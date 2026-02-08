"""Risk Assessment Agent — uses Gemini 2.5 Pro with function calling to assess
a patient's cancer risk against the NICE NG12 guidelines.

Supports two LLM providers (set LLM_PROVIDER env var):
  - "google_genai"  — Google AI Studio via google-generativeai SDK (API key)
  - "vertex_ai"     — Google Cloud Vertex AI via google-cloud-aiplatform SDK

Architecture improvements over baseline:
  - Manual agentic loop (no AutomaticFunctionCallingResponder)
  - Structured JSON output via Gemini response_schema
  - Multi-query RAG: symptom-level + demographic pre-fetch

Grounding guardrails:
  1. Relevance gate  — low-relevance chunks filtered before the model sees them
  2. Tool-use check  — flags if the model skipped the guideline search
  3. Citation check   — ensures the final output includes citations
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from app.config import (
    GOOGLE_API_KEY, GEMINI_MODEL, LLM_PROVIDER,
    GCP_PROJECT_ID, GCP_LOCATION, MAX_AGENT_STEPS,
    LLM_TIMEOUT_SECONDS,
)
from app.patient_data import get_patient
from app.rag import search_guidelines, search_guidelines_with_timeout
from app.tracing import AgentStepTracer
from app.guardrails import MEDICAL_DISCLAIMER
from app.llm_judge import quick_safety_check, run_background_evaluation, _SAFETY_WARNING

logger = logging.getLogger(__name__)

# ── Initialise the LLM SDK ──────────────────────────────────────────────
if LLM_PROVIDER == "vertex_ai":
    import vertexai
    from vertexai.generative_models import (
        GenerativeModel, GenerationConfig, Content, Part, Tool,
        FunctionDeclaration,
    )
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
else:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)

# ── System prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a Clinical Decision Support Agent that evaluates cancer risk using the
NICE NG12 guidelines ("Suspected cancer: recognition and referral").

## Workflow
1. Retrieve the patient's record with get_patient_data.
2. For EACH symptom the patient has, call search_clinical_guidelines with a
   targeted query (e.g. "hemoptysis urgent referral lung cancer criteria").
3. Cross-reference the patient's age, gender, smoking history, symptom list,
   and symptom duration against the retrieved guideline passages.
4. Decide on the risk level.

## Risk Levels (use exactly one of these strings)
- "Urgent Referral (2-week wait)"  – meets NG12 criteria for 2WW referral
- "Urgent Investigation"           – meets NG12 criteria for direct-access tests
- "Non-urgent"                     – symptoms present but NG12 thresholds not met
- "No cancer indicators"           – nothing in NG12 applies to this patient

## Output
Respond with ONLY a JSON object matching the required schema.

## Rules
- Base your assessment ONLY on retrieved guideline text.  Never invent criteria.
- Include ALL citations that support your reasoning.
- If the guidelines are unclear or insufficient, state that in your reasoning.
- Also consider the pre-fetched guideline passages provided in the initial context.
"""

# ── Structured output schema for Gemini ──────────────────────────────────
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "patient_id": {"type": "string"},
        "patient_name": {"type": "string"},
        "risk_level": {
            "type": "string",
            "enum": [
                "Urgent Referral (2-week wait)",
                "Urgent Investigation",
                "Non-urgent",
                "No cancer indicators",
            ],
        },
        "cancer_type_suspected": {"type": "string", "nullable": True},
        "reasoning": {"type": "string"},
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "page": {"type": "integer"},
                    "chunk_id": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
                "required": ["source", "page", "chunk_id", "excerpt"],
            },
        },
    },
    "required": [
        "patient_id", "patient_name", "risk_level",
        "reasoning", "recommendations", "citations",
    ],
}

# ── Guardrail constants ──────────────────────────────────────────────────
RELEVANCE_DISTANCE_THRESHOLD = 1.5

_GROUNDING_WARNING = (
    " WARNING: This assessment may not be fully grounded in NG12 guidelines."
    " The model either did not search the guidelines or did not cite specific passages."
    " Please verify independently."
)


def _filter_relevant_chunks(chunks: list[dict]) -> list[dict]:
    """Guardrail 1 — drop chunks with distance above threshold."""
    return [c for c in chunks if c.get("distance", 0) <= RELEVANCE_DISTANCE_THRESHOLD]


def _check_tool_was_called(tracked: list[dict]) -> bool:
    """Guardrail 2 — True if the search tool was actually invoked."""
    return len(tracked) > 0


def _check_citations_in_json(result: dict) -> bool:
    """Guardrail 3 — True if the parsed result has at least one citation."""
    citations = result.get("citations", [])
    if citations:
        return True
    # Also check if the reasoning text mentions page numbers
    reasoning = result.get("reasoning", "")
    return bool(re.search(r"(p\.\d+|page\s*\d+|\[NG12)", reasoning, re.IGNORECASE))


# ── LLM timeout wrapper ────────────────────────────────────────────────

def _send_message_with_timeout(chat, message, timeout=LLM_TIMEOUT_SECONDS):
    """Wrap chat.send_message with a timeout using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(chat.send_message, message)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.error("LLM call timed out after %ds", timeout)
            raise TimeoutError(f"LLM call timed out after {timeout}s")


# ── Multi-query RAG pre-fetch ───────────────────────────────────────────

def _build_prefetch_queries(patient: dict) -> list[str]:
    """Generate targeted RAG queries from patient data."""
    queries = []

    # One query per symptom
    for symptom in patient.get("symptoms", []):
        queries.append(f"{symptom} cancer referral criteria NG12")

    # Demographic query
    age = patient.get("age", "")
    gender = patient.get("gender", "")
    smoking = patient.get("smoking_history", "")
    queries.append(f"{age}yo {gender} {smoking} cancer risk NG12")

    return queries


def _prefetch_context(patient: dict) -> tuple[list[dict], str]:
    """Run multi-query RAG and return unique chunks + formatted context text."""
    queries = _build_prefetch_queries(patient)
    seen_ids: set[str] = set()
    all_chunks: list[dict] = []

    for query in queries:
        results = search_guidelines_with_timeout(query, top_k=5)
        for chunk in results:
            cid = chunk.get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(chunk)

    # Format as context text for the prompt
    if not all_chunks:
        return all_chunks, ""

    lines = ["## Pre-fetched NG12 Guideline Passages"]
    for i, c in enumerate(all_chunks, 1):
        lines.append(
            f"[{i}] (page {c.get('page', '?')}, id={c.get('chunk_id', '?')}) "
            f"{c.get('text', '')[:500]}"
        )
    return all_chunks, "\n".join(lines)


# ── Manual agentic loop (Vertex AI) ─────────────────────────────────────

def _run_vertex_agent_loop(
    patient_id: str,
    prompt: str,
    get_patient_data,
    search_clinical_guidelines,
) -> str:
    """Manual tool-calling loop for Vertex AI — replaces AutomaticFunctionCallingResponder."""

    tool_map = {
        "get_patient_data": get_patient_data,
        "search_clinical_guidelines": search_clinical_guidelines,
    }

    # Define tool declarations manually for full control
    patient_func_decl = FunctionDeclaration(
        name="get_patient_data",
        description="Retrieve a patient's medical record from the hospital database.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The unique patient identifier (e.g. 'PT-101').",
                },
            },
            "required": ["patient_id"],
        },
    )
    search_func_decl = FunctionDeclaration(
        name="search_clinical_guidelines",
        description="Search the NICE NG12 cancer referral guidelines for relevant passages.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Clinical search terms about symptoms, referral criteria, or cancer types.",
                },
            },
            "required": ["query"],
        },
    )
    tool = Tool(function_declarations=[patient_func_decl, search_func_decl])

    model = GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        tools=[tool],
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
        ),
    )

    chat = model.start_chat()
    logger.info("Starting manual agent loop for %s (Vertex AI)", patient_id)

    response = _send_message_with_timeout(chat, prompt)

    for step in range(MAX_AGENT_STEPS):
        # Check if the response contains function calls
        candidate = response.candidates[0]
        function_calls = [
            part.function_call
            for part in candidate.content.parts
            if part.function_call and part.function_call.name
        ]

        if not function_calls:
            # Model returned final text — extract it
            text_parts = [
                part.text for part in candidate.content.parts if part.text
            ]
            return "".join(text_parts).strip()

        # Execute each function call and collect responses
        fn_response_parts = []
        for fc in function_calls:
            fn_name = fc.name
            fn_args = dict(fc.args) if fc.args else {}
            logger.info("  Step %d: calling %s(%s)", step + 1, fn_name, fn_args)

            with AgentStepTracer(fn_name, step=step + 1, patient_id=patient_id):
                if fn_name in tool_map:
                    fn_result = tool_map[fn_name](**fn_args)
                else:
                    fn_result = {"error": f"Unknown function: {fn_name}"}

            fn_response_parts.append(
                Part.from_function_response(
                    name=fn_name,
                    response={"result": fn_result},
                )
            )

        # Send function responses back to the model
        response = _send_message_with_timeout(chat, fn_response_parts)

    logger.warning("Agent loop hit MAX_AGENT_STEPS (%d) for %s", MAX_AGENT_STEPS, patient_id)
    # Return whatever text we have
    text_parts = [
        part.text for part in response.candidates[0].content.parts if part.text
    ]
    return "".join(text_parts).strip() if text_parts else "{}"


def assess_patient(patient_id: str) -> dict:
    """Run the full risk-assessment pipeline for one patient.

    Returns a dict matching the RiskAssessmentResponse schema.
    """
    # Closure list to collect every chunk the agent retrieves
    tracked_citations: list[dict] = []

    # ── Tool functions (defined here so they capture tracked_citations) ──

    def get_patient_data(patient_id: str) -> dict:
        """Retrieve a patient's medical record from the hospital database."""
        result = get_patient(patient_id)
        if result:
            return result
        return {"error": f"Patient {patient_id} not found"}

    def search_clinical_guidelines(query: str) -> list:
        """Search the NICE NG12 cancer referral guidelines for relevant passages."""
        raw_results = search_guidelines_with_timeout(query, top_k=5)

        # Guardrail 1: filter out low-relevance chunks
        relevant = _filter_relevant_chunks(raw_results)
        if len(relevant) < len(raw_results):
            logger.info(
                "Relevance filter: kept %d/%d chunks (threshold=%.2f)",
                len(relevant), len(raw_results), RELEVANCE_DISTANCE_THRESHOLD,
            )

        tracked_citations.extend(relevant)
        return relevant

    # ── Multi-query RAG pre-fetch ────────────────────────────────────────
    patient = get_patient(patient_id)
    prefetch_chunks = []
    context_text = ""
    if patient:
        prefetch_chunks, context_text = _prefetch_context(patient)
        tracked_citations.extend(prefetch_chunks)
        logger.info("Pre-fetched %d unique chunks for %s", len(prefetch_chunks), patient_id)

    # ── Build the prompt ─────────────────────────────────────────────────
    prompt = (
        f"Assess the cancer risk for patient ID: {patient_id}. "
        f"First retrieve their record, then search the NG12 guidelines for "
        f"each of their symptoms individually. Finally, provide the complete "
        f"risk assessment as a JSON object."
    )

    if context_text:
        prompt += (
            f"\n\nThe following guideline passages were pre-fetched based on the "
            f"patient's symptoms and demographics. Use them as additional context, "
            f"but still call search_clinical_guidelines for thorough coverage.\n\n"
            f"{context_text}"
        )

    # ── Run the agent ────────────────────────────────────────────────────
    if LLM_PROVIDER == "vertex_ai":
        raw_text = _run_vertex_agent_loop(
            patient_id, prompt, get_patient_data, search_clinical_guidelines,
        )
    else:
        # google_genai fallback — keep automatic function calling
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            tools=[get_patient_data, search_clinical_guidelines],
            system_instruction=SYSTEM_PROMPT,
        )
        chat = model.start_chat(enable_automatic_function_calling=True)

        logger.info("Starting assessment for %s (Google AI)", patient_id)
        response = _send_message_with_timeout(chat, prompt)
        raw_text = response.text.strip()

    logger.info("Gemini response received (%d chars)", len(raw_text))

    # ── Parse the JSON response ──────────────────────────────────────────
    try:
        # Strip markdown code fences if the model added them (fallback safety)
        cleaned = raw_text
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3]

        result = json.loads(cleaned.strip())

        # Ensure citations exist — fall back to tracked chunks if model omitted them
        if not result.get("citations") and tracked_citations:
            result["citations"] = _format_citations(tracked_citations)

        # ── Guardrail 2: flag if model never searched guidelines ─────
        if not _check_tool_was_called(tracked_citations):
            logger.warning("Assessment [%s]: model did NOT search guidelines", patient_id)
            result["reasoning"] = result.get("reasoning", "") + _GROUNDING_WARNING
            result.setdefault("grounding_flags", []).append("no_retrieval")

        # ── Guardrail 3: flag if result lacks citations ──────────────
        elif not _check_citations_in_json(result):
            logger.warning("Assessment [%s]: no citations found in output", patient_id)
            result["reasoning"] = result.get("reasoning", "") + _GROUNDING_WARNING
            result.setdefault("grounding_flags", []).append("missing_citations")

        # Append medical disclaimer to every response
        result["disclaimer"] = MEDICAL_DISCLAIMER

        # ── Safety gate: check → re-generate if unsafe → check again ──
        safety = quick_safety_check(result, patient, tracked_citations)

        if not safety["safe"]:
            logger.warning(
                "Safety check FAILED for %s — re-generating with corrective feedback",
                patient_id,
            )

            # Re-generate: send corrective feedback to the LLM
            corrective_prompt = (
                f"Your previous assessment was flagged as potentially UNSAFE by a "
                f"clinical safety evaluator. The safety concern is:\n\n"
                f"{safety['reasoning']}\n\n"
                f"Please re-assess patient {patient_id} with extra caution. "
                f"Err on the side of a higher risk level if there is any doubt. "
                f"A missed urgent referral is more dangerous than an unnecessary one. "
                f"Return the corrected JSON assessment."
            )

            if LLM_PROVIDER == "vertex_ai":
                corrected_text = _run_vertex_agent_loop(
                    patient_id, corrective_prompt,
                    get_patient_data, search_clinical_guidelines,
                )
            else:
                corrective_model = genai.GenerativeModel(
                    model_name=GEMINI_MODEL,
                    system_instruction=SYSTEM_PROMPT,
                )
                corrective_chat = corrective_model.start_chat()
                # Provide original context + correction request
                corrective_response = _send_message_with_timeout(
                    corrective_chat,
                    f"Original assessment:\n{raw_text}\n\n{corrective_prompt}",
                )
                corrected_text = corrective_response.text.strip()

            # Parse corrected response
            try:
                corrected_cleaned = corrected_text
                if corrected_cleaned.startswith("```"):
                    lines = corrected_cleaned.split("\n")
                    corrected_cleaned = "\n".join(lines[1:])
                    if corrected_cleaned.rstrip().endswith("```"):
                        corrected_cleaned = corrected_cleaned.rstrip()[:-3]

                corrected_result = json.loads(corrected_cleaned.strip())
                corrected_result["disclaimer"] = MEDICAL_DISCLAIMER
                corrected_result["safety_corrected"] = True

                # Run safety check again on corrected result
                safety_retry = quick_safety_check(corrected_result, patient, tracked_citations)
                if not safety_retry["safe"]:
                    # Still unsafe after retry — return with warning
                    corrected_result["safety_warning"] = _SAFETY_WARNING
                    corrected_result.setdefault("grounding_flags", []).append("safety_concern")
                    logger.warning("Safety still FAILED after re-generation for %s", patient_id)

                result = corrected_result
            except (json.JSONDecodeError, Exception):
                logger.warning("Could not parse corrected response, using original with warning")
                result["safety_warning"] = _SAFETY_WARNING
                result.setdefault("grounding_flags", []).append("safety_concern")

        result["evaluation_status"] = "processing"

        # ── Full evaluation (background — 5 criteria, does NOT block) ──
        import threading
        eval_thread = threading.Thread(
            target=run_background_evaluation,
            args=(result.copy(), patient, tracked_citations),
            daemon=True,
        )
        eval_thread.start()

        return result

    except json.JSONDecodeError:
        logger.warning("Could not parse Gemini output as JSON")
        return {
            "patient_id": patient_id,
            "patient_name": "Unknown",
            "risk_level": "Error - could not parse model response",
            "cancer_type_suspected": None,
            "reasoning": raw_text,
            "recommendations": ["Please retry the assessment"],
            "citations": _format_citations(tracked_citations),
        }



def _format_citations(chunks: list[dict]) -> list[dict]:
    """De-duplicate and format raw chunks into citation dicts."""
    seen = set()
    citations = []
    for c in chunks:
        cid = c.get("chunk_id", "")
        if cid in seen:
            continue
        seen.add(cid)
        citations.append(
            {
                "source": c.get("source", "NG12 PDF"),
                "page": c.get("page", 0),
                "chunk_id": cid,
                "excerpt": c.get("text", "")[:300],
            }
        )
    return citations
