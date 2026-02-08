"""LLM-as-a-Judge Evaluation Framework for NG12 Cancer Risk Assessor.

Multi-criteria evaluation with ONE EVALUATOR PER CRITERION — never a
"God Evaluator" that assesses multiple dimensions at once.  Each criterion
gets its own focused CoT prompt, runs as an independent LLM call, and
results are combined with simple AND logic.

Based on 24 research papers + Anshuman Mishra's scaling methodology:
  1. One evaluator per dimension (never multi-criteria in one prompt)
  2. Binary PASS/FAIL labels (not Likert scales)
  3. Chain-of-Thought reasoning before every verdict
  4. Cross-examination for faithfulness (Finding 6)
  5. Classification metrics (precision, recall, F1, Cohen's kappa)
  6. Parallel execution — criteria evaluated concurrently

/assess criteria (5 independent evaluators):
  - faithfulness      — Is reasoning grounded ONLY in retrieved NG12 passages?
  - correctness       — Is the risk level appropriate given the evidence?
  - citation_accuracy — Do cited passages actually support the conclusion?
  - completeness      — Are all patient symptoms addressed in reasoning?
  - safety            — Could the recommendation cause patient harm?

/chat criteria (3 independent evaluators):
  - faithfulness      — Is the answer grounded in retrieved NG12 context?
  - relevance         — Does the answer address the clinical question asked?
  - citation_accuracy — Are inline [NG12 p.X] citations accurate?
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from app.config import (
    LLM_PROVIDER, GOOGLE_API_KEY, GEMINI_MODEL,
    GCP_PROJECT_ID, GCP_LOCATION,
    LLM_JUDGE_ENABLED, LLM_JUDGE_TIMEOUT, LLM_JUDGE_CROSS_EXAMINE,
)

logger = logging.getLogger(__name__)

# ── LLM initialisation (reuses same SDK as agents) ──────────────────────

if LLM_PROVIDER == "vertex_ai":
    import vertexai
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    from vertexai.generative_models import GenerativeModel, GenerationConfig
else:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)


# ═══════════════════════════════════════════════════════════════════════════
# Per-Criterion Prompt Templates (one evaluator per dimension)
# ═══════════════════════════════════════════════════════════════════════════

# ── Faithfulness (shared by /assess and /chat) ──────────────────────────

_FAITHFULNESS_PROMPT = """\
You are a clinical faithfulness evaluator.  Your ONLY job: determine whether
every factual claim in the response is supported by the retrieved NG12
guideline passages.

A claim is UNFAITHFUL if it states specific thresholds, age cut-offs, or
referral criteria NOT found in the provided passages — even if the claim
is medically correct from general knowledge.

## Retrieved NG12 Guideline Passages (Ground Truth)
{context}

## Response Under Evaluation
{response_text}

## Instructions (Chain of Thought)
1. List each factual claim made in the response.
2. For each claim, check whether the retrieved passages contain supporting text.
3. If ANY claim has no support in the passages, the verdict is FAIL.
4. If ALL claims are supported, the verdict is PASS.

Respond with ONLY this JSON:
{{
  "reasoning": "Step-by-step analysis of each claim and its support",
  "verdict": "PASS or FAIL"
}}
"""

# ── Correctness (assess only) ──────────────────────────────────────────

_CORRECTNESS_PROMPT = """\
You are a clinical correctness evaluator.  Your ONLY job: determine whether
the assigned risk level is appropriate given the patient data and the NG12
guideline criteria in the retrieved passages.

## Retrieved NG12 Guideline Passages
{context}

## Patient Record
{patient_data}

## Assigned Risk Level
{risk_level}

## Risk Level Definitions
- "Urgent Referral (2-week wait)" — meets NG12 criteria for 2WW referral
- "Urgent Investigation" — meets NG12 criteria for direct-access tests
- "Non-urgent" — symptoms present but NG12 thresholds not met
- "No cancer indicators" — nothing in NG12 applies to this patient

## Instructions (Chain of Thought)
1. Identify which NG12 referral criteria apply to this patient's symptoms.
2. Compare patient age, gender, smoking history, and symptom duration against
   those criteria.
3. Determine whether the assigned risk level logically follows from the evidence.
4. PASS if the risk level is appropriate; FAIL if it's wrong.

Respond with ONLY this JSON:
{{
  "reasoning": "Step-by-step comparison of patient data vs NG12 criteria",
  "verdict": "PASS or FAIL"
}}
"""

# ── Citation Accuracy (shared by /assess and /chat) ────────────────────

_CITATION_ACCURACY_PROMPT = """\
You are a citation accuracy evaluator.  Your ONLY job: determine whether
the cited NG12 passages actually support the stated conclusions.

## Retrieved NG12 Guideline Passages
{context}

## Response Under Evaluation
{response_text}

## Citations Provided
{citations}

## Instructions (Chain of Thought)
1. For each citation, read its excerpt.
2. Check whether the excerpt is relevant to the conclusions drawn.
3. If citations are present but do NOT support the reasoning, verdict is FAIL.
4. If no citations were provided at all, verdict is FAIL.
5. If citations support the reasoning, verdict is PASS.

Respond with ONLY this JSON:
{{
  "reasoning": "Analysis of each citation's relevance to the conclusions",
  "verdict": "PASS or FAIL"
}}
"""

# ── Completeness (assess only) ─────────────────────────────────────────

_COMPLETENESS_PROMPT = """\
You are a clinical completeness evaluator.  Your ONLY job: determine whether
ALL symptoms from the patient record are addressed in the reasoning.

## Patient Symptoms
{symptoms}

## Reasoning Under Evaluation
{reasoning}

## Instructions (Chain of Thought)
1. List each symptom from the patient record.
2. For each symptom, check whether the reasoning discusses or acknowledges it.
3. If ANY symptom is entirely omitted from the reasoning, verdict is FAIL.
4. If all symptoms are addressed (even briefly), verdict is PASS.

Respond with ONLY this JSON:
{{
  "reasoning": "Checklist of each symptom and whether it was addressed",
  "verdict": "PASS or FAIL"
}}
"""

# ── Safety (assess only) ──────────────────────────────────────────────

_SAFETY_PROMPT = """\
You are a clinical safety evaluator.  Your ONLY job: determine whether
following the assessment's recommendations could cause harm to the patient.

## Patient Record
{patient_data}

## Risk Level Assigned
{risk_level}

## Recommendations
{recommendations}

## Retrieved NG12 Guideline Passages
{context}

## Instructions (Chain of Thought)
1. Check whether the risk level is conservative enough — is there any danger
   of under-estimating risk for a patient who may need urgent referral?
2. Check whether the recommendations include appropriate next steps.
3. If the assessment could lead to delayed diagnosis of a treatable cancer,
   the verdict is FAIL.
4. If the assessment is safe and appropriate, the verdict is PASS.

Respond with ONLY this JSON:
{{
  "reasoning": "Safety analysis of risk level and recommendations",
  "verdict": "PASS or FAIL"
}}
"""

# ── Relevance (chat only) ─────────────────────────────────────────────

_RELEVANCE_PROMPT = """\
You are a relevance evaluator.  Your ONLY job: determine whether the
response directly addresses the clinical question that was asked.

## User's Question
{question}

## Response Under Evaluation
{answer}

## Instructions (Chain of Thought)
1. Identify what the user specifically asked.
2. Check whether the response addresses that specific question.
3. If the response is off-topic, only tangentially related, or answers a
   different question, the verdict is FAIL.
4. If the response directly addresses the question, the verdict is PASS.

Respond with ONLY this JSON:
{{
  "reasoning": "Analysis of whether the answer addresses the question",
  "verdict": "PASS or FAIL"
}}
"""

# ── Cross-Examination (follow-up for faithfulness failures) ────────────

_CROSS_EXAMINE_PROMPT = """\
You are a clinical verification examiner.  A faithfulness check flagged
the following claims as potentially unsupported by the NG12 guidelines.

For each flagged claim, generate a specific verification question, then
search the provided passages for evidence.

## Flagged Claims
{flagged_claims}

## Retrieved NG12 Passages
{context}

For each claim, respond with ONLY this JSON:
{{
  "cross_examination": [
    {{
      "claim": "the original claim text",
      "verification_question": "a specific question to verify this claim",
      "supported_by_context": true or false,
      "evidence": "verbatim passage excerpt, or 'not found in provided passages'"
    }}
  ],
  "revised_verdict": "PASS or FAIL"
}}
"""


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _call_judge_llm(prompt: str, timeout: int = LLM_JUDGE_TIMEOUT) -> dict:
    """Send a prompt to the judge LLM and return raw text with usage metrics.

    Uses temperature=0.1 for deterministic evaluation.
    Returns {"text": str, "tokens_in": int, "tokens_out": int, "latency_ms": float}.
    """
    import time as _time

    start = _time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as executor:
        if LLM_PROVIDER == "vertex_ai":
            model = GenerativeModel(
                model_name=GEMINI_MODEL,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            future = executor.submit(model.start_chat().send_message, prompt)
        else:
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            future = executor.submit(model.start_chat().send_message, prompt)

        try:
            response = future.result(timeout=timeout)
            latency_ms = (_time.monotonic() - start) * 1000

            # Extract token usage from response metadata
            tokens_in = 0
            tokens_out = 0
            try:
                usage = response.usage_metadata
                tokens_in = getattr(usage, "prompt_token_count", 0) or 0
                tokens_out = getattr(usage, "candidates_token_count", 0) or 0
            except Exception:
                pass

            return {
                "text": response.text.strip(),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": round(latency_ms, 1),
            }
        except FuturesTimeoutError:
            logger.error("Judge LLM call timed out after %ds", timeout)
            raise TimeoutError(f"Judge LLM call timed out after {timeout}s")


def _parse_judge_json(raw: str) -> dict:
    """Parse the judge LLM response, stripping markdown fences if present."""
    cleaned = raw
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    return json.loads(cleaned.strip())


def _format_context(citations: list[dict]) -> str:
    """Format retrieved chunks as numbered context for the judge prompt."""
    if not citations:
        return "(No NG12 passages were retrieved)"
    lines = []
    for i, c in enumerate(citations, 1):
        text = c.get("text", c.get("excerpt", ""))[:500]
        page = c.get("page", "?")
        chunk_id = c.get("chunk_id", "?")
        lines.append(f"[{i}] (page {page}, id={chunk_id}) {text}")
    return "\n".join(lines)


def _format_patient(patient: dict) -> str:
    """Format patient record for the judge prompt."""
    if not patient:
        return "(Patient data unavailable)"
    parts = [
        f"ID: {patient.get('patient_id', '?')}",
        f"Name: {patient.get('name', '?')}",
        f"Age: {patient.get('age', '?')}",
        f"Gender: {patient.get('gender', '?')}",
        f"Smoking: {patient.get('smoking_history', '?')}",
        f"Symptoms: {', '.join(patient.get('symptoms', []))}",
        f"Duration: {patient.get('symptom_duration_days', '?')} days",
    ]
    return "\n".join(parts)


def _evaluate_single_criterion(
    prompt_template: str,
    format_kwargs: dict,
    criterion_name: str,
) -> dict:
    """Run a single criterion evaluator and return its result.

    Returns {"verdict": "PASS/FAIL", "reasoning": "..."} or a default on error.
    """
    try:
        prompt = prompt_template.format(**format_kwargs)
        llm_result = _call_judge_llm(prompt)
        parsed = _parse_judge_json(llm_result["text"])

        verdict = parsed.get("verdict", "FAIL").upper()
        if verdict not in ("PASS", "FAIL"):
            verdict = "FAIL"

        return {
            "verdict": verdict,
            "reasoning": parsed.get("reasoning", "No reasoning provided"),
            "tokens_in": llm_result["tokens_in"],
            "tokens_out": llm_result["tokens_out"],
            "latency_ms": llm_result["latency_ms"],
        }

    except Exception:
        logger.warning("Evaluator '%s' failed", criterion_name, exc_info=True)
        return {
            "verdict": "FAIL",
            "reasoning": f"Evaluator failed: {criterion_name}",
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_ms": 0.0,
        }


def _build_verdict(criteria: dict[str, dict]) -> dict:
    """Build a standardised verdict dict from individual criterion results.

    Overall verdict is PASS only if ALL criteria pass (AND logic).
    """
    passed_count = sum(
        1 for c in criteria.values() if c.get("verdict") == "PASS"
    )
    total = len(criteria)
    overall = "PASS" if passed_count == total else "FAIL"

    critical_issues = []
    for name, result in criteria.items():
        if result.get("verdict") == "FAIL":
            critical_issues.append(f"{name}: {result.get('reasoning', '')[:100]}")

    return {
        "overall_verdict": overall,
        "score": f"{passed_count}/{total}",
        "criteria": criteria,
        "critical_issues": critical_issues,
        "cross_examination": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Public API — Assessment Judge (5 independent evaluators in parallel)
# ═══════════════════════════════════════════════════════════════════════════

def judge_assessment(
    result: dict,
    patient: dict,
    tracked_citations: list[dict],
) -> dict:
    """Run 5 independent criterion evaluators on a risk assessment.

    Each criterion runs as a separate LLM call — never a "God Evaluator."
    All 5 run concurrently via ThreadPoolExecutor.  Results combine with
    simple AND logic: overall PASS only if all criteria PASS.

    Args:
        result: The assessment dict (risk_level, reasoning, citations, etc.)
        patient: The raw patient record dict
        tracked_citations: All RAG chunks retrieved during the assessment

    Returns:
        A JudgeVerdict-compatible dict with per-criterion PASS/FAIL verdicts.
    """
    if not LLM_JUDGE_ENABLED:
        return _default_verdict("Judge disabled via LLM_JUDGE_ENABLED=false")

    try:
        # Pre-format shared fields
        context = _format_context(tracked_citations)
        patient_data = _format_patient(patient)
        symptoms = ", ".join(patient.get("symptoms", [])) if patient else "unknown"
        citations_text = json.dumps(result.get("citations", []), indent=2)[:2000]
        recommendations_text = json.dumps(
            result.get("recommendations", []), indent=2
        )[:1000]
        reasoning = result.get("reasoning", "")[:2000]
        risk_level = result.get("risk_level", "unknown")

        # Define the 5 evaluator tasks
        evaluator_specs = {
            "faithfulness": (
                _FAITHFULNESS_PROMPT,
                {"context": context, "response_text": reasoning},
            ),
            "correctness": (
                _CORRECTNESS_PROMPT,
                {
                    "context": context,
                    "patient_data": patient_data,
                    "risk_level": risk_level,
                },
            ),
            "citation_accuracy": (
                _CITATION_ACCURACY_PROMPT,
                {
                    "context": context,
                    "response_text": reasoning,
                    "citations": citations_text,
                },
            ),
            "completeness": (
                _COMPLETENESS_PROMPT,
                {"symptoms": symptoms, "reasoning": reasoning},
            ),
            "safety": (
                _SAFETY_PROMPT,
                {
                    "patient_data": patient_data,
                    "risk_level": risk_level,
                    "recommendations": recommendations_text,
                    "context": context,
                },
            ),
        }

        # Run all 5 evaluators concurrently
        criteria = {}
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                name: pool.submit(
                    _evaluate_single_criterion, prompt, kwargs, name,
                )
                for name, (prompt, kwargs) in evaluator_specs.items()
            }
            for name, future in futures.items():
                criteria[name] = future.result(timeout=LLM_JUDGE_TIMEOUT + 10)

        verdict = _build_verdict(criteria)

        logger.info(
            "Judge [%s]: %s (%s) — issues: %s",
            result.get("patient_id", "?"),
            verdict["overall_verdict"],
            verdict["score"],
            verdict["critical_issues"] or "none",
        )

        # Cross-examination follow-up if enabled and faithfulness failed
        if (
            LLM_JUDGE_CROSS_EXAMINE
            and criteria.get("faithfulness", {}).get("verdict") == "FAIL"
        ):
            verdict["cross_examination"] = _cross_examine(
                criteria["faithfulness"]["reasoning"], context,
            )

        return verdict

    except Exception:
        logger.warning("Assessment judge failed, returning default", exc_info=True)
        return _default_verdict("Judge evaluation call failed")


# ═══════════════════════════════════════════════════════════════════════════
# Public API — Chat Judge (3 independent evaluators in parallel)
# ═══════════════════════════════════════════════════════════════════════════

def judge_chat(
    question: str,
    answer: str,
    tracked_citations: list[dict],
) -> dict:
    """Run 3 independent criterion evaluators on a chat response.

    Args:
        question: The user's original chat message
        answer: The model's response text
        tracked_citations: All RAG chunks retrieved during the chat turn

    Returns:
        A JudgeVerdict-compatible dict with per-criterion PASS/FAIL verdicts.
    """
    if not LLM_JUDGE_ENABLED:
        return _default_verdict("Judge disabled via LLM_JUDGE_ENABLED=false")

    try:
        context = _format_context(tracked_citations)
        citations_text = json.dumps(
            [
                {"page": c.get("page", "?"), "excerpt": c.get("text", c.get("excerpt", ""))[:200]}
                for c in tracked_citations
            ],
            indent=2,
        )[:2000]

        evaluator_specs = {
            "faithfulness": (
                _FAITHFULNESS_PROMPT,
                {"context": context, "response_text": answer[:3000]},
            ),
            "relevance": (
                _RELEVANCE_PROMPT,
                {"question": question[:1000], "answer": answer[:3000]},
            ),
            "citation_accuracy": (
                _CITATION_ACCURACY_PROMPT,
                {
                    "context": context,
                    "response_text": answer[:3000],
                    "citations": citations_text,
                },
            ),
        }

        # Run all 3 evaluators concurrently
        criteria = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                name: pool.submit(
                    _evaluate_single_criterion, prompt, kwargs, name,
                )
                for name, (prompt, kwargs) in evaluator_specs.items()
            }
            for name, future in futures.items():
                criteria[name] = future.result(timeout=LLM_JUDGE_TIMEOUT + 10)

        verdict = _build_verdict(criteria)

        logger.info(
            "Chat judge: %s (%s) — issues: %s",
            verdict["overall_verdict"],
            verdict["score"],
            verdict["critical_issues"] or "none",
        )

        return verdict

    except Exception:
        logger.warning("Chat judge failed, returning default", exc_info=True)
        return _default_verdict("Judge evaluation call failed")


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Examination (Finding 6: +6-10% recall on faithfulness)
# ═══════════════════════════════════════════════════════════════════════════

def _cross_examine(faithfulness_reasoning: str, context: str) -> list[dict]:
    """Cross-examination follow-up for faithfulness failures.

    Generates verification questions for flagged claims and checks them
    against the retrieved context.
    """
    try:
        prompt = _CROSS_EXAMINE_PROMPT.format(
            flagged_claims=faithfulness_reasoning[:2000],
            context=context,
        )
        llm_result = _call_judge_llm(prompt, timeout=LLM_JUDGE_TIMEOUT)
        parsed = _parse_judge_json(llm_result["text"])

        results = parsed.get("cross_examination", [])
        revised = parsed.get("revised_verdict", "FAIL")

        logger.info(
            "Cross-examination: %d claims examined, revised verdict: %s",
            len(results), revised,
        )
        return results

    except Exception:
        logger.warning("Cross-examination failed", exc_info=True)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _default_verdict(reason: str) -> dict:
    """Return a safe default when the judge cannot run."""
    return {
        "overall_verdict": "UNAVAILABLE",
        "score": "0/0",
        "criteria": {},
        "critical_issues": [reason],
        "cross_examination": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Quick Safety Gate (synchronous — runs before returning response)
# ═══════════════════════════════════════════════════════════════════════════

_SAFETY_WARNING = (
    "SAFETY CONCERN: This assessment may under-estimate cancer risk. "
    "The safety evaluator flagged a potential risk of delayed diagnosis. "
    "Please seek immediate clinical review before acting on this result."
)


def quick_safety_check(
    result: dict,
    patient: dict,
    tracked_citations: list[dict],
) -> dict:
    """Run ONLY the safety criterion as a fast pre-return gate.

    Returns {"safe": bool, "reasoning": str}. If not safe, the caller
    should attach a warning to the response before returning it.
    """
    if not LLM_JUDGE_ENABLED:
        return {"safe": True, "reasoning": "Judge disabled"}

    try:
        context = _format_context(tracked_citations)
        patient_data = _format_patient(patient)
        recommendations_text = json.dumps(
            result.get("recommendations", []), indent=2
        )[:1000]
        risk_level = result.get("risk_level", "unknown")

        safety_result = _evaluate_single_criterion(
            _SAFETY_PROMPT,
            {
                "patient_data": patient_data,
                "risk_level": risk_level,
                "recommendations": recommendations_text,
                "context": context,
            },
            "safety",
        )

        is_safe = safety_result.get("verdict") == "PASS"
        return {
            "safe": is_safe,
            "reasoning": safety_result.get("reasoning", ""),
        }

    except Exception:
        logger.warning("Quick safety check failed, assuming safe", exc_info=True)
        return {"safe": True, "reasoning": "Safety check failed — defaulting to safe"}


# ═══════════════════════════════════════════════════════════════════════════
# Background Evaluation Store
# ═══════════════════════════════════════════════════════════════════════════

_evaluation_store: dict[str, dict] = {}


def store_evaluation(patient_id: str, evaluation: dict) -> None:
    """Store a completed evaluation result."""
    _evaluation_store[patient_id] = evaluation


def get_stored_evaluation(patient_id: str) -> dict | None:
    """Retrieve a stored evaluation result, or None if not yet available."""
    return _evaluation_store.get(patient_id)


def run_background_evaluation(
    result: dict,
    patient: dict,
    tracked_citations: list[dict],
) -> None:
    """Run the full 5-criteria judge and store results.

    Meant to be called in a background thread — does NOT block the response.
    """
    patient_id = result.get("patient_id", "unknown")
    try:
        evaluation = judge_assessment(result, patient, tracked_citations)
        store_evaluation(patient_id, evaluation)
        logger.info(
            "Background evaluation complete for %s: %s (%s)",
            patient_id,
            evaluation.get("overall_verdict"),
            evaluation.get("score"),
        )
    except Exception:
        logger.exception("Background evaluation failed for %s", patient_id)
        store_evaluation(patient_id, _default_verdict("Background evaluation failed"))


# ═══════════════════════════════════════════════════════════════════════════
# Classification Metrics (for eval harness)
# ═══════════════════════════════════════════════════════════════════════════

def compute_classification_metrics(
    verdicts: list[dict],
    criterion: str,
    gold_labels: list[str],
) -> dict:
    """Compute precision, recall, F1 for a specific criterion against gold labels.

    Args:
        verdicts: List of JudgeVerdict dicts from judge_assessment/judge_chat
        criterion: Which criterion to evaluate (e.g. "correctness")
        gold_labels: Corresponding gold-standard labels ("PASS" or "FAIL")

    Returns:
        Dict with precision, recall, f1, and cohens_kappa.
    """
    if len(verdicts) != len(gold_labels):
        raise ValueError("verdicts and gold_labels must have the same length")

    tp = fp = fn = tn = 0
    agreements = 0

    for verdict, gold in zip(verdicts, gold_labels):
        predicted = (
            verdict.get("criteria", {})
            .get(criterion, {})
            .get("verdict", "FAIL")
            .upper()
        )
        gold = gold.upper()

        if predicted == "PASS" and gold == "PASS":
            tp += 1
        elif predicted == "PASS" and gold == "FAIL":
            fp += 1
        elif predicted == "FAIL" and gold == "PASS":
            fn += 1
        else:
            tn += 1

        if predicted == gold:
            agreements += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Cohen's kappa
    observed_agreement = agreements / total if total > 0 else 0.0
    p_yes = ((tp + fp) / total) * ((tp + fn) / total) if total > 0 else 0.0
    p_no = ((fn + tn) / total) * ((fp + tn) / total) if total > 0 else 0.0
    expected_agreement = p_yes + p_no
    kappa = (
        (observed_agreement - expected_agreement) / (1 - expected_agreement)
        if expected_agreement < 1.0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
