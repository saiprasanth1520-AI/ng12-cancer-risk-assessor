"""Unit tests for the LLM-as-a-Judge framework.

Tests cover structural logic (JSON parsing, verdict aggregation, per-criterion
evaluation, metrics computation) without requiring a live LLM connection.
Integration tests that call the actual judge LLM are in test_evaluation.py.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from app.llm_judge import (
    _parse_judge_json,
    _format_context,
    _format_patient,
    _build_verdict,
    _default_verdict,
    _evaluate_single_criterion,
    judge_assessment,
    judge_chat,
    compute_classification_metrics,
)


# ── _parse_judge_json ──────────────────────────────────────────────────

class TestParseJudgeJson:
    def test_plain_json(self):
        raw = '{"verdict": "PASS", "reasoning": "ok"}'
        result = _parse_judge_json(raw)
        assert result["verdict"] == "PASS"

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"verdict": "FAIL"}\n```'
        result = _parse_judge_json(raw)
        assert result["verdict"] == "FAIL"

    def test_json_with_bare_fences(self):
        raw = '```\n{"key": "value"}\n```'
        result = _parse_judge_json(raw)
        assert result["key"] == "value"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_judge_json("not json at all")


# ── _format_context ────────────────────────────────────────────────────

class TestFormatContext:
    def test_empty_citations(self):
        result = _format_context([])
        assert "No NG12 passages" in result

    def test_formats_chunks(self):
        chunks = [
            {"text": "Hemoptysis referral", "page": 42, "chunk_id": "c1"},
            {"text": "Lung cancer criteria", "page": 43, "chunk_id": "c2"},
        ]
        result = _format_context(chunks)
        assert "[1]" in result
        assert "[2]" in result
        assert "page 42" in result
        assert "Hemoptysis referral" in result

    def test_uses_excerpt_fallback(self):
        chunks = [{"excerpt": "Some excerpt", "page": 1, "chunk_id": "x"}]
        result = _format_context(chunks)
        assert "Some excerpt" in result


# ── _format_patient ────────────────────────────────────────────────────

class TestFormatPatient:
    def test_empty_patient(self):
        result = _format_patient({})
        assert "unavailable" in result

    def test_none_patient(self):
        result = _format_patient(None)
        assert "unavailable" in result

    def test_full_patient(self):
        patient = {
            "patient_id": "PT-101",
            "name": "John Doe",
            "age": 55,
            "gender": "male",
            "smoking_history": "current",
            "symptoms": ["hemoptysis", "fatigue"],
            "symptom_duration_days": 14,
        }
        result = _format_patient(patient)
        assert "PT-101" in result
        assert "John Doe" in result
        assert "hemoptysis, fatigue" in result


# ── _build_verdict ─────────────────────────────────────────────────────

class TestBuildVerdict:
    def test_all_pass(self):
        criteria = {
            "faithfulness": {"verdict": "PASS", "reasoning": "good"},
            "correctness": {"verdict": "PASS", "reasoning": "right"},
        }
        result = _build_verdict(criteria)
        assert result["overall_verdict"] == "PASS"
        assert result["score"] == "2/2"
        assert result["criteria"]["faithfulness"]["verdict"] == "PASS"
        assert result["critical_issues"] == []

    def test_one_fail(self):
        criteria = {
            "faithfulness": {"verdict": "FAIL", "reasoning": "bad claim found"},
            "correctness": {"verdict": "PASS", "reasoning": "right"},
        }
        result = _build_verdict(criteria)
        assert result["overall_verdict"] == "FAIL"
        assert result["score"] == "1/2"
        assert len(result["critical_issues"]) == 1
        assert "faithfulness" in result["critical_issues"][0]

    def test_all_fail(self):
        criteria = {
            "faithfulness": {"verdict": "FAIL", "reasoning": "bad"},
            "correctness": {"verdict": "FAIL", "reasoning": "wrong"},
            "safety": {"verdict": "FAIL", "reasoning": "unsafe"},
        }
        result = _build_verdict(criteria)
        assert result["overall_verdict"] == "FAIL"
        assert result["score"] == "0/3"
        assert len(result["critical_issues"]) == 3

    def test_empty_criteria(self):
        result = _build_verdict({})
        assert result["overall_verdict"] == "PASS"
        assert result["score"] == "0/0"

    def test_five_criteria_mixed(self):
        criteria = {
            "faithfulness": {"verdict": "PASS", "reasoning": "ok"},
            "correctness": {"verdict": "PASS", "reasoning": "ok"},
            "citation_accuracy": {"verdict": "PASS", "reasoning": "ok"},
            "completeness": {"verdict": "FAIL", "reasoning": "missed fatigue"},
            "safety": {"verdict": "PASS", "reasoning": "ok"},
        }
        result = _build_verdict(criteria)
        assert result["overall_verdict"] == "FAIL"
        assert result["score"] == "4/5"
        assert result["cross_examination"] is None


# ── _evaluate_single_criterion (mocked LLM) ───────────────────────────

class TestEvaluateSingleCriterion:
    @patch("app.llm_judge._call_judge_llm")
    def test_successful_evaluation(self, mock_llm):
        mock_llm.return_value = {
            "text": json.dumps({
                "reasoning": "All claims grounded in context",
                "verdict": "PASS",
            }),
            "tokens_in": 150,
            "tokens_out": 30,
            "latency_ms": 1200.5,
        }
        result = _evaluate_single_criterion(
            "prompt {context}",
            {"context": "some context"},
            "faithfulness",
        )
        assert result["verdict"] == "PASS"
        assert "grounded" in result["reasoning"]
        assert result["tokens_in"] == 150
        assert result["tokens_out"] == 30
        assert result["latency_ms"] == 1200.5

    @patch("app.llm_judge._call_judge_llm")
    def test_fail_verdict(self, mock_llm):
        mock_llm.return_value = {
            "text": json.dumps({
                "reasoning": "Claim about age cutoff not in passages",
                "verdict": "FAIL",
            }),
            "tokens_in": 140,
            "tokens_out": 25,
            "latency_ms": 900.0,
        }
        result = _evaluate_single_criterion(
            "prompt {context}",
            {"context": "some context"},
            "faithfulness",
        )
        assert result["verdict"] == "FAIL"

    @patch("app.llm_judge._call_judge_llm")
    def test_invalid_verdict_defaults_to_fail(self, mock_llm):
        mock_llm.return_value = {
            "text": json.dumps({
                "reasoning": "unsure",
                "verdict": "MAYBE",
            }),
            "tokens_in": 100,
            "tokens_out": 20,
            "latency_ms": 800.0,
        }
        result = _evaluate_single_criterion(
            "prompt {context}",
            {"context": "some context"},
            "faithfulness",
        )
        assert result["verdict"] == "FAIL"

    @patch("app.llm_judge._call_judge_llm")
    def test_llm_error_returns_fail(self, mock_llm):
        mock_llm.side_effect = TimeoutError("timeout")
        result = _evaluate_single_criterion(
            "prompt {context}",
            {"context": "some context"},
            "faithfulness",
        )
        assert result["verdict"] == "FAIL"
        assert "failed" in result["reasoning"].lower()


# ── _default_verdict ───────────────────────────────────────────────────

class TestDefaultVerdict:
    def test_returns_unavailable(self):
        result = _default_verdict("some reason")
        assert result["overall_verdict"] == "UNAVAILABLE"
        assert result["score"] == "0/0"
        assert "some reason" in result["critical_issues"]


# ── judge_assessment (mocked per-criterion evaluators) ─────────────────

class TestJudgeAssessmentMocked:
    @patch("app.llm_judge.LLM_JUDGE_ENABLED", False)
    def test_disabled_returns_default(self):
        result = judge_assessment({}, {}, [])
        assert result["overall_verdict"] == "UNAVAILABLE"

    @patch("app.llm_judge._evaluate_single_criterion")
    def test_five_criteria_all_pass(self, mock_eval):
        mock_eval.return_value = {"verdict": "PASS", "reasoning": "ok"}

        assessment = {
            "patient_id": "PT-101",
            "risk_level": "Urgent Referral (2-week wait)",
            "reasoning": "Hemoptysis suggests lung cancer per NG12",
            "citations": [{"page": 42, "excerpt": "hemoptysis", "chunk_id": "c1", "source": "NG12"}],
            "recommendations": ["Urgent chest X-ray"],
        }
        patient = {
            "patient_id": "PT-101",
            "name": "John",
            "age": 55,
            "gender": "male",
            "smoking_history": "current",
            "symptoms": ["hemoptysis", "fatigue"],
            "symptom_duration_days": 14,
        }

        result = judge_assessment(assessment, patient, [])
        assert result["overall_verdict"] == "PASS"
        assert result["score"] == "5/5"
        assert mock_eval.call_count == 5

    @patch("app.llm_judge._evaluate_single_criterion")
    def test_mixed_verdicts(self, mock_eval):
        # Return PASS for all except completeness
        def side_effect(prompt, kwargs, criterion_name):
            if criterion_name == "completeness":
                return {"verdict": "FAIL", "reasoning": "missed fatigue"}
            return {"verdict": "PASS", "reasoning": "ok"}

        mock_eval.side_effect = side_effect

        result = judge_assessment(
            {"patient_id": "PT-101", "risk_level": "Urgent Referral (2-week wait)",
             "reasoning": "test", "citations": [], "recommendations": []},
            {"symptoms": ["hemoptysis", "fatigue"]},
            [],
        )
        assert result["overall_verdict"] == "FAIL"
        assert result["score"] == "4/5"
        assert result["criteria"]["completeness"]["verdict"] == "FAIL"

    @patch("app.llm_judge._evaluate_single_criterion")
    def test_evaluator_exception_returns_default(self, mock_eval):
        mock_eval.side_effect = Exception("total failure")
        result = judge_assessment({"patient_id": "PT-101"}, {}, [])
        assert result["overall_verdict"] == "UNAVAILABLE"


# ── judge_chat (mocked per-criterion evaluators) ──────────────────────

class TestJudgeChatMocked:
    @patch("app.llm_judge._evaluate_single_criterion")
    def test_three_criteria_all_pass(self, mock_eval):
        mock_eval.return_value = {"verdict": "PASS", "reasoning": "ok"}

        result = judge_chat(
            "What symptoms need urgent lung cancer referral?",
            "Per NG12, hemoptysis requires 2WW referral [NG12 p.42]",
            [{"text": "hemoptysis referral", "page": 42, "chunk_id": "c1"}],
        )
        assert result["overall_verdict"] == "PASS"
        assert result["score"] == "3/3"
        assert mock_eval.call_count == 3

    @patch("app.llm_judge._evaluate_single_criterion")
    def test_relevance_fails(self, mock_eval):
        def side_effect(prompt, kwargs, criterion_name):
            if criterion_name == "relevance":
                return {"verdict": "FAIL", "reasoning": "off topic"}
            return {"verdict": "PASS", "reasoning": "ok"}

        mock_eval.side_effect = side_effect

        result = judge_chat("What about lung cancer?", "Here's info on diabetes.", [])
        assert result["overall_verdict"] == "FAIL"
        assert result["score"] == "2/3"

    @patch("app.llm_judge.LLM_JUDGE_ENABLED", False)
    def test_disabled_returns_default(self):
        result = judge_chat("q", "a", [])
        assert result["overall_verdict"] == "UNAVAILABLE"


# ── compute_classification_metrics ─────────────────────────────────────

class TestClassificationMetrics:
    def test_perfect_scores(self):
        verdicts = [
            {"criteria": {"correctness": {"verdict": "PASS"}}},
            {"criteria": {"correctness": {"verdict": "FAIL"}}},
            {"criteria": {"correctness": {"verdict": "PASS"}}},
        ]
        gold = ["PASS", "FAIL", "PASS"]
        result = compute_classification_metrics(verdicts, "correctness", gold)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["cohens_kappa"] == 1.0

    def test_all_wrong(self):
        verdicts = [
            {"criteria": {"correctness": {"verdict": "FAIL"}}},
            {"criteria": {"correctness": {"verdict": "PASS"}}},
        ]
        gold = ["PASS", "FAIL"]
        result = compute_classification_metrics(verdicts, "correctness", gold)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_mixed_results(self):
        verdicts = [
            {"criteria": {"safety": {"verdict": "PASS"}}},
            {"criteria": {"safety": {"verdict": "PASS"}}},
            {"criteria": {"safety": {"verdict": "FAIL"}}},
            {"criteria": {"safety": {"verdict": "FAIL"}}},
        ]
        gold = ["PASS", "FAIL", "PASS", "FAIL"]
        result = compute_classification_metrics(verdicts, "safety", gold)
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5
        assert result["confusion_matrix"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_classification_metrics([{}], "x", ["PASS", "FAIL"])

    def test_missing_criterion_defaults_to_fail(self):
        verdicts = [{"criteria": {}}]
        gold = ["FAIL"]
        result = compute_classification_metrics(verdicts, "correctness", gold)
        assert result["confusion_matrix"]["tn"] == 1
