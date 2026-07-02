import json
from types import SimpleNamespace

import smolagents.secb.review.reviewer as reviewer_module
from smolagents.secb.review.config import FinalizationReviewConfig
from smolagents.secb.review.reviewer import FinalizationReviewer, ReviewResult
from smolagents.secb.review.tools import (
    EmitFinalizationVerdictTool,
    FinalizationReviewToolState,
    ReadSynthesisLogTool,
)


class _Memory:
    steps = []


class _Agent:
    max_steps = 75
    step_number = 20


def _reviewer(tmp_path, *, guard_decide_fn=None):
    reviewer = FinalizationReviewer(
        config=FinalizationReviewConfig(enabled=True),
        static_context={
            "instance_id": "example.cve",
            "work_dir": "/src/example",
            "sanitizer_report": "AddressSanitizer target report",
        },
        synthesis_log_path=tmp_path / "synthesis_log.jsonl",
        log_dir=str(tmp_path),
        guard_decide_fn=guard_decide_fn,
    )
    reviewer.agent_ref = _Agent()
    return reviewer


def _state() -> FinalizationReviewToolState:
    return FinalizationReviewToolState(
        steps=[],
        valid_step_indices=set(),
        synthesis_log_records=[],
    )


def test_reviewer_prompt_is_strict_first_and_has_three_verdicts(tmp_path):
    rendered, _ = _reviewer(tmp_path)._render_prompt(
        {
            "block_index": 0,
            "artifact_status": "semantic pass",
            "stop_reason": "success",
        }
    )
    assert "Prefer strict success" in rendered
    assert "strict failure reason" in rendered
    assert "CALL_C_WITH_FAILURE_TABLE" not in rendered


def test_allow_success_requires_current_loose_pass():
    state = _state()
    state.repro_checked = True
    state.repro_result = {
        "validation": "no_match",
        "grader_passes": {
            "loose": False,
            "caller": False,
            "semantic": False,
            "strict": False,
        },
    }
    EmitFinalizationVerdictTool(state).forward(
        verdict="ALLOW_SUCCESS",
        reasoning="solver claimed success",
    )
    assert state.emitted_verdict == "CONTINUE_LOCAL"
    assert "does not pass even the loose" in state.emitted_attached_action


def test_allow_success_accepts_semantic_and_records_tier():
    state = _state()
    state.repro_checked = True
    state.repro_result = {
        "validation": "semantic",
        "grader_passes": {
            "loose": True,
            "caller": True,
            "semantic": True,
            "strict": False,
        },
        "strict_failure_reason": "line mismatch",
    }
    EmitFinalizationVerdictTool(state).forward(
        verdict="ALLOW_SUCCESS",
        reasoning="semantic pass; strict line mismatch is toolchain-specific",
    )
    assert state.emitted_verdict == "ALLOW_SUCCESS"


def test_allow_exhausted_still_requires_no_sanitizer():
    state = _state()
    state.repro_checked = True
    state.repro_result = {
        "validation": "no_match",
        "sanitizer_family": "AddressSanitizer",
    }
    EmitFinalizationVerdictTool(state).forward(
        verdict="ALLOW_EXHAUSTED",
        reasoning="claimed exhaustion",
    )
    assert state.emitted_verdict == "CONTINUE_LOCAL"
    assert "sanitizer firing" in state.emitted_attached_action


def test_unknown_verdict_normalizes_to_continue_local():
    state = _state()
    result = EmitFinalizationVerdictTool(state).forward(
        verdict="CALL_C_WITH_FAILURE_TABLE",
        reasoning="retired verdict",
    )
    assert result == "OK: verdict=CONTINUE_LOCAL emitted."
    assert state.emitted_verdict == "CONTINUE_LOCAL"


def test_synthesis_log_projection_is_compact_and_v1_compatible():
    state = _state()
    state.synthesis_log_records = [
        {
            "schema_version": "v1",
            "invocation_index": 1,
            "input": {"blocker_summary": "old blocker"},
            "raw_response": "do not expose",
            "c_tool_calls": [{"tool": "read_workspace_file"}],
            "emitted_payload": {
                "helper_files": ["Helpers/a.py"],
                "validation_outcome": "did_not_run",
                "candidate_table": [{"large": "row"}],
            },
        }
    ]
    result = json.loads(ReadSynthesisLogTool(state).forward())
    assert result[0]["delegated_problem"] == "old blocker"
    assert result[0]["validation"] == "did_not_run"
    assert "raw_response" not in result[0]
    assert "candidate_table" not in result[0]
    assert "c_tool_calls" not in result[0]


def test_reviewer_logs_repro_result(tmp_path, monkeypatch):
    reviewer = _reviewer(
        tmp_path,
        guard_decide_fn=lambda *args: SimpleNamespace(
            decision="allow",
            reason="test",
            remaining_steps=55,
            latest_gate="target_frame_seen",
        ),
    )

    def fake_run_b_agent(bundle, valid_step_indices):
        del bundle, valid_step_indices
        return ReviewResult(
            verdict="ALLOW_SUCCESS",
            reasoning="strict pass",
            attached_action="",
            evidence_steps=[],
            b_steps_used=1,
            b_tool_calls=[],
            degraded=False,
            degraded_reason="",
            prompt_hash="hash",
            dropped_invalid_evidence=[],
            repro_result={
                "validation": "strict",
                "grader_passes": {"strict": True},
            },
        )

    monkeypatch.setattr(reviewer, "_run_b_agent", fake_run_b_agent)
    allow, _ = reviewer.invoke(
        artifact_status="strict pass",
        stop_reason="success",
        memory=_Memory(),
    )
    assert allow is True
    record = json.loads((tmp_path / "finalization_review.jsonl").read_text())
    assert record["schema_version"] == "v2"
    assert record["repro_result"]["validation"] == "strict"
    assert "policy_mode" not in record
    assert "evidence_relation" not in record


def test_low_solver_step_budget_does_not_bypass_review(tmp_path, monkeypatch):
    reviewer = _reviewer(tmp_path)
    reviewer.agent_ref = SimpleNamespace(max_steps=75, step_number=74)
    called = {"value": False}

    def fake_run_b_agent(bundle, valid_step_indices):
        del bundle, valid_step_indices
        called["value"] = True
        return ReviewResult(
            verdict="CONTINUE_LOCAL",
            reasoning="success unsupported",
            attached_action="Resolve the target mismatch.",
            evidence_steps=[],
            b_steps_used=1,
            b_tool_calls=[],
            degraded=False,
            degraded_reason="",
            prompt_hash="hash",
            dropped_invalid_evidence=[],
        )

    monkeypatch.setattr(reviewer, "_run_b_agent", fake_run_b_agent)
    allow, _ = reviewer.invoke(
        artifact_status="no target signal",
        stop_reason="evidence_exhaustion",
        memory=_Memory(),
    )
    assert called["value"] is True
    assert allow is False


def test_degraded_reviewer_blocks(tmp_path, monkeypatch):
    reviewer = _reviewer(tmp_path)
    monkeypatch.setattr(reviewer_module, "LiteLLMModel", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(reviewer_module, "_run_with_timeout", lambda *args, **kwargs: None)
    result = reviewer._run_b_agent(
        {
            "steps": [],
            "valid_step_indices": set(),
            "synthesis_log_records": [],
            "work_dir": "/src/example",
            "target_signal": "AddressSanitizer target report",
        },
        set(),
    )
    assert result.verdict == "CONTINUE_LOCAL"
    assert result.degraded is True


def test_degraded_reviewer_classifies_role_budget_exhaustion(tmp_path, monkeypatch):
    reviewer = _reviewer(tmp_path)
    monkeypatch.setattr(reviewer_module, "LiteLLMModel", lambda **kwargs: SimpleNamespace())

    def stop_for_budget(agent, *args, **kwargs):
        del args, kwargs
        agent._termination_reason = "cost_budget_exhausted"

    monkeypatch.setattr(reviewer_module, "_run_with_timeout", stop_for_budget)
    result = reviewer._run_b_agent(
        {
            "steps": [],
            "valid_step_indices": set(),
            "synthesis_log_records": [],
            "work_dir": "/src/example",
            "target_signal": "AddressSanitizer target report",
        },
        set(),
    )

    assert result.verdict == "CONTINUE_LOCAL"
    assert result.degraded is True
    assert result.degraded_reason == "cost_budget_exhausted"
    assert result.termination_reason == "cost_budget_exhausted"
