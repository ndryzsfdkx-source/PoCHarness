"""Hermetic proof that sanitizer_report does not reach B/C in description_mode.

Three check-points, each exercising real production code:
  1. cli.py boundary  — _build_agent_config_and_synthesis_context (the sole channel to B/C)
  2. synthesis.py     — oracle_output in C's bundle (monkeypatched backend captures the dict)
  3. finalization_guard.py:226 — target_signal fallback reads self.context["sanitizer_report"]

Each check has a description_mode=True absence-assert AND a description_mode=False positive
control so the tests can actually fail if the boundary regresses.
"""
import json
from pathlib import Path

import pytest

from smolagents.cli import _build_agent_config_and_synthesis_context
from smolagents.secb.harness.config import SynthesisConfig
from smolagents.secb.harness.finalization_guard import SynthesisFinalizationGuard
from smolagents.secb.harness.synthesis import SynthesisBackendResult, SynthesisOrchestrator

SENTINEL = "LEAK_SENTINEL_ASAN_heap_bof_0xdeadbeef"

_BASE_CONFIG = {
    "model": {"type": "LiteLLMModel", "model_id": "gpt-4o"},
    "agent": {"pocharness": {"enabled": True}},
    "managed_agents": {},
}


def _fake_instance() -> dict:
    return {
        "instance_id": "test.cve-0000-00000",
        "work_dir": "/work",
        "bug_description": "heap-buffer-overflow in foo_parse",
        "sanitizer_report": SENTINEL,
    }


def _make_payload(n: int = 1) -> dict:
    return {
        "helper_files": [f"Helpers/h{n}.py"],
        "how_to_run": f"python3 Helpers/h{n}.py",
        "validation": "not_run",
        "grader_passes": {},
        "strict_failure_reason": "",
        "crossed_gate": "",
        "failed_gate": "target_frame_seen",
        "harness_evidence": "",
        "notes": f"helper {n}",
    }


# ---------------------------------------------------------------------------
# Check 1 — cli.py:_build_agent_config_and_synthesis_context (the boundary)
# ---------------------------------------------------------------------------

def test_synthesis_context_excludes_sentinel_in_description_mode():
    # Explicit description_mode=true in config (the path used by the poc-desc A+B+C configs)
    config = {**_BASE_CONFIG, "agent": {"pocharness": {"enabled": True, "description_mode": True}}}
    agent_cfg, ctx = _build_agent_config_and_synthesis_context(
        _fake_instance(), config, "poc-desc"
    )
    assert SENTINEL not in ctx
    assert SENTINEL not in json.dumps(agent_cfg)


def test_synthesis_context_excludes_sentinel_via_task_type_derivation():
    # description_mode derived from task_type="poc-desc" (no explicit TOML key)
    config = {**_BASE_CONFIG}
    agent_cfg, ctx = _build_agent_config_and_synthesis_context(
        _fake_instance(), config, "poc-desc"
    )
    assert SENTINEL not in ctx
    assert SENTINEL not in json.dumps(agent_cfg)


def test_synthesis_context_includes_sentinel_when_not_description_mode():
    # Positive control — poc-san must keep sanitizer_report so the check can fail
    config = {**_BASE_CONFIG}
    agent_cfg, ctx = _build_agent_config_and_synthesis_context(
        _fake_instance(), config, "poc-san"
    )
    assert ctx.get("sanitizer_report") == SENTINEL
    assert SENTINEL in json.dumps(agent_cfg)


# ---------------------------------------------------------------------------
# Check 2 — synthesis.py:185-192 oracle_output in C's bundle
# ---------------------------------------------------------------------------

def _capturing_run(captured: list):
    def fake_run(self, bundle):
        del self
        captured.append(dict(bundle))
        return SynthesisBackendResult(
            parse_status="ok",
            emitted_payload=_make_payload(),
            c_steps_used=1,
            c_tool_calls=[],
            degraded=False,
            prompt="prompt",
        )
    return fake_run


def test_oracle_output_empty_in_description_mode(tmp_path: Path, monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        "smolagents.secb.harness.synthesis.SynthesisAgentBackend.run",
        _capturing_run(captured),
    )
    # synthesis_context built by the real boundary — no sanitizer_report present
    _, ctx = _build_agent_config_and_synthesis_context(
        _fake_instance(), {**_BASE_CONFIG, "agent": {"pocharness": {"enabled": True, "description_mode": True}}}, "poc-desc"
    )
    orchestrator = SynthesisOrchestrator(
        SynthesisConfig(enabled=True, log_dir=str(tmp_path), description_mode=True),
        synthesis_context=ctx,
    )
    orchestrator.invoke(
        delegated_problem="diagnose crash",
        candidate_testcase_path="/testcase/poc",
        repro_output="rejected",
        target_signal="AddressSanitizer in foo_parse",
    )
    assert len(captured) == 1
    assert captured[0]["oracle_output"] == ""
    assert SENTINEL not in captured[0]["oracle_output"]


def test_oracle_output_contains_sentinel_when_not_description_mode(tmp_path: Path, monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        "smolagents.secb.harness.synthesis.SynthesisAgentBackend.run",
        _capturing_run(captured),
    )
    # synthesis_context built by the real boundary — sanitizer_report present (poc-san path)
    _, ctx = _build_agent_config_and_synthesis_context(
        _fake_instance(), _BASE_CONFIG, "poc-san"
    )
    orchestrator = SynthesisOrchestrator(
        SynthesisConfig(enabled=True, log_dir=str(tmp_path), description_mode=False),
        synthesis_context=ctx,
    )
    orchestrator.invoke(
        delegated_problem="diagnose crash",
        candidate_testcase_path="/testcase/poc",
        repro_output="rejected",
        target_signal="",  # empty so fallback to sanitizer_report exercises the else branch
    )
    assert len(captured) == 1
    assert SENTINEL in captured[0]["oracle_output"]


# ---------------------------------------------------------------------------
# Check 3 — finalization_guard.py:226 target_signal fallback (self.context)
# ---------------------------------------------------------------------------

def test_guard_context_has_no_sentinel_in_description_mode():
    # Context from the real boundary — sanitizer_report stripped
    _, ctx = _build_agent_config_and_synthesis_context(
        _fake_instance(), {**_BASE_CONFIG, "agent": {"pocharness": {"enabled": True, "description_mode": True}}}, "poc-desc"
    )
    guard = SynthesisFinalizationGuard(synthesis_context=ctx)
    # Line 226 fallback: str(self.context.get("sanitizer_report") or "")
    assert guard.context.get("sanitizer_report") is None
    assert SENTINEL not in str(guard.context.get("sanitizer_report") or "")


def test_guard_context_has_sentinel_when_not_description_mode():
    # Positive control — poc-san context must carry the sentinel
    _, ctx = _build_agent_config_and_synthesis_context(
        _fake_instance(), _BASE_CONFIG, "poc-san"
    )
    guard = SynthesisFinalizationGuard(synthesis_context=ctx)
    assert guard.context.get("sanitizer_report") == SENTINEL
