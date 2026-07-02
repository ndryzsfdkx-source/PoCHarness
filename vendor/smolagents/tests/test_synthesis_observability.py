import json
from pathlib import Path
from types import SimpleNamespace

from smolagents.secb.harness.config import SynthesisConfig
from smolagents.secb.harness.synthesis import SynthesisBackendResult, SynthesisOrchestrator


def _step(step_number, command):
    return SimpleNamespace(
        step_number=step_number,
        tool_calls=[SimpleNamespace(name="cmd", arguments={"command": command})],
    )


def _orchestrator(tmp_path, monkeypatch, result):
    monkeypatch.setattr(
        "smolagents.secb.harness.synthesis.SynthesisAgentBackend.run",
        lambda self, bundle: result,
    )
    return SynthesisOrchestrator(
        SynthesisConfig(enabled=True, log_dir=str(tmp_path)),
        synthesis_context={"work_dir": str(tmp_path)},
    )


def _invoke(orchestrator, candidate):
    return orchestrator.invoke(
        delegated_problem="diagnose parser sequence",
        candidate_testcase_path=str(candidate),
        repro_output="rejected",
        target_signal="AddressSanitizer in target",
    )


def test_observability_v2_tracks_successful_emission(tmp_path: Path, monkeypatch):
    candidate = tmp_path / "poc"
    candidate.write_bytes(b"v1")
    result = SynthesisBackendResult(
        parse_status="ok",
        emitted_payload={
            "helper_files": ["Helpers/helper.py"],
            "how_to_run": "python3 Helpers/helper.py",
            "validation": "not_run",
        },
        c_steps_used=1,
        c_tool_calls=[],
        degraded=False,
        prompt="prompt",
    )
    orchestrator = _orchestrator(tmp_path, monkeypatch, result)
    orchestrator.agent_ref = SimpleNamespace(
        step_number=5,
        memory=SimpleNamespace(steps=[]),
    )
    _invoke(orchestrator, candidate)
    candidate.write_bytes(b"v2")
    orchestrator.agent_ref.memory.steps = [
        _step(6, "python3 Helpers/helper.py > /testcase/poc")
    ]
    orchestrator.finalize_observability()

    record = json.loads(
        (tmp_path / "synthesis_observability.jsonl").read_text().splitlines()[0]
    )
    assert record["schema_version"] == "v2"
    assert record["result_status"] == "emitted"
    assert record["a_step_at_result"] == 5
    assert record["a_followed"] is True
    assert record["a_adopted_into_testcase"] is True


def test_observability_v2_tracks_degraded_partial_files(tmp_path: Path, monkeypatch):
    candidate = tmp_path / "poc"
    candidate.write_bytes(b"v1")
    result = SynthesisBackendResult(
        parse_status="failed",
        emitted_payload=None,
        c_steps_used=30,
        c_tool_calls=[],
        degraded=True,
        prompt="prompt",
        degraded_reason="policy_block",
        partial_helper_files=["Helpers/partial.py"],
    )
    orchestrator = _orchestrator(tmp_path, monkeypatch, result)
    orchestrator.agent_ref = SimpleNamespace(
        step_number=12,
        memory=SimpleNamespace(steps=[]),
    )
    rendered = _invoke(orchestrator, candidate)
    assert "Helpers/partial.py" in rendered

    orchestrator.agent_ref.memory.steps = [
        _step(13, "python3 Helpers/partial.py > /testcase/poc")
    ]
    orchestrator.finalize_observability()
    record = json.loads(
        (tmp_path / "synthesis_observability.jsonl").read_text().splitlines()[0]
    )
    assert record["result_status"] == "degraded_partial"
    assert record["a_followed"] is True


def test_observability_is_idempotent(tmp_path: Path, monkeypatch):
    result = SynthesisBackendResult(
        parse_status="failed",
        emitted_payload=None,
        c_steps_used=30,
        c_tool_calls=[],
        degraded=True,
        prompt="prompt",
        partial_helper_files=["Helpers/partial.py"],
    )
    orchestrator = _orchestrator(tmp_path, monkeypatch, result)
    orchestrator.agent_ref = SimpleNamespace(
        step_number=1,
        memory=SimpleNamespace(steps=[]),
    )
    _invoke(orchestrator, tmp_path / "missing")
    orchestrator.finalize_observability()
    orchestrator.finalize_observability()
    assert len(
        (tmp_path / "synthesis_observability.jsonl").read_text().splitlines()
    ) == 1
