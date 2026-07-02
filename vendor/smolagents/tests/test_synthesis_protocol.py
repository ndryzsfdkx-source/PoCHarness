import json
from pathlib import Path
from types import SimpleNamespace

import smolagents.secb.harness.agent as agent_module
from smolagents.memory import ActionStep
from smolagents.monitoring import Timing
from smolagents.secb.harness.agent import (
    SynthesisHelperAgent,
    SynthesisAgentBackend,
    SynthesisBackendResult,
    _classify_degraded_reason,
)
from smolagents.secb.harness.config import SynthesisAgentBackendConfig, SynthesisConfig
from smolagents.secb.harness.synthesis import (
    SynthesisOrchestrator,
    _render_degraded,
    _render_observation,
)
from smolagents.secb.harness.tool import RequestSynthesisHelperTool
from smolagents.secb.harness.tools import EmitHelperTool, SynthesisToolState


TARGET = (
    "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n"
    "#0 0x1234 in target /src/target.c:10:2"
)


def _state(tmp_path: Path) -> SynthesisToolState:
    return SynthesisToolState(
        bundle={"invocation_index": 1, "oracle_output": TARGET},
        work_dir=str(tmp_path),
        backend_config=SynthesisAgentBackendConfig(),
    )


def _payload(index: int = 1) -> dict:
    return {
        "helper_files": [f"Helpers/h{index}.py"],
        "how_to_run": f"python3 Helpers/h{index}.py",
        "validation": "not_run",
        "grader_passes": {},
        "strict_failure_reason": "",
        "crossed_gate": "",
        "failed_gate": "target_frame_seen",
        "harness_evidence": "",
        "notes": f"helper {index}",
    }


def test_backend_config_filters_retired_tools_and_keeps_continuation_cap():
    cfg = SynthesisAgentBackendConfig.from_config(
        {
            "max_c_steps": 20,
            "continuation_max_c_steps": 12,
            "tools_allow": [
                "write_helper",
                "strict_crash_compare",
                "attempt_ledger_append",
                "emit_helper",
            ],
        }
    )
    assert cfg.max_c_steps == 20
    assert cfg.continuation_max_c_steps == 12
    assert cfg.tools_allow == ("write_helper", "emit_helper")


def test_specialist_prompt_receives_general_delegation(tmp_path: Path):
    backend = SynthesisAgentBackend(
        config=SynthesisConfig(enabled=True, log_dir=str(tmp_path)),
        static_context={
            "instance_id": "example.cve",
            "work_dir": str(tmp_path),
            "bug_description": "confirmed parser regression",
            "sanitizer_report": TARGET,
        },
        invocation_index=1,
    )
    rendered = backend._render_prompt(
        bundle={
            "delegated_problem": "reconstruct the semantic record sequence",
            "candidate_testcase_path": "/testcase/poc",
            "target_signal": "AddressSanitizer in target",
            "source_hints": ["parser.c"],
            "attempts_summary": "single-record input rejected",
            "prior_synthesis": [],
        }
    )
    assert "Synthesis Helper" in rendered
    assert "reconstruct the semantic record sequence" in rendered
    assert "attachments, reference PoCs, or fetch additional Git history" in rendered
    assert "field_sweep" not in rendered


def test_emit_accepts_nonempty_small_helper_and_derives_not_run(tmp_path: Path):
    helpers = tmp_path / "Helpers"
    helpers.mkdir()
    (helpers / "tiny.py").write_text("print('x')\n", encoding="utf-8")
    state = _state(tmp_path)

    result = EmitHelperTool(state).forward(
        helper_files=["Helpers/tiny.py"],
        how_to_run="python3 Helpers/tiny.py",
        failed_gate="parser state",
    )

    assert result == "OK: emitted manifest with 1 helper file(s)."
    assert state.emitted_payload["validation"] == "not_run"
    assert state.emitted_payload["failed_gate"] == "parser state"
    assert set(state.emitted_payload) == {
        "helper_files",
        "how_to_run",
        "validation",
        "grader_passes",
        "strict_failure_reason",
        "crossed_gate",
        "failed_gate",
        "harness_evidence",
        "notes",
    }


def test_emit_rejects_missing_empty_and_outside_files(tmp_path: Path):
    state = _state(tmp_path)
    tool = EmitHelperTool(state)
    assert "not found" in tool.forward(
        helper_files=["Helpers/missing.py"],
        how_to_run="python3 Helpers/missing.py",
    )

    helpers = tmp_path / "Helpers"
    helpers.mkdir()
    (helpers / "empty.py").write_text("", encoding="utf-8")
    assert "is empty" in tool.forward(
        helper_files=["Helpers/empty.py"],
        how_to_run="python3 Helpers/empty.py",
    )
    assert "must be under Helpers/" in tool.forward(
        helper_files=["/etc/passwd"],
        how_to_run="cat /etc/passwd",
    )


def test_emit_uses_latest_runtime_grading(tmp_path: Path):
    helpers = tmp_path / "Helpers"
    helpers.mkdir()
    (helpers / "probe.py").write_text("print('x')\n", encoding="utf-8")
    state = _state(tmp_path)
    state.latest_repro_result = {
        "validation": "semantic",
        "grader_passes": {
            "loose": True,
            "caller": True,
            "semantic": True,
            "strict": False,
        },
        "strict_failure_reason": "line mismatch",
    }
    EmitHelperTool(state).forward(
        helper_files=["Helpers/probe.py"],
        how_to_run="python3 Helpers/probe.py",
    )
    assert state.emitted_payload["validation"] == "semantic"
    assert state.emitted_payload["grader_passes"]["strict"] is False
    assert state.emitted_payload["strict_failure_reason"] == "line mismatch"
    assert "Validation: semantic" in _render_observation(1, state.emitted_payload)


def test_request_tool_uses_general_schema():
    class FakeOrchestrator:
        def invoke(self, **kwargs):
            self.kwargs = kwargs
            return "ok"

    fake = FakeOrchestrator()
    result = RequestSynthesisHelperTool(orchestrator=fake).forward(
        delegated_problem="diagnose coupled length invariants",
        candidate_testcase_path="/testcase/poc",
        repro_output="parser rejected record 2",
        target_signal="AddressSanitizer in target",
        source_hints=["parser.c"],
        attempts_summary="single-field changes failed",
        continue_from=2,
    )
    assert result == "ok"
    assert fake.kwargs["delegated_problem"] == "diagnose coupled length invariants"
    assert fake.kwargs["attempts_summary"] == "single-field changes failed"
    assert fake.kwargs["continue_from"] == 2
    assert "helper_type_hint" not in fake.kwargs
    assert "failure_table" not in fake.kwargs


def test_request_tool_rejects_non_testcase_path():
    tool = RequestSynthesisHelperTool(orchestrator=SimpleNamespace())
    result = tool.forward(
        delegated_problem="diagnose parser gate",
        candidate_testcase_path="/tmp/poc",
        repro_output="rejected",
        target_signal="AddressSanitizer in target",
    )
    assert result == "ERROR: candidate_testcase_path must be under /testcase/."


def test_explicit_continuation_injects_only_referenced_result(tmp_path: Path, monkeypatch):
    captured = []
    counter = {"value": 0}

    def fake_run(self, bundle):
        del self
        captured.append(dict(bundle))
        counter["value"] += 1
        return SynthesisBackendResult(
            parse_status="ok",
            emitted_payload=_payload(counter["value"]),
            c_steps_used=1,
            c_tool_calls=[],
            degraded=False,
            prompt="prompt",
        )

    monkeypatch.setattr(
        "smolagents.secb.harness.synthesis.SynthesisAgentBackend.run",
        fake_run,
    )
    orchestrator = SynthesisOrchestrator(
        SynthesisConfig(enabled=True, log_dir=str(tmp_path)),
        synthesis_context={"work_dir": str(tmp_path), "sanitizer_report": TARGET},
    )
    common = {
        "delegated_problem": "diagnose parser state",
        "candidate_testcase_path": "/testcase/poc",
        "repro_output": "rejected",
        "target_signal": "AddressSanitizer in target",
    }
    orchestrator.invoke(**common)
    orchestrator.invoke(**common, continue_from=1)
    orchestrator.invoke(**common, continue_from=99)

    assert captured[0]["continuation_mode"] is False
    assert captured[1]["continuation_mode"] is True
    assert [r["invocation_index"] for r in captured[1]["prior_synthesis"]] == [1]
    assert captured[2]["continuation_mode"] is False
    assert captured[2]["prior_synthesis"] == []


def test_degraded_partial_result_can_be_continued(tmp_path: Path, monkeypatch):
    captured = []

    def fake_run(self, bundle):
        del self
        captured.append(dict(bundle))
        if len(captured) == 1:
            return SynthesisBackendResult(
                parse_status="failed",
                emitted_payload=None,
                c_steps_used=30,
                c_tool_calls=[],
                degraded=True,
                prompt="prompt",
                degraded_reason="max_steps_exhausted",
                partial_helper_files=["Helpers/partial.py"],
            )
        return SynthesisBackendResult(
            parse_status="ok",
            emitted_payload=_payload(2),
            c_steps_used=1,
            c_tool_calls=[],
            degraded=False,
            prompt="prompt",
        )

    monkeypatch.setattr(
        "smolagents.secb.harness.synthesis.SynthesisAgentBackend.run",
        fake_run,
    )
    orchestrator = SynthesisOrchestrator(
        SynthesisConfig(enabled=True, log_dir=str(tmp_path)),
        synthesis_context={"work_dir": str(tmp_path), "sanitizer_report": TARGET},
    )
    kwargs = {
        "delegated_problem": "reconstruct sequence",
        "candidate_testcase_path": "/testcase/poc",
        "repro_output": "rejected",
        "target_signal": "AddressSanitizer in target",
    }
    first = orchestrator.invoke(**kwargs)
    orchestrator.invoke(**kwargs, continue_from=1)
    assert "Helpers/partial.py" in first
    assert captured[1]["prior_synthesis"][0]["degraded"] is True
    assert captured[1]["prior_synthesis"][0]["helper_files"] == ["Helpers/partial.py"]


def test_log_v2_is_compact(tmp_path: Path):
    orchestrator = SynthesisOrchestrator(
        SynthesisConfig(enabled=True, log_dir=str(tmp_path))
    )
    result = SynthesisBackendResult(
        parse_status="failed",
        emitted_payload=None,
        c_steps_used=31,
        c_tool_calls=[],
        degraded=True,
        prompt="prompt",
        raw_response="large raw model response",
        degraded_reason="max_steps_exhausted",
        partial_helper_files=["Helpers/partial.py"],
        step_budget={
            "mode": "fresh",
            "work_cap": 30,
            "work_steps": 30,
            "recovery_used": True,
            "recovery_steps": 1,
        },
    )
    orchestrator._write_log(
        invocation_index=1,
        a_step=20,
        bundle={
            "delegated_problem": "reconstruct sequence",
            "candidate_testcase_path": "/testcase/poc",
            "repro_output": "rejected",
            "target_signal": "AddressSanitizer in target",
            "source_hints": [],
            "attempts_summary": None,
            "continue_from": None,
            "continuation_mode": False,
        },
        result=result,
    )
    record = json.loads((tmp_path / "synthesis_log.jsonl").read_text())
    assert record["schema_version"] == "v2"
    assert record["step_budget"]["recovery_used"] is True
    assert "raw_response" not in record
    assert "helper_type_hint" not in record["input"]
    assert "failure_table" not in record["input"]


def test_degraded_reason_classifies_policy_block(tmp_path: Path):
    state = _state(tmp_path)
    reason = _classify_degraded_reason(
        state=state,
        error_message="backend content filter policy violation",
        termination_reason="",
        c_steps_used=12,
        max_c_steps=30,
    )
    assert reason == "policy_block"


def test_degraded_reason_prefers_role_budget_termination(tmp_path: Path):
    state = _state(tmp_path)
    reason = _classify_degraded_reason(
        state=state,
        error_message="",
        termination_reason="cost_budget_exhausted",
        c_steps_used=5,
        max_c_steps=50,
    )
    assert reason == "cost_budget_exhausted"


def test_degraded_render_includes_partial_files():
    result = SynthesisBackendResult(
        parse_status="failed",
        emitted_payload=None,
        c_steps_used=30,
        c_tool_calls=[],
        degraded=True,
        prompt="",
        degraded_reason="max_steps_exhausted",
        partial_helper_files=["Helpers/partial.py"],
    )
    rendered = _render_degraded(2, result)
    assert "[SYNTHESIS SPECIALIST #2" in rendered
    assert "Helpers/partial.py" in rendered


def test_emit_terminality_requires_acceptance_or_retry_exhaustion(tmp_path: Path):
    fake_self = SimpleNamespace(max_emit_attempts=SynthesisHelperAgent.max_emit_attempts)
    decide = SynthesisHelperAgent._emit_is_terminal
    rejected = _state(tmp_path)
    rejected.emit_attempts = 1
    assert decide(fake_self, rejected) is False
    rejected.emit_attempts = SynthesisHelperAgent.max_emit_attempts
    assert decide(fake_self, rejected) is True
    rejected.emitted_payload = _payload()
    assert decide(fake_self, rejected) is True


def test_backend_uses_one_emission_only_recovery_turn(tmp_path: Path, monkeypatch):
    class FakeAgent:
        def __init__(self, tools, **kwargs):
            del kwargs
            self.tools = {tool.name: tool for tool in tools}
            self.memory = SimpleNamespace(steps=[])
            self.monitor = SimpleNamespace()
            self.calls = 0

        def run(self, task, reset=True, max_steps=None, **kwargs):
            del task, kwargs
            self.calls += 1
            if reset:
                helper = tmp_path / "Helpers" / "recover.py"
                helper.parent.mkdir(exist_ok=True)
                helper.write_text("print('recovered')\n", encoding="utf-8")
                self.memory.steps.extend(
                    ActionStep(step_number=i, timing=Timing(start_time=0))
                    for i in range(1, 31)
                )
                return "work cap"
            assert max_steps == 1
            assert set(self.tools) == {"write_helper", "emit_helper"}
            self.tools["emit_helper"].forward(
                helper_files=["Helpers/recover.py"],
                how_to_run="python3 Helpers/recover.py",
            )
            self.memory.steps.append(
                ActionStep(step_number=1, timing=Timing(start_time=0))
            )
            return "emitted"

    monkeypatch.setattr(agent_module, "LiteLLMModel", lambda **kwargs: object())
    monkeypatch.setattr(agent_module, "SynthesisHelperAgent", FakeAgent)
    backend = SynthesisAgentBackend(
        config=SynthesisConfig(
            enabled=True,
            log_dir=str(tmp_path),
            agent_backend=SynthesisAgentBackendConfig(max_c_steps=30),
        ),
        static_context={"work_dir": str(tmp_path), "sanitizer_report": TARGET},
        invocation_index=1,
    )
    result = backend.run(
        {
            "delegated_problem": "materialize final helper",
            "candidate_testcase_path": "/testcase/poc",
            "repro_output": "rejected",
            "target_signal": "AddressSanitizer in target",
            "continuation_mode": False,
        }
    )
    assert result.degraded is False, result.error_message
    assert result.step_budget == {
        "mode": "fresh",
        "work_cap": 30,
        "work_steps": 30,
        "recovery_used": True,
        "recovery_steps": 1,
    }


def test_backend_recovers_early_stop_after_helper_was_written(tmp_path: Path, monkeypatch):
    class FakeAgent:
        def __init__(self, tools, **kwargs):
            del kwargs
            self.tools = {tool.name: tool for tool in tools}
            self.memory = SimpleNamespace(steps=[])
            self.monitor = SimpleNamespace()
            self.calls = 0

        def run(self, task, reset=True, max_steps=None, **kwargs):
            del task, kwargs
            self.calls += 1
            if reset:
                assert max_steps is None
                self.tools["write_helper"].forward(
                    filename="Helpers/early.py",
                    content="print('ready')\n",
                )
                self.memory.steps.extend(
                    ActionStep(step_number=i, timing=Timing(start_time=0))
                    for i in range(1, 4)
                )
                return "stopped before emitting"
            assert max_steps == 1
            assert set(self.tools) == {"write_helper", "emit_helper"}
            self.tools["emit_helper"].forward(
                helper_files=["Helpers/early.py"],
                how_to_run="python3 Helpers/early.py",
            )
            self.memory.steps.append(ActionStep(step_number=1, timing=Timing(start_time=0)))
            return "emitted"

    monkeypatch.setattr(agent_module, "LiteLLMModel", lambda **kwargs: object())
    monkeypatch.setattr(agent_module, "SynthesisHelperAgent", FakeAgent)
    backend = SynthesisAgentBackend(
        config=SynthesisConfig(
            enabled=True,
            log_dir=str(tmp_path),
            agent_backend=SynthesisAgentBackendConfig(max_c_steps=30),
        ),
        static_context={"work_dir": str(tmp_path), "sanitizer_report": TARGET},
        invocation_index=1,
    )

    result = backend.run(
        {
            "delegated_problem": "emit the completed helper",
            "candidate_testcase_path": "/testcase/poc",
            "repro_output": "rejected",
            "target_signal": "AddressSanitizer in target",
            "continuation_mode": False,
        }
    )

    assert result.degraded is False, result.error_message
    assert result.partial_helper_files == ["Helpers/early.py"]
    assert result.step_budget == {
        "mode": "fresh",
        "work_cap": 30,
        "work_steps": 3,
        "recovery_used": True,
        "recovery_steps": 1,
    }


def test_backend_does_not_recover_early_stop_without_helper(tmp_path: Path, monkeypatch):
    class FakeAgent:
        def __init__(self, tools, **kwargs):
            del kwargs
            self.tools = {tool.name: tool for tool in tools}
            self.memory = SimpleNamespace(steps=[])
            self.monitor = SimpleNamespace()
            self.calls = 0

        def run(self, task, reset=True, max_steps=None, **kwargs):
            del task, kwargs
            self.calls += 1
            assert reset is True
            assert max_steps is None
            self.memory.steps.extend(
                ActionStep(step_number=i, timing=Timing(start_time=0))
                for i in range(1, 4)
            )
            return "stopped without an artifact"

    agents = []

    def build_agent(*args, **kwargs):
        agent = FakeAgent(*args, **kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(agent_module, "LiteLLMModel", lambda **kwargs: object())
    monkeypatch.setattr(agent_module, "SynthesisHelperAgent", build_agent)
    backend = SynthesisAgentBackend(
        config=SynthesisConfig(
            enabled=True,
            log_dir=str(tmp_path),
            agent_backend=SynthesisAgentBackendConfig(max_c_steps=30),
        ),
        static_context={"work_dir": str(tmp_path), "sanitizer_report": TARGET},
        invocation_index=1,
    )

    result = backend.run(
        {
            "delegated_problem": "find a construction path",
            "candidate_testcase_path": "/testcase/poc",
            "repro_output": "rejected",
            "target_signal": "AddressSanitizer in target",
            "continuation_mode": False,
        }
    )

    assert agents[0].calls == 1
    assert result.degraded is True
    assert result.degraded_reason == "emit_not_called"
    assert result.partial_helper_files == []
    assert result.step_budget == {
        "mode": "fresh",
        "work_cap": 30,
        "work_steps": 3,
        "recovery_used": False,
        "recovery_steps": 0,
    }
