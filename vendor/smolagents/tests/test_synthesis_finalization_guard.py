import json
from pathlib import Path

import smolagents.secb.harness.finalization_guard as finalization_guard
from smolagents.memory import ActionStep, Timing, ToolCall
from smolagents.secb.harness.finalization_guard import (
    GUARD_LOG_NAME,
    SynthesisFinalizationGuard,
)
from smolagents.secb.harness.config import SynthesisFinalizationGuardConfig


TARGET_REPORT = (
    "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014\n"
    "READ of size 4 at 0x602000000014 thread T0\n"
    "    #0 0x1234 in sycc420_to_rgb /src/openjpeg/src/bin/common/color.c:379:42\n"
)


def _step(tool_name: str, args: dict, observations: str = "", step_number: int = 1) -> ActionStep:
    return ActionStep(
        step_number=step_number,
        timing=Timing(start_time=0.0),
        tool_calls=[ToolCall(name=tool_name, arguments=args, id=f"call_{step_number}")],
        observations=observations,
    )


def _c_step(
    step_number: int = 2,
    gate: str = "packet decode consistency",
    target_signal: str = TARGET_REPORT,
) -> ActionStep:
    return _step(
        "request_synthesis_helper",
        {
            "blocker_summary": "blocked at decoder gate",
            "helper_type_hint": "format_constructor",
            "candidate_testcase_path": "/testcase/poc.j2k",
            "repro_output": "no sanitizer",
            "target_signal": target_signal,
        },
        observations=(
            "[SYNTHESIS HELPER #1 - type=format_constructor]\n"
            f"Failed gate: {gate}\n"
            "[END SYNTHESIS HELPER]"
        ),
        step_number=step_number,
    )


class _Memory:
    def __init__(self, steps):
        self.steps = steps


class _Agent:
    def __init__(self, *, max_steps=75, step_number=20):
        self.max_steps = max_steps
        self.step_number = step_number


def _guard(tmp_path: Path) -> SynthesisFinalizationGuard:
    return SynthesisFinalizationGuard(
        guard_config={"enabled": True, "min_remaining_steps": 8},
        synthesis_context={"instance_id": "openjpeg.cve-2021-3575", "sanitizer_report": TARGET_REPORT},
        log_dir=str(tmp_path),
    )


def _log_records(tmp_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (tmp_path / GUARD_LOG_NAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_guard_allows_no_c_finalization(tmp_path: Path):
    guard = _guard(tmp_path)
    memory = _Memory([_step("cmd", {"command": "secb repro"}, "target crash reached", 1)])

    assert guard.final_answer_check("A solved this locally.", memory, _Agent(step_number=12)) is True

    record = _log_records(tmp_path)[0]
    assert record["decision"] == "allow"
    assert record["reason"] == "no_synthesis_helper_used"


def test_guard_no_c_path_does_not_scan_final_answer_for_next_action(tmp_path: Path, monkeypatch):
    def fail_if_called(final_answer):
        raise AssertionError("next-action scan should not run without synthesis helper use")

    monkeypatch.setattr(finalization_guard, "_final_answer_names_concrete_next_action", fail_if_called)
    guard = _guard(tmp_path)
    memory = _Memory([_step("cmd", {"command": "secb repro"}, "target crash reached", 1)])

    assert guard.final_answer_check(
        "A promising next step would be to inspect another branch.",
        memory,
        _Agent(step_number=12),
    ) is True

    record = _log_records(tmp_path)[0]
    assert record["reason"] == "no_synthesis_helper_used"
    assert record["concrete_next_action_in_final"] is False


def test_guard_logs_unfollowed_latest_c_gate(tmp_path: Path):
    guard = _guard(tmp_path)
    memory = _Memory([_c_step(step_number=10)])

    assert guard.final_answer_check("I cannot reproduce it.", memory, _Agent(step_number=11)) is True

    record = _log_records(tmp_path)[0]
    assert record["decision"] == "block"
    assert record["reason"] == "latest_gate_unattempted:packet decode consistency"
    assert record["latest_gate"] == "packet decode consistency"
    assert record["follow_up_after_latest_c"] is False


def test_guard_logs_final_answer_that_names_concrete_next_action(tmp_path: Path):
    guard = _guard(tmp_path)
    memory = _Memory(
        [
            _c_step(step_number=10),
            _step("cmd", {"command": "python3 Helpers/ctor.py && secb repro"}, "still no sanitizer", 11),
        ]
    )

    assert guard.final_answer_check(
        "A promising next step would be to inspect the packet decode branch.",
        memory,
        _Agent(step_number=12),
    ) is True

    record = _log_records(tmp_path)[0]
    assert record["decision"] == "block"
    assert record["reason"] == "final_answer_names_concrete_next_action"
    assert record["follow_up_after_latest_c"] is True
    assert record["concrete_next_action_in_final"] is True


def test_guard_allows_low_budget_final_even_with_gate(tmp_path: Path):
    guard = _guard(tmp_path)
    memory = _Memory([_c_step(step_number=68)])

    assert guard.final_answer_check(
        "I would need to inspect another decode branch.",
        memory,
        _Agent(max_steps=75, step_number=70),
    ) is True

    assert _log_records(tmp_path)[0]["reason"] == "low_step_budget"


def test_guard_allows_when_target_signal_was_observed(tmp_path: Path):
    guard = _guard(tmp_path)
    target_obs = (
        "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014\n"
        "READ of size 4 at 0x602000000014 thread T0\n"
        "    #0 0x1234 in sycc420_to_rgb /src/openjpeg/src/bin/common/color.c:379:42\n"
    )
    memory = _Memory(
        [
            _c_step(step_number=10),
            _step("cmd", {"command": "secb repro"}, target_obs, 11),
        ]
    )

    assert guard.final_answer_check(
        "A promising next step would be to minimize it.",
        memory,
        _Agent(step_number=12),
    ) is True

    record = _log_records(tmp_path)[0]
    assert record["reason"] == "target_signal_observed"
    assert record["target_reached"] is True


def test_guard_default_config_is_opt_in():
    assert SynthesisFinalizationGuardConfig.from_config(None).enabled is False


def test_degraded_second_c_does_not_hide_previous_gate_in_telemetry(tmp_path: Path):
    guard = _guard(tmp_path)
    degraded = _step(
        "request_synthesis_helper",
        {
            "blocker_summary": "same blocker",
            "helper_type_hint": "mutation_loop",
            "candidate_testcase_path": "/testcase/poc",
            "repro_output": "timeout",
            "target_signal": TARGET_REPORT,
        },
        observations=(
            "[SYNTHESIS HELPER #2 - degraded: emit_rejected]\n"
            "Synthesis helper unavailable this step, continue.\n"
            "[END SYNTHESIS HELPER]"
        ),
        step_number=11,
    )
    memory = _Memory([_c_step(step_number=10, gate="header_magic_check"), degraded])

    assert guard.final_answer_check(
        "No source-grounded next action remains.",
        memory,
        _Agent(step_number=12),
    ) is True

    record = _log_records(tmp_path)[0]
    assert record["decision"] == "block"
    assert record["reason"] == "latest_gate_unattempted:header_magic_check"
    assert record["latest_c_step"] == 10
    assert record["latest_gate"] == "header_magic_check"
    assert record["follow_up_after_latest_c"] is False


def test_guard_prefers_a_request_target_signal_over_context_report(tmp_path: Path):
    context_report = TARGET_REPORT.replace("sycc420_to_rgb", "opj_t1_decode_cblk")
    guard = SynthesisFinalizationGuard(
        guard_config={"enabled": True, "min_remaining_steps": 8},
        synthesis_context={"instance_id": "openjpeg.cve-2021-3575", "sanitizer_report": context_report},
        log_dir=str(tmp_path),
    )
    target_obs = (
        "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014\n"
        "READ of size 4 at 0x602000000014 thread T0\n"
        "    #0 0x1234 in sycc420_to_rgb /src/openjpeg/src/bin/common/color.c:379:42\n"
    )
    memory = _Memory(
        [
            _c_step(
                step_number=10,
                gate="packet decode consistency",
                target_signal="AddressSanitizer heap-buffer-overflow in sycc420_to_rgb",
            ),
            _step("cmd", {"command": "secb repro"}, target_obs, 11),
        ]
    )

    assert guard.final_answer_check("A promising next step would be to minimize it.", memory, _Agent(step_number=12))
    assert _log_records(tmp_path)[0]["reason"] == "target_signal_observed"
