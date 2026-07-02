import json
import subprocess
from pathlib import Path

import smolagents.secb.harness.tools as synthesis_tools
from smolagents.secb.sanitizer.parser import grade_sanitizer_outputs
from smolagents.secb.harness.config import SynthesisAgentBackendConfig
from smolagents.secb.harness.tools import RunSecbReproForHelperTool, SynthesisToolState


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


def test_four_grader_summary_reports_strict_match():
    result = grade_sanitizer_outputs(TARGET, TARGET)
    assert result["highest_passing_grader"] == "strict"
    assert result["grader_passes"] == {
        "loose": True,
        "caller": True,
        "semantic": True,
        "strict": True,
    }
    assert result["strict_failure_reason"] == ""
    assert result["top_frame"]["function"] == "target"


def test_four_grader_summary_reports_no_match():
    result = grade_sanitizer_outputs(TARGET, "clean exit")
    assert result["highest_passing_grader"] == "no_match"
    assert not any(result["grader_passes"].values())
    assert result["strict_failure_reason"]


def test_run_helper_repro_returns_four_grader_result(tmp_path: Path, monkeypatch):
    helpers = tmp_path / "Helpers"
    helpers.mkdir()
    (helpers / "case.bin").write_bytes(b"candidate")
    testcase_root = tmp_path / "testcase"

    monkeypatch.setattr(synthesis_tools, "DEFAULT_TESTCASE_ROOT", testcase_root)
    monkeypatch.setattr(
        synthesis_tools,
        "_load_shape",
        lambda: {"expected_testcase_filename": "poc"},
    )
    monkeypatch.setattr(
        synthesis_tools.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["secb", "repro"],
            returncode=1,
            stdout="",
            stderr=TARGET,
        ),
    )

    state = _state(tmp_path)
    result = json.loads(
        RunSecbReproForHelperTool(state).forward("Helpers/case.bin")
    )
    assert result["validation"] == "strict"
    assert result["grader_passes"]["strict"] is True
    assert state.latest_repro_result["validation"] == "strict"
    assert state.helper_repro_runs_used == 1


def test_run_helper_repro_timeout_does_not_claim_validation(tmp_path: Path, monkeypatch):
    helpers = tmp_path / "Helpers"
    helpers.mkdir()
    (helpers / "case.bin").write_bytes(b"candidate")
    monkeypatch.setattr(synthesis_tools, "DEFAULT_TESTCASE_ROOT", tmp_path / "testcase")
    monkeypatch.setattr(
        synthesis_tools,
        "_load_shape",
        lambda: {"expected_testcase_filename": "poc"},
    )

    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["secb", "repro"], timeout=30)

    monkeypatch.setattr(synthesis_tools.subprocess, "run", timeout_run)
    state = _state(tmp_path)
    output = RunSecbReproForHelperTool(state).forward("Helpers/case.bin")
    assert output.startswith("ERROR: secb repro timed out")
    assert state.latest_repro_result is None
    assert state.helper_repro_runs_used == 0
