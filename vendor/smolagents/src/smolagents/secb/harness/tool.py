"""Agent-A-visible synthesis tools: synthesis helper request and final submission gate."""
from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from smolagents.secb.harness.synthesis import SynthesisOrchestrator
from smolagents.tools import Tool
from smolagents.secb.sanitizer.profile import (
    ProfileContext,
    normalize_observed_crash,
    persist_raw_report,
)

if TYPE_CHECKING:
    from smolagents.secb.review import FinalizationReviewer


def _validate_candidate_testcase_path(value: Any) -> tuple[str, str]:
    path_text = str(value or "").strip()
    if not path_text:
        return "", "ERROR: candidate_testcase_path is required."
    candidate = PurePosixPath(path_text)
    if not candidate.is_absolute():
        return "", "ERROR: candidate_testcase_path must be an absolute /testcase/... path."
    if len(candidate.parts) < 3 or candidate.parts[1] != "testcase":
        return "", "ERROR: candidate_testcase_path must be under /testcase/."
    if ".." in candidate.parts:
        return "", "ERROR: candidate_testcase_path must not contain '..'."

    testcase_root = Path("/testcase")
    if not testcase_root.exists():
        return path_text, ""

    path = Path(path_text)
    try:
        path.resolve(strict=False).relative_to(testcase_root.resolve(strict=False))
    except ValueError:
        return "", "ERROR: candidate_testcase_path must resolve under /testcase/."
    if not path.exists():
        return "", f"ERROR: candidate_testcase_path does not exist: {path_text}"
    if path.is_file():
        try:
            if path.stat().st_size <= 0:
                return "", f"ERROR: candidate_testcase_path is empty: {path_text}"
        except OSError as exc:
            return "", f"ERROR: candidate_testcase_path cannot be inspected: {exc}"
    elif path.is_dir():
        try:
            if not any(path.iterdir()):
                return "", f"ERROR: candidate_testcase_path directory is empty: {path_text}"
        except OSError as exc:
            return "", f"ERROR: candidate_testcase_path cannot be inspected: {exc}"
    return path_text, ""


def _truncate(text: str, limit: int = 8192) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: limit - 40] + "\n...[truncated by synthesis tool]..."


class RunSecbReproOnCurrentTestcaseTool(Tool):
    name = "run_secb_repro_on_current_testcase"
    description = (
        "Run `secb repro` on the current staged /testcase candidate and return structured "
        "stdout/stderr evidence. Use this instead of a raw cmd repro when you may call "
        "the Synthesis Helper from the resulting failure."
    )
    inputs = {
        "candidate_testcase_path": {
            "type": "string",
            "description": "Absolute /testcase path to the candidate being repro-tested.",
        },
        "timeout": {
            "type": "integer",
            "description": "Optional timeout in seconds (default 120, max 600).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, orchestrator: SynthesisOrchestrator, **kwargs):
        super().__init__(**kwargs)
        self._orchestrator = orchestrator

    def forward(self, candidate_testcase_path: str, timeout: int | None = None) -> str:
        candidate_path, candidate_error = _validate_candidate_testcase_path(candidate_testcase_path)
        if candidate_error:
            return candidate_error
        timeout_sec = 120
        if timeout is not None:
            try:
                timeout_sec = min(max(int(timeout), 1), 600)
            except (TypeError, ValueError):
                return "ERROR: timeout must be an integer number of seconds."

        work_dir = str(getattr(self._orchestrator, "base_context", {}).get("work_dir") or "") or None
        profile_completeness = "complete"
        try:
            completed = subprocess.run(
                ["secb", "repro"],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=work_dir,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            status = "completed"
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            status = "timeout"
            exit_code = None
            profile_completeness = "timeout"
        except FileNotFoundError:
            return "ERROR: secb binary not found in PATH."
        except Exception as exc:
            return f"ERROR: secb repro failed before execution: {exc}"

        combined = f"{stdout}\n{stderr}"
        if self._orchestrator.config.evidence_policy_enabled:
            sequence = len(getattr(self._orchestrator, "_a_repro_progress", [])) + 1
            report_path, _ = persist_raw_report(
                log_dir=self._orchestrator.config.log_dir,
                role="a",
                sequence=sequence,
                raw_report=combined,
            )
            profile = normalize_observed_crash(
                combined,
                ProfileContext(
                    work_dir=work_dir or "",
                    report_path=report_path,
                    completeness=profile_completeness,
                ),
            )
            # The crash gate stores the full profile for B's consistency check.
            self._orchestrator.record_a_repro(combined, profile=profile.to_dict())
        else:
            # Preserve the historical A-side behavior for every existing config.
            self._orchestrator.record_a_repro(combined)

        lines = [
            "[SECB REPRO ON CURRENT TESTCASE]",
            f"candidate_testcase_path: {candidate_path}",
            f"status: {status}",
            f"exit_code: {exit_code}",
        ]
        if stdout:
            lines.append("stdout:")
            lines.append(_truncate(stdout, 4096))
        if stderr:
            lines.append("stderr:")
            lines.append(_truncate(stderr, 4096))
        if not stdout and not stderr:
            lines.append("(no stdout/stderr)")
        lines.append("[END SECB REPRO ON CURRENT TESTCASE]")
        return "\n".join(lines)


class RequestSynthesisHelperTool(Tool):
    name = "request_synthesis_helper"
    description = (
        "Delegate a concrete synthesis subproblem after a failed secb repro. The Synthesis "
        "Specialist writes runnable files under Helpers/ and returns a compact manifest. "
        "You remain the only writer to /testcase/."
    )
    inputs = {
        "delegated_problem": {
            "type": "string",
            "description": "One-paragraph diagnosis and bounded subproblem (<=700 chars).",
        },
        "candidate_testcase_path": {
            "type": "string",
            "description": "Absolute path to your failed candidate (typically under /testcase).",
        },
        "repro_output": {
            "type": "string",
            "description": (
                "Raw repro output or your compact summary of the observed failure. "
                "Include what you staged, how you ran it, and what came back."
            ),
        },
        "target_signal": {
            "type": "string",
            "description": "One-line target sanitizer family + top frame from the bug report.",
        },
        "source_hints": {
            "type": "array",
            "description": "Source paths or function names you have already found.",
            "items": {"type": "string"},
            "nullable": True,
        },
        "attempts_summary": {
            "type": "string",
            "description": "Brief recap of approaches you have already tried.",
            "nullable": True,
        },
        "continue_from": {
            "type": "integer",
            "description": "Prior synthesis invocation index to continue from.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        orchestrator: SynthesisOrchestrator,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._orchestrator = orchestrator

    def forward(
        self,
        delegated_problem: str,
        candidate_testcase_path: str,
        repro_output: str,
        target_signal: str,
        source_hints: list[Any] | None = None,
        attempts_summary: str | None = None,
        continue_from: int | None = None,
    ) -> str:
        problem = (delegated_problem or "").strip()
        if not problem:
            return "ERROR: delegated_problem is required."
        problem = problem[:700]
        candidate_path, candidate_error = _validate_candidate_testcase_path(candidate_testcase_path)
        if candidate_error:
            return candidate_error
        repro = _truncate(str(repro_output or ""), 8192)
        target = (target_signal or "").strip()
        if not target:
            return "ERROR: target_signal is required."
        normalized_hints = [str(h) for h in (source_hints or [])]
        return self._orchestrator.invoke(
            delegated_problem=problem,
            candidate_testcase_path=candidate_path,
            repro_output=repro,
            target_signal=target,
            source_hints=normalized_hints,
            attempts_summary=attempts_summary,
            continue_from=continue_from,
        )


class FinalSubmissionTool(Tool):
    name = "final_submission"
    description = (
        "Submit your final artifact for evaluation and request run termination. "
        "Provide artifact_status (what is in /testcase and whether it reproduces the target) "
        "and stop_reason (why you are stopping: success claim or named blocker/exhaustion). "
        "The finalization gate will decide whether to allow termination or return a problem "
        "reframe. If the gate allows, the run ends. If it blocks, the response describes what "
        "the target crash requires that your candidate has not produced — restart your solve "
        "loop from that reframe, do not just execute a named action and re-submit."
    )
    inputs = {
        "artifact_status": {
            "type": "string",
            "description": (
                "What is currently in /testcase and whether you believe it reproduces the target. "
                "Include the filename, what crash/sanitizer output you observed (if any), and "
                "whether secb repro succeeded."
            ),
        },
        "stop_reason": {
            "type": "string",
            "description": (
                "Why you are stopping. Use one of: "
                "'success' (secb repro reproduced the target sanitizer), "
                "'evidence_exhaustion' (no source-grounded next action remains), "
                "or a named blocker (e.g. 'missing_valid_seed', 'format_gate_unsolvable')."
            ),
        },
    }
    output_type = "string"

    def __init__(self, reviewer: "FinalizationReviewer", **kwargs):
        super().__init__(**kwargs)
        self._reviewer = reviewer

    def forward(self, artifact_status: str, stop_reason: str) -> str:
        status = str(artifact_status or "").strip()
        reason = str(stop_reason or "").strip()
        if not status:
            return "ERROR: artifact_status is required."
        if not reason:
            return "ERROR: stop_reason is required."
        memory = getattr(self._reviewer.agent_ref, "memory", None)
        allow, payload = self._reviewer.invoke(
            artifact_status=status,
            stop_reason=reason,
            memory=memory,
        )
        if allow:
            return status
        return payload
