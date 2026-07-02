"""B-finalization gate tools. Reuses read-only trajectory tools; adds synthesis-specific tools."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smolagents.tools import Tool

from smolagents.secb.sanitizer.parser import grade_sanitizer_outputs
from smolagents.secb.sanitizer.profile import (
    ObservedCrashProfile,
    ProfileContext,
    ReplayConsistency,
    compare_replays,
    normalize_observed_crash,
    persist_raw_report,
    profile_from_dict,
)
from smolagents.secb.harness.description_judge import assess_description_match
from smolagents.secb.review.evidence import (
    ALLOW_UNVERIFIED,
    INSUFFICIENT_EVIDENCE,
    EvidenceRelation,
    assess_evidence_relation,
    build_target_profile,
    extract_literal_target_profile,
)
from smolagents.secb.trajectory.tools import (
    TrajectoryToolState,
    ReadATrajectoryTool,
    ReadAStepTool,
    FindWorkspacePathsTool,
    ReadWorkspaceFileTool,
    GrepSourceTool,
)
from smolagents.secb.harness.workspace import WorkspaceInspector

HARNESS_SHAPE_PATH = "/workspace/harness_shape.json"


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


ALLOWED_VERDICTS = {"ALLOW_SUCCESS", "ALLOW_EXHAUSTED", ALLOW_UNVERIFIED, "CONTINUE_LOCAL"}
ALLOW_VERDICTS = {"ALLOW_SUCCESS", "ALLOW_EXHAUSTED", ALLOW_UNVERIFIED}


@dataclass
class FinalizationReviewToolState:
    steps: list[Any]
    valid_step_indices: set[int]
    synthesis_log_records: list[dict[str, Any]]
    work_dir: str = ""
    target_signal: str = ""
    default_head_steps: int = 6
    default_tail_steps: int = 12
    default_max_observation_chars: int = 2000
    repro_checked: bool = False
    repro_result: dict[str, Any] | None = None
    description_mode: bool = False
    judge_mode: str = "advisory"
    bug_description: str = ""
    instance_id: str = ""
    judge_model: Any = None
    description_match_result: dict[str, Any] | None = None
    evidence_policy_enabled: bool = False
    evidence_policy_mode: str = "advisory"
    target_profile_mode: str = "literal_only"
    model_version: str = ""
    log_dir: str = "/tmp/synthesis"
    review_index: int = 0
    observed_profile: dict[str, Any] | None = None
    target_profile: dict[str, Any] | None = None
    replay_consistency: dict[str, Any] | None = None
    evidence_relation: dict[str, Any] | None = None
    emitted_verdict: str = ""
    emitted_reasoning: str = ""
    emitted_attached_action: str = ""
    emitted_evidence_steps: list[int] = field(default_factory=list)
    raw_emit_payload: str = ""
    dropped_invalid_evidence: list[dict[str, Any]] = field(default_factory=list)

    @property
    def emit_called(self) -> bool:
        return bool(self.emitted_verdict)


def _make_trajectory_state(state: FinalizationReviewToolState) -> TrajectoryToolState:
    return TrajectoryToolState(
        steps=state.steps,
        valid_step_indices=state.valid_step_indices,
        default_head_steps=state.default_head_steps,
        default_tail_steps=state.default_tail_steps,
        default_max_observation_chars=state.default_max_observation_chars,
    )


class ReadSynthesisLogTool(Tool):
    name = "read_synthesis_log"
    description = (
        "Return a compact projection of Synthesis Helper invocations: delegated problem, "
        "status, helper files, validation tier, and remaining gate."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, state: FinalizationReviewToolState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def forward(self) -> str:
        if not self._state.synthesis_log_records:
            return "No synthesis helper invocations recorded in this run."
        compact = []
        for record in self._state.synthesis_log_records[-6:]:
            inp = record.get("input") or {}
            payload = record.get("emitted_payload") or {}
            compact.append(
                {
                    "invocation_index": record.get("invocation_index"),
                    "delegated_problem": (
                        inp.get("delegated_problem") or inp.get("blocker_summary") or ""
                    )[:500],
                    "degraded": bool(record.get("degraded")),
                    "degraded_reason": record.get("degraded_reason") or "",
                    "partial_helper_files": record.get("partial_helper_files") or [],
                    "helper_files": payload.get("helper_files") or [],
                    "validation": (
                        payload.get("validation")
                        or payload.get("validation_outcome")
                        or "not_run"
                    ),
                    "strict_failure_reason": payload.get("strict_failure_reason") or "",
                    "crossed_gate": payload.get("crossed_gate") or "",
                    "failed_gate": payload.get("failed_gate") or "",
                    "step_budget": record.get("step_budget") or {},
                }
            )
        return json.dumps(compact, indent=2)


class RunSecbReproOnCurrentTestcaseTool(Tool):
    name = "run_secb_repro_on_current_testcase"
    description = (
        "Run `secb repro` against the current /testcase/ state as A left it — no artifact staging. "
        "Returns the highest passing grader, all four grader booleans, strict failure reason, "
        "sanitizer family, top frame, and bounded output evidence. "
        "Use this to obtain ground-truth harness evidence before emitting ALLOW_EXHAUSTED."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, state: FinalizationReviewToolState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def forward(self) -> str:
        shape_path = Path(HARNESS_SHAPE_PATH)
        if not shape_path.exists():
            result = {
                "error": f"{HARNESS_SHAPE_PATH} not found (harness shape probe did not run)",
                "exit_code": None,
                "validation": "environment_failure",
            }
            self._state.repro_checked = True
            self._state.repro_result = result
            return json.dumps(result)

        try:
            json.loads(shape_path.read_text(encoding="utf-8"))
        except Exception as exc:
            result = {
                "error": f"Failed to parse {HARNESS_SHAPE_PATH}: {exc}",
                "exit_code": None,
                "validation": "environment_failure",
            }
            self._state.repro_checked = True
            self._state.repro_result = result
            return json.dumps(result)

        # The crash gate needs a reproducibility partner for the consistency check. Prefer A's
        # recorded profile (a genuine cross-state A/B comparison). A often runs `secb repro`
        # via the generic cmd tool, which records no profile; when A's profile is absent, fall
        # back to a self-confirmation repro on the finalized candidate. That fallback degrades
        # consistency from "A and B agree" to "the crash reproduces on the final candidate":
        # identity fields are trivially equal across two back-to-back B runs (see
        # compare_replays), so only the crash fingerprint is actually being re-verified.
        evidence_active = self._state.description_mode and self._state.evidence_policy_enabled
        a_profile: ObservedCrashProfile | None = None
        if evidence_active:
            a_profiles_path = Path(self._state.log_dir) / "a_repro_profiles.jsonl"
            if a_profiles_path.exists():
                try:
                    for line in a_profiles_path.read_text(encoding="utf-8").splitlines():
                        record = json.loads(line)
                        candidate = profile_from_dict(record.get("observed_profile"))
                        if candidate is not None:
                            a_profile = candidate
                except Exception:
                    a_profile = None
        need_confirmation = evidence_active and a_profile is None
        confirm_combined: str | None = None

        # Snapshot /testcase so repro side effects do not contaminate the solver artifact.
        testcase_dir = Path("/testcase")
        snapshot_dir: Path | None = None
        restore_warning = ""
        if testcase_dir.exists():
            try:
                tmp = tempfile.mkdtemp(prefix="b_repro_snapshot_")
                shutil.copytree(str(testcase_dir), str(Path(tmp) / "testcase"))
                snapshot_dir = Path(tmp)
            except Exception as exc:
                restore_warning = f"WARNING: snapshot failed, /testcase not restored: {exc}"

        work_dir = self._state.work_dir or None
        completed = None
        repro_error = ""
        try:
            completed = subprocess.run(
                ["secb", "repro"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=work_dir,
            )
            if need_confirmation:
                # Second independent repro inside the same snapshot guard, so the single
                # restore below still protects A's continuation candidate.
                try:
                    confirm = subprocess.run(
                        ["secb", "repro"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=work_dir,
                    )
                    confirm_combined = (confirm.stdout or "") + "\n" + (confirm.stderr or "")
                except Exception:
                    confirm_combined = None
        except subprocess.TimeoutExpired:
            repro_error = "secb repro timed out (60s)"
        except FileNotFoundError:
            repro_error = "secb binary not found in PATH"
        except Exception as exc:
            repro_error = str(exc)
        finally:
            if snapshot_dir is not None:
                try:
                    # /testcase is a Docker bind-mount: clear children without removing
                    # the mountpoint itself, then copy snapshot children back in.
                    for child in testcase_dir.iterdir():
                        if child.is_dir():
                            shutil.rmtree(str(child))
                        else:
                            child.unlink(missing_ok=True)
                    for child in (snapshot_dir / "testcase").iterdir():
                        dst = testcase_dir / child.name
                        if child.is_dir():
                            shutil.copytree(str(child), str(dst))
                        else:
                            shutil.copy2(str(child), str(dst))
                except Exception as exc:
                    restore_warning = f"WARNING: failed to restore /testcase: {exc}"
                finally:
                    shutil.rmtree(str(snapshot_dir), ignore_errors=True)

        if repro_error:
            result = {
                "error": repro_error + (f" {restore_warning}" if restore_warning else ""),
                "exit_code": None,
                "validation": "environment_failure",
            }
            self._state.repro_checked = True
            self._state.repro_result = result
            return json.dumps(result)

        stdout = (completed.stdout or "")[:1200]
        stderr = (completed.stderr or "")[:1200]
        combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
        grading = grade_sanitizer_outputs(self._state.target_signal, combined)
        result = {
            "exit_code": completed.returncode,
            "validation": grading["highest_passing_grader"],
            "grader_passes": grading["grader_passes"],
            "strict_failure_reason": grading["strict_failure_reason"],
            "sanitizer_family": grading["sanitizer_family"],
            "crash_type": grading["crash_type"],
            "top_frame": grading["top_frame"],
            "stdout_truncated": stdout,
            "stderr_truncated": stderr,
        }
        if self._state.description_mode and self._state.evidence_policy_enabled:
            advisory_shadow = self._state.evidence_policy_mode == "advisory"
            if advisory_shadow:
                # Resolve the actual legacy tool result before any shadow-only work.
                match = assess_description_match(
                    self._state.bug_description,
                    combined,
                    model=self._state.judge_model,
                    instance_id=self._state.instance_id,
                )
                result["description_match_result"] = match
                self._state.description_match_result = match
            report_path = ""
            report_hash = ""
            try:
                report_path, report_hash = persist_raw_report(
                    log_dir=self._state.log_dir,
                    role="b",
                    sequence=self._state.review_index + 1,
                    raw_report=combined,
                )
                profile = normalize_observed_crash(
                    combined,
                    ProfileContext(
                        work_dir=self._state.work_dir,
                        report_path=report_path,
                        completeness="complete",
                        restore_warning=restore_warning,
                    ),
                )
                self._state.observed_profile = profile.to_dict()
                if a_profile is None and confirm_combined is not None:
                    # No recorded A profile: use the self-confirmation repro as the
                    # reproducibility partner on the finalized candidate.
                    a_profile = normalize_observed_crash(
                        confirm_combined,
                        ProfileContext(
                            work_dir=self._state.work_dir,
                            completeness="complete",
                            restore_warning=restore_warning,
                        ),
                    )
                replay = compare_replays(a_profile, profile)
                cache_path = Path(self._state.log_dir) / "finalization_evidence" / "target_profile.json"
                target = build_target_profile(
                    self._state.bug_description,
                    mode="literal_only" if advisory_shadow else self._state.target_profile_mode,
                    model=None if advisory_shadow else self._state.judge_model,
                    model_version=self._state.model_version,
                    cache_path=cache_path,
                )
                relation = assess_evidence_relation(
                    self._state.bug_description,
                    target,
                    profile,
                    replay,
                    model=None if advisory_shadow else self._state.judge_model,
                )
            except Exception as exc:
                # Evidence collection degrades to an abstention. Advisory keeps its already
                # resolved legacy result; enforce follows the bounded challenge policy.
                profile = ObservedCrashProfile(report_completeness="environment_failure")
                target = extract_literal_target_profile(self._state.bug_description)
                replay = ReplayConsistency(False, "error", ("evidence_collection_failure",))
                relation = EvidenceRelation(
                    INSUFFICIENT_EVIDENCE,
                    reasoning="evidence collection failed",
                    judge_error=str(exc)[:1000],
                )
            self._state.observed_profile = profile.to_dict()
            self._state.target_profile = target.to_dict()
            self._state.replay_consistency = replay.to_dict()
            self._state.evidence_relation = relation.to_dict()
            if not advisory_shadow:
                result.update(
                    {
                        "report_hash": report_hash,
                        "report_path": report_path,
                        "observed_profile": profile.to_dict(),
                        "target_profile": target.to_dict(),
                        "replay_consistency": replay.to_dict(),
                        "evidence_relation": relation.to_dict(),
                    }
                )
        elif self._state.description_mode:
            match = assess_description_match(
                self._state.bug_description,
                combined,
                model=self._state.judge_model,
                instance_id=self._state.instance_id,
            )
            result["description_match_result"] = match
            self._state.description_match_result = match
        if restore_warning:
            result["restore_warning"] = restore_warning
        self._state.repro_checked = True
        self._state.repro_result = result
        return json.dumps(result)


_REPRO_NOT_CALLED_ACTION = (
    "Run `secb repro` against your current /testcase candidate and inspect the output. "
    "If no sanitizer fires, return to the source path driving the target crash and identify "
    "what precondition your candidate has not satisfied before submitting again."
)
_REPRO_ENV_FAILURE_ACTION = (
    "The harness check could not run cleanly (environment error or restore failure). "
    "Run `secb repro` directly, confirm no sanitizer fires, then re-examine the source "
    "path driving the crash before submitting again."
)
_REPRO_SANITIZER_FIRED_ACTION = (
    "The harness shows a sanitizer firing on your current candidate — stopping is not earned. "
    "Inspect the sanitizer output, compare it against the target signal, and continue narrowing "
    "toward the exact crash family and top frame."
)


def _allow_exhausted_block_reason(state: "FinalizationReviewToolState") -> str | None:
    """Return a CONTINUE_LOCAL action string if ALLOW_EXHAUSTED must be blocked, else None."""
    if not state.repro_checked:
        return _REPRO_NOT_CALLED_ACTION
    r = state.repro_result
    if r is None or r.get("validation") == "environment_failure":
        return _REPRO_ENV_FAILURE_ACTION
    if r.get("restore_warning"):
        return _REPRO_ENV_FAILURE_ACTION
    if r.get("sanitizer_family"):
        return _REPRO_SANITIZER_FIRED_ACTION
    return None


def _build_mismatch_redirect_action(match: dict) -> str:
    """Build a specific CONTINUE_LOCAL action text from the description-match result.

    Uses the deterministic pre-filter fields (determ_verdict, described_class,
    described_functions, crash_type, observed_top_frame) already present in the
    description_match_result dict to produce a crash-specific redirect message.
    Falls back to a generic message when fields are absent.
    """
    determ = match.get("determ_verdict") or ""
    obs_type = match.get("crash_type") or match.get("observed_family") or "(unknown)"
    obs_frame = (match.get("observed_top_frame") or {}).get("function") or "(unknown)"
    desc_class = match.get("described_class") or ""
    desc_funcs: list = match.get("described_functions") or []
    desc_target = ", ".join(desc_funcs) if desc_funcs else ""

    if determ == "type_mismatch" and desc_class:
        tail = f" in {desc_target}" if desc_target else ""
        return (
            f"Your candidate produced a {obs_type} crash (top frame: {obs_frame}), "
            f"but the described vulnerability is a {desc_class}{tail}. "
            "These are different bugs — the crash you found is not the target vulnerability. "
            f"Identify the source path leading to a {desc_class}{tail} "
            "and re-direct your PoC toward that path."
        )
    if determ == "frame_mismatch" and desc_target:
        return (
            f"Your candidate crashes in {obs_frame} (type: {obs_type}), "
            f"but the described vulnerability targets {desc_target}. "
            "The described target function was not found in the observed call stack. "
            f"Locate the code path leading to {desc_target} and ensure your PoC "
            "exercises that path before claiming success."
        )
    return (
        "The description-consistency judge did not confirm this crash matches "
        "the described vulnerability. Inspect the mismatch between your observed "
        "crash and the bug description, then continue from that discrepancy."
    )


def _allow_success_block_reason(state: "FinalizationReviewToolState") -> str | None:
    if not state.repro_checked:
        return _REPRO_NOT_CALLED_ACTION
    r = state.repro_result or {}
    if r.get("validation") == "environment_failure":
        return _REPRO_ENV_FAILURE_ACTION
    if state.description_mode:
        # No sanitizer-report oracle exists in description mode. judge_mode="gate" hard-gates
        # on the non-oracle description-consistency judge; judge_mode="advisory" trusts the
        # B sub-agent's own emitted verdict instead (the loose-grader check below is
        # unreachable here since grader_passes is always no-oracle/False post boundary-fix).
        if state.judge_mode == "gate":
            match = state.description_match_result or {}
            if not match.get("matched"):
                return _build_mismatch_redirect_action(match)
        return None
    if not (r.get("grader_passes") or {}).get("loose"):
        return (
            "The current harness result does not pass even the loose target grader. "
            "Continue from the observed mismatch before claiming success."
        )
    return None


class EmitFinalizationVerdictTool(Tool):
    name = "emit_finalization_verdict"
    description = (
        "Emit the finalization verdict. Terminal call — use exactly once. "
        "Verdict must be one of: ALLOW_SUCCESS, ALLOW_EXHAUSTED, CONTINUE_LOCAL. "
        "ALLOW verdicts terminate the run. CONTINUE_LOCAL requires a non-empty attached_action."
    )
    inputs = {
        "verdict": {
            "type": "string",
            "description": "One of: ALLOW_SUCCESS, ALLOW_EXHAUSTED, CONTINUE_LOCAL.",
        },
        "reasoning": {
            "type": "string",
            "description": "Short justification for the verdict (1-3 sentences).",
        },
        "attached_action": {
            "type": "string",
            "description": (
                "Required for CONTINUE_LOCAL. "
                "A problem reframe for the PoC Solver, not a task checklist. State what the "
                "target crash requires that its candidate has not produced, and name the "
                "source-grounded investigation frame it should restart from. Do not list steps. "
                "Do not reference verdict labels."
            ),
            "nullable": True,
        },
        "evidence_steps": {
            "type": "array",
            "description": "PoC Solver step numbers supporting this verdict.",
            "items": {"type": "integer"},
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, state: FinalizationReviewToolState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def forward(
        self,
        verdict: str,
        reasoning: str,
        attached_action: str | None = None,
        evidence_steps: list[Any] | None = None,
    ) -> str:
        normalized = str(verdict or "").strip().upper()
        if normalized not in ALLOWED_VERDICTS:
            normalized = "CONTINUE_LOCAL"

        # ALLOW_UNVERIFIED is policy-owned. Legacy and advisory B prompts cannot emit it
        # directly; the reviewer may apply it after the evidence relation is resolved.
        if normalized == ALLOW_UNVERIFIED and (
            not self._state.evidence_policy_enabled
            or self._state.evidence_policy_mode != "enforce"
        ):
            normalized = "CONTINUE_LOCAL"

        if normalized == "ALLOW_SUCCESS":
            block_reason = _allow_success_block_reason(self._state)
            if block_reason is not None:
                normalized = "CONTINUE_LOCAL"
                attached_action = block_reason

        if normalized == "ALLOW_EXHAUSTED":
            block_reason = _allow_exhausted_block_reason(self._state)
            if block_reason is not None:
                normalized = "CONTINUE_LOCAL"
                attached_action = block_reason

        is_block = normalized not in ALLOW_VERDICTS
        action = str(attached_action or "").strip()
        if is_block and not action:
            action = (
                "Return to the source path driving the target crash and identify what "
                "precondition the candidate has not satisfied. Restart the solve loop from "
                "that source-level requirement — do not re-submit without exploring a new approach."
            )

        kept = []
        dropped = []
        for value in evidence_steps or []:
            step_number = _coerce_int(value)
            if step_number is None or step_number not in self._state.valid_step_indices:
                dropped.append({"entry": value, "reason": "invalid_evidence_step"})
                continue
            kept.append(step_number)

        payload = {
            "verdict": normalized,
            "reasoning": str(reasoning or ""),
            "attached_action": action,
            "evidence_steps": kept,
        }
        self._state.emitted_verdict = normalized
        self._state.emitted_reasoning = str(reasoning or "")
        self._state.emitted_attached_action = action
        self._state.emitted_evidence_steps = kept
        self._state.raw_emit_payload = json.dumps(payload)
        self._state.dropped_invalid_evidence = dropped

        suffix = f"; dropped {len(dropped)} invalid evidence step(s)." if dropped else "."
        return f"OK: verdict={normalized} emitted{suffix}"


def build_review_tools(
    *,
    tools_allow: tuple[str, ...],
    state: FinalizationReviewToolState,
    inspector: WorkspaceInspector,
) -> list[Tool]:
    trajectory_state = _make_trajectory_state(state)
    factories = {
        "read_a_trajectory": lambda: ReadATrajectoryTool(trajectory_state),
        "read_a_step": lambda: ReadAStepTool(trajectory_state),
        "find_workspace_paths": lambda: FindWorkspacePathsTool(inspector),
        "read_workspace_file": lambda: ReadWorkspaceFileTool(inspector),
        "grep_source": lambda: GrepSourceTool(inspector),
        "read_synthesis_log": lambda: ReadSynthesisLogTool(state),
        "run_secb_repro_on_current_testcase": lambda: RunSecbReproOnCurrentTestcaseTool(state),
        "emit_finalization_verdict": lambda: EmitFinalizationVerdictTool(state),
    }
    return [factories[name]() for name in tools_allow]
