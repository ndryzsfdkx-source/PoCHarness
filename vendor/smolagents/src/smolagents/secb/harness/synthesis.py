"""Stateful Synthesis Helper orchestrator."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from smolagents.secb.cost_budget import CostBudgetLedger
from smolagents.secb.harness.agent import SynthesisAgentBackend, SynthesisBackendResult
from smolagents.secb.harness.config import SynthesisConfig
from smolagents.secb.harness.crash_compare import (
    PROGRESS_ACCEPTED_NO_SIGNAL,
    PROGRESS_ARTIFACT_MISSING,
    PROGRESS_ENVIRONMENT_FAILURE,
    PROGRESS_HARNESS_REJECT,
    PROGRESS_SANITIZER_SEEN,
    PROGRESS_TARGET_FAMILY_SEEN,
    PROGRESS_TARGET_FRAME_SEEN,
    classify_helper_repro_progress,
)


DEGRADED_MANIFEST_BODY = "Synthesis helper unavailable this step, continue."

# Progress ladder for A-side repro outcomes (low → high). Used to decide whether the
# first failing gate moved after a C emission (downstream_passed).
_PROGRESS_RANK = {
    PROGRESS_ARTIFACT_MISSING: 0,
    PROGRESS_ENVIRONMENT_FAILURE: 1,
    PROGRESS_HARNESS_REJECT: 2,
    PROGRESS_ACCEPTED_NO_SIGNAL: 3,
    PROGRESS_SANITIZER_SEEN: 4,
    PROGRESS_TARGET_FAMILY_SEEN: 5,
    PROGRESS_TARGET_FRAME_SEEN: 6,
}
# A repro that reached at least a real sanitizer signal counts as downstream gate movement.
_SANITIZER_RANK = _PROGRESS_RANK[PROGRESS_SANITIZER_SEEN]


def _compact_prior_result(
    invocation_index: int,
    payload: dict[str, Any] | None,
    *,
    degraded: bool,
    partial_helper_files: list[str] | None = None,
    degraded_reason: str | None = None,
) -> dict[str, Any]:
    """Compact, prompt-safe view of a prior synthesis invocation."""
    payload = payload or {}
    return {
        "invocation_index": invocation_index,
        "degraded": degraded,
        "degraded_reason": degraded_reason or "",
        "helper_files": list(
            payload.get("helper_files") or partial_helper_files or []
        )[:8],
        "how_to_run": payload.get("how_to_run") or "",
        "validation": payload.get("validation") or "not_run",
        "crossed_gate": payload.get("crossed_gate") or "",
        "failed_gate": payload.get("failed_gate") or "",
        "notes": (payload.get("notes") or "")[:300],
    }


def _render_observation(invocation_index: int, payload: dict[str, Any]) -> str:
    files = payload.get("helper_files") or []
    how_to_run = payload.get("how_to_run") or "(no command provided)"
    validation = payload.get("validation") or "not_run"
    crossed_gate = payload.get("crossed_gate") or ""
    failed_gate = payload.get("failed_gate") or ""
    harness_evidence = payload.get("harness_evidence") or ""
    notes = payload.get("notes") or ""

    lines = [
        f"[SYNTHESIS SPECIALIST #{invocation_index}]",
        "Files written:",
    ]
    if files:
        for rel in files:
            lines.append(f"  - {rel}")
    else:
        lines.append("  (none)")
    lines.extend(
        [
            "How to run:",
            f"  {how_to_run}",
            f"Validation: {validation}",
        ]
    )
    if crossed_gate:
        lines.append(f"Crossed gate: {crossed_gate}")
    if failed_gate:
        lines.append(f"Failed gate: {failed_gate}")
    if harness_evidence:
        lines.append(f"Harness evidence: {harness_evidence}")
    if notes:
        lines.append(f"Notes: {notes}")
    lines.append("[END SYNTHESIS SPECIALIST]")
    return "\n".join(lines)


def _render_degraded(invocation_index: int, result: SynthesisBackendResult) -> str:
    reason = result.degraded_reason or "unknown"
    lines = [
        f"[SYNTHESIS SPECIALIST #{invocation_index} — degraded: {reason}]",
        DEGRADED_MANIFEST_BODY,
    ]
    if result.partial_helper_files:
        lines.append("Partial helper files written:")
        for rel in result.partial_helper_files:
            lines.append(f"  - {rel}")
    detail = result.emit_error or result.error_message
    if detail:
        lines.append(f"Reason: {detail}")
    lines.append("[END SYNTHESIS SPECIALIST]")
    return "\n".join(lines)


class SynthesisOrchestrator:
    """Owns per-instance state across all request_synthesis_helper invocations."""

    def __init__(
        self,
        synthesis_config: SynthesisConfig | dict[str, Any] | None = None,
        synthesis_context: dict[str, Any] | None = None,
        *,
        default_model_id: str | None = None,
        model_transport_kwargs: dict | None = None,
        cost_budget: CostBudgetLedger | None = None,
        description_mode: bool = False,
    ):
        self.config = (
            synthesis_config
            if isinstance(synthesis_config, SynthesisConfig)
            else SynthesisConfig.from_config(
                synthesis_config, default_model_id=default_model_id, description_mode=description_mode
            )
        )
        if model_transport_kwargs:
            self.config.model_transport_kwargs = dict(model_transport_kwargs)
        self.base_context = dict(synthesis_context or {})
        self.cost_budget = cost_budget
        self.invocations_used = 0
        self._invocation_lock = threading.Lock()
        self.agent_ref = None
        self.last_repro_attempt: dict[str, Any] | None = None
        self._prior_results: dict[int, dict[str, Any]] = {}
        # Queue completed emissions and degraded partial outputs for run-end
        # adoption and gate-movement resolution.
        self._pending_observations: list[dict[str, Any]] = []
        self._a_repro_progress: list[dict[str, Any]] = []
        self._last_target_signal: str = ""
        self._observability_written = False
        log_dir = Path(self.config.log_dir)
        self._log_path = log_dir / "synthesis_log.jsonl"
        self._observability_path = log_dir / "synthesis_observability.jsonl"

    def invoke(
        self,
        *,
        delegated_problem: str,
        candidate_testcase_path: str,
        repro_output: str,
        target_signal: str,
        source_hints: list[str] | None = None,
        attempts_summary: str | None = None,
        continue_from: int | None = None,
    ) -> str:
        with self._invocation_lock:
            self.invocations_used += 1
            invocation_index = self.invocations_used

        a_step = getattr(self.agent_ref, "step_number", None)
        if target_signal:
            self._last_target_signal = str(target_signal)
        prior_result = self._prior_results.get(continue_from) if continue_from is not None else None
        bundle: dict[str, Any] = {
            "delegated_problem": str(delegated_problem or ""),
            "candidate_testcase_path": str(candidate_testcase_path or ""),
            "repro_output": str(repro_output or ""),
            "target_signal": str(target_signal or ""),
            # description_mode: never populate from the dataset oracle, and do not fall back
            # to A's self-supplied target_signal either -- C's validation tool must not grade
            # against anything oracle-shaped in this mode (poc-desc_redesign_spec.md SS4/SS5).
            "oracle_output": (
                ""
                if self.config.description_mode
                else str(self.base_context.get("sanitizer_report") or target_signal or "")
            ),
            "source_hints": list(source_hints or []),
            "attempts_summary": attempts_summary,
            "continue_from": continue_from,
            "continuation_mode": prior_result is not None,
            "prior_synthesis": [prior_result] if prior_result is not None else [],
            "invocation_index": invocation_index,
            "instance_id": self.base_context.get("instance_id", ""),
            "work_dir": self.base_context.get("work_dir", ""),
        }

        try:
            backend = SynthesisAgentBackend(
                config=self.config,
                static_context=self.base_context,
                invocation_index=invocation_index,
                cost_budget=self.cost_budget,
            )
            result = backend.run(bundle)
        except Exception as exc:
            result = SynthesisBackendResult(
                parse_status="failed",
                emitted_payload=None,
                c_steps_used=0,
                c_tool_calls=[],
                degraded=True,
                prompt="",
                error_message=f"Backend error: {exc}",
                raw_response="",
                degraded_reason="backend_exception",
                emit_error="",
                partial_helper_files=[],
            )

        self._write_log(
            invocation_index=invocation_index,
            a_step=a_step,
            bundle=bundle,
            result=result,
        )

        prior_record = _compact_prior_result(
            invocation_index,
            result.emitted_payload,
            degraded=result.degraded,
            partial_helper_files=result.partial_helper_files,
            degraded_reason=result.degraded_reason,
        )
        self._prior_results[invocation_index] = prior_record

        observable_files = (
            list((result.emitted_payload or {}).get("helper_files") or [])
            or list(result.partial_helper_files)
        )
        candidate_path = bundle.get("candidate_testcase_path") or ""
        if observable_files:
            self._pending_observations.append(
                {
                    "invocation_index": invocation_index,
                    "a_step_at_result": a_step,
                    "result_status": "degraded_partial" if result.degraded else "emitted",
                    "helper_basenames": [Path(str(f)).name for f in observable_files],
                    "candidate_testcase_path": candidate_path,
                    "testcase_sha_at_result": self._hash_path(candidate_path),
                }
            )
        if result.degraded or result.emitted_payload is None:
            return _render_degraded(invocation_index, result)
        return _render_observation(invocation_index, result.emitted_payload)

    def _write_log(
        self,
        *,
        invocation_index: int,
        a_step: int | None,
        bundle: dict[str, Any],
        result: SynthesisBackendResult,
    ) -> None:
        try:
            log_dir = self._log_path.parent
            log_dir.mkdir(parents=True, exist_ok=True)
            repro_output = bundle.get("repro_output") or ""
            repro_sha = hashlib.sha256(repro_output.encode("utf-8", errors="replace")).hexdigest()
            input_record = {
                "delegated_problem": bundle.get("delegated_problem"),
                "candidate_testcase_path": bundle.get("candidate_testcase_path"),
                "repro_output_sha256": repro_sha,
                "repro_output_head": repro_output[:1024],
                "target_signal": bundle.get("target_signal"),
                "source_hints": bundle.get("source_hints"),
                "attempts_summary": bundle.get("attempts_summary"),
                "continue_from": bundle.get("continue_from"),
                "continuation_mode": bundle.get("continuation_mode"),
            }
            prompt_hash = (
                hashlib.sha256(result.prompt.encode("utf-8", errors="replace")).hexdigest()
                if result.prompt
                else None
            )
            record = {
                "schema_version": "v2",
                "invocation_index": invocation_index,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "a_step": a_step,
                "input": input_record,
                "c_steps_used": result.c_steps_used,
                "step_budget": result.step_budget,
                "c_input_tokens": result.c_input_tokens,
                "c_output_tokens": result.c_output_tokens,
                "c_termination_reason": result.termination_reason,
                "c_tool_calls": result.c_tool_calls,
                "degraded": result.degraded,
                "degraded_reason": result.degraded_reason,
                "parse_status": result.parse_status,
                "error_message": result.error_message,
                "emit_error": result.emit_error,
                "partial_helper_files": result.partial_helper_files,
                "emitted_payload": result.emitted_payload,
                "prompt_hash": prompt_hash,
                "a_followed": None,
                "a_adopted_into_testcase": None,
                "downstream_passed": None,
            }
            if result.c_cost is not None:
                record["c_cost"] = result.c_cost
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            import sys

            print(f"WARNING: Synthesis log write failed: {exc}", file=sys.stderr)

    @staticmethod
    def _hash_path(path_text: str) -> str:
        """SHA-256 of a file's bytes, or '' if not a readable regular file."""
        try:
            path = Path(str(path_text or ""))
            if path.is_file():
                return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            pass
        return ""

    @staticmethod
    def _step_argtext(step: Any) -> str:
        """Serialize a memory step's tool-call names + arguments (what A actively did)."""
        parts: list[str] = []
        for call in getattr(step, "tool_calls", None) or []:
            name = getattr(call, "name", "")
            if name:
                parts.append(str(name))
            args = getattr(call, "arguments", None)
            if isinstance(args, str):
                parts.append(args)
            else:
                try:
                    parts.append(json.dumps(args))
                except TypeError:
                    parts.append(str(args))
        return "\n".join(parts)

    def record_a_repro(self, observed_output: str, *, profile: dict[str, Any] | None = None) -> None:
        """Record an A-side `secb repro` outcome so downstream gate movement is observable.

        Called by RunSecbReproOnCurrentTestcaseTool. Classifies the observed output against
        the most recent target signal and stores its progress rank. Never raises.
        """
        try:
            progress = classify_helper_repro_progress(
                self._last_target_signal or "",
                str(observed_output or ""),
                artifact_created=True,
            )
            entry = {
                "a_step": getattr(self.agent_ref, "step_number", None),
                "progress_gate": progress.get("progress_gate"),
                "rank": _PROGRESS_RANK.get(progress.get("progress_gate"), 0),
                "sanitizer_seen": bool(progress.get("sanitizer_seen")),
            }
            if profile is not None:
                entry["observed_profile"] = profile
            self._a_repro_progress.append(entry)
            self.last_repro_attempt = entry
            if profile is not None:
                evidence_path = Path(self.config.log_dir) / "a_repro_profiles.jsonl"
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                with evidence_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, sort_keys=True) + "\n")
        except Exception as exc:
            import sys

            print(f"WARNING: record_a_repro failed: {exc}", file=sys.stderr)

    def finalize_observability(self) -> None:
        """Resolve queued observations into the sidecar. Call once at run end. Never raises."""
        if self._observability_written:
            return
        self._observability_written = True
        if not self._pending_observations:
            return
        try:
            memory = getattr(self.agent_ref, "memory", None)
            steps: list[tuple[int, str]] = []
            for step in getattr(memory, "steps", None) or []:
                step_number = getattr(step, "step_number", None)
                if step_number is None:
                    continue
                steps.append((int(step_number), self._step_argtext(step)))

            records: list[dict[str, Any]] = []
            for obs in self._pending_observations:
                result_step = obs.get("a_step_at_result")
                after_text = "\n".join(
                    text for (sn, text) in steps if result_step is None or sn > result_step
                )
                basenames = [bn for bn in (obs.get("helper_basenames") or []) if bn]
                a_followed = any(bn in after_text for bn in basenames)
                current_sha = self._hash_path(obs.get("candidate_testcase_path"))
                a_adopted = bool(
                    current_sha and current_sha != obs.get("testcase_sha_at_result")
                )
                downstream_passed = any(
                    (
                        result_step is None
                        or e.get("a_step") is None
                        or e["a_step"] > result_step
                    )
                    and int(e.get("rank", 0)) >= _SANITIZER_RANK
                    for e in self._a_repro_progress
                )
                records.append(
                    {
                        "schema_version": "v2",
                        "invocation_index": obs.get("invocation_index"),
                        "result_status": obs.get("result_status"),
                        "a_step_at_result": result_step,
                        "a_followed": a_followed,
                        "a_adopted_into_testcase": a_adopted,
                        "a_adopted_is_proxy": True,
                        "downstream_passed": downstream_passed,
                    }
                )

            self._observability_path.parent.mkdir(parents=True, exist_ok=True)
            with self._observability_path.open("a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
        except Exception as exc:
            import sys

            print(f"WARNING: observability sidecar write failed: {exc}", file=sys.stderr)


def create_synthesis_orchestrator(
    synthesis_config: dict[str, Any] | SynthesisConfig | None = None,
    synthesis_context: dict[str, Any] | None = None,
    *,
    default_model_id: str | None = None,
    model_transport_kwargs: dict | None = None,
    cost_budget: CostBudgetLedger | None = None,
    description_mode: bool = False,
) -> SynthesisOrchestrator:
    return SynthesisOrchestrator(
        synthesis_config=synthesis_config,
        synthesis_context=synthesis_context,
        default_model_id=default_model_id,
        model_transport_kwargs=model_transport_kwargs,
        cost_budget=cost_budget,
        description_mode=description_mode,
    )
