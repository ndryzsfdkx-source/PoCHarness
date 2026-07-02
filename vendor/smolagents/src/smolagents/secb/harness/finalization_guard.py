"""Rule-based finalization guard for synthesis-enabled SEC-bench PoC runs."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...memory import ActionStep
from .config import SynthesisFinalizationGuardConfig
from .crash_compare import STATUS_EXACT_MATCH, strict_crash_compare


GUARD_LOG_NAME = "synthesis_finalization_guard.jsonl"
IGNORED_FOLLOWUP_TOOLS = {"artifact_guard", "final_answer", "consult_trajectory_advisor"}

_NEXT_ACTION_RE = re.compile(
    r"\b(?:next\s+(?:step|action)|promising\s+next|remaining\s+(?:direction|blocker|work)|"
    r"(?:could|should|would\s+need\s+to|need\s+to))\b"
    r"(?P<body>.{0,240})"
    r"\b(?:inspect|refine|call|run|mutate|try|sweep|adjust|generate|construct|rebuild|instrument)\b",
    re.IGNORECASE | re.DOTALL,
)
_NO_NEXT_ACTION_RE = re.compile(
    r"\b(?:no\s+concrete\s+next\s+action|no\s+source-grounded\s+next|no\s+remaining\s+action)\b",
    re.IGNORECASE,
)
_GATE_PATTERNS = (
    re.compile(r"^Next gate targeted:\s*(?P<gate>.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Failed gate:\s*(?P<gate>.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Current blocking gate:\s*(?P<gate>.+)$", re.IGNORECASE | re.MULTILINE),
)
_PLAIN_TARGET_FRAME_RE = re.compile(r"\bin\s+([A-Za-z_][A-Za-z0-9_:\.<>~\-]*)\b")


@dataclass
class GuardDecision:
    decision: str
    reason: str
    c_used: bool
    target_reached: bool
    remaining_steps: int
    latest_c_step: int | None
    latest_gate: str
    follow_up_after_latest_c: bool
    concrete_next_action_in_final: bool


def _iter_action_steps(memory) -> list[ActionStep]:
    steps: list[ActionStep] = []
    for step in getattr(memory, "steps", []) or []:
        if isinstance(step, ActionStep):
            steps.append(step)
    return steps


def _tool_args(tool_call) -> dict[str, Any]:
    args = getattr(tool_call, "arguments", {}) or {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_name(tool_call) -> str:
    return str(getattr(tool_call, "name", "") or "")


def _has_tool(step: ActionStep, tool_name: str) -> bool:
    return any(_tool_name(tc) == tool_name for tc in getattr(step, "tool_calls", []) or [])


def _request_target_signal(step: ActionStep) -> str:
    for tc in getattr(step, "tool_calls", []) or []:
        if _tool_name(tc) == "request_synthesis_helper":
            return str(_tool_args(tc).get("target_signal") or "")
    return ""


def _latest_request_target_signal(c_steps: list[ActionStep]) -> str:
    for step in reversed(c_steps):
        signal = _request_target_signal(step).strip()
        if signal:
            return signal
    return ""


def _is_degraded_synthesis_step(step: ActionStep) -> bool:
    observations = str(getattr(step, "observations", "") or "").lower()
    return (
        _has_tool(step, "request_synthesis_helper")
        and ("degraded:" in observations or "synthesis helper unavailable" in observations)
    )


def _find_latest_gate(step: ActionStep) -> str:
    observations = str(getattr(step, "observations", "") or "")
    for pattern in _GATE_PATTERNS:
        match = pattern.search(observations)
        if match:
            gate = match.group("gate").strip()
            if gate and gate.lower() not in {"n/a", "none", "(unknown)", "unknown"}:
                return gate[:300]
    return ""


def _target_reached(steps: list[ActionStep], target_signal: str) -> bool:
    if not target_signal:
        return False
    plain_frame_match = _PLAIN_TARGET_FRAME_RE.search(str(target_signal))
    plain_frame = plain_frame_match.group(1) if plain_frame_match else ""
    for step in steps:
        observations = str(getattr(step, "observations", "") or "")
        if not observations:
            continue
        try:
            comparison = strict_crash_compare(target_signal, observations)
            if comparison.get("status") == STATUS_EXACT_MATCH:
                return True
            target = comparison.get("target") or {}
            observed = comparison.get("observed") or {}
            if (
                plain_frame
                and target.get("family")
                and observed.get("family") == target.get("family")
                and observed.get("top_function") == plain_frame
            ):
                return True
        except Exception:
            continue
    return False


def _latest_c_gate(c_steps: list[ActionStep]) -> tuple[ActionStep | None, str]:
    for step in reversed(c_steps):
        gate = _find_latest_gate(step)
        if gate:
            return step, gate
    return None, ""


def _has_follow_up_after(step_number: int, steps: list[ActionStep]) -> bool:
    for step in steps:
        if int(getattr(step, "step_number", 0) or 0) <= step_number:
            continue
        if _is_degraded_synthesis_step(step):
            continue
        for tc in getattr(step, "tool_calls", []) or []:
            name = _tool_name(tc)
            if name and name not in IGNORED_FOLLOWUP_TOOLS:
                return True
    return False


def _final_answer_names_concrete_next_action(final_answer: Any) -> bool:
    text = str(final_answer or "")
    if not text.strip():
        return False
    if _NO_NEXT_ACTION_RE.search(text):
        return False
    return bool(_NEXT_ACTION_RE.search(text))


class SynthesisFinalizationGuard:
    """Rejects early final_answer when a synthesis run still has a concrete next gate."""

    def __init__(
        self,
        *,
        guard_config: SynthesisFinalizationGuardConfig | dict[str, Any] | None = None,
        synthesis_context: dict[str, Any] | None = None,
        log_dir: str = "/tmp/synthesis",
    ):
        self.config = (
            guard_config
            if isinstance(guard_config, SynthesisFinalizationGuardConfig)
            else SynthesisFinalizationGuardConfig.from_config(guard_config)
        )
        self.context = dict(synthesis_context or {})
        self.log_path = Path(log_dir) / GUARD_LOG_NAME

    def final_answer_check(self, final_answer, memory, agent) -> bool:
        """Instrumentation-only: log the rule-guard decision but never block.

        The B finalization gate is the sole decision authority. The guard's block
        decision is written to synthesis_finalization_guard.jsonl for reference.
        """
        decision = self.decide(final_answer, memory, agent)
        self._write_log(decision, final_answer)
        return True

    def decide(self, final_answer, memory, agent) -> GuardDecision:
        if not self.config.enabled:
            return self._allow("guard_disabled", final_answer, memory, agent)

        steps = _iter_action_steps(memory)
        c_steps = [step for step in steps if _has_tool(step, "request_synthesis_helper")]
        max_steps = int(getattr(agent, "max_steps", 0) or 0)
        current_step = int(getattr(agent, "step_number", 0) or 0)
        remaining_steps = max(max_steps - current_step, 0)

        if not c_steps:
            return GuardDecision(
                decision="allow",
                reason="no_synthesis_helper_used",
                c_used=False,
                target_reached=False,
                remaining_steps=remaining_steps,
                latest_c_step=None,
                latest_gate="",
                follow_up_after_latest_c=False,
                concrete_next_action_in_final=False,
            )

        concrete_next = _final_answer_names_concrete_next_action(final_answer)
        latest_c_with_gate, latest_gate = _latest_c_gate(c_steps)
        latest_c = latest_c_with_gate or c_steps[-1]
        latest_c_step = int(getattr(latest_c, "step_number", 0) or 0)
        target_signal = _latest_request_target_signal(c_steps) or str(self.context.get("sanitizer_report") or "")
        target_reached = _target_reached(steps, target_signal)
        follow_up = _has_follow_up_after(latest_c_step, steps)

        if target_reached:
            reason = "target_signal_observed"
        elif remaining_steps < self.config.min_remaining_steps:
            reason = "low_step_budget"
        elif latest_gate and not follow_up:
            return GuardDecision(
                decision="block",
                reason=f"latest_gate_unattempted:{latest_gate}",
                c_used=True,
                target_reached=False,
                remaining_steps=remaining_steps,
                latest_c_step=latest_c_step,
                latest_gate=latest_gate,
                follow_up_after_latest_c=False,
                concrete_next_action_in_final=concrete_next,
            )
        elif concrete_next:
            return GuardDecision(
                decision="block",
                reason="final_answer_names_concrete_next_action",
                c_used=True,
                target_reached=False,
                remaining_steps=remaining_steps,
                latest_c_step=latest_c_step,
                latest_gate=latest_gate,
                follow_up_after_latest_c=follow_up,
                concrete_next_action_in_final=True,
            )
        else:
            reason = "evidence_exhaustion_claim_accepted"

        return GuardDecision(
            decision="allow",
            reason=reason,
            c_used=True,
            target_reached=target_reached,
            remaining_steps=remaining_steps,
            latest_c_step=latest_c_step,
            latest_gate=latest_gate,
            follow_up_after_latest_c=follow_up,
            concrete_next_action_in_final=concrete_next,
        )

    def _decide(self, final_answer, memory, agent) -> GuardDecision:
        """Backward-compat shim — callers should use decide()."""
        return self.decide(final_answer, memory, agent)

    def _allow(self, reason: str, final_answer, memory, agent) -> GuardDecision:
        steps = _iter_action_steps(memory)
        max_steps = int(getattr(agent, "max_steps", 0) or 0)
        current_step = int(getattr(agent, "step_number", 0) or 0)
        return GuardDecision(
            decision="allow",
            reason=reason,
            c_used=any(_has_tool(step, "request_synthesis_helper") for step in steps),
            target_reached=False,
            remaining_steps=max(max_steps - current_step, 0),
            latest_c_step=None,
            latest_gate="",
            follow_up_after_latest_c=False,
            concrete_next_action_in_final=False,
        )

    def _write_log(self, decision: GuardDecision, final_answer: Any) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "schema_version": "v1",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "instance_id": self.context.get("instance_id"),
                "decision": decision.decision,
                "reason": decision.reason,
                "c_used": decision.c_used,
                "target_reached": decision.target_reached,
                "remaining_steps": decision.remaining_steps,
                "latest_c_step": decision.latest_c_step,
                "latest_gate": decision.latest_gate,
                "follow_up_after_latest_c": decision.follow_up_after_latest_c,
                "concrete_next_action_in_final": decision.concrete_next_action_in_final,
                "final_answer_head": str(final_answer or "")[:500],
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            import sys

            print(f"WARNING: synthesis finalization guard log write failed: {exc}", file=sys.stderr)


def create_synthesis_finalization_guard(
    *,
    guard_config: SynthesisFinalizationGuardConfig | dict[str, Any] | None = None,
    synthesis_context: dict[str, Any] | None = None,
    log_dir: str = "/tmp/synthesis",
) -> SynthesisFinalizationGuard:
    return SynthesisFinalizationGuard(
        guard_config=guard_config,
        synthesis_context=synthesis_context,
        log_dir=log_dir,
    )
