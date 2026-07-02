"""B finalization gate reviewer — spawns a sub-agent when A tries to finalize."""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.resources
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Template
from rich.panel import Panel
from rich.text import Text

from smolagents.agents import ToolCallingAgent, ToolOutput
from smolagents.memory import ActionStep, ToolCall
from smolagents.models import LiteLLMModel
from smolagents.monitoring import LogLevel
from smolagents.secb.cost_budget import BudgetedModel, CostBudgetLedger
from smolagents.secb.sanitizer.profile import ObservedCrashProfile, profile_from_dict
from smolagents.secb.harness._agent_utils import collect_agent_cost
from smolagents.secb.harness.workspace import WorkspaceInspector

from .config import FinalizationReviewConfig
from .evidence import (
    EvidencePolicyState,
    EvidenceRelation,
    INSUFFICIENT_EVIDENCE,
    apply_evidence_policy,
    extract_literal_target_profile,
    target_profile_from_dict,
)
from .log import write_finalization_review_log
from .tools import (
    ALLOW_VERDICTS,
    FinalizationReviewToolState,
    build_review_tools,
)


def _read_template_text(template_path: str) -> str:
    if "/" in template_path or "\\" in template_path:
        if os.path.isabs(template_path) or os.path.exists(template_path):
            return Path(template_path).read_text(encoding="utf-8")
        anchor = importlib.resources.files("smolagents")
        for part in template_path.split("/"):
            anchor = anchor.joinpath(part)
        return anchor.read_text(encoding="utf-8")
    return (
        importlib.resources.files("smolagents.secb.review.prompts")
        .joinpath(template_path)
        .read_text(encoding="utf-8")
    )


def _truncate(text: str, max_chars: int = 200) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _iter_action_steps(memory) -> list[Any]:
    return [step for step in getattr(memory, "steps", []) or [] if isinstance(step, ActionStep)]


def _collect_valid_step_indices(steps: list[Any]) -> set[int]:
    indices = set()
    for step in steps:
        n = getattr(step, "step_number", None)
        if n is not None:
            indices.add(int(n))
    return indices


def _load_synthesis_log(log_path: Path) -> list[dict[str, Any]]:
    records = []
    if not log_path.exists():
        return records
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return records


def _compute_post_c_tool_calls(steps: list[Any]) -> list[dict[str, Any]]:
    """Slice of tool calls after the last request_synthesis_helper, for helper adoption inference."""
    last_c_step = -1
    for step in steps:
        calls = getattr(step, "tool_calls", []) or []
        if any(getattr(tc, "name", "") == "request_synthesis_helper" for tc in calls):
            last_c_step = int(getattr(step, "step_number", -1) or -1)

    if last_c_step < 0:
        return []
    result = []
    for step in steps:
        if int(getattr(step, "step_number", -1) or -1) <= last_c_step:
            continue
        for tc in getattr(step, "tool_calls", []) or []:
            result.append({"step": getattr(step, "step_number", None), "tool": getattr(tc, "name", "")})
    return result


@dataclass
class ReviewResult:
    verdict: str
    reasoning: str
    attached_action: str
    evidence_steps: list[int]
    b_steps_used: int
    b_tool_calls: list[dict[str, Any]]
    degraded: bool
    degraded_reason: str
    prompt_hash: str
    dropped_invalid_evidence: list[dict[str, Any]]
    repro_result: dict[str, Any] | None = None
    b_input_tokens: int = 0
    b_output_tokens: int = 0
    b_cost: float | None = None
    termination_reason: str = ""
    observed_profile: dict[str, Any] | None = None
    target_profile: dict[str, Any] | None = None
    replay_consistency: dict[str, Any] | None = None
    evidence_relation: dict[str, Any] | None = None


class PoCReviewerAgent(ToolCallingAgent):
    """ToolCallingAgent where emit_finalization_verdict is terminal (mirrors AgentBToolCallingAgent)."""

    terminal_tool_name = "emit_finalization_verdict"

    def process_tool_calls(self, chat_message, memory_step: ActionStep):
        assert chat_message.tool_calls is not None
        executed_calls: list[ToolCall] = []
        outputs: list[ToolOutput] = []

        for chat_tool_call in chat_message.tool_calls:
            tool_call = ToolCall(
                name=chat_tool_call.function.name,
                arguments=chat_tool_call.function.arguments,
                id=chat_tool_call.id,
            )
            executed_calls.append(tool_call)
            yield tool_call

            tool_name = tool_call.name
            tool_arguments = tool_call.arguments or {}
            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            tool_call_result = self.execute_tool_call(tool_name, tool_arguments)
            observation = str(tool_call_result).strip()
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",
                level=LogLevel.INFO,
            )
            is_terminal = tool_name in {"final_answer", self.terminal_tool_name}
            tool_output = ToolOutput(
                id=tool_call.id,
                output=tool_call_result,
                is_final_answer=is_terminal,
                observation=observation,
                tool_call=tool_call,
            )
            outputs.append(tool_output)
            if is_terminal:
                chat_message.tool_calls = [chat_tool_call]
            yield tool_output

            if is_terminal:
                break

        memory_step.tool_calls = executed_calls
        memory_step.observations = memory_step.observations or ""
        for tool_output in outputs:
            memory_step.observations += tool_output.observation + "\n"
        memory_step.observations = (
            memory_step.observations.rstrip("\n") if memory_step.observations else memory_step.observations
        )


class FinalizationReviewer:
    """Manages B-gate finalization reviews: spawns a B sub-agent on each final_submission call."""

    def __init__(
        self,
        *,
        config: FinalizationReviewConfig,
        static_context: dict[str, Any],
        synthesis_log_path: Path,
        log_dir: str,
        guard_decide_fn: Any = None,
        cost_budget: CostBudgetLedger | None = None,
    ):
        self._config = config
        self._static_context = dict(static_context or {})
        self._synthesis_log_path = synthesis_log_path
        self._log_dir = log_dir
        self._guard_decide_fn = guard_decide_fn
        self._cost_budget = cost_budget
        self.block_index: int = 0
        self.last_verdict: str = ""
        self.agent_ref: Any = None
        self._evidence_policy_state = EvidencePolicyState()

    def invoke(
        self,
        *,
        artifact_status: str,
        stop_reason: str,
        memory: Any,
    ) -> tuple[bool, str]:
        """Run the B gate. Returns (allow_finalization, payload_or_action)."""
        steps = _iter_action_steps(memory)
        valid_step_indices = _collect_valid_step_indices(steps)
        max_steps = int(getattr(self.agent_ref, "max_steps", 0) or 0)
        current_step = int(getattr(self.agent_ref, "step_number", 0) or 0)
        remaining_steps = max(max_steps - current_step, 0)

        # Rule guard decision (instrumentation, non-binding).
        rule_guard_decision: dict[str, Any] = {}
        if self._guard_decide_fn is not None:
            try:
                rg = self._guard_decide_fn(artifact_status, memory, self.agent_ref)
                rule_guard_decision = {
                    "decision": rg.decision,
                    "reason": rg.reason,
                    "remaining_steps": rg.remaining_steps,
                    "latest_gate": rg.latest_gate,
                }
            except Exception as exc:
                rule_guard_decision = {"error": str(exc)}
        b_rule_guard_decision = {
            key: value
            for key, value in rule_guard_decision.items()
            if key != "remaining_steps"
        }

        # Gate disabled → always allow.
        if not self._config.enabled:
            self.last_verdict = "ALLOW_EXHAUSTED"
            return True, artifact_status

        # max_blocks cap: allow through when cap is reached (None = unlimited).
        if self._config.max_blocks is not None and self.block_index >= self._config.max_blocks:
            self.last_verdict = "ALLOW_EXHAUSTED"
            self._write_log(
                verdict="ALLOW_EXHAUSTED",
                reasoning="max_blocks_reached",
                attached_action="",
                evidence_steps=[],
                b_steps_used=0,
                b_tool_calls=[],
                degraded=False,
                degraded_reason="",
                prompt_hash="",
                rule_guard_decision=rule_guard_decision,
                remaining_steps=remaining_steps,
                block_index=self.block_index,
                block_index_input=self.block_index,
                block_count_after=self.block_index,
                artifact_status=artifact_status,
                stop_reason=stop_reason,
                dropped_invalid_evidence=[],
            )
            return True, artifact_status

        # Spawn B sub-agent.
        block_index_input = self.block_index
        synthesis_log_records = _load_synthesis_log(self._synthesis_log_path)
        post_c_tool_calls = _compute_post_c_tool_calls(steps)
        bundle = {
            "steps": steps,
            "valid_step_indices": valid_step_indices,
            "synthesis_log_records": synthesis_log_records,
            "post_c_tool_calls": post_c_tool_calls,
            "rule_guard_decision": b_rule_guard_decision,
            "block_index": block_index_input,
            "artifact_status": artifact_status,
            "stop_reason": stop_reason,
            "instance_id": self._static_context.get("instance_id", ""),
            "work_dir": self._static_context.get("work_dir", ""),
            "bug_description": self._static_context.get("bug_description", ""),
            "sanitizer_report": self._static_context.get("sanitizer_report", ""),
            "target_signal": (
                self._static_context.get("target_signal")
                or self._static_context.get("sanitizer_report", "")
            ),
        }

        result = self._run_b_agent(bundle, valid_step_indices)
        proposed_verdict = result.verdict
        recommended_verdict = proposed_verdict
        policy_applied = False
        challenge = None
        if self._config.evidence_policy.enabled:
            observed = profile_from_dict(result.observed_profile)
            if observed is None:
                observed = ObservedCrashProfile(report_completeness="environment_failure")
            target = (
                target_profile_from_dict(result.target_profile)
                if result.target_profile
                else extract_literal_target_profile(str(bundle.get("bug_description") or ""))
            )
            relation = (
                EvidenceRelation(**result.evidence_relation)
                if result.evidence_relation
                else EvidenceRelation(
                    INSUFFICIENT_EVIDENCE,
                    reasoning="B did not produce a complete evidence relation",
                )
            )
            policy = apply_evidence_policy(
                mode=self._config.evidence_policy.mode,
                proposed_verdict=proposed_verdict,
                relation=relation,
                target=target,
                observed=observed,
                state=self._evidence_policy_state,
                remaining_steps=remaining_steps,
                min_remaining_steps=self._config.min_remaining_steps,
                review_index=block_index_input,
                degraded=result.degraded,
                hard_gate_scope=self._config.evidence_policy.hard_gate_scope,
            )
            result.verdict = policy.actual_verdict
            recommended_verdict = policy.recommended_verdict
            policy_applied = policy.applied
            challenge = policy.challenge.to_dict() if policy.challenge else None
            if policy.attached_action:
                result.attached_action = policy.attached_action
        self.last_verdict = result.verdict

        allow = result.verdict in ALLOW_VERDICTS
        if not allow:
            self.block_index += 1

        self._write_log(
            verdict=result.verdict,
            reasoning=result.reasoning,
            attached_action=result.attached_action,
            evidence_steps=result.evidence_steps,
            b_steps_used=result.b_steps_used,
            b_tool_calls=result.b_tool_calls,
            degraded=result.degraded,
            degraded_reason=result.degraded_reason,
            prompt_hash=result.prompt_hash,
            rule_guard_decision=rule_guard_decision,
            remaining_steps=remaining_steps,
            block_index=self.block_index,
            block_index_input=block_index_input,
            block_count_after=self.block_index,
            artifact_status=artifact_status,
            stop_reason=stop_reason,
            dropped_invalid_evidence=result.dropped_invalid_evidence,
            b_input_tokens=result.b_input_tokens,
            b_output_tokens=result.b_output_tokens,
            b_cost=result.b_cost,
            b_termination_reason=result.termination_reason,
            repro_result=result.repro_result,
            proposed_verdict=proposed_verdict,
            recommended_verdict=recommended_verdict,
            policy_mode=(
                self._config.evidence_policy.mode
                if self._config.evidence_policy.enabled
                else "legacy"
            ),
            policy_applied=policy_applied,
            challenge=challenge,
            observed_profile=result.observed_profile,
            target_profile=result.target_profile,
            replay_consistency=result.replay_consistency,
            evidence_relation=result.evidence_relation,
        )

        if allow:
            return True, artifact_status
        return False, result.attached_action

    def _run_b_agent(self, bundle: dict[str, Any], valid_step_indices: set[int]) -> ReviewResult:
        rendered_prompt, prompt_hash = self._render_prompt(bundle)
        starting_cost = (
            float(self._cost_budget.snapshot()["observed_usd"])
            if self._cost_budget is not None and self._cost_budget.enabled
            else 0.0
        )
        work_dir = bundle.get("work_dir", "")
        inspector = WorkspaceInspector(self._config.workspace, work_dir=work_dir)
        model_id = self._config.model_id or "openai/gpt-5.4-2026-03-05"
        transport = self._static_context.get("model_transport_kwargs") or {}
        model = LiteLLMModel(
            model_id=model_id,
            api_key=transport.get("api_key") or os.getenv("OPENAI_API_KEY"),
            api_base=transport.get("api_base"),
            service_tier=transport.get("service_tier", "default"),
        )
        if self._cost_budget is not None and self._cost_budget.enabled:
            model = BudgetedModel(
                model,
                ledger=self._cost_budget,
                role="b",
                service_tier=transport.get("service_tier", "default"),
            )
        # judge_model reuses B's own model/transport for the description-mode judge call
        # (assess_description_match) -- extra cost lands under the same "b" budget role.
        state = FinalizationReviewToolState(
            steps=list(bundle.get("steps") or []),
            valid_step_indices=valid_step_indices,
            synthesis_log_records=bundle.get("synthesis_log_records") or [],
            work_dir=str(bundle.get("work_dir") or ""),
            target_signal=str(bundle.get("target_signal") or ""),
            description_mode=self._config.description_mode,
            judge_mode=self._config.judge_mode,
            bug_description=str(bundle.get("bug_description") or ""),
            instance_id=str(bundle.get("instance_id") or ""),
            judge_model=model,
            evidence_policy_enabled=self._config.evidence_policy.enabled,
            evidence_policy_mode=self._config.evidence_policy.mode,
            target_profile_mode=self._config.evidence_policy.target_profile_mode,
            model_version=model_id,
            log_dir=self._log_dir,
            review_index=self.block_index,
        )
        tools = build_review_tools(
            tools_allow=self._config.tools_allow,
            state=state,
            inspector=inspector,
        )
        agent = PoCReviewerAgent(
            tools=tools,
            model=model,
            max_steps=max(int(self._config.max_b_steps), 1),
            verbosity_level=LogLevel(0),
            instructions=rendered_prompt,
            stream_outputs=False,
        )
        task = (
            "Review the PoC Solver's finalization claim and emit exactly one verdict "
            "via emit_finalization_verdict."
        )
        error_message = ""
        try:
            _run_with_timeout(agent, task, timeout_seconds=max(int(self._config.barrier_timeout_seconds), 1))
        except Exception as exc:
            error_message = str(exc)

        b_steps_used = _collect_steps_used(agent)
        b_tool_calls = _collect_tool_calls(agent)
        b_input_tokens, b_output_tokens = collect_agent_cost(agent)
        termination_reason = str(getattr(agent, "_termination_reason", "") or "")
        b_cost = None
        if self._cost_budget is not None and self._cost_budget.enabled:
            b_cost = max(float(self._cost_budget.snapshot()["observed_usd"]) - starting_cost, 0.0)

        if not state.emit_called:
            degraded_reason = (
                termination_reason
                if termination_reason in {"cost_budget_exhausted", "pricing_unavailable"}
                else error_message or "emit_not_called"
            )
            return ReviewResult(
                verdict="CONTINUE_LOCAL",
                reasoning=f"b_gate_degraded: {degraded_reason}",
                attached_action=(
                    "Agent B could not complete its review. Return to the source path driving "
                    "the target crash, identify what precondition your candidate has not "
                    "satisfied, and restart your solve loop from that requirement before stopping."
                ),
                evidence_steps=[],
                b_steps_used=b_steps_used,
                b_tool_calls=b_tool_calls,
                degraded=True,
                degraded_reason=degraded_reason,
                prompt_hash=prompt_hash,
                dropped_invalid_evidence=[],
                repro_result=state.repro_result,
                b_input_tokens=b_input_tokens,
                b_output_tokens=b_output_tokens,
                b_cost=b_cost,
                termination_reason=termination_reason,
                observed_profile=state.observed_profile,
                target_profile=state.target_profile,
                replay_consistency=state.replay_consistency,
                evidence_relation=state.evidence_relation,
            )
        return ReviewResult(
            verdict=state.emitted_verdict,
            reasoning=state.emitted_reasoning,
            attached_action=state.emitted_attached_action,
            evidence_steps=state.emitted_evidence_steps,
            b_steps_used=b_steps_used,
            b_tool_calls=b_tool_calls,
            degraded=False,
            degraded_reason="",
            prompt_hash=prompt_hash,
            dropped_invalid_evidence=state.dropped_invalid_evidence,
            repro_result=state.repro_result,
            b_input_tokens=b_input_tokens,
            b_output_tokens=b_output_tokens,
            b_cost=b_cost,
            termination_reason=termination_reason,
            observed_profile=state.observed_profile,
            target_profile=state.target_profile,
            replay_consistency=state.replay_consistency,
            evidence_relation=state.evidence_relation,
        )

    def _render_prompt(self, bundle: dict[str, Any]) -> tuple[str, str]:
        template_text = _read_template_text(self._config.prompt_template)
        template = Template(template_text)
        rendered = template.render(context=self._static_context, inputs=bundle)
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        return f"{rendered}\n\nPrompt hash: {digest}", digest

    def _write_log(self, **kwargs) -> None:
        write_finalization_review_log(
            log_dir=self._log_dir,
            instance_id=self._static_context.get("instance_id", ""),
            **kwargs,
        )


def _run_with_timeout(agent: ToolCallingAgent, task: str, *, timeout_seconds: int) -> Any:
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGALRM)

        def _timeout_handler(signum, frame):
            del signum, frame
            raise TimeoutError(f"B finalization reviewer timed out after {timeout_seconds}s")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        try:
            return agent.run(task)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(agent.run, task)
    try:
        result = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"B finalization reviewer timed out after {timeout_seconds}s") from exc
    executor.shutdown(wait=True)
    return result


def _collect_steps_used(agent: ToolCallingAgent) -> int:
    steps = [
        getattr(step, "step_number", 0) or 0
        for step in getattr(getattr(agent, "memory", None), "steps", []) or []
        if isinstance(step, ActionStep)
    ]
    return max(steps) if steps else 0


def _collect_tool_calls(agent: ToolCallingAgent) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for step in getattr(getattr(agent, "memory", None), "steps", []) or []:
        if not isinstance(step, ActionStep):
            continue
        obs_head = _truncate(" ".join(str(getattr(step, "observations", "") or "").split()), 300)
        for call in getattr(step, "tool_calls", None) or []:
            arguments = getattr(call, "arguments", {}) or {}
            try:
                summary = json.dumps(arguments, sort_keys=True)
            except TypeError:
                summary = str(arguments)
            calls.append({
                "step": getattr(step, "step_number", None),
                "tool": getattr(call, "name", ""),
                "args_summary": _truncate(summary, 200),
                "obs_head": obs_head,
            })
    return calls


def create_finalization_reviewer(
    *,
    review_config: FinalizationReviewConfig,
    static_context: dict[str, Any],
    synthesis_log_path: Path,
    log_dir: str,
    model_transport_kwargs: dict[str, Any] | None = None,
    guard_decide_fn: Any = None,
    cost_budget: CostBudgetLedger | None = None,
) -> FinalizationReviewer:
    ctx = dict(static_context or {})
    if model_transport_kwargs:
        ctx["model_transport_kwargs"] = model_transport_kwargs
    return FinalizationReviewer(
        config=review_config,
        static_context=ctx,
        synthesis_log_path=synthesis_log_path,
        log_dir=log_dir,
        guard_decide_fn=guard_decide_fn,
        cost_budget=cost_budget,
    )
