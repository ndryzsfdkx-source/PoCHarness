"""ToolCallingAgent backends for the Synthesis Helper and PoC Solver."""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.resources
import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Template
from rich.panel import Panel
from rich.text import Text

from smolagents.agents import ToolCallingAgent, ToolOutput
from smolagents.memory import ActionStep, ToolCall
from smolagents.models import LiteLLMModel
from smolagents.monitoring import LogLevel, Timing
from smolagents.secb.cost_budget import BudgetedModel, CostBudgetLedger
from smolagents.secb.harness._agent_utils import collect_agent_cost
from smolagents.secb.harness.config import SynthesisConfig
from smolagents.secb.harness.tools import (
    SynthesisToolState,
    build_synthesis_tools,
)
from smolagents.secb.harness.workspace import WorkspaceInspector
from smolagents.utils import AgentMaxStepsError


if TYPE_CHECKING:
    from smolagents.secb.review import FinalizationReviewer


def _read_template_text(template_path: str) -> str:
    if "/" in template_path or "\\" in template_path:
        if os.path.isabs(template_path) or os.path.exists(template_path):
            return Path(template_path).read_text(encoding="utf-8")
        anchor = importlib.resources.files("smolagents")
        for part in template_path.split("/"):
            anchor = anchor.joinpath(part)
        return anchor.read_text(encoding="utf-8")
    return (
        importlib.resources.files("smolagents.secb.harness.prompts")
        .joinpath(template_path)
        .read_text(encoding="utf-8")
    )


def _truncate(text: str, max_chars: int = 200) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


@dataclass
class SynthesisBackendResult:
    parse_status: str
    emitted_payload: dict[str, Any] | None
    c_steps_used: int
    c_tool_calls: list[dict[str, Any]]
    degraded: bool
    prompt: str
    error_message: str = ""
    raw_response: str = ""
    degraded_reason: str | None = None
    emit_error: str = ""
    partial_helper_files: list[str] = field(default_factory=list)
    c_input_tokens: int = 0
    c_output_tokens: int = 0
    c_cost: float | None = None
    termination_reason: str = ""
    step_budget: dict[str, Any] = field(default_factory=dict)


def _classify_degraded_reason(
    *,
    state: SynthesisToolState,
    error_message: str,
    termination_reason: str,
    c_steps_used: int,
    max_c_steps: int,
) -> str:
    """Return the most specific reason an Agent-C invocation degraded."""
    if termination_reason in {"cost_budget_exhausted", "pricing_unavailable"}:
        return termination_reason
    if state.emit_called and state.emitted_payload is None:
        return "emit_rejected"
    error_lower = str(error_message).lower()
    if "cost_budget_exhausted" in error_lower:
        return "cost_budget_exhausted"
    if "pricing_unavailable" in error_lower:
        return "pricing_unavailable"
    if any(marker in error_lower for marker in ("content filter", "policy", "safety")):
        return "policy_block"
    if "timed out" in error_lower:
        return "timeout"
    if error_message:
        return "backend_exception"
    if c_steps_used >= max(int(max_c_steps), 1):
        return "max_steps_exhausted"
    return "emit_not_called"


class SynthesisAgentBackend:
    """Single-use backend for one Synthesis Helper invocation."""

    def __init__(
        self,
        *,
        config: SynthesisConfig,
        static_context: dict[str, Any],
        invocation_index: int,
        cost_budget: CostBudgetLedger | None = None,
    ):
        self._config = config
        self._static_context = dict(static_context or {})
        self._invocation_index = invocation_index
        self._cost_budget = cost_budget

    def run(self, bundle: dict[str, Any]) -> SynthesisBackendResult:
        rendered_prompt = self._render_prompt(bundle=bundle)
        starting_cost = (
            float(self._cost_budget.snapshot()["observed_usd"])
            if self._cost_budget is not None and self._cost_budget.enabled
            else 0.0
        )
        work_dir = str(self._static_context.get("work_dir") or "")
        max_c_steps = int(self._config.agent_backend.max_c_steps)
        if (
            bundle.get("continuation_mode")
            and self._config.agent_backend.continuation_max_c_steps is not None
        ):
            max_c_steps = int(self._config.agent_backend.continuation_max_c_steps)
        inspector = WorkspaceInspector(
            self._config.agent_backend.workspace,
            work_dir=work_dir,
        )
        transport = self._config.model_transport_kwargs or {}
        model = LiteLLMModel(
            model_id=self._config.model_id,
            api_key=transport.get("api_key") or os.getenv("OPENAI_API_KEY"),
            api_base=transport.get("api_base"),
            service_tier=transport.get("service_tier", "default"),
        )
        if self._cost_budget is not None and self._cost_budget.enabled:
            model = BudgetedModel(
                model,
                ledger=self._cost_budget,
                role="c",
                service_tier=transport.get("service_tier", "default"),
            )
        state = SynthesisToolState(
            bundle=bundle,
            work_dir=work_dir,
            backend_config=self._config.agent_backend,
        )
        tools = build_synthesis_tools(
            tools_allow=self._config.agent_backend.tools_allow,
            state=state,
            inspector=inspector,
        )
        agent = SynthesisHelperAgent(
            tools=tools,
            model=model,
            max_steps=max(max_c_steps, 1),
            verbosity_level=LogLevel(0),
            instructions=rendered_prompt,
            stream_outputs=False,
        )
        task = (
            "Resolve the delegated subproblem with runnable files under Helpers/, "
            "then produce one accepted emit_helper emission. If an emission is rejected, "
            "correct it and retry."
        )
        error_message = ""
        recovery_used = False
        work_steps = 0
        recovery_steps = 0
        try:
            output = self._run_with_timeout(
                agent,
                task,
                timeout_seconds=max(int(self._config.agent_backend.barrier_timeout_seconds), 1),
            )
            raw_output = str(output)
            work_steps = self._collect_steps_used(agent)
            # The normal agent can stop before its step cap without calling
            # emit_helper. If it already wrote useful helper files, give it the
            # same bounded emission-only recovery used at the cap. Do not spend
            # a recovery turn on an early stop that produced nothing.
            if state.emitted_payload is None and (
                work_steps >= max_c_steps or bool(state.helper_files_written)
            ):
                recovery_used = True
                agent.tools = {
                    name: tool
                    for name, tool in agent.tools.items()
                    if name in {"write_helper", "emit_helper"}
                }
                recovery_task = (
                    "Recovery turn: use the preserved work. Write only any missing final helper "
                    "content, then call emit_helper. Do not investigate further."
                )
                recovery_output = self._run_with_timeout(
                    agent,
                    recovery_task,
                    timeout_seconds=max(
                        int(self._config.agent_backend.barrier_timeout_seconds),
                        1,
                    ),
                    reset=False,
                    max_steps=1,
                )
                raw_output = str(recovery_output)
        except Exception as exc:
            raw_output = f"Synthesis Helper error: {exc}"
            error_message = str(exc)

        tool_calls = self._collect_tool_calls(agent)
        total_steps = self._collect_steps_used(agent)
        if recovery_used:
            recovery_steps = max(total_steps - work_steps, 0)
        else:
            work_steps = total_steps
        c_steps_used = total_steps
        step_budget = {
            "mode": "continuation" if bundle.get("continuation_mode") else "fresh",
            "work_cap": max_c_steps,
            "work_steps": work_steps,
            "recovery_used": recovery_used,
            "recovery_steps": recovery_steps,
        }
        c_input_tokens, c_output_tokens = collect_agent_cost(agent)
        termination_reason = str(getattr(agent, "_termination_reason", "") or "")
        c_cost = None
        if self._cost_budget is not None and self._cost_budget.enabled:
            c_cost = max(float(self._cost_budget.snapshot()["observed_usd"]) - starting_cost, 0.0)
        if state.emitted_payload is None:
            degraded_reason = _classify_degraded_reason(
                state=state,
                error_message=error_message,
                termination_reason=termination_reason,
                c_steps_used=c_steps_used,
                max_c_steps=max_c_steps,
            )
            return SynthesisBackendResult(
                parse_status="failed",
                emitted_payload=None,
                c_steps_used=c_steps_used,
                c_tool_calls=tool_calls,
                degraded=True,
                prompt=rendered_prompt,
                error_message=error_message
                or termination_reason
                or state.emit_error
                or "emit_helper not called before max_c_steps.",
                raw_response=state.raw_terminal_observation or raw_output,
                degraded_reason=degraded_reason,
                emit_error=state.emit_error,
                partial_helper_files=list(state.helper_files_written),
                c_input_tokens=c_input_tokens,
                c_output_tokens=c_output_tokens,
                c_cost=c_cost,
                termination_reason=termination_reason,
                step_budget=step_budget,
            )
        return SynthesisBackendResult(
            parse_status="ok",
            emitted_payload=state.emitted_payload,
            c_steps_used=c_steps_used,
            c_tool_calls=tool_calls,
            degraded=False,
            prompt=rendered_prompt,
            error_message="",
            raw_response=state.raw_emit_payload or raw_output,
            degraded_reason=None,
            emit_error="",
            partial_helper_files=list(state.helper_files_written),
            c_input_tokens=c_input_tokens,
            c_output_tokens=c_output_tokens,
            c_cost=c_cost,
            termination_reason=termination_reason,
            step_budget=step_budget,
        )

    def _render_prompt(self, *, bundle: dict[str, Any]) -> str:
        template_text = _read_template_text(self._config.agent_backend.prompt_template)
        template = Template(template_text)
        context = {
            "instance_id": self._static_context.get("instance_id", ""),
            "work_dir": self._static_context.get("work_dir", ""),
            "bug_description": self._static_context.get("bug_description", ""),
            "sanitizer_report": self._static_context.get("sanitizer_report", ""),
            "invocation_index": self._invocation_index,
            "prompt_version": "synthesis_specialist/v2",
        }
        rendered = template.render(context=context, inputs=bundle)
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        return f"{rendered}\n\nPrompt hash: {digest}"

    @staticmethod
    def _run_with_timeout(
        agent: ToolCallingAgent,
        task: str,
        *,
        timeout_seconds: int,
        reset: bool = True,
        max_steps: int | None = None,
    ) -> Any:
        if threading.current_thread() is threading.main_thread():
            previous_handler = signal.getsignal(signal.SIGALRM)

            def _timeout_handler(signum, frame):
                del signum, frame
                raise TimeoutError(f"Synthesis Helper timed out after {timeout_seconds}s")

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            try:
                return agent.run(task, reset=reset, max_steps=max_steps)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            agent.run,
            task,
            reset=reset,
            max_steps=max_steps,
        )
        try:
            result = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(f"Synthesis Helper timed out after {timeout_seconds}s") from exc
        executor.shutdown(wait=True)
        return result

    @staticmethod
    def _collect_steps_used(agent: ToolCallingAgent) -> int:
        steps = [
            step
            for step in getattr(getattr(agent, "memory", None), "steps", []) or []
            if isinstance(step, ActionStep)
            and not isinstance(getattr(step, "error", None), AgentMaxStepsError)
        ]
        return len(steps)

    @staticmethod
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
                calls.append(
                    {
                        "step": getattr(step, "step_number", None),
                        "tool": getattr(call, "name", ""),
                        "args_summary": _truncate(summary, 200),
                        "obs_head": obs_head,
                    }
                )
        return calls


class SynthesisHelperAgent(ToolCallingAgent):
    """Compatibility class whose terminal tool is emit_helper."""

    terminal_tool_name = "emit_helper"
    # Backstop so a thrashing C cannot re-emit indefinitely. Once this many emit
    # attempts have been rejected, the next rejection terminates the invocation
    # (degraded). max_steps is the outer bound; this just gives a clean cap.
    max_emit_attempts = 4

    def _handle_max_steps_reached(self, task: str) -> Any:
        """End the work phase without spending another model call on free text."""
        del task
        now = time.time()
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps.", self.logger),
            timing=Timing(start_time=now, end_time=now),
        )
        final_memory_step.action_output = "work step cap reached"
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        return final_memory_step.action_output

    def _emit_is_terminal(self, emit_state: "SynthesisToolState | None") -> bool:
        """emit_helper ends the turn only when accepted, or after the retry budget is spent."""
        if emit_state is None:
            return False
        accepted = emit_state.emitted_payload is not None
        exhausted = emit_state.emit_attempts >= self.max_emit_attempts
        return accepted or exhausted

    def process_tool_calls(self, chat_message, memory_step: ActionStep):
        """Execute tool calls sequentially and stop after the first emit_helper.

        Same mechanic as advisor's AgentBToolCallingAgent — emit_helper plays the role of
        final_answer for this sub-agent.
        """
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
            # emit_helper is terminal ONLY when the emission was accepted. A rejected
            # emit leaves the loop running so C can correct (e.g. run repro, then re-emit)
            # instead of dying with its already-written helper files discarded.
            is_terminal = tool_name == "final_answer"
            if tool_name == self.terminal_tool_name:
                emit_tool = self.tools.get(self.terminal_tool_name)
                is_terminal = self._emit_is_terminal(getattr(emit_tool, "_state", None))
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


class PoCSolverAgent(ToolCallingAgent):
    """Main-agent subclass used when the synthesis scaffold is active.

    Makes `final_submission` conditionally terminal based on the B reviewer's verdict.
    When B allows finalization, the tool output is marked is_final_answer=True and the
    run terminates normally. When B blocks, is_final_answer=False and A continues with
    B's attached_action as the observation.

    Only used when [agent.synthesis] is enabled — does not affect the clean baseline.
    """

    def __init__(self, *args, reviewer: "FinalizationReviewer | None" = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._reviewer = reviewer
        # Remove final_answer so A must go through final_submission → B gate.
        # ToolCallingAgent.__init__ always injects it via setdefault; pop it here.
        self.tools.pop("final_answer", None)

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

            if tool_name == "final_submission":
                # Terminal only when B explicitly allowed. Empty verdict (e.g. validation
                # error returned before invoke()) is treated as block so A can retry.
                is_terminal = (
                    self._reviewer is None
                    or self._reviewer.last_verdict
                    in {"ALLOW_SUCCESS", "ALLOW_EXHAUSTED", "ALLOW_UNVERIFIED"}
                )
            else:
                is_terminal = tool_name == "final_answer"

            tool_output = ToolOutput(
                id=tool_call.id,
                output=tool_call_result,
                is_final_answer=is_terminal,
                observation=observation,
                tool_call=tool_call,
            )
            outputs.append(tool_output)
            if is_terminal:
                # Required: prevent _step_stream from raising on multi-tool messages.
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
