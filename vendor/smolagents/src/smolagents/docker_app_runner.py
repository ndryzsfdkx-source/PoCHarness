"""Runner script for executing agents inside Docker containers for SEC-bench evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Union

# Import smolagents - should be installed locally via install_local_smolagents before this script runs
from smolagents import CodeAgent, ToolCallingAgent
from smolagents.default_tools import TOOL_MAPPING
from smolagents.memory import ActionStep
from smolagents.models import InferenceClientModel, LiteLLMModel, OpenAIModel, TransformersModel
from smolagents.monitoring import LogLevel
from smolagents.secb.cost_budget import BudgetedModel, CostBudgetRegistry
from smolagents.secb.review import FinalizationReviewConfig, create_finalization_reviewer
from smolagents.secb.harness import (
    ArtifactGuardTool,
    FinalSubmissionTool,
    RequestSynthesisHelperTool,
    RunSecbReproOnCurrentTestcaseTool,
    SynthesisConfig,
    create_synthesis_finalization_guard,
    create_synthesis_orchestrator,
    run_harness_shape_probe,
)
from smolagents.secb.harness.agent import PoCSolverAgent


def _build_model(model_config: dict[str, Any]) -> Any:
    """Build a model from configuration."""
    model_type = model_config.get("type", "InferenceClientModel")
    model_id = model_config.get("model_id", "")

    if model_type == "LiteLLMModel":
        try:
            import litellm

            litellm.drop_params = model_id.startswith("openai/gpt-5")
        except ImportError:
            pass

    if model_type == "InferenceClientModel":
        return InferenceClientModel(
            model_id=model_id,
            token=model_config.get("api_key") or os.getenv("HF_API_KEY"),
            provider=model_config.get("provider"),
        )
    elif model_type == "OpenAIModel":
        return OpenAIModel(
            model_id=model_id,
            api_key=model_config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            api_base=model_config.get("api_base"),
        )
    elif model_type == "LiteLLMModel":
        return LiteLLMModel(
            model_id=model_id,
            api_key=model_config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            api_base=model_config.get("api_base"),
            service_tier=model_config.get("service_tier", "default"),
        )
    elif model_type == "TransformersModel":
        return TransformersModel(model_id=model_id, device_map="auto")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _build_tools(tool_names: list[str]) -> list[Any]:
    """Build tools from tool names."""
    tools = []
    for tool_name in tool_names:
        if tool_name in TOOL_MAPPING:
            tools.append(TOOL_MAPPING[tool_name]())
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    return tools


def _should_enable_synthesis(agent_config: dict[str, Any]) -> bool:
    synthesis_config = agent_config.get("pocharness") or {}
    return bool(synthesis_config.get("enabled"))


def _append_action_step_callback(step_callbacks, callback):
    if step_callbacks is None:
        return {ActionStep: callback}
    if ActionStep not in step_callbacks:
        step_callbacks[ActionStep] = callback
        return step_callbacks
    existing = step_callbacks[ActionStep]
    step_callbacks[ActionStep] = [*existing, callback] if isinstance(existing, list) else [existing, callback]
    return step_callbacks


def _append_final_answer_check(final_answer_checks, check):
    if final_answer_checks is None:
        return [check]
    return [*final_answer_checks, check]


def _model_transport_with_role_overrides(
    base_transport: dict[str, Any],
    role_config: dict[str, Any] | None,
) -> dict[str, Any]:
    transport = dict(base_transport)
    role_config = role_config or {}
    if role_config.get("service_tier"):
        transport["service_tier"] = role_config["service_tier"]
    return transport


def _build_agent(
    agent_config: dict[str, Any],
    inherited_model_config: dict[str, Any] | None = None,
    *,
    cost_budgets: CostBudgetRegistry | None = None,
    budget_role: str = "a",
) -> Any:
    """Build an agent, optionally with managed subagents, from configuration."""
    merged_model_config = dict(inherited_model_config or {})
    merged_model_config.update(agent_config.get("model") or {})
    synthesis_enabled = _should_enable_synthesis(agent_config)
    if cost_budgets is None:
        cost_budgets = CostBudgetRegistry.from_config(agent_config.get("cost_budget"))
    cost_budget = cost_budgets.ledger_for(budget_role)

    model = _build_model(merged_model_config)
    if cost_budget.enabled:
        model = BudgetedModel(
            model,
            ledger=cost_budget,
            role=budget_role,
            service_tier=merged_model_config.get("service_tier"),
        )
    tools = _build_tools(agent_config.get("tools", []))
    agent_type = agent_config.get("agent_type", "ToolCallingAgent")
    verbosity_level = LogLevel(agent_config.get("verbosity_level", 1))

    managed_agents = [
        _build_agent(
            managed_agent_config,
            inherited_model_config=merged_model_config,
            cost_budgets=cost_budgets,
            budget_role=f"managed:{managed_agent_config.get('name') or 'unnamed'}",
        )
        for managed_agent_config in agent_config.get("managed_agents", [])
    ]

    step_callbacks = None
    final_answer_checks = None

    synthesis = None
    finalization_guard = None
    finalization_reviewer = None
    if synthesis_enabled:
        model_transport_kwargs = {
            "api_key": merged_model_config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            "api_base": merged_model_config.get("api_base"),
            "service_tier": merged_model_config.get("service_tier", "default"),
        }
        raw_synthesis_config = agent_config.get("pocharness") or {}
        c_model_transport_kwargs = _model_transport_with_role_overrides(
            model_transport_kwargs,
            raw_synthesis_config.get("helper"),
        )
        b_model_transport_kwargs = _model_transport_with_role_overrides(
            model_transport_kwargs,
            raw_synthesis_config.get("reviewer"),
        )
        # c_disabled=true: wire B and A's artifact guard without registering C tools.
        c_disabled = bool(raw_synthesis_config.get("c_disabled", False))
        # Resolved once in cli.py and threaded through agent_config; B/C never recompute
        # this from task_type themselves. A config that didn't set the TOML key still gets
        # this resolved value via the description_mode= kwarg below.
        description_mode = bool(agent_config.get("description_mode"))

        if not c_disabled:
            synthesis = create_synthesis_orchestrator(
                synthesis_config=raw_synthesis_config,
                synthesis_context=agent_config.get("synthesis_context"),
                default_model_id=merged_model_config.get("model_id"),
                model_transport_kwargs=c_model_transport_kwargs,
                cost_budget=cost_budgets.ledger_for("c"),
                description_mode=description_mode,
            )
            synthesis_config = synthesis.config
        else:
            synthesis_config = SynthesisConfig.from_config(
                raw_synthesis_config,
                default_model_id=merged_model_config.get("model_id"),
                description_mode=description_mode,
            )

        # Rule guard: instrumentation-only (never blocks); always register so it logs.
        finalization_guard = create_synthesis_finalization_guard(
            guard_config=synthesis_config.finalization_guard,
            synthesis_context=agent_config.get("synthesis_context"),
            log_dir=synthesis_config.log_dir,
        )
        final_answer_checks = _append_final_answer_check(
            final_answer_checks,
            finalization_guard.final_answer_check,
        )

        # B finalization gate: spawns B sub-agent when A calls final_submission.
        from pathlib import Path as _Path
        synthesis_log_path = _Path(synthesis_config.log_dir) / "synthesis_log.jsonl"
        review_config = FinalizationReviewConfig.from_config(
            raw_synthesis_config.get("reviewer"),
            fallback_model_id=synthesis_config.model_id,
            description_mode=description_mode,
        )
        finalization_reviewer = create_finalization_reviewer(
            review_config=review_config,
            static_context=agent_config.get("synthesis_context") or {},
            synthesis_log_path=synthesis_log_path,
            log_dir=synthesis_config.log_dir,
            model_transport_kwargs=b_model_transport_kwargs,
            guard_decide_fn=finalization_guard.decide,
            cost_budget=cost_budgets.ledger_for("b") if review_config.enabled else None,
        )
        if synthesis is not None:
            tools.append(RunSecbReproOnCurrentTestcaseTool(orchestrator=synthesis))
            tools.append(
                RequestSynthesisHelperTool(orchestrator=synthesis)
            )
        tools.append(ArtifactGuardTool())
        tools.append(FinalSubmissionTool(reviewer=finalization_reviewer))

    common_kwargs = {
        "max_steps": agent_config.get("max_steps", 20),
        "verbosity_level": verbosity_level,
        "managed_agents": managed_agents or None,
        "instructions": agent_config.get("instructions"),
        "planning_interval": agent_config.get("planning_interval"),
        "name": agent_config.get("name"),
        "description": agent_config.get("description"),
        "provide_run_summary": agent_config.get("provide_run_summary", False),
        "step_callbacks": step_callbacks,
        "final_answer_checks": final_answer_checks,
    }
    common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}

    if synthesis_enabled and agent_type == "ToolCallingAgent":
        # Use PoCSolverAgent when synthesis is active so final_submission
        # is conditionally terminal based on the B reviewer's verdict.
        agent = PoCSolverAgent(
            tools=tools,
            model=model,
            stream_outputs=False,
            reviewer=finalization_reviewer,
            **common_kwargs,
        )
    elif agent_type == "ToolCallingAgent":
        agent = ToolCallingAgent(
            tools=tools,
            model=model,
            stream_outputs=False,
            **common_kwargs,
        )
    elif agent_type == "CodeAgent":
        agent = CodeAgent(
            tools=tools,
            model=model,
            stream_outputs=False,
            additional_authorized_imports=agent_config.get("additional_imports", []),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    if synthesis is not None:
        synthesis.agent_ref = agent
    # Must be outside the synthesis-not-None guard: when c_disabled=True,
    # synthesis=None but finalization_reviewer is still active and needs agent_ref
    # to compute remaining_steps. Without this, remaining_steps=0 and low-budget
    # deferral fires immediately, skipping B entirely.
    if finalization_reviewer is not None:
        finalization_reviewer.agent_ref = agent
    # The cached harness shape is shared infrastructure for A's artifact guard,
    # B's repro tool, and C. In A+B mode c_disabled=True leaves synthesis=None,
    # but B still requires the probe output.
    if synthesis_enabled:
        try:
            synthesis_context = agent_config.get("synthesis_context") or {}
            run_harness_shape_probe(work_dir=synthesis_context.get("work_dir"))
        except Exception as exc:  # never block A on probe failure
            print(f"WARNING: harness_shape_probe failed: {exc}", file=sys.stderr)
    if synthesis_enabled:
        agent._synthesis_log_dir = synthesis_config.log_dir
    # Stash the orchestrator so the run loop can flush the observability sidecar at run end.
    agent._synthesis_orchestrator = synthesis
    agent._cost_budgets = cost_budgets
    agent._cost_budget = cost_budget
    return agent


def _sum_jsonl_costs(path: Path, cost_key: str, in_key: str, out_key: str) -> tuple[float, int, int]:
    """Sum cost/token fields across all records in a JSONL file. Returns (cost, input, output)."""
    total_cost, total_in, total_out = 0.0, 0, 0
    try:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    total_cost += float(rec.get(cost_key, 0) or 0)
                    total_in += int(rec.get(in_key, 0) or 0)
                    total_out += int(rec.get(out_key, 0) or 0)
                except Exception:
                    pass
    except Exception:
        pass
    return total_cost, total_in, total_out


def _write_meta_json(
    artifacts_dir: str,
    agent: Union[ToolCallingAgent, CodeAgent],
    result: Any,
) -> None:
    """Write metadata JSON file for the agent run."""
    try:
        # Ensure artifacts directory exists
        out_dir = Path(artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Extract model name
        model_name = getattr(agent.model, "model_id", None) or agent.model.__class__.__name__

        # Extract agent name
        agent_name = agent.__class__.__name__

        # Extract tool names
        tools_attr = getattr(agent, "tools", {})
        if isinstance(tools_attr, dict):
            tool_names = sorted(list(tools_attr.keys()))
        elif isinstance(tools_attr, list):
            tool_names = sorted([getattr(t, "name", t.__class__.__name__) for t in tools_attr])
        else:
            tool_names = []

        # Count steps and aggregate token usage from agent.memory.steps
        steps_count = 0
        input_tokens = 0
        output_tokens = 0

        for step in agent.memory.steps:
            # Count steps that have step_number (ActionStep)
            if hasattr(step, "step_number"):
                steps_count += 1

            # Extract token usage from step
            tu = getattr(step, "token_usage", None)
            if tu is not None:
                input_tokens += getattr(tu, "input_tokens", 0)
                output_tokens += getattr(tu, "output_tokens", 0)

        # If result has token_usage, prefer that (more accurate aggregation)
        if hasattr(result, "token_usage") and result.token_usage is not None:
            input_tokens = result.token_usage.input_tokens
            output_tokens = result.token_usage.output_tokens

        # Calculate A cost using litellm
        cost = 0.0
        try:
            from litellm import completion_cost

            fake_response = {
                "model": model_name.split("/")[-1],
                "usage": {"prompt_tokens": int(input_tokens), "completion_tokens": int(output_tokens)},
            }
            cost = float(completion_cost(fake_response))
        except Exception:
            pass

        # Sum B and C costs from their per-invocation JSONL logs (written by synthesis scaffold).
        b_cost, b_input_tokens, b_output_tokens = 0.0, 0, 0
        c_cost, c_input_tokens, c_output_tokens = 0.0, 0, 0
        synthesis_log_dir = getattr(agent, "_synthesis_log_dir", None)
        if synthesis_log_dir:
            log_dir_path = Path(synthesis_log_dir)
            b_cost, b_input_tokens, b_output_tokens = _sum_jsonl_costs(
                log_dir_path / "finalization_review.jsonl",
                "b_cost", "b_input_tokens", "b_output_tokens",
            )
            c_cost, c_input_tokens, c_output_tokens = _sum_jsonl_costs(
                log_dir_path / "synthesis_log.jsonl",
                "c_cost", "c_input_tokens", "c_output_tokens",
            )

        # Build metadata dictionary
        meta = {
            "model": model_name,
            "agent": agent_name,
            "tools": tool_names,
            "steps": steps_count,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        if synthesis_log_dir:
            meta["b_cost"] = b_cost
            meta["b_input_tokens"] = b_input_tokens
            meta["b_output_tokens"] = b_output_tokens
            meta["c_cost"] = c_cost
            meta["c_input_tokens"] = c_input_tokens
            meta["c_output_tokens"] = c_output_tokens
            meta["total_cost"] = cost + b_cost + c_cost
            meta["total_input_tokens"] = input_tokens + b_input_tokens + c_input_tokens
            meta["total_output_tokens"] = output_tokens + b_output_tokens + c_output_tokens

        cost_budget_source = getattr(agent, "_cost_budgets", None) or getattr(agent, "_cost_budget", None)
        if cost_budget_source is not None and cost_budget_source.enabled:
            budget = cost_budget_source.snapshot()
            cost_by_role = budget["cost_by_role"]
            input_by_role = budget["input_tokens_by_role"]
            output_by_role = budget["output_tokens_by_role"]
            meta["cost"] = cost_by_role.get("a", 0.0)
            meta["input_tokens"] = input_by_role.get("a", 0)
            meta["output_tokens"] = output_by_role.get("a", 0)
            meta["b_cost"] = cost_by_role.get("b", 0.0)
            meta["b_input_tokens"] = input_by_role.get("b", 0)
            meta["b_output_tokens"] = output_by_role.get("b", 0)
            meta["c_cost"] = cost_by_role.get("c", 0.0)
            meta["c_input_tokens"] = input_by_role.get("c", 0)
            meta["c_output_tokens"] = output_by_role.get("c", 0)
            meta["overhead_cost"] = meta["b_cost"] + meta["c_cost"]
            managed_roles = [role for role in cost_by_role if role.startswith("managed:")]
            meta["managed_cost"] = sum(cost_by_role[role] for role in managed_roles)
            meta["managed_input_tokens"] = sum(input_by_role.get(role, 0) for role in managed_roles)
            meta["managed_output_tokens"] = sum(output_by_role.get(role, 0) for role in managed_roles)
            meta["total_cost"] = budget["observed_usd"]
            meta["total_input_tokens"] = sum(input_by_role.values())
            meta["total_output_tokens"] = sum(output_by_role.values())
            meta["p_cost"] = cost_by_role.get("p", 0.0)
            meta["p_input_tokens"] = input_by_role.get("p", 0)
            meta["p_output_tokens"] = output_by_role.get("p", 0)
            meta["cost_budget_enabled"] = True
            meta["cost_budget_mode"] = budget.get("mode", "shared")
            meta["cost_budget_limit_usd"] = budget["limit_usd"]
            meta["cost_budget_observed_usd"] = budget["observed_usd"]
            meta["cost_budget_exhausted"] = budget["exhausted"]
            meta["cost_budget_overshoot_usd"] = budget["overshoot_usd"]
            meta["cost_budget_status"] = budget["status"]
            meta["cost_by_role"] = cost_by_role
            meta["model_calls_by_role"] = budget["model_calls_by_role"]
            if budget.get("budget_by_role"):
                meta["cost_budget_by_role"] = budget["budget_by_role"]
            if budget["pricing_error"]:
                meta["cost_budget_pricing_error"] = budget["pricing_error"]

        termination_reason = getattr(agent, "_termination_reason", "")
        if termination_reason:
            meta["termination_reason"] = termination_reason

        # Add docker_image if available
        docker_image = os.getenv("DOCKER_IMAGE")
        if docker_image:
            meta["docker_image"] = docker_image
        if getattr(agent, "managed_agents", None):
            meta["managed_agents"] = sorted(list(agent.managed_agents.keys()))

        # Write meta.json using Path.write_text (more robust)
        meta_path = out_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=False), encoding="utf-8")

    except Exception as e:
        # Log warning but don't fail the entire run
        import traceback

        print(f"Warning: failed to write meta.json: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


def _apply_extra_build_flags() -> None:
    """Merge SECB_EXTRA_CFLAGS/CXXFLAGS into the container's build environment."""
    for src_var, dst_var in [("SECB_EXTRA_CFLAGS", "CFLAGS"), ("SECB_EXTRA_CXXFLAGS", "CXXFLAGS")]:
        extra = os.environ.get(src_var, "").strip()
        if extra:
            existing = os.environ.get(dst_var, "")
            os.environ[dst_var] = f"{existing} {extra}".strip()

def main() -> None:
    """Main entry point for the Docker container runner."""
    _apply_extra_build_flags()

    # Parse command-line arguments (fallback to environment variables)
    parser = argparse.ArgumentParser(description="Run smolagents application inside container")
    parser.add_argument("--config", help="Path to agent config JSON inside container")
    parser.add_argument("--task", help="Task string to run")
    parser.add_argument("--artifacts-dir", help="Directory to write trajectory and outputs")
    parser.add_argument("--max-steps", type=int, help="Optional max steps override")
    args = parser.parse_args()

    # Load agent config (prefer CLI arg, fallback to env var, then default)
    config_path = args.config or os.environ.get("SMOLAGENTS_CONFIG_PATH", "/app/agent_config.json")
    with open(config_path, "r") as f:
        agent_config = json.load(f)

    # Load task (prefer CLI arg, fallback to env var, then default)
    if args.task:
        task = args.task
    else:
        task_path = os.environ.get("SMOLAGENTS_TASK_PATH", "/app/task.txt")
        with open(task_path, "r") as f:
            task = f.read()

    # Resolve either the legacy shared pool or independent per-role ledgers.
    cost_budgets = CostBudgetRegistry.from_config(agent_config.get("cost_budget"))

    # Create agent tree
    if args.max_steps is not None:
        agent_config = dict(agent_config)
        agent_config["max_steps"] = args.max_steps
    agent: Union[ToolCallingAgent, CodeAgent] = _build_agent(agent_config, cost_budgets=cost_budgets)

    # Run agent
    try:
        result = agent.run(task, return_full_result=True)

        # Resolve synthesis observability into its sidecar now that the run is complete.
        _orchestrator = getattr(agent, "_synthesis_orchestrator", None)
        if _orchestrator is not None:
            _orchestrator.finalize_observability()

        # Extract output
        if hasattr(result, "output"):
            output = result.output
        else:
            output = result

        # Save result
        # Prefer CLI arg for artifacts_dir, fallback to env var, then default
        artifacts_dir = args.artifacts_dir or os.environ.get("SMOLAGENTS_ARTIFACTS_DIR", "/app/artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        with open(os.path.join(artifacts_dir, "output.json"), "w") as f:
            json.dump(
                {
                    "output": str(output) if output is not None else "",
                    "steps": result.steps if hasattr(result, "steps") else [],
                    "state": getattr(result, "state", None),
                    "termination_reason": getattr(agent, "_termination_reason", ""),
                },
                f,
            )

        # Save trajectory
        if hasattr(result, "steps"):
            with open(os.path.join(artifacts_dir, "trajectory.jsonl"), "w") as f:
                for step in result.steps:
                    f.write(json.dumps(step) + "\n")

        # Save metadata
        _write_meta_json(artifacts_dir, agent, result)

        sys.exit(0)
    except Exception as e:
        import traceback

        # Prefer CLI arg for artifacts_dir, fallback to env var, then default
        artifacts_dir = args.artifacts_dir or os.environ.get("SMOLAGENTS_ARTIFACTS_DIR", "/app/artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        with open(os.path.join(artifacts_dir, "error.txt"), "w") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
