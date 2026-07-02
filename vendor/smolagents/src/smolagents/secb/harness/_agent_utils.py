"""Shared agent introspection helpers for B and C token logging."""
from __future__ import annotations

from typing import Any


def collect_agent_cost(agent: Any) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) for a completed sub-agent run.

    Walks agent.memory.steps and aggregates token_usage. Returns zeros on failure.

    Note: cost is NOT returned here. The naive litellm estimate this function used
    to compute was systematically wrong (no flex-tier pricing, no prefix-caching
    discount). Authoritative per-role costs come from the BudgetedModel ledger,
    written to meta.json by docker_app_runner.py when cost_budget_enabled=True.
    """
    input_tokens = 0
    output_tokens = 0
    for step in getattr(getattr(agent, "memory", None), "steps", []) or []:
        tu = getattr(step, "token_usage", None)
        if tu is not None:
            input_tokens += getattr(tu, "input_tokens", 0)
            output_tokens += getattr(tu, "output_tokens", 0)
    return input_tokens, output_tokens
