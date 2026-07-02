"""Cost ledgers and model wrappers for one SEC-bench instance."""
from __future__ import annotations

import math
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Generator

from smolagents.monitoring import TokenUsage
from smolagents.utils import AgentBudgetExceeded


DEFAULT_MAX_TOTAL_COST_USD = 10.0


@dataclass(frozen=True)
class CostBudgetConfig:
    enabled: bool = False
    max_total_cost_usd: float = DEFAULT_MAX_TOTAL_COST_USD

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "CostBudgetConfig":
        config = config or {}
        limit = float(config.get("max_total_cost_usd", DEFAULT_MAX_TOTAL_COST_USD))
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("agent.cost_budget.max_total_cost_usd must be a finite positive number.")
        return cls(
            enabled=bool(config.get("enabled", False)),
            max_total_cost_usd=limit,
        )


class CostBudgetRegistry:
    """Resolve shared or independent per-role ledgers for one instance.

    Flat legacy configs keep one shared ledger. ``mode = "per_role"`` creates
    one cumulative ledger per configured role, so exhausting an auxiliary
    agent cannot consume another role's allowance.
    """

    VALID_MODES = {"shared", "per_role"}

    def __init__(
        self,
        *,
        mode: str,
        ledgers: dict[str, "CostBudgetLedger"],
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown cost budget mode: {mode!r}")
        if mode == "shared" and set(ledgers) != {"shared"}:
            raise ValueError("Shared cost budget registry requires exactly one 'shared' ledger.")
        if mode == "per_role" and not ledgers:
            raise ValueError("Per-role cost budget registry requires at least one role.")
        self.mode = mode
        self._ledgers = dict(ledgers)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "CostBudgetRegistry":
        config = config or {}
        enabled = bool(config.get("enabled", False))
        mode = str(config.get("mode") or "shared").strip().lower()
        if mode not in cls.VALID_MODES:
            raise ValueError(
                "agent.cost_budget.mode must be 'shared' or 'per_role', "
                f"got: {mode!r}"
            )

        if mode == "shared":
            ledger = CostBudgetLedger(CostBudgetConfig.from_config(config))
            return cls(mode=mode, ledgers={"shared": ledger})

        if "max_total_cost_usd" in config:
            raise ValueError(
                "agent.cost_budget.max_total_cost_usd is ambiguous in per_role mode; "
                "configure agent.cost_budget.roles.<role>.max_total_cost_usd instead."
            )
        roles = config.get("roles") or {}
        if not isinstance(roles, dict) or not roles:
            raise ValueError("agent.cost_budget.roles must be a non-empty table in per_role mode.")

        ledgers: dict[str, CostBudgetLedger] = {}
        for raw_role, raw_role_config in roles.items():
            role = str(raw_role).strip()
            if not role:
                raise ValueError("agent.cost_budget.roles contains an empty role name.")
            if not isinstance(raw_role_config, dict):
                raise ValueError(f"agent.cost_budget.roles.{role} must be a table.")
            role_config = dict(raw_role_config)
            role_config["enabled"] = enabled
            try:
                parsed = CostBudgetConfig.from_config(role_config)
            except ValueError as exc:
                raise ValueError(f"Invalid cost budget for role {role!r}: {exc}") from exc
            ledgers[role] = CostBudgetLedger(parsed)
        return cls(mode=mode, ledgers=ledgers)

    @property
    def enabled(self) -> bool:
        return any(ledger.enabled for ledger in self._ledgers.values())

    def ledger_for(self, role: str) -> "CostBudgetLedger":
        if self.mode == "shared":
            return self._ledgers["shared"]
        try:
            return self._ledgers[role]
        except KeyError as exc:
            configured = ", ".join(sorted(self._ledgers)) or "none"
            raise ValueError(
                f"No per-role cost budget configured for {role!r}; configured roles: {configured}."
            ) from exc

    def snapshot(self) -> dict[str, Any]:
        if self.mode == "shared":
            snapshot = self._ledgers["shared"].snapshot()
            return {**snapshot, "mode": "shared", "budget_by_role": {}}

        role_snapshots = {role: ledger.snapshot() for role, ledger in self._ledgers.items()}
        cost_by_role: dict[str, float] = {}
        input_by_role: dict[str, int] = {}
        output_by_role: dict[str, int] = {}
        calls_by_role: dict[str, int] = {}
        pricing_errors: list[str] = []
        for role, snapshot in role_snapshots.items():
            cost_by_role[role] = float(sum(snapshot["cost_by_role"].values()))
            input_by_role[role] = int(sum(snapshot["input_tokens_by_role"].values()))
            output_by_role[role] = int(sum(snapshot["output_tokens_by_role"].values()))
            calls_by_role[role] = int(sum(snapshot["model_calls_by_role"].values()))
            if snapshot["pricing_error"]:
                pricing_errors.append(f"{role}: {snapshot['pricing_error']}")

        statuses = {snapshot["status"] for snapshot in role_snapshots.values()}
        if "pricing_unavailable" in statuses:
            status = "pricing_unavailable"
        elif "exhausted" in statuses:
            status = "role_exhausted"
        else:
            status = "within_budget"

        budget_by_role = {
            role: {
                "enabled": snapshot["enabled"],
                "limit_usd": snapshot["limit_usd"],
                "observed_usd": snapshot["observed_usd"],
                "exhausted": snapshot["exhausted"],
                "overshoot_usd": snapshot["overshoot_usd"],
                "status": snapshot["status"],
                "pricing_error": snapshot["pricing_error"],
            }
            for role, snapshot in role_snapshots.items()
        }
        observed = sum(snapshot["observed_usd"] for snapshot in role_snapshots.values())
        return {
            "enabled": self.enabled,
            "mode": "per_role",
            "limit_usd": sum(snapshot["limit_usd"] for snapshot in role_snapshots.values()),
            "observed_usd": observed,
            "exhausted": any(snapshot["exhausted"] for snapshot in role_snapshots.values()),
            "overshoot_usd": sum(snapshot["overshoot_usd"] for snapshot in role_snapshots.values()),
            "status": status,
            "pricing_error": "; ".join(pricing_errors),
            "cost_by_role": cost_by_role,
            "input_tokens_by_role": input_by_role,
            "output_tokens_by_role": output_by_role,
            "model_calls_by_role": calls_by_role,
            "budget_by_role": budget_by_role,
        }


def _usage_counts(value: Any) -> tuple[int, int]:
    usage = getattr(value, "token_usage", None)
    if usage is None:
        usage = getattr(value, "usage", None)
    if usage is None and isinstance(value, dict):
        usage = value.get("usage")
    if usage is None:
        return 0, 0

    def _read(*names: str) -> int:
        for name in names:
            if isinstance(usage, dict) and name in usage:
                return int(usage.get(name) or 0)
            attr = getattr(usage, name, None)
            if attr is not None:
                return int(attr or 0)
        return 0

    return _read("input_tokens", "prompt_tokens"), _read("output_tokens", "completion_tokens")


def _estimate_cost(
    *,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    raw_response: Any,
    service_tier: str | None,
) -> float:
    from litellm import completion_cost

    if raw_response is not None:
        try:
            return float(completion_cost(completion_response=raw_response, service_tier=service_tier))
        except Exception:
            pass
    if input_tokens == 0 and output_tokens == 0:
        raise ValueError("model response did not include token usage")
    response = {
        "model": model_id,
        "usage": {
            "prompt_tokens": int(input_tokens),
            "completion_tokens": int(output_tokens),
        },
    }
    return float(completion_cost(completion_response=response, service_tier=service_tier))


class CostBudgetLedger:
    """Thread-safe cumulative ledger for one shared pool or agent role."""

    def __init__(self, config: CostBudgetConfig):
        self.config = config
        self._state_lock = threading.Lock()
        self._call_lock = threading.RLock()
        self._observed_cost_usd = 0.0
        self._status = "within_budget"
        self._pricing_error = ""
        self._cost_by_role: dict[str, float] = defaultdict(float)
        self._input_tokens_by_role: dict[str, int] = defaultdict(int)
        self._output_tokens_by_role: dict[str, int] = defaultdict(int)
        self._model_calls_by_role: dict[str, int] = defaultdict(int)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def ensure_available(self) -> None:
        if not self.enabled:
            return
        with self._state_lock:
            status = self._status
            observed = self._observed_cost_usd
        if status != "within_budget":
            reason = "pricing_unavailable" if status == "pricing_unavailable" else "cost_budget_exhausted"
            raise AgentBudgetExceeded(
                reason=reason,
                observed_cost=observed,
                limit=self.config.max_total_cost_usd,
            )

    @contextmanager
    def call_slot(self) -> Generator[None, None, None]:
        if not self.enabled:
            yield
            return
        with self._call_lock:
            self.ensure_available()
            yield

    def record_response(
        self,
        *,
        role: str,
        model_id: str,
        response: Any,
        service_tier: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        input_tokens, output_tokens = _usage_counts(response)
        raw_response = getattr(response, "raw", None)
        if raw_response is None and getattr(response, "usage", None) is not None:
            raw_response = response
        try:
            cost = _estimate_cost(
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_response=raw_response,
                service_tier=service_tier,
            )
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(f"invalid completion cost: {cost}")
        except Exception as exc:
            with self._state_lock:
                self._model_calls_by_role[role] += 1
                self._input_tokens_by_role[role] += input_tokens
                self._output_tokens_by_role[role] += output_tokens
                self._status = "pricing_unavailable"
                self._pricing_error = str(exc)
            return

        with self._state_lock:
            self._model_calls_by_role[role] += 1
            self._input_tokens_by_role[role] += input_tokens
            self._output_tokens_by_role[role] += output_tokens
            self._cost_by_role[role] += cost
            self._observed_cost_usd += cost
            if self._observed_cost_usd >= self.config.max_total_cost_usd:
                self._status = "exhausted"

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            observed = self._observed_cost_usd
            status = self._status
            return {
                "enabled": self.enabled,
                "limit_usd": self.config.max_total_cost_usd,
                "observed_usd": observed,
                "exhausted": status == "exhausted",
                "overshoot_usd": max(observed - self.config.max_total_cost_usd, 0.0),
                "status": status,
                "pricing_error": self._pricing_error,
                "cost_by_role": dict(self._cost_by_role),
                "input_tokens_by_role": dict(self._input_tokens_by_role),
                "output_tokens_by_role": dict(self._output_tokens_by_role),
                "model_calls_by_role": dict(self._model_calls_by_role),
            }


class BudgetedModel:
    """Proxy that accounts for one role's model calls in a shared ledger."""

    def __init__(
        self,
        model: Any,
        *,
        ledger: CostBudgetLedger,
        role: str,
        service_tier: str | None = None,
    ):
        self._model = model
        self._ledger = ledger
        self._role = role
        self._service_tier = service_tier

    @property
    def model_id(self) -> str | None:
        return getattr(self._model, "model_id", None)

    def generate(self, *args, **kwargs):
        with self._ledger.call_slot():
            response = self._model.generate(*args, **kwargs)
            self._ledger.record_response(
                role=self._role,
                model_id=str(self.model_id or "unknown"),
                response=response,
                service_tier=self._service_tier,
            )
            return response

    def generate_stream(self, *args, **kwargs):
        def _stream():
            # The slot is intentionally acquired on first iteration, when the
            # upstream model call actually starts. Callers must consume streams
            # immediately rather than queueing unstarted generators.
            with self._ledger.call_slot():
                input_tokens = 0
                output_tokens = 0
                for event in self._model.generate_stream(*args, **kwargs):
                    if getattr(event, "token_usage", None) is not None:
                        input_tokens += int(event.token_usage.input_tokens or 0)
                        output_tokens += int(event.token_usage.output_tokens or 0)
                    yield event
                self._ledger.record_response(
                    role=self._role,
                    model_id=str(self.model_id or "unknown"),
                    response=SimpleNamespace(
                        token_usage=TokenUsage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        ),
                        raw=None,
                    ),
                    service_tier=self._service_tier,
                )

        return _stream()

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


def call_with_budget(
    *,
    ledger: CostBudgetLedger | None,
    role: str,
    model_id: str,
    service_tier: str | None,
    call: Callable[[], Any],
) -> Any:
    """Account for a direct model call that does not use a smolagents Model."""
    if ledger is None or not ledger.enabled:
        return call()
    with ledger.call_slot():
        response = call()
        ledger.record_response(
            role=role,
            model_id=model_id,
            response=response,
            service_tier=service_tier,
        )
        return response
