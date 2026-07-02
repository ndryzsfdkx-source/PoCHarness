import math
from types import SimpleNamespace

import litellm
import pytest

from smolagents.agents import ToolCallingAgent
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
)
from smolagents.monitoring import TokenUsage
from smolagents.secb.cost_budget import (
    BudgetedModel,
    CostBudgetConfig,
    CostBudgetLedger,
    CostBudgetRegistry,
)
from smolagents.tools import Tool
from smolagents.utils import AgentBudgetExceeded


class FakeModel:
    model_id = "openai/gpt-5.4-2026-03-05"

    def __init__(self, *, input_tokens: int = 40, output_tokens: int = 10):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    def generate(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(name="noop", arguments={}),
                    id=f"call-{self.calls}",
                    type="function",
                )
            ],
            token_usage=TokenUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            ),
        )

    def parse_tool_calls(self, message):
        return message


class FakeStreamingModel:
    model_id = "openai/gpt-5.4-2026-03-05"

    def generate_stream(self, *args, **kwargs):
        del args, kwargs
        yield SimpleNamespace(token_usage=TokenUsage(input_tokens=30, output_tokens=0))
        yield SimpleNamespace(token_usage=None)
        yield SimpleNamespace(token_usage=TokenUsage(input_tokens=0, output_tokens=12))


class NoopTool(Tool):
    name = "noop"
    description = "Return a local observation without changing state."
    inputs = {}
    output_type = "string"

    def forward(self):
        return "ok"


def _token_pricing(*, completion_response=None, **kwargs):
    del kwargs
    usage = completion_response["usage"]
    return (usage["prompt_tokens"] + usage["completion_tokens"]) / 100.0


def test_cost_budget_config_defaults_and_validation():
    assert CostBudgetConfig.from_config(None) == CostBudgetConfig(enabled=False, max_total_cost_usd=10.0)
    assert CostBudgetConfig.from_config({"enabled": True}) == CostBudgetConfig(
        enabled=True,
        max_total_cost_usd=10.0,
    )
    assert CostBudgetConfig.from_config(
        {"enabled": True, "max_total_cost_usd": 7.5}
    ).max_total_cost_usd == 7.5

    for invalid in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError):
            CostBudgetConfig.from_config({"enabled": True, "max_total_cost_usd": invalid})


def test_per_role_registry_validates_shape_and_missing_roles():
    with pytest.raises(ValueError, match="mode must be"):
        CostBudgetRegistry.from_config({"enabled": True, "mode": "unknown"})
    with pytest.raises(ValueError, match="non-empty table"):
        CostBudgetRegistry.from_config({"enabled": True, "mode": "per_role"})
    with pytest.raises(ValueError, match="ambiguous"):
        CostBudgetRegistry.from_config(
            {
                "enabled": True,
                "mode": "per_role",
                "max_total_cost_usd": 2.5,
                "roles": {"a": {"max_total_cost_usd": 2.5}},
            }
        )

    registry = CostBudgetRegistry.from_config(
        {
            "enabled": True,
            "mode": "per_role",
            "roles": {"a": {"max_total_cost_usd": 2.5}},
        }
    )
    with pytest.raises(ValueError, match="No per-role cost budget configured for 'c'"):
        registry.ledger_for("c")


def test_per_role_registry_isolates_exhaustion_and_aggregates_exact_costs(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", _token_pricing)
    registry = CostBudgetRegistry.from_config(
        {
            "enabled": True,
            "mode": "per_role",
            "roles": {
                "a": {"max_total_cost_usd": 1.0},
                "b": {"max_total_cost_usd": 0.4},
                "c": {"max_total_cost_usd": 0.7},
            },
        }
    )
    a_model = FakeModel(input_tokens=40, output_tokens=10)
    b_model = FakeModel(input_tokens=30, output_tokens=10)
    c_model = FakeModel(input_tokens=50, output_tokens=20)

    BudgetedModel(c_model, ledger=registry.ledger_for("c"), role="c").generate([])
    assert registry.ledger_for("c").snapshot()["status"] == "exhausted"
    assert registry.ledger_for("a").snapshot()["status"] == "within_budget"
    assert registry.ledger_for("b").snapshot()["status"] == "within_budget"

    # C cannot call again, but exhausting it does not block A or B.
    with pytest.raises(AgentBudgetExceeded):
        BudgetedModel(c_model, ledger=registry.ledger_for("c"), role="c").generate([])
    BudgetedModel(a_model, ledger=registry.ledger_for("a"), role="a").generate([])
    BudgetedModel(b_model, ledger=registry.ledger_for("b"), role="b").generate([])

    snapshot = registry.snapshot()
    assert snapshot["mode"] == "per_role"
    assert snapshot["status"] == "role_exhausted"
    assert snapshot["cost_by_role"] == {"a": 0.5, "b": 0.4, "c": 0.7}
    assert snapshot["observed_usd"] == pytest.approx(1.6)
    assert snapshot["limit_usd"] == pytest.approx(2.1)
    assert snapshot["model_calls_by_role"] == {"a": 1, "b": 1, "c": 1}
    assert snapshot["budget_by_role"]["a"]["status"] == "within_budget"
    assert snapshot["budget_by_role"]["b"]["status"] == "exhausted"
    assert snapshot["budget_by_role"]["c"]["status"] == "exhausted"


@pytest.mark.parametrize("exhausted_role", ["a", "b", "c"])
def test_exhausting_one_role_leaves_every_other_role_available(monkeypatch, exhausted_role):
    monkeypatch.setattr(litellm, "completion_cost", _token_pricing)
    registry = CostBudgetRegistry.from_config(
        {
            "enabled": True,
            "mode": "per_role",
            "roles": {
                "a": {"max_total_cost_usd": 0.5},
                "b": {"max_total_cost_usd": 0.5},
                "c": {"max_total_cost_usd": 0.5},
            },
        }
    )

    BudgetedModel(
        FakeModel(input_tokens=40, output_tokens=10),
        ledger=registry.ledger_for(exhausted_role),
        role=exhausted_role,
    ).generate([])

    assert registry.ledger_for(exhausted_role).snapshot()["status"] == "exhausted"
    for role in {"a", "b", "c"} - {exhausted_role}:
        assert registry.ledger_for(role).snapshot()["status"] == "within_budget"
        registry.ledger_for(role).ensure_available()


def test_per_role_overshoot_stays_with_responsible_role(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", _token_pricing)
    registry = CostBudgetRegistry.from_config(
        {
            "enabled": True,
            "mode": "per_role",
            "roles": {
                "a": {"max_total_cost_usd": 1.0},
                "b": {"max_total_cost_usd": 1.0},
                "c": {"max_total_cost_usd": 0.6},
            },
        }
    )

    BudgetedModel(
        FakeModel(input_tokens=50, output_tokens=20),
        ledger=registry.ledger_for("c"),
        role="c",
    ).generate([])

    snapshot = registry.snapshot()
    assert snapshot["budget_by_role"]["c"]["overshoot_usd"] == pytest.approx(0.1)
    assert snapshot["budget_by_role"]["a"]["overshoot_usd"] == 0.0
    assert snapshot["budget_by_role"]["b"]["overshoot_usd"] == 0.0
    registry.ledger_for("a").ensure_available()
    registry.ledger_for("b").ensure_available()


def test_legacy_registry_keeps_one_shared_ledger():
    registry = CostBudgetRegistry.from_config(
        {"enabled": True, "max_total_cost_usd": 2.5}
    )

    assert registry.mode == "shared"
    assert registry.ledger_for("a") is registry.ledger_for("b")
    assert registry.ledger_for("b") is registry.ledger_for("c")


def test_shared_ledger_aggregates_roles_and_allows_one_call_overshoot(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", _token_pricing)
    ledger = CostBudgetLedger(CostBudgetConfig(enabled=True, max_total_cost_usd=1.0))
    a_model = FakeModel(input_tokens=30, output_tokens=20)
    c_model = FakeModel(input_tokens=50, output_tokens=25)

    BudgetedModel(a_model, ledger=ledger, role="a").generate([])
    BudgetedModel(c_model, ledger=ledger, role="c").generate([])

    snapshot = ledger.snapshot()
    assert snapshot["status"] == "exhausted"
    assert snapshot["observed_usd"] == 1.25
    assert snapshot["overshoot_usd"] == 0.25
    assert snapshot["cost_by_role"] == {"a": 0.5, "c": 0.75}
    assert snapshot["model_calls_by_role"] == {"a": 1, "c": 1}

    with pytest.raises(AgentBudgetExceeded) as exc:
        BudgetedModel(a_model, ledger=ledger, role="a").generate([])
    assert exc.value.reason == "cost_budget_exhausted"
    assert a_model.calls == 1


def test_pricing_failure_stops_before_next_model_call(monkeypatch):
    def fail_pricing(*args, **kwargs):
        del args, kwargs
        raise ValueError("missing price")

    monkeypatch.setattr(litellm, "completion_cost", fail_pricing)
    ledger = CostBudgetLedger(CostBudgetConfig(enabled=True))
    model = FakeModel()
    wrapped = BudgetedModel(model, ledger=ledger, role="a")

    wrapped.generate([])
    assert ledger.snapshot()["status"] == "pricing_unavailable"
    with pytest.raises(AgentBudgetExceeded) as exc:
        wrapped.generate([])
    assert exc.value.reason == "pricing_unavailable"
    assert model.calls == 1


def test_generate_stream_accumulates_usage(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", _token_pricing)
    ledger = CostBudgetLedger(CostBudgetConfig(enabled=True, max_total_cost_usd=10.0))
    wrapped = BudgetedModel(FakeStreamingModel(), ledger=ledger, role="a")

    events = list(wrapped.generate_stream([]))

    assert len(events) == 3
    snapshot = ledger.snapshot()
    assert snapshot["input_tokens_by_role"] == {"a": 30}
    assert snapshot["output_tokens_by_role"] == {"a": 12}
    assert snapshot["cost_by_role"] == {"a": 0.42}
    assert snapshot["model_calls_by_role"] == {"a": 1}


def test_missing_usage_stops_safely(monkeypatch):
    def unexpected_pricing_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError("completion_cost must not be called without token usage")

    monkeypatch.setattr(litellm, "completion_cost", unexpected_pricing_call)
    ledger = CostBudgetLedger(CostBudgetConfig(enabled=True))

    ledger.record_response(
        role="a",
        model_id="openai/gpt-5.4-2026-03-05",
        response=SimpleNamespace(token_usage=None, raw=None),
    )

    snapshot = ledger.snapshot()
    assert snapshot["status"] == "pricing_unavailable"
    assert "did not include token usage" in snapshot["pricing_error"]


def test_budget_exception_from_step_callback_terminates_gracefully(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", _token_pricing)
    ledger = CostBudgetLedger(CostBudgetConfig(enabled=True, max_total_cost_usd=0.25))
    fake_model = FakeModel()

    def budget_callback(step, agent=None):
        del step, agent
        ledger.ensure_available()

    agent = ToolCallingAgent(
        tools=[NoopTool()],
        model=BudgetedModel(fake_model, ledger=ledger, role="a"),
        max_steps=5,
        step_callbacks=[budget_callback],
    )

    result = agent.run("test callback termination", return_full_result=True)

    assert result.state == "budget_exhausted"
    assert agent._termination_reason == "cost_budget_exhausted"
    assert fake_model.calls == 1


@pytest.mark.parametrize(
    ("pricing", "expected_state", "expected_reason"),
    [
        (_token_pricing, "budget_exhausted", "cost_budget_exhausted"),
        (lambda **kwargs: (_ for _ in ()).throw(ValueError("no price")), "pricing_unavailable", "pricing_unavailable"),
    ],
)
def test_agent_terminates_gracefully_without_final_answer_checks(
    monkeypatch,
    pricing,
    expected_state,
    expected_reason,
):
    monkeypatch.setattr(litellm, "completion_cost", pricing)
    ledger = CostBudgetLedger(CostBudgetConfig(enabled=True, max_total_cost_usd=0.25))
    fake_model = FakeModel()
    final_check_calls = []
    agent = ToolCallingAgent(
        tools=[NoopTool()],
        model=BudgetedModel(fake_model, ledger=ledger, role="a"),
        max_steps=5,
        final_answer_checks=[lambda *args, **kwargs: final_check_calls.append(True)],
    )

    result = agent.run("test budget termination", return_full_result=True)

    assert result.state == expected_state
    assert agent._termination_reason == expected_reason
    assert "current artifact will be preserved" in str(result.output)
    assert fake_model.calls == 1
    assert final_check_calls == []
