from smolagents.docker_app_runner import (
    _append_final_answer_check,
    _build_agent,
)
from smolagents.secb.cost_budget import BudgetedModel
from smolagents import docker_app_runner


def test_append_final_answer_check_composes_with_existing_checks():
    def first(*args):
        return True

    def second(*args):
        return True

    assert _append_final_answer_check(None, first) == [first]
    assert _append_final_answer_check([first], second) == [first, second]


def test_build_agent_shares_instance_budget_with_managed_agents():
    agent = _build_agent(
        {
            "model": {
                "type": "LiteLLMModel",
                "model_id": "openai/gpt-5.4-mini-2026-03-17",
            },
            "agent_type": "ToolCallingAgent",
            "tools": [],
            "cost_budget": {
                "enabled": True,
                "max_total_cost_usd": 10.0,
            },
            "managed_agents": [
                {
                    "name": "helper",
                    "description": "Offline test helper.",
                    "agent_type": "ToolCallingAgent",
                    "tools": [],
                    "max_steps": 1,
                }
            ],
        }
    )

    helper = agent.managed_agents["helper"]
    assert isinstance(agent.model, BudgetedModel)
    assert isinstance(helper.model, BudgetedModel)
    assert agent._cost_budget is helper._cost_budget
    assert agent.model._ledger is helper.model._ledger


def test_build_agent_wires_independent_abc_ledgers(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_app_runner, "run_harness_shape_probe", lambda **kwargs: None)
    agent = _build_agent(
        {
            "model": {
                "type": "LiteLLMModel",
                "model_id": "openai/gpt-5.4-mini-2026-03-17",
            },
            "agent_type": "ToolCallingAgent",
            "tools": [],
            "max_steps": 2,
            "cost_budget": {
                "enabled": True,
                "mode": "per_role",
                "roles": {
                    "a": {"max_total_cost_usd": 2.5},
                    "b": {"max_total_cost_usd": 1.0},
                    "c": {"max_total_cost_usd": 2.5},
                },
            },
            "pocharness": {
                "enabled": True,
                "log_dir": str(tmp_path),
                "reviewer": {"enabled": True},
            },
            "synthesis_context": {"work_dir": str(tmp_path)},
        }
    )

    a_ledger = agent.model._ledger
    b_ledger = agent._reviewer._cost_budget
    c_ledger = agent._synthesis_orchestrator.cost_budget
    assert len({id(a_ledger), id(b_ledger), id(c_ledger)}) == 3
    assert a_ledger is agent._cost_budgets.ledger_for("a")
    assert b_ledger is agent._cost_budgets.ledger_for("b")
    assert c_ledger is agent._cost_budgets.ledger_for("c")


def test_build_agent_allows_role_local_bc_service_tiers(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_app_runner, "run_harness_shape_probe", lambda **kwargs: None)
    agent = _build_agent(
        {
            "model": {
                "type": "LiteLLMModel",
                "model_id": "openai/gpt-5.4-mini-2026-03-17",
                "service_tier": "flex",
            },
            "agent_type": "ToolCallingAgent",
            "tools": [],
            "max_steps": 2,
            "cost_budget": {
                "enabled": True,
                "mode": "per_role",
                "roles": {
                    "a": {"max_total_cost_usd": 2.5},
                    "b": {"max_total_cost_usd": 1.0},
                    "c": {"max_total_cost_usd": 2.5},
                },
            },
            "pocharness": {
                "enabled": True,
                "log_dir": str(tmp_path),
                "helper": {"service_tier": "default"},
                "reviewer": {
                    "enabled": True,
                    "service_tier": "default",
                },
            },
            "synthesis_context": {"work_dir": str(tmp_path)},
        }
    )

    assert agent.model._service_tier == "flex"
    assert agent._reviewer._static_context["model_transport_kwargs"]["service_tier"] == "default"
    assert agent._synthesis_orchestrator.config.model_transport_kwargs["service_tier"] == "default"


def test_build_agent_keeps_artifact_guard_and_probes_harness_when_c_is_disabled(
    tmp_path, monkeypatch
):
    probe_calls = []
    monkeypatch.setattr(
        docker_app_runner,
        "run_harness_shape_probe",
        lambda **kwargs: probe_calls.append(kwargs),
    )
    agent = _build_agent(
        {
            "model": {
                "type": "LiteLLMModel",
                "model_id": "openai/gpt-5.4-mini-2026-03-17",
            },
            "agent_type": "ToolCallingAgent",
            "tools": [],
            "max_steps": 2,
            "cost_budget": {
                "enabled": True,
                "mode": "per_role",
                "roles": {
                    "a": {"max_total_cost_usd": 2.5},
                    "b": {"max_total_cost_usd": 1.0},
                },
            },
            "pocharness": {
                "enabled": True,
                "c_disabled": True,
                "log_dir": str(tmp_path),
                "reviewer": {"enabled": True},
            },
            "synthesis_context": {"work_dir": str(tmp_path)},
        }
    )

    assert set(agent.tools) == {"artifact_guard", "final_submission"}
    assert agent._synthesis_orchestrator is None
    assert agent._reviewer is not None
    assert agent._reviewer._cost_budget is agent._cost_budgets.ledger_for("b")
    assert probe_calls == [{"work_dir": str(tmp_path)}]
