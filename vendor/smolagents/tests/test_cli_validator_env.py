"""Tests for runtime env propagation and task prompt helpers."""
from smolagents.cli import _apply_runtime_env, _create_agent_config_json, _create_task_prompt


class DummyRuntime:
    def __init__(self):
        self.environment = {}


def test_apply_runtime_env_sets_core_env_vars(monkeypatch):
    runtime = DummyRuntime()
    config = {
        "model": {"type": "LiteLLMModel", "model_id": "openai/gpt-5.4-mini-2026-03-17"},
        "agent": {},
    }

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _apply_runtime_env(runtime, config)

    assert runtime.environment["SMOLAGENTS_CONFIG_PATH"] == "/app/agent_config.json"
    assert runtime.environment["SMOLAGENTS_TASK_PATH"] == "/app/task.txt"
    assert runtime.environment["OPENAI_API_KEY"] == "test-key"


def test_create_agent_config_json_preserves_cost_budget_section():
    config = {
        "agent": {
            "cost_budget": {
                "enabled": True,
                "max_total_cost_usd": 10.0,
            }
        }
    }

    agent_config = _create_agent_config_json(config)

    assert agent_config["cost_budget"] == {
        "enabled": True,
        "max_total_cost_usd": 10.0,
    }


def test_create_agent_config_json_preserves_per_role_cost_budgets():
    budget = {
        "enabled": True,
        "mode": "per_role",
        "roles": {
            "a": {"max_total_cost_usd": 2.5},
            "b": {"max_total_cost_usd": 1.0},
            "c": {"max_total_cost_usd": 2.5},
        },
    }

    agent_config = _create_agent_config_json({"agent": {"cost_budget": budget}})

    assert agent_config["cost_budget"] == budget


