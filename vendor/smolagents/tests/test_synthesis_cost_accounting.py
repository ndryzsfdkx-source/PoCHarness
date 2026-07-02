import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from smolagents.docker_app_runner import _sum_jsonl_costs, _write_meta_json
from smolagents.secb.cost_budget import CostBudgetConfig, CostBudgetLedger, CostBudgetRegistry
from smolagents.secb.harness.agent import SynthesisBackendResult
from smolagents.secb.harness.config import SynthesisConfig
from smolagents.secb.review.log import write_finalization_review_log
from smolagents.secb.harness.synthesis import SynthesisOrchestrator


def test_finalization_review_log_records_b_token_fields_without_naive_cost(tmp_path: Path):
    write_finalization_review_log(
        log_dir=str(tmp_path),
        verdict="CONTINUE_LOCAL",
        reasoning="not exhausted",
        attached_action="Keep investigating the source-level gate.",
        evidence_steps=[3],
        b_steps_used=2,
        b_tool_calls=[{"step": 1, "tool": "read_synthesis_log"}],
        degraded=False,
        degraded_reason="",
        prompt_hash="abc123",
        rule_guard_decision={"decision": "allow"},
        remaining_steps=55,
        block_index=1,
        artifact_status="no sanitizer",
        stop_reason="evidence_exhaustion",
        instance_id="upx.ossfuzz-42531672",
        b_input_tokens=120,
        b_output_tokens=30,
    )

    record = json.loads((tmp_path / "finalization_review.jsonl").read_text(encoding="utf-8"))

    assert record["b_input_tokens"] == 120
    assert record["b_output_tokens"] == 30
    assert "b_cost" not in record


def test_finalization_review_log_records_ledger_cost_and_budget_reason(tmp_path: Path):
    write_finalization_review_log(
        log_dir=str(tmp_path),
        verdict="CONTINUE_LOCAL",
        reasoning="B allowance exhausted",
        attached_action="Continue locally.",
        evidence_steps=[],
        b_steps_used=3,
        b_tool_calls=[],
        degraded=True,
        degraded_reason="cost_budget_exhausted",
        prompt_hash="abc123",
        rule_guard_decision={},
        remaining_steps=30,
        block_index=1,
        artifact_status="partial",
        stop_reason="success",
        instance_id="example.cve",
        b_input_tokens=30,
        b_output_tokens=10,
        b_cost=0.4,
        b_termination_reason="cost_budget_exhausted",
    )

    record = json.loads((tmp_path / "finalization_review.jsonl").read_text(encoding="utf-8"))
    assert record["b_cost"] == 0.4
    assert record["b_termination_reason"] == "cost_budget_exhausted"
    assert record["degraded_reason"] == "cost_budget_exhausted"


def test_synthesis_log_records_c_token_fields_without_naive_cost(tmp_path: Path):
    orchestrator = SynthesisOrchestrator(SynthesisConfig(enabled=True, log_dir=str(tmp_path)))
    result = SynthesisBackendResult(
        parse_status="ok",
        emitted_payload={"helper_files": ["Helpers/mutate.py"]},
        c_steps_used=4,
        c_tool_calls=[{"step": 1, "tool": "write_helper"}],
        degraded=False,
        prompt="agent c prompt",
        raw_response="ok",
        c_input_tokens=300,
        c_output_tokens=80,
    )

    orchestrator._write_log(
        invocation_index=1,
        a_step=9,
        bundle={
            "delegated_problem": "blocked on parser gate",
            "candidate_testcase_path": "/testcase/poc",
            "repro_output": "no sanitizer",
            "target_signal": "AddressSanitizer heap-buffer-overflow in target",
            "source_hints": [],
            "attempts_summary": None,
            "continue_from": None,
            "continuation_mode": False,
        },
        result=result,
    )

    record = json.loads((tmp_path / "synthesis_log.jsonl").read_text(encoding="utf-8"))

    assert record["c_input_tokens"] == 300
    assert record["c_output_tokens"] == 80
    assert "c_cost" not in record


def test_synthesis_log_preserves_partial_files_and_ledger_budget_reason(tmp_path: Path):
    orchestrator = SynthesisOrchestrator(SynthesisConfig(enabled=True, log_dir=str(tmp_path)))
    result = SynthesisBackendResult(
        parse_status="failed",
        emitted_payload=None,
        c_steps_used=5,
        c_tool_calls=[{"step": 4, "tool": "write_helper"}],
        degraded=True,
        prompt="agent c prompt",
        degraded_reason="cost_budget_exhausted",
        partial_helper_files=["Helpers/partial.py"],
        c_input_tokens=50,
        c_output_tokens=20,
        c_cost=0.7,
        termination_reason="cost_budget_exhausted",
    )

    orchestrator._write_log(
        invocation_index=1,
        a_step=20,
        bundle={
            "delegated_problem": "construct parser seed",
            "candidate_testcase_path": "/testcase/poc",
            "repro_output": "no sanitizer",
            "target_signal": "target",
            "source_hints": [],
            "attempts_summary": None,
            "continue_from": None,
            "continuation_mode": False,
        },
        result=result,
    )

    record = json.loads((tmp_path / "synthesis_log.jsonl").read_text(encoding="utf-8"))
    assert record["c_cost"] == 0.7
    assert record["c_termination_reason"] == "cost_budget_exhausted"
    assert record["degraded_reason"] == "cost_budget_exhausted"
    assert record["partial_helper_files"] == ["Helpers/partial.py"]


def test_sum_jsonl_costs_skips_bad_rows(tmp_path: Path):
    log_path = tmp_path / "costs.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"cost": 1.5, "in": 100, "out": 20}),
                "{not json",
                json.dumps({"cost": 2.0, "in": 50, "out": 10}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert _sum_jsonl_costs(log_path, "cost", "in", "out") == (3.5, 150, 30)


def test_meta_json_includes_total_abc_costs(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(
            completion_cost=lambda response: (
                response["usage"]["prompt_tokens"] + response["usage"]["completion_tokens"]
            )
            / 100.0
        ),
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "finalization_review.jsonl").write_text(
        json.dumps({"b_cost": 0.5, "b_input_tokens": 100, "b_output_tokens": 25}) + "\n",
        encoding="utf-8",
    )
    (log_dir / "synthesis_log.jsonl").write_text(
        json.dumps({"c_cost": 1.25, "c_input_tokens": 300, "c_output_tokens": 75}) + "\n",
        encoding="utf-8",
    )
    step = SimpleNamespace(
        step_number=1,
        token_usage=SimpleNamespace(input_tokens=40, output_tokens=10),
    )
    agent = SimpleNamespace(
        model=SimpleNamespace(model_id="openai/gpt-5.4-2026-03-05"),
        memory=SimpleNamespace(steps=[step]),
        tools={},
        _synthesis_log_dir=str(log_dir),
    )

    _write_meta_json(str(tmp_path / "artifacts"), agent, SimpleNamespace())
    meta = json.loads((tmp_path / "artifacts" / "meta.json").read_text(encoding="utf-8"))

    assert meta["cost"] == 0.5
    assert meta["b_cost"] == 0.5
    assert meta["c_cost"] == 1.25
    assert meta["total_cost"] == 2.25
    assert meta["total_input_tokens"] == 440
    assert meta["total_output_tokens"] == 110


def test_meta_json_without_synthesis_keeps_legacy_cost_shape(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion_cost=lambda response: 0.1),
    )
    step = SimpleNamespace(
        step_number=1,
        token_usage=SimpleNamespace(input_tokens=4, output_tokens=2),
    )
    agent = SimpleNamespace(
        model=SimpleNamespace(model_id="openai/gpt-5.4-2026-03-05"),
        memory=SimpleNamespace(steps=[step]),
        tools={},
    )

    _write_meta_json(str(tmp_path / "artifacts"), agent, SimpleNamespace())
    meta = json.loads((tmp_path / "artifacts" / "meta.json").read_text(encoding="utf-8"))

    assert meta["cost"] == 0.1
    assert "b_cost" not in meta
    assert "c_cost" not in meta
    assert "total_cost" not in meta


def test_meta_json_uses_live_instance_budget_ledger(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(
            completion_cost=lambda completion_response=None, **kwargs: (
                completion_response["usage"]["prompt_tokens"]
                + completion_response["usage"]["completion_tokens"]
            )
            / 100.0
        ),
    )
    ledger = CostBudgetLedger(CostBudgetConfig(enabled=True, max_total_cost_usd=1.0))
    for role, input_tokens, output_tokens in (("a", 40, 10), ("b", 30, 10), ("c", 50, 20)):
        ledger.record_response(
            role=role,
            model_id="openai/gpt-5.4-2026-03-05",
            response=SimpleNamespace(
                token_usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
                raw=None,
            ),
            service_tier="flex",
        )

    step = SimpleNamespace(
        step_number=1,
        token_usage=SimpleNamespace(input_tokens=40, output_tokens=10),
    )
    agent = SimpleNamespace(
        model=SimpleNamespace(model_id="openai/gpt-5.4-2026-03-05"),
        memory=SimpleNamespace(steps=[step]),
        tools={},
        _cost_budget=ledger,
        _termination_reason="cost_budget_exhausted",
    )

    _write_meta_json(str(tmp_path / "artifacts"), agent, SimpleNamespace())
    meta = json.loads((tmp_path / "artifacts" / "meta.json").read_text(encoding="utf-8"))

    assert meta["cost"] == 0.5
    assert meta["b_cost"] == 0.4
    assert meta["c_cost"] == 0.7
    assert meta["total_cost"] == 1.6
    assert meta["total_input_tokens"] == 120
    assert meta["total_output_tokens"] == 40
    assert meta["cost_budget_limit_usd"] == 1.0
    assert meta["cost_budget_observed_usd"] == 1.6
    assert meta["cost_budget_overshoot_usd"] == pytest.approx(0.6)
    assert meta["cost_budget_status"] == "exhausted"
    assert meta["termination_reason"] == "cost_budget_exhausted"
    assert meta["model_calls_by_role"] == {"a": 1, "b": 1, "c": 1}


def test_live_budget_ledger_overrides_synthesis_jsonl_and_exposes_auxiliary_roles(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(
            completion_cost=lambda completion_response=None, **kwargs: (
                completion_response["usage"]["prompt_tokens"]
                + completion_response["usage"]["completion_tokens"]
            )
            / 100.0
        ),
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "finalization_review.jsonl").write_text(
        json.dumps({"b_cost": 99.0, "b_input_tokens": 9900, "b_output_tokens": 990}) + "\n",
        encoding="utf-8",
    )
    (log_dir / "synthesis_log.jsonl").write_text(
        json.dumps({"c_cost": 88.0, "c_input_tokens": 8800, "c_output_tokens": 880}) + "\n",
        encoding="utf-8",
    )
    ledger = CostBudgetLedger(CostBudgetConfig(enabled=True, max_total_cost_usd=100.0))
    role_usage = (
        ("a", 40, 10),
        ("b", 30, 10),
        ("c", 50, 20),
        ("p", 20, 5),
        ("managed:helper", 15, 5),
    )
    for role, input_tokens, output_tokens in role_usage:
        ledger.record_response(
            role=role,
            model_id="openai/gpt-5.4-2026-03-05",
            response=SimpleNamespace(
                token_usage=SimpleNamespace(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                raw=None,
            ),
        )
    agent = SimpleNamespace(
        model=SimpleNamespace(model_id="openai/gpt-5.4-2026-03-05"),
        memory=SimpleNamespace(steps=[]),
        tools={},
        _synthesis_log_dir=str(log_dir),
        _cost_budget=ledger,
    )

    _write_meta_json(str(tmp_path / "artifacts"), agent, SimpleNamespace())
    meta = json.loads((tmp_path / "artifacts" / "meta.json").read_text(encoding="utf-8"))

    assert meta["b_cost"] == 0.4
    assert meta["c_cost"] == 0.7
    assert meta["p_cost"] == 0.25
    assert meta["managed_cost"] == 0.2
    assert meta["total_cost"] == pytest.approx(2.05)
    assert (
        meta["cost"]
        + meta["b_cost"]
        + meta["c_cost"]
        + meta["p_cost"]
        + meta["managed_cost"]
    ) == pytest.approx(meta["total_cost"])


def test_meta_json_reports_separated_role_budgets_and_exact_overhead(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(
            completion_cost=lambda completion_response=None, **kwargs: (
                completion_response["usage"]["prompt_tokens"]
                + completion_response["usage"]["completion_tokens"]
            )
            / 100.0
        ),
    )
    registry = CostBudgetRegistry.from_config(
        {
            "enabled": True,
            "mode": "per_role",
            "roles": {
                "a": {"max_total_cost_usd": 2.5},
                "b": {"max_total_cost_usd": 1.0},
                "c": {"max_total_cost_usd": 2.5},
            },
        }
    )
    for role, input_tokens, output_tokens in (("a", 40, 10), ("b", 30, 10), ("c", 50, 20)):
        registry.ledger_for(role).record_response(
            role=role,
            model_id="openai/gpt-5.4-2026-03-05",
            response=SimpleNamespace(
                token_usage=SimpleNamespace(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                raw=None,
            ),
        )
    agent = SimpleNamespace(
        model=SimpleNamespace(model_id="openai/gpt-5.4-2026-03-05"),
        memory=SimpleNamespace(steps=[]),
        tools={},
        _cost_budgets=registry,
        _cost_budget=registry.ledger_for("a"),
    )

    _write_meta_json(str(tmp_path / "artifacts"), agent, SimpleNamespace())
    meta = json.loads((tmp_path / "artifacts" / "meta.json").read_text(encoding="utf-8"))

    assert meta["cost_budget_mode"] == "per_role"
    assert meta["cost"] == 0.5
    assert meta["b_cost"] == 0.4
    assert meta["c_cost"] == 0.7
    assert meta["overhead_cost"] == pytest.approx(1.1)
    assert meta["total_cost"] == pytest.approx(1.6)
    assert meta["cost_budget_limit_usd"] == 6.0
    assert meta["cost_by_role"] == {"a": 0.5, "b": 0.4, "c": 0.7}
    assert meta["cost_budget_by_role"]["a"]["limit_usd"] == 2.5
    assert meta["cost_budget_by_role"]["b"]["limit_usd"] == 1.0
    assert meta["cost_budget_by_role"]["c"]["limit_usd"] == 2.5
