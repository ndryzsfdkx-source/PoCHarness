from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SRC_DIR / "analyze_run.py"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SPEC = importlib.util.spec_from_file_location("analyze_run", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
analyze_run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyze_run
SPEC.loader.exec_module(analyze_run)


def test_parse_step_preserves_string_final_answer_arguments() -> None:
    step = analyze_run._parse_step(
        {
            "step_number": 3,
            "tool_calls": [
                {
                    "function": {
                        "name": "final_answer",
                        "arguments": "I’m sorry, but I can’t reliably produce a working PoC.",
                    }
                }
            ],
            "is_final_answer": True,
        }
    )

    assert step is not None
    assert step.tool_name == "final_answer"
    assert step.tool_args == "I’m sorry, but I can’t reliably produce a working PoC."
    assert step.all_tool_calls is None


def test_render_step_includes_string_final_answer_text() -> None:
    step = analyze_run.Step(
        step_number=7,
        tool_name="final_answer",
        tool_args="I’m sorry, but I can’t reliably produce a working PoC.",
        observations=None,
        error=None,
        timing=None,
        token_usage=None,
        is_final_answer=True,
        all_tool_calls=None,
    )

    rendered = analyze_run.render_step(step, max_chars=400)

    assert "final_answer" in rendered
    assert "can’t reliably produce a working PoC" in rendered


def _instance(instance_id: str, meta: dict, *, success: bool = False):
    return analyze_run.InstanceData(
        instance_id=instance_id,
        meta=meta,
        steps=[],
        task_prompt="",
        eval_result={"success": success, "reason": ""},
        eval_results_by_grader={},
        testcase_files=[],
    )


def test_render_eval_report_prefers_total_abc_cost_and_tokens() -> None:
    run = analyze_run.EvalRun(
        path=Path("/tmp/evals/demo"),
        name="demo",
        config=None,
        instances=[
            _instance(
                "demo.one",
                {
                    "steps": 1,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cost": 1.0,
                    "b_cost": 0.5,
                    "c_cost": 1.5,
                    "total_input_tokens": 100,
                    "total_output_tokens": 50,
                    "total_cost": 3.0,
                },
                success=True,
            ),
            _instance(
                "demo.two",
                {
                    "steps": 2,
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "cost": 0.4,
                    "total_input_tokens": 40,
                    "total_output_tokens": 20,
                    "total_cost": 1.0,
                },
            ),
        ],
    )

    rendered = analyze_run.render_eval_report(
        run,
        instance_filter=None,
        max_chars=400,
        show_trajectory=False,
    )

    assert "| Total tokens | 140 in / 70 out |" in rendered
    assert "| Total cost | $4.00 |" in rendered
    assert "| demo.one | PASS | 1 | 100 / 50 | $3.00 |" in rendered
    assert "A/B/C cost: A=$1.00 | B=$0.50 | C=$1.50 | Total=$3.00" in rendered


def test_render_eval_report_keeps_legacy_a_only_meta_fallback() -> None:
    run = analyze_run.EvalRun(
        path=Path("/tmp/evals/legacy"),
        name="legacy",
        config=None,
        instances=[
            _instance(
                "legacy.one",
                {"steps": 1, "input_tokens": 10, "output_tokens": 5, "cost": 0.25},
            )
        ],
    )

    rendered = analyze_run.render_eval_report(
        run,
        instance_filter=None,
        max_chars=400,
        show_trajectory=False,
    )

    assert "| Total tokens | 10 in / 5 out |" in rendered
    assert "| Total cost | $0.25 |" in rendered
    assert "| legacy.one | FAIL | 1 | 10 / 5 | $0.25 |" in rendered
    assert "A/B/C cost:" not in rendered


def test_render_eval_report_surfaces_instance_budget_termination() -> None:
    run = analyze_run.EvalRun(
        path=Path("/tmp/evals/budget"),
        name="budget",
        config={
            "model": {"model_id": "openai/gpt-5.4-2026-03-05"},
            "task": {"type": "poc-san", "prompt_template": "poc-san.j2"},
            "agent": {
                "tools": ["cmd"],
                "max_steps": 90,
                "cost_budget": {"enabled": True},
            },
        },
        instances=[
            _instance(
                "budget.one",
                {
                    "steps": 3,
                    "total_cost": 10.25,
                    "cost_budget_enabled": True,
                    "cost_budget_limit_usd": 10.0,
                    "cost_budget_observed_usd": 10.25,
                    "cost_budget_overshoot_usd": 0.25,
                    "cost_budget_status": "exhausted",
                    "termination_reason": "cost_budget_exhausted",
                },
            )
        ],
    )

    rendered = analyze_run.render_eval_report(
        run,
        instance_filter=None,
        max_chars=400,
        show_trajectory=False,
    )

    assert "- Instance cost cap: $10.00" in rendered
    assert "| Budget-terminated | 1 |" in rendered
    assert "status=exhausted | limit=$10.00 | observed=$10.25 | overshoot=$0.25" in rendered
    assert "- Termination reason: `cost_budget_exhausted`" in rendered


def test_render_eval_report_surfaces_per_role_caps_costs_and_overhead() -> None:
    budgets = {
        "a": {"limit_usd": 2.5, "observed_usd": 0.5, "overshoot_usd": 0.0, "status": "within_budget"},
        "b": {"limit_usd": 1.0, "observed_usd": 0.4, "overshoot_usd": 0.0, "status": "within_budget"},
        "c": {"limit_usd": 2.5, "observed_usd": 0.7, "overshoot_usd": 0.0, "status": "within_budget"},
    }
    run = analyze_run.EvalRun(
        path=Path("/tmp/evals/per-role"),
        name="per-role",
        config={
            "model": {"model_id": "openai/gpt-5.5-2026-04-23"},
            "task": {"type": "poc-desc", "prompt_template": "poc-desc-pocharness.j2"},
            "agent": {
                "tools": ["cmd"],
                "max_steps": 75,
                "cost_budget": {
                    "enabled": True,
                    "mode": "per_role",
                    "roles": {
                        "a": {"max_total_cost_usd": 2.5},
                        "b": {"max_total_cost_usd": 1.0},
                        "c": {"max_total_cost_usd": 2.5},
                    },
                },
            },
        },
        instances=[
            _instance(
                "per-role.one",
                {
                    "steps": 2,
                    "cost": 0.5,
                    "b_cost": 0.4,
                    "c_cost": 0.7,
                    "overhead_cost": 1.1,
                    "total_cost": 1.6,
                    "cost_by_role": {"a": 0.5, "b": 0.4, "c": 0.7},
                    "cost_budget_enabled": True,
                    "cost_budget_mode": "per_role",
                    "cost_budget_limit_usd": 6.0,
                    "cost_budget_observed_usd": 1.6,
                    "cost_budget_overshoot_usd": 0.0,
                    "cost_budget_status": "within_budget",
                    "cost_budget_by_role": budgets,
                },
            )
        ],
    )

    rendered = analyze_run.render_eval_report(
        run,
        instance_filter=None,
        max_chars=400,
        show_trajectory=False,
    )

    assert "Per-role instance cost caps: A=$2.50 | B=$1.00 | C=$2.50" in rendered
    assert "| A cost | $0.50 |" in rendered
    assert "| B cost | $0.40 |" in rendered
    assert "| C cost | $0.70 |" in rendered
    assert "| B+C overhead cost | $1.10 |" in rendered
    assert "- A budget: status=within_budget | limit=$2.50 | observed=$0.50" in rendered


def test_render_instance_uses_complete_role_cost_breakdown() -> None:
    rendered = analyze_run.render_instance(
        _instance(
            "budget.roles",
            {
                "steps": 2,
                "cost": 1.0,
                "b_cost": 0.5,
                "c_cost": 0.75,
                "total_cost": 2.5,
                "cost_by_role": {
                    "a": 1.0,
                    "advisor": 0.1,
                    "b": 0.5,
                    "c": 0.75,
                    "controller": 0.15,
                },
            },
        ),
        max_chars=400,
        show_trajectory=False,
    )

    assert "Role costs: a=$1.00 | advisor=$0.10 | b=$0.50 | c=$0.75 | controller=$0.15" in rendered
    assert "Total=$2.50" in rendered
    assert "A/B/C cost:" not in rendered


def test_independent_synthesis_audit_flags_external_retrieval() -> None:
    steps = [
        analyze_run.Step(
            step_number=31,
            tool_name="cmd",
            tool_args={"command": "curl -L https://github.com/example/repo/poc -o /tmp/poc"},
            observations="downloaded",
            error=None,
            timing=None,
            token_usage=None,
            is_final_answer=False,
        )
    ]
    audit = analyze_run.audit_independent_synthesis(steps)
    assert audit["status"] == "flagged"
    assert audit["evidence"][0]["step"] == 31
    assert audit["evidence"][0]["signal"] in {"curl", "https://"}


def test_independent_synthesis_audit_allows_local_history() -> None:
    steps = [
        analyze_run.Step(
            step_number=4,
            tool_name="cmd",
            tool_args={"command": "git log --oneline -- parser.c && git show HEAD~1:parser.c"},
            observations="local history",
            error=None,
            timing=None,
            token_usage=None,
            is_final_answer=False,
        )
    ]
    assert analyze_run.audit_independent_synthesis(steps)["status"] == "clean"


def test_c_event_detail_reads_v2_and_v1_records() -> None:
    inst = _instance("example", {})
    v2 = analyze_run._c_event_detail(
        inst,
        {
            "invocation_index": 2,
            "input": {"delegated_problem": "reconstruct record order"},
            "emitted_payload": {"validation": "semantic"},
        },
    )
    v1 = analyze_run._c_event_detail(
        inst,
        {
            "invocation_index": 1,
            "input": {"blocker_summary": "legacy blocker"},
            "emitted_payload": {"validation_outcome": "did_not_run"},
        },
    )
    assert v2["delegated_problem"] == "reconstruct record order"
    assert v2["validation_outcome"] == "semantic"
    assert v1["delegated_problem"] == "legacy blocker"
    assert v1["validation_outcome"] == "did_not_run"
