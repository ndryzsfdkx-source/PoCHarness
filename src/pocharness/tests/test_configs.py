from pathlib import Path
import tomllib


CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

MODEL_PAIRS = (
    ("all300_solver_only_gpt55.toml", "all300_pocharness_gpt55.toml", "openai/gpt-5.5-2026-04-23"),
    ("all300_solver_only_gpt54mini.toml", "all300_pocharness_gpt54mini.toml", "openai/gpt-5.4-mini-2026-03-17"),
)


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("rb") as handle:
        return tomllib.load(handle)


def test_reported_pairs_share_model_dataset_and_solver_budget():
    for baseline_name, scaffold_name, model_id in MODEL_PAIRS:
        baseline = _load(baseline_name)
        scaffold = _load(scaffold_name)

        assert scaffold["model"] == baseline["model"]
        assert scaffold["model"]["model_id"] == model_id
        assert scaffold["dataset"] == baseline["dataset"]
        assert baseline["dataset"]["split"] == "eval"
        assert "instance_ids" not in baseline["dataset"]

        a_budget = scaffold["agent"]["cost_budget"]["roles"]["a"]["max_total_cost_usd"]
        assert a_budget == 2.5
        assert baseline["agent"]["cost_budget"]["max_total_cost_usd"] == a_budget


def test_reported_pairs_differ_only_in_scaffold_and_prompt():
    for baseline_name, scaffold_name, _model_id in MODEL_PAIRS:
        baseline = _load(baseline_name)
        scaffold = _load(scaffold_name)

        assert "pocharness" not in baseline["agent"]
        assert "pocharness" in scaffold["agent"]

        assert baseline["task"]["prompt_template"] == "poc-desc-floor.j2"
        assert scaffold["task"]["prompt_template"] == "poc-desc-pocharness.j2"
        assert scaffold["task"]["type"] == baseline["task"]["type"] == "poc-desc"


def test_pocharness_configs_wire_b_and_c():
    for name in (
        "all300_pocharness_gpt55.toml",
        "all300_pocharness_gpt54mini.toml",
    ):
        config = _load(name)
        pocharness = config["agent"]["pocharness"]
        assert pocharness["enabled"] is True
        assert pocharness["description_mode"] is True
        assert pocharness["reviewer"]["enabled"] is True
        assert "emit_finalization_verdict" in pocharness["reviewer"]["tools_allow"]
        assert "emit_helper" in pocharness["helper"]["tools_allow"]
