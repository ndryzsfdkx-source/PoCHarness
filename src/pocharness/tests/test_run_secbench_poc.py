from __future__ import annotations

import base64
import importlib.util
from io import BytesIO
from pathlib import Path
import sys
import tarfile
import tomllib

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_secbench_poc.py"
SPEC = importlib.util.spec_from_file_location("run_secbench_poc", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_secbench_poc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_secbench_poc
SPEC.loader.exec_module(run_secbench_poc)


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[dataset]
split = "eval"

[output]
output_dir = "{tmp_path / 'runs' / 'demo'}"

[task]
type = "poc-san"
""".strip()
        + "\n"
    )
    return config_path


def make_context(tmp_path: Path) -> run_secbench_poc.RunContext:
    return run_secbench_poc.RunContext(
        repo_root=tmp_path,
        secbench_root=tmp_path / "vendor" / "sec-bench-evaluator",
        local_smolagents_repo=tmp_path / "vendor" / "smolagents",
        local_smolagents_src=tmp_path / "vendor" / "smolagents" / "src",
        repo_python=Path(sys.executable),
    )


def test_default_main_runs_generate_then_eval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    fake_run_dir = tmp_path / "runs" / "demo" / "20260409_010203"
    fake_eval_dir = tmp_path / "runs" / "evals" / "demo_20260409_010203"
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(run_secbench_poc, "get_run_context", lambda: make_context(tmp_path))

    def fake_generate_stage(**kwargs):
        calls.append(("generate", kwargs["base_output_dir"]))
        return fake_run_dir

    def fake_eval_stage(**kwargs):
        calls.append(("eval", kwargs["run_dir"]))
        return fake_eval_dir

    monkeypatch.setattr(run_secbench_poc, "run_generation_stage", fake_generate_stage)
    monkeypatch.setattr(run_secbench_poc, "run_eval_stage", fake_eval_stage)
    monkeypatch.setattr(run_secbench_poc, "run_analysis_stage", lambda **kwargs: pytest.fail("analysis should not run"))

    fake_run_dir.mkdir(parents=True)
    assert run_secbench_poc.main(["--config", str(config_path)]) == 0
    assert calls == [
        ("generate", tmp_path / "runs" / "demo"),
        ("eval", fake_run_dir),
    ]


def test_skip_eval_keeps_generation_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    calls: list[str] = []
    fake_run_dir = tmp_path / "runs" / "demo" / "20260409_010203"

    monkeypatch.setattr(run_secbench_poc, "get_run_context", lambda: make_context(tmp_path))
    monkeypatch.setattr(
        run_secbench_poc,
        "run_generation_stage",
        lambda **kwargs: calls.append("generate") or fake_run_dir,
    )
    monkeypatch.setattr(run_secbench_poc, "run_eval_stage", lambda **kwargs: pytest.fail("eval should not run"))

    fake_run_dir.mkdir(parents=True)
    assert run_secbench_poc.main(["--config", str(config_path), "--skip-eval"]) == 0
    assert calls == ["generate"]


def test_analyze_flag_appends_analysis(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    fake_run_dir = tmp_path / "runs" / "demo" / "20260409_010203"
    fake_eval_dir = tmp_path / "runs" / "evals" / "demo_20260409_010203"
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(run_secbench_poc, "get_run_context", lambda: make_context(tmp_path))
    monkeypatch.setattr(run_secbench_poc, "run_generation_stage", lambda **kwargs: fake_run_dir)
    monkeypatch.setattr(run_secbench_poc, "run_eval_stage", lambda **kwargs: fake_eval_dir)

    def fake_analysis_stage(**kwargs):
        calls.append(("analyze", kwargs["eval_dir"]))
        return fake_eval_dir / "analysis.md"

    monkeypatch.setattr(run_secbench_poc, "run_analysis_stage", fake_analysis_stage)

    fake_run_dir.mkdir(parents=True)
    assert run_secbench_poc.main(["--config", str(config_path), "--analyze"]) == 0
    assert calls == [("analyze", fake_eval_dir)]


def test_eval_stage_requires_run_dir_without_generation(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)

    with pytest.raises(SystemExit):
        run_secbench_poc.parse_args(["--config", str(config_path), "--stages", "eval"])


def test_analyze_only_requires_eval_dir() -> None:
    with pytest.raises(SystemExit):
        run_secbench_poc.parse_args(["--stages", "analyze"])


def test_eval_then_analyze_uses_existing_run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    run_dir = tmp_path / "runs" / "demo" / "20260409_010203"
    run_dir.mkdir(parents=True)
    (run_dir / "output.jsonl").write_text("{}\n")
    fake_eval_dir = tmp_path / "runs" / "evals" / "demo_20260409_010203"
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(run_secbench_poc, "get_run_context", lambda: make_context(tmp_path))

    def fake_eval_stage(**kwargs):
        calls.append(("eval", kwargs["run_dir"]))
        return fake_eval_dir

    def fake_analysis_stage(**kwargs):
        calls.append(("analyze", kwargs["eval_dir"]))
        return fake_eval_dir / "analysis.md"

    monkeypatch.setattr(run_secbench_poc, "run_eval_stage", fake_eval_stage)
    monkeypatch.setattr(run_secbench_poc, "run_analysis_stage", fake_analysis_stage)

    assert run_secbench_poc.main(
        [
            "--config",
            str(config_path),
            "--stages",
            "eval",
            "analyze",
            "--run-dir",
            str(run_dir),
        ]
    ) == 0
    assert calls == [
        ("eval", run_dir.resolve()),
        ("analyze", fake_eval_dir),
    ]


def test_analyze_only_uses_existing_eval_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    eval_dir = tmp_path / "runs" / "evals" / "demo_20260409_010203"
    eval_dir.mkdir(parents=True)
    calls: list[Path] = []

    monkeypatch.setattr(run_secbench_poc, "get_run_context", lambda: make_context(tmp_path))
    monkeypatch.setattr(
        run_secbench_poc,
        "run_analysis_stage",
        lambda **kwargs: calls.append(kwargs["eval_dir"]) or (kwargs["eval_dir"] / "analysis.md"),
    )

    assert run_secbench_poc.main(["--stages", "analyze", "--eval-dir", str(eval_dir)]) == 0
    assert calls == [eval_dir.resolve()]


def test_eval_output_dir_nests_under_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "model_a" / "exp1" / "20260409_010203"

    eval_dir = run_secbench_poc.build_eval_output_dir(run_dir)

    assert eval_dir.parent == run_dir / "eval"


def test_stage_eval_input_dir_rehydrates_testcase_with_readable_permissions(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo" / "20260409_010203"
    instance_dir = run_dir / "demo.instance"
    eval_dir = run_dir / "eval" / "20260409_010204"

    instance_dir.mkdir(parents=True)
    (run_dir / "output.jsonl").write_text('{"instance_id":"demo.instance","test_result":{"poc_artifact":"stub"}}\n')

    archive_buffer = BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        payload = b"artifact-bytes"

        file_info = tarfile.TarInfo("./nested/poc.sh")
        file_info.size = len(payload)
        file_info.mode = 0o700
        archive.addfile(file_info, BytesIO(payload))

        unreadable_info = tarfile.TarInfo("./output.cache")
        unreadable_info.size = 0
        unreadable_info.mode = 0o600
        archive.addfile(unreadable_info, BytesIO(b""))

    (instance_dir / "poc_artifact.txt").write_text(base64.b64encode(archive_buffer.getvalue()).decode())

    staged_dir = run_secbench_poc.stage_eval_input_dir(run_dir, eval_dir)

    assert staged_dir == eval_dir
    assert (eval_dir / "output.jsonl").read_text() == (run_dir / "output.jsonl").read_text()
    # Testcase is rehydrated in place under the run dir's instance dir, not duplicated into eval_dir.
    assert (instance_dir / "testcase" / "nested" / "poc.sh").read_bytes() == b"artifact-bytes"
    assert (instance_dir / "testcase" / "nested" / "poc.sh").stat().st_mode & 0o777 == 0o755
    assert (instance_dir / "testcase" / "output.cache").stat().st_mode & 0o777 == 0o644


def test_run_eval_stage_threads_graders_to_eval_instances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = make_context(tmp_path)
    run_dir = tmp_path / "runs" / "demo" / "20260409_010203"
    run_dir.mkdir(parents=True)
    (run_dir / "output.jsonl").write_text("{}\n")
    commands: list[list[str]] = []

    def fake_stage_eval_input_dir(source: Path, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "output.jsonl").write_text("{}\n")
        return destination

    monkeypatch.setattr(run_secbench_poc, "stage_eval_input_dir", fake_stage_eval_input_dir)
    monkeypatch.setattr(run_secbench_poc, "run_command", lambda command, cwd, env=None: commands.append(command))

    eval_dir = run_secbench_poc.run_eval_stage(
        context=context,
        config={"dataset": {"split": "eval"}},
        run_dir=run_dir,
        graders="semantic,strict",
        grader_line_tolerance=7,
    )

    assert eval_dir.parent == run_dir / "eval"
    assert commands == [[
        sys.executable,
        "-m",
        "secb.evaluator.eval_instances",
        "--input-dir",
        str(eval_dir),
        "--type",
        "poc",
        "--split",
        "eval",
        "--agent",
        "smolagent",
        "--graders",
        "semantic,strict",
        "--grader-line-tolerance",
        "7",
    ]]
