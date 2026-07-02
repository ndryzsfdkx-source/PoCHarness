#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from io import BytesIO
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


STAGE_ORDER = ("generate", "eval", "analyze")
SITE_ENV_VAR = "SECB_SITE"
VALID_TIERS = ("main", "smoke")
MANIFEST_NAME = "manifest.json"
CONFIG_SNAPSHOT_NAME = "config_snapshot.toml"


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    secbench_root: Path
    local_smolagents_repo: Path
    local_smolagents_src: Path
    repo_python: Path


def run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True, env=env)


def with_local_smolagents_pythonpath(context: RunContext, env: dict[str, str] | None = None) -> dict[str, str]:
    run_env = (env or os.environ).copy()
    if context.local_smolagents_repo.exists() and context.local_smolagents_src.exists():
        existing_pythonpath = run_env.get("PYTHONPATH", "")
        run_env["PYTHONPATH"] = (
            f"{context.local_smolagents_src}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(context.local_smolagents_src)
        )
    if sys.platform == "darwin":
        hf_root = Path("/tmp/secb-hf")
        datasets_cache = hf_root / "datasets"
        hub_cache = hf_root / "hub"
        eval_home = Path("/tmp/secb-eval-home")
        for path in (hf_root, datasets_cache, hub_cache, eval_home):
            path.mkdir(parents=True, exist_ok=True)
        run_env["HF_HOME"] = str(hf_root)
        run_env["HF_DATASETS_CACHE"] = str(datasets_cache)
        run_env["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
        run_env["HOME"] = str(eval_home)
    return run_env


def resolve_latest_run_dir(base_output_dir: Path, before_dirs: set[Path]) -> Path:
    after_dirs = {path.resolve() for path in base_output_dir.iterdir() if path.is_dir()}
    new_dirs = sorted(after_dirs - before_dirs, key=lambda path: path.stat().st_mtime)
    if new_dirs:
        return new_dirs[-1]
    if after_dirs:
        return max(after_dirs, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(f"No run directories found under {base_output_dir}")


def resolve_site() -> str:
    """A short tag identifying the machine a run came from, used to disambiguate
    run directories when reproducing across multiple machines. Defaults to
    "local"; set SECB_SITE to something else if you run on more than one host."""
    return os.environ.get(SITE_ENV_VAR, "local").strip() or "local"


def resolve_tier(config: dict, smoke_override: bool) -> str:
    if smoke_override:
        return "smoke"
    tier = config.get("output", {}).get("tier", "main")
    if tier not in VALID_TIERS:
        raise SystemExit(f"Invalid [output].tier {tier!r}; expected one of {VALID_TIERS}.")
    return tier


def git_snapshot(repo_root: Path) -> dict:
    def _git(*git_args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *git_args], cwd=repo_root, check=True, capture_output=True, text=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


def manifest_path_for_run(run_dir: Path) -> Path:
    return run_dir / MANIFEST_NAME


def write_generation_manifest(
    *,
    run_dir: Path,
    site: str,
    tier: str,
    config_path: Path,
    config: dict,
    repo_root: Path,
    instance_id: str | None,
    num_workers: int,
) -> None:
    snapshot_path = run_dir / CONFIG_SNAPSHOT_NAME
    shutil.copy2(config_path, snapshot_path)
    manifest = {
        "schema_version": 1,
        "experiment": config_path.stem,
        "run_id": run_dir.name,
        "site": site,
        "tier": tier,
        "config_name": config_path.name,
        "config_snapshot": CONFIG_SNAPSHOT_NAME,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_id": config.get("model", {}).get("model_id"),
        "task_type": config.get("task", {}).get("type"),
        "dataset": config.get("dataset", {}),
        "instance_filter": instance_id,
        "num_workers": num_workers,
        **git_snapshot(repo_root),
        "stages": {
            "generate": {"completed_at": datetime.now().astimezone().isoformat(timespec="seconds")},
        },
    }
    manifest_path_for_run(run_dir).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest written: {manifest_path_for_run(run_dir)}")


def record_manifest_stage(run_dir: Path, stage: str, entry: dict) -> None:
    """Append a stage record to the run manifest, if one exists (legacy runs may lack it)."""
    manifest_path = manifest_path_for_run(run_dir)
    if not manifest_path.exists():
        print(f"No {MANIFEST_NAME} in {run_dir}; skipping manifest update (legacy run).")
        return
    manifest = json.loads(manifest_path.read_text())
    stages = manifest.setdefault("stages", {})
    stages.setdefault(stage, []).append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def build_eval_output_dir(run_dir: Path) -> Path:
    """Nested evaluation directory inside the run dir; re-evals stack as siblings."""
    evals_root = run_dir / "eval"
    candidate = evals_root / datetime.now().strftime("%Y%m%d_%H%M")
    if candidate.exists():
        candidate = evals_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    return candidate


SECBENCH_EVAL_ROOT_ENV_VAR = "SECBENCH_EVAL_ROOT"


def get_run_context() -> RunContext:
    repo_root = Path(__file__).resolve().parents[2]
    # secb.evaluator is vendored (extended with our four-grader PoC oracle) at
    # vendor/sec-bench-evaluator/; SECBENCH_EVAL_ROOT overrides for a different checkout.
    secbench_root = Path(
        os.environ.get(SECBENCH_EVAL_ROOT_ENV_VAR, str(repo_root / "vendor" / "sec-bench-evaluator"))
    ).expanduser()
    return RunContext(
        repo_root=repo_root,
        secbench_root=secbench_root,
        local_smolagents_repo=repo_root / "vendor" / "smolagents",
        local_smolagents_src=repo_root / "vendor" / "smolagents" / "src",
        repo_python=repo_root / ".conda" / "bin" / "python",
    )


def resolve_runner_python(context: RunContext) -> str:
    if context.repo_python.exists():
        return str(context.repo_python)
    return sys.executable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SEC-bench PoC generation, evaluation, and analysis.")
    parser.add_argument("--config", help="Path to the local TOML config.")
    parser.add_argument("--instance-id", help="Optional single SEC-bench instance to run.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of smolagent workers to use.")
    parser.add_argument(
        "--output-dir",
        help="Optional local output directory override for generation. Evaluation nests inside the run dir.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Force the smoke tier for this run (overrides [output].tier; output under runs/smoke/).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        help="Pipeline stages to run. Default: generate eval.",
    )
    parser.add_argument(
        "--run-dir",
        help="Existing timestamped generation run directory to evaluate when `generate` is not selected.",
    )
    parser.add_argument(
        "--eval-dir",
        help="Existing evaluation directory to analyze when `eval` is not selected.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Backward-compatible alias for `--stages generate`.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Append the analysis stage after evaluation.",
    )
    parser.add_argument(
        "--graders",
        default="loose,caller,semantic,strict",
        help=(
            "Comma-separated PoC graders to emit during evaluation. "
            "Default: loose,caller,semantic,strict."
        ),
    )
    parser.add_argument(
        "--grader-line-tolerance",
        type=int,
        default=10,
        help="Maximum allowed line drift for structured PoC graders (default: 10).",
    )
    parser.add_argument(
        "--analysis-grader",
        choices=["loose", "caller", "semantic", "strict"],
        help="Primary grader view to use when generating analysis.md (default: loose if available).",
    )
    args = parser.parse_args(argv)

    if args.skip_eval and args.stages:
        parser.error("`--skip-eval` cannot be combined with `--stages`.")

    requested_stages = list(args.stages) if args.stages else ["generate", "eval"]
    if args.skip_eval:
        requested_stages = ["generate"]
    if args.analyze and "analyze" not in requested_stages:
        requested_stages.append("analyze")
    args.stages = [stage for stage in STAGE_ORDER if stage in requested_stages]

    requires_config = "generate" in args.stages or "eval" in args.stages
    if requires_config and not args.config:
        parser.error("`--config` is required when running generate or eval stages.")
    if not requires_config and not args.eval_dir:
        parser.error("`--eval-dir` is required for analyze-only runs.")

    if "generate" not in args.stages:
        if args.instance_id:
            parser.error("`--instance-id` only applies to the generate stage.")
        if args.num_workers != 1:
            parser.error("`--num-workers` only applies to the generate stage.")
        if args.output_dir:
            parser.error("`--output-dir` only applies to the generate stage.")
        if args.smoke:
            parser.error("`--smoke` only applies to the generate stage.")
        if "eval" in args.stages and not args.run_dir:
            parser.error("`--run-dir` is required when running eval without generate.")
    elif args.run_dir:
        parser.error("`--run-dir` cannot be used when generate is selected.")

    if "analyze" in args.stages and "eval" not in args.stages:
        if "generate" in args.stages:
            parser.error("Analyze requires eval output. Use `--stages generate eval analyze` or analyze-only with `--eval-dir`.")
    elif args.eval_dir:
        parser.error("`--eval-dir` can only be used for analyze-only runs.")

    return args


def resolve_config_path(config_arg: str, repo_root: Path) -> Path:
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    return config_path


def load_config(config_path: Path) -> dict:
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    task_type = config.get("task", {}).get("type", "")
    if not task_type.startswith("poc"):
        raise ValueError(f"Expected a PoC task config, got: {task_type!r}")
    return config


def resolve_generation_output_root(
    config: dict,
    output_dir_override: str | None,
    *,
    tier: str,
    config_path: Path,
    repo_root: Path,
) -> Path:
    """Default: runs/<tier>/<config-stem>/. `[output].output_dir` and --output-dir override."""
    if output_dir_override:
        return Path(output_dir_override).resolve()
    configured_output_dir = config.get("output", {}).get("output_dir")
    if configured_output_dir:
        configured = Path(configured_output_dir)
        return configured.resolve() if configured.is_absolute() else (repo_root / configured).resolve()
    return repo_root / "runs" / tier / config_path.stem


def validate_run_dir(path_str: str) -> Path:
    run_dir = Path(path_str).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not (run_dir / "output.jsonl").exists():
        raise FileNotFoundError(f"Run directory does not contain output.jsonl: {run_dir}")
    return run_dir


def validate_eval_dir(path_str: str) -> Path:
    eval_dir = Path(path_str).resolve()
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")
    return eval_dir


def _normalize_tar_member_path(name: str) -> Path | None:
    parts = [part for part in Path(name).parts if part not in ("", ".")]
    if not parts:
        return None
    if any(part == ".." for part in parts):
        raise ValueError(f"Unsafe path in PoC artifact: {name}")
    return Path(*parts)


def _extract_poc_artifact_to_testcase(artifact_b64: str, testcase_dir: Path) -> None:
    testcase_dir.mkdir(parents=True, exist_ok=True)
    testcase_dir.chmod(0o755)
    if not artifact_b64:
        return

    try:
        archive_bytes = base64.b64decode(artifact_b64)
    except ValueError as exc:
        raise ValueError("Invalid base64 PoC artifact") from exc

    with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            rel_path = _normalize_tar_member_path(member.name)
            if rel_path is None:
                continue

            target_path = testcase_dir / rel_path
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                target_path.chmod(0o755)
                continue

            if not member.isfile():
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                continue

            with extracted, target_path.open("wb") as destination:
                shutil.copyfileobj(extracted, destination)

            target_path.chmod(0o755 if member.mode & 0o111 else 0o644)


def stage_eval_input_dir(run_dir: Path, eval_output_dir: Path) -> Path:
    """Prepare the nested eval dir: stage output.jsonl for the evaluator (it reads the
    PoC base64 from there) and extract per-instance testcase/ trees into the run dir
    for the aggregate analyzers. No other duplication of the run tree."""
    eval_output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run_dir / "output.jsonl", eval_output_dir / "output.jsonl")

    for item in sorted(run_dir.iterdir()):
        if not item.is_dir() or item.name == "eval":
            continue
        artifact_file = item / "poc_artifact.txt"
        if artifact_file.exists() and not (item / "testcase").exists():
            _extract_poc_artifact_to_testcase(artifact_file.read_text().strip(), item / "testcase")

    return eval_output_dir


def run_generation_stage(
    *,
    context: RunContext,
    config_path: Path,
    base_output_dir: Path,
    instance_id: str | None,
    num_workers: int,
    output_dir_override: str | None,
    site: str,
) -> Path:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    before_dirs = {path.resolve() for path in base_output_dir.iterdir() if path.is_dir()}

    run_env = with_local_smolagents_pythonpath(context)
    if context.local_smolagents_repo.exists() and context.local_smolagents_src.exists():
        command = [
            resolve_runner_python(context),
            "-m",
            "smolagents.cli",
            "secb-run",
            "--config",
            str(config_path),
            "--num-workers",
            str(num_workers),
        ]
    else:
        command = [
            "smolagent",
            "secb-run",
            "--config",
            str(config_path),
            "--num-workers",
            str(num_workers),
        ]

    if instance_id:
        command.extend(["--instance-id", instance_id])
    # Always pass the resolved output root: the tier/site-aware default lives in
    # this orchestrator, not in the vendor CLI (whose own fallback is ./secb_results).
    command.extend(["--output-dir", str(base_output_dir)])

    run_command(command, cwd=context.repo_root, env=run_env)
    run_dir = resolve_latest_run_dir(base_output_dir, before_dirs)
    # The vendor CLI creates a bare-timestamp dir; suffix the site tag here so the
    # vendor stays untouched (local-vs-upstream rule).
    if not run_dir.name.endswith(f"_{site}"):
        sited_dir = run_dir.with_name(f"{run_dir.name}_{site}")
        run_dir.rename(sited_dir)
        run_dir = sited_dir
    print(f"Run output: {run_dir}")
    return run_dir


def run_eval_stage(
    *,
    context: RunContext,
    config: dict,
    run_dir: Path,
    graders: str = "loose,caller,semantic,strict",
    grader_line_tolerance: int = 10,
) -> Path:
    dataset_split = config.get("dataset", {}).get("split", "eval")
    eval_output_dir = build_eval_output_dir(run_dir)
    stage_eval_input_dir(run_dir, eval_output_dir)
    command = [
        resolve_runner_python(context),
        "-m",
        "secb.evaluator.eval_instances",
        "--input-dir",
        str(eval_output_dir),
        "--type",
        "poc",
        "--split",
        dataset_split,
        "--agent",
        "smolagent",
        "--graders",
        graders,
        "--grader-line-tolerance",
        str(grader_line_tolerance),
    ]
    run_command(command, cwd=context.secbench_root, env=with_local_smolagents_pythonpath(context))
    # The staged output.jsonl is a byte copy of the run dir's; drop it once the
    # evaluator has consumed it so eval/ holds only reports/digests/analysis.
    staged_output = eval_output_dir / "output.jsonl"
    if staged_output.exists():
        staged_output.unlink()
    record_manifest_stage(
        run_dir,
        "eval",
        {
            "eval_dir": str(eval_output_dir.relative_to(run_dir)),
            "graders": graders,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )
    print(f"Evaluation output: {eval_output_dir}")
    return eval_output_dir


def run_analysis_stage(*, context: RunContext, eval_dir: Path, analysis_grader: str | None = None) -> Path:
    analyze_script = Path(__file__).resolve().parent / "analyze_run.py"
    analysis_output = eval_dir / "analysis.md"
    command = [
        resolve_runner_python(context),
        str(analyze_script),
        "--eval-dir",
        str(eval_dir),
        "--output",
        str(analysis_output),
    ]
    if analysis_grader:
        command.extend(["--grader", analysis_grader])
    run_command(command, cwd=context.repo_root)
    print(f"Analysis output: {analysis_output}")
    return analysis_output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context = get_run_context()

    config: dict | None = None
    config_path: Path | None = None
    if args.config:
        config_path = resolve_config_path(args.config, context.repo_root)
        config = load_config(config_path)

    run_dir: Path | None = None
    eval_dir: Path | None = None

    if "generate" in args.stages:
        assert config is not None
        assert config_path is not None
        site = resolve_site()
        tier = resolve_tier(config, args.smoke)
        base_output_dir = resolve_generation_output_root(
            config,
            args.output_dir,
            tier=tier,
            config_path=config_path,
            repo_root=context.repo_root,
        )
        run_dir = run_generation_stage(
            context=context,
            config_path=config_path,
            base_output_dir=base_output_dir,
            instance_id=args.instance_id,
            num_workers=args.num_workers,
            output_dir_override=args.output_dir,
            site=site,
        )
        write_generation_manifest(
            run_dir=run_dir,
            site=site,
            tier=tier,
            config_path=config_path,
            config=config,
            repo_root=context.repo_root,
            instance_id=args.instance_id,
            num_workers=args.num_workers,
        )

    if "eval" in args.stages:
        assert config is not None
        if run_dir is None:
            run_dir = validate_run_dir(args.run_dir)
            print(f"Run input: {run_dir}")
        eval_dir = run_eval_stage(
            context=context,
            config=config,
            run_dir=run_dir,
            graders=args.graders,
            grader_line_tolerance=args.grader_line_tolerance,
        )

    if "analyze" in args.stages:
        if eval_dir is None:
            eval_dir = validate_eval_dir(args.eval_dir)
            print(f"Evaluation input: {eval_dir}")
        run_analysis_stage(context=context, eval_dir=eval_dir, analysis_grader=args.analysis_grader)
        if eval_dir.parent.name == "eval":
            record_manifest_stage(
                eval_dir.parent.parent,
                "analyze",
                {
                    "eval_dir": f"eval/{eval_dir.name}",
                    "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                },
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
