#!/usr/bin/env python3
"""Post-run analysis tool for SEC-bench PoC generation experiments.

Reads trajectory, evaluation, and metadata artifacts produced by the
PoC generation + evaluation pipeline and renders human-readable
Markdown reports.

Usage:
    # Single eval run
    python analyze_run.py --eval-dir .../evals/e5-combined_run1_20260326_150736

    # Single instance deep-dive
    python analyze_run.py --eval-dir ... --instance-id gpac.cve-2022-3178

    # Cross-config comparison
    python analyze_run.py --eval-dirs .../evals/e5-baseline_run1_* .../evals/e5-tooling_run1_*

    # Raw run (no eval results)
    python analyze_run.py --run-dir .../e5-combined_run1/20260326_150736
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCallInfo:
    name: str
    args: object


@dataclass
class Step:
    step_number: int
    tool_name: str  # primary (first) tool name — kept for backward compat
    tool_args: object  # primary (first) tool args — kept for backward compat
    observations: str | None
    error: dict | None
    timing: dict | None
    token_usage: dict | None
    is_final_answer: bool
    all_tool_calls: list[ToolCallInfo] | None = None  # all tool calls in step


@dataclass
class InstanceData:
    instance_id: str
    meta: dict
    steps: list[Step]
    task_prompt: str
    eval_result: dict | None
    eval_results_by_grader: dict[str, dict]
    testcase_files: list[tuple[str, int]]  # (name, size)
    # B/C/P scaffold artifacts (empty/None for runs that predate them)
    synthesis_log: list[dict] = field(default_factory=list)
    synthesis_ledger: list[dict] = field(default_factory=list)
    synthesis_obs: list[dict] = field(default_factory=list)
    finalization_reviews: list[dict] = field(default_factory=list)
    finalization_guard: list[dict] = field(default_factory=list)
    plan: dict | None = None
    path: Path | None = None
    task_context: dict | None = None
    independent_synthesis_audit: dict = field(default_factory=dict)


@dataclass
class EvalRun:
    path: Path
    name: str
    config: dict | None
    available_graders: list[str] = field(default_factory=list)
    report_label: str = "evaluation report"
    primary_grader: str | None = None
    instances: list[InstanceData] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    """Tolerant JSONL reader: returns [] on missing file, skips blank/corrupt lines."""
    records: list[dict] = []
    if not path.exists():
        return records
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
    except OSError:
        return records
    return records


def load_report(path: Path) -> dict[str, dict]:
    """Read a grader report JSONL file, keyed by instance_id."""
    return {
        entry["instance_id"]: entry
        for entry in load_jsonl(path)
        if "instance_id" in entry
    }


def resolve_instance_root(eval_dir: Path) -> Path:
    """Directory holding the per-instance artifact dirs for an eval dir.

    Legacy layout (runs/evals/<exp>_<ts>/): instance dirs are full copies inside
    the eval dir itself. Nested layout (<run-dir>/eval/<ts>/): the eval dir holds
    only reports/digests/analysis; instance dirs live in the parent run dir.
    """
    if any(p.is_dir() and (p / "artifacts").exists() for p in eval_dir.iterdir()):
        return eval_dir
    if eval_dir.parent.name == "eval":
        return eval_dir.parent.parent
    return eval_dir


def iter_instance_dirs(eval_dir: Path) -> list[Path]:
    """Instance dirs for an eval dir, layout-agnostic (see resolve_instance_root)."""
    root = resolve_instance_root(eval_dir)
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and p.name != "eval" and (p / "artifacts").exists()
    )


def discover_report_paths(eval_dir: Path) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    for grader in ("loose", "caller", "semantic", "strict"):
        path = eval_dir / f"report_{grader}_smolagent.jsonl"
        if path.exists():
            reports[grader] = path

    legacy = eval_dir / "report_sanitizer.jsonl"
    if legacy.exists():
        if legacy.is_symlink():
            try:
                target = legacy.resolve()
                for grader, path in reports.items():
                    if path.resolve() == target:
                        break
                else:
                    reports["sanitizer"] = legacy
            except OSError:
                reports["sanitizer"] = legacy
        elif "loose" not in reports:
            reports["loose"] = legacy
    return reports


def resolve_primary_grader(available_reports: dict[str, Path], preferred_grader: str | None) -> str | None:
    if preferred_grader:
        if preferred_grader in available_reports:
            return preferred_grader
        raise FileNotFoundError(f"Requested grader report not found: {preferred_grader}")

    for grader in ("loose", "caller", "semantic", "strict", "sanitizer"):
        if grader in available_reports:
            return grader
    return next(iter(available_reports), None)


def _parse_step(raw: dict) -> Step | None:
    """Parse a single trajectory.jsonl line into a Step."""
    if "task" in raw and "step_number" not in raw:
        return None  # task line (step 0)

    step_number = raw.get("step_number", 0)
    raw_tool_calls = raw.get("tool_calls") or []
    all_tcs: list[ToolCallInfo] = []

    if isinstance(raw_tool_calls, list):
        for tc in raw_tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            if isinstance(fn, dict):
                name = fn.get("name", "?")
                args = fn.get("arguments", {})
                all_tcs.append(ToolCallInfo(name=name, args=args))

    tool_name = all_tcs[0].name if all_tcs else "?"
    tool_args = all_tcs[0].args if all_tcs else {}

    error = raw.get("error")
    if error and not raw_tool_calls:
        tool_name = "ERROR"

    return Step(
        step_number=step_number,
        tool_name=tool_name,
        tool_args=tool_args,
        observations=raw.get("observations"),
        error=error,
        timing=raw.get("timing"),
        token_usage=raw.get("token_usage"),
        is_final_answer=bool(raw.get("is_final_answer")),
        all_tool_calls=all_tcs if len(all_tcs) > 1 else None,
    )


def load_trajectory(path: Path) -> tuple[str, list[Step]]:
    """Load trajectory.jsonl. Returns (task_prompt, steps)."""
    task_prompt = ""
    steps: list[Step] = []
    if not path.exists():
        return task_prompt, steps
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            if "task" in raw and "step_number" not in raw:
                task_prompt = raw.get("task", "")
                continue
            step = _parse_step(raw)
            if step is not None:
                steps.append(step)
    return task_prompt, steps


def load_synthesis_log(art_dir: Path) -> list[dict]:
    """One record per C invocation, sorted by invocation_index."""
    records = load_jsonl(art_dir / "synthesis_log.jsonl")
    return sorted(records, key=lambda r: r.get("invocation_index") or 0)


def load_synthesis_ledger(art_dir: Path) -> list[dict]:
    """C-side attempt ledger (one row per helper write+run); field sets vary."""
    return load_jsonl(art_dir / "synthesis_attempt_ledger.jsonl")


def load_synthesis_observability(art_dir: Path) -> list[dict]:
    """Run-end adoption sidecar (absent on pre-73a210b runs)."""
    return load_jsonl(art_dir / "synthesis_observability.jsonl")


def load_finalization_reviews(art_dir: Path) -> list[dict]:
    """One record per B review, file order preserved (anchoring relies on it)."""
    return load_jsonl(art_dir / "finalization_review.jsonl")


def load_finalization_guard(art_dir: Path) -> list[dict]:
    """Non-binding rule-guard decisions (instrumentation only)."""
    return load_jsonl(art_dir / "synthesis_finalization_guard.jsonl")


def load_plan(art_dir: Path, meta: dict) -> dict | None:
    """P plan record: planning log file if present, else meta-derived summary."""
    records = load_jsonl(art_dir / "planning_log.jsonl")
    if records:
        return records[0]
    plan_result = meta.get("plan_result")
    if isinstance(plan_result, dict):
        return plan_result
    # P writes only p_* keys into meta; p_steps_used is the "P ran" discriminator.
    if "p_steps_used" in meta:
        return {
            "p_steps_used": meta.get("p_steps_used"),
            "degraded": meta.get("p_degraded"),
            "degraded_reason": meta.get("p_degraded_reason", ""),
        }
    return None


_EXTERNAL_ACCESS_RE = re.compile(
    r"(?ix)"
    r"(https?://|github\.com|gitlab\.com|"
    r"(?:^|\s)(?:curl|wget)(?:\s|$)|"
    r"(?:^|\s)git\s+(?:clone|fetch|pull)(?:\s|$)|"
    r"(?:^|\s)gh\s+(?:api|repo\s+clone)(?:\s|$)|"
    r"(?:requests\.(?:get|post)|urllib\.request)|"
    r"(?:^|\s)(?:pip|pip3)\s+install(?:\s|$)|"
    r"(?:^|\s)(?:apt|apt-get|npm|yarn)\s+(?:install|add)(?:\s|$))"
)


def audit_independent_synthesis(steps: list[Step]) -> dict:
    """Flag likely external retrieval using the existing Solver trajectory."""
    if not steps:
        return {"status": "unknown", "evidence": []}
    evidence: list[dict] = []
    inspected = 0
    for step in steps:
        calls = step.all_tool_calls or [ToolCallInfo(step.tool_name, step.tool_args)]
        for call in calls:
            if call.name not in {"cmd", "python_interpreter"}:
                continue
            inspected += 1
            try:
                arg_text = (
                    call.args
                    if isinstance(call.args, str)
                    else json.dumps(call.args, sort_keys=True)
                )
            except (TypeError, ValueError):
                continue
            match = _EXTERNAL_ACCESS_RE.search(arg_text)
            if match:
                evidence.append(
                    {
                        "step": step.step_number,
                        "tool": call.name,
                        "signal": match.group(0)[:120],
                        "args_head": arg_text[:300],
                    }
                )
    if evidence:
        return {"status": "flagged", "evidence": evidence[:5]}
    if inspected == 0:
        return {"status": "unknown", "evidence": []}
    return {"status": "clean", "evidence": []}


def list_testcase_files(testcase_dir: Path) -> list[tuple[str, int]]:
    """List files in testcase/ with their sizes."""
    if not testcase_dir.exists():
        return []
    files = []
    for f in sorted(testcase_dir.iterdir()):
        if f.is_file():
            files.append((f.name, f.stat().st_size))
    return files


def list_poc_tar_contents(base64_path: Path) -> list[tuple[str, int]]:
    """Decode poc.tar.gz.base64 and list file names + sizes."""
    if not base64_path.exists():
        return []
    try:
        raw = base64_path.read_text().strip()
        data = base64.b64decode(raw)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            return [(m.name, m.size) for m in tar.getmembers() if m.isfile()]
    except Exception:
        return []


def build_dataset_index(dataset_dir: Path | None) -> dict[str, dict]:
    """Index SEC-bench dataset records by instance_id across eval-oss/eval-cve jsonl."""
    index: dict[str, dict] = {}
    if dataset_dir is None or not dataset_dir.exists():
        return index
    for name in ("eval-oss.jsonl", "eval-cve.jsonl"):
        path = dataset_dir / name
        for rec in load_jsonl(path):
            iid = rec.get("instance_id")
            if iid and iid not in index:
                index[iid] = rec
    return index


def task_context_from_record(rec: dict | None) -> dict | None:
    """Compact task-definition block for the digest: what the target vuln actually is.

    The `patch` (ground-truth fix) is the highest-value field — it pins the exact
    crashing condition, letting an interpreter judge how close the run got.
    """
    if not rec:
        return None
    def head(key: str, n: int) -> str | None:
        v = rec.get(key)
        if not v:
            return None
        s = str(v).strip()
        return s[:n] + ("\n…[truncated]" if len(s) > n else "")
    # Sanitizer reports carry a long shadow-byte dump after the crash frame — useless
    # for interpretation. Cut it at the "Shadow bytes" marker, then head-truncate.
    san = rec.get("sanitizer_report")
    if san:
        san = str(san)
        cut = san.find("Shadow bytes")
        if cut != -1:
            san = san[:cut].rstrip()
        san = san[:1800] + ("\n…[truncated]" if len(san) > 1800 else "")
    return {
        "project_name": rec.get("project_name"),
        "repo": rec.get("repo"),
        "lang": rec.get("lang"),
        "sanitizer": rec.get("sanitizer"),
        "expected_exit_code": rec.get("exit_code"),
        "bug_description": head("bug_description", 1200),
        "sanitizer_report": san or None,
        "patch": head("patch", 6000),
        "bug_report": head("bug_report", 800),
    }


def load_instance(
    instance_dir: Path,
    eval_result: dict | None = None,
    eval_results_by_grader: dict[str, dict] | None = None,
    dataset_index: dict[str, dict] | None = None,
) -> InstanceData:
    """Load all data for a single instance."""
    instance_id = instance_dir.name
    art_dir = instance_dir / "artifacts"
    meta = load_meta(art_dir / "meta.json")
    task_prompt, steps = load_trajectory(art_dir / "trajectory.jsonl")
    testcase_files = list_testcase_files(instance_dir / "testcase")
    return InstanceData(
        instance_id=instance_id,
        meta=meta,
        steps=steps,
        task_prompt=task_prompt,
        eval_result=eval_result,
        eval_results_by_grader=eval_results_by_grader or {},
        testcase_files=testcase_files,
        synthesis_log=load_synthesis_log(art_dir),
        synthesis_ledger=load_synthesis_ledger(art_dir),
        synthesis_obs=load_synthesis_observability(art_dir),
        finalization_reviews=load_finalization_reviews(art_dir),
        finalization_guard=load_finalization_guard(art_dir),
        plan=load_plan(art_dir, meta),
        path=instance_dir,
        task_context=task_context_from_record((dataset_index or {}).get(instance_id)),
        independent_synthesis_audit=audit_independent_synthesis(steps),
    )


def find_config_for_run_dir(run_dir: Path, configs_dir: Path | None) -> dict | None:
    """Resolve the config for a run dir in the nested layout.

    Prefers the run's own config_snapshot.toml (written by the orchestrator /
    backfilled by migration), then the manifest's config_name, then the
    experiment-dir-name match against configs/.
    """
    snapshot = run_dir / "config_snapshot.toml"
    if snapshot.exists():
        with snapshot.open("rb") as f:
            return tomllib.load(f)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and configs_dir is not None:
        try:
            config_name = json.loads(manifest_path.read_text()).get("config_name")
        except (json.JSONDecodeError, OSError):
            config_name = None
        if config_name and (configs_dir / config_name).exists():
            with (configs_dir / config_name).open("rb") as f:
                return tomllib.load(f)
    if configs_dir is not None:
        return find_config(run_dir.parent.name, configs_dir)
    return None


def find_config(eval_dir_name: str, configs_dir: Path) -> dict | None:
    """Try to resolve the TOML config from an eval directory name.

    Eval dirs are named: {config_stem}_{YYYYMMDD_HHMMSS}[_rerun_...]
    Config files are: {experiment}[_{variant}][_run{N}].toml
    """
    # Strip timestamp suffix(es): _YYYYMMDD_HHMMSS or _rerun_YYYYMMDD_HHMM
    name = re.sub(r"_rerun_\d{8}_\d{4}$", "", eval_dir_name)
    name = re.sub(r"_\d{8}_\d{6}$", "", name)

    if not configs_dir.exists():
        return None
    for toml_path in configs_dir.glob("*.toml"):
        # Check if the config stem matches
        stem = toml_path.stem  # e.g., smolagent_gpt-5.4-mini_flex_e5-combined_run1
        if stem.endswith(name) or name in stem:
            with open(toml_path, "rb") as f:
                return tomllib.load(f)
    return None


def load_eval_dir(
    path: Path,
    configs_dir: Path | None = None,
    preferred_grader: str | None = None,
    report_path: Path | None = None,
    dataset_index: dict[str, dict] | None = None,
) -> EvalRun:
    """Load an eval directory: report + all instances."""
    if report_path is not None:
        available_reports = {"custom": report_path}
        primary_grader = "custom"
        report_label = report_path.name
    else:
        available_reports = discover_report_paths(path)
        primary_grader = resolve_primary_grader(available_reports, preferred_grader)
        report_label = (
            available_reports[primary_grader].name
            if primary_grader is not None and primary_grader in available_reports
            else "evaluation report"
        )

    reports_by_grader = {grader: load_report(report) for grader, report in available_reports.items()}
    primary_report = reports_by_grader.get(primary_grader or "", {})

    nested_run_dir = path.parent.parent if path.parent.name == "eval" else None
    if nested_run_dir is not None:
        config = find_config_for_run_dir(nested_run_dir, configs_dir)
        name = f"{nested_run_dir.parent.name}/{nested_run_dir.name}/eval/{path.name}"
    else:
        config = find_config(path.name, configs_dir) if configs_dir else None
        name = path.name

    instances = []
    for subdir in iter_instance_dirs(path):
        eval_result = primary_report.get(subdir.name)
        eval_results_by_grader = {
            grader: report[subdir.name]
            for grader, report in reports_by_grader.items()
            if subdir.name in report
        }
        instances.append(load_instance(subdir, eval_result, eval_results_by_grader, dataset_index))

    return EvalRun(
        path=path,
        name=name,
        config=config,
        available_graders=list(available_reports.keys()),
        report_label=report_label,
        primary_grader=primary_grader,
        instances=instances,
    )


def load_run_dir(
    path: Path,
    configs_dir: Path | None = None,
    dataset_index: dict[str, dict] | None = None,
) -> EvalRun:
    """Load a raw run directory (no eval results)."""
    config = find_config_for_run_dir(path, configs_dir)
    instances = []
    for subdir in sorted(path.iterdir()):
        if subdir.is_dir() and subdir.name != "eval" and (subdir / "artifacts").exists():
            instances.append(load_instance(subdir, dataset_index=dataset_index))
    return EvalRun(path=path, name=path.parent.name, config=config, instances=instances)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def truncate(text: str | None, max_chars: int) -> str:
    if text is None:
        return ""
    text = text.strip()
    # Collapse consecutive whitespace for table display
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def fmt_tokens(n: int | None) -> str:
    if n is None:
        return "—"
    return f"{n:,}"


def fmt_cost(value: float | int | None) -> str:
    if value is None:
        return "—"
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def meta_input_tokens(meta: dict) -> int | None:
    return meta.get("total_input_tokens", meta.get("input_tokens"))


def meta_output_tokens(meta: dict) -> int | None:
    return meta.get("total_output_tokens", meta.get("output_tokens"))


def meta_cost(meta: dict) -> float | int | None:
    return meta.get("total_cost", meta.get("cost"))


def budget_terminated(meta: dict) -> bool:
    return meta.get("termination_reason") in {"cost_budget_exhausted", "pricing_unavailable"}


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    return f"{seconds:.1f}s"


def fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}K"
    return f"{size / (1024 * 1024):.1f}M"


def count_tool_usage(steps: list[Step]) -> Counter:
    c: Counter = Counter()
    skip = {"?", "ERROR", "final_answer"}
    for s in steps:
        if s.all_tool_calls:
            for tc in s.all_tool_calls:
                if tc.name not in skip:
                    c[tc.name] += 1
        elif s.tool_name not in skip:
            c[s.tool_name] += 1
    return c


def tool_usage_str(steps: list[Step]) -> str:
    c = count_tool_usage(steps)
    if not c:
        return "—"
    parts = [f"{name} ({cnt}x)" for name, cnt in c.most_common()]
    return ", ".join(parts)


def result_str(eval_result: dict | None) -> str:
    if eval_result is None:
        return "N/A"
    return "PASS" if eval_result.get("success") else "FAIL"


def result_reason(eval_result: dict | None) -> str:
    if eval_result is None:
        return ""
    return eval_result.get("reason", "")


def grader_result_str(eval_results_by_grader: dict[str, dict], grader: str) -> str:
    return result_str(eval_results_by_grader.get(grader))


# ---------------------------------------------------------------------------
# Timeline (interleaved A/B/C/P view)
# ---------------------------------------------------------------------------

# Canonical gate ladder (see docs/scaffold/communication.md); C may emit
# non-canonical gate names — treat those as informational only.
GATE_LADDER = [
    "artifact_created",
    "harness_accepted",
    "parser_branch_reached",
    "target_family_seen",
    "target_frame_seen",
]


@dataclass
class TimelineEvent:
    a_step: int | None  # anchor step in A's trajectory (None = unanchored)
    order: int  # tiebreak for same-step events
    kind: str  # a_run | c_invocation | b_review | plan
    summary: str
    detail: dict = field(default_factory=dict)


def step_tool_names(step: Step) -> list[str]:
    if step.all_tool_calls:
        return [tc.name for tc in step.all_tool_calls]
    return [step.tool_name]


def final_submission_steps(steps: list[Step]) -> list[int]:
    """Ordered A-step numbers whose tool calls include final_submission."""
    return [s.step_number for s in steps if "final_submission" in step_tool_names(s)]


def adoption_for_invocation(inst: InstanceData, rec: dict) -> dict:
    """Adoption flags for a C invocation: observability sidecar first, inline fallback.

    Fields sourced from the sidecar may be proxy-inferred (temporal ordering, not
    artifact provenance) — ``a_adopted_is_proxy: True`` is preserved so callers and
    renderers can distinguish proven adoption from a heuristic estimate.
    """
    inv = rec.get("invocation_index")
    for obs in inst.synthesis_obs:
        if obs.get("invocation_index") == inv:
            return {
                "a_followed": obs.get("a_followed"),
                "a_adopted_into_testcase": obs.get("a_adopted_into_testcase"),
                "downstream_passed": obs.get("downstream_passed"),
                "a_adopted_is_proxy": obs.get("a_adopted_is_proxy"),
            }
    return {
        "a_followed": rec.get("a_followed"),
        "a_adopted_into_testcase": rec.get("a_adopted_into_testcase"),
        "downstream_passed": rec.get("downstream_passed"),
        "a_adopted_is_proxy": rec.get("a_adopted_is_proxy"),
    }


def _c_event_detail(inst: InstanceData, rec: dict) -> dict:
    inp = rec.get("input") or {}
    payload = rec.get("emitted_payload") or {}
    detail = {
        "invocation_index": rec.get("invocation_index"),
        "a_step": rec.get("a_step"),
        "delegated_problem": (
            inp.get("delegated_problem") or inp.get("blocker_summary")
            if isinstance(inp, dict)
            else None
        ),
        "c_steps_used": rec.get("c_steps_used"),
        "step_budget": rec.get("step_budget") or {},
        "degraded": rec.get("degraded"),
        "degraded_reason": rec.get("degraded_reason") or None,
        "validation_outcome": payload.get("validation") or payload.get("validation_outcome"),
        "crossed_gate": payload.get("crossed_gate"),
        "failed_gate": payload.get("failed_gate"),
        "helper_files": payload.get("helper_files"),
    }
    detail.update(adoption_for_invocation(inst, rec))
    return detail


def _c_event_summary(detail: dict) -> str:
    inv = detail.get("invocation_index", "?")
    step = f" @step{detail['a_step']}" if detail.get("a_step") is not None else ""
    bits = []
    if detail.get("delegated_problem"):
        bits.append(f"problem={str(detail['delegated_problem'])[:80]}")
    if detail.get("c_steps_used") is not None:
        bits.append(f"{detail['c_steps_used']} C-steps")
    if detail.get("degraded"):
        bits.append(f"DEGRADED ({detail.get('degraded_reason') or 'unknown'})")
    else:
        if detail.get("validation_outcome"):
            bits.append(f"validation={detail['validation_outcome']}")
        if detail.get("crossed_gate"):
            bits.append(f"crossed={detail['crossed_gate']}")
        if detail.get("failed_gate"):
            bits.append(f"failed={detail['failed_gate']}")
    adopted = detail.get("a_adopted_into_testcase")
    if adopted is not None:
        bits.append(f"adopted={adopted}")
    head = f"C invocation #{inv}{step}"
    return f"{head} ({', '.join(bits)})" if bits else head


def _b_event_detail(rec: dict, a_step: int | None, note: str | None) -> dict:
    guard = rec.get("rule_guard_decision") or {}
    detail = {
        "a_step": a_step,
        "verdict": rec.get("verdict"),
        "block_index": rec.get("block_index"),
        "evidence_steps": rec.get("evidence_steps"),
        "stop_reason": rec.get("stop_reason"),
        "remaining_steps": rec.get("remaining_steps"),
        "degraded": rec.get("degraded"),
        "degraded_reason": rec.get("degraded_reason") or None,
        "rule_guard_decision": guard.get("decision") if isinstance(guard, dict) else guard,
        "relation": (rec.get("evidence_relation") or {}).get("relation"),
        "recommended_verdict": rec.get("recommended_verdict"),
        "policy_mode": rec.get("policy_mode") or "legacy",
        "challenge": rec.get("challenge") or None,
    }
    if note:
        detail["note"] = note
    return detail


def _b_event_summary(detail: dict) -> str:
    step = f"@step{detail['a_step']}" if detail.get("a_step") is not None else "(unanchored)"
    verdict = detail.get("verdict") or "?"
    bits = []
    if detail.get("block_index") is not None:
        bits.append(f"block {detail['block_index']}")
    if detail.get("evidence_steps"):
        bits.append(f"evidence {detail['evidence_steps']}")
    if detail.get("degraded"):
        bits.append(f"DEGRADED ({detail.get('degraded_reason') or 'unknown'})")
    if detail.get("note"):
        bits.append(detail["note"])
    if detail.get("relation"):
        bits.append(f"relation={detail['relation']}")
    recommended = detail.get("recommended_verdict")
    if recommended and recommended != verdict:
        bits.append(f"shadow={recommended}")
    suffix = f" ({', '.join(bits)})" if bits else ""
    return f"A final_submission {step} → B: {verdict}{suffix}"


def build_timeline(inst: InstanceData) -> list[TimelineEvent]:
    """Interleave A trajectory, C invocations, B reviews, and P plan into one timeline."""
    events: list[TimelineEvent] = []
    order = 0

    if inst.plan is not None:
        p_steps = inst.plan.get("p_steps_used")
        degraded = inst.plan.get("degraded")
        bits = []
        if p_steps is not None:
            bits.append(f"{p_steps} P-steps")
        if degraded:
            bits.append(f"DEGRADED ({inst.plan.get('degraded_reason') or 'unknown'})")
        elif degraded is False:
            bits.append("ladder injected into A's task")
        summary = f"P planning phase ({', '.join(bits)})" if bits else "P planning phase"
        events.append(TimelineEvent(a_step=0, order=order, kind="plan", summary=summary, detail=dict(inst.plan)))
        order += 1

    # C invocations anchored by synthesis_log.a_step.
    for rec in inst.synthesis_log:
        detail = _c_event_detail(inst, rec)
        events.append(
            TimelineEvent(
                a_step=rec.get("a_step"),
                order=order,
                kind="c_invocation",
                summary=_c_event_summary(detail),
                detail=detail,
            )
        )
        order += 1

    # B reviews anchored by zipping file order against ordered final_submission steps.
    submission_steps = final_submission_steps(inst.steps)
    count_mismatch = (
        bool(inst.finalization_reviews)
        and len(submission_steps) != len(inst.finalization_reviews)
    )
    for i, rec in enumerate(inst.finalization_reviews):
        a_step = submission_steps[i] if i < len(submission_steps) else None
        note = "anchor uncertain: review/submission count mismatch" if count_mismatch else None
        detail = _b_event_detail(rec, a_step, note)
        events.append(
            TimelineEvent(
                a_step=a_step,
                order=order,
                kind="b_review",
                summary=_b_event_summary(detail),
                detail=detail,
            )
        )
        order += 1

    # Collapse plain-A step ranges between anchors into a_run segments.
    anchor_steps = {e.a_step for e in events if e.a_step is not None and e.kind != "plan"}
    segment: list[int] = []
    for step in inst.steps:
        n = step.step_number
        if n in anchor_steps:
            if segment:
                events.append(_a_run_event(segment, order))
                order += 1
                segment = []
        else:
            segment.append(n)
    if segment:
        events.append(_a_run_event(segment, order))
        order += 1

    big = 10**9
    events.sort(key=lambda e: (e.a_step if e.a_step is not None else big, e.order))
    return events


def _a_run_event(segment: list[int], order: int) -> TimelineEvent:
    lo, hi = segment[0], segment[-1]
    label = f"A steps {lo}–{hi}" if lo != hi else f"A step {lo}"
    return TimelineEvent(
        a_step=lo,
        order=order,
        kind="a_run",
        summary=f"{label} ({len(segment)} step{'s' if len(segment) != 1 else ''})",
        detail={"first_step": lo, "last_step": hi, "step_count": len(segment)},
    )


def has_scaffold_data(inst: InstanceData) -> bool:
    return bool(inst.synthesis_log or inst.finalization_reviews or inst.plan)


def render_timeline(events: list[TimelineEvent]) -> str:
    lines = ["\n### Run Timeline"]
    for e in events:
        lines.append(f"- {e.summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _blockquote(text: str, max_chars: int) -> str:
    """Wrap truncated text as a Markdown blockquote (each line prefixed with >)."""
    t = truncate(text, max_chars)
    if not t:
        return "> *(empty)*"
    return "\n".join(f"> {line}" for line in t.split("\n"))


def _render_tool_args(tool: str, tool_args: object, lines: list[str], max_chars: int) -> None:
    """Render a single tool call's arguments into *lines*."""
    if not isinstance(tool_args, dict):
        if tool == "final_answer":
            answer = tool_args
            if not isinstance(answer, str):
                answer = json.dumps(answer)
            if answer:
                lines.append(f"> {truncate(answer, max_chars)}")
            return

        if tool_args not in ({}, None, ""):
            args_str = tool_args if isinstance(tool_args, str) else json.dumps(tool_args)
            lines.append(f"> {truncate(args_str, max_chars)}")
        return

    if tool == "cmd":
        cmd = tool_args.get("command", "")
        if cmd:
            lines.append("```")
            lines.append(truncate(cmd, max_chars * 2))
            lines.append("```")
    elif tool == "python_interpreter":
        code = tool_args.get("code", "")
        if code:
            lines.append("```python")
            code_lines = code.split("\n")
            preview = "\n".join(code_lines[:10])
            if len(code_lines) > 10:
                preview += f"\n# ... ({len(code_lines) - 10} more lines)"
            lines.append(preview)
            lines.append("```")
    elif tool == "sanitizer_parser":
        lines.append("> parse sanitizer output")
    elif tool == "gdb":
        cmd = tool_args.get("command", "")
        gdb_commands = tool_args.get("gdb_commands", "")
        work_dir = tool_args.get("work_dir", "")
        timeout = tool_args.get("timeout")
        if cmd:
            lines.append("```")
            lines.append(truncate(cmd, max_chars))
            lines.append("```")
        meta_bits = []
        if work_dir:
            meta_bits.append(f"work_dir=`{work_dir}`")
        if timeout is not None:
            meta_bits.append(f"timeout={timeout}s")
        if meta_bits:
            lines.append(f"> {' · '.join(meta_bits)}")
        if gdb_commands:
            lines.append(f"> gdb: `{truncate(gdb_commands, max_chars)}`")
    elif tool == "structured_gdb":
        cmd = tool_args.get("command", "")
        if cmd:
            lines.append("```")
            lines.append(truncate(cmd, max_chars))
            lines.append("```")
    elif tool == "binary_helper":
        action = tool_args.get("action", "?")
        fmt = tool_args.get("format", "?")
        path = tool_args.get("path", "")
        extras = []
        for key in ("width", "height", "n_colors"):
            val = tool_args.get(key)
            if val is not None:
                extras.append(f"{key}={val}")
        extras_str = f" [{', '.join(extras)}]" if extras else ""
        path_str = f" → `{path}`" if path else ""
        lines.append(f"> `{action}` ({fmt}){extras_str}{path_str}")
    elif tool == "oracle_seed":
        instance_id = tool_args.get("instance_id", "?")
        variant = tool_args.get("variant", "default")
        lines.append(f"> materialize seed for `{instance_id}` (variant=`{variant}`)")
    elif tool == "magick_probe":
        path = tool_args.get("path", "?")
        mode = tool_args.get("mode", "uil")
        lines.append(f"> probe `{path}` (mode=`{mode}`)")
    elif tool == "ubsan":
        cmd = tool_args.get("command", "")
        if cmd:
            lines.append("```")
            lines.append(truncate(cmd, max_chars))
            lines.append("```")
    elif tool == "final_answer":
        answer = tool_args.get("answer", "")
        if not isinstance(answer, str):
            answer = json.dumps(answer)
        if answer:
            lines.append(f"> {truncate(answer, max_chars)}")
    elif tool == "create_tool":
        name = tool_args.get("tool_name", tool_args.get("name", "?"))
        lines.append(f"> create tool: `{name}`")
    else:
        args_str = json.dumps(tool_args)
        if args_str != "{}":
            lines.append(f"> {truncate(args_str, max_chars)}")


def render_step(step: Step, max_chars: int) -> str:
    """Render a single trajectory step as a Markdown block."""
    tool = step.tool_name
    lines: list[str] = []

    # Token usage
    tu = step.token_usage or {}
    in_tok = tu.get("input_tokens")
    out_tok = tu.get("output_tokens")
    tokens = f"{fmt_tokens(in_tok)}→{fmt_tokens(out_tok)}" if in_tok is not None else ""

    # Timing
    timing = step.timing or {}
    duration = fmt_duration(timing.get("duration"))

    # Header line
    meta_parts = []
    if tokens:
        meta_parts.append(f"{tokens} tokens")
    if duration != "—":
        meta_parts.append(duration)
    meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""

    if tool == "ERROR":
        err = step.error or {}
        err_type = err.get("type", "Error") if isinstance(err, dict) else "Error"
        err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        lines.append(f"**Step {step.step_number}** — `ERROR`{meta_str}")
        lines.append(f"> **{err_type}**: {truncate(err_msg, max_chars)}")
        return "\n".join(lines)

    # Header — show all tool names when parallel calls exist
    if step.all_tool_calls:
        tool_labels = ", ".join(f"`{tc.name}`" for tc in step.all_tool_calls)
        lines.append(f"**Step {step.step_number}** — {tool_labels}{meta_str}")
    else:
        lines.append(f"**Step {step.step_number}** — `{tool}`{meta_str}")

    # Arguments — render each tool call when parallel, otherwise just the primary
    if step.all_tool_calls:
        for tc in step.all_tool_calls:
            _render_tool_args(tc.name, tc.args, lines, max_chars)
    else:
        _render_tool_args(tool, step.tool_args, lines, max_chars)

    # Observation / result
    obs = step.observations
    if tool == "sanitizer_parser" and obs:
        try:
            parsed = json.loads(obs)
            crash = parsed.get("crash_type", "?")
            func = parsed.get("top_function", "?")
            lines.append(f"> **Result**: {crash} in `{func}`")
        except (json.JSONDecodeError, TypeError):
            lines.append(_blockquote(obs, max_chars))
    elif tool == "final_answer":
        pass  # answer already shown above
    elif tool == "ERROR":
        pass
    elif obs:
        lines.append(_blockquote(obs, max_chars))

    return "\n".join(lines)


def render_trajectory(steps: list[Step], max_chars: int) -> str:
    """Render the full trajectory as block-per-step Markdown."""
    if not steps:
        return "_No agent steps recorded._\n"

    blocks = []
    for step in steps:
        blocks.append(render_step(step, max_chars))
    return "\n\n".join(blocks) + "\n"


def count_role_tool_usage(records: list[dict], key: str) -> Counter:
    """Flatten {step,tool,args_summary} tool-call lists from B/C records into a Counter."""
    c: Counter = Counter()
    for rec in records:
        calls = rec.get(key) or []
        if not isinstance(calls, list):
            continue
        for tc in calls:
            if isinstance(tc, dict) and tc.get("tool"):
                c[tc["tool"]] += 1
    return c


def render_cost_by_role(meta: dict) -> list[str]:
    """Role-cost summary lines (reused by Metadata section and the digest)."""
    lines: list[str] = []
    if "total_cost" not in meta:
        return lines
    if meta.get("cost_by_role"):
        role_costs = " | ".join(
            f"{role}={fmt_cost(value)}"
            for role, value in sorted(meta["cost_by_role"].items())
        )
        lines.append(f"- Role costs: {role_costs} | Total={fmt_cost(meta.get('total_cost'))}")
        if meta.get("model_calls_by_role"):
            role_calls = " | ".join(
                f"{role}={calls}"
                for role, calls in sorted(meta["model_calls_by_role"].items())
            )
            lines.append(f"- Model calls by role: {role_calls}")
    else:
        lines.append(
            "- A/B/C cost: "
            f"A={fmt_cost(meta.get('cost'))} | "
            f"B={fmt_cost(meta.get('b_cost'))} | "
            f"C={fmt_cost(meta.get('c_cost'))} | "
            f"Total={fmt_cost(meta.get('total_cost'))}"
        )
    return lines


def render_budget_by_role(meta: dict) -> list[str]:
    """Render independent role limits/statuses when per-role budgeting is active."""
    budgets = meta.get("cost_budget_by_role") or {}
    return [
        f"- {role.upper()} budget: status={budget.get('status', 'unknown')} | "
        f"limit={fmt_cost(budget.get('limit_usd'))} | "
        f"observed={fmt_cost(budget.get('observed_usd'))} | "
        f"overshoot={fmt_cost(budget.get('overshoot_usd'))}"
        for role, budget in sorted(budgets.items())
    ]


def _role_token_keys(role: str) -> tuple[str, str]:
    # Bare input_tokens/output_tokens are A's; other roles use prefixed keys.
    if role == "a":
        return "input_tokens", "output_tokens"
    return f"{role}_input_tokens", f"{role}_output_tokens"


def render_role_table(meta: dict) -> list[str]:
    """Per-role cost/calls/tokens table from meta (empty if no cost_by_role)."""
    cost_by_role = meta.get("cost_by_role") or {}
    if not cost_by_role:
        return []
    calls_by_role = meta.get("model_calls_by_role") or {}
    lines = [
        "| Role | Cost | Model calls | Input tokens | Output tokens |",
        "|------|------|-------------|--------------|---------------|",
    ]
    for role in sorted(cost_by_role):
        in_key, out_key = _role_token_keys(role)
        lines.append(
            f"| {role.upper()} | {fmt_cost(cost_by_role.get(role))} "
            f"| {calls_by_role.get(role, '—')} "
            f"| {fmt_tokens(meta.get(in_key))} | {fmt_tokens(meta.get(out_key))} |"
        )
    return lines


def _fmt_flag(value) -> str:
    if value is None:
        return "—"
    return "yes" if value else "no"


def _fmt_flag_maybe_proxy(value, is_proxy: bool) -> str:
    """Like _fmt_flag but appends '(proxy)' when the value is a heuristic estimate."""
    base = _fmt_flag(value)
    if is_proxy and value is not None:
        return f"{base} (proxy)"
    return base


def render_agent_stats(inst: InstanceData) -> str:
    """Per-agent tool usage, invocation/verdict summaries, and cost/tokens by role."""
    parts: list[str] = ["\n### Agent Statistics"]
    audit = inst.independent_synthesis_audit or {}
    parts.append(
        f"\n**Independent-synthesis audit:** {audit.get('status', 'unknown')}"
    )
    for item in audit.get("evidence") or []:
        parts.append(
            f"- step {item.get('step')}: `{item.get('tool')}` "
            f"signal `{item.get('signal')}`"
        )

    role_table = render_role_table(inst.meta)
    if role_table:
        parts.append("\n**Cost / tokens by role**\n")
        parts.extend(role_table)

    # Tool usage per agent
    usage_rows: list[tuple[str, Counter]] = []
    a_usage = count_tool_usage(inst.steps)
    if a_usage:
        usage_rows.append(("A", a_usage))
    c_usage = count_role_tool_usage(inst.synthesis_log, "c_tool_calls")
    if c_usage:
        usage_rows.append(("C", c_usage))
    b_usage = count_role_tool_usage(inst.finalization_reviews, "b_tool_calls")
    if b_usage:
        usage_rows.append(("B", b_usage))
    if usage_rows:
        parts.append("\n**Tool usage by agent**\n")
        parts.append("| Agent | Tools used |")
        parts.append("|-------|------------|")
        for label, counter in usage_rows:
            usage = ", ".join(f"{name} ({cnt}x)" for name, cnt in counter.most_common())
            parts.append(f"| {label} | {usage} |")

    # C invocation summary
    if inst.synthesis_log:
        parts.append(f"\n**C invocations: {len(inst.synthesis_log)}**\n")
        parts.append("| # | A-step | Problem | C-steps | Outcome | Crossed / failed gate | Followed | Adopted | Downstream pass |")
        parts.append("|---|--------|---------|---------|---------|----------------------|----------|---------|-----------------|")
        for rec in inst.synthesis_log:
            d = _c_event_detail(inst, rec)
            outcome = (
                f"DEGRADED ({d.get('degraded_reason') or 'unknown'})"
                if d.get("degraded")
                else (d.get("validation_outcome") or "—")
            )
            gates = f"{d.get('crossed_gate') or '—'} / {d.get('failed_gate') or '—'}"
            parts.append(
                f"| {d.get('invocation_index', '?')} | {d.get('a_step', '—')} "
                f"| {_md_cell(d.get('delegated_problem'), 80)} | {d.get('c_steps_used', '—')} "
                f"| {outcome} | {gates} "
                f"| {_fmt_flag_maybe_proxy(d.get('a_followed'), bool(d.get('a_adopted_is_proxy')))} "
                f"| {_fmt_flag_maybe_proxy(d.get('a_adopted_into_testcase'), bool(d.get('a_adopted_is_proxy')))} "
                f"| {_fmt_flag_maybe_proxy(d.get('downstream_passed'), bool(d.get('a_adopted_is_proxy')))} |"
            )

    # B review summary
    if inst.finalization_reviews:
        verdicts = Counter(rec.get("verdict") or "?" for rec in inst.finalization_reviews)
        tally = ", ".join(f"{v} ({n}x)" for v, n in verdicts.most_common())
        parts.append(f"\n**B reviews: {len(inst.finalization_reviews)}** — {tally}\n")
        parts.append("| # | Verdict | Relation | Recommendation | Policy | Block | Evidence steps | Guard | B-steps |")
        parts.append("|---|---------|----------|----------------|--------|-------|----------------|-------|---------|")
        for i, rec in enumerate(inst.finalization_reviews, 1):
            guard = rec.get("rule_guard_decision") or {}
            guard_str = guard.get("decision", "—") if isinstance(guard, dict) else str(guard)
            parts.append(
                f"| {i} | {rec.get('verdict') or '?'} "
                f"| {(rec.get('evidence_relation') or {}).get('relation') or '—'} "
                f"| {rec.get('recommended_verdict') or rec.get('verdict') or '?'} "
                f"| {rec.get('policy_mode') or 'legacy'} | {rec.get('block_index', '—')} "
                f"| {rec.get('evidence_steps') or '—'} | {guard_str} | {rec.get('b_steps_used', '—')} |"
            )

    return "\n".join(parts)


def render_instance(inst: InstanceData, max_chars: int, show_trajectory: bool = True) -> str:
    """Render full analysis for a single instance."""
    parts: list[str] = []

    # Header + result
    parts.append(f"\n## Instance: {inst.instance_id}\n")
    result = result_str(inst.eval_result)
    reason = result_reason(inst.eval_result)
    parts.append(f"### Result: {result}")
    if inst.eval_results_by_grader:
        grader_bits = [
            f"{grader}={result_str(inst.eval_results_by_grader.get(grader))}"
            for grader in ("loose", "caller", "semantic", "strict", "sanitizer", "custom")
            if grader in inst.eval_results_by_grader
        ]
        if grader_bits:
            parts.append(f"**Graders**: {', '.join(grader_bits)}")
    if reason:
        parts.append(f"**Reason**: {reason}\n")

    # Metadata
    meta = inst.meta
    steps_count = meta.get("steps", len(inst.steps))
    in_tok = fmt_tokens(meta_input_tokens(meta))
    out_tok = fmt_tokens(meta_output_tokens(meta))
    cost = fmt_cost(meta_cost(meta))
    tools_available = ", ".join(meta.get("tools", []))
    tools_used = tool_usage_str(inst.steps)

    parts.append("\n### Metadata")
    parts.append(f"- Steps: {steps_count} | Input tokens: {in_tok} | Output tokens: {out_tok} | Cost: {cost}")
    parts.extend(render_cost_by_role(meta))
    if meta.get("cost_budget_enabled"):
        parts.append(
            "- Cost budget: "
            f"status={meta.get('cost_budget_status', 'unknown')} | "
            f"limit={fmt_cost(meta.get('cost_budget_limit_usd'))} | "
            f"observed={fmt_cost(meta.get('cost_budget_observed_usd'))} | "
            f"overshoot={fmt_cost(meta.get('cost_budget_overshoot_usd'))}"
        )
        parts.extend(render_budget_by_role(meta))
    if meta.get("termination_reason"):
        parts.append(f"- Termination reason: `{meta['termination_reason']}`")
    parts.append(f"- Tools available: {tools_available}")
    parts.append(f"- Tools actually used: {tools_used}")

    # B/C/P-aware sections (skip entirely for runs that predate the scaffold)
    if has_scaffold_data(inst):
        parts.append(render_timeline(build_timeline(inst)))
        parts.append(render_agent_stats(inst))

    # Testcase files
    if inst.testcase_files:
        parts.append("\n### Testcase Files")
        file_strs = [f"`{name}` ({fmt_size(size)})" for name, size in inst.testcase_files]
        parts.append(", ".join(file_strs))

    # Trajectory
    if show_trajectory:
        parts.append("\n### Trajectory\n")
        parts.append(render_trajectory(inst.steps, max_chars))

    return "\n".join(parts)


def render_eval_report(
    run: EvalRun, instance_filter: str | None, max_chars: int, show_trajectory: bool
) -> str:
    """Render full analysis report for an eval run."""
    parts: list[str] = []

    # Title
    parts.append(f"# Analysis: {run.name}\n")
    if run.primary_grader:
        parts.append(f"_Primary grader view: `{run.primary_grader}` via `{run.report_label}`_\n")

    # Config section
    if run.config:
        model = run.config.get("model", {}).get("model_id", "?")
        prompt = run.config.get("task", {}).get("prompt_template", "?")
        tools = ", ".join(run.config.get("agent", {}).get("tools", []))
        max_steps = run.config.get("agent", {}).get("max_steps", "?")
        cost_budget = run.config.get("agent", {}).get("cost_budget") or {}
        task_type = run.config.get("task", {}).get("type", "?")
        parts.append("## Config\n")
        parts.append(f"- Model: {model}")
        parts.append(f"- Task type: {task_type}")
        parts.append(f"- Prompt: {prompt}")
        parts.append(f"- Tools: {tools}")
        parts.append(f"- Max steps: {max_steps}")
        if cost_budget.get("enabled"):
            if cost_budget.get("mode") == "per_role":
                role_caps = " | ".join(
                    f"{role.upper()}={fmt_cost(role_cfg.get('max_total_cost_usd'))}"
                    for role, role_cfg in sorted((cost_budget.get("roles") or {}).items())
                )
                parts.append(f"- Per-role instance cost caps: {role_caps}")
            else:
                parts.append(f"- Instance cost cap: {fmt_cost(cost_budget.get('max_total_cost_usd', 10.0))}")

    # Filter instances if requested
    instances = run.instances
    if instance_filter:
        instances = [i for i in instances if i.instance_id == instance_filter]

    # Summary table
    if not instance_filter:
        total = len(instances)
        passed = sum(1 for i in instances if i.eval_result and i.eval_result.get("success"))
        failed = total - passed
        total_in = sum(meta_input_tokens(i.meta) or 0 for i in instances)
        total_out = sum(meta_output_tokens(i.meta) or 0 for i in instances)
        cost_values = [meta_cost(i.meta) for i in instances]
        total_cost = sum(float(value) for value in cost_values if value is not None)
        avg_steps = sum(i.meta.get("steps", 0) for i in instances) / max(total, 1)
        budget_terminated_count = sum(1 for i in instances if budget_terminated(i.meta))

        parts.append("\n## Summary\n")
        parts.append("| Metric | Value |")
        parts.append("|--------|-------|")
        parts.append(f"| Instances | {total} |")
        pct = f"{passed / total * 100:.0f}%" if total else "—"
        parts.append(f"| Passed | {passed} ({pct}) |")
        parts.append(f"| Failed | {failed} |")
        parts.append(f"| Avg steps | {avg_steps:.1f} |")
        parts.append(f"| Total tokens | {fmt_tokens(total_in)} in / {fmt_tokens(total_out)} out |")
        if any(value is not None for value in cost_values):
            parts.append(f"| Total cost | {fmt_cost(total_cost)} |")
        role_names = sorted(
            {
                role
                for instance in instances
                for role in (instance.meta.get("cost_by_role") or {})
            }
        )
        role_totals = {
            role: sum(
                float((instance.meta.get("cost_by_role") or {}).get(role, 0.0) or 0.0)
                for instance in instances
            )
            for role in role_names
        }
        for role in role_names:
            parts.append(f"| {role.upper()} cost | {fmt_cost(role_totals[role])} |")
        if "b" in role_totals or "c" in role_totals:
            parts.append(
                f"| B+C overhead cost | {fmt_cost(role_totals.get('b', 0.0) + role_totals.get('c', 0.0))} |"
            )
        if budget_terminated_count:
            parts.append(f"| Budget-terminated | {budget_terminated_count} |")

        if run.available_graders:
            parts.append("\n## Grader Summary\n")
            parts.append("| Grader | Passed | Failed |")
            parts.append("|--------|--------|--------|")
            for grader in run.available_graders:
                grader_passed = sum(
                    1 for i in instances if i.eval_results_by_grader.get(grader, {}).get("success")
                )
                grader_failed = sum(1 for i in instances if grader in i.eval_results_by_grader) - grader_passed
                parts.append(f"| {grader} | {grader_passed} | {grader_failed} |")

        # Results table
        parts.append("\n## Results\n")
        grader_columns = [
            grader for grader in ("loose", "caller", "semantic", "strict", "sanitizer", "custom")
            if grader in run.available_graders
        ]
        if grader_columns:
            header = "| Instance | " + " | ".join(grader_columns) + " | Steps | Tokens (in/out) | Cost | Tools Used |"
            sep = "|----------|" + "|".join(["--------"] * len(grader_columns)) + "|-------|-----------------|------|------------|"
            parts.append(header)
            parts.append(sep)
        else:
            parts.append("| Instance | Result | Steps | Tokens (in/out) | Cost | Tools Used |")
            parts.append("|----------|--------|-------|-----------------|------|------------|")
        for inst in instances:
            steps = inst.meta.get("steps", len(inst.steps))
            in_t = fmt_tokens(meta_input_tokens(inst.meta))
            out_t = fmt_tokens(meta_output_tokens(inst.meta))
            cost = fmt_cost(meta_cost(inst.meta))
            tools = tool_usage_str(inst.steps)
            if grader_columns:
                grader_cells = [grader_result_str(inst.eval_results_by_grader, grader) for grader in grader_columns]
                parts.append(
                    f"| {inst.instance_id} | "
                    + " | ".join(grader_cells)
                    + f" | {steps} | {in_t} / {out_t} | {cost} | {tools} |"
                )
            else:
                res = result_str(inst.eval_result)
                parts.append(f"| {inst.instance_id} | {res} | {steps} | {in_t} / {out_t} | {cost} | {tools} |")

    # Per-instance sections
    for inst in instances:
        parts.append(render_instance(inst, max_chars, show_trajectory))

    return "\n".join(parts)


def render_comparison(runs: list[EvalRun]) -> str:
    """Render cross-config comparison report."""
    parts: list[str] = []
    parts.append(f"# Comparison: {len(runs)} eval runs\n")

    # Collect all instance IDs
    all_ids: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for inst in run.instances:
            if inst.instance_id not in seen:
                all_ids.append(inst.instance_id)
                seen.add(inst.instance_id)

    # Short run names (strip common prefix/timestamp for readability)
    run_names = [run.name for run in runs]

    # Build lookup: run_name -> instance_id -> InstanceData
    lookup: dict[str, dict[str, InstanceData]] = {}
    for run in runs:
        lookup[run.name] = {i.instance_id: i for i in run.instances}

    # Pass/Fail matrix
    parts.append("## Pass/Fail Matrix\n")
    header = "| Instance | " + " | ".join(run_names) + " |"
    sep = "|----------|" + "|".join(["------"] * len(runs)) + "|"
    parts.append(header)
    parts.append(sep)
    for iid in all_ids:
        cells = []
        for rn in run_names:
            inst = lookup.get(rn, {}).get(iid)
            if inst:
                cells.append(result_str(inst.eval_result))
            else:
                cells.append("—")
        parts.append(f"| {iid} | " + " | ".join(cells) + " |")

    # Aggregate metrics
    parts.append("\n## Aggregate Metrics\n")
    parts.append("| Config | Pass Rate | Avg Steps | Avg Tokens (in) | Avg Tokens (out) | Avg Cost |")
    parts.append("|--------|-----------|-----------|-----------------|------------------|----------|")
    for run in runs:
        total = len(run.instances)
        passed = sum(1 for i in run.instances if i.eval_result and i.eval_result.get("success"))
        pct = f"{passed}/{total} ({passed / total * 100:.0f}%)" if total else "—"
        avg_steps = sum(i.meta.get("steps", 0) for i in run.instances) / max(total, 1)
        avg_in = sum(meta_input_tokens(i.meta) or 0 for i in run.instances) / max(total, 1)
        avg_out = sum(meta_output_tokens(i.meta) or 0 for i in run.instances) / max(total, 1)
        avg_cost = sum(float(meta_cost(i.meta) or 0.0) for i in run.instances) / max(total, 1)
        parts.append(
            f"| {run.name} | {pct} | {avg_steps:.1f} | "
            f"{fmt_tokens(int(avg_in))} | {fmt_tokens(int(avg_out))} | {fmt_cost(avg_cost)} |"
        )

    # Tool usage comparison
    all_tools: set[str] = set()
    run_tool_counts: dict[str, Counter] = {}
    for run in runs:
        combined: Counter = Counter()
        for inst in run.instances:
            combined += count_tool_usage(inst.steps)
        run_tool_counts[run.name] = combined
        all_tools |= set(combined.keys())

    tool_list = sorted(all_tools - {"final_answer"})
    if tool_list:
        parts.append("\n## Tool Usage (total calls across all instances)\n")
        header = "| Config | " + " | ".join(tool_list) + " |"
        sep = "|--------|" + "|".join(["---"] * len(tool_list)) + "|"
        parts.append(header)
        parts.append(sep)
        for run in runs:
            cells = [str(run_tool_counts[run.name].get(t, 0)) for t in tool_list]
            parts.append(f"| {run.name} | " + " | ".join(cells) + " |")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Digest (compact per-instance summary for LLM interpretation — see
# .claude/skills/interpret-run/SKILL.md)
# ---------------------------------------------------------------------------

DIGEST_VOCAB_VERSION = "v1"


def _head(text, max_chars: int) -> str | None:
    """Whitespace-collapsed prefix of *text*, or None when empty."""
    if text is None:
        return None
    s = re.sub(r"\s+", " ", str(text)).strip()
    if not s:
        return None
    return s[:max_chars] + ("..." if len(s) > max_chars else "")


def _primary_arg(tool: str, args: object) -> str:
    if isinstance(args, dict):
        for key in ("command", "code", "answer", "blocker_summary", "artifact_status", "path"):
            if args.get(key):
                return str(args[key])
        return json.dumps(args)
    if isinstance(args, str):
        return args
    return json.dumps(args)


def condense_step(step: Step) -> dict:
    tools = step_tool_names(step)
    primary_args = step.all_tool_calls[0].args if step.all_tool_calls else step.tool_args
    error = None
    if step.error:
        err = step.error
        error = _head(err.get("message", str(err)) if isinstance(err, dict) else str(err), 120)
    return {
        "step": step.step_number,
        "tools": tools,
        "args_summary": _head(_primary_arg(tools[0], primary_args), 120),
        "obs_head": _head(step.observations, 200),
        "error": error,
        "is_final": step.is_final_answer,
    }


def _gate_progress(inst: InstanceData) -> dict:
    """Deepest canonical gate reached across C emissions and reviewer repros."""
    canonical_rank = {gate: i for i, gate in enumerate(GATE_LADDER)}
    deepest = None
    deepest_rank = -1
    raw_gates: list[str] = []
    for rec in inst.synthesis_log:
        payload = rec.get("emitted_payload") or {}
        gate = payload.get("crossed_gate")
        if not gate:
            continue
        raw_gates.append(gate)
        rank = canonical_rank.get(gate, -1)
        if rank > deepest_rank:
            deepest_rank = rank
            deepest = gate
    for rec in inst.finalization_reviews:
        validation = (rec.get("repro_result") or {}).get("validation")
        gate = (
            "target_frame_seen"
            if validation in {"caller", "semantic", "strict"}
            else "target_family_seen"
            if validation == "loose"
            else None
        )
        rank = canonical_rank.get(gate, -1)
        if gate and rank > deepest_rank:
            deepest_rank = rank
            deepest = gate
    guard_gate = None
    if inst.finalization_guard:
        guard_gate = inst.finalization_guard[-1].get("latest_gate") or None
    return {
        "gate_ladder": GATE_LADDER,
        "deepest_canonical_gate": deepest,
        "crossed_gates_raw": raw_gates,
        "guard_latest_gate": guard_gate,
    }


def condense_role_calls(calls: list) -> list[dict]:
    """Condense a B/C `*_tool_calls` list ({step, tool, args_summary, obs_head?}).

    `obs_head` exists only on runs at/after the scaffold logging change; older runs
    yield action-only entries (obs_head absent → None).
    """
    out: list[dict] = []
    if not isinstance(calls, list):
        return out
    for c in calls:
        if not isinstance(c, dict):
            continue
        out.append(
            {
                "step": c.get("step"),
                "tool": c.get("tool"),
                "args_summary": _head(c.get("args_summary"), 120),
                "obs_head": _head(c.get("obs_head"), 200),
            }
        )
    return out


def _digest_c_invocations(inst: InstanceData) -> list[dict]:
    out = []
    for rec in inst.synthesis_log:
        d = _c_event_detail(inst, rec)
        payload = rec.get("emitted_payload") or {}
        d["parse_status"] = rec.get("parse_status")
        d["emit_error"] = _head(rec.get("emit_error"), 200)
        d["notes"] = _head(payload.get("notes"), 280)
        d["how_to_run"] = _head(payload.get("how_to_run"), 200)
        d["trajectory"] = condense_role_calls(rec.get("c_tool_calls"))
        out.append(d)
    return out


def _digest_b_reviews(inst: InstanceData) -> list[dict]:
    submission_steps = final_submission_steps(inst.steps)
    out = []
    for i, rec in enumerate(inst.finalization_reviews):
        guard = rec.get("rule_guard_decision") or {}
        out.append(
            {
                "a_step": submission_steps[i] if i < len(submission_steps) else None,
                "verdict": rec.get("verdict"),
                "proposed_verdict": rec.get("proposed_verdict") or rec.get("verdict"),
                "recommended_verdict": rec.get("recommended_verdict") or rec.get("verdict"),
                "policy_mode": rec.get("policy_mode") or "legacy",
                "policy_applied": bool(rec.get("policy_applied")),
                "evidence_relation": rec.get("evidence_relation") or None,
                "replay_consistency": rec.get("replay_consistency") or None,
                "challenge": rec.get("challenge") or None,
                "block_index": rec.get("block_index"),
                "evidence_steps": rec.get("evidence_steps"),
                "stop_reason": rec.get("stop_reason"),
                "remaining_steps": rec.get("remaining_steps"),
                "degraded": rec.get("degraded"),
                "degraded_reason": rec.get("degraded_reason") or None,
                "reasoning": _head(rec.get("reasoning"), 300),
                "attached_action": _head(rec.get("attached_action"), 300),
                "c_synthesis_assessment": _head(rec.get("c_synthesis_assessment"), 300),
                "rule_guard_decision": guard if isinstance(guard, dict) else {"decision": str(guard)},
                "b_steps_used": rec.get("b_steps_used"),
                "trajectory": condense_role_calls(rec.get("b_tool_calls")),
            }
        )
    return out


def build_digest(inst: InstanceData) -> dict:
    meta = inst.meta
    return {
        "vocab_version": DIGEST_VOCAB_VERSION,
        "instance_id": inst.instance_id,
        "instance_dir": str(inst.path),
        "task_context": inst.task_context,
        "result": {
            "primary": {
                "result": result_str(inst.eval_result),
                "reason": result_reason(inst.eval_result) or None,
            },
            "by_grader": {
                grader: {
                    "result": result_str(res),
                    "reason": res.get("reason") or None,
                }
                for grader, res in inst.eval_results_by_grader.items()
            },
        },
        "meta_summary": {
            "steps": meta.get("steps", len(inst.steps)),
            "model": meta.get("model"),
            "total_cost": meta_cost(meta),
            "total_input_tokens": meta_input_tokens(meta),
            "total_output_tokens": meta_output_tokens(meta),
            "cost_by_role": meta.get("cost_by_role"),
            "model_calls_by_role": meta.get("model_calls_by_role"),
            "termination_reason": meta.get("termination_reason"),
            "cost_budget_status": meta.get("cost_budget_status"),
            "cost_budget_mode": meta.get("cost_budget_mode"),
            "cost_budget_by_role": meta.get("cost_budget_by_role"),
        },
        "plan": inst.plan,
        "independent_synthesis_audit": inst.independent_synthesis_audit,
        "timeline": [
            {"kind": e.kind, "a_step": e.a_step, "summary": e.summary, **e.detail}
            for e in build_timeline(inst)
        ],
        "c_invocations": _digest_c_invocations(inst),
        "b_reviews": _digest_b_reviews(inst),
        "gate_progress": _gate_progress(inst),
        "steps": [condense_step(s) for s in inst.steps],
    }


def _md_cell(value, max_chars: int = 120) -> str:
    s = _head(value, max_chars)
    if not s:
        return "—"
    return s.replace("|", "\\|")


def _render_role_trajectory(parts: list[str], label: str, calls: list) -> None:
    """Render a condensed B/C trajectory table (actions; observations if logged)."""
    if not calls:
        return
    has_obs = any(c.get("obs_head") for c in calls)
    parts.append(f"- {label}:")
    if has_obs:
        parts.append("  | Step | Tool | Args | Observation |")
        parts.append("  |------|------|------|-------------|")
        for c in calls:
            parts.append(
                f"  | {c.get('step', '?')} | {c.get('tool') or '?'} "
                f"| {_md_cell(c.get('args_summary'))} | {_md_cell(c.get('obs_head'), 200)} |"
            )
    else:
        # Older runs: actions only (no observations logged).
        seq = " → ".join(f"{c.get('tool') or '?'}" for c in calls)
        parts.append(f"  {seq}")


def render_digest_md(digest: dict) -> str:
    parts: list[str] = [f"# Digest: {digest['instance_id']}\n"]

    res = digest["result"]
    parts.append(f"**Result**: {res['primary']['result']}")
    if res["by_grader"]:
        grader_bits = [f"{g}={r['result']}" for g, r in res["by_grader"].items()]
        parts.append(f"**Graders**: {', '.join(grader_bits)}")
    if res["primary"]["reason"]:
        parts.append(f"**Reason**: {res['primary']['reason']}")

    tc = digest.get("task_context")
    if tc:
        parts.append("\n## Task context (ground truth)")
        ident = " | ".join(
            f"{k}={tc[k]}" for k in ("project_name", "repo", "lang", "sanitizer", "expected_exit_code")
            if tc.get(k) is not None
        )
        if ident:
            parts.append(f"- {ident}")
        if tc.get("bug_description"):
            parts.append(f"- **Bug**: {tc['bug_description']}")
        if tc.get("sanitizer_report"):
            parts.append(f"- **Sanitizer report (head)**:\n```\n{tc['sanitizer_report']}\n```")
        if tc.get("patch"):
            parts.append(f"- **Ground-truth patch** (the exact condition the PoC must trigger):\n```diff\n{tc['patch']}\n```")

    m = digest["meta_summary"]
    parts.append("\n## Meta")
    parts.append(
        f"- Steps: {m.get('steps')} | Cost: {fmt_cost(m.get('total_cost'))} "
        f"| Tokens: {fmt_tokens(m.get('total_input_tokens'))} in / {fmt_tokens(m.get('total_output_tokens'))} out"
    )
    if m.get("cost_by_role"):
        parts.append(
            "- Cost by role: "
            + " | ".join(f"{r}={fmt_cost(v)}" for r, v in sorted(m["cost_by_role"].items()))
        )
    if m.get("model_calls_by_role"):
        parts.append(
            "- Model calls by role: "
            + " | ".join(f"{r}={v}" for r, v in sorted(m["model_calls_by_role"].items()))
        )
    if m.get("termination_reason"):
        parts.append(f"- Termination: `{m['termination_reason']}`")
    if m.get("cost_budget_status"):
        parts.append(f"- Budget status: {m['cost_budget_status']}")
    parts.extend(render_budget_by_role(m))

    if digest.get("plan"):
        p = digest["plan"]
        parts.append("\n## P plan")
        parts.append(
            f"- P-steps: {p.get('p_steps_used', '—')} | degraded: {p.get('degraded', '—')}"
            + (f" ({p['degraded_reason']})" if p.get("degraded_reason") else "")
        )

    audit = digest.get("independent_synthesis_audit") or {}
    parts.append("\n## Independent synthesis audit")
    parts.append(f"- Status: `{audit.get('status', 'unknown')}`")
    for item in audit.get("evidence") or []:
        parts.append(
            f"- Step {item.get('step')}: `{item.get('tool')}` matched "
            f"`{item.get('signal')}`"
        )

    if digest["timeline"]:
        parts.append("\n## Timeline")
        for e in digest["timeline"]:
            parts.append(f"- {e['summary']}")

    if digest["c_invocations"]:
        parts.append("\n## C invocations")
        for d in digest["c_invocations"]:
            outcome = (
                f"DEGRADED ({d.get('degraded_reason') or 'unknown'})"
                if d.get("degraded")
                else (d.get("validation_outcome") or "—")
            )
            parts.append(
                f"\n### C#{d.get('invocation_index', '?')} @step{d.get('a_step', '?')} — "
                f"{outcome}"
            )
            if d.get("delegated_problem"):
                parts.append(f"- Delegated problem: {d['delegated_problem']}")
            parts.append(
                f"- C-steps: {d.get('c_steps_used', '—')} | crossed: {d.get('crossed_gate') or '—'} "
                f"| failed: {d.get('failed_gate') or '—'}"
            )
            _is_proxy = bool(d.get("a_adopted_is_proxy"))
            parts.append(
                f"- Adoption: followed={_fmt_flag_maybe_proxy(d.get('a_followed'), _is_proxy)}, "
                f"adopted={_fmt_flag_maybe_proxy(d.get('a_adopted_into_testcase'), _is_proxy)}, "
                f"downstream_passed={_fmt_flag_maybe_proxy(d.get('downstream_passed'), _is_proxy)}"
            )
            if d.get("helper_files"):
                parts.append(f"- Helper files: {', '.join(d['helper_files'])}")
            if d.get("emit_error"):
                parts.append(f"- Emit error: {d['emit_error']}")
            if d.get("notes"):
                parts.append(f"- Notes: {d['notes']}")
            _render_role_trajectory(parts, "C trajectory", d.get("trajectory"))

    if digest["b_reviews"]:
        parts.append("\n## B reviews")
        for i, r in enumerate(digest["b_reviews"], 1):
            step = f"@step{r['a_step']}" if r.get("a_step") is not None else "(unanchored)"
            parts.append(f"\n### B review {i} {step} — {r.get('verdict') or '?'}")
            parts.append(
                f"- Block: {r.get('block_index', '—')} | evidence: {r.get('evidence_steps') or '—'} "
                f"| guard: {(r.get('rule_guard_decision') or {}).get('decision', '—')} "
                f"| remaining A-steps: {r.get('remaining_steps', '—')}"
            )
            if r.get("reasoning"):
                parts.append(f"- Reasoning: {r['reasoning']}")
            if r.get("attached_action"):
                parts.append(f"- Attached action: {r['attached_action']}")
            if r.get("c_synthesis_assessment"):
                parts.append(f"- C assessment: {r['c_synthesis_assessment']}")
            _render_role_trajectory(parts, "B trajectory", r.get("trajectory"))

    gp = digest["gate_progress"]
    if gp.get("crossed_gates_raw") or gp.get("guard_latest_gate"):
        parts.append("\n## Gate progress")
        parts.append(f"- Deepest canonical gate: {gp.get('deepest_canonical_gate') or '—'}")
        if gp.get("crossed_gates_raw"):
            parts.append(f"- Crossed gates (raw, per C emission): {', '.join(gp['crossed_gates_raw'])}")
        if gp.get("guard_latest_gate"):
            parts.append(f"- Guard latest gate: {gp['guard_latest_gate']}")

    if digest["steps"]:
        parts.append("\n## Condensed A trajectory")
        parts.append("| Step | Tools | Args | Observation | Error |")
        parts.append("|------|-------|------|-------------|-------|")
        for s in digest["steps"]:
            tools = ", ".join(s["tools"])
            final_mark = " (final)" if s.get("is_final") else ""
            parts.append(
                f"| {s['step']}{final_mark} | {tools} | {_md_cell(s.get('args_summary'))} "
                f"| {_md_cell(s.get('obs_head'), 200)} | {_md_cell(s.get('error'))} |"
            )

    return "\n".join(parts) + "\n"


def write_digests(instances: list[InstanceData], digest_root: Path | None = None) -> list[Path]:
    """Write digest.json + digest.md per instance. Returns written paths.

    Default: into each instance dir root (legacy layout, where instance dirs are
    eval-dir copies). With digest_root (nested layout): into
    digest_root/<instance_id>/, keeping digests eval-specific while instance
    artifacts stay in the run dir.
    """
    written: list[Path] = []
    for inst in instances:
        if inst.path is None:
            continue
        digest = build_digest(inst)
        target_dir = inst.path if digest_root is None else digest_root / inst.instance_id
        target_dir.mkdir(parents=True, exist_ok=True)
        json_path = target_dir / "digest.json"
        md_path = target_dir / "digest.md"
        json_path.write_text(json.dumps(digest, indent=1, default=str) + "\n")
        md_path.write_text(render_digest_md(digest))
        written.extend([json_path, md_path])
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_configs_dir() -> Path | None:
    """Try to find the configs/ directory relative to this script."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent.parent / "configs",  # <repo_root>/configs
        script_dir / "configs",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_dataset_dir() -> Path | None:
    """Find the SEC-bench dataset data dir (eval-oss/eval-cve jsonl) from the repo root."""
    script_dir = Path(__file__).resolve().parent
    # script_dir = <repo_root>/src/pocharness; this package doesn't vendor the
    # dataset snapshot, so this candidate won't exist unless the caller adds one.
    candidates = [
        script_dir.parent.parent / "references" / "datasets" / "01.SEC-bench" / "data",
    ]
    for c in candidates:
        if (c / "eval-oss.jsonl").exists() or (c / "eval-cve.jsonl").exists():
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze SEC-bench PoC generation run artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eval-dir", type=Path, help="Path to a single eval directory.")
    mode.add_argument("--eval-dirs", type=Path, nargs="+", help="Paths to multiple eval dirs for comparison.")
    mode.add_argument("--run-dir", type=Path, help="Path to a raw run directory (no eval results).")

    parser.add_argument("--instance-id", help="Filter to a single instance.")
    parser.add_argument("--output", "-o", type=Path, help="Write output to file (default: stdout).")
    parser.add_argument("--max-chars", type=int, default=500, help="Max chars for observation previews.")
    parser.add_argument(
        "--full-trajectory",
        action="store_true",
        help="Render the full per-step A trajectory in analysis.md (off by default; the digest "
        "carries a condensed trajectory and raw trajectory.jsonl has the full detail).",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Deprecated no-op: the full trajectory is already off by default. Use --full-trajectory to include it.",
    )
    parser.add_argument("--configs-dir", type=Path, help="Path to configs directory for config resolution.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Path to the SEC-bench dataset data dir (eval-oss/eval-cve jsonl) for task-context "
        "injection into digests. Not shipped in this package by default; omit to skip.",
    )
    parser.add_argument(
        "--grader",
        choices=["loose", "caller", "semantic", "strict"],
        help="Primary grader view to use for summary/comparison (default: loose if available).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Explicit report JSONL to analyze instead of auto-discovering report_*.jsonl files.",
    )
    parser.add_argument(
        "--no-digest",
        action="store_true",
        help="Skip writing per-instance digest.json/digest.md files.",
    )
    parser.add_argument(
        "--digest-only",
        action="store_true",
        help="Write per-instance digests only; skip the analysis.md report.",
    )
    args = parser.parse_args()

    if args.report_path and args.eval_dirs:
        parser.error("--report-path only supports single --eval-dir mode.")
    if args.no_digest and args.digest_only:
        parser.error("--no-digest and --digest-only are mutually exclusive.")
    if args.digest_only and args.eval_dirs:
        parser.error("--digest-only does not apply to --eval-dirs comparison mode.")

    configs_dir = args.configs_dir or resolve_configs_dir()
    dataset_dir = args.dataset_dir or resolve_dataset_dir()
    dataset_index = build_dataset_index(dataset_dir)
    show_trajectory = args.full_trajectory

    run = None
    if args.eval_dir:
        run = load_eval_dir(args.eval_dir, configs_dir, args.grader, args.report_path, dataset_index)
        output = render_eval_report(run, args.instance_id, args.max_chars, show_trajectory)
    elif args.eval_dirs:
        runs = [load_eval_dir(d, configs_dir, args.grader) for d in args.eval_dirs]
        output = render_comparison(runs)
    elif args.run_dir:
        run = load_run_dir(args.run_dir, configs_dir, dataset_index)
        output = render_eval_report(run, args.instance_id, args.max_chars, show_trajectory)
    else:
        parser.error("One of --eval-dir, --eval-dirs, or --run-dir is required.")
        return 1

    # Per-instance digests (single-run modes only; comparison mode untouched)
    if run is not None and not args.no_digest:
        digest_instances = run.instances
        if args.instance_id:
            digest_instances = [i for i in digest_instances if i.instance_id == args.instance_id]
        # Nested layout: instances live in the run dir, but digests are eval-specific,
        # so they land under the eval dir.
        digest_root = None
        if args.eval_dir and resolve_instance_root(args.eval_dir) != args.eval_dir:
            digest_root = args.eval_dir
        written = write_digests(digest_instances, digest_root)
        if written:
            print(f"Digests written: {len(written) // 2} instance(s)")

    if args.digest_only:
        return 0

    # Determine output path: explicit --output, or auto-generate inside the input dir
    output_path = args.output
    if output_path is None:
        input_dir = args.eval_dir or args.run_dir
        if input_dir:
            output_path = input_dir / "analysis.md"
        # For --eval-dirs comparison mode, default to stdout
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n")
        print(f"Report written to: {output_path}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
