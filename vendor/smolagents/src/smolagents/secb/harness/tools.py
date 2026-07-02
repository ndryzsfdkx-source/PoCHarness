"""Synthesis Helper internal tools."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smolagents.secb.sanitizer.parser import grade_sanitizer_outputs
from smolagents.secb.trajectory.format import truncate
from smolagents.secb.harness.artifact_guard import _load_shape
from smolagents.secb.harness.config import SynthesisAgentBackendConfig
from smolagents.secb.harness.workspace import WorkspaceInspector
from smolagents.tools import Tool


HARNESS_SHAPE_PATH = "/workspace/harness_shape.json"
DEFAULT_TESTCASE_ROOT = Path("/testcase")


@dataclass
class SynthesisToolState:
    """Mutable state shared across all C-internal tools for one invocation."""

    bundle: dict[str, Any]
    work_dir: str
    backend_config: SynthesisAgentBackendConfig
    helper_repro_runs_used: int = 0
    helper_files_written: list[str] = field(default_factory=list)
    latest_repro_result: dict[str, Any] | None = None
    emitted_payload: dict[str, Any] | None = None
    raw_emit_payload: str = ""
    emit_called: bool = False
    emit_error: str = ""
    emit_attempts: int = 0
    raw_terminal_observation: str = ""


def _helpers_dir(state: SynthesisToolState) -> Path:
    """Resolve <work_dir>/Helpers/ (the canonical write anchor for C)."""
    return Path(state.work_dir) / "Helpers"


# ---------------------------------------------------------------------------
# Tools


class ReadTestcaseTool(Tool):
    name = "read_testcase"
    description = (
        "Read the PoC Solver's failed candidate testcase. Returns text or hex based on `mode`. "
        "Use `mode='hex'` for binary formats and `mode='strings'` to extract printable runs."
    )
    inputs = {
        "path": {"type": "string", "description": "Absolute path of the candidate (e.g. /testcase/poc)."},
        "max_bytes": {"type": "integer", "description": "Byte cap; defaults to workspace.read_max_chars.", "nullable": True},
        "mode": {"type": "string", "description": "One of: raw, hex, strings.", "nullable": True},
    }
    output_type = "string"

    def __init__(self, state: SynthesisToolState, inspector: WorkspaceInspector, **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._inspector = inspector

    def forward(self, path: str, max_bytes: int | None = None, mode: str | None = None) -> str:
        cap = int(max_bytes or self._inspector.config.read_max_chars)
        normalized_mode = (mode or "raw").lower()
        if normalized_mode not in {"raw", "hex", "strings"}:
            return "ERROR: mode must be one of: raw, hex, strings"
        data, error = self._inspector.read_bytes(path, max_bytes=cap)
        if error:
            return error
        assert data is not None
        if normalized_mode == "raw":
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="replace")
            return f"path={path} bytes={len(data)}\n{truncate(text, cap)}"
        if normalized_mode == "hex":
            hex_lines = []
            for offset in range(0, len(data), 16):
                chunk = data[offset : offset + 16]
                hex_part = " ".join(f"{b:02x}" for b in chunk)
                ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
                hex_lines.append(f"{offset:08x}  {hex_part:<47}  {ascii_part}")
            return f"path={path} bytes={len(data)}\n" + "\n".join(hex_lines)
        # strings mode
        runs: list[str] = []
        current: list[int] = []
        for b in data:
            if 32 <= b < 127:
                current.append(b)
            else:
                if len(current) >= 4:
                    runs.append(bytes(current).decode("ascii", errors="replace"))
                current = []
        if len(current) >= 4:
            runs.append(bytes(current).decode("ascii", errors="replace"))
        body = "\n".join(runs) if runs else "(no printable runs >=4 chars)"
        return f"path={path} bytes={len(data)} runs={len(runs)}\n{truncate(body, cap)}"


class ReadReproOutputTool(Tool):
    name = "read_repro_output"
    description = (
        "Return the full `secb repro` output that the PoC Solver captured "
        "for the failed candidate."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, state: SynthesisToolState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def forward(self) -> str:
        return str(self._state.bundle.get("repro_output") or "(no repro_output provided)")


class ReadHarnessShapeTool(Tool):
    name = "read_harness_shape"
    description = (
        "Read the cached harness shape JSON (expected testcase filename, build/repro "
        "commands, primary binary, argv shape) written by the startup probe."
    )
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        path = Path(HARNESS_SHAPE_PATH)
        if not path.exists():
            return f"ERROR: {HARNESS_SHAPE_PATH} not found (harness shape probe did not run)."
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"ERROR: failed to read {path}: {exc}"


class FindWorkspacePathsTool(Tool):
    name = "find_workspace_paths"
    description = "Discover allow-listed files or directories under a workspace root."
    inputs = {
        "root_path": {"type": "string", "description": "Absolute allow-listed directory to search under."},
        "name_pattern": {"type": "string", "description": "Glob for the basename, e.g. '*.c'.", "nullable": True},
        "file_type": {"type": "string", "description": "One of: any, file, dir.", "nullable": True},
        "executable_only": {"type": "boolean", "description": "Only executable files.", "nullable": True},
        "max_depth": {"type": "integer", "description": "Maximum directory depth.", "nullable": True},
        "max_matches": {"type": "integer", "description": "Maximum matches to return.", "nullable": True},
    }
    output_type = "string"

    def __init__(self, inspector: WorkspaceInspector, **kwargs):
        super().__init__(**kwargs)
        self._inspector = inspector

    def forward(
        self,
        root_path: str,
        name_pattern: str | None = None,
        file_type: str | None = None,
        executable_only: bool | None = False,
        max_depth: int | None = None,
        max_matches: int | None = None,
    ) -> str:
        return self._inspector.find_paths(
            root_path,
            name_pattern=name_pattern,
            file_type=file_type or "any",
            executable_only=bool(executable_only),
            max_depth=max_depth,
            max_matches=max_matches,
        )


class ReadWorkspaceFileTool(Tool):
    name = "read_workspace_file"
    description = "Read one allow-listed workspace file."
    inputs = {
        "path": {"type": "string", "description": "Allow-listed file path."},
        "max_chars": {"type": "integer", "description": "Maximum characters.", "nullable": True},
    }
    output_type = "string"

    def __init__(self, inspector: WorkspaceInspector, **kwargs):
        super().__init__(**kwargs)
        self._inspector = inspector

    def forward(self, path: str, max_chars: int | None = None) -> str:
        return self._inspector.read_file(path, max_chars=max_chars)


class GrepSourceTool(Tool):
    name = "grep_source"
    description = "Run read-only grep over an allow-listed source path."
    inputs = {
        "pattern": {"type": "string", "description": "Literal grep pattern."},
        "path": {"type": "string", "description": "Absolute source path."},
        "max_matches": {"type": "integer", "description": "Max matches.", "nullable": True},
    }
    output_type = "string"

    def __init__(self, inspector: WorkspaceInspector, **kwargs):
        super().__init__(**kwargs)
        self._inspector = inspector

    def forward(self, pattern: str, path: str, max_matches: int | None = None) -> str:
        return self._inspector.grep_source(pattern, path, max_matches=max_matches)


class ReadPriorHelpersTool(Tool):
    name = "read_prior_helpers"
    description = (
        "List files currently under Helpers/ and compact prior invocation context supplied "
        "by the runtime."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, state: SynthesisToolState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def forward(self) -> str:
        helpers = _helpers_dir(self._state)
        listing: list[str] = []
        if helpers.exists() and helpers.is_dir():
            for entry in sorted(helpers.rglob("*")):
                if entry.is_file():
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    listing.append(f"{entry.relative_to(helpers)} ({size}B)")
        else:
            listing.append("(Helpers/ does not exist yet)")

        sections: list[str] = ["Helpers/:"]
        sections.extend(f"  {item}" for item in listing)
        prior = self._state.bundle.get("prior_synthesis") or []
        if prior:
            sections.extend(["", "Prior synthesis:"])
            sections.extend(f"  {json.dumps(item, sort_keys=True)}" for item in prior[-3:])
        return "\n".join(sections)


class WriteHelperTool(Tool):
    name = "write_helper"
    description = (
        "Write a helper file under Helpers/. Filename must begin with 'Helpers/' and be "
        "relative; absolute paths, '..', and glob characters are rejected."
    )
    inputs = {
        "filename": {"type": "string", "description": "Relative path that must start with Helpers/."},
        "content": {"type": "string", "description": "File contents to write (overwrites if exists)."},
    }
    output_type = "string"

    def __init__(self, state: SynthesisToolState, inspector: WorkspaceInspector, **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._inspector = inspector

    def forward(self, filename: str, content: str) -> str:
        resolved, error = self._inspector.resolve_write_path(filename)
        if error:
            return error
        assert resolved is not None
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content or "", encoding="utf-8")
        except Exception as exc:
            return f"ERROR: write failed for {resolved}: {exc}"
        rel = str(resolved.relative_to(Path(self._state.work_dir)))
        if rel not in self._state.helper_files_written:
            self._state.helper_files_written.append(rel)
        return f"OK: wrote {len(content or '')} chars to {rel}"


class RunHelperTool(Tool):
    name = "run_helper"
    description = (
        "Execute a helper from Helpers/ to confirm it is runnable. Captures stdout/stderr "
        "(truncated). The interpreter is inferred from the file's extension (.py -> python3, "
        ".sh -> bash) or directly if marked executable."
    )
    inputs = {
        "filename": {"type": "string", "description": "Relative path under Helpers/."},
        "args": {"type": "array", "description": "Argv tail.", "items": {"type": "string"}, "nullable": True},
        "timeout": {"type": "integer", "description": "Seconds (1-60).", "nullable": True},
    }
    output_type = "string"

    def __init__(self, state: SynthesisToolState, inspector: WorkspaceInspector, **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._inspector = inspector

    def forward(self, filename: str, args: list[str] | None = None, timeout: int | None = None) -> str:
        resolved, error = self._inspector.resolve_write_path(filename)
        if error:
            return error
        assert resolved is not None
        if not resolved.exists() or not resolved.is_file():
            return f"ERROR: helper not found: {resolved}"
        cap_timeout = min(max(int(timeout or 30), 1), 60)
        suffix = resolved.suffix.lower()
        if suffix == ".py":
            command = ["python3", str(resolved), *([str(a) for a in (args or [])])]
        elif suffix in {".sh", ".bash"}:
            command = ["bash", str(resolved), *([str(a) for a in (args or [])])]
        elif os.access(resolved, os.X_OK):
            command = [str(resolved), *([str(a) for a in (args or [])])]
        else:
            return f"ERROR: unsupported helper extension '{suffix}' and not executable: {resolved}"
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=cap_timeout,
                cwd=self._state.work_dir,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: helper timed out after {cap_timeout}s: {' '.join(shlex.quote(c) for c in command)}"
        except FileNotFoundError as exc:
            return f"ERROR: interpreter not found: {exc}"
        cap = self._inspector.config.read_max_chars
        return (
            f"exit_code={completed.returncode}\n"
            f"command={' '.join(shlex.quote(c) for c in command)}\n"
            f"stdout:\n{truncate(completed.stdout or '', cap)}\n"
            f"stderr:\n{truncate(completed.stderr or '', cap)}"
        )


class RunSecbReproForHelperTool(Tool):
    name = "run_secb_repro_for_helper"
    description = (
        "Run `secb repro` against an artifact (typically the helper's output staged at the "
        "harness-expected path)."
    )
    inputs = {
        "artifact_path": {"type": "string", "description": "Path the helper staged for the harness."},
    }
    output_type = "string"

    def __init__(self, state: SynthesisToolState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def forward(self, artifact_path: str) -> str:
        description_mode = bool(self._state.backend_config.description_mode)
        # description_mode: grade against "" regardless of bundle contents -- oracle_output is
        # already forced empty upstream (synthesis.py), but A's self-supplied target_signal is
        # still present in the bundle and must not be used as a grading fallback either. This
        # still yields real sanitizer_family/crash_type/top_frame (parsed from observed output,
        # not the expected side), just no oracle-derived pass/fail.
        target_signal = (
            ""
            if description_mode
            else str(
                self._state.bundle.get("oracle_output")
                or self._state.bundle.get("target_signal")
                or ""
            )
        )

        # Resolve artifact_path (relative paths are anchored to work_dir)
        src = Path(artifact_path)
        if not src.is_absolute():
            src = Path(self._state.work_dir) / src
        if not src.exists() or not src.is_file():
            return f"ERROR: artifact not found: {artifact_path}"

        # Determine the expected testcase filename from harness_shape.json
        shape = _load_shape()
        expected_filename = (shape or {}).get("expected_testcase_filename")
        if not expected_filename:
            return (
                "ERROR: harness_shape.json missing or has no expected_testcase_filename. "
                "Cannot stage artifact to /testcase/."
            )

        dst = DEFAULT_TESTCASE_ROOT / expected_filename

        # Guard: artifact must live under Helpers/, not at the harness path itself
        if src.resolve() == dst.resolve():
            return (
                f"ERROR: artifact_path resolves to the harness dst ({dst}). "
                "Write the artifact under Helpers/ first, then pass that path here."
            )

        # Snapshot the PoC Solver's current candidate so staging is transient.
        # /testcase must reflect the solver's last write after this call returns:
        # _collect_artifacts packages /testcase at end-of-run for grading.
        existed = dst.exists()
        original: bytes | None = None
        if existed:
            try:
                original = dst.read_bytes()
            except Exception as exc:
                return f"ERROR: failed to snapshot the PoC Solver candidate at {dst}: {exc}"

        # Stage C's artifact into /testcase for the duration of the repro call
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
        except Exception as exc:
            return f"ERROR: failed to stage {artifact_path} -> {dst}: {exc}"

        restore_warning = ""
        repro_error = ""
        completed = None
        try:
            completed = subprocess.run(
                ["secb", "repro"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self._state.work_dir,
            )
        except subprocess.TimeoutExpired:
            repro_error = "ERROR: secb repro timed out (30s)."
        except FileNotFoundError:
            repro_error = "ERROR: secb binary not found in PATH."
        finally:
            # Restore the PoC Solver candidate so the /testcase invariant holds.
            # Restore failure must not mask the repro result — append a warning.
            try:
                if existed and original is not None:
                    dst.write_bytes(original)
                else:
                    dst.unlink(missing_ok=True)
            except Exception as exc:
                restore_warning = f"\nWARNING: failed to restore /testcase/{expected_filename}: {exc}"

        if repro_error:
            return repro_error + restore_warning

        assert completed is not None
        self._state.helper_repro_runs_used += 1
        combined = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        grading = grade_sanitizer_outputs(target_signal, combined)
        result = {
            "artifact_path": artifact_path,
            "staged_to": str(dst),
            "exit_code": completed.returncode,
            "validation": grading["highest_passing_grader"],
            "grader_passes": grading["grader_passes"],
            "strict_failure_reason": grading["strict_failure_reason"],
            "sanitizer_family": grading["sanitizer_family"],
            "crash_type": grading["crash_type"],
            "top_frame": grading["top_frame"],
            "stdout": truncate(completed.stdout or "", 2500),
            "stderr": truncate(completed.stderr or "", 2500),
        }
        if restore_warning:
            result["restore_warning"] = restore_warning.strip()
        self._state.latest_repro_result = result
        return json.dumps(result, indent=2)


class EmitHelperTool(Tool):
    name = "emit_helper"
    description = (
        "Terminal: emit the synthesis helper manifest shown to the PoC Solver. "
        "Call exactly once after writing (and optionally running) helper files."
    )
    inputs = {
        "helper_files": {
            "type": "array",
            "description": "Relative paths under Helpers/ that were written this invocation.",
            "items": {"type": "string"},
        },
        "how_to_run": {
            "type": "string",
            "description": "Exact command(s) the PoC Solver should run.",
        },
        "crossed_gate": {
            "type": "string",
            "description": "Optional condition the work demonstrably satisfied.",
            "nullable": True,
        },
        "failed_gate": {
            "type": "string",
            "description": "Optional remaining condition or invariant.",
            "nullable": True,
        },
        "harness_evidence": {
            "type": "string",
            "description": "Optional compact evidence supporting the diagnosis.",
            "nullable": True,
        },
        "notes": {"type": "string", "description": "<=500 chars.", "nullable": True},
    }
    output_type = "string"

    def __init__(self, state: SynthesisToolState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def forward(
        self,
        helper_files: list[str],
        how_to_run: str,
        crossed_gate: str | None = None,
        failed_gate: str | None = None,
        harness_evidence: str | None = None,
        notes: str | None = "",
    ) -> str:
        self._state.emit_called = True
        self._state.emit_attempts += 1
        # Clear any error from a prior rejected attempt so a successful retry is clean.
        self._state.emit_error = ""

        def _reject(message: str) -> str:
            self._state.emit_error = message
            self._state.raw_terminal_observation = message
            return message

        files = [str(f) for f in (helper_files or [])]
        if not files:
            return _reject("ERROR: helper_files must be non-empty. Write at least one helper before emit_helper.")
        work_dir = Path(self._state.work_dir)
        helpers_anchor = (work_dir / "Helpers").resolve(strict=False)
        helpers_prefix = str(helpers_anchor).rstrip("/") + "/"
        validated: list[str] = []
        for rel in files:
            path = (work_dir / rel).resolve(strict=False)
            if not str(path).startswith(helpers_prefix):
                return _reject(f"ERROR: helper_files path must be under Helpers/: {rel}")
            if not path.exists() or not path.is_file():
                return _reject(f"ERROR: helper file not found on disk: {rel}")
            if path.stat().st_size <= 0:
                return _reject(f"ERROR: helper file is empty: {rel}")
            validated.append(rel)
        how_to_run_text = str(how_to_run or "").strip()
        if not how_to_run_text:
            return _reject("ERROR: how_to_run is required.")
        crossed_gate_text = str(crossed_gate or "").strip()[:300]
        failed_gate_text = str(failed_gate or "").strip()[:300]
        harness_evidence_text = str(harness_evidence or "").strip()[:500]

        payload = {
            "helper_files": validated,
            "how_to_run": how_to_run_text,
            "validation": (
                (self._state.latest_repro_result or {}).get("validation") or "not_run"
            ),
            "grader_passes": (
                (self._state.latest_repro_result or {}).get("grader_passes") or {}
            ),
            "strict_failure_reason": (
                (self._state.latest_repro_result or {}).get("strict_failure_reason") or ""
            ),
            "crossed_gate": crossed_gate_text,
            "failed_gate": failed_gate_text,
            "harness_evidence": harness_evidence_text,
            "notes": str(notes or "")[:500],
        }
        self._state.emitted_payload = payload
        self._state.raw_emit_payload = json.dumps(payload)
        message = f"OK: emitted manifest with {len(validated)} helper file(s)."
        self._state.raw_terminal_observation = message
        return message


def build_synthesis_tools(
    *,
    tools_allow: tuple[str, ...],
    state: SynthesisToolState,
    inspector: WorkspaceInspector,
) -> list[Tool]:
    factories = {
        "read_testcase": lambda: ReadTestcaseTool(state, inspector),
        "read_repro_output": lambda: ReadReproOutputTool(state),
        "read_harness_shape": lambda: ReadHarnessShapeTool(),
        "find_workspace_paths": lambda: FindWorkspacePathsTool(inspector),
        "read_workspace_file": lambda: ReadWorkspaceFileTool(inspector),
        "grep_source": lambda: GrepSourceTool(inspector),
        "read_prior_helpers": lambda: ReadPriorHelpersTool(state),
        "write_helper": lambda: WriteHelperTool(state, inspector),
        "run_helper": lambda: RunHelperTool(state, inspector),
        "run_secb_repro_for_helper": lambda: RunSecbReproForHelperTool(state),
        "emit_helper": lambda: EmitHelperTool(state),
    }
    return [factories[name]() for name in tools_allow]
