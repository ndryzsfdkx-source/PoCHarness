"""Workspace inspection for Agent-C: read/grep/find plus a Helpers/ write guard."""
from __future__ import annotations

import fnmatch
import os
import subprocess
from pathlib import Path

from smolagents.secb.trajectory.format import truncate
from smolagents.secb.harness.config import SynthesisWorkspaceConfig


def _normalize_prefix(prefix: str) -> str:
    if prefix == "/":
        return prefix
    return prefix.rstrip("/") + "/"


class WorkspaceInspector:
    """Read/grep/find under allow-listed paths; write only into Helpers/."""

    def __init__(self, config: SynthesisWorkspaceConfig, *, work_dir: str | None = None):
        self.config = config
        self.work_dir = Path(work_dir) if work_dir else None
        self.allow = tuple(_normalize_prefix(item) for item in config.read_allow_paths)
        self.deny = tuple(_normalize_prefix(item) for item in config.read_deny_paths)
        self.write_paths = tuple(_normalize_prefix(item) for item in config.write_paths)

    def _resolve_under_work_dir(self, raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute() or self.work_dir is None:
            return path.resolve(strict=False)
        return (self.work_dir / path).resolve(strict=False)

    def _resolve_allowed_path(self, path: str) -> tuple[Path | None, str | None]:
        raw = str(path or "").strip()
        if not raw:
            return None, "ERROR: path not in allow-list"
        if any(ch in raw for ch in "*?[]"):
            return None, "ERROR: path not in allow-list"
        if "bug_report" in raw:
            return None, "ERROR: path not in allow-list"
        resolved = self._resolve_under_work_dir(raw)
        rendered = str(resolved)
        as_dir = rendered.rstrip("/") + "/"
        if any(as_dir.startswith(prefix) or rendered == prefix.rstrip("/") for prefix in self.deny):
            return None, "ERROR: path not in allow-list"
        # Allow-list match against absolute prefixes or against the relative
        # write_paths (Helpers/), resolved under work_dir.
        absolute_allows = [p for p in self.allow if p.startswith("/")]
        relative_allows = [p for p in self.allow if not p.startswith("/")]
        if any(as_dir.startswith(prefix) or rendered == prefix.rstrip("/") for prefix in absolute_allows):
            return resolved, None
        for prefix in relative_allows:
            if self.work_dir is None:
                continue
            anchor = (self.work_dir / prefix).resolve(strict=False)
            anchor_dir = str(anchor).rstrip("/") + "/"
            if as_dir.startswith(anchor_dir) or rendered == str(anchor):
                return resolved, None
        return None, "ERROR: path not in allow-list"

    def resolve_write_path(self, filename: str) -> tuple[Path | None, str | None]:
        """Resolve a Helpers/-relative filename for write. Rejects absolute paths,
        parent traversal, glob chars, bug_report, and anything outside write_paths."""
        raw = str(filename or "").strip()
        if not raw:
            return None, "ERROR: filename was empty"
        if any(ch in raw for ch in "*?[]"):
            return None, "ERROR: filename contains glob characters"
        if "bug_report" in raw:
            return None, "ERROR: filename not allowed"
        candidate = Path(raw)
        if candidate.is_absolute():
            return None, "ERROR: filename must be relative (under Helpers/)"
        if ".." in candidate.parts:
            return None, "ERROR: filename contains '..'"
        if self.work_dir is None:
            return None, "ERROR: work_dir unset; cannot resolve Helpers/"
        # If the filename does not already start with a write_paths prefix,
        # require Helpers/ to be the first component.
        first = candidate.parts[0] if candidate.parts else ""
        if (first + "/") not in self.write_paths:
            return None, f"ERROR: filename must start with one of: {', '.join(self.write_paths)}"
        resolved = (self.work_dir / candidate).resolve(strict=False)
        # Confirm the resolved path is still under an allowed write anchor.
        as_dir = str(resolved).rstrip("/") + "/"
        for prefix in self.write_paths:
            anchor = (self.work_dir / prefix).resolve(strict=False)
            anchor_dir = str(anchor).rstrip("/") + "/"
            if as_dir.startswith(anchor_dir):
                return resolved, None
        return None, "ERROR: resolved path outside write_paths"

    def read_file(self, path: str, max_chars: int | None = None) -> str:
        resolved, error = self._resolve_allowed_path(path)
        if error:
            return error
        assert resolved is not None
        if not resolved.exists() or not resolved.is_file():
            return f"ERROR: file not found: {resolved}"
        cap = min(max(int(max_chars or self.config.read_max_chars), 1), 32000)
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"ERROR: read failed for {resolved}: {exc}"
        return f"path={resolved}\n{truncate(text, cap)}"

    def read_bytes(self, path: str, max_bytes: int | None = None) -> tuple[bytes | None, str | None]:
        resolved, error = self._resolve_allowed_path(path)
        if error:
            return None, error
        assert resolved is not None
        if not resolved.exists() or not resolved.is_file():
            return None, f"ERROR: file not found: {resolved}"
        cap = min(max(int(max_bytes or self.config.read_max_chars), 1), 65536)
        try:
            with resolved.open("rb") as f:
                return f.read(cap), None
        except Exception as exc:
            return None, f"ERROR: read failed for {resolved}: {exc}"

    def grep_source(self, pattern: str, path: str, max_matches: int | None = None) -> str:
        resolved, error = self._resolve_allowed_path(path)
        if error:
            return error
        assert resolved is not None
        if not resolved.exists():
            return f"ERROR: path not found: {resolved}"
        pattern_text = str(pattern or "")
        if not pattern_text:
            return "ERROR: pattern was empty"
        match_cap = min(max(int(max_matches or self.config.grep_max_matches), 1), 200)
        command = ["grep", "-rn", "--binary-files=without-match", "-m", str(match_cap), pattern_text, str(resolved)]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(int(self.config.grep_timeout_secs), 1),
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: grep timed out for {resolved}"
        except FileNotFoundError:
            return "ERROR: grep not available"
        output = completed.stdout if completed.returncode in (0, 1) else completed.stderr
        lines: list[str] = []
        for line in (output or "").splitlines()[:match_cap]:
            lines.append(truncate(line, 200))
        if not lines:
            return f"No matches under {resolved}."
        return "\n".join(lines)

    def find_paths(
        self,
        root_path: str,
        *,
        name_pattern: str | None = None,
        file_type: str = "any",
        executable_only: bool = False,
        max_depth: int | None = None,
        max_matches: int | None = None,
    ) -> str:
        resolved, error = self._resolve_allowed_path(root_path)
        if error:
            return error
        assert resolved is not None
        if not resolved.exists():
            return f"ERROR: path not found: {resolved}"
        if not resolved.is_dir():
            return f"ERROR: root path is not a directory: {resolved}"

        normalized_type = str(file_type or "any").strip().lower()
        if normalized_type not in {"any", "file", "dir"}:
            return "ERROR: file_type must be one of: any, file, dir"

        pattern = str(name_pattern or "*").strip() or "*"
        depth_cap = max(int(max_depth or 6), 0)
        match_cap = min(max(int(max_matches or self.config.grep_max_matches), 1), 200)

        root_depth = len(resolved.parts)
        matches: list[str] = []
        for current_root, dirnames, filenames in os.walk(resolved):
            current_path = Path(current_root)
            current_depth = len(current_path.parts) - root_depth
            if current_depth >= depth_cap:
                dirnames[:] = []

            if normalized_type in {"any", "dir"} and current_depth > 0:
                if fnmatch.fnmatch(current_path.name, pattern):
                    matches.append(str(current_path))
                    if len(matches) >= match_cap:
                        break

            if normalized_type in {"any", "file"}:
                for filename in filenames:
                    if not fnmatch.fnmatch(filename, pattern):
                        continue
                    candidate = current_path / filename
                    if executable_only and not os.access(candidate, os.X_OK):
                        continue
                    matches.append(str(candidate))
                    if len(matches) >= match_cap:
                        break
                if len(matches) >= match_cap:
                    break

        if not matches:
            return (
                f"No matches under {resolved} "
                f"(pattern={pattern}, file_type={normalized_type}, executable_only={executable_only})."
            )
        return "\n".join(matches)
