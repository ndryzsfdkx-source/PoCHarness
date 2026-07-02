"""artifact_guard A-loop tool + harness_shape_probe startup helper."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from smolagents.tools import Tool


HARNESS_SHAPE_PATH = Path("/workspace/harness_shape.json")
DEFAULT_SECB_SCRIPT = Path("/usr/local/bin/secb")
DEFAULT_TESTCASE_ROOT = Path("/testcase")
HARNESS_SHAPE_PROBE_VERSION = "synthesis_harness_shape/v2"


# ---------------------------------------------------------------------------
# Harness shape probe


_TESTCASE_PATH_RE = re.compile(r"/testcase/([A-Za-z0-9_\-.]+)")
_REPRO_CMD_RE = re.compile(r"repro\s*\(\)\s*\{[^}]+\}", re.DOTALL)
_BUILD_CMD_RE = re.compile(r"build\s*\(\)\s*\{[^}]+\}", re.DOTALL)


def _extract_testcase_filename(script_text: str) -> str | None:
    """Return the first /testcase/<filename> referenced in the secb script, if any."""
    match = _TESTCASE_PATH_RE.search(script_text)
    if not match:
        return None
    return match.group(1)


def _extract_block(script_text: str, fn_name: str) -> str | None:
    pattern = re.compile(rf"{fn_name}\s*\(\)\s*\{{(?P<body>[^}}]*)\}}", re.DOTALL)
    match = pattern.search(script_text)
    if not match:
        return None
    return match.group("body").strip()


def run_harness_shape_probe(
    *,
    work_dir: str | None = None,
    secb_script: Path = DEFAULT_SECB_SCRIPT,
    output_path: Path = HARNESS_SHAPE_PATH,
) -> dict[str, Any]:
    """Parse /usr/local/bin/secb and cache a harness-shape JSON.

    Idempotent: if `output_path` exists and its `_secb_sha256` matches the current
    script, returns the cached payload unchanged. Otherwise writes a new one.
    """
    payload: dict[str, Any] = {
        "probe_version": HARNESS_SHAPE_PROBE_VERSION,
        "secb_script": str(secb_script),
        "work_dir": work_dir,
        "expected_testcase_filename": None,
        "expected_testcase_dir": str(DEFAULT_TESTCASE_ROOT),
        "build_block": None,
        "repro_block": None,
        "warnings": [],
    }
    try:
        script_text = secb_script.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        payload["warnings"].append(f"{secb_script} not found")
        _write_shape(output_path, payload)
        return payload
    except Exception as exc:
        payload["warnings"].append(f"failed to read {secb_script}: {exc}")
        _write_shape(output_path, payload)
        return payload

    sha = hashlib.sha256(script_text.encode("utf-8", errors="replace")).hexdigest()
    payload["_secb_sha256"] = sha

    # Cache-hit short-circuit.
    if output_path.exists():
        try:
            previous = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                isinstance(previous, dict)
                and previous.get("_secb_sha256") == sha
                and previous.get("probe_version") == HARNESS_SHAPE_PROBE_VERSION
            ):
                return previous
        except Exception:
            pass

    payload["build_block"] = _extract_block(script_text, "build")
    payload["repro_block"] = _extract_block(script_text, "repro")
    # The build/patch blocks often mention repo_changes.diff or model_patch.diff.
    # The handoff filename that matters for PoC generation is the path consumed
    # by repro(), so prefer that block and only fall back to the whole wrapper.
    payload["expected_testcase_filename"] = _extract_testcase_filename(
        payload["repro_block"] or script_text
    )
    _write_shape(output_path, payload)
    return payload


def _write_shape(output_path: Path, payload: dict[str, Any]) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        import sys

        print(f"WARNING: harness_shape write failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# artifact_guard tool


def _load_shape() -> dict[str, Any] | None:
    if not HARNESS_SHAPE_PATH.exists():
        return None
    try:
        return json.loads(HARNESS_SHAPE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _hexdump(data: bytes, length: int = 64) -> str:
    sample = data[:length]
    lines = []
    for offset in range(0, len(sample), 16):
        chunk = sample[offset : offset + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{offset:08x}  {hex_part:<47}  {ascii_part}")
    return "\n".join(lines)


def _inspect_testcase(root: Path, shape: dict[str, Any] | None) -> list[str]:
    lines: list[str] = []
    if not root.exists():
        lines.append(f"/testcase does not exist: {root}")
        return lines
    if not root.is_dir():
        lines.append(f"/testcase is not a directory: {root}")
        return lines

    entries = sorted(root.iterdir(), key=lambda p: p.name)
    if not entries:
        lines.append("/testcase is empty.")
        return lines

    lines.append(f"/testcase contents ({len(entries)} entr{'y' if len(entries)==1 else 'ies'}):")
    for entry in entries:
        try:
            stat = entry.stat()
        except OSError as exc:
            lines.append(f"  {entry.name}  <stat failed: {exc}>")
            continue
        kind = "dir" if entry.is_dir() else "file"
        lines.append(f"  {entry.name}  {kind}  {stat.st_size}B")

    expected_name = (shape or {}).get("expected_testcase_filename") if shape else None
    if expected_name:
        present = (root / expected_name)
        if present.exists():
            lines.append(f"Expected filename '{expected_name}' is PRESENT.")
        else:
            lines.append(
                f"Expected filename '{expected_name}' is MISSING. Files present: "
                + ", ".join(e.name for e in entries)
            )
    elif shape is None:
        lines.append("(harness shape unavailable; cannot verify expected filename)")
    else:
        lines.append("(harness shape did not record an expected filename)")

    # Hexdump the primary file (expected if known, else the largest file).
    primary: Path | None = None
    if expected_name:
        candidate = root / expected_name
        if candidate.exists() and candidate.is_file():
            primary = candidate
    if primary is None:
        files = [e for e in entries if e.is_file()]
        if files:
            primary = max(files, key=lambda p: p.stat().st_size if p.exists() else 0)
    if primary is not None:
        try:
            with primary.open("rb") as f:
                data = f.read(128)
            lines.append(f"hexdump (first 64B) of {primary.name}:")
            lines.append(_hexdump(data, 64))
        except Exception as exc:
            lines.append(f"hexdump failed for {primary}: {exc}")
    return lines


class ArtifactGuardTool(Tool):
    name = "artifact_guard"
    description = (
        "Inspect /testcase before finalising: verifies the expected filename (from "
        "harness_shape.json) is present, lists current files with sizes, and prints a "
        "short hexdump. Useful to surface stale artifacts and wrong filenames before "
        "running secb repro."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Root to inspect (defaults to /testcase).",
            "nullable": True,
        }
    }
    output_type = "string"

    def forward(self, path: str | None = None) -> str:
        root = Path(path) if path else DEFAULT_TESTCASE_ROOT
        shape = _load_shape()
        report_lines = ["[ARTIFACT GUARD]"]
        if shape is None:
            report_lines.append("harness_shape.json: not found (probe did not run).")
        else:
            report_lines.append(
                f"harness_shape.json: expected_testcase_filename="
                f"{shape.get('expected_testcase_filename')!r}"
            )
        report_lines.extend(_inspect_testcase(root, shape))
        report_lines.append("[END ARTIFACT GUARD]")
        return "\n".join(report_lines)
