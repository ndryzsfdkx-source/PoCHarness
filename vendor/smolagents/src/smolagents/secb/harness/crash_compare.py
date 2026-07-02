"""Strict crash comparison for Agent C: classify observed vs target sanitizer signal."""
from __future__ import annotations

import re
from typing import Any


STATUS_EXACT_MATCH = "exact_match"
STATUS_SAME_FAMILY_WRONG_FRAME = "same_family_wrong_frame"
STATUS_WRONG_CRASH = "wrong_crash"
STATUS_NO_SANITIZER = "no_sanitizer"
STATUS_HARNESS_REJECT = "harness_reject"
STATUS_ENVIRONMENT_FAILURE = "environment_failure"

PROGRESS_ARTIFACT_MISSING = "artifact_missing"
PROGRESS_ENVIRONMENT_FAILURE = "environment_failure"
PROGRESS_HARNESS_REJECT = "harness_reject"
PROGRESS_ACCEPTED_NO_SIGNAL = "accepted_no_signal"
PROGRESS_SANITIZER_SEEN = "sanitizer_seen"
PROGRESS_TARGET_FAMILY_SEEN = "target_family_seen"
PROGRESS_TARGET_FRAME_SEEN = "target_frame_seen"

# Sanitizer family detection (top-level error tags emitted by ASan/UBSan/MSan/LSan/TSan).
_SANITIZER_FAMILY_PATTERNS = [
    ("ASAN", re.compile(r"AddressSanitizer", re.IGNORECASE)),
    ("UBSAN", re.compile(r"UndefinedBehaviorSanitizer|runtime error:", re.IGNORECASE)),
    ("MSAN", re.compile(r"MemorySanitizer", re.IGNORECASE)),
    ("LSAN", re.compile(r"LeakSanitizer", re.IGNORECASE)),
    ("TSAN", re.compile(r"ThreadSanitizer", re.IGNORECASE)),
    ("SEGV", re.compile(r"segmentation fault|SIGSEGV|signal SEGV", re.IGNORECASE)),
]

# ASan-style access type, e.g. "heap-buffer-overflow", "use-after-free".
_ASAN_ACCESS_RE = re.compile(
    r"(?:AddressSanitizer:\s*)([a-zA-Z0-9_-]+)"
)
# UBSan access kind embedded in "runtime error: <kind>"
_UBSAN_KIND_RE = re.compile(r"runtime error:\s*([a-zA-Z0-9_ \-]+)")
# Top frame: first "#0 0x... in <fn> /path:line"
_TOP_FRAME_RE = re.compile(
    r"#0\s+0x[0-9a-fA-F]+\s+in\s+([A-Za-z0-9_:\.<>~\-]+)\s*(?:[^/\n]*)(/[^\s:]+)?(?::(\d+))?"
)
# Harness rejection markers (target binary refuses the input before vuln path).
_HARNESS_REJECT_RE = re.compile(
    r"\b(invalid\s+(?:file|input|format)|unsupported\s+format|magic\s+(?:bytes|mismatch)|"
    r"unknown\s+input\s+file\s+format|not\s+a\s+valid|bad\s+magic|cannot\s+open|fatal\s+error)\b",
    re.IGNORECASE,
)
# Environment / harness infra failures.
_ENV_FAILURE_RE = re.compile(
    r"\b(secb\s+repro\s+failed\s+with\s+exit|docker\s+error|command\s+not\s+found|"
    r"permission\s+denied|no\s+such\s+file\s+or\s+directory|timed\s+out|"
    r"artifact\s+not\s+found|failed\s+to\s+(?:snapshot|stage)|harness_shape)\b",
    re.IGNORECASE,
)
_ERROR_LINE_RE = re.compile(
    r"(error|failed|invalid|unsupported|bad\s+magic|cannot\s+open|not\s+generated|"
    r"no\s+such\s+file|stream\s+end|timed\s+out)",
    re.IGNORECASE,
)


def _detect_family(text: str) -> str | None:
    for name, pattern in _SANITIZER_FAMILY_PATTERNS:
        if pattern.search(text):
            return name
    return None


def _detect_access_type(text: str, family: str | None) -> str | None:
    if family in ("ASAN", "MSAN", "LSAN", "TSAN"):
        match = _ASAN_ACCESS_RE.search(text)
        if match:
            return match.group(1).lower()
    if family == "UBSAN":
        match = _UBSAN_KIND_RE.search(text)
        if match:
            return match.group(1).strip().lower()
    return None


def _detect_top_frame(text: str) -> tuple[str | None, str | None, int | None]:
    match = _TOP_FRAME_RE.search(text)
    if not match:
        return None, None, None
    function = match.group(1)
    source = match.group(2)
    line_str = match.group(3)
    line = int(line_str) if line_str and line_str.isdigit() else None
    return function, source, line


def _summarize(text: str) -> dict[str, Any]:
    # NB: strict_crash_compare summarizes the *freeform* target_signal on one side (an
    # agent/human-authored string like "AddressSanitizer heap-buffer-overflow in foo"),
    # not a well-formed report. The keyword detectors tolerate that; normalize_observed_crash
    # requires report structure and returns family=None for it, collapsing every compare to
    # wrong_crash. Keep the tolerant detectors here.
    family = _detect_family(text)
    access = _detect_access_type(text, family)
    function, source, line = _detect_top_frame(text)
    return {
        "family": family,
        "access_type": access,
        "top_function": function,
        "top_source": source,
        "top_line": line,
    }


def _first_error_line(text: str) -> str | None:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if line and _ERROR_LINE_RE.search(line):
            return line
    return None


def strict_crash_compare(target_signal: str, observed_output: str) -> dict[str, Any]:
    """Classify observed_output against target_signal.

    Returns a dict with:
      status: enum (one of STATUS_*)
      target: parsed target_signal fields
      observed: parsed observed_output fields
      matching_dims: list of fields that matched (family, access_type, top_function, ...)
    """
    target_text = str(target_signal or "")
    observed_text = str(observed_output or "")

    target = _summarize(target_text)
    observed = _summarize(observed_text)

    # Environment/harness infrastructure failure takes precedence over no_sanitizer:
    # the run didn't actually exercise the target binary.
    if _ENV_FAILURE_RE.search(observed_text) and observed["family"] is None:
        return {
            "status": STATUS_ENVIRONMENT_FAILURE,
            "target": target,
            "observed": observed,
            "matching_dims": [],
        }

    if observed["family"] is None:
        # No sanitizer signal at all. Distinguish a harness rejection (input refused
        # before vuln path) from a silent no-op.
        if _HARNESS_REJECT_RE.search(observed_text):
            return {
                "status": STATUS_HARNESS_REJECT,
                "target": target,
                "observed": observed,
                "matching_dims": [],
            }
        return {
            "status": STATUS_NO_SANITIZER,
            "target": target,
            "observed": observed,
            "matching_dims": [],
        }

    matching_dims: list[str] = []
    if target["family"] and observed["family"] == target["family"]:
        matching_dims.append("family")
    if target["access_type"] and observed["access_type"] == target["access_type"]:
        matching_dims.append("access_type")
    if target["top_function"] and observed["top_function"] == target["top_function"]:
        matching_dims.append("top_function")
    if target["top_source"] and observed["top_source"] == target["top_source"]:
        matching_dims.append("top_source")
    if target["top_line"] and observed["top_line"] == target["top_line"]:
        matching_dims.append("top_line")

    if "family" not in matching_dims:
        return {
            "status": STATUS_WRONG_CRASH,
            "target": target,
            "observed": observed,
            "matching_dims": matching_dims,
        }

    # Same family. Is the top frame correct?
    target_has_frame = bool(target["top_function"])
    frame_matches = "top_function" in matching_dims
    if target_has_frame and frame_matches:
        return {
            "status": STATUS_EXACT_MATCH,
            "target": target,
            "observed": observed,
            "matching_dims": matching_dims,
        }
    return {
        "status": STATUS_SAME_FAMILY_WRONG_FRAME,
        "target": target,
        "observed": observed,
        "matching_dims": matching_dims,
    }


def classify_helper_repro_progress(
    target_signal: str,
    observed_output: str,
    *,
    artifact_created: bool = True,
) -> dict[str, Any]:
    """Classify how far a helper-generated artifact progressed under `secb repro`.

    This is intentionally coarser than `strict_crash_compare`: it reports the first
    operational gate Agent A should care about after C validates a helper output.
    """
    observed_text = str(observed_output or "")
    first_error_line = _first_error_line(observed_text)
    comparison = strict_crash_compare(target_signal, observed_text)
    observed = comparison.get("observed") or {}
    status = comparison.get("status")

    if not artifact_created:
        if re.search(r"artifact\s+not\s+found", observed_text, re.IGNORECASE):
            progress_gate = PROGRESS_ARTIFACT_MISSING
        else:
            progress_gate = PROGRESS_ENVIRONMENT_FAILURE
        harness_accepted = False
    elif status == STATUS_ENVIRONMENT_FAILURE:
        progress_gate = PROGRESS_ENVIRONMENT_FAILURE
        harness_accepted = False
    elif status == STATUS_HARNESS_REJECT:
        progress_gate = PROGRESS_HARNESS_REJECT
        harness_accepted = False
    else:
        harness_accepted = True
        if status == STATUS_EXACT_MATCH:
            progress_gate = PROGRESS_TARGET_FRAME_SEEN
        elif status == STATUS_SAME_FAMILY_WRONG_FRAME:
            progress_gate = PROGRESS_TARGET_FAMILY_SEEN
        elif status == STATUS_WRONG_CRASH:
            progress_gate = PROGRESS_SANITIZER_SEEN
        else:
            progress_gate = PROGRESS_ACCEPTED_NO_SIGNAL

    return {
        "artifact_created": bool(artifact_created),
        "harness_accepted": harness_accepted,
        "first_error_line": first_error_line,
        "sanitizer_seen": bool(observed.get("family")),
        "sanitizer_family": observed.get("family"),
        "top_frame": observed.get("top_function"),
        "progress_gate": progress_gate,
    }
