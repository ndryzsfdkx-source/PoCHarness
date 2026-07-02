"""Canonical, non-oracle crash evidence profiles for poc-desc runtime and replay."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .parser import StackFrame, parse_sanitizer_output


PARSER_VERSION = "observed_crash_profile/v1"
RUNTIME_FUNCTION_PREFIXES = (
    "__asan_",
    "__lsan_",
    "__ubsan_",
    "__sanitizer_",
    "__interceptor_",
)
RUNTIME_FUNCTIONS = {
    "malloc",
    "free",
    "calloc",
    "realloc",
    "memcpy",
    "memmove",
    "memset",
    "memcmp",
    "strcmp",
    "strlen",
    "operator new",
    "operator delete",
}
_ASAN_KIND_RE = re.compile(r"AddressSanitizer:\s*([A-Za-z0-9_-]+)", re.I)
_LSAN_RE = re.compile(r"LeakSanitizer:\s*detected memory leaks", re.I)
_LEAK_SUMMARY_RE = re.compile(
    r"SUMMARY:\s+(?:AddressSanitizer|LeakSanitizer):\s+(\d+)\s+byte\(s\) leaked in (\d+) allocation",
    re.I,
)
_SIGNAL_RE = re.compile(r"\b(?:SIGSEGV|signal SEGV|segmentation fault)\b", re.I)
_NULL_ADDRESS_RE = re.compile(r"(?:unknown address\s+)?0x0+\b|zero page", re.I)


@dataclass(frozen=True)
class ProfileContext:
    candidate_dir: str = "/testcase"
    work_dir: str = ""
    harness_shape_path: str = "/workspace/harness_shape.json"
    repro_command: tuple[str, ...] = ("secb", "repro")
    docker_image: str = ""
    report_path: str = ""
    completeness: str = "complete"
    restore_warning: str = ""


@dataclass
class ObservedCrashProfile:
    parser_version: str = PARSER_VERSION
    sanitizer_family: str | None = None
    crash_type: str | None = None
    access_type: str | None = None
    access_size: int | None = None
    crash_address: str | None = None
    all_frames: list[dict[str, Any]] = field(default_factory=list)
    project_frames: list[dict[str, Any]] = field(default_factory=list)
    first_project_frame: dict[str, Any] | None = None
    lsan_status: str = "not_present"
    leaked_bytes: int | None = None
    leaked_allocations: int | None = None
    candidate_hash: str = ""
    report_hash: str = ""
    report_path: str = ""
    repro_command: list[str] = field(default_factory=lambda: ["secb", "repro"])
    docker_image: str = ""
    source_head: str = ""
    source_diff_hash: str = ""
    harness_shape_hash: str = ""
    report_completeness: str = "complete"
    restore_warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_crash_payload(self) -> dict[str, Any]:
        """Return only semantic crash fields, excluding runtime provenance."""
        return {
            "parser_version": self.parser_version,
            "sanitizer_family": self.sanitizer_family,
            "crash_type": self.crash_type,
            "access_type": self.access_type,
            "access_size": self.access_size,
            "crash_address": self.crash_address,
            "all_frames": self.all_frames,
            "project_frames": self.project_frames,
            "first_project_frame": self.first_project_frame,
            "lsan_status": self.lsan_status,
            "leaked_bytes": self.leaked_bytes,
            "leaked_allocations": self.leaked_allocations,
        }

    def crash_fingerprint(self) -> str:
        # Runtime addresses vary across identical A/B replays under ASLR. Stability is
        # defined by semantic type/access and symbolized frame identity, not raw PCs.
        payload_dict = self.canonical_crash_payload()
        payload_dict["crash_address"] = None
        for key in ("all_frames", "project_frames"):
            payload_dict[key] = [
                {name: value for name, value in frame.items() if name != "address"}
                for frame in payload_dict[key]
            ]
        if payload_dict.get("first_project_frame"):
            payload_dict["first_project_frame"] = {
                name: value
                for name, value in payload_dict["first_project_frame"].items()
                if name != "address"
            }
        payload = json.dumps(payload_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ReplayConsistency:
    consistent: bool
    status: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"consistent": self.consistent, "status": self.status, "reasons": list(self.reasons)}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    try:
        return _sha256(Path(path).read_bytes())
    except OSError:
        return ""


def hash_candidate_tree(path: str | Path = "/testcase") -> str:
    """Hash a testcase tree deterministically without following symlinks."""
    root = Path(path)
    if not root.exists():
        return ""
    digest = hashlib.sha256()
    entries = [root, *sorted(root.rglob("*"), key=lambda p: p.as_posix())]
    for entry in entries:
        try:
            info = entry.lstat()
            rel = "." if entry == root else entry.relative_to(root).as_posix()
            digest.update(rel.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            digest.update(oct(stat.S_IMODE(info.st_mode)).encode("ascii"))
            digest.update(b"\0")
            if entry.is_symlink():
                digest.update(b"L")
                digest.update(os.readlink(entry).encode("utf-8", errors="surrogateescape"))
            elif entry.is_file():
                digest.update(b"F")
                with entry.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            elif entry.is_dir():
                digest.update(b"D")
            else:
                digest.update(b"O")
            digest.update(b"\0")
        except OSError:
            return ""
    return digest.hexdigest()


def _git_output(work_dir: str, *args: str) -> bytes:
    if not work_dir:
        return b""
    try:
        result = subprocess.run(
            ["git", "-C", work_dir, *args],
            capture_output=True,
            check=False,
            timeout=10,
        )
        return result.stdout if result.returncode == 0 else b""
    except Exception:
        return b""


def source_identity(work_dir: str) -> tuple[str, str]:
    head = _git_output(work_dir, "rev-parse", "HEAD").decode("utf-8", errors="replace").strip()
    diff = _git_output(work_dir, "diff", "--binary", "--no-ext-diff", "HEAD")
    return head, _sha256(diff) if diff or head else ""


def _normalize_family(raw_family: str | None, raw: str) -> str | None:
    if _LSAN_RE.search(raw) and not re.search(
        r"ERROR:\s+AddressSanitizer:\s*(?!detected memory leaks)", raw, re.I
    ):
        return "LSAN"
    value = str(raw_family or "").lower()
    if "address" in value:
        return "ASAN"
    if "leak" in value:
        return "LSAN"
    if "undefined" in value:
        return "UBSAN"
    if "memory" in value:
        return "MSAN"
    if "thread" in value:
        return "TSAN"
    if _LSAN_RE.search(raw):
        return "LSAN"
    if _SIGNAL_RE.search(raw):
        return "SEGV"
    return None


def _normalize_crash_type(raw_type: str | None, family: str | None, raw: str) -> str | None:
    kind = str(raw_type or "").strip().lower()
    asan = _ASAN_KIND_RE.search(raw)
    if asan:
        kind = asan.group(1).lower()
    if "deadlysignal" in raw.lower() or kind == "deadlysignal":
        return "deadlysignal"
    if family == "LSAN" and _LSAN_RE.search(raw):
        return "memory-leak"
    if not kind and family == "SEGV":
        return "segv"
    return kind or None


def _is_runtime_frame(frame: StackFrame) -> bool:
    fn = str(frame.function or "").strip()
    lowered = fn.lower()
    if any(lowered.startswith(prefix) for prefix in RUNTIME_FUNCTION_PREFIXES):
        return True
    if lowered in RUNTIME_FUNCTIONS:
        return True
    path = str(frame.file or "").lower()
    return any(
        token in path
        for token in (
            "compiler-rt/lib/asan",
            "sanitizer_common",
            "/libc",
            "/libstdc++",
            "libasan.so",
        )
    )


def _is_project_frame(frame: StackFrame, work_dir: str) -> bool:
    if _is_runtime_frame(frame):
        return False
    path = str(frame.file or "")
    if not path:
        return False
    normalized_work = str(Path(work_dir)) if work_dir else ""
    if normalized_work and (path == normalized_work or path.startswith(normalized_work.rstrip("/") + "/")):
        return True
    project_name = Path(normalized_work).name if normalized_work else ""
    return bool(project_name and f"/{project_name}/" in path)


def normalize_observed_crash(
    raw_report: str | bytes,
    context: ProfileContext | None = None,
) -> ObservedCrashProfile:
    """Normalize a full repro report using the same path at runtime and offline."""
    context = context or ProfileContext()
    raw_bytes = raw_report if isinstance(raw_report, bytes) else str(raw_report or "").encode(
        "utf-8", errors="replace"
    )
    raw = raw_bytes.decode("utf-8", errors="replace")
    parsed = parse_sanitizer_output(raw)
    family = _normalize_family(parsed.sanitizer, raw)
    crash_type = _normalize_crash_type(parsed.crash_type, family, raw)
    frames = [frame.to_dict() for frame in parsed.stack_frames]
    project_frames = [frame.to_dict() for frame in parsed.stack_frames if _is_project_frame(frame, context.work_dir)]
    leak_match = _LEAK_SUMMARY_RE.search(raw)
    has_lsan = bool(_LSAN_RE.search(raw))
    has_non_lsan_signal = bool(
        re.search(r"ERROR:\s+AddressSanitizer:\s*(?!detected memory leaks)", raw, re.I)
        or _SIGNAL_RE.search(raw)
    )
    source_head, source_diff_hash = source_identity(context.work_dir)
    completeness = context.completeness
    if context.restore_warning:
        completeness = "restore_failure"
    return ObservedCrashProfile(
        sanitizer_family=family,
        crash_type=crash_type,
        access_type=parsed.access_type,
        access_size=parsed.access_size,
        crash_address=parsed.crash_address,
        all_frames=frames,
        project_frames=project_frames,
        first_project_frame=project_frames[0] if project_frames else None,
        lsan_status=("present" if has_non_lsan_signal else "only_signal") if has_lsan else "not_present",
        leaked_bytes=int(leak_match.group(1)) if leak_match else None,
        leaked_allocations=int(leak_match.group(2)) if leak_match else None,
        candidate_hash=hash_candidate_tree(context.candidate_dir),
        report_hash=_sha256(raw_bytes),
        report_path=context.report_path,
        repro_command=list(context.repro_command),
        docker_image=context.docker_image or os.getenv("DOCKER_IMAGE", ""),
        source_head=source_head,
        source_diff_hash=source_diff_hash,
        harness_shape_hash=hash_file(context.harness_shape_path),
        report_completeness=completeness,
        restore_warning=context.restore_warning,
    )


def compare_replays(
    a_profile: ObservedCrashProfile | None,
    b_profile: ObservedCrashProfile | None,
) -> ReplayConsistency:
    if a_profile is None or b_profile is None:
        return ReplayConsistency(False, "missing", ("a_or_b_profile_missing",))
    reasons: list[str] = []
    if a_profile.report_completeness != "complete" or b_profile.report_completeness != "complete":
        reasons.append("report_incomplete")
    for field_name in ("candidate_hash", "source_head", "source_diff_hash", "harness_shape_hash", "docker_image"):
        left = getattr(a_profile, field_name)
        right = getattr(b_profile, field_name)
        if not left or not right:
            reasons.append(f"{field_name}_missing")
        elif left != right:
            reasons.append(f"{field_name}_mismatch")
    if a_profile.crash_fingerprint() != b_profile.crash_fingerprint():
        reasons.append("crash_profile_mismatch")
    return ReplayConsistency(not reasons, "consistent" if not reasons else "inconsistent", tuple(reasons))


def profile_from_dict(value: dict[str, Any] | None) -> ObservedCrashProfile | None:
    if not value:
        return None
    allowed = set(ObservedCrashProfile.__dataclass_fields__)
    return ObservedCrashProfile(**{key: item for key, item in value.items() if key in allowed})


def persist_raw_report(
    *,
    log_dir: str | Path,
    role: str,
    sequence: int,
    raw_report: str | bytes,
) -> tuple[str, str]:
    data = raw_report if isinstance(raw_report, bytes) else str(raw_report or "").encode("utf-8", errors="replace")
    digest = _sha256(data)
    evidence_dir = Path(log_dir) / "finalization_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    name = f"{role}_{sequence:03d}_{digest[:12]}.log"
    path = evidence_dir / name
    path.write_bytes(data)
    return str(Path("finalization_evidence") / name), digest
