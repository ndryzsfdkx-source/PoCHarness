"""Unit tests for deterministic description-consistency prefilter.

Covers extract_described_facts, compare_described_vs_observed, and the top_frames
extension to extract_observed_facts.  Seven retrospective cases from a wrong-type
audit are used as fixtures; each uses synthetic ASAN output (not oracle
data) that the existing parsers (_detect_family / _detect_access_type /
parse_sanitizer_output) can process correctly.

Expected outcomes:
  exiv2-14857    -> type_mismatch       (invalid-free described; HBO observed)
  mruby-0614     -> insufficient_signal  (no class, no function extractable)
  gpac-4679      -> type_mismatch       (double-free described; SEGV observed)
  gpac-48013     -> type_mismatch       (double-free described; SEGV observed)
  njs-34029      -> frame_mismatch      (OOB-read OK type match; described func absent)
  faad2-20357    -> frame_mismatch      (null-deref OK type match; described func absent)
  njs-46462      -> frame_mismatch      (SEGV OK type match; described func absent)
"""
import pytest

from smolagents.secb.harness.description_judge import (
    compare_described_vs_observed,
    extract_described_facts,
    extract_observed_facts,
)


# ---------------------------------------------------------------------------
# Synthetic ASAN output helpers
# ---------------------------------------------------------------------------

def _asan_hbo(frames: list[str]) -> str:
    """Minimal ASAN heap-buffer-overflow output with given frames."""
    frame_lines = "\n".join(f"    #{i} 0x{0x55+i:x} in {fn}" for i, fn in enumerate(frames))
    return (
        "==1234==ERROR: AddressSanitizer: heap-buffer-overflow on address 0xdeadbeef\n"
        "READ of size 4 at 0xdeadbeef\n"
        f"{frame_lines}\n"
    )


def _asan_segv(frames: list[str]) -> str:
    """Minimal ASAN SEGV output with given frames."""
    frame_lines = "\n".join(f"    #{i} 0x{0x55+i:x} in {fn}" for i, fn in enumerate(frames))
    return (
        "==1234==ERROR: AddressSanitizer: SEGV on unknown address 0x0 (pc 0x1 bp 0x2 sp 0x3 T0)\n"
        f"{frame_lines}\n"
    )


# ---------------------------------------------------------------------------
# Case 1: exiv2-14857  (type_mismatch)
# invalid-free described; observed crash is HBO
# ---------------------------------------------------------------------------

EXIV2_DESC = (
    "An invalid-free vulnerability exists in the `Image` class in exiv2. "
    "When processing a malformed TIFF file, the code frees an already-freed buffer."
)
EXIV2_OBS = _asan_hbo([
    "Image::printIFDStructure /src/exiv2/src/image.cpp:100",
    "Exiv2::BasicIo::read /src/exiv2/src/basicio.cpp:200",
])


def test_exiv2_described_facts():
    f = extract_described_facts(EXIV2_DESC)
    assert f["described_class"] == "invalid-free"
    assert "Image" in f["described_functions"]


def test_exiv2_type_mismatch():
    described = extract_described_facts(EXIV2_DESC)
    observed = extract_observed_facts(EXIV2_OBS)
    result = compare_described_vs_observed(described, observed)
    assert result["verdict"] == "type_mismatch"
    assert result["type_match"] is False


# ---------------------------------------------------------------------------
# Case 2: mruby-0614  (insufficient_signal)
# Description has no crash class keyword and no named function.
# ---------------------------------------------------------------------------

MRUBY_DESC = (
    "An out-of-range ptr offset access causes heap corruption in mruby "
    "when evaluating certain bytecode sequences."
)
MRUBY_OBS = _asan_hbo([
    "bin_to_uint16 /src/mruby/src/value.c:50",
    "mrb_vm_exec /src/mruby/src/vm.c:200",
])


def test_mruby_described_facts():
    f = extract_described_facts(MRUBY_DESC)
    # "out-of-range" does NOT match any pattern (intentional ceiling case)
    assert f["described_class"] is None
    assert f["described_functions"] == []


def test_mruby_insufficient_signal():
    described = extract_described_facts(MRUBY_DESC)
    observed = extract_observed_facts(MRUBY_OBS)
    result = compare_described_vs_observed(described, observed)
    assert result["verdict"] == "insufficient_signal"


# ---------------------------------------------------------------------------
# Case 3: gpac-4679  (type_mismatch)
# double-free described; observed crash is SEGV
# ---------------------------------------------------------------------------

GPAC4679_DESC = (
    "Double-free in `gf_filterpacket_del` when processing filter packets in GPAC. "
    "The function frees the same packet buffer twice under error paths."
)
GPAC4679_OBS = _asan_segv([
    "gf_filter_pck_new_ref /src/gpac/src/filter_pck.c:200",
    "gf_filter_pck_new_alloc /src/gpac/src/filter_pck.c:210",
])


def test_gpac4679_described_facts():
    f = extract_described_facts(GPAC4679_DESC)
    assert f["described_class"] == "double-free"
    assert "gf_filterpacket_del" in f["described_functions"]


def test_gpac4679_type_mismatch():
    described = extract_described_facts(GPAC4679_DESC)
    observed = extract_observed_facts(GPAC4679_OBS)
    result = compare_described_vs_observed(described, observed)
    assert result["verdict"] == "type_mismatch"
    assert result["type_match"] is False


# ---------------------------------------------------------------------------
# Case 4: gpac-48013  (type_mismatch, same pattern as 4679)
# ---------------------------------------------------------------------------

GPAC48013_DESC = (
    "Double-free vulnerability triggered via `gf_filterpacket_del` in the GPAC "
    "mp4box filter pipeline. Reproducible with a crafted .mp4 input."
)
GPAC48013_OBS = _asan_segv([
    "gf_filter_pck_new_alloc /src/gpac/src/filter_pck.c:210",
    "gf_filter_pid_get_packet /src/gpac/src/filter_pid.c:300",
])


def test_gpac48013_type_mismatch():
    described = extract_described_facts(GPAC48013_DESC)
    observed = extract_observed_facts(GPAC48013_OBS)
    result = compare_described_vs_observed(described, observed)
    assert result["verdict"] == "type_mismatch"
    assert result["type_match"] is False


# ---------------------------------------------------------------------------
# Case 5: njs-34029  (frame_mismatch)
# OOB-read described (type-compatible with HBO); described function absent from top-N
# ---------------------------------------------------------------------------

NJS34029_DESC = (
    "OOB-read in `njs_scope_value` leading to heap memory corruption in njs. "
    "The scope lookup reads past the end of the allocated scope array."
)
NJS34029_OBS = _asan_hbo([
    "njs_function_lambda_frame /src/njs/src/njs_function.c:300",
    "njs_vmcode_call /src/njs/src/njs_vmcode.c:400",
    "njs_vmcode_interpreter /src/njs/src/njs_vmcode.c:600",
])


def test_njs34029_described_facts():
    f = extract_described_facts(NJS34029_DESC)
    assert f["described_class"] == "oob-read"
    assert "njs_scope_value" in f["described_functions"]


def test_njs34029_frame_mismatch():
    described = extract_described_facts(NJS34029_DESC)
    observed = extract_observed_facts(NJS34029_OBS)
    # Type IS compatible (oob-read vs HBO) but described function not in top-N
    assert _check_type_compatible_for_test(described, observed) is True
    result = compare_described_vs_observed(described, observed)
    assert result["verdict"] == "frame_mismatch"
    assert result["frame_match"] is False


# ---------------------------------------------------------------------------
# Case 6: faad2-20357  (frame_mismatch)
# null-deref SEGV described; SEGV observed (type-compatible); described func absent
# ---------------------------------------------------------------------------

FAAD2_DESC = (
    "Null-deref SEGV in `sbr_process_channel` during AAC SBR decoding in faad2. "
    "The channel pointer is NULL when processing certain SBR extension data."
)
FAAD2_OBS = _asan_segv([
    "ifilter_bank /src/faad2/src/filterbank.c:400",
    "sbr_qmf_analysis_32 /src/faad2/src/sbr_qmf.c:500",
])


def test_faad2_described_facts():
    f = extract_described_facts(FAAD2_DESC)
    assert f["described_class"] == "null-deref"
    assert "sbr_process_channel" in f["described_functions"]


def test_faad2_frame_mismatch():
    described = extract_described_facts(FAAD2_DESC)
    observed = extract_observed_facts(FAAD2_OBS)
    result = compare_described_vs_observed(described, observed)
    assert result["verdict"] == "frame_mismatch"
    assert result["frame_match"] is False


# ---------------------------------------------------------------------------
# Case 7: njs-46462  (frame_mismatch)
# SEGV described; SEGV observed (type-compatible); described function absent
# ---------------------------------------------------------------------------

NJS46462_DESC = (
    "SEGV crash in `njs_object_set_prototype` when setting prototype chain in njs. "
    "The object pointer is stale after an internal GC move."
)
NJS46462_OBS = _asan_segv([
    "njs_promise_resolve /src/njs/src/njs_promise.c:500",
    "njs_vmcode_interpreter /src/njs/src/njs_vmcode.c:600",
    "njs_function_lambda_frame /src/njs/src/njs_function.c:300",
])


def test_njs46462_described_facts():
    f = extract_described_facts(NJS46462_DESC)
    assert f["described_class"] == "SEGV"
    assert "njs_object_set_prototype" in f["described_functions"]


def test_njs46462_frame_mismatch():
    described = extract_described_facts(NJS46462_DESC)
    observed = extract_observed_facts(NJS46462_OBS)
    result = compare_described_vs_observed(described, observed)
    assert result["verdict"] == "frame_mismatch"
    assert result["frame_match"] is False


# ---------------------------------------------------------------------------
# Top-frames extension
# ---------------------------------------------------------------------------

def test_extract_observed_facts_returns_top_frames():
    obs = extract_observed_facts(NJS34029_OBS)
    assert "top_frames" in obs
    assert isinstance(obs["top_frames"], list)
    assert len(obs["top_frames"]) > 0
    assert "function" in obs["top_frames"][0]


def test_extract_observed_facts_top_frames_respects_top_n():
    obs = extract_observed_facts(NJS34029_OBS, top_n=2)
    assert len(obs["top_frames"]) <= 2


def test_extract_observed_facts_backward_compat():
    # Keys from before the top_frames extension must still be present.
    obs = extract_observed_facts(EXIV2_OBS)
    assert "sanitizer_family" in obs
    assert "crash_type" in obs
    assert "top_frame" in obs


# ---------------------------------------------------------------------------
# Aggregate: 6/7 caught, mruby fails open
# ---------------------------------------------------------------------------

_ALL_CASES = [
    (EXIV2_DESC, EXIV2_OBS, "type_mismatch"),
    (MRUBY_DESC, MRUBY_OBS, "insufficient_signal"),
    (GPAC4679_DESC, GPAC4679_OBS, "type_mismatch"),
    (GPAC48013_DESC, GPAC48013_OBS, "type_mismatch"),
    (NJS34029_DESC, NJS34029_OBS, "frame_mismatch"),
    (FAAD2_DESC, FAAD2_OBS, "frame_mismatch"),
    (NJS46462_DESC, NJS46462_OBS, "frame_mismatch"),
]

_CAUGHT_VERDICTS = {"type_mismatch", "frame_mismatch"}


def test_six_of_seven_caught():
    caught = 0
    for desc, obs_raw, expected_verdict in _ALL_CASES:
        described = extract_described_facts(desc)
        observed = extract_observed_facts(obs_raw)
        result = compare_described_vs_observed(described, observed)
        assert result["verdict"] == expected_verdict, (
            f"Expected {expected_verdict!r}, got {result['verdict']!r} for desc={desc[:60]!r}"
        )
        if result["verdict"] in _CAUGHT_VERDICTS:
            caught += 1
    assert caught == 6, f"Expected 6/7 caught, got {caught}/7"


def test_mruby_fails_open():
    described = extract_described_facts(MRUBY_DESC)
    observed = extract_observed_facts(MRUBY_OBS)
    result = compare_described_vs_observed(described, observed)
    assert result["verdict"] == "insufficient_signal"


# ---------------------------------------------------------------------------
# Helper used inline above (keeps test readable without importing private fn)
# ---------------------------------------------------------------------------

from smolagents.secb.harness.description_judge import _check_type_compatible  # noqa: E402


def _check_type_compatible_for_test(described: dict, observed: dict) -> bool | None:
    return _check_type_compatible(
        described.get("described_class"),
        observed.get("sanitizer_family"),
        observed.get("crash_type"),
    )
