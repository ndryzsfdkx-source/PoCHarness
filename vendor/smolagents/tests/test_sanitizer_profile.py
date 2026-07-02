from copy import deepcopy

from smolagents.secb.sanitizer.profile import (
    ObservedCrashProfile,
    ProfileContext,
    compare_replays,
    normalize_observed_crash,
)


ASAN = """==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1234
READ of size 4 at 0x1234 thread T0
    #0 0x1111 in __asan_memcpy /src/llvm-project/compiler-rt/lib/asan/x.cpp:1
    #1 0x2222 in target_fn /src/example/parser.c:42
SUMMARY: AddressSanitizer: heap-buffer-overflow /src/example/parser.c:42 in target_fn
"""


def test_normalize_observed_profile_extracts_project_frames(tmp_path):
    testcase = tmp_path / "testcase"
    testcase.mkdir()
    (testcase / "poc_file").write_bytes(b"abc")
    harness = tmp_path / "harness_shape.json"
    harness.write_text("{}")
    profile = normalize_observed_crash(
        ASAN,
        ProfileContext(
            candidate_dir=str(testcase),
            work_dir="/src/example",
            harness_shape_path=str(harness),
        ),
    )
    assert profile.sanitizer_family == "ASAN"
    assert profile.crash_type == "heap-buffer-overflow"
    assert profile.access_type == "READ"
    assert profile.first_project_frame["function"] == "target_fn"
    assert profile.candidate_hash
    assert profile.harness_shape_hash

    # Runtime, offline replay, and calibration all use this exact normalization entrypoint.
    replayed = normalize_observed_crash(
        ASAN,
        ProfileContext(
            candidate_dir=str(testcase),
            work_dir="/src/example",
            harness_shape_path=str(harness),
        ),
    )
    assert replayed.to_dict() == profile.to_dict()
    assert profile.report_hash


def test_replay_fingerprint_ignores_runtime_addresses():
    first = normalize_observed_crash(ASAN, ProfileContext(work_dir="/src/example"))
    second = normalize_observed_crash(
        ASAN.replace("0x1234", "0xabcd").replace("0x1111", "0xaaaa").replace("0x2222", "0xbbbb"),
        ProfileContext(work_dir="/src/example"),
    )
    assert first.crash_fingerprint() == second.crash_fingerprint()


def test_compare_replays_checks_provenance_and_semantics():
    payload = dict(
        sanitizer_family="ASAN",
        crash_type="heap-buffer-overflow",
        candidate_hash="candidate",
        source_head="head",
        source_diff_hash="diff",
        harness_shape_hash="harness",
        docker_image="image",
    )
    first = ObservedCrashProfile(**payload)
    second = ObservedCrashProfile(**deepcopy(payload))
    assert compare_replays(first, second).consistent is True
    second.candidate_hash = "changed"
    result = compare_replays(first, second)
    assert result.consistent is False
    assert "candidate_hash_mismatch" in result.reasons


def test_deadlysignal_and_lsan_are_explicit_states():
    deadly = normalize_observed_crash(
        "AddressSanitizer:DEADLYSIGNAL\nSUMMARY: AddressSanitizer: SEGV x.c:1 in f",
    )
    assert deadly.sanitizer_family == "ASAN"
    assert deadly.crash_type == "deadlysignal"

    leak = normalize_observed_crash(
        "==1==ERROR: LeakSanitizer: detected memory leaks\n"
        "SUMMARY: AddressSanitizer: 12 byte(s) leaked in 1 allocation(s)."
    )
    assert leak.sanitizer_family == "LSAN"
    assert leak.crash_type == "memory-leak"
    assert leak.lsan_status == "only_signal"
    assert leak.leaked_bytes == 12
