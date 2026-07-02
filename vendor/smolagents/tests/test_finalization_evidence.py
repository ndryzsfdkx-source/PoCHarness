import json
from types import SimpleNamespace

from smolagents.secb.review.evidence import (
    ALLOW_SUCCESS,
    ALLOW_UNVERIFIED,
    CONFIRMED_MISMATCH,
    CONTINUE_LOCAL,
    INSUFFICIENT_EVIDENCE,
    PROBABLE_MISMATCH,
    SUPPORTED_MATCH,
    EvidencePolicyState,
    EvidenceRelation,
    apply_evidence_policy,
    assess_evidence_relation,
    build_target_profile,
    extract_literal_target_profile,
)
from smolagents.secb.review.config import FinalizationReviewConfig
from smolagents.secb.review.reviewer import FinalizationReviewer, ReviewResult
from smolagents.secb.review.tools import (
    ALLOW_VERDICTS,
    EmitFinalizationVerdictTool,
    FinalizationReviewToolState,
)
from smolagents.secb.sanitizer.profile import ObservedCrashProfile, ReplayConsistency
from smolagents.secb.harness.config import SynthesisConfig


KNOWN_LITERAL_MISSES = [
    (
        "When processing a file, a Memory Leak occurs in the ReadOnePNGImage() function in coders/png.c.",
        "ReadOnePNGImage",
    ),
    ("In the ReadDCMImage function in coders/dcm.c a pointer is lost.", "ReadDCMImage"),
    ("A heap-based buffer overflow at MagickCore/statistic.c in EvaluateImages.", "EvaluateImages"),
    ("Memory leaks in AcquireMagickMemory because of an AnnotateImage error.", "AcquireMagickMemory"),
    ("Memory leaks at AcquireMagickMemory because of an error in MagickWand/mogrify.c.", "AcquireMagickMemory"),
    ("A null pointer dereference via htmlescape ../../programs/escape.c:29.", "htmlescape"),
    ("A heap based buffer overflow via htmlescape ../../programs/escape.c:48.", "htmlescape"),
    (
        "There is a stack-based buffer over-read in the function ReadNextStructField() in mat5.c.",
        "ReadNextStructField",
    ),
    ("The init_copy function in kernel.c makes initialize_copy calls.", "init_copy"),
    ("the dispatch (and dispatch_linked) code performs a one-byte read", "dispatch_linked"),
    ("code execution via the yr_execute_cod function in the exe.c component", "yr_execute_cod"),
]


def _profile(**overrides):
    values = dict(
        sanitizer_family="ASAN",
        crash_type="heap-buffer-overflow",
        access_type="READ",
        project_frames=[{"index": 1, "address": "", "function": "other", "file": "/src/p/other.c", "line": 1}],
        first_project_frame={"index": 1, "address": "", "function": "other", "file": "/src/p/other.c", "line": 1},
        candidate_hash="candidate",
        report_completeness="complete",
    )
    values.update(overrides)
    return ObservedCrashProfile(**values)


def test_known_literal_misses_recover_exact_spans():
    for description, expected in KNOWN_LITERAL_MISSES:
        profile = extract_literal_target_profile(description)
        facts = [fact for fact in profile.facts_for("function") if fact.value == expected]
        assert facts, (description, expected, profile.to_dict())
        fact = facts[0]
        assert description[fact.evidence_span[0] : fact.evidence_span[1]] == expected


class _Model:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def generate(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return SimpleNamespace(content=json.dumps(self.payload))


def test_hybrid_profile_drops_bad_spans_and_cannot_self_authorize_hard_gate(tmp_path):
    description = "A crash in parse_file while reading a PNG."
    model = _Model(
        {
            "facts": [
                {
                    "field": "bug_class",
                    "value": "use-after-free",
                    "evidence_text": "invented",
                    "evidence_span": [0, 8],
                    "provenance": "explicit",
                    "usable_for_hard_gate": True,
                },
                {
                    "field": "input_format",
                    "value": "PNG",
                    "evidence_text": "PNG",
                    "evidence_span": [36, 39],
                    "provenance": "paraphrased",
                    "usable_for_hard_gate": True,
                },
            ]
        }
    )
    cache = tmp_path / "profile.json"
    first = build_target_profile(
        description,
        mode="hybrid_llm",
        model=model,
        model_version="test",
        cache_path=cache,
    )
    second = build_target_profile(
        description,
        mode="hybrid_llm",
        model=model,
        model_version="test",
        cache_path=cache,
    )
    assert model.calls == 1
    assert not first.facts_for("bug_class")
    assert all(not fact.usable_for_hard_gate for fact in first.facts)
    assert second.cache_key == first.cache_key


def test_relation_hard_mismatch_requires_explicit_type_and_consistent_replay():
    target = extract_literal_target_profile("An invalid free in parse_file in parser.c.")
    observed = _profile()
    relation = assess_evidence_relation(
        "An invalid free in parse_file in parser.c.",
        target,
        observed,
        ReplayConsistency(True, "consistent"),
    )
    assert relation.relation == CONFIRMED_MISMATCH

    inconsistent = assess_evidence_relation(
        "An invalid free in parse_file in parser.c.",
        target,
        observed,
        ReplayConsistency(False, "inconsistent", ("candidate_hash_mismatch",)),
    )
    assert inconsistent.relation == INSUFFICIENT_EVIDENCE


def test_relation_supported_requires_positive_type_and_path():
    target = extract_literal_target_profile(
        "A heap-buffer-overflow in parse_file in parser.c."
    )
    observed = _profile(
        project_frames=[{"function": "parse_file", "file": "/src/p/parser.c", "line": 10}],
        first_project_frame={"function": "parse_file", "file": "/src/p/parser.c", "line": 10},
    )
    relation = assess_evidence_relation(
        "A heap-buffer-overflow in parse_file in parser.c.",
        target,
        observed,
        ReplayConsistency(True, "consistent"),
    )
    assert relation.relation == SUPPORTED_MATCH


def test_lsan_and_incomplete_report_never_create_unsupported_positive_match():
    stable = ReplayConsistency(True, "consistent")
    non_leak = extract_literal_target_profile(
        "A heap-buffer-overflow in parse_file in parser.c."
    )
    lsan_only = _profile(
        sanitizer_family="LSAN",
        crash_type="memory-leak",
        lsan_status="only_signal",
        project_frames=[],
        first_project_frame=None,
    )
    assert assess_evidence_relation(
        "A heap-buffer-overflow in parse_file in parser.c.",
        non_leak,
        lsan_only,
        stable,
    ).relation == INSUFFICIENT_EVIDENCE

    leak = extract_literal_target_profile("A memory leak in parse_file in parser.c.")
    lsan_with_path = _profile(
        sanitizer_family="LSAN",
        crash_type="memory-leak",
        lsan_status="only_signal",
        project_frames=[{"function": "parse_file", "file": "/src/p/parser.c", "line": 3}],
        first_project_frame={"function": "parse_file", "file": "/src/p/parser.c", "line": 3},
    )
    assert assess_evidence_relation(
        "A memory leak in parse_file in parser.c.",
        leak,
        lsan_with_path,
        stable,
    ).relation == SUPPORTED_MATCH

    lsan_with_path.report_completeness = "truncated"
    assert assess_evidence_relation(
        "A memory leak in parse_file in parser.c.",
        leak,
        lsan_with_path,
        stable,
    ).relation == INSUFFICIENT_EVIDENCE


def test_policy_is_shadow_in_advisory_and_bounded_in_enforce():
    target = extract_literal_target_profile("An invalid free in parse_file in parser.c.")
    observed = _profile()
    relation = EvidenceRelation(CONFIRMED_MISMATCH, "mismatch", "mismatch")
    state = EvidencePolicyState()
    shadow = apply_evidence_policy(
        mode="advisory",
        proposed_verdict=ALLOW_SUCCESS,
        relation=relation,
        target=target,
        observed=observed,
        state=state,
        remaining_steps=20,
        min_remaining_steps=8,
        review_index=0,
    )
    assert shadow.actual_verdict == ALLOW_SUCCESS
    assert shadow.recommended_verdict == CONTINUE_LOCAL
    assert state.challenge_count == 0

    first = apply_evidence_policy(
        mode="enforce",
        proposed_verdict=ALLOW_SUCCESS,
        relation=relation,
        target=target,
        observed=observed,
        state=state,
        remaining_steps=20,
        min_remaining_steps=8,
        review_index=0,
    )
    assert first.actual_verdict == CONTINUE_LOCAL
    assert state.challenge_count == 1
    # Under explicit_type_only a PROBABLE_MISMATCH does not force a redirect: the
    # deterministic policy falls through to B's proposed verdict (here ALLOW_SUCCESS).
    # The CONFIRMED-only narrowing replaces the prior "challenge everything, then
    # ALLOW_UNVERIFIED" behavior on non-CONFIRMED relations.
    second = apply_evidence_policy(
        mode="enforce",
        proposed_verdict=ALLOW_SUCCESS,
        relation=EvidenceRelation(PROBABLE_MISMATCH),
        target=target,
        observed=observed,
        state=state,
        remaining_steps=19,
        min_remaining_steps=8,
        review_index=1,
    )
    assert second.actual_verdict == ALLOW_SUCCESS
    assert second.applied is False

    recovered = apply_evidence_policy(
        mode="enforce",
        proposed_verdict=ALLOW_SUCCESS,
        relation=EvidenceRelation(SUPPORTED_MATCH, "match", "match"),
        target=target,
        observed=observed,
        state=state,
        remaining_steps=18,
        min_remaining_steps=8,
        review_index=2,
    )
    assert recovered.actual_verdict == ALLOW_SUCCESS


def test_insufficient_evidence_first_enforce_falls_through():
    """A first enforce finalization landing INSUFFICIENT_EVIDENCE must not force a
    challenge under explicit_type_only — the verdict follows B's own proposal and no
    challenge state is consumed (the deterministic policy owns only CONFIRMED hard vetoes)."""
    target = extract_literal_target_profile("An invalid free in parse_file in parser.c.")
    observed = _profile()
    state = EvidencePolicyState()
    decision = apply_evidence_policy(
        mode="enforce",
        proposed_verdict=ALLOW_SUCCESS,
        relation=EvidenceRelation(INSUFFICIENT_EVIDENCE),
        target=target,
        observed=observed,
        state=state,
        remaining_steps=20,
        min_remaining_steps=8,
        review_index=0,
    )
    assert decision.actual_verdict == ALLOW_SUCCESS
    assert decision.applied is False
    assert decision.challenge is None
    assert state.challenge_count == 0


def test_hard_gate_scope_governs_override_surface():
    """Under explicit_type_only, only CONFIRMED_MISMATCH forces a redirect; a first-pass
    PROBABLE_MISMATCH or INSUFFICIENT_EVIDENCE falls through to B's proposed verdict with
    no challenge consumed."""
    target = extract_literal_target_profile("An invalid free in parse_file in parser.c.")
    observed = _profile()

    confirmed_state = EvidencePolicyState()
    confirmed = apply_evidence_policy(
        mode="enforce",
        proposed_verdict=ALLOW_SUCCESS,
        relation=EvidenceRelation(CONFIRMED_MISMATCH, "mismatch", "mismatch"),
        target=target,
        observed=observed,
        state=confirmed_state,
        remaining_steps=20,
        min_remaining_steps=8,
        review_index=0,
        hard_gate_scope="explicit_type_only",
    )
    assert confirmed.actual_verdict == CONTINUE_LOCAL
    assert confirmed.applied is True
    assert confirmed_state.challenge_count == 1

    for relation in (EvidenceRelation(PROBABLE_MISMATCH), EvidenceRelation(INSUFFICIENT_EVIDENCE)):
        state = EvidencePolicyState()
        decision = apply_evidence_policy(
            mode="enforce",
            proposed_verdict=ALLOW_SUCCESS,
            relation=relation,
            target=target,
            observed=observed,
            state=state,
            remaining_steps=20,
            min_remaining_steps=8,
            review_index=0,
            hard_gate_scope="explicit_type_only",
        )
        assert decision.actual_verdict == ALLOW_SUCCESS, relation.relation
        assert decision.applied is False, relation.relation
        assert state.challenge_count == 0, relation.relation


def test_evidence_policy_config_is_opt_in_and_rejects_double_gate():
    assert SynthesisConfig.from_config({"enabled": True}).evidence_policy_enabled is False
    assert SynthesisConfig.from_config(
        {
            "enabled": True,
            "reviewer": {"evidence_policy": {"enabled": True}},
        }
    ).evidence_policy_enabled is True
    legacy = FinalizationReviewConfig.from_config({"enabled": True}, description_mode=True)
    assert legacy.evidence_policy.enabled is False
    advisory = FinalizationReviewConfig.from_config(
        {
            "enabled": True,
            "evidence_policy": {
                "enabled": True,
                "mode": "advisory",
                "target_profile_mode": "literal_only",
            },
        },
        description_mode=True,
    )
    assert advisory.prompt_template.endswith("poc_reviewer_desc.j2")
    config = FinalizationReviewConfig.from_config(
        {
            "enabled": True,
            "judge_mode": "advisory",
            "evidence_policy": {
                "enabled": True,
                "mode": "enforce",
                "hard_gate_scope": "explicit_type_only",
                "max_evidence_challenges": 1,
                "target_profile_mode": "hybrid_llm",
            },
        },
        description_mode=True,
    )
    assert config.evidence_policy.mode == "enforce"
    assert config.prompt_template.endswith("poc_reviewer_evidence_gate.j2")

    try:
        FinalizationReviewConfig.from_config(
            {
                "judge_mode": "gate",
                "evidence_policy": {"enabled": True},
            },
            description_mode=True,
        )
    except ValueError as exc:
        assert "must remain 'advisory'" in str(exc)
    else:
        raise AssertionError("double gate configuration was accepted")


def test_allow_unverified_is_terminal_but_policy_owned():
    assert ALLOW_UNVERIFIED in ALLOW_VERDICTS
    state = FinalizationReviewToolState(
        steps=[],
        valid_step_indices=set(),
        synthesis_log_records=[],
        evidence_policy_enabled=True,
        evidence_policy_mode="enforce",
    )
    EmitFinalizationVerdictTool(state).forward(
        verdict=ALLOW_UNVERIFIED,
        reasoning="bounded evidence challenge ended",
    )
    assert state.emitted_verdict == ALLOW_UNVERIFIED

    legacy = FinalizationReviewToolState(
        steps=[], valid_step_indices=set(), synthesis_log_records=[]
    )
    EmitFinalizationVerdictTool(legacy).forward(
        verdict=ALLOW_UNVERIFIED,
        reasoning="not policy-owned",
    )
    assert legacy.emitted_verdict == CONTINUE_LOCAL


def test_reviewer_applies_one_redirect_then_logs_allow_unverified_v3(tmp_path, monkeypatch):
    config = FinalizationReviewConfig.from_config(
        {
            "enabled": True,
            "min_remaining_steps": 8,
            "evidence_policy": {
                "enabled": True,
                "mode": "enforce",
                "hard_gate_scope": "explicit_type_only",
                "max_evidence_challenges": 1,
                "target_profile_mode": "literal_only",
            },
        },
        description_mode=True,
    )
    reviewer = FinalizationReviewer(
        config=config,
        static_context={
            "instance_id": "example.cve",
            "work_dir": "/src/example",
            "bug_description": "An invalid free in parse_file in parser.c.",
        },
        synthesis_log_path=tmp_path / "synthesis_log.jsonl",
        log_dir=str(tmp_path),
    )
    reviewer.agent_ref = SimpleNamespace(max_steps=75, step_number=20)
    observed = _profile().to_dict()
    target = extract_literal_target_profile(
        "An invalid free in parse_file in parser.c."
    ).to_dict()

    def fake_run(*args, **kwargs):
        del args, kwargs
        return ReviewResult(
            verdict=ALLOW_SUCCESS,
            reasoning="candidate looks acceptable",
            attached_action="",
            evidence_steps=[],
            b_steps_used=1,
            b_tool_calls=[],
            degraded=False,
            degraded_reason="",
            prompt_hash="hash",
            dropped_invalid_evidence=[],
            observed_profile=observed,
            target_profile=target,
            replay_consistency={"consistent": True, "status": "consistent", "reasons": []},
            evidence_relation=EvidenceRelation(
                CONFIRMED_MISMATCH, "mismatch", "mismatch"
            ).to_dict(),
        )

    monkeypatch.setattr(reviewer, "_run_b_agent", fake_run)
    first_allow, _ = reviewer.invoke(
        artifact_status="crash", stop_reason="success", memory=SimpleNamespace(steps=[])
    )
    second_allow, _ = reviewer.invoke(
        artifact_status="same crash", stop_reason="success", memory=SimpleNamespace(steps=[])
    )
    assert first_allow is False
    assert second_allow is True
    records = [
        json.loads(line)
        for line in (tmp_path / "finalization_review.jsonl").read_text().splitlines()
    ]
    assert [record["schema_version"] for record in records] == ["v3", "v3"]
    assert [record["verdict"] for record in records] == [
        CONTINUE_LOCAL,
        ALLOW_UNVERIFIED,
    ]
    assert records[0]["challenge"]["max_follow_up_attempts"] == 1


_HBO_REPORT = (
    "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014 at pc 0x000\n"
    "READ of size 4 at 0x602000000014 thread T0\n"
    "    #0 0x4a1 in parse_file {src}/parser.c:10:5\n"
    "    #1 0x4b2 in main {src}/main.c:3:1\n"
)


def _grounded_repro_env(tmp_path):
    """Build a tmp git work_dir + testcase + harness shape so profile identity fields populate."""
    import subprocess as _sp

    src = tmp_path / "src"
    src.mkdir()
    _sp.run(["git", "init", "-q", str(src)], check=True)
    (src / "parser.c").write_text("int x;\n")
    _sp.run(["git", "-C", str(src), "add", "-A"], check=True)
    _sp.run(
        ["git", "-C", str(src), "-c", "user.email=t@t.t", "-c", "user.name=t", "commit", "-qm", "x"],
        check=True,
    )
    testcase = tmp_path / "testcase"
    testcase.mkdir()
    (testcase / "poc").write_bytes(b"AAAA")
    harness = tmp_path / "harness_shape.json"
    harness.write_text("{}")
    report = _HBO_REPORT.format(src=str(src))
    return src, testcase, harness, report


def test_evidence_chain_fires_confirmed_mismatch_via_self_confirmation(tmp_path):
    """Full seam: repro report -> profile -> self-confirmation consistency -> hard gate fires.

    This is the path the prior tests never exercised (they hand-built a consistent replay).
    Regression guard for the a_repro_profiles.jsonl wiring gap: when A's profile is absent, a
    second identical B repro must still satisfy replay consistency so an explicit type mismatch
    reaches CONFIRMED_MISMATCH instead of collapsing to INSUFFICIENT_EVIDENCE.
    """
    from smolagents.secb.sanitizer.profile import (
        ProfileContext,
        compare_replays,
        normalize_observed_crash,
    )

    src, testcase, harness, report = _grounded_repro_env(tmp_path)
    ctx = ProfileContext(
        candidate_dir=str(testcase),
        work_dir=str(src),
        harness_shape_path=str(harness),
        docker_image="img",
        completeness="complete",
    )
    observed = normalize_observed_crash(report, ctx)
    confirmation = normalize_observed_crash(report, ctx)  # stands in for the fallback run
    assert observed.sanitizer_family == "ASAN"
    assert observed.crash_type == "heap-buffer-overflow"
    assert observed.candidate_hash and observed.source_head and observed.harness_shape_hash

    replay = compare_replays(confirmation, observed)
    assert replay.consistent is True

    description = "An invalid free in parse_file in parser.c."
    target = extract_literal_target_profile(description)
    relation = assess_evidence_relation(description, target, observed, replay)
    assert relation.type_relation == "mismatch"
    assert relation.relation == CONFIRMED_MISMATCH


def test_run_repro_tool_runs_confirmation_when_a_profile_absent(tmp_path, monkeypatch):
    """The B repro tool must fire a second self-confirmation repro when a_repro_profiles.jsonl
    is missing, and record an evidence relation rather than skipping the evidence path."""
    import smolagents.secb.review.tools as tools_mod

    src, _testcase, harness, report = _grounded_repro_env(tmp_path)
    monkeypatch.setattr(tools_mod, "HARNESS_SHAPE_PATH", str(harness))

    calls = {"n": 0}
    real_run = tools_mod.subprocess.run

    def fake_run(cmd, *args, **kwargs):
        # Only intercept `secb repro`; let real git calls in the profiler pass through so
        # source identity is genuine.
        if cmd and cmd[0] == "secb":
            calls["n"] += 1
            return SimpleNamespace(returncode=1, stdout=report, stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(tools_mod.subprocess, "run", fake_run)

    state = tools_mod.FinalizationReviewToolState(
        steps=[],
        valid_step_indices=set(),
        synthesis_log_records=[],
        work_dir=str(src),
        target_signal="AddressSanitizer invalid-free",
        description_mode=True,
        bug_description="An invalid free in parse_file in parser.c.",
        evidence_policy_enabled=True,
        evidence_policy_mode="enforce",
        target_profile_mode="literal_only",
        log_dir=str(tmp_path),  # no a_repro_profiles.jsonl here -> fallback path
    )
    tool = tools_mod.RunSecbReproOnCurrentTestcaseTool(state)
    tool.forward()

    # One primary repro + one self-confirmation repro because A's profile is absent.
    assert calls["n"] == 2
    assert state.observed_profile is not None
    assert state.replay_consistency is not None
    assert state.evidence_relation is not None
