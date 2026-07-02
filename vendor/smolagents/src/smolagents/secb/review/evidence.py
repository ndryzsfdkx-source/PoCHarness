"""Target extraction, evidence relation, and bounded policy for the Agent B crash gate."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from smolagents.secb.sanitizer.profile import ObservedCrashProfile, ReplayConsistency


TARGET_PROFILE_VERSION = "target_profile/v1"
RELATION_PROMPT_VERSION = "evidence_relation/v1"

CONFIRMED_MISMATCH = "CONFIRMED_MISMATCH"
PROBABLE_MISMATCH = "PROBABLE_MISMATCH"
SUPPORTED_MATCH = "SUPPORTED_MATCH"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
RELATIONS = {
    CONFIRMED_MISMATCH,
    PROBABLE_MISMATCH,
    SUPPORTED_MATCH,
    INSUFFICIENT_EVIDENCE,
}

ALLOW_SUCCESS = "ALLOW_SUCCESS"
ALLOW_EXHAUSTED = "ALLOW_EXHAUSTED"
ALLOW_UNVERIFIED = "ALLOW_UNVERIFIED"
CONTINUE_LOCAL = "CONTINUE_LOCAL"


@dataclass
class TargetFact:
    field: str
    value: str
    evidence_text: str
    evidence_span: tuple[int, int]
    provenance: str = "explicit"
    usable_for_hard_gate: bool = False
    source: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_span"] = list(self.evidence_span)
        return value


@dataclass
class TargetProfile:
    description_hash: str
    prompt_version: str = TARGET_PROFILE_VERSION
    model_version: str = ""
    facts: list[TargetFact] = field(default_factory=list)
    cache_key: str = ""
    extractor_error: str = ""

    def facts_for(self, field_name: str) -> list[TargetFact]:
        return [fact for fact in self.facts if fact.field == field_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "description_hash": self.description_hash,
            "prompt_version": self.prompt_version,
            "model_version": self.model_version,
            "facts": [fact.to_dict() for fact in self.facts],
            "cache_key": self.cache_key,
            "extractor_error": self.extractor_error,
        }


@dataclass
class EvidenceRelation:
    relation: str
    type_relation: str = "unknown"
    path_relation: str = "unknown"
    reasoning: str = ""
    llm_relation: str = ""
    llm_confidence: float = 0.0
    llm_reasoning: str = ""
    judge_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceChallenge:
    challenge_id: str
    candidate_hash: str
    required_new_evidence: str
    challenge_count: int
    max_follow_up_attempts: int = 1
    prior_evidence_summary: str = ""
    issued_review_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidencePolicyState:
    challenge_count: int = 0
    last_challenge: EvidenceChallenge | None = None


@dataclass
class PolicyDecision:
    actual_verdict: str
    recommended_verdict: str
    attached_action: str
    challenge: EvidenceChallenge | None = None
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual_verdict": self.actual_verdict,
            "recommended_verdict": self.recommended_verdict,
            "attached_action": self.attached_action,
            "challenge": self.challenge.to_dict() if self.challenge else None,
            "applied": self.applied,
        }


_BUG_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bheap[- ]use[- ]after[- ]free\b", re.I), "use-after-free"),
    (re.compile(r"\buse[- ]after[- ]free\b", re.I), "use-after-free"),
    (re.compile(r"\bdouble[- ]free\b", re.I), "double-free"),
    (re.compile(r"\b(?:invalid|bad)\s+free\b", re.I), "invalid-free"),
    (re.compile(r"\bheap[- ](?:based[- ])?buffer[- ]overflow\b", re.I), "heap-buffer-overflow"),
    (re.compile(r"\bstack[- ](?:based[- ])?buffer[- ]overflow\b", re.I), "stack-buffer-overflow"),
    (re.compile(r"\bglobal[- ]buffer[- ]overflow\b", re.I), "global-buffer-overflow"),
    (re.compile(r"\b(?:out[- ]of[- ]bounds|out[- ]of[- ]bound|oob)\s+read\b|\bbuffer over-read\b", re.I), "oob-read"),
    (re.compile(r"\b(?:out[- ]of[- ]bounds|out[- ]of[- ]bound|oob)\s+write\b", re.I), "oob-write"),
    (re.compile(r"\bnull(?:[- ]pointer)?\s+deref(?:erence)?\b", re.I), "null-deref"),
    (re.compile(r"\b(?:sigsegv|segv|segmentation fault|segfault)\b", re.I), "SEGV"),
    (re.compile(r"\bmemory leaks?\b", re.I), "memory-leak"),
]
_QUOTED_SYMBOL_RE = re.compile(r"[`'\"]([A-Za-z_~][A-Za-z0-9_:~<>.-]{2,})[`'\"]")
_CALL_SYMBOL_RE = re.compile(r"\b([A-Za-z_~][A-Za-z0-9_:~<>]*)\s*\(\)")
_QUALIFIED_SYMBOL_RE = re.compile(r"\b([A-Za-z_~][A-Za-z0-9_~]+(?:::[A-Za-z_~][A-Za-z0-9_~]+)+)\b")
_FUNCTION_PREFIX_RE = re.compile(
    r"\b(?:function|routine|method)\s+(?:named\s+)?([A-Za-z_~][A-Za-z0-9_:~<>]*)\b",
    re.I,
)
_FUNCTION_SUFFIX_RE = re.compile(
    r"\b([A-Za-z_~][A-Za-z0-9_:~<>]*)\s+(?:function|routine|method)\b",
    re.I,
)
_IN_SYMBOL_RE = re.compile(
    r"\b(?:in|via|at|within|from)\s+(?:the\s+)?([A-Za-z_~][A-Za-z0-9_:~<>]*)\b",
    re.I,
)
_AND_CODE_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(and\s+([A-Za-z_][A-Za-z0-9_]*)\)\s+code\b",
    re.I,
)
_VIA_FILE_SYMBOL_RE = re.compile(
    r"\bvia\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:\.\.?/|/)?"
    r"[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*\.(?:c|cc|cpp|cxx)(?::\d+)?",
    re.I,
)
_FILE_RE = re.compile(
    r"(?<![A-Za-z0-9_])((?:\.\.?/|/)?[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*"
    r"\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx)(?::\d+)?)\b",
    re.I,
)
_SYMBOL_STOPWORDS = {
    "a",
    "an",
    "the",
    "github",
    "repository",
    "image",
    "file",
    "component",
    "homebrew",
    "crafted",
    "memory",
    "buffer",
    "pointer",
    "in",
    "via",
    "at",
}


def _description_hash(description: str) -> str:
    return hashlib.sha256(str(description or "").encode("utf-8")).hexdigest()


def _cache_key(description: str, model_version: str) -> str:
    raw = f"{_description_hash(description)}\0{TARGET_PROFILE_VERSION}\0{model_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _looks_like_symbol(value: str) -> bool:
    lower = value.lower()
    return (
        lower not in _SYMBOL_STOPWORDS
        and len(value) >= 3
        and not value.isupper()
        and (
            "_" in value
            or "::" in value
            or any(char.isupper() for char in value[1:])
        )
    )


def _fact(
    *,
    description: str,
    field_name: str,
    value: str,
    start: int,
    end: int,
    provenance: str = "explicit",
    source: str = "deterministic",
) -> TargetFact | None:
    if start < 0 or end <= start or end > len(description):
        return None
    evidence_text = description[start:end]
    hard = field_name == "bug_class" and provenance == "explicit" and any(
        normalized == value and pattern.fullmatch(evidence_text.strip())
        for pattern, normalized in _BUG_PATTERNS
    )
    return TargetFact(
        field=field_name,
        value=value,
        evidence_text=evidence_text,
        evidence_span=(start, end),
        provenance=provenance,
        usable_for_hard_gate=hard,
        source=source,
    )


def extract_literal_target_profile(description: str) -> TargetProfile:
    text = str(description or "")
    facts: list[TargetFact] = []
    seen: set[tuple[str, str, int, int]] = set()

    def add(field_name: str, value: str, start: int, end: int) -> None:
        key = (field_name, value, start, end)
        if key in seen:
            return
        item = _fact(
            description=text,
            field_name=field_name,
            value=value,
            start=start,
            end=end,
        )
        if item is not None:
            seen.add(key)
            facts.append(item)

    for pattern, normalized in _BUG_PATTERNS:
        match = pattern.search(text)
        if match:
            add("bug_class", normalized, match.start(), match.end())
            break

    symbol_patterns: Iterable[re.Pattern[str]] = (
        _QUOTED_SYMBOL_RE,
        _CALL_SYMBOL_RE,
        _QUALIFIED_SYMBOL_RE,
        _FUNCTION_PREFIX_RE,
        _FUNCTION_SUFFIX_RE,
        _IN_SYMBOL_RE,
    )
    for pattern in symbol_patterns:
        for match in pattern.finditer(text):
            value = match.group(1)
            if _looks_like_symbol(value):
                add("function", value, match.start(1), match.end(1))
    for match in _AND_CODE_RE.finditer(text):
        for group in (1, 2):
            value = match.group(group)
            add("function", value, match.start(group), match.end(group))
    for match in _VIA_FILE_SYMBOL_RE.finditer(text):
        add("function", match.group(1), match.start(1), match.end(1))
    for match in _FILE_RE.finditer(text):
        add("file", match.group(1), match.start(1), match.end(1))

    return TargetProfile(description_hash=_description_hash(text), facts=facts)


_TARGET_PROMPT = """Extract structured target-vulnerability facts from the bug description.
Return only facts supported by the exact original text. Do not repair typos or infer hidden names.
Allowed fields: bug_class, function, file, subsystem, input_format, trigger_constraint.
For every fact return value, exact evidence_text, zero-based [start,end] evidence_span, and
provenance explicit or paraphrased. Respond as JSON: {{\"facts\": [...]}}.

Bug description:
{description}
"""


def _validated_model_facts(description: str, payload: dict[str, Any]) -> list[TargetFact]:
    facts: list[TargetFact] = []
    for raw in payload.get("facts") or []:
        if not isinstance(raw, dict):
            continue
        field_name = str(raw.get("field") or "").strip()
        if field_name not in {
            "bug_class",
            "function",
            "file",
            "subsystem",
            "input_format",
            "trigger_constraint",
        }:
            continue
        span = raw.get("evidence_span")
        if not isinstance(span, list) or len(span) != 2:
            continue
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start or end > len(description):
            continue
        evidence = description[start:end]
        if evidence != str(raw.get("evidence_text") or ""):
            continue
        provenance = str(raw.get("provenance") or "paraphrased").strip().lower()
        if provenance not in {"explicit", "paraphrased"}:
            provenance = "paraphrased"
        value = str(raw.get("value") or "").strip()
        if not value:
            continue
        # A model-proposed hard type is accepted only when the exact evidence span maps to
        # the same frozen deterministic class vocabulary.
        if field_name == "bug_class":
            normalized = next(
                (
                    cls
                    for pattern, cls in _BUG_PATTERNS
                    if pattern.fullmatch(evidence.strip())
                ),
                None,
            )
            if normalized is None:
                provenance = "paraphrased"
            else:
                value = normalized
        item = _fact(
            description=description,
            field_name=field_name,
            value=value,
            start=start,
            end=end,
            provenance=provenance,
            source="llm",
        )
        if item is not None:
            facts.append(item)
    return facts


def build_target_profile(
    description: str,
    *,
    mode: str = "literal_only",
    model: Any = None,
    model_version: str = "",
    cache_path: str | Path | None = None,
) -> TargetProfile:
    profile = extract_literal_target_profile(description)
    profile.model_version = model_version
    profile.cache_key = _cache_key(description, model_version)
    path = Path(cache_path) if cache_path else None
    if path and path.exists():
        try:
            cached = target_profile_from_dict(json.loads(path.read_text(encoding="utf-8")))
            if cached.cache_key == profile.cache_key:
                return cached
        except Exception:
            pass
    if mode == "hybrid_llm" and model is not None:
        try:
            prompt = _TARGET_PROMPT.format(description=str(description or "")[:6000])
            response = model.generate(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.content)
            model_facts = _validated_model_facts(str(description or ""), payload)
            existing = {(fact.field, fact.value, fact.evidence_span) for fact in profile.facts}
            profile.facts.extend(
                fact
                for fact in model_facts
                if (fact.field, fact.value, fact.evidence_span) not in existing
            )
        except Exception as exc:
            profile.extractor_error = str(exc)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile.to_dict(), indent=2) + "\n", encoding="utf-8")
    return profile


def target_profile_from_dict(value: dict[str, Any]) -> TargetProfile:
    facts = [
        TargetFact(
            field=str(raw.get("field") or ""),
            value=str(raw.get("value") or ""),
            evidence_text=str(raw.get("evidence_text") or ""),
            evidence_span=tuple(raw.get("evidence_span") or (0, 0)),
            provenance=str(raw.get("provenance") or "paraphrased"),
            usable_for_hard_gate=bool(raw.get("usable_for_hard_gate")),
            source=str(raw.get("source") or "deterministic"),
        )
        for raw in value.get("facts") or []
    ]
    return TargetProfile(
        description_hash=str(value.get("description_hash") or ""),
        prompt_version=str(value.get("prompt_version") or TARGET_PROFILE_VERSION),
        model_version=str(value.get("model_version") or ""),
        facts=facts,
        cache_key=str(value.get("cache_key") or ""),
        extractor_error=str(value.get("extractor_error") or ""),
    )


def _observed_kind(profile: ObservedCrashProfile) -> str | None:
    value = str(profile.crash_type or "").lower()
    aliases = {
        "heap-use-after-free": "use-after-free",
        "use-after-free": "use-after-free",
        "double-free": "double-free",
        "attempting-double-free": "double-free",
        "invalid-free": "invalid-free",
        "bad-free": "invalid-free",
        "alloc-dealloc-mismatch": "invalid-free",
        "heap-buffer-overflow": "heap-buffer-overflow",
        "stack-buffer-overflow": "stack-buffer-overflow",
        "global-buffer-overflow": "global-buffer-overflow",
        "segv": "SEGV",
        "deadlysignal": "DEADLYSIGNAL",
        "memory-leak": "memory-leak",
    }
    if value.startswith("attempting free on address"):
        return "invalid-free"
    return aliases.get(value, profile.crash_type)


def type_relation(target_class: str | None, observed: ObservedCrashProfile) -> str:
    if not target_class or not observed.sanitizer_family:
        return "unknown"
    actual = _observed_kind(observed)
    if target_class == "memory-leak":
        return "match" if observed.lsan_status != "not_present" else "mismatch"
    if observed.lsan_status == "only_signal":
        return "unknown"
    if target_class in {"null-deref", "SEGV"} and actual in {"SEGV", "DEADLYSIGNAL"}:
        # DEADLYSIGNAL is compatible, but generic signal-only evidence is not a strong
        # positive match without null-address or project-path evidence.
        return "weak_match" if actual == "DEADLYSIGNAL" else "match"
    if target_class == actual:
        return "match"
    if target_class in {"oob-read", "oob-write"} and actual in {
        "heap-buffer-overflow",
        "stack-buffer-overflow",
        "global-buffer-overflow",
    }:
        expected_access = "READ" if target_class == "oob-read" else "WRITE"
        return "match" if observed.access_type == expected_access else "unknown"
    known = {
        "use-after-free",
        "double-free",
        "invalid-free",
        "heap-buffer-overflow",
        "stack-buffer-overflow",
        "global-buffer-overflow",
        "oob-read",
        "oob-write",
        "null-deref",
        "SEGV",
        "memory-leak",
    }
    return "mismatch" if target_class in known and actual in known else "unknown"


def _path_relation(target: TargetProfile, observed: ObservedCrashProfile) -> str:
    functions = [fact.value for fact in target.facts_for("function")]
    files = [Path(fact.value.split(":", 1)[0]).name for fact in target.facts_for("file")]
    if not functions and not files:
        return "unknown"
    if not observed.project_frames:
        return "unknown"
    for frame in observed.project_frames:
        function = str(frame.get("function") or "")
        basename = Path(str(frame.get("file") or "")).name
        if any(name and (name == function or name in function) for name in functions):
            return "match"
        if basename and basename in files:
            return "match"
    return "mismatch"


_RELATION_PROMPT = """Judge the relation between a target bug description and a candidate crash.
Use only the supplied description and candidate evidence. Return JSON with relation one of
PROBABLE_MISMATCH, SUPPORTED_MATCH, INSUFFICIENT_EVIDENCE, confidence, and reasoning.
Never return CONFIRMED_MISMATCH; deterministic policy owns hard mismatches.

Description:
{description}

Target profile:
{target}

Observed profile:
{observed}
"""


def assess_evidence_relation(
    description: str,
    target: TargetProfile,
    observed: ObservedCrashProfile,
    replay: ReplayConsistency,
    *,
    model: Any = None,
) -> EvidenceRelation:
    hard_classes = [fact.value for fact in target.facts_for("bug_class") if fact.usable_for_hard_gate]
    target_class = hard_classes[0] if hard_classes else None
    t_relation = type_relation(target_class, observed)
    p_relation = _path_relation(target, observed)
    if observed.report_completeness != "complete" or not replay.consistent:
        return EvidenceRelation(
            INSUFFICIENT_EVIDENCE,
            type_relation=t_relation,
            path_relation=p_relation,
            reasoning="profile incomplete or A/B replay inconsistent",
        )
    if t_relation == "mismatch":
        return EvidenceRelation(
            CONFIRMED_MISMATCH,
            type_relation=t_relation,
            path_relation=p_relation,
            reasoning="explicit target type is deterministically incompatible with the stable candidate crash",
        )

    llm_relation = ""
    llm_confidence = 0.0
    llm_reasoning = ""
    judge_error = ""
    if model is not None:
        try:
            response = model.generate(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _RELATION_PROMPT.format(
                                    description=str(description or "")[:6000],
                                    target=json.dumps(target.to_dict(), sort_keys=True)[:8000],
                                    observed=json.dumps(observed.canonical_crash_payload(), sort_keys=True)[:8000],
                                ),
                            }
                        ],
                    }
                ],
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.content)
            candidate = str(payload.get("relation") or "").upper()
            if candidate in {PROBABLE_MISMATCH, SUPPORTED_MATCH, INSUFFICIENT_EVIDENCE}:
                llm_relation = candidate
            llm_confidence = max(0.0, min(float(payload.get("confidence", 0.0)), 1.0))
            llm_reasoning = str(payload.get("reasoning") or "")[:1000]
        except Exception as exc:
            judge_error = str(exc)

    target_is_leak = target_class == "memory-leak"
    lsan_positive = observed.lsan_status != "not_present" and target_is_leak
    positive_type = t_relation == "match" or lsan_positive
    positive_path = p_relation == "match"
    if positive_type and positive_path:
        relation = SUPPORTED_MATCH
        reasoning = "stable repro has positive target type and project-path evidence"
    elif p_relation == "mismatch" or llm_relation == PROBABLE_MISMATCH:
        relation = PROBABLE_MISMATCH
        reasoning = "location or relation evidence indicates a possible mismatch"
    else:
        relation = INSUFFICIENT_EVIDENCE
        reasoning = "available evidence does not positively support the target and is not a hard mismatch"
    return EvidenceRelation(
        relation,
        type_relation=t_relation,
        path_relation=p_relation,
        reasoning=reasoning,
        llm_relation=llm_relation,
        llm_confidence=llm_confidence,
        llm_reasoning=llm_reasoning,
        judge_error=judge_error,
    )


def _required_new_evidence(relation: EvidenceRelation, observed: ObservedCrashProfile) -> str:
    if relation.relation == CONFIRMED_MISMATCH:
        return "candidate_change_with_complete_repro"
    if relation.path_relation == "mismatch":
        return "target_symbol_in_project_stack_or_verified_source_edge"
    if not observed.project_frames:
        return "first_project_frame"
    return "complete_structured_repro"


def _challenge_action(relation: EvidenceRelation, target: TargetProfile, observed: ObservedCrashProfile) -> str:
    target_classes = [fact.value for fact in target.facts_for("bug_class")]
    target_functions = [fact.value for fact in target.facts_for("function")]
    observed_frame = (observed.first_project_frame or {}).get("function") or "no project frame"
    if relation.relation == CONFIRMED_MISMATCH:
        return (
            f"The stable candidate crash is {observed.crash_type or observed.sanitizer_family}, "
            "but the description explicitly requires "
            f"{target_classes[0] if target_classes else 'a different crash type'}. "
            "Change the candidate toward the described crash class and produce a complete "
            "structured repro before submitting again."
        )
    if relation.path_relation == "mismatch" and target_functions:
        return (
            f"The candidate's first project frame is {observed_frame}, while the description names "
            f"{', '.join(target_functions)}. Produce one concrete result showing the target "
            "symbol in the project stack "
            "or a verified caller/callee source edge before submitting again."
        )
    return (
        "The current evidence cannot connect this sanitizer result to the described vulnerability. "
        "Produce one complete structured repro with a first project frame before submitting again."
    )


def _enforced_recommendation(
    *,
    proposed_verdict: str,
    relation: EvidenceRelation,
    target: TargetProfile,
    observed: ObservedCrashProfile,
    state: EvidencePolicyState,
    remaining_steps: int,
    min_remaining_steps: int,
    review_index: int,
    degraded: bool,
    mutate: bool,
    hard_gate_scope: str = "explicit_type_only",
) -> PolicyDecision:
    no_signal = observed.sanitizer_family is None
    if relation.relation == SUPPORTED_MATCH:
        if proposed_verdict == ALLOW_SUCCESS:
            return PolicyDecision(ALLOW_SUCCESS, ALLOW_SUCCESS, "", applied=True)
        return PolicyDecision(proposed_verdict, proposed_verdict, "", applied=False)
    # explicit_type_only: the deterministic policy only forces a redirect on the
    # CONFIRMED_MISMATCH hard veto. PROBABLE_MISMATCH and INSUFFICIENT_EVIDENCE fall
    # through to B's own proposed verdict (B may still choose CONTINUE_LOCAL on its own
    # judgment, but the deterministic policy will not force it). This keeps the runtime
    # surface equal to the advertised precision; broadening it would re-introduce the
    # demonstrated correct-solve false-redirect harm (see
    # analysis/agent-b-crash-gate/prelaunch-calibration/: 24/50 strict passes).
    if hard_gate_scope == "explicit_type_only" and relation.relation != CONFIRMED_MISMATCH:
        return PolicyDecision(proposed_verdict, proposed_verdict, "", applied=False)
    if state.challenge_count >= 1:
        verdict = ALLOW_EXHAUSTED if no_signal and not degraded else ALLOW_UNVERIFIED
        return PolicyDecision(verdict, verdict, "", applied=True)
    if no_signal and proposed_verdict == ALLOW_EXHAUSTED and remaining_steps < min_remaining_steps:
        return PolicyDecision(ALLOW_EXHAUSTED, ALLOW_EXHAUSTED, "", applied=True)
    actionable = remaining_steps >= min_remaining_steps
    if actionable:
        required = _required_new_evidence(relation, observed)
        candidate_prefix = (observed.candidate_hash or "missing")[:12]
        challenge = EvidenceChallenge(
            challenge_id=f"b-evidence-v2:{review_index}:{candidate_prefix}",
            candidate_hash=observed.candidate_hash,
            required_new_evidence=required,
            challenge_count=1,
            prior_evidence_summary=(
                f"relation={relation.relation}; type={relation.type_relation}; "
                f"path={relation.path_relation}; crash={observed.crash_type or observed.sanitizer_family}"
            ),
            issued_review_index=review_index,
        )
        if mutate:
            state.challenge_count = 1
            state.last_challenge = challenge
        action = _challenge_action(relation, target, observed)
        return PolicyDecision(CONTINUE_LOCAL, CONTINUE_LOCAL, action, challenge=challenge, applied=True)
    verdict = ALLOW_EXHAUSTED if no_signal and not degraded else ALLOW_UNVERIFIED
    return PolicyDecision(verdict, verdict, "", applied=True)


def apply_evidence_policy(
    *,
    mode: str,
    proposed_verdict: str,
    relation: EvidenceRelation,
    target: TargetProfile,
    observed: ObservedCrashProfile,
    state: EvidencePolicyState,
    remaining_steps: int,
    min_remaining_steps: int,
    review_index: int,
    degraded: bool = False,
    hard_gate_scope: str = "explicit_type_only",
) -> PolicyDecision:
    recommendation = _enforced_recommendation(
        proposed_verdict=proposed_verdict,
        relation=relation,
        target=target,
        observed=observed,
        state=state,
        remaining_steps=remaining_steps,
        min_remaining_steps=min_remaining_steps,
        review_index=review_index,
        degraded=degraded,
        mutate=mode == "enforce",
        hard_gate_scope=hard_gate_scope,
    )
    if mode == "advisory":
        return PolicyDecision(
            actual_verdict=proposed_verdict,
            recommended_verdict=recommendation.recommended_verdict,
            attached_action="",
            challenge=recommendation.challenge,
            applied=False,
        )
    return recommendation
