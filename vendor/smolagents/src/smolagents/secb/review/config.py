"""Config for B's finalization gate."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smolagents.secb.harness.config import SynthesisWorkspaceConfig


DEFAULT_POC_REVIEWER_PROMPT = "secb/review/prompts/poc_reviewer.j2"
DEFAULT_POC_REVIEWER_DESC_PROMPT = "secb/review/prompts/poc_reviewer_desc.j2"
DEFAULT_POC_REVIEWER_EVIDENCE_GATE_PROMPT = (
    "secb/review/prompts/poc_reviewer_evidence_gate.j2"
)

DEFAULT_FINALIZATION_REVIEW_TOOLS = (
    "read_a_trajectory",
    "read_a_step",
    "find_workspace_paths",
    "read_workspace_file",
    "grep_source",
    "read_synthesis_log",
    "run_secb_repro_on_current_testcase",
    "emit_finalization_verdict",
)


@dataclass(frozen=True)
class EvidencePolicyConfig:
    """Opt-in poc-desc evidence policy; absence preserves the legacy B path."""

    enabled: bool = False
    mode: str = "advisory"
    hard_gate_scope: str = "explicit_type_only"
    max_evidence_challenges: int = 1
    target_profile_mode: str = "literal_only"

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "EvidencePolicyConfig":
        config = config or {}
        enabled = bool(config.get("enabled", False))
        mode = str(config.get("mode") or "advisory").strip().lower()
        if mode not in {"advisory", "enforce"}:
            raise ValueError(
                "agent.pocharness.reviewer.evidence_policy.mode must be "
                f"'advisory' or 'enforce', got: {mode!r}"
            )
        hard_gate_scope = str(
            config.get("hard_gate_scope") or "explicit_type_only"
        ).strip().lower()
        if hard_gate_scope != "explicit_type_only":
            raise ValueError(
                "agent.pocharness.reviewer.evidence_policy.hard_gate_scope currently "
                "supports only 'explicit_type_only'."
            )
        max_challenges = int(config.get("max_evidence_challenges", 1))
        if max_challenges != 1:
            raise ValueError(
                "agent.pocharness.reviewer.evidence_policy.max_evidence_challenges "
                "must be 1 when evidence_policy is enabled."
            )
        target_profile_mode = str(
            config.get("target_profile_mode")
            or ("literal_only" if mode == "advisory" else "hybrid_llm")
        ).strip().lower()
        if target_profile_mode not in {"literal_only", "hybrid_llm"}:
            raise ValueError(
                "agent.pocharness.reviewer.evidence_policy.target_profile_mode must be "
                f"'literal_only' or 'hybrid_llm', got: {target_profile_mode!r}"
            )
        return cls(
            enabled=enabled,
            mode=mode,
            hard_gate_scope=hard_gate_scope,
            max_evidence_challenges=max_challenges,
            target_profile_mode=target_profile_mode,
        )


@dataclass
class FinalizationReviewConfig:
    """Config for B's finalization gate (orthogonal to C's synthesis)."""
    enabled: bool = False
    model_id: str = ""
    max_b_steps: int = 12
    barrier_timeout_seconds: int = 120
    max_blocks: int | None = None   # None = unlimited blocks
    min_remaining_steps: int = 8
    tools_allow: tuple[str, ...] = DEFAULT_FINALIZATION_REVIEW_TOOLS
    prompt_template: str = DEFAULT_POC_REVIEWER_PROMPT
    workspace: SynthesisWorkspaceConfig = field(default_factory=SynthesisWorkspaceConfig)
    description_mode: bool = False
    judge_mode: str = "advisory"   # "gate" | "advisory" -- see poc-desc_redesign_spec.md SS5
    evidence_policy: EvidencePolicyConfig = field(default_factory=EvidencePolicyConfig)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        fallback_model_id: str = "",
        description_mode: bool = False,
    ) -> "FinalizationReviewConfig":
        config = config or {}
        tools_allow = tuple(config.get("tools_allow") or DEFAULT_FINALIZATION_REVIEW_TOOLS)
        unknown = sorted(set(tools_allow) - set(DEFAULT_FINALIZATION_REVIEW_TOOLS))
        if unknown:
            raise ValueError(f"Unknown agent.pocharness.reviewer.tools_allow entries: {unknown}")
        if "emit_finalization_verdict" not in tools_allow:
            raise ValueError("agent.pocharness.reviewer.tools_allow must include 'emit_finalization_verdict'.")
        raw_max_blocks = config.get("max_blocks")
        max_blocks = None if raw_max_blocks is None else int(raw_max_blocks)
        judge_mode = str(config.get("judge_mode") or "advisory").strip().lower()
        if judge_mode not in {"gate", "advisory"}:
            raise ValueError(f"agent.pocharness.reviewer.judge_mode must be 'gate' or 'advisory', got: {judge_mode!r}")
        resolved_description_mode = bool(config.get("description_mode", description_mode))
        evidence_policy = EvidencePolicyConfig.from_config(config.get("evidence_policy"))
        if evidence_policy.enabled and judge_mode != "advisory":
            raise ValueError(
                "agent.pocharness.reviewer.judge_mode must remain 'advisory' when "
                "evidence_policy.enabled=true; the crash gate owns enforcement."
            )
        prompt_template = str(config.get("prompt_template") or DEFAULT_POC_REVIEWER_PROMPT)
        if resolved_description_mode and prompt_template == DEFAULT_POC_REVIEWER_PROMPT:
            prompt_template = DEFAULT_POC_REVIEWER_DESC_PROMPT
        if evidence_policy.enabled and evidence_policy.mode == "enforce" and prompt_template in {
            DEFAULT_POC_REVIEWER_PROMPT,
            DEFAULT_POC_REVIEWER_DESC_PROMPT,
        }:
            prompt_template = DEFAULT_POC_REVIEWER_EVIDENCE_GATE_PROMPT
        return cls(
            enabled=bool(config.get("enabled", False)),
            model_id=str(config.get("model_id") or fallback_model_id or ""),
            max_b_steps=int(config.get("max_b_steps", 12)),
            barrier_timeout_seconds=int(config.get("barrier_timeout_seconds", 120)),
            max_blocks=max_blocks,
            min_remaining_steps=int(config.get("min_remaining_steps", 8)),
            tools_allow=tools_allow,
            prompt_template=prompt_template,
            workspace=SynthesisWorkspaceConfig.from_config(config.get("workspace")),
            description_mode=resolved_description_mode,
            judge_mode=judge_mode,
            evidence_policy=evidence_policy,
        )
