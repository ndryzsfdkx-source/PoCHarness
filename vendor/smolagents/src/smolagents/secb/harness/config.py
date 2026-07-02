"""Config objects for the SEC-bench on-the-fly synthesis helper (Agent C)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_SYNTHESIS_MODEL = "openai/gpt-5.4-2026-03-05"
DEFAULT_SYNTHESIS_LOG_DIR = "/tmp/synthesis"
DEFAULT_SYNTHESIS_HELPER_PROMPT = "secb/harness/prompts/synthesis_helper.j2"
DEFAULT_SYNTHESIS_HELPER_DESC_PROMPT = "secb/harness/prompts/synthesis_helper_desc.j2"

DEFAULT_SYNTHESIS_TOOLS = (
    "read_testcase",
    "read_repro_output",
    "read_harness_shape",
    "find_workspace_paths",
    "read_workspace_file",
    "grep_source",
    "read_prior_helpers",
    "write_helper",
    "run_helper",
    "run_secb_repro_for_helper",
    "emit_helper",
)
RETIRED_SYNTHESIS_TOOLS = {"strict_crash_compare", "attempt_ledger_append"}

DEFAULT_READ_ALLOW_PATHS = ("/src/", "/usr/local/bin/", "/usr/bin/", "/testcase/", "Helpers/")
DEFAULT_READ_DENY_PATHS = ("/eval/", "/secb/")
DEFAULT_WRITE_PATHS = ("Helpers/",)


@dataclass
class SynthesisWorkspaceConfig:
    read_allow_paths: tuple[str, ...] = DEFAULT_READ_ALLOW_PATHS
    read_deny_paths: tuple[str, ...] = DEFAULT_READ_DENY_PATHS
    write_paths: tuple[str, ...] = DEFAULT_WRITE_PATHS
    read_max_chars: int = 8000
    grep_max_matches: int = 50
    grep_timeout_secs: int = 5

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "SynthesisWorkspaceConfig":
        config = config or {}
        return cls(
            read_allow_paths=tuple(config.get("read_allow_paths") or DEFAULT_READ_ALLOW_PATHS),
            read_deny_paths=tuple(config.get("read_deny_paths") or DEFAULT_READ_DENY_PATHS),
            write_paths=tuple(config.get("write_paths") or DEFAULT_WRITE_PATHS),
            read_max_chars=int(config.get("read_max_chars", 8000)),
            grep_max_matches=int(config.get("grep_max_matches", 50)),
            grep_timeout_secs=int(config.get("grep_timeout_secs", 5)),
        )


@dataclass
class SynthesisAgentBackendConfig:
    max_c_steps: int = 30
    continuation_max_c_steps: int | None = None
    barrier_timeout_seconds: int = 180
    tools_allow: tuple[str, ...] = DEFAULT_SYNTHESIS_TOOLS
    prompt_template: str = DEFAULT_SYNTHESIS_HELPER_PROMPT
    workspace: SynthesisWorkspaceConfig = field(default_factory=SynthesisWorkspaceConfig)
    description_mode: bool = False

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        description_mode: bool = False,
    ) -> "SynthesisAgentBackendConfig":
        config = config or {}
        configured_tools = tuple(config.get("tools_allow") or DEFAULT_SYNTHESIS_TOOLS)
        unknown = sorted(
            set(configured_tools) - set(DEFAULT_SYNTHESIS_TOOLS) - RETIRED_SYNTHESIS_TOOLS
        )
        if unknown:
            raise ValueError(f"Unknown synthesis.agent_backend.tools_allow entries: {unknown}")
        tools_allow = tuple(
            name for name in configured_tools if name not in RETIRED_SYNTHESIS_TOOLS
        )
        if "emit_helper" not in tools_allow:
            raise ValueError("synthesis.agent_backend.tools_allow must include 'emit_helper'.")
        raw_continuation_max = config.get("continuation_max_c_steps")
        resolved_description_mode = bool(config.get("description_mode", description_mode))
        prompt_template = str(config.get("prompt_template") or DEFAULT_SYNTHESIS_HELPER_PROMPT)
        if resolved_description_mode and prompt_template == DEFAULT_SYNTHESIS_HELPER_PROMPT:
            prompt_template = DEFAULT_SYNTHESIS_HELPER_DESC_PROMPT
        return cls(
            max_c_steps=int(config.get("max_c_steps", 30)),
            continuation_max_c_steps=(
                None if raw_continuation_max is None else int(raw_continuation_max)
            ),
            barrier_timeout_seconds=int(config.get("barrier_timeout_seconds", 180)),
            tools_allow=tools_allow,
            prompt_template=prompt_template,
            workspace=SynthesisWorkspaceConfig.from_config(config.get("workspace")),
            description_mode=resolved_description_mode,
        )


@dataclass
class SynthesisFinalizationGuardConfig:
    enabled: bool = False
    min_remaining_steps: int = 8

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "SynthesisFinalizationGuardConfig":
        config = config or {}
        return cls(
            enabled=bool(config.get("enabled", False)),
            min_remaining_steps=int(config.get("min_remaining_steps", 8)),
        )


@dataclass
class SynthesisConfig:
    enabled: bool = False
    model_id: str = DEFAULT_SYNTHESIS_MODEL
    log_dir: str = DEFAULT_SYNTHESIS_LOG_DIR
    agent_backend: SynthesisAgentBackendConfig = field(default_factory=SynthesisAgentBackendConfig)
    finalization_guard: SynthesisFinalizationGuardConfig = field(default_factory=SynthesisFinalizationGuardConfig)
    model_transport_kwargs: dict = field(default_factory=dict)
    description_mode: bool = False
    evidence_policy_enabled: bool = False

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        default_model_id: str | None = None,
        description_mode: bool = False,
    ) -> "SynthesisConfig":
        config = config or {}
        model_id = str(config.get("model_id") or default_model_id or DEFAULT_SYNTHESIS_MODEL)
        resolved_description_mode = bool(config.get("description_mode", description_mode))
        finalization_review = config.get("reviewer") or {}
        evidence_policy = finalization_review.get("evidence_policy") or {}
        return cls(
            enabled=bool(config.get("enabled", False)),
            model_id=model_id,
            log_dir=str(config.get("log_dir") or DEFAULT_SYNTHESIS_LOG_DIR),
            agent_backend=SynthesisAgentBackendConfig.from_config(
                config.get("helper"), description_mode=resolved_description_mode
            ),
            finalization_guard=SynthesisFinalizationGuardConfig.from_config(config.get("finalization_guard")),
            description_mode=resolved_description_mode,
            evidence_policy_enabled=bool(evidence_policy.get("enabled", False)),
        )
