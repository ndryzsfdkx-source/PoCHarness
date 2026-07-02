"""Logging for B finalization review records."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

LOG_NAME = "finalization_review.jsonl"


def write_finalization_review_log(
    *,
    log_dir: str,
    verdict: str,
    reasoning: str,
    attached_action: str,
    evidence_steps: list[int],
    b_steps_used: int,
    b_tool_calls: list[dict[str, Any]],
    degraded: bool,
    degraded_reason: str,
    prompt_hash: str,
    rule_guard_decision: dict[str, Any],
    remaining_steps: int,
    block_index: int,
    artifact_status: str,
    stop_reason: str,
    instance_id: str,
    block_index_input: int | None = None,
    block_count_after: int | None = None,
    dropped_invalid_evidence: list[dict[str, Any]] | None = None,
    b_input_tokens: int = 0,
    b_output_tokens: int = 0,
    b_cost: float | None = None,
    b_termination_reason: str = "",
    repro_result: dict[str, Any] | None = None,
    proposed_verdict: str = "",
    recommended_verdict: str = "",
    policy_mode: str = "legacy",
    policy_applied: bool = False,
    challenge: dict[str, Any] | None = None,
    observed_profile: dict[str, Any] | None = None,
    target_profile: dict[str, Any] | None = None,
    replay_consistency: dict[str, Any] | None = None,
    evidence_relation: dict[str, Any] | None = None,
) -> None:
    try:
        log_path = Path(log_dir) / LOG_NAME
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": "v3" if policy_mode != "legacy" else "v2",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "instance_id": instance_id,
            "verdict": verdict,
            "reasoning": reasoning,
            "attached_action": attached_action,
            "evidence_steps": evidence_steps,
            "b_steps_used": b_steps_used,
            "b_input_tokens": b_input_tokens,
            "b_output_tokens": b_output_tokens,
            "b_termination_reason": b_termination_reason,
            "b_tool_calls": b_tool_calls,
            "degraded": degraded,
            "degraded_reason": degraded_reason,
            "prompt_hash": prompt_hash,
            "rule_guard_decision": rule_guard_decision,
            "remaining_steps": remaining_steps,
            "block_index": block_index,
            "block_index_input": block_index if block_index_input is None else block_index_input,
            "block_count_after": block_index if block_count_after is None else block_count_after,
            "artifact_status": artifact_status[:500],
            "stop_reason": stop_reason[:500],
            "repro_result": repro_result or {},
            "dropped_invalid_evidence": dropped_invalid_evidence or [],
        }
        if policy_mode != "legacy":
            record.update(
                {
                    "proposed_verdict": proposed_verdict or verdict,
                    "recommended_verdict": recommended_verdict or verdict,
                    "policy_mode": policy_mode,
                    "policy_applied": bool(policy_applied),
                    "challenge": challenge or {},
                    "observed_profile": observed_profile or {},
                    "target_profile": target_profile or {},
                    "replay_consistency": replay_consistency or {},
                    "evidence_relation": evidence_relation or {},
                }
            )
        if b_cost is not None:
            record["b_cost"] = b_cost
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        import sys
        print(f"WARNING: finalization_review log write failed: {exc}", file=sys.stderr)
