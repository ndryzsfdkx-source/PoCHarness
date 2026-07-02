"""Shared trajectory formatting helpers."""
from __future__ import annotations

import json
import re
from typing import Any

from ...memory import ActionStep


TRAJECTORY_FORMATTED = "formatted"
TRAJECTORY_RAW = "raw"
VALID_TRAJECTORY_MODES = {TRAJECTORY_FORMATTED, TRAJECTORY_RAW}


def truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    head = max_chars // 2
    tail = max_chars - head
    return value[:head] + "\n...[truncated]...\n" + value[-tail:]


def extract_tool_argument(tool_call) -> str:
    arguments = getattr(tool_call, "arguments", "")
    if isinstance(arguments, dict):
        if "command" in arguments:
            return str(arguments.get("command", ""))
        return json.dumps(arguments, sort_keys=True)
    return str(arguments)


def format_tool_calls(step: ActionStep) -> str:
    parts = []
    for tc in getattr(step, "tool_calls", None) or []:
        parts.append(f"{tc.name}: {extract_tool_argument(tc)}")
    return "; ".join(parts) or "(no tool call)"


def clean_observation(observation: str) -> str:
    lines = []
    blank = False
    for line in observation.splitlines():
        stripped = line.rstrip()
        if not stripped:
            if not blank:
                lines.append("")
            blank = True
            continue
        blank = False
        lines.append(stripped)
    return "\n".join(lines).strip()


def format_step_full(step: ActionStep, max_observation_chars: int) -> dict[str, Any]:
    return {
        "step_number": step.step_number,
        "model_output": truncate(str(step.model_output or ""), 2000),
        "tool_calls": format_tool_calls(step),
        "observations": truncate(str(step.observations or ""), max_observation_chars),
    }


def format_step_readable(step: ActionStep, max_observation_chars: int) -> str:
    model_output = truncate(str(step.model_output or ""), max_observation_chars)
    observations = truncate(
        clean_observation(str(step.observations or "")),
        max_observation_chars,
    )
    return (
        f"Step {step.step_number}\n"
        f"Tool calls: {format_tool_calls(step)}\n"
        f"Model output:\n{model_output or '(empty)'}\n"
        f"Observation summary:\n{observations or '(empty)'}"
    )


def format_step_summary(step: ActionStep) -> str:
    obs = str(step.observations or "").replace("\n", "\\n")
    if len(obs) > 200:
        obs = obs[:200] + "...[truncated]"
    return f"step {step.step_number}: tools={format_tool_calls(step)}; obs={obs}"


def truncate_trajectory(
    steps: list[ActionStep],
    *,
    head_n: int,
    tail_n: int,
    max_observation_chars: int,
) -> dict[str, Any]:
    head_n = max(head_n, 0)
    tail_n = max(tail_n, 0)
    if len(steps) <= head_n + tail_n:
        return {
            "head": [format_step_full(step, max_observation_chars) for step in steps],
            "middle_summary": [],
            "tail": [],
            "truncation_flag": "",
        }
    head = steps[:head_n]
    tail = steps[-tail_n:] if tail_n else []
    middle_end = len(steps) - tail_n if tail_n else len(steps)
    middle = steps[head_n:middle_end]
    return {
        "head": [format_step_full(step, max_observation_chars) for step in head],
        "middle_summary": [format_step_summary(step) for step in middle],
        "tail": [format_step_full(step, max_observation_chars) for step in tail],
        "truncation_flag": f"[middle truncated: {len(middle)} steps]",
    }


def build_online_trajectory(steps: list[ActionStep], max_observation_chars: int) -> dict[str, Any]:
    formatted_blocks: list[str] = []
    repeat_base: str | None = None
    previous_body: str | None = None
    previous_block_index: int | None = None
    repeat_count = 1
    for step in steps:
        body = format_step_readable(step, max_observation_chars)
        comparable = re.sub(r"^Step \d+\n", "", body)
        if comparable == previous_body and previous_block_index is not None:
            repeat_count += 1
            formatted_blocks[previous_block_index] = f"{repeat_base}\n(repeated x{repeat_count})"
            continue
        previous_body = comparable
        previous_block_index = len(formatted_blocks)
        repeat_base = body
        repeat_count = 1
        formatted_blocks.append(body)
    return {
        "formatted": "\n\n---\n\n".join(formatted_blocks),
        "raw": [format_step_full(step, max_observation_chars) for step in steps],
    }
