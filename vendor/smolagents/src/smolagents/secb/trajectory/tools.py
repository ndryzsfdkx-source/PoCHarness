"""Read-only tools over Agent A's trajectory and the allow-listed workspace."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from smolagents.tools import Tool

from .format import build_online_trajectory, format_step_full, truncate, truncate_trajectory


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class TrajectoryToolState:
    steps: list[Any]
    valid_step_indices: set[int]
    default_head_steps: int
    default_tail_steps: int
    default_max_observation_chars: int


def _no_steps_message() -> str:
    return "No Agent A steps are available yet."


class ReadATrajectoryTool(Tool):
    name = "read_a_trajectory"
    description = "Return a head/middle/tail view of Agent A's trajectory."
    inputs = {
        "head": {"type": "integer", "description": "Number of head steps to include.", "nullable": True},
        "tail": {"type": "integer", "description": "Number of tail steps to include.", "nullable": True},
        "max_obs_chars": {"type": "integer", "description": "Per-step observation cap.", "nullable": True},
    }
    output_type = "string"

    def __init__(self, state: TrajectoryToolState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def forward(self, head: int | None = None, tail: int | None = None, max_obs_chars: int | None = None) -> str:
        if not self._state.steps:
            return _no_steps_message()
        cap = int(max_obs_chars or self._state.default_max_observation_chars)
        view = truncate_trajectory(
            self._state.steps,
            head_n=int(head if head is not None else self._state.default_head_steps),
            tail_n=int(tail if tail is not None else self._state.default_tail_steps),
            max_observation_chars=cap,
        )
        readable = build_online_trajectory(self._state.steps, cap)["formatted"]
        return f"--- structured ---\n{json.dumps(view, indent=2)}\n--- readable ---\n{readable}"


class ReadAStepTool(Tool):
    name = "read_a_step"
    description = "Return one Agent A step at full fidelity by step number."
    inputs = {
        "step_number": {"type": "integer", "description": "The step number to fetch."},
        "max_obs_chars": {"type": "integer", "description": "Observation cap.", "nullable": True},
    }
    output_type = "string"

    def __init__(self, state: TrajectoryToolState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def forward(self, step_number: int, max_obs_chars: int | None = None) -> str:
        if not self._state.steps:
            return _no_steps_message()
        step_number = _coerce_int(step_number)
        if step_number is None or step_number not in self._state.valid_step_indices:
            valid = sorted(self._state.valid_step_indices)
            return f"ERROR: step {step_number} not in valid steps {valid}"
        matched = [s for s in self._state.steps if getattr(s, "step_number", None) == step_number]
        if not matched:
            return f"ERROR: step {step_number} not found in trajectory"
        cap = int(max_obs_chars or self._state.default_max_observation_chars)
        return json.dumps(format_step_full(matched[0], cap), indent=2)


class ReadWorkspaceFileTool(Tool):
    name = "read_workspace_file"
    description = (
        "Read one allow-listed workspace file when a specific file, wrapper, script, "
        "binary entry point, or source path matters to the advice. Denies testcase, "
        "secb, and eval internals."
    )
    inputs = {
        "path": {"type": "string", "description": "Absolute file path to read."},
        "max_chars": {"type": "integer", "description": "Maximum characters to return.", "nullable": True},
    }
    output_type = "string"

    def __init__(self, inspector: Any, **kwargs):
        super().__init__(**kwargs)
        self._inspector = inspector

    def forward(self, path: str, max_chars: int | None = None) -> str:
        return self._inspector.read_file(path, max_chars=max_chars)


class FindWorkspacePathsTool(Tool):
    name = "find_workspace_paths"
    description = (
        "Discover allow-listed files or directories under a workspace root before "
        "choosing what to read or grep. Useful when you know the project or binary "
        "root but not the exact filename."
    )
    inputs = {
        "root_path": {"type": "string", "description": "Absolute allow-listed directory to search under."},
        "name_pattern": {
            "type": "string",
            "description": "Glob pattern for the basename, for example '*.c' or 'magick*'.",
            "nullable": True,
        },
        "file_type": {
            "type": "string",
            "description": "One of: any, file, dir.",
            "nullable": True,
        },
        "executable_only": {
            "type": "boolean",
            "description": "When true, only return executable files.",
            "nullable": True,
        },
        "max_depth": {
            "type": "integer",
            "description": "Maximum directory depth below root_path to search.",
            "nullable": True,
        },
        "max_matches": {
            "type": "integer",
            "description": "Maximum matches to return.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, inspector: Any, **kwargs):
        super().__init__(**kwargs)
        self._inspector = inspector

    def forward(
        self,
        root_path: str,
        name_pattern: str | None = None,
        file_type: str | None = None,
        executable_only: bool | None = False,
        max_depth: int | None = None,
        max_matches: int | None = None,
    ) -> str:
        return self._inspector.find_paths(
            root_path,
            name_pattern=name_pattern,
            file_type=file_type or "any",
            executable_only=bool(executable_only),
            max_depth=max_depth,
            max_matches=max_matches,
        )


class GrepSourceTool(Tool):
    name = "grep_source"
    description = (
        "Run read-only grep over an allow-listed source path. Useful for locating "
        "source symbols, functions, stack-frame names, error strings, or nearby "
        "source context before making source-based recommendations."
    )
    inputs = {
        "pattern": {"type": "string", "description": "Literal grep pattern."},
        "path": {"type": "string", "description": "Absolute source path to search."},
        "max_matches": {"type": "integer", "description": "Maximum matches to return.", "nullable": True},
    }
    output_type = "string"

    def __init__(self, inspector: Any, **kwargs):
        super().__init__(**kwargs)
        self._inspector = inspector

    def forward(self, pattern: str, path: str, max_matches: int | None = None) -> str:
        return self._inspector.grep_source(pattern, path, max_matches=max_matches)
