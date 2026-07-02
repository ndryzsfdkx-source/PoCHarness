"""Shared trajectory formatting and read-only trajectory/workspace tools.

Used by the B finalization gate and the C synthesis helper to render and
inspect Agent A's trajectory without depending on any retired subsystem.
"""
from .format import (
    TRAJECTORY_FORMATTED,
    TRAJECTORY_RAW,
    VALID_TRAJECTORY_MODES,
    build_online_trajectory,
    format_step_full,
    truncate,
    truncate_trajectory,
)
from .tools import (
    FindWorkspacePathsTool,
    GrepSourceTool,
    ReadAStepTool,
    ReadATrajectoryTool,
    ReadWorkspaceFileTool,
    TrajectoryToolState,
)


__all__ = [
    "TRAJECTORY_FORMATTED",
    "TRAJECTORY_RAW",
    "VALID_TRAJECTORY_MODES",
    "FindWorkspacePathsTool",
    "GrepSourceTool",
    "ReadAStepTool",
    "ReadATrajectoryTool",
    "ReadWorkspaceFileTool",
    "TrajectoryToolState",
    "build_online_trajectory",
    "format_step_full",
    "truncate",
    "truncate_trajectory",
]
