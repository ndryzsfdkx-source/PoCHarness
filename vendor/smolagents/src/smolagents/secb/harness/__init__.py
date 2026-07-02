"""On-the-fly synthesis helper (Agent C) and A-side synthesis tools for SEC-bench PoC runs."""
from .artifact_guard import ArtifactGuardTool, run_harness_shape_probe
from .config import (
    SynthesisAgentBackendConfig,
    SynthesisConfig,
    SynthesisFinalizationGuardConfig,
    SynthesisWorkspaceConfig,
)
from .finalization_guard import SynthesisFinalizationGuard, create_synthesis_finalization_guard
from .synthesis import SynthesisOrchestrator, create_synthesis_orchestrator
from .tool import FinalSubmissionTool, RequestSynthesisHelperTool, RunSecbReproOnCurrentTestcaseTool


__all__ = [
    "ArtifactGuardTool",
    "FinalSubmissionTool",
    "RequestSynthesisHelperTool",
    "RunSecbReproOnCurrentTestcaseTool",
    "SynthesisAgentBackendConfig",
    "SynthesisConfig",
    "SynthesisFinalizationGuard",
    "SynthesisFinalizationGuardConfig",
    "SynthesisOrchestrator",
    "SynthesisWorkspaceConfig",
    "create_synthesis_finalization_guard",
    "create_synthesis_orchestrator",
    "run_harness_shape_probe",
]
