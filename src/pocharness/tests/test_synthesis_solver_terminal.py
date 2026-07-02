"""Unit tests for PoCSolverAgent.process_tool_calls terminal-wiring.

Covers three cases of the conditional-terminal logic without a full smolagents
run loop: ALLOW verdict → is_final_answer=True; BLOCK verdict → False; no
reviewer (synthesis off) → terminal. Neither test_finalization_review.py
(reviewer verdict logic) nor test_synthesis_protocol.py (Agent C emit/
continuation) cover this path.
"""
import sys
import types
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so we can import without a full LiteLLM/Docker environment
# ---------------------------------------------------------------------------

def _make_stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# Stub out heavy external imports that aren't needed for this spike
for _name in ["docker", "litellm", "openai"]:
    if _name not in sys.modules:
        sys.modules[_name] = _make_stub_module(_name)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction
from smolagents.memory import ActionStep
from smolagents.monitoring import Timing
from smolagents.secb.harness.agent import PoCSolverAgent


# ---------------------------------------------------------------------------
# Stub FinalizationReviewer
# ---------------------------------------------------------------------------
class StubReviewer:
    def __init__(self, allow: bool):
        self.last_verdict = "ALLOW_SUCCESS" if allow else ""
        self._allow = allow

    def invoke(self, artifact_status, stop_reason, memory):
        if self._allow:
            self.last_verdict = "ALLOW_SUCCESS"
            return True, artifact_status
        else:
            self.last_verdict = ""
            return False, "Inspect the target source and retry."


# ---------------------------------------------------------------------------
# Stub Model — returns a single final_submission tool call
# ---------------------------------------------------------------------------
def _make_final_submission_message():
    return ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatMessageToolCall(
                id="call_001",
                type="function",
                function=ChatMessageToolCallFunction(
                    name="final_submission",
                    arguments={
                        "artifact_status": "No crash observed.",
                        "stop_reason": "evidence_exhaustion",
                    },
                ),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Stub FinalSubmissionTool
# ---------------------------------------------------------------------------
class StubFinalSubmissionTool:
    name = "final_submission"
    description = "stub"

    def __init__(self, reviewer):
        self._reviewer = reviewer

    def __call__(self, artifact_status, stop_reason, memory=None):
        if self._reviewer is None:
            return artifact_status
        allow, payload = self._reviewer.invoke(artifact_status, stop_reason, memory)
        if allow:
            return artifact_status
        return payload


# ---------------------------------------------------------------------------
# Helper: build a minimal PoCSolverAgent and call process_tool_calls
# ---------------------------------------------------------------------------
def _run_process_tool_calls(reviewer):
    tool = StubFinalSubmissionTool(reviewer=reviewer)

    # We need a real PoCSolverAgent but bypass __init__ dependencies.
    # Use object.__new__ + manual attribute setup matching what ToolCallingAgent needs.
    agent = object.__new__(PoCSolverAgent)
    agent._reviewer = reviewer

    # Minimal attribute set used inside process_tool_calls:
    # - self.execute_tool_call(tool_name, arguments) -> str
    # - self.logger (for Panel log calls)
    class _NullLogger:
        def log(self, *a, **kw): pass

    agent.logger = _NullLogger()

    def _execute_tool_call(tool_name, arguments):
        if tool_name == "final_submission":
            return tool(
                artifact_status=arguments.get("artifact_status", ""),
                stop_reason=arguments.get("stop_reason", ""),
            )
        raise ValueError(f"unexpected tool: {tool_name}")

    agent.execute_tool_call = _execute_tool_call
    agent.state = {}  # used by base process_tool_calls for state lookup

    chat_message = _make_final_submission_message()
    memory_step = ActionStep(step_number=1, timing=Timing(start_time=0.0))

    from smolagents.memory import ToolCall
    from smolagents.agents import ToolOutput

    tool_calls_out = []
    tool_outputs_out = []
    for item in agent.process_tool_calls(chat_message, memory_step):
        if isinstance(item, ToolCall):
            tool_calls_out.append(item)
        elif isinstance(item, ToolOutput):
            tool_outputs_out.append(item)

    return tool_outputs_out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_allow_makes_final_submission_terminal():
    """ALLOW_SUCCESS verdict → is_final_answer=True."""
    reviewer = StubReviewer(allow=True)
    outputs = _run_process_tool_calls(reviewer)
    assert len(outputs) == 1, f"expected 1 ToolOutput, got {len(outputs)}"
    assert outputs[0].is_final_answer is True, (
        f"Expected is_final_answer=True on ALLOW; got {outputs[0].is_final_answer}"
    )


def test_block_makes_final_submission_non_terminal():
    """BLOCK verdict (last_verdict='') → is_final_answer=False."""
    reviewer = StubReviewer(allow=False)
    outputs = _run_process_tool_calls(reviewer)
    assert len(outputs) == 1, f"expected 1 ToolOutput, got {len(outputs)}"
    assert outputs[0].is_final_answer is False, (
        f"Expected is_final_answer=False on BLOCK; got {outputs[0].is_final_answer}"
    )


def test_no_reviewer_makes_final_submission_terminal():
    """No reviewer (synthesis off) → final_submission is terminal."""
    outputs = _run_process_tool_calls(reviewer=None)
    assert len(outputs) == 1
    assert outputs[0].is_final_answer is True
