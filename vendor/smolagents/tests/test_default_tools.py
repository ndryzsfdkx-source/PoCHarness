# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
from unittest.mock import patch

import pytest

from smolagents.agent_types import _AGENT_TYPE_MAPPING
from smolagents.default_tools import (
    TOOL_MAPPING,
    DuckDuckGoSearchTool,
    GDBTool,
    PythonInterpreterTool,
)

from ._tool_test_helpers import ToolTesterMixin
from .utils.markers import require_run_all


class DefaultToolTests(unittest.TestCase):
    def test_oracle_seed_is_not_in_default_tool_mapping(self):
        assert "oracle_seed" not in TOOL_MAPPING

    @require_run_all
    def test_ddgs_with_kwargs(self):
        result = DuckDuckGoSearchTool(timeout=20)("DeepSeek parent company")
        assert isinstance(result, str)


class TestPythonInterpreterTool(ToolTesterMixin):
    def setup_method(self):
        self.tool = PythonInterpreterTool(authorized_imports=["numpy"])
        self.tool.setup()

    def test_exact_match_arg(self):
        result = self.tool("(2 / 2) * 4")
        assert result == "Stdout:\n\nOutput: 4.0"

    def test_exact_match_kwarg(self):
        result = self.tool(code="(2 / 2) * 4")
        assert result == "Stdout:\n\nOutput: 4.0"

    def test_agent_type_output(self):
        inputs = ["2 * 2"]
        output = self.tool(*inputs, sanitize_inputs_outputs=True)
        output_type = _AGENT_TYPE_MAPPING[self.tool.output_type]
        assert isinstance(output, output_type)

    def test_agent_types_inputs(self):
        inputs = ["2 * 2"]
        _inputs = []

        for _input, expected_input in zip(inputs, self.tool.inputs.values()):
            input_type = expected_input["type"]
            if isinstance(input_type, list):
                _inputs.append([_AGENT_TYPE_MAPPING[_input_type](_input) for _input_type in input_type])
            else:
                _inputs.append(_AGENT_TYPE_MAPPING[input_type](_input))

        # Should not raise an error
        output = self.tool(*inputs, sanitize_inputs_outputs=True)
        output_type = _AGENT_TYPE_MAPPING[self.tool.output_type]
        assert isinstance(output, output_type)

    def test_imports_work(self):
        result = self.tool("import numpy as np")
        assert "import from numpy is not allowed" not in result.lower()

    def test_unauthorized_imports_fail(self):
        with pytest.raises(Exception) as e:
            self.tool("import sympy as sp")
        assert "sympy" in str(e).lower()


class DummyCompletedProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class TestGDBTool:
    @patch("smolagents.default_tools._ensure_system_packages", return_value=(True, "already installed"))
    @patch("smolagents.default_tools.subprocess.run")
    def test_gdb_sets_breakpoints_before_running_target(self, mock_run, _mock_packages):
        mock_run.side_effect = [
            DummyCompletedProcess(stdout="Breakpoint 1 at 0x1234\n"),
            DummyCompletedProcess(stdout="Program received signal SIGSEGV\n#0 0x1234 in target\n"),
        ]

        result = GDBTool().forward(
            command="/bin/target /tmp/input",
            work_dir="/tmp",
            gdb_commands="break target_func; run; bt; info locals",
        )

        assert mock_run.call_count == 2
        setup_args = mock_run.call_args_list[0].args[0]
        run_args = mock_run.call_args_list[1].args[0]
        setup_joined = " ".join(setup_args)
        run_joined = " ".join(run_args)
        assert "-ex break target_func" in setup_joined
        assert "-ex run" not in setup_joined
        assert run_joined.index("break target_func") < run_joined.index("run")
        assert run_joined.count("-ex run") == 1
        assert run_joined.index("run") < run_joined.index("bt 30")
        assert "Program received signal SIGSEGV" in result

    @patch("smolagents.default_tools._ensure_system_packages", return_value=(True, "already installed"))
    @patch("smolagents.default_tools.subprocess.run")
    def test_gdb_returns_early_when_breakpoint_setup_fails(self, mock_run, _mock_packages):
        mock_run.return_value = DummyCompletedProcess(stdout='Function "missing_func" not defined.\n')

        result = GDBTool().forward(
            command="/bin/target /tmp/input",
            work_dir="/tmp",
            gdb_commands="break missing_func; run; bt",
        )

        assert mock_run.call_count == 1
        assert "GDB breakpoint setup failed before target execution" in result
        assert "Function \"missing_func\" not defined" in result

    @patch("smolagents.default_tools._ensure_system_packages", return_value=(True, "already installed"))
    @patch("smolagents.default_tools.subprocess.run")
    def test_gdb_uses_direct_argv_for_simple_commands(self, mock_run, _mock_packages):
        mock_run.return_value = DummyCompletedProcess(stdout="Program exited normally.\n")

        GDBTool().forward(command="/bin/target --flag value", work_dir="/tmp")

        gdb_args = mock_run.call_args.args[0]
        assert gdb_args[-3:] == ["/bin/target", "--flag", "value"]
        assert "bash" not in gdb_args

    @patch("smolagents.default_tools._ensure_system_packages", return_value=(True, "already installed"))
    @patch("smolagents.default_tools.subprocess.run")
    def test_gdb_keeps_shell_for_shell_commands(self, mock_run, _mock_packages):
        mock_run.return_value = DummyCompletedProcess(stdout="Program exited normally.\n")

        GDBTool().forward(command="/bin/target < /tmp/input", work_dir="/tmp")

        gdb_args = mock_run.call_args.args[0]
        assert gdb_args[-3:] == ["bash", "-lc", "exec /bin/target < /tmp/input"]

    @patch("smolagents.default_tools._ensure_system_packages", return_value=(True, "already installed"))
    @patch("smolagents.default_tools.subprocess.run")
    def test_gdb_preserves_simple_env_assignments_without_shell(self, mock_run, _mock_packages):
        mock_run.return_value = DummyCompletedProcess(stdout="Program exited normally.\n")

        GDBTool().forward(command="USE_ZEND_ALLOC=0 /bin/target --flag", work_dir="/tmp")

        gdb_args = mock_run.call_args.args[0]
        assert gdb_args[-3:] == ["USE_ZEND_ALLOC=0", "/bin/target", "--flag"]
        assert gdb_args[-4].endswith("/env")
        assert "bash" not in gdb_args


