"""Tests for sanitizer parser agent tool."""
import json
from smolagents.secb.sanitizer.tool import SanitizerParserTool


ASAN_OUTPUT = """
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014 at pc 0x55555555abcd bp 0x7fffffffd870 sp 0x7fffffffd868
READ of size 4 at 0x602000000014 thread T0
    #0 0x55555555abcc in WriteUILImage coders/uil.c:248:21
    #1 0x55555555dcba in WriteImage MagickCore/constitute.c:1159:13
""".strip()

ASAN_UAF_OUTPUT = """
==67890==ERROR: AddressSanitizer: heap-use-after-free on address 0x60d000001234 at pc 0x55555556abcd bp 0x7fffffffd870 sp 0x7fffffffd868
READ of size 8 at 0x60d000001234 thread T0
    #0 0x55555556abcc in mrb_ary_splat src/vm.c:1234:5
""".strip()


def test_tool_attributes():
    tool = SanitizerParserTool()
    assert tool.name == "sanitizer_parser"
    assert "raw_output" in tool.inputs
    assert tool.output_type == "string"


def test_tool_parses_asan():
    tool = SanitizerParserTool()
    result = tool.forward(raw_output=ASAN_OUTPUT)
    parsed = json.loads(result)
    assert parsed["sanitizer"] == "AddressSanitizer"
    assert parsed["crash_type"] == "heap-buffer-overflow"
    assert parsed["stack_frames"][0]["function"] == "WriteUILImage"


def test_tool_compare_mode_same():
    tool = SanitizerParserTool()
    result = tool.forward(
        raw_output=ASAN_OUTPUT,
        compare_to=ASAN_OUTPUT,
    )
    parsed = json.loads(result)
    assert "comparison" in parsed
    assert parsed["comparison"]["type_match"] is True
    assert parsed["comparison"]["top_frame_match"] is True


def test_tool_compare_mode_different():
    tool = SanitizerParserTool()
    result = tool.forward(
        raw_output=ASAN_OUTPUT,
        compare_to=ASAN_UAF_OUTPUT,
    )
    parsed = json.loads(result)
    assert parsed["comparison"]["type_match"] is False
    assert "explanation" in parsed["comparison"]


def test_tool_no_sanitizer():
    tool = SanitizerParserTool()
    result = tool.forward(raw_output="Exit code: 0\nNo errors.")
    parsed = json.loads(result)
    assert parsed["sanitizer"] is None
