"""Agent-callable sanitizer parser tool.

Exposes the shared sanitizer_parser module as a smolagents Tool,
letting the agent parse raw sanitizer output into structured JSON
and optionally compare two reports.
"""
from __future__ import annotations

import json

from .parser import parse_sanitizer_output
from ...tools import Tool


class SanitizerParserTool(Tool):
    name = "sanitizer_parser"
    description = (
        "Parse raw sanitizer output (ASan, MSan, UBSan) into structured JSON with "
        "crash type, access info, and stack frames. Optionally compare two reports "
        "to check if they match. Use this to understand what `secb repro` output means "
        "and to compare your crash against the target."
    )
    inputs = {
        "raw_output": {
            "type": "string",
            "description": "Raw sanitizer stderr/stdout to parse.",
        },
        "compare_to": {
            "type": "string",
            "description": (
                "Optional: another raw sanitizer output to compare against. "
                "Returns a comparison showing type match, top frame match, "
                "and a human-readable explanation of the gap."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        raw_output: str,
        compare_to: str | None = None,
    ) -> str:
        report = parse_sanitizer_output(raw_output)
        result = report.to_dict()

        if compare_to:
            other = parse_sanitizer_output(compare_to)
            comparison = report.compare_to(other)

            # Add human-readable explanation
            parts = []
            if comparison["type_match"]:
                parts.append(f"Crash type matches: {comparison['expected_type']}")
            else:
                parts.append(
                    f"Crash type MISMATCH: expected {comparison['expected_type']}, "
                    f"got {comparison['actual_type']}"
                )
            if comparison["top_frame_match"]:
                parts.append(f"Top frame matches: {comparison['expected_top_function']}")
            else:
                parts.append(
                    f"Top frame MISMATCH: expected {comparison['expected_top_function']}, "
                    f"got {comparison['actual_top_function']}"
                )
            comparison["explanation"] = ". ".join(parts)
            result["comparison"] = comparison

        return json.dumps(result, indent=2)
