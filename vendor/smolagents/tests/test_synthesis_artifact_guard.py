from pathlib import Path

from smolagents.secb.harness.artifact_guard import run_harness_shape_probe


def test_harness_shape_probe_prefers_repro_testcase_path(tmp_path: Path):
    secb = tmp_path / "secb"
    output = tmp_path / "harness_shape.json"
    secb.write_text(
        """#!/bin/bash
build() {
    if [[ -f /testcase/repo_changes.diff ]]; then
        git apply --check /testcase/repo_changes.diff || true
    fi
}

repro() {
    /src/upx/build/debug/upx -d /testcase/poc
}
""",
        encoding="utf-8",
    )

    shape = run_harness_shape_probe(
        work_dir="/src/upx",
        secb_script=secb,
        output_path=output,
    )

    assert shape["expected_testcase_filename"] == "poc"
    assert "repo_changes.diff" in shape["build_block"]
    assert "/testcase/poc" in shape["repro_block"]
