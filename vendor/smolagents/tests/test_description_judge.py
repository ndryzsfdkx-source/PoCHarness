"""Tests for the description_mode crash-vs-description judge (assess_description_match).

Focus: the non-oracle judge must (a) extract sanitizer facts deterministically, (b) scrub
the target-identifying harness banner / instance_id before the LLM sees the output, and
(c) fail closed on missing signal or judge error -- no live model call required.
"""
from smolagents.secb.harness.description_judge import (
    _scrub_observed_output,
    assess_description_match,
    extract_observed_facts,
)


SAMPLE_OUTPUT = """Created testcase directory at /src/exiv2/testcase and linked to /testcase
/root/extracted/poc9-isoSpeed
REPRODUCING THE ISSUE FOR exiv2.cve-2018-19607...
==1234==ERROR: AddressSanitizer: SEGV on unknown address
    #0 0x55 in Exiv2::isoSpeed /src/exiv2/src/easyaccess.cpp:120:5
"""


def test_scrub_removes_banner_and_instance_id():
    scrubbed = _scrub_observed_output(SAMPLE_OUTPUT, "exiv2.cve-2018-19607")
    assert "REPRODUCING THE ISSUE FOR" not in scrubbed
    assert "exiv2.cve-2018-19607" not in scrubbed
    assert "cve-2018-19607" not in scrubbed  # bare CVE tail also redacted


def test_scrub_strips_banner_without_instance_id():
    scrubbed = _scrub_observed_output(SAMPLE_OUTPUT, "")
    assert "REPRODUCING THE ISSUE FOR" not in scrubbed


def test_scrub_preserves_sanitizer_facts():
    scrubbed = _scrub_observed_output(SAMPLE_OUTPUT, "exiv2.cve-2018-19607")
    facts = extract_observed_facts(scrubbed)
    assert facts["sanitizer_family"] == "ASAN"
    assert facts["top_frame"]["function"] == "Exiv2::isoSpeed"


def test_judge_fails_closed_on_no_sanitizer_signal():
    # No sanitizer family -> nothing to compare; returns matched=False without a model call.
    result = assess_description_match("some bug", "no crash here", model=None)
    assert result["matched"] is False
    assert result["observed_family"] is None


def test_judge_fails_closed_on_model_error():
    class _Boom:
        def generate(self, *a, **k):
            raise RuntimeError("transport down")

    result = assess_description_match(SAMPLE_OUTPUT, SAMPLE_OUTPUT, model=_Boom())
    assert result["matched"] is False
    assert result["judge_error"] == "transport down"


def test_judge_scrubs_before_model_sees_output():
    # Capture the prompt the model receives; assert the CVE banner/id never reach it.
    captured = {}

    class _Capture:
        def generate(self, messages, **k):
            captured["text"] = messages[0]["content"][0]["text"]

            class _R:
                content = '{"matched": true, "confidence": 0.9, "reasoning": "ok"}'

            return _R()

    assess_description_match(
        "an Exiv2 isoSpeed null deref", SAMPLE_OUTPUT, model=_Capture(),
        instance_id="exiv2.cve-2018-19607",
    )
    assert "REPRODUCING THE ISSUE FOR" not in captured["text"]
    assert "cve-2018-19607" not in captured["text"]
