"""Tests for sanitizer output parser."""
import pytest
from smolagents.secb.sanitizer.parser import (
    grade,
    parse_sanitizer_output,
    SanitizerReport,
    _unqualified_function_name,
    semantic_match,
    _is_source_file,
)


ASAN_HBO_OUTPUT = """
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014 at pc 0x55555555abcd bp 0x7fffffffd870 sp 0x7fffffffd868
READ of size 4 at 0x602000000014 thread T0
    #0 0x55555555abcc in WriteUILImage coders/uil.c:248:21
    #1 0x55555555dcba in WriteImage MagickCore/constitute.c:1159:13
    #2 0x55555555ef01 in ConvertImageCommand MagickWand/convert.c:3254:12
    #3 0x7ffff7a23d8f in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x23d8f)

0x602000000014 is located 0 bytes to the right of 4-byte region [0x602000000010,0x602000000014)
allocated by thread T0 here:
    #0 0x7ffff7b44808 in malloc (/usr/lib/x86_64-linux-gnu/libasan.so.5+0x107808)
    #1 0x55555555aabb in AcquireQuantumMemory MagickCore/memory.c:551:10
""".strip()

ASAN_UAF_OUTPUT = """
=================================================================
==67890==ERROR: AddressSanitizer: heap-use-after-free on address 0x60d000001234 at pc 0x55555556abcd bp 0x7fffffffd870 sp 0x7fffffffd868
READ of size 8 at 0x60d000001234 thread T0
    #0 0x55555556abcc in mrb_ary_splat src/vm.c:1234:5
    #1 0x55555556bcda in mrb_vm_exec src/vm.c:2345:7
""".strip()

ASAN_SEGV_OUTPUT = """
=================================================================
==31165==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000022 (pc 0x00000053426f bp 0x7ffef5d7b170 sp 0x7ffef5d7a3e0 T0)
==31165==The signal is caused by a READ memory access.
==31165==Hint: address points to the zero page.
    #0 0x53426f in njs_dump_is_recursive /root/njs/src/njs_vmcode.c:1234:5
    #1 0x534abc in njs_vm_execute /root/njs/src/njs_vm.c:5678:7
""".strip()

PLAIN_TEXT_DESCRIPTION = """
In ImageMagick 7.0.7-1 Q16, the PersistPixelCache function in magick/cache.c
mishandles the pixel cache nexus, which allows remote attackers to cause a denial
of service (NULL pointer dereference in the function GetVirtualPixels in
MagickCore/cache.c) via a crafted file.
""".strip()

NO_SANITIZER_OUTPUT = """
Processing complete.
Exit code: 0
""".strip()


def test_parse_asan_heap_buffer_overflow():
    report = parse_sanitizer_output(ASAN_HBO_OUTPUT)
    assert report.sanitizer == "AddressSanitizer"
    assert report.crash_type == "heap-buffer-overflow"
    assert report.access_type == "READ"
    assert report.access_size == 4
    assert len(report.stack_frames) >= 2
    assert report.stack_frames[0].function == "WriteUILImage"
    assert report.stack_frames[0].file == "coders/uil.c"
    assert report.stack_frames[0].line == 248


def test_parse_asan_heap_use_after_free():
    report = parse_sanitizer_output(ASAN_UAF_OUTPUT)
    assert report.sanitizer == "AddressSanitizer"
    assert report.crash_type == "heap-use-after-free"
    assert report.stack_frames[0].function == "mrb_ary_splat"


def test_parse_no_sanitizer():
    report = parse_sanitizer_output(NO_SANITIZER_OUTPUT)
    assert report.sanitizer is None
    assert report.crash_type is None
    assert len(report.stack_frames) == 0


def test_report_to_dict():
    report = parse_sanitizer_output(ASAN_HBO_OUTPUT)
    d = report.to_dict()
    assert d["sanitizer"] == "AddressSanitizer"
    assert d["crash_type"] == "heap-buffer-overflow"
    assert isinstance(d["stack_frames"], list)
    assert d["stack_frames"][0]["function"] == "WriteUILImage"


def test_compare_reports_same_type():
    report = parse_sanitizer_output(ASAN_HBO_OUTPUT)
    diff = report.compare_to(report)
    assert diff["type_match"] is True
    assert diff["top_frame_match"] is True


def test_compare_reports_different_type():
    report_hbo = parse_sanitizer_output(ASAN_HBO_OUTPUT)
    report_uaf = parse_sanitizer_output(ASAN_UAF_OUTPUT)
    diff = report_hbo.compare_to(report_uaf)
    assert diff["type_match"] is False


def test_parse_asan_segv():
    report = parse_sanitizer_output(ASAN_SEGV_OUTPUT)
    assert report.sanitizer == "AddressSanitizer"
    assert report.crash_type == "SEGV"
    assert report.crash_address == "0x000000000022"
    assert len(report.stack_frames) >= 1
    assert report.stack_frames[0].function == "njs_dump_is_recursive"
    assert report.stack_frames[0].file == "/root/njs/src/njs_vmcode.c"
    assert report.stack_frames[0].line == 1234


def test_plain_text_returns_none():
    """Plain text descriptions must return crash_type=None — no guessing."""
    report = parse_sanitizer_output(PLAIN_TEXT_DESCRIPTION)
    assert report.sanitizer is None
    assert report.crash_type is None
    assert len(report.stack_frames) == 0


# ---------------------------------------------------------------------------
# C++ frame parsing (regression tests for the openexr.cve-2020-16588 bug where
# signatures with spaces and anonymous-namespace qualifiers were dropped,
# causing the post-harness controller to hallucinate a "wrong code path"
# diagnosis on a correctly-crashing PoC).
# ---------------------------------------------------------------------------

OPENEXR_TARGET = """
==5549==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000 (pc 0x402db3 bp 0x7fbf849cd800 sp 0x7ffe12fc8050 T0)
    #0 0x402db2 in generatePreview /home/dungnguyen/gueb-testing/openexr/OpenEXR/exrmakepreview/makePreview.cpp:132
    #1 0x402db2 in makePreview(char const*, char const*, int, float, bool) /home/dungnguyen/gueb-testing/openexr/OpenEXR/exrmakepreview/makePreview.cpp:162
    #2 0x402187 in main /home/dungnguyen/gueb-testing/openexr/OpenEXR/exrmakepreview/main.cpp:185
""".strip()

OPENEXR_ACTUAL_RUNTIME = """
AddressSanitizer:DEADLYSIGNAL
=================================================================
==2945==ERROR: AddressSanitizer: SEGV on unknown address (pc 0x6268d330ffc3 bp 0x7fffd997a0d0 sp 0x7fffd9979fe0 T0)
    #0 0x6268d330ffc3 in (anonymous namespace)::generatePreview(char const*, char const*, int, float, bool, Imf_2_3::InputFile&) /src/openexr/OpenEXR/exrmakepreview/makePreview.cpp:134
    #1 0x6268d3310c88 in makePreview(char const*, char const*, int, float, bool) /src/openexr/OpenEXR/exrmakepreview/makePreview.cpp:162
    #2 0x6268d3305986 in main /src/openexr/OpenEXR/exrmakepreview/main.cpp:189
""".strip()

OPENEXR_ACTUAL_BINARY_OFFSET = """
==1680==ERROR: AddressSanitizer: SEGV on unknown address (pc 0x582832b92fc3 bp 0x7ffd5e7c2a30 sp 0x7ffd5e7c2940 T0)
    #0 0x582832b92fc3 in makePreview(char const*, char const*, int, float, bool) (/src/openexr/bin/exrmakepreview+0x109fc3)
    #1 0x582832b917a4 in main (/src/openexr/bin/exrmakepreview+0x1087a4)
""".strip()

MRUBY_EXPECTED = """
=================================================================
==20001==ERROR: AddressSanitizer: SEGV on unknown address 0x52efc6aeca66 (pc 0x111111 bp 0x222222 sp 0x333333 T0)
==20001==The signal is caused by a WRITE memory access.
    #0 0x111111 in str_init_embed /src/mruby/src/string.c:73:5
    #1 0x222222 in str_new /src/mruby/src/string.c:101:3
    #2 0x333333 in unpack_hex /src/mruby/mrbgems/mruby-pack/src/pack.c:812:7
""".strip()

MRUBY_ACTUAL = """
=================================================================
==20002==ERROR: AddressSanitizer: SEGV on unknown address 0x52efc6aeca66 (pc 0xabcdef bp 0x222222 sp 0x333333 T0)
==20002==The signal is caused by a WRITE memory access.
    #0 0xabcdef in str_init_embed /src/mruby/src/string.c:73:5
    #1 0xbcdef0 in str_new /src/mruby/src/string.c:101:3
    #2 0xcdef01 in unpack_hex /src/mruby/mrbgems/mruby-pack/src/pack.c:812:7
""".strip()

MRUBY_NO_LINE_SOURCE_PATH = """
=================================================================
==20003==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x603000000010 at pc 0xabcdef bp 0x222222 sp 0x333333
READ of size 8 at 0x603000000010 thread T0
    #0 0xabcdef in mrb_funcall_with_block /src/mruby/src/vm.c
    #1 0xbcdef0 in mrb_funcall_argv /src/mruby/src/vm.c
""".strip()

PHP_EXPECTED = """
=================================================================
==30001==ERROR: AddressSanitizer: heap-use-after-free on address 0x603000000010 at pc 0x123456 bp 0x234567 sp 0x345678
READ of size 8 at 0x603000000010 thread T0
    #0 0x123456 in _php_stream_free /src/php-src/main/streams/streams.c:373:9
    #1 0x234567 in exif_read_from_file /src/php-src/ext/exif/exif.c:4411:5
""".strip()

PHP_ACTUAL = """
=================================================================
==30002==ERROR: AddressSanitizer: heap-use-after-free on address 0x603000000010 at pc 0x654321 bp 0x234567 sp 0x345678
READ of size 8 at 0x603000000010 thread T0
    #0 0x654321 in _php_stream_free /src/php-src/main/streams/streams.c:373:9
    #1 0x765432 in exif_read_from_file /src/php-src/ext/exif/exif.c:4411:5
""".strip()

PHP_ACTUAL_MISSING_LINE = """
=================================================================
==30003==ERROR: AddressSanitizer: heap-use-after-free on address 0x603000000010 at pc 0x654321 bp 0x234567 sp 0x345678
READ of size 8 at 0x603000000010 thread T0
    #0 0x654321 in _php_stream_free (/src/php-src/main/streams/streams.c+0x123)
    #1 0x765432 in exif_read_from_file (/src/php-src/ext/exif/exif.c+0x456)
""".strip()

EXIV2_EXPECTED = """
==11745==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x60200000ed80 at pc 0x000000620b4d bp 0x7ffccda7af80 sp 0x7ffccda7af70
READ of size 1 at 0x60200000ed80 thread T0
    #0 0x620b4c in Exiv2::Image::byteSwap4(Exiv2::DataBuf&, unsigned long, bool) /home/fuzz/exiv2/0.26/src/image.cpp:269
    #1 0x620b4c in Exiv2::Image::printIFDStructure(Exiv2::BasicIo&, std::ostream&, Exiv2::PrintStructureOption, unsigned int, bool, char, int) /home/fuzz/exiv2/0.26/src/image.cpp:444
""".strip()

EXIV2_ACTUAL_WRONG_BASENAME = """
==11746==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x60200000ed80 at pc 0x000000620b4d bp 0x7ffccda7af80 sp 0x7ffccda7af70
READ of size 1 at 0x60200000ed80 thread T0
    #0 0x620b4c in Exiv2::IptcData::printStructure /src/exiv2/src/iptc.cpp:362:9
    #1 0x620b4d in Exiv2::TiffImage::readMetadata /src/exiv2/src/tiffimage.cpp:191:5
SUMMARY: AddressSanitizer: heap-buffer-overflow /src/exiv2/src/image.cpp:269 Exiv2::Image::byteSwap4(Exiv2::DataBuf&, unsigned long, bool)
""".strip()

NJS_EXPECTED = """
=================================================================
==31165==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000022 (pc 0x00000053426f bp 0x7ffef5d7b170 sp 0x7ffef5d7a3e0 T0)
==31165==The signal is caused by a READ memory access.
==31165==Hint: address points to the zero page.
    #0 0x53426f in njs_dump_is_recursive /root/njs/src/njs_json.c:2100:5
    #1 0x53426f in njs_vm_value_dump /root/njs/src/njs_json.c:2113:13
    #2 0x4e0374 in njs_vm_retval_dump /root/njs/src/njs_vm.c:1004:12
""".strip()

NJS_ACTUAL_WRONG_FUNCTION = """
AddressSanitizer:DEADLYSIGNAL
=================================================================
==1643==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000018 (pc 0x5562ab25f4ac bp 0x7ffd7baf75c0 sp 0x7ffd7baf7500 T0)
==1643==The signal is caused by a READ memory access.
==1643==Hint: address points to the zero page.
    #0 0x5562ab25f4ac in njs_value_own_enumerate /src/njs/src/njs_value.c:241:17
    #1 0x5562ab2dc194 in njs_json_push_stringify_state /src/njs/src/njs_json.c:1004:23
    #2 0x5562ab2da6db in njs_vm_value_dump /src/njs/src/njs_json.c:2119:21
""".strip()

OPENEXR_ACTUAL_TOP3_NOT_FRAME0 = """
AddressSanitizer:DEADLYSIGNAL
=================================================================
==2946==ERROR: AddressSanitizer: SEGV on unknown address (pc 0x6268d330ffc3 bp 0x7fffd997a0d0 sp 0x7fffd9979fe0 T0)
    #0 0x6268d330fe00 in someWrapper /src/openexr/OpenEXR/exrmakepreview/makePreview.cpp:133
    #1 0x6268d330ffc3 in generatePreview /src/openexr/OpenEXR/exrmakepreview/makePreview.cpp:133
    #2 0x6268d3310c88 in makePreview(char const*, char const*, int, float, bool) /src/openexr/OpenEXR/exrmakepreview/makePreview.cpp:162
""".strip()

OPENEXR_ACTUAL_CALLER_ONLY = """
AddressSanitizer:DEADLYSIGNAL
=================================================================
==2947==ERROR: AddressSanitizer: SEGV on unknown address (pc 0x6268d3310c88 bp 0x7fffd997a0d0 sp 0x7fffd9979fe0 T0)
    #0 0x6268d3310c88 in makePreview(char const*, char const*, int, float, bool) /src/openexr/OpenEXR/exrmakepreview/makePreview.cpp:132
    #1 0x6268d3305986 in main /src/openexr/OpenEXR/exrmakepreview/main.cpp:189
""".strip()

UNRELATED_SEGV_SAME_TYPE = """
AddressSanitizer:DEADLYSIGNAL
=================================================================
==2948==ERROR: AddressSanitizer: SEGV on unknown address (pc 0x6268d3300000 bp 0x7fffd997a0d0 sp 0x7fffd9979fe0 T0)
    #0 0x6268d3300000 in jpeg_decode_scanline /src/openexr/jpeg.cpp:450
    #1 0x6268d3300001 in someUtility /src/openexr/util.cpp:10
    #2 0x6268d3305986 in main /src/openexr/OpenEXR/exrmakepreview/main.cpp:185
""".strip()


BASENAME_ONLY_SOURCE_FRAMES = """
=================================================================
==4100==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x503000000288 at pc 0x111111 bp 0x222222 sp 0x333333
READ of size 4 at 0x503000000288 thread T0
    #0 0x111111 in sycc420_to_rgb color.c
    #1 0x222222 in GetPixelAlpha xpm.c
""".strip()

OPENJPEG_3575_EXPECTED = """
=================================================================
==674==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x503000000288 at pc 0x641aed9a915d bp 0x7fff927eed20 sp 0x7fff927eed18
READ of size 4 at 0x503000000288 thread T0
    #0 0x641aed9a915c in sycc420_to_rgb /src/openjpeg/src/bin/common/color.c:379
    #1 0x641aed9a6e9e in color_sycc_to_rgb /src/openjpeg/src/bin/common/color.c:416
""".strip()

OPENJPEG_3575_ACTUAL_BASENAME_ONLY = """
=================================================================
==674==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x503000000288 at pc 0x641aed9a915d bp 0x7fff927eed20 sp 0x7fff927eed18
READ of size 4 at 0x503000000288 thread T0
    #0 0x641aed9a915c in sycc420_to_rgb color.c
    #1 0x641aed9a6e9e in color_sycc_to_rgb (/src/openjpeg/build/bin/opj_decompress+0x151e9e)
""".strip()

IMAGEMAGICK_0284_EXPECTED = """
=================================================================
==13628==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x509000001a8c at pc 0x5fc1e88762d0 bp 0x7fff3f7078c0 sp 0x7fff3f7078b8
READ of size 4 at 0x509000001a8c thread T0
    #0 0x5fc1e88762cf in GetPixelAlpha /src/imagemagick/MagickCore/pixel-accessor.h:59
    #1 0x5fc1e8870d15 in WritePICONImage /src/imagemagick/coders/xpm.c:807
""".strip()

IMAGEMAGICK_0284_ACTUAL_BASENAME_ONLY = """
=================================================================
==13628==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x509000001a8c at pc 0x5fc1e88762d0 bp 0x7fff3f7078c0 sp 0x7fff3f7078b8
READ of size 4 at 0x509000001a8c thread T0
    #0 0x5fc1e88762cf in GetPixelAlpha xpm.c
    #1 0x5fc1e8870d15 in WritePICONImage xpm.c
    #2 0x5fc1e89bebac in WriteImage (/src/imagemagick/utilities/magick+0x9edbac)
""".strip()


def test_parse_strips_trailing_source_basename_without_line():
    report = parse_sanitizer_output(BASENAME_ONLY_SOURCE_FRAMES)

    assert report.stack_frames[0].function == "sycc420_to_rgb"
    assert report.stack_frames[0].file == "color.c"
    assert report.stack_frames[0].line is None
    assert report.stack_frames[1].function == "GetPixelAlpha"
    assert report.stack_frames[1].file == "xpm.c"
    assert report.stack_frames[1].line is None


def test_parse_cpp_signature_with_spaces():
    """Function signatures with spaces between arg types must parse intact."""
    report = parse_sanitizer_output(OPENEXR_TARGET)
    assert report.crash_type == "SEGV"
    assert len(report.stack_frames) == 3
    assert report.stack_frames[0].function == "generatePreview"
    # The old regex dropped this frame because of the spaces in the signature.
    assert report.stack_frames[1].function == "makePreview(char const*, char const*, int, float, bool)"
    assert report.stack_frames[1].file.endswith("makePreview.cpp")
    assert report.stack_frames[1].line == 162
    assert report.stack_frames[2].function == "main"


def test_parse_anonymous_namespace_prefix():
    """The `(anonymous namespace)::` prefix must not break frame parsing."""
    report = parse_sanitizer_output(OPENEXR_ACTUAL_RUNTIME)
    assert len(report.stack_frames) == 3
    assert "generatePreview" in report.stack_frames[0].function
    assert report.stack_frames[0].function.startswith("(anonymous namespace)::")
    assert report.stack_frames[0].file.endswith("makePreview.cpp")


def test_parse_binary_plus_offset_location():
    """Frames with `(binary+0xOFFSET)` locations (no debug info) must parse."""
    report = parse_sanitizer_output(OPENEXR_ACTUAL_BINARY_OFFSET)
    assert len(report.stack_frames) == 2
    assert report.stack_frames[0].function == "makePreview(char const*, char const*, int, float, bool)"
    assert report.stack_frames[0].file == "/src/openexr/bin/exrmakepreview"
    assert report.stack_frames[0].line is None
    assert report.stack_frames[1].function == "main"


def test_openexr_target_vs_actual_matches():
    """Post-harness gating must treat the openexr runtime crash as matching
    its target report. The old parser caused this to come back as
    top_frame_match=False because it silently dropped C++ frames and left
    `main` as the top, which made the controller hallucinate a "wrong code
    path" diagnosis on a correctly-crashing PoC."""
    target = parse_sanitizer_output(OPENEXR_TARGET)
    actual = parse_sanitizer_output(OPENEXR_ACTUAL_RUNTIME)
    diff = target.compare_to(actual)
    assert diff["type_match"] is True
    assert diff["top_frame_match"] is True


def test_unqualified_function_name_variants():
    assert _unqualified_function_name(None) == ""
    assert _unqualified_function_name("") == ""
    assert _unqualified_function_name("generatePreview") == "generatePreview"
    assert _unqualified_function_name("foo::bar::baz") == "baz"
    assert (
        _unqualified_function_name("makePreview(char const*, int)")
        == "makePreview"
    )
    assert (
        _unqualified_function_name(
            "(anonymous namespace)::generatePreview(char const*, ...)"
        )
        == "generatePreview"
    )
    assert (
        _unqualified_function_name("Imf_2_3::InputFile::read(int)")
        == "read"
    )
    assert _unqualified_function_name("sycc420_to_rgb color.c") == "sycc420_to_rgb"
    assert _unqualified_function_name("GetPixelAlpha xpm.c") == "GetPixelAlpha"


@pytest.mark.parametrize(
    ("raw_name", "expected_name"),
    [
        ("sycc420_to_rgb.constprop.0", "sycc420_to_rgb"),
        ("sycc420_to_rgb.constprop.0 color.c", "sycc420_to_rgb"),
        ("decode_tile.isra.0", "decode_tile"),
        ("parse_header.part.0", "parse_header"),
        ("GetPixelAlpha.cold", "GetPixelAlpha"),
        ("GetPixelAlpha.cold.1", "GetPixelAlpha"),
        ("ns::Decoder::decode_tile.isra.0(int)", "decode_tile"),
        ("(anonymous namespace)::helper.part.0(char const*)", "helper"),
    ],
)
def test_unqualified_function_name_strips_compiler_generated_suffixes(
    raw_name,
    expected_name,
):
    assert _unqualified_function_name(raw_name) == expected_name


@pytest.mark.parametrize("grader_name", ["semantic", "strict"])
def test_openjpeg_basename_only_frame_matches_expected_function(grader_name):
    actual = parse_sanitizer_output(OPENJPEG_3575_ACTUAL_BASENAME_ONLY)
    assert actual.stack_frames[0].function == "sycc420_to_rgb"

    result = grade(
        parse_sanitizer_output(OPENJPEG_3575_EXPECTED),
        actual,
        grader=grader_name,
        N=10,
    )

    assert result["pass"] is True
    if grader_name == "strict":
        assert (
            result["gate_results_per_gate"]["L"]["reason"]
            == "location skipped: source basename without line info"
        )


@pytest.mark.parametrize("grader_name", ["semantic", "strict"])
def test_imagemagick_basename_only_frame_matches_expected_function(grader_name):
    actual = parse_sanitizer_output(IMAGEMAGICK_0284_ACTUAL_BASENAME_ONLY)
    assert actual.stack_frames[0].function == "GetPixelAlpha"

    result = grade(
        parse_sanitizer_output(IMAGEMAGICK_0284_EXPECTED),
        actual,
        grader=grader_name,
        N=10,
    )

    assert result["pass"] is True
    if grader_name == "strict":
        assert (
            result["gate_results_per_gate"]["L"]["reason"]
            == "location skipped: source basename without line info"
        )


@pytest.mark.parametrize("grader_name", ["caller", "semantic", "strict"])
def test_full_source_path_without_line_matches_expected_function(grader_name):
    report = parse_sanitizer_output(MRUBY_NO_LINE_SOURCE_PATH)

    assert report.stack_frames[0].function == "mrb_funcall_with_block"
    assert report.stack_frames[0].file == "vm.c"
    assert report.stack_frames[0].line is None

    result = grade(report, report, grader=grader_name, N=10)

    assert result["pass"] is True
    if grader_name == "strict":
        assert (
            result["gate_results_per_gate"]["L"]["reason"]
            == "location skipped: source basename without line info"
        )


def test_strict_grade_matches_compiler_generated_suffix_variant():
    actual = parse_sanitizer_output(
        OPENJPEG_3575_ACTUAL_BASENAME_ONLY.replace(
            "sycc420_to_rgb color.c",
            "sycc420_to_rgb.constprop.0 color.c",
        )
    )

    result = grade(
        parse_sanitizer_output(OPENJPEG_3575_EXPECTED),
        actual,
        grader="strict",
        N=10,
    )

    assert result["pass"] is True


def test_top_frame_match_fuzzy_on_inlined_functions():
    """If the runtime inlines the vulnerable function into its caller, the
    target's unqualified name still appears in the raw sanitizer output.
    The controller should not second-guess that."""
    target = parse_sanitizer_output(OPENEXR_TARGET)
    actual_with_inlining = OPENEXR_ACTUAL_RUNTIME  # contains "generatePreview"
    actual = parse_sanitizer_output(actual_with_inlining)
    assert target.compare_to(actual)["top_frame_match"] is True


def test_compare_to_returns_location_metadata_when_requested():
    target = parse_sanitizer_output(OPENEXR_TARGET)
    actual = parse_sanitizer_output(OPENEXR_ACTUAL_RUNTIME)

    diff = target.compare_to(actual, location_tolerance=10)

    assert diff["type_match"] is True
    assert diff["top_frame_match"] is True
    assert diff["basename_match"] is True
    assert diff["expected_file"].endswith("makePreview.cpp")
    assert diff["actual_file"].endswith("makePreview.cpp")
    assert diff["expected_line"] == 132
    assert diff["actual_line"] == 134
    assert diff["line_delta"] == 2


def test_compare_to_inlining_fallback_requires_matching_basename():
    expected = parse_sanitizer_output(EXIV2_EXPECTED)
    actual = parse_sanitizer_output(EXIV2_ACTUAL_WRONG_BASENAME)

    diff = expected.compare_to(actual)

    assert diff["type_match"] is True
    assert diff["top_frame_match"] is False


def test_semantic_match_passes_for_clean_e11b_cases():
    cases = [
        (MRUBY_EXPECTED, MRUBY_ACTUAL, 0),
        (OPENEXR_TARGET, OPENEXR_ACTUAL_RUNTIME, 2),
        (PHP_EXPECTED, PHP_ACTUAL, 0),
    ]

    for expected_raw, actual_raw, expected_delta in cases:
        result = semantic_match(
            parse_sanitizer_output(expected_raw),
            parse_sanitizer_output(actual_raw),
            N=10,
        )
        assert result["pass"] is True
        assert (
            result["gate_results_per_gate"]["L"]["details"]["line_delta"]
            == expected_delta
        )


def test_semantic_match_rejects_wrong_function_path():
    result = semantic_match(
        parse_sanitizer_output(NJS_EXPECTED),
        parse_sanitizer_output(NJS_ACTUAL_WRONG_FUNCTION),
        N=10,
    )

    assert result["pass"] is False
    assert result["reason"] == "top-frame function mismatch"


def test_semantic_match_rejects_exiv2_overmatch_after_fallback_fix():
    result = semantic_match(
        parse_sanitizer_output(EXIV2_EXPECTED),
        parse_sanitizer_output(EXIV2_ACTUAL_WRONG_BASENAME),
        N=10,
    )

    assert result["pass"] is False
    assert result["reason"] == "top-frame function mismatch"


def test_semantic_match_line_tolerance_boundary():
    target = parse_sanitizer_output(OPENEXR_TARGET)
    delta_10 = parse_sanitizer_output(
        OPENEXR_ACTUAL_RUNTIME.replace("makePreview.cpp:134", "makePreview.cpp:142", 1)
    )
    delta_11 = parse_sanitizer_output(
        OPENEXR_ACTUAL_RUNTIME.replace("makePreview.cpp:134", "makePreview.cpp:143", 1)
    )

    assert semantic_match(target, delta_10, N=10)["pass"] is True

    result = semantic_match(target, delta_11, N=10)
    assert result["pass"] is False
    assert result["reason"] == "line drift 11 exceeds tolerance 10"


def test_semantic_match_rejects_missing_line_info_with_matching_basename():
    result = semantic_match(
        parse_sanitizer_output(PHP_EXPECTED),
        parse_sanitizer_output(PHP_ACTUAL_MISSING_LINE),
        N=10,
    )

    assert result["pass"] is False
    assert result["reason"] == "missing line info for location check"


# ---------------------------------------------------------------------------
# No-debug-info: binary path in actual frame should skip basename check
# ---------------------------------------------------------------------------

IMAGEMAGICK_19302_EXPECTED = """
=================================================================
==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x60300000ed80 at pc 0x000000111111 bp 0x7ffd00000010 sp 0x7ffd00000000 T0
READ of size 8 at 0x60300000ed80 thread T0
    #0 0x111111 in ComplexImages /src/imagemagick/MagickCore/fourier.c:512:5
    #1 0x222222 in MagickCommandGenesis /src/imagemagick/MagickCore/magick.c:100
""".strip()

IMAGEMAGICK_19302_ACTUAL_NO_DEBUG = """
=================================================================
==2==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x60300000ed80 at pc 0x5ddfbf656874 bp 0x7ffd00000010 sp 0x7ffd00000000 T0
READ of size 8 at 0x60300000ed80 thread T0
    #0 0x5ddfbf656874 in ComplexImages (/src/imagemagick/utilities/magick+0x123456)
    #1 0x5ddfbf657000 in ConvertImageCommand (/src/imagemagick/utilities/magick+0x124000)
""".strip()

PHP_ACTUAL_NO_DEBUG = """
=================================================================
==30004==ERROR: AddressSanitizer: heap-use-after-free on address 0x603000000010 at pc 0x654321 bp 0x234567 sp 0x345678
READ of size 8 at 0x603000000010 thread T0
    #0 0x5ff6511ce1f1 in _php_stream_free (/src/php-src/sapi/cli/php+0xabcdef)
    #1 0x5ff6511cf000 in php_stream_free_enclosed (/src/php-src/sapi/cli/php+0xabc000)
""".strip()

IMAGEMAGICK_14400_UNKNOWN_SIGNAL = """
=================================================================
==14212==ERROR: AddressSanitizer: UNKNOWN SIGNAL on unknown address 0x7ffed6369120 (pc 0x5e2f3563b42d bp 0x7ffed6369d30 sp 0x7ffed6369bc0 T0)
    #0 0x5e2f3563b42d in GetVirtualPixels (/src/imagemagick/utilities/magick+0x86c42d)
    #1 0x5e2f3533b354 in WriteAAIImage aai.c
SUMMARY: AddressSanitizer: UNKNOWN SIGNAL (/src/imagemagick/utilities/magick+0x86c42d) in GetVirtualPixels
==14212==ABORTING
""".strip()


def test_is_source_file():
    assert _is_source_file("/src/foo/bar.c") is True
    assert _is_source_file("/src/foo/bar.cpp") is True
    assert _is_source_file("/src/foo/bar.h") is True
    assert _is_source_file("/src/imagemagick/utilities/magick") is False
    assert _is_source_file("/src/php-src/sapi/cli/php") is False
    assert _is_source_file(None) is False


def test_strict_grade_passes_when_actual_frame_is_binary_no_debug_info():
    """Function matches but Docker image has no DWARF info — should pass strict."""
    result = grade(
        parse_sanitizer_output(IMAGEMAGICK_19302_EXPECTED),
        parse_sanitizer_output(IMAGEMAGICK_19302_ACTUAL_NO_DEBUG),
        grader="strict",
        N=10,
    )
    assert result["pass"] is True
    assert result["reason"] == "strict match"
    assert "no debug info" in result["gate_results_per_gate"]["L"]["reason"]


def test_strict_grade_php_binary_path_passes():
    result = grade(
        parse_sanitizer_output(PHP_EXPECTED),
        parse_sanitizer_output(PHP_ACTUAL_NO_DEBUG),
        grader="strict",
        N=10,
    )
    assert result["pass"] is True
    assert "no debug info" in result["gate_results_per_gate"]["L"]["reason"]


def test_semantic_match_rejects_genuine_basename_mismatch_with_source_info():
    """Both frames have source files but different basenames — should still fail."""
    result = semantic_match(
        parse_sanitizer_output(EXIV2_EXPECTED),
        parse_sanitizer_output(EXIV2_ACTUAL_WRONG_BASENAME),
        N=10,
    )
    assert result["pass"] is False
    assert result["reason"] == "top-frame function mismatch"


def test_parse_unknown_signal_crash_type():
    report = parse_sanitizer_output(IMAGEMAGICK_14400_UNKNOWN_SIGNAL)
    assert report.sanitizer == "AddressSanitizer"
    assert report.crash_type == "UNKNOWN SIGNAL"
    assert report.stack_frames[0].function == "GetVirtualPixels"


def test_strict_grade_rejects_top3_match_that_is_not_frame0():
    result = grade(
        parse_sanitizer_output(OPENEXR_TARGET),
        parse_sanitizer_output(OPENEXR_ACTUAL_TOP3_NOT_FRAME0),
        grader="strict",
        N=10,
    )
    assert result["pass"] is False
    assert result["reason"].startswith("frame 0 mismatch")


def test_caller_grade_passes_beta_path_without_inline_metadata():
    result = grade(
        parse_sanitizer_output(OPENEXR_TARGET),
        parse_sanitizer_output(OPENEXR_ACTUAL_CALLER_ONLY),
        grader="caller",
        N=10,
    )
    assert result["pass"] is True
    assert result["gate_results_per_gate"]["F-inline"]["pass"] is False
    assert result["gate_results_per_gate"]["F-caller"]["pass"] is True
    assert result["gate_results_per_gate"]["L-dual"]["details"]["passed_branch"] == "beta"


def test_caller_grade_rejects_same_type_unrelated_crash():
    result = grade(
        parse_sanitizer_output(OPENEXR_TARGET),
        parse_sanitizer_output(UNRELATED_SEGV_SAME_TYPE),
        grader="caller",
        N=10,
    )
    assert result["pass"] is False
    assert result["reason"] == "no caller-tolerant function match"


def test_all_graders_return_no_oracle_when_expected_report_is_missing():
    actual = parse_sanitizer_output(MRUBY_ACTUAL)
    for grader in ("loose", "caller", "semantic", "strict"):
        result = grade(None, actual, grader=grader, N=10)
        assert result["pass"] is False
        assert result["reason"] == "no_oracle"


@pytest.mark.parametrize(
    ("actual_raw", "expected_passes"),
    [
        (MRUBY_ACTUAL, {"loose": True, "caller": True, "semantic": True, "strict": True}),
        (
            OPENEXR_ACTUAL_TOP3_NOT_FRAME0,
            {"loose": True, "caller": True, "semantic": True, "strict": False},
        ),
        (
            OPENEXR_ACTUAL_CALLER_ONLY,
            {"loose": True, "caller": True, "semantic": False, "strict": False},
        ),
        (
            UNRELATED_SEGV_SAME_TYPE,
            {"loose": True, "caller": False, "semantic": False, "strict": False},
        ),
    ],
)
def test_grader_ordering_transitions(actual_raw, expected_passes):
    expected = parse_sanitizer_output(OPENEXR_TARGET if "openexr" in actual_raw.lower() else MRUBY_EXPECTED)
    actual = parse_sanitizer_output(actual_raw)
    results = {
        grader: grade(expected, actual, grader=grader, N=10)["pass"]
        for grader in ("loose", "caller", "semantic", "strict")
    }
    assert results == expected_passes
    assert results["strict"] <= results["semantic"] <= results["caller"] <= results["loose"]


def test_grader_ordering_transitions_no_oracle():
    actual = parse_sanitizer_output(MRUBY_ACTUAL)
    results = {
        grader: grade(None, actual, grader=grader, N=10)
        for grader in ("loose", "caller", "semantic", "strict")
    }
    for result in results.values():
        assert result["pass"] is False
        assert result["reason"] == "no_oracle"
