"""Host-only regressions for headers parsed even by a simple Ascend kernel.

Predicate tests use explicit SDK-shaped stubs to check wrapper dispatch. They do
not establish the numerical behavior of an NPU predicate instruction.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def compile_and_run(tmp_path, code):
    compiler = os.environ.get("TILELANG_HEADER_TEST_CXX") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a local C++ compiler is required")
    source = tmp_path / "check.cc"
    source.write_text(code)
    executable = tmp_path / "check"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(ROOT / "src"), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_special_value_bit_patterns_without_cxx20_library(tmp_path):
    compile_and_run(
        tmp_path,
        """
#include <cstdint>
using half = _Float16;
struct bfloat16_t { uint16_t bits; };
#include "tl_templates/ascend/numeric_limits.h"
using namespace tl::limits;
static_assert(__builtin_bit_cast(uint32_t, kFloatInf) == 0x7F800000u);
static_assert(__builtin_bit_cast(uint32_t, kFloatNInf) == 0xFF800000u);
static_assert(__builtin_bit_cast(uint32_t, kFloatNaN) == 0x7FC00000u);
static_assert(__builtin_bit_cast(uint16_t, kHalfInf) == 0x7C00u);
static_assert(__builtin_bit_cast(uint16_t, kHalfNInf) == 0xFC00u);
static_assert(__builtin_bit_cast(uint16_t, kHalfNaN) == 0x7E00u);
static_assert(kBf16Inf.bits == 0x7F80u);
static_assert(kBf16NInf.bits == 0xFF80u);
static_assert(kBf16NaN.bits == 0x7FC0u);
int main() { return 0; }
""",
    )


def test_predicate_wrapper_dispatch_with_sdk_shaped_stubs(tmp_path):
    header = (ROOT / "src/tl_templates/ascend/simd_inst.h").read_text()
    start = header.index("template <typename Part>\n__simd_callee__ inline vector_bool ppack")
    end = header.index("#define TL_SIMD_PINTLV_IMPL", start)
    compile_and_run(
        tmp_path,
        """
#define __simd_callee__
struct vector_bool { int value; };
enum Part { LOWER, HIGHER };
void asc_pack(vector_bool& dst, vector_bool src) { dst.value = src.value + 1; }
void asc_pack_v2(vector_bool& dst, vector_bool src) { dst.value = src.value + 2; }
void asc_unpack_lower(vector_bool& dst, vector_bool src) { dst.value = src.value + 3; }
void asc_unpack_upper(vector_bool& dst, vector_bool src) { dst.value = src.value + 4; }
namespace tl {
"""
        + header[start:end]
        + """
}
int main() {
  vector_bool src{10};
  return !(tl::ppack(src, LOWER).value == 11 && tl::ppack(src, HIGHER).value == 12 &&
           tl::punpack(src, LOWER).value == 13 && tl::punpack(src, HIGHER).value == 14);
}
""",
    )


def test_quarter_pack_store_uses_element_offset_with_sdk_shaped_stub(tmp_path):
    header = (ROOT / "src/tl_templates/ascend/simd_inst.h").read_text()
    start = header.index("template <typename T>\n__simd_callee__ inline void vsts_pack_quarter")
    end = header.index("template <typename T>", start + len("template <typename T>"))
    compile_and_run(
        tmp_path,
        """
#include <cstdint>
#include <type_traits>
#define __simd_callee__
#define __ubuf__
struct vector_bool {};
template<typename T> struct Vec { int payload; };
template<typename T> using vec_t = Vec<T>;
using vector_u32 = vec_t<uint32_t>;
using fp8_e4_t = uint8_t;
using fp8_e5_t = uint8_t;
using float4_e2m1x2_t = uint8_t;
using float4_e1m2x2_t = uint8_t;
void* written = nullptr;
template<typename T> void asc_storealign_pack_v2(T* dst, vec_t<T>, vector_bool) { written = dst; }
"""
        + header[start:end]
        + """
int main() {
  uint32_t words[16]{};
  uint8_t bytes[64]{};
  vsts_pack_quarter(vec_t<uint32_t>{}, words, 2, vector_bool{});
  if (written != words + 2) return 1;
  vsts_pack_quarter(vec_t<uint8_t>{}, bytes, 4, vector_bool{});
  return written != bytes + 4;
}
""",
    )
