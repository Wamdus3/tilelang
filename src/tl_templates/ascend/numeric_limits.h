#pragma once

#include <cstdint>

namespace tl {
namespace limits {

// --- float (fp32) ---
// Bisheng supports constexpr bit casts, even when its host GCC standard library
// predates C++20's std::bit_cast (for example GCC 9 on Ubuntu 20.04).
inline constexpr float kFloatInf = __builtin_bit_cast(float, 0x7F800000u);
inline constexpr float kFloatNInf = __builtin_bit_cast(float, 0xFF800000u);
inline constexpr float kFloatNaN = __builtin_bit_cast(float, 0x7FC00000u);

// --- half (fp16) ---
inline constexpr half kHalfInf = __builtin_bit_cast(half, uint16_t{0x7C00u});
inline constexpr half kHalfNInf = __builtin_bit_cast(half, uint16_t{0xFC00u});
inline constexpr half kHalfNaN = __builtin_bit_cast(half, uint16_t{0x7E00u});

// --- bfloat16 ---
inline constexpr bfloat16_t kBf16Inf =
    __builtin_bit_cast(bfloat16_t, uint16_t{0x7F80u});
inline constexpr bfloat16_t kBf16NInf =
    __builtin_bit_cast(bfloat16_t, uint16_t{0xFF80u});
inline constexpr bfloat16_t kBf16NaN =
    __builtin_bit_cast(bfloat16_t, uint16_t{0x7FC0u});

} // namespace limits
} // namespace tl
