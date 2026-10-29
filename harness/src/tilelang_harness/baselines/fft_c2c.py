"""Untuned split-complex forward radix-2 FFT seed; NPU validation not claimed.

One Ascend vector block owns a row. Four N-element shared buffers keep source
and destination disjoint at each butterfly stage. Full-stage dependencies
require synchronization; this seed makes no performance claim.
"""


def build(B, N, dtype="float32"):
    import tilelang.ascend.language as T

    if type(B) is not int or B <= 0 or type(N) is not int or N <= 0:
        raise ValueError("B and N must be positive integers")
    if dtype not in ("float32", "float64") or N & (N - 1) or B * N > 4_194_304:
        raise ValueError("FFT requires float32/64, power-of-two N and B*N <= 4194304")
    if N > 4096:
        raise NotImplementedError("pinned device seed has no long-FFT baseline; measurement unavailable")
    bits = N.bit_length() - 1
    threads = min(128, N)

    @T.prim_func
    def main(
        real: T.Tensor((B, N), dtype),
        imag: T.Tensor((B, N), dtype),
        out_real: T.Tensor((B, N), dtype),
        out_imag: T.Tensor((B, N), dtype),
    ):
        with T.Kernel(B) as row:
            source_real = T.alloc_shared((N,), dtype)
            source_imag = T.alloc_shared((N,), dtype)
            dest_real = T.alloc_shared((N,), dtype)
            dest_imag = T.alloc_shared((N,), dtype)
            T.copy(real[row, :N], source_real)
            T.copy(imag[row, :N], source_imag)
            with T.SimtVF(threads=threads):
                reverse = T.alloc_fragment((N,), "int32")
                for i in T.Parallel(N):
                    reverse[i] = 0
                for bit in T.unroll(bits):
                    for i in T.Parallel(N):
                        reverse[i] = reverse[i] * 2 + (i // T.shift_left(1, bit)) % 2
                for i in T.Parallel(N):
                    dest_real[i] = source_real[reverse[i]]
                    dest_imag[i] = source_imag[reverse[i]]
                for i in T.Parallel(N):
                    source_real[i] = dest_real[i]
                    source_imag[i] = dest_imag[i]
                for stage in T.unroll(bits):
                    for i in T.Parallel(N):
                        half = T.shift_left(1, stage)
                        j = i % half
                        low = (i // (2 * half)) * (2 * half) + j
                        angle = T.const(-6.283185307179586, dtype) * T.Cast(dtype, j) / (2 * half)
                        cosine = T.cos(angle)
                        sine = T.sin(angle)
                        tr = source_real[low + half] * cosine - source_imag[low + half] * sine
                        ti = source_real[low + half] * sine + source_imag[low + half] * cosine
                        sign = T.if_then_else(i % (2 * half) < half, T.const(1, dtype), T.const(-1, dtype))
                        dest_real[i] = source_real[low] + sign * tr
                        dest_imag[i] = source_imag[low] + sign * ti
                    for i in T.Parallel(N):
                        source_real[i] = dest_real[i]
                        source_imag[i] = dest_imag[i]
            T.copy(source_real, out_real[row, :N])
            T.copy(source_imag, out_imag[row, :N])

    return main
