"""Vectorized CPU baselines for testing the judge, not NPU performance baselines."""

import numpy as np


def vecadd(inputs, epsilon):
    return {"out": np.add(inputs["a"], inputs["b"])}


def rmsnorm(inputs, epsilon):
    x, weight = inputs["x"], inputs["weight"]
    scale = np.reciprocal(np.sqrt(np.mean(np.square(x), axis=-1, dtype=np.float32) + epsilon))
    return {"out": (x * scale[:, None] * weight[None, :]).astype(np.float32), "rstd": scale.astype(np.float32)}


def fft_c2c(inputs, epsilon):
    """Independent radix-2 DIT reference, without np.fft or torch FFT calls.

    Work in float64 components to validate the judge independently of rounding
    choices in generated kernels. Row chunks bound temporary memory for long FFTs.
    """
    length = inputs["real"].shape[-1]
    if length < 1 or length > 1_048_576 or length & (length - 1):
        raise ValueError("FFT workload length must be a power of two <= 1048576")
    bits = length.bit_length() - 1
    indices = np.arange(length, dtype=np.int64)
    reverse = np.zeros(length, dtype=np.int64)
    for _ in range(bits):
        reverse = (reverse << 1) | (indices & 1)
        indices >>= 1
    out_real = np.empty_like(inputs["real"])
    out_imag = np.empty_like(inputs["imag"])
    rows_per_chunk = max(1, 262_144 // length)
    for start in range(0, out_real.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, out_real.shape[0])
        real = inputs["real"][start:end, reverse].astype(np.float64)
        imag = inputs["imag"][start:end, reverse].astype(np.float64)
        width = 2
        while width <= length:
            half = width // 2
            angle = -2.0 * np.pi * np.arange(half, dtype=np.float64) / width
            cosine, sine = np.cos(angle), np.sin(angle)
            real_tiles = real.reshape(real.shape[0], -1, width)
            imag_tiles = imag.reshape(imag.shape[0], -1, width)
            ar, ai = real_tiles[..., :half].copy(), imag_tiles[..., :half].copy()
            br, bi = real_tiles[..., half:].copy(), imag_tiles[..., half:].copy()
            tr, ti = br * cosine - bi * sine, br * sine + bi * cosine
            real_tiles[..., :half], real_tiles[..., half:] = ar + tr, ar - tr
            imag_tiles[..., :half], imag_tiles[..., half:] = ai + ti, ai - ti
            width *= 2
        out_real[start:end], out_imag[start:end] = real, imag
    return {"out_real": out_real, "out_imag": out_imag}
