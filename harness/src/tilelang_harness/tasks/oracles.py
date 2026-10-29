"""Scalar float64 oracles, separate from candidate and baseline implementations.

These finite-input P0 oracles return float32 out tensors and, for RMSNorm,
the required float32 reciprocal standard deviation for every row.
"""

import math

import numpy as np


def vecadd(inputs, epsilon):
    a, b = inputs["a"], inputs["b"]
    out = np.fromiter((float(x) + float(y) for x, y in zip(a, b)), dtype=np.float32, count=a.size)
    return {"out": out}


def rmsnorm(inputs, epsilon):
    x, weight = inputs["x"], inputs["weight"]
    out = np.empty_like(x)
    rstd = np.empty(x.shape[0], dtype=np.float32)
    for row_index, row in enumerate(x):
        scale = 1.0 / math.sqrt(math.fsum(float(v) ** 2 for v in row) / row.size + epsilon)
        rstd[row_index] = scale
        for col, value in enumerate(row):
            out[row_index, col] = float(value) * scale * float(weight[col])
    return {"out": out, "rstd": rstd}


def fft_c2c(inputs, epsilon):
    """Forward, unnormalized complex FFT; float64 components before NumPy FFT.

    The physical ABI splits complex64/128 into float32/64 component buffers.
    NumPy's independent FFT implementation evaluates in complex128.
    """
    values = inputs["real"].astype(np.float64) + 1j * inputs["imag"].astype(np.float64)
    transformed = np.fft.fft(values, axis=-1, norm="backward")
    dtype = inputs["real"].dtype
    return {
        "out_real": np.ascontiguousarray(transformed.real, dtype=dtype),
        "out_imag": np.ascontiguousarray(transformed.imag, dtype=dtype),
    }
