"""SSD host references; bfloat16 host inputs use exactly rounded float32 storage.

This keeps the offline harness NumPy-only. Device ABI remains real bfloat16.
"""

import numpy as np


def make_inputs(task, case, variant=0):
    d = case.sizes()
    rng = np.random.default_rng(case.seed + variant)
    b, nc, q, h, p, g, n = (d[k] for k in ("B", "NC", "Q", "H", "P", "G", "N"))

    def component(shape):
        value = (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(np.float32)
        if task.inputs[0].dtype == "float16":
            return value.astype(np.float16)
        bits = value.view(np.uint32)
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
        return rounded.view(np.float32)

    seq = np.zeros((b, nc, q), dtype=np.int32)
    if ".seq." in task.task_id:
        seq[:] = np.arange(q)[None, None, :] // max(1, q // 3)
        seq[:, 0, :] = -1  # Invalid final token excludes the entire chunk.
        if nc > 1:
            seq[:, 1, q // 2 :] = 7 + variant
    return {
        "x": component((b, nc * q, h, p)),
        "Bmat": component((b, nc * q, g, n)),
        "dt": (rng.random((b, h, nc, q), dtype=np.float32) * 0.1 + 0.01),
        "dA": -np.cumsum(rng.random((b, h, nc, q), dtype=np.float32), axis=-1),
        "seq_idx": seq.reshape(b, nc * q),
    }


def oracle(inputs, epsilon, *, has_seq_idx=True):
    x, bm, dt, da, seq = (inputs[k] for k in ("x", "Bmat", "dt", "dA", "seq_idx"))
    b, h, nc, q = dt.shape
    p, n, g = x.shape[-1], bm.shape[-1], bm.shape[-2]
    result = np.empty((b, nc, h, p, n), np.float32)
    for bi in range(b):
        for ci in range(nc):
            valid = True
            if has_seq_idx:
                ids = seq[bi, ci * q : (ci + 1) * q]
                valid = (ids == ids[-1]) & (ids[-1] >= 0)
            for hi in range(h):
                scale = np.exp(np.minimum(da[bi, hi, ci, -1].astype(np.float64) - da[bi, hi, ci].astype(np.float64), 0))
                scale *= dt[bi, hi, ci].astype(np.float64) * valid
                left = x[bi, ci * q : (ci + 1) * q, hi].astype(np.float64)
                right = bm[bi, ci * q : (ci + 1) * q, hi // (h // g)].astype(np.float64)
                result[bi, ci, hi] = np.einsum("qp,qn,q->pn", left, right, scale, optimize=True)
    return {"states": result}


def host_baseline(inputs, epsilon, *, has_seq_idx=True):
    x, bm, dt, da, seq = (inputs[k] for k in ("x", "Bmat", "dt", "dA", "seq_idx"))
    b, h, nc, q = dt.shape
    p, n, g = x.shape[-1], bm.shape[-1], bm.shape[-2]
    result = np.empty((b, nc, h, p, n), np.float32)
    for bi in range(b):
        for ci in range(nc):
            for hi in range(h):
                scale = dt[bi, hi, ci] * np.exp(np.minimum(da[bi, hi, ci, -1] - da[bi, hi, ci], 0))
                if has_seq_idx:
                    ids = seq[bi, ci * q : (ci + 1) * q]
                    scale *= (ids == ids[-1]) & (ids[-1] >= 0)
                left = x[bi, ci * q : (ci + 1) * q, hi].astype(np.float32)
                right = bm[bi, ci * q : (ci + 1) * q, hi // (h // g)].astype(np.float32)
                result[bi, ci, hi] = (left * scale[:, None]).T @ right
    return {"states": result}
