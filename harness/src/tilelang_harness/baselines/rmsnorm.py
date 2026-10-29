"""Untuned per-row Ascend RMSNorm seed with both required outputs.

Adapted from examples/ascend/example_rmsnorm.py's reducer lifecycle. A block
owns one complete row, so there is no cross-row scratch reuse or tile loop.
"""


def build(M, N, epsilon=1e-6):
    import tilelang
    import tilelang.ascend.language as T

    if type(M) is not int or M <= 0 or type(N) is not int or N <= 0:
        raise ValueError("M and N must be positive integers")
    tile = tilelang.next_power_of_2(N)
    threads = min(128, tile)

    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        W: T.Tensor((N,), "float32"),
        Y: T.Tensor((M, N), "float32"),
        RSTD: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M) as row:
            x = T.alloc_shared((N,), "float32")
            w = T.alloc_shared((N,), "float32")
            y = T.alloc_shared((N,), "float32")
            rstd = T.alloc_shared((8,), "float32")
            T.copy(X[row, :N], x)
            T.copy(W, w)
            with T.SimtVF(threads=threads):
                values = T.alloc_fragment((tile,), "float32")
                for i in T.Parallel(tile):
                    values[i] = T.if_then_else(i < N, x[i], T.float32(0))
                accum = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(accum)
                for i in T.Parallel(tile):
                    T.reducer_update(accum[0], values[i] * values[i])
                scale = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(accum, scale)
                for i in T.Parallel(1):
                    scale[i] = T.rsqrt(scale[i] / N + epsilon)
                for i in T.Parallel(tile):
                    if i < N:
                        y[i] = values[i] * scale[0] * w[i]
                T.copy(scale, rstd[:1])
            T.copy(y, Y[row, :N])
            T.copy(rstd[:1], RSTD[row : row + 1])

    return main
