"""Untuned Ascend seed: one independent tile per block, including a final tail.

Derived from examples/ascend/example_vecadd.py. The single-tile scope has no
cross-iteration pipeline; this seed is for P1 device validation, not a tuned result.
"""


def build(N, tile=256):
    import tilelang.ascend.language as T

    if type(N) is not int or N <= 0 or type(tile) is not int or tile <= 0:
        raise ValueError("N and tile must be positive integers")

    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, tile)) as bx:
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
            # The final partial DMA must not leave any VF read lane uninitialized.
            T.clear(a)
            T.clear(b)
            T.copy(A[bx * tile : bx * tile + tile], a)
            T.copy(B[bx * tile : bx * tile + tile], b)
            with T.SimtVF(threads=128):
                for i in T.Parallel(tile):
                    c[i] = a[i] + b[i]
            T.copy(c, C[bx * tile : bx * tile + tile])

    return main
