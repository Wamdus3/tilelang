"""Small Ascend test kernels adapted from tile-ai/tilelang 562cbc7c (Apache-2.0)."""


def check_vector(kernel):
    import torch

    a = torch.randn(4096)
    b = torch.randn_like(a)
    output = kernel(a.npu(), b.npu()).cpu()
    torch.testing.assert_close(output, a + b, rtol=1e-5, atol=1e-6)
    return {"passed": True, "max_abs_error": (output - (a + b)).abs().max().item()}


def vector_add(n=4096, tile=2048):
    import tilelang.ascend.language as T

    blocks = 1
    if tile not in (1024, 2048, 4096) or n % (blocks * tile):
        raise ValueError("Unsupported tile or nondivisible input")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(blocks) as bx:
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
            for step in T.serial(n // (blocks * tile)):
                begin = (step * blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                T.copy(B[begin : begin + tile], b)
                with T.SimtVF(threads=256):
                    for i in T.Parallel(tile):
                        c[i] = a[i] + b[i]
                T.copy(c, C[begin : begin + tile])

    return main


def cube_gemm(size=128):
    import tilelang.ascend.language as T

    blocks, tile = 1, 128
    if size <= 0 or size % tile:
        raise ValueError("GEMM size must be a positive multiple of 128")
    tiles = size // tile

    @T.prim_func
    def main(A: T.Tensor((size, size), "bfloat16"), B: T.Tensor((size, size), "bfloat16"), C: T.Tensor((size, size), "bfloat16")):
        with T.Kernel(blocks) as bx:
            a = T.alloc_l1((tile, tile), "bfloat16")
            b = T.alloc_l1((tile, tile), "bfloat16")
            c = T.alloc_l0c((tile, tile), "float32")
            for block in T.Persistent([tiles * tiles], blocks, bx):
                row, col = block // tiles, block % tiles
                for k in T.serial(tiles):
                    T.copy(A[row * tile : (row + 1) * tile, k * tile : (k + 1) * tile], a)
                    T.copy(B[col * tile : (col + 1) * tile, k * tile : (k + 1) * tile], b)
                    T.gemm(a, b, c, transpose_B=True, clear_accum=k == 0)
                T.copy(c, C[row * tile : (row + 1) * tile, col * tile : (col + 1) * tile])

    return main


def _measure(kernel, inputs, device, backend):
    assert all(tensor.device == device for tensor in inputs)
    return kernel.get_profiler().do_bench(
        input_tensors=inputs, device=device, backend=backend, n_warmup=5, n_repeat=10, early_stop_baseline=None
    )
