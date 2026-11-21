"""Shared NPU fixtures for Ascend profiler tests."""

import pytest
import torch

import tilelang


@pytest.fixture(scope="module")
def npu_device():
    pytest.importorskip("torch_npu")
    return torch.device("npu", torch.npu.current_device())


@pytest.fixture(scope="module")
def vector_kernel(npu_device):
    from testing.ascend._ascend_test_utils import check_vector, vector_add

    kernel = tilelang.compile(vector_add(), target="ascend", execution_backend="tvm_ffi", out_idx=-1)
    check_vector(kernel)
    inputs = [torch.randn(4096, device=npu_device) for _ in range(2)]
    return kernel, inputs


@pytest.fixture(scope="module")
def cube_kernel(npu_device):
    from testing.ascend._ascend_test_utils import cube_gemm

    kernel = tilelang.compile(cube_gemm(), target="ascend", execution_backend="tvm_ffi", out_idx=-1)
    inputs = [torch.randn(128, 128, device=npu_device, dtype=torch.bfloat16) * 0.1 for _ in range(2)]
    kernel.get_profiler().assert_allclose(
        lambda a, b: (a.float() @ b.float().T).to(torch.bfloat16),
        input_tensors=inputs,
        atol=0.02,
        rtol=0.02,
        max_mismatched_ratio=0,
    )
    return kernel, inputs
