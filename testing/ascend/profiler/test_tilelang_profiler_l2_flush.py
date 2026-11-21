"""NPU L2 cache flush."""

import pytest
import torch

import tilelang.testing
from tilelang.profiler import torch_bench


pytestmark = [pytest.mark.perf, *tilelang.testing.requires_ascend.marks()]


def test_npu_l2_cache_flush(npu_device):
    torch_bench._flush_cache(torch.empty(0, dtype=torch.int32, device=npu_device))


if __name__ == "__main__":
    tilelang.testing.main()
