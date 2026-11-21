"""NPU latency measurement with event and msprof backends."""

import math

import pytest
import tilelang.testing
from testing.ascend._ascend_test_utils import _measure


pytestmark = [pytest.mark.perf, *tilelang.testing.requires_ascend.marks()]


@pytest.mark.parametrize("backend", ["event", "msprof"])
def test_profiler_npu_latency(vector_kernel, npu_device, backend, record_property):
    kernel, inputs = vector_kernel
    latency = _measure(kernel, inputs, npu_device, backend)
    record_property("latency_ms", latency)
    assert math.isfinite(latency) and latency > 0


if __name__ == "__main__":
    tilelang.testing.main()
