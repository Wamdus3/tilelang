"""AIC/AIV pipeline metrics collected by msprof."""

from dataclasses import asdict
import math

import pytest
import tilelang.testing
from testing.ascend._ascend_test_utils import _measure


pytestmark = [pytest.mark.perf, *tilelang.testing.requires_ascend.marks()]


@pytest.mark.parametrize(
    "kernel_fixture, required",
    [
        pytest.param("vector_kernel", ("aiv_total_cycles", "aiv_vec"), id="aiv"),
        pytest.param("cube_kernel", ("aic_total_cycles", "aic_mad"), id="aic"),
    ],
)
def test_msprof_pipeline_metrics(request, npu_device, kernel_fixture, required, record_property):
    kernel, inputs = request.getfixturevalue(kernel_fixture)
    metrics = asdict(_measure(kernel, inputs, npu_device, "msprof_detail"))
    for name, value in metrics.items():
        record_property(name, value)
    assert math.isfinite(metrics["dur_ns"]) and metrics["dur_ns"] > 0
    for name, value in metrics.items():
        if name.startswith(("aic_", "aiv_")) and not name.endswith("total_cycles"):
            assert math.isfinite(value) and 0 <= value <= 1, (name, value)
    assert all(math.isfinite(metrics[name]) and metrics[name] > 0 for name in required)


if __name__ == "__main__":
    tilelang.testing.main()
