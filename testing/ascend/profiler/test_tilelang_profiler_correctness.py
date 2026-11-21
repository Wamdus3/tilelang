"""Explicit NPU correctness checks."""

import pytest
import tilelang.testing


pytestmark = [pytest.mark.perf, *tilelang.testing.requires_ascend.marks()]


def test_profiler_npu_correct_reference(vector_kernel):
    kernel, inputs = vector_kernel
    kernel.get_profiler().assert_allclose(lambda a, b: a + b, input_tensors=inputs, atol=1e-5, rtol=1e-5, max_mismatched_ratio=0)


def test_profiler_npu_rejects_incorrect_reference(vector_kernel):
    kernel, inputs = vector_kernel
    with pytest.raises(AssertionError):
        kernel.get_profiler().assert_allclose(lambda a, b: a + b + 4, input_tensors=inputs, atol=1e-5, rtol=1e-5, max_mismatched_ratio=0)


if __name__ == "__main__":
    tilelang.testing.main()
