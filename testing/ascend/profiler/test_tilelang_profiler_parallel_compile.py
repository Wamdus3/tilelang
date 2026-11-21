"""Candidate compilation worker limits, concurrency, and NPU correctness."""

import concurrent.futures
import threading

import pytest
import torch

import tilelang.testing
from tilelang.autotuner import AutoTuner
from tilelang.autotuner import tuner as tuner_module
from tilelang.env import env
from testing.ascend._ascend_test_utils import vector_add


pytestmark = pytest.mark.perf


@pytest.mark.parametrize("requested, cap, expected", [(4, 2, 2), (20, -1, 8), (-1, -1, 4)])
def test_compile_worker_limits(monkeypatch, requested, cap, expected):
    monkeypatch.setattr(tuner_module, "get_available_cpu_count", lambda: 8)
    monkeypatch.setattr(env, "TILELANG_AUTO_TUNING_CPU_COUNTS", str(requested))
    monkeypatch.setattr(env, "TILELANG_AUTO_TUNING_MAX_CPU_COUNT", str(cap))
    monkeypatch.setattr(env, "TILELANG_AUTO_TUNING_CPU_UTILITIES", "0.5")
    tuner = AutoTuner(lambda tile: tile, [{"tile": 1024}])
    assert tuner._resolve_num_compile_workers() == expected


@pytest.mark.parametrize("workers", [1, 2])
def test_compile_executor_overlap_and_failure_isolation(monkeypatch, workers):
    monkeypatch.setattr(env, "TILELANG_AUTO_TUNING_CPU_COUNTS", str(workers))
    monkeypatch.setattr(env, "TILELANG_AUTO_TUNING_MAX_CPU_COUNT", str(workers))
    monkeypatch.setattr(tuner_module, "get_available_cpu_count", lambda: 8)
    configs = [{"tile": i} for i in range(4)]
    tuner = AutoTuner(lambda tile: tile, configs)
    barrier = threading.Barrier(workers)
    lock = threading.Lock()
    active = peak = 0

    def compile_one(tile):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait(timeout=10)
            if tile == 3:
                raise ValueError("intentional candidate failure")
            return tile
        finally:
            with lock:
                active -= 1

    pool, futures, _, _ = tuner._prepare_compile_execution(configs, False, 1, compile_one, None)
    try:
        results = [row for future in futures for row in future.result(timeout=30)]
    finally:
        pool.shutdown(wait=True)
    assert peak == workers
    assert len(results) == 4
    failures = [row for row in results if row[3] is not None]
    assert len(failures) == 1
    assert failures[0][1] == {"tile": 3}
    assert isinstance(failures[0][3], ValueError)
    assert sorted(kernel for _, _, kernel, error in results if error is None) == [0, 1, 2]


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_candidate_compilation(monkeypatch, npu_device, workers, record_property):
    monkeypatch.setattr(env, "TILELANG_AUTO_TUNING_CPU_COUNTS", str(workers))
    monkeypatch.setattr(env, "TILELANG_AUTO_TUNING_MAX_CPU_COUNT", str(workers))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "1")
    configs = [{"tile": tile} for tile in (1024, 2048, 4096)]
    tuner = AutoTuner(vector_add, configs)
    tuner.set_compile_args(target="ascend", execution_backend="tvm_ffi", out_idx=-1)
    resolved = tuner._resolve_num_compile_workers()
    assert 1 <= resolved <= workers
    lock = threading.Lock()
    active = peak = 0

    def compile_one(**config):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return tuner._default_compile(**config)
        finally:
            with lock:
                active -= 1

    pool, futures, _, _ = tuner._prepare_compile_execution(configs, False, 1, compile_one, tuner._default_elaborate)
    try:
        results = [row for future in concurrent.futures.as_completed(futures) for row in future.result()]
    finally:
        pool.shutdown(wait=True)
    record_property("resolved_workers", resolved)
    record_property("peak_compile_callbacks", peak)
    assert len(results) == len(configs)
    assert 1 <= peak <= min(resolved, len(configs))
    if resolved > 1:
        assert peak > 1
    errors = [(config, repr(error)) for _, config, _, error in results if error is not None]
    assert not errors, errors
    a, b = [torch.randn(4096, device=npu_device) for _ in range(2)]
    expected = a.cpu() + b.cpu()
    for _, _, kernel, _ in results:
        torch.testing.assert_close(kernel(a, b).cpu(), expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    tilelang.testing.main()
