"""Real candidate compilation, native tuning result persistence, and reuse."""

import json
import os
from pathlib import Path
import shutil

import pytest
import tilelang.testing

pytestmark = [pytest.mark.perf, *tilelang.testing.requires_ascend.marks()]


def test_real_tuning_result_is_saved_and_reused_in_memory(real_baseline, record_property):
    cold = real_baseline[0]
    assert cold["compile_count"] == 3
    assert len(cold["saved_tuning_results"]) == 1
    first, second = cold["runs"]
    assert [t["status"] for t in first["trials"]] == ["ok"] * 3
    assert second["same_result_object"]
    assert [t["status"] for t in second["trials"]] == ["cache_hit"]
    saved = cold["saved_tuning_results"][0]
    assert saved["config"] == first["config"]
    assert saved["latency"]["latency"] == first["latency_ms"] > 0
    assert all(candidate["passed"] for candidate in cold["candidate_correctness"])
    record_property("best_config", json.dumps(saved["config"]))
    record_property("latency_ms", first["latency_ms"])
    record_property("latency_scope", cold["tuning_latency_scope"])


def test_real_tuning_result_is_reused_in_a_fresh_process(real_baseline):
    cold, fresh, _ = real_baseline
    assert cold["pid"] != fresh["pid"]
    assert fresh["compile_count"] == 0
    assert [t["status"] for t in fresh["runs"][0]["trials"]] == ["cache_hit"]
    assert fresh["runs"][0]["config"] == cold["runs"][0]["config"]
    assert fresh["runs"][0]["latency_ms"] == cold["runs"][0]["latency_ms"]
    assert fresh["runs"][0]["correctness"]["passed"]


def test_real_kernel_compile_reuses_disk_and_memory_cache(real_baseline):
    direct = real_baseline[2]
    assert direct["compile_count"] == 0
    assert direct["runs"][1]["same_kernel_object"]
    assert all(run["correctness"]["passed"] for run in direct["runs"])


@pytest.mark.parametrize("scenario", ["missing-file", "corrupt-file", "absent-root", "corrupt-root"])
def test_tuning_result_load_errors(real_baseline, cache_lab, tmp_path, monkeypatch, scenario):
    from tilelang.autotuner import AutoTuner
    from tilelang.env import env
    from tilelang.jit.kernel import JITKernel
    from testing.ascend._ascend_test_utils import check_vector, vector_add

    root = tmp_path / "cache"
    shutil.copytree(cache_lab.cache, root)
    saved = real_baseline[0]["saved_tuning_results"][0]
    entry = root / Path(saved["path"]).relative_to(cache_lab.cache)
    roots = [root]
    if scenario == "absent-root":
        roots.insert(0, tmp_path / "absent")
    elif scenario == "corrupt-root":
        shutil.copytree(root, tmp_path / "fallback")
        roots.append(tmp_path / "fallback")
    config = entry / "best_config.json"
    if scenario == "missing-file":
        config.unlink()
    elif scenario.startswith("corrupt"):
        config.write_text("invalid json")
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", os.pathsep.join(map(str, roots)))
    monkeypatch.setattr(JITKernel, "_compile_and_create_adapter", lambda *a, **kw: pytest.fail("Cache loading must not compile"))
    tuner = AutoTuner(vector_add, [{"tile": v} for v in (1024, 2048, 4096)])
    tuner.set_compile_args(target="ascend", execution_backend="tvm_ffi", out_idx=-1)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if scenario == "absent-root":
        result = tuner._load_result_from_disk(entry.name)
        assert result.config == saved["config"]
        assert result.latency == saved["latency"]["latency"]
        check_vector(result.kernel)
    else:
        error = FileNotFoundError if scenario == "missing-file" else json.JSONDecodeError
        with pytest.raises(error):
            tuner._load_result_from_disk(entry.name)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


if __name__ == "__main__":
    tilelang.testing.main()
