"""Isolated native compilation, tuning and cache loading for Ascend tests."""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from testing.ascend._ascend_test_utils import check_vector, vector_add


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--mode", choices=["tune", "direct"], default="tune")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-failure", action="store_true")
    args = parser.parse_args()
    import torch_npu  # noqa: F401

    args.output = args.output.absolute()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["TILELANG_CACHE_DIR"] = str((args.cache_dir or args.output / "cache").absolute())
    os.environ["TILELANG_DISABLE_CACHE"] = "0"
    os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "0"
    os.environ.setdefault("TILELANG_AUTO_TUNING_CPU_COUNTS", "2")
    os.environ.setdefault("TILELANG_AUTO_TUNING_MAX_CPU_COUNT", "2")
    report = {"mode": args.mode, "pid": os.getpid(), "runs": []}
    report["tuning_backend"] = "wall"
    report["tuning_latency_scope"] = "native AutoTuner wall host-call latency"
    compilations = []

    def audit(event, values):
        if (
            event == "subprocess.Popen"
            and isinstance(values[1], (list, tuple))
            and any(str(arg).endswith("tl_kernel.asc") for arg in values[1])
        ):
            compilations.append(values[1])

    sys.addaudithook(audit)
    import tilelang
    from tilelang.autotuner import AutoTuner
    from tilelang.cache.kernel_cache import KernelCache
    from tilelang.cache.ascend_binary_cache import AscendBinaryCache

    roots = {
        "kernel": Path(KernelCache._get_write_cache_root()),
        "binary": Path(AscendBinaryCache._get_write_cache_root()),
        "autotune": AutoTuner._get_cache_dir(tilelang.env.get_cache_write_dir()),
    }
    report["cache_roots"] = {k: str(v) for k, v in roots.items()}
    stages = {
        "kernel": Path(KernelCache._get_staging_root()),
        "binary": Path(AscendBinaryCache._get_staging_root()),
        "autotune": roots["autotune"].parent / ".staging",
    }
    for path in [*roots.values(), *stages.values()]:
        path.mkdir(parents=True, exist_ok=True)
    # A real syscall control, separate from the production cache operations.
    report["replace_probes"] = {}
    for name, final in roots.items():
        src, dst = stages[name] / f"probe-{os.getpid()}", final / f"probe-{os.getpid()}"
        src.write_bytes(b"new")
        dst.write_bytes(b"old")
        try:
            os.replace(src, dst)
            observation = {"errno": None, "source_exists": src.exists(), "destination": dst.read_text()}
        except OSError as exc:
            observation = {"errno": exc.errno, "source": src.read_text(), "destination": dst.read_text()}
            assert exc.errno == 18 and observation["source"] == "new" and observation["destination"] == "old"
        finally:
            src.unlink(missing_ok=True)
            dst.unlink(missing_ok=True)
        report["replace_probes"][name] = observation

    if args.mode == "tune":
        configs = [{"tile": v} for v in (1024, 2048, 4096)]
        tuner = AutoTuner(vector_add, configs)
        tuner.set_compile_args(out_idx=-1, target="ascend", execution_backend="tvm_ffi")
        tuner.set_profile_args(warmup=5, rep=10, timeout=0, backend="wall", cache_input_tensors=True)
        last_result = None
        for iteration in range(2):
            trials = []
            try:
                result = tuner.run(warmup=5, rep=10, timeout=0, on_trial=trials.append)
            except Exception:
                report["runs"].append(
                    {
                        "iteration": iteration,
                        "trials": trials,
                        "error": traceback.format_exc(),
                    }
                )
                if not args.expect_failure:
                    raise
                assert trials and all(r["status"] == "compile_error" for r in trials), trials
                assert all("Invalid cross-device link" in r["error"] or "EXDEV" in r["error"] for r in trials)
                continue
            assert not args.expect_failure, "Expected BinaryCache EXDEV did not stop compilation"
            report["runs"].append(
                {
                    "iteration": iteration,
                    "trials": trials,
                    "config": result.config,
                    "latency_ms": result.latency,
                    "same_result_object": result is last_result,
                    "correctness": check_vector(result.kernel),
                }
            )
            assert result.latency is not None and result.latency > 0
            measured = [trial for trial in trials if trial["status"] == "ok"]
            if measured:
                best = min(measured, key=lambda trial: trial["latency_ms"])
                assert result.config == best["config"] and result.latency == best["latency_ms"]
            if iteration == 1:
                assert result is last_result and [r["status"] for r in trials] == ["cache_hit"]
            last_result = result
        if not args.expect_failure:
            report["candidate_correctness"] = []
            for config in configs:
                candidate = tilelang.compile(vector_add(**config), out_idx=-1, target="ascend", execution_backend="tvm_ffi")
                report["candidate_correctness"].append({"config": config, **check_vector(candidate)})
    else:
        last_kernel = None
        for iteration in range(2):
            try:
                kernel = tilelang.compile(vector_add(tile=2048), out_idx=-1, target="ascend", execution_backend="tvm_ffi")
            except Exception as error:
                if not args.expect_failure:
                    raise
                assert "[Errno 18]" in str(error) and "Invalid cross-device link" in str(error)
                report["runs"].append({"error": str(error)})
                continue
            assert not args.expect_failure
            report["runs"].append(
                {
                    "iteration": iteration,
                    "correctness": check_vector(kernel),
                    "same_kernel_object": kernel is last_kernel,
                }
            )
            if iteration == 1:
                assert kernel is last_kernel
            last_kernel = kernel

    report["artifacts"] = {name: [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()] for name, root in roots.items()}
    report["saved_tuning_results"] = []
    for config_path in sorted(roots["autotune"].glob("*/best_config.json")):
        report["saved_tuning_results"].append(
            {
                "path": str(config_path.parent),
                "config": json.loads(config_path.read_text()),
                "latency": json.loads((config_path.parent / "latency.json").read_text()),
            }
        )
    report["staging_leftovers"] = {k: [str(p) for p in v.iterdir()] for k, v in stages.items()}
    # Binary staging lives below the same version namespace; every private
    # staging entry must have been cleaned by production code.
    assert all(not items for items in report["staging_leftovers"].values())
    report["compile_count"] = len(compilations)
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
