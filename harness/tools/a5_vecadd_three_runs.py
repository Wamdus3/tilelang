"""One fixed vecadd example, compiled once and launched exactly three times.

No benchmark, warmup, autotuning, suite, or retry. Run only with explicit device
authorization. The default also requires no other process on the selected NPU.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path


def device_snapshot(index, allow_resident):
    result = subprocess.run(["npu-smi", "info"], check=True, capture_output=True, text=True, timeout=15)
    text = result.stdout
    devices = {}
    current = None
    for line in text.splitlines():
        match = re.match(r"^\|\s*(\d+)\s*\|\s*(Ascend950\S*)\s*\|\s*(\S+)", line)
        if match:
            current = int(match[1])
            devices[current] = {"name": match[2], "health": match[3]}
        usage = re.match(r"^\|\s*\|\s*\|\s*([\da-fA-F:.]+)\s*\|\s*(\d+)(?=\s)", line)
        if usage and current is not None:
            devices[current].update(bus_id=usage[1], utilization_percent=int(usage[2]))
    if index not in devices:
        raise RuntimeError("selected device is not an Ascend950")
    selected = devices[index]
    if selected.get("health") != "OK" or selected.get("utilization_percent") != 0:
        raise RuntimeError("selected NPU is unhealthy or not idle: " + repr(selected))
    if "Process id" not in text:
        raise RuntimeError("npu-smi process table is missing; cannot confirm availability")
    process_table = text.split("Process id", 1)[1]
    other_pids = []
    for line in process_table.splitlines():
        match = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line)
        if match and int(match[1]) == index and int(match[2]) != os.getpid():
            other_pids.append(int(match[2]))
    selected["other_process_ids"] = other_pids
    if other_pids and not allow_resident:
        raise RuntimeError("selected NPU has other processes: " + repr(other_pids))
    return {"selected": selected, "raw": text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-resident-processes", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "PREPARING",
        "operator": "vecadd",
        "shape": [1024],
        "dtype": "float32",
        "tile": 256,
        "requested_launches": 3,
        "attempted_launches": 0,
        "completed_launches": 0,
        "warmup_launches": 0,
        "profiling": "NOT_RUN",
        "autotune": "NOT_RUN",
        "allow_resident_processes": args.allow_resident_processes,
        "runs": [],
        "physical_device_index": args.device_index,
        "visibility": {key: os.environ.get(key) for key in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")},
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        # Refuse ambiguous logical-to-physical mapping.
        if any(report["visibility"].values()):
            raise RuntimeError("run without pre-existing Ascend visibility remapping")
        report["before"] = device_snapshot(args.device_index, args.allow_resident_processes)
        from tilelang_harness.device import lease

        with lease():
            import numpy as np
            import tilelang
            import torch
            import torch_npu
            from tilelang.backend.target import determine_target

            import tilelang_harness.baselines.vecadd as seed
            from tilelang_harness.baselines.vecadd import build

            report["versions"] = {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "tilelang": tilelang.__version__,
            }
            report["tilelang_file"] = tilelang.__file__
            seed_bytes = Path(seed.__file__).read_bytes()
            report["seed_sha256"] = hashlib.sha256(seed_bytes).hexdigest()
            (args.output / "vecadd_seed.py").write_bytes(seed_bytes)
            torch.npu.set_device(args.device_index)
            report["device_name"] = torch.npu.get_device_name(args.device_index)
            report["status"] = "COMPILING_OPERATOR"
            save()
            with determine_target("ascend", return_object=True):
                kernel = tilelang.compile(build(1024, tile=256), target="ascend", out_idx=[2])
            source = kernel.get_kernel_source()
            (args.output / "kernel_source.cpp").write_text(source)
            report["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            report["execution_backend"] = kernel.execution_backend
            # Check again after compilation, before any TileLang launch.
            report["before_launch"] = device_snapshot(args.device_index, args.allow_resident_processes)
            host_a = ((np.arange(1024, dtype=np.float32) % 17) - 8) / 8
            host_b = ((np.arange(1024, dtype=np.float32) % 13) - 6) / 4
            reference = host_a + host_b
            device = torch.device(f"npu:{args.device_index}")
            a = torch.from_numpy(host_a.copy()).to(device)
            b = torch.from_numpy(host_b.copy()).to(device)
            torch.npu.synchronize()

            def run_once(number):
                report["status"] = "RUNNING"
                report["attempted_launches"] += 1
                save()
                output = kernel(a, b)
                torch.npu.synchronize()
                report["completed_launches"] += 1
                if isinstance(output, (list, tuple)):
                    if len(output) != 1:
                        raise RuntimeError("expected one output")
                    output = output[0]
                actual = output.cpu().numpy()
                np.testing.assert_array_equal(actual, reference)
                np.testing.assert_array_equal(a.cpu().numpy(), host_a)
                np.testing.assert_array_equal(b.cpu().numpy(), host_b)
                record = {
                    "run": number,
                    "status": "PASS",
                    "verified_elements": 1024,
                    "max_abs_error": float(np.max(np.abs(actual - reference))),
                }
                report["runs"].append(record)
                save()
                print(json.dumps(record), flush=True)

            # User's exact budget: three launches, no warmup or hidden profiler loop.
            run_once(1)
            run_once(2)
            run_once(3)
            report["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001 - Persist diagnostics for arbitrary SDK failures.
        report.update(status="ERROR", error=str(exc), traceback=traceback.format_exc())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
