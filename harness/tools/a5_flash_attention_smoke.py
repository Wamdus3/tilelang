"""One repository MHA forward invocation; no benchmark, warmup, or retries."""

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from a5_vecadd_three_runs import device_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-resident-processes", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "PREPARING",
        "operator": "flash_attention_mha_forward",
        "shape": [512, 128],
        "input_dtype": "bfloat16",
        "output_dtype": "float32",
        "tiling": {"block_q": 128, "block_kv": 128, "num_stages": 3},
        "requested_launches": 1,
        "attempted_launches": 0,
        "completed_launches": 0,
        "warmup_launches": 0,
        "profiling": "NOT_RUN",
        "autotune": "NOT_RUN",
        "physical_device_index": args.device_index,
        "runs": [],
        "allow_resident_processes": args.allow_resident_processes,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        if any(os.environ.get(key) for key in ("ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")):
            raise RuntimeError("pre-existing device visibility remapping is not allowed")
        source_dir = args.source_dir.resolve()
        expected_hashes = json.loads((source_dir / "source-manifest.json").read_text())
        for name in ("core.py", "example_mha.py"):
            content = (source_dir / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected_hashes[name]:
                raise ValueError("source snapshot checksum mismatch: " + name)
            (args.output / name).write_bytes(content)
        report["example_source_sha256"] = expected_hashes
        report["before"] = device_snapshot(args.device_index, allow_resident=args.allow_resident_processes)
        from tilelang_harness.device import lease

        with lease():
            import tilelang
            import torch
            import torch_npu
            from tilelang.backend.target import determine_target

            torch.set_num_threads(2)
            report["versions"] = {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "tilelang": tilelang.__version__,
            }
            report["tilelang_file"] = tilelang.__file__
            report["ascend_home"] = os.environ.get("ASCEND_HOME_PATH")
            sys.path.insert(0, str(source_dir))
            from example_mha import mha, ref_program

            generator = torch.Generator(device="cpu").manual_seed(42)
            q_host = torch.randn((512, 128), generator=generator).to(torch.bfloat16)
            k_host = torch.randn((512, 128), generator=generator).to(torch.bfloat16)
            v_host = torch.randn((512, 128), generator=generator).to(torch.bfloat16)
            # The repository reference runs on CPU with the quantized input values.
            reference = ref_program(q_host.float(), k_host.float(), v_host.float())
            report["reference"] = "repository ref_program, CPU float32 from BF16 inputs"
            torch.npu.set_device(args.device_index)
            report["device_name"] = torch.npu.get_device_name(args.device_index)
            report["status"] = "COMPILING_OPERATOR"
            save()
            with determine_target("ascend", return_object=True):
                kernel = tilelang.compile(
                    mha(D_val=128, SEQ_LEN=512),
                    target="ascend",
                    out_idx=-1,
                    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
                )
            source = kernel.get_kernel_source()
            (args.output / "kernel_source.cpp").write_text(source)
            report["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            report["execution_backend"] = kernel.execution_backend
            report["before_launch"] = device_snapshot(args.device_index, allow_resident=args.allow_resident_processes)
            device = torch.device(f"npu:{args.device_index}")
            q = q_host.to(device)
            k = k_host.to(device)
            v = v_host.to(device)
            torch.npu.synchronize()
            report.update(status="RUNNING", attempted_launches=1)
            save()
            output = kernel(q, k, v)  # The only TileLang kernel invocation.
            torch.npu.synchronize()
            report["completed_launches"] = 1
            save()
            if isinstance(output, (tuple, list)):
                if len(output) != 1:
                    raise RuntimeError("expected one MHA output")
                output = output[0]
            actual = output.cpu().float()
            assert tuple(actual.shape) == (512, 128), actual.shape
            assert bool(torch.isfinite(actual).all()), "nonfinite MHA output"
            max_abs_error = float((actual - reference).abs().max())
            reference_max = float(reference.abs().max())
            relative_error = max_abs_error / max(reference_max, 1e-12)
            # Match the repository example's normalized maximum error criterion.
            assert relative_error < 0.05, f"relative maximum error {relative_error} >= 0.05"
            torch.testing.assert_close(q.cpu(), q_host, rtol=0, atol=0)
            torch.testing.assert_close(k.cpu(), k_host, rtol=0, atol=0)
            torch.testing.assert_close(v.cpu(), v_host, rtol=0, atol=0)
            report["runs"].append(
                {
                    "run": 1,
                    "status": "PASS",
                    "verified_elements": actual.numel(),
                    "max_abs_error": max_abs_error,
                    "relative_max_error": relative_error,
                    "relative_max_error_limit": 0.05,
                    "inputs_unchanged": True,
                }
            )
            report["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001 - Persist diagnostics for arbitrary SDK failures.
        report.update(status="ERROR", error=str(exc), traceback=traceback.format_exc())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
