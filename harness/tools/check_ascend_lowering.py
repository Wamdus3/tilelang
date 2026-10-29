"""WSL-only local source-generation check of the bundled Ascend seeds.

Never vendor-compiles, launches kernels, measures performance, or connects to a
server. This is optional P0 lowering evidence; the P1 CompilerBridge will own
build identity and device execution. Every fixed shape is lowered explicitly.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    import tilelang
    from tilelang.backend.target import determine_target

    from tilelang_harness.tasks.registry import load_task, source_path

    results = []
    for name in ("vecadd", "rmsnorm"):
        task = load_task(name)
        source = source_path(task.device_baseline)
        spec = importlib.util.spec_from_file_location(f"p0_{name}_baseline", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for case in task.cases:
            stem = f"{name}_{case.case_id}_{task.task_digest[:12]}"
            record = {
                "task_id": task.task_id,
                "task_digest": task.task_digest,
                "case_id": case.case_id,
                "builder_args": case.sizes(),
                "baseline_sha256": task.device_baseline.sha256,
            }
            if name == "rmsnorm":
                record["builder_args"]["epsilon"] = task.epsilon
            phase = "builder"
            try:
                with determine_target("ascend", return_object=True):
                    prim = module.build(**record["builder_args"])
                    phase = "lowering"
                    # Layout inference consumes the active target context as well.
                    # Both flags are explicit: no vendor compilation or device execution.
                    artifact = tilelang.lower(
                        prim, target="ascend", enable_host_codegen=False, enable_device_compile=False
                    )
                code = artifact.kernel_source
                if not code or not code.strip():
                    raise ValueError("Ascend lowering produced no source")
                phase = "artifact_export"
                code_path = args.output / f"{stem}.cpp"
                code_path.write_text(code, encoding="utf-8")
                input_ir_path = args.output / f"{stem}.input.tir"
                input_ir_path.write_text(prim.script(), encoding="utf-8")
                lowered_ir_path = args.output / f"{stem}.lowered.tir"
                lowered_ir_path.write_text(artifact.device_mod.script(), encoding="utf-8")
                record.update(
                    status="PASS",
                    source_path=str(code_path),
                    input_ir_path=str(input_ir_path),
                    lowered_ir_path=str(lowered_ir_path),
                    source_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
                )
            except Exception as exc:  # noqa: BLE001 - Preserve per-case compiler diagnostics and keep coverage.
                trace_path = args.output / f"{stem}.error.txt"
                trace_path.write_text(traceback.format_exc(), encoding="utf-8")
                record.update(
                    status="FAIL", phase=phase, error=f"{type(exc).__name__}: {exc}", traceback_path=str(trace_path)
                )
            results.append(record)
    report = {
        "schema_version": 1,
        "scope": "ascend_source_lowering_only",
        "tilelang_version": tilelang.__version__,
        "tilelang_python_path": tilelang.__file__,
        "status": "PASS" if all(x["status"] == "PASS" for x in results) else "FAIL",
        "results": results,
        "vendor_build": "NOT_RUN",
        "device_correctness": "NOT_RUN",
        "measurement": "NOT_RUN",
        "eligible_for_promotion": False,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(results), "report": str(report_path)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
