"""One process per attempt. This process never opens campaign SQLite."""

import importlib.util
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

from .artifacts import digest, read_json, write_json
from .budget import BudgetExhausted, ResourceBudget, limits_for
from .bundle import verify_materialized
from .compiler import compiler_identity, lower_case, runner_identity
from .contracts import ContractError, TaskSpec, content_id
from .judge import check_candidate, selfcheck
from .protocol import validate_job
from .state import LAYERS
from .tasks.registry import verify_sources


def empty_result(job):
    return {
        key: job[key]
        for key in (
            "schema_version",
            "job_id",
            "attempt_id",
            "candidate_id",
            "task_digest",
            "mode",
            "case_ids",
            "runner_digest",
            "device_index",
            "measure",
        )
    } | {
        "status": "ERROR",
        "category": "WORKER_ERROR",
        "cases": [],
        "builds": {},
        "evidence": dict.fromkeys(LAYERS, "NOT_RUN"),
        "artifacts": {},
        "resources": {
            "limits": limits_for(job),
            "used": {"launches": 0, "vendor_builds": 0},
            "accounting_complete": False,
        },
    }


def execute(work: Path):
    job = validate_job(read_json(work / "job.json"))
    result = empty_result(job)
    budget = ResourceBudget(limits_for(job), work / "resources.json")
    current_stage = "startup"

    def stage(value):
        nonlocal current_stage
        current_stage = value
        write_json(work / "stage.json", {"stage": value, "time": time.time()})

    started = time.monotonic()
    try:
        # A controller crash between spawn and PID commit cannot release an unregistered worker.
        while not (work / "release.json").exists():
            if time.time() > job["created"] + job["timeout_s"]:
                raise ContractError("worker was not released before its deadline")
            time.sleep(0.02)
        if read_json(work / "release.json") != {"attempt_id": job["attempt_id"]}:
            raise ContractError("invalid worker release")
        stage("verification")
        identity = runner_identity()
        if content_id(identity) != job["runner_digest"]:
            raise ContractError("runner changed after scheduling")
        task = TaskSpec.from_dict(job["task"])
        verify_sources(task)
        if task.task_digest != job["task_digest"]:
            raise ContractError("task identity mismatch")
        bundle = job["bundle"]
        if content_id({k: v for k, v in bundle.items() if k != "candidate_id"}) != job["candidate_id"]:
            raise ContractError("candidate identity mismatch")
        source = work / "candidate"
        verify_materialized(bundle, source)
        stage("reference_selfcheck")
        reference_task = task
        if job["mode"] == "device-smoke":
            selected = next(case for case in task.cases if case.case_id == job["case_ids"][0])
            reference_task = replace(task, cases=(selected,))
        baseline = selfcheck(reference_task)
        baseline["original_task_digest"] = task.task_digest
        write_json(work / "artifacts" / "host-baseline.json", baseline)
        if baseline["status"] != "PASS":
            raise ContractError("trusted host baseline/reference selfcheck failed")
        if job["mode"] == "device-smoke":
            stage("device_environment")
            from .device import environment

            torch, device_env = environment(job["device_index"])
            write_json(work / "artifacts" / "device-environment.json", device_env)
            from .npu_status import idle_snapshot

            write_json(
                work / "artifacts" / "device-before-build.json",
                idle_snapshot(job["device_index"], job.get("allow_resident_processes", False)),
            )
        if job["mode"] in ("lower", "device-smoke"):
            stage("compiler_import")
            import tilelang

            tilelang.disable_cache()
            pack = compiler_identity(tilelang)
            write_json(work / "artifacts" / "compiler-pack.json", pack)
        stage("candidate_import")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location("_tlh_solution", source / "solution.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        entry = bundle["manifest"]["entrypoint"].split(":")[1]
        candidate = getattr(module, entry)
        if not callable(candidate):
            raise TypeError("candidate entrypoint is not callable")
        if job["mode"] == "host":
            stage("host_correctness")
            report = check_candidate(task, candidate)
            write_json(work / "artifacts" / "host-judge.json", report)
            result.update(
                status=report["status"],
                cases=report["cases"],
                scope="isolated_host_diagnostic",
                category="HOST_CHECK" if report["status"] != "ERROR" else "REFERENCE_ERROR",
            )
        elif job["mode"] == "lower":
            for index, case in enumerate(task.cases):
                build = lower_case(task, case, bundle, candidate, work / "artifacts" / f"case-{index}", pack, stage)
                result["builds"][case.case_id] = build
                result["cases"].append({"case_id": case.case_id, "status": "PASS"})
            result.update(status="PASS", category="LOWERED", scope="ascend_source_lowering_only")
            result["evidence"]["lowering"] = "PASS"
        else:
            from .device import Invocation, compile_handle, compiled_identity, trusted_builder, verify_one
            from .evaluation import measure_pair

            case = next(case for case in task.cases if case.case_id == job["case_ids"][0])
            destination = work / "artifacts" / "case-0"
            build, prim = lower_case(task, case, bundle, candidate, destination, pack, stage, return_prim=True)
            result["evidence"]["lowering"] = "PASS"
            result["builds"][case.case_id] = build
            stage("vendor_build")
            budget.consume("vendor_builds")
            kernel, options = compile_handle(task, prim)
            result["compiled_builds"] = {case.case_id: compiled_identity(build, kernel, options, destination)}
            result["evidence"]["vendor_build"] = "PASS"
            invoke = Invocation(task, kernel, torch, job["device_index"], budget)
            write_json(
                destination / "device-before-launch.json",
                idle_snapshot(job["device_index"], job.get("allow_resident_processes", False)),
            )
            stage("device_correctness")
            correctness = verify_one(task, case, invoke)
            write_json(destination / "correctness.json", correctness)
            result["cases"] = correctness["cases"]
            result["evidence"]["device_correctness"] = correctness["status"]
            result.update(status=correctness["status"], category="DEVICE_SMOKE", scope="single_case_npu_smoke")
            if job["measure"] and correctness["status"] == "PASS":
                # Compile/validate the pinned baseline in this SAME leased worker/environment.
                baseline_bundle = {
                    "candidate_id": content_id(task.to_dict()["device_baseline"]),
                    "manifest": {"config": {}},
                }
                baseline_destination = work / "artifacts" / "baseline"
                baseline_build, baseline_prim = lower_case(
                    task,
                    case,
                    baseline_bundle,
                    trusted_builder(task),
                    baseline_destination,
                    pack,
                    lambda name: stage("baseline_" + name),
                    return_prim=True,
                )
                write_json(baseline_destination / "lowering-build.json", baseline_build)
                stage("baseline_vendor_build")
                budget.consume("vendor_builds")
                baseline_kernel, baseline_options = compile_handle(task, baseline_prim)
                result["baseline_build"] = compiled_identity(
                    baseline_build, baseline_kernel, baseline_options, baseline_destination
                )
                stage("baseline_correctness")
                baseline_invoke = Invocation(task, baseline_kernel, torch, job["device_index"], budget)
                baseline_report = verify_one(task, case, baseline_invoke)
                write_json(baseline_destination / "correctness.json", baseline_report)
                if baseline_report["status"] != "PASS":
                    raise ContractError("device baseline/reference verification failed")
                stage("measurement")
                result["measurement"] = measure_pair(task, case, baseline_invoke, invoke)
                # Recheck the same handles after timing. Cached/stale outputs cannot become timing evidence.
                stage("post_measurement_correctness")
                if any(verify_one(task, case, item)["status"] != "PASS" for item in (baseline_invoke, invoke)):
                    raise ContractError("post-measurement correctness failed")
                result["evidence"]["measurement"] = "PASS"
            torch.npu.synchronize()
        stage("post_verification")
        verify_materialized(bundle, source)
        verify_sources(task)
        if content_id(runner_identity()) != job["runner_digest"]:
            raise ContractError("runner source changed during execution")
    except BaseException as exc:  # noqa: BLE001 - Persist a result even when candidate code exits the worker.
        category = {
            "candidate_import": "IMPORT_ERROR",
            "compiler_import": "ENVIRONMENT_UNAVAILABLE",
            "builder": "BUILDER_ERROR",
            "lowering": "LOWERING_ERROR",
            "artifact_export": "ARTIFACT_ERROR",
            "reference_selfcheck": "REFERENCE_ERROR",
            "post_verification": "SOURCE_DRIFT",
            "device_environment": "ENVIRONMENT_UNAVAILABLE",
            "vendor_build": "VENDOR_BUILD_ERROR",
            "device_correctness": "DEVICE_ERROR",
            "baseline_vendor_build": "BASELINE_BUILD_ERROR",
            "baseline_correctness": "REFERENCE_ERROR",
            "measurement": "MEASUREMENT_ERROR",
            "post_measurement_correctness": "POST_MEASUREMENT_ERROR",
            "baseline_builder": "BASELINE_BUILD_ERROR",
            "baseline_lowering": "BASELINE_BUILD_ERROR",
            "baseline_artifact_export": "ARTIFACT_ERROR",
        }.get(current_stage, "WORKER_ERROR")
        if isinstance(exc, BudgetExhausted):
            category = "BUDGET_EXHAUSTED"
            if "torch" in locals():
                torch.npu.synchronize()
        if str(exc).startswith("DEVICE_BUSY:"):
            category = "DEVICE_BUSY"
        result.update(
            status="INCOMPLETE" if category in ("ENVIRONMENT_UNAVAILABLE", "DEVICE_BUSY") else "ERROR",
            category=category,
            error=f"{type(exc).__name__}: {exc}",
            stage=current_stage,
        )
        if current_stage in ("builder", "lowering", "artifact_export"):
            result["evidence"]["lowering"] = "ERROR"
        if current_stage == "vendor_build":
            result["evidence"]["vendor_build"] = "ERROR"
        if current_stage == "device_correctness":
            result["evidence"]["device_correctness"] = "ERROR"
        if current_stage in ("measurement", "post_measurement_correctness"):
            result["evidence"]["measurement"] = "ERROR"
        if current_stage == "post_verification":
            result["evidence"] = {k: "ERROR" if v == "PASS" else v for k, v in result["evidence"].items()}
        if (
            category not in ("BUDGET_EXHAUSTED", "DEVICE_BUSY")
            and job["mode"] == "device-smoke"
            and current_stage
            in (
                "vendor_build",
                "device_correctness",
                "baseline_vendor_build",
                "baseline_correctness",
                "measurement",
                "post_measurement_correctness",
                "candidate_import",
                "post_verification",
            )
        ):
            from .device import quarantine

            quarantine(category, job["attempt_id"])
        (work / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    result["resources"] = budget.snapshot()
    result["wall_seconds"] = time.monotonic() - started
    for path in sorted((work / "artifacts").rglob("*")):
        if path.is_file() and not path.is_symlink():
            result["artifacts"][path.relative_to(work).as_posix()] = digest(path.read_bytes())
    result["evaluation_id"] = content_id({"job": job, "result": result})
    write_json(work / "result.json", result)


def main():
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
    execute(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
