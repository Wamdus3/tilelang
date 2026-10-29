"""Strict on-disk job protocol, validated before importing a candidate or touching a device."""

from .bundle import parse_manifest
from .contracts import ContractError, TaskSpec, content_id, fields, integer, number


def validate_job(job):
    if not isinstance(job, dict):
        raise ContractError("job must be an object")
    optional = {"resource_limits", "allow_resident_processes", "experiment_trial_id"} & set(job)
    fields(
        job,
        {
            "schema_version",
            "job_id",
            "attempt_id",
            "candidate_id",
            "task_digest",
            "task",
            "bundle",
            "mode",
            "case_ids",
            "device_index",
            "measure",
            "timeout_s",
            "created",
            "runner_digest",
        }
        | optional,
        "job",
    )
    if type(job["schema_version"]) is not int or job["schema_version"] != 1:
        raise ContractError("unsupported job schema")
    for name in ("job_id", "attempt_id", "candidate_id", "task_digest", "runner_digest"):
        value = job[name]
        size = 32 if name in ("job_id", "attempt_id") else 64
        if not isinstance(value, str) or len(value) != size or any(c not in "0123456789abcdef" for c in value):
            raise ContractError(f"invalid job {name}")
    timeout = number(job["timeout_s"], "timeout_s", minimum=0.05)
    if timeout > 86400:
        raise ContractError("timeout exceeds one day")
    number(job["created"], "created", minimum=1)
    if "experiment_trial_id" in job:
        value = job["experiment_trial_id"]
        if not isinstance(value, str) or len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
            raise ContractError("invalid experiment trial identity")
    task = TaskSpec.from_dict(job["task"])
    if task.task_digest != job["task_digest"]:
        raise ContractError("job task identity mismatch")
    bundle = job["bundle"]
    fields(bundle, {"schema_version", "candidate_id", "task_digest", "manifest", "files"}, "candidate bundle")
    if content_id({k: v for k, v in bundle.items() if k != "candidate_id"}) != job["candidate_id"]:
        raise ContractError("job candidate identity mismatch")
    if bundle["candidate_id"] != job["candidate_id"] or bundle["task_digest"] != task.task_digest:
        raise ContractError("job candidate/task binding mismatch")
    manifest = parse_manifest(bundle["manifest"])
    if job["mode"] not in ("host", "lower", "device-smoke"):
        raise ContractError("unsupported job mode")
    if manifest["abi"] != ("numpy" if job["mode"] == "host" else "primfunc"):
        raise ContractError("job candidate ABI mismatch")
    if type(job["measure"]) is not bool or not isinstance(job["case_ids"], list):
        raise ContractError("invalid measurement/case selection")
    expected = [case.case_id for case in task.cases]
    if job["mode"] == "device-smoke":
        if len(job["case_ids"]) != 1 or job["case_ids"][0] not in expected:
            raise ContractError("device smoke requires exactly one registered case")
        if type(job["device_index"]) is not int or not 0 <= job["device_index"] < 64:
            raise ContractError("invalid NPU device index")
    elif job["case_ids"] != expected or job["device_index"] is not None or job["measure"]:
        raise ContractError("host/lower jobs require full case coverage and no device options")
    if "allow_resident_processes" in job and (
        type(job["allow_resident_processes"]) is not bool
        or (job["mode"] != "device-smoke" and job["allow_resident_processes"])
    ):
        raise ContractError("invalid resident process policy")
    if "resource_limits" in job:
        limits = fields(job["resource_limits"], {"launches", "vendor_builds"}, "resource limits")
        for key, value in limits.items():
            integer(value, key, 0)
            if value > 1000000:
                raise ContractError("per-job resource limit exceeds 1000000")
        if job["mode"] != "device-smoke" and any(limits.values()):
            raise ContractError("host/lower cannot reserve device resources")
        if job["mode"] == "device-smoke" and (
            limits["launches"] < (12 if job["measure"] else 3) or limits["vendor_builds"] < (2 if job["measure"] else 1)
        ):
            raise ContractError("resource budget cannot cover required correctness calls/builds")
    return job
