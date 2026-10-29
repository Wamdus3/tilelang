"""Linux process watchdog and recovery; imports/builds share a hard wall deadline."""

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from .artifacts import digest, read_json, write_json
from .bundle import materialize, relative_name
from .compiler import runner_identity
from .contracts import ContractError, content_id, number
from .store import new_id
from .worker import empty_result


def process_identity(pid):
    try:
        # comm can contain spaces and parentheses, hence split after the LAST ')'.
        values = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": int(values[19]),
            "session": int(values[3]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "state": values[0],
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(identity):
    current = process_identity(identity["pid"])
    return (
        current is not None
        and current["state"] != "Z"
        and all(current[k] == identity[k] for k in ("pid", "start_ticks", "session", "boot_id"))
    )


def group_members(identity):
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != identity["boot_id"]:
        return []
    leader = process_identity(identity["pid"])
    if leader and leader["start_ticks"] != identity["start_ticks"]:
        return []  # PID reused: never signal a different process/session.
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            member = process_identity(int(path.name))
            if (
                member
                and member["session"] == identity["session"]
                and member["state"] != "Z"
                and member["start_ticks"] >= identity["start_ticks"]
            ):
                result.append(member)
    return result


def stop_group(identity, exclude_pid=None):
    for sig, grace in ((signal.SIGTERM, 0.3), (signal.SIGKILL, 1.0)):
        members = [m for m in group_members(identity) if m["pid"] != exclude_pid]
        if not members:
            return
        # Include ordinary subprocesses in other process groups within this worker session.
        for member in members:
            if alive(member):
                with suppress(ProcessLookupError):
                    os.kill(member["pid"], sig)
        until = time.monotonic() + grace
        while time.monotonic() < until and any(m["pid"] != exclude_pid for m in group_members(identity)):
            time.sleep(0.02)
    if any(m["pid"] != exclude_pid for m in group_members(identity)):
        raise ContractError("worker descendants remain alive; campaign stays unfinished")


def _failure(job, work, category, message):
    result = empty_result(job)
    if (work / "resources.json").is_file():
        result["resources"] = {**read_json(work / "resources.json"), "accounting_complete": False}
    try:
        stage = read_json(work / "stage.json")
        label = stage.get("stage", "unknown") if isinstance(stage, dict) else "unknown"
    except (OSError, ContractError):
        label = "unknown"
    result.update(status="ERROR", category=category, error=message, stage=label)
    result["evaluation_id"] = content_id({"job": job, "result": result})
    return result


def ingest(campaign, attempt, fallback=None):
    job = campaign.objects.json(attempt["job_ref"])
    work = campaign.root / "work" / job["attempt_id"]
    if fallback is None:
        result = read_json(work / "result.json")
        from .state import reduce_result

        reduce_result(job, result)
        if result.get("evaluation_id") != content_id(
            {"job": job, "result": {k: v for k, v in result.items() if k != "evaluation_id"}}
        ):
            raise ContractError("evaluation checksum mismatch")
        for name, key in result.get("artifacts", {}).items():
            relative_name(name)
            if not name.startswith("artifacts/"):
                raise ContractError("worker artifacts must be under artifacts/")
            path = work / name
            if path.is_symlink() or not path.resolve().is_relative_to(work.resolve()):
                raise ContractError("artifact escapes worker directory")
            data = path.read_bytes()
            if digest(data) != key:
                raise ContractError(f"worker artifact checksum mismatch: {name}")
            campaign.objects.put(data)
    else:
        result = fallback
    # Logs are controller-owned evidence even if import crashes before a result exists.
    result["logs"] = {
        name: campaign.objects.put((work / name).read_bytes())
        for name in ("stdout.log", "stderr.log", "traceback.txt", "stage.json")
        if (work / name).is_file()
    }
    result["evaluation_id"] = content_id(
        {"job": job, "result": {k: v for k, v in result.items() if k != "evaluation_id"}}
    )
    campaign.finish(job["attempt_id"], result)
    return result


def _await(campaign, attempt, process=None):
    job = campaign.objects.json(attempt["job_ref"])
    identity = campaign.objects.json(attempt["process_ref"])
    work = campaign.root / "work" / job["attempt_id"]
    timed_out = False
    while alive(identity):
        if time.time() >= job["created"] + job["timeout_s"] + 3:
            timed_out = True
            break
        time.sleep(0.025)
    stop_group(identity)
    if process is not None:
        process.wait(timeout=3)
    termination = read_json(work / "termination.json") if (work / "termination.json").is_file() else {}
    if job["mode"] == "device-smoke" and not termination:
        from .device import quarantine

        quarantine("WATCHDOG_EXIT", job["attempt_id"])
        return ingest(
            campaign, attempt, _failure(job, work, "WATCHDOG_EXIT", "device worker termination is unconfirmed")
        )
    if timed_out or termination.get("reason") == "TIMEOUT":
        if job["mode"] == "device-smoke":
            from .device import quarantine

            quarantine("TIMEOUT", job["attempt_id"])
        result = _failure(job, work, "TIMEOUT", "hard wall deadline exceeded; worker session terminated")
        return ingest(campaign, attempt, result)
    if (work / "result.json").is_file():
        try:
            return ingest(campaign, attempt)
        except (ContractError, OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
            result = _failure(job, work, "RESULT_PROTOCOL_ERROR", str(exc))
            result["rejected_result"] = campaign.objects.put((work / "result.json").read_bytes())
            return ingest(campaign, attempt, result)
    result = _failure(job, work, "WORKER_EXIT", "worker exited without a complete result")
    if job["mode"] == "device-smoke":
        from .device import quarantine

        quarantine("WORKER_EXIT", job["attempt_id"])
    return ingest(campaign, attempt, result)


def run(
    campaign,
    candidate_id,
    mode="lower",
    timeout_s=120,
    case_id=None,
    device_index=0,
    measure=False,
    max_launches=3,
    allow_resident_processes=False,
    trial_id=None,
):
    timeout_s = number(timeout_s, "timeout_s", minimum=0.05)
    if timeout_s > 86400:
        raise ContractError("timeout exceeds one day")
    if mode not in ("host", "lower", "device-smoke"):
        raise ContractError("mode must be host, lower or device-smoke")
    if type(device_index) is not int or not 0 <= device_index < 64 or type(measure) is not bool:
        raise ContractError("invalid device index/measurement flag")
    if mode != "device-smoke" and (case_id is not None or measure or device_index != 0):
        raise ContractError("device options require device-smoke mode")
    with campaign.lock():
        task = campaign.task()
        bundle = campaign.bundle(candidate_id)
        abi = "numpy" if mode == "host" else "primfunc"
        if bundle["manifest"]["abi"] != abi:
            raise ContractError(f"{mode} requires candidate ABI {abi}")
        if bundle["task_digest"] != task.task_digest:
            raise ContractError("candidate is frozen for a different task")
        case_ids = [case.case_id for case in task.cases]
        if mode == "device-smoke":
            if case_id not in case_ids:
                raise ContractError("device-smoke requires one explicit --case from the fixed task")
            case_ids = [case_id]
        identity = runner_identity()
        attempt_id = new_id()
        job = {
            "schema_version": 1,
            "job_id": new_id(),
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "task_digest": task.task_digest,
            "task": task.to_dict(),
            "bundle": bundle,
            "mode": mode,
            "case_ids": case_ids,
            "device_index": device_index if mode == "device-smoke" else None,
            "measure": measure,
            "timeout_s": timeout_s,
            "created": time.time(),
            "runner_digest": content_id(identity),
        }
        job["resource_limits"] = {
            "launches": max_launches if mode == "device-smoke" else 0,
            "vendor_builds": (2 if measure else 1) if mode == "device-smoke" else 0,
        }
        job["allow_resident_processes"] = allow_resident_processes
        if trial_id is not None:
            job["experiment_trial_id"] = trial_id
        from .protocol import validate_job

        validate_job(job)
        campaign.reserve(job)
        work = campaign.root / "work" / attempt_id
        work.mkdir(parents=True)
        materialize(bundle, work / "candidate", campaign.objects)
        write_json(work / "job.json", job)
        write_json(work / "runner.json", identity)
        env = os.environ.copy()
        paths = [str(Path(__file__).resolve().parents[1])]
        # Freeze interpreter import search locations as absolute paths, independent of worker cwd.
        paths.extend(str(Path(path or os.getcwd()).resolve()) for path in sys.path)
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with (work / "stdout.log").open("wb") as stdout, (work / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-m", "tilelang_harness.watchdog", str(work)],
                cwd=work,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
            identity = process_identity(process.pid)
            if identity is None:
                process.wait()
                return ingest(
                    campaign,
                    campaign.attempt(attempt_id),
                    _failure(job, work, "STARTUP_ERROR", "worker exited before registration"),
                )
            campaign.started(attempt_id, identity)
            write_json(work / "release.json", {"attempt_id": attempt_id})
            return _await(campaign, campaign.attempt(attempt_id), process)


def resume(campaign):
    results = []
    with campaign.lock():
        for attempt in campaign.attempts():
            if attempt["status"] == "FINISHED":
                continue
            job = campaign.objects.json(attempt["job_ref"])
            work = campaign.root / "work" / job["attempt_id"]
            if attempt["process_ref"]:
                # Wait on precisely the recorded PID incarnation. Never launch a replacement here.
                result = _await(campaign, attempt)
            else:
                # An unregistered process cannot pass the release handshake. Wait out that deadline.
                remaining = job["created"] + job["timeout_s"] - time.time()
                if remaining > 0:
                    results.append(
                        {"attempt_id": job["attempt_id"], "status": "STARTING", "retry_after_seconds": remaining}
                    )
                    continue
                work.mkdir(parents=True, exist_ok=True)
                result = ingest(
                    campaign, attempt, _failure(job, work, "INTERRUPTED", "controller stopped before release")
                )
            results.append(result)
    return results


def replay(campaign, attempt_id, timeout_s=None):
    old = campaign.attempt(attempt_id)
    if old["status"] != "FINISHED":
        raise ContractError("resume unfinished attempts before replay")
    job = campaign.objects.json(old["job_ref"])
    return run(
        campaign,
        old["candidate_id"],
        job["mode"],
        timeout_s if timeout_s is not None else job["timeout_s"],
        job["case_ids"][0] if job["mode"] == "device-smoke" else None,
        job.get("device_index") or 0,
        job.get("measure", False),
        job.get("resource_limits", {}).get("launches", 3),
        job.get("allow_resident_processes", False),
    )
