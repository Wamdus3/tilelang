"""Detached watchdog survives a controller exit and enforces the job deadline."""

import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from .artifacts import read_json, write_json
from .contracts import ContractError, content_id
from .protocol import validate_job
from .runner import process_identity, stop_group
from .worker import empty_result


def main():
    work = Path(sys.argv[1]).resolve()
    job = validate_job(read_json(work / "job.json"))
    identity = process_identity(os.getpid())
    if job["mode"] == "device-smoke":
        from .device import lease

        context = lease()
    else:
        context = nullcontext(None)
    try:
        with context as lease_fd:
            supervise(work, job, identity, lease_fd)
    except ContractError as exc:
        if not str(exc).startswith(("DEVICE_BUSY:", "DEVICE_QUARANTINED:")):
            raise
        result = empty_result(job)
        result.update(status="INCOMPLETE", category=str(exc).split(":")[0], error=str(exc))
        result["evaluation_id"] = content_id({"job": job, "result": result})
        write_json(work / "result.json", result)
        write_json(work / "termination.json", {"reason": "NOT_STARTED", "finished": time.time()})


def supervise(work, job, identity, lease_fd):
    deadline = time.monotonic() + max(0, job["created"] + job["timeout_s"] - time.time())
    child = subprocess.Popen(
        [sys.executable, "-m", "tilelang_harness.worker", str(work)],
        close_fds=True,
        pass_fds=() if lease_fd is None else (lease_fd,),
    )
    reason = "EXIT"
    try:
        while child.poll() is None:
            if time.monotonic() >= deadline:
                reason = "TIMEOUT"
                break
            time.sleep(0.02)
    finally:
        if job["mode"] == "device-smoke" and (reason == "TIMEOUT" or child.poll() not in (None, 0)):
            from .device import quarantine

            quarantine(reason, job["attempt_id"])
        stop_group(identity, exclude_pid=os.getpid())
        child.wait(timeout=3)
        write_json(
            work / "termination.json", {"reason": reason, "exit_code": child.returncode, "finished": time.time()}
        )


if __name__ == "__main__":
    main()
