"""Detached command watchdog; a controller restart never repeats a model request."""

import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import read_json, write_json
from .runner import process_identity, stop_group


def main():
    started = time.monotonic()
    work = Path(sys.argv[1]).resolve()
    request = read_json(work / "invocation.json")
    deadline = request["deadline"]
    identity = process_identity(os.getpid())
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    while not (work / "release.json").exists():
        if time.time() >= deadline:
            write_json(work / "termination.json", {"status": "INTERRUPTED", "reason": "not released"})
            return
        time.sleep(0.02)
    if time.time() >= deadline:
        write_json(work / "termination.json", {"status": "TIMEOUT", "reason": "deadline before command startup"})
        return
    process = None
    result = {"status": "ERROR", "reason": "command startup failed"}
    try:
        command = request["command"] + [
            "--request",
            str(work / "context.json"),
            "--response",
            str(work / "response.json"),
        ]
        with (work / "stdout.log").open("wb") as stdout, (work / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=request["cwd"], stdout=stdout, stderr=stderr)
            try:
                code = process.wait(timeout=max(0.001, deadline - time.time()))
                result = {"status": "PASS" if code == 0 else "ERROR", "exit_code": code}
            except subprocess.TimeoutExpired:
                result = {"status": "TIMEOUT"}
    except Exception as exc:  # noqa: BLE001 - The runtime boundary must persist command startup failures.
        result = {"status": "ERROR", "reason": str(exc)}
    finally:
        stop_group(identity, exclude_pid=os.getpid())
        if process is not None:
            process.wait(timeout=3)
        write_json(work / "termination.json", {**result, "wall_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
