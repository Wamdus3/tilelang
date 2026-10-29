"""Read-only preflight, also repeated after vendor compilation before dispatch."""

import os
import re
import subprocess

from .contracts import ContractError


def idle_snapshot(index, allow_resident=False):
    if any(os.environ.get(key) for key in ("ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")):
        raise ContractError("DEVICE_BUSY: ambiguous physical device visibility mapping")
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=15)
    devices, current = {}, None
    for line in result.stdout.splitlines():
        match = re.match(r"^\|\s*(\d+)\s*\|\s*(Ascend950\S*)\s*\|\s*(\S+)", line)
        if match:
            current = int(match[1])
            devices[current] = {"name": match[2], "health": match[3]}
        usage = re.match(r"^\|\s*\|\s*\|\s*([\da-fA-F:.]+)\s*\|\s*(\d+)(?=\s)", line)
        if usage and current is not None:
            devices[current].update(bus_id=usage[1], utilization=int(usage[2]))
    selected = devices.get(index, {})
    if selected.get("health") != "OK" or selected.get("utilization") != 0:
        raise ContractError("DEVICE_BUSY: selected Ascend950 is unavailable, unhealthy, or active")
    if "Process id" not in result.stdout:
        raise ContractError("DEVICE_BUSY: process table missing")
    pids = []
    for line in result.stdout.split("Process id", 1)[1].splitlines():
        match = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line)
        if match and int(match[1]) == index and int(match[2]) != os.getpid():
            pids.append(int(match[2]))
    if pids and not allow_resident:
        raise ContractError("DEVICE_BUSY: selected NPU has resident processes")
    return {**selected, "resident_pids": pids, "raw": result.stdout}
