"""Read-only preflight; compatible with system Python 3.8. Never imports TileLang/torch.

May be streamed to an approved server's `python3 -`; no file transfer/install/build required.
"""

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    packages = {}
    for name in ("torch", "torch-npu", "tilelang", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    executables = {name: shutil.which(name) for name in ("python3", "bisheng", "npu-smi")}
    status = None
    if executables["npu-smi"]:
        result = subprocess.run(
            [executables["npu-smi"], "info"], capture_output=True, text=True, timeout=20, check=False
        )
        status = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    # List executable locations only; never activate or alter shared environments.
    conda_root = Path("/data/miniconda3/envs")
    envs = (
        sorted(str(p / "bin/python") for p in conda_root.iterdir() if (p / "bin/python").is_file())
        if conda_root.is_dir()
        else []
    )
    print(
        json.dumps(
            {
                "scope": "read_only_environment_probe",
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "libc": platform.libc_ver(),
                "packages": packages,
                "executables": executables,
                "npu_smi": status,
                "existing_env_pythons": envs,
                "minimum_harness_python": "3.10",
                "no_device_kernel_executed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
