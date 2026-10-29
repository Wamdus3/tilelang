"""Verify installed repository wheels through full lowering/replay campaigns in local WSL."""

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "microsoft" not in platform.release().lower():
        parser.error("this multi-case verification tool is local WSL only; never run on A5")
    prefix, output = args.prefix.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "PYTHONNOUSERSITE": "1",
        "LD_LIBRARY_PATH": str(prefix / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    env.pop("PYTHONPATH", None)

    def cli(*arguments):
        result = subprocess.run(
            [str(prefix / "bin/python"), "-I", "-m", "tilelang_harness", *map(str, arguments)],
            cwd=output,
            env=env,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"CLI failed ({result.returncode}): {result.stdout}\n{result.stderr}")
        return json.loads(result.stdout)

    results = []
    for name in ("vecadd", "rmsnorm"):
        source, campaign = output / f"{name}-source", output / name
        cli("seed", name, "--destination", source)
        cli("init", name, "--run-dir", campaign, "--max-attempts", "2")
        candidate = cli("submit", source, "--run-dir", campaign)
        first = cli("run", candidate["candidate_id"], "--run-dir", campaign, "--mode", "lower", "--timeout-s", "180")
        second = cli("replay", first["attempt_id"], "--run-dir", campaign)
        report = cli("report", "--run-dir", campaign, "--output", output / f"{name}-report.json")
        if first["status"] != "PASS" or second["status"] != "PASS":
            raise ValueError("installed wheel lowering/replay failed")
        if first["builds"] != second["builds"] or first["evaluation_id"] == second["evaluation_id"]:
            raise ValueError("replay build identity or independent evaluation identity mismatch")
        if report["eligible_for_promotion"]:
            raise ValueError("source lowering cannot grant device promotion")
        results.append(
            {
                "task": name,
                "case_count": len(first["case_ids"]),
                "status": "PASS",
                "attempt_ids": [first["attempt_id"], second["attempt_id"]],
                "build_sets_match": True,
            }
        )
        print(f"{name}: packaged Ascend lowering and replay PASS", flush=True)
    (output / "report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "scope": "installed_portable_wheel_local_wsl_lowering",
                "results": results,
                "device_execution": "NOT_RUN",
                "eligible_for_promotion": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
