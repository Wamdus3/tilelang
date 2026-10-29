"""Check a built harness wheel from a temporary cwd without repository imports."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    package = destination / "unpacked"
    with zipfile.ZipFile(args.wheel) as wheel:
        wheel.extractall(package)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package)
    with tempfile.TemporaryDirectory(prefix="tlh-wheel-") as cwd:

        def cli(*arguments):
            result = subprocess.run(
                [sys.executable, "-m", "tilelang_harness", *map(str, arguments)],
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                timeout=90,
                check=True,
            )
            return json.loads(result.stdout)

        inspection = subprocess.run(
            [
                sys.executable,
                "-c",
                "import tilelang_harness, sys; print(tilelang_harness.__file__); assert 'tilelang' not in sys.modules",
            ],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        assert Path(inspection.stdout.strip()).is_relative_to(package)
        baseline = cli("selfcheck", "all", "--output", destination / "selfcheck.json")
        assert baseline["status"] == "PASS"
        results = []
        for name in ("vecadd", "rmsnorm"):
            source = Path(cwd) / name
            run_dir = destination / name
            cli("seed", name, "--destination", source, "--abi", "numpy")
            cli("init", name, "--run-dir", run_dir, "--max-attempts", 2)
            frozen = cli("submit", source, "--run-dir", run_dir)
            first = cli("run", frozen["candidate_id"], "--run-dir", run_dir, "--mode", "host", "--timeout-s", 30)
            second = cli("replay", first["attempt_id"], "--run-dir", run_dir)
            report = cli("report", "--run-dir", run_dir, "--output", destination / f"{name}-report.json")
            assert first["status"] == second["status"] == "PASS"
            assert first["evaluation_id"] != second["evaluation_id"]
            assert not report["eligible_for_promotion"]
            results.append({"task": name, "status": "PASS", "attempts": len(report["attempts"])})
    payload = {
        "scope": "installed_wheel_host_campaign_and_replay",
        "status": "PASS",
        "results": results,
        "package": str(package),
        "eligible_for_promotion": False,
    }
    (destination / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
