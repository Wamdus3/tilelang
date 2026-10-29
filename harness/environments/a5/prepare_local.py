"""Download Linux wheels locally and emit a hashed offline installation lock.

Run with CPython 3.11 inside local WSL. Never connects to A5 or builds TileLang.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

import tomllib
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pins():
    return [
        Requirement(line)
        for line in (CONFIG / "runtime.in").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def check_roots():
    repository = tomllib.loads((ROOT / "pyproject.toml").read_text())
    harness = tomllib.loads((ROOT / "harness" / "pyproject.toml").read_text())
    locked = {canonicalize_name(req.name): req for req in pins()}
    environment = {
        **default_environment(),
        "python_version": "3.11",
        "python_full_version": "3.11.15",
        "platform_system": "Linux",
        "sys_platform": "linux",
        "platform_machine": "x86_64",
    }
    for value in repository["project"]["dependencies"] + harness["project"]["dependencies"]:
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        pin = locked.get(canonicalize_name(requirement.name))
        if pin is None:
            raise ValueError(f"repository runtime dependency has no A5 pin: {requirement}")
        versions = [item.version for item in pin.specifier if item.operator == "=="]
        if len(versions) != 1 or not requirement.specifier.contains(versions[0]):
            raise ValueError(f"A5 pin {pin} violates repository requirement {requirement}")


def download(output, skip_download=False):
    output.mkdir(parents=True, exist_ok=True)
    wheels = output / "wheels"
    wheels.mkdir(exist_ok=True)
    base = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(wheels),
        "--only-binary=:all:",
        "--python-version",
        "311",
        "--implementation",
        "cp",
        "--abi",
        "cp311",
        "--abi",
        "abi3",
        "--abi",
        "none",
    ]
    # Include only ABI tags no newer than the destination's glibc.
    for minor in range(31, 4, -1):
        base += ["--platform", f"manylinux_2_{minor}_x86_64"]
    for platform in ("manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"):
        base += ["--platform", platform]
    if not skip_download:
        subprocess.run(
            base + ["--no-deps", "torch==2.10.0+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True
        )
        subprocess.run(
            base + ["--find-links", str(wheels), "-r", str(CONFIG / "runtime.in"), "-r", str(CONFIG / "build.in")],
            check=True,
        )
    entries = []
    lock = []
    names = set()
    versions = {}
    dependencies = []
    for path in sorted(wheels.glob("*.whl")):
        with zipfile.ZipFile(path) as wheel:
            files = [name for name in wheel.namelist() if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
            if len(files) != 1:
                raise ValueError(f"ambiguous wheel metadata: {path.name}")
            metadata = BytesParser().parsebytes(wheel.read(files[0]))
        name, version = metadata["Name"], metadata["Version"]
        if canonicalize_name(name) in names:
            raise ValueError(f"multiple versions/wheels for {name}; use a fresh output directory")
        names.add(canonicalize_name(name))
        versions[canonicalize_name(name)] = version
        dependencies.extend(Requirement(item) for item in metadata.get_all("Requires-Dist", []))
        key = sha256(path)
        entries.append(
            {"file": path.name, "name": name, "version": version, "sha256": key, "bytes": path.stat().st_size}
        )
        lock.append(f"{name}=={version} --hash=sha256:{key}")
    roots = pins() + [
        Requirement(line)
        for line in (CONFIG / "build.in").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    environment = {
        **default_environment(),
        "python_version": "3.11",
        "python_full_version": "3.11.15",
        "platform_system": "Linux",
        "sys_platform": "linux",
        "platform_machine": "x86_64",
        "extra": "",
    }
    for requirement in roots + dependencies:
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        value = versions.get(canonicalize_name(requirement.name))
        if value is None or not requirement.specifier.contains(value):
            raise ValueError(f"offline wheels do not satisfy dependency: {requirement}")
    (output / "requirements-offline.txt").write_text("\n".join(lock) + "\n")
    manifest = {
        "schema_version": 1,
        "target": {"python": "3.11", "arch": "x86_64", "max_glibc": "2.31", "cann": "9.1.0"},
        "inputs": {
            name: sha256(path)
            for name, path in {
                "pyproject.toml": ROOT / "pyproject.toml",
                "harness/pyproject.toml": ROOT / "harness/pyproject.toml",
                "runtime.in": CONFIG / "runtime.in",
                "build.in": CONFIG / "build.in",
            }.items()
        },
        "wheels": entries,
        "tilelang": "must be built from this checkout locally; no upstream tilelang wheel is accepted",
    }
    (output / "wheel-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PREPARED",
                "wheel_count": len(entries),
                "bytes": sum(entry["bytes"] for entry in entries),
                "output": str(output),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-download", action="store_true", help="regenerate locks for already downloaded wheels")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11) or sys.platform != "linux":
        parser.error("run with CPython 3.11 in local WSL")
    check_roots()
    download(args.output.resolve(), args.skip_download)


if __name__ == "__main__":
    main()
