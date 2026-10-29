"""Assemble an audited offline bundle after local native wheel build and validation."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from audit_abi import audit


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--packing-python", required=True, type=Path)
    parser.add_argument("--validation-dir", type=Path, help="fresh directory for this validation attempt")
    args = parser.parse_args()
    output, prefix = args.output.resolve(), args.prefix.resolve()
    validation = args.validation_dir.resolve() if args.validation_dir else output / "portable-lowering"
    packer = args.packing_python.parent / "conda-pack"
    if not packer.is_file():
        raise FileNotFoundError("packing interpreter must have the conda-pack CLI installed")
    repo = Path(__file__).resolve().parents[3]
    wheels = output / "repository-wheels"
    native = list(wheels.glob("tilelang-*.whl"))
    harness = list(wheels.glob("tilelang_harness-*.whl"))
    if len(native) != 1 or len(harness) != 1:
        raise ValueError("requires exactly one locally built repository wheel and one harness wheel")
    with zipfile.ZipFile(native[0]) as archive:
        if "tilelang/ascend/language/__init__.py" not in archive.namelist():
            raise ValueError("repository wheel does not contain the Ascend dialect")
    for name in ("pyproject.toml", "harness/pyproject.toml"):
        inputs = json.loads((output / "wheel-manifest.json").read_text())["inputs"]
        if inputs[name] != file_hash(repo / name):
            raise ValueError("repository requirements changed since dependency resolution")
    env = {
        **os.environ,
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "PYTHONNOUSERSITE": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LD_LIBRARY_PATH": str(prefix / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    python = prefix / "bin/python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(native[0]),
            str(harness[0]),
        ],
        env=env,
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], env=env, check=True)
    # Execute from outside the checkout so an editable/development import cannot mask a broken wheel.
    probe = """
import json, pathlib, sys, torch, torch_c_dlpack_ext, tilelang, tilelang_harness
from tilelang.backend.target import determine_target
import tilelang.ascend.language as T
prefix = pathlib.Path(sys.prefix).resolve()
assert pathlib.Path(tilelang.__file__).resolve().is_relative_to(prefix)
assert callable(T.SimtVF)
with determine_target('ascend', return_object=True):
    pass
print(json.dumps({'python':sys.version, 'prefix':str(prefix), 'torch':torch.__version__,
 'tilelang':tilelang.__version__, 'tilelang_file':tilelang.__file__,
 'harness':tilelang_harness.__version__, 'device_execution':'NOT_RUN'}))
"""
    result = subprocess.run(
        [str(python), "-I", "-c", probe],
        cwd=prefix.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "installed-wheel-import.json").write_text(result.stdout)
    (output / "installed-wheel-import.stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError("installed wheel import failed; see installed-wheel-import.stderr.log")
    subprocess.run(
        [
            str(python),
            str(Path(__file__).parent / "verify_local.py"),
            "--prefix",
            str(prefix),
            "--output",
            str(validation),
        ],
        cwd=prefix.parent,
        env=env,
        check=True,
    )
    report = audit([prefix], (2, 31))
    (output / "environment-abi.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "PASS":
        raise ValueError("environment contains ELF binaries incompatible with A5")
    bundle = output / "bundle"
    bundle.mkdir(exist_ok=True)
    archive = bundle / "conda-env.tar.gz"
    if archive.exists():
        raise FileExistsError("refusing to overwrite a previously packed environment")
    subprocess.run(
        [
            str(args.packing_python),
            str(packer),
            "--prefix",
            str(prefix),
            "--output",
            str(archive),
            "--format",
            "tar.gz",
            "--n-threads",
            "8",
            "--compress-level",
            "4",
        ],
        env=env,
        check=True,
    )
    numeric = [
        tuple(map(int, value[6:].split("."))) for item in report["artifacts"] for value in item["glibc_requirements"]
    ]
    cache = prefix.parent / "native-build" / "CMakeCache.txt"
    settings = {}
    for line in cache.read_text().splitlines():
        if not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            settings[key.split(":", 1)[0]] = value
    profile = {key: settings[key] for key in ("USE_ASCEND", "USE_CUDA", "USE_ROCM", "USE_LLVM")}
    if profile != {"USE_ASCEND": "ON", "USE_CUDA": "OFF", "USE_ROCM": "OFF", "USE_LLVM": "OFF"}:
        raise ValueError("native build profile differs from the declared Ascend-only target")
    manifest = {
        "schema_version": 1,
        "status": "READY_FOR_OFFLINE_INSTALL",
        "archive_sha256": file_hash(archive),
        "archive_bytes": archive.stat().st_size,
        "minimum_glibc": ".".join(map(str, max(numeric))),
        "target_glibc": "2.31",
        "python": "3.11.15",
        "cann": "9.1.0",
        "build_location": "local_wsl",
        "native_build_profile": profile,
        "repository_wheels": {p.name: file_hash(p) for p in native + harness},
        "dependency_manifest_sha256": file_hash(output / "wheel-manifest.json"),
        "environment_abi_sha256": file_hash(output / "environment-abi.json"),
        "lowering_report_sha256": file_hash(validation / "report.json"),
        "device_execution": "NOT_RUN",
    }
    (bundle / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(Path(__file__).parent / "install_offline.py", bundle / "install_offline.py")
    shutil.copy2(output / "wheel-manifest.json", bundle / "wheel-manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
