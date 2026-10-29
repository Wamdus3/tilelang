"""Offline, build-free installation of the locally prepared personal Conda environment.

Compatible with the A5 system Python 3.8. Does not install packages, touch devices,
modify shell profiles, or rebuild TileLang. Existing prefixes are never replaced.
"""

import argparse
import hashlib
import json
import os
import platform
import posixpath
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_member(member):
    name = member.name
    if not name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
        raise ValueError("unsafe archive member: " + name)
    if member.isdev() or member.isfifo():
        raise ValueError("unsupported special archive file: " + name)
    if member.issym() or member.islnk():
        base = posixpath.dirname(name) if member.issym() else ""
        target = posixpath.normpath(posixpath.join(base, member.linkname))
        if PurePosixPath(member.linkname).is_absolute() or target == ".." or target.startswith("../"):
            raise ValueError("archive link escapes environment: " + name)


def install(bundle, workspace):
    bundle, workspace = bundle.resolve(), workspace.resolve()
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "READY_FOR_OFFLINE_INSTALL":
        raise ValueError("bundle is not a verified offline environment")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("environment requires Linux x86_64")
    libc, version = platform.libc_ver()
    minimum = tuple(map(int, manifest["minimum_glibc"].split(".")))
    if libc != "glibc" or tuple(map(int, version.split("."))) < minimum:
        raise ValueError("host glibc is older than the audited environment requirement")
    archive = bundle / "conda-env.tar.gz"
    if file_hash(archive) != manifest["archive_sha256"]:
        raise ValueError("environment archive checksum mismatch")
    if not workspace.is_dir() or workspace.stat().st_uid != os.getuid():
        raise ValueError("workspace must exist and belong to the current user")
    prefix = workspace / "envs" / "tilelang-950"
    if os.path.commonpath((str(workspace), str(prefix.resolve()))) != str(workspace):
        raise ValueError("environment prefix escapes workspace")
    if prefix.exists():
        raise FileExistsError("environment already exists; refusing to replace " + str(prefix))
    activation = workspace / "activate-tilelang-950.sh"
    if activation.exists():
        raise FileExistsError("activation script already exists; refusing to replace it")
    # Validate all paths and links before creating the target environment.
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            validate_member(member)
        prefix.mkdir(parents=True)
        (prefix / ".install-incomplete.json").write_text(json.dumps(manifest, indent=2))
        source.extractall(prefix)
    env = os.environ.copy()
    env.update(
        PATH=str(prefix / "bin") + os.pathsep + env.get("PATH", ""),
        PYTHONNOUSERSITE="1",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    # Prefix relocation only, with no package resolver, network, or source build.
    with (workspace / "conda-unpack.log").open("w") as log:
        subprocess.run(
            [str(prefix / "bin/python"), str(prefix / "bin/conda-unpack")],
            cwd=workspace,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=True,
        )
    script = f"""# Source this file in your shell; no global shell profiles are modified.
source '{prefix}/bin/activate'
if [ -r /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -r /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh ]; then
    source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh
fi
export LD_LIBRARY_PATH='{prefix}/lib'${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PIP_NO_INDEX=1
export PIP_CACHE_DIR='{workspace}/pip-cache'
export XDG_CACHE_HOME='{workspace}/cache'
export TORCH_EXTENSIONS_DIR='{workspace}/cache/torch_extensions'
export ASCEND_PROCESS_LOG_PATH='{workspace}/logs/ascend'
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$ASCEND_PROCESS_LOG_PATH"
"""
    activation.write_text(script)
    report = {
        "schema_version": 1,
        "status": "INSTALLED",
        "prefix": str(prefix),
        "activation": str(activation),
        "archive_sha256": manifest["archive_sha256"],
        "tilelang_build_location": "local_wsl",
        "device_execution": "NOT_RUN",
    }
    (workspace / "environment-install.json").write_text(json.dumps(report, indent=2) + "\n")
    (prefix / ".install-incomplete.json").unlink()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.bundle, args.workspace), indent=2))


if __name__ == "__main__":
    main()
