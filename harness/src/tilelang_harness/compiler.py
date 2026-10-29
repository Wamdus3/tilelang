"""CompilerBridge for real Ascend source lowering, with explicit build identities."""

import importlib.metadata
import os
import platform
import shutil
import sys
from pathlib import Path

from .artifacts import digest
from .contracts import FFT_WORKLOAD_TASKS, content_id


def runner_identity():
    root = Path(__file__).parent
    return {
        "python": platform.python_version(),
        "executable": digest(Path(sys.executable).read_bytes()),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy",)},
        "harness": {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in sorted(root.rglob("*.py"))},
    }


def compiler_identity(tilelang):
    root = Path(tilelang.__file__).resolve().parent
    source = {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in sorted(root.rglob("*.py"))}
    dependencies = {}
    for name in ("tvm", "tvm_ffi"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            directory = Path(module.__file__).resolve().parent
            dependencies[name] = {
                p.relative_to(directory).as_posix(): digest(p.read_bytes()) for p in sorted(directory.rglob("*.py"))
            }
    # Include native compiler sources for development checkouts and the actual mapped libraries.
    native_source = {}
    for directory in (root.parent / "src", root.parent / "include", root / "src", root / "include"):
        if directory.is_dir():
            for p in sorted(directory.rglob("*")):
                if p.is_file() and p.suffix in (".cc", ".cpp", ".h", ".cuh", ".cu"):
                    native_source[p.relative_to(root.parent).as_posix()] = digest(p.read_bytes())
    native = {}
    maps = Path("/proc/self/maps")
    if maps.exists():
        for line in maps.read_text().splitlines():
            path = line.split()[-1]
            if path.startswith("/") and any(name in Path(path).name for name in ("libtilelang", "libtvm", "libascend")):
                p = Path(path)
                if p.is_file():
                    native[str(p)] = digest(p.read_bytes())
    toolchain = {}
    for name in ("bisheng", "ld.lld", "ccec"):
        preferred = Path(os.environ.get("BISHENG_HOME", "")) / "bin" / name
        path = str(preferred) if os.environ.get("BISHENG_HOME") and preferred.is_file() else shutil.which(name)
        toolchain[name] = {"path": path, "sha256": digest(Path(path).read_bytes())} if path else None
    sdk_root = Path(os.environ["ASCEND_HOME_PATH"]) if os.environ.get("ASCEND_HOME_PATH") else None
    sdk = {}
    if sdk_root and sdk_root.is_dir():
        headers = sdk_root / "asc" / "include"
        if headers.is_dir():
            sdk["asc_headers"] = {
                p.relative_to(headers).as_posix(): digest(p.read_bytes())
                for p in sorted(headers.rglob("*.h"))
                if p.is_file()
            }
        info = sdk_root / "x86_64-linux" / "ascend_toolkit_install.info"
        if info.is_file():
            sdk["toolkit_info"] = digest(info.read_bytes())
    return {
        "tilelang_version": str(getattr(tilelang, "__version__", "unknown")),
        "python_dependencies": dependencies,
        "vendor_toolchain_on_path": toolchain,
        "sdk": sdk,
        "python_sources": source,
        "native_sources": native_source,
        "loaded_native": native,
        "environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("TILELANG_", "TVM_", "ASCEND_"))},
    }


def lower_case(task, case, bundle, builder, destination, compiler_pack, stage, return_prim=False):
    import tilelang
    from tilelang.backend.target import determine_target

    kwargs = {**bundle["manifest"]["config"], **case.sizes()}
    if task.operator == "rmsnorm":
        kwargs["epsilon"] = task.epsilon
    if task.task_id in FFT_WORKLOAD_TASKS:
        kwargs["dtype"] = task.inputs[0].dtype
    if task.operator == "ssd_chunk_state_fwd":
        kwargs.pop("L")
        kwargs.update(dtype=task.inputs[0].dtype, dt_dtype=task.inputs[2].dtype, has_seq_idx=".seq." in task.task_id)
    options = {"target": "ascend", "enable_host_codegen": False, "enable_device_compile": False}
    with determine_target("ascend", return_object=True):
        stage("builder")
        prim = builder(**kwargs)
        stage("lowering")
        artifact = tilelang.lower(prim, **options)
    stage("artifact_export")
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {
        "input.tir": prim.script(),
        "lowered.tir": artifact.device_mod.script(),
        "kernel.cpp": artifact.kernel_source,
    }
    hashes = {}
    for name, value in outputs.items():
        data = value.encode("utf-8")
        (destination / name).write_bytes(data)
        hashes[name] = digest(data)
    identity = {
        "schema_version": 1,
        "candidate_id": bundle["candidate_id"],
        "case_id": case.case_id,
        "builder_kwargs": kwargs,
        "options": options,
        "compiler_pack": content_id(compiler_pack),
        "artifacts": hashes,
        "binary": None,
        "allocation_audit": "unknown",
    }
    build = {"build_id": content_id(identity), **identity}
    return (build, prim) if return_prim else build
