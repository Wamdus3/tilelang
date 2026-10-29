"""Environment import verification only: never creates tensors or runs NPU operators."""

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--include-npu", action="store_true", help="import installed torch_npu with existing CANN")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    report = {
        "schema_version": 1,
        "scope": "environment_imports_only",
        "device_execution": "NOT_RUN",
        "prefix": str(prefix),
        "python": platform.python_version(),
        "libc": platform.libc_ver(),
    }
    try:
        if Path(sys.prefix).resolve() != prefix:
            raise ValueError("wrong Python interpreter prefix")
        modules = ["numpy", "torch"]
        if args.include_npu:
            modules.append("torch_npu")
        modules.extend(["torch_c_dlpack_ext", "tilelang", "tilelang_harness"])
        origins = {}
        for name in modules:
            module = importlib.import_module(name)
            origin = Path(module.__file__).resolve()
            if not origin.is_relative_to(prefix):
                raise ValueError("module loaded outside personal environment: " + str(origin))
            origins[name] = str(origin)
        import tilelang.ascend.language as T
        from tilelang.backend.target import determine_target

        if not callable(T.SimtVF):
            raise TypeError("Ascend SimtVF API unavailable")
        with determine_target("ascend", return_object=True):
            pass
        mapped = []
        for line in Path("/proc/self/maps").read_text().splitlines():
            path = line.split()[-1]
            if path.startswith("/") and any(
                part in Path(path).name for part in ("libtilelang", "libtvm_", "libtorch", "libc10", "libz3")
            ):
                actual = Path(path).resolve()
                if not actual.is_relative_to(prefix):
                    raise ValueError("native library loaded outside personal environment: " + str(actual))
                mapped.append(str(actual))
        report.update(
            status="PASS",
            module_origins=origins,
            native_libraries=sorted(set(mapped)),
            torch_npu_import="PASS" if args.include_npu else "NOT_RUN",
            packages={
                name: importlib.metadata.version(name)
                for name in ("tilelang", "tilelang-harness", "numpy", "torch", "torch-npu", "apache-tvm-ffi")
            },
        )
    except Exception as exc:  # noqa: BLE001 - Report arbitrary dependency import failures.
        report.update(status="ERROR", error=str(exc), traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
