"""Read ELF version requirements without importing/executing wheel or environment code."""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def requirements(path):
    result = subprocess.run(["readelf", "-W", "--version-info", str(path)], text=True, capture_output=True, check=True)
    needs = result.stdout.partition("Version needs section")[2]
    versions = sorted(set(re.findall(r"Name: (GLIBC_[A-Za-z0-9_.]+)", needs)))
    return versions


def audit_file(path, name, maximum):
    versions = requirements(path)
    incompatible = []
    for version in versions:
        suffix = version.removeprefix("GLIBC_")
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", suffix) or tuple(map(int, suffix.split("."))) > maximum:
            incompatible.append(version)
    return {"file": name, "glibc_requirements": versions, "incompatible": incompatible}


def audit(paths, maximum):
    records = []
    with tempfile.TemporaryDirectory(prefix="tlh-abi-") as temporary:
        elf = Path(temporary) / "artifact.elf"
        for source in paths:
            files = sorted(source.rglob("*")) if source.is_dir() else [source]
            for path in files:
                if not path.is_file() or path.is_symlink():
                    continue
                if path.suffix == ".whl":
                    with zipfile.ZipFile(path) as wheel:
                        for entry in wheel.infolist():
                            if entry.is_dir():
                                continue
                            with wheel.open(entry) as stream:
                                header = stream.read(4)
                                if header != b"\x7fELF":
                                    continue
                                with elf.open("wb") as target:
                                    target.write(header)
                                    shutil.copyfileobj(stream, target)
                            records.append(audit_file(elf, f"{path.name}!{entry.filename}", maximum))
                else:
                    with path.open("rb") as stream:
                        header = stream.read(4)
                    if header == b"\x7fELF":
                        records.append(audit_file(path, str(path), maximum))
    if not records:
        raise ValueError("no ELF artifacts found; nothing was verified")
    return {
        "schema_version": 1,
        "scope": "elf_glibc_requirements_only",
        "max_glibc": ".".join(map(str, maximum)),
        "status": "INCOMPATIBLE" if any(item["incompatible"] for item in records) else "PASS",
        "elf_count": len(records),
        "artifacts": records,
        "device_execution": "NOT_RUN",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-glibc", default="2.31")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.paths, tuple(map(int, args.max_glibc.split("."))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "artifacts"}))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
