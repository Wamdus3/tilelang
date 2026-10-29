"""Freeze an explicit candidate directory without importing candidate Python."""

import math
from pathlib import Path, PurePosixPath

from .artifacts import ArtifactStore, digest, read_json, write_json
from .contracts import ContractError, content_id, fields
from .tasks.registry import source_path, verify_sources

MAX_BUNDLE_BYTES = 16 * 1024 * 1024


def relative_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not PurePosixPath(value).parts
        or "\\" in value
        or ":" in value
        or PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
        or str(PurePosixPath(value)) != value
    ):
        raise ContractError("bundle paths must be normalized relative POSIX paths")
    return value


def parse_manifest(value):
    fields(value, {"schema_version", "abi", "entrypoint", "config"}, "candidate manifest")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ContractError("unsupported candidate schema")
    if value["abi"] not in ("numpy", "primfunc"):
        raise ContractError("candidate ABI must be numpy or primfunc")
    expected = "solution.py:run" if value["abi"] == "numpy" else "solution.py:build"
    if value["entrypoint"] != expected:
        raise ContractError(f"explicit entrypoint required: {expected}")
    if not isinstance(value["config"], dict):
        raise ContractError("builder config must be a JSON object")
    for key, item in value["config"].items():
        if not key.isidentifier() or key in (
            "B",
            "M",
            "N",
            "epsilon",
            "dtype",
            "L",
            "NC",
            "Q",
            "H",
            "P",
            "G",
            "dt_dtype",
            "has_seq_idx",
        ):
            raise ContractError(f"config cannot override the task ABI: {key}")
        if type(item) not in (bool, int, float, str) or (isinstance(item, str) and len(item) > 1024):
            raise ContractError("builder config values must be scalars")
        if type(item) is float and not math.isfinite(item):
            raise ContractError("builder config must be finite")
    if value["abi"] == "numpy" and value["config"]:
        raise ContractError("numpy diagnostic ABI has no builder config")
    content_id(value)  # Also reject nonfinite JSON numbers.
    return value


def freeze(source: Path, task, objects: ArtifactStore) -> dict:
    if source.is_symlink() or not source.is_dir():
        raise ContractError("candidate must be a real directory")
    manifest = parse_manifest(read_json(source / "candidate.json"))
    files, names, total = {}, set(), 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"candidate symlink is forbidden: {path.name}")
        name = relative_name(path.relative_to(source).as_posix())
        if any(part in (".git", "__pycache__", ".pytest_cache") for part in path.relative_to(source).parts):
            continue
        if not path.is_file():
            if not path.is_dir():
                raise ContractError(f"unsupported bundle entry: {name}")
            continue
        if name.casefold() in names:
            raise ContractError(f"case-insensitive bundle path collision: {name}")
        names.add(name.casefold())
        data = path.read_bytes()
        total += len(data)
        if total > MAX_BUNDLE_BYTES or len(files) >= 512:
            raise ContractError("candidate bundle exceeds the size/file budget")
        files[name] = objects.put(data)
    if "solution.py" not in files:
        raise ContractError("candidate requires solution.py")
    # Bind the manifest parsed from the same frozen bytes, preventing a racing edit.
    import json

    if parse_manifest(json.loads(objects.get(files["candidate.json"]))) != manifest:
        raise ContractError("candidate changed during freezing")
    body = {"schema_version": 1, "task_digest": task.task_digest, "manifest": manifest, "files": files}
    return {"candidate_id": content_id(body), **body}


def materialize(bundle: dict, destination: Path, objects: ArtifactStore):
    body = {k: v for k, v in bundle.items() if k != "candidate_id"}
    if content_id(body) != bundle["candidate_id"]:
        raise ContractError("candidate identity mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    for name, key in bundle["files"].items():
        path = destination / relative_name(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(objects.get(key))


def verify_materialized(bundle: dict, root: Path):
    for name, key in bundle["files"].items():
        path = root / relative_name(name)
        if path.is_symlink() or not path.is_file() or digest(path.read_bytes()) != key:
            raise ContractError(f"frozen candidate drift: {name}")


def seed(task, destination: Path, abi: str = "primfunc"):
    verify_sources(task)
    destination.mkdir(parents=True, exist_ok=False)
    if abi == "primfunc":
        data = source_path(task.device_baseline).read_bytes()
    else:
        # Copy the trusted host implementation, but keep its source registry untouched.
        data = source_path(task.host_baseline).read_bytes()
        if task.operator == "ssd_chunk_state_fwd":
            masked = ".seq." in task.task_id
            data += (
                f"\ndef run(inputs, epsilon):\n    return host_baseline(inputs, epsilon, has_seq_idx={masked!r})\n"
            ).encode()
        else:
            data += f"\nrun = {task.operator}\n".encode()
    (destination / "solution.py").write_bytes(data)
    write_json(
        destination / "candidate.json",
        {
            "schema_version": 1,
            "abi": abi,
            "entrypoint": "solution.py:build" if abi == "primfunc" else "solution.py:run",
            "config": {},
        },
    )
