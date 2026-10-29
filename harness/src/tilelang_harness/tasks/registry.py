"""Task loading is offline and checks content identity without importing candidates."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np

from ..contracts import CaseSpec, ContractError, SourceIdentity, TaskSpec
from . import host_baselines, oracles, ssd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PACKAGE_ROOT / "fixtures"
TASK_NAMES = (
    "vecadd",
    "rmsnorm",
    "fft_c2c",
    "fft_c2c_complex64",
    "fft_c2c_complex128",
    "ssd_chunk_state_fwd_float16_plain",
    "ssd_chunk_state_fwd_bfloat16_plain",
    "ssd_chunk_state_fwd_float16_seq",
    "ssd_chunk_state_fwd_bfloat16_seq",
)
LOADED_SOURCE_HASHES = {
    f"tasks/{module.__name__.rsplit('.', 1)[-1]}.py": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    for module in (host_baselines, oracles, ssd)
}


def source_path(source: SourceIdentity, root: Path = PACKAGE_ROOT) -> Path:
    root = root.resolve()
    path = (root / source.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ContractError(f"source outside package or missing: {source.path}")
    return path


def verify_sources(task: TaskSpec, root: Path = PACKAGE_ROOT) -> None:
    for source in (task.oracle, task.host_baseline, task.device_baseline):
        digest = hashlib.sha256(source_path(source, root).read_bytes()).hexdigest()
        if digest != source.sha256:
            raise ContractError(f"source digest mismatch: {source.path}")
        if source.path in LOADED_SOURCE_HASHES and LOADED_SOURCE_HASHES[source.path] != source.sha256:
            raise ContractError(f"loaded implementation differs from source: restart the process: {source.path}")
    if task.operator == "ssd_chunk_state_fwd":
        if (task.oracle.path, task.oracle.entrypoint) != ("tasks/ssd.py", "oracle") or (
            task.host_baseline.path,
            task.host_baseline.entrypoint,
        ) != ("tasks/ssd.py", "host_baseline"):
            raise ContractError("unsupported SSD reference binding")
        if (task.device_baseline.path, task.device_baseline.entrypoint) != (
            "baselines/ssd_chunk_state_fwd.py",
            "build",
        ):
            raise ContractError("unsupported SSD baseline binding")
        return
    # Oracles are selected from a closed, trusted registry, not arbitrary task-provided imports.
    for name, identity in (("oracles", task.oracle), ("host_baselines", task.host_baseline)):
        if identity.path != f"tasks/{name}.py" or identity.entrypoint != task.operator:
            raise ContractError(f"unsupported {name} binding")
    if task.device_baseline.path != f"baselines/{task.operator}.py" or task.device_baseline.entrypoint != "build":
        raise ContractError("unsupported device baseline binding")


def load_task(path_or_id: str | Path) -> TaskSpec:
    value = Path(path_or_id)
    if str(path_or_id) in TASK_NAMES:
        value = FIXTURES / f"{path_or_id}.json"
    try:
        data = json.loads(
            value.read_text(encoding="utf-8"), parse_constant=_invalid_constant, object_pairs_hook=_unique_object
        )
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot read task {value}: {exc}") from exc
    task = TaskSpec.from_dict(data)
    verify_sources(task)
    return task


def _invalid_constant(value):
    raise ContractError(f"nonfinite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def make_inputs(task: TaskSpec, case: CaseSpec, variant: int = 0) -> dict[str, np.ndarray]:
    if task.operator == "ssd_chunk_state_fwd":
        return ssd.make_inputs(task, case, variant)
    rng = np.random.default_rng(case.seed + variant)
    sizes = case.sizes()
    inputs = {}
    for index, tensor in enumerate(task.inputs):
        dtype = np.dtype(tensor.dtype)
        shape = tuple(sizes[axis] for axis in tensor.shape)
        if case.pattern == "zeros":
            values = np.zeros(shape, dtype=dtype)
        elif case.pattern == "alternating":
            values = np.where(np.arange(np.prod(shape)).reshape(shape) % 2, -1.0, 1.0)
        else:
            scale = {"normal": 1.0, "large": 1e3, "tiny": 1e-6}[case.pattern]
            values = rng.normal(0.0, scale, shape)
        values = np.asarray(values, dtype=dtype)
        if variant:
            # Deterministic change even for constant patterns and N=1; detects stale-result reuse.
            values = values * dtype.type(-0.75) + dtype.type((index + 1) * 0.25)
        inputs[tensor.name] = np.ascontiguousarray(values)
    return inputs


def oracle_for(task: TaskSpec):
    verify_sources(task)
    if task.operator == "ssd_chunk_state_fwd":
        return partial(ssd.oracle, has_seq_idx=".seq." in task.task_id)
    return getattr(oracles, task.operator)


def host_baseline_for(task: TaskSpec):
    verify_sources(task)
    if task.operator == "ssd_chunk_state_fwd":
        return partial(ssd.host_baseline, has_seq_idx=".seq." in task.task_id)
    return getattr(host_baselines, task.operator)
