"""Explicit single-case NPU smoke adapter. Hardware validation is still required."""

import importlib.util
import os
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from .artifacts import write_json
from .contracts import ContractError, content_id
from .judge import check_candidate
from .tasks.registry import source_path


def lease_directory():
    # Serialize all NPU jobs of this user, including different visibility mappings/campaigns.
    root = Path(tempfile.gettempdir()) / f"tilelang-harness-{os.getuid()}"
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise ContractError("unsafe device lease directory")
    return root


def quarantine(reason, attempt_id):
    write_json(
        lease_directory() / "quarantine.json",
        {
            "reason": reason,
            "attempt_id": attempt_id,
            "recovery": "operator must verify NPU health before explicitly removing this marker",
        },
    )


@contextmanager
def lease():
    import fcntl

    root = lease_directory()
    with (root / "npu.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("DEVICE_BUSY: another harness NPU worker holds the lease") from exc
        if (root / "quarantine.json").exists():
            raise ContractError("DEVICE_QUARANTINED: health confirmation required")
        # Close rather than LOCK_UN: a surviving worker inherits this open description,
        # and must retain the lease if its watchdog is killed unexpectedly.
        yield stream.fileno()


def environment(index):
    import torch
    import torch_npu

    if not torch.npu.is_available() or not 0 <= index < torch.npu.device_count():
        raise RuntimeError(f"NPU {index} is unavailable")
    torch.npu.set_device(index)
    torch.npu.synchronize()
    return torch, {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": f"npu:{index}",
        "name": torch.npu.get_device_name(index),
        "visibility": {k: os.environ.get(k) for k in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")},
    }


class Invocation:
    """Map the fixed numpy judge ABI to the SAME prepared NPU handle on every call."""

    def __init__(self, task, kernel, torch, index, budget=None):
        self.task, self.kernel, self.torch = task, kernel, torch
        self.device = torch.device(f"npu:{index}")
        self.owners, self.tensors = None, None
        self.budget = budget

    def prepare(self, inputs):
        owners = tuple(inputs[t.name] for t in self.task.inputs)
        if self.owners is None or any(a is not b for a, b in zip(self.owners, owners)):
            self.owners = owners
            self.tensors = [self.torch.from_numpy(value.copy()).to(self.device) for value in owners]
        else:
            for tensor, value in zip(self.tensors, owners):
                tensor.copy_(self.torch.from_numpy(value.copy()))
        self.torch.npu.synchronize()

    def call(self):
        if self.budget is not None:
            self.budget.consume("launches")
        return self.kernel(*self.tensors)

    def __call__(self, inputs, epsilon):
        self.prepare(inputs)
        outputs = self.call()
        self.torch.npu.synchronize()
        if isinstance(outputs, self.torch.Tensor):
            outputs = [outputs]
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(self.task.outputs):
            raise ValueError("device output set mismatch")
        for value in outputs:
            if not isinstance(value, self.torch.Tensor) or value.device != self.device or not value.is_contiguous():
                raise ValueError("device output ABI/layout mismatch")
            if any(value.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in self.tensors):
                raise ValueError("device output aliases an input")
        # Copy back inputs too: otherwise CPU copies would conceal illegal device mutation.
        for spec, owner, tensor in zip(self.task.inputs, self.owners, self.tensors):
            if (
                tuple(tensor.shape) != owner.shape
                or tensor.dtype != getattr(self.torch, spec.dtype)
                or tensor.device != self.device
                or not tensor.is_contiguous()
            ):
                raise ValueError("device input ABI mutation")
            owner[...] = tensor.cpu().numpy()
        return {spec.name: value.cpu().numpy().copy() for spec, value in zip(self.task.outputs, outputs)}


def trusted_builder(task):
    spec = importlib.util.spec_from_file_location("_tlh_device_baseline", source_path(task.device_baseline))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, task.device_baseline.entrypoint)


def compile_handle(task, prim):
    import tilelang
    from tilelang.backend.target import determine_target

    options = {
        "target": "ascend",
        "execution_backend": "auto",
        "out_idx": list(range(len(task.inputs), len(task.inputs) + len(task.outputs))),
        "pass_configs": {},
        "compile_flags": [],
    }
    with determine_target("ascend", return_object=True):
        kernel = tilelang.compile(prim, **options)
    return kernel, {**options, "resolved_execution_backend": kernel.execution_backend}


def verify_one(task, case, invocation):
    # The subset is explicit. The enclosing result retains the original task identity and coverage.
    report = check_candidate(replace(task, cases=(case,)), invocation)
    report["scope"] = "single_case_device_correctness"
    report["original_task_digest"] = task.task_digest
    return report


def compiled_identity(build, kernel, options, destination):
    from .artifacts import digest

    data = kernel.get_kernel_source().encode("utf-8")
    (destination / "compiled-kernel.cpp").write_bytes(data)
    body = {
        "lowering_build_id": build["build_id"],
        "options": options,
        "compiled_source": digest(data),
        "binary": None,
        "binary_note": "live handle is worker-local; binary reload is not assumed",
    }
    return {"build_id": content_id(body), **body}
