"""Independent P0 host correctness judge. It never produces device/timing evidence."""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import numpy as np

from .contracts import TaskSpec
from .tasks.registry import host_baseline_for, make_inputs, oracle_for, verify_sources


def _validate_outputs(task, case, actual, expected, inputs):
    errors = []
    required = {tensor.name for tensor in task.outputs}
    if not isinstance(actual, dict) or set(actual) != required:
        return ["OUTPUT_SET_MISMATCH"]
    sizes = case.sizes()
    for tensor in task.outputs:
        value = actual[tensor.name]
        shape = tuple(sizes[axis] for axis in tensor.shape)
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.dtype(tensor.dtype):
            errors.append(f"OUTPUT_ABI:{tensor.name}")
            continue
        if any(np.shares_memory(value, v) for v in inputs.values()):
            errors.append(f"OUTPUT_ALIASES_INPUT:{tensor.name}")
        if not value.flags.c_contiguous:
            errors.append(f"OUTPUT_LAYOUT:{tensor.name}")
        if not np.isfinite(value).all():
            errors.append(f"OUTPUT_NONFINITE:{tensor.name}")
        elif not np.allclose(
            value,
            expected[tensor.name],
            rtol=task.accuracy.rtol,
            atol=task.accuracy.atol,
            equal_nan=task.accuracy.equal_nan,
        ):
            errors.append(f"NUMERICAL_MISMATCH:{tensor.name}")
    return errors


def check_candidate(task: TaskSpec, candidate) -> dict:
    """Trusted in-process CPU callable check; process isolation is a P1 responsibility.

    Each fixed case is tested twice on the same objects (including an in-place
    input change), then with distinct same-shape input objects. Inputs may not
    be mutated. Oracle buffers are disjoint from candidate-owned storage.
    """
    verify_sources(task)
    oracle = oracle_for(task)
    results = []
    for case in task.cases:
        inputs = make_inputs(task, case)
        invocations = []
        for phase in ("initial", "mutated_inputs", "new_inputs"):
            if phase == "mutated_inputs":
                changed = make_inputs(task, case, variant=1)
                for name in inputs:
                    inputs[name][...] = changed[name]
            elif phase == "new_inputs":
                inputs = make_inputs(task, case, variant=2)
            pristine = {k: v.copy() for k, v in inputs.items()}
            original_inputs = dict(inputs)
            oracle_inputs = {k: v.copy() for k, v in pristine.items()}
            try:
                expected = oracle(oracle_inputs, task.epsilon)
                oracle_errors = _validate_outputs(task, case, expected, expected, oracle_inputs)
                if oracle_errors:
                    raise ValueError(f"invalid oracle output: {oracle_errors}")
            except Exception as exc:  # noqa: BLE001 - Preserve arbitrary oracle failures as reference diagnostics.
                invocations.append({"phase": phase, "status": "ERROR", "errors": [f"REFERENCE_ERROR:{exc}"]})
                break
            try:
                actual = candidate(inputs, task.epsilon)
                errors = _validate_outputs(task, case, actual, expected, original_inputs)
                if set(inputs) != set(pristine) or any(
                    inputs.get(k) is not original_inputs[k]
                    or original_inputs[k].dtype != pristine[k].dtype
                    or original_inputs[k].shape != pristine[k].shape
                    or not original_inputs[k].flags.writeable
                    or not original_inputs[k].flags.c_contiguous
                    or not np.array_equal(original_inputs[k], pristine[k], equal_nan=True)
                    for k in pristine
                ):
                    errors.append("INPUT_MUTATION")
                    # A failed candidate must not poison the next reference invocation.
                    inputs = {k: v.copy() for k, v in pristine.items()}
                invocations.append({"phase": phase, "status": "FAIL" if errors else "PASS", "errors": errors})
            except Exception as exc:
                from .budget import BudgetExhausted

                if isinstance(exc, BudgetExhausted):
                    raise
                invocations.append({"phase": phase, "status": "FAIL", "errors": [f"CANDIDATE_ERROR:{exc}"]})
                inputs = {k: v.copy() for k, v in pristine.items()}
        status = (
            "ERROR"
            if any(x["status"] == "ERROR" for x in invocations)
            else ("PASS" if len(invocations) == 3 and all(x["status"] == "PASS" for x in invocations) else "FAIL")
        )
        results.append({"case_id": case.case_id, "status": status, "invocations": invocations})
    status = (
        "ERROR"
        if any(x["status"] == "ERROR" for x in results)
        else ("PASS" if all(x["status"] == "PASS" for x in results) else "FAIL")
    )
    return {
        "schema_version": 1,
        "task_id": task.task_id,
        "task_digest": task.task_digest,
        "task_snapshot": task.to_dict(),
        "host_environment": {"python": platform.python_version(), "numpy": np.__version__},
        "judge_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "host_judge_selfcheck",
        "execution_backend": "numpy_cpu",
        "status": status,
        "expected_cases": len(task.cases),
        "completed_cases": len(results),
        "cases": results,
        "evidence": {
            "lowering": "NOT_RUN",
            "vendor_build": "NOT_RUN",
            "device_correctness": "NOT_RUN",
            "measurement": "NOT_RUN",
        },
        "eligible_for_promotion": False,
    }


def selfcheck(task: TaskSpec) -> dict:
    return check_candidate(task, host_baseline_for(task))
