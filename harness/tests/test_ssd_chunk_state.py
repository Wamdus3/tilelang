import copy

import numpy as np
import pytest

from tilelang_harness.bundle import seed
from tilelang_harness.contracts import ContractError, TaskSpec
from tilelang_harness.tasks.registry import host_baseline_for, load_task, make_inputs, oracle_for
from tilelang_harness.tasks.ssd import host_baseline, oracle


def tiny_task():
    task = load_task("ssd_chunk_state_fwd_float16_seq").to_dict()
    task["cases"] = [
        {
            "case_id": "edge",
            "dimensions": {"B": 1, "NC": 3, "Q": 4, "L": 12, "H": 4, "G": 2, "P": 2, "N": 3},
            "seed": 42,
            "pattern": "normal",
            "weight": 1,
        }
    ]
    return TaskSpec.from_dict(task)


def test_ssd_group_mask_clamp_and_reference_independence():
    task = tiny_task()
    inputs = make_inputs(task, task.cases[0])
    inputs["x"].fill(1)
    inputs["Bmat"][:, :, 0].fill(2)
    inputs["Bmat"][:, :, 1].fill(3)
    inputs["dt"].fill(1)
    inputs["dA"].fill(0)
    inputs["dA"][..., -1] = 1  # Positive differences must clamp to zero.
    inputs["seq_idx"][:] = [-1, -1, -1, -1, 0, 1, 1, 1, 0, 0, 0, -1]
    expected = np.zeros((1, 3, 4, 2, 3), np.float32)
    expected[:, 1, :2] = 6
    expected[:, 1, 2:] = 9
    for reference in (oracle, host_baseline):
        actual = reference(inputs, 1e-6)["states"]
        np.testing.assert_array_equal(actual, expected)


def test_ssd_dtypes_and_workload_coverage():
    names = ("float16_plain", "bfloat16_plain", "float16_seq", "bfloat16_seq")
    cases = []
    for name in names:
        task = load_task(f"ssd_chunk_state_fwd_{name}")
        cases.extend(case.case_id for case in task.cases)
        assert task.accuracy.rtol == 1e-3
        assert task.accuracy.atol == (1.6e-2 if name.startswith("bfloat16") else 1e-3)
    assert len(cases) == 6 and len(set(cases)) == 6
    task = tiny_task().to_dict()
    task["task_id"] = "ssd_chunk_state_fwd.bfloat16.seq.tileops.v1"
    task["inputs"][0]["dtype"] = task["inputs"][1]["dtype"] = "bfloat16"
    parsed = TaskSpec.from_dict(task)
    inputs = make_inputs(parsed, parsed.cases[0])
    assert np.all((inputs["x"].view(np.uint32) & 0xFFFF) == 0)


def test_ssd_rejects_invalid_shape_and_widened_accuracy():
    source = tiny_task().to_dict()
    invalid = copy.deepcopy(source)
    invalid["cases"][0]["dimensions"]["L"] += 1
    with pytest.raises(ContractError, match="L=NC"):
        TaskSpec.from_dict(invalid)
    invalid = copy.deepcopy(source)
    invalid["accuracy"]["atol"] = 1.6e-2
    with pytest.raises(ContractError, match="tolerances exceed"):
        TaskSpec.from_dict(invalid)


@pytest.mark.parametrize("mode", ["seq", "plain"])
def test_ssd_host_seed_entrypoint(tmp_path, mode):
    task_dict = tiny_task().to_dict()
    task_dict["task_id"] = f"ssd_chunk_state_fwd.float16.{mode}.tileops.v1"
    task = TaskSpec.from_dict(task_dict)
    destination = tmp_path / "seed"
    seed(task, destination, "numpy")
    namespace = {}
    exec((destination / "solution.py").read_text(), namespace)  # noqa: S102 - Exercise the trusted generated host seed.
    inputs = make_inputs(task, task.cases[0])
    inputs["seq_idx"].fill(-1)
    expected = oracle_for(task)(inputs, task.epsilon)["states"]
    np.testing.assert_allclose(
        namespace["run"](inputs, task.epsilon)["states"], expected, rtol=task.accuracy.rtol, atol=task.accuracy.atol
    )


def test_ssd_plain_ignores_arbitrary_dummy_sequence_values():
    task_dict = tiny_task().to_dict()
    task_dict["task_id"] = "ssd_chunk_state_fwd.float16.plain.tileops.v1"
    task = TaskSpec.from_dict(task_dict)
    inputs = make_inputs(task, task.cases[0])
    inputs["x"].fill(1)
    inputs["Bmat"].fill(2)
    inputs["dt"].fill(1)
    inputs["dA"].fill(0)
    expected = np.full((1, 3, 4, 2, 3), 8, dtype=np.float32)
    for dummy in (
        np.full((1, 12), -1, np.int32),
        np.array([[-2147483648, 7, -3, -1, 0, 1, 2, 3, 4, 5, 6, 2147483647]], dtype=np.int32),
    ):
        inputs["seq_idx"][...] = dummy
        for reference in (oracle_for(task), host_baseline_for(task)):
            np.testing.assert_array_equal(reference(inputs, task.epsilon)["states"], expected)
