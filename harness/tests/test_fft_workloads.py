"""Exact upstream workload coverage, true FP64 ABI and long host references."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tilelang_harness.bundle import parse_manifest
from tilelang_harness.contracts import ContractError, TaskSpec
from tilelang_harness.judge import check_candidate
from tilelang_harness.tasks import host_baselines, oracles
from tilelang_harness.tasks.registry import load_task, make_inputs

FIXTURES = Path(__file__).parents[1] / "src/tilelang_harness/fixtures"


def test_exact_upstream_shape_dtype_pairs_and_provenance():
    catalog = json.loads((FIXTURES / "fft_c2c.tileops.json").read_text())
    assert (
        hashlib.sha256((FIXTURES / "fft_c2c.tileops.yaml").read_bytes()).hexdigest() == catalog["upstream_spec_sha256"]
    )
    expected = [
        ([], 4096, "complex64"),
        ([64], 4096, "complex64"),
        ([64], 4096, "complex128"),
        ([64], 16384, "complex64"),
        ([8], 65536, "complex64"),
        ([256], 1024, "complex64"),
        ([4096], 512, "complex64"),
        ([65536], 64, "complex64"),
        ([4], 1048576, "complex128"),
        ([300], 1024, "complex64"),
    ]
    actual = [(w["original_batch"], w["dimensions"]["N"], w["logical_dtype"]) for w in catalog["workloads"]]
    assert actual == expected
    for name, count, dtype in [("fft_c2c_complex64", 8, "float32"), ("fft_c2c_complex128", 2, "float64")]:
        task = load_task(name)
        assert TaskSpec.from_dict(task.to_dict()) == task
        assert len(task.cases) == count
        assert {t.dtype for t in task.inputs + task.outputs} == {dtype}
        assert task.invocation.input_domain == f"finite_{dtype}"
        mapping = {w["case_id"]: w for w in catalog["workloads"] if w["task"] == name}
        assert set(mapping) == {c.case_id for c in task.cases}
        for case in task.cases:
            assert case.sizes() == mapping[case.case_id]["dimensions"]
            assert mapping[case.case_id]["logical_shape"] == mapping[case.case_id]["original_batch"] + [
                case.sizes()["N"]
            ]
    assert load_task("fft_c2c").task_id == "fft_c2c.complex64.pilot.v1"
    assert len(load_task("fft_c2c").cases) == 8


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("length", [8, 8192, 65536])
def test_long_independent_radix2_reference_preserves_dtype_and_matches_numpy(dtype, length):
    rng = np.random.default_rng(2468)
    inputs = {name: rng.normal(size=(2, length)).astype(dtype) for name in ("real", "imag")}
    original = {name: array.copy() for name, array in inputs.items()}
    reference = oracles.fft_c2c(inputs, 1e-6)
    actual = host_baselines.fft_c2c(inputs, 1e-6)
    for name in actual:
        assert actual[name].dtype == np.dtype(dtype)
        assert actual[name].flags.c_contiguous
        np.testing.assert_allclose(
            actual[name],
            reference[name],
            atol=1e-10 if dtype == "float64" else 1e-4,
            rtol=1e-10 if dtype == "float64" else 1e-4,
        )
    for name, value in inputs.items():
        np.testing.assert_array_equal(value, original[name])


def test_double_reference_does_not_round_inputs_to_float32():
    real = np.array([[1 + 2**-40, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float64)
    inputs = {"real": real, "imag": np.zeros_like(real)}
    for function in (oracles.fft_c2c, host_baselines.fft_c2c):
        result = function(inputs, 1e-6)
        assert result["out_real"][0, 0] == 2 + 2**-40
        assert result["out_real"].dtype == np.float64


def test_double_judge_rejects_downcast_even_if_returned_in_float64():
    task = load_task("fft_c2c_complex128")
    case = replace(task.cases[0], dimensions=(("B", 2), ("N", 8)))
    task = replace(task, cases=(case,))

    def lossy(inputs, epsilon):
        rounded = {k: v.astype(np.float32) for k, v in inputs.items()}
        return {k: v.astype(np.float64) for k, v in oracles.fft_c2c(rounded, epsilon).items()}

    assert check_candidate(task, host_baselines.fft_c2c)["status"] == "PASS"
    report = check_candidate(task, lossy)
    assert report["status"] == "FAIL"
    assert any("NUMERICAL_MISMATCH" in error for call in report["cases"][0]["invocations"] for error in call["errors"])


def test_double_inputs_are_reproducible_and_retain_double_precision():
    task = load_task("fft_c2c_complex128")
    case = replace(task.cases[0], dimensions=(("B", 1), ("N", 64)))
    initial, repeat, changed = [make_inputs(task, case, variant) for variant in (0, 0, 1)]
    for name in initial:
        assert initial[name].dtype == np.float64
        np.testing.assert_array_equal(initial[name], repeat[name])
        assert not np.array_equal(initial[name], changed[name])
        assert np.any(initial[name] != initial[name].astype(np.float32).astype(np.float64))


@pytest.mark.parametrize("fault", ["dtype", "invocation", "too_large", "non_power_two"])
def test_full_workload_contract_rejects_invalid_precision_or_resource_bounds(fault):
    data = load_task("fft_c2c_complex128").to_dict()
    if fault == "dtype":
        data["inputs"][0]["dtype"] = "float32"
    elif fault == "invocation":
        data["invocation"]["input_domain"] = "finite_float32"
    elif fault == "too_large":
        data["cases"][-1]["dimensions"]["B"] = 5
    else:
        data["cases"][0]["dimensions"]["N"] = 4095
    with pytest.raises(ContractError):
        TaskSpec.from_dict(data)


def test_dtype_cannot_be_overridden_by_candidate_config():
    with pytest.raises(ContractError, match="task ABI"):
        parse_manifest(
            {"schema_version": 1, "abi": "primfunc", "entrypoint": "solution.py:build", "config": {"dtype": "float32"}}
        )
