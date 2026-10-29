"""FFT pilot ABI and independent mathematical checks; no NPU execution."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tilelang_harness.bundle import parse_manifest
from tilelang_harness.contracts import ContractError, TaskSpec
from tilelang_harness.judge import check_candidate, selfcheck
from tilelang_harness.tasks import host_baselines, oracles
from tilelang_harness.tasks.registry import TASK_NAMES, load_task, make_inputs


def direct_dft(real, imag):
    """Small O(N^2) analytical evaluator, independent of both FFT algorithms."""
    values = real.astype(np.float64) + 1j * imag.astype(np.float64)
    n = values.shape[-1]
    matrix = np.exp(-2j * np.pi * np.arange(n)[:, None] * np.arange(n)[None, :] / n)
    return values @ matrix.T


@pytest.mark.parametrize("length", [1, 2, 8, 16, 64])
def test_fft_both_references_match_direct_dft_and_preserve_inputs(length):
    rng = np.random.default_rng(739 + length)
    inputs = {name: rng.normal(size=(3, length)).astype(np.float32) for name in ("real", "imag")}
    original = {name: value.copy() for name, value in inputs.items()}
    expected = direct_dft(**inputs)
    for reference in (oracles.fft_c2c, host_baselines.fft_c2c):
        actual = reference(inputs, 1e-6)
        for key, component in (("out_real", expected.real), ("out_imag", expected.imag)):
            assert actual[key].dtype == np.float32
            assert actual[key].flags.c_contiguous
            np.testing.assert_allclose(actual[key], component, rtol=1e-5, atol=1e-5)
        for key, value in inputs.items():
            np.testing.assert_array_equal(value, original[key])


def test_fft_forward_sign_no_normalization_and_batch_flattening():
    real, imag = np.zeros((2, 3, 8), np.float32), np.zeros((2, 3, 8), np.float32)
    real[0, 0, 1] = 1
    real[1, 2, :] = 2
    imag[0, 1, 0] = 3
    inputs = {"real": real.reshape(6, 8), "imag": imag.reshape(6, 8)}
    actual = oracles.fft_c2c(inputs, 1e-6)
    assert actual["out_imag"][0, 1] == pytest.approx(-np.sqrt(0.5), abs=1e-7)
    assert actual["out_real"][5, 0] == 16  # A normalized transform would give 2.
    np.testing.assert_array_equal(actual["out_real"][5, 1:], 0)
    np.testing.assert_array_equal(actual["out_imag"][1], 3)
    unbatched = oracles.fft_c2c({"real": real[0, 0][None, :], "imag": imag[0, 0][None, :]}, 1e-6)
    np.testing.assert_array_equal(unbatched["out_real"][0], actual["out_real"][0])


def test_fft_frozen_pilot_cases_and_full_host_lifecycles():
    task = load_task("fft_c2c")
    assert task.task_id == "fft_c2c.complex64.pilot.v1"
    assert [(c.sizes()["B"], c.sizes()["N"]) for c in task.cases] == [
        (1, 1),
        (2, 8),
        (3, 64),
        (3, 512),
        (256, 1024),
        (300, 1024),
        (1, 4096),
        (64, 4096),
    ]
    assert [case.seed for case in task.cases] == list(range(20261023, 20261031))
    assert task.cases[2].pattern == "alternating"
    assert task.accuracy.rtol == task.accuracy.atol == 1e-4
    assert TaskSpec.from_dict(task.to_dict()) == task
    result = selfcheck(task)
    assert result["status"] == "PASS", result
    assert result["completed_cases"] == 8
    assert all(len(case["invocations"]) == 3 for case in result["cases"])
    assert not result["eligible_for_promotion"]
    assert set(result["evidence"].values()) == {"NOT_RUN"}


@pytest.mark.parametrize(
    "dimensions", [{"B": 1, "N": 3}, {"B": 1, "N": 8192}, {"B": 1000, "N": 1024}, {"B": True, "N": 8}]
)
def test_fft_rejects_shapes_outside_pilot(dimensions):
    data = load_task("fft_c2c").to_dict()
    data["cases"][0]["dimensions"] = dimensions
    with pytest.raises(ContractError):
        TaskSpec.from_dict(data)


@pytest.mark.parametrize("fault", ["missing_imag", "alias", "mutation", "normalized", "inverse"])
def test_fft_judge_rejects_complex_abi_and_math_faults(fault):
    task = load_task("fft_c2c")
    task = replace(task, cases=(task.cases[1],))

    def broken(inputs, epsilon):
        result = host_baselines.fft_c2c(inputs, epsilon)
        if fault == "missing_imag":
            del result["out_imag"]
        elif fault == "alias":
            result["out_real"] = inputs["real"]
        elif fault == "mutation":
            inputs["imag"].fill(0)
        elif fault == "normalized":
            result = {name: value / np.float32(8) for name, value in result.items()}
        else:
            result["out_imag"] *= -1
        return result

    assert check_candidate(task, broken)["status"] == "FAIL"


def test_fft_input_patterns_are_reproducible_and_changed():
    task = load_task("fft_c2c")
    for case in (task.cases[0], task.cases[2]):
        first, repeated, changed = [make_inputs(task, case, variant) for variant in (0, 0, 1)]
        for name in ("real", "imag"):
            np.testing.assert_array_equal(first[name], repeated[name])
            assert not np.array_equal(first[name], changed[name])


def test_fft_dimension_cannot_be_overridden_by_candidate_config():
    with pytest.raises(ContractError, match="task ABI"):
        parse_manifest({"schema_version": 1, "abi": "primfunc", "entrypoint": "solution.py:build", "config": {"B": 1}})


def test_fft_public_cli_includes_registered_pilot(capsys):
    from tilelang_harness.cli import main

    assert main(["list"]) == 0
    assert json.loads(capsys.readouterr().out)["tasks"] == list(TASK_NAMES)
    assert "fft_c2c" in TASK_NAMES
    scope = Path(__file__).parents[1] / "src/tilelang_harness/fixtures/fft_c2c.scope.json"
    assert "complex128" in json.loads(scope.read_text())["excluded"]


def test_fft_device_compile_output_indices_are_derived_from_abi(monkeypatch):
    from tilelang_harness.device import compile_handle

    calls = []
    fake = ModuleType("tilelang")
    fake.compile = lambda prim, **kwargs: calls.append(kwargs) or type("Kernel", (), {"execution_backend": "fake"})()
    target = ModuleType("tilelang.backend.target")
    from contextlib import nullcontext

    target.determine_target = lambda *args, **kwargs: nullcontext()
    monkeypatch.setitem(sys.modules, "tilelang", fake)
    monkeypatch.setitem(sys.modules, "tilelang.backend.target", target)
    compile_handle(load_task("fft_c2c"), object())
    assert calls[0]["out_idx"] == [2, 3]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("fft_c2c", {"B": 2, "N": 8}),
        ("fft_c2c_complex64", {"B": 64, "N": 4096, "dtype": "float32"}),
        ("fft_c2c_complex128", {"B": 4, "N": 1048576, "dtype": "float64"}),
    ],
)
def test_fft_builder_receives_dimensions_without_epsilon(monkeypatch, tmp_path, name, expected):
    from contextlib import nullcontext

    from tilelang_harness.compiler import lower_case

    calls = []
    fake = ModuleType("tilelang")
    scripted = type("Scripted", (), {"script": lambda self: "fake_tir"})()
    fake.lower = lambda prim, **kwargs: type("Artifact", (), {"device_mod": scripted, "kernel_source": "fake_cpp"})()
    target = ModuleType("tilelang.backend.target")
    target.determine_target = lambda *args, **kwargs: nullcontext()
    monkeypatch.setitem(sys.modules, "tilelang", fake)
    monkeypatch.setitem(sys.modules, "tilelang.backend.target", target)
    task = load_task(name)

    def builder(**kwargs):
        calls.append(kwargs)
        return scripted

    result = lower_case(
        task,
        task.cases[1],
        {"manifest": {"config": {}}, "candidate_id": "x"},
        builder,
        tmp_path / "lower",
        {},
        lambda stage: None,
    )
    assert calls == [expected]
    assert result["builder_kwargs"] == expected
