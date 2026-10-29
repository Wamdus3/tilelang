"""P0 acceptance tests: reject broken judges/contracts rather than testing plumbing alone."""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tilelang_harness.contracts import ContractError, TaskSpec
from tilelang_harness.judge import check_candidate, selfcheck
from tilelang_harness.tasks import oracles
from tilelang_harness.tasks.registry import host_baseline_for, load_task, make_inputs, verify_sources


@pytest.mark.parametrize("name", ["vecadd", "rmsnorm"])
def test_host_baseline_all_cases_and_input_lifecycle(name):
    task = load_task(name)
    result = selfcheck(task)
    assert result["status"] == "PASS", result
    assert result["completed_cases"] == result["expected_cases"] == len(task.cases)
    assert all(len(case["invocations"]) == 3 for case in result["cases"])
    assert set(result["evidence"].values()) == {"NOT_RUN"}
    assert not result["eligible_for_promotion"]
    assert TaskSpec.from_dict(task.to_dict()) == task


def test_oracles_have_analytical_sanity_cases():
    inputs = {"x": np.array([[3.0, 4.0]], dtype=np.float32), "weight": np.array([2.0, -1.0], dtype=np.float32)}
    expected_scale = 1 / math.sqrt(12.5 + 1e-6)
    result = oracles.rmsnorm(inputs, 1e-6)
    np.testing.assert_allclose(result["rstd"], [expected_scale], rtol=1e-7)
    np.testing.assert_allclose(result["out"], [[6 * expected_scale, -4 * expected_scale]], rtol=1e-7)
    added = oracles.vecadd({"a": np.array([1, -2], np.float32), "b": np.array([4, 2], np.float32)}, 1e-6)
    np.testing.assert_array_equal(added["out"], [5, 0])


@pytest.mark.parametrize("fault", ["zero", "tail", "dtype", "nan", "none", "shape", "alias", "mutation"])
def test_vecadd_fault_injection(fault):
    task = load_task("vecadd")
    baseline = host_baseline_for(task)

    def broken(inputs, epsilon):
        result = baseline(inputs, epsilon)
        if fault == "zero":
            result["out"].fill(0)
        elif fault == "tail":
            result["out"][-1] = 12345
        elif fault == "dtype":
            result["out"] = result["out"].astype(np.float64)
        elif fault == "nan":
            result["out"][-1] = np.nan
        elif fault == "none":
            return {"out": None}
        elif fault == "shape":
            result["out"] = result["out"].reshape(-1, 1)
        elif fault == "alias":
            return {"out": inputs["a"]}
        elif fault == "mutation":
            inputs["a"].fill(0)
        return result

    result = check_candidate(task, broken)
    assert result["status"] == "FAIL", result
    assert not result["eligible_for_promotion"]


@pytest.mark.parametrize("fault", ["missing", "incorrect"])
def test_rmsnorm_requires_auxiliary_output(fault):
    task = load_task("rmsnorm")
    baseline = host_baseline_for(task)

    def broken(inputs, epsilon):
        result = baseline(inputs, epsilon)
        if fault == "missing":
            del result["rstd"]
        else:
            result["rstd"] = np.zeros_like(result["rstd"])
        return result

    assert check_candidate(task, broken)["status"] == "FAIL"


def test_output_layout_must_match_the_frozen_contract():
    task = load_task("rmsnorm")
    baseline = host_baseline_for(task)

    def wrong_layout(inputs, epsilon):
        result = baseline(inputs, epsilon)
        result["out"] = np.asfortranarray(result["out"])
        return result

    result = check_candidate(task, wrong_layout)
    assert result["status"] == "FAIL"
    assert any("OUTPUT_LAYOUT:out" in call["errors"] for case in result["cases"] for call in case["invocations"])


def test_rejects_pointer_cached_and_same_shape_stale_results():
    task = load_task("vecadd")
    baseline = host_baseline_for(task)
    cache = {}

    def stale(inputs, epsilon):
        key = inputs["a"].shape
        if key not in cache:
            cache[key] = baseline(inputs, epsilon)
        return cache[key]

    result = check_candidate(task, stale)
    assert result["status"] == "FAIL"
    assert all(case["invocations"][0]["status"] == "PASS" for case in result["cases"])
    assert all(case["invocations"][1]["status"] == "FAIL" for case in result["cases"])
    assert all(case["invocations"][2]["status"] == "FAIL" for case in result["cases"])


def test_reference_error_is_not_a_candidate_pass(monkeypatch):
    task = load_task("vecadd")

    def broken_reference(*args):
        raise RuntimeError("oracle unavailable")

    monkeypatch.setattr(oracles, "vecadd", broken_reference)
    result = selfcheck(task)
    assert result["status"] == "ERROR"
    assert not result["eligible_for_promotion"]


@pytest.mark.parametrize("fault", ["mapping", "replace", "readonly"])
def test_candidate_input_corruption_does_not_poison_the_reference(fault):
    task = load_task("vecadd")
    baseline = host_baseline_for(task)

    def broken(inputs, epsilon):
        result = baseline(inputs, epsilon)
        if fault == "mapping":
            inputs.clear()
        elif fault == "replace":
            original = inputs["a"]
            inputs["a"] = original.copy()
            original.fill(0)
        else:
            inputs["a"].flags.writeable = False
        return result

    result = check_candidate(task, broken)
    assert result["status"] == "FAIL"
    assert all(case["status"] == "FAIL" for case in result["cases"])
    assert not any(
        "REFERENCE_ERROR" in error
        for case in result["cases"]
        for invocation in case["invocations"]
        for error in invocation["errors"]
    )


def test_cli_distinguishes_reference_errors(monkeypatch, capsys):
    from tilelang_harness import cli

    monkeypatch.setattr(cli, "selfcheck", lambda task: {"status": "ERROR"})
    assert cli.main(["selfcheck", "vecadd"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("target", "cuda"),
        ("cases", []),
        ("epsilon", float("nan")),
        ("epsilon", 0),
        ("unknown", "ignored?"),
    ],
)
def test_strict_root_schema(field, value):
    data = load_task("vecadd").to_dict()
    data[field] = value
    with pytest.raises(ContractError):
        TaskSpec.from_dict(data)


@pytest.mark.parametrize(
    "change",
    [
        "missing_output",
        "duplicate_case",
        "dimension",
        "bool_dimension",
        "dtype",
        "protocol",
        "unit",
        "mutation",
        "source_path",
        "loose_tolerance",
    ],
)
def test_rejects_inconsistent_contracts(change):
    data = load_task("rmsnorm").to_dict()
    if change == "missing_output":
        data["outputs"].pop()
    elif change == "duplicate_case":
        data["cases"].append(copy.deepcopy(data["cases"][0]))
    elif change == "dimension":
        data["cases"][0]["dimensions"]["K"] = 7
    elif change == "bool_dimension":
        data["cases"][0]["dimensions"]["N"] = True
    elif change == "dtype":
        data["outputs"][0]["dtype"] = "float16"
    elif change == "protocol":
        data["measurement"]["boundary"] = "fastest_kernel"
    elif change == "unit":
        data["measurement"]["repeat"] = data["measurement"].pop("repeat_ms")
    elif change == "mutation":
        data["invocation"]["input_mutation"] = "allowed"
    elif change == "source_path":
        data["oracle"]["path"] = "../outside.py"
    elif change == "loose_tolerance":
        data["accuracy"]["rtol"] = 1.0
    with pytest.raises(ContractError):
        TaskSpec.from_dict(data)


def test_source_drift_and_unregistered_reference_are_rejected(tmp_path):
    task = load_task("vecadd")
    from tilelang_harness.tasks.registry import PACKAGE_ROOT

    for source in (task.oracle, task.host_baseline, task.device_baseline):
        path = tmp_path / source.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((PACKAGE_ROOT / source.path).read_bytes())
    verify_sources(task, tmp_path)
    (tmp_path / task.oracle.path).write_text("# changed oracle", encoding="utf-8")
    with pytest.raises(ContractError, match="digest mismatch"):
        verify_sources(task, tmp_path)


def test_identity_binds_cases_accuracy_protocol_and_source():
    task = load_task("vecadd")
    original = task.to_dict()
    reordered = dict(reversed(list(original.items())))
    assert TaskSpec.from_dict(reordered).task_digest == task.task_digest
    for field in ("cases", "accuracy", "measurement", "device_baseline"):
        data = copy.deepcopy(original)
        if field == "cases":
            data[field][0]["seed"] += 1
        elif field == "accuracy":
            data[field]["atol"] /= 2
        elif field == "measurement":
            data[field]["repeat_ms"] += 1
        else:
            data[field]["sha256"] = "0" * 64
        assert TaskSpec.from_dict(data).task_digest != task.task_digest


def test_deterministic_inputs_and_variants():
    task = load_task("rmsnorm")
    for case in task.cases:
        first = make_inputs(task, case)
        repeated = make_inputs(task, case)
        changed = make_inputs(task, case, 1)
        for key in first:
            np.testing.assert_array_equal(first[key], repeated[key])
            assert first[key].flags.c_contiguous
            assert not np.array_equal(first[key], changed[key])


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate JSON key"):
        load_task(path)


def test_nonfinite_and_out_of_range_json_numbers_rejected(tmp_path):
    task = load_task("vecadd").to_dict()
    for invalid in (float("inf"), float("nan"), 10**500):
        task["epsilon"] = invalid
        with pytest.raises(ContractError):
            TaskSpec.from_dict(task)
    path = tmp_path / "nonfinite.json"
    path.write_text('{"epsilon": NaN}', encoding="utf-8")
    with pytest.raises(ContractError, match="nonfinite JSON"):
        load_task(path)


def test_refreezing_source_cannot_relabel_an_already_loaded_module(tmp_path):
    import hashlib

    from tilelang_harness.tasks.registry import PACKAGE_ROOT

    task = load_task("vecadd")
    for source in (task.oracle, task.host_baseline, task.device_baseline):
        path = tmp_path / source.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((PACKAGE_ROOT / source.path).read_bytes())
    path = tmp_path / task.oracle.path
    path.write_bytes(path.read_bytes() + b"\n# A different oracle version\n")
    data = task.to_dict()
    data["oracle"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ContractError, match="loaded implementation differs"):
        verify_sources(TaskSpec.from_dict(data), tmp_path)


def test_cli_offline_from_unrelated_directory(tmp_path):
    # CLI is exercised through the public module entry point; no TileLang import is needed.
    import os

    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "OPENBLAS_NUM_THREADS": "2"}
    output = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, "-m", "tilelang_harness", "selfcheck", "all", "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,  # Includes the six full SSD workloads and their three input lifecycles.
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(output.read_text())
    assert report == json.loads(result.stdout)
    assert report["status"] == "PASS"
    from tilelang_harness.tasks.registry import TASK_NAMES

    assert len(report["results"]) == len(TASK_NAMES)
    assert {row["task_id"] for row in report["results"]} == {load_task(name).task_id for name in TASK_NAMES}
    bad = subprocess.run(
        [sys.executable, "-m", "tilelang_harness", "validate", "missing.json"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert bad.returncode == 2
    assert json.loads(bad.stdout)["status"] == "ERROR"
