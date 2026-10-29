"""Versioned, closed task contracts. JSON input is validated before execution."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

FFT_WORKLOAD_TASKS = {
    "fft_c2c.complex64.tileops.v1": "float32",
    "fft_c2c.complex128.tileops.v1": "float64",
}
SSD_TASK_PREFIX = "ssd_chunk_state_fwd."


class ContractError(ValueError):
    """Invalid or unsupported experiment contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def fields(value: Any, expected: set[str], where: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ContractError(f"{where}: missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}")
    return value


def text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where}: expected nonempty string")
    return value


def integer(value: Any, where: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{where}: expected integer >= {minimum}")
    return value


def number(value: Any, where: str, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise ContractError(f"{where}: expected finite number >= {minimum}")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ContractError(f"{where}: number outside float64 range") from exc
    if not math.isfinite(result) or result < minimum:
        raise ContractError(f"{where}: expected finite number >= {minimum}")
    return result


def sequence(value: Any, where: str) -> list:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{where}: expected nonempty list")
    return value


@dataclass(frozen=True)
class SourceIdentity:
    path: str
    sha256: str
    entrypoint: str

    @classmethod
    def parse(cls, data: Any) -> SourceIdentity:
        fields(data, {"path", "sha256", "entrypoint"}, "source")
        path = text(data["path"], "source.path")
        parts = PurePosixPath(path)
        if parts.is_absolute() or ".." in parts.parts or "\\" in path or ":" in path:
            raise ContractError("source.path: expected relative POSIX path without traversal")
        digest = text(data["sha256"], "source.sha256")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ContractError("source.sha256: expected lowercase SHA256")
        entrypoint = text(data["entrypoint"], "source.entrypoint")
        if not entrypoint.isidentifier():
            raise ContractError("source.entrypoint: expected a function name")
        return cls(path, digest, entrypoint)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[str, ...]
    dtype: str

    @classmethod
    def parse(cls, data: Any) -> TensorSpec:
        fields(data, {"name", "shape", "dtype"}, "tensor")
        name = text(data["name"], "tensor.name")
        shape = tuple(text(x, "tensor.shape") for x in sequence(data["shape"], "tensor.shape"))
        if any(not x.isidentifier() for x in shape) or not name.isidentifier():
            raise ContractError("tensor: names and dimensions must be identifiers")
        if data["dtype"] not in ("float16", "bfloat16", "float32", "float64", "int32"):
            raise ContractError("unsupported tensor dtype")
        return cls(name, shape, data["dtype"])


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    dimensions: tuple[tuple[str, int], ...]
    seed: int
    pattern: str
    weight: float

    def sizes(self) -> dict[str, int]:
        return dict(self.dimensions)

    @classmethod
    def parse(cls, data: Any, axes: set[str], max_elements: int = 1_000_000) -> CaseSpec:
        fields(data, {"case_id", "dimensions", "seed", "pattern", "weight"}, "case")
        dims = fields(data["dimensions"], axes, "case.dimensions")
        dimensions = tuple((k, integer(v, f"dimension.{k}")) for k, v in sorted(dims.items()))
        # P0 bounds prevent malformed offline input from allocating arbitrary amounts of memory.
        if math.prod(v for _, v in dimensions) > max_elements:
            raise ContractError(f"case exceeds the {max_elements}-element limit")
        pattern = data["pattern"]
        if pattern not in ("normal", "zeros", "alternating", "large", "tiny"):
            raise ContractError("unsupported input pattern")
        return cls(
            text(data["case_id"], "case.case_id"),
            dimensions,
            integer(data["seed"], "case.seed", 0),
            pattern,
            number(data["weight"], "case.weight", 1e-12),
        )


@dataclass(frozen=True)
class InvocationContract:
    input_mutation: str
    output_alias: str
    layout: str
    input_domain: str
    reference_access: str

    @classmethod
    def parse(cls, data: Any, dtype: str = "float32") -> InvocationContract:
        expected = {
            "input_mutation": "forbidden",
            "output_alias": "forbidden",
            "layout": "c_contiguous",
            "input_domain": f"finite_{dtype}",
            "reference_access": "allowed",
        }
        fields(data, set(expected), "invocation")
        if data != expected:
            raise ContractError("unsupported P0 invocation contract")
        return cls(**data)


@dataclass(frozen=True)
class AccuracySpec:
    rtol: float
    atol: float
    equal_nan: bool

    @classmethod
    def parse(cls, data: Any, max_atol: float = 1e-3) -> AccuracySpec:
        fields(data, {"rtol", "atol", "equal_nan"}, "accuracy")
        if data["equal_nan"] is not False:
            raise ContractError("P0 finite-input tasks require equal_nan=false")
        rtol = number(data["rtol"], "accuracy.rtol")
        atol = number(data["atol"], "accuracy.atol")
        if rtol > 1e-3 or atol > max_atol:
            raise ContractError("tolerances exceed the registered task limit")
        return cls(rtol, atol, False)


@dataclass(frozen=True)
class MeasurementProtocol:
    timer: str
    boundary: str
    warmup_ms: float
    repeat_ms: float
    independent_runs: int
    aggregation: str
    input_reset: str

    @classmethod
    def parse(cls, data: Any) -> MeasurementProtocol:
        fields(
            data,
            {"timer", "boundary", "warmup_ms", "repeat_ms", "independent_runs", "aggregation", "input_reset"},
            "measurement",
        )
        if data["timer"] != "npu_event" or data["boundary"] != "complete_callable":
            raise ContractError("P0 declares npu_event timing of the complete callable")
        if data["aggregation"] != "weighted_geomean_speedup" or data["input_reset"] != "fresh_inputs_per_run":
            raise ContractError("unsupported aggregation or reset policy")
        return cls(
            data["timer"],
            data["boundary"],
            number(data["warmup_ms"], "warmup_ms"),
            number(data["repeat_ms"], "repeat_ms", 0.001),
            integer(data["independent_runs"], "independent_runs", 3),
            data["aggregation"],
            data["input_reset"],
        )


@dataclass(frozen=True)
class TaskSpec:
    schema_version: int
    task_id: str
    operator: str
    target: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    cases: tuple[CaseSpec, ...]
    epsilon: float
    invocation: InvocationContract
    accuracy: AccuracySpec
    measurement: MeasurementProtocol
    oracle: SourceIdentity
    host_baseline: SourceIdentity
    device_baseline: SourceIdentity

    @classmethod
    def from_dict(cls, data: Any) -> TaskSpec:
        fields(
            data,
            {
                "schema_version",
                "task_id",
                "operator",
                "target",
                "inputs",
                "outputs",
                "cases",
                "epsilon",
                "invocation",
                "accuracy",
                "measurement",
                "oracle",
                "host_baseline",
                "device_baseline",
            },
            "task",
        )
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ContractError("unsupported schema_version")
        if data["target"] != "ascend" or data["operator"] not in (
            "vecadd",
            "rmsnorm",
            "fft_c2c",
            "ssd_chunk_state_fwd",
        ):
            raise ContractError("unsupported ascend task")
        inputs = tuple(TensorSpec.parse(x) for x in sequence(data["inputs"], "inputs"))
        outputs = tuple(TensorSpec.parse(x) for x in sequence(data["outputs"], "outputs"))
        names = [x.name for x in inputs + outputs]
        if len(set(names)) != len(names):
            raise ContractError("tensor names must be unique")
        expected = {
            "vecadd": (("a", ("N",)), ("b", ("N",)), ("out", ("N",))),
            "rmsnorm": (("x", ("M", "N")), ("weight", ("N",)), ("out", ("M", "N")), ("rstd", ("M",))),
            "fft_c2c": (("real", ("B", "N")), ("imag", ("B", "N")), ("out_real", ("B", "N")), ("out_imag", ("B", "N"))),
            "ssd_chunk_state_fwd": (
                ("x", ("B", "L", "H", "P")),
                ("Bmat", ("B", "L", "G", "N")),
                ("dt", ("B", "H", "NC", "Q")),
                ("dA", ("B", "H", "NC", "Q")),
                ("seq_idx", ("B", "L")),
                ("states", ("B", "NC", "H", "P", "N")),
            ),
        }[data["operator"]]
        ssd = data["operator"] == "ssd_chunk_state_fwd"
        if len(inputs) != (5 if ssd else 2) or tuple((x.name, x.shape) for x in inputs + outputs) != expected:
            raise ContractError("tensor ABI does not match the registered operator")
        full_fft = data["operator"] == "fft_c2c" and data["task_id"] in FFT_WORKLOAD_TASKS
        dtype = FFT_WORKLOAD_TASKS[data["task_id"]] if full_fft else "float32"
        if ssd:
            if data["task_id"] not in {
                f"{SSD_TASK_PREFIX}{dt}.{mode}.tileops.v1"
                for dt in ("float16", "bfloat16")
                for mode in ("plain", "seq")
            }:
                raise ContractError("unsupported SSD task identity")
            component = data["task_id"].split(".")[1]
            if tuple(t.dtype for t in inputs + outputs) != (
                component,
                component,
                "float32",
                "float32",
                "int32",
                "float32",
            ):
                raise ContractError("SSD requires x/Bmat component dtype, fp32 dt/dA/states and int32 seq_idx")
            dtype = "mixed_ssd"
        elif any(tensor.dtype != dtype for tensor in inputs + outputs):
            raise ContractError(f"task requires uniform {dtype} tensor ABI")
        axes = {axis for x in inputs + outputs for axis in x.shape}
        max_elements = 10**20 if ssd else (4_194_304 if full_fft else 1_000_000)
        cases = tuple(CaseSpec.parse(x, axes, max_elements) for x in sequence(data["cases"], "cases"))
        if len({x.case_id for x in cases}) != len(cases):
            raise ContractError("duplicate case_id")
        if len(cases) > 64:
            raise ContractError("P0 supports at most 64 cases per task")
        if data["operator"] == "fft_c2c":
            max_length = 1_048_576 if full_fft else 4096
            for case in cases:
                length = case.sizes()["N"]
                if length > max_length or length & (length - 1):
                    raise ContractError(f"fft_c2c requires power-of-two N <= {max_length}")
        if ssd:
            for case in cases:
                dims = case.sizes()
                if dims["L"] != dims["NC"] * dims["Q"] or dims["H"] % dims["G"]:
                    raise ContractError("SSD requires L=NC*Q and H divisible by G")
                if max(math.prod(dims[axis] for axis in tensor.shape) for tensor in inputs + outputs) > 268_435_456:
                    raise ContractError("SSD tensor exceeds the 268435456-element limit")
        epsilon = number(data["epsilon"], "epsilon", 1e-12)
        return cls(
            1,
            text(data["task_id"], "task_id"),
            data["operator"],
            data["target"],
            inputs,
            outputs,
            cases,
            epsilon,
            InvocationContract.parse(data["invocation"], dtype),
            AccuracySpec.parse(data["accuracy"], 1.6e-2 if ssd and inputs[0].dtype == "bfloat16" else 1e-3),
            MeasurementProtocol.parse(data["measurement"]),
            *(SourceIdentity.parse(data[k]) for k in ("oracle", "host_baseline", "device_baseline")),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        for case in data["cases"]:
            case["dimensions"] = dict(case["dimensions"])
        return json.loads(canonical_json(data))

    @property
    def task_digest(self) -> str:
        return content_id(self.to_dict())
