"""Paired complete-call NPU event measurements; never synthesize missing samples."""

import math
import statistics
import time
from dataclasses import asdict

from .contracts import ContractError
from .tasks.registry import make_inputs


def summarize_samples(samples, expected):
    if len(samples) != expected or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
        raise ContractError("measurement requires all finite positive millisecond samples")
    median = statistics.median(samples)
    spread = (max(samples) - min(samples)) / median
    return {"samples_ms": samples, "median_ms": median, "relative_range": spread}


def event_sample(invocation, warmup_ms, repeat_ms, clock=time.monotonic):
    """Reserve before recording events, keeping durable quota I/O out of the sample."""

    def prepare_call():
        if invocation.budget is not None:
            invocation.budget.consume("launches")
        return lambda: invocation.kernel(*invocation.tensors)

    warmups = 0
    until = clock() + warmup_ms / 1000
    while clock() < until:
        prepare_call()()
        invocation.torch.npu.synchronize()
        warmups += 1
    values = []
    until = clock() + repeat_ms / 1000
    while not values or clock() < until:
        call = prepare_call()
        start = invocation.torch.npu.Event(enable_timing=True)
        end = invocation.torch.npu.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        summarize_samples([elapsed], 1)
        values.append(elapsed)
    return {"mean_ms": statistics.mean(values), "call_samples_ms": values, "warmup_calls": warmups}


def measure_pair(task, case, baseline, candidate):

    protocol = task.measurement
    samples = {"baseline": [], "candidate": []}
    invocations = {"baseline": baseline, "candidate": candidate}
    orders = []
    details = {"baseline": [], "candidate": []}
    for index in range(protocol.independent_runs):
        inputs = make_inputs(task, case, variant=index)
        for invocation in invocations.values():
            invocation.prepare(inputs)
        order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        orders.append(order)
        for name in order:
            invocation = invocations[name]
            sample = event_sample(invocation, protocol.warmup_ms, protocol.repeat_ms)
            details[name].append(sample)
            samples[name].append(sample["mean_ms"])
    summaries = {key: summarize_samples(value, protocol.independent_runs) for key, value in samples.items()}
    return {
        "case_id": case.case_id,
        "protocol": asdict(protocol),
        "backend": "event",
        "device": str(candidate.device),
        "unit": "ms",
        "orders": orders,
        "implementation": "bounded_single_call_events_v1",
        "call_details": details,
        "time_budget_unit": "host_wall_ms",
        "cache_policy": "no explicit cache flush",
        **summaries,
        "case_speedup": summaries["baseline"]["median_ms"] / summaries["candidate"]["median_ms"],
        "high_noise": any(x["relative_range"] > 0.10 for x in summaries.values()),
        "workload_score": None,
        "eligible_for_promotion": False,
    }
