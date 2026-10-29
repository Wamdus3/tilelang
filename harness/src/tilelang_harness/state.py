"""Deterministic evidence reducer. No model participates in state transitions."""

from itertools import pairwise

from .contracts import ContractError, content_id

LAYERS = ("lowering", "vendor_build", "device_correctness", "measurement")


def reduce_result(job, result):
    if not isinstance(result, dict):
        raise ContractError("worker result must be an object")
    if (
        not isinstance(result.get("builds"), dict)
        or not isinstance(result.get("cases"), list)
        or not isinstance(result.get("artifacts"), dict)
    ):
        raise ContractError("invalid worker result collections")
    for key in (
        "job_id",
        "attempt_id",
        "candidate_id",
        "task_digest",
        "mode",
        "case_ids",
        "runner_digest",
        "device_index",
        "measure",
    ):
        if result.get(key) != job[key]:
            raise ContractError(f"worker result identity mismatch: {key}")
    if (
        type(result.get("schema_version")) is not int
        or result["schema_version"] != 1
        or result.get("status") not in ("PASS", "FAIL", "ERROR", "INCOMPLETE")
    ):
        raise ContractError("invalid worker result status/schema")
    evidence = result.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(LAYERS):
        raise ContractError("invalid evidence layers")
    if any(value not in ("PASS", "FAIL", "ERROR", "NOT_RUN") for value in evidence.values()):
        raise ContractError("invalid evidence status")
    phase = "FROZEN"
    if "resource_limits" in job:
        accounting = result.get("resources", {})
        if accounting.get("limits") != job["resource_limits"]:
            raise ContractError("resource limit identity mismatch")
        used = accounting.get("used", {})
        if set(used) != set(job["resource_limits"]) or any(
            type(value) is not int or not 0 <= value <= job["resource_limits"][key] for key, value in used.items()
        ):
            raise ContractError("invalid resource accounting")
        if result["status"] == "PASS" and accounting.get("accounting_complete") is not True:
            raise ContractError("PASS requires complete resource accounting")
        if (
            result["status"] == "PASS"
            and job["mode"] == "device-smoke"
            and not job["measure"]
            and used != {"launches": 3, "vendor_builds": 1}
        ):
            raise ContractError("device PASS requires exactly three correctness dispatches and one build")
    if job["mode"] == "host":
        if any(value != "NOT_RUN" for value in evidence.values()):
            raise ContractError("host diagnostics cannot provide device evidence")
    elif job["mode"] == "lower":
        if any(evidence[key] != "NOT_RUN" for key in LAYERS[1:]):
            raise ContractError("lowering cannot provide vendor/device evidence")
        if evidence["lowering"] == "PASS":
            phase = "LOWERED"
    elif job["mode"] == "device-smoke":
        if len(job["case_ids"]) != 1:
            raise ContractError("device smoke must select exactly one case")
        if not job["measure"] and evidence["measurement"] != "NOT_RUN":
            raise ContractError("measurement was not requested")
        for previous, layer in pairwise(LAYERS):
            if evidence[layer] == "PASS" and evidence[previous] != "PASS":
                raise ContractError("device evidence requires previous layers to pass")
        # A smoke attempt never grants full workload coverage or promotion.
    else:
        raise ContractError("unsupported execution mode")
    expected = job["case_ids"]
    for case_id, build in result.get("builds", {}).items():
        if (
            case_id not in expected
            or build.get("build_id") != content_id({k: v for k, v in build.items() if k != "build_id"})
            or build.get("candidate_id") != job["candidate_id"]
            or build.get("case_id") != case_id
        ):
            raise ContractError("invalid per-case BuildID")
    for case_id, build in result.get("compiled_builds", {}).items():
        if (
            case_id not in result.get("builds", {})
            or build.get("build_id") != content_id({k: v for k, v in build.items() if k != "build_id"})
            or build.get("lowering_build_id") != result["builds"][case_id]["build_id"]
        ):
            raise ContractError("invalid compiled BuildID")
    if job["mode"] == "device-smoke" and evidence["measurement"] == "PASS":
        from .evaluation import summarize_samples

        measurement = result.get("measurement", {})
        protocol = job["task"]["measurement"]
        if (
            measurement.get("protocol") != protocol
            or measurement.get("unit") != "ms"
            or measurement.get("backend") != "event"
            or measurement.get("case_id") != expected[0]
            or measurement.get("device") != f"npu:{job['device_index']}"
        ):
            raise ContractError("measurement protocol/environment mismatch")
        for key in ("baseline", "candidate"):
            summary = measurement.get(key, {})
            if summary != summarize_samples(summary.get("samples_ms", []), protocol["independent_runs"]):
                raise ContractError("invalid measurement summary")
        if "resource_limits" in job:
            import statistics

            if measurement.get("implementation") != "bounded_single_call_events_v1":
                raise ContractError("unrecognized measurement implementation")
            orders = [
                ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
                for index in range(protocol["independent_runs"])
            ]
            if [list(order) for order in measurement.get("orders", [])] != orders:
                raise ContractError("invalid paired measurement order")
            details = measurement.get("call_details", {})
            if set(details) != {"baseline", "candidate"}:
                raise ContractError("missing raw event measurements")
            launches = 12  # baseline/candidate correctness before and after timing, three each.
            for key in details:
                if len(details[key]) != protocol["independent_runs"]:
                    raise ContractError("incomplete event measurement rounds")
                for index, sample in enumerate(details[key]):
                    values = sample.get("call_samples_ms", [])
                    if not values:
                        raise ContractError("empty raw event measurements")
                    summarize_samples(values, len(values))
                    warmups = sample.get("warmup_calls")
                    if (
                        not values
                        or type(warmups) is not int
                        or warmups < 0
                        or (
                            sample.get("mean_ms") != statistics.mean(values)
                            or sample["mean_ms"] != measurement[key]["samples_ms"][index]
                        )
                    ):
                        raise ContractError("invalid raw event measurement")
                    launches += warmups + len(values)
            if result["resources"]["used"] != {"launches": launches, "vendor_builds": 2}:
                raise ContractError("measurement dispatch accounting mismatch")
    if result["status"] == "PASS":
        cases = result.get("cases", [])
        if [case.get("case_id") for case in cases] != expected or any(case.get("status") != "PASS" for case in cases):
            raise ContractError("PASS requires complete, ordered case coverage")
        if job["mode"] in ("lower", "device-smoke") and (
            evidence["lowering"] != "PASS" or set(result.get("builds", {})) != set(expected)
        ):
            raise ContractError("lowering PASS requires a BuildID for every case")
        if job["mode"] == "device-smoke":
            if evidence["vendor_build"] != "PASS" or evidence["device_correctness"] != "PASS":
                raise ContractError("device smoke PASS requires compiled, correct output")
            if set(result.get("compiled_builds", {})) != set(expected):
                raise ContractError("device smoke requires compiled BuildID coverage")
            if job["measure"] and evidence["measurement"] != "PASS":
                raise ContractError("requested measurement is missing")
    return {
        "phase": phase,
        "outcome": result["status"],
        "evidence": evidence,
        "eligible_for_promotion": False,
        "promotion_reason": "NPU build, correctness, measurement and independent review remain required",
    }
