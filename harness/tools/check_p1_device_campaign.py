"""One explicitly authorized P1 vecadd case, three correctness calls, no timing."""

import argparse
from pathlib import Path

from tilelang_harness.artifacts import write_json
from tilelang_harness.bundle import freeze, seed
from tilelang_harness.runner import run
from tilelang_harness.store import Campaign
from tilelang_harness.tasks.registry import load_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--allow-resident-processes", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    task = load_task("vecadd")
    seed(task, args.output / "source")
    campaign = Campaign.create(args.output / "campaign", task, max_attempts=1, device_launches=3)
    bundle = freeze(args.output / "source", task, campaign.objects)
    campaign.add_candidate(bundle)
    result = run(
        campaign,
        bundle["candidate_id"],
        "device-smoke",
        timeout_s=180,
        case_id="n257_normal",
        device_index=args.device_index,
        max_launches=3,
        allow_resident_processes=args.allow_resident_processes,
    )
    report = campaign.report()
    write_json(args.output / "report.json", report)
    passed = (
        result["status"] == "PASS"
        and result["resources"]["used"] == {"launches": 3, "vendor_builds": 1}
        and result["evidence"]["device_correctness"] == "PASS"
        and result["evidence"]["measurement"] == "NOT_RUN"
    )
    summary = {
        "status": "PASS" if passed else result["status"],
        "scope": "p1_single_case_device_campaign",
        "task": task.task_id,
        "case_id": "n257_normal",
        "physical_device_index": args.device_index,
        "attempt_id": result["attempt_id"],
        "resources": result["resources"],
        "evidence": result["evidence"],
        "category": result["category"],
        "eligible_for_promotion": report["eligible_for_promotion"],
    }
    write_json(args.output / "validation.json", summary)
    print(__import__("json").dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
