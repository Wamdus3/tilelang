"""Exercise frozen candidates, real lowering, replay and authoritative report export in local WSL."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "harness" / "src"), str(REPO)]

from tilelang_harness.artifacts import write_json
from tilelang_harness.bundle import freeze, seed
from tilelang_harness.runner import replay, resume, run
from tilelang_harness.store import Campaign
from tilelang_harness.tasks.registry import load_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="fresh directory, never an existing campaign")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in ("vecadd", "rmsnorm"):
        task = load_task(name)
        source = args.output / f"{name}-source"
        seed(task, source)
        campaign = Campaign.create(args.output / name, task, max_attempts=3)
        bundle = freeze(source, task, campaign.objects)
        campaign.add_candidate(bundle)
        first = run(campaign, bundle["candidate_id"], "lower", timeout_s=300)
        if first["status"] != "PASS":
            raise RuntimeError(first)
        second = replay(campaign, first["attempt_id"])
        if second["status"] != "PASS" or second["evaluation_id"] == first["evaluation_id"]:
            raise RuntimeError(second)
        assert resume(campaign) == []
        report = campaign.report()
        write_json(args.output / f"{name}-report.json", report)
        entry = {
            "task": name,
            "status": "PASS",
            "case_count": len(task.cases),
            "candidate_id": bundle["candidate_id"],
            "attempt_ids": [first["attempt_id"], second["attempt_id"]],
            "build_sets_match": first["builds"] == second["builds"],
            "evidence": first["evidence"],
            "eligible_for_promotion": False,
        }
        results.append(entry)
        print(f"{name}: real lowering + independent replay PASS ({len(task.cases)} cases)", flush=True)
    write_json(
        args.output / "report.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "scope": "local_wsl_campaign_lowering_and_replay",
            "results": results,
            "eligible_for_promotion": False,
        },
    )


if __name__ == "__main__":
    main()
