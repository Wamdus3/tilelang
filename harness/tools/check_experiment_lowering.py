"""Real local WSL integration using fixed baseline responses, NOT a model benchmark."""

import argparse
from pathlib import Path

from tilelang_harness.artifacts import write_json
from tilelang_harness.experiments import Experiment
from tilelang_harness.runner import replay
from tilelang_harness.store import Campaign
from tilelang_harness.tasks.registry import load_task, source_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    spec = {
        "schema_version": 1,
        "name": "fixed-baseline-pipeline-check-not-a-model-study",
        "tasks": ["vecadd", "rmsnorm"],
        "arms": [
            {
                "name": "linked_original_skill",
                "skill_links": [".agents/skills/tilelang-ascend/SKILL.md"],
                "skill_workflow": "links",
                "generator": {
                    "kind": "file",
                    "command": [],
                    "model": "fixed-baseline-fixture-not-a-model",
                    "parameters": {},
                    "timeout_s": 30,
                    "token_reservation": 1,
                },
            }
        ],
        "replicates": 1,
        "seed": 42,
        "max_repairs": 0,
        "evaluation_mode": "lower",
        "evaluation_timeout_s": 300,
        "budget": {"generations": 2, "reserved_tokens": 2, "evaluations": 2, "wall_seconds": 1200},
    }
    experiment = Experiment.create(args.output, spec, repo)
    for _ in range(2):
        waiting = experiment.run()
        request_dir = Path(waiting["request_directory"])
        trial = next(row for row in experiment.trials() if row["trial_id"] == waiting["trial_id"])
        task = next(load_task(name) for name in ("vecadd", "rmsnorm") if load_task(name).task_id == trial["task_id"])
        write_json(
            request_dir / "response.json",
            {
                "schema_version": 1,
                "files": {"solution.py": source_path(task.device_baseline).read_text()},
                "config": {},
                "plan": "Fixed trusted baseline; protocol integration only, no model call.",
                "usage": {"input_tokens": None, "output_tokens": None},
            },
        )
    report = experiment.run()
    assert report["status"] == "COMPLETE"
    assert all(row["result"]["status"] == "PASS" for row in report["trials"]), report
    replays = []
    for trial in report["trials"]:
        campaign = Campaign(experiment.root / "campaigns" / trial["task_id"])
        fresh = replay(campaign, trial["attempt_id"], timeout_s=300)
        assert fresh["status"] == "PASS", fresh
        original = next(
            row["result"]
            for row in report["campaigns"][trial["task_id"]]["attempts"]
            if row["attempt_id"] == trial["attempt_id"]
        )
        assert fresh["builds"] == original["builds"]
        assert fresh["attempt_id"] != original["attempt_id"]
        replays.append(
            {
                "task_id": trial["task_id"],
                "attempt_id": fresh["attempt_id"],
                "case_count": len(fresh["cases"]),
                "build_sets_match": True,
            }
        )
    write_json(experiment.root / "report.json", experiment.report())
    write_json(
        experiment.root / "validation.json",
        {
            "status": "PASS",
            "scope": "local_wsl_lowering_and_replay",
            "generator": "fixed_baseline_fixture_not_LLM",
            "device_execution": "NOT_RUN",
            "replays": replays,
        },
    )
    print("PASS: fixed responses, linked original skill, 13 case lowerings and independent replays; no NPU or LLM")


if __name__ == "__main__":
    main()
