"""Task checks and deterministic local campaigns. No remote transport."""

import argparse
import json
from pathlib import Path

from .contracts import ContractError
from .judge import selfcheck
from .tasks.registry import TASK_NAMES, load_task


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tlh")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list registered task fixtures")
    validate = commands.add_parser("validate", help="validate a task and its source hashes offline")
    validate.add_argument("task", help="registered task name or a JSON path")
    check = commands.add_parser("selfcheck", help="check host baselines against independent CPU oracles")
    check.add_argument("task", nargs="?", default="all", help="registered task name, all, or a JSON path")
    check.add_argument("--output", type=Path)
    init = commands.add_parser("init", help="create an immutable task campaign")
    init.add_argument("task")
    init.add_argument("--run-dir", required=True, type=Path)
    init.add_argument("--max-attempts", type=int, default=20)
    init.add_argument("--device-launches", type=int, default=0, help="campaign launch quota; 0 disables devices")
    seed = commands.add_parser("seed", help="copy a registered baseline into an explicit candidate bundle")
    seed.add_argument("task")
    seed.add_argument("--destination", required=True, type=Path)
    seed.add_argument("--abi", choices=("numpy", "primfunc"), default="primfunc")
    submit = commands.add_parser("submit", help="freeze all files in a candidate directory without importing it")
    submit.add_argument("source", type=Path)
    submit.add_argument("--run-dir", required=True, type=Path)
    run = commands.add_parser("run", help="run one frozen candidate in a deadline-limited Linux worker")
    run.add_argument("candidate_id")
    run.add_argument("--mode", choices=("host", "lower", "device-smoke"), default="lower")
    run.add_argument("--timeout-s", type=float, default=120)
    run.add_argument("--case", help="one required case ID for device-smoke")
    run.add_argument("--device-index", type=int, default=0)
    run.add_argument("--measure", action="store_true", help="paired NPU event samples for the selected case only")
    run.add_argument("--max-launches", type=int, default=3, help="per-job callable limit, reserved before execution")
    run.add_argument("--allow-resident-processes", action="store_true", help="allow residents only at zero utilization")
    run.add_argument("--trial-id", help="link an explicit single-case evaluation to a generation trial")
    quota = commands.add_parser("device-quota", help="explicitly set total campaign launch quota")
    quota.add_argument("--run-dir", type=Path, required=True)
    quota.add_argument("--launches", type=int, required=True)
    replay = commands.add_parser("replay", help="independently execute an old attempt using its frozen candidate")
    replay.add_argument("attempt_id")
    replay.add_argument("--timeout-s", type=float)
    resume = commands.add_parser("resume", help="reconcile recorded workers/results without launching replacements")
    report = commands.add_parser("report", help="export authoritative events and results")
    report.add_argument("--output", type=Path)
    for command in (run, replay, resume, report):
        command.add_argument("--run-dir", required=True, type=Path)
    experiment_init = commands.add_parser("experiment-init", help="freeze a generation comparison specification")
    experiment_init.add_argument("spec", type=Path)
    experiment_init.add_argument("--repo-root", type=Path, default=Path.cwd())
    experiment_init.add_argument("--run-dir", type=Path, required=True)
    for name in ("experiment-run", "experiment-resume", "experiment-report"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-dir", type=Path, required=True)
        if name == "experiment-report":
            sub.add_argument("--output", type=Path)
    skill_read = commands.add_parser(
        "skill-read", help="deliver one bounded original skill chunk and record provenance"
    )
    skill_read.add_argument("--run-dir", type=Path, required=True)
    skill_read.add_argument("--trial-id", required=True)
    skill_read.add_argument("--chunk", required=True)
    skill_status = commands.add_parser(
        "skill-status", help="show the next unread chunk without dumping the whole context"
    )
    skill_status.add_argument("--run-dir", type=Path, required=True)
    skill_status.add_argument("--trial-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in ("skill-read", "skill-status"):
            from .experiments import Experiment

            experiment = Experiment(args.run_dir)
            payload = (
                experiment.read_skill(args.trial_id, args.chunk)
                if args.command == "skill-read"
                else experiment.skill_status(args.trial_id)
            )
        elif args.command.startswith("experiment-"):
            from .artifacts import read_json
            from .experiments import Experiment

            if args.command == "experiment-init":
                experiment = Experiment.create(args.run_dir, read_json(args.spec), args.repo_root)
                payload = {"status": "CREATED", "experiment_id": experiment.report()["experiment_id"]}
            else:
                experiment = Experiment(args.run_dir)
                payload = experiment.report() if args.command == "experiment-report" else experiment.run()
        elif args.command == "list":
            payload = {"schema_version": 1, "tasks": list(TASK_NAMES)}
        elif args.command == "validate":
            task = load_task(args.task)
            payload = {"status": "VALID", "task_digest": task.task_digest, "task": task.to_dict()}
        elif args.command == "selfcheck":
            tasks = TASK_NAMES if args.task == "all" else (args.task,)
            results = [selfcheck(load_task(task)) for task in tasks]
            status = (
                "ERROR"
                if any(r["status"] == "ERROR" for r in results)
                else ("PASS" if all(r["status"] == "PASS" for r in results) else "FAIL")
            )
            payload = {
                "schema_version": 1,
                "status": status,
                "scope": "host_judge_selfcheck",
                "results": results,
            }
        elif args.command == "seed":
            from .bundle import seed

            seed(load_task(args.task), args.destination, args.abi)
            payload = {"status": "CREATED", "source": str(args.destination.resolve())}
        else:
            from .store import Campaign

            if args.command == "init":
                campaign = Campaign.create(args.run_dir, load_task(args.task), args.max_attempts, args.device_launches)
                payload = {"status": "CREATED", "run_dir": str(campaign.root)}
            else:
                campaign = Campaign(args.run_dir)
                if args.command == "submit":
                    from .bundle import freeze

                    bundle = freeze(args.source, campaign.task(), campaign.objects)
                    campaign.add_candidate(bundle)
                    payload = {"status": "FROZEN", "candidate_id": bundle["candidate_id"]}
                elif args.command == "report":
                    payload = campaign.report()
                elif args.command == "device-quota":
                    campaign.authorize_device(args.launches)
                    payload = {"status": "QUOTA_SET", "launches": args.launches}
                else:
                    from . import runner

                    if args.command == "run":
                        payload = runner.run(
                            campaign,
                            args.candidate_id,
                            args.mode,
                            args.timeout_s,
                            args.case,
                            args.device_index,
                            args.measure,
                            args.max_launches,
                            args.allow_resident_processes,
                            args.trial_id,
                        )
                    elif args.command == "replay":
                        payload = runner.replay(campaign, args.attempt_id, args.timeout_s)
                    else:
                        payload = {"status": "RECONCILED", "results": runner.resume(campaign)}
        output = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.command in ("selfcheck", "report", "experiment-report") and args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        print(output, end="")
        return {"ERROR": 2, "FAIL": 1}.get(payload.get("status"), 0)
    except (ContractError, OSError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}))
        return 2
