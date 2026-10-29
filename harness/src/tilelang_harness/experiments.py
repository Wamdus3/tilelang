"""Deterministic, resumable generation comparisons using the existing P1 evaluator."""

import json
import os
import random
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .artifacts import ArtifactStore, read_json, write_json
from .bundle import freeze
from .compiler import runner_identity
from .contracts import ContractError, TaskSpec, content_id, fields, integer, number, text
from .generation import (
    parse_generator,
    parse_response,
    response_bundle,
    skill_links,
    verify_generator,
    verify_links,
    write_context,
)
from .runner import alive, process_identity, resume, run, stop_group
from .skill_workflow import (
    PROFILE,
    STAGES,
    chunk_text,
    make_manifest,
    parse_stage_response,
    profile_for,
    workflow_context,
)
from .store import Campaign, new_id
from .tasks.registry import TASK_NAMES, load_task, verify_sources


def normalize_spec(value, repo_root):
    fields(
        value,
        {
            "schema_version",
            "name",
            "tasks",
            "arms",
            "replicates",
            "seed",
            "max_repairs",
            "evaluation_mode",
            "evaluation_timeout_s",
            "budget",
        },
        "experiment",
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ContractError("unsupported experiment schema")
    text(value["name"], "experiment name")
    integer(value["replicates"], "replicates")
    integer(value["seed"], "seed", 0)
    integer(value["max_repairs"], "max_repairs", 0)
    if value["max_repairs"] > 10:
        raise ContractError("max_repairs exceeds 10")
    if value["evaluation_mode"] not in ("host", "lower"):
        raise ContractError(
            "experiments only auto-evaluate host/lower; device jobs require explicit single-case commands"
        )
    number(value["evaluation_timeout_s"], "evaluation timeout", 0.05)
    if not isinstance(value["tasks"], list) or not value["tasks"]:
        raise ContractError("tasks must be a nonempty list")
    tasks = {}
    for name in value["tasks"]:
        text(name, "task name")
        task = load_task(name if name in TASK_NAMES else repo_root / name)
        if task.task_id in tasks:
            raise ContractError("duplicate task identity")
        tasks[task.task_id] = task.to_dict()
    if not isinstance(value["arms"], list) or not value["arms"]:
        raise ContractError("arms must be a nonempty list")
    arms = {}
    for arm in value["arms"]:
        fields(arm, {"name", "generator", "skill_links"} | ({"skill_workflow"} & set(arm)), "arm")
        name = text(arm["name"], "arm name")
        if name in arms:
            raise ContractError("duplicate arm")
        arms[name] = {
            "generator": parse_generator(arm["generator"], repo_root),
            "skill_links": skill_links(repo_root, arm["skill_links"]),
            "skill_workflow": profile_for(arm, value["evaluation_mode"]),
        }
        if arms[name]["skill_workflow"] == PROFILE:
            for stage in STAGES:
                make_manifest(repo_root, arms[name]["skill_links"], stage)
    budget = fields(value["budget"], {"generations", "reserved_tokens", "evaluations", "wall_seconds"}, "budget")
    for key in ("generations", "reserved_tokens", "evaluations"):
        integer(budget[key], key)
    number(budget["wall_seconds"], "wall budget", 0.05)
    count = len(tasks) * len(arms) * value["replicates"]
    stage_counts = [len(STAGES) if arm["skill_workflow"] == PROFILE else 1 for arm in arms.values()]
    initial_calls = len(tasks) * value["replicates"] * sum(stage_counts)
    initial_tokens = (
        len(tasks)
        * value["replicates"]
        * sum(arm["generator"]["token_reservation"] * stages for arm, stages in zip(arms.values(), stage_counts))
    )
    if budget["generations"] < initial_calls or budget["reserved_tokens"] < initial_tokens:
        raise ContractError("budget must cover all initial generation trials and their skill stages")
    if budget["evaluations"] > 10000 or count > 10000:
        raise ContractError("experiment exceeds 10000 trials/evaluations")
    return {**value, "tasks": tasks, "arms": arms, "repo_root": str(repo_root), "runner": runner_identity()}


class Experiment:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not (self.root / "experiment.sqlite").is_file():
            raise ContractError("experiment does not exist")
        self.objects = ArtifactStore(self.root / "objects")

    @contextmanager
    def db(self):
        with sqlite3.connect(self.root / "experiment.sqlite", timeout=10) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db

    @contextmanager
    def lock(self):
        if os.name != "posix":
            raise ContractError("experiment execution requires Linux/WSL")
        import fcntl

        with (self.root / "controller.lock").open("a+b") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ContractError("another experiment controller is active") from exc
            yield

    @classmethod
    def create(cls, root, value, repo_root):
        repo_root = Path(repo_root).resolve()
        spec = normalize_spec(value, repo_root)
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=False)
        objects = ArtifactStore(root / "objects")
        ref = objects.put_json(spec)
        with sqlite3.connect(root / "experiment.sqlite") as db:
            db.executescript("""
                PRAGMA synchronous=FULL;
                CREATE TABLE settings (id INTEGER PRIMARY KEY CHECK(id=1), spec_ref TEXT NOT NULL,
                    created REAL NOT NULL, generations INTEGER NOT NULL, tokens INTEGER NOT NULL,
                    evaluations INTEGER NOT NULL);
                CREATE TABLE trials (trial_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, arm TEXT NOT NULL,
                    replica INTEGER NOT NULL, round INTEGER NOT NULL, parent TEXT UNIQUE, root_id TEXT NOT NULL,
                    state TEXT NOT NULL, context_ref TEXT, response_ref TEXT, process_ref TEXT, execution_ref TEXT,
                    candidate_id TEXT, attempt_id TEXT, result_ref TEXT, workflow_ref TEXT);
                CREATE TABLE skill_reads (trial_id TEXT NOT NULL, request_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL, delivery_ref TEXT NOT NULL,
                    PRIMARY KEY(trial_id, request_id, chunk_id));
                CREATE TABLE events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, trial_id TEXT,
                    kind TEXT NOT NULL, payload_ref TEXT NOT NULL, created REAL NOT NULL);
            """)
            db.execute("INSERT INTO settings VALUES (1,?,?,0,0,0)", (ref, time.time()))
            trials = [
                (task_id, name, replica)
                for replica in range(spec["replicates"])
                for task_id in spec["tasks"]
                for name in spec["arms"]
            ]
            random.Random(spec["seed"]).shuffle(trials)
            for task_id, name, replica in trials:
                trial_id = new_id()
                db.execute(
                    "INSERT INTO trials(trial_id,task_id,arm,replica,round,parent,root_id,state) "
                    "VALUES (?,?,?,?,0,NULL,?,'QUEUED')",
                    (trial_id, task_id, name, replica, trial_id),
                )
        for task_id, task in spec["tasks"].items():
            Campaign.create(root / "campaigns" / task_id, TaskSpec.from_dict(task), spec["budget"]["evaluations"])
        return cls(root)

    def settings(self):
        with self.db() as db:
            return dict(db.execute("SELECT * FROM settings WHERE id=1").fetchone())

    def spec(self):
        return self.objects.json(self.settings()["spec_ref"])

    def trials(self):
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM trials ORDER BY rowid")]

    def event(self, db, trial_id, kind, payload):
        ref = self.objects.put_json(payload)
        db.execute(
            "INSERT INTO events(trial_id,kind,payload_ref,created) VALUES (?,?,?,?)", (trial_id, kind, ref, time.time())
        )

    def update(self, trial_id, **values):
        allowed = {
            "state",
            "context_ref",
            "response_ref",
            "process_ref",
            "execution_ref",
            "candidate_id",
            "attempt_id",
            "result_ref",
            "workflow_ref",
        }
        if not set(values) <= allowed:
            raise ContractError("unknown trial update")
        with self.db() as db:
            db.execute(
                "UPDATE trials SET " + ",".join(key + "=?" for key in values) + " WHERE trial_id=?",
                (*values.values(), trial_id),
            )
            self.event(db, trial_id, "TRIAL_UPDATED", values)

    def remaining(self):
        settings, spec = self.settings(), self.spec()
        return settings["created"] + spec["budget"]["wall_seconds"] - time.time()

    def workflow(self, trial):
        return self.objects.json(trial["workflow_ref"]) if trial.get("workflow_ref") else {"steps": []}

    def staged(self, trial):
        return self.spec()["arms"][trial["arm"]].get("skill_workflow") == PROFILE

    def request_work(self, trial):
        work = self.root / "requests" / trial["trial_id"]
        if self.staged(trial):
            work /= STAGES[len(self.workflow(trial)["steps"])]
        return work

    def read_skill(self, trial_id, chunk_id):
        """Deliver one bounded original excerpt and persist its delivery identity."""
        trial = next((row for row in self.trials() if row["trial_id"] == trial_id), None)
        if trial is None or trial["state"] not in ("WAITING_RESPONSE", "GENERATING") or not self.staged(trial):
            raise ContractError("skill-read requires an active staged request")
        context = self.objects.json(trial["context_ref"])
        workflow = context["skill_workflow"]
        chunk = next((item for item in workflow["chunks"] if item["chunk_id"] == chunk_id), None)
        if chunk is None:
            raise ContractError("chunk is not required by this stage")
        excerpt = chunk_text(Path(context["repository_root"]), chunk)
        delivery = {
            "request_id": workflow["request_id"],
            "chunk": chunk,
            "content": excerpt,
            "scope": "bounded_tool_delivery; receiver must re-read truncated returns",
        }
        ref = self.objects.put_json(delivery)
        with self.db() as db:
            current = db.execute("SELECT context_ref,state FROM trials WHERE trial_id=?", (trial_id,)).fetchone()
            if current["context_ref"] != trial["context_ref"] or current["state"] not in (
                "WAITING_RESPONSE",
                "GENERATING",
            ):
                raise ContractError("skill request advanced during reading")
            db.execute(
                "INSERT OR IGNORE INTO skill_reads VALUES (?,?,?,?)", (trial_id, workflow["request_id"], chunk_id, ref)
            )
        return {**delivery, "delivery_id": ref}

    def skill_status(self, trial_id):
        trial = next((row for row in self.trials() if row["trial_id"] == trial_id), None)
        if trial is None or trial["state"] not in ("WAITING_RESPONSE", "GENERATING") or not self.staged(trial):
            raise ContractError("skill-status requires an active staged request")
        workflow = self.objects.json(trial["context_ref"])["skill_workflow"]
        with self.db() as db:
            rows = db.execute(
                "SELECT chunk_id,delivery_ref FROM skill_reads WHERE trial_id=? AND request_id=?",
                (trial_id, workflow["request_id"]),
            ).fetchall()
        for row in rows:
            self.objects.get(row["delivery_ref"])
        delivered = {row["chunk_id"] for row in rows}
        pending = [chunk for chunk in workflow["chunks"] if chunk["chunk_id"] not in delivered]
        return {
            "stage": workflow["stage"],
            "request_id": workflow["request_id"],
            "request_directory": str(self.request_work(trial)),
            "required_chunks": len(workflow["chunks"]),
            "delivered_chunks": len(delivered),
            "next_chunk": pending[0] if pending else None,
            "required_checks": workflow["required_checks"],
            "stage_deliverable": workflow["stage_deliverable"],
            "assurance": workflow["assurance"],
        }

    def lowering_files(self, trial):
        """Export exact per-case CAS artifacts, not mutable worker-directory files."""
        progress = self.workflow(trial)
        if "lower_result" not in progress:
            return {}
        campaign = Campaign(self.root / "campaigns" / trial["task_id"])
        attempts = campaign.report()["attempts"]
        result = next(row["result"] for row in attempts if row["attempt_id"] == progress["lower_result"]["attempt_id"])
        files = {}
        for case_id, build in result["builds"].items():
            for name, checksum in build["artifacts"].items():
                files[f"lowering/{case_id}/{name}"] = campaign.objects.get(checksum)
        return files

    def check_sources(self):
        spec = self.spec()
        if runner_identity() != spec["runner"]:
            raise ContractError("runner changed; create a new experiment")
        for task in spec["tasks"].values():
            verify_sources(TaskSpec.from_dict(task))
        for arm in spec["arms"].values():
            verify_links(Path(spec["repo_root"]), arm["skill_links"])
            verify_generator(arm["generator"])

    def reserve_generation(self, trial, generator):
        spec = self.spec()
        remaining = self.remaining()
        with self.db() as db:
            setting = db.execute("SELECT * FROM settings").fetchone()
            if (
                setting["generations"] >= spec["budget"]["generations"]
                or setting["tokens"] + generator["token_reservation"] > spec["budget"]["reserved_tokens"]
                or remaining <= 0
            ):
                return False
            db.execute(
                "UPDATE settings SET generations=generations+1,tokens=tokens+?", (generator["token_reservation"],)
            )
            db.execute("UPDATE trials SET state='PREPARING' WHERE trial_id=?", (trial["trial_id"],))
            self.event(db, trial["trial_id"], "GENERATION_RESERVED", {"tokens": generator["token_reservation"]})
        return True

    def finish(self, trial, outcome, allow_repair=True):
        ref = self.objects.put_json(outcome)
        spec = self.spec()
        with self.db() as db:
            row = db.execute("SELECT state FROM trials WHERE trial_id=?", (trial["trial_id"],)).fetchone()
            if row["state"] == "FINISHED":
                return
            db.execute("UPDATE trials SET state='FINISHED',result_ref=? WHERE trial_id=?", (ref, trial["trial_id"]))
            self.event(db, trial["trial_id"], "TRIAL_FINISHED", {"result_ref": ref})
            if allow_repair and outcome["status"] != "PASS" and trial["round"] < spec["max_repairs"]:
                db.execute(
                    "INSERT OR IGNORE INTO trials(trial_id,task_id,arm,replica,round,parent,root_id,state) "
                    "VALUES (?,?,?,?,?,?,?,'QUEUED')",
                    (
                        new_id(),
                        trial["task_id"],
                        trial["arm"],
                        trial["replica"],
                        trial["round"] + 1,
                        trial["trial_id"],
                        trial["root_id"],
                    ),
                )

    def prepare(self, trial):
        spec = self.spec()
        arm = spec["arms"][trial["arm"]]
        work = self.request_work(trial)
        work.mkdir(parents=True, exist_ok=False)
        feedback = previous = None
        if trial["parent"]:
            parent = next(row for row in self.trials() if row["trial_id"] == trial["parent"])
            feedback = self.objects.json(parent["result_ref"])
            previous = {
                "trial_id": parent["trial_id"],
                "candidate_id": parent["candidate_id"],
                "directory": str(self.root / "requests" / parent["trial_id"]),
            }
        task = TaskSpec.from_dict(spec["tasks"][trial["task_id"]])
        workflow = None
        if self.staged(trial):
            progress = self.workflow(trial)
            stage = STAGES[len(progress["steps"])]
            workflow = workflow_context(
                Path(spec["repo_root"]),
                arm["skill_links"],
                stage,
                trial["trial_id"],
                progress,
                [case.case_id for case in task.cases],
            )
            workflow["experiment_directory"] = str(self.root)
            lower_files = self.lowering_files(trial)
            for name, data in lower_files.items():
                destination = work / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            # Keep the actual byte hashes accessible to adapters without loading
            # large lowering files into their prompt.
            workflow["lowering_artifacts"] = {name: self.objects.put(data) for name, data in lower_files.items()}
            workflow["prior_files"] = {}
            for step in progress["steps"]:
                prior = self.objects.json(step["response_ref"])
                for name, source in prior["files"].items():
                    destination = work / "prior" / step["stage"] / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(source, encoding="utf-8")
                    workflow["prior_files"][destination.relative_to(work).as_posix()] = self.objects.put(
                        source.encode("utf-8")
                    )
        context = write_context(
            work,
            task,
            arm["skill_links"],
            Path(spec["repo_root"]),
            spec["evaluation_mode"],
            spec["seed"] + trial["replica"],
            feedback,
            previous,
            workflow,
        )
        context.update(
            model=arm["generator"]["model"],
            parameters=arm["generator"]["parameters"],
            token_reservation=arm["generator"]["token_reservation"],
            trial_id=trial["trial_id"],
        )
        write_json(work / "context.json", context)
        self.update(
            trial["trial_id"],
            context_ref=self.objects.put_json(context),
            response_ref=None,
            process_ref=None,
            execution_ref=None,
            state="WAITING_RESPONSE" if arm["generator"]["kind"] == "file" else "GENERATING",
        )
        if arm["generator"]["kind"] == "command":
            timeout = min(arm["generator"]["timeout_s"], max(0, self.remaining()))
            write_json(
                work / "invocation.json",
                {"command": arm["generator"]["command"], "cwd": spec["repo_root"], "deadline": time.time() + timeout},
            )
            env = os.environ.copy()
            paths = [
                str(Path(__file__).resolve().parents[1]),
                *[str(Path(p or os.getcwd()).resolve()) for p in sys.path],
            ]
            env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
            with (work / "watchdog.log").open("wb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "tilelang_harness.generator_worker", str(work)],
                    cwd=work,
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                identity = process_identity(process.pid)
                if identity is None:
                    process.wait()
                    raise ContractError("generator watchdog exited before registration")
                self.update(trial["trial_id"], process_ref=self.objects.put_json(identity))
                write_json(work / "release.json", {"trial_id": trial["trial_id"]})

    def collect(self, trial):
        spec = self.spec()
        work = self.request_work(trial)
        generator = spec["arms"][trial["arm"]]["generator"]
        if generator["kind"] == "command":
            if trial["process_ref"]:
                identity = self.objects.json(trial["process_ref"])
                while alive(identity):
                    time.sleep(0.05)
                stop_group(identity)  # Also clean descendants after a killed watchdog.
            elif (work / "invocation.json").exists() and time.time() < read_json(work / "invocation.json")["deadline"]:
                return False
            terminal = read_json(work / "termination.json") if (work / "termination.json").exists() else {}
            logs = {}
            for name in ("termination.json", "stdout.log", "stderr.log", "watchdog.log"):
                if (work / name).is_file():
                    with (work / name).open("rb") as stream:
                        logs[name] = self.objects.put(stream.read(16 * 1024 * 1024))
            self.update(trial["trial_id"], execution_ref=self.objects.put_json(logs))
            if terminal.get("status") != "PASS":
                self.finish(
                    trial,
                    {
                        "status": "ERROR",
                        "scope": "generation",
                        "category": "GENERATOR_FAILED",
                        "diagnostic": terminal or {"status": "INTERRUPTED"},
                    },
                )
                return True
        if not (work / "response.json").exists():
            if generator["kind"] == "file" and self.remaining() > 0:
                return False
            self.finish(trial, {"status": "ERROR", "scope": "generation", "category": "MISSING_RESPONSE"})
            return True
        with (work / "response.json").open("rb") as stream:
            raw = stream.read(16 * 1024 * 1024 + 1)
        if trial["response_ref"] and self.objects.get(trial["response_ref"]) != raw:
            raise ContractError("generator response changed after collection")
        self.update(trial["trial_id"], response_ref=self.objects.put(raw))
        self.check_sources()
        if read_json(work / "context.json") != self.objects.json(trial["context_ref"]):
            raise ContractError("generation context changed")
        try:
            if self.staged(trial):
                return self.collect_stage(trial, work, generator)
            response = parse_response(work / "response.json", generator["token_reservation"])
            source = work / "candidate"
            response_bundle(response, source, spec["evaluation_mode"])
            campaign = Campaign(self.root / "campaigns" / trial["task_id"])
            bundle = freeze(source, campaign.task(), campaign.objects)
            campaign.add_candidate(bundle, skill_trial_id=trial["trial_id"], require_skill_gate=False)
        except (ContractError, OSError, TypeError, KeyError) as exc:
            token_violation = "token usage above" in str(exc)
            if token_violation:
                # A misbehaving adapter cannot be undone, but cannot trigger more calls.
                with self.db() as db:
                    db.execute("UPDATE settings SET tokens=?", (spec["budget"]["reserved_tokens"],))
            self.finish(
                trial,
                {
                    "status": "ERROR",
                    "scope": "generation",
                    "category": "INVALID_PROPOSAL",
                    "error": str(exc),
                    "skill_stage": self.objects.json(trial["context_ref"]).get("skill_workflow", {}).get("stage"),
                },
                allow_repair=not token_violation,
            )
            return True
        self.update(trial["trial_id"], state="GENERATED", candidate_id=bundle["candidate_id"])
        return True

    def collect_stage(self, trial, work, generator):
        context = self.objects.json(trial["context_ref"])
        workflow = context["skill_workflow"]
        with self.db() as db:
            reads = db.execute(
                "SELECT chunk_id,delivery_ref FROM skill_reads WHERE trial_id=? AND request_id=?",
                (trial["trial_id"], workflow["request_id"]),
            ).fetchall()
        deliveries = {row["chunk_id"]: row["delivery_ref"] for row in reads}
        for ref in deliveries.values():
            self.objects.get(ref)
        response = parse_stage_response(
            work / "response.json",
            context,
            deliveries,
            Path(context["repository_root"]),
            generator["token_reservation"],
            self.lowering_files(trial),
        )
        stage = workflow["stage"]
        progress = self.workflow(trial)
        # Preserve every stage's raw response, context and command logs even
        # after the trial's active request pointers move on to the next stage.
        current = next(row for row in self.trials() if row["trial_id"] == trial["trial_id"])
        progress["steps"].append(
            {
                "stage": stage,
                "context_ref": trial["context_ref"],
                "response_ref": current["response_ref"],
                "execution_ref": current["execution_ref"],
                "deliveries": deliveries,
                "assurance": workflow["assurance"],
            }
        )
        values = {"workflow_ref": self.objects.put_json(progress), "state": "QUEUED"}
        if stage == "implementation":
            proposal = {key: response[key] for key in ("schema_version", "files", "config", "plan", "usage")}
            source = self.root / "requests" / trial["trial_id"] / "candidate"
            response_bundle(proposal, source, "lower")
            campaign = Campaign(self.root / "campaigns" / trial["task_id"])
            bundle = freeze(source, campaign.task(), campaign.objects)
            campaign.add_candidate(bundle, skill_trial_id=trial["trial_id"])
            values.update(state="GENERATED", candidate_id=bundle["candidate_id"])
        if stage == "validation":
            values["state"] = "WORKFLOW_COMPLETE"
        self.update(trial["trial_id"], **values)
        return True

    def finalize_workflow(self, trial):
        progress = self.workflow(trial)
        campaign = Campaign(self.root / "campaigns" / trial["task_id"])
        for step in progress["steps"]:
            refs = [step["context_ref"], step["response_ref"], *step["deliveries"].values()]
            if step["execution_ref"]:
                refs.append(step["execution_ref"])
                refs.extend(self.objects.json(step["execution_ref"]).values())
            for ref in refs:
                campaign.objects.put(self.objects.get(ref))
        campaign.complete_skill_gate(trial["candidate_id"], trial["trial_id"], progress)
        self.finish(
            trial,
            {
                **progress["lower_result"],
                "skill_workflow": {
                    "status": "CONTRACT_CHECKED",
                    "profile": PROFILE,
                    "assurance": "bounded_delivery_and_agent_reported_evidence; not semantic or hardware proof",
                    "device_readiness": "AGENT_REPORTED_REQUIRES_EXPLICIT_AUTHORIZATION",
                },
            },
        )

    def accept_evaluation(self, trial, outcome):
        if not self.staged(trial) or outcome["status"] != "PASS":
            self.finish(trial, outcome)
            return
        progress = self.workflow(trial)
        progress["lower_result"] = outcome
        self.update(trial["trial_id"], workflow_ref=self.objects.put_json(progress), state="QUEUED")

    def evaluate(self, trial):
        spec = self.spec()
        remaining = self.remaining()
        with self.db() as db:
            setting = db.execute("SELECT evaluations FROM settings").fetchone()
            if setting[0] >= spec["budget"]["evaluations"] or remaining < 0.05:
                allowed = False
            else:
                db.execute("UPDATE settings SET evaluations=evaluations+1")
                db.execute("UPDATE trials SET state='EVALUATING' WHERE trial_id=?", (trial["trial_id"],))
                allowed = True
        if not allowed:
            self.finish(
                trial,
                {"status": "INCOMPLETE", "scope": spec["evaluation_mode"], "category": "EVALUATION_BUDGET_EXHAUSTED"},
                allow_repair=False,
            )
            return
        campaign = Campaign(self.root / "campaigns" / trial["task_id"])
        result = run(
            campaign,
            trial["candidate_id"],
            spec["evaluation_mode"],
            min(spec["evaluation_timeout_s"], max(0.05, self.remaining())),
            trial_id=trial["trial_id"],
        )
        self.update(trial["trial_id"], attempt_id=result["attempt_id"])
        self.check_compiler_environment(campaign)
        self.accept_evaluation(
            trial,
            {
                "status": result["status"],
                "scope": spec["evaluation_mode"],
                "category": result["category"],
                "attempt_id": result["attempt_id"],
                "error": result.get("error"),
                "evidence": result["evidence"],
                "wall_seconds": result.get("wall_seconds"),
                "stage": result.get("stage"),
                "cases": result.get("cases", []),
                "evaluation_directory": str(campaign.root / "work" / result["attempt_id"]),
            },
        )

    def recover_evaluation(self, trial):
        campaign = Campaign(self.root / "campaigns" / trial["task_id"])
        resume(campaign)
        self.check_compiler_environment(campaign)
        for row in campaign.report()["attempts"]:
            if row["job"].get("experiment_trial_id") == trial["trial_id"] and "result" in row:
                result = row["result"]
                self.update(trial["trial_id"], attempt_id=row["attempt_id"])
                self.accept_evaluation(
                    trial,
                    {
                        "status": result["status"],
                        "scope": self.spec()["evaluation_mode"],
                        "category": result["category"],
                        "error": result.get("error"),
                        "attempt_id": row["attempt_id"],
                        "evidence": result["evidence"],
                        "cases": result.get("cases", []),
                        "evaluation_directory": str(campaign.root / "work" / row["attempt_id"]),
                    },
                )
                return
        self.finish(
            trial,
            {
                "status": "INCOMPLETE",
                "scope": self.spec()["evaluation_mode"],
                "category": "INTERRUPTED_BEFORE_EVALUATION_RESERVATION",
            },
        )

    @staticmethod
    def compiler_packs(campaign):
        packs = set()
        for attempt in campaign.report()["attempts"]:
            for name, ref in attempt.get("result", {}).get("artifacts", {}).items():
                if name.endswith("/compiler-pack.json"):
                    packs.add(content_id(campaign.objects.json(ref)))
        return packs

    def check_compiler_environment(self, campaign):
        packs = self.compiler_packs(campaign)
        for task_id in self.spec()["tasks"]:
            other = self.root / "campaigns" / task_id
            if other.resolve() != campaign.root:
                packs.update(self.compiler_packs(Campaign(other)))
        if len(packs) > 1:
            raise ContractError("compiler environment changed between trials; create a new experiment")

    def run(self):
        with self.lock():
            self.check_sources()
            while True:
                pending = [row for row in self.trials() if row["state"] != "FINISHED"]
                if not pending:
                    break
                # Continue released operations before reserving a new model call.
                active = next((row for row in pending if row["state"] != "QUEUED"), pending[0])
                state = active["state"]
                if state == "QUEUED":
                    generator = self.spec()["arms"][active["arm"]]["generator"]
                    if not self.reserve_generation(active, generator):
                        self.finish(
                            active,
                            {"status": "INCOMPLETE", "scope": "generation", "category": "GENERATION_BUDGET_EXHAUSTED"},
                            allow_repair=False,
                        )
                    else:
                        self.prepare(active)
                elif state == "PREPARING":
                    self.finish(
                        active,
                        {
                            "status": "INCOMPLETE",
                            "scope": "generation",
                            "category": "INTERRUPTED_BEFORE_GENERATOR_RELEASE",
                        },
                    )
                elif state in ("GENERATING", "WAITING_RESPONSE"):
                    if not self.collect(active):
                        return {
                            "status": "WAITING_RESPONSE",
                            "trial_id": active["trial_id"],
                            "request_directory": str(self.request_work(active)),
                            "skill_stage": self.objects.json(active["context_ref"])
                            .get("skill_workflow", {})
                            .get("stage"),
                        }
                elif state == "GENERATED":
                    self.evaluate(active)
                elif state == "EVALUATING":
                    self.recover_evaluation(active)
                elif state == "WORKFLOW_COMPLETE":
                    self.finalize_workflow(active)
                else:
                    raise ContractError("unknown trial state")
            return self.report()

    def report(self):
        spec = self.spec()
        rows = []
        for trial in self.trials():
            item = dict(trial)
            if trial.get("workflow_ref"):
                progress = self.workflow(trial)
                item["skill_workflow"] = progress
                for step in progress["steps"]:
                    self.objects.get(step["context_ref"])
                    step["reported_usage"] = self.objects.json(step["response_ref"])["usage"]
                    for ref in step["deliveries"].values():
                        self.objects.get(ref)
                    if step["execution_ref"]:
                        logs = self.objects.json(step["execution_ref"])
                        for ref in logs.values():
                            self.objects.get(ref)
                        if logs.get("termination.json"):
                            step["generation_termination"] = self.objects.json(logs["termination.json"])
            for name in ("context", "response", "result"):
                if trial[name + "_ref"]:
                    raw = self.objects.get(trial[name + "_ref"])
                    if name != "response":
                        item[name] = self.objects.json(trial[name + "_ref"])
                    else:
                        item["raw_response_bytes"] = len(raw)
                        try:
                            parsed = json.loads(raw)
                            item["reported_usage"] = parsed.get("usage") if isinstance(parsed, dict) else None
                        except (ValueError, UnicodeError):
                            item["reported_usage"] = None
            if trial["execution_ref"]:
                item["generation_logs"] = self.objects.json(trial["execution_ref"])
                for ref in item["generation_logs"].values():
                    self.objects.get(ref)
                terminal_ref = item["generation_logs"].get("termination.json")
                if terminal_ref:
                    item["generation_termination"] = self.objects.json(terminal_ref)
            if self.staged(trial):
                with self.db() as db:
                    deliveries = [
                        dict(row)
                        for row in db.execute(
                            "SELECT request_id,chunk_id,delivery_ref FROM skill_reads WHERE trial_id=? ORDER BY request_id,chunk_id",
                            (trial["trial_id"],),
                        )
                    ]
                for delivery in deliveries:
                    self.objects.get(delivery["delivery_ref"])
                item["skill_deliveries"] = deliveries
            rows.append(item)
        campaigns = {name: Campaign(self.root / "campaigns" / name).report() for name in spec["tasks"]}
        compiler_packs = {
            name: sorted(self.compiler_packs(Campaign(self.root / "campaigns" / name))) for name in spec["tasks"]
        }
        groups = []
        for task_id in spec["tasks"]:
            for arm in spec["arms"]:
                selected = [row for row in rows if row["task_id"] == task_id and row["arm"] == arm]
                initial = [row for row in selected if row["round"] == 0]
                passed_roots = {row["root_id"] for row in selected if row.get("result", {}).get("status") == "PASS"}
                groups.append(
                    {
                        "task_id": task_id,
                        "arm": arm,
                        "planned_trials": spec["replicates"],
                        "first_attempt_passes": sum(row.get("result", {}).get("status") == "PASS" for row in initial),
                        "within_budget_passes": len(passed_roots),
                        "finished_generations": sum(row["response_ref"] is not None for row in selected),
                        "pending_trials": sum(row["state"] != "FINISHED" for row in selected),
                        "unique_candidates": len({row["candidate_id"] for row in selected if row["candidate_id"]}),
                        "success_scope": spec["evaluation_mode"],
                        "device_correctness": "NOT_RUN",
                        "measurement": "NOT_RUN",
                    }
                )
                group = groups[-1]
                group["first_attempt_success_rate"] = group["first_attempt_passes"] / spec["replicates"]
                group["within_budget_success_rate"] = len(passed_roots) / spec["replicates"]
                failures = {}
                for row in selected:
                    result = row.get("result", {})
                    if result.get("status") != "PASS" and result.get("category"):
                        key = result["category"]
                        failures[key] = failures.get(key, 0) + 1
                group["failure_categories"] = failures
                usages = []
                for row in selected:
                    steps = row.get("skill_workflow", {}).get("steps", [])
                    usages.extend(step["reported_usage"] for step in steps)
                    if row["response_ref"] and row["response_ref"] not in {step["response_ref"] for step in steps}:
                        usages.append(row.get("reported_usage"))
                known = [
                    usage
                    for usage in usages
                    if isinstance(usage, dict)
                    and all(
                        type(usage.get(key)) is int and usage[key] >= 0 for key in ("input_tokens", "output_tokens")
                    )
                ]
                group["reported_tokens_known"] = sum(usage["input_tokens"] + usage["output_tokens"] for usage in known)
                group["responses_with_unknown_usage"] = len(usages) - len(known)
                group["finished_generations"] = len(usages)
                wall_seconds = 0
                for row in selected:
                    steps = row.get("skill_workflow", {}).get("steps", [])
                    wall_seconds += sum(step.get("generation_termination", {}).get("wall_seconds", 0) for step in steps)
                    if row["execution_ref"] not in {step["execution_ref"] for step in steps}:
                        wall_seconds += row.get("generation_termination", {}).get("wall_seconds", 0)
                group["generator_wall_seconds_known"] = wall_seconds
                group["skill_workflow"] = spec["arms"][arm].get("skill_workflow", "links")
        for trial in rows:
            trial["device_attempts"] = [
                row
                for row in campaigns[trial["task_id"]]["attempts"]
                if row["job"].get("experiment_trial_id") == trial["trial_id"]
                and row["candidate_id"] == trial["candidate_id"]
                and row["job"]["mode"] == "device-smoke"
            ]
        with self.db() as db:
            events = [dict(row) for row in db.execute("SELECT * FROM events ORDER BY sequence")]
        for event in events:
            self.objects.get(event["payload_ref"])
        reserved = [event["trial_id"] for event in events if event["kind"] == "GENERATION_RESERVED"]
        for group in groups:
            selected = [row for row in rows if row["task_id"] == group["task_id"] and row["arm"] == group["arm"]]
            group["generation_reservations"] = sum(reserved.count(row["trial_id"]) for row in selected)
            group["reserved_tokens"] = (
                group["generation_reservations"] * spec["arms"][group["arm"]]["generator"]["token_reservation"]
            )
        return {
            "schema_version": 1,
            "status": "COMPLETE" if all(row["state"] == "FINISHED" for row in rows) else "INCOMPLETE",
            "experiment_id": content_id(spec),
            "spec": spec,
            "usage": self.settings(),
            "compiler_packs_by_task": compiler_packs,
            "compiler_environment_comparable": len({pack for packs in compiler_packs.values() for pack in packs}) <= 1,
            "groups": groups,
            "trials": rows,
            "campaigns": campaigns,
            "events": events,
            "eligible_for_promotion": False,
            "device_execution": "RECORDED_SEPARATELY" if any(row["device_attempts"] for row in rows) else "NOT_RUN",
        }
