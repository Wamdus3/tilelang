"""Single-writer campaign metadata and append-only events in SQLite."""

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .artifacts import ArtifactStore
from .contracts import ContractError
from .state import reduce_result


class Campaign:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.objects = ArtifactStore(self.root / "objects")
        if not (self.root / "run.sqlite").is_file():
            raise ContractError("campaign does not exist; use init first")

    @classmethod
    def create(cls, root, task, max_attempts=20, device_launches=0):
        from .contracts import integer

        integer(device_launches, "device_launches", 0)
        if type(max_attempts) is not int or not 1 <= max_attempts <= 10000:
            raise ContractError("attempt budget must be between 1 and 10000")
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=False)
        objects = ArtifactStore(root / "objects")
        task_ref = objects.put_json(task.to_dict())
        with sqlite3.connect(root / "run.sqlite") as db:
            db.executescript("""
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                CREATE TABLE campaign (id INTEGER PRIMARY KEY CHECK(id=1), task_ref TEXT NOT NULL,
                    max_attempts INTEGER NOT NULL);
                CREATE TABLE candidates (candidate_id TEXT PRIMARY KEY, bundle_ref TEXT NOT NULL);
                CREATE TABLE resources (id INTEGER PRIMARY KEY CHECK(id=1), launch_limit INTEGER NOT NULL,
                    reserved_launches INTEGER NOT NULL);
                CREATE TABLE attempts (attempt_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL,
                    job_ref TEXT NOT NULL, status TEXT NOT NULL, process_ref TEXT, result_ref TEXT,
                    decision_ref TEXT);
                CREATE TABLE events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_key TEXT UNIQUE NOT NULL,
                    kind TEXT NOT NULL, attempt_id TEXT, payload_ref TEXT NOT NULL, created REAL NOT NULL);
            """)
            db.execute("INSERT INTO campaign VALUES (1, ?, ?)", (task_ref, max_attempts))
            db.execute("INSERT INTO resources VALUES (1, ?, 0)", (device_launches,))
            db.execute(
                "INSERT INTO events(event_key,kind,payload_ref,created) VALUES (?,?,?,?)",
                ("campaign", "CAMPAIGN_CREATED", task_ref, time.time()),
            )
        return cls(root)

    @contextmanager
    def connection(self):
        with sqlite3.connect(self.root / "run.sqlite", timeout=10) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db

    @contextmanager
    def lock(self):
        # Linux flock also works in WSL. Never emulate a durable lease with a pid file.
        if __import__("os").name != "posix":
            raise ContractError("execution/recovery requires Linux; use local WSL")
        import fcntl

        with (self.root / "controller.lock").open("a+b") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ContractError("another controller is operating this campaign") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def settings(self):
        with self.connection() as db:
            row = db.execute("SELECT * FROM campaign WHERE id=1").fetchone()
        return dict(row)

    def task(self):
        from .contracts import TaskSpec
        from .tasks.registry import verify_sources

        task = TaskSpec.from_dict(self.objects.json(self.settings()["task_ref"]))
        verify_sources(task)
        return task

    def authorize_device(self, launches):
        from .contracts import integer

        integer(launches, "device launch quota", 0)
        with self.connection() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS resources (id INTEGER PRIMARY KEY CHECK(id=1), "
                "launch_limit INTEGER NOT NULL, reserved_launches INTEGER NOT NULL)"
            )
            db.execute("INSERT OR IGNORE INTO resources VALUES (1,0,0)")
            old = dict(db.execute("SELECT * FROM resources WHERE id=1").fetchone())
            if launches < old["reserved_launches"]:
                raise ContractError("quota cannot be smaller than already reserved launches")
            db.execute("UPDATE resources SET launch_limit=? WHERE id=1", (launches,))
            self._event(
                db,
                "quota:" + new_id(),
                "DEVICE_QUOTA_SET",
                None,
                self.objects.put_json({"previous": old, "launch_limit": launches}),
            )

    def add_candidate(self, bundle, skill_trial_id=None, require_skill_gate=True):
        ref = self.objects.put_json(bundle)
        gate_ref = (
            None
            if require_skill_gate
            else self.objects.put_json({"profile": "links", "steps": [], "assurance": "links_only_no_skill_workflow"})
        )
        with self.connection() as db:
            old = db.execute(
                "SELECT bundle_ref FROM candidates WHERE candidate_id=?", (bundle["candidate_id"],)
            ).fetchone()
            if old is not None and old[0] != ref:
                raise ContractError("candidate identity conflict")
            db.execute("INSERT OR IGNORE INTO candidates VALUES (?,?)", (bundle["candidate_id"], ref))
            db.execute(
                "INSERT OR IGNORE INTO events(event_key,kind,payload_ref,created) VALUES (?,?,?,?)",
                ("candidate:" + bundle["candidate_id"], "CANDIDATE_FROZEN", ref, time.time()),
            )
            if skill_trial_id is not None:
                # The candidate and its pending gate enter the database in one
                # transaction; a crash cannot expose an ungated candidate.
                db.execute(
                    "CREATE TABLE IF NOT EXISTS skill_gates (candidate_id TEXT NOT NULL, "
                    "trial_id TEXT NOT NULL, evidence_ref TEXT, PRIMARY KEY(candidate_id,trial_id))"
                )
                db.execute(
                    "INSERT OR IGNORE INTO skill_gates VALUES (?,?,?)",
                    (bundle["candidate_id"], skill_trial_id, gate_ref),
                )

    def complete_skill_gate(self, candidate_id, trial_id, evidence):
        ref = self.objects.put_json(evidence)
        with self.connection() as db:
            row = db.execute(
                "SELECT evidence_ref FROM skill_gates WHERE candidate_id=? AND trial_id=?", (candidate_id, trial_id)
            ).fetchone()
            if row is None:
                raise ContractError("unknown skill gate")
            if row[0] is not None:
                if row[0] != ref:
                    raise ContractError("conflicting skill gate evidence")
                self.objects.json(row[0])
                return
            db.execute(
                "UPDATE skill_gates SET evidence_ref=? WHERE candidate_id=? AND trial_id=?",
                (ref, candidate_id, trial_id),
            )
            self._event(db, "skill-gate:" + trial_id, "SKILL_WORKFLOW_CHECKED", None, ref)

    def bundle(self, candidate_id):
        with self.connection() as db:
            row = db.execute("SELECT bundle_ref FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        if row is None:
            raise ContractError("unknown CandidateID")
        return self.objects.json(row[0])

    def reserve(self, job):
        ref = self.objects.put_json(job)
        with self.connection() as db:
            if db.execute("SELECT count(*) FROM attempts WHERE status IN ('STARTING','RUNNING')").fetchone()[0]:
                raise ContractError("unfinished attempt exists; resume before scheduling another")
            budget = db.execute("SELECT max_attempts FROM campaign WHERE id=1").fetchone()[0]
            if db.execute("SELECT count(*) FROM attempts").fetchone()[0] >= budget:
                raise ContractError("campaign attempt budget exhausted")
            if job["mode"] == "device-smoke":
                if db.execute("SELECT name FROM sqlite_master WHERE name='skill_gates'").fetchone():
                    gates = db.execute(
                        "SELECT trial_id,evidence_ref FROM skill_gates WHERE candidate_id=?", (job["candidate_id"],)
                    ).fetchall()
                    if gates:
                        gate = next((row for row in gates if row["trial_id"] == job.get("experiment_trial_id")), None)
                        strict = any(
                            row["evidence_ref"] is None
                            or self.objects.json(row["evidence_ref"]).get("profile") != "links"
                            for row in gates
                        )
                        if gate is None and (strict or job.get("experiment_trial_id") is not None):
                            raise ContractError("skill workflow audit incomplete or explicit trial association missing")
                        if gate is not None:
                            if gate["evidence_ref"] is None:
                                raise ContractError("skill workflow audit incomplete")
                            self.objects.json(gate["evidence_ref"])
                if not db.execute("SELECT name FROM sqlite_master WHERE name='resources'").fetchone():
                    raise ContractError("legacy campaign has no device quota; create a new campaign")
                limits = db.execute("SELECT * FROM resources WHERE id=1").fetchone()
                requested = job["resource_limits"]["launches"]
                if limits["reserved_launches"] + requested > limits["launch_limit"]:
                    raise ContractError("campaign device launch budget exhausted")
                db.execute("UPDATE resources SET reserved_launches=reserved_launches+? WHERE id=1", (requested,))
            db.execute(
                "INSERT INTO attempts(attempt_id,candidate_id,job_ref,status) VALUES (?,?,?,'STARTING')",
                (job["attempt_id"], job["candidate_id"], ref),
            )
            self._event(db, job["attempt_id"] + ":reserved", "JOB_RESERVED", job["attempt_id"], ref)

    def started(self, attempt_id, process):
        ref = self.objects.put_json(process)
        with self.connection() as db:
            if (
                db.execute(
                    "UPDATE attempts SET status='RUNNING',process_ref=? WHERE attempt_id=? AND status='STARTING'",
                    (ref, attempt_id),
                ).rowcount
                != 1
            ):
                raise ContractError("invalid STARTING -> RUNNING transition")
            self._event(db, attempt_id + ":started", "WORKER_STARTED", attempt_id, ref)

    @staticmethod
    def _event(db, key, kind, attempt_id, ref):
        db.execute(
            "INSERT INTO events(event_key,kind,attempt_id,payload_ref,created) VALUES (?,?,?,?,?)",
            (key, kind, attempt_id, ref, time.time()),
        )

    def attempts(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY rowid")]

    def attempt(self, attempt_id):
        for attempt in self.attempts():
            if attempt["attempt_id"] == attempt_id:
                return attempt
        raise ContractError("unknown AttemptID")

    def finish(self, attempt_id, result):
        attempt = self.attempt(attempt_id)
        job = self.objects.json(attempt["job_ref"])
        decision = reduce_result(job, result)
        # Artifact writes complete before the one authoritative metadata transaction.
        ref = self.objects.put_json(result)
        decision_ref = self.objects.put_json(decision)
        with self.connection() as db:
            current = db.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if current["result_ref"] is not None:
                if current["result_ref"] != ref:
                    raise ContractError("conflicting duplicate result")
                return False
            if current["status"] not in ("STARTING", "RUNNING"):
                raise ContractError("invalid terminal transition")
            db.execute(
                "UPDATE attempts SET status='FINISHED',result_ref=?,decision_ref=? WHERE attempt_id=?",
                (ref, decision_ref, attempt_id),
            )
            self._event(db, attempt_id + ":result", "RESULT_COMMITTED", attempt_id, ref)
        return True

    def report(self):
        task = self.task()
        attempts = []
        for row in self.attempts():
            entry = {"attempt_id": row["attempt_id"], "candidate_id": row["candidate_id"], "status": row["status"]}
            entry["job"] = self.objects.json(row["job_ref"])
            for ref in entry["job"]["bundle"]["files"].values():
                self.objects.get(ref)
            for field in ("result", "decision"):
                if row[field + "_ref"]:
                    entry[field] = self.objects.json(row[field + "_ref"])
            if "result" in entry:
                for category in ("artifacts", "logs"):
                    for ref in entry["result"].get(category, {}).values():
                        self.objects.get(ref)
                if entry["result"].get("rejected_result"):
                    self.objects.get(entry["result"]["rejected_result"])
            attempts.append(entry)
        with self.connection() as db:
            events = [dict(row) for row in db.execute("SELECT * FROM events ORDER BY sequence")]
            resources = (
                dict(db.execute("SELECT * FROM resources WHERE id=1").fetchone())
                if db.execute("SELECT name FROM sqlite_master WHERE name='resources'").fetchone()
                else None
            )
            gates = (
                [dict(row) for row in db.execute("SELECT * FROM skill_gates ORDER BY candidate_id,trial_id")]
                if db.execute("SELECT name FROM sqlite_master WHERE name='skill_gates'").fetchone()
                else []
            )
        for gate in gates:
            if gate["evidence_ref"]:
                evidence = self.objects.json(gate["evidence_ref"])
                for step in evidence["steps"]:
                    for key in ("context_ref", "response_ref"):
                        self.objects.get(step[key])
                    for ref in step["deliveries"].values():
                        self.objects.get(ref)
        return {
            "schema_version": 1,
            "task_digest": task.task_digest,
            "task_snapshot": task.to_dict(),
            "max_attempts": self.settings()["max_attempts"],
            "attempts": attempts,
            "events": events,
            "device_quota": resources,
            "skill_gates": gates,
            "eligible_for_promotion": False,
        }


def new_id():
    return uuid.uuid4().hex
