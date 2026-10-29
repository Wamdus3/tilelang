"""Skill delivery/provenance fixtures, never model benchmarks or NPU tests."""

import copy
import os
import sys
from pathlib import Path

import pytest

from tilelang_harness.artifacts import digest, read_json, write_json
from tilelang_harness.contracts import ContractError
from tilelang_harness.experiments import Experiment
from tilelang_harness.generation import skill_links
from tilelang_harness.skill_workflow import ASCEND_SKILL, MAX_CHUNK_BYTES, STAGES, make_manifest
from tilelang_harness.store import Campaign


def skill_fixture(root):
    main = root / ASCEND_SKILL
    main.parent.mkdir(parents=True)
    main.write_text(
        "# Original fixture guidance\n[structure](references/kernel-structure.md)\n"
        "[sync](references/synchronization.md)\n[profiling](references/profiling.md)\n"
        "Design original storage guidance.\n## 1. Mental Model\nImplementation API guidance.\n"
        "## 10. Common Pitfalls\nAudit generated addresses.\n"
        "## 12. Performance\nSeparate validation levels.\n"
    )
    references = main.parent / "references"
    references.mkdir()
    (references / "kernel-structure.md").write_text("# Structure\nChoose a VF for each stage.\n")
    (references / "synchronization.md").write_text(
        "# Synchronization\n### Establish an automatic tiled pipeline\nClassify and budget storage.\n"
        "### Tune performance\nInspect versions and read footprints.\n"
    )
    (references / "profiling.md").write_text("# Profiling\nKeep raw samples and separate device claims.\n")


def specification(profile=None):
    arm = {
        "name": "fixture",
        "skill_links": [ASCEND_SKILL],
        "generator": {
            "kind": "file",
            "command": [],
            "model": "protocol-fixture-not-a-model",
            "parameters": {},
            "timeout_s": 10,
            "token_reservation": 100,
        },
    }
    if profile is not None:
        arm["skill_workflow"] = profile
    return {
        "schema_version": 1,
        "name": "skill-protocol-fixture",
        "tasks": ["vecadd"],
        "arms": [arm],
        "replicates": 1,
        "seed": 42,
        "max_repairs": 0,
        "evaluation_mode": "lower",
        "evaluation_timeout_s": 10,
        "budget": {"generations": 4, "reserved_tokens": 400, "evaluations": 1, "wall_seconds": 120},
    }


def create(tmp_path, monkeypatch):
    skill_fixture(tmp_path)
    experiment = Experiment.create(tmp_path / "run", specification(), tmp_path)

    def lower_files(self, trial):
        return {
            f"lowering/{case['case_id']}/lowered.tir": b"fixture lowered allocation and schedule\n"
            for case in self.spec()["tasks"][trial["task_id"]]["cases"]
        }

    def evaluate(self, trial):
        # Wiring fixture only: this deliberately does NOT perform a real lower.
        with self.db() as db:
            db.execute("UPDATE settings SET evaluations=evaluations+1")
        self.update(trial["trial_id"], attempt_id="fixture-attempt")
        self.accept_evaluation(
            trial,
            {
                "status": "PASS",
                "scope": "lower",
                "category": "FIXTURE_ONLY",
                "attempt_id": "fixture-attempt",
                "evidence": {"lowering": "PASS"},
            },
        )

    monkeypatch.setattr(Experiment, "lowering_files", lower_files)
    monkeypatch.setattr(Experiment, "evaluate", evaluate)
    return experiment


def proposal(experiment, waiting):
    context = read_json(Path(waiting["request_directory"]) / "context.json")
    flow = context["skill_workflow"]
    reads = []
    for chunk in flow["chunks"]:
        delivery = experiment.read_skill(waiting["trial_id"], chunk["chunk_id"])
        reads.append({"chunk_id": chunk["chunk_id"], "delivery_id": delivery["delivery_id"]})
    files = {"notes.md": "Protocol fixture evidence, not a technical kernel audit.\n"}
    if flow["stage"] == "implementation":
        files["solution.py"] = "def build(N):\n    raise RuntimeError('fixture, never executed')\n"
    checks = []
    cases = flow["case_ids"] if flow["stage"] == "lowering_audit" else [None]
    for case in cases:
        evidence = {
            "path": "files/notes.md",
            "sha256": digest(files["notes.md"].encode()),
            "start_line": 1,
            "end_line": 1,
        }
        if case is not None:
            name = f"lowering/{case}/lowered.tir"
            evidence.update(path=name, sha256=flow["lowering_artifacts"][name])
        for check in flow["required_checks"]:
            chunk = next(item for item in flow["chunks"] if item["path"] in flow["required_citation_sources"][check])
            original = experiment.read_skill(waiting["trial_id"], chunk["chunk_id"])
            citation = {"chunk_id": chunk["chunk_id"], "quote": original["content"].splitlines()[0]}
            checks.append(
                {
                    "id": check,
                    "case_id": case,
                    "status": "PASS",
                    "reason": "Protocol fixture only.",
                    "citations": [citation],
                    "evidence": [copy.deepcopy(evidence)],
                }
            )
    return {
        "schema_version": 1,
        "stage": flow["stage"],
        "request_id": flow["request_id"],
        "reads": reads,
        "checks": checks,
        "files": files,
        "config": {},
        "plan": "Protocol fixture only.",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def submit(experiment, waiting, value):
    write_json(Path(waiting["request_directory"]) / "response.json", value)
    return experiment.run()


def test_default_profile_reserves_all_four_requests(tmp_path):
    skill_fixture(tmp_path)
    value = specification()
    value["budget"]["generations"] = 3
    with pytest.raises(ContractError, match="skill stages"):
        Experiment.create(tmp_path / "bad", value, tmp_path)
    value["budget"]["generations"] = 4
    value["budget"]["reserved_tokens"] = 399
    with pytest.raises(ContractError, match="skill stages"):
        Experiment.create(tmp_path / "bad", value, tmp_path)


def test_original_chunks_are_bounded_and_no_closure_file_is_lost(tmp_path):
    skill_fixture(tmp_path)
    links = skill_links(tmp_path, [ASCEND_SKILL])
    routed = set()
    for stage in STAGES:
        for chunk in make_manifest(tmp_path, links, stage):
            routed.add(chunk["path"])
            lines = (tmp_path / chunk["path"]).read_text().splitlines(keepends=True)
            data = "".join(lines[chunk["start_line"] - 1 : chunk["end_line"]]).encode()
            assert digest(data) == chunk["sha256"]
            assert len(data) <= MAX_CHUNK_BYTES
    assert routed == {link["path"] for link in links}


def test_missing_skill_section_rejected_before_experiment_creation(tmp_path):
    skill_fixture(tmp_path)
    path = tmp_path / ASCEND_SKILL
    path.write_text(path.read_text().replace("## 10. Common Pitfalls", "## Renamed section"))
    with pytest.raises(ContractError, match="anchor missing"):
        Experiment.create(tmp_path / "run", specification(), tmp_path)
    assert not (tmp_path / "run").exists()


@pytest.mark.skipif(os.name != "posix", reason="WSL controller lock required")
def test_four_steps_and_recovery_preserve_all_evidence(tmp_path, monkeypatch):
    experiment = create(tmp_path, monkeypatch)
    waiting = experiment.run()
    for stage in STAGES:
        assert waiting["skill_stage"] == stage
        assert experiment.settings()["evaluations"] == (0 if stage in STAGES[:2] else 1)
        context = read_json(Path(waiting["request_directory"]) / "context.json")
        if stage != "design":
            assert context["skill_workflow"]["prior_files"]
        if stage in STAGES[2:]:
            campaign = Campaign(experiment.root / "campaigns" / experiment.trials()[0]["task_id"])
            assert campaign.report()["skill_gates"][0]["evidence_ref"] is None
        before = experiment.settings()
        assert experiment.run() == waiting  # No extra call on file-protocol resume.
        assert experiment.settings() == before
        waiting = submit(experiment, waiting, proposal(experiment, waiting))
    report = waiting
    assert report["status"] == "COMPLETE"
    assert report["usage"]["generations"] == 4
    assert report["usage"]["evaluations"] == 1
    assert report["groups"][0]["generation_reservations"] == 4
    assert report["groups"][0]["finished_generations"] == 4
    assert report["groups"][0]["reported_tokens_known"] == 120
    assert report["groups"][0]["within_budget_passes"] == 1
    assert report["trials"][0]["result"]["skill_workflow"]["status"] == "CONTRACT_CHECKED"
    assert len(report["trials"][0]["skill_workflow"]["steps"]) == 4
    assert report["campaigns"][experiment.trials()[0]["task_id"]]["skill_gates"][0]["evidence_ref"]
    assert experiment.run()["usage"] == report["usage"]
    # A historical stage's receipts, not just the final response, remain checked.
    step = report["trials"][0]["skill_workflow"]["steps"][0]
    experiment.objects.path(next(iter(step["deliveries"].values()))).write_bytes(b"corrupt")
    with pytest.raises(ContractError, match="corruption"):
        experiment.report()


@pytest.mark.skipif(os.name != "posix", reason="WSL controller lock required")
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_read",
        "fake_delivery",
        "missing_check",
        "fake_quote",
        "stale_request",
        "blocked",
        "wrong_hash",
        "bad_line",
        "duplicate_check",
        "foreign_case",
    ],
)
def test_bad_design_cannot_reach_evaluation(tmp_path, monkeypatch, mutation):
    experiment = create(tmp_path, monkeypatch)
    waiting = experiment.run()
    value = proposal(experiment, waiting)
    if mutation == "missing_read":
        value["reads"].pop()
    elif mutation == "fake_delivery":
        value["reads"][0]["delivery_id"] = "0" * 64
    elif mutation == "missing_check":
        value["checks"].pop()
    elif mutation == "fake_quote":
        value["checks"][0]["citations"][0]["quote"] = "invented guidance"
    elif mutation == "stale_request":
        value["request_id"] = "stale"
    elif mutation == "blocked":
        value["checks"][0]["status"] = "BLOCKED"
    elif mutation == "wrong_hash":
        value["checks"][0]["evidence"][0]["sha256"] = "0" * 64
    elif mutation == "bad_line":
        value["checks"][0]["evidence"][0]["end_line"] = 1000
    elif mutation == "duplicate_check":
        value["checks"].append(value["checks"][0])
    else:
        value["checks"][0]["case_id"] = "another-case"
    report = submit(experiment, waiting, value)
    assert report["usage"]["evaluations"] == 0
    assert report["trials"][0]["result"]["category"] == "INVALID_PROPOSAL"


@pytest.mark.skipif(os.name != "posix", reason="WSL controller lock required")
def test_reading_directly_is_not_a_framework_delivery(tmp_path, monkeypatch):
    experiment = create(tmp_path, monkeypatch)
    waiting = experiment.run()
    value = proposal(experiment, waiting)
    with experiment.db() as db:
        db.execute("DELETE FROM skill_reads")
    report = submit(experiment, waiting, value)
    assert report["usage"]["evaluations"] == 0
    assert "unverified skill delivery" in report["trials"][0]["result"]["error"]


@pytest.mark.skipif(os.name != "posix", reason="WSL controller lock required")
def test_post_lower_audit_rejects_foreign_artifact_and_blocks_device(tmp_path, monkeypatch):
    experiment = create(tmp_path, monkeypatch)
    waiting = experiment.run()
    for _ in range(2):
        waiting = submit(experiment, waiting, proposal(experiment, waiting))
    value = proposal(experiment, waiting)
    case = value["checks"][0]["case_id"]
    other = next(item for item in value["checks"] if item["case_id"] != case)
    value["checks"][0]["evidence"] = other["evidence"]
    report = submit(experiment, waiting, value)
    trial = report["trials"][0]
    assert trial["result"]["category"] == "INVALID_PROPOSAL"
    assert report["groups"][0]["within_budget_passes"] == 0
    campaign = Campaign(experiment.root / "campaigns" / trial["task_id"])
    campaign.authorize_device(3)
    for association in (None, trial["trial_id"], "wrong-trial"):
        job = {
            "mode": "device-smoke",
            "attempt_id": "not-launched",
            "candidate_id": trial["candidate_id"],
            "resource_limits": {"launches": 3},
            "experiment_trial_id": association,
        }
        with pytest.raises(ContractError, match="audit incomplete"):
            campaign.reserve(job)
    assert campaign.report()["device_quota"]["reserved_launches"] == 0


@pytest.mark.skipif(os.name != "posix", reason="WSL controller required")
def test_duplicate_candidate_preserves_each_routes_gate(tmp_path, monkeypatch):
    experiment = create(tmp_path, monkeypatch)
    waiting = experiment.run()
    for _ in range(2):
        waiting = submit(experiment, waiting, proposal(experiment, waiting))
    trial = experiment.trials()[0]
    campaign = Campaign(experiment.root / "campaigns" / trial["task_id"])
    campaign.authorize_device(3)
    bundle = campaign.bundle(trial["candidate_id"])
    direct_trial = "c" * 32
    campaign.add_candidate(bundle, skill_trial_id=direct_trial, require_skill_gate=False)
    job = {
        "mode": "device-smoke",
        "attempt_id": "fixture-reservation-no-worker",
        "candidate_id": trial["candidate_id"],
        "resource_limits": {"launches": 3},
        "experiment_trial_id": trial["trial_id"],
    }
    with pytest.raises(ContractError, match="audit incomplete"):
        campaign.reserve(job)
    # Same source does not let the unreviewed harness trial borrow the direct
    # route's exemption. The direct trial can still reserve its own job.
    job["experiment_trial_id"] = direct_trial
    campaign.reserve(job)
    assert campaign.attempts()[0]["attempt_id"] == job["attempt_id"]


@pytest.mark.skipif(os.name != "posix", reason="WSL controller required")
def test_finalization_recovers_after_gate_commit_without_new_calls(tmp_path, monkeypatch):
    experiment = create(tmp_path, monkeypatch)
    waiting = experiment.run()
    for _ in range(3):
        waiting = submit(experiment, waiting, proposal(experiment, waiting))
    original = experiment.finish

    def interrupted(*args, **kwargs):
        raise RuntimeError("fixture controller crash after gate commit")

    monkeypatch.setattr(experiment, "finish", interrupted)
    with pytest.raises(RuntimeError, match="controller crash"):
        submit(experiment, waiting, proposal(experiment, waiting))
    assert experiment.trials()[0]["state"] == "WORKFLOW_COMPLETE"
    before = experiment.settings()
    monkeypatch.setattr(experiment, "finish", original)
    report = experiment.run()
    assert report["usage"] == before
    campaign = report["campaigns"][experiment.trials()[0]["task_id"]]
    assert sum(event["kind"] == "SKILL_WORKFLOW_CHECKED" for event in campaign["events"]) == 1


@pytest.mark.skipif(os.name != "posix", reason="WSL controller lock required")
def test_explicit_links_ablation_keeps_legacy_response_protocol(tmp_path):
    skill_fixture(tmp_path)
    experiment = Experiment.create(tmp_path / "run", specification("links"), tmp_path)
    waiting = experiment.run()
    context = read_json(Path(waiting["request_directory"]) / "context.json")
    assert "skill_workflow" not in context
    assert context["skill_links"]


@pytest.mark.skipif(os.name != "posix", reason="WSL watchdog required")
def test_command_runtime_can_read_stages_while_controller_is_waiting(tmp_path, monkeypatch):
    experiment = create(tmp_path, monkeypatch)
    # Recreate a fresh experiment with a command fixture. Imports test helpers
    # solely to exercise protocol wiring, not to assess generation quality.
    script = tmp_path / "adapter.py"
    script.write_text(
        "import argparse,sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_skill_workflow import proposal\n"
        "from tilelang_harness.artifacts import read_json,write_json\n"
        "from tilelang_harness.experiments import Experiment\n"
        "p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--response');a=p.parse_args()\n"
        "c=read_json(Path(a.request));e=Experiment(c['skill_workflow']['experiment_directory'])\n"
        "v=proposal(e,{'trial_id':c['trial_id'],'request_directory':str(Path(a.request).parent)})\n"
        "print(c['skill_workflow']['stage']);write_json(Path(a.response),v)\n"
    )
    value = specification()
    value["arms"][0]["generator"].update(kind="command", command=[sys.executable, str(script)])
    experiment = Experiment.create(tmp_path / "command-run", value, tmp_path)
    report = experiment.run()
    assert report["groups"][0]["within_budget_passes"] == 1
    assert report["groups"][0]["finished_generations"] == 4
    assert report["groups"][0]["reported_tokens_known"] == 120
    steps = report["trials"][0]["skill_workflow"]["steps"]
    total = sum(step["generation_termination"]["wall_seconds"] for step in steps)
    assert report["groups"][0]["generator_wall_seconds_known"] == total > 0
    assert all(step["execution_ref"] for step in steps)


@pytest.mark.skipif(os.name != "posix", reason="WSL controller required")
def test_skill_status_only_advances_after_bounded_delivery(tmp_path, monkeypatch, capsys):
    from tilelang_harness.cli import main

    experiment = create(tmp_path, monkeypatch)
    waiting = experiment.run()
    args = ["--run-dir", str(experiment.root), "--trial-id", waiting["trial_id"]]
    before = experiment.skill_status(waiting["trial_id"])
    assert before["delivered_chunks"] == 0
    assert main(["skill-read", *args, "--chunk", before["next_chunk"]["chunk_id"]]) == 0
    delivered = __import__("json").loads(capsys.readouterr().out)
    assert delivered["delivery_id"]
    assert len(delivered["content"].encode("utf-8")) <= MAX_CHUNK_BYTES
    assert main(["skill-status", *args]) == 0
    after = __import__("json").loads(capsys.readouterr().out)
    assert after["delivered_chunks"] == 1
    assert after["next_chunk"]["chunk_id"] != before["next_chunk"]["chunk_id"]


@pytest.mark.skipif(
    os.name != "posix" or os.environ.get("TLH_RUN_SKILL_LOWERING") != "1",
    reason="opt-in real local WSL lowering, no vendor/NPU",
)
def test_original_skill_with_real_lowering_artifact_handoff(tmp_path):
    """Fixed baseline fixture, NOT an agent skill-understanding experiment."""
    from tilelang_harness.tasks.registry import load_task, source_path

    repo = Path(__file__).resolve().parents[2]
    value = specification()
    value["evaluation_timeout_s"] = 300
    value["budget"]["wall_seconds"] = 1200
    experiment = Experiment.create(tmp_path / "real-lower", value, repo)
    waiting = experiment.run()
    for stage in STAGES:
        assert waiting["skill_stage"] == stage
        response = proposal(experiment, waiting)
        if stage == "implementation":
            response["files"]["solution.py"] = source_path(load_task("vecadd").device_baseline).read_text()
        waiting = submit(experiment, waiting, response)
    assert waiting["trials"][0]["result"]["status"] == "PASS"
    task_id = experiment.trials()[0]["task_id"]
    attempts = waiting["campaigns"][task_id]["attempts"]
    assert len(attempts) == 1
    assert len(attempts[0]["result"]["builds"]) == len(load_task("vecadd").cases)
    assert attempts[0]["result"]["evidence"]["vendor_build"] == "NOT_RUN"
    write_json(experiment.root / "fixture-report.json", waiting)
