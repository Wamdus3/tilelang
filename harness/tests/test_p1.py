"""Real process/SQLite fault injection. These tests never impersonate NPU evidence."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tilelang_harness.artifacts import ArtifactStore, read_json, write_json
from tilelang_harness.bundle import freeze, materialize, seed
from tilelang_harness.cli import main
from tilelang_harness.contracts import ContractError
from tilelang_harness.evaluation import summarize_samples
from tilelang_harness.protocol import validate_job
from tilelang_harness.runner import alive, process_identity, replay, resume, run
from tilelang_harness.state import reduce_result
from tilelang_harness.store import Campaign, new_id
from tilelang_harness.tasks.registry import load_task

pytestmark = pytest.mark.skipif(os.name != "posix", reason="worker requires Linux/WSL")


@pytest.fixture
def setup(tmp_path):
    task = load_task("vecadd")
    source = tmp_path / "source"
    seed(task, source, "numpy")
    campaign = Campaign.create(tmp_path / "run", task)
    bundle = freeze(source, task, campaign.objects)
    campaign.add_candidate(bundle)
    return campaign, bundle, source


def replace_candidate(setup, code):
    campaign, _, source = setup
    (source / "solution.py").write_text(code)
    bundle = freeze(source, campaign.task(), campaign.objects)
    campaign.add_candidate(bundle)
    return bundle["candidate_id"]


def test_freeze_dependency_config_and_task_identity(setup, tmp_path):
    campaign, bundle, source = setup
    assert freeze(source, campaign.task(), campaign.objects) == bundle
    (source / "helper.py").write_text("TILE = 256\n")
    dependent = freeze(source, campaign.task(), campaign.objects)
    assert dependent["candidate_id"] != bundle["candidate_id"]
    assert freeze(source, load_task("rmsnorm"), campaign.objects)["candidate_id"] != dependent["candidate_id"]
    materialize(bundle, tmp_path / "copy", campaign.objects)
    assert not (tmp_path / "copy" / "helper.py").exists()
    assert (tmp_path / "copy" / "solution.py").read_bytes() == campaign.objects.get(bundle["files"]["solution.py"])


def test_freeze_does_not_import(setup, tmp_path):
    marker = tmp_path / "side-effect"
    replace_candidate(setup, f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    assert not marker.exists()


@pytest.mark.parametrize("mutation", ["symlink", "collision", "missing_entry", "config", "duplicate"])
def test_bad_bundle_rejected(setup, tmp_path, mutation):
    campaign, _, source = setup
    if mutation == "symlink":
        (source / "escape.py").symlink_to(tmp_path / "outside")
    elif mutation == "collision":
        (source / "Helper.py").touch()
        (source / "helper.py").touch()
    elif mutation == "missing_entry":
        (source / "solution.py").unlink()
    elif mutation == "config":
        manifest = read_json(source / "candidate.json")
        manifest["config"] = {"N": 1}
        write_json(source / "candidate.json", manifest)
    else:
        (source / "candidate.json").write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ContractError):
        freeze(source, campaign.task(), campaign.objects)


def test_cas_corruption_is_not_a_cache_hit(tmp_path):
    objects = ArtifactStore(tmp_path)
    key = objects.put(b"evidence")
    objects.path(key).write_bytes(b"modified")
    with pytest.raises(ContractError, match="corruption"):
        objects.get(key)
    with pytest.raises(ContractError, match="corruption"):
        objects.put(b"evidence")


def test_host_worker_replay_and_idempotent_commit(setup):
    campaign, bundle, source = setup
    result = run(campaign, bundle["candidate_id"], "host", 20)
    assert result["status"] == "PASS"
    assert set(result["evidence"].values()) == {"NOT_RUN"}
    rows = campaign.attempts()
    assert rows[0]["status"] == "FINISHED"
    before = len(campaign.report()["events"])
    assert campaign.finish(rows[0]["attempt_id"], result) is False
    assert len(campaign.report()["events"]) == before
    changed = dict(result, error="conflicting result")
    with pytest.raises(ContractError, match="conflicting duplicate"):
        campaign.finish(rows[0]["attempt_id"], changed)
    (source / "solution.py").write_text("raise RuntimeError('mutable source must not be replayed')\n")
    fresh = replay(campaign, result["attempt_id"])
    assert fresh["status"] == "PASS"
    assert fresh["candidate_id"] == result["candidate_id"]
    assert fresh["evaluation_id"] != result["evaluation_id"]
    assert fresh["job_id"] != result["job_id"]
    assert resume(campaign) == []
    assert not campaign.report()["eligible_for_promotion"]
    for row in campaign.report()["attempts"]:
        assert row["decision"]["phase"] == "FROZEN"
        assert not row["decision"]["eligible_for_promotion"]
        for key in row["result"]["artifacts"].values():
            campaign.objects.get(key)


def test_wrong_output_is_correctness_failure(setup):
    candidate_id = replace_candidate(setup, "def run(inputs, epsilon):\n    return {}\n")
    result = run(setup[0], candidate_id, "host", 20)
    assert result["status"] == "FAIL"
    assert result["category"] == "HOST_CHECK"


@pytest.mark.parametrize(
    "code,category",
    [
        ("raise RuntimeError('import failed')", "IMPORT_ERROR"),
        ("import os\nos._exit(73)", "WORKER_EXIT"),
    ],
)
def test_import_failure_and_abrupt_exit(setup, code, category):
    candidate_id = replace_candidate(setup, code)
    result = run(setup[0], candidate_id, "host", 20)
    assert result["status"] == "ERROR"
    assert result["category"] == category
    assert set(result["evidence"].values()) == {"NOT_RUN"}


def test_import_deadline_terminates_descendants(setup):
    code = """
import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
Path('child.pid').write_text(str(child.pid))
while True:
    time.sleep(0.1)
"""
    candidate_id = replace_candidate(setup, code)
    result = run(setup[0], candidate_id, "host", 5)
    assert result["category"] == "TIMEOUT"
    assert result["stage"] == "candidate_import"
    work = setup[0].root / "work" / result["attempt_id"]
    child = process_identity(int((work / "child.pid").read_text()))
    assert child is None or not alive(child)


def wait_file(path, timeout=15):
    until = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= until:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.05)


def test_controller_crash_keeps_watchdog_and_recovery_does_not_rerun(setup, tmp_path):
    marker = tmp_path / "imported"
    candidate_id = replace_candidate(
        setup,
        f"""
from pathlib import Path
import time
with Path({str(marker)!r}).open('a') as f:
    f.write('once\\n')
while True:
    time.sleep(0.1)
""",
    )
    campaign = setup[0]
    script = (
        "from pathlib import Path; from tilelang_harness.store import Campaign; "
        "from tilelang_harness.runner import run; "
        f"run(Campaign(Path({str(campaign.root)!r})), {candidate_id!r}, 'host', 7)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    parent = subprocess.Popen(
        [sys.executable, "-c", script], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    try:
        wait_file(marker)
        row = campaign.attempts()[0]
        identity = campaign.objects.json(row["process_ref"])
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=3)
        assert alive(identity)
        work = campaign.root / "work" / row["attempt_id"]
        # Detached watchdog must enforce the deadline even before any resume command.
        wait_file(work / "termination.json")
        assert read_json(work / "termination.json")["reason"] == "TIMEOUT"
        recovered = resume(campaign)
        assert recovered[0]["category"] == "TIMEOUT"
        assert marker.read_text() == "once\n"
        assert len(campaign.attempts()) == 1
        assert resume(campaign) == []
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if campaign.attempts() and campaign.attempts()[0]["status"] != "FINISHED":
            resume(campaign)
        if parent.stderr:
            parent.stderr.close()


def test_budget_and_live_attempt_prevent_duplicate_scheduling(setup):
    campaign, bundle, _ = setup
    result = run(campaign, bundle["candidate_id"], "host", 20)
    job = dict(campaign.objects.json(campaign.attempts()[0]["job_ref"]), attempt_id=new_id(), job_id=new_id())
    job["created"] = time.time()
    campaign.reserve(job)
    with pytest.raises(ContractError, match="unfinished attempt"):
        run(campaign, bundle["candidate_id"], "host", 20)
    assert resume(campaign)[0]["status"] == "STARTING"
    assert len(campaign.attempts()) == 2
    # Simulate a crash before spawn/release, then recover the reservation as interrupted.
    # A very short reservation deadline avoids artificial clock sleeps.
    recovery = Campaign.create(campaign.root.parent / "recovery", campaign.task(), max_attempts=1)
    recovery.add_candidate(bundle)
    old_job = dict(job, created=time.time() - 100)
    recovery.reserve(old_job)
    assert resume(recovery)[0]["category"] == "INTERRUPTED"
    with pytest.raises(ContractError, match="budget exhausted"):
        recovery.reserve(dict(old_job, attempt_id=new_id(), job_id=new_id()))
    assert result["status"] == "PASS"


@pytest.mark.parametrize(
    "key,value",
    [
        ("task_digest", "wrong"),
        ("candidate_id", "wrong"),
        ("attempt_id", "wrong"),
        ("runner_digest", "wrong"),
        ("case_ids", []),
        ("mode", "lower"),
    ],
)
def test_mismatched_results_never_committed(setup, key, value):
    campaign, bundle, _ = setup
    result = run(campaign, bundle["candidate_id"], "host", 20)
    job = campaign.objects.json(campaign.attempts()[0]["job_ref"])
    with pytest.raises(ContractError, match="identity mismatch"):
        reduce_result(job, dict(result, **{key: value}))


def test_partial_or_host_device_evidence_rejected(setup):
    campaign, bundle, _ = setup
    result = run(campaign, bundle["candidate_id"], "host", 20)
    job = campaign.objects.json(campaign.attempts()[0]["job_ref"])
    with pytest.raises(ContractError, match="complete, ordered"):
        reduce_result(job, dict(result, cases=result["cases"][:-1]))
    with pytest.raises(ContractError, match="host diagnostics"):
        reduce_result(job, dict(result, evidence=dict.fromkeys(result["evidence"], "PASS")))


def test_cli_roundtrip_and_no_device_mode(tmp_path, capsys):
    source, root = tmp_path / "seed", tmp_path / "campaign"
    assert main(["seed", "vecadd", "--destination", str(source), "--abi", "numpy"]) == 0
    assert main(["init", "vecadd", "--run-dir", str(root), "--max-attempts", "1"]) == 0
    capsys.readouterr()
    assert main(["submit", str(source), "--run-dir", str(root)]) == 0
    candidate_id = json.loads(capsys.readouterr().out)["candidate_id"]
    assert main(["run", candidate_id, "--run-dir", str(root), "--mode", "host"]) == 0
    capsys.readouterr()
    report = tmp_path / "report.json"
    assert main(["report", "--run-dir", str(root), "--output", str(report)]) == 0
    assert read_json(report)["attempts"][0]["result"]["status"] == "PASS"
    with pytest.raises(SystemExit):
        main(["run", candidate_id, "--run-dir", str(root), "--mode", "device"])


def test_controller_lease_and_mode_contract(setup):
    campaign, bundle, _ = setup
    with campaign.lock(), pytest.raises(ContractError, match="another controller"):
        run(campaign, bundle["candidate_id"], "host")
    with pytest.raises(ContractError, match="requires candidate ABI"):
        run(campaign, bundle["candidate_id"], "lower")
    with pytest.raises(ContractError, match="timeout"):
        run(campaign, bundle["candidate_id"], "host", float("inf"))


def device_job(setup):
    campaign, _, source = setup
    manifest = read_json(source / "candidate.json")
    manifest.update(abi="primfunc", entrypoint="solution.py:build")
    write_json(source / "candidate.json", manifest)
    bundle = freeze(source, campaign.task(), campaign.objects)
    campaign.add_candidate(bundle)
    from tilelang_harness.compiler import runner_identity
    from tilelang_harness.contracts import content_id

    task = campaign.task()
    return {
        "schema_version": 1,
        "job_id": new_id(),
        "attempt_id": new_id(),
        "candidate_id": bundle["candidate_id"],
        "task_digest": task.task_digest,
        "task": task.to_dict(),
        "bundle": bundle,
        "mode": "device-smoke",
        "case_ids": [task.cases[0].case_id],
        "device_index": 0,
        "measure": False,
        "created": time.time(),
        "timeout_s": 20,
        "runner_digest": content_id(runner_identity()),
    }


def test_device_requires_explicit_single_case_without_launching(setup):
    job = device_job(setup)
    assert validate_job(job) == job
    with pytest.raises(ContractError, match="one explicit"):
        run(setup[0], job["candidate_id"], "device-smoke")
    with pytest.raises(ContractError, match="exactly one"):
        validate_job(dict(job, case_ids=[case.case_id for case in setup[0].task().cases]))
    with pytest.raises(ContractError, match="device index"):
        validate_job(dict(job, device_index=True))
    with pytest.raises(ContractError, match="job.*unknown"):
        validate_job(dict(job, extra="ignored options are forbidden"))
    assert setup[0].attempts() == []


def test_missing_device_environment_is_incomplete_not_pass(setup, tmp_path, monkeypatch):
    from tilelang_harness import device, worker

    job = device_job(setup)
    work = tmp_path / "worker"
    work.mkdir()
    materialize(job["bundle"], work / "candidate", setup[0].objects)
    write_json(work / "job.json", job)
    write_json(work / "release.json", {"attempt_id": job["attempt_id"]})

    def unavailable(index):
        raise ImportError("test: NPU environment missing")

    monkeypatch.setattr(device, "environment", unavailable)
    worker.execute(work)
    result = read_json(work / "result.json")
    baseline_report = read_json(work / "artifacts" / "host-baseline.json")
    assert [case["case_id"] for case in baseline_report["cases"]] == job["case_ids"]
    assert result["status"] == "INCOMPLETE"
    assert result["category"] == "ENVIRONMENT_UNAVAILABLE"
    assert set(result["evidence"].values()) == {"NOT_RUN"}
    assert not reduce_result(job, result)["eligible_for_promotion"]


def test_device_lease_serializes_campaigns_and_quarantine_requires_health_ack(tmp_path, monkeypatch):
    from tilelang_harness import device

    monkeypatch.setattr(device, "lease_directory", lambda: tmp_path)
    with device.lease(), pytest.raises(ContractError, match="DEVICE_BUSY"), device.lease():
        pytest.fail("second lease acquired")
    device.quarantine("TIMEOUT", "attempt")
    with pytest.raises(ContractError, match="DEVICE_QUARANTINED"), device.lease():
        pytest.fail("quarantined device acquired")


@pytest.mark.parametrize("samples", [[1, 2], [1, 0, 1], [1, -1, 1], [1, float("nan"), 1], [1, float("inf"), 1]])
def test_incomplete_or_invalid_timing_never_scores(samples):
    with pytest.raises(ContractError, match="finite positive"):
        summarize_samples(samples, 3)


def test_valid_timing_retains_raw_samples_and_noise():
    summary = summarize_samples([1.0, 1.1, 2.0], 3)
    assert summary["samples_ms"] == [1.0, 1.1, 2.0]
    assert summary["median_ms"] == 1.1
    assert summary["relative_range"] > 0.1


def test_report_rejects_corruption_in_nested_evidence(setup):
    campaign, bundle, _ = setup
    result = run(campaign, bundle["candidate_id"], "host", 20)
    ref = next(iter(result["artifacts"].values()))
    campaign.objects.path(ref).write_bytes(b"corrupted evidence")
    with pytest.raises(ContractError, match="corruption"):
        campaign.report()


def test_invalid_worker_result_is_retained_but_never_accepted_as_pass(setup):
    candidate_id = replace_candidate(
        setup,
        """
from pathlib import Path
import os
Path('result.json').write_text('{}')
os._exit(0)
""",
    )
    result = run(setup[0], candidate_id, "host", 20)
    assert result["category"] == "RESULT_PROTOCOL_ERROR"
    assert result["status"] == "ERROR"
    assert setup[0].objects.get(result["rejected_result"]) == b"{}"
    assert setup[0].attempts()[0]["status"] == "FINISHED"
    assert resume(setup[0]) == []
