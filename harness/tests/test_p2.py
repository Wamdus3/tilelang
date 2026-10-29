"""Generation-protocol fixtures, not LLM benchmarks; never run NPU kernels."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tilelang_harness.artifacts import read_json, write_json
from tilelang_harness.budget import BudgetExhausted, ResourceBudget
from tilelang_harness.bundle import freeze, seed
from tilelang_harness.contracts import ContractError
from tilelang_harness.device import Invocation
from tilelang_harness.experiments import Experiment
from tilelang_harness.generation import parse_response, skill_links, verify_links
from tilelang_harness.protocol import validate_job
from tilelang_harness.runner import alive, process_identity
from tilelang_harness.store import Campaign, new_id
from tilelang_harness.tasks.registry import load_task

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Linux watchdog/flock required")

CODE = "def run(inputs, epsilon):\n    return {'out': inputs['a'] + inputs['b']}\n"


def proposal(code=CODE):
    return {
        "schema_version": 1,
        "files": {"solution.py": code},
        "config": {},
        "plan": "Protocol-test fixture only, no LLM invocation.",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def spec(command=None, arms=1, replicates=1, repairs=0):
    return {
        "schema_version": 1,
        "name": "protocol-fixture-not-an-LLM-study",
        "tasks": ["vecadd"],
        "arms": [
            {
                "name": f"arm-{i}",
                "skill_links": [],
                "generator": {
                    "kind": "command" if command else "file",
                    "command": command or [],
                    "model": "fixture-not-a-model",
                    "parameters": {},
                    "timeout_s": 10,
                    "token_reservation": 100,
                },
            }
            for i in range(arms)
        ],
        "replicates": replicates,
        "seed": 42,
        "max_repairs": repairs,
        "evaluation_mode": "host",
        "evaluation_timeout_s": 20,
        "budget": {
            "generations": arms * replicates * (repairs + 1),
            "reserved_tokens": arms * replicates * (repairs + 1) * 100,
            "evaluations": arms * replicates * (repairs + 1),
            "wall_seconds": 120,
        },
    }


def test_skill_is_only_linked_and_reference_closure_is_locked(tmp_path):
    (tmp_path / "SKILL.md").write_text("# Original\n[details](ref.md)\n")
    (tmp_path / "ref.md").write_text("Original reference\n")
    value = spec()
    value["arms"][0]["skill_links"] = ["SKILL.md"]
    experiment = Experiment.create(tmp_path / "run", value, tmp_path)
    waiting = experiment.run()
    work = Path(waiting["request_directory"])
    assert "Original reference" not in (work / "context.md").read_text()
    assert not (work / "SKILL.md").exists()
    links = read_json(work / "context.json")["skill_links"]
    assert {link["path"] for link in links} == {"SKILL.md", "ref.md"}
    (tmp_path / "ref.md").write_text("Changed reference\n")
    with pytest.raises(ContractError, match="skill changed"):
        experiment.run()


def test_skills_reject_missing_and_escaping_links(tmp_path):
    (tmp_path / "SKILL.md").write_text("[outside](../outside.md)")
    with pytest.raises(ContractError, match="escaping"):
        skill_links(tmp_path, ["SKILL.md"])
    with pytest.raises(ContractError):
        verify_links(tmp_path, [{"path": "missing.md", "sha256": "0" * 64}])


def test_file_adapter_repairs_and_preserves_failed_trial(tmp_path):
    experiment = Experiment.create(tmp_path / "run", spec(repairs=1), tmp_path)
    waiting = experiment.run()
    write_json(Path(waiting["request_directory"]) / "response.json", proposal("def run(inputs, epsilon): return {}\n"))
    waiting = experiment.run()
    context = read_json(Path(waiting["request_directory"]) / "context.json")
    assert context["role"] == "diagnoser_implementer"
    assert context["feedback"]["cases"][0]["invocations"][0]["errors"]
    assert context["previous_candidate"]["candidate_id"]
    write_json(Path(waiting["request_directory"]) / "response.json", proposal())
    report = experiment.run()
    assert report["status"] == "COMPLETE"
    assert report["usage"]["generations"] == report["usage"]["evaluations"] == 2
    group = report["groups"][0]
    assert group["first_attempt_passes"] == 0
    assert group["within_budget_passes"] == 1
    assert group["reported_tokens_known"] == 60
    assert group["failure_categories"] == {"HOST_CHECK": 1}
    assert report["trials"][1]["parent"] == report["trials"][0]["trial_id"]
    assert report["device_execution"] == "NOT_RUN"


def test_duplicate_sources_are_separate_generations_and_evaluations(tmp_path):
    experiment = Experiment.create(tmp_path / "run", spec(arms=2, replicates=2), tmp_path)
    for _ in range(4):
        waiting = experiment.run()
        write_json(Path(waiting["request_directory"]) / "response.json", proposal())
    report = experiment.run()
    assert report["usage"]["generations"] == report["usage"]["evaluations"] == 4
    assert len({row["candidate_id"] for row in report["trials"]}) == 1
    assert len({row["attempt_id"] for row in report["trials"]}) == 4
    assert all(group["first_attempt_success_rate"] == 1 for group in report["groups"])
    before = experiment.settings()
    assert experiment.run()["usage"] == before  # No new model call or evaluation on resume.


@pytest.mark.parametrize("mutation", ["path", "tokens", "schema", "nan"])
def test_bad_generator_response_is_rejected(tmp_path, mutation):
    value = proposal()
    if mutation == "path":
        value["files"]["../escape.py"] = "bad"
    elif mutation == "tokens":
        value["usage"]["input_tokens"] = 1000
    elif mutation == "schema":
        value["extra"] = "ignored fields forbidden"
    else:
        value["config"]["tile"] = float("nan")
    path = tmp_path / "response.json"
    if mutation == "nan":
        path.write_text('{"invalid":NaN}')
    else:
        write_json(path, value)
    with pytest.raises(ContractError):
        parse_response(path, 100)


def test_budget_exhaustion_and_unknown_tokens_do_not_become_success(tmp_path):
    value = spec(repairs=1)
    value["budget"]["generations"] = 1
    experiment = Experiment.create(tmp_path / "run", value, tmp_path)
    waiting = experiment.run()
    bad = proposal("def run(inputs, epsilon): return {}\n")
    bad["usage"] = {"input_tokens": None, "output_tokens": None}
    write_json(Path(waiting["request_directory"]) / "response.json", bad)
    report = experiment.run()
    assert report["status"] == "COMPLETE"
    assert report["usage"]["generations"] == 1
    assert report["groups"][0]["responses_with_unknown_usage"] == 1
    assert report["groups"][0]["within_budget_passes"] == 0
    assert report["trials"][-1]["result"]["category"] == "GENERATION_BUDGET_EXHAUSTED"


def command_adapter(tmp_path, prefix=""):
    script = tmp_path / "adapter.py"
    script.write_text(
        "import argparse, json\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--response'); a=p.parse_args()\n"
        + prefix
        + "\nPath(a.response).write_text("
        + repr(__import__("json").dumps(proposal()))
        + ")\n"
    )
    return [sys.executable, str(script)]


def test_command_generator_and_evaluator_roundtrip(tmp_path):
    experiment = Experiment.create(tmp_path / "run", spec(command_adapter(tmp_path)), tmp_path)
    report = experiment.run()
    assert report["status"] == "COMPLETE"
    assert report["groups"][0]["within_budget_success_rate"] == 1
    assert report["groups"][0]["generator_wall_seconds_known"] > 0
    assert report["usage"]["tokens"] == 100
    assert report["trials"][0]["context"]["skill_links"] == []


def test_generator_preserves_virtualenv_and_locks_executable(tmp_path):
    from tilelang_harness.generation import parse_generator

    generator = spec([sys.executable, "-c", "pass"])["arms"][0]["generator"]
    locked = parse_generator(generator, tmp_path)
    assert locked["command"][0] == str(Path(sys.executable).absolute())
    assert str(Path(sys.executable).resolve()) in locked["command_files"]


def test_generator_logs_are_content_addressed(tmp_path):
    experiment = Experiment.create(tmp_path / "run", spec(command_adapter(tmp_path, "print('fixture log')")), tmp_path)
    report = experiment.run()
    row = report["trials"][0]
    (experiment.root / "requests" / row["trial_id"] / "termination.json").write_text('{"status":"fake"}')
    assert experiment.report()["trials"][0]["generation_termination"]["status"] == "PASS"
    checksum = row["generation_logs"]["stdout.log"]
    experiment.objects.path(checksum).write_bytes(b"corrupt")
    with pytest.raises(ContractError, match="corruption"):
        experiment.report()


def test_command_timeout_kills_descendants_and_keeps_failure(tmp_path):
    marker = tmp_path / "child.pid"
    command = command_adapter(
        tmp_path,
        "import subprocess,sys,time\n"
        "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])\n"
        f"Path({str(marker)!r}).write_text(str(c.pid))\ntime.sleep(120)",
    )
    value = spec(command)
    value["arms"][0]["generator"]["timeout_s"] = 10
    experiment = Experiment.create(tmp_path / "run", value, tmp_path)
    report = experiment.run()
    assert report["trials"][0]["generation_termination"]["status"] == "TIMEOUT"
    assert report["trials"][0]["result"]["category"] == "GENERATOR_FAILED"
    assert report["usage"]["generations"] == 1
    assert not alive(process_identity(int(marker.read_text())) or {"pid": -1})


def test_controller_crash_does_not_repeat_generator(tmp_path):
    marker = tmp_path / "model-calls.txt"
    command = command_adapter(
        tmp_path, f"with Path({str(marker)!r}).open('a') as f: f.write('call\\n')\nimport time\ntime.sleep(2)"
    )
    experiment = Experiment.create(tmp_path / "run", spec(command), tmp_path)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    controller = subprocess.Popen(
        [sys.executable, "-m", "tilelang_harness", "experiment-run", "--run-dir", str(experiment.root)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        until = time.monotonic() + 10
        while not marker.exists():
            assert time.monotonic() < until
            time.sleep(0.025)
        os.kill(controller.pid, signal.SIGKILL)
        controller.wait(timeout=3)
        report = experiment.run()
        assert report["groups"][0]["within_budget_passes"] == 1
        assert report["usage"]["generations"] == 1
        assert marker.read_text() == "call\n"
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait()


def test_corrupted_experiment_evidence_is_rejected(tmp_path):
    experiment = Experiment.create(tmp_path / "run", spec(), tmp_path)
    waiting = experiment.run()
    trial = experiment.trials()[0]
    experiment.objects.path(trial["context_ref"]).write_bytes(b"modified")
    with pytest.raises(ContractError, match="corruption"):
        experiment.report()
    assert waiting["status"] == "WAITING_RESPONSE"


def test_experiment_cannot_auto_run_device_jobs(tmp_path):
    value = spec()
    value["evaluation_mode"] = "device-smoke"
    with pytest.raises(ContractError, match="explicit single-case"):
        Experiment.create(tmp_path / "run", value, tmp_path)


def test_launch_limit_counts_failed_calls_and_never_dispatches_fourth(tmp_path):
    budget = ResourceBudget({"launches": 3, "vendor_builds": 1}, tmp_path / "resources.json")
    dispatched = []

    def kernel():
        dispatched.append(1)
        if len(dispatched) == 2:
            raise RuntimeError("vendor call failed")

    invocation = Invocation(None, kernel, SimpleNamespace(device=lambda _: "npu:0"), 0, budget)
    invocation.tensors = []
    invocation.call()
    with pytest.raises(RuntimeError):
        invocation.call()
    invocation.call()
    with pytest.raises(BudgetExhausted):
        invocation.call()
    assert len(dispatched) == 3
    assert read_json(tmp_path / "resources.json")["used"]["launches"] == 3


def test_campaign_quota_is_conservative_and_persistent(tmp_path):
    task = load_task("vecadd")
    source = tmp_path / "source"
    seed(task, source)
    campaign = Campaign.create(tmp_path / "campaign", task, device_launches=3)
    bundle = freeze(source, task, campaign.objects)
    campaign.add_candidate(bundle)
    from tilelang_harness.compiler import runner_identity
    from tilelang_harness.contracts import content_id

    job = {
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
        "resource_limits": {"launches": 3, "vendor_builds": 1},
    }
    validate_job(job)
    campaign.reserve(job)
    assert Campaign(campaign.root).report()["device_quota"]["reserved_launches"] == 3
    with pytest.raises(ContractError, match="already reserved"):
        campaign.authorize_device(2)
    from tilelang_harness.worker import empty_result

    campaign.finish(job["attempt_id"], empty_result(job))
    another = dict(job, job_id=new_id(), attempt_id=new_id())
    with pytest.raises(ContractError, match="device launch budget"):
        campaign.reserve(another)
    campaign.authorize_device(6)
    campaign.reserve(another)
    assert campaign.report()["device_quota"]["reserved_launches"] == 6


def test_event_measurement_reserves_outside_timed_region():
    from tilelang_harness.evaluation import event_sample

    order = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            order.append("event")

        def synchronize(self):
            order.append("sync")

        def elapsed_time(self, other):
            return 1.0

    invocation = SimpleNamespace(
        budget=SimpleNamespace(consume=lambda _: order.append("reserve")),
        kernel=lambda: order.append("kernel"),
        tensors=[],
        torch=SimpleNamespace(npu=SimpleNamespace(Event=Event)),
    )
    sample = event_sample(invocation, 0, 0)
    assert sample["call_samples_ms"] == [1.0]
    assert order == ["reserve", "event", "kernel", "event", "sync"]


@pytest.mark.parametrize("value", [None, [], 3])
def test_job_requires_object(value):
    with pytest.raises(ContractError, match="job must be an object"):
        validate_job(value)


def test_killed_generator_watchdog_does_not_leave_adapter_running(tmp_path):
    marker = tmp_path / "adapter.pid"
    command = command_adapter(
        tmp_path, f"import os,time\nPath({str(marker)!r}).write_text(str(os.getpid()))\ntime.sleep(120)"
    )
    experiment = Experiment.create(tmp_path / "run", spec(command), tmp_path)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    controller = subprocess.Popen(
        [sys.executable, "-m", "tilelang_harness", "experiment-run", "--run-dir", str(experiment.root)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        until = time.monotonic() + 10
        while not marker.exists():
            assert time.monotonic() < until
            time.sleep(0.025)
        trial = experiment.trials()[0]
        identity = experiment.objects.json(trial["process_ref"])
        os.kill(identity["pid"], signal.SIGKILL)
        controller.wait(timeout=10)
        report = experiment.report()
        assert report["usage"]["generations"] == 1
        assert report["trials"][0]["result"]["category"] == "GENERATOR_FAILED"
        assert not alive(process_identity(int(marker.read_text())) or {"pid": -1})
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait()
