"""Fixtures for real Ascend cache tests; subprocesses isolate native memory caches."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def cache_lab(tmp_path_factory, record_testsuite_property):
    if sys.platform != "linux":
        pytest.skip("Cache filesystem probes require Linux")
    pytest.importorskip("torch_npu")
    work = tmp_path_factory.mktemp("ascend-cache")
    record_testsuite_property("cache_artifacts", str(work))
    tests = Path(__file__).resolve().parent
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tests.parents[2]), environment.get("PYTHONPATH")]))
    environment.update(
        TILELANG_AUTO_TUNING_CPU_COUNTS="2",
        TILELANG_AUTO_TUNING_MAX_CPU_COUNT="2",
        TILELANG_DISABLE_CACHE="0",
        TILELANG_AUTO_TUNING_DISABLE_CACHE="0",
    )

    def run_phase(cache, output, mode="tune", binary_failure=False):
        output.mkdir(parents=True)
        command = [
            sys.executable,
            str(tests / "_ascend_cache_worker.py"),
            "--mode",
            mode,
            "--output",
            str(output / "report"),
            "--cache-dir",
            str(cache),
        ]
        if binary_failure:
            command.append("--expect-failure")
        env = dict(environment, TILELANG_CACHE_DIR=str(cache), HOME=str(output))
        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
            process = subprocess.run(command, env=env, cwd=work, stdout=stdout, stderr=stderr, timeout=300)
        assert process.returncode == 0, f"{command}: returned {process.returncode}; inspect {output}/stderr.log"
        return json.loads((output / "report/result.json").read_text())

    return SimpleNamespace(work=work, run_phase=run_phase)


@pytest.fixture(scope="session")
def real_baseline(cache_lab):
    cache_lab.cache = cache_lab.work / "seed-cache"
    reports = [
        cache_lab.run_phase(cache_lab.cache, cache_lab.work / phase, mode)
        for phase, mode in [("cold", "tune"), ("fresh-process", "tune"), ("direct-compile", "direct")]
    ]
    cache_lab.relative_roots = {kind: Path(path).relative_to(cache_lab.cache) for kind, path in reports[0]["cache_roots"].items()}
    return reports


@pytest.fixture(scope="session")
def cache_mounts(cache_lab, real_baseline):
    """Mount only disposable test directories; all NPU work stays on the host."""
    image = os.environ.get("TILELANG_CACHE_TEST_IMAGE")
    if not image or not shutil.which("docker"):
        pytest.skip("Cross-mount cases require Docker and TILELANG_CACHE_TEST_IMAGE naming an existing local image")
    image_id = subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
    data = cache_lab.work / "container-data"
    data.mkdir()
    scenarios = [f"{layer}-{mount}" for layer in ("kernel", "binary", "autotune") for mount in ("bind", "tmpfs")]
    name = "tilelang-cache-test-" + uuid.uuid4().hex[:12]
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,src={data},dst={data}",
        "--entrypoint",
        "/bin/sleep",
    ]
    for scenario in scenarios:
        for relative in cache_lab.relative_roots.values():
            (data / scenario / "cache" / relative).mkdir(parents=True, exist_ok=True)
        layer, mount = scenario.split("-")
        final = data / scenario / "cache" / cache_lab.relative_roots[layer]
        if mount == "bind":
            backing = data / scenario / "separate-final"
            backing.mkdir()
            command += ["--mount", f"type=bind,src={backing},dst={final}"]
        else:
            command += ["--tmpfs", f"{final}:rw,uid={os.getuid()},gid={os.getgid()},mode=0700"]
    command += [image_id, "3600"]
    dump(cache_lab.work / "container-command.json", command)
    with (cache_lab.work / "container.log").open("w") as log:
        container = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    base = None
    try:
        pid = None
        for _ in range(100):
            inspection = subprocess.run(["docker", "inspect", name, "--format", "{{.State.Pid}}"], capture_output=True, text=True)
            if inspection.returncode == 0 and int(inspection.stdout.strip()) > 0:
                pid = int(inspection.stdout.strip())
                break
            assert container.poll() is None, "Filesystem container exited during startup"
            time.sleep(0.1)
        assert pid, "Filesystem container startup timed out"
        # Matching absolute paths keep TVM realpath-normalized libraries loadable.
        base = Path(f"/proc/{pid}/root") / data.relative_to("/")
        mountinfo = Path(f"/proc/{pid}/mountinfo")
        assert base.is_dir(), "Cannot access the same-UID filesystem container"
        (cache_lab.work / "container-mountinfo.txt").write_text(mountinfo.read_text())
        dump(cache_lab.work / "container.json", dict(name=name, pid=pid, image=image_id))
        yield base
    finally:
        try:
            if base is not None:
                for layer in ("kernel", "binary", "autotune"):
                    scenario = f"{layer}-tmpfs"
                    source = base / scenario / "cache" / cache_lab.relative_roots[layer]
                    if source.exists():
                        shutil.copytree(source, cache_lab.work / scenario / "tmpfs-final-snapshot", dirs_exist_ok=True)
        finally:
            subprocess.run(["docker", "stop", "--time", "2", name], capture_output=True, timeout=30)
            container.wait(timeout=30)
