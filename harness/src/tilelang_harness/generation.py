"""Generator contracts and links to existing skills; no model SDK dependency."""

import re
import shutil
from pathlib import Path

from .artifacts import digest, read_json, write_json
from .bundle import MAX_BUNDLE_BYTES, relative_name
from .contracts import FFT_WORKLOAD_TASKS, ContractError, fields, integer, number, text


def skill_links(root, names):
    if not isinstance(names, list) or len(set(names)) != len(names):
        raise ContractError("skill_links must be a unique list of repository-relative paths")
    root = root.resolve()
    result, visited, pending = [], set(), list(names)
    while pending:
        name = pending.pop(0)
        if name in visited:
            continue
        visited.add(name)
        if len(visited) > 256:
            raise ContractError("skill reference closure exceeds 256 files")
        path = (root / relative_name(name)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ContractError("skill link escapes repository or is missing")
        result.append({"path": name, "sha256": digest(path.read_bytes())})
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.strip("<>").split("#", 1)[0]
            if not target or ":" in target or not target.endswith(".md"):
                continue
            dependency = (path.parent / target).resolve()
            if not dependency.is_relative_to(root) or not dependency.is_file():
                raise ContractError(f"missing or escaping linked skill reference: {target}")
            pending.append(dependency.relative_to(root).as_posix())
    return sorted(result, key=lambda item: item["path"])


def verify_links(root, links):
    if skill_links(root, [link["path"] for link in links]) != links:
        raise ContractError("linked skill changed; create a new experiment instead of mixing guidance versions")


def parse_generator(value, root):
    fields(value, {"kind", "command", "model", "parameters", "timeout_s", "token_reservation"}, "generator")
    if value["kind"] not in ("command", "file"):
        raise ContractError("generator kind must be command or file")
    command = value["command"]
    if not isinstance(command, list) or any(not isinstance(arg, str) or not arg for arg in command):
        raise ContractError("generator command must be an argv list")
    if (value["kind"] == "command") != bool(command):
        raise ContractError("only command generators require a nonempty command")
    command = list(command)
    if command:
        executable = Path(command[0])
        if not executable.is_absolute() and (root / executable).is_file():
            executable = root / executable
        else:
            executable = Path(shutil.which(command[0]) or command[0])
        if not executable.is_file():
            raise ContractError("generator executable does not exist")
        # Keep a venv's executable symlink: resolving it changes Python's sys.prefix.
        command[0] = str(executable.absolute())
    text(value["model"], "model")
    if not isinstance(value["parameters"], dict):
        raise ContractError("model parameters must be an object")
    timeout = number(value["timeout_s"], "generator timeout", 0.05)
    if timeout > 86400:
        raise ContractError("generator timeout exceeds one day")
    integer(value["token_reservation"], "token reservation")
    # Local script arguments are locked by content when they name existing files.
    files = {}
    for arg in command:
        path = Path(arg)
        path = path if path.is_absolute() else root / path
        if path.is_file():
            files[str(path.resolve())] = digest(path.read_bytes())
    return {**value, "command": command, "command_files": files}


def verify_generator(generator):
    for name, checksum in generator["command_files"].items():
        if digest(Path(name).read_bytes()) != checksum:
            raise ContractError("generator command file changed")


def write_context(work, task, links, repo_root, mode, seed, feedback=None, previous=None, workflow=None):
    public = task.to_dict()
    # Source implementations of the oracle/baseline are not pasted into generation prompts.
    for name in ("oracle", "host_baseline", "device_baseline", "measurement"):
        public.pop(name, None)
    request = {
        "schema_version": 1,
        "task": public,
        "seed": seed,
        "role": "implementer" if previous is None else "diagnoser_implementer",
        "candidate_contract": {
            "abi": "numpy" if mode == "host" else "primfunc",
            "entrypoint": "solution.py:run" if mode == "host" else "solution.py:build",
            "builder_kwargs": (
                "B,N,dtype; dtype is the task component dtype float32/float64"
                if task.task_id in FFT_WORKLOAD_TASKS
                else "B,NC,Q,H,P,G,N,dtype,dt_dtype,has_seq_idx; real fp16/bf16 inputs, fp32 output"
                if task.operator == "ssd_chunk_state_fwd"
                else "case dimensions; only RMSNorm also receives epsilon; FFT pilot receives B,N"
            ),
        },
        "skill_links": links,
        "repository_root": str(repo_root),
        "feedback": feedback,
        "previous_candidate": previous,
        "response_contract": {
            "schema_version": 1,
            "files": {"solution.py": "UTF-8 source"},
            "config": {},
            "plan": "implementation assumptions",
            "usage": {"input_tokens": "integer or null", "output_tokens": "integer or null"},
        },
    }
    if workflow is not None:
        from .skill_workflow import response_contract

        request["skill_workflow"] = workflow
        request["response_contract"] = response_contract(workflow["stage"])
    write_json(work / "context.json", request)
    lines = [
        "# Generation context",
        "",
        "Task and ABI: [context.json](context.json)",
        "",
        "Read the existing guidance through these links; its contents are not rewritten here:",
        "",
    ]
    for link in links:
        lines.append(f"- [{link['path']}]({(repo_root / link['path']).as_uri()}) SHA256={link['sha256']}")
    if workflow is not None:
        lines += [
            "",
            "## Required skill stage: " + workflow["stage"],
            "",
            "Start with skill-status --run-dir <experiment> --trial-id <trial> for the next unread chunk.",
            "Use skill-read for ONE chunk per tool return; do not concatenate reads.",
            "If a return is truncated, request that chunk again before acknowledging it.",
            "Submit every required check with original citations and inspectable evidence.",
            "NOT_APPLICABLE requires a concrete scope reason; BLOCKED cannot advance.",
            "These are delivery/provenance checks, not proof of understanding or correctness.",
            "See skill_workflow and response_contract in context.json for the exact schema.",
            "",
        ]
    (work / "context.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return request


def parse_response(path, token_reservation):
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ContractError("generator response exceeds size budget")
    value = read_json(path)
    fields(value, {"schema_version", "files", "config", "plan", "usage"}, "generator response")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ContractError("unsupported generator response schema")
    text(value["plan"], "plan")
    files = value["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= 500 or "solution.py" not in files:
        raise ContractError("response requires solution.py and at most 500 files")
    for name, source in files.items():
        relative_name(name)
        if name == "candidate.json" or any(part.startswith(".") or part == "__pycache__" for part in Path(name).parts):
            raise ContractError("reserved candidate file name")
        if not isinstance(source, str):
            raise ContractError("candidate files must be UTF-8 strings")
    if sum(len(source.encode()) for source in files.values()) > MAX_BUNDLE_BYTES:
        raise ContractError("candidate exceeds size budget")
    fields(value["usage"], {"input_tokens", "output_tokens"}, "usage")
    for name, count in value["usage"].items():
        if count is not None:
            integer(count, name, 0)
    counts = list(value["usage"].values())
    if all(count is not None for count in counts) and sum(counts) > token_reservation:
        raise ContractError("generator reported token usage above its reservation")
    return value


def response_bundle(value, destination, mode):
    if destination.is_symlink():
        raise ContractError("unsafe candidate destination")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in value["files"].items():
        path = destination / relative_name(name)
        if path.is_symlink() or not path.resolve().is_relative_to(destination.resolve()):
            raise ContractError("unsafe materialized generator path")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise ContractError("generated candidate changed during recovery")
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "abi": "numpy" if mode == "host" else "primfunc",
        "entrypoint": "solution.py:run" if mode == "host" else "solution.py:build",
        "config": value["config"],
    }
    from .bundle import parse_manifest

    parse_manifest(manifest)
    existing = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if existing - (set(value["files"]) | {"candidate.json"}):
        raise ContractError("unexpected generated candidate files")
    if (destination / "candidate.json").exists() and read_json(destination / "candidate.json") != manifest:
        raise ContractError("generated manifest changed")
    write_json(destination / "candidate.json", manifest)
