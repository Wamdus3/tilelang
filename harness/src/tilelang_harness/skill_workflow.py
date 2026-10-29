"""Bounded original-skill delivery and evidence contracts, without a model SDK.

These checks establish delivery and provenance, not that an agent understood a
document or that its technical conclusions are correct.
"""

from .artifacts import digest, read_json
from .bundle import MAX_BUNDLE_BYTES, relative_name
from .contracts import ContractError, content_id, fields, integer, text

PROFILE = "ascend-kernel-v1"
ASCEND_SKILL = ".agents/skills/tilelang-ascend/SKILL.md"
STAGES = ("design", "implementation", "lowering_audit", "validation")
CHECKS = {
    "design": ("stage_core_pipe_map", "physical_storage_budget", "vf_selection", "pipeline_budget"),
    "implementation": ("api_contracts", "address_alignment", "initialization", "ownership", "pipeline"),
    "lowering_audit": (
        "materialized_storage",
        "versions_and_epochs",
        "ready_reuse",
        "address_and_footprint",
        "generated_schedule",
    ),
    "validation": ("validation_levels", "measurement_protocol", "coverage_and_limits"),
}
STRUCTURE = ".agents/skills/tilelang-ascend/references/kernel-structure.md"
SYNC = ".agents/skills/tilelang-ascend/references/synchronization.md"
PROFILING = ".agents/skills/tilelang-ascend/references/profiling.md"
CHECK_SOURCES = {
    "stage_core_pipe_map": (ASCEND_SKILL, STRUCTURE),
    "physical_storage_budget": (SYNC,),
    "vf_selection": (STRUCTURE,),
    "pipeline_budget": (SYNC,),
    "api_contracts": (ASCEND_SKILL,),
    "address_alignment": (ASCEND_SKILL, STRUCTURE),
    "initialization": (SYNC,),
    "ownership": (SYNC,),
    "pipeline": (SYNC,),
    "materialized_storage": (SYNC,),
    "versions_and_epochs": (SYNC,),
    "ready_reuse": (SYNC,),
    "address_and_footprint": (ASCEND_SKILL, SYNC),
    "generated_schedule": (SYNC,),
    "validation_levels": (ASCEND_SKILL, PROFILING),
    "measurement_protocol": (PROFILING,),
    "coverage_and_limits": (ASCEND_SKILL, PROFILING),
}
# Small enough for an ordinary tool return, even with line numbers and metadata.
MAX_CHUNK_BYTES = 2400
MAX_CHUNK_LINES = 40


def profile_for(arm, mode):
    default = PROFILE if mode == "lower" and ASCEND_SKILL in arm["skill_links"] else "links"
    profile = arm.get("skill_workflow", default)
    if profile not in ("links", PROFILE):
        raise ContractError("skill_workflow must be links or ascend-kernel-v1")
    if profile == PROFILE and (mode != "lower" or ASCEND_SKILL not in arm["skill_links"]):
        raise ContractError("ascend-kernel-v1 requires lower mode and the original Ascend skill")
    return profile


def _sections(lines, start, end=None):
    begin = next((i for i, line in enumerate(lines) if line.startswith(start)), None)
    stop = next((i for i, line in enumerate(lines) if end and line.startswith(end)), len(lines))
    if begin is None or stop <= begin:
        raise ContractError(f"skill workflow anchor missing or out of order: {start}")
    return begin, stop


def _ranges(path, lines, stage):
    if path == ASCEND_SKILL:
        mental, _ = _sections(lines, "## 1. Mental Model")
        pitfalls, _ = _sections(lines, "## 10. Common Pitfalls")
        performance, _ = _sections(lines, "## 12. Performance")
        return {
            "design": [(0, mental)],
            "implementation": [(mental, pitfalls)],
            "lowering_audit": [(0, mental), (pitfalls, performance)],
            "validation": [(performance, len(lines))],
        }[stage]
    if path.endswith("/references/kernel-structure.md"):
        return [(0, len(lines))] if stage in ("design", "implementation") else []
    if path.endswith("/references/synchronization.md"):
        begin, end = _sections(lines, "### Establish an automatic tiled pipeline", "### Tune performance")
        return (
            [(begin, end)]
            if stage == "design"
            else ([(0, len(lines))] if stage in ("implementation", "lowering_audit") else [])
        )
    if path.endswith("/references/profiling.md"):
        return [(0, len(lines))] if stage == "validation" else []
    # Build guide and compiler tools are consulted before implementation and
    # again when auditing actual lowering. Every file in the closure is routed.
    return [(0, len(lines))] if stage in ("implementation", "lowering_audit") else []


def make_manifest(root, links, stage):
    if stage not in STAGES:
        raise ContractError("unknown skill stage")
    required = {
        ASCEND_SKILL,
        ".agents/skills/tilelang-ascend/references/kernel-structure.md",
        ".agents/skills/tilelang-ascend/references/synchronization.md",
        ".agents/skills/tilelang-ascend/references/profiling.md",
    }
    if not required <= {link["path"] for link in links}:
        raise ContractError("Ascend workflow is missing required linked references")
    chunks = []
    for link in links:
        data = (root / relative_name(link["path"])).read_bytes()
        if digest(data) != link["sha256"]:
            raise ContractError("linked skill changed")
        lines = data.decode("utf-8").splitlines(keepends=True)
        for begin, end in _ranges(link["path"], lines, stage):
            cursor = begin
            while cursor < end:
                stop, size = cursor, 0
                while stop < end and stop - cursor < MAX_CHUNK_LINES:
                    length = len(lines[stop].encode("utf-8"))
                    if length > MAX_CHUNK_BYTES:
                        raise ContractError("skill contains an oversized line; split it before creating an experiment")
                    if size + length > MAX_CHUNK_BYTES:
                        break
                    size += length
                    stop += 1
                chunk = {
                    "path": link["path"],
                    "source_sha256": link["sha256"],
                    "start_line": cursor + 1,
                    "end_line": stop,
                    "sha256": digest("".join(lines[cursor:stop]).encode("utf-8")),
                }
                chunk["chunk_id"] = content_id(chunk)
                chunks.append(chunk)
                cursor = stop
    return chunks


def chunk_text(root, chunk):
    path = (root / relative_name(chunk["path"])).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ContractError("skill chunk escapes repository")
    data = path.read_bytes()
    if digest(data) != chunk["source_sha256"]:
        raise ContractError("skill changed since the request was created")
    lines = data.decode("utf-8").splitlines(keepends=True)
    result = "".join(lines[chunk["start_line"] - 1 : chunk["end_line"]])
    if digest(result.encode("utf-8")) != chunk["sha256"] or len(result.encode("utf-8")) > MAX_CHUNK_BYTES:
        raise ContractError("skill chunk checksum/size mismatch")
    return result


def workflow_context(root, links, stage, trial_id, progress, case_ids):
    binding = {"profile": PROFILE, "stage": stage, "trial_id": trial_id, "history": progress, "case_ids": case_ids}
    return {
        **binding,
        "request_id": content_id(binding),
        "chunks": make_manifest(root, links, stage),
        "required_checks": list(CHECKS[stage]),
        "required_citation_sources": {check: list(CHECK_SOURCES[check]) for check in CHECKS[stage]},
        "stage_deliverable": {
            "design": "Stage/core/PIPE/VF table; physical bytes including padding, aliases, conversion scratch and "
            "versions; caller-loop stage bound and explicit reasons for serial/SIMT alternatives.",
            "implementation": "Source plus selected API signature/dtype/scope/alignment contracts and evidence "
            "for unsupported alternatives; write/read footprints and pipeline ownership.",
            "lowering_audit": "For EVERY case inspect actual allocations/versions, offsets, guards, address "
            "alignment, initialization and ready/reuse across epochs. Cite lowered.tir/kernel.cpp.",
            "validation": "Separate lower/vendor/device correctness/performance status; fixed case coverage, "
            "timing boundary/repetitions, pending hardware work and authorization limits.",
        }[stage],
        "check_scope": "each_case" if stage == "lowering_audit" else "whole_candidate",
        "read_command": "python -m tilelang_harness skill-read --run-dir <experiment> --trial-id <trial> --chunk <id>",
        "assurance": "bounded_delivery_and_agent_reported_evidence; not semantic or hardware proof",
    }


def response_contract(stage):
    return {
        "schema_version": 1,
        "stage": stage,
        "request_id": "copy skill_workflow.request_id",
        "reads": [{"chunk_id": "every required chunk", "delivery_id": "from skill-read"}],
        "checks": [
            {
                "id": "required check",
                "case_id": "required case for lowering_audit, otherwise null",
                "status": "PASS or NOT_APPLICABLE; BLOCKED rejects the candidate",
                "reason": "concrete decision, deviation reason or unresolved constraint",
                "citations": [{"chunk_id": "required chunk", "quote": "exact nonempty excerpt, <=240 chars"}],
                "evidence": [
                    {
                        "path": "files/<submitted file> or lowering/<artifact path>",
                        "sha256": "SHA256 of referenced bytes",
                        "start_line": 1,
                        "end_line": 1,
                    }
                ],
            }
        ],
        "files": {"plan.md": "design; use solution.py plus audit notes for implementation"},
        "config": {},
        "plan": "summary of this stage",
        "usage": {"input_tokens": "integer or null", "output_tokens": "integer or null"},
    }


def parse_stage_response(path, context, deliveries, root, token_reservation, lowering_files):
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ContractError("generator response exceeds size budget")
    value = read_json(path)
    fields(
        value,
        {"schema_version", "stage", "request_id", "reads", "checks", "files", "config", "plan", "usage"},
        "skill stage response",
    )
    workflow = context["skill_workflow"]
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ContractError("unsupported skill response schema")
    if value["stage"] != workflow["stage"] or value["request_id"] != workflow["request_id"]:
        raise ContractError("stale or wrong-stage skill response")
    text(value["plan"], "stage plan")
    fields(value["usage"], {"input_tokens", "output_tokens"}, "usage")
    for name, count in value["usage"].items():
        if count is not None:
            integer(count, name, 0)
    counts = list(value["usage"].values())
    if all(count is not None for count in counts) and sum(counts) > token_reservation:
        raise ContractError("generator reported token usage above its reservation")
    files = value["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= 500:
        raise ContractError("stage requires 1..500 evidence files")
    for name, source in files.items():
        relative_name(name)
        if name == "candidate.json" or any(part.startswith(".") for part in name.split("/")):
            raise ContractError("reserved stage file name")
        if not isinstance(source, str):
            raise ContractError("stage files must be UTF-8 strings")
    if sum(len(source.encode("utf-8")) for source in files.values()) > MAX_BUNDLE_BYTES:
        raise ContractError("stage files exceed size budget")
    if not isinstance(value["config"], dict):
        raise ContractError("config must be an object")
    if workflow["stage"] == "implementation" and "solution.py" not in files:
        raise ContractError("implementation requires solution.py")
    if workflow["stage"] != "implementation" and ("solution.py" in files or value["config"]):
        raise ContractError("only the implementation stage may submit code/config")
    chunks = {item["chunk_id"]: item for item in workflow["chunks"]}
    if not isinstance(value["reads"], list):
        raise ContractError("skill reads must be a list")
    seen = set()
    for read in value["reads"]:
        fields(read, {"chunk_id", "delivery_id"}, "skill read")
        key = text(read["chunk_id"], "chunk_id")
        if key not in chunks or key in seen or deliveries.get(key) != read["delivery_id"]:
            raise ContractError("missing, duplicate or unverified skill delivery")
        seen.add(key)
    if seen != set(chunks):
        raise ContractError("incomplete skill reading; request every missing chunk separately")
    if not isinstance(value["checks"], list):
        raise ContractError("skill checks must be a list")
    cases = workflow["case_ids"] if workflow["stage"] == "lowering_audit" else [None]
    expected = {(check, case) for check in CHECKS[workflow["stage"]] for case in cases}
    seen = set()
    available = {"files/" + name: source.encode("utf-8") for name, source in files.items()}
    available.update(lowering_files)
    for check in value["checks"]:
        fields(check, {"id", "case_id", "status", "reason", "citations", "evidence"}, "skill check")
        if check["case_id"] is not None:
            text(check["case_id"], "check case_id")
        key = (text(check["id"], "check id"), check["case_id"])
        if key not in expected or key in seen:
            raise ContractError("unexpected, duplicate or wrong-case skill check")
        seen.add(key)
        if check["status"] not in ("PASS", "NOT_APPLICABLE"):
            raise ContractError("blocked/failed skill check; repair before continuing")
        text(check["reason"], "concrete skill decision/reason")
        if not isinstance(check["citations"], list) or not check["citations"]:
            raise ContractError("each decision requires original skill citations")
        cited_paths = set()
        for cite in check["citations"]:
            fields(cite, {"chunk_id", "quote"}, "skill citation")
            chunk = chunks.get(text(cite["chunk_id"], "citation chunk"))
            quote = text(cite["quote"], "citation quote")
            if chunk is None or len(quote) > 240 or quote not in chunk_text(root, chunk):
                raise ContractError("skill quote does not match a required original chunk")
            cited_paths.add(chunk["path"])
        if not cited_paths.intersection(CHECK_SOURCES[check["id"]]):
            raise ContractError("decision cites unrelated guidance; consult its required skill reference")
        if not isinstance(check["evidence"], list) or not check["evidence"]:
            raise ContractError("each decision requires inspectable evidence")
        for evidence in check["evidence"]:
            fields(evidence, {"path", "sha256", "start_line", "end_line"}, "check evidence")
            name = text(evidence["path"], "evidence path")
            data = available.get(name)
            if data is None or digest(data) != evidence["sha256"]:
                raise ContractError("evidence does not match this submission/lowering")
            begin = integer(evidence["start_line"], "evidence start")
            end = integer(evidence["end_line"], "evidence end")
            if end < begin or end > len(data.decode("utf-8").splitlines()):
                raise ContractError("evidence line range is invalid")
        if workflow["stage"] == "lowering_audit":
            # A prose-only audit or an artifact from another case is insufficient.
            prefix = "lowering/" + check["case_id"] + "/"
            if not any(e["path"] in (prefix + "lowered.tir", prefix + "kernel.cpp") for e in check["evidence"]):
                raise ContractError("each lowering check requires an artifact from its exact case")
    if seen != expected:
        raise ContractError("missing required skill checks/case coverage")
    return value
