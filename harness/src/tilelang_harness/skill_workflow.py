"""Kernel structure and original-source evidence contracts, without a model SDK.

These checks bind agent-reported decisions to guidance and candidate/lowering
evidence. They do not prove understanding, semantics or device correctness.
"""

from .artifacts import digest, read_json
from .bundle import MAX_BUNDLE_BYTES, relative_name
from .contracts import ContractError, content_id, fields, integer, text

PROFILE = "ascend-kernel-v2"
ASSURANCE = "agent_reported_kernel_contracts_and_source_evidence; not semantic or hardware proof"
ASCEND_SKILL = ".agents/skills/tilelang-ascend/SKILL.md"
STAGES = ("design", "implementation", "lowering_audit", "validation")
CHECKS = {
    "design": (
        "tile_decomposition",
        "stage_core_pipe_map",
        "physical_storage_budget",
        "vf_selection",
        "pipeline_budget",
    ),
    "implementation": (
        "tile_decomposition",
        "api_contracts",
        "address_alignment",
        "initialization",
        "ownership",
        "pipeline",
    ),
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
    "tile_decomposition": (STRUCTURE, ASCEND_SKILL),
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
KERNEL_STRUCTURE = {
    "macro_stages": "Capture compile-time tiling options before defining T.macro stages. Use meaningful "
    "producer/consumer buffers and core/PIPE roles; a simple inline stage needs a concrete scope reason.",
    "tile_parallelism": "Specify tile extents, index/axis mapping, tails and masks. Explain how T.Parallel "
    "maps work inside the selected VF. Choose SIMD/SIMT per stage; explicit T.simd.* intrinsics are valid "
    "when appropriate, with their lane/predicate mapping documented.",
    "orchestration": "The PrimFunc owns allocations, layouts/version annotations and the caller tile loop. "
    "Separate recurrent state from per-tile scratch. Macro boundaries do not create pipeline tasks or barriers.",
}


def profile_for(arm, mode):
    default = PROFILE if mode == "lower" and ASCEND_SKILL in arm["skill_links"] else "links"
    profile = arm.get("skill_workflow", default)
    if profile not in ("links", PROFILE):
        raise ContractError("skill_workflow must be links or ascend-kernel-v2")
    if profile == PROFILE and (mode != "lower" or ASCEND_SKILL not in arm["skill_links"]):
        raise ContractError("ascend-kernel-v2 requires lower mode and the original Ascend skill")
    return profile


def verify_workflow_sources(root, links):
    required = {ASCEND_SKILL, STRUCTURE, SYNC, PROFILING}
    if not required <= {link["path"] for link in links}:
        raise ContractError("Ascend workflow is missing required linked references")
    for link in links:
        path = (root / relative_name(link["path"])).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ContractError("skill reference escapes repository or is missing")
        if digest(path.read_bytes()) != link["sha256"]:
            raise ContractError("linked skill changed")


def workflow_context(root, links, stage, trial_id, progress, case_ids):
    if stage not in STAGES:
        raise ContractError("unknown skill stage")
    verify_workflow_sources(root, links)
    binding = {"profile": PROFILE, "stage": stage, "trial_id": trial_id, "history": progress, "case_ids": case_ids}
    return {
        **binding,
        "request_id": content_id(binding),
        "kernel_structure": dict(KERNEL_STRUCTURE),
        "required_checks": list(CHECKS[stage]),
        "required_citation_sources": {check: list(CHECK_SOURCES[check]) for check in CHECKS[stage]},
        "stage_deliverable": {
            "design": "T.macro stage and producer/consumer plan; tile extents, T.Parallel/VF index mapping and "
            "tail/mask coverage; core/PIPE roles; physical bytes including padding, aliases, conversion scratch "
            "and versions; caller-loop stage bound and reasons for inline/serial/SIMT alternatives.",
            "implementation": "Source evidence for the planned T.macro stages and T.Parallel/VF tile mapping "
            "(or justified inline/explicit-SIMD forms); selected API signature/dtype/scope/alignment contracts; "
            "write/read footprints and pipeline ownership.",
            "lowering_audit": "For EVERY case inspect actual allocations/versions, offsets, guards, address "
            "alignment, initialization and ready/reuse across epochs. Cite lowered.tir/kernel.cpp.",
            "validation": "Separate lower/vendor/device correctness/performance status; fixed case coverage, "
            "timing boundary/repetitions, pending hardware work and authorization limits.",
        }[stage],
        "check_scope": "each_case" if stage == "lowering_audit" else "whole_candidate",
        "assurance": ASSURANCE,
    }


def response_contract(stage):
    return {
        "schema_version": 1,
        "stage": stage,
        "request_id": "copy skill_workflow.request_id",
        "checks": [
            {
                "id": "required check",
                "case_id": "required case for lowering_audit, otherwise null",
                "status": "PASS or NOT_APPLICABLE; BLOCKED rejects the candidate",
                "reason": "concrete decision, deviation reason or unresolved constraint",
                "citations": [
                    {
                        "path": "repository-relative original guidance path from skill_links",
                        "sha256": "source SHA256 from skill_links",
                        "start_line": 1,
                        "end_line": 1,
                        "quote": "exact nonempty excerpt from these lines, <=240 chars",
                    }
                ],
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


def _citation_path(cite, root, sources):
    fields(cite, {"path", "sha256", "start_line", "end_line", "quote"}, "skill citation")
    name = relative_name(cite["path"])
    if name not in sources or cite["sha256"] != sources[name]:
        raise ContractError("skill citation does not match frozen guidance")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ContractError("skill citation escapes repository or is missing")
    data = path.read_bytes()
    if digest(data) != sources[name]:
        raise ContractError("cited skill changed since the request was created")
    lines = data.decode("utf-8").splitlines(keepends=True)
    begin = integer(cite["start_line"], "citation start")
    end = integer(cite["end_line"], "citation end")
    if end < begin or end > len(lines):
        raise ContractError("skill citation line range is invalid")
    quote = text(cite["quote"], "citation quote")
    if len(quote) > 240 or quote not in "".join(lines[begin - 1 : end]):
        raise ContractError("skill quote does not match the cited original lines")
    return name


def parse_stage_response(path, context, root, token_reservation, lowering_files):
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ContractError("generator response exceeds size budget")
    value = read_json(path)
    fields(
        value,
        {"schema_version", "stage", "request_id", "checks", "files", "config", "plan", "usage"},
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
    sources = {link["path"]: link["sha256"] for link in context["skill_links"]}
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
            cited_paths.add(_citation_path(cite, root, sources))
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
        if (
            workflow["stage"] == "implementation"
            and check["id"] == "tile_decomposition"
            and not any(e["path"].startswith("files/") and e["path"].endswith(".py") for e in check["evidence"])
        ):
            raise ContractError("tile decomposition requires submitted Python source evidence")
        if workflow["stage"] == "lowering_audit":
            # A prose-only audit or an artifact from another case is insufficient.
            prefix = "lowering/" + check["case_id"] + "/"
            if not any(e["path"] in (prefix + "lowered.tir", prefix + "kernel.cpp") for e in check["evidence"]):
                raise ContractError("each lowering check requires an artifact from its exact case")
    if seen != expected:
        raise ContractError("missing required skill checks/case coverage")
    return value
