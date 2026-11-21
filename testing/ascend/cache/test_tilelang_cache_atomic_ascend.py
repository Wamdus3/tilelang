"""Same-mount publication and real Docker bind/tmpfs EXDEV with compiled kernels."""

import errno
import json

import pytest
import tilelang.testing


pytestmark = [pytest.mark.perf, *tilelang.testing.requires_ascend.marks()]


@pytest.mark.parametrize(
    "scenario", ["unified", "kernel-bind", "binary-bind", "autotune-bind", "kernel-tmpfs", "binary-tmpfs", "autotune-tmpfs"]
)
def test_real_cache_publication(request, cache_lab, real_baseline, scenario, record_property):
    layer = None if scenario == "unified" else scenario.split("-")[0]
    if layer is None:
        reports = real_baseline
    else:
        base = request.getfixturevalue("cache_mounts")
        reports = [
            cache_lab.run_phase(
                base / scenario / "cache",
                cache_lab.work / scenario / phase,
                mode,
                binary_failure=layer == "binary",
            )
            for phase, mode in [("cold", "tune"), ("fresh-process", "tune"), ("direct-compile", "direct")]
        ]
    counts = [report["compile_count"] for report in reports]
    record_property("compile_counts", json.dumps(counts))
    for report in reports:
        for kind, probe in report["replace_probes"].items():
            if kind == layer:
                assert probe == {"errno": errno.EXDEV, "source": "new", "destination": "old"}
            else:
                assert probe == {"errno": None, "source_exists": False, "destination": "new"}
        assert all(not contents for contents in report["staging_leftovers"].values())
    cold, fresh, direct = reports
    if layer == "binary":
        assert counts == [6, 6, 2]
        assert all(not report["artifacts"]["binary"] and not report["saved_tuning_results"] for report in reports)
    else:
        assert counts == [3, 0, 0]
        assert [trial["status"] for trial in cold["runs"][0]["trials"]] == ["ok"] * 3
        if layer == "autotune":
            assert not cold["saved_tuning_results"] and not fresh["saved_tuning_results"]
            assert [trial["status"] for trial in fresh["runs"][0]["trials"]] == ["ok"] * 3
        else:
            assert [trial["status"] for trial in fresh["runs"][0]["trials"]] == ["cache_hit"]
            assert len(cold["saved_tuning_results"]) == 1
            saved = cold["saved_tuning_results"][0]
            assert saved["config"] == cold["runs"][0]["config"] == fresh["runs"][0]["config"]
            assert saved["latency"]["latency"] == cold["runs"][0]["latency_ms"] == fresh["runs"][0]["latency_ms"]
        assert bool(cold["artifacts"]["kernel"]) == (layer != "kernel")
        assert bool(direct["artifacts"]["kernel"]) == (layer != "kernel")


if __name__ == "__main__":
    tilelang.testing.main()
