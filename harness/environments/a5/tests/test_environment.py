"""Local-only guards for dependency closure, ABI auditing and offline extraction."""

import ast
import importlib.util
import json
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize(
    "name,link",
    [
        ("../escape", None),
        ("/absolute", None),
        ("dir/file", "../../escape"),
        ("dir/file", "/outside"),
    ],
)
def test_archive_cannot_escape_prefix(name, link):
    installer = module("install_offline")
    entry = tarfile.TarInfo(name)
    if link:
        entry.type, entry.linkname = tarfile.SYMTYPE, link
    with pytest.raises(ValueError, match="unsafe|escapes"):
        installer.validate_member(entry)


def test_internal_relative_link_is_allowed():
    installer = module("install_offline")
    entry = tarfile.TarInfo("bin/python")
    entry.type, entry.linkname = tarfile.SYMTYPE, "../lib/python3.11/python"
    installer.validate_member(entry)


@pytest.mark.parametrize("version", ["GLIBC_2.38", "GLIBC_ABI_DT_RELR", "GLIBC_PRIVATE"])
def test_new_or_private_glibc_is_rejected(monkeypatch, tmp_path, version):
    checker = module("audit_abi")
    monkeypatch.setattr(checker, "requirements", lambda path: ["GLIBC_2.17", version])
    report = checker.audit_file(tmp_path / "not-executed", "artifact", (2, 31))
    assert report["incompatible"] == [version]


def test_installer_is_parseable_by_server_system_python():
    ast.parse((TOOLS / "install_offline.py").read_text(), feature_version=(3, 8))


def test_abi_parser_ignores_symbol_definitions(monkeypatch, tmp_path):
    checker = module("audit_abi")
    output = "Version definition section:\n Name: GLIBC_2.99\nVersion needs section:\n Name: GLIBC_2.28\n"
    monkeypatch.setattr(checker.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=output))
    assert checker.requirements(tmp_path / "library") == ["GLIBC_2.28"]


def test_empty_audit_never_passes(tmp_path):
    checker = module("audit_abi")
    with pytest.raises(ValueError, match="no ELF"):
        checker.audit([tmp_path], (2, 31))


def test_wheel_closure_and_nested_vendor_metadata(tmp_path, monkeypatch):
    prepare = module("prepare_local")
    config = tmp_path / "config"
    config.mkdir()
    (config / "runtime.in").write_text("root-package==1.0\n")
    (config / "build.in").write_text("")
    monkeypatch.setattr(prepare, "CONFIG", config)
    output = tmp_path / "offline"
    wheels = output / "wheels"
    wheels.mkdir(parents=True)
    with zipfile.ZipFile(wheels / "root_package-1.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr(
            "root_package-1.0.dist-info/METADATA", "Name: root-package\nVersion: 1.0\nRequires-Dist: child>=2\n"
        )
        wheel.writestr("root/_vendor/nested-1.0.dist-info/METADATA", "Name: nested\nVersion: 1.0\n")
    with pytest.raises(ValueError, match="child>=2"):
        prepare.download(output, skip_download=True)
    assert not (output / "wheel-manifest.json").exists()
    with zipfile.ZipFile(wheels / "child-2.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr("child-2.0.dist-info/METADATA", "Name: child\nVersion: 2.0\n")
    prepare.download(output, skip_download=True)
    manifest = json.loads((output / "wheel-manifest.json").read_text())
    assert {item["name"] for item in manifest["wheels"]} == {"root-package", "child"}


def test_pins_cover_repository_runtime():
    module("prepare_local").check_roots()
