"""Durable content-addressed files; SQLite only points at committed objects."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .contracts import ContractError, canonical_json


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ContractError(f"nonfinite JSON number: {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError) as exc:
        raise ContractError(f"invalid JSON {path}: {exc}") from exc


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if os.name == "posix":
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value):
    atomic_write(path, canonical_json(value).encode("utf-8"))


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root

    def path(self, key: str) -> Path:
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ContractError("invalid artifact digest")
        return self.root / key[:2] / key[2:]

    def put(self, data: bytes) -> str:
        key = digest(data)
        path = self.path(key)
        if path.exists():
            self.get(key)
        else:
            atomic_write(path, data)
        return key

    def get(self, key: str) -> bytes:
        data = self.path(key).read_bytes()
        if digest(data) != key:
            raise ContractError(f"artifact corruption: {key}")
        return data

    def put_json(self, value) -> str:
        return self.put(canonical_json(value).encode("utf-8"))

    def json(self, key: str):
        # get() verifies the content before parsing it.
        return json.loads(self.get(key))
