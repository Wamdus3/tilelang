"""Explicit maintenance command to bind bundled task fixtures to their source bytes.

Default is check-only. --write deliberately creates a new task content identity;
it must only follow a review of the changed oracle/baselines, never run in a judge.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="update fixture hashes after reviewing source changes")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "src" / "tilelang_harness"
    changes = []
    for path in sorted((root / "fixtures").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "oracle" not in data:
            continue  # Scope/provenance catalogs are not executable task contracts.
        for key in ("oracle", "host_baseline", "device_baseline"):
            source = data[key]
            digest = hashlib.sha256((root / source["path"]).read_bytes()).hexdigest()
            if digest != source["sha256"]:
                changes.append(f"{path.stem}:{key}")
                source["sha256"] = digest
        if args.write:
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "UPDATED" if args.write else ("DRIFT" if changes else "MATCH"), "changes": changes}))
    return 1 if changes and not args.write else 0


if __name__ == "__main__":
    raise SystemExit(main())
