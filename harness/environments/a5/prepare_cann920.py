"""Prepare the exact official CANN 9.2.0 Conda packages locally, with SHA256 checks."""

import argparse
import fcntl
import hashlib
import http.client
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHANNEL = "https://repo.huaweicloud.com/ascend/repos/conda/linux-64/"
NAMES = ("ascend-cann-toolkit", "ascend-cann-950-ops")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".download.lock").open("a+b")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with urllib.request.urlopen(CHANNEL + "repodata.json", timeout=60) as response:
        raw = response.read()
    (output / "repodata.json").write_bytes(raw)
    data = json.loads(raw)
    entries = []
    for name in NAMES:
        matches = [
            (file, entry)
            for file, entry in data.get("packages.conda", {}).items()
            if entry["name"] == name and entry["version"] == "9.2.0"
        ]
        if len(matches) != 1:
            raise ValueError("expected exactly one official stable 9.2.0 package for " + name)
        file, entry = matches[0]
        entries.append({**entry, "file": file, "url": CHANNEL + file})
    manifest = {
        "schema_version": 1,
        "source": CHANNEL,
        "version": "9.2.0",
        "packages": entries,
        "status": "DOWNLOADING",
        "device_execution": "NOT_RUN",
    }
    manifest_file = output / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

    def download(entry):
        destination = output / entry["file"]
        if destination.exists():
            if destination.stat().st_size == entry["size"] and sha256(destination) == entry["sha256"]:
                print(entry["file"] + ": verified existing package", flush=True)
                return
            raise ValueError("existing package has unexpected contents: " + str(destination))
        temporary = destination.with_suffix(".part")
        parts = output / (entry["file"] + ".parts")
        parts.mkdir(exist_ok=True)
        chunk = 16 * 1024 * 1024
        ranges = [(start, min(start + chunk, entry["size"]) - 1) for start in range(0, entry["size"], chunk)]

        def fetch(bounds):
            start, end = bounds
            part = parts / f"{start:012d}"
            length = end - start + 1
            if part.exists() and part.stat().st_size == length:
                return length
            for attempt in range(1, 5):
                try:
                    request = urllib.request.Request(entry["url"], headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=45) as response, part.open("wb") as stream:
                        expected = f"bytes {start}-{end}/{entry['size']}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            raise ValueError("server did not honor the requested package range")
                        shutil.copyfileobj(response, stream, 1024 * 1024)
                    if part.stat().st_size != length:
                        raise ValueError("incomplete package range")
                    return length
                except (OSError, ValueError, http.client.HTTPException) as exc:
                    print(f"{entry['file']} range {start}-{end}: attempt {attempt}: {exc}", flush=True)
                    if attempt == 4:
                        raise
                    time.sleep(attempt)

        count, reported = 0, -1
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in as_completed([pool.submit(fetch, bounds) for bounds in ranges]):
                count += future.result()
                progress = count * 100 // entry["size"]
                if progress // 10 != reported:
                    reported = progress // 10
                    print(f"{entry['file']}: {progress}% ({count}/{entry['size']} bytes)", flush=True)
        with temporary.open("wb") as stream:
            for start, _ in ranges:
                with (parts / f"{start:012d}").open("rb") as source:
                    shutil.copyfileobj(source, stream, 8 * 1024 * 1024)
        if temporary.stat().st_size != entry["size"] or sha256(temporary) != entry["sha256"]:
            raise ValueError("official package size or SHA256 mismatch: " + entry["file"])
        temporary.rename(destination)
        print(entry["file"] + ": SHA256 PASS", flush=True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(download, entries))
    manifest["status"] = "DOWNLOAD_VERIFIED"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "bytes": sum(item["size"] for item in entries)}), flush=True)


if __name__ == "__main__":
    main()
