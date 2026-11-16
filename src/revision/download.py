"""List or download selected files from a public Zenodo record with checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def download(entry, output):
    name = entry["key"]
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Unsafe archive filename")
    algorithm, expected = entry["checksum"].split(":", 1)
    if algorithm not in {"md5", "sha256"}:
        raise ValueError("Unsupported repository checksum")
    destination = output / name
    if destination.exists():
        raise FileExistsError(destination)
    temporary = output / (name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    digest = hashlib.new(algorithm)
    try:
        with (
            temporary.open("xb") as stream,
            urllib.request.urlopen(entry["links"]["self"]) as source,
        ):
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
                stream.write(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Repository checksum mismatch: {name}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=int, required=True, help="Public Zenodo record number")
    parser.add_argument(
        "--file", action="append", default=[], help="Exact filename; repeat to select files"
    )
    parser.add_argument("--output", type=Path, default=Path.cwd())
    args = parser.parse_args()
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{args.record}") as response:
        record = json.load(response)
    entries = {entry["key"]: entry for entry in record["files"]}
    if not args.file:
        print(
            json.dumps(
                [
                    {key: entry[key] for key in ("key", "size", "checksum")}
                    for entry in entries.values()
                ],
                indent=2,
            )
        )
        return
    if missing := set(args.file) - entries.keys():
        raise ValueError(f"Record does not contain files: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.file:
        print(download(entries[name], args.output))


if __name__ == "__main__":
    main()
