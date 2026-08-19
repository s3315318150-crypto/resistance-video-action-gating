#!/usr/bin/env python3
"""Create an immutable SHA-256 sidecar for a prediction artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def freeze(input_path: Path, output_path: Path) -> dict[str, Any]:
    source = input_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite freeze record: {target}")
    json.loads(source.read_text(encoding="utf-8"))
    record = {
        "schema_version": "1.0",
        "artifact_type": "prediction_artifact_freeze",
        "frozen_at": utc_now(),
        "input_path": str(source),
        "sha256": sha256_file(source),
        "ground_truth_accessed": False,
        "input_modified": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Prediction JSON to freeze.")
    parser.add_argument("--output", required=True, type=Path, help="New freeze sidecar JSON.")
    args = parser.parse_args()
    record = freeze(args.input, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": record["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
