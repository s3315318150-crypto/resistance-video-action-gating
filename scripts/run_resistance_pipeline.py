#!/usr/bin/env python3
"""Run the dataset-agnostic local preprocessing and action pipeline.

This is the one-command entrypoint for private local videos. Source media and
derived images stay in ignored directories. The runner produces start/end
segments, seven-stage actions, and a freshly generated wiring episode config.
Rubric aggregation is intentionally separate until all rubric-specific
evidence artifacts are supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def discover_videos(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    videos = sorted(path.resolve() for path in root.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise ValueError(f"no_supported_videos:{root}")
    return videos


def run_phase(name: str, command: list[str], run_root: Path, dry_run: bool) -> dict[str, Any]:
    record: dict[str, Any] = {"phase": name, "command": command, "status": "planned" if dry_run else "running"}
    if dry_run:
        return record
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    record.update({"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
    write_json(run_root / "logs" / f"{name}.json", record)
    if completed.returncode != 0:
        record["status"] = "failed"
        raise RuntimeError(f"phase_failed:{name}:{completed.returncode}")
    record["status"] = "completed"
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data" / "videos")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "resistance_pipeline")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true", help="Write the command plan without opening videos or calling Qwen")
    parser.add_argument("--boundary-interval-seconds", type=float, default=10.0)
    parser.add_argument("--boundary-max-model-edge", type=int, default=480)
    parser.add_argument("--action-sample-interval-seconds", type=float, default=2.0)
    parser.add_argument("--action-max-model-edge", type=int, default=640)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.boundary_interval_seconds <= 0 or args.action_sample_interval_seconds <= 0:
        raise ValueError("sampling_intervals_must_be_positive")
    videos = discover_videos(args.video_dir.resolve())
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_root = args.output_root.resolve() / run_id
    if run_root.exists():
        raise FileExistsError(f"run_directory_exists:{run_root}")
    run_root.mkdir(parents=True)
    marker_root = run_root / "marker_filter"
    boundary_root = run_root / "experiment_boundary"
    action_root = run_root / "actions"
    action_run_id = "v2"
    action_summary = action_root / action_run_id / "summary.json"
    wiring_config = run_root / "generated_configs" / "wiring_sequence.json"
    phases = [
        (
            "01_marker_filter",
            [sys.executable, str(ROOT / "scripts" / "filter_redundant_video_frames.py"), *[str(path) for path in videos], "--output-dir", str(marker_root)],
        ),
        (
            "02_experiment_boundary",
            [
                sys.executable,
                str(ROOT / "scripts" / "qwen_experiment_segment_judge.py"),
                "--input-dir",
                str(marker_root),
                "--output-dir",
                str(boundary_root),
                "--interval-seconds",
                str(args.boundary_interval_seconds),
                "--max-model-edge",
                str(args.boundary_max_model_edge),
                "--prompt-profile",
                "voltmeter_resistance",
                "--timestamp-watermark",
            ],
        ),
        (
            "03_action_v2",
            [
                sys.executable,
                str(ROOT / "scripts" / "qwen_experiment_action_hierarchical_v2.py"),
                "--segment-source",
                str(boundary_root / "summary.json"),
                "--schema",
                str(ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v2.json"),
                "--output-root",
                str(action_root),
                "--run-id",
                action_run_id,
                "--sample-interval-seconds",
                str(args.action_sample_interval_seconds),
                "--max-model-edge",
                str(args.action_max_model_edge),
                "--reduce-recovery-policy",
                "local_partial",
            ],
        ),
        (
            "04_wiring_config",
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_wiring_sequence_config.py"),
                "--action-summary",
                str(action_summary),
                "--video-root",
                str(args.video_dir.resolve()),
                "--output",
                str(wiring_config),
                "--evidence-root",
                str(run_root / "wiring_stable_frames"),
                "--wiring-output-root",
                str(run_root / "wiring_results"),
            ],
        ),
    ]
    report: dict[str, Any] = {
        "schema_version": "resistance_pipeline.run.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dry_run": bool(args.dry_run),
        "video_dir": str(args.video_dir.resolve()),
        "videos": [path.name for path in videos],
        "phases": [],
        "outputs": {
            "boundary_summary": str((boundary_root / "summary.json").resolve()),
            "action_summary": str(action_summary.resolve()),
            "wiring_config": str(wiring_config.resolve()),
        },
        "ten_rubric_status": "rubric_specific_artifacts_required",
        "rubric_specific_artifacts_required": [1, 2, 3, 4, 5, 6, 7, 9],
    }
    write_json(run_root / "run_report.json", report)
    try:
        for name, command in phases:
            record = run_phase(name, command, run_root, args.dry_run)
            report["phases"].append(record)
            write_json(run_root / "run_report.json", report)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}:{exc}"
        write_json(run_root / "run_report.json", report)
        raise
    report["status"] = "planned" if args.dry_run else "completed"
    report["qwen_configuration_present"] = bool(os.getenv("QWEN_API_BASE_URL") and os.getenv("QWEN_API_TOKEN"))
    write_json(run_root / "run_report.json", report)
    print(json.dumps({"status": report["status"], "video_count": len(videos), "run_root": str(run_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
