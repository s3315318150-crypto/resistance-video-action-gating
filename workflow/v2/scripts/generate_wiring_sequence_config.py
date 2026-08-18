#!/usr/bin/env python3
"""Generate per-run wiring episodes and stable-frame evidence from v2 stages.

The generator is dataset-agnostic. It discovers videos from the action summary,
derives each wiring episode from observed stages, samples the following stable
measurement/recording interval, and exports fresh primary/backup JPEG evidence.
It never reuses timestamps or images from the historical five-video config.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
WIRING_STAGES = {"circuit_wiring", "circuit_rewiring"}
STABLE_STAGES = {"measurement_1", "recording_1", "measurement_2", "recording_2"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_expected:{path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_from_root(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def safe_id(value: str) -> str:
    stem = Path(value).stem
    match = re.match(r"([0-9]+)(?:_|$)", Path(value).name)
    if match:
        return match.group(1)
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.-")
    if not result:
        raise ValueError(f"video_id_unavailable:{value!r}")
    return result


def finite_interval(item: dict[str, Any]) -> tuple[float, float] | None:
    try:
        start = float(item["start_seconds"])
        end = float(item["end_seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        return None
    return start, end


def observed_stages(document: dict[str, Any]) -> list[dict[str, Any]]:
    raw_runs = document.get("observed_stage_runs")
    if isinstance(raw_runs, list):
        stages: list[dict[str, Any]] = []
        for item in raw_runs:
            if not isinstance(item, dict):
                continue
            interval = finite_interval(item)
            stage = str(item.get("stage") or "")
            if interval is None or not stage:
                continue
            normalized: dict[str, Any] = {
                "stage": stage,
                "start": interval[0],
                "end": interval[1],
            }
            for field in (
                "stage_semantics",
                "stage_window_semantics",
                "merged_stage_semantics",
                "merged_measurement_recording",
                "merged_stage",
                "contains_measurement_evidence",
                "measurement_subintervals",
            ):
                if field in item:
                    normalized[field] = item[field]
            stages.append(normalized)
        if stages:
            return sorted(stages, key=lambda item: (item["start"], item["end"], item["stage"]))

    stages: list[dict[str, Any]] = []
    for item in document.get("timeline_segments", []):
        if not isinstance(item, dict) or item.get("kind") != "observed_stage":
            continue
        interval = finite_interval(item)
        stage = str(item.get("stage") or "")
        if interval is None or not stage:
            continue
        stages.append({"stage": stage, "start": interval[0], "end": interval[1]})
    return sorted(stages, key=lambda item: (item["start"], item["end"], item["stage"]))


def stable_intervals(stage: dict[str, Any]) -> list[dict[str, Any]]:
    stage_name = str(stage.get("stage") or "")
    merged = stage_name in {"recording_1", "recording_2"} and (
        stage.get("merged_measurement_recording") is True
        or stage.get("merged_stage") is True
        or stage.get("stage_semantics") == "measurement_and_recording_cycle"
        or stage.get("stage_window_semantics") == "measurement_and_recording_cycle"
        or stage.get("merged_stage_semantics") == "measurement_and_recording_cycle"
    )
    if merged:
        intervals: list[dict[str, Any]] = []
        raw = stage.get("measurement_subintervals")
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            interval = finite_interval(item)
            if interval is None:
                continue
            intervals.append(
                {
                    "start": interval[0],
                    "end": interval[1],
                    "source": f"{stage_name}.measurement_action",
                }
            )
        return sorted(intervals, key=lambda item: (item["start"], item["end"]))
    if stage_name not in STABLE_STAGES:
        return []
    return [{"start": float(stage["start"]), "end": float(stage["end"]), "source": stage_name}]


def episode_windows(stages: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    """Pair every wiring stage with the next measurement or recording interval."""
    episodes: list[dict[str, Any]] = []
    wiring_items = [item for item in stages if item["stage"] in WIRING_STAGES]
    for index, wiring in enumerate(wiring_items, start=1):
        stable_candidates = [
            interval
            for item in stages
            for interval in stable_intervals(item)
            if interval["end"] > wiring["end"]
        ]
        following = next(
            iter(sorted(stable_candidates, key=lambda item: (item["start"], item["end"]))),
            None,
        )
        if following is None:
            stable_start = wiring["end"]
            stable_end = min(duration, stable_start + 20.0)
            stable_source = "post_wiring_fallback"
        else:
            stable_start = max(wiring["end"], following["start"])
            stable_end = min(duration, following["end"])
            stable_source = following["source"]
        if stable_end <= stable_start:
            stable_start = min(duration, wiring["end"])
            stable_end = min(duration, stable_start + 5.0)
            stable_source = "short_post_wiring_fallback"
        if stable_end <= stable_start:
            continue
        episodes.append(
            {
                "episode_id": f"episode_{index:02d}",
                "source_stage": wiring["stage"],
                "wiring_window_seconds": [round(wiring["start"], 3), round(wiring["end"], 3)],
                "recording_window_seconds": [round(stable_start, 3), round(stable_end, 3)],
                "stable_window_source": stable_source,
            }
        )
    return episodes


def candidate_timestamps(start: float, end: float, interval: float, limit: int) -> list[float]:
    if interval <= 0 or limit < 2 or end <= start:
        raise ValueError("invalid_stable_sampling_parameters")
    values: list[float] = []
    current = start
    while current <= end + 1e-6:
        values.append(min(current, end))
        current += interval
    if not values or values[-1] < end - 1e-3:
        values.append(end)
    if len(values) > limit:
        indexes = np.linspace(0, len(values) - 1, limit).round().astype(int)
        values = [values[int(index)] for index in indexes]
    return sorted(set(round(value, 3) for value in values))


def frame_quality(frame: np.ndarray, previous: np.ndarray | None, following: np.ndarray | None) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (320, max(1, round(gray.shape[0] * 320 / gray.shape[1]))), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
    exposure = 1.0 - min(1.0, abs(float(small.mean()) - 128.0) / 128.0)
    differences: list[float] = []
    for neighbour in (previous, following):
        if neighbour is None:
            continue
        other = cv2.cvtColor(neighbour, cv2.COLOR_BGR2GRAY)
        other = cv2.resize(other, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        differences.append(float(cv2.absdiff(small, other).mean()) / 255.0)
    motion = sum(differences) / len(differences) if differences else 0.5
    return {"sharpness": sharpness, "exposure": exposure, "staticness": 1.0 - min(1.0, motion * 4.0)}


def read_frame(capture: cv2.VideoCapture, timestamp: float) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
    ok, frame = capture.read()
    return frame if ok and frame is not None else None


def select_stable_frames(
    video: Path,
    timestamps: Iterable[float],
    output_dir: Path,
    backup_gap_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"video_open_failed:{video}")
    candidates: list[dict[str, Any]] = []
    try:
        for timestamp in timestamps:
            frame = read_frame(capture, timestamp)
            if frame is None:
                continue
            previous = read_frame(capture, max(0.0, timestamp - 0.4))
            following = read_frame(capture, timestamp + 0.4)
            candidates.append({"timestamp": timestamp, "frame": frame, **frame_quality(frame, previous, following)})
    finally:
        capture.release()
    if len(candidates) < 2:
        raise RuntimeError(f"stable_frame_candidates_insufficient:{video}")
    sharp_values = [math.log1p(item["sharpness"]) for item in candidates]
    low, high = min(sharp_values), max(sharp_values)
    for item, sharp in zip(candidates, sharp_values):
        normalized_sharp = (sharp - low) / (high - low) if high > low else 0.5
        item["quality"] = 0.55 * normalized_sharp + 0.20 * item["exposure"] + 0.25 * item["staticness"]
    ranked = sorted(candidates, key=lambda item: (-item["quality"], item["timestamp"]))
    primary = ranked[0]
    backup = next(
        (item for item in ranked[1:] if abs(item["timestamp"] - primary["timestamp"]) >= backup_gap_seconds),
        ranked[1],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    for role, item in (("primary", primary), ("backup", backup)):
        path = output_dir / f"stable_{role}_{item['timestamp']:010.3f}s.jpg"
        if not cv2.imwrite(str(path), item["frame"], [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"stable_frame_write_failed:{path}")
        selected.append(
            {
                "role": role,
                "timestamp_seconds": round(float(item["timestamp"]), 3),
                "path": str(path.resolve()),
                "quality": round(float(item["quality"]), 4),
                "sharpness": round(float(item["sharpness"]), 3),
                "exposure": round(float(item["exposure"]), 4),
                "staticness": round(float(item["staticness"]), 4),
            }
        )
    return selected[0], selected[1]


def video_duration(video: Path) -> float:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"video_open_failed:{video}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"video_metadata_invalid:{video}")
    return frames / fps


def discover_video(source_name: str, video_root: Path) -> Path:
    direct = video_root / Path(source_name).name
    if direct.is_file():
        return direct.resolve()
    wanted = safe_id(source_name)
    matches = [
        path
        for path in video_root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and safe_id(path.name) == wanted
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"video_match_expected_one:{source_name}:{len(matches)}")
    return matches[0].resolve()


def generate(
    action_summary: Path,
    video_root: Path,
    output: Path,
    evidence_root: Path,
    wiring_output_root: Path,
    sample_interval: float,
    sample_limit: int,
    backup_gap: float,
) -> dict[str, Any]:
    summary = read_json(action_summary)
    videos: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for record in summary.get("records", []):
        if not isinstance(record, dict) or record.get("status") == "processing_failed":
            continue
        source_name = str(record.get("source_video_id") or "")
        result_path = resolve_from_root(str(record.get("result_path") or ""))
        video = discover_video(source_name, video_root)
        duration = video_duration(video)
        windows = episode_windows(observed_stages(read_json(result_path)), duration)
        video_id = safe_id(source_name)
        episodes: list[dict[str, Any]] = []
        for window in windows:
            stable_start, stable_end = window["recording_window_seconds"]
            times = candidate_timestamps(stable_start, stable_end, sample_interval, sample_limit)
            primary, backup = select_stable_frames(
                video,
                times,
                evidence_root / f"video_{video_id}" / window["episode_id"],
                backup_gap,
            )
            episodes.append(
                {
                    **window,
                    "stable_primary_seconds": primary["timestamp_seconds"],
                    "stable_backup_seconds": backup["timestamp_seconds"],
                    "stable_primary": primary["path"],
                    "stable_backup": backup["path"],
                    "adjudication": None,
                    "selection_diagnostics": {"primary": primary, "backup": backup, "candidate_count": len(times)},
                }
            )
        videos.append(
            {
                "video_id": video_id,
                "enabled": True,
                "source_video": str(video),
                "cache_dir": str((evidence_root / f"video_{video_id}" / "cache").resolve()),
                "episodes": episodes,
            }
        )
        diagnostics.append({"video_id": video_id, "episode_count": len(episodes), "source_video": str(video)})
    if not videos:
        raise ValueError("no_action_records_available")
    config = {
        "schema_version": "wiring_sequence_batch_config.generated.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_from_action_summary": str(action_summary.resolve()),
        "output_root": str(wiring_output_root.resolve()),
        "videos": videos,
        "diagnostics": diagnostics,
    }
    write_json(output, config)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-summary", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=ROOT / "data" / "videos")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--wiring-output-root", type=Path, required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--backup-gap-seconds", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_interval_seconds <= 0 or args.sample_limit < 2 or args.backup_gap_seconds < 0:
        raise ValueError("invalid_sampling_arguments")
    config = generate(
        args.action_summary.resolve(),
        args.video_root.resolve(),
        args.output.resolve(),
        args.evidence_root.resolve(),
        args.wiring_output_root.resolve(),
        args.sample_interval_seconds,
        args.sample_limit,
        args.backup_gap_seconds,
    )
    print(json.dumps({"status": "completed", "video_count": len(config["videos"]), "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
