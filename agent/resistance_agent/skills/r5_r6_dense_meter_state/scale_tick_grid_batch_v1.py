"""Run per-frame printed-scale detection and robust temporal grid consensus."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import cv2

from scale_tick_grid_v1 import detect_scale_ticks, draw_overlay, pointer_grid_position


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _robust_endpoint_indices(values: list[float]) -> set[int]:
    if not values:
        return set()
    center = float(median(values))
    deviations = [abs(value - center) for value in values]
    mad = float(median(deviations))
    tolerance = max(3.0, 3.0 * mad)
    return {index for index, value in enumerate(values) if abs(value - center) <= tolerance}


def _mode(values: list[float]) -> float | None:
    if not values:
        return None
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def consensus_grid(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        item
        for item in frame_results
        if item.get("grid", {}).get("fitted")
        and item.get("grid", {}).get("state") == "grid_candidate"
    ]
    if not usable:
        return {"status": "no_grid_candidate", "used_frame_count": 0}
    zeros = [float(item["grid"]["zero_angle_deg"]) for item in usable]
    fulls = [float(item["grid"]["full_angle_deg"]) for item in usable]
    keep = _robust_endpoint_indices(zeros) & _robust_endpoint_indices(fulls)
    retained = [item for index, item in enumerate(usable) if index in keep]
    rejected = [item for index, item in enumerate(usable) if index not in keep]
    if not retained:
        return {"status": "no_consensus_after_endpoint_outlier_rejection", "used_frame_count": 0}
    zero = float(median([float(item["grid"]["zero_angle_deg"]) for item in retained]))
    full = float(median([float(item["grid"]["full_angle_deg"]) for item in retained]))
    total = int(median([int(item["grid"]["total_major_divisions"]) for item in retained]))
    positions = []
    for item in retained:
        angle = float(item["pointer_angle_deg"])
        raw = (zero - angle) / (zero - full) * total
        positions.append(
            {
                "timestamp_seconds": item.get("timestamp_seconds"),
                "pointer_angle_deg": round(angle, 6),
                "raw_major_division_index": round(raw, 6),
                "nearest_major_division_index": min(total, max(0, int(math.floor(raw + 0.5)))),
            }
        )
    median_index = float(median([item["nearest_major_division_index"] for item in positions]))
    consensus_index = int(math.floor(median_index + 0.5))
    ranges = [float(item["range_max_value"]) for item in retained if item.get("range_max_value") is not None]
    range_max = _mode(ranges)
    reading = consensus_index / total * range_max if range_max is not None else None
    return {
        "status": "grid_consensus_candidate",
        "method": "per_frame_printed_tick_fit_then_mad_endpoint_rejection_and_median",
        "fixed_pivot_used": False,
        "input_candidate_count": len(usable),
        "used_frame_count": len(retained),
        "rejected_frame_count": len(rejected),
        "rejected_timestamps_seconds": [item.get("timestamp_seconds") for item in rejected],
        "consensus_zero_angle_deg": round(zero, 6),
        "consensus_full_angle_deg": round(full, 6),
        "total_major_divisions": total,
        "frame_positions": positions,
        "consensus_major_division_index": consensus_index,
        "range_max_value": range_max,
        "reading_candidate": round(reading, 6) if reading is not None else None,
    }


def run_batch(source_result: Path, role: str, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    source = _load_json(source_result)
    videos = [item for item in source.get("videos", []) if isinstance(item, dict)]
    if len(videos) != 1:
        raise ValueError(f"Expected one current-run video record, found {len(videos)}")
    observations = videos[0].get("roles", {}).get(role, {}).get("observations", [])
    frame_results = []
    for sequence, observation in enumerate(observations, start=1):
        pointer = observation.get("pointer", {})
        face_value = observation.get("rectified_face_path")
        if not pointer.get("detected") or pointer.get("anchor") is None or pointer.get("angle_deg") is None or not face_value:
            continue
        face_path = Path(face_value)
        face = cv2.imread(str(face_path), cv2.IMREAD_COLOR)
        if face is None:
            continue
        ticks = detect_scale_ticks(face, pointer["anchor"])
        pointer_position = pointer_grid_position(ticks["regular_grid"], float(pointer["angle_deg"]))
        timestamp = observation.get("timestamp_seconds")
        overlay_path = output_dir / f"frame_{sequence:04d}_{float(timestamp or 0):09.3f}s_{role}_grid.jpg"
        if not cv2.imwrite(
            str(overlay_path), draw_overlay(face, ticks, float(pointer["angle_deg"])),
            [cv2.IMWRITE_JPEG_QUALITY, 97],
        ):
            raise OSError(overlay_path)
        frame_results.append(
            {
                "timestamp_seconds": timestamp,
                "source_face": str(face_path.resolve()),
                "pointer_angle_deg": pointer["angle_deg"],
                "dynamic_pointer_anchor": pointer["anchor"],
                "grid": ticks["regular_grid"],
                "per_frame_position": pointer_position,
                "range_max_value": observation.get("range", {}).get("range_max_value"),
                "selected_port": observation.get("range", {}).get("selected_port"),
                "overlay": str(overlay_path.resolve()),
            }
        )
    result = {
        "schema_version": "scale-tick-grid-batch-v1",
        "source_result": str(source_result.resolve()),
        "role": role,
        "frame_results": frame_results,
        "consensus": consensus_grid(frame_results),
        "fixed_pivot_used": False,
        "per_video_angle_override_used": False,
        "qwen_called": False,
        "excel_accessed": False,
        "score_computed": False,
    }
    result_path = output_dir / "scale_tick_grid_batch_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automatic printed-grid reading across meter frames")
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("ammeter", "voltmeter"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_batch(args.source_result.resolve(), args.role, args.output_dir.resolve())
    print(json.dumps(
        {
            "output": str((args.output_dir.resolve() / "scale_tick_grid_batch_result.json")),
            "consensus": result["consensus"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
