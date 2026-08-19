"""Convert a calibrated pointer-angle candidate into a discrete dial tick candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2


def _nearest_tick(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _find_observation(payload: dict[str, Any], role: str, timestamp: float) -> dict[str, Any]:
    matches = [
        item
        for item in payload.get("observations", [])
        if str(item.get("role")) == role
        and abs(float(item.get("timestamp_seconds", -1.0)) - timestamp) < 1e-6
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {role} observation at {timestamp}s, found {len(matches)}")
    return matches[0]


def _alignment_status(review: dict[str, Any], role: str, timestamp: float) -> tuple[bool, str | None]:
    if isinstance(review.get("reviews"), list):
        matches = [
            item
            for item in review["reviews"]
            if str(item.get("role")) == role
            and abs(float(item.get("timestamp_seconds", -1.0)) - timestamp) < 1e-6
        ]
        if len(matches) != 1:
            return False, "No unique visual tick-grid review exists for this role and timestamp."
        item = matches[0]
        return bool(item.get("tick_grid_visual_alignment_confirmed")), item.get("reason")
    if str(review.get("role")) != role or abs(float(review.get("timestamp_seconds", -1.0)) - timestamp) >= 1e-6:
        return False, "Pointer-alignment review does not match this role and timestamp."
    return bool(review.get("pointer_geometry_valid")), review.get("evidence_insufficient_reason")


def _draw_tick_overlay(
    observation: dict[str, Any],
    output_path: Path,
    total_divisions: int,
    raw_tick: float | None,
    nearest_tick: int | None,
    alignment_valid: bool,
) -> None:
    image = cv2.imread(str(observation["original_face_copy_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(observation["original_face_copy_path"])
    pointer = observation["pointer"]
    geometry = observation["geometry"]
    pivot = tuple(float(value) for value in pointer["pivot"])
    zero = float(geometry["zero_angle_deg"])
    full = float(geometry["full_angle_deg"])
    direction = str(geometry["sweep_direction"])
    sweep = (full - zero) % 360.0 if direction == "increasing" else (zero - full) % 360.0
    radius = min(image.shape[1] * 0.47, image.shape[0] * 0.58)
    for index in range(total_divisions + 1):
        fraction = index / total_divisions
        angle = zero + sweep * fraction if direction == "increasing" else zero - sweep * fraction
        radians = math.radians(angle)
        outer = (
            int(round(pivot[0] + radius * math.cos(radians))),
            int(round(pivot[1] - radius * math.sin(radians))),
        )
        length = 17 if index % 5 == 0 else 10
        inner = (
            int(round(pivot[0] + (radius - length) * math.cos(radians))),
            int(round(pivot[1] - (radius - length) * math.sin(radians))),
        )
        color = (0, 215, 255) if index == nearest_tick else (220, 80, 220)
        cv2.line(image, inner, outer, color, 3 if index == nearest_tick else 1, cv2.LINE_AA)
        if index % 5 == 0:
            cv2.putText(image, str(index), (outer[0] + 3, outer[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    status = "alignment_confirmed" if alignment_valid else "alignment_not_confirmed"
    label = f"tick={raw_tick:.3f} nearest={nearest_tick} {status}" if raw_tick is not None else status
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (255, 255, 255), -1)
    cv2.putText(image, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 97]):
        raise OSError(f"Could not write {output_path}")


def count_ticks(
    geometry_path: Path,
    alignment_review_path: Path,
    output_dir: Path,
    role: str,
    timestamp: float,
    total_divisions: int,
    range_max_value: float,
    unit: str,
) -> dict[str, Any]:
    if total_divisions <= 0 or range_max_value <= 0:
        raise ValueError("total divisions and range max must be positive")
    geometry_payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    review = json.loads(alignment_review_path.read_text(encoding="utf-8"))
    observation = _find_observation(geometry_payload, role, timestamp)
    geometry = observation.get("geometry", {})
    ratio = geometry.get("raw_ratio")
    state = str(geometry.get("ratio_state", "calibration_missing"))
    raw_tick = float(ratio) * total_divisions if ratio is not None else None
    nearest_tick = None
    if raw_tick is not None and state == "within_scale":
        nearest_tick = min(total_divisions, max(0, _nearest_tick(raw_tick)))
    smallest_division = range_max_value / total_divisions
    provisional_reading = nearest_tick * smallest_division if nearest_tick is not None else None
    alignment_valid, alignment_reason = _alignment_status(review, role, timestamp)
    result = {
        "schema_version": "meter-tick-count-v1",
        "source_geometry_results": str(geometry_path.resolve()),
        "source_alignment_review": str(alignment_review_path.resolve()),
        "frame": observation["frame"],
        "timestamp_seconds": timestamp,
        "role": role,
        "pointer_angle_candidate_deg": observation.get("pointer", {}).get("angle_deg"),
        "pointer_alignment_confirmed": alignment_valid,
        "total_divisions": total_divisions,
        "raw_tick_index": round(raw_tick, 6) if raw_tick is not None else None,
        "nearest_tick_index": nearest_tick,
        "tick_state": state,
        "range_max_value": range_max_value,
        "smallest_division": round(smallest_division, 6),
        "unit": unit,
        "provisional_reading_from_tick_candidate": round(provisional_reading, 6) if provisional_reading is not None else None,
        "reading": round(provisional_reading, 6) if alignment_valid and provisional_reading is not None else None,
        "reading_status": "confirmed_tick_reading" if alignment_valid and provisional_reading is not None else "withheld_until_pointer_alignment",
        "reason": None if alignment_valid else alignment_reason,
        "qwen_called": False,
        "excel_accessed": False,
        "score_computed": False,
        "original_files_modified": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    overlay_path = output_dir / "tick_overlay.jpg"
    _draw_tick_overlay(observation, overlay_path, total_divisions, raw_tick, nearest_tick, alignment_valid)
    result["tick_overlay_path"] = str(overlay_path.resolve())
    result_path = output_dir / "tick_count_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Count discrete meter ticks from a pointer-angle candidate")
    parser.add_argument("--geometry-results", required=True, type=Path)
    parser.add_argument("--alignment-review", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--role", choices=("ammeter", "voltmeter"), required=True)
    parser.add_argument("--timestamp", required=True, type=float)
    parser.add_argument("--total-divisions", required=True, type=int)
    parser.add_argument("--range-max", required=True, type=float)
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    result = count_ticks(
        args.geometry_results.resolve(),
        args.alignment_review.resolve(),
        args.output_dir.resolve(),
        args.role,
        args.timestamp,
        args.total_divisions,
        args.range_max,
        args.unit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
