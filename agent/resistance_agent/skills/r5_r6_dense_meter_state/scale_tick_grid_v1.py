"""Detect printed meter scale ticks around a dynamically detected pointer anchor.

This module does not use a template pivot or a per-video zero-angle override.
Short printed tick segments are detected on each rectified face and ordered by
their direction from the current pointer line's lower endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _line_distance(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    direction = second - first
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return float("inf")
    cross = float(direction[0] * (point - first)[1] - direction[1] * (point - first)[0])
    return abs(cross) / length


def _angle_from_anchor(anchor: np.ndarray, point: np.ndarray) -> float:
    delta = point - anchor
    return math.degrees(math.atan2(-float(delta[1]), float(delta[0]))) % 360.0


def _cluster_angles(candidates: list[dict[str, Any]], tolerance_deg: float) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: item["angle_deg"], reverse=True)
    clusters: list[list[dict[str, Any]]] = []
    for candidate in ordered:
        if not clusters or abs(candidate["angle_deg"] - np.mean([item["angle_deg"] for item in clusters[-1]])) > tolerance_deg:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)
    result = []
    for cluster in clusters:
        best = max(cluster, key=lambda item: (item["length_px"], -item["anchor_distance_px"]))
        result.append(
            {
                **best,
                "angle_deg": round(float(np.median([item["angle_deg"] for item in cluster])), 6),
                "cluster_size": len(cluster),
            }
        )
    return result


def radial_dark_response(
    gray: np.ndarray,
    anchor: np.ndarray,
    angle_values: np.ndarray,
    min_radius_px: float,
    max_radius_px: float,
) -> np.ndarray:
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    radii = np.arange(min_radius_px, max_radius_px + 0.5, 1.0, dtype=np.float64)
    window = np.ones(13, dtype=np.float64) / 13.0
    result = np.zeros(len(angle_values), dtype=np.float64)
    height, width = gray.shape
    for index, angle in enumerate(angle_values):
        radians = math.radians(float(angle))
        xs = np.rint(anchor[0] + radii * math.cos(radians)).astype(np.int32)
        ys = np.rint(anchor[1] - radii * math.sin(radians)).astype(np.int32)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        samples = np.zeros_like(radii)
        samples[valid] = binary[ys[valid], xs[valid]].astype(np.float64) / 255.0
        continuous = np.convolve(samples, window, mode="valid")
        result[index] = float(np.max(continuous)) if continuous.size else 0.0
    return cv2.GaussianBlur(result.reshape(1, -1), (1, 0), sigmaX=0, sigmaY=0).reshape(-1)


def _interpolate_response(angle_values: np.ndarray, response: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.interp(targets, angle_values, response)


def fit_regular_tick_grid(
    angle_values: np.ndarray,
    response: np.ndarray,
    *,
    total_major_divisions: int = 15,
    zero_search_deg: tuple[float, float] = (112.0, 132.0),
    full_search_deg: tuple[float, float] = (38.0, 65.0),
    search_step_deg: float = 0.25,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for zero in np.arange(zero_search_deg[0], zero_search_deg[1] + 1e-9, search_step_deg):
        for full in np.arange(full_search_deg[0], full_search_deg[1] + 1e-9, search_step_deg):
            sweep = zero - full
            if not 58.0 <= sweep <= 92.0:
                continue
            ticks = np.linspace(zero, full, total_major_divisions + 1)
            gaps = (ticks[:-1] + ticks[1:]) / 2.0
            tick_values = _interpolate_response(angle_values, response, ticks)
            gap_values = _interpolate_response(angle_values, response, gaps)
            contrast = float(np.mean(tick_values) - 0.55 * np.mean(gap_values))
            coverage = float(np.mean(tick_values >= 0.46))
            endpoint_support = float((tick_values[0] + tick_values[-1]) / 2.0)
            score = contrast + 0.18 * coverage + 0.08 * endpoint_support
            candidate = {
                "zero_angle_deg": float(zero),
                "full_angle_deg": float(full),
                "tick_angles_deg": ticks,
                "tick_responses": tick_values,
                "gap_response_mean": float(np.mean(gap_values)),
                "coverage": coverage,
                "score": score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    if best is None:
        return {"fitted": False, "reason": "no_grid_search_candidate"}
    return {
        "fitted": True,
        "method": "per_frame_regular_grid_fit_on_radial_dark_response",
        "total_major_divisions": total_major_divisions,
        "zero_angle_deg": round(best["zero_angle_deg"], 6),
        "full_angle_deg": round(best["full_angle_deg"], 6),
        "tick_angles_deg": [round(float(value), 6) for value in best["tick_angles_deg"]],
        "tick_responses": [round(float(value), 6) for value in best["tick_responses"]],
        "gap_response_mean": round(best["gap_response_mean"], 6),
        "coverage": round(best["coverage"], 6),
        "fit_score": round(best["score"], 6),
        "state": "grid_candidate" if best["coverage"] >= 0.5 else "grid_weak",
    }


def pointer_grid_position(grid: dict[str, Any], pointer_angle_deg: float) -> dict[str, Any]:
    if not grid.get("fitted"):
        return {"matched": False, "reason": "grid_not_fitted"}
    zero = float(grid["zero_angle_deg"])
    full = float(grid["full_angle_deg"])
    total = int(grid["total_major_divisions"])
    raw = (zero - float(pointer_angle_deg)) / (zero - full) * total
    nearest = int(math.floor(raw + 0.5))
    return {
        "matched": -0.5 <= raw <= total + 0.5,
        "pointer_angle_deg": round(float(pointer_angle_deg), 6),
        "raw_major_division_index": round(raw, 6),
        "nearest_major_division_index": min(total, max(0, nearest)),
        "total_major_divisions": total,
    }


def detect_scale_ticks(
    face: np.ndarray,
    anchor: list[float] | tuple[float, float],
    *,
    angle_min_deg: float = 35.0,
    angle_max_deg: float = 135.0,
    min_radius_px: float = 180.0,
    max_radius_px: float = 330.0,
    max_anchor_distance_px: float = 18.0,
    min_line_length_px: float = 6.0,
    max_line_length_px: float = 45.0,
    cluster_tolerance_deg: float = 1.0,
) -> dict[str, Any]:
    if face is None or face.size == 0:
        raise ValueError("face image is empty")
    anchor_array = np.asarray(anchor, dtype=np.float64)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.Canny(blurred, 35, 105)

    height, width = gray.shape
    mask = np.zeros_like(edges)
    y_limit = min(height, int(round(anchor_array[1] - 105.0)))
    cv2.rectangle(mask, (0, 35), (width - 1, max(36, y_limit)), 255, -1)
    edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720.0, threshold=10, minLineLength=5, maxLineGap=3)

    line_rows = np.empty((0, 4), dtype=np.int32) if lines is None else np.asarray(lines).reshape(-1, 4)
    candidates: list[dict[str, Any]] = []
    for raw in line_rows:
        first = np.asarray(raw[:2], dtype=np.float64)
        second = np.asarray(raw[2:], dtype=np.float64)
        length = float(np.linalg.norm(second - first))
        if not (min_line_length_px <= length <= max_line_length_px):
            continue
        midpoint = (first + second) / 2.0
        radius = float(np.linalg.norm(midpoint - anchor_array))
        if not (min_radius_px <= radius <= max_radius_px):
            continue
        angle = _angle_from_anchor(anchor_array, midpoint)
        if not (angle_min_deg <= angle <= angle_max_deg):
            continue
        anchor_distance = _line_distance(anchor_array, first, second)
        if anchor_distance > max_anchor_distance_px:
            continue
        candidates.append(
            {
                "line": [int(value) for value in raw],
                "midpoint": [round(float(value), 3) for value in midpoint],
                "angle_deg": angle,
                "radius_px": round(radius, 3),
                "length_px": round(length, 3),
                "anchor_distance_px": round(anchor_distance, 3),
            }
        )

    clusters = _cluster_angles(candidates, cluster_tolerance_deg)
    angle_values = np.arange(angle_min_deg, angle_max_deg + 1e-9, 0.1, dtype=np.float64)
    response = radial_dark_response(gray, anchor_array, angle_values, min_radius_px, max_radius_px)
    regular_grid = fit_regular_tick_grid(angle_values, response)
    return {
        "method": "per_frame_hough_short_radial_ticks_dynamic_pointer_anchor",
        "fixed_pivot_used": False,
        "anchor": [round(float(value), 3) for value in anchor_array],
        "raw_line_count": int(len(line_rows)),
        "radial_candidate_count": len(candidates),
        "tick_cluster_count": len(clusters),
        "tick_clusters": clusters,
        "regular_grid": regular_grid,
        "radial_response": {
            "angle_start_deg": round(float(angle_values[0]), 6),
            "angle_step_deg": 0.1,
            "values": [round(float(value), 6) for value in response],
        },
    }


def match_pointer_to_ticks(ticks: dict[str, Any], pointer_angle_deg: float) -> dict[str, Any]:
    clusters = ticks.get("tick_clusters", [])
    if len(clusters) < 2:
        return {"matched": False, "reason": "fewer_than_two_tick_clusters"}
    angles = np.asarray([float(item["angle_deg"]) for item in clusters], dtype=np.float64)
    order = np.argsort(-angles)
    angles = angles[order]
    nearest = int(np.argmin(np.abs(angles - float(pointer_angle_deg))))
    return {
        "matched": True,
        "pointer_angle_deg": round(float(pointer_angle_deg), 6),
        "nearest_detected_cluster_index": nearest,
        "nearest_detected_tick_angle_deg": round(float(angles[nearest]), 6),
        "angle_error_deg": round(abs(float(angles[nearest]) - float(pointer_angle_deg)), 6),
        "ordered_detected_tick_angles_deg": [round(float(value), 6) for value in angles],
        "semantics": "diagnostic cluster order only; endpoints must be established from a complete regular tick run",
    }


def draw_overlay(face: np.ndarray, ticks: dict[str, Any], pointer_angle_deg: float | None) -> np.ndarray:
    output = face.copy()
    anchor = tuple(int(round(value)) for value in ticks["anchor"])
    cv2.circle(output, anchor, 6, (255, 80, 0), -1, cv2.LINE_AA)
    for index, item in enumerate(ticks["tick_clusters"]):
        x1, y1, x2, y2 = item["line"]
        cv2.line(output, (x1, y1), (x2, y2), (255, 0, 255), 2, cv2.LINE_AA)
        midpoint = tuple(int(round(value)) for value in item["midpoint"])
        cv2.putText(output, str(index), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 90, 220), 1, cv2.LINE_AA)
    grid = ticks.get("regular_grid", {})
    if grid.get("fitted"):
        for index, angle in enumerate(grid["tick_angles_deg"]):
            radians = math.radians(float(angle))
            inner = (
                int(round(anchor[0] + 215 * math.cos(radians))),
                int(round(anchor[1] - 215 * math.sin(radians))),
            )
            outer = (
                int(round(anchor[0] + 285 * math.cos(radians))),
                int(round(anchor[1] - 285 * math.sin(radians))),
            )
            cv2.line(output, inner, outer, (0, 210, 255), 1, cv2.LINE_AA)
            cv2.putText(output, str(index), outer, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 80, 200), 1, cv2.LINE_AA)
    if pointer_angle_deg is not None:
        radians = math.radians(pointer_angle_deg)
        tip = (
            int(round(anchor[0] + 290 * math.cos(radians))),
            int(round(anchor[1] - 290 * math.sin(radians))),
        )
        cv2.line(output, anchor, tip, (0, 200, 0), 2, cv2.LINE_AA)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect printed scale ticks on one rectified meter face")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--anchor-x", required=True, type=float)
    parser.add_argument("--anchor-y", required=True, type=float)
    parser.add_argument("--pointer-angle", type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.image.resolve()), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    result = detect_scale_ticks(image, [args.anchor_x, args.anchor_y])
    if args.pointer_angle is not None:
        result["pointer_match"] = match_pointer_to_ticks(result, args.pointer_angle)
        result["regular_grid_pointer"] = pointer_grid_position(result["regular_grid"], args.pointer_angle)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "scale_tick_grid_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    overlay = draw_overlay(image, result, args.pointer_angle)
    overlay_path = args.output_dir / "scale_tick_grid_overlay.jpg"
    if not cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 97]):
        raise OSError(overlay_path)
    print(json.dumps({"result": str(result_path.resolve()), "overlay": str(overlay_path.resolve()), **{key: result[key] for key in ("raw_line_count", "radial_candidate_count", "tick_cluster_count")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
