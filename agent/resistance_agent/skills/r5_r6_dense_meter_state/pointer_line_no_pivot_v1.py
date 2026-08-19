"""Detect a teaching-meter pointer as a line without a fixed pivot point."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _point(value: np.ndarray) -> list[float]:
    return [round(float(value[0]), 3), round(float(value[1]), 3)]


def _red_mask(face: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 65, 45]), np.array([16, 255, 255])),
        cv2.inRange(hsv, np.array([168, 55, 40]), np.array([179, 255, 255])),
    )
    # The black border of a red lead is also not pointer evidence.
    return cv2.dilate(mask, np.ones((31, 31), np.uint8), iterations=1)


def _sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    map_x = np.clip(x, 0, width - 1).astype(np.float32).reshape(1, -1)
    map_y = np.clip(y, 0, height - 1).astype(np.float32).reshape(1, -1)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR).reshape(-1)


def _longest_run(values: np.ndarray, max_gap: int = 3) -> int:
    best = 0
    start = 0
    gaps = 0
    for index, value in enumerate(values):
        if not bool(value):
            gaps += 1
        while gaps > max_gap and start <= index:
            if not bool(values[start]):
                gaps -= 1
            start += 1
        best = max(best, index - start + 1)
    return best


def _angle_in_range(angle: float, minimum: float, maximum: float, tolerance: float = 7.0) -> bool:
    low = min(minimum, maximum) - tolerance
    high = max(minimum, maximum) + tolerance
    return low <= angle <= high


def _ray_metrics(
    gray: np.ndarray,
    red: np.ndarray,
    anchor: np.ndarray,
    angle_deg: float,
) -> dict[str, float | np.ndarray]:
    height, width = gray.shape
    radians = math.radians(angle_deg)
    direction = np.float32([math.cos(radians), -math.sin(radians)])
    perpendicular = np.float32([-direction[1], direction[0]])
    limits = []
    if direction[0] > 1e-6:
        limits.append((width - 1.0 - anchor[0]) / direction[0])
    elif direction[0] < -1e-6:
        limits.append((0.0 - anchor[0]) / direction[0])
    if direction[1] > 1e-6:
        limits.append((height - 1.0 - anchor[1]) / direction[1])
    elif direction[1] < -1e-6:
        limits.append((0.0 - anchor[1]) / direction[1])
    positive = [value for value in limits if value > 0]
    ray_limit = min(positive) * 0.94 if positive else 0.0
    radii = np.arange(18.0, ray_limit, 1.0, dtype=np.float32)
    if len(radii) < 80:
        return {
            "score": 0.0,
            "dark_fraction": 0.0,
            "contrast_fraction": 0.0,
            "continuity": 0.0,
            "red_fraction": 1.0,
            "tip": anchor.copy(),
        }
    x = anchor[0] + radii * direction[0]
    y = anchor[1] + radii * direction[1]
    centers = np.vstack(
        [_sample(gray, x + offset * perpendicular[0], y + offset * perpendicular[1]) for offset in (-1.0, 0.0, 1.0)]
    ).min(axis=0)
    sides = np.vstack(
        [_sample(gray, x + offset * perpendicular[0], y + offset * perpendicular[1]) for offset in (-8.0, -6.0, 6.0, 8.0)]
    ).mean(axis=0)
    occluded = _sample(red, x, y) > 0
    valid = ~occluded
    if int(valid.sum()) < max(60, int(0.55 * len(valid))):
        return {
            "score": 0.0,
            "dark_fraction": 0.0,
            "contrast_fraction": 0.0,
            "continuity": 0.0,
            "red_fraction": float(occluded.mean()),
            "tip": anchor + ray_limit * direction,
        }
    center = centers[valid]
    side = sides[valid]
    support = (center < 178.0) & ((side - center) > 5.0)
    darkness = np.clip((190.0 - center) / 110.0, 0.0, 1.0)
    contrast = np.clip((side - center) / 60.0, 0.0, 1.0)
    dark_fraction = float(np.mean(center < 178.0))
    contrast_fraction = float(np.mean((side - center) > 6.0))
    continuity = float(_longest_run(support) / max(1, len(support)))
    score = 0.38 * float(darkness.mean()) + 0.28 * float(contrast.mean()) + 0.34 * continuity
    return {
        "score": score,
        "dark_fraction": dark_fraction,
        "contrast_fraction": contrast_fraction,
        "continuity": continuity,
        "red_fraction": float(occluded.mean()),
        "tip": anchor + ray_limit * direction,
    }


def detect_pointer_line(
    face: np.ndarray,
    zero_angle_deg: float,
    full_angle_deg: float,
) -> dict[str, Any]:
    """Find a long dark line crossing the scale and a broad lower-center hub zone."""
    gray_raw = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(cv2.createCLAHE(2.0, (8, 8)).apply(gray_raw), (3, 3), 0.0)
    red = _red_mask(face)
    prepared = gray.copy()
    prepared[red > 0] = 255
    edges = cv2.Canny(prepared, 45, 135)
    height, width = gray.shape
    edges[: int(0.04 * height)] = 0
    edges[int(0.91 * height) :] = 0
    edges[:, : int(0.06 * width)] = 0
    edges[:, int(0.94 * width) :] = 0
    lines = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 720.0,
        threshold=22,
        minLineLength=max(55, int(0.13 * height)),
        maxLineGap=28,
    )
    candidates: list[dict[str, Any]] = []
    anchor_y = 0.76 * height
    if lines is not None:
        for raw in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = map(float, raw)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < max(110.0, 0.21 * height) or abs(dy) < 12.0:
                continue
            angle = math.degrees(math.atan2(-dy, dx)) % 180.0
            if not _angle_in_range(angle, zero_angle_deg, full_angle_deg):
                continue
            parameter = (anchor_y - y1) / dy
            anchor_x = x1 + parameter * dx
            if not 0.26 * width <= anchor_x <= 0.74 * width:
                continue
            extrapolation = 0.0 if 0.0 <= parameter <= 1.0 else min(abs(parameter), abs(parameter - 1.0)) * length
            if extrapolation > 0.30 * height:
                continue
            y_top, y_bottom = min(y1, y2), max(y1, y2)
            if y_top > 0.52 * height or y_bottom < 0.43 * height:
                continue
            anchor = np.float32([anchor_x, anchor_y])
            metrics = _ray_metrics(gray, red, anchor, angle)
            vertical_span = (y_bottom - y_top) / height
            length_fraction = length / math.hypot(width, height)
            total_score = (
                float(metrics["score"])
                + 0.24 * min(1.0, length_fraction / 0.42)
                + 0.16 * min(1.0, vertical_span / 0.55)
                - 0.25 * float(metrics["red_fraction"])
            )
            candidates.append(
                {
                    "angle_deg": angle,
                    "anchor": anchor,
                    "tip": metrics["tip"],
                    "line": [int(x1), int(y1), int(x2), int(y2)],
                    "line_length_px": length,
                    "vertical_span_fraction": vertical_span,
                    "extrapolation_px": extrapolation,
                    "total_score": total_score,
                    **{key: value for key, value in metrics.items() if key != "tip"},
                }
            )
    candidates.sort(key=lambda item: float(item["total_score"]), reverse=True)
    clusters: list[dict[str, Any]] = []
    for candidate in candidates:
        cluster = next(
            (
                item
                for item in clusters
                if abs(float(item["representative"]["angle_deg"]) - float(candidate["angle_deg"])) <= 2.5
                and abs(float(item["representative"]["anchor"][0]) - float(candidate["anchor"][0])) <= 18.0
            ),
            None,
        )
        if cluster is None:
            clusters.append({"representative": candidate, "size": 1})
        else:
            cluster["size"] += 1
    clusters.sort(key=lambda item: float(item["representative"]["total_score"]), reverse=True)
    best_cluster = clusters[0] if clusters else None
    best = best_cluster["representative"] if best_cluster else None
    second_score = float(clusters[1]["representative"]["total_score"]) if len(clusters) > 1 else 0.0
    reasons: list[str] = []
    if best is None:
        reasons.append("no_long_dark_line_crosses_scale_and_hub_zone")
    else:
        if float(best["line_length_px"]) < 120.0:
            reasons.append("pointer_line_too_short")
        if float(best["dark_fraction"]) < 0.18:
            reasons.append("pointer_line_dark_support_too_sparse")
        if float(best["contrast_fraction"]) < 0.08:
            reasons.append("pointer_line_side_contrast_too_low")
        if float(best["continuity"]) < 0.10:
            reasons.append("pointer_line_not_continuous")
        if float(best["red_fraction"]) > 0.12:
            reasons.append("pointer_line_too_occluded_by_red_lead")
        if float(best["total_score"]) - second_score < 0.015:
            reasons.append("multiple_pointer_lines_ambiguous")
    if best is None:
        return {
            "detected": False,
            "state": "evidence_insufficient",
            "angle_deg": None,
            "anchor": None,
            "tip": None,
            "reasons": reasons,
            "candidate_count": 0,
            "localization_method": "pivot_free_long_dark_line",
        }
    return {
        "detected": not reasons,
        "state": "angle_candidate" if not reasons else "evidence_insufficient",
        "angle_deg": round(float(best["angle_deg"]), 6),
        "anchor": _point(best["anchor"]),
        "tip": _point(best["tip"]),
        "line": best["line"],
        "line_length_px": round(float(best["line_length_px"]), 6),
        "score": round(float(best["total_score"]), 6),
        "peak_margin": round(float(best["total_score"]) - second_score, 6),
        "dark_fraction": round(float(best["dark_fraction"]), 6),
        "contrast_fraction": round(float(best["contrast_fraction"]), 6),
        "continuity": round(float(best["continuity"]), 6),
        "red_occlusion_fraction": round(float(best["red_fraction"]), 6),
        "reasons": reasons,
        "candidate_count": len(candidates),
        "line_cluster_count": len(clusters),
        "supporting_line_count": int(best_cluster["size"]) if best_cluster else 0,
        "localization_method": "pivot_free_long_dark_line",
        "manual_review_required": True,
        "reading_value": None,
        "reading_computed": False,
    }
