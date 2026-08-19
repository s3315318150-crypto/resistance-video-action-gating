"""Pivot-free pointer detection with explicit scale-arc and hub-zone reach.

This detector is an experiment successor to ``pointer_line_no_pivot_v1``.  It
does not consume a fixed pivot.  A Hough segment must visibly span from the
upper scale region into the broad lower mechanical-axis region before it can
be scored as a pointer.  Short strokes contained in the central A/V glyph or
other printed text are rejected before candidate ranking.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from pointer_line_no_pivot_v1 import _point, _ray_metrics


def _angle_in_expanded_range(angle: float, zero: float, full: float, tolerance: float = 16.0) -> bool:
    return min(zero, full) - tolerance <= angle <= max(zero, full) + tolerance


def line_zone_diagnostics(line: list[int] | tuple[int, int, int, int], shape: tuple[int, ...]) -> dict[str, Any]:
    height, width = shape[:2]
    x1, y1, x2, y2 = map(float, line)
    top_y = min(y1, y2)
    bottom_y = max(y1, y2)
    scale_reached = top_y <= 0.42 * height
    hub_reached = bottom_y >= 0.62 * height

    samples = max(2, int(round(math.hypot(x2 - x1, y2 - y1))))
    xs = np.linspace(x1, x2, samples)
    ys = np.linspace(y1, y2, samples)
    glyph = (
        (xs >= 0.39 * width)
        & (xs <= 0.61 * width)
        & (ys >= 0.40 * height)
        & (ys <= 0.70 * height)
    )
    glyph_fraction = float(np.mean(glyph))
    contained_in_glyph = bool(glyph_fraction >= 0.72 and not (scale_reached and hub_reached))
    return {
        "scale_arc_reached": scale_reached,
        "hub_zone_reached": hub_reached,
        "line_top_y_ratio": round(top_y / height, 6),
        "line_bottom_y_ratio": round(bottom_y / height, 6),
        "glyph_box_overlap_fraction": round(glyph_fraction, 6),
        "line_contained_in_center_glyph": contained_in_glyph,
    }


def detect_pointer_line_arc_hub(
    face: np.ndarray,
    occlusion_mask: np.ndarray,
    zero_angle_deg: float,
    full_angle_deg: float,
) -> dict[str, Any]:
    """Find a dark segment that visibly joins the scale arc and hub zone."""
    gray_raw = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(cv2.createCLAHE(2.0, (8, 8)).apply(gray_raw), (3, 3), 0.0)
    prepared = gray.copy()
    prepared[occlusion_mask > 0] = 255
    edges = cv2.Canny(prepared, 42, 128)
    height, width = gray.shape
    edges[: int(0.04 * height)] = 0
    edges[int(0.91 * height) :] = 0
    edges[:, : int(0.05 * width)] = 0
    edges[:, int(0.95 * width) :] = 0

    lines = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 720.0,
        threshold=20,
        minLineLength=max(80, int(0.17 * height)),
        maxLineGap=32,
    )
    candidates: list[dict[str, Any]] = []
    rejected_counts = {
        "angle_outside_expanded_scale_sector": 0,
        "line_does_not_reach_scale_arc": 0,
        "line_does_not_reach_hub_zone": 0,
        "line_contained_in_center_glyph": 0,
        "line_misses_dynamic_hub_x_zone": 0,
    }
    anchor_y = 0.76 * height
    if lines is not None:
        for raw in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = map(float, raw)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < max(105.0, 0.20 * height) or abs(dy) < 12.0:
                continue
            angle = math.degrees(math.atan2(-dy, dx)) % 180.0
            if not _angle_in_expanded_range(angle, zero_angle_deg, full_angle_deg):
                rejected_counts["angle_outside_expanded_scale_sector"] += 1
                continue
            geometry = line_zone_diagnostics([int(x1), int(y1), int(x2), int(y2)], face.shape)
            if not geometry["scale_arc_reached"]:
                rejected_counts["line_does_not_reach_scale_arc"] += 1
                continue
            if not geometry["hub_zone_reached"]:
                rejected_counts["line_does_not_reach_hub_zone"] += 1
                continue
            if geometry["line_contained_in_center_glyph"]:
                rejected_counts["line_contained_in_center_glyph"] += 1
                continue

            parameter = (anchor_y - y1) / dy
            anchor_x = x1 + parameter * dx
            if not 0.22 * width <= anchor_x <= 0.78 * width:
                rejected_counts["line_misses_dynamic_hub_x_zone"] += 1
                continue
            extrapolation = 0.0 if 0.0 <= parameter <= 1.0 else min(abs(parameter), abs(parameter - 1.0)) * length
            if extrapolation > 0.12 * height:
                rejected_counts["line_misses_dynamic_hub_x_zone"] += 1
                continue

            anchor = np.float32([anchor_x, anchor_y])
            metrics = _ray_metrics(gray, occlusion_mask, anchor, angle)
            vertical_span = abs(y2 - y1) / height
            length_fraction = length / math.hypot(width, height)
            total_score = (
                float(metrics["score"])
                + 0.28 * min(1.0, length_fraction / 0.42)
                + 0.24 * min(1.0, vertical_span / 0.55)
                - 0.30 * float(metrics["red_fraction"])
                - 0.12 * float(geometry["glyph_box_overlap_fraction"])
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
                    **geometry,
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
    if best is None:
        reasons = ["no_dark_line_visibly_connects_scale_arc_and_hub_zone"]
        return {
            "detected": False,
            "state": "evidence_insufficient",
            "angle_deg": None,
            "anchor": None,
            "tip": None,
            "line": None,
            "reasons": reasons,
            "candidate_count": 0,
            "rejected_candidate_counts": rejected_counts,
            "localization_method": "pivot_free_scale_arc_to_dynamic_hub_line",
            "search_angle_tolerance_deg": 16.0,
            "reading_value": None,
            "reading_computed": False,
        }

    reasons: list[str] = []
    if float(best["line_length_px"]) < 120.0:
        reasons.append("pointer_line_too_short")
    if float(best["dark_fraction"]) < 0.18:
        reasons.append("pointer_line_dark_support_too_sparse")
    if float(best["contrast_fraction"]) < 0.08:
        reasons.append("pointer_line_side_contrast_too_low")
    if float(best["continuity"]) < 0.10:
        reasons.append("pointer_line_not_continuous")
    if float(best["red_fraction"]) > 0.12:
        reasons.append("pointer_line_too_occluded_by_wire_mask")
    if float(best["total_score"]) - second_score < 0.015:
        reasons.append("multiple_pointer_lines_ambiguous")
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
        "wire_occlusion_fraction": round(float(best["red_fraction"]), 6),
        "scale_arc_reached": bool(best["scale_arc_reached"]),
        "hub_zone_reached": bool(best["hub_zone_reached"]),
        "line_top_y_ratio": best["line_top_y_ratio"],
        "line_bottom_y_ratio": best["line_bottom_y_ratio"],
        "glyph_box_overlap_fraction": best["glyph_box_overlap_fraction"],
        "line_contained_in_center_glyph": bool(best["line_contained_in_center_glyph"]),
        "reasons": reasons,
        "candidate_count": len(candidates),
        "line_cluster_count": len(clusters),
        "supporting_line_count": int(best_cluster["size"]) if best_cluster else 0,
        "rejected_candidate_counts": rejected_counts,
        "localization_method": "pivot_free_scale_arc_to_dynamic_hub_line",
        "search_angle_tolerance_deg": 16.0,
        "manual_review_required": True,
        "reading_value": None,
        "reading_computed": False,
    }
