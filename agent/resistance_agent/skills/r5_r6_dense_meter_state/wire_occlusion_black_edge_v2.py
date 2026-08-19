"""Mask red leads together with their immediately adjacent dark borders.

The mask is an occlusion mask only.  It never fills, reconstructs, or
interpolates a pointer.  Dark pixels are added only inside a narrow band around
retained red-wire components, so unrelated black scale marks remain evidence.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

import pointer_line_no_pivot_v1 as pointer_detector


def _angle_difference(first: float, second: float) -> float:
    difference = abs((first - second) % 180.0)
    return min(difference, 180.0 - difference)


def _retained_red_components(face: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    raw = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 65, 45]), np.array([16, 255, 255])),
        cv2.inRange(hsv, np.array([168, 55, 40]), np.array([179, 255, 255])),
    )
    interior = np.zeros(raw.shape, dtype=np.uint8)
    interior[12 : int(face.shape[0] * 0.86), 12 : face.shape[1] - 12] = 255
    raw = cv2.bitwise_and(raw, interior)
    raw = cv2.morphologyEx(
        raw,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats((raw > 0).astype(np.uint8), 8)
    kept = np.zeros_like(raw)
    components: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        aspect = max(width, height) / max(1.0, min(width, height))
        if area < 35 or not (area >= 180 or (aspect >= 2.0 and max(width, height) >= 24)):
            continue
        component_mask = labels == label
        kept[component_mask] = 255
        points = np.column_stack(np.where(component_mask)[::-1]).astype(np.float32)
        angle = None
        if len(points) >= 8:
            centered = points - points.mean(axis=0, keepdims=True)
            covariance = centered.T @ centered / max(1, len(centered) - 1)
            values, vectors = np.linalg.eigh(covariance)
            axis = vectors[:, int(np.argmax(values))]
            angle = math.degrees(math.atan2(-float(axis[1]), float(axis[0]))) % 180.0
        components.append(
            {
                "label": label,
                "bbox": [x, y, width, height],
                "area": area,
                "aspect_ratio": round(aspect, 6),
                "orientation_deg": round(angle, 6) if angle is not None else None,
            }
        )
    return kept, components


def detect_wire_black_edge_mask(
    face: np.ndarray,
    red_core_dilate_px: int = 5,
    dark_edge_search_px: int = 18,
    dark_value_max: int = 158,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a joint red-body and adjacent-dark-edge occlusion mask."""
    red_core, components = _retained_red_components(face)
    core_radius = max(1, int(red_core_dilate_px))
    search_radius = max(core_radius + 1, int(dark_edge_search_px))
    core = cv2.dilate(
        red_core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * core_radius + 1, 2 * core_radius + 1)),
    )
    search = cv2.dilate(
        red_core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * search_radius + 1, 2 * search_radius + 1)),
    )

    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Cable borders can be neutral black or dark, highly saturated red.  The
    # value limit captures both while the narrow search band prevents scale
    # text elsewhere from entering the mask.
    dark = np.where((gray <= dark_value_max) | (hsv[:, :, 2] <= dark_value_max - 8), 255, 0).astype(np.uint8)
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    adjacent_dark = cv2.bitwise_and(dark, search)
    adjacent_dark = cv2.bitwise_and(adjacent_dark, cv2.bitwise_not(core))

    # Retain only dark groups that actually touch the conservative core after
    # a small bridge.  This removes nearby text that merely happens to lie in
    # the wider search band.
    bridge = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((adjacent_dark > 0).astype(np.uint8), 8)
    retained_dark = np.zeros_like(adjacent_dark)
    dark_components = 0
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 5 or not np.any(component & (bridge > 0)):
            continue
        retained_dark[component] = 255
        dark_components += 1

    # Perspective warps use exact black for source pixels outside the frame.
    # Treat only border-connected exact-black regions as invalid evidence.
    warp_black = np.where(np.max(face, axis=2) <= 4, 255, 0).astype(np.uint8)
    invalid_warp = np.zeros_like(warp_black)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((warp_black > 0).astype(np.uint8), 8)
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        touches_border = x == 0 or y == 0 or x + width >= face.shape[1] or y + height >= face.shape[0]
        if touches_border and area >= 20:
            invalid_warp[labels == label] = 255
    if np.any(invalid_warp):
        invalid_warp = cv2.dilate(invalid_warp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    joint = cv2.bitwise_or(core, retained_dark)
    joint = cv2.bitwise_or(joint, invalid_warp)
    joint = cv2.morphologyEx(
        joint,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    stats_payload = {
        "red_core_fraction": round(float(np.mean(red_core > 0)), 6),
        "red_body_fraction": round(float(np.mean(core > 0)), 6),
        "adjacent_dark_fraction": round(float(np.mean(retained_dark > 0)), 6),
        "joint_mask_fraction": round(float(np.mean(joint > 0)), 6),
        "invalid_warp_fraction": round(float(np.mean(invalid_warp > 0)), 6),
        # Compatibility with dense_wire_search_v2 candidate sorting.
        "wire_fraction": round(float(np.mean(red_core > 0)), 6),
        "dilated_fraction": round(float(np.mean(joint > 0)), 6),
        "component_count": len(components),
        "dark_component_count": dark_components,
        "red_core_dilate_px": core_radius,
        "dark_edge_search_px": search_radius,
        "dark_value_max": int(dark_value_max),
        "components": components,
        "method": "hsv_red_components_plus_connected_adjacent_dark_edges_no_inpaint",
    }
    return joint, stats_payload


def line_wire_diagnostics(
    line: list[int] | tuple[int, int, int, int] | None,
    joint_mask: np.ndarray,
    red_core: np.ndarray,
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    if line is None:
        return {
            "line_mask_overlap_fraction": None,
            "line_near_red_fraction": None,
            "nearest_wire_angle_difference_deg": None,
            "pointer_parallel_to_wire_edge": False,
        }
    x1, y1, x2, y2 = map(float, line)
    count = max(2, int(round(math.hypot(x2 - x1, y2 - y1))))
    xs = np.clip(np.rint(np.linspace(x1, x2, count)).astype(int), 0, joint_mask.shape[1] - 1)
    ys = np.clip(np.rint(np.linspace(y1, y2, count)).astype(int), 0, joint_mask.shape[0] - 1)
    overlap = float(np.mean(joint_mask[ys, xs] > 0))
    near_red = cv2.dilate(red_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49)))
    near_fraction = float(np.mean(near_red[ys, xs] > 0))
    line_angle = math.degrees(math.atan2(-(y2 - y1), x2 - x1)) % 180.0
    differences = [
        _angle_difference(line_angle, float(item["orientation_deg"]))
        for item in components
        if item.get("orientation_deg") is not None
    ]
    nearest_difference = min(differences) if differences else None
    parallel = bool(
        nearest_difference is not None
        and nearest_difference <= 12.0
        and near_fraction >= 0.45
    )
    return {
        "line_angle_deg": round(line_angle, 6),
        "line_mask_overlap_fraction": round(overlap, 6),
        "line_near_red_fraction": round(near_fraction, 6),
        "nearest_wire_angle_difference_deg": round(nearest_difference, 6) if nearest_difference is not None else None,
        "pointer_parallel_to_wire_edge": parallel,
    }


def detect_pointer_with_wire_black_edge_mask(
    face: np.ndarray,
    zero_angle: float,
    full_angle: float,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    mask, mask_stats = detect_wire_black_edge_mask(face)
    red_core, components = _retained_red_components(face)
    original = pointer_detector._red_mask
    pointer_detector._red_mask = lambda _face: mask
    try:
        result = pointer_detector.detect_pointer_line(face, zero_angle, full_angle)
    finally:
        pointer_detector._red_mask = original
    diagnostics = line_wire_diagnostics(result.get("line"), mask, red_core, components)
    result.update(diagnostics)
    reasons = list(result.get("reasons", []))
    if diagnostics["pointer_parallel_to_wire_edge"] and "pointer_parallel_to_wire_edge" not in reasons:
        reasons.append("pointer_parallel_to_wire_edge")
    if (diagnostics["line_mask_overlap_fraction"] or 0.0) > 0.18 and "pointer_overlaps_wire_black_edge_mask" not in reasons:
        reasons.append("pointer_overlaps_wire_black_edge_mask")
    result["reasons"] = reasons
    result["detected"] = bool(result.get("detected")) and not reasons
    result["state"] = "angle_candidate" if result["detected"] else "evidence_insufficient"
    result["occlusion_method"] = "red_body_plus_adjacent_dark_edge_no_inpaint"
    return mask, mask_stats, result
