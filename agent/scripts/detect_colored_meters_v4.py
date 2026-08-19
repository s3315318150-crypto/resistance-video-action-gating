#!/usr/bin/env python3
"""Conservative terminal-color anchored meter detection V4.

This local detector uses the red/green terminal board as an anchor and searches
for a structured, neutral/light dial interior near it. V4 adds face-level
deduplication, component and housing checks, explicit insufficiency reasons,
and independent face/terminal ROI artifacts. It is intentionally label-blind:
it does not call a model, read Excel, inspect video metadata, or assign a
score. Pointer angles are geometric candidates only; state calibration is
left to a later temporal stage.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA_VERSION = "1.2"
ARTIFACT_TYPE = "colored_meter_detection_v4"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    denom = aw * ah + bw * bh - inter
    return float(inter / denom) if denom else 0.0


def union_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = min(ax, bx), min(ay, by)
    x2, y2 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return x1, y1, x2 - x1, y2 - y1


def make_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, np.array([0, 70, 40], np.uint8), np.array([30, 255, 255], np.uint8))
    orange |= cv2.inRange(hsv, np.array([165, 65, 40], np.uint8), np.array([179, 255, 255], np.uint8))
    red = cv2.inRange(hsv, np.array([0, 95, 45], np.uint8), np.array([14, 255, 255], np.uint8))
    red |= cv2.inRange(hsv, np.array([168, 80, 40], np.uint8), np.array([179, 255, 255], np.uint8))
    green = cv2.inRange(hsv, np.array([35, 45, 20], np.uint8), np.array([100, 255, 230], np.uint8))
    white = cv2.inRange(hsv, np.array([0, 0, 125], np.uint8), np.array([179, 105, 255], np.uint8))

    # Keep thin wires from being connected across the entire scene. The
    # terminal board itself is large enough to survive these kernels.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    orange = cv2.morphologyEx(orange, cv2.MORPH_OPEN, open_kernel)
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, close_kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, open_kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, close_kernel)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, open_kernel)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, close_kernel)
    return {"orange": orange, "red": red, "green": green, "white": white}


def light_face_components(masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Return compact light panels that can seed a meter-face search.

    The white mask alone is not an identity signal.  It is used here only to
    propose locally bounded face windows near a colored terminal anchor.  A
    broad workbench region is excluded before this list reaches face_search.
    """

    white = masks["white"]
    height, width = white.shape[:2]
    image_area = float(max(1, height * width))
    closed = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    components: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        box_area = float(max(1, w * h))
        relative_area = area / image_area
        density = area / box_area
        aspect = w / max(float(h), 1.0)
        if (
            w < 110
            or h < 80
            or relative_area < 0.00012
            or relative_area > 0.060
            or density < 0.34
            or aspect < 0.35
            or aspect > 2.35
        ):
            continue
        rectangularity = density
        score = clamp(
            0.56 * clamp((density - 0.34) / 0.50)
            + 0.24 * clamp((min(w, h) - 100.0) / 450.0)
            + 0.20 * clamp(1.0 - abs(aspect - 1.0) / 1.35)
        )
        components.append(
            {
                "bbox": [x, y, w, h],
                "area": area,
                "density": round(float(density), 6),
                "rectangularity": round(float(rectangularity), 6),
                "aspect_ratio": round(float(aspect), 6),
                "score": round(float(score), 6),
            }
        )
    return sorted(components, key=lambda item: float(item["score"]), reverse=True)[:80]


def integral(mask: np.ndarray) -> np.ndarray:
    return cv2.integral((mask > 0).astype(np.uint8), sdepth=cv2.CV_64F)


def rect_sum(integral_image: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    x2, y2 = x + w, y + h
    return float(
        integral_image[y2, x2]
        - integral_image[y, x2]
        - integral_image[y2, x]
        + integral_image[y, x]
    )


def ratio_from_integral(integral_image: np.ndarray, box: tuple[int, int, int, int]) -> float:
    _x, _y, w, h = box
    return rect_sum(integral_image, box) / max(float(w * h), 1.0)


def nms(items: list[dict[str, Any]], limit: int, iou_threshold: float = 0.35) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: float(value["score"]), reverse=True):
        box = tuple(int(v) for v in item["bbox"])
        if any(box_iou(box, tuple(int(v) for v in kept["bbox"])) > iou_threshold for kept in selected):
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def face_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    """IoU using the dial ROI instead of the larger meter/terminal union box."""

    return box_iou(tuple(int(v) for v in a["face"]["bbox"]), tuple(int(v) for v in b["face"]["bbox"]))


def face_containment(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Fraction of the smaller face ROI covered by the intersection."""

    ax, ay, aw, ah = (int(v) for v in a["face"]["bbox"])
    bx, by, bw, bh = (int(v) for v in b["face"]["bbox"])
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return intersection / max(float(min(aw * ah, bw * bh)), 1.0)


def faces_conflict(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two role hypotheses visibly describe the same face region."""

    if face_iou(a, b) >= 0.24 or face_containment(a, b) >= 0.52:
        return True
    ax, ay, aw, ah = (int(v) for v in a["face"]["bbox"])
    bx, by, bw, bh = (int(v) for v in b["face"]["bbox"])
    distance = math.hypot((ax + aw / 2.0) - (bx + bw / 2.0), (ay + ah / 2.0) - (by + bh / 2.0))
    reference = max(1.0, min(min(aw, ah), min(bw, bh)))
    return distance < 0.50 * reference and 0.45 <= (aw * ah) / max(float(bw * bh), 1.0) <= 2.2


def face_nms(items: list[dict[str, Any]], limit: int, iou_threshold: float = 0.55) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: float(value["score"]), reverse=True):
        ix, iy, iw, ih = (int(v) for v in item["face"]["bbox"])
        icx, icy = ix + iw / 2.0, iy + ih / 2.0
        duplicate = False
        for kept in selected:
            if face_iou(item, kept) > min(iou_threshold, 0.28) or face_containment(item, kept) >= 0.50:
                duplicate = True
                break
            kx, ky, kw, kh = (int(v) for v in kept["face"]["bbox"])
            kcx, kcy = kx + kw / 2.0, ky + kh / 2.0
            center_distance = math.hypot(icx - kcx, icy - kcy)
            reference_scale = max(1.0, min(min(iw, ih), min(kw, kh)))
            area_ratio = (iw * ih) / max(float(kw * kh), 1.0)
            if center_distance < 0.72 * reference_scale and 0.35 <= area_ratio <= 2.9:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def scan_terminal_anchors(
    masks: dict[str, np.ndarray], role: str, max_anchors: int = 8
) -> list[dict[str, Any]]:
    """Find dense red/green board-like windows with an integral-image scan."""

    primary = masks["green" if role == "ammeter" else "red"]
    opposing = masks["red" if role == "ammeter" else "green"]
    height, width = primary.shape[:2]
    primary_integral = integral(primary)
    opposing_integral = integral(opposing)
    orange_integral = integral(masks["orange"])
    # Board dimensions are expressed as fractions so the detector scales with
    # 4K and lower-resolution exports. The stride keeps the scan bounded.
    size_factors = ((0.10, 0.065), (0.13, 0.080), (0.16, 0.100), (0.20, 0.120))
    candidates: list[dict[str, Any]] = []
    threshold = 0.16 if role == "ammeter" else 0.20
    for width_factor, height_factor in size_factors:
        box_width = max(48, int(width * width_factor))
        box_height = max(36, int(height * height_factor))
        stride_x = max(36, int(box_width * 0.24))
        stride_y = max(28, int(box_height * 0.30))
        for y in range(0, max(1, height - box_height + 1), stride_y):
            for x in range(0, max(1, width - box_width + 1), stride_x):
                box = (x, y, box_width, box_height)
                primary_ratio = ratio_from_integral(primary_integral, box)
                if primary_ratio < threshold:
                    continue
                opposing_ratio = ratio_from_integral(opposing_integral, box)
                orange_ratio = ratio_from_integral(orange_integral, box)
                local = primary[y : y + box_height, x : x + box_width]
                component_fill = 0.0
                component_rectangularity = 0.0
                if local.size:
                    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(local, connectivity=8)
                    if count > 1:
                        local_stats = stats[1:]
                        largest = local_stats[int(np.argmax(local_stats[:, cv2.CC_STAT_AREA]))]
                        component_area = float(largest[cv2.CC_STAT_AREA])
                        component_box_area = float(max(1, largest[cv2.CC_STAT_WIDTH] * largest[cv2.CC_STAT_HEIGHT]))
                        component_fill = component_area / max(float(box_width * box_height), 1.0)
                        component_rectangularity = component_area / component_box_area
                # A solid board is dense in its own color; a wire is thin and
                # usually has a much lower density in a board-sized window.
                score = clamp(
                    0.78 * primary_ratio
                    + 0.14 * orange_ratio
                    + 0.06 * component_rectangularity
                    - 0.08 * min(opposing_ratio, 1.0)
                )
                candidates.append(
                    {
                        "bbox": [x, y, box_width, box_height],
                        "primary_ratio": round(float(primary_ratio), 6),
                        "opposing_ratio": round(float(opposing_ratio), 6),
                        "orange_ratio": round(float(orange_ratio), 6),
                        "component_fill": round(float(component_fill), 6),
                        "component_rectangularity": round(float(component_rectangularity), 6),
                        "score": round(float(score), 6),
                        "role_hint": role,
                    }
                )
    return nms(candidates, max_anchors, iou_threshold=0.30)


def face_metrics(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    box: tuple[int, int, int, int],
) -> dict[str, Any]:
    x, y, w, h = box
    roi = image[y : y + h, x : x + w]
    if roi.size == 0:
        return {
            "bbox": list(box),
            "white_ratio": 0.0,
            "dark_ratio": 0.0,
            "edge_density": 0.0,
            "structure_score": 0.0,
            "dial_likeness": 0.0,
        }
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 45, 150)
    white_ratio = float(np.mean(masks["white"][y : y + h, x : x + w] > 0))
    dark_ratio = float(np.mean(gray < 110))
    edge_density = float(np.mean(edges > 0))
    aspect_ratio = min(w, h) / max(w, h, 1)
    # A real dial keeps a light, structured interior even when its outer
    # border is oblique. Batteries, clothing and table regions tend to fail
    # this central-window check or contain a single large dark blob.
    ix0, iy0 = int(0.14 * w), int(0.12 * h)
    ix1, iy1 = max(ix0 + 1, int(0.86 * w)), max(iy0 + 1, int(0.86 * h))
    interior_gray = gray[iy0:iy1, ix0:ix1]
    interior_white = masks["white"][y + iy0 : y + iy1, x + ix0 : x + ix1]
    interior_edges = edges[iy0:iy1, ix0:ix1]
    interior_white_ratio = float(np.mean(interior_white > 0)) if interior_white.size else 0.0
    interior_dark_ratio = float(np.mean(interior_gray < 110)) if interior_gray.size else 0.0
    interior_edge_density = float(np.mean(interior_edges > 0)) if interior_edges.size else 0.0
    interior_hsv = hsv_roi[iy0:iy1, ix0:ix1]
    interior_neutral_ratio = (
        float(np.mean((interior_hsv[:, :, 1] < 80) & (interior_hsv[:, :, 2] > 110)))
        if interior_hsv.size
        else 0.0
    )
    upper = gray[int(0.08 * h) : int(0.55 * h), int(0.10 * w) : int(0.90 * w)]
    upper_dark_ratio = float(np.mean(upper < 105)) if upper.size else 0.0
    white_score = clamp((white_ratio - 0.25) / 0.60)
    dark_score = clamp((dark_ratio - 0.035) / 0.20)
    edge_score = clamp((edge_density - 0.020) / 0.090)
    interior_white_score = clamp((interior_white_ratio - 0.34) / 0.46)
    interior_edge_score = clamp((interior_edge_density - 0.018) / 0.12)
    aspect_score = clamp((aspect_ratio - 0.52) / 0.43)
    scale_band_score = clamp((upper_dark_ratio - 0.035) / 0.20)
    # White workbench regions have high white_ratio but low edge/dark density.
    structure_score = 0.40 * white_score + 0.38 * edge_score + 0.22 * dark_score
    if white_ratio > 0.90 and edge_density < 0.030:
        structure_score *= 0.35
    dial_likeness = clamp(
        0.30 * interior_white_score
        + 0.20 * interior_edge_score
        + 0.16 * scale_band_score
        + 0.15 * aspect_score
        + 0.10 * clamp((interior_dark_ratio - 0.045) / 0.26)
        + 0.09 * clamp((interior_neutral_ratio - 0.35) / 0.45)
    )
    return {
        "bbox": list(box),
        "bbox_xyxy": [x, y, x + w, y + h],
        "white_ratio": round(white_ratio, 6),
        "dark_ratio": round(dark_ratio, 6),
        "edge_density": round(edge_density, 6),
        "aspect_ratio": round(float(aspect_ratio), 6),
        "interior_white_ratio": round(interior_white_ratio, 6),
        "interior_dark_ratio": round(interior_dark_ratio, 6),
        "interior_edge_density": round(interior_edge_density, 6),
        "interior_neutral_ratio": round(interior_neutral_ratio, 6),
        "upper_dark_ratio": round(upper_dark_ratio, 6),
        "dial_likeness": round(float(dial_likeness), 6),
        "structure_score": round(float(clamp(structure_score)), 6),
    }


def component_anchor_proximity(component_box: tuple[int, int, int, int], anchor_box: tuple[int, int, int, int]) -> float:
    """Score spatial proximity without assuming a perfectly vertical camera."""

    cx, cy, cw, ch = component_box
    ax, ay, aw, ah = anchor_box
    distance = math.hypot((cx + cw / 2.0) - (ax + aw / 2.0), (cy + ch / 2.0) - (ay + ah / 2.0))
    reference = max(1.0, 0.58 * math.hypot(cw, ch) + 0.42 * math.hypot(aw, ah))
    return clamp(1.0 - distance / (1.85 * reference))


def face_search(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    anchor: dict[str, Any],
    face_components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Search light face panels near an anchor, with a grid-search fallback."""

    ax, ay, aw, ah = (int(v) for v in anchor["bbox"])
    height, width = image.shape[:2]
    anchor_box = (ax, ay, aw, ah)
    candidates: list[dict[str, Any]] = []
    anchor_center_x = ax + aw / 2.0
    anchor_center_y = ay + ah / 2.0

    def add_candidate(
        face_box: tuple[int, int, int, int],
        placement: str,
        source: str,
        component: dict[str, Any] | None,
        proximity: float,
        clipped_top: bool = False,
    ) -> None:
        metrics = face_metrics(image, masks, face_box)
        if float(metrics["structure_score"]) < 0.18:
            return
        component_score = float(component["score"]) if component is not None else 0.0
        metrics["light_component_support"] = round(component_score, 6)
        metrics["terminal_proximity"] = round(proximity, 6)
        score = float(metrics["structure_score"])
        if component is not None:
            score = clamp(0.68 * score + 0.20 * component_score + 0.12 * proximity)
        candidates.append(
            {
                "bbox": list(face_box),
                "metrics": metrics,
                "score": score,
                "anchor_bbox": [ax, ay, aw, ah],
                "placement": placement,
                "face_source": source,
                "light_face_component": component,
                "clipped_top": clipped_top,
            }
        )

    # Prefer bounded neutral/light panels. This prevents a coarse search box
    # from covering both meters or drifting onto a white tabletop.
    for component in face_components:
        component_box = tuple(int(value) for value in component["bbox"])
        proximity = component_anchor_proximity(component_box, anchor_box)
        if proximity < 0.26:
            continue
        cx, cy, cw, ch = component_box
        pad_x, pad_y = max(6, int(0.035 * cw)), max(6, int(0.040 * ch))
        left, top = max(0, cx - pad_x), max(0, cy - pad_y)
        right, bottom = min(width, cx + cw + pad_x), min(height, cy + ch + pad_y)
        if right - left < 100 or bottom - top < 100:
            continue
        relation_x = (cx + cw / 2.0 - anchor_center_x) / max(float(cw), 1.0)
        relation_y = (cy + ch / 2.0 - anchor_center_y) / max(float(ch), 1.0)
        placement = "component_near_anchor"
        if relation_y < -0.25:
            placement = "component_above_anchor"
        elif relation_y > 0.25:
            placement = "component_below_anchor"
        elif relation_x < -0.25:
            placement = "component_left_anchor"
        elif relation_x > 0.25:
            placement = "component_right_anchor"
        add_candidate((left, top, right - left, bottom - top), placement, "light_face_component", component, proximity)

    # The ordinary layout is ``dial above terminal``. Wide camera angles can
    # put the dial above-left, above-right, or beside the terminal. Retain the
    # grid path as a lower-ranked fallback for blurred or partly occluded faces.
    placements = (
        ("above", 0.0, -0.55),
        ("above_left", -0.45, -0.38),
        ("above_right", 0.45, -0.38),
        ("left", -0.58, 0.0),
        ("right", 0.58, 0.0),
        ("below_left", -0.40, 0.38),
        ("below_right", 0.40, 0.38),
    )
    for width_scale in (0.78, 0.90, 1.05, 1.20, 1.40):
        for height_scale in (0.78, 0.92, 1.08, 1.24, 1.36):
            face_width = max(80, int(aw * width_scale))
            face_height = max(80, int(aw * height_scale))
            for offset in (-0.14, 0.0, 0.14):
                for placement, px, py in placements:
                    center_x = anchor_center_x + px * face_width + offset * aw
                    center_y = anchor_center_y + py * face_height
                    left = int(round(center_x - face_width / 2.0))
                    top = int(round(center_y - face_height / 2.0))
                    right = left + face_width
                    bottom = top + face_height
                    clip_left, clip_top = max(0, left), max(0, top)
                    clip_right, clip_bottom = min(width, right), min(height, bottom)
                    if clip_right - clip_left < 100 or clip_bottom - clip_top < 100:
                        continue
                    face_box = (clip_left, clip_top, clip_right - clip_left, clip_bottom - clip_top)
                    proximity = component_anchor_proximity(face_box, anchor_box)
                    add_candidate(face_box, placement, "anchor_grid_fallback", None, proximity, clipped_top=bool(top < 0))
    return nms(candidates, 5, iou_threshold=0.42)


def find_pointer_axis(gray: np.ndarray) -> dict[str, Any]:
    """Find a possible pivot hub; absence is preserved as insufficient evidence."""

    height, width = gray.shape[:2]
    minimum = min(width, height)
    if minimum < 70:
        return {"detected": False, "reason": "face_roi_too_small"}
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(18, int(0.16 * minimum)),
        param1=90,
        param2=13,
        minRadius=max(3, int(0.010 * minimum)),
        maxRadius=max(8, int(0.095 * minimum)),
    )
    if circles is None:
        return {"detected": False, "reason": "no_hough_hub"}
    best: tuple[float, float, float, float] | None = None
    for cx, cy, radius in circles[0]:
        if not (0.22 * width <= cx <= 0.78 * width and 0.43 * height <= cy <= 0.90 * height):
            continue
        horizontal = abs(cx / max(width, 1) - 0.50)
        vertical = abs(cy / max(height, 1) - 0.66)
        radius_score = clamp(1.0 - abs(radius / max(minimum, 1) - 0.042) / 0.060)
        score = clamp(0.42 * (1.0 - horizontal / 0.28) + 0.38 * (1.0 - vertical / 0.27) + 0.20 * radius_score)
        if best is None or score > best[0]:
            best = (score, float(cx), float(cy), float(radius))
    if best is None:
        return {"detected": False, "reason": "hough_hubs_outside_expected_face_region"}
    score, cx, cy, radius = best
    return {
        "detected": True,
        "center_xy": [round(cx, 2), round(cy, 2)],
        "radius": round(radius, 2),
        "confidence": round(score, 6),
        "reason": "hough_circle_geometry_only",
    }


def detect_pointer(image: np.ndarray, face_box: tuple[int, int, int, int]) -> dict[str, Any]:
    x, y, w, h = face_box
    roi = image[y : y + h, x : x + w]
    if roi.size == 0:
        return {"detected": False, "state": "uncertain", "reason": "empty_face_roi"}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(enhanced, 35, 130)
    axis = find_pointer_axis(enhanced)
    center = tuple(axis["center_xy"]) if axis.get("detected") else (w / 2.0, h * 0.62)
    radius = min(w, h) * 0.44
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    profile = np.zeros(len(angles), dtype=np.float32)
    distances = np.linspace(radius * 0.13, radius * 0.72, 55)
    offsets = np.linspace(-1.5, 1.5, 3)
    for index, angle in enumerate(angles):
        xs = center[0] + np.cos(angle) * distances[:, None] - np.sin(angle) * offsets[None, :]
        ys = center[1] - np.sin(angle) * distances[:, None] - np.cos(angle) * offsets[None, :]
        xi = np.rint(xs).astype(np.int32)
        yi = np.rint(ys).astype(np.int32)
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if np.any(valid):
            values = enhanced[yi[valid], xi[valid]].astype(np.float32)
            profile[index] = float(0.78 * np.mean(values < 120) + 0.22 * np.mean(edges[yi[valid], xi[valid]] > 0))
    smoothed = np.convolve(np.concatenate([profile[-3:], profile, profile[:3]]), np.ones(7, dtype=np.float32) / 7.0, mode="valid")
    order = np.argsort(smoothed)[::-1]
    best_index = int(order[0]) if len(order) else 0
    best = float(smoothed[best_index]) if len(smoothed) else 0.0
    second = float(smoothed[order[1]]) if len(order) > 1 else 0.0
    confidence = clamp(0.58 * best + 0.42 * max(0.0, best - second) * 4.0)
    angle_deg = math.degrees(float(angles[best_index])) if len(angles) else None
    endpoint = None
    if angle_deg is not None:
        endpoint = [
            round(center[0] + math.cos(math.radians(angle_deg)) * radius * 0.72, 2),
            round(center[1] - math.sin(math.radians(angle_deg)) * radius * 0.72, 2),
        ]
    return {
        "detected": bool(best > 0.13),
        "angle_deg": None if angle_deg is None else round(float(angle_deg), 4),
        "confidence": round(float(confidence), 6),
        "center_xy": [round(center[0], 2), round(center[1], 2)],
        "radius": round(float(radius), 2),
        "endpoint_xy": endpoint,
        "state": "uncertain",
        "reason": "zero_full_scale_calibration_and_temporal_stability_not_supplied",
        "axis_candidate": axis,
        "profile_peak": round(best, 6),
    }


def color_components(mask: np.ndarray, min_area: int = 3500) -> list[dict[str, Any]]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rows: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area < min_area:
            continue
        ratio = w / max(h, 1)
        if ratio < 0.35 or ratio > 5.5:
            continue
        rows.append(
            {
                "bbox": [x, y, w, h],
                "area": area,
                "density": round(float(area / max(w * h, 1)), 6),
            }
        )
    return rows


def nearby_terminal_features(
    masks: dict[str, np.ndarray],
    face_box: tuple[int, int, int, int],
    components: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Search color windows around a dial without assuming a vertical layout."""

    x, y, w, h = face_box
    height, width = masks["green"].shape[:2]
    placements = (
        ("below", 0.0, 0.86, 0.78, 0.25),
        ("below_left", -0.38, 0.72, 0.58, 0.25),
        ("below_right", 0.38, 0.72, 0.58, 0.25),
        ("left", -0.82, 0.0, 0.32, 0.58),
        ("right", 0.82, 0.0, 0.32, 0.58),
        ("above", 0.0, -0.84, 0.78, 0.22),
    )
    best = {
        "green": {"ratio": 0.0, "bbox": None, "placement": None, "component_confidence": 0.0, "component": None},
        "red": {"ratio": 0.0, "bbox": None, "placement": None, "component_confidence": 0.0, "component": None},
    }
    for placement, dx, dy, width_scale, height_scale in placements:
        box_width = max(40, int(w * width_scale))
        box_height = max(30, int(h * height_scale))
        center_x = x + w / 2.0 + dx * w
        center_y = y + h / 2.0 + dy * h
        left = max(0, min(width - box_width, int(round(center_x - box_width / 2.0))))
        top = max(0, min(height - box_height, int(round(center_y - box_height / 2.0))))
        box = (left, top, box_width, box_height)
        for color in ("green", "red"):
            ratio = float(np.mean(masks[color][top : top + box_height, left : left + box_width] > 0))
            if ratio > float(best[color]["ratio"]):
                best[color] = {
                    "ratio": round(ratio, 6),
                    "bbox": list(box),
                    "placement": placement,
                    "component_confidence": 0.0,
                    "component": None,
                }
    if components is None:
        components = {"green": color_components(masks["green"]), "red": color_components(masks["red"])}
    face_cx, face_cy = x + w / 2.0, y + h / 2.0
    for color in ("green", "red"):
        for component in components[color]:
            bx, by, bw, bh = (int(v) for v in component["bbox"])
            component_cx, component_cy = bx + bw / 2.0, by + bh / 2.0
            distance = math.hypot(component_cx - face_cx, component_cy - face_cy) / max(w, h, 1)
            if distance > 1.8:
                continue
            proximity = clamp(1.0 - distance / 1.8)
            density_score = clamp(float(component["density"]) / 0.40)
            area_score = clamp(float(component["area"]) / 30000.0)
            confidence = 0.45 * density_score + 0.35 * area_score + 0.20 * proximity
            if confidence > float(best[color]["component_confidence"]):
                best[color]["component_confidence"] = round(float(confidence), 6)
                best[color]["component"] = component
                component_cx, component_cy = bx + bw / 2.0, by + bh / 2.0
                dx = component_cx - face_cx
                dy = component_cy - face_cy
                best[color]["component_relative_valid"] = bool(dy >= -0.15 * h or abs(dx) >= 0.65 * w)
    for color in ("green", "red"):
        best[color].setdefault("component_relative_valid", False)
    return best


def hough_dial_candidates(image: np.ndarray, masks: dict[str, np.ndarray], limit: int = 12) -> list[dict[str, Any]]:
    """Find dial-like faces first, then classify them by nearby terminal color."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 0.50
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)
    circles = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=70,
        param1=100,
        param2=30,
        minRadius=65,
        maxRadius=420,
    )
    if circles is None:
        return []
    height, width = image.shape[:2]
    components = {"green": color_components(masks["green"]), "red": color_components(masks["red"])}
    candidates: list[dict[str, Any]] = []
    for cx_small, cy_small, radius_small in circles[0][:80]:
        cx, cy, radius = float(cx_small / scale), float(cy_small / scale), float(radius_small / scale)
        face_width = max(100, int(1.70 * radius))
        face_height = max(100, int(1.50 * radius))
        left = max(0, min(width - face_width, int(round(cx - 0.85 * radius))))
        top = max(0, min(height - face_height, int(round(cy - 0.75 * radius))))
        face_box = (left, top, min(face_width, width - left), min(face_height, height - top))
        metrics = face_metrics(image, masks, face_box)
        if (
            float(metrics["structure_score"]) < 0.54
            or float(metrics["white_ratio"]) < 0.55
            or float(metrics["dark_ratio"]) < 0.07
            or float(metrics["edge_density"]) < 0.045
            or float(metrics.get("interior_white_ratio", 0.0)) < 0.52
            or float(metrics.get("interior_neutral_ratio", 0.0)) < 0.54
            or float(metrics.get("dial_likeness", 0.0)) < 0.42
            or face_box[2] * face_box[3] > image.shape[0] * image.shape[1] * 0.06
        ):
            continue
        nearby = nearby_terminal_features(masks, face_box, components)
        green_ratio = float(nearby["green"]["ratio"])
        red_ratio = float(nearby["red"]["ratio"])
        green_component_confidence = float(nearby["green"]["component_confidence"])
        if (
            green_component_confidence >= 0.55
            and bool(nearby["green"].get("component_relative_valid"))
            and green_ratio >= 0.10
        ):
            role, role_basis, primary_ratio = "ammeter", "nearby_green_terminal_board", green_ratio
            terminal = nearby["green"]
            threshold = 0.11
        elif red_ratio >= 0.16:
            role, role_basis, primary_ratio = "voltmeter", "nearby_red_terminal_board", red_ratio
            terminal = nearby["red"]
            threshold = 0.16
        else:
            continue
        x, y, w, h = face_box
        housing_box = (
            max(0, int(x - 0.12 * w)),
            max(0, int(y - 0.08 * h)),
            min(width - max(0, int(x - 0.12 * w)), int(1.24 * w)),
            min(height - max(0, int(y - 0.08 * h)), int(1.16 * h)),
        )
        hx, hy, hw, hh = housing_box
        housing_ratio = float(np.mean(masks["orange"][hy : hy + hh, hx : hx + hw] > 0))
        role_score = clamp((primary_ratio - threshold) / (0.50 - threshold))
        score = clamp(
            0.46 * float(metrics["structure_score"])
            + 0.18 * float(metrics.get("dial_likeness", 0.0))
            + 0.26 * role_score
            + 0.16 * clamp(housing_ratio / 0.18)
        )
        candidates.append(
            {
                "bbox": list(union_box(face_box, tuple(int(v) for v in terminal["bbox"])) if terminal["bbox"] else face_box),
                "face": metrics,
                "terminal_anchor": terminal,
                "role": role,
                "role_basis": role_basis,
                "terminal_score": round(float(role_score), 6),
                "housing_ratio": round(float(housing_ratio), 6),
                "shape_score": None,
                "score": round(float(score), 6),
                "detector_source": "hough_dial_fallback",
                "circle": {"center_xy": [round(cx, 2), round(cy, 2)], "radius": round(radius, 2)},
                "candidate_reason": [
                    "hough_circle_support",
                    "light_structured_dial_interior",
                    "nearby_terminal_component",
                    "orange_housing_support",
                ],
                "rejection_reason": None,
                "evidence_insufficient_reason": ["hough_fallback_requires_temporal_confirmation"],
            }
        )
    return nms(candidates, limit, iou_threshold=0.45)


def build_role_candidate(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    anchor: dict[str, Any],
    role: str,
    face_components: list[dict[str, Any]],
    rejections: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    face_candidates = face_search(image, masks, anchor, face_components)
    output: list[dict[str, Any]] = []
    ax, ay, aw, ah = (int(v) for v in anchor["bbox"])
    primary_ratio = float(anchor["primary_ratio"])
    threshold = 0.16 if role == "ammeter" else 0.20
    terminal_score = clamp((primary_ratio - threshold) / (0.50 - threshold))
    for face_item in face_candidates:
        face_box = tuple(int(v) for v in face_item["bbox"])
        face_info = face_item["metrics"]
        # A board-sized crop can contain batteries or a hand and still have
        # some edges. Require all three dial cues before it becomes a meter
        # candidate; missing cues are evidence insufficiency, not a fail.
        rejection_reason = None
        if float(face_info["structure_score"]) < 0.60:
            rejection_reason = "face_structure_below_threshold"
        elif float(face_info["white_ratio"]) < 0.42:
            rejection_reason = "face_white_ratio_below_threshold"
        elif float(face_info["dark_ratio"]) < 0.07:
            rejection_reason = "face_dark_ratio_below_threshold"
        elif float(face_info["edge_density"]) < 0.045:
            rejection_reason = "face_edge_density_below_threshold"
        elif float(face_info.get("interior_white_ratio", 0.0)) < 0.54:
            rejection_reason = "dial_interior_not_light_enough"
        elif float(face_info.get("interior_neutral_ratio", 0.0)) < 0.54:
            rejection_reason = "dial_interior_color_not_neutral"
        elif float(face_info.get("dial_likeness", 0.0)) < 0.46:
            rejection_reason = "dial_geometry_insufficient"
        if rejection_reason:
            if rejections is not None:
                rejections.append({"role": role, "face_bbox": list(face_box), "rejection_reason": rejection_reason})
            continue
        meter_box = union_box(face_box, (ax, ay, aw, ah))
        x, y, w, h = meter_box
        housing_ratio = float(np.mean(masks["orange"][y : y + h, x : x + w] > 0))
        if housing_ratio < 0.04:
            if rejections is not None:
                rejections.append({"role": role, "face_bbox": list(face_box), "rejection_reason": "orange_housing_support_insufficient"})
            continue
        box_shape = clamp(1.0 - abs((w / max(h, 1)) - 0.90) / 1.50)
        score = clamp(
            0.48 * float(face_item["score"])
            + 0.18 * float(face_info.get("dial_likeness", 0.0))
            + 0.24 * terminal_score
            + 0.10 * box_shape
            + 0.04 * clamp(housing_ratio / 0.20)
        )
        candidate_reason = [
            "terminal_color_anchor",
            "light_structured_dial_interior",
            "dark_scale_or_pointer_edges",
            "orange_housing_support",
        ]
        if face_item.get("face_source") == "light_face_component":
            candidate_reason.append("bounded_light_face_component")
        else:
            candidate_reason.append("anchor_grid_face_fallback")
        output.append(
            {
                "bbox": list(meter_box),
                "face": face_item["metrics"],
                "terminal_anchor": anchor,
                "role": role,
                "role_basis": "green_terminal_board" if role == "ammeter" else "red_terminal_board",
                "detector_source": "terminal_anchor_search",
                "terminal_score": round(float(terminal_score), 6),
                "housing_ratio": round(float(housing_ratio), 6),
                "shape_score": round(float(box_shape), 6),
                "score": round(float(score), 6),
                "face_source": face_item.get("face_source"),
                "light_face_component": face_item.get("light_face_component"),
                "candidate_reason": candidate_reason,
                "rejection_reason": None,
                "evidence_insufficient_reason": [],
            }
        )
    return face_nms(output, 5, iou_threshold=0.55)


def draw_debug(image: np.ndarray, candidates: list[dict[str, Any]], pair: dict[str, Any]) -> np.ndarray:
    canvas = image.copy()
    for index, item in enumerate(candidates, start=1):
        x, y, w, h = (int(v) for v in item["bbox"])
        fx, fy, fw, fh = (int(v) for v in item["face"]["bbox"])
        role = str(item["role"])
        color = (0, 180, 0) if role == "ammeter" else (0, 0, 255)
        thickness = 8 if item is pair.get(role) else 3
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (255, 80, 0), 3)
        terminal = item.get("terminal_anchor") or {}
        terminal_box = terminal.get("bbox")
        if terminal_box:
            tx, ty, tw, th = (int(v) for v in terminal_box)
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (255, 190, 0), 2)
        pointer = item.get("pointer")
        if pointer and pointer.get("endpoint_xy"):
            cx, cy = pointer["center_xy"]
            ex, ey = pointer["endpoint_xy"]
            cv2.line(canvas, (fx + int(cx), fy + int(cy)), (fx + int(ex), fy + int(ey)), (0, 255, 0), 4)
            cv2.circle(canvas, (fx + int(cx), fy + int(cy)), 8, (0, 0, 255), -1)
        label = (
            f"{index} {role} S={float(item['score']):.2f} "
            f"D={float(item['face'].get('dial_likeness', 0.0)):.2f} "
            f"src={str(item.get('detector_source', 'unknown'))[:7]}"
        )
        cv2.putText(canvas, label, (x, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return canvas


def save_roi_artifacts(
    image: np.ndarray,
    candidates: list[dict[str, Any]],
    roi_dir: Path | None,
    stem: str,
) -> None:
    if roi_dir is None:
        return
    face_dir = roi_dir / "face"
    terminal_dir = roi_dir / "terminal"
    face_dir.mkdir(parents=True, exist_ok=True)
    terminal_dir.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]

    def crop(box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        return image[y0:y1, x0:x1]

    for index, item in enumerate(candidates, start=1):
        face_box = tuple(int(v) for v in item["face"]["bbox"])
        face_path = face_dir / f"{stem}_candidate_{index:02d}_face.jpg"
        face_roi = crop(face_box)
        if face_roi.size:
            cv2.imwrite(str(face_path), face_roi, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
            item["face_roi_path"] = str(face_path.resolve())
        terminal = item.get("terminal_anchor") or {}
        terminal_box = terminal.get("bbox")
        if terminal_box:
            terminal_path = terminal_dir / f"{stem}_candidate_{index:02d}_terminal.jpg"
            terminal_roi = crop(tuple(int(v) for v in terminal_box))
            if terminal_roi.size:
                cv2.imwrite(str(terminal_path), terminal_roi, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
                item["terminal_roi_path"] = str(terminal_path.resolve())


def analyze_image(
    path: Path,
    debug_path: Path | None = None,
    roi_dir: Path | None = None,
) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"path": str(path.resolve()), "valid": False, "errors": ["image_decode_failed"]}
    mask_set = make_masks(image)
    face_components = light_face_components(mask_set)
    ammeter_anchors = scan_terminal_anchors(mask_set, "ammeter")
    voltmeter_anchors = scan_terminal_anchors(mask_set, "voltmeter")
    candidates: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    for anchor in ammeter_anchors:
        candidates.extend(build_role_candidate(image, mask_set, anchor, "ammeter", face_components, rejected_candidates))
    for anchor in voltmeter_anchors:
        candidates.extend(build_role_candidate(image, mask_set, anchor, "voltmeter", face_components, rejected_candidates))
    best_by_role = {
        role: max((item for item in candidates if item.get("role") == role), key=lambda value: float(value["score"]), default=None)
        for role in ("ammeter", "voltmeter")
    }
    fallback_needed = any(
        best_by_role[role] is None
        or float(best_by_role[role]["score"]) < 0.70
        or float(best_by_role[role]["face"].get("dial_likeness", 0.0)) < 0.50
        for role in ("ammeter", "voltmeter")
    )
    if fallback_needed:
        candidates.extend(hough_dial_candidates(image, mask_set, limit=12))
    # Different roles may legitimately overlap while the image is oblique;
    # perform NMS within a role so a red false positive cannot erase the only
    # green ammeter hypothesis.
    role_pools = {
        "ammeter": face_nms([item for item in candidates if item.get("role") == "ammeter"], 6, iou_threshold=0.55),
        "voltmeter": face_nms([item for item in candidates if item.get("role") == "voltmeter"], 6, iou_threshold=0.55),
    }
    candidates = role_pools["ammeter"] + role_pools["voltmeter"]
    for item in candidates:
        item["pointer"] = detect_pointer(image, tuple(int(v) for v in item["face"]["bbox"]))
    by_role: dict[str, list[dict[str, Any]]] = {"ammeter": [], "voltmeter": []}
    for item in candidates:
        by_role.setdefault(str(item["role"]), []).append(item)
    for role in by_role:
        by_role[role].sort(key=lambda value: float(value["score"]), reverse=True)
        by_role[role] = by_role[role][:3]
        for item in by_role[role]:
            if item.get("detector_source") == "hough_dial_fallback":
                item.setdefault("evidence_insufficient_reason", []).append("fallback_identity_needs_temporal_confirmation")
    candidates = by_role["ammeter"] + by_role["voltmeter"]
    for index, item in enumerate(candidates, start=1):
        item["candidate_id"] = f"candidate_{index:02d}"

    role_conflicts: list[dict[str, Any]] = []
    for ammeter in by_role["ammeter"]:
        for voltmeter in by_role["voltmeter"]:
            if not faces_conflict(ammeter, voltmeter):
                continue
            reason = "cross_role_face_overlap_identity_conflict"
            for item in (ammeter, voltmeter):
                reasons = item.setdefault("evidence_insufficient_reason", [])
                if reason not in reasons:
                    reasons.append(reason)
            role_conflicts.append(
                {
                    "ammeter_candidate_id": ammeter["candidate_id"],
                    "voltmeter_candidate_id": voltmeter["candidate_id"],
                    "face_iou": round(face_iou(ammeter, voltmeter), 6),
                    "face_containment": round(face_containment(ammeter, voltmeter), 6),
                    "rejection_reason": reason,
                }
            )
    pair: dict[str, Any] = {
        "status": "incomplete",
        "ammeter": None,
        "voltmeter": None,
        "same_source_image": True,
        "evidence_insufficient_reason": "same-frame double-role evidence did not clear conservative gates",
    }
    pair_options: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for ammeter in by_role["ammeter"]:
        for voltmeter in by_role["voltmeter"]:
            if faces_conflict(ammeter, voltmeter):
                continue
            fallback_pair = (
                ammeter.get("detector_source") == "hough_dial_fallback"
                or voltmeter.get("detector_source") == "hough_dial_fallback"
            )
            pair_threshold = 0.78 if fallback_pair else 0.70
            strong_fallback = (
                not fallback_pair
                or (
                    float(ammeter["face"]["structure_score"]) >= 0.70
                    and float(voltmeter["face"]["structure_score"]) >= 0.70
                    and float(ammeter["face"].get("dial_likeness", 0.0)) >= 0.50
                    and float(voltmeter["face"].get("dial_likeness", 0.0)) >= 0.50
                )
            )
            if (
                strong_fallback
                and float(ammeter["score"]) >= pair_threshold
                and float(voltmeter["score"]) >= pair_threshold
                and float(ammeter["face"].get("dial_likeness", 0.0)) >= 0.50
                and float(voltmeter["face"].get("dial_likeness", 0.0)) >= 0.50
            ):
                pair_options.append((float(ammeter["score"]) + float(voltmeter["score"]), ammeter, voltmeter))
    if pair_options:
        _score, ammeter, voltmeter = max(pair_options, key=lambda item: item[0])
        pair.update({"status": "paired", "ammeter": ammeter, "voltmeter": voltmeter, "evidence_insufficient_reason": None})
    elif by_role["ammeter"] or by_role["voltmeter"]:
        for item in candidates:
            reasons = item.setdefault("evidence_insufficient_reason", [])
            if "pair_gate_not_cleared" not in reasons:
                reasons.append("pair_gate_not_cleared")
    save_roi_artifacts(image, candidates, roi_dir, path.stem)
    if debug_path:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), draw_debug(image, candidates, pair), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "path": str(path.resolve()),
        "valid": True,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "detector": "terminal_anchor_face_search_v4",
        "terminal_anchor_counts": {"ammeter": len(ammeter_anchors), "voltmeter": len(voltmeter_anchors)},
        "light_face_component_count": len(face_components),
        "light_face_components": face_components,
        "candidates": candidates,
        "role_candidates": by_role,
        "cross_role_conflicts": role_conflicts,
        "rejected_candidates": rejected_candidates[:40],
        "selected_pair": pair,
        "qwen_called": False,
        "excel_accessed": False,
        "source_video_accessed": False,
        "score_computed": False,
        "warnings": [
            "Terminal color is an identity cue, not proof of wiring or range.",
            "Pointer state remains uncertain until zero/full-scale calibration and temporal stability are available.",
            "A paired result means both candidates came from this image; it does not prove a measurement phase.",
        ],
    }


def source_frame_number(path: str) -> int | None:
    match = re.search(r"frame_(\d+)", Path(path).name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def candidate_observation(measurement: dict[str, Any], image_index: int, candidate: dict[str, Any]) -> dict[str, Any]:
    x, y, w, h = (float(value) for value in candidate["face"]["bbox"])
    image_width = max(float(measurement.get("image_width", 1)), 1.0)
    image_height = max(float(measurement.get("image_height", 1)), 1.0)
    return {
        "candidate_id": candidate.get("candidate_id"),
        "source_image_path": measurement.get("path"),
        "image_index": image_index,
        "frame_number": source_frame_number(str(measurement.get("path", ""))),
        "face_bbox": candidate["face"]["bbox"],
        "normalized_center": [round((x + w / 2.0) / image_width, 6), round((y + h / 2.0) / image_height, 6)],
        "normalized_size": [round(w / image_width, 6), round(h / image_height, 6)],
        "detector_source": candidate.get("detector_source"),
        "score": candidate.get("score"),
        "dial_likeness": candidate.get("face", {}).get("dial_likeness"),
    }


def build_role_tracks(measurements: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    """Track candidate positions across supplied still frames from one event."""

    tracks: list[dict[str, Any]] = []
    for image_index, measurement in enumerate(measurements):
        if not measurement.get("valid"):
            continue
        candidates = measurement.get("role_candidates", {}).get(role, [])
        if not isinstance(candidates, list):
            continue
        for candidate in sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True):
            observation = candidate_observation(measurement, image_index, candidate)
            ox, oy = observation["normalized_center"]
            ow, oh = observation["normalized_size"]
            compatible: list[tuple[float, dict[str, Any]]] = []
            for track in tracks:
                if any(int(item["image_index"]) == image_index for item in track["observations"]):
                    continue
                reference = track["observations"][-1]
                rx, ry = reference["normalized_center"]
                rw, rh = reference["normalized_size"]
                distance = math.hypot(ox - rx, oy - ry)
                scale_ratio = (ow * oh) / max(float(rw * rh), 1e-6)
                if distance <= 0.135 and 0.48 <= scale_ratio <= 2.10:
                    compatible.append((distance, track))
            if compatible:
                _distance, selected = min(compatible, key=lambda item: item[0])
                selected["observations"].append(observation)
            else:
                tracks.append({"role": role, "observations": [observation]})
    output: list[dict[str, Any]] = []
    for index, track in enumerate(tracks, start=1):
        observations = track["observations"]
        xs = [float(item["normalized_center"][0]) for item in observations]
        ys = [float(item["normalized_center"][1]) for item in observations]
        mean_x, mean_y = float(np.mean(xs)), float(np.mean(ys))
        jitter = max((math.hypot(x - mean_x, y - mean_y) for x, y in zip(xs, ys)), default=0.0)
        frame_numbers = [item["frame_number"] for item in observations if item["frame_number"] is not None]
        scores = [float(item.get("score") or 0.0) for item in observations]
        output.append(
            {
                "track_id": f"{role}_track_{index:02d}",
                "role": role,
                "observation_count": len(observations),
                "source_image_count": len({int(item["image_index"]) for item in observations}),
                "frame_numbers": frame_numbers,
                "normalized_center_mean": [round(mean_x, 6), round(mean_y, 6)],
                "normalized_center_jitter": round(float(jitter), 6),
                "mean_candidate_score": round(float(np.mean(scores)), 6) if scores else 0.0,
                "stability_state": "stable_candidate_track"
                if len(observations) >= 2 and jitter <= 0.095
                else "insufficient_temporal_support",
                "observations": observations,
            }
        )
    return sorted(
        output,
        key=lambda item: (int(item["source_image_count"]), float(item["mean_candidate_score"])),
        reverse=True,
    )


def associate_event_candidates(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize same-event evidence without asserting a measurement phase."""

    result: dict[str, Any] = {
        "status": "evidence_insufficient",
        "association_scope": "supplied_event_still_frames_only",
        "same_source_image_required": False,
        "supports_rubric_scoring": False,
        "ammeter_track": None,
        "voltmeter_track": None,
        "frame_gap": None,
        "switch_closure_phase_candidate": {
            "state": "unknown",
            "reason": "switch_detection_is_not_implemented_in_the_candidate_layer",
        },
        "evidence_insufficient_reason": "at_least_two_event_frames_per_role_are_required_for_temporal_association",
    }
    if len(measurements) < 2:
        return result
    role_tracks = {
        "ammeter": build_role_tracks(measurements, "ammeter"),
        "voltmeter": build_role_tracks(measurements, "voltmeter"),
    }
    result["role_tracks"] = role_tracks
    eligible_ammeter = [
        item
        for item in role_tracks["ammeter"]
        if item["source_image_count"] >= 2 and item["stability_state"] == "stable_candidate_track"
    ]
    eligible_voltmeter = [
        item
        for item in role_tracks["voltmeter"]
        if item["source_image_count"] >= 2 and item["stability_state"] == "stable_candidate_track"
    ]
    if not eligible_ammeter or not eligible_voltmeter:
        result["evidence_insufficient_reason"] = "one_or_both_roles_lack_a_stable_two_frame_candidate_track"
        return result
    # A Hough circle can remain stable on a battery, a desk feature, or a
    # broad two-meter crop. It is useful as a candidate source but cannot by
    # itself establish a cross-frame role track.
    direct_ammeter = [
        item
        for item in eligible_ammeter
        if all(observation.get("detector_source") == "terminal_anchor_search" for observation in item["observations"])
    ]
    direct_voltmeter = [
        item
        for item in eligible_voltmeter
        if all(observation.get("detector_source") == "terminal_anchor_search" for observation in item["observations"])
    ]
    if not direct_ammeter or not direct_voltmeter:
        result["evidence_insufficient_reason"] = "hough_fallback_track_requires_independent_terminal_anchor_confirmation"
        return result
    best: tuple[int, float, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    for ammeter_track in direct_ammeter:
        for voltmeter_track in direct_voltmeter:
            for ammeter_observation in ammeter_track["observations"]:
                for voltmeter_observation in voltmeter_track["observations"]:
                    a_frame, v_frame = ammeter_observation["frame_number"], voltmeter_observation["frame_number"]
                    if a_frame is not None and v_frame is not None:
                        frame_gap = abs(int(a_frame) - int(v_frame))
                        within_window = frame_gap <= 90
                    else:
                        frame_gap = abs(int(ammeter_observation["image_index"]) - int(voltmeter_observation["image_index"]))
                        within_window = frame_gap <= 1
                    if not within_window:
                        continue
                    acx, acy = ammeter_observation["normalized_center"]
                    vcx, vcy = voltmeter_observation["normalized_center"]
                    if math.hypot(acx - vcx, acy - vcy) < 0.075:
                        continue
                    quality = float(ammeter_track["mean_candidate_score"]) + float(voltmeter_track["mean_candidate_score"])
                    candidate = (frame_gap, -quality, ammeter_track, voltmeter_track, ammeter_observation, voltmeter_observation)
                    if best is None or candidate[:2] < best[:2]:
                        best = candidate
    if best is None:
        result["evidence_insufficient_reason"] = "stable_role_tracks_do_not_have_a_nonoverlapping_temporally_near_observation_pair"
        return result
    frame_gap, _neg_quality, ammeter_track, voltmeter_track, ammeter_observation, voltmeter_observation = best
    result.update(
        {
            "status": "temporally_associated_candidates",
            "ammeter_track": ammeter_track,
            "voltmeter_track": voltmeter_track,
            "ammeter_observation": ammeter_observation,
            "voltmeter_observation": voltmeter_observation,
            "frame_gap": frame_gap,
            "evidence_insufficient_reason": "identity_and_temporal_cooccurrence_only;_no_switch_or_rubric_inference",
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative local analog-meter candidate search V4.")
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug-dir")
    parser.add_argument("--roi-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else None
    roi_dir = Path(args.roi_dir).expanduser().resolve() if args.roi_dir else None
    measurements = []
    for index, raw_path in enumerate(args.image, start=1):
        path = Path(raw_path).expanduser().resolve()
        debug_path = debug_dir / f"{index:03d}_{path.stem}_colored_v4.jpg" if debug_dir else None
        measurements.append(analyze_image(path, debug_path, roi_dir))
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "colored_meter_detection_v4_sequence",
        "image_count": len(measurements),
        "measurements": measurements,
        "event_candidate_association": associate_event_candidates(measurements),
        "qwen_called": False,
        "excel_accessed": False,
        "source_video_accessed": False,
        "score_computed": False,
        "warnings": [
            "Event association may use different still frames from the supplied event; it is not a Rubric result.",
            "Temporal association does not establish switch closure, wiring correctness, pointer normality, or appropriate range.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(safe_json(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
