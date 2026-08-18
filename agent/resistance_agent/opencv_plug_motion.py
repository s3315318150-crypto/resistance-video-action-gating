"""Pure-OpenCV three-frame banana-plug motion verification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class Base:
    track_id: int
    box: tuple[int, int, int, int]
    center: tuple[float, float]
    short_side: float
    area: float
    contour: np.ndarray


def orange_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([2, 105, 55]), np.array([28, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def base_candidates(frame: np.ndarray) -> list[Base]:
    height, width = frame.shape[:2]
    frame_area = float(height * width)
    output: list[Base] = []
    for contour in cv2.findContours(
        orange_mask(frame), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]:
        area = float(cv2.contourArea(contour))
        if not frame_area * 0.0006 <= area <= frame_area * 0.075:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if min(box_width, box_height) < 18 or max(box_width, box_height) < 45:
            continue
        if max(box_width, box_height) / max(min(box_width, box_height), 1) > 7.5:
            continue
        moments = cv2.moments(contour)
        center = (
            (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            if moments["m00"]
            else (x + box_width / 2.0, y + box_height / 2.0)
        )
        output.append(
            Base(
                -1,
                (x, y, box_width, box_height),
                center,
                float(min(box_width, box_height)),
                area,
                contour,
            )
        )
    return output


def assign_base_tracks(
    previous: list[Base], current: list[Base], next_track_id: int
) -> tuple[list[Base], int]:
    pairs: list[tuple[float, int, int]] = []
    for previous_index, old in enumerate(previous):
        for current_index, new in enumerate(current):
            scale = max(old.short_side, new.short_side, 1.0)
            distance = math.dist(old.center, new.center) / scale
            area_change = abs(math.log(max(new.area, 1.0) / max(old.area, 1.0)))
            pairs.append((distance + 0.35 * area_change, previous_index, current_index))
    used_previous: set[int] = set()
    used_current: set[int] = set()
    for cost, previous_index, current_index in sorted(pairs):
        if cost > 1.25 or previous_index in used_previous or current_index in used_current:
            continue
        current[current_index].track_id = previous[previous_index].track_id
        used_previous.add(previous_index)
        used_current.add(current_index)
    for candidate in current:
        if candidate.track_id < 0:
            candidate.track_id = next_track_id
            next_track_id += 1
    return current, next_track_id


def color_masks(frame: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 125, 45]), np.array([8, 255, 255])),
        cv2.inRange(hsv, np.array([170, 105, 40]), np.array([179, 255, 255])),
    )
    black = cv2.inRange(gray, 0, 78)
    skin = cv2.inRange(hsv, np.array([0, 18, 65]), np.array([27, 190, 255]))
    orange = orange_mask(frame)
    output: dict[str, np.ndarray] = {}
    for name, mask in (("red", red), ("black", black)):
        cleaned = mask.copy()
        cleaned[skin > 0] = 0
        cleaned[orange > 0] = 0
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        output[name] = cv2.morphologyEx(
            cleaned, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
    return output


def _mask_ratio(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = box
    crop = mask[top:bottom, left:right]
    return float(cv2.countNonZero(crop)) / max(crop.size, 1)


def handle_boxes(frame: np.ndarray, limit: int = 8) -> list[list[int]]:
    """Locate dark switch grips so their motion cannot become plug motion."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orange = cv2.inRange(hsv, np.array([0, 105, 55]), np.array([28, 255, 255]))
    green = cv2.inRange(hsv, np.array([30, 55, 30]), np.array([105, 255, 255]))
    dark = cv2.inRange(gray, 0, 95)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ranked: list[tuple[float, list[int]]] = []
    for contour in cv2.findContours(
        dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]:
        area = float(cv2.contourArea(contour))
        if area < 120 or area > width * height * 0.012:
            continue
        (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
        length, thickness = max(rect_width, rect_height), min(rect_width, rect_height)
        if length < width * 0.025 or length > width * 0.22 or thickness < 5:
            continue
        aspect = length / max(thickness, 1.0)
        rectangularity = area / max(length * thickness, 1.0)
        if not 2.4 <= aspect <= 14.0 or rectangularity < 0.28:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        pad_x = int(max(box_width * 1.5, length * 0.8))
        pad_y = int(max(box_height * 2.0, length * 0.7))
        context = (
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(width, x + box_width + pad_x),
            min(height, y + box_height + pad_y),
        )
        orange_ratio = _mask_ratio(orange, context)
        if orange_ratio < 0.012:
            continue
        green_ratio = _mask_ratio(green, context)
        score = (
            min(aspect / 5.0, 1.4)
            + min(rectangularity / 0.65, 1.2)
            + 1.8 * min(orange_ratio / 0.12, 1.0)
            - 1.8 * min(green_ratio / 0.10, 1.0)
        )
        ranked.append((score, [x, y, box_width, box_height]))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in ranked[:limit]]


def component_features(
    frame: np.ndarray,
    base: Base,
    color: str,
    mask: np.ndarray,
    detected_handles: list[list[int]],
) -> list[dict[str, Any]]:
    height, width = mask.shape
    base_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(base_mask, [base.contour], -1, 255, -1)
    outside_distance = cv2.distanceTransform(
        cv2.bitwise_not(base_mask), cv2.DIST_L2, 5
    )
    x, y, box_width, box_height = base.box
    radius = int(round(base.short_side * 2.2))
    local = np.zeros(mask.shape, np.uint8)
    cv2.rectangle(
        local,
        (max(0, x - radius), max(0, y - radius)),
        (
            min(width - 1, x + box_width + radius),
            min(height - 1, y + box_height + radius),
        ),
        255,
        -1,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_and(mask, local), 8
    )
    output: list[dict[str, Any]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not 24 <= area <= int(width * height * 0.035):
            continue
        ys, xs = np.nonzero(labels == label)
        gaps = outside_distance[ys, xs]
        near_index = int(np.argmin(gaps))
        near = np.array([float(xs[near_index]), float(ys[near_index])])
        base_center = np.array(base.center)
        radial = near - base_center
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm < 1.0:
            continue
        radial /= radial_norm
        points = np.column_stack([xs, ys]).astype(np.float32)
        projections = (points - near) @ radial
        min_gap = float(np.min(gaps)) / max(base.short_side, 1.0)
        max_gap = float(np.max(gaps)) / max(base.short_side, 1.0)
        radial_extent = float(np.ptp(projections)) / max(base.short_side, 1.0)
        component_x, component_y, component_width, component_height = cv2.boundingRect(
            points.astype(np.int32)
        )
        thickness = min(component_width, component_height) / max(base.short_side, 1.0)
        near_pixels = int(np.count_nonzero(gaps <= base.short_side * 0.22))
        far_pixels = int(np.count_nonzero(gaps >= base.short_side * 0.42))
        cable_continuation = min(far_pixels / max(near_pixels, 1), 1.0)
        handle_overlap = 0.0
        component_box_area = max(component_width * component_height, 1)
        if color == "black":
            for handle_x, handle_y, handle_width, handle_height in detected_handles:
                intersection = max(
                    0,
                    min(component_x + component_width, handle_x + handle_width)
                    - max(component_x, handle_x),
                ) * max(
                    0,
                    min(component_y + component_height, handle_y + handle_height)
                    - max(component_y, handle_y),
                )
                handle_overlap = max(handle_overlap, intersection / component_box_area)
        normalized_centroid = (
            (float(np.mean(xs)) - base.center[0]) / base.short_side,
            (float(np.mean(ys)) - base.center[1]) / base.short_side,
        )
        is_lead = (
            min_gap <= 0.28
            and max_gap >= 0.42
            and radial_extent >= 0.35
            and cable_continuation >= 0.10
            and (color != "black" or handle_overlap < 0.72 or max_gap >= 0.75)
        )
        output.append(
            {
                "color": color,
                "component_box_xywh": [
                    component_x,
                    component_y,
                    component_width,
                    component_height,
                ],
                "area": area,
                "min_gap_norm": round(min_gap, 5),
                "max_gap_norm": round(max_gap, 5),
                "radial_extent_norm": round(radial_extent, 5),
                "thickness_norm": round(thickness, 5),
                "cable_continuation": round(cable_continuation, 5),
                "handle_overlap": round(handle_overlap, 5),
                "centroid_norm": [round(float(value), 5) for value in normalized_centroid],
                "is_lead": bool(is_lead),
            }
        )
    return output


def frame_candidates(
    frame: np.ndarray,
    previous_bases: list[Base],
    next_track_id: int,
) -> tuple[list[Base], int, dict[int, list[dict[str, Any]]]]:
    bases, next_track_id = assign_base_tracks(
        previous_bases, base_candidates(frame), next_track_id
    )
    masks = color_masks(frame)
    handles = handle_boxes(frame)
    by_base: dict[int, list[dict[str, Any]]] = {}
    for base in bases:
        candidates: list[dict[str, Any]] = []
        for color, mask in masks.items():
            candidates.extend(component_features(frame, base, color, mask, handles))
        by_base[base.track_id] = candidates
    return bases, next_track_id, by_base


def match_three(
    previous: list[dict[str, Any]],
    middle: list[dict[str, Any]],
    following: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    for center in middle:
        if not center["is_lead"]:
            continue
        center_point = np.asarray(center["centroid_norm"], dtype=float)
        neighbors: list[dict[str, Any] | None] = []
        for rows in (previous, following):
            ranked = sorted(
                (
                    (
                        float(
                            np.linalg.norm(
                                np.asarray(item["centroid_norm"], dtype=float)
                                - center_point
                            )
                        ),
                        item,
                    )
                    for item in rows
                    if item["color"] == center["color"] and item["is_lead"]
                ),
                key=lambda pair: pair[0],
            )
            neighbors.append(ranked[0][1] if ranked and ranked[0][0] <= 0.90 else None)
        before, after = neighbors
        if before is None or after is None:
            continue
        points = [
            np.asarray(item["centroid_norm"], dtype=float)
            for item in (before, center, after)
        ]
        steps = [points[1] - points[0], points[2] - points[1]]
        coherence = float(
            np.dot(steps[0], steps[1])
            / max(np.linalg.norm(steps[0]) * np.linalg.norm(steps[1]), 1e-9)
        )
        displacement = float(np.linalg.norm(points[2] - points[0]))
        gaps = [float(item["min_gap_norm"]) for item in (before, center, after)]
        gap_steps = [gaps[1] - gaps[0], gaps[2] - gaps[1]]
        thickness = float(
            np.median([item["thickness_norm"] for item in (before, center, after)])
        )
        contact_flip = min(gaps[0], gaps[2]) <= max(0.28, thickness) and max(
            gaps[0], gaps[2]
        ) > min(gaps[0], gaps[2]) + max(thickness, 0.08)
        monotonic = (
            gap_steps[0] * gap_steps[1] >= 0
            and max(abs(value) for value in gap_steps) > 1e-5
        )
        real_transition = bool(
            contact_flip
            and monotonic
            and coherence >= -0.15
            and displacement >= max(thickness, 0.08)
        )
        occupancy_delta = abs(gaps[2] - gaps[0])
        occupancy_transition = bool(
            not contact_flip
            and monotonic
            and coherence >= 0.25
            and occupancy_delta >= max(0.12 * thickness, 0.02)
            and displacement >= max(0.25 * thickness, 0.04)
        )
        wiring_activity = real_transition or occupancy_transition
        confidence = float(
            np.clip(
                0.45
                + 0.20 * min(displacement / max(thickness, 0.08), 1.5) / 1.5
                + 0.20 * max(coherence, 0.0)
                + 0.15 * min(abs(gaps[2] - gaps[0]) / max(thickness, 0.08), 1.0),
                0.0,
                0.99,
            )
        )
        tracks.append(
            {
                "color": center["color"],
                "boxes": [
                    item["component_box_xywh"] for item in (before, center, after)
                ],
                "gaps_norm": [round(value, 5) for value in gaps],
                "gap_delta_norm": round(gaps[2] - gaps[0], 5),
                "relative_displacement_norm": round(displacement, 5),
                "direction_coherence": round(coherence, 5),
                "connector_thickness_norm": round(thickness, 5),
                "contact_flip": bool(contact_flip),
                "monotonic": bool(monotonic),
                "real_transition": real_transition,
                "occupancy_transition": occupancy_transition,
                "wiring_activity": wiring_activity,
                "transition_kind": (
                    "contact_flip"
                    if real_transition
                    else "occupancy_change"
                    if occupancy_transition
                    else None
                ),
                "confidence": round(confidence, 4),
            }
        )
    return tracks


def transitions_for_triple(
    previous: dict[str, Any], middle: dict[str, Any], following: dict[str, Any]
) -> list[dict[str, Any]]:
    common_ids = (
        set(previous["candidates"])
        & set(middle["candidates"])
        & set(following["candidates"])
    )
    transitions: list[dict[str, Any]] = []
    for track_id in sorted(common_ids):
        for track in match_three(
            previous["candidates"][track_id],
            middle["candidates"][track_id],
            following["candidates"][track_id],
        ):
            if track["wiring_activity"]:
                transitions.append(
                    {
                        "window_id": middle["window_id"],
                        "stage": middle["stage"],
                        "timestamp_seconds": middle["timestamp_seconds"],
                        "frame_number": middle["frame_number"],
                        "base_track_id": track_id,
                        "support_frames": [
                            {
                                "window_id": sample["window_id"],
                                "stage": sample["stage"],
                                "timestamp_seconds": sample["timestamp_seconds"],
                                "frame_number": sample["frame_number"],
                                "role": role,
                            }
                            for role, sample in zip(
                                ("before", "center", "after"),
                                (previous, middle, following),
                            )
                        ],
                        **track,
                    }
                )
    return transitions
