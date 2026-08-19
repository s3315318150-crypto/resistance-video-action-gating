from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np


def pointer_angle_deg(pivot: tuple[float, float], point: tuple[float, float]) -> float:
    px, py = pivot
    x, y = point
    return float(math.degrees(math.atan2(py - y, x - px)) % 360.0)


def angular_distance_deg(a: float, b: float) -> float:
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


def frame_time_seconds(path: str | Path) -> float | None:
    match = re.search(r"_(\d+(?:\.\d+)?)s(?:_|$)", Path(path).stem)
    return None if match is None else float(match.group(1))


def summarize_temporal_stability(
    frame_results: list[dict[str, Any]],
    roles: list[str],
    min_frames: int = 3,
    max_gap_seconds: float = 1.5,
    max_mad_deg: float = 2.0,
) -> dict[str, Any]:
    if min_frames < 1:
        raise ValueError("min_frames must be positive")
    if max_gap_seconds <= 0.0:
        raise ValueError("max_gap_seconds must be positive")
    if max_mad_deg < 0.0:
        raise ValueError("max_mad_deg cannot be negative")

    summary: dict[str, Any] = {}
    for role in roles:
        candidates: list[dict[str, Any]] = []
        for frame_index, frame_result in enumerate(frame_results):
            for role_result in frame_result.get("roles", []):
                if role_result.get("role") != role:
                    continue
                pointer = role_result.get("pointer", {})
                angle = pointer.get("angle_deg")
                if not pointer.get("detected") or angle is None or not math.isfinite(float(angle)):
                    continue
                candidates.append(
                    {
                        "frame_index": frame_index,
                        "frame": frame_result.get("frame"),
                        "annotated_path": frame_result.get("annotated_path"),
                        "timestamp_seconds": frame_time_seconds(str(frame_result.get("frame", ""))),
                        "angle_deg": float(angle),
                    }
                )

        runs: list[list[dict[str, Any]]] = []
        for candidate in candidates:
            if not runs:
                runs.append([candidate])
                continue
            previous = runs[-1][-1]
            consecutive_input = candidate["frame_index"] == previous["frame_index"] + 1
            previous_time = previous["timestamp_seconds"]
            current_time = candidate["timestamp_seconds"]
            acceptable_time_gap = (
                previous_time is None
                or current_time is None
                or 0.0 <= current_time - previous_time <= max_gap_seconds
            )
            if consecutive_input and acceptable_time_gap:
                runs[-1].append(candidate)
            else:
                runs.append([candidate])

        windows: list[dict[str, Any]] = []
        for run in runs:
            angles = np.asarray([item["angle_deg"] for item in run], dtype=np.float64)
            median = float(np.median(angles))
            mad = float(np.median(np.abs(angles - median)))
            if len(run) < min_frames:
                status = "evidence_insufficient"
                reason = "fewer_than_minimum_consecutive_candidates"
            elif mad > max_mad_deg:
                status = "unstable_candidate"
                reason = "angle_mad_above_threshold"
            else:
                status = "stable_candidate"
                reason = None
            windows.append(
                {
                    "status": status,
                    "reason": reason,
                    "candidate_count": len(run),
                    "start_frame": run[0]["frame"],
                    "end_frame": run[-1]["frame"],
                    "start_timestamp_seconds": run[0]["timestamp_seconds"],
                    "end_timestamp_seconds": run[-1]["timestamp_seconds"],
                    "median_angle_deg": _json_number(median),
                    "angle_mad_deg": _json_number(mad),
                    "angle_min_deg": _json_number(angles.min()),
                    "angle_max_deg": _json_number(angles.max()),
                    "frames": [
                        {
                            "frame": item["frame"],
                            "annotated_path": item["annotated_path"],
                            "timestamp_seconds": item["timestamp_seconds"],
                            "angle_deg": _json_number(item["angle_deg"]),
                        }
                        for item in run
                    ],
                }
            )
        summary[role] = {
            "candidate_count": len(candidates),
            "stable_window_count": sum(item["status"] == "stable_candidate" for item in windows),
            "windows": windows,
        }
    return {
        "semantics": "Temporal stability groups same-role candidates from consecutive sampled frames. It remains candidate evidence, not a reading or Rubric decision.",
        "minimum_consecutive_candidates": min_frames,
        "maximum_gap_seconds": max_gap_seconds,
        "maximum_mad_deg": max_mad_deg,
        "roles": summary,
    }


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    area_first = abs(float(cv2.contourArea(first)))
    area_second = abs(float(cv2.contourArea(second)))
    if area_first <= 0.0 or area_second <= 0.0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = area_first + area_second - float(intersection)
    return 0.0 if union <= 0.0 else float(intersection / union)


def _json_number(value: float | np.floating[Any]) -> float:
    return round(float(value), 6)


def _point_list(point: np.ndarray | tuple[float, float]) -> list[float]:
    return [_json_number(point[0]), _json_number(point[1])]


@dataclass(frozen=True)
class TrackingConfig:
    image_scale: float = 0.5
    ratio_test: float = 0.72
    ransac_threshold_px: float = 7.0
    min_good_matches: int = 12
    min_inliers: int = 10
    min_inlier_ratio: float = 0.30
    min_quad_area_ratio: float = 0.005
    max_quad_area_ratio: float = 0.15
    max_edge_length_ratio: float = 3.0
    min_light_ratio: float = 0.42
    role_collision_iou: float = 0.30
    collision_strength_ratio: float = 1.40


@dataclass(frozen=True)
class ScanConfig:
    angle_min_deg: float = 25.0
    angle_max_deg: float = 155.0
    angle_step_deg: float = 0.25
    radial_start_px: float = 45.0
    radial_end_fraction: float = 0.90
    red_dilate_px: int = 7
    min_score: float = 0.16
    min_dark_fraction: float = 0.20
    min_continuity: float = 0.15
    min_peak_margin: float = 0.008
    max_red_occlusion_fraction: float = 0.35
    hough_pivot_distance_px: float = 24.0
    hough_line_angle_tolerance_deg: float = 7.0
    hough_min_line_fraction: float = 0.40
    hough_min_far_radius_fraction: float = 0.48
    hough_max_near_radius_fraction: float = 0.28
    hough_cluster_tolerance_deg: float = 4.0
    hough_refine_window_deg: float = 3.0


@dataclass
class MeterTemplate:
    role: str
    reference_frame: Path
    quad: np.ndarray
    pivot_source: np.ndarray
    canonical_size: tuple[int, int]
    source_keypoints: list[cv2.KeyPoint]
    source_descriptors: np.ndarray
    reference_face: np.ndarray
    canonical_pivot: np.ndarray

    @classmethod
    def build(
        cls,
        role: str,
        reference_frame: Path,
        quad: np.ndarray,
        pivot_source: np.ndarray,
        canonical_size: tuple[int, int],
        sift: cv2.SIFT,
    ) -> "MeterTemplate":
        image = cv2.imread(str(reference_frame), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read reference frame: {reference_frame}")
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) < 20:
            raise ValueError(f"Reference template {role} has too few SIFT features")

        width, height = canonical_size
        destination = np.float32(
            [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]]
        )
        source_to_face = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
        reference_face = cv2.warpPerspective(image, source_to_face, canonical_size)
        canonical_pivot = cv2.perspectiveTransform(
            pivot_source.reshape(1, 1, 2).astype(np.float32), source_to_face
        ).reshape(2)
        return cls(
            role=role,
            reference_frame=reference_frame,
            quad=quad.astype(np.float32),
            pivot_source=pivot_source.astype(np.float32),
            canonical_size=canonical_size,
            source_keypoints=keypoints,
            source_descriptors=descriptors,
            reference_face=reference_face,
            canonical_pivot=canonical_pivot,
        )


def _quad_edge_ratio(quad: np.ndarray) -> float:
    lengths = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    minimum = min(lengths)
    return float("inf") if minimum <= 1e-6 else max(lengths) / minimum


def _face_quality(face: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    return {
        "mean_brightness": _json_number(gray.mean()),
        "light_ratio": _json_number(np.mean(gray >= 135)),
        "dark_ratio": _json_number(np.mean(gray <= 90)),
        "edge_density": _json_number(np.mean(edges > 0)),
        "laplacian_variance": _json_number(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def track_template(
    template: MeterTemplate,
    frame: np.ndarray,
    target_keypoints: list[cv2.KeyPoint],
    target_descriptors: np.ndarray | None,
    config: TrackingConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": template.role,
        "tracked": False,
        "status": "evidence_insufficient",
        "reasons": [],
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
    }
    if target_descriptors is None:
        result["reasons"] = ["target_has_no_sift_descriptors"]
        return result

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(template.source_descriptors, target_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < config.ratio_test * second.distance]
    result["good_matches"] = len(good)
    if len(good) < config.min_good_matches:
        result["reasons"] = ["too_few_feature_matches"]
        return result

    source = np.float32([template.source_keypoints[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    target = np.float32([target_keypoints[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    target /= float(config.image_scale)
    homography, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, config.ransac_threshold_px)
    if homography is None or inlier_mask is None or not np.all(np.isfinite(homography)):
        result["reasons"] = ["homography_failed"]
        return result

    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(good)
    projected_quad = cv2.perspectiveTransform(template.quad.reshape(-1, 1, 2), homography).reshape(-1, 2)
    projected_pivot = cv2.perspectiveTransform(
        template.pivot_source.reshape(1, 1, 2), homography
    ).reshape(2)
    area = abs(float(cv2.contourArea(projected_quad.astype(np.float32))))
    image_area = float(frame.shape[0] * frame.shape[1])
    area_ratio = area / image_area
    edge_ratio = _quad_edge_ratio(projected_quad)

    result.update(
        {
            "inliers": inliers,
            "inlier_ratio": _json_number(inlier_ratio),
            "match_strength": _json_number(inliers * inlier_ratio),
            "homography": [[_json_number(value) for value in row] for row in homography],
            "quad": [_point_list(point) for point in projected_quad],
            "pivot_source_projection": _point_list(projected_pivot),
            "quad_area_ratio": _json_number(area_ratio),
            "edge_length_ratio": _json_number(edge_ratio),
        }
    )

    reasons: list[str] = []
    if inliers < config.min_inliers:
        reasons.append("too_few_homography_inliers")
    if inlier_ratio < config.min_inlier_ratio:
        reasons.append("low_homography_inlier_ratio")
    if not cv2.isContourConvex(projected_quad.astype(np.float32)):
        reasons.append("projected_face_not_convex")
    if not config.min_quad_area_ratio <= area_ratio <= config.max_quad_area_ratio:
        reasons.append("projected_face_area_out_of_range")
    if edge_ratio > config.max_edge_length_ratio:
        reasons.append("projected_face_shape_implausible")
    height, width = frame.shape[:2]
    if np.any(projected_quad[:, 0] < 0.0) or np.any(projected_quad[:, 0] >= width):
        reasons.append("projected_face_clipped_horizontally")
    if np.any(projected_quad[:, 1] < 0.0) or np.any(projected_quad[:, 1] >= height):
        reasons.append("projected_face_clipped_vertically")

    destination = np.float32(
        [
            [0.0, 0.0],
            [template.canonical_size[0] - 1.0, 0.0],
            [template.canonical_size[0] - 1.0, template.canonical_size[1] - 1.0],
            [0.0, template.canonical_size[1] - 1.0],
        ]
    )
    if not reasons:
        frame_to_face = cv2.getPerspectiveTransform(projected_quad.astype(np.float32), destination)
        face = cv2.warpPerspective(frame, frame_to_face, template.canonical_size)
        quality = _face_quality(face)
        result["face_quality"] = quality
        if quality["light_ratio"] < config.min_light_ratio:
            reasons.append("warped_face_not_light_dial_like")
        else:
            result["face"] = face
            result["frame_to_face"] = frame_to_face

    result["reasons"] = reasons
    result["tracked"] = not reasons
    result["status"] = "tracked_candidate" if not reasons else "evidence_insufficient"
    return result


def resolve_role_collisions(tracks: dict[str, dict[str, Any]], config: TrackingConfig) -> None:
    if "ammeter" not in tracks or "voltmeter" not in tracks:
        return
    ammeter = tracks["ammeter"]
    voltmeter = tracks["voltmeter"]
    if not ammeter.get("tracked") or not voltmeter.get("tracked"):
        return
    first = np.float32(ammeter["quad"])
    second = np.float32(voltmeter["quad"])
    overlap = polygon_iou(first, second)
    ammeter["other_role_iou"] = _json_number(overlap)
    voltmeter["other_role_iou"] = _json_number(overlap)
    if overlap >= config.role_collision_iou:
        strength_a = float(ammeter.get("match_strength", 0.0))
        strength_v = float(voltmeter.get("match_strength", 0.0))
        stronger = max(strength_a, strength_v)
        weaker = min(strength_a, strength_v)
        if weaker > 0.0 and stronger / weaker >= config.collision_strength_ratio:
            rejected = ammeter if strength_a < strength_v else voltmeter
            rejected["tracked"] = False
            rejected["status"] = "evidence_insufficient"
            rejected["reasons"].append("role_collision_weaker_match")
        else:
            for item in (ammeter, voltmeter):
                item["tracked"] = False
                item["status"] = "evidence_insufficient"
                item["reasons"].append("ambiguous_same_face_role_collision")
        return

    center_a = first.mean(axis=0)
    center_v = second.mean(axis=0)
    if center_a[0] >= center_v[0]:
        for item in (ammeter, voltmeter):
            item["tracked"] = False
            item["status"] = "evidence_insufficient"
            item["reasons"].append("fixed_layout_role_order_conflict")


def _red_occlusion_mask(face: np.ndarray, dilate_px: int) -> np.ndarray:
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, np.array([0, 75, 65]), np.array([14, 255, 255]))
    high_red = cv2.inRange(hsv, np.array([168, 65, 55]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(low_red, high_red)
    if dilate_px > 0:
        size = dilate_px * 2 + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return mask


def _ray_limit(pivot: np.ndarray, angle_deg: float, width: int, height: int) -> float:
    radians = math.radians(angle_deg)
    dx = math.cos(radians)
    dy = -math.sin(radians)
    limits: list[float] = []
    if dx > 1e-6:
        limits.append((width - 1.0 - pivot[0]) / dx)
    elif dx < -1e-6:
        limits.append((0.0 - pivot[0]) / dx)
    if dy > 1e-6:
        limits.append((height - 1.0 - pivot[1]) / dy)
    elif dy < -1e-6:
        limits.append((0.0 - pivot[1]) / dy)
    positive = [value for value in limits if value > 0.0]
    return 0.0 if not positive else min(positive)


def _longest_tolerant_run(values: np.ndarray, max_gap: int = 3) -> int:
    best = 0
    start = 0
    gap = 0
    for index, value in enumerate(values):
        if value:
            gap = 0
            best = max(best, index - start + 1)
        else:
            gap += 1
            if gap > max_gap:
                start = index + 1
                gap = 0
    return best


def _sample_gray(gray: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        gray,
        x.astype(np.float32).reshape(1, -1),
        y.astype(np.float32).reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    ).reshape(-1)


def _hough_support(
    gray: np.ndarray,
    red_mask: np.ndarray,
    pivot: np.ndarray,
    config: ScanConfig,
) -> dict[str, Any]:
    prepared = gray.copy()
    prepared[red_mask > 0] = 255
    edges = cv2.Canny(prepared, 25, 100)
    lines = cv2.HoughLinesP(edges, 1.0, np.pi / 720.0, 20, minLineLength=55, maxLineGap=28)
    candidates: list[dict[str, Any]] = []
    if lines is not None:
        for raw in np.asarray(lines).reshape(-1, 4):
            first = raw[:2].astype(np.float32)
            second = raw[2:].astype(np.float32)
            vector = second - first
            length = float(np.linalg.norm(vector))
            if length <= 1e-6:
                continue
            offset = pivot - first
            distance = abs(float(vector[0] * offset[1] - vector[1] * offset[0])) / length
            if distance > config.hough_pivot_distance_px:
                continue
            radius_first = float(np.linalg.norm(first - pivot))
            radius_second = float(np.linalg.norm(second - pivot))
            far = first if radius_first >= radius_second else second
            near = second if radius_first >= radius_second else first
            near_radius = min(radius_first, radius_second)
            far_radius = max(radius_first, radius_second)
            angle = pointer_angle_deg(tuple(pivot), tuple(far))
            if not config.angle_min_deg <= angle <= config.angle_max_deg:
                continue
            line_angle = pointer_angle_deg(tuple(near), tuple(far))
            line_angle_difference = angular_distance_deg(line_angle, angle)
            if line_angle_difference > config.hough_line_angle_tolerance_deg:
                continue
            ray_limit = _ray_limit(pivot, angle, gray.shape[1], gray.shape[0]) * config.radial_end_fraction
            if ray_limit <= 1.0:
                continue
            line_fraction = length / ray_limit
            far_fraction = far_radius / ray_limit
            near_fraction = near_radius / ray_limit
            if line_fraction < config.hough_min_line_fraction:
                continue
            if far_fraction < config.hough_min_far_radius_fraction:
                continue
            if near_fraction > config.hough_max_near_radius_fraction:
                continue
            score = (
                1.7 * line_fraction
                + far_fraction
                - 0.025 * distance
                - 0.7 * near_fraction
                - 0.05 * line_angle_difference
            )
            item = {
                "supported": True,
                "angle_deg": _json_number(angle),
                "line_angle_deg": _json_number(line_angle),
                "line_angle_difference_deg": _json_number(line_angle_difference),
                "pivot_line_distance_px": _json_number(distance),
                "line_length_px": _json_number(length),
                "line_length_fraction": _json_number(line_fraction),
                "far_radius_px": _json_number(far_radius),
                "far_radius_fraction": _json_number(far_fraction),
                "near_radius_fraction": _json_number(near_fraction),
                "line": [int(value) for value in raw],
                "score": _json_number(score),
            }
            candidates.append(item)
    if not candidates:
        return {"supported": False, "reason": "no_long_radial_hough_line_from_pivot_to_scale"}

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    seed = candidates[0]
    cluster = [
        item
        for item in candidates
        if angular_distance_deg(float(item["angle_deg"]), float(seed["angle_deg"]))
        <= config.hough_cluster_tolerance_deg
    ]
    weights = np.asarray([max(1e-6, float(item["line_length_px"])) for item in cluster])
    cluster_angle = float(
        np.average(np.asarray([float(item["angle_deg"]) for item in cluster]), weights=weights)
    )
    best = dict(seed)
    best["cluster_angle_deg"] = _json_number(cluster_angle)
    best["cluster_size"] = len(cluster)
    best["accepted_line_count"] = len(candidates)
    best["top_lines"] = candidates[:5]
    return best


def scan_pointer(
    face: np.ndarray,
    pivot: np.ndarray,
    config: ScanConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    gray_raw = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.GaussianBlur(clahe.apply(gray_raw), (3, 3), 0.0)
    red_mask = _red_occlusion_mask(face, config.red_dilate_px)
    height, width = gray.shape
    angles = np.arange(config.angle_min_deg, config.angle_max_deg + 1e-6, config.angle_step_deg)
    metrics: list[dict[str, Any]] = []
    scores: list[float] = []

    for angle in angles:
        ray_limit = _ray_limit(pivot, float(angle), width, height) * config.radial_end_fraction
        if ray_limit <= config.radial_start_px + 20.0:
            metrics.append({"score": 0.0, "dark_fraction": 0.0, "continuity": 0.0, "red_fraction": 1.0})
            scores.append(0.0)
            continue
        radii = np.arange(config.radial_start_px, ray_limit, 1.0, dtype=np.float32)
        radians = math.radians(float(angle))
        dx = math.cos(radians)
        dy = -math.sin(radians)
        perpendicular_x = -dy
        perpendicular_y = dx
        x = pivot[0] + radii * dx
        y = pivot[1] + radii * dy
        centers = np.vstack(
            [_sample_gray(gray, x + offset * perpendicular_x, y + offset * perpendicular_y) for offset in (-1.0, 0.0, 1.0)]
        ).min(axis=0)
        sides = np.vstack(
            [_sample_gray(gray, x + offset * perpendicular_x, y + offset * perpendicular_y) for offset in (-8.0, -5.0, 5.0, 8.0)]
        ).mean(axis=0)
        red = _sample_gray(red_mask, x, y) > 0
        valid = ~red
        valid_count = int(valid.sum())
        if valid_count < max(30, int(0.55 * len(radii))):
            metrics.append({"score": 0.0, "dark_fraction": 0.0, "continuity": 0.0, "red_fraction": _json_number(red.mean())})
            scores.append(0.0)
            continue
        center_valid = centers[valid]
        side_valid = sides[valid]
        absolute_darkness = np.clip((190.0 - center_valid) / 105.0, 0.0, 1.0)
        local_contrast = np.clip((side_valid - center_valid) / 65.0, 0.0, 1.0)
        support = (center_valid < 180.0) & ((side_valid - center_valid) > 4.0)
        weighted = 0.68 * absolute_darkness + 0.32 * local_contrast
        score = float(weighted.mean())
        dark_fraction = float(np.mean(center_valid < 180.0))
        continuity = float(_longest_tolerant_run(support) / max(1, len(support)))
        item = {
            "score": score,
            "dark_fraction": dark_fraction,
            "contrast_fraction": float(np.mean((side_valid - center_valid) > 6.0)),
            "continuity": continuity,
            "red_fraction": float(red.mean()),
            "ray_limit_px": float(ray_limit),
        }
        metrics.append(item)
        scores.append(score + 0.06 * continuity)

    score_array = np.asarray(scores, dtype=np.float32)
    smooth = cv2.GaussianBlur(score_array.reshape(1, -1), (0, 0), sigmaX=2.0).reshape(-1)
    hough = _hough_support(gray, red_mask, pivot, config)
    if hough.get("supported"):
        hough_angle = float(hough["cluster_angle_deg"])
        eligible = np.asarray(
            [angular_distance_deg(float(angle), hough_angle) <= config.hough_refine_window_deg for angle in angles]
        )
        constrained = np.where(eligible, smooth, -np.inf)
        best_index = int(np.argmax(constrained))
    else:
        best_index = int(np.argmax(smooth))
    best_angle = float(angles[best_index])
    exclusion = int(round(4.0 / config.angle_step_deg))
    comparison = smooth.copy()
    comparison[max(0, best_index - exclusion) : min(len(comparison), best_index + exclusion + 1)] = -np.inf
    second_score = float(np.max(comparison)) if np.any(np.isfinite(comparison)) else 0.0
    peak_margin = float(smooth[best_index] - second_score)
    best_metrics = metrics[best_index]

    reasons: list[str] = []
    if best_metrics["score"] < config.min_score:
        reasons.append("radial_dark_line_score_too_low")
    if best_metrics["dark_fraction"] < config.min_dark_fraction:
        reasons.append("radial_dark_support_too_sparse")
    if best_metrics["continuity"] < config.min_continuity:
        reasons.append("radial_dark_line_not_continuous")
    if peak_margin < config.min_peak_margin and not hough.get("supported"):
        reasons.append("radial_angle_peak_ambiguous")
    if best_metrics["red_fraction"] > config.max_red_occlusion_fraction:
        reasons.append("candidate_ray_too_occluded_by_red_lead")
    if not hough.get("supported"):
        reasons.append("no_long_radial_hough_line_from_pivot_to_scale")

    tip_radius = float(best_metrics.get("ray_limit_px", 0.0))
    if hough.get("supported"):
        tip_radius = min(tip_radius, max(config.radial_start_px, float(hough["far_radius_px"])))
    radians = math.radians(best_angle)
    tip = np.float32(
        [pivot[0] + tip_radius * math.cos(radians), pivot[1] - tip_radius * math.sin(radians)]
    )
    result = {
        "detected": not reasons,
        "state": "angle_candidate" if not reasons else "evidence_insufficient",
        "angle_deg": _json_number(best_angle),
        "pivot": _point_list(pivot),
        "tip": _point_list(tip),
        "tip_radius_px": _json_number(tip_radius),
        "score": _json_number(best_metrics["score"]),
        "smoothed_peak_score": _json_number(smooth[best_index]),
        "peak_margin": _json_number(peak_margin),
        "dark_fraction": _json_number(best_metrics["dark_fraction"]),
        "contrast_fraction": _json_number(best_metrics.get("contrast_fraction", 0.0)),
        "continuity": _json_number(best_metrics["continuity"]),
        "red_occlusion_fraction": _json_number(best_metrics["red_fraction"]),
        "hough_support": hough,
        "global_radial_peak_angle_deg": _json_number(float(angles[int(np.argmax(smooth))])),
        "reasons": reasons,
        "manual_review_required": True,
        "reading_value": None,
        "reading_computed": False,
    }

    debug = face.copy()
    red_overlay = np.zeros_like(debug)
    red_overlay[:, :, 2] = red_mask
    debug = cv2.addWeighted(debug, 1.0, red_overlay, 0.25, 0.0)
    color = (0, 180, 0) if not reasons else (0, 165, 255)
    pivot_int = tuple(np.round(pivot).astype(int))
    tip_int = tuple(np.round(tip).astype(int))
    cv2.line(debug, pivot_int, tip_int, color, 3, cv2.LINE_AA)
    cv2.circle(debug, pivot_int, 8, (255, 80, 0), -1, cv2.LINE_AA)
    cv2.circle(debug, tip_int, 7, color, -1, cv2.LINE_AA)
    for boundary_angle in (config.angle_min_deg, config.angle_max_deg):
        limit = _ray_limit(pivot, boundary_angle, width, height) * config.radial_end_fraction
        radians_boundary = math.radians(boundary_angle)
        endpoint = (
            int(round(pivot[0] + limit * math.cos(radians_boundary))),
            int(round(pivot[1] - limit * math.sin(radians_boundary))),
        )
        cv2.line(debug, pivot_int, endpoint, (255, 255, 0), 1, cv2.LINE_AA)
    label = f"{result['state']} angle={best_angle:.2f} score={best_metrics['score']:.3f}"
    cv2.putText(debug, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return result, debug, red_mask


def _load_configuration(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("value_mapping_enabled", False):
        raise ValueError("Fixed-meter candidate mode does not permit value mapping")
    templates = payload.get("templates")
    if not isinstance(templates, list) or not templates:
        raise ValueError("Configuration must contain at least one template")
    return payload, templates


def _clean_track_for_json(track: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in track.items()
        if key not in {"face", "frame_to_face"}
    }


def run_fixed_meter_search(config_path: Path, frames_dir: Path, output_dir: Path) -> dict[str, Any]:
    payload, template_payloads = _load_configuration(config_path)
    canonical_size = tuple(int(value) for value in payload.get("canonical_size", [640, 520]))
    if len(canonical_size) != 2 or min(canonical_size) <= 0:
        raise ValueError("canonical_size must be [width, height]")
    tracking_config = TrackingConfig(**payload.get("tracking", {}))
    scan_config = ScanConfig(**payload.get("scan", {}))
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.02, edgeThreshold=12)

    templates: list[MeterTemplate] = []
    for item in template_payloads:
        reference = Path(item["reference_frame"])
        if not reference.is_absolute():
            reference = (config_path.parent / reference).resolve()
        templates.append(
            MeterTemplate.build(
                role=str(item["role"]),
                reference_frame=reference,
                quad=np.float32(item["quad"]),
                pivot_source=np.float32(item["pivot"]),
                canonical_size=canonical_size,
                sift=sift,
            )
        )

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No JPEG frames found in {frames_dir}")
    annotated_dir = output_dir / "annotated_frames"
    faces_dir = output_dir / "face_rois"
    masks_dir = output_dir / "red_masks"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    frame_results: list[dict[str, Any]] = []
    valid_by_role = {template.role: 0 for template in templates}
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            frame_results.append({"frame": str(frame_path.resolve()), "status": "read_failed"})
            continue
        small = cv2.resize(
            frame,
            None,
            fx=tracking_config.image_scale,
            fy=tracking_config.image_scale,
            interpolation=cv2.INTER_AREA,
        )
        target_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        target_keypoints, target_descriptors = sift.detectAndCompute(target_gray, None)
        tracks = {
            template.role: track_template(
                template,
                frame,
                target_keypoints,
                target_descriptors,
                tracking_config,
            )
            for template in templates
        }
        resolve_role_collisions(tracks, tracking_config)

        annotated = frame.copy()
        role_results: list[dict[str, Any]] = []
        for template in templates:
            track = tracks[template.role]
            pointer: dict[str, Any] = {
                "detected": False,
                "state": "evidence_insufficient",
                "angle_deg": None,
                "reasons": ["meter_face_not_reliably_tracked"],
                "reading_value": None,
                "reading_computed": False,
            }
            if track.get("tracked"):
                pointer, face_debug, red_mask = scan_pointer(
                    track["face"], template.canonical_pivot, scan_config
                )
                face_name = f"{frame_path.stem}_{template.role}_debug.jpg"
                raw_face_name = f"{frame_path.stem}_{template.role}_raw.jpg"
                mask_name = f"{frame_path.stem}_{template.role}_red_mask.png"
                cv2.imwrite(str(faces_dir / face_name), face_debug, [cv2.IMWRITE_JPEG_QUALITY, 96])
                cv2.imwrite(str(faces_dir / raw_face_name), track["face"], [cv2.IMWRITE_JPEG_QUALITY, 96])
                cv2.imwrite(str(masks_dir / mask_name), red_mask)
                track["face_debug_path"] = str((faces_dir / face_name).resolve())
                track["face_roi_path"] = str((faces_dir / raw_face_name).resolve())
                track["red_mask_path"] = str((masks_dir / mask_name).resolve())

                face_to_frame = np.linalg.inv(track["frame_to_face"])
                source_pivot = cv2.perspectiveTransform(
                    np.float32(pointer["pivot"]).reshape(1, 1, 2), face_to_frame
                ).reshape(2)
                source_tip = cv2.perspectiveTransform(
                    np.float32(pointer["tip"]).reshape(1, 1, 2), face_to_frame
                ).reshape(2)
                pointer["source_pivot"] = _point_list(source_pivot)
                pointer["source_tip"] = _point_list(source_tip)
                quad = np.round(np.float32(track["quad"])).astype(np.int32)
                color = (40, 190, 40) if pointer["detected"] else (0, 165, 255)
                cv2.polylines(annotated, [quad], True, color, 4, cv2.LINE_AA)
                cv2.line(
                    annotated,
                    tuple(np.round(source_pivot).astype(int)),
                    tuple(np.round(source_tip).astype(int)),
                    color,
                    4,
                    cv2.LINE_AA,
                )
                cv2.circle(annotated, tuple(np.round(source_pivot).astype(int)), 9, (255, 80, 0), -1)
                cv2.circle(annotated, tuple(np.round(source_tip).astype(int)), 8, color, -1)
                if pointer["detected"]:
                    valid_by_role[template.role] += 1
                label_point = tuple(quad[0])
                label = f"{template.role}: {pointer['state']}"
                cv2.putText(
                    annotated,
                    label,
                    (max(5, label_point[0]), max(30, label_point[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            role_results.append(
                {
                    "role": template.role,
                    "track": _clean_track_for_json(track),
                    "pointer": pointer,
                    "manual_review_required": True,
                }
            )

        annotated_path = annotated_dir / f"{frame_path.stem}_fixed_meter.jpg"
        cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 94])
        frame_results.append(
            {
                "frame": str(frame_path.resolve()),
                "annotated_path": str(annotated_path.resolve()),
                "resolution": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
                "roles": role_results,
                "qwen_called": False,
                "excel_accessed": False,
                "score_computed": False,
            }
        )

    elapsed = perf_counter() - started
    stability_payload = payload.get("stability", {})
    temporal_stability = summarize_temporal_stability(
        frame_results,
        [template.role for template in templates],
        min_frames=int(stability_payload.get("min_frames", 3)),
        max_gap_seconds=float(stability_payload.get("max_gap_seconds", 1.5)),
        max_mad_deg=float(stability_payload.get("max_mad_deg", 2.0)),
    )
    summary = {
        "schema_version": "fixed-meter-opencv-candidate-v1",
        "input_frames_directory": str(frames_dir.resolve()),
        "configuration_path": str(config_path.resolve()),
        "frame_count": len(frame_paths),
        "processed_frame_count": sum("roles" in item for item in frame_results),
        "angle_candidates_by_role": valid_by_role,
        "temporal_stability": temporal_stability,
        "elapsed_seconds": _json_number(elapsed),
        "candidate_semantics": "An angle candidate requires successful fixed-face tracking and radial black-line checks. It is not a meter reading or a Rubric decision.",
        "manual_review_required": True,
        "value_mapping_enabled": False,
        "qwen_called": False,
        "excel_accessed": False,
        "score_computed": False,
        "results": frame_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "fixed_meter_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed teaching-meter OpenCV pointer candidate search")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_fixed_meter_search(
        config_path=args.config.resolve(),
        frames_dir=args.frames_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
