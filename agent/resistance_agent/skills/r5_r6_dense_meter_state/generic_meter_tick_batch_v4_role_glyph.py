"""Generic teaching-meter locator and pivot-free line reader.

The physical meter model is calibrated once. Every input frame is localized
again; pointer detection does not consume a fixed or projected pivot point.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from fixed_meter_opencv import ScanConfig
from pointer_line_arc_hub_v2 import detect_pointer_line_arc_hub
from wire_occlusion_black_edge_v2 import (
    _retained_red_components,
    detect_wire_black_edge_mask,
    line_wire_diagnostics,
)


TIME_PATTERN = re.compile(r"_(\d{5}\.\d{3})s(?:_|\.)")
CANONICAL_CORNERS = np.float32([[0, 0], [639, 0], [639, 519], [0, 519]])
STRICT_HOUGH_MIN_FAR_RADIUS_FRACTION = 0.57
STRICT_RADIAL_MIN_CONTINUITY = 0.10


def parse_time(path: Path) -> float | None:
    match = TIME_PATTERN.search(path.name)
    return float(match.group(1)) if match else None


def nearest_tick(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def directed_delta(target: float, start: float, direction: str) -> float:
    if direction == "increasing":
        return (target - start + 360.0) % 360.0
    if direction == "decreasing":
        return (start - target + 360.0) % 360.0
    raise ValueError("direction must be increasing or decreasing")


def angle_to_ratio(angle: float, zero: float, full: float, direction: str) -> float:
    sweep = directed_delta(full, zero, direction)
    if sweep <= 1e-6:
        raise ValueError("zero/full calibration has no positive sweep")
    return directed_delta(angle, zero, direction) / sweep


def _point(value: np.ndarray | list[float]) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(2)
    return [round(float(array[0]), 3), round(float(array[1]), 3)]


def _quad_iou(first: np.ndarray, second: np.ndarray) -> float:
    area_a = abs(float(cv2.contourArea(first.astype(np.float32))))
    area_b = abs(float(cv2.contourArea(second.astype(np.float32))))
    intersection, _ = cv2.intersectConvexConvex(first.astype(np.float32), second.astype(np.float32))
    union = area_a + area_b - float(intersection)
    return 0.0 if union <= 0 else float(intersection) / union


def _edge_ratio(quad: np.ndarray) -> float:
    lengths = [float(np.linalg.norm(quad[(index + 1) % 4] - quad[index])) for index in range(4)]
    return max(lengths) / max(min(lengths), 1e-6)


@dataclass
class FaceTemplate:
    role: str
    image: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray
    zero_angle: float
    full_angle: float
    direction: str
    ranges: list[dict[str, Any]]
    ports: list[dict[str, Any]]


@dataclass
class TerminalTemplate:
    role: str
    image: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray
    quad: np.ndarray
    ports: list[dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_face_templates(config_path: Path, terminal_annotation_dir: Path, sift: cv2.SIFT) -> dict[str, FaceTemplate]:
    config = _load_json(config_path)
    result: dict[str, FaceTemplate] = {}
    destination = CANONICAL_CORNERS.copy()
    for item in config["templates"]:
        reference = Path(item["reference_frame"])
        if not reference.is_absolute():
            reference = (config_path.parent / reference).resolve()
        frame = cv2.imread(str(reference), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(reference)
        source_quad = np.asarray(item["quad"], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source_quad, destination)
        face = cv2.warpPerspective(frame, transform, (640, 520))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        # Suppress the broad moving hub/lead zone without relying on a fixed pivot.
        cv2.rectangle(mask, (205, 310), (435, 519), 0, -1)
        keypoints, descriptors = sift.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) < 30:
            raise ValueError(f"Too few canonical features for {item['role']}")
        geometry = item["geometry_calibration"]
        ranges = [dict(value, port_id=key) for key, value in item["range_port_layout"].items()]
        terminal_payload = _load_json(terminal_annotation_dir / f"{item['role']}_terminal_annotation.json")
        ports = []
        for port in terminal_payload["ports"]:
            canonical_center = port.get("canonical_center")
            source_center = np.asarray(
                canonical_center if canonical_center is not None else port["center"],
                dtype=np.float32,
            )
            source_radius = float(port.get("canonical_radius_px", port["radius_px"]))
            source_points = np.float32([source_center, source_center + [source_radius, 0]]).reshape(-1, 1, 2)
            canonical_points = cv2.perspectiveTransform(source_points, transform).reshape(-1, 2)
            converted = dict(port)
            converted["canonical_center"] = _point(canonical_points[0])
            converted["canonical_radius_px"] = round(float(np.linalg.norm(canonical_points[1] - canonical_points[0])), 3)
            ports.append(converted)
        result[item["role"]] = FaceTemplate(
            role=item["role"],
            image=face,
            keypoints=keypoints,
            descriptors=descriptors,
            zero_angle=float(geometry["zero_angle_deg"]),
            full_angle=float(geometry["full_angle_deg"]),
            direction=str(geometry["sweep_direction"]),
            ranges=ranges,
            ports=ports,
        )
    return result


def build_terminal_templates(annotation_dir: Path, sift: cv2.SIFT) -> dict[str, TerminalTemplate]:
    result: dict[str, TerminalTemplate] = {}
    for role in ("ammeter", "voltmeter"):
        annotation_path = annotation_dir / f"{role}_terminal_annotation.json"
        payload = _load_json(annotation_path)
        configured_image = Path(str(payload["source_filename"]))
        image_path = (
            configured_image
            if configured_image.is_absolute()
            else annotation_dir / configured_image
        )
        if not image_path.is_file():
            image_path = annotation_dir / "reference_frames" / configured_image.name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 20:
            raise ValueError(f"Too few terminal features for {role}")
        origin = np.asarray(payload["crop_origin_px"], dtype=np.float32)
        quad = np.asarray(payload["source_quad"], dtype=np.float32) - origin
        ports = []
        for port in payload["ports"]:
            converted = dict(port)
            converted["center"] = _point(np.asarray(port["center"], dtype=np.float32) - origin)
            ports.append(converted)
        result[role] = TerminalTemplate(role, image, keypoints, descriptors, quad, ports)
    return result


def detect_frame_features(frame: np.ndarray, sift: cv2.SIFT, max_width: int) -> dict[str, Any]:
    scale = min(1.0, max_width / frame.shape[1])
    work = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return {"scale": scale, "keypoints": keypoints, "descriptors": descriptors, "shape": work.shape}


def match_template(
    source_keypoints: list[cv2.KeyPoint],
    source_descriptors: np.ndarray,
    source_quad: np.ndarray,
    target: dict[str, Any],
    frame_shape: tuple[int, ...],
    ratio_test: float = 0.80,
    excluded_quads: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"tracked": False, "reasons": []}
    target_descriptors = target["descriptors"]
    if target_descriptors is None:
        result["reasons"] = ["target_has_no_sift_descriptors"]
        return result
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
    good = []
    for pair in pairs:
        if len(pair) != 2 or pair[0].distance >= ratio_test * pair[1].distance:
            continue
        target_point = np.asarray(target["keypoints"][pair[0].trainIdx].pt, dtype=np.float32) / float(target["scale"])
        if excluded_quads and any(cv2.pointPolygonTest(quad.astype(np.float32), tuple(target_point), False) >= 0 for quad in excluded_quads):
            continue
        good.append(pair[0])
    result["good_matches"] = len(good)
    if len(good) < 10:
        result["reasons"] = ["too_few_feature_matches"]
        return result
    source = np.float32([source_keypoints[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    target_points = np.float32([target["keypoints"][item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    target_points /= float(target["scale"])
    homography, inlier_mask = cv2.findHomography(source, target_points, cv2.RANSAC, 6.0)
    if homography is None or inlier_mask is None or not np.all(np.isfinite(homography)):
        result["reasons"] = ["homography_failed"]
        return result
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(good)
    quad = cv2.perspectiveTransform(source_quad.reshape(-1, 1, 2), homography).reshape(-1, 2)
    area_ratio = abs(float(cv2.contourArea(quad.astype(np.float32)))) / (frame_shape[0] * frame_shape[1])
    reasons: list[str] = []
    if inliers < 8:
        reasons.append("too_few_homography_inliers")
    if inlier_ratio < 0.28:
        reasons.append("low_homography_inlier_ratio")
    if not cv2.isContourConvex(quad.astype(np.float32)):
        reasons.append("projected_quad_not_convex")
    if not 0.001 <= area_ratio <= 0.18:
        reasons.append("projected_quad_area_out_of_range")
    if _edge_ratio(quad) > 4.2:
        reasons.append("projected_quad_shape_implausible")
    height, width = frame_shape[:2]
    if np.any(quad[:, 0] < -3) or np.any(quad[:, 0] >= width + 3) or np.any(quad[:, 1] < -3) or np.any(quad[:, 1] >= height + 3):
        reasons.append("projected_quad_outside_frame")
    result.update(
        {
            "tracked": not reasons,
            "reasons": reasons,
            "inliers": inliers,
            "inlier_ratio": round(inlier_ratio, 6),
            "match_strength": round(inliers * inlier_ratio, 6),
            "homography": homography,
            "quad": quad,
            "quad_area_ratio": round(area_ratio, 6),
        }
    )
    return result


def _edge_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    region = (slice(105, 390), slice(145, 500))
    first_edge = cv2.Canny(first_gray[region], 45, 130).astype(np.float32)
    second_edge = cv2.Canny(second_gray[region], 45, 130).astype(np.float32)
    first_edge -= first_edge.mean()
    second_edge -= second_edge.mean()
    denominator = float(np.linalg.norm(first_edge) * np.linalg.norm(second_edge))
    return 0.0 if denominator <= 1e-6 else float(np.sum(first_edge * second_edge) / denominator)


ROLE_GLYPH_CROPS = {
    "ammeter": (270, 170, 390, 310),
    "voltmeter": (270, 175, 390, 320),
}


def _glyph_preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _scaled_template_score(search_image: np.ndarray, template: np.ndarray) -> float:
    best = -1.0
    for scale in np.linspace(0.55, 1.35, 17):
        resized = cv2.resize(template, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_AREA)
        if resized.shape[0] >= search_image.shape[0] or resized.shape[1] >= search_image.shape[1]:
            continue
        response = cv2.matchTemplate(search_image, resized, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(cv2.minMaxLoc(response)[1]))
    return best


def _role_glyph_diagnostics(
    face: np.ndarray,
    requested_role: str,
    templates: dict[str, FaceTemplate],
) -> dict[str, Any]:
    """Use the fixed A/V face glyph only to reject a confident role mismatch."""
    search = _glyph_preprocess(face[100:400, 170:490])
    scores: dict[str, float] = {}
    for role, template in templates.items():
        x1, y1, x2, y2 = ROLE_GLYPH_CROPS[role]
        glyph = _glyph_preprocess(template.image[y1:y2, x1:x2])
        scores[role] = _scaled_template_score(search, glyph)
    other_role = "voltmeter" if requested_role == "ammeter" else "ammeter"
    requested_score = scores[requested_role]
    other_score = scores[other_role]
    margin = requested_score - other_score
    confident_other = other_score >= 0.35 and margin <= -0.03
    state = "supports_other_role" if confident_other else "supports_requested_role" if requested_score >= 0.40 and margin >= 0.08 else "inconclusive"
    return {
        "state": state,
        "requested_role": requested_role,
        "scores": {key: round(value, 6) for key, value in scores.items()},
        "requested_minus_other_margin": round(margin, 6),
        "confident_other_role": confident_other,
        "method": "canonical_center_glyph_scaled_template_match",
    }


def _source_panel_identity(frame: np.ndarray, quad: np.ndarray) -> dict[str, float]:
    top_center = (quad[0] + quad[1]) * 0.5
    bottom_center = (quad[2] + quad[3]) * 0.5
    down = bottom_center - top_center
    width_vector = quad[2] - quad[3]
    polygon = np.float32(
        [
            quad[3] - 0.10 * width_vector,
            quad[2] + 0.10 * width_vector,
            quad[2] + 0.82 * down + 0.10 * width_vector,
            quad[3] + 0.82 * down - 0.10 * width_vector,
        ]
    )
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.bitwise_and(cv2.inRange(hsv, np.array([34, 50, 25]), np.array([100, 255, 210])), mask)
    red = cv2.bitwise_and(
        cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 65, 45]), np.array([16, 255, 255])),
            cv2.inRange(hsv, np.array([168, 55, 40]), np.array([179, 255, 255])),
        ),
        mask,
    )
    area = max(1, int(np.count_nonzero(mask)))
    components = cv2.connectedComponentsWithStats((green > 0).astype(np.uint8), 8)[2]
    largest_green = int(components[1:, cv2.CC_STAT_AREA].max()) if len(components) > 1 else 0
    return {
        "green_fraction": float(np.count_nonzero(green)) / area,
        "red_fraction": float(np.count_nonzero(red)) / area,
        "largest_green_component_fraction": largest_green / area,
    }


def _candidate_diagnostics(
    frame: np.ndarray,
    quad: np.ndarray,
    face: np.ndarray,
    extended: np.ndarray,
    role: str,
    templates: dict[str, FaceTemplate],
) -> dict[str, Any]:
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    light_ratio = float(np.mean(gray >= 125))
    edge_density = float(np.mean(cv2.Canny(gray, 50, 140) > 0))
    similarities = {name: round(_edge_correlation(face, template.image), 6) for name, template in templates.items()}
    other_role = "voltmeter" if role == "ammeter" else "ammeter"
    role_margin = similarities[role] - similarities[other_role]
    panel = _face_color_identity(extended[500:, :])
    source_panel = _source_panel_identity(frame, quad)
    role_glyph = _role_glyph_diagnostics(face, role, templates)
    panel_color_available = source_panel["green_fraction"] + source_panel["red_fraction"] >= 0.015
    if panel_color_available and role == "ammeter":
        panel_role_supported = source_panel["largest_green_component_fraction"] >= 0.018
    elif panel_color_available:
        panel_role_supported = source_panel["largest_green_component_fraction"] < 0.012 and source_panel["red_fraction"] >= 0.025
    else:
        panel_role_supported = role_margin >= 0.015
    return {
        "light_ratio": round(light_ratio, 6),
        "edge_density": round(edge_density, 6),
        "role_similarities": similarities,
        "role_margin": round(role_margin, 6),
        "panel_color": {key: round(value, 6) for key, value in panel.items()},
        "source_panel_color": {key: round(value, 6) for key, value in source_panel.items()},
        "panel_color_available": panel_color_available,
        "panel_role_supported": panel_role_supported,
        "role_glyph": role_glyph,
        "dial_like": light_ratio >= 0.40 and 0.015 <= edge_density <= 0.30,
    }


def locate_role_face(
    frame: np.ndarray,
    role: str,
    templates: dict[str, FaceTemplate],
    target: dict[str, Any],
    max_candidates: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for source_role, source_template in templates.items():
        exclusions: list[np.ndarray] = []
        for _ in range(max_candidates):
            candidate = match_template(
                source_template.keypoints,
                source_template.descriptors,
                CANONICAL_CORNERS,
                target,
                frame.shape,
                excluded_quads=exclusions,
            )
            if candidate.get("homography") is None or candidate.get("quad") is None:
                break
            candidate["source_template_role"] = source_role
            try:
                inverse_homography = np.linalg.inv(candidate["homography"])
            except np.linalg.LinAlgError:
                candidate["tracked"] = False
                candidate.setdefault("reasons", []).append("singular_homography")
                candidate["selection_score"] = -1e9
                candidates.append(candidate)
                break
            face = cv2.warpPerspective(frame, inverse_homography, (640, 520))
            extended = cv2.warpPerspective(frame, inverse_homography, (640, 780))
            diagnostics = _candidate_diagnostics(frame, candidate["quad"], face, extended, role, templates)
            candidate["candidate_diagnostics"] = diagnostics
            hard_reasons = [reason for reason in candidate.get("reasons", []) if reason != "low_homography_inlier_ratio"]
            soft_geometry = not hard_reasons and int(candidate.get("inliers", 0)) >= 9
            role_supported = bool(diagnostics["panel_role_supported"]) and not bool(
                diagnostics["role_glyph"]["confident_other_role"]
            )
            candidate["tracked"] = bool(soft_geometry and diagnostics["dial_like"] and role_supported)
            if candidate["tracked"]:
                candidate["reasons"] = []
            elif diagnostics["role_glyph"]["confident_other_role"]:
                candidate.setdefault("reasons", []).append("center_glyph_supports_other_meter_role")
            elif not role_supported:
                candidate.setdefault("reasons", []).append("rectified_center_supports_other_meter_role")
            candidate["selection_score"] = round(
                2.5 * diagnostics["role_margin"]
                + 0.8 * diagnostics["role_glyph"]["requested_minus_other_margin"]
                + 0.025 * float(candidate.get("match_strength", 0.0))
                + 0.25 * diagnostics["light_ratio"],
                6,
            )
            candidates.append(candidate)
            quad = candidate["quad"]
            if cv2.isContourConvex(quad.astype(np.float32)) and 0.0005 <= float(candidate.get("quad_area_ratio", 0.0)) <= 0.25:
                exclusions.append(quad)
            else:
                break
    valid = [candidate for candidate in candidates if candidate.get("tracked")]
    selected = max(valid, key=lambda item: item["selection_score"]) if valid else max(candidates, key=lambda item: item["selection_score"], default={"tracked": False, "reasons": ["no_face_candidate"]})
    return selected, candidates


def _face_color_identity(face: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    lower = hsv[int(face.shape[0] * 0.66) :, :]
    green = cv2.inRange(lower, np.array([34, 50, 25]), np.array([100, 255, 210]))
    red_a = cv2.inRange(lower, np.array([0, 65, 45]), np.array([16, 255, 255]))
    red_b = cv2.inRange(lower, np.array([168, 55, 40]), np.array([179, 255, 255]))
    return {"green_fraction": float(np.mean(green > 0)), "red_fraction": float(np.mean(cv2.bitwise_or(red_a, red_b) > 0))}


def rectify_face(frame: np.ndarray, track: dict[str, Any]) -> np.ndarray:
    inverse = np.linalg.inv(track["homography"])
    return cv2.warpPerspective(frame, inverse, (640, 520))


def _port_occupancy(frame: np.ndarray, center: np.ndarray, radius: float) -> dict[str, float]:
    x, y = np.round(center).astype(int)
    radius_i = max(10, int(round(radius * 1.6)))
    x1, x2 = max(0, x - radius_i), min(frame.shape[1], x + radius_i + 1)
    y1, y2 = max(0, y - radius_i * 2), min(frame.shape[0], y + radius_i + 1)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {"score": 0.0, "dark_fraction": 0.0, "edge_density": 0.0, "red_fraction": 0.0}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 70, 45]), np.array([16, 255, 255])),
        cv2.inRange(hsv, np.array([168, 60, 40]), np.array([179, 255, 255])),
    )
    dark = float(np.mean(gray < 82))
    edges = float(np.mean(cv2.Canny(gray, 60, 160) > 0))
    red_fraction = float(np.mean(red > 0))
    score = 0.70 * dark + 0.22 * edges + 0.08 * min(red_fraction, 0.30)
    return {"score": round(score, 6), "dark_fraction": round(dark, 6), "edge_density": round(edges, 6), "red_fraction": round(red_fraction, 6)}


def infer_range(
    frame: np.ndarray,
    terminal: TerminalTemplate,
    terminal_track: dict[str, Any],
) -> dict[str, Any]:
    if not terminal_track.get("tracked"):
        return {"status": "not_localized", "range_max_value": None, "reason": terminal_track.get("reasons", ["terminal_not_localized"])[0]}
    homography = terminal_track["homography"]
    observations = []
    for port in terminal.ports:
        center = np.asarray(port["center"], dtype=np.float32)
        radius = float(port["radius_px"])
        points = np.float32([center, center + [radius, 0]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
        projected_radius = float(np.linalg.norm(projected[1] - projected[0]))
        occupancy = _port_occupancy(frame, projected[0], projected_radius)
        observations.append({**port, "projected_center": _point(projected[0]), "projected_radius_px": round(projected_radius, 3), "occupancy": occupancy})
    ranged = [item for item in observations if item.get("range_max_value") is not None]
    ranged.sort(key=lambda item: float(item["range_max_value"]))
    if len(ranged) != 2:
        return {"status": "invalid_calibration", "range_max_value": None, "ports": observations}
    difference = ranged[0]["occupancy"]["score"] - ranged[1]["occupancy"]["score"]
    selected = ranged[0] if difference >= 0.018 else ranged[1] if difference <= -0.018 else None
    return {
        "status": "selected" if selected else "ambiguous",
        "range_max_value": float(selected["range_max_value"]) if selected else None,
        "unit": selected.get("unit") if selected else ranged[0].get("unit"),
        "selected_port": selected.get("id") if selected else None,
        "score_difference_low_minus_high": round(difference, 6),
        "ports": observations,
        "reason": None if selected else "middle_and_right_port_occupancy_scores_too_close",
    }


def infer_range_from_face_track(frame: np.ndarray, template: FaceTemplate, face_track: dict[str, Any]) -> dict[str, Any]:
    if not face_track.get("tracked"):
        return {"status": "not_localized", "range_max_value": None, "reason": "face_not_localized"}
    homography = face_track["homography"]
    observations = []
    for port in template.ports:
        center = np.asarray(port["canonical_center"], dtype=np.float32)
        radius = float(port["canonical_radius_px"])
        points = np.float32([center, center + [radius, 0]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
        projected_radius = float(np.linalg.norm(projected[1] - projected[0]))
        occupancy = _port_occupancy(frame, projected[0], projected_radius)
        observations.append(
            {
                **port,
                "projected_center": _point(projected[0]),
                "projected_radius_px": round(projected_radius, 3),
                "occupancy": occupancy,
            }
        )
    ranged = sorted(
        [item for item in observations if item.get("range_max_value") is not None],
        key=lambda item: float(item["range_max_value"]),
    )
    if len(ranged) != 2:
        return {"status": "invalid_calibration", "range_max_value": None, "ports": observations}
    difference = ranged[0]["occupancy"]["score"] - ranged[1]["occupancy"]["score"]
    selected = ranged[0] if difference >= 0.018 else ranged[1] if difference <= -0.018 else None
    return {
        "status": "selected" if selected else "ambiguous",
        "range_max_value": float(selected["range_max_value"]) if selected else None,
        "unit": selected.get("unit") if selected else ranged[0].get("unit"),
        "selected_port": selected.get("id") if selected else None,
        "score_difference_low_minus_high": round(difference, 6),
        "ports": observations,
        "reason": None if selected else "middle_and_right_port_occupancy_scores_too_close",
        "projection_source": "face_homography_and_one_time_terminal_layout",
    }


def draw_tick_face(face: np.ndarray, template: FaceTemplate, pointer: dict[str, Any], raw_tick: float | None, tick: int | None) -> np.ndarray:
    image = face.copy()
    color = (0, 185, 0) if pointer.get("detected") else (0, 165, 255)
    if pointer.get("line") is not None:
        x1, y1, x2, y2 = pointer["line"]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 4, cv2.LINE_AA)
    label = f"raw_tick={raw_tick:.2f} nearest={tick}" if raw_tick is not None else "pointer unavailable"
    cv2.rectangle(image, (0, 0), (640, 38), (255, 255, 255), -1)
    cv2.putText(image, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    return image


def process_observation(
    frame_path: Path,
    expected_role: str,
    face_templates: dict[str, FaceTemplate],
    terminal_templates: dict[str, TerminalTemplate],
    sift: cv2.SIFT,
    scan_config: ScanConfig,
    output_dir: Path,
    max_feature_width: int,
) -> dict[str, Any]:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(frame_path)
    features = detect_frame_features(frame, sift, max_feature_width)
    face_tracks: dict[str, dict[str, Any]] = {}
    face_candidates: dict[str, list[dict[str, Any]]] = {}
    for role, template in face_templates.items():
        face_tracks[role], face_candidates[role] = locate_role_face(frame, role, face_templates, features)
        if face_tracks[role].get("tracked"):
            candidate_face = rectify_face(frame, face_tracks[role])
            face_tracks[role]["identity_color"] = _face_color_identity(candidate_face)
    first, second = face_tracks.get("ammeter", {}), face_tracks.get("voltmeter", {})
    if first.get("tracked") and second.get("tracked") and _quad_iou(first["quad"], second["quad"]) > 0.45:
        green = first.get("identity_color", {}).get("green_fraction", 0.0)
        if green >= 0.015:
            second["tracked"] = False
            second["reasons"].append("same_face_green_panel_supports_ammeter")
        else:
            weaker = first if first["match_strength"] < second["match_strength"] else second
            weaker["tracked"] = False
            weaker["reasons"].append("same_face_role_collision_weaker_match")

    template = face_templates[expected_role]
    track = face_tracks[expected_role]
    terminal_template = terminal_templates[expected_role]
    terminal_track = match_template(
        terminal_template.keypoints,
        terminal_template.descriptors,
        terminal_template.quad,
        features,
        frame.shape,
        ratio_test=0.82,
    )
    role_dir = output_dir / expected_role
    face_dir = role_dir / "faces"
    overlay_dir = role_dir / "overlays"
    terminal_dir = role_dir / "terminal_overlays"
    for directory in (face_dir, overlay_dir, terminal_dir):
        directory.mkdir(parents=True, exist_ok=True)
    timestamp = parse_time(frame_path)
    stem = frame_path.stem
    pointer: dict[str, Any] = {"detected": False, "angle_deg": None, "anchor": None, "tip": None, "reasons": ["face_not_localized"]}
    raw_tick = None
    tick = None
    face_path = None
    face_overlay_path = None
    annotated = frame.copy()
    if track.get("tracked"):
        face = rectify_face(frame, track)
        wire_mask, wire_stats = detect_wire_black_edge_mask(face)
        pointer = detect_pointer_line_arc_hub(
            face,
            wire_mask,
            template.zero_angle,
            template.full_angle,
        )
        red_core, wire_components = _retained_red_components(face)
        pointer.update(line_wire_diagnostics(pointer.get("line"), wire_mask, red_core, wire_components))
        pointer["wire_mask_fraction"] = wire_stats.get("joint_mask_fraction")
        if pointer.get("angle_deg") is not None:
            ratio = angle_to_ratio(float(pointer["angle_deg"]), template.zero_angle, template.full_angle, template.direction)
            raw_tick = ratio * 30.0
            if -1.5 <= raw_tick <= 31.5:
                tick = min(30, max(0, nearest_tick(raw_tick)))
        face_path = face_dir / f"{stem}_{expected_role}_rectified.jpg"
        face_overlay_path = face_dir / f"{stem}_{expected_role}_ticks.jpg"
        cv2.imwrite(str(face_path), face, [cv2.IMWRITE_JPEG_QUALITY, 97])
        cv2.imwrite(str(face_overlay_path), draw_tick_face(face, template, pointer, raw_tick, tick), [cv2.IMWRITE_JPEG_QUALITY, 97])
        quad_i = np.round(track["quad"]).astype(np.int32)
        cv2.polylines(annotated, [quad_i], True, (40, 190, 40), 4, cv2.LINE_AA)
        if pointer.get("line") is not None:
            x1, y1, x2, y2 = pointer["line"]
            projected = cv2.perspectiveTransform(
                np.float32([[x1, y1], [x2, y2]]).reshape(-1, 1, 2), track["homography"]
            ).reshape(-1, 2)
            cv2.line(annotated, tuple(np.round(projected[0]).astype(int)), tuple(np.round(projected[1]).astype(int)), (0, 190, 0) if pointer.get("detected") else (0, 165, 255), 5, cv2.LINE_AA)
    else:
        for other in face_tracks.values():
            if other.get("quad") is not None:
                cv2.polylines(annotated, [np.round(other["quad"]).astype(np.int32)], True, (0, 165, 255), 3, cv2.LINE_AA)

    range_result = infer_range_from_face_track(frame, template, track)
    if range_result.get("status") == "not_localized":
        range_result = infer_range(frame, terminal_template, terminal_track)
    if range_result.get("ports"):
        for port in range_result["ports"]:
            center = tuple(np.round(port["projected_center"]).astype(int))
            cv2.circle(annotated, center, max(8, int(round(port["projected_radius_px"]))), (255, 180, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, str(port["id"]), (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    elif terminal_track.get("tracked"):
        cv2.polylines(annotated, [np.round(terminal_track["quad"]).astype(np.int32)], True, (255, 180, 0), 3, cv2.LINE_AA)
    range_max = range_result.get("range_max_value")
    reading = tick * float(range_max) / 30.0 if tick is not None and range_max is not None and pointer.get("detected") else None
    readings_by_range = [
        {"port_id": item["port_id"], "range_max_value": item["range_max_value"], "unit": item["unit"], "reading": round(tick * float(item["range_max_value"]) / 30.0, 6) if tick is not None else None}
        for item in template.ranges
    ]
    label = f"{expected_role} tick={tick} range={range_max} reading={reading}"
    cv2.rectangle(annotated, (0, 0), (min(frame.shape[1], 1250), 52), (255, 255, 255), -1)
    cv2.putText(annotated, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    overlay_path = overlay_dir / f"{stem}_{expected_role}_overlay.jpg"
    terminal_overlay_path = terminal_dir / f"{stem}_{expected_role}_terminal.jpg"
    cv2.imwrite(str(overlay_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 94])
    cv2.imwrite(str(terminal_overlay_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return {
        "frame": str(frame_path.resolve()),
        "timestamp_seconds": timestamp,
        "expected_role": expected_role,
        "face_localized": bool(track.get("tracked")),
        "face_track": {key: value for key, value in track.items() if key not in {"homography", "quad"}},
        "face_candidate_count": len(face_candidates.get(expected_role, [])),
        "face_quad": [_point(value) for value in track["quad"]] if track.get("quad") is not None else None,
        "rectified_face_path": str(face_path.resolve()) if face_path else None,
        "face_tick_overlay_path": str(face_overlay_path.resolve()) if face_overlay_path else None,
        "full_overlay_path": str(overlay_path.resolve()),
        "terminal_overlay_path": str(terminal_overlay_path.resolve()),
        "pointer": {
            "detected": bool(pointer.get("detected")),
            "angle_deg": pointer.get("angle_deg"),
            "confidence": pointer.get("score"),
            "state": pointer.get("state"),
            "reasons": pointer.get("reasons", []),
            "dark_fraction": pointer.get("dark_fraction"),
            "continuity": pointer.get("continuity"),
            "peak_margin": pointer.get("peak_margin"),
            "localization_method": pointer.get("localization_method"),
            "fixed_pivot_used": False,
            "anchor": pointer.get("anchor"),
            "line": pointer.get("line"),
            "tip": _point(pointer["tip"]) if pointer.get("tip") is not None else None,
        },
        "tick": {"total_divisions": 30, "raw_tick_index": round(raw_tick, 6) if raw_tick is not None else None, "nearest_tick_index": tick},
        "range": range_result,
        "readings_by_range": readings_by_range,
        "reading": round(reading, 6) if reading is not None else None,
        "reading_unit": "A" if expected_role == "ammeter" else "V",
        "reading_status": "automatic_tick_and_range_candidate" if reading is not None else "candidate_incomplete",
        "qwen_called": False,
        "excel_accessed": False,
        "score_computed": False,
    }


def summarize_role(observations: list[dict[str, Any]], role: str) -> dict[str, Any]:
    strict = [item for item in observations if item.get("pointer", {}).get("detected") and item.get("tick", {}).get("nearest_tick_index") is not None]
    soft = [
        item
        for item in observations
        if item.get("face_localized")
        and item.get("tick", {}).get("nearest_tick_index") is not None
        and set(item.get("pointer", {}).get("reasons", [])) == {"multiple_pointer_lines_ambiguous"}
    ]
    soft_ticks = [int(item["tick"]["nearest_tick_index"]) for item in soft]
    temporal_soft_supported = len(soft_ticks) >= 2 and max(soft_ticks) - min(soft_ticks) <= 2
    valid = strict
    ticks = [int(item["tick"]["nearest_tick_index"]) for item in valid]
    sorted_ticks = sorted(ticks)
    clusters: list[list[int]] = []
    for start in range(len(sorted_ticks)):
        cluster = [value for value in sorted_ticks[start:] if value - sorted_ticks[start] <= 3]
        clusters.append(cluster)
    dominant_ticks = max(clusters, key=lambda values: (len(values), -max(values) + min(values)), default=[])
    dominant_is_majority = bool(dominant_ticks) and (
        len(dominant_ticks) == len(ticks) or len(dominant_ticks) >= 2 and len(dominant_ticks) > len(ticks) / 2.0
    )
    tick_consistent = dominant_is_majority
    median_tick = nearest_tick(float(median(dominant_ticks))) if dominant_is_majority else None
    remaining_consensus = dominant_ticks.copy() if dominant_is_majority else []
    tick_outliers = []
    for value in ticks:
        if value in remaining_consensus:
            remaining_consensus.remove(value)
        else:
            tick_outliers.append(value)
    selected_ranges = [
        float(item["range"]["range_max_value"])
        for item in observations
        if item.get("face_localized") and item.get("range", {}).get("range_max_value") is not None
    ]
    range_counts = {value: selected_ranges.count(value) for value in set(selected_ranges)}
    range_max = None
    if range_counts:
        ranked_ranges = sorted(range_counts.items(), key=lambda item: (item[1], -item[0]), reverse=True)
        if len(ranked_ranges) == 1 or ranked_ranges[0][1] > ranked_ranges[1][1]:
            range_max = float(ranked_ranges[0][0])
    reading = median_tick * range_max / 30.0 if median_tick is not None and range_max is not None else None
    return {
        "role": role,
        "observation_count": len(observations),
        "strict_pointer_count": len(strict),
        "temporal_soft_pointer_count_diagnostic_only": len(soft) if temporal_soft_supported else 0,
        "valid_pointer_count": len(valid),
        "pointer_validation_source": "pivot_free_long_dark_line" if strict else None,
        "tick_candidates": ticks,
        "tick_consensus_candidates": dominant_ticks,
        "tick_outliers_rejected": sorted(tick_outliers),
        "tick_consistent_within_3_divisions": tick_consistent,
        "median_tick_index": median_tick,
        "range_candidates": selected_ranges,
        "range_consistent": range_max is not None,
        "range_max_value": range_max,
        "reading": round(reading, 6) if reading is not None else None,
        "unit": "A" if role == "ammeter" else "V",
        "status": "reading_candidate" if reading is not None else "candidate_incomplete",
        "reason": None if reading is not None else "pointer_ticks_inconsistent" if ticks and not tick_consistent else "range_candidates_inconsistent_or_missing" if median_tick is not None else "no_valid_pointer",
        "evidence_quality": "multi_frame" if len(valid) >= 2 else "single_frame" if len(valid) == 1 else "no_valid_pointer",
    }


def make_visual_review_sheet(video_results: list[dict[str, Any]], output_dir: Path) -> Path:
    cell_width, cell_height = 640, 560
    canvas = np.full((cell_height * len(video_results), cell_width * 2, 3), 245, dtype=np.uint8)
    for row, video in enumerate(video_results):
        for column, role in enumerate(("ammeter", "voltmeter")):
            role_payload = video["roles"][role]
            observations = role_payload["observations"]
            candidates = [item for item in observations if item.get("face_tick_overlay_path")]
            candidates.sort(
                key=lambda item: (
                    bool(item.get("pointer", {}).get("detected")),
                    item.get("tick", {}).get("nearest_tick_index") is not None,
                ),
                reverse=True,
            )
            image_path = Path(candidates[0]["face_tick_overlay_path"]) if candidates else None
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path else None
            if image is None:
                image = np.full((520, 640, 3), 232, dtype=np.uint8)
                cv2.putText(image, "NO RECTIFIED FACE", (120, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
            image = cv2.resize(image, (cell_width, 520), interpolation=cv2.INTER_AREA)
            y, x = row * cell_height, column * cell_width
            canvas[y + 40 : y + cell_height, x : x + cell_width] = image
            summary = role_payload["summary"]
            label = f"{video['video_id']} {role} {summary['status']} reading={summary['reading']} {summary['unit']}"
            cv2.putText(canvas, label, (x + 8, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    path = output_dir / "visual_review_contact_sheet.jpg"
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise OSError(path)
    return path


def run_batch(manifest_path: Path, config_path: Path, terminal_annotation_dir: Path, output_dir: Path, max_feature_width: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    manifest = _load_json(manifest_path)
    sift = cv2.SIFT_create(nfeatures=14000, contrastThreshold=0.012, edgeThreshold=14)
    face_templates = build_face_templates(config_path, terminal_annotation_dir, sift)
    terminal_templates = build_terminal_templates(terminal_annotation_dir, sift)
    scan_config = ScanConfig()
    started = perf_counter()
    video_results = []
    for video in manifest["videos"]:
        video_id = str(video["video_id"])
        frames_dir = Path(video["frames_dir"])
        if not frames_dir.is_absolute():
            frames_dir = (manifest_path.parent / frames_dir).resolve()
        observations_by_role: dict[str, list[dict[str, Any]]] = {"ammeter": [], "voltmeter": []}
        for role, times in video["role_times_seconds"].items():
            for timestamp in times:
                token = f"_{float(timestamp):09.3f}s_"
                matches = sorted(path for path in frames_dir.glob("*.jpg") if token in path.name)
                if len(matches) != 1:
                    observations_by_role[role].append({"timestamp_seconds": timestamp, "expected_role": role, "reading_status": "input_frame_not_unique", "match_count": len(matches), "qwen_called": False, "excel_accessed": False, "score_computed": False})
                    continue
                observation = process_observation(matches[0], role, face_templates, terminal_templates, sift, scan_config, output_dir / video_id, max_feature_width)
                observations_by_role[role].append(observation)
        video_results.append(
            {
                "video_id": video_id,
                "closed_event_id": video.get("closed_event_id"),
                "roles": {
                    role: {"summary": summarize_role(items, role), "observations": items}
                    for role, items in observations_by_role.items()
                },
            }
        )
    contact_sheet_path = make_visual_review_sheet(video_results, output_dir)
    result = {
        "schema_version": "generic-fixed-meter-tick-batch-v4-role-glyph",
        "method": "one_time_model_template_then_per_frame_sift_homography_pivot_free_long_line_30_tick_mapping",
        "fixed_pivot_used": False,
        "manifest": str(manifest_path.resolve()),
        "calibration": str(config_path.resolve()),
        "terminal_calibration_directory": str(terminal_annotation_dir.resolve()),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "video_count": len(video_results),
        "videos": video_results,
        "visual_review_contact_sheet": str(contact_sheet_path.resolve()),
        "no_video_specific_meter_geometry": True,
        "qwen_called": False,
        "excel_accessed": False,
        "score_computed": False,
    }
    result_path = output_dir / "generic_meter_tick_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic fixed-model teaching-meter tick batch reader")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--terminal-annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-feature-width", type=int, default=2400)
    args = parser.parse_args()
    result = run_batch(args.manifest.resolve(), args.calibration.resolve(), args.terminal_annotations.resolve(), args.output_dir.resolve(), args.max_feature_width)
    compact = {
        "output": str((args.output_dir.resolve() / "generic_meter_tick_results.json")),
        "elapsed_seconds": result["elapsed_seconds"],
        "videos": [
            {"video_id": item["video_id"], "ammeter": item["roles"]["ammeter"]["summary"], "voltmeter": item["roles"]["voltmeter"]["summary"]}
            for item in result["videos"]
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
