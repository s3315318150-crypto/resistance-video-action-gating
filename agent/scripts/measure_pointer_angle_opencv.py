#!/usr/bin/env python3
"""Measure an analog meter pointer angle with OpenCV only.

This module is deliberately a geometry measurement layer.  It does not call a
model, read Excel, assign a Rubric score, or infer a meter's semantic identity.
The safest mode supplies a calibration JSON containing the dial center, face
radius, and zero/full-scale directions.  Without that calibration the script
may estimate a face circle, but it reports an uncalibrated/uncertain state
instead of inventing a normalized deflection.

Examples:
    python scripts/measure_pointer_angle_opencv.py `
        --image meter.jpg `
        --calibration meter_calibration.json `
        --output measurement.json `
        --debug-dir debug

    python scripts/measure_pointer_angle_opencv.py `
        --image frame_001.jpg --image frame_002.jpg `
        --calibration meter_calibration.json `
        --output sequence_measurement.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


SCHEMA_VERSION = "1.0"
ARTIFACT_TYPE = "opencv_pointer_angle_measurement"


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def wrap_angle_deg(value: float) -> float:
    """Return an angle in [-180, 180)."""

    return (float(value) + 180.0) % 360.0 - 180.0


def angle_deg(center: tuple[float, float], point: tuple[float, float]) -> float:
    """Use mathematical y-up angles, so calibration is resolution independent."""

    dx = float(point[0]) - float(center[0])
    dy = -(float(point[1]) - float(center[1]))
    return math.degrees(math.atan2(dy, dx))


def directed_delta_deg(start: float, end: float, direction: str) -> float:
    delta = wrap_angle_deg(float(end) - float(start))
    return delta if direction == "ccw" else -delta


def angle_in_calibrated_arc(
    angle: float,
    geometry: dict[str, Any],
    margin_fraction: float = 0.12,
) -> bool:
    """Check whether an angle lies on the calibrated zero-to-full-scale arc."""

    zero = geometry.get("zero_angle_deg")
    full = geometry.get("full_scale_angle_deg")
    if zero is None or full is None:
        return True
    direction = str(geometry.get("direction", "cw"))
    arc = directed_delta_deg(float(zero), float(full), direction)
    if abs(arc) < 20.0:
        return False
    position = directed_delta_deg(float(zero), float(angle), direction)
    margin = abs(arc) * float(margin_fraction)
    return -margin <= position <= abs(arc) + margin


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def resize_calibrated_geometry(
    calibration: dict[str, Any], width: int, height: int
) -> dict[str, Any]:
    """Resolve absolute or normalized calibration coordinates for this image."""

    center_value = calibration.get("center")
    if (
        isinstance(center_value, (list, tuple))
        and len(center_value) == 2
        and all(finite_number(item) for item in center_value)
    ):
        center = (float(center_value[0]), float(center_value[1]))
        center_source = "absolute"
    else:
        normalized = calibration.get("center_normalized")
        if (
            isinstance(normalized, (list, tuple))
            and len(normalized) == 2
            and all(finite_number(item) for item in normalized)
        ):
            center = (float(normalized[0]) * width, float(normalized[1]) * height)
            center_source = "normalized"
        else:
            raise ValueError("calibration_center_missing")

    radius: float | None = None
    radius_source = ""
    if finite_number(calibration.get("face_radius")):
        radius = float(calibration["face_radius"])
        radius_source = "absolute"
    elif finite_number(calibration.get("face_radius_ratio")):
        radius = float(calibration["face_radius_ratio"]) * min(width, height)
        radius_source = "ratio"
    if radius is None or radius <= 0:
        raise ValueError("calibration_face_radius_missing")

    direction = str(calibration.get("direction", "cw")).lower().strip()
    if direction not in {"cw", "ccw"}:
        raise ValueError("calibration_direction_invalid")

    zero_angle = calibration.get("zero_angle_deg")
    full_angle = calibration.get("full_scale_angle_deg")
    if not finite_number(zero_angle) or not finite_number(full_angle):
        zero_angle = None
        full_angle = None

    return {
        "center": [round(center[0], 4), round(center[1], 4)],
        "face_radius": round(radius, 4),
        "center_source": center_source,
        "radius_source": radius_source,
        "zero_angle_deg": None if zero_angle is None else float(zero_angle),
        "full_scale_angle_deg": None if full_angle is None else float(full_angle),
        "direction": direction,
        "calibration_complete": zero_angle is not None and full_angle is not None,
    }


def face_edge_support(
    edges: np.ndarray, center: tuple[float, float], radius: float
) -> float:
    angles = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    x = np.rint(center[0] + np.cos(angles) * radius).astype(np.int32)
    y = np.rint(center[1] - np.sin(angles) * radius).astype(np.int32)
    valid = (x >= 0) & (x < edges.shape[1]) & (y >= 0) & (y < edges.shape[0])
    return float(np.mean(edges[y[valid], x[valid]] > 0)) if np.any(valid) else 0.0


def estimate_face_geometry(image: np.ndarray) -> dict[str, Any]:
    """Conservatively estimate a dial circle for debug/triage purposes.

    The estimate is not enough to establish zero/full-scale calibration.  It is
    intentionally reported with a confidence and can be rejected by callers.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.medianBlur(enhanced, 5)
    edges = cv2.Canny(enhanced, 60, 160)
    height, width = gray.shape[:2]
    short = float(min(width, height))
    min_radius = max(24, int(round(short * 0.28)))
    max_radius = max(min_radius + 4, int(round(short * 0.50)))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(30, int(round(short * 0.22))),
        param1=100,
        param2=26,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return {
            "available": False,
            "confidence": 0.0,
            "reason": "face_circle_not_found",
            "candidates": [],
        }

    image_center = (width / 2.0, height / 2.0)
    candidates: list[dict[str, Any]] = []
    for raw_x, raw_y, raw_radius in circles[0][:30]:
        center = (float(raw_x), float(raw_y))
        radius = float(raw_radius)
        if not (0.08 * width <= center[0] <= 0.92 * width):
            continue
        if not (0.08 * height <= center[1] <= 0.92 * height):
            continue
        y_grid, x_grid = np.ogrid[:height, :width]
        face_mask = (x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2 <= (radius * 0.80) ** 2
        pixels = int(np.count_nonzero(face_mask))
        if pixels == 0:
            continue
        neutral = (hsv[:, :, 1] <= 110) & (hsv[:, :, 2] >= 80) & face_mask
        neutral_ratio = float(np.count_nonzero(neutral) / pixels)
        ring = face_edge_support(edges, center, radius)
        center_distance = math.hypot(center[0] - image_center[0], center[1] - image_center[1]) / short
        center_score = clamp(1.0 - center_distance / 0.55)
        radius_score = clamp(1.0 - abs(radius / short - 0.38) / 0.20)
        score = 0.45 * ring + 0.25 * clamp(neutral_ratio / 0.60) + 0.20 * center_score + 0.10 * radius_score
        candidates.append(
            {
                "center": [round(center[0], 3), round(center[1], 3)],
                "face_radius": round(radius, 3),
                "face_edge_support": round(ring, 6),
                "neutral_face_ratio": round(neutral_ratio, 6),
                "score": round(score, 6),
            }
        )
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    if not candidates:
        return {"available": False, "confidence": 0.0, "reason": "face_circle_rejected", "candidates": []}
    best = candidates[0]
    second = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
    confidence = clamp(0.55 * float(best["score"]) + 0.45 * max(0.0, float(best["score"]) - second) * 4.0)
    return {
        "available": True,
        "confidence": round(confidence, 6),
        "center": best["center"],
        "face_radius": best["face_radius"],
        "candidates": candidates[:8],
        "reason": "estimated_from_hough_face_circle",
    }


def ray_samples(
    gray: np.ndarray,
    edges: np.ndarray,
    center: tuple[float, float],
    radius: float,
    angle: float,
) -> tuple[float, float, float]:
    """Return dark support, edge support, and valid sample ratio for one ray."""

    distances = np.linspace(radius * 0.10, radius * 0.72, 64)
    offsets = np.linspace(-2.0, 2.0, 5)
    x_values = center[0] + np.cos(angle) * distances[:, None] - np.sin(angle) * offsets[None, :]
    y_values = center[1] - np.sin(angle) * distances[:, None] - np.cos(angle) * offsets[None, :]
    x = np.rint(x_values).astype(np.int32)
    y = np.rint(y_values).astype(np.int32)
    valid = (x >= 0) & (x < gray.shape[1]) & (y >= 0) & (y < gray.shape[0])
    if not np.any(valid):
        return 0.0, 0.0, 0.0
    gray_values = gray[y[valid], x[valid]].astype(np.float32)
    edge_values = edges[y[valid], x[valid]] > 0
    median = float(np.median(gray_values))
    deviation = float(np.std(gray_values))
    threshold = max(70.0, min(155.0, median - 0.35 * deviation))
    dark_support = float(np.mean(gray_values <= threshold))
    edge_support = float(np.mean(edge_values))
    valid_ratio = float(np.mean(valid))
    return dark_support, edge_support, valid_ratio


def detect_pointer(
    image: np.ndarray,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(enhanced, 45, 140)
    center = (float(geometry["center"][0]), float(geometry["center"][1]))
    radius = float(geometry["face_radius"])
    angles = np.linspace(-math.pi, math.pi, 720, endpoint=False)
    profile = np.zeros(len(angles), dtype=np.float32)
    valid_profile = np.zeros(len(angles), dtype=np.float32)
    for index, angle in enumerate(angles):
        dark, edge, valid = ray_samples(gray, edges, center, radius, float(angle))
        # The pointer generally contributes a long inner dark/edge ray.  Scale
        # ticks are mostly farther out and therefore receive less weight.
        profile[index] = float(0.72 * dark + 0.28 * edge)
        valid_profile[index] = valid

    kernel = np.ones(9, dtype=np.float32) / 9.0
    padded = np.concatenate([profile[-4:], profile, profile[:4]])
    smoothed = np.convolve(padded, kernel, mode="valid")
    baseline = float(np.median(smoothed))
    spread = float(np.percentile(smoothed, 95) - baseline)
    top_indices = np.argsort(smoothed)[::-1]
    profile_candidates: list[dict[str, Any]] = []
    for index in top_indices:
        candidate_angle = math.degrees(float(angles[index]))
        if any(abs(wrap_angle_deg(candidate_angle - item["angle_deg"])) < 5.0 for item in profile_candidates):
            continue
        score = float(smoothed[index])
        if not angle_in_calibrated_arc(candidate_angle, geometry):
            continue
        profile_candidates.append(
            {
                "angle_deg": round(candidate_angle, 4),
                "profile_score": round(score, 6),
                "support": round(clamp((score - baseline) / max(spread, 1e-6)), 6),
            }
        )
        if len(profile_candidates) >= 12:
            break

    min_line_length = max(20, int(round(radius * 0.18)))
    lines = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 180.0,
        threshold=max(16, int(round(radius * 0.10))),
        minLineLength=min_line_length,
        maxLineGap=max(6, int(round(radius * 0.025))),
    )
    line_candidates: list[dict[str, Any]] = []
    if lines is not None:
        # OpenCV versions may return (N, 1, 4) or (N, 4).
        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = (float(value) for value in line)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < min_line_length:
                continue
            distance = abs((center[0] - x1) * dy - (center[1] - y1) * dx) / max(length, 1e-6)
            endpoint_distances = [math.hypot(x1 - center[0], y1 - center[1]), math.hypot(x2 - center[0], y2 - center[1])]
            near_endpoint = min(endpoint_distances)
            far_endpoint = max(endpoint_distances)
            if (
                distance > radius * 0.12
                or near_endpoint > radius * 0.45
                or far_endpoint < radius * 0.22
                or far_endpoint > radius * 1.08
            ):
                continue
            endpoint = (x1, y1) if endpoint_distances[0] >= endpoint_distances[1] else (x2, y2)
            candidate_angle = angle_deg(center, endpoint)
            if not angle_in_calibrated_arc(candidate_angle, geometry):
                continue
            dark, edge, valid = ray_samples(gray, edges, center, radius, math.radians(candidate_angle))
            score = (
                0.45 * (0.72 * dark + 0.28 * edge)
                + 0.25 * clamp(length / (radius * 0.72))
                + 0.20 * clamp(1.0 - distance / (radius * 0.12))
                + 0.10 * clamp((far_endpoint / radius - 0.22) / 0.86)
            )
            line_candidates.append(
                {
                    "angle_deg": round(candidate_angle, 4),
                    "line_length": round(length, 4),
                    "center_distance": round(distance, 4),
                    "ray_score": round(0.72 * dark + 0.28 * edge, 6),
                    "support": round(valid, 6),
                    "score": round(score, 6),
                }
            )
    line_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    all_candidates = profile_candidates + line_candidates[:12]
    all_candidates.sort(key=lambda item: float(item.get("score", item.get("support", 0.0))), reverse=True)
    if not all_candidates:
        return {
            "detected": False,
            "angle_deg": None,
            "confidence": 0.0,
            "reason": "pointer_line_not_found",
            "profile_candidates": profile_candidates,
            "line_candidates": line_candidates[:12],
        }

    selected = all_candidates[0]
    selected_score = float(selected.get("score", selected.get("profile_score", 0.0)))
    second_score = float(all_candidates[1].get("score", all_candidates[1].get("profile_score", 0.0))) if len(all_candidates) > 1 else 0.0
    confidence = clamp(0.55 * selected_score + 0.45 * max(0.0, selected_score - second_score) * 5.0)
    return {
        "detected": True,
        "angle_deg": float(selected["angle_deg"]),
        "confidence": round(confidence, 6),
        "selected_source": "hough_line_or_radial_profile",
        "selected_candidate": selected,
        "profile_candidates": profile_candidates,
        "line_candidates": line_candidates[:12],
    }


def classify_angle(
    angle: float | None,
    confidence: float,
    geometry: dict[str, Any],
    zero_tolerance: float = 0.05,
    near_full_fraction: float = 0.90,
    overrun_tolerance: float = 0.05,
) -> dict[str, Any]:
    if angle is None:
        return {
            "needle_state": "uncertain",
            "position_class": "uncertain",
            "deflection_fraction": None,
            "reason": "pointer_angle_missing",
        }
    if confidence < 0.30:
        return {
            "needle_state": "uncertain",
            "position_class": "low_confidence",
            "deflection_fraction": None,
            "reason": "pointer_angle_confidence_below_threshold",
        }
    zero = geometry.get("zero_angle_deg")
    full = geometry.get("full_scale_angle_deg")
    if zero is None or full is None:
        return {
            "needle_state": "uncertain",
            "position_class": "uncalibrated",
            "deflection_fraction": None,
            "reason": "zero_or_full_scale_calibration_missing",
        }
    direction = str(geometry.get("direction", "cw"))
    arc = directed_delta_deg(float(zero), float(full), direction)
    if abs(arc) < 20.0:
        return {
            "needle_state": "uncertain",
            "position_class": "uncertain",
            "deflection_fraction": None,
            "reason": "calibration_arc_too_small",
        }
    position = directed_delta_deg(float(zero), float(angle), direction)
    fraction = position / arc
    if fraction < -zero_tolerance:
        state, position_class = "reverse", "reverse"
    elif abs(fraction) <= zero_tolerance:
        state, position_class = "zero", "zero"
    elif fraction >= 1.0 + overrun_tolerance:
        state, position_class = "overrange", "overrange"
    elif fraction >= 1.0:
        state, position_class = "overrange", "at_full_scale"
    elif fraction >= near_full_fraction:
        state, position_class = "uncertain", "near_full_scale"
    else:
        state, position_class = "normal_rightward", "within_scale"
    return {
        "needle_state": state,
        "position_class": position_class,
        "deflection_fraction": round(float(fraction), 6),
        "reason": "calibrated_pointer_position",
    }


def measure_image(
    path: Path,
    calibration: dict[str, Any] | None,
    debug_path: Path | None,
) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "path": str(path.resolve()),
            "valid": False,
            "errors": ["image_decode_failed"],
        }
    height, width = image.shape[:2]
    warnings: list[str] = []
    if calibration is None:
        estimated = estimate_face_geometry(image)
        if not estimated.get("available"):
            return {
                "path": str(path.resolve()),
                "valid": False,
                "width": width,
                "height": height,
                "errors": [str(estimated.get("reason", "face_geometry_missing"))],
                "face_geometry": estimated,
            }
        geometry = {
            "center": estimated["center"],
            "face_radius": estimated["face_radius"],
            "zero_angle_deg": None,
            "full_scale_angle_deg": None,
            "direction": "cw",
            "source": "auto_estimate",
        }
        warnings.append("zero_and_full_scale_require_explicit_calibration")
        face_geometry = estimated
    else:
        try:
            geometry = resize_calibrated_geometry(calibration, width, height)
        except ValueError as exc:
            return {
                "path": str(path.resolve()),
                "valid": False,
                "width": width,
                "height": height,
                "errors": [str(exc)],
            }
        face_geometry = {"available": True, "source": "provided_calibration"}

    pointer = detect_pointer(image, geometry)
    classification = classify_angle(pointer.get("angle_deg"), float(pointer.get("confidence", 0.0)), geometry)
    geometry_confidence = float(face_geometry.get("confidence", 1.0))
    if geometry.get("source") == "auto_estimate" and geometry_confidence < 0.50:
        warnings.append("auto_face_geometry_low_confidence")
        classification = {
            "needle_state": "uncertain",
            "position_class": "low_geometry_confidence",
            "deflection_fraction": None,
            "reason": "auto_face_geometry_confidence_below_threshold",
        }
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "valid": bool(pointer.get("detected")) and geometry_confidence >= 0.50 and float(pointer.get("confidence", 0.0)) >= 0.30,
        "width": width,
        "height": height,
        "geometry": geometry,
        "face_geometry": face_geometry,
        "pointer": pointer,
        "classification": classification,
        "warnings": warnings,
        "errors": [] if pointer.get("detected") else [str(pointer.get("reason", "pointer_not_found"))],
    }

    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        overlay = image.copy()
        center = (int(round(float(geometry["center"][0]))), int(round(float(geometry["center"][1]))))
        radius = int(round(float(geometry["face_radius"])))
        cv2.circle(overlay, center, radius, (255, 180, 0), 3)
        cv2.circle(overlay, center, 6, (0, 0, 255), -1)
        if pointer.get("angle_deg") is not None:
            theta = math.radians(float(pointer["angle_deg"]))
            endpoint = (
                int(round(center[0] + math.cos(theta) * radius * 0.86)),
                int(round(center[1] - math.sin(theta) * radius * 0.86)),
            )
            cv2.line(overlay, center, endpoint, (0, 255, 0), 4, cv2.LINE_AA)
        for key, color in (("zero_angle_deg", (255, 0, 0)), ("full_scale_angle_deg", (0, 165, 255))):
            if geometry.get(key) is not None:
                theta = math.radians(float(geometry[key]))
                endpoint = (
                    int(round(center[0] + math.cos(theta) * radius * 0.82)),
                    int(round(center[1] - math.sin(theta) * radius * 0.82)),
                )
                cv2.line(overlay, center, endpoint, color, 2, cv2.LINE_AA)
        label = f"angle={pointer.get('angle_deg')} state={classification.get('needle_state')} conf={pointer.get('confidence')}"
        cv2.rectangle(overlay, (10, 10), (min(width - 10, 900), 48), (255, 255, 255), -1)
        cv2.putText(overlay, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(debug_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
        result["debug_path"] = str(debug_path.resolve())
    return result


def aggregate_sequence(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in results if item.get("valid") and item.get("pointer", {}).get("angle_deg") is not None]
    if not valid:
        return {
            "valid": False,
            "needle_state": "uncertain",
            "deflection_fraction": None,
            "reason": "no_valid_pointer_measurements",
            "valid_frame_count": 0,
        }
    states = [str(item.get("classification", {}).get("needle_state", "uncertain")) for item in valid]
    state_counts: dict[str, int] = {}
    for state in states:
        state_counts[state] = state_counts.get(state, 0) + 1
    dominant_state = max(state_counts, key=state_counts.get)
    angles = np.asarray([float(item["pointer"]["angle_deg"]) for item in valid], dtype=np.float64)
    reference = float(angles[0])
    unwrapped = np.asarray([reference + wrap_angle_deg(float(value) - reference) for value in angles], dtype=np.float64)
    median_angle = float(np.median(unwrapped))
    mad = float(np.median(np.abs(unwrapped - median_angle)))
    consensus = state_counts[dominant_state] / len(states)
    stable = len(valid) == 1 or (consensus >= 0.60 and mad <= 8.0)
    classification = valid[0].get("classification", {})
    fractions = [item.get("classification", {}).get("deflection_fraction") for item in valid]
    fractions = [float(item) for item in fractions if finite_number(item)]
    return {
        "valid": bool(stable),
        "needle_state": dominant_state if stable else "uncertain",
        "deflection_fraction": round(float(np.median(fractions)), 6) if fractions else None,
        "median_angle_deg": round(median_angle, 4),
        "angle_mad_deg": round(mad, 4),
        "state_counts": state_counts,
        "state_consensus": round(consensus, 6),
        "stability": "single_frame_not_assessed" if len(valid) == 1 else ("stable" if stable else "unstable"),
        "valid_frame_count": len(valid),
        "source_classification_example": classification,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure analog meter pointer angle with OpenCV.")
    parser.add_argument("--image", action="append", required=True, help="Meter ROI image; repeat for a temporal sequence.")
    parser.add_argument("--output", required=True, help="Output JSON measurement report.")
    parser.add_argument("--calibration", help="Optional calibration JSON with center/radius/zero/full-scale angles.")
    parser.add_argument("--debug-dir", help="Optional directory for overlay images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [Path(value).expanduser().resolve() for value in args.image]
    calibration: dict[str, Any] | None = None
    errors: list[str] = []
    if args.calibration:
        calibration_path = Path(args.calibration).expanduser().resolve()
        try:
            value = read_json(calibration_path)
            if not isinstance(value, dict):
                raise ValueError("calibration_not_object")
            calibration = value
        except Exception as exc:
            errors.append(f"calibration_read_failed:{type(exc).__name__}:{exc}")

    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else None
    measurements: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        debug_path = debug_dir / f"{index:03d}_{path.stem}_pointer_angle.jpg" if debug_dir else None
        measurements.append(measure_image(path, calibration, debug_path))

    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "measurement_only": True,
        "qwen_called": False,
        "excel_accessed": False,
        "video_accessed": False,
        "score_computed": False,
        "calibration_path": str(Path(args.calibration).expanduser().resolve()) if args.calibration else None,
        "image_count": len(paths),
        "measurements": measurements,
        "aggregate": aggregate_sequence(measurements),
        "errors": errors,
        "warnings": [
            "OpenCV geometry is not semantic meter identification.",
            "Normalized deflection requires a trusted zero/full-scale calibration.",
            "A low-confidence or ambiguous pointer is reported as uncertain.",
        ],
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
