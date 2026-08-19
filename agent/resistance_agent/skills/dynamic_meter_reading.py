"""Situation-driven analog-meter candidate selection shared by meter rubrics."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean
from typing import Any

import cv2


SKILL_VERSION = "dynamic_meter_reading.v3"
METER_ROLES = {"ammeter", "voltmeter"}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _candidate_coordinate_size(detection: dict[str, Any]) -> tuple[float, float]:
    """Return the image size used by exported candidate boxes."""
    width = detection.get("source_image_width") or detection.get("image_width") or 1.0
    height = detection.get("source_image_height") or detection.get("image_height") or 1.0
    return float(width), float(height)


def _xywh(candidate: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = candidate.get("face_bbox")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return x, y, width, height


def _face_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    diagnostics = candidate.get("opencv_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    value = diagnostics.get("face")
    return value if isinstance(value, dict) else {}


def _intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return width * height


def _same_face(
    first: dict[str, Any],
    second: dict[str, Any],
    image_width: float,
    image_height: float,
) -> bool:
    a, b = _xywh(first), _xywh(second)
    if a is None or b is None:
        return False
    intersection = _intersection(a, b)
    a_area, b_area = a[2] * a[3], b[2] * b[3]
    union = a_area + b_area - intersection
    iou = intersection / max(union, 1.0)
    containment = intersection / max(min(a_area, b_area), 1.0)
    acx, acy = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
    bcx, bcy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
    normalized_distance = math.hypot(
        (acx - bcx) / max(image_width, 1.0),
        (acy - bcy) / max(image_height, 1.0),
    )
    area_ratio = min(a_area, b_area) / max(a_area, b_area, 1.0)
    return iou >= 0.24 or containment >= 0.52 or (normalized_distance < 0.045 and area_ratio >= 0.42)


def score_face_completeness(
    candidate: dict[str, Any],
    image_width: float,
    image_height: float,
) -> dict[str, Any]:
    """Score visible dial structure without using video identity or fixed coordinates."""
    metrics = _face_metrics(candidate)
    box = _xywh(candidate)
    if box is None:
        return {"score": 0.0, "status": "weak", "reasons": ["face_bbox_missing"]}
    x, y, width, height = box
    dial = _clamp(float(metrics.get("dial_likeness") or 0.0))
    structure = _clamp(float(metrics.get("structure_score") or 0.0))
    neutral = _clamp(float(metrics.get("interior_neutral_ratio") or 0.0))
    edge = _clamp(float(metrics.get("interior_edge_density") or 0.0) / 0.12)
    scale_band = _clamp(float(metrics.get("upper_dark_ratio") or 0.0) / 0.18)
    aspect = min(width, height) / max(width, height, 1.0)
    aspect_score = _clamp((aspect - 0.46) / 0.42)
    touches_edge = x <= 1.0 or y <= 1.0 or x + width >= image_width - 1.0 or y + height >= image_height - 1.0
    boundary_score = 0.35 if touches_edge else 1.0
    layout = 0.0
    layout_details: dict[str, Any] = {}
    face_path = candidate.get("face_path")
    if isinstance(face_path, str):
        image = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
        if image is not None and image.size:
            image = cv2.resize(image, (240, 180), interpolation=cv2.INTER_AREA)
            edges = cv2.Canny(image, 45, 150)
            upper = edges[: int(edges.shape[0] * 0.58)]
            bands = [upper[:, start : start + upper.shape[1] // 5] for start in range(0, upper.shape[1], max(1, upper.shape[1] // 5))]
            band_density = [float((band > 0).mean()) if band.size else 0.0 for band in bands[:5]]
            scale_spread = sum(value >= 0.018 for value in band_density) / 5.0
            border = max(2, int(min(edges.shape) * 0.08))
            border_density = [
                float((edges[:border, :] > 0).mean()),
                float((edges[-border:, :] > 0).mean()),
                float((edges[:, :border] > 0).mean()),
                float((edges[:, -border:] > 0).mean()),
            ]
            bezel_support = _clamp(sum(value >= 0.018 for value in border_density) / 4.0)
            pointer = candidate.get("detector_pointer") or {}
            pointer_confidence = _clamp(float(pointer.get("confidence") or 0.0))
            hub_support = pointer_confidence if pointer.get("detected") else 0.15
            radial_support = _clamp(0.55 * scale_spread + 0.45 * pointer_confidence)
            layout = 0.30 * scale_spread + 0.25 * hub_support + 0.25 * radial_support + 0.20 * bezel_support
            layout_details = {
                "scale_spread": round(scale_spread, 6),
                "hub_support": round(hub_support, 6),
                "radial_support": round(radial_support, 6),
                "bezel_support": round(bezel_support, 6),
                "band_edge_density": [round(value, 6) for value in band_density],
            }
    visible_fraction = boundary_score
    score = (
        0.20 * visible_fraction
        + 0.20 * layout
        + 0.18 * dial
        + 0.14 * structure
        + 0.10 * neutral
        + 0.08 * edge
        + 0.06 * scale_band
        + 0.04 * aspect_score
    )
    reasons: list[str] = []
    if dial < 0.42:
        reasons.append("dial_likeness_low")
    if structure < 0.48:
        reasons.append("face_structure_low")
    if touches_edge:
        reasons.append("face_touches_image_boundary")
    if score >= 0.52 and visible_fraction >= 0.70 and dial >= 0.42 and structure >= 0.48:
        status = "complete"
    elif score >= 0.38:
        status = "partial"
    else:
        status = "weak"
    return {
        "score": round(score, 6),
        "status": status,
        "visible_fraction": round(visible_fraction, 6),
        "layout": round(layout, 6),
        "layout_details": layout_details,
        "reasons": reasons,
    }


def _selection_score(candidate: dict[str, Any]) -> float:
    completeness = candidate.get("face_completeness") or {}
    pointer = candidate.get("detector_pointer") or {}
    return (
        0.52 * float(candidate.get("quality") or 0.0)
        + 0.38 * float(completeness.get("score") or 0.0)
        + 0.10 * float(pointer.get("confidence") or 0.0)
    )


def _render_pointer_overlay(
    candidate: dict[str, Any],
    image_width: int,
    image_height: int,
) -> str | None:
    face_path = candidate.get("face_path")
    box = _xywh(candidate)
    pointer = candidate.get("detector_pointer")
    if not isinstance(face_path, str) or box is None or not isinstance(pointer, dict):
        return None
    image = cv2.imread(face_path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    x, y, width, height = box
    pad_x, pad_y = int(round(width * 0.08)), int(round(height * 0.08))
    offset_x = x - max(0.0, x - pad_x)
    offset_y = y - max(0.0, y - pad_y)
    center = pointer.get("center_xy")
    endpoint = pointer.get("endpoint_xy")
    if (
        bool(pointer.get("detected"))
        and isinstance(center, list)
        and len(center) == 2
        and isinstance(endpoint, list)
        and len(endpoint) == 2
    ):
        start = (int(round(float(center[0]) + offset_x)), int(round(float(center[1]) + offset_y)))
        end = (int(round(float(endpoint[0]) + offset_x)), int(round(float(endpoint[1]) + offset_y)))
        cv2.line(image, start, end, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(image, start, 5, (0, 255, 0), -1, cv2.LINE_AA)
    label = f"{candidate.get('candidate_id', 'candidate')} pointer={float(pointer.get('confidence') or 0.0):.2f}"
    cv2.rectangle(image, (0, 0), (min(image.shape[1] - 1, 520), 34), (255, 255, 255), -1)
    cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
    output = Path(face_path).with_name(Path(face_path).stem + "_pointer_overlay.jpg")
    if not cv2.imwrite(str(output), image, [int(cv2.IMWRITE_JPEG_QUALITY), 96]):
        return None
    return str(output.resolve())


def enrich_frame(frame: dict[str, Any], *, render_overlays: bool = True) -> dict[str, Any]:
    """Rank one frame's candidates and suppress duplicate views of one meter."""
    detection = frame.get("detection") if isinstance(frame.get("detection"), dict) else {}
    coordinate_width, coordinate_height = _candidate_coordinate_size(detection)
    image_width = int(round(coordinate_width))
    image_height = int(round(coordinate_height))
    candidates = [item for item in frame.get("candidates") or [] if isinstance(item, dict)]
    for candidate in candidates:
        completeness = score_face_completeness(candidate, image_width, image_height)
        candidate["face_completeness"] = completeness
        candidate["selection_score"] = round(_selection_score(candidate), 6)
        role = str(candidate.get("role_hint") or "unknown")
        reasons = candidate.get("opencv_diagnostics", {}).get("evidence_insufficient_reason") or []
        reliable = (
            role in METER_ROLES
            and candidate.get("detector_source") == "terminal_anchor_search"
            and completeness["status"] == "complete"
            and "cross_role_face_overlap_identity_conflict" not in reasons
        )
        candidate["identity_hint"] = role if reliable else "unknown"
        candidate["identity_hint_reliable"] = reliable
        candidate["selected_for_model"] = False
        candidate.pop("suppressed_as_duplicate_of", None)

    ranked = sorted(candidates, key=_selection_score, reverse=True)
    preferred_ids: list[str] = []
    pair = detection.get("selected_pair") if isinstance(detection.get("selected_pair"), dict) else {}
    if pair.get("status") == "paired":
        for role in ("ammeter", "voltmeter"):
            raw = pair.get(role)
            if isinstance(raw, dict) and isinstance(raw.get("candidate_id"), str):
                preferred_ids.append(raw["candidate_id"])
    preferred = [
        candidate
        for candidate_id in preferred_ids
        for candidate in ranked
        if candidate.get("candidate_id") == candidate_id
    ]
    ordered = preferred + [candidate for candidate in ranked if candidate not in preferred]
    selected: list[dict[str, Any]] = []
    for candidate in ordered:
        duplicate = next(
            (
                existing
                for existing in selected
                if _same_face(candidate, existing, float(image_width), float(image_height))
            ),
            None,
        )
        if duplicate is not None:
            candidate["suppressed_as_duplicate_of"] = duplicate.get("candidate_id")
            continue
        selected.append(candidate)
        if len(selected) >= 2:
            break
    for candidate in selected:
        candidate["selected_for_model"] = True
        if render_overlays:
            candidate["pointer_overlay_path"] = _render_pointer_overlay(candidate, image_width, image_height)
    frame["candidates"] = candidates
    frame["model_candidates"] = selected
    frame["candidate_selection"] = {
        "skill_version": SKILL_VERSION,
        "status": "two_distinct_faces" if len(selected) == 2 else "single_face" if selected else "no_face",
        "selected_candidate_ids": [str(candidate.get("candidate_id")) for candidate in selected],
        "suppressed_duplicate_ids": [
            str(candidate.get("candidate_id"))
            for candidate in candidates
            if candidate.get("suppressed_as_duplicate_of")
        ],
        "detector_pair_status": pair.get("status"),
        "candidate_coordinate_space": detection.get("candidate_coordinate_space") or "detection_image_pixels",
        "candidate_coordinate_size": [image_width, image_height],
    }
    return frame


def _track_candidates(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    orb = cv2.ORB_create(nfeatures=320, scaleFactor=1.2, nlevels=6)

    def visual_features(candidate: dict[str, Any]) -> tuple[Any, Any]:
        path = candidate.get("face_path")
        image = cv2.imread(path, cv2.IMREAD_COLOR) if isinstance(path, str) else None
        if image is None or image.size == 0:
            return None, None
        scale = min(1.0, 360.0 / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _points, descriptors = orb.detectAndCompute(gray, None)
        return histogram, descriptors

    def visual_similarity(first: dict[str, Any], second: dict[str, Any]) -> float:
        first_hist, first_descriptors = first.get("visual_features", (None, None))
        second_hist, second_descriptors = second.get("visual_features", (None, None))
        histogram_score = 0.0
        if first_hist is not None and second_hist is not None:
            correlation = float(cv2.compareHist(first_hist, second_hist, cv2.HISTCMP_CORREL))
            histogram_score = _clamp((correlation + 1.0) / 2.0)
        orb_score = 0.0
        if first_descriptors is not None and second_descriptors is not None:
            matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(first_descriptors, second_descriptors, k=2)
            good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.76 * pair[1].distance]
            orb_score = _clamp(len(good) / 14.0)
        return 0.42 * histogram_score + 0.58 * orb_score

    tracks: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        detection = frame.get("detection") if isinstance(frame.get("detection"), dict) else {}
        image_width, image_height = _candidate_coordinate_size(detection)
        for candidate in frame.get("model_candidates") or []:
            box = _xywh(candidate)
            if box is None:
                continue
            x, y, width, height = box
            observation = {
                "frame_index": frame_index,
                "candidate": candidate,
                "center": [(x + width / 2.0) / image_width, (y + height / 2.0) / image_height],
                "area": width * height / max(image_width * image_height, 1.0),
                "visual_features": visual_features(candidate),
            }
            compatible: list[tuple[float, float, dict[str, Any]]] = []
            for track in tracks:
                previous = track["observations"][-1]
                frame_gap = frame_index - int(previous["frame_index"])
                if frame_gap <= 0 or frame_gap > 2:
                    continue
                distance = math.dist(observation["center"], previous["center"])
                area_ratio = observation["area"] / max(float(previous["area"]), 1e-9)
                appearance = visual_similarity(observation, previous)
                spatial_support = math.exp(-distance / 0.16)
                scale_support = math.exp(-2.0 * abs(math.log(max(area_ratio, 1e-9))))
                score = 0.45 * appearance + 0.35 * spatial_support + 0.20 * scale_support
                if (
                    0.28 <= area_ratio <= 3.6
                    and (distance <= 0.17 or (appearance >= 0.48 and distance <= 0.58))
                    and score >= 0.39
                ):
                    compatible.append((score, appearance, track))
            if compatible:
                score, appearance, selected = max(compatible, key=lambda item: item[0])
                selected["observations"].append(observation)
                selected["match_scores"].append(score)
                selected["appearance_scores"].append(appearance)
            else:
                tracks.append({"observations": [observation], "match_scores": [], "appearance_scores": []})

    summaries: list[dict[str, Any]] = []
    for index, track in enumerate(tracks, start=1):
        observations = track["observations"]
        centers = [item["center"] for item in observations]
        mean_center = [fmean(item[axis] for item in centers) for axis in (0, 1)]
        jitter = max((math.dist(center, mean_center) for center in centers), default=0.0)
        role_weights = {role: 0.0 for role in METER_ROLES}
        for observation in observations:
            candidate = observation["candidate"]
            role = str(candidate.get("role_hint") or "")
            if role in role_weights:
                source_weight = 1.0 if candidate.get("detector_source") == "terminal_anchor_search" else 0.35
                role_weights[role] += source_weight * max(0.05, float(candidate.get("selection_score") or 0.0))
        winner = max(role_weights, key=role_weights.get)
        total = sum(role_weights.values())
        role_confidence = role_weights[winner] / total if total else 0.0
        mean_match_score = fmean(track.get("match_scores") or [0.0])
        stable = (
            len({item["frame_index"] for item in observations}) >= 2
            and (jitter <= 0.18 or mean_match_score >= 0.48)
        )
        identity = winner if stable and role_confidence >= 0.65 else "unknown"
        track_id = f"meter_track_{index:02d}"
        for observation in observations:
            observation["candidate"]["track_id"] = track_id
            observation["candidate"]["track_identity_hint"] = identity
        summaries.append(
            {
                "track_id": track_id,
                "source_frame_count": len({item["frame_index"] for item in observations}),
                "mean_center": [round(value, 6) for value in mean_center],
                "center_jitter": round(jitter, 6),
                "mean_selection_score": round(fmean(float(item["candidate"].get("selection_score") or 0.0) for item in observations), 6),
                "mean_track_match_score": round(mean_match_score, 6),
                "mean_appearance_score": round(fmean(track.get("appearance_scores") or [0.0]), 6),
                "identity_hint": identity,
                "identity_hint_confidence": round(role_confidence, 6),
                "stability": "stable" if stable else "single_or_unstable",
            }
        )
    summaries.sort(
        key=lambda item: (int(item["source_frame_count"]), float(item["mean_selection_score"])),
        reverse=True,
    )
    if len(summaries) >= 2 and summaries[0]["identity_hint"] == summaries[1]["identity_hint"] != "unknown":
        weaker = min(summaries[:2], key=lambda item: float(item["identity_hint_confidence"]))
        weaker["identity_hint"] = "unknown"
        weaker["identity_conflict"] = "two_tracks_claimed_same_role"
        for frame in frames:
            for candidate in frame.get("model_candidates") or []:
                if candidate.get("track_id") == weaker["track_id"]:
                    candidate["track_identity_hint"] = "unknown"
    return summaries


def prepare_frames(frames: list[dict[str, Any]], *, render_overlays: bool = True) -> dict[str, Any]:
    """Apply frame-level selection and short-window temporal identity checks."""
    for frame in frames:
        enrich_frame(frame, render_overlays=render_overlays)
    tracks = _track_candidates(frames)
    return {
        "skill_version": SKILL_VERSION,
        "selection_basis": "visible_face_structure_spatial_distinctness_and_short_window_tracks",
        "video_id_used": False,
        "historical_artifact_used": False,
        "tracks": tracks,
    }


def candidate_media_paths(frame: dict[str, Any]) -> list[Path]:
    """Return face, terminal and enhanced context for selected distinct faces."""
    paths: list[Path] = []
    for candidate in frame.get("model_candidates") or []:
        for key in ("face_path", "terminal_path", "enhanced_path"):
            value = candidate.get(key)
            if not isinstance(value, str):
                continue
            path = Path(value)
            if path.is_file() and path not in paths:
                paths.append(path)
    return paths


def panorama_location_prompt() -> str:
    return """Locate physical rectangular analog meters in this full experiment frame.
Identify an ammeter only from a visible A glyph and analog arc, and a voltmeter only from a visible V glyph and analog arc. Ignore the power supply, battery holder, rheostat, resistor, switch, paper and wires. Return a tight box including arc scale, needle/hub and terminal panel. Coordinates are integers from 0 to 1000 relative to the full image. Do not assign the same physical box to both roles. Replace nulls from visible pixels; omit a role that is not visible. Return JSON only:
{"meters":[{"identity":"ammeter","bbox_normalized_1000":[null,null,null,null],"face_visible":true,"confidence":null,"evidence":"visible glyph and analog arc"}]}"""


def validate_panorama_locations(value: dict[str, Any]) -> dict[str, Any]:
    meters = value.get("meters") if isinstance(value, dict) else None
    if not isinstance(meters, list):
        raise ValueError("panorama_meters_missing")
    parsed: list[dict[str, Any]] = []
    for item in meters:
        if not isinstance(item, dict):
            continue
        identity = str(item.get("identity") or "").lower()
        box = item.get("bbox_normalized_1000")
        if identity not in METER_ROLES or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            values = [int(round(float(number))) for number in box]
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            continue
        width, height = x2 - x1, y2 - y1
        # A readable meter crop must include the arc, needle/hub and terminal
        # panel. Very shallow strips are terminal panels or clipped apparatus,
        # even when a VLM assigns them an A/V role.
        if width < 120 or height < 140 or width * height < 24000:
            continue
        confidence = item.get("confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else 0.0
        parsed.append(
            {
                "identity": identity,
                "bbox_normalized_1000": values,
                "face_visible": bool(item.get("face_visible")),
                "confidence": round(_clamp(confidence), 4),
                "evidence": str(item.get("evidence") or ""),
            }
        )
    by_role = {item["identity"]: item for item in parsed if item["face_visible"]}
    if set(by_role) != METER_ROLES:
        raise ValueError("panorama_both_meter_roles_not_visible")
    ammeter_box = by_role["ammeter"]["bbox_normalized_1000"]
    voltmeter_box = by_role["voltmeter"]["bbox_normalized_1000"]
    ammeter_xywh = [ammeter_box[0], ammeter_box[1], ammeter_box[2] - ammeter_box[0], ammeter_box[3] - ammeter_box[1]]
    voltmeter_xywh = [voltmeter_box[0], voltmeter_box[1], voltmeter_box[2] - voltmeter_box[0], voltmeter_box[3] - voltmeter_box[1]]
    first, second = tuple(float(number) for number in ammeter_xywh), tuple(float(number) for number in voltmeter_xywh)
    intersection = _intersection(first, second)
    first_area, second_area = first[2] * first[3], second[2] * second[3]
    union = first_area + second_area - intersection
    iou = intersection / max(union, 1.0)
    containment = intersection / max(min(first_area, second_area), 1.0)
    if iou >= 0.30 or containment >= 0.58:
        raise ValueError("panorama_cross_role_boxes_overlap")
    return {
        "meters": [by_role[role] for role in ("ammeter", "voltmeter")],
        "cross_role_iou": round(iou, 6),
        "cross_role_containment": round(containment, 6),
    }


def export_panorama_crops(
    frame_path: Path,
    locations: dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> dict[str, dict[str, Any]]:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to decode panorama frame: {frame_path}")
    height, width = image.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, dict[str, Any]] = {}
    for item in locations.get("meters") or []:
        identity = str(item["identity"])
        x1, y1, x2, y2 = item["bbox_normalized_1000"]
        left, top = int(x1 * width / 1000), int(y1 * height / 1000)
        right, bottom = int(x2 * width / 1000), int(y2 * height / 1000)
        pad_x, pad_y = int((right - left) * 0.08), int((bottom - top) * 0.08)
        left, top = max(0, left - pad_x), max(0, top - pad_y)
        right, bottom = min(width, right + pad_x), min(height, bottom + pad_y)
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        path = output_dir / f"{prefix}_{identity}.jpg"
        if not cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96]):
            raise RuntimeError(f"unable to write panorama meter crop: {path}")
        output[identity] = {
            "image_path": str(path.resolve()),
            "source_bbox_xyxy": [left, top, right, bottom],
            "location_confidence": item.get("confidence"),
            "location_evidence": item.get("evidence"),
        }
    if set(output) != METER_ROLES:
        raise ValueError("panorama_meter_crop_export_incomplete")
    return output
