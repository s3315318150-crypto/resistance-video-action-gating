"""Pure-OpenCV dynamic knife-switch localization and state reduction."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ANALYSIS_SIZE = (256, 128)

# The orange switch body is often split by green terminals, red posts, or a
# hand.  A slightly wider support mask keeps those pieces available for the
# contact-pair test; the final score still rejects large battery holders.
ORANGE_MASK_HUE_MAX = 30
ORANGE_MASK_SAT_MIN = 140
ORANGE_MASK_VALUE_MIN = 50


def _rotated_crop(image: np.ndarray, rect: tuple[Any, Any, float]) -> np.ndarray:
    center, size, angle = rect
    width, height = size
    if width < height:
        width, height = height, width
        angle += 90.0
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return cv2.getRectSubPix(
        rotated,
        (max(1, round(width)), max(1, round(height))),
        center,
    )


def _contact_pair_features(normalized: np.ndarray) -> tuple[float, list[list[float]]]:
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, np.array([0, 145, 55]), np.array([27, 255, 255]))
    support = cv2.dilate(
        orange,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 13)),
    )
    metal = cv2.inRange(hsv, np.array([0, 0, 65]), np.array([179, 125, 255]))
    metal = cv2.bitwise_and(metal, support)
    metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    metal[:3, :] = 0
    metal[-3:, :] = 0
    metal[:, :3] = 0
    metal[:, -3:] = 0
    count, _, stats, centroids = cv2.connectedComponentsWithStats(metal)
    blobs: list[dict[str, Any]] = []
    image_area = normalized.shape[0] * normalized.shape[1]
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if not image_area * 0.0008 <= area <= image_area * 0.09:
            continue
        aspect = width / max(height, 1)
        if not 0.25 <= aspect <= 4.0:
            continue
        cx, cy = map(float, centroids[label])
        if cy <= normalized.shape[0] * 0.78:
            blobs.append({"area": area, "center": (cx, cy), "aspect": aspect})
    best_score = 0.0
    best_pair: list[list[float]] = []
    for first_index, first in enumerate(blobs):
        for second in blobs[first_index + 1 :]:
            ax, ay = first["center"]
            bx, by = second["center"]
            dx = abs(ax - bx) / normalized.shape[1]
            dy = abs(ay - by) / normalized.shape[0]
            if not 0.18 <= dx <= 0.72 or dy > 0.32:
                continue
            separation = math.exp(-abs(dx - 0.42) / 0.26)
            alignment = math.exp(-dy / 0.13)
            size_similarity = min(first["area"], second["area"]) / max(
                first["area"], second["area"]
            )
            square_like = math.sqrt(
                min(first["aspect"], 1.0 / first["aspect"])
                * min(second["aspect"], 1.0 / second["aspect"])
            )
            score = (
                separation
                * alignment
                * math.sqrt(size_similarity)
                * (0.55 + 0.45 * square_like)
            )
            if score > best_score:
                best_score = score
                best_pair = [[round(ax, 2), round(ay, 2)], [round(bx, 2), round(by, 2)]]
    best_pair.sort(key=lambda point: point[0])
    return float(best_score), best_pair


def _battery_like_score(
    long_side: float,
    short_side: float,
    frame_width: int,
    green_ratio: float,
    base_value_median: float,
) -> float:
    """Score geometry that is characteristic of a long battery holder.

    This is deliberately a soft penalty.  The locator remains usable when a
    real switch is partially occluded, while a large orange tray with green
    cells is kept below a compact support with paired contacts.
    """
    long_ratio = long_side / max(float(frame_width), 1.0)
    oversized = float(np.clip((long_ratio - 0.22) / 0.30, 0.0, 1.0))
    elongated = float(
        np.clip((long_side / max(short_side, 1.0) - 2.8) / 2.4, 0.0, 1.0)
    )
    green_cells = float(np.clip(green_ratio / 0.12, 0.0, 1.0))
    dim_holder = float(np.clip((145.0 - base_value_median) / 90.0, 0.0, 1.0))
    return float(
        np.clip(
            0.50 * oversized
            + 0.25 * oversized * elongated
            + 0.15 * oversized * green_cells
            + 0.10 * dim_holder,
            0.0,
            1.0,
        )
    )


def bridge_features(normalized: np.ndarray) -> dict[str, float]:
    """Measure copper-blade continuity between the two switch contacts."""
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    height, width = normalized.shape[:2]
    band_height = max(8, round(height * 0.44))
    base = hsv[round(height * 0.45) :, round(width * 0.15) : round(width * 0.85)]
    base_orange = (base[:, :, 0] <= 24) & (base[:, :, 1] >= 140)
    base_value = (
        float(np.median(base[:, :, 2][base_orange])) if np.any(base_orange) else 226.0
    )
    value_cutoff = int(np.clip(base_value * 0.82, 145, 205))
    bridge = cv2.inRange(
        hsv[:band_height],
        np.array([0, 45, 35]),
        np.array([27, 255, value_cutoff]),
    )
    bridge = cv2.morphologyEx(
        bridge,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
    )
    left = round(width * 0.25)
    right = round(width * 0.75)
    central = bridge[:, left:right] > 0
    dark_ratio = float(np.mean(central)) if central.size else 0.0
    column_coverage = (
        float(np.mean(np.mean(central, axis=0) >= 0.15)) if central.size else 0.0
    )
    max_span = 0.0
    for contour in cv2.findContours(
        bridge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]:
        x, _, component_width, _ = cv2.boundingRect(contour)
        if x + component_width >= left and x <= right:
            max_span = max(max_span, component_width / width)
    continuity = min(max_span, column_coverage)
    return {
        "bridge_score": round(float(np.clip(continuity, 0.0, 1.0)), 4),
        "bridge_span": round(float(max_span), 4),
        "bridge_column_coverage": round(float(column_coverage), 4),
        "bridge_dark_ratio": round(float(dark_ratio), 4),
        "base_value_median": round(base_value, 2),
        "value_cutoff": float(value_cutoff),
    }


def component_candidates(image: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
    """Find switch supports from orange bases and paired contacts.

    The support mask is intentionally broader than a thin orange contour.  A
    component can therefore be a short base with terminals rather than only a
    long orange bar.  Dynamic geometry and the contact-pair score suppress
    orange battery holders without using a video-specific ROI.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(
        hsv,
        np.array([0, ORANGE_MASK_SAT_MIN, ORANGE_MASK_VALUE_MIN]),
        np.array([ORANGE_MASK_HUE_MAX, 255, 255]),
    )
    orange = cv2.morphologyEx(
        orange,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 7)),
    )
    orange = cv2.morphologyEx(orange, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    frame_area = image.shape[0] * image.shape[1]
    records: list[dict[str, Any]] = []
    for contour in cv2.findContours(
        orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]:
        area = float(cv2.contourArea(contour))
        if area < frame_area * 0.00035 or area > frame_area * 0.08:
            continue
        rect = cv2.minAreaRect(contour)
        (_, _), (rect_width, rect_height), _ = rect
        long_side, short_side = max(rect_width, rect_height), min(rect_width, rect_height)
        if short_side < 8 or long_side < max(20.0, image.shape[1] * 0.025):
            continue
        aspect = long_side / max(short_side, 1.0)
        if not 1.0 <= aspect <= 10.0:
            continue
        normalized = _rotated_crop(image, rect)
        if normalized.size == 0:
            continue
        normalized = cv2.resize(normalized, ANALYSIS_SIZE, interpolation=cv2.INTER_CUBIC)
        normalized_hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
        normalized_gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        contour_x, contour_y, contour_width, contour_height = cv2.boundingRect(contour)
        context_pad = round(long_side * 0.75)
        context = hsv[
            max(0, contour_y - context_pad) : min(
                image.shape[0], contour_y + contour_height + context_pad
            ),
            max(0, contour_x - context_pad) : min(
                image.shape[1], contour_x + contour_width + context_pad
            ),
        ]
        context_colored = (context[:, :, 1] > 130) & (context[:, :, 2] > 60)
        context_green_ratio = float(
            np.mean(
                (context[:, :, 0] >= 30)
                & (context[:, :, 0] <= 100)
                & (context[:, :, 1] >= 60)
                & (context[:, :, 2] >= 35)
            )
        )
        context_red_ratio = float(
            np.mean(
                context_colored
                & ((context[:, :, 0] < 3) | (context[:, :, 0] > 170))
            )
        )
        orange_ratio = cv2.countNonZero(
            cv2.inRange(
                normalized_hsv,
                np.array([0, ORANGE_MASK_SAT_MIN, ORANGE_MASK_VALUE_MIN]),
                np.array([ORANGE_MASK_HUE_MAX, 255, 255]),
            )
        ) / float(normalized.shape[0] * normalized.shape[1])
        colored = (normalized_hsv[:, :, 1] > 130) & (normalized_hsv[:, :, 2] > 60)
        red_ratio = float(
            np.mean(
                colored
                & ((normalized_hsv[:, :, 0] < 3) | (normalized_hsv[:, :, 0] > 170))
            )
        )
        green_ratio = cv2.countNonZero(
            cv2.inRange(
                normalized_hsv,
                np.array([30, 60, 35]),
                np.array([100, 255, 255]),
            )
        ) / float(normalized.shape[0] * normalized.shape[1])
        dark_ratio = float(np.mean(normalized_gray < 70))
        metal = cv2.inRange(
            normalized_hsv,
            np.array([0, 0, 90]),
            np.array([179, 115, 255]),
        )
        metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        inner = metal[12:116, 10:246]
        count, _, stats, centroids = cv2.connectedComponentsWithStats(inner)
        metal_blobs: list[tuple[int, float, float]] = []
        for label in range(1, count):
            blob_area = int(stats[label, cv2.CC_STAT_AREA])
            if 30 <= blob_area <= 4000:
                cx, cy = centroids[label]
                metal_blobs.append((blob_area, float(cx + 10), float(cy + 12)))
        metal_blobs.sort(reverse=True)
        separated_metal = 0.0
        for _, ax, ay in metal_blobs[:8]:
            for _, bx, by in metal_blobs[:8]:
                dx = abs(ax - bx) / 256.0
                dy = abs(ay - by) / 128.0
                separated_metal = max(separated_metal, dx * math.exp(-3.0 * dy))
        contact_pair_score, contact_pair = _contact_pair_features(normalized)
        state_features = bridge_features(normalized)
        # Below this brightness the clamped blade threshold turns most of the
        # crop dark and can fabricate a full-width bridge from an occluder.
        if float(state_features["base_value_median"]) < 110.0:
            continue
        fill = area / max(long_side * short_side, 1.0)
        aspect_score = math.exp(-abs(math.log(max(aspect, 0.01) / 1.85)) / 0.75)
        orange_score = max(0.0, 1.0 - abs(orange_ratio - 0.42) / 0.48)
        battery_like = _battery_like_score(
            long_side,
            short_side,
            image.shape[1],
            green_ratio,
            float(state_features["base_value_median"]),
        )
        terminal_color_score = 0.5 * min(green_ratio / 0.12, 1.0) + 0.5 * min(
            red_ratio / 0.30, 1.0
        )
        score = (
            0.18 * aspect_score
            + 0.14 * orange_score
            + 0.13 * min(separated_metal / 0.55, 1.0)
            + 0.32 * contact_pair_score
            + 0.09 * min(dark_ratio / 0.15, 1.0)
            + 0.08 * min(fill / 0.70, 1.0)
            + 0.08 * terminal_color_score
            - 0.28 * min(context_green_ratio / 0.10, 1.0)
            - 0.26 * min(context_red_ratio / 0.20, 1.0)
            - 0.30 * battery_like
        )
        records.append(
            {
                "score": round(float(score), 4),
                "center": [round(float(rect[0][0]), 1), round(float(rect[0][1]), 1)],
                "size": [round(float(long_side), 1), round(float(short_side), 1)],
                "angle": round(float(rect[2]), 2),
                "aspect": round(float(aspect), 3),
                "fill": round(float(fill), 3),
                "orange_ratio": round(float(orange_ratio), 3),
                "green_ratio": round(float(green_ratio), 3),
                "red_ratio": round(float(red_ratio), 3),
                "context_green_ratio": round(context_green_ratio, 3),
                "context_red_ratio": round(context_red_ratio, 3),
                "dark_ratio": round(float(dark_ratio), 3),
                "separated_metal": round(float(separated_metal), 3),
                "contact_pair_score": round(float(contact_pair_score), 3),
                "contact_pair": contact_pair,
                "battery_like_score": round(float(battery_like), 3),
                "detection_mode": "orange_support_and_contact_pair",
                **state_features,
                "box": cv2.boxPoints(rect).astype(int).tolist(),
                "crop": normalized,
            }
        )
    records.sort(key=lambda item: float(item["score"]), reverse=True)
    return records[:limit]


def _descriptor(path: str) -> dict[str, Any]:
    image = cv2.imread(path)
    if image is None:
        raise OSError(f"unable to read switch candidate crop: {path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram)
    orb = cv2.ORB_create(nfeatures=350, fastThreshold=8)
    _, features = orb.detectAndCompute(gray, None)
    edge = cv2.Canny(gray, 60, 150)
    edge_vector = cv2.resize(edge, (64, 32), interpolation=cv2.INTER_AREA).astype(
        np.float32
    ).reshape(-1)
    edge_vector /= max(float(np.linalg.norm(edge_vector)), 1e-6)
    return {"histogram": histogram, "orb": features, "edge": edge_vector}


def _orb_similarity(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None or len(first) < 4 or len(second) < 4:
        return 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(first, second, k=2)
    good = [match for match, other in pairs if match.distance < 0.78 * other.distance]
    if not good:
        return 0.0
    count_score = min(len(good) / 20.0, 1.0)
    distance_score = max(
        0.0, 1.0 - float(np.median([item.distance for item in good])) / 90.0
    )
    return math.sqrt(count_score * distance_score)


def _groups(frames: list[dict[str, Any]], gap_seconds: float = 2.0) -> list[list[dict[str, Any]]]:
    output: list[list[dict[str, Any]]] = []
    for frame in frames:
        if (
            not output
            or frame["window_id"] != output[-1][-1]["window_id"]
            or float(frame["timestamp_seconds"])
            - float(output[-1][-1]["timestamp_seconds"])
            > gap_seconds
        ):
            output.append([])
        output[-1].append(frame)
    return output


def _choose_neighbor(frame: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[tuple[float, float, dict[str, Any]]] = []
    for match in frame["matches"]:
        center = match["center"]
        previous_center = previous["center"]
        gap = math.hypot(center[0] - previous_center[0], center[1] - previous_center[1])
        if gap <= 90.0:
            candidates.append((float(match["total"]) - gap / 360.0, gap, match))
    if not candidates:
        return None
    _, gap, selected = max(candidates, key=lambda item: item[0])
    if float(selected["total"]) < 0.24 or float(selected["histogram_similarity"]) < 0.34:
        return None
    return {**selected, "track_step_distance": round(gap, 3)}


def _track_group(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available = [
        (frame_index, match)
        for frame_index, frame in enumerate(group)
        for match in frame["matches"]
    ]
    if not available:
        return []
    anchor_index, anchor = max(available, key=lambda item: float(item[1]["total"]))
    selected: dict[int, dict[str, Any]] = {
        anchor_index: {**anchor, "track_step_distance": 0.0, "anchor": True}
    }
    previous = selected[anchor_index]
    for index in range(anchor_index + 1, len(group)):
        match = _choose_neighbor(group[index], previous)
        if match is not None:
            selected[index] = match
            previous = match
    previous = selected[anchor_index]
    for index in range(anchor_index - 1, -1, -1):
        match = _choose_neighbor(group[index], previous)
        if match is not None:
            selected[index] = match
            previous = match
    output: list[dict[str, Any]] = []
    for index, match in sorted(selected.items()):
        if float(match["total"]) < 0.34 and float(match["track_step_distance"]) > 28.0:
            continue
        if float(match["total"]) < 0.40 and not (
            float(match["total"]) >= 0.34
            and float(match["histogram_similarity"]) >= 0.65
        ):
            continue
        output.append(
            {
                "window_id": group[index]["window_id"],
                "stage": group[index]["stage"],
                "timestamp_seconds": float(group[index]["timestamp_seconds"]),
                "frame_number": int(group[index]["frame_number"]),
                "candidate_index": int(match["candidate_index"]),
                "center": match["center"],
                "identity_score": float(match["total"]),
                "object_score": float(match["object_score"]),
                "histogram_similarity": float(match["histogram_similarity"]),
                "orb_similarity": float(match["orb_similarity"]),
                "track_step_distance": float(match["track_step_distance"]),
                "crop_path": match["crop_path"],
                **match["bridge"],
            }
        )
    return output


def _candidate_object_score(candidate: dict[str, Any]) -> float:
    """Rank a current-frame switch candidate without a fixed video template."""
    # Red posts and green terminal blocks are part of the switch assembly, so
    # they are positive support cues rather than clutter penalties.  The
    # battery-like term remains a soft negative cue for oversized trays.
    return (
        float(candidate["score"])
        + 0.08 * float(candidate.get("contact_pair_score") or 0.0)
        - 0.15 * float(candidate.get("battery_like_score") or 0.0)
    )


def _match_group_candidates(
    group: list[dict[str, Any]], seed: dict[str, Any]
) -> list[dict[str, Any]]:
    """Match one temporal group against a seed from the same group."""
    seed_descriptor = _descriptor(seed["crop_path"])
    matched_frames: list[dict[str, Any]] = []
    for frame in group:
        matches: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(frame.get("candidates", []), start=1):
            descriptor = _descriptor(candidate["crop_path"])
            histogram_similarity = max(
                0.0,
                float(
                    cv2.compareHist(
                        seed_descriptor["histogram"],
                        descriptor["histogram"],
                        cv2.HISTCMP_CORREL,
                    )
                ),
            )
            orb_similarity = _orb_similarity(seed_descriptor["orb"], descriptor["orb"])
            edge_similarity = max(
                0.0, float(np.dot(seed_descriptor["edge"], descriptor["edge"]))
            )
            size_ratio = math.exp(
                -abs(
                    math.log(
                        max(float(candidate["size"][0] * candidate["size"][1]), 1.0)
                        / max(float(seed["size"][0] * seed["size"][1]), 1.0)
                    )
                )
            )
            adjusted_object_score = max(_candidate_object_score(candidate), 0.0)
            total = (
                0.38 * adjusted_object_score
                + 0.20 * histogram_similarity
                + 0.18 * orb_similarity
                + 0.12 * edge_similarity
                + 0.12 * size_ratio
            )
            matches.append(
                {
                    "candidate_index": candidate_index,
                    "total": round(total, 4),
                    "object_score": round(adjusted_object_score, 4),
                    "histogram_similarity": round(histogram_similarity, 4),
                    "orb_similarity": round(orb_similarity, 4),
                    "edge_similarity": round(edge_similarity, 4),
                    "size_similarity": round(size_ratio, 4),
                    "bridge": candidate["bridge"],
                    "center": candidate["center"],
                    "crop_path": candidate["crop_path"],
                }
            )
        matches.sort(key=lambda item: float(item["total"]), reverse=True)
        matched_frames.append(
            {
                **{
                    key: frame[key]
                    for key in ("window_id", "stage", "timestamp_seconds", "frame_number")
                },
                "matches": matches,
            }
        )
    return matched_frames


def cluster_threshold(observations: list[dict[str, Any]]) -> tuple[float, list[float]]:
    """Fit deterministic one-dimensional two-means on current-run geometry."""
    values = np.asarray(
        [float(item["bridge_score"]) for item in observations], dtype=np.float64
    )
    if len(values) < 4:
        return 0.80, []
    centers = np.asarray([np.quantile(values, 0.25), np.quantile(values, 0.75)])
    for _ in range(50):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for label in (0, 1):
            members = values[labels == label]
            if len(members):
                updated[label] = float(np.mean(members))
        if np.allclose(updated, centers, atol=1e-6):
            break
        centers = updated
    ordered = sorted(float(value) for value in centers)
    if ordered[1] - ordered[0] < 0.18 or ordered[1] < 0.70:
        return 0.80, ordered
    return float(np.mean(ordered)), ordered


def _smooth_states(observations: list[dict[str, Any]], threshold: float) -> None:
    for group in _groups(observations, gap_seconds=1.0):
        raw = np.asarray([float(item["bridge_score"]) for item in group])
        for index, item in enumerate(group):
            left = max(0, index - 2)
            right = min(len(group), index + 3)
            smoothed = float(np.median(raw[left:right]))
            item["smoothed_bridge_score"] = round(smoothed, 4)
            item["state"] = "closed" if smoothed >= threshold else "open"


def _annotate_closed_persistence(
    observations: list[dict[str, Any]], max_gap_seconds: float = 0.5
) -> None:
    """Record temporal support so brief occluders cannot become closed switches."""
    for item in observations:
        item["closed_persistence_count"] = 0
        item["closed_persistence_duration_seconds"] = 0.0
    for group in _groups(observations, gap_seconds=max_gap_seconds):
        run: list[dict[str, Any]] = []

        def flush() -> None:
            if not run:
                return
            count = len(run)
            duration = max(
                0.0,
                float(run[-1]["timestamp_seconds"])
                - float(run[0]["timestamp_seconds"]),
            )
            for member in run:
                member["closed_persistence_count"] = count
                member["closed_persistence_duration_seconds"] = round(duration, 4)

        for item in group:
            if item.get("state") != "closed":
                flush()
                run = []
                continue
            if run and (
                float(item["timestamp_seconds"])
                - float(run[-1]["timestamp_seconds"])
                > max_gap_seconds
            ):
                flush()
                run = []
            run.append(item)
        flush()


def analyze_candidate_records(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the switch from current-run candidates and classify its state."""
    all_candidates = [
        candidate
        for frame in frames
        for candidate in frame.get("candidates", [])
        if isinstance(candidate.get("crop_path"), str)
    ]
    if not all_candidates:
        return {
            "state_threshold": 0.80,
            "state_cluster_centers": [],
            "tracked_observation_count": 0,
            "observations": [],
        }
    seed = max(all_candidates, key=_candidate_object_score)
    observations: list[dict[str, Any]] = []
    for group in _groups(frames):
        group_candidates = [
            candidate
            for frame in group
            for candidate in frame.get("candidates", [])
            if isinstance(candidate.get("crop_path"), str)
        ]
        if not group_candidates:
            continue
        group_seed = max(group_candidates, key=_candidate_object_score)
        matched_frames = _match_group_candidates(group, group_seed)
        observations.extend(_track_group(matched_frames))
    observations.sort(key=lambda item: (item["window_id"], item["timestamp_seconds"]))
    threshold, centers = cluster_threshold(observations)
    _smooth_states(observations, threshold)
    _annotate_closed_persistence(observations)
    return {
        "seed_crop_path": seed["crop_path"],
        "tracking_mode": "current_temporal_group_seed_v2",
        "state_threshold_source": (
            "current_video_two_cluster_midpoint"
            if len(centers) == 2
            and centers[1] - centers[0] >= 0.18
            and centers[1] >= 0.70
            else "geometry_fallback"
        ),
        "state_threshold": round(threshold, 4),
        "state_cluster_centers": [round(value, 4) for value in centers],
        "tracked_observation_count": len(observations),
        "open_observation_count": sum(item["state"] == "open" for item in observations),
        "closed_observation_count": sum(item["state"] == "closed" for item in observations),
        "observations": observations,
    }


def save_contact_sheet(observations: list[dict[str, Any]], output: Path) -> None:
    chosen = observations[:: max(1, len(observations) // 30)]
    cells: list[np.ndarray] = []
    for item in chosen:
        image = cv2.imread(item["crop_path"])
        if image is None:
            continue
        image = cv2.resize(image, (320, 160), interpolation=cv2.INTER_CUBIC)
        color = (0, 190, 0) if item["state"] == "open" else (0, 0, 220)
        cv2.rectangle(image, (1, 1), (318, 158), color, 4)
        label = f"{item['timestamp_seconds']:.1f}s {item['state']}  b={item['bridge_score']:.2f}  id={item['identity_score']:.2f}"
        cv2.rectangle(image, (3, 132), (317, 157), (20, 20, 20), -1)
        cv2.putText(
            image,
            label,
            (9, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(image)
    if not cells:
        return
    columns = 4
    blank = np.full_like(cells[0], 245)
    while len(cells) % columns:
        cells.append(blank)
    sheet = np.vstack(
        [
            np.hstack(cells[index : index + columns])
            for index in range(0, len(cells), columns)
        ]
    )
    cv2.imwrite(str(output), sheet)
