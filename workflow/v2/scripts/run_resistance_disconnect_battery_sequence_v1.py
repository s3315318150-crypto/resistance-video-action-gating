#!/usr/bin/env python3
"""Evaluate rubric 8 by reconstructing a battery-terminal rewire episode.

The upstream seven-stage model supplies ``circuit_rewiring`` intervals.  When
it misses a short rewire and repeats ``recording_1`` or ``measurement_1``, the
short gap between the repeated stages is also searched as an independent
recovery episode.  Each episode retrieves evidence for the battery holder and
knife switch, asks Qwen for frame-level observations, and delegates the binary
decision to ``resistance_disconnect_battery_sequence_core``.  The model is
never asked to emit the final score.

The target operation is a lead relocation on a fixed two-cell holder::

    T0 -- cell 1 -- T1 -- cell 2 -- T2

``T0-T2`` must become ``T0-T1`` or ``T1-T2``.  Physical battery removal is
not required and is not used as the primary signal.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - the project fallback is tested elsewhere
    OpenAI = None  # type: ignore[assignment,misc]

from resistance_disconnect_battery_sequence_core import aggregate_episodes, classify_relocation, normalize_terminals


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "resistance_disconnect_battery_sequence_v3_dynamic_roi"
DEFAULT_API_BASE_URL = os.getenv("QWEN_API_BASE_URL", "").strip()
DEFAULT_API_TOKEN = os.getenv("QWEN_API_TOKEN", "").strip()
DEFAULT_MODEL = os.getenv("QWEN_MODEL", "qwen").strip() or "qwen"
SCHEMA_ID = "resistance_7stage_no_battery_v2"
ALGORITHM_ID = "resistance_disconnect_battery_sequence_v3_dynamic_roi"
RECOVERY_STAGE_IDS = ("recording_1", "measurement_1")
RECOVERY_MIN_GAP_SECONDS = 0.5
RECOVERY_MAX_GAP_SECONDS = 45.0

# The direct-contact verifier is deliberately narrower than the topology
# summary.  Keeping these values in one place makes the evidence contract
# explicit and lets replay/tests use the same selection policy.
DIRECT_CONTACT_VERIFIER_VERSION = "resistance_disconnect_battery_direct_contact.v2"
DIRECT_CONTACT_MARGIN_SECONDS = 5.0
DIRECT_CONTACT_MAX_FRAMES = 12
TERMINAL_PAIR_VERIFIER_VERSION = "resistance_disconnect_battery_terminal_pair.v1"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def video_id_from_name(value: str) -> str:
    name = Path(value).name
    match = re.match(r"(\d+)(?:_|$)", name)
    return match.group(1) if match else Path(name).stem


def discover_video(source_name: str, video_root: Path) -> Path:
    root = video_root.expanduser().resolve()
    relative = Path(Path(source_name).name)
    source = (root / relative).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise FileNotFoundError(f"Source video is missing under the configured root: {relative}")
    return source


def stage_intervals(document: Mapping[str, Any]) -> dict[str, list[list[float]]]:
    values: dict[str, list[list[float]]] = {}
    raw = document.get("observed_stage_intervals")
    if not isinstance(raw, list):
        raw = [
            item
            for item in document.get("timeline_segments", [])
            if isinstance(item, Mapping) and item.get("kind") == "observed_stage"
        ]
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        stage = str(item.get("stage") or "")
        start = finite(item.get("start_seconds"))
        end = finite(item.get("end_seconds"))
        if stage and start is not None and end is not None and start <= end:
            values.setdefault(stage, []).append([start, end])
    return values


def intervals_overlap(left: list[float], right: list[float]) -> bool:
    """Return whether two closed time intervals overlap or touch."""
    return max(left[0], right[0]) <= min(left[1], right[1])


R8_TIME_MODES = {
    "rewiring_recovery",
    "wiring_transition",
    "broad_transition_search",
}


def episode_candidates(
    intervals: Mapping[str, list[list[float]]],
    time_mode: str = "rewiring_recovery",
) -> list[dict[str, Any]]:
    """Build independent rewire candidates from stage evidence.

    A second occurrence of the same first-recording or first-measurement stage
    means the upstream state machine may have skipped a brief intervening
    rewire.  Its short gap is searched only when it does not overlap an already
    supplied circuit_rewiring interval.
    """
    if time_mode not in R8_TIME_MODES:
        raise ValueError(f"unsupported R8 time mode: {time_mode}")

    candidates: list[dict[str, Any]] = []
    accepted_intervals: list[list[float]] = []

    def add_stage_intervals(
        stage: str,
        *,
        episode_kind: str,
        candidate_source: str,
    ) -> None:
        for value in intervals.get(stage, []):
            interval = list(value)
            if len(interval) != 2 or interval[0] > interval[1]:
                continue
            if time_mode == "broad_transition_search" and any(
                intervals_overlap(interval, previous) for previous in accepted_intervals
            ):
                continue
            candidates.append(
                {
                    "interval": interval,
                    "episode_kind": episode_kind,
                    "candidate_source": candidate_source,
                    "recovery_anchor": None,
                }
            )
            accepted_intervals.append(interval)

    if time_mode == "rewiring_recovery":
        add_stage_intervals(
            "circuit_rewiring",
            episode_kind="rewire",
            candidate_source="stage_circuit_rewiring",
        )
    elif time_mode == "wiring_transition":
        add_stage_intervals(
            "circuit_wiring",
            episode_kind="wiring",
            candidate_source="stage_circuit_wiring",
        )
    else:
        add_stage_intervals(
            "circuit_wiring",
            episode_kind="wiring",
            candidate_source="stage_circuit_wiring",
        )
        add_stage_intervals(
            "circuit_rewiring",
            episode_kind="rewire",
            candidate_source="stage_circuit_rewiring",
        )

    if time_mode in {"rewiring_recovery", "broad_transition_search"}:
        for stage in RECOVERY_STAGE_IDS:
            stage_values = sorted(
                (list(value) for value in intervals.get(stage, [])),
                key=lambda value: (value[0], value[1]),
            )
            for before, after in zip(stage_values, stage_values[1:]):
                gap = [before[1], after[0]]
                gap_seconds = gap[1] - gap[0]
                if not RECOVERY_MIN_GAP_SECONDS <= gap_seconds <= RECOVERY_MAX_GAP_SECONDS:
                    continue
                if any(intervals_overlap(gap, interval) for interval in accepted_intervals):
                    continue
                candidates.append(
                    {
                        "interval": gap,
                        "episode_kind": "recovery",
                        "candidate_source": "repeated_stage_gap_recovery",
                        "recovery_anchor": {
                            "stage": stage,
                            "before_interval_seconds": before,
                            "after_interval_seconds": after,
                        },
                    }
                )
                accepted_intervals.append(gap)
    return sorted(candidates, key=lambda item: (item["interval"][0], item["interval"][1], item["episode_kind"]))


def video_duration(path: Path) -> tuple[float, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            raise RuntimeError(f"Invalid video metadata for {path}: fps={fps}, frames={count}")
        return fps, count / fps, count
    finally:
        capture.release()


def expand_roi(roi: tuple[float, float, float, float], margin: float = 0.08) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = roi
    width, height = x2 - x1, y2 - y1
    return (
        max(0.0, x1 - width * margin),
        max(0.0, y1 - height * margin),
        min(1.0, x2 + width * margin),
        min(1.0, y2 + height * margin),
    )


def crop_normalized(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi
    left = max(0, min(width - 2, round(x1 * width)))
    top = max(0, min(height - 2, round(y1 * height)))
    right = max(left + 2, min(width, round(x2 * width)))
    bottom = max(top + 2, min(height, round(y2 * height)))
    return frame[top:bottom, left:right]


def resize_longest(frame: np.ndarray, longest: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, float(longest) / max(height, width))
    if scale >= 1.0:
        return frame.copy()
    return cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def banner(frame: np.ndarray, label: str, longest: int = 1600) -> np.ndarray:
    image = resize_longest(frame, longest)
    height, width = image.shape[:2]
    bar_height = max(28, round(height * 0.09))
    result = cv2.copyMakeBorder(image, 0, bar_height, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(result, label, (10, height + bar_height - 9), cv2.FONT_HERSHEY_SIMPLEX, max(0.45, width / 1500), (255, 255, 255), 1, cv2.LINE_AA)
    return result


@dataclass
class FrameRecord:
    frame_id: str
    frame_number: int
    timestamp_seconds: float
    panorama_path: str
    battery_path: str
    sharpness: float
    motion: float
    battery_motion: float
    battery_roi: tuple[float, float, float, float] | None = None
    roi_mode: str = "fallback"
    fallback_battery_path: str | None = None
    localization_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_number": self.frame_number,
            "timestamp_seconds": self.timestamp_seconds,
            "panorama_path": self.panorama_path,
            "battery_path": self.battery_path,
            "sharpness": round(self.sharpness, 3),
            "motion": round(self.motion, 3),
            "battery_motion": round(self.battery_motion, 3),
            "battery_roi": list(self.battery_roi) if self.battery_roi else None,
            "roi_mode": self.roi_mode,
            "fallback_battery_path": self.fallback_battery_path,
            "localization_path": self.localization_path,
        }


class BatteryRoiTracker:
    """Track a live panorama detection through adjacent frames."""

    def __init__(self, initial_roi: tuple[float, float, float, float] | None = None) -> None:
        self.last_roi = expand_roi(initial_roi, 0.12) if initial_roi else None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.orb = cv2.ORB_create(nfeatures=1800)

    def locate(self, frame: np.ndarray) -> tuple[tuple[float, float, float, float] | None, str]:
        target = resize_longest(frame, 960)
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        target_small = cv2.resize(gray, (480, 270), interpolation=cv2.INTER_AREA)
        keypoints, descriptors = self.orb.detectAndCompute(target_small, None)
        mode = "dynamic_seed" if self.last_roi else "panorama_fallback"
        candidate = self.last_roi
        if self.previous_descriptors is not None and descriptors is not None and self.previous_keypoints is not None and keypoints:
            matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.previous_descriptors, descriptors, k=2)
            good = [first for first, second in matches if first.distance < 0.75 * second.distance]
            if len(good) >= 6:
                source = np.float32([self.previous_keypoints[item.queryIdx].pt for item in good])
                destination = np.float32([keypoints[item.trainIdx].pt for item in good])
                affine, mask = cv2.estimateAffinePartial2D(source, destination, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                inliers = int(mask.sum()) if mask is not None else 0
                if affine is not None and inliers >= 5 and candidate is not None:
                    points = np.float32([[[candidate[0] * 480, candidate[1] * 270], [candidate[2] * 480, candidate[1] * 270], [candidate[2] * 480, candidate[3] * 270], [candidate[0] * 480, candidate[3] * 270]]])
                    projected = cv2.transform(points, affine)[0]
                    left = max(0.0, float(projected[:, 0].min()) / 480)
                    top = max(0.0, float(projected[:, 1].min()) / 270)
                    right = min(1.0, float(projected[:, 0].max()) / 480)
                    bottom = min(1.0, float(projected[:, 1].max()) / 270)
                    area = (right - left) * (bottom - top)
                    if right - left >= 0.04 and bottom - top >= 0.04 and area <= 0.65:
                        candidate = (left, top, right, bottom)
                        self.last_roi = candidate
                        mode = "tracked_sequential"
        self.previous_keypoints = keypoints
        self.previous_descriptors = descriptors
        return candidate, mode


def target_frame_numbers(start: float, end: float, fps: float, sample_fps: float) -> list[int]:
    if end < start or sample_fps <= 0:
        return []
    first, last = round(start * fps), round(end * fps)
    step = fps / sample_fps
    values: list[int] = []
    index = 0
    while True:
        number = round(first + index * step)
        if number > last:
            break
        if not values or number != values[-1]:
            values.append(number)
        index += 1
    return values


def extract_frames(
    video_path: Path,
    video_id: str,
    episode_id: str,
    start: float,
    end: float,
    sample_fps: float,
    output_dir: Path,
    initial_roi: tuple[float, float, float, float] | None = None,
    *,
    save_localization_images: bool = False,
) -> list[FrameRecord]:
    fps, duration, _ = video_duration(video_path)
    start = max(0.0, min(duration, start))
    end = max(start, min(duration, end))
    targets = target_frame_numbers(start, end, fps, sample_fps)
    if not targets:
        return []
    frames_dir = output_dir / "frames"
    battery_dir = output_dir / "battery_roi"
    fallback_dir = output_dir / "battery_reference_roi"
    localization_dir = output_dir / "localization_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    battery_dir.mkdir(parents=True, exist_ok=True)
    fallback_dir.mkdir(parents=True, exist_ok=True)
    if save_localization_images:
        localization_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    records: list[FrameRecord] = []
    previous_small: np.ndarray | None = None
    previous_battery: np.ndarray | None = None
    tracker = BatteryRoiTracker(initial_roi)
    target_index = 0
    current = targets[0]
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, targets[0])
        while target_index < len(targets):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Decode failed at frame {current} in {video_path}")
            if current == targets[target_index]:
                timestamp = round(current / fps, 3)
                frame_id = f"{video_id}_{episode_id}_f{current:08d}"
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
                motion = float(np.mean(cv2.absdiff(small, previous_small))) if previous_small is not None else 0.0
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                tracked_roi, roi_mode = tracker.locate(frame)
                battery = crop_normalized(frame, tracked_roi) if tracked_roi is not None else frame
                battery_gray = cv2.cvtColor(battery, cv2.COLOR_BGR2GRAY)
                battery_small = cv2.resize(battery_gray, (96, 54), interpolation=cv2.INTER_AREA)
                battery_motion = float(np.mean(cv2.absdiff(battery_small, previous_battery))) if previous_battery is not None else 0.0
                panorama_path = frames_dir / f"{frame_id}.jpg"
                battery_path = battery_dir / f"{frame_id}.jpg"
                fallback_path = fallback_dir / f"{frame_id}.jpg"
                localization_path = localization_dir / f"{frame_id}.jpg"
                context = frame
                if not cv2.imwrite(str(panorama_path), banner(frame, f"FRAME ID={frame_id} | VIDEO T={timestamp:.3f}s | PANORAMA", 1600), [cv2.IMWRITE_JPEG_QUALITY, 93]):
                    raise RuntimeError(f"Could not write {panorama_path}")
                if not cv2.imwrite(str(battery_path), banner(battery, f"FRAME ID={frame_id} | VIDEO T={timestamp:.3f}s | BATTERY ROI", 1200), [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Could not write {battery_path}")
                if not cv2.imwrite(str(fallback_path), banner(context, f"FRAME ID={frame_id} | VIDEO T={timestamp:.3f}s | PANORAMA CONTEXT", 1200), [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Could not write {fallback_path}")
                saved_localization_path = None
                if save_localization_images:
                    if not cv2.imwrite(str(localization_path), resize_longest(frame, 1600), [cv2.IMWRITE_JPEG_QUALITY, 94]):
                        raise RuntimeError(f"Could not write {localization_path}")
                    saved_localization_path = str(localization_path.resolve())
                records.append(FrameRecord(frame_id, current, timestamp, str(panorama_path.resolve()), str(battery_path.resolve()), sharpness, motion, battery_motion, tracked_roi, roi_mode, str(fallback_path.resolve()), saved_localization_path))
                previous_small, previous_battery = small, battery_small
                target_index += 1
            current += 1
    finally:
        capture.release()
    return records


def select_uniform(records: list[FrameRecord], maximum: int) -> list[FrameRecord]:
    if len(records) <= maximum:
        return records
    indexes = np.linspace(0, len(records) - 1, maximum).round().astype(int)
    return [records[int(index)] for index in indexes]


def paired_image(record: FrameRecord, output_path: Path) -> Path:
    panorama = cv2.imread(record.panorama_path)
    battery = cv2.imread(record.battery_path)
    fallback = cv2.imread(record.fallback_battery_path or record.battery_path)
    if panorama is None or battery is None or fallback is None:
        raise RuntimeError(f"Cannot read paired evidence for {record.frame_id}")
    tile_height = 506
    panorama = cv2.resize(panorama, (720, tile_height), interpolation=cv2.INTER_AREA)
    battery = cv2.resize(battery, (360, tile_height), interpolation=cv2.INTER_AREA)
    fallback = cv2.resize(fallback, (360, tile_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((tile_height + 42, panorama.shape[1] + battery.shape[1] + fallback.shape[1], 3), 28, dtype=np.uint8)
    canvas[42:, : panorama.shape[1]] = panorama
    canvas[42:, panorama.shape[1] : panorama.shape[1] + battery.shape[1]] = battery
    canvas[42:, panorama.shape[1] + battery.shape[1] :] = fallback
    cv2.putText(canvas, "PANORAMA", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "TRACKED ROI", (panorama.shape[1] + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "PANORAMA CONTEXT", (panorama.shape[1] + battery.shape[1] + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Cannot write paired evidence: {output_path}")
    return output_path.resolve()


def make_contact_sheet(records: list[FrameRecord], output_path: Path, mode: str, columns: int = 4) -> Path:
    if not records:
        raise ValueError("Cannot make a contact sheet without frames")
    tiles: list[np.ndarray] = []
    for record in records:
        if mode == "paired":
            panorama = cv2.imread(record.panorama_path)
            battery = cv2.imread(record.battery_path)
            fallback = cv2.imread(record.fallback_battery_path or record.battery_path)
            if panorama is None or battery is None or fallback is None:
                raise RuntimeError(f"Cannot read paired evidence for {record.frame_id}")
            height = 220
            left = cv2.resize(panorama, (240, height), interpolation=cv2.INTER_AREA)
            middle = cv2.resize(battery, (120, height), interpolation=cv2.INTER_AREA)
            right = cv2.resize(fallback, (120, height), interpolation=cv2.INTER_AREA)
            image = np.concatenate([left, middle, right], axis=1)
            tile_width = 480
        else:
            path = Path(record.panorama_path if mode == "panorama" else record.battery_path)
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Cannot read evidence image: {path}")
            image = resize_longest(image, 480)
            image = cv2.resize(image, (480, max(1, round(image.shape[0] * 480 / image.shape[1]))), interpolation=cv2.INTER_AREA)
            tile_width = 480
        label_height = 28
        tile = np.full((image.shape[0] + label_height, tile_width, 3), 245, dtype=np.uint8)
        tile[: image.shape[0], : image.shape[1]] = image
        cv2.putText(tile, f"{record.frame_id}  t={record.timestamp_seconds:.3f}s", (5, image.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(tile)
    tile_height = max(tile.shape[0] for tile in tiles)
    rows = math.ceil(len(tiles) / columns)
    canvas = np.full((rows * tile_height, columns * 480, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[row * tile_height : row * tile_height + tile.shape[0], column * 480 : column * 480 + tile.shape[1]] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Cannot write contact sheet: {output_path}")
    return output_path.resolve()


def split_chunks(records: list[FrameRecord], maximum: int = 12) -> list[list[FrameRecord]]:
    return [records[index : index + maximum] for index in range(0, len(records), maximum)]


def json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    return value


def image_data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def qwen_call(client: Any, model: str, prompt: str, images: list[Path]) -> tuple[dict[str, Any] | None, str | None, str]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": image_data_url(path)}} for path in images)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=5000,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as exc:  # The caller records the failed evidence packet and continues.
        return None, f"{type(exc).__name__}: {exc}", ""
    raw = str(response.choices[0].message.content or "")
    try:
        return json_object(raw), None, raw
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}", raw


def screen_prompt(records: list[FrameRecord], episode: Mapping[str, Any]) -> str:
    ids = ", ".join(record.frame_id for record in records)
    return f"""你正在取证伏安法实验的评价8。图片按时间升序排列，每张图包含全景或电池盒局部，图片文字中的 FRAME ID 是唯一合法引用。可用 FRAME ID：{ids}。

只做候选检索，不输出 pass/fail，不依据实验常识补造动作。
目标换接不是把电池拿下来，而是橙色长条电池盒上的外接导线从两个外侧端子之一移到中间抽头：
T0--电池1--T1--电池2--T2；目标状态是 T0-T2 变成 T0-T1 或 T1-T2。
刀闸开关必须看刀片与固定夹口的间隙，不能只看黑色手柄方向。
排除滑动变阻器、电表、普通导线和整理拆线。

返回单个 JSON：
{{"battery_candidate_frame_ids":[],"switch_candidate_frame_ids":[],"cleanup_like_frame_ids":[],"battery_object_visible_frame_ids":[],"notes":[]}}
所有数组元素必须原样来自可用 FRAME ID。episode={episode.get('episode_id')}，核心段={episode.get('core_interval_seconds')}。"""


def localization_prompt(records: list[FrameRecord], episode: Mapping[str, Any]) -> str:
    mapping = ", ".join(
        f"image {index}={record.frame_id}" for index, record in enumerate(records, start=1)
    )
    return f"""你只负责在当前视频的独立全景帧中定位电池盒和三个电源端子，不做评分。输入不是联系表，每张图片都是一个完整原始全景，映射为：{mapping}。

电池盒是带一节或两节圆柱电池、金属接触片/槽位及外接端子的长条器材。排除滑动变阻器、刀闸开关、电流表、电压表和定值电阻。
对每张能确认电池盒的图片返回一个框，bbox_normalized=[x1,y1,x2,y2] 必须相对于该张独立全景图片，范围 0..1，并覆盖完整电池盒和可见端子。不得返回联系表格子坐标，不得沿用其他图片或其他视频的坐标；看不清就省略。

只输出单个 JSON：
{{"battery_detections":[{{"frame_id":"合法ID","bbox_normalized":[0.0,0.0,1.0,1.0],"confidence":0.0,"evidence":"直接可见特征"}}]}}
episode={episode.get('episode_id')}。"""


def facts_prompt(records: list[FrameRecord], episode: Mapping[str, Any]) -> str:
    ids = ", ".join(record.frame_id for record in records)
    return f"""你是逐帧视觉观察器，不是评分器。按图片时间顺序观察同一个重新连线 episode。每张配对图左侧是全景，右侧是同一时间的电池盒 ROI。可用 FRAME ID 只有：{ids}。

电池盒拓扑固定写成 T0(左外端子)-电池1-T1(中间抽头)-电池2-T2(右外端子)。只在直接看清外接导线实际接触的端子时填写 battery_terminals；看不清就 null。battery_terminals 表示外接导线占用的端子对，不是电池数量。确认的目标变化是稳定 T0,T2 到稳定 T0,T1 或 T1,T2。不要把手接近、遮挡或同端拔插写成完成变化。
刀闸状态必须依据刀片是否与固定夹口有间隙：open=有间隙，closed=刀片落入夹口，不能仅凭手柄方向猜测。

对每个 FRAME ID 返回一条 observation，不能遗漏或新增 ID。字段固定为：
{{"observations":[{{"frame_id":"...","battery_object":"confirmed|rejected|unknown","battery_terminals":["T0","T1","T2"]|null,"terminal_state_stable":true|false,"direct_battery_contact":true|false,"terminal_action":"none|disconnect|reconnect|relocate|uncertain","switch_state":"open|closed|unknown","switch_action":"opening|closing|none|uncertain","confidence":0.0,"evidence":"简短直接观察"}}]}}
    不要输出 decision、pass、fail 或 predicted_score。episode={episode.get('episode_id')}，扩展段={episode.get('expanded_interval_seconds')}。"""


def structured_summary_prompt(records: list[FrameRecord], episode: Mapping[str, Any]) -> str:
    ids = ", ".join(record.frame_id for record in records)
    return f"""只做视觉事实提取，不做评价。以下是同一个 episode 的按时间排序图片；每张图依次给出全景、跟踪 ROI、原始参考 ROI。ROI 可能跟踪失败，必须以全景中的真实器材为准。合法 FRAME ID 只有：{ids}。

先辨认对象：电池盒必须是画面中清楚带两节圆柱电池、金属接触片或槽位的长条盒。滑动变阻器、刀闸、电表、定值电阻或其他橙色器材都不是电池盒。只有手或插头直接接触该电池盒固定端子时，才可填写 direct_contact_frame_ids。
若确认电池盒，三个电气抽头沿串联方向定义为 T0、T1、T2。外接导线端子对可以是 [T0,T2]、[T0,T1]、[T1,T2] 或 null。不要预设发生了改变，不要根据两节实体电池反推端子，必须看到导线插头与固定端子的关系。若只是操作其他器材，将 battery_object 写 rejected、terminal_rewire.completed 写 false、端子对写 null。
刀闸 open 必须看到刀片与固定夹口有间隙；closed 必须看到刀片落入夹口。

核心重新连线区间是 {episode.get('core_interval_seconds')} 秒。battery_before 必须选换接前最后一个清楚稳定的 T0-T2；terminal_rewire 的起止图必须位于核心区间附近；battery_after 必须选换接完成后第一个清楚稳定的一节配置；closed_after_frame_id 必须严格晚于换接完成，closed_during_frame_ids 只能列换接尚未完成期间的闭合证据，不能把正常的后续闭合列入其中。

端子字段必须自洽：若 battery_before=[T0,T2] 且 terminal_rewire 写 from_terminal=T2、to_terminal=T1、completed=true，则 battery_after 必须是 [T0,T1]；若 from_terminal=T0、to_terminal=T1，则 battery_after 必须是 [T1,T2]。如果稳定后态仍与前态相同，就不能把 terminal_rewire.completed 写成 true。输出前必须用这一规则复核一次，但仍只依据图片中实际可见的插头位置填写。

请比较前后稳定画面，并只输出这个 JSON（字段不可改名）：
{{
  "battery_object": "confirmed|rejected|unknown",
  "direct_contact_frame_ids": [],
  "battery_before": {{"frame_id": "合法ID或null", "terminals": ["T0","T2"] 或 null, "stable": true 或 false}},
  "battery_after": {{"frame_id": "合法ID或null", "terminals": ["T0","T2"] 或 ["T0","T1"] 或 ["T1","T2"] 或 null, "stable": true 或 false}},
  "terminal_rewire": {{"start_frame_id": "合法ID或null", "end_frame_id": "合法ID或null", "from_terminal": "T0或T2或null", "to_terminal": "T1或null", "completed": true 或 false}},
  "switch": {{"open_before_frame_id": "合法ID或null", "closed_after_frame_id": "合法ID或null", "closed_during_frame_ids": []}}
}}
    不能输出 pass、fail、decision 或任何实验常识推断。episode={episode.get('episode_id')}。"""


def validate_structured_summary(summary: Mapping[str, Any], records: list[FrameRecord], episode: Mapping[str, Any]) -> list[str]:
    by_id = {record.frame_id: record.timestamp_seconds for record in records}
    errors: list[str] = []

    def section(name: str) -> Mapping[str, Any]:
        value = summary.get(name)
        if not isinstance(value, Mapping):
            errors.append(f"{name}_not_object")
            return {}
        return value

    before = section("battery_before")
    after = section("battery_after")
    rewire = section("terminal_rewire")
    switch = section("switch")
    if summary.get("battery_object") != "confirmed":
        errors.append("battery_object_not_confirmed")
    contact_ids = summary.get("direct_contact_frame_ids")
    if not isinstance(contact_ids, list) or not contact_ids:
        errors.append("direct_battery_contact_missing")
        contact_ids = []
    for frame_id in contact_ids:
        if not isinstance(frame_id, str) or frame_id not in by_id:
            errors.append("direct_contact_frame_id_invalid")

    ids = {
        "before": before.get("frame_id"),
        "after": after.get("frame_id"),
        "start": rewire.get("start_frame_id"),
        "end": rewire.get("end_frame_id"),
        "open": switch.get("open_before_frame_id"),
        "close": switch.get("closed_after_frame_id"),
    }
    for name, frame_id in ids.items():
        if not isinstance(frame_id, str) or frame_id not in by_id:
            errors.append(f"{name}_frame_id_invalid")
    if before.get("stable") is not True:
        errors.append("before_not_stable")
    if after.get("stable") is not True:
        errors.append("after_not_stable")
    if rewire.get("completed") is not True:
        errors.append("rewire_not_completed")
    relocation = classify_relocation(before.get("terminals"), after.get("terminals"))
    if not relocation["completed"]:
        errors.append("terminal_pair_not_two_to_one")
    if rewire.get("completed") is True:
        moved = relocation.get("moved_lead")
        if not isinstance(moved, Mapping):
            errors.append("rewire_claim_conflicts_with_terminal_pairs")
        elif rewire.get("from_terminal") != moved.get("from") or rewire.get("to_terminal") != moved.get("to"):
            errors.append("rewire_endpoints_conflict_with_terminal_pairs")
    if all(isinstance(frame_id, str) and frame_id in by_id for frame_id in ids.values()):
        times = {name: by_id[str(frame_id)] for name, frame_id in ids.items()}
        if not (times["before"] < times["start"] <= times["end"] <= times["after"]):
            errors.append("battery_event_order_invalid")
        if not (times["open"] < times["start"]):
            errors.append("switch_open_not_before_rewire")
        if not (times["close"] > times["end"]):
            errors.append("switch_close_not_after_rewire")
        core_start, core_end = episode["core_interval_seconds"]
        if times["start"] < core_start - 2.5 or times["end"] > core_end + 2.5:
            errors.append("rewire_outside_core_context")
        valid_contact_times = [by_id[frame_id] for frame_id in contact_ids if isinstance(frame_id, str) and frame_id in by_id]
        if not any(times["start"] <= value <= times["end"] for value in valid_contact_times):
            errors.append("no_direct_contact_during_rewire")
        closed_during = switch.get("closed_during_frame_ids")
        if isinstance(closed_during, list):
            for frame_id in closed_during:
                if not isinstance(frame_id, str) or frame_id not in by_id:
                    errors.append("closed_during_frame_id_invalid")
                elif not (times["start"] <= by_id[frame_id] <= times["end"]):
                    errors.append("closed_during_outside_rewire")
    return sorted(set(errors))


def structured_summary_to_observations(summary: Mapping[str, Any], records: list[FrameRecord]) -> tuple[list[dict[str, Any]], list[str]]:
    by_id = {record.frame_id: record for record in records}
    errors: list[str] = []
    object_state = summary.get("battery_object") if summary.get("battery_object") in {"confirmed", "rejected", "unknown"} else "unknown"
    contact_values = summary.get("direct_contact_frame_ids")
    contact_ids = {item for item in contact_values if isinstance(item, str) and item in by_id} if isinstance(contact_values, list) else set()

    # Contact evidence can authorize the topology only when it falls inside
    # the summary's cited terminal-relocation interval.  A stale citation
    # from a later switch operation must not make the stable terminal pairs
    # usable.
    rewire_bounds = summary.get("terminal_rewire")
    if isinstance(rewire_bounds, Mapping):
        start_id = rewire_bounds.get("start_frame_id")
        end_id = rewire_bounds.get("end_frame_id")
        start_time = by_id[start_id].timestamp_seconds if isinstance(start_id, str) and start_id in by_id else None
        end_time = by_id[end_id].timestamp_seconds if isinstance(end_id, str) and end_id in by_id else None
        if start_time is not None and end_time is not None and start_time <= end_time:
            contact_ids = {
                frame_id
                for frame_id in contact_ids
                if start_time <= by_id[frame_id].timestamp_seconds <= end_time
            }
    object_verified = object_state == "confirmed" and bool(contact_ids)
    observations = {
        record.frame_id: {
            "frame_id": record.frame_id,
            "timestamp_seconds": record.timestamp_seconds,
            "battery_object": "unknown",
            "battery_terminals": None,
            "terminal_state_stable": False,
            "direct_battery_contact": False,
            "terminal_action": "none",
            "terminal_rewire_completed": False,
            "switch_state": "unknown",
            "switch_action": "none",
            "confidence": 0.35,
            "evidence": "",
        }
        for record in records
    }

    def section(name: str) -> Mapping[str, Any]:
        value = summary.get(name)
        return value if isinstance(value, Mapping) else {}

    def assign_frame(frame_id: Any, *, terminals: Any = None, stable: bool = False, action: str = "none", direct: bool = False, rewire_completed: bool = False, switch: str | None = None, evidence: str = "") -> None:
        if not isinstance(frame_id, str) or frame_id == "null" or frame_id not in by_id:
            if frame_id not in (None, "null"):
                errors.append(f"invalid_summary_frame_id:{frame_id}")
            return
        item = observations[frame_id]
        if terminals is not None or action != "none" or direct:
            item["battery_object"] = object_state
        if terminals is not None:
            item["battery_terminals"] = terminals if object_verified else None
            item["terminal_state_stable"] = stable and object_verified
        item["direct_battery_contact"] = direct
        item["terminal_action"] = action
        item["terminal_rewire_completed"] = rewire_completed
        if switch is not None:
            item["switch_state"] = switch
        if evidence:
            item["evidence"] = evidence
        item["confidence"] = 0.8

    before = section("battery_before")
    after = section("battery_after")
    rewire = section("terminal_rewire")
    switch = section("switch")
    assign_frame(before.get("frame_id"), terminals=before.get("terminals"), stable=before.get("stable") is True, evidence="Model-selected stable pre-rewire terminal state.")
    assign_frame(after.get("frame_id"), terminals=after.get("terminals"), stable=after.get("stable") is True, evidence="Model-selected stable post-rewire terminal state.")
    start_id = rewire.get("start_frame_id")
    end_id = rewire.get("end_frame_id")
    assign_frame(start_id, action="disconnect", direct=object_verified and start_id in contact_ids, evidence="Model-selected start of the terminal relocation.")
    completed = rewire.get("completed") is True
    assign_frame(
        end_id,
        action="reconnect",
        direct=object_verified and end_id in contact_ids,
        rewire_completed=completed and object_verified,
        evidence="Model-selected completed terminal reconnection.",
    )
    for frame_id in contact_ids:
        assign_frame(frame_id, action="uncertain", direct=True, evidence="Direct contact with a fixed battery-holder terminal.")
    assign_frame(switch.get("open_before_frame_id"), switch="open", evidence="Model-selected open knife-switch state before rewire.")
    assign_frame(switch.get("closed_after_frame_id"), switch="closed", evidence="Model-selected closed knife-switch state after rewire.")
    closed_during = switch.get("closed_during_frame_ids")
    if isinstance(closed_during, list):
        for frame_id in closed_during:
            assign_frame(frame_id, switch="closed", evidence="Model-selected closed state during rewire.")
    if not completed:
        errors.append("summary_rewire_not_completed")
    # A summary action can be cited at a frame between the two stable states.
    if completed and object_verified and isinstance(start_id, str) and start_id in observations:
        observations[start_id]["terminal_action"] = "relocate"
    return [observations[record.frame_id] for record in records], sorted(set(errors))


def select_direct_contact_records(
    records: list[FrameRecord],
    episode: Mapping[str, Any],
    margin_seconds: float = DIRECT_CONTACT_MARGIN_SECONDS,
    maximum: int = DIRECT_CONTACT_MAX_FRAMES,
) -> list[FrameRecord]:
    """Select only dense battery-ROI frames around the core rewire interval.

    ``records`` is expected to come from the dense extractor.  The verifier is
    intentionally independent from coarse screening candidates: a short hand
    or plug contact can be missed by a candidate selector.  We retain a
    temporally uniform backbone and add the highest local battery-motion frames
    so a brief contact remains represented when the packet is bounded.
    """

    if not records:
        return []
    try:
        core_start, core_end = (float(value) for value in episode["core_interval_seconds"])
    except (KeyError, TypeError, ValueError):
        return []
    try:
        expanded_start, expanded_end = (float(value) for value in episode["expanded_interval_seconds"])
    except (KeyError, TypeError, ValueError):
        expanded_start = min(record.timestamp_seconds for record in records)
        expanded_end = max(record.timestamp_seconds for record in records)
    margin = max(0.0, float(margin_seconds))
    left = max(expanded_start, core_start - margin)
    right = min(expanded_end, core_end + margin)
    candidates = [
        record
        for record in records
        if left <= record.timestamp_seconds <= right
    ]
    candidates.sort(key=lambda record: record.timestamp_seconds)
    if not candidates:
        return []
    if maximum <= 0 or len(candidates) <= maximum:
        return candidates

    # The uniform backbone gives the model temporal context.  Motion-ranked
    # frames preserve short plug/hand interactions that uniform sampling can
    # otherwise skip.  Endpoints and core boundaries are always retained.
    backbone_count = max(1, int(round(maximum * 0.6)))
    backbone = select_uniform(candidates, min(backbone_count, len(candidates)))
    ranked = sorted(
        candidates,
        key=lambda record: (record.battery_motion, record.sharpness),
        reverse=True,
    )[: max(1, maximum - len(backbone))]
    anchors: list[FrameRecord] = list(backbone)
    for timestamp in (left, right, core_start, core_end):
        nearest = min(candidates, key=lambda record: abs(record.timestamp_seconds - timestamp))
        if nearest not in anchors:
            anchors.append(nearest)
    for record in ranked:
        if record not in anchors:
            anchors.append(record)
        if len(anchors) >= maximum:
            break
    return sorted(anchors[:maximum], key=lambda record: record.timestamp_seconds)


def direct_contact_verifier_prompt(records: list[FrameRecord], episode: Mapping[str, Any]) -> str:
    """Build a neutral, frame-by-frame prompt for the narrow contact check."""

    ids = ", ".join(record.frame_id for record in records)
    return f"""你是一个独立的窄范围视觉事实核验器。这里只核验电池盒端子附近的直接接触，不能评分，也不能推断换接是否完成。

输入图片按视频时间升序排列。每张图都是同一时刻的三联证据：左侧全景，中间跟踪 ROI，右侧原始参考 ROI；三部分下方的 FRAME ID 相同。ROI 可能漂移，所以必须用全景和两种局部图交叉确认对象，不要把任何橙色器材自动当成电池盒。可用 FRAME ID：{ids}。

逐帧判断：
1. battery_object=confirmed 只有在同一画面中清楚看见橙色长条电池盒、两节圆柱电池（或两个独立电池槽）以及固定金属端子/弹片；只露出一节、只看见导线、或对象是滑动变阻器、刀闸（带金属刀片）、电表端子、定值电阻等时写 rejected 或 unknown。
2. contact_mode=active 表示这一帧中能直接看到手指、导线插头或金属夹的金属接触端，正在实际贴住或插入上述两节电池盒的固定端子/金属接触片；只有 contact_mode=active 才能令 direct_contact=true。contact_mode=static 表示导线已经接好但当前帧没有正在操作，必须令 direct_contact=false。contact_mode=none/uncertain 也必须令 direct_contact=false。插头接在别的橙色盒、刀闸、变阻器或电表端子一律为 false。仅仅靠近、悬空、导线经过、手遮挡或根据前后帧补造都必须为 false。
3. 不要判断 T0/T1/T2、串联节数、开关状态、动作顺序，也不要预设任何变化。只有当前帧直接显示的接触才可引用。

必须为每一个可用 FRAME ID 返回一条 observation，不能遗漏、重复或新增 ID。只输出一个 JSON 对象，不要 Markdown：
{{
  "battery_object": "confirmed|rejected|unknown",
  "direct_contact_frame_ids": [],
  "observations": [
    {{"frame_id": "合法ID", "battery_object": "confirmed|rejected|unknown", "contact_mode": "active|static|none|uncertain", "direct_contact": true, "evidence": "只写当前帧直接看到的事实"}}
  ]
}}
顶层 direct_contact_frame_ids 必须是 observations 中 direct_contact=true 且 battery_object=confirmed 的 FRAME ID 集合。episode={episode.get('episode_id')}；核心区间={episode.get('core_interval_seconds')}。"""


def validate_direct_contact_verifier_response(
    response: Mapping[str, Any] | None,
    records: list[FrameRecord],
) -> list[str]:
    """Validate the narrow verifier schema without making topology claims."""

    errors: list[str] = []
    known = {record.frame_id for record in records}
    if not isinstance(response, Mapping):
        return ["response_not_object"]
    top_object = response.get("battery_object")
    if top_object not in {"confirmed", "rejected", "unknown"}:
        errors.append("battery_object_invalid")
    raw_observations = response.get("observations")
    if not isinstance(raw_observations, list):
        errors.append("observations_not_list")
        raw_observations = []
    observed_ids: list[str] = []
    direct_from_observations: list[str] = []
    for item in raw_observations:
        if not isinstance(item, Mapping):
            errors.append("observation_not_object")
            continue
        frame_id = item.get("frame_id")
        if not isinstance(frame_id, str) or frame_id not in known or frame_id in observed_ids:
            errors.append("observation_frame_id_invalid_or_duplicate")
            continue
        observed_ids.append(frame_id)
        object_state = item.get("battery_object")
        if object_state not in {"confirmed", "rejected", "unknown"}:
            errors.append(f"observation_battery_object_invalid:{frame_id}")
        contact_mode = item.get("contact_mode")
        if contact_mode not in {"active", "static", "none", "uncertain"}:
            errors.append(f"observation_contact_mode_invalid:{frame_id}")
        direct = item.get("direct_contact")
        if not isinstance(direct, bool):
            # Accept the older spelling when replaying a hand-authored cache,
            # but keep the canonical output as ``direct_contact``.
            direct = item.get("direct_battery_contact")
        if not isinstance(direct, bool):
            errors.append(f"observation_direct_contact_invalid:{frame_id}")
            direct = False
        if direct and object_state != "confirmed":
            errors.append(f"direct_contact_object_not_confirmed:{frame_id}")
        if direct and contact_mode != "active":
            errors.append(f"direct_contact_not_active:{frame_id}")
        if direct and object_state == "confirmed" and contact_mode == "active":
            direct_from_observations.append(frame_id)
    if len(observed_ids) != len(known) or set(observed_ids) != known:
        errors.append("observation_frame_coverage_incomplete")

    raw_direct_ids = response.get("direct_contact_frame_ids")
    if not isinstance(raw_direct_ids, list):
        errors.append("direct_contact_frame_ids_not_list")
        raw_direct_ids = []
    direct_ids: list[str] = []
    for frame_id in raw_direct_ids:
        if not isinstance(frame_id, str) or frame_id not in known or frame_id in direct_ids:
            errors.append("direct_contact_frame_id_invalid_or_duplicate")
            continue
        direct_ids.append(frame_id)
    if set(direct_ids) != set(direct_from_observations):
        errors.append("direct_contact_frame_ids_inconsistent")
    return sorted(set(errors))


def normalize_direct_contact_verifier_response(
    response: Mapping[str, Any] | None,
    records: list[FrameRecord],
) -> tuple[dict[str, Any], list[str]]:
    """Return a conservative normalized verifier result and its errors."""

    errors = validate_direct_contact_verifier_response(response, records)
    known = {record.frame_id: record for record in records}
    raw_observations = response.get("observations") if isinstance(response, Mapping) else []
    by_id: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_observations, list):
        for item in raw_observations:
            if isinstance(item, Mapping) and isinstance(item.get("frame_id"), str):
                frame_id = item["frame_id"]
                if frame_id in known and frame_id not in by_id:
                    by_id[frame_id] = item
    observations: list[dict[str, Any]] = []
    for record in records:
        item = by_id.get(record.frame_id, {})
        object_state = item.get("battery_object")
        if object_state not in {"confirmed", "rejected", "unknown"}:
            object_state = "unknown"
        direct = item.get("direct_contact")
        if not isinstance(direct, bool):
            direct = item.get("direct_battery_contact")
        direct = direct is True and object_state == "confirmed" and item.get("contact_mode") == "active"
        observations.append(
            {
                "frame_id": record.frame_id,
                "timestamp_seconds": record.timestamp_seconds,
                "battery_object": object_state,
                "contact_mode": item.get("contact_mode") if item.get("contact_mode") in {"active", "static", "none", "uncertain"} else "uncertain",
                "direct_contact": direct,
                "direct_battery_contact": direct,
                "evidence": str(item.get("evidence") or ""),
            }
        )
    # A malformed response never contributes contact evidence.  This keeps a
    # partially parsed cache from creating a reducer pass.
    usable = not errors
    direct_ids = [item["frame_id"] for item in observations if usable and item["direct_contact"]]
    object_values = [item["battery_object"] for item in observations]
    top_object = response.get("battery_object") if isinstance(response, Mapping) else None
    if top_object not in {"confirmed", "rejected", "unknown"}:
        top_object = "confirmed" if any(value == "confirmed" for value in object_values) else "unknown"
    return (
        {
            "status": "parsed" if isinstance(response, Mapping) else "invalid",
            "battery_object": top_object,
            "direct_contact_frame_ids": direct_ids,
            "observations": observations,
            "validation_errors": errors,
            "frame_ids": [record.frame_id for record in records],
            "usable": usable,
        },
        errors,
    )


def run_direct_contact_verifier(
    client: Any,
    model: str,
    records: list[FrameRecord],
    episode: Mapping[str, Any],
    output_dir: Path,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Run/cache the narrow battery contact verifier using ROI images only."""

    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_direct_contact_records(records, episode)
    frame_ids = [record.frame_id for record in selected]
    response_path = output_dir / "direct_contact_verifier.json"
    window: list[float] | None = None
    try:
        core_start, core_end = (float(value) for value in episode["core_interval_seconds"])
        expanded_start, expanded_end = (float(value) for value in episode["expanded_interval_seconds"])
        window = [max(expanded_start, core_start - DIRECT_CONTACT_MARGIN_SECONDS), min(expanded_end, core_end + DIRECT_CONTACT_MARGIN_SECONDS)]
    except (KeyError, TypeError, ValueError):
        pass

    def unavailable(status: str, error: str) -> dict[str, Any]:
        return {
            "status": status,
            "battery_object": "unknown",
            "direct_contact_frame_ids": [],
            "observations": [],
            "validation_errors": [error],
            "error": error,
            "raw": "",
            "frame_ids": frame_ids,
            "window_seconds": window,
            "cache_hit": False,
            "usable": False,
        }

    if not selected:
        result = unavailable("not_available", "no_dense_battery_roi_frames")
        write_json(response_path, {**result, "verifier_version": DIRECT_CONTACT_VERIFIER_VERSION})
        return result

    if use_cache and response_path.is_file():
        try:
            cached = read_json(response_path)
            cached_ids = cached.get("frame_ids") if isinstance(cached, Mapping) else None
            parsed = cached.get("parsed") if isinstance(cached, Mapping) else None
            if parsed is None and isinstance(cached, Mapping) and isinstance(cached.get("observations"), list):
                parsed = cached
            version_matches = cached.get("verifier_version") == DIRECT_CONTACT_VERIFIER_VERSION if isinstance(cached, Mapping) else False
            if version_matches and cached_ids == frame_ids and isinstance(parsed, Mapping):
                normalized, errors = normalize_direct_contact_verifier_response(parsed, selected)
                if not errors:
                    return {
                        **normalized,
                        "error": cached.get("error"),
                        "raw": str(cached.get("raw") or ""),
                        "window_seconds": window,
                        "cache_hit": True,
                    }
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    if client is None:
        return unavailable("not_run", "direct_contact_verifier_not_run")

    # A compact three-panel image preserves panorama context plus both ROI
    # hypotheses.  Twelve packets keep the VLM request below its GPU memory
    # limit while retaining the high-motion contact candidates.
    try:
        images = [
            paired_image(record, output_dir / "direct_contact_paired_frames" / f"{record.frame_id}.jpg")
            for record in selected
        ]
    except RuntimeError:
        result = unavailable("not_available", "battery_roi_image_missing")
        write_json(response_path, {**result, "verifier_version": DIRECT_CONTACT_VERIFIER_VERSION})
        return result

    parsed, error, raw = qwen_call(client, model, direct_contact_verifier_prompt(selected, episode), images)
    validation_errors = validate_direct_contact_verifier_response(parsed, selected) if isinstance(parsed, Mapping) else [error or "response_not_object"]
    payload = {
        "verifier_version": DIRECT_CONTACT_VERIFIER_VERSION,
        "parsed": parsed,
        "error": error,
        "validation_errors": validation_errors,
        "raw": raw,
        "frame_ids": frame_ids,
        "window_seconds": window,
        "image_paths": [str(path.resolve()) for path in images],
    }
    write_json(response_path, payload)

    # One targeted schema retry mirrors the existing fact-observation path.
    if parsed is None or validation_errors:
        retry_reason = error or ",".join(validation_errors)
        retry_prompt = direct_contact_verifier_prompt(selected, episode) + f"\n上一版未通过本地契约：{retry_reason}。逐帧补齐所有 FRAME ID，只输出指定 JSON。"
        retry_parsed, retry_error, retry_raw = qwen_call(client, model, retry_prompt, images)
        retry_errors = validate_direct_contact_verifier_response(retry_parsed, selected) if isinstance(retry_parsed, Mapping) else [retry_error or "response_not_object"]
        write_json(
            response_path.with_name("direct_contact_verifier_retry.json"),
            {
                **payload,
                "parsed": retry_parsed,
                "error": retry_error,
                "validation_errors": retry_errors,
                "raw": retry_raw,
            },
        )
        parsed, error, raw, validation_errors = retry_parsed, retry_error, retry_raw, retry_errors
        # Keep the canonical cache at the final attempted response.  A valid
        # retry must be reusable on deterministic replay instead of forcing a
        # fresh model call because the first response was malformed.
        write_json(
            response_path,
            {
                **payload,
                "parsed": parsed,
                "error": error,
                "validation_errors": validation_errors,
                "raw": raw,
            },
        )

    normalized, _ = normalize_direct_contact_verifier_response(parsed, selected)
    return {
        **normalized,
        "error": error,
        "raw": raw,
        "window_seconds": window,
        "cache_hit": False,
    }


def merge_frame_records(*groups: Iterable[FrameRecord]) -> list[FrameRecord]:
    """Merge records by frame ID while preserving chronological order."""

    by_id: dict[str, FrameRecord] = {}
    for group in groups:
        for record in group:
            by_id.setdefault(record.frame_id, record)
    return sorted(by_id.values(), key=lambda record: (record.timestamp_seconds, record.frame_id))


def usable_direct_contact_ids(
    verifier_result: Mapping[str, Any] | None,
    records: list[FrameRecord],
    interval: tuple[float, float] | list[float] | None = None,
) -> list[str]:
    """Return only validated direct-contact IDs, optionally time-bounded."""

    if not isinstance(verifier_result, Mapping) or verifier_result.get("usable") is not True:
        return []
    raw_ids = verifier_result.get("direct_contact_frame_ids")
    known = {record.frame_id: record for record in records}
    if not isinstance(raw_ids, list):
        return []
    bounds: tuple[float, float] | None = None
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        try:
            bounds = (float(interval[0]), float(interval[1]))
        except (TypeError, ValueError):
            bounds = None
    result: list[str] = []
    for frame_id in raw_ids:
        if not isinstance(frame_id, str) or frame_id not in known:
            continue
        timestamp = known[frame_id].timestamp_seconds
        if bounds is not None and not (bounds[0] <= timestamp <= bounds[1]):
            continue
        if frame_id not in result:
            result.append(frame_id)
    return result


def merge_direct_contact_summary(
    summary: Mapping[str, Any],
    verifier_result: Mapping[str, Any] | None,
    records: list[FrameRecord],
    episode: Mapping[str, Any],
) -> dict[str, Any]:
    """Add independently verified contact IDs to a topology summary."""

    merged = dict(summary)
    raw_existing = summary.get("direct_contact_frame_ids")
    existing_candidates = [item for item in raw_existing if isinstance(item, str)] if isinstance(raw_existing, list) else []
    rewire = summary.get("terminal_rewire")
    by_id = {record.frame_id: record.timestamp_seconds for record in records}
    if isinstance(rewire, Mapping):
        before = summary.get("battery_before")
        after = summary.get("battery_after")
        transition = classify_relocation(
            before.get("terminals") if isinstance(before, Mapping) else None,
            after.get("terminals") if isinstance(after, Mapping) else None,
        )
        start_id = rewire.get("start_frame_id")
        after_id = after.get("frame_id") if isinstance(after, Mapping) else None
        core = episode.get("core_interval_seconds") if isinstance(episode, Mapping) else None
        if (
            rewire.get("completed") is True
            and transition.get("completed") is True
            and isinstance(start_id, str)
            and isinstance(after_id, str)
            and start_id in by_id
            and after_id in by_id
            and isinstance(core, (list, tuple))
            and len(core) == 2
        ):
            contact_window = [float(core[0]) - DIRECT_CONTACT_MARGIN_SECONDS, float(core[1]) + 2.5]
            episode_contact_ids = usable_direct_contact_ids(verifier_result, records, contact_window)
            aligned_ids = [
                frame_id
                for frame_id in episode_contact_ids
                if by_id[start_id] <= by_id[frame_id] < by_id[after_id]
            ]
            if aligned_ids:
                latest_contact_id = max(aligned_ids, key=lambda frame_id: by_id[frame_id])
                current_end_id = rewire.get("end_frame_id")
                if not isinstance(current_end_id, str) or current_end_id not in by_id or by_id[current_end_id] < by_id[latest_contact_id]:
                    rewire = dict(rewire)
                    rewire["end_frame_id"] = latest_contact_id
                    merged["terminal_rewire"] = rewire
                    merged["direct_contact_temporal_alignment"] = {
                        "applied": True,
                        "source": "independent_live_contact_verifier",
                        "previous_end_frame_id": current_end_id,
                        "aligned_end_frame_id": latest_contact_id,
                    }
    interval: tuple[float, float] | list[float] | None = None
    if isinstance(rewire, Mapping):
        start_id, end_id = rewire.get("start_frame_id"), rewire.get("end_frame_id")
        if isinstance(start_id, str) and isinstance(end_id, str) and start_id in by_id and end_id in by_id:
            interval = (by_id[start_id], by_id[end_id])
    if interval is None:
        interval = episode.get("core_interval_seconds") if isinstance(episode, Mapping) else None
    known = {record.frame_id: record for record in records}
    existing: list[str] = []
    for frame_id in existing_candidates:
        if frame_id not in known:
            continue
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            try:
                if not (float(interval[0]) <= known[frame_id].timestamp_seconds <= float(interval[1])):
                    continue
            except (TypeError, ValueError):
                pass
        if frame_id not in existing:
            existing.append(frame_id)
    verified_ids = usable_direct_contact_ids(verifier_result, records, interval)
    merged["direct_contact_frame_ids"] = list(dict.fromkeys(existing + verified_ids))
    if verified_ids:
        merged["battery_object"] = "confirmed"
    return merged


def merge_direct_contact_observations(
    observations: list[dict[str, Any]],
    verifier_result: Mapping[str, Any] | None,
    records: list[FrameRecord],
    interval: tuple[float, float] | list[float] | None = None,
) -> list[dict[str, Any]]:
    """Overlay validated contact facts on a fallback frame-observation list."""

    by_id = {item.get("frame_id"): dict(item) for item in observations if isinstance(item, Mapping)}
    known = {record.frame_id: record for record in records}
    for frame_id in usable_direct_contact_ids(verifier_result, records, interval):
        record = known[frame_id]
        item = by_id.setdefault(
            frame_id,
            {
                "frame_id": frame_id,
                "timestamp_seconds": record.timestamp_seconds,
                "battery_terminals": None,
                "terminal_state_stable": False,
                "switch_state": "unknown",
                "switch_action": "none",
                "terminal_action": "none",
                "confidence": 0.65,
                "evidence": "Direct-contact verifier observed plug/hand contact with a fixed battery-holder terminal.",
            },
        )
        item["battery_object"] = "confirmed"
        item["direct_battery_contact"] = True
        item["terminal_action"] = item.get("terminal_action") if item.get("terminal_action") not in {None, "none"} else "relocate"
        item["confidence"] = max(float(item.get("confidence") or 0.0), 0.65)
    return sorted(by_id.values(), key=lambda item: (float(item.get("timestamp_seconds") or 0.0), str(item.get("frame_id") or "")))


def run_structured_summary(
    client: Any,
    model: str,
    records: list[FrameRecord],
    episode: Mapping[str, Any],
    output_dir: Path,
    use_cache: bool = True,
    direct_contact_repair_available: bool = False,
) -> tuple[dict[str, Any] | None, str | None, str]:
    selected = select_uniform(records, min(14, len(records)))
    images = [paired_image(record, output_dir / "paired_frames" / f"{record.frame_id}.jpg") for record in selected]
    response_path = output_dir / "structured_summary.json"
    if use_cache and response_path.is_file():
        try:
            cached = read_json(response_path)
            cached_parsed = cached.get("parsed")
            if isinstance(cached_parsed, Mapping):
                cached_errors = validate_structured_summary(cached_parsed, selected, episode)
                repairable = {"direct_battery_contact_missing", "no_direct_contact_during_rewire"}
                if not cached_errors or (direct_contact_repair_available and set(cached_errors) <= repairable):
                    return dict(cached_parsed), cached.get("error"), str(cached.get("raw") or "")
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if client is None:
        return None, "structured_summary_not_run", ""
    parsed, error, raw = qwen_call(client, model, structured_summary_prompt(selected, episode), images)
    validation_errors = validate_structured_summary(parsed, selected, episode) if isinstance(parsed, Mapping) else []
    write_json(response_path, {"parsed": parsed, "error": error, "validation_errors": validation_errors, "raw": raw, "frame_ids": [record.frame_id for record in selected]})
    repairable = {"direct_battery_contact_missing", "no_direct_contact_during_rewire"}
    requires_retry = parsed is None or (
        bool(validation_errors)
        and not (direct_contact_repair_available and set(validation_errors) <= repairable)
    )
    if requires_retry:
        retry_reason = error or ",".join(validation_errors)
        retry_prompt = structured_summary_prompt(selected, episode) + f"\n上一版未通过本地契约：{retry_reason}。重点核对换接后的稳定端子对是否与 from_terminal/to_terminal 一致；不能同时声称 T2->T1 已完成又把换接后写回 T0-T2。重新核对帧号和时间顺序，只输出指定 JSON 对象，不要 Markdown。"
        parsed, retry_error, retry_raw = qwen_call(client, model, retry_prompt, images)
        retry_validation = validate_structured_summary(parsed, selected, episode) if isinstance(parsed, Mapping) else []
        write_json(response_path.with_name("structured_summary_retry.json"), {"parsed": parsed, "error": retry_error, "validation_errors": retry_validation, "raw": retry_raw, "frame_ids": [record.frame_id for record in selected]})
        combined_error = retry_error or (",".join(retry_validation) if retry_validation else None)
        return parsed, combined_error, retry_raw
    return parsed, error, raw


def select_terminal_pair_records(records: list[FrameRecord], episode: Mapping[str, Any]) -> list[FrameRecord]:
    """Select a compact stable-state packet on both sides of the core gap."""
    try:
        core_start, core_end = (float(value) for value in episode["core_interval_seconds"])
    except (KeyError, TypeError, ValueError):
        return []
    ordered = sorted(records, key=lambda item: (item.timestamp_seconds, item.frame_id))
    before = [item for item in ordered if item.timestamp_seconds < core_start]
    after = [item for item in ordered if item.timestamp_seconds > core_end]
    return before[-2:] + after[:3]


def terminal_pair_verifier_prompt(records: list[FrameRecord], episode: Mapping[str, Any]) -> str:
    core_start, core_end = episode["core_interval_seconds"]
    before_ids = [item.frame_id for item in records if item.timestamp_seconds < core_start]
    after_ids = [item.frame_id for item in records if item.timestamp_seconds > core_end]
    return f"""只比较伏安法实验电池盒换线前后的稳定接线，不判断开关，不输出分数。每张图片依次含全景、跟踪 ROI、参考 ROI；ROI 失效时以全景为准。

目标对象是装有两节圆柱电池的长条电池盒，不是刀闸、变阻器或电表。沿长条电池盒串联方向定义 T0(一侧外端)-电池1-T1(两节之间的中间连接端)-电池2-T2(另一侧外端)。只看插入固定端子的两根外接导线插头位置，不要按实体电池数量猜测。

换线前候选 FRAME ID：{', '.join(before_ids)}。
换线后候选 FRAME ID：{', '.join(after_ids)}。
分别选择最清楚且手已离开的稳定帧。两节串联使用外端对 [T0,T2]；一节使用 [T0,T1] 或 [T1,T2]。若黑色或红色插头从外端移动到两节之间的金属连接端，换线后必须写对应的一节端子对。看不清时 stable=false、terminals=null，不能用动作顺序补造。

只输出一个 JSON：
{{
  "battery_object": "confirmed|rejected|unknown",
  "before": {{"frame_id": "合法ID或null", "terminals": ["T0","T2"] 或 ["T0","T1"] 或 ["T1","T2"] 或 null, "stable": true 或 false, "evidence": "直接可见事实"}},
  "after": {{"frame_id": "合法ID或null", "terminals": ["T0","T2"] 或 ["T0","T1"] 或 ["T1","T2"] 或 null, "stable": true 或 false, "evidence": "直接可见事实"}}
}}
不能输出 decision、pass、fail。episode={episode.get('episode_id')}，核心窗={episode.get('core_interval_seconds')}。"""


def normalize_terminal_pair_verifier_response(
    response: Mapping[str, Any] | None,
    records: list[FrameRecord],
    episode: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    known = {item.frame_id: item for item in records}
    try:
        core_start, core_end = (float(value) for value in episode["core_interval_seconds"])
    except (KeyError, TypeError, ValueError):
        core_start = core_end = 0.0
        errors.append("core_interval_invalid")
    if not isinstance(response, Mapping):
        response = {}
        errors.append("response_not_object")
    if response.get("battery_object") != "confirmed":
        errors.append("battery_object_not_confirmed")

    normalized: dict[str, Any] = {"battery_object": response.get("battery_object", "unknown")}
    for name in ("before", "after"):
        value = response.get(name)
        if not isinstance(value, Mapping):
            value = {}
            errors.append(f"{name}_not_object")
        frame_id = value.get("frame_id")
        record = known.get(frame_id) if isinstance(frame_id, str) else None
        if record is None:
            errors.append(f"{name}_frame_id_invalid")
        elif name == "before" and record.timestamp_seconds >= core_start:
            errors.append("before_not_before_core")
        elif name == "after" and record.timestamp_seconds <= core_end:
            errors.append("after_not_after_core")
        pair = normalize_terminals(value.get("terminals"))
        if pair not in {("T0", "T1"), ("T0", "T2"), ("T1", "T2")}:
            pair = None
            errors.append(f"{name}_terminal_pair_invalid")
        if value.get("stable") is not True:
            errors.append(f"{name}_not_stable")
        normalized[name] = {
            "frame_id": frame_id if record is not None else None,
            "terminals": list(pair) if pair is not None else None,
            "stable": value.get("stable") is True,
            "evidence": str(value.get("evidence") or ""),
        }
    normalized["transition"] = classify_relocation(
        normalized["before"]["terminals"], normalized["after"]["terminals"]
    )
    normalized["usable"] = not errors
    return normalized, sorted(set(errors))


def run_terminal_pair_verifier(
    client: Any,
    model: str,
    records: list[FrameRecord],
    episode: Mapping[str, Any],
    output_dir: Path,
    use_cache: bool = True,
) -> dict[str, Any]:
    selected = select_terminal_pair_records(records, episode)
    frame_ids = [item.frame_id for item in selected]
    response_path = output_dir / "terminal_pair_verifier.json"
    if len(selected) < 2:
        return {"status": "not_available", "usable": False, "validation_errors": ["stable_side_frames_missing"], "frame_ids": frame_ids}
    if use_cache and response_path.is_file():
        try:
            cached = read_json(response_path)
            if cached.get("verifier_version") == TERMINAL_PAIR_VERIFIER_VERSION and cached.get("frame_ids") == frame_ids:
                normalized, errors = normalize_terminal_pair_verifier_response(cached.get("parsed"), selected, episode)
                if not errors:
                    return {**normalized, "status": "parsed", "validation_errors": [], "cache_hit": True, "frame_ids": frame_ids}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if client is None:
        return {"status": "not_run", "usable": False, "validation_errors": ["terminal_pair_verifier_not_run"], "frame_ids": frame_ids}

    images = [paired_image(item, output_dir / "terminal_pair_frames" / f"{item.frame_id}.jpg") for item in selected]
    parsed, error, raw = qwen_call(client, model, terminal_pair_verifier_prompt(selected, episode), images)
    normalized, errors = normalize_terminal_pair_verifier_response(parsed, selected, episode)
    payload = {
        "verifier_version": TERMINAL_PAIR_VERIFIER_VERSION,
        "parsed": parsed,
        "error": error,
        "validation_errors": errors,
        "raw": raw,
        "frame_ids": frame_ids,
        "image_paths": [str(path.resolve()) for path in images],
    }
    write_json(response_path, payload)
    if errors:
        retry_prompt = terminal_pair_verifier_prompt(selected, episode) + f"\n上一版未通过字段契约：{','.join(errors)}。只按前后稳定帧中插头实际所在端子修正 JSON。"
        retry_parsed, retry_error, retry_raw = qwen_call(client, model, retry_prompt, images)
        retry_normalized, retry_errors = normalize_terminal_pair_verifier_response(retry_parsed, selected, episode)
        write_json(
            response_path.with_name("terminal_pair_verifier_retry.json"),
            {**payload, "parsed": retry_parsed, "error": retry_error, "validation_errors": retry_errors, "raw": retry_raw},
        )
        normalized, errors, error, raw = retry_normalized, retry_errors, retry_error, retry_raw
    return {
        **normalized,
        "status": "parsed" if not errors else "invalid",
        "validation_errors": errors,
        "error": error,
        "raw": raw,
        "cache_hit": False,
        "frame_ids": frame_ids,
    }


def merge_terminal_pair_summary(summary: Mapping[str, Any], verifier: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Replace contradictory topology fields with a focused direct comparison."""
    transition = verifier.get("transition")
    if verifier.get("usable") is not True or not isinstance(transition, Mapping) or transition.get("completed") is not True:
        return dict(summary), False
    before = verifier.get("before")
    after = verifier.get("after")
    moved = transition.get("moved_lead")
    rewire = summary.get("terminal_rewire")
    if not all(isinstance(value, Mapping) for value in (before, after, moved, rewire)):
        return dict(summary), False
    merged = dict(summary)
    merged["battery_before"] = {
        "frame_id": before.get("frame_id"),
        "terminals": before.get("terminals"),
        "stable": before.get("stable") is True,
    }
    merged["battery_after"] = {
        "frame_id": after.get("frame_id"),
        "terminals": after.get("terminals"),
        "stable": after.get("stable") is True,
    }
    merged["terminal_rewire"] = {
        **dict(rewire),
        "from_terminal": moved.get("from"),
        "to_terminal": moved.get("to"),
        "completed": True,
    }
    return merged, True


def valid_ids(value: Any, known: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item in known]


def normalize_fact_observations(raw: Mapping[str, Any], records: list[FrameRecord]) -> tuple[list[dict[str, Any]], list[str]]:
    known = {record.frame_id: record for record in records}
    errors: list[str] = []
    values = raw.get("observations")
    if not isinstance(values, list):
        return [], ["observations_not_list"]
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in values:
        if not isinstance(item, Mapping):
            errors.append("observation_not_object")
            continue
        frame_id = item.get("frame_id")
        if not isinstance(frame_id, str) or frame_id not in known or frame_id in by_id:
            errors.append("observation_frame_id_invalid_or_duplicate")
            continue
        by_id[frame_id] = item
    if len(by_id) != len(records):
        errors.append("observation_frame_coverage_incomplete")
    normalized: list[dict[str, Any]] = []
    for record in records:
        item = by_id.get(record.frame_id, {})
        terminals = item.get("battery_terminals")
        if terminals is not None and (not isinstance(terminals, list) or len(terminals) != 2):
            terminals = None
            errors.append(f"invalid_terminal_pair:{record.frame_id}")
        state = item.get("switch_state")
        if state not in {"open", "closed", "unknown"}:
            state = "unknown"
            if item:
                errors.append(f"invalid_switch_state:{record.frame_id}")
        normalized.append(
            {
                "frame_id": record.frame_id,
                "timestamp_seconds": record.timestamp_seconds,
                "battery_object": item.get("battery_object", "unknown"),
                "battery_terminals": terminals,
                "terminal_state_stable": item.get("terminal_state_stable") is True,
                "direct_battery_contact": item.get("direct_battery_contact") is True,
                "terminal_action": item.get("terminal_action", "none"),
                "terminal_rewire_completed": item.get("terminal_rewire_completed") is True,
                "switch_state": state,
                "switch_action": item.get("switch_action", "none"),
                "confidence": max(0.0, min(1.0, finite(item.get("confidence")) or 0.35)),
                "evidence": str(item.get("evidence") or ""),
            }
        )
    return normalized, sorted(set(errors))


def load_observation_file(root: Path | None, video_id: str, episode_id: str) -> dict[str, Any] | None:
    if root is None:
        return None
    candidates = []
    if root.is_file():
        candidates.append(root)
    else:
        candidates.extend(
            [
                root / f"video_{video_id}" / episode_id / "facts.json",
                root / f"video_{video_id}" / "facts.json",
                root / video_id / episode_id / "facts.json",
                root / video_id / "facts.json",
            ]
        )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping) and isinstance(value.get("episodes"), list):
            for episode in value["episodes"]:
                if isinstance(episode, Mapping) and str(episode.get("episode_id")) == episode_id:
                    return dict(episode)
        if isinstance(value, Mapping):
            return dict(value)
    return None


def create_client(base_url: str, token: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package is unavailable; use --prepare-only or --observations-root")
    if not base_url.strip() or not token.strip():
        raise RuntimeError("QWEN_API_BASE_URL and QWEN_API_TOKEN are required")
    return OpenAI(base_url=base_url, api_key=token, timeout=120, max_retries=0)


def candidate_times(
    records: list[FrameRecord],
    screen: Mapping[str, Any] | None,
    episode: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    by_id = {record.frame_id: record for record in records}
    known = set(by_id)
    battery_ids = valid_ids(screen.get("battery_candidate_frame_ids"), known) if screen else []
    switch_ids = valid_ids(screen.get("switch_candidate_frame_ids"), known) if screen else []
    battery_times = [by_id[item].timestamp_seconds for item in battery_ids]
    switch_times = [by_id[item].timestamp_seconds for item in switch_ids]
    core_start, core_end = episode["core_interval_seconds"]
    core_records = [record for record in records if core_start <= record.timestamp_seconds <= core_end]
    battery_times += [record.timestamp_seconds for record in sorted(core_records, key=lambda x: x.battery_motion, reverse=True)[:4]]
    switch_times += [record.timestamp_seconds for record in sorted(core_records, key=lambda x: x.motion, reverse=True)[:4]]
    return sorted(set(round(value, 3) for value in battery_times)), sorted(set(round(value, 3) for value in switch_times))


def select_dense_records(
    records: list[FrameRecord],
    battery_times: list[float],
    switch_times: list[float],
    episode: Mapping[str, Any],
    maximum: int = 18,
) -> list[FrameRecord]:
    by_time = {record.timestamp_seconds: record for record in records}
    start, end = episode["expanded_interval_seconds"]
    core_start, core_end = episode["core_interval_seconds"]
    anchors = [start, min(end, start + 1.0), max(start, core_start - 1.0), core_start, core_end, min(end, core_end + 1.0), end]
    for value in battery_times + switch_times:
        anchors.extend([value - 1.5, value - 0.75, value, value + 0.75, value + 1.5])
    available = sorted(records, key=lambda record: record.timestamp_seconds)
    selected: list[FrameRecord] = []
    for anchor in anchors:
        nearest = min(available, key=lambda record: abs(record.timestamp_seconds - anchor))
        if nearest not in selected:
            selected.append(nearest)
    selected.sort(key=lambda record: record.timestamp_seconds)
    if len(selected) > maximum:
        selected = select_uniform(selected, maximum)
    return selected


def build_screen_sheets(records: list[FrameRecord], output_dir: Path) -> list[Path]:
    sheets: list[Path] = []
    for index, chunk in enumerate(split_chunks(records, 12), start=1):
        sheets.append(make_contact_sheet(chunk, output_dir / f"screen_{index:02d}.jpg", "panorama"))
    return sheets


def select_screening_records(
    records: list[FrameRecord],
    episode: Mapping[str, Any],
    maximum: int = 24,
) -> list[FrameRecord]:
    """Bound coarse VLM requests while retaining temporal and motion coverage."""
    if len(records) <= maximum:
        return sorted(records, key=lambda record: record.timestamp_seconds)
    uniform_limit = maximum // 2
    selected = select_uniform(records, uniform_limit)
    core_start, core_end = episode["core_interval_seconds"]
    core = [record for record in records if core_start <= record.timestamp_seconds <= core_end]
    ranked = sorted(
        core or records,
        key=lambda record: (record.motion + record.battery_motion, record.sharpness),
        reverse=True,
    )
    for record in ranked:
        if record not in selected:
            selected.append(record)
        if len(selected) >= maximum:
            break
    return sorted(selected, key=lambda record: record.timestamp_seconds)


def run_screening(
    client: Any,
    model: str,
    records: list[FrameRecord],
    episode: Mapping[str, Any],
    output_dir: Path,
    use_cache: bool = True,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    battery_ids: list[str] = []
    switch_ids: list[str] = []
    cleanup_ids: list[str] = []
    visible_ids: list[str] = []
    for index, chunk in enumerate(split_chunks(records, 12), start=1):
        sheet = output_dir / f"screen_{index:02d}.jpg"
        make_contact_sheet(chunk, sheet, "panorama")
        response_path = output_dir / f"screen_{index:02d}.json"
        parsed: dict[str, Any] | None = None
        error: str | None = None
        raw = ""
        cached_failure = False
        if use_cache and response_path.is_file():
            try:
                cached = read_json(response_path)
                expected_ids = [record.frame_id for record in chunk]
                if isinstance(cached, Mapping) and cached.get("frame_ids") == expected_ids:
                    parsed = cached.get("parsed")
                    error = cached.get("error")
                    raw = str(cached.get("raw") or "")
                    cached_failure = parsed is None and isinstance(error, str) and bool(error)
            except (OSError, ValueError, json.JSONDecodeError):
                parsed = None
        if parsed is None and client is not None and not cached_failure:
            parsed, error, raw = qwen_call(client, model, screen_prompt(chunk, episode), [sheet])
            write_json(response_path, {"parsed": parsed, "error": error, "raw": raw, "frame_ids": [r.frame_id for r in chunk]})
        if parsed is None:
            reports.append({"chunk_index": index, "frame_ids": [r.frame_id for r in chunk], "status": "unavailable", "error": error or "screening_not_run"})
            continue
        known = {r.frame_id for r in chunk}
        b = valid_ids(parsed.get("battery_candidate_frame_ids"), known)
        s = valid_ids(parsed.get("switch_candidate_frame_ids"), known)
        c = valid_ids(parsed.get("cleanup_like_frame_ids"), known)
        visible = valid_ids(parsed.get("battery_object_visible_frame_ids"), known)
        battery_ids.extend(b); switch_ids.extend(s); cleanup_ids.extend(c)
        visible_ids.extend(visible)
        reports.append({"chunk_index": index, "frame_ids": [r.frame_id for r in chunk], "status": "parsed", "battery_candidate_frame_ids": b, "switch_candidate_frame_ids": s, "cleanup_like_frame_ids": c, "battery_object_visible_frame_ids": visible, "notes": parsed.get("notes", [])})
    result = {
        "battery_candidate_frame_ids": sorted(set(battery_ids)),
        "switch_candidate_frame_ids": sorted(set(switch_ids)),
        "cleanup_like_frame_ids": sorted(set(cleanup_ids)),
        "battery_object_visible_frame_ids": sorted(set(visible_ids)),
    }
    return result, reports


def select_localization_records(
    records: list[FrameRecord],
    screening: Mapping[str, Any] | None,
    maximum: int = 24,
) -> list[FrameRecord]:
    by_id = {record.frame_id: record for record in records}
    ids: list[str] = []
    if screening:
        known = set(by_id)
        ids.extend(valid_ids(screening.get("battery_object_visible_frame_ids"), known))
        ids.extend(valid_ids(screening.get("battery_candidate_frame_ids"), known))
    selected = [by_id[frame_id] for frame_id in dict.fromkeys(ids)]
    for record in select_uniform(records, min(12, len(records))):
        if record not in selected:
            selected.append(record)
    selected.sort(key=lambda record: record.timestamp_seconds)
    return select_uniform(selected, maximum) if len(selected) > maximum else selected


def run_dynamic_localization(
    client: Any,
    model: str,
    records: list[FrameRecord],
    episode: Mapping[str, Any],
    output_dir: Path,
    minimum_confidence: float,
    use_cache: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detections: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for index, chunk in enumerate(split_chunks(records, 8), start=1):
        response_path = output_dir / f"localize_{index:02d}.json"
        parsed: dict[str, Any] | None = None
        error: str | None = None
        raw = ""
        cached_failure = False
        if use_cache and response_path.is_file():
            cached = read_json(response_path)
            expected_ids = [record.frame_id for record in chunk]
            if isinstance(cached, Mapping) and cached.get("frame_ids") == expected_ids:
                parsed = cached.get("parsed")
                error = cached.get("error")
                raw = str(cached.get("raw") or "")
                cached_failure = parsed is None and isinstance(error, str) and bool(error)
        images = [Path(record.localization_path or record.panorama_path) for record in chunk]
        if parsed is None and client is not None and not cached_failure:
            parsed, error, raw = qwen_call(client, model, localization_prompt(chunk, episode), images)
            write_json(
                response_path,
                {
                    "parsed": parsed,
                    "error": error,
                    "raw": raw,
                    "frame_ids": [record.frame_id for record in chunk],
                    "image_paths": [str(path.resolve()) for path in images],
                },
            )
        known = {record.frame_id for record in chunk}
        normalized = normalize_dynamic_battery_detections(
            parsed.get("battery_detections") if parsed else None,
            known,
            minimum_confidence,
        )
        detections.extend(normalized)
        reports.append(
            {
                "chunk_index": index,
                "frame_ids": [record.frame_id for record in chunk],
                "status": "parsed" if parsed is not None else "unavailable",
                "detections": normalized,
                "error": error,
            }
        )
    return detections, reports


def run_fact_observation(client: Any, model: str, records: list[FrameRecord], episode: Mapping[str, Any], output_dir: Path, use_cache: bool = True) -> tuple[dict[str, Any] | None, str | None, str]:
    pair_dir = output_dir / "paired_frames"
    images = [paired_image(record, pair_dir / f"{record.frame_id}.jpg") for record in records]
    response_path = output_dir / "facts.json"
    if use_cache and response_path.is_file():
        try:
            cached = read_json(response_path)
            return cached.get("parsed"), cached.get("error"), str(cached.get("raw") or "")
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if client is None:
        return None, "facts_not_run", ""
    parsed, error, raw = qwen_call(client, model, facts_prompt(records, episode), images)
    write_json(response_path, {"parsed": parsed, "error": error, "raw": raw, "frame_ids": [r.frame_id for r in records]})
    if parsed is None and error:
        # One targeted format retry; no blind retry loop.
        retry_prompt = facts_prompt(records, episode) + "\n上一版不是合法 JSON。只输出一个完整 JSON 对象，不要 Markdown，不要解释。"
        parsed, retry_error, retry_raw = qwen_call(client, model, retry_prompt, images)
        write_json(response_path.with_name("facts_retry.json"), {"parsed": parsed, "error": retry_error, "raw": retry_raw, "frame_ids": [r.frame_id for r in records]})
        return parsed, retry_error, retry_raw
    return parsed, error, raw


def make_episode(
    video_id: str,
    episode_index: int,
    source_video: str,
    result_path: str,
    interval: list[float],
    intervals: Mapping[str, list[list[float]]],
    duration: float,
    *,
    episode_kind: str = "rewire",
    candidate_source: str = "stage_circuit_rewiring",
    recovery_anchor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    start, end = interval
    cleanup_values = [item[0] for item in intervals.get("material_cleanup", []) if item and finite(item[0]) is not None]
    cleanup_cutoff = min(cleanup_values) if cleanup_values else duration
    expanded_start = max(0.0, start - 10.0)
    expanded_end = min(duration, end + 10.0, cleanup_cutoff)
    return {
        "episode_id": f"{video_id}_{episode_kind}_{episode_index:02d}",
        "video_id": video_id,
        "source_video": source_video,
        "action_result_path": result_path,
        "stage_schema_id": SCHEMA_ID,
        "core_interval_seconds": [round(start, 3), round(end, 3)],
        "expanded_interval_seconds": [round(expanded_start, 3), round(max(expanded_start, expanded_end), 3)],
        "cleanup_cutoff_seconds": round(cleanup_cutoff, 3),
        "candidate_source": candidate_source,
        "recovery_anchor": dict(recovery_anchor) if recovery_anchor is not None else None,
        "sampling_policy": {"coarse_fps": 2.0, "core_fps": 5.0, "transition_fps": 10.0, "margin_seconds": 10.0},
    }


def process_episode(
    episode: dict[str, Any],
    video_path: Path,
    output_dir: Path,
    client: Any,
    model: str,
    supplied_observations: Mapping[str, Any] | None,
    prepare_only: bool,
    *,
    coarse_fps: float = 2.0,
    core_fps: float = 5.0,
    transition_fps: float = 10.0,
    dynamic_roi_min_confidence: float = 0.45,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    start, end = episode["expanded_interval_seconds"]
    core_start, core_end = episode["core_interval_seconds"]
    episode["sampling_policy"] = {
        "coarse_fps": coarse_fps,
        "core_fps": core_fps,
        "transition_fps": transition_fps,
        "margin_seconds": 10.0,
    }
    coarse = extract_frames(
        video_path,
        episode["video_id"],
        episode["episode_id"],
        start,
        end,
        coarse_fps,
        output_dir / "coarse",
        save_localization_images=True,
    )
    write_json(output_dir / "coarse_manifest.json", {"frames": [item.as_dict() for item in coarse]})
    screening_records = select_screening_records(coarse, episode)
    screening, screen_reports = run_screening(
        None if prepare_only else client,
        model,
        screening_records,
        episode,
        output_dir / "screening",
    ) if coarse else (None, [])
    localization_records = select_localization_records(coarse, screening)
    dynamic_detections, localization_reports = run_dynamic_localization(
        None if prepare_only else client,
        model,
        localization_records,
        episode,
        output_dir / "localization",
        dynamic_roi_min_confidence,
    ) if localization_records else ([], [])
    write_json(
        output_dir / "dynamic_roi_manifest.json",
        {
            "schema_version": "resistance_dynamic_battery_roi.v1",
            "source": "current_video_panorama_only",
            "video_id_routing_used": False,
            "configured_roi_used": False,
            "reference_frame_used": False,
            "minimum_confidence": dynamic_roi_min_confidence,
            "candidate_frame_ids": [record.frame_id for record in localization_records],
            "detections": dynamic_detections,
            "chunks": localization_reports,
        },
    )
    battery_times, switch_times = candidate_times(coarse, screening, episode) if coarse else ([], [])
    # Dense records cover the core plus a narrow +/-5 second contact window.
    # Candidate activity ranges remain available for switch/topology evidence,
    # but the direct-contact verifier later filters strictly to this window.
    contact_left = max(start, core_start - DIRECT_CONTACT_MARGIN_SECONDS)
    contact_right = min(end, core_end + DIRECT_CONTACT_MARGIN_SECONDS)
    dense_ranges: list[tuple[float, float]] = [(contact_left, contact_right), (core_start, core_end)]
    for value in battery_times + switch_times:
        dense_ranges.append((max(start, value - 2.5), min(end, value + 2.5)))
    merged: list[list[float]] = []
    for left, right in sorted(dense_ranges):
        if merged and left <= merged[-1][1] + 0.5:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    dense: list[FrameRecord] = []
    dense_seed_records: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(merged, start=1):
        initial, source_frame_id = nearest_dynamic_roi(
            coarse,
            dynamic_detections,
            (left + right) / 2.0,
        )
        sample_fps = core_fps if (left, right) == (core_start, core_end) else transition_fps
        dense_seed_records.append(
            {
                "range_seconds": [round(left, 3), round(right, 3)],
                "source_frame_id": source_frame_id,
                "bbox_normalized": list(initial) if initial else None,
                "sample_fps": sample_fps,
            }
        )
        dense.extend(
            extract_frames(
                video_path,
                episode["video_id"],
                episode["episode_id"] + f"_d{index}",
                left,
                right,
                sample_fps,
                output_dir / "dense" / f"range_{index:02d}",
                initial,
            )
        )
    write_json(
        output_dir / "dense_roi_seeds.json",
        {
            "source": "nearest_live_panorama_detection",
            "configured_roi_used": False,
            "seeds": dense_seed_records,
        },
    )
    by_id = {item.frame_id: item for item in dense}
    # Keep a bounded, temporally ordered evidence packet while retaining all extracted files on disk.
    dense_sorted = sorted(by_id.values(), key=lambda item: item.timestamp_seconds)
    dense_selected = select_dense_records(dense_sorted, battery_times, switch_times, episode, maximum=18)
    direct_records = select_direct_contact_records(dense_sorted, episode)
    write_json(
        output_dir / "dense_manifest.json",
        {"frames": [item.as_dict() for item in dense_selected], "all_extracted_count": len(dense)},
    )
    write_json(
        output_dir / "direct_contact_manifest.json",
        {
            "verifier_version": DIRECT_CONTACT_VERIFIER_VERSION,
            "frames": [item.as_dict() for item in direct_records],
            "window_seconds": [
                max(start, core_start - DIRECT_CONTACT_MARGIN_SECONDS),
                min(end, core_end + DIRECT_CONTACT_MARGIN_SECONDS),
            ],
        },
    )
    # Run the contact check before the topology summary so its independent
    # direct-contact evidence can repair a summary that missed a brief plug
    # touch.  Supplied observations are deterministic replay input and should
    # not trigger a second model call.
    if supplied_observations is None:
        direct_result = run_direct_contact_verifier(
            None if prepare_only else client,
            model,
            direct_records,
            episode,
            output_dir / "facts",
        )
    else:
        direct_result = {
            "status": "not_run",
            "battery_object": "unknown",
            "direct_contact_frame_ids": [],
            "observations": [],
            "validation_errors": ["supplied_observations_replay"],
            "error": "supplied_observations_replay",
            "raw": "",
            "frame_ids": [record.frame_id for record in direct_records],
            "window_seconds": [
                max(start, core_start - DIRECT_CONTACT_MARGIN_SECONDS),
                min(end, core_end + DIRECT_CONTACT_MARGIN_SECONDS),
            ],
            "cache_hit": False,
            "usable": False,
        }
    # Topology summaries generally use a smaller packet.  Conversion receives
    # the union so a contact frame retained only by the narrow verifier still
    # contributes its direct-action evidence without fusing another episode.
    fact_records = merge_frame_records(dense_selected, direct_records)
    facts = None
    fact_error = None
    raw_facts = ""
    structured_summary = None
    terminal_pair_result: dict[str, Any] = {
        "status": "not_run",
        "usable": False,
        "validation_errors": [],
        "frame_ids": [],
        "applied": False,
    }
    structured_error = None
    structured_raw = ""
    validation_errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    if supplied_observations is not None:
        facts = dict(supplied_observations)
        fact_error = None
        raw_facts = "supplied_observations"
    elif not prepare_only and dense_selected:
        structured_summary, structured_error, structured_raw = run_structured_summary(
            client,
            model,
            dense_selected,
            episode,
            output_dir / "facts",
            direct_contact_repair_available=bool(direct_result.get("direct_contact_frame_ids")),
        )
        if structured_summary is not None:
            initial_summary_errors = validate_structured_summary(structured_summary, fact_records, episode)
            if any(
                error in initial_summary_errors
                for error in (
                    "terminal_pair_not_two_to_one",
                    "rewire_claim_conflicts_with_terminal_pairs",
                    "rewire_endpoints_conflict_with_terminal_pairs",
                )
            ):
                terminal_pair_result = run_terminal_pair_verifier(
                    client,
                    model,
                    dense_selected,
                    episode,
                    output_dir / "facts",
                )
                structured_summary, applied = merge_terminal_pair_summary(structured_summary, terminal_pair_result)
                terminal_pair_result["applied"] = applied
            structured_summary = merge_direct_contact_summary(structured_summary, direct_result, fact_records, episode)
            summary_observations, conversion_errors = structured_summary_to_observations(structured_summary, fact_records)
            facts = {"observations": summary_observations}
            normalized = summary_observations
            fact_error = structured_error
            raw_facts = structured_raw
            validation_errors = validate_structured_summary(structured_summary, fact_records, episode)
            validation_errors.extend(conversion_errors)
        else:
            facts, fact_error, raw_facts = run_fact_observation(client, model, dense_selected, episode, output_dir / "facts")
            normalized, validation_errors = normalize_fact_observations(facts or {}, dense_selected)
            normalized = merge_direct_contact_observations(
                normalized,
                direct_result,
                fact_records,
                episode.get("core_interval_seconds"),
            )
    if supplied_observations is not None or prepare_only or not dense_selected:
        normalized, validation_errors = normalize_fact_observations(facts or {}, dense_selected)
    elif structured_summary is None:
        normalized, validation_errors = normalize_fact_observations(facts or {}, dense_selected)
        normalized = merge_direct_contact_observations(
            normalized,
            direct_result,
            fact_records,
            episode.get("core_interval_seconds"),
        )
    validation_errors = sorted(set(validation_errors + list(direct_result.get("validation_errors") or [])))
    reducer_episode = {**episode, "observations": normalized}
    reducer = aggregate_episodes([reducer_episode], video_id=episode["video_id"])
    return {
        **episode,
        "roi": None,
        "roi_mode": "qwen_live_panorama_then_adjacent_frame_tracking",
        "configured_roi_used": False,
        "reference_frame_used": False,
        "dynamic_roi_detection_count": len(dynamic_detections),
        "screening": {"summary": screening, "chunks": screen_reports, "battery_candidate_times": battery_times, "switch_candidate_times": switch_times},
        "dynamic_localization": {"detections": dynamic_detections, "chunks": localization_reports},
        "facts": {"status": "parsed" if facts is not None else "not_available", "validation_errors": validation_errors, "raw": raw_facts, "structured_summary": structured_summary, "structured_error": structured_error, "observations": normalized, "direct_contact_verifier": direct_result, "terminal_pair_verifier": terminal_pair_result},
        "direct_contact_verifier": direct_result,
        "terminal_pair_verifier": terminal_pair_result,
        "reducer": reducer,
        "decision": reducer["decision"],
        "predicted_score": reducer["predicted_score"],
        "confidence": reducer["confidence"],
        "reason": reducer["episodes"][0].get("reason") if reducer.get("episodes") else "No episode evidence",
        "evidence_quality": "high" if not validation_errors and facts is not None else "low",
    }


def aggregate_processed_episodes(episodes: list[Mapping[str, Any]], video_id: str) -> dict[str, Any]:
    """Aggregate already-reduced episodes without re-reading nested evidence."""
    passing = [item for item in episodes if item.get("decision") == "pass"]
    confidence_values = [finite(item.get("confidence")) for item in (passing or episodes)]
    confidence = max((value for value in confidence_values if value is not None), default=0.25)
    return {
        "schema_version": "resistance_disconnect_battery_sequence_v1.aggregate.v1",
        "video_id": str(video_id),
        "decision": "pass" if passing else "fail",
        "predicted_score": 1 if passing else 0,
        "confidence": round(confidence, 3),
        "reason_code": "at_least_one_episode_passed" if passing else "no_episode_passed",
        "passing_episode_ids": [str(item.get("episode_id")) for item in passing],
        "episodes": [item.get("reducer") for item in episodes],
        "diagnostics": {"aggregation": "processed_episode_results_only", "cross_episode_evidence_fusion": False},
    }


def load_supplied_map(root: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if root is None:
        return {}
    paths = [root] if root.is_file() else list(root.rglob("*.json"))
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        values = value.get("episodes") if isinstance(value, Mapping) else None
        if not isinstance(values, list):
            values = [value] if isinstance(value, Mapping) else []
        for item in values:
            if not isinstance(item, Mapping):
                continue
            key = (str(item.get("video_id") or ""), str(item.get("episode_id") or ""))
            if key[0] and key[1] and isinstance(item.get("observations"), list):
                result[key] = dict(item)
    return result


def normalize_dynamic_battery_detections(
    value: Any,
    known: set[str],
    minimum_confidence: float = 0.45,
) -> list[dict[str, Any]]:
    """Validate frame-bound live detections without consulting video identity."""
    best: dict[str, dict[str, Any]] = {}
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, Mapping):
            continue
        frame_id = item.get("frame_id")
        bbox = item.get("bbox_normalized")
        confidence = finite(item.get("confidence"))
        if not isinstance(frame_id, str) or frame_id not in known:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4 or confidence is None:
            continue
        coordinates = [finite(coordinate) for coordinate in bbox]
        if any(coordinate is None for coordinate in coordinates):
            continue
        x1, y1, x2, y2 = (float(coordinate) for coordinate in coordinates)
        width, height = x2 - x1, y2 - y1
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            continue
        if width < 0.04 or height < 0.04 or width * height > 0.75:
            continue
        confidence = max(0.0, min(1.0, confidence))
        if confidence < minimum_confidence:
            continue
        normalized = {
            "frame_id": frame_id,
            "bbox_normalized": [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)],
            "confidence": round(confidence, 4),
            "source": "qwen_live_panorama",
        }
        previous = best.get(frame_id)
        if previous is None or normalized["confidence"] > previous["confidence"]:
            best[frame_id] = normalized
    return [best[frame_id] for frame_id in sorted(best)]


def nearest_dynamic_roi(
    records: list[FrameRecord],
    detections: list[Mapping[str, Any]],
    timestamp_seconds: float,
) -> tuple[tuple[float, float, float, float] | None, str | None]:
    by_id = {record.frame_id: record for record in records}
    usable = [item for item in detections if isinstance(item.get("frame_id"), str) and item["frame_id"] in by_id]
    if not usable:
        return None, None
    selected = min(
        usable,
        key=lambda item: abs(by_id[str(item["frame_id"])].timestamp_seconds - timestamp_seconds),
    )
    bbox = selected.get("bbox_normalized")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, None
    return tuple(float(value) for value in bbox), str(selected["frame_id"])  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video-root", type=Path, default=ROOT / "data" / "videos")
    parser.add_argument("--roi-config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--observations-root", type=Path, default=None, help="Offline facts JSON for deterministic replay")
    parser.add_argument("--video-ids", default="", help="Optional comma-separated action-record ids; empty means all records")
    parser.add_argument("--source-video", type=Path, default=None, help="Current run's isolated source video")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--api-token", default=DEFAULT_API_TOKEN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--coarse-fps", type=float, default=2.0)
    parser.add_argument("--core-fps", type=float, default=5.0)
    parser.add_argument("--transition-fps", type=float, default=10.0)
    parser.add_argument("--dynamic-roi-min-confidence", type=float, default=0.45)
    parser.add_argument(
        "--time-mode",
        choices=tuple(sorted(R8_TIME_MODES)),
        default="rewiring_recovery",
        help="Situation-selected interval source for the dynamic R8 executor",
    )
    args = parser.parse_args(argv)
    action_summary = args.action_summary if args.action_summary.is_absolute() else ROOT / args.action_summary
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    video_root = args.video_root if args.video_root.is_absolute() else ROOT / args.video_root
    if not action_summary.is_file():
        raise FileNotFoundError(action_summary)
    if any(value <= 0 for value in (args.coarse_fps, args.core_fps, args.transition_fps)):
        raise ValueError("sampling fps values must be positive")
    if not 0.0 <= args.dynamic_roi_min_confidence <= 1.0:
        raise ValueError("dynamic ROI minimum confidence must be between 0 and 1")
    if output_root.exists() and any(output_root.iterdir()) and not args.no_cache:
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output_root}; use --no-cache with a new output root")
    output_root.mkdir(parents=True, exist_ok=True)
    action_document = read_json(action_summary)
    supplied = load_supplied_map(args.observations_root)
    client = None if args.prepare_only or supplied else create_client(args.api_base_url, args.api_token)
    wanted = {item.strip() for item in str(args.video_ids).split(",") if item.strip()}
    source_override = args.source_video.resolve() if args.source_video else None
    if source_override is not None and not source_override.is_file():
        raise FileNotFoundError(source_override)
    records: list[dict[str, Any]] = []
    for source_record in action_document.get("records", []):
        if not isinstance(source_record, Mapping):
            continue
        source_name = str(source_record.get("source_video_id") or "")
        try:
            video_id = video_id_from_name(source_name)
        except ValueError:
            continue
        if wanted and video_id not in wanted:
            continue
        result_path = Path(str(source_record.get("result_path") or ""))
        if not result_path.is_absolute():
            result_path = action_summary.parent / result_path
        action_result = read_json(result_path)
        intervals = stage_intervals(action_result)
        source_video = (
            source_override
            if source_override is not None
            else discover_video(source_name, video_root)
        )
        _, duration, _ = video_duration(source_video)
        episodes: list[dict[str, Any]] = []
        episode_indexes: dict[str, int] = {}
        for candidate in episode_candidates(intervals, args.time_mode):
            episode_kind = str(candidate["episode_kind"])
            episode_indexes[episode_kind] = episode_indexes.get(episode_kind, 0) + 1
            episode = make_episode(
                video_id,
                episode_indexes[episode_kind],
                str(source_video),
                str(result_path.resolve()),
                candidate["interval"],
                intervals,
                duration,
                episode_kind=episode_kind,
                candidate_source=str(candidate["candidate_source"]),
                recovery_anchor=candidate.get("recovery_anchor"),
            )
            ep_dir = output_root / f"video_{video_id}" / episode["episode_id"]
            supplied_item = supplied.get((video_id, episode["episode_id"]))
            episodes.append(
                process_episode(
                    episode,
                    source_video,
                    ep_dir,
                    client,
                    args.model,
                    supplied_item,
                    args.prepare_only,
                    coarse_fps=args.coarse_fps,
                    core_fps=args.core_fps,
                    transition_fps=args.transition_fps,
                    dynamic_roi_min_confidence=args.dynamic_roi_min_confidence,
                )
            )
        aggregate = aggregate_processed_episodes(episodes, video_id=video_id)
        video_record = {
            "video_id": video_id,
            "source_video": str(source_video),
            "source_video_size_bytes": source_video.stat().st_size,
            "action_result_path": str(result_path.resolve()),
            "stage_schema_id": SCHEMA_ID,
            "time_mode": args.time_mode,
            "dynamic_r8_execution": True,
            "episodes": episodes,
            "decision": aggregate["decision"],
            "predicted_score": aggregate["predicted_score"],
            "confidence": aggregate["confidence"],
            "reason": "; ".join(str(item.get("reason")) for item in episodes if item.get("reason")) or f"No R8 episode candidate was supplied for time_mode={args.time_mode}.",
            "diagnostics": aggregate,
        }
        write_json(output_root / f"video_{video_id}" / "result.json", video_record)
        records.append(video_record)
    summary = {
        "schema_version": "resistance_disconnect_battery_sequence_v1.summary.v1",
        "algorithm_id": ALGORITHM_ID,
        "stage_schema_id": SCHEMA_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "action_summary": str(action_summary.resolve()),
        "roi_config": None,
        "reference_frames": {},
        "configured_roi_used": False,
        "reference_frame_used": False,
        "video_id_routing_used": False,
        "deprecated_roi_config_ignored": args.roi_config is not None,
        "dynamic_roi": {
            "source": "qwen_live_panorama",
            "minimum_confidence": args.dynamic_roi_min_confidence,
            "tracking": "adjacent_frame_orb_affine",
        },
        "time_mode": args.time_mode,
        "dynamic_r8_execution": True,
        "prepare_only": bool(args.prepare_only),
        "videos": records,
        "decision_counts": {"pass": sum(item["decision"] == "pass" for item in records), "fail": sum(item["decision"] == "fail" for item in records)},
        "excel_accessed": False,
        "source_videos_modified": False,
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps({"status": "completed", "videos": len(records), "decision_counts": summary["decision_counts"], "output": str(output_root.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
