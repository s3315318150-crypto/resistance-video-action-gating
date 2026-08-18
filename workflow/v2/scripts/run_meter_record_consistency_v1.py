#!/usr/bin/env python3
"""Verify first paper U/I values against analog meters from the same cycle.

The paper values are read from the completed adaptive record-sheet workflow.
They are never included in the Qwen request.  For each video, this script
extracts a short, dense window immediately before first writing, crops the two
analog meters, adds a relative-video-time information bar, and asks Qwen for
an independent meter reading.  The local reducer then compares U and I with
range-aware one-division tolerances and emits a binary pass/fail result.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC_CONFIG = ROOT / "configs" / "meter_record_consistency.example.json"
DEFAULT_PAPER_SUMMARY = ROOT / "examples" / "paper_records.example.json"
DEFAULT_VIDEO_ROOT = ROOT / "data" / "videos"
DEFAULT_OUTPUT = ROOT / "outputs" / "meter_record_consistency_v1"
API_BASE_URL = os.getenv("QWEN_API_BASE_URL", "").strip()
API_TOKEN = os.getenv("QWEN_API_TOKEN", "").strip()
MODEL = os.getenv("QWEN_MODEL", "qwen").strip() or "qwen"


@dataclass(frozen=True)
class VideoSpec:
    video_id: str
    source_video: str
    timestamps: tuple[float, ...]
    meter_roi_normalized_xyxy: tuple[float, float, float, float]
    precision_timestamp: float
    ammeter_roi_normalized_xyxy: tuple[float, float, float, float]
    voltmeter_roi_normalized_xyxy: tuple[float, float, float, float]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_expected:{path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def normalized_roi(value: Any, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field}_must_have_four_numbers")
    roi = tuple(float(item) for item in value)
    x1, y1, x2, y2 = roi
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{field}_outside_normalized_image")
    return roi


def load_video_specs(path: Path) -> tuple[VideoSpec, ...]:
    config = read_json(path)
    rows = config.get("videos")
    if not isinstance(rows, list) or not rows:
        raise ValueError("spec_config_videos_missing")
    specs: list[VideoSpec] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"spec_config_video_invalid:{index}")
        video_id = str(row.get("video_id") or "").strip()
        source_video = str(row.get("source_video") or "").strip()
        timestamps_raw = row.get("timestamps_seconds")
        precision_rois = row.get("precision_rois")
        if not video_id or video_id in seen:
            raise ValueError(f"spec_config_video_id_invalid:{index}")
        if not source_video or Path(source_video).is_absolute():
            raise ValueError(f"spec_config_source_video_invalid:{video_id}")
        if not isinstance(timestamps_raw, list) or not timestamps_raw:
            raise ValueError(f"spec_config_timestamps_missing:{video_id}")
        timestamps = tuple(float(item) for item in timestamps_raw)
        if any(item < 0.0 for item in timestamps):
            raise ValueError(f"spec_config_timestamp_negative:{video_id}")
        if not isinstance(precision_rois, dict):
            raise ValueError(f"spec_config_precision_rois_missing:{video_id}")
        specs.append(
            VideoSpec(
                video_id=video_id,
                source_video=source_video,
                timestamps=timestamps,
                meter_roi_normalized_xyxy=normalized_roi(
                    row.get("meter_roi_normalized_xyxy"),
                    f"meter_roi:{video_id}",
                ),
                precision_timestamp=float(row.get("precision_timestamp_seconds")),
                ammeter_roi_normalized_xyxy=normalized_roi(
                    precision_rois.get("ammeter"),
                    f"ammeter_roi:{video_id}",
                ),
                voltmeter_roi_normalized_xyxy=normalized_roi(
                    precision_rois.get("voltmeter"),
                    f"voltmeter_roi:{video_id}",
                ),
            )
        )
        seen.add(video_id)
    return tuple(specs)


def resolve_source_video(video_root: Path, source_video: str) -> Path:
    root = video_root.expanduser().resolve()
    source = (root / source_video).resolve()
    if not source.is_relative_to(root):
        raise ValueError(f"source_video_outside_video_root:{source_video}")
    return source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_decimal(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*(?:约|≈|~)?\s*(\d+(?:\.\d+)?)\s*", value)
        if not match:
            return None
        result = float(match.group(1))
    else:
        return None
    return result if np.isfinite(result) and result >= 0.0 else None


def one_division_tolerance(role: str, selected_range: Any) -> float:
    numeric_range = normalize_decimal(selected_range)
    if numeric_range is None:
        numeric_range = 3.0 if role == "voltmeter" else 0.6
    floor = 0.05 if role == "voltmeter" else 0.01
    return round(max(floor, numeric_range / 30.0 * 1.25), 6)


def compare_value(
    paper_value: Any, meter_value: Any, role: str, selected_range: Any
) -> dict[str, Any]:
    paper = normalize_decimal(paper_value)
    meter = normalize_decimal(meter_value)
    tolerance = one_division_tolerance(role, selected_range)
    difference = None if paper is None or meter is None else abs(paper - meter)
    matched = difference is not None and difference <= tolerance + 1e-9
    return {
        "paper_value": paper,
        "meter_value": meter,
        "absolute_difference": None if difference is None else round(difference, 6),
        "tolerance": tolerance,
        "matched": bool(matched),
    }


def parse_model_json(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("qwen_json_object_missing")
        text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("qwen_json_object_expected")
    return value


def validate_observation(value: dict[str, Any]) -> dict[str, Any]:
    consensus = value.get("consensus")
    if not isinstance(consensus, dict):
        raise ValueError("qwen_consensus_missing")
    for role in ("ammeter", "voltmeter"):
        meter = consensus.get(role)
        if not isinstance(meter, dict):
            raise ValueError(f"qwen_{role}_missing")
        meter["value"] = normalize_decimal(meter.get("value"))
        meter["selected_range"] = normalize_decimal(meter.get("selected_range"))
        confidence = meter.get("confidence", 0.0)
        meter["confidence"] = min(1.0, max(0.0, float(confidence or 0.0)))
    frames = value.get("per_frame")
    if not isinstance(frames, list):
        value["per_frame"] = []
    value["evidence"] = str(value.get("evidence") or "")
    return value


def resize_long_side(image: np.ndarray, maximum: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, maximum / max(width, height))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def enhance_meter_crop(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)


def add_information_bar(
    image: np.ndarray,
    frame_number: int,
    timestamp_seconds: float,
    label_suffix: str = "METER ROI",
) -> np.ndarray:
    bar_height = max(64, int(round(image.shape[0] * 0.065)))
    output = cv2.copyMakeBorder(
        image, 0, bar_height, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18)
    )
    label = (
        f"FRAME {frame_number:08d} | VIDEO T={timestamp_seconds:.3f}s | {label_suffix}"
    )
    font_scale = max(0.8, min(1.8, image.shape[1] / 1300.0))
    cv2.putText(
        output,
        label,
        (24, image.shape[0] + int(bar_height * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        max(2, int(round(font_scale * 2))),
        cv2.LINE_AA,
    )
    return output


def extract_evidence_frames(
    spec: VideoSpec, video_root: Path, video_dir: Path
) -> list[dict[str, Any]]:
    source = resolve_source_video(video_root, spec.source_video)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"video_open_failed:{source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0.0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"video_metadata_invalid:{source}")
    evidence_dir = video_dir / "meter_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    try:
        for ordinal, requested_seconds in enumerate(spec.timestamps, start=1):
            requested_frame = min(
                frame_count - 1, max(0, int(round(requested_seconds * fps)))
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, requested_frame)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"frame_decode_failed:{spec.video_id}:{requested_frame}"
                )
            actual_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp = actual_frame / fps
            height, width = frame.shape[:2]
            x1n, y1n, x2n, y2n = spec.meter_roi_normalized_xyxy
            x1, y1 = int(round(x1n * width)), int(round(y1n * height))
            x2, y2 = int(round(x2n * width)), int(round(y2n * height))
            crop = frame[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
            if crop.size == 0:
                raise RuntimeError(f"empty_meter_roi:{spec.video_id}:{actual_frame}")
            crop = resize_long_side(enhance_meter_crop(crop), 2200)
            prepared = add_information_bar(crop, actual_frame, timestamp)
            path = evidence_dir / (
                f"{ordinal:02d}_frame_{actual_frame:08d}_{timestamp:010.3f}s_meter.jpg"
            )
            if not cv2.imwrite(str(path), prepared, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                raise RuntimeError(f"frame_write_failed:{path}")
            rows.append(
                {
                    "frame_id": f"frame_{actual_frame:08d}",
                    "frame_number": actual_frame,
                    "requested_timestamp_seconds": requested_seconds,
                    "timestamp_seconds": round(timestamp, 6),
                    "meter_roi_normalized_xyxy": list(
                        spec.meter_roi_normalized_xyxy
                    ),
                    "image_path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
            )
    finally:
        capture.release()
    return rows


def extract_precision_evidence(
    spec: VideoSpec, video_root: Path, video_dir: Path
) -> list[dict[str, Any]]:
    source = resolve_source_video(video_root, spec.source_video)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"video_open_failed:{source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    requested_frame = min(
        frame_count - 1, max(0, int(round(spec.precision_timestamp * fps)))
    )
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, requested_frame)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"precision_frame_decode_failed:{spec.video_id}:{requested_frame}"
            )
        actual_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        timestamp = actual_frame / fps
    finally:
        capture.release()
    height, width = frame.shape[:2]
    precision_dir = video_dir / "precision_evidence"
    precision_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    role_rois = (
        ("ammeter", spec.ammeter_roi_normalized_xyxy),
        ("voltmeter", spec.voltmeter_roi_normalized_xyxy),
    )
    for role, normalized_roi in role_rois:
        x1n, y1n, x2n, y2n = normalized_roi
        x1, y1 = int(round(x1n * width)), int(round(y1n * height))
        x2, y2 = int(round(x2n * width)), int(round(y2n * height))
        crop = frame[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
        if crop.size == 0:
            raise RuntimeError(f"empty_precision_roi:{spec.video_id}:{role}")
        crop = resize_long_side(enhance_meter_crop(crop), 1800)
        prepared = add_information_bar(
            crop, actual_frame, timestamp, f"{role.upper()} PRECISION ROI"
        )
        path = precision_dir / (
            f"frame_{actual_frame:08d}_{timestamp:010.3f}s_{role}.jpg"
        )
        if not cv2.imwrite(str(path), prepared, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"precision_frame_write_failed:{path}")
        rows.append(
            {
                "role": role,
                "frame_id": f"frame_{actual_frame:08d}",
                "frame_number": actual_frame,
                "timestamp_seconds": round(timestamp, 6),
                "roi_normalized_xyxy": list(normalized_roi),
                "image_path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    return rows


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_prompt(frame_rows: list[dict[str, Any]]) -> str:
    frame_list = ", ".join(
        f"{row['frame_id']}@{row['timestamp_seconds']:.3f}s" for row in frame_rows
    )
    return f"""你是伏安法测电阻实验的模拟电表读数核验器。输入是同一次第一轮测量末段的 4 张连续仪表区域图：{frame_list}。每张图底部都有视频相对时间和 frame_id。图中没有提供纸面标准答案，你也不得尝试从记录纸、动作顺序或实验常识反推读数。

目标仪表说明：
1. 橙红色外壳、表盘中央写 A 的是电流表；其接线面板通常偏绿色。表盘同时印有 0-3 A 外刻度和 0-0.6 A 内刻度。必须先看实际插头接在 0.6 A 还是 3 A 量程端子，再选择对应刻度。
2. 橙红色外壳、表盘中央写 V 的是电压表；其接线面板通常偏红色。表盘同时印有 0-15 V 外刻度和 0-3 V 内刻度。必须先看实际插头接在 3 V 还是 15 V 量程端子，再选择对应刻度。
3. 指针是从表盘转轴延伸到刻度弧的细直线。不要把红色导线、刻度线、玻璃反光、阴影或表壳边缘当作指针。
4. 先逐帧检查可见性、量程端子和指针位置，再用相邻帧中重复出现的稳定位置形成一致读数。短暂运动模糊或手部遮挡不能覆盖其他清晰帧。
5. 如果至少一帧能基本读取，就给出最可能数值和较低置信度；只有四帧中目标表都完全不在画面内时，value 才可为 null。
6. 只做视觉读数，不评价学生、不读取纸面、不输出 pass/fail。

只返回一个合法 JSON 对象，不要 Markdown：
{{
  "per_frame": [
    {{
      "frame_id": "frame_00000000",
      "timestamp_seconds": 0.0,
      "ammeter": {{"visible": true, "selected_range": 0.6, "scale_reading": 0.0, "value": 0.0, "confidence": 0.0, "evidence": "可见依据"}},
      "voltmeter": {{"visible": true, "selected_range": 3.0, "scale_reading": 0.0, "value": 0.0, "confidence": 0.0, "evidence": "可见依据"}}
    }}
  ],
  "consensus": {{
    "ammeter": {{"selected_range": 0.6, "value": 0.0, "confidence": 0.0, "supporting_frame_ids": ["frame_00000000"], "evidence": "量程和稳定指针依据"}},
    "voltmeter": {{"selected_range": 3.0, "value": 0.0, "confidence": 0.0, "supporting_frame_ids": ["frame_00000000"], "evidence": "量程和稳定指针依据"}}
  }},
  "evidence": "两表的总体直接观察说明"
}}"""


def call_qwen(
    client: OpenAI,
    frame_rows: list[dict[str, Any]],
    raw_path: Path,
    retries: int,
) -> dict[str, Any]:
    prompt = build_prompt(frame_rows)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for row in frame_rows:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(Path(row["image_path"]))},
            }
        )
    errors: list[dict[str, str]] = []
    for attempt in range(1, retries + 2):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=1800,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(raw_path, completion.model_dump(mode="json"))
            message = completion.choices[0].message.content or ""
            return {
                "attempt": attempt,
                "finish_reason": completion.choices[0].finish_reason,
                "observation": validate_observation(parse_model_json(message)),
                "prior_errors": errors,
                "prompt": prompt,
            }
        except Exception as error:
            errors.append({"type": type(error).__name__, "message": str(error)})
    raise RuntimeError(f"qwen_request_failed:{errors}")


def build_precision_prompt(rows: list[dict[str, Any]]) -> str:
    frame_id = rows[0]["frame_id"]
    timestamp = float(rows[0]["timestamp_seconds"])
    return f"""你是模拟电表精细读数核验器。两张图来自同一源帧 {frame_id}、视频相对时间 {timestamp:.3f}s：图1只裁出电流表 A，图2只裁出电压表 V。图中没有纸面答案，不得从实验常识推断。

必须遵守以下读表规则：
1. 表盘零位是印刷刻度弧的最左端，不是画面竖直方向，也不是刻度弧中点。仪表可能倾斜，必须看指针与印刷刻度弧的交点。不要把表盘顶部或正中位置误读为零。
2. A 表三孔从左到右为公共负端、中间 0.6A、右端 3A。V 表三孔从左到右为公共负端、中间 3V、右端 15V。先逐孔报告 occupied/empty；右端空着时不得选择高量程。
3. A 表印刷主刻度沿弧从左到右依次为：外圈 `0, 1, 2, 3`，内圈 `0, 0.2, 0.4, 0.6`。所以指针若穿过印刷的 `1 / 0.2` 主刻度，即使它在图中恰好竖直，也应读为 0.2A，不是零。
4. V 表印刷主刻度沿弧从左到右依次为：外圈 `0, 5, 10, 15`，内圈 `0, 1, 2, 3`。所以指针若穿过印刷的 `5 / 1` 主刻度，即使它在图中恰好竖直，也应读为 1V，不是零。
5. A 表同时报告外圈 0-3 的原始位置和内圈 0-0.6 的原始位置。V 表同时报告外圈 0-15 和内圈 0-3 的原始位置。
6. 指针是从圆形转轴向刻度弧延伸的细线。红色导线、反光边缘、印刷刻度线不是指针。
7. `value` 必须由实际占用插孔选择量程后得到，不能直接抄另一圈刻度。

只返回一个合法 JSON，不要 Markdown：
{{
  "ammeter": {{
    "terminal_occupancy_left_middle_right": ["occupied", "occupied", "empty"],
    "selected_range": 0.6,
    "outer_0_to_3_position": 0.0,
    "inner_0_to_0_6_position": 0.0,
    "value": 0.0,
    "confidence": 0.0,
    "evidence": "插孔和指针交点"
  }},
  "voltmeter": {{
    "terminal_occupancy_left_middle_right": ["occupied", "occupied", "empty"],
    "selected_range": 3.0,
    "outer_0_to_15_position": 0.0,
    "inner_0_to_3_position": 0.0,
    "value": 0.0,
    "confidence": 0.0,
    "evidence": "插孔和指针交点"
  }},
  "evidence": "总体直接观察"
}}"""


def call_qwen_precision(
    client: OpenAI,
    rows: list[dict[str, Any]],
    raw_path: Path,
    retries: int,
) -> dict[str, Any]:
    prompt = build_precision_prompt(rows)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for row in rows:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(Path(row["image_path"]))},
            }
        )
    errors: list[dict[str, str]] = []
    for attempt in range(1, retries + 2):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=1200,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            write_json(raw_path, completion.model_dump(mode="json"))
            parsed = parse_model_json(completion.choices[0].message.content or "")
            for role in ("ammeter", "voltmeter"):
                meter = parsed.get(role)
                if not isinstance(meter, dict):
                    raise ValueError(f"precision_{role}_missing")
                meter["selected_range"] = normalize_decimal(
                    meter.get("selected_range")
                )
                meter["value"] = normalize_decimal(meter.get("value"))
                meter["confidence"] = min(
                    1.0, max(0.0, float(meter.get("confidence") or 0.0))
                )
            parsed["evidence"] = str(parsed.get("evidence") or "")
            return {
                "attempt": attempt,
                "finish_reason": completion.choices[0].finish_reason,
                "observation": parsed,
                "prior_errors": errors,
                "prompt": prompt,
            }
        except Exception as error:
            errors.append({"type": type(error).__name__, "message": str(error)})
    raise RuntimeError(f"qwen_precision_request_failed:{errors}")


def paper_rows_by_id(
    summary: dict[str, Any], specs: tuple[VideoSpec, ...]
) -> dict[str, dict[str, Any]]:
    rows = summary.get("results")
    if not isinstance(rows, list):
        raise ValueError("paper_summary_results_missing")
    result = {
        str(row.get("video_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("video_id") is not None
    }
    missing = {spec.video_id for spec in specs} - set(result)
    if missing:
        raise ValueError(f"paper_summary_video_missing:{sorted(missing)}")
    return result


def reduce_result(
    spec: VideoSpec,
    paper: dict[str, Any],
    qwen_result: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    precision_result: dict[str, Any],
    precision_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    observation = qwen_result["observation"]
    precision = precision_result["observation"]
    ammeter = precision["ammeter"]
    voltmeter = precision["voltmeter"]
    voltage = compare_value(
        paper.get("u_value"),
        voltmeter.get("value"),
        "voltmeter",
        voltmeter.get("selected_range"),
    )
    current = compare_value(
        paper.get("i_value"),
        ammeter.get("value"),
        "ammeter",
        ammeter.get("selected_range"),
    )
    passed = voltage["matched"] and current["matched"]
    confidence = min(float(ammeter["confidence"]), float(voltmeter["confidence"]))
    if paper.get("u_value") is None or paper.get("i_value") is None:
        confidence = max(confidence, float(paper.get("confidence") or 0.0))
    return {
        "video_id": spec.video_id,
        "result": "pass" if passed else "fail",
        "predicted_score": 1 if passed else 0,
        "confidence": round(confidence, 3),
        "paper": {
            "u_value": paper.get("u_value"),
            "i_value": paper.get("i_value"),
            "source_result": paper.get("result"),
            "evidence_seconds": paper.get("evidence_seconds", []),
        },
        "meters": {
            "voltmeter": voltmeter,
            "ammeter": ammeter,
            "evidence": precision.get("evidence"),
            "temporal_consensus": observation.get("consensus"),
            "temporal_evidence": observation.get("evidence"),
        },
        "comparison": {"voltage": voltage, "current": current},
        "meter_evidence_frames": frame_rows,
        "precision_evidence": precision_rows,
        "qwen_attempt": qwen_result["attempt"],
        "qwen_prior_errors": qwen_result["prior_errors"],
        "qwen_precision_attempt": precision_result["attempt"],
        "qwen_precision_prior_errors": precision_result["prior_errors"],
        "reason": (
            "纸面 U1/I1 与第一轮测量末段的电压表、电流表读数均在一个最小分度容差内。"
            if passed
            else "纸面字段缺失，或至少一个纸面数值与同轮仪表读数超出一个最小分度容差。"
        ),
    }


def write_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# 第一组纸面记录与仪表读数一致性核验 v1",
        "",
        "Qwen 只接收第一轮书写前的仪表区域，不接收纸面候选值。程序在本地比较 U1/I1。",
        "",
        "| 视频 ID | 纸面 U1/I1 | 仪表 U/I | 差值 U/I | 结果 | 置信度 |",
        "|---|---|---|---|---|---:|",
    ]
    for row in results:
        paper = row["paper"]
        meters = row["meters"]
        comparison = row["comparison"]
        pu = paper["u_value"] if paper["u_value"] is not None else "未读出"
        pi = paper["i_value"] if paper["i_value"] is not None else "未读出"
        mu = meters["voltmeter"].get("value")
        mi = meters["ammeter"].get("value")
        du = comparison["voltage"].get("absolute_difference")
        di = comparison["current"].get("absolute_difference")
        lines.append(
            f"| {row['video_id']} | {pu} V / {pi} A | "
            f"{mu} V / {mi} A | {du} V / {di} A | `{row['result']}` | {row['confidence']:.2f} |"
        )
    lines.extend(
        [
            "",
            "容差按所选量程的约 30 小格计算，并放宽到 1.25 个最小分度；字段缺失或超差均输出 fail。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-config", type=Path, default=DEFAULT_SPEC_CONFIG)
    parser.add_argument("--paper-summary", type=Path, default=DEFAULT_PAPER_SUMMARY)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-ids", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    spec_config = args.spec_config.expanduser().resolve()
    paper_summary = args.paper_summary.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing_to_overwrite_nonempty_output:{output}")
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(paper_summary)
    spec_hash = sha256_file(spec_config)
    shutil.copy2(paper_summary, output / "source_summary.original.json")
    shutil.copy2(spec_config, output / "source_specs.original.json")
    all_specs = load_video_specs(spec_config)
    requested = set(args.video_ids or [spec.video_id for spec in all_specs])
    known = {spec.video_id for spec in all_specs}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown_video_ids:{sorted(unknown)}")
    specs = [spec for spec in all_specs if spec.video_id in requested]
    paper_by_id = paper_rows_by_id(read_json(paper_summary), tuple(specs))

    client = None
    if not args.dry_run:
        missing = [
            name
            for name, value in (
                ("QWEN_API_BASE_URL", API_BASE_URL),
                ("QWEN_API_TOKEN", API_TOKEN),
            )
            if not value
        ]
        if missing:
            raise RuntimeError("missing_qwen_configuration:" + ",".join(missing))
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_TOKEN,
            timeout=180.0,
            max_retries=0,
        )
    results: list[dict[str, Any]] = []
    request_manifest: list[dict[str, Any]] = []
    for spec in specs:
        video_dir = output / f"video_{spec.video_id}"
        frame_rows = extract_evidence_frames(spec, video_root, video_dir)
        precision_rows = extract_precision_evidence(spec, video_root, video_dir)
        manifest_row = {
            "video_id": spec.video_id,
            "source_video": str(resolve_source_video(video_root, spec.source_video)),
            "paper_values_sent_to_qwen": False,
            "student_identity_sent_to_qwen": False,
            "frames": frame_rows,
            "precision_frames": precision_rows,
            "prompt": build_prompt(frame_rows),
            "precision_prompt": build_precision_prompt(precision_rows),
        }
        request_manifest.append(manifest_row)
        if args.dry_run:
            continue
        assert client is not None
        qwen_result = call_qwen(
            client,
            frame_rows,
            video_dir / "raw_response.json",
            max(0, int(args.retries)),
        )
        write_json(video_dir / "parsed_observation.json", qwen_result["observation"])
        precision_result = call_qwen_precision(
            client,
            precision_rows,
            video_dir / "precision_raw_response.json",
            max(0, int(args.retries)),
        )
        write_json(
            video_dir / "precision_observation.json",
            precision_result["observation"],
        )
        result = reduce_result(
            spec,
            paper_by_id[spec.video_id],
            qwen_result,
            frame_rows,
            precision_result,
            precision_rows,
        )
        write_json(video_dir / "result.json", result)
        results.append(result)

    write_json(
        output / "request_manifest.json",
        {
            "schema_version": "meter_record_consistency_request.v1",
            "generated_at": utc_now(),
            "paper_summary_sha256": source_hash,
            "spec_config_sha256": spec_hash,
            "qwen_received_paper_values": False,
            "requests": request_manifest,
        },
    )
    report = {
        "schema_version": "meter_record_consistency.v1",
        "artifact_type": "first_record_meter_consistency_binary_results",
        "generated_at": utc_now(),
        "status": "prepared" if args.dry_run else "completed",
        "source_paper_summary": str(paper_summary),
        "source_paper_summary_sha256": source_hash,
        "source_spec_config_sha256": spec_hash,
        "qwen_received_paper_values": False,
        "comparison_policy": (
            "pass only when both U1 and I1 match the independently observed meters "
            "within 1.25 minor divisions; otherwise fail"
        ),
        "results": results,
        "counts": {
            "videos": len(specs),
            "pass": sum(row["result"] == "pass" for row in results),
            "fail": sum(row["result"] == "fail" for row in results),
            "qwen_requests": 0 if args.dry_run else len(results) * 2,
        },
    }
    write_json(output / "meter_record_consistency.json", report)
    if not args.dry_run:
        write_markdown(output / "README.md", results)
    print(json.dumps(report["counts"], ensure_ascii=False))
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
