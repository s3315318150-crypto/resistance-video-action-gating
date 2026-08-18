#!/usr/bin/env python3
"""Run online Qwen A/B tests for the three P0 temporal retrieval ideas.

This is an isolated experiment.  It consumes saved stage plans and a separate
gold comparison file, reads source videos without modifying them, and writes a
new output directory.  Gold intervals are used only for local metrics and are
never included in model prompts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

import cv2


SCHEMA_VERSION = "night_exploration.online_p0_ab.v1"
DEFAULT_ENDPOINT = "https://cossin.ecnu.edu.cn/skill/api/qwen/v1"
DEFAULT_MODEL = "qwen"
TOLERANCE_SECONDS = 2.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def safe_slug(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(value).stem).strip("._-")
    return f"{prefix[:36] or 'video'}__{digest}"


def resolve_video(source_video_id: str, roots: list[Path]) -> Path:
    exact = [root / source_video_id for root in roots]
    matches = [path.resolve() for path in exact if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    stem = Path(source_video_id).stem
    discovered = []
    for root in roots:
        discovered.extend(path.resolve() for path in root.glob(f"{stem}.*") if path.is_file())
    unique = sorted(set(discovered))
    if len(unique) != 1:
        raise FileNotFoundError(f"video_resolution_failed:{source_video_id}:matches={len(unique)}")
    return unique[0]


def source_records(plan_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = plan_payload.get("records")
    if isinstance(records, list):
        return [item for item in records if isinstance(item, dict)]
    raise ValueError("plan_records_missing")


def cleanup_gold(gold_payload: dict[str, Any]) -> dict[str, tuple[float, float]]:
    output: dict[str, tuple[float, float]] = {}
    for record in gold_payload.get("records", []):
        video_id = str(record.get("source_video_id") or "")
        for run in record.get("actual_stage_runs", []):
            if isinstance(run, list) and len(run) >= 3 and run[0] == "material_cleanup":
                output[video_id] = (float(run[1]), float(run[2]))
    return output


def rubric_zero_plan(record: dict[str, Any]) -> dict[str, Any]:
    matches = [item for item in record.get("rubric_plans", []) if int(item.get("rubric_id", -1)) == 0]
    if len(matches) != 1:
        raise ValueError(f"rubric_zero_plan_count:{len(matches)}")
    return matches[0]


def planned_times(plan: dict[str, Any]) -> list[float]:
    values = {
        round(float(value), 3)
        for window in plan.get("candidate_windows", [])
        for value in window.get("planned_sample_times_seconds", [])
    }
    return sorted(values)


def uniform_times(start: float, end: float, count: int) -> list[float]:
    if count <= 0 or end < start:
        raise ValueError("invalid_uniform_request")
    if count == 1 or math.isclose(start, end):
        return [round((start + end) / 2.0, 3)]
    return [round(start + index * (end - start) / (count - 1), 3) for index in range(count)]


def partition(start: float, end: float, count: int, prefix: str) -> list[dict[str, Any]]:
    if end <= start or count <= 0:
        raise ValueError("invalid_partition")
    width = (end - start) / count
    clips = []
    for index in range(count):
        clip_start = start + index * width
        clip_end = end if index == count - 1 else start + (index + 1) * width
        clips.append(
            {
                "clip_id": f"{prefix}_c{index + 1:02d}",
                "start_seconds": clip_start,
                "end_seconds": clip_end,
                "sample_seconds": [
                    clip_start + fraction * (clip_end - clip_start)
                    for fraction in (0.2, 0.5, 0.8)
                ],
            }
        )
    return clips


def resize(frame: Any, max_edge: int) -> Any:
    height, width = frame.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def add_watermark(frame: Any, frame_id: str, timestamp: float) -> Any:
    label = f"VIDEO T={timestamp:.3f}s | FRAME ID={frame_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.52, min(0.82, frame.shape[1] / 1050.0))
    thickness = 2
    (_, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    y = frame.shape[0] - 12
    top = max(0, y - text_height - baseline - 12)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0, frame)
    cv2.putText(frame, label, (12, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


class FrameExtractor:
    def __init__(self, video: Path, output: Path, max_edge: int) -> None:
        self.video = video
        self.output = output
        self.max_edge = max_edge
        self.output.mkdir(parents=True, exist_ok=True)
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise OSError(f"video_open_failed:{video}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cache: dict[tuple[str, int], dict[str, Any]] = {}

    def close(self) -> None:
        self.capture.release()

    def extract(self, method: str, frame_id: str, timestamp: float) -> dict[str, Any]:
        frame_number = min(self.frame_count - 1, max(0, int(round(timestamp * self.fps))))
        key = (method, frame_number)
        if key in self.cache:
            previous = self.cache[key]
            return {**previous, "frame_id": frame_id}
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise OSError(f"frame_read_failed:{self.video}:{frame_number}")
        actual = round(frame_number / self.fps, 6)
        image = add_watermark(resize(frame, self.max_edge), frame_id, actual)
        target_dir = self.output / method
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{frame_id}_{frame_number:08d}_{actual:010.3f}s.jpg"
        if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise OSError(f"frame_write_failed:{target}")
        record = {
            "frame_id": frame_id,
            "timestamp_seconds": actual,
            "frame_number": frame_number,
            "path": str(target.resolve()),
        }
        self.cache[key] = record
        return record

    def extract_times(self, method: str, prefix: str, times: list[float]) -> list[dict[str, Any]]:
        return [
            self.extract(method, f"{prefix}_{index + 1:03d}", timestamp)
            for index, timestamp in enumerate(times)
        ]


def parse_json_text(text: str) -> dict[str, Any]:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(clean[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response_not_object")
    return value


def image_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class QwenClient:
    def __init__(self, endpoint: str, token: str, model: str, timeout: float) -> None:
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.token = token
        self.model = model
        self.timeout = timeout
        self.call_count = 0
        self.image_exposures = 0

    def call_json(
        self,
        prompt: str,
        frames: list[dict[str, Any]],
        validator: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        last_error = "unknown"
        attempts = []
        for attempt_index in range(2):
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": image_url(Path(frame["path"]))}}
                for frame in frames
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "stream": False,
                "max_tokens": 1600,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            started = time.monotonic()
            self.call_count += 1
            self.image_exposures += len(frames)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw_response = json.loads(response.read().decode("utf-8"))
                raw_content = str(raw_response["choices"][0]["message"].get("content") or "")
                parsed = parse_json_text(raw_content)
                valid = validator(parsed) if validator else True
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "raw_model_content": raw_content,
                        "parsed_result": parsed,
                        "valid": bool(valid),
                        "usage": raw_response.get("usage"),
                    }
                )
                if valid:
                    return {"status": "valid", "result": parsed, "attempts": attempts}
                last_error = "validator_rejected_response"
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}:{exc}"
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "error": last_error,
                        "valid": False,
                    }
                )
        return {"status": "failed", "error": last_error, "attempts": attempts}


def direct_prompt(frame_ids: list[str], method_label: str) -> str:
    return f"""你正在判断伏安法测电阻视频中的最终整理材料阶段。图片按视频时间顺序给出，合法 FRAME ID 为：{', '.join(frame_ids)}。

目标动作必须依据画面直接可见事实判断：学生拆卸或断开导线、收拢仪器，并把橙红色仪器放回桌面左上角；换座位可以作为实验结束的辅助证据。普通挪动一件器材不等于整理完成。

这是 {method_label} 取帧。请选择：
1. start_frame_id：第一张明确看到最终拆线或整理已经开始的图片；
2. end_frame_id：第一张明确看到整理已经完成、仪器已归位桌面左上角或学生已换座位的图片；
3. rubric_decision：观察到完成态时为 pass，否则为 fail。

只能复制给出的 FRAME ID。不要输出秒数，不要根据实验步骤猜测被遮挡的动作。即使证据较弱也必须在 pass/fail 中选择最可能的一项。

只输出 JSON：
{{"start_frame_id":"..."|null,"end_frame_id":"..."|null,"rubric_decision":"pass"|"fail","confidence":0.0,"visible_evidence":"..."}}"""


def validate_direct(value: dict[str, Any], frame_ids: set[str]) -> bool:
    start = value.get("start_frame_id")
    end = value.get("end_frame_id")
    return (
        value.get("rubric_decision") in {"pass", "fail"}
        and (start is None or start in frame_ids)
        and (end is None or end in frame_ids)
    )


def frame_time(frame_id: Any, frames: list[dict[str, Any]]) -> float | None:
    by_id = {frame["frame_id"]: float(frame["timestamp_seconds"]) for frame in frames}
    return by_id.get(frame_id)


def run_direct(
    client: QwenClient | None,
    frames: list[dict[str, Any]],
    method_label: str,
) -> dict[str, Any]:
    if client is None:
        return {"status": "prepared", "frames": frames}
    ids = [frame["frame_id"] for frame in frames]
    response = client.call_json(
        direct_prompt(ids, method_label),
        frames,
        lambda value: validate_direct(value, set(ids)),
    )
    result = response.get("result") or {}
    return {
        "status": response["status"],
        "frames": frames,
        "prompt": direct_prompt(ids, method_label),
        "qwen": response,
        "predicted_start_seconds": frame_time(result.get("start_frame_id"), frames),
        "predicted_end_seconds": frame_time(result.get("end_frame_id"), frames),
        "rubric_decision": result.get("rubric_decision", "fail"),
    }


def clip_batch_prompt(clips: list[dict[str, Any]], boundary_kind: str) -> str:
    if boundary_kind == "start":
        target = "最终整理材料开始的转折：从继续实验转为拆卸导线或集中收拢仪器"
    else:
        target = "最终整理材料完成的转折：导线和仪器已经整理完，橙红色仪器已回桌面左上角，或学生换座位且不再实验"
    mapping = "\n".join(
        f"- {clip['clip_id']}: {', '.join(frame['frame_id'] for frame in clip['frames'])}"
        for clip in clips
    )
    return f"""以下图片分属于多个互不重叠、按时间排列的短片段。目标是：{target}。

片段与合法 FRAME ID：
{mapping}

对每个片段只回答该转折是否直接出现在片段的三张图片中。不要输出或猜测秒数；不要因为片段位于视频后部就判 yes。target_probability 是画面包含该转折的概率，必须在 0 到 1 之间。selected_frame_id 只能从该片段的三个 ID 中选择最接近转折的一张；没有直接证据时为 null。

只输出 JSON：
{{"clips":[{{"clip_id":"...","answer":"yes"|"no","target_probability":0.0,"selected_frame_id":"..."|null,"evidence":"..."}}]}}"""


def validate_clip_scores(value: dict[str, Any], clips: list[dict[str, Any]]) -> bool:
    rows = value.get("clips")
    if not isinstance(rows, list):
        return False
    expected = {clip["clip_id"] for clip in clips}
    observed = {str(row.get("clip_id")) for row in rows if isinstance(row, dict)}
    if observed != expected:
        return False
    frame_ids = {
        clip["clip_id"]: {frame["frame_id"] for frame in clip["frames"]}
        for clip in clips
    }
    for row in rows:
        try:
            probability = float(row["target_probability"])
        except (KeyError, TypeError, ValueError):
            return False
        if not 0.0 <= probability <= 1.0 or row.get("answer") not in {"yes", "no"}:
            return False
        selected = row.get("selected_frame_id")
        if selected is not None and selected not in frame_ids[str(row["clip_id"])]:
            return False
    return True


def materialize_clips(
    extractor: FrameExtractor,
    method: str,
    clips: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for clip in clips:
        item = dict(clip)
        item["frames"] = [
            extractor.extract(method, f"{clip['clip_id']}_f{index + 1}", timestamp)
            for index, timestamp in enumerate(clip["sample_seconds"])
        ]
        output.append(item)
    return output


def run_binary_boundary_scan(
    client: QwenClient | None,
    extractor: FrameExtractor,
    extent: tuple[float, float],
    boundary_kind: str,
) -> dict[str, Any]:
    candidates = partition(extent[0], extent[1], 8, f"{boundary_kind}_r0")
    rounds = []
    for round_index in range(3):
        materialized = materialize_clips(extractor, f"yes_no_{boundary_kind}_r{round_index}", candidates)
        prompt = clip_batch_prompt(materialized, boundary_kind)
        if client is None:
            rounds.append({"round": round_index, "clips": materialized, "prompt": prompt, "status": "prepared"})
            break
        all_frames = [frame for clip in materialized for frame in clip["frames"]]
        response = client.call_json(
            prompt,
            all_frames,
            lambda value: validate_clip_scores(value, materialized),
        )
        if response["status"] != "valid":
            rounds.append({"round": round_index, "clips": materialized, "prompt": prompt, "qwen": response})
            return {"status": "failed", "rounds": rounds, "predicted_seconds": None}
        rows = response["result"]["clips"]
        scores = {str(row["clip_id"]): float(row["target_probability"]) for row in rows}
        selected = sorted(
            materialized,
            key=lambda clip: (-scores[clip["clip_id"]], float(clip["start_seconds"]), clip["clip_id"]),
        )[0]
        selected_row = next(row for row in rows if row["clip_id"] == selected["clip_id"])
        rounds.append(
            {
                "round": round_index,
                "clips": materialized,
                "prompt": prompt,
                "qwen": response,
                "selected_clip_id": selected["clip_id"],
                "selected_probability": scores[selected["clip_id"]],
            }
        )
        if round_index < 2:
            candidates = partition(
                float(selected["start_seconds"]),
                float(selected["end_seconds"]),
                4,
                f"{boundary_kind}_r{round_index + 1}",
            )
    if client is None:
        return {"status": "prepared", "rounds": rounds, "predicted_seconds": None}
    selected_frame = selected_row.get("selected_frame_id")
    predicted = frame_time(selected_frame, selected["frames"])
    if predicted is None:
        predicted = (float(selected["start_seconds"]) + float(selected["end_seconds"])) / 2.0
    return {
        "status": "valid",
        "rounds": rounds,
        "predicted_seconds": round(predicted, 6),
        "final_clip": {
            "clip_id": selected["clip_id"],
            "start_seconds": selected["start_seconds"],
            "end_seconds": selected["end_seconds"],
            "selected_frame_id": selected_frame,
        },
    }


def run_yes_no(
    client: QwenClient | None,
    extractor: FrameExtractor,
    extent: tuple[float, float],
) -> dict[str, Any]:
    start_scan = run_binary_boundary_scan(client, extractor, extent, "start")
    end_scan = run_binary_boundary_scan(client, extractor, extent, "end")
    status = "valid" if start_scan["status"] == end_scan["status"] == "valid" else (
        "prepared" if client is None else "failed"
    )
    return {
        "status": status,
        "start_scan": start_scan,
        "end_scan": end_scan,
        "predicted_start_seconds": start_scan.get("predicted_seconds"),
        "predicted_end_seconds": end_scan.get("predicted_seconds"),
        "rubric_decision": "pass" if status == "valid" else "fail",
    }


def boundary_prompt(frame_ids: list[str], kind: str) -> str:
    if kind == "start":
        definition = "最终整理开始：学生从继续实验转为拆卸导线或集中收拢器材"
        fields = '"last_before_frame_id":"..."|null,"first_during_frame_id":"..."|null'
    else:
        definition = "最终整理完成：线路和器材已经整理完，橙红色仪器已放回桌面左上角，或学生换座位且不再继续实验"
        fields = '"last_during_frame_id":"..."|null,"first_after_frame_id":"..."|null'
    return f"""这些图片来自候选边界附近，按时间顺序排列。合法 FRAME ID：{', '.join(frame_ids)}。

请为以下边界分别绑定边界前和边界后的直接视觉证据：{definition}。
只能复制合法 FRAME ID，不要输出秒数，不要按实验常识猜测。若相邻抽帧之间发生转折，分别选择转折前最后一张和转折后第一张。

只输出 JSON：{{{fields},"confidence":0.0,"visible_evidence":"..."}}"""


def run_boundary_binding(
    client: QwenClient | None,
    extractor: FrameExtractor,
    guided: dict[str, Any],
    extent: tuple[float, float],
) -> dict[str, Any]:
    predicted_start = guided.get("predicted_start_seconds")
    predicted_end = guided.get("predicted_end_seconds")
    if predicted_start is None or predicted_end is None:
        return {"status": "not_run", "reason": "guided_direct_boundary_missing"}
    passes = {}
    predictions = {}
    for kind, center in (("start", float(predicted_start)), ("end", float(predicted_end))):
        lower = max(extent[0], center - 5.0)
        upper = min(extent[1], center + 5.0)
        count = max(2, int(math.floor(upper - lower)) + 1)
        frames = extractor.extract_times(
            f"boundary_{kind}", f"boundary_{kind}", uniform_times(lower, upper, count)
        )
        ids = [frame["frame_id"] for frame in frames]
        prompt = boundary_prompt(ids, kind)
        if client is None:
            passes[kind] = {"status": "prepared", "frames": frames, "prompt": prompt}
            continue
        valid_fields = (
            ("last_before_frame_id", "first_during_frame_id")
            if kind == "start" else ("last_during_frame_id", "first_after_frame_id")
        )
        response = client.call_json(
            prompt,
            frames,
            lambda value, fields=valid_fields, allowed=set(ids): all(
                value.get(field) is None or value.get(field) in allowed for field in fields
            ),
        )
        result = response.get("result") or {}
        selected_field = "first_during_frame_id" if kind == "start" else "first_after_frame_id"
        predictions[kind] = frame_time(result.get(selected_field), frames)
        passes[kind] = {"status": response["status"], "frames": frames, "prompt": prompt, "qwen": response}
    if client is None:
        return {"status": "prepared", "passes": passes}
    return {
        "status": "valid" if all(item["status"] == "valid" for item in passes.values()) else "failed",
        "passes": passes,
        "predicted_start_seconds": predictions.get("start") or predicted_start,
        "predicted_end_seconds": predictions.get("end") or predicted_end,
        "rubric_decision": guided.get("rubric_decision", "fail"),
    }


def error_metrics(method: dict[str, Any], gold: tuple[float, float]) -> dict[str, Any]:
    start = method.get("predicted_start_seconds")
    end = method.get("predicted_end_seconds")
    start_error = abs(float(start) - gold[0]) if start is not None else None
    end_error = abs(float(end) - gold[1]) if end is not None else None
    return {
        "gold_start_seconds": gold[0],
        "gold_end_seconds": gold[1],
        "start_absolute_error_seconds": round(start_error, 6) if start_error is not None else None,
        "end_absolute_error_seconds": round(end_error, 6) if end_error is not None else None,
        "both_boundaries_within_2s": (
            start_error is not None and end_error is not None
            and start_error <= TOLERANCE_SECONDS and end_error <= TOLERANCE_SECONDS
        ),
        "rubric_decision_correct": method.get("rubric_decision") == "pass",
    }


def aggregate(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rows = [record["methods"][method]["metrics"] for record in records]
    starts = [row["start_absolute_error_seconds"] for row in rows if row["start_absolute_error_seconds"] is not None]
    ends = [row["end_absolute_error_seconds"] for row in rows if row["end_absolute_error_seconds"] is not None]
    return {
        "method": method,
        "video_count": len(rows),
        "start_boundary_coverage": len(starts),
        "end_boundary_coverage": len(ends),
        "mean_start_absolute_error_seconds": round(sum(starts) / len(starts), 6) if starts else None,
        "mean_end_absolute_error_seconds": round(sum(ends) / len(ends), 6) if ends else None,
        "both_boundaries_within_2s_count": sum(row["both_boundaries_within_2s"] for row in rows),
        "rubric_pass_accuracy_count": sum(row["rubric_decision_correct"] for row in rows),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Online Qwen P0 A/B",
        "",
        f"Status: `{result['status']}`; model: `{result['model']}`; rubric: cleanup and return.",
        "",
        "Gold intervals were used only for local scoring and were never sent to Qwen.",
        "",
        "| Method | Start MAE (s) | End MAE (s) | Both <=2s | Rubric pass accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result.get("summary", []):
        lines.append(
            f"| {row['method']} | {row['mean_start_absolute_error_seconds']} | "
            f"{row['mean_end_absolute_error_seconds']} | {row['both_boundaries_within_2s_count']}/{row['video_count']} | "
            f"{row['rubric_pass_accuracy_count']}/{row['video_count']} |"
        )
    lines.extend([
        "",
        f"Qwen calls: {result['qwen_usage']['call_count']}; image exposures: {result['qwen_usage']['image_exposures']}.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    plan_payload = read_json(args.plan)
    gold = cleanup_gold(read_json(args.gold))
    client = None if args.prepare_only else QwenClient(args.endpoint, args.token, args.model, args.timeout)
    records = []
    for index, record in enumerate(source_records(plan_payload), start=1):
        video_id = str(record["source_video_id"])
        if video_id not in gold:
            raise ValueError(f"gold_cleanup_missing:{video_id}")
        video = resolve_video(video_id, args.video_root)
        extent_raw = record["experiment_interval_seconds"]
        extent = (float(extent_raw[0]), float(extent_raw[1]))
        rubric_plan = rubric_zero_plan(record)
        guided_times = planned_times(rubric_plan)
        if not guided_times:
            raise ValueError(f"guided_times_missing:{video_id}")
        uniform = uniform_times(extent[0], extent[1], len(guided_times))
        video_dir = args.output / "videos" / safe_slug(video_id)
        extractor = FrameExtractor(video, video_dir / "frames", args.max_edge)
        try:
            uniform_frames = extractor.extract_times("uniform_direct", "uniform", uniform)
            guided_frames = extractor.extract_times("rubric_guided_direct", "guided", guided_times)
            calls_before = client.call_count if client else 0
            images_before = client.image_exposures if client else 0
            uniform_result = run_direct(client, uniform_frames, "全实验区间均匀")
            guided_result = run_direct(client, guided_frames, "Rubric 引导候选区间")
            yes_no_result = run_yes_no(client, extractor, extent)
            boundary_result = run_boundary_binding(client, extractor, guided_result, extent)
            methods = {
                "uniform_direct": uniform_result,
                "rubric_guided_direct": guided_result,
                "yes_no_temporal": yes_no_result,
                "boundary_evidence_binding": boundary_result,
            }
            for method in methods.values():
                method["metrics"] = error_metrics(method, gold[video_id])
            item = {
                "source_video_id": video_id,
                "source_video": str(video),
                "experiment_extent_seconds": list(extent),
                "gold_cleanup_test_only": list(gold[video_id]),
                "gold_sent_to_qwen": False,
                "equal_direct_image_budget": len(guided_times),
                "methods": methods,
                "qwen_calls": (client.call_count - calls_before) if client else 0,
                "qwen_image_exposures": (client.image_exposures - images_before) if client else 0,
            }
            write_json(video_dir / "result.json", item)
            records.append(item)
            print(json.dumps({"video": index, "total": len(source_records(plan_payload)), "source_video_id": video_id, "status": "completed" if client else "prepared"}, ensure_ascii=False), flush=True)
        finally:
            extractor.close()
    method_names = [
        "uniform_direct", "rubric_guided_direct", "yes_no_temporal", "boundary_evidence_binding"
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared" if args.prepare_only else "completed",
        "model": args.model,
        "endpoint_host": urllib.parse.urlparse(args.endpoint).netloc,
        "rubric_id": 0,
        "rubric_key": "cleanup_and_return",
        "gold_is_test_only": True,
        "gold_sent_to_qwen": False,
        "records": records,
        "summary": [aggregate(records, method) for method in method_names],
        "qwen_usage": {
            "call_count": client.call_count if client else 0,
            "image_exposures": client.image_exposures if client else 0,
        },
    }
    write_json(args.output / "comparison.json", result)
    write_markdown(args.output / "comparison.md", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--video-root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--token", default="EMPTY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-edge", type=int, default=640)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    args.gold = args.gold.resolve()
    args.video_root = [path.resolve() for path in args.video_root]
    args.output = args.output.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output), "qwen_calls": result["qwen_usage"]["call_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
