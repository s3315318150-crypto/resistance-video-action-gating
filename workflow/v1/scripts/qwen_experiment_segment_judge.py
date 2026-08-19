#!/usr/bin/env python3
"""Use Qwen to select an experiment segment from a full video timeline.

The visual rules are supplied by the user:

* Start when the orange-red instrument moves from the upper-left to the middle
  of the table, or at video start when it is already in the middle.
* End only after the wiring is dismantled and the instrument is returned to
  the upper-left.

The script sends an ordered, uniformly sampled low-resolution timeline to
Qwen.  Qwen selects frame IDs rather than timestamps; timestamps are mapped
locally after the response, so the model cannot be led by a proposed interval.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import cv2
try:
    from openai import OpenAI
except ModuleNotFoundError:
    from types import SimpleNamespace

    class OpenAI:
        """Minimal OpenAI-compatible client for the existing Qwen endpoint."""

        def __init__(self, *, base_url: str, api_key: str, timeout: float = 180, max_retries: int = 0) -> None:
            self._base_url = base_url.rstrip("/")
            self._api_key = api_key
            self._timeout = timeout
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, *, model: str, messages: list[dict[str, Any]], max_tokens: int, temperature: float, extra_body: dict[str, Any] | None = None) -> Any:
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if extra_body:
                payload.update(extra_body)
            request = urllib.request.Request(
                self._base_url + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            choices = raw.get("choices") if isinstance(raw, dict) else None
            if not isinstance(choices, list) or not choices:
                raise ValueError("Qwen response has no choices")
            choice = choices[0] if isinstance(choices[0], dict) else {}
            message = choice.get("message") if isinstance(choice, dict) else {}
            content = message.get("content") if isinstance(message, dict) else ""
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=choice.get("finish_reason", "unknown"),
                        message=SimpleNamespace(content=content),
                    )
                ]
            )


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "outputs" / "marker_filter"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "experiment_boundary"
API_BASE_URL = os.getenv("QWEN_API_BASE_URL", "").strip()
API_TOKEN = os.getenv("QWEN_API_TOKEN", "").strip()
MODEL = os.getenv("QWEN_MODEL", "qwen").strip() or "qwen"


def require_qwen_configuration() -> None:
    """Reject model calls unless credentials are provided explicitly."""
    missing = [
        name
        for name, value in (
            ("QWEN_API_BASE_URL", API_BASE_URL),
            ("QWEN_API_TOKEN", API_TOKEN),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )


def safe_error_message(exc: Exception) -> str:
    """Keep diagnostics useful without persisting a configured endpoint."""
    message = str(exc)
    return message.replace(API_BASE_URL, "<QWEN_API_BASE_URL>") if API_BASE_URL else message


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def image_data_url(path: Path) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def resize_for_model(frame: Any, longest_edge: int) -> Any:
    height, width = frame.shape[:2]
    scale = min(1.0, longest_edge / max(height, width))
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def add_relative_timestamp_banner(frame: Any, image_id: str, timestamp_seconds: float) -> Any:
    """Add visual frame/time identifiers without covering source pixels."""
    height, width = frame.shape[:2]
    banner_height = max(40, int(round(height * 0.12)))
    banner = cv2.copyMakeBorder(frame, 0, banner_height, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    label = f"FRAME ID={image_id}  |  VIDEO T={timestamp_seconds:.1f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.65, min(1.05, width / 760.0))
    thickness = max(2, int(round(scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    x = 14
    y = height + max(text_height + 6, (banner_height + text_height - baseline) // 2)
    cv2.putText(banner, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return banner


def parse_json(content: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Qwen response was not a JSON object")
    return value


def extract_timeline(
    manifest: dict[str, Any],
    output_dir: Path,
    interval_seconds: float,
    longest_edge: int,
    timestamp_watermark: bool = False,
) -> list[dict[str, Any]]:
    source = Path(str(manifest["source_video"]))
    metadata = manifest["video_metadata"]
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    duration = frame_count / fps
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {source}")
    records: list[dict[str, Any]] = []
    try:
        timestamps = [index * interval_seconds for index in range(int(math.floor(duration / interval_seconds)) + 1)]
        if not timestamps or timestamps[-1] < duration - 0.01:
            timestamps.append(duration)
        for index, timestamp in enumerate(timestamps, start=1):
            frame_number = min(frame_count - 1, max(0, int(round(timestamp * fps))))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Unable to read frame {frame_number} from {source}")
            image_id = f"frame_{index:03d}"
            path = output_dir / f"{image_id}_{frame_number:08d}_{frame_number / fps:010.3f}s.jpg"
            relative_seconds = round(frame_number / fps, 3)
            image = resize_for_model(frame, longest_edge)
            if timestamp_watermark:
                image = add_relative_timestamp_banner(image, image_id, relative_seconds)
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 84]):
                raise RuntimeError(f"Unable to write frame: {path}")
            records.append({
                "image_id": image_id,
                "frame_number": frame_number,
                "timestamp_seconds": relative_seconds,
                "path": str(path.resolve()),
                "relative_timestamp_banner": (
                    f"FRAME ID={image_id} | VIDEO T={relative_seconds:.1f}s"
                    if timestamp_watermark else None
                ),
            })
    finally:
        capture.release()
    return records


def build_baseline_prompt(image_ids: list[str]) -> str:
    return f"""你在判断一个物理实验视频中一次有效实验的时间片段。图片来自同一个视频，按时间顺序提供，ID 依次为：{", ".join(image_ids)}。

判定规则：
1. 实验开始：目标仪器从桌面左上角被拿出或移开；拿出动作完成后第一个能确认仪器已离开桌面左上角的抽帧，就是实验开始。
2. 实验结束：线路已被拆卸，并且该仪器已重新放回画面左上区域。两项必须同时可见。
3. 若视频多次开始或重做，选择最后一次满足开始规则的实验；其结束必须满足规则 2。

只依据实际可见像素。镜头移动、遮挡、看不清仪器位置或仅出现其他橘红色零件时，不能当作开始或结束。不能确认时将相应 frame_id 设为 null，不得猜测。

只输出一个合法 JSON 对象：
{{
  "start_frame_id": "frame_001" | null,
  "start_reason": "removed_from_upper_left" | "uncertain",
  "end_frame_id": "frame_001" | null,
  "end_reason": "wiring_dismantled_and_instrument_returned_upper_left" | "not_observed" | "uncertain",
  "confidence": 0.0,
  "visible_evidence": "仅描述可见动作和位置",
  "needs_dense_scan": true
}}"""


def build_voltmeter_resistance_prompt(image_ids: list[str]) -> str:
    return f"""你是一名严谨的中学物理实验视频判读员。请从同一段视频的有序抽帧中，找出最后一次有效的“伏安法测电阻”实验区间。图片严格按时间顺序提供，ID 依次为：{", ".join(image_ids)}。图片 ID 只用于引用画面，不代表你看到了精确秒数。

时间标签说明：部分图片底部左侧可能有黑色信息条和白色文字，例如 `FRAME ID=frame_016 | VIDEO T=75.0s`。`FRAME ID` 是该图片唯一可用的 frame_id；`VIDEO T` 是该图片在视频中的相对时间，不是右下角摄像机自带的日期时间。它们只帮助你核对前后顺序和时间位置；你仍必须依据可见动作选择 frame_id，不能只因时间标签推测开始、结束或阶段。

frame_id 与时间文字必须严格区分：`VIDEO T=75.0s` 绝不是 `frame_075`。所有 JSON 中出现的 frame_id 必须从本提示开头给出的 ID 列表或图片底部左侧的 `FRAME ID=` 文字中原样复制，例如 `frame_016`；不得依据时间数字临时编造 frame_id，也不得将秒数改写为 frame_id。

目标对象和实验背景：
* 目标仪器是画面中的橘红色电池盒或电池组装置。不要把橘红色导线、开关、表笔或其他零件误认成目标仪器。
* 实验中使用的仪器整体都可能是橙红色的，包括电池盒、开关、电表等；颜色只能作为辅助线索，必须结合仪器的形状、所在位置、与导线的连接关系及前后动作来区分不同仪器。
* 起始收纳区是桌面左上角的区域，不是整张画面的左上角。只有目标仪器明确放在桌面左上角时，才可视为在起始收纳区。
* 实验操作区是桌面中央由橙色虚线画出的矩形方框，方框内写有“测量区”三个字。
* 本实验通过测量待测电阻两端电压 U 和通过它的电流 I，并以 R = U / I 求电阻。通常电流表串联在电路中，电压表并联在待测电阻两端；画面中也可能有开关、导线和记录纸。
* 上述接线关系仅用于理解操作阶段。表盘、接线端子或数值看不清时，不能臆测具体读数、元件名称或连接正确性。

这段视频中可能出现的典型顺序如下。它们是识别线索，不是每一步都必须清晰出现：
1. 连线：摆放电池盒、开关、待测电阻和电表，连接导线。
2. 第一次测量：闭合电路、观察电流表/电压表并进行测量。
3. 第一次记录：在纸上记录读数或计算。
4. 重新连线：拆开、修改或重新插接实验线路以继续实验。
5. 第二次测量：重新连线后再次观察仪表。
6. 第二次记录：再次在纸上记录。
7. 整理材料：拆下导线、断开连接，并把目标仪器放回画面左上区域。学生也可能明确换到另一个座位继续操作；这种换座位表示原位置上的本次实验结束。

任务和边界规则：
1. 开始帧：先比较按时间相邻的抽帧，找到最后一次明确看到目标仪器从桌面左上角被拿出或移开的动作。目标仪器一旦从桌面左上角被拿出，就视为本次实验开始；开始帧选择拿出动作完成后第一个能清楚确认目标仪器已离开桌面左上角的抽帧。
2. 开始后的验证：目标仪器被拿出后，应至少可见连线、观察仪表、记录数据或重新连线中的一种实验活动。若拿出后很快又放回左上角且没有后续实验活动，这是未完成尝试，不作为最终实验开始。
3. 整理材料终止限制：若清楚看到进入 `material_cleanup`，即线路已拆卸或断开且目标仪器已放回桌面左上角，则将该帧设为 cleanup_frame_id 和 end_frame_id。从该帧开始，后面的所有图片均不再参与本次实验的开始、结束、重做或阶段判断；不得因为后续画面再次出现器材移动而另选更晚开始或结束。
4. 重做处理：只在 cleanup_frame_id 之前，若较早一次从桌面左上角拿出后又归位、重新连线或重新开始，才把最后一次符合“从桌面左上角拿出”规则且有后续实验活动的动作作为候选开始；不要把未完成的早期尝试当最终实验。
5. 结束帧：在 cleanup_frame_id 未观察到时，只选择最后一次实验之后，第一个清楚满足换座位条件的抽帧：实验者明确离开原座位、换到另一个座位，并在新位置继续操作。条件表示原座位上的本次实验结束。
6. 换座位必须能从相邻画面确认原座位和新座位的变化，并看到实验者在新位置继续操作。仅起身、侧身取物、短暂离开座位、伸手到桌子另一侧、他人经过或镜头移动，都不是换座位。
7. 重新连线、暂停书写、移动镜头、拿起其他材料，都不能当作整理材料或结束。若视频结束时仍在测量、记录或线路仍连着，且未确认整理材料或换座位，end_frame_id 必须为 null。
8. 仅依据可见像素。请比较候选帧前后的画面来确认位置和线路状态；看不清、被手遮挡或无法确认时，选择 null 或标记不确定，不得猜测。

输出要求：
* start_frame_id 非空时，start_reason 必须为 removed_from_upper_left。
* end_frame_id 非空时，end_reason 必须为 wiring_dismantled_and_instrument_returned_upper_left 或 student_changed_seat。若未观察到结束，end_frame_id 必须为 null，end_reason 必须为 not_observed 或 uncertain。
* cleanup_frame_id 只在明确整理材料时填写；此时它必须与 end_frame_id 相同，end_reason 必须为 wiring_dismantled_and_instrument_returned_upper_left。未观察到整理材料时 cleanup_frame_id 必须为 null。
* stage_observations 返回 3 至 5 条按时间排序的关键可见事实，覆盖开头、最后一次开始、关键实验过程和视频末尾。每条 observation 不超过 30 个汉字。stage 只能是 preparation、circuit_wiring、measurement_1、recording_1、circuit_rewiring、measurement_2、recording_2、material_cleanup、other 之一。不要因为预期流程而虚构未看到的阶段。

只输出一个合法 JSON 对象：
{{
  "start_frame_id": "frame_001" | null,
  "start_reason": "removed_from_upper_left" | "uncertain",
  "start_evidence": "不超过50字，引用相邻 frame_id 说明目标仪器从桌面左上角被拿出的变化",
  "end_frame_id": "frame_001" | null,
  "end_reason": "wiring_dismantled_and_instrument_returned_upper_left" | "student_changed_seat" | "not_observed" | "uncertain",
  "end_evidence": "不超过50字，说明拆线归位、换座位或缺少的结束证据",
  "cleanup_frame_id": "frame_001" | null,
  "cleanup_evidence": "不超过50字；未观察到整理材料则为空字符串",
  "stage_observations": [
    {{"frame_id": "frame_001", "stage": "preparation", "observation": "只描述实际可见内容"}}
  ],
  "confidence": 0.0,
  "visible_evidence": "不超过80字，概括最后一次实验的开始、过程和结束证据",
  "needs_dense_scan": true,
  "uncertainty": "不超过50字；无不确定性则写空字符串"
}}"""


def build_switch_closure_prompt(image_ids: list[str]) -> str:
    return f"""你是一名严谨的中学物理实验视频判读员。请从同一段伏安法测电阻实验视频的有序抽帧中，识别“实验过程中开关第一次明确闭合”的时间点。图片严格按时间顺序提供，ID 依次为：{", ".join(image_ids)}。图片 ID 只用于引用画面，不代表你看到了精确秒数。

实验背景：
* 视频包含摆放器材、连接导线、闭合或断开开关、观察电流表和电压表、记录数据、重新连线后再次测量等过程。
* 实验中使用的仪器整体都可能是橙红色的，包括电池盒、开关、电表等；颜色只能作为辅助线索，必须结合开关的形状、所在位置、与导线的连接关系及前后拨动动作来识别。
* 目标是实验电路中的机械开关或按键开关，不是滑动变阻器、电池盒、导线接头、表笔或手指。
* 开关闭合表示开关从“断开/分离/断路”状态变为“闭合/接通”状态。只有能从相邻画面确认状态发生变化，才可判定闭合。

识别规则：
1. 按时间顺序比较相邻抽帧，先找到开关仍明显断开的画面，再找同一次操作后开关已经明确闭合的第一张抽帧。
2. switch_frame_id 选择第一次明确确认开关闭合的抽帧，不是手刚伸向开关、正在拨动、手遮挡开关或状态看不清的画面。
3. 连接导线、整理材料、拿起电池、移动其他器材、观察仪表、写字，都不能单独当作闭合。手离开后开关的最终可见状态必须支持闭合判断。
4. 如果开关在视频第一帧就已经明确闭合，且没有更早的断开画面，switch_frame_id 可以为 frame_001，reason 使用 already_closed_at_video_start。
5. 如果第一次闭合后又断开、换电池或重复测量，仍只返回全视频中最早一次明确闭合的时间。
6. 只依据实际可见像素。开关太小、被手或导线遮挡、镜头移动或抽帧间隔导致无法判断时，switch_frame_id 必须为 null，不得根据电表读数或实验流程猜测。

只输出一个合法 JSON 对象：
{{
  "switch_frame_id": "frame_001" | null,
  "switch_reason": "first_switch_closed" | "already_closed_at_video_start" | "not_observed" | "uncertain",
  "switch_evidence": "不超过80字，引用闭合前后相邻 frame_id，描述开关可见状态变化",
  "observations": [
    {{"frame_id": "frame_001", "observation": "不超过30字，只描述开关的可见状态"}}
  ],
  "confidence": 0.0,
  "uncertainty": "不超过50字；无不确定性则写空字符串"
}}"""


def build_prompt(image_ids: list[str], profile: str) -> str:
    if profile == "baseline":
        return build_baseline_prompt(image_ids)
    if profile == "voltmeter_resistance":
        return build_voltmeter_resistance_prompt(image_ids)
    if profile == "switch_closure":
        return build_switch_closure_prompt(image_ids)
    raise ValueError(f"Unknown prompt profile: {profile}")


def validate_response(value: dict[str, Any], image_ids: set[str], profile: str) -> list[str]:
    errors: list[str] = []
    if profile == "switch_closure":
        switch_id = value.get("switch_frame_id")
        if switch_id is not None and switch_id not in image_ids:
            errors.append("switch_frame_id_invalid")
        if value.get("switch_reason") not in {
            "first_switch_closed", "already_closed_at_video_start",
            "not_observed", "uncertain",
        }:
            errors.append("switch_reason_invalid")
        if value.get("switch_reason") == "already_closed_at_video_start" and switch_id != "frame_001":
            errors.append("already_closed_must_be_first_frame")
        if switch_id is not None and value.get("switch_reason") in {"not_observed", "uncertain"}:
            errors.append("switch_reason_does_not_support_selected_frame")
        if switch_id is None and value.get("switch_reason") == "first_switch_closed":
            errors.append("switch_reason_without_selected_frame")
        if not isinstance(value.get("switch_evidence"), str) or not value["switch_evidence"].strip():
            errors.append("switch_evidence_invalid")
        if not isinstance(value.get("confidence"), (int, float)) or not 0.0 <= float(value["confidence"]) <= 1.0:
            errors.append("confidence_invalid")
        if not isinstance(value.get("uncertainty"), str):
            errors.append("uncertainty_invalid")
        observations = value.get("observations")
        if not isinstance(observations, list) or not 2 <= len(observations) <= 5:
            errors.append("observations_invalid")
        elif any(
            not isinstance(item, dict)
            or item.get("frame_id") not in image_ids
            or not isinstance(item.get("observation"), str)
            or not item["observation"].strip()
            or len(item["observation"]) > 30
            for item in observations
        ):
            errors.append("observation_entry_invalid")
        return sorted(set(errors))
    for key in ("start_frame_id", "end_frame_id"):
        value_id = value.get(key)
        if value_id is not None and value_id not in image_ids:
            errors.append(f"{key}_invalid")
    if profile in {"baseline", "voltmeter_resistance"}:
        start_reasons = {"removed_from_upper_left", "uncertain"}
    else:
        start_reasons = {"uncertain"}
    if value.get("start_reason") not in start_reasons:
        errors.append("start_reason_invalid")
    end_reasons = {"wiring_dismantled_and_instrument_returned_upper_left", "not_observed", "uncertain"}
    if profile == "voltmeter_resistance":
        end_reasons.add("student_changed_seat")
    if value.get("end_reason") not in end_reasons:
        errors.append("end_reason_invalid")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        errors.append("confidence_invalid")
    if not isinstance(value.get("visible_evidence"), str) or not value["visible_evidence"].strip():
        errors.append("visible_evidence_invalid")
    if not isinstance(value.get("needs_dense_scan"), bool):
        errors.append("needs_dense_scan_invalid")
    if profile == "voltmeter_resistance":
        start_id = value.get("start_frame_id")
        start_reason = value.get("start_reason")
        if start_id is not None and start_reason == "uncertain":
            errors.append("start_reason_does_not_support_selected_frame")
        end_id = value.get("end_frame_id")
        end_reason = value.get("end_reason")
        confirmed_end_reasons = {
            "wiring_dismantled_and_instrument_returned_upper_left",
            "student_changed_seat",
        }
        if end_id is not None and end_reason not in confirmed_end_reasons:
            errors.append("end_reason_does_not_support_selected_frame")
        if end_id is None and end_reason in confirmed_end_reasons:
            errors.append("end_reason_without_selected_frame")
        cleanup_id = value.get("cleanup_frame_id")
        if cleanup_id is not None and cleanup_id not in image_ids:
            errors.append("cleanup_frame_id_invalid")
        if cleanup_id is not None:
            if cleanup_id != end_id:
                errors.append("cleanup_frame_must_equal_end_frame")
            if end_reason != "wiring_dismantled_and_instrument_returned_upper_left":
                errors.append("cleanup_requires_wiring_return_end_reason")
            if not isinstance(value.get("cleanup_evidence"), str) or not value["cleanup_evidence"].strip():
                errors.append("cleanup_evidence_invalid")
        elif not isinstance(value.get("cleanup_evidence"), str):
            errors.append("cleanup_evidence_invalid")
        if not isinstance(value.get("start_evidence"), str) or not value["start_evidence"].strip():
            errors.append("start_evidence_invalid")
        if not isinstance(value.get("end_evidence"), str) or not value["end_evidence"].strip():
            errors.append("end_evidence_invalid")
        if not isinstance(value.get("uncertainty"), str):
            errors.append("uncertainty_invalid")
        observations = value.get("stage_observations")
        allowed_stages = {
            "preparation", "circuit_wiring", "measurement_1", "recording_1", "circuit_rewiring",
            "measurement_2", "recording_2", "material_cleanup", "other",
        }
        if not isinstance(observations, list) or not 3 <= len(observations) <= 5:
            errors.append("stage_observations_invalid")
        elif any(
            not isinstance(item, dict)
            or item.get("frame_id") not in image_ids
            or item.get("stage") not in allowed_stages
            or not isinstance(item.get("observation"), str)
            or not item["observation"].strip()
            or len(item["observation"]) > 30
            for item in observations
        ):
            errors.append("stage_observation_entry_invalid")
    if profile == "switch_closure":
        switch_id = value.get("switch_frame_id")
        if switch_id is not None and switch_id not in image_ids:
            errors.append("switch_frame_id_invalid")
        if value.get("switch_reason") not in {
            "first_switch_closed", "already_closed_at_video_start",
            "not_observed", "uncertain",
        }:
            errors.append("switch_reason_invalid")
        if value.get("switch_reason") == "already_closed_at_video_start" and switch_id != "frame_001":
            errors.append("already_closed_must_be_first_frame")
        if switch_id is not None and value.get("switch_reason") in {"not_observed", "uncertain"}:
            errors.append("switch_reason_does_not_support_selected_frame")
        if switch_id is None and value.get("switch_reason") == "first_switch_closed":
            errors.append("switch_reason_without_selected_frame")
        if not isinstance(value.get("switch_evidence"), str) or not value["switch_evidence"].strip():
            errors.append("switch_evidence_invalid")
        if not isinstance(value.get("uncertainty"), str):
            errors.append("uncertainty_invalid")
        observations = value.get("observations")
        if not isinstance(observations, list) or not 2 <= len(observations) <= 5:
            errors.append("observations_invalid")
        elif any(
            not isinstance(item, dict)
            or item.get("frame_id") not in image_ids
            or not isinstance(item.get("observation"), str)
            or not item["observation"].strip()
            or len(item["observation"]) > 30
            for item in observations
        ):
            errors.append("observation_entry_invalid")
    return sorted(set(errors))


def call_qwen(client: OpenAI, frames: list[dict[str, Any]], max_tokens: int, prompt_profile: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": build_prompt([str(frame["image_id"]) for frame in frames], prompt_profile)}]
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(Path(frame["path"]))}})
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = completion.choices[0]
    raw = choice.message.content or ""
    result: dict[str, Any] = {"finish_reason": choice.finish_reason or "unknown", "raw_model_content": raw, "parsed": False}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = parse_json(raw)
            result["parsed_result"] = parsed
            result["validation_errors"] = validate_response(parsed, {str(frame["image_id"]) for frame in frames}, prompt_profile)
            result["parsed"] = True
        except (json.JSONDecodeError, ValueError) as exc:
            result["parse_error"] = str(exc)
    else:
        result["parse_error"] = "Qwen did not return text content"
    return result


def attach_timestamps(result: dict[str, Any], frames: list[dict[str, Any]], prompt_profile: str) -> dict[str, Any]:
    parsed = result.get("parsed_result")
    if not isinstance(parsed, dict):
        return {"start_seconds": None, "end_seconds": None, "segment_valid": False, "segment_errors": ["qwen_response_not_parsed"]}
    by_id = {str(frame["image_id"]): frame for frame in frames}
    if prompt_profile == "switch_closure":
        switch_id = parsed.get("switch_frame_id")
        switch = by_id.get(switch_id) if isinstance(switch_id, str) else None
        errors = list(result.get("validation_errors", []))
        if switch is None:
            errors.append("switch_not_observed")
        return {
            "switch_seconds": switch.get("timestamp_seconds") if switch else None,
            "switch_valid": not errors,
            "switch_errors": sorted(set(errors)),
        }
    start_id, end_id = parsed.get("start_frame_id"), parsed.get("end_frame_id")
    cleanup_id = parsed.get("cleanup_frame_id") if prompt_profile == "voltmeter_resistance" else None
    start = by_id.get(start_id) if isinstance(start_id, str) else None
    end = by_id.get(end_id) if isinstance(end_id, str) else None
    cleanup = by_id.get(cleanup_id) if isinstance(cleanup_id, str) else None
    errors = list(result.get("validation_errors", []))
    if start is None:
        errors.append("start_not_observed")
    if end is None:
        errors.append("end_not_observed")
    if start is not None and end is not None and int(end["frame_number"]) <= int(start["frame_number"]):
        errors.append("end_not_after_start")
    return {
        "start_seconds": start.get("timestamp_seconds") if start else None,
        "end_seconds": end.get("timestamp_seconds") if end else None,
        "cleanup_seconds": cleanup.get("timestamp_seconds") if cleanup else None,
        "segment_valid": not errors,
        "segment_errors": sorted(set(errors)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--max-model-edge", type=int, default=480)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument(
        "--timestamp-watermark",
        action="store_true",
        help="Render visual frame/time identifiers in a bottom-left information strip on every model image.",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=("baseline", "voltmeter_resistance", "switch_closure"),
        default="baseline",
        help="Qwen prompt template to use.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Only process a source video ID. May be passed more than once.",
    )
    args = parser.parse_args(argv)
    if args.interval_seconds <= 0 or args.max_model_edge <= 0 or args.max_tokens <= 0:
        parser.error("interval-seconds, max-model-edge, and max-tokens must be positive")
    manifests = sorted(args.input_dir.glob("*.marker_filter.json"))
    if args.video_id:
        requested_ids = set(args.video_id)
        manifests = [
            path
            for path in manifests
            if str(json.loads(path.read_text(encoding="utf-8")).get("source_video_id", path.stem)) in requested_ids
        ]
    if not manifests:
        parser.error(f"No manifests found under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require_qwen_configuration()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_TOKEN, timeout=180, max_retries=0)
    records: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        video_id = str(manifest.get("source_video_id", manifest_path.stem))
        frames = extract_timeline(
            manifest,
            args.output_dir / "inputs" / slug(video_id),
            args.interval_seconds,
            args.max_model_edge,
            timestamp_watermark=args.timestamp_watermark,
        )
        record: dict[str, Any] = {
            "source_video_id": video_id,
            "source_manifest": str(manifest_path.resolve()),
            "prompt_profile": args.prompt_profile,
            "input_frame_count": len(frames),
            "input_frames": frames,
        }
        try:
            qwen = call_qwen(client, frames, args.max_tokens, args.prompt_profile)
            record["qwen"] = qwen
            record["segment"] = attach_timestamps(qwen, frames, args.prompt_profile)
            record["status"] = "completed"
        except Exception as exc:
            record["status"] = "qwen_request_failed"
            record["error_type"] = type(exc).__name__
            record["error"] = safe_error_message(exc)
        records.append(record)
        path = args.output_dir / f"{slug(video_id)}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"video": video_id, "status": record["status"], "segment": record.get("segment")}, ensure_ascii=False), flush=True)
    summary = {
        "schema_version": "qwen_experiment_segment_judge.v1",
        "model": MODEL,
        "prompt_profile": args.prompt_profile,
        "interval_seconds": args.interval_seconds,
        "max_model_edge": args.max_model_edge,
        "timestamp_watermark": args.timestamp_watermark,
        "timestamp_watermark_format": "FRAME ID=<frame-id> | VIDEO T=<video-relative-seconds>s" if args.timestamp_watermark else None,
        "records": records,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary={summary_path.resolve()}")
    return 0 if all(record.get("status") == "completed" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
