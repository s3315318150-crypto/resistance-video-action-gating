#!/usr/bin/env python3
"""Run lenient meter-polarity assessment on measurement/recording frames only."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parent.parent
RUBRIC_ID = "resistance.meter_polarity_lenient_v15_apparatus_priors"
# Historical experiment manifests are intentionally not bundled.  The standalone
# helper accepts all data roots explicitly; the live Agent path supplies only the
# current run's stage summary and dynamic frame candidates.
DEFAULT_STAGE_MANIFEST = None
DEFAULT_REFERENCE_MANIFEST = None
DEFAULT_DETECTOR_ROOT = None
DEFAULT_OUTPUT_ROOT = None
OBSERVATION_PHASES = ("measurement", "recording")
OBSERVATION_STAGES = {"measurement_1", "measurement_2", "recording_1", "recording_2"}
METER_STATES = {"likely_correct", "likely_incorrect", "unclear"}
VIOLATION_TYPES = {"reversed", "not_connected", "wrong_terminal", "none", "unclear"}
TERMINAL_DESTINATIONS = {"source_positive_side", "source_negative_side", "unconnected", "unclear"}
POINTER_STATES = {"normal_positive_deflection", "reverse_below_zero", "zero_or_unclear"}
READING_SIGNS = {"positive", "negative", "zero", "unclear"}
READING_SIGN_MIN_CONFIDENCE = 0.85
TERMINAL_REVERSAL_MIN_CONFIDENCE = 0.85
TERMINAL_REVERSAL_MIN_EVIDENCE_FRAMES = 2
EVIDENCE_QUALITY = {"high", "medium", "low"}
SOURCE_POSITIVE_END_IDENTITIES = {"gold_raised", "plus_mark", "other_visible", "unclear"}
SOURCE_NEGATIVE_END_IDENTITIES = {"green_flat", "minus_mark", "other_visible", "unclear"}
SOURCE_SERIES_BRIDGE_STATES = {"gold_to_green", "not_visible", "not_applicable", "conflict"}
WIRE_COLOR_TERM = (
    r"(?:深|浅|亮|暗)?(?:金黄|银白|红|黑|蓝|黄|白|绿|灰|橙|紫|棕|褐|粉|青|金|银)(?:色)?|透明|彩色"
)
WIRE_OBJECT_TERM = (
    r"(?:连接线|测试线|导线|电线|线缆|线材|香蕉插头|香蕉头|插头|接线头|接头|接线夹|夹子|线)"
)
WIRE_COLOR_REFERENCE_PATTERNS = (
    re.compile(
        rf"(?:(?:{WIRE_COLOR_TERM})(?:[/／、和或]?)){{1,3}}{WIRE_OBJECT_TERM}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{WIRE_OBJECT_TERM}(?:的)?(?:外皮|绝缘层|颜色)?(?:为|是|呈|采用|涂成)?(?:{WIRE_COLOR_TERM})",
        re.IGNORECASE,
    ),
    re.compile(rf"{WIRE_OBJECT_TERM}(?:的)?(?:颜色|色彩|色泽|线色)"),
    re.compile(rf"(?:按|依据|根据|利用|通过)[^，。；\n]{{0,8}}颜色[^，。；\n]{{0,12}}{WIRE_OBJECT_TERM}"),
    re.compile(
        r"(?:red|black|blue|yellow|white|green|gray|grey|orange|purple|brown|pink|gold|silver)\s*"
        r"(?:wire|lead|cable|plug|connector)|(?:wire|lead|cable|plug|connector)\s*(?:color|colour)",
        re.IGNORECASE,
    ),
)
RESPONSE_FIELDS = {
    "rubric_id",
    "confidence",
    "evidence_quality",
    "source_polarity_state",
    "source_positive_end_identity",
    "source_negative_end_identity",
    "source_series_bridge_state",
    "source_positive_evidence",
    "ammeter",
    "voltmeter",
    "observation_summary",
}
METER_FIELDS = {
    "positive_terminal_destination",
    "negative_terminal_destination",
    "pointer_state",
    "confidence",
    "evidence",
    "evidence_seconds",
}
ROI_ORDER = ("ammeter", "voltmeter", "battery_holder", "fixed_resistor")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def finite_probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def parse_model_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Model response is not a JSON object")
    return value


def prompt_text(stage_frames: list[dict[str, Any]], available_rois: list[str]) -> str:
    frame_summary = [
        {
            "stage": item["stage"],
            "phase": item["observation_phase"],
            "timestamp_seconds": item["timestamp_seconds"],
        }
        for item in stage_frames
    ]
    return f"""你是中学物理伏安法测电阻视频的端点视觉观察器。你只记录画面事实，不评分。

评分项：电流表、电压表正负接线柱连接正确。电流表应当“+”进“-”出；电压表“+”应接在被测电阻电路拓扑上靠近电源正极的一端。这里的“靠近”是电路连接关系，不是画面中的几何距离。

输入证据全部来自测量阶段和记录阶段，不包含连线阶段或重新连线阶段：{frame_summary}
每个时间点先给一张全景图，再按实际可用情况给局部图。可用局部类型：{available_rois}。名为“电表候选”的宽局部只是计算机视觉给出的空间候选，可能误检；候选没有预设 A/V 身份，你必须从表盘字母和画面上下文自行识别电流表或电压表。

采用宽松观察规则：
1. 不输出 pass、fail、correct、incorrect 或 reversed；评分由本地程序根据端点观察生成。你只填写指定的视觉观察字段。
2. 本任务不分析连线动作，不要求精确捕捉插拔过程。只观察测量阶段和记录阶段已经存在的电路状态。
3. 一张较清楚的帧可以判断；两张或多张邻近帧也可以互补。一帧可用于识别端子印字或器材金黄/绿色端，另一帧可用于观察插头落点、导线几何走向或遮挡后的状态。
4. 邻近帧不必严格连续，器材位置也不必完全一致；同一视频的测量帧和记录帧可以互补。
5. 不要求完整追踪整根导线。端子印字、端子几何位置、插头落点、导线几何连续性、电源端方向和遮挡前后变化可以共同作为依据。
6. 导线交叉不自动代表连接。完全忽略导线外皮和插头的颜色；不得用红色、黑色或其他导线颜色区分、追踪或关联导线，也不得在 evidence 中用颜色描述导线。只能依靠同一帧内连续可见的几何路径、插头落点以及相邻帧中的空间连续性追踪导线。
7. 测量阶段画面优先用于观察电路；记录阶段若仍能看见器材，可补充确认状态连续性。纸面书写本身不能证明电表极性。
8. 对每块表只观察正接线柱和负接线柱各自导线远端属于 source_positive_side、source_negative_side、unconnected 或 unclear，不解释这种组合是否正确。
9. 模糊、遮挡、画面晃动、候选误检或没有看全时填写 unclear，不把看不清改写成某种连接。
10. 证据接近均衡时选择 unclear，同时降低 confidence。
11. 只依据图片，不使用文件名、学生身份、人工分数、历史预测或标准答案。
12. 只有局部或全景中确实看见表盘身份、接线柱区域和插头落点时，才可声称某个正负端子或量程端子已连接；看不清时写 unclear，不得虚构精确端子细节。
13. 测量时指针持续反向偏转可支持 reversed，但不能在看不见端子和拓扑时单独定案。相邻两张以上画面可互补确认同一处错误，不要求每张都完整显示。
14. 本项目使用同一类金黄/绿色外观电池：金黄色凸起端直接作为单节电池正极视觉证据，绿色宽平端直接作为单节电池负极视觉证据。颜色端清楚时可以直接确定该节电池极性，并可跨相邻帧持续跟踪同一端；凸起/平端、可见 +/− 印字和串联桥片继续作为互相校验的证据。两节串联时，桥片必须连接一节的金黄色正端与另一节的绿色负端；真正对外输出只能剩下一个未接桥片的金黄色正端和一个未接桥片的绿色负端。禁止声称桥片连接绿色-绿色或金黄-金黄，也禁止把两个金黄色正端或两个绿色负端同时写成对外输出；画面若无法满足该结构，source_polarity_state 写 unclear。
15. 导线或插头的任何颜色都不属于本任务证据。红色、黑色既不能判断正负极，也不能帮助区分或跟踪“同一根导线”，更不能提高置信度。禁止在 source_positive_evidence、两块表的 evidence 和 observation_summary 中用“红线”“黑线”“红色/黑色导线或插头”等说法描述推理。若去掉导线颜色后无法确认连续路径，相关 destination 必须写 unclear。
16. 电表自身两个接线柱都有插头，不等于该电表已经接入电路。必须继续检查两根导线的远端；若任一必要远端在全景或相邻帧中明显悬空、香蕉插头未插入任何器材，该端填写 unconnected。
17. 按固定顺序观察：先写电源正端的可见依据，再识别 A/V，再分别填写正接线柱和负接线柱导线远端的 destination。不能跳过电源极性就声称某端属于 source_positive_side。
18. destination 描述的是电路拓扑高低电势侧，不要求导线直接连到电池。不得根据你认为实验“应该正确”来倒填 destination；必须忠实记录实际画面去向。
19. 本项目电流表正视、面板朝上且画面未镜像时，三个接线柱从左到右固定为：左侧“-”公共负接线柱、中间正接线柱（小量程，通常标 0.6）、右侧正接线柱（大量程，通常标 3）。该布局可在印字局部模糊时作为辅助先验；若器材旋转、倒置、透视方向不确定或画面可能镜像，必须先恢复面板朝向，不能直接套用画面左中右。面板“-”、0.6、3、15 等印字清楚时始终以实际印字为准。positive_terminal_destination 只按端子印字与无颜色的几何连续路径追踪实际使用的量程数字端，negative_terminal_destination 同样追踪“-”公共端。若两张以上相邻测量帧共同确认上述端子去向交换，且该端点观察 confidence >= 0.85，即使另一张表针观察暂时显示正常偏转，也保留端子接反事实，不要用单次表针结果覆盖多帧端子证据。
20. 只观察测量阶段的指针。这类弧形表盘的零刻度在刻度弧最左端，不在竖直方向；指针竖直或向右上方指向刻度数字，属于从左端零位向刻度增大方向偏转，填 normal_positive_deflection。只有指针越过最左端零刻度、向刻度弧外侧反打或顶住左侧挡针，才填 reverse_below_zero。停在最左端零位、被遮挡、看不清或仅记录阶段可见填 zero_or_unclear。不要用指针推测导线去向。

只返回一个合法 JSON 对象，不使用 Markdown，不增加字段：
{{
  "rubric_id": "{RUBRIC_ID}",
  "confidence": 0.0,
  "evidence_quality": "high | medium | low",
  "source_polarity_state": "visibly_established | unclear",
  "source_positive_end_identity": "gold_raised | plus_mark | other_visible | unclear",
  "source_negative_end_identity": "green_flat | minus_mark | other_visible | unclear",
  "source_series_bridge_state": "gold_to_green | not_visible | not_applicable | conflict",
  "source_positive_evidence": "说明电池凸点/平端、正负标识和串联桥片如何确定对外正端；看不清则明确写不清楚",
  "ammeter": {{
    "positive_terminal_destination": "source_positive_side | source_negative_side | unconnected | unclear",
    "negative_terminal_destination": "source_positive_side | source_negative_side | unconnected | unclear",
    "pointer_state": "normal_positive_deflection | reverse_below_zero | zero_or_unclear",
    "confidence": 0.0,
    "evidence": "综合哪些可见局部得出判断",
    "evidence_seconds": [0.0]
  }},
  "voltmeter": {{
    "positive_terminal_destination": "source_positive_side | source_negative_side | unconnected | unclear",
    "negative_terminal_destination": "source_positive_side | source_negative_side | unconnected | unclear",
    "pointer_state": "normal_positive_deflection | reverse_below_zero | zero_or_unclear",
    "confidence": 0.0,
    "evidence": "综合哪些可见局部得出判断",
    "evidence_seconds": [0.0]
  }},
  "observation_summary": "只总结可见端点去向和不清楚之处，不评价是否正确"
}}
"""


def pointer_prompt_text() -> str:
    return """只观察测量阶段画面中标有 A 和 V 的表盘指针，不分析接线、电池、导线或评分。
弧形刻度零位在最左端。指针指向竖直或右上方的刻度数字是 normal_positive_deflection；只有越过最左端零刻度向刻度外反打才是 reverse_below_zero；停在最左端、遮挡或看不清是 zero_or_unclear。
相邻测量帧可以互补。导线或插头颜色完全不参与观察，也不要在 evidence 中描述红线、黑线或其他导线颜色。只返回一个 JSON 对象，不使用 Markdown，不增加字段：
{
  "ammeter": "normal_positive_deflection | reverse_below_zero | zero_or_unclear",
  "voltmeter": "normal_positive_deflection | reverse_below_zero | zero_or_unclear",
  "confidence": 0.0,
  "evidence": "说明 A/V 指针相对最左端零位的位置"
}
"""


def reading_sign_prompt_text() -> str:
    return """你只观察测量阶段和记录阶段中的电流、电压读数符号，不分析导线拓扑，不评分。

允许同一阶段或相邻帧互补：一帧看清 A/V 身份，另一帧看清指针或纸面数值。分别观察：
1. ammeter_face_sign：标有 A 的电流表。指针从最左端零位向刻度数字方向偏转是 positive；只有越过最左端零刻度、向刻度弧外反打才是 negative；停在零位是 zero；遮挡或看不清是 unclear。
2. voltmeter_face_sign：标有 V 的电压表，规则同上。
3. recorded_current_sign：记录纸上明确属于 I 或电流/A 的最终数值。数值前紧邻清楚的“-”才是 negative；清楚正数是 positive；清楚为0是 zero；没有看清数值与符号的对应关系是 unclear。
4. recorded_voltage_sign：记录纸上明确属于 U 或电压/V 的最终数值，规则同上。

宽松约束：
- 纸面横线、表格边框、题号短横、公式中的减法符号，以及不紧邻 I/U 最终数值的“-”都不能当负读数。
- `U=___`、`I=___`、空白格、只有单位或没有写入清楚数字时必须填 unclear；不能因为没看见负号就填 positive。
- A/V 表盘身份、指针和纸面读数可以来自相邻帧，但不得把不同仪表或不同数据行拼成一个负读数。
- 正数、零、遮挡、模糊或冲突不得改写成 negative；冲突时填 unclear。
- 导线或插头颜色完全不参与观察，也不要在 evidence 中描述红线、黑线或其他导线颜色。
- 不使用文件名、学生身份、人工分数、历史预测或标准答案，不输出评分结果。

只返回一个合法 JSON 对象，不使用 Markdown，不增加字段：
{
  "ammeter_face_sign": "positive | negative | zero | unclear",
  "voltmeter_face_sign": "positive | negative | zero | unclear",
  "recorded_current_sign": "positive | negative | zero | unclear",
  "recorded_voltage_sign": "positive | negative | zero | unclear",
  "confidence": 0.0,
  "evidence_seconds": [0.0],
  "evidence": "说明负号与哪个 A/V 表针或 I/U 最终数值直接对应；没有负读数则说明看见的是正数、零或不清楚"
}
"""


def select_stage_video(manifest: dict[str, Any], video_id: str) -> dict[str, Any]:
    videos = manifest.get("videos")
    if not isinstance(videos, list):
        raise ValueError("Stage manifest videos must be a list")
    video = next((item for item in videos if str(item.get("video_id")) == video_id), None)
    if not isinstance(video, dict):
        raise ValueError(f"Video {video_id} is not present in the measurement/recording manifest")
    return video


def evenly_sample(items: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if len(items) <= maximum:
        return list(items)
    if maximum == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (maximum - 1)) for index in range(maximum)]
    return [items[index] for index in dict.fromkeys(indices)]


def detector_frame_numbers(detector_root: Path, video_id: str) -> set[int]:
    json_dir = detector_root / f"video_{video_id}" / "json"
    if not json_dir.is_dir():
        return set()
    numbers: set[int] = set()
    for path in json_dir.glob("frame_*_colored_v4.json"):
        match = re.match(r"frame_(\d+)_colored_v4\.json$", path.name)
        if match:
            numbers.add(int(match.group(1)))
    return numbers


def select_stage_frames(
    video: dict[str, Any],
    maximum_per_phase: int = 4,
    preferred_frame_numbers: set[int] | None = None,
) -> list[dict[str, Any]]:
    frames = video.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Video stage frames must be a list")
    selected: list[dict[str, Any]] = []
    for phase in OBSERVATION_PHASES:
        candidates = [
            item
            for item in frames
            if isinstance(item, dict)
            and item.get("stage") in OBSERVATION_STAGES
            and item.get("observation_phase") == phase
            and isinstance(item.get("timestamp_seconds"), (int, float))
            and isinstance(item.get("output_frame_path"), str)
        ]
        candidates.sort(key=lambda item: float(item["timestamp_seconds"]))
        preferred = [
            item
            for item in candidates
            if preferred_frame_numbers
            and isinstance(item.get("frame_number"), int)
            and item["frame_number"] in preferred_frame_numbers
        ]
        pool = preferred if preferred else candidates
        selected.extend(evenly_sample(pool, maximum_per_phase) if pool else [])
    selected.sort(key=lambda item: float(item["timestamp_seconds"]))
    if not selected:
        raise ValueError("No measurement or recording frames are available")
    return selected


def detector_candidates_for_frame(
    detector_root: Path,
    video_id: str,
    frame_number: int,
    maximum: int = 3,
) -> tuple[list[dict[str, Any]], str | None]:
    path = detector_root / f"video_{video_id}" / "json" / f"frame_{frame_number:08d}_colored_v4.json"
    if not path.is_file():
        return [], None
    payload = read_json(path)
    frame_index = payload.get("frame_index")
    summaries = frame_index.get("candidate_summaries") if isinstance(frame_index, dict) else None
    if not isinstance(summaries, list):
        return [], str(path.resolve())
    candidates: list[dict[str, Any]] = []
    for item in summaries:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(value, (int, float)) for value in box)
            or float(box[2]) <= 0
            or float(box[3]) <= 0
        ):
            continue
        candidates.append(
            {
                "candidate_id": str(item.get("candidate_id", f"candidate_{len(candidates) + 1:02d}")),
                "bbox_xywh": [int(round(float(value))) for value in box],
                "score": float(item.get("score", 0.0)),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:maximum], str(path.resolve())


def expanded_xywh_box(
    box: list[int],
    frame_size: tuple[int, int],
    horizontal_padding: float = 0.18,
    top_padding: float = 0.15,
    bottom_padding: float = 0.32,
) -> tuple[int, int, int, int]:
    frame_width, frame_height = frame_size
    x, y, width, height = box
    return (
        max(0, round(x - width * horizontal_padding)),
        max(0, round(y - height * top_padding)),
        min(frame_width, round(x + width * (1.0 + horizontal_padding))),
        min(frame_height, round(y + height * (1.0 + bottom_padding))),
    )


def reference_boxes(
    manifest: dict[str, Any], video_id: str
) -> tuple[dict[str, list[int]], tuple[int, int] | None, str | None]:
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        return {}, None, None
    source = next(
        (
            item
            for item in sources
            if isinstance(item, dict) and str(item.get("source_id", "")).startswith(f"video_{video_id}_")
        ),
        None,
    )
    if not isinstance(source, dict):
        return {}, None, None
    image_size = source.get("image_size")
    reference_size = None
    if (
        isinstance(image_size, list)
        and len(image_size) == 2
        and all(isinstance(value, int) and value > 0 for value in image_size)
    ):
        reference_size = (image_size[0], image_size[1])
    boxes: dict[str, list[int]] = {}
    crops = source.get("crops")
    if isinstance(crops, list):
        for crop in crops:
            if not isinstance(crop, dict):
                continue
            instrument = crop.get("instrument")
            box = crop.get("bbox_xyxy")
            if instrument in ROI_ORDER and isinstance(box, list) and len(box) == 4:
                boxes[str(instrument)] = [int(value) for value in box]
    return boxes, reference_size, str(source.get("source_id"))


def load_stage_frame(frame_path: Path, maximum_width: int = 1280) -> Any:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Unable to open extracted stage frame: {frame_path}")
    height, width = frame.shape[:2]
    if width > maximum_width:
        frame = cv2.resize(
            frame,
            (maximum_width, max(1, round(height * maximum_width / width))),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def expanded_scaled_box(
    box: list[int], reference_size: tuple[int, int], frame_size: tuple[int, int], padding: float = 0.08
) -> tuple[int, int, int, int]:
    reference_width, reference_height = reference_size
    frame_width, frame_height = frame_size
    x1, y1, x2, y2 = box
    x1 = round(x1 * frame_width / reference_width)
    x2 = round(x2 * frame_width / reference_width)
    y1 = round(y1 * frame_height / reference_height)
    y2 = round(y2 * frame_height / reference_height)
    pad_x = round((x2 - x1) * padding)
    pad_y = round((y2 - y1) * padding)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(frame_width, x2 + pad_x),
        min(frame_height, y2 + pad_y),
    )


def enhance_crop(crop: Any, minimum_long_edge: int = 900) -> Any:
    height, width = crop.shape[:2]
    long_edge = max(height, width)
    if long_edge < minimum_long_edge:
        scale = minimum_long_edge / max(1, long_edge)
        crop = cv2.resize(
            crop,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)


def write_jpeg(path: Path, image: Any, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"Unable to write image: {path}")


def build_media_groups(
    stage_frames: list[dict[str, Any]],
    boxes: dict[str, list[int]],
    reference_size: tuple[int, int] | None,
    media_dir: Path,
    detector_root: Path | None = None,
    video_id: str | None = None,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for index, stage_frame in enumerate(stage_frames, start=1):
        timestamp = float(stage_frame["timestamp_seconds"])
        stage = str(stage_frame["stage"])
        phase = str(stage_frame["observation_phase"])
        source_frame_path = Path(str(stage_frame["output_frame_path"])).resolve()
        source_frame = cv2.imread(str(source_frame_path), cv2.IMREAD_COLOR)
        if source_frame is None:
            raise RuntimeError(f"Unable to open extracted stage frame: {source_frame_path}")
        frame = load_stage_frame(source_frame_path)
        height, width = frame.shape[:2]
        frame_path = media_dir / f"group_{index:02d}_{timestamp:010.3f}s_{stage}_overview.jpg"
        write_jpeg(frame_path, frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        group: dict[str, Any] = {
            "group_index": index,
            "timestamp_seconds": timestamp,
            "stage": stage,
            "observation_phase": phase,
            "stage_interval_seconds": stage_frame.get("stage_interval_seconds"),
            "source_stage_frame": str(source_frame_path),
            "source_stage_frame_sha256": sha256_file(source_frame_path),
            "overview": str(frame_path.resolve()),
            "overview_sha256": sha256_file(frame_path),
            "overview_sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
            "rois": [],
            "dynamic_detector_json": None,
        }
        if reference_size is not None:
            for instrument in ROI_ORDER:
                box = boxes.get(instrument)
                if box is None:
                    continue
                x1, y1, x2, y2 = expanded_scaled_box(box, reference_size, (width, height))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = enhance_crop(frame[y1:y2, x1:x2])
                crop_path = media_dir / f"group_{index:02d}_{timestamp:010.3f}s_{stage}_{instrument}.jpg"
                write_jpeg(crop_path, crop, quality=92)
                group["rois"].append(
                    {
                        "instrument": instrument,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "path": str(crop_path.resolve()),
                        "sha256": sha256_file(crop_path),
                    }
                )
        frame_number = stage_frame.get("frame_number")
        if detector_root is not None and video_id is not None and isinstance(frame_number, int):
            candidates, detector_json = detector_candidates_for_frame(
                detector_root, video_id, frame_number
            )
            group["dynamic_detector_json"] = detector_json
            source_height, source_width = source_frame.shape[:2]
            for candidate_index, candidate in enumerate(candidates, start=1):
                x1, y1, x2, y2 = expanded_xywh_box(
                    candidate["bbox_xywh"], (source_width, source_height)
                )
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = enhance_crop(source_frame[y1:y2, x1:x2])
                crop_path = media_dir / (
                    f"group_{index:02d}_{timestamp:010.3f}s_{stage}_"
                    f"meter_candidate_{candidate_index:02d}.jpg"
                )
                write_jpeg(crop_path, crop, quality=92)
                group["rois"].append(
                    {
                        "instrument": "meter_candidate",
                        "candidate_label": f"候选{candidate_index}",
                        "detector_identity_trusted": False,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "detector_bbox_xywh": candidate["bbox_xywh"],
                        "detector_score": round(float(candidate["score"]), 6),
                        "path": str(crop_path.resolve()),
                        "sha256": sha256_file(crop_path),
                    }
                )
            if candidates:
                raw_boxes = [candidate["bbox_xywh"] for candidate in candidates]
                union_x1 = min(box[0] for box in raw_boxes)
                union_y1 = min(box[1] for box in raw_boxes)
                union_x2 = max(box[0] + box[2] for box in raw_boxes)
                union_y2 = max(box[1] + box[3] for box in raw_boxes)
                union_width = max(1, union_x2 - union_x1)
                union_height = max(1, union_y2 - union_y1)
                x1 = max(0, round(union_x1 - union_width * 0.22))
                y1 = max(0, round(union_y1 - union_height * 0.22))
                x2 = min(source_width, round(union_x2 + union_width * 0.22))
                y2 = min(source_height, round(union_y2 + union_height * 0.22))
                topology_crop = source_frame[y1:y2, x1:x2]
                crop_height, crop_width = topology_crop.shape[:2]
                if max(crop_height, crop_width) > 1600:
                    scale = 1600 / max(crop_height, crop_width)
                    topology_crop = cv2.resize(
                        topology_crop,
                        (round(crop_width * scale), round(crop_height * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                topology_crop = enhance_crop(topology_crop)
                topology_path = media_dir / (
                    f"group_{index:02d}_{timestamp:010.3f}s_{stage}_topology_context.jpg"
                )
                write_jpeg(topology_path, topology_crop, quality=92)
                group["rois"].append(
                    {
                        "instrument": "topology_context",
                        "detector_identity_trusted": False,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "path": str(topology_path.resolve()),
                        "sha256": sha256_file(topology_path),
                    }
                )
        groups.append(group)
    return groups


def data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def request_content(prompt: str, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for group in groups:
        timestamp = float(group["timestamp_seconds"])
        content.append(
            {
                "type": "text",
                "text": (
                    f"证据组 {group['group_index']}，阶段 {group['stage']}，"
                    f"时间 {timestamp:.3f} 秒。下一张为全景图。"
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url(Path(group["overview"]))},
            }
        )
        for roi in group["rois"]:
            if roi["instrument"] == "meter_candidate":
                roi_label = (
                    f"同一证据组的电表宽候选 {roi.get('candidate_label', '')}；"
                    "它可能误检且没有预设 A/V 身份，请先看表盘字母，再检查全部接线柱和插头落点。"
                )
            elif roi["instrument"] == "topology_context":
                roi_label = (
                    "同一证据组的高分辨率拓扑上下文；它没有预设器材身份。"
                    "请检查电池凸点/平端和串联桥片、导线远端、悬空香蕉插头及 A/V 接线柱之间的关系。"
                )
            else:
                roi_label = f"同一证据组的 {roi['instrument']} 增强局部。"
            content.append(
                {
                    "type": "text",
                    "text": roi_label,
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url(Path(roi["path"]))},
                }
            )
    return content


def build_client() -> tuple[Any, str]:
    from openai import OpenAI

    base_url = os.getenv("QWEN_API_BASE_URL", "").strip()
    token = os.getenv("QWEN_API_TOKEN", "").strip()
    model = os.getenv("QWEN_MODEL", "qwen").strip() or "qwen"
    if not base_url or not token:
        raise RuntimeError(
            "missing_qwen_configuration:QWEN_API_BASE_URL,QWEN_API_TOKEN"
        )
    client = OpenAI(
        base_url=base_url,
        api_key=token,
        timeout=180.0,
        max_retries=0,
    )
    return client, model


def call_qwen(client: Any, model: str, prompt: str, groups: list[dict[str, Any]]) -> dict[str, str]:
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": request_content(prompt, groups)}],
        max_tokens=1800,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = completion.choices[0]
    return {
        "finish_reason": choice.finish_reason or "unknown",
        "content": choice.message.content or "",
    }


def measurement_pointer_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **group,
            "rois": [
                roi for roi in group["rois"] if roi.get("instrument") == "meter_candidate"
            ],
        }
        for group in groups
        if group.get("observation_phase") == "measurement"
    ]


def reading_sign_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **group,
            "rois": [
                roi
                for roi in group["rois"]
                if roi.get("instrument") in {"ammeter", "voltmeter", "meter_candidate"}
            ],
        }
        for group in groups
        if group.get("observation_phase") in OBSERVATION_PHASES
    ]


def validate_pointer_response(content: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        value = parse_model_json(content)
    except Exception as exc:
        return None, [f"json_invalid:{type(exc).__name__}"]
    expected = {"ammeter", "voltmeter", "confidence", "evidence"}
    errors = [f"missing:{field}" for field in sorted(expected - set(value))]
    errors.extend(f"unexpected:{field}" for field in sorted(set(value) - expected))
    for meter_name in ("ammeter", "voltmeter"):
        if value.get(meter_name) not in POINTER_STATES:
            errors.append(f"{meter_name}_invalid")
    if not finite_probability(value.get("confidence")):
        errors.append("confidence_invalid")
    if not isinstance(value.get("evidence"), str) or not value["evidence"].strip():
        errors.append("evidence_missing")
    elif any(pattern.search(value["evidence"]) for pattern in WIRE_COLOR_REFERENCE_PATTERNS):
        errors.append("wire_color_evidence_forbidden")
    return (None, sorted(set(errors))) if errors else (value, [])


def validate_reading_sign_response(content: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        value = parse_model_json(content)
    except Exception as exc:
        return None, [f"json_invalid:{type(exc).__name__}"]
    sign_fields = {
        "ammeter_face_sign",
        "voltmeter_face_sign",
        "recorded_current_sign",
        "recorded_voltage_sign",
    }
    expected = sign_fields | {"confidence", "evidence_seconds", "evidence"}
    errors = [f"missing:{field}" for field in sorted(expected - set(value))]
    errors.extend(f"unexpected:{field}" for field in sorted(set(value) - expected))
    for field in sign_fields:
        if value.get(field) not in READING_SIGNS:
            errors.append(f"{field}_invalid")
    if not finite_probability(value.get("confidence")):
        errors.append("confidence_invalid")
    evidence_seconds = value.get("evidence_seconds")
    if not isinstance(evidence_seconds, list) or any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in evidence_seconds
    ):
        errors.append("evidence_seconds_invalid")
    if not isinstance(value.get("evidence"), str) or not value["evidence"].strip():
        errors.append("evidence_missing")
    elif any(pattern.search(value["evidence"]) for pattern in WIRE_COLOR_REFERENCE_PATTERNS):
        errors.append("wire_color_evidence_forbidden")
    return (None, sorted(set(errors))) if errors else (value, [])


def validate_response(
    content: str,
    meter_context_supplied: bool = True,
    pointer_overrides: dict[str, str] | None = None,
    pointer_observation_confidence: float | None = None,
    reading_sign_observation: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        value = parse_model_json(content)
    except Exception as exc:
        return None, [f"json_invalid:{type(exc).__name__}"]
    errors: list[str] = []
    errors.extend(f"missing:{field}" for field in sorted(RESPONSE_FIELDS - set(value)))
    errors.extend(f"unexpected:{field}" for field in sorted(set(value) - RESPONSE_FIELDS))
    if value.get("rubric_id") != RUBRIC_ID:
        errors.append("rubric_id_invalid")
    if not finite_probability(value.get("confidence")):
        errors.append("confidence_invalid")
    if value.get("evidence_quality") not in EVIDENCE_QUALITY:
        errors.append("evidence_quality_invalid")
    source_polarity_state = value.get("source_polarity_state")
    if source_polarity_state not in {"visibly_established", "unclear"}:
        errors.append("source_polarity_state_invalid")
    positive_end_identity = value.get("source_positive_end_identity")
    negative_end_identity = value.get("source_negative_end_identity")
    bridge_state = value.get("source_series_bridge_state")
    if positive_end_identity not in SOURCE_POSITIVE_END_IDENTITIES:
        errors.append("source_positive_end_identity_invalid")
    if negative_end_identity not in SOURCE_NEGATIVE_END_IDENTITIES:
        errors.append("source_negative_end_identity_invalid")
    if bridge_state not in SOURCE_SERIES_BRIDGE_STATES:
        errors.append("source_series_bridge_state_invalid")
    if source_polarity_state == "visibly_established":
        if positive_end_identity not in {"gold_raised", "plus_mark", "other_visible"}:
            errors.append("source_positive_end_identity_not_positive")
        if negative_end_identity not in {"green_flat", "minus_mark", "other_visible"}:
            errors.append("source_negative_end_identity_not_negative")
        if bridge_state == "conflict":
            errors.append("source_series_bridge_conflict")
    source_positive_evidence = value.get("source_positive_evidence")
    if not isinstance(source_positive_evidence, str) or not source_positive_evidence.strip():
        errors.append("source_positive_evidence_missing")
    elif any(pattern.search(source_positive_evidence) for pattern in WIRE_COLOR_REFERENCE_PATTERNS):
        errors.append("source_polarity_wire_color_evidence_forbidden")
    elif source_polarity_state == "visibly_established":
        if not re.search(
            r"金黄|绿色|凸|平|标识|标记|桥片|串联|电池.*[+＋正]|[+＋正].*电池",
            source_positive_evidence,
        ):
            errors.append("source_positive_evidence_not_visual")
    for meter_name in ("ammeter", "voltmeter"):
        meter = value.get(meter_name)
        if not isinstance(meter, dict):
            errors.append(f"{meter_name}_invalid")
            continue
        errors.extend(
            f"{meter_name}_missing:{field}" for field in sorted(METER_FIELDS - set(meter))
        )
        errors.extend(
            f"{meter_name}_unexpected:{field}" for field in sorted(set(meter) - METER_FIELDS)
        )
        for field in ("positive_terminal_destination", "negative_terminal_destination"):
            if meter.get(field) not in TERMINAL_DESTINATIONS:
                errors.append(f"{meter_name}_{field}_invalid")
            elif source_polarity_state == "unclear" and meter.get(field) not in {"unconnected", "unclear"}:
                errors.append(f"{meter_name}_{field}_requires_visible_source_polarity")
        if meter.get("pointer_state") not in POINTER_STATES:
            errors.append(f"{meter_name}_pointer_state_invalid")
        if not finite_probability(meter.get("confidence")):
            errors.append(f"{meter_name}_confidence_invalid")
        if not isinstance(meter.get("evidence"), str) or not meter["evidence"].strip():
            errors.append(f"{meter_name}_evidence_missing")
        else:
            if any(pattern.search(meter["evidence"]) for pattern in WIRE_COLOR_REFERENCE_PATTERNS):
                errors.append(f"{meter_name}_wire_color_evidence_forbidden")
            if not meter_context_supplied and re.search(
                r"(?:正|负|\+|\-|量程|接线柱).{0,12}(?:端|柱|插|接)|(?:端|柱|插|接).{0,12}(?:正|负|\+|\-|量程)",
                meter["evidence"],
            ):
                errors.append(f"{meter_name}_terminal_claim_without_meter_context")
        evidence_seconds = meter.get("evidence_seconds")
        if not isinstance(evidence_seconds, list) or any(
            not isinstance(item, (int, float)) or isinstance(item, bool) for item in evidence_seconds
        ):
            errors.append(f"{meter_name}_evidence_seconds_invalid")
    if not isinstance(value.get("observation_summary"), str) or not value["observation_summary"].strip():
        errors.append("observation_summary_missing")
    elif any(
        pattern.search(value["observation_summary"])
        for pattern in WIRE_COLOR_REFERENCE_PATTERNS
    ):
        errors.append("observation_summary_wire_color_evidence_forbidden")
    if errors:
        return None, sorted(set(errors))

    applied_pointer_overrides: dict[str, dict[str, str]] = {}
    for meter_name, pointer_state in (pointer_overrides or {}).items():
        if meter_name not in {"ammeter", "voltmeter"} or pointer_state not in POINTER_STATES:
            continue
        if value[meter_name]["pointer_state"] != pointer_state:
            applied_pointer_overrides[meter_name] = {
                "integrated_observation": value[meter_name]["pointer_state"],
                "focused_observation": pointer_state,
            }
        value[meter_name]["pointer_state"] = pointer_state

    applied_negative_readings: dict[str, dict[str, Any]] = {}
    for meter_name in ("ammeter", "voltmeter"):
        meter = value[meter_name]
        positive_destination = meter["positive_terminal_destination"]
        negative_destination = meter["negative_terminal_destination"]
        pointer_state = meter["pointer_state"]
        sign_fields = (
            ("ammeter_face_sign", "recorded_current_sign")
            if meter_name == "ammeter"
            else ("voltmeter_face_sign", "recorded_voltage_sign")
        )
        negative_fields: list[str] = []
        reading_confidence = None
        if isinstance(reading_sign_observation, dict):
            reading_confidence = reading_sign_observation.get("confidence")
            if (
                finite_probability(reading_confidence)
                and float(reading_confidence) >= READING_SIGN_MIN_CONFIDENCE
            ):
                negative_fields = [
                    field for field in sign_fields if reading_sign_observation.get(field) == "negative"
                ]
        if negative_fields:
            applied_negative_readings[meter_name] = {
                "fields": negative_fields,
                "confidence": float(reading_confidence),
                "evidence_seconds": reading_sign_observation.get("evidence_seconds"),
                "evidence": reading_sign_observation.get("evidence"),
            }
            derived_state, derived_violation = "likely_incorrect", "reversed"
        elif "unconnected" in {positive_destination, negative_destination}:
            derived_state, derived_violation = "likely_incorrect", "not_connected"
        elif (
            positive_destination == "source_positive_side"
            and negative_destination == "source_negative_side"
        ):
            if pointer_state == "reverse_below_zero":
                derived_state, derived_violation = "unclear", "unclear"
            else:
                derived_state, derived_violation = "likely_correct", "none"
        elif (
            positive_destination == "source_negative_side"
            and negative_destination == "source_positive_side"
        ):
            evidence_seconds = meter.get("evidence_seconds")
            multi_frame_endpoint_support = (
                finite_probability(meter.get("confidence"))
                and float(meter["confidence"]) >= TERMINAL_REVERSAL_MIN_CONFIDENCE
                and isinstance(evidence_seconds, list)
                and len(
                    {
                        round(float(second), 3)
                        for second in evidence_seconds
                        if isinstance(second, (int, float)) and not isinstance(second, bool)
                    }
                )
                >= TERMINAL_REVERSAL_MIN_EVIDENCE_FRAMES
            )
            focused_pointer_is_reliable_normal = (
                pointer_state == "normal_positive_deflection"
                and finite_probability(pointer_observation_confidence)
                and float(pointer_observation_confidence) >= TERMINAL_REVERSAL_MIN_CONFIDENCE
            )
            if pointer_state == "normal_positive_deflection" and (
                not multi_frame_endpoint_support or focused_pointer_is_reliable_normal
            ):
                derived_state, derived_violation = "unclear", "unclear"
            else:
                derived_state, derived_violation = "likely_incorrect", "reversed"
        else:
            derived_state, derived_violation = "unclear", "unclear"
        meter["state"] = derived_state
        meter["violation_type"] = derived_violation

    incorrect_meters = [
        meter_name
        for meter_name in ("ammeter", "voltmeter")
        if value[meter_name]["state"] == "likely_incorrect"
    ]
    local_result = "fail" if incorrect_meters else "pass"
    value["model_reported_result"] = None
    value["meter_state_overrides"] = {}
    value["pointer_state_overrides"] = applied_pointer_overrides
    value["negative_reading_applied"] = applied_negative_readings
    value["local_result_applied"] = True
    value["result"] = local_result
    if local_result == "pass":
        value["fail_trigger"] = None
    else:
        triggers = [
            f"{meter_name}:{value[meter_name]['violation_type']}" for meter_name in incorrect_meters
        ]
        value["fail_trigger"] = "、".join(triggers)
    value["reason"] = (
        "本地端点归并发现 " + value["fail_trigger"]
        if local_result == "fail"
        else "本地端点归并未发现可见接反或明显未接入；不清楚项按宽松规则通过。"
    )
    original_confidence = float(value["confidence"])
    confidence_ceiling = 1.0
    if any(value[meter_name]["state"] == "unclear" for meter_name in ("ammeter", "voltmeter")):
        confidence_ceiling = min(confidence_ceiling, 0.65)
    if value["evidence_quality"] == "low":
        confidence_ceiling = min(confidence_ceiling, 0.60)
    if value["source_polarity_state"] == "unclear":
        confidence_ceiling = min(confidence_ceiling, 0.65)
    value["confidence"] = min(original_confidence, confidence_ceiling)
    value["confidence_calibrated"] = value["confidence"] != original_confidence
    return value, []


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lenient binary ammeter/voltmeter polarity assessment.")
    parser.add_argument("--video-id", required=True, help="Input/output association only; never selects an algorithm.")
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--detector-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-groups-per-phase", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    stage_manifest_path = args.stage_manifest.expanduser().resolve()
    stage_manifest = read_json(stage_manifest_path)
    video = select_stage_video(stage_manifest, str(args.video_id))
    detector_root = args.detector_root.expanduser().resolve()
    preferred_frames = detector_frame_numbers(detector_root, str(args.video_id))
    selected_stage_frames = select_stage_frames(
        video, args.max_groups_per_phase, preferred_frame_numbers=preferred_frames
    )

    output_dir = args.output_root.expanduser().resolve() / f"video_{args.video_id}"
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {output_dir}")
    media_dir = output_dir / "media"
    output_dir.mkdir(parents=True, exist_ok=False)

    reference_manifest_path = args.reference_manifest.expanduser().resolve()
    reference_manifest = read_json(reference_manifest_path) if reference_manifest_path.is_file() else {}
    boxes, reference_size, reference_source_id = reference_boxes(reference_manifest, str(args.video_id))
    polarity_boxes = {
        instrument: box
        for instrument, box in boxes.items()
        if instrument in {"ammeter", "voltmeter"}
    }
    if preferred_frames:
        # Static boxes drift badly in moving-camera video. Candidate-backed wide crops
        # are tied to the current frame, so do not mix in stale instrument labels.
        polarity_boxes = {}
        reference_size = None
    groups = build_media_groups(
        selected_stage_frames,
        polarity_boxes,
        reference_size,
        media_dir,
        detector_root=detector_root,
        video_id=str(args.video_id),
    )
    available_rois = sorted(
        {
            str(roi["instrument"])
            for group in groups
            for roi in group["rois"]
            if isinstance(roi, dict) and isinstance(roi.get("instrument"), str)
        }
    )
    prompt = prompt_text(groups, available_rois)
    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    pointer_prompt = pointer_prompt_text()
    pointer_prompt_path = output_dir / "pointer_prompt.txt"
    pointer_prompt_path.write_text(pointer_prompt, encoding="utf-8")
    reading_prompt = reading_sign_prompt_text()
    reading_prompt_path = output_dir / "reading_sign_prompt.txt"
    reading_prompt_path.write_text(reading_prompt, encoding="utf-8")
    input_manifest = {
        "schema_version": "1.0",
        "artifact_type": "meter_polarity_measurement_recording_input_manifest",
        "rubric_id": RUBRIC_ID,
        "video_id": str(args.video_id),
        "stage_manifest": str(stage_manifest_path),
        "stage_manifest_sha256": sha256_file(stage_manifest_path),
        "stage_scope": sorted({str(group["stage"]) for group in groups}),
        "observation_phase_scope": sorted({str(group["observation_phase"]) for group in groups}),
        "wiring_stage_accessed": False,
        "reference_manifest": str(reference_manifest_path),
        "reference_source_id": reference_source_id,
        "dynamic_detector_root": str(detector_root),
        "dynamic_detector_frame_count": sum(
            1 for group in groups if group.get("dynamic_detector_json")
        ),
        "dynamic_meter_candidate_count": sum(
            1
            for group in groups
            for roi in group["rois"]
            if roi.get("instrument") == "meter_candidate"
        ),
        "timestamps_seconds": [group["timestamp_seconds"] for group in groups],
        "available_rois": available_rois,
        "groups": groups,
        "prompt_path": str(prompt_path.resolve()),
        "prompt_sha256": sha256_text(prompt),
        "pointer_prompt_path": str(pointer_prompt_path.resolve()),
        "pointer_prompt_sha256": sha256_text(pointer_prompt),
        "reading_sign_prompt_path": str(reading_prompt_path.resolve()),
        "reading_sign_prompt_sha256": sha256_text(reading_prompt),
        "excel_accessed": False,
        "labels_accessed": False,
    }
    write_json(output_dir / "input_manifest.json", input_manifest)

    if args.prepare_only:
        write_json(
            output_dir / "run_report.json",
            {
                "status": "prepared",
                "qwen_called": False,
                "result": None,
                "input_manifest": str((output_dir / "input_manifest.json").resolve()),
            },
        )
        print(json.dumps({"status": "prepared", "output_dir": str(output_dir)}, ensure_ascii=False))
        return 0

    client, model = build_client()
    focused_pointer: dict[str, Any] | None = None
    pointer_errors: list[str] = []
    pointer_raw_paths: list[str] = []
    focused_groups = measurement_pointer_groups(groups)
    if focused_groups:
        for attempt in (1, 2):
            attempt_prompt = pointer_prompt
            if attempt == 2:
                attempt_prompt += "\n上一响应格式或证据无效；只按指定四个字段返回合法 JSON，并删除所有导线和插头颜色描述。"
            pointer_raw = call_qwen(client, model, attempt_prompt, focused_groups)
            pointer_raw_path = output_dir / f"pointer_raw_response_attempt_{attempt:02d}.json"
            write_json(pointer_raw_path, pointer_raw)
            pointer_raw_paths.append(str(pointer_raw_path.resolve()))
            focused_pointer, pointer_errors = validate_pointer_response(pointer_raw["content"])
            if focused_pointer is not None:
                break
    pointer_overrides = (
        {meter_name: str(focused_pointer[meter_name]) for meter_name in ("ammeter", "voltmeter")}
        if focused_pointer is not None
        else {}
    )
    focused_reading_sign: dict[str, Any] | None = None
    reading_sign_errors: list[str] = []
    reading_sign_raw_paths: list[str] = []
    focused_reading_groups = reading_sign_groups(groups)
    if focused_reading_groups:
        for attempt in (1, 2):
            attempt_prompt = reading_prompt
            if attempt == 2:
                attempt_prompt += "\n上一响应格式或证据无效；只按指定七个字段返回合法 JSON，并删除所有导线和插头颜色描述。"
            reading_raw = call_qwen(client, model, attempt_prompt, focused_reading_groups)
            reading_raw_path = output_dir / f"reading_sign_raw_response_attempt_{attempt:02d}.json"
            write_json(reading_raw_path, reading_raw)
            reading_sign_raw_paths.append(str(reading_raw_path.resolve()))
            focused_reading_sign, reading_sign_errors = validate_reading_sign_response(
                reading_raw["content"]
            )
            if focused_reading_sign is not None:
                break
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    raw_paths: list[str] = []
    for attempt in (1, 2):
        attempt_prompt = prompt
        if attempt == 2:
            attempt_prompt += (
                "\n上一响应没有通过本地证据与 JSON 校验："
                + json.dumps(errors, ensure_ascii=False)
                + "。请重新观察并纠正这些问题，只返回指定 JSON。"
                "导线或插头颜色必须完全忽略；不得照搬被拒绝的红黑线描述。"
            )
        raw = call_qwen(client, model, attempt_prompt, groups)
        raw_path = output_dir / f"raw_response_attempt_{attempt:02d}.json"
        write_json(raw_path, raw)
        raw_paths.append(str(raw_path.resolve()))
        meter_context_supplied = any(
            roi.get("instrument") in {"ammeter", "voltmeter", "meter_candidate"}
            for group in groups
            for roi in group["rois"]
        )
        parsed, errors = validate_response(
            raw["content"],
            meter_context_supplied=meter_context_supplied,
            pointer_overrides=pointer_overrides,
            pointer_observation_confidence=(
                float(focused_pointer["confidence"])
                if isinstance(focused_pointer, dict) and finite_probability(focused_pointer.get("confidence"))
                else None
            ),
            reading_sign_observation=focused_reading_sign,
        )
        if parsed is not None:
            break

    if parsed is None:
        write_json(
            output_dir / "run_report.json",
            {
                "status": "response_invalid",
                "qwen_called": True,
                "attempts": len(raw_paths),
                "validation_errors": errors,
                "raw_responses": raw_paths,
                "model": model,
            },
        )
        raise RuntimeError(f"Qwen response failed validation: {errors}")

    result = {
        "schema_version": "1.0",
        "artifact_type": "meter_polarity_measurement_recording_binary_result",
        "rubric_id": RUBRIC_ID,
        "video_id": str(args.video_id),
        "result": parsed["result"],
        "predicted_score": 1 if parsed["result"] == "pass" else 0,
        "confidence": float(parsed["confidence"]),
        "evidence_quality": parsed["evidence_quality"],
        "source_polarity_state": parsed["source_polarity_state"],
        "source_positive_end_identity": parsed["source_positive_end_identity"],
        "source_negative_end_identity": parsed["source_negative_end_identity"],
        "source_series_bridge_state": parsed["source_series_bridge_state"],
        "source_positive_evidence": parsed["source_positive_evidence"],
        "ammeter": parsed["ammeter"],
        "voltmeter": parsed["voltmeter"],
        "fail_trigger": parsed["fail_trigger"],
        "reason": parsed["reason"],
        "model_reported_result": parsed["model_reported_result"],
        "local_result_applied": parsed["local_result_applied"],
        "meter_state_overrides": parsed["meter_state_overrides"],
        "pointer_state_overrides": parsed["pointer_state_overrides"],
        "negative_reading_applied": parsed["negative_reading_applied"],
        "focused_pointer_observation": focused_pointer,
        "focused_pointer_validation_errors": pointer_errors,
        "focused_reading_sign_observation": focused_reading_sign,
        "focused_reading_sign_validation_errors": reading_sign_errors,
        "confidence_calibrated": parsed["confidence_calibrated"],
        "stage_scope": sorted({str(group["stage"]) for group in groups}),
        "observation_phase_scope": sorted({str(group["observation_phase"]) for group in groups}),
        "wiring_stage_accessed": False,
        "timestamps_seconds": [group["timestamp_seconds"] for group in groups],
        "input_manifest": str((output_dir / "input_manifest.json").resolve()),
        "raw_responses": raw_paths,
        "pointer_raw_responses": pointer_raw_paths,
        "reading_sign_raw_responses": reading_sign_raw_paths,
        "model": model,
        "binary_lenient": True,
        "excel_accessed": False,
        "labels_accessed": False,
    }
    result_path = output_dir / "result.json"
    write_json(result_path, result)
    write_json(
        output_dir / "run_report.json",
        {
            "status": "completed",
            "qwen_called": True,
            "attempts": len(raw_paths),
            "result": parsed["result"],
            "confidence": float(parsed["confidence"]),
            "result_path": str(result_path.resolve()),
            "result_sha256": sha256_file(result_path),
            "validation_errors": [],
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "result": parsed["result"],
                "confidence": float(parsed["confidence"]),
                "result_path": str(result_path.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
