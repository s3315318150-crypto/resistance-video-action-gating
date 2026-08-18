# R8 动态电池 ROI 与断开-换接-闭合事件链

## 1. 算法定位

| 字段 | 值 |
|---|---|
| Algorithm ID | `resistance_disconnect_battery_sequence_v3_dynamic_roi` |
| Rubric | R8：更换电池组前先断开开关 |
| 主结果 | `pass` / `fail` |
| Agent Skills | `battery.recovery_episode` / `battery.wiring_transition` / `battery.broad_transition_search` |
| 主脚本 | `scripts/run_resistance_disconnect_battery_sequence_v1.py` |
| 确定性 reducer | `scripts/resistance_disconnect_battery_sequence_core.py` |
| Agent wrapper | `agent/scripts/run_rubric8_specialized.py` |

评价目标是确认以下有序链：

```text
开关断开
  -> 电池盒仍为两节有效连接
  -> 学生直接操作电池端子
  -> 完成两节到一节的导线换接
  -> 一节连接稳定
  -> 开关重新闭合
```

这里的“换电池”不要求把电芯从电池盒中取出。对三抽头串联电池盒：

```text
T0 -- cell 1 -- T1 -- cell 2 -- T2
```

有效变化是：

```text
before: T0-T2  -> 2 cells
after:  T0-T1  -> 1 cell
    or: T1-T2  -> 1 cell
```

## 2. 为什么是通用情境 Skill

该算法不保存视频 8、16、24、32、38 的固定时间点、固定 ROI 或预测结果。

实时选择逻辑：

- 有 `circuit_rewiring` 情境：使用 `battery.recovery_episode`，搜索重接线和重复测量之间的 recovery gap；
- 没有重接线但有 `circuit_wiring`：使用 `battery.wiring_transition`，搜索当前视频的初次接线窗口；
- 两种接线阶段都不可靠：使用 `battery.broad_transition_search`，合并当前视频可观察的接线、重接线和 recovery gap 窗口。

三个 Skill 只改变 episode 的时间来源，后续都调用同一个动态电池 ROI、相邻帧跟踪、直接接触 verifier 和有序 reducer。它们不按视频 ID、文件名、SHA-256 或历史结果分支。

算法结果中固定写入：

```json
{
  "configured_roi_used": false,
  "reference_frame_used": false,
  "video_id_routing_used": false
}
```

`video_id` 只用于将当前视频与动作记录、输出目录绑定，不选择算法分支或结果。

## 3. 输入

### 3.1 必需输入

| 输入 | 作用 |
|---|---|
| 当前 MP4 | 重新解码、抽帧和动态定位 |
| action summary | 提供 `circuit_rewiring`、`recording_1`、`measurement_1` 等阶段时间段 |
| Qwen endpoint/model | 执行全景粗筛、电池定位和结构化动作观察 |
| Skill parameters | 控制抽帧密度和动态 ROI 门槛 |

Excel 真值、教师错项和冻结预测不是算法输入。

### 3.2 默认参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `coarse_fps` | `2.0` | 候选窗全景粗筛 |
| `core_fps` | `5.0` | 核心换线区间 |
| `transition_fps` | `10.0` | 端点、开关与短转场密集帧 |
| `dynamic_roi_min_confidence` | `0.45` | 接受 Qwen 电池定位的最低置信度 |
| `time_mode` | `rewiring_recovery` | 由当前阶段情境选择 episode 时间来源 |

这些参数对所有视频使用同一套默认值，不包含视频 ID 对照表。它们仍属于开发集上调试过的全局参数，需用新视频盲测验证。

## 4. 处理流程

### 4.1 生成候选 episode

算法按 `time_mode` 生成独立 episode：

1. `rewiring_recovery`：`circuit_rewiring` 直接提供的换线 episode，加上重复 `recording_1`/`measurement_1` 阶段之间的 recovery-gap episode；
2. `wiring_transition`：当前 `circuit_wiring` 阶段的接线 episode；
3. `broad_transition_search`：当前 `circuit_wiring`、`circuit_rewiring` 和 recovery-gap 的去重并集。

默认向核心区间前后各扩展 `10 s`，用于捕获换接前后的稳定状态。每个 episode 独立 reducer，禁止把一个 episode 的开关证据与另一个 episode 的端子证据拼接。

### 4.2 粗筛候选时刻

在扩展窗中按 `coarse_fps` 抽取全景帧。OpenCV 计算清晰度与运动强度，再从均匀时刻与高运动时刻中选择有界证据包。

Qwen 粗筛只返回候选 `frame_id`：

- 电池盒可见帧；
- 手部/导线操作帧；
- 开关候选帧；
- 类似整理阶段的帧。

模型返回的 `frame_id` 必须存在于当前请求的帧组中，否则本地校验拒绝该证据。

### 4.3 动态电池 ROI

定位请求向 Qwen 发送当前视频的独立全景帧，而不是带缩放排版的联系表。返回字段：

```json
{
  "frame_id": "16_16_recovery_01_f00012750",
  "bbox_normalized": [0.08, 0.72, 0.42, 0.98],
  "confidence": 0.95,
  "evidence": "battery holder and terminals are visible"
}
```

本地校验：

- `frame_id` 必须属于当前请求；
- bbox 必须是四个归一化数值；
- `x2 > x1`、`y2 > y1`；
- 置信度不低于 `dynamic_roi_min_confidence`。

定位成功后，相邻帧使用 ORB 特征与 affine 跟踪传播 ROI。视角突变或特征失败时，保留全景帧作为诊断证据，不加载视频专属坐标。

### 4.4 密集相邻帧

算法将核心换线区间、直接接触窗和粗筛候选时刻合并为密集区间：

- 核心区间使用 `core_fps`；
- 短转场和候选时刻附近使用 `transition_fps`；
- 全部抽取帧留在磁盘；
- 发给模型的证据包有上限，并按时间排序。

每个密集帧保存：

- 时间戳；
- `frame_id`；
- 原始全景帧；
- 动态电池 ROI；
- ROI 来源帧与跟踪模式；
- 清晰度与运动诊断。

### 4.5 两条观察路径

算法对同一 episode 使用两类 Qwen 观察：

1. **direct-contact verifier**：检查手或导线是否真正触及电池盒端子；
2. **structured topology summary**：描述开关状态、换接前后端子对和换接完成时刻。

如果结构化总结的前后端子对矛盾，再调用 terminal-pair verifier。两条路径只能融合同一 episode 且真实 `frame_id` 可验证的证据。

### 4.6 确定性 reducer

Qwen 不直接输出 R8 分数。本地 reducer 按时间顺序检查：

1. 换接开始前开关为 `open`；
2. 稳定前态是 `T0-T2`；
3. 存在直接端子操作；
4. 稳定后态是 `T0-T1` 或 `T1-T2`；
5. 有效节数从 2 变为 1；
6. 重新闭合发生在换接完成之后。

任意一个 episode 完整通过，视频 R8 即为 `pass`；所有 episode 均不通过才是 `fail`。

## 5. 接口失败与二分类兜底

项目契约要求即使视觉证据不完整，主结果仍为 `pass/fail`。因此 Qwen 超时时仍可能生成：

```json
{
  "decision": "fail",
  "confidence": 0.25,
  "evidence_quality": "low"
}
```

这个二分类工件对于保持 Pipeline 契约有用，但不一定是有效的算法准确率样本。评测时必须同时检查：

- `facts.status`；
- `facts.observations` 数量；
- `direct_contact_verifier.usable`；
- `validation_errors`；
- `evidence_quality`；
- Qwen 请求的 `error`。

`observation_count = 0` 且关键请求全部超时时，该 `fail` 应报告为“接口兜底二分类”，不能宣称模型看到了错误操作。

## 6. 输出结构

```text
<output-root>/
  summary.json
  video_<id>/
    result.json
    <episode_id>/
      coarse_manifest.json
      dynamic_roi_manifest.json
      dense_roi_seeds.json
      dense_manifest.json
      direct_contact_manifest.json
      coarse/
        frames/
        localization_frames/
        battery_roi/
      dense/
        range_*/
          frames/
          battery_roi/
      screening/
        screen_*.json
      localization/
        localize_*.json
      facts/
        direct_contact_verifier.json
        structured_summary.json
        paired_frames/
```

`result.json` 中最重要的字段：

| 字段 | 含义 |
|---|---|
| `decision` | R8 二分类结果 |
| `confidence` | reducer 置信度 |
| `episodes` | 每个候选窗的独立结果 |
| `dynamic_roi_detection_count` | 当前 episode 的动态电池定位数 |
| `facts.observations` | 经本地校验的时序观察 |
| `reducer.episodes[].ordered_chain` | 通过时的完整有序证据链 |
| `evidence_quality` | 证据完整性诊断 |

## 7. 运行

### 7.1 从 Agent 执行

```powershell
python agent\run_agent.py `
  --video-id 16 `
  --run-id r8_dynamic_video16 `
  --mode execute `
  --scheduler deterministic
```

Agent 会先运行阶段与 Skill Router；三个 R8 Skill 都调用本专项执行器，`time_mode` 由当前阶段情况绑定后传入 runner。

### 7.2 单独运行 R8

```powershell
python agent\scripts\run_rubric8_specialized.py `
  --source-video "data\videos\sample.mp4" `
  --action-summary "agent\runs\sample_execute\action_summary.json" `
  --output-root "outputs\r8_dynamic_video16" `
  --video-ids "16" `
  --time-mode "rewiring_recovery" `
  --no-cache
```

`--time-mode` 也可以是 `wiring_transition` 或 `broad_transition_search`。正式 Agent execute 不读取 Excel，不加载历史 R8 预测；`specialized_best` 只保留给显式 regression/replay，默认配置已关闭。

专项结果缓存同时校验当前视频 SHA、动态算法 ID、`time_mode` 和 `dynamic_r8_execution` 标记。改变 Skill 情境不会误复用另一种时间窗的旧结果。每次真实重跑仍应使用新的 `output-root`。

## 8. 2026-08-16 五视频重跑

预测在读取 Excel 前冻结于：

```text
outputs/resistance_disconnect_battery_sequence_v3_dynamic_roi_five_retry_20260816/frozen_predictions.json
```

| 视频 | 预测 | 置信度 | 证据质量 | observations | Excel R8 | 说明 |
|---:|---|---:|---|---:|---|---|
| 8 | `pass` | 0.80 | high | 27 | `0.5（存疑）` | Excel 不是明确二分类 |
| 16 | `pass` | 0.80 | high | 52 | `1` | 有效有序链，一致 |
| 24 | `fail` | 0.25 | low | 25 | `0（错误）` | 有效观察，未形成完整链，一致 |
| 32 | `pass` | 0.80 | high | 28 | `1` | 有效有序链，一致 |
| 38 | `fail` | 0.25 | low | 0 | `0（错误）` | 关键请求超时；二分类一致但不是有效视觉命中 |

评测口径：

- Excel 明确 0/1 的冻结输出一致率：`4/4 = 100%`；
- 再排除视频 38 的接口兜底：`3/3 = 100%`；
- 视频 8 是 `0.5（存疑）`，不应在未声明映射规则时直接计入二分类准确率。

这五个视频长期用于开发和回归，上述数字不是新视频泛化证明。

## 9. 当前限制

1. Qwen 网关可能间歇性超时，大证据包的风险高于单图 preflight。
2. `2/5/10 fps`、`10 s` 边距和 `0.45` 阈值仍是开发集调试后的全局默认值。
3. 当前端子 reducer 针对三抽头串联电池盒；器材形态变化时需要新的设备 schema，不能通过视频 ID 适配。
4. 五个视频已参与开发，新视频必须使用冻结后评测。

## 10. 最小验收标准

一次可用的 R8 运行至少需要：

- `summary.json` 和 `video_<id>/result.json` 可解析；
- `decision` 只是 `pass/fail`；
- 动态 ROI 的 `frame_id` 和 bbox 通过本地校验；
- 保存原帧、ROI、模型原文和时间戳；
- `pass` 必须包含 `ordered_chain`；
- 接口错误与算法视觉结论分开报告；
- 预测冻结前未读取 Excel。

回到 Agent 总说明：[README.md](../../../README.md)。
