# Rubric 8：断开开关后改变串联电池节数

## 1. 算法目标

该专项算法判断伏安法测电阻视频是否完成以下有序操作：

```text
先断开开关
-> 改变串联电池节数
-> 换接期间保持断开
-> 换接完成后重新闭合开关
```

这里的“换电池”不是从电池盒中取出一节电池，而是改变固定两节电池盒的接线端点。三个抽头表示为：

```text
T0 -- cell 1 -- T1 -- cell 2 -- T2
```

有效变化必须是稳定接线从两节外端 `T0-T2` 改为单节端点对 `T0-T1` 或 `T1-T2`。

## 2. 代码入口

| 文件 | 职责 |
|---|---|
| `scripts/run_resistance_disconnect_battery_sequence_v1.py` | 生成动作窗口、recovery episode、局部帧和结构化观察 |
| `scripts/resistance_disconnect_battery_sequence_core.py` | 按开关、端点换接和顺序约束输出 `pass/fail` |
| `configs/resistance_disconnect_battery_rois.example.json` | 视频 ROI 和参考帧配置模板 |
| `tests/test_run_resistance_disconnect_battery_sequence_v1.py` | workflow、窗口和观察转换测试 |
| `tests/test_resistance_disconnect_battery_sequence_core.py` | 本地 reducer 时序测试 |

算法 ID：`resistance_disconnect_battery_sequence_v2_recovery_windows`

## 3. Episode 生成

每个 `circuit_rewiring=[s,e]` 独立生成一个 episode，默认使用 `[s-10,e+10]` 扩展窗补足换线前后的稳定状态。扩展窗不改变核心动作区间。

如果 `recording_1` 或 `measurement_1` 重复出现，系统还会检查相邻同类阶段之间的短暂 gap，并为该 gap 建立独立 recovery episode。该机制用于找回没有被七阶段分割显式标为 `circuit_rewiring` 的短换线动作。

不同 episode 的观察不能拼接。一个 episode 内必须独立形成完整证据链。

## 4. 视觉证据

### 4.1 换线前稳定状态

需要观察到电池盒两节外端接入：

```text
battery_before = T0-T2
effective_cells_before = 2
```

### 4.2 直接端点操作

不把普通手部运动、移动电池盒或触碰其他橙色仪器当作换电池证据。必须看到手或插头直接操作固定电池盒端点，并且该证据位于本次 terminal relocation 的起止范围内。

### 4.3 明确重接完成

模型必须给出重接完成帧。若该帧同时属于 `direct_contact_frame_ids`，workflow 保留更强的完成语义：

```json
{
  "direct_battery_contact": true,
  "terminal_action": "reconnect",
  "terminal_rewire_completed": true
}
```

这避免通用接触字段把同一帧已经确认的 `reconnect` 覆盖成 `uncertain`。

### 4.4 换线后稳定状态

完成后需要观察到稳定单节端点对：

```text
battery_after = T0-T1 or T1-T2
effective_cells_after = 1
```

## 5. 本地顺序 Reducer

最终 reducer 检查：

```text
switch open
-> stable T0-T2
-> direct terminal relocation
-> explicit reconnect completion
-> stable T0-T1/T1-T2
-> switch closed
```

闭合时间必须严格晚于 `explicit reconnect completion`。任一 episode 独立满足完整链条时，视频结果为 `pass`；全部 episode 均不满足时为 `fail`。

Qwen 只产生帧级观察。本地 reducer 决定最终二分类，防止模型根据实验常识直接猜测结果。

## 6. 运行方式

准备动作分割汇总和匿名 ROI 配置后运行：

```powershell
python scripts/run_resistance_disconnect_battery_sequence_v1.py `
  --action-summary outputs/qwen_experiment_action_hierarchical_v2/<run-id>/summary.json `
  --roi-config configs/resistance_disconnect_battery_rois.example.json `
  --video-root data/videos `
  --output-root outputs/resistance_disconnect_battery_sequence_v2_recovery_windows
```

只运行指定视频：

```powershell
python scripts/run_resistance_disconnect_battery_sequence_v1.py `
  --action-summary <summary.json> `
  --roi-config <roi-config.json> `
  --video-root <video-directory> `
  --video-ids 16 `
  --output-root outputs/rubric8_video16
```

使用保存的结构化观察做离线确定性重放：

```powershell
python scripts/run_resistance_disconnect_battery_sequence_v1.py `
  --action-summary <summary.json> `
  --roi-config <roi-config.json> `
  --video-root <video-directory> `
  --observations-root <saved-observations-directory> `
  --output-root outputs/rubric8_replay
```

## 7. 五视频结果

正式 recovery-window 运行结果：

| 视频 | 结果 | 置信度 | Episode 结论 |
|---:|---|---:|---|
| 8 | `pass` | 0.80 | `8_rewire_01: pass` |
| 16 | `pass` | 0.80 | `16_recovery_01: pass`；原 rewire episode 为 fail |
| 24 | `fail` | 0.25 | `24_rewire_01: fail` |
| 32 | `pass` | 0.80 | `32_rewire_01: pass` |
| 38 | `fail` | 0.25 | `38_rewire_01: fail` |

16 号 recovery episode 的完整证据链：

| 证据 | 相对视频时间 | 状态 |
|---|---:|---|
| 换线前开关 | 412.0 秒 | `open` |
| 换线前稳定端点 | 418.2 秒 | `T0-T2`，两节 |
| 换线开始 | 425.5 秒 | 直接端点操作 |
| 重接完成 | 428.7 秒 | `T2 -> T1` |
| 换线后开关 | 433.5 秒 | `closed` |
| 换线后稳定端点 | 436.5 秒 | `T0-T1`，一节 |

教师标准中视频 8 为 `0.5（存疑）`，其余四个样本具有明确二分类标签：

- 严格排除视频 8：`4/4 = 100%`；
- 将视频 8 映射为 `pass`：`5/5 = 100%`。

该准确率只描述当前五视频冻结评测，不代表新视频泛化准确率。

## 8. 输出结构

汇总文件：

```text
outputs/<run-id>/summary.json
```

每个 episode 保留：

- 核心和扩展时间窗；
- 输入帧及其相对时间；
- 电池盒对象确认；
- 换线前后端点状态；
- 直接端点接触帧；
- 开关断开、期间状态和重新闭合帧；
- 本地 reducer 的 reason code、置信度和诊断。

## 9. 约束

- 主结果固定为 `pass` 或 `fail`；
- 原始视频和教师评分文件只读；
- 教师答案、Excel 标签和最终分数不发送给 Qwen；
- 不跨 episode 合并证据；
- 不把普通 `circuit_rewiring` 自动视为换电池；
- 不因画面模糊直接停止，先使用邻帧、ROI 和密集候选；
- 运行时代码不包含姓名、固定视频时间或五视频专用判断。
