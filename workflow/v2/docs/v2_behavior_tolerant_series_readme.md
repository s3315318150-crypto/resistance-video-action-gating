# v2 行为容错系列

本系列从 `v2-temporal-guard` 独立派生。四个版本逐项累积，但均有独立入口、算法 ID 和输出目录；原 v2、Temporal Guard 和 v3 的源码与行为不变。

## 版本

| 一键选项 | 独立入口 | 新增能力 |
|---|---|---|
| `v2-behavior-tolerant` | `qwen_experiment_action_hierarchical_v2_behavior_tolerant.py` | 带惩罚状态图、纠错微循环、集中书写 |
| `v2-behavior-tolerant-aux` | `qwen_experiment_action_hierarchical_v2_behavior_tolerant_aux.py` | 辅助动作及冲突隔离 |
| `v2-behavior-tolerant-boundary` | `qwen_experiment_action_hierarchical_v2_behavior_tolerant_boundary.py` | 跨 Map 窗口边缘复核 |
| `v2-behavior-tolerant-adaptive` | `qwen_experiment_action_hierarchical_v2_behavior_tolerant_adaptive.py` | 保留基础帧的活动补帧 |

## 1. 行为容错状态图

`qwen_hierarchical_v2_behavior_tolerant_reduce.py` 使用宽度为 6 的确定性 beam search。它只解释 Reduce 已接受的事件，不改变 Map 时间。

- 第一次测量后的接线同时形成 `correction_loop` 与 `second_cycle` 两条假设。
- 测量时长通过 12 秒范围内的连续短促分数参与判断，不使用 `<5 秒`硬门槛。
- “接触不良、插紧、纠正”等证据提高纠错得分；“换接、改接、另一端、重新配置”等证据提高正式第二轮得分。
- 后续 30 秒内是否再次测量或书写作为通用前瞻特征。
- `recording_2 -> measurement_2 -> recording_2` 可以发生，但回退转移记录 `-0.25` 惩罚。
- 两轮测量后只出现一次连续书写时，该事件保留为一个区间，并输出 `batched_recording=true` 与两个 `recording_search_aliases`。
- 明确重新连线后直接书写仍兼容 v2 的 `recording_2`，但会记录 `inferred_stage=true`、`measurement_2_observed=false`，不把未观察到的测量伪装成视觉证据。

相同分数按直接阶段事件数、推断事件数和事件 ID 稳定排序，重复运行不会随机选择不同路径。

## 2. 辅助动作

Map 可以额外输出 `auxiliary_action`，但七阶段列表没有变化。

| subtype | 含义 | 对状态图的作用 |
|---|---|---|
| `battery_configuration_change` | 电池盒端子或接入配置直接发生变化 | 随后 20 秒提高正式重新连线得分 |
| `teacher_intervention` | 老师进入并指导 | 仅诊断 |
| `seat_change` | 换座位或换人 | 只能强化已存在的整理候选 |
| `conversation` | 直接可见的闲聊 | 只能强化已存在的整理候选 |
| `phone_use` | 使用手机 | 仅诊断 |
| `off_task_behavior` | 其他离题行为 | 仅诊断 |
| `other_action` | 其他七阶段外可见行为 | 仅诊断 |

辅助事件不会与同时发生的接线、测量、书写竞争，也不会因 Reduce 文本拒绝而静默消失。最终结果写入 `auxiliary_events`；换座位或闲聊本身不能生成 `material_cleanup`。

## 3. 窗口边缘复核

基础 Map 仍使用 60 秒窗口和 10 秒重叠。只有动作触及窗口首尾，或相邻窗口报告同类连续动作时，才创建边缘候选。

1. 在窗口边缘 `T` 建立 `T-10s` 至 `T+10s` 的辅助范围。
2. 第一遍按 1 秒抽帧。
3. 第一遍无效或回答 `uncertain` 时，第二遍按 0.5 秒抽帧。
4. 模型只能确认候选的原动作类型；本地契约拒绝改成其他动作。
5. 新事件只用于延伸或合并同类动作，原 Map 事件始终保留。

每个候选的提示词、帧、模型原文和采用状态写入 `map/boundary_bridges/`，汇总写入 `boundary_bridge_reviews`。

## 4. 补充式动态采样

该版本不使用 v3 的替换式 TCS。原 `--sample-interval-seconds` 产生的全部基础帧都会进入模型请求。

- 全锁定区间按 0.5 秒低清扫描一次。
- 使用相位相关估计全局平移，将前帧对齐后再计算帧差，降低摄像机晃动影响。
- 活动分数由补偿帧差和 HSV 直方图变化组成，并在当前窗口按 P20/P90 归一化。
- 每 10 秒时间桶最多补一帧。
- 每个窗口补帧预算为基础帧数的 `floor(25%)`。
- 补帧与任意基础帧至少相隔 0.5 秒。

`result.json` 的 `sampling.dynamic_supplement` 保存基础帧、补充帧、时间桶、预算和全局运动补偿方法。

## 运行

一键运行最终累积版：

```powershell
python scripts/run_resistance_pipeline.py `
  --video-dir data/videos `
  --output-root outputs/resistance_pipeline `
  --action-version v2-behavior-tolerant-adaptive
```

只运行动作分割：

```powershell
python scripts/qwen_experiment_action_hierarchical_v2_behavior_tolerant_adaptive.py `
  --segment-source outputs/experiment_boundary/summary.json `
  --schema configs/action_schemas/resistance_7stage_no_battery_v2_behavior_tolerant_aux.json `
  --output-root outputs/qwen_experiment_action_hierarchical_v2_behavior_tolerant_adaptive `
  --sample-interval-seconds 2 `
  --max-model-edge 640
```

## 五视频离线 A/B

黄金基准位于 `tests/fixtures/v2_behavior_tolerant_golden.json`，只供测试和离线评测使用，运行时代码不会读取。验收条件为阶段数量与顺序一致，起止边界误差不超过 2 秒。

| 版本 | 通过数 | 阶段差异 | 超差边界 |
|---|---:|---:|---:|
| `v2-behavior-tolerant` | 5/5 | 0 | 0 |
| `v2-behavior-tolerant-aux` | 5/5 | 0 | 0 |
| `v2-behavior-tolerant-boundary` | 5/5 | 0 | 0 |
| `v2-behavior-tolerant-adaptive` | 5/5 | 0 | 0 |

复现命令：

```powershell
python scripts/compare_v2_behavior_tolerant.py `
  --replay-root <保存的-v2-temporal-guard-回放目录> `
  --variant v2-behavior-tolerant-adaptive `
  --output outputs/v2_behavior_tolerant_ab/v2-behavior-tolerant-adaptive
```

该比较使用已保存的 accepted events，验证 Reduce 后的确定性阶段解释。辅助动作、边缘复核和动态采样对新 Map 观察的影响仍需要网关可用时做完整在线 A/B。

## 当前验证

- 全部单元测试：179 项通过。
- 四个一键选项 dry-run：通过。
- 五视频离线黄金回放：四版均 5/5。
- 2026-08-12 在线探测：本地 adaptive 预扫描成功，所有基础帧均保留；Qwen Map 请求返回 `APIConnectionError: Connection error.`，因此未生成可比较的在线阶段结果。

项目没有加入姓名、视频 ID、固定秒数或证据图片专用分支。原始视频、抽帧图片和在线输出不进入仓库。
