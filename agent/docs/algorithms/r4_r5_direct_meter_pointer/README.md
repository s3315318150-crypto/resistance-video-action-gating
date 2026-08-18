# R4 复用 R5 电表指针证据

## 1. 算法定位

| 字段 | 值 |
|---|---|
| Algorithm ID | `r4_meter_polarity_v21_r5_direct_meter_pointer` |
| Rubric | R4：电表正负接线柱正确 |
| 主结果 | `pass` / `fail` |
| R5 证据版本 | `r56_temporal_meter_v5_live_closed_stable_cv_v3` |
| Agent 融合策略 | `current_run_r5_direct_meter_pointer` |
| R4 reducer | `resistance_agent/polarity_rubric.py` |
| Agent 工具封装 | `resistance_agent/toolkit.py` |
| Skill 注册 | `resistance_agent/skills/executors.py` |

R4 v21 不再通过导线颜色或跨画面端点追踪判断极性。它直接消费当前 run 的 R5 电表指针结果：

```text
当前视频 Temporal Guard / Rubric 边界
  -> R5/R6 共享抽帧与动态电表 ROI
  -> Qwen 观察通电状态、A/V 身份和指针方向
  -> R5 本地多帧 reducer
  -> 当前 run 的 rubric_5.json
  -> R4 校验来源并直接生成 rubric_4.json
```

R4 不会再次解码视频，不会再次调用 Qwen，也不会重新生成 ROI。

## 2. 修改原因

R4 v20 已能通过金黄色凸起端和绿色宽平端识别电源正负极，但后续仍需要把电表接线柱沿导线追踪到电源两侧。困难视频中常见：

- 导线交叉；
- 手部和器材遮挡；
- 导线离开画面；
- 全景能看到电源，但看不清表盘；
- ROI 能看到表盘，但看不到完整导线；
- 不同时间帧中的局部证据被错误拼成同一拓扑。

五视频 v20 回归只有 `2/5 = 40%`。主要错项是视频 16、24 被错误触发 `ammeter:reversed`，视频 8 又因指针融合覆盖端点结果而错误输出 `pass`。

R5 本身已经直接观察通电测量时的表盘状态，并在五个开发视频中输出：

```text
8  -> fail
16 -> pass
24 -> pass
32 -> pass
38 -> fail
```

因此 v21 采用“一次取证、多项消费”：R5 负责真实视频取证，R4 只复用同一份当前 run 证据。

## 3. R5 取证内容

R5 的输入来自当前视频本次执行：

1. 根据当前 Temporal Guard 和边界结果选择测量窗口；
2. 从测量窗口和记录前邻域提取候选帧；
3. 动态定位电流表和电压表，不使用按视频编号保存的坐标；
4. 保存原帧、表盘 ROI、宽 ROI、增强 ROI 和端子 ROI；
5. 让 Qwen 逐图组输出可见观察；
6. 本地 reducer 对多帧结果进行归并。

主要观察字段：

```text
image_group
circuit_state: energized / deenergized / unclear
identity: ammeter / voltmeter / unknown
pointer_state: normal_rightward / zero / reverse / overrange / uncertain
pointer_scale_position: near_zero / low / mid / high / near_full / uncertain
confidence
evidence
```

断电图组中的零位不作为异常测量。稳定异常需要两个不同图组支持，或者单个图组达到高置信度门槛。

## 4. R5 到 R4 的映射

当前映射是明确的二分类复用：

| R5 | R4 | 说明 |
|---|---|---|
| `pass` | `pass` | 当前测量证据支持正常指针状态 |
| `fail` | `fail` | 当前测量证据未通过 R5 指针规则 |

R4 继承：

- `decision`；
- `predicted_score`；
- `confidence`；
- R5 原始 `reason`；
- A/V `needle_states`；
- `identity_observations`；
- 证据时间点；
- 原帧和 ROI 路径。

R4 的 `reason` 增加统一前缀：

```text
current_run_r5_direct_meter_pointer:<R5 reason>
```

诊断字段固定声明：

```json
{
  "decision_basis": "current_run_r5_direct_meter_pointer",
  "source_rubric_id": 5,
  "endpoint_topology_used": false,
  "wire_color_used": false,
  "historical_artifacts_used": false
}
```

## 5. 当前 run 依赖校验

正式 execute 不接受任意位置的 R5 文件。`toolkit.run_polarity_rubric` 会依次检查：

1. `rubric_5.json` 必须位于当前 run 的 `rubrics` 目录；
2. 文件名必须是 `rubric_5.json`；
3. schema 必须是 `resistance_agent_rubric_result.v2`；
4. `rubric_id` 必须是 `5`；
5. `video_id` 和 `source_video_id` 必须与当前 run 一致；
6. `decision` 与 `predicted_score` 必须一致；
7. `execution_mode` 必须是 `execute_visual_evidence`；
8. `routing_policy` 必须是 `live_situation_skills.v1`；
9. `source_artifact` 必须指向同一 run 的 `meter_rubrics/meter_evidence_report.json`；
10. 证据报告必须声明未读取 Excel、未发送真值、未使用历史回退。

R4 保存 R5 文件 SHA-256。已有 R4 的算法版本或 R5 哈希不一致时，旧 R4 结果不会被复用。

如果单独调用 R4 且当前 run 尚未生成 R5，toolkit 会先执行 `run_meter_rubrics`。如果绕过 toolkit 直接调用 polarity 模块而不提供当前 run R5，模块立即报错：

```text
current-run R5 result is required for R4 execute
```

## 6. Agent 执行顺序

完整十项 bundle 中，R5/R6 本来就在 R4 之前：

```text
run_switch_rubric       -> R3
run_series_rubric       -> R1
run_meter_rubrics       -> R5 / R6
run_remaining_rubrics   -> R0 / R2 / R8
run_polarity_rubric     -> R4，复用当前 run R5
```

因此 R4 的额外成本只包括 JSON 校验、哈希计算和结果写入，不增加抽帧数、图片数或 Qwen 请求数。

## 7. 反过拟合约束

正式路径满足以下约束：

- 不根据 `video_id`、文件名、学生姓名或 SHA-256 选择算法；
- 不读取该视频以前保存的 R4/R5 预测；
- 不读取历史时间窗或固定 ROI；
- 不读取 Excel 或人工真值；
- R5 ROI 来自当前视频当前帧的动态定位；
- R4 只接受同一 run 的 R5 结果与证据报告；
- 仅改变 `video_id` 和文件名时，R4 reducer 的输出逐字段保持一致。

`video_id` 和哈希只用于文件关联、完整性验证与缓存失效，不参与 `pass/fail` 选择。

## 8. 五视频开发集回归

本次回归先读取已经冻结的 R5 证据生成 R4 预测，写入 `frozen_predictions.json`；预测冻结后才读取 `实验标准.xlsx`。

| 视频 | R4 v21 | 置信度 | Excel R4 | 正确 |
|---|---|---:|---|---|
| 8 | `fail` | 0.585 | `fail` | 是 |
| 16 | `pass` | 0.900 | `pass` | 是 |
| 24 | `pass` | 1.000 | `pass` | 是 |
| 32 | `pass` | 1.000 | `pass` | 是 |
| 38 | `fail` | 1.000 | `fail` | 是 |

指标：

```text
正确数       5/5
准确率       100%
pass recall  100%
fail recall  100%
```

回归文件：

```text
outputs/r4_r5_direct_v21_replay_20260817/frozen_predictions.json
outputs/r4_r5_direct_v21_replay_20260817/evaluation.json
```

冻结预测 SHA-256：

```text
88F88DA9BD997245350EC1BAA440B4C713BF8EDD2EE8794597CB4446AFABCE51
```

评测文件 SHA-256：

```text
9BC92B6605945A939BA22F855139EAA8C0AC8CA9C60DAA2450EF750C7EDA2D8F
```

这组结果属于五个开发视频的 `replay/regression`，不能作为新视频泛化准确率。

## 9. 输出示例

```json
{
  "rubric_id": 4,
  "decision": "pass",
  "predicted_score": 1,
  "confidence": 0.9,
  "reason": "current_run_r5_direct_meter_pointer:visible_measurement_window_has_normal_pointer_deflection;other_meter_low_visibility",
  "diagnostics": {
    "algorithm_version": "r4_meter_polarity_v21_r5_direct_meter_pointer",
    "decision_basis": "current_run_r5_direct_meter_pointer",
    "source_rubric_id": 5,
    "r5_decision": "pass",
    "endpoint_topology_used": false,
    "wire_color_used": false,
    "historical_artifacts_used": false
  }
}
```

## 10. 测试

在 Agent 目录运行：

```powershell
cd <repository-root>
python -m unittest tests.test_toolkit
```

当前结果：

```text
Ran 159 tests
OK
```

测试覆盖：

- R5 `pass/fail` 到 R4 的确定性映射；
- R4 不使用端点拓扑和导线颜色；
- 改变视频身份字段不会改变 reducer 输出；
- polarity 模块存在 R5 时不解码视频；
- polarity 模块缺少 R5 时拒绝执行；
- toolkit 将 R5 路径传入 R4；
- R5 文件哈希控制 R4 幂等复用；
- 正式 Agent 状态机仍能写入 `rubric_4.json`。

## 11. 已知限制

R4 和 R5 现在不是两个独立视觉判断，它们共享同一个二分类结论。这会带来明确的语义风险：

- R5 因“没有找到正常偏转”而 `fail`，不一定证明正负接线柱接反；
- R5 因始终为零、超量程或电路未可靠通电而 `fail`，原因可能不是极性；
- 五个开发视频中 R4 与 R5 标签恰好一致，不代表其他学生视频也一定一致；
- 当前 100% 只能说明开发集回归恢复，不证明泛化能力。

因此下一步应冻结 v21 后，用未参与开发的新视频执行完整 R5 -> R4 流程。评测时至少分别报告：

- R4 总体准确率；
- `pass` 召回率；
- `fail` 召回率；
- R5 fail 原因分布；
- R4 与 R5 标签不一致的样本；
- 低置信度和单表可见样本数量。
