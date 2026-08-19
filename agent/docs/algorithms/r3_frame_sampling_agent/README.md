# R3 自适应抽帧 Agent

## 1. 定位

该模块是在原 R3 OpenCV 判定器外增加的独立抽帧 Agent。它不修改原
`opencv_switch_overlap.py` 和 `switch_rubric.py`，而是先以原算法的固定
`5 fps` 扫描作为基线，再根据当前 run 的视觉证据质量决定是否申请短时补帧。

当前实现保留独立 `standalone_experiment` 入口，同时已注册为正式
`switch.adaptive_frame_sampling` Skill。正式路径由 `run_switch_rubric` 生成
`resistance_agent_rubric_result.v2`，可以进入 `validate_run` 和 `finalize_run`。

主要文件：

- `resistance_agent/r3_frame_sampling_agent.py`：证据质量判断、补帧计划和融合；
- `resistance_agent/r3_frame_agent_adapter.py`：读取当前 run 阶段结果并生成 R3 实验工件；
- `resistance_agent/skills/executors.py`：注册正式 Skill、默认预算和参数校验；
- `resistance_agent/skills/router.py`：根据当前接线阶段选择 `window_mode`；
- `resistance_agent/toolkit.py`：写入正式 R3 工件和 Agent 状态；
- `run_r3_frame_agent.py`：命令行入口；
- `tests/test_r3_frame_sampling_agent.py`：预算、隔离、边界和融合测试。

配套的身份、时间、画质和盲测工具位于
[`../r3_stress_suite/README.md`](../r3_stress_suite/README.md)。

核心数据流：

```text
当前视频副本 + 当前 run 阶段
  -> Skill Router 选择接线窗口模式
  -> 固定 5 fps OpenCV 基线
  -> 证据质量诊断
  -> 最多两轮受限补帧
  -> 当前 run 统一阈值重算
  -> persistent closed AND wiring_active
  -> pass/fail + confidence + 证据帧
```

## 2. 输入与输出

输入只包含：

- 当前视频；
- 当前 run 内生成的阶段结果；
- 当前 run 本次 OpenCV 扫描产生的帧、动态 ROI 和状态观察。

输出主结果固定为 `pass` 或 `fail`，并保存：

- `frame_id`、数值 `image_group`、`image_group_id`、原始帧号和时间点；
- 原帧和动态开关 ROI 路径；
- 触发补帧的原因；
- 每轮请求、帧预算和停止原因；
- 共享阈值及融合诊断。

正式执行还会写入：

- `runs/<run>/rubrics/rubric_3.json`；
- `state.json.r3_frame_agent_report`；
- `state.json.rubric_evidence_reports["3"]`；
- Skill execution fingerprint 和完整实时路由审计字段。

## 3. 运行流程

1. 使用原 R3 OpenCV 算法在当前接线阶段按固定 `5 fps` 做基线扫描。
2. 评估当前证据的开关可见率、接线动作、闭合持续性、阈值边界和阶段边界活动。
3. 证据不清、临界或冲突时，在对应接线阶段内生成短窗口补帧请求。
4. 补帧使用独立的 `5 fps` 相位偏移采样，默认相位偏移为 `0.1 s`。
5. 将基线和补帧产生的原始观察放回同一个当前 run 共享阈值下重新融合。
6. 达到明确反例、证据充分、无新帧、预算耗尽或最大轮数时停止。

局部补帧窗口自己的 `fail` 不会直接覆盖最终结果。最终 `fail` 必须由所有当前
run 观察在同一共享阈值下确认“接线动作发生时，开关闭合状态持续存在”。

## 4. 补帧触发条件

Agent 当前支持以下统一触发原因：

| 触发原因 | 含义 | 请求类型 |
|---|---|---|
| `no_switch_observation` | 当前扫描没有可靠开关观察 | `seek_clearer_frame` |
| `low_switch_coverage` | 接线阶段开关覆盖率过低 | `seek_clearer_frame` |
| `no_wiring_activity_observed` | 未观察到接线活动 | `neighbor_burst` |
| `switch_not_visible_during_wiring` | 接线动作帧中开关不可见 | `neighbor_burst` |
| `closed_persistence_boundary` | 闭合持续帧数接近判定边界 | `neighbor_burst` |
| `state_threshold_margin` | 状态分数接近共享阈值 | `neighbor_burst` |
| `stage_edge_activity` | 接线活动靠近阶段边缘 | `expand_within_stage` |
| `stage_boundary_ambiguity` | 第二轮仍需核对阶段边界 | `expand_within_stage` |

请求按“触发原因多样性、优先级、时间覆盖”排序。长动作片段先取中点，再向片段
两端扩展，避免同一种低质量原因只覆盖视频前段或占满预算。

## 5. 默认预算与边界

| 参数 | 默认值 |
|---|---:|
| 基线采样率 | `5 fps` |
| 补帧采样率 | `5 fps` |
| 补帧相位偏移 | `0.1 s` |
| 最大轮数 | `2` |
| 每轮最大请求数 | `3` |
| 最大新增补帧数 | `64` |

所有 `candidate_window` 都必须位于父级 `circuit_wiring` 或
`circuit_rewiring` 时间窗内。阶段外帧不会参与 R3 判分，报告固定记录
`outside_stage_frames_scored=false`。不同补帧请求之间不会重复解码；在非整倍帧率
视频中，补帧短窗可以重新包含少量基线帧作为连续状态上下文，但新增帧预算只统计
基线中没有的帧。

## 6. 反过拟合约束

正式运行遵守以下约束：

- 不按 `video_id`、文件名、学生姓名或 SHA-256 选择算法、时间窗、ROI、阈值或结论；
- 不读取历史时间窗、固定 ROI、旧预测、人工复核、Excel 或历史最佳工件；
- 阶段文件必须位于当前 run 内，且只能解析唯一的当前阶段记录；
- ROI 由原 R3 判定器从当前帧动态定位；
- 相同视觉情况使用相同触发规则、预算和融合方式；
- 证据质量只影响补帧和置信度，最终结果仍为二分类。

审计字段固定包括：

```json
{
  "selection_basis": "current_video_observed_situation_only",
  "video_id_used_for_routing": false,
  "historical_artifacts_used": false,
  "fixed_video_roi_used": false
}
```

## 7. 命令

正式 execute 使用已有工具链：

```text
plan_live_skills
-> run_rubric_bundle(rubric_ids=[3])
-> validate_run
-> finalize_run
```

Router 对所有视频统一选择 `switch.adaptive_frame_sampling`，只根据当前 run 是否观察到
接线、重接线来设置 `all_wiring_runs`、`initial_wiring_only` 或 `broad_search`。
`max_rounds`、`max_requests_per_round` 和 `max_supplemental_frames` 会实际传给 Agent。

独立实验入口仍可在 `agent` 目录运行：

```powershell
python run_r3_frame_agent.py `
  --video "runs/<run>/input_video/current.mp4" `
  --stage-summary "runs/<run>/boundary_refinement/rubric_boundaries/summary.json" `
  --output-dir "outputs/r3_frame_agent_current_run" `
  --association-id "current-run-association"
```

定向测试：

```powershell
python -m unittest tests.test_r3_frame_sampling_agent -v
```

## 8. 当前限制

- Agent 仍复用原 R3 动态 ROI 和状态特征，因此补帧能增加时间证据，不能自动修复所有 ROI 定位错误；
- 开发视频上的结果只能视为开发集回归，泛化效果需要冻结实现后用新视频盲测；
- 独立 CLI 生成的 `standalone_experiment` 工件仍不冒充正式状态机结果；只有 Toolkit 正式路径写入 run 状态和 `rubric_3.json`。

## 9. 判定语义与停止条件

R3 的评分语义是“接线或改线时开关应保持断开”。OpenCV 基线一旦已经找到共享阈值下
成立的持续闭合与接线动作同帧反例，Agent 立即输出 `fail`，停止原因是
`baseline_counterexample_confirmed`，不会为了增加帧数而稀释明确反例。因此视频 16、24
的补帧请求数为 0 是预期行为，不表示 Agent 没有执行。

基线为 `pass` 时，Agent 才根据证据质量决定是否补帧。所有停止原因如下：

| `stop_reason` | 含义 |
|---|---|
| `baseline_counterexample_confirmed` | 基线已经找到明确反例 |
| `shared_threshold_counterexample_confirmed` | 补帧融合后找到明确反例 |
| `evidence_sufficient_or_no_new_frames` | 证据足够或规划器没有可申请的新帧 |
| `supplemental_frame_budget_exhausted` | 新增帧达到预算上限 |
| `no_new_frames` | 本轮请求没有解码出新物理帧 |
| `maximum_rounds_reached` | 完成两轮后仍无明确反例 |

最终 `pass` 不等于“每一帧都清晰”，而是“在统一预算内没有发现满足严格同帧条件的
反例”。若最终仍存在低覆盖、遮挡或阈值边界问题，结果保持二分类，但置信度会限制为
不高于 `0.55`，并使用
`no_counterexample_after_bounded_sampling_with_low_evidence_quality` 说明原因。

## 10. 当前 run 共享融合

每个补帧短窗都会独立运行原 OpenCV 分析器，但短窗自己的局部 `decision` 不进入最终投票。
Agent 汇总所有基线和补帧的原始 `bridge_score`、开关观察、插头转换和接线活动帧，然后：

1. 对当前 run 全部开关观察重新计算一次二聚类阈值；
2. 使用相同阈值重新平滑开闭状态；
3. 重新标记连续闭合支持，至少 3 个时间连续观察才算 persistent closed；
4. 把接线转换扩展到真实支持帧；
5. 只在同一个物理帧同时满足 persistent closed 和 wiring active 时生成反例。

报告中的 `shared_threshold_fusion` 保存最终阈值、簇中心和阈值来源。
`cross_scan_persistence_fusion=false` 表示不会把两个互不连续短窗中的零散 closed 观察拼成
一段持续闭合；补帧增加证据覆盖，但不能制造跨窗口的虚假持续性。

## 11. 正式状态机与工件

正式执行顺序为：

```text
inspect_video
-> create_run
-> run_full_pipeline
-> refine_rubric_boundaries
-> plan_live_skills
-> run_switch_rubric
-> validate_run / freeze
```

`inspect_video` 既接受配置目录中的唯一 ID/文件名，也接受明确存在的本地视频路径；路径只用于
定位当前输入。`create_run` 随后复制视频并核对内容，后续 R3 只读取 run 内副本。

核心输出包括：

| 路径或字段 | 内容 |
|---|---|
| `baseline_5fps/opencv_switch_overlap_report.json` | 固定 5 fps 原始报告 |
| `plans/round_<n>.json` | 每轮证据质量、锚点和请求预算 |
| `supplemental/<request_id>/` | 独立补帧扫描与动态 ROI |
| `evidence_frames/` | 最终使用的原帧副本 |
| `r3_frame_sampling_agent_report.json` | 基线、补帧、融合和审计总报告 |
| `rubrics/rubric_3.json` | 正式二分类评分结果 |
| `frozen_r3/predictions_frozen.json` | 盲测冻结结果 |

每个证据帧保存 `image_group`、`frame_id`、`image_group_id`、真实帧号、时间点、阶段、
基线/补帧来源、请求 ID、触发原因、开关状态、接线活动和 ROI 路径。这样可以从最终结论
回溯到同一时间点的原帧和局部候选，而不需要依赖视频 ID 找历史图片。

## 12. 2026-08-18 六视频稳定性运行

本轮使用同一个 `r3_frame_sampling_agent.v1`、同一默认预算和当前 run 阶段，统一处理
视频 8、16、24、32、38，以及不同实验类型的视频 39。六个视频均未读取 Excel参与判断。

| 视频 | 基线帧 | 最终帧 | 开关观察 基线→最终 | 覆盖率 基线→最终 | 请求 / 新帧 | 判定 | 工件运行跨度 |
|---|---:|---:|---:|---:|---:|---|---:|
| 8 | 592 | 646 | 334→378 | 56.42%→58.51% | 6 / 54 | pass | 254.123s |
| 16 | 813 | 813 | 213→213 | 26.20%→26.20% | 0 / 0 | fail | 518.871s |
| 24 | 762 | 762 | 77→77 | 10.10%→10.10% | 0 / 0 | fail | 613.020s |
| 32 | 542 | 585 | 120→149 | 22.14%→25.47% | 6 / 43 | pass | 402.656s |
| 38 | 732 | 786 | 190→221 | 25.96%→28.12% | 6 / 54 | pass | 470.281s |
| 39 | 112 | 165 | 35→80 | 31.25%→48.48% | 6 / 53 | pass | 110.091s |

合计：基线 `3553` 帧、最终 `3757` 帧、新增 `204` 帧、`24` 个请求。六条 Agent
判定均与各自固定 5 fps 基线一致，`decision_change_count=0`。运行跨度来自 Agent 输出目录
首末工件修改时间，字段为 `runtime_seconds`、来源为
`agent_output_artifact_mtime_span`；它用于比较执行开销，不是精确的模型计费时间。

汇总文件：

- `outputs/r3_stability_summary_20260818_all6/summary.json`；
- `outputs/r3_stability_summary_20260818_all6/summary.csv`；
- `outputs/r3_stability_summary_20260818_all6/stress_overview.json`。

这些结果只能说明当前六个输入上的稳定性。视频 8、16、24、32、38 已参与算法开发；
视频 39 属于不同实验类型，其 Excel 第 3 项语义与本 Agent 的 R3 契约不同。因此本表不报告
新视频准确率，也不把 6 条稳定结果写成泛化准确率。

## 13. 压力测试结果

### 13.1 身份无关性

视频 39 被复制为随机匿名文件名，并同步重绑定当前阶段摘要。原文件与匿名副本得到完全相同的
候选窗口、参数、判定、基线/最终帧数、开关观察数和请求数，`metric_differences={}`、
`passed=true`。这证明当前测试范围内没有文件名或 `video_id` 路由。

### 13.2 时间鲁棒性

- 采样相位 `-0.1s`、`+0.1s`：均为 `pass`，覆盖率均为 `41.82%`；
- 接线边界 `-2s`、`+2s`：均为 `pass`，覆盖率为 `54.61%`、`68.06%`；
- 接线边界 `-5s`、`+5s`：均为 `pass`，覆盖率为 `41.41%`、`62.50%`；
- 所有时间测试的 `decision_flip_count=0`。

覆盖率变化说明分段误差会改变证据量，但当前测试没有改变最终二分类。5 fps 周期为
`0.2s`，所以 `-0.1s` 与 `+0.1s` 在窗口内部归一化为同一个采样相位。

### 13.3 画质压力

视频 39 的 720p 副本为 `1280x720`，原视频保持不变。变体运行结果为 `pass`，基线
`112` 帧、最终 `159` 帧、6 个请求，真实运行约 `128.408s`。这只是一项画质冒烟测试，
不能代表全部低清、模糊、过曝或重压缩条件。

## 14. 一键盲测与回归验证

新视频一键入口：

```powershell
python scripts/run_r3_blind_execute.py `
  --video-ref "<new-video-path-or-id>" `
  --run-id "blind_r3_<timestamp>" `
  --config "config.json"
```

命令不接受 Excel 路径，执行完成后先写
`runs/<run>/frozen_r3/predictions_frozen.json`，再允许单独进行离线评测。视频 39 的真实入口
验证耗时 `315.2s`，结果为 `pass`、置信度 `0.55`，并确认：

```json
{
  "status": "r3_prediction_frozen",
  "video_id_used_for_routing": false,
  "historical_artifacts_used": false,
  "fixed_video_roi_used": false,
  "excel_read": false,
  "ground_truth_read": false,
  "source_video_unchanged": true
}
```

本轮专项测试 `34/34` 通过；提交时仅保留可在公开仓库复现的 Agent 测试。
路由审计没有在正式 R3 execute 链中发现视频专属算法分支、历史回退或固定视频 ROI；源码
密钥扫描没有发现明文 `sk-*` 密钥。完整验证记录位于：

- `outputs/r3_stability_summary_20260818_all6/routing_security_audit.json`；
- `deliveries/r3_longrun_20260818/VERIFICATION.txt`。
