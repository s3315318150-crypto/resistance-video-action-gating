# R1 动态拓扑与遮挡取证算法

## 1. 算法定位

| 字段 | 当前值 |
|---|---|
| Rubric | R1：电流表是否串联接入电路 |
| Algorithm version | `r1_occlusion_aware_dynamic_topology_v8_activity_context` |
| 主结果 | `pass` / `fail` |
| 主要执行器 | `series.adaptive_terminal_sampling` |
| 主实现 | `resistance_agent/series_rubric.py` |
| Skill 注册 | `resistance_agent/skills/executors.py` |
| 路由 | `resistance_agent/skills/router.py`、`toolkit.py` |

该版本是 Agent 正式 execute 路径中的 R1 实现。它从当前 run 重新生成阶段、候选帧、动态 ROI 和模型观察，不读取某个视频以前的时间窗、ROI 或预测结果。

## 2. 判定目标

R1 的物理目标是确认电流表位于主回路中，而不是直接跨接电源或悬空。当前判定规则为：

1. 接线或改线阶段确认电流表直接跨接电源，结果永久为 `fail`。
2. 测量/观测阶段确认电流表导线非串联或端点悬空，结果永久为 `fail`。
3. 仅在接线动作中的临时插拔、移动端子或短暂悬空，不直接判错。
4. 观察阶段的恢复帧只有在 `activity_context=measurement_action` 时才参与观察阶段违规判定。
5. 没有确认反例时仍输出二分类结果；证据质量只影响 `confidence` 和诊断字段。

## 3. 执行流程

正式 execute 的调用顺序是：

```text
inspect_video
  -> create_run
  -> run_full_pipeline
  -> refine_rubric_boundaries
  -> plan_live_skills
  -> run_series_rubric
  -> validate_run
  -> finalize_run
```

R1 执行器内部流程：

```text
当前 run 阶段
  -> 生成 circuit_wiring / circuit_rewiring 候选窗
  -> 当前视频 2 FPS 粗扫
  -> 阶段边界 5 FPS 密集取证
  -> 动态定位电流表、端子和导线 ROI
  -> Qwen 全景/ROI 结构化观察
  -> 同一 run 内的拓扑与时间融合
  -> pass/fail + confidence + 证据工件
```

分段阶段由当前视频重新识别 `wiring_action`、`measurement_action`、`writing_action` 和 `cleanup_action`，再生成七阶段记录。R1 只使用与接线、改线和相关测量动作相邻的当前 run 时间窗。

## 4. 抽帧和 ROI

默认参数由 Skill 传入执行器，并对相同视觉情况保持一致：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `sampling_interval_seconds` | `0.5` | 粗扫时间间隔，相当于 2 FPS |
| `dense_sampling_fps` | `2.0` | 阶段边界密集扫描 |
| `transition_sampling_fps` | `5.0` | 接线动作转场附近的补充帧 |
| `max_samples_per_window` | `36` | 单个粗扫窗的候选上限 |
| `stable_frames_per_stage_run` | `2` | 每个阶段的稳定帧数量 |
| `view_recovery_frames_per_stage_run` | `1` | 视角恢复帧数量 |
| `max_transition_anchors` | `4` | 转场锚点上限 |
| `transition_radius_seconds` | `1.0` | 锚点前后扩展范围 |
| `max_supplemental_rounds` | `1` | 冲突复核轮数 |
| `max_supplemental_frames` | `12` | 补充复核帧上限 |
| `roi_target_long_edge` | `1400` | ROI 目标长边 |

ROI 优先从当前帧中的电流表标识、彩色端子面板、端子结构和导线候选动态生成。系统同时保存原始帧、增强帧、原生 ROI 和放大 ROI，便于回溯同一时刻的证据。

## 5. Qwen 观察格式

每个图片组使用当前 run 内的 `image_group` 和 `frame_binding` 绑定。模型观察主要包括：

- `activity_context`：`wiring_action`、`measurement_action`、`writing_action` 或 `unclear`；
- `path_relation`：导线端点之间的关系；
- `direct_across_state`：是否确认直接跨接电源；
- `final_topology`：当前可见拓扑；
- `terminal_evidence`：设备、端子、远端点、连接状态和中间元件；
- `loose_lead_endpoints`：悬空端点；
- `direct_observations` 与 `derived_observations`：直接观察和推导观察分开保存；
- `confidence`：诊断置信度，不是第三类结果。

模型返回的 `image_group` 必须属于本次请求，随后由本地代码绑定到真实帧号、时间点和 ROI 路径。

## 6. 融合逻辑

融合器按以下顺序处理当前 run 的观察：

1. 合并粗扫、转场密集帧和补充复核帧。
2. 过滤不属于当前评分阶段的观察。
3. 优先处理接线阶段确认的直接跨电源证据。
4. 再检查测量阶段的非串联拓扑和悬空电流表导线。
5. 对相邻时间点进行聚类，避免把单个模糊帧直接扩展成连续事件。
6. 输出唯一的 `pass` 或 `fail`，并记录决定性帧和时间点。

当前版本中的关键判定策略是单调的：一旦当前 run 发现规则明确的 R1 违规，后续正常帧不能把它改回 `pass`。

## 7. 输出工件

每个 run 主要输出：

```text
runs/<run>/
  state.json
  skills/live_skill_plan.json
  series_rubric/series_evidence_report.json
  rubrics/rubric_1.json
  series_rubric/frame_agent/
    selected/
    transition_bursts/
    initial_qwen/
    supplemental_qwen/
```

`rubric_1.json` 至少包含：

- `decision`、`predicted_score`、`confidence`；
- `reason`、`decision_branch`、`path_relation`；
- `supporting_frame_ids`、`supporting_timestamps_seconds`；
- `direct_observations`、`derived_observations`；
- `diagnostics` 中的稳定通过、直接跨接和观测阶段违规统计；
- 当前 run 的路由审计字段。

## 8. 反过拟合约束

正式 execute 遵守以下约束：

- `video_id`、文件名和学生姓名只用于输入/输出关联，不选择算法、时间窗、ROI、阈值或结论；
- 不读取该视频以前保存的时间窗、固定 ROI、预测、人工复核或 Excel 真值；
- 每次从当前视频重新识别阶段并生成候选帧；
- ROI 必须来自当前帧动态定位；
- 相同视觉情况选择相同 Skill 和参数；
- 阶段缺失时使用统一 broad search；
- Excel 只在预测冻结后离线比较；
- 主结果固定为 `pass` 或 `fail`。

审计字段应保持：

```json
{
  "selection_basis": "current_video_observed_situation_only",
  "video_id_used_for_routing": false,
  "historical_artifacts_used": false,
  "fixed_video_roi_used": false
}
```

## 9. 五视频开发集回归

本次 R1 v8 批次在预测冻结后才读取 Excel，结果如下：

| 视频 | Agent 结果 | Excel | 置信度 | 是否正确 |
|---|---|---|---:|---|
| 8 | `fail` | `fail` | 0.94 | 是 |
| 16 | `pass` | `pass` | 0.55 | 是 |
| 24 | `pass` | `pass` | 0.88 | 是 |
| 32 | `fail` | `pass` | 0.94 | 否 |
| 38 | `pass` | `fail` | 0.88 | 否 |

结果：`3/5 = 60%`，平衡准确率 `58.3%`。这只是开发集回归，不能作为新视频泛化准确率。

冻结结果：

```text
agent/runs/<run-id>/rubric_1.json
```

## 10. 已确认问题

### 10.1 视频 32：同一观察内部字段冲突

在 `272.1s` 的观察中，模型同时写出：

- `direct_across_state=confirmed`；
- `final_topology=single_series_loop`；
- 直接观察为“所有核心设备连接成闭合回路”。

当前融合优先采信了 `direct_across_state=confirmed`，造成 false fail。下一版需要增加同一观察的一致性约束：直接跨接结论必须与端子配对证据一致，不能只依赖单个枚举字段。

### 10.2 视频 38：不可见电流表被当成完整串联

决定性帧 `370.5s` 的直接观察写明“电流表不在可见回路中”，但 `final_topology=single_series_loop` 仍被当成通过证据，造成 false pass。下一版需要要求通过证据明确包含电流表两端和主回路连接，不能用缺少关键器件的局部回路替代完整串联证据。

### 10.3 当前性能开销偏大

长视频会按每个阶段窗累积候选帧，并为同一时刻保存多份原帧和 ROI。视频 38 本次产生 116 个证据组和约 2192 个中间文件，随后串行执行 29 个 Qwen 上下文批次。后续应先做候选帧全局上限、重复 ROI 去重和批量请求优化，但不能改变当前 run 的证据语义。

## 11. 测试

在当前代码状态下运行：

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

最近一次结果：`315 tests / OK`。

专项代码位置：

- `resistance_agent/series_rubric.py`
- `resistance_agent/skills/executors.py`
- `resistance_agent/skills/router.py`
- `tests/`
