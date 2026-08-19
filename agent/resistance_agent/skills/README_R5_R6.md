# Agent R5/R6 当前算法与变更说明

## 1. 文档范围

本文描述 Agent 当前实际执行的两个电表评分项，以及 R4 对其当前 run 证据的复用：

| Rubric | 名称 | 主输出 |
|---|---|---|
| R5 | 指针正常偏转 | `pass` / `fail` |
| R6 | 电表量程合适 | `pass` / `fail` |

当前总版本为：

```text
r56_temporal_meter_v5_live_closed_stable_cv_v3
```

R5 与 R6 共用时间窗、原帧、电表检测、动态 ROI 和部分语义观察，但使用不同的最终归并规则：

- R5 主要回答“两块表在通电测量时是否出现正常正向偏转”；
- R6 主要回答“稳定指针是否处于可接受状态，是否出现超量程、明显反转或始终为零”。

R6 的当前主路径是确定性几何分类。它是“量程合适”的视觉代理，不等同于完整识别每根导线接入了 `0.6A/3A/3V/15V` 中的哪个端子。

## 2. 代码位置

| 文件 | 作用 |
|---|---|
| `resistance_agent/meter_rubrics.py` | R5/R6 共享取证、Qwen 观察校验、R5 reducer、R6 基础 reducer 和 Agent 集成 |
| `resistance_agent/skills/dynamic_meter_reading.py` | 动态表盘候选选择、跨帧跟踪、A/V 身份提示、ROI 质量控制 |
| `resistance_agent/skills/closed_stable_stage_producer.py` | 根据当前 Temporal Guard 结果生成四阶段清单并启动 1.0s/0.1s CPU 搜索 |
| `resistance_agent/skills/closed_stable_r6_cv_v3.py` | 将稳定指针角度映射为几何状态并生成 R6 二分类 |
| `config.json` | R5/R6 模型入口、共享标定、搜索步长和超时配置 |
| `tests/test_toolkit.py` | R5 reducer、R6 几何分类、四阶段生成和 Agent 状态写入测试 |

## 3. 版本变化

### 3.1 旧路径

早期 R5/R6 主要使用：

```text
Temporal Guard
  -> 少量候选帧
  -> OpenCV 表盘 ROI
  -> Qwen 判断指针与端子状态
  -> R5/R6 reducer
```

主要问题：

1. OpenCV 找不到表盘时容易把“没有证据”直接变成 `fail`；
2. 单帧 ROI 可能裁偏、漂移或把电流表和电压表角色弄反；
3. R6 只看某个最终候选时，可能漏掉较早阶段已经出现的超量程；
4. 旧五视频阶段搜索文件不能用于新视频的当前运行取证。

### 3.2 当前 V5 路径

当前版本新增：

1. 动态表盘候选与短窗口跨帧身份跟踪；
2. 原帧、表盘、端子、宽 ROI 和增强 ROI 成组保存；
3. R4 v15 的严格帧绑定反转证据可覆盖 R5 的弱语义结果；
4. R6 在当前 Agent 运行中生成自己的四阶段搜索结果；
5. 1.0 秒粗搜定位候选，候选附近使用 0.1 秒密集帧做角度稳定共识；
6. 强制扫描全部四阶段，不在发现第一组稳定候选后提前停止；
7. Temporal Guard 缺少某轮记录时，按动作阶段间隙生成统一代理窗；
8. 当前运行缓存必须同时匹配视频哈希、manifest 哈希、结果哈希和源视频完整性。

路由依据是当前视频的阶段结构和可见状态，不是视频 ID。

## 4. 共享取证流程

```text
当前视频副本
  -> Temporal Guard / Rubric 边界结果
  -> 候选时间窗
  -> 抽取原帧
  -> OpenCV 检测模拟电表
  -> 动态 ROI、表盘/端子裁剪和增强
  -> 短窗口跨帧跟踪与 A/V 身份提示
  -> 选择最多 4 个信息量较高的图组
  -> Qwen 输出可见状态 JSON
  -> R5 reducer
  -> 当前视频四阶段 1.0s/0.1s 稳定角度生产器
  -> R6 deterministic reducer
  -> rubric_5.json / rubric_6.json
```

共享取证保存：

- 原始时间点和帧号；
- 未裁剪原帧；
- 表盘 face ROI；
- 包含表身和上下文的 wide ROI；
- 对比度增强 ROI；
- 端子 ROI；
- OpenCV 指针诊断；
- Qwen 原始观察与 schema 校验结果；
- 视频、manifest 和结果 SHA-256。

OpenCV 在 R5 的语义路径中主要用于候选定位和诊断，不单独替代最终语义判断。R6 的 closed-stable 主路径则直接使用经过多帧稳定共识的 OpenCV 几何角度。

## 5. 时间窗与抽帧

### 5.1 R5/Qwen 候选时间窗

`meter_rubrics.candidate_windows()` 按以下顺序建立候选窗：

1. 所有显式 `measurement_*` 阶段，前后各扩 2 秒；
2. `recording_1` 前 15 秒到记录早期；
3. `recording_1` 前 60 秒的宽恢复窗；
4. 最后一次接线到第一次记录之间的过渡窗；
5. 若仍没有窗口，使用接线尾部或视频中段的统一回退窗。

抽帧规则：

- 显式测量窗约每 1.5 秒取一帧；
- 记录邻域约每 3 秒取一帧；
- 宽恢复窗约每 5 秒取一帧；
- 单视频最多取 28 个初始候选时间点；
- 最终送给 Qwen 的图组最多 4 组，并尽量保持至少 2 秒时间间隔。

### 5.2 R6 四阶段时间窗

固定阶段顺序：

```text
measurement_1
recording_1
measurement_2
recording_2
```

生成规则：

1. Temporal Guard 直接给出的测量或记录阶段优先使用；
2. 有记录阶段但没有测量阶段时，使用记录开始前最多 8 秒作为测量代理；
3. 某轮记录阶段缺失时，使用该轮前一动作结束到后一动作开始的间隙；
4. 间隙末端最多 8 秒作为记录代理，记录代理之前最多 8 秒作为测量代理；
5. 视频 24 的第二轮就是由 `circuit_rewiring` 结束到 `material_cleanup` 开始的统一规则生成，不包含视频 24 专属条件。

每个阶段对电流表和电压表分别执行：

```text
1.0 秒粗搜
  -> 最佳角度候选
  -> 候选前后 0.7 秒
  -> 0.1 秒密集搜索
  -> 连续帧稳定角度共识
```

`force_full_stage_scan=true` 会临时关闭底层脚本的“找到稳定候选后提前停止”，搜索结束后再恢复原 Temporal Guard 的 `segmentation_claim` 语义。

## 6. 动态表盘与 ROI

### 6.1 OpenCV 初选

`detect_colored_meters_v4.py` 在缩小后的检测图上寻找候选，并把坐标映射回源图像像素。候选质量综合考虑：

- 检测分数；
- 表盘相似度；
- 表盘结构完整度；
- 指针检测置信度；
- 原帧清晰度；
- 时间窗优先级。

### 6.2 动态候选选择

`dynamic_meter_reading.py` 会：

- 合并空间上重复的 A/V 假设；
- 保留两个物理上不同的表盘；
- 根据可见 A/V 字样、表盘结构、端子布局和位置给出身份提示；
- 在短时间窗中跟踪候选，防止单帧角色错配；
- 身份冲突时把较弱轨迹降为 `unknown`，而不是强制分配错误角色。

Qwen 收到的每个时间组由原图和对应候选的 face、terminal、enhanced wide 视图组成。模型必须从可见像素重新确认角色，不能把本地 `role_hint` 当作真值。

## 7. R5：指针正常偏转

### 7.1 Qwen 观察字段

Qwen 只输出观察，不直接评分。主要字段：

```text
image_group
circuit_state: energized / deenergized / unclear
identity: ammeter / voltmeter / unknown
pointer_state: normal_rightward / zero / reverse / overrange / uncertain
pointer_scale_position: near_zero / low / mid / high / near_full / uncertain
terminal_occupancy_left_middle_right
selected_range_label
plugged_terminal_visible
range_assessment
confidence
evidence
```

断电图组中的零位属于正常静止状态，不进入异常测量判断。`unclear` 图组仍保留，但会通过置信度和多帧规则降低单帧误判影响。

### 7.2 多帧状态归并

对于电流表和电压表分别按置信度累加 `pointer_state`。每条观察至少贡献 `0.05` 权重；相同总权重时优先 `normal_rightward`。

稳定异常必须满足：

- 状态为 `zero`、`reverse` 或 `overrange`；
- 图组不是明确 `deenergized`；
- 观察置信度至少 `0.65`；
- 至少来自两个不同图组，或者某一图组置信度至少 `0.90`。

### 7.3 R4 复用 R5 当前 run 证据

正式 execute 的依赖方向固定为 `R5 -> R4`。R5 不再读取 R4 端点结果；R4 v21 直接采用 R5 的二分类、指针状态、原帧和动态 ROI 路径。

R4 只接受同时满足以下条件的 R5：

- `rubric_5.json` 位于同一 run 的 `rubrics` 目录；
- 视频 ID、源视频名和 `routing_policy` 与当前 run 一致；
- `execution_mode=execute_visual_evidence`；
- `source_artifact` 指向同一 run 的 `meter_rubrics/meter_evidence_report.json`；
- 证据报告声明未读取 Excel、未发送真值且未使用历史回退。

R4 不再使用导线颜色、端点拓扑或视频专属固定 ROI。旧 R4-to-R5 有符号覆盖代码只保留给显式 replay/regression，正式 toolkit 不传入该依赖。

### 7.4 R5 判定顺序

1. 有严格帧绑定的反转覆盖：`fail`；
2. 电流表和电压表都归并为正常正偏：`pass`；
3. 任一电表出现稳定异常：`fail`；
4. 可确认电路处于通电状态，且至少一块表出现置信度不低于 `0.45` 的正常正偏：`pass`，另一块表按低可见性记录；
5. 时间搜索和 ROI 增强后仍没有找到正常偏转：`fail`。

主结果始终是二分类。`uncertain` 只允许出现在诊断字段，不能成为最终类别。

### 7.5 R5 置信度

初始值来自 Qwen `overall_confidence`，再与两种电表的多帧共识合并：

- 两块表都不确定时乘 `0.65`；
- 至少一块表可归并时，置信度不低于两块表共识均值的一半；
- 严格反转覆盖存在时，置信度不低于该覆盖证据的置信度。

## 8. R6：电表量程合适

### 8.1 两层实现

R6 先与 R5 一起生成一个 Qwen 基础结果，然后优先执行当前视频 closed-stable 几何路径：

```text
Qwen 基础 R6
  -> 当前视频四阶段生产器成功
     -> closed_stable_r6_cv_v3 覆盖基础结果
  -> 当前视频生产器失败
     -> 保留 Qwen 基础 R6
```

正式 `execute` 不读取五视频历史阶段搜索文件。当前视频生产器失败时，R6 保留本次运行已经生成的 Qwen 二分类结果，并在诊断字段记录失败原因。历史几何结果只允许由显式 `replay/regression` 调用传入。

### 8.2 Qwen 基础 R6

基础 reducer 只有在端子和指针状态互相支持时才接受明确量程错误：

- `too_low` 必须同时看到已连接的小量程端子，以及 `near_full/overrange` 指针；
- `too_high` 必须同时看到已连接的大量程端子，以及 `near_zero/low` 指针；
- 端子标签必须与实际 occupied 位置一致；
- 明确错误观察置信度至少 `0.65`；
- 合法量程观察要求 `appropriate`、指针位于 `low/mid/high` 且置信度至少 `0.45`。

基础判定顺序：

1. 明确端子-指针不匹配或超量程：`fail`；
2. 明确量程合适：`pass`；
3. R5 已正常且没有可见量程冲突：`pass`；
4. 仍未显示量程合适：`fail`。

### 8.3 Closed-stable 几何比例

对每个稳定阶段候选计算未截断比例：

```text
ratio = signed(pointer_angle - zero_angle)
        / signed(full_scale_angle - zero_angle)
```

`signed` 由标定中的 `sweep_direction` 决定。比例不会先截断到 `[0, 1]`，因此能够区分反转和超量程。

状态定义：

| 状态 | 条件 |
|---|---|
| `zero_band_candidate` | 指针位于零位角不确定区间 |
| `negative_deflection_candidate` | 指针越过零位并朝反方向移动 |
| `positive_but_too_small_candidate` | 正向比例小于 `0.05` |
| `normal_positive_deflection_candidate` | 合法正向比例 |
| `full_scale_band_candidate` | 位于满量程端点不确定区间 |
| `overrange_candidate` | 越过满量程端点和不确定区间 |

默认角度不确定度：

```text
zero uncertainty = 1.5 deg
full-scale uncertainty = 1.5 deg
pointer uncertainty = 0.5 deg
```

明显反转阈值：

```text
ratio <= -0.10
```

`-0.10 < ratio < 0` 视为近零边界，不允许单个弱负角度覆盖其他阶段的正常正偏证据。

### 8.4 R6 全阶段异常优先归并

R6 不只读取角色最终选择的阶段，而是检查四阶段内全部稳定候选：

1. 任一角色出现明确超量程：`fail`，置信度 `0.95`；
2. 任一角色出现明显反转：`fail`，置信度 `0.95`；
3. 任一角色的全部可用稳定候选都在零位或近零边界：`fail`，置信度 `0.95`；
4. 两种电表都有合法非零候选：`pass`，置信度 `0.95`；
5. 只有一种电表有合法非零候选，另一种缺失：`pass`，置信度 `0.72`；
6. 两种电表都没有可检查候选：`fail`，置信度 `0.38`。

第五条是项目“二分类优先”的低门槛回退：可见的一块表提供合法证据时，不因另一块表遮挡或定位失败而停止评分，但会降低置信度。

### 8.5 当前 R6 的含义边界

当前 closed-stable 主路径能可靠检查：

- 超量程；
- 明显反转；
- 始终停在零位；
- 合法非零稳定偏转。

它不能单独证明：

- 学生具体接入了哪个量程孔；
- 大量程导致的偏转过小一定不合适；
- 精确电流、电压读数正确；
- 开关闭合事件由本算法直接观察到。

输出固定保留：

```json
{
  "closed_stable_binding": "measurement_or_recording_stable_consensus_proxy",
  "switch_closure_directly_observed": false
}
```

## 9. 当前运行、缓存与完整性

当前四阶段生产器只复用同一次运行中经过验证的 checkpoint。以下条件必须全部满足：

- skill 版本一致；
- 源视频绝对路径一致；
- 源视频 SHA-256 一致；
- 当前 manifest SHA-256 一致；
- 结果文件存在且 SHA-256 一致；
- 结果只绑定一个当前视频；
- 源视频完整性字段为 unchanged；
- 结果中的视频结束哈希与当前源视频哈希一致。

任一条件不满足都会创建新的 `search_<n>` 目录并重新搜索。原视频只读，生产器运行前后再次比较 SHA-256。

这与“按视频 ID 读取历史最佳工件”不同：ID 只用于绑定输出身份，不能决定时间窗、算法分支或最终类别。

## 10. Agent 集成顺序

`run_meter_rubrics` 的执行顺序：

1. 校验 execute 模式和当前运行状态；
2. 读取当前 Temporal Guard 或边界精修记录；
3. 对当前运行视频副本生成共享 R5/R6 图像证据；
4. 调用 Qwen 并校验单个 JSON 对象；
5. 归并 R5 和基础 R6；
6. 正式 execute 不加载 R4 历史证据；
7. 启动当前视频四阶段生产器；
8. 用 closed-stable CV V3 覆盖基础 R6；
9. 写入 `rubric_5.json`、`rubric_6.json` 和 `meter_evidence_report.json`；
10. 更新 Agent `state.json`，供 `validate_run` 和 `finalize_run` 使用。

R5 和 R6 属于同一个 producer group。调用 `run_rubric_bundle([5])` 时会共同生成 R5/R6，避免重复解码视频和重复调用 Qwen。

当请求包含 R4 时，`run_polarity_rubric` 在 R5/R6 之后执行轻量归并：它只校验当前 run 的 R5 及证据报告，然后写出 `rubric_4.json`，不会再次解码视频或调用 Qwen。

## 11. 配置

当前核心配置：

```json
{
  "stage_producer": {
    "enabled": true,
    "measurement_lead_seconds": 8.0,
    "force_full_stage_scan": true,
    "coarse_seconds": 1.0,
    "dense_seconds": 0.1,
    "dense_radius_seconds": 0.7,
    "max_feature_width": 2400,
    "timeout_seconds": 3600
  },
  "routing_policy": "current-stage geometry; no prediction or Excel routing"
}
```

生产环境依赖共享的 CPU 表盘脚本和同型号电表标定：

```text
<optional-local-stage-producer>/
```

本项目假设老师提供的新视频继续使用相同实验设备和拍摄条件，因此共享表型标定属于设备配置，不属于按学生或视频 ID 适配。

## 12. 输出字段

### 12.1 `rubric_5.json`

主要字段：

```text
decision
predicted_score
confidence
reason
diagnostics.needle_states
diagnostics.effective_needle_states
diagnostics.identity_observations
diagnostics.candidate_windows
diagnostics.qwen_observation
diagnostics.signed_pointer_evidence
diagnostics.original_frame_paths
diagnostics.roi_paths
```

### 12.2 `rubric_6.json`

主要字段：

```text
decision
predicted_score
confidence
reason
diagnostics.confidence_level
diagnostics.overrange_roles
diagnostics.strong_reverse_roles
diagnostics.zero_only_roles
diagnostics.assessable_roles
diagnostics.missing_roles
diagnostics.roles.<role>.stage_observations
diagnostics.roles.<role>.evidence_paths
diagnostics.source_stage_results
```

### 12.3 `meter_evidence_report.json`

同时保存：

- 算法版本；
- 视频 SHA-256；
- 时间窗和选中帧；
- 动态表盘轨迹；
- Qwen 原观察；
- R4 有符号指针证据；
- 当前四阶段生产器 summary；
- R5/R6 完整 reducer 输出；
- `excel_accessed=false`；
- `ground_truth_sent_to_model=false`。

## 13. 五视频当前 R6 实跑结果

本次使用全新输出目录，五个视频全部 `checkpoint_reused=false`。视觉运行没有读取 Excel，也没有调用 Qwen；Excel 只在预测冻结后用于评测。

| 视频 | R6 | 置信度 | A 稳定阶段 | V 稳定阶段 | 证据图 | 规则 |
|---:|---|---:|---:|---:|---:|---|
| 8 | `fail` | 0.95 | 2 | 2 | 11 | `explicit_overrange_in_closed_stable_proxy` |
| 16 | `pass` | 0.95 | 3 | 4 | 224 | `both_roles_have_nonzero_legal_candidate` |
| 24 | `pass` | 0.72 | 1 | 0 | 24 | `one_role_legal_other_missing_fallback` |
| 32 | `pass` | 0.95 | 3 | 3 | 137 | `both_roles_have_nonzero_legal_candidate` |
| 38 | `fail` | 0.95 | 1 | 0 | 5 | `all_found_stable_candidates_at_zero` |

冻结后与 `实验标准.xlsx` 的 R6 最终得分比较：

```text
5/5 correct
accuracy = 100%
pass recall = 3/3
fail recall = 2/2
false positive = 0
false negative = 0
```

批次汇总：

```text
agent/.tmp/
  live_closed_stable_stage_producer_all5_fresh_20260816_run01/
  batch_summary.json
```

这次批次只重跑了 R6 的实时四阶段生产器。它不能作为“当前 R5 五视频实时准确率”的证据；R5 需要包含 Qwen 和 R4 帧绑定覆盖的独立冻结运行后再单独评测。

## 14. 测试

当前 R5/R6 专项测试覆盖：

- 两轮测量/记录窗口生成；
- 显式 measurement 优先；
- 缺失第二轮时按阶段间隙补齐；
- 生产器提前停止抑制和语义恢复；
- 完整四阶段 role summary 重建；
- R6 异常优先级；
- 弱负角度不能覆盖合法正偏；
- R5/R6 工件写入和 Agent 状态更新；
- Qwen schema、端子支持、稳定异常和反转覆盖 reducer。

核心专项测试：

```powershell
cd <repository-root>
python -m unittest -v tests.test_toolkit.ResistanceAgentToolkitTests.test_closed_stable_stage_producer_builds_two_cycle_current_video_windows
python -m unittest -v tests.test_toolkit.ResistanceAgentToolkitTests.test_closed_stable_stage_producer_fills_missing_second_cycle_from_stage_gap
python -m unittest -v tests.test_toolkit.ResistanceAgentToolkitTests.test_closed_stable_r6_cv_v3_abnormal_priority
python -m unittest -v tests.test_toolkit.ResistanceAgentToolkitTests.test_meter_rubrics_write_binary_results_and_update_state
```

2026-08-16 当前八个四阶段/R6 专项目标运行结果：

```text
Ran 8 tests
OK
```

完整 `tests.test_toolkit` 在当前 worktree 中仍有与本算法无关的缺失视频、发布目录、历史 evaluation fixture 和旧 17/18 MCP 工具数量断言；应同时查看专项测试结果，不把这些环境错误归因到 R5/R6。

## 15. 修改算法时应改哪里

| 要修改的行为 | 代码位置 |
|---|---|
| R5 时间窗 | `meter_rubrics.candidate_windows()` |
| R5 抽帧密度 | `meter_rubrics.sampling_timestamps()` |
| ROI 检测和增强 | `meter_rubrics._export_candidates()` |
| 动态表盘选择与角色跟踪 | `skills/dynamic_meter_reading.py` |
| Qwen 观察字段和提示词 | `meter_rubrics._prompt()` |
| R5 判定优先级 | `meter_rubrics.reduce_results()` |
| R4 反转证据接入 | `meter_rubrics.load_signed_pointer_evidence()` |
| 四阶段生成和缺失阶段回退 | `skills/closed_stable_stage_producer.build_stage_intervals()` |
| 1.0s/0.1s 搜索参数 | `config.json` 的 `stage_producer` |
| 指针角度到比例的映射 | `skills/closed_stable_r6_cv_v3.classify_pointer_scale_state()` |
| R6 最终规则 | `skills/closed_stable_r6_cv_v3.classify_video()` |
| Agent/MCP 参数传递 | `resistance_agent/toolkit.py::run_meter_rubrics()` |

修改后至少执行：

1. 对应 reducer 单元测试；
2. 四阶段 manifest 测试；
3. 一个视频的无缓存当前运行；
4. 检查原帧、ROI、角度、证据路径和 JSON 字段；
5. 冻结预测；
6. 最后才读取 Excel 计算准确率；
7. 确认源视频运行前后 SHA-256 一致。

## 16. 不允许引入的适配方式

后续优化不得：

- 根据视频 ID 选择不同时间窗、阈值或最终类别；
- 把 Excel 真值发送给 Qwen 或视觉生产器；
- 视觉运行前读取 Excel 决定分支；
- 将旧视频预测直接复制为当前运行结果；
- 把 OpenCV 定位失败直接解释成学生操作失败；
- 用第三类结果替代最终 `pass/fail`；
- 覆盖、重编码或修改原始视频。

允许复用：

- 同型号电表的共享几何标定；
- 同一套阶段生成规则；
- 同一套阈值和 reducer；
- 当前运行内经过视频/manifest/结果哈希校验的 checkpoint；
- 一次取证后由 R5/R6 共同消费的原帧、ROI 和模型观察。

## 17. 当前限制与下一步

当前主要限制：

1. R5 的最终语义路径仍需要 Qwen；
2. R5 最多选择 4 个图组，短暂异常可能仍被漏掉；
3. R6 四阶段 CPU 搜索在长视频上较慢；
4. R6 的 closed-stable 绑定来自动作阶段代理，不是直接开关闭合检测；
5. R6 没有完整识别实际量程端子，当前主要依据稳定指针状态；
6. 视频 24 的电压表没有稳定候选，当前通过单角色合法证据输出中置信度 `pass`；
7. R5 当前版本需要重新进行一次与本次 R6 相同口径的五视频无缓存冻结评测。

建议优化顺序：

1. 为 R5 增加候选附近的短窗口密集帧共识，降低对单个 Qwen 图组的依赖；
2. 复用 R6 已生成的表盘 ROI 和阶段候选，减少 R5 重复解码；
3. 在不改变二分类主结果的前提下，增加实际端子量程识别；
4. 优化长视频的特征缓存和两种电表共享计算；
5. 用老师新增视频按相同设备条件进行盲测。
