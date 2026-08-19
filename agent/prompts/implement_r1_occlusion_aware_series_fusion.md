# Agent 实施提示词：改造 R1 电流表串联检测

你正在仓库的 `agent` 目录中修改 Agent 版。请直接完成代码修改、测试和一次本地验证，不要只给设计方案。

## 目标

把 Agent 版 Rubric 1“电流表串联”改造成以下通用实时算法：

1. 当前视频本次运行重新识别实验阶段；
2. 当前帧动态定位电表、电池盒、开关和待测定值电阻；
3. 电表只露出一半时，使用 `A/V` 字样和红绿接线面板确认身份；
4. 以 2 FPS 粗扫描、局部连续帧和严格复核检测电流表是否直接跨接电池；
5. 导线太细、遮挡或交叉时，融合端点、相邻帧、六图局部拓扑和最终连接图，仍输出 `pass/fail`；
6. 正式 `execute` 不读取视频专属配置、固定 ROI、历史预测或人工答案。

最终 R1 主结果只能是：

```json
{
  "decision": "pass | fail",
  "binary_score": 1,
  "confidence": 0.0
}
```

`binary_score` 必须与 `decision` 一致：`pass=1`，`fail=0`。

## 开始前必须阅读

先阅读并遵守：

- 项目根目录 `AGENTS.md`；
- `agent/prompts/tool_scheduler_anti_overfitting.md`；
- `agent/resistance_agent/series_rubric.py`；
- `agent/resistance_agent/skills/router.py`；
- `agent/resistance_agent/skills/executors.py`；
- `agent/resistance_agent/toolkit.py`；
- `scripts/run_ammeter_battery_direct_state_2fps.py`；
- `scripts/detect_ammeter_region.py`；
- `scripts/fuse_series_circuit_evidence.py`；
- `docs/series_circuit_evidence_fusion_readme.md`；
- `docs/ammeter_battery_direct_state_2fps_green_red_identity_readme.md`。

复用算法思想和结构化字段，但正式 `execute` 不得读取上述脚本以前为五个开发视频生成的输出。

## 第一步：审计并清理正式路径

搜索 Agent 代码和正式配置中的以下内容：

```text
supported_video_ids
if video_id ==
video_38_graph
按 video_id 索引的 ROI/时间窗
historical fallback
历史 summary/result/prediction
Excel 或人工真值
```

要求：

- `video_id`、文件名和 SHA 只能用于关联输入、输出和缓存；
- 正式 `execute` 删除所有按视频身份选择算法、参数、ROI、时间窗和结论的逻辑；
- 正式 `execute` 禁止读取旧 `outputs` 中该视频的阶段、ROI、预测和人工复核；
- replay/regression 路径可以保留历史工件，但必须与正式 `execute` 明确隔离；
- 不得用某个开发视频的时间点或坐标作为默认回退；
- 阶段缺失时使用所有视频一致的 `broad_search`。

审计结果必须写入运行工件，并包含：

```json
{
  "selection_basis": "current_video_observed_situation_only",
  "observed_stages": [],
  "selected_skills": [
    {
      "rubric_ids": [1],
      "skill_id": "",
      "parameters": {},
      "selected_by": ""
    }
  ],
  "video_id_used_for_routing": false,
  "historical_artifacts_used": false,
  "fixed_video_roi_used": false
}
```

## 第二步：从当前视频生成阶段

正式运行时重新检测：

```text
wiring_action
measurement_action
writing_action
cleanup_action
```

再形成：

```text
circuit_wiring
measurement_1
recording_1
circuit_rewiring
measurement_2
recording_2
material_cleanup
```

R1 优先扫描当前实验的 `circuit_wiring` 和 `circuit_rewiring`，并在接线完成后的稳定时刻恢复最终拓扑。若阶段缺失，使用统一 `broad_search` 扫描当前实验有效区间。不要加载历史时间窗。

如果视频包含多个实验，使用当前 run 检测到的 terminal cleanup/新实验边界隔离，不能把后一个实验拼到前一个实验。

## 第三步：动态定位器材

从当前帧动态定位以下器材和端点：

```text
ammeter
voltmeter
battery_holder
single_pole_switch
fixed_resistor
```

ROI 必须由当前帧产生，并可通过光流、ORB/affine 或目标跟踪传播到相邻帧。镜头变化或跟踪失败时重新检测，不允许查按视频 ID 保存的固定 ROI。

### 部分电表身份

电表不要求完整进入画面。身份规则：

```text
A 或橙色电表底座上的墨绿色接线面板 -> ammeter
V 或橙色电表底座上的红色接线面板 -> voltmeter
任一清楚线索即可确认身份
```

限制：

- 绿色电池外皮、绿色插头不能当成绿色电表面板；
- 红色导线、桌面红色虚线不能当成红色电表面板；
- 颜色候选必须与橙色电表底座、接线柱结构或局部表盘结构空间一致；
- 红色电压表证据用于排除 `ammeter` 误认；
- 颜色只确认身份，不能直接触发 `fail`。

保存每个时间点的动态 bbox、跟踪来源和身份依据：

```json
{
  "frame_id": "",
  "timestamp_seconds": 0.0,
  "bbox_xyxy": [],
  "identity": "ammeter | voltmeter | unknown",
  "identity_basis": "A | V | green_terminal_panel | red_terminal_panel | combined",
  "roi_source": "dynamic_detection | temporal_tracking"
}
```

## 第四步：2 FPS 异构窗口扫描

在当前实验有效区间内按 `2 FPS` 扫描。使用通用参数，不按视频身份改变：

- 粗窗口：`16 秒`；
- 每个粗窗口生成全景时间表、器材走线区、动态电流表端子 ROI、动态电池端子 ROI、红绿身份图；
- 对走线区计算连续帧变化，至少选择一次变化最大的时刻；
- 对候选建立约 `4 秒`连续帧细窗口；
- 细层输出二值 `present/absent`，置信度作为诊断；
- 每个细层 `present` 使用前、中、后三张独立全景执行严格复核。

严格复核输出：

```json
{
  "verification": "confirmed | rejected",
  "ammeter_present": true,
  "battery_holder_present": true,
  "two_distinct_ammeter_terminals_connected": true,
  "two_distinct_battery_terminals_connected": true,
  "direct_pairing_visible": true,
  "path_relation": "direct | via_component | occluded_likely_direct | no_connection",
  "intermediate_components": [],
  "confidence": 0.0,
  "evidence": ""
}
```

`path_relation` 和 `intermediate_components` 必须是结构化字段，不能只把“经过开关”写在自然语言里。

## 第五步：三路证据融合

融合三条当前 run 的证据：

1. 最终连接图：电池、电流表、开关和待测电阻是否形成单一串联环路；
2. 六图 CV+VLM：同一稳定时刻的全景、端点 ROI 和走线路径 ROI；
3. 2 FPS 过程扫描：接线期间是否出现电流表双边直跨电池。

不要简单多数投票。使用以下通用规则：

### 直接 fail

满足任一项即可 `fail`：

- 2 FPS 严格层确认两根导线直接覆盖两个不同电流表端子和两个不同电池端子；
- 当前 run 最终连接图以高置信度确认核心器材没有形成单一串联环路；
- 六图在拓扑充分可见、无需推断时确认结构化违规。

### 遮挡补偿 fail

导线太细或被遮挡，无法在单帧完整追踪时，只有以下条件全部成立才允许 `fail`：

- 六图给出高置信度结构化违规；
- 2 FPS 细层在相邻帧给出独立阳性；
- 两个不同电流表端点和两个不同电池端点均有可见接线；
- `path_relation` 不是 `via_component`；
- 没有当前 run 的强最终串联拓扑反证；
- 所有证据来自当前视频本次运行。

不要因为导线没有完整显示而停工，也不要把“端点都有线”单独当成违规。若相邻帧结构化确认路径经过开关、电阻或电压表，则不能使用遮挡补偿分支判直跨。

### pass

没有直接 fail 或遮挡补偿 fail 时输出 `pass`。低可见性只降低 `confidence`，不产生第三类主结果。

## 第六步：区分观察与推断

最终 R1 结果同时保存：

```json
{
  "decision": "pass | fail",
  "binary_score": 1,
  "confidence": 0.0,
  "final_series_circuit": "pass | fail",
  "temporary_direct_across_battery": "pass | fail",
  "decision_branch": "direct_violation | occlusion_corroboration | binary_fallback",
  "direct_observations": [],
  "derived_observations": [],
  "supporting_frame_ids": [],
  "supporting_timestamps_seconds": [],
  "reason": ""
}
```

VLM 置信度不能直接当成校准概率；融合置信度必须由固定、视频身份无关的规则生成。

## 第七步：测试要求

至少增加以下回归测试：

1. 只露出绿色面板的半块电流表仍识别为 `ammeter`；
2. 红色面板识别为 `voltmeter`，不会被当成电流表；
3. 绿色电池和绿色插头不触发电流表身份；
4. 红导线不触发电压表面板；
5. 单条电池到电流表连接边输出 `pass`；
6. 两条直接边输出 `fail`；
7. 导线经过开关或电阻输出 `pass`；
8. 遮挡场景满足全部多分支条件时输出 `fail`；
9. 强最终串联拓扑反证可以阻止遮挡补偿误报；
10. 相同视觉观察只改变 `video_id`，Skill、参数和最终结果必须完全相同；
11. 正式 `execute` 断言 `historical_artifacts_used=false`；
12. 正式 `execute` 断言 `fixed_video_roi_used=false`；
13. 最终结果只能为 `pass/fail`；
14. JSON 可重新解析，证据帧路径存在。

测试不得引用视频 8、16、24、32、38 的固定时间点或固定 ROI 来决定预期结果。可以使用合成观察对象和匿名临时视频 ID。

## 第八步：实际验证

完成代码后：

1. 运行相关单元测试；
2. 使用一个当前输入视频执行新的正式 `execute`，生成全新 `run-id`；
3. 检查 `plan_live_skills` 审计字段；
4. 检查动态 ROI、红绿身份图、粗/细/严格窗口和最终 R1 JSON；
5. 在预测冻结前不要读取 `实验标准.xlsx`；
6. 若使用开发视频，只能报告 regression/development 结果，不得称为泛化准确率。

原始视频和 Excel 只读，输出写入新的 `outputs` 或 Agent run 目录，不覆盖历史结果。

## 完成交付

最终回复必须简短说明：

- 修改了哪些正式 `execute` 文件；
- 删除了哪些视频专属、历史回退或固定 ROI 依赖；
- R1 输出是 `pass` 还是 `fail`；
- 置信度和决定性证据时间；
- `plan_live_skills` 四个反过拟合审计字段；
- 测试命令、测试数量和结果；
- 新输出文件的绝对路径；
- 尚未经过新视频 holdout 验证这一限制。
