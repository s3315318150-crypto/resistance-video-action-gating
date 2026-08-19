# AI 开发上下文

本文件是代码型 AI 进入本仓库后的第一入口。它描述当前正式实现、修改边界和验证方法。算法细节以代码和各目录 README 为准；不要根据开发视频的编号或历史结果推断正式行为。

## 1. 项目目标

本项目从中学“伏安法测电阻”实验视频中定位动作与视觉证据，并对每个 Rubric 输出二分类结果：

```text
pass | fail
```

`confidence`、可见性、候选帧、模型原文和冲突原因是诊断信息，不能替代主结果，也不能产生 `uncertain`、`abstained` 或 `needs_review` 等第三类评分。

## 2. 仓库路线

| 路径 | 用途 | 修改建议 |
|---|---|---|
| `workflow/v1/` | 原始发布版本和历史基线 | 默认保持兼容，不作为新 Agent 的依赖入口 |
| `workflow/v2/` | 当前推荐的阶段定位与边界精修 Workflow | Agent 通过仓库相对路径复用 |
| `agent/` | 当前视频情境驱动的 Skill Router、动态取证、融合和 MCP 入口 | 新的 Agent 功能在这里实现 |

先读：

1. `README.md`
2. `agent/README.md`
3. `agent/config.json`
4. `agent/resistance_agent/orchestrator.py`
5. `agent/resistance_agent/toolkit.py`
6. 当前 Rubric 对应的 `agent/docs/algorithms/<algorithm>/README.md`

## 3. 正式 Agent 流程

正式 `execute` 的顺序是：

```text
inspect_video
  -> create_run
  -> run_full_pipeline
  -> refine_rubric_boundaries
  -> plan_live_skills
  -> run_rubric_bundle
  -> 必要时 request_additional_evidence
  -> validate_run
  -> finalize_run
```

`run_rubric_bundle` 会按生产组执行“一次取证、多项消费”，不要为同组 Rubric 重复跑整段视频：

| 生产工具 | Rubric | 主要实现 |
|---|---|---|
| `run_switch_rubric` | R3 接线时开关断开 | `switch_rubric.py`、`opencv_switch_*.py` |
| `run_series_rubric` | R1 电流表串联 | `series_rubric.py`、`r1_frame_sampling_agent.py` |
| `run_meter_rubrics` | R5、R6 电表指针与状态 | `meter_rubrics.py`、`skills/cpu_tick_meter_reading.py` |
| `run_remaining_rubrics` | R0、R2、R8 | `remaining_rubrics.py` |
| `run_polarity_rubric` | R4 正负接线柱 | `polarity_rubric.py`，复用当前 run 的 R5 指针证据 |

当前 Agent 正式发布 `R0-R6、R8`。R7、R9 尚未进入 Agent 正式 execute，不要在没有完整实现、测试和文档时只修改常量宣称已支持。

## 4. 反过拟合硬约束

正式 `execute` 只能依据当前视频、当前 run 生成的阶段、帧、动态 ROI 和视觉观察。

- `video_id`、文件名和学生姓名只用于输入输出关联。
- 不得用视频身份选择 Skill、时间窗、ROI、阈值、提示词、融合分支或结论。
- 不得读取历史预测、旧时间窗、旧 ROI、人工错误说明或 Excel 真值。
- 不得新增 `if video_id == ...`、`video_38_graph`、按视频编号配置表或固定视频 ROI。
- 阶段缺失时使用统一 `broad_search` 或统一二分类兜底。
- Excel 只能在预测冻结后用于离线评测，不能进入模型请求或同次调参。
- 开发视频参与过规则或阈值设计后，只能报告“开发集回归”，不能报告为新视频泛化准确率。

`plan_live_skills` 或等价结果必须保留：

```json
{
  "selection_basis": "current_video_observed_situation_only",
  "observed_stages": [],
  "selected_skills": [],
  "video_id_used_for_routing": false,
  "historical_artifacts_used": false,
  "fixed_video_roi_used": false
}
```

修改路由、抽帧、ROI 或融合代码后，必须搜索视频专属分支、历史结果回退和固定 ROI，并为“相同视觉情况、不同文件身份”补回归测试。完整调度提示词见 `agent/prompts/tool_scheduler_anti_overfitting.md`。

## 5. R5/R6 当前实现

当前版本为：

```text
r56_temporal_meter_v6_cpu_tick_grid
```

R5/R6 不是单文件算法，而是以下组合：

| 模块 | 职责 |
|---|---|
| `agent/resistance_agent/meter_rubrics.py` | 当前 run 帧选择、Qwen 观察、CPU 调用、R6 reducer 调用和最终融合 |
| `agent/resistance_agent/skills/closed_stable_stage_producer.py` | 可选的当前视频四阶段搜索窗口生产 |
| `agent/resistance_agent/skills/closed_stable_r6_cv_v3.py` | 零位、反偏、正常偏转、满量程、超量程与 R6 二分类 |
| `agent/resistance_agent/skills/cpu_tick_meter_reading.py` | 印刷刻度、30 格换算、reverse/overrange 和直接证据融合 |
| `agent/resistance_agent/skills/r5_r6_dense_meter_state/` | SIFT 表头定位、透视校正、导线屏蔽、arc-to-hub 指针检测和多帧刻度共识 |
| `agent/assets/meter_calibration/` | 匿名共享设备标定，不包含视频身份或学生信息 |

详细导航见 `agent/docs/algorithms/r5_r6_dense_meter_state/AI_CONTEXT.md`。

## 6. 修改一个 Rubric 的最小步骤

1. 找到该 Rubric 的 producer 和 reducer，不先扩展整个 Agent 框架。
2. 明确真正需要的可见证据与 `pass`/`fail` 规则。
3. 从当前阶段粗定位，再在候选窗口密集抽取邻帧。
4. ROI 从当前帧动态定位；定位失败时扩大搜索或降低置信度，不把定位失败直接写成学生操作失败。
5. 多帧冲突使用 Rubric 内预先定义的一致 tie-break。
6. 只修复实际复现的问题，并增加一个针对性测试。
7. 先运行专项测试，再运行 Agent 和 Workflow V2 全量测试。

## 7. 运行入口

在仓库根目录执行：

```powershell
python agent\run_agent.py `
  --scheduler deterministic `
  --mode execute `
  --video-ref data\videos\sample.mp4 `
  --run-id sample_execute
```

MCP stdio 服务：

```powershell
python agent\run_mcp_server.py
```

结果写入 `agent/runs/<run-id>/`。原视频只读，运行不得覆盖视频或历史预测。

## 8. 验证清单

```powershell
python -m compileall -q agent workflow\v2
python -m unittest discover -s agent\tests -v
python -m unittest discover -s workflow\v2\tests -v
python agent\run_agent.py --help
git diff --check
```

同时确认：

- JSON 文件均可解析；
- 每个正式 Rubric 主结果是 `pass` 或 `fail`；
- 新代码没有视频 ID 路由、历史工件回退或固定视频 ROI；
- 正式 execute 不读取 Excel 或 replay 结果；
- 配置与代码只使用仓库相对路径；
- 未把实验诊断或 forced-binary 中间产物混入正式结果。

## 9. 发布边界

可以提交通用代码、配置、测试、文档和匿名设备标定。不要提交：

- 原始视频或视频片段；
- Excel、学生姓名或人工标签；
- `outputs/`、`agent/runs/`、模型原始响应和本地日志；
- 真实 API Key、Token、私有接口地址或 `.env`；
- 五个开发视频专属配置、旧时间窗、历史预测或人工复核结果；
- 本机绝对路径和未发布的外部模块依赖。

凭据仅通过 `.env.example` 中声明的环境变量注入。公开文档和测试使用匿名文件名与占位符。

## 10. 工作原则

- 先阅读现有实现，再做最小可运行改动。
- 优先复用现有 producer、Skill、OpenCV 模块和输出 schema。
- 一张可用帧或短邻帧窗口即可启动二分类；低质量证据影响置信度，不自动停工。
- 不为了理论异常增加多层 Gate、重复校验或第三类结果。
- 不删除或回滚无关改动，不重写 `main` 历史，不强制推送。
- 汇报时区分开发集回归、replay/regression 和真正的冻结后新视频盲测。
