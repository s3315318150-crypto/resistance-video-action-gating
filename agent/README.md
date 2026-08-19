# 伏安法视频理解 Agent

Agent 版以 [Workflow V2](../workflow/v2/) 为工程基线，复用稳定视觉能力，并对调度、抽帧、ROI、补充取证和结果融合进行适配与增强；不是简单包装，也不读取 V2 历史预测。

## 改进点

- 每次从当前视频重新识别接线、测量、记录、改线和整理阶段；当前发布集为 `R0-R6、R8`。
- Skill Router 只根据当前 run 已观察到的阶段与证据质量选择执行器。
- 从原生视频帧动态定位仪表、端子、纸面、开关和电池区域，不按视频身份读取固定 ROI。
- 遮挡、冲突或低置信度时，在当前阶段附近申请有限邻帧，再回到同一二分类 reducer。
- 多个 Rubric 复用同一 run 的已生成证据，但不读取历史预测或 Excel 标签。
- R5/R6 使用动态表头定位、导线遮挡屏蔽、相邻帧指针共识、印刷刻度读数和本地二分类融合；详见 [`R5/R6 AI Context`](./docs/algorithms/r5_r6_dense_meter_state/AI_CONTEXT.md) 与 [算法说明](./docs/algorithms/r5_r6_dense_meter_state/README.md)。
- 每项主结果固定为 `pass` 或 `fail`；可见性与置信度只进入诊断字段。

## 执行结构

```mermaid
flowchart LR
    A[当前视频] --> B[Workflow V2 阶段定位]
    B --> C[Rubric 边界精修]
    C --> D[情境 Skill Router]
    D --> E[动态抽帧与 ROI]
    E --> F[OpenCV / VLM 观察]
    F --> G[本地二分类融合]
    G --> H[当前 run 结果]
```

`plan_live_skills` 的审计结果固定包含：

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

视频 ID 和文件名只用于关联输入与输出，不参与 Skill、时间窗、ROI、阈值和结论选择。正式 `execute` 不接受历史 Temporal Guard 回退。

## 当前开发集回归

本次 Agent 发布暂不包含 `R7`、`R9` 记录纸评分。两项仍保留在 Workflow V1/V2 的历史路线中，后续单独接入 Agent。Agent 的正式 execute、MCP schema、Skill 覆盖、校验和最终结果均只处理 `R0-R6、R8` 八项。

| Rubric | 准确率 | 说明 |
|---|---:|---|
| R3 接线时开关断开 | `5/5 = 100%` | fresh execute；pass recall `3/3`，fail recall `2/2` |
| R6 电表状态判断 | `5/5 = 100%` | 无 checkpoint fresh 执行；当前主要是稳定指针状态代理 |
| R2 电压表并联 | `5/5 = 100%` | 视频 8 的 Excel `0.5（存疑）` 按当前二分类展示映射为 pass；五个映射后样本全部为正类，尚未验证 fail recall |
| R8 换电池前断开开关 | `4/4 = 100%` | 视频 8 的非二分类标签不计入；其中一次结果来自接口失败后的 fail 兜底 |
| R1 电流表串联 | `4/5 = 80%` | 视频 8 误判为 pass；pass recall `3/3`，fail recall `1/2` |

以上数字仅是视频 8、16、24、32、38 的开发集回归，不是新视频泛化准确率。R2 若严格只统计 Excel 原始 0/1 标签，口径为 `4/4 = 100%`。

## 安装

在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r agent\requirements.txt
```

把视频放在 `data/videos/`。复制 [`.env.example`](./.env.example) 中的变量到本地环境；仓库内不保存地址和凭据。

## 运行

确定性调度不调用额外的调度模型：

```powershell
python agent\run_agent.py `
  --scheduler deterministic `
  --mode execute `
  --video-ref data\videos\sample.mp4 `
  --run-id sample_execute
```

仅规划和检查输入：

```powershell
python agent\run_agent.py `
  --scheduler deterministic `
  --mode prepare `
  --video-ref data\videos\sample.mp4 `
  --run-id sample_prepare
```

MCP stdio 入口为：

```powershell
python agent\run_mcp_server.py
```

运行目录写入 `agent/runs/<run-id>/`，不会覆盖原视频。Workflow V2 的脚本由仓库内相对路径 `workflow/v2/` 调用。

## 测试

```powershell
python -m compileall -q agent
python -m unittest discover -s agent\tests -v
```

仓库根目录的完整验证还应运行 Workflow V2 测试：

```powershell
python -m compileall -q agent workflow\v2
python -m unittest discover -s workflow\v2\tests -v
```

算法说明见 [`docs/algorithms`](./docs/algorithms/)，反过拟合调度约束见 [`prompts/tool_scheduler_anti_overfitting.md`](./prompts/tool_scheduler_anti_overfitting.md)。本仓库不包含原始视频、Excel、输出包、模型原始响应或真实凭据。
