# Resistance Video Action Gating

伏安法测电阻实验视频的七阶段动作分割、Rubric 独立视觉取证和纸面记录/仪表读数一致性核验原型。

该项目把“什么时候发生了什么动作”和“某个评分项是否有直接视觉证据”分开处理。七阶段分割只提供检索时间窗，不能直接充当评分证据。评分层最终只输出 `pass` 或 `fail`；置信度、可见性和候选帧只作为诊断信息。

> 这是研究原型，不是可直接投入教学评分的完整产品。仓库不包含真实学生视频、姓名、Excel 标注、专家分数、原始模型响应或历史实验输出。

## 核心流程

```mermaid
flowchart LR
    A[全视频边界检测] --> B[七阶段动作分割]
    B --> C[层级 Map/Reduce 与终态截断]
    C --> D[Rubric 独立动作门控]
    D --> E[原始帧与紧密 ROI]
    E --> F[请求体预检与结构化观察]
    F --> G[本地确定性二分类]
    G --> H[预测工件冻结]
    H --> I[冻结后离线评测]
```

七个动作阶段：

| 标识 | 含义 |
|---|---|
| `circuit_wiring` | 电路连接 |
| `measurement_1` | 第一次测量 |
| `recording_1` | 记录第一组数据 |
| `circuit_rewiring` | 电路改接 |
| `measurement_2` | 第二次测量 |
| `recording_2` | 记录第二组数据 |
| `material_cleanup` | 拆除与整理 |

## 算法文件清单

| 用途 | 入口或核心文件 |
|---|---|
| 推荐七阶段动作分割 | `scripts/qwen_experiment_action_hierarchical_v2.py` |
| 七阶段契约、提示词与 Reduce | `scripts/qwen_hierarchical_v1_contract.py`、`scripts/qwen_hierarchical_v1_prompts.py`、`scripts/qwen_hierarchical_v1_reduce.py` |
| 评价 8 断开换电池组 | `scripts/run_resistance_disconnect_battery_sequence_v1.py` |
| 评价 8 本地确定性 reducer | `scripts/resistance_disconnect_battery_sequence_core.py` |
| 纸面记录与仪表读数核验 | `scripts/run_meter_record_consistency_v1.py` |

## 十项评分链

`scripts/run_all_rubrics_v2.py` 是十项评分的公开编排入口。它读取动作分割结果与各项专用证据工件，并把最终评分统一为 `pass` 或 `fail`；它不读取 Excel 标签，也不修改原始视频。

| 指标 | 评分项 | 主要证据来源 |
|---:|---|---|
| 0 | 拆除整理归位 | `material_cleanup` 与整理动作证据 |
| 1 | 电流表串联 | 接线 episode 或电流表-电源短路专用证据 |
| 2 | 电压表并联 | 记录期同帧电压表/电阻端点证据 |
| 3 | 接线时开关断开 | 接线动作与刀闸状态的时间重叠证据 |
| 4 | 电表正负接线柱正确 | 测量/记录期端子、指针和读数符号证据 |
| 5 | 指针正常偏转 | 双表表盘 ROI 与连续帧指针状态 |
| 6 | 电表量程合适 | 量程端子、档位和读数证据 |
| 7 | 正确记录第一组数据 | 第一组纸面 U/I 与仪表读数核验 |
| 8 | 换电池前先断开开关 | `T0-T2 -> T0-T1/T1-T2` 与开关时序 |
| 9 | 正确记录第二组数据 | 第二轮测量语境、仪表读数和纸面 U/I |

先运行各评分项的证据提取器，将其输出路径填入匿名配置，再运行总评：

```powershell
python scripts/run_all_rubrics_v2.py `
  --action-summary outputs/qwen_experiment_action_hierarchical_v2/<run-id>/summary.json `
  --artifact-config configs/ten_rubrics_artifacts.example.json `
  --output-root outputs/all_rubrics_v2
```

配置中的未提供项不会被虚构为通过；逐视频 `result.json` 会记录每一项的来源、证据时间窗、置信度和二分类映射。

例如，指针类证据应在 `measurement_1/2` 搜索；记录纸证据应在 `recording_1/2` 搜索；整理动作应在 `material_cleanup` 或受控末段扫描中搜索。动作标签本身不证明任何 Rubric 通过或失败。

## 环境

- Python 3.10+
- OpenCV
- NumPy
- OpenAI-compatible Python client

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

模型连接信息只从环境变量读取。代码没有默认服务地址：

```powershell
$env:QWEN_API_BASE_URL = "<your-openai-compatible-base-url>"
$env:QWEN_API_TOKEN = "<your-token>"
$env:QWEN_MODEL = "<your-model-name>"
```

不要把真实值写入 `.env.example` 或提交到 Git。

## 快速开始

将获得授权且已匿名化的视频放在本地 `data/videos/`。该目录被 Git 忽略。

### 1. 建立全视频时间清单

```powershell
python scripts/filter_redundant_video_frames.py `
  data/videos/sample_001.mp4 `
  --output-dir outputs/marker_filter
```

### 2. 锁定有效实验边界

```powershell
python scripts/qwen_experiment_segment_judge.py `
  --input-dir outputs/marker_filter `
  --output-dir outputs/experiment_boundary `
  --prompt-profile voltmeter_resistance `
  --timestamp-watermark
```

模型只选择匿名帧 ID；时间戳由本地代码映射。请求失败时脚本保存脱敏错误并返回非零退出码。

### 3. 执行推荐的七阶段动作分割 v2

v2 使用重叠时间窗 Map、全局 Reduce、本地单调状态机和边界复核。默认启用 `local_partial` 恢复：局部冲突只隔离冲突事件；首个可信完成态整理会锁定实验终点，后续动作写入 `ignored_noise_events`。

```powershell
python scripts/qwen_experiment_action_hierarchical_v2.py `
  --segment-source outputs/experiment_boundary/summary.json `
  --schema configs/action_schemas/resistance_7stage_no_battery_v2.json `
  --output-root outputs/qwen_experiment_action_hierarchical_v2
```

`qwen_experiment_action_minute.py`、`qwen_experiment_action_minute_merge.py`、`qwen_experiment_action_segment.py` 和 `qwen_experiment_action_stepwise.py` 保留为经典对照与替换实现。

### 4. Rubric 独立取证

整理动作证据包：

```powershell
python scripts/build_cleanup_action_guided_v1.py `
  --summary outputs/action_minutes/action_segments_summary.json `
  --source-dir data/videos `
  --output-dir outputs/cleanup_evidence
```

记录阶段的电压表并联拓扑候选包：

```powershell
python scripts/build_voltmeter_parallel_action_guided_v1.py `
  --summary outputs/action_minutes/action_segments_summary.json `
  --source-dir data/videos `
  --output-dir outputs/voltmeter_parallel_evidence
```

每个 builder 只负责检索和构建证据包，不读取标签，也不计算学生分数。

### 5. 模型请求前预检

```powershell
python scripts/preflight_qwen_request.py `
  --manifest outputs/voltmeter_parallel_evidence/sample_001/events/event_01/event_packet_manifest.json `
  --output outputs/preflight/event_01.json
```

默认硬预算为 8 张图、10 MiB JPEG、14 MiB 预计 Base64、单图 2 MiB。预检还检查解码、SHA-256 重复、近重复和近似全幅 ROI。预检失败只代表本次请求不应发送，不能写成学生不合格。

### 6. 执行一次结构化视觉观察

通用观察入口不读取标签，也不输出分数或最终 Rubric 决策：

```powershell
python scripts/run_qwen_structured_observation.py `
  --manifest outputs/voltmeter_parallel_evidence/sample_001/events/event_01/event_packet_manifest.json `
  --rubric-config configs/rubrics/structured_observation.example.json `
  --output outputs/observations/event_01.json
```

它会在请求前强制运行同一套媒体预检，每次命令最多发送一次请求。该入口只生成观察，不直接评分；评分器使用 Rubric 的 tie-break 规则将观察收敛为 `pass` 或 `fail`。

### 7. 核验第一组纸面记录与仪表读数

先在匿名配置中填写第一轮书写前的连续时间点、双表 ROI 和同一清晰帧上的 A/V 精细 ROI。Qwen 不接收纸面数值：它先对连续帧形成读数共识，再对同一源帧的两块电表分别精读；程序最后才在本地将仪表读数和纸面 U1/I1 比较。

```powershell
python scripts/run_meter_record_consistency_v1.py `
  --spec-config configs/meter_record_consistency.example.json `
  --paper-summary examples/paper_records.example.json `
  --video-root data/videos `
  --output outputs/meter_record_consistency_v1
```

默认容差是所选量程约 30 个小格的 `1.25` 倍，即 3 V 量程为 `0.125 V`，0.6 A 量程为 `0.025 A`。U1 或 I1 缺失、不可读或超差时输出 `fail`。

### 8. 评价 8：断开开关后改变串联电池节数

该评分项不把“拿出电池”或一般 `circuit_rewiring` 当作通过证据。固定两节电池盒的三个抽头是 `T0 -- cell 1 -- T1 -- cell 2 -- T2`；通过条件是同一个独立 episode 内，稳定连接由 `T0-T2` 改为 `T0-T1` 或 `T1-T2`，并且换接前开关断开、换接期间没有闭合证据、换接完成后开关重新闭合。

脚本会从 `circuit_rewiring` 生成 episode，也会在重复 `recording_1` 或 `measurement_1` 之间搜索短暂间隔，避免遗漏短换线。所有端点和开关证据必须留在同一个 episode 内，不能跨时间拼接。

```powershell
python scripts/run_resistance_disconnect_battery_sequence_v1.py `
  --action-summary outputs/qwen_experiment_action_hierarchical_v2/<run-id>/summary.json `
  --roi-config configs/resistance_disconnect_battery_rois.example.json `
  --video-root data/videos `
  --output-root outputs/resistance_disconnect_battery_sequence_v1
```

Qwen 只返回帧级的端点、直接接触和开关观察；`resistance_disconnect_battery_sequence_core.py` 在本地生成最终 `pass/fail`。离线重放可通过 `--observations-root` 提供已保存的 observations，不再调用模型。

### 9. 冻结预测工件

```powershell
python scripts/freeze_artifact.py `
  --input outputs/predictions/prediction_manifest.json `
  --output outputs/predictions/prediction_manifest.freeze.json
```

只有冻结并记录 SHA-256 后，才能在项目外部读取标签进行离线比较。留出集标签不得进入检索、提示词或规则调整。

## 二分类约定

评分层主结果固定为：

```json
{
  "decision": "pass",
  "predicted_score": 1,
  "confidence": 0.78,
  "evidence_quality": "medium"
}
```

证据不完整时先扩大时间窗、使用相邻帧、增强 ROI，再按 Rubric 预设的冲突规则选择最可能类别。观察层仍可记录 `uncertain`，但不能把它直接作为评分层第三类。

`configs/pipeline/evidence_artifact_contract_v1.json` 与对应校验测试保留旧版 `abstained` 工件兼容性；它只用于重放历史工件，当前评分结果不得走该分支。

## 测试

```powershell
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
```

测试不读取视频、不调用网络，也不需要 Qwen 凭据。

## 仓库边界

公开仓库只包含通用代码、配置、文档和匿名 JSON 示例。数据治理要求见 [docs/privacy.md](docs/privacy.md)，架构与工件关系见 [docs/architecture.md](docs/architecture.md)。

## License

[MIT](LICENSE)
