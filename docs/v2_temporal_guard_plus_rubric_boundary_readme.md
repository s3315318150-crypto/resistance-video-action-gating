# v2 Temporal Guard + Rubric 边界精修算法

## 1. 算法名称

中文名：**伏安法七阶段 v2 时序保护 + Rubric 边界精修算法**

算法 ID：`v2-temporal-guard-rubric-boundary`

通用入口：`scripts/refine_v2_temporal_guard_boundaries_rubric.py`

该算法是 v2 分段后的第二阶段处理。它保留 v2 Temporal Guard 已经识别出的阶段、阶段数量和时间段，只在相邻阶段的交界附近抽取更密集的候选帧，让原 v2 边界提示词给出可复核的阶段切换时间。

## 2. 适用范围

适合以下场景：

- v2 已经识别出连线、测量、记录、重新连线和整理等阶段；
- 需要把约 2 秒粒度的粗边界细化到 0.5 秒候选帧；
- 需要保存阶段切换前后帧、Qwen 原文和置信度；
- 不希望边界精修改变 Reduce 和状态机已经确定的阶段顺序。

该算法不会：

- 从零完成七阶段分段；
- 发现 v2 完全漏掉的新阶段；
- 删除、增加或重命名 v2 阶段；
- 把精修时间自动覆盖回原 v2 `observed_stage_runs`。

## 3. 输入

`--action-summary` 接受两种结构：

1. 正常运行生成的 `qwen_experiment_action_hierarchical_v2_temporal_guard/summary.json`；
2. 离线确定性重放生成的 `stored_successful_map_and_reduce_temporal_guard_replay` summary。

每个视频必须能解析出：

- `source_manifest` 和原视频元数据；
- `observed_stage_intervals`；
- `observed_stage_runs`；
- 锁定的实验开始和结束时间。

运行时代码不读取姓名、固定视频 ID、五视频时间或黄金标注。黄金夹具只存在于独立 A/B 评测脚本中。

## 4. 处理流程

```text
Temporal Guard v2 已有分段
    ↓
由 observed_stage_intervals 构建相邻阶段边界
    ↓
十项 Rubric 配置生成相关候选证据窗口
    ↓
在粗边界前后 10 秒内选择距离最近的 Rubric 候选中心
    ↓
截取中心前后各 3 秒，共约 6 秒
    ↓
每 0.5 秒抽一帧，最大边长 640px
    ↓
调用原 v2 build_boundary_prompt
    ↓
校验 last_from_frame_id 和 first_to_frame_id
    ↓
输出 refined_boundaries；原 v2 阶段时间段保持不变
```

候选检索使用 Rubric `0、3、5、7、9`：

- `0`：最终整理和归位；
- `3`：接线期间开关及手部操作；
- `5`：测量和读表；
- `7`：第一次记录；
- `9`：第二次记录。

## 5. 运行方法

先配置 Qwen：

```powershell
$env:QWEN_API_BASE_URL = "https://cossin.ecnu.edu.cn/skill/api/qwen/v1"
$env:QWEN_API_TOKEN = "EMPTY"
$env:QWEN_MODEL = "qwen"
```

对任意 Temporal Guard v2 批次运行：

```powershell
python scripts/refine_v2_temporal_guard_boundaries_rubric.py `
  --action-summary outputs/qwen_experiment_action_hierarchical_v2_temporal_guard/<run-id>/summary.json `
  --output-root outputs/v2_temporal_guard_rubric_boundary `
  --run-id <new-run-id>
```

只准备图片和提示词，不调用 Qwen：

```powershell
python scripts/refine_v2_temporal_guard_boundaries_rubric.py `
  --action-summary <summary.json> `
  --output-root <output-dir> `
  --run-id <new-run-id> `
  --prepare-only
```

输出目录必须是新的，程序拒绝覆盖已有运行。

## 6. 输出字段

批次 `summary.json` 包含：

- `source_stage_runs_unchanged=true`；
- `golden_fixture_used=false`；
- `video_count`、`boundary_count` 和 `qwen_call_count`；
- 每个视频的 `result_path`。

每个视频 `result.json` 包含：

- `source_observed_stage_runs`：原 v2 时间段，原样保存；
- `source_observed_stage_intervals`：原 v2 事件区间；
- `refined_boundaries`：Rubric 精修建议；
- `retrieval_traces`：候选范围、候选数量和采用来源；
- `rubric_retrieval_plan`：十项 Rubric 的完整候选计划；
- `rejected_refined_boundaries`：不满足全局时间单调性的结果。

若 Qwen 返回无效 JSON、非法帧号或低于契约要求的结果，程序保留 v2 粗边界并标记 `needs_review=true`，不会编造新时间。

## 7. 五视频在线 A/B

确认基线为用户指定的 Temporal Guard v2，恢复事件数为 `[16, 0, 0, 0, 2]`。在任何 Qwen 调用前，评测脚本逐视频检查阶段名称、数量和时间与本地黄金夹具完全一致。

| 方法 | 保持 v2 阶段时间段 | 黄金边界 ±2 秒 | 边界 MAE | Qwen 调用 | 图片输入 |
|---|---:|---:|---:|---:|---:|
| Rubric 边界精修 | 5/5 | **19/20** | **1.125 秒** | 20 | 260 |
| Yes/No 边界实验 | 5/5 | 14/20 | 2.025 秒 | 41 | 510 |

Rubric 唯一超过 ±2 秒的边界是 `sample_002` 的“连线 → 第一次记录”：v2 为 `05:24`，Rubric 建议为 `05:21.5`，提前 `2.5 秒`。

这里的 `5/5` 表示原 v2 阶段时间段被只读保留，不表示 Rubric 独立从零分段达到 5/5。`19/20` 才是本次边界建议与 v2 黄金边界的比较结果。

## 8. 成本

五视频原 v2 在线识别共记录 `74` 次 Qwen 调用：Map `41` 次、Reduce `7` 次、边界细化 `26` 次。Temporal Guard 离线重放不调用 Qwen。

Rubric 精修在已有 v2 结果上新增 `20` 次调用，每个相邻阶段转换通常调用一次；五视频共输入 `260` 张边界帧。完整从头运行的实测调用量约为 `74 + 20 = 94` 次。

## 9. 验证与限制

正式 A/B 评测入口：

`experiments/night_exploration_20260812/scripts/v2_temporal_guard_boundary_retrieval_ab_v4.py`

评测结果保存在本地忽略目录：

`outputs/v2_tg_bo_ab_20260812/comparison.json`

当前结果证明 Rubric 检索能为正确的 v2 粗分段提供密集边界证据；尚未证明它比独立人工标注更准确，也没有证明它能替代 v2 完成从零分段。
