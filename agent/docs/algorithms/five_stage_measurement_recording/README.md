# 五阶段测量记录分段

## 目标

七阶段中的 `measurement_1/2` 与 `recording_1/2` 在真实视频里经常连续或重叠。当前 Agent 因此使用五阶段：

1. `circuit_wiring`
2. `recording_1`，语义为第一轮“测量 + 记录”
3. `circuit_rewiring`
4. `recording_2`，语义为第二轮“测量 + 记录”
5. `material_cleanup`

`recording_1/2` 是兼容 ID，不能只按字面理解为写字。每个合并阶段都带有：

- `stage_semantics=measurement_and_recording_cycle`
- `measurement_subintervals`
- `writing_subintervals`
- `contains_measurement_evidence`
- `contains_writing_evidence`

只有模型本次直接观察到相应子动作时，子区间和 `contains_*` 才会声明证据；合并阶段本身不伪造未观察到的测量动作。

## Fresh 数据流

```text
当前视频
  -> marker filter
  -> 当前 run 的实验起止检测
  -> 60 秒重叠 Map 窗口，2 秒抽样
  -> Qwen 只输出 wiring / measurement / writing / cleanup 基础动作
  -> 本地五阶段时序归约
  -> 1 fps 边界首轮复核
  -> 需要时 0.5 秒邻帧精修
  -> 当前 run 的五阶段摘要
  -> plan_live_skills
  -> Rubric 专项抽帧 Agent
```

正式入口是 `run_full_pipeline`。历史 `run_action_segmentation()` 只保留为未注册 helper，不在 MCP schema 或 registry 中。正式 Pipeline 的动作阶段读取本次 Pipeline 刚生成的 `experiment_boundary/summary.json`，不读取 `config.json` 中 legacy helper 的 `segment_source`。

### 粗定位与 canonical 阶段的边界

实验起止检测和粗粒度边界判定只负责给当前 run 划出“在哪一段视频取证”。粗定位实现可能使用 `battery_change` 等兼容标签，但这些标签不会直接选择 Skill、ROI、阈值或结论。动作 Map/Reduce 重新依据当前帧中可见的基础动作归一化阶段：重复接线才形成 `circuit_rewiring`，测量与书写才合并到对应 `recording_1/2`。下游只消费归一化后的五阶段和其子区间；旧标签不会从历史摘要回填当前 run。

## Skill 消费方式

- R1/R3/R8 使用 `circuit_wiring` 和 `circuit_rewiring`。
- R5/R6 优先消费合并阶段内的 `measurement_subintervals`；缺失时在同一合并周期做统一 `broad_search`，Router 不会仅因 `recording_1/2` 容器存在就选择 `explicit_measurement`。
- R0 使用 `material_cleanup`。
- Skill Router 按当前 run 实际观察到的阶段和轮次选择参数，不按视频 ID、文件名、哈希或历史结果路由。

## 真实视频 8 验证

2026-08-18 对视频 8 的只读硬链接执行 fresh Pipeline，输出位于：

```text
agent/runs/<run-id>/
```

实际结果：

| 阶段 | 时间窗（秒） | 直接基础动作 |
|---|---:|---|
| `circuit_wiring` | `0.0-106.0` | `wiring_action` |
| `recording_1` | `108.0-114.0` | `writing_action` |
| `circuit_rewiring` | `116.0-122.0` | `wiring_action` |
| `recording_2` | `124.0-138.0` | `writing_action` |
| `material_cleanup` | `164.0-184.133` | `cleanup_action` |

边界精修保留了全部五个源阶段，生成 4 个边界并调用 Qwen 4 次。四个精修边界分别为 `106.5-107.0`、`114.5-115.0`、`123.0-123.5`、`164.0-164.5` 秒。

这次 Map 没有直接输出 `measurement_action`，所以两个合并周期的 `measurement_subintervals` 为空。下游会把它们作为当前周期的搜索范围并启动统一宽搜索，但不会把本次结果表述为“测量动作已直接检出”，也不会因此提高显式测量 Skill 的证据等级。

## 已验证命令

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
# Ran 256 tests ... OK

..\.venv\Scripts\python.exe -m unittest discover `
  -s ..\workflow\v2\tests `
  -p "test_qwen_hierarchical_five_stage.py"
# Ran 4 tests ... OK

..\.venv\Scripts\python.exe -m unittest discover `
  -s ..\workflow\v2\tests `
  -p "test_refine_v2_temporal_guard_boundaries_rubric.py"
# Ran 7 tests ... OK
```

这些结果验证的是实现与开发视频回归，不是新视频泛化准确率。泛化准确率必须在算法冻结后用未参与开发的新视频评测。
