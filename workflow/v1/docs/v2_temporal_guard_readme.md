# v2 Temporal Guard

`qwen_experiment_action_hierarchical_v2_temporal_guard.py` 是独立的 v2 实验入口。它不修改或覆盖：

- `qwen_experiment_action_hierarchical_v2.py`
- `qwen_hierarchical_v1_reduce.py`
- `resistance_7stage_no_battery_v2.json`
- 任何已有 v2 输出目录

## 解决的问题

原 v2 的 Map 会产生按时间排列的直接视觉事件，但全局 Qwen Reduce 可以用 `duplicate` 或 `conflicts_with_stronger_evidence` 拒绝任意事件。JSON 契约只检查 ID 和结构是否完整，不验证被拒绝事件与所谓“更强事件”是否真的重叠。因此，后面的拆线或整理可能错误删除前面已经观察到的接线、书写和重新连线。

本版本在原 `salvage_reduce_response()` 之后增加本地时序保护：

1. `duplicate` 只有在被拒绝事件与已接受的同类事件源帧重叠时才合法。
2. `conflicts_with_stronger_evidence` 只有在被拒绝事件与已接受的异类事件源帧重叠时才合法。
3. 没有合法重叠见证事件的拒绝会被恢复，并记录 `non_overlapping_rejection_restored`。
4. 最终整理之前的事件可以恢复；最终整理开始后仍保持原终态截断，不恢复录像冗余。
5. 恢复后的真实重叠冲突继续交给原 `resolve_accepted_conflicts()`，不会同时保留同一时刻互斥的动作。

输出诊断字段：

```json
{
  "temporal_guard": {
    "policy": "restore_preterminal_duplicate_or_conflict_rejection_without_overlapping_accepted_witness",
    "restored_event_ids": ["evt_0002"],
    "restored_event_count": 1,
    "terminal_cleanup_event_id": "evt_0007"
  }
}
```

## 单独运行

```powershell
python scripts/qwen_experiment_action_hierarchical_v2_temporal_guard.py `
  --segment-source outputs/experiment_boundary/summary.json `
  --schema configs/action_schemas/resistance_7stage_no_battery_v2.json `
  --output-root outputs/qwen_experiment_action_hierarchical_v2_temporal_guard
```

## 一键流水线

```powershell
python scripts/run_resistance_pipeline.py `
  --video-dir data/videos `
  --output-root outputs/resistance_pipeline `
  --action-version v2-temporal-guard
```

默认 `--action-version` 仍为 `v2`。只有显式选择 `v2-temporal-guard` 才启用新规则。
