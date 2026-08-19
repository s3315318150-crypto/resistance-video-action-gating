# R3 泛化压力测试与一键盲测

## 目标

该工具只评估 R3 的身份无关性、时间稳定性、画质稳定性和抽帧效率，不把开发视频
或跨实验视频的结果报告为新视频泛化准确率。所有测试只读取视频副本和当前 run 阶段，
不读取 Excel、人工真值、历史时间窗、固定 ROI 或历史最佳判断。

主要入口：

- `scripts/run_r3_stress_suite.py`：身份、时间、画质和六视频汇总；
- `scripts/run_r3_blind_execute.py`：新视频一键执行并在 Excel 访问前冻结 R3；
- `resistance_agent/r3_stress_suite.py`：变体生成、执行和统计实现；
- `tests/test_r3_stress_suite.py`：相位、边界、画质、身份和汇总测试。

## 身份无关性

工具复制当前视频，使用随机匿名文件名，并把直接阶段摘要中的
`source_video_id` 同步改为匿名文件名。原文件和匿名副本分别运行相同 R3 参数，比较：

- `decision`、`predicted_score`；
- 候选时间窗；
- 基线与最终抽帧数；
- 开关观察数；
- 补帧请求数和新增帧数。

视频内容 SHA-256 必须相同，任一比较字段变化都会使 `passed=false`。

```powershell
.venv\Scripts\python.exe agent\scripts\run_r3_stress_suite.py identity `
  --video "<current-run-video>" `
  --stage-summary "<current-run-boundary-summary>" `
  --output-dir "agent\outputs\r3_stress\identity"
```

## 时间鲁棒性

相位测试为 OpenCV 5 fps 采样器增加独立 `sampling_phase_offset_seconds`。默认正式
execute 仍为 `0.0s`。5 fps 周期为 `0.2s`，所以 `-0.1s` 与 `+0.1s` 在阶段内部会
归一化为同一相位；保留两项是为了显式验证周期等价性。

边界测试将所有当前接线阶段整体平移 `-5s`、`-2s`、`+2s`、`+5s`，并限制在视频
时长内。它不加载历史窗口，也不修改原阶段文件。

```powershell
.venv\Scripts\python.exe agent\scripts\run_r3_stress_suite.py temporal `
  --video "<current-run-video>" `
  --stage-summary "<current-run-boundary-summary>" `
  --output-dir "agent\outputs\r3_stress\temporal"
```

## 画质压力测试

OpenCV 在独立目录生成以下完整视频副本：

- `1080p`、`720p`；
- `blur`：`5x5` 高斯模糊；
- `brightness`：像素亮度增加 `25`；
- `recompress`：使用 `mp4v` 重新编码。

帧率、帧数和时长保持不变，音频不保留。变体只用于视觉 R3 测试，原视频 SHA-256
在前后都必须一致。

```powershell
.venv\Scripts\python.exe agent\scripts\run_r3_stress_suite.py quality `
  --video "<current-run-video>" `
  --stage-summary "<current-run-boundary-summary>" `
  --variant 720p --variant blur `
  --output-dir "agent\outputs\r3_stress\quality"
```

先只生成视频而不执行 R3：增加 `--generate-only`。

## 固定 5 fps 与 Agent 汇总

每个 Agent 报告已保存本次运行的 `baseline_report_path`，因此汇总不会重新分析视频，
也不会读取 Excel。

```powershell
.venv\Scripts\python.exe agent\scripts\run_r3_stress_suite.py aggregate `
  --agent-report "8=<agent-report-path>" `
  --agent-report "16=<agent-report-path>" `
  --output-dir "agent\outputs\r3_stress\all_videos"
```

输出 `summary.json` 和 `summary.csv`，字段包括基线/最终抽帧数、开关观察数、覆盖率、
补帧请求数和判定变化。报告固定记录 `accuracy_claimed=false`。

## 一键盲测

```powershell
.venv\Scripts\python.exe agent\scripts\run_r3_blind_execute.py `
  --video-ref "<new-video-path-or-id>" `
  --run-id "blind_r3_<timestamp>" `
  --config "agent\config.json"
```

执行顺序固定为：

```text
inspect_video
-> create_run
-> run_full_pipeline
-> refine_rubric_boundaries
-> plan_live_skills
-> run_switch_rubric
-> freeze_prediction
```

冻结文件为 `runs/<run>/frozen_r3/predictions_frozen.json`。脚本不接受 Excel 路径，
并固定输出 `excel_read=false`、`ground_truth_read=false`。
