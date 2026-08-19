# R5/R6 AI Context

This is the navigation file for the Agent R5/R6 production path. Read it
before changing the meter reducer, then read `README.md` for the algorithm
details.

## Production path

```text
agent/resistance_agent/toolkit.py
  -> resistance_agent/meter_rubrics.py
  -> current-run Qwen observations
  -> closed_stable_stage_producer.py (optional current-run four-stage search)
  -> closed_stable_r6_cv_v3.py (R6 geometric state reducer)
  -> cpu_tick_meter_reading.py (printed-grid reading and R5/R6 fusion)
  -> repository-local OpenCV modules
  -> pass/fail fusion
  -> agent/runs/<run-id>/rubrics/rubric_5.json and rubric_6.json
```

The current algorithm version is `r56_temporal_meter_v6_cpu_tick_grid`.
The only primary outcomes are `pass` and `fail`.

## Files

| File | Responsibility |
|---|---|
| `agent/resistance_agent/meter_rubrics.py` | R5/R6 orchestration, Qwen observations, CPU reader invocation, R6 reducer invocation and final fusion |
| `agent/resistance_agent/skills/closed_stable_r6_cv_v3.py` | Zero, reverse, normal, full-scale and overrange geometry states, followed by binary R6 reduction |
| `agent/resistance_agent/skills/closed_stable_stage_producer.py` | Optional current-video stage-window production and OpenCV search; it does not load video-specific ROIs or historical windows |
| `agent/resistance_agent/skills/cpu_tick_meter_reading.py` | CPU printed-grid reading and direct binary evidence fusion |
| `agent/resistance_agent/skills/r5_r6_dense_meter_state/generic_meter_tick_batch_v4_role_glyph.py` | SIFT face matching, perspective normalization, role glyph and terminal evidence |
| `agent/resistance_agent/skills/r5_r6_dense_meter_state/wire_occlusion_black_edge_v2.py` | Red lead and adjacent black-edge occlusion mask |
| `agent/resistance_agent/skills/r5_r6_dense_meter_state/pointer_line_arc_hub_v2.py` | Scale-arc to dynamic hub pointer geometry |
| `agent/resistance_agent/skills/r5_r6_dense_meter_state/scale_tick_grid_v1.py` | Per-frame printed tick grid |
| `agent/resistance_agent/skills/r5_r6_dense_meter_state/scale_tick_grid_batch_v1.py` | Multi-frame grid endpoint consensus |
| `agent/resistance_agent/skills/r5_r6_dense_meter_state/count_meter_ticks_v1.py` | Thirty-division conversion and half-up rounding |
| `agent/assets/meter_calibration/` | Anonymous shared-device references and calibration only |

The development workspace originally invoked a coarse-to-fine OpenCV entry
point from a separate `gauge_reader_cpu_baseline` directory. The public Agent
packages the required runtime components under
`agent/resistance_agent/skills/r5_r6_dense_meter_state/` and uses only
repository-relative paths. Private absolute paths are not part of the release.

`AI_CONTEXT.md` is navigation documentation. It is not imported by the Agent
and does not implement or modify scoring behavior.

## Constraints

- Routing uses current-run stages, frames, ROIs and observations only.
- Video identity is an association field, never an algorithm, threshold, ROI,
  prompt or result selector.
- No historical prediction, Excel label, fixed video ROI or private experiment
  output is read by formal `execute`.
- A failed locator lowers evidence quality and keeps the existing binary
  reducer; it does not create a third primary class.

## Minimal checks

```powershell
$env:PYTHONPATH = "agent"
python -m unittest discover -s agent\tests -p "test_cpu_tick_meter_reading.py" -v
python -m compileall -q agent
```

When modifying the shared CPU components, also run the focused tests listed in
`README.md` and the complete Agent and Workflow V2 test suites.
