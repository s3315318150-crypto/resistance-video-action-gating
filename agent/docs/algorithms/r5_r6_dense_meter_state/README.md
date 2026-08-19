# Agent R5/R6: Dynamic Meter State and Printed Tick Evidence

This document describes the Agent implementation of Rubrics 5 and 6. It is a
current-run evidence chain, not a replay of the five development videos.

## What changed

The previous Agent path relied mainly on Qwen observations and the closed
stable-state reducer. The current path adds a deterministic CPU evidence
layer:

```text
current measurement frames
  -> SIFT face localization on every frame
  -> perspective normalization to 640 x 520
  -> A/V role confirmation
  -> red lead and adjacent black-edge masking
  -> CLAHE, Canny and Hough pointer candidates
  -> scale-arc to dynamic-hub geometry check
  -> printed tick endpoints on each accepted frame
  -> adjacent-frame endpoint consensus
  -> thirty-division reading and range assessment
  -> R5/R6 binary fusion
```

The shared device references are in
`agent/assets/meter_calibration/`. They contain no student, video or Excel
metadata. Every input frame is localized again; no video-number ROI is used.

## Implementation map

R5/R6 is a composed execution path rather than one standalone file:

| Layer | Repository file | Role |
|---|---|---|
| Agent orchestration | `agent/resistance_agent/meter_rubrics.py` | Select current-run evidence, request Qwen observations, invoke CPU and R6 reducers, and fuse the final binary results |
| R6 geometry reducer | `agent/resistance_agent/skills/closed_stable_r6_cv_v3.py` | Classify zero, reverse, normal, full-scale and overrange states, then emit R6 `pass` or `fail` |
| Current-video stage producer | `agent/resistance_agent/skills/closed_stable_stage_producer.py` | Build search windows from current-run stages and invoke an optional repository-local OpenCV producer |
| CPU tick fusion | `agent/resistance_agent/skills/cpu_tick_meter_reading.py` | Locate printed ticks, apply the thirty-division conversion, recognize reverse/overrange evidence and fuse direct CPU evidence into R5/R6 |
| OpenCV core | `agent/resistance_agent/skills/r5_r6_dense_meter_state/` | Face matching, perspective normalization, wire masking, arc-to-hub pointer geometry and multi-frame tick consensus |

The private development workspace used a separate coarse-to-fine OpenCV
entry point with a one-second coarse scan and 0.1-second dense search. The
public release does not retain that machine-specific absolute path. Runtime
components and calibration assets used by the published path are packaged in
the repository and referenced relatively.

## Face localization

The CPU locator extracts SIFT features from the current frame, applies a Lowe
ratio match against the anonymous ammeter and voltmeter templates, and uses a
RANSAC homography to rectify the candidate face. The face is normalized to
`640 x 520`. The role glyph and dial structure reject a wrong A/V template.
Terminal occupancy is measured from the current frame and the shared terminal
layout; it is not copied from an old run.

## Pointer evidence

Before line detection, HSV red components are dilated and adjacent dark
components are included only when they touch a retained red component. The
mask is an occlusion mask; pixels are never inpainted. A pointer candidate is
accepted only when a dark line reaches both the upper scale arc and the lower
dynamic hub zone. Lines trapped in the central A/V glyph, short text strokes,
wire-parallel edges and ambiguous alternatives are rejected.

The detector uses CLAHE, a small Gaussian blur, Canny edges and HoughLinesP.
Candidate angle and hub intersection are computed from the current rectified
face, so camera movement does not reuse a fixed pixel pivot.

## Multi-frame printed grid

For each accepted pointer frame, `scale_tick_grid_v1.py` detects the visible
printed scale endpoints and maps the pointer angle to the grid. The batch
reducer rejects endpoint outliers with a median/MAD rule, then keeps the
consensus endpoint pair and median pointer position. The scale has thirty
small divisions:

```text
reading = nearest_tick * selected_range_max / 30
```

The ratio is deliberately not clipped before state classification. A negative
ratio is reverse deflection; a ratio above one is an overrange candidate. Both
states keep `reading: null` rather than fabricating a clipped number.

## R5 and R6 fusion

R5 asks whether the energized measurement shows a normal positive pointer
deflection. Explicit CPU `reverse` or `overrange` evidence overrides a weak
pass. A CPU `normal_rightward` result can repair only the known weak failure
reason `no_normal_pointer_deflection_found_after_temporal_and_roi_search`.

R6 asks whether the stable pointer/range state is suitable. CPU `too_low` or
`too_high` range evidence produces `fail`; direct `appropriate` evidence can
repair only the weak missing-range reason. The Qwen and existing current-run
reducer remain in the report as parallel observations. CPU evidence does not
introduce `uncertain`, `needs_review` or another primary class.

Every report records the current image group, frame number, timestamp, dynamic
face/ROI paths, pointer geometry, printed-grid result and confidence. The
audit fields are always:

```json
{
  "selection_basis": "current_run_active_measurement_frames_only",
  "video_id_used_for_routing": false,
  "historical_artifacts_used": false,
  "fixed_video_roi_used": false
}
```

## Running and testing

The normal Agent entry point remains:

```powershell
python agent\run_agent.py `
  --scheduler deterministic `
  --mode execute `
  --video-ref data\videos\sample.mp4 `
  --run-id sample_execute
```

Focused CPU tests:

```powershell
$env:PYTHONPATH = "agent"
python -m unittest discover -s agent\tests -p "test_cpu_tick_meter_reading.py" -v
```

The development-set examples from the original experiment remain regression
evidence only. They are not a claim of accuracy on unseen videos.
