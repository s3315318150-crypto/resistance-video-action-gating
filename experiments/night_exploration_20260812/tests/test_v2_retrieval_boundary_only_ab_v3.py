from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v2_retrieval_boundary_only_ab_v3.py"
SPEC = importlib.util.spec_from_file_location("v2_retrieval_boundary_only_ab_v3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class BoundaryOnlyRetrievalTests(unittest.TestCase):
    def test_golden_boundaries_collapse_consecutive_same_stage_runs(self) -> None:
        result = MODULE.golden_boundaries([
            ["circuit_wiring", 0.0, 10.0],
            ["circuit_wiring", 20.0, 30.0],
            ["recording_1", 40.0, 50.0],
        ])
        self.assertEqual(1, len(result))
        self.assertEqual(40.0, result[0]["selected_seconds"])

    def test_boundary_comparison_matches_transition_without_video_identity(self) -> None:
        result = MODULE.compare_boundaries(
            [{
                "boundary_id": "b001",
                "from_stage": "circuit_wiring",
                "to_stage": "recording_1",
                "selected_seconds": 40.5,
            }],
            [{
                "from_stage": "circuit_wiring",
                "to_stage": "recording_1",
                "selected_seconds": 40.0,
            }],
        )
        self.assertEqual(1, result["within_2_seconds_count"])
        self.assertEqual(0.5, result["mean_absolute_error_seconds"])

    def test_rubric_range_is_bounded_and_uses_nearest_candidate(self) -> None:
        prepared = {"fixed_start": 0.0, "fixed_end": 100.0}
        boundary = {"coarse_selected_seconds": 50.0}
        start, end, trace = MODULE.rubric_boundary_range(
            prepared,
            boundary,
            [{
                "start_seconds": 40.0,
                "end_seconds": 60.0,
                "planned_times": [42.0, 49.5, 58.0],
            }],
        )
        self.assertEqual(46.5, start)
        self.assertEqual(52.5, end)
        self.assertEqual(49.5, trace["selected_center_seconds"])

    def test_original_boundary_prompt_is_the_imported_builder(self) -> None:
        self.assertIs(
            MODULE.base.engine.build_boundary_prompt,
            MODULE.base.mature_prompts.build_boundary_prompt,
        )


if __name__ == "__main__":
    unittest.main()
