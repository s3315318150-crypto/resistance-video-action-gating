from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qwen_hierarchical_v3_sampling import motion_consistency_score, scan_activity, select_timestamps  # noqa: E402


class HierarchicalV3SamplingTests(unittest.TestCase):
    def test_fixed_anchors_remain_and_activity_peak_gets_extra_frame(self) -> None:
        activity = [
            {"timestamp_seconds": float(index) / 2.0, "activity_score": 1.0 if index == 15 else 0.01}
            for index in range(41)
        ]
        selected = select_timestamps(0.0, 20.0, 7, activity)
        self.assertEqual(7, len(selected))
        for anchor in (0.0, 5.0, 10.0, 15.0, 20.0):
            self.assertIn(anchor, selected)
        self.assertIn(7.5, selected)

    def test_budget_is_never_exceeded_when_anchors_outnumber_budget(self) -> None:
        selected = select_timestamps(0.0, 60.0, 6, [])
        self.assertLessEqual(len(selected), 6)
        self.assertEqual(0.0, selected[0])
        self.assertEqual(60.0, selected[-1])

    def test_motion_consistency_is_diagnostic_and_action_specific(self) -> None:
        self.assertGreater(motion_consistency_score("cleanup_action", 0.8), motion_consistency_score("measurement_action", 0.8))
        self.assertGreaterEqual(motion_consistency_score("uncertain", 0.5), 0.99)

    def test_activity_scan_reads_forward_and_detects_visual_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "activity.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
            self.assertTrue(writer.isOpened())
            for index in range(20):
                value = 0 if index < 10 else 255
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            writer.release()
            samples = scan_activity(path, 0.0, 1.9, 0.5)
        self.assertGreaterEqual(len(samples), 4)
        self.assertGreater(max(item["activity_score"] for item in samples), 0.5)


if __name__ == "__main__":
    unittest.main()
