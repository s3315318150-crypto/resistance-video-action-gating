from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_cleanup_action_guided_v1 as cleanup_builder  # noqa: E402
import build_voltmeter_parallel_action_guided_v1 as voltmeter_builder  # noqa: E402
from qwen_hierarchical_v1_prompts import (  # noqa: E402
    build_boundary_prompt,
    build_map_prompt,
    build_reduce_prompt,
)


class PublicReleasePrivacyTests(unittest.TestCase):
    def test_private_source_name_is_not_inserted_into_qwen_prompts(self) -> None:
        source_name = "student-private-name_phy-resistance.mp4"
        frames = [{"image_id": "frame_00000001"}, {"image_id": "frame_00000002"}]
        map_prompt = build_map_prompt(
            source_name,
            {"window_id": "window_0001", "window_seconds": [0.0, 2.0]},
            frames,
        )
        reduce_prompt = build_reduce_prompt(
            source_name,
            [
                {
                    "event_id": "evt_0001",
                    "action_type": "wiring_action",
                    "first_frame_id": "frame_00000001",
                    "last_frame_id": "frame_00000002",
                    "representative_frame_id": "frame_00000001",
                    "first_seconds": 0.0,
                    "last_seconds": 2.0,
                    "evidence": "visible wiring",
                    "confidence": 0.8,
                }
            ],
        )
        boundary_prompt = build_boundary_prompt(
            source_name,
            {
                "boundary_id": "boundary_0001",
                "from_stage": "circuit_wiring",
                "to_stage": "measurement_1",
            },
            frames,
            {"circuit_wiring": "连接电路", "measurement_1": "第一次测量"},
        )

        for prompt in (map_prompt, reduce_prompt, boundary_prompt):
            self.assertNotIn(source_name, prompt)
            self.assertNotIn("student-private-name", prompt)
            self.assertIn("匿名", prompt)

    def test_evidence_builders_accept_arbitrary_supported_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "anonymous-input.MP4"
            expected.touch()
            (root / "another-video.mov").touch()
            (root / "notes.txt").touch()

            for module in (cleanup_builder, voltmeter_builder):
                candidates = module.source_video_candidates(root)
                resolved, method = module.find_source_video(expected.name, candidates)
                self.assertEqual(expected.resolve(), resolved)
                self.assertEqual("exact_source_video_id", method)
                self.assertEqual(2, len(candidates))


if __name__ == "__main__":
    unittest.main()
