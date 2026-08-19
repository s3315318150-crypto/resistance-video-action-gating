from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPT = EXPERIMENT_ROOT / "scripts" / "v2_retrieval_only_ab_v1.py"
SPEC = importlib.util.spec_from_file_location("v2_retrieval_only_ab_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class V2RetrievalOnlyABTests(unittest.TestCase):
    def test_mature_pipeline_functions_are_bound_by_identity(self) -> None:
        MODULE.bind_mature_v2_pipeline()
        self.assertIs(MODULE.engine.salvage_reduce_response, MODULE.mature_reduce.salvage_reduce_response)
        self.assertIs(MODULE.engine.select_events, MODULE.mature_reduce.select_events)
        self.assertIs(MODULE.engine.assign_seven_stages, MODULE.mature_reduce.assign_seven_stages)
        self.assertIs(MODULE.engine.build_map_prompt, MODULE.mature_prompts.build_map_prompt)

    def test_source_bindings_have_real_hashes(self) -> None:
        bindings = MODULE.source_bindings()
        self.assertIn("state_machine_and_event_reducer", bindings)
        for binding in bindings.values():
            self.assertTrue(Path(binding["path"]).is_file())
            self.assertRegex(binding["sha256"], r"^[0-9a-f]{64}$")

    def test_saved_v2_source_result_is_the_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_result = root / "source.json"
            source_result.write_text(
                json.dumps({"observed_stage_runs": [{"stage": "old"}]}),
                encoding="utf-8",
            )
            summary = {
                "records": [{
                    "source_video_id": "anonymous.mp4",
                    "source_result": str(source_result),
                }]
            }
            item = MODULE.baseline_sources(summary)[0]
            self.assertEqual("old", item["baseline"]["observed_stage_runs"][0]["stage"])
            self.assertEqual(source_result.resolve(), item["reference_result"])

    def test_selector_validator_uses_engine_image_ids(self) -> None:
        clips = [{"clip_id": "c1", "frames": [{"image_id": "frame_0001"}]}]
        self.assertTrue(MODULE.validate_selector_response({
            "clips": [{
                "clip_id": "c1",
                "answer": "yes",
                "target_probability": 0.8,
                "selected_frame_id": "frame_0001",
            }]
        }, clips))
        self.assertFalse(MODULE.validate_selector_response({
            "clips": [{
                "clip_id": "c1",
                "answer": "yes",
                "target_probability": 0.8,
                "selected_frame_id": "invented",
            }]
        }, clips))

    def test_supplemental_window_calls_original_map_prompt(self) -> None:
        prepared = {
            "fixed_start": 0.0,
            "fixed_end": 20.0,
            "video_id": "anonymous.mp4",
            "video_dir": Path("unused"),
            "prepared_windows": [],
        }
        frames = [
            {"image_id": "f1", "timestamp_seconds": 2.0},
            {"image_id": "f2", "timestamp_seconds": 4.0},
        ]
        args = MODULE.default_engine_args()
        with tempfile.TemporaryDirectory() as temporary:
            prepared["video_dir"] = Path(temporary)
            with mock.patch.object(MODULE, "frames_for_times", return_value=frames), mock.patch.object(
                MODULE.mature_prompts,
                "build_map_prompt",
                wraps=MODULE.mature_prompts.build_map_prompt,
            ) as prompt_builder:
                windows = MODULE.make_map_windows(
                    prepared,
                    [{"start_seconds": 2.0, "end_seconds": 4.0, "source": "test"}],
                    "test",
                    args,
                )
            self.assertEqual(1, len(windows))
            prompt_builder.assert_called_once()
            prompt = next(Path(temporary).rglob("prompt.txt")).read_text(encoding="utf-8")
            self.assertIn("分层算法的 Map 步骤", prompt)
            self.assertIn("本任务没有“换电池”基础动作或阶段", prompt)

    def test_gold_is_only_used_by_comparator(self) -> None:
        gold_marker = "GOLD_ONLY_478_491"
        prompt = MODULE.selector_prompt(
            [{"clip_id": "c1", "frames": [{"image_id": "f1"}]}],
            "cleanup_action",
        )
        self.assertNotIn(gold_marker, prompt)
        comparison = MODULE.compare_runs(
            [["material_cleanup", 478.0, 491.0]],
            [["material_cleanup", 478.0, 491.0]],
        )
        self.assertTrue(comparison["within_2_seconds"])


if __name__ == "__main__":
    unittest.main()
