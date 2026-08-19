from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_resistance_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_resistance_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RunResistancePipelineTests(unittest.TestCase):
    def test_dry_run_discovers_arbitrary_video_and_writes_four_phase_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_dir = root / "videos"
            video_dir.mkdir()
            (video_dir / "new class sample.mp4").write_bytes(b"not-opened-in-dry-run")
            output_root = root / "outputs"

            result = MODULE.main(
                [
                    "--video-dir",
                    str(video_dir),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "anonymous_test",
                    "--dry-run",
                ]
            )
            report = json.loads(
                (output_root / "anonymous_test" / "run_report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(0, result)
        self.assertEqual(["new class sample.mp4"], report["videos"])
        self.assertEqual("planned", report["status"])
        self.assertEqual(4, len(report["phases"]))
        self.assertEqual("04_wiring_config", report["phases"][-1]["phase"])
        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 9], report["rubric_specific_artifacts_required"])

    def test_video_discovery_ignores_non_video_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("x", encoding="utf-8")
            (root / "sample.webm").write_bytes(b"x")

            videos = MODULE.discover_videos(root)

        self.assertEqual(["sample.webm"], [path.name for path in videos])

    def test_v3_is_an_explicit_opt_in_for_the_one_command_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_dir = root / "videos"
            video_dir.mkdir()
            (video_dir / "anonymous.mp4").write_bytes(b"dry-run")
            output_root = root / "outputs"
            MODULE.main(
                [
                    "--video-dir",
                    str(video_dir),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "v3_plan",
                    "--action-version",
                    "v3",
                    "--dry-run",
                ]
            )
            report = json.loads((output_root / "v3_plan" / "run_report.json").read_text(encoding="utf-8"))
        action_phase = report["phases"][2]
        self.assertEqual("v3", report["action_version"])
        self.assertEqual("03_action_v3", action_phase["phase"])
        self.assertTrue(any("qwen_experiment_action_hierarchical_v3.py" in item for item in action_phase["command"]))
        action_summary = report["outputs"]["action_summary"].replace("\\", "/")
        self.assertTrue(action_summary.endswith("actions/v3/summary.json"))

    def test_v2_temporal_guard_is_independent_and_reuses_v2_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            (videos / "anonymous_sample.mp4").write_bytes(b"not-opened-in-dry-run")
            output = root / "outputs"
            result = MODULE.main(
                [
                    "--video-dir",
                    str(videos),
                    "--output-root",
                    str(output),
                    "--run-id",
                    "guarded_plan",
                    "--action-version",
                    "v2-temporal-guard",
                    "--dry-run",
                ]
            )
            report = json.loads((output / "guarded_plan" / "run_report.json").read_text(encoding="utf-8"))
        self.assertEqual(0, result)
        action_phase = next(item for item in report["phases"] if item["phase"] == "03_action_v2-temporal-guard")
        self.assertTrue(
            any("qwen_experiment_action_hierarchical_v2_temporal_guard.py" in item for item in action_phase["command"])
        )
        schema_index = action_phase["command"].index("--schema") + 1
        self.assertTrue(action_phase["command"][schema_index].endswith("resistance_7stage_no_battery_v2.json"))
        action_summary = report["outputs"]["action_summary"].replace("\\", "/")
        self.assertTrue(action_summary.endswith("actions/v2-temporal-guard/summary.json"))


if __name__ == "__main__":
    unittest.main()
