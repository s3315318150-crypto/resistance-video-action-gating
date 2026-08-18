from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = AGENT_ROOT.parent
sys.path.insert(0, str(AGENT_ROOT / "resistance_agent"))

from skills.router import select_live_skills  # noqa: E402
from toolkit import load_config  # noqa: E402


class PublicReleaseContractTests(unittest.TestCase):
    def test_config_uses_repository_local_workflow_v2(self) -> None:
        config = load_config(AGENT_ROOT / "config.json")
        workflow = config["workflow"]
        self.assertEqual("workflow/v2", workflow["release_root"])
        self.assertTrue((REPOSITORY_ROOT / workflow["release_root"]).is_dir())
        self.assertNotIn("replay", config)
        self.assertEqual("QWEN_API_BASE_URL", config["models"]["qwen"]["base_url_env"])
        self.assertEqual("QWEN_API_TOKEN", config["models"]["qwen"]["api_key_env"])

    def test_dynamic_rubric8_runner_matches_agent_cli_contract(self) -> None:
        runner = REPOSITORY_ROOT / "workflow" / "v2" / "scripts" / "run_resistance_disconnect_battery_sequence_v1.py"
        completed = subprocess.run(
            [sys.executable, str(runner), "--help"],
            cwd=REPOSITORY_ROOT / "workflow" / "v2",
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        for option in (
            "--source-video",
            "--dynamic-roi-min-confidence",
            "--time-mode",
            "--api-base-url",
            "--api-token",
        ):
            self.assertIn(option, completed.stdout)
        source = runner.read_text(encoding="utf-8")
        self.assertNotIn("source_video_sha256", source)
        self.assertNotIn("https://", source)

    def test_routing_is_independent_of_video_identity(self) -> None:
        stage_runs = [
            {"stage": "circuit_wiring", "start_seconds": 2.0, "end_seconds": 8.0},
            {"stage": "measurement_1", "start_seconds": 9.0, "end_seconds": 12.0},
            {"stage": "recording_1", "start_seconds": 13.0, "end_seconds": 16.0},
            {"stage": "material_cleanup", "start_seconds": 20.0, "end_seconds": 24.0},
        ]
        plans = []
        with tempfile.TemporaryDirectory(dir=AGENT_ROOT) as temporary:
            root = Path(temporary)
            for index, source_id in enumerate(("alpha.mp4", "renamed-input.mp4")):
                summary = root / f"summary_{index}.json"
                summary.write_text(
                    json.dumps(
                        {
                            "records": [
                                {
                                    "source_video_id": source_id,
                                    "source_observed_stage_runs": stage_runs,
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                plans.append(
                    select_live_skills(
                        source_video_id=source_id,
                        boundary_summary_path=summary,
                        action_summary_path=None,
                        allowed_root=root,
                    )
                )

        self.assertEqual(plans[0]["selected_skills"], plans[1]["selected_skills"])
        for plan in plans:
            self.assertEqual("current_video_observed_situation_only", plan["selection_basis"])
            self.assertFalse(plan["video_id_used_for_routing"])
            self.assertFalse(plan["historical_artifacts_used"])
            self.assertFalse(plan["fixed_video_roi_used"])

    def test_recording_only_situation_selects_pre_recording_polarity_skill(self) -> None:
        stage_runs = [
            {"stage": "circuit_wiring", "start_seconds": 2.0, "end_seconds": 8.0},
            {"stage": "recording_1", "start_seconds": 13.0, "end_seconds": 16.0},
        ]
        with tempfile.TemporaryDirectory(dir=AGENT_ROOT) as temporary:
            root = Path(temporary)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps({"source_observed_stage_runs": stage_runs}),
                encoding="utf-8",
            )
            plan = select_live_skills(
                source_video_id="anonymous-input.mp4",
                boundary_summary_path=summary,
                action_summary_path=None,
                allowed_root=root,
            )

        polarity = next(item for item in plan["selected_skills"] if item["rubric_ids"] == [4])
        self.assertEqual("polarity.pre_recording_dynamic_roi", polarity["skill_id"])

    def test_polarity_runner_uses_public_environment_configuration(self) -> None:
        runner = AGENT_ROOT / "scripts" / "run_qwen_meter_polarity_lenient.py"
        source = runner.read_text(encoding="utf-8")

        self.assertIn('os.getenv("QWEN_API_BASE_URL", "")', source)
        self.assertIn('os.getenv("QWEN_API_TOKEN", "")', source)
        self.assertIn('os.getenv("QWEN_MODEL", "qwen")', source)
        self.assertNotIn("qwen_pointer_wang_yizan_holdout", source)
        self.assertNotIn('"base_url": base_url', source)

    def test_rubric8_agent_wrapper_has_standard_help(self) -> None:
        runner = AGENT_ROOT / "scripts" / "run_rubric8_specialized.py"
        completed = subprocess.run(
            [sys.executable, str(runner), "--help"],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("--source-video", completed.stdout)

    def test_live_modules_have_no_development_video_defaults_or_fixed_rois(self) -> None:
        module_root = AGENT_ROOT / "resistance_agent"
        for name in ("meter_rubrics.py", "switch_rubric.py"):
            source = (module_root / name).read_text(encoding="utf-8")
            self.assertNotIn("all_five", source)
            self.assertNotIn("DEFAULT_ACTION_SUMMARY", source)
            self.assertNotIn("fallback_action_summary_path or", source)

        toolkit_source = (module_root / "toolkit.py").read_text(encoding="utf-8")
        self.assertNotIn('"run_record_rubrics"', toolkit_source)

    def test_agent_release_publishes_r0_to_r6_and_r8_only(self) -> None:
        toolkit_source = (AGENT_ROOT / "resistance_agent" / "toolkit.py").read_text(encoding="utf-8")
        self.assertIn("PUBLISHED_RUBRIC_IDS", toolkit_source)
        self.assertNotIn('"run_record_rubrics"', toolkit_source)
        self.assertNotIn('"record.two_cycle_consistency"', toolkit_source)
        self.assertFalse((AGENT_ROOT / "resistance_agent" / "record_rubrics.py").exists())


if __name__ == "__main__":
    unittest.main()
