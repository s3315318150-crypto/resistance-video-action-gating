from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_all_rubrics_v2.py"
SPEC = importlib.util.spec_from_file_location("run_all_rubrics_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AllRubricsV2Tests(unittest.TestCase):
    def test_anonymous_artifact_config_lists_all_ten_sources(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "ten_rubrics_artifacts.example.json").read_text(
                encoding="utf-8"
            )
        )
        artifacts = config["artifacts"]
        self.assertIn("wiring_episodes", artifacts)
        self.assertIn("rubric8", artifacts)
        self.assertIn("second_record", artifacts)

    def test_result_is_always_binary(self) -> None:
        self.assertEqual(MODULE.result("pass", "x", "ok", 0.5)["predicted_score"], 1)
        self.assertEqual(MODULE.result("fail", "x", "no", 0.5)["predicted_score"], 0)
        with self.assertRaises(ValueError):
            MODULE.result("abstained", "x", "no", 0.0)

    def test_artifact_binary_accepts_binary_fields_only(self) -> None:
        self.assertEqual("pass", MODULE.artifact_binary({"result": "pass"}, "result"))
        self.assertEqual("fail", MODULE.artifact_binary({"result": "uncertain"}, "result"))
        self.assertEqual("pass", MODULE.artifact_binary({"predicted_score": 1}, "result"))

    def test_stage_intervals_groups_repeated_stages(self) -> None:
        values = MODULE.stage_intervals(
            [
                {"stage": "recording_1", "start_seconds": 1, "end_seconds": 2},
                {"stage": "recording_1", "start_seconds": 3, "end_seconds": 4},
            ]
        )
        self.assertEqual(values, {"recording_1": [[1.0, 2.0], [3.0, 4.0]]})

    def test_markdown_has_ten_evaluation_columns(self) -> None:
        evaluations = {
            str(index): {"decision": "pass"} for index, _, _ in MODULE.RUBRICS
        }
        text = MODULE.markdown([{"video_id": "sample_001", "evaluations": evaluations}])
        self.assertEqual(text.splitlines()[0], "# v2 十项评价结果")
        self.assertEqual(text.count("| pass"), 10)

    def test_loads_dedicated_rubric8_sequence_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            MODULE.write_json(
                path,
                {
                    "videos": [
                        {
                            "video_id": "8",
                            "decision": "pass",
                            "predicted_score": 1,
                            "episodes": [{"episode_id": "8_rewire_01"}],
                        },
                        {
                            "video_id": "38",
                            "decision": "fail",
                            "predicted_score": 0,
                            "episodes": [{"episode_id": "38_rewire_01"}],
                        },
                    ]
                },
            )

            values = MODULE.load_rubric8_sequence_results(path)

        self.assertEqual({"8", "38"}, set(values))
        self.assertEqual("pass", values["8"]["decision"])
        self.assertEqual("fail", values["38"]["decision"])
        self.assertEqual("38_rewire_01", values["38"]["episodes"][0]["episode_id"])

    def test_dispatches_dedicated_rubric8_summary_outside_legacy_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            MODULE.write_json(
                path,
                {
                    "videos": [
                        {
                            "video_id": "16",
                            "decision": "pass",
                            "predicted_score": 1,
                            "episodes": [{"episode_id": "16_recovery_01", "decision": "pass"}],
                        }
                    ]
                },
            )

            values = MODULE.load_battery_results(path)

        self.assertEqual("pass", values["16"]["decision"])
        self.assertEqual("16_recovery_01", values["16"]["episodes"][0]["episode_id"])

    def test_loads_dedicated_ammeter_source_series_result(self) -> None:
        document = {
            "rubric_id": "resistance.ammeter_source_short_circuit_v1",
            "predictions": [
                {
                    "video_id": "8_sample.mp4",
                    "automated_outcome": "scored",
                    "decision": "不是",
                    "predicted_score": 0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parsed_predictions.json"
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            results = MODULE.load_ammeter_source_series_results(path)
        self.assertEqual(results["8"]["predicted_score"], 0)

    def test_rejects_inconsistent_ammeter_source_series_result(self) -> None:
        document = {
            "rubric_id": "resistance.ammeter_source_short_circuit_v1",
            "predictions": [
                {
                    "video_id": "8_sample.mp4",
                    "automated_outcome": "scored",
                    "decision": "不是",
                    "predicted_score": 1,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parsed_predictions.json"
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.load_ammeter_source_series_results(path)

    def test_cli_replays_all_ten_anonymous_artifacts(self) -> None:
        """The public entrypoint produces ten binary results from generic inputs."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            action_result = root / "action_result.json"
            action_result.write_text(
                json.dumps(
                    {
                        "timeline_segments": [
                            {"kind": "observed_stage", "stage": "circuit_wiring", "start_seconds": 0, "end_seconds": 2},
                            {"kind": "observed_stage", "stage": "measurement_1", "start_seconds": 2, "end_seconds": 3},
                            {"kind": "observed_stage", "stage": "recording_1", "start_seconds": 3, "end_seconds": 4},
                            {"kind": "observed_stage", "stage": "circuit_rewiring", "start_seconds": 4, "end_seconds": 5},
                            {"kind": "observed_stage", "stage": "measurement_2", "start_seconds": 5, "end_seconds": 6},
                            {"kind": "observed_stage", "stage": "recording_2", "start_seconds": 6, "end_seconds": 7},
                            {"kind": "observed_stage", "stage": "material_cleanup", "start_seconds": 7, "end_seconds": 8},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            action_summary = root / "action_summary.json"
            action_summary.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "source_video_id": "sample_001.mp4",
                                "result_path": str(action_result),
                                "status": "completed",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            def write(name: str, value: dict) -> str:
                path = root / name
                path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
                return str(path)

            polarity_root = root / "polarity" / "video_sample_001"
            polarity_root.mkdir(parents=True)
            (polarity_root / "result.json").write_text(
                json.dumps({"video_id": "sample_001", "result": "pass"}, ensure_ascii=False),
                encoding="utf-8",
            )
            config = {
                "artifacts": {
                    "wiring_episodes": write("wiring.json", {"episodes": [{"video_id": "sample_001", "series_circuit": {"decision": "pass"}, "voltmeter_parallel": {"decision": "pass"}}]}),
                    "deepseek": "",
                    "switch": write("switch.json", {"results": [{"video_id": "sample_001", "decision": "pass"}]}),
                    "polarity_root": str(polarity_root.parent),
                    "first_record": write("first.json", {"results": [{"video_id": "sample_001", "result": "pass"}]}),
                    "opencv": write("opencv.json", {"videos": [{"video_id": "sample_001", "rubric_5": {"decision": {"automated_outcome": "scored", "predicted_score": 1}}, "rubric_6": {"decision": {"automated_outcome": "scored", "predicted_score": 1}}}]}),
                    "rubric8": write("rubric8.json", {"videos": [{"video_id": "sample_001", "decision": "pass"}]}),
                    "second_record": write("second.json", {"results": [{"video_id": "sample_001", "result": "pass"}]}),
                    "ammeter_source_series": "",
                }
            }
            config_path = root / "artifacts.json"
            config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            output = root / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--action-summary",
                    str(action_summary),
                    "--artifact-config",
                    str(config_path),
                    "--output-root",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            result = json.loads((output / "video_sample_001" / "result.json").read_text(encoding="utf-8"))

        self.assertEqual(set(str(index) for index in range(10)), set(result["evaluations"]))
        self.assertTrue(all(item["decision"] == "pass" for item in result["evaluations"].values()))


if __name__ == "__main__":
    unittest.main()
