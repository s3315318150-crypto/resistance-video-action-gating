from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import qwen_hierarchical_v2_segment_frame_agent as agent
import qwen_experiment_action_hierarchical_v2_screenshot_guard_agent as entrypoint
import qwen_experiment_action_hierarchical_v2_frame_agent as original_entrypoint


def event(action: str, start: float, end: float, source: str) -> dict:
    first = int(start * 10)
    last = int(end * 10)
    return {
        "source_event_id": source,
        "window_id": "w000",
        "action_type": action,
        "first_frame_id": f"frame_{first:06d}",
        "last_frame_id": f"frame_{last:06d}",
        "representative_frame_id": f"frame_{first:06d}",
        "first_frame_number": first,
        "last_frame_number": last,
        "representative_frame_number": first,
        "first_seconds": start,
        "last_seconds": end,
        "representative_seconds": start,
        "evidence": action,
        "confidence": 0.9,
    }


class SegmentFrameAgentTests(unittest.TestCase):
    def test_original_v2_frame_agent_does_not_bind_screenshot_guard_reduce(self) -> None:
        import qwen_experiment_action_hierarchical_v1 as engine

        original = {
            "map": engine._run_map,
            "reduce": engine._run_reduce,
            "salvage": engine.salvage_reduce_response,
            "select": engine.select_events,
        }
        try:
            original_entrypoint.bind_frame_agent()
            self.assertEqual(original_entrypoint.ALGORITHM_ID, engine.ALGORITHM_ID)
            self.assertIs(original_entrypoint._ORIGINAL_SALVAGE_REDUCE_RESPONSE, engine.salvage_reduce_response)
            self.assertIs(original_entrypoint._ORIGINAL_SELECT_EVENTS, engine.select_events)
            self.assertNotIn("screenshot_guard", engine.salvage_reduce_response.__module__)
            self.assertNotIn("screenshot_guard", engine.select_events.__module__)
        finally:
            engine._run_map = original["map"]
            engine._run_reduce = original["reduce"]
            engine.salvage_reduce_response = original["salvage"]
            engine.select_events = original["select"]

    def test_entrypoint_binds_agent_identity_and_resolution_defaults(self) -> None:
        import qwen_experiment_action_hierarchical_v1 as engine
        import qwen_hierarchical_v1_contract as contract

        original = {
            "map": engine._run_map,
            "reduce": engine._run_reduce,
            "algorithm": engine.ALGORITHM_ID,
            "version": engine.ALGORITHM_SCHEMA_VERSION,
            "schema": engine.STAGE_SCHEMA_ID,
            "default_schema": engine.DEFAULT_SCHEMA,
            "default_output_root": engine.DEFAULT_OUTPUT_ROOT,
            "salvage": engine.salvage_reduce_response,
            "select": engine.select_events,
            "contract_schema": contract.STAGE_SCHEMA_ID,
            "contract_stages": contract.STAGES,
        }
        try:
            entrypoint.bind_screenshot_guard_agent()
            self.assertEqual(entrypoint.ALGORITHM_ID, engine.ALGORITHM_ID)
            self.assertEqual(entrypoint.ALGORITHM_SCHEMA_VERSION, engine.ALGORITHM_SCHEMA_VERSION)
            self.assertEqual(entrypoint.STAGE_SCHEMA_ID, engine.STAGE_SCHEMA_ID)
            argv = entrypoint.normalized_argv([])
            self.assertEqual("1280", argv[argv.index("--max-model-edge") + 1])
            self.assertEqual("90", argv[argv.index("--jpeg-quality") + 1])
        finally:
            engine._run_map = original["map"]
            engine._run_reduce = original["reduce"]
            engine.ALGORITHM_ID = original["algorithm"]
            engine.ALGORITHM_SCHEMA_VERSION = original["version"]
            engine.STAGE_SCHEMA_ID = original["schema"]
            engine.DEFAULT_SCHEMA = original["default_schema"]
            engine.DEFAULT_OUTPUT_ROOT = original["default_output_root"]
            engine.salvage_reduce_response = original["salvage"]
            engine.select_events = original["select"]
            contract.STAGE_SCHEMA_ID = original["contract_schema"]
            contract.STAGES = original["contract_stages"]

    def test_requests_only_when_measurement_is_missing(self) -> None:
        missing = [event("wiring_action", 0, 10, "w1"), event("writing_action", 20, 30, "r1")]
        present = [
            event("wiring_action", 0, 10, "w1"),
            event("measurement_action", 12, 18, "m1"),
            event("writing_action", 20, 30, "r1"),
        ]
        self.assertEqual(1, len(agent.plan_frame_requests(missing, 0, 40)))
        self.assertEqual([], agent.plan_frame_requests(present, 0, 40))

    def test_plan_is_limited_to_two_experiment_cycles(self) -> None:
        events = [
            event("wiring_action", 0, 10, "w1"),
            event("writing_action", 20, 30, "r1"),
            event("wiring_action", 40, 50, "w2"),
            event("writing_action", 60, 70, "r2"),
            event("wiring_action", 80, 90, "w3"),
            event("writing_action", 100, 110, "r3"),
        ]
        requests = agent.plan_frame_requests(events, 0, 120)
        self.assertEqual(2, len(requests))
        self.assertEqual([1, 2], [item["cycle_index"] for item in requests])

    def test_second_round_runs_only_after_first_round_misses_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepared = {
                "video_id": "anonymous.mp4",
                "video_dir": Path(directory),
                "fixed_start": 0.0,
                "fixed_end": 80.0,
                "fps": 10.0,
                "frame_count": 500,
            }
            args = argparse.Namespace(max_model_edge=1280, max_attempts=1, map_max_tokens=100)
            engine = SimpleNamespace(
                assign_seven_stages=lambda events, _terminal: {
                    "observed_stage_intervals": [],
                }
            )
            map_events = [
                event("wiring_action", 0, 10, "w1"),
                event("writing_action", 20, 30, "r1"),
                event("wiring_action", 40, 50, "w2"),
                event("writing_action", 60, 70, "r2"),
            ]
            calls: list[tuple[str, int]] = []

            def fake_request(**kwargs):
                request = kwargs["request"]
                round_number = kwargs["round_number"]
                calls.append((request["request_id"], round_number))
                result_events = []
                if request["request_id"] == "measurement_gap_001" and round_number == 1:
                    result_events = [
                        event("measurement_action", 12, 18, "supplemental_measurement"),
                        event("writing_action", 18, 19, "supplemental_writing"),
                    ]
                record = {
                    **request,
                    "request_id": f"{request['request_id']}_round_{round_number}",
                    "attempts": [{}],
                    "input_frames": [],
                }
                return result_events, record, 1

            with mock.patch.object(agent, "_run_request", side_effect=fake_request):
                supplemental, report = agent.run_segment_frame_agent(
                    engine=engine,
                    prepared=prepared,
                    client=object(),
                    args=args,
                    map_events=map_events,
                )

        self.assertEqual(
            [("measurement_gap_001", 1), ("measurement_gap_002", 1), ("measurement_gap_002", 2)],
            calls,
        )
        self.assertEqual(["measurement_action"], [item["action_type"] for item in supplemental])
        self.assertEqual(3, report["frame_request_count"])
        self.assertEqual(3, report["qwen_request_count"])
        self.assertFalse(report["video_id_used_for_routing"])
        self.assertFalse(report["historical_artifacts_used"])
        self.assertFalse(report["fixed_video_roi_used"])

    def test_budget_is_four_requests_and_sixty_four_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepared = {
                "video_id": "anonymous.mp4",
                "video_dir": Path(directory),
                "fixed_start": 0.0,
                "fixed_end": 80.0,
                "fps": 10.0,
                "frame_count": 500,
            }
            args = argparse.Namespace(max_model_edge=1280, max_attempts=1, map_max_tokens=100)
            engine = SimpleNamespace(assign_seven_stages=lambda events, _terminal: {"observed_stage_intervals": []})
            map_events = [
                event("wiring_action", 0, 10, "w1"),
                event("writing_action", 20, 30, "r1"),
                event("wiring_action", 40, 50, "w2"),
                event("writing_action", 60, 70, "r2"),
            ]

            def fake_request(**kwargs):
                request = kwargs["request"]
                return [], {
                    **request,
                    "request_id": f"{request['request_id']}_round_{kwargs['round_number']}",
                    "attempts": [{}],
                    "input_frames": [],
                }, 17 if kwargs["round_number"] == 1 else 15

            with mock.patch.object(agent, "_run_request", side_effect=fake_request):
                _, report = agent.run_segment_frame_agent(
                    engine=engine,
                    prepared=prepared,
                    client=object(),
                    args=args,
                    map_events=map_events,
                )
        self.assertEqual(4, report["frame_request_count"])
        self.assertEqual(64, report["supplemental_frame_count"])

    def test_reduce_preserves_base_events_and_inserts_only_supplemental_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base_events = [
                event("wiring_action", 0, 10, "w1"),
                event("writing_action", 20, 30, "r1"),
                event("cleanup_action", 40, 45, "c1"),
            ]
            supplemental = {
                **event("measurement_action", 12, 18, "m1"),
                "segment_frame_agent_source": agent.SUPPLEMENTAL_EVENT_SOURCE,
            }
            reduced_inputs: list[list[dict]] = []

            def fake_map(prepared, client, args):
                return [], [], []

            def fake_reduce(prepared, map_events, client, args):
                reduced_inputs.append(list(map_events))
                selected = agent.deduplicate_map_events(map_events)
                cleanup_id = next(
                    item["event_id"]
                    for item in selected
                    if item["action_type"] == "cleanup_action"
                )
                return selected, {
                    "selection": {"terminal_cleanup_event_id": cleanup_id},
                    "accepted_events": selected,
                }, []

            engine = SimpleNamespace(_run_map=fake_map, _run_reduce=fake_reduce)
            agent.install(engine)
            selected, result, _ = engine._run_reduce(
                {"video_dir": Path(directory)},
                [*base_events, supplemental],
                object(),
                argparse.Namespace(),
            )

            self.assertEqual(base_events, reduced_inputs[0])
            self.assertEqual(
                ["wiring_action", "measurement_action", "writing_action", "cleanup_action"],
                [item["action_type"] for item in selected],
            )
            merge = result["segment_frame_agent_merge"]
            self.assertTrue(merge["base_events_preserved"])
            self.assertEqual(["agent_evt_0001"], merge["inserted_event_ids"])
            self.assertEqual([], merge["rejected_supplemental_events"])

    def test_supplemental_measurement_cannot_replace_existing_measurement(self) -> None:
        base = agent.deduplicate_map_events(
            [
                event("wiring_action", 0, 10, "w1"),
                event("measurement_action", 12, 18, "m1"),
                event("writing_action", 20, 30, "r1"),
            ]
        )
        supplemental = [
            {
                **event("measurement_action", 14, 16, "supplemental"),
                "segment_frame_agent_source": agent.SUPPLEMENTAL_EVENT_SOURCE,
            }
        ]

        merged, record = agent.merge_supplemental_measurements(base, supplemental, None)

        self.assertEqual(base, merged)
        self.assertEqual([], record["inserted_event_ids"])
        self.assertEqual("overlaps_existing_measurement", record["rejected_supplemental_events"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
