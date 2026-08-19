from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT / "resistance_agent"))

from opencv_plug_motion import match_three  # noqa: E402
from opencv_switch_overlap import analyze_opencv_switch_overlap, fuse_same_frame_records  # noqa: E402
from opencv_switch_state import (  # noqa: E402
    _annotate_closed_persistence,
    bridge_features,
    cluster_threshold,
    component_candidates,
)


def _lead(
    x: float,
    gap: float,
    *,
    is_lead: bool = True,
    color: str = "black",
) -> dict[str, object]:
    return {
        "color": color,
        "component_box_xywh": [0, 0, 12, 36],
        "centroid_norm": [x, 0.4],
        "min_gap_norm": gap,
        "thickness_norm": 0.08,
        "is_lead": is_lead,
    }


class OpenCvSwitchOverlapTests(unittest.TestCase):
    def test_dynamic_support_prefers_paired_switch_over_battery_holder(self) -> None:
        image = np.full((320, 480, 3), (230, 230, 230), dtype=np.uint8)
        # Compact switch support with paired metal contacts and red terminal posts.
        cv2.rectangle(image, (110, 70), (250, 145), (0, 120, 230), -1)
        cv2.rectangle(image, (130, 108), (230, 140), (0, 155, 0), -1)
        for x in (145, 215):
            cv2.circle(image, (x, 95), 13, (0, 0, 190), -1)
            cv2.circle(image, (x, 95), 6, (210, 210, 210), -1)
        cv2.rectangle(image, (140, 83), (220, 91), (40, 40, 40), -1)
        # Long orange holder with green cells is a competing false candidate.
        cv2.rectangle(image, (290, 190), (445, 245), (0, 120, 230), -1)
        for x in (310, 390):
            cv2.rectangle(image, (x, 200), (x + 36, 235), (0, 190, 70), -1)
            cv2.rectangle(image, (x + 4, 204), (x + 32, 231), (20, 210, 200), -1)

        candidates = component_candidates(image, limit=5)
        self.assertGreaterEqual(len(candidates), 2)
        selected = candidates[0]
        battery = max(candidates, key=lambda item: float(item["battery_like_score"]))
        self.assertAlmostEqual(180.0, selected["center"][0], delta=12.0)
        self.assertAlmostEqual(107.5, selected["center"][1], delta=12.0)
        self.assertEqual("orange_support_and_contact_pair", selected["detection_mode"])
        self.assertGreater(float(selected["contact_pair_score"]), 0.50)
        self.assertGreater(float(selected["score"]), float(battery["score"]))
        self.assertGreater(float(battery["battery_like_score"]), 0.20)

    def test_real_three_frame_contact_transition_is_detected(self) -> None:
        tracks = match_three(
            [_lead(0.0, 0.02)],
            [_lead(0.12, 0.13)],
            [_lead(0.24, 0.26)],
        )
        self.assertEqual(1, len(tracks))
        self.assertTrue(tracks[0]["contact_flip"])
        self.assertTrue(tracks[0]["monotonic"])
        self.assertTrue(tracks[0]["real_transition"])

    def test_whole_apparatus_translation_is_not_plug_motion(self) -> None:
        tracks = match_three(
            [_lead(0.35, 0.01)],
            [_lead(0.35, 0.01)],
            [_lead(0.35, 0.01)],
        )
        self.assertEqual(1, len(tracks))
        self.assertFalse(tracks[0]["contact_flip"])
        self.assertFalse(tracks[0]["real_transition"])

    def test_switch_handle_without_cable_is_not_a_plug(self) -> None:
        tracks = match_three(
            [_lead(0.0, 0.02, is_lead=False)],
            [_lead(0.12, 0.13, is_lead=False)],
            [_lead(0.24, 0.26, is_lead=False)],
        )
        self.assertEqual([], tracks)

    def test_bridge_geometry_separates_closed_from_open(self) -> None:
        orange = (0, 105, 245)
        copper = (20, 70, 120)
        closed = np.full((128, 256, 3), orange, dtype=np.uint8)
        opened = closed.copy()
        cv2.rectangle(closed, (45, 18), (215, 34), copper, -1)
        cv2.rectangle(opened, (45, 18), (78, 34), copper, -1)
        closed_score = bridge_features(closed)["bridge_score"]
        open_score = bridge_features(opened)["bridge_score"]
        self.assertGreater(closed_score, open_score + 0.30)

    def test_cluster_threshold_is_deterministic_and_label_free(self) -> None:
        observations = [
            {"bridge_score": value}
            for value in (0.20, 0.25, 0.31, 0.84, 0.90, 0.96)
        ]
        first = cluster_threshold(observations)
        second = cluster_threshold(observations)
        self.assertEqual(first, second)
        self.assertGreater(first[0], first[1][0])
        self.assertLess(first[0], first[1][1])

    def test_low_geometry_clusters_do_not_invent_a_closed_state(self) -> None:
        observations = [
            {"bridge_score": value}
            for value in (0.18, 0.22, 0.28, 0.36, 0.42, 0.48)
        ]
        threshold, centers = cluster_threshold(observations)
        self.assertEqual(0.80, threshold)
        self.assertEqual(2, len(centers))

    def test_different_frames_do_not_form_a_failure(self) -> None:
        sampled = [
            {
                "window_id": "w1",
                "stage": "circuit_wiring",
                "timestamp_seconds": 1.0,
                "frame_number": frame_number,
            }
            for frame_number in (5, 6)
        ]
        switch = [
            {
                **sampled[0],
                "state": "closed",
                "bridge_score": 0.9,
                "identity_score": 0.8,
                "crop_path": "switch.jpg",
            }
        ]
        plug = [{**sampled[1], "real_transition": True, "confidence": 0.9}]
        _, overlaps = fuse_same_frame_records(sampled, switch, plug)
        self.assertEqual([], overlaps)

    def test_exact_same_frame_closed_and_transition_forms_failure(self) -> None:
        sampled = [
            {
                "window_id": "w1",
                "stage": "circuit_wiring",
                "timestamp_seconds": 1.0,
                "frame_number": 5,
            }
        ]
        switch = [
            {
                **sampled[0],
                "state": "closed",
                "bridge_score": 0.9,
                "identity_score": 0.8,
                "crop_path": "switch.jpg",
            }
        ]
        plug = [{**sampled[0], "real_transition": True, "confidence": 0.9}]
        frames, overlaps = fuse_same_frame_records(sampled, switch, plug)
        self.assertTrue(frames[0]["same_frame_overlap"])
        self.assertEqual(1, len(overlaps))

    def test_two_frame_closed_run_is_not_supported(self) -> None:
        sampled = [
            {
                "window_id": "w1",
                "stage": "circuit_wiring",
                "timestamp_seconds": 1.0,
                "frame_number": 5,
            }
        ]
        switch = [
            {
                **sampled[0],
                "state": "closed",
                "closed_persistence_count": 2,
                "closed_persistence_duration_seconds": 0.2,
                "bridge_score": 0.95,
                "identity_score": 0.9,
                "crop_path": "brief_occluder.jpg",
            }
        ]
        plug = [{**sampled[0], "wiring_activity": True, "confidence": 0.9}]
        frames, overlaps = fuse_same_frame_records(sampled, switch, plug)
        self.assertFalse(frames[0]["switch_state_temporally_supported"])
        self.assertFalse(frames[0]["same_frame_overlap"])
        self.assertEqual([], overlaps)

    def test_three_frame_closed_run_remains_a_failure(self) -> None:
        sampled = [
            {
                "window_id": "w1",
                "stage": "circuit_wiring",
                "timestamp_seconds": 1.0,
                "frame_number": 5,
            }
        ]
        switch = [
            {
                **sampled[0],
                "state": "closed",
                "closed_persistence_count": 3,
                "closed_persistence_duration_seconds": 0.4,
                "bridge_score": 0.95,
                "identity_score": 0.9,
                "crop_path": "persistent_switch.jpg",
            }
        ]
        plug = [{**sampled[0], "wiring_activity": True, "confidence": 0.9}]
        frames, overlaps = fuse_same_frame_records(sampled, switch, plug)
        self.assertTrue(frames[0]["switch_state_temporally_supported"])
        self.assertTrue(frames[0]["same_frame_overlap"])
        self.assertEqual(1, len(overlaps))

    def test_closed_persistence_is_annotated_per_contiguous_run(self) -> None:
        observations = [
            {"window_id": "w1", "timestamp_seconds": timestamp, "state": state}
            for timestamp, state in (
                (0.0, "closed"),
                (0.2, "closed"),
                (0.4, "open"),
                (1.0, "closed"),
                (1.2, "closed"),
                (1.4, "closed"),
            )
        ]
        _annotate_closed_persistence(observations)
        self.assertEqual([2, 2, 0, 3, 3, 3], [
            item["closed_persistence_count"] for item in observations
        ])
        self.assertEqual(0.2, observations[0]["closed_persistence_duration_seconds"])
        self.assertEqual(0.4, observations[-1]["closed_persistence_duration_seconds"])

    def test_terminal_occupancy_change_marks_wiring_activity(self) -> None:
        tracks = match_three(
            [_lead(0.00, 0.32)],
            [_lead(0.12, 0.39)],
            [_lead(0.24, 0.44)],
        )
        self.assertEqual(1, len(tracks))
        self.assertFalse(tracks[0]["contact_flip"])
        self.assertTrue(tracks[0]["occupancy_transition"])
        self.assertTrue(tracks[0]["wiring_activity"])

    def test_support_frames_are_same_frame_fusion_candidates(self) -> None:
        sampled = [
            {
                "window_id": "w1",
                "stage": "circuit_wiring",
                "timestamp_seconds": time,
                "frame_number": frame,
            }
            for time, frame in ((1.0, 5), (1.2, 6), (1.4, 7))
        ]
        switch = [
            {
                **sampled[0],
                "state": "closed",
                "bridge_score": 0.9,
                "identity_score": 0.8,
                "crop_path": "switch.jpg",
            }
        ]
        plug = [
            {
                **sampled[1],
                "real_transition": False,
                "wiring_activity": True,
                "confidence": 0.8,
                "support_frames": sampled,
            }
        ]
        frames, overlaps = fuse_same_frame_records(sampled, switch, plug)
        self.assertTrue(frames[0]["wiring_active"])
        self.assertTrue(frames[0]["same_frame_overlap"])
        self.assertEqual(1, len(overlaps))

    def test_analyzer_rejects_declared_roi_that_it_does_not_execute(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported switch ROI mode"):
            analyze_opencv_switch_overlap(
                Path("unused.mp4"),
                [{"start_seconds": 0.0, "end_seconds": 1.0}],
                Path("unused"),
                roi_mode="fixed_video_roi",
            )


if __name__ == "__main__":
    unittest.main()
