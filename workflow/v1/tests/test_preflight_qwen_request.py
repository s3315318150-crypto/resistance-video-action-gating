from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load_script("preflight_qwen_request.py")


class PreflightTests(unittest.TestCase):
    def test_relative_media_path_is_resolved_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media = root / "media"
            media.mkdir()
            image_path = media / "frame.jpg"
            image = np.full((120, 180, 3), 127, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "selected_candidates": [
                            {
                                "candidate_id": "candidate_001",
                                "frame": {"path": "media/frame.jpg"},
                                "rois": [],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = preflight.build_report(manifest_path)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["image_count"], 1)
            self.assertEqual(Path(report["media"][0]["path"]), image_path.resolve())

    def test_too_many_images_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = []
            for index in range(9):
                path = root / f"frame_{index}.jpg"
                image = np.full((32, 32, 3), index * 20, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(path), image))
                candidates.append(
                    {
                        "candidate_id": f"candidate_{index}",
                        "frame": {"path": path.name},
                        "rois": [],
                    }
                )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"selected_candidates": candidates}), encoding="utf-8")
            report = preflight.build_report(manifest_path)
            self.assertFalse(report["request_should_be_sent"])
            self.assertIn("too_many_images", {item["code"] for item in report["errors"]})


if __name__ == "__main__":
    unittest.main()
