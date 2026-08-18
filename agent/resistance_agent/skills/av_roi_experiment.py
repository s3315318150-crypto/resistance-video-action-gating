"""Experimental A/V glyph gate for dynamic meter ROI candidates.

This module is intentionally isolated from the live R5/R6 path. It audits an
already-produced face crop and accepts it only when a visible A/V glyph wins a
template comparison with a readable-face geometry check. It does not use a
video id, fixed coordinates, historical crops, or model labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FONT_FACES = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
)
FONT_SCALES = (0.65, 0.8, 0.95, 1.1, 1.3, 1.55, 1.8, 2.1)
GLYPHS = ("A", "V")


def _templates(glyph: str) -> list[np.ndarray]:
    templates: list[np.ndarray] = []
    for face in FONT_FACES:
        for scale in FONT_SCALES:
            (width, height), baseline = cv2.getTextSize(glyph, face, scale, 2)
            canvas = np.full((height + baseline + 12, width + 12), 255, dtype=np.uint8)
            cv2.putText(canvas, glyph, (6, height + 4), face, scale, 0, 2, cv2.LINE_AA)
            templates.append(canvas)
    return templates


def _gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError(f"unable to decode face crop: {path}")
    scale = min(1.0, 720.0 / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def _central_region(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    left, top = int(width * 0.06), int(height * 0.04)
    right, bottom = int(width * 0.94), int(height * 0.84)
    return image[top:bottom, left:right], (left, top)


def _best_glyph_score(image: np.ndarray, glyph: str) -> dict[str, Any]:
    region, offset = _central_region(image)
    best: tuple[float, tuple[int, int], tuple[int, int]] | None = None
    for template in _templates(glyph):
        if template.shape[0] >= region.shape[0] or template.shape[1] >= region.shape[1]:
            continue
        response = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _min_value, max_value, _min_location, max_location = cv2.minMaxLoc(response)
        item = (float(max_value), max_location, (template.shape[1], template.shape[0]))
        if best is None or item[0] > best[0]:
            best = item
    if best is None:
        return {"glyph": glyph, "score": 0.0, "location": None, "size": None}
    return {
        "glyph": glyph,
        "score": round(best[0], 6),
        "location": [int(best[1][0] + offset[0]), int(best[1][1] + offset[1])],
        "size": [int(best[2][0]), int(best[2][1])],
    }


def _geometry(image: np.ndarray) -> dict[str, float]:
    edges = cv2.Canny(image, 45, 150)
    white_ratio = float(np.mean(image >= 150))
    edge_density = float(np.mean(edges > 0))
    return {
        "white_ratio": round(white_ratio, 6),
        "edge_density": round(edge_density, 6),
    }


def audit_face(
    face_path: str | Path,
    *,
    minimum_glyph_score: float = 0.50,
    minimum_margin: float = 0.035,
) -> dict[str, Any]:
    path = Path(face_path).resolve()
    image = _gray(path)
    scores = {glyph: _best_glyph_score(image, glyph) for glyph in GLYPHS}
    ordered = sorted(scores.values(), key=lambda item: float(item["score"]), reverse=True)
    winner = ordered[0]
    runner_up = ordered[1]
    margin = float(winner["score"]) - float(runner_up["score"])
    identity = (
        winner["glyph"]
        if float(winner["score"]) >= minimum_glyph_score and margin >= minimum_margin
        else "unknown"
    )
    geometry = _geometry(image)
    return {
        "schema_version": "av_roi_experiment.v1",
        "face_path": str(path),
        "identity": {"glyph": identity, "winner": winner, "runner_up": runner_up, "margin": round(margin, 6)},
        "geometry": geometry,
        "accepted": identity in GLYPHS,
        "thresholds": {
            "minimum_glyph_score": minimum_glyph_score,
            "minimum_margin": minimum_margin,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit face crops for visible A/V glyphs.")
    parser.add_argument("face_paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"schema_version": "av_roi_experiment_batch.v1", "faces": [audit_face(path) for path in args.face_paths]}
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
