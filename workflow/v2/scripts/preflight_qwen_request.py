#!/usr/bin/env python3
"""Deterministically preflight image payloads before a Qwen request.

The script reads only an evidence-search manifest and the image files named by
that manifest.  It never calls a model, opens a video, or reads labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


MIB = 1024 * 1024
MAX_IMAGES = 8
CLEANUP_TERMINAL_SCAN_MAX_IMAGES = 12
MAX_TOTAL_JPEG_BYTES = 10 * MIB
MAX_ESTIMATED_BASE64_BYTES = 14 * MIB
MAX_IMAGE_WIDTH = 1600
MAX_SINGLE_IMAGE_BYTES = 2 * MIB
NEAR_DUPLICATE_THUMBNAIL_MAD_MAX = 3.0
THUMBNAIL_SIZE = (32, 32)
REDUNDANT_ROI_DIMENSION_RATIO_MIN = 0.95
MAX_SOURCE_ROI_AREA_RATIO = 0.80


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_issue(issues: list[dict[str, Any]], code: str, message: str, **details: Any) -> None:
    issue: dict[str, Any] = {"code": code, "message": message}
    issue.update(details)
    issues.append(issue)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_media_path(raw_path: str, manifest_path: Path) -> Path:
    """Resolve relative media references from the manifest directory."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def estimated_base64_bytes(byte_count: int) -> int:
    return ((byte_count + 2) // 3) * 4


def request_image_limit(manifest: dict[str, Any]) -> tuple[int, str | None]:
    """Keep the normal eight-image limit, with one bounded cleanup fallback."""
    if (
        manifest.get("artifact_type") == "resistance_cleanup_action_guided_event_packet_v3"
        and manifest.get("candidate_kind") == "terminal_segment_fallback"
    ):
        sampling = manifest.get("sampling")
        declared = sampling.get("max_images") if isinstance(sampling, dict) else None
        if not isinstance(declared, int) or isinstance(declared, bool) or declared < 1 or declared > CLEANUP_TERMINAL_SCAN_MAX_IMAGES:
            return MAX_IMAGES, "cleanup_terminal_scan_image_budget_invalid"
        return declared, None
    return MAX_IMAGES, None


def extract_references(manifest: dict[str, Any], errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = manifest.get("selected_candidates")
    if not isinstance(candidates, list):
        add_issue(
            errors,
            "manifest_selected_candidates_invalid",
            "selected_candidates must be a list.",
        )
        return []

    references: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            add_issue(
                errors,
                "manifest_candidate_invalid",
                "A selected candidate must be an object.",
                candidate_index=candidate_index,
            )
            continue
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            candidate_id = f"candidate_{candidate_index + 1:03d}"

        frame = candidate.get("frame")
        frame_path = frame.get("path") if isinstance(frame, dict) else None
        if isinstance(frame_path, str) and frame_path.strip():
            references.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "field": "frame.path",
                    "roi_index": None,
                    "roi_role": None,
                    "roi_bbox_xyxy": None,
                    "raw_path": frame_path,
                }
            )
        else:
            add_issue(
                errors,
                "missing_media_file",
                "Candidate frame.path is missing.",
                candidate_id=candidate_id,
                field="frame.path",
            )

        rois = candidate.get("rois", [])
        if rois is None:
            rois = []
        if not isinstance(rois, list):
            add_issue(
                errors,
                "manifest_rois_invalid",
                "Candidate rois must be a list when present.",
                candidate_id=candidate_id,
            )
            continue
        for roi_index, roi in enumerate(rois):
            if not isinstance(roi, dict):
                add_issue(
                    errors,
                    "manifest_roi_invalid",
                    "A ROI must be an object.",
                    candidate_id=candidate_id,
                    roi_index=roi_index,
                )
                continue
            role = roi.get("role") if isinstance(roi.get("role"), str) else None
            for field in ("crop_path", "enhanced_or_rectified_path"):
                raw_path = roi.get(field)
                if raw_path is None:
                    continue
                if not isinstance(raw_path, str) or not raw_path.strip():
                    add_issue(
                        errors,
                        "missing_media_file",
                        "ROI media path is not a non-empty string.",
                        candidate_id=candidate_id,
                        roi_index=roi_index,
                        field=f"rois[].{field}",
                    )
                    continue
                references.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_index": candidate_index,
                        "field": f"rois[].{field}",
                        "roi_index": roi_index,
                        "roi_role": role,
                        "roi_bbox_xyxy": roi.get("bbox_xyxy"),
                        "roi_source_frame_width": roi.get("source_frame_width"),
                        "roi_source_frame_height": roi.get("source_frame_height"),
                        "roi_source_area_ratio": roi.get("source_area_ratio"),
                        "raw_path": raw_path,
                    }
                )
    return references


def inspect_media(
    references: list[dict[str, Any]],
    manifest_path: Path,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Inspect the same unique path list used by the Qwen runner."""
    inspected: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    for reference in references:
        raw_path = reference["raw_path"]
        path = resolve_media_path(raw_path, manifest_path)
        path_key = str(path)
        if path_key in by_path:
            existing = by_path[path_key]
            existing["references"].append(reference)
            add_issue(
                warnings,
                "duplicate_path_reference",
                "The same media path is referenced more than once and will be sent once.",
                path=path_key,
                original_reference_count=len(existing["references"]),
            )
            continue

        record: dict[str, Any] = {
            "path": path_key,
            "references": [reference],
            "exists": False,
            "regular_file": False,
            "decoded": False,
            "file_size_bytes": None,
            "width": None,
            "height": None,
            "sha256": None,
            "thumbnail": None,
        }
        by_path[path_key] = record
        inspected.append(record)

        try:
            file_stat = path.stat()
        except OSError:
            add_issue(
                errors,
                "missing_media_file",
                "Referenced media file does not exist.",
                path=path_key,
                references=record["references"],
            )
            continue
        if not stat.S_ISREG(file_stat.st_mode):
            add_issue(
                errors,
                "missing_media_file",
                "Referenced media path is not a regular file.",
                path=path_key,
                references=record["references"],
            )
            continue

        record["exists"] = True
        record["regular_file"] = True
        record["file_size_bytes"] = int(file_stat.st_size)
        if file_stat.st_size > MAX_SINGLE_IMAGE_BYTES:
            add_issue(
                errors,
                "image_too_large",
                "A single image exceeds the maximum JPEG byte budget.",
                path=path_key,
                file_size_bytes=int(file_stat.st_size),
                max_single_image_bytes=MAX_SINGLE_IMAGE_BYTES,
            )

        try:
            image = cv2.imread(path_key, cv2.IMREAD_COLOR)
        except cv2.error:
            image = None
        if image is None or image.size == 0:
            add_issue(
                errors,
                "image_decode_failed",
                "OpenCV could not decode the referenced image.",
                path=path_key,
            )
            continue

        height, width = image.shape[:2]
        record["decoded"] = True
        record["width"] = int(width)
        record["height"] = int(height)
        if width > MAX_IMAGE_WIDTH:
            add_issue(
                warnings,
                "image_width_exceeds_recommended_max",
                "Image width exceeds the recommended request-dimension budget; payload limits remain hard gates.",
                path=path_key,
                width=int(width),
                max_image_width=MAX_IMAGE_WIDTH,
            )
        try:
            record["sha256"] = sha256_file(path)
        except OSError:
            add_issue(
                errors,
                "missing_media_file",
                "Referenced image could not be read for hashing.",
                path=path_key,
            )
            continue
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        record["thumbnail"] = cv2.resize(grayscale, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
    return inspected


def exact_duplicate_groups(records: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["decoded"] and isinstance(record["sha256"], str):
            groups[record["sha256"]].append(record)
    result: list[dict[str, Any]] = []
    for digest, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        group = {
            "sha256": digest,
            "count": len(members),
            "paths": [member["path"] for member in members],
        }
        result.append(group)
        add_issue(
            warnings,
            "exact_duplicate_images",
            "Different request paths contain byte-identical image content.",
            **group,
        )
    return result


def near_duplicate_groups(records: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [record for record in records if record["decoded"] and record["thumbnail"] is not None]
    parent = list(range(len(usable)))
    pairs: list[dict[str, Any]] = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(usable):
        for right_index in range(left_index + 1, len(usable)):
            right = usable[right_index]
            if left["sha256"] == right["sha256"]:
                continue
            difference = cv2.absdiff(left["thumbnail"], right["thumbnail"])
            thumbnail_mad = float(cv2.mean(difference)[0])
            if thumbnail_mad <= NEAR_DUPLICATE_THUMBNAIL_MAD_MAX:
                union(left_index, right_index)
                pairs.append(
                    {
                        "left_path": left["path"],
                        "right_path": right["path"],
                        "thumbnail_mean_absolute_difference": round(thumbnail_mad, 6),
                    }
                )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(usable):
        grouped[find(index)].append(record)
    result: list[dict[str, Any]] = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        member_paths = {member["path"] for member in members}
        group_pairs = [
            pair
            for pair in pairs
            if pair["left_path"] in member_paths and pair["right_path"] in member_paths
        ]
        group = {
            "count": len(members),
            "paths": sorted(member_paths),
            "comparison_method": "32x32_grayscale_thumbnail_mean_absolute_difference",
            "threshold": NEAR_DUPLICATE_THUMBNAIL_MAD_MAX,
            "pairs": group_pairs,
        }
        result.append(group)
        add_issue(
            warnings,
            "near_duplicate_images",
            "Images are visually near-duplicate at thumbnail scale.",
            count=len(members),
            paths=group["paths"],
        )
    return result


def redundant_full_frame_rois(records: list[dict[str, Any]], errors: list[dict[str, Any]]) -> int:
    frames: dict[str, dict[str, Any]] = {}
    roi_media: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record in records:
        if not record["decoded"]:
            continue
        for reference in record["references"]:
            if reference["field"] == "frame.path":
                frames[reference["candidate_id"]] = record
            elif reference["field"] in {"rois[].crop_path", "rois[].enhanced_or_rectified_path"}:
                roi_index = reference["roi_index"]
                if isinstance(roi_index, int):
                    roi_media[(reference["candidate_id"], roi_index)].append((record, reference))

    count = 0
    for (candidate_id, _), media in roi_media.items():
        frame = frames.get(candidate_id)
        if frame is None:
            continue
        reference = media[0][1]
        width_ratio: float | None = None
        height_ratio: float | None = None
        measurement_source: str | None = None
        thumbnail_mad: float | None = None
        bbox = reference.get("roi_bbox_xyxy")
        source_width = reference.get("roi_source_frame_width")
        source_height = reference.get("roi_source_frame_height")
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox)
            and float(bbox[2]) > float(bbox[0])
            and float(bbox[3]) > float(bbox[1])
        ):
            if (
                isinstance(source_width, (int, float))
                and not isinstance(source_width, bool)
                and isinstance(source_height, (int, float))
                and not isinstance(source_height, bool)
                and float(source_width) > 0
                and float(source_height) > 0
            ):
                width_ratio = (float(bbox[2]) - float(bbox[0])) / float(source_width)
                height_ratio = (float(bbox[3]) - float(bbox[1])) / float(source_height)
                measurement_source = "source_frame_bbox_xyxy"
            else:
                width_ratio = (float(bbox[2]) - float(bbox[0])) / float(frame["width"])
                height_ratio = (float(bbox[3]) - float(bbox[1])) / float(frame["height"])
                measurement_source = "roi_bbox_xyxy"
        else:
            crop_media = [item for item in media if item[1]["field"] == "rois[].crop_path"]
            comparable = crop_media or media
            probe = comparable[0][0]
            width_ratio = float(probe["width"]) / float(frame["width"])
            height_ratio = float(probe["height"]) / float(frame["height"])
            measurement_source = "decoded_image_dimensions"
            if comparable[0][1]["field"] == "rois[].enhanced_or_rectified_path":
                difference = cv2.absdiff(probe["thumbnail"], frame["thumbnail"])
                thumbnail_mad = float(cv2.mean(difference)[0])
                if thumbnail_mad > NEAR_DUPLICATE_THUMBNAIL_MAD_MAX:
                    width_ratio = height_ratio = 0.0
                    measurement_source = "enhanced_image_dimensions_not_visually_full_frame"
        if (
            width_ratio >= REDUNDANT_ROI_DIMENSION_RATIO_MIN
            and height_ratio >= REDUNDANT_ROI_DIMENSION_RATIO_MIN
        ):
            count += 1
            details: dict[str, Any] = {
                "candidate_id": candidate_id,
                "roi_role": reference["roi_role"],
                "frame_path": frame["path"],
                "roi_paths": [record["path"] for record, _ in media],
                "width_ratio": round(width_ratio, 6),
                "height_ratio": round(height_ratio, 6),
                "measurement_source": measurement_source,
                "threshold": REDUNDANT_ROI_DIMENSION_RATIO_MIN,
            }
            if thumbnail_mad is not None:
                details["thumbnail_mean_absolute_difference"] = round(thumbnail_mad, 6)
            add_issue(
                errors,
                "redundant_full_frame_roi",
                "ROI is nearly the same dimensions as its candidate panorama and adds redundant payload.",
                **details,
            )
    return count


def oversized_source_rois(references: list[dict[str, Any]], errors: list[dict[str, Any]]) -> int:
    """Reject detail media whose recorded source crop covers almost all of a 4K frame."""
    seen: set[tuple[str, int]] = set()
    count = 0
    for reference in references:
        if reference["field"] not in {"rois[].crop_path", "rois[].enhanced_or_rectified_path"}:
            continue
        roi_index = reference["roi_index"]
        if not isinstance(roi_index, int):
            continue
        key = (reference["candidate_id"], roi_index)
        if key in seen:
            continue
        ratio = reference.get("roi_source_area_ratio")
        if (
            isinstance(ratio, (int, float))
            and not isinstance(ratio, bool)
            and float(ratio) > MAX_SOURCE_ROI_AREA_RATIO
        ):
            seen.add(key)
            count += 1
            add_issue(
                errors,
                "roi_source_area_too_large",
                "Detail ROI covers more than the allowed share of its original source frame.",
                candidate_id=reference["candidate_id"],
                roi_role=reference["roi_role"],
                source_area_ratio=round(float(ratio), 6),
                max_source_area_ratio=MAX_SOURCE_ROI_AREA_RATIO,
            )
    return count


def media_report(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": record["path"],
        "references": record["references"],
        "exists": record["exists"],
        "regular_file": record["regular_file"],
        "decoded": record["decoded"],
        "file_size_bytes": record["file_size_bytes"],
        "width": record["width"],
        "height": record["height"],
        "sha256": record["sha256"],
    }


def build_report(manifest_path: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "qwen_request_preflight",
        "valid": False,
        "request_should_be_sent": False,
        "errors": errors,
        "warnings": warnings,
        "manifest_path": str(manifest_path.resolve()),
        "image_count": 0,
        "total_jpeg_bytes": 0,
        "estimated_base64_bytes": 0,
        "max_dimensions": {"width": 0, "height": 0},
        "duplicate_groups": [],
        "near_duplicate_groups": [],
        "redundant_roi_count": 0,
        "oversized_source_roi_count": 0,
        "checked_rules": [
            "manifest_json_parse",
            "selected_candidate_media_path_collection",
            "regular_file_exists",
            "opencv_image_decode",
            "image_dimensions_and_single_image_budget",
            "total_jpeg_payload_budget",
            "estimated_base64_payload_budget",
            "exact_sha256_duplicates",
            "near_duplicate_thumbnail_comparison",
            "same_candidate_redundant_full_frame_roi",
            "source_roi_area_budget",
            "request_image_count_budget",
        ],
        "budget": {
            "max_images": MAX_IMAGES,
            "max_total_jpeg_mib": 10,
            "max_estimated_base64_mib": 14,
            "max_image_width": MAX_IMAGE_WIDTH,
            "max_single_image_mib": 2,
            "image_width_policy": "warning_only_when_byte_budgets_pass",
            "max_source_roi_area_ratio": MAX_SOURCE_ROI_AREA_RATIO,
        },
        "qwen_called": False,
        "video_accessed": False,
        "excel_accessed": False,
        "labels_accessed": False,
        "score_computed": False,
    }

    if not manifest_path.is_file():
        add_issue(errors, "manifest_missing", "Manifest file does not exist.", path=str(manifest_path))
        return report
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        add_issue(errors, "manifest_parse_failed", "Manifest JSON could not be parsed.", error_type=type(exc).__name__)
        return report
    if not isinstance(value, dict):
        add_issue(errors, "manifest_root_invalid", "Manifest root must be a JSON object.")
        return report

    image_limit, image_limit_error = request_image_limit(value)
    report["budget"]["max_images"] = image_limit
    if image_limit_error:
        add_issue(
            errors,
            image_limit_error,
            "The terminal scan declares an unsupported image budget.",
            declared_max_images=value.get("sampling", {}).get("max_images") if isinstance(value.get("sampling"), dict) else None,
            maximum=CLEANUP_TERMINAL_SCAN_MAX_IMAGES,
        )

    references = extract_references(value, errors)
    if not references:
        add_issue(
            errors,
            "no_media_images",
            "Manifest does not contain any candidate frame or ROI image to send.",
        )
    records = inspect_media(references, manifest_path, errors, warnings)
    decoded_records = [record for record in records if record["decoded"]]
    total_bytes = sum(int(record["file_size_bytes"]) for record in decoded_records)
    report["image_count"] = len(decoded_records)
    report["total_jpeg_bytes"] = total_bytes
    report["estimated_base64_bytes"] = sum(
        estimated_base64_bytes(int(record["file_size_bytes"])) for record in decoded_records
    )
    if decoded_records:
        report["max_dimensions"] = {
            "width": max(int(record["width"]) for record in decoded_records),
            "height": max(int(record["height"]) for record in decoded_records),
        }
    if report["image_count"] > image_limit:
        add_issue(
            errors,
            "too_many_images",
            "Request image count exceeds the maximum budget.",
            image_count=report["image_count"],
            max_images=image_limit,
        )
    if total_bytes > MAX_TOTAL_JPEG_BYTES:
        add_issue(
            errors,
            "total_payload_too_large",
            "Total JPEG payload exceeds the maximum budget.",
            total_jpeg_bytes=total_bytes,
            max_total_jpeg_bytes=MAX_TOTAL_JPEG_BYTES,
        )
    if report["estimated_base64_bytes"] > MAX_ESTIMATED_BASE64_BYTES:
        add_issue(
            errors,
            "estimated_base64_payload_too_large",
            "Estimated Base64 payload exceeds the maximum budget.",
            estimated_base64_bytes=report["estimated_base64_bytes"],
            max_estimated_base64_bytes=MAX_ESTIMATED_BASE64_BYTES,
        )
    report["duplicate_groups"] = exact_duplicate_groups(decoded_records, warnings)
    report["near_duplicate_groups"] = near_duplicate_groups(decoded_records, warnings)
    report["redundant_roi_count"] = redundant_full_frame_rois(decoded_records, errors)
    report["oversized_source_roi_count"] = oversized_source_rois(references, errors)
    report["media"] = [media_report(record) for record in records]
    report["referenced_media_count"] = len(references)
    report["valid"] = not errors
    report["request_should_be_sent"] = report["valid"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight Qwen image payloads without calling Qwen.")
    parser.add_argument("--manifest", required=True, help="Evidence-search manifest JSON path.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report = build_report(manifest_path)
    write_json(output_path, report)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
