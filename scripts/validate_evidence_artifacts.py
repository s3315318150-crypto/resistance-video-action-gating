#!/usr/bin/env python3
"""Validate local JSON evidence artifacts without invoking models or media tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROI_VALUES = {"full", "partial", "not_visible"}
DECISION_VALUES = {"pass", "fail", "abstained"}
AUTOMATED_OUTCOME_VALUES = {"scored", "abstained"}
SWITCH_STATE_VALUES = {"open", "closed", "uncertain"}
NEEDLE_STATE_V2_VALUES = {
    "normal_rightward",
    "zero",
    "reverse",
    "overrange",
    "uncertain",
}
NEEDLE_DIRECTION_VALUES = {
    "right",
    "rightward",
    "positive",
    "positive_rightward",
    "zero",
    "reverse",
    "uncertain",
}
EVALUATION_STATUS_VALUES = {"matched", "mismatched", "ground_truth_not_found"}
ARTIFACT_TYPES = {
    "frame_manifest",
    "evidence_availability_review",
    "dynamic_roi_results",
    "qwen_structured_result",
    "evidence_package",
    "prediction_package",
    "offline_evaluation",
    "validation_gate",
}

# These keys describe files produced by the local evidence pipeline.  source_video
# and workbook_path are deliberately excluded: the validator must not touch videos
# or Excel files.
LOCAL_PATH_KEYS = {
    "contact_sheet_path",
    "output_path",
    "debug_image",
    "enhanced_crop",
    "evidence_availability_review",
    "evidence_package_path",
    "evidence_review",
    "frame_manifest",
    "frame_path",
    "frozen_derived_rule_source",
    "holdout_rubric_config",
    "input_images",
    "original_crop",
    "prediction_package_path",
    "qwen_aggregation_results",
    "qwen_raw_response",
    "qwen_validated_result",
    "raw_response_path",
    "rubric_config",
    "selected_source_frame",
    "source_frame",
}
LOCAL_PATH_SUFFIXES = ("_path", "_file", "_file_path")
NO_DEREFERENCE_PATH_KEYS = {"source_video", "workbook_path"}
FORBIDDEN_QWEN_KEYS = {
    "score",
    "predicted_score",
    "automated_decision",
    "predicted_decision",
    "final_decision",
    "final_rubric_decision",
    "rubric_decision",
}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
CONTRACT_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline" / "evidence_artifact_contract_v1.json"


class Report:
    """Collect deterministic, machine-readable validation results."""

    def __init__(self, artifact_type: str, input_path: Path) -> None:
        self.artifact_type = artifact_type
        self.input_path = str(input_path)
        self.errors: list[dict[str, str]] = []
        self.warnings: list[dict[str, str]] = []
        self.checked_rules: list[str] = []
        self._seen_rules: set[str] = set()

    def rule(self, name: str) -> None:
        if name not in self._seen_rules:
            self._seen_rules.add(name)
            self.checked_rules.append(name)

    def error(self, code: str, path: str, message: str) -> None:
        self.errors.append({"code": code, "path": path, "message": message})

    def warning(self, code: str, path: str, message: str) -> None:
        self.warnings.append({"code": code, "path": path, "message": message})

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": not self.errors,
            "errors": self.errors,
            "warnings": self.warnings,
            "artifact_type": self.artifact_type,
            "input_path": self.input_path,
            "checked_rules": self.checked_rules,
        }


def is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def is_present(value: Any) -> bool:
    return value is not None and value != ""


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def as_path(value: str, input_path: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else input_path.parent / candidate


def is_local_path_key(key: str) -> bool:
    return key in LOCAL_PATH_KEYS or key.endswith(LOCAL_PATH_SUFFIXES)


def looks_like_local_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith(("\\\\", "./", "../"))


def check_required(
    report: Report,
    data: dict[str, Any],
    fields: Iterable[str],
    root: str = "$",
    nullable: Iterable[str] = (),
) -> None:
    report.rule("required_fields")
    nullable_set = set(nullable)
    for field in fields:
        if field not in data or (data[field] is None and field not in nullable_set):
            report.error("MISSING_REQUIRED_FIELD", f"{root}.{field}", "Required field is missing or null.")


def check_enum(report: Report, value: Any, allowed: set[str], path: str, code: str) -> None:
    report.rule("enum_values")
    if value is not None and value not in allowed:
        report.error(code, path, f"Expected one of {sorted(allowed)}; received {value!r}.")


def find_local_path_references(
    value: Any,
    input_path: Path,
    report: Report,
    path: str = "$",
    key: str | None = None,
) -> None:
    """Stat referenced local files without opening media, videos, or workbooks."""
    if is_mapping(value):
        for child_key, child_value in value.items():
            child_path = f"{path}.{child_key}"
            if child_key in NO_DEREFERENCE_PATH_KEYS and isinstance(child_value, str):
                report.rule("external_sources_not_dereferenced")
                report.warning(
                    "EXTERNAL_REFERENCE_NOT_DEREFERENCED",
                    child_path,
                    f"{child_key} is recorded for provenance but is not opened by this validator.",
                )
                continue
            find_local_path_references(child_value, input_path, report, child_path, child_key)
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            find_local_path_references(item, input_path, report, f"{path}[{index}]", key)
        return

    if not isinstance(value, str) or not key or not (is_local_path_key(key) or looks_like_local_path(value)):
        return
    if value.startswith(("http://", "https://")):
        report.warning("NONLOCAL_PATH_REFERENCE", path, "URL was not checked as a local artifact path.")
        return

    report.rule("local_path_references")
    referenced_path = as_path(value, input_path)
    if not referenced_path.exists():
        report.error("MISSING_REFERENCED_PATH", path, f"Referenced local path does not exist: {referenced_path}")


def check_schema_metadata(data: dict[str, Any], report: Report) -> None:
    report.rule("schema_metadata")
    if "schema_version" not in data:
        report.warning(
            "LEGACY_SCHEMA_METADATA_MISSING",
            "$.schema_version",
            "Historical artifact has no schema_version; compatibility rules were applied.",
        )
    if "artifact_type" not in data:
        report.warning(
            "LEGACY_SCHEMA_METADATA_MISSING",
            "$.artifact_type",
            "Historical artifact has no artifact_type; the CLI artifact type was used.",
        )


def check_common_enums(data: dict[str, Any], report: Report, path: str = "$") -> None:
    """Check field-name-specific enums without conflating distinct status vocabularies."""
    if is_mapping(data):
        for key, value in data.items():
            child_path = f"{path}.{key}"
            if key in {"roi_status", "dynamic_roi_status", "semantic_visibility"}:
                check_enum(report, value, ROI_VALUES, child_path, "INVALID_ROI_STATUS")
            elif key in {"automated_decision", "predicted_decision"}:
                check_enum(report, value, DECISION_VALUES, child_path, "INVALID_DECISION")
            elif key == "circuit_state":
                check_enum(report, value, SWITCH_STATE_VALUES, child_path, "INVALID_CIRCUIT_STATE")
            elif key == "needle_state":
                check_enum(report, value, NEEDLE_STATE_V2_VALUES, child_path, "INVALID_NEEDLE_STATE_V2")
            elif key in {"needle_direction", "needle_direction_raw", "needle_direction_normalized"}:
                check_enum(report, value, NEEDLE_DIRECTION_VALUES, child_path, "INVALID_NEEDLE_DIRECTION")
            elif key == "state" and (
                "switch" in path.lower() or "switch" in key.lower() or "assessment" in path.lower()
            ):
                check_enum(report, value, SWITCH_STATE_VALUES, child_path, "INVALID_SWITCH_STATE")
            check_common_enums(value, report, child_path)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            check_common_enums(item, report, f"{path}[{index}]")


def validate_frame_manifest(data: dict[str, Any], report: Report) -> None:
    check_required(report, data, ("source_video", "frames"))
    report.rule("frame_manifest_structure")
    if not isinstance(data.get("frames"), list):
        report.error("INVALID_FRAMES", "$.frames", "frames must be an array.")
        return
    for index, frame in enumerate(data["frames"]):
        if not is_mapping(frame):
            report.error("INVALID_FRAME_RECORD", f"$.frames[{index}]", "Frame record must be an object.")
            continue
        check_required(report, frame, ("timestamp_seconds",), f"$.frames[{index}]")
        if not ("output_path" in frame or "frame_path" in frame):
            report.error(
                "MISSING_FRAME_OUTPUT_PATH",
                f"$.frames[{index}]",
                "Frame record must include output_path or frame_path.",
            )


def validate_evidence_availability_review(data: dict[str, Any], report: Report) -> None:
    check_required(report, data, ("rubric_id", "source_video", "candidate_reviews", "review_method"))
    report.rule("evidence_availability_review_structure")
    candidates = data.get("candidate_reviews")
    if not isinstance(candidates, list):
        report.error("INVALID_CANDIDATE_REVIEWS", "$.candidate_reviews", "candidate_reviews must be an array.")
        return
    for index, candidate in enumerate(candidates):
        if not is_mapping(candidate):
            report.error("INVALID_CANDIDATE_REVIEW", f"$.candidate_reviews[{index}]", "Candidate review must be an object.")
            continue
        check_required(report, candidate, ("timestamp_seconds", "frame_path", "usable_for_qwen_evidence"), f"$.candidate_reviews[{index}]")
        for field in ("switch_visibility", "voltmeter_visibility", "ammeter_visibility"):
            if field in candidate:
                check_enum(report, candidate[field], ROI_VALUES, f"$.candidate_reviews[{index}].{field}", "INVALID_ROI_STATUS")


def device_mapping(frame: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    devices = frame.get("devices")
    if is_mapping(devices):
        return devices, False
    direct = {name: frame[name] for name in ("voltmeter", "ammeter", "switch") if name in frame}
    return (direct or None), bool(direct)


def device_bbox(device: dict[str, Any]) -> Any:
    """Accept the two dynamic-ROI bbox layouts produced by existing pilots."""
    mapped_bbox = device.get("mapped_bbox")
    if mapped_bbox is not None:
        return mapped_bbox

    bbox = device.get("bbox")
    if is_mapping(bbox):
        return first_value(bbox, ("mapped_bbox_xyxy", "mapped_bbox"))
    return bbox


def validate_dynamic_roi_results(data: dict[str, Any], report: Report) -> None:
    check_required(report, data, ("frames",))
    report.rule("dynamic_roi_device_geometry")
    frames = data.get("frames")
    if not isinstance(frames, list):
        report.error("INVALID_FRAMES", "$.frames", "frames must be an array.")
        return
    for frame_index, frame in enumerate(frames):
        frame_path = f"$.frames[{frame_index}]"
        if not is_mapping(frame):
            report.error("INVALID_FRAME_RECORD", frame_path, "Frame record must be an object.")
            continue
        check_required(report, frame, ("timestamp_seconds",), frame_path)
        devices, legacy_direct = device_mapping(frame)
        if devices is None:
            report.error("MISSING_DEVICE_RECORDS", frame_path, "Expected a devices object with all three devices.")
            continue
        if legacy_direct:
            report.warning(
                "LEGACY_DYNAMIC_DEVICE_LAYOUT",
                frame_path,
                "Accepted direct device fields; current contract uses frame.devices.",
            )
        for name in ("voltmeter", "ammeter", "switch"):
            device_path = f"{frame_path}.devices.{name}"
            device = devices.get(name)
            if not is_mapping(device):
                report.error("MISSING_DEVICE_RESULT", device_path, "Device result must be an object.")
                continue
            status = device.get("status")
            check_enum(report, status, ROI_VALUES, f"{device_path}.status", "INVALID_ROI_STATUS")
            if "semantic_visibility" in device:
                check_enum(
                    report,
                    device["semantic_visibility"],
                    ROI_VALUES,
                    f"{device_path}.semantic_visibility",
                    "INVALID_ROI_STATUS",
                )
            if "switch_state_assessment_allowed" in device:
                allowed = device["switch_state_assessment_allowed"]
                if allowed is not None and not isinstance(allowed, bool):
                    report.error(
                        "INVALID_SWITCH_ASSESSMENT_PERMISSION",
                        f"{device_path}.switch_state_assessment_allowed",
                        "switch_state_assessment_allowed must be true, false, or null.",
                    )
            if status == "not_visible":
                for crop_name in ("original_crop", "enhanced_crop"):
                    if is_present(device.get(crop_name)):
                        report.error(
                            "CROP_PRESENT_FOR_NOT_VISIBLE",
                            f"{device_path}.{crop_name}",
                            "not_visible devices must not include a crop path.",
                        )
            elif status in {"full", "partial"}:
                bbox = device_bbox(device)
                if not (isinstance(bbox, list) and len(bbox) == 4 and all(is_number(item) for item in bbox)):
                    report.error(
                        "MISSING_OR_INVALID_BBOX",
                        device_path,
                        "full or partial device must include a four-number mapped_bbox, bbox, or bbox.mapped_bbox_xyxy.",
                    )
                for field in ("inlier_count", "area_ratio_to_reference", "visible_area_ratio"):
                    if not is_number(device.get(field)):
                        report.error(
                            "MISSING_OR_INVALID_GEOMETRY",
                            f"{device_path}.{field}",
                            "full or partial device requires a numeric geometry field.",
                        )


def qwen_records(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    evaluations = data.get("evaluations")
    if isinstance(evaluations, list):
        return [(f"$.evaluations[{index}]", item) for index, item in enumerate(evaluations) if is_mapping(item)]
    return [("$", data)]


def find_forbidden_qwen_fields(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if is_mapping(value):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_QWEN_KEYS:
                yield key, child_path
            yield from find_forbidden_qwen_fields(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from find_forbidden_qwen_fields(child, f"{path}[{index}]")


def validate_qwen_structured_result(data: dict[str, Any], report: Report) -> None:
    report.rule("qwen_parse_and_safety")
    if not isinstance(data.get("evaluations"), list) and "json_parsed" not in data:
        report.error(
            "MISSING_QWEN_RESULT_RECORDS",
            "$",
            "Expected evaluations array or a single json_parsed result record.",
        )
    if isinstance(data.get("evaluations"), list) and any(not is_mapping(item) for item in data["evaluations"]):
        report.error("INVALID_QWEN_EVALUATION_RECORD", "$.evaluations", "Every evaluation must be an object.")
    for record_path, record in qwen_records(data):
        parsed = record.get("json_parsed")
        safety_valid = record.get("safety_valid")
        if not isinstance(parsed, bool):
            report.error("INVALID_JSON_PARSED", f"{record_path}.json_parsed", "json_parsed must be a boolean.")
        if not isinstance(safety_valid, bool):
            report.error("INVALID_SAFETY_VALID", f"{record_path}.safety_valid", "safety_valid must be a boolean.")
        structured = record.get("structured_result", record.get("result"))
        if parsed is True and not is_mapping(structured):
            report.error(
                "MISSING_STRUCTURED_RESULT",
                record_path,
                "json_parsed=true requires a structured_result or result object.",
            )
        if safety_valid is True:
            errors = record.get("validation_errors")
            if not isinstance(errors, list) or errors:
                report.error(
                    "SAFETY_VALID_WITH_ERRORS",
                    f"{record_path}.validation_errors",
                    "safety_valid=true requires an empty validation_errors array.",
                )
        if is_mapping(structured):
            for assessment_name, assessment in structured.items():
                if not assessment_name.endswith("_assessment") or not is_mapping(assessment):
                    continue
                roi_status = assessment.get("roi_status")
                check_enum(report, roi_status, ROI_VALUES, f"{record_path}.{assessment_name}.roi_status", "INVALID_ROI_STATUS")
                if roi_status == "not_visible" and assessment.get("status") != "not_assessable":
                    report.error(
                        "NOT_VISIBLE_ASSESSMENT_NOT_REJECTED",
                        f"{record_path}.{assessment_name}.status",
                        "A not_visible ROI must be marked not_assessable.",
                    )
            for key, key_path in find_forbidden_qwen_fields(structured, record_path):
                report.error(
                    "FORBIDDEN_QWEN_SCORING_FIELD",
                    key_path,
                    f"Qwen structured result must not contain {key}.",
                )


def meter_evidence(data: dict[str, Any], device: str) -> dict[str, Any] | None:
    grouped = data.get("meter_evidence")
    if is_mapping(grouped) and is_mapping(grouped.get(device)):
        return grouped[device]
    direct = data.get(f"{device}_evidence")
    return direct if is_mapping(direct) else None


def first_value(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def evidence_status(meter: dict[str, Any]) -> Any:
    return first_value(meter, ("assessment_status", "status"))


def evidence_roi(meter: dict[str, Any]) -> Any:
    return first_value(meter, ("dynamic_roi_status", "roi_status"))


def evidence_direction(meter: dict[str, Any]) -> Any:
    return first_value(meter, ("needle_direction_normalized", "needle_direction", "needle_direction_raw"))


def direct_switch_claimed(data: dict[str, Any], circuit: dict[str, Any]) -> tuple[bool, str | None]:
    direct = circuit.get("direct_switch_evidence")
    claim = False
    status: str | None = None
    if is_mapping(direct):
        claim = direct.get("available") is True or direct.get("established") is True
        status = direct.get("status")
    switch = data.get("switch_evidence")
    if is_mapping(switch):
        claim = claim or switch.get("direct_switch_closure_claimed") is True or switch.get("used_for_decision") is True
        status = status or switch.get("assessment_status")
        structured = switch.get("structured_assessment")
        if is_mapping(structured):
            status = status or structured.get("status")
    if circuit.get("method") == "direct_switch_evidence":
        claim = True
    return claim, status


def check_derived_meter_evidence(data: dict[str, Any], circuit: dict[str, Any], report: Report) -> None:
    report.rule("derived_meter_evidence")
    derived = circuit.get("derived_meter_evidence")
    if not is_mapping(derived):
        report.error(
            "MISSING_DERIVED_METER_EVIDENCE",
            "$.circuit_closed_evidence.derived_meter_evidence",
            "derived_meter_evidence method requires a structured derived_meter_evidence object.",
        )
        return
    established = first_value(derived, ("established", "available"))
    if established is not True:
        report.error(
            "DERIVED_METER_EVIDENCE_NOT_ESTABLISHED",
            "$.circuit_closed_evidence.derived_meter_evidence",
            "Derived meter evidence must be explicitly established or available.",
        )
    scope_note = circuit.get("scope_note")
    if not isinstance(scope_note, str) or not scope_note.strip():
        report.error(
            "MISSING_DERIVED_SCOPE_NOTE",
            "$.circuit_closed_evidence.scope_note",
            "Derived meter evidence requires a non-empty scope_note.",
        )
    elif not any(token in scope_note.lower() for token in ("switch", "mechanical", "rubric", "开关", "机械")):
        report.warning(
            "DERIVED_SCOPE_NOTE_NOT_EXPLICIT",
            "$.circuit_closed_evidence.scope_note",
            "scope_note exists but does not explicitly limit reuse for direct mechanical switch assessment.",
        )

    criteria = derived.get("criteria") if is_mapping(derived.get("criteria")) else {}
    voltmeter = meter_evidence(data, "voltmeter")
    ammeter = meter_evidence(data, "ammeter")
    for name, meter in (("voltmeter", voltmeter), ("ammeter", ammeter)):
        meter_path = f"$.{name}_evidence"
        if meter is None:
            report.error("MISSING_METER_EVIDENCE", meter_path, "Derived evidence requires both meter evidence objects.")
            continue
        if evidence_roi(meter) != "full":
            report.error("DERIVED_METER_ROI_NOT_FULL", meter_path, "Derived evidence requires a full meter ROI.")
        if evidence_status(meter) != "assessable":
            report.error("DERIVED_METER_NOT_ASSESSABLE", meter_path, "Derived evidence requires assessable meter status.")
        if meter.get("needle_deflected") is not True:
            report.error("DERIVED_NEEDLE_NOT_DEFLECTED", meter_path, "Derived evidence requires needle_deflected=true.")
        direction = evidence_direction(meter)
        if direction not in {"right", "rightward"}:
            report.error(
                "DERIVED_NEEDLE_DIRECTION_NOT_RIGHTWARD",
                meter_path,
                "Derived evidence requires needle direction right or rightward.",
            )

    # The reference package may express conditions in a criteria object.  If it does,
    # insist that it agrees with the per-meter evidence rather than trusting it alone.
    expected_criteria = {
        "voltmeter_dynamic_roi_status": "full",
        "ammeter_dynamic_roi_status": "full",
        "voltmeter_assessment_status": "assessable",
        "ammeter_assessment_status": "assessable",
        "voltmeter_needle_deflected": True,
        "ammeter_needle_deflected": True,
        "voltmeter_needle_direction_normalized": "right",
        "ammeter_needle_direction_normalized": "right",
    }
    for key, expected in expected_criteria.items():
        if key in criteria and criteria[key] != expected:
            report.error(
                "DERIVED_CRITERIA_CONFLICT",
                f"$.circuit_closed_evidence.derived_meter_evidence.criteria.{key}",
                f"Expected {expected!r}; received {criteria[key]!r}.",
            )

    declared_same_frame = first_value(derived, ("same_source_frame",))
    if declared_same_frame is None:
        declared_same_frame = criteria.get("same_source_frame")
    source_frames = []
    for meter in (voltmeter, ammeter):
        if meter is not None and is_present(meter.get("source_frame")):
            source_frames.append(meter["source_frame"])
    common_frame = first_value(derived, ("source_frame",)) or data.get("selected_source_frame") or data.get("source_frame")
    if declared_same_frame is not True:
        report.error(
            "DERIVED_METERS_NOT_SAME_FRAME",
            "$.circuit_closed_evidence.derived_meter_evidence",
            "Derived evidence must explicitly declare same_source_frame=true.",
        )
    if len(source_frames) == 2 and source_frames[0] != source_frames[1]:
        report.error(
            "DERIVED_METER_SOURCE_FRAME_MISMATCH",
            "$.voltmeter_evidence.source_frame",
            "Voltmeter and ammeter evidence reference different source frames.",
        )
    elif len(source_frames) < 2 and not is_present(common_frame):
        report.error(
            "MISSING_DERIVED_SOURCE_FRAME",
            "$.source_frame",
            "Derived evidence requires a shared source frame reference.",
        )
    elif len(source_frames) < 2:
        report.warning(
            "LEGACY_SHARED_SOURCE_FRAME",
            "$.source_frame",
            "Using the package-level source frame with same_source_frame=true for legacy meter records.",
        )

    safety_values = []
    for meter in (voltmeter, ammeter):
        if meter is not None and "safety_valid" in meter:
            safety_values.append(meter["safety_valid"])
    if "safety_valid" in criteria:
        safety_values.append(criteria["safety_valid"])
    if "local_safety_validation_passed" in derived:
        safety_values.append(derived["local_safety_validation_passed"])
    decision_validation = data.get("decision_validation")
    if is_mapping(decision_validation) and "local_safety_validation_passed" in decision_validation:
        safety_values.append(decision_validation["local_safety_validation_passed"])
    if not safety_values or any(value is not True for value in safety_values):
        report.error(
            "DERIVED_METER_SAFETY_NOT_VALID",
            "$.circuit_closed_evidence.derived_meter_evidence",
            "Derived evidence requires successful local safety validation.",
        )


def validate_evidence_or_prediction_package(data: dict[str, Any], report: Report, artifact_type: str) -> None:
    base_fields = (
        "rubric_id",
        "source_video",
        "automated_decision",
        "automated_outcome",
        "abstention_reason",
        "predicted_score",
    )
    check_required(report, data, base_fields, nullable=("predicted_score", "abstention_reason"))
    if artifact_type == "prediction_package":
        check_required(report, data, ("predicted_score", "circuit_closed_evidence", "provenance"), nullable=("predicted_score",))
    report.rule("decision_and_score_consistency")
    decision = data.get("automated_decision")
    check_enum(report, decision, DECISION_VALUES, "$.automated_decision", "INVALID_DECISION")
    score = data.get("predicted_score") if "predicted_score" in data else data.get("score")
    outcome = data.get("automated_outcome")
    check_enum(report, outcome, AUTOMATED_OUTCOME_VALUES, "$.automated_outcome", "INVALID_AUTOMATED_OUTCOME")
    if decision == "abstained" and is_present(score):
        report.error(
            "ABSTAINED_WITH_SCORE",
            "$.predicted_score" if "predicted_score" in data else "$.score",
            "abstained must use a null or absent score.",
        )
    if outcome == "abstained":
        if is_present(score):
            report.error(
                "ABSTAINED_OUTCOME_WITH_SCORE",
                "$.predicted_score" if "predicted_score" in data else "$.score",
                "automated_outcome=abstained requires a null score.",
            )
        reason = data.get("abstention_reason")
        if not isinstance(reason, str) or not reason.strip():
            report.error(
                "ABSTENTION_REASON_MISSING",
                "$.abstention_reason",
                "automated_outcome=abstained requires a non-empty abstention_reason.",
            )
    if outcome == "scored" and (isinstance(score, bool) or not isinstance(score, (int, float)) or score not in (0, 1)):
        report.error(
            "SCORED_OUTCOME_INVALID_SCORE",
            "$.predicted_score" if "predicted_score" in data else "$.score",
            "automated_outcome=scored requires predicted_score 0 or 1.",
        )
    if outcome == "scored" and data.get("abstention_reason") is not None:
        report.error(
            "SCORED_OUTCOME_WITH_ABSTENTION_REASON",
            "$.abstention_reason",
            "automated_outcome=scored requires abstention_reason=null.",
        )
    if (decision == "abstained") != (outcome == "abstained"):
        report.error(
            "DECISION_OUTCOME_MISMATCH",
            "$.automated_outcome",
            "automated_decision=abstained and automated_outcome=abstained must agree.",
        )

    circuit = data.get("circuit_closed_evidence")
    if circuit is None and decision == "abstained":
        report.warning(
            "ABSTAINED_WITHOUT_CIRCUIT_EVIDENCE",
            "$.circuit_closed_evidence",
            "Abstained package has no circuit evidence; pass/fail evidence checks were not applied.",
        )
        return
    if not is_mapping(circuit):
        report.error("MISSING_CIRCUIT_EVIDENCE", "$.circuit_closed_evidence", "Expected a circuit_closed_evidence object.")
        return

    report.rule("switch_evidence_scope")
    direct_claim, switch_status = direct_switch_claimed(data, circuit)
    if switch_status == "not_assessable" and direct_claim:
        report.error(
            "DIRECT_SWITCH_FROM_NOT_ASSESSABLE",
            "$.circuit_closed_evidence.direct_switch_evidence",
            "A not_assessable switch cannot be recorded as direct_switch_evidence.",
        )
    if circuit.get("method") == "derived_meter_evidence":
        check_derived_meter_evidence(data, circuit, report)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_contract_definition(report: Report) -> None:
    """Confirm that the checked rules have their companion local contract."""
    report.rule("contract_definition")
    try:
        with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
            contract = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        report.error("CONTRACT_UNAVAILABLE", str(CONTRACT_PATH), f"Could not load v1 contract: {exc}")
        return
    if not is_mapping(contract) or contract.get("contract_id") != "evidence_artifact_contract_v1":
        report.error("INVALID_CONTRACT", str(CONTRACT_PATH), "Contract id must be evidence_artifact_contract_v1.")
        return
    artifact_types = contract.get("artifact_types")
    if not is_mapping(artifact_types) or not ARTIFACT_TYPES.issubset(artifact_types):
        report.error("INVALID_CONTRACT", str(CONTRACT_PATH), "Contract must define every supported artifact type.")
    enums = contract.get("common_enums")
    expected_enums = {
        "roi_status": ROI_VALUES,
        "decision": DECISION_VALUES,
        "switch_state": SWITCH_STATE_VALUES,
        "needle_state_v2": NEEDLE_STATE_V2_VALUES,
    }
    if not is_mapping(enums):
        report.error("INVALID_CONTRACT", str(CONTRACT_PATH), "Contract must define common_enums.")
        return
    for name, expected_values in expected_enums.items():
        if set(enums.get(name, [])) != expected_values:
            report.error(
                "CONTRACT_ENUM_MISMATCH",
                f"{CONTRACT_PATH}:{name}",
                "Contract enum differs from validator enum.",
            )


def validate_offline_evaluation(data: dict[str, Any], report: Report, input_path: Path) -> None:
    check_required(
        report,
        data,
        (
            "source_video",
            "rubric_id",
            "prediction_package_path",
            "prediction_package_sha256",
            "predicted_decision",
            "predicted_score",
            "evaluation_status",
            "prediction_frozen_before_ground_truth_access",
            "ground_truth_shared_with_model",
        ),
        nullable=("predicted_score",),
    )
    report.rule("offline_evaluation_freeze_invariants")
    check_enum(report, data.get("predicted_decision"), DECISION_VALUES, "$.predicted_decision", "INVALID_DECISION")
    check_enum(report, data.get("evaluation_status"), EVALUATION_STATUS_VALUES, "$.evaluation_status", "INVALID_EVALUATION_STATUS")
    sha = data.get("prediction_package_sha256")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        report.error("INVALID_PREDICTION_SHA256", "$.prediction_package_sha256", "Expected a 64-character SHA-256 hex string.")
    if data.get("prediction_frozen_before_ground_truth_access") is not True:
        report.error(
            "PREDICTION_NOT_FROZEN_BEFORE_GROUND_TRUTH",
            "$.prediction_frozen_before_ground_truth_access",
            "Offline evaluation must state prediction_frozen_before_ground_truth_access=true.",
        )
    if data.get("ground_truth_shared_with_model") is not False:
        report.error(
            "GROUND_TRUTH_SHARED_WITH_MODEL",
            "$.ground_truth_shared_with_model",
            "Offline evaluation must state ground_truth_shared_with_model=false.",
        )
    if data.get("upstream_files_modified") is True:
        report.error("UPSTREAM_FILES_MODIFIED", "$.upstream_files_modified", "Offline evaluation must not modify upstream files.")

    prediction_path_value = data.get("prediction_package_path")
    if isinstance(prediction_path_value, str):
        prediction_path = as_path(prediction_path_value, input_path)
        if prediction_path.exists() and isinstance(sha, str) and SHA256_RE.fullmatch(sha):
            report.rule("offline_prediction_hash_binding")
            actual = sha256_file(prediction_path)
            if actual.lower() != sha.lower():
                report.error(
                    "PREDICTION_SHA256_MISMATCH",
                    "$.prediction_package_sha256",
                    "prediction_package_sha256 does not match the referenced prediction package.",
                )


def validate_validation_gate(data: dict[str, Any], report: Report, input_path: Path) -> None:
    check_required(report, data, ("rubric_id", "status", "frozen_artifacts", "required_new_data", "prohibited_actions"))
    report.rule("validation_gate_requirements")
    if data.get("status") != "blocked_waiting_for_data":
        report.warning(
            "UNRECOGNIZED_VALIDATION_GATE_STATUS",
            "$.status",
            "This v1 gate has specialized requirements for blocked_waiting_for_data.",
        )
        return
    for field in ("required_new_data", "prohibited_actions", "frozen_artifacts"):
        if not isinstance(data.get(field), list) or not data[field]:
            report.error(
                "MISSING_BLOCKED_GATE_REQUIREMENT",
                f"$.{field}",
                "blocked_waiting_for_data requires a non-empty list.",
            )
    frozen_artifacts = data.get("frozen_artifacts")
    if isinstance(frozen_artifacts, list):
        for index, artifact in enumerate(frozen_artifacts):
            artifact_path = f"$.frozen_artifacts[{index}]"
            if not is_mapping(artifact):
                report.error("INVALID_FROZEN_ARTIFACT", artifact_path, "Frozen artifact entry must be an object.")
                continue
            if not isinstance(artifact.get("path"), str) or not isinstance(artifact.get("sha256"), str):
                report.error(
                    "INVALID_FROZEN_ARTIFACT",
                    artifact_path,
                    "Frozen artifact must include string path and sha256 fields.",
                )
                continue
            expected_hash = artifact["sha256"]
            if not SHA256_RE.fullmatch(expected_hash):
                report.error("INVALID_FROZEN_ARTIFACT_SHA256", f"{artifact_path}.sha256", "Expected SHA-256 hex string.")
                continue
            referenced_path = as_path(artifact["path"], input_path)
            if referenced_path.exists():
                report.rule("frozen_artifact_hash_binding")
                actual_hash = sha256_file(referenced_path)
                if actual_hash.lower() != expected_hash.lower():
                    report.error(
                        "FROZEN_ARTIFACT_SHA256_MISMATCH",
                        f"{artifact_path}.sha256",
                        "Frozen artifact SHA-256 does not match the referenced file.",
                    )
    v2_schema = data.get("v2_schema")
    if isinstance(v2_schema, list):
        for index, state in enumerate(v2_schema):
            check_enum(report, state, NEEDLE_STATE_V2_VALUES, f"$.v2_schema[{index}]", "INVALID_NEEDLE_STATE_V2")


def validate_artifact(data: dict[str, Any], artifact_type: str, input_path: Path, report: Report) -> None:
    check_schema_metadata(data, report)
    check_common_enums(data, report)
    if artifact_type == "frame_manifest":
        validate_frame_manifest(data, report)
    elif artifact_type == "evidence_availability_review":
        validate_evidence_availability_review(data, report)
    elif artifact_type == "dynamic_roi_results":
        validate_dynamic_roi_results(data, report)
    elif artifact_type == "qwen_structured_result":
        validate_qwen_structured_result(data, report)
    elif artifact_type in {"evidence_package", "prediction_package"}:
        validate_evidence_or_prediction_package(data, report, artifact_type)
    elif artifact_type == "offline_evaluation":
        validate_offline_evaluation(data, report, input_path)
    elif artifact_type == "validation_gate":
        validate_validation_gate(data, report, input_path)


def read_json(path: Path, report: Report) -> dict[str, Any] | None:
    report.rule("json_parse")
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        report.error("INPUT_NOT_FOUND", "$", f"Input JSON does not exist: {path}")
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        report.error("JSON_PARSE_ERROR", "$", f"Could not parse input JSON: {exc}")
        return None
    if not is_mapping(data):
        report.error("INVALID_ROOT_TYPE", "$", "Artifact root must be a JSON object.")
        return None
    return data


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate local JSON evidence artifacts without reading videos, Excel, or external services."
    )
    parser.add_argument("--artifact-type", choices=sorted(ARTIFACT_TYPES), required=True)
    parser.add_argument("--input", required=True, help="Absolute or relative JSON artifact path.")
    parser.add_argument("--output", required=True, help="JSON validation report path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if input_path == output_path:
        print("Refusing to overwrite the input artifact with a validation report.", file=sys.stderr)
        return 2

    report = Report(args.artifact_type, input_path)
    report.rule("input_read_only")
    check_contract_definition(report)
    data = read_json(input_path, report)
    if data is not None:
        find_local_path_references(data, input_path, report)
        validate_artifact(data, args.artifact_type, input_path, report)
    write_report(output_path, report.as_dict())
    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
