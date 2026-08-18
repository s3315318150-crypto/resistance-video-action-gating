#!/usr/bin/env python3
"""Run one label-blind Qwen observation after deterministic media preflight."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

import preflight_qwen_request as preflight


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    target = path.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def qwen_settings() -> tuple[str, str, str]:
    base_url = os.getenv("QWEN_API_BASE_URL", "").strip()
    token = os.getenv("QWEN_API_TOKEN", "").strip()
    model = os.getenv("QWEN_MODEL", "qwen").strip() or "qwen"
    missing = [name for name, value in (("QWEN_API_BASE_URL", base_url), ("QWEN_API_TOKEN", token)) if not value]
    if missing:
        raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))
    return base_url, token, model


def safe_error_message(exc: Exception, base_url: str) -> str:
    message = str(exc)
    return message.replace(base_url, "<QWEN_API_BASE_URL>") if base_url else message


def data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def strip_json_fence(content: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)


def parse_model_json(content: str) -> dict[str, Any]:
    value = json.loads(strip_json_fence(content))
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(config.get("rubric_id"), str) or not config["rubric_id"].strip():
        errors.append("rubric_id_invalid")
    if not isinstance(config.get("observation_instruction"), str) or not config["observation_instruction"].strip():
        errors.append("observation_instruction_invalid")
    values = config.get("observation_enum")
    if not isinstance(values, list) or len(values) < 2 or any(not isinstance(item, str) or not item for item in values):
        errors.append("observation_enum_invalid")
    enum_values = values if isinstance(values, list) else []
    if any(item in {"pass", "fail"} for item in enum_values):
        errors.append("observation_enum_must_not_contain_score_decisions")
    return errors


def candidate_ids(manifest: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for index, candidate in enumerate(manifest.get("selected_candidates", []), start=1):
        if not isinstance(candidate, dict):
            continue
        candidate_id = candidate.get("candidate_id")
        result.append(candidate_id if isinstance(candidate_id, str) and candidate_id else f"candidate_{index:03d}")
    return result


def build_prompt(config: dict[str, Any], ids: list[str]) -> str:
    enum_text = " | ".join(config["observation_enum"])
    rules = config.get("evidence_rules")
    rule_text = "\n".join(f"- {item}" for item in rules) if isinstance(rules, list) else "- Only use directly visible evidence."
    return f"""You are making a label-blind visual observation for rubric `{config['rubric_id']}`.

Task:
{config['observation_instruction']}

Evidence rules:
{rule_text}
- Do not infer failure from an occluded or missing object.
- If the evidence is unreadable, incomplete, or temporally insufficient, use `uncertain` when that value is available.
- Do not output a score, pass/fail decision, student identity, or final rubric decision.

Candidate IDs available in this request: {", ".join(ids)}
`observation` must be exactly one of: {enum_text}

Return exactly one JSON object and no Markdown:
{{
  "rubric_id": "{config['rubric_id']}",
  "observation": "{config['observation_enum'][0]}",
  "cited_candidate_ids": ["{ids[0]}"],
  "confidence": 0.0,
  "evidence": "directly visible evidence only",
  "uncertainty": "empty string when none"
}}"""


def validate_response(
    value: dict[str, Any],
    config: dict[str, Any],
    ids: list[str],
) -> list[str]:
    errors: list[str] = []
    if value.get("rubric_id") != config.get("rubric_id"):
        errors.append("rubric_id_mismatch")
    if value.get("observation") not in set(config.get("observation_enum", [])):
        errors.append("observation_invalid")
    cited = value.get("cited_candidate_ids")
    if not isinstance(cited, list) or any(item not in ids for item in cited):
        errors.append("cited_candidate_ids_invalid")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        errors.append("confidence_invalid")
    if not isinstance(value.get("evidence"), str):
        errors.append("evidence_invalid")
    if not isinstance(value.get("uncertainty"), str):
        errors.append("uncertainty_invalid")
    forbidden = {"score", "predicted_score", "automated_decision", "rubric_decision", "final_decision"}

    def contains_forbidden_key(item: Any) -> bool:
        if isinstance(item, dict):
            return bool(forbidden.intersection(item)) or any(contains_forbidden_key(child) for child in item.values())
        if isinstance(item, list):
            return any(contains_forbidden_key(child) for child in item)
        return False

    if contains_forbidden_key(value):
        errors.append("forbidden_scoring_field")
    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--rubric-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=800)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    manifest_path = args.manifest.expanduser().resolve()
    config_path = args.rubric_config.expanduser().resolve()
    manifest = read_object(manifest_path)
    config = read_object(config_path)
    config_errors = validate_config(config)
    media_preflight = preflight.build_report(manifest_path)
    ids = candidate_ids(manifest)
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "qwen_structured_observation",
        "rubric_id": config.get("rubric_id"),
        "manifest_path": str(manifest_path),
        "rubric_config_path": str(config_path),
        "preflight_valid": media_preflight["valid"],
        "preflight_errors": media_preflight["errors"],
        "config_errors": config_errors,
        "qwen_called": False,
        "call_count": 0,
        "json_parsed": False,
        "safety_valid": False,
        "validation_errors": [],
        "observation_status": "not_sent",
        "score_computed": False,
        "labels_accessed": False,
    }
    if config_errors or not media_preflight["valid"]:
        result["observation_status"] = "abstained_before_request"
        result["validation_errors"] = config_errors + [item["code"] for item in media_preflight["errors"]]
        write_new_json(args.output, result)
        return 1

    base_url = ""
    try:
        base_url, token, model = qwen_settings()
        client = OpenAI(base_url=base_url, api_key=token, timeout=180, max_retries=0)
        content: list[dict[str, Any]] = [{"type": "text", "text": build_prompt(config, ids)}]
        for media in media_preflight["media"]:
            reference = media["references"][0]
            content.append(
                {
                    "type": "text",
                    "text": f"candidate_id={reference['candidate_id']}; media_field={reference['field']}",
                }
            )
            content.append({"type": "image_url", "image_url": {"url": data_url(Path(media["path"]))}})
        result["qwen_called"] = True
        result["call_count"] = 1
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=args.max_tokens,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = completion.choices[0]
        raw = choice.message.content or ""
        result["finish_reason"] = choice.finish_reason or "unknown"
        result["raw_model_content"] = raw
        parsed = parse_model_json(raw)
        result["json_parsed"] = True
        result["structured_result"] = parsed
        errors = validate_response(parsed, config, ids)
        result["validation_errors"] = errors
        result["safety_valid"] = not errors
        result["observation_status"] = "completed" if not errors else "abstained_invalid_response"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = safe_error_message(exc, base_url)
        result["observation_status"] = "request_failed"
        result["validation_errors"] = ["qwen_request_or_parse_failed"]
    write_new_json(args.output, result)
    return 0 if result["safety_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
