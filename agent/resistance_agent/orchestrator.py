#!/usr/bin/env python3
"""Deterministic, DeepSeek, and OpenAI schedulers over bounded MCP tools."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from .toolkit import (
        AGENT_ROOT,
        DEFAULT_CONFIG,
        TOOL_SCHEMAS,
        ToolError,
        call_tool,
        load_config,
        read_json,
        redact,
        resolve_inside,
        sanitize_run_id,
        write_json,
    )
except ImportError:
    from toolkit import (  # type: ignore
        AGENT_ROOT,
        DEFAULT_CONFIG,
        TOOL_SCHEMAS,
        ToolError,
        call_tool,
        load_config,
        read_json,
        redact,
        resolve_inside,
        sanitize_run_id,
        write_json,
    )


SYSTEM_PROMPT = """你是伏安法测电阻视频流水线的调度 Agent。
你只选择注册过的 MCP 工具，不直接看像素、不虚构工件、不执行任意命令。
当前 Agent 发布集为 R0-R9；R7/R9 使用同轮记录取证和 R4/R5/R6 前置门控。
replay 模式：inspect_video -> create_run -> load_rubric_bundle(rubric_ids=[0,1,2,3,4,5,6,8]) ->
validate_run -> finalize_run。
prepare 模式：inspect_video -> create_run -> run_full_pipeline(dry_run=true) ->
refine_rubric_boundaries(execute=false) -> inspect_run_status，然后结束，绝不伪装成十项评分完成。
 execute 模式：inspect_video -> create_run -> run_full_pipeline(dry_run=false) ->
 refine_rubric_boundaries(execute=true) -> plan_live_skills -> run_adaptive_frame_agent ->
 run_rubric_bundle(rubric_ids=[0,1,2,3,4,5,6,8]) ->
inspect_run_status -> validate_run -> finalize_run。
如果 R5/R6 电表证据被遮挡、冲突、只有单帧支持或置信度过低，
使用生产组返回的 adaptive_request_template 原样调用 request_additional_evidence，
然后重新调用对应的 run_rubric_bundle，电表组重跑 [5,6]。
plan_live_skills 根据当前视频的阶段、重接线次数、测量次数和记录轮次选择 Skills；
禁止依据 video_id、文件名、SHA 或历史结果工件选择算法、ROI 或结论。
run_rubric_bundle 会把同组 Rubric 合并为一次取证生产调用；不要再调用单项生产工具。
每次只调用一个工具；相同工具的参数错误最多修正一次；create_run 返回 existing 时直接继续，禁止再次创建。
"""


def openai_tools(mode: str) -> list[dict[str, Any]]:
    """Return the legacy Chat Completions tool shape used by DeepSeek."""
    allowed = {
        "replay": {"inspect_video", "create_run", "load_rubric_bundle", "inspect_run_status", "validate_run", "finalize_run"},
        "prepare": {"inspect_video", "create_run", "run_full_pipeline", "refine_rubric_boundaries", "inspect_run_status"},
     "execute": {"inspect_video", "create_run", "run_full_pipeline", "refine_rubric_boundaries", "plan_live_skills", "run_adaptive_frame_agent", "run_rubric_bundle", "request_additional_evidence", "inspect_run_status", "validate_run", "finalize_run"},
    }[mode]
    return [
        {
            "type": "function",
            "function": {
                "name": item["name"],
                "description": item["description"],
                "parameters": item["inputSchema"],
            },
        }
        for item in TOOL_SCHEMAS
        if item["name"] in allowed
    ]


def responses_tools(mode: str) -> list[dict[str, Any]]:
    """Return Responses API function tools backed by the same MCP registry."""
    return [
        {
            "type": "function",
            "name": item["function"]["name"],
            "description": item["function"]["description"],
            "parameters": item["function"]["parameters"],
            # Existing MCP schemas intentionally leave pinned runtime fields
            # optional, so strict mode would reject otherwise valid schemas.
            "strict": False,
        }
        for item in openai_tools(mode)
    ]


def _tool_call_value(call: Any) -> tuple[str, dict[str, Any]]:
    name = str(call.function.name)
    arguments = json.loads(call.function.arguments or "{}")
    if not isinstance(arguments, dict):
        raise ToolError("model tool arguments must be an object")
    return name, arguments


def _response_tool_call_value(call: Any) -> tuple[str, dict[str, Any]]:
    name = str(call.name)
    try:
        arguments = json.loads(call.arguments or "{}")
    except json.JSONDecodeError as exc:
        raise ToolError(f"model tool arguments are invalid JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise ToolError("model tool arguments must be an object")
    return name, arguments


def _response_input_item(item: Any) -> dict[str, Any] | None:
    """Convert state-bearing SDK output into a proxy-compatible input item."""
    if isinstance(item, dict):
        value = dict(item)
    elif hasattr(item, "model_dump"):
        value = item.model_dump(exclude_none=True)
    else:
        value = {key: value for key, value in vars(item).items() if value is not None}
    if value.get("type") not in {"function_call", "message"}:
        return None
    # Some compatible gateways reject Responses output IDs (item_*) when the
    # same item is supplied as input. call_id retains the tool binding.
    value.pop("id", None)
    value.pop("status", None)
    return value


def _response_input_items(output: list[Any]) -> list[dict[str, Any]]:
    return [value for item in output if (value := _response_input_item(item)) is not None]


def _invoke(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """In-process MCP adapter; the stdio server uses the same schemas/registry."""
    return call_tool(name, arguments)


def _compact_final(document: dict[str, Any], final_path: str) -> dict[str, Any]:
    return {
        "status": document.get("status"),
        "run_id": document.get("run_id"),
        "video_id": document.get("video_id"),
        "source_video_id": document.get("source_video_id"),
        "mode": document.get("mode"),
        "result_count": len(document.get("results", [])),
        "decision_counts": document.get("decision_counts", {}),
        "final_result_path": final_path,
    }


def _pin_openai_arguments(
    name: str,
    arguments: dict[str, Any],
    *,
    run_id: str,
    video_ref: str,
    mode: str,
    config_path: Path,
) -> dict[str, Any]:
    """Bind model-selected tools to the user-requested run and mode."""
    pinned = dict(arguments)
    if name in {item["name"] for item in TOOL_SCHEMAS if "run_id" in item["inputSchema"]["properties"]}:
        pinned["run_id"] = run_id
    if name == "inspect_video":
        pinned.update(video_ref=video_ref, config_path=str(config_path))
    elif name == "create_run":
        pinned.update(
            run_id=run_id,
            video_ref=video_ref,
            mode=mode,
            config_path=str(config_path),
        )
    elif name == "run_full_pipeline":
        pinned["dry_run"] = mode == "prepare"
    elif name == "refine_rubric_boundaries":
        pinned["execute"] = mode == "execute"
    elif name == "load_rubric_bundle":
        pinned["rubric_ids"] = list(range(10))
    return pinned


def _completed_final(
    run_id: str, mode: str, transcript: list[dict[str, Any]]
) -> dict[str, Any] | None:
    state_path = RUNS_ROOT / run_id / "state.json"
    state = read_json(state_path) if state_path.is_file() else {}
    expected_status = {
        "replay": "completed",
        "prepare": "boundaries_planned",
        "execute": "completed",
    }[mode]
    inspected = bool(transcript and transcript[-1].get("tool") == "inspect_run_status")
    if state.get("status") != expected_status or (mode != "replay" and not inspected):
        return None
    if state.get("final_result"):
        final_path = str(resolve_inside(state["final_result"], AGENT_ROOT))
        return _compact_final(read_json(Path(final_path)), final_path)
    return _invoke("inspect_run_status", {"run_id": run_id})


def run_deterministic(
    run_id: str,
    video_ref: str,
    mode: str,
    config_path: Path,
) -> dict[str, Any]:
    transcript: list[dict[str, Any]] = []

    def invoke(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        value = _invoke(name, arguments)
        transcript.append({"tool": name, "arguments": arguments, "result": value})
        return value

    invoke("inspect_video", {"video_ref": video_ref, "config_path": str(config_path)})
    invoke(
        "create_run",
        {"run_id": run_id, "video_ref": video_ref, "mode": mode, "config_path": str(config_path)},
    )
    if mode == "replay":
        invoke("load_rubric_bundle", {"run_id": run_id, "rubric_ids": list(range(10))})
        invoke("validate_run", {"run_id": run_id})
        final = invoke("finalize_run", {"run_id": run_id})
        return {"scheduler": "deterministic", "transcript": transcript, "final": final}

    pipeline = invoke("run_full_pipeline", {"run_id": run_id, "dry_run": mode == "prepare"})
    boundary = invoke(
        "refine_rubric_boundaries", {"run_id": run_id, "execute": mode == "execute"}
    )
    skill_plan = invoke("plan_live_skills", {"run_id": run_id}) if mode == "execute" else None
    frame_agent = (
        invoke("run_adaptive_frame_agent", {"run_id": run_id})
        if mode == "execute"
        else None
    )
    rubric_bundle = (
        invoke("run_rubric_bundle", {"run_id": run_id, "rubric_ids": list(range(10))})
        if mode == "execute"
        else None
    )
    if mode == "execute" and rubric_bundle is not None:
        queue = list(rubric_bundle.get("producer_calls") or [])
        adaptive_rounds: dict[str, int] = {}
        while queue:
            producer = queue.pop(0)
            template = producer.get("adaptive_request_template")
            if producer.get("adaptive_evidence_recommended") is not True or not isinstance(template, dict):
                continue
            profile = str(template.get("evidence_profile") or "meter_pair")
            cycle = None
            key = f"{profile}:{cycle}"
            limit = 2
            if adaptive_rounds.get(key, 0) >= limit:
                continue
            adaptive_rounds[key] = adaptive_rounds.get(key, 0) + 1
            adaptive_result = invoke(
                "request_additional_evidence",
                {"run_id": run_id, **template},
            )
            acquired = (
                int(adaptive_result.get("selected_frame_count") or 0) > 0
            )
            if not acquired:
                continue
            rerun_ids = [5, 6]
            rerun = invoke(
                "run_rubric_bundle",
                {"run_id": run_id, "rubric_ids": rerun_ids},
            )
            queue.extend(list(rerun.get("producer_calls") or []))
    status = invoke("inspect_run_status", {"run_id": run_id})
    if mode == "execute" and not status["missing_rubrics"]:
        invoke("validate_run", {"run_id": run_id})
        final = invoke("finalize_run", {"run_id": run_id})
        return {"scheduler": "deterministic", "transcript": transcript, "final": final}
    result = {
        "status": status["status"],
        "pipeline_status": pipeline["status"],
        "video_id": status["video_id"],
        "run_dir": status["run_dir"],
        "run_report": pipeline["run_report"],
        "boundary_refinement": boundary.get("summary_path") or boundary.get("plan_path"),
        "skill_plan": skill_plan,
        "frame_agent": frame_agent,
        "rubric_bundle": rubric_bundle,
        "rubric_specific_artifacts_required": pipeline["rubric_specific_artifacts_required"],
        "completed_rubrics": status["completed_rubrics"],
        "missing_rubrics": status["missing_rubrics"],
    }
    return {"scheduler": "deterministic", "transcript": transcript, "final": result}


def run_deepseek(
    run_id: str,
    video_ref: str,
    mode: str,
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = config.get("models", {}).get("deepseek", {})
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ToolError("DEEPSEEK_API_KEY is required for --scheduler deepseek")
    base_url = os.getenv("DEEPSEEK_BASE_URL", str(settings.get("base_url") or ""))
    model = os.getenv("DEEPSEEK_MODEL", str(settings.get("model") or ""))
    if not base_url or not model:
        raise ToolError("DEEPSEEK_BASE_URL and DEEPSEEK_MODEL are required for --scheduler deepseek")
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=float(config["limits"]["model_timeout_seconds"]),
        max_retries=int(settings.get("max_retries", 2)),
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"run_id={run_id}; video_ref={video_ref}; mode={mode}; "
                f"config_path={config_path}。执行该模式的完整合法步骤。"
            ),
        },
    ]
    transcript: list[dict[str, Any]] = []
    retries: dict[str, int] = {}
    for step in range(int(config["limits"]["max_agent_steps"])):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools(mode),
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        calls = list(message.tool_calls or [])
        if not calls:
            state_path = RUNS_ROOT / run_id / "state.json"
            state = read_json(state_path) if state_path.is_file() else {}
            expected_status = {
                "replay": "completed",
                "prepare": "boundaries_planned",
                "execute": "completed",
            }[mode]
            complete = state.get("status") == expected_status
            inspected = bool(transcript and transcript[-1].get("tool") == "inspect_run_status")
            if not complete or (mode != "replay" and not inspected):
                messages.append({"role": "user", "content": "当前模式的 Pipeline 尚未完成，继续调用下一项必需工具。"})
                continue
            if state.get("final_result"):
                final_path = str(resolve_inside(state["final_result"], AGENT_ROOT))
                final = _compact_final(read_json(Path(final_path)), final_path)
            else:
                final = _invoke("inspect_run_status", {"run_id": run_id})
            return {
                "scheduler": "deepseek",
                "model": model,
                "base_url": base_url,
                "steps": step + 1,
                "transcript": transcript,
                "final_message": message.content,
                "final": final,
            }
        for call in calls:
            name, arguments = _tool_call_value(call)
            if "run_id" in arguments and arguments["run_id"] != run_id:
                value = {"error": "run_id does not match", "is_error": True}
            elif name == "create_run" and (
                arguments.get("video_ref") != video_ref or arguments.get("mode", "replay") != mode
            ):
                value = {"error": "create_run must use requested video_ref and mode", "is_error": True}
            elif name == "run_full_pipeline" and bool(arguments.get("dry_run", False)) != (mode == "prepare"):
                value = {"error": "dry_run must match prepare mode", "is_error": True}
            elif name == "refine_rubric_boundaries" and bool(arguments.get("execute", False)) != (mode == "execute"):
                value = {"error": "execute must match the run mode", "is_error": True}
            else:
                if name in {"inspect_video", "create_run"}:
                    arguments["config_path"] = str(config_path)
                try:
                    value = _invoke(name, arguments)
                    value["is_error"] = False
                except ToolError as exc:
                    retries[name] = retries.get(name, 0) + 1
                    value = {"error": str(exc), "is_error": True}
                    if retries[name] > 1:
                        raise ToolError(f"tool {name} failed twice: {exc}") from exc
            transcript.append(
                {"step": step + 1, "tool": name, "arguments": redact(arguments), "result": redact(value)}
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(redact(value), ensure_ascii=False)}
            )
    raise ToolError("agent exceeded max_agent_steps")


def run_openai(
    run_id: str,
    video_ref: str,
    mode: str,
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Use the OpenAI Responses API as the bounded tool scheduler."""
    settings = config.get("models", {}).get("openai", {})
    api_key_env = str(settings.get("api_key_env") or "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ToolError(f"{api_key_env} is required for --scheduler openai")
    base_url = os.getenv(
        str(settings.get("base_url_env") or "OPENAI_BASE_URL"),
        str(settings.get("base_url") or ""),
    )
    model = os.getenv(
        str(settings.get("model_env") or "OPENAI_MODEL"),
        str(settings.get("model") or ""),
    )
    if not base_url or not model:
        raise ToolError(f"{settings.get('base_url_env') or 'OPENAI_BASE_URL'} and {settings.get('model_env') or 'OPENAI_MODEL'} are required for --scheduler openai")
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=float(config["limits"]["model_timeout_seconds"]),
        max_retries=0,
    )
    tools = responses_tools(mode)
    allowed = {item["name"] for item in tools}
    common: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "reasoning": {"effort": str(settings.get("reasoning_effort", "low"))},
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "max_tool_calls": 1,
        "store": True,
    }
    conversation_input: list[Any] = [
        {
            "role": "user",
            "content": (
                f"run_id={run_id}; video_ref={video_ref}; mode={mode}; "
                f"config_path={config_path}。执行该模式的完整合法步骤。"
            ),
        }
    ]
    response = client.responses.create(
        **common,
        input=conversation_input,
    )
    transcript: list[dict[str, Any]] = []
    retries: dict[str, int] = {}
    response_ids: list[str] = []
    for step in range(int(config["limits"]["max_agent_steps"])):
        response_ids.append(str(response.id))
        calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
        if len(calls) > 1:
            raise ToolError("OpenAI scheduler returned multiple tool calls in one step")
        if not calls:
            final = _completed_final(run_id, mode, transcript)
            if final is not None:
                return {
                    "scheduler": "openai",
                    "model": model,
                    "base_url": base_url,
                    "steps": step + 1,
                    "response_ids": response_ids,
                    "transcript": transcript,
                    "final_message": response.output_text,
                    "final": final,
                }
            conversation_input.extend(_response_input_items(response.output))
            conversation_input.append(
                {
                    "role": "user",
                    "content": "当前模式的 Pipeline 尚未完成，继续调用下一项必需工具。",
                }
            )
            response = client.responses.create(
                **common,
                input=conversation_input,
            )
            continue

        call = calls[0]
        name, model_arguments = _response_tool_call_value(call)
        if name not in allowed:
            raise ToolError(f"OpenAI scheduler selected unregistered tool: {name}")
        arguments = _pin_openai_arguments(
            name,
            model_arguments,
            run_id=run_id,
            video_ref=video_ref,
            mode=mode,
            config_path=config_path,
        )
        try:
            value = _invoke(name, arguments)
            value["is_error"] = False
        except ToolError as exc:
            retries[name] = retries.get(name, 0) + 1
            value = {"error": str(exc), "is_error": True}
            if retries[name] > int(config["limits"].get("tool_parameter_retries", 1)):
                raise ToolError(f"tool {name} failed twice: {exc}") from exc
        safe_arguments = redact(arguments)
        safe_value = redact(value)
        transcript.append(
            {
                "step": step + 1,
                "response_id": str(response.id),
                "call_id": str(call.call_id),
                "tool": name,
                "model_arguments": redact(model_arguments),
                "arguments": safe_arguments,
                "result": safe_value,
            }
        )
        if name in {"finalize_run", "inspect_run_status"}:
            final = _completed_final(run_id, mode, transcript)
            if final is not None:
                return {
                    "scheduler": "openai",
                    "model": model,
                    "base_url": base_url,
                    "steps": step + 1,
                    "response_ids": response_ids,
                    "transcript": transcript,
                    "final_message": "local run reached its completed state",
                    "final": final,
                }
        conversation_input.extend(_response_input_items(response.output))
        conversation_input.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(safe_value, ensure_ascii=False),
            }
        )
        response = client.responses.create(
            **common,
            input=conversation_input,
        )
    raise ToolError("OpenAI agent exceeded max_agent_steps")


RUNS_ROOT = AGENT_ROOT / "runs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-ref", "--video-id", dest="video_ref", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("replay", "prepare", "execute"), default="replay")
    parser.add_argument(
        "--scheduler",
        choices=("deterministic", "deepseek", "openai"),
        default="deterministic",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    run_id = sanitize_run_id(args.run_id)
    config_path = resolve_inside(args.config, AGENT_ROOT)
    config = load_config(config_path)
    if args.scheduler == "deepseek":
        result = run_deepseek(run_id, str(args.video_ref), args.mode, config_path, config)
    elif args.scheduler == "openai":
        result = run_openai(run_id, str(args.video_ref), args.mode, config_path, config)
    else:
        result = run_deterministic(run_id, str(args.video_ref), args.mode, config_path)
    run_dir = resolve_inside(RUNS_ROOT / run_id, AGENT_ROOT)
    trace_path = run_dir / "agent_trace.json"
    write_json(trace_path, redact(result))
    final = result["final"]
    payload = {
        "status": final["status"],
        "video_id": final["video_id"],
        "final_result": final.get("final_result_path"),
        "run_report": final.get("run_report"),
        "boundary_refinement": final.get("boundary_refinement"),
        "trace": str(trace_path.resolve()),
    }
    if "decision_counts" in final:
        payload.update(final["decision_counts"])
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
