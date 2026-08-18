#!/usr/bin/env python3
"""Dependency-free MCP 2025-03-26 stdio server for the project tools."""

from __future__ import annotations

import json
import sys
from typing import Any, BinaryIO

try:
    from .toolkit import TOOL_SCHEMAS, ToolError, call_tool
except ImportError:
    from toolkit import TOOL_SCHEMAS, ToolError, call_tool  # type: ignore


SERVER_INFO = {"name": "resistance-video-agent", "version": "1.1.0"}
PROTOCOL_VERSION = "2025-03-26"


def read_message(stream: BinaryIO) -> dict[str, Any] | None:
    first = stream.readline()
    if not first:
        return None
    if first.lstrip().startswith(b"{"):
        return json.loads(first.decode("utf-8"))
    headers: dict[str, str] = {}
    line = first
    while line not in {b"\r\n", b"\n", b""}:
        name, separator, value = line.decode("ascii").partition(":")
        if not separator:
            raise ValueError("invalid MCP header")
        headers[name.strip().lower()] = value.strip()
        line = stream.readline()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        raise ValueError("missing Content-Length")
    return json.loads(stream.read(length).decode("utf-8"))


def write_message(stream: BinaryIO, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    stream.write(payload + b"\n")
    stream.flush()


def response(
    request_id: Any,
    result: Any = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    payload["error" if error else "result"] = error if error else result
    return payload


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    if request_id is None:
        return None
    if method == "initialize":
        return response(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            },
        )
    if method == "ping":
        return response(request_id, {})
    if method == "tools/list":
        return response(request_id, {"tools": TOOL_SCHEMAS})
    if method == "tools/call":
        params = request.get("params") or {}
        try:
            value = call_tool(str(params.get("name") or ""), params.get("arguments"))
            return response(
                request_id,
                {
                    "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
                    "structuredContent": value,
                    "isError": False,
                },
            )
        except (ToolError, TypeError, ValueError) as exc:
            value = {"error": str(exc), "error_type": type(exc).__name__}
            return response(
                request_id,
                {
                    "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
                    "structuredContent": value,
                    "isError": True,
                },
            )
    return response(request_id, error={"code": -32601, "message": f"method not found: {method}"})


def main() -> int:
    while True:
        try:
            request = read_message(sys.stdin.buffer)
            if request is None:
                return 0
            answer = handle(request)
            if answer is not None:
                write_message(sys.stdout.buffer, answer)
        except Exception as exc:
            write_message(sys.stdout.buffer, response(None, error={"code": -32603, "message": str(exc)}))


if __name__ == "__main__":
    raise SystemExit(main())
