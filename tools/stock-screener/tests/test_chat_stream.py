import json
import os
import stat

import pytest

from moonshot import control_server
from moonshot.control_server import ChatStore


def fake_claude(tmp_path, monkeypatch, body: str) -> None:
    script = tmp_path / "claude"
    script.write_text("#!/usr/bin/env python3\nimport json, sys, time\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


def delta(text):
    return {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}}


def emit(*events):
    return "".join(f"print(json.dumps({e!r}), flush=True)\n" for e in events)


def test_streams_text_then_done_and_remembers_the_session(tmp_path, monkeypatch):
    tool = {"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "tool_use", "name": "WebSearch"}}}
    result = {"type": "result", "is_error": False, "result": "final answer", "session_id": "abc"}
    fake_claude(tmp_path, monkeypatch, emit(delta("preamble"), tool, delta("final "), delta("answer"), result))
    chat = ChatStore()
    events = list(chat.stream("tick", "why?", lambda t: "context"))
    kinds = [e["type"] for e in events]
    assert kinds == ["delta", "reset", "status", "delta", "delta", "done"]
    assert events[2]["text"] == "Searching the web…"
    assert events[-1]["answer"] == "final answer"
    assert chat.sessions["TICK"] == "abc"


def test_cli_error_result_is_reported_and_no_session_is_stored(tmp_path, monkeypatch):
    fake_claude(tmp_path, monkeypatch, emit({"type": "result", "is_error": True, "result": "rate limited", "session_id": "x"}))
    chat = ChatStore()
    events = list(chat.stream("tick", "why?", lambda t: "context"))
    assert events == [{"type": "error", "error": "rate limited"}]
    assert "TICK" not in chat.sessions


def test_a_silent_cli_is_stopped_after_the_idle_timeout(tmp_path, monkeypatch):
    fake_claude(tmp_path, monkeypatch, "time.sleep(30)\n")
    monkeypatch.setattr(control_server, "CHAT_IDLE_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(control_server, "CHAT_HEARTBEAT_SECONDS", 0.3)
    events = list(ChatStore().stream("tick", "why?", lambda t: "context"))
    assert events[-1]["type"] == "error" and "no output" in events[-1]["error"]
    assert any(e["type"] == "ping" for e in events)


def test_a_slow_but_steady_cli_is_not_cut_off(tmp_path, monkeypatch):
    body = "for i in range(4):\n    print(json.dumps(" + repr(delta("x")) + "), flush=True)\n    time.sleep(0.5)\n" + emit(
        {"type": "result", "is_error": False, "result": "xxxx", "session_id": "s"})
    fake_claude(tmp_path, monkeypatch, body)
    monkeypatch.setattr(control_server, "CHAT_IDLE_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(control_server, "CHAT_HEARTBEAT_SECONDS", 0.2)
    events = list(ChatStore().stream("tick", "why?", lambda t: "context"))
    assert events[-1]["type"] == "done"
