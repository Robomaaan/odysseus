"""Paperclip HTTP-adapter heartbeats run the agent with PAPERCLIP_* env."""
import asyncio
import json
import sys

import pytest

from src import paperclip_bridge as pb
from src import workspaces
from tests.test_telegram_bridge import FakeSessionManager, Msg


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    clip = tmp_path / "paperclip"
    (clip / "skills" / "paperclip").mkdir(parents=True)
    (clip / "skills" / "paperclip" / "SKILL.md").write_text("# Paperclip Skill\nHeartbeat-Prozedur", encoding="utf-8")
    mgr = tmp_path / "The-Manager"
    mgr.mkdir()
    cfg = tmp_path / "projects.json"
    cfg.write_text(json.dumps({"projects": [{"name": "paperclip", "path": str(clip)},
                                            {"name": "the-manager", "path": str(mgr)}]}), encoding="utf-8")
    monkeypatch.setenv(workspaces.CONFIG_ENV, str(cfg))
    workspaces._cache["key"] = None
    monkeypatch.setenv("PAPERCLIP_API_URL", "http://127.0.0.1:3100/")
    monkeypatch.setenv("PAPERCLIP_API_KEY", "pc-default")
    monkeypatch.setenv("PAPERCLIP_KEY_CTO", "pc-cto")
    monkeypatch.delenv("PAPERCLIP_SKILL_PATH", raising=False)
    import core.models
    import src.endpoint_resolver as er
    monkeypatch.setattr(core.models, "ChatMessage", Msg)
    monkeypatch.setattr(er, "resolve_endpoint", lambda prefix, owner=None: ("http://llm/v1/chat/completions", "qwen3:32b", {}))
    return mgr


def body(**extra):
    return {"agentId": "agent-cto-123456", "runId": "run-1",
            "context": {"taskId": "ISS-7", "wakeReason": "comment", "commentId": "c-9",
                        "issueIds": ["ISS-7", "ISS-8"], "paperclipWake": {"comments": [{"body": "los"}]}},
            **extra}


async def test_heartbeat_runs_with_env_skill_and_project(tmp_path, _env):
    seen = {}

    async def runner(endpoint_url, model, headers, messages, owner, sid, on_event):
        from src.workspaces import ACTIVE_WORKDIR, EXTRA_ENV
        seen.update(env=dict(EXTRA_ENV.get()), workdir=ACTIVE_WORKDIR.get(), messages=messages,
                    owner=owner, model=model)
        return "ISS-7 kommentiert, nichts offen."

    notified = []

    async def notify(text):
        notified.append(text)

    bridge = pb.PaperclipBridge(FakeSessionManager(), str(tmp_path / "pc.json"), agent_runner=runner, notify=notify)
    res = bridge.accept(body(project="the-manager", apiKeyEnv="PAPERCLIP_KEY_CTO", model="llama3.3:70b"), "admin")
    await bridge._running["agent-cto-123456"]

    env = seen["env"]
    assert env["PAPERCLIP_API_KEY"] == "pc-cto"
    assert env["PAPERCLIP_API_URL"] == "http://127.0.0.1:3100"
    assert env["PAPERCLIP_TASK_ID"] == "ISS-7" and env["PAPERCLIP_WAKE_COMMENT_ID"] == "c-9"
    assert env["PAPERCLIP_LINKED_ISSUE_IDS"] == "ISS-7,ISS-8"
    assert json.loads(env["PAPERCLIP_WAKE_PAYLOAD_JSON"]) == {"comments": [{"body": "los"}]}
    assert seen["workdir"].lower() == str(_env.resolve()).lower()
    assert seen["model"] == "llama3.3:70b" and seen["owner"] == "admin"
    assert "Heartbeat-Prozedur" in seen["messages"][0]["content"]
    assert "X-Paperclip-Run-Id: run-1" in seen["messages"][-1]["content"]
    assert "pc-cto" not in seen["messages"][-1]["content"]  # key only via env
    hist = bridge.sm.sessions[res["sessionId"]].history
    assert hist[-1].content == "ISS-7 kommentiert, nichts offen."
    assert notified and "ISS-7" in notified[0]
    from src.workspaces import EXTRA_ENV
    assert EXTRA_ENV.get() == {}


async def test_same_task_reuses_session_and_busy_is_rejected(tmp_path):
    gate = asyncio.Event()

    async def runner(*a):
        await gate.wait()
        return "ok"

    bridge = pb.PaperclipBridge(FakeSessionManager(), str(tmp_path / "pc.json"), agent_runner=runner)
    first = bridge.accept(body(), "admin")
    with pytest.raises(RuntimeError, match="still running"):
        bridge.accept(body(runId="run-2"), "admin")
    gate.set()
    await bridge._running["agent-cto-123456"]
    second = bridge.accept(body(runId="run-2"), "admin")
    await bridge._running["agent-cto-123456"]
    assert first["sessionId"] == second["sessionId"]


@pytest.mark.parametrize("bad, msg", [
    ({"agentId": "", "runId": "r", "context": {}}, "required"),
    (body(apiKeyEnv="PATH"), "PAPERCLIP_"),
    (body(apiKeyEnv="PAPERCLIP_MISSING"), "must be set"),
    (body(project="gibts-nicht"), "unknown project"),
])
def test_invalid_heartbeats(tmp_path, bad, msg):
    bridge = pb.PaperclipBridge(FakeSessionManager(), str(tmp_path / "pc.json"))
    with pytest.raises(ValueError, match=msg):
        bridge.accept(bad, "admin")


def test_route_requires_api_token_and_maps_errors(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.middleware.base import BaseHTTPMiddleware

    async def runner(*a):
        return "ok"

    bridge = pb.PaperclipBridge(FakeSessionManager(), str(tmp_path / "pc.json"), agent_runner=runner)
    app = FastAPI()
    app.include_router(pb.setup_paperclip_routes(bridge))

    class FakeTokenAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.headers.get("authorization") == "Bearer ody_test":
                request.state.api_token = True
                request.state.api_token_scopes = ["chat"]
                request.state.api_token_owner = "admin"
            return await call_next(request)

    app.add_middleware(FakeTokenAuth)
    with TestClient(app) as client:
        assert client.post("/api/paperclip/heartbeat", json=body()).status_code == 403
        auth = {"Authorization": "Bearer ody_test"}
        assert client.post("/api/paperclip/heartbeat", json={"x": 1}, headers=auth).status_code == 400
        r = client.post("/api/paperclip/heartbeat", json=body(), headers=auth)
        assert r.status_code == 202 and r.json()["accepted"] is True
