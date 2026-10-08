"""Telegram bridge: allowlist, commands, agent turns persisted as Odysseus sessions."""
import asyncio
import json

import pytest

from src import workspaces
from src import telegram_bridge as tb

OWNER_ID = 4242
CHAT_ID = 4242


class FakeApi:
    def __init__(self):
        self.calls = []
        self._next_id = 100

    async def call(self, method, **params):
        self.calls.append((method, params))
        if method == "sendMessage":
            self._next_id += 1
            return {"message_id": self._next_id}
        return True

    async def close(self):
        pass

    def texts(self):
        return [p["text"] for m, p in self.calls if m == "sendMessage"]


class Msg:
    def __init__(self, role, content, metadata=None):
        self.role, self.content, self.metadata = role, content, metadata

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class Sess:
    def __init__(self):
        self.history = []


class FakeSessionManager:
    def __init__(self):
        self.sessions = {}
        self.created = []

    def create_session(self, session_id, name, endpoint_url, model, owner=None, rag=False):
        self.sessions[session_id] = Sess()
        self.created.append((session_id, name, owner))

    def get_session(self, sid):
        return self.sessions[sid]

    def add_message(self, sid, message):
        self.sessions[sid].history.append(message)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    repo = tmp_path / "The-Manager"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("Manager-Regeln", encoding="utf-8")
    cfg = tmp_path / "projects.json"
    cfg.write_text(json.dumps({"default": "the-manager",
                               "projects": [{"name": "the-manager", "path": str(repo)}]}), encoding="utf-8")
    monkeypatch.setenv(workspaces.CONFIG_ENV, str(cfg))
    workspaces._cache["key"] = None
    import core.models
    monkeypatch.setattr(core.models, "ChatMessage", Msg)
    monkeypatch.setattr(tb.TelegramBridge, "_resolve_model",
                        lambda self, chat: ("http://localhost:11434/v1/chat/completions",
                                            chat.get("model") or "qwen3:14b", {}))
    return repo


def make_bridge(tmp_path, runner):
    return tb.TelegramBridge(FakeApi(), FakeSessionManager(), {OWNER_ID}, "admin",
                             state_path=str(tmp_path / "tg.json"), agent_runner=runner)


def update(text, user=OWNER_ID):
    return {"update_id": 1, "message": {"from": {"id": user}, "chat": {"id": CHAT_ID}, "text": text}}


async def wait_runs(bridge):
    for task in list(bridge._runs.values()):
        try:
            await task
        except asyncio.CancelledError:
            pass


async def test_unlisted_user_is_ignored(tmp_path):
    async def runner(*a):
        raise AssertionError("must not run")
    bridge = make_bridge(tmp_path, runner)
    await bridge.handle_update(update("hallo", user=999))
    assert bridge.api.calls == []


async def test_turn_runs_in_project_and_persists(tmp_path, _env, monkeypatch):
    seen = {}
    monkeypatch.setattr(tb, "TYPING_INTERVAL", 0.01)
    monkeypatch.setattr(tb, "STATUS_EDIT_INTERVAL", 0)

    async def runner(endpoint_url, model, headers, messages, owner, sid, on_event):
        seen.update(workdir=workspaces.ACTIVE_WORKDIR.get(), messages=messages, owner=owner, model=model)
        on_event({"type": "tool_start", "tool": "bash"})
        await asyncio.sleep(0.1)  # long enough for the status message to appear
        return "Erledigt."

    bridge = make_bridge(tmp_path, runner)
    await bridge.handle_update(update("Status von The Manager?"))
    await wait_runs(bridge)

    assert seen["workdir"] == str(_env.resolve()) or seen["workdir"].lower() == str(_env.resolve()).lower()
    assert seen["owner"] == "admin" and seen["model"] == "qwen3:14b"
    assert "Manager-Regeln" in seen["messages"][0]["content"]
    assert seen["messages"][-1] == {"role": "user", "content": "Status von The Manager?"}
    assert workspaces.ACTIVE_WORKDIR.get() is None
    sid = bridge.chats[str(CHAT_ID)]["session_id"]
    hist = bridge.sm.sessions[sid].history
    assert [(m.role, m.content) for m in hist] == [("user", "Status von The Manager?"), ("assistant", "Erledigt.")]
    assert bridge.api.texts()[-1] == "Erledigt."
    assert any(m == "editMessageText" and p["text"].startswith("Fertig – 1 Werkzeugaufruf")
               for m, p in bridge.api.calls)
    # state survives a restart
    assert json.loads((tmp_path / "tg.json").read_text())[str(CHAT_ID)]["session_id"] == sid


async def test_history_is_sent_on_next_turn(tmp_path):
    captured = []

    async def runner(endpoint_url, model, headers, messages, owner, sid, on_event):
        captured.append(messages)
        return f"Antwort {len(captured)}"

    bridge = make_bridge(tmp_path, runner)
    await bridge.handle_update(update("eins"))
    await wait_runs(bridge)
    await bridge.handle_update(update("zwei"))
    await wait_runs(bridge)
    roles = [(m["role"], m["content"]) for m in captured[1][1:]]
    assert roles == [("user", "eins"), ("assistant", "Antwort 1"), ("user", "zwei")]


async def test_busy_and_stop(tmp_path):
    gate = asyncio.Event()

    async def runner(*a):
        await gate.wait()
        return "spät"

    bridge = make_bridge(tmp_path, runner)
    await bridge.handle_update(update("lang"))
    await asyncio.sleep(0)
    await bridge.handle_update(update("noch was"))
    assert "arbeite noch" in bridge.api.texts()[-1]
    await bridge.handle_update(update("/stop"))
    await wait_runs(bridge)
    assert "Abgebrochen." in bridge.api.texts()
    assert "spät" not in bridge.api.texts()


async def test_commands(tmp_path):
    async def runner(*a):
        return "ok"
    bridge = make_bridge(tmp_path, runner)
    await bridge.handle_update(update("/projekte"))
    assert "the-manager" in bridge.api.texts()[-1]
    await bridge.handle_update(update("/projekt gibts-nicht"))
    assert "Unbekanntes Projekt" in bridge.api.texts()[-1]
    await bridge.handle_update(update("/modell llama3.1:8b"))
    assert bridge.api.texts()[-1] == "Modell: llama3.1:8b"
    await bridge.handle_update(update("/status"))
    assert "Modell: llama3.1:8b" in bridge.api.texts()[-1]


def test_split_text_respects_limit():
    text = ("Absatz " * 300 + "\n\n") * 5
    chunks = tb.split_text(text, limit=1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(chunks).replace(" ", "").replace("\n", "") == text.replace(" ", "").replace("\n", "")


def test_from_env_requires_allowlist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "")
    assert tb.TelegramBridge.from_env(FakeSessionManager()) is None
    assert tb.parse_allowed("1, 2,x,-3") == {1, 2, -3}
