"""
telegram_bridge.py

Chat with the Odysseus agent from Telegram.

Long-polls the Bot API (no inbound port needed), runs each message through the
full agent loop with tools and MCP, and stores the conversation as a normal
Odysseus session so it can be continued in the web UI.

Environment:
    TELEGRAM_BOT_TOKEN       bot token from @BotFather; bridge is off without it
    TELEGRAM_ALLOWED_USERS   comma-separated numeric Telegram user ids (required;
                             the bridge refuses to start with an empty list)
    TELEGRAM_OWNER           Odysseus user the runs execute as (default:
                             ODYSSEUS_ADMIN_USER or "admin"); must be admin for
                             shell and MCP tools
    TELEGRAM_HISTORY         past messages sent to the model per turn (default 20)

Commands: /hilfe /projekte /projekt <name> /neu /stop /status /modell [name]
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Awaitable, Callable, Dict, List, Optional

from src.headless_agent import run_agent, run_scope
from src.workspaces import get_project, list_projects, project_context

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
MAX_TEXT = 4000           # Telegram hard limit is 4096
POLL_TIMEOUT = 50
TYPING_INTERVAL = 4.5
STATUS_EDIT_INTERVAL = 3.0

HELP_TEXT = (
    "Odysseus-Hauptagent\n\n"
    "Schreib einfach, was zu tun ist. Befehle:\n"
    "/projekte – verfügbare Projekte\n"
    "/projekt <name> – Projekt wechseln (startet neue Session)\n"
    "/neu – neue Session im aktuellen Projekt\n"
    "/stop – laufende Antwort abbrechen\n"
    "/status – Projekt, Modell, Session\n"
    "/modell [name] – Modell anzeigen oder für diesen Chat setzen"
)

TELEGRAM_SYSTEM = (
    "Du sprichst über Telegram mit deinem Auftraggeber. Antworte knapp und in "
    "reinem Text ohne Markdown-Tabellen; lange Ausgaben fasst du zusammen und "
    "nennst den Pfad der Datei. Frag nach, bevor du etwas Destruktives tust "
    "(löschen, force-push, Migrationen, fremde Daten überschreiben)."
)


def split_text(text: str, limit: int = MAX_TEXT) -> List[str]:
    """Split on paragraph/line boundaries so each chunk fits one Telegram message."""
    text = text.strip() or "(keine Antwort)"
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    chunks.append(text)
    return chunks


def parse_allowed(raw: str) -> set:
    return {int(x) for x in (p.strip() for p in (raw or "").split(",")) if x.lstrip("-").isdigit()}


class TelegramApi:
    """Minimal async Bot API client."""

    def __init__(self, token: str, client=None):
        import httpx
        self._base = f"{API_BASE}/bot{token}"
        self._client = client or httpx.AsyncClient(timeout=POLL_TIMEOUT + 10)

    async def call(self, method: str, **params):
        r = await self._client.post(f"{self._base}/{method}", json=params)
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram {method}: {data.get('description', r.status_code)}")
        return data.get("result")

    async def close(self):
        await self._client.aclose()


# (endpoint_url, model, headers, messages, owner, session_id, on_event) -> final text
AgentRunner = Callable[..., Awaitable[str]]


class TelegramBridge:
    def __init__(self, api: TelegramApi, session_manager, allowed_users: set, owner: str,
                 state_path: str, history_limit: int = 20, agent_runner: AgentRunner = run_agent):
        self.api = api
        self.sm = session_manager
        self.allowed = allowed_users
        self.owner = owner
        self.state_path = state_path
        self.history_limit = history_limit
        self.agent_runner = agent_runner
        self.chats: Dict[str, Dict] = self._load_state()
        self._runs: Dict[int, asyncio.Task] = {}
        self._offset = 0
        self._stopping = False

    @classmethod
    def from_env(cls, session_manager) -> Optional["TelegramBridge"]:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            return None
        allowed = parse_allowed(os.environ.get("TELEGRAM_ALLOWED_USERS", ""))
        if not allowed:
            logger.error("Telegram bridge disabled: TELEGRAM_ALLOWED_USERS is empty")
            return None
        from src.constants import DATA_DIR
        owner = (os.environ.get("TELEGRAM_OWNER") or os.environ.get("ODYSSEUS_ADMIN_USER") or "admin").strip()
        return cls(
            TelegramApi(token), session_manager, allowed, owner,
            state_path=os.path.join(DATA_DIR, "telegram_state.json"),
            history_limit=int(os.environ.get("TELEGRAM_HISTORY", "20") or 20),
        )

    # ── state ────────────────────────────────────────────────────────────
    def _load_state(self) -> Dict[str, Dict]:
        try:
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self):
        from core.atomic_io import atomic_write_json
        atomic_write_json(self.state_path, self.chats, indent=2)

    def _chat(self, chat_id: int) -> Dict:
        return self.chats.setdefault(str(chat_id), {"session_id": None, "project": None, "model": None})

    # ── polling ──────────────────────────────────────────────────────────
    async def run(self):
        logger.info(f"Telegram bridge started for {len(self.allowed)} user(s), owner={self.owner}")
        backoff = 1.0
        while not self._stopping:
            try:
                updates = await self.api.call(
                    "getUpdates", offset=self._offset, timeout=POLL_TIMEOUT, allowed_updates=["message"])
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Telegram getUpdates failed: {e}; retry in {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            for upd in updates or []:
                self._offset = max(self._offset, upd["update_id"] + 1)
                try:
                    await self.handle_update(upd)
                except Exception:
                    logger.exception("Telegram update handling failed")

    async def stop(self):
        self._stopping = True
        for task in list(self._runs.values()):
            task.cancel()
        await self.api.close()

    async def broadcast(self, text: str):
        """Send to every allowed user (private chats share the user id)."""
        for user_id in self.allowed:
            try:
                await self.send(user_id, text)
            except Exception as e:
                logger.warning(f"Telegram broadcast to {user_id} failed: {e}")

    async def send(self, chat_id: int, text: str):
        for chunk in split_text(text):
            await self.api.call("sendMessage", chat_id=chat_id, text=chunk)

    # ── dispatch ─────────────────────────────────────────────────────────
    async def handle_update(self, upd: Dict):
        msg = upd.get("message") or {}
        user_id = (msg.get("from") or {}).get("id")
        chat_id = (msg.get("chat") or {}).get("id")
        if chat_id is None:
            return
        if user_id not in self.allowed:
            logger.warning(f"Telegram: ignored message from unlisted user id {user_id}")
            return
        text = (msg.get("text") or msg.get("caption") or "").strip()
        if not text:
            await self.send(chat_id, "Ich verarbeite bisher nur Text.")
            return
        if text.startswith("/"):
            await self.handle_command(chat_id, text)
            return
        if chat_id in self._runs and not self._runs[chat_id].done():
            await self.send(chat_id, "Ich arbeite noch an der letzten Nachricht. /stop bricht ab.")
            return
        self._runs[chat_id] = asyncio.create_task(self.run_turn(chat_id, text))

    async def handle_command(self, chat_id: int, text: str):
        cmd, _, arg = text.partition(" ")
        cmd = cmd.split("@", 1)[0].lower()
        arg = arg.strip()
        chat = self._chat(chat_id)
        if cmd in ("/start", "/hilfe", "/help"):
            await self.send(chat_id, HELP_TEXT)
        elif cmd == "/projekte":
            projects = list_projects()
            active = (get_project(chat["project"]) or get_project(None))
            lines = [f"{'▶' if active and p.name == active.name else '•'} {p.name} – {p.description or p.path}"
                     for p in projects]
            await self.send(chat_id, "\n".join(lines) or "Keine Projekte konfiguriert (ODYSSEUS_PROJECTS).")
        elif cmd == "/projekt":
            project = get_project(arg) if arg else None
            if not project:
                await self.send(chat_id, f"Unbekanntes Projekt {arg!r}. /projekte zeigt die Liste.")
                return
            chat.update(project=project.name, session_id=None)
            self._save_state()
            await self.send(chat_id, f"Projekt: {project.name}\n{project.path}\nNeue Session beginnt mit der nächsten Nachricht.")
        elif cmd == "/neu":
            chat["session_id"] = None
            self._save_state()
            await self.send(chat_id, "Neue Session beginnt mit der nächsten Nachricht.")
        elif cmd == "/stop":
            task = self._runs.get(chat_id)
            if task and not task.done():
                task.cancel()
                await self.send(chat_id, "Abgebrochen.")
            else:
                await self.send(chat_id, "Es läuft gerade nichts.")
        elif cmd == "/status":
            project = get_project(chat["project"])
            endpoint_url, model, _ = self._resolve_model(chat)
            running = chat_id in self._runs and not self._runs[chat_id].done()
            await self.send(chat_id, "\n".join([
                f"Projekt: {project.name if project else '–'}",
                f"Modell: {model or '– (in Odysseus Settings ein Default-Modell setzen)'}",
                f"Endpoint: {endpoint_url or '–'}",
                f"Session: {chat['session_id'] or 'neu'}",
                f"Läuft: {'ja' if running else 'nein'}",
            ]))
        elif cmd == "/modell":
            if arg:
                chat["model"] = None if arg in ("-", "default", "standard") else arg
                self._save_state()
            _, model, _ = self._resolve_model(chat)
            await self.send(chat_id, f"Modell: {model or '–'}")
        else:
            await self.send(chat_id, "Unbekannter Befehl. /hilfe zeigt alle.")

    # ── agent turn ───────────────────────────────────────────────────────
    def _resolve_model(self, chat: Dict):
        from src.endpoint_resolver import resolve_endpoint
        endpoint_url, model, headers = resolve_endpoint("default", owner=self.owner)
        return endpoint_url, chat.get("model") or model, headers or {}

    def _ensure_session(self, chat: Dict, chat_id: int, endpoint_url: str, model: str, project) -> str:
        sid = chat.get("session_id")
        if sid:
            try:
                self.sm.get_session(sid)
                return sid
            except Exception:
                pass
        sid = str(uuid.uuid4())
        label = project.name if project else "Allgemein"
        self.sm.create_session(session_id=sid, name=f"Telegram · {label}",
                               endpoint_url=endpoint_url, model=model, owner=self.owner)
        chat["session_id"] = sid
        self._save_state()
        return sid

    async def run_turn(self, chat_id: int, text: str):
        from core.models import ChatMessage
        chat = self._chat(chat_id)
        project = get_project(chat.get("project"))
        endpoint_url, model, headers = self._resolve_model(chat)
        if not endpoint_url or not model:
            await self.send(chat_id, "Kein Modell konfiguriert. In Odysseus unter Settings ein Default-Modell setzen.")
            return
        sid = self._ensure_session(chat, chat_id, endpoint_url, model, project)
        history = [m.to_dict() for m in self.sm.get_session(sid).history[-self.history_limit:]
                   if m.role in ("user", "assistant")]
        system = TELEGRAM_SYSTEM + ("\n\n" + project_context(project) if project else "")
        messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": text}]
        self.sm.add_message(sid, ChatMessage(role="user", content=text, metadata={"source": "telegram"}))

        status = {"id": None, "tools": [], "last_edit": 0.0}

        def on_event(ev: Dict):
            if ev.get("type") == "tool_start":
                status["tools"].append(str(ev.get("tool") or "?"))

        async def keep_typing():
            while True:
                try:
                    await self.api.call("sendChatAction", chat_id=chat_id, action="typing")
                    await self._update_status(chat_id, status)
                except Exception as e:
                    logger.debug(f"Telegram typing/status failed: {e}")
                await asyncio.sleep(TYPING_INTERVAL)

        typing = asyncio.create_task(keep_typing())
        try:
            with run_scope(project.path if project else None):
                reply = await self.agent_runner(endpoint_url, model, headers, messages, self.owner, sid, on_event)
        except asyncio.CancelledError:
            self.sm.add_message(sid, ChatMessage(role="assistant", content="(abgebrochen)", metadata={"source": "telegram"}))
            raise
        except Exception as e:
            logger.exception("Telegram agent run failed")
            await self.send(chat_id, f"Fehler im Agentenlauf: {type(e).__name__}: {e}")
            return
        finally:
            typing.cancel()
        self.sm.add_message(sid, ChatMessage(role="assistant", content=reply, metadata={"source": "telegram"}))
        if status["id"]:
            await self._update_status(chat_id, status, final=True)
        await self.send(chat_id, reply)

    async def _update_status(self, chat_id: int, status: Dict, final: bool = False):
        """One editable message listing the tools used so far."""
        tools = status["tools"]
        if not tools:
            return
        now = time.monotonic()
        if not final and now - status["last_edit"] < STATUS_EDIT_INTERVAL:
            return
        recent = ", ".join(tools[-6:])
        text = f"{'Fertig' if final else 'Arbeite'} – {len(tools)} Werkzeugaufruf(e): {recent}"
        if status.get("text") == text:
            return
        if status["id"] is None:
            res = await self.api.call("sendMessage", chat_id=chat_id, text=text)
            status["id"] = res["message_id"]
        else:
            await self.api.call("editMessageText", chat_id=chat_id, message_id=status["id"], text=text)
        status.update(text=text, last_edit=now)
