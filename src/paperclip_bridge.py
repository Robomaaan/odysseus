"""
paperclip_bridge.py

Lets Paperclip (github.com/paperclipai/paperclip) employ Odysseus as an agent.

In Paperclip, create an agent with adapter "http":
    url:     http://<odysseus-host>:7000/api/paperclip/heartbeat
    headers: {"Authorization": "Bearer ody_..."}      (Odysseus API token, scope chat)
    payloadTemplate (optional):
             {"project": "the-manager", "model": "qwen3:32b", "apiKeyEnv": "PAPERCLIP_KEY_CTO"}

Each heartbeat is acknowledged with 202 and runs the agent in the background
with the Paperclip skill as instructions. The agent talks back to Paperclip's
REST API itself; the PAPERCLIP_* variables are injected into its bash/python
tools for that run only.

Environment:
    PAPERCLIP_API_URL        Paperclip base URL, e.g. http://127.0.0.1:3100
    PAPERCLIP_API_KEY        default agent API key (override per agent via apiKeyEnv)
    PAPERCLIP_SKILL_PATH     path to skills/paperclip/SKILL.md (default: looked up
                             in the "paperclip" project from ODYSSEUS_PROJECTS)
    PAPERCLIP_NOTIFY_TELEGRAM  "1" → send each run's result to Telegram
"""

import asyncio
import json
import logging
import os
import re
import uuid
from typing import Dict, Optional

from src.headless_agent import run_agent, run_scope
from src.workspaces import get_project, project_context

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 6
MAX_SKILL_CHARS = 30_000
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")

WAKE_FIELDS = {
    "taskId": "PAPERCLIP_TASK_ID", "issueId": "PAPERCLIP_TASK_ID",
    "wakeReason": "PAPERCLIP_WAKE_REASON",
    "wakeCommentId": "PAPERCLIP_WAKE_COMMENT_ID", "commentId": "PAPERCLIP_WAKE_COMMENT_ID",
    "approvalId": "PAPERCLIP_APPROVAL_ID", "approvalStatus": "PAPERCLIP_APPROVAL_STATUS",
}


def wake_env(agent_id: str, run_id: str, context: Dict, api_url: str, api_key: str) -> Dict[str, str]:
    """Mirror the env vars Paperclip's local adapters inject (see claude-local)."""
    env = {"PAPERCLIP_AGENT_ID": agent_id, "PAPERCLIP_RUN_ID": run_id,
           "PAPERCLIP_API_URL": api_url, "PAPERCLIP_API_KEY": api_key}
    if context.get("companyId"):
        env["PAPERCLIP_COMPANY_ID"] = str(context["companyId"])
    for key, var in WAKE_FIELDS.items():
        value = context.get(key)
        if isinstance(value, str) and value.strip() and var not in env:
            env[var] = value.strip()
    issue_ids = [i for i in context.get("issueIds") or [] if isinstance(i, str) and i.strip()]
    if issue_ids:
        env["PAPERCLIP_LINKED_ISSUE_IDS"] = ",".join(issue_ids)
    if context.get("paperclipWake") is not None:
        env["PAPERCLIP_WAKE_PAYLOAD_JSON"] = json.dumps(context["paperclipWake"])
    return env


def heartbeat_prompt(env: Dict[str, str]) -> str:
    lines = ["Paperclip-Heartbeat. Führe die Heartbeat-Prozedur aus dem Paperclip-Skill aus.",
             f"Paperclip API: {env['PAPERCLIP_API_URL']}/api (Bearer-Token in $PAPERCLIP_API_KEY).",
             f"Bei jeder ändernden Anfrage Header X-Paperclip-Run-Id: {env['PAPERCLIP_RUN_ID']} setzen."]
    for var in ("PAPERCLIP_AGENT_ID", "PAPERCLIP_TASK_ID", "PAPERCLIP_WAKE_REASON",
                "PAPERCLIP_WAKE_COMMENT_ID", "PAPERCLIP_APPROVAL_ID", "PAPERCLIP_APPROVAL_STATUS",
                "PAPERCLIP_LINKED_ISSUE_IDS"):
        if env.get(var):
            lines.append(f"{var}={env[var]}")
    if env.get("PAPERCLIP_WAKE_PAYLOAD_JSON"):
        lines.append("PAPERCLIP_WAKE_PAYLOAD_JSON (zuerst auswerten):\n" + env["PAPERCLIP_WAKE_PAYLOAD_JSON"][:6000])
    lines.append("Antworte am Ende mit 2–4 Sätzen: was du getan hast und was offen ist.")
    return "\n".join(lines)


def load_skill() -> str:
    path = os.environ.get("PAPERCLIP_SKILL_PATH", "").strip()
    if not path:
        project = get_project("paperclip")
        if project:
            path = os.path.join(project.path, "skills", "paperclip", "SKILL.md")
    if not path or not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read(MAX_SKILL_CHARS)


class PaperclipBridge:
    def __init__(self, session_manager, state_path: str, agent_runner=run_agent, notify=None):
        self.sm = session_manager
        self.state_path = state_path
        self.agent_runner = agent_runner
        self.notify = notify  # async callable(text) or None
        self._running: Dict[str, asyncio.Task] = {}
        try:
            with open(state_path, encoding="utf-8") as f:
                self.sessions: Dict[str, str] = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.sessions = {}

    def is_busy(self, agent_id: str) -> bool:
        task = self._running.get(agent_id)
        return bool(task and not task.done())

    def accept(self, body: Dict, owner: str) -> Dict:
        """Validate a heartbeat and start its run. Raises ValueError/RuntimeError."""
        agent_id = str(body.get("agentId") or "").strip()
        run_id = str(body.get("runId") or "").strip()
        context = body.get("context") or {}
        if not agent_id or not run_id or not isinstance(context, dict):
            raise ValueError("agentId, runId and context are required")
        if self.is_busy(agent_id):
            raise RuntimeError(f"agent {agent_id} is still running")
        api_url = os.environ.get("PAPERCLIP_API_URL", "").strip().rstrip("/")
        key_env = str(body.get("apiKeyEnv") or "PAPERCLIP_API_KEY")
        if not _ENV_NAME.match(key_env) or not key_env.startswith("PAPERCLIP_"):
            raise ValueError("apiKeyEnv must name a PAPERCLIP_* variable")
        api_key = os.environ.get(key_env, "").strip()
        if not api_url or not api_key:
            raise ValueError(f"PAPERCLIP_API_URL and {key_env} must be set on the Odysseus host")
        project = get_project(body.get("project")) if body.get("project") else None
        if body.get("project") and not project:
            raise ValueError(f"unknown project {body.get('project')!r}")
        env = wake_env(agent_id, run_id, context, api_url, api_key)
        key = f"{agent_id}:{env.get('PAPERCLIP_TASK_ID', '-')}"
        sid = self.sessions.get(key) or str(uuid.uuid4())
        self._running[agent_id] = asyncio.create_task(
            self._run(key, sid, env, project, body.get("model"), owner))
        return {"accepted": True, "sessionId": sid}

    def _ensure_session(self, key: str, sid: str, endpoint_url: str, model: str, owner: str, env: Dict):
        try:
            self.sm.get_session(sid)
            return
        except Exception:
            pass
        task = env.get("PAPERCLIP_TASK_ID", "allgemein")
        self.sm.create_session(session_id=sid, name=f"Paperclip · {env['PAPERCLIP_AGENT_ID'][:8]} · {task[:24]}",
                               endpoint_url=endpoint_url, model=model, owner=owner)
        self.sessions[key] = sid
        from core.atomic_io import atomic_write_json
        atomic_write_json(self.state_path, self.sessions, indent=2)

    async def _run(self, key: str, sid: str, env: Dict, project, model_override: Optional[str], owner: str):
        from core.models import ChatMessage
        from src.endpoint_resolver import resolve_endpoint
        endpoint_url, model, headers = resolve_endpoint("default", owner=owner)
        model = model_override or model
        if not endpoint_url or not model:
            logger.error("Paperclip heartbeat: no default model configured in Odysseus")
            return
        self._ensure_session(key, sid, endpoint_url, model, owner, env)
        skill = load_skill()
        system = "\n\n".join(filter(None, [
            "Du arbeitest als Agent einer Paperclip-Firma." + (" Paperclip-Skill:" if skill else
                " Der Paperclip-Skill wurde nicht gefunden; nutze GET /api/agents/me und die Issue-API."),
            skill,
            project_context(project) if project else "",
        ]))
        prompt = heartbeat_prompt(env)
        history = [m.to_dict() for m in self.sm.get_session(sid).history[-HISTORY_LIMIT:]
                   if m.role in ("user", "assistant")]
        messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": prompt}]
        self.sm.add_message(sid, ChatMessage(role="user", content=prompt, metadata={"source": "paperclip"}))
        try:
            with run_scope(project.path if project else None, env):
                reply = await self.agent_runner(endpoint_url, model, headers or {}, messages, owner, sid,
                                                lambda ev: None)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Paperclip heartbeat run failed")
            reply = f"Fehler im Heartbeat-Lauf: {type(e).__name__}: {e}"
        self.sm.add_message(sid, ChatMessage(role="assistant", content=reply, metadata={"source": "paperclip"}))
        if self.notify:
            try:
                await self.notify(f"Paperclip {env['PAPERCLIP_AGENT_ID'][:8]}"
                                  f" · {env.get('PAPERCLIP_TASK_ID', '-')}\n{reply}")
            except Exception as e:
                logger.warning(f"Paperclip notify failed: {e}")


def setup_paperclip_routes(bridge: PaperclipBridge):
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import JSONResponse

    router = APIRouter(prefix="/api/paperclip", tags=["paperclip"])

    @router.post("/heartbeat")
    async def heartbeat(request: Request):
        if not getattr(request.state, "api_token", False):
            raise HTTPException(403, "This endpoint requires an Odysseus API token")
        if "chat" not in set(getattr(request.state, "api_token_scopes", []) or []):
            raise HTTPException(403, "API token is not scoped for chat")
        owner = getattr(request.state, "api_token_owner", None)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "JSON body required")
        if not isinstance(body, dict):
            raise HTTPException(400, "JSON object required")
        try:
            result = bridge.accept(body, owner)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return JSONResponse(result, status_code=202)

    return router
