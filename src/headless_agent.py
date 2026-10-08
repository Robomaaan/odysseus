"""
headless_agent.py

Run the Odysseus agent loop without a browser (Telegram, Paperclip heartbeats)
and return the final assistant text.
"""

import json
from typing import Callable, Dict, List

from src.workspaces import ACTIVE_WORKDIR, EXTRA_ENV


async def run_agent(endpoint_url: str, model: str, headers: Dict, messages: List[Dict],
                    owner: str, session_id: str, on_event: Callable[[Dict], None]) -> str:
    from src.agent_loop import stream_agent_loop
    try:
        from src.endpoint_resolver import resolve_chat_fallback_candidates
        fallbacks = resolve_chat_fallback_candidates(owner)
    except Exception:
        fallbacks = []
    text = ""
    async for event_str in stream_agent_loop(
        endpoint_url=endpoint_url, model=model, messages=messages, headers=headers,
        session_id=session_id, owner=owner, fallbacks=fallbacks,
    ):
        if not event_str.startswith("data: ") or event_str.startswith("data: [DONE]"):
            continue
        try:
            data = json.loads(event_str[6:])
        except json.JSONDecodeError:
            continue
        if "delta" in data:
            text += data["delta"]
        else:
            on_event(data)
    return text


class run_scope:
    """Pin workdir and extra tool env for one agent run (context-local)."""

    def __init__(self, workdir=None, env=None):
        self._workdir, self._env = workdir, env or {}

    def __enter__(self):
        self._t1 = ACTIVE_WORKDIR.set(self._workdir)
        self._t2 = EXTRA_ENV.set(dict(self._env))
        return self

    def __exit__(self, *exc):
        EXTRA_ENV.reset(self._t2)
        ACTIVE_WORKDIR.reset(self._t1)
        return False
