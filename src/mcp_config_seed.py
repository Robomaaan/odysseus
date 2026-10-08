"""
mcp_config_seed.py

Declarative MCP server config for headless deployments.

When ODYSSEUS_MCP_CONFIG points at a JSON file, its servers are upserted into
the mcp_servers table on startup, keyed by a stable id derived from the name.
Servers added through the UI are left alone. Secrets never belong in the file:
header values reference environment variables as ${NAME} and are resolved at
connect time (see mcp_manager.expand_header_values).

File format:
    {
      "servers": [
        {"name": "the-manager", "transport": "http",
         "url": "https://example.vercel.app/mcp",
         "headers": {"Authorization": "Bearer ${MANAGER_MCP_TOKEN}"}},
        {"name": "files", "transport": "stdio",
         "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "G:/KI-Firma/repos"],
         "env": {}}
      ]
    }
"""

import json
import logging
import os
import re
from typing import Dict, List

from src.mcp_manager import REMOTE_TRANSPORTS

logger = logging.getLogger(__name__)

CONFIG_ENV = "ODYSSEUS_MCP_CONFIG"
ID_PREFIX = "cfg-"


def server_id_for(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise ValueError(f"MCP server name {name!r} has no usable characters")
    return ID_PREFIX + slug


def parse_config(data: Dict) -> List[Dict]:
    """Validate the config dict and return normalized server rows."""
    rows = []
    seen = set()
    for i, srv in enumerate(data.get("servers") or []):
        name = (srv.get("name") or "").strip()
        transport = srv.get("transport", "stdio")
        if not name:
            raise ValueError(f"servers[{i}]: name is required")
        if transport not in ("stdio", *REMOTE_TRANSPORTS):
            raise ValueError(f"servers[{i}] {name}: unknown transport {transport!r}")
        if transport == "stdio" and not srv.get("command"):
            raise ValueError(f"servers[{i}] {name}: command is required for stdio")
        if transport in REMOTE_TRANSPORTS and not srv.get("url"):
            raise ValueError(f"servers[{i}] {name}: url is required for {transport}")
        sid = server_id_for(name)
        if sid in seen:
            raise ValueError(f"servers[{i}] {name}: duplicate name")
        seen.add(sid)
        # Remote transports keep their headers in the env column.
        env = srv.get("headers") if transport in REMOTE_TRANSPORTS else srv.get("env")
        rows.append({
            "id": sid,
            "name": name,
            "transport": transport,
            "command": srv.get("command"),
            "args": json.dumps(srv.get("args") or []),
            "env": json.dumps(env or {}),
            "url": srv.get("url"),
            "is_enabled": bool(srv.get("enabled", True)),
            "disabled_tools": json.dumps(srv["disabled_tools"]) if srv.get("disabled_tools") else None,
        })
    return rows


def seed_from_env() -> int:
    """Upsert servers from $ODYSSEUS_MCP_CONFIG. Returns the number of rows written."""
    path = os.environ.get(CONFIG_ENV, "").strip()
    if not path:
        return 0
    with open(path, encoding="utf-8") as f:
        rows = parse_config(json.load(f))

    from core.database import McpServer, SessionLocal
    db = SessionLocal()
    try:
        for row in rows:
            srv = db.query(McpServer).filter(McpServer.id == row["id"]).first()
            if srv is None:
                db.add(McpServer(**row))
            else:
                for key, value in row.items():
                    setattr(srv, key, value)
        db.commit()
    finally:
        db.close()
    logger.info(f"MCP config: {len(rows)} server(s) from {path}")
    return len(rows)
