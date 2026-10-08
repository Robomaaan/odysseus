"""
mcp_manager.py

Manages connections to MCP (Model Context Protocol) tool servers.
Each server exposes tools that are made available to the agent loop.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Remote transports ("sse", "http") reuse the server's `env` column as HTTP
# headers. Values may reference process env vars as ${NAME} so bearer tokens
# stay in the environment instead of the database.
REMOTE_TRANSPORTS = ("sse", "http")
REMOTE_CONNECT_TIMEOUT = 20
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_header_values(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Resolve ${NAME} references in header values from os.environ.

    Raises ValueError naming the missing variable instead of silently sending
    an empty or literal "${NAME}" credential.
    """
    resolved: Dict[str, str] = {}
    for key, value in (headers or {}).items():
        def _sub(match):
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"Header {key!r} references unset environment variable {name}")
            return os.environ[name]
        resolved[str(key)] = _ENV_REF.sub(_sub, str(value))
    return resolved

def _format_mcp_connection_error(name: str, command: str = "", args: Optional[List[str]] = None, error: Exception = None) -> str:
    """Return a user-actionable MCP connection error message."""
    args = args or []
    raw_error = str(error) if error else "Unknown error"
    command_line = " ".join([command or "", *args]).strip()
    lower_command = command_line.lower()

    if "@playwright/mcp" in lower_command:
        return (
            f"{raw_error}\n\n"
            "Browser MCP could not start. On fresh installs, cache the Playwright MCP package once before connecting:\n\n"
            "npx -y @playwright/mcp@latest --version\n\n"
            "Then restart Odysseus and reconnect the Browser MCP server."
        )

    return raw_error



class _RemoteHandle:
    """Stands in for an AsyncExitStack: closing signals the connection's worker task."""

    def __init__(self, stop, worker):
        self._stop = stop
        self._worker = worker

    async def aclose(self):
        self._stop.set()
        try:
            await asyncio.wait_for(self._worker, timeout=5)
        except BaseException:
            self._worker.cancel()


def _unwrap_connect_error(exc: BaseException) -> Exception:
    """Surface the real cause (e.g. HTTP 401) from anyio task-group wrappers."""
    seen = 0
    while seen < 10:
        seen += 1
        inner = getattr(exc, "exceptions", None)
        if inner:
            # Prefer a non-cancellation leaf (the HTTP error) over CancelledError.
            picks = [e for e in inner if not isinstance(e, asyncio.CancelledError)]
            exc = (picks or list(inner))[0]
            continue
        cause = exc.__cause__ or exc.__context__
        if isinstance(exc, (asyncio.CancelledError, RuntimeError)) and cause is not None:
            exc = cause
            continue
        break
    if isinstance(exc, Exception):
        return exc
    return RuntimeError(f"connection aborted ({type(exc).__name__})")


class McpManager:
    """Manages MCP server connections and tool routing."""

    def __init__(self):
        # server_id -> connection state
        self._connections: Dict[str, Dict[str, Any]] = {}
        # server_id -> list of tool schemas
        self._tools: Dict[str, List[Dict]] = {}
        # server_id -> MCP ClientSession
        self._sessions: Dict[str, Any] = {}
        # server_id -> exit stack (for cleanup)
        self._stacks: Dict[str, Any] = {}
        # Tracking updates to tools/connections for RAG indexing
        self._generation = 0

    async def connect_server(
        self,
        server_id: str,
        name: str,
        transport: str,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        url: Optional[str] = None,
    ) -> bool:
        """Connect to an MCP server via stdio, SSE or Streamable HTTP transport."""
        try:
            if transport == "stdio":
                res = await self._connect_stdio(server_id, name, command, args or [], env or {})
            elif transport in REMOTE_TRANSPORTS:
                res = await self._connect_remote(server_id, name, url, transport, expand_header_values(env))
            else:
                logger.error(f"Unknown MCP transport: {transport}")
                res = False
            if res:
                self._generation += 1
            return res
        except Exception as e:
            logger.error(f"Failed to connect MCP server {name} ({server_id}): {e}")
            error_message = _format_mcp_connection_error(name, command or "", args or [], e)
            self._connections[server_id] = {"status": "error", "error": error_message, "name": name}
            self._generation += 1
            return False

    async def _connect_stdio(self, server_id: str, name: str, command: str, args: List[str], env: Dict[str, str]) -> bool:
        """Connect to an MCP server via stdio transport."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from contextlib import AsyncExitStack

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env={**os.environ, **env} if env else None,
            )

            stack = AsyncExitStack()
            try:
                transport = await stack.enter_async_context(stdio_client(server_params))
                read_stream, write_stream = transport
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))

                await session.initialize()

                # Discover tools
                tools_result = await session.list_tools()
            except Exception:
                await stack.aclose()
                raise
            tools = []
            for tool in tools_result.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                })

            self._sessions[server_id] = session
            self._stacks[server_id] = stack
            self._tools[server_id] = tools
            # Extract identity hints from env vars (e.g. email address, API name)
            # so tool descriptions can distinguish between multiple instances of
            # the same MCP server (e.g. two email accounts).
            identity_hints = []
            for k, v in (env or {}).items():
                k_lower = k.lower()
                if any(x in k_lower for x in ['email_address', 'account', 'user', 'username']):
                    identity_hints.append(v)
            identity = ", ".join(identity_hints) if identity_hints else ""

            self._connections[server_id] = {
                "status": "connected",
                "name": name,
                "transport": "stdio",
                "tool_count": len(tools),
                "identity": identity,
            }

            logger.info(f"MCP server connected: {name} ({server_id}) - {len(tools)} tools via stdio")
            return True

        except ImportError:
            logger.warning("MCP package not installed. Install with: pip install mcp")
            self._connections[server_id] = {"status": "error", "error": "mcp package not installed", "name": name}
            return False

    async def _connect_remote(self, server_id: str, name: str, url: str, transport_name: str,
                              headers: Dict[str, str]) -> bool:
        """Connect to a remote MCP server via SSE or Streamable HTTP transport.

        The mcp client context managers use anyio task groups that must be
        entered and exited in the same task; a failed handshake (e.g. HTTP 401)
        otherwise surfaces as CancelledError / "exit cancel scope in a different
        task" and escapes connect_server. A dedicated worker task owns the
        connection for its whole lifetime and reports the outcome via a future.
        """
        try:
            from mcp import ClientSession  # noqa: F401  (fail fast if missing)
        except ImportError:
            logger.warning("MCP package not installed. Install with: pip install mcp")
            self._connections[server_id] = {"status": "error", "error": "mcp package not installed", "name": name}
            return False

        loop = asyncio.get_running_loop()
        ready: asyncio.Future = loop.create_future()
        stop = asyncio.Event()
        worker = asyncio.create_task(
            self._remote_worker(url, transport_name, headers, ready, stop),
            name=f"mcp-remote-{server_id}",
        )
        try:
            session, tools, label = await asyncio.wait_for(asyncio.shield(ready), timeout=REMOTE_CONNECT_TIMEOUT)
        except BaseException as e:
            stop.set()
            if not ready.done():
                worker.cancel()
            try:
                await asyncio.wait_for(worker, timeout=5)
            except BaseException:
                pass
            if isinstance(e, asyncio.TimeoutError):
                raise RuntimeError(f"timed out after {REMOTE_CONNECT_TIMEOUT}s connecting to {url}") from None
            if isinstance(e, asyncio.CancelledError) and not ready.done():
                raise  # we ourselves were cancelled
            raise

        self._sessions[server_id] = session
        self._stacks[server_id] = _RemoteHandle(stop, worker)
        self._tools[server_id] = tools
        self._connections[server_id] = {
            "status": "connected",
            "name": name,
            "transport": transport_name,
            "tool_count": len(tools),
        }
        logger.info(f"MCP server connected: {name} ({server_id}) - {len(tools)} tools via {label}")
        return True

    @staticmethod
    async def _remote_worker(url: str, transport_name: str, headers: Dict[str, str],
                             ready: "asyncio.Future", stop: "asyncio.Event"):
        from mcp import ClientSession
        from contextlib import AsyncExitStack
        try:
            async with AsyncExitStack() as stack:
                if transport_name == "http":
                    import mcp.client.streamable_http as _sh
                    if hasattr(_sh, "streamable_http_client"):
                        # mcp >= 2: headers live on the httpx client we pass in.
                        http_client = await stack.enter_async_context(
                            _sh.create_mcp_http_client(headers=headers or None))
                        client = _sh.streamable_http_client(url, http_client=http_client)
                    else:
                        client = _sh.streamablehttp_client(url, headers=headers or None)
                    label = "Streamable HTTP"
                else:
                    from mcp.client.sse import sse_client
                    client = sse_client(url, headers=headers or None)
                    label = "SSE"
                transport = await stack.enter_async_context(client)
                # streamablehttp_client yields (read, write, get_session_id)
                session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
                await session.initialize()
                tools_result = await session.list_tools()
                tools = [{
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                } for tool in tools_result.tools]
                ready.set_result((session, tools, label))
                await stop.wait()
        except BaseException as e:
            if not ready.done():
                ready.set_exception(_unwrap_connect_error(e))
            elif not isinstance(e, asyncio.CancelledError):
                logger.warning(f"MCP remote connection to {url} ended: {type(e).__name__}: {e}")

    async def disconnect_server(self, server_id: str):
        """Disconnect from an MCP server."""
        stack = self._stacks.pop(server_id, None)
        if stack:
            try:
                await stack.aclose()
            except Exception as e:
                logger.warning(f"Error closing MCP server {server_id}: {e}")

        self._sessions.pop(server_id, None)
        self._tools.pop(server_id, None)
        self._connections.pop(server_id, None)
        self._generation += 1
        logger.info(f"MCP server disconnected: {server_id}")

    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        ids = list(self._sessions.keys())
        for sid in ids:
            await self.disconnect_server(sid)

    async def connect_all_enabled(self):
        """Connect to all enabled MCP servers from the database."""
        from src.database import McpServer, SessionLocal

        db = SessionLocal()
        try:
            servers = db.query(McpServer).filter(McpServer.is_enabled == True).all()
            for srv in servers:
                args = json.loads(srv.args) if srv.args else []
                env = json.loads(srv.env) if srv.env else {}
                await self.connect_server(
                    server_id=srv.id,
                    name=srv.name,
                    transport=srv.transport,
                    command=srv.command,
                    args=args,
                    env=env,
                    url=srv.url,
                )
        finally:
            db.close()

    async def call_tool(self, qualified_name: str, arguments: Dict) -> Dict:
        """Call an MCP tool by its qualified name (mcp__{server_id}__{tool_name}).

        Returns a result dict compatible with agent_tools format.
        """
        parts = qualified_name.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            return {"error": f"Invalid MCP tool name: {qualified_name}", "exit_code": 1}

        server_id = parts[1]
        tool_name = parts[2]

        session = self._sessions.get(server_id)
        if not session:
            return {"error": f"MCP server not connected: {server_id}", "exit_code": 1}

        try:
            result = await self._do_call(session, tool_name, arguments)
        except Exception as e:
            # Auto-reconnect for builtin servers whose subprocess may have died
            if self.is_builtin(server_id):
                logger.warning(f"MCP call failed for {qualified_name}, attempting reconnect: {e}")
                reconnected = await self._reconnect_builtin(server_id)
                if reconnected:
                    session = self._sessions.get(server_id)
                    if session:
                        try:
                            result = await self._do_call(session, tool_name, arguments)
                        except Exception as e2:
                            logger.error(f"MCP tool call failed after reconnect: {qualified_name}: {e2}")
                            return {"error": str(e2), "exit_code": 1}
                    else:
                        return {"error": f"Reconnected but no session for {server_id}", "exit_code": 1}
                else:
                    logger.error(f"MCP reconnect failed for {server_id}")
                    return {"error": f"MCP server crashed and reconnect failed: {server_id}", "exit_code": 1}
            else:
                logger.error(f"MCP tool call failed: {qualified_name}: {e}")
                return {"error": str(e), "exit_code": 1}

        return result

    async def _do_call(self, session, tool_name: str, arguments: Dict) -> Dict:
        """Execute a single MCP tool call and return result dict."""
        result = await session.call_tool(tool_name, arguments)
        output_parts = []
        images = []
        for content in result.content:
            if hasattr(content, 'text'):
                output_parts.append(content.text)
            elif getattr(content, 'type', '') == 'image' and hasattr(content, 'data'):
                # Image content (e.g. Playwright screenshots)
                mime = getattr(content, 'mimeType', 'image/png')
                images.append({"data": content.data, "mimeType": mime})
                output_parts.append(f"[Screenshot captured ({mime})]")
            elif hasattr(content, 'data'):
                output_parts.append(str(content.data))

        output = "\n".join(output_parts)
        is_error = getattr(result, 'isError', False)

        result_dict = {
            "stdout": output if not is_error else "",
            "stderr": output if is_error else "",
            "exit_code": 1 if is_error else 0,
        }
        if images:
            result_dict["images"] = images
        return result_dict

    async def _reconnect_builtin(self, server_id: str) -> bool:
        """Tear down and reconnect a crashed builtin MCP server."""
        import sys
        from src.builtin_mcp import _BUILTIN_SERVERS

        if server_id not in _BUILTIN_SERVERS:
            return False

        script_rel, name = _BUILTIN_SERVERS[server_id]
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(base_dir, script_rel)

        # Clean up old connection
        await self.disconnect_server(server_id)

        try:
            ok = await self.connect_server(
                server_id=server_id,
                name=name,
                transport="stdio",
                command=sys.executable,
                args=[script_path],
                env={"PYTHONPATH": base_dir},
            )
            if ok:
                logger.info(f"Reconnected builtin MCP server: {name}")
            return ok
        except Exception as e:
            logger.error(f"Failed to reconnect builtin MCP server {name}: {e}")
            return False

    def get_all_openai_schemas(self, disabled_map: Optional[Dict[str, set]] = None) -> List[Dict]:
        """Return all MCP tools in OpenAI function-calling format.

        Tool names are namespaced as mcp__{server_id}__{tool_name}.
        disabled_map: optional {server_id: set_of_disabled_tool_names} to filter out.
        """
        schemas = []
        for server_id, tools in self._tools.items():
            # Skip builtin Python servers — they use the code-block tool format
            # But include NPX-based builtins (like browser) which need function calling
            if self.is_builtin(server_id) and server_id != "builtin_browser":
                continue
            conn = self._connections.get(server_id, {})
            server_name = conn.get("name", server_id)
            disabled = (disabled_map or {}).get(server_id, set())

            identity = conn.get("identity", "")
            label = f"{server_name} ({identity})" if identity else server_name

            for tool in tools:
                if tool["name"] in disabled:
                    continue
                qualified = f"mcp__{server_id}__{tool['name']}"
                schema = {
                    "type": "function",
                    "function": {
                        "name": qualified,
                        "description": f"[MCP:{label}] {tool['description']}",
                        "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
                schemas.append(schema)

        return schemas

    def get_all_tools(self, disabled_map: Optional[Dict[str, set]] = None) -> List[Dict]:
        """Return a flat list of all discovered tools with server info."""
        result = []
        for server_id, tools in self._tools.items():
            conn = self._connections.get(server_id, {})
            disabled = (disabled_map or {}).get(server_id, set())
            for tool in tools:
                result.append({
                    "server_id": server_id,
                    "server_name": conn.get("name", server_id),
                    "name": tool["name"],
                    "qualified_name": f"mcp__{server_id}__{tool['name']}",
                    "description": tool.get("description", ""),
                    "is_disabled": tool["name"] in disabled,
                })
        return result

    def is_builtin(self, server_id: str) -> bool:
        """Check if a server is a built-in (auto-registered) server."""
        return server_id.startswith("builtin_") or server_id in {
            "image_gen",
            "memory",
            "rag",
            "email",
        }

    def get_server_status(self, server_id: str) -> Dict:
        """Get connection status for a server."""
        return self._connections.get(server_id, {"status": "disconnected"})

    def get_all_statuses(self) -> Dict[str, Dict]:
        """Get connection statuses for all servers."""
        return dict(self._connections)

    _cached_prompt_desc = None
    _cached_prompt_desc_key = None

    def get_tool_descriptions_for_prompt(self, disabled_map: Optional[Dict[str, set]] = None) -> str:
        """Generate text describing MCP tools for the agent system prompt. Cached."""
        cache_key = (
            frozenset((k, frozenset(v)) for k, v in (disabled_map or {}).items()),
            len(self._tools),
            self._generation,
        )
        if self._cached_prompt_desc is not None and self._cached_prompt_desc_key == cache_key:
            return self._cached_prompt_desc
        tools = self.get_all_tools(disabled_map)
        if not tools:
            return ""

        lines = ["\n\nYou also have access to external MCP tool servers. These tools are called via native function calling:"]
        by_server = {}
        for t in tools:
            # Skip builtin Python servers — they're already in the agent prompt
            # But include NPX-based builtins (like browser) which aren't hardcoded
            if self.is_builtin(t["server_id"]) and t["server_id"] != "builtin_browser":
                continue
            if t.get("is_disabled"):
                continue
            sn = t["server_name"]
            if sn not in by_server:
                by_server[sn] = []
            by_server[sn].append(t)

        if not by_server:
            return ""

        for server_name, server_tools in by_server.items():
            # Include identity (e.g. email address) if available
            sid = server_tools[0]["server_id"] if server_tools else ""
            identity = self._connections.get(sid, {}).get("identity", "")
            label = f"{server_name} ({identity})" if identity else server_name
            lines.append(f"\n**{label}:**")
            for t in server_tools:
                # Truncate long descriptions
                desc = t['description'][:120] + '...' if len(t['description']) > 120 else t['description']
                lines.append(f"  - {t['qualified_name']}: {desc}")

        result = "\n".join(lines)
        self._cached_prompt_desc = result
        self._cached_prompt_desc_key = cache_key
        return result
