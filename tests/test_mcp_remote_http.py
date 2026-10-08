"""Streamable HTTP MCP transport with bearer headers (The Manager speaks only this)."""
import asyncio
import socket
import threading
import time

import pytest

from src.mcp_manager import McpManager, expand_header_values

TOKEN = "s3cret-test-token"


def test_expand_header_values_resolves_env(monkeypatch):
    monkeypatch.setenv("MGR_TOKEN", "abc")
    assert expand_header_values({"Authorization": "Bearer ${MGR_TOKEN}", "X": "plain"}) == {
        "Authorization": "Bearer abc", "X": "plain"}


def test_expand_header_values_missing_env_fails(monkeypatch):
    monkeypatch.delenv("MGR_TOKEN_MISSING", raising=False)
    with pytest.raises(ValueError, match="MGR_TOKEN_MISSING"):
        expand_header_values({"Authorization": "Bearer ${MGR_TOKEN_MISSING}"})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mcp_http_server():
    pytest.importorskip("uvicorn")
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP as MCPServer
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    import uvicorn

    srv = MCPServer("test-manager")

    @srv.tool()
    def list_projects() -> str:
        """Return project names."""
        return "KI-Firma, Odysseus"

    app = srv.streamable_http_app()

    class BearerAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.headers.get("authorization") != f"Bearer {TOKEN}":
                return JSONResponse({"error": "MACHINE_CREDENTIAL_INVALID"}, status_code=401)
            return await call_next(request)

    app.add_middleware(BearerAuth)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=5)


async def test_http_transport_with_bearer(mcp_http_server, monkeypatch):
    monkeypatch.setenv("TEST_MGR_TOKEN", TOKEN)
    mgr = McpManager()
    ok = await mgr.connect_server("cfg-the-manager", "the-manager", "http", url=mcp_http_server,
                                  env={"Authorization": "Bearer ${TEST_MGR_TOKEN}"})
    try:
        assert ok, mgr.get_server_status("cfg-the-manager")
        assert mgr.get_server_status("cfg-the-manager")["transport"] == "http"
        result = await mgr.call_tool("mcp__cfg-the-manager__list_projects", {})
        assert result["exit_code"] == 0
        assert "KI-Firma" in result["stdout"]
    finally:
        await mgr.disconnect_all()


async def test_http_transport_wrong_token_reports_error(mcp_http_server, monkeypatch):
    monkeypatch.setenv("TEST_MGR_TOKEN", "wrong")
    mgr = McpManager()
    ok = await asyncio.wait_for(
        mgr.connect_server("cfg-bad", "bad", "http", url=mcp_http_server,
                           env={"Authorization": "Bearer ${TEST_MGR_TOKEN}"}),
        timeout=20)
    assert not ok
    status = mgr.get_server_status("cfg-bad")
    assert status["status"] == "error"
    assert "401" in status["error"], status["error"]
    await mgr.disconnect_all()
