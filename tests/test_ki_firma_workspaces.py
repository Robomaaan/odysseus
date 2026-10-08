"""Project workspaces and declarative MCP config."""
import json
import os

import pytest

from src import workspaces
from src.mcp_config_seed import parse_config, server_id_for


@pytest.fixture
def projects(tmp_path, monkeypatch):
    repo = tmp_path / "KI-Firma-Control"
    repo.mkdir()
    (repo / "README.md").write_text("# Eigene KI-Firma\nZielarchitektur", encoding="utf-8")
    other = tmp_path / "paperclip"
    other.mkdir()
    cfg = tmp_path / "projects.json"
    cfg.write_text(json.dumps({
        "default": "ki-firma-control",
        "projects": [
            {"name": "ki-firma-control", "path": str(repo), "description": "Steuer-Repo"},
            {"name": "paperclip", "path": str(other)},
        ],
    }), encoding="utf-8")
    monkeypatch.setenv(workspaces.CONFIG_ENV, str(cfg))
    workspaces._cache["key"] = None
    return repo, other


def test_lookup_default_and_case_insensitive(projects):
    repo, other = projects
    assert workspaces.get_project(None).path == os.path.realpath(repo)
    assert workspaces.get_project("PaperClip").path == os.path.realpath(other)
    assert workspaces.get_project("gibts-nicht") is None


def test_project_context_includes_readme(projects):
    ctx = workspaces.project_context(workspaces.get_project("ki-firma-control"))
    assert "Arbeitsverzeichnis" in ctx and "Zielarchitektur" in ctx


def test_relative_paths_anchor_at_active_workdir(projects):
    repo, _ = projects
    token = workspaces.ACTIVE_WORKDIR.set(str(repo))
    try:
        assert workspaces.resolve_relative("docs/x.md") == os.path.join(str(repo), "docs/x.md")
        assert workspaces.resolve_relative(str(repo / "abs.md")) == str(repo / "abs.md")
    finally:
        workspaces.ACTIVE_WORKDIR.reset(token)
    assert workspaces.resolve_relative("docs/x.md") == "docs/x.md"


def test_file_tools_may_read_project_files(projects):
    from src.tool_execution import _resolve_tool_path
    repo, _ = projects
    token = workspaces.ACTIVE_WORKDIR.set(str(repo))
    try:
        assert _resolve_tool_path("README.md") == os.path.realpath(repo / "README.md")
    finally:
        workspaces.ACTIVE_WORKDIR.reset(token)


async def test_bash_runs_in_active_workdir(projects):
    from src.tool_execution import _direct_fallback
    repo, _ = projects
    token = workspaces.ACTIVE_WORKDIR.set(str(repo))
    try:
        res = await _direct_fallback("python", "import os; print(os.getcwd())")
    finally:
        workspaces.ACTIVE_WORKDIR.reset(token)
    assert os.path.realpath(res["output"].strip()) == os.path.realpath(repo)


def test_parse_config_maps_headers_to_env():
    rows = parse_config({"servers": [
        {"name": "The Manager", "transport": "http", "url": "https://m/mcp",
         "headers": {"Authorization": "Bearer ${MANAGER_MCP_TOKEN}"}},
        {"name": "files", "command": "npx", "args": ["-y", "x"]},
    ]})
    assert rows[0]["id"] == "cfg-the-manager"
    assert json.loads(rows[0]["env"]) == {"Authorization": "Bearer ${MANAGER_MCP_TOKEN}"}
    assert rows[1]["transport"] == "stdio" and json.loads(rows[1]["args"]) == ["-y", "x"]


@pytest.mark.parametrize("bad, msg", [
    ({"transport": "http", "url": "u"}, "name is required"),
    ({"name": "a", "transport": "ws", "url": "u"}, "unknown transport"),
    ({"name": "a", "transport": "http"}, "url is required"),
    ({"name": "a"}, "command is required"),
])
def test_parse_config_rejects_invalid(bad, msg):
    with pytest.raises(ValueError, match=msg):
        parse_config({"servers": [bad]})


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        parse_config({"servers": [{"name": "A", "command": "x"}, {"name": "a", "command": "y"}]})
    assert server_id_for("Paperclip API") == "cfg-paperclip-api"
