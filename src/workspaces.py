"""
workspaces.py

Named project workspaces (git repos or plain folders) the agent can work in.

ODYSSEUS_PROJECTS points at a JSON file:
    {
      "default": "ki-firma-control",
      "projects": [
        {"name": "ki-firma-control", "path": "G:/KI-Firma/repos/KI-Firma-Control",
         "description": "Steuer-Repo der KI-Firma: Architektur, Status, Abnahme"}
      ]
    }

Callers that run the agent for a project (e.g. the Telegram bridge) set
ACTIVE_WORKDIR for the duration of the run. The bash/python tools then start
in that directory and relative read_file/write_file paths resolve against it.
Every configured project path is also an allowed root for file tools.
"""

import json
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, List, Optional

CONFIG_ENV = "ODYSSEUS_PROJECTS"
CONTEXT_FILES = ("CLAUDE.md", "AGENTS.md", "README.md")
MAX_CONTEXT_CHARS = 4000

ACTIVE_WORKDIR: ContextVar[Optional[str]] = ContextVar("odysseus_active_workdir", default=None)
# Extra environment for bash/python tools during one run (e.g. PAPERCLIP_RUN_ID).
EXTRA_ENV: ContextVar[Dict[str, str]] = ContextVar("odysseus_extra_env", default={})


@dataclass(frozen=True)
class Project:
    name: str
    path: str
    description: str = ""


_cache: Dict[str, object] = {"key": None, "projects": [], "default": None}


def _load() -> None:
    path = os.environ.get(CONFIG_ENV, "").strip()
    try:
        key = (path, os.path.getmtime(path)) if path else None
    except OSError:
        key = (path, None)
    if key == _cache["key"]:
        return
    projects: List[Project] = []
    default = None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for p in data.get("projects") or []:
            name = (p.get("name") or "").strip()
            ppath = (p.get("path") or "").strip()
            if name and ppath:
                projects.append(Project(name, os.path.realpath(os.path.expanduser(ppath)), p.get("description", "")))
        default = data.get("default")
    _cache.update(key=key, projects=projects, default=default)


def list_projects() -> List[Project]:
    _load()
    return list(_cache["projects"])


def get_project(name: Optional[str]) -> Optional[Project]:
    """Look up a project by exact or case-insensitive name; None → default."""
    _load()
    wanted = name or _cache["default"]
    if not wanted:
        return None
    for p in _cache["projects"]:
        if p.name == wanted or p.name.lower() == str(wanted).lower():
            return p
    return None


def project_roots() -> List[str]:
    return [p.path for p in list_projects()]


def resolve_relative(raw_path: str) -> str:
    """Anchor a relative path at the active workdir, if one is set."""
    workdir = ACTIVE_WORKDIR.get()
    if workdir and not os.path.isabs(os.path.expanduser(raw_path)):
        return os.path.join(workdir, raw_path)
    return raw_path


def project_context(project: Project) -> str:
    """System-prompt block describing the project and its agent instructions."""
    lines = [
        f"## Aktives Projekt: {project.name}",
        f"Arbeitsverzeichnis: {project.path}",
        "Shell-Befehle laufen in diesem Verzeichnis; relative Dateipfade beziehen sich darauf.",
    ]
    if project.description:
        lines.append(f"Beschreibung: {project.description}")
    for fname in CONTEXT_FILES:
        fpath = os.path.join(project.path, fname)
        if os.path.isfile(fpath):
            with open(fpath, encoding="utf-8", errors="replace") as f:
                text = f.read(MAX_CONTEXT_CHARS + 1)
            if len(text) > MAX_CONTEXT_CHARS:
                text = text[:MAX_CONTEXT_CHARS] + "\n[… gekürzt]"
            lines += ["", f"### {fname}", text.strip()]
            break
    return "\n".join(lines)
