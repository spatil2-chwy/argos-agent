"""FastAPI entrypoint for the Argos observability dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from argos_src.observability.dashboard_data import DEFAULT_LOG_PATH, load_dashboard_snapshot


LOG_PATH_ENV = "ARGOS_DASHBOARD_LOG_PATH"
REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIST = REPO_ROOT / "dashboard" / "dist"


def _resolve_log_path() -> Path:
    selected = os.getenv(LOG_PATH_ENV) or str(DEFAULT_LOG_PATH)
    path = Path(selected)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Argos Observability Dashboard",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        path = _resolve_log_path()
        return {
            "ok": True,
            "log_path": str(path),
            "log_exists": path.exists(),
            "frontend_built": DASHBOARD_DIST.exists(),
        }

    @app.get("/api/snapshot")
    def snapshot() -> dict[str, Any]:
        path = _resolve_log_path()
        if not path.exists():
            return load_dashboard_snapshot(path)
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"Not a log file: {path}")
        return load_dashboard_snapshot(path)

    if DASHBOARD_DIST.exists():
        assets = DASHBOARD_DIST / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="dashboard-assets")

        @app.get("/{path:path}", include_in_schema=False)
        def dashboard(path: str) -> FileResponse:
            candidate = (DASHBOARD_DIST / path).resolve()
            if candidate.is_file() and DASHBOARD_DIST in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(DASHBOARD_DIST / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def missing_frontend() -> dict[str, str]:
            return {
                "message": "Dashboard API is running. Build the Vite frontend in dashboard/ to serve the UI here.",
            }

    return app


app = create_app()
