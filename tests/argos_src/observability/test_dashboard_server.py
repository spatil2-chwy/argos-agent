from argos_src.observability.dashboard_data import load_dashboard_snapshot
from argos_src.observability.dashboard_server import _resolve_log_path, create_app


def test_dashboard_app_exposes_api_routes() -> None:
    app = create_app()

    paths = {route.path for route in app.routes}

    assert "/api/health" in paths
    assert "/api/snapshot" in paths


def test_dashboard_log_path_resolution_ignores_query_style_absolute_paths(tmp_path) -> None:
    log_path = tmp_path / "latency.log"

    assert _resolve_log_path() != log_path.resolve()


def test_dashboard_log_path_resolution_uses_env_override(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "latency.log"

    monkeypatch.setenv("ARGOS_DASHBOARD_LOG_PATH", str(log_path))

    assert _resolve_log_path() == log_path.resolve()


def test_dashboard_snapshot_route_has_no_log_path_query_parameter() -> None:
    app = create_app()
    snapshot_route = next(route for route in app.routes if route.path == "/api/snapshot")

    assert "log_path" not in {param.name for param in snapshot_route.dependant.query_params}


def test_dashboard_snapshot_loader_reads_log_file(tmp_path) -> None:
    log_path = tmp_path / "latency.log"
    log_path.write_text(
        "ts=2026-07-07 10:00:00.000 | component=realtime | "
        "event=response_usage | req_id=rt-1\n",
        encoding="utf-8",
    )

    body = load_dashboard_snapshot(log_path)

    assert body["source"] == str(log_path)
    assert body["summary"]["interaction_count"] == 1
    assert body["interactions"][0]["status"] == "complete"
