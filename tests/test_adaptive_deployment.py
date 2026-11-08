"""Existing interface and deployment boundary checks. No research/API calls."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.server.production import ResearchAuth, prepare_storage


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCH_PASSWORD", "fixture-password-with-24-chars")
    monkeypatch.setenv("RESEARCH_USERNAME", "greg")
    monkeypatch.setenv("REPORT_STORE_PATH", str(tmp_path / "reports.json"))
    from backend.server.app import app
    with TestClient(ResearchAuth(app), base_url="https://testserver") as c:
        yield c


def auth():
    return ("greg", "fixture-password-with-24-chars")


def test_authenticated_existing_interface_and_static_assets(client):
    response = client.get("/", auth=auth())
    assert response.status_code == 200
    assert 'id="task"' in response.text and "deep" in response.text
    assert client.get("/site/scripts.js", auth=auth()).status_code == 200
    assert client.get("/openapi.json", auth=auth()).status_code == 200


@pytest.mark.parametrize("path", ["/", "/api/reports", "/outputs/test.json", "/docs", "/site/scripts.js"])
def test_private_endpoints_require_authentication(client, path):
    assert client.get(path).status_code == 401


def test_health_exposes_only_readiness_and_mutations_reject_cross_origin(client):
    assert client.get("/__health").status_code == 200
    response = client.post("/api/reports", auth=auth(), headers={"Origin": "https://other.example"}, json={})
    assert response.status_code == 401


def test_websocket_keeps_original_ping_interface_and_rejects_unauthenticated(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws"):
            pass
    basic = "Basic " + base64.b64encode(":".join(auth()).encode()).decode()
    with client.websocket_connect("/ws", headers={"Authorization": basic, "Origin": "https://testserver"}) as ws:
        ws.send_text("ping")
        assert ws.receive_text() == "pong"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"Authorization": basic, "Origin": "https://other.example"}):
            pass


def test_missing_password_fails_closed(client, monkeypatch):
    monkeypatch.delenv("RESEARCH_PASSWORD")
    assert client.get("/").status_code == 503


def test_volume_path_and_output_symlink_are_repeatable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("outputs").mkdir()
    root = tmp_path / "volume"
    prepare_storage(root)
    (root / "outputs" / "retained.txt").write_text("saved")
    prepare_storage(root)
    assert Path("outputs/retained.txt").read_text() == "saved"


def test_storage_does_not_delete_existing_nonempty_local_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("outputs").mkdir()
    Path("outputs/keep.txt").write_text("keep")
    with pytest.raises(OSError):
        prepare_storage(tmp_path / "volume")
    assert Path("outputs/keep.txt").read_text() == "keep"


@pytest.mark.asyncio
async def test_browser_disconnect_does_not_destroy_stream_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from backend.server.server_utils import CustomLogsHandler
    socket = SimpleNamespace(send_json=AsyncMock(side_effect=RuntimeError("closed")))
    handler = CustomLogsHandler(socket, "fixture")
    await handler.send_json({"type": "logs", "content": "evidence", "output": "Fixture progress"})
    assert handler.websocket is None
    assert json.loads(Path(handler.log_file).read_text())["events"]


@pytest.mark.asyncio
async def test_final_download_paths_preserve_native_knowledge_and_trace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from backend.server import server_utils as su
    artifacts = {"json": "outputs/run.knowledge.json", "trace": "outputs/run.trace.json", "state": "outputs/run.state.json", "md": "outputs/run.md"}
    socket = SimpleNamespace(send_json=AsyncMock(), state=SimpleNamespace())
    async def start(*args):
        socket.state.research_artifacts = artifacts
        return "fixture report"
    monkeypatch.setattr(su, "generate_report_files", AsyncMock(return_value={"pdf": "report.pdf", "md": "legacy.md"}))
    data = "start " + json.dumps({"task": "fixture", "report_type": "deep", "report_source": "web", "tone": "Objective"})
    await su.handle_start_command(socket, data, SimpleNamespace(start_streaming=start))
    output = socket.send_json.call_args.args[0]["output"]
    assert output["json"] == artifacts["json"] and output["trace"] == artifacts["trace"]
