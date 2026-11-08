"""Container readiness and exact Knowledge contract checks, with no live research."""
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    real_connect = socket.socket.connect
    def offline(self, address, *args, **kwargs):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise RuntimeError("Readiness verification must not reach external APIs")
        return real_connect(self, address, *args, **kwargs)
    socket.socket.connect = offline
    from fastapi.testclient import TestClient
    from backend.server.production import ResearchAuth
    from gpt_researcher.skills.research_state import ResearchState, StateStore
    from gpt_researcher.skills.knowledge_export import export_knowledge, validate_export
    password = os.environ.get("RESEARCH_PASSWORD", "")
    assert len(password) >= 16, "RESEARCH_PASSWORD must contain at least 16 characters"
    with tempfile.TemporaryDirectory() as directory:
        previous = os.getcwd()
        os.chdir(directory)
        os.environ["REPORT_STORE_PATH"] = str(Path(directory) / "reports.json")
        try:
            from backend.server.app import app
            with TestClient(ResearchAuth(app), base_url="https://testserver") as client:
                assert client.get("/__health").status_code == 200
                assert client.get("/").status_code == 401
                response = client.get("/", auth=(os.getenv("RESEARCH_USERNAME", "greg"), password))
                assert response.status_code == 200 and 'id="task"' in response.text
                assert client.get("/site/scripts.js", auth=(os.getenv("RESEARCH_USERNAME", "greg"), password)).status_code == 200
            store = StateStore(Path(directory) / "state")
            state = store.open("Readiness fixture, not a research assignment", {"fixture": True})
            state.status = "partial"
            state.stopping_reason = "Readiness contract check; no research performed"
            store.save(state)
            validate_export(export_knowledge(store.load(state.run_id)))
            store.close()
        finally:
            os.chdir(previous)
    print("ADAPTIVE_READINESS_PASS " + json.dumps({"interface": True, "static_assets": True,
        "authentication": True, "sqlite_roundtrip": True, "knowledge_schema": "1.2", "live_research_performed": False}), flush=True)


if __name__ == "__main__":
    main()
