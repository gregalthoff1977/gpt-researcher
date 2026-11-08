"""Railway entry point: initialize the durable volume, drop root, require auth."""
import base64
import hmac
import os
from pathlib import Path
from urllib.parse import urlsplit


class ResearchAuth:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        if kind not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        if kind == "http" and scope.get("path") == "/__health" and scope.get("method") in ("GET", "HEAD"):
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain"), (b"cache-control", b"no-store"), (b"x-research-engine", b"adaptive-consumer-v1")]})
            return await send({"type": "http.response.body", "body": b"" if scope["method"] == "HEAD" else b"ok"})
        password = os.getenv("RESEARCH_PASSWORD", "")
        ready = len(password) >= 16
        credentials = base64.b64encode((os.getenv("RESEARCH_USERNAME", "greg") + ":" + password).encode())
        headers = dict(scope.get("headers", []))
        authorized = ready and hmac.compare_digest(headers.get(b"authorization", b""), b"Basic " + credentials)
        origin = headers.get(b"origin", b"").decode("utf-8", errors="replace")
        # Basic auth is ambient in browsers: reject cross-origin mutations/WS.
        if origin and (kind == "websocket" or scope.get("method") not in ("GET", "HEAD")):
            authorized = authorized and urlsplit(origin).netloc == headers.get(b"host", b"").decode()
        if authorized:
            return await self.app(scope, receive, send)
        if kind == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        response_headers = [(b"content-type", b"text/plain"), (b"cache-control", b"no-store")]
        if ready:
            response_headers.append((b"www-authenticate", b'Basic realm="GPT Researcher", charset="UTF-8"'))
        await send({"type": "http.response.start", "status": 401 if ready else 503, "headers": response_headers})
        await send({"type": "http.response.body", "body": b"Authentication required" if ready else b"Set RESEARCH_PASSWORD to at least 16 characters in Railway Variables."})


def prepare_storage(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    state, output = root / "research", root / "outputs"
    state.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    os.environ["RESEARCH_STATE_DIR"] = str(state)
    os.environ["REPORT_STORE_PATH"] = str(root / "reports.json")
    target = Path("outputs")
    if not target.is_symlink():
        if target.exists():
            target.rmdir()  # Fail instead of deleting or hiding nonempty local output.
        target.symlink_to(output.resolve(), target_is_directory=True)
    elif target.resolve() != output.resolve():
        raise RuntimeError("Existing outputs symlink does not point at the configured durable volume")
    return root, state, output


def main():
    paths = prepare_storage(os.getenv("RESEARCH_DATA_DIR", "/data"))
    if os.geteuid() == 0:
        import pwd
        account = pwd.getpwnam("gpt-researcher")
        for path in paths:
            os.chown(path, account.pw_uid, account.pw_gid)
        os.setgroups([])
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
    from main import app
    import uvicorn
    uvicorn.run(ResearchAuth(app), host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), workers=1)


if __name__ == "__main__":
    main()
