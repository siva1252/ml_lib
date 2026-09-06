"""Local inference helper. Cloud/Docker packaging can wrap this later."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from mlverdict.production.artifact import ModelArtifact


def predict_records(artifact: ModelArtifact, records: list[dict[str, Any]]) -> list[Any]:
    preds = artifact.predict(records)
    return [ _jsonable(p) for p in preds ]


def _jsonable(value: Any) -> Any:
    try:
        return value.item()
    except Exception:
        return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def serve(artifact_path: str | Path, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    """Stdlib JSON prediction API. POST /predict with {\"records\": [...]}."""
    artifact = ModelArtifact.load(artifact_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
            return

        def do_GET(self) -> None:  # pragma: no cover - trivial health
            if self.path in {"/", "/health"}:
                self._send(200, {"status": "ok"})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                records = payload["records"]
                preds = predict_records(artifact, records)
                self._send(200, {"predictions": preds})
            except Exception as exc:
                self._send(400, {"error": str(exc)})

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, port), Handler)
