"""Local inference helper for development and internal use.

The stdlib server has no authentication or TLS. Bind it to localhost (the default) or put it
behind a proper gateway before exposing it to a network.
"""

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


MAX_BODY_BYTES = 10 * 1024 * 1024
MAX_RECORDS = 10_000


def serve(
    artifact_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    max_body_bytes: int = MAX_BODY_BYTES,
    max_records: int = MAX_RECORDS,
) -> ThreadingHTTPServer:
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
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send(400, {"error": "invalid Content-Length"})
                return
            if length <= 0:
                self._send(400, {"error": "empty body; send {\"records\": [...]}"})
                return
            if length > max_body_bytes:
                self._send(413, {"error": f"body exceeds {max_body_bytes} bytes"})
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send(400, {"error": f"invalid JSON: {exc}"})
                return
            records = payload.get("records") if isinstance(payload, dict) else None
            if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
                self._send(400, {"error": "body must be {\"records\": [{...}, ...]}"})
                return
            if len(records) > max_records:
                self._send(413, {"error": f"at most {max_records} records per request"})
                return
            try:
                preds = predict_records(artifact, records)
            except Exception as exc:
                self._send(422, {"error": str(exc)})
                return
            self._send(200, {"predictions": preds})

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, port), Handler)
