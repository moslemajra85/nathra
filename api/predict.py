from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
from urllib.parse import urlparse

from predictor import ModelNotConfigured, ValidationError, metadata, predict


ROOT = Path(__file__).resolve().parent.parent
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self._send_json({"ok": True})

    def do_HEAD(self):
        if self._is_api_request():
            self._send_json(metadata(), include_body=False)
            return

        self._serve_static(include_body=False)

    def do_GET(self):
        if self._is_api_request():
            self._send_json(metadata())
            return

        self._serve_static()

    def do_POST(self):
        if not self._is_api_request():
            self._send_json(
                {"error": "not_found", "message": "Unknown API route."},
                status=404,
            )
            return

        try:
            content_length = int(self.headers.get("content-length", "0"))
            if content_length > 20_000:
                self._send_json(
                    {"error": "payload_too_large", "message": "Request body is too large."},
                    status=413,
                )
                return

            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body or "{}")
            self._send_json(predict(payload))
        except json.JSONDecodeError:
            self._send_json(
                {"error": "invalid_json", "message": "Request body must be valid JSON."},
                status=400,
            )
        except ValidationError as error:
            self._send_json(
                {
                    "error": "validation_error",
                    "message": "Some fields are missing or invalid.",
                    "fields": error.errors,
                },
                status=400,
            )
        except ModelNotConfigured as error:
            self._send_json(
                {
                    "error": "model_not_configured",
                    "message": str(error),
                    "nextStep": "Export model.pkl and scaler.pkl from the notebook into the model/ directory.",
                },
                status=503,
            )
        except Exception as error:
            self._send_json(
                {
                    "error": "prediction_failed",
                    "message": "The model could not generate a prediction.",
                    "detail": str(error),
                },
                status=500,
            )

    def _is_api_request(self):
        return urlparse(self.path).path == "/api/predict"

    def _serve_static(self, include_body=True):
        request_path = urlparse(self.path).path
        relative_path = "index.html" if request_path in ("", "/") else request_path.lstrip("/")
        file_path = (ROOT / relative_path).resolve()

        if ROOT not in file_path.parents and file_path != ROOT:
            self._send_json({"error": "not_found"}, status=404)
            return

        if not file_path.exists() or not file_path.is_file():
            self._send_json({"error": "not_found"}, status=404)
            return

        body = file_path.read_bytes() if include_body else b""
        content_type = STATIC_TYPES.get(file_path.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("cache-control", "public, max-age=0, must-revalidate")
        self.send_header("content-length", str(file_path.stat().st_size))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def _send_json(self, payload, status=200, include_body=True):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)
