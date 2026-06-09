from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

from predictor import ModelNotConfigured, ValidationError, metadata, predict


ROOT = Path(__file__).parent


class LocalHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_OPTIONS(self):
        if self._is_api_request():
            self._send_json({"ok": True})
        else:
            super().do_OPTIONS()

    def do_GET(self):
        if self._is_api_request():
            self._send_json(metadata())
            return

        path = urlparse(self.path).path
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        if not self._is_api_request():
            self.send_error(404)
            return

        try:
            content_length = int(self.headers.get("content-length", "0"))
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

    def _is_api_request(self):
        return urlparse(self.path).path == "/api/predict"

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    port = 3000
    server = ThreadingHTTPServer(("127.0.0.1", port), LocalHandler)
    print(f"Serving Nathra at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
