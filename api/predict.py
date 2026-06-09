from http.server import BaseHTTPRequestHandler
import json

from predictor import ModelNotConfigured, ValidationError, metadata, predict


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self._send_json({"ok": True})

    def do_GET(self):
        self._send_json(metadata())

    def do_POST(self):
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
