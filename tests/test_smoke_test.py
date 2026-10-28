import contextlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import time
import unittest

from scripts.smoke_test import DATASET, MODEL, QUOTE, run


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.scenario = "success"
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.handle_request()

            def do_POST(self):
                self.handle_request()

            def handle_request(self):
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                owner.requests.append((self.path, self.headers.get("Authorization"), payload))
                status = 200
                body = {"status": "ok"}
                if owner.scenario == "auth" and self.path != "/health":
                    status, body = 401, {"detail": "do not print this secret"}
                elif self.path == "/v1/models":
                    body = {"data": [{"id": "wrong" if owner.scenario == "wrong_model" else MODEL}]}
                elif self.path == "/v1/chat/completions":
                    body = {
                        "model": "wrong" if owner.scenario == "wrong_completion_model" else MODEL,
                        "choices": [{
                            "finish_reason": "length" if owner.scenario == "truncated" else "stop",
                            "message": {"content": json.dumps({"dataset_name": DATASET, "quote": QUOTE})},
                        }],
                    }
                    if owner.scenario == "invalid_content":
                        body["choices"][0]["message"]["content"] = "not JSON"
                    if owner.scenario == "invented_evidence":
                        body["choices"][0]["message"]["content"] = json.dumps({"dataset_name": DATASET, "quote": "invented"})
                    if owner.scenario == "unavailable":
                        status, body = 503, {"detail": {"error": {"code": "llm_backend_unavailable"}}}
                    if owner.scenario == "other_503":
                        status, body = 503, {"detail": {"error": {"code": "something_else"}}}
                if owner.scenario == "timeout":
                    time.sleep(0.1)
                encoded = b"not JSON" if owner.scenario == "invalid_json" else json.dumps(body).encode()
                if (owner.scenario, self.path) in {
                    ("empty_health", "/health"), ("empty_models", "/v1/models"),
                    ("empty_completion", "/v1/chat/completions"),
                }:
                    encoded = b""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def check(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            run(self.base_url, **kwargs)

    def test_success_sends_auth_schema_and_non_thinking(self):
        self.check(token="test-secret")
        self.assertEqual([r[0] for r in self.requests], ["/health", "/v1/models", "/v1/chat/completions"])
        self.assertTrue(all(r[1] == "Bearer test-secret" for r in self.requests))
        payload = self.requests[-1][2]
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(payload["max_tokens"], 128)

    def test_unavailable_gateway(self):
        self.scenario = "unavailable"
        self.check(expect_unavailable=True)
        self.assertEqual([r[0] for r in self.requests], ["/health", "/v1/chat/completions"])

    def test_empty_health_response(self):
        self.scenario = "empty_health"
        self.check()
        self.assertEqual(self.requests[-1][0], "/v1/chat/completions")

    def test_failures_identify_stage(self):
        for scenario, stage in [
            ("auth", "/v1/models"), ("wrong_model", "/v1/models"),
            ("wrong_completion_model", "/v1/chat/completions"),
            ("invalid_json", "/health"), ("invalid_content", "/v1/chat/completions"),
            ("truncated", "/v1/chat/completions"), ("invented_evidence", "/v1/chat/completions"),
            ("unavailable", "/v1/chat/completions"),
            ("empty_models", "/v1/models"), ("empty_completion", "/v1/chat/completions"),
        ]:
            with self.subTest(scenario=scenario):
                self.scenario = scenario
                with self.assertRaisesRegex(ValueError, stage) as caught:
                    self.check()
                self.assertNotIn("secret", str(caught.exception))

    def test_unrelated_503_is_not_accepted(self):
        self.scenario = "other_503"
        with self.assertRaisesRegex(ValueError, "not a backend-unavailable"):
            self.check(expect_unavailable=True)

    def test_timeout(self):
        self.scenario = "timeout"
        with self.assertRaisesRegex(ValueError, "/health failed"):
            self.check(timeout=0.02)

    def test_rejects_credentials_in_url(self):
        with self.assertRaisesRegex(ValueError, "must not contain credentials"):
            run("https://user:secret@example.test")


if __name__ == "__main__":
    unittest.main()
