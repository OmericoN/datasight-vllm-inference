"""Check a running vLLM backend or CPU gateway; requires only Python's stdlib."""

import argparse
import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MODEL = "Qwen/Qwen3.8-27B"
DATASET = "DataSight Smoke Dataset"
QUOTE = "We used the DataSight Smoke Dataset for our analysis."


def request_json(base_url, path, token, timeout, payload=None):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    request = Request(base_url + path, data=body, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            return response.status, json.loads(body) if body else None
    except HTTPError as exc:
        # Never print upstream bodies: they can contain prompts or credentials.
        with exc:
            try:
                error = json.load(exc)
            except (ValueError, UnicodeError):
                error = None
        return exc.code, error


def run(base_url, model=MODEL, token="", timeout=120, expect_unavailable=False):
    url = urlsplit(base_url)
    if url.scheme not in {"http", "https"} or not url.netloc:
        raise ValueError("base URL must be an HTTP(S) service root")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("base URL must not contain credentials, query parameters, or fragments")
    base_url = base_url.rstrip("/")
    started = time.monotonic()
    stage = "/health"
    try:
        status, _ = request_json(base_url, stage, token, timeout)
        if status != 200:
            raise ValueError(f"HTTP {status}; expected 200")
        print(f"PASS {stage}")

        if not expect_unavailable:
            stage = "/v1/models"
            status, models = request_json(base_url, stage, token, timeout)
            if status != 200:
                raise ValueError(f"HTTP {status}; expected 200")
            if not isinstance(models, dict) or not isinstance(models.get("data"), list):
                raise ValueError("invalid model listing")
            if not any(
                isinstance(item, dict) and item.get("id") == model
                and item.get("available") is not False
                for item in models["data"]
            ):
                raise ValueError("expected model is absent or unavailable")
            print(f"PASS {stage}: {model}")

        stage = "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": (
                    "Extract the dataset name and copy the entire evidence sentence "
                    f"exactly into quote. Return only JSON. Sentence: {QUOTE}"
                ),
            }],
            "stream": False,
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "smoke_dataset",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "dataset_name": {"type": "string"},
                            "quote": {"type": "string"},
                        },
                        "required": ["dataset_name", "quote"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        status, completion = request_json(base_url, stage, token, timeout, payload)
        if expect_unavailable:
            if status != 503 or not isinstance(completion, dict):
                raise ValueError(f"HTTP {status}; expected gateway 503 backend-unavailable error")
            error = completion.get("detail", {})
            if not isinstance(error, dict) or not isinstance(error.get("error"), dict):
                raise ValueError("invalid gateway error shape")
            if error["error"].get("code") != "llm_backend_unavailable":
                raise ValueError("503 was not a backend-unavailable error")
            print(f"PASS {stage}: expected unavailable backend")
        else:
            if status != 200:
                raise ValueError(f"HTTP {status}; expected 200")
            if not isinstance(completion, dict) or completion.get("model") != model:
                raise ValueError("completion returned the wrong model")
            choices = completion.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("expected one completion choice")
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
                raise ValueError("completion was truncated or did not stop normally")
            message = choice.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError("completion has no text content")
            extracted = json.loads(message["content"])
            if extracted != {"dataset_name": DATASET, "quote": QUOTE}:
                raise ValueError("extracted name or evidence did not match the synthetic source")
            print(f"PASS {stage}: grounded, schema-constrained JSON")
    except (ValueError, UnicodeError, URLError, OSError) as exc:
        # URL errors can echo sensitive proxy URLs; report their type only.
        reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        raise ValueError(f"{stage} failed after {time.monotonic() - started:.2f}s: {reason}") from None
    print(f"PASS smoke test in {time.monotonic() - started:.2f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="service root URL, without /v1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=float, default=120, help="per-request timeout in seconds")
    parser.add_argument("--expect-unavailable", action="store_true")
    args = parser.parse_args()
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be finite and positive")
    try:
        run(args.base_url, args.model, os.environ.get("SMOKE_API_KEY", ""), args.timeout, args.expect_unavailable)
    except ValueError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
