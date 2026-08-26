"""Hermetic tests for the Tavily-backed Web Search Function."""

from __future__ import annotations

import json
import http.client
import os
import socket
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))
import handler  # noqa: E402


VALID_ENV = {
    "TAVILY_API_KEY": "tvly-test-api-key",
    "INGRESS_BEARER_TOKEN": "test-ingress-token",
}


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.payload if size < 0 else self.payload[:size]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


class FakeOpener:
    def __init__(self, response: FakeResponse | BaseException) -> None:
        self.response = response
        self.requests: list[tuple[object, float]] = []

    def open(self, request: object, timeout: float) -> FakeResponse:
        self.requests.append((request, timeout))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class ReadErrorResponse(FakeResponse):
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        raise self.error


class WebSearchFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = mock.patch.dict(os.environ, VALID_ENV, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def provider_payload(self, *items: dict[str, object]) -> dict[str, object]:
        return {"results": list(items)}

    def search_with_response(self, payload: object) -> tuple[list[handler.SearchResult], FakeOpener, FakeResponse]:
        response = FakeResponse(payload)
        opener = FakeOpener(response)
        with mock.patch.object(handler.urllib.request, "build_opener", return_value=opener) as build:
            results = handler.search_tavily(
                handler.SearchRequest("current news", (), "medium", None),
                api_key=VALID_ENV["TAVILY_API_KEY"],
                timeout_seconds=4.0,
            )
        self.assertIsInstance(build.call_args.args[0], handler.RejectRedirectHandler)
        return results, opener, response

    def test_parses_valid_strict_request_and_returns_normalized_results(self) -> None:
        response = FakeResponse(
            self.provider_payload(
                {
                    "title": "A result",
                    "url": "https://Search.Test:443/story#fragment",
                    "content": "A short description",
                    "published_date": "provider value is intentionally ignored",
                }
            )
        )
        opener = FakeOpener(response)
        with mock.patch.object(handler.urllib.request, "build_opener", return_value=opener):
            payload = handler.handler(b'{"query":" current news ","search_context_size":"medium"}', object())

        self.assertEqual(
            payload,
            {
                "results": [
                    {
                        "title": "A result",
                        "url": "https://search.test/story",
                        "snippet": "A short description",
                        "published_at": None,
                    }
                ]
            },
        )
        request, timeout = opener.requests[0]
        self.assertEqual(timeout, handler.PROVIDER_TIMEOUT_SECONDS)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.full_url, "https://api.tavily.com/search")
        self.assertEqual(request.get_header("Authorization"), "Bearer tvly-test-api-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(
            json.loads(request.data),
            {
                "query": "current news",
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
        )
        self.assertEqual(response.read_sizes, [handler.MAX_PROVIDER_RESPONSE_BYTES + 1])

    def test_fc3_http_trigger_unwraps_request_body_and_wraps_json_response(self) -> None:
        response = FakeResponse(self.provider_payload())
        opener = FakeOpener(response)
        event = json.dumps(
            {
                "version": "v1",
                "rawPath": "/invoke",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer test-ingress-token",
                },
                "queryParameters": {},
                "body": '{"query":"current news"}',
                "isBase64Encoded": False,
                "requestContext": {
                    "http": {"method": "POST", "path": "/invoke"}
                },
            }
        ).encode()

        with mock.patch.object(handler.urllib.request, "build_opener", return_value=opener):
            payload = handler.handler(event, object())

        self.assertEqual(payload["statusCode"], 200)
        self.assertEqual(payload["headers"], {"Content-Type": "application/json"})
        self.assertIs(payload["isBase64Encoded"], False)
        self.assertEqual(json.loads(payload["body"]), {"results": []})
        self.assertEqual(len(opener.requests), 1)

    def test_fc3_http_trigger_rejects_missing_or_wrong_ingress_bearer(self) -> None:
        for authorization in (None, "Bearer wrong-token"):
            with self.subTest(authorization=authorization):
                headers = {"Content-Type": "application/json"}
                if authorization is not None:
                    headers["Authorization"] = authorization
                event = json.dumps(
                    {
                        "version": "v1",
                        "rawPath": "/invoke",
                        "headers": headers,
                        "queryParameters": {},
                        "body": '{"query":"current news"}',
                        "isBase64Encoded": False,
                        "requestContext": {
                            "http": {"method": "POST", "path": "/invoke"}
                        },
                    }
                ).encode()
                opener = FakeOpener(FakeResponse(self.provider_payload()))
                with mock.patch.object(
                    handler.urllib.request, "build_opener", return_value=opener
                ):
                    payload = handler.handler(event, object())

                self.assertEqual(payload["statusCode"], 401)
                self.assertEqual(payload["headers"], {"Content-Type": "application/json"})
                self.assertEqual(
                    json.loads(payload["body"]),
                    {
                        "ok": False,
                        "error": {
                            "type": "unauthorized",
                            "message": "The web search request is unauthorized.",
                            "retryable": False,
                        },
                        "stdout": "",
                        "stderr": "",
                    },
                )
                self.assertEqual(opener.requests, [])

    def test_rejects_missing_and_oversized_queries(self) -> None:
        missing = handler.handler(b"{}", object())
        oversized = handler.handler(
            json.dumps({"query": "x" * (handler.MAX_QUERY_BYTES + 1)}).encode(),
            object(),
        )

        self.assertEqual(missing["ok"], False)
        self.assertEqual(oversized["ok"], False)
        self.assertEqual(missing["error"]["type"], "invalid_request")
        self.assertEqual(oversized["error"]["type"], "invalid_request")

    def test_rejects_unencodable_input_and_discards_unencodable_provider_text(self) -> None:
        malformed_query = handler.handler(b'{"query":"\\ud800"}', object())
        self.assertEqual(malformed_query["error"]["type"], "invalid_request")

        results, _, _ = self.search_with_response(
            self.provider_payload(
                {"title": chr(0xD800), "url": "https://result.test/bad", "content": "bad"},
                {"title": "good", "url": "https://result.test/good", "content": "good"},
            )
        )
        self.assertEqual([result.url for result in results], ["https://result.test/good"])

    def test_rejects_unknown_and_malformed_trusted_options(self) -> None:
        for value in (
            {"query": "news", "surprise": True},
            {"query": "news", "allowed_domains": ["https://allowed.test"]},
            {"query": "news", "search_context_size": "huge"},
            {"query": "news", "search_context_size": "small"},
            {"query": "news", "search_context_size": "large"},
            {"query": "news", "user_location": {"type": "precise"}},
            {"query": "news", "user_location": {"country": 42}},
        ):
            with self.subTest(value=value):
                payload = handler.handler(json.dumps(value).encode(), object())
                self.assertEqual(payload["ok"], False)
                self.assertEqual(payload["error"]["type"], "invalid_request")

    def test_context_sizes_map_to_tavily_depth_and_result_count(self) -> None:
        expected = {
            "low": ("fast", 3),
            "medium": ("basic", 5),
            "high": ("advanced", 10),
        }
        for size, (search_depth, max_results) in expected.items():
            with self.subTest(size=size):
                response = FakeResponse(self.provider_payload())
                opener = FakeOpener(response)
                with mock.patch.object(handler.urllib.request, "build_opener", return_value=opener):
                    payload = handler.handler(
                        json.dumps({"query": "news", "search_context_size": size}).encode(),
                        object(),
                    )
                self.assertEqual(payload, {"results": []})
                request_body = json.loads(opener.requests[0][0].data)
                self.assertEqual(request_body["search_depth"], search_depth)
                self.assertEqual(request_body["max_results"], max_results)

    def test_provider_failures_are_sanitized_application_errors(self) -> None:
        cases: list[tuple[str, object, str]] = [
            ("timeout", socket.timeout(), "provider_timeout"),
            (
                "non_2xx",
                urllib.error.HTTPError("https://provider.invalid/path", 503, "failure", {}, None),
                "provider_failure",
            ),
            ("invalid_json", b"not-json", "provider_response_invalid"),
        ]
        for label, result, expected_type in cases:
            with self.subTest(label=label):
                if isinstance(result, bytes):
                    response = FakeResponse({})
                    response.payload = result
                    opener = FakeOpener(response)
                else:
                    opener = FakeOpener(result)
                with mock.patch.object(handler.urllib.request, "build_opener", return_value=opener):
                    payload = handler.handler(b'{"query":"news"}', object())
                self.assertEqual(payload["ok"], False)
                self.assertEqual(payload["error"]["type"], expected_type)
                self.assertNotIn("provider.invalid", json.dumps(payload))
                self.assertNotIn(VALID_ENV["TAVILY_API_KEY"], json.dumps(payload))

    def test_handler_sanitizes_malformed_and_truncated_http_transport(self) -> None:
        cases: list[tuple[str, FakeResponse | BaseException, str]] = [
            (
                "incomplete_read",
                ReadErrorResponse(http.client.IncompleteRead(b"provider-body-with-secret", 64)),
                "provider_response_invalid",
            ),
            (
                "bad_status_line",
                http.client.BadStatusLine("provider status with tvly-test-api-key"),
                "provider_response_invalid",
            ),
        ]
        for label, response, expected_type in cases:
            with self.subTest(label=label):
                opener = FakeOpener(response)
                with mock.patch.object(handler.urllib.request, "build_opener", return_value=opener):
                    payload = handler.handler(b'{"query":"news"}', object())
                serialized = json.dumps(payload)
                self.assertEqual(payload["error"]["type"], expected_type)
                self.assertNotIn("provider-body-with-secret", serialized)
                self.assertNotIn("tvly-test-api-key", serialized)

    def test_handler_rejects_oversized_provider_body(self) -> None:
        response = FakeResponse("x" * (handler.MAX_PROVIDER_RESPONSE_BYTES + 1))
        opener = FakeOpener(response)
        with mock.patch.object(handler.urllib.request, "build_opener", return_value=opener):
            payload = handler.handler(b'{"query":"news"}', object())

        self.assertEqual(payload["error"]["type"], "provider_response_invalid")
        self.assertEqual(response.read_sizes, [handler.MAX_PROVIDER_RESPONSE_BYTES + 1])

    def test_handler_rejects_redirects_as_application_failures(self) -> None:
        opener = FakeOpener(handler.RedirectRejected("provider redirect"))
        with mock.patch.object(handler.urllib.request, "build_opener", return_value=opener):
            payload = handler.handler(b'{"query":"news"}', object())

        self.assertEqual(payload["error"]["type"], "provider_failure")

    def test_handler_rejects_missing_or_malformed_tavily_key(self) -> None:
        for api_key in (None, "", "bad\nkey"):
            environment = {**VALID_ENV}
            if api_key is None:
                environment.pop("TAVILY_API_KEY")
            else:
                environment["TAVILY_API_KEY"] = api_key
            with self.subTest(api_key=api_key), mock.patch.dict(
                os.environ, environment, clear=True
            ), mock.patch.object(handler.urllib.request, "build_opener") as build:
                payload = handler.handler(b'{"query":"news"}', object())
                self.assertEqual(payload["error"]["type"], "configuration_error")
                build.assert_not_called()

    def test_rejects_non_https_result_urls(self) -> None:
        results, _, _ = self.search_with_response(
            self.provider_payload(
                {"title": "bad", "url": "http://result.test/story", "content": "bad"},
                {"title": "good", "url": "https://result.test/story", "content": "good"},
            )
        )
        self.assertEqual([result.url for result in results], ["https://result.test/story"])

    def test_rejects_all_redirects(self) -> None:
        redirect_handler = handler.RejectRedirectHandler()
        with self.assertRaises(handler.RedirectRejected):
            redirect_handler.redirect_request(None, None, 302, "Found", {}, "https://elsewhere.example")
        with self.assertRaises(handler.RedirectRejected):
            redirect_handler.http_error_302(None, None, 302, "Found", {})

    def test_filters_domains_at_dns_label_boundaries(self) -> None:
        response = FakeResponse(
            self.provider_payload(
                {"title": "root", "url": "https://example.com/a", "content": "root"},
                {"title": "subdomain", "url": "https://news.example.com/a", "content": "sub"},
                {"title": "bad prefix", "url": "https://badexample.com/a", "content": "bad"},
                {"title": "bad suffix", "url": "https://example.com.bad/a", "content": "bad"},
            )
        )
        opener = FakeOpener(response)
        with mock.patch.object(handler.urllib.request, "build_opener", return_value=opener):
            payload = handler.handler(
                b'{"query":"news","allowed_domains":["example.com"]}', object()
            )

        self.assertEqual(
            [result["url"] for result in payload["results"]],
            ["https://example.com/a", "https://news.example.com/a"],
        )
        self.assertEqual(
            json.loads(opener.requests[0][0].data)["include_domains"],
            ["example.com"],
        )

    def test_deduplicates_normalized_https_urls_stably(self) -> None:
        results, _, _ = self.search_with_response(
            self.provider_payload(
                {"title": "first", "url": "https://RESULT.test:443/a#one", "content": "first"},
                {"title": "duplicate", "url": "https://result.test/a#two", "content": "duplicate"},
                {"title": "next", "url": "https://result.test/b", "content": "next"},
            )
        )
        self.assertEqual([result.url for result in results], ["https://result.test/a", "https://result.test/b"])
        self.assertEqual(results[0].title, "first")

    def test_caps_utf8_output_without_breaking_schema_or_encoding(self) -> None:
        long_text = "你好" * (handler.MAX_SNIPPET_BYTES + 100)
        response = FakeResponse(
            self.provider_payload(
                *[
                    {
                        "title": "标题" * 1000,
                        "url": f"https://result.test/{index}",
                        "content": long_text,
                    }
                    # Seven rows stay under the provider's 512 KiB body cap
                    # while exceeding the function's 32 KiB serialized cap.
                    for index in range(7)
                ]
            )
        )
        opener = FakeOpener(response)
        with mock.patch.object(handler.urllib.request, "build_opener", return_value=opener):
            payload = handler.handler(b'{"query":"news","search_context_size":"high"}', object())
        encoded = handler.serialize_json(payload)

        self.assertLessEqual(len(encoded), handler.MAX_SERIALIZED_OUTPUT_BYTES)
        self.assertEqual(json.loads(encoded), payload)
        self.assertIn("results", payload)
        self.assertTrue(payload["truncated"])
        for result in payload["results"]:
            self.assertIsNone(result["published_at"])


class WebSearchHelperTests(unittest.TestCase):
    def test_success_payload_caps_utf8_without_breaking_schema_or_encoding(self) -> None:
        long_text = "你好" * (handler.MAX_SNIPPET_BYTES + 100)
        results = [
            handler.SearchResult("标题" * 1000, f"https://result.test/{index}", long_text, None)
            for index in range(10)
        ]
        payload = handler.success_payload(results)
        encoded = handler.serialize_json(payload)

        self.assertLessEqual(len(encoded), handler.MAX_SERIALIZED_OUTPUT_BYTES)
        self.assertEqual(json.loads(encoded), payload)
        self.assertIn("results", payload)
        self.assertTrue(payload["truncated"])
        for result in payload["results"]:
            self.assertIsNone(result["published_at"])


if __name__ == "__main__":
    unittest.main()
