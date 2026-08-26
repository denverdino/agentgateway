"""Alibaba Cloud Function Compute Web Search adapter for Tavily.

This module deliberately uses only the Python standard library.  The gateway
supplies the trusted Web Search options; model-generated input never selects a
network destination, credential, service identifier, or timeout.
"""

from __future__ import annotations

import json
import http.client
import hmac
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Mapping


# Resource caps are intentionally small enough for a Function Compute HTTP
# response and are enforced before JSON output is returned to the gateway.
MAX_EVENT_BYTES = 16 * 1024
MAX_HTTP_TRIGGER_EVENT_BYTES = 32 * 1024
MAX_INGRESS_BEARER_TOKEN_BYTES = 4096
MAX_QUERY_BYTES = 4 * 1024
MAX_ALLOWED_DOMAINS = 50
MAX_PROVIDER_RESPONSE_BYTES = 512 * 1024
MAX_PROVIDER_RESULT_ITEMS = 100
MAX_TITLE_BYTES = 1024
MAX_SNIPPET_BYTES = 4096
MAX_RESULT_URL_BYTES = 4096
MAX_SERIALIZED_OUTPUT_BYTES = 32 * 1024

PROVIDER_TIMEOUT_SECONDS = 10.0
MAX_PROVIDER_TIMEOUT_SECONDS = 30.0
CONTEXT_TOP_K = {"low": 3, "medium": 5, "high": 10}
CONTEXT_SEARCH_DEPTH = {"low": "fast", "medium": "basic", "high": "advanced"}
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_REQUEST_FIELDS = frozenset(
    {"query", "allowed_domains", "search_context_size", "user_location"}
)
_LOCATION_FIELDS = frozenset({"type", "country", "region", "city", "timezone"})


class WebSearchError(Exception):
    """Base class whose details are never returned to a caller."""


class RequestValidationError(WebSearchError):
    pass


class ConfigurationError(WebSearchError):
    pass


class ProviderTimeout(WebSearchError):
    pass


class ProviderFailure(WebSearchError):
    pass


class ProviderResponseInvalid(WebSearchError):
    pass


class RedirectRejected(ProviderFailure):
    pass


class RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject every redirect instead of allowing urllib to follow one."""

    def redirect_request(
        self,
        req: object,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> object:
        raise RedirectRejected()

    def http_error_301(
        self, req: object, fp: object, code: int, msg: str, headers: object
    ) -> object:
        raise RedirectRejected()

    def http_error_302(
        self, req: object, fp: object, code: int, msg: str, headers: object
    ) -> object:
        raise RedirectRejected()

    def http_error_303(
        self, req: object, fp: object, code: int, msg: str, headers: object
    ) -> object:
        raise RedirectRejected()

    def http_error_307(
        self, req: object, fp: object, code: int, msg: str, headers: object
    ) -> object:
        raise RedirectRejected()

    def http_error_308(
        self, req: object, fp: object, code: int, msg: str, headers: object
    ) -> object:
        raise RedirectRejected()


@dataclass(frozen=True)
class SearchRequest:
    query: str
    allowed_domains: tuple[str, ...]
    search_context_size: str
    # The function validates this trusted option but does not send location
    # data to Tavily because its country filter is not equivalent to the
    # Responses API's complete approximate-location object.
    user_location: Mapping[str, str | None] | None


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    published_at: None = None


def handler(event: bytes, context: object) -> dict[str, object]:
    """Run one trusted Web Search request.

    Expected validation and provider failures are application-level values.  A
    successful value intentionally has no ``ok`` member: AgentGateway's typed
    Web Search backend adds ``ok: true`` after validating this response shape.
    """

    del context
    is_http_trigger = is_fc3_http_trigger_event(event)
    if is_http_trigger and not authenticate_fc3_http_trigger(event):
        return render_handler_response(
            application_error(
                "unauthorized", "The web search request is unauthorized.", False
            ),
            True,
            status_code=401,
        )
    try:
        request_event = fc3_http_trigger_body(event) if is_http_trigger else event
    except RequestValidationError:
        return render_handler_response(
            application_error(
                "invalid_request", "The web search request is invalid.", False
            ),
            is_http_trigger,
        )
    try:
        request = parse_search_request(request_event)
    except RequestValidationError:
        return render_handler_response(
            application_error(
                "invalid_request", "The web search request is invalid.", False
            ),
            is_http_trigger,
        )

    try:
        api_key = load_configuration()
    except ConfigurationError:
        return render_handler_response(
            application_error(
                "configuration_error", "The web search service is unavailable.", False
            ),
            is_http_trigger,
        )

    try:
        results = search_tavily(
            request,
            api_key=api_key,
            timeout_seconds=PROVIDER_TIMEOUT_SECONDS,
        )
    except ProviderTimeout:
        payload = application_error(
            "provider_timeout", "The web search provider timed out.", True
        )
    except ProviderResponseInvalid:
        payload = application_error(
            "provider_response_invalid",
            "The web search provider returned an invalid response.",
            True,
        )
    except (RedirectRejected, ProviderFailure):
        payload = application_error(
            "provider_failure", "The web search provider request failed.", True
        )
    except ConfigurationError:
        # This branch also covers a direct search_tavily caller supplying an
        # invalid operator configuration; do not reveal the rejected value.
        payload = application_error(
            "configuration_error", "The web search service is unavailable.", False
        )
    else:
        payload = success_payload(results)

    return render_handler_response(payload, is_http_trigger)


def is_fc3_http_trigger_event(event: object) -> bool:
    if not isinstance(event, bytes) or len(event) > MAX_HTTP_TRIGGER_EVENT_BYTES:
        return False
    try:
        value = json.loads(event)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("version") == "v1"
        and "requestContext" in value
        and "body" in value
    )


def fc3_http_trigger_body(event: bytes) -> bytes:
    try:
        value = json.loads(event)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RequestValidationError() from None
    request_context = value.get("requestContext") if isinstance(value, dict) else None
    http = request_context.get("http") if isinstance(request_context, dict) else None
    body = value.get("body") if isinstance(value, dict) else None
    if (
        not isinstance(http, dict)
        or http.get("method") != "POST"
        or not isinstance(body, str)
        or value.get("isBase64Encoded") is not False
        or not is_utf8_encodable(body)
    ):
        raise RequestValidationError()
    encoded = body.encode("utf-8")
    if len(encoded) > MAX_EVENT_BYTES:
        raise RequestValidationError()
    return encoded


def authenticate_fc3_http_trigger(event: bytes) -> bool:
    expected = os.environ.get("INGRESS_BEARER_TOKEN")
    if (
        not isinstance(expected, str)
        or not expected
        or not is_utf8_encodable(expected)
        or _utf8_len(expected) > MAX_INGRESS_BEARER_TOKEN_BYTES
    ):
        return False
    try:
        value = json.loads(event)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    headers = value.get("headers") if isinstance(value, dict) else None
    if not isinstance(headers, dict):
        return False
    authorization = next(
        (
            header_value
            for header_name, header_value in headers.items()
            if isinstance(header_name, str)
            and header_name.casefold() == "authorization"
            and isinstance(header_value, str)
        ),
        None,
    )
    if authorization is None:
        return False
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return False
    return hmac.compare_digest(authorization[len(prefix) :], expected)


def render_handler_response(
    payload: dict[str, object], is_http_trigger: bool, status_code: int = 200
) -> dict[str, object]:
    if not is_http_trigger:
        return payload
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "isBase64Encoded": False,
        "body": serialize_json(payload).decode("utf-8"),
    }


def parse_search_request(event: bytes) -> SearchRequest:
    """Decode and strictly validate the flattened trusted FC request."""

    if not isinstance(event, bytes) or len(event) > MAX_EVENT_BYTES:
        raise RequestValidationError()
    try:
        value = json.loads(event)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RequestValidationError() from None
    if not isinstance(value, dict) or set(value) - _REQUEST_FIELDS:
        raise RequestValidationError()

    query = value.get("query")
    if not isinstance(query, str):
        raise RequestValidationError()
    query = query.strip()
    if not query or not is_utf8_encodable(query) or _utf8_len(query) > MAX_QUERY_BYTES:
        raise RequestValidationError()

    allowed_domains = parse_allowed_domains(value.get("allowed_domains"))
    context_size = value.get("search_context_size", "medium")
    if not isinstance(context_size, str) or context_size not in CONTEXT_TOP_K:
        raise RequestValidationError()

    user_location = parse_user_location(value.get("user_location"))
    return SearchRequest(query, allowed_domains, context_size, user_location)


def parse_allowed_domains(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > MAX_ALLOWED_DOMAINS:
        raise RequestValidationError()

    normalized: list[str] = []
    seen: set[str] = set()
    for domain in value:
        if not isinstance(domain, str):
            raise RequestValidationError()
        normalized_domain = normalize_domain(domain)
        if normalized_domain not in seen:
            seen.add(normalized_domain)
            normalized.append(normalized_domain)
    return tuple(normalized)


def parse_user_location(value: object) -> Mapping[str, str | None] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - _LOCATION_FIELDS:
        raise RequestValidationError()

    kind = value.get("type", "approximate")
    if kind != "approximate":
        raise RequestValidationError()
    normalized: dict[str, str | None] = {"type": "approximate"}
    for field in ("country", "region", "city", "timezone"):
        if field not in value:
            continue
        field_value = value[field]
        if field_value is not None and (
            not isinstance(field_value, str)
            or not is_utf8_encodable(field_value)
            or _utf8_len(field_value) > 256
        ):
            raise RequestValidationError()
        normalized[field] = field_value
    return normalized


def load_configuration() -> str:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key or not is_utf8_encodable(api_key) or _has_control_characters(api_key):
        raise ConfigurationError()
    return api_key


def search_tavily(
    request: SearchRequest,
    *,
    api_key: str,
    timeout_seconds: float,
) -> list[SearchResult]:
    """Call the fixed Tavily endpoint and normalize its bounded response."""

    if (
        not isinstance(api_key, str)
        or not api_key
        or not is_utf8_encodable(api_key)
        or _has_control_characters(api_key)
    ):
        raise ConfigurationError()
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
        or timeout_seconds > MAX_PROVIDER_TIMEOUT_SECONDS
    ):
        raise ConfigurationError()

    provider_request: dict[str, object] = {
        "query": request.query,
        "search_depth": CONTEXT_SEARCH_DEPTH[request.search_context_size],
        "max_results": CONTEXT_TOP_K[request.search_context_size],
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }
    if request.allowed_domains:
        provider_request["include_domains"] = list(request.allowed_domains)
    body = json.dumps(
        provider_request,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    outbound = urllib.request.Request(
        TAVILY_SEARCH_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    opener = urllib.request.build_opener(RejectRedirectHandler())
    try:
        with opener.open(outbound, timeout=float(timeout_seconds)) as response:
            raw_response = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
    except RedirectRejected:
        raise
    except (socket.timeout, TimeoutError):
        raise ProviderTimeout() from None
    except urllib.error.HTTPError:
        raise ProviderFailure() from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            raise ProviderTimeout() from None
        raise ProviderFailure() from None
    except http.client.HTTPException:
        raise ProviderResponseInvalid() from None
    except (OSError, ValueError):
        raise ProviderFailure() from None

    if len(raw_response) > MAX_PROVIDER_RESPONSE_BYTES:
        raise ProviderResponseInvalid()
    try:
        payload = json.loads(raw_response)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ProviderResponseInvalid() from None

    return filter_and_normalize_results(parse_provider_results(payload), request)


def parse_provider_results(payload: object) -> list[SearchResult]:
    if not isinstance(payload, dict):
        raise ProviderResponseInvalid()
    items = payload.get("results")
    if not isinstance(items, list):
        raise ProviderResponseInvalid()

    results: list[SearchResult] = []
    for item in items[:MAX_PROVIDER_RESULT_ITEMS]:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        link = item.get("url")
        snippet = item.get("content")
        if (
            not isinstance(title, str)
            or not is_utf8_encodable(title)
            or not isinstance(link, str)
            or not isinstance(snippet, str)
            or not is_utf8_encodable(snippet)
        ):
            continue
        url = normalize_result_url(link)
        if url is None:
            continue
        results.append(SearchResult(title=title, url=url, snippet=snippet))
    return results


def filter_and_normalize_results(
    results: Iterable[SearchResult], request: SearchRequest
) -> list[SearchResult]:
    """Apply DNS-boundary filters and stable normalized-URL de-duplication."""

    accepted: list[SearchResult] = []
    seen_urls: set[str] = set()
    for result in results:
        url = normalize_result_url(result.url)
        if url is None:
            continue
        host = urllib.parse.urlsplit(url).hostname
        if host is None:
            continue
        if request.allowed_domains and not any(
            domain_matches(host, domain) for domain in request.allowed_domains
        ):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        accepted.append(
            SearchResult(
                title=result.title,
                url=url,
                snippet=result.snippet,
                # Provider timestamps are not part of the gateway's normalized
                # first-release contract, so never forward them.
                published_at=None,
            )
        )
        if len(accepted) == CONTEXT_TOP_K[request.search_context_size]:
            break
    return accepted


def normalize_result_url(value: str) -> str | None:
    """Normalize a safe HTTPS result URL, returning ``None`` for unsafe URLs."""

    if (
        not isinstance(value, str)
        or not value
        or not is_utf8_encodable(value)
        or _utf8_len(value) > MAX_RESULT_URL_BYTES
    ):
        return None
    if _has_control_characters(value):
        return None
    try:
        parts = urllib.parse.urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    if (
        parts.scheme.lower() != "https"
        or not parts.netloc
        or "@" in parts.netloc
        or port not in (None, 443)
        or parts.hostname is None
    ):
        return None
    try:
        host = normalize_domain(parts.hostname)
    except RequestValidationError:
        return None
    path = parts.path or "/"
    normalized = f"https://{host}{path}"
    if parts.query:
        normalized = f"{normalized}?{parts.query}"
    return normalized


def normalize_domain(value: str) -> str:
    """Lowercase an optional FQDN dot and require conventional DNS labels."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise RequestValidationError()
    if value.endswith("."):
        value = value[:-1]
        if value.endswith("."):
            raise RequestValidationError()
    value = value.lower()
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        raise RequestValidationError() from None
    if not value or len(value) > 253 or _has_control_characters(value):
        raise RequestValidationError()
    labels = value.split(".")
    for label in labels:
        if not label or len(label) > 63 or label[0] == "-" or label[-1] == "-":
            raise RequestValidationError()
        if any(not ("a" <= char <= "z" or "0" <= char <= "9" or char == "-") for char in label):
            raise RequestValidationError()
    return value


def domain_matches(host: str, allowed_domain: str) -> bool:
    """Match the exact domain or a subdomain at a DNS label boundary."""

    return host == allowed_domain or host.endswith("." + allowed_domain)


def success_payload(results: Iterable[SearchResult]) -> dict[str, object]:
    """Bound output deterministically while retaining the typed result schema."""

    encoded_results: list[dict[str, object]] = []
    text_was_truncated = False
    for result in results:
        title, title_truncated = truncate_utf8(result.title, MAX_TITLE_BYTES)
        snippet, snippet_truncated = truncate_utf8(result.snippet, MAX_SNIPPET_BYTES)
        url, url_truncated = truncate_utf8(result.url, MAX_RESULT_URL_BYTES)
        text_was_truncated = text_was_truncated or title_truncated or snippet_truncated or url_truncated
        encoded_results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "published_at": None,
            }
        )

    included: list[dict[str, object]] = []
    output_was_truncated = text_was_truncated
    for result in encoded_results:
        candidate = [*included, result]
        candidate_payload = _success_payload(candidate, output_was_truncated)
        if len(serialize_json(candidate_payload)) > MAX_SERIALIZED_OUTPUT_BYTES:
            output_was_truncated = True
            break
        included = candidate

    payload = _success_payload(included, output_was_truncated)
    while included and len(serialize_json(payload)) > MAX_SERIALIZED_OUTPUT_BYTES:
        included.pop()
        output_was_truncated = True
        payload = _success_payload(included, output_was_truncated)

    # The named cap is deliberately larger than the schema-only payload.  This
    # guard protects that invariant if an operator changes the constant.
    if len(serialize_json(payload)) > MAX_SERIALIZED_OUTPUT_BYTES:
        return {"results": [], "truncated": True}
    return payload


def _success_payload(
    results: list[dict[str, object]], truncated: bool
) -> dict[str, object]:
    payload: dict[str, object] = {"results": results}
    if truncated:
        payload["truncated"] = True
    return payload


def application_error(
    error_type: str, message: str, retryable: bool
) -> dict[str, object]:
    """Return the exact model-visible application-error envelope."""

    return {
        "ok": False,
        "error": {"type": error_type, "message": message, "retryable": retryable},
        "stdout": "",
        "stderr": "",
    }


def serialize_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def truncate_utf8(value: str, max_bytes: int) -> tuple[str, bool]:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return "", True
    if len(encoded) <= max_bytes:
        return value, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


def _utf8_len(value: str) -> int:
    return len(value.encode("utf-8"))


def is_utf8_encodable(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _has_control_characters(value: str) -> bool:
    return any(ord(character) <= 0x1F or ord(character) == 0x7F for character in value)
