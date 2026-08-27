import assert from "node:assert/strict";
import { test } from "node:test";

import {
  MAX_PROVIDER_RESPONSE_BYTES,
  createRequestHandler,
} from "./server.mjs";

const ENVIRONMENT = {
  WEATHERAPI_KEY: "weatherapi-test-key",
  INGRESS_BEARER_TOKEN: "ingress-test-token",
};

function mcpRequest(body, authorization = "Bearer ingress-test-token") {
  return new Request("https://function.example.test/mcp", {
    method: "POST",
    headers: {
      accept: "application/json, text/event-stream",
      authorization,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

async function json(response) {
  const body = await response.json();
  return { status: response.status, body };
}

function handler(fetchImpl = async () => {
  throw new Error("unexpected provider request");
}, providerTimeoutMs) {
  return createRequestHandler({ environment: ENVIRONMENT, fetchImpl, providerTimeoutMs });
}

test("health endpoint is available without bearer authentication", async () => {
  const response = await handler()(new Request("https://function.example.test/healthz"));

  assert.deepEqual(await json(response), {
    status: 200,
    body: { status: "ok" },
  });
});

test("MCP endpoint authenticates before parsing the request body", async () => {
  const request = new Request("https://function.example.test/mcp", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "not-json",
  });

  const response = await handler()(request);

  assert.equal(response.status, 401);
  assert.equal(response.headers.get("www-authenticate"), "Bearer");
  assert.deepEqual(await response.json(), {
    error: "unauthorized",
  });
});

test("MCP initialize uses stateless JSON response mode", async () => {
  const response = await handler()(mcpRequest({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2025-03-26",
      capabilities: {},
      clientInfo: { name: "test-client", version: "1.0.0" },
    },
  }));

  const result = await json(response);
  assert.equal(result.status, 200);
  assert.equal(response.headers.get("content-type"), "application/json");
  assert.equal(response.headers.get("mcp-session-id"), null);
  assert.equal(result.body.jsonrpc, "2.0");
  assert.equal(result.body.id, 1);
  assert.equal(result.body.result.serverInfo.name, "weatherapi-mcp-fc");
  assert.deepEqual(result.body.result.capabilities, { tools: { listChanged: true } });
});

test("tools/list exposes only current weather and forecast", async () => {
  const response = await handler()(mcpRequest({
    jsonrpc: "2.0",
    id: 2,
    method: "tools/list",
    params: {},
  }));

  const result = await json(response);
  assert.equal(result.status, 200);
  assert.deepEqual(
    result.body.result.tools.map((tool) => tool.name),
    ["get_current_weather", "get_forecast"],
  );
  assert.deepEqual(result.body.result.tools[0].inputSchema.required, ["q"]);
  assert.deepEqual(result.body.result.tools[0].outputSchema.required, ["location", "current"]);
  assert.deepEqual(result.body.result.tools[1].outputSchema.required, ["forecast"]);
  assert.equal(
    result.body.result.tools[1].outputSchema.properties.forecast.properties.forecastday
      .items.properties.day.properties.mintemp_c.type,
    "number",
  );
});

test("get_current_weather calls only the fixed WeatherAPI origin", async () => {
  const calls = [];
  const response = await handler(async (url, options) => {
    calls.push({ url: new URL(url), options });
    return Response.json({ location: { name: "Beijing" }, current: { temp_c: 28 } });
  })(mcpRequest({
    jsonrpc: "2.0",
    id: 3,
    method: "tools/call",
    params: {
      name: "get_current_weather",
      arguments: { q: "Beijing", aqi: "no" },
    },
  }));

  const result = await json(response);
  assert.equal(result.status, 200);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url.origin, "https://api.weatherapi.com");
  assert.equal(calls[0].url.pathname, "/v1/current.json");
  assert.equal(calls[0].url.searchParams.get("key"), "weatherapi-test-key");
  assert.equal(calls[0].url.searchParams.get("q"), "Beijing");
  assert.equal(calls[0].options.signal instanceof AbortSignal, true);
  assert.deepEqual(JSON.parse(result.body.result.content[0].text), {
    location: { name: "Beijing" },
    current: { temp_c: 28 },
  });
  assert.deepEqual(result.body.result.structuredContent, {
    location: { name: "Beijing" },
    current: { temp_c: 28 },
  });
});

test("get_forecast returns structured content matching its advertised output schema", async () => {
  const providerForecast = {
    location: { name: "Shanghai" },
    current: { temp_c: 31 },
    forecast: {
      forecastday: [{
        date: "2026-08-27",
        day: { mintemp_c: 26.2, maxtemp_c: 33, avgtemp_c: 29.6 },
        hour: [{ time: "2026-08-27 00:00", temp_c: 28 }],
      }],
    },
  };
  const response = await handler(async () => Response.json(providerForecast))(mcpRequest({
    jsonrpc: "2.0",
    id: 4,
    method: "tools/call",
    params: {
      name: "get_forecast",
      arguments: { q: "Shanghai", days: 3 },
    },
  }));

  const result = await json(response);
  assert.equal(result.status, 200);
  const expected = {
    forecast: {
      forecastday: [{
        date: "2026-08-27",
        day: { mintemp_c: 26.2, maxtemp_c: 33 },
      }],
    },
  };
  assert.deepEqual(result.body.result.structuredContent, expected);
  assert.deepEqual(JSON.parse(result.body.result.content[0].text), expected);
});

test("get_forecast validates days before calling WeatherAPI", async () => {
  let calls = 0;
  const response = await handler(async () => {
    calls += 1;
    return Response.json({});
  })(mcpRequest({
    jsonrpc: "2.0",
    id: 4,
    method: "tools/call",
    params: {
      name: "get_forecast",
      arguments: { q: "Beijing", days: 15 },
    },
  }));

  const result = await json(response);
  assert.equal(result.status, 200);
  assert.equal(calls, 0);
  assert.equal(result.body.result.isError, true);
  assert.match(result.body.result.content[0].text, /input validation error/i);
});

test("WeatherAPI failures are sanitized and do not expose provider content", async () => {
  const response = await handler(async () => new Response(
    JSON.stringify({ error: { message: "secret provider detail" } }),
    { status: 403, headers: { "content-type": "application/json" } },
  ))(mcpRequest({
    jsonrpc: "2.0",
    id: 5,
    method: "tools/call",
    params: {
      name: "get_current_weather",
      arguments: { q: "Beijing" },
    },
  }));

  const result = await json(response);
  assert.equal(result.body.result.isError, true);
  assert.equal(result.body.result.content[0].text, "WeatherAPI request failed.");
  assert.doesNotMatch(JSON.stringify(result.body), /secret provider detail|weatherapi-test-key/);
});

test("oversized WeatherAPI responses are rejected", async () => {
  const oversized = "x".repeat(MAX_PROVIDER_RESPONSE_BYTES + 1);
  const response = await handler(async () => new Response(
    JSON.stringify({ value: oversized }),
    { status: 200, headers: { "content-type": "application/json" } },
  ))(mcpRequest({
    jsonrpc: "2.0",
    id: 6,
    method: "tools/call",
    params: {
      name: "get_current_weather",
      arguments: { q: "Beijing" },
    },
  }));

  const result = await json(response);
  assert.equal(result.body.result.isError, true);
  assert.equal(result.body.result.content[0].text, "WeatherAPI response was invalid.");
});

test("WeatherAPI timeout remains active while reading the response body", { timeout: 1_000 }, async () => {
  const response = await handler(async (_url, { signal }) => new Response(new ReadableStream({
    start(controller) {
      signal.addEventListener("abort", () => {
        controller.error(new DOMException("aborted", "AbortError"));
      }, { once: true });
    },
  }), { status: 200 }), 10)(mcpRequest({
    jsonrpc: "2.0",
    id: 7,
    method: "tools/call",
    params: {
      name: "get_current_weather",
      arguments: { q: "Beijing" },
    },
  }));

  const result = await json(response);
  assert.equal(result.body.result.isError, true);
  assert.equal(result.body.result.content[0].text, "WeatherAPI response was invalid.");
});

test("unknown paths are not routed into MCP", async () => {
  const response = await handler()(new Request("https://function.example.test/other"));

  assert.equal(response.status, 404);
});
