import { timingSafeEqual } from "node:crypto";
import http from "node:http";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { z } from "zod";

export const MAX_MCP_REQUEST_BYTES = 64 * 1024;
export const MAX_PROVIDER_RESPONSE_BYTES = 512 * 1024;
const MAX_CREDENTIAL_BYTES = 4 * 1024;
const PROVIDER_TIMEOUT_MS = 10_000;
const WEATHERAPI_ORIGIN = "https://api.weatherapi.com";
const WEATHERAPI_BASE_PATH = "/v1";

class ProviderFailure extends Error {}
class ProviderResponseInvalid extends Error {}

function jsonResponse(body, status = 200, headers = {}) {
  return Response.json(body, {
    status,
    headers: {
      "cache-control": "no-store",
      ...headers,
    },
  });
}

function credential(value) {
  return typeof value === "string"
    && value.length > 0
    && Buffer.byteLength(value, "utf8") <= MAX_CREDENTIAL_BYTES
    ? value
    : null;
}

function bearerMatches(request, expectedToken) {
  const authorization = request.headers.get("authorization");
  if (!authorization?.startsWith("Bearer ")) return false;
  const supplied = Buffer.from(authorization.slice("Bearer ".length), "utf8");
  const expected = Buffer.from(expectedToken, "utf8");
  return supplied.length === expected.length && timingSafeEqual(supplied, expected);
}

async function readBounded(stream, maximumBytes) {
  if (stream === null) return new Uint8Array();
  const reader = stream.getReader();
  const chunks = [];
  let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > maximumBytes) throw new ProviderResponseInvalid();
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const body = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

function validateEnvironment(environment) {
  const apiKey = credential(environment.WEATHERAPI_KEY);
  const ingressToken = credential(environment.INGRESS_BEARER_TOKEN);
  return apiKey && ingressToken ? { apiKey, ingressToken } : null;
}

async function weatherRequest(fetchImpl, apiKey, endpoint, parameters, providerTimeoutMs) {
  const url = new URL(WEATHERAPI_BASE_PATH + endpoint, WEATHERAPI_ORIGIN);
  url.searchParams.set("key", apiKey);
  for (const [name, value] of Object.entries(parameters)) {
    url.searchParams.set(name, String(value));
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), providerTimeoutMs);
  try {
    let response;
    try {
      response = await fetchImpl(url, {
        method: "GET",
        headers: { accept: "application/json" },
        redirect: "error",
        signal: controller.signal,
      });
    } catch {
      throw new ProviderFailure();
    }

    if (!response.ok) throw new ProviderFailure();
    try {
      const body = await readBounded(response.body, MAX_PROVIDER_RESPONSE_BYTES);
      return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(body));
    } catch (error) {
      if (error instanceof ProviderResponseInvalid) throw error;
      throw new ProviderResponseInvalid();
    }
  } finally {
    clearTimeout(timeout);
  }
}

function providerError(error) {
  return {
    content: [{
      type: "text",
      text: error instanceof ProviderResponseInvalid
        ? "WeatherAPI response was invalid."
        : "WeatherAPI request failed.",
    }],
    isError: true,
  };
}

export function createMcpServer(fetchImpl, apiKey, providerTimeoutMs = PROVIDER_TIMEOUT_MS) {
  const server = new McpServer(
    { name: "weatherapi-mcp-fc", version: "1.0.0" },
    { capabilities: { tools: { listChanged: true } } },
  );

  server.registerTool("get_current_weather", {
    description: "Get real-time current weather for a location, optionally including air quality data.",
    inputSchema: {
      q: z.string().trim().min(1).max(256).describe("City, coordinates, postcode, airport code, or IP address."),
      aqi: z.enum(["yes", "no"]).default("no"),
    },
  }, async ({ q, aqi }) => {
    try {
      const result = await weatherRequest(fetchImpl, apiKey, "/current.json", { q, aqi }, providerTimeoutMs);
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    } catch (error) {
      return providerError(error);
    }
  });

  server.registerTool("get_forecast", {
    description: "Get a 1 to 14 day weather forecast with current conditions and hourly data.",
    inputSchema: {
      q: z.string().trim().min(1).max(256).describe("City, coordinates, postcode, airport code, or IP address."),
      days: z.number().int().min(1).max(14).default(3),
      alerts: z.enum(["yes", "no"]).default("no"),
      aqi: z.enum(["yes", "no"]).default("no"),
    },
  }, async ({ q, days, alerts, aqi }) => {
    try {
      const result = await weatherRequest(fetchImpl, apiKey, "/forecast.json", {
        q,
        days,
        alerts,
        aqi,
      }, providerTimeoutMs);
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    } catch (error) {
      return providerError(error);
    }
  });

  return server;
}

async function boundedMcpRequest(request) {
  const contentLength = Number(request.headers.get("content-length"));
  if (Number.isFinite(contentLength) && contentLength > MAX_MCP_REQUEST_BYTES) return null;
  try {
    const body = await readBounded(request.body, MAX_MCP_REQUEST_BYTES);
    return new Request(request.url, {
      method: request.method,
      headers: request.headers,
      body,
    });
  } catch {
    return null;
  }
}

export function createRequestHandler({
  environment = process.env,
  fetchImpl = fetch,
  providerTimeoutMs = PROVIDER_TIMEOUT_MS,
} = {}) {
  return async function handleHttpRequest(request) {
    const url = new URL(request.url);
    if (url.pathname === "/healthz" && request.method === "GET") {
      return jsonResponse({ status: "ok" });
    }
    if (url.pathname !== "/mcp") return jsonResponse({ error: "not_found" }, 404);

    const configuration = validateEnvironment(environment);
    if (!configuration) return jsonResponse({ error: "service_unavailable" }, 503);
    if (!bearerMatches(request, configuration.ingressToken)) {
      return jsonResponse({ error: "unauthorized" }, 401, {
        "www-authenticate": "Bearer",
      });
    }
    if (request.method !== "POST") {
      return jsonResponse({ error: "method_not_allowed" }, 405, { allow: "POST" });
    }

    const boundedRequest = await boundedMcpRequest(request);
    if (boundedRequest === null) {
      return jsonResponse({
        jsonrpc: "2.0",
        error: { code: -32600, message: "Request is too large." },
        id: null,
      }, 413);
    }

    const server = createMcpServer(fetchImpl, configuration.apiKey, providerTimeoutMs);
    const transport = new WebStandardStreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableJsonResponse: true,
    });
    try {
      await server.connect(transport);
      return await transport.handleRequest(boundedRequest);
    } catch {
      return jsonResponse({
        jsonrpc: "2.0",
        error: { code: -32603, message: "Internal server error." },
        id: null,
      }, 500);
    }
  };
}

async function nodeRequest(req, maximumBytes) {
  const chunks = [];
  let length = 0;
  for await (const chunk of req) {
    length += chunk.length;
    if (length > maximumBytes) throw new ProviderResponseInvalid();
    chunks.push(chunk);
  }
  const host = req.headers.host ?? "127.0.0.1";
  return new Request(`http://${host}${req.url ?? "/"}`, {
    method: req.method,
    headers: req.headers,
    body: chunks.length === 0 ? undefined : Buffer.concat(chunks),
  });
}

async function writeNodeResponse(response, res) {
  res.writeHead(response.status, Object.fromEntries(response.headers.entries()));
  if (response.body === null) return res.end();
  const reader = response.body.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    res.write(value);
  }
  res.end();
}

export function startServer({ environment = process.env, fetchImpl = fetch } = {}) {
  const port = Number.parseInt(environment.CAPort ?? "9000", 10);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("CAPort must be a valid TCP port.");
  }
  const handler = createRequestHandler({ environment, fetchImpl });
  const server = http.createServer(async (req, res) => {
    try {
      const request = await nodeRequest(req, MAX_MCP_REQUEST_BYTES);
      await writeNodeResponse(await handler(request), res);
    } catch {
      if (!res.headersSent) {
        await writeNodeResponse(jsonResponse({ error: "invalid_request" }, 400), res);
      } else {
        res.destroy();
      }
    }
  });
  return server.listen(port, "0.0.0.0", () => {
    process.stdout.write(`weatherapi-mcp-fc listening on ${port}\n`);
  });
}

if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  startServer();
}
