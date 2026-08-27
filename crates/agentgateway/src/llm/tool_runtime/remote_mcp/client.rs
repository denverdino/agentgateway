use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

use futures_util::StreamExt;
use rmcp::model::{
	CallToolRequest, CallToolRequestParams, CallToolResult, ClientCapabilities, ClientRequest,
	Implementation, InitializeRequest, InitializeRequestParams, InitializedNotification,
	JsonRpcRequest, ListToolsRequest, PaginatedRequestParams, RequestId, ServerJsonRpcMessage,
	ServerResult,
};
use secrecy::{ExposeSecret, SecretString};
use serde_json::Map;

use super::{
	MAX_DESCRIPTION_BYTES, MAX_DISCOVERED_TOOLS, MAX_DISCOVERY_BYTES, MAX_DISCOVERY_PAGES,
	MAX_INPUT_SCHEMA_BYTES, MAX_OUTPUT_SCHEMA_BYTES, MAX_TOOL_NAME_BYTES,
};
use crate::Strng;
use crate::http::backendtls::SYSTEM_TRUST;
use crate::mcp::upstream::{IncomingRequestContext, McpHttpClient, McpStreamableClient, Upstream};
use crate::proxy::httpproxy::PolicyClient;
use crate::store::BackendPolicies;
use crate::types::agent::{ResourceName, SimpleBackend, Target};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const RESPONSES_REMOTE_TARGET: &str = "responses_remote_mcp";

#[derive(Clone, Debug)]
pub(crate) struct RemoteClientTool {
	pub name: String,
	pub description: Option<String>,
	pub input_schema: serde_json::Value,
	pub output_schema: Option<serde_json::Value>,
}

#[derive(Debug)]
pub(crate) enum RemoteCallError {
	Application,
	Infrastructure,
}

/// Request-scoped MCP client used by the Responses Tool Runtime.
///
/// This intentionally reuses the proxy's outbound MCP transport instead of
/// creating a second reqwest/rmcp transport stack. One instance owns one MCP
/// session and can therefore serve all tool calls in the same Response.
pub(crate) struct RemoteClient {
	upstream: Arc<Upstream>,
	context: IncomingRequestContext,
	target_name: String,
	next_id: AtomicI64,
}

impl RemoteClient {
	pub(crate) async fn connect(
		policy_client: PolicyClient,
		extensions: &::http::Extensions,
		server_url: &str,
		authorization: Option<&SecretString>,
		allowed_tools: Option<&[String]>,
		allow_private_http: bool,
		deadline: tokio::time::Instant,
	) -> anyhow::Result<(Self, Vec<RemoteClientTool>)> {
		let endpoint = Endpoint::resolve(server_url, allow_private_http).await?;
		let mut headers = ::http::HeaderMap::new();
		headers.insert(::http::header::HOST, endpoint.host_header.parse()?);
		if let Some(authorization) = authorization {
			let value = authorization.expose_secret();
			let value = if value.starts_with("Bearer ") {
				value.to_owned()
			} else {
				format!("Bearer {value}")
			};
			headers.insert(::http::header::AUTHORIZATION, value.parse()?);
		}
		let context = IncomingRequestContext::outbound(extensions, headers);
		let policies = BackendPolicies {
			backend_tls: endpoint.https.then(|| {
				let mut tls = SYSTEM_TRUST.clone();
				tls.hostname_override = Some(
					endpoint
						.host
						.clone()
						.try_into()
						.expect("validated DNS name"),
				);
				tls
			}),
			..Default::default()
		};
		let backend = SimpleBackend::Opaque(
			ResourceName::new(Strng::from("responses-mcp"), Strng::from("")),
			Target::Address(endpoint.address),
		);
		let http = McpHttpClient::new(
			policy_client,
			backend,
			policies,
			true,
			RESPONSES_REMOTE_TARGET.to_owned(),
		);
		let transport = McpStreamableClient::new(http, Strng::from(endpoint.path))?;
		let client = Self {
			upstream: Arc::new(Upstream::McpStreamable(transport)),
			context,
			target_name: RESPONSES_REMOTE_TARGET.to_owned(),
			next_id: AtomicI64::new(1),
		};
		client.initialize(deadline).await?;
		let tools = client.list_tools(allowed_tools, deadline).await?;
		Ok((client, tools))
	}

	async fn initialize(&self, deadline: tokio::time::Instant) -> anyhow::Result<()> {
		let params = InitializeRequestParams::new(
			ClientCapabilities::default(),
			Implementation::new(
				"agentgateway".to_owned(),
				env!("CARGO_PKG_VERSION").to_owned(),
			),
		);
		let request = InitializeRequest::new(params);
		match self
			.request(request.into(), phase_deadline(deadline, CONNECT_TIMEOUT))
			.await
			.map_err(|_| anyhow::anyhow!("remote MCP initialize failed"))?
		{
			ServerResult::InitializeResult(_) => {},
			_ => anyhow::bail!("remote MCP server returned an invalid initialize result"),
		}
		self
			.upstream
			.generic_notification(
				&self.target_name,
				InitializedNotification::default().into(),
				&self.context,
			)
			.await
			.map_err(|error| anyhow::anyhow!(error.to_string()))?;
		Ok(())
	}

	async fn list_tools(
		&self,
		allowed_tools: Option<&[String]>,
		deadline: tokio::time::Instant,
	) -> anyhow::Result<Vec<RemoteClientTool>> {
		let allowed =
			allowed_tools.map(|items| items.iter().map(String::as_str).collect::<HashSet<_>>());
		let mut cursor = None;
		let mut tools = Vec::new();
		let mut discovered_names = HashSet::new();
		let mut seen_cursors = HashSet::new();
		let mut budget = DiscoveryBudget::default();
		loop {
			budget.begin_page()?;
			let params = cursor
				.take()
				.map(|cursor| PaginatedRequestParams::default().with_cursor(Some(cursor)));
			let request = ListToolsRequest {
				params,
				..Default::default()
			};
			let ServerResult::ListToolsResult(page) = self
				.request(request.into(), phase_deadline(deadline, REQUEST_TIMEOUT))
				.await
				.map_err(|_| anyhow::anyhow!("remote MCP tools/list failed"))?
			else {
				anyhow::bail!("remote MCP server returned an invalid tools/list result");
			};
			for tool in page.tools {
				let name = tool.name.into_owned();
				let description = tool.description.map(|value| value.into_owned());
				let input_schema = serde_json::Value::Object((*tool.input_schema).clone());
				let output_schema = tool
					.output_schema
					.map(|schema| serde_json::Value::Object((*schema).clone()));
				budget.record_tool(
					&name,
					description.as_deref(),
					&input_schema,
					output_schema.as_ref(),
				)?;
				if !discovered_names.insert(name.clone()) {
					anyhow::bail!("remote MCP server advertised a duplicate tool name");
				}
				if allowed
					.as_ref()
					.is_some_and(|allowed| !allowed.contains(name.as_str()))
				{
					continue;
				}
				tools.push(RemoteClientTool {
					name,
					description,
					input_schema,
					output_schema,
				});
			}
			let Some(next_cursor) = page.next_cursor else {
				break;
			};
			budget.record_cursor(&next_cursor)?;
			if !seen_cursors.insert(next_cursor.clone()) {
				anyhow::bail!("remote MCP server repeated a pagination cursor");
			}
			cursor = Some(next_cursor);
		}
		if let Some(allowed) = allowed {
			let found = tools
				.iter()
				.map(|tool| tool.name.as_str())
				.collect::<HashSet<_>>();
			if !allowed.is_subset(&found) {
				anyhow::bail!("an allowed remote MCP tool was not advertised by the server");
			}
		}
		Ok(tools)
	}

	pub(crate) async fn call_tool(
		&self,
		name: &str,
		arguments: Map<String, serde_json::Value>,
		deadline: Option<std::time::Instant>,
	) -> Result<CallToolResult, RemoteCallError> {
		let timeout = deadline
			.and_then(|deadline| deadline.checked_duration_since(std::time::Instant::now()))
			.unwrap_or(REQUEST_TIMEOUT)
			.min(REQUEST_TIMEOUT);
		let request =
			CallToolRequest::new(CallToolRequestParams::new(name.to_owned()).with_arguments(arguments));
		let deadline = tokio::time::Instant::now() + timeout;
		let ServerResult::CallToolResult(result) = self.request(request.into(), deadline).await? else {
			return Err(RemoteCallError::Infrastructure);
		};
		Ok(result)
	}

	async fn request(
		&self,
		request: ClientRequest,
		deadline: tokio::time::Instant,
	) -> Result<ServerResult, RemoteCallError> {
		let id = RequestId::Number(self.next_id.fetch_add(1, Ordering::Relaxed));
		tokio::time::timeout_at(deadline, async {
			let mut messages = self
				.upstream
				.generic_stream(
					&self.target_name,
					JsonRpcRequest::new(id.clone(), request),
					&self.context,
				)
				.await
				.map_err(|_| RemoteCallError::Infrastructure)?;
			while let Some(message) = messages.next().await {
				match message.map_err(|_| RemoteCallError::Infrastructure)? {
					ServerJsonRpcMessage::Response(response) if response.id == id => {
						return Ok(response.result);
					},
					ServerJsonRpcMessage::Error(error) if error.id.as_ref() == Some(&id) => {
						return Err(RemoteCallError::Application);
					},
					ServerJsonRpcMessage::Request(_) => return Err(RemoteCallError::Infrastructure),
					_ => {},
				}
			}
			Err(RemoteCallError::Infrastructure)
		})
		.await
		.map_err(|_| RemoteCallError::Infrastructure)?
	}
}

#[derive(Default)]
struct DiscoveryBudget {
	pages: usize,
	tools: usize,
	bytes: usize,
}

impl DiscoveryBudget {
	fn begin_page(&mut self) -> anyhow::Result<()> {
		self.pages = self
			.pages
			.checked_add(1)
			.ok_or_else(|| anyhow::anyhow!("remote MCP discovery page limit exceeded"))?;
		if self.pages > MAX_DISCOVERY_PAGES {
			anyhow::bail!("remote MCP discovery page limit exceeded");
		}
		Ok(())
	}

	fn record_tool(
		&mut self,
		name: &str,
		description: Option<&str>,
		input_schema: &serde_json::Value,
		output_schema: Option<&serde_json::Value>,
	) -> anyhow::Result<()> {
		self.tools = self
			.tools
			.checked_add(1)
			.ok_or_else(|| anyhow::anyhow!("remote MCP discovery tool limit exceeded"))?;
		if self.tools > MAX_DISCOVERED_TOOLS
			|| name.is_empty()
			|| name.len() > MAX_TOOL_NAME_BYTES
			|| description.is_some_and(|value| value.len() > MAX_DESCRIPTION_BYTES)
		{
			anyhow::bail!("remote MCP discovery metadata limit exceeded");
		}
		let schema_bytes = serde_json::to_vec(input_schema)?.len();
		let output_schema_bytes = output_schema
			.map(serde_json::to_vec)
			.transpose()?
			.map_or(0, |schema| schema.len());
		if schema_bytes > MAX_INPUT_SCHEMA_BYTES || output_schema_bytes > MAX_OUTPUT_SCHEMA_BYTES {
			anyhow::bail!("remote MCP discovery schema limit exceeded");
		}
		self.add_bytes(
			name
				.len()
				.saturating_add(description.map_or(0, str::len))
				.saturating_add(schema_bytes)
				.saturating_add(output_schema_bytes),
		)
	}

	fn record_cursor(&mut self, cursor: &str) -> anyhow::Result<()> {
		if cursor.len() > MAX_DESCRIPTION_BYTES {
			anyhow::bail!("remote MCP discovery cursor limit exceeded");
		}
		self.add_bytes(cursor.len())
	}

	fn add_bytes(&mut self, bytes: usize) -> anyhow::Result<()> {
		self.bytes = self
			.bytes
			.checked_add(bytes)
			.ok_or_else(|| anyhow::anyhow!("remote MCP discovery byte limit exceeded"))?;
		if self.bytes > MAX_DISCOVERY_BYTES {
			anyhow::bail!("remote MCP discovery byte limit exceeded");
		}
		Ok(())
	}
}

fn phase_deadline(
	request_deadline: tokio::time::Instant,
	phase_timeout: Duration,
) -> tokio::time::Instant {
	request_deadline.min(tokio::time::Instant::now() + phase_timeout)
}

impl Drop for RemoteClient {
	fn drop(&mut self) {
		let Ok(runtime) = tokio::runtime::Handle::try_current() else {
			return;
		};
		let upstream = self.upstream.clone();
		let context = self.context.clone();
		let target_name = self.target_name.clone();
		runtime.spawn(async move {
			if let Err(error) = upstream.delete(&target_name, &context).await {
				tracing::debug!(target = %target_name, %error, "failed to close remote MCP session");
			}
		});
	}
}

struct Endpoint {
	host: String,
	host_header: String,
	address: SocketAddr,
	path: String,
	https: bool,
}

impl Endpoint {
	async fn resolve(server_url: &str, allow_private_http: bool) -> anyhow::Result<Self> {
		let url = url::Url::parse(server_url)?;
		if (!allow_private_http && url.scheme() != "https")
			|| (allow_private_http && !matches!(url.scheme(), "http" | "https"))
			|| !url.username().is_empty()
			|| url.password().is_some()
			|| url.fragment().is_some()
		{
			anyhow::bail!("invalid remote MCP server URL");
		}
		let host = url
			.host_str()
			.ok_or_else(|| anyhow::anyhow!("remote MCP URL has no host"))?;
		let port = url
			.port_or_known_default()
			.ok_or_else(|| anyhow::anyhow!("remote MCP URL has no port"))?;
		let addresses = tokio::net::lookup_host((host, port))
			.await?
			.collect::<Vec<_>>();
		if addresses.is_empty()
			|| (!allow_private_http && addresses.iter().any(|address| !is_public_ip(address.ip())))
		{
			anyhow::bail!("remote MCP URL resolves to a non-public address");
		}
		let default_port =
			(url.scheme() == "https" && port == 443) || (url.scheme() == "http" && port == 80);
		let authority_host = if host.contains(':') {
			format!("[{host}]")
		} else {
			host.to_owned()
		};
		let host_header = if default_port {
			authority_host
		} else {
			format!("{authority_host}:{port}")
		};
		let mut path = url.path().to_owned();
		if path.is_empty() {
			path.push('/');
		}
		if let Some(query) = url.query() {
			path.push('?');
			path.push_str(query);
		}
		Ok(Self {
			host: host.to_owned(),
			host_header,
			address: addresses[0],
			path,
			https: url.scheme() == "https",
		})
	}
}

fn is_public_ip(ip: IpAddr) -> bool {
	match ip {
		IpAddr::V4(ip) => is_public_ipv4(ip),
		IpAddr::V6(ip) => is_public_ipv6(ip),
	}
}

fn is_public_ipv4(ip: Ipv4Addr) -> bool {
	let [a, b, c, d] = ip.octets();
	!(a == 0
		|| a == 10
		|| a == 127
		|| (a == 100 && (64..=127).contains(&b))
		|| (a == 169 && b == 254)
		|| (a == 172 && (16..=31).contains(&b))
		|| (a == 192 && b == 0 && c == 0 && !matches!(d, 9 | 10))
		|| (a == 192 && b == 0 && c == 2)
		|| (a == 192 && b == 88 && c == 99)
		|| (a == 192 && b == 168)
		|| (a == 198 && (b == 18 || b == 19))
		|| (a == 198 && b == 51 && c == 100)
		|| (a == 203 && b == 0 && c == 113)
		|| a >= 224
		|| (a == 255 && b == 255 && c == 255 && d == 255))
}

fn is_public_ipv6(ip: Ipv6Addr) -> bool {
	let segments = ip.segments();

	// The public IPv6 space is the IANA global-unicast allocation, plus the
	// globally reachable well-known NAT64 prefix. Keep this classification
	// explicit: `IpAddr::is_global` is unstable and broader "unicast" helpers
	// include locally routed and special-purpose ranges that are unsafe for an
	// SSRF boundary.
	let well_known_nat64 = segments[0] == 0x0064
		&& segments[1] == 0xff9b
		&& segments[2] == 0
		&& segments[3] == 0
		&& segments[4] == 0
		&& segments[5] == 0;
	if well_known_nat64 {
		let embedded_ipv4 = Ipv4Addr::new(
			(segments[6] >> 8) as u8,
			segments[6] as u8,
			(segments[7] >> 8) as u8,
			segments[7] as u8,
		);
		return is_public_ipv4(embedded_ipv4);
	}
	if (segments[0] & 0xe000) != 0x2000 {
		return false;
	}

	// 2001::/23 is reserved for IETF protocol assignments. Only the entries
	// explicitly marked globally reachable in the IANA registry are public.
	if segments[0] == 0x2001 && segments[1] <= 0x01ff {
		let globally_reachable_exception = (segments[1] == 0x0001
			&& segments[2] == 0
			&& segments[3] == 0
			&& segments[4] == 0
			&& segments[5] == 0
			&& segments[6] == 0
			&& matches!(segments[7], 1..=3))
			|| segments[1] == 0x0003
			|| (segments[1] == 0x0004 && segments[2] == 0x0112)
			|| (segments[1] & 0xfff0) == 0x0020
			|| (segments[1] & 0xfff0) == 0x0030;
		if !globally_reachable_exception {
			return false;
		}
	}

	// Documentation, deprecated 6to4, and documentation-v2 are not public.
	!(segments[0] == 0x2001 && segments[1] == 0x0db8
		|| segments[0] == 0x2002
		|| (segments[0] == 0x3fff && (segments[1] & 0xf000) == 0))
}

#[cfg(test)]
mod tests {
	use std::net::{Ipv4Addr, Ipv6Addr};
	use std::str::FromStr;

	use super::{DiscoveryBudget, is_public_ipv4, is_public_ipv6};
	use crate::llm::tool_runtime::remote_mcp::{
		MAX_DISCOVERED_TOOLS, MAX_DISCOVERY_BYTES, MAX_DISCOVERY_PAGES, MAX_INPUT_SCHEMA_BYTES,
	};

	#[test]
	fn publicly_reachable_ipv4_matches_iana_special_purpose_boundaries() {
		let cases = [
			("0.0.0.0", false),
			("0.255.255.255", false),
			("1.0.0.0", true),
			("9.255.255.255", true),
			("10.0.0.0", false),
			("10.255.255.255", false),
			("11.0.0.0", true),
			("100.63.255.255", true),
			("100.64.0.0", false),
			("100.127.255.255", false),
			("100.128.0.0", true),
			("126.255.255.255", true),
			("127.0.0.0", false),
			("127.255.255.255", false),
			("128.0.0.0", true),
			("169.253.255.255", true),
			("169.254.0.0", false),
			("169.254.255.255", false),
			("169.255.0.0", true),
			("172.15.255.255", true),
			("172.16.0.0", false),
			("172.31.255.255", false),
			("172.32.0.0", true),
			("192.0.0.0", false),
			("192.0.0.8", false),
			("192.0.0.9", true),
			("192.0.0.10", true),
			("192.0.0.11", false),
			("192.0.0.255", false),
			("192.0.1.255", true),
			("192.0.2.0", false),
			("192.0.2.255", false),
			("192.0.3.0", true),
			("192.31.196.1", true),
			("192.52.193.1", true),
			("192.88.98.255", true),
			("192.88.99.0", false),
			("192.88.99.2", false),
			("192.88.99.255", false),
			("192.88.100.0", true),
			("192.167.255.255", true),
			("192.168.0.0", false),
			("192.168.255.255", false),
			("192.169.0.0", true),
			("192.175.48.1", true),
			("198.17.255.255", true),
			("198.18.0.0", false),
			("198.19.255.255", false),
			("198.20.0.0", true),
			("198.51.100.0", false),
			("198.51.100.255", false),
			("203.0.113.0", false),
			("203.0.113.255", false),
			("223.255.255.255", true),
			("224.0.0.0", false),
			("239.255.255.255", false),
			("240.0.0.0", false),
			("255.255.255.255", false),
		];

		for (address, expected) in cases {
			let address = Ipv4Addr::from_str(address).unwrap();
			assert_eq!(is_public_ipv4(address), expected, "address={address}");
		}
	}

	#[test]
	fn publicly_reachable_ipv6_matches_iana_special_purpose_boundaries() {
		let cases = [
			("::", false),
			("::1", false),
			("::ffff:8.8.8.8", false),
			("::ffff:127.0.0.1", false),
			("64:ff9a:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("64:ff9b::", false),
			("64:ff9b::8.8.8.8", true),
			("64:ff9b::10.0.0.1", false),
			("64:ff9b::127.0.0.1", false),
			("64:ff9b::169.254.1.1", false),
			("64:ff9b::192.0.2.1", false),
			("64:ff9b::224.0.0.1", false),
			("64:ff9b::240.0.0.1", false),
			("64:ff9b::255.255.255.255", false),
			("64:ff9b:0:ffff:ffff:ffff:ffff:ffff", false),
			("64:ff9b:1::", false),
			("64:ff9b:1:ffff:ffff:ffff:ffff:ffff", false),
			("64:ff9b:2::", false),
			("ff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("100::", false),
			("100::ffff:ffff:ffff:ffff", false),
			("100:0:0:1::", false),
			("100:0:0:1:ffff:ffff:ffff:ffff", false),
			("100:0:0:2::", false),
			("2000:ffff:ffff:ffff:ffff:ffff:ffff:ffff", true),
			("2001::", false),
			("2001:1::", false),
			("2001:1::1", true),
			("2001:1::2", true),
			("2001:1::3", true),
			("2001:1::4", false),
			("2001:2::", false),
			("2001:3::", true),
			("2001:3:ffff:ffff:ffff:ffff:ffff:ffff", true),
			("2001:4:111:ffff:ffff:ffff:ffff:ffff", false),
			("2001:4:112::", true),
			("2001:4:112:ffff:ffff:ffff:ffff:ffff", true),
			("2001:4:113::", false),
			("2001:20::", true),
			("2001:3f:ffff:ffff:ffff:ffff:ffff:ffff", true),
			("2001:40::", false),
			("2001:1ff:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("2001:200::", true),
			("2001:db7:ffff:ffff:ffff:ffff:ffff:ffff", true),
			("2001:db8::", false),
			("2001:db8:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("2001:db9::", true),
			("2002::", false),
			("2002:ffff:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("2003::", true),
			("3ffe:ffff:ffff:ffff:ffff:ffff:ffff:ffff", true),
			("3fff::", false),
			("3fff:fff:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("3fff:1000::", true),
			("5eff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("5f00::", false),
			("5f00:ffff:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("5f01::", false),
			("fbff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("fc00::", false),
			("fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("fe00::", false),
			("fe7f:ffff:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("fe80::", false),
			("febf:ffff:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("fec0::", false),
			("feff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", false),
			("ff00::", false),
			("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", false),
		];

		for (address, expected) in cases {
			let address = Ipv6Addr::from_str(address).unwrap();
			assert_eq!(is_public_ipv6(address), expected, "address={address}");
		}
	}

	#[test]
	fn discovery_budget_stops_at_tool_page_and_schema_limits() {
		let mut pages = DiscoveryBudget::default();
		for _ in 0..MAX_DISCOVERY_PAGES {
			pages.begin_page().unwrap();
		}
		assert!(pages.begin_page().is_err());
		let mut bytes = DiscoveryBudget {
			bytes: MAX_DISCOVERY_BYTES,
			..Default::default()
		};
		assert!(bytes.record_cursor("x").is_err());

		let mut tools = DiscoveryBudget::default();
		for index in 0..MAX_DISCOVERED_TOOLS {
			tools
				.record_tool(
					&format!("tool_{index}"),
					None,
					&serde_json::json!({"type": "object"}),
					None,
				)
				.unwrap();
		}
		assert!(
			tools
				.record_tool(
					"one_too_many",
					None,
					&serde_json::json!({"type": "object"}),
					None,
				)
				.is_err()
		);

		let oversized_schema = serde_json::json!({"description": "x".repeat(MAX_INPUT_SCHEMA_BYTES)});
		assert!(
			DiscoveryBudget::default()
				.record_tool("large", None, &oversized_schema, None)
				.is_err()
		);
		assert!(
			DiscoveryBudget::default()
				.record_tool(
					"large_output",
					None,
					&serde_json::json!({"type": "object"}),
					Some(&oversized_schema),
				)
				.is_err()
		);
	}
}
