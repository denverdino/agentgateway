use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use agent_core::prelude::Strng;
use secrecy::SecretString;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::remote_mcp::{
	MAX_ALLOWED_TOOLS, MAX_DESCRIPTION_BYTES, MAX_SERVER_LABEL_BYTES, MAX_TOOL_NAME_BYTES,
};
use super::schema::{ArgumentSchema, code_interpreter_parameters, web_search_parameters};
use super::{
	Activation, AllowedCaller, AllowedCallers, CODE_INTERPRETER_FUNCTION, DeferredTool,
	MAX_DEFERRED_TOOLS, PreparedToolRuntime, ProgrammaticToolSpec, RemoteMcpServer,
	ResponsesRequestExt, ToolRegistry, ToolRuntimeError, ToolSearchState, WEB_SEARCH_FUNCTION,
};
use crate::llm::types::responses;

const RESERVED_PREFIX: &str = "_agentgateway_";

pub(crate) fn prepare(
	request: &mut responses::Request,
	registry: Option<&Arc<ToolRegistry>>,
) -> Result<Activation, ToolRuntimeError> {
	let client_tools = request.rest_field("tools").cloned();
	let declarations = declarations(request)?;
	let mut seen_function_names = HashSet::new();
	let mut seen_builtin_types = HashSet::<String>::new();
	let mut function_names = Vec::new();
	let mut managed_function_present = false;
	let mut builtin_present = false;
	let mut remote_mcp_present = false;
	let mut programmatic_requested = false;
	let mut tool_search_requested = false;
	let mut seen_mcp_labels = HashSet::new();
	let mut pending_remote_mcp = Vec::new();
	let mut rewritten = Vec::with_capacity(declarations.len());
	let mut trusted_options = HashMap::new();
	let mut argument_schemas = HashMap::new();
	let mut programmatic_tools = Vec::new();
	let mut deferred_tools = Vec::new();

	for declaration in declarations {
		let kind = declaration
			.get("type")
			.and_then(Value::as_str)
			.ok_or_else(|| {
				ToolRuntimeError::invalid_request("tool declaration requires a string type")
			})?;
		match kind {
			"programmatic_tool_calling" => {
				if programmatic_requested
					|| declaration
						.as_object()
						.is_some_and(|value| value.len() != 1)
				{
					return Err(ToolRuntimeError::invalid_request(
						"programmatic_tool_calling may be declared once without options",
					));
				}
				programmatic_requested = true;
			},
			"tool_search" => {
				if !seen_builtin_types.insert(kind.to_owned()) {
					return Err(ToolRuntimeError::invalid_request(format!(
						"duplicate tool declaration for {kind}"
					)));
				}
				if declaration
					.as_object()
					.is_some_and(|value| value.len() != 1)
				{
					return Err(ToolRuntimeError::invalid_request(
						"tool_search may be declared once without options",
					));
				}
				if registry.is_none() {
					return Err(ToolRuntimeError::invalid_request(
						"tool_search requires llm.toolRuntime configuration",
					));
				}
				tool_search_requested = true;
			},
			"function" => {
				let mut target = FunctionTarget {
					seen_function_names: &mut seen_function_names,
					function_names: &mut function_names,
					managed_function_present: &mut managed_function_present,
					argument_schemas: &mut argument_schemas,
					deferred_tools: &mut deferred_tools,
					rewritten: &mut rewritten,
				};
				accept_function_declaration(&mut target, registry, declaration, None, false)?;
			},
			// A namespace declaration must never reach the model: `collect_model_calls` rejects any call
			// carrying a namespace. Flatten it into its member functions and keep the namespace name only
			// as search text.
			"namespace" => {
				let namespace: NamespaceDeclaration =
					serde_json::from_value(declaration).map_err(|error| {
						ToolRuntimeError::invalid_request(format!("invalid namespace declaration: {error}"))
					})?;
				if namespace.name.is_empty() || namespace.name.len() > MAX_TOOL_NAME_BYTES {
					return Err(ToolRuntimeError::invalid_request(
						"namespace name must contain 1 to 128 bytes",
					));
				}
				if namespace.tools.is_empty() {
					return Err(ToolRuntimeError::invalid_request(
						"namespace must declare at least one tool",
					));
				}
				let defer_loading = namespace.defer_loading.unwrap_or(false);
				for mut tool in namespace.tools {
					// Compose the namespace context into the member description, the same way a remote MCP
					// server label and description are composed onto its tools.
					if let Some(namespace_description) = namespace.description.as_deref()
						&& let Some(tool) = tool.as_object_mut()
					{
						let composed = match tool.get("description").and_then(Value::as_str) {
							Some(description) => format!(
								"[{}] {namespace_description}\n\n{description}",
								namespace.name
							),
							None => format!("[{}] {namespace_description}", namespace.name),
						};
						tool.insert("description".to_owned(), Value::String(composed));
					}
					let mut target = FunctionTarget {
						seen_function_names: &mut seen_function_names,
						function_names: &mut function_names,
						managed_function_present: &mut managed_function_present,
						argument_schemas: &mut argument_schemas,
						deferred_tools: &mut deferred_tools,
						rewritten: &mut rewritten,
					};
					accept_function_declaration(
						&mut target,
						registry,
						tool,
						Some(namespace.name.as_str()),
						defer_loading,
					)?;
				}
			},
			"web_search" => {
				if !seen_builtin_types.insert(kind.to_owned()) {
					return Err(ToolRuntimeError::invalid_request(format!(
						"duplicate tool declaration for {kind}"
					)));
				}
				builtin_registry(registry, WEB_SEARCH_FUNCTION, "web_search")?;
				let (options, allowed_callers) = web_search_options(&declaration)?;
				trusted_options.insert(Strng::from(WEB_SEARCH_FUNCTION), options);
				builtin_present = true;
				if allowed_callers.programmatic {
					programmatic_tools.push(ProgrammaticToolSpec {
						public_name: Strng::from("web_search"),
						internal_name: Strng::from(WEB_SEARCH_FUNCTION),
						description: "Search the web for current information and return relevant sources."
							.to_owned(),
						input_schema: web_search_parameters(),
						output_schema: None,
					});
				}
				if allowed_callers.direct {
					rewritten.push(web_search_schema());
				}
			},
			"code_interpreter" => {
				if !seen_builtin_types.insert(kind.to_owned()) {
					return Err(ToolRuntimeError::invalid_request(format!(
						"duplicate tool declaration for {kind}"
					)));
				}
				builtin_registry(registry, CODE_INTERPRETER_FUNCTION, "code_interpreter")?;
				let allowed_callers = code_interpreter_options(&declaration)?;
				builtin_present = true;
				if allowed_callers.programmatic {
					programmatic_tools.push(ProgrammaticToolSpec {
						public_name: Strng::from("code_interpreter"),
						internal_name: Strng::from(CODE_INTERPRETER_FUNCTION),
						description: "Execute Python code in an isolated sandbox and return stdout and stderr."
							.to_owned(),
						input_schema: code_interpreter_parameters(),
						output_schema: None,
					});
				}
				if allowed_callers.direct {
					rewritten.push(code_interpreter_schema());
				}
			},
			"mcp" => {
				if registry.is_none() {
					return Err(ToolRuntimeError::invalid_request(
						"remote MCP requires llm.toolRuntime configuration",
					));
				}
				let server = remote_mcp_options(&declaration)?;
				if !seen_mcp_labels.insert(server.server_label.clone()) {
					return Err(ToolRuntimeError::invalid_request(format!(
						"duplicate MCP server_label {}",
						server.server_label
					)));
				}
				pending_remote_mcp.push(server);
				remote_mcp_present = true;
			},
			// The gateway consumes the `tool_search` declaration, so a passthrough type that kept
			// `defer_loading` would reach the provider with nothing to satisfy it, and the gateway holds no
			// catalog entry to search it back into context.
			_ => {
				if declaration
					.get("defer_loading")
					.and_then(Value::as_bool)
					.unwrap_or(false)
				{
					return Err(ToolRuntimeError::invalid_request(format!(
						"defer_loading is only supported on function, mcp and namespace tools, not {kind}"
					)));
				}
				rewritten.push(declaration)
			},
		}
	}

	if !managed_function_present
		&& !builtin_present
		&& !remote_mcp_present
		&& !programmatic_requested
		&& !tool_search_requested
	{
		return Ok(Activation::Inactive);
	}
	if !tool_search_requested
		&& (!deferred_tools.is_empty() || pending_remote_mcp.iter().any(|server| server.defer_loading))
	{
		return Err(ToolRuntimeError::invalid_request(
			"defer_loading requires a tool_search declaration",
		));
	}
	if let Some(forced) = forced_function_name(request)
		&& deferred_tools.iter().any(|tool| tool.public_name == forced)
	{
		return Err(ToolRuntimeError::invalid_request(format!(
			"tool_choice cannot force the deferred function {forced}"
		)));
	}
	if programmatic_requested
		&& !registry.is_some_and(|registry| registry.has_internal(CODE_INTERPRETER_FUNCTION))
	{
		return Err(ToolRuntimeError::invalid_request(
			"programmatic_tool_calling requires a configured code_interpreter E2B backend",
		));
	}
	let registry = registry.expect("built-ins and managed functions require a registry");
	for name in function_names {
		if !registry.resolves_function(&name) {
			return Err(ToolRuntimeError::invalid_request(format!(
				"every function tool must be registered when managed tool runtime is active; {name} is not registered"
			)));
		}
	}
	unsupported_active_request_fields(request)?;
	let client_streaming = request.stream == Some(true);
	let include_obfuscation = if client_streaming {
		match request.rest_field("stream_options") {
			None | Some(Value::Null) => true,
			Some(Value::Object(options)) => match options.get("include_obfuscation") {
				None | Some(Value::Null) => true,
				Some(Value::Bool(include)) => *include,
				Some(_) => {
					return Err(ToolRuntimeError::invalid_request(
						"stream_options.include_obfuscation must be a boolean",
					));
				},
			},
			Some(_) => {
				return Err(ToolRuntimeError::invalid_request(
					"stream_options must be an object",
				));
			},
		}
	} else {
		false
	};
	if client_streaming {
		request.stream = Some(false);
		if let Some(rest) = request.rest.as_object_mut() {
			rest.remove("stream_options");
		}
	}

	let parallel = match request.rest_field("parallel_tool_calls") {
		None => true,
		Some(Value::Bool(parallel)) => *parallel,
		Some(_) => {
			return Err(ToolRuntimeError::invalid_request(
				"parallel_tool_calls must be a boolean",
			));
		},
	};
	request.replace_rest_field("tools", Value::Array(rewritten));
	let client_tools = sanitize_client_tools(client_tools).map(Box::new);
	let prepared_registry = Arc::new(registry.with_request_data(trusted_options, argument_schemas));
	let deadline = super::RuntimeDeadline::new(prepared_registry.limits.total_timeout);
	let programmatic_tools = programmatic_tools
		.into_iter()
		.map(|tool| (tool.public_name.clone(), tool))
		.collect();
	let programmatic_catalog_bytes = super::programmatic_catalog_bytes(&programmatic_tools)?;
	let mut prepared = PreparedToolRuntime {
		state: super::ManagedToolState {
			registry: prepared_registry,
			parallel,
			deadline,
			programmatic_requested,
			programmatic_tools: Arc::new(programmatic_tools),
			programmatic_catalog_bytes,
		},
		canonical_request: request.clone(),
		client_streaming,
		include_obfuscation,
		client_tools,
		accumulated_usage: Box::new(None),
		pending_remote_mcp,
		tool_search: tool_search_requested.then(|| Box::new(ToolSearchState::new(deferred_tools))),
	};
	prepared.refresh_programmatic_schema()?;
	prepared.refresh_tool_search_schema()?;
	*request = prepared.canonical_request.clone();
	Ok(Activation::Active(prepared))
}

/// The `prepare` locals a function declaration contributes to. Bundled so both the `function` and
/// `namespace` arms can share one acceptor without exceeding the argument-count lint.
struct FunctionTarget<'a> {
	seen_function_names: &'a mut HashSet<String>,
	function_names: &'a mut Vec<String>,
	managed_function_present: &'a mut bool,
	argument_schemas: &'a mut HashMap<Strng, Arc<ArgumentSchema>>,
	deferred_tools: &'a mut Vec<DeferredTool>,
	rewritten: &'a mut Vec<Value>,
}

fn accept_function_declaration(
	target: &mut FunctionTarget<'_>,
	registry: Option<&Arc<ToolRegistry>>,
	mut declaration: Value,
	label: Option<&str>,
	inherited_defer_loading: bool,
) -> Result<(), ToolRuntimeError> {
	if declaration.get("type").and_then(Value::as_str) != Some("function") {
		return Err(ToolRuntimeError::invalid_request(
			"namespace tools must be function declarations",
		));
	}
	let name = declaration
		.get("name")
		.and_then(Value::as_str)
		.ok_or_else(|| ToolRuntimeError::invalid_request("function tool requires a string name"))?;
	if name.starts_with(RESERVED_PREFIX) {
		return Err(ToolRuntimeError::invalid_request(format!(
			"function tool name {name} uses the reserved _agentgateway_ namespace"
		)));
	}
	if !target.seen_function_names.insert(name.to_owned()) {
		return Err(ToolRuntimeError::invalid_request(format!(
			"duplicate tool declaration for function {name}"
		)));
	}
	let name = Strng::from(name);
	if registry.is_some_and(|registry| registry.resolves_function(&name)) {
		*target.managed_function_present = true;
		let parameters = declaration.get("parameters").ok_or_else(|| {
			ToolRuntimeError::invalid_request("managed function parameters must be a valid JSON Schema")
		})?;
		let schema = ArgumentSchema::compile(parameters).map_err(|()| {
			ToolRuntimeError::invalid_request("managed function parameters must be a valid JSON Schema")
		})?;
		target.argument_schemas.insert(name.clone(), schema);
	}
	target.function_names.push(name.to_string());
	let defer_loading = match declaration
		.as_object_mut()
		.and_then(|declaration| declaration.remove("defer_loading"))
	{
		None | Some(Value::Null) => inherited_defer_loading,
		Some(Value::Bool(defer_loading)) => defer_loading || inherited_defer_loading,
		Some(_) => {
			return Err(ToolRuntimeError::invalid_request(
				"defer_loading must be a boolean",
			));
		},
	};
	if !defer_loading {
		target.rewritten.push(declaration);
		return Ok(());
	}
	if target.deferred_tools.len() >= MAX_DEFERRED_TOOLS {
		return Err(ToolRuntimeError::invalid_request(format!(
			"deferred tool catalog exceeds the {MAX_DEFERRED_TOOLS}-tool limit"
		)));
	}
	let description = super::truncate_utf8(
		declaration
			.get("description")
			.and_then(Value::as_str)
			.unwrap_or_default()
			.to_owned(),
		MAX_DESCRIPTION_BYTES,
	);
	target.deferred_tools.push(DeferredTool {
		internal_name: name.clone(),
		public_name: name,
		label: label.map(str::to_owned),
		description,
		declaration,
	});
	Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NamespaceDeclaration {
	#[serde(rename = "type")]
	_tool_type: String,
	name: String,
	#[serde(default)]
	description: Option<String>,
	tools: Vec<Value>,
	#[serde(default)]
	defer_loading: Option<bool>,
}

/// The function name a `tool_choice: {"type":"function","name":...}` forces, if any.
fn forced_function_name(request: &responses::Request) -> Option<&str> {
	let tool_choice = request.rest_field("tool_choice")?;
	if tool_choice.get("type").and_then(Value::as_str) != Some("function") {
		return None;
	}
	tool_choice.get("name").and_then(Value::as_str)
}

fn sanitize_client_tools(client_tools: Option<Value>) -> Option<Value> {
	let mut client_tools = client_tools?;
	if let Some(tools) = client_tools.as_array_mut() {
		for tool in tools {
			if tool.get("type").and_then(Value::as_str) == Some("mcp")
				&& let Some(tool) = tool.as_object_mut()
			{
				tool.remove("authorization");
			}
		}
	}
	Some(client_tools)
}

fn remote_mcp_options(declaration: &Value) -> Result<RemoteMcpServer, ToolRuntimeError> {
	let declaration: RemoteMcpDeclaration =
		serde_json::from_value(declaration.clone()).map_err(|error| {
			ToolRuntimeError::invalid_request(format!("invalid mcp declaration: {error}"))
		})?;
	if declaration.tool_type != "mcp" {
		return Err(ToolRuntimeError::invalid_request(
			"mcp declaration requires type mcp",
		));
	}
	if declaration.server_label.is_empty() || declaration.server_label.len() > MAX_SERVER_LABEL_BYTES
	{
		return Err(ToolRuntimeError::invalid_request(
			"mcp server_label must contain 1 to 128 bytes",
		));
	}
	if declaration
		.server_description
		.as_ref()
		.is_some_and(|description| description.len() > MAX_DESCRIPTION_BYTES)
	{
		return Err(ToolRuntimeError::invalid_request(
			"mcp server_description exceeds the 4096-byte limit",
		));
	}
	let server_url = url::Url::parse(&declaration.server_url)
		.map_err(|_| ToolRuntimeError::invalid_request("mcp server_url must be a valid HTTPS URL"))?;
	if server_url.scheme() != "https"
		|| server_url.host_str().is_none()
		|| !server_url.username().is_empty()
		|| server_url.password().is_some()
		|| server_url.fragment().is_some()
	{
		return Err(ToolRuntimeError::invalid_request(
			"mcp server_url must be a valid HTTPS URL without userinfo or a fragment",
		));
	}
	if !matches!(
		declaration
			.require_approval
			.as_ref()
			.and_then(Value::as_str),
		Some("never" | "auto")
	) {
		return Err(ToolRuntimeError::invalid_request(
			"mcp require_approval must be auto or never",
		));
	}
	if let Some(tools) = declaration.allowed_tools.as_ref() {
		let mut unique = HashSet::new();
		if tools.is_empty()
			|| tools.len() > MAX_ALLOWED_TOOLS
			|| tools.iter().any(|name| {
				name.is_empty() || name.len() > MAX_TOOL_NAME_BYTES || !unique.insert(name.as_str())
			}) {
			return Err(ToolRuntimeError::invalid_request(
				"mcp allowed_tools must contain 1 to 128 unique names of at most 128 bytes",
			));
		}
	}
	let allowed_callers = parse_allowed_callers(declaration.allowed_callers, false)?;
	let defer_loading = declaration.defer_loading.unwrap_or(false);
	if defer_loading && allowed_callers.programmatic {
		return Err(ToolRuntimeError::invalid_request(
			"defer_loading cannot be combined with programmatic allowed_callers",
		));
	}
	Ok(RemoteMcpServer {
		server_label: declaration.server_label,
		server_description: declaration.server_description,
		server_url: declaration.server_url,
		authorization: declaration.authorization.map(SecretString::from),
		allowed_tools: declaration.allowed_tools,
		allowed_callers,
		defer_loading,
	})
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RemoteMcpDeclaration {
	#[serde(rename = "type")]
	tool_type: String,
	server_label: String,
	#[serde(default)]
	server_description: Option<String>,
	server_url: String,
	#[serde(default)]
	authorization: Option<String>,
	#[serde(default)]
	allowed_tools: Option<Vec<String>>,
	#[serde(default)]
	require_approval: Option<Value>,
	#[serde(default)]
	allowed_callers: Option<Vec<AllowedCaller>>,
	#[serde(default)]
	defer_loading: Option<bool>,
}

fn declarations(request: &responses::Request) -> Result<Vec<Value>, ToolRuntimeError> {
	match request.rest_field("tools") {
		None | Some(Value::Null) => Ok(Vec::new()),
		Some(Value::Array(tools)) => tools
			.iter()
			.cloned()
			.map(|tool| {
				if tool.is_object() {
					Ok(tool)
				} else {
					Err(ToolRuntimeError::invalid_request(
						"tool declarations must be objects",
					))
				}
			})
			.collect(),
		Some(_) => Err(ToolRuntimeError::invalid_request("tools must be an array")),
	}
}

fn builtin_registry<'a>(
	registry: Option<&'a Arc<ToolRegistry>>,
	internal_name: &str,
	builtin: &str,
) -> Result<&'a Arc<ToolRegistry>, ToolRuntimeError> {
	let registry = registry.ok_or_else(|| {
		ToolRuntimeError::invalid_request(format!("builtin tool {builtin} is not configured"))
	})?;
	if !registry.has_internal(internal_name) {
		return Err(ToolRuntimeError::invalid_request(format!(
			"builtin tool {builtin} is not configured"
		)));
	}
	Ok(registry)
}

fn unsupported_active_request_fields(request: &responses::Request) -> Result<(), ToolRuntimeError> {
	for field in ["background", "conversation"] {
		if request
			.rest_field(field)
			.is_some_and(|value| !value.is_null())
			&& (field == "conversation" || request.rest_field(field) == Some(&Value::Bool(true)))
		{
			return Err(ToolRuntimeError::unsupported(field));
		}
	}
	Ok(())
}

fn web_search_options(declaration: &Value) -> Result<(Value, AllowedCallers), ToolRuntimeError> {
	let declaration: WebSearchDeclaration =
		serde_json::from_value(declaration.clone()).map_err(|error| {
			ToolRuntimeError::invalid_request(format!("invalid web_search declaration: {error}"))
		})?;
	if declaration.tool_type != "web_search" {
		return Err(ToolRuntimeError::invalid_request(
			"web_search declaration requires type web_search",
		));
	}
	let mut options = Map::new();
	match declaration.filters {
		None | Some(Nullable::Null(_)) => {},
		Some(Nullable::Value(filters)) => {
			if let Some(allowed_domains) = filters.allowed_domains {
				options.insert(
					"allowed_domains".to_owned(),
					match allowed_domains {
						Nullable::Null(_) => Value::Null,
						Nullable::Value(domains) => {
							serde_json::to_value(domains).expect("string domains serialize")
						},
					},
				);
			}
		},
	}
	match declaration.search_context_size {
		None => {},
		Some(Nullable::Null(_)) => {
			return Err(ToolRuntimeError::invalid_request(
				"web_search search_context_size must be low, medium, or high",
			));
		},
		Some(Nullable::Value(search_context_size)) => {
			options.insert(
				"search_context_size".to_owned(),
				Value::String(search_context_size.contract_value().to_owned()),
			);
		},
	}
	match declaration.user_location {
		None => {},
		Some(Nullable::Null(_)) => {
			options.insert("user_location".to_owned(), Value::Null);
		},
		Some(Nullable::Value(location)) => {
			options.insert("user_location".to_owned(), location.into_json()?);
		},
	}
	Ok((
		Value::Object(options),
		parse_allowed_callers(declaration.allowed_callers, false)?,
	))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WebSearchDeclaration {
	#[serde(rename = "type")]
	tool_type: String,
	#[serde(default)]
	filters: Option<Nullable<WebSearchFilters>>,
	#[serde(default)]
	search_context_size: Option<Nullable<SearchContextSize>>,
	#[serde(default)]
	user_location: Option<Nullable<WebSearchUserLocation>>,
	#[serde(default)]
	allowed_callers: Option<Vec<AllowedCaller>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WebSearchFilters {
	#[serde(default)]
	allowed_domains: Option<Nullable<Vec<String>>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum SearchContextSize {
	Low,
	Medium,
	High,
}

impl SearchContextSize {
	fn contract_value(self) -> &'static str {
		match self {
			Self::Low => "low",
			Self::Medium => "medium",
			Self::High => "high",
		}
	}
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WebSearchUserLocation {
	#[serde(rename = "type", default)]
	kind: Option<Nullable<WebSearchUserLocationType>>,
	#[serde(default)]
	country: Option<Nullable<String>>,
	#[serde(default)]
	region: Option<Nullable<String>>,
	#[serde(default)]
	city: Option<Nullable<String>>,
	#[serde(default)]
	timezone: Option<Nullable<String>>,
}

impl WebSearchUserLocation {
	fn into_json(self) -> Result<Value, ToolRuntimeError> {
		let kind = match self.kind {
			None | Some(Nullable::Value(WebSearchUserLocationType::Approximate)) => "approximate",
			Some(Nullable::Null(_)) => {
				return Err(ToolRuntimeError::invalid_request(
					"web_search user_location.type must be approximate",
				));
			},
		};
		let mut location = Map::from_iter([(String::from("type"), Value::String(kind.to_owned()))]);
		for (field, value) in [
			("country", self.country),
			("region", self.region),
			("city", self.city),
			("timezone", self.timezone),
		] {
			if let Some(value) = value {
				location.insert(
					field.to_owned(),
					match value {
						Nullable::Null(_) => Value::Null,
						Nullable::Value(value) => Value::String(value),
					},
				);
			}
		}
		Ok(Value::Object(location))
	}
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum WebSearchUserLocationType {
	Approximate,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum Nullable<T> {
	Value(T),
	Null(()),
}

impl<T> Default for Nullable<T> {
	fn default() -> Self {
		Self::Null(())
	}
}

fn code_interpreter_options(declaration: &Value) -> Result<AllowedCallers, ToolRuntimeError> {
	let declaration = declaration
		.as_object()
		.expect("tool declaration is an object");
	if declaration
		.keys()
		.any(|key| key != "type" && key != "container" && key != "allowed_callers")
	{
		let unsupported = declaration
			.keys()
			.find(|key| {
				key.as_str() != "type" && key.as_str() != "container" && key.as_str() != "allowed_callers"
			})
			.expect("key was found");
		return Err(ToolRuntimeError::invalid_request(format!(
			"unsupported code_interpreter option {unsupported}"
		)));
	}
	let container = declaration
		.get("container")
		.and_then(Value::as_object)
		.ok_or_else(|| {
			ToolRuntimeError::invalid_request("code_interpreter container must be { type: auto }")
		})?;
	if container.len() != 1 || container.get("type") != Some(&Value::String("auto".to_owned())) {
		return Err(ToolRuntimeError::invalid_request(
			"code_interpreter container must be { type: auto }",
		));
	}
	let callers = declaration
		.get("allowed_callers")
		.cloned()
		.map(serde_json::from_value::<Vec<AllowedCaller>>)
		.transpose()
		.map_err(|error| {
			ToolRuntimeError::invalid_request(format!(
				"invalid code_interpreter allowed_callers: {error}"
			))
		})?;
	parse_allowed_callers(callers, false)
}

pub(super) fn parse_allowed_callers(
	callers: Option<Vec<AllowedCaller>>,
	default_programmatic: bool,
) -> Result<AllowedCallers, ToolRuntimeError> {
	let Some(callers) = callers else {
		return Ok(AllowedCallers {
			direct: !default_programmatic,
			programmatic: default_programmatic,
		});
	};
	if callers.is_empty() {
		return Err(ToolRuntimeError::invalid_request(
			"allowed_callers must contain direct, programmatic, or both",
		));
	}
	let mut parsed = AllowedCallers {
		direct: false,
		programmatic: false,
	};
	for caller in callers {
		let duplicate = match caller {
			AllowedCaller::Direct => std::mem::replace(&mut parsed.direct, true),
			AllowedCaller::Programmatic => std::mem::replace(&mut parsed.programmatic, true),
		};
		if duplicate {
			return Err(ToolRuntimeError::invalid_request(
				"allowed_callers must not contain duplicates",
			));
		}
	}
	Ok(parsed)
}

fn web_search_schema() -> Value {
	json!({
	 "type": "function",
	 "name": WEB_SEARCH_FUNCTION,
	 "description": "Search the web for current information and return relevant sources.",
	 "strict": true,
	 "parameters": web_search_parameters()
	})
}

fn code_interpreter_schema() -> Value {
	json!({
		"type": "function",
		"name": CODE_INTERPRETER_FUNCTION,
		"description": "Execute Python code in an isolated sandbox and return stdout and stderr.",
		"strict": true,
		"parameters": code_interpreter_parameters()
	})
}
