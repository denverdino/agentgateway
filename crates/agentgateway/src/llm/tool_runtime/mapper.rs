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
	Activation, AllowedCaller, AllowedCallers, CODE_INTERPRETER_FUNCTION, PreparedToolRuntime,
	ProgrammaticToolSpec, RemoteMcpServer, ResponsesRequestExt, ToolRegistry, ToolRuntimeError,
	WEB_SEARCH_FUNCTION,
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
	let mut seen_mcp_labels = HashSet::new();
	let mut pending_remote_mcp = Vec::new();
	let mut rewritten = Vec::with_capacity(declarations.len());
	let mut trusted_options = HashMap::new();
	let mut argument_schemas = HashMap::new();
	let mut programmatic_tools = Vec::new();

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
			"function" => {
				let name = declaration
					.get("name")
					.and_then(Value::as_str)
					.ok_or_else(|| {
						ToolRuntimeError::invalid_request("function tool requires a string name")
					})?;
				if name.starts_with(RESERVED_PREFIX) {
					return Err(ToolRuntimeError::invalid_request(format!(
						"function tool name {name} uses the reserved _agentgateway_ namespace"
					)));
				}
				if !seen_function_names.insert(name.to_owned()) {
					return Err(ToolRuntimeError::invalid_request(format!(
						"duplicate tool declaration for function {name}"
					)));
				}
				if registry.is_some_and(|registry| registry.resolves_function(name)) {
					managed_function_present = true;
					let parameters = declaration.get("parameters").ok_or_else(|| {
						ToolRuntimeError::invalid_request(
							"managed function parameters must be a valid JSON Schema",
						)
					})?;
					let schema = ArgumentSchema::compile(parameters).map_err(|()| {
						ToolRuntimeError::invalid_request(
							"managed function parameters must be a valid JSON Schema",
						)
					})?;
					argument_schemas.insert(Strng::from(name), schema);
				}
				function_names.push(name.to_owned());
				rewritten.push(declaration);
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
			_ => rewritten.push(declaration),
		}
	}

	if !managed_function_present && !builtin_present && !remote_mcp_present && !programmatic_requested
	{
		return Ok(Activation::Inactive);
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
	let client_tools = sanitize_client_tools(client_tools);
	let prepared_registry = Arc::new(registry.with_request_data(trusted_options, argument_schemas));
	let deadline = super::RuntimeDeadline::new(prepared_registry.limits.total_timeout);
	let programmatic_tools = programmatic_tools
		.into_iter()
		.map(|tool| (tool.public_name.clone(), tool))
		.collect();
	let programmatic_catalog_bytes = super::programmatic_catalog_bytes(&programmatic_tools)?;
	let mut prepared = PreparedToolRuntime {
		registry: prepared_registry,
		canonical_request: request.clone(),
		programmatic_requested,
		programmatic_tools: Arc::new(programmatic_tools),
		programmatic_catalog_bytes,
		parallel,
		client_streaming,
		include_obfuscation,
		client_tools,
		deadline,
		pending_remote_mcp,
	};
	prepared.refresh_programmatic_schema()?;
	*request = prepared.canonical_request.clone();
	Ok(Activation::Active(prepared))
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
	Ok(RemoteMcpServer {
		server_label: declaration.server_label,
		server_description: declaration.server_description,
		server_url: declaration.server_url,
		authorization: declaration.authorization.map(SecretString::from),
		allowed_tools: declaration.allowed_tools,
		allowed_callers: parse_allowed_callers(declaration.allowed_callers)?,
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
		parse_allowed_callers(declaration.allowed_callers)?,
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
enum Nullable<T> {
	Value(T),
	Null(()),
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
	parse_allowed_callers(callers)
}

fn parse_allowed_callers(
	callers: Option<Vec<AllowedCaller>>,
) -> Result<AllowedCallers, ToolRuntimeError> {
	let Some(callers) = callers else {
		return Ok(AllowedCallers::default());
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
