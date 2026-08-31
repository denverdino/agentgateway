use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use agent_core::prelude::Strng;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{MessagesActivation, MessagesRequestExt, PreparedMessagesRuntime};
use crate::llm::tool_runtime::conversation::ManagedToolState;
use crate::llm::tool_runtime::mapper::{Nullable, parse_allowed_callers};
use crate::llm::tool_runtime::schema::{
	ArgumentSchema, code_interpreter_parameters, web_search_parameters,
};
use crate::llm::tool_runtime::{
	AllowedCaller, AllowedCallers, CODE_INTERPRETER_FUNCTION, PROGRAMMATIC_FUNCTION,
	ProgrammaticToolSpec, RuntimeDeadline, ToolRegistry, ToolRuntimeError, WEB_SEARCH_FUNCTION,
};
use crate::llm::types::messages;

const RESERVED_PREFIX: &str = "_agentgateway_";
/// The lowest code-execution version whose declaration can host Programmatic Tool Calling.
const PTC_CODE_EXECUTION_VERSIONS: &[&str] =
	&["code_execution_20260120", "code_execution_20260521"];
const DIRECT_ONLY_CODE_EXECUTION_VERSIONS: &[&str] =
	&["code_execution_20250522", "code_execution_20250825"];
/// From this Web Search version on, Anthropic defaults `allowed_callers` to the code-execution tool.
const WEB_SEARCH_PROGRAMMATIC_DEFAULT_VERSION: &str = "web_search_20260209";
const SUPPORTED_WEB_SEARCH_VERSIONS: &[&str] = &[
	"web_search_20250305",
	WEB_SEARCH_PROGRAMMATIC_DEFAULT_VERSION,
];

#[allow(dead_code)]
pub(crate) fn prepare(
	request: &mut messages::Request,
	registry: Option<&Arc<ToolRegistry>>,
) -> Result<MessagesActivation, ToolRuntimeError> {
	let Some(declarations) = request
		.rest_field("tools")
		.and_then(Value::as_array)
		.cloned()
	else {
		return Ok(MessagesActivation::Inactive);
	};

	let mut plan = Plan::default();
	for declaration in &declarations {
		classify(&mut plan, registry, declaration)?;
	}
	if !plan.activates() {
		return Ok(MessagesActivation::Inactive);
	}
	let registry = registry.ok_or_else(|| {
		ToolRuntimeError::invalid_request("managed tools require llm.toolRuntime configuration")
	})?;

	let mut rewritten = Vec::with_capacity(declarations.len());
	let mut programmatic_tools = HashMap::new();
	let mut trusted_options = HashMap::new();
	let mut argument_schemas = HashMap::new();
	for declaration in declarations {
		emit(
			&plan,
			registry,
			declaration,
			&mut rewritten,
			&mut programmatic_tools,
			&mut trusted_options,
			&mut argument_schemas,
		)?;
	}
	reject_unsupported_request_fields(request)?;
	validate_tool_choice(request, &plan)?;

	let parallel = !request
		.rest_field("tool_choice")
		.and_then(|choice| choice.get("disable_parallel_tool_use"))
		.and_then(Value::as_bool)
		.unwrap_or(false);
	let client_streaming = request.stream == Some(true);
	if client_streaming {
		request.stream = Some(false);
	}

	let prepared_registry = Arc::new(registry.with_request_data(trusted_options, argument_schemas));
	let deadline = RuntimeDeadline::new(prepared_registry.limits.total_timeout);
	let mut state = ManagedToolState {
		registry: prepared_registry,
		parallel,
		deadline,
		programmatic_requested: plan.programmatic_runtime_needed(),
		programmatic_tools: Arc::new(HashMap::new()),
		programmatic_catalog_bytes: 0,
	};
	for spec in programmatic_tools.into_values() {
		state.insert_programmatic_tool(spec)?;
	}
	if let Some(catalog) = state.programmatic_catalog_json()? {
		rewritten.push(program_runtime_declaration(&catalog));
	}
	request.replace_rest_field("tools", Value::Array(rewritten));

	Ok(MessagesActivation::Active(Box::new(
		PreparedMessagesRuntime {
			state,
			canonical_request: request.clone(),
			client_streaming,
			accumulated_usage: None,
		},
	)))
}

#[derive(Default)]
struct Plan {
	code_execution: Option<CodeExecution>,
	code_execution_present: bool,
	web_search_present: bool,
	managed_function_present: bool,
	programmatic_callers_requested: bool,
	programmatic_only: HashSet<String>,
	seen_names: HashSet<String>,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum CodeExecution {
	ProgramCapable,
	DirectOnly,
}

impl Plan {
	fn activates(&self) -> bool {
		self.code_execution.is_some() || self.web_search_present || self.managed_function_present
	}

	fn programmatic_runtime_needed(&self) -> bool {
		self.code_execution == Some(CodeExecution::ProgramCapable)
			&& self.programmatic_callers_requested
	}
}

fn classify(
	plan: &mut Plan,
	registry: Option<&Arc<ToolRegistry>>,
	declaration: &Value,
) -> Result<(), ToolRuntimeError> {
	let name = declaration.get("name").and_then(Value::as_str);
	if let Some(name) = name
		&& !plan.seen_names.insert(name.to_owned())
	{
		return Err(ToolRuntimeError::invalid_request(format!(
			"duplicate tool declaration for {name}"
		)));
	}
	match declaration.get("type").and_then(Value::as_str) {
		Some(version) if PTC_CODE_EXECUTION_VERSIONS.contains(&version) => {
			require_declaration_name(name, "code_execution")?;
			if plan.code_execution_present {
				return Err(ToolRuntimeError::invalid_request(
					"duplicate tool declaration for code_execution",
				));
			}
			plan.code_execution_present = true;
			plan.code_execution = Some(CodeExecution::ProgramCapable);
		},
		Some(version) if DIRECT_ONLY_CODE_EXECUTION_VERSIONS.contains(&version) => {
			require_declaration_name(name, "code_execution")?;
			if plan.code_execution_present {
				return Err(ToolRuntimeError::invalid_request(
					"duplicate tool declaration for code_execution",
				));
			}
			plan.code_execution_present = true;
			plan.code_execution = Some(CodeExecution::DirectOnly);
		},
		Some(version) if SUPPORTED_WEB_SEARCH_VERSIONS.contains(&version) => {
			require_declaration_name(name, "web_search")?;
			if plan.web_search_present {
				return Err(ToolRuntimeError::invalid_request(
					"duplicate tool declaration for web_search",
				));
			}
			plan.web_search_present = true;
			let callers = callers(declaration, web_search_defaults_programmatic(version))?;
			if callers.programmatic {
				plan.programmatic_callers_requested = true;
				if !callers.direct {
					plan.programmatic_only.insert("web_search".to_owned());
				}
			}
		},
		None | Some("custom") => {
			let name = name
				.ok_or_else(|| ToolRuntimeError::invalid_request("custom tool requires a string name"))?;
			if registry.is_some_and(|registry| registry.resolves_function(name)) {
				plan.managed_function_present = true;
				let callers = callers(declaration, false)?;
				if callers.programmatic {
					plan.programmatic_callers_requested = true;
					if !callers.direct {
						plan.programmatic_only.insert(name.to_owned());
					}
				}
			}
		},
		Some(_) => {},
	}
	Ok(())
}

fn require_declaration_name(name: Option<&str>, expected: &str) -> Result<(), ToolRuntimeError> {
	if name == Some(expected) {
		return Ok(());
	}
	Err(ToolRuntimeError::invalid_request(format!(
		"built-in tool declaration requires name {expected}"
	)))
}

fn emit(
	plan: &Plan,
	registry: &Arc<ToolRegistry>,
	declaration: Value,
	rewritten: &mut Vec<Value>,
	programmatic_tools: &mut HashMap<Strng, ProgrammaticToolSpec>,
	trusted_options: &mut HashMap<Strng, Value>,
	argument_schemas: &mut HashMap<Strng, Arc<ArgumentSchema>>,
) -> Result<(), ToolRuntimeError> {
	let kind = declaration.get("type").and_then(Value::as_str);
	match kind {
		Some(version)
			if PTC_CODE_EXECUTION_VERSIONS.contains(&version)
				|| DIRECT_ONLY_CODE_EXECUTION_VERSIONS.contains(&version) =>
		{
			require_builtin(registry, CODE_INTERPRETER_FUNCTION, "code execution")?;
			rewritten.push(json!({
				"name": CODE_INTERPRETER_FUNCTION,
				"description": "Execute Python code in an isolated sandbox and return stdout and stderr.",
				"input_schema": code_interpreter_parameters(),
			}));
			Ok(())
		},
		Some(version) if SUPPORTED_WEB_SEARCH_VERSIONS.contains(&version) => {
			require_builtin(registry, WEB_SEARCH_FUNCTION, "web search")?;
			let callers = callers(&declaration, web_search_defaults_programmatic(version))?;
			trusted_options.insert(
				Strng::from(WEB_SEARCH_FUNCTION),
				web_search_trusted_options(&declaration)?,
			);
			if callers.programmatic {
				if !plan.programmatic_runtime_needed() {
					return Err(ToolRuntimeError::invalid_request(
						"web search allowed_callers names a code execution version, but no code execution tool is declared; set allowed_callers to [\"direct\"]",
					));
				}
				programmatic_tools.insert(
					Strng::from("web_search"),
					ProgrammaticToolSpec {
						public_name: Strng::from("web_search"),
						internal_name: Strng::from(WEB_SEARCH_FUNCTION),
						description: "Search the web for current information and return relevant sources."
							.to_owned(),
						input_schema: web_search_parameters(),
						output_schema: None,
					},
				);
			}
			if callers.direct {
				rewritten.push(json!({
					"name": WEB_SEARCH_FUNCTION,
					"description": "Search the web for current information and return relevant sources.",
					"input_schema": web_search_parameters(),
				}));
			}
			Ok(())
		},
		None | Some("custom") => {
			let mut declaration = declaration;
			let name = Strng::from(
				declaration
					.get("name")
					.and_then(Value::as_str)
					.ok_or_else(|| ToolRuntimeError::invalid_request("custom tool requires a string name"))?,
			);
			if name.starts_with(RESERVED_PREFIX) {
				return Err(ToolRuntimeError::invalid_request(format!(
					"tool name {name} uses the reserved _agentgateway_ namespace"
				)));
			}
			if !registry.resolves_function(&name) {
				return Err(ToolRuntimeError::invalid_request(format!(
					"every custom tool must be registered when the managed tool runtime is active; {name} is not registered"
				)));
			}
			let input_schema = declaration.get("input_schema").ok_or_else(|| {
				ToolRuntimeError::invalid_request("managed tool input_schema must be a valid JSON Schema")
			})?;
			reject_recursive_schema_refs(input_schema)?;
			let schema = ArgumentSchema::compile(input_schema).map_err(|()| {
				ToolRuntimeError::invalid_request("managed tool input_schema must be a valid JSON Schema")
			})?;
			argument_schemas.insert(name.clone(), schema);
			let callers = callers(&declaration, false)?;
			if callers.programmatic {
				if !plan.programmatic_runtime_needed() {
					return Err(ToolRuntimeError::invalid_request(format!(
						"tool {name} allowed_callers names a code execution version, but no code execution tool is declared"
					)));
				}
				programmatic_tools.insert(
					name.clone(),
					ProgrammaticToolSpec {
						public_name: name.clone(),
						internal_name: name.clone(),
						description: declaration
							.get("description")
							.and_then(Value::as_str)
							.unwrap_or_default()
							.to_owned(),
						input_schema: input_schema.clone(),
						output_schema: declaration.get("output_schema").cloned(),
					},
				);
			}
			if callers.direct {
				if let Some(object) = declaration.as_object_mut() {
					object.remove("allowed_callers");
				}
				rewritten.push(declaration);
			}
			Ok(())
		},
		Some(kind) => Err(ToolRuntimeError::invalid_request(format!(
			"tool type {kind} is not supported while the managed tool runtime is active"
		))),
	}
}

fn program_runtime_declaration(catalog: &str) -> Value {
	json!({
		"name": PROGRAMMATIC_FUNCTION,
		"description": format!(
			"Write one Python program. Call authorized tools sequentially with tools.call(name, arguments) and finish with program_output(value). Do not call these tools directly. For a catalog entry with output_schema, output_schema describes the structuredContent field in the successful object returned by tools.call. Available tools: {catalog}"
		),
		"input_schema": {
			"type": "object",
			"properties": {"code": {"type": "string"}},
			"required": ["code"],
			"additionalProperties": false
		}
	})
}

fn web_search_defaults_programmatic(version: &str) -> bool {
	version == WEB_SEARCH_PROGRAMMATIC_DEFAULT_VERSION
}

fn callers(
	declaration: &Value,
	default_programmatic: bool,
) -> Result<AllowedCallers, ToolRuntimeError> {
	#[derive(Deserialize)]
	struct Callers {
		#[serde(default)]
		allowed_callers: Option<Nullable<Vec<String>>>,
	}
	let declared: Callers = serde_json::from_value(declaration.clone())
		.map_err(|error| ToolRuntimeError::invalid_request(format!("invalid tool: {error}")))?;
	let normalized = match declared.allowed_callers {
		None => None,
		Some(Nullable::Null(_)) => {
			return Err(ToolRuntimeError::invalid_request(
				"allowed_callers must contain direct, programmatic, or both",
			));
		},
		Some(Nullable::Value(callers)) => Some(
			callers
				.into_iter()
				.map(|caller| match caller.as_str() {
					"code_execution_20260521" => "programmatic".to_owned(),
					"code_execution_20260120" => "programmatic".to_owned(),
					other => other.to_owned(),
				})
				.collect::<Vec<_>>(),
		),
	};
	let parsed = normalized
		.map(|callers| {
			callers
				.into_iter()
				.map(|caller| serde_json::from_value::<AllowedCaller>(Value::String(caller)))
				.collect::<Result<Vec<_>, _>>()
		})
		.transpose()
		.map_err(|error| {
			ToolRuntimeError::invalid_request(format!("invalid tool allowed_callers: {error}"))
		})?;
	parse_allowed_callers(parsed, default_programmatic)
}

fn web_search_trusted_options(declaration: &Value) -> Result<Value, ToolRuntimeError> {
	#[derive(Deserialize)]
	struct WebSearchDeclaration {
		#[serde(default)]
		allowed_domains: Nullable<Vec<String>>,
		#[serde(default)]
		blocked_domains: Nullable<Vec<String>>,
		#[serde(default)]
		user_location: Nullable<WebSearchUserLocation>,
		#[serde(default)]
		max_uses: Nullable<u64>,
	}
	#[derive(Deserialize)]
	struct WebSearchUserLocation {
		#[serde(rename = "type")]
		kind: WebSearchUserLocationType,
		#[serde(default)]
		country: Option<String>,
		#[serde(default)]
		region: Option<String>,
		#[serde(default)]
		city: Option<String>,
		#[serde(default)]
		timezone: Option<String>,
	}
	#[derive(Deserialize)]
	#[serde(rename_all = "snake_case")]
	enum WebSearchUserLocationType {
		Approximate,
	}

	let declared: WebSearchDeclaration = serde_json::from_value(declaration.clone())
		.map_err(|error| ToolRuntimeError::invalid_request(format!("invalid web search: {error}")))?;
	let mut options = serde_json::Map::new();
	for (field, value) in [
		("allowed_domains", declared.allowed_domains),
		("blocked_domains", declared.blocked_domains),
	] {
		match value {
			Nullable::Value(domains) => {
				options.insert(field.to_owned(), json!(domains));
			},
			Nullable::Null(_) if declaration.get(field).is_some() => {
				return Err(ToolRuntimeError::invalid_request(format!(
					"invalid web search: {field} must be an array of strings"
				)));
			},
			Nullable::Null(_) => {},
		}
	}
	match declared.user_location {
		Nullable::Null(_) if declaration.get("user_location").is_some() => {
			return Err(ToolRuntimeError::invalid_request(
				"invalid web search: user_location must be an approximate location object",
			));
		},
		Nullable::Null(_) => {},
		Nullable::Value(location) => {
			let WebSearchUserLocationType::Approximate = location.kind;
			let mut value =
				serde_json::Map::from_iter([("type".to_owned(), Value::String("approximate".to_owned()))]);
			for (field, member) in [
				("country", location.country),
				("region", location.region),
				("city", location.city),
				("timezone", location.timezone),
			] {
				if let Some(member) = member {
					value.insert(field.to_owned(), Value::String(member));
				}
			}
			options.insert("user_location".to_owned(), Value::Object(value));
		},
	}
	match declared.max_uses {
		Nullable::Value(max_uses) => {
			options.insert("max_uses".to_owned(), json!(max_uses));
		},
		Nullable::Null(_) if declaration.get("max_uses").is_some() => {
			return Err(ToolRuntimeError::invalid_request(
				"invalid web search: max_uses must be a non-negative integer",
			));
		},
		Nullable::Null(_) => {},
	}
	Ok(Value::Object(options))
}

const MAX_LOCAL_REF_DEPTH: usize = 128;

fn schema_invalid() -> ToolRuntimeError {
	ToolRuntimeError::invalid_request("managed tool input_schema must be a valid JSON Schema")
}

fn reject_recursive_schema_refs(schema: &Value) -> Result<(), ToolRuntimeError> {
	walk_schema_refs(schema, schema, 0, &mut Vec::new())
}

fn walk_schema_refs(
	root: &Value,
	node: &Value,
	depth: usize,
	resolving: &mut Vec<String>,
) -> Result<(), ToolRuntimeError> {
	if depth > MAX_LOCAL_REF_DEPTH {
		return Err(schema_invalid());
	}

	if let Some(reference) = node.get("$ref").and_then(Value::as_str)
		&& let Some(target) = reference.strip_prefix('#')
	{
		if !target.is_empty() && !target.starts_with('/') {
			return Err(schema_invalid());
		}
		let target_node = root.pointer(target).ok_or_else(schema_invalid)?;
		let target = target.to_owned();
		if resolving.iter().any(|ancestor| ancestor == &target) {
			return Err(schema_invalid());
		}
		resolving.push(target);
		walk_schema_refs(root, target_node, depth + 1, resolving)?;
		resolving.pop();
	}

	match node {
		Value::Array(items) => {
			for item in items {
				walk_schema_refs(root, item, depth + 1, resolving)?;
			}
		},
		Value::Object(object) => {
			for value in object.values() {
				walk_schema_refs(root, value, depth + 1, resolving)?;
			}
		},
		_ => {},
	}
	Ok(())
}

fn require_builtin(
	registry: &Arc<ToolRegistry>,
	internal_name: &str,
	label: &str,
) -> Result<(), ToolRuntimeError> {
	if registry.has_internal(internal_name) {
		return Ok(());
	}
	Err(ToolRuntimeError::invalid_request(format!(
		"{label} requires a configured {label} backend in llm.toolRuntime"
	)))
}

/// Request-level fields the managed runtime cannot honor.
const UNSUPPORTED_ACTIVE_FIELDS: &[&str] = &["container", "mcp_servers"];

fn reject_unsupported_request_fields(request: &messages::Request) -> Result<(), ToolRuntimeError> {
	for field in UNSUPPORTED_ACTIVE_FIELDS {
		if request
			.rest_field(field)
			.is_some_and(|value| !value.is_null())
		{
			return Err(ToolRuntimeError::invalid_request(format!(
				"{field} is not supported while the managed tool runtime is active"
			)));
		}
	}
	Ok(())
}

fn validate_tool_choice(
	request: &mut messages::Request,
	plan: &Plan,
) -> Result<(), ToolRuntimeError> {
	let Some(choice) = request.rest_field("tool_choice").cloned() else {
		return Ok(());
	};
	if choice.is_null() {
		return Ok(());
	}
	// Anthropic rejects this combination, and honoring it would serialize the program's own
	// sequential tool calls behind a flag the program cannot see.
	if plan.programmatic_runtime_needed()
		&& choice
			.get("disable_parallel_tool_use")
			.and_then(Value::as_bool)
			.unwrap_or(false)
	{
		return Err(ToolRuntimeError::invalid_request(
			"tool_choice.disable_parallel_tool_use cannot be combined with programmatic tool calling",
		));
	}
	let Some(choice_type) = choice.get("type").and_then(Value::as_str) else {
		return Err(ToolRuntimeError::invalid_request(
			"tool_choice type must be auto, any, none, or tool",
		));
	};
	match choice_type {
		"auto" | "any" | "none" => Ok(()),
		"tool" => {
			let Some(name) = choice.get("name").and_then(Value::as_str) else {
				return Err(ToolRuntimeError::invalid_request(
					"tool_choice tool requires a string name",
				));
			};
			// A forced tool must reach the model, so it cannot be one we withheld.
			if plan.programmatic_only.contains(name) {
				return Err(ToolRuntimeError::invalid_request(format!(
					"tool_choice cannot force {name}, whose allowed_callers omits direct"
				)));
			}
			if name == "code_execution" {
				if plan.code_execution.is_some() {
					let mut rewritten = choice;
					if let Some(object) = rewritten.as_object_mut() {
						object.insert(
							"name".to_owned(),
							Value::String(CODE_INTERPRETER_FUNCTION.to_owned()),
						);
					}
					request.replace_rest_field("tool_choice", rewritten);
					return Ok(());
				}
				return Err(ToolRuntimeError::invalid_request(
					"tool_choice names code_execution, but no code execution tool is declared",
				));
			}
			if name == WEB_SEARCH_FUNCTION || name == PROGRAMMATIC_FUNCTION {
				return Err(ToolRuntimeError::invalid_request(format!(
					"tool_choice cannot name reserved tool {name}"
				)));
			}
			Ok(())
		},
		_ => Err(ToolRuntimeError::invalid_request(
			"tool_choice type must be auto, any, none, or tool",
		)),
	}
}
