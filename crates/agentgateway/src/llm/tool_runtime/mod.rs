//! Operator-owned configuration and bounded execution for the Responses tool runtime.

mod backend;
mod config;
mod e2b;
mod http_backend;
mod mapper;
mod program;
mod registry;
mod remote_mcp;
pub(crate) mod runner;
mod schema;
mod telemetry;
mod transport;
mod validation;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_core::prelude::Strng;
use agent_llm::types::responses;
pub(crate) use backend::{
	ManagedToolCall, SandboxCleanupOutcome, SandboxOperationReport, ToolApplicationError,
	ToolBackend, ToolBatchExecution, ToolBatchInfrastructureError, ToolBatchMetadata,
	ToolExecutionContext, ToolExecutionResult, ToolInfrastructureError, bound_replay_output,
	parse_arguments,
};
pub use config::{
	BuiltinTool, ManagedToolConfig, RuntimeLimits, ToolBackendConfig, ToolRuntimeConfig,
};
pub(crate) use e2b::E2bSandboxBackend;
use futures_util::stream::{FuturesUnordered, StreamExt};
pub(crate) use http_backend::HttpToolBackend;
pub(crate) use mapper::prepare;
pub(crate) use program::{
	ProgramOutcome, ProgramReplayEntry, ProgramSandbox, ProgramSandboxExecution,
	ProgramSandboxRequest, build_sandbox_request, fit_replay_entry, has_replay_entry_capacity,
	parse_sandbox_outcome, program_protocol_stdout_max_bytes,
};
use rand::RngExt;
pub(crate) use registry::ToolRegistry;
pub(crate) use remote_mcp::{RemoteMcpBackend, RemoteMcpTool};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
pub(crate) use telemetry::ToolExecutionRecord;
pub use telemetry::{
	SandboxOperation, SandboxOperationDurationLabels, SandboxOperationLabels,
	SandboxOperationOutcome, ToolBackendLabel, ToolCallDurationLabels, ToolCallLabels,
	ToolExecutionOutcome, ToolLabels, ToolRuntimeLimit, ToolRuntimeLimitLabels, ToolRuntimeOutcome,
	ToolRuntimeOutcomeLabels,
};

use self::telemetry::ToolRuntimeTelemetry;
use self::transport::truncate_utf8_bytes;
use crate::http::Response;
use crate::proxy::httpproxy::PolicyClient;

trait ResponsesRequestExt {
	fn rest_field(&self, name: &str) -> Option<&Value>;
	fn replace_rest_field(&mut self, name: &str, value: Value);
}

impl ResponsesRequestExt for responses::Request {
	fn rest_field(&self, name: &str) -> Option<&Value> {
		self.rest.as_object().and_then(|rest| rest.get(name))
	}

	fn replace_rest_field(&mut self, name: &str, value: Value) {
		if let Some(rest) = self.rest.as_object_mut() {
			rest.insert(name.to_owned(), value);
		} else {
			let mut rest = serde_json::Map::new();
			rest.insert(name.to_owned(), value);
			self.rest = Value::Object(rest);
		}
	}
}

fn aggregate_usage(aggregate: &mut Option<responses::Usage>, round: &responses::Usage) {
	let Some(current) = aggregate.as_mut() else {
		*aggregate = Some(round.clone());
		return;
	};

	current.input_tokens = current.input_tokens.saturating_add(round.input_tokens);
	current.output_tokens = current.output_tokens.saturating_add(round.output_tokens);
	current.total_tokens = Some(current.input_tokens.saturating_add(current.output_tokens));

	match (
		&mut current.input_tokens_details,
		&round.input_tokens_details,
	) {
		(None, None) | (Some(_), None) => {},
		(None, Some(round)) => current.input_tokens_details = Some(round.clone()),
		(Some(current), Some(round)) => {
			current.cached_tokens = saturating_add_optional(current.cached_tokens, round.cached_tokens);
			current.cache_write_tokens =
				saturating_add_optional(current.cache_write_tokens, round.cache_write_tokens);
		},
	}

	match (
		&mut current.output_tokens_details,
		&round.output_tokens_details,
	) {
		(None, None) | (Some(_), None) => {},
		(None, Some(round)) => current.output_tokens_details = Some(round.clone()),
		(Some(current), Some(round)) => {
			current.reasoning_tokens =
				saturating_add_optional(current.reasoning_tokens, round.reasoning_tokens);
		},
	}
}

fn saturating_add_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
	match (left, right) {
		(None, None) => None,
		(left, right) => Some(
			left
				.unwrap_or_default()
				.saturating_add(right.unwrap_or_default()),
		),
	}
}

pub(crate) const WEB_SEARCH_FUNCTION: &str = "_agentgateway_web_search";
pub(crate) const CODE_INTERPRETER_FUNCTION: &str = "_agentgateway_code_interpreter";
pub(crate) const PROGRAMMATIC_FUNCTION: &str = "_agentgateway_programmatic_tool_calling";
pub(crate) const REMOTE_MCP_FUNCTION_PREFIX: &str = "_agentgateway_mcp_";
pub(crate) const SANDBOX_MAX_BATCH_EXECUTIONS: usize = 8;
pub(crate) const SANDBOX_MAX_CALL_ID_BYTES: usize = 256;
pub(crate) const SANDBOX_MAX_CODE_BYTES: usize = 32 * 1024;
pub(crate) const SANDBOX_MAX_BATCH_DEADLINE: Duration = Duration::from_secs(24 * 60 * 60);
const PROGRAMMATIC_MAX_CATALOG_BYTES: usize = remote_mcp::MAX_DISCOVERY_BYTES;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub(super) enum AllowedCaller {
	Direct,
	Programmatic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AllowedCallers {
	pub(super) direct: bool,
	pub(super) programmatic: bool,
}

impl Default for AllowedCallers {
	fn default() -> Self {
		Self {
			direct: true,
			programmatic: false,
		}
	}
}

#[derive(Clone, Debug)]
pub(crate) struct ProgrammaticToolSpec {
	pub(crate) public_name: Strng,
	pub(crate) internal_name: Strng,
	pub(crate) description: String,
	pub(crate) input_schema: Value,
	pub(crate) output_schema: Option<Value>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RuntimeDeadline(tokio::time::Instant);

impl RuntimeDeadline {
	pub(crate) fn new(timeout: Duration) -> Self {
		Self(tokio::time::Instant::now() + timeout)
	}

	pub(crate) fn remaining(self) -> Duration {
		self
			.0
			.saturating_duration_since(tokio::time::Instant::now())
	}

	pub(crate) fn instant(self) -> tokio::time::Instant {
		self.0
	}
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedToolRuntime {
	pub(crate) registry: Arc<ToolRegistry>,
	pub(crate) canonical_request: responses::Request,
	programmatic_requested: bool,
	programmatic_tools: Arc<HashMap<Strng, ProgrammaticToolSpec>>,
	programmatic_catalog_bytes: usize,
	pub(crate) parallel: bool,
	pub(crate) client_streaming: bool,
	pub(crate) include_obfuscation: bool,
	pub(crate) client_tools: Option<Value>,
	pub(crate) deadline: RuntimeDeadline,
	pub(crate) pending_remote_mcp: Vec<RemoteMcpServer>,
}

#[derive(Debug)]
pub(crate) enum CollectedToolCalls {
	Direct(Vec<ManagedToolCall>),
	Programmatic { call_id: Strng, code: String },
}

#[derive(Clone, Debug)]
pub(crate) struct RemoteMcpServer {
	pub server_label: String,
	pub server_description: Option<String>,
	pub server_url: String,
	pub authorization: Option<secrecy::SecretString>,
	pub allowed_tools: Option<Vec<String>>,
	pub allowed_callers: AllowedCallers,
}

impl PreparedToolRuntime {
	pub(crate) fn append_round_history(&mut self, raw_output: Vec<Value>, outputs: Vec<Value>) {
		self.canonical_request.append_raw_input_values(raw_output);
		self.canonical_request.append_raw_input_values(outputs);
		if self.canonical_request.rest_field("tool_choice") != Some(&json!("auto")) {
			self
				.canonical_request
				.replace_rest_field("tool_choice", json!("auto"));
		}
	}

	pub(crate) async fn initialize_remote_mcp(
		&mut self,
		client: PolicyClient,
		extensions: &::http::Extensions,
		deadline: RuntimeDeadline,
	) -> Result<(), ToolRuntimeError> {
		for server_index in 0..self.pending_remote_mcp.len() {
			let server = self.pending_remote_mcp[server_index].clone();
			let (backend, tools) =
				RemoteMcpBackend::connect(client.clone(), extensions, &server, deadline)
					.await
					.map_err(ToolRuntimeError::infrastructure)?;
			self.install_remote_mcp_tools(server_index, backend, tools)?;
		}
		self.pending_remote_mcp.clear();
		self.refresh_programmatic_schema()?;
		if self
			.canonical_request
			.rest_field("tool_choice")
			.and_then(Value::as_object)
			.is_some_and(|choice| choice.get("type").and_then(Value::as_str) == Some("mcp"))
		{
			return Err(ToolRuntimeError::invalid_request(
				"mcp tool_choice does not match an imported tool",
			));
		}
		Ok(())
	}

	fn install_remote_mcp_tools(
		&mut self,
		server_index: usize,
		backend: Arc<dyn ToolBackend>,
		tools: Vec<RemoteMcpTool>,
	) -> Result<(), ToolRuntimeError> {
		let server = self.pending_remote_mcp.get(server_index).ok_or_else(|| {
			ToolRuntimeError::invalid_configuration("remote MCP server index is invalid")
		})?;
		let server_label = server.server_label.clone();
		let server_description = server.server_description.clone();
		let allowed_callers = server.allowed_callers;
		let mut declarations = self
			.canonical_request
			.rest_field("tools")
			.and_then(Value::as_array)
			.cloned()
			.unwrap_or_default();
		let mut registry = (*self.registry).clone();
		for (tool_index, tool) in tools.into_iter().enumerate() {
			let internal_name = Strng::from(format!(
				"{REMOTE_MCP_FUNCTION_PREFIX}{server_index}_{tool_index}"
			));
			let public_name = Strng::from(format!("{}.{}", server_label, tool.remote_name));
			let argument_schema = schema::ArgumentSchema::compile(&tool.input_schema)
				.map_err(|()| ToolRuntimeError::infrastructure(ToolInfrastructureError::backend()))?;
			let description = match (&server_description, tool.description) {
				(Some(server_description), Some(tool_description)) => format!(
					"[{}] {server_description}\n\n{tool_description}",
					server_label
				),
				(Some(description), None) => format!("[{}] {description}", server_label),
				(None, Some(description)) => format!("[{}] {description}", server_label),
				(None, None) => format!("Tool from remote MCP server {}", server_label),
			};
			let description = truncate_utf8(description, remote_mcp::MAX_DESCRIPTION_BYTES);
			if allowed_callers.direct {
				declarations.push(json!({
					"type": "function",
					"name": internal_name,
					"description": description,
					"strict": false,
					"parameters": tool.input_schema,
				}));
			}
			if allowed_callers.programmatic {
				if self.programmatic_tools.contains_key(public_name.as_str()) {
					return Err(ToolRuntimeError::invalid_request(
						"duplicate programmatic tool name",
					));
				}
				let spec = ProgrammaticToolSpec {
					public_name: public_name.clone(),
					internal_name: internal_name.clone(),
					description: description.clone(),
					input_schema: tool.input_schema.clone(),
					output_schema: tool.output_schema.clone(),
				};
				let next_size = checked_programmatic_catalog_add(
					self.programmatic_catalog_bytes,
					!self.programmatic_tools.is_empty(),
					&spec,
				)?;
				Arc::make_mut(&mut self.programmatic_tools).insert(public_name.clone(), spec);
				self.programmatic_catalog_bytes = next_size;
			}
			let forced_mcp_tool = self
				.canonical_request
				.rest_field("tool_choice")
				.and_then(Value::as_object)
				.filter(|choice| choice.get("type").and_then(Value::as_str) == Some("mcp"))
				.filter(|choice| {
					choice.get("server_label").and_then(Value::as_str) == Some(server_label.as_str())
				})
				.and_then(|choice| choice.get("name").and_then(Value::as_str))
				== Some(tool.remote_name.as_str());
			if forced_mcp_tool && allowed_callers.direct {
				self.canonical_request.replace_rest_field(
					"tool_choice",
					json!({"type": "function", "name": internal_name}),
				);
			}
			registry = registry.with_remote_tool(
				internal_name,
				public_name,
				argument_schema,
				json!({"remote_tool_name": tool.remote_name}),
				backend.clone(),
			)?;
		}
		self
			.canonical_request
			.replace_rest_field("tools", Value::Array(declarations));
		self.registry = Arc::new(registry);
		Ok(())
	}

	#[cfg(test)]
	pub(crate) fn install_remote_mcp_tools_for_test(
		&mut self,
		server_index: usize,
		backend: Arc<dyn ToolBackend>,
		tools: Vec<RemoteMcpTool>,
	) -> Result<(), ToolRuntimeError> {
		self.install_remote_mcp_tools(server_index, backend, tools)?;
		self.refresh_programmatic_schema()
	}

	pub(crate) fn refresh_programmatic_schema(&mut self) -> Result<(), ToolRuntimeError> {
		let mut declarations = self
			.canonical_request
			.rest_field("tools")
			.and_then(Value::as_array)
			.cloned()
			.unwrap_or_default();
		declarations
			.retain(|tool| tool.get("name").and_then(Value::as_str) != Some(PROGRAMMATIC_FUNCTION));
		if !self.programmatic_requested {
			self
				.canonical_request
				.replace_rest_field("tools", Value::Array(declarations));
			return Ok(());
		}
		if self.programmatic_tools.is_empty() {
			if self.pending_remote_mcp.is_empty() {
				return Err(ToolRuntimeError::invalid_request(
					"programmatic_tool_calling requires at least one programmatic tool",
				));
			}
			self
				.canonical_request
				.replace_rest_field("tools", Value::Array(declarations));
			return Ok(());
		}

		let mut catalog = self.programmatic_tools.values().collect::<Vec<_>>();
		catalog.sort_by(|left, right| left.public_name.cmp(&right.public_name));
		let catalog = catalog
			.into_iter()
			.map(programmatic_catalog_entry)
			.collect::<Vec<_>>();
		let catalog = serde_json::to_string(&catalog).map_err(|_| ToolRuntimeError::internal())?;
		if catalog.len() > PROGRAMMATIC_MAX_CATALOG_BYTES {
			return Err(ToolRuntimeError::invalid_request(
				"programmatic tool catalog exceeds the 2097152-byte limit",
			));
		}
		declarations.push(json!({
			"type": "function",
			"name": PROGRAMMATIC_FUNCTION,
			"description": format!(
				"Write one Python program. Call authorized tools sequentially with tools.call(name, arguments) and finish with program_output(value). Do not call these tools directly. For a catalog entry with output_schema, output_schema describes the structuredContent field in the successful object returned by tools.call. Available tools: {catalog}"
			),
			"strict": true,
			"parameters": {
				"type": "object",
				"properties": {"code": {"type": "string"}},
				"required": ["code"],
				"additionalProperties": false
			}
		}));
		self
			.canonical_request
			.replace_rest_field("tools", Value::Array(declarations));
		Ok(())
	}

	pub(crate) fn resolve_programmatic_call(
		&self,
		execution_id: &str,
		sequence: usize,
		public_name: &str,
		arguments: Value,
	) -> Result<ManagedToolCall, ToolRuntimeError> {
		let spec = self.programmatic_tools.get(public_name).ok_or_else(|| {
			ToolRuntimeError::invalid_request("programmatic call used an undeclared tool")
		})?;
		let registered = self
			.registry
			.by_internal_name
			.get(spec.internal_name.as_str())
			.ok_or_else(|| {
				ToolRuntimeError::invalid_configuration("programmatic tool is not registered")
			})?;
		if !registered
			.argument_schema
			.as_ref()
			.is_some_and(|schema| schema.is_valid(&arguments))
		{
			return Err(ToolRuntimeError::invalid_request(
				"programmatic tool arguments do not match declared schema",
			));
		}
		Ok(ManagedToolCall {
			public_name: registered.public_name.clone(),
			internal_name: spec.internal_name.clone(),
			call_id: Strng::from(format!("programmatic_{execution_id}_{sequence}")),
			arguments,
			trusted_options: registered
				.trusted_options
				.clone()
				.unwrap_or_else(|| Value::Object(Default::default())),
		})
	}

	/// Authorize and parse the function calls emitted by one model Response.
	///
	/// A configured tool is not sufficient authorization on its own: the tool
	/// must also have been declared in this request's canonical tool list.
	#[cfg(test)]
	pub(crate) fn collect_calls(
		&self,
		response: &responses::Response,
	) -> Result<Vec<ManagedToolCall>, ToolRuntimeError> {
		match self.collect_model_calls(response)? {
			CollectedToolCalls::Direct(calls) => Ok(calls),
			CollectedToolCalls::Programmatic { .. } => Err(ToolRuntimeError::invalid_request(
				"programmatic calls require the managed runner",
			)),
		}
	}

	pub(crate) fn collect_model_calls(
		&self,
		response: &responses::Response,
	) -> Result<CollectedToolCalls, ToolRuntimeError> {
		let declared = self
			.canonical_request
			.rest_field("tools")
			.and_then(Value::as_array)
			.into_iter()
			.flatten()
			.filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
			.filter_map(|tool| tool.get("name").and_then(Value::as_str))
			.collect::<HashSet<_>>();
		let mut seen_call_ids = HashSet::new();
		let mut calls = Vec::new();
		let mut programmatic = None;
		for item in &response.output {
			let responses::typed::OutputItem::FunctionCall(call) = item else {
				continue;
			};
			if !matches!(
				call.status.as_ref(),
				Some(responses::typed::OutputStatus::Completed)
			) {
				return Err(ToolRuntimeError::invalid_request(
					"model generated a managed tool call without completed status",
				));
			}
			if call.call_id.is_empty() || !seen_call_ids.insert(call.call_id.as_str()) {
				return Err(ToolRuntimeError::invalid_request(
					"model generated an empty or duplicate managed tool call id",
				));
			}
			if call.name == PROGRAMMATIC_FUNCTION {
				if !self.programmatic_requested || !declared.contains(PROGRAMMATIC_FUNCTION) {
					return Err(ToolRuntimeError::invalid_request(
						"model generated an undeclared programmatic tool call",
					));
				}
				if call.call_id.len() > SANDBOX_MAX_CALL_ID_BYTES {
					return Err(ToolRuntimeError::limit(
						"Sandbox call id exceeds the 256-byte limit",
						ToolRuntimeLimit::SandboxCallId,
					));
				}
				if programmatic.is_some() || !calls.is_empty() || call.namespace.is_some() {
					return Err(ToolRuntimeError::invalid_request(
						"model generated an invalid programmatic tool batch",
					));
				}
				programmatic = Some((Strng::from(call.call_id.clone()), call.arguments.as_str()));
				continue;
			}
			if programmatic.is_some() {
				return Err(ToolRuntimeError::invalid_request(
					"model mixed programmatic and direct tool calls in one response",
				));
			}
			if call.namespace.is_some() || !declared.contains(call.name.as_str()) {
				return Err(ToolRuntimeError::invalid_request(format!(
					"model generated undeclared managed tool {}",
					call.name
				)));
			}
			let registered = self
				.registry
				.by_internal_name
				.get(call.name.as_str())
				.ok_or_else(|| {
					ToolRuntimeError::invalid_request(format!(
						"model generated unregistered managed tool {}",
						call.name
					))
				})?;
			let arguments = parse_arguments(
				call.arguments.as_bytes(),
				self.registry.limits.max_arguments_bytes,
			)?;
			let argument_schema = registered.argument_schema.as_ref().ok_or_else(|| {
				ToolRuntimeError::invalid_configuration("managed function argument schema was not compiled")
			})?;
			if !argument_schema.is_valid(&arguments) {
				return Err(ToolRuntimeError::invalid_request(
					"tool arguments do not match declared schema",
				));
			}
			calls.push(ManagedToolCall {
				public_name: registered.public_name.clone(),
				internal_name: Strng::from(call.name.clone()),
				call_id: Strng::from(call.call_id.clone()),
				arguments,
				trusted_options: registered
					.trusted_options
					.clone()
					.unwrap_or_else(|| Value::Object(Default::default())),
			});
		}
		if let Some((call_id, arguments)) = programmatic {
			let arguments = parse_arguments(
				arguments.as_bytes(),
				self.registry.limits.max_arguments_bytes,
			)?;
			let object = arguments.as_object().ok_or_else(|| {
				ToolRuntimeError::invalid_request("programmatic call arguments must be an object")
			})?;
			if object.len() != 1 {
				return Err(ToolRuntimeError::invalid_request(
					"programmatic call requires only the code argument",
				));
			}
			let code = object.get("code").and_then(Value::as_str).ok_or_else(|| {
				ToolRuntimeError::invalid_request("programmatic call requires string code")
			})?;
			if code.len() > SANDBOX_MAX_CODE_BYTES {
				return Err(ToolRuntimeError::invalid_request(
					"programmatic code exceeds the 32768-byte limit",
				));
			}
			return Ok(CollectedToolCalls::Programmatic {
				call_id,
				code: code.to_owned(),
			});
		}
		validate_sandbox_contract(&self.registry, &calls, self.parallel)?;
		Ok(CollectedToolCalls::Direct(calls))
	}
}

fn programmatic_catalog_entry_bytes(
	spec: &ProgrammaticToolSpec,
) -> Result<usize, ToolRuntimeError> {
	serde_json::to_vec(&programmatic_catalog_entry(spec))
		.map(|value| value.len())
		.map_err(|_| ToolRuntimeError::internal())
}

fn programmatic_catalog_entry(spec: &ProgrammaticToolSpec) -> Value {
	let mut entry = json!({
		"name": spec.public_name,
		"description": spec.description,
		"input_schema": spec.input_schema,
	});
	if let Some(output_schema) = &spec.output_schema {
		entry["output_schema"] = output_schema.clone();
	}
	entry
}

fn checked_programmatic_catalog_add(
	current_bytes: usize,
	has_existing_entry: bool,
	spec: &ProgrammaticToolSpec,
) -> Result<usize, ToolRuntimeError> {
	let entry_bytes = programmatic_catalog_entry_bytes(spec)?;
	let next = current_bytes
		.checked_add(usize::from(has_existing_entry))
		.and_then(|value| value.checked_add(entry_bytes))
		.ok_or_else(ToolRuntimeError::internal)?;
	if next > PROGRAMMATIC_MAX_CATALOG_BYTES {
		return Err(ToolRuntimeError::invalid_request(
			"programmatic tool catalog exceeds the 2097152-byte limit",
		));
	}
	Ok(next)
}

fn programmatic_catalog_bytes(
	tools: &HashMap<Strng, ProgrammaticToolSpec>,
) -> Result<usize, ToolRuntimeError> {
	let mut bytes = 2usize;
	let mut has_existing_entry = false;
	for spec in tools.values() {
		bytes = checked_programmatic_catalog_add(bytes, has_existing_entry, spec)?;
		has_existing_entry = true;
	}
	Ok(bytes)
}

fn truncate_utf8(mut value: String, max_bytes: usize) -> String {
	if value.len() <= max_bytes {
		return value;
	}
	let mut boundary = max_bytes;
	while !value.is_char_boundary(boundary) {
		boundary -= 1;
	}
	value.truncate(boundary);
	value
}

fn validate_sandbox_contract(
	registry: &ToolRegistry,
	calls: &[ManagedToolCall],
	parallel: bool,
) -> Result<(), ToolRuntimeError> {
	let is_sandbox = |call: &ManagedToolCall| {
		registry
			.by_internal_name
			.get(call.internal_name.as_str())
			.is_some_and(|registered| matches!(registered.backend, Some(ToolBackendConfig::E2b { .. })))
	};
	let mut batch_size = 0usize;
	for call in calls {
		if !is_sandbox(call) {
			if !parallel {
				batch_size = 0;
			}
			continue;
		}
		if call.call_id.len() > SANDBOX_MAX_CALL_ID_BYTES {
			return Err(ToolRuntimeError::limit(
				"Sandbox call id exceeds the 256-byte limit",
				ToolRuntimeLimit::SandboxCallId,
			));
		}
		let code = call
			.arguments
			.get("code")
			.and_then(Value::as_str)
			.ok_or_else(|| ToolRuntimeError::invalid_request("Sandbox code arguments are invalid"))?;
		if code.len() > SANDBOX_MAX_CODE_BYTES {
			return Err(ToolRuntimeError::limit(
				"Sandbox code exceeds the 32768-byte limit",
				ToolRuntimeLimit::SandboxCode,
			));
		}
		batch_size = batch_size.checked_add(1).ok_or_else(|| {
			ToolRuntimeError::limit(
				"Sandbox batch limit exceeded",
				ToolRuntimeLimit::SandboxBatch,
			)
		})?;
		if batch_size > SANDBOX_MAX_BATCH_EXECUTIONS {
			return Err(ToolRuntimeError::limit(
				"Sandbox batch limit exceeded (maximum 8)",
				ToolRuntimeLimit::SandboxBatch,
			));
		}
	}
	Ok(())
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum Activation {
	Inactive,
	Active(PreparedToolRuntime),
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub(crate) struct ToolRuntimeError {
	message: String,
	infrastructure: Option<ToolInfrastructureError>,
	deadline_exceeded: bool,
	limit: Option<ToolRuntimeLimit>,
}

impl ToolRuntimeError {
	pub(crate) fn invalid_configuration(message: impl Into<String>) -> Self {
		Self {
			message: format!("invalid managed tool configuration: {}", message.into()),
			infrastructure: Some(ToolInfrastructureError::configuration()),
			deadline_exceeded: false,
			limit: None,
		}
	}

	pub(crate) fn invalid_request(message: impl Into<String>) -> Self {
		Self {
			message: format!("invalid managed tool request: {}", message.into()),
			infrastructure: None,
			deadline_exceeded: false,
			limit: None,
		}
	}

	pub(crate) fn limit(message: impl Into<String>, limit: ToolRuntimeLimit) -> Self {
		Self {
			message: format!("invalid managed tool request: {}", message.into()),
			infrastructure: None,
			deadline_exceeded: false,
			limit: Some(limit),
		}
	}

	pub(crate) fn unsupported(field: &str) -> Self {
		Self::invalid_request(format!(
			"{field} is unsupported while managed tool runtime is active"
		))
	}

	fn infrastructure(error: ToolInfrastructureError) -> Self {
		Self {
			message: error.to_string(),
			infrastructure: Some(error),
			deadline_exceeded: false,
			limit: None,
		}
	}

	pub(crate) fn deadline_exceeded() -> Self {
		Self {
			message: ToolInfrastructureError::timeout().to_string(),
			infrastructure: Some(ToolInfrastructureError::timeout()),
			deadline_exceeded: true,
			limit: Some(ToolRuntimeLimit::Deadline),
		}
	}

	pub(crate) fn internal() -> Self {
		Self::infrastructure(ToolInfrastructureError::internal())
	}

	#[cfg(test)]
	pub(crate) fn infrastructure_error(&self) -> Option<ToolInfrastructureError> {
		self.infrastructure
	}

	pub(crate) fn exhausted_limit(&self) -> Option<ToolRuntimeLimit> {
		self.limit
	}

	pub(crate) fn telemetry_outcome(&self) -> ToolRuntimeOutcome {
		if self.deadline_exceeded || self.infrastructure == Some(ToolInfrastructureError::Timeout) {
			ToolRuntimeOutcome::Timeout
		} else if self.infrastructure.is_some() {
			ToolRuntimeOutcome::InfrastructureError
		} else {
			ToolRuntimeOutcome::InvalidRequest
		}
	}

	pub(crate) fn into_openai_response(self) -> crate::http::Response {
		let (status, body) = match (self.deadline_exceeded, self.infrastructure) {
			(true, _) => (
				::http::StatusCode::GATEWAY_TIMEOUT,
				ToolInfrastructureError::Timeout.to_openai_error(),
			),
			(false, Some(infrastructure)) => (
				::http::StatusCode::BAD_GATEWAY,
				infrastructure.to_openai_error(),
			),
			(false, None) => (
				::http::StatusCode::BAD_REQUEST,
				json!({
					"error": {
						"type": "invalid_request_error",
						"message": self.message,
						"code": "managed_tool_request_invalid",
					}
				}),
			),
		};
		let body = serde_json::to_vec(&body).expect("tool runtime error response serializes");
		::http::Response::builder()
			.status(status)
			.header(::http::header::CONTENT_TYPE, "application/json")
			.body(crate::http::Body::from(body))
			.expect("tool runtime error response builds")
	}
}

#[derive(Clone)]
struct BoundBackend {
	backend: Arc<dyn ToolBackend>,
	label: ToolBackendLabel,
}

/// One request-wide budget. Construct it before the first model round and
/// carry it through every tool batch so the absolute deadline never resets.
pub(crate) struct RuntimeBudget {
	deadline: tokio::time::Instant,
	max_rounds: usize,
	rounds: usize,
	max_tool_calls: usize,
	tool_calls: usize,
	max_parallel_tool_calls: usize,
	max_output_bytes: usize,
	request_id: Option<Strng>,
	backends: HashMap<Strng, BoundBackend>,
	program_sandbox: Option<Arc<dyn ProgramSandbox>>,
	program_sandbox_executions: usize,
	next_execution_index: usize,
	telemetry: ToolRuntimeTelemetry,
}

impl RuntimeBudget {
	#[cfg(test)]
	pub(crate) fn new(
		registry: &ToolRegistry,
		client: PolicyClient,
	) -> Result<Self, ToolRuntimeError> {
		Self::new_at(
			registry,
			client,
			RuntimeDeadline::new(registry.limits.total_timeout),
		)
	}

	pub(crate) fn new_at(
		registry: &ToolRegistry,
		client: PolicyClient,
		deadline: RuntimeDeadline,
	) -> Result<Self, ToolRuntimeError> {
		let telemetry = ToolRuntimeTelemetry::new(client.inputs.metrics.clone());
		let mut backends = HashMap::with_capacity(registry.by_internal_name.len());
		let mut program_sandbox = None;
		for (internal_name, registered) in registry.by_internal_name.iter() {
			if let Some(request_backend) = &registered.request_backend {
				backends.insert(
					internal_name.clone(),
					BoundBackend {
						backend: request_backend.backend.clone(),
						label: request_backend.label,
					},
				);
				continue;
			}
			let (backend, label): (Arc<dyn ToolBackend>, ToolBackendLabel) = match registered
				.backend
				.as_ref()
				.expect("configured tools have a configured backend")
			{
				ToolBackendConfig::Http {
					url,
					timeout,
					bearer_token,
				} => (
					Arc::new(
						HttpToolBackend::new(
							client.clone(),
							url.clone(),
							*timeout,
							bearer_token.clone(),
							registry.limits.max_output_bytes,
						)
						.map_err(|error| {
							let error = Self::backend_construction_error(error);
							telemetry.record_request(error.telemetry_outcome());
							error
						})?,
					),
					ToolBackendLabel::Http,
				),
				ToolBackendConfig::E2b {
					api_url,
					domain,
					timeout,
					api_key,
				} => {
					let backend = Arc::new(
						E2bSandboxBackend::new(
							client.clone(),
							api_url.clone(),
							domain.clone(),
							*timeout,
							api_key.clone(),
							registry.limits.max_output_bytes,
						)
						.map_err(|error| {
							let error = Self::backend_construction_error(error);
							telemetry.record_request(error.telemetry_outcome());
							error
						})?,
					);
					program_sandbox = Some(backend.clone() as Arc<dyn ProgramSandbox>);
					(backend as Arc<dyn ToolBackend>, ToolBackendLabel::E2b)
				},
			};
			backends.insert(internal_name.clone(), BoundBackend { backend, label });
		}
		Ok(Self::from_backends(
			registry,
			backends,
			program_sandbox,
			telemetry,
			deadline,
		))
	}

	fn backend_construction_error(error: ToolInfrastructureError) -> ToolRuntimeError {
		match error {
			ToolInfrastructureError::Configuration => {
				ToolRuntimeError::invalid_configuration(error.to_string())
			},
			_ => ToolRuntimeError::infrastructure(error),
		}
	}

	fn from_backends(
		registry: &ToolRegistry,
		backends: HashMap<Strng, BoundBackend>,
		program_sandbox: Option<Arc<dyn ProgramSandbox>>,
		telemetry: ToolRuntimeTelemetry,
		deadline: RuntimeDeadline,
	) -> Self {
		Self {
			deadline: deadline.instant(),
			max_rounds: registry.limits.max_rounds,
			rounds: 0,
			max_tool_calls: registry.limits.max_tool_calls,
			tool_calls: 0,
			max_parallel_tool_calls: registry.limits.max_parallel_tool_calls,
			max_output_bytes: registry.limits.max_output_bytes,
			request_id: None,
			backends,
			program_sandbox,
			program_sandbox_executions: 0,
			next_execution_index: 0,
			telemetry,
		}
	}

	#[cfg(test)]
	pub(crate) fn with_test_backends(
		registry: &ToolRegistry,
		backends: HashMap<String, Arc<dyn ToolBackend>>,
	) -> Self {
		Self::from_backends(
			registry,
			Self::bind_test_backends(registry, backends),
			None,
			ToolRuntimeTelemetry::default(),
			RuntimeDeadline::new(registry.limits.total_timeout),
		)
	}

	#[cfg(test)]
	pub(crate) fn with_test_backends_and_metrics(
		registry: &ToolRegistry,
		backends: HashMap<String, Arc<dyn ToolBackend>>,
		metrics: Arc<crate::telemetry::metrics::Metrics>,
	) -> Self {
		Self::from_backends(
			registry,
			Self::bind_test_backends(registry, backends),
			None,
			ToolRuntimeTelemetry::new(metrics),
			RuntimeDeadline::new(registry.limits.total_timeout),
		)
	}

	#[cfg(test)]
	pub(crate) fn with_test_backends_and_program_sandbox(
		registry: &ToolRegistry,
		backends: HashMap<String, Arc<dyn ToolBackend>>,
		program_sandbox: Arc<dyn ProgramSandbox>,
	) -> Self {
		Self::from_backends(
			registry,
			Self::bind_test_backends(registry, backends),
			Some(program_sandbox),
			ToolRuntimeTelemetry::default(),
			RuntimeDeadline::new(registry.limits.total_timeout),
		)
	}

	#[cfg(test)]
	fn bind_test_backends(
		registry: &ToolRegistry,
		backends: HashMap<String, Arc<dyn ToolBackend>>,
	) -> HashMap<Strng, BoundBackend> {
		backends
			.into_iter()
			.map(|(internal_name, backend)| {
				let registered = registry
					.by_internal_name
					.get(internal_name.as_str())
					.expect("test backend must name a registered tool");
				let label = match registered.backend.as_ref() {
					Some(ToolBackendConfig::Http { .. }) => ToolBackendLabel::Http,
					Some(ToolBackendConfig::E2b { .. }) => ToolBackendLabel::E2b,
					None => {
						registered
							.request_backend
							.as_ref()
							.expect("dynamic tool has a request backend")
							.label
					},
				};
				(Strng::from(internal_name), BoundBackend { backend, label })
			})
			.collect()
	}

	pub(crate) fn set_request_id(&mut self, request_id: Option<Strng>) {
		self.request_id = request_id;
	}

	pub(crate) fn remaining(&self) -> std::time::Duration {
		self
			.deadline
			.saturating_duration_since(tokio::time::Instant::now())
	}

	pub(crate) fn tool_calls(&self) -> usize {
		self.tool_calls
	}

	pub(crate) fn max_output_bytes(&self) -> usize {
		self.max_output_bytes
	}

	#[cfg(test)]
	pub(crate) fn program_sandbox_executions(&self) -> usize {
		self.program_sandbox_executions
	}

	pub(crate) async fn execute_program_sandbox(
		&mut self,
		request: ProgramSandboxRequest,
	) -> Result<ProgramSandboxExecution, ToolRuntimeError> {
		let sandbox = self.program_sandbox.clone().ok_or_else(|| {
			ToolRuntimeError::invalid_configuration("program sandbox backend is not bound")
		})?;
		let remaining = self.remaining();
		if remaining.is_zero() {
			let error = ToolRuntimeError::deadline_exceeded();
			self.record_error(&error);
			return Err(error);
		}
		let execution_index = self.next_execution_index;
		self.next_execution_index = self
			.next_execution_index
			.checked_add(1)
			.ok_or_else(|| ToolRuntimeError::invalid_configuration("program execution index overflow"))?;
		self.program_sandbox_executions =
			self
				.program_sandbox_executions
				.checked_add(1)
				.ok_or_else(|| {
					ToolRuntimeError::invalid_configuration("program sandbox execution count overflow")
				})?;
		for operation in [
			SandboxOperation::Execute,
			SandboxOperation::Create,
			SandboxOperation::Cleanup,
			SandboxOperation::Terminate,
		] {
			self
				.telemetry
				.start_sandbox_operation(execution_index, operation);
		}
		let context = ToolExecutionContext {
			request_id: self.request_id.clone(),
			deadline: Some(Instant::now() + remaining),
		};
		match tokio::time::timeout_at(self.deadline, sandbox.run(request, context)).await {
			Ok(Ok(execution)) => {
				finish_sandbox_metadata(&self.telemetry, execution_index, execution.metadata, None);
				Ok(execution)
			},
			Ok(Err(error)) => {
				let fallback = (error.error == ToolInfrastructureError::Timeout)
					.then_some(SandboxOperationOutcome::Timeout);
				finish_sandbox_metadata(
					&self.telemetry,
					execution_index,
					error.metadata,
					fallback.or(Some(SandboxOperationOutcome::Failure)),
				);
				let error = ToolRuntimeError::infrastructure(error.error);
				self.record_error(&error);
				Err(error)
			},
			Err(_) => {
				record_unfinished_sandbox_operations(
					&self.telemetry,
					&[PendingSandboxOperation { execution_index }],
					SandboxOperationOutcome::Timeout,
				);
				let error = ToolRuntimeError::deadline_exceeded();
				self.record_error(&error);
				Err(error)
			},
		}
	}

	#[cfg(test)]
	pub(crate) fn rounds(&self) -> usize {
		self.rounds
	}

	/// Reserve the next model round against the same request-wide deadline and
	/// round limit used by every prior model and tool call.
	pub(crate) fn start_model_round(&mut self) -> Result<(), ToolRuntimeError> {
		if self.remaining().is_zero() {
			let error = ToolRuntimeError::deadline_exceeded();
			self.record_error(&error);
			return Err(error);
		}
		let Some(next) = self.rounds.checked_add(1) else {
			let error = ToolRuntimeError::limit(
				"managed tool round limit exceeded",
				ToolRuntimeLimit::Rounds,
			);
			self.record_error(&error);
			return Err(error);
		};
		if next > self.max_rounds {
			let error = ToolRuntimeError::limit(
				format!(
					"managed tool round limit exceeded (maximum {})",
					self.max_rounds
				),
				ToolRuntimeLimit::Rounds,
			);
			self.record_error(&error);
			return Err(error);
		}
		self.rounds = next;
		self.telemetry.start_model_round(next);
		Ok(())
	}

	pub(crate) fn record_model_round(&self, outcome: ToolRuntimeOutcome, duration: Duration) {
		self
			.telemetry
			.record_model_round(self.rounds, outcome, duration);
	}

	pub(crate) fn finish_request(&self, outcome: ToolRuntimeOutcome) {
		self.telemetry.record_request(outcome);
	}

	pub(crate) fn record_error(&self, error: &ToolRuntimeError) {
		if let Some(limit) = error.exhausted_limit() {
			self.telemetry.record_limit(limit);
		}
	}

	#[cfg(test)]
	pub(crate) fn record_sandbox_operation(
		&self,
		operation: SandboxOperation,
		outcome: SandboxOperationOutcome,
	) {
		self
			.telemetry
			.observe_sandbox_operation(operation, outcome, Duration::from_millis(1));
	}

	#[cfg(test)]
	pub(crate) fn execution_records(&self) -> Vec<ToolExecutionRecord> {
		self.telemetry.records()
	}

	fn reserve_calls(&mut self, count: usize) -> Result<usize, ToolRuntimeError> {
		let start = self.next_execution_index;
		let total = self.tool_calls.checked_add(count).ok_or_else(|| {
			ToolRuntimeError::limit(
				"managed tool call limit exceeded",
				ToolRuntimeLimit::ToolCalls,
			)
		})?;
		if total > self.max_tool_calls {
			return Err(ToolRuntimeError::limit(
				format!(
					"managed tool call limit exceeded (maximum {})",
					self.max_tool_calls
				),
				ToolRuntimeLimit::ToolCalls,
			));
		}
		let next_execution_index = self
			.next_execution_index
			.checked_add(count)
			.ok_or_else(|| ToolRuntimeError::invalid_configuration("tool execution index overflow"))?;
		self.tool_calls = total;
		self.next_execution_index = next_execution_index;
		Ok(start)
	}
}

impl Drop for RuntimeBudget {
	fn drop(&mut self) {
		self.telemetry.finish_unfinished();
		self.telemetry.record_request(ToolRuntimeOutcome::Cancelled);
	}
}

#[derive(Clone, Debug)]
pub(crate) struct ToolRuntimeSummary {
	pub(crate) usage: Option<responses::Usage>,
	#[cfg(test)]
	pub(crate) rounds: usize,
	#[cfg(test)]
	pub(crate) tool_calls: usize,
	pub(crate) client_streaming: bool,
	pub(crate) include_obfuscation: bool,
	pub(crate) client_tools: Option<Value>,
}

pub(crate) struct ManagedFinalResponse {
	pub(crate) response: responses::Response,
	pub(crate) raw_output: Vec<Value>,
	pub(crate) raw_upstream: Response,
	pub(crate) summary: ToolRuntimeSummary,
}

pub(crate) fn finalize_managed_response(
	mut response: responses::Response,
	summary: &ToolRuntimeSummary,
) -> responses::Response {
	response.usage = summary.usage.clone();
	if let Some(client_tools) = summary.client_tools.clone() {
		if let Some(rest) = response.rest.as_object_mut() {
			rest.insert("tools".to_owned(), client_tools);
		} else {
			response.rest = json!({"tools": client_tools});
		}
	}
	response
}

/// Encode a fully processed managed Response as OpenAI Responses SSE events.
///
/// Managed execution intentionally buffers model rounds so intermediate calls
/// and reserved tool names never reach the client. The final canonical response
/// is then emitted as one valid event stream.
#[cfg(test)]
pub(crate) fn encode_streaming_response(
	response: &responses::Response,
	include_obfuscation: bool,
) -> Result<Vec<u8>, serde_json::Error> {
	encode_streaming_response_value(serde_json::to_value(response)?, include_obfuscation)
}

pub(crate) fn serialize_managed_response(
	response: &responses::Response,
	raw_output: &[Value],
) -> Result<Vec<u8>, serde_json::Error> {
	serde_json::to_vec(&merge_final_output(response, raw_output)?)
}

pub(crate) fn encode_managed_streaming_response(
	response: &responses::Response,
	raw_output: &[Value],
	include_obfuscation: bool,
) -> Result<Vec<u8>, serde_json::Error> {
	encode_streaming_response_value(
		merge_final_output(response, raw_output)?,
		include_obfuscation,
	)
}

fn merge_final_output(
	response: &responses::Response,
	raw_output: &[Value],
) -> Result<Value, serde_json::Error> {
	let mut final_response = serde_json::to_value(response)?;
	let typed_output = final_response
		.get_mut("output")
		.and_then(Value::as_array_mut)
		.ok_or_else(|| {
			serde_json::Error::io(std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				"managed Responses output was not an array",
			))
		})?;
	let mut typed_output = std::mem::take(typed_output).into_iter();
	let mut merged = Vec::with_capacity(raw_output.len());
	for raw_item in raw_output {
		let is_unknown = raw_item
			.get("type")
			.and_then(Value::as_str)
			.is_some_and(|kind| !super::is_known_responses_output_item_type(kind));
		if is_unknown {
			merged.push(raw_item.clone());
		} else {
			merged.push(typed_output.next().ok_or_else(|| {
				serde_json::Error::io(std::io::Error::new(
					std::io::ErrorKind::InvalidData,
					"managed Responses output item count changed",
				))
			})?);
		}
	}
	if typed_output.next().is_some() {
		return Err(serde_json::Error::io(std::io::Error::new(
			std::io::ErrorKind::InvalidData,
			"managed Responses output item count changed",
		)));
	}
	final_response["output"] = Value::Array(merged);
	Ok(final_response)
}

fn encode_streaming_response_value(
	mut final_response: Value,
	include_obfuscation: bool,
) -> Result<Vec<u8>, serde_json::Error> {
	fn push_event(
		body: &mut Vec<u8>,
		sequence_number: &mut u64,
		event_type: &str,
		mut event: Value,
	) -> Result<(), serde_json::Error> {
		*sequence_number += 1;
		event["type"] = Value::String(event_type.to_owned());
		event["sequence_number"] = Value::from(*sequence_number);
		body.extend_from_slice(b"event: ");
		body.extend_from_slice(event_type.as_bytes());
		body.extend_from_slice(b"\ndata: ");
		serde_json::to_writer(&mut *body, &event)?;
		body.extend_from_slice(b"\n\n");
		Ok(())
	}

	if let Some(usage) = final_response
		.get_mut("usage")
		.and_then(Value::as_object_mut)
	{
		usage
			.entry("input_tokens_details")
			.or_insert_with(|| json!({"cached_tokens": 0}));
		usage
			.entry("output_tokens_details")
			.or_insert_with(|| json!({"reasoning_tokens": 0}));
	}
	let mut initial_response = final_response.clone();
	initial_response["status"] = Value::String("in_progress".to_owned());
	initial_response["output"] = Value::Array(Vec::new());
	initial_response["usage"] = Value::Null;
	initial_response["completed_at"] = Value::Null;
	initial_response["error"] = Value::Null;
	initial_response["incomplete_details"] = Value::Null;

	let mut body = Vec::new();
	let mut sequence_number = 0;
	push_event(
		&mut body,
		&mut sequence_number,
		"response.created",
		json!({"response": initial_response}),
	)?;
	push_event(
		&mut body,
		&mut sequence_number,
		"response.in_progress",
		json!({"response": initial_response}),
	)?;

	for (output_index, item) in final_response["output"]
		.as_array()
		.into_iter()
		.flatten()
		.enumerate()
	{
		let output_index = output_index as u64;
		let item_id = item.get("id").and_then(Value::as_str).unwrap_or_default();
		let is_message = item.get("type").and_then(Value::as_str) == Some("message");
		let mut added_item = item.clone();
		added_item["status"] = Value::String("in_progress".to_owned());
		if is_message {
			added_item["content"] = Value::Array(Vec::new());
		} else if added_item.get("type").and_then(Value::as_str) == Some("reasoning") {
			if added_item.get("summary").is_some() {
				added_item["summary"] = Value::Array(Vec::new());
			}
			if added_item.get("content").is_some() {
				added_item["content"] = Value::Array(Vec::new());
			}
		}
		push_event(
			&mut body,
			&mut sequence_number,
			"response.output_item.added",
			json!({"output_index": output_index, "item": added_item}),
		)?;

		if is_message && let Some(content) = item.get("content").and_then(Value::as_array) {
			for (content_index, part) in content.iter().enumerate() {
				let content_index = content_index as u64;
				let part_type = part.get("type").and_then(Value::as_str);
				let mut added_part = part.clone();
				match part_type {
					Some("output_text") => added_part["text"] = Value::String(String::new()),
					Some("refusal") => added_part["refusal"] = Value::String(String::new()),
					_ => {},
				}
				push_event(
					&mut body,
					&mut sequence_number,
					"response.content_part.added",
					json!({
						"item_id": item_id,
						"output_index": output_index,
						"content_index": content_index,
						"part": added_part,
					}),
				)?;

				if let Some(text) = part.get("text").and_then(Value::as_str) {
					let mut delta_event = json!({
						"item_id": item_id,
						"output_index": output_index,
						"content_index": content_index,
						"delta": text,
						"logprobs": part.get("logprobs").cloned().unwrap_or(Value::Null),
					});
					if include_obfuscation {
						delta_event["obfuscation"] =
							Value::String(format!("{:032x}", rand::rng().random::<u128>()));
					}
					push_event(
						&mut body,
						&mut sequence_number,
						"response.output_text.delta",
						delta_event,
					)?;
					push_event(
						&mut body,
						&mut sequence_number,
						"response.output_text.done",
						json!({
							"item_id": item_id,
							"output_index": output_index,
							"content_index": content_index,
							"text": text,
							"logprobs": part.get("logprobs").cloned().unwrap_or(Value::Null),
						}),
					)?;
				} else if let Some(refusal) = part.get("refusal").and_then(Value::as_str) {
					let mut delta_event = json!({
						"item_id": item_id,
						"output_index": output_index,
						"content_index": content_index,
						"delta": refusal,
					});
					if include_obfuscation {
						delta_event["obfuscation"] =
							Value::String(format!("{:032x}", rand::rng().random::<u128>()));
					}
					push_event(
						&mut body,
						&mut sequence_number,
						"response.refusal.delta",
						delta_event,
					)?;
					push_event(
						&mut body,
						&mut sequence_number,
						"response.refusal.done",
						json!({
							"item_id": item_id,
							"output_index": output_index,
							"content_index": content_index,
							"refusal": refusal,
						}),
					)?;
				}

				push_event(
					&mut body,
					&mut sequence_number,
					"response.content_part.done",
					json!({
						"item_id": item_id,
						"output_index": output_index,
						"content_index": content_index,
						"part": part,
					}),
				)?;
			}
		}

		push_event(
			&mut body,
			&mut sequence_number,
			"response.output_item.done",
			json!({"output_index": output_index, "item": item}),
		)?;
	}

	let terminal_event = match final_response.get("status").and_then(Value::as_str) {
		Some("completed") => "response.completed",
		Some("incomplete") => "response.incomplete",
		Some("failed") => "response.failed",
		Some(status) => {
			return Err(serde_json::Error::io(std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				format!("unsupported terminal Responses status {status}"),
			)));
		},
		None => {
			return Err(serde_json::Error::io(std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				"missing terminal Responses status",
			)));
		},
	};
	push_event(
		&mut body,
		&mut sequence_number,
		terminal_event,
		json!({"response": final_response}),
	)?;
	Ok(body)
}

struct IndexedCall {
	index: usize,
	execution_index: usize,
	call: ManagedToolCall,
	tool: Strng,
}

struct BackendOperation {
	backend: Arc<dyn ToolBackend>,
	backend_label: ToolBackendLabel,
	calls: Vec<IndexedCall>,
}

struct IndexedOutput {
	index: usize,
	value: Value,
}

#[derive(Clone)]
struct PendingExecutionRecord {
	execution_index: usize,
	tool: Strng,
	backend: ToolBackendLabel,
}

#[derive(Clone, Copy)]
struct PendingSandboxOperation {
	execution_index: usize,
}

/// Execute all managed calls emitted by one model Response.
pub(crate) async fn execute_batch(
	registry: &ToolRegistry,
	calls: Vec<ManagedToolCall>,
	parallel: bool,
	budget: &mut RuntimeBudget,
) -> Result<Vec<Value>, ToolRuntimeError> {
	if let Err(error) = validate_sandbox_contract(registry, &calls, parallel) {
		budget.record_error(&error);
		return Err(error);
	}
	let execution_start = match budget.reserve_calls(calls.len()) {
		Ok(start) => start,
		Err(error) => {
			budget.record_error(&error);
			return Err(error);
		},
	};
	if calls.is_empty() {
		return Ok(Vec::new());
	}
	if budget.remaining().is_zero() {
		let error = ToolRuntimeError::deadline_exceeded();
		budget.record_error(&error);
		return Err(error);
	}

	let operations = build_operations(registry, calls, parallel, budget, execution_start)?;
	let pending_records = operations
		.iter()
		.flat_map(|operation| {
			operation.calls.iter().map(|call| PendingExecutionRecord {
				execution_index: call.execution_index,
				tool: call.tool.clone(),
				backend: operation.backend_label,
			})
		})
		.collect::<Vec<_>>();
	let pending_sandbox_operations = operations
		.iter()
		.filter(|operation| operation.backend_label == ToolBackendLabel::E2b)
		.map(|operation| PendingSandboxOperation {
			execution_index: operation.calls[0].execution_index,
		})
		.collect::<Vec<_>>();
	for call in &pending_records {
		budget
			.telemetry
			.start_call(call.execution_index, call.tool.clone(), call.backend);
	}
	for operation in &pending_sandbox_operations {
		budget
			.telemetry
			.start_sandbox_operation(operation.execution_index, SandboxOperation::Execute);
	}
	let max_parallel = if parallel {
		budget.max_parallel_tool_calls
	} else {
		1
	};
	let execution = run_operations(
		operations,
		max_parallel,
		budget.deadline,
		budget.request_id.clone(),
		budget.max_output_bytes,
		budget.telemetry.clone(),
	);
	let mut outputs = match tokio::time::timeout_at(budget.deadline, execution).await {
		Ok(Ok(outputs)) => outputs,
		Ok(Err(error)) => {
			let timed_out = budget.remaining().is_zero();
			record_unfinished(
				&budget.telemetry,
				&pending_records,
				if timed_out {
					ToolExecutionOutcome::Timeout
				} else {
					ToolExecutionOutcome::Cancelled
				},
			);
			record_unfinished_sandbox_operations(
				&budget.telemetry,
				&pending_sandbox_operations,
				if timed_out {
					SandboxOperationOutcome::Timeout
				} else {
					SandboxOperationOutcome::Cancelled
				},
			);
			let error = if timed_out {
				ToolRuntimeError::deadline_exceeded()
			} else {
				error
			};
			budget.record_error(&error);
			return Err(error);
		},
		Err(_) => {
			record_unfinished(
				&budget.telemetry,
				&pending_records,
				ToolExecutionOutcome::Timeout,
			);
			record_unfinished_sandbox_operations(
				&budget.telemetry,
				&pending_sandbox_operations,
				SandboxOperationOutcome::Timeout,
			);
			let error = ToolRuntimeError::deadline_exceeded();
			budget.record_error(&error);
			return Err(error);
		},
	};
	outputs.sort_by_key(|output| output.index);
	Ok(outputs.into_iter().map(|output| output.value).collect())
}

fn build_operations(
	registry: &ToolRegistry,
	calls: Vec<ManagedToolCall>,
	parallel: bool,
	budget: &RuntimeBudget,
	execution_start: usize,
) -> Result<Vec<BackendOperation>, ToolRuntimeError> {
	let mut resolved = Vec::with_capacity(calls.len());
	for (index, call) in calls.into_iter().enumerate() {
		let registered = registry
			.by_internal_name
			.get(call.internal_name.as_str())
			.ok_or_else(|| {
				ToolRuntimeError::invalid_request(format!(
					"model generated unregistered managed tool {}",
					call.internal_name
				))
			})?;
		let bound = budget
			.backends
			.get(call.internal_name.as_str())
			.ok_or_else(|| {
				ToolRuntimeError::invalid_configuration(format!(
					"managed tool backend {} is not bound",
					registered.public_name
				))
			})?;
		resolved.push((
			bound.clone(),
			IndexedCall {
				index,
				execution_index: execution_start + index,
				call,
				// Remote MCP names are client/server controlled and therefore cannot
				// become metric labels. Configured tools retain their operator-bounded
				// public name.
				tool: if bound.label == ToolBackendLabel::RemoteMcp {
					Strng::from("remote_mcp")
				} else {
					registered.public_name.clone()
				},
			},
		));
	}

	if parallel {
		let mut operations = Vec::new();
		let mut sandbox: Option<BackendOperation> = None;
		for (bound, call) in resolved {
			if bound.label == ToolBackendLabel::E2b {
				sandbox
					.get_or_insert_with(|| BackendOperation {
						backend: bound.backend.clone(),
						backend_label: bound.label,
						calls: Vec::new(),
					})
					.calls
					.push(call);
			} else {
				operations.push(BackendOperation {
					backend: bound.backend,
					backend_label: bound.label,
					calls: vec![call],
				});
			}
		}
		if let Some(sandbox) = sandbox {
			operations.push(sandbox);
		}
		operations.sort_by_key(|operation| operation.calls[0].index);
		return Ok(operations);
	}

	let mut operations: Vec<BackendOperation> = Vec::new();
	for (bound, call) in resolved {
		if bound.label == ToolBackendLabel::E2b
			&& let Some(previous) = operations.last_mut()
			&& previous.backend_label == ToolBackendLabel::E2b
		{
			previous.calls.push(call);
		} else {
			operations.push(BackendOperation {
				backend: bound.backend,
				backend_label: bound.label,
				calls: vec![call],
			});
		}
	}
	Ok(operations)
}

fn record_unfinished(
	telemetry: &ToolRuntimeTelemetry,
	pending: &[PendingExecutionRecord],
	outcome: ToolExecutionOutcome,
) {
	for call in pending {
		telemetry.record(
			call.execution_index,
			ToolExecutionRecord {
				tool: call.tool.clone(),
				backend: call.backend,
				outcome,
			},
			false,
		);
	}
}

fn record_unfinished_sandbox_operations(
	telemetry: &ToolRuntimeTelemetry,
	pending: &[PendingSandboxOperation],
	outcome: SandboxOperationOutcome,
) {
	for operation in pending {
		telemetry.finish_sandbox_operation(
			operation.execution_index,
			SandboxOperation::Execute,
			outcome,
		);
		telemetry.finish_started_sandbox_operation(
			operation.execution_index,
			SandboxOperation::Cleanup,
			outcome,
		);
		for lifecycle in [SandboxOperation::Create, SandboxOperation::Terminate] {
			telemetry.finish_started_sandbox_operation(operation.execution_index, lifecycle, outcome);
		}
	}
}

async fn run_operations(
	operations: Vec<BackendOperation>,
	max_parallel: usize,
	deadline: tokio::time::Instant,
	request_id: Option<Strng>,
	max_output_bytes: usize,
	telemetry: ToolRuntimeTelemetry,
) -> Result<Vec<IndexedOutput>, ToolRuntimeError> {
	let mut queued = operations.into_iter();
	let mut in_flight = FuturesUnordered::new();
	for operation in queued.by_ref().take(max_parallel) {
		in_flight.push(execute_operation(
			operation,
			deadline,
			request_id.clone(),
			max_output_bytes,
			telemetry.clone(),
		));
	}

	let mut outputs = Vec::new();
	while let Some(result) = in_flight.next().await {
		outputs.extend(result?);
		if let Some(operation) = queued.next() {
			in_flight.push(execute_operation(
				operation,
				deadline,
				request_id.clone(),
				max_output_bytes,
				telemetry.clone(),
			));
		}
	}
	Ok(outputs)
}

async fn execute_operation(
	operation: BackendOperation,
	deadline: tokio::time::Instant,
	request_id: Option<Strng>,
	max_output_bytes: usize,
	telemetry: ToolRuntimeTelemetry,
) -> Result<Vec<IndexedOutput>, ToolRuntimeError> {
	let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
	if remaining.is_zero() {
		return Err(ToolRuntimeError::deadline_exceeded());
	}
	let context = ToolExecutionContext {
		request_id,
		deadline: Some(Instant::now() + remaining),
	};
	let BackendOperation {
		backend,
		backend_label,
		calls,
	} = operation;
	for call in &calls {
		telemetry.mark_call_executing(call.execution_index);
	}
	let sandbox_operation_index = calls[0].execution_index;
	let backend_calls = calls.iter().map(|call| call.call.clone()).collect();
	if backend_label == ToolBackendLabel::E2b {
		telemetry.start_sandbox_operation(sandbox_operation_index, SandboxOperation::Create);
		telemetry.start_sandbox_operation(sandbox_operation_index, SandboxOperation::Cleanup);
		telemetry.start_sandbox_operation(sandbox_operation_index, SandboxOperation::Terminate);
	}
	let result = backend.execute_batch(backend_calls, context).await;
	let result = match result {
		Ok(batch) => {
			if backend_label == ToolBackendLabel::E2b {
				finish_sandbox_metadata(&telemetry, sandbox_operation_index, batch.metadata, None);
			}
			Ok(batch.results)
		},
		Err(error) => {
			if backend_label == ToolBackendLabel::E2b {
				let fallback = (error.error == ToolInfrastructureError::Timeout)
					.then_some(SandboxOperationOutcome::Timeout);
				finish_sandbox_metadata(
					&telemetry,
					sandbox_operation_index,
					error.metadata,
					fallback,
				);
			}
			Err(error.error)
		},
	};
	let results = match result {
		Ok(results) if results.len() == calls.len() => results,
		Ok(_) => {
			if backend_label == ToolBackendLabel::E2b {
				telemetry.finish_sandbox_operation(
					sandbox_operation_index,
					SandboxOperation::Execute,
					SandboxOperationOutcome::Failure,
				);
			}
			for call in &calls {
				telemetry.record(
					call.execution_index,
					ToolExecutionRecord {
						tool: call.tool.clone(),
						backend: backend_label,
						outcome: ToolExecutionOutcome::InfrastructureError,
					},
					false,
				);
			}
			return Err(ToolRuntimeError::infrastructure(
				ToolInfrastructureError::backend(),
			));
		},
		Err(error) => {
			if backend_label == ToolBackendLabel::E2b {
				let outcome = if error == ToolInfrastructureError::Timeout {
					SandboxOperationOutcome::Timeout
				} else {
					SandboxOperationOutcome::Failure
				};
				telemetry.finish_sandbox_operation(
					sandbox_operation_index,
					SandboxOperation::Execute,
					outcome,
				);
			}
			let outcome = if error == ToolInfrastructureError::Timeout {
				ToolExecutionOutcome::Timeout
			} else {
				ToolExecutionOutcome::InfrastructureError
			};
			for call in &calls {
				telemetry.record(
					call.execution_index,
					ToolExecutionRecord {
						tool: call.tool.clone(),
						backend: backend_label,
						outcome,
					},
					false,
				);
			}
			return Err(ToolRuntimeError::infrastructure(error));
		},
	};

	let outputs = calls
		.into_iter()
		.zip(results)
		.map(|(call, result)| {
			let outcome = match &result {
				ToolExecutionResult::ApplicationError(_) => ToolExecutionOutcome::ApplicationError,
				ToolExecutionResult::Function(_)
				| ToolExecutionResult::WebSearch(_)
				| ToolExecutionResult::Python(_) => ToolExecutionOutcome::Success,
			};
			let output = result.into_model_output(max_output_bytes)?;
			let truncated = output
				.get("truncated")
				.and_then(Value::as_bool)
				.unwrap_or(false);
			let output = serde_json::to_string(&output)
				.map_err(|_| ToolRuntimeError::infrastructure(ToolInfrastructureError::internal()))?;
			telemetry.record(
				call.execution_index,
				ToolExecutionRecord {
					tool: call.tool,
					backend: backend_label,
					outcome,
				},
				truncated,
			);
			let value = json!({
				"type": "function_call_output",
				"call_id": call.call.call_id,
				"output": output,
			});
			Ok(IndexedOutput {
				index: call.index,
				value,
			})
		})
		.collect::<Result<Vec<_>, _>>();
	if backend_label == ToolBackendLabel::E2b {
		telemetry.finish_sandbox_operation(
			sandbox_operation_index,
			SandboxOperation::Execute,
			if outputs.is_ok() {
				SandboxOperationOutcome::Success
			} else {
				SandboxOperationOutcome::Failure
			},
		);
	}
	outputs
}

fn finish_sandbox_metadata(
	telemetry: &ToolRuntimeTelemetry,
	execution_index: usize,
	metadata: ToolBatchMetadata,
	fallback: Option<SandboxOperationOutcome>,
) {
	for (operation, report) in [
		(SandboxOperation::Create, metadata.sandbox_create),
		(SandboxOperation::Execute, metadata.sandbox_execute),
		(SandboxOperation::Terminate, metadata.sandbox_terminate),
	] {
		if let Some(report) = report {
			telemetry.finish_sandbox_operation_with_duration(
				execution_index,
				operation,
				match report.outcome {
					SandboxCleanupOutcome::Success => SandboxOperationOutcome::Success,
					SandboxCleanupOutcome::Failure => SandboxOperationOutcome::Failure,
				},
				report.duration,
			);
		} else if let Some(fallback) = fallback {
			telemetry.finish_started_sandbox_operation(execution_index, operation, fallback);
		}
	}
	let Some(outcome) = metadata.sandbox_cleanup else {
		if let Some(fallback) = fallback {
			telemetry.finish_sandbox_operation(execution_index, SandboxOperation::Cleanup, fallback);
		} else {
			telemetry.abandon_sandbox_operation(execution_index, SandboxOperation::Cleanup);
		}
		return;
	};
	let outcome = match outcome {
		SandboxCleanupOutcome::Success => SandboxOperationOutcome::Success,
		SandboxCleanupOutcome::Failure => SandboxOperationOutcome::Failure,
	};
	if let Some(duration) = metadata.sandbox_cleanup_duration {
		telemetry.finish_sandbox_operation_with_duration(
			execution_index,
			SandboxOperation::Cleanup,
			outcome,
			duration,
		);
	} else {
		telemetry.finish_started_sandbox_operation(execution_index, SandboxOperation::Cleanup, outcome);
	}
}
