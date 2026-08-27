use std::fmt;
use std::future::Future;
use std::time::{Duration, Instant};

use agent_core::prelude::Strng;
use async_trait::async_trait;
use serde::Serialize;
use serde_json::{Map, Value, json};

use super::{ToolRuntimeError, ToolRuntimeLimit};

/// A model-selected call after the runtime has authorized its tool name.
///
/// Destinations, credentials, and other operator configuration are deliberately
/// absent. They belong to the backend implementation, not to a model call.
#[derive(Clone, Debug)]
pub struct ManagedToolCall {
	pub public_name: Strng,
	pub internal_name: Strng,
	pub call_id: Strng,
	pub arguments: Value,
	pub trusted_options: Value,
}

/// Request-scoped information a backend may use for correlation and deadlines.
#[derive(Clone, Debug, Default)]
pub struct ToolExecutionContext {
	pub request_id: Option<Strng>,
	pub deadline: Option<Instant>,
}

/// An application-level failure is returned to the model as a structured value.
#[derive(Clone, Debug, Serialize)]
pub struct ToolApplicationError {
	#[serde(rename = "type")]
	pub r#type: String,
	pub message: String,
	pub retryable: bool,
	#[serde(default)]
	pub stdout: String,
	#[serde(default)]
	pub stderr: String,
}

impl ToolApplicationError {
	pub fn new(
		error_type: impl Into<String>,
		message: impl Into<String>,
		retryable: bool,
		stdout: impl Into<String>,
		stderr: impl Into<String>,
	) -> Self {
		Self {
			r#type: error_type.into(),
			message: message.into(),
			retryable,
			stdout: stdout.into(),
			stderr: stderr.into(),
		}
	}

	pub fn execution_error(
		message: impl Into<String>,
		retryable: bool,
		stdout: impl Into<String>,
		stderr: impl Into<String>,
	) -> Self {
		Self::new("execution_error", message, retryable, stdout, stderr)
	}
}

/// An infrastructure failure terminates the request and is never model output.
///
/// The enum intentionally stores no provider response body, URL, credential, or
/// other raw detail. This makes accidental client serialization harmless.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolInfrastructureError {
	Authentication,
	Timeout,
	Backend,
	Configuration,
	Internal,
}

impl ToolInfrastructureError {
	pub const fn authentication() -> Self {
		Self::Authentication
	}

	pub const fn timeout() -> Self {
		Self::Timeout
	}

	pub const fn backend() -> Self {
		Self::Backend
	}

	pub const fn configuration() -> Self {
		Self::Configuration
	}

	pub const fn internal() -> Self {
		Self::Internal
	}

	/// Return the only representation suitable for an OpenAI-compatible client.
	pub fn to_openai_error(self) -> Value {
		json!({
			"error": {
				"type": "tool_infrastructure_error",
				"message": self.public_message(),
				"code": self.public_code(),
			}
		})
	}

	#[cfg(test)]
	pub(crate) fn sanitized_openai_error(self) -> Value {
		self.to_openai_error()
	}

	fn public_message(self) -> &'static str {
		match self {
			Self::Authentication => "managed tool backend authentication failed",
			Self::Timeout => "managed tool execution timed out",
			Self::Backend => "managed tool backend request failed",
			Self::Configuration => "managed tool backend configuration is invalid",
			Self::Internal => "managed tool runtime failed",
		}
	}

	fn public_code(self) -> &'static str {
		match self {
			Self::Authentication => "tool_backend_authentication_failed",
			Self::Timeout => "tool_execution_timeout",
			Self::Backend => "tool_backend_request_failed",
			Self::Configuration => "tool_backend_configuration_invalid",
			Self::Internal => "tool_runtime_internal_error",
		}
	}
}

impl fmt::Display for ToolInfrastructureError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(self.public_message())
	}
}

impl std::error::Error for ToolInfrastructureError {}

/// Normalized backend output. Application errors remain values; infrastructure
/// errors use the ToolBackend Rust error channel.
#[derive(Clone, Debug, Serialize)]
pub enum ToolExecutionResult {
	Function(Value),
	WebSearch(Value),
	Python(Value),
	ApplicationError(ToolApplicationError),
}

/// Adapter-owned Sandbox cleanup outcome carried outside model-visible results.
///
/// The closed enum prevents response content, Sandbox IDs, or other untrusted
/// values from becoming telemetry labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandboxCleanupOutcome {
	Success,
	Failure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SandboxOperationReport {
	pub outcome: SandboxCleanupOutcome,
	pub duration: Duration,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToolBatchMetadata {
	pub sandbox_create: Option<SandboxOperationReport>,
	pub sandbox_execute: Option<SandboxOperationReport>,
	pub sandbox_terminate: Option<SandboxOperationReport>,
	pub sandbox_cleanup: Option<SandboxCleanupOutcome>,
	pub sandbox_cleanup_duration: Option<Duration>,
}

#[derive(Clone, Debug)]
pub struct ToolBatchExecution {
	pub results: Vec<ToolExecutionResult>,
	pub metadata: ToolBatchMetadata,
}

impl ToolBatchExecution {
	pub fn new(results: Vec<ToolExecutionResult>) -> Self {
		Self {
			results,
			metadata: ToolBatchMetadata::default(),
		}
	}
}

#[derive(Clone, Debug)]
pub struct ToolBatchInfrastructureError {
	pub error: ToolInfrastructureError,
	pub metadata: ToolBatchMetadata,
}

impl ToolBatchInfrastructureError {
	pub fn new(error: ToolInfrastructureError, metadata: ToolBatchMetadata) -> Self {
		Self { error, metadata }
	}
}

impl From<ToolInfrastructureError> for ToolBatchInfrastructureError {
	fn from(error: ToolInfrastructureError) -> Self {
		Self::new(error, ToolBatchMetadata::default())
	}
}

impl ToolExecutionResult {
	pub fn web_search(value: Value) -> Self {
		Self::WebSearch(value)
	}

	pub fn python(value: Value) -> Self {
		Self::Python(value)
	}

	pub fn function(value: Value) -> Self {
		Self::Function(value)
	}

	pub fn into_model_output(self, max_bytes: usize) -> Result<Value, ToolRuntimeError> {
		let (mut output, typed) = match self {
			Self::Function(value) => (ordinary_success_output(value), true),
			Self::WebSearch(value) => (web_search_output(value)?, true),
			Self::Python(value) => (python_output(value)?, true),
			Self::ApplicationError(error) => (application_error_output(error), false),
		};
		bound_output(&mut output, max_bytes, typed)?;
		Ok(Value::Object(output))
	}
}

fn ordinary_success_output(value: Value) -> Map<String, Value> {
	let mut output = match value {
		Value::Object(map) => map,
		value => Map::from_iter([(String::from("result"), value)]),
	};
	output.insert(String::from("ok"), Value::Bool(true));
	output
}

fn web_search_output(value: Value) -> Result<Map<String, Value>, ToolRuntimeError> {
	let Value::Object(mut value) = value else {
		return Err(ToolRuntimeError::invalid_request(
			"web search result must be a JSON object",
		));
	};
	let Some(results) = value.remove("results") else {
		return Err(ToolRuntimeError::invalid_request(
			"web search result requires a results array",
		));
	};
	if !results.is_array() {
		return Err(ToolRuntimeError::invalid_request(
			"web search result results must be an array",
		));
	}
	let mut output = Map::from_iter([
		("ok".into(), Value::Bool(true)),
		("results".into(), results),
	]);
	if let Some(truncated) = value.remove("truncated") {
		if !truncated.is_boolean() {
			return Err(ToolRuntimeError::invalid_request(
				"web search result truncated must be a boolean",
			));
		}
		output.insert("truncated".into(), truncated);
	}
	if let Some(field) = value.keys().next() {
		return Err(ToolRuntimeError::invalid_request(format!(
			"web search result contains unsupported field {field}"
		)));
	}
	Ok(output)
}

fn python_output(value: Value) -> Result<Map<String, Value>, ToolRuntimeError> {
	let Value::Object(value) = value else {
		return Err(ToolRuntimeError::invalid_request(
			"python result must be a JSON object",
		));
	};
	if let Some(field) = value.keys().find(|field| {
		!matches!(
			field.as_str(),
			"exit_code" | "stdout" | "stderr" | "timed_out" | "truncated" | "artifacts"
		)
	}) {
		return Err(ToolRuntimeError::invalid_request(format!(
			"python result contains unsupported field {field}"
		)));
	}
	let required = [
		("exit_code", "an integer"),
		("stdout", "a string"),
		("stderr", "a string"),
		("timed_out", "a boolean"),
		("truncated", "a boolean"),
		("artifacts", "an array"),
	];
	for (field, expected) in required {
		let Some(value) = value.get(field) else {
			return Err(ToolRuntimeError::invalid_request(format!(
				"python result requires {field} ({expected})"
			)));
		};
		let valid = match field {
			"exit_code" => value.is_i64() || value.is_u64(),
			"stdout" | "stderr" => value.is_string(),
			"timed_out" | "truncated" => value.is_boolean(),
			"artifacts" => value.is_array(),
			_ => false,
		};
		if !valid {
			return Err(ToolRuntimeError::invalid_request(format!(
				"python result {field} must be {expected}"
			)));
		}
	}
	Ok(Map::from_iter([
		("ok".into(), Value::Bool(true)),
		("exit_code".into(), value["exit_code"].clone()),
		("stdout".into(), value["stdout"].clone()),
		("stderr".into(), value["stderr"].clone()),
		("timed_out".into(), value["timed_out"].clone()),
		("truncated".into(), value["truncated"].clone()),
		("artifacts".into(), value["artifacts"].clone()),
	]))
}

fn application_error_output(error: ToolApplicationError) -> Map<String, Value> {
	Map::from_iter([
		("ok".into(), Value::Bool(false)),
		(
			"error".into(),
			json!({
				"type": error.r#type,
				"message": error.message,
				"retryable": error.retryable,
			}),
		),
		("stdout".into(), Value::String(error.stdout)),
		("stderr".into(), Value::String(error.stderr)),
	])
}

fn bound_output(
	output: &mut Map<String, Value>,
	max_bytes: usize,
	typed: bool,
) -> Result<(), ToolRuntimeError> {
	if max_bytes == 0 {
		return Err(ToolRuntimeError::limit(
			"tool output exceeds the configured byte limit",
			ToolRuntimeLimit::Output,
		));
	}
	if serialized_len(output) <= max_bytes {
		return Ok(());
	}

	output.insert("truncated".into(), Value::Bool(true));
	while serialized_len(output) > max_bytes {
		if !trim_largest_value(output) {
			return Err(ToolRuntimeError::limit(
				"tool output exceeds the configured byte limit",
				ToolRuntimeLimit::Output,
			));
		}
	}
	if typed {
		output.insert("truncated".into(), Value::Bool(true));
	}
	Ok(())
}

pub(crate) fn bound_replay_output(
	output: &mut Value,
	max_bytes: usize,
) -> Result<(), ToolRuntimeError> {
	let Value::Object(output) = output else {
		return Err(ToolRuntimeError::internal());
	};
	bound_output(output, max_bytes, true)
}

fn serialized_len(output: &Map<String, Value>) -> usize {
	serde_json::to_vec(output).map_or(usize::MAX, |bytes| bytes.len())
}

fn trim_largest_value(value: &mut Map<String, Value>) -> bool {
	let Some((_, largest)) = value
		.iter_mut()
		.max_by_key(|(_, value)| serialized_value_len(value))
	else {
		return false;
	};
	trim_value(largest)
}

fn trim_value(value: &mut Value) -> bool {
	match value {
		Value::String(text) => {
			if text.is_empty() {
				return false;
			}
			let target = text.len().saturating_sub((text.len() / 4).max(1));
			let truncated = super::truncate_utf8_bytes(text, target);
			if truncated.len() == text.len() {
				return false;
			}
			*text = truncated;
			true
		},
		Value::Array(values) => {
			if let Some((_, largest)) = values
				.iter_mut()
				.enumerate()
				.max_by_key(|(_, value)| serialized_value_len(value))
				&& trim_value(largest)
			{
				return true;
			}
			values.pop().is_some()
		},
		Value::Object(values) => {
			if let Some((_, largest)) = values
				.iter_mut()
				.max_by_key(|(_, value)| serialized_value_len(value))
				&& trim_value(largest)
			{
				return true;
			}
			let Some(key) = values.keys().next().cloned() else {
				return false;
			};
			values.remove(&key).is_some()
		},
		Value::Null | Value::Bool(_) | Value::Number(_) => false,
	}
}

fn serialized_value_len(value: &Value) -> usize {
	serde_json::to_vec(value).map_or(usize::MAX, |bytes| bytes.len())
}

/// Parse model-supplied arguments with a byte bound and an object requirement.
pub fn parse_arguments(
	input: impl AsRef<[u8]>,
	max_bytes: usize,
) -> Result<Value, ToolRuntimeError> {
	let input = input.as_ref();
	if input.len() > max_bytes {
		return Err(ToolRuntimeError::limit(
			format!(
				"tool arguments exceed the configured {} byte limit",
				max_bytes
			),
			ToolRuntimeLimit::Arguments,
		));
	}
	let text = std::str::from_utf8(input)
		.map_err(|_| ToolRuntimeError::invalid_request("tool arguments must be valid UTF-8 JSON"))?;
	let value: Value = serde_json::from_str(text).map_err(|error| {
		ToolRuntimeError::invalid_request(format!("invalid tool arguments JSON: {error}"))
	})?;
	if !value.is_object() {
		return Err(ToolRuntimeError::invalid_request(
			"tool arguments must be a JSON object",
		));
	}
	Ok(value)
}

#[async_trait]
pub(crate) trait ToolBackend: Send + Sync {
	async fn execute_batch(
		&self,
		calls: Vec<ManagedToolCall>,
		context: ToolExecutionContext,
	) -> Result<ToolBatchExecution, ToolBatchInfrastructureError>;
}

pub(crate) async fn execute_sequentially<F, Fut>(
	mut execute_one: F,
	calls: Vec<ManagedToolCall>,
	context: ToolExecutionContext,
) -> Result<ToolBatchExecution, ToolBatchInfrastructureError>
where
	F: FnMut(ManagedToolCall, ToolExecutionContext) -> Fut,
	Fut: Future<Output = Result<ToolExecutionResult, ToolInfrastructureError>>,
{
	let mut results = Vec::with_capacity(calls.len());
	for call in calls {
		results.push(execute_one(call, context.clone()).await?);
	}
	Ok(ToolBatchExecution::new(results))
}
