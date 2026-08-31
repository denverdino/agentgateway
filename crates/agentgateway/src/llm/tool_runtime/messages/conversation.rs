use std::collections::HashSet;

use agent_core::prelude::Strng;
use async_trait::async_trait;
use serde_json::{Value, json};

use super::{MessagesRequestExt, PreparedMessagesRuntime};
use crate::llm::BufferedMessagesRound;
use crate::llm::tool_runtime::backend::ManagedToolCall;
use crate::llm::tool_runtime::conversation::{ManagedConversation, ManagedToolState};
use crate::llm::tool_runtime::messages::{ManagedMessagesResponse, MessagesRuntimeSummary};
use crate::llm::tool_runtime::telemetry::ManagedFormat;
use crate::llm::tool_runtime::{
	CollectedToolCalls, PROGRAMMATIC_FUNCTION, RuntimeBudget, SANDBOX_MAX_CALL_ID_BYTES,
	SANDBOX_MAX_CODE_BYTES, ToolExecutionResult, ToolRuntimeError, ToolRuntimeLimit,
	validate_sandbox_contract,
};
use crate::llm::types::messages;

impl PreparedMessagesRuntime {
	#[allow(dead_code)]
	pub(crate) fn collect_model_calls(
		&self,
		response: &messages::Response,
	) -> Result<CollectedToolCalls, ToolRuntimeError> {
		let declared = self
			.canonical_request
			.rest_field("tools")
			.and_then(Value::as_array)
			.into_iter()
			.flatten()
			.filter_map(|tool| tool.get("name").and_then(Value::as_str))
			.collect::<HashSet<_>>();
		let mut seen_call_ids = HashSet::new();
		let mut calls = Vec::new();
		let mut program = None;
		for block in &response.content {
			if block.rest.get("type").and_then(Value::as_str) != Some("tool_use") {
				continue;
			}
			if response.stop_reason.as_deref() == Some("max_tokens") {
				return Err(ToolRuntimeError::invalid_request(
					"model stopped at max_tokens with an incomplete tool_use block",
				));
			}
			let id = block
				.rest
				.get("id")
				.and_then(Value::as_str)
				.unwrap_or_default();
			if id.is_empty() || !seen_call_ids.insert(id) {
				return Err(ToolRuntimeError::invalid_request(
					"model generated an empty or duplicate managed tool_use id",
				));
			}
			let name = block
				.rest
				.get("name")
				.and_then(Value::as_str)
				.ok_or_else(|| {
					ToolRuntimeError::invalid_request("tool_use block requires a string name")
				})?;
			let arguments = block.rest.get("input").cloned().ok_or_else(|| {
				ToolRuntimeError::invalid_request("tool_use block requires an input object")
			})?;
			if !arguments.is_object() {
				return Err(ToolRuntimeError::invalid_request(
					"tool_use input must be an object",
				));
			}
			let serialized = serde_json::to_vec(&arguments).map_err(|_| ToolRuntimeError::internal())?;
			if serialized.len() > self.state.registry.limits.max_arguments_bytes {
				return Err(ToolRuntimeError::invalid_request(
					"tool arguments exceed the configured maxArgumentsBytes limit",
				));
			}
			if !declared.contains(name) {
				return Err(ToolRuntimeError::invalid_request(format!(
					"model generated undeclared managed tool {name}"
				)));
			}
			if name == PROGRAMMATIC_FUNCTION {
				if id.len() > SANDBOX_MAX_CALL_ID_BYTES {
					return Err(ToolRuntimeError::limit(
						"Sandbox call id exceeds the 256-byte limit",
						ToolRuntimeLimit::SandboxCallId,
					));
				}
				if program.is_some() || !calls.is_empty() {
					return Err(ToolRuntimeError::invalid_request(
						"model generated an invalid programmatic tool batch",
					));
				}
				let object = arguments.as_object().expect("checked above");
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
				program = Some((Strng::from(id), code.to_owned()));
				continue;
			}
			if program.is_some() {
				return Err(ToolRuntimeError::invalid_request(
					"model mixed programmatic and direct tool calls in one response",
				));
			}
			let registered = self
				.state
				.registry
				.by_internal_name
				.get(name)
				.ok_or_else(|| {
					ToolRuntimeError::invalid_request(format!(
						"model generated unregistered managed tool {name}"
					))
				})?;
			let argument_schema = registered.argument_schema.as_ref().ok_or_else(|| {
				ToolRuntimeError::invalid_configuration("managed tool argument schema was not compiled")
			})?;
			if !argument_schema.is_valid(&arguments) {
				return Err(ToolRuntimeError::invalid_request(
					"tool arguments do not match declared schema",
				));
			}
			calls.push(ManagedToolCall {
				public_name: registered.public_name.clone(),
				internal_name: Strng::from(name),
				call_id: Strng::from(id),
				arguments,
				trusted_options: registered
					.trusted_options
					.clone()
					.unwrap_or_else(|| Value::Object(Default::default())),
			});
		}
		if let Some((call_id, code)) = program {
			return Ok(CollectedToolCalls::Programmatic { call_id, code });
		}
		validate_sandbox_contract(&self.state.registry, &calls, self.state.parallel)?;
		Ok(CollectedToolCalls::Direct(calls))
	}
}

#[async_trait]
impl ManagedConversation for PreparedMessagesRuntime {
	type Request = messages::Request;
	type Round = BufferedMessagesRound;
	type Final = ManagedMessagesResponse;

	fn state(&self) -> &ManagedToolState {
		&self.state
	}

	fn state_mut(&mut self) -> &mut ManagedToolState {
		&mut self.state
	}

	fn model_request(&self) -> &messages::Request {
		&self.canonical_request
	}

	fn format(&self) -> ManagedFormat {
		ManagedFormat::Messages
	}

	fn collect_model_calls(
		&self,
		round: &BufferedMessagesRound,
	) -> Result<CollectedToolCalls, ToolRuntimeError> {
		PreparedMessagesRuntime::collect_model_calls(self, &round.response)
	}

	fn accumulate_usage(&mut self, round: &BufferedMessagesRound) {
		match self.accumulated_usage.as_mut() {
			Some(aggregate) => super::aggregate_messages_usage(aggregate, &round.response.usage),
			None => self.accumulated_usage = Some(round.response.usage.clone()),
		}
	}

	fn append_round_history(&mut self, round: BufferedMessagesRound, outputs: Vec<Value>) {
		let assistant = round
			.response
			.content
			.into_iter()
			.map(|block| {
				messages::ContentPart::Unknown(serde_json::to_value(block).unwrap_or(Value::Null))
			})
			.collect::<Vec<_>>();
		self
			.canonical_request
			.messages
			.push(messages::RequestMessage {
				role: "assistant".to_owned(),
				content: Some(messages::ContentBlock::Array(assistant)),
				rest: Value::Null,
			});
		self
			.canonical_request
			.messages
			.push(messages::RequestMessage {
				role: "user".to_owned(),
				content: Some(messages::ContentBlock::Array(
					outputs
						.into_iter()
						.map(messages::ContentPart::Unknown)
						.collect(),
				)),
				rest: Value::Null,
			});
		if self.canonical_request.rest_field("tool_choice") != Some(&json!({"type": "auto"})) {
			self
				.canonical_request
				.replace_rest_field("tool_choice", json!({"type": "auto"}));
		}
	}

	fn tool_output_item(
		&self,
		call_id: &Strng,
		result: ToolExecutionResult,
		max_output_bytes: usize,
	) -> Result<Value, ToolRuntimeError> {
		let is_error = matches!(result, ToolExecutionResult::ApplicationError(_));
		let output = result.into_model_output(max_output_bytes)?;
		let output = serde_json::to_string(&output).map_err(|_| ToolRuntimeError::internal())?;
		Ok(tool_result_block(call_id, output, is_error))
	}

	fn adapt_batch_output_item(&self, output: Value) -> Result<Value, ToolRuntimeError> {
		let call_id = output
			.get("call_id")
			.and_then(Value::as_str)
			.ok_or_else(ToolRuntimeError::internal)?;
		let content = output
			.get("output")
			.and_then(Value::as_str)
			.ok_or_else(ToolRuntimeError::internal)?;
		let is_error = serde_json::from_str::<Value>(content)
			.ok()
			.and_then(|value| value.get("ok").and_then(Value::as_bool))
			== Some(false);
		Ok(tool_result_block(call_id, content.to_owned(), is_error))
	}

	fn finalize(
		self,
		round: BufferedMessagesRound,
		#[cfg_attr(not(test), allow(unused_variables))] budget: &RuntimeBudget,
	) -> ManagedMessagesResponse {
		let mut response = round.response;
		if let Some(usage) = self.accumulated_usage.clone() {
			response.usage = usage;
		}
		ManagedMessagesResponse {
			summary: MessagesRuntimeSummary {
				usage: response.usage.clone(),
				client_streaming: self.client_streaming,
				#[cfg(test)]
				rounds: budget.rounds(),
				#[cfg(test)]
				tool_calls: budget.tool_calls(),
			},
			response,
			raw_upstream: round.reconstructed_upstream,
		}
	}
}

fn tool_result_block(call_id: &str, content: String, is_error: bool) -> Value {
	let mut block = json!({
		"type": "tool_result",
		"tool_use_id": call_id,
		"content": content,
	});
	if is_error && let Some(object) = block.as_object_mut() {
		object.insert("is_error".to_owned(), Value::Bool(true));
	}
	block
}
