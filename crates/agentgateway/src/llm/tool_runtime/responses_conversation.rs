use agent_core::prelude::Strng;
use async_trait::async_trait;
use serde_json::Value;

use super::conversation::{ManagedConversation, ManagedToolState};
use super::telemetry::ManagedFormat;
use super::{
	CollectedToolCalls, ManagedFinalResponse, PreparedToolRuntime, RuntimeBudget,
	ToolExecutionResult, ToolRuntimeError, ToolRuntimeSummary, aggregate_usage,
	finalize_managed_response,
};
use crate::llm::BufferedResponsesRound;
use crate::llm::types::responses;

#[async_trait]
impl ManagedConversation for PreparedToolRuntime {
	type Request = responses::Request;
	type Round = BufferedResponsesRound;
	type Final = ManagedFinalResponse;

	fn state(&self) -> &ManagedToolState {
		&self.state
	}

	fn state_mut(&mut self) -> &mut ManagedToolState {
		&mut self.state
	}

	fn model_request(&self) -> &responses::Request {
		&self.canonical_request
	}

	fn format(&self) -> ManagedFormat {
		ManagedFormat::Responses
	}

	fn collect_model_calls(
		&self,
		round: &BufferedResponsesRound,
	) -> Result<CollectedToolCalls, ToolRuntimeError> {
		PreparedToolRuntime::collect_model_calls(self, &round.response)
	}

	fn accumulate_usage(&mut self, round: &BufferedResponsesRound) {
		if let Some(usage) = round.response.usage.as_ref() {
			aggregate_usage(&mut self.accumulated_usage, usage);
		}
	}

	fn append_round_history(&mut self, round: BufferedResponsesRound, outputs: Vec<Value>) {
		PreparedToolRuntime::append_round_history(self, round.raw_output, outputs);
	}

	fn tool_output_item(
		&self,
		call_id: &Strng,
		result: ToolExecutionResult,
		max_output_bytes: usize,
	) -> Result<Value, ToolRuntimeError> {
		let output = result.into_model_output(max_output_bytes)?;
		let output = serde_json::to_string(&output).map_err(|_| ToolRuntimeError::internal())?;
		Ok(serde_json::json!({
			"type": "function_call_output",
			"call_id": call_id,
			"output": output,
		}))
	}

	#[cfg_attr(not(test), allow(unused_variables))]
	fn finalize(self, round: BufferedResponsesRound, budget: &RuntimeBudget) -> ManagedFinalResponse {
		let summary = ToolRuntimeSummary {
			usage: *self.accumulated_usage,
			#[cfg(test)]
			rounds: budget.rounds(),
			#[cfg(test)]
			tool_calls: budget.tool_calls(),
			client_streaming: self.client_streaming,
			include_obfuscation: self.include_obfuscation,
			client_tools: self.client_tools.as_deref().cloned(),
		};
		let response = finalize_managed_response(round.response, &summary);
		ManagedFinalResponse {
			response,
			raw_output: round.raw_output,
			raw_upstream: round.reconstructed_upstream,
			summary,
		}
	}

	fn execute_tool_search(&mut self, query: &str) -> Result<ToolExecutionResult, ToolRuntimeError> {
		PreparedToolRuntime::execute_tool_search(self, query)
	}
}
