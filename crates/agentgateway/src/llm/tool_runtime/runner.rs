use std::fmt;
use std::time::Instant;

use super::conversation::{ManagedConversation, ModelRoundTrip};
use super::{
	CollectedToolCalls, ProgramOutcome, ProgramReplayEntry, RuntimeBudget, TOOL_SEARCH_FUNCTION,
	ToolApplicationError, ToolExecutionResult, ToolRuntimeError, ToolRuntimeOutcome,
	build_sandbox_request, execute_batch, fit_replay_entry, has_replay_entry_capacity,
	parse_sandbox_outcome,
};
use crate::http::Response;
pub(crate) enum ModelRound<R> {
	Success(Box<R>),
	UpstreamError(Box<Response>),
	InfrastructureError(Box<Response>),
}

pub(crate) enum RunError {
	Runtime(ToolRuntimeError),
	UpstreamResponse(Box<Response>),
	DirectResponse(Box<Response>),
}

impl fmt::Debug for RunError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Runtime(error) => formatter.debug_tuple("Runtime").field(error).finish(),
			Self::UpstreamResponse(response) => formatter
				.debug_struct("UpstreamResponse")
				.field("status", &response.status())
				.finish(),
			Self::DirectResponse(response) => formatter
				.debug_struct("DirectResponse")
				.field("status", &response.status())
				.finish(),
		}
	}
}

pub(crate) async fn run<C, T>(
	mut conversation: C,
	mut budget: RuntimeBudget,
	round_trip: &mut T,
) -> Result<C::Final, RunError>
where
	C: ManagedConversation,
	T: ModelRoundTrip<C::Request, C::Round>,
{
	budget.set_managed_format(conversation.format());
	loop {
		if let Err(error) = budget.start_model_round() {
			return Err(finish_runtime_error(&mut budget, error));
		}
		let round_started = Instant::now();
		let remaining = budget.remaining();
		let round = match tokio::time::timeout(
			remaining,
			round_trip.execute_round(conversation.model_request(), remaining),
		)
		.await
		{
			Err(_) => {
				budget.record_model_round(ToolRuntimeOutcome::Timeout, round_started.elapsed());
				let error = ToolRuntimeError::deadline_exceeded();
				budget.record_error(&error);
				budget.finish_request(ToolRuntimeOutcome::Timeout);
				return Err(RunError::Runtime(error));
			},
			Ok(Err(error)) => {
				let outcome = error.telemetry_outcome();
				budget.record_model_round(outcome, round_started.elapsed());
				return Err(finish_runtime_error(&mut budget, error));
			},
			Ok(Ok(round)) => round,
		};
		let round = match round {
			ModelRound::InfrastructureError(response) => {
				budget.record_model_round(
					ToolRuntimeOutcome::InfrastructureError,
					round_started.elapsed(),
				);
				budget.finish_request(ToolRuntimeOutcome::InfrastructureError);
				return Err(RunError::DirectResponse(response));
			},
			ModelRound::UpstreamError(response) => {
				budget.record_model_round(
					ToolRuntimeOutcome::ApplicationError,
					round_started.elapsed(),
				);
				if budget.tool_calls() > 0 {
					return Err(finish_runtime_error(
						&mut budget,
						ToolRuntimeError::internal(),
					));
				}
				budget.finish_request(ToolRuntimeOutcome::ApplicationError);
				return Err(RunError::UpstreamResponse(response));
			},
			ModelRound::Success(round) => *round,
		};
		if budget.remaining().is_zero() {
			budget.record_model_round(ToolRuntimeOutcome::Timeout, round_started.elapsed());
			return Err(finish_runtime_error(
				&mut budget,
				ToolRuntimeError::deadline_exceeded(),
			));
		}
		conversation.accumulate_usage(&round);
		let collected = match conversation.collect_model_calls(&round) {
			Ok(calls) => calls,
			Err(error) => {
				let outcome = error.telemetry_outcome();
				budget.record_error(&error);
				budget.record_model_round(outcome, round_started.elapsed());
				budget.finish_request(outcome);
				return Err(RunError::Runtime(error));
			},
		};
		budget.record_model_round(ToolRuntimeOutcome::Success, round_started.elapsed());
		let calls_empty = match &collected {
			CollectedToolCalls::Direct(calls) => calls.is_empty(),
			CollectedToolCalls::Programmatic { .. } => false,
		};
		if calls_empty {
			if budget.remaining().is_zero() {
				return Err(finish_runtime_error(
					&mut budget,
					ToolRuntimeError::deadline_exceeded(),
				));
			}
			let final_response = conversation.finalize(round, &budget);
			budget.finish_request(ToolRuntimeOutcome::Success);
			return Ok(final_response);
		}
		let outputs = match Box::pin(execute_collected(&mut conversation, collected, &mut budget)).await
		{
			Ok(outputs) => outputs,
			Err(error) => return Err(finish_runtime_error(&mut budget, error)),
		};
		conversation.append_round_history(round, outputs);
	}
}

async fn execute_collected<C: ManagedConversation>(
	conversation: &mut C,
	collected: CollectedToolCalls,
	budget: &mut RuntimeBudget,
) -> Result<Vec<serde_json::Value>, ToolRuntimeError> {
	match collected {
		CollectedToolCalls::Direct(calls) => {
			Box::pin(execute_direct(conversation, calls, budget)).await
		},
		CollectedToolCalls::Programmatic { call_id, code } => {
			execute_program(conversation, call_id, code, budget).await
		},
	}
}

async fn execute_direct<C: ManagedConversation>(
	conversation: &mut C,
	calls: Vec<super::ManagedToolCall>,
	budget: &mut RuntimeBudget,
) -> Result<Vec<serde_json::Value>, ToolRuntimeError> {
	let searches = calls
		.iter()
		.filter(|call| call.internal_name == TOOL_SEARCH_FUNCTION)
		.count();
	if searches == 0 {
		let outputs = execute_batch(
			&conversation.state().registry,
			calls,
			conversation.state().parallel,
			budget,
		)
		.await?;
		return outputs
			.into_iter()
			.map(|output| conversation.adapt_batch_output_item(output))
			.collect();
	}
	if let Err(error) = budget.reserve_calls(searches) {
		budget.record_error(&error);
		return Err(error);
	}
	let mut outputs = Vec::with_capacity(calls.len());
	let mut backed = Vec::with_capacity(calls.len() - searches);
	for call in calls {
		if call.internal_name == TOOL_SEARCH_FUNCTION {
			let query = call
				.arguments
				.get("query")
				.and_then(serde_json::Value::as_str)
				.ok_or_else(ToolRuntimeError::internal)?
				.to_owned();
			let result = conversation.execute_tool_search(&query)?;
			outputs.push(conversation.tool_output_item(
				&call.call_id,
				result,
				budget.max_output_bytes(),
			)?);
		} else {
			outputs.push(serde_json::Value::Null);
			backed.push((outputs.len() - 1, call));
		}
	}
	if !backed.is_empty() {
		let (slots, calls): (Vec<usize>, Vec<super::ManagedToolCall>) = backed.into_iter().unzip();
		let executed = execute_batch(
			&conversation.state().registry,
			calls,
			conversation.state().parallel,
			budget,
		)
		.await?;
		if executed.len() != slots.len() {
			return Err(ToolRuntimeError::internal());
		}
		for (slot, output) in slots.into_iter().zip(executed) {
			outputs[slot] = conversation.adapt_batch_output_item(output)?;
		}
	}
	Ok(outputs)
}

async fn execute_program<C: ManagedConversation>(
	conversation: &C,
	call_id: agent_core::prelude::Strng,
	code: String,
	budget: &mut RuntimeBudget,
) -> Result<Vec<serde_json::Value>, ToolRuntimeError> {
	tracing::debug!(
		code_bytes = code.len(),
		"generated programmatic tool call code"
	);
	// The source inlines model-supplied tool arguments, so it stays off the content-free debug path.
	tracing::trace!(code = %code, "programmatic tool call program source");
	let mut replay = Vec::new();
	let program_execution_id = uuid::Uuid::new_v4().simple().to_string();
	let program_started = Instant::now();
	loop {
		let nonce = uuid::Uuid::new_v4().to_string();
		let request = build_sandbox_request(&code, &replay, &nonce, budget.max_output_bytes())?;
		let sandbox_started = Instant::now();
		let execution = budget.execute_program_sandbox(request).await?;
		tracing::debug!(
			replay_entries = replay.len(),
			duration_ms = sandbox_started.elapsed().as_millis(),
			"programmatic sandbox replay completed"
		);
		let stdout = match execution.result {
			ToolExecutionResult::Python(value) => value
				.get("stdout")
				.and_then(serde_json::Value::as_str)
				.ok_or_else(|| ToolRuntimeError::invalid_request("program sandbox omitted stdout"))?
				.to_owned(),
			ToolExecutionResult::ApplicationError(error) => {
				return synthetic_program_output(
					conversation,
					call_id,
					ToolExecutionResult::ApplicationError(error),
					budget.max_output_bytes(),
				);
			},
			ToolExecutionResult::Function(_) | ToolExecutionResult::WebSearch(_) => {
				return Err(ToolRuntimeError::invalid_configuration(
					"program sandbox returned a non-Python result",
				));
			},
		};
		match parse_sandbox_outcome(&stdout, &nonce, budget.max_output_bytes())? {
			ProgramOutcome::Pending(pending) => {
				if pending.sequence != replay.len() {
					return Err(ToolRuntimeError::invalid_request(
						"program sandbox returned a divergent call sequence",
					));
				}
				let mut replay_entry = ProgramReplayEntry {
					sequence: pending.sequence,
					public_name: pending.public_name.clone(),
					arguments: pending.arguments.clone(),
					output: serde_json::Value::Null,
				};
				if !has_replay_entry_capacity(&replay, &replay_entry, budget.max_output_bytes()) {
					return synthetic_program_output(
						conversation,
						call_id,
						ToolExecutionResult::ApplicationError(ToolApplicationError::new(
							"program_replay_limit",
							"program replay transcript has no capacity for another tool result",
							false,
							"",
							"",
						)),
						budget.max_output_bytes(),
					);
				}
				let call = conversation.state().resolve_programmatic_call(
					&program_execution_id,
					pending.sequence,
					pending.public_name.as_str(),
					pending.arguments.clone(),
				)?;
				let tool_started = Instant::now();
				let mut outputs =
					execute_batch(&conversation.state().registry, vec![call], false, budget).await?;
				tracing::debug!(
					sequence = pending.sequence,
					duration_ms = tool_started.elapsed().as_millis(),
					"programmatic nested tool completed"
				);
				let output = outputs.pop().ok_or_else(ToolRuntimeError::internal)?;
				let output = output
					.get("output")
					.and_then(serde_json::Value::as_str)
					.ok_or_else(ToolRuntimeError::internal)?;
				let output = serde_json::from_str(output).map_err(|_| ToolRuntimeError::internal())?;
				replay_entry.output = output;
				replay.push(fit_replay_entry(
					&replay,
					replay_entry,
					budget.max_output_bytes(),
				)?);
			},
			ProgramOutcome::Completed(value) => {
				tracing::debug!(
					replay_entries = replay.len(),
					duration_ms = program_started.elapsed().as_millis(),
					"programmatic tool call completed"
				);
				return synthetic_program_output(
					conversation,
					call_id,
					ToolExecutionResult::function(serde_json::json!({"result": value})),
					budget.max_output_bytes(),
				);
			},
			ProgramOutcome::ContractError { message } => {
				return Err(ToolRuntimeError::invalid_request(format!(
					"programmatic replay contract failed: {message}"
				)));
			},
			ProgramOutcome::ApplicationError {
				error_type,
				message,
			} => {
				return synthetic_program_output(
					conversation,
					call_id,
					ToolExecutionResult::ApplicationError(ToolApplicationError::new(
						error_type, message, false, "", "",
					)),
					budget.max_output_bytes(),
				);
			},
		}
	}
}

fn synthetic_program_output<C: ManagedConversation>(
	conversation: &C,
	call_id: agent_core::prelude::Strng,
	result: ToolExecutionResult,
	max_output_bytes: usize,
) -> Result<Vec<serde_json::Value>, ToolRuntimeError> {
	Ok(vec![conversation.tool_output_item(
		&call_id,
		result,
		max_output_bytes,
	)?])
}

fn finish_runtime_error(budget: &mut RuntimeBudget, error: ToolRuntimeError) -> RunError {
	let outcome = error.telemetry_outcome();
	budget.finish_request(outcome);
	RunError::Runtime(error)
}
