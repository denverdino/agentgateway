use std::fmt;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use super::{
	CollectedToolCalls, ManagedFinalResponse, PreparedToolRuntime, ProgramOutcome,
	ProgramReplayEntry, RuntimeBudget, ToolApplicationError, ToolExecutionResult, ToolRuntimeError,
	ToolRuntimeOutcome, ToolRuntimeSummary, aggregate_usage, build_sandbox_request, execute_batch,
	finalize_managed_response, fit_replay_entry, has_replay_entry_capacity, parse_sandbox_outcome,
};
use crate::http::Response;
use crate::llm::BufferedResponsesRound;
use crate::llm::types::responses;

#[async_trait]
pub(crate) trait ResponsesRoundTrip {
	async fn execute_round(
		&mut self,
		request: &responses::Request,
		remaining: Duration,
	) -> Result<ModelRound, ToolRuntimeError>;
}

pub(crate) enum ModelRound {
	Success(Box<BufferedResponsesRound>),
	UpstreamError(Box<Response>),
	InfrastructureError(Box<Response>),
}

pub(crate) type RunResult = ManagedFinalResponse;

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

pub(crate) async fn run(
	mut runtime: PreparedToolRuntime,
	mut budget: RuntimeBudget,
	round_trip: &mut impl ResponsesRoundTrip,
) -> Result<RunResult, RunError> {
	let mut accumulated_usage: Option<responses::Usage> = None;
	loop {
		if let Err(error) = budget.start_model_round() {
			return Err(finish_runtime_error(&mut budget, error));
		}
		let round_started = Instant::now();
		let remaining = budget.remaining();
		let round = match tokio::time::timeout(
			remaining,
			round_trip.execute_round(&runtime.canonical_request, remaining),
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
		if let Some(usage) = round.response.usage.as_ref() {
			aggregate_usage(&mut accumulated_usage, usage);
		}
		let collected = match runtime.collect_model_calls(&round.response) {
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
			let summary = summary(&runtime, &budget, accumulated_usage);
			let response = finalize_managed_response(round.response, &summary);
			budget.finish_request(ToolRuntimeOutcome::Success);
			return Ok(ManagedFinalResponse {
				response,
				raw_output: round.raw_output,
				raw_upstream: round.reconstructed_upstream,
				summary,
			});
		}

		let outputs = match Box::pin(execute_collected(&runtime, collected, &mut budget)).await {
			Ok(outputs) => outputs,
			Err(error) => return Err(finish_runtime_error(&mut budget, error)),
		};
		runtime.append_round_history(round.raw_output, outputs);
	}
}

async fn execute_collected(
	runtime: &PreparedToolRuntime,
	collected: CollectedToolCalls,
	budget: &mut RuntimeBudget,
) -> Result<Vec<serde_json::Value>, ToolRuntimeError> {
	match collected {
		CollectedToolCalls::Direct(calls) => {
			execute_batch(&runtime.registry, calls, runtime.parallel, budget).await
		},
		CollectedToolCalls::Programmatic { call_id, code } => {
			execute_program(runtime, call_id, code, budget).await
		},
	}
}

async fn execute_program(
	runtime: &PreparedToolRuntime,
	call_id: agent_core::prelude::Strng,
	code: String,
	budget: &mut RuntimeBudget,
) -> Result<Vec<serde_json::Value>, ToolRuntimeError> {
	tracing::debug!(
		code_bytes = code.len(),
		"generated programmatic tool call code"
	);
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
				let call = runtime.resolve_programmatic_call(
					&program_execution_id,
					pending.sequence,
					pending.public_name.as_str(),
					pending.arguments.clone(),
				)?;
				let tool_started = Instant::now();
				let mut outputs = execute_batch(&runtime.registry, vec![call], false, budget).await?;
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

fn synthetic_program_output(
	call_id: agent_core::prelude::Strng,
	result: ToolExecutionResult,
	max_output_bytes: usize,
) -> Result<Vec<serde_json::Value>, ToolRuntimeError> {
	let output = result.into_model_output(max_output_bytes)?;
	let output = serde_json::to_string(&output).map_err(|_| ToolRuntimeError::internal())?;
	Ok(vec![serde_json::json!({
		"type": "function_call_output",
		"call_id": call_id,
		"output": output,
	})])
}

fn summary(
	runtime: &PreparedToolRuntime,
	_budget: &RuntimeBudget,
	usage: Option<responses::Usage>,
) -> ToolRuntimeSummary {
	ToolRuntimeSummary {
		usage,
		#[cfg(test)]
		rounds: _budget.rounds(),
		#[cfg(test)]
		tool_calls: _budget.tool_calls(),
		client_streaming: runtime.client_streaming,
		include_obfuscation: runtime.include_obfuscation,
		client_tools: runtime.client_tools.clone(),
	}
}

fn finish_runtime_error(budget: &mut RuntimeBudget, error: ToolRuntimeError) -> RunError {
	let outcome = error.telemetry_outcome();
	budget.finish_request(outcome);
	RunError::Runtime(error)
}
