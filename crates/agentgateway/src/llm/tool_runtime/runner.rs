use std::fmt;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use super::{
	ManagedFinalResponse, PreparedToolRuntime, RuntimeBudget, ToolRuntimeError, ToolRuntimeOutcome,
	ToolRuntimeSummary, aggregate_usage, execute_batch, finalize_managed_response,
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
		let calls = match runtime.collect_calls(&round.response) {
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
		if calls.is_empty() {
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

		let outputs = match execute_batch(&runtime.registry, calls, runtime.parallel, &mut budget).await
		{
			Ok(outputs) => outputs,
			Err(error) => return Err(finish_runtime_error(&mut budget, error)),
		};
		runtime.append_round_history(round.raw_output, outputs);
	}
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
