use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use agent_core::prelude::Strng;
use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue, LabelValueEncoder};

use crate::telemetry::metrics::Metrics;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ToolBackendLabel {
	Http,
	E2b,
	RemoteMcp,
}

impl ToolBackendLabel {
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Http => "http",
			Self::E2b => "e2b",
			Self::RemoteMcp => "remote_mcp",
		}
	}
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ToolExecutionOutcome {
	Queued,
	Executing,
	Success,
	ApplicationError,
	InfrastructureError,
	Timeout,
	Cancelled,
}

impl ToolExecutionOutcome {
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Queued => "queued",
			Self::Executing => "executing",
			Self::Success => "success",
			Self::ApplicationError => "application_error",
			Self::InfrastructureError => "infrastructure_error",
			Self::Timeout => "timeout",
			Self::Cancelled => "cancelled",
		}
	}
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ToolRuntimeOutcome {
	Success,
	ApplicationError,
	InvalidRequest,
	InfrastructureError,
	Timeout,
	Cancelled,
}

impl ToolRuntimeOutcome {
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Success => "success",
			Self::ApplicationError => "application_error",
			Self::InvalidRequest => "invalid_request",
			Self::InfrastructureError => "infrastructure_error",
			Self::Timeout => "timeout",
			Self::Cancelled => "cancelled",
		}
	}
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum SandboxOperation {
	Create,
	Execute,
	Terminate,
	Cleanup,
}

impl SandboxOperation {
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Create => "create",
			Self::Execute => "execute",
			Self::Terminate => "terminate",
			Self::Cleanup => "cleanup",
		}
	}
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ToolRuntimeLimit {
	Deadline,
	Rounds,
	ToolCalls,
	Arguments,
	Output,
	SandboxBatch,
	SandboxCode,
	SandboxCallId,
}

impl ToolRuntimeLimit {
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Deadline => "deadline",
			Self::Rounds => "rounds",
			Self::ToolCalls => "tool_calls",
			Self::Arguments => "arguments",
			Self::Output => "output",
			Self::SandboxBatch => "sandbox_batch",
			Self::SandboxCode => "sandbox_code",
			Self::SandboxCallId => "sandbox_call_id",
		}
	}
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum SandboxOperationOutcome {
	Success,
	Failure,
	Timeout,
	Cancelled,
}

impl SandboxOperationOutcome {
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Success => "success",
			Self::Failure => "failure",
			Self::Timeout => "timeout",
			Self::Cancelled => "cancelled",
		}
	}
}

macro_rules! encode_bounded_label {
	($label:ty) => {
		impl EncodeLabelValue for $label {
			fn encode(&self, encoder: &mut LabelValueEncoder) -> Result<(), std::fmt::Error> {
				use std::fmt::Write;
				encoder.write_str(self.as_str())
			}
		}
	};
}

encode_bounded_label!(ToolBackendLabel);
encode_bounded_label!(ToolExecutionOutcome);
encode_bounded_label!(ToolRuntimeOutcome);
encode_bounded_label!(SandboxOperation);
encode_bounded_label!(SandboxOperationOutcome);
encode_bounded_label!(ToolRuntimeLimit);

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ToolRuntimeOutcomeLabels {
	pub outcome: ToolRuntimeOutcome,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ToolCallLabels {
	pub tool: String,
	pub backend: ToolBackendLabel,
	pub outcome: ToolExecutionOutcome,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ToolCallDurationLabels {
	pub tool: String,
	pub backend: ToolBackendLabel,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct SandboxOperationLabels {
	pub operation: SandboxOperation,
	pub outcome: SandboxOperationOutcome,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct SandboxOperationDurationLabels {
	pub operation: SandboxOperation,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ToolRuntimeLimitLabels {
	pub limit: ToolRuntimeLimit,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ToolLabels {
	pub tool: String,
}

/// Content-free execution observation. `tool` comes from operator configuration;
/// model-controlled call IDs and arguments are deliberately absent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolExecutionRecord {
	pub tool: Strng,
	pub backend: ToolBackendLabel,
	pub outcome: ToolExecutionOutcome,
}

struct StartedCall {
	started: Instant,
	_span: tracing::Span,
	tool: Strng,
	backend: ToolBackendLabel,
	executing: bool,
}

#[derive(Default)]
struct ToolRuntimeTelemetryState {
	records: BTreeMap<usize, ToolExecutionRecord>,
	started_calls: HashMap<usize, StartedCall>,
	started_rounds: HashMap<usize, (Instant, tracing::Span)>,
	recorded_rounds: BTreeSet<usize>,
	started_sandbox_operations: HashMap<(usize, SandboxOperation), Instant>,
	recorded_sandbox_operations: BTreeSet<(usize, SandboxOperation)>,
}

#[derive(Clone)]
pub(crate) struct ToolRuntimeTelemetry {
	state: Arc<Mutex<ToolRuntimeTelemetryState>>,
	metrics: Option<Arc<Metrics>>,
	request_recorded: Arc<AtomicBool>,
}

impl Default for ToolRuntimeTelemetry {
	fn default() -> Self {
		Self {
			state: Arc::new(Mutex::new(ToolRuntimeTelemetryState::default())),
			metrics: None,
			request_recorded: Arc::new(AtomicBool::new(false)),
		}
	}
}

impl ToolRuntimeTelemetry {
	pub(crate) fn new(metrics: Arc<Metrics>) -> Self {
		Self {
			metrics: Some(metrics),
			..Self::default()
		}
	}

	fn lock_state(&self) -> MutexGuard<'_, ToolRuntimeTelemetryState> {
		self
			.state
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner)
	}

	pub(crate) fn start_model_round(&self, round: usize) {
		let mut unused = Some((
			Instant::now(),
			tracing::info_span!("agentgateway.tool_runtime.model_round"),
		));
		{
			let mut state = self.lock_state();
			if !state.recorded_rounds.contains(&round) {
				state
					.started_rounds
					.entry(round)
					.or_insert_with(|| unused.take().expect("new model round has a span"));
			}
		}
		drop(unused);
	}

	pub(crate) fn record_model_round(
		&self,
		round: usize,
		outcome: ToolRuntimeOutcome,
		duration: Duration,
	) {
		let (recorded, span) = {
			let mut state = self.lock_state();
			if !state.recorded_rounds.insert(round) {
				(false, None)
			} else {
				(
					true,
					state.started_rounds.remove(&round).map(|(_, span)| span),
				)
			}
		};
		drop(span);
		if !recorded {
			return;
		}
		self.observe_model_round(outcome, duration);
	}

	fn observe_model_round(&self, outcome: ToolRuntimeOutcome, duration: Duration) {
		let Some(metrics) = &self.metrics else {
			return;
		};
		metrics
			.tool_runtime_model_rounds
			.get_or_create(&ToolRuntimeOutcomeLabels { outcome })
			.inc();
		metrics
			.tool_runtime_model_round_duration
			.observe(duration.as_secs_f64());
	}

	pub(crate) fn start_call(&self, execution_index: usize, tool: Strng, backend: ToolBackendLabel) {
		let queued = ToolExecutionRecord {
			tool: tool.clone(),
			backend,
			outcome: ToolExecutionOutcome::Queued,
		};
		let mut unused = Some(StartedCall {
			started: Instant::now(),
			_span: tracing::info_span!("agentgateway.tool_runtime.call"),
			tool,
			backend,
			executing: false,
		});
		let mut inserted = false;
		{
			let mut state = self.lock_state();
			if !state.records.contains_key(&execution_index) {
				let before = unused.is_some();
				state
					.started_calls
					.entry(execution_index)
					.or_insert_with(|| unused.take().expect("new tool call has a span"));
				inserted = before && unused.is_none();
			}
		}
		drop(unused);
		if inserted {
			self.observe_call_transition(queued);
		}
	}

	pub(crate) fn mark_call_executing(&self, execution_index: usize) {
		let record = {
			let mut state = self.lock_state();
			let Some(started) = state.started_calls.get_mut(&execution_index) else {
				return;
			};
			if started.executing {
				return;
			}
			started.executing = true;
			ToolExecutionRecord {
				tool: started.tool.clone(),
				backend: started.backend,
				outcome: ToolExecutionOutcome::Executing,
			}
		};
		self.observe_call_transition(record);
	}

	pub(crate) fn record(
		&self,
		execution_index: usize,
		record: ToolExecutionRecord,
		truncated: bool,
	) {
		let mut started = None;
		let recorded = {
			let mut state = self.lock_state();
			if state.records.contains_key(&execution_index) {
				false
			} else {
				started = state.started_calls.remove(&execution_index);
				state.records.insert(execution_index, record.clone());
				true
			}
		};
		if !recorded {
			return;
		}
		let duration = started
			.as_ref()
			.map_or(Duration::ZERO, |started| started.started.elapsed());
		drop(started);
		self.observe_call(record, truncated, duration);
	}

	fn observe_call(&self, record: ToolExecutionRecord, truncated: bool, duration: Duration) {
		let Some(metrics) = &self.metrics else {
			return;
		};
		let tool = record.tool.to_string();
		metrics
			.tool_runtime_calls
			.get_or_create(&ToolCallLabels {
				tool: tool.clone(),
				backend: record.backend,
				outcome: record.outcome,
			})
			.inc();
		metrics
			.tool_runtime_call_duration
			.get_or_create(&ToolCallDurationLabels {
				tool: tool.clone(),
				backend: record.backend,
			})
			.observe(duration.as_secs_f64());
		if truncated {
			metrics
				.tool_runtime_output_truncations
				.get_or_create(&ToolLabels { tool })
				.inc();
		}
	}

	fn observe_call_transition(&self, record: ToolExecutionRecord) {
		let Some(metrics) = &self.metrics else {
			return;
		};
		metrics
			.tool_runtime_calls
			.get_or_create(&ToolCallLabels {
				tool: record.tool.to_string(),
				backend: record.backend,
				outcome: record.outcome,
			})
			.inc();
	}

	pub(crate) fn record_limit(&self, limit: ToolRuntimeLimit) {
		if let Some(metrics) = &self.metrics {
			metrics
				.tool_runtime_limit_exhaustions
				.get_or_create(&ToolRuntimeLimitLabels { limit })
				.inc();
		}
	}

	pub(crate) fn record_request(&self, outcome: ToolRuntimeOutcome) {
		if self
			.request_recorded
			.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
			.is_err()
		{
			return;
		}
		if let Some(metrics) = &self.metrics {
			metrics
				.tool_runtime_requests
				.get_or_create(&ToolRuntimeOutcomeLabels { outcome })
				.inc();
		}
	}

	pub(crate) fn start_sandbox_operation(
		&self,
		execution_index: usize,
		operation: SandboxOperation,
	) {
		let key = (execution_index, operation);
		let mut state = self.lock_state();
		if !state.recorded_sandbox_operations.contains(&key) {
			state
				.started_sandbox_operations
				.entry(key)
				.or_insert_with(Instant::now);
		}
	}

	pub(crate) fn finish_sandbox_operation(
		&self,
		execution_index: usize,
		operation: SandboxOperation,
		outcome: SandboxOperationOutcome,
	) {
		self.finish_sandbox_operation_inner((execution_index, operation), outcome, None, false);
	}

	pub(crate) fn finish_sandbox_operation_with_duration(
		&self,
		execution_index: usize,
		operation: SandboxOperation,
		outcome: SandboxOperationOutcome,
		duration: Duration,
	) {
		self.finish_sandbox_operation_inner(
			(execution_index, operation),
			outcome,
			Some(duration),
			false,
		);
	}

	pub(crate) fn finish_started_sandbox_operation(
		&self,
		execution_index: usize,
		operation: SandboxOperation,
		outcome: SandboxOperationOutcome,
	) {
		self.finish_sandbox_operation_inner((execution_index, operation), outcome, None, true);
	}

	fn finish_sandbox_operation_inner(
		&self,
		key: (usize, SandboxOperation),
		outcome: SandboxOperationOutcome,
		duration: Option<Duration>,
		require_started: bool,
	) {
		let started = {
			let mut state = self.lock_state();
			if (require_started && !state.started_sandbox_operations.contains_key(&key))
				|| !state.recorded_sandbox_operations.insert(key)
			{
				return;
			}
			state.started_sandbox_operations.remove(&key)
		};
		let duration = duration
			.or_else(|| started.map(|started| started.elapsed()))
			.unwrap_or_default();
		self.observe_sandbox_operation(key.1, outcome, duration);
	}

	pub(crate) fn abandon_sandbox_operation(
		&self,
		execution_index: usize,
		operation: SandboxOperation,
	) {
		self
			.lock_state()
			.started_sandbox_operations
			.remove(&(execution_index, operation));
	}

	pub(crate) fn observe_sandbox_operation(
		&self,
		operation: SandboxOperation,
		outcome: SandboxOperationOutcome,
		duration: Duration,
	) {
		if let Some(metrics) = &self.metrics {
			metrics
				.tool_runtime_sandbox_operations
				.get_or_create(&SandboxOperationLabels { operation, outcome })
				.inc();
			metrics
				.tool_runtime_sandbox_operation_duration
				.get_or_create(&SandboxOperationDurationLabels { operation })
				.observe(duration.as_secs_f64());
		}
	}

	pub(crate) fn finish_unfinished(&self) {
		let mut rounds = HashMap::new();
		let mut calls = HashMap::new();
		let mut sandbox_operations = HashMap::new();
		let (recorded_rounds, recorded_calls, recorded_sandbox_operations) = {
			let mut state = self.lock_state();
			std::mem::swap(&mut rounds, &mut state.started_rounds);
			std::mem::swap(&mut calls, &mut state.started_calls);
			std::mem::swap(
				&mut sandbox_operations,
				&mut state.started_sandbox_operations,
			);
			let recorded_rounds = rounds
				.keys()
				.copied()
				.filter(|round| state.recorded_rounds.insert(*round))
				.collect::<BTreeSet<_>>();
			let recorded_calls = calls
				.iter()
				.filter_map(|(execution_index, started)| {
					if state.records.contains_key(execution_index) {
						return None;
					}
					state.records.insert(
						*execution_index,
						ToolExecutionRecord {
							tool: started.tool.clone(),
							backend: started.backend,
							outcome: ToolExecutionOutcome::Cancelled,
						},
					);
					Some(*execution_index)
				})
				.collect::<BTreeSet<_>>();
			let recorded_sandbox_operations = sandbox_operations
				.keys()
				.copied()
				.filter(|key| state.recorded_sandbox_operations.insert(*key))
				.collect::<BTreeSet<_>>();
			(recorded_rounds, recorded_calls, recorded_sandbox_operations)
		};

		for (round, (started, span)) in rounds {
			let duration = started.elapsed();
			drop(span);
			if recorded_rounds.contains(&round) {
				self.observe_model_round(ToolRuntimeOutcome::Cancelled, duration);
			}
		}
		for (execution_index, started) in calls {
			let duration = started.started.elapsed();
			let record = ToolExecutionRecord {
				tool: started.tool.clone(),
				backend: started.backend,
				outcome: ToolExecutionOutcome::Cancelled,
			};
			drop(started);
			if recorded_calls.contains(&execution_index) {
				self.observe_call(record, false, duration);
			}
		}
		for ((execution_index, operation), started) in sandbox_operations {
			if recorded_sandbox_operations.contains(&(execution_index, operation)) {
				self.observe_sandbox_operation(
					operation,
					SandboxOperationOutcome::Cancelled,
					started.elapsed(),
				);
			}
		}
	}

	#[cfg(test)]
	pub(crate) fn records(&self) -> Vec<ToolExecutionRecord> {
		self.lock_state().records.values().cloned().collect()
	}
}

#[cfg(test)]
mod tests {
	use std::collections::HashMap;
	use std::panic::{AssertUnwindSafe, catch_unwind};
	use std::sync::atomic::AtomicUsize;
	use std::sync::{TryLockError, atomic};

	use prometheus_client::encoding::prometheus_protobuf;
	use prometheus_client::registry::Registry as PrometheusRegistry;
	use tracing::Subscriber;
	use tracing_subscriber::Layer;
	use tracing_subscriber::layer::{Context, SubscriberExt};

	use super::*;
	use crate::HistogramMode;
	use crate::llm::tool_runtime::{SandboxCleanupOutcome, ToolBatchMetadata};
	use crate::telemetry::metrics::Metrics;

	fn runtime_budget(telemetry: ToolRuntimeTelemetry) -> super::super::RuntimeBudget {
		super::super::RuntimeBudget {
			deadline: tokio::time::Instant::now() + Duration::from_secs(30),
			max_rounds: 4,
			rounds: 0,
			max_tool_calls: 4,
			tool_calls: 0,
			max_parallel_tool_calls: 2,
			max_output_bytes: 1024,
			request_id: None,
			backends: HashMap::new(),
			program_sandbox: None,
			program_sandbox_executions: 0,
			next_execution_index: 0,
			telemetry,
		}
	}

	fn telemetry_with_metrics() -> (PrometheusRegistry, ToolRuntimeTelemetry, Arc<Metrics>) {
		let mut registry = PrometheusRegistry::default();
		let metrics = Arc::new(Metrics::new(
			agent_core::metrics::sub_registry(&mut registry),
			Default::default(),
			HistogramMode::Classic,
		));
		(
			registry,
			ToolRuntimeTelemetry::new(metrics.clone()),
			metrics,
		)
	}

	fn sandbox_duration(registry: &PrometheusRegistry) -> (u64, f64) {
		let families = prometheus_protobuf::encode(registry).unwrap();
		let histogram = families
			.iter()
			.find(|family| family.name == "agentgateway_tool_runtime_sandbox_operation_duration_seconds")
			.and_then(|family| family.metric.first())
			.and_then(|metric| metric.histogram.as_ref())
			.expect("cleanup duration histogram is recorded");
		(histogram.sample_count, histogram.sample_sum)
	}

	fn cleanup_metadata(duration: Option<Duration>) -> ToolBatchMetadata {
		ToolBatchMetadata {
			sandbox_cleanup: Some(SandboxCleanupOutcome::Success),
			sandbox_cleanup_duration: duration,
			..Default::default()
		}
	}

	fn start_cleanup_at(telemetry: &ToolRuntimeTelemetry, execution_index: usize, started: Instant) {
		telemetry
			.lock_state()
			.started_sandbox_operations
			.insert((execution_index, SandboxOperation::Cleanup), started);
	}

	#[test]
	fn sandbox_cleanup_metadata_explicit_duration_takes_precedence() {
		let (registry, telemetry, _metrics) = telemetry_with_metrics();
		start_cleanup_at(&telemetry, 1, Instant::now() - Duration::from_secs(30));

		super::super::finish_sandbox_metadata(
			&telemetry,
			1,
			cleanup_metadata(Some(Duration::from_secs(2))),
			None,
		);

		assert_eq!(sandbox_duration(&registry), (1, 2.0));
	}

	#[test]
	fn sandbox_cleanup_metadata_without_duration_uses_recorded_start() {
		let (registry, telemetry, _metrics) = telemetry_with_metrics();
		start_cleanup_at(&telemetry, 2, Instant::now() - Duration::from_secs(5));

		super::super::finish_sandbox_metadata(&telemetry, 2, cleanup_metadata(None), None);

		let (count, sum) = sandbox_duration(&registry);
		assert_eq!(count, 1);
		assert!(
			sum >= 5.0,
			"recorded start supplies elapsed duration: {sum}"
		);
	}

	#[test]
	fn sandbox_cleanup_metadata_duplicate_is_suppressed() {
		let (registry, telemetry, metrics) = telemetry_with_metrics();
		start_cleanup_at(&telemetry, 3, Instant::now() - Duration::from_secs(30));

		super::super::finish_sandbox_metadata(
			&telemetry,
			3,
			cleanup_metadata(Some(Duration::from_secs(2))),
			None,
		);
		super::super::finish_sandbox_metadata(
			&telemetry,
			3,
			cleanup_metadata(Some(Duration::from_secs(9))),
			None,
		);

		assert_eq!(sandbox_duration(&registry), (1, 2.0));
		assert_eq!(
			metrics
				.tool_runtime_sandbox_operations
				.get_or_create(&SandboxOperationLabels {
					operation: SandboxOperation::Cleanup,
					outcome: SandboxOperationOutcome::Success,
				})
				.get(),
			1
		);
	}

	#[derive(Clone)]
	struct ReentrantCloseLayer {
		state: Arc<Mutex<ToolRuntimeTelemetryState>>,
		closes: Arc<AtomicUsize>,
		blocked: Arc<AtomicUsize>,
	}

	impl<S> Layer<S> for ReentrantCloseLayer
	where
		S: Subscriber,
	{
		fn on_close(&self, _id: tracing::span::Id, _context: Context<'_, S>) {
			self.closes.fetch_add(1, atomic::Ordering::SeqCst);
			if let Err(TryLockError::WouldBlock) = self.state.try_lock() {
				self.blocked.fetch_add(1, atomic::Ordering::SeqCst);
			}
		}
	}

	#[test]
	fn span_close_callbacks_are_reentrant_through_actual_budget_drop() {
		let telemetry = ToolRuntimeTelemetry::default();
		let closes = Arc::new(AtomicUsize::new(0));
		let blocked = Arc::new(AtomicUsize::new(0));
		let subscriber = tracing_subscriber::registry().with(ReentrantCloseLayer {
			state: telemetry.state.clone(),
			closes: closes.clone(),
			blocked: blocked.clone(),
		});

		tracing::subscriber::with_default(subscriber, || {
			let budget = runtime_budget(telemetry.clone());
			telemetry.start_model_round(1);
			telemetry.start_model_round(1);
			telemetry.start_call(0, Strng::from("operator_tool"), ToolBackendLabel::Http);
			telemetry.start_call(0, Strng::from("operator_tool"), ToolBackendLabel::Http);

			telemetry.start_model_round(2);
			telemetry.record_model_round(2, ToolRuntimeOutcome::Success, Duration::from_millis(1));
			telemetry.start_call(1, Strng::from("operator_tool"), ToolBackendLabel::Http);
			telemetry.record(
				1,
				ToolExecutionRecord {
					tool: Strng::from("operator_tool"),
					backend: ToolBackendLabel::Http,
					outcome: ToolExecutionOutcome::Success,
				},
				false,
			);

			telemetry.start_model_round(3);
			telemetry.start_call(2, Strng::from("operator_tool"), ToolBackendLabel::Http);
			drop(budget);
		});

		assert!(
			closes.load(atomic::Ordering::SeqCst) >= 6,
			"all duplicate, finished, and dropped spans must be enabled"
		);
		assert_eq!(
			blocked.load(atomic::Ordering::SeqCst),
			0,
			"span close callback could not reenter the telemetry state"
		);
	}

	#[derive(Clone)]
	struct OneShotPanickingCloseLayer {
		state: Arc<Mutex<ToolRuntimeTelemetryState>>,
		panicked: Arc<AtomicBool>,
	}

	impl<S> Layer<S> for OneShotPanickingCloseLayer
	where
		S: Subscriber,
	{
		fn on_close(&self, _id: tracing::span::Id, _context: Context<'_, S>) {
			assert!(
				self.state.try_lock().is_ok(),
				"span close callback ran under the telemetry state lock"
			);
			if !self.panicked.swap(true, atomic::Ordering::SeqCst) {
				panic!("deliberate one-shot close callback panic");
			}
		}
	}

	#[test]
	fn span_close_panic_during_actual_budget_drop_does_not_poison_state() {
		let telemetry = ToolRuntimeTelemetry::default();
		let subscriber = tracing_subscriber::registry().with(OneShotPanickingCloseLayer {
			state: telemetry.state.clone(),
			panicked: Arc::new(AtomicBool::new(false)),
		});

		let result = catch_unwind(AssertUnwindSafe(|| {
			tracing::subscriber::with_default(subscriber, || {
				let budget = runtime_budget(telemetry.clone());
				telemetry.start_model_round(1);
				drop(budget);
			});
		}));
		assert!(
			result.is_err(),
			"subscriber close panic must reach the caller"
		);
		assert!(
			telemetry.state.try_lock().is_ok(),
			"close-time panic poisoned telemetry state"
		);
	}

	#[test]
	fn unfinished_cleanup_tolerates_a_poisoned_state_mutex() {
		let telemetry = ToolRuntimeTelemetry::default();
		telemetry.start_model_round(1);
		telemetry.start_call(0, Strng::from("operator_tool"), ToolBackendLabel::Http);
		let state = telemetry.state.clone();
		assert!(
			catch_unwind(AssertUnwindSafe(|| {
				let _guard = state.lock().unwrap();
				panic!("deliberately poison telemetry state");
			}))
			.is_err()
		);

		assert!(
			catch_unwind(AssertUnwindSafe(|| telemetry.finish_unfinished())).is_ok(),
			"Drop-time telemetry must not panic while unwinding"
		);
	}
}
