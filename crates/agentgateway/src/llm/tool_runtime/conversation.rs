use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use agent_core::prelude::Strng;
use serde_json::Value;

use super::backend::ManagedToolCall;
use super::telemetry::ManagedFormat;
use super::{
	CollectedToolCalls, ModelRound, PROGRAMMATIC_MAX_CATALOG_BYTES, ProgrammaticToolSpec,
	RuntimeBudget, RuntimeDeadline, ToolExecutionResult, ToolRegistry, ToolRuntimeError,
	checked_programmatic_catalog_add, programmatic_catalog_entry,
};

/// The half of the managed runtime that does not depend on the inbound wire format.
#[derive(Clone, Debug)]
pub(crate) struct ManagedToolState {
	pub(crate) registry: Arc<ToolRegistry>,
	pub(crate) parallel: bool,
	pub(crate) deadline: RuntimeDeadline,
	pub(crate) programmatic_requested: bool,
	pub(crate) programmatic_tools: Arc<HashMap<Strng, ProgrammaticToolSpec>>,
	pub(crate) programmatic_catalog_bytes: usize,
}

impl ManagedToolState {
	/// Serialize the program catalog, or `None` when no program runtime is requested.
	///
	/// Returns `Ok(None)` for "declare nothing" and `Err` only for a real violation, so each format
	/// can decide whether an empty catalog is fatal — it is not while remote MCP discovery is pending.
	pub(crate) fn programmatic_catalog_json(&self) -> Result<Option<String>, ToolRuntimeError> {
		if !self.programmatic_requested || self.programmatic_tools.is_empty() {
			return Ok(None);
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
		Ok(Some(catalog))
	}

	pub(crate) fn insert_programmatic_tool(
		&mut self,
		spec: ProgrammaticToolSpec,
	) -> Result<(), ToolRuntimeError> {
		if self
			.programmatic_tools
			.contains_key(spec.public_name.as_str())
		{
			return Err(ToolRuntimeError::invalid_request(
				"duplicate programmatic tool name",
			));
		}
		let next_size = checked_programmatic_catalog_add(
			self.programmatic_catalog_bytes,
			!self.programmatic_tools.is_empty(),
			&spec,
		)?;
		Arc::make_mut(&mut self.programmatic_tools).insert(spec.public_name.clone(), spec);
		self.programmatic_catalog_bytes = next_size;
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

	pub(crate) fn programmatic_requested(&self) -> bool {
		self.programmatic_requested
	}

	pub(crate) fn has_programmatic_tools(&self) -> bool {
		!self.programmatic_tools.is_empty()
	}
}

/// The half of the managed runtime that depends on the inbound wire format.
#[async_trait]
pub(crate) trait ManagedConversation: Send {
	/// Canonical request handed to the round trip each iteration.
	type Request: Send + Sync;
	/// Buffered and translated upstream round for this wire format.
	type Round: Send;
	/// Client-facing managed result for this wire format.
	type Final: Send;

	fn state(&self) -> &ManagedToolState;
	#[allow(dead_code)]
	fn state_mut(&mut self) -> &mut ManagedToolState;
	fn model_request(&self) -> &Self::Request;

	/// The inbound wire format this conversation serves, for request-level telemetry.
	fn format(&self) -> ManagedFormat;

	fn collect_model_calls(
		&self,
		round: &Self::Round,
	) -> Result<CollectedToolCalls, ToolRuntimeError>;
	fn accumulate_usage(&mut self, round: &Self::Round);
	fn append_round_history(&mut self, round: Self::Round, outputs: Vec<Value>);
	fn tool_output_item(
		&self,
		call_id: &Strng,
		result: ToolExecutionResult,
		max_output_bytes: usize,
	) -> Result<Value, ToolRuntimeError>;
	fn adapt_batch_output_item(&self, output: Value) -> Result<Value, ToolRuntimeError> {
		Ok(output)
	}
	fn finalize(self, round: Self::Round, budget: &RuntimeBudget) -> Self::Final;

	/// Gateway-executed tool search. Only the Responses route declares the search function, so a
	/// format that never declares it never reaches this.
	fn execute_tool_search(&mut self, _query: &str) -> Result<ToolExecutionResult, ToolRuntimeError> {
		Err(ToolRuntimeError::internal())
	}
}

/// One authenticated model round against a pinned upstream.
#[async_trait]
pub(crate) trait ModelRoundTrip<Q: ?Sized, R>: Send {
	async fn execute_round(
		&mut self,
		request: &Q,
		remaining: Duration,
	) -> Result<ModelRound<R>, ToolRuntimeError>;
}
