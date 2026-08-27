mod client;

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use self::client::{RemoteCallError, RemoteClient, RemoteClientTool};
use super::backend::execute_sequentially;
use super::{
	ManagedToolCall, RemoteMcpServer, RuntimeDeadline, ToolApplicationError, ToolBackend,
	ToolBatchExecution, ToolBatchInfrastructureError, ToolExecutionContext, ToolExecutionResult,
	ToolInfrastructureError,
};
use crate::proxy::httpproxy::PolicyClient;

pub(super) const MAX_DISCOVERED_TOOLS: usize = 128;
pub(super) const MAX_DISCOVERY_PAGES: usize = 128;
pub(super) const MAX_DISCOVERY_BYTES: usize = 2 * 1024 * 1024;
pub(super) const MAX_INPUT_SCHEMA_BYTES: usize = 64 * 1024;
pub(super) const MAX_OUTPUT_SCHEMA_BYTES: usize = 64 * 1024;
pub(super) const MAX_TOOL_NAME_BYTES: usize = 128;
pub(super) const MAX_DESCRIPTION_BYTES: usize = 4096;
pub(super) const MAX_SERVER_LABEL_BYTES: usize = 128;
pub(super) const MAX_ALLOWED_TOOLS: usize = MAX_DISCOVERED_TOOLS;

#[derive(Clone, Debug)]
pub struct RemoteMcpTool {
	pub remote_name: String,
	pub description: Option<String>,
	pub input_schema: Value,
	pub output_schema: Option<Value>,
}

pub struct RemoteMcpBackend {
	client: RemoteClient,
}

impl RemoteMcpBackend {
	pub(crate) async fn connect(
		policy_client: PolicyClient,
		extensions: &::http::Extensions,
		server: &RemoteMcpServer,
		deadline: RuntimeDeadline,
	) -> Result<(Arc<Self>, Vec<RemoteMcpTool>), ToolInfrastructureError> {
		Self::connect_inner(policy_client, extensions, server, cfg!(test), deadline).await
	}

	#[cfg(test)]
	pub(crate) async fn connect_for_test(
		policy_client: PolicyClient,
		server: &RemoteMcpServer,
		deadline: RuntimeDeadline,
	) -> Result<(Arc<Self>, Vec<RemoteMcpTool>), ToolInfrastructureError> {
		Self::connect_inner(
			policy_client,
			&::http::Extensions::new(),
			server,
			true,
			deadline,
		)
		.await
	}

	async fn connect_inner(
		policy_client: PolicyClient,
		extensions: &::http::Extensions,
		server: &RemoteMcpServer,
		allow_private_http: bool,
		deadline: RuntimeDeadline,
	) -> Result<(Arc<Self>, Vec<RemoteMcpTool>), ToolInfrastructureError> {
		let connect = RemoteClient::connect(
			policy_client,
			extensions,
			&server.server_url,
			server.authorization.as_ref(),
			server.allowed_tools.as_deref(),
			allow_private_http,
			deadline.instant(),
		);
		let (client, tools) = match tokio::time::timeout_at(deadline.instant(), connect).await {
			Err(_) => return Err(ToolInfrastructureError::timeout()),
			Ok(Err(_)) if deadline.remaining().is_zero() => {
				return Err(ToolInfrastructureError::timeout());
			},
			Ok(Err(_)) => return Err(ToolInfrastructureError::backend()),
			Ok(Ok(connected)) => connected,
		};
		let tools = validate_tools(tools)?;
		Ok((Arc::new(Self { client }), tools))
	}

	async fn execute_one(
		&self,
		call: ManagedToolCall,
		context: ToolExecutionContext,
	) -> Result<ToolExecutionResult, ToolInfrastructureError> {
		let remote_name = call
			.trusted_options
			.get("remote_tool_name")
			.and_then(Value::as_str)
			.ok_or_else(ToolInfrastructureError::configuration)?;
		let arguments = call.arguments.as_object().cloned().unwrap_or_else(Map::new);
		match self
			.client
			.call_tool(remote_name, arguments, context.deadline)
			.await
		{
			Ok(result) => serde_json::to_value(result)
				.map(ToolExecutionResult::function)
				.map_err(|_| ToolInfrastructureError::internal()),
			Err(RemoteCallError::Application) => Ok(ToolExecutionResult::ApplicationError(
				ToolApplicationError::new(
					"mcp_tool_error",
					"remote MCP tool call failed",
					false,
					"",
					"",
				),
			)),
			Err(RemoteCallError::Infrastructure) => Err(ToolInfrastructureError::backend()),
		}
	}
}

fn validate_tools(
	tools: Vec<RemoteClientTool>,
) -> Result<Vec<RemoteMcpTool>, ToolInfrastructureError> {
	if tools.len() > MAX_DISCOVERED_TOOLS {
		return Err(ToolInfrastructureError::backend());
	}
	let mut names = HashSet::new();
	let mut total_bytes = 0usize;
	tools
		.into_iter()
		.map(|tool| {
			let schema_bytes = serde_json::to_vec(&tool.input_schema)
				.map_err(|_| ToolInfrastructureError::backend())?
				.len();
			let output_schema_bytes = tool
				.output_schema
				.as_ref()
				.map(serde_json::to_vec)
				.transpose()
				.map_err(|_| ToolInfrastructureError::backend())?
				.map_or(0, |schema| schema.len());
			total_bytes = total_bytes
				.checked_add(tool.name.len())
				.and_then(|total| total.checked_add(tool.description.as_ref().map_or(0, String::len)))
				.and_then(|total| total.checked_add(schema_bytes))
				.and_then(|total| total.checked_add(output_schema_bytes))
				.ok_or_else(ToolInfrastructureError::backend)?;
			if tool.name.is_empty()
				|| tool.name.len() > MAX_TOOL_NAME_BYTES
				|| !names.insert(tool.name.clone())
				|| tool
					.description
					.as_ref()
					.is_some_and(|value| value.len() > MAX_DESCRIPTION_BYTES)
				|| schema_bytes > MAX_INPUT_SCHEMA_BYTES
				|| output_schema_bytes > MAX_OUTPUT_SCHEMA_BYTES
				|| total_bytes > MAX_DISCOVERY_BYTES
			{
				return Err(ToolInfrastructureError::backend());
			}
			Ok(RemoteMcpTool {
				remote_name: tool.name,
				description: tool.description,
				input_schema: tool.input_schema,
				output_schema: tool.output_schema,
			})
		})
		.collect()
}

#[async_trait]
impl ToolBackend for RemoteMcpBackend {
	async fn execute_batch(
		&self,
		calls: Vec<ManagedToolCall>,
		context: ToolExecutionContext,
	) -> Result<ToolBatchExecution, ToolBatchInfrastructureError> {
		execute_sequentially(
			|call, context| self.execute_one(call, context),
			calls,
			context,
		)
		.await
	}
}
