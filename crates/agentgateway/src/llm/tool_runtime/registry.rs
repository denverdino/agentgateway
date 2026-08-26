use std::collections::HashMap;
use std::sync::Arc;

use agent_core::prelude::Strng;
use serde::{Serialize, Serializer};
use serde_json::Value;

use super::schema::{ArgumentSchema, code_interpreter_parameters, web_search_parameters};
use super::{
	BuiltinTool, CODE_INTERPRETER_FUNCTION, ManagedToolConfig, RuntimeLimits, ToolBackend,
	ToolBackendConfig, ToolBackendLabel, ToolRuntimeConfig, ToolRuntimeError, WEB_SEARCH_FUNCTION,
};

#[derive(Clone)]
pub(crate) struct RequestBackend {
	pub backend: Arc<dyn ToolBackend>,
	pub label: ToolBackendLabel,
}

impl std::fmt::Debug for RequestBackend {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("RequestBackend")
			.field("label", &self.label)
			.finish_non_exhaustive()
	}
}

/// Operator-approved backend and request-scoped trusted options for one managed tool.
#[derive(Clone, Debug)]
pub(crate) struct RegisteredTool {
	pub(crate) public_name: Strng,
	pub(crate) backend: Option<ToolBackendConfig>,
	pub(crate) request_backend: Option<RequestBackend>,
	pub(crate) trusted_options: Option<Value>,
	pub(super) argument_schema: Option<Arc<ArgumentSchema>>,
}

/// Immutable lookup tables used to authorize model tool calls.
#[derive(Clone, Debug)]
pub(crate) struct ToolRegistry {
	pub(crate) limits: RuntimeLimits,
	pub(crate) by_internal_name: Arc<HashMap<Strng, RegisteredTool>>,
	pub(crate) by_public_name: Arc<HashMap<Strng, Strng>>,
	source: Arc<ToolRuntimeConfig>,
}

impl Serialize for ToolRegistry {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: Serializer,
	{
		self.source.serialize(serializer)
	}
}

impl ToolRegistry {
	pub(crate) fn compile(config: ToolRuntimeConfig) -> Result<Self, ToolRuntimeError> {
		super::validation::validate_config(&config)?;
		let source = Arc::new(config.clone());
		let mut by_internal_name = HashMap::new();
		let mut by_public_name = HashMap::new();
		let mut builtins = HashMap::new();

		for ManagedToolConfig {
			name,
			builtin,
			backend,
		} in config.tools
		{
			if name.is_empty() {
				return Err(ToolRuntimeError::invalid_configuration(
					"tool name must not be empty",
				));
			}
			if name.starts_with("_agentgateway_") {
				return Err(ToolRuntimeError::invalid_configuration(format!(
					"tool name {name} uses the reserved _agentgateway_ prefix"
				)));
			}
			match (builtin, &backend) {
				(Some(BuiltinTool::CodeInterpreter), ToolBackendConfig::Http { .. }) => {
					return Err(ToolRuntimeError::invalid_configuration(
						"codeInterpreter requires e2b",
					));
				},
				(Some(BuiltinTool::WebSearch), ToolBackendConfig::E2b { .. }) => {
					return Err(ToolRuntimeError::invalid_configuration(
						"webSearch requires http",
					));
				},
				(None, ToolBackendConfig::E2b { .. }) => {
					return Err(ToolRuntimeError::invalid_configuration(
						"ordinary functions require http",
					));
				},
				_ => {},
			}

			let public_name = Strng::from(name);
			let internal_name = match builtin {
				Some(BuiltinTool::WebSearch) => Strng::from(WEB_SEARCH_FUNCTION),
				Some(BuiltinTool::CodeInterpreter) => Strng::from(CODE_INTERPRETER_FUNCTION),
				None => public_name.clone(),
			};
			let argument_schema = match builtin {
				Some(BuiltinTool::WebSearch) => Some(
					ArgumentSchema::compile(&web_search_parameters()).map_err(|()| {
						ToolRuntimeError::invalid_configuration("builtin argument schema is invalid")
					})?,
				),
				Some(BuiltinTool::CodeInterpreter) => Some(
					ArgumentSchema::compile(&code_interpreter_parameters()).map_err(|()| {
						ToolRuntimeError::invalid_configuration("builtin argument schema is invalid")
					})?,
				),
				None => None,
			};
			if let Some(builtin) = builtin
				&& builtins.insert(builtin, ()).is_some()
			{
				return Err(ToolRuntimeError::invalid_configuration(format!(
					"duplicate builtin {builtin:?}"
				)));
			}
			if by_public_name
				.insert(public_name.clone(), internal_name.clone())
				.is_some()
			{
				return Err(ToolRuntimeError::invalid_configuration(format!(
					"duplicate tool name {public_name}"
				)));
			}
			if by_internal_name
				.insert(
					internal_name.clone(),
					RegisteredTool {
						public_name,
						backend: Some(backend),
						request_backend: None,
						trusted_options: None,
						argument_schema,
					},
				)
				.is_some()
			{
				return Err(ToolRuntimeError::invalid_configuration(format!(
					"duplicate internal tool name {internal_name}"
				)));
			}
		}

		Ok(Self {
			limits: config.limits,
			by_internal_name: Arc::new(by_internal_name),
			by_public_name: Arc::new(by_public_name),
			source,
		})
	}

	pub(crate) fn has_internal(&self, name: &str) -> bool {
		self.by_internal_name.contains_key(name)
	}

	pub(crate) fn resolves_function(&self, name: &str) -> bool {
		self
			.by_public_name
			.get(name)
			.is_some_and(|internal| internal.as_str() == name)
	}

	pub(super) fn with_request_data(
		&self,
		options: HashMap<Strng, Value>,
		argument_schemas: HashMap<Strng, Arc<ArgumentSchema>>,
	) -> Self {
		let mut by_internal_name = (*self.by_internal_name).clone();
		for (name, trusted_options) in options {
			if let Some(tool) = by_internal_name.get_mut(&name) {
				tool.trusted_options = Some(trusted_options);
			}
		}
		for (name, argument_schema) in argument_schemas {
			if let Some(tool) = by_internal_name.get_mut(&name) {
				tool.argument_schema = Some(argument_schema);
			}
		}
		Self {
			limits: self.limits.clone(),
			by_internal_name: Arc::new(by_internal_name),
			by_public_name: self.by_public_name.clone(),
			source: self.source.clone(),
		}
	}

	pub(super) fn with_remote_tool(
		&self,
		internal_name: Strng,
		public_name: Strng,
		argument_schema: Arc<ArgumentSchema>,
		trusted_options: Value,
		backend: Arc<dyn ToolBackend>,
	) -> Result<Self, ToolRuntimeError> {
		let mut by_internal_name = (*self.by_internal_name).clone();
		let mut by_public_name = (*self.by_public_name).clone();
		if by_internal_name.contains_key(internal_name.as_str())
			|| by_public_name.contains_key(public_name.as_str())
		{
			return Err(ToolRuntimeError::invalid_request(
				"duplicate imported remote MCP tool",
			));
		}
		by_public_name.insert(public_name.clone(), internal_name.clone());
		by_internal_name.insert(
			internal_name,
			RegisteredTool {
				public_name,
				backend: None,
				request_backend: Some(RequestBackend {
					backend,
					label: ToolBackendLabel::RemoteMcp,
				}),
				trusted_options: Some(trusted_options),
				argument_schema: Some(argument_schema),
			},
		);
		Ok(Self {
			limits: self.limits.clone(),
			by_internal_name: Arc::new(by_internal_name),
			by_public_name: Arc::new(by_public_name),
			source: self.source.clone(),
		})
	}

	#[cfg(test)]
	pub(crate) fn trusted_options(&self, internal_name: &str) -> Option<&Value> {
		self
			.by_internal_name
			.get(internal_name)
			.and_then(|tool| tool.trusted_options.as_ref())
	}
}
