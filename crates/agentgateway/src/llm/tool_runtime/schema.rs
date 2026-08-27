use std::error::Error;
use std::sync::Arc;
use std::{fmt, io};

use serde_json::{Value, json};

/// A compiled JSON Schema whose diagnostics never cross the client boundary.
pub(super) struct ArgumentSchema(jsonschema::Validator);

impl ArgumentSchema {
	pub(super) fn compile(schema: &Value) -> Result<Arc<Self>, ()> {
		jsonschema::options()
			.with_retriever(RejectExternalReferences)
			.build(schema)
			.map(Self)
			.map(Arc::new)
			.map_err(|_| ())
	}

	pub(super) fn is_valid(&self, instance: &Value) -> bool {
		self.0.is_valid(instance)
	}
}

impl fmt::Debug for ArgumentSchema {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str("ArgumentSchema([COMPILED])")
	}
}

#[derive(Debug)]
struct RejectExternalReferences;

impl jsonschema::Retrieve for RejectExternalReferences {
	fn retrieve(
		&self,
		_uri: &jsonschema::Uri<String>,
	) -> Result<Value, Box<dyn Error + Send + Sync>> {
		Err(Box::new(io::Error::new(
			io::ErrorKind::Unsupported,
			"external JSON Schema references are disabled",
		)))
	}
}

pub(super) fn web_search_parameters() -> Value {
	json!({
		"type": "object",
		"properties": { "query": { "type": "string" } },
		"required": ["query"],
		"additionalProperties": false
	})
}

pub(super) fn code_interpreter_parameters() -> Value {
	json!({
		"type": "object",
		"properties": { "code": { "type": "string" } },
		"required": ["code"],
		"additionalProperties": false
	})
}
