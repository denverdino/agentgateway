mod conversation;
mod mapper;
#[allow(dead_code)]
mod stream;
#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub(crate) use mapper::prepare;
#[allow(unused_imports)]
pub(crate) use stream::encode_managed_streaming_response;

use serde_json::Value;

use super::conversation::ManagedToolState;
use crate::llm::types::messages;

/// Read and write flattened top-level request fields the typed struct does not model.
///
/// Mirrors `ResponsesRequestExt` (`tool_runtime/mod.rs:62`) so no change to `crates/llm` is needed.
#[allow(dead_code)]
pub(crate) trait MessagesRequestExt {
	fn rest_field(&self, name: &str) -> Option<&Value>;
	fn replace_rest_field(&mut self, name: &str, value: Value);
	fn remove_rest_field(&mut self, name: &str);
}

impl MessagesRequestExt for messages::Request {
	fn rest_field(&self, name: &str) -> Option<&Value> {
		self.rest.as_object()?.get(name)
	}

	fn replace_rest_field(&mut self, name: &str, value: Value) {
		if !self.rest.is_object() {
			self.rest = Value::Object(Default::default());
		}
		let rest = self
			.rest
			.as_object_mut()
			.expect("rest was just made an object");
		rest.insert(name.to_owned(), value);
	}

	fn remove_rest_field(&mut self, name: &str) {
		if let Some(rest) = self.rest.as_object_mut() {
			rest.remove(name);
		}
	}
}

#[allow(dead_code)]
pub(crate) enum MessagesActivation {
	Inactive,
	Active(Box<PreparedMessagesRuntime>),
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct PreparedMessagesRuntime {
	pub(crate) state: ManagedToolState,
	pub(crate) canonical_request: messages::Request,
	pub(crate) client_streaming: bool,
	pub(crate) accumulated_usage: Option<messages::Usage>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct MessagesRuntimeSummary {
	pub(crate) usage: messages::Usage,
	pub(crate) client_streaming: bool,
	#[cfg(test)]
	pub(crate) rounds: usize,
	#[cfg(test)]
	pub(crate) tool_calls: usize,
}

#[allow(dead_code)]
pub(crate) struct ManagedMessagesResponse {
	pub(crate) response: messages::Response,
	pub(crate) raw_upstream: crate::http::Response,
	pub(crate) summary: MessagesRuntimeSummary,
}

/// Saturating per-field usage addition across internal model rounds.
#[allow(dead_code)]
pub(crate) fn aggregate_messages_usage(aggregate: &mut messages::Usage, round: &messages::Usage) {
	aggregate.input_tokens = aggregate.input_tokens.saturating_add(round.input_tokens);
	aggregate.output_tokens = aggregate.output_tokens.saturating_add(round.output_tokens);
	aggregate.cache_creation_input_tokens = saturating_option_add(
		aggregate.cache_creation_input_tokens,
		round.cache_creation_input_tokens,
	);
	aggregate.cache_read_input_tokens = saturating_option_add(
		aggregate.cache_read_input_tokens,
		round.cache_read_input_tokens,
	);
	if aggregate.service_tier != round.service_tier {
		aggregate.service_tier = None;
	}
	remove_unaggregated_usage_breakdowns(&mut aggregate.rest);
}

const UNAGGREGATED_USAGE_BREAKDOWNS: &[&str] = &[
	"input_tokens_details",
	"output_tokens_details",
	"cache_creation",
];

fn remove_unaggregated_usage_breakdowns(rest: &mut Value) {
	let Some(fields) = rest.as_object_mut() else {
		return;
	};
	for field in UNAGGREGATED_USAGE_BREAKDOWNS {
		fields.remove(*field);
	}
	if fields.is_empty() {
		*rest = Value::Null;
	}
}

#[allow(dead_code)]
fn saturating_option_add(left: Option<u64>, right: Option<u64>) -> Option<u64> {
	match (left, right) {
		(None, None) => None,
		(left, right) => Some(left.unwrap_or(0).saturating_add(right.unwrap_or(0))),
	}
}

/// Serialize a completed managed Messages response.
///
/// Unlike the Responses path there is no raw/typed merge: `messages::Content` keeps every unknown
/// field through `#[serde(flatten)]`, so the typed value already re-serializes verbatim.
#[allow(dead_code)]
pub(crate) fn serialize_managed_response(
	response: &messages::Response,
) -> Result<Vec<u8>, serde_json::Error> {
	serde_json::to_vec(response)
}
