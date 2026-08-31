use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_core::prelude::Strng;
use serde_json::{Value, json};

use super::{MessagesActivation, MessagesRequestExt, prepare};
use crate::llm::tool_runtime::backend::ToolApplicationError;
use crate::llm::tool_runtime::conversation::{ManagedConversation, ModelRoundTrip};
use crate::llm::tool_runtime::tests::{
	BatchBackend, ReplayProgramSandbox, ReplayProgramState, registry, tool,
};
use crate::llm::tool_runtime::{
	BuiltinTool, RuntimeBudget, RuntimeDeadline, ToolBackend, ToolExecutionResult, ToolRegistry,
	ToolRuntimeError,
};
use crate::llm::types::messages;
use crate::llm::{AIProvider, ChatFormat, InputFormat, RouteType};

fn anthropic() -> AIProvider {
	AIProvider::Anthropic(agent_llm::anthropic::Provider { model: None })
}

fn messages_llm_request() -> crate::llm::LLMRequest {
	crate::llm::LLMRequest {
		input_tokens: None,
		input_format: InputFormat::Messages,
		cache_convention: crate::llm::CacheTokenConvention::pending(),
		request_model: "claude-x".into(),
		provider: "test-provider".into(),
		streaming: false,
		params: Default::default(),
		prompt: None,
		provider_state: None,
	}
}

fn managed_registry() -> Arc<ToolRegistry> {
	registry(vec![
		tool("query_orders", None),
		tool("web_search", Some(BuiltinTool::WebSearch)),
		tool("code_interpreter", Some(BuiltinTool::CodeInterpreter)),
	])
}

fn declared(request: &messages::Request) -> Vec<String> {
	request
		.rest_field("tools")
		.and_then(Value::as_array)
		.into_iter()
		.flatten()
		.filter_map(|tool| tool.get("name").and_then(Value::as_str))
		.map(str::to_owned)
		.collect()
}

fn active(value: Value) -> (messages::Request, Box<super::PreparedMessagesRuntime>) {
	let mut request = request(value);
	let MessagesActivation::Active(runtime) =
		prepare(&mut request, Some(&managed_registry())).unwrap()
	else {
		panic!("request must activate the managed runtime");
	};
	(request, runtime)
}

fn request(value: Value) -> messages::Request {
	serde_json::from_value(value).expect("valid Messages request")
}

fn tool_use(id: &str, name: &str, input: Value) -> Value {
	json!({"type": "tool_use", "id": id, "name": name, "input": input})
}

fn model_response(stop_reason: &str, content: Vec<Value>) -> messages::Response {
	serde_json::from_value(json!({
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"model": "smart",
		"stop_reason": stop_reason,
		"content": content,
		"usage": {"input_tokens": 5, "output_tokens": 2}
	}))
	.expect("valid Messages response")
}

fn direct_runtime() -> Box<super::PreparedMessagesRuntime> {
	active(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "orders"}],
		"tools": [{
			"name": "query_orders",
			"input_schema": {
				"type": "object",
				"properties": {"since": {"type": "string"}},
				"required": ["since"]
			}
		}]
	}))
	.1
}

fn buffered(response: messages::Response) -> crate::llm::BufferedMessagesRound {
	crate::llm::BufferedMessagesRound {
		response,
		reconstructed_upstream: crate::http::Response::new(crate::http::Body::empty()),
	}
}

fn stream(response: messages::Response) -> String {
	String::from_utf8(super::encode_managed_streaming_response(&response).unwrap()).unwrap()
}

fn managed_messages_tools_value() -> Value {
	json!([
		{"type": "code_execution_20260120", "name": "code_execution"},
		{
			"name": "query_orders",
			"input_schema": {"type": "object", "properties": {"since": {"type": "string"}}},
			"allowed_callers": ["code_execution_20260120"]
		}
	])
}

#[tokio::test]
async fn messages_runtime_records_the_messages_format_label() {
	let (_, runtime) = active(json!({
		"model": "m",
		"max_tokens": 32,
		"messages": [{"role": "user", "content": "hi"}],
		"tools": managed_messages_tools_value()
	}));
	assert_eq!(
		runtime.format().as_str(),
		"messages",
		"the messages conversation must label its own format"
	);
}

#[test]
fn messages_translation_accepts_anthropic_and_openai_upstreams() {
	let provider = anthropic();
	let translation = provider
		.messages_translation_for_route(RouteType::Messages)
		.expect("anthropic upstream is a valid messages target");
	assert_eq!(translation.input, InputFormat::Messages);
	assert_eq!(translation.output, ChatFormat::AnthropicMessages);

	let openai = AIProvider::OpenAI(agent_llm::openai::Provider {
		model: None,
		moderation: None,
	});
	let translation = openai
		.messages_translation_for_route(RouteType::Completions)
		.expect("completions upstream is a valid messages target");
	assert_eq!(translation.output, ChatFormat::OpenAICompletions);
}

#[test]
fn messages_translation_rejects_a_non_chat_upstream_route() {
	let error = match anthropic().messages_translation_for_route(RouteType::Embeddings) {
		Ok(_) => panic!("embeddings is not a chat route"),
		Err(error) => error,
	};
	assert!(matches!(
		error,
		crate::llm::AIError::UnsupportedConversion(_)
	));
}

#[test]
fn messages_rerender_round_trips_the_canonical_request() {
	let provider = anthropic();
	let mut request = request(json!({
		"model": "claude-x",
		"max_tokens": 64,
		"messages": [{"role": "user", "content": "hi"}],
		"metadata": {"user_id": "u-1"}
	}));
	request.stream = Some(false);
	let mut llm_request = messages_llm_request();
	let body = provider
		.rerender_messages_request(
			None,
			&request,
			RouteType::Messages,
			&::http::HeaderMap::new(),
			&mut None,
			&mut llm_request,
		)
		.expect("rerender");
	let value: Value = serde_json::from_slice(&body).expect("json body");
	assert_eq!(value["model"], "claude-x");
	assert_eq!(value["stream"], false);
	// Unmodelled top-level fields survive the round trip through `rest`.
	assert_eq!(value["metadata"]["user_id"], "u-1");
}

#[test]
fn messages_rerender_uses_vertex_anthropic_renderer() {
	let provider = AIProvider::Vertex(agent_llm::vertex::Provider {
		project_id: agent_core::strng::new("test-project"),
		model: None,
		region: Some(agent_core::strng::new("us-central1")),
	});
	let request = request(json!({
		"model": "claude-haiku-4-5-20251001",
		"max_tokens": 64,
		"messages": [{
			"role": "user",
			"content": [{
				"type": "text",
				"text": "hi",
				"cache_control": {"type": "ephemeral", "scope": "tools"}
			}]
		}],
		"output_format": {"type": "json_schema", "schema": {"type": "object"}}
	}));

	let mut llm_request = messages_llm_request();
	let body = provider
		.rerender_messages_request(
			None,
			&request,
			RouteType::Messages,
			&::http::HeaderMap::new(),
			&mut None,
			&mut llm_request,
		)
		.expect("rerender");
	let value: Value = serde_json::from_slice(&body).expect("json body");

	assert!(value.get("model").is_none());
	assert_eq!(value["anthropic_version"], "vertex-2023-10-16");
	assert!(value.get("output_format").is_none());
	assert!(
		!value["messages"][0]["content"][0]["cache_control"]
			.as_object()
			.expect("cache control object")
			.contains_key("scope")
	);
}

#[test]
fn managed_messages_stream_encodes_a_text_only_final_round() {
	let mut response = model_response("end_turn", vec![json!({"type": "text", "text": "22 C"})]);
	response.usage.input_tokens = 24;
	response.usage.output_tokens = 7;
	response.usage.cache_read_input_tokens = Some(2);
	insta::assert_snapshot!(stream(response));
}

#[test]
fn managed_messages_stream_encodes_thinking_with_its_signature() {
	let response = model_response(
		"end_turn",
		vec![
			json!({"type": "thinking", "thinking": "step one", "signature": "sig-abc"}),
			json!({"type": "text", "text": "done"}),
		],
	);
	insta::assert_snapshot!(stream(response));
}

#[test]
fn managed_messages_stream_emits_unmodelled_blocks_whole() {
	let response = model_response(
		"end_turn",
		vec![json!({"type": "redacted_thinking", "data": "opaque"})],
	);
	let encoded = stream(response);
	assert!(encoded.contains("\"type\":\"redacted_thinking\""));
	assert!(!encoded.contains("content_block_delta"));
	insta::assert_snapshot!(encoded);
}

#[test]
fn managed_messages_stream_has_no_done_sentinel_and_splits_usage() {
	let mut response = model_response("end_turn", vec![json!({"type": "text", "text": "hi"})]);
	response.usage.input_tokens = 11;
	response.usage.output_tokens = 5;
	let encoded = stream(response);
	assert!(!encoded.contains("[DONE]"));
	assert!(encoded.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
	let start = encoded.lines().nth(1).unwrap();
	assert!(start.contains("\"input_tokens\":11"), "{start}");
	assert!(start.contains("\"output_tokens\":0"), "{start}");
	assert!(encoded.contains("\"output_tokens\":5"));
}

struct FakeMessagesRoundTrip {
	rounds: VecDeque<crate::llm::tool_runtime::runner::ModelRound<crate::llm::BufferedMessagesRound>>,
	requests: Vec<Value>,
}

#[async_trait::async_trait]
impl ModelRoundTrip<messages::Request, crate::llm::BufferedMessagesRound>
	for FakeMessagesRoundTrip
{
	async fn execute_round(
		&mut self,
		request: &messages::Request,
		_remaining: std::time::Duration,
	) -> Result<
		crate::llm::tool_runtime::runner::ModelRound<crate::llm::BufferedMessagesRound>,
		ToolRuntimeError,
	> {
		self
			.requests
			.push(serde_json::to_value(request).expect("serializable request"));
		Ok(self.rounds.pop_front().expect("configured fake round"))
	}
}

fn successful_messages_round(
	stop_reason: &str,
	content: Vec<Value>,
) -> crate::llm::tool_runtime::runner::ModelRound<crate::llm::BufferedMessagesRound> {
	crate::llm::tool_runtime::runner::ModelRound::Success(Box::new(buffered(model_response(
		stop_reason,
		content,
	))))
}

#[tokio::test]
async fn runner_replays_a_program_and_returns_only_the_final_message() {
	let (_, runtime) = active(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "total the orders"}],
		"tools": [
			{"type": "code_execution_20260120", "name": "code_execution"},
			{
				"name": "query_orders",
				"input_schema": {"type": "object", "properties": {"since": {"type": "string"}}},
				"allowed_callers": ["code_execution_20260120"]
			}
		]
	}));
	let state = Arc::new(ReplayProgramState {
		outcomes: Mutex::new(VecDeque::from([
			json!({
				"version": 1, "kind": "pending", "sequence": 0, "name": "query_orders",
				"arguments": {"since": "2026-01-01"}
			}),
			json!({"version": 1, "kind": "completed", "output": {"total": 1499}}),
		])),
		replay_lengths: Mutex::new(Vec::new()),
		replays: Mutex::new(Vec::new()),
	});
	let budget = RuntimeBudget::with_test_backends_and_program_sandbox(
		runtime.state.registry.as_ref(),
		HashMap::from([(
			"query_orders".to_owned(),
			Arc::new(BatchBackend) as Arc<dyn ToolBackend>,
		)]),
		Arc::new(ReplayProgramSandbox {
			state: state.clone(),
		}),
	);
	let mut round_trip = FakeMessagesRoundTrip {
		rounds: VecDeque::from([
			successful_messages_round(
				"tool_use",
				vec![tool_use(
					"toolu_p",
					"_agentgateway_programmatic_tool_calling",
					json!({"code": "program_output(tools.call('query_orders', {'since': '2026-01-01'}))"}),
				)],
			),
			successful_messages_round("end_turn", vec![json!({"type": "text", "text": "$1499"})]),
		]),
		requests: Vec::new(),
	};

	let final_response = crate::llm::tool_runtime::runner::run(*runtime, budget, &mut round_trip)
		.await
		.expect("managed program run completes");

	assert_eq!(final_response.summary.rounds, 2);
	let serialized = serde_json::to_value(&final_response.response).unwrap();
	assert_eq!(serialized["content"][0]["text"], "$1499");
	assert!(
		!serde_json::to_string(&serialized)
			.unwrap()
			.contains("_agentgateway_"),
		"client-visible body must not name a reserved function"
	);

	// The second model round sees the synthetic tool_result, not the program or the replay.
	let second = &round_trip.requests[1];
	assert_eq!(second["messages"][2]["content"][0]["type"], "tool_result");
	assert_eq!(
		second["messages"][2]["content"][0]["tool_use_id"],
		"toolu_p"
	);
	assert!(
		second["messages"][2]["content"][0]["content"]
			.as_str()
			.unwrap()
			.contains("1499")
	);
	assert_eq!(state.replay_lengths.lock().unwrap().as_slice(), &[0, 1]);
}

#[tokio::test]
async fn a_program_syntax_error_becomes_a_tool_result_with_is_error() {
	let (_, runtime) = active(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "total the orders"}],
		"tools": [
			{"type": "code_execution_20260120", "name": "code_execution"},
			{
				"name": "query_orders",
				"input_schema": {"type": "object"},
				"allowed_callers": ["code_execution_20260120"]
			}
		]
	}));
	let state = Arc::new(ReplayProgramState {
		outcomes: Mutex::new(VecDeque::from([json!({
			"version": 1, "kind": "error", "error_type": "program_error",
			"message": "SyntaxError: invalid syntax"
		})])),
		replay_lengths: Mutex::new(Vec::new()),
		replays: Mutex::new(Vec::new()),
	});
	let budget = RuntimeBudget::with_test_backends_and_program_sandbox(
		runtime.state.registry.as_ref(),
		HashMap::from([(
			"query_orders".to_owned(),
			Arc::new(BatchBackend) as Arc<dyn ToolBackend>,
		)]),
		Arc::new(ReplayProgramSandbox { state }),
	);
	let mut round_trip = FakeMessagesRoundTrip {
		rounds: VecDeque::from([
			successful_messages_round(
				"tool_use",
				vec![tool_use(
					"toolu_p",
					"_agentgateway_programmatic_tool_calling",
					json!({"code": "def ("}),
				)],
			),
			successful_messages_round("end_turn", vec![json!({"type": "text", "text": "sorry"})]),
		]),
		requests: Vec::new(),
	};

	crate::llm::tool_runtime::runner::run(*runtime, budget, &mut round_trip)
		.await
		.expect("an application error must let the model recover");

	let repair = &round_trip.requests[1]["messages"][2]["content"][0];
	assert_eq!(repair["type"], "tool_result");
	assert_eq!(repair["is_error"], true);
	assert!(
		repair["content"]
			.as_str()
			.unwrap()
			.contains("program_error")
	);
}

fn test_budget() -> RuntimeBudget {
	RuntimeBudget::new_at(
		&managed_registry(),
		crate::test_helpers::policy_client(),
		RuntimeDeadline::new(Duration::from_secs(30)),
	)
	.expect("valid test budget")
}

#[test]
fn append_round_history_writes_assistant_then_tool_results_in_call_order() {
	let mut runtime = direct_runtime();
	let round = buffered(model_response(
		"tool_use",
		vec![
			json!({"type": "text", "text": "checking"}),
			tool_use("toolu_1", "query_orders", json!({"since": "a"})),
			tool_use("toolu_2", "query_orders", json!({"since": "b"})),
		],
	));
	let outputs = vec![
		runtime
			.tool_output_item(
				&Strng::from("toolu_1"),
				ToolExecutionResult::function(json!({"rows": 1})),
				4096,
			)
			.unwrap(),
		runtime
			.tool_output_item(
				&Strng::from("toolu_2"),
				ToolExecutionResult::ApplicationError(ToolApplicationError::new(
					"tool_error",
					"backend refused",
					false,
					"",
					"",
				)),
				4096,
			)
			.unwrap(),
	];
	runtime.append_round_history(round, outputs);

	let serialized = serde_json::to_value(&runtime.canonical_request).unwrap();
	let messages = serialized["messages"].as_array().unwrap();
	assert_eq!(messages.len(), 3);
	assert_eq!(messages[1]["role"], "assistant");
	assert_eq!(messages[1]["content"][1]["type"], "tool_use");
	assert_eq!(messages[2]["role"], "user");
	assert_eq!(messages[2]["content"][0]["tool_use_id"], "toolu_1");
	assert_eq!(messages[2]["content"][0].get("is_error"), None);
	assert_eq!(messages[2]["content"][1]["tool_use_id"], "toolu_2");
	assert_eq!(messages[2]["content"][1]["is_error"], true);
	assert_eq!(serialized["tool_choice"], json!({"type": "auto"}));
}

#[test]
fn thinking_signature_and_cache_control_survive_an_appended_round() {
	let mut runtime = direct_runtime();
	let thinking = json!({
		"type": "thinking",
		"thinking": "step one",
		"signature": "sig-abcdef0123456789"
	});
	let cached = json!({
		"type": "text",
		"text": "prefix",
		"cache_control": {"type": "ephemeral", "ttl": "5m"}
	});
	let round = buffered(model_response(
		"tool_use",
		vec![
			thinking.clone(),
			cached.clone(),
			tool_use("toolu_1", "query_orders", json!({"since": "a"})),
		],
	));
	let outputs = vec![
		runtime
			.tool_output_item(
				&Strng::from("toolu_1"),
				ToolExecutionResult::function(json!({"rows": 1})),
				4096,
			)
			.unwrap(),
	];
	runtime.append_round_history(round, outputs);

	let serialized = serde_json::to_value(&runtime.canonical_request).unwrap();
	let content = serialized["messages"][1]["content"].as_array().unwrap();
	assert_eq!(content[0], thinking);
	assert_eq!(content[1], cached);
}

#[test]
fn accumulate_usage_sums_rounds_and_finalize_reports_the_aggregate() {
	let mut runtime = direct_runtime();
	let mut first = model_response("tool_use", vec![]);
	first.usage.input_tokens = 10;
	first.usage.output_tokens = 3;
	first.usage.cache_read_input_tokens = Some(2);
	let mut last = model_response("end_turn", vec![json!({"type": "text", "text": "22 C"})]);
	last.usage.input_tokens = 14;
	last.usage.output_tokens = 4;

	runtime.accumulate_usage(&buffered(first));
	runtime.accumulate_usage(&buffered(last.clone()));
	let budget = test_budget();
	let final_response = runtime.finalize(buffered(last), &budget);

	assert_eq!(final_response.response.usage.input_tokens, 24);
	assert_eq!(final_response.response.usage.output_tokens, 7);
	assert_eq!(
		final_response.response.usage.cache_read_input_tokens,
		Some(2)
	);
	assert_eq!(
		final_response.response.stop_reason.as_deref(),
		Some("end_turn")
	);
	let serialized = serde_json::to_value(&final_response.response).unwrap();
	assert_eq!(serialized.get("tools"), None);
	assert_eq!(serialized.get("container"), None);
}

#[test]
fn collect_reads_object_input_and_uses_the_block_id_as_the_call_id() {
	let runtime = direct_runtime();
	let response = model_response(
		"tool_use",
		vec![
			json!({"type": "text", "text": "checking"}),
			tool_use("toolu_1", "query_orders", json!({"since": "2026-01-01"})),
		],
	);
	let crate::llm::tool_runtime::CollectedToolCalls::Direct(calls) =
		runtime.collect_model_calls(&response).unwrap()
	else {
		panic!("direct calls expected");
	};
	assert_eq!(calls.len(), 1);
	assert_eq!(calls[0].call_id.as_str(), "toolu_1");
	assert_eq!(calls[0].public_name.as_str(), "query_orders");
	assert_eq!(calls[0].arguments, json!({"since": "2026-01-01"}));
}

#[test]
fn a_round_without_tool_use_blocks_is_terminal() {
	let runtime = direct_runtime();
	let response = model_response("end_turn", vec![json!({"type": "text", "text": "done"})]);
	let crate::llm::tool_runtime::CollectedToolCalls::Direct(calls) =
		runtime.collect_model_calls(&response).unwrap()
	else {
		panic!("direct calls expected");
	};
	assert!(calls.is_empty());
}

#[test]
fn collect_rejects_empty_duplicate_and_undeclared_tool_use_blocks() {
	let runtime = direct_runtime();
	for content in [
		vec![tool_use("", "query_orders", json!({"since": "x"}))],
		vec![
			tool_use("toolu_1", "query_orders", json!({"since": "x"})),
			tool_use("toolu_1", "query_orders", json!({"since": "y"})),
		],
		vec![tool_use("toolu_1", "not_declared", json!({}))],
		vec![tool_use("toolu_1", "query_orders", json!({"since": 5}))],
		vec![tool_use("toolu_1", "query_orders", json!("since"))],
	] {
		assert!(
			runtime
				.collect_model_calls(&model_response("tool_use", content))
				.is_err()
		);
	}
}

#[test]
fn collect_rejects_a_tool_use_truncated_by_max_tokens() {
	let runtime = direct_runtime();
	let response = model_response(
		"max_tokens",
		vec![tool_use("toolu_1", "query_orders", json!({"since": "x"}))],
	);
	let error = runtime.collect_model_calls(&response).unwrap_err();
	assert!(error.to_string().contains("max_tokens"), "{error}");
}

#[test]
fn collect_extracts_the_program_and_rejects_it_mixed_with_direct_calls() {
	let runtime = active(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "orders"}],
		"tools": [
			{"type": "code_execution_20260120", "name": "code_execution"},
			{
				"name": "query_orders",
				"input_schema": {"type": "object"},
				"allowed_callers": ["code_execution_20260120"]
			}
		]
	}))
	.1;

	let program = tool_use(
		"toolu_p",
		"_agentgateway_programmatic_tool_calling",
		json!({"code": "program_output(1)"}),
	);
	let crate::llm::tool_runtime::CollectedToolCalls::Programmatic { call_id, code } = runtime
		.collect_model_calls(&model_response("tool_use", vec![program.clone()]))
		.unwrap()
	else {
		panic!("programmatic call expected");
	};
	assert_eq!(call_id.as_str(), "toolu_p");
	assert_eq!(code, "program_output(1)");

	let mixed = model_response(
		"tool_use",
		vec![
			program,
			tool_use(
				"toolu_1",
				"_agentgateway_code_interpreter",
				json!({"code": "1"}),
			),
		],
	);
	assert!(runtime.collect_model_calls(&mixed).is_err());
}

#[test]
fn recursive_input_schema_ref_cycle_is_rejected() {
	let mut request = request(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "orders"}],
		"tools": [{
			"name": "query_orders",
			"input_schema": {
				"type": "object",
				"properties": {"self": {"$ref": "#"}}
			}
		}]
	}));
	let error = match prepare(&mut request, Some(&managed_registry())) {
		Ok(_) => panic!("recursive schema must be rejected"),
		Err(error) => error,
	};
	assert!(error.to_string().contains("input_schema"), "{error}");
}

#[test]
fn indirect_input_schema_ref_cycle_is_rejected() {
	let mut request = request(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "orders"}],
		"tools": [{
			"name": "query_orders",
			"input_schema": {
				"type": "object",
				"$defs": {
					"a": {"$ref": "#/$defs/b"},
					"b": {"$ref": "#/$defs/a"}
				},
				"properties": {"cycle": {"$ref": "#/$defs/a"}}
			}
		}]
	}));
	let error = match prepare(&mut request, Some(&managed_registry())) {
		Ok(_) => panic!("indirect recursive schema must be rejected"),
		Err(error) => error,
	};
	assert!(error.to_string().contains("input_schema"), "{error}");
}

#[test]
fn unresolved_input_schema_local_ref_is_rejected() {
	let mut request = request(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "orders"}],
		"tools": [{
			"name": "query_orders",
			"input_schema": {
				"type": "object",
				"properties": {"missing": {"$ref": "#/$defs/missing"}}
			}
		}]
	}));
	let error = match prepare(&mut request, Some(&managed_registry())) {
		Ok(_) => panic!("unresolved local ref schema must be rejected"),
		Err(error) => error,
	};
	assert!(error.to_string().contains("input_schema"), "{error}");
}

fn rejection(value: Value) -> String {
	let mut request = request(value);
	match prepare(&mut request, Some(&managed_registry())) {
		Ok(_) => panic!("declaration must be rejected"),
		Err(error) => error.to_string(),
	}
}

#[test]
fn active_runtime_rejects_unsupported_declarations_and_fields() {
	let base = |tools: Value, extra: Value| {
		let mut value = json!({
			"model": "smart",
			"max_tokens": 16,
			"messages": [{"role": "user", "content": "x"}],
			"tools": tools
		});
		if let (Some(value), Some(extra)) = (value.as_object_mut(), extra.as_object()) {
			for (key, field) in extra {
				value.insert(key.clone(), field.clone());
			}
		}
		value
	};
	let managed = json!({"name": "query_orders", "input_schema": {"type": "object"}});
	let code = json!({"type": "code_execution_20260120", "name": "code_execution"});

	// An unregistered custom tool: active mode is fail-closed.
	assert!(
		rejection(base(
			json!([managed, {"name": "unregistered", "input_schema": {"type": "object"}}]),
			json!({})
		))
		.contains("is not registered")
	);
	// A reserved name.
	assert!(
		rejection(base(
			json!([managed, {"name": "_agentgateway_x", "input_schema": {"type": "object"}}]),
			json!({})
		))
		.contains("reserved")
	);
	// Server toolsets we cannot serve.
	for kind in [
		"mcp_toolset",
		"computer_toolset_20260120",
		"browser_toolset_20260120",
	] {
		assert!(
			rejection(base(
				json!([managed, {"type": kind, "name": kind}]),
				json!({})
			))
			.contains("is not supported"),
			"{kind}"
		);
	}
	// Request fields this design holds no state for.
	assert!(
		rejection(base(json!([managed]), json!({"container": "container_1"}))).contains("container")
	);
	assert!(
		rejection(base(
			json!([managed]),
			json!({"mcp_servers": [{"type": "url", "url": "https://x", "name": "x"}]})
		))
		.contains("mcp_servers")
	);
	// A duplicate declaration.
	assert!(rejection(base(json!([managed.clone(), managed]), json!({}))).contains("duplicate"));
	// A code execution declaration with no configured E2B backend.
	let mut request = request(base(json!([code, managed]), json!({})));
	let error = match prepare(
		&mut request,
		Some(&registry(vec![tool("query_orders", None)])),
	) {
		Ok(_) => panic!("code execution requires a backend"),
		Err(error) => error,
	};
	assert!(error.to_string().contains("code execution"), "{error}");
}

#[test]
fn tool_choice_cannot_force_a_programmatic_only_tool() {
	let message = rejection(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "x"}],
		"tool_choice": {"type": "tool", "name": "query_orders"},
		"tools": [
			{"type": "code_execution_20260120", "name": "code_execution"},
			{
				"name": "query_orders",
				"input_schema": {"type": "object"},
				"allowed_callers": ["code_execution_20260120"]
			}
		]
	}));
	assert!(message.contains("query_orders"), "{message}");
}

#[test]
fn disable_parallel_tool_use_is_rejected_with_programmatic_tool_calling() {
	let message = rejection(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "x"}],
		"tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
		"tools": [
			{"type": "code_execution_20260120", "name": "code_execution"},
			{
				"name": "query_orders",
				"input_schema": {"type": "object"},
				"allowed_callers": ["code_execution_20260120"]
			}
		]
	}));
	assert!(message.contains("disable_parallel_tool_use"), "{message}");
}

#[test]
fn a_programmatic_caller_without_code_execution_is_rejected() {
	let message = rejection(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "x"}],
		"tools": [{
			"name": "query_orders",
			"input_schema": {"type": "object"},
			"allowed_callers": ["code_execution_20260120"]
		}]
	}));
	assert!(message.contains("code execution"), "{message}");
}

#[test]
fn the_web_search_programmatic_default_is_rejected_without_code_execution() {
	let message = rejection(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "x"}],
		"tools": [
			{"name": "query_orders", "input_schema": {"type": "object"}},
			{"type": "web_search_20260209", "name": "web_search"}
		]
	}));
	assert!(message.contains("allowed_callers"), "{message}");
	assert!(message.contains("direct"), "{message}");
}

#[test]
fn messages_request_extension_reads_replaces_and_preserves_unknown_fields() {
	let mut request = request(json!({
			"model": "smart",
			"max_tokens": 16,
			"messages": [{"role": "user", "content": "hi"}],
			"future_request_field": {"nested": [1, true, null]}
	}));

	assert_eq!(
		request.rest_field("future_request_field"),
		Some(&json!({"nested": [1, true, null]}))
	);
	request.replace_rest_field("tool_choice", json!({"type": "auto"}));

	let serialized = serde_json::to_value(request).unwrap();
	assert_eq!(serialized["tool_choice"], json!({"type": "auto"}));
	assert_eq!(
		serialized["future_request_field"],
		json!({"nested": [1, true, null]})
	);
	assert_eq!(serialized["messages"][0]["content"], json!("hi"));
}

#[test]
fn messages_request_extension_creates_rest_when_absent() {
	let mut request = request(json!({
			"model": "smart",
			"max_tokens": 16,
			"messages": []
	}));

	assert_eq!(request.rest_field("tools"), None);
	request.replace_rest_field("tools", json!([{"name": "x"}]));
	assert_eq!(request.rest_field("tools"), Some(&json!([{"name": "x"}])));

	request.remove_rest_field("tools");
	assert_eq!(request.rest_field("tools"), None);
	assert_eq!(serde_json::to_value(request).unwrap().get("tools"), None);
}

#[test]
fn messages_usage_aggregation_saturates_and_sums_cache_fields() {
	let mut aggregate = messages::Usage {
		input_tokens: u64::MAX - 1,
		output_tokens: 10,
		cache_creation_input_tokens: Some(3),
		cache_read_input_tokens: None,
		service_tier: Some("standard".to_owned()),
		rest: json!({
			"output_tokens_details": {"reasoning_tokens": 2},
			"cache_creation": {"ephemeral_5m_input_tokens": 1},
			"diagnostic": "first"
		}),
	};
	super::aggregate_messages_usage(
		&mut aggregate,
		&messages::Usage {
			input_tokens: 5,
			output_tokens: 7,
			cache_creation_input_tokens: Some(4),
			cache_read_input_tokens: Some(9),
			service_tier: Some("priority".to_owned()),
			rest: json!({
				"output_tokens_details": {"reasoning_tokens": 8},
				"cache_creation": {"ephemeral_5m_input_tokens": 3},
				"diagnostic": "second"
			}),
		},
	);

	assert_eq!(aggregate.input_tokens, u64::MAX);
	assert_eq!(aggregate.output_tokens, 17);
	assert_eq!(aggregate.cache_creation_input_tokens, Some(7));
	assert_eq!(aggregate.cache_read_input_tokens, Some(9));
	assert_eq!(aggregate.service_tier, None);
	assert!(aggregate.rest.get("output_tokens_details").is_none());
	assert!(aggregate.rest.get("cache_creation").is_none());
}

#[test]
fn no_managed_declaration_leaves_the_runtime_inactive() {
	let mut request = request(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{"type": "mcp_toolset", "mcp_server_name": "s"}]
	}));
	assert!(matches!(
		prepare(&mut request, Some(&managed_registry())).unwrap(),
		MessagesActivation::Inactive
	));
	// An inactive request is forwarded byte-for-byte, including declarations we would reject.
	assert_eq!(declared(&request), Vec::<String>::new());
	assert_eq!(
		request.rest_field("tools").unwrap()[0]["type"],
		"mcp_toolset"
	);
}

#[test]
fn programmatic_tool_is_withheld_and_the_program_runtime_is_declared() {
	let (request, runtime) = active(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "total the orders"}],
		"tools": [
			{"type": "code_execution_20260120", "name": "code_execution"},
			{
				"name": "query_orders",
				"description": "Query orders.",
				"input_schema": {"type": "object", "properties": {"since": {"type": "string"}}},
				"allowed_callers": ["code_execution_20260120"]
			}
		]
	}));

	let names = declared(&request);
	assert!(names.contains(&"_agentgateway_programmatic_tool_calling".to_owned()));
	assert!(names.contains(&"_agentgateway_code_interpreter".to_owned()));
	assert!(!names.contains(&"query_orders".to_owned()));
	assert_eq!(
		runtime.canonical_request.rest_field("tools"),
		request.rest_field("tools")
	);

	let program = request
		.rest_field("tools")
		.and_then(Value::as_array)
		.unwrap()
		.iter()
		.find(|tool| tool["name"] == "_agentgateway_programmatic_tool_calling")
		.unwrap();
	assert!(
		program["description"]
			.as_str()
			.unwrap()
			.contains("query_orders")
	);
	assert_eq!(program["input_schema"]["required"], json!(["code"]));
	assert_eq!(program.get("type"), None);
	assert_eq!(program.get("allowed_callers"), None);
}

#[test]
fn direct_managed_function_keeps_its_client_visible_name_and_schema() {
	let (request, _) = active(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "orders"}],
		"tools": [{
			"name": "query_orders",
			"description": "Query orders.",
			"input_schema": {"type": "object", "properties": {"since": {"type": "string"}}}
		}]
	}));

	let tools = request
		.rest_field("tools")
		.and_then(Value::as_array)
		.unwrap();
	assert_eq!(tools.len(), 1);
	assert_eq!(tools[0]["name"], "query_orders");
	assert_eq!(
		tools[0]["input_schema"]["properties"]["since"]["type"],
		"string"
	);
}

#[test]
fn code_execution_20260521_normalizes_to_20260120() {
	let (request, _) = active(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "compute"}],
		"tools": [
			{"type": "code_execution_20260521", "name": "code_execution"},
			{
				"name": "query_orders",
				"input_schema": {"type": "object"},
				"allowed_callers": ["code_execution_20260521"]
			}
		]
	}));
	let names = declared(&request);
	assert!(names.contains(&"_agentgateway_programmatic_tool_calling".to_owned()));
	assert!(!names.contains(&"query_orders".to_owned()));
}

#[test]
fn pre_ptc_code_execution_versions_declare_only_the_direct_interpreter() {
	for version in ["code_execution_20250522", "code_execution_20250825"] {
		let (request, _) = active(json!({
			"model": "smart",
			"max_tokens": 16,
			"messages": [{"role": "user", "content": "compute"}],
			"tools": [{"type": version, "name": "code_execution"}]
		}));
		let names = declared(&request);
		assert_eq!(names, vec!["_agentgateway_code_interpreter".to_owned()]);
	}
}

#[test]
fn messages_web_search_retains_all_valid_trusted_options() {
	let (_, runtime) = active(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "search"}],
		"tools": [{
			"type": "web_search_20250305",
			"name": "web_search",
			"allowed_domains": ["docs.example.com"],
			"blocked_domains": ["ads.example.com"],
			"user_location": {
				"type": "approximate",
				"country": "US",
				"region": "California",
				"city": "San Francisco",
				"timezone": "America/Los_Angeles"
			},
			"max_uses": 3
		}]
	}));

	assert_eq!(
		runtime
			.state
			.registry
			.trusted_options("_agentgateway_web_search"),
		Some(&json!({
			"allowed_domains": ["docs.example.com"],
			"blocked_domains": ["ads.example.com"],
			"user_location": {
				"type": "approximate",
				"country": "US",
				"region": "California",
				"city": "San Francisco",
				"timezone": "America/Los_Angeles"
			},
			"max_uses": 3
		}))
	);
}

#[test]
fn messages_web_search_rejects_malformed_trusted_options() {
	for (label, options) in [
		(
			"allowed_domains_string",
			json!({"allowed_domains": "example.com"}),
		),
		(
			"blocked_domains_string",
			json!({"blocked_domains": "example.com"}),
		),
		("max_uses_fractional", json!({"max_uses": 1.5})),
		("max_uses_negative", json!({"max_uses": -1})),
		(
			"location_type",
			json!({"user_location": {"type": "precise"}}),
		),
		(
			"location_member",
			json!({"user_location": {"type": "approximate", "city": 3}}),
		),
		("allowed_domains_null", json!({"allowed_domains": null})),
		("blocked_domains_null", json!({"blocked_domains": null})),
		("user_location_null", json!({"user_location": null})),
		("max_uses_null", json!({"max_uses": null})),
	] {
		let mut declaration = json!({
			"type": "web_search_20250305",
			"name": "web_search"
		});
		declaration
			.as_object_mut()
			.unwrap()
			.extend(options.as_object().unwrap().clone());
		let message = rejection(json!({
			"model": "smart",
			"max_tokens": 16,
			"messages": [{"role": "user", "content": "search"}],
			"tools": [declaration]
		}));
		assert!(message.contains("invalid web search"), "{label}: {message}");
	}
}

#[test]
fn duplicate_builtin_declarations_are_rejected_by_identity() {
	for (label, tools) in [
		(
			"same_code_execution_version",
			json!([
				{"type": "code_execution_20260120", "name": "code_execution"},
				{"type": "code_execution_20260120", "name": "code_execution"}
			]),
		),
		(
			"mixed_code_execution",
			json!([
				{"type": "code_execution_20260120", "name": "code_execution"},
				{"type": "code_execution_20250825", "name": "code_execution"}
			]),
		),
		(
			"mixed_web_search",
			json!([
				{"type": "web_search_20250305", "name": "web_search"},
				{"type": "web_search_20260209", "name": "web_search"}
			]),
		),
	] {
		let message = rejection(json!({
			"model": "smart",
			"max_tokens": 16,
			"messages": [{"role": "user", "content": "x"}],
			"tools": tools
		}));
		assert!(message.contains("duplicate"), "{label}: {message}");
	}
}

#[test]
fn builtin_declarations_require_their_exact_names() {
	for (label, declaration, expected) in [
		(
			"code_missing",
			json!({"type": "code_execution_20260120"}),
			"code_execution",
		),
		(
			"code_wrong",
			json!({"type": "code_execution_20260120", "name": "python"}),
			"code_execution",
		),
		(
			"search_missing",
			json!({"type": "web_search_20250305"}),
			"web_search",
		),
		(
			"search_wrong",
			json!({"type": "web_search_20250305", "name": "search"}),
			"web_search",
		),
	] {
		let message = rejection(json!({
			"model": "smart",
			"max_tokens": 16,
			"messages": [{"role": "user", "content": "x"}],
			"tools": [declaration]
		}));
		assert!(message.contains(expected), "{label}: {message}");
	}
}

#[test]
fn web_search_20260209_defaults_to_the_programmatic_caller() {
	let (request, _) = active(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "search"}],
		"tools": [
			{"type": "code_execution_20260120", "name": "code_execution"},
			{
				"type": "web_search_20260209",
				"name": "web_search",
				"allowed_domains": ["example.com"],
				"max_uses": 3
			}
		]
	}));
	let names = declared(&request);
	assert!(names.contains(&"_agentgateway_programmatic_tool_calling".to_owned()));
	assert!(!names.contains(&"_agentgateway_web_search".to_owned()));
}

#[test]
fn earlier_web_search_versions_default_to_the_direct_caller() {
	let (request, _) = active(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "search"}],
		"tools": [{"type": "web_search_20250305", "name": "web_search"}]
	}));
	let tools = request
		.rest_field("tools")
		.and_then(Value::as_array)
		.unwrap();
	assert_eq!(tools.len(), 1);
	assert_eq!(tools[0]["name"], "_agentgateway_web_search");
	assert_eq!(tools[0]["input_schema"]["required"], json!(["query"]));
	assert_eq!(tools[0].get("allowed_domains"), None);
}

#[test]
fn streaming_requests_run_internal_rounds_non_streaming() {
	let (request, runtime) = active(json!({
		"model": "smart",
		"max_tokens": 16,
		"stream": true,
		"messages": [{"role": "user", "content": "orders"}],
		"tools": [{"name": "query_orders", "input_schema": {"type": "object"}}]
	}));
	assert!(runtime.client_streaming);
	assert_eq!(request.stream, Some(false));
}

#[test]
fn disable_parallel_tool_use_inverts_the_parallel_flag() {
	let (_, runtime) = active(json!({
		"model": "smart",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": "orders"}],
		"tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
		"tools": [{"name": "query_orders", "input_schema": {"type": "object"}}]
	}));
	assert!(!runtime.state.parallel);
}
