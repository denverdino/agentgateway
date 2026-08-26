use std::collections::{BTreeSet, HashMap, VecDeque};
use std::future::pending;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use http_body_util::BodyExt;
use prometheus_client::encoding::prometheus_protobuf;
use prometheus_client::encoding::prometheus_protobuf::prometheus_data_model::{
	Metric, MetricFamily, MetricType,
};
use prometheus_client::registry::Registry as PrometheusRegistry;
use secrecy::SecretString;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::backend::execute_sequentially;
use super::{
	Activation, BuiltinTool, ManagedToolConfig, PreparedToolRuntime, ResponsesRequestExt,
	RuntimeBudget, RuntimeDeadline, RuntimeLimits, SandboxOperation, SandboxOperationLabels,
	SandboxOperationOutcome, ToolBackendConfig, ToolExecutionOutcome, ToolRegistry,
	ToolRuntimeConfig, ToolRuntimeOutcome, aggregate_usage, encode_streaming_response, execute_batch,
	finalize_managed_response, prepare,
};
use super::{
	E2bSandboxBackend, HttpToolBackend, ManagedToolCall, RemoteMcpBackend, RemoteMcpServer,
	ToolApplicationError, ToolBackend, ToolBatchExecution, ToolBatchInfrastructureError,
	ToolBatchMetadata, ToolExecutionContext, ToolExecutionResult, ToolInfrastructureError,
	parse_arguments,
};

mod config_tests;
mod proxy;
use crate::HistogramMode;
use crate::llm::types::responses;
use crate::proxy::httpproxy::PolicyClient;
use crate::telemetry::metrics::{Metrics, OutboundCallSubtype};

fn registry(tools: Vec<ManagedToolConfig>) -> Arc<ToolRegistry> {
	Arc::new(
		ToolRegistry::compile(ToolRuntimeConfig {
			limits: RuntimeLimits {
				total_timeout: Duration::from_secs(120),
				max_rounds: 8,
				max_tool_calls: 16,
				max_parallel_tool_calls: 4,
				max_arguments_bytes: 65_536,
				max_output_bytes: 1_048_576,
			},
			tools,
		})
		.expect("valid runtime configuration"),
	)
}

fn tool(name: &str, builtin: Option<BuiltinTool>) -> ManagedToolConfig {
	let backend = match builtin {
		Some(BuiltinTool::CodeInterpreter) => ToolBackendConfig::E2b {
			api_url: "http://127.0.0.1:18080".parse().unwrap(),
			domain: "sandbox.example.com".into(),
			timeout: Duration::from_secs(5),
			api_key: SecretString::from("operator-secret"),
		},
		Some(BuiltinTool::WebSearch) | None => ToolBackendConfig::Http {
			url: "http://127.0.0.1:18080".parse().unwrap(),
			timeout: Duration::from_secs(5),
			bearer_token: Some(SecretString::from("operator-secret")),
		},
	};
	ManagedToolConfig {
		name: name.to_owned(),
		builtin,
		backend,
	}
}

#[test]
fn registry_compile_rejects_invalid_runtime_semantics() {
	let valid_limits = RuntimeLimits {
		total_timeout: Duration::from_secs(30),
		max_rounds: 2,
		max_tool_calls: 4,
		max_parallel_tool_calls: 2,
		max_arguments_bytes: 1024,
		max_output_bytes: 4096,
	};
	let valid_tool = || tool("managed", None);
	let mut cases = Vec::new();

	let mut limits = valid_limits.clone();
	limits.max_rounds = 0;
	cases.push((
		"zero max rounds",
		ToolRuntimeConfig {
			limits,
			tools: vec![valid_tool()],
		},
	));
	cases.push((
		"empty tools",
		ToolRuntimeConfig {
			limits: valid_limits.clone(),
			tools: vec![],
		},
	));
	cases.push((
		"zero backend timeout",
		ToolRuntimeConfig {
			limits: valid_limits.clone(),
			tools: vec![ManagedToolConfig {
				name: "managed".into(),
				builtin: None,
				backend: ToolBackendConfig::Http {
					url: "https://tools.example.com".parse().unwrap(),
					timeout: Duration::ZERO,
					bearer_token: None,
				},
			}],
		},
	));
	cases.push((
		"insecure public endpoint",
		ToolRuntimeConfig {
			limits: valid_limits.clone(),
			tools: vec![ManagedToolConfig {
				name: "managed".into(),
				builtin: None,
				backend: ToolBackendConfig::Http {
					url: "http://tools.example.com".parse().unwrap(),
					timeout: Duration::from_secs(1),
					bearer_token: None,
				},
			}],
		},
	));
	cases.push((
		"e2b url is not an origin",
		ToolRuntimeConfig {
			limits: valid_limits.clone(),
			tools: vec![ManagedToolConfig {
				name: "python".into(),
				builtin: Some(BuiltinTool::CodeInterpreter),
				backend: ToolBackendConfig::E2b {
					api_url: "https://api.e2b.example.com/path?query=1".parse().unwrap(),
					domain: "sandbox.example.com".into(),
					timeout: Duration::from_secs(1),
					api_key: SecretString::from("secret"),
				},
			}],
		},
	));
	cases.push((
		"invalid e2b domain",
		ToolRuntimeConfig {
			limits: valid_limits,
			tools: vec![ManagedToolConfig {
				name: "python".into(),
				builtin: Some(BuiltinTool::CodeInterpreter),
				backend: ToolBackendConfig::E2b {
					api_url: "https://api.e2b.example.com".parse().unwrap(),
					domain: "bad/domain".into(),
					timeout: Duration::from_secs(1),
					api_key: SecretString::from("secret"),
				},
			}],
		},
	));

	for (name, config) in cases {
		assert!(ToolRegistry::compile(config).is_err(), "{name}");
	}
}

#[test]
fn tool_runtime_shared_helpers_classify_transport_and_truncate_utf8() {
	assert_eq!(
		super::transport::classify_status(http::StatusCode::UNAUTHORIZED, false),
		ToolInfrastructureError::Authentication
	);
	assert_eq!(
		super::transport::classify_status(http::StatusCode::GATEWAY_TIMEOUT, true),
		ToolInfrastructureError::Timeout
	);
	assert_eq!(
		super::transport::classify_status(http::StatusCode::BAD_GATEWAY, false),
		ToolInfrastructureError::Backend
	);
	assert_eq!(
		super::transport::classify_proxy_error(crate::proxy::ProxyError::RequestTimeout),
		ToolInfrastructureError::Timeout
	);
	assert_eq!(
		super::transport::classify_proxy_error(crate::proxy::ProxyError::NoValidBackends),
		ToolInfrastructureError::Backend
	);
	assert_eq!(super::truncate_utf8_bytes("a你b", 2), "a");
	assert_eq!(super::truncate_utf8_bytes("a你b", 4), "a你");
}

#[tokio::test]
async fn e2b_batch_reuses_one_sandbox_and_cleans_each_context_in_order() {
	let server = MockServer::start().await;
	Mock::given(method("POST"))
		.and(path("/sandboxes"))
		.respond_with(ResponseTemplate::new(201).set_body_json(json!({
			"clientID": "client",
			"envdVersion": "0.1.0",
			"sandboxID": "sandbox_1",
			"templateID": "code-interpreter-v1",
			"domain": "sandbox.example.com",
			"envdAccessToken": "envd-token",
			"trafficAccessToken": "traffic-token"
		})))
		.mount(&server)
		.await;
	Mock::given(method("POST"))
		.and(path("/contexts"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({
			"id": "context_1", "language": "python", "cwd": "/home/user"
		})))
		.mount(&server)
		.await;
	Mock::given(method("POST"))
		.and(path("/execute"))
		.respond_with(ResponseTemplate::new(200).set_body_string(
			"{\"type\":\"stdout\",\"text\":\"ok\\n\",\"timestamp\":1}\n{\"type\":\"number_of_executions\",\"execution_count\":1}\n",
		))
		.mount(&server)
		.await;
	Mock::given(method("DELETE"))
		.and(path("/contexts/context_1"))
		.respond_with(ResponseTemplate::new(204))
		.mount(&server)
		.await;
	Mock::given(method("DELETE"))
		.and(path("/sandboxes/sandbox_1"))
		.respond_with(ResponseTemplate::new(204))
		.mount(&server)
		.await;

	let backend = E2bSandboxBackend::new(
		policy_client(),
		server.uri().parse().unwrap(),
		"sandbox.example.com".into(),
		Duration::from_secs(2),
		SecretString::from("operator-key"),
		4096,
	)
	.unwrap();
	let batch = backend
		.execute_batch(
			vec![
				managed_call(
					super::CODE_INTERPRETER_FUNCTION,
					"call_1",
					json!({"code": "print(1)"}),
				),
				managed_call(
					super::CODE_INTERPRETER_FUNCTION,
					"call_2",
					json!({"code": "print(2)"}),
				),
			],
			ToolExecutionContext::default(),
		)
		.await
		.unwrap();
	assert_eq!(
		batch.metadata.sandbox_create.unwrap().outcome,
		super::SandboxCleanupOutcome::Success
	);
	assert_eq!(
		batch.metadata.sandbox_execute.unwrap().outcome,
		super::SandboxCleanupOutcome::Success
	);
	assert_eq!(
		batch.metadata.sandbox_terminate.unwrap().outcome,
		super::SandboxCleanupOutcome::Success
	);
	assert_eq!(
		batch.metadata.sandbox_cleanup,
		Some(super::SandboxCleanupOutcome::Success)
	);
	assert!(batch.metadata.sandbox_cleanup_duration.is_some());
	assert_eq!(batch.results.len(), 2);
	for result in batch.results {
		assert_eq!(
			result.into_model_output(4096).unwrap(),
			json!({
				"ok": true, "exit_code": 0, "stdout": "ok\n", "stderr": "",
				"timed_out": false, "truncated": false, "artifacts": []
			})
		);
	}

	let requests = server.received_requests().await.unwrap();
	assert_eq!(
		requests
			.iter()
			.filter(|r| r.url.path() == "/sandboxes")
			.count(),
		1
	);
	assert_eq!(
		requests
			.iter()
			.filter(|r| r.url.path() == "/contexts")
			.count(),
		2
	);
	assert_eq!(
		requests
			.iter()
			.filter(|r| r.url.path() == "/execute")
			.count(),
		2
	);
	assert_eq!(
		requests
			.iter()
			.filter(|r| r.url.path() == "/contexts/context_1")
			.count(),
		2
	);
	assert_eq!(
		requests
			.iter()
			.filter(|r| r.url.path() == "/sandboxes/sandbox_1")
			.count(),
		1
	);
	let create = requests
		.iter()
		.find(|r| r.url.path() == "/sandboxes")
		.unwrap();
	assert_eq!(create.headers["x-api-key"], "operator-key");
	assert_eq!(
		serde_json::from_slice::<Value>(&create.body).unwrap(),
		json!({"templateID": "code-interpreter-v1", "timeout": 2, "secure": true})
	);
	let execute = requests
		.iter()
		.find(|r| r.url.path() == "/execute")
		.unwrap();
	assert_eq!(execute.headers["x-access-token"], "envd-token");
	assert_eq!(execute.headers["e2b-traffic-access-token"], "traffic-token");
	let execute_body: Value = serde_json::from_slice(&execute.body).unwrap();
	assert_eq!(execute_body["context_id"], "context_1");
	assert_eq!(execute_body["language"], Value::Null);
	assert_eq!(execute_body["env_vars"], Value::Null);
}

async fn wait_for_e2b_request(server: &MockServer, request_path: &str) {
	tokio::time::timeout(Duration::from_secs(2), async {
		loop {
			if server
				.received_requests()
				.await
				.unwrap()
				.iter()
				.any(|request| request.url.path() == request_path)
			{
				return;
			}
			tokio::task::yield_now().await;
		}
	})
	.await
	.unwrap_or_else(|_| panic!("timed out waiting for {request_path}"));
}

#[tokio::test]
async fn e2b_cancelled_batch_still_terminates_sandbox_once() {
	let server = MockServer::start().await;
	Mock::given(method("POST"))
		.and(path("/sandboxes"))
		.respond_with(ResponseTemplate::new(201).set_body_json(json!({
			"sandboxID": "sandbox_cancelled",
			"domain": "sandbox.example.com"
		})))
		.mount(&server)
		.await;
	Mock::given(method("POST"))
		.and(path("/contexts"))
		.respond_with(
			ResponseTemplate::new(200)
				.set_delay(Duration::from_secs(30))
				.set_body_json(json!({"id": "never"})),
		)
		.mount(&server)
		.await;
	Mock::given(method("DELETE"))
		.and(path("/sandboxes/sandbox_cancelled"))
		.respond_with(ResponseTemplate::new(204))
		.mount(&server)
		.await;

	let backend = E2bSandboxBackend::new(
		policy_client(),
		server.uri().parse().unwrap(),
		"sandbox.example.com".into(),
		Duration::from_secs(24 * 60 * 60),
		SecretString::from("operator-key"),
		4096,
	)
	.unwrap();
	let task = tokio::spawn(async move {
		backend
			.execute_batch(
				vec![managed_call(
					super::CODE_INTERPRETER_FUNCTION,
					"cancelled",
					json!({"code": "pass"}),
				)],
				ToolExecutionContext {
					deadline: Some(Instant::now() + Duration::from_secs(2)),
					..Default::default()
				},
			)
			.await
	});
	wait_for_e2b_request(&server, "/contexts").await;
	task.abort();
	let _ = task.await;
	wait_for_e2b_request(&server, "/sandboxes/sandbox_cancelled").await;

	let requests = server.received_requests().await.unwrap();
	assert_eq!(
		requests
			.iter()
			.filter(|request| request.url.path() == "/sandboxes/sandbox_cancelled")
			.count(),
		1
	);
	let create: Value = serde_json::from_slice(
		&requests
			.iter()
			.find(|request| request.url.path() == "/sandboxes")
			.unwrap()
			.body,
	)
	.unwrap();
	assert!(
		create["timeout"]
			.as_u64()
			.is_some_and(|timeout| timeout <= 2)
	);
}

#[tokio::test]
async fn e2b_invalid_created_domain_is_cleaned_with_failure_metadata() {
	let server = MockServer::start().await;
	Mock::given(method("POST"))
		.and(path("/sandboxes"))
		.respond_with(ResponseTemplate::new(201).set_body_json(json!({
			"sandboxID": "sandbox_bad_domain",
			"domain": "bad/domain"
		})))
		.mount(&server)
		.await;
	Mock::given(method("DELETE"))
		.and(path("/sandboxes/sandbox_bad_domain"))
		.respond_with(ResponseTemplate::new(204))
		.mount(&server)
		.await;

	let backend = E2bSandboxBackend::new(
		policy_client(),
		server.uri().parse().unwrap(),
		"sandbox.example.com".into(),
		Duration::from_secs(2),
		SecretString::from("operator-key"),
		4096,
	)
	.unwrap();
	let error = backend
		.execute_batch(
			vec![managed_call(
				super::CODE_INTERPRETER_FUNCTION,
				"invalid-domain",
				json!({"code": "pass"}),
			)],
			ToolExecutionContext::default(),
		)
		.await
		.expect_err("provider-returned invalid domain must fail");
	assert_eq!(error.error, ToolInfrastructureError::Backend);
	assert_eq!(
		error.metadata.sandbox_create.unwrap().outcome,
		super::SandboxCleanupOutcome::Failure
	);
	assert_eq!(error.metadata.sandbox_execute, None);
	assert_eq!(
		error.metadata.sandbox_terminate.unwrap().outcome,
		super::SandboxCleanupOutcome::Success
	);
	assert_eq!(
		error.metadata.sandbox_cleanup,
		Some(super::SandboxCleanupOutcome::Success)
	);
	assert!(error.metadata.sandbox_cleanup_duration.is_some());
	let requests = server.received_requests().await.unwrap();
	assert_eq!(
		requests
			.iter()
			.filter(|request| request.url.path() == "/sandboxes/sandbox_bad_domain")
			.count(),
		1
	);
	assert!(
		requests
			.iter()
			.all(|request| request.url.path() != "/contexts")
	);
}

fn request(value: Value) -> responses::Request {
	serde_json::from_value(value).expect("valid Responses request")
}

#[test]
fn responses_request_extension_reads_replaces_and_preserves_unknown_fields() {
	let mut request = request(json!({
		"input": "hello",
		"model": "test",
		"future_request_field": {"nested": [1, true, null]}
	}));

	assert_eq!(
		request.rest_field("future_request_field"),
		Some(&json!({"nested": [1, true, null]}))
	);
	request.replace_rest_field("tool_choice", json!("auto"));

	let serialized = serde_json::to_value(request).unwrap();
	assert_eq!(serialized["tool_choice"], json!("auto"));
	assert_eq!(
		serialized["future_request_field"],
		json!({"nested": [1, true, null]})
	);
}

#[test]
fn responses_request_extension_preserves_unknown_raw_input_items() {
	let unknown_item = json!({
		"type": "future_output_item",
		"id": "item_123",
		"nested": {"z": [1, true, null], "a": "unchanged"}
	});
	let mut request = request(json!({"input": "hello", "model": "test"}));

	request.append_raw_input_values([unknown_item.clone()]);
	request.replace_rest_field("tool_choice", json!("auto"));

	let serialized = serde_json::to_value(request).unwrap();
	assert_eq!(serialized["input"][0]["type"], json!("message"));
	assert_eq!(serialized["input"][0]["content"][0]["text"], json!("hello"));
	assert_eq!(serialized["input"][1], unknown_item);
}

#[test]
fn runtime_usage_aggregation_saturates_and_preserves_none_plus_none() {
	let mut aggregate = None;
	let first_round = responses::Usage {
		input_tokens: u64::MAX - 1,
		output_tokens: 10,
		input_tokens_details: Some(responses::UsageInputDetails {
			cached_tokens: Some(u64::MAX - 1),
			cache_write_tokens: None,
			rest: Value::Null,
		}),
		output_tokens_details: Some(responses::UsageOutputDetails {
			reasoning_tokens: None,
			rest: Value::Null,
		}),
		total_tokens: Some(1),
		rest: Value::Null,
	};
	let second_round = responses::Usage {
		input_tokens: 10,
		output_tokens: u64::MAX,
		input_tokens_details: Some(responses::UsageInputDetails {
			cached_tokens: Some(10),
			cache_write_tokens: Some(3),
			rest: Value::Null,
		}),
		output_tokens_details: Some(responses::UsageOutputDetails {
			reasoning_tokens: None,
			rest: Value::Null,
		}),
		total_tokens: Some(4),
		rest: Value::Null,
	};

	aggregate_usage(&mut aggregate, &first_round);
	aggregate_usage(&mut aggregate, &second_round);

	let aggregate = aggregate.unwrap();
	assert_eq!(aggregate.input_tokens, u64::MAX);
	assert_eq!(aggregate.output_tokens, u64::MAX);
	assert_eq!(aggregate.total_tokens, Some(u64::MAX));
	let input_details = aggregate.input_tokens_details.unwrap();
	assert_eq!(input_details.cached_tokens, Some(u64::MAX));
	assert_eq!(input_details.cache_write_tokens, Some(3));
	let output_details = aggregate.output_tokens_details.unwrap();
	assert_eq!(output_details.reasoning_tokens, None);
}

fn response_with_calls(calls: Vec<Value>) -> responses::Response {
	serde_json::from_value(json!({
		"id": "resp_schema_validation",
		"status": "completed",
		"model": "test",
		"output": calls,
	}))
	.expect("valid Responses response")
}

#[test]
fn managed_final_response_applies_summary_without_retranslation() {
	let final_round = response_with_calls(vec![json!({
		"type": "message",
		"id": "msg_final",
		"role": "assistant",
		"status": "completed",
		"content": []
	})]);
	let summary = super::ToolRuntimeSummary {
		usage: Some(responses::Usage {
			input_tokens: 20,
			output_tokens: 22,
			input_tokens_details: None,
			output_tokens_details: None,
			total_tokens: Some(42),
			rest: Value::Null,
		}),
		rounds: 2,
		tool_calls: 1,
		client_streaming: false,
		include_obfuscation: false,
		client_tools: Some(json!([{"type": "function", "name": "client_tool"}])),
	};

	let finalized = finalize_managed_response(final_round, &summary);

	assert_eq!(finalized.usage.unwrap().total_tokens, Some(42));
	assert_eq!(
		finalized.rest["tools"],
		json!([{"type": "function", "name": "client_tool"}])
	);
}

fn function_call(name: &str, call_id: &str, arguments: &str) -> Value {
	json!({
		"type": "function_call",
		"id": format!("fc_{call_id}"),
		"call_id": call_id,
		"name": name,
		"arguments": arguments,
		"status": "completed",
	})
}

fn tools(request: &responses::Request) -> &Vec<Value> {
	request
		.rest
		.get("tools")
		.and_then(Value::as_array)
		.expect("tools array")
}

#[test]
fn remote_mcp_declaration_activates_runtime_and_redacts_authorization() {
	let mut request = request(json!({
		"input": "roll dice",
		"model": "test",
		"tools": [{
			"type": "mcp",
			"server_label": "dice",
			"server_description": "Roll dice",
			"server_url": "https://mcp.example.com/mcp",
			"authorization": "secret-oauth-token",
			"allowed_tools": ["roll"],
			"require_approval": "never"
		}]
	}));

	let Activation::Active(prepared) =
		prepare(&mut request, Some(&registry(vec![tool("managed", None)])))
			.expect("remote MCP activates the managed runtime")
	else {
		panic!("remote MCP must activate the managed runtime");
	};

	assert!(
		tools(&request).is_empty(),
		"MCP tools are discovered asynchronously"
	);
	assert_eq!(prepared.pending_remote_mcp.len(), 1);
	assert_eq!(prepared.pending_remote_mcp[0].server_label, "dice");
	assert_eq!(
		prepared.pending_remote_mcp[0].allowed_tools.as_deref(),
		Some(&["roll".to_owned()][..])
	);
	let client_tools = prepared.client_tools.as_ref().unwrap().as_array().unwrap();
	assert_eq!(client_tools[0]["server_url"], "https://mcp.example.com/mcp");
	assert!(client_tools[0].get("authorization").is_none());
}

#[test]
fn remote_mcp_declaration_rejects_oversized_client_controlled_metadata() {
	let cases = [
		json!({
			"type": "mcp",
			"server_label": "x".repeat(129),
			"server_url": "https://mcp.example.com/mcp",
			"require_approval": "never"
		}),
		json!({
			"type": "mcp",
			"server_label": "safe",
			"server_description": "x".repeat(4097),
			"server_url": "https://mcp.example.com/mcp",
			"require_approval": "never"
		}),
		json!({
			"type": "mcp",
			"server_label": "safe",
			"server_url": "https://mcp.example.com/mcp",
			"allowed_tools": (0..129).map(|index| format!("tool_{index}")).collect::<Vec<_>>(),
			"require_approval": "never"
		}),
		json!({
			"type": "mcp",
			"server_label": "safe",
			"server_url": "https://mcp.example.com/mcp",
			"allowed_tools": ["x".repeat(129)],
			"require_approval": "never"
		}),
	];

	for declaration in cases {
		let mut request = request(json!({"input": "test", "tools": [declaration]}));
		prepare(&mut request, Some(&registry(vec![tool("managed", None)])))
			.expect_err("oversized MCP metadata must be rejected before discovery");
	}
}

#[test]
fn remote_mcp_auto_approval_executes_without_an_approval_round_trip() {
	let mut request = request(json!({
		"input": "check Beijing weather",
		"model": "test",
		"tools": [{
			"type": "mcp",
			"server_label": "weather",
			"server_url": "https://mcp.weatherapi.com/mcp",
			"allowed_tools": ["get_forecast", "get_current_weather"],
			"require_approval": "auto"
		}]
	}));

	let Activation::Active(prepared) =
		prepare(&mut request, Some(&registry(vec![tool("managed", None)])))
			.expect("auto approval executes in AgentGateway")
	else {
		panic!("remote MCP must activate the managed runtime");
	};

	assert_eq!(prepared.pending_remote_mcp.len(), 1);
	assert_eq!(prepared.pending_remote_mcp[0].server_label, "weather");
	assert_eq!(
		prepared.pending_remote_mcp[0].allowed_tools.as_deref(),
		Some(&["get_forecast".to_owned(), "get_current_weather".to_owned(),][..])
	);
}

#[test]
fn remote_mcp_rejects_approval_flows_that_agentgateway_cannot_resume() {
	for require_approval in [json!("always"), json!({"always": {"tool_names": ["roll"]}})] {
		let mut request = request(json!({
			"input": "roll dice",
			"model": "test",
			"tools": [{
				"type": "mcp",
				"server_label": "dice",
				"server_url": "https://mcp.example.com/mcp",
				"require_approval": require_approval
			}]
		}));

		let error = prepare(&mut request, Some(&registry(vec![tool("managed", None)])))
			.expect_err("approval flow is unsupported");
		assert!(
			error
				.to_string()
				.contains("require_approval must be auto or never")
		);
	}
}

#[test]
fn remote_mcp_rejects_unsafe_server_urls_before_discovery() {
	for server_url in [
		"http://mcp.example.com/mcp",
		"https://user:secret@mcp.example.com/mcp",
		"https://mcp.example.com/mcp#fragment",
		"not-a-url",
	] {
		let mut request = request(json!({
			"input": "test",
			"tools": [{
				"type": "mcp",
				"server_label": "unsafe",
				"server_url": server_url,
				"require_approval": "never"
			}]
		}));
		let error = prepare(&mut request, Some(&registry(vec![tool("managed", None)])))
			.expect_err("unsafe MCP URL must fail before network discovery");
		assert!(error.to_string().contains("server_url"));
	}
}

#[tokio::test]
async fn remote_mcp_discovery_obeys_total_runtime_deadline() {
	let listener = tokio::net::TcpListener::bind("localhost:0").await.unwrap();
	let address = listener.local_addr().unwrap();
	tokio::spawn(async move {
		let router = axum::Router::new().route(
			"/mcp",
			axum::routing::post(|| async {
				tokio::time::sleep(Duration::from_secs(1)).await;
				axum::http::StatusCode::NO_CONTENT
			}),
		);
		axum::serve(listener, router).await.unwrap();
	});

	let server = RemoteMcpServer {
		server_label: "slow".into(),
		server_description: None,
		server_url: format!("http://localhost:{}/mcp", address.port()),
		authorization: None,
		allowed_tools: None,
	};
	let started = Instant::now();
	let result = RemoteMcpBackend::connect_for_test(
		policy_client(),
		&server,
		RuntimeDeadline::new(Duration::from_millis(30)),
	)
	.await;
	let error = match result {
		Ok(_) => panic!("stalled discovery must respect the runtime deadline"),
		Err(error) => error,
	};

	assert_eq!(error, ToolInfrastructureError::Timeout);
	assert!(started.elapsed() < Duration::from_millis(250));
}

#[tokio::test]
async fn remote_mcp_discovers_filters_and_executes_over_one_session() {
	use std::sync::Arc;

	use rmcp::handler::server::ServerHandler;
	use rmcp::model::{
		CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ListToolsResult,
		PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
	};
	use rmcp::service::RequestContext;
	use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
	use rmcp::transport::streamable_http_server::{
		StreamableHttpServerConfig, StreamableHttpService,
	};
	use rmcp::{ErrorData as McpError, RoleServer};

	#[derive(Clone)]
	struct DiceServer;

	impl ServerHandler for DiceServer {
		fn get_info(&self) -> ServerInfo {
			ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
		}

		async fn list_tools(
			&self,
			_request: Option<PaginatedRequestParams>,
			_: RequestContext<RoleServer>,
		) -> Result<ListToolsResult, McpError> {
			let schema = Arc::new(
				json!({
					"type": "object",
					"properties": {"sides": {"type": "integer"}},
					"required": ["sides"],
					"additionalProperties": false
				})
				.as_object()
				.unwrap()
				.clone(),
			);
			Ok(ListToolsResult::with_all_items(vec![
				Tool::new("roll", "Roll one die", schema.clone()),
				Tool::new("hidden", "Must be filtered", schema),
			]))
		}

		async fn call_tool(
			&self,
			request: CallToolRequestParams,
			_: RequestContext<RoleServer>,
		) -> Result<CallToolResponse, McpError> {
			assert_eq!(request.name.as_ref(), "roll");
			assert_eq!(request.arguments.as_ref().unwrap()["sides"], 6);
			Ok(CallToolResult::success(vec![ContentBlock::text("4")]).into())
		}
	}

	let service = StreamableHttpService::new(
		|| Ok(DiceServer),
		LocalSessionManager::default().into(),
		StreamableHttpServerConfig::default().with_json_response(true),
	);
	let captured_headers = Arc::new(Mutex::new(Vec::new()));
	let capture = captured_headers.clone();
	let listener = tokio::net::TcpListener::bind("localhost:0").await.unwrap();
	let address = listener.local_addr().unwrap();
	tokio::spawn(async move {
		let router =
			axum::Router::new()
				.nest_service("/mcp", service)
				.layer(axum::middleware::from_fn(
					move |request: axum::extract::Request, next: axum::middleware::Next| {
						let capture = capture.clone();
						async move {
							capture.lock().unwrap().push(request.headers().clone());
							next.run(request).await
						}
					},
				));
		axum::serve(listener, router).await.unwrap();
	});

	let server = RemoteMcpServer {
		server_label: "dice".into(),
		server_description: Some("Dice tools".into()),
		server_url: format!("http://localhost:{}/mcp", address.port()),
		authorization: Some(SecretString::from("mcp-oauth-token")),
		allowed_tools: Some(vec!["roll".into()]),
	};
	let (backend, tools) = RemoteMcpBackend::connect_for_test(
		policy_client(),
		&server,
		RuntimeDeadline::new(Duration::from_secs(5)),
	)
	.await
	.unwrap();
	assert_eq!(tools.len(), 1);
	assert_eq!(tools[0].remote_name, "roll");
	assert_eq!(tools[0].description.as_deref(), Some("Roll one die"));
	assert_eq!(tools[0].input_schema["required"], json!(["sides"]));

	let result = execute_one(
		backend.as_ref(),
		ManagedToolCall {
			public_name: "dice.roll".into(),
			internal_name: "_agentgateway_mcp_0_0".into(),
			call_id: "call_1".into(),
			arguments: json!({"sides": 6}),
			trusted_options: json!({"remote_tool_name": "roll"}),
		},
		ToolExecutionContext::default(),
	)
	.await
	.unwrap()
	.into_model_output(4096)
	.unwrap();
	assert_eq!(result["ok"], true);
	assert_eq!(result["content"][0]["text"], "4");
	let second = execute_one(
		backend.as_ref(),
		ManagedToolCall {
			public_name: "dice.roll".into(),
			internal_name: "_agentgateway_mcp_0_0".into(),
			call_id: "call_2".into(),
			arguments: json!({"sides": 6}),
			trusted_options: json!({"remote_tool_name": "roll"}),
		},
		ToolExecutionContext::default(),
	)
	.await
	.unwrap()
	.into_model_output(4096)
	.unwrap();
	assert_eq!(second["content"][0]["text"], "4");
	assert!(captured_headers.lock().unwrap().iter().all(|headers| {
		headers
			.get("authorization")
			.and_then(|value| value.to_str().ok())
			== Some("Bearer mcp-oauth-token")
	}));
	assert!(captured_headers.lock().unwrap().iter().all(|headers| {
		headers.get("host").and_then(|value| value.to_str().ok())
			== Some(format!("localhost:{}", address.port()).as_str())
	}));
	let session_ids = captured_headers
		.lock()
		.unwrap()
		.iter()
		.filter_map(|headers| {
			headers
				.get("mcp-session-id")
				.and_then(|value| value.to_str().ok())
				.map(str::to_owned)
		})
		.collect::<BTreeSet<_>>();
	assert_eq!(
		session_ids.len(),
		1,
		"all post-initialize calls reuse one MCP session"
	);
}

#[tokio::test]
async fn remote_mcp_tools_are_installed_as_namespaced_model_functions() {
	let mut request = request(json!({
		"input": "roll dice",
		"model": "test",
		"tool_choice": {"type": "mcp", "server_label": "dice", "name": "roll"},
		"tools": [{
			"type": "mcp",
			"server_label": "dice",
			"server_description": "Dice utilities",
			"server_url": "https://mcp.example.com/mcp",
			"allowed_tools": ["roll"],
			"require_approval": "never"
		}]
	}));
	let Activation::Active(mut prepared) =
		prepare(&mut request, Some(&registry(vec![tool("managed", None)]))).unwrap()
	else {
		panic!("expected active runtime");
	};

	prepared
		.install_remote_mcp_tools_for_test(
			0,
			Arc::new(BatchBackend),
			vec![super::RemoteMcpTool {
				remote_name: "roll".into(),
				description: Some("Roll one die".into()),
				input_schema: json!({
					"type": "object",
					"properties": {"sides": {"type": "integer"}},
					"required": ["sides"],
					"additionalProperties": false
				}),
			}],
		)
		.unwrap();

	assert_eq!(tools(&prepared.canonical_request).len(), 1);
	assert_eq!(tools(&prepared.canonical_request)[0]["type"], "function");
	assert_eq!(
		tools(&prepared.canonical_request)[0]["name"],
		"_agentgateway_mcp_0_0"
	);
	assert_eq!(
		tools(&prepared.canonical_request)[0]["description"],
		"[dice] Dice utilities\n\nRoll one die"
	);
	assert_eq!(
		prepared.canonical_request.rest_field("tool_choice"),
		Some(&json!({"type": "function", "name": "_agentgateway_mcp_0_0"}))
	);

	let calls = prepared
		.collect_calls(&response_with_calls(vec![function_call(
			"_agentgateway_mcp_0_0",
			"mcp_call_1",
			"{\"sides\":6}",
		)]))
		.unwrap();
	assert_eq!(calls[0].public_name, "dice.roll");
	assert_eq!(calls[0].trusted_options["remote_tool_name"], "roll");
	let mut budget = RuntimeBudget::new(&prepared.registry, policy_client()).unwrap();
	let outputs = execute_batch(&prepared.registry, calls, true, &mut budget)
		.await
		.unwrap();
	assert_eq!(outputs[0]["type"], "function_call_output");
	assert!(
		outputs[0]["output"]
			.as_str()
			.unwrap()
			.contains("mcp_call_1")
	);
	assert_eq!(budget.execution_records()[0].tool, "remote_mcp");

	prepared.append_round_history(Vec::new(), Vec::new());
	assert_eq!(
		prepared.canonical_request.rest_field("tool_choice"),
		Some(&json!("auto")),
		"a forced first-round tool choice must not force every continuation round"
	);
}

#[tokio::test]
async fn remote_mcp_composed_declaration_description_is_bounded() {
	let mut request = request(json!({
		"input": "test",
		"tools": [{
			"type": "mcp",
			"server_label": "remote",
			"server_description": "s".repeat(4096),
			"server_url": "https://mcp.example.com/mcp",
			"require_approval": "never"
		}]
	}));
	let Activation::Active(mut prepared) =
		prepare(&mut request, Some(&registry(vec![tool("managed", None)]))).unwrap()
	else {
		panic!("expected active runtime");
	};
	prepared
		.install_remote_mcp_tools_for_test(
			0,
			Arc::new(BatchBackend),
			vec![super::RemoteMcpTool {
				remote_name: "bounded".into(),
				description: Some("t".repeat(4096)),
				input_schema: json!({"type": "object"}),
			}],
		)
		.unwrap();

	let description = tools(&prepared.canonical_request)[0]["description"]
		.as_str()
		.unwrap();
	assert!(
		description.len() <= 4096,
		"description bytes={}",
		description.len()
	);
}

#[test]
fn managed_function_schema_and_unrelated_request_fields_are_unchanged() {
	let raw = json!({
		"input": "weather",
		"model": "test",
		"custom_field": { "unchanged": true },
		"tools": [{
			"type": "function",
			"name": "managed",
			"description": "A client-managed function",
			"strict": false,
			"parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
		}]
	});
	let mut request = request(raw.clone());

	let activation = prepare(&mut request, Some(&registry(vec![tool("managed", None)])))
		.expect("managed request activates runtime");

	assert!(matches!(activation, Activation::Active(_)));
	assert_eq!(serde_json::to_value(request).unwrap(), raw);
}

#[tokio::test]
async fn managed_function_arguments_must_match_the_advertised_json_schema() {
	let schema = json!({
		"type": "object",
		"properties": { "city": { "type": "string" } },
		"required": ["city"],
		"additionalProperties": false,
	});
	let mut request = request(json!({
		"input": "weather",
		"tools": [{ "type": "function", "name": "managed", "parameters": schema }],
	}));
	let Activation::Active(prepared) =
		prepare(&mut request, Some(&registry(vec![tool("managed", None)]))).unwrap()
	else {
		panic!("managed request must activate the runtime");
	};

	for (label, arguments) in [
		("wrong type", r#"{"city":7}"#),
		("missing required", r#"{}"#),
		(
			"additional property",
			r#"{"city":"Hangzhou","schema_secret":"do-not-leak"}"#,
		),
	] {
		let response = response_with_calls(vec![function_call("managed", "call_1", arguments)]);
		let error = prepared.collect_calls(&response).expect_err(label);
		let public = error.into_openai_response();
		assert_eq!(public.status(), ::http::StatusCode::BAD_REQUEST, "{label}");
		let body = public.into_body().collect().await.unwrap().to_bytes();
		let body = std::str::from_utf8(&body).unwrap();
		assert!(
			body.contains("arguments do not match declared schema"),
			"{label}: {body}"
		);
		assert!(!body.contains(arguments), "{label}: {body}");
		assert!(!body.contains("schema_secret"), "{label}: {body}");
		assert!(body.len() < 512, "client error must stay bounded: {body}");
	}

	let response = response_with_calls(vec![function_call(
		"managed",
		"call_valid",
		r#"{"city":"Hangzhou"}"#,
	)]);
	let calls = prepared
		.collect_calls(&response)
		.expect("valid schema-conforming arguments");
	assert_eq!(calls.len(), 1);
	assert_eq!(calls[0].arguments, json!({"city": "Hangzhou"}));
}

#[tokio::test]
async fn malformed_or_invalid_managed_function_schemas_fail_sanitized_at_request_preparation() {
	for (label, parameters) in [
		("missing parameters", None),
		(
			"invalid schema",
			Some(json!({
				"type": "object",
				"required": "schema_secret_do_not_leak",
			})),
		),
		(
			"external schema reference",
			Some(json!({
				"$ref": "https://schemas.invalid/schema_secret_do_not_leak.json",
			})),
		),
	] {
		let mut declaration = json!({ "type": "function", "name": "managed" });
		if let Some(parameters) = parameters {
			declaration
				.as_object_mut()
				.unwrap()
				.insert("parameters".to_owned(), parameters);
		}
		let mut request = request(json!({ "input": "weather", "tools": [declaration] }));
		let error =
			prepare(&mut request, Some(&registry(vec![tool("managed", None)]))).expect_err(label);
		let public = error.into_openai_response();
		assert_eq!(public.status(), ::http::StatusCode::BAD_REQUEST, "{label}");
		let body = public.into_body().collect().await.unwrap().to_bytes();
		let body = std::str::from_utf8(&body).unwrap();
		assert!(body.contains("valid JSON Schema"), "{label}: {body}");
		assert!(
			!body.contains("schema_secret_do_not_leak"),
			"{label}: {body}"
		);
		assert!(body.len() < 512, "client error must stay bounded: {body}");
	}
}

fn prepared_code_runtime(parallel: bool) -> PreparedToolRuntime {
	let mut request = request(json!({
		"input": "calculate",
		"parallel_tool_calls": parallel,
		"tools": [{ "type": "code_interpreter", "container": { "type": "auto" } }],
	}));
	let Activation::Active(prepared) = prepare(
		&mut request,
		Some(&registry(vec![tool(
			"code_interpreter",
			Some(BuiltinTool::CodeInterpreter),
		)])),
	)
	.unwrap() else {
		panic!("code request must activate the runtime");
	};
	prepared
}

#[test]
fn gateway_enforces_sandbox_batch_boundary_before_adapter_execution() {
	let prepared = prepared_code_runtime(true);
	let calls = |count| {
		response_with_calls(
			(0..count)
				.map(|index| {
					function_call(
						super::CODE_INTERPRETER_FUNCTION,
						&format!("call-{index}"),
						r#"{"code":"pass"}"#,
					)
				})
				.collect(),
		)
	};
	assert_eq!(
		prepared.collect_calls(&calls(8)).expect("8 is valid").len(),
		8,
	);
	let error = prepared
		.collect_calls(&calls(9))
		.expect_err("9 exceeds the contract");
	assert!(error.to_string().contains("Sandbox batch limit"), "{error}");
}

#[test]
fn gateway_enforces_sandbox_code_utf8_byte_boundary() {
	let prepared = prepared_code_runtime(true);
	for (size, accepted) in [(32 * 1024, true), (32 * 1024 + 1, false)] {
		let arguments = serde_json::to_string(&json!({"code": "x".repeat(size)})).unwrap();
		let response = response_with_calls(vec![function_call(
			super::CODE_INTERPRETER_FUNCTION,
			"code-boundary",
			&arguments,
		)]);
		let result = prepared.collect_calls(&response);
		assert_eq!(result.is_ok(), accepted, "code size {size}: {result:?}");
	}
}

#[test]
fn gateway_enforces_sandbox_call_id_utf8_byte_boundary() {
	let prepared = prepared_code_runtime(true);
	for (size, accepted) in [(256, true), (257, false)] {
		let call_id = "i".repeat(size);
		let response = response_with_calls(vec![function_call(
			super::CODE_INTERPRETER_FUNCTION,
			&call_id,
			r#"{"code":"pass"}"#,
		)]);
		let result = prepared.collect_calls(&response);
		assert_eq!(result.is_ok(), accepted, "call ID size {size}: {result:?}");
	}
}

#[test]
fn inactive_request_preserves_unmanaged_tools_and_request_fields() {
	let raw = json!({
		"input": "weather",
		"custom_field": { "unchanged": true },
		"tools": [{ "type": "function", "name": "client_only", "parameters": {} }]
	});
	let mut request = request(raw.clone());

	let activation = prepare(&mut request, Some(&registry(vec![tool("managed", None)])))
		.expect("unmanaged request remains inactive");

	assert!(matches!(activation, Activation::Inactive));
	assert_eq!(serde_json::to_value(request).unwrap(), raw);
}

#[test]
fn maps_web_search_to_exact_function_schema_and_retains_trusted_options() {
	let mut request = request(json!({
		"input": "latest news",
		"tools": [{
			"type": "web_search",
			"filters": { "allowed_domains": ["allowed.test"] },
			"search_context_size": "medium",
			"user_location": { "country": "CN" }
		}]
	}));

	let activation = prepare(
		&mut request,
		Some(&registry(vec![tool(
			"web_search",
			Some(BuiltinTool::WebSearch),
		)])),
	)
	.expect("web search activates runtime");

	let Activation::Active(prepared) = activation else {
		panic!("expected active runtime");
	};
	assert_eq!(
		tools(&prepared.canonical_request),
		&vec![json!({
			"type": "function",
			"name": "_agentgateway_web_search",
			"description": "Search the web for current information and return relevant sources.",
			"strict": true,
			"parameters": {
				"type": "object",
				"properties": { "query": { "type": "string" } },
				"required": ["query"],
				"additionalProperties": false
			}
		})]
	);
	assert_eq!(
		prepared
			.registry
			.trusted_options("_agentgateway_web_search"),
		Some(&json!({
			"allowed_domains": ["allowed.test"],
			"search_context_size": "medium",
			"user_location": { "type": "approximate", "country": "CN" }
		}))
	);
	assert!(
		!serde_json::to_value(&prepared.canonical_request)
			.unwrap()
			.to_string()
			.contains("allowed_domains")
	);
}

#[test]
fn preserves_public_web_search_context_sizes_in_http_contract() {
	for context_size in ["low", "medium", "high"] {
		let mut request = request(json!({
			"input": "latest news",
			"tools": [{
				"type": "web_search",
				"search_context_size": context_size
			}]
		}));

		let activation = prepare(
			&mut request,
			Some(&registry(vec![tool(
				"web_search",
				Some(BuiltinTool::WebSearch),
			)])),
		)
		.expect("web search activates runtime");
		let Activation::Active(prepared) = activation else {
			panic!("expected active runtime");
		};
		assert_eq!(
			prepared
				.registry
				.trusted_options("_agentgateway_web_search"),
			Some(&json!({"search_context_size": context_size})),
			"{context_size} must be preserved in the HTTP backend contract"
		);
	}
}

#[test]
fn rejects_malformed_web_search_options() {
	for (label, declaration, expected) in [
		(
			"allowed_domains_string",
			json!({ "type": "web_search", "filters": { "allowed_domains": "allowed.test" } }),
			"invalid web_search declaration",
		),
		(
			"search_context_size_invalid",
			json!({ "type": "web_search", "search_context_size": "large" }),
			"invalid web_search declaration",
		),
		(
			"user_location_invalid_type",
			json!({ "type": "web_search", "user_location": { "type": "precise" } }),
			"invalid web_search declaration",
		),
		(
			"user_location_unknown_field",
			json!({ "type": "web_search", "user_location": { "latitude": 30 } }),
			"invalid web_search declaration",
		),
		(
			"filters_unknown_field",
			json!({ "type": "web_search", "filters": { "blocked_domains": ["allowed.test"] } }),
			"invalid web_search declaration",
		),
		(
			"unknown_web_search_option",
			json!({ "type": "web_search", "provider_hint": "untrusted" }),
			"invalid web_search declaration",
		),
	] {
		let mut request = request(json!({ "input": "news", "tools": [declaration] }));
		let error = prepare(
			&mut request,
			Some(&registry(vec![tool(
				"web_search",
				Some(BuiltinTool::WebSearch),
			)])),
		)
		.expect_err(label);
		assert!(error.to_string().contains(expected), "{label}: {error}");
	}
}

#[test]
fn maps_code_interpreter_to_exact_function_schema() {
	let mut request = request(json!({
		"input": "calculate",
		"tools": [{ "type": "code_interpreter", "container": { "type": "auto" } }]
	}));

	let activation = prepare(
		&mut request,
		Some(&registry(vec![tool(
			"code_interpreter",
			Some(BuiltinTool::CodeInterpreter),
		)])),
	)
	.expect("code interpreter activates runtime");

	let Activation::Active(prepared) = activation else {
		panic!("expected active runtime");
	};
	assert_eq!(
		tools(&prepared.canonical_request),
		&vec![json!({
			"type": "function",
			"name": "_agentgateway_code_interpreter",
			"description": "Execute Python code in an isolated sandbox and return stdout and stderr.",
			"strict": true,
			"parameters": {
				"type": "object",
				"properties": { "code": { "type": "string" } },
				"required": ["code"],
				"additionalProperties": false
			}
		})]
	);
}

#[test]
fn parallel_tool_calls_false_is_retained_in_prepared_runtime() {
	let mut request = request(json!({
		"input": "weather",
		"parallel_tool_calls": false,
		"tools": [{ "type": "function", "name": "managed", "parameters": { "type": "object" } }]
	}));

	let activation = prepare(&mut request, Some(&registry(vec![tool("managed", None)])))
		.expect("managed function activates runtime");

	let Activation::Active(prepared) = activation else {
		panic!("expected active runtime");
	};
	assert!(!prepared.parallel);
}

#[test]
fn streaming_is_retained_for_the_client_and_disabled_for_internal_rounds() {
	let mut streaming_request = request(json!({
		"input": "weather",
		"stream": true,
		"stream_options": {"include_obfuscation": false},
		"tools": [{ "type": "function", "name": "managed", "parameters": { "type": "object" } }]
	}));

	let activation = prepare(
		&mut streaming_request,
		Some(&registry(vec![tool("managed", None)])),
	)
	.expect("streaming managed request activates runtime");

	let Activation::Active(prepared) = activation else {
		panic!("expected active runtime");
	};
	assert!(prepared.client_streaming);
	assert!(!prepared.include_obfuscation);
	assert_eq!(streaming_request.stream, Some(false));
	assert_eq!(prepared.canonical_request.stream, Some(false));
	assert_eq!(streaming_request.rest_field("stream_options"), None);
	assert_eq!(
		prepared.canonical_request.rest_field("stream_options"),
		None
	);

	let mut default_request = request(json!({
		"input": "weather",
		"stream": true,
		"tools": [{ "type": "function", "name": "managed", "parameters": { "type": "object" } }]
	}));
	let Activation::Active(default_prepared) = prepare(
		&mut default_request,
		Some(&registry(vec![tool("managed", None)])),
	)
	.unwrap() else {
		panic!("expected active runtime");
	};
	assert!(default_prepared.include_obfuscation);
}

#[test]
fn streaming_rejects_invalid_obfuscation_options() {
	for stream_options in [json!(true), json!({"include_obfuscation": "false"})] {
		let mut request = request(json!({
			"input": "weather",
			"stream": true,
			"stream_options": stream_options,
			"tools": [{ "type": "function", "name": "managed", "parameters": { "type": "object" } }]
		}));
		let error = prepare(&mut request, Some(&registry(vec![tool("managed", None)])))
			.expect_err("invalid stream options must be rejected");
		assert!(error.to_string().contains("stream_options"), "{error}");
	}
}

#[test]
fn streaming_encoder_uses_the_response_status_terminal_event() {
	for (status, terminal_event) in [
		("completed", "response.completed"),
		("incomplete", "response.incomplete"),
		("failed", "response.failed"),
	] {
		let response = serde_json::from_value::<responses::Response>(json!({
			"id": format!("resp_{status}"),
			"status": status,
			"model": "test",
			"output": []
		}))
		.unwrap();

		let stream = String::from_utf8(encode_streaming_response(&response, false).unwrap()).unwrap();
		let last = stream
			.trim_end()
			.rsplit("\n\n")
			.next()
			.expect("terminal SSE frame");
		assert!(
			last.starts_with(&format!("event: {terminal_event}\n")),
			"status={status}: {last}"
		);
	}
	let unknown = serde_json::from_value::<responses::Response>(json!({
		"id": "resp_cancelled",
		"status": "cancelled",
		"model": "test",
		"output": []
	}))
	.unwrap();
	assert!(encode_streaming_response(&unknown, false).is_err());
}

#[test]
fn streaming_encoder_obfuscates_deltas_by_default_but_allows_opt_out() {
	let response = response_with_calls(vec![json!({
		"type": "message",
		"id": "msg_1",
		"role": "assistant",
		"status": "completed",
		"content": [{"type": "output_text", "text": "hello", "annotations": []}]
	})]);

	let obfuscated = String::from_utf8(encode_streaming_response(&response, true).unwrap()).unwrap();
	let plain = String::from_utf8(encode_streaming_response(&response, false).unwrap()).unwrap();
	let delta = |stream: &str| {
		stream
			.split("\n\n")
			.filter_map(|frame| frame.lines().find_map(|line| line.strip_prefix("data: ")))
			.map(|data| serde_json::from_str::<Value>(data).unwrap())
			.find(|event| event["type"] == "response.output_text.delta")
			.unwrap()
	};

	assert!(
		delta(&obfuscated)["obfuscation"]
			.as_str()
			.is_some_and(|value| !value.is_empty())
	);
	assert_eq!(delta(&plain).get("obfuscation"), None);
}

#[test]
fn rejects_non_boolean_parallel_tool_calls() {
	for value in [json!(null), json!("false"), json!(0), json!([]), json!({})] {
		let mut request = request(json!({
			"input": "weather",
			"parallel_tool_calls": value,
			"tools": [{ "type": "function", "name": "managed", "parameters": { "type": "object" } }]
		}));

		let error = prepare(&mut request, Some(&registry(vec![tool("managed", None)])))
			.expect_err("parallel_tool_calls must be a boolean");

		assert!(error.to_string().contains("parallel_tool_calls"), "{error}");
		assert!(error.to_string().contains("boolean"), "{error}");
	}
}

#[test]
fn rejects_mixed_managed_and_unmanaged_function_declarations_without_mutating_request() {
	let mut request = request(json!({
		"input": "weather",
		"tools": [
			{ "type": "function", "name": "managed", "parameters": { "type": "object" } },
			{ "type": "function", "name": "client_only", "parameters": { "type": "object" } }
		]
	}));
	let before = request.clone();

	let error = prepare(&mut request, Some(&registry(vec![tool("managed", None)])))
		.expect_err("mixed declarations are unsafe");

	assert!(error.to_string().contains("every function"));
	assert_eq!(
		serde_json::to_value(request).unwrap(),
		serde_json::to_value(before).unwrap()
	);
}

#[test]
fn rejects_reserved_duplicate_and_unsupported_builtin_declarations() {
	for (label, tool_declarations, expected) in [
		(
			"reserved",
			json!([{ "type": "function", "name": "_agentgateway_web_search", "parameters": {} }]),
			"reserved",
		),
		(
			"duplicate",
			json!([
				{ "type": "function", "name": "managed", "parameters": {} },
				{ "type": "function", "name": "managed", "parameters": {} }
			]),
			"duplicate",
		),
		(
			"duplicate_builtin",
			json!([
				{ "type": "code_interpreter", "container": { "type": "auto" } },
				{ "type": "code_interpreter", "container": { "type": "auto" } }
			]),
			"duplicate",
		),
		(
			"explicit_container",
			json!([{ "type": "code_interpreter", "container": "container_123" }]),
			"container",
		),
		(
			"files",
			json!([{ "type": "code_interpreter", "container": { "type": "auto" }, "file_ids": ["file_123"] }]),
			"file",
		),
		(
			"language",
			json!([{ "type": "code_interpreter", "container": { "type": "auto" }, "language": "javascript" }]),
			"language",
		),
	] {
		let mut request = request(json!({ "input": "test", "tools": tool_declarations }));
		let error = prepare(
			&mut request,
			Some(&registry(vec![
				tool("managed", None),
				tool("code_interpreter", Some(BuiltinTool::CodeInterpreter)),
			])),
		)
		.expect_err(label);
		assert!(
			error.to_string().to_lowercase().contains(expected),
			"{label}: {error}"
		);
	}
}

#[test]
fn rejects_active_background_and_conversation() {
	for (field, value) in [
		("background", json!(true)),
		("conversation", json!("conv_123")),
	] {
		let mut body = json!({
			"input": "weather",
			"tools": [{ "type": "function", "name": "managed", "parameters": {} }]
		});
		body
			.as_object_mut()
			.unwrap()
			.insert(field.to_owned(), value);
		let mut request = request(body);

		let error =
			prepare(&mut request, Some(&registry(vec![tool("managed", None)]))).expect_err(field);
		assert!(error.to_string().contains(field), "{error}");
	}
}

#[test]
fn rejects_builtin_missing_from_registry() {
	let mut request = request(json!({
		"input": "news",
		"tools": [{ "type": "web_search" }]
	}));

	let error = prepare(&mut request, Some(&registry(vec![tool("managed", None)])))
		.expect_err("builtin must be operator configured");

	assert!(error.to_string().contains("web_search"));
}

#[test]
fn application_errors_are_model_visible_structured_output() {
	let result = ToolExecutionResult::ApplicationError(ToolApplicationError::new(
		"execution_error",
		"NameError: name 'x' is not defined",
		true,
		"",
		"Traceback...",
	));

	assert_eq!(
		result.into_model_output(1024).unwrap(),
		json!({
			"ok": false,
			"error": {
				"type": "execution_error",
				"message": "NameError: name 'x' is not defined",
				"retryable": true
			},
			"stdout": "",
			"stderr": "Traceback..."
		})
	);
}

#[test]
fn infrastructure_errors_have_only_sanitized_openai_mapping() {
	let error = ToolInfrastructureError::authentication();
	let public = error.to_openai_error();
	assert_eq!(
		public,
		json!({
			"error": {
				"type": "tool_infrastructure_error",
				"message": "managed tool backend authentication failed",
				"code": "tool_backend_authentication_failed"
			}
		})
	);
	assert!(!serde_json::to_string(&public).unwrap().contains("secret"));
	assert_eq!(
		error.to_string(),
		"managed tool backend authentication failed"
	);
}

#[test]
fn arguments_are_strict_json_objects_and_respect_byte_limit() {
	assert_eq!(
		parse_arguments(r#"{"city":"Hangzhou"}"#, 64).unwrap(),
		json!({"city": "Hangzhou"})
	);
	for raw in ["[]", "null", "not json"] {
		let error = parse_arguments(raw, 64).expect_err(raw);
		assert!(error.to_string().contains("arguments"), "{raw}: {error}");
	}
	let error = parse_arguments(r#"{"city":"Hangzhou"}"#, 4).expect_err("byte limit");
	assert!(error.to_string().contains("byte"));
}

#[test]
fn python_output_is_normalized_and_truncated_on_utf8_boundaries() {
	let result = ToolExecutionResult::python(json!({
		"exit_code": 0,
		"stdout": "你好世界",
		"stderr": "",
		"timed_out": false,
		"truncated": false,
		"artifacts": []
	}));
	let output = result.into_model_output(105).unwrap();
	let stdout = output["stdout"].as_str().unwrap();
	assert!("你好世界".starts_with(stdout));
	assert!(stdout.len() < "你好世界".len());
	assert!(output["truncated"].as_bool().unwrap());
	assert!(output["ok"].as_bool().unwrap());
	assert!(output["artifacts"].is_array());
	assert!(serde_json::to_vec(&output).unwrap().len() <= 105);
}

#[test]
fn ordinary_success_does_not_infer_python_from_overlapping_keys() {
	let output = ToolExecutionResult::function(json!({
		"stdout": "ordinary",
		"artifacts": ["caller-defined"]
	}))
	.into_model_output(1024)
	.unwrap();
	assert_eq!(
		output,
		json!({
			"ok": true,
			"stdout": "ordinary",
			"artifacts": ["caller-defined"]
		})
	);
}

#[test]
fn typed_builtin_results_validate_their_shapes() {
	assert_eq!(
		ToolExecutionResult::web_search(json!({"results": []}))
			.into_model_output(1024)
			.unwrap(),
		json!({"ok": true, "results": []})
	);
	assert_eq!(
		ToolExecutionResult::python(json!({
			"exit_code": 0,
			"stdout": "",
			"stderr": "",
			"timed_out": false,
			"truncated": false,
			"artifacts": []
		}))
		.into_model_output(1024)
		.unwrap()["ok"],
		true
	);
	for result in [
		ToolExecutionResult::web_search(json!({})),
		ToolExecutionResult::web_search(json!({"results": "not-an-array"})),
		ToolExecutionResult::web_search(json!({"results": [], "unexpected": true})),
		ToolExecutionResult::python(json!({
			"exit_code": "zero",
			"stdout": "",
			"stderr": "",
			"timed_out": false,
			"truncated": false,
			"artifacts": []
		})),
		ToolExecutionResult::python(json!({
			"exit_code": 0,
			"stdout": "",
			"stderr": "",
			"timed_out": false,
			"truncated": false
		})),
		ToolExecutionResult::python(json!({
			"exit_code": 0,
			"stdout": "",
			"stderr": "",
			"timed_out": false,
			"truncated": false,
			"artifacts": [],
			"unexpected": true
		})),
	] {
		assert!(result.into_model_output(1024).is_err());
	}
}

#[test]
fn nested_web_results_and_application_error_text_are_bounded() {
	let search = ToolExecutionResult::web_search(json!({
		"results": [
			{"title": "这是一个非常长的结果标题", "snippet": "更多内容", "url": "https://example.test/1"},
			{"title": "drop this result", "snippet": "drop this result", "url": "https://example.test/2"}
		]
	}))
	.into_model_output(125)
	.unwrap();
	assert!(search["truncated"].as_bool().unwrap());
	assert!(serde_json::to_vec(&search).unwrap().len() <= 125);
	assert!(search["results"].is_array());

	let application = ToolExecutionResult::ApplicationError(ToolApplicationError::new(
		"execution_error",
		"这是一个非常长的错误消息，需要截断",
		true,
		"",
		"",
	))
	.into_model_output(120)
	.unwrap();
	assert!(application["truncated"].as_bool().unwrap());
	assert!(application["error"]["message"].is_string());
	assert!(serde_json::to_vec(&application).unwrap().len() <= 120);
}

struct BatchBackend;

#[async_trait::async_trait]
impl ToolBackend for BatchBackend {
	async fn execute_batch(
		&self,
		calls: Vec<ManagedToolCall>,
		_context: ToolExecutionContext,
	) -> Result<ToolBatchExecution, ToolBatchInfrastructureError> {
		Ok(ToolBatchExecution::new(
			calls
				.into_iter()
				.map(|call| ToolExecutionResult::function(json!({"call": call.call_id})))
				.collect(),
		))
	}
}

#[tokio::test]
async fn tool_backend_uses_single_batch_contract() {
	let backend = BatchBackend;
	let calls = ["first", "second"]
		.into_iter()
		.map(|call_id| ManagedToolCall {
			public_name: call_id.into(),
			internal_name: call_id.into(),
			call_id: call_id.into(),
			arguments: json!({}),
			trusted_options: json!({}),
		})
		.collect();
	let batch = backend
		.execute_batch(calls, ToolExecutionContext::default())
		.await
		.unwrap();
	assert_eq!(batch.results.len(), 2);
	assert_eq!(batch.metadata, ToolBatchMetadata::default());
	assert_eq!(
		batch.results[0].clone().into_model_output(1024).unwrap()["call"],
		"first"
	);
	assert_eq!(
		batch.results[1].clone().into_model_output(1024).unwrap()["call"],
		"second"
	);
}

struct OutcomeBackend;

#[async_trait::async_trait]
impl ToolBackend for OutcomeBackend {
	async fn execute_batch(
		&self,
		calls: Vec<ManagedToolCall>,
		context: ToolExecutionContext,
	) -> Result<ToolBatchExecution, ToolBatchInfrastructureError> {
		execute_sequentially(
			|call, _context| async move {
				match call.call_id.as_str() {
					"application" => Ok(ToolExecutionResult::ApplicationError(
						ToolApplicationError::execution_error("bad input", true, "", ""),
					)),
					"infrastructure" => Err(ToolInfrastructureError::backend()),
					_ => Ok(ToolExecutionResult::function(json!({"call": call.call_id}))),
				}
			},
			calls,
			context,
		)
		.await
	}
}

#[tokio::test]
async fn batch_keeps_application_errors_but_stops_on_infrastructure_errors() {
	let backend = OutcomeBackend;
	let call = |call_id: &str| ManagedToolCall {
		public_name: call_id.into(),
		internal_name: call_id.into(),
		call_id: call_id.into(),
		arguments: json!({}),
		trusted_options: json!({}),
	};
	let batch = backend
		.execute_batch(
			vec![call("application"), call("success")],
			ToolExecutionContext::default(),
		)
		.await
		.unwrap();
	let values = batch.results;
	assert!(matches!(
		values[0],
		ToolExecutionResult::ApplicationError(_)
	));
	assert!(matches!(values[1], ToolExecutionResult::Function(_)));

	let error = backend
		.execute_batch(
			vec![call("infrastructure"), call("success")],
			ToolExecutionContext::default(),
		)
		.await
		.expect_err("infrastructure errors stop the batch");
	assert_eq!(error.error, ToolInfrastructureError::Backend);
}

#[tokio::test]
async fn sequential_execution_never_invokes_calls_after_infrastructure_failure() {
	let invoked = Arc::new(Mutex::new(Vec::new()));
	let observed = invoked.clone();
	let result = execute_sequentially(
		move |call, _context| {
			let invoked = observed.clone();
			async move {
				invoked.lock().unwrap().push(call.call_id.to_string());
				if call.call_id.as_str() == "stop" {
					Err(ToolInfrastructureError::backend())
				} else {
					Ok(ToolExecutionResult::function(json!({"ok": true})))
				}
			}
		},
		vec![
			managed_call("tool", "first", json!({})),
			managed_call("tool", "stop", json!({})),
			managed_call("tool", "must_not_run", json!({})),
		],
		ToolExecutionContext::default(),
	)
	.await;
	assert_eq!(result.unwrap_err().error, ToolInfrastructureError::Backend);
	assert_eq!(
		*invoked.lock().unwrap(),
		vec!["first".to_owned(), "stop".to_owned()]
	);
}

fn managed_call(internal_name: &str, call_id: &str, arguments: Value) -> ManagedToolCall {
	ManagedToolCall {
		public_name: internal_name.into(),
		internal_name: internal_name.into(),
		call_id: call_id.into(),
		arguments,
		trusted_options: json!({}),
	}
}

async fn execute_one<B: ToolBackend + ?Sized>(
	backend: &B,
	call: ManagedToolCall,
	context: ToolExecutionContext,
) -> Result<ToolExecutionResult, ToolInfrastructureError> {
	let mut batch = backend
		.execute_batch(vec![call], context)
		.await
		.map_err(|error| error.error)?;
	if batch.results.len() != 1 {
		return Err(ToolInfrastructureError::internal());
	}
	Ok(batch.results.pop().expect("one result checked"))
}

fn policy_client() -> PolicyClient {
	crate::test_helpers::policy_client()
}

fn http_backend(
	server: &MockServer,
	timeout: Duration,
	max_response_bytes: usize,
) -> HttpToolBackend {
	HttpToolBackend::new(
		policy_client(),
		server.uri().parse().unwrap(),
		timeout,
		Some(SecretString::from("operator-token")),
		max_response_bytes,
	)
	.expect("valid test backend")
}

#[tokio::test]
async fn http_backend_sends_authenticated_json_envelope_and_returns_typed_function_success() {
	let server = MockServer::start().await;
	Mock::given(method("POST"))
		.and(path("/"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({"forecast": "sunny"})))
		.mount(&server)
		.await;
	let backend = http_backend(&server, Duration::from_secs(2), 1024);

	let result = execute_one(
		&backend,
		managed_call("get_weather", "call_123", json!({"city": "Hangzhou"})),
		ToolExecutionContext {
			request_id: Some("request_123".into()),
			deadline: None,
		},
	)
	.await
	.unwrap()
	.into_model_output(1024)
	.unwrap();

	assert_eq!(result, json!({"ok": true, "forecast": "sunny"}));
	let requests = server.received_requests().await.unwrap();
	assert_eq!(requests.len(), 1);
	assert_eq!(
		requests[0].headers["authorization"],
		"Bearer operator-token"
	);
	assert_eq!(requests[0].headers["content-type"], "application/json");
	assert_eq!(
		serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
		json!({
			"tool_name": "get_weather",
			"call_id": "call_123",
			"arguments": {"city": "Hangzhou"},
			"context": {"request_id": "request_123", "deadline_ms": 2000}
		})
	);
}

#[tokio::test]
async fn http_backend_flattens_web_search_with_only_trusted_options() {
	let server = MockServer::start().await;
	Mock::given(method("POST"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({"results": []})))
		.mount(&server)
		.await;
	let backend = http_backend(&server, Duration::from_secs(2), 1024);
	let mut call = managed_call(
		super::WEB_SEARCH_FUNCTION,
		"search_1",
		json!({"query": "today's news"}),
	);
	call.trusted_options = json!({
		"allowed_domains": ["trusted.example"],
		"search_context_size": "medium"
	});

	let result = execute_one(&backend, call, ToolExecutionContext::default())
		.await
		.unwrap()
		.into_model_output(1024)
		.unwrap();
	assert_eq!(result, json!({"ok": true, "results": []}));
	let requests = server.received_requests().await.unwrap();
	assert_eq!(
		serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
		json!({
			"query": "today's news",
			"allowed_domains": ["trusted.example"],
			"search_context_size": "medium"
		})
	);
}

#[tokio::test]
async fn http_backend_keeps_structured_200_application_errors_model_visible() {
	let server = MockServer::start().await;
	Mock::given(method("POST"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({
			"ok": false,
			"error": {"type": "invalid_city", "message": "unknown city", "retryable": false},
			"stdout": "",
			"stderr": ""
		})))
		.mount(&server)
		.await;

	let result = execute_one(
		&http_backend(&server, Duration::from_secs(2), 1024),
		managed_call("get_weather", "call_1", json!({"city": "missing"})),
		ToolExecutionContext::default(),
	)
	.await
	.unwrap()
	.into_model_output(1024)
	.unwrap();
	assert_eq!(
		result,
		json!({
			"ok": false,
			"error": {"type": "invalid_city", "message": "unknown city", "retryable": false},
			"stdout": "",
			"stderr": ""
		})
	);
}

#[tokio::test]
async fn http_backend_times_out_once_without_retrying_execution() {
	let server = MockServer::start().await;
	Mock::given(method("POST"))
		.respond_with(
			ResponseTemplate::new(200)
				.set_delay(Duration::from_millis(150))
				.set_body_json(json!({"late": true})),
		)
		.mount(&server)
		.await;

	let error = execute_one(
		&http_backend(&server, Duration::from_millis(20), 1024),
		managed_call("slow", "call_1", json!({})),
		ToolExecutionContext {
			request_id: None,
			deadline: Some(Instant::now() + Duration::from_secs(1)),
		},
	)
	.await
	.expect_err("configured per-call timeout must win");
	assert_eq!(error, ToolInfrastructureError::Timeout);
	assert_eq!(
		server.received_requests().await.unwrap().len(),
		1,
		"an HTTP tool invocation is never retried"
	);
}

#[tokio::test]
async fn http_backend_timeout_covers_the_response_body() {
	use tokio::io::{AsyncReadExt, AsyncWriteExt};

	let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
	let address = listener.local_addr().unwrap();
	let server = tokio::spawn(async move {
		let (mut socket, _) = listener.accept().await.unwrap();
		let mut request = [0u8; 4096];
		let _ = socket.read(&mut request).await.unwrap();
		socket
			.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\n")
			.await
			.unwrap();
		socket.flush().await.unwrap();
		tokio::time::sleep(Duration::from_millis(150)).await;
		let _ = socket.write_all(br#"{"ok":true}"#).await;
	});
	let backend = HttpToolBackend::new(
		policy_client(),
		format!("http://{address}").parse().unwrap(),
		Duration::from_millis(20),
		None,
		1024,
	)
	.unwrap();

	let error = execute_one(
		&backend,
		managed_call("slow_body", "call_1", json!({})),
		ToolExecutionContext::default(),
	)
	.await
	.expect_err("the per-call timeout includes body consumption");
	assert_eq!(error, ToolInfrastructureError::Timeout);
	server.abort();
}

#[tokio::test]
async fn http_backend_non_success_and_auth_failures_are_sanitized_and_not_retried() {
	for (status, expected) in [
		(401, ToolInfrastructureError::Authentication),
		(503, ToolInfrastructureError::Backend),
	] {
		let server = MockServer::start().await;
		Mock::given(method("POST"))
			.respond_with(
				ResponseTemplate::new(status)
					.set_body_string("secret backend body https://internal.example"),
			)
			.mount(&server)
			.await;
		let error = execute_one(
			&http_backend(&server, Duration::from_secs(2), 1024),
			managed_call("tool", "call_1", json!({})),
			ToolExecutionContext::default(),
		)
		.await
		.expect_err("non-2xx is infrastructure failure");
		assert_eq!(error, expected);
		let public = error.sanitized_openai_error().to_string();
		assert!(!public.contains("secret"));
		assert!(!public.contains("internal.example"));
		assert_eq!(server.received_requests().await.unwrap().len(), 1);
	}
}

#[tokio::test]
async fn http_backend_rejects_oversized_and_invalid_json_bodies_without_leaking_them() {
	for (body, limit) in [
		("0123456789abcdef", 8usize),
		("not-json secret-response", 1024usize),
	] {
		let server = MockServer::start().await;
		Mock::given(method("POST"))
			.respond_with(ResponseTemplate::new(200).set_body_string(body))
			.mount(&server)
			.await;
		let error = execute_one(
			&http_backend(&server, Duration::from_secs(2), limit),
			managed_call("tool", "call_1", json!({})),
			ToolExecutionContext::default(),
		)
		.await
		.expect_err("body must be bounded valid JSON");
		assert_eq!(error, ToolInfrastructureError::Backend);
		assert!(!error.sanitized_openai_error().to_string().contains(body));
	}
}

#[tokio::test]
async fn http_backend_connectivity_failure_is_sanitized() {
	let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
	let address = listener.local_addr().unwrap();
	drop(listener);
	let backend = HttpToolBackend::new(
		policy_client(),
		format!("http://{address}").parse().unwrap(),
		Duration::from_millis(100),
		None,
		1024,
	)
	.unwrap();

	let error = execute_one(
		&backend,
		managed_call("tool", "call_1", json!({})),
		ToolExecutionContext::default(),
	)
	.await
	.expect_err("closed port must fail");
	assert_eq!(error, ToolInfrastructureError::Backend);
	assert!(
		!error
			.sanitized_openai_error()
			.to_string()
			.contains(&address.to_string())
	);
}

#[tokio::test]
async fn generic_http_backend_never_invokes_the_reserved_python_function() {
	let server = MockServer::start().await;
	Mock::given(method("POST"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({"unexpected": true})))
		.mount(&server)
		.await;

	let error = execute_one(
		&http_backend(&server, Duration::from_secs(2), 1024),
		managed_call(
			super::CODE_INTERPRETER_FUNCTION,
			"python_1",
			json!({"code": "print(1)"}),
		),
		ToolExecutionContext::default(),
	)
	.await
	.expect_err("code execution requires the dedicated sandbox adapter");
	assert_eq!(error, ToolInfrastructureError::Configuration);
	assert!(server.received_requests().await.unwrap().is_empty());
}

#[test]
fn tool_transport_outbound_subtypes_are_bounded_labels() {
	assert_eq!(OutboundCallSubtype::ToolHttp.as_str(), "ToolHttp");
	assert_eq!(OutboundCallSubtype::ToolE2b.as_str(), "ToolE2b");
}

#[derive(Clone)]
enum ControlledOutcome {
	SuccessAfter(Duration),
	ApplicationErrorAfter(Duration),
	InfrastructureAfterStarts(usize),
	Pending,
}

#[derive(Default)]
struct ControlledState {
	active: AtomicUsize,
	max_active: AtomicUsize,
	started: Mutex<Vec<String>>,
	completed: Mutex<Vec<String>>,
	cancelled: Mutex<Vec<String>>,
	batches: Mutex<Vec<Vec<String>>>,
	starts_changed: tokio::sync::Notify,
}

impl ControlledState {
	fn record_start(&self, call_id: &str) -> ActiveCall<'_> {
		self.started.lock().unwrap().push(call_id.to_owned());
		let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
		self.max_active.fetch_max(active, Ordering::SeqCst);
		self.starts_changed.notify_waiters();
		ActiveCall {
			state: self,
			call_id: call_id.to_owned(),
			completed: false,
		}
	}

	async fn wait_for_starts(&self, count: usize) {
		loop {
			let notified = self.starts_changed.notified();
			if self.started.lock().unwrap().len() >= count {
				return;
			}
			notified.await;
		}
	}
}

struct ActiveCall<'a> {
	state: &'a ControlledState,
	call_id: String,
	completed: bool,
}

impl ActiveCall<'_> {
	fn complete(mut self) {
		self.completed = true;
		self
			.state
			.completed
			.lock()
			.unwrap()
			.push(self.call_id.clone());
	}
}

impl Drop for ActiveCall<'_> {
	fn drop(&mut self) {
		self.state.active.fetch_sub(1, Ordering::SeqCst);
		if !self.completed {
			self
				.state
				.cancelled
				.lock()
				.unwrap()
				.push(self.call_id.clone());
		}
	}
}

struct ControlledBackend {
	state: Arc<ControlledState>,
	outcomes: HashMap<String, ControlledOutcome>,
}

impl ControlledBackend {
	fn new(
		state: Arc<ControlledState>,
		outcomes: impl IntoIterator<Item = (&'static str, ControlledOutcome)>,
	) -> Self {
		Self {
			state,
			outcomes: outcomes
				.into_iter()
				.map(|(call_id, outcome)| (call_id.to_owned(), outcome))
				.collect(),
		}
	}

	async fn controlled_result(
		&self,
		call: ManagedToolCall,
	) -> Result<ToolExecutionResult, ToolInfrastructureError> {
		let call_id = call.call_id.to_string();
		let active = self.state.record_start(&call_id);
		let outcome = self
			.outcomes
			.get(&call_id)
			.cloned()
			.unwrap_or(ControlledOutcome::SuccessAfter(Duration::ZERO));
		let result = match outcome {
			ControlledOutcome::SuccessAfter(delay) => {
				tokio::time::sleep(delay).await;
				Ok(ToolExecutionResult::function(json!({"call": call_id})))
			},
			ControlledOutcome::ApplicationErrorAfter(delay) => {
				tokio::time::sleep(delay).await;
				Ok(ToolExecutionResult::ApplicationError(
					ToolApplicationError::execution_error("bad input", false, "", ""),
				))
			},
			ControlledOutcome::InfrastructureAfterStarts(count) => {
				self.state.wait_for_starts(count).await;
				Err(ToolInfrastructureError::backend())
			},
			ControlledOutcome::Pending => pending().await,
		};
		active.complete();
		result
	}
}

#[async_trait::async_trait]
impl ToolBackend for ControlledBackend {
	async fn execute_batch(
		&self,
		calls: Vec<ManagedToolCall>,
		_context: ToolExecutionContext,
	) -> Result<ToolBatchExecution, ToolBatchInfrastructureError> {
		if calls.len() == 1 && calls[0].internal_name.as_str() != super::CODE_INTERPRETER_FUNCTION {
			let result = self
				.controlled_result(calls.into_iter().next().unwrap())
				.await?;
			return Ok(ToolBatchExecution::new(vec![result]));
		}
		let call_ids = calls
			.iter()
			.map(|call| call.call_id.to_string())
			.collect::<Vec<_>>();
		self.state.batches.lock().unwrap().push(call_ids.clone());
		let batch_id = format!("batch:{}", call_ids.join(","));
		let active = self.state.record_start(&batch_id);
		let mut results = Vec::with_capacity(calls.len());
		for call in calls {
			let call_id = call.call_id.to_string();
			let outcome = self
				.outcomes
				.get(&call_id)
				.cloned()
				.unwrap_or(ControlledOutcome::SuccessAfter(Duration::ZERO));
			match outcome {
				ControlledOutcome::SuccessAfter(delay) => {
					tokio::time::sleep(delay).await;
					results.push(ToolExecutionResult::function(json!({"call": call_id})));
				},
				ControlledOutcome::ApplicationErrorAfter(delay) => {
					tokio::time::sleep(delay).await;
					results.push(ToolExecutionResult::ApplicationError(
						ToolApplicationError::execution_error("bad input", false, "", ""),
					));
				},
				ControlledOutcome::InfrastructureAfterStarts(count) => {
					self.state.wait_for_starts(count).await;
					return Err(ToolInfrastructureError::backend().into());
				},
				ControlledOutcome::Pending => pending().await,
			}
		}
		active.complete();
		Ok(ToolBatchExecution::new(results))
	}
}

fn runtime_limits(
	max_tool_calls: usize,
	max_parallel_tool_calls: usize,
	total_timeout: Duration,
) -> RuntimeLimits {
	RuntimeLimits {
		total_timeout,
		max_rounds: 8,
		max_tool_calls,
		max_parallel_tool_calls,
		max_arguments_bytes: 65_536,
		max_output_bytes: 1_048_576,
	}
}

fn controlled_registry(
	limits: RuntimeLimits,
	ordinary_tools: &[&str],
	sandbox: bool,
) -> Arc<ToolRegistry> {
	let mut tools = ordinary_tools
		.iter()
		.map(|name| tool(name, None))
		.collect::<Vec<_>>();
	if sandbox {
		tools.push(ManagedToolConfig {
			name: "python".to_owned(),
			builtin: Some(BuiltinTool::CodeInterpreter),
			backend: ToolBackendConfig::E2b {
				api_url: SENSITIVE_SANDBOX_ENDPOINT.parse().unwrap(),
				domain: "sandbox.example.com".into(),
				timeout: Duration::from_secs(30),
				api_key: SecretString::from("operator-secret"),
			},
		});
	}
	Arc::new(
		ToolRegistry::compile(ToolRuntimeConfig { limits, tools }).expect("valid controlled registry"),
	)
}

fn controlled_budget(
	registry: &ToolRegistry,
	backends: impl IntoIterator<Item = (&'static str, Arc<dyn ToolBackend>)>,
) -> RuntimeBudget {
	RuntimeBudget::with_test_backends(
		registry,
		backends
			.into_iter()
			.map(|(name, backend)| (name.to_owned(), backend))
			.collect(),
	)
}

struct FakeResponsesRoundTrip {
	rounds: VecDeque<super::runner::ModelRound>,
	requests: Vec<Value>,
}

#[async_trait::async_trait]
impl super::runner::ResponsesRoundTrip for FakeResponsesRoundTrip {
	async fn execute_round(
		&mut self,
		request: &responses::Request,
		_remaining: Duration,
	) -> Result<super::runner::ModelRound, super::ToolRuntimeError> {
		self
			.requests
			.push(serde_json::to_value(request).expect("serializable request"));
		Ok(self.rounds.pop_front().expect("configured fake round"))
	}
}

fn successful_model_round(output: Vec<Value>) -> super::runner::ModelRound {
	super::runner::ModelRound::Success(Box::new(crate::llm::BufferedResponsesRound {
		response: response_with_calls(output.clone()),
		raw_output: output,
		reconstructed_upstream: crate::http::Response::new(crate::http::Body::empty()),
	}))
}

#[tokio::test]
async fn runner_rejects_non_completed_function_calls_before_backend_side_effects() {
	for status in [None, Some("in_progress"), Some("incomplete")] {
		let mut client_request = request(json!({
			"input": "run",
			"tools": [{
				"type": "function",
				"name": "managed",
				"parameters": {"type": "object", "additionalProperties": true}
			}]
		}));
		let Activation::Active(runtime) = prepare(
			&mut client_request,
			Some(&registry(vec![tool("managed", None)])),
		)
		.unwrap() else {
			panic!("managed request must activate the runtime");
		};
		let state = Arc::new(ControlledState::default());
		let backend = ControlledBackend::new(state.clone(), []);
		let budget = controlled_budget(&runtime.registry, [("managed", Arc::new(backend) as _)]);
		let mut call = function_call("managed", "call_1", "{}");
		match status {
			Some(status) => call["status"] = json!(status),
			None => {
				call.as_object_mut().unwrap().remove("status");
			},
		}
		let mut round_trip = FakeResponsesRoundTrip {
			rounds: VecDeque::from([successful_model_round(vec![call])]),
			requests: Vec::new(),
		};

		let error = match super::runner::run(runtime, budget, &mut round_trip).await {
			Ok(_) => panic!("only completed function calls may execute"),
			Err(error) => error,
		};
		let super::runner::RunError::Runtime(error) = error else {
			panic!("expected runtime validation error: {error:?}");
		};
		assert!(error.to_string().contains("completed status"), "{error}");
		assert!(
			state.started.lock().unwrap().is_empty(),
			"status {status:?} invoked a backend"
		);
	}
}

#[tokio::test]
async fn runner_owns_round_history_usage_and_summary() {
	let mut client_request = request(json!({
		"input": "run twice",
		"model": "test",
		"tools": [{
			"type": "function",
			"name": "managed",
			"parameters": {"type": "object", "additionalProperties": true}
		}]
	}));
	let Activation::Active(runtime) = prepare(
		&mut client_request,
		Some(&registry(vec![tool("managed", None)])),
	)
	.unwrap() else {
		panic!("managed request must activate the runtime");
	};
	let backend = ControlledBackend::new(Arc::new(ControlledState::default()), []);
	let budget = controlled_budget(&runtime.registry, [("managed", Arc::new(backend) as _)]);
	let mut round_trip = FakeResponsesRoundTrip {
		rounds: VecDeque::from([
			successful_model_round(vec![function_call("managed", "call_1", r#"{"x":1}"#)]),
			successful_model_round(vec![function_call("managed", "call_2", r#"{"x":2}"#)]),
			successful_model_round(vec![json!({
				"type": "message",
				"id": "msg_final",
				"role": "assistant",
				"status": "completed",
				"content": []
			})]),
		]),
		requests: Vec::new(),
	};

	let result = super::runner::run(runtime, budget, &mut round_trip)
		.await
		.expect("runner succeeds");

	assert_eq!(round_trip.requests.len(), 3);
	assert_eq!(result.summary.rounds, 3);
	assert_eq!(result.summary.tool_calls, 2);
	let final_input = round_trip.requests[2]["input"]
		.as_array()
		.expect("continuation input is an array");
	assert_eq!(
		final_input
			.iter()
			.filter(|item| item["type"] == "function_call_output")
			.count(),
		2
	);
}

fn batch_call(name: &str, call_id: &str) -> ManagedToolCall {
	managed_call(name, call_id, json!({"value": call_id}))
}

fn code_call(call_id: &str) -> ManagedToolCall {
	managed_call(
		super::CODE_INTERPRETER_FUNCTION,
		call_id,
		json!({"code": format!("print({call_id:?})")}),
	)
}

fn output_payload(output: &Value) -> Value {
	serde_json::from_str(output["output"].as_str().expect("string tool output"))
		.expect("JSON tool output")
}

const SENSITIVE_QUERY: &str = "tenant revenue forecast";
const SENSITIVE_CODE: &str = "print(customer_private_balance)";
const SENSITIVE_STDOUT: &str = "private stdout";
const SENSITIVE_STDERR: &str = "private stderr";
const SENSITIVE_RESULT_URL: &str = "https://results.invalid/private/artifact";
const SENSITIVE_ENDPOINT_IP: &str = "203.0.113.77";
const SENSITIVE_SANDBOX_ENDPOINT: &str = "https://sandbox-control.example.invalid";
const SENSITIVE_REQUEST_ID: &str = "request-sensitive-tenant-42";
const SENSITIVE_SANDBOX_TARGET: &str = "sandbox-target-sensitive";
const SENSITIVE_SANDBOX_TEMPLATE: &str = "sandbox-template-sensitive";
const SENSITIVE_SANDBOX_ID: &str = "sandbox-id-sensitive";

struct RuntimeMetricsBackend;

#[async_trait::async_trait]
impl ToolBackend for RuntimeMetricsBackend {
	async fn execute_batch(
		&self,
		calls: Vec<ManagedToolCall>,
		context: ToolExecutionContext,
	) -> Result<ToolBatchExecution, ToolBatchInfrastructureError> {
		execute_sequentially(
			|call, _context| async move {
				match call.call_id.as_str() {
					"call_application_sensitive" => Ok(ToolExecutionResult::ApplicationError(
						ToolApplicationError::execution_error(
							"private application failure",
							false,
							SENSITIVE_STDOUT,
							SENSITIVE_STDERR,
						),
					)),
					"call_truncation_sensitive" => Ok(ToolExecutionResult::function(json!({
						"result_url": SENSITIVE_RESULT_URL,
						"payload": SENSITIVE_CODE.repeat(64),
					}))),
					"call_timeout_sensitive" => pending().await,
					_ => Ok(ToolExecutionResult::function(json!({
						"result_url": SENSITIVE_RESULT_URL,
						"sandbox_target": SENSITIVE_SANDBOX_TARGET,
						"sandbox_template": SENSITIVE_SANDBOX_TEMPLATE,
						"sandbox_id": SENSITIVE_SANDBOX_ID,
					}))),
				}
			},
			calls,
			context,
		)
		.await
	}
}

fn runtime_metrics(mode: HistogramMode) -> (PrometheusRegistry, Arc<Metrics>) {
	let mut registry = PrometheusRegistry::default();
	let metrics = Arc::new(Metrics::new(
		agent_core::metrics::sub_registry(&mut registry),
		Default::default(),
		mode,
	));
	(registry, metrics)
}

fn sandbox_operation_count(
	metrics: &Metrics,
	operation: SandboxOperation,
	outcome: SandboxOperationOutcome,
) -> u64 {
	metrics
		.tool_runtime_sandbox_operations
		.get_or_create(&SandboxOperationLabels { operation, outcome })
		.get()
}

struct InvalidSandboxResultBackend;

#[async_trait::async_trait]
impl ToolBackend for InvalidSandboxResultBackend {
	async fn execute_batch(
		&self,
		calls: Vec<ManagedToolCall>,
		_context: ToolExecutionContext,
	) -> Result<ToolBatchExecution, ToolBatchInfrastructureError> {
		Ok(ToolBatchExecution::new(
			calls
				.into_iter()
				.map(|_| ToolExecutionResult::python(json!({"unexpected": true})))
				.collect(),
		))
	}
}

#[tokio::test(start_paused = true)]
async fn sandbox_operation_timeout_is_finalized_once() {
	let (_prometheus, metrics) = runtime_metrics(HistogramMode::Classic);
	let configured = controlled_registry(runtime_limits(1, 1, Duration::from_secs(5)), &[], true);
	let state = Arc::new(ControlledState::default());
	let backend: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
		state,
		[("pending", ControlledOutcome::Pending)],
	));
	let mut budget = controlled_budget_with_metrics(
		&configured,
		[(super::CODE_INTERPRETER_FUNCTION, backend)],
		metrics.clone(),
	);

	let error = execute_batch(&configured, vec![code_call("pending")], true, &mut budget)
		.await
		.expect_err("request deadline expires the Sandbox batch");
	assert_eq!(
		error.infrastructure_error(),
		Some(ToolInfrastructureError::Timeout)
	);
	assert_eq!(
		sandbox_operation_count(
			&metrics,
			SandboxOperation::Execute,
			SandboxOperationOutcome::Timeout,
		),
		1
	);
	assert_eq!(
		sandbox_operation_count(
			&metrics,
			SandboxOperation::Execute,
			SandboxOperationOutcome::Cancelled,
		),
		0
	);
	assert_eq!(
		sandbox_operation_count(
			&metrics,
			SandboxOperation::Cleanup,
			SandboxOperationOutcome::Timeout,
		),
		1
	);
}

#[tokio::test(start_paused = true)]
async fn sandbox_operation_sibling_failure_is_cancelled_once() {
	let (_prometheus, metrics) = runtime_metrics(HistogramMode::Classic);
	let configured = controlled_registry(
		runtime_limits(2, 2, Duration::from_secs(30)),
		&["ordinary"],
		true,
	);
	let state = Arc::new(ControlledState::default());
	let backend: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
		state,
		[
			(
				"ordinary_failure",
				ControlledOutcome::InfrastructureAfterStarts(2),
			),
			("sandbox_pending", ControlledOutcome::Pending),
		],
	));
	let mut budget = controlled_budget_with_metrics(
		&configured,
		[
			("ordinary", backend.clone()),
			(super::CODE_INTERPRETER_FUNCTION, backend),
		],
		metrics.clone(),
	);

	execute_batch(
		&configured,
		vec![
			batch_call("ordinary", "ordinary_failure"),
			code_call("sandbox_pending"),
		],
		true,
		&mut budget,
	)
	.await
	.expect_err("ordinary failure cancels the in-flight Sandbox batch");
	assert_eq!(
		sandbox_operation_count(
			&metrics,
			SandboxOperation::Execute,
			SandboxOperationOutcome::Cancelled,
		),
		1
	);
	assert_eq!(
		sandbox_operation_count(
			&metrics,
			SandboxOperation::Cleanup,
			SandboxOperationOutcome::Cancelled,
		),
		1
	);
}

#[tokio::test(start_paused = true)]
async fn queued_sandbox_operation_sibling_failure_is_cancelled_once() {
	let (_prometheus, metrics) = runtime_metrics(HistogramMode::Classic);
	let configured = controlled_registry(
		runtime_limits(2, 1, Duration::from_secs(30)),
		&["ordinary"],
		true,
	);
	let state = Arc::new(ControlledState::default());
	let backend: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
		state.clone(),
		[
			(
				"ordinary_failure",
				ControlledOutcome::InfrastructureAfterStarts(1),
			),
			("sandbox_queued", ControlledOutcome::Pending),
		],
	));
	let mut budget = controlled_budget_with_metrics(
		&configured,
		[
			("ordinary", backend.clone()),
			(super::CODE_INTERPRETER_FUNCTION, backend),
		],
		metrics.clone(),
	);

	execute_batch(
		&configured,
		vec![
			batch_call("ordinary", "ordinary_failure"),
			code_call("sandbox_queued"),
		],
		true,
		&mut budget,
	)
	.await
	.expect_err("ordinary failure cancels the queued Sandbox batch");
	assert!(state.batches.lock().unwrap().is_empty());
	assert_eq!(
		sandbox_operation_count(
			&metrics,
			SandboxOperation::Execute,
			SandboxOperationOutcome::Cancelled,
		),
		1
	);
	assert_eq!(
		sandbox_operation_count(
			&metrics,
			SandboxOperation::Cleanup,
			SandboxOperationOutcome::Cancelled,
		),
		0,
		"a never-started Sandbox batch has no cleanup lifecycle"
	);
}

#[tokio::test(start_paused = true)]
async fn sandbox_operation_budget_drop_is_cancelled_once() {
	let (_prometheus, metrics) = runtime_metrics(HistogramMode::Classic);
	let configured = controlled_registry(runtime_limits(1, 1, Duration::from_secs(30)), &[], true);
	let state = Arc::new(ControlledState::default());
	let backend: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
		state.clone(),
		[("pending_drop", ControlledOutcome::Pending)],
	));
	let mut budget = controlled_budget_with_metrics(
		&configured,
		[(super::CODE_INTERPRETER_FUNCTION, backend)],
		metrics.clone(),
	);
	let mut execution = Box::pin(execute_batch(
		&configured,
		vec![code_call("pending_drop")],
		true,
		&mut budget,
	));
	tokio::select! {
		_ = state.wait_for_starts(1) => {},
		_ = &mut execution => panic!("pending Sandbox batch unexpectedly completed"),
	}
	drop(execution);
	drop(budget);

	assert_eq!(
		sandbox_operation_count(
			&metrics,
			SandboxOperation::Execute,
			SandboxOperationOutcome::Cancelled,
		),
		1
	);
	assert_eq!(
		sandbox_operation_count(
			&metrics,
			SandboxOperation::Cleanup,
			SandboxOperationOutcome::Cancelled,
		),
		1
	);
}

#[tokio::test]
async fn sandbox_operation_normalization_failure_is_failure_not_success() {
	let (_prometheus, metrics) = runtime_metrics(HistogramMode::Classic);
	let configured = controlled_registry(runtime_limits(1, 1, Duration::from_secs(30)), &[], true);
	let backend: Arc<dyn ToolBackend> = Arc::new(InvalidSandboxResultBackend);
	let mut budget = controlled_budget_with_metrics(
		&configured,
		[(super::CODE_INTERPRETER_FUNCTION, backend)],
		metrics.clone(),
	);

	execute_batch(
		&configured,
		vec![code_call("invalid_result")],
		true,
		&mut budget,
	)
	.await
	.expect_err("invalid typed Sandbox result fails normalization");
	assert_eq!(
		sandbox_operation_count(
			&metrics,
			SandboxOperation::Execute,
			SandboxOperationOutcome::Failure,
		),
		1
	);
	assert_eq!(
		sandbox_operation_count(
			&metrics,
			SandboxOperation::Execute,
			SandboxOperationOutcome::Success,
		),
		0
	);
	assert_eq!(
		sandbox_operation_count(
			&metrics,
			SandboxOperation::Execute,
			SandboxOperationOutcome::Cancelled,
		),
		0
	);
}

fn controlled_budget_with_metrics(
	registry: &ToolRegistry,
	backends: impl IntoIterator<Item = (&'static str, Arc<dyn ToolBackend>)>,
	metrics: Arc<Metrics>,
) -> RuntimeBudget {
	RuntimeBudget::with_test_backends_and_metrics(
		registry,
		backends
			.into_iter()
			.map(|(name, backend)| (name.to_owned(), backend))
			.collect(),
		metrics,
	)
}

fn metric_labels(metric: &Metric) -> HashMap<&str, &str> {
	metric
		.label
		.iter()
		.map(|label| (label.name.as_str(), label.value.as_str()))
		.collect()
}

fn counter_value(family: &MetricFamily, expected: &[(&str, &str)]) -> Option<f64> {
	family
		.metric
		.iter()
		.find(|metric| {
			let labels = metric_labels(metric);
			labels.len() == expected.len()
				&& expected
					.iter()
					.all(|(key, value)| labels.get(key).copied() == Some(*value))
		})
		.and_then(|metric| metric.counter.as_ref())
		.map(|counter| counter.value)
}

fn assert_runtime_registry_schema(mode: HistogramMode, families: &[MetricFamily]) {
	let runtime_family_count = families
		.iter()
		.filter(|family| family.name.starts_with("agentgateway_tool_runtime_"))
		.count();
	let runtime = families
		.iter()
		.filter(|family| family.name.starts_with("agentgateway_tool_runtime_"))
		.map(|family| (family.name.as_str(), family))
		.collect::<HashMap<_, _>>();
	let expected_names = BTreeSet::from([
		"agentgateway_tool_runtime_requests_total",
		"agentgateway_tool_runtime_model_rounds_total",
		"agentgateway_tool_runtime_model_round_duration_seconds",
		"agentgateway_tool_runtime_calls_total",
		"agentgateway_tool_runtime_call_duration_seconds",
		"agentgateway_tool_runtime_sandbox_operations_total",
		"agentgateway_tool_runtime_sandbox_operation_duration_seconds",
		"agentgateway_tool_runtime_limit_exhaustions_total",
		"agentgateway_tool_runtime_output_truncations_total",
	]);
	assert_eq!(runtime_family_count, expected_names.len(), "mode: {mode:?}");
	assert_eq!(
		runtime.keys().copied().collect::<BTreeSet<_>>(),
		expected_names,
		"mode: {mode:?}"
	);

	let schemas = [
		(
			"agentgateway_tool_runtime_requests_total",
			&["outcome"][..],
			MetricType::Counter,
			"",
		),
		(
			"agentgateway_tool_runtime_model_rounds_total",
			&["outcome"][..],
			MetricType::Counter,
			"",
		),
		(
			"agentgateway_tool_runtime_model_round_duration_seconds",
			&[][..],
			MetricType::Histogram,
			"seconds",
		),
		(
			"agentgateway_tool_runtime_calls_total",
			&["backend", "outcome", "tool"][..],
			MetricType::Counter,
			"",
		),
		(
			"agentgateway_tool_runtime_call_duration_seconds",
			&["backend", "tool"][..],
			MetricType::Histogram,
			"seconds",
		),
		(
			"agentgateway_tool_runtime_sandbox_operations_total",
			&["operation", "outcome"][..],
			MetricType::Counter,
			"",
		),
		(
			"agentgateway_tool_runtime_sandbox_operation_duration_seconds",
			&["operation"][..],
			MetricType::Histogram,
			"seconds",
		),
		(
			"agentgateway_tool_runtime_limit_exhaustions_total",
			&["limit"][..],
			MetricType::Counter,
			"",
		),
		(
			"agentgateway_tool_runtime_output_truncations_total",
			&["tool"][..],
			MetricType::Counter,
			"",
		),
	];
	let runtime_outcomes = [
		"success",
		"application_error",
		"invalid_request",
		"infrastructure_error",
		"timeout",
		"cancelled",
	];
	let call_outcomes = [
		"queued",
		"executing",
		"success",
		"application_error",
		"infrastructure_error",
		"timeout",
		"cancelled",
	];
	let sandbox_outcomes = ["success", "failure", "timeout", "cancelled"];
	for (name, expected_labels, expected_type, expected_unit) in schemas {
		let family = runtime[name];
		assert_eq!(
			family.r#type, expected_type as i32,
			"{name}; mode: {mode:?}"
		);
		assert_eq!(family.unit, expected_unit, "{name}; mode: {mode:?}");
		assert!(!family.metric.is_empty(), "{name}; mode: {mode:?}");
		for metric in &family.metric {
			let labels = metric_labels(metric);
			assert_eq!(
				metric.label.len(),
				expected_labels.len(),
				"{name}; mode: {mode:?}"
			);
			assert_eq!(
				labels.keys().copied().collect::<BTreeSet<_>>(),
				expected_labels.iter().copied().collect::<BTreeSet<_>>(),
				"{name}; mode: {mode:?}"
			);
			match expected_type {
				MetricType::Counter => {
					assert!(metric.counter.is_some(), "{name}; mode: {mode:?}");
					assert!(metric.gauge.is_none(), "{name}; mode: {mode:?}");
					assert!(metric.summary.is_none(), "{name}; mode: {mode:?}");
					assert!(metric.untyped.is_none(), "{name}; mode: {mode:?}");
					assert!(metric.histogram.is_none(), "{name}; mode: {mode:?}");
				},
				MetricType::Histogram => {
					assert!(metric.counter.is_none(), "{name}; mode: {mode:?}");
					assert!(metric.gauge.is_none(), "{name}; mode: {mode:?}");
					assert!(metric.summary.is_none(), "{name}; mode: {mode:?}");
					assert!(metric.untyped.is_none(), "{name}; mode: {mode:?}");
					let histogram = metric
						.histogram
						.as_ref()
						.expect("histogram family contains histogram samples");
					assert_eq!(
						!histogram.bucket.is_empty(),
						matches!(mode, HistogramMode::Classic | HistogramMode::Both),
						"classic representation for {name}; mode: {mode:?}"
					);
					assert_eq!(
						!histogram.positive_span.is_empty(),
						matches!(mode, HistogramMode::Native | HistogramMode::Both),
						"native representation for {name}; mode: {mode:?}"
					);
				},
				_ => panic!("unexpected runtime metric type for {name}"),
			}

			for (key, value) in labels {
				match key {
					"tool" => assert!(["operator_search", "operator_long", "python"].contains(&value)),
					"backend" => assert!(["http", "e2b"].contains(&value)),
					"operation" => {
						assert!(["create", "execute", "terminate", "cleanup"].contains(&value))
					},
					"limit" => assert!(
						[
							"deadline",
							"rounds",
							"tool_calls",
							"arguments",
							"output",
							"sandbox_batch",
							"sandbox_code",
							"sandbox_call_id",
						]
						.contains(&value)
					),
					"outcome" if name == "agentgateway_tool_runtime_calls_total" => {
						assert!(call_outcomes.contains(&value))
					},
					"outcome" if name == "agentgateway_tool_runtime_sandbox_operations_total" => {
						assert!(sandbox_outcomes.contains(&value))
					},
					"outcome" => assert!(runtime_outcomes.contains(&value)),
					_ => panic!("unexpected label after exact schema validation"),
				}
			}
		}
	}

	let requests = runtime["agentgateway_tool_runtime_requests_total"];
	assert_eq!(
		counter_value(requests, &[("outcome", "success")]),
		Some(1.0)
	);
	assert_eq!(
		counter_value(requests, &[("outcome", "timeout")]),
		Some(1.0)
	);
	assert_eq!(
		counter_value(requests, &[("outcome", "infrastructure_error")]),
		None
	);
	let calls = runtime["agentgateway_tool_runtime_calls_total"];
	for outcome in ["queued", "executing"] {
		assert_eq!(
			counter_value(
				calls,
				&[
					("tool", "operator_search"),
					("backend", "http"),
					("outcome", outcome),
				],
			),
			Some(3.0),
			"every operator_search call records its {outcome} transition"
		);
	}
	assert_eq!(
		counter_value(
			calls,
			&[
				("tool", "operator_search"),
				("backend", "http"),
				("outcome", "application_error"),
			],
		),
		Some(1.0)
	);
	assert_eq!(
		counter_value(
			runtime["agentgateway_tool_runtime_limit_exhaustions_total"],
			&[("limit", "deadline")],
		),
		Some(1.0),
		"the absolute deadline is recorded through the closed limit label"
	);
	assert_eq!(
		counter_value(
			calls,
			&[
				("tool", "python"),
				("backend", "e2b"),
				("outcome", "success"),
			],
		),
		Some(2.0),
		"individual calls remain distinct from one Sandbox batch"
	);
	assert_eq!(
		counter_value(
			calls,
			&[
				("tool", "operator_search"),
				("backend", "http"),
				("outcome", "timeout"),
			],
		),
		Some(1.0)
	);
	let sandbox = runtime["agentgateway_tool_runtime_sandbox_operations_total"];
	assert_eq!(
		counter_value(sandbox, &[("operation", "execute"), ("outcome", "success")],),
		Some(1.0)
	);
	assert_eq!(
		counter_value(sandbox, &[("operation", "cleanup"), ("outcome", "failure")],),
		Some(1.0)
	);
	assert_eq!(
		counter_value(
			runtime["agentgateway_tool_runtime_output_truncations_total"],
			&[("tool", "operator_long")],
		),
		Some(1.0)
	);
}

#[tokio::test(start_paused = true)]
async fn runtime_metrics_are_bounded_and_content_free() {
	for mode in [
		HistogramMode::Classic,
		HistogramMode::Native,
		HistogramMode::Both,
	] {
		let (prometheus, metrics) = runtime_metrics(mode);
		let configured = controlled_registry(
			RuntimeLimits {
				max_output_bytes: 256,
				..runtime_limits(16, 8, Duration::from_secs(30))
			},
			&["operator_search", "operator_long"],
			true,
		);
		let backend: Arc<dyn ToolBackend> = Arc::new(RuntimeMetricsBackend);
		let mut completed = controlled_budget_with_metrics(
			&configured,
			[
				("operator_search", backend.clone()),
				("operator_long", backend.clone()),
				(super::CODE_INTERPRETER_FUNCTION, backend.clone()),
			],
			metrics.clone(),
		);
		completed.set_request_id(Some(SENSITIVE_REQUEST_ID.into()));
		completed.start_model_round().unwrap();
		let outputs = execute_batch(
			&configured,
			vec![
				managed_call(
					"operator_search",
					"call_success_sensitive",
					json!({"query": SENSITIVE_QUERY}),
				),
				managed_call(
					"operator_search",
					"call_application_sensitive",
					json!({"query": SENSITIVE_QUERY}),
				),
				managed_call(
					"operator_long",
					"call_truncation_sensitive",
					json!({"query": SENSITIVE_QUERY}),
				),
				managed_call(
					super::CODE_INTERPRETER_FUNCTION,
					"call_sandbox_one_sensitive",
					json!({
						"code": SENSITIVE_CODE,
						"target": SENSITIVE_SANDBOX_TARGET,
						"template": SENSITIVE_SANDBOX_TEMPLATE,
						"sandbox_id": SENSITIVE_SANDBOX_ID,
					}),
				),
				managed_call(
					super::CODE_INTERPRETER_FUNCTION,
					"call_sandbox_two_sensitive",
					json!({"code": SENSITIVE_CODE}),
				),
			],
			true,
			&mut completed,
		)
		.await
		.unwrap();
		assert_eq!(outputs.len(), 5);
		assert!(output_payload(&outputs[2])["truncated"].as_bool().unwrap());
		completed.record_model_round(ToolRuntimeOutcome::Success, Duration::from_millis(7));
		completed.finish_request(ToolRuntimeOutcome::Success);
		completed.finish_request(ToolRuntimeOutcome::InfrastructureError);
		completed.record_sandbox_operation(SandboxOperation::Cleanup, SandboxOperationOutcome::Failure);

		let timed = controlled_registry(
			runtime_limits(1, 1, Duration::from_secs(5)),
			&["operator_search"],
			false,
		);
		let mut timed_out =
			controlled_budget_with_metrics(&timed, [("operator_search", backend)], metrics);
		timed_out.set_request_id(Some(SENSITIVE_REQUEST_ID.into()));
		timed_out.start_model_round().unwrap();
		let error = execute_batch(
			&timed,
			vec![managed_call(
				"operator_search",
				"call_timeout_sensitive",
				json!({"query": SENSITIVE_QUERY}),
			)],
			true,
			&mut timed_out,
		)
		.await
		.expect_err("absolute deadline expires the pending call");
		assert_eq!(
			error.infrastructure_error(),
			Some(ToolInfrastructureError::Timeout)
		);
		timed_out.record_model_round(ToolRuntimeOutcome::Timeout, Duration::from_secs(5));
		timed_out.finish_request(ToolRuntimeOutcome::Timeout);

		let families =
			prometheus_protobuf::encode(&prometheus).expect("runtime metrics protobuf encoding succeeds");
		assert_runtime_registry_schema(mode, &families);
		let encoded = prometheus_protobuf::encode_to_vec(&prometheus)
			.expect("runtime metrics protobuf wire encoding succeeds");
		for (category, sensitive) in [
			("query", SENSITIVE_QUERY),
			("code", SENSITIVE_CODE),
			("stdout", SENSITIVE_STDOUT),
			("stderr", SENSITIVE_STDERR),
			("result URL", SENSITIVE_RESULT_URL),
			("endpoint IP", SENSITIVE_ENDPOINT_IP),
			("Sandbox endpoint", SENSITIVE_SANDBOX_ENDPOINT),
			("request ID", SENSITIVE_REQUEST_ID),
			("Sandbox target", SENSITIVE_SANDBOX_TARGET),
			("Sandbox template", SENSITIVE_SANDBOX_TEMPLATE),
			("Sandbox ID", SENSITIVE_SANDBOX_ID),
			("call ID", "call_success_sensitive"),
			("call ID", "call_application_sensitive"),
			("call ID", "call_truncation_sensitive"),
			("call ID", "call_sandbox_one_sensitive"),
			("call ID", "call_sandbox_two_sensitive"),
			("call ID", "call_timeout_sensitive"),
			("credential", "operator-secret"),
		] {
			assert!(
				!encoded
					.windows(sensitive.len())
					.any(|window| window == sensitive.as_bytes()),
				"runtime metrics leaked {category}; mode: {mode:?}"
			);
		}
	}
}

#[tokio::test(start_paused = true)]
async fn runtime_budget_binds_configured_backends_once_with_one_absolute_deadline() {
	let registry = controlled_registry(
		runtime_limits(16, 4, Duration::from_secs(30)),
		&["tool"],
		true,
	);
	let budget = RuntimeBudget::new(&registry, policy_client()).expect("valid bound backends");

	assert_eq!(budget.tool_calls(), 0);
	assert_eq!(budget.remaining(), Duration::from_secs(30));
}

#[tokio::test(start_paused = true)]
async fn execute_batch_bounds_parallel_backend_operations() {
	let registry = controlled_registry(
		runtime_limits(16, 2, Duration::from_secs(30)),
		&["tool"],
		false,
	);
	let state = Arc::new(ControlledState::default());
	let backend: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
		state.clone(),
		[
			(
				"one",
				ControlledOutcome::SuccessAfter(Duration::from_secs(1)),
			),
			(
				"two",
				ControlledOutcome::SuccessAfter(Duration::from_secs(1)),
			),
			(
				"three",
				ControlledOutcome::SuccessAfter(Duration::from_secs(1)),
			),
			(
				"four",
				ControlledOutcome::SuccessAfter(Duration::from_secs(1)),
			),
		],
	));
	let mut budget = controlled_budget(&registry, [("tool", backend)]);

	let outputs = execute_batch(
		&registry,
		["one", "two", "three", "four"]
			.into_iter()
			.map(|call_id| batch_call("tool", call_id))
			.collect(),
		true,
		&mut budget,
	)
	.await
	.unwrap();

	assert_eq!(outputs.len(), 4);
	assert_eq!(state.max_active.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn execute_batch_sequential_mode_never_overlaps() {
	let registry = controlled_registry(
		runtime_limits(16, 4, Duration::from_secs(30)),
		&["tool"],
		false,
	);
	let state = Arc::new(ControlledState::default());
	let backend: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
		state.clone(),
		[
			(
				"one",
				ControlledOutcome::SuccessAfter(Duration::from_secs(1)),
			),
			(
				"two",
				ControlledOutcome::SuccessAfter(Duration::from_secs(1)),
			),
			(
				"three",
				ControlledOutcome::SuccessAfter(Duration::from_secs(1)),
			),
		],
	));
	let mut budget = controlled_budget(&registry, [("tool", backend)]);

	execute_batch(
		&registry,
		["one", "two", "three"]
			.into_iter()
			.map(|call_id| batch_call("tool", call_id))
			.collect(),
		false,
		&mut budget,
	)
	.await
	.unwrap();

	assert_eq!(state.max_active.load(Ordering::SeqCst), 1);
	assert_eq!(*state.started.lock().unwrap(), vec!["one", "two", "three"]);
}

#[tokio::test(start_paused = true)]
async fn execute_batch_restores_call_order_and_call_id_after_out_of_order_completion() {
	let registry = controlled_registry(
		runtime_limits(16, 3, Duration::from_secs(30)),
		&["tool"],
		false,
	);
	let state = Arc::new(ControlledState::default());
	let backend: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
		state.clone(),
		[
			(
				"slow",
				ControlledOutcome::SuccessAfter(Duration::from_secs(3)),
			),
			(
				"fast",
				ControlledOutcome::SuccessAfter(Duration::from_secs(1)),
			),
			(
				"middle",
				ControlledOutcome::SuccessAfter(Duration::from_secs(2)),
			),
		],
	));
	let mut budget = controlled_budget(&registry, [("tool", backend)]);

	let outputs = execute_batch(
		&registry,
		["slow", "fast", "middle"]
			.into_iter()
			.map(|call_id| batch_call("tool", call_id))
			.collect(),
		true,
		&mut budget,
	)
	.await
	.unwrap();

	assert_eq!(
		state.completed.lock().unwrap().as_slice(),
		["fast", "middle", "slow"]
	);
	assert_eq!(
		outputs
			.iter()
			.map(|output| output["call_id"].as_str().unwrap())
			.collect::<Vec<_>>(),
		["slow", "fast", "middle"]
	);
	assert_eq!(output_payload(&outputs[0])["call"], "slow");
	assert_eq!(output_payload(&outputs[1])["call"], "fast");
	assert_eq!(output_payload(&outputs[2])["call"], "middle");
}

#[tokio::test(start_paused = true)]
async fn application_error_does_not_cancel_sibling_calls() {
	let registry = controlled_registry(
		runtime_limits(16, 2, Duration::from_secs(30)),
		&["tool"],
		false,
	);
	let state = Arc::new(ControlledState::default());
	let backend: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
		state.clone(),
		[
			(
				"application",
				ControlledOutcome::ApplicationErrorAfter(Duration::from_secs(1)),
			),
			(
				"success",
				ControlledOutcome::SuccessAfter(Duration::from_secs(2)),
			),
		],
	));
	let mut budget = controlled_budget(&registry, [("tool", backend)]);

	let outputs = execute_batch(
		&registry,
		vec![
			batch_call("tool", "application"),
			batch_call("tool", "success"),
		],
		true,
		&mut budget,
	)
	.await
	.unwrap();

	assert!(!output_payload(&outputs[0])["ok"].as_bool().unwrap());
	assert_eq!(output_payload(&outputs[1])["call"], "success");
	assert!(state.cancelled.lock().unwrap().is_empty());
}

#[tokio::test(start_paused = true)]
async fn output_normalization_failure_is_not_recorded_as_success_or_application_error() {
	for (call_id, outcome) in [
		("success", ControlledOutcome::SuccessAfter(Duration::ZERO)),
		(
			"application",
			ControlledOutcome::ApplicationErrorAfter(Duration::ZERO),
		),
	] {
		let mut limits = runtime_limits(1, 1, Duration::from_secs(30));
		limits.max_output_bytes = 1;
		let registry = controlled_registry(limits, &["tool"], false);
		let state = Arc::new(ControlledState::default());
		let backend: Arc<dyn ToolBackend> =
			Arc::new(ControlledBackend::new(state.clone(), [(call_id, outcome)]));
		let mut budget = controlled_budget(&registry, [("tool", backend)]);

		let error = execute_batch(
			&registry,
			vec![batch_call("tool", call_id)],
			true,
			&mut budget,
		)
		.await
		.expect_err("unrepresentable output must fail normalization");

		assert!(error.to_string().contains("output exceeds"), "{call_id}");
		assert_eq!(state.completed.lock().unwrap().as_slice(), [call_id]);
		assert_eq!(budget.execution_records().len(), 1);
		assert_eq!(
			budget.execution_records()[0].outcome,
			ToolExecutionOutcome::Cancelled,
			"{call_id} must not retain a pre-normalization outcome"
		);
	}
}

#[tokio::test(start_paused = true)]
async fn infrastructure_error_cancels_in_flight_and_never_launches_queued_calls() {
	let registry = controlled_registry(
		runtime_limits(16, 2, Duration::from_secs(30)),
		&["tool"],
		false,
	);
	let state = Arc::new(ControlledState::default());
	let backend: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
		state.clone(),
		[
			("fail", ControlledOutcome::InfrastructureAfterStarts(2)),
			("in_flight", ControlledOutcome::Pending),
			("queued", ControlledOutcome::SuccessAfter(Duration::ZERO)),
		],
	));
	let mut budget = controlled_budget(&registry, [("tool", backend)]);

	let error = execute_batch(
		&registry,
		vec![
			batch_call("tool", "fail"),
			batch_call("tool", "in_flight"),
			batch_call("tool", "queued"),
		],
		true,
		&mut budget,
	)
	.await
	.expect_err("infrastructure failure must fail the request");

	assert_eq!(
		error.infrastructure_error(),
		Some(ToolInfrastructureError::Backend)
	);
	assert_eq!(
		*state.started.lock().unwrap(),
		vec!["fail", "in_flight"],
		"queued work must not start after the infrastructure failure"
	);
	assert_eq!(
		*state.cancelled.lock().unwrap(),
		vec!["in_flight"],
		"already-running siblings are dropped"
	);
}

#[tokio::test(start_paused = true)]
async fn max_tool_calls_counts_individual_sandbox_calls_not_batch_operations() {
	let registry = controlled_registry(
		runtime_limits(2, 2, Duration::from_secs(30)),
		&["tool"],
		true,
	);
	let sandbox_state = Arc::new(ControlledState::default());
	let ordinary_state = Arc::new(ControlledState::default());
	let sandbox: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
		sandbox_state.clone(),
		std::iter::empty(),
	));
	let ordinary: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
		ordinary_state.clone(),
		std::iter::empty(),
	));
	let mut budget = controlled_budget(
		&registry,
		[
			(super::CODE_INTERPRETER_FUNCTION, sandbox),
			("tool", ordinary),
		],
	);

	execute_batch(
		&registry,
		vec![code_call("one"), code_call("two")],
		true,
		&mut budget,
	)
	.await
	.unwrap();
	let error = execute_batch(
		&registry,
		vec![batch_call("tool", "three")],
		true,
		&mut budget,
	)
	.await
	.expect_err("third model call exceeds the request budget");

	assert!(error.to_string().contains("tool call limit"));
	assert!(ordinary_state.started.lock().unwrap().is_empty());
	assert_eq!(sandbox_state.batches.lock().unwrap().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn total_deadline_is_absolute_across_execute_batch_rounds() {
	let registry = controlled_registry(
		runtime_limits(16, 1, Duration::from_secs(5)),
		&["tool"],
		false,
	);
	let state = Arc::new(ControlledState::default());
	let backend: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
		state.clone(),
		[
			(
				"first",
				ControlledOutcome::SuccessAfter(Duration::from_secs(3)),
			),
			(
				"second",
				ControlledOutcome::SuccessAfter(Duration::from_secs(3)),
			),
		],
	));
	let mut budget = controlled_budget(&registry, [("tool", backend)]);
	let started = tokio::time::Instant::now();

	execute_batch(
		&registry,
		vec![batch_call("tool", "first")],
		true,
		&mut budget,
	)
	.await
	.unwrap();
	let error = execute_batch(
		&registry,
		vec![batch_call("tool", "second")],
		true,
		&mut budget,
	)
	.await
	.expect_err("second round only has the original deadline remaining");

	assert_eq!(
		error.infrastructure_error(),
		Some(ToolInfrastructureError::Timeout)
	);
	assert_eq!(
		tokio::time::Instant::now() - started,
		Duration::from_secs(5)
	);
	assert_eq!(state.cancelled.lock().unwrap().as_slice(), ["second"]);
	assert_eq!(budget.execution_records().len(), 2);
	assert_eq!(
		budget.execution_records()[1].outcome,
		ToolExecutionOutcome::Timeout
	);
	assert_eq!(budget.execution_records()[1].tool.as_str(), "tool");
	assert!(
		!format!("{:?}", budget.execution_records()[1]).contains("second"),
		"bounded telemetry must exclude call_id"
	);
}

#[tokio::test(start_paused = true)]
async fn sandbox_grouping_obeys_parallel_flag_and_keeps_outputs_in_original_order() {
	for (parallel, expected_batches) in [
		(true, vec![vec!["code_one", "code_two"]]),
		(false, vec![vec!["code_one"], vec!["code_two"]]),
	] {
		let registry = controlled_registry(
			runtime_limits(16, 4, Duration::from_secs(30)),
			&["tool"],
			true,
		);
		let sandbox_state = Arc::new(ControlledState::default());
		let sandbox: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
			sandbox_state.clone(),
			[
				(
					"code_one",
					ControlledOutcome::SuccessAfter(Duration::from_secs(1)),
				),
				(
					"code_two",
					ControlledOutcome::SuccessAfter(Duration::from_secs(1)),
				),
			],
		));
		let ordinary: Arc<dyn ToolBackend> = Arc::new(ControlledBackend::new(
			sandbox_state.clone(),
			[(
				"ordinary",
				ControlledOutcome::SuccessAfter(Duration::from_secs(2)),
			)],
		));
		let mut budget = controlled_budget(
			&registry,
			[
				(super::CODE_INTERPRETER_FUNCTION, sandbox),
				("tool", ordinary),
			],
		);

		let outputs = execute_batch(
			&registry,
			vec![
				code_call("code_one"),
				batch_call("tool", "ordinary"),
				code_call("code_two"),
			],
			parallel,
			&mut budget,
		)
		.await
		.unwrap();

		assert_eq!(
			*sandbox_state.batches.lock().unwrap(),
			expected_batches
				.into_iter()
				.map(|batch| batch.into_iter().map(str::to_owned).collect())
				.collect::<Vec<Vec<String>>>(),
			"parallel={parallel}"
		);
		assert_eq!(
			outputs
				.iter()
				.map(|output| output["call_id"].as_str().unwrap())
				.collect::<Vec<_>>(),
			["code_one", "ordinary", "code_two"],
			"parallel={parallel}"
		);
		assert_eq!(
			sandbox_state.max_active.load(Ordering::SeqCst),
			if parallel { 2 } else { 1 },
			"the Sandbox batch counts as one operation under the shared limit"
		);
	}
}
