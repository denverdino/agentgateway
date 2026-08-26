use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use ::http::Method;
use bytes::Bytes;
use http_body::Frame;
use http_body_util::StreamBody;
use secrecy::SecretString;
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;
use wiremock::{Mock, Respond, ResponseTemplate};

use crate::test_helpers::proxymock;
use crate::types::agent::{Backend, BackendTrafficPolicy, ResourceName, Target};
use crate::{http, llm};

#[derive(Clone)]
struct ResponseSequence {
	responses: Arc<Vec<serde_json::Value>>,
	next: Arc<AtomicUsize>,
}

impl Respond for ResponseSequence {
	fn respond(&self, _request: &wiremock::Request) -> ResponseTemplate {
		let index = self.next.fetch_add(1, Ordering::SeqCst);
		let response = self
			.responses
			.get(index)
			.or_else(|| self.responses.last())
			.expect("response sequence must not be empty")
			.clone();
		ResponseTemplate::new(200).set_body_json(response)
	}
}

#[derive(Clone)]
struct StatusResponseSequence {
	responses: Arc<Vec<(u16, serde_json::Value)>>,
	next: Arc<AtomicUsize>,
}

impl Respond for StatusResponseSequence {
	fn respond(&self, _request: &wiremock::Request) -> ResponseTemplate {
		let index = self.next.fetch_add(1, Ordering::SeqCst);
		let (status, response) = self
			.responses
			.get(index)
			.or_else(|| self.responses.last())
			.expect("response sequence must not be empty");
		ResponseTemplate::new(*status).set_body_json(response.clone())
	}
}

struct StalledBodyServer {
	address: SocketAddr,
	body_dropped: Arc<tokio::sync::Notify>,
	requests: Arc<AtomicUsize>,
	handle: tokio::task::JoinHandle<()>,
}

impl StalledBodyServer {
	async fn start() -> Self {
		let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
		let address = listener.local_addr().unwrap();
		let body_dropped = Arc::new(tokio::sync::Notify::new());
		let requests = Arc::new(AtomicUsize::new(0));
		let server_body_dropped = body_dropped.clone();
		let server_requests = requests.clone();
		let handle = tokio::spawn(async move {
			while let Ok((socket, _)) = listener.accept().await {
				let body_dropped = server_body_dropped.clone();
				let requests = server_requests.clone();
				tokio::spawn(async move {
					let service = hyper::service::service_fn(move |_request| {
						let body_dropped = body_dropped.clone();
						let requests = requests.clone();
						async move {
							requests.fetch_add(1, Ordering::SeqCst);
							let (sender, receiver) =
								tokio::sync::mpsc::channel::<Result<Frame<Bytes>, crate::http::Error>>(1);
							sender
								.send(Ok(Frame::data(Bytes::from_static(b"{\"id\":"))))
								.await
								.unwrap();
							tokio::spawn(async move {
								sender.closed().await;
								body_dropped.notify_one();
							});
							Ok::<_, Infallible>(
								::http::Response::builder()
									.status(200)
									.header(::http::header::CONTENT_TYPE, "application/json")
									.body(http::Body::new(StreamBody::new(ReceiverStream::new(
										receiver,
									))))
									.unwrap(),
							)
						}
					});
					let _ = hyper::server::conn::http1::Builder::new()
						.serve_connection(hyper_util::rt::TokioIo::new(socket), service)
						.await;
				});
			}
		});
		Self {
			address,
			body_dropped,
			requests,
			handle,
		}
	}
}

impl Drop for StalledBodyServer {
	fn drop(&mut self) {
		self.handle.abort();
	}
}

fn responses_body(
	output: serde_json::Value,
	input_tokens: u64,
	output_tokens: u64,
) -> serde_json::Value {
	json!({
		"id": format!("resp_{input_tokens}_{output_tokens}"),
		"object": "response",
		"created_at": 1,
		"status": "completed",
		"model": "gpt-test",
		"output": output,
		"usage": {
			"input_tokens": input_tokens,
			"output_tokens": output_tokens,
			"total_tokens": input_tokens + output_tokens
		}
	})
}

fn function_call(name: &str, call_id: &str, arguments: &str) -> serde_json::Value {
	json!({
		"type": "function_call",
		"id": format!("fc_{call_id}"),
		"call_id": call_id,
		"name": name,
		"arguments": arguments,
		"status": "completed"
	})
}

fn final_message(text: &str) -> serde_json::Value {
	json!({
		"type": "message",
		"id": "msg_final",
		"role": "assistant",
		"status": "completed",
		"content": [{
			"type": "output_text",
			"text": text,
			"annotations": []
		}]
	})
}

fn responses_request(tool_name: &str) -> Vec<u8> {
	serde_json::to_vec(&json!({
		"model": "gpt-test",
		"input": "weather",
		"tools": [{
			"type": "function",
			"name": tool_name,
			"parameters": {"type": "object"}
		}]
	}))
	.unwrap()
}

async fn send_responses_request(
	io: hyper_util::client::legacy::Client<proxymock::MemoryConnector, http::Body>,
	tool_name: &str,
) -> http::Response {
	proxymock::send_request_body(
		io,
		Method::POST,
		"http://lo/v1/responses",
		&responses_request(tool_name),
	)
	.await
}

async fn response_json(response: http::Response) -> serde_json::Value {
	serde_json::from_slice(&proxymock::read_body_raw(response.into_body()).await).unwrap()
}

fn managed_tool_backend(
	model: &wiremock::MockServer,
	tool: &wiremock::MockServer,
	limits: llm::tool_runtime::RuntimeLimits,
) -> crate::types::agent::BackendWithPolicies {
	managed_tool_backend_for_models(&[("mock-openai", model)], tool, limits)
}

fn managed_tool_backend_for_models(
	models: &[(&str, &wiremock::MockServer)],
	tool: &wiremock::MockServer,
	limits: llm::tool_runtime::RuntimeLimits,
) -> crate::types::agent::BackendWithPolicies {
	let models = models
		.iter()
		.map(|(name, model)| (*name, *model.address()))
		.collect::<Vec<_>>();
	managed_tool_backend_for_addresses(&models, tool, limits)
}

fn managed_tool_backend_for_addresses(
	models: &[(&str, SocketAddr)],
	tool: &wiremock::MockServer,
	limits: llm::tool_runtime::RuntimeLimits,
) -> crate::types::agent::BackendWithPolicies {
	let runtime = llm::tool_runtime::ToolRuntimeConfig {
		limits,
		tools: vec![llm::tool_runtime::ManagedToolConfig {
			name: "managed".to_owned(),
			builtin: None,
			backend: llm::tool_runtime::ToolBackendConfig::Http {
				url: format!("http://{}/invoke", tool.address()).parse().unwrap(),
				timeout: Duration::from_secs(5),
				bearer_token: Some(SecretString::from("operator-secret")),
			},
		}],
	};
	let runtime = Arc::new(
		llm::tool_runtime::ToolRegistry::compile(runtime).expect("valid managed tool test runtime"),
	);
	let providers = models
		.iter()
		.map(|(name, model)| {
			let mut policy = Arc::unwrap_or_clone(llm::model_router::default_route_types());
			policy.tool_runtime = Some(runtime.clone());
			policy.final_transformations = Some(HashMap::from([(
				"operator_round_marker".to_owned(),
				Arc::new(crate::cel::Expression::new_strict("true").unwrap()),
			)]));
			let provider_name = agent_core::strng::new(*name);
			let provider = llm::NamedAIProvider {
				name: provider_name.clone(),
				provider: llm::AIProvider::OpenAI(llm::openai::Provider {
					model: None,
					moderation: None,
				}),
				provider_backend: None,
				host_override: Some(Target::Address(*model)),
				path_override: None,
				path_prefix: None,
				tokenize: false,
				inline_policies: vec![
					BackendTrafficPolicy::backend_auth(crate::http::auth::BackendAuthKind::Key {
						value: SecretString::from("model-backend-secret"),
						location: None,
					}),
					BackendTrafficPolicy::AI(Arc::new(policy)),
				],
			};
			(provider_name, provider)
		})
		.collect();
	Backend::AI(
		ResourceName::new("managed-llm".into(), "".into()),
		llm::AIBackend {
			providers: crate::types::loadbalancer::EndpointSet::new(vec![providers]),
		},
	)
	.into()
}

fn runtime_limits() -> llm::tool_runtime::RuntimeLimits {
	llm::tool_runtime::RuntimeLimits {
		total_timeout: Duration::from_secs(30),
		max_rounds: 4,
		max_tool_calls: 4,
		max_parallel_tool_calls: 2,
		max_arguments_bytes: 1024,
		max_output_bytes: 4096,
	}
}

async fn managed_tool_proxy(
	model: &wiremock::MockServer,
	tool: &wiremock::MockServer,
	limits: llm::tool_runtime::RuntimeLimits,
) -> (
	proxymock::TestBind,
	hyper_util::client::legacy::Client<proxymock::MemoryConnector, http::Body>,
) {
	let bind = proxymock::setup_proxy_test("{}")
		.expect("proxy test harness")
		.with_raw_backend(managed_tool_backend(model, tool, limits))
		.with_bind(proxymock::simple_bind())
		.with_route(proxymock::basic_named_route("/managed-llm".into()));
	let io = bind.serve_http(proxymock::BIND_KEY);
	(bind, io)
}

#[tokio::test]
async fn responses_managed_tool_call_loops_and_returns_only_final_answer() {
	let model = wiremock::MockServer::start().await;
	Mock::given(wiremock::matchers::method("POST"))
		.respond_with(ResponseSequence {
			responses: Arc::new(vec![
				responses_body(
					json!([function_call(
						"managed",
						"call_1",
						"{\"city\":\"Hangzhou\"}",
					)]),
					10,
					3,
				),
				responses_body(json!([final_message("It is 22 C.")]), 14, 4),
			]),
			next: Arc::new(AtomicUsize::new(0)),
		})
		.mount(&model)
		.await;
	let tool = wiremock::MockServer::start().await;
	Mock::given(wiremock::matchers::method("POST"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({"temperature_c": 22})))
		.mount(&tool)
		.await;
	let (_bind, io) = managed_tool_proxy(&model, &tool, runtime_limits()).await;

	let response = proxymock::send_request_body(
		io,
		Method::POST,
		"http://lo/v1/responses",
		br#"{
			"model":"gpt-test",
			"input":"weather",
			"tools":[{
				"type":"function",
				"name":"managed",
				"parameters":{"type":"object"}
			}]
		}"#,
	)
	.await;

	assert_eq!(response.status(), 200);
	let body: serde_json::Value =
		serde_json::from_slice(&proxymock::read_body_raw(response.into_body()).await).unwrap();
	assert_eq!(body["output"].as_array().unwrap().len(), 1);
	assert_eq!(body["output"][0]["type"], "message");
	assert_eq!(body["output"][0]["content"][0]["text"], "It is 22 C.");
	assert_eq!(body["usage"]["input_tokens"], 24);
	assert_eq!(body["usage"]["output_tokens"], 7);
	assert_eq!(body["usage"]["total_tokens"], 31);

	let model_requests = model.received_requests().await.unwrap();
	assert_eq!(model_requests.len(), 2);
	assert!(model_requests.iter().all(|request| {
		request
			.headers
			.get("authorization")
			.and_then(|value| value.to_str().ok())
			== Some("Bearer model-backend-secret")
	}));
	assert!(model_requests.iter().all(|request| {
		serde_json::from_slice::<serde_json::Value>(&request.body).unwrap()["operator_round_marker"]
			== true
	}));
	let second: serde_json::Value = serde_json::from_slice(&model_requests[1].body).unwrap();
	let input = second["input"].as_array().unwrap();
	assert_eq!(input[1]["type"], "function_call");
	assert_eq!(input[1]["call_id"], "call_1");
	assert_eq!(input[2]["type"], "function_call_output");
	assert_eq!(input[2]["call_id"], "call_1");
	assert_eq!(tool.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn responses_managed_tool_runtime_streams_only_the_final_answer() {
	let model = wiremock::MockServer::start().await;
	let mut final_response = responses_body(
		json!([
			{
				"type": "reasoning",
				"id": "reasoning_final",
				"summary": [{"type": "summary_text", "text": "private reasoning"}]
			},
			final_message("streamed final")
		]),
		3,
		4,
	);
	final_response["tools"] = json!([{
		"type": "function",
		"name": "_agentgateway_internal_echo",
		"parameters": {"type": "object"}
	}]);
	Mock::given(wiremock::matchers::method("POST"))
		.respond_with(ResponseSequence {
			responses: Arc::new(vec![
				responses_body(json!([function_call("managed", "call_1", "{}")]), 2, 1),
				final_response,
			]),
			next: Arc::new(AtomicUsize::new(0)),
		})
		.mount(&model)
		.await;
	let tool = wiremock::MockServer::start().await;
	Mock::given(wiremock::matchers::method("POST"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
		.mount(&tool)
		.await;
	let (_bind, io) = managed_tool_proxy(&model, &tool, runtime_limits()).await;
	let request = serde_json::to_vec(&json!({
		"model": "gpt-test",
		"input": "weather",
		"stream": true,
		"tools": [{
			"type": "function",
			"name": "managed",
			"parameters": {"type": "object"}
		}]
	}))
	.unwrap();

	let response =
		proxymock::send_request_body(io, Method::POST, "http://lo/v1/responses", &request).await;

	assert_eq!(response.status(), 200);
	assert_eq!(
		response
			.headers()
			.get(::http::header::CONTENT_TYPE)
			.unwrap(),
		"text/event-stream"
	);
	let body = String::from_utf8(
		proxymock::read_body_raw(response.into_body())
			.await
			.to_vec(),
	)
	.unwrap();
	assert!(body.contains("event: response.created\n"), "{body}");
	assert!(
		body.contains("event: response.output_text.delta\n"),
		"{body}"
	);
	assert!(body.contains("streamed final"), "{body}");
	assert!(body.contains("event: response.completed\n"), "{body}");
	assert!(!body.contains("_agentgateway_"), "{body}");
	for data in body
		.split("\n\n")
		.filter_map(|frame| frame.lines().find_map(|line| line.strip_prefix("data: ")))
	{
		serde_json::from_str::<llm::types::responses::typed::ResponseStreamEvent>(data)
			.unwrap_or_else(|error| panic!("invalid Responses SSE event: {error}; data={data}"));
	}
	let completed = body
		.split("\n\n")
		.find_map(|frame| {
			frame
				.lines()
				.find_map(|line| line.strip_prefix("data: "))
				.and_then(|data| serde_json::from_str::<serde_json::Value>(data).ok())
				.filter(|event| event["type"] == "response.completed")
		})
		.expect("response.completed event");
	let text_deltas = body
		.split("\n\n")
		.filter_map(|frame| frame.lines().find_map(|line| line.strip_prefix("data: ")))
		.filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
		.filter(|event| event["type"] == "response.output_text.delta")
		.filter_map(|event| event["delta"].as_str().map(str::to_owned))
		.collect::<Vec<_>>();
	assert_eq!(text_deltas, vec!["streamed final"]);
	let reasoning_added = body
		.split("\n\n")
		.filter_map(|frame| frame.lines().find_map(|line| line.strip_prefix("data: ")))
		.filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
		.find(|event| {
			event["type"] == "response.output_item.added" && event["item"]["id"] == "reasoning_final"
		})
		.expect("reasoning item added event");
	assert_eq!(reasoning_added["item"]["summary"], json!([]));
	assert_eq!(completed["response"]["usage"]["input_tokens"], 5);
	assert_eq!(completed["response"]["usage"]["output_tokens"], 5);
	assert_eq!(completed["response"]["usage"]["total_tokens"], 10);
	assert_eq!(completed["response"]["tools"][0]["name"], "managed");

	let model_requests = model.received_requests().await.unwrap();
	assert_eq!(model_requests.len(), 2);
	assert!(model_requests.iter().all(|request| {
		serde_json::from_slice::<serde_json::Value>(&request.body).unwrap()["stream"] == false
	}));
	assert_eq!(tool.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn responses_managed_tool_runtime_pins_initial_provider_for_every_round() {
	let model_a = wiremock::MockServer::start().await;
	let model_b = wiremock::MockServer::start().await;
	for model in [&model_a, &model_b] {
		Mock::given(wiremock::matchers::method("POST"))
			.respond_with(ResponseSequence {
				responses: Arc::new(vec![
					responses_body(json!([function_call("managed", "call_1", "{}")]), 1, 1),
					responses_body(json!([final_message("pinned")]), 1, 1),
				]),
				next: Arc::new(AtomicUsize::new(0)),
			})
			.mount(model)
			.await;
	}
	let tool = wiremock::MockServer::start().await;
	Mock::given(wiremock::matchers::method("POST"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
		.mount(&tool)
		.await;
	let bind = proxymock::setup_proxy_test("{}")
		.expect("proxy test harness")
		.with_raw_backend(managed_tool_backend_for_models(
			&[("model-a", &model_a), ("model-b", &model_b)],
			&tool,
			runtime_limits(),
		))
		.with_bind(proxymock::simple_bind())
		.with_route(proxymock::basic_named_route("/managed-llm".into()));
	let io = bind.serve_http(proxymock::BIND_KEY);

	let response = send_responses_request(io, "managed").await;
	assert_eq!(response.status(), 200);
	let body = response_json(response).await;
	assert_eq!(body["output"][0]["content"][0]["text"], "pinned");
	let model_a_calls = model_a.received_requests().await.unwrap().len();
	let model_b_calls = model_b.received_requests().await.unwrap().len();
	assert!(
		(model_a_calls == 2 && model_b_calls == 0) || (model_a_calls == 0 && model_b_calls == 2),
		"all rounds must stay on one provider; model-a={model_a_calls}, model-b={model_b_calls}"
	);
	assert_eq!(tool.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn responses_client_cannot_override_managed_tool_backend_credentials_or_limits() {
	let model = wiremock::MockServer::start().await;
	Mock::given(wiremock::matchers::method("POST"))
		.respond_with(ResponseSequence {
			responses: Arc::new(vec![
				responses_body(json!([function_call("managed", "call_1", "{}")]), 1, 1),
				responses_body(json!([final_message("trusted configuration won")]), 1, 1),
			]),
			next: Arc::new(AtomicUsize::new(0)),
		})
		.mount(&model)
		.await;
	let operator_tool = wiremock::MockServer::start().await;
	Mock::given(wiremock::matchers::method("POST"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
		.mount(&operator_tool)
		.await;
	let untrusted_tool = wiremock::MockServer::start().await;
	let (_bind, io) = managed_tool_proxy(&model, &operator_tool, runtime_limits()).await;
	let request = serde_json::to_vec(&json!({
		"model": "gpt-test",
		"input": "weather",
		"max_tool_calls": 0,
		"total_timeout_ms": 1,
		"sandbox_template": "client-template",
		"tools": [{
			"type": "function",
			"name": "managed",
			"parameters": {"type": "object"},
			"url": format!("http://{}/invoke", untrusted_tool.address()),
			"bearer_token": "client-secret",
			"sandbox_template": "client-template"
		}]
	}))
	.unwrap();

	let response =
		proxymock::send_request_body(io, Method::POST, "http://lo/v1/responses", &request).await;
	assert_eq!(response.status(), 200);
	let body = response_json(response).await;
	assert_eq!(
		body["output"][0]["content"][0]["text"],
		"trusted configuration won"
	);
	assert_eq!(untrusted_tool.received_requests().await.unwrap().len(), 0);
	let operator_requests = operator_tool.received_requests().await.unwrap();
	assert_eq!(operator_requests.len(), 1);
	assert_eq!(
		operator_requests[0]
			.headers
			.get("authorization")
			.and_then(|value| value.to_str().ok()),
		Some("Bearer operator-secret")
	);
	let tool_request = String::from_utf8_lossy(&operator_requests[0].body);
	assert!(!tool_request.contains("client-secret"));
	assert!(!tool_request.contains("client-template"));
	assert_eq!(model.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn responses_managed_tool_runtime_deadline_covers_body_after_headers() {
	let model = StalledBodyServer::start().await;
	let tool = wiremock::MockServer::start().await;
	let mut limits = runtime_limits();
	limits.total_timeout = Duration::from_millis(40);
	let bind = proxymock::setup_proxy_test("{}")
		.expect("proxy test harness")
		.with_raw_backend(managed_tool_backend_for_addresses(
			&[("stalled-model", model.address)],
			&tool,
			limits,
		))
		.with_bind(proxymock::simple_bind())
		.with_route(proxymock::basic_named_route("/managed-llm".into()));
	let io = bind.serve_http(proxymock::BIND_KEY);

	let response = tokio::time::timeout(
		Duration::from_secs(1),
		send_responses_request(io, "managed"),
	)
	.await
	.expect("the runtime deadline must stop a stalled upstream body");
	assert_eq!(response.status(), 504);
	let body = response_json(response).await;
	assert_eq!(body["error"]["type"], "tool_infrastructure_error");
	assert_eq!(body["error"]["code"], "tool_execution_timeout");
	assert_eq!(model.requests.load(Ordering::SeqCst), 1);
	assert_eq!(tool.received_requests().await.unwrap().len(), 0);
	tokio::time::timeout(Duration::from_secs(1), model.body_dropped.notified())
		.await
		.expect("timed-out response body must be dropped");
}

#[tokio::test]
async fn responses_active_runtime_sanitizes_later_model_error_without_retry() {
	let model = wiremock::MockServer::start().await;
	let call_id = "call_sensitive_1";
	let function_round = responses_body(json!([function_call("managed", call_id, "{}")]), 1, 1);
	Mock::given(wiremock::matchers::method("POST"))
		.respond_with(StatusResponseSequence {
			responses: Arc::new(vec![
				(200, function_round.clone()),
				(
					500,
					json!({
						"error": {
							"type": "server_error",
							"message": "SENSITIVE_MODEL_ECHO",
							"code": "upstream_failed"
						},
						"echo": {
							"stdout": "SENSITIVE_STDOUT",
							"stderr": "SENSITIVE_STDERR",
							"result_url": "https://results.invalid/private",
							"call_id": call_id
						}
					}),
				),
				(200, function_round),
			]),
			next: Arc::new(AtomicUsize::new(0)),
		})
		.mount(&model)
		.await;
	let tool = wiremock::MockServer::start().await;
	Mock::given(wiremock::matchers::method("POST"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({
			"stdout": "SENSITIVE_STDOUT",
			"stderr": "SENSITIVE_STDERR",
			"result_url": "https://results.invalid/private"
		})))
		.mount(&tool)
		.await;
	let mut bind = proxymock::setup_proxy_test("{}")
		.expect("proxy test harness")
		.with_raw_backend(managed_tool_backend(&model, &tool, runtime_limits()))
		.with_bind(proxymock::simple_bind())
		.with_route(proxymock::basic_named_route("/managed-llm".into()));
	bind
		.attach_route_policy(json!({
			"retry": {"attempts": 1, "codes": [500, 502]}
		}))
		.await;
	let io = bind.serve_http(proxymock::BIND_KEY);

	let response = send_responses_request(io, "managed").await;
	assert_eq!(response.status(), 502);
	let body = response_json(response).await;
	assert_eq!(body["error"]["type"], "tool_infrastructure_error");
	assert_eq!(body["error"]["code"], "tool_runtime_internal_error");
	let serialized = body.to_string();
	for sensitive in [
		"SENSITIVE_MODEL_ECHO",
		"SENSITIVE_STDOUT",
		"SENSITIVE_STDERR",
		"https://results.invalid/private",
		call_id,
	] {
		assert!(!serialized.contains(sensitive));
	}
	assert_eq!(model.received_requests().await.unwrap().len(), 2);
	assert_eq!(tool.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn responses_unmanaged_function_keeps_one_call_passthrough_path() {
	let model = wiremock::MockServer::start().await;
	Mock::given(wiremock::matchers::method("POST"))
		.respond_with(ResponseTemplate::new(200).set_body_json(responses_body(
			json!([function_call("client_only", "call_client", "{\"value\":1}")]),
			7,
			2,
		)))
		.mount(&model)
		.await;
	let tool = wiremock::MockServer::start().await;
	let (_bind, io) = managed_tool_proxy(&model, &tool, runtime_limits()).await;

	let response = send_responses_request(io, "client_only").await;
	assert_eq!(response.status(), 200);
	let body = response_json(response).await;
	assert_eq!(body["output"][0]["type"], "function_call");
	assert_eq!(body["output"][0]["name"], "client_only");
	assert_eq!(body["output"][0]["call_id"], "call_client");
	assert_eq!(body["usage"]["total_tokens"], 9);
	assert_eq!(model.received_requests().await.unwrap().len(), 1);
	assert_eq!(tool.received_requests().await.unwrap().len(), 0);
}
