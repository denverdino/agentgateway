use std::fs;
use std::path::{Path, PathBuf};

use agent_core::strng;
use http_body_util::BodyExt;
use serde_json::{Value, json};

use super::*;
use crate::http::x_headers::TRACEPARENT;

fn llm_request_with_tokens(input_tokens: Option<u64>) -> LLMRequest {
	LLMRequest {
		input_tokens,
		input_format: InputFormat::Completions,
		cache_convention: CacheTokenConvention::pending(),
		request_model: "test-model".into(),
		provider: "test-provider".into(),
		streaming: true,
		params: Default::default(),
		prompt: None,
		provider_state: None,
	}
}

#[test]
fn vertex_gemini_uses_native_completions_and_compat_fallbacks() {
	let provider = AIProvider::Vertex(vertex::Provider {
		project_id: strng::new("test-project"),
		model: None,
		region: None,
	});
	let model = Some("google/gemini-2.5-flash-lite");

	assert_eq!(
		provider
			.chat_translation(InputFormat::Completions, model, None)
			.unwrap()
			.output,
		ChatFormat::VertexGemini
	);
	for input in [InputFormat::Messages, InputFormat::Responses] {
		assert_eq!(
			provider
				.chat_translation(input, model, None)
				.unwrap()
				.output,
			ChatFormat::OpenAICompletions
		);
	}
}

#[test]
fn gemini_inbound_selects_native_translation_only_for_gemini_upstreams() {
	let vertex = AIProvider::Vertex(vertex::Provider {
		project_id: strng::new("test-project"),
		model: None,
		region: None,
	});
	assert_eq!(
		vertex
			.chat_translation(InputFormat::Gemini, Some("gemini-2.5-flash"), None)
			.unwrap()
			.output,
		ChatFormat::VertexGemini
	);
	// Vertex with a non-Gemini model has no Gemini-input translation.
	assert!(
		vertex
			.chat_translation(InputFormat::Gemini, Some("claude-sonnet-4-5"), None)
			.is_err()
	);

	let gemini = AIProvider::Gemini(gemini::Provider { model: None });
	assert_eq!(
		gemini
			.chat_translation(InputFormat::Gemini, Some("gemini-2.5-flash"), None)
			.unwrap()
			.output,
		ChatFormat::VertexGemini
	);
	// Completions inbound on the Gemini API provider prefers our native conversion over the
	// OpenAI-compat shim, matching Vertex with a Gemini model.
	assert_eq!(
		gemini
			.chat_translation(InputFormat::Completions, Some("gemini-2.5-flash"), None)
			.unwrap()
			.output,
		ChatFormat::VertexGemini
	);
	// Messages and Responses clients still ride the compat shim: there is no conversion from
	// those formats to native Gemini.
	for input in [InputFormat::Messages, InputFormat::Responses] {
		assert_eq!(
			gemini
				.chat_translation(input, Some("gemini-2.5-flash"), None)
				.unwrap()
				.output,
			ChatFormat::OpenAICompletions
		);
	}
}

#[test]
fn gemini_inbound_to_non_gemini_upstream_is_unsupported() {
	let anthropic = AIProvider::Anthropic(anthropic::Provider { model: None });
	let Err(err) = anthropic.chat_translation(InputFormat::Gemini, Some("claude-opus-4"), None)
	else {
		panic!("expected unsupported conversion");
	};
	assert!(matches!(err, AIError::UnsupportedConversion(_)));
	let msg = err.to_string();
	assert!(msg.contains("Gemini") && msg.contains("anthropic"), "{msg}");

	let vertex = AIProvider::Vertex(vertex::Provider {
		project_id: strng::new("test-project"),
		model: None,
		region: None,
	});
	let Err(err) = vertex.chat_translation(InputFormat::Gemini, Some("claude-sonnet-4-5"), None)
	else {
		panic!("expected unsupported conversion");
	};
	let msg = err.to_string();
	assert!(msg.contains("Gemini") && msg.contains("vertex"), "{msg}");
}

#[test]
fn custom_provider_generate_content_advertises_the_native_chat_format() {
	let provider = custom_provider(custom::ProviderFormat::GenerateContent);

	// Native Gemini input takes the direct passthrough.
	assert_eq!(
		provider
			.chat_translation(InputFormat::Gemini, Some("gemini-2.5-flash"), None)
			.unwrap()
			.output,
		ChatFormat::VertexGemini
	);
	// Completions input prefers our native conversion over the compat shim, exactly like
	// Vertex with a Gemini model (the CHAT_TRANSLATIONS quirk).
	assert_eq!(
		provider
			.chat_translation(InputFormat::Completions, Some("gemini-2.5-flash"), None)
			.unwrap()
			.output,
		ChatFormat::VertexGemini
	);

	// A custom provider that does not declare the format has no Gemini-input translation.
	let undeclared = custom_provider(custom::ProviderFormat::Completions);
	assert!(
		undeclared
			.chat_translation(InputFormat::Gemini, Some("gemini-2.5-flash"), None)
			.is_err()
	);
}

#[tokio::test]
async fn custom_provider_completions_inbound_renders_native_gemini() {
	// With generateContent declared, the CHAT_TRANSLATIONS quirk applies to custom providers
	// too: OpenAI-compat input converts to the native request, and the conversion must not
	// assume a Vertex provider.
	let provider = custom_provider(custom::ProviderFormat::GenerateContent);
	let req = ::http::Request::builder()
		.uri("/v1/chat/completions")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hello"}]}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: mut forwarded,
		llm_request,
		upstream_route_type,
		..
	} = provider
		.process_completions_request(
			&openai_test_backend_info(),
			None,
			req,
			false,
			&mut None,
			None,
		)
		.await
		.expect("completions request should process")
	else {
		panic!("expected forwarded request");
	};

	assert_eq!(upstream_route_type, RouteType::GenerateContent);
	assert!(matches!(
		llm_request.provider_state,
		Some(ProviderState::VertexGemini)
	));

	provider
		.setup_request(
			&mut forwarded,
			upstream_route_type,
			Some(&llm_request),
			None,
			None,
			false,
		)
		.expect("setup_request should succeed");
	assert_eq!(
		forwarded.uri().path(),
		"/v1beta/models/gemini-2.5-flash:generateContent"
	);

	let body = forwarded.into_body().collect().await.unwrap().to_bytes();
	let json: Value = serde_json::from_slice(&body).expect("forwarded body should be JSON");
	assert!(json.get("contents").is_some(), "{json}");
	assert!(json.get("messages").is_none(), "{json}");
}

#[test]
fn custom_provider_declaring_gemini_count_tokens_renders_passthrough() {
	// countTokens has no cross-provider conversion, so the render gate must accept exactly
	// the providers that speak it natively, including a custom provider declaring it.
	let req: types::gemini::CountTokensRequest =
		serde_json::from_value(json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}))
			.unwrap();

	let provider = custom_provider(custom::ProviderFormat::GeminiCountTokens);
	let body = provider
		.render_gemini_count_tokens_request(&req, "gemini-2.5-flash")
		.expect("declared format renders passthrough");
	let json: Value = serde_json::from_slice(&body).unwrap();
	assert!(json.get("contents").is_some(), "{json}");

	let undeclared = custom_provider(custom::ProviderFormat::Completions);
	assert!(
		undeclared
			.render_gemini_count_tokens_request(&req, "gemini-2.5-flash")
			.is_err()
	);
}

#[test]
fn gemini_render_is_passthrough_with_unknown_fields() {
	let provider = AIProvider::Vertex(vertex::Provider {
		project_id: strng::new("test-project"),
		model: None,
		region: None,
	});
	let translation = provider
		.chat_translation(InputFormat::Gemini, Some("gemini-2.5-flash"), None)
		.unwrap();

	let raw = json!({
		"contents": [{
			"role": "user",
			"parts": [
				{ "text": "describe this", "someNewPartField": 1 },
				{ "inlineData": { "mimeType": "image/png", "data": "AAAA" } }
			],
			"someNewContentField": true
		}],
		"systemInstruction": { "parts": [{ "text": "be brief" }] },
		"tools": [
			{ "functionDeclarations": [{
				"name": "get_weather",
				"parameters": { "type": "object" },
				"behavior": "BLOCKING"
			}] },
			{ "googleSearch": {} }
		],
		"toolConfig": { "functionCallingConfig": { "mode": "AUTO", "someNewKnob": 1 } },
		"generationConfig": {
			"temperature": 0.5,
			"thinkingConfig": { "thinkingLevel": "high", "someNewField": true },
			"responseModalities": ["TEXT"]
		},
		"safetySettings": [{
			"category": "HARM_CATEGORY_HATE_SPEECH",
			"threshold": "BLOCK_NONE",
			"someNewField": 1
		}],
		"modelArmorConfig": { "promptTemplateName": "projects/p/locations/l/templates/t" }
	});
	let inner: types::gemini::GenerateContentRequest =
		serde_json::from_value(raw.clone()).expect("valid request");
	let rendered = translation
		.render_request(
			types::ChatRequest::Gemini(inner),
			&ChatRequestContext {
				provider: &provider,
				headers: &HeaderMap::new(),
				prompt_caching: None,
			},
		)
		.expect("render");
	assert!(matches!(
		rendered.provider_state,
		Some(ProviderState::VertexGemini)
	));
	let out: Value = serde_json::from_slice(&rendered.body).expect("valid body");
	assert_eq!(
		out, raw,
		"render must pass unknown fields through untouched"
	);
}

#[test]
fn gemini_error_passes_google_shape_through() {
	let provider = AIProvider::Vertex(vertex::Provider {
		project_id: strng::new("test-project"),
		model: None,
		region: None,
	});
	let translation = provider
		.chat_translation(InputFormat::Gemini, Some("gemini-2.5-flash"), None)
		.unwrap();
	assert!(matches!(
		provider.chat_error_format(translation, Some("gemini-2.5-flash")),
		ChatErrorFormat::Google
	));

	let body = bytes::Bytes::from_static(
		br#"{"error":{"code":400,"message":"bad request","status":"INVALID_ARGUMENT"}}"#,
	);
	let out = translation
		.error(
			&body,
			::http::StatusCode::BAD_REQUEST,
			ChatErrorFormat::Google,
		)
		.expect("error translation");
	assert_eq!(out, body);
}

#[test]
fn strip_alt_query_removes_only_alt() {
	let mut req = crate::http::tests_common::request(
		"https://example.com/v1beta/models/m:streamGenerateContent?alt=sse&key=abc",
		http::Method::POST,
		&[],
	);
	strip_alt_query(&mut req);
	assert_eq!(req.uri().query(), Some("key=abc"));

	let mut req = crate::http::tests_common::request(
		"https://example.com/v1beta/models/m:streamGenerateContent?alt=sse",
		http::Method::POST,
		&[],
	);
	strip_alt_query(&mut req);
	assert_eq!(req.uri().query(), None);

	let mut req = crate::http::tests_common::request(
		"https://example.com/v1beta/models/m:generateContent?key=abc",
		http::Method::POST,
		&[],
	);
	strip_alt_query(&mut req);
	assert_eq!(req.uri().query(), Some("key=abc"));
}

#[test]
fn streaming_amend_on_drop_updates_local_rate_limit() {
	let rate_limit =
		crate::http::localratelimit::RateLimit::try_from(crate::http::localratelimit::RateLimitSpec {
			max_tokens: 10,
			tokens_per_fill: 10,
			fill_interval: std::time::Duration::from_secs(60),
			limit_type: crate::http::localratelimit::RateLimitType::Tokens,
		})
		.unwrap();
	let log = AsyncLog::default();
	log.store(Some(LLMInfo {
		request: llm_request_with_tokens(Some(2)),
		response: LLMResponse {
			input_tokens: Some(2),
			output_tokens: Some(4),
			..Default::default()
		},
	}));

	let mut amend = AmendOnDrop::new(
		log,
		LLMResponsePolicies {
			local_rate_limit: vec![rate_limit.clone()],
			..Default::default()
		},
		None,
		None,
	);
	amend.report_usage();

	assert!(
		rate_limit
			.check_llm_request(&llm_request_with_tokens(Some(7)))
			.is_err()
	);
	assert!(
		rate_limit
			.check_llm_request(&llm_request_with_tokens(Some(6)))
			.is_ok()
	);
}

#[test]
fn streaming_amend_on_drop_uses_cache_inclusive_input_tokens() {
	let rate_limit =
		crate::http::localratelimit::RateLimit::try_from(crate::http::localratelimit::RateLimitSpec {
			max_tokens: 10,
			tokens_per_fill: 10,
			fill_interval: std::time::Duration::from_secs(60),
			limit_type: crate::http::localratelimit::RateLimitType::Tokens,
		})
		.unwrap();
	let mut request = llm_request_with_tokens(Some(5));
	request.cache_convention = CacheTokenConvention::InputExcludesCache;
	let log = AsyncLog::default();
	log.store(Some(LLMInfo {
		request,
		response: LLMResponse {
			input_tokens: Some(2),
			cached_input_tokens: Some(2),
			cache_creation_input_tokens: Some(1),
			output_tokens: Some(4),
			..Default::default()
		},
	}));

	let mut amend = AmendOnDrop::new(
		log,
		LLMResponsePolicies {
			local_rate_limit: vec![rate_limit.clone()],
			..Default::default()
		},
		None,
		None,
	);
	amend.report_usage();

	assert!(
		rate_limit
			.check_llm_request(&llm_request_with_tokens(Some(7)))
			.is_err()
	);
	assert!(
		rate_limit
			.check_llm_request(&llm_request_with_tokens(Some(6)))
			.is_ok()
	);
}

fn test_root() -> &'static Path {
	Path::new("../llm/src/tests")
}

fn fixture_path(relative_path: &str) -> PathBuf {
	test_root().join(relative_path)
}

#[test]
fn response_prompt_guard_headers_copies_request_traceparent() {
	let traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
		.parse()
		.unwrap();
	let mut response_headers = ::http::HeaderMap::new();
	response_headers.insert("x-upstream", "value".parse().unwrap());

	let headers = response_prompt_guard_headers(&response_headers, Some(&traceparent));

	assert_eq!(headers.get("x-upstream").unwrap(), "value");
	assert_eq!(
		headers.get(TRACEPARENT).unwrap(),
		"00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
	);
	assert!(!response_headers.contains_key(TRACEPARENT));
}

#[test]
fn response_prompt_guard_headers_overwrites_upstream_traceparent() {
	let traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
		.parse()
		.unwrap();
	let mut response_headers = ::http::HeaderMap::new();
	response_headers.insert(
		TRACEPARENT,
		"00-11111111111111111111111111111111-2222222222222222-01"
			.parse()
			.unwrap(),
	);

	let headers = response_prompt_guard_headers(&response_headers, Some(&traceparent));

	assert_eq!(
		headers.get(TRACEPARENT).unwrap(),
		"00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
	);
	assert_eq!(
		response_headers.get(TRACEPARENT).unwrap(),
		"00-11111111111111111111111111111111-2222222222222222-01"
	);
}

#[tokio::test]
async fn test_passthrough() {
	let input_path = fixture_path("requests/completions/full.json");
	let openai_str = &fs::read_to_string(&input_path).expect("Failed to read input file");
	let openai_raw: Value = serde_json::from_str(openai_str).expect("Failed to parse input json");
	let openai: types::completions::Request =
		serde_json::from_str(openai_str).expect("Failed to parse input JSON");
	let t = serde_json::to_string_pretty(&openai).unwrap();
	let t2 = serde_json::to_string_pretty(&openai_raw).unwrap();
	assert_eq!(
		serde_json::from_str::<Value>(&t).unwrap(),
		serde_json::from_str::<Value>(&t2).unwrap(),
		"{t}\n{t2}"
	);
}

fn openai_inline_moderation_param() -> openai::ModerationParam {
	openai::ModerationParam {
		model: strng::new("omni-moderation-latest"),
		policy: Some(openai::ModerationPolicyParam {
			input: Some(openai::ModerationConfigParam {
				mode: openai::ModerationMode::Block,
			}),
			output: Some(openai::ModerationConfigParam {
				mode: openai::ModerationMode::Score,
			}),
		}),
	}
}

fn openai_inline_moderation_value() -> Value {
	json!({
		"model": "omni-moderation-latest",
		"policy": {
			"input": { "mode": "block" },
			"output": { "mode": "score" }
		}
	})
}

fn openai_test_backend_info() -> crate::http::auth::BackendInfo {
	let inputs = crate::test_helpers::proxymock::setup_proxy_test("{}")
		.unwrap()
		.pi;
	crate::http::auth::BackendInfo {
		target: crate::types::agent::BackendTarget::Invalid,
		call_target: Target::from(("api.openai.com", 443)),
		inputs,
	}
}

#[tokio::test]
async fn openai_inline_moderation_injected_for_completions() {
	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: Some(openai_inline_moderation_param()),
	});
	let backend_info = openai_test_backend_info();
	let req = ::http::Request::builder()
		.uri("/v1/chat/completions")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "gpt-5",
				"messages": [{"role": "user", "content": "hello"}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		upstream_route_type,
		..
	} = provider
		.process_completions_request(&backend_info, None, req, false, &mut None, None)
		.await
		.expect("OpenAI completions request should process")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(upstream_route_type, RouteType::Completions);
	assert_eq!(
		forwarded_json["moderation"],
		openai_inline_moderation_value()
	);
}

#[tokio::test]
async fn openai_inline_moderation_overrides_client_value_for_completions() {
	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: Some(openai_inline_moderation_param()),
	});
	let backend_info = openai_test_backend_info();
	let req = ::http::Request::builder()
		.uri("/v1/chat/completions")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "gpt-5",
				"messages": [{"role": "user", "content": "hello"}],
				"moderation": {
					"model": "client-selected-model",
					"policy": {
						"input": { "mode": "score" },
						"output": { "mode": "score" }
					}
				}
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded, ..
	} = provider
		.process_completions_request(&backend_info, None, req, false, &mut None, None)
		.await
		.expect("OpenAI completions request should process")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(
		forwarded_json["moderation"],
		openai_inline_moderation_value()
	);
}

#[tokio::test]
async fn openai_client_moderation_passthrough_without_config() {
	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let backend_info = openai_test_backend_info();
	let req = ::http::Request::builder()
		.uri("/v1/chat/completions")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
					"model": "gpt-5",
					"messages": [{"role": "user", "content": "hello"}],
					"moderation": {
						"model": "client-selected-model",
						"policy": {
							"input": {
								"mode": "future-mode",
								"future_option": { "enabled": true }
							},
							"output": { "mode": "block" }
						}
					},
					"future_top_level": { "enabled": true }
				}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded, ..
	} = provider
		.process_completions_request(&backend_info, None, req, false, &mut None, None)
		.await
		.expect("OpenAI completions request should process")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(
		forwarded_json["moderation"],
		json!({
			"model": "client-selected-model",
			"policy": {
				"input": {
					"mode": "future-mode",
					"future_option": { "enabled": true }
				},
				"output": { "mode": "block" }
			}
		})
	);
	assert_eq!(
		forwarded_json["future_top_level"],
		json!({ "enabled": true })
	);
}

#[tokio::test]
async fn openai_inline_moderation_injected_for_responses() {
	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: Some(openai_inline_moderation_param()),
	});
	let backend_info = openai_test_backend_info();
	let req = ::http::Request::builder()
		.uri("/v1/responses")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "gpt-5",
				"input": "hello"
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		upstream_route_type,
		..
	} = provider
		.process_responses_request(&backend_info, None, req, false, &mut None, None)
		.await
		.expect("OpenAI responses request should process")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(upstream_route_type, RouteType::Responses);
	assert_eq!(
		forwarded_json["moderation"],
		openai_inline_moderation_value()
	);
}

#[tokio::test]
async fn responses_inactive_path_reuses_chat_renderer() {
	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let backend_info = openai_test_backend_info();
	let request_body = br#"{"model":"gpt-5","input":"hello","stream":false}"#.to_vec();
	let request = ::http::Request::builder()
		.uri("/v1/responses")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(request_body.clone()))
		.unwrap();
	let RequestResult::Success {
		request: inactive_request,
		llm_request: inactive_info,
		..
	} = provider
		.process_responses_request(&backend_info, None, request, false, &mut None, None)
		.await
		.expect("inactive Responses request should process")
	else {
		panic!("expected forwarded request");
	};

	let request = ::http::Request::builder()
		.uri("/v1/responses")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(request_body))
		.unwrap();
	let (parts, mut request) = provider
		.read_body_and_default_model::<types::responses::Request>(None, request, &mut None)
		.await
		.expect("Responses request should parse");
	provider.apply_model_alias(None, &mut request);
	let prepared = provider
		.prepare_chat_request(
			&backend_info,
			None,
			InputFormat::Responses,
			request,
			parts,
			false,
			&mut None,
			None,
		)
		.await
		.expect("Responses request preparation should succeed")
		.expect("Responses request should not be guardrail-rejected");
	let PreparedChatRequest {
		request,
		parts,
		translation,
		llm_request,
	} = prepared;
	let RequestResult::Success {
		request: shared_request,
		llm_request: shared_info,
		..
	} = provider
		.render_prepared_chat_request(
			None,
			types::ChatRequest::Responses(request),
			parts,
			translation,
			llm_request,
			&mut None,
			None,
		)
		.expect("shared Responses request rendering should succeed")
	else {
		panic!("expected shared forwarded request");
	};

	let inactive_body: Value =
		serde_json::from_slice(&inactive_request.collect().await.unwrap().to_bytes())
			.expect("inactive Responses body should be JSON");
	let shared_body: Value =
		serde_json::from_slice(&shared_request.collect().await.unwrap().to_bytes())
			.expect("shared Responses body should be JSON");
	assert_eq!(inactive_body, shared_body);
	assert_eq!(
		shared_body,
		json!({
			"model": "gpt-5",
			"input": "hello",
			"stream": false
		})
	);
	assert_eq!(inactive_info.input_format, InputFormat::Responses);
	assert!(!inactive_info.streaming);
	assert_eq!(shared_info.input_format, InputFormat::Responses);
	assert!(!shared_info.streaming);
}

#[tokio::test]
async fn responses_tool_runtime_activates_after_request_policy_defaults() {
	use std::collections::HashMap;
	use std::sync::Arc;
	use std::time::Duration;

	use secrecy::SecretString;

	use crate::llm::tool_runtime::{
		ManagedToolConfig, RuntimeLimits, ToolBackendConfig, ToolRegistry, ToolRuntimeConfig,
	};

	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let backend_info = openai_test_backend_info();
	let policy = Policy {
		tool_runtime: Some(Arc::new(
			ToolRegistry::compile(ToolRuntimeConfig {
				limits: RuntimeLimits {
					total_timeout: Duration::from_secs(30),
					max_rounds: 4,
					max_tool_calls: 4,
					max_parallel_tool_calls: 2,
					max_arguments_bytes: 1024,
					max_output_bytes: 4096,
				},
				tools: vec![ManagedToolConfig {
					name: "managed".to_owned(),
					builtin: None,
					backend: ToolBackendConfig::Http {
						url: "http://127.0.0.1:1/invoke".parse().unwrap(),
						timeout: Duration::from_secs(5),
						bearer_token: Some(SecretString::from("operator-secret")),
					},
				}],
			})
			.unwrap(),
		)),
		defaults: Some(HashMap::from([(
			"tools".to_owned(),
			json!([{
				"type": "function",
				"name": "managed",
				"description": "added by policy",
				"parameters": {"type": "object"}
			}]),
		)])),
		..Default::default()
	};
	let request = ::http::Request::builder()
		.uri("/v1/responses")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(br#"{"model":"gpt-5","input":"hello"}"#.to_vec()))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		tool_runtime: Some(runtime),
		..
	} = provider
		.process_responses_request(
			&backend_info,
			Some(&policy),
			request,
			false,
			&mut None,
			None,
		)
		.await
		.expect("managed Responses request should process")
	else {
		panic!("expected active managed tool runtime");
	};

	let crate::llm::PreparedManagedRuntime::Responses(runtime) = runtime else {
		panic!("expected Responses runtime");
	};
	let forwarded = forwarded.collect().await.unwrap().to_bytes();
	let forwarded: Value = serde_json::from_slice(&forwarded).unwrap();
	assert_eq!(forwarded["tools"][0]["name"], "managed");
	assert_eq!(
		serde_json::to_value(runtime.canonical_request).unwrap()["tools"][0]["description"],
		"added by policy"
	);
}

#[tokio::test]
async fn messages_tool_runtime_activates_after_request_policy_prompts() {
	use std::sync::Arc;
	use std::time::Duration;

	use secrecy::SecretString;

	use crate::llm::policy::PromptEnrichment;
	use crate::llm::tool_runtime::{
		ManagedToolConfig, RuntimeLimits, ToolBackendConfig, ToolRegistry, ToolRuntimeConfig,
	};

	let provider = AIProvider::Anthropic(anthropic::Provider { model: None });
	let backend_info = openai_test_backend_info();
	let policy = Policy {
		tool_runtime: Some(Arc::new(
			ToolRegistry::compile(ToolRuntimeConfig {
				limits: RuntimeLimits {
					total_timeout: Duration::from_secs(30),
					max_rounds: 4,
					max_tool_calls: 4,
					max_parallel_tool_calls: 2,
					max_arguments_bytes: 1024,
					max_output_bytes: 4096,
				},
				tools: vec![ManagedToolConfig {
					name: "managed".to_owned(),
					builtin: None,
					backend: ToolBackendConfig::Http {
						url: "http://127.0.0.1:1/invoke".parse().unwrap(),
						timeout: Duration::from_secs(5),
						bearer_token: Some(SecretString::from("operator-secret")),
					},
				}],
			})
			.unwrap(),
		)),
		prompts: Some(PromptEnrichment {
			prepend: vec![SimpleChatCompletionMessage {
				role: strng::new("system"),
				content: strng::new("policy system"),
			}],
			append: vec![],
		}),
		..Default::default()
	};
	let request = ::http::Request::builder()
		.uri("/v1/messages")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model":"claude-x",
				"max_tokens":64,
				"messages":[{"role":"user","content":"hello"}],
				"tools":[{"name":"managed","input_schema":{"type":"object"}}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		tool_runtime: Some(runtime),
		..
	} = provider
		.process_messages_request(
			&backend_info,
			Some(&policy),
			request,
			false,
			&mut None,
			None,
		)
		.await
		.expect("managed Messages request should process")
	else {
		panic!("expected active managed tool runtime");
	};

	let crate::llm::PreparedManagedRuntime::Messages(runtime) = runtime else {
		panic!("expected Messages runtime");
	};
	let forwarded = forwarded.collect().await.unwrap().to_bytes();
	let forwarded: Value = serde_json::from_slice(&forwarded).unwrap();
	let canonical = serde_json::to_value(runtime.canonical_request).unwrap();

	assert_eq!(forwarded["system"][0]["text"], "policy system");
	assert_eq!(canonical["system"][0]["text"], "policy system");
	assert_eq!(forwarded["tools"][0]["name"], "managed");
	assert_eq!(canonical["tools"][0]["name"], "managed");
}

#[derive(Clone)]
struct Task12ResponseSequence {
	responses: std::sync::Arc<Vec<Value>>,
	next: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl wiremock::Respond for Task12ResponseSequence {
	fn respond(&self, _request: &wiremock::Request) -> wiremock::ResponseTemplate {
		let index = self.next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
		let response = self
			.responses
			.get(index)
			.or_else(|| self.responses.last())
			.expect("Task 12 response sequence is non-empty")
			.clone();
		wiremock::ResponseTemplate::new(200).set_body_json(response)
	}
}

#[derive(Clone)]
struct Task12BackendBarrier {
	inner: std::sync::Arc<(
		std::sync::Mutex<Task12BackendBarrierState>,
		std::sync::Condvar,
	)>,
	timeout: std::time::Duration,
}

#[derive(Default)]
struct Task12BackendBarrierState {
	arrived: std::collections::HashSet<&'static str>,
	confirmed: std::collections::HashSet<&'static str>,
}

impl Task12BackendBarrier {
	fn new(timeout: std::time::Duration) -> Self {
		Self {
			inner: std::sync::Arc::new((
				std::sync::Mutex::new(Task12BackendBarrierState::default()),
				std::sync::Condvar::new(),
			)),
			timeout,
		}
	}

	fn arrive_and_wait(&self, label: &'static str) -> bool {
		let (state, changed) = &*self.inner;
		let deadline = std::time::Instant::now() + self.timeout;
		let mut state = state.lock().expect("Task 12 backend barrier lock");
		state.arrived.insert(label);
		changed.notify_all();
		while state.arrived.len() < 2 {
			let remaining = deadline.saturating_duration_since(std::time::Instant::now());
			if remaining.is_zero() {
				return false;
			}
			let (next, result) = changed
				.wait_timeout(state, remaining)
				.expect("Task 12 backend barrier wait");
			state = next;
			if result.timed_out() && state.arrived.len() < 2 {
				return false;
			}
		}
		state.confirmed.insert(label);
		true
	}

	fn confirmed_labels(&self) -> std::collections::HashSet<&'static str> {
		self
			.inner
			.0
			.lock()
			.expect("Task 12 backend barrier lock")
			.confirmed
			.clone()
	}
}

#[derive(Clone)]
struct Task12DelayedToolResponse {
	label: &'static str,
	barrier: Task12BackendBarrier,
	delay: std::time::Duration,
	body: Value,
	status: u16,
}

impl wiremock::Respond for Task12DelayedToolResponse {
	fn respond(&self, _request: &wiremock::Request) -> wiremock::ResponseTemplate {
		if !self.barrier.arrive_and_wait(self.label) {
			return wiremock::ResponseTemplate::new(500);
		}
		wiremock::ResponseTemplate::new(self.status)
			.set_delay(self.delay)
			.set_body_json(self.body.clone())
	}
}

fn task12_responses_body(id: &str, output: Value, input_tokens: u64, output_tokens: u64) -> Value {
	json!({
		"id": id,
		"object": "response",
		"created_at": 1,
		"status": "completed",
		"model": "mock-model",
		"output": output,
		"usage": {
			"input_tokens": input_tokens,
			"output_tokens": output_tokens,
			"total_tokens": input_tokens + output_tokens
		}
	})
}

fn task12_tool_runtime_backend(
	model: &wiremock::MockServer,
	web_search: &wiremock::MockServer,
	sandbox: &wiremock::MockServer,
) -> crate::types::agent::BackendWithPolicies {
	use std::sync::Arc;
	use std::time::Duration;

	use secrecy::SecretString;

	use crate::llm::tool_runtime::{
		BuiltinTool, ManagedToolConfig, RuntimeLimits, ToolBackendConfig, ToolRegistry,
		ToolRuntimeConfig,
	};
	use crate::types::agent::{Backend, BackendTrafficPolicy, ResourceName, Target};

	let runtime = Arc::new(
		ToolRegistry::compile(ToolRuntimeConfig {
			limits: RuntimeLimits {
				total_timeout: Duration::from_secs(10),
				max_rounds: 4,
				max_tool_calls: 4,
				max_parallel_tool_calls: 2,
				max_arguments_bytes: 4096,
				max_output_bytes: 16_384,
			},
			tools: vec![
				ManagedToolConfig {
					name: "web_search".to_owned(),
					builtin: Some(BuiltinTool::WebSearch),
					backend: ToolBackendConfig::Http {
						url: format!("http://{}/invoke", web_search.address())
							.parse()
							.expect("loopback Web Search URL"),
						timeout: Duration::from_secs(3),
						bearer_token: Some(SecretString::from("task12-web-token")),
					},
				},
				ManagedToolConfig {
					name: "code_interpreter".to_owned(),
					builtin: Some(BuiltinTool::CodeInterpreter),
					backend: ToolBackendConfig::E2b {
						api_url: format!("http://{}", sandbox.address())
							.parse()
							.expect("loopback E2B API URL"),
						domain: "sandbox.example.com".into(),
						timeout: Duration::from_secs(3),
						api_key: SecretString::from("task12-sandbox-token"),
					},
				},
			],
		})
		.expect("valid mixed-tool runtime"),
	);
	let mut policy = Arc::unwrap_or_clone(crate::llm::model_router::default_route_types());
	policy.tool_runtime = Some(runtime);
	let provider_name = agent_core::strng::new("task12-mock-openai");
	let provider = crate::llm::NamedAIProvider {
		name: provider_name.clone(),
		provider: AIProvider::OpenAI(openai::Provider {
			model: None,
			moderation: None,
		}),
		provider_backend: None,
		host_override: Some(Target::Address(*model.address())),
		path_override: None,
		path_prefix: None,
		tokenize: false,
		inline_policies: vec![
			BackendTrafficPolicy::backend_auth(crate::http::auth::BackendAuthKind::Key {
				value: SecretString::from("task12-model-token"),
				location: None,
			}),
			BackendTrafficPolicy::AI(Arc::new(policy)),
		],
	};
	Backend::AI(
		ResourceName::new("task12-managed-llm".into(), "".into()),
		crate::llm::AIBackend {
			providers: crate::types::loadbalancer::EndpointSet::new(vec![vec![(
				provider_name,
				provider,
			)]]),
		},
	)
	.into()
}

#[test]
fn task12_backend_barrier_requires_both_concurrent_arrivals() {
	use std::time::{Duration, Instant};

	let serial = Task12BackendBarrier::new(Duration::from_millis(40));
	let started = Instant::now();
	assert!(!serial.arrive_and_wait("web_search"));
	assert!(
		started.elapsed() < Duration::from_millis(250),
		"a serial backend must fail its bounded handshake quickly"
	);

	let concurrent = Task12BackendBarrier::new(Duration::from_secs(1));
	let peer = concurrent.clone();
	let web = std::thread::spawn(move || peer.arrive_and_wait("web_search"));
	let sandbox = concurrent.arrive_and_wait("sandbox");
	assert!(sandbox);
	assert!(web.join().unwrap());
	assert_eq!(
		concurrent.confirmed_labels(),
		["sandbox", "web_search"].into_iter().collect()
	);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn responses_tool_runtime_dual_builtin_cross_component_acceptance() {
	use std::sync::Arc;
	use std::time::Duration;

	use ::http::Method;
	use wiremock::Mock;

	use crate::llm::tool_runtime::{
		SandboxOperation, SandboxOperationLabels, SandboxOperationOutcome, ToolBackendLabel,
		ToolCallLabels, ToolExecutionOutcome, ToolRuntimeOutcome, ToolRuntimeOutcomeLabels,
	};
	use crate::test_helpers::proxymock;

	const INTERMEDIATE_TEXT: &str = "task12-intermediate-must-not-leak";
	const WEB_CALL_ID: &str = "call-web-first";
	const CODE_CALL_ID: &str = "call-code-second";
	const CODE: &str = "print(6 * 7)";

	let intermediate = vec![
		json!({
			"type": "message",
			"id": "msg_intermediate",
			"role": "assistant",
			"status": "completed",
			"content": [{
				"type": "output_text",
				"text": INTERMEDIATE_TEXT,
				"annotations": []
			}]
		}),
		json!({
			"type": "function_call",
			"id": "fc_web",
			"call_id": WEB_CALL_ID,
			"name": "_agentgateway_web_search",
			"arguments": "{\"query\":\"current agentgateway release\"}",
			"status": "completed"
		}),
		json!({
			"type": "function_call",
			"id": "fc_code",
			"call_id": CODE_CALL_ID,
			"name": "_agentgateway_code_interpreter",
			"arguments": format!("{{\"code\":{}}}", serde_json::to_string(CODE).unwrap()),
			"status": "completed"
		}),
	];
	let final_output = json!([{
		"type": "message",
		"id": "msg_final",
		"role": "assistant",
		"status": "completed",
		"content": [{
			"type": "output_text",
			"text": "The searched release value is 42.",
			"annotations": []
		}]
	}]);
	let model = wiremock::MockServer::start().await;
	Mock::given(wiremock::matchers::method("POST"))
		.respond_with(Task12ResponseSequence {
			responses: Arc::new(vec![
				task12_responses_body("resp-intermediate", json!(intermediate.clone()), 11, 7),
				task12_responses_body("resp-final", final_output, 13, 5),
			]),
			next: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
		})
		.mount(&model)
		.await;

	let backend_barrier = Task12BackendBarrier::new(Duration::from_secs(1));
	let web_search = wiremock::MockServer::start().await;
	Mock::given(wiremock::matchers::method("POST"))
		.respond_with(Task12DelayedToolResponse {
			label: "web_search",
			barrier: backend_barrier.clone(),
			delay: Duration::from_millis(500),
			body: json!({
				"results": [{
					"title": "AgentGateway release",
					"url": "https://agentgateway.dev/release",
					"snippet": "The stable release value is 42.",
					"published_at": null
				}]
			}),
			status: 200,
		})
		.mount(&web_search)
		.await;
	let sandbox = wiremock::MockServer::start().await;
	Mock::given(wiremock::matchers::method("POST"))
		.and(wiremock::matchers::path("/sandboxes"))
		.respond_with(Task12DelayedToolResponse {
			label: "sandbox",
			barrier: backend_barrier.clone(),
			delay: Duration::from_millis(40),
			body: json!({
				"clientID": "task12-client",
				"envdVersion": "0.1.0",
				"sandboxID": "task12-sandbox",
				"templateID": "code-interpreter-v1",
				"domain": "sandbox.example.com",
				"envdAccessToken": "task12-envd-token",
				"trafficAccessToken": "task12-traffic-token"
			}),
			status: 201,
		})
		.mount(&sandbox)
		.await;
	Mock::given(wiremock::matchers::method("POST"))
		.and(wiremock::matchers::path("/contexts"))
		.respond_with(wiremock::ResponseTemplate::new(200).set_body_json(json!({
			"id": "task12-context", "language": "python", "cwd": "/home/user"
		})))
		.mount(&sandbox)
		.await;
	Mock::given(wiremock::matchers::method("POST"))
		.and(wiremock::matchers::path("/execute"))
		.respond_with(wiremock::ResponseTemplate::new(200).set_body_string(
			"{\"type\":\"stdout\",\"text\":\"42\\n\",\"timestamp\":1}\n{\"type\":\"number_of_executions\",\"execution_count\":1}\n",
		))
		.mount(&sandbox)
		.await;
	Mock::given(wiremock::matchers::method("DELETE"))
		.and(wiremock::matchers::path("/contexts/task12-context"))
		.respond_with(wiremock::ResponseTemplate::new(204))
		.mount(&sandbox)
		.await;
	Mock::given(wiremock::matchers::method("DELETE"))
		.and(wiremock::matchers::path("/sandboxes/task12-sandbox"))
		.respond_with(wiremock::ResponseTemplate::new(204))
		.mount(&sandbox)
		.await;

	let bind = proxymock::setup_proxy_test("{}")
		.expect("Task 12 proxy harness")
		.with_raw_backend(task12_tool_runtime_backend(&model, &web_search, &sandbox))
		.with_bind(proxymock::simple_bind())
		.with_route(proxymock::basic_named_route("/task12-managed-llm".into()));
	let io = bind.serve_http(proxymock::BIND_KEY);
	let request = serde_json::to_vec(&json!({
		"model": "smart",
		"input": "Search the current release and calculate its stable value.",
		"tools": [
			{"type": "web_search"},
			{"type": "code_interpreter", "container": {"type": "auto"}}
		],
		"parallel_tool_calls": true,
		"stream": false
	}))
	.unwrap();
	let response =
		proxymock::send_request_body(io, Method::POST, "http://lo/v1/responses", &request).await;

	assert_eq!(response.status(), 200);
	let response: Value =
		serde_json::from_slice(&proxymock::read_body_raw(response.into_body()).await).unwrap();
	assert_eq!(response["id"], "resp-final");
	assert_eq!(response["model"], "mock-model");
	assert_eq!(response["output"].as_array().unwrap().len(), 1);
	assert_eq!(response["output"][0]["type"], "message");
	assert_eq!(
		response["output"][0]["content"][0]["text"],
		"The searched release value is 42."
	);
	assert_eq!(response["usage"]["input_tokens"], 24);
	assert_eq!(response["usage"]["output_tokens"], 12);
	assert_eq!(response["usage"]["total_tokens"], 36);
	let final_wire = serde_json::to_string(&response).unwrap();
	for intermediate_value in [
		INTERMEDIATE_TEXT,
		WEB_CALL_ID,
		CODE_CALL_ID,
		"The stable release value is 42.",
		"42\\n",
	] {
		assert!(
			!final_wire.contains(intermediate_value),
			"only the final model round may reach the client"
		);
	}

	let model_requests = model.received_requests().await.unwrap();
	assert_eq!(model_requests.len(), 2, "exactly two model rounds");
	let first_model_request: Value = serde_json::from_slice(&model_requests[0].body).unwrap();
	assert_eq!(first_model_request["stream"], false);
	assert_eq!(
		first_model_request["tools"]
			.as_array()
			.unwrap()
			.iter()
			.map(|tool| tool["name"].as_str().unwrap())
			.collect::<Vec<_>>(),
		["_agentgateway_web_search", "_agentgateway_code_interpreter"]
	);
	let second_model_request: Value = serde_json::from_slice(&model_requests[1].body).unwrap();
	let canonical_input = second_model_request["input"].as_array().unwrap();
	assert_eq!(canonical_input.len(), 6);
	assert_eq!(&canonical_input[1..4], intermediate.as_slice());
	assert_eq!(canonical_input[4]["type"], "function_call_output");
	assert_eq!(canonical_input[4]["call_id"], WEB_CALL_ID);
	assert_eq!(canonical_input[5]["type"], "function_call_output");
	assert_eq!(canonical_input[5]["call_id"], CODE_CALL_ID);
	let web_output: Value =
		serde_json::from_str(canonical_input[4]["output"].as_str().unwrap()).unwrap();
	let code_output: Value =
		serde_json::from_str(canonical_input[5]["output"].as_str().unwrap()).unwrap();
	assert_eq!(web_output["ok"], true);
	assert_eq!(web_output["results"][0]["title"], "AgentGateway release");
	assert_eq!(code_output["ok"], true);
	assert_eq!(code_output["stdout"], "42\n");

	let web_requests = web_search.received_requests().await.unwrap();
	assert_eq!(web_requests.len(), 1);
	assert_eq!(
		web_requests[0]
			.headers
			.get("authorization")
			.and_then(|value| value.to_str().ok()),
		Some("Bearer task12-web-token")
	);
	let web_request: Value = serde_json::from_slice(&web_requests[0].body).unwrap();
	assert_eq!(web_request["query"], "current agentgateway release");
	let sandbox_requests = sandbox.received_requests().await.unwrap();
	assert_eq!(sandbox_requests.len(), 5, "one direct E2B lifecycle");
	let create = sandbox_requests
		.iter()
		.find(|request| request.url.path() == "/sandboxes")
		.unwrap();
	assert_eq!(
		create
			.headers
			.get("x-api-key")
			.and_then(|value| value.to_str().ok()),
		Some("task12-sandbox-token")
	);
	let sandbox_request: Value = serde_json::from_slice(&create.body).unwrap();
	assert_eq!(sandbox_request["templateID"], "code-interpreter-v1");
	let execute = sandbox_requests
		.iter()
		.find(|request| request.url.path() == "/execute")
		.unwrap();
	let execute_request: Value = serde_json::from_slice(&execute.body).unwrap();
	assert_eq!(execute_request["context_id"], "task12-context");
	assert_ne!(
		execute_request["code"], CODE,
		"gateway wraps code to bound output"
	);

	assert_eq!(
		backend_barrier.confirmed_labels(),
		["sandbox", "web_search"].into_iter().collect(),
		"each delayed backend must observe its peer before either completes"
	);

	let web_labels = ToolCallLabels {
		tool: "web_search".to_owned(),
		backend: ToolBackendLabel::Http,
		outcome: ToolExecutionOutcome::Success,
	};
	let code_labels = ToolCallLabels {
		tool: "code_interpreter".to_owned(),
		backend: ToolBackendLabel::E2b,
		outcome: ToolExecutionOutcome::Success,
	};
	assert_eq!(
		bind
			.pi
			.metrics
			.tool_runtime_calls
			.get_or_create(&web_labels)
			.get(),
		1
	);
	assert_eq!(
		bind
			.pi
			.metrics
			.tool_runtime_calls
			.get_or_create(&code_labels)
			.get(),
		1
	);
	assert_eq!(
		bind
			.pi
			.metrics
			.tool_runtime_model_rounds
			.get_or_create(&ToolRuntimeOutcomeLabels {
				outcome: ToolRuntimeOutcome::Success,
			})
			.get(),
		2
	);
	assert_eq!(
		bind
			.pi
			.metrics
			.tool_runtime_sandbox_operations
			.get_or_create(&SandboxOperationLabels {
				operation: SandboxOperation::Execute,
				outcome: SandboxOperationOutcome::Success,
			})
			.get(),
		1
	);
	assert_eq!(
		bind
			.pi
			.metrics
			.tool_runtime_sandbox_operations
			.get_or_create(&SandboxOperationLabels {
				operation: SandboxOperation::Cleanup,
				outcome: SandboxOperationOutcome::Success,
			})
			.get(),
		1
	);
	let bounded_telemetry = format!("{web_labels:?}{code_labels:?}");
	for content in [INTERMEDIATE_TEXT, WEB_CALL_ID, CODE_CALL_ID, CODE, "42\n"] {
		assert!(
			!bounded_telemetry.contains(content),
			"telemetry labels must remain content-free"
		);
	}
}

mod task12_live_tools {
	use std::collections::HashMap;
	use std::path::{Path, PathBuf};
	use std::time::Duration;

	use serde_json::{Value, json};

	const LIVE_KEYS: [&str; 6] = [
		"AGENTGATEWAY_LIVE_TOOLS",
		"FC_WEB_SEARCH_URL",
		"FC_WEB_SEARCH_TOKEN",
		"E2B_API_KEY",
		"E2B_API_URL",
		"E2B_DOMAIN",
	];
	const MAX_LIVE_RESPONSE_BYTES: usize = 1_048_576;

	#[test]
	fn dotenv_candidates_normalize_worktree_and_main_checkout_roots() {
		let manifest = Path::new("checkout")
			.join(".worktrees")
			.join("task12")
			.join("crates")
			.join("agentgateway");
		assert_eq!(
			dotenv_candidates_for_manifest(&manifest),
			vec![
				Path::new("checkout")
					.join(".worktrees")
					.join("task12")
					.join(".env"),
				Path::new("checkout").join(".env"),
			]
		);

		let main_manifest = Path::new("checkout").join("crates").join("agentgateway");
		assert_eq!(
			dotenv_candidates_for_manifest(&main_manifest),
			vec![Path::new("checkout").join(".env")]
		);
	}

	#[test]
	fn live_response_validation_errors_are_content_free() {
		const EXTERNAL_VALUE: &str = "external-content-must-not-escape";
		let web = LiveResponse {
			status: 502,
			body: serde_json::to_vec(&json!({"error": EXTERNAL_VALUE})).unwrap(),
		};
		let web_error = validate_live_web_search_response(&web).unwrap_err();
		assert_eq!(web_error, "live Web Search returned non-200 status");
		assert!(!web_error.contains(EXTERNAL_VALUE));
	}
	struct LiveConfiguration {
		web_search_url: String,
		web_search_token: String,
		e2b_api_key: String,
		e2b_api_url: String,
		e2b_domain: String,
	}

	impl LiveConfiguration {
		fn load() -> Option<Self> {
			let mut values = HashMap::new();
			for key in LIVE_KEYS {
				if let Ok(value) = std::env::var(key)
					&& !value.is_empty()
				{
					values.insert(key.to_owned(), value);
				}
			}
			for file in dotenv_candidates() {
				load_known_dotenv_values(&file, &mut values);
			}

			let mut missing = LIVE_KEYS
				.into_iter()
				.filter(|key| !values.contains_key(*key))
				.collect::<Vec<_>>();
			if values
				.get("AGENTGATEWAY_LIVE_TOOLS")
				.is_some_and(|gate| gate != "1")
			{
				missing.push("AGENTGATEWAY_LIVE_TOOLS");
			}
			missing.sort_unstable();
			missing.dedup();
			if !missing.is_empty() {
				eprintln!("SKIPPED live tools: missing {}", missing.join(", "));
				return None;
			}

			Some(Self {
				web_search_url: values.remove("FC_WEB_SEARCH_URL").unwrap(),
				web_search_token: values.remove("FC_WEB_SEARCH_TOKEN").unwrap(),
				e2b_api_key: values.remove("E2B_API_KEY").unwrap(),
				e2b_api_url: values.remove("E2B_API_URL").unwrap(),
				e2b_domain: values.remove("E2B_DOMAIN").unwrap(),
			})
		}
	}

	fn dotenv_candidates_for_manifest(manifest_dir: &Path) -> Vec<PathBuf> {
		let Some(root) = manifest_dir.parent().and_then(Path::parent) else {
			return Vec::new();
		};
		let mut candidates = vec![root.join(".env")];
		if root
			.parent()
			.and_then(Path::file_name)
			.is_some_and(|name| name == ".worktrees")
			&& let Some(main_checkout) = root.parent().and_then(Path::parent)
		{
			candidates.push(main_checkout.join(".env"));
		}
		candidates
	}

	fn dotenv_candidates() -> Vec<PathBuf> {
		dotenv_candidates_for_manifest(Path::new(env!("CARGO_MANIFEST_DIR")))
	}

	fn load_known_dotenv_values(path: &Path, values: &mut HashMap<String, String>) {
		let Ok(contents) = std::fs::read_to_string(path) else {
			return;
		};
		for line in contents.lines() {
			let line = line.trim();
			if line.is_empty() || line.starts_with('#') {
				continue;
			}
			let line = line.strip_prefix("export ").unwrap_or(line);
			let Some((key, raw_value)) = line.split_once('=') else {
				continue;
			};
			let key = key.trim();
			if !LIVE_KEYS.contains(&key) || values.contains_key(key) {
				continue;
			}
			let raw_value = raw_value.trim();
			let value = if raw_value.len() >= 2
				&& ((raw_value.starts_with('"') && raw_value.ends_with('"'))
					|| (raw_value.starts_with('\'') && raw_value.ends_with('\'')))
			{
				&raw_value[1..raw_value.len() - 1]
			} else {
				raw_value
			};
			if !value.is_empty() {
				values.insert(key.to_owned(), value.to_owned());
			}
		}
	}

	struct LiveResponse {
		status: u16,
		body: Vec<u8>,
	}

	async fn live_post(url: String, token: String, payload: Value) -> LiveResponse {
		let client = reqwest::Client::builder()
			.timeout(Duration::from_secs(30))
			.redirect(reqwest::redirect::Policy::none())
			.build()
			.expect("live FC HTTP client must build");
		let call = async move {
			let mut response = client
				.post(url)
				.bearer_auth(token)
				.json(&payload)
				.send()
				.await
				.map_err(|_| ())?;
			let status = response.status().as_u16();
			let mut body = Vec::new();
			while let Some(chunk) = response.chunk().await.map_err(|_| ())? {
				if body.len().saturating_add(chunk.len()) > MAX_LIVE_RESPONSE_BYTES {
					return Err(());
				}
				body.extend_from_slice(&chunk);
			}
			Ok(LiveResponse { status, body })
		};
		match tokio::time::timeout(Duration::from_secs(30), call).await {
			Ok(Ok(response)) => response,
			Ok(Err(())) | Err(_) => {
				panic!("live FC call failed or exceeded its 30-second limit")
			},
		}
	}

	fn live_json(response: &LiveResponse) -> Result<Value, &'static str> {
		serde_json::from_slice(&response.body).map_err(|_| "live FC response was not valid JSON")
	}

	fn validate_live_web_search_response(response: &LiveResponse) -> Result<(), &'static str> {
		if response.status != 200 {
			return Err("live Web Search returned non-200 status");
		}
		let body = live_json(response)?;
		if !body.get("results").is_some_and(Value::is_array) {
			return Err("live Web Search response omitted its results array");
		}
		Ok(())
	}

	#[tokio::test]
	#[ignore = "opt-in: requires AGENTGATEWAY_LIVE_TOOLS=1 and Web Search/E2B configuration"]
	async fn web_search_function_smoke() {
		let Some(configuration) = LiveConfiguration::load() else {
			return;
		};
		let response = live_post(
			configuration.web_search_url,
			configuration.web_search_token,
			json!({"query": "Alibaba Cloud Function Compute documentation"}),
		)
		.await;
		assert!(
			validate_live_web_search_response(&response).is_ok(),
			"live Web Search response failed fixed contract validation"
		);
	}

	#[tokio::test]
	#[ignore = "opt-in: creates one E2B Sandbox through the direct backend and verifies cleanup"]
	async fn e2b_sandbox_smoke_and_cleanup() {
		use secrecy::SecretString;

		use crate::llm::tool_runtime::{
			E2bSandboxBackend, ManagedToolCall, ToolBackend, ToolExecutionContext,
		};

		let Some(configuration) = LiveConfiguration::load() else {
			return;
		};
		let backend = E2bSandboxBackend::new(
			crate::test_helpers::policy_client(),
			configuration
				.e2b_api_url
				.parse()
				.expect("valid E2B_API_URL"),
			configuration.e2b_domain,
			Duration::from_secs(30),
			SecretString::from(configuration.e2b_api_key),
			MAX_LIVE_RESPONSE_BYTES,
		)
		.expect("valid direct E2B backend configuration");
		let result = backend
			.execute_batch(
				vec![ManagedToolCall {
					public_name: "code_interpreter".into(),
					internal_name: "_agentgateway_code_interpreter".into(),
					call_id: "task12-live-sandbox".into(),
					arguments: json!({"code": "print(6 * 7)"}),
					trusted_options: json!({}),
				}],
				ToolExecutionContext {
					request_id: None,
					deadline: Some(std::time::Instant::now() + Duration::from_secs(30)),
				},
			)
			.await
			.expect("direct E2B live execution must succeed")
			.results
			.pop()
			.expect("direct E2B live execution returns one result")
			.into_model_output(MAX_LIVE_RESPONSE_BYTES)
			.expect("direct E2B live output must normalize");
		assert_eq!(result.get("stdout").and_then(Value::as_str), Some("42\n"));
	}
}

#[tokio::test]
async fn openai_inline_moderation_injected_after_messages_translation() {
	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: Some(openai_inline_moderation_param()),
	});
	let backend_info = openai_test_backend_info();
	let req = ::http::Request::builder()
		.uri("/v1/messages")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "gpt-5",
				"max_tokens": 64,
				"messages": [{"role": "user", "content": "hello"}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		upstream_route_type,
		..
	} = provider
		.process_messages_request(&backend_info, None, req, false, &mut None, None)
		.await
		.expect("Anthropic messages request should translate to OpenAI completions")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(upstream_route_type, RouteType::Completions);
	assert_eq!(
		forwarded_json["moderation"],
		openai_inline_moderation_value()
	);
}

#[test]
fn openai_response_passthrough_preserves_moderation_fields() {
	let completion_response: types::completions::Response = serde_json::from_value(json!({
		"model": "gpt-5",
		"usage": null,
		"choices": [],
		"moderation": {
			"input": { "flagged": false },
			"output": { "flagged": true }
		}
	}))
	.expect("completion response should deserialize");
	let completion_roundtrip =
		serde_json::to_value(completion_response).expect("completion response should serialize");
	assert_eq!(
		completion_roundtrip["moderation"],
		json!({
			"input": { "flagged": false },
			"output": { "flagged": true }
		})
	);

	let responses_response: types::responses::Response = serde_json::from_value(json!({
		"id": "resp_123",
		"status": "completed",
		"output": [],
		"model": "gpt-5",
		"moderation": {
			"input": { "flagged": false },
			"output": { "flagged": true }
		}
	}))
	.expect("responses response should deserialize");
	let responses_roundtrip =
		serde_json::to_value(responses_response).expect("responses response should serialize");
	assert_eq!(
		responses_roundtrip["moderation"],
		json!({
			"input": { "flagged": false },
			"output": { "flagged": true }
		})
	);
}

#[tokio::test]
async fn openai_provider_normalizes_max_tokens_before_forwarding() {
	use crate::http::auth::BackendInfo;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("api.openai.com", 443)),
		inputs,
	};
	let req = ::http::Request::builder()
		.uri("/v1/chat/completions")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "gpt-5.4",
				"max_tokens": 1024,
				"messages": [{"role": "user", "content": "hello"}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		llm_request,
		..
	} = provider
		.process_completions_request(&backend_info, None, req, false, &mut None, None)
		.await
		.expect("OpenAI completions request should process")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert!(forwarded_json.get("max_tokens").is_none());
	assert_eq!(forwarded_json["max_completion_tokens"], json!(1024));
	assert_eq!(llm_request.params.max_tokens, Some(1024));
}

#[tokio::test]
async fn openai_provider_normalizes_max_tokens_after_model_alias() {
	use crate::http::auth::BackendInfo;
	use crate::llm::policy::Policy;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("api.openai.com", 443)),
		inputs,
	};
	let policy = Policy {
		model_aliases: std::collections::HashMap::from([(
			strng::new("fast-model"),
			strng::new("gpt-5.4"),
		)]),
		..Default::default()
	};
	let req = ::http::Request::builder()
		.uri("/v1/chat/completions")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "fast-model",
				"max_tokens": 1024,
				"messages": [{"role": "user", "content": "hello"}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		llm_request,
		..
	} = provider
		.process_completions_request(&backend_info, Some(&policy), req, false, &mut None, None)
		.await
		.expect("OpenAI completions request should process")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(forwarded_json["model"], json!("gpt-5.4"));
	assert!(forwarded_json.get("max_tokens").is_none());
	assert_eq!(forwarded_json["max_completion_tokens"], json!(1024));
	assert_eq!(llm_request.request_model, "gpt-5.4");
	assert_eq!(llm_request.params.max_tokens, Some(1024));
}

#[tokio::test]
async fn openai_provider_preserves_max_tokens_for_non_gpt_models() {
	use crate::http::auth::BackendInfo;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("localhost", 11434)),
		inputs,
	};
	let req = ::http::Request::builder()
		.uri("/v1/chat/completions")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "llama3.1",
				"max_tokens": 1024,
				"messages": [{"role": "user", "content": "hello"}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		llm_request,
		..
	} = provider
		.process_completions_request(&backend_info, None, req, false, &mut None, None)
		.await
		.expect("OpenAI-compatible completions request should process")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(forwarded_json["max_tokens"], json!(1024));
	assert!(forwarded_json.get("max_completion_tokens").is_none());
	assert_eq!(llm_request.params.max_tokens, Some(1024));
}

#[tokio::test]
async fn count_tokens_resolves_model_alias_once_for_upstream_request() {
	use crate::http::auth::BackendInfo;
	use crate::llm::policy::Policy;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::Anthropic(anthropic::Provider { model: None });
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("api.anthropic.com", 443)),
		inputs,
	};
	let policy = Policy {
		model_aliases: std::collections::HashMap::from([
			(strng::new("short-name"), strng::new("middle-name")),
			(strng::new("middle-name"), strng::new("final-name")),
		]),
		..Default::default()
	};
	let req = ::http::Request::builder()
		.uri("/v1/messages/count_tokens")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "short-name",
				"messages": [{"role": "user", "content": "hello"}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		llm_request,
		..
	} = provider
		.process_count_tokens_request(&backend_info, req, Some(&policy), &mut None)
		.await
		.expect("count_tokens request should process")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(forwarded_json["model"], json!("middle-name"));
	assert_eq!(llm_request.request_model, "middle-name");
}

#[tokio::test]
async fn count_tokens_uses_native_endpoint_after_model_alias() {
	use crate::http::auth::BackendInfo;
	use crate::llm::policy::Policy;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::Vertex(vertex::Provider {
		model: None,
		region: None,
		project_id: strng::new("test-project"),
	});
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("us-central1-aiplatform.googleapis.com", 443)),
		inputs,
	};
	let policy = Policy {
		model_aliases: std::collections::HashMap::from([(
			strng::new("short-name"),
			strng::new("claude-3-5-sonnet"),
		)]),
		..Default::default()
	};
	let req = ::http::Request::builder()
		.uri("/v1/messages/count_tokens")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "short-name",
				"messages": [{"role": "user", "content": "hello"}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		llm_request,
		upstream_route_type,
		..
	} = provider
		.process_count_tokens_request(&backend_info, req, Some(&policy), &mut None)
		.await
		.expect("count_tokens request should process")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(upstream_route_type, RouteType::AnthropicTokenCount);
	assert_eq!(forwarded_json["model"], json!("claude-3-5-sonnet"));
	assert_eq!(llm_request.request_model, "claude-3-5-sonnet");
}

fn gemini_generate_content_request(uri: &str) -> ::http::Request<Body> {
	::http::Request::builder()
		.uri(uri)
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"contents": [{"role": "user", "parts": [{"text": "hello"}]}],
				"someNewTopLevelField": {"a": true}
			}"#
				.to_vec(),
		))
		.unwrap()
}

fn vertex_backend_info() -> crate::http::auth::BackendInfo {
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;
	crate::http::auth::BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("aiplatform.googleapis.com", 443)),
		inputs: setup_proxy_test("{}").unwrap().pi,
	}
}

#[tokio::test]
async fn gemini_generate_content_forwards_unknown_top_level_fields() {
	let provider = AIProvider::Vertex(vertex::Provider {
		model: None,
		region: None,
		project_id: strng::new("test-project"),
	});
	let req = gemini_generate_content_request(
		"https://example.com/v1beta/models/gemini-2.5-flash:generateContent",
	);

	let RequestResult::Success {
		request: forwarded,
		llm_request,
		..
	} = provider
		.process_gemini_request(&vertex_backend_info(), None, req, false, &mut None, None)
		.await
		.expect("generateContent request should process")
	else {
		panic!("expected forwarded request");
	};
	assert_eq!(llm_request.request_model, "gemini-2.5-flash");
	assert!(!llm_request.streaming);

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");
	assert_eq!(forwarded_json["someNewTopLevelField"], json!({"a": true}));
	assert_eq!(forwarded_json["contents"][0]["parts"][0]["text"], "hello");
	assert!(
		forwarded_json.get("model").is_none(),
		"the model rides the path, not the body: {forwarded_json}"
	);
}

#[tokio::test]
async fn gemini_stream_without_alt_sse_is_rejected_with_google_shaped_400() {
	let provider = AIProvider::Vertex(vertex::Provider {
		model: None,
		region: None,
		project_id: strng::new("test-project"),
	});
	for uri in [
		"https://example.com/v1beta/models/gemini-2.5-flash:streamGenerateContent",
		"https://example.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=json",
	] {
		let RequestResult::Rejected(resp) = provider
			.process_gemini_request(
				&vertex_backend_info(),
				None,
				gemini_generate_content_request(uri),
				false,
				&mut None,
				None,
			)
			.await
			.expect("the non-SSE streaming variant is a client error, not a gateway failure")
		else {
			panic!("expected a direct response for {uri}");
		};

		assert_eq!(resp.status(), ::http::StatusCode::BAD_REQUEST);
		let body = resp.into_body().collect().await.unwrap().to_bytes();
		let body: Value = serde_json::from_slice(&body).expect("error body should be JSON");
		assert_eq!(body["error"]["code"], json!(400));
		assert_eq!(body["error"]["status"], json!("INVALID_ARGUMENT"));
		assert!(
			body["error"]["message"]
				.as_str()
				.is_some_and(|m| m.contains("alt=sse")),
			"{body}"
		);
	}
}

fn gemini_count_tokens_request(uri: &str) -> ::http::Request<Body> {
	::http::Request::builder()
		.uri(uri)
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"contents": [{"role": "user", "parts": [{"text": "hello"}]}],
				"someNewField": {"a": true}
			}"#
				.to_vec(),
		))
		.unwrap()
}

#[tokio::test]
async fn gemini_count_tokens_passes_body_through_on_vertex() {
	use crate::http::auth::BackendInfo;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::Vertex(vertex::Provider {
		model: None,
		region: None,
		project_id: strng::new("test-project"),
	});
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("aiplatform.googleapis.com", 443)),
		inputs,
	};
	let req =
		gemini_count_tokens_request("https://example.com/v1beta/models/gemini-2.5-flash:countTokens");

	let RequestResult::Success {
		request: forwarded,
		llm_request,
		upstream_route_type,
		..
	} = provider
		.process_gemini_count_tokens_request(&backend_info, None, req, &mut None)
		.await
		.expect("countTokens request should process")
	else {
		panic!("expected forwarded request");
	};

	assert_eq!(upstream_route_type, RouteType::GeminiCountTokens);
	assert_eq!(llm_request.input_format, InputFormat::GeminiCountTokens);
	assert_eq!(llm_request.request_model, "gemini-2.5-flash");
	assert!(!llm_request.streaming);

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");
	assert_eq!(forwarded_json["someNewField"], json!({"a": true}));
	assert_eq!(forwarded_json["contents"][0]["parts"][0]["text"], "hello");
	assert!(
		forwarded_json.get("model").is_none(),
		"the model rides the path, not the body: {forwarded_json}"
	);
}

#[tokio::test]
async fn gemini_count_tokens_applies_model_alias_and_rewrites_upstream_path() {
	use crate::http::auth::BackendInfo;
	use crate::llm::policy::Policy;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::Gemini(gemini::Provider { model: None });
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from((gemini::DEFAULT_HOST_STR, 443)),
		inputs,
	};
	let policy = Policy {
		model_aliases: std::collections::HashMap::from([(
			strng::new("fast"),
			strng::new("gemini-2.5-flash"),
		)]),
		..Default::default()
	};
	let req = gemini_count_tokens_request("https://example.com/v1beta/models/fast:countTokens");

	let RequestResult::Success {
		request: mut forwarded,
		llm_request,
		upstream_route_type,
		..
	} = provider
		.process_gemini_count_tokens_request(&backend_info, Some(&policy), req, &mut None)
		.await
		.expect("countTokens request should process")
	else {
		panic!("expected forwarded request");
	};
	assert_eq!(llm_request.request_model, "gemini-2.5-flash");

	provider
		.setup_request(
			&mut forwarded,
			upstream_route_type,
			Some(&llm_request),
			None,
			None,
			false,
		)
		.expect("setup_request should succeed");
	assert_eq!(
		forwarded.uri().path(),
		"/v1beta/models/gemini-2.5-flash:countTokens"
	);
	assert_eq!(forwarded.uri().query(), None);
	assert_eq!(
		forwarded.uri().authority().map(|a| a.as_str()),
		Some(gemini::DEFAULT_HOST_STR)
	);
}

#[tokio::test]
async fn gemini_count_tokens_on_non_gemini_upstream_is_unsupported() {
	use crate::http::auth::BackendInfo;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::Anthropic(anthropic::Provider { model: None });
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("api.anthropic.com", 443)),
		inputs,
	};
	let req =
		gemini_count_tokens_request("https://example.com/v1beta/models/claude-opus-4:countTokens");

	let err = provider
		.process_gemini_count_tokens_request(&backend_info, None, req, &mut None)
		.await
		.expect_err("countTokens against a non-Gemini upstream must be rejected");
	assert!(matches!(err, AIError::UnsupportedConversion(_)), "{err}");
}

#[test]
fn gemini_count_tokens_response_reports_total_tokens() {
	let provider = AIProvider::Gemini(gemini::Provider { model: None });
	let req = LLMRequest {
		input_tokens: None,
		input_format: InputFormat::GeminiCountTokens,
		cache_convention: CacheTokenConvention::pending(),
		request_model: "gemini-2.5-flash".into(),
		provider: "gcp.gemini".into(),
		streaming: false,
		params: Default::default(),
		prompt: None,
		provider_state: None,
	};
	let body = br#"{"totalTokens":31,"promptTokensDetails":[{"modality":"TEXT","tokenCount":31}]}"#;
	let buffered = BufferedResponse {
		parts: ::http::Response::new(()).into_parts().0,
		bytes: bytes::Bytes::from_static(body),
	};

	let log = AsyncLog::<llm::LLMInfo>::default();
	let resp = provider
		.process_gemini_count_tokens_response(req, buffered, None, &log)
		.expect("countTokens response should process");
	assert_eq!(
		log.take().expect("llm info").response.count_tokens,
		Some(31)
	);
	assert!(resp.headers().get(header::CONTENT_LENGTH).is_none());
}

#[tokio::test]
async fn vertex_anthropic_messages_prepares_vertex_body() {
	use crate::http::auth::BackendInfo;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::Vertex(vertex::Provider {
		model: None,
		region: Some(strng::new("us-central1")),
		project_id: strng::new("test-project"),
	});
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("us-central1-aiplatform.googleapis.com", 443)),
		inputs,
	};
	let req = ::http::Request::builder()
		.uri("/v1/messages")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "claude-haiku-4-5-20251001",
				"max_tokens": 64,
				"messages": [{"role": "user", "content": "say hi"}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		upstream_route_type,
		..
	} = provider
		.process_messages_request(&backend_info, None, req, false, &mut None, None)
		.await
		.expect("Vertex Anthropic messages request should process")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(upstream_route_type, RouteType::Messages);
	assert!(forwarded_json.get("model").is_none());
	assert_eq!(
		forwarded_json["anthropic_version"],
		json!("vertex-2023-10-16")
	);
}

#[tokio::test]
async fn provider_model_is_set_before_llm_transformations() {
	use crate::http::auth::BackendInfo;
	use crate::llm::policy::Policy;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::OpenAI(openai::Provider {
		model: Some("gcp/failover-model".into()),
		moderation: None,
	});
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("api.openai.com", 443)),
		inputs,
	};
	let policy = Policy {
		transformations: Some(
			[(
				"model".to_string(),
				std::sync::Arc::new(
					crate::cel::Expression::new_strict(r#"llmRequest.model.stripPrefix("gcp/")"#).unwrap(),
				),
			)]
			.into_iter()
			.collect(),
		),
		..Default::default()
	};
	let req = ::http::Request::builder()
		.uri("/v1/chat/completions")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "public-model",
				"messages": [{"role": "user", "content": "hello"}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		llm_request,
		..
	} = provider
		.process_completions_request(&backend_info, Some(&policy), req, false, &mut None, None)
		.await
		.expect("OpenAI completions request should process")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(forwarded_json["model"], json!("failover-model"));
	assert_eq!(llm_request.request_model, "failover-model");
}

#[tokio::test]
async fn messages_to_completions_final_transformation() {
	use crate::llm::policy::Policy;

	async fn create_llm_request(vec_body: Vec<u8>, policy: Option<&Policy>) -> (Request, RouteType) {
		let provider = AIProvider::OpenAI(openai::Provider {
			model: None,
			moderation: None,
		});
		let backend_info = openai_test_backend_info();
		let req = ::http::Request::builder()
			.uri("/v1/messages")
			.header(::http::header::CONTENT_TYPE, "application/json")
			.body(Body::from(vec_body))
			.unwrap();
		let RequestResult::Success {
			request: forwarded,
			upstream_route_type,
			..
		} = provider
			.process_messages_request(&backend_info, policy, req, false, &mut None, None)
			.await
			.expect("Anthropic messages request should translate to OpenAI completions")
		else {
			panic!("expected forwarded request");
		};
		(forwarded, upstream_route_type)
	}
	let expr = |e: &str| std::sync::Arc::new(crate::cel::Expression::new_strict(e).unwrap());

	let policy = Policy {
		final_transformations: Some(
			[
				// Only true final-conversion: `system` became messages[0].
				(
					"converted_message_count".to_string(),
					expr("llmRequest.messages.size()"),
				),
				// Mutate a field carried through the conversion.
				("max_tokens".to_string(), expr("32")),
				("reasoning_effort".to_string(), expr("fail(\"remove\")")),
			]
			.into_iter()
			.collect(),
		),
		..Default::default()
	};

	let vec_body = br#"{
				"model": "gpt-4o",
				"max_tokens": 64,
				"system": "be brief",
				"messages": [{"role": "user", "content": "hello"}],
				"tools": [{
					"name": "get_weather",
					"description": "Look up the weather",
					"input_schema": {
						"type": "object",
						"properties": {"city": {"type": "string"}},
						"required": ["city"]
					}
				}]
			}"#
		.to_vec();

	let (forwarded, upstream_route_type) = create_llm_request(vec_body.clone(), Some(&policy)).await;
	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(upstream_route_type, RouteType::Completions);
	// The request really was converted to completions format.
	assert_eq!(forwarded_json["messages"][0]["role"], json!("system"));
	// 2 (system + user), not the 1 message the client sent.
	assert_eq!(forwarded_json["converted_message_count"], json!(2));
	assert_eq!(forwarded_json["max_tokens"], json!(32));
	// Indexing returns Null for a missing key too, so assert on key presence.
	let reasoning_effort = forwarded_json.get("reasoning_effort");
	assert!(
		reasoning_effort.is_none(),
		"reasoning_effort should be removed, got: {reasoning_effort:?}"
	);

	let (forwarded, upstream_route_type) = create_llm_request(vec_body, None).await;
	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");
	assert_eq!(upstream_route_type, RouteType::Completions);
	// The request really was converted to completions format.
	assert_eq!(forwarded_json["messages"][0]["role"], json!("system"));
	// 2 (system + user), not the 1 message the client sent.
	assert_eq!(forwarded_json["max_completion_tokens"], json!(64));
	// Indexing returns Null for a missing key too, so assert on key presence.
	let reasoning_effort = forwarded_json.get("reasoning_effort");
	assert!(
		reasoning_effort.is_some(),
		"reasoning_effort should not be empty, got: {reasoning_effort:?}"
	);
}

#[tokio::test]
async fn detect_final_transformations_skip_opaque_bodies() {
	use crate::llm::policy::Policy;

	async fn create_detect_request(
		content_type: &str,
		body: &[u8],
		policy: Option<&Policy>,
	) -> (Request, RouteType) {
		let provider = AIProvider::OpenAI(openai::Provider {
			model: None,
			moderation: None,
		});
		let backend_info = openai_test_backend_info();
		let req = ::http::Request::builder()
			.uri("/v1/passthrough")
			.header(::http::header::CONTENT_TYPE, content_type)
			.body(Body::from(body.to_vec()))
			.unwrap();
		let RequestResult::Success {
			request: forwarded,
			upstream_route_type,
			..
		} = provider
			.process_detect_request(&backend_info, policy, req, &mut None)
			.await
			.expect("detect request should process")
		else {
			panic!("expected forwarded request");
		};
		(forwarded, upstream_route_type)
	}

	let expr = |e: &str| std::sync::Arc::new(crate::cel::Expression::new_strict(e).unwrap());
	let policy = Policy {
		final_transformations: Some(
			[("max_tokens".to_string(), expr("32"))]
				.into_iter()
				.collect(),
		),
		..Default::default()
	};

	// Inner spaces and key order survive only if the body is never round-tripped through serde.
	let json_body = br#"{ "model": "gpt-4o", "max_tokens": 64 }"#;

	// A non-JSON content type is passed through opaquely even when the body happens to parse as
	// JSON, so final transformations must not rewrite (or re-serialize) it.
	let (forwarded, route_type) = create_detect_request("text/plain", json_body, Some(&policy)).await;
	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	assert_eq!(route_type, RouteType::Detect);
	assert_eq!(
		forwarded_body.as_ref(),
		json_body.as_slice(),
		"passthrough body must be forwarded byte-for-byte"
	);

	// A body that fails to parse falls back to raw passthrough, which must not become an error.
	let raw_body = b"\x00\x01not json at all";
	let (forwarded, route_type) =
		create_detect_request("application/octet-stream", raw_body, Some(&policy)).await;
	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	assert_eq!(route_type, RouteType::Detect);
	assert_eq!(forwarded_body.as_ref(), raw_body.as_slice());

	// Malformed JSON under a JSON content type takes the same raw fallback.
	let bad_json = br#"{"model": "gpt-4o", "#;
	let (forwarded, route_type) =
		create_detect_request("application/json", bad_json, Some(&policy)).await;
	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	assert_eq!(route_type, RouteType::Detect);
	assert_eq!(forwarded_body.as_ref(), bad_json.as_slice());

	// A genuine JSON detect body is still transformed: this is the behavior the guard preserves.
	let (forwarded, route_type) =
		create_detect_request("application/json", json_body, Some(&policy)).await;
	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");
	assert_eq!(route_type, RouteType::Detect);
	assert_eq!(forwarded_json["max_tokens"], json!(32));
	assert_eq!(forwarded_json["model"], json!("gpt-4o"));

	// Without a policy the JSON body is unchanged apart from the parse/serialize round trip.
	let (forwarded, _) = create_detect_request("application/json", json_body, None).await;
	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");
	assert_eq!(forwarded_json["max_tokens"], json!(64));
}

#[tokio::test]
async fn bedrock_transformed_provider_model_is_used_for_upstream_path() {
	use crate::http::auth::BackendInfo;
	use crate::llm::policy::Policy;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::bedrock(bedrock::Provider {
		model: Some(strng::new(
			"bedrock-runtime/us/anthropic.claude-3-5-sonnet-20241022-v2:0",
		)),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	});
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("bedrock-runtime.us-east-1.amazonaws.com", 443)),
		inputs,
	};
	let policy = Policy {
		transformations: Some(
			[(
				"model".to_string(),
				std::sync::Arc::new(
					crate::cel::Expression::new_strict(
						r#"llmRequest.model.stripPrefix("bedrock-runtime/us/")"#,
					)
					.unwrap(),
				),
			)]
			.into_iter()
			.collect(),
		),
		..Default::default()
	};
	let expected_model = "anthropic.claude-3-5-sonnet-20241022-v2:0";

	let req = ::http::Request::builder()
		.uri("https://gateway.example.com/v1/chat/completions")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			json!({
				"model": "client-model",
				"messages": [{"role": "user", "content": "hello"}],
				"stream": true,
			})
			.to_string(),
		))
		.unwrap();

	let RequestResult::Success {
		request: mut forwarded,
		llm_request,
		upstream_route_type,
		..
	} = provider
		.process_completions_request(&backend_info, Some(&policy), req, false, &mut None, None)
		.await
		.expect("Bedrock completions request should process")
	else {
		panic!("expected forwarded request");
	};

	assert_eq!(llm_request.request_model, expected_model);
	provider
		.setup_request(
			&mut forwarded,
			upstream_route_type,
			Some(&llm_request),
			None,
			None,
			false,
		)
		.expect("Bedrock upstream request should be finalized");
	assert_eq!(
		forwarded.uri().path(),
		format!("/model/{expected_model}/converse-stream")
	);
}

#[tokio::test]
async fn bedrock_provider_model_overrides_client_model() {
	use crate::http::auth::BackendInfo;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let configured_model = "anthropic.claude-3-5-sonnet-20241022-v2:0";
	let provider = AIProvider::bedrock(bedrock::Provider {
		model: Some(strng::new(configured_model)),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	});
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("bedrock-runtime.us-east-1.amazonaws.com", 443)),
		inputs,
	};
	let req = ::http::Request::builder()
		.uri("https://gateway.example.com/v1/chat/completions")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "client-model",
				"messages": [{"role": "user", "content": "hello"}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: mut forwarded,
		llm_request,
		upstream_route_type,
		..
	} = provider
		.process_completions_request(&backend_info, None, req, false, &mut None, None)
		.await
		.expect("Bedrock completions request should process")
	else {
		panic!("expected forwarded request");
	};

	assert_eq!(llm_request.request_model, configured_model);
	provider
		.setup_request(
			&mut forwarded,
			upstream_route_type,
			Some(&llm_request),
			None,
			None,
			false,
		)
		.expect("Bedrock upstream request should be finalized");
	assert_eq!(
		forwarded.uri().path(),
		format!("/model/{configured_model}/converse")
	);
}

#[tokio::test]
async fn llm_transformations_can_set_missing_model() {
	use crate::http::auth::BackendInfo;
	use crate::llm::policy::Policy;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("api.openai.com", 443)),
		inputs,
	};
	let policy = Policy {
		transformations: Some(
			[(
				"model".to_string(),
				std::sync::Arc::new(crate::cel::Expression::new_strict(r#""transformed-model""#).unwrap()),
			)]
			.into_iter()
			.collect(),
		),
		..Default::default()
	};
	let req = ::http::Request::builder()
		.uri("/v1/chat/completions")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"messages": [{"role": "user", "content": "hello"}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		llm_request,
		..
	} = provider
		.process_completions_request(&backend_info, Some(&policy), req, false, &mut None, None)
		.await
		.expect("OpenAI completions request should process")
	else {
		panic!("expected forwarded request");
	};

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");

	assert_eq!(forwarded_json["model"], json!("transformed-model"));
	assert_eq!(llm_request.request_model, "transformed-model");
}

#[tokio::test]
async fn copilot_anthropic_model_uses_messages_route() {
	use crate::http::auth::BackendInfo;
	use crate::test_helpers::proxymock::setup_proxy_test;
	use crate::types::agent::BackendTarget;

	let provider = AIProvider::Copilot(copilot::Provider { model: None });
	let inputs = setup_proxy_test("{}").unwrap().pi;
	let backend_info = BackendInfo {
		target: BackendTarget::Invalid,
		call_target: Target::from(("api.githubcopilot.com", 443)),
		inputs,
	};
	let req = ::http::Request::builder()
		.uri("/v1/messages")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{
				"model": "claude-sonnet-4",
				"max_tokens": 64,
				"messages": [{"role": "user", "content": "say hi"}]
			}"#
				.to_vec(),
		))
		.unwrap();

	let RequestResult::Success {
		request: forwarded,
		llm_request,
		upstream_route_type,
		..
	} = provider
		.process_messages_request(&backend_info, None, req, false, &mut None, None)
		.await
		.expect("Copilot Anthropic messages request should process")
	else {
		panic!("expected forwarded request");
	};

	assert_eq!(upstream_route_type, RouteType::Messages);
	assert_eq!(
		llm_request.cache_convention,
		CacheTokenConvention::InputExcludesCache
	);

	let mut setup_req =
		crate::http::tests_common::request("https://example.com/v1/messages", http::Method::POST, &[]);
	provider
		.setup_request(
			&mut setup_req,
			upstream_route_type,
			Some(&llm_request),
			None,
			None,
			false,
		)
		.expect("setup_request should succeed");
	assert_eq!(setup_req.uri().path(), "/v1/messages");

	let forwarded_body = forwarded.collect().await.unwrap().to_bytes();
	let forwarded_json: Value =
		serde_json::from_slice(&forwarded_body).expect("forwarded request should be JSON");
	assert_eq!(forwarded_json["model"], json!("claude-sonnet-4"));
	assert_eq!(forwarded_json["max_tokens"], json!(64));
}

#[test]
fn copilot_embeddings_response_adds_missing_openai_fields() {
	let provider = AIProvider::Copilot(copilot::Provider { model: None });
	let mut request = llm_request_with_tokens(None);
	request.input_format = InputFormat::Embeddings;
	request.request_model = "text-embedding-3-small".into();
	let response = Bytes::from_static(
		br#"{"data":[{"embedding":[0.5,-0.25],"index":0,"object":"embedding"}],"usage":{"prompt_tokens":2,"total_tokens":2}}"#,
	);

	let (llm_response, body) = provider
		.process_embeddings_response(&request, &::http::HeaderMap::new(), response)
		.expect("Copilot embeddings response should normalize");
	let body: Value = serde_json::from_slice(&body).expect("normalized response should be JSON");

	assert_eq!(body["object"], json!("list"));
	assert_eq!(body["model"], json!("text-embedding-3-small"));
	assert_eq!(body["data"][0]["embedding"], json!([0.5, -0.25]));
	assert_eq!(llm_response.input_tokens, Some(2));
	assert_eq!(llm_response.total_tokens, Some(2));
}

#[test]
fn copilot_embeddings_response_preserves_missing_usage() {
	let provider = AIProvider::Copilot(copilot::Provider { model: None });
	let mut request = llm_request_with_tokens(None);
	request.input_format = InputFormat::Embeddings;
	request.request_model = "text-embedding-3-small".into();
	let response =
		Bytes::from_static(br#"{"data":[{"embedding":[0.5],"index":0,"object":"embedding"}]}"#);

	let (_, body) = provider
		.process_embeddings_response(&request, &::http::HeaderMap::new(), response)
		.expect("Copilot embeddings response should normalize");
	let body: Value = serde_json::from_slice(&body).expect("normalized response should be JSON");

	assert!(body.get("usage").is_none());
}

#[test]
fn copilot_embeddings_response_preserves_explicit_openai_fields() {
	let provider = AIProvider::Copilot(copilot::Provider { model: None });
	let mut request = llm_request_with_tokens(None);
	request.input_format = InputFormat::Embeddings;
	request.request_model = "requested-model".into();
	let response = Bytes::from_static(
		br#"{"object":"upstream-list","model":"upstream-model","data":[{"embedding":[0.5],"index":0,"object":"embedding"}],"usage":{"prompt_tokens":1,"total_tokens":1}}"#,
	);

	let (_, body) = provider
		.process_embeddings_response(&request, &::http::HeaderMap::new(), response)
		.expect("Copilot embeddings response should preserve explicit fields");
	let body: Value = serde_json::from_slice(&body).expect("normalized response should be JSON");

	assert_eq!(body["object"], json!("upstream-list"));
	assert_eq!(body["model"], json!("upstream-model"));
}

#[test]
fn copilot_embeddings_parse_error_logs_normalized_response() {
	#[derive(Clone)]
	struct LogWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

	impl std::io::Write for LogWriter {
		fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
			self.0.lock().unwrap().extend_from_slice(buf);
			Ok(buf.len())
		}

		fn flush(&mut self) -> std::io::Result<()> {
			Ok(())
		}
	}

	let logs = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
	let writer = LogWriter(logs.clone());
	let subscriber = tracing_subscriber::fmt()
		.with_ansi(false)
		.without_time()
		.with_writer(move || writer.clone())
		.finish();

	tracing::subscriber::with_default(subscriber, || {
		let provider = AIProvider::Copilot(copilot::Provider { model: None });
		let mut request = llm_request_with_tokens(None);
		request.input_format = InputFormat::Embeddings;
		request.request_model = "text-embedding-3-small".into();
		let response = Bytes::from_static(br#"{"usage":"invalid"}"#);

		assert!(
			provider
				.process_embeddings_response(&request, &::http::HeaderMap::new(), response)
				.is_err()
		);
	});

	let logs = String::from_utf8(logs.lock().unwrap().clone()).unwrap();
	assert!(logs.contains(r#""object":"list""#), "{logs}");
	assert!(
		logs.contains(r#""model":"text-embedding-3-small""#),
		"{logs}"
	);
}

#[test]
fn non_copilot_embeddings_response_still_requires_openai_fields() {
	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let mut request = llm_request_with_tokens(None);
	request.input_format = InputFormat::Embeddings;
	let response = Bytes::from_static(
		br#"{"data":[{"embedding":[0.5],"index":0,"object":"embedding"}],"usage":{"prompt_tokens":1,"total_tokens":1}}"#,
	);

	assert!(
		provider
			.process_embeddings_response(&request, &::http::HeaderMap::new(), response)
			.is_err()
	);
}

#[test]
fn openai_token_limit_normalization_keeps_explicit_max_completion_tokens() {
	let mut request: types::completions::Request = serde_json::from_value(json!({
		"model": "gpt-5.4",
		"max_tokens": 1024,
		"max_completion_tokens": 2048,
		"messages": [{"role": "user", "content": "hello"}]
	}))
	.expect("valid completions request");

	request.normalize_openai_token_limit();

	assert_eq!(request.max_tokens, None);
	assert_eq!(request.max_completion_tokens, Some(2048));
}

#[test]
fn test_adaptive_thinking_without_effort_maps_to_high_reasoning_effort() {
	let request: types::messages::Request = serde_json::from_value(json!({
		"model": "claude-opus-4-6",
		"max_tokens": 256,
		"thinking": {
			"type": "adaptive"
		},
		"messages": [
			{
				"role": "user",
				"content": "Give one concise insight."
			}
		]
	}))
	.expect("valid messages request");

	let translated = conversion::completions::from_messages::translate(&request)
		.expect("messages->completions translation");
	let translated: Value =
		serde_json::from_slice(&translated).expect("translated request should be valid json");

	assert_eq!(translated.get("reasoning_effort"), Some(&json!("high")));
}

#[test]
fn test_completions_reasoning_effort_maps_to_enabled_thinking_budget() {
	let request: types::completions::Request = serde_json::from_value(json!({
		"model": "claude-opus-4-6",
		"messages": [
			{ "role": "user", "content": "Give one concise insight." }
		],
		"reasoning_effort": "minimal"
	}))
	.expect("valid completions request");

	let translated = conversion::messages::from_completions::translate(&request)
		.expect("completions->messages translation");
	let translated: Value =
		serde_json::from_slice(&translated).expect("translated request should be valid json");

	assert_eq!(
		translated["thinking"],
		json!({
			"type": "enabled",
			"budget_tokens": 1024
		})
	);
	assert!(translated.get("output_config").is_none());
}

#[test]
fn test_completions_json_schema_response_format_maps_to_anthropic_output_config() {
	let request: types::completions::Request = serde_json::from_value(json!({
		"model": "claude-opus-4-6",
		"messages": [
			{ "role": "user", "content": "Return one short summary." }
		],
		"response_format": {
			"type": "json_schema",
			"json_schema": {
				"name": "summary_schema",
				"schema": {
					"type": "object",
					"properties": { "summary": { "type": "string" } },
					"required": ["summary"],
					"additionalProperties": false
				}
			}
		}
	}))
	.expect("valid completions request");

	let translated = conversion::messages::from_completions::translate(&request)
		.expect("completions->messages translation");
	let translated: Value =
		serde_json::from_slice(&translated).expect("translated request should be valid json");

	assert_eq!(
		translated["output_config"]["format"],
		json!({
			"type": "json_schema",
			"schema": {
				"type": "object",
				"properties": { "summary": { "type": "string" } },
				"required": ["summary"],
				"additionalProperties": false
			}
		})
	);
}

#[test]
fn test_messages_output_config_format_maps_to_openai_response_format() {
	let request: types::messages::Request = serde_json::from_value(json!({
		"model": "claude-opus-4-6",
		"max_tokens": 256,
		"output_config": {
			"format": {
				"type": "json_schema",
				"schema": {
					"type": "object",
					"properties": { "answer": { "type": "number" } },
					"required": ["answer"],
					"additionalProperties": false
				}
			}
		},
		"messages": [
			{
				"role": "user",
				"content": "What is 2+2?"
			}
		]
	}))
	.expect("valid messages request");

	let translated = conversion::completions::from_messages::translate(&request)
		.expect("messages->completions translation");
	let translated: Value =
		serde_json::from_slice(&translated).expect("translated request should be valid json");

	assert_eq!(translated["response_format"]["type"], json!("json_schema"));
	assert_eq!(
		translated["response_format"]["json_schema"]["name"],
		json!("structured_output")
	);
	assert_eq!(
		translated["response_format"]["json_schema"]["schema"],
		json!({
			"type": "object",
			"properties": { "answer": { "type": "number" } },
			"required": ["answer"],
			"additionalProperties": false
		})
	);
}

/// Verifies that `process_response` routes a non-success response through
/// the buffered error path even when the request has `streaming: true`.
///
/// Constructs a Bedrock 400 JSON error response and passes it through
/// `process_response` with a streaming `LLMRequest`. Asserts the returned
/// body is non-empty, valid JSON, and preserves the original error message.
#[tokio::test]
async fn process_response_routes_streaming_error_to_buffered_path() {
	use crate::proxy::httpproxy::PolicyClient;
	use crate::test_helpers::proxymock::setup_proxy_test;

	let bedrock = AIProvider::bedrock(bedrock::Provider {
		model: Some(strng::new("anthropic.claude-3-5-sonnet-20241022-v2:0")),
		region: strng::new("us-west-2"),
		guardrail_identifier: None,
		guardrail_version: None,
	});

	let error_json = r#"{"message":"Expected toolResult blocks at messages.2.content for the following Ids: tooluse_abc123"}"#;

	let req = LLMRequest {
		input_tokens: None,
		input_format: InputFormat::Completions,
		cache_convention: CacheTokenConvention::pending(),
		request_model: "input-model".into(),
		provider: Default::default(),
		streaming: true,
		params: Default::default(),
		prompt: None,
		provider_state: None,
	};

	let body = Body::from(error_json.as_bytes().to_vec());
	let mut resp = Response::new(body);
	*resp.status_mut() = ::http::StatusCode::BAD_REQUEST;
	resp.headers_mut().insert(
		::http::header::CONTENT_TYPE,
		"application/json".parse().unwrap(),
	);

	let client = PolicyClient::new(setup_proxy_test("{}").unwrap().pi);

	let result = bedrock
		.process_response(
			client,
			req,
			LLMResponsePolicies::default(),
			None,
			Default::default(),
			None,
			resp.into(),
		)
		.await
		.expect("process_response should succeed for error responses");

	assert_eq!(result.status(), ::http::StatusCode::BAD_REQUEST);

	let result_body = result.collect().await.unwrap().to_bytes();
	assert!(
		!result_body.is_empty(),
		"error response body must not be empty",
	);

	let parsed: Value =
		serde_json::from_slice(&result_body).expect("translated error should be valid JSON");

	let message = parsed
		.pointer("/error/message")
		.and_then(|v| v.as_str())
		.unwrap_or_default();
	assert!(
		message.contains("toolResult"),
		"translated error should preserve the original message, got: {message}",
	);
}

fn buffered_responses_request(model: &str) -> LLMRequest {
	LLMRequest {
		input_tokens: None,
		input_format: InputFormat::Responses,
		cache_convention: CacheTokenConvention::pending(),
		request_model: model.into(),
		provider: "test-provider".into(),
		streaming: false,
		params: Default::default(),
		prompt: None,
		provider_state: None,
	}
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct IntermediateRoundExtension(&'static str);

#[tokio::test]
async fn translate_buffered_responses_round_preserves_native_upstream_for_final_processing() {
	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let req = buffered_responses_request("gpt-4.1-mini");
	let upstream_body = fs::read(fixture_path("response/responses/tool.json"))
		.expect("Failed to read Responses tool-call fixture");
	let mut upstream = ::http::Response::builder()
		.status(::http::StatusCode::ACCEPTED)
		.header(::http::header::CONTENT_TYPE, "application/json")
		.header("x-upstream-round", "native-responses")
		.header(::http::header::CONTENT_LENGTH, upstream_body.len())
		.body(Body::from(upstream_body.clone()))
		.unwrap();
	upstream
		.extensions_mut()
		.insert(IntermediateRoundExtension("preserve-across-translation"));

	let round = provider
		.translate_buffered_responses_round(&req, RouteType::Responses, upstream)
		.await
		.expect("native Responses tool call should translate");

	assert_eq!(round.response.id, "resp_tool");
	assert_eq!(round.response.status, "completed");
	let types::responses::typed::OutputItem::Message(message) = &round.response.output[0] else {
		panic!("expected native message output before the function call");
	};
	assert_eq!(message.id, "msg_text");
	let [types::responses::typed::OutputMessageContent::OutputText(text)] =
		message.content.as_slice()
	else {
		panic!("expected the native message's single output-text item");
	};
	assert_eq!(text.text, "I will check that.");
	let types::responses::typed::OutputItem::FunctionCall(call) = &round.response.output[1] else {
		panic!("expected native function call output");
	};
	assert_eq!(call.call_id, "call_weather");
	assert_eq!(call.name, "get_weather");
	assert_eq!(call.arguments, r#"{"location":"San Francisco"}"#);

	assert_eq!(
		round.reconstructed_upstream.status(),
		::http::StatusCode::ACCEPTED
	);
	assert_eq!(
		round.reconstructed_upstream.headers()["x-upstream-round"],
		"native-responses"
	);
	assert_eq!(
		round.reconstructed_upstream.headers()[::http::header::CONTENT_TYPE],
		"application/json"
	);
	assert!(
		!round
			.reconstructed_upstream
			.headers()
			.contains_key(::http::header::CONTENT_LENGTH)
	);
	assert!(
		round
			.reconstructed_upstream
			.extensions()
			.get::<crate::cel::LLMContext>()
			.is_none(),
		"intermediate translation must not run final response side effects"
	);
	assert_eq!(
		round
			.reconstructed_upstream
			.extensions()
			.get::<IntermediateRoundExtension>(),
		Some(&IntermediateRoundExtension("preserve-across-translation")),
		"intermediate translation must preserve arbitrary upstream response extensions"
	);
	let reconstructed_body = round
		.reconstructed_upstream
		.into_body()
		.collect()
		.await
		.unwrap()
		.to_bytes();
	assert_eq!(reconstructed_body.as_ref(), upstream_body.as_slice());
}

#[tokio::test]
async fn translate_buffered_responses_round_converts_openai_completions_tool_calls() {
	// A catalog can pin a Copilot model to Completions even when its built-in heuristic would
	// select Responses. Intermediate translation must honor the already-selected upstream route.
	let provider = AIProvider::Copilot(copilot::Provider { model: None });
	let req = buffered_responses_request("grok-2");
	let upstream_body = fs::read(fixture_path("response/completions/tool_call.json"))
		.expect("Failed to read Completions tool-call fixture");
	let upstream = ::http::Response::builder()
		.header("x-upstream-round", "openai-completions")
		.body(Body::from(upstream_body.clone()))
		.unwrap();

	let round = provider
		.translate_buffered_responses_round(&req, RouteType::Completions, upstream)
		.await
		.expect("OpenAI Completions tool calls should translate to Responses");

	assert_eq!(round.response.status, "completed");
	assert_eq!(round.response.output.len(), 2);
	let types::responses::typed::OutputItem::FunctionCall(first) = &round.response.output[0] else {
		panic!("expected first converted function call");
	};
	assert_eq!(first.call_id, "call_abc123");
	assert_eq!(first.name, "get_weather");
	let types::responses::typed::OutputItem::FunctionCall(second) = &round.response.output[1] else {
		panic!("expected second converted function call");
	};
	assert_eq!(second.call_id, "call_xyz789");
	assert_eq!(second.name, "search_web");
	assert_eq!(
		round.reconstructed_upstream.headers()["x-upstream-round"],
		"openai-completions"
	);
	assert!(
		round
			.reconstructed_upstream
			.extensions()
			.get::<crate::cel::LLMContext>()
			.is_none()
	);
	let reconstructed_body = round
		.reconstructed_upstream
		.into_body()
		.collect()
		.await
		.unwrap()
		.to_bytes();
	assert_eq!(reconstructed_body.as_ref(), upstream_body.as_slice());
}

#[tokio::test]
async fn translate_buffered_responses_round_converts_bedrock_converse_tool_calls() {
	let provider = AIProvider::bedrock(bedrock::Provider {
		model: Some(strng::new("anthropic.claude-3-5-sonnet-20241022-v2:0")),
		region: strng::new("us-west-2"),
		guardrail_identifier: None,
		guardrail_version: None,
	});
	let req = buffered_responses_request("anthropic.claude-3-5-sonnet-20241022-v2:0");
	let upstream_body = fs::read(fixture_path("response/bedrock/tool.json"))
		.expect("Failed to read Bedrock Converse tool-call fixture");
	let upstream = ::http::Response::builder()
		.header("x-upstream-round", "bedrock-converse")
		.body(Body::from(upstream_body.clone()))
		.unwrap();

	let round = provider
		.translate_buffered_responses_round(&req, RouteType::Responses, upstream)
		.await
		.expect("Bedrock Converse tool calls should translate to Responses");

	assert_eq!(round.response.status, "completed");
	assert_eq!(round.response.output.len(), 2);
	let types::responses::typed::OutputItem::FunctionCall(first) = &round.response.output[0] else {
		panic!("expected first Bedrock function call");
	};
	assert_eq!(first.call_id, "tooluse_kZJMlvQmRJ6eAyJE5GIl7Q");
	assert_eq!(first.name, "top_song");
	assert_eq!(first.arguments, r#"{"sign":"WZPZ"}"#);
	let types::responses::typed::OutputItem::FunctionCall(second) = &round.response.output[1] else {
		panic!("expected second Bedrock function call");
	};
	assert_eq!(second.call_id, "tooluse_kZJMlvQmRJ6eAyxxx");
	assert_eq!(second.name, "hello");
	assert_eq!(second.arguments, r#"{"sign":"world"}"#);
	assert_eq!(
		round.reconstructed_upstream.headers()["x-upstream-round"],
		"bedrock-converse"
	);
	assert!(
		round
			.reconstructed_upstream
			.extensions()
			.get::<crate::cel::LLMContext>()
			.is_none()
	);
	let reconstructed_body = round
		.reconstructed_upstream
		.into_body()
		.collect()
		.await
		.unwrap()
		.to_bytes();
	assert_eq!(reconstructed_body.as_ref(), upstream_body.as_slice());
}

#[tokio::test]
async fn upstream_encoding_is_applied_after_messages_response_translation() {
	use crate::proxy::httpproxy::PolicyClient;
	use crate::test_helpers::proxymock::setup_proxy_test;

	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let mut req = llm_request_with_tokens(None);
	req.input_format = InputFormat::Messages;
	req.request_model = "gpt-4o".into();
	req.streaming = false;
	let upstream_body = br#"{"id":"chatcmpl-1","object":"chat.completion","created":0,"model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Hello!"}}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#;
	let compressed = crate::http::compression::encode_body(upstream_body, "br")
		.await
		.unwrap();
	let upstream = ::http::Response::builder()
		.header(::http::header::CONTENT_ENCODING, "br")
		.header(::http::header::CONTENT_LENGTH, compressed.len())
		.body(Body::from(compressed))
		.unwrap();

	let response = provider
		.process_response(
			PolicyClient::new(setup_proxy_test("{}").unwrap().pi),
			req,
			LLMResponsePolicies::default(),
			None,
			Default::default(),
			None,
			upstream.into(),
		)
		.await
		.unwrap();

	// Keep the response plain while later response policies can still replace its body.
	assert!(
		!response
			.headers()
			.contains_key(::http::header::CONTENT_ENCODING)
	);
	assert!(
		response
			.extensions()
			.get::<crate::cel::BufferedBody>()
			.is_none()
	);
	let (parts, body) = response.into_parts();
	let mut body: Value = serde_json::from_slice(&body.collect().await.unwrap().to_bytes()).unwrap();
	assert_eq!(body["type"], "message");
	assert_eq!(body["content"][0]["text"], "Hello!");
	// Stand in for a later response-body policy mutation.
	body["policy_applied"] = true.into();
	let mut response = Response::from_parts(parts, Body::from(serde_json::to_vec(&body).unwrap()));

	encode_deferred_response(&mut response);
	assert_eq!(response.headers()[::http::header::CONTENT_ENCODING], "br");
	assert!(
		!response
			.headers()
			.contains_key(::http::header::CONTENT_LENGTH)
	);
	let content_encoding = response.headers().typed_get::<ContentEncoding>();
	let (_, body) = crate::http::compression::to_bytes_with_decompression(
		response.into_body(),
		content_encoding.as_ref(),
		1024 * 1024,
	)
	.await
	.unwrap();
	let body: Value = serde_json::from_slice(&body).unwrap();
	assert_eq!(body["type"], "message");
	assert_eq!(body["policy_applied"], true);
}

async fn managed_terminal_response(client_streaming: bool) -> Response {
	use crate::proxy::httpproxy::PolicyClient;
	use crate::test_helpers::proxymock::setup_proxy_test;

	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let req = LLMRequest {
		input_tokens: None,
		input_format: InputFormat::Responses,
		cache_convention: CacheTokenConvention::pending(),
		request_model: "gpt-test".into(),
		provider: "test-provider".into(),
		streaming: client_streaming,
		params: Default::default(),
		prompt: None,
		provider_state: None,
	};
	let typed_response = serde_json::from_value(json!({
		"id": "resp_typed",
		"status": "completed",
		"model": "gpt-test",
		"output": [
			{
				"type": "message",
				"id": "msg_typed_0",
				"role": "assistant",
				"status": "completed",
				"content": [{"type": "output_text", "text": "typed first", "annotations": []}]
			},
			{
				"type": "message",
				"id": "msg_typed_2",
				"role": "assistant",
				"status": "completed",
				"content": [{"type": "output_text", "text": "typed last", "annotations": []}]
			}
		],
		"tools": [{"type": "function", "name": "client_tool"}],
		"usage": {"input_tokens": 20, "output_tokens": 22, "total_tokens": 42}
	}))
	.unwrap();
	let raw_body = serde_json::to_vec(&json!({
		"id": "resp_raw",
		"status": "completed",
		"model": "gpt-test",
		"output": [{
			"type": "message",
			"id": "msg_raw",
			"role": "assistant",
			"status": "completed",
			"content": [{"type": "output_text", "text": "raw final", "annotations": []}]
		}],
		"tools": [{"type": "function", "name": "_agentgateway_internal"}],
		"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
	}))
	.unwrap();
	let raw_upstream = ::http::Response::builder()
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(raw_body))
		.unwrap();
	let managed = tool_runtime::ManagedFinalResponse {
		response: typed_response,
		raw_output: vec![
			json!({
				"type": "message",
				"id": "msg_raw_0",
				"role": "assistant",
				"status": "completed",
				"content": [{"type": "output_text", "text": "raw first", "annotations": []}]
			}),
			json!({
				"type": "future_output_item",
				"id": "future_1",
				"future": {"nested": [1, true, null]}
			}),
			json!({
				"type": "message",
				"id": "msg_raw_2",
				"role": "assistant",
				"status": "completed",
				"content": [{"type": "output_text", "text": "raw last", "annotations": []}]
			}),
		],
		raw_upstream,
		summary: tool_runtime::ToolRuntimeSummary {
			usage: None,
			rounds: 2,
			tool_calls: 1,
			client_streaming,
			include_obfuscation: false,
			client_tools: None,
		},
	};

	provider
		.process_response(
			PolicyClient::new(setup_proxy_test("{}").unwrap().pi),
			req,
			LLMResponsePolicies::default(),
			None,
			Default::default(),
			None,
			ResponseProcessingInput::Managed(Box::new(managed)),
		)
		.await
		.unwrap()
}

#[tokio::test]
async fn managed_json_terminal_round_preserves_unknown_output_item_position() {
	let response = managed_terminal_response(false).await;
	let body: Value = serde_json::from_slice(&response.collect().await.unwrap().to_bytes()).unwrap();
	assert_eq!(body["output"][0]["id"], "msg_typed_0");
	assert_eq!(body["output"][1]["type"], "future_output_item");
	assert_eq!(
		body["output"][1]["future"],
		json!({"nested": [1, true, null]})
	);
	assert_eq!(body["output"][2]["id"], "msg_typed_2");
	assert_eq!(body["tools"][0]["name"], "client_tool");
	assert_eq!(body["usage"]["total_tokens"], 42);
	assert!(!body.to_string().contains("raw first"));
}

#[tokio::test]
async fn managed_sse_terminal_round_preserves_unknown_output_item_position() {
	let response = managed_terminal_response(true).await;

	assert_eq!(
		response.headers()[::http::header::CONTENT_TYPE],
		"text/event-stream"
	);
	let body = String::from_utf8(response.collect().await.unwrap().to_bytes().to_vec()).unwrap();
	assert!(body.contains("event: response.completed\n"), "{body}");
	assert!(body.contains("typed first"), "{body}");
	assert!(body.contains("typed last"), "{body}");
	assert!(body.contains("future_output_item"), "{body}");
	assert!(body.contains("client_tool"), "{body}");
	assert!(body.contains(r#""total_tokens":42"#), "{body}");
	assert!(!body.contains("raw first"), "{body}");
	assert!(!body.contains("_agentgateway_internal"), "{body}");
	let terminal = body
		.split("\n\n")
		.find(|frame| frame.starts_with("event: response.completed\n"))
		.unwrap();
	let data = terminal
		.lines()
		.find_map(|line| line.strip_prefix("data: "))
		.unwrap();
	let event: Value = serde_json::from_str(data).unwrap();
	assert_eq!(event["response"]["output"][0]["id"], "msg_typed_0");
	assert_eq!(event["response"]["output"][1]["type"], "future_output_item");
	assert_eq!(event["response"]["output"][2]["id"], "msg_typed_2");
}

#[test]
fn openai_completions_error_translates_to_messages_client() {
	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let mut req = llm_request_with_tokens(None);
	req.input_format = InputFormat::Messages;
	req.request_model = "gpt-4o".into();

	let error = Bytes::from_static(
		br#"{"error":{"message":"bad request","type":"invalid_request_error","param":null,"code":400}}"#,
	);
	let translated = provider
		.process_error(&req, ::http::StatusCode::BAD_REQUEST, &error, None)
		.expect("OpenAI error should translate to messages error");
	let body: Value = serde_json::from_slice(&translated).expect("translated error should be JSON");

	assert_eq!(body["type"], json!("error"));
	assert_eq!(body["error"]["type"], json!("invalid_request_error"));
	assert_eq!(body["error"]["message"], json!("bad request"));
}

#[test]
fn custom_messages_error_translates_to_completions_client() {
	let provider = custom_provider(custom::ProviderFormat::Messages);
	let mut req = llm_request_with_tokens(None);
	req.input_format = InputFormat::Completions;
	req.request_model = "claude-test".into();

	let error = Bytes::from_static(
		br#"{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}"#,
	);
	let translated = provider
		.process_error(&req, ::http::StatusCode::BAD_REQUEST, &error, None)
		.expect("Anthropic error should translate to completions error");
	let body: Value = serde_json::from_slice(&translated).expect("translated error should be JSON");

	assert_eq!(body["error"]["type"], json!("invalid_request_error"));
	assert_eq!(body["error"]["message"], json!("bad request"));
}

#[test]
fn foundry_claude_messages_error_uses_anthropic_shape() {
	let provider = AIProvider::azure(azure::Provider {
		model: None,
		resource_name: strng::new("example"),
		resource_type: azure::AzureResourceType::Foundry,
		api_version: None,
		project_name: Some(strng::new("project")),
	});
	let mut req = llm_request_with_tokens(None);
	req.input_format = InputFormat::Messages;
	req.request_model = "claude-haiku-4-5".into();

	let error = Bytes::from_static(
		br#"{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}"#,
	);
	let translated = provider
		.process_error(&req, ::http::StatusCode::BAD_REQUEST, &error, None)
		.expect("Foundry Claude messages error should stay Anthropic-shaped");
	let body: Value = serde_json::from_slice(&translated).expect("translated error should be JSON");

	assert_eq!(body["type"], json!("error"));
	assert_eq!(body["error"]["type"], json!("invalid_request_error"));
	assert_eq!(body["error"]["message"], json!("bad request"));
}

#[tokio::test]
async fn process_streaming_bedrock_completions_normalizes_sse_headers_and_done() {
	use crate::proxy::httpproxy::PolicyClient;
	use crate::test_helpers::proxymock::setup_proxy_test;
	let bedrock = AIProvider::bedrock(bedrock::Provider {
		model: Some(strng::new("openai.gpt-oss-120b-1:0")),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	});

	let body = Body::from(
		fs::read(fixture_path("response/bedrock/basic.bin"))
			.expect("failed to read Bedrock streaming fixture"),
	);
	let mut resp = Response::new(body);
	resp.headers_mut().insert(
		::http::header::CONTENT_TYPE,
		"application/vnd.amazon.eventstream".parse().unwrap(),
	);
	resp.headers_mut().insert(
		crate::http::x_headers::X_AMZN_REQUESTID,
		"request_id".parse().unwrap(),
	);

	let client = PolicyClient::new(setup_proxy_test("{}").unwrap().pi);
	let translated = bedrock
		.process_streaming(
			client,
			LLMRequest {
				input_tokens: None,
				input_format: InputFormat::Completions,
				cache_convention: CacheTokenConvention::pending(),
				request_model: "input-model".into(),
				provider: Default::default(),
				streaming: true,
				params: Default::default(),
				prompt: None,
				provider_state: None,
			},
			LLMResponsePolicies::default(),
			None,
			Default::default(),
			None,
			resp,
		)
		.expect("Bedrock streaming translation should succeed");

	crate::http::tests_common::assert_header(
		&translated,
		::http::header::CONTENT_TYPE,
		"text/event-stream",
	);

	let body = translated.collect().await.unwrap().to_bytes();
	let text = String::from_utf8(body.to_vec()).expect("stream should be valid UTF-8");
	assert!(
		text.ends_with("data: [DONE]\n\n"),
		"translated Bedrock completions stream must end with [DONE], got:\n{text}",
	);
	assert!(
		!text.contains("event: \n"),
		"translated Bedrock completions stream must not emit empty event fields:\n{text}",
	);
}

#[test]
fn setup_request_openai_applies_prefixed_path_without_host_override() {
	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let mut req = crate::http::tests_common::request(
		"https://example.com/v1/messages?trace=repro",
		http::Method::POST,
		&[],
	);

	provider
		.setup_request(
			&mut req,
			RouteType::Messages,
			None,
			None,
			Some("/v1/custom"),
			false,
		)
		.expect("setup_request should succeed");

	assert_eq!(
		req.uri().authority().map(|a| a.as_str()),
		Some("api.openai.com")
	);
	assert_eq!(req.uri().path(), "/v1/custom/chat/completions");
	assert_eq!(req.uri().query(), Some("trace=repro"));
}

#[test]
fn setup_request_openai_normalizes_trailing_slash_in_path_prefix() {
	let provider = AIProvider::OpenAI(openai::Provider {
		model: None,
		moderation: None,
	});
	let mut req = crate::http::tests_common::request(
		"https://example.com/v1/messages?trace=repro",
		http::Method::POST,
		&[],
	);

	provider
		.setup_request(
			&mut req,
			RouteType::Messages,
			None,
			None,
			Some("/v1/custom/"),
			false,
		)
		.expect("setup_request should succeed");

	assert_eq!(req.uri().path(), "/v1/custom/chat/completions");
	assert_eq!(req.uri().query(), Some("trace=repro"));
}

#[test]
fn setup_request_custom_path_override_wins_over_format_path() {
	let provider = AIProvider::Custom(custom::Provider {
		model: None,
		provider_override: None,
		formats: vec![custom::ProviderFormatConfig {
			format: custom::ProviderFormat::Messages,
			path: Some(strng::literal!("/api/messages")),
		}],
	});
	let llm_request = LLMRequest {
		input_tokens: None,
		input_format: InputFormat::Completions,
		cache_convention: CacheTokenConvention::pending(),
		request_model: "input-model".into(),
		provider: Default::default(),
		streaming: false,
		params: Default::default(),
		prompt: None,
		provider_state: None,
	};
	let mut req = crate::http::tests_common::request(
		"https://proxy.example.com/v1/chat/completions?trace=repro",
		http::Method::POST,
		&[],
	);

	provider
		.setup_request(
			&mut req,
			RouteType::Completions,
			Some(&llm_request),
			Some("/override/messages"),
			None,
			true,
		)
		.expect("setup_request should succeed");

	assert_eq!(req.uri().path(), "/override/messages");
	assert_eq!(req.uri().query(), None);
}

#[test]
fn setup_request_custom_generate_content_defaults_to_the_native_path() {
	// A static configured path cannot carry the model or the streaming method, so the
	// default for the native Gemini chat format is the canonical Gemini API shape.
	let provider = custom_provider(custom::ProviderFormat::GenerateContent);
	for (streaming, expected_path, expected_query) in [
		(
			false,
			"/v1beta/models/gemini-2.5-flash:generateContent",
			None,
		),
		(
			true,
			"/v1beta/models/gemini-2.5-flash:streamGenerateContent",
			Some("alt=sse"),
		),
	] {
		let llm_request = LLMRequest {
			input_tokens: None,
			input_format: InputFormat::Gemini,
			cache_convention: CacheTokenConvention::pending(),
			request_model: "gemini-2.5-flash".into(),
			provider: Default::default(),
			streaming,
			params: Default::default(),
			prompt: None,
			provider_state: Some(ProviderState::VertexGemini),
		};
		let mut req = crate::http::tests_common::request(
			"https://gemini.example.com/v1beta/models/gemini-2.5-flash:generateContent",
			http::Method::POST,
			&[],
		);

		provider
			.setup_request(
				&mut req,
				RouteType::GenerateContent,
				Some(&llm_request),
				None,
				None,
				false,
			)
			.expect("setup_request should succeed");

		assert_eq!(req.uri().path(), expected_path, "streaming={streaming}");
		assert_eq!(req.uri().query(), expected_query, "streaming={streaming}");
	}
}

#[test]
fn setup_request_custom_count_tokens_defaults_to_the_native_path() {
	// Regression: with no configured path, countTokens used to fall through to the
	// OpenAI default and land on /v1/chat/completions.
	let provider = custom_provider(custom::ProviderFormat::GeminiCountTokens);
	let llm_request = LLMRequest {
		input_tokens: None,
		input_format: InputFormat::GeminiCountTokens,
		cache_convention: CacheTokenConvention::pending(),
		request_model: "gemini-2.5-flash".into(),
		provider: Default::default(),
		streaming: false,
		params: Default::default(),
		prompt: None,
		provider_state: None,
	};
	let mut req = crate::http::tests_common::request(
		"https://gemini.example.com/v1beta/models/gemini-2.5-flash:countTokens",
		http::Method::POST,
		&[],
	);

	provider
		.setup_request(
			&mut req,
			RouteType::GeminiCountTokens,
			Some(&llm_request),
			None,
			None,
			false,
		)
		.expect("setup_request should succeed");

	assert_eq!(
		req.uri().path(),
		"/v1beta/models/gemini-2.5-flash:countTokens"
	);
	assert_eq!(req.uri().query(), None);
}

fn llm_request_for_path(request_model: &str) -> LLMRequest {
	LLMRequest {
		input_tokens: None,
		input_format: InputFormat::Messages,
		cache_convention: CacheTokenConvention::pending(),
		request_model: request_model.into(),
		provider: Default::default(),
		streaming: false,
		params: Default::default(),
		prompt: None,
		provider_state: None,
	}
}

fn assert_prefixed_host_override_path(
	provider: AIProvider,
	request_model: &str,
	expected_path: &str,
	expected_query: Option<&str>,
) {
	let llm_request = llm_request_for_path(request_model);
	let mut req = crate::http::tests_common::request(
		"https://proxy.example.com/v1/messages?trace=repro",
		http::Method::POST,
		&[],
	);

	provider
		.setup_request(
			&mut req,
			RouteType::Messages,
			Some(&llm_request),
			None,
			Some("/proxy/"),
			true,
		)
		.expect("setup_request should succeed");

	assert_eq!(req.uri().path(), expected_path);
	assert_eq!(req.uri().query(), expected_query);
}

fn native_gemini_llm_request(request_model: &str, streaming: bool) -> LLMRequest {
	LLMRequest {
		input_tokens: None,
		input_format: InputFormat::Gemini,
		cache_convention: CacheTokenConvention::pending(),
		request_model: request_model.into(),
		provider: Default::default(),
		streaming,
		params: Default::default(),
		prompt: None,
		provider_state: Some(ProviderState::VertexGemini),
	}
}

#[test]
fn setup_request_gemini_native_builds_generate_content_path() {
	let provider = AIProvider::Gemini(gemini::Provider { model: None });
	let llm_request = native_gemini_llm_request("gemini-2.5-flash", false);
	let mut req = crate::http::tests_common::request(
		"https://example.com/v1beta/models/gemini-2.5-flash:generateContent",
		http::Method::POST,
		&[],
	);

	provider
		.setup_request(
			&mut req,
			RouteType::Completions,
			Some(&llm_request),
			None,
			None,
			false,
		)
		.expect("setup_request should succeed");

	assert_eq!(
		req.uri().authority().map(|a| a.as_str()),
		Some("generativelanguage.googleapis.com")
	);
	assert_eq!(
		req.uri().path(),
		"/v1beta/models/gemini-2.5-flash:generateContent"
	);
	assert_eq!(req.uri().query(), None);
}

#[test]
fn setup_request_gemini_native_streaming_adds_alt_sse_and_keeps_client_query() {
	let provider = AIProvider::Gemini(gemini::Provider { model: None });
	// The client's own alt=sse is dropped in favour of the path-provided one, while any other
	// parameter such as key survives.
	let llm_request = native_gemini_llm_request("models/gemini-2.5-flash", true);
	let mut req = crate::http::tests_common::request(
		"https://example.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=abc",
		http::Method::POST,
		&[],
	);

	provider
		.setup_request(
			&mut req,
			RouteType::Completions,
			Some(&llm_request),
			None,
			None,
			false,
		)
		.expect("setup_request should succeed");

	assert_eq!(
		req.uri().path(),
		"/v1beta/models/gemini-2.5-flash:streamGenerateContent"
	);
	assert_eq!(req.uri().query(), Some("alt=sse&key=abc"));
}

#[test]
fn setup_request_gemini_native_streaming_keeps_client_alt_with_host_override() {
	// hostOverride without pathPrefix forwards the client's URI verbatim, so alt=sse must
	// still be there: the upstream would otherwise answer with the JSON-array variant.
	for provider in [
		AIProvider::Gemini(gemini::Provider { model: None }),
		AIProvider::Vertex(vertex::Provider {
			model: None,
			region: None,
			project_id: strng::new("test-project"),
		}),
	] {
		let llm_request = native_gemini_llm_request("gemini-2.5-flash", true);
		let mut req = crate::http::tests_common::request(
			"https://proxy.example.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
			http::Method::POST,
			&[],
		);

		provider
			.setup_request(
				&mut req,
				RouteType::Completions,
				Some(&llm_request),
				None,
				None,
				true,
			)
			.expect("setup_request should succeed");

		assert_eq!(
			req.uri().path(),
			"/v1beta/models/gemini-2.5-flash:streamGenerateContent"
		);
		assert_eq!(req.uri().query(), Some("alt=sse"));
	}
}

#[test]
fn setup_request_gemini_without_native_state_keeps_compat_path() {
	let provider = AIProvider::Gemini(gemini::Provider { model: None });
	let llm_request = LLMRequest {
		provider_state: None,
		..native_gemini_llm_request("gemini-2.5-flash", false)
	};
	let mut req = crate::http::tests_common::request(
		"https://example.com/v1/chat/completions",
		http::Method::POST,
		&[],
	);

	provider
		.setup_request(
			&mut req,
			RouteType::Completions,
			Some(&llm_request),
			None,
			None,
			false,
		)
		.expect("setup_request should succeed");

	assert_eq!(req.uri().path(), "/v1beta/openai/chat/completions");
}

#[test]
fn setup_request_gemini_applies_path_prefix_with_host_override() {
	assert_prefixed_host_override_path(
		AIProvider::Gemini(gemini::Provider { model: None }),
		"gemini-2.5-pro",
		"/proxy/v1beta/openai/chat/completions",
		Some("trace=repro"),
	);
}

#[test]
fn setup_request_vertex_applies_path_prefix_with_host_override() {
	assert_prefixed_host_override_path(
		AIProvider::Vertex(vertex::Provider {
			model: None,
			region: Some(strng::new("us-central1")),
			project_id: strng::new("example-project"),
		}),
		"gemini-2.5-pro",
		"/proxy/v1/projects/example-project/locations/us-central1/endpoints/openapi/chat/completions",
		Some("trace=repro"),
	);
}

#[test]
fn setup_request_bedrock_applies_path_prefix_with_host_override() {
	assert_prefixed_host_override_path(
		AIProvider::bedrock(bedrock::Provider {
			model: None,
			region: strng::new("us-east-1"),
			guardrail_identifier: None,
			guardrail_version: None,
		}),
		"anthropic.claude-3-5-sonnet-20241022-v2:0",
		"/proxy/model/anthropic.claude-3-5-sonnet-20241022-v2:0/converse",
		Some("trace=repro"),
	);
}

#[test]
fn setup_request_bedrock_sets_signing_region_with_host_override() {
	let provider = AIProvider::bedrock(bedrock::Provider {
		model: None,
		region: strng::new("ca-central-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	});
	let mut req = crate::http::tests_common::request(
		"https://bedrock-vpce.example.com/model/example/converse",
		http::Method::POST,
		&[],
	);

	provider
		.setup_request(&mut req, RouteType::Messages, None, None, None, true)
		.expect("setup_request should succeed");

	assert_eq!(
		req.uri().authority().map(|authority| authority.as_str()),
		Some("bedrock-vpce.example.com")
	);
	assert_eq!(
		req
			.extensions()
			.get::<bedrock::AwsRegion>()
			.map(|region| region.region.as_str()),
		Some("ca-central-1")
	);
}

#[test]
fn setup_request_azure_applies_path_prefix_with_host_override() {
	assert_prefixed_host_override_path(
		AIProvider::azure(azure::Provider {
			model: None,
			resource_name: strng::new("example"),
			resource_type: azure::AzureResourceType::OpenAI,
			api_version: Some(strng::new("2024-02-15-preview")),
			project_name: None,
		}),
		"gpt-4.1",
		"/proxy/openai/deployments/gpt-4.1/chat/completions",
		Some("api-version=2024-02-15-preview&trace=repro"),
	);
}

#[test]
fn completions_response_missing_message_and_usage_fields() {
	// Gemini's OpenAI-compat endpoint can omit `message` from choices and
	// `completion_tokens` from usage. Verify deserialization succeeds with defaults.
	let json = r#"{
		"id": "1",
		"object": "chat.completion",
		"created": 0,
		"model": "google/gemini-2.5-flash",
		"choices": [{"index": 0, "finish_reason": "length"}],
		"usage": {"prompt_tokens": 5, "total_tokens": 12}
	}"#;
	let resp: types::completions::Response = serde_json::from_str(json).unwrap();
	assert_eq!(resp.choices.len(), 1);
	assert_eq!(resp.choices[0].message.content, None);
	assert_eq!(resp.choices[0].message.role, None);
	let usage = resp.usage.unwrap();
	assert_eq!(usage.prompt_tokens, 5);
	assert_eq!(usage.completion_tokens, 0);
	assert_eq!(usage.total_tokens, 12);
}

#[test]
fn completions_to_messages_response_allows_missing_openai_metadata() {
	let body = Bytes::from_static(
		br#"{
			"id": "chatcmpl-1",
			"model": "gpt-5-mini",
			"choices": [{
				"message": {"role": "assistant", "content": "hi"},
				"finish_reason": "stop"
			}],
			"usage": {
				"completion_tokens": 16,
				"prompt_tokens": 9,
				"prompt_tokens_details": {"cached_tokens": 0},
				"total_tokens": 25
			},
			"copilot_usage": {
				"token_details": []
			}
		}"#,
	);

	conversion::completions::from_messages::translate_response(&body)
		.expect("messages response translation should not require OpenAI metadata");
}

#[tokio::test]
async fn bedrock_from_messages_stream_captures_completion() {
	let input_bytes =
		fs::read(fixture_path("response/bedrock/basic.bin")).expect("Failed to read fixture");
	let body = Body::from(input_bytes);
	let log = AsyncLog::default();
	let log2 = log.clone();
	let llmresp = LLMInfo {
		request: LLMRequest {
			input_tokens: None,
			input_format: InputFormat::Messages,
			cache_convention: CacheTokenConvention::pending(),
			request_model: "us.anthropic.claude-haiku-4-5-20251001-v1:0".into(),
			provider: "bedrock".into(),
			streaming: true,
			params: Default::default(),
			prompt: None,
			provider_state: None,
		},
		response: LLMResponse::default(),
	};
	log.store(Some(llmresp));
	let logger = AmendOnDrop::new(log, LLMResponsePolicies::default(), None, None).into_llm();
	let buffer_limit = 1024 * 1024;
	let body = conversion::bedrock::from_messages::translate_stream(
		body,
		buffer_limit,
		logger,
		"us.anthropic.claude-haiku-4-5-20251001-v1:0",
		"msg_123",
		llm::LogContentFields {
			completion: true,
			tool_calls: true,
		},
		None,
	);
	let _ = body.collect().await.unwrap();
	let info = log2
		.take()
		.expect("log should have LLMInfo after stream completes");
	let completion = info
		.response
		.completion
		.expect("completion should be set for bedrock streaming");
	assert!(
		!completion.join("").is_empty(),
		"completion should contain response text"
	);
}

#[tokio::test]
async fn bedrock_from_messages_stream_skips_completion_when_disabled() {
	let input_bytes =
		fs::read(fixture_path("response/bedrock/basic.bin")).expect("Failed to read fixture");
	let body = Body::from(input_bytes);
	let log = AsyncLog::default();
	let log2 = log.clone();
	let llmresp = LLMInfo {
		request: LLMRequest {
			input_tokens: None,
			input_format: InputFormat::Messages,
			cache_convention: CacheTokenConvention::pending(),
			request_model: "us.anthropic.claude-haiku-4-5-20251001-v1:0".into(),
			provider: "bedrock".into(),
			streaming: true,
			params: Default::default(),
			prompt: None,
			provider_state: None,
		},
		response: LLMResponse::default(),
	};
	log.store(Some(llmresp));
	let logger = AmendOnDrop::new(log, LLMResponsePolicies::default(), None, None).into_llm();
	let buffer_limit = 1024 * 1024;
	let body = conversion::bedrock::from_messages::translate_stream(
		body,
		buffer_limit,
		logger,
		"us.anthropic.claude-haiku-4-5-20251001-v1:0",
		"msg_123",
		llm::LogContentFields::default(),
		None,
	);
	let _ = body.collect().await.unwrap();
	let info = log2
		.take()
		.expect("log should have LLMInfo after stream completes");
	assert!(
		info.response.completion.is_none(),
		"completion should not be set when log_content.completion is false"
	);
	assert!(
		info.response.output_messages.is_none(),
		"output messages should not be set when log_content.tool_calls is false"
	);
}

#[tokio::test]
async fn bedrock_from_messages_stream_captures_tool_calls() {
	let input_bytes =
		fs::read(fixture_path("response/bedrock/tool.bin")).expect("Failed to read fixture");
	let body = Body::from(input_bytes);
	let log = AsyncLog::default();
	let log2 = log.clone();
	let llmresp = LLMInfo {
		request: LLMRequest {
			input_tokens: None,
			input_format: InputFormat::Messages,
			cache_convention: CacheTokenConvention::pending(),
			request_model: "us.anthropic.claude-haiku-4-5-20251001-v1:0".into(),
			provider: "bedrock".into(),
			streaming: true,
			params: Default::default(),
			prompt: None,
			provider_state: None,
		},
		response: LLMResponse::default(),
	};
	log.store(Some(llmresp));
	let logger = AmendOnDrop::new(log, LLMResponsePolicies::default(), None, None).into_llm();
	let body = conversion::bedrock::from_messages::translate_stream(
		body,
		1024 * 1024,
		logger,
		"us.anthropic.claude-haiku-4-5-20251001-v1:0",
		"msg_123",
		llm::LogContentFields {
			completion: false,
			tool_calls: true,
		},
		None,
	);
	let _ = body.collect().await.unwrap();
	let info = log2
		.take()
		.expect("log should have LLMInfo after stream completes");
	assert!(info.response.completion.is_none());
	let output_messages = info
		.response
		.output_messages
		.expect("output messages should be set for Bedrock tool calls");
	assert_eq!(
		output_messages[0].finish_reason.as_deref(),
		Some("tool_use")
	);
	let tool_calls = output_messages[0].tool_calls();
	assert_eq!(tool_calls.len(), 2);
	assert_eq!(tool_calls[0].name.as_str(), "top_song");
	assert_eq!(tool_calls[0].arguments, serde_json::json!({"sign": "WZPZ"}));
	assert_eq!(tool_calls[1].name.as_str(), "hello");
	assert_eq!(
		tool_calls[1].arguments,
		serde_json::json!({"sign": "world"})
	);
}

#[tokio::test]
async fn messages_passthrough_stream_captures_completion() {
	let input_path = fixture_path("response/anthropic/stream_basic.json");
	let input_bytes = fs::read(&input_path).expect("Failed to read fixture");
	let body = Body::from(input_bytes);
	let log = AsyncLog::default();
	let log2 = log.clone();
	let llmresp = LLMInfo {
		request: LLMRequest {
			input_tokens: None,
			input_format: InputFormat::Messages,
			cache_convention: CacheTokenConvention::pending(),
			request_model: "claude-haiku-4-5-20251001".into(),
			provider: "anthropic".into(),
			streaming: true,
			params: Default::default(),
			prompt: None,
			provider_state: None,
		},
		response: LLMResponse::default(),
	};
	log.store(Some(llmresp));
	let logger = AmendOnDrop::new(log, LLMResponsePolicies::default(), None, None).into_llm();
	let buffer_limit = 1024 * 1024;
	let body = conversion::messages::passthrough_stream(
		body,
		buffer_limit,
		logger,
		llm::LogContentFields {
			completion: true,
			tool_calls: true,
		},
	);
	// Consume the body to drive the stream to completion
	let _ = body.collect().await.unwrap();
	let info = log2
		.take()
		.expect("log should have LLMInfo after stream completes");
	let completion = info
		.response
		.completion
		.expect("completion should be set for messages streaming");
	assert_eq!(
		completion.join(""),
		"Hi there! How are you doing today? Is there anything I can help you with?"
	);
}

#[tokio::test]
async fn messages_passthrough_stream_skips_completion_when_disabled() {
	let input_path = fixture_path("response/anthropic/stream_basic.json");
	let input_bytes = fs::read(&input_path).expect("Failed to read fixture");
	let body = Body::from(input_bytes);
	let log = AsyncLog::default();
	let log2 = log.clone();
	let llmresp = LLMInfo {
		request: LLMRequest {
			input_tokens: None,
			input_format: InputFormat::Messages,
			cache_convention: CacheTokenConvention::pending(),
			request_model: "claude-haiku-4-5-20251001".into(),
			provider: "anthropic".into(),
			streaming: true,
			params: Default::default(),
			prompt: None,
			provider_state: None,
		},
		response: LLMResponse::default(),
	};
	log.store(Some(llmresp));
	let logger = AmendOnDrop::new(log, LLMResponsePolicies::default(), None, None).into_llm();
	let buffer_limit = 1024 * 1024;
	let body = conversion::messages::passthrough_stream(
		body,
		buffer_limit,
		logger,
		llm::LogContentFields::default(),
	);
	let _ = body.collect().await.unwrap();
	let info = log2
		.take()
		.expect("log should have LLMInfo after stream completes");
	assert!(
		info.response.completion.is_none(),
		"completion should not be set when log_content.completion is false"
	);
	assert!(
		info.response.output_messages.is_none(),
		"output messages should not be set when log_content.tool_calls is false"
	);
}

#[tokio::test]
async fn messages_passthrough_stream_captures_tool_calls() {
	let input_bytes =
		fs::read(fixture_path("response/anthropic/stream_tool.json")).expect("Failed to read fixture");
	let body = Body::from(input_bytes);
	let log = AsyncLog::default();
	let log2 = log.clone();
	let llmresp = LLMInfo {
		request: LLMRequest {
			input_tokens: None,
			input_format: InputFormat::Messages,
			cache_convention: CacheTokenConvention::pending(),
			request_model: "claude-haiku-4-5-20251001".into(),
			provider: "anthropic".into(),
			streaming: true,
			params: Default::default(),
			prompt: None,
			provider_state: None,
		},
		response: LLMResponse::default(),
	};
	log.store(Some(llmresp));
	let logger = AmendOnDrop::new(log, LLMResponsePolicies::default(), None, None).into_llm();
	let body = conversion::messages::passthrough_stream(
		body,
		1024 * 1024,
		logger,
		llm::LogContentFields {
			completion: false,
			tool_calls: true,
		},
	);
	let _ = body.collect().await.unwrap();
	let info = log2
		.take()
		.expect("log should have LLMInfo after stream completes");
	assert!(info.response.completion.is_none());
	let output_messages = info
		.response
		.output_messages
		.expect("output messages should be set for Anthropic tool calls");
	assert_eq!(
		output_messages[0].finish_reason.as_deref(),
		Some("tool_use")
	);
	let tool_calls = output_messages[0].tool_calls();
	assert_eq!(tool_calls.len(), 1);
	assert_eq!(tool_calls[0].id.as_str(), "toolu_01A");
	assert_eq!(tool_calls[0].name.as_str(), "get_weather");
	assert_eq!(
		tool_calls[0].arguments,
		serde_json::json!({"location": "San Francisco"})
	);
}

#[tokio::test]
async fn responses_passthrough_stream_captures_completion_and_tool_calls() {
	let input_path = fixture_path("response/responses/stream.json");
	let input_bytes = fs::read(&input_path).expect("Failed to read fixture");
	let body = Body::from(input_bytes);
	let log = AsyncLog::default();
	let log2 = log.clone();
	let llmresp = LLMInfo {
		request: LLMRequest {
			input_tokens: None,
			input_format: InputFormat::Responses,
			cache_convention: CacheTokenConvention::pending(),
			request_model: "gpt-4.1-mini".into(),
			provider: "openai".into(),
			streaming: true,
			params: Default::default(),
			prompt: None,
			provider_state: None,
		},
		response: LLMResponse::default(),
	};
	log.store(Some(llmresp));
	let logger = AmendOnDrop::new(log, LLMResponsePolicies::default(), None, None).into_llm();
	let buffer_limit = 1024 * 1024;
	let body = conversion::responses::passthrough_stream(
		body,
		buffer_limit,
		logger,
		llm::LogContentFields {
			completion: true,
			tool_calls: true,
		},
	);
	let _ = body.collect().await.unwrap();
	let info = log2
		.take()
		.expect("log should have LLMInfo after stream completes");
	let completion = info
		.response
		.completion
		.expect("completion should be set for responses streaming");
	assert_eq!(completion.join(""), "Hello");
	let output_messages = info
		.response
		.output_messages
		.expect("output messages should be set for responses streaming");
	assert_eq!(
		output_messages[0].finish_reason.as_deref(),
		Some("completed")
	);
	let tool_calls = output_messages[0].tool_calls();
	assert_eq!(tool_calls.len(), 1);
	assert_eq!(tool_calls[0].id.as_str(), "call_xxx");
	assert_eq!(tool_calls[0].name.as_str(), "get_weather");
	assert_eq!(
		tool_calls[0].arguments,
		serde_json::json!({"location": "San Francisco"})
	);
}

#[tokio::test]
async fn responses_passthrough_stream_preserves_moderation_chunks() {
	let input_bytes = br#"event: response.completed
data: {"type":"response.completed","sequence_number":1,"response":{"created_at":123,"id":"resp_123","model":"gpt-5","object":"response","output":[],"status":"completed","moderation":{"input":{"flagged":false},"output":{"flagged":true}}}}

data: [DONE]

"#;
	let body = Body::from(input_bytes.to_vec());
	let log = AsyncLog::default();
	let llmresp = LLMInfo {
		request: LLMRequest {
			input_tokens: None,
			input_format: InputFormat::Responses,
			cache_convention: CacheTokenConvention::pending(),
			request_model: "gpt-5".into(),
			provider: "openai".into(),
			streaming: true,
			params: Default::default(),
			prompt: None,
			provider_state: None,
		},
		response: LLMResponse::default(),
	};
	log.store(Some(llmresp));
	let logger = AmendOnDrop::new(log, LLMResponsePolicies::default(), None, None).into_llm();
	let buffer_limit = 1024 * 1024;
	let body = conversion::responses::passthrough_stream(
		body,
		buffer_limit,
		logger,
		llm::LogContentFields::default(),
	);
	let output = body.collect().await.unwrap().to_bytes();
	let text = String::from_utf8(output.to_vec()).expect("stream should be valid UTF-8");

	assert!(text.contains(r#""moderation":{"input":{"flagged":false},"output":{"flagged":true}}"#));
	assert!(text.contains("data: [DONE]"));
}

#[tokio::test]
async fn responses_passthrough_stream_skips_completion_when_disabled() {
	let input_path = fixture_path("response/responses/stream.json");
	let input_bytes = fs::read(&input_path).expect("Failed to read fixture");
	let body = Body::from(input_bytes);
	let log = AsyncLog::default();
	let log2 = log.clone();
	let llmresp = LLMInfo {
		request: LLMRequest {
			input_tokens: None,
			input_format: InputFormat::Responses,
			cache_convention: CacheTokenConvention::pending(),
			request_model: "gpt-4.1-mini".into(),
			provider: "openai".into(),
			streaming: true,
			params: Default::default(),
			prompt: None,
			provider_state: None,
		},
		response: LLMResponse::default(),
	};
	log.store(Some(llmresp));
	let logger = AmendOnDrop::new(log, LLMResponsePolicies::default(), None, None).into_llm();
	let buffer_limit = 1024 * 1024;
	let body = conversion::responses::passthrough_stream(
		body,
		buffer_limit,
		logger,
		llm::LogContentFields::default(),
	);
	let _ = body.collect().await.unwrap();
	let info = log2
		.take()
		.expect("log should have LLMInfo after stream completes");
	assert!(
		info.response.completion.is_none(),
		"completion should not be set when log_content.completion is false"
	);
	assert!(
		info.response.output_messages.is_none(),
		"output messages should not be set when log_content.tool_calls is false"
	);
}

fn vertex_provider(model: &str) -> AIProvider {
	AIProvider::Vertex(vertex::Provider {
		model: Some(strng::new(model)),
		region: None,
		project_id: strng::new("test-project"),
	})
}

fn custom_provider(format: custom::ProviderFormat) -> AIProvider {
	AIProvider::Custom(custom::Provider {
		model: None,
		provider_override: None,
		formats: vec![custom::ProviderFormatConfig { format, path: None }],
	})
}

#[tokio::test]
async fn read_body_decodes_gzip_request_before_json_parse() {
	// Regression: a gzip-compressed request body (Content-Encoding: gzip) must be
	// decompressed before the JSON parse. Clients such as the Claude Code harness
	// gzip request bodies above a size threshold; previously the reader handed the
	// raw compressed bytes to serde_json and failed with a misleading
	// "LLM request body must be valid JSON" 400, even for tiny payloads.
	let provider = custom_provider(custom::ProviderFormat::Messages);

	let plaintext =
		br#"{"model":"claude-sonnet-4-5","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#;
	let gz = crate::http::compression::encode_body(plaintext, "gzip")
		.await
		.expect("gzip encode");
	// The payload is genuinely compressed (gzip magic) and tiny, so this exercises
	// content-encoding decoding rather than the buffer-size path.
	assert_eq!(&gz[..2], &[0x1f, 0x8b]);

	let req = ::http::Request::builder()
		.uri("/v1/messages")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.header(::http::header::CONTENT_ENCODING, "gzip")
		.body(Body::from(gz.to_vec()))
		.unwrap();

	let (parts, parsed) = provider
		.read_body_and_default_model::<types::messages::Request>(None, req, &mut None)
		.await
		.expect("gzip request body should decode and parse as JSON");

	assert_eq!(parsed.model.as_deref(), Some("claude-sonnet-4-5"));
	// The encoding header is stripped now that the body is plaintext.
	assert!(
		parts
			.headers
			.get(::http::header::CONTENT_ENCODING)
			.is_none()
	);
}

#[tokio::test]
async fn read_body_still_parses_plaintext_request() {
	// A plaintext (unencoded) request body must continue to parse unchanged — the
	// decompression path is a no-op when no Content-Encoding is present.
	let provider = custom_provider(custom::ProviderFormat::Messages);

	let req = ::http::Request::builder()
		.uri("/v1/messages")
		.header(::http::header::CONTENT_TYPE, "application/json")
		.body(Body::from(
			br#"{"model":"claude-sonnet-4-5","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#
				.to_vec(),
		))
		.unwrap();

	let (_parts, parsed) = provider
		.read_body_and_default_model::<types::messages::Request>(None, req, &mut None)
		.await
		.expect("plaintext request body should parse as JSON");

	assert_eq!(parsed.model.as_deref(), Some("claude-sonnet-4-5"));
}

#[test]
fn custom_provider_name_falls_back_to_custom() {
	let provider = custom_provider(custom::ProviderFormat::Completions);
	assert_eq!(provider.provider(), strng::literal!("custom"));
}

#[test]
fn custom_provider_override_drives_provider_name() {
	let provider = AIProvider::Custom(custom::Provider {
		model: None,
		provider_override: Some(strng::literal!("cohere")),
		formats: vec![custom::ProviderFormatConfig {
			format: custom::ProviderFormat::Rerank,
			path: None,
		}],
	});
	assert_eq!(provider.provider(), strng::literal!("cohere"));
}

#[test]
fn vertex_anthropic_model_uses_exclusive_convention() {
	let provider = vertex_provider("anthropic/claude-sonnet-4-5");
	assert_eq!(
		cache_convention_for(&provider, None, "anthropic/claude-sonnet-4-5"),
		CacheTokenConvention::InputExcludesCache,
	);
}

#[test]
fn vertex_non_anthropic_model_uses_inclusive_convention() {
	let provider = vertex_provider("gemini-2.0-flash");
	assert_eq!(
		cache_convention_for(&provider, None, "gemini-2.0-flash"),
		CacheTokenConvention::InputIncludesCache,
	);
}

#[test]
fn custom_messages_backend_uses_exclusive_convention() {
	let provider = custom_provider(custom::ProviderFormat::Messages);
	assert_eq!(
		cache_convention_for(
			&provider,
			Some(custom::ProviderFormat::Messages),
			"some-model"
		),
		CacheTokenConvention::InputExcludesCache,
	);
}

#[test]
fn custom_completions_backend_uses_inclusive_convention() {
	let provider = custom_provider(custom::ProviderFormat::Completions);
	assert_eq!(
		cache_convention_for(
			&provider,
			Some(custom::ProviderFormat::Completions),
			"some-model"
		),
		CacheTokenConvention::InputIncludesCache,
	);
}

#[test]
fn fixed_providers_classify_by_family() {
	assert_eq!(
		cache_convention_for(
			&AIProvider::Anthropic(anthropic::Provider { model: None }),
			None,
			"claude-sonnet-4-5"
		),
		CacheTokenConvention::InputExcludesCache,
	);
	assert_eq!(
		cache_convention_for(
			&AIProvider::OpenAI(openai::Provider {
				model: None,
				moderation: None,
			}),
			Some(custom::ProviderFormat::Completions),
			"gpt-4o"
		),
		CacheTokenConvention::InputIncludesCache,
	);
}

#[test]
fn query_requests_sse_matches_alt_query_parameter() {
	let uri = |s: &str| s.parse::<::http::Uri>().expect("valid uri");
	assert!(query_requests_sse(&uri(
		"/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
	)));
	assert!(query_requests_sse(&uri(
		"/v1beta/models/gemini-2.5-flash:streamGenerateContent?key=abc&alt=sse"
	)));
	assert!(!query_requests_sse(&uri(
		"/v1beta/models/gemini-2.5-flash:streamGenerateContent"
	)));
	assert!(!query_requests_sse(&uri(
		"/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=json"
	)));
	assert!(!query_requests_sse(&uri(
		"/v1beta/models/gemini-2.5-flash:streamGenerateContent?halt=sse"
	)));
}
