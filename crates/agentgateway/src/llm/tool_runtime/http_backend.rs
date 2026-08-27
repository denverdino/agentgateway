use std::time::{Duration, Instant};

use ::http::{HeaderValue, Method, StatusCode, Uri, header};
use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::backend::execute_sequentially;
use super::{
	CODE_INTERPRETER_FUNCTION, ManagedToolCall, ToolApplicationError, ToolBackend,
	ToolBatchExecution, ToolBatchInfrastructureError, ToolExecutionContext, ToolExecutionResult,
	ToolInfrastructureError, WEB_SEARCH_FUNCTION,
};
use crate::http::filters::BackendRequestTimeout;
use crate::proxy::httpproxy::PolicyClient;
use crate::telemetry::metrics::{OutboundCallKind, OutboundCallSubtype};
use crate::types::agent::{Backend, BackendTrafficPolicy, ResourceName};

#[derive(Clone)]
pub struct HttpToolBackend {
	transport: HttpToolTransport,
}

impl HttpToolBackend {
	pub fn new(
		client: PolicyClient,
		url: Uri,
		timeout: Duration,
		bearer_token: Option<SecretString>,
		max_response_bytes: usize,
	) -> Result<Self, ToolInfrastructureError> {
		Ok(Self {
			transport: HttpToolTransport::new(
				client,
				url,
				timeout,
				bearer_token,
				max_response_bytes,
				OutboundCallSubtype::ToolHttp,
			)?,
		})
	}

	fn request_body(&self, call: &ManagedToolCall) -> Result<Value, ToolInfrastructureError> {
		if call.internal_name.as_str() == CODE_INTERPRETER_FUNCTION {
			return Err(ToolInfrastructureError::configuration());
		}
		if call.internal_name.as_str() == WEB_SEARCH_FUNCTION {
			return web_search_request(call);
		}
		serde_json::to_value(FunctionRequest {
			tool_name: call.public_name.as_str(),
			call_id: call.call_id.as_str(),
			arguments: &call.arguments,
			context: FunctionRequestContext {
				request_id: None,
				deadline_ms: 0,
			},
		})
		.map_err(|_| ToolInfrastructureError::internal())
	}

	async fn execute_one(
		&self,
		call: ManagedToolCall,
		context: ToolExecutionContext,
	) -> Result<ToolExecutionResult, ToolInfrastructureError> {
		let timeout = self.transport.effective_timeout(&context)?;
		let mut request = self.request_body(&call)?;
		if call.internal_name.as_str() != WEB_SEARCH_FUNCTION {
			let request_context = request
				.get_mut("context")
				.and_then(Value::as_object_mut)
				.ok_or_else(ToolInfrastructureError::internal)?;
			request_context.insert(
				"request_id".into(),
				context
					.request_id
					.as_ref()
					.map_or(Value::Null, |id| Value::String(id.to_string())),
			);
			request_context.insert("deadline_ms".into(), Value::from(timeout_millis(timeout)));
		}

		let response = self.transport.post_json(request, timeout).await?;
		if let Some(result) = parse_application_error(&response)? {
			return Ok(result);
		}
		if call.internal_name.as_str() == WEB_SEARCH_FUNCTION {
			let result = ToolExecutionResult::web_search(response);
			validate_typed_result(&result)?;
			Ok(result)
		} else {
			Ok(ToolExecutionResult::function(response))
		}
	}
}

#[async_trait]
impl ToolBackend for HttpToolBackend {
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

#[derive(Serialize)]
struct FunctionRequest<'a> {
	tool_name: &'a str,
	call_id: &'a str,
	arguments: &'a Value,
	context: FunctionRequestContext<'a>,
}

#[derive(Serialize)]
struct FunctionRequestContext<'a> {
	request_id: Option<&'a str>,
	deadline_ms: u64,
}

fn web_search_request(call: &ManagedToolCall) -> Result<Value, ToolInfrastructureError> {
	let arguments = call
		.arguments
		.as_object()
		.ok_or_else(ToolInfrastructureError::configuration)?;
	if arguments.len() != 1 || !arguments.get("query").is_some_and(Value::is_string) {
		return Err(ToolInfrastructureError::configuration());
	}
	let mut flattened = call
		.trusted_options
		.as_object()
		.cloned()
		.ok_or_else(ToolInfrastructureError::configuration)?;
	if flattened.contains_key("query") {
		return Err(ToolInfrastructureError::configuration());
	}
	flattened.insert("query".into(), arguments["query"].clone());
	Ok(Value::Object(flattened))
}

#[derive(Clone)]
pub(crate) struct HttpToolTransport {
	client: PolicyClient,
	url: Uri,
	timeout: Duration,
	bearer_token: Option<SecretString>,
	max_response_bytes: usize,
	subtype: OutboundCallSubtype,
	policies: Vec<BackendTrafficPolicy>,
}

pub(crate) struct HttpToolJsonResponse {
	pub(crate) status: StatusCode,
	pub(crate) body: Result<Option<Value>, ToolInfrastructureError>,
}

impl HttpToolTransport {
	pub(crate) fn new(
		client: PolicyClient,
		url: Uri,
		timeout: Duration,
		bearer_token: Option<SecretString>,
		max_response_bytes: usize,
		subtype: OutboundCallSubtype,
	) -> Result<Self, ToolInfrastructureError> {
		if timeout.is_zero() || max_response_bytes == 0 || url.host().is_none() {
			return Err(ToolInfrastructureError::configuration());
		}
		let policies = match url.scheme_str() {
			Some("http") => Vec::new(),
			Some("https") => vec![BackendTrafficPolicy::BackendTLS(
				crate::http::backendtls::SYSTEM_TRUST.clone(),
			)],
			_ => return Err(ToolInfrastructureError::configuration()),
		};
		Ok(Self {
			client,
			url,
			timeout,
			bearer_token,
			max_response_bytes,
			subtype,
			policies,
		})
	}

	pub(crate) fn effective_timeout(
		&self,
		context: &ToolExecutionContext,
	) -> Result<Duration, ToolInfrastructureError> {
		let Some(deadline) = context.deadline else {
			return Ok(self.timeout);
		};
		let remaining = deadline.saturating_duration_since(Instant::now());
		if remaining.is_zero() {
			return Err(ToolInfrastructureError::timeout());
		}
		Ok(self.timeout.min(remaining))
	}

	pub(crate) async fn post_json(
		&self,
		body: Value,
		timeout: Duration,
	) -> Result<Value, ToolInfrastructureError> {
		let response = self.post_json_response(body, timeout).await?;
		if !response.status.is_success() {
			return Err(super::transport::classify_status(response.status, false));
		}
		response.body?.ok_or_else(ToolInfrastructureError::backend)
	}

	pub(crate) async fn post_json_response(
		&self,
		body: Value,
		timeout: Duration,
	) -> Result<HttpToolJsonResponse, ToolInfrastructureError> {
		tokio::time::timeout(timeout, self.post_json_response_inner(body, timeout))
			.await
			.map_err(|_| ToolInfrastructureError::timeout())?
	}

	async fn post_json_response_inner(
		&self,
		body: Value,
		timeout: Duration,
	) -> Result<HttpToolJsonResponse, ToolInfrastructureError> {
		let bytes = serde_json::to_vec(&body).map_err(|_| ToolInfrastructureError::internal())?;
		let mut builder = ::http::Request::builder()
			.method(Method::POST)
			.uri(self.url.clone())
			.header(header::CONTENT_TYPE, "application/json");
		if let Some(token) = &self.bearer_token {
			let mut value = HeaderValue::from_str(&format!("Bearer {}", token.expose_secret()))
				.map_err(|_| ToolInfrastructureError::configuration())?;
			value.set_sensitive(true);
			builder = builder.header(header::AUTHORIZATION, value);
		}
		let mut request = builder
			.body(crate::http::Body::from(bytes))
			.map_err(|_| ToolInfrastructureError::configuration())?;
		request
			.extensions_mut()
			.insert(BackendRequestTimeout(timeout));

		let backend = Backend::Dynamic(
			ResourceName::new("_managed-tool-http".into(), "".into()),
			None,
		);
		let response = self
			.client
			.with_outbound(OutboundCallKind::Primary, self.subtype)
			.call_with_explicit_policies_list(request, backend, self.policies.clone())
			.await
			.map_err(super::transport::classify_proxy_error)?;
		let status = response.status();
		let body = if status.is_success() {
			match crate::http::read_body_with_limit(response.into_body(), self.max_response_bytes).await {
				Ok(bytes) => serde_json::from_slice(&bytes)
					.map(Some)
					.map_err(|_| ToolInfrastructureError::backend()),
				Err(_) => Err(ToolInfrastructureError::backend()),
			}
		} else {
			Ok(None)
		};
		Ok(HttpToolJsonResponse { status, body })
	}
}

fn timeout_millis(timeout: Duration) -> u64 {
	u64::try_from(timeout.as_millis().max(1)).unwrap_or(u64::MAX)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ApplicationErrorEnvelope {
	ok: bool,
	error: ApplicationErrorDetails,
	stdout: String,
	stderr: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ApplicationErrorDetails {
	#[serde(rename = "type")]
	error_type: String,
	message: String,
	retryable: bool,
}

pub(crate) fn parse_application_error(
	value: &Value,
) -> Result<Option<ToolExecutionResult>, ToolInfrastructureError> {
	if value.get("ok") != Some(&Value::Bool(false)) {
		return Ok(None);
	}
	let parsed: ApplicationErrorEnvelope =
		serde_json::from_value(value.clone()).map_err(|_| ToolInfrastructureError::backend())?;
	if parsed.ok {
		return Err(ToolInfrastructureError::backend());
	}
	Ok(Some(ToolExecutionResult::ApplicationError(
		ToolApplicationError::new(
			parsed.error.error_type,
			parsed.error.message,
			parsed.error.retryable,
			parsed.stdout,
			parsed.stderr,
		),
	)))
}

pub(crate) fn validate_typed_result(
	result: &ToolExecutionResult,
) -> Result<(), ToolInfrastructureError> {
	result
		.clone()
		.into_model_output(usize::MAX)
		.map(|_| ())
		.map_err(|_| ToolInfrastructureError::backend())
}
