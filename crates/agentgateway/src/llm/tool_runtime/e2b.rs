use std::time::{Duration, Instant};

use ::http::{HeaderValue, Method, StatusCode, Uri, header};
use async_trait::async_trait;
use base64::Engine;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
	CODE_INTERPRETER_FUNCTION, ManagedToolCall, ProgramSandbox, ProgramSandboxExecution,
	ProgramSandboxRequest, SANDBOX_MAX_BATCH_DEADLINE, SANDBOX_MAX_BATCH_EXECUTIONS,
	SANDBOX_MAX_CALL_ID_BYTES, SANDBOX_MAX_CODE_BYTES, SandboxCleanupOutcome, SandboxOperationReport,
	ToolApplicationError, ToolBackend, ToolBatchExecution, ToolBatchInfrastructureError,
	ToolBatchMetadata, ToolExecutionContext, ToolExecutionResult, ToolInfrastructureError,
	program_protocol_stdout_max_bytes,
};
use crate::http::filters::BackendRequestTimeout;
use crate::proxy::httpproxy::PolicyClient;
use crate::telemetry::metrics::{OutboundCallKind, OutboundCallSubtype};
use crate::types::agent::{Backend, BackendTrafficPolicy, ResourceName};

const TEMPLATE: &str = "code-interpreter-v1";
const PYTHON_CWD: &str = "/home/user";
const JUPYTER_PORT: u16 = 49_999;
const CONTROL_BODY_LIMIT: usize = 64 * 1024;
const PROTOCOL_OVERHEAD: usize = 64 * 1024;
const CLEANUP_ATTEMPTS: usize = 2;
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_ERROR_MESSAGE_BYTES: usize = 1024;

struct PythonSource {
	code: String,
	env_vars: Option<Map<String, Value>>,
}

#[derive(Clone)]
pub struct E2bSandboxBackend {
	client: PolicyClient,
	api_url: Uri,
	domain: String,
	timeout: Duration,
	api_key: SecretString,
	max_response_bytes: usize,
}

impl E2bSandboxBackend {
	pub fn new(
		client: PolicyClient,
		api_url: Uri,
		domain: String,
		timeout: Duration,
		api_key: SecretString,
		max_response_bytes: usize,
	) -> Result<Self, ToolInfrastructureError> {
		if timeout.is_zero()
			|| timeout > SANDBOX_MAX_BATCH_DEADLINE
			|| max_response_bytes == 0
			|| api_url.host().is_none()
			|| api_url.path() != "/"
			|| api_url.query().is_some()
			|| api_key.expose_secret().is_empty()
			|| !super::validation::valid_domain(&domain)
		{
			return Err(ToolInfrastructureError::configuration());
		}
		match api_url.scheme_str() {
			Some("https") => {},
			Some("http") if super::validation::is_loopback(api_url.host().expect("host checked")) => {},
			_ => return Err(ToolInfrastructureError::configuration()),
		}
		Ok(Self {
			client,
			api_url,
			domain,
			timeout,
			api_key,
			max_response_bytes,
		})
	}

	fn validate_calls(&self, calls: &[ManagedToolCall]) -> Result<(), ToolInfrastructureError> {
		if calls.is_empty() || calls.len() > SANDBOX_MAX_BATCH_EXECUTIONS {
			return Err(ToolInfrastructureError::configuration());
		}
		let mut ids = std::collections::HashSet::with_capacity(calls.len());
		for call in calls {
			let arguments = call
				.arguments
				.as_object()
				.ok_or_else(ToolInfrastructureError::configuration)?;
			let valid = call.internal_name.as_str() == CODE_INTERPRETER_FUNCTION
				&& call.call_id.len() <= SANDBOX_MAX_CALL_ID_BYTES
				&& !call.call_id.is_empty()
				&& ids.insert(call.call_id.as_str())
				&& call
					.trusted_options
					.as_object()
					.is_some_and(serde_json::Map::is_empty)
				&& arguments.len() == 1
				&& arguments
					.get("code")
					.and_then(Value::as_str)
					.is_some_and(|code| code.len() <= SANDBOX_MAX_CODE_BYTES);
			if !valid {
				return Err(ToolInfrastructureError::configuration());
			}
		}
		Ok(())
	}

	async fn execute_batch_inner(
		&self,
		calls: Vec<ManagedToolCall>,
		context: ToolExecutionContext,
	) -> Result<ToolBatchExecution, ToolBatchInfrastructureError> {
		if calls.is_empty() {
			return Ok(ToolBatchExecution::new(Vec::new()));
		}
		self.validate_calls(&calls)?;
		let sources = calls
			.into_iter()
			.map(|call| PythonSource {
				code: call.arguments["code"]
					.as_str()
					.expect("validated code argument")
					.to_owned(),
				env_vars: None,
			})
			.collect();
		self
			.execute_python_sources(sources, context, self.max_response_bytes)
			.await
	}

	async fn execute_python_sources(
		&self,
		sources: Vec<PythonSource>,
		context: ToolExecutionContext,
		max_output_bytes: usize,
	) -> Result<ToolBatchExecution, ToolBatchInfrastructureError> {
		if sources.is_empty() || sources.len() > SANDBOX_MAX_BATCH_EXECUTIONS {
			return Err(ToolInfrastructureError::configuration().into());
		}
		for source in &sources {
			if source.code.len() > SANDBOX_MAX_CODE_BYTES
				|| source.env_vars.as_ref().is_some_and(|env_vars| {
					serde_json::to_vec(env_vars)
						.map(|value| value.len() > self.max_response_bytes.saturating_mul(2))
						.unwrap_or(true)
				}) {
				return Err(ToolInfrastructureError::configuration().into());
			}
		}
		let deadline = context.deadline.map_or_else(
			|| Instant::now() + self.timeout,
			|value| value.min(Instant::now() + self.timeout),
		);
		let remaining = deadline.saturating_duration_since(Instant::now());
		if remaining.is_zero() {
			return Err(ToolInfrastructureError::timeout().into());
		}
		let cleanup_reserve = Duration::from_secs(2).min(remaining / 4);
		let operation_deadline = deadline - cleanup_reserve;
		let mut metadata = ToolBatchMetadata::default();

		let create_started = Instant::now();
		let sandbox = match self.create_sandbox(operation_deadline).await {
			Ok(sandbox) => sandbox,
			Err(error) => {
				metadata.sandbox_create = Some(operation_report(false, create_started));
				return Err(ToolBatchInfrastructureError::new(error, metadata));
			},
		};
		let mut lease = SandboxLease::new(self.clone(), sandbox.sandbox_id.clone());
		if sandbox
			.domain
			.as_deref()
			.is_some_and(|domain| !super::validation::valid_domain(domain))
		{
			metadata.sandbox_create = Some(operation_report(false, create_started));
			cleanup_lease(&mut lease, deadline, &mut metadata).await;
			return Err(ToolBatchInfrastructureError::new(
				ToolInfrastructureError::backend(),
				metadata,
			));
		}
		metadata.sandbox_create = Some(operation_report(true, create_started));

		let execute_started = Instant::now();
		let mut remaining_output = max_output_bytes;
		let mut results = Vec::with_capacity(sources.len());
		let mut execution_error = None;
		for source in &sources {
			match self
				.execute_source(&sandbox, source, operation_deadline, remaining_output)
				.await
			{
				Ok(result) => {
					remaining_output = remaining_output.saturating_sub(result_output_bytes(&result));
					results.push(result);
				},
				Err(error) => {
					execution_error = Some(error);
					break;
				},
			}
		}
		metadata.sandbox_execute = Some(operation_report(execution_error.is_none(), execute_started));

		let terminated = cleanup_lease(&mut lease, deadline, &mut metadata).await;

		if let Some(error) = execution_error {
			return Err(ToolBatchInfrastructureError::new(error, metadata));
		}
		if !terminated {
			return Err(ToolBatchInfrastructureError::new(
				ToolInfrastructureError::backend(),
				metadata,
			));
		}
		Ok(ToolBatchExecution { results, metadata })
	}

	async fn create_sandbox(&self, deadline: Instant) -> Result<Sandbox, ToolInfrastructureError> {
		let uri = join_uri(&self.api_url, "/sandboxes")?;
		let timeout_seconds = provider_timeout_seconds(self.timeout, deadline)?;
		let response = self
			.request(
				Method::POST,
				uri,
				Some(json!({
					"templateID": TEMPLATE,
					"timeout": timeout_seconds,
					"secure": true
				})),
				Auth::ApiKey,
				CONTROL_BODY_LIMIT,
				deadline,
			)
			.await?;
		if response.status != StatusCode::CREATED {
			return Err(super::transport::classify_status(response.status, true));
		}
		let sandbox: Sandbox =
			serde_json::from_slice(&response.body).map_err(|_| ToolInfrastructureError::backend())?;
		if !valid_sandbox_id(&sandbox.sandbox_id) {
			return Err(ToolInfrastructureError::backend());
		}
		Ok(sandbox)
	}

	async fn execute_source(
		&self,
		sandbox: &Sandbox,
		source: &PythonSource,
		deadline: Instant,
		max_output_bytes: usize,
	) -> Result<ToolExecutionResult, ToolInfrastructureError> {
		let origin = self.data_plane_origin(sandbox)?;
		let headers = Auth::Sandbox(sandbox);
		let context_response = self
			.request(
				Method::POST,
				join_uri(&origin, "/contexts")?,
				Some(json!({"language": "python", "cwd": PYTHON_CWD})),
				headers,
				CONTROL_BODY_LIMIT,
				deadline,
			)
			.await?;
		if !context_response.status.is_success() {
			return Err(super::transport::classify_status(
				context_response.status,
				true,
			));
		}
		let context: CodeContext = serde_json::from_slice(&context_response.body)
			.map_err(|_| ToolInfrastructureError::backend())?;
		if !valid_path_segment(&context.id) {
			return Err(ToolInfrastructureError::backend());
		}

		let (wrapped, truncation_marker) = bounded_python_source(&source.code, max_output_bytes);
		let execution = self
			.request(
				Method::POST,
				join_uri(&origin, "/execute")?,
				Some(json!({
					"code": wrapped,
					"context_id": context.id,
					"language": Value::Null,
					"env_vars": source.env_vars
				})),
				headers,
				max_output_bytes.saturating_add(PROTOCOL_OVERHEAD),
				deadline,
			)
			.await;

		let remove = self
			.request(
				Method::DELETE,
				join_uri(&origin, &format!("/contexts/{}", context.id))?,
				None,
				headers,
				0,
				deadline,
			)
			.await;
		let execution = execution?;
		let remove = remove?;
		if !remove.status.is_success() {
			return Err(super::transport::classify_status(remove.status, true));
		}
		if !execution.status.is_success() {
			return Err(super::transport::classify_status(execution.status, true));
		}
		parse_execution(&execution.body, max_output_bytes, &truncation_marker)
	}

	async fn terminate_sandbox(&self, sandbox_id: &str, deadline: Instant) -> bool {
		let Ok(uri) = join_uri(&self.api_url, &format!("/sandboxes/{sandbox_id}")) else {
			return false;
		};
		for _ in 0..CLEANUP_ATTEMPTS {
			if let Ok(response) = self
				.request(Method::DELETE, uri.clone(), None, Auth::ApiKey, 0, deadline)
				.await
				&& matches!(
					response.status,
					StatusCode::NO_CONTENT | StatusCode::NOT_FOUND
				) {
				return true;
			}
		}
		false
	}

	fn data_plane_origin(&self, sandbox: &Sandbox) -> Result<Uri, ToolInfrastructureError> {
		if self
			.api_url
			.host()
			.is_some_and(super::validation::is_loopback)
		{
			return join_uri(&self.api_url, "/");
		}
		let domain = sandbox.domain.as_deref().unwrap_or(&self.domain);
		if !super::validation::valid_domain(domain) {
			return Err(ToolInfrastructureError::backend());
		}
		format!("https://{JUPYTER_PORT}-{}.{domain}", sandbox.sandbox_id)
			.parse()
			.map_err(|_| ToolInfrastructureError::backend())
	}

	async fn request(
		&self,
		method: Method,
		uri: Uri,
		body: Option<Value>,
		auth: Auth<'_>,
		body_limit: usize,
		deadline: Instant,
	) -> Result<E2bResponse, ToolInfrastructureError> {
		let remaining = deadline.saturating_duration_since(Instant::now());
		if remaining.is_zero() {
			return Err(ToolInfrastructureError::timeout());
		}
		let bytes = body
			.map(|body| serde_json::to_vec(&body).map_err(|_| ToolInfrastructureError::internal()))
			.transpose()?
			.unwrap_or_default();
		let mut builder = ::http::Request::builder().method(method).uri(uri.clone());
		if !bytes.is_empty() {
			builder = builder.header(header::CONTENT_TYPE, "application/json");
		}
		match auth {
			Auth::ApiKey => {
				let mut value = HeaderValue::from_str(self.api_key.expose_secret())
					.map_err(|_| ToolInfrastructureError::configuration())?;
				value.set_sensitive(true);
				builder = builder.header("X-API-Key", value);
			},
			Auth::Sandbox(sandbox) => {
				if let Some(token) = &sandbox.envd_access_token {
					let mut value =
						HeaderValue::from_str(token).map_err(|_| ToolInfrastructureError::backend())?;
					value.set_sensitive(true);
					builder = builder.header("X-Access-Token", value);
				}
				if let Some(token) = &sandbox.traffic_access_token {
					let mut value =
						HeaderValue::from_str(token).map_err(|_| ToolInfrastructureError::backend())?;
					value.set_sensitive(true);
					builder = builder.header("E2B-Traffic-Access-Token", value);
				}
			},
		}
		let mut request = builder
			.body(crate::http::Body::from(bytes))
			.map_err(|_| ToolInfrastructureError::configuration())?;
		request
			.extensions_mut()
			.insert(BackendRequestTimeout(remaining));
		let policies = match uri.scheme_str() {
			Some("http") => Vec::new(),
			Some("https") => vec![BackendTrafficPolicy::BackendTLS(
				crate::http::backendtls::SYSTEM_TRUST.clone(),
			)],
			_ => return Err(ToolInfrastructureError::configuration()),
		};
		let backend = Backend::Dynamic(
			ResourceName::new("_managed-tool-e2b".into(), "".into()),
			None,
		);
		let response = tokio::time::timeout(
			remaining,
			self
				.client
				.with_outbound(OutboundCallKind::Primary, OutboundCallSubtype::ToolE2b)
				.call_with_explicit_policies_list(request, backend, policies),
		)
		.await
		.map_err(|_| ToolInfrastructureError::timeout())?
		.map_err(super::transport::classify_proxy_error)?;
		let status = response.status();
		let body = if body_limit == 0 || !status.is_success() {
			Vec::new()
		} else {
			crate::http::read_body_with_limit(response.into_body(), body_limit)
				.await
				.map_err(|_| ToolInfrastructureError::backend())?
				.to_vec()
		};
		Ok(E2bResponse { status, body })
	}
}

struct SandboxLease {
	backend: E2bSandboxBackend,
	sandbox_id: Option<String>,
}

impl SandboxLease {
	fn new(backend: E2bSandboxBackend, sandbox_id: String) -> Self {
		Self {
			backend,
			sandbox_id: Some(sandbox_id),
		}
	}

	fn start_cleanup(&mut self) -> Option<tokio::task::JoinHandle<bool>> {
		let runtime = tokio::runtime::Handle::try_current().ok()?;
		let sandbox_id = self.sandbox_id.take()?;
		let backend = self.backend.clone();
		Some(runtime.spawn(async move {
			backend
				.terminate_sandbox(&sandbox_id, Instant::now() + CLEANUP_TIMEOUT)
				.await
		}))
	}

	async fn cleanup_until(&mut self, request_deadline: Instant) -> bool {
		let Some(cleanup) = self.start_cleanup() else {
			return false;
		};
		let remaining = request_deadline.saturating_duration_since(Instant::now());
		if remaining.is_zero() {
			drop(cleanup);
			return false;
		}
		matches!(tokio::time::timeout(remaining, cleanup).await, Ok(Ok(true)))
	}
}

impl Drop for SandboxLease {
	fn drop(&mut self) {
		if self.sandbox_id.is_some() && self.start_cleanup().is_none() {
			tracing::warn!("unable to schedule E2B sandbox cleanup");
		}
	}
}

async fn cleanup_lease(
	lease: &mut SandboxLease,
	request_deadline: Instant,
	metadata: &mut ToolBatchMetadata,
) -> bool {
	let started = Instant::now();
	let terminated = lease.cleanup_until(request_deadline).await;
	metadata.sandbox_terminate = Some(operation_report(terminated, started));
	metadata.sandbox_cleanup = Some(if terminated {
		SandboxCleanupOutcome::Success
	} else {
		SandboxCleanupOutcome::Failure
	});
	metadata.sandbox_cleanup_duration = Some(started.elapsed());
	terminated
}

fn provider_timeout_seconds(
	configured_timeout: Duration,
	deadline: Instant,
) -> Result<u64, ToolInfrastructureError> {
	let remaining = deadline.saturating_duration_since(Instant::now());
	if remaining.is_zero() {
		return Err(ToolInfrastructureError::timeout());
	}
	let lifetime = configured_timeout.min(remaining);
	Ok(
		lifetime
			.as_secs()
			.saturating_add(u64::from(lifetime.subsec_nanos() != 0))
			.max(1),
	)
}

#[async_trait]
impl ToolBackend for E2bSandboxBackend {
	async fn execute_batch(
		&self,
		calls: Vec<ManagedToolCall>,
		context: ToolExecutionContext,
	) -> Result<ToolBatchExecution, ToolBatchInfrastructureError> {
		self.execute_batch_inner(calls, context).await
	}
}

#[async_trait]
impl ProgramSandbox for E2bSandboxBackend {
	async fn run(
		&self,
		request: ProgramSandboxRequest,
		context: ToolExecutionContext,
	) -> Result<ProgramSandboxExecution, ToolBatchInfrastructureError> {
		let execution = self
			.execute_python_sources(
				vec![PythonSource {
					code: request.source,
					env_vars: Some(request.env_vars),
				}],
				context,
				program_protocol_stdout_max_bytes(self.max_response_bytes),
			)
			.await?;
		let ToolBatchExecution {
			mut results,
			metadata,
		} = execution;
		let Some(result) = results.pop() else {
			return Err(ToolBatchInfrastructureError::new(
				ToolInfrastructureError::internal(),
				metadata,
			));
		};
		Ok(ProgramSandboxExecution { result, metadata })
	}
}

#[derive(Clone, Copy)]
enum Auth<'a> {
	ApiKey,
	Sandbox(&'a Sandbox),
}

struct E2bResponse {
	status: StatusCode,
	body: Vec<u8>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sandbox {
	#[serde(rename = "sandboxID")]
	sandbox_id: String,
	#[serde(default)]
	domain: Option<String>,
	#[serde(default, rename = "envdAccessToken")]
	envd_access_token: Option<String>,
	#[serde(default, rename = "trafficAccessToken")]
	traffic_access_token: Option<String>,
}

#[derive(Deserialize)]
struct CodeContext {
	id: String,
}

#[derive(Default)]
struct ParsedExecution {
	stdout: String,
	stderr: String,
	error: Option<(String, String)>,
}

fn parse_execution(
	body: &[u8],
	max_output_bytes: usize,
	truncation_marker: &str,
) -> Result<ToolExecutionResult, ToolInfrastructureError> {
	let text = std::str::from_utf8(body).map_err(|_| ToolInfrastructureError::backend())?;
	let mut parsed = ParsedExecution::default();
	for line in text.lines().filter(|line| !line.is_empty()) {
		let event: Value =
			serde_json::from_str(line).map_err(|_| ToolInfrastructureError::backend())?;
		let kind = event
			.get("type")
			.and_then(Value::as_str)
			.ok_or_else(ToolInfrastructureError::backend)?;
		match kind {
			"stdout" => parsed.stdout.push_str(required_string(&event, "text")?),
			"stderr" => parsed.stderr.push_str(required_string(&event, "text")?),
			"error" => {
				parsed.error = Some((
					required_string(&event, "name")?.to_owned(),
					required_string(&event, "value")?.to_owned(),
				));
			},
			"result" | "number_of_executions" => {},
			_ => return Err(ToolInfrastructureError::backend()),
		}
		if parsed.stdout.len().saturating_add(parsed.stderr.len())
			> max_output_bytes.saturating_add(truncation_marker.len())
		{
			return Err(ToolInfrastructureError::backend());
		}
	}
	let truncated = parsed.stderr.ends_with(truncation_marker);
	if truncated {
		parsed
			.stderr
			.truncate(parsed.stderr.len() - truncation_marker.len());
	}
	if parsed.stdout.len().saturating_add(parsed.stderr.len()) > max_output_bytes {
		return Err(ToolInfrastructureError::backend());
	}
	if let Some((name, value)) = parsed.error {
		let message = super::truncate_utf8_bytes(&format!("{name}: {value}"), MAX_ERROR_MESSAGE_BYTES);
		return Ok(ToolExecutionResult::ApplicationError(
			ToolApplicationError::execution_error(message, false, parsed.stdout, parsed.stderr),
		));
	}
	Ok(ToolExecutionResult::python(json!({
		"exit_code": 0,
		"stdout": parsed.stdout,
		"stderr": parsed.stderr,
		"timed_out": false,
		"truncated": truncated,
		"artifacts": []
	})))
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, ToolInfrastructureError> {
	value
		.get(field)
		.and_then(Value::as_str)
		.ok_or_else(ToolInfrastructureError::backend)
}

fn bounded_python_source(code: &str, max_output_bytes: usize) -> (String, String) {
	let encoded = base64::engine::general_purpose::STANDARD.encode(code.as_bytes());
	let marker = format!("\n__AG_OUTPUT_TRUNCATED_{}__\n", uuid::Uuid::new_v4());
	let source = format!(
		r#"import base64 as __ag_base64
import sys as __ag_sys

class __AGWriter:
    def __init__(self, target, budget):
        self.target = target
        self.budget = budget
    def write(self, value):
        encoded = value.encode("utf-8", errors="replace")
        remaining = self.budget[0]
        chunk = encoded[:remaining].decode("utf-8", errors="ignore")
        self.budget[0] -= len(chunk.encode("utf-8"))
        if len(encoded) > remaining:
            self.budget[1] = True
        self.target.write(chunk)
        return len(value)
    def flush(self):
        return self.target.flush()
    def isatty(self):
        return False
    @property
    def encoding(self):
        return "utf-8"

__ag_budget = [{max_output_bytes}, False]
__ag_stdout, __ag_stderr = __ag_sys.stdout, __ag_sys.stderr
__ag_sys.stdout = __AGWriter(__ag_stdout, __ag_budget)
__ag_sys.stderr = __AGWriter(__ag_stderr, __ag_budget)
try:
    exec(compile(__ag_base64.b64decode("{encoded}").decode("utf-8"), "<agentgateway>", "exec"), {{"__name__": "__main__"}})
finally:
    __ag_sys.stdout, __ag_sys.stderr = __ag_stdout, __ag_stderr
    if __ag_budget[1]:
        __ag_stderr.write({marker:?})
"#
	);
	(source, marker)
}

fn join_uri(base: &Uri, path: &str) -> Result<Uri, ToolInfrastructureError> {
	let scheme = base
		.scheme_str()
		.ok_or_else(ToolInfrastructureError::configuration)?;
	let authority = base
		.authority()
		.ok_or_else(ToolInfrastructureError::configuration)?;
	format!("{scheme}://{authority}{path}")
		.parse()
		.map_err(|_| ToolInfrastructureError::configuration())
}

fn valid_sandbox_id(value: &str) -> bool {
	!value.is_empty()
		&& value.len() <= 128
		&& value
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn valid_path_segment(value: &str) -> bool {
	valid_sandbox_id(value)
}

fn operation_report(success: bool, started: Instant) -> SandboxOperationReport {
	SandboxOperationReport {
		outcome: if success {
			SandboxCleanupOutcome::Success
		} else {
			SandboxCleanupOutcome::Failure
		},
		duration: started.elapsed(),
	}
}

fn result_output_bytes(result: &ToolExecutionResult) -> usize {
	match result {
		ToolExecutionResult::Python(value) => ["stdout", "stderr"]
			.into_iter()
			.filter_map(|field| value.get(field).and_then(Value::as_str))
			.map(str::len)
			.sum(),
		ToolExecutionResult::ApplicationError(error) => error.stdout.len() + error.stderr.len(),
		_ => 0,
	}
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;
	use std::sync::atomic::{AtomicUsize, Ordering};
	use std::time::{Duration, Instant};

	use secrecy::SecretString;
	use serde_json::json;
	use wiremock::matchers::{method, path};
	use wiremock::{Mock, MockServer, ResponseTemplate};

	use super::{E2bSandboxBackend, bounded_python_source, parse_execution};

	#[test]
	fn output_marker_is_removed_and_sets_truncated() {
		let (_source, marker) = bounded_python_source("print('large')", 5);
		let body = format!(
			"{{\"type\":\"stdout\",\"text\":\"hello\",\"timestamp\":1}}\n{{\"type\":\"stderr\",\"text\":{},\"timestamp\":2}}\n",
			serde_json::to_string(&marker).unwrap()
		);
		let output = parse_execution(body.as_bytes(), 5, &marker)
			.unwrap()
			.into_model_output(1024)
			.unwrap();
		assert_eq!(
			output,
			json!({
				"ok": true, "exit_code": 0, "stdout": "hello", "stderr": "",
				"timed_out": false, "truncated": true, "artifacts": []
			})
		);
	}

	#[tokio::test]
	async fn sandbox_termination_retries_exactly_once() {
		let server = MockServer::start().await;
		let attempts = Arc::new(AtomicUsize::new(0));
		let responder_attempts = attempts.clone();
		Mock::given(method("DELETE"))
			.and(path("/sandboxes/sandbox_1"))
			.respond_with(move |_: &wiremock::Request| {
				if responder_attempts.fetch_add(1, Ordering::SeqCst) == 0 {
					ResponseTemplate::new(503)
				} else {
					ResponseTemplate::new(204)
				}
			})
			.mount(&server)
			.await;
		let backend = E2bSandboxBackend::new(
			crate::test_helpers::policy_client(),
			server.uri().parse().unwrap(),
			"sandbox.example.com".into(),
			Duration::from_secs(2),
			SecretString::from("operator-key"),
			4096,
		)
		.unwrap();

		assert!(
			backend
				.terminate_sandbox("sandbox_1", Instant::now() + Duration::from_secs(2))
				.await
		);
		assert_eq!(attempts.load(Ordering::SeqCst), 2);
	}
}
