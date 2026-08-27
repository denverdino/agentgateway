use agent_core::prelude::Strng;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{
	SANDBOX_MAX_CODE_BYTES, ToolBatchInfrastructureError, ToolBatchMetadata, ToolExecutionContext,
	ToolExecutionResult, ToolRuntimeError,
};

const PROTOCOL_PREFIX: &str = "__AGENTGATEWAY_PTC_V1__";
const PROTOCOL_VERSION: u64 = 1;
const MAX_ERROR_BYTES: usize = 1024;
const MAX_NONCE_BYTES: usize = 128;

const PYTHON_WRAPPER: &str = include_str!("program_wrapper.py");

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ProgramReplayEntry {
	pub(crate) sequence: usize,
	#[serde(rename = "name")]
	pub(crate) public_name: Strng,
	pub(crate) arguments: Value,
	pub(crate) output: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ProgramPendingCall {
	pub(crate) sequence: usize,
	pub(crate) public_name: Strng,
	pub(crate) arguments: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ProgramOutcome {
	Pending(ProgramPendingCall),
	Completed(Value),
	ContractError { message: String },
	ApplicationError { error_type: String, message: String },
}

#[derive(Clone, Debug)]
pub(crate) struct ProgramSandboxRequest {
	pub(crate) source: String,
	pub(crate) env_vars: Map<String, Value>,
}

pub(crate) struct ProgramSandboxExecution {
	pub(crate) result: ToolExecutionResult,
	pub(crate) metadata: ToolBatchMetadata,
}

#[async_trait]
pub(crate) trait ProgramSandbox: Send + Sync {
	async fn run(
		&self,
		request: ProgramSandboxRequest,
		context: ToolExecutionContext,
	) -> Result<ProgramSandboxExecution, ToolBatchInfrastructureError>;
}

pub(crate) fn build_sandbox_request(
	code: &str,
	replay: &[ProgramReplayEntry],
	nonce: &str,
	max_bytes: usize,
) -> Result<ProgramSandboxRequest, ToolRuntimeError> {
	if code.len() > SANDBOX_MAX_CODE_BYTES {
		return Err(ToolRuntimeError::invalid_request(
			"programmatic code exceeds the 32768-byte limit",
		));
	}
	if nonce.is_empty()
		|| nonce.len() > MAX_NONCE_BYTES
		|| !nonce
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
	{
		return Err(ToolRuntimeError::internal());
	}
	let replay = serde_json::to_vec(replay).map_err(|_| ToolRuntimeError::internal())?;
	if replay.len() > max_bytes {
		return Err(ToolRuntimeError::invalid_request(
			"programmatic replay exceeds the configured output limit",
		));
	}
	let encoded_code = STANDARD.encode(code.as_bytes());
	let encoded_replay = STANDARD.encode(replay);
	Ok(ProgramSandboxRequest {
		source: PYTHON_WRAPPER.to_owned(),
		env_vars: Map::from_iter([
			(
				"AGENTGATEWAY_PTC_CODE".to_owned(),
				Value::String(encoded_code),
			),
			(
				"AGENTGATEWAY_PTC_REPLAY".to_owned(),
				Value::String(encoded_replay),
			),
			(
				"AGENTGATEWAY_PTC_NONCE".to_owned(),
				Value::String(nonce.to_owned()),
			),
		]),
	})
}

pub(crate) fn parse_sandbox_outcome(
	stdout: &str,
	nonce: &str,
	max_bytes: usize,
) -> Result<ProgramOutcome, ToolRuntimeError> {
	if stdout.len() > protocol_stdout_max_bytes(max_bytes, nonce.len()) {
		return Err(ToolRuntimeError::invalid_request(
			"programmatic protocol output exceeds the configured output limit",
		));
	}
	let expected = format!("{PROTOCOL_PREFIX}{nonce}:");
	let mut frames = stdout
		.lines()
		.filter_map(|line| line.strip_prefix(&expected));
	let frame = frames.next().ok_or_else(|| {
		ToolRuntimeError::invalid_request("program sandbox returned no protocol outcome")
	})?;
	if frames.next().is_some() {
		return Err(ToolRuntimeError::invalid_request(
			"program sandbox returned multiple protocol outcomes",
		));
	}
	let payload = URL_SAFE_NO_PAD.decode(frame).map_err(|_| {
		ToolRuntimeError::invalid_request("program sandbox returned an invalid protocol outcome")
	})?;
	if payload.len() > max_bytes {
		return Err(ToolRuntimeError::invalid_request(
			"programmatic protocol output exceeds the configured output limit",
		));
	}
	let payload: Value = serde_json::from_slice(&payload).map_err(|_| {
		ToolRuntimeError::invalid_request("program sandbox returned an invalid protocol outcome")
	})?;
	if payload.get("version").and_then(Value::as_u64) != Some(PROTOCOL_VERSION) {
		return Err(ToolRuntimeError::invalid_request(
			"program sandbox returned an unsupported protocol version",
		));
	}
	match payload.get("kind").and_then(Value::as_str) {
		Some("pending") => {
			let sequence = payload
				.get("sequence")
				.and_then(Value::as_u64)
				.and_then(|value| usize::try_from(value).ok())
				.ok_or_else(|| ToolRuntimeError::invalid_request("invalid pending sequence"))?;
			let public_name = payload
				.get("name")
				.and_then(Value::as_str)
				.filter(|name| !name.is_empty())
				.ok_or_else(|| ToolRuntimeError::invalid_request("invalid pending tool name"))?;
			let arguments = payload
				.get("arguments")
				.filter(|value| value.is_object())
				.cloned()
				.ok_or_else(|| ToolRuntimeError::invalid_request("invalid pending arguments"))?;
			Ok(ProgramOutcome::Pending(ProgramPendingCall {
				sequence,
				public_name: Strng::from(public_name),
				arguments,
			}))
		},
		Some("completed") => Ok(ProgramOutcome::Completed(
			payload.get("output").cloned().unwrap_or(Value::Null),
		)),
		Some("contract_error") => Ok(ProgramOutcome::ContractError {
			message: bounded_protocol_string(payload.get("message"), "program replay contract failed"),
		}),
		Some("error") => {
			let error_type = bounded_protocol_string(payload.get("error_type"), "program_error");
			let message = bounded_protocol_string(payload.get("message"), "program execution failed");
			Ok(ProgramOutcome::ApplicationError {
				error_type,
				message,
			})
		},
		_ => Err(ToolRuntimeError::invalid_request(
			"program sandbox returned an invalid outcome kind",
		)),
	}
}

pub(crate) fn program_protocol_stdout_max_bytes(max_payload_bytes: usize) -> usize {
	protocol_stdout_max_bytes(max_payload_bytes, MAX_NONCE_BYTES)
}

pub(crate) fn has_replay_entry_capacity(
	replay: &[ProgramReplayEntry],
	entry: &ProgramReplayEntry,
	max_bytes: usize,
) -> bool {
	let mut candidate = replay.to_vec();
	candidate.push(ProgramReplayEntry {
		output: replay_truncated_output(),
		..entry.clone()
	});
	serde_json::to_vec(&candidate).is_ok_and(|candidate| candidate.len() <= max_bytes)
}

pub(crate) fn fit_replay_entry(
	replay: &[ProgramReplayEntry],
	mut entry: ProgramReplayEntry,
	max_bytes: usize,
) -> Result<ProgramReplayEntry, ToolRuntimeError> {
	let mut candidate = replay.to_vec();
	candidate.push(entry.clone());
	if serde_json::to_vec(&candidate)
		.map_err(|_| ToolRuntimeError::internal())?
		.len()
		<= max_bytes
	{
		return Ok(entry);
	}

	let output = std::mem::replace(&mut entry.output, Value::Null);
	let mut envelope = replay.to_vec();
	envelope.push(entry.clone());
	let envelope_bytes = serde_json::to_vec(&envelope)
		.map_err(|_| ToolRuntimeError::internal())?
		.len();
	let output_bytes = max_bytes
		.checked_sub(envelope_bytes)
		.and_then(|remaining| remaining.checked_add(4))
		.ok_or_else(|| {
			ToolRuntimeError::invalid_request(
				"programmatic replay has no capacity for another tool result",
			)
		})?;
	entry.output = output;
	if super::bound_replay_output(&mut entry.output, output_bytes).is_err() {
		entry.output = replay_truncated_output();
	}
	Ok(entry)
}

fn replay_truncated_output() -> Value {
	serde_json::json!({
		"ok": false,
		"error": {
			"type": "output_truncated",
			"message": "tool output did not fit in the program replay transcript",
			"retryable": false
		},
		"truncated": true
	})
}

fn protocol_stdout_max_bytes(max_payload_bytes: usize, nonce_bytes: usize) -> usize {
	let encoded_payload_bytes =
		(max_payload_bytes / 3)
			.saturating_mul(4)
			.saturating_add(match max_payload_bytes % 3 {
				0 => 0,
				1 => 2,
				_ => 3,
			});
	encoded_payload_bytes
		.saturating_add(PROTOCOL_PREFIX.len())
		.saturating_add(nonce_bytes)
		.saturating_add(2)
}

fn bounded_protocol_string(value: Option<&Value>, fallback: &str) -> String {
	let value = value.and_then(Value::as_str).unwrap_or(fallback);
	let mut end = value.len().min(MAX_ERROR_BYTES);
	while !value.is_char_boundary(end) {
		end -= 1;
	}
	value[..end].to_owned()
}

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;

	fn protocol_line(nonce: &str, payload: Value) -> String {
		let payload = serde_json::to_vec(&payload).unwrap();
		format!(
			"{PROTOCOL_PREFIX}{nonce}:{}",
			URL_SAFE_NO_PAD.encode(payload)
		)
	}

	#[test]
	fn python_wrapper_requests_first_unresolved_call() {
		let request = build_sandbox_request(
			"fact = tools.call('web_search', {'query':'AgentGateway'})\nprogram_output(fact)",
			&[],
			"nonce-1",
			1_048_576,
		)
		.unwrap();
		assert_eq!(request.env_vars["AGENTGATEWAY_PTC_NONCE"], "nonce-1");
		assert!(request.source.contains("class _Tools"));
		assert!(request.source.contains("def program_output"));
		assert!(request.source.contains("allow_nan=False"));
		assert!(request.source.contains("sort_keys=True"));
		assert_eq!(request.env_vars.len(), 3);
	}

	#[test]
	fn protocol_parser_accepts_only_matching_nonce_and_version() {
		let line = protocol_line(
			"nonce-1",
			json!({
				"version":1, "kind":"pending", "sequence":0,
				"name":"web_search", "arguments":{"query":"AgentGateway"}
			}),
		);
		assert_eq!(
			parse_sandbox_outcome(&line, "nonce-1", 4096).unwrap(),
			ProgramOutcome::Pending(ProgramPendingCall {
				sequence: 0,
				public_name: "web_search".into(),
				arguments: json!({"query":"AgentGateway"}),
			})
		);
		assert!(parse_sandbox_outcome(&line, "different", 4096).is_err());
		let wrong_version = protocol_line("nonce-1", json!({"version":2,"kind":"completed"}));
		assert!(parse_sandbox_outcome(&wrong_version, "nonce-1", 4096).is_err());
		assert_eq!(
			parse_sandbox_outcome(
				&protocol_line(
					"nonce-1",
					json!({"version":1,"kind":"contract_error","message":"replay diverged"}),
				),
				"nonce-1",
				4096,
			)
			.unwrap(),
			ProgramOutcome::ContractError {
				message: "replay diverged".to_owned()
			}
		);
	}

	#[test]
	fn builder_serializes_exact_replay_and_enforces_bounds() {
		let replay = [ProgramReplayEntry {
			sequence: 0,
			public_name: "weather.get_forecast".into(),
			arguments: json!({"q":"Shanghai","days":3}),
			output: json!({"forecast":[]}),
		}];
		let request = build_sandbox_request("program_output(1)", &replay, "n-1", 4096).unwrap();
		let encoded = request.env_vars["AGENTGATEWAY_PTC_REPLAY"]
			.as_str()
			.unwrap();
		let decoded: Value = serde_json::from_slice(&STANDARD.decode(encoded).unwrap()).unwrap();
		assert_eq!(decoded[0]["name"], "weather.get_forecast");
		assert!(
			build_sandbox_request(&"x".repeat(SANDBOX_MAX_CODE_BYTES + 1), &[], "n", 4096).is_err()
		);
		assert!(build_sandbox_request("pass", &replay, "n", 1).is_err());
	}

	#[test]
	fn builder_allows_independently_bounded_code_and_replay() {
		let replay = [ProgramReplayEntry {
			sequence: 0,
			public_name: "tool".into(),
			arguments: json!({}),
			output: json!({"value": "x".repeat(330)}),
		}];
		assert!(serde_json::to_vec(&replay).unwrap().len() <= 512);

		build_sandbox_request(&"x".repeat(400), &replay, "n", 512)
			.expect("bounded code and replay must fit independent environment limits");
	}

	#[test]
	fn parser_bounds_errors_and_rejects_ambiguous_frames() {
		let line = protocol_line(
			"n",
			json!({
				"version":1,
				"kind":"error",
				"error_type":"é".repeat(800),
				"message":"m".repeat(2048)
			}),
		);
		let ProgramOutcome::ApplicationError {
			error_type,
			message,
		} = parse_sandbox_outcome(&line, "n", 8192).unwrap()
		else {
			panic!("expected application error")
		};
		assert!(error_type.len() <= MAX_ERROR_BYTES);
		assert!(message.len() <= MAX_ERROR_BYTES);
		assert!(parse_sandbox_outcome(&format!("{line}\n{line}"), "n", 16384).is_err());
	}

	#[test]
	fn parser_accepts_encoded_frame_when_decoded_payload_fits_limit() {
		let line = protocol_line(
			"n",
			json!({
				"version":1,
				"kind":"completed",
				"output":"x".repeat(3500)
			}),
		);
		assert!(line.len() > 4096, "fixture must exercise base64 expansion");

		assert_eq!(
			parse_sandbox_outcome(&line, "n", 4096).unwrap(),
			ProgramOutcome::Completed(Value::String("x".repeat(3500)))
		);
	}
}
