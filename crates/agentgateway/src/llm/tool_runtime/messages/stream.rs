use serde_json::{Value, json};

use crate::llm::types::messages;

pub(crate) fn encode_managed_streaming_response(
	response: &messages::Response,
) -> Result<Vec<u8>, serde_json::Error> {
	let mut out = Vec::new();
	push_event(
		&mut out,
		"message_start",
		&json!({"type": "message_start", "message": message_shell(response)?}),
	)?;
	for (index, block) in response.content.iter().enumerate() {
		let full = serde_json::to_value(block)?;
		let kind = full.get("type").and_then(Value::as_str).unwrap_or_default();
		let (start, deltas) = match kind {
			"text" => (
				json!({"type": "text", "text": ""}),
				vec![json!({"type": "text_delta", "text": block.text.clone().unwrap_or_default()})],
			),
			"thinking" => {
				let thinking = full
					.get("thinking")
					.and_then(Value::as_str)
					.unwrap_or_default();
				let mut deltas = vec![json!({"type": "thinking_delta", "thinking": thinking})];
				if let Some(signature) = full.get("signature").and_then(Value::as_str) {
					deltas.push(json!({"type": "signature_delta", "signature": signature}));
				}
				(json!({"type": "thinking", "thinking": ""}), deltas)
			},
			"tool_use" => {
				let input = full.get("input").cloned().unwrap_or_else(|| json!({}));
				(
					json!({
						"type": "tool_use",
						"id": full.get("id").cloned().unwrap_or(Value::Null),
						"name": full.get("name").cloned().unwrap_or(Value::Null),
						"input": {},
					}),
					vec![json!({
						"type": "input_json_delta",
						"partial_json": serde_json::to_string(&input)?,
					})],
				)
			},
			_ => (full, Vec::new()),
		};
		push_event(
			&mut out,
			"content_block_start",
			&json!({"type": "content_block_start", "index": index, "content_block": start}),
		)?;
		for delta in deltas {
			push_event(
				&mut out,
				"content_block_delta",
				&json!({"type": "content_block_delta", "index": index, "delta": delta}),
			)?;
		}
		push_event(
			&mut out,
			"content_block_stop",
			&json!({"type": "content_block_stop", "index": index}),
		)?;
	}
	push_event(
		&mut out,
		"message_delta",
		&json!({
			"type": "message_delta",
			"delta": {
				"stop_reason": response.stop_reason,
				"stop_sequence": response.stop_sequence,
			},
			"usage": {"output_tokens": response.usage.output_tokens},
		}),
	)?;
	push_event(&mut out, "message_stop", &json!({"type": "message_stop"}))?;
	Ok(out)
}

fn message_shell(response: &messages::Response) -> Result<Value, serde_json::Error> {
	let mut shell = serde_json::to_value(response)?;
	let object = shell
		.as_object_mut()
		.expect("a struct always serializes to a JSON object");
	object.insert("content".to_owned(), json!([]));
	object.insert("stop_reason".to_owned(), Value::Null);
	object.insert("stop_sequence".to_owned(), Value::Null);
	if let Some(usage) = object.get_mut("usage").and_then(Value::as_object_mut) {
		usage.insert("output_tokens".to_owned(), json!(0));
	}
	Ok(shell)
}

fn push_event(out: &mut Vec<u8>, event: &str, data: &Value) -> Result<(), serde_json::Error> {
	out.extend_from_slice(b"event: ");
	out.extend_from_slice(event.as_bytes());
	out.extend_from_slice(b"\ndata: ");
	out.extend_from_slice(&serde_json::to_vec(data)?);
	out.extend_from_slice(b"\n\n");
	Ok(())
}
