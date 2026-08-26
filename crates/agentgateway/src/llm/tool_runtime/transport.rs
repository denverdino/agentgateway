use ::http::StatusCode;

use super::ToolInfrastructureError;

pub(super) fn classify_proxy_error(error: crate::proxy::ProxyError) -> ToolInfrastructureError {
	match error {
		crate::proxy::ProxyError::RequestTimeout | crate::proxy::ProxyError::UpstreamCallTimeout => {
			ToolInfrastructureError::timeout()
		},
		_ => ToolInfrastructureError::backend(),
	}
}

pub(super) fn classify_status(
	status: StatusCode,
	gateway_timeout_is_timeout: bool,
) -> ToolInfrastructureError {
	if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
		ToolInfrastructureError::authentication()
	} else if gateway_timeout_is_timeout && status == StatusCode::GATEWAY_TIMEOUT {
		ToolInfrastructureError::timeout()
	} else {
		ToolInfrastructureError::backend()
	}
}

pub(super) fn truncate_utf8_bytes(value: &str, max_bytes: usize) -> String {
	if value.len() <= max_bytes {
		return value.to_owned();
	}
	let mut end = max_bytes;
	while !value.is_char_boundary(end) {
		end -= 1;
	}
	value[..end].to_owned()
}
