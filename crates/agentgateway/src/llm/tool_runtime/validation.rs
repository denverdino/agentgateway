use std::net::IpAddr;
use std::time::Duration;

use ::http::Uri;

use super::{
	RuntimeLimits, SANDBOX_MAX_BATCH_DEADLINE, ToolBackendConfig, ToolRuntimeConfig, ToolRuntimeError,
};

pub(super) fn validate_config(config: &ToolRuntimeConfig) -> Result<(), ToolRuntimeError> {
	validate_limits(&config.limits)?;
	if config.tools.is_empty() {
		return Err(invalid("tools must contain at least one tool"));
	}
	for tool in &config.tools {
		match &tool.backend {
			ToolBackendConfig::Http { url, timeout, .. } => {
				validate_endpoint("url", url)?;
				validate_backend_timeout("url", *timeout, None)?;
			},
			ToolBackendConfig::E2b {
				api_url,
				domain,
				timeout,
				..
			} => {
				validate_endpoint("apiUrl", api_url)?;
				if api_url.path() != "/" || api_url.query().is_some() {
					return Err(invalid(
						"tools[].backend.apiUrl must be an origin without path or query",
					));
				}
				validate_e2b_domain(domain)?;
				validate_backend_timeout("apiUrl", *timeout, Some(SANDBOX_MAX_BATCH_DEADLINE))?;
			},
		}
	}
	Ok(())
}

pub(super) fn validate_limits(limits: &RuntimeLimits) -> Result<(), ToolRuntimeError> {
	for (name, value) in [
		("maxRounds", limits.max_rounds),
		("maxToolCalls", limits.max_tool_calls),
		("maxParallelToolCalls", limits.max_parallel_tool_calls),
		("maxArgumentsBytes", limits.max_arguments_bytes),
		("maxOutputBytes", limits.max_output_bytes),
	] {
		if value == 0 {
			return Err(invalid(format!("{name} must be greater than zero")));
		}
	}
	if limits.total_timeout.is_zero() {
		return Err(invalid("totalTimeout must be greater than zero"));
	}
	if limits.max_parallel_tool_calls > limits.max_tool_calls {
		return Err(invalid("maxParallelToolCalls must not exceed maxToolCalls"));
	}
	Ok(())
}

pub(super) fn validate_backend_timeout(
	field: &str,
	timeout: Duration,
	maximum: Option<Duration>,
) -> Result<(), ToolRuntimeError> {
	if timeout.is_zero() {
		return Err(invalid(format!(
			"tools[].backend.{field} timeout must be greater than zero"
		)));
	}
	if maximum.is_some_and(|maximum| timeout > maximum) {
		return Err(invalid(format!(
			"tools[].backend.{field} timeout must not exceed 24 hours"
		)));
	}
	Ok(())
}

pub(super) fn validate_e2b_domain(domain: &str) -> Result<(), ToolRuntimeError> {
	if !valid_domain(domain) {
		return Err(invalid("tools[].backend.domain must be a valid hostname"));
	}
	Ok(())
}

pub(super) fn validate_endpoint(field: &str, endpoint: &Uri) -> Result<(), ToolRuntimeError> {
	let Some(host) = endpoint.host() else {
		return Err(invalid(format!(
			"tools[].backend.{field} must include a host"
		)));
	};
	if endpoint.scheme_str() != Some("https")
		&& !(endpoint.scheme_str() == Some("http") && is_loopback(host))
	{
		return Err(invalid(format!(
			"tools[].backend.{field} must use https unless its host is loopback"
		)));
	}
	Ok(())
}

pub(super) fn valid_domain(domain: &str) -> bool {
	!domain.is_empty()
		&& domain.len() <= 253
		&& domain.is_ascii()
		&& !domain.starts_with('.')
		&& !domain.ends_with('.')
		&& !domain.contains(['/', ':', '@', '*'])
		&& domain.split('.').all(|label| {
			!label.is_empty()
				&& label.len() <= 63
				&& !label.starts_with('-')
				&& !label.ends_with('-')
				&& label
					.bytes()
					.all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
		})
}

pub(super) fn is_loopback(host: &str) -> bool {
	host.eq_ignore_ascii_case("localhost")
		|| host
			.parse::<IpAddr>()
			.map(|address| address.is_loopback())
			.unwrap_or(false)
}

fn invalid(message: impl Into<String>) -> ToolRuntimeError {
	ToolRuntimeError::invalid_configuration(format!("llm.toolRuntime.{}", message.into()))
}
