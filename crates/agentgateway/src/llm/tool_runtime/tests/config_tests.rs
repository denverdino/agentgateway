use std::time::Duration;

use super::{ToolRegistry, ToolRuntimeConfig};

fn compile_yaml(yaml: &str) -> Result<(), String> {
	let config: ToolRuntimeConfig = serde_yaml::from_str(yaml).map_err(|error| error.to_string())?;
	ToolRegistry::compile(config)
		.map(|_| ())
		.map_err(|error| error.to_string())
}

#[test]
fn config_deserializes_with_defaults() {
	let config: ToolRuntimeConfig = serde_yaml::from_str(
		r#"
maxRounds: 3
tools:
- name: lookup
  backend:
    type: http
    url: http://127.0.0.1:18080
    timeout: 5s
"#,
	)
	.unwrap();

	assert_eq!(config.limits.max_rounds, 3);
	assert_eq!(config.limits.max_tool_calls, 16);
}

#[test]
fn config_explicit_null_limits_use_defaults() {
	let config: ToolRuntimeConfig = serde_yaml::from_str(
		r#"
maxRounds: null
maxToolCalls: null
maxParallelToolCalls: null
totalTimeout: null
maxArgumentsBytes: null
maxOutputBytes: null
tools:
- name: lookup
  backend:
    type: http
    url: http://127.0.0.1:18080
    timeout: 5s
"#,
	)
	.unwrap();

	assert_eq!(config.limits.max_rounds, 8);
	assert_eq!(config.limits.max_tool_calls, 16);
	assert_eq!(config.limits.max_parallel_tool_calls, 4);
	assert_eq!(config.limits.total_timeout, Duration::from_secs(120));
	assert_eq!(config.limits.max_arguments_bytes, 65_536);
	assert_eq!(config.limits.max_output_bytes, 1_048_576);
}

#[test]
fn config_accepts_legacy_function_compute_http_as_http() {
	let config: ToolRuntimeConfig = serde_yaml::from_str(
		r#"
tools:
- name: get_weather
  backend:
    type: functionComputeHttp
    url: https://weather.example.com
    timeout: 5s
"#,
	)
	.expect("legacy HTTP backend discriminator remains compatible");
	let normalized = serde_yaml::to_string(&config).unwrap();
	assert!(normalized.contains("type: http"));
	assert!(!normalized.contains("functionComputeHttp"));
}

#[test]
fn config_rejects_invalid_operator_configuration() {
	let cases = [
		(
			"duplicate public names",
			r#"
tools:
- name: get_weather
  backend: { type: http, url: https://weather.example.com, timeout: 5s }
- name: get_weather
  backend: { type: http, url: https://weather-2.example.com, timeout: 5s }
"#,
			"duplicate tool name",
		),
		(
			"duplicate builtin",
			r#"
tools:
- name: web_search_one
  builtin: webSearch
  backend: { type: http, url: https://search-one.example.com, timeout: 5s }
- name: web_search_two
  builtin: webSearch
  backend: { type: http, url: https://search-two.example.com, timeout: 5s }
"#,
			"duplicate builtin",
		),
		(
			"reserved prefix",
			r#"
tools:
- name: _agentgateway_internal
  backend: { type: http, url: https://weather.example.com, timeout: 5s }
"#,
			"_agentgateway_",
		),
		(
			"non HTTPS destination",
			r#"
tools:
- name: get_weather
  backend: { type: http, url: http://weather.example.com, timeout: 5s }
"#,
			"https",
		),
		(
			"zero limits",
			r#"
maxRounds: 0
tools:
- name: get_weather
  backend: { type: http, url: https://weather.example.com, timeout: 5s }
"#,
			"maxRounds",
		),
		(
			"parallel calls above total calls",
			r#"
maxToolCalls: 1
maxParallelToolCalls: 2
tools:
- name: get_weather
  backend: { type: http, url: https://weather.example.com, timeout: 5s }
"#,
			"maxParallelToolCalls",
		),
		("empty tool list", "tools: []", "at least one tool"),
		(
			"missing E2B API key",
			r#"
tools:
- name: code_interpreter
  builtin: codeInterpreter
  backend: { type: e2b, apiUrl: https://api.example.com, domain: example.com, timeout: 30s }
"#,
			"apiKey",
		),
		(
			"E2B API URL is not an origin",
			r#"
tools:
- name: code_interpreter
  builtin: codeInterpreter
  backend: { type: e2b, apiKey: secret, apiUrl: https://api.example.com/private, domain: example.com, timeout: 30s }
"#,
			"apiUrl",
		),
		(
			"invalid E2B domain",
			r#"
tools:
- name: code_interpreter
  builtin: codeInterpreter
  backend: { type: e2b, apiKey: secret, apiUrl: https://api.example.com, domain: https://example.com, timeout: 30s }
"#,
			"domain",
		),
	];

	for (name, yaml, expected_error) in cases {
		let error = compile_yaml(yaml).expect_err(name);
		assert!(
			error.contains(expected_error),
			"{name}: expected error containing {expected_error:?}, got {error}"
		);
	}

	for limit in [
		"maxRounds",
		"maxToolCalls",
		"maxParallelToolCalls",
		"totalTimeout",
		"maxArgumentsBytes",
		"maxOutputBytes",
	] {
		let yaml = format!(
			r#"
{limit}: 0{}
tools:
- name: get_weather
  backend: {{ type: http, url: https://weather.example.com, timeout: 5s }}
"#,
			if limit == "totalTimeout" { "s" } else { "" }
		);
		let error = compile_yaml(&yaml).expect_err("zero runtime limit must be rejected");
		assert!(
			error.contains(limit),
			"expected zero {limit} error, got {error}"
		);
	}

	let loopback = compile_yaml(
		r#"
tools:
- name: get_weather
  backend:
    type: http
    url: http://127.0.0.1:8080/weather
    timeout: 5s
"#,
	);
	assert!(
		loopback.is_ok(),
		"loopback HTTP endpoints are permitted for tests: {loopback:?}"
	);

	let non_http_loopback = compile_yaml(
		r#"
tools:
- name: get_weather
  backend:
    type: http
    url: ftp://127.0.0.1:8080/weather
    timeout: 5s
"#,
	);
	assert!(
		non_http_loopback.is_err(),
		"only HTTP loopback endpoints are permitted for tests"
	);
}

#[test]
fn config_e2b_timeout_matches_sandbox_deadline_boundary() {
	let config = |timeout: &str| {
		format!(
			r#"
totalTimeout: 86401s
tools:
- name: code_interpreter
  builtin: codeInterpreter
  backend:
    type: e2b
    apiKey: operator-token
    apiUrl: https://api.sandbox.invalid
    domain: sandbox.invalid
    timeout: {timeout}
"#,
		)
	};

	compile_yaml(&config("86400s"))
		.expect("the E2B backend accepts an exact 24-hour backend deadline");
	let error = compile_yaml(&config("86400001ms"))
		.expect_err("a backend deadline one millisecond over the adapter cap must fail at startup");
	assert!(
		error.contains("must not exceed 24 hours"),
		"unexpected startup error: {error}"
	);
}

#[test]
fn config_rejects_incompatible_tool_backend_pairs_without_leaking_secrets() {
	for (name, yaml, expected_error) in [
		(
			"code interpreter on ordinary HTTP backend",
			r#"
tools:
- name: code_interpreter
  builtin: codeInterpreter
  backend:
    type: http
    url: https://functions.invalid/code
    timeout: 5s
    bearerToken: should-never-leak-token
"#,
			"codeInterpreter requires e2b",
		),
		(
			"web search on E2B",
			r#"
tools:
- name: web_search
  builtin: webSearch
  backend:
    type: e2b
    apiKey: should-never-leak-token
    apiUrl: https://api.sandbox.invalid
    domain: sandbox.invalid
    timeout: 5s
"#,
			"webSearch requires http",
		),
		(
			"ordinary function on E2B",
			r#"
tools:
- name: get_weather
  backend:
    type: e2b
    apiKey: should-never-leak-token
    apiUrl: https://api.sandbox.invalid
    domain: sandbox.invalid
    timeout: 5s
"#,
			"ordinary functions require http",
		),
	] {
		let error = compile_yaml(yaml).expect_err(name);
		assert!(error.contains(expected_error), "{name}: {error}");
		assert!(
			!error.contains("should-never-leak-token"),
			"{name}: {error}"
		);
	}
}
