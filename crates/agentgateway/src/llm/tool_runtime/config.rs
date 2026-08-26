use std::time::Duration;

use ::http::Uri;
use secrecy::SecretString;

use crate::{apply, schema, schema_enum};

#[apply(schema!)]
#[cfg_attr(feature = "schema", schemars(rename = "LocalToolRuntimeConfig"))]
pub struct ToolRuntimeConfig {
	#[serde(flatten)]
	#[cfg_attr(feature = "schema", schemars(flatten))]
	pub limits: RuntimeLimits,
	pub tools: Vec<ManagedToolConfig>,
}

#[apply(schema!)]
pub struct RuntimeLimits {
	#[serde(
		default = "default_max_rounds",
		deserialize_with = "deserialize_max_rounds"
	)]
	#[cfg_attr(
		feature = "schema",
		schemars(with = "Option<usize>", transform = remove_schema_default)
	)]
	pub max_rounds: usize,
	#[serde(
		default = "default_max_tool_calls",
		deserialize_with = "deserialize_max_tool_calls"
	)]
	#[cfg_attr(
		feature = "schema",
		schemars(with = "Option<usize>", transform = remove_schema_default)
	)]
	pub max_tool_calls: usize,
	#[serde(
		default = "default_max_parallel_tool_calls",
		deserialize_with = "deserialize_max_parallel_tool_calls"
	)]
	#[cfg_attr(
		feature = "schema",
		schemars(with = "Option<usize>", transform = remove_schema_default)
	)]
	pub max_parallel_tool_calls: usize,
	#[serde(
		default = "default_total_timeout",
		serialize_with = "crate::serdes::serde_dur::serialize",
		deserialize_with = "deserialize_total_timeout"
	)]
	#[cfg_attr(
		feature = "schema",
		schemars(with = "Option<String>", transform = set_null_schema_default)
	)]
	pub total_timeout: Duration,
	#[serde(
		default = "default_max_arguments_bytes",
		deserialize_with = "deserialize_max_arguments_bytes"
	)]
	#[cfg_attr(
		feature = "schema",
		schemars(with = "Option<usize>", transform = remove_schema_default)
	)]
	pub max_arguments_bytes: usize,
	#[serde(
		default = "default_max_output_bytes",
		deserialize_with = "deserialize_max_output_bytes"
	)]
	#[cfg_attr(
		feature = "schema",
		schemars(with = "Option<usize>", transform = remove_schema_default)
	)]
	pub max_output_bytes: usize,
}

fn deserialize_default_on_null<'de, D, T>(
	deserializer: D,
	default: fn() -> T,
) -> Result<T, D::Error>
where
	D: serde::Deserializer<'de>,
	T: serde::Deserialize<'de>,
{
	<Option<T> as serde::Deserialize>::deserialize(deserializer)
		.map(|value| value.unwrap_or_else(default))
}

fn deserialize_max_rounds<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
	D: serde::Deserializer<'de>,
{
	deserialize_default_on_null(deserializer, default_max_rounds)
}

fn deserialize_max_tool_calls<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
	D: serde::Deserializer<'de>,
{
	deserialize_default_on_null(deserializer, default_max_tool_calls)
}

fn deserialize_max_parallel_tool_calls<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
	D: serde::Deserializer<'de>,
{
	deserialize_default_on_null(deserializer, default_max_parallel_tool_calls)
}

fn deserialize_total_timeout<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
	D: serde::Deserializer<'de>,
{
	crate::serdes::serde_dur_option::deserialize(deserializer)
		.map(|value| value.unwrap_or_else(default_total_timeout))
}

fn deserialize_max_arguments_bytes<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
	D: serde::Deserializer<'de>,
{
	deserialize_default_on_null(deserializer, default_max_arguments_bytes)
}

fn deserialize_max_output_bytes<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
	D: serde::Deserializer<'de>,
{
	deserialize_default_on_null(deserializer, default_max_output_bytes)
}

// Preserve the schema emitted by the former Option-backed local limits while the canonical
// runtime fields retain their concrete deserialization defaults.
#[cfg(feature = "schema")]
fn remove_schema_default(schema: &mut schemars::Schema) {
	schema.remove("default");
}

#[cfg(feature = "schema")]
fn set_null_schema_default(schema: &mut schemars::Schema) {
	schema.insert("default".into(), serde_json::Value::Null);
}

impl Default for RuntimeLimits {
	fn default() -> Self {
		Self {
			max_rounds: default_max_rounds(),
			max_tool_calls: default_max_tool_calls(),
			max_parallel_tool_calls: default_max_parallel_tool_calls(),
			total_timeout: default_total_timeout(),
			max_arguments_bytes: default_max_arguments_bytes(),
			max_output_bytes: default_max_output_bytes(),
		}
	}
}

const fn default_max_rounds() -> usize {
	8
}

const fn default_max_tool_calls() -> usize {
	16
}

const fn default_max_parallel_tool_calls() -> usize {
	4
}

const fn default_total_timeout() -> Duration {
	Duration::from_secs(120)
}

const fn default_max_arguments_bytes() -> usize {
	65_536
}

const fn default_max_output_bytes() -> usize {
	1_048_576
}

#[apply(schema!)]
#[cfg_attr(feature = "schema", schemars(rename = "LocalManagedTool"))]
pub struct ManagedToolConfig {
	pub name: String,
	pub builtin: Option<BuiltinTool>,
	pub backend: ToolBackendConfig,
}

#[apply(schema_enum!)]
#[cfg_attr(feature = "schema", schemars(rename = "LocalBuiltinTool"))]
pub enum BuiltinTool {
	WebSearch,
	CodeInterpreter,
}

#[apply(schema!)]
#[serde(tag = "type")]
#[cfg_attr(feature = "schema", schemars(rename = "LocalToolBackend"))]
pub enum ToolBackendConfig {
	#[serde(alias = "functionComputeHttp")]
	Http {
		#[serde(with = "http_serde::uri")]
		#[cfg_attr(feature = "schema", schemars(with = "String"))]
		url: Uri,
		#[serde(with = "crate::serdes::serde_dur")]
		#[cfg_attr(feature = "schema", schemars(with = "String"))]
		timeout: Duration,
		#[serde(
			rename = "bearerToken",
			default,
			skip_serializing_if = "Option::is_none",
			serialize_with = "crate::serdes::ser_redact",
			deserialize_with = "crate::serdes::deser_key_from_file_option"
		)]
		#[cfg_attr(
			feature = "schema",
			schemars(with = "Option<crate::serdes::FileOrInline>")
		)]
		bearer_token: Option<SecretString>,
	},
	E2b {
		#[serde(rename = "apiUrl")]
		#[serde(with = "http_serde::uri")]
		#[cfg_attr(feature = "schema", schemars(with = "String"))]
		api_url: Uri,
		domain: String,
		#[serde(with = "crate::serdes::serde_dur")]
		#[cfg_attr(feature = "schema", schemars(with = "String"))]
		timeout: Duration,
		#[serde(
			rename = "apiKey",
			serialize_with = "crate::serdes::ser_redact",
			deserialize_with = "crate::serdes::deser_key_from_file"
		)]
		#[cfg_attr(feature = "schema", schemars(with = "crate::serdes::FileOrInline"))]
		api_key: SecretString,
	},
}
