use crate::mistral_v3::config::MistralV3Config;
use crate::mistral_v3::model::MistralV3ForCausalLM;
use std::collections::HashMap;
use thiserror::Error;
use trustformers_core::errors::Result as TFResult;

// ─────────────────────────────────────────────────────────────────────────────
// Tool definition types
// ─────────────────────────────────────────────────────────────────────────────

/// A function/tool definition for Mistral v0.3 function calling
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Tool name (must be unique within a session)
    pub name: String,
    /// Human-readable description of what the tool does
    pub description: String,
    /// Named parameters accepted by this tool
    pub parameters: HashMap<String, ToolParameter>,
}

/// A single parameter within a `ToolDefinition`
#[derive(Debug, Clone)]
pub struct ToolParameter {
    /// JSON Schema type string: `"string"`, `"integer"`, `"boolean"`, `"array"`, `"object"`
    pub param_type: String,
    /// Human-readable description
    pub description: String,
    /// Whether this parameter must be provided in a tool call
    pub required: bool,
    /// Optional enumerated allowed values (for `"string"` params)
    pub enum_values: Option<Vec<String>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool call parsed from model output
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed tool call extracted from `[TOOL_CALLS] [...]` in model output
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Name of the tool to invoke
    pub name: String,
    /// Arguments as a JSON object
    pub arguments: HashMap<String, serde_json::Value>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur when parsing or validating tool calls
#[derive(Debug, Error)]
pub enum ToolParseError {
    #[error("No tool calls found in output")]
    NoToolCalls,
    #[error("JSON parse error: {0}")]
    JsonError(String),
    #[error("Unknown tool: {0}")]
    UnknownTool(String),
    #[error("Missing required param: {0}")]
    MissingParam(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// MistralV3FunctionCaller
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a `MistralV3ForCausalLM` with a registry of tool definitions,
/// providing prompt formatting and tool call parsing / validation.
pub struct MistralV3FunctionCaller {
    inner: MistralV3ForCausalLM,
    tools: Vec<ToolDefinition>,
}

impl MistralV3FunctionCaller {
    /// Create a new function caller with the given model and tools.
    pub fn new(inner: MistralV3ForCausalLM, tools: Vec<ToolDefinition>) -> Self {
        Self { inner, tools }
    }

    /// Access the underlying model configuration.
    pub fn config(&self) -> &MistralV3Config {
        self.inner.config()
    }

    /// Format the system prompt listing all available tools.
    ///
    /// Format: `[AVAILABLE_TOOLS] [{...},...] [/AVAILABLE_TOOLS]`
    ///
    /// Each tool is serialised as:
    /// ```json
    /// {"type":"function","function":{"name":"...","description":"...","parameters":{...}}}
    /// ```
    pub fn format_tool_prompt(&self) -> String {
        let tool_jsons: Vec<serde_json::Value> = self
            .tools
            .iter()
            .map(|tool| {
                let params: serde_json::Map<String, serde_json::Value> = tool
                    .parameters
                    .iter()
                    .map(|(k, v)| {
                        let mut obj = serde_json::Map::new();
                        obj.insert("type".to_string(), serde_json::json!(v.param_type));
                        obj.insert("description".to_string(), serde_json::json!(v.description));
                        if let Some(ref enums) = v.enum_values {
                            obj.insert("enum".to_string(), serde_json::json!(enums));
                        }
                        (k.clone(), serde_json::Value::Object(obj))
                    })
                    .collect();

                let required: Vec<String> = tool
                    .parameters
                    .iter()
                    .filter(|(_, v)| v.required)
                    .map(|(k, _)| k.clone())
                    .collect();

                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": params,
                            "required": required,
                        }
                    }
                })
            })
            .collect();

        let tools_json = serde_json::to_string(&tool_jsons).unwrap_or_else(|_| "[]".to_string());

        format!("[AVAILABLE_TOOLS] {tools_json}[/AVAILABLE_TOOLS]")
    }

    /// Parse `[TOOL_CALLS] [{"name": "...", "arguments": {...}}]` from model output.
    ///
    /// Returns all tool calls found in the marker.  Text outside the marker is
    /// ignored.
    pub fn parse_tool_calls(output: &str) -> Result<Vec<ToolCall>, ToolParseError> {
        const MARKER: &str = "[TOOL_CALLS]";

        let marker_pos = output.find(MARKER).ok_or(ToolParseError::NoToolCalls)?;
        let after_marker = &output[marker_pos + MARKER.len()..].trim_start();

        // Find the JSON array: starts with '[' and ends at the matching ']'
        let json_start = after_marker
            .find('[')
            .ok_or_else(|| ToolParseError::JsonError("no JSON array found".to_string()))?;
        let json_candidate = &after_marker[json_start..];

        // Find balanced closing bracket
        let json_str = balanced_json_array(json_candidate)
            .ok_or_else(|| ToolParseError::JsonError("unbalanced brackets".to_string()))?;

        let raw: Vec<serde_json::Value> =
            serde_json::from_str(json_str).map_err(|e| ToolParseError::JsonError(e.to_string()))?;

        let mut calls = Vec::with_capacity(raw.len());
        for item in raw {
            let name = item
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolParseError::JsonError("tool call missing 'name'".to_string()))?
                .to_string();

            let arguments_val = item.get("arguments").cloned().unwrap_or(serde_json::json!({}));
            let arguments: HashMap<String, serde_json::Value> =
                if let serde_json::Value::Object(map) = arguments_val {
                    map.into_iter().collect()
                } else {
                    return Err(ToolParseError::JsonError(
                        "'arguments' must be a JSON object".to_string(),
                    ));
                };

            calls.push(ToolCall { name, arguments });
        }

        if calls.is_empty() {
            return Err(ToolParseError::NoToolCalls);
        }

        Ok(calls)
    }

    /// Validate a parsed tool call against the registered tool definitions.
    ///
    /// Checks:
    /// 1. The tool name is known.
    /// 2. All required parameters are present in `call.arguments`.
    pub fn validate_tool_call(&self, call: &ToolCall) -> Result<(), ToolParseError> {
        let definition = self
            .tools
            .iter()
            .find(|t| t.name == call.name)
            .ok_or_else(|| ToolParseError::UnknownTool(call.name.clone()))?;

        for (param_name, param_def) in &definition.parameters {
            if param_def.required && !call.arguments.contains_key(param_name) {
                return Err(ToolParseError::MissingParam(param_name.clone()));
            }
        }

        Ok(())
    }

    /// Run a forward pass on the underlying model.
    pub fn forward(&self, input_ids: Vec<u32>) -> TFResult<trustformers_core::tensor::Tensor> {
        self.inner.forward(input_ids)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: extract a balanced JSON array string
// ─────────────────────────────────────────────────────────────────────────────

/// Walk the string starting at the first `[` and return the slice up to and
/// including the matching `]`, respecting nested brackets and strings.
fn balanced_json_array(s: &str) -> Option<&str> {
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escape_next = false;
    let mut end_idx = None;

    for (byte_idx, ch) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if in_string {
            if ch == '\\' {
                escape_next = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    end_idx = Some(byte_idx + ch.len_utf8());
                    break;
                }
            },
            _ => {},
        }
    }

    end_idx.map(|end| &s[..end])
}
