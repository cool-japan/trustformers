use crate::mistral_v3::config::MistralV3Config;
use crate::mistral_v3::model::MistralV3ForCausalLM;
use std::collections::HashMap;
use thiserror::Error;
use trustformers_core::errors::Result as TFResult;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors specific to Mistral v0.3 tasks and function calling
#[derive(Debug, Error)]
pub enum MistralV3Error {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("no tool calls found in model output")]
    NoToolCalls,
    #[error("JSON parse error: {0}")]
    JsonParse(String),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("missing required parameter: {0}")]
    MissingRequiredParam(String),
    #[error("tensor operation error: {0}")]
    TensorOp(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool calling special tokens (v0.3 additions)
// ─────────────────────────────────────────────────────────────────────────────

/// Mistral v0.3 special tokens for function / tool calling
#[derive(Debug, Clone)]
pub struct ToolUseTokens {
    /// Marks the start of model-generated tool calls
    pub tool_call_start: String,
    /// Marks the end of model-generated tool calls
    pub tool_call_end: String,
    /// Marks the start of tool execution results injected into the prompt
    pub tool_results_start: String,
    /// Marks the end of tool execution results
    pub tool_results_end: String,
    /// Marks the start of the available-tools block in the system prompt
    pub available_tools_start: String,
    /// Marks the end of the available-tools block
    pub available_tools_end: String,
}

impl Default for ToolUseTokens {
    fn default() -> Self {
        Self {
            tool_call_start: "[TOOL_CALLS]".to_string(),
            tool_call_end: "[/TOOL_CALLS]".to_string(),
            tool_results_start: "[TOOL_RESULTS]".to_string(),
            tool_results_end: "[/TOOL_RESULTS]".to_string(),
            available_tools_start: "[AVAILABLE_TOOLS]".to_string(),
            available_tools_end: "[/AVAILABLE_TOOLS]".to_string(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool schema types
// ─────────────────────────────────────────────────────────────────────────────

/// A single parameter in a tool's schema
#[derive(Debug, Clone)]
pub struct ToolParameter {
    /// Parameter name
    pub name: String,
    /// JSON Schema type: `"string"`, `"integer"`, `"boolean"`, `"array"`, `"object"`
    pub param_type: String,
    /// Human-readable description
    pub description: String,
    /// Whether this parameter must be provided
    pub required: bool,
}

/// A complete tool definition for Mistral v0.3 function calling
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Unique tool name
    pub name: String,
    /// Human-readable description of what the tool does
    pub description: String,
    /// Ordered list of parameters
    pub parameters: Vec<ToolParameter>,
}

impl ToolDefinition {
    /// Collect all required parameter names
    pub fn required_params(&self) -> Vec<&str> {
        self.parameters.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool call request (parsed from model output)
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed tool call extracted from the model's `[TOOL_CALLS]...[/TOOL_CALLS]` output
#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    /// Name of the tool to invoke
    pub tool_name: String,
    /// String-typed arguments (caller is responsible for further parsing)
    pub arguments: HashMap<String, String>,
    /// Unique identifier for this call (used to match results)
    pub call_id: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Prompt formatting
// ─────────────────────────────────────────────────────────────────────────────

/// Format a system prompt that lists available tools and appends the user message.
///
/// Output structure:
/// ```text
/// [AVAILABLE_TOOLS] [{"type":"function","function":{...}},...] [/AVAILABLE_TOOLS]
///
/// {user_message}
/// ```
pub fn format_tool_call_prompt(tools: &[ToolDefinition], user_message: &str) -> String {
    let tokens = ToolUseTokens::default();

    let tool_jsons: Vec<serde_json::Value> = tools
        .iter()
        .map(|tool| {
            // Build properties and required arrays
            let mut properties = serde_json::Map::new();
            let mut required_names: Vec<String> = Vec::new();

            for param in &tool.parameters {
                let mut prop = serde_json::Map::new();
                prop.insert("type".to_string(), serde_json::json!(param.param_type));
                prop.insert(
                    "description".to_string(),
                    serde_json::json!(param.description),
                );
                properties.insert(param.name.clone(), serde_json::Value::Object(prop));
                if param.required {
                    required_names.push(param.name.clone());
                }
            }

            serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required_names,
                    }
                }
            })
        })
        .collect();

    let tools_json = serde_json::to_string(&tool_jsons).unwrap_or_else(|_| "[]".to_string());

    format!(
        "{} {}{}\n\n{}",
        tokens.available_tools_start, tools_json, tokens.available_tools_end, user_message
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool call response parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Parse `[TOOL_CALLS]...[/TOOL_CALLS]` from a model response string.
///
/// Expects the inner content to be a JSON array of objects with the shape:
/// ```json
/// [{"name": "tool_name", "arguments": {"key": "value"}, "id": "call_id"}, ...]
/// ```
///
/// Returns `None` if no `[TOOL_CALLS]` marker is found; returns `Some(vec)` even
/// if `[/TOOL_CALLS]` is absent (graceful truncation).
pub fn parse_tool_call_response(
    response: &str,
    tokens: &ToolUseTokens,
) -> Option<Vec<ToolCallRequest>> {
    let start_pos = response.find(&tokens.tool_call_start)?;
    let after_start = &response[start_pos + tokens.tool_call_start.len()..];

    // Content up to the closing marker, or end of string
    let content = if let Some(end_pos) = after_start.find(&tokens.tool_call_end) {
        &after_start[..end_pos]
    } else {
        after_start
    }
    .trim();

    // Find the JSON array
    let json_start = content.find('[')?;
    let json_candidate = &content[json_start..];
    let json_str = balanced_json_array(json_candidate)?;

    let raw: Vec<serde_json::Value> = serde_json::from_str(json_str).ok()?;
    if raw.is_empty() {
        return None;
    }

    let mut calls = Vec::with_capacity(raw.len());
    for item in raw {
        let tool_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();

        let call_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();

        let arguments_val = item.get("arguments").cloned().unwrap_or(serde_json::json!({}));

        let arguments: HashMap<String, String> =
            if let serde_json::Value::Object(map) = arguments_val {
                map.into_iter()
                    .map(|(k, v)| {
                        let val_str = match &v {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        (k, val_str)
                    })
                    .collect()
            } else {
                HashMap::new()
            };

        if !tool_name.is_empty() {
            calls.push(ToolCallRequest {
                tool_name,
                arguments,
                call_id,
            });
        }
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MistralV3ForCausalLM re-export wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Mistral v0.3 causal language model with function-calling utilities.
///
/// Thin wrapper around `MistralV3ForCausalLM` that attaches a tool registry
/// and the v0.3 special tokens, providing `format_prompt` and
/// `parse_response` helpers.
pub struct MistralV3WithTools {
    inner: MistralV3ForCausalLM,
    tools: Vec<ToolDefinition>,
    tokens: ToolUseTokens,
}

impl MistralV3WithTools {
    /// Create a new tool-aware Mistral v0.3 instance.
    pub fn new(inner: MistralV3ForCausalLM, tools: Vec<ToolDefinition>) -> Self {
        Self {
            inner,
            tools,
            tokens: ToolUseTokens::default(),
        }
    }

    /// Access the underlying model configuration.
    pub fn config(&self) -> &MistralV3Config {
        self.inner.config()
    }

    /// Format a prompt that embeds all registered tools.
    pub fn format_prompt(&self, user_message: &str) -> String {
        format_tool_call_prompt(&self.tools, user_message)
    }

    /// Parse tool calls from a model response string.
    pub fn parse_response(&self, response: &str) -> Option<Vec<ToolCallRequest>> {
        parse_tool_call_response(response, &self.tokens)
    }

    /// Run a forward pass on the underlying model.
    pub fn forward(&self, input_ids: Vec<u32>) -> TFResult<trustformers_core::tensor::Tensor> {
        self.inner.forward(input_ids)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: extract a balanced JSON array string
// ─────────────────────────────────────────────────────────────────────────────

/// Walk the string starting at `[` and return the slice up to and including
/// the matching `]`, respecting nested brackets and string literals.
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
            match ch {
                '\\' => escape_next = true,
                '"' => in_string = false,
                _ => {},
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
