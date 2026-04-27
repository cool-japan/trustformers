//! OpenAI-compatible API layer for TrustformeRS
//!
//! Provides structs and logic compatible with the OpenAI API v1 format so that
//! existing OpenAI client libraries can talk to TrustformeRS without modification.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Chat Completions ─────────────────────────────────────────────────────────

/// The role of a participant in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
    Function,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A tool call embedded in a chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallMessage {
    pub id: String,
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// The function identifier and its serialised arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON string of arguments.
    pub arguments: String,
}

/// `POST /v1/chat/completions` request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Number of completions to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// A tool that the model may call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

/// Metadata describing a callable function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema object describing the function parameters.
    pub parameters: serde_json::Value,
}

/// Controls which tool (if any) is called by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// One of `"none"`, `"auto"`, or `"required"`.
    String(String),
    /// Explicit function choice: `{"type": "function", "function": {"name": "…"}}`.
    Object {
        r#type: String,
        function: FunctionName,
    },
}

/// Wrapper holding just a function name, used inside [`ToolChoice::Object`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionName {
    pub name: String,
}

/// `POST /v1/chat/completions` response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    /// Always `"chat.completion"`.
    pub object: String,
    /// Unix timestamp of creation.
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// One candidate completion within a [`ChatCompletionResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    /// One of `"stop"`, `"length"`, `"tool_calls"`, or `"content_filter"`.
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

// ─── Text Completions ─────────────────────────────────────────────────────────

/// `POST /v1/completions` request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: CompletionPrompt,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
}

/// Prompt for a text completion: a single string or a batch of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Single(String),
    Multiple(Vec<String>),
}

/// `POST /v1/completions` response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    /// Always `"text_completion"`.
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageStats,
}

/// One candidate in a [`CompletionResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

// ─── Embeddings ───────────────────────────────────────────────────────────────

/// `POST /v1/embeddings` request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    /// `"float"` or `"base64"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Input for an embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Pre-tokenised input as token IDs.
    Tokens(Vec<u32>),
    Single(String),
    Multiple(Vec<String>),
}

/// `POST /v1/embeddings` response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Always `"list"`.
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// One embedding vector within an [`EmbeddingResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    /// Always `"embedding"`.
    pub object: String,
    pub index: u32,
    pub embedding: Vec<f32>,
}

/// Token-usage statistics for an embedding call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// ─── Models endpoint ──────────────────────────────────────────────────────────

/// Metadata about a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    /// Always `"model"`.
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// `GET /v1/models` response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    /// Always `"list"`.
    pub object: String,
    pub data: Vec<ModelInfo>,
}

// ─── Error format ─────────────────────────────────────────────────────────────

/// Top-level OpenAI error envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiError {
    pub error: OpenAiErrorBody,
}

/// Payload inside an [`OpenAiError`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl OpenAiError {
    /// Construct an `invalid_request_error`.
    pub fn invalid_request(message: impl Into<String>, param: Option<String>) -> Self {
        Self {
            error: OpenAiErrorBody {
                message: message.into(),
                error_type: "invalid_request_error".to_string(),
                param,
                code: None,
            },
        }
    }

    /// Construct a model-not-found error.
    pub fn model_not_found(model: &str) -> Self {
        Self {
            error: OpenAiErrorBody {
                message: format!("The model `{model}` does not exist"),
                error_type: "invalid_request_error".to_string(),
                param: Some("model".to_string()),
                code: Some("model_not_found".to_string()),
            },
        }
    }

    /// Construct a rate-limit error.
    pub fn rate_limit_exceeded() -> Self {
        Self {
            error: OpenAiErrorBody {
                message: "Rate limit reached for requests".to_string(),
                error_type: "requests".to_string(),
                param: None,
                code: Some("rate_limit_exceeded".to_string()),
            },
        }
    }

    /// Construct an internal server error.
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiErrorBody {
                message: message.into(),
                error_type: "server_error".to_string(),
                param: None,
                code: Some("internal_error".to_string()),
            },
        }
    }
}

// ─── Shared ───────────────────────────────────────────────────────────────────

/// Token-usage statistics shared by chat and text completion responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by the OpenAI-compatibility layer.
#[derive(Debug, thiserror::Error)]
pub enum OpenAiCompatError {
    #[error("Empty messages")]
    EmptyMessages,
    #[error("Empty model name")]
    EmptyModel,
    #[error("Temperature out of range: {0}")]
    InvalidTemperature(f32),
    #[error("top_p out of range: {0}")]
    InvalidTopP(f32),
    #[error("Invalid max_tokens: 0")]
    InvalidMaxTokens,
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Model not allowed: {0}")]
    ModelNotAllowed(String),
}

// ─── Builder / Handler ────────────────────────────────────────────────────────

/// Builds OpenAI-compatible responses from internal TrustformeRS outputs.
pub struct OpenAiResponseBuilder;

impl OpenAiResponseBuilder {
    // ── Private helpers ──────────────────────────────────────────────────────

    /// Produce a short deterministic hash string from two inputs.
    fn short_hash(a: &str, b: u64) -> String {
        // Simple djb2-style mix; no crypto needed here.
        let mut h: u64 = 5381;
        for byte in a.bytes() {
            h = h.wrapping_mul(33).wrapping_add(u64::from(byte));
        }
        h = h.wrapping_mul(33).wrapping_add(b);
        format!("{h:016x}")
    }

    /// Derive a stable-ish creation timestamp from model and a seed value.
    fn created_at(model: &str, seed: u64) -> u64 {
        let mut h: u64 = 5381;
        for byte in model.bytes() {
            h = h.wrapping_mul(33).wrapping_add(u64::from(byte));
        }
        h = h.wrapping_mul(33).wrapping_add(seed);
        // Keep in a reasonable recent range without std::time dependency in tests.
        1_700_000_000u64.wrapping_add(h % 10_000_000)
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Create a chat completion response from generated text.
    ///
    /// - `id` is `chatcmpl-{hash of model + timestamp}`
    /// - `object` is `"chat.completion"`
    /// - `choices[0].message.role` is [`ChatRole::Assistant`]
    /// - `choices[0].message.content` is `Some(generated_text)`
    /// - `choices[0].finish_reason` is `"stop"`
    pub fn chat_completion(
        model: &str,
        _messages: &[ChatMessage],
        generated_text: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> ChatCompletionResponse {
        let seed = prompt_tokens as u64 * 1_000_000 + completion_tokens as u64;
        let created = Self::created_at(model, seed);
        let id = format!("chatcmpl-{}", Self::short_hash(model, created));

        ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model: model.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: Some(generated_text.to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".to_string(),
                logprobs: None,
            }],
            usage: UsageStats {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            system_fingerprint: None,
        }
    }

    /// Create a text completion response from generated text.
    pub fn completion(
        model: &str,
        prompt: &str,
        generated_text: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> CompletionResponse {
        let seed = prompt.len() as u64;
        let created = Self::created_at(model, seed);
        let id = format!("cmpl-{}", Self::short_hash(model, created));

        CompletionResponse {
            id,
            object: "text_completion".to_string(),
            created,
            model: model.to_string(),
            choices: vec![CompletionChoice {
                text: generated_text.to_string(),
                index: 0,
                finish_reason: "stop".to_string(),
                logprobs: None,
            }],
            usage: UsageStats {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        }
    }

    /// Create an embedding response from a list of embedding vectors.
    pub fn embedding(model: &str, inputs: &[&str], embeddings: &[Vec<f32>]) -> EmbeddingResponse {
        let prompt_tokens: u32 = inputs.iter().map(|s| Self::count_tokens(s)).sum();

        let data: Vec<EmbeddingData> = embeddings
            .iter()
            .enumerate()
            .map(|(i, vec)| EmbeddingData {
                object: "embedding".to_string(),
                index: i as u32,
                embedding: vec.clone(),
            })
            .collect();

        EmbeddingResponse {
            object: "list".to_string(),
            data,
            model: model.to_string(),
            usage: EmbeddingUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
            },
        }
    }

    /// Build a model-list response from a slice of model ID strings.
    pub fn model_list(models: &[&str]) -> ModelListResponse {
        let data: Vec<ModelInfo> = models
            .iter()
            .enumerate()
            .map(|(i, id)| ModelInfo {
                id: id.to_string(),
                object: "model".to_string(),
                created: 1_700_000_000u64 + i as u64,
                owned_by: "trustformers".to_string(),
            })
            .collect();

        ModelListResponse {
            object: "list".to_string(),
            data,
        }
    }

    /// Validate a [`ChatCompletionRequest`].
    ///
    /// Checks:
    /// - messages not empty
    /// - model not empty
    /// - temperature in `[0, 2]` if set
    /// - top_p in `[0, 1]` if set
    /// - max_tokens > 0 if set
    pub fn validate_chat_request(req: &ChatCompletionRequest) -> Result<(), OpenAiCompatError> {
        if req.messages.is_empty() {
            return Err(OpenAiCompatError::EmptyMessages);
        }
        if req.model.trim().is_empty() {
            return Err(OpenAiCompatError::EmptyModel);
        }
        if let Some(temp) = req.temperature {
            if !(0.0..=2.0).contains(&temp) {
                return Err(OpenAiCompatError::InvalidTemperature(temp));
            }
        }
        if let Some(top_p) = req.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                return Err(OpenAiCompatError::InvalidTopP(top_p));
            }
        }
        if let Some(max_tokens) = req.max_tokens {
            if max_tokens == 0 {
                return Err(OpenAiCompatError::InvalidMaxTokens);
            }
        }
        Ok(())
    }

    /// Count approximate tokens in text using the word-based approximation `len / 4`.
    pub fn count_tokens(text: &str) -> u32 {
        (text.len() / 4) as u32
    }

    /// Extract the content of the first `system` message, if any.
    pub fn extract_system(messages: &[ChatMessage]) -> Option<String> {
        messages.iter().find_map(|msg| {
            if matches!(msg.role, ChatRole::System) {
                msg.content.clone()
            } else {
                None
            }
        })
    }

    /// Convert a slice of chat messages into a plain-text prompt.
    ///
    /// Format: `"{role}: {content}\n"` for each message.
    pub fn messages_to_prompt(messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            let role_str = match msg.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => "tool",
                ChatRole::Function => "function",
            };
            let content = msg.content.as_deref().unwrap_or("");
            out.push_str(&format!("{role_str}: {content}\n"));
        }
        out
    }
}

// Keep HashMap import used by potential downstream consumers; suppress dead-code
// lint for the field-less helper without making the public API less clear.
const _: Option<HashMap<String, String>> = None;

// ─── FinishReason ─────────────────────────────────────────────────────────────

/// Reason why the model stopped generating tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    FunctionCall,
}

// ─── TokenUsage alias ────────────────────────────────────────────────────────

/// Alias for [`UsageStats`] matching the OpenAI `usage` field name `TokenUsage`.
pub type TokenUsage = UsageStats;

// ─── SSE Streaming ────────────────────────────────────────────────────────────

/// The incremental content delta in a streaming chunk choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<ChatRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// One candidate delta within a [`StreamChunk`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: MessageDelta,
    pub finish_reason: Option<FinishReason>,
}

/// Server-Sent Events chunk for streaming chat completions.
///
/// Object type is always `"chat.completion.chunk"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

impl StreamChunk {
    /// Format this chunk as an SSE data line: `data: {json}\n\n`.
    pub fn to_sse_line(&self) -> Result<String, OpenAiCompatError> {
        let json = serde_json::to_string(self)
            .map_err(|e| OpenAiCompatError::SerializationError(e.to_string()))?;
        Ok(format!("data: {json}\n\n"))
    }

    /// Create the initial chunk that carries the role.
    pub fn first_chunk(id: &str, model: &str, created: u64) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            choices: vec![StreamChoice {
                index: 0,
                delta: MessageDelta {
                    role: Some(ChatRole::Assistant),
                    content: None,
                },
                finish_reason: None,
            }],
        }
    }

    /// Create a content delta chunk.
    pub fn content_chunk(id: &str, model: &str, created: u64, content: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            choices: vec![StreamChoice {
                index: 0,
                delta: MessageDelta {
                    role: None,
                    content: Some(content.to_string()),
                },
                finish_reason: None,
            }],
        }
    }

    /// Create the final stop chunk.
    pub fn stop_chunk(id: &str, model: &str, created: u64, reason: FinishReason) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            choices: vec![StreamChoice {
                index: 0,
                delta: MessageDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(reason),
            }],
        }
    }
}

// ─── OpenAiApiRouter ─────────────────────────────────────────────────────────

/// Routes incoming OpenAI-compatible requests to the appropriate handler.
pub struct OpenAiApiRouter {
    allowed_models: Vec<String>,
}

impl OpenAiApiRouter {
    /// Construct a new router with a list of allowed model identifiers.
    ///
    /// An empty `allowed_models` list means all models are permitted.
    pub fn new(allowed_models: Vec<String>) -> Self {
        Self { allowed_models }
    }

    /// Check whether a model name is allowed by this router.
    fn is_model_allowed(&self, model: &str) -> bool {
        self.allowed_models.is_empty() || self.allowed_models.iter().any(|m| m == model)
    }

    /// Handle a `POST /v1/chat/completions` request.
    pub fn route_chat_completion(
        &self,
        req: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, OpenAiCompatError> {
        OpenAiResponseBuilder::validate_chat_request(req)?;
        if !self.is_model_allowed(&req.model) {
            return Err(OpenAiCompatError::ModelNotAllowed(req.model.clone()));
        }
        let prompt_tokens: u32 = req
            .messages
            .iter()
            .filter_map(|m| m.content.as_deref())
            .map(OpenAiResponseBuilder::count_tokens)
            .sum();
        let generated_text = "[stub chat response]";
        let completion_tokens = OpenAiResponseBuilder::count_tokens(generated_text);
        Ok(OpenAiResponseBuilder::chat_completion(
            &req.model,
            &req.messages,
            generated_text,
            prompt_tokens,
            completion_tokens,
        ))
    }

    /// Handle a `POST /v1/completions` request.
    pub fn route_completion(
        &self,
        req: &CompletionRequest,
    ) -> Result<CompletionResponse, OpenAiCompatError> {
        if req.model.trim().is_empty() {
            return Err(OpenAiCompatError::EmptyModel);
        }
        if !self.is_model_allowed(&req.model) {
            return Err(OpenAiCompatError::ModelNotAllowed(req.model.clone()));
        }
        let prompt_text = match &req.prompt {
            CompletionPrompt::Single(s) => s.clone(),
            CompletionPrompt::Multiple(v) => v.join(" "),
        };
        let prompt_tokens = OpenAiResponseBuilder::count_tokens(&prompt_text);
        let generated_text = "[stub completion]";
        let completion_tokens = OpenAiResponseBuilder::count_tokens(generated_text);
        Ok(OpenAiResponseBuilder::completion(
            &req.model,
            &prompt_text,
            generated_text,
            prompt_tokens,
            completion_tokens,
        ))
    }

    /// Handle a `POST /v1/embeddings` request.
    pub fn route_embeddings(
        &self,
        req: &EmbeddingRequest,
    ) -> Result<EmbeddingResponse, OpenAiCompatError> {
        if req.model.trim().is_empty() {
            return Err(OpenAiCompatError::EmptyModel);
        }
        if !self.is_model_allowed(&req.model) {
            return Err(OpenAiCompatError::ModelNotAllowed(req.model.clone()));
        }
        let texts: Vec<String> = match &req.input {
            EmbeddingInput::Single(s) => vec![s.clone()],
            EmbeddingInput::Multiple(v) => v.clone(),
            EmbeddingInput::Tokens(ids) => {
                vec![ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" ")]
            },
        };
        let inputs_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        // Stub embeddings: each input gets a 4-dim zero vector.
        let embeddings: Vec<Vec<f32>> = inputs_ref.iter().map(|_| vec![0.0f32; 4]).collect();
        Ok(OpenAiResponseBuilder::embedding(
            &req.model,
            &inputs_ref,
            &embeddings,
        ))
    }

    /// Convert an [`OpenAiCompatError`] into an [`OpenAiError`] response body.
    pub fn format_error_response(err: &OpenAiCompatError) -> OpenAiError {
        match err {
            OpenAiCompatError::EmptyMessages => OpenAiError::invalid_request(
                "messages array must not be empty",
                Some("messages".to_string()),
            ),
            OpenAiCompatError::EmptyModel => {
                OpenAiError::invalid_request("model must not be empty", Some("model".to_string()))
            },
            OpenAiCompatError::InvalidTemperature(t) => OpenAiError::invalid_request(
                format!("temperature {t} is out of range [0, 2]"),
                Some("temperature".to_string()),
            ),
            OpenAiCompatError::InvalidTopP(p) => OpenAiError::invalid_request(
                format!("top_p {p} is out of range [0, 1]"),
                Some("top_p".to_string()),
            ),
            OpenAiCompatError::InvalidMaxTokens => OpenAiError::invalid_request(
                "max_tokens must be > 0",
                Some("max_tokens".to_string()),
            ),
            OpenAiCompatError::SerializationError(msg) => {
                OpenAiError::internal_error(format!("serialization error: {msg}"))
            },
            OpenAiCompatError::ModelNotAllowed(model) => OpenAiError::model_not_found(model),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a minimal valid ChatCompletionRequest.
    fn minimal_chat_req() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: Some("Hello".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            top_p: None,
            n: None,
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            user: None,
        }
    }

    // ── 1. ChatMessage serialization roundtrip ────────────────────────────────
    #[test]
    fn test_chat_message_serde_roundtrip() {
        let msg = ChatMessage {
            role: ChatRole::User,
            content: Some("Hello, world!".to_string()),
            name: Some("Alice".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        let back: ChatMessage = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(back.role, ChatRole::User));
        assert_eq!(back.content.as_deref(), Some("Hello, world!"));
        assert_eq!(back.name.as_deref(), Some("Alice"));
    }

    // ── 2. ChatCompletionRequest deserialize from JSON ────────────────────────
    #[test]
    fn test_chat_completion_request_deserialize() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "max_tokens": 256
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.model, "gpt-4");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(256));
    }

    // ── 3. CompletionRequest serialize / deserialize ──────────────────────────
    #[test]
    fn test_completion_request_serde() {
        let req = CompletionRequest {
            model: "text-davinci-003".to_string(),
            prompt: CompletionPrompt::Single("Once upon a time".to_string()),
            max_tokens: Some(100),
            temperature: Some(1.0),
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            echo: Some(false),
        };
        let json = serde_json::to_string(&req).expect("serialize");
        let back: CompletionRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.model, "text-davinci-003");
        assert_eq!(back.max_tokens, Some(100));
        assert!(matches!(back.prompt, CompletionPrompt::Single(_)));
    }

    // ── 4. EmbeddingRequest single input ─────────────────────────────────────
    #[test]
    fn test_embedding_request_single_input() {
        let json = r#"{"model": "text-embedding-ada-002", "input": "Hello"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.model, "text-embedding-ada-002");
        assert!(matches!(req.input, EmbeddingInput::Single(_)));
    }

    // ── 5. EmbeddingRequest multiple input ───────────────────────────────────
    #[test]
    fn test_embedding_request_multiple_input() {
        let json = r#"{"model": "text-embedding-ada-002", "input": ["Hello", "World"]}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).expect("deserialize");
        assert!(matches!(req.input, EmbeddingInput::Multiple(_)));
        if let EmbeddingInput::Multiple(v) = &req.input {
            assert_eq!(v.len(), 2);
        }
    }

    // ── 6. chat_completion response structure (id starts with "chatcmpl-") ────
    #[test]
    fn test_chat_completion_response_id_prefix() {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: Some("Hi".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let resp = OpenAiResponseBuilder::chat_completion("gpt-4", &messages, "Hello!", 10, 5);
        assert!(resp.id.starts_with("chatcmpl-"), "id = {}", resp.id);
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.choices.len(), 1);
        assert!(matches!(resp.choices[0].message.role, ChatRole::Assistant));
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.choices[0].finish_reason, "stop");
    }

    // ── 7. completion response (object = "text_completion") ──────────────────
    #[test]
    fn test_completion_response_object_field() {
        let resp = OpenAiResponseBuilder::completion("davinci", "Say hi", "Hi!", 5, 3);
        assert_eq!(resp.object, "text_completion");
        assert_eq!(resp.choices[0].text, "Hi!");
        assert_eq!(resp.choices[0].finish_reason, "stop");
    }

    // ── 8. embedding response (object = "list") ───────────────────────────────
    #[test]
    fn test_embedding_response_object_field() {
        let inputs = ["Hello", "World"];
        let embeddings = vec![vec![0.1f32, 0.2], vec![0.3f32, 0.4]];
        let resp = OpenAiResponseBuilder::embedding("ada", &inputs, &embeddings);
        assert_eq!(resp.object, "list");
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].object, "embedding");
        assert_eq!(resp.data[0].embedding, vec![0.1f32, 0.2]);
    }

    // ── 9. model_list response ────────────────────────────────────────────────
    #[test]
    fn test_model_list_response() {
        let resp = OpenAiResponseBuilder::model_list(&["gpt-4", "gpt-3.5-turbo"]);
        assert_eq!(resp.object, "list");
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].id, "gpt-4");
        assert_eq!(resp.data[0].object, "model");
        assert_eq!(resp.data[1].id, "gpt-3.5-turbo");
    }

    // ── 10. validate_chat_request valid ───────────────────────────────────────
    #[test]
    fn test_validate_chat_request_valid() {
        let mut req = minimal_chat_req();
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        req.max_tokens = Some(128);
        assert!(OpenAiResponseBuilder::validate_chat_request(&req).is_ok());
    }

    // ── 11. validate empty messages ───────────────────────────────────────────
    #[test]
    fn test_validate_empty_messages() {
        let mut req = minimal_chat_req();
        req.messages = vec![];
        let err = OpenAiResponseBuilder::validate_chat_request(&req).unwrap_err();
        assert!(matches!(err, OpenAiCompatError::EmptyMessages));
    }

    // ── 12. validate empty model ──────────────────────────────────────────────
    #[test]
    fn test_validate_empty_model() {
        let mut req = minimal_chat_req();
        req.model = "  ".to_string();
        let err = OpenAiResponseBuilder::validate_chat_request(&req).unwrap_err();
        assert!(matches!(err, OpenAiCompatError::EmptyModel));
    }

    // ── 13. validate invalid temperature ─────────────────────────────────────
    #[test]
    fn test_validate_invalid_temperature() {
        let mut req = minimal_chat_req();
        req.temperature = Some(2.5);
        let err = OpenAiResponseBuilder::validate_chat_request(&req).unwrap_err();
        assert!(matches!(err, OpenAiCompatError::InvalidTemperature(_)));
    }

    // ── 14. validate invalid top_p ────────────────────────────────────────────
    #[test]
    fn test_validate_invalid_top_p() {
        let mut req = minimal_chat_req();
        req.top_p = Some(-0.1);
        let err = OpenAiResponseBuilder::validate_chat_request(&req).unwrap_err();
        assert!(matches!(err, OpenAiCompatError::InvalidTopP(_)));
    }

    // ── 15. count_tokens approximation ────────────────────────────────────────
    #[test]
    fn test_count_tokens_approximation() {
        // "Hello" has 5 chars → 5/4 = 1
        assert_eq!(OpenAiResponseBuilder::count_tokens("Hello"), 1);
        // 40-char string → 10 tokens
        let text = "a".repeat(40);
        assert_eq!(OpenAiResponseBuilder::count_tokens(&text), 10);
    }

    // ── 16. extract_system (finds system role) ────────────────────────────────
    #[test]
    fn test_extract_system_message() {
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: Some("You are a helpful assistant.".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: ChatRole::User,
                content: Some("Hello".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];
        let system = OpenAiResponseBuilder::extract_system(&messages);
        assert_eq!(system.as_deref(), Some("You are a helpful assistant."));
    }

    // ── 17. messages_to_prompt format ─────────────────────────────────────────
    #[test]
    fn test_messages_to_prompt_format() {
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: Some("Be helpful.".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: ChatRole::User,
                content: Some("Hi".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];
        let prompt = OpenAiResponseBuilder::messages_to_prompt(&messages);
        assert_eq!(prompt, "system: Be helpful.\nuser: Hi\n");
    }

    // ── 18. OpenAiError constructors ──────────────────────────────────────────
    #[test]
    fn test_open_ai_error_constructors() {
        let e = OpenAiError::invalid_request("bad param", Some("temperature".to_string()));
        assert_eq!(e.error.error_type, "invalid_request_error");
        assert_eq!(e.error.param.as_deref(), Some("temperature"));

        let e = OpenAiError::model_not_found("gpt-99");
        assert!(e.error.message.contains("gpt-99"));
        assert_eq!(e.error.code.as_deref(), Some("model_not_found"));

        let e = OpenAiError::rate_limit_exceeded();
        assert_eq!(e.error.code.as_deref(), Some("rate_limit_exceeded"));

        let e = OpenAiError::internal_error("oops");
        assert_eq!(e.error.error_type, "server_error");
        assert_eq!(e.error.message, "oops");
    }

    // ── 19. OpenAiError JSON serialization ───────────────────────────────────
    #[test]
    fn test_open_ai_error_json_serialization() {
        let e = OpenAiError::model_not_found("unknown-model");
        let json = serde_json::to_string(&e).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert!(v["error"]["message"].as_str().is_some());
        assert_eq!(v["error"]["type"].as_str(), Some("invalid_request_error"));
    }

    // ── 20. UsageStats total = prompt + completion ───────────────────────────
    #[test]
    fn test_usage_stats_total() {
        let resp = OpenAiResponseBuilder::chat_completion("gpt-4", &[], "Hi", 100, 50);
        let usage = &resp.usage;
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
    }

    // ── 21. ToolChoice String variant deserialization ─────────────────────────
    #[test]
    fn test_tool_choice_string_variant() {
        let json =
            r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}],"tool_choice":"auto"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        match req.tool_choice {
            Some(ToolChoice::String(s)) => assert_eq!(s, "auto"),
            other => panic!("expected ToolChoice::String, got {other:?}"),
        }
    }

    // ── 22. FinishReason serde roundtrip ──────────────────────────────────────
    #[test]
    fn test_finish_reason_serde_roundtrip() {
        let variants = [
            (FinishReason::Stop, "\"stop\""),
            (FinishReason::Length, "\"length\""),
            (FinishReason::ContentFilter, "\"content_filter\""),
            (FinishReason::ToolCalls, "\"tool_calls\""),
            (FinishReason::FunctionCall, "\"function_call\""),
        ];
        for (variant, expected_json) in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            assert_eq!(
                json, expected_json,
                "serialization mismatch for {variant:?}"
            );
            let back: FinishReason = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, variant);
        }
    }

    // ── 23. StreamChunk::to_sse_line produces data: ... \n\n ─────────────────
    #[test]
    fn test_stream_chunk_to_sse_line() {
        let chunk = StreamChunk {
            id: "chatcmpl-abc".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1_700_000_000,
            model: "gpt-4".to_string(),
            choices: vec![StreamChoice {
                index: 0,
                delta: MessageDelta {
                    role: None,
                    content: Some("Hello".to_string()),
                },
                finish_reason: None,
            }],
        };
        let line = chunk.to_sse_line().expect("to_sse_line");
        assert!(line.starts_with("data: "), "line = {line}");
        assert!(
            line.ends_with("\n\n"),
            "line does not end with \\n\\n: {line:?}"
        );
    }

    // ── 24. SSE line ends with exactly \n\n ───────────────────────────────────
    #[test]
    fn test_sse_line_ends_with_double_newline() {
        let chunk = StreamChunk::content_chunk("id1", "model1", 0, "token");
        let line = chunk.to_sse_line().expect("to_sse_line");
        let bytes = line.as_bytes();
        let last_two: &[u8] = &bytes[bytes.len().saturating_sub(2)..];
        assert_eq!(last_two, b"\n\n");
    }

    // ── 25. MessageDelta with None fields skips them in JSON ─────────────────
    #[test]
    fn test_message_delta_skips_none_fields() {
        let delta = MessageDelta {
            role: None,
            content: Some("hello".to_string()),
        };
        let json = serde_json::to_string(&delta).expect("serialize");
        assert!(!json.contains("role"), "role should be absent: {json}");
        assert!(
            json.contains("content"),
            "content should be present: {json}"
        );
    }

    // ── 26. StreamChoice serde roundtrip ──────────────────────────────────────
    #[test]
    fn test_stream_choice_serde_roundtrip() {
        let choice = StreamChoice {
            index: 2,
            delta: MessageDelta {
                role: Some(ChatRole::Assistant),
                content: None,
            },
            finish_reason: Some(FinishReason::Stop),
        };
        let json = serde_json::to_string(&choice).expect("serialize");
        let back: StreamChoice = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.index, 2);
        assert!(back.finish_reason.as_ref().is_some_and(|r| *r == FinishReason::Stop));
    }

    // ── 27. OpenAiApiRouter::new stores allowed models ────────────────────────
    #[test]
    fn test_router_new_stores_models() {
        let router = OpenAiApiRouter::new(vec!["gpt-4".to_string(), "claude".to_string()]);
        assert!(router.is_model_allowed("gpt-4"));
        assert!(router.is_model_allowed("claude"));
        assert!(!router.is_model_allowed("unknown"));
    }

    // ── 28. OpenAiApiRouter route_chat_completion happy path ─────────────────
    #[test]
    fn test_router_route_chat_completion_happy() {
        let router = OpenAiApiRouter::new(vec!["gpt-4".to_string()]);
        let req = minimal_chat_req();
        let resp = router.route_chat_completion(&req).expect("route_chat_completion");
        assert!(resp.id.starts_with("chatcmpl-"));
        assert_eq!(resp.model, "gpt-4");
        assert_eq!(resp.choices.len(), 1);
    }

    // ── 29. OpenAiApiRouter route_chat_completion disallowed model ────────────
    #[test]
    fn test_router_route_chat_completion_disallowed_model() {
        let router = OpenAiApiRouter::new(vec!["gpt-3".to_string()]);
        let req = minimal_chat_req(); // uses "gpt-4"
        let err = router.route_chat_completion(&req).unwrap_err();
        assert!(matches!(err, OpenAiCompatError::ModelNotAllowed(_)));
    }

    // ── 30. OpenAiApiRouter route_completion happy path ───────────────────────
    #[test]
    fn test_router_route_completion_happy() {
        let router = OpenAiApiRouter::new(vec!["davinci".to_string()]);
        let req = CompletionRequest {
            model: "davinci".to_string(),
            prompt: CompletionPrompt::Single("Hello".to_string()),
            max_tokens: Some(50),
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            echo: None,
        };
        let resp = router.route_completion(&req).expect("route_completion");
        assert_eq!(resp.model, "davinci");
        assert_eq!(resp.object, "text_completion");
    }

    // ── 31. OpenAiApiRouter route_embeddings happy path ───────────────────────
    #[test]
    fn test_router_route_embeddings_happy() {
        let router = OpenAiApiRouter::new(vec!["ada".to_string()]);
        let req = EmbeddingRequest {
            model: "ada".to_string(),
            input: EmbeddingInput::Single("Hello world".to_string()),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        let resp = router.route_embeddings(&req).expect("route_embeddings");
        assert_eq!(resp.object, "list");
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].embedding.len(), 4);
    }

    // ── 32. format_error_response for each variant ────────────────────────────
    #[test]
    fn test_format_error_response_variants() {
        let cases: Vec<OpenAiCompatError> = vec![
            OpenAiCompatError::EmptyMessages,
            OpenAiCompatError::EmptyModel,
            OpenAiCompatError::InvalidTemperature(2.5),
            OpenAiCompatError::InvalidTopP(-0.1),
            OpenAiCompatError::InvalidMaxTokens,
            OpenAiCompatError::SerializationError("json fail".to_string()),
            OpenAiCompatError::ModelNotAllowed("unknown".to_string()),
        ];
        for err in &cases {
            let oai_err = OpenAiApiRouter::format_error_response(err);
            // Every converted error should have a non-empty message.
            assert!(!oai_err.error.message.is_empty(), "empty message for {err}");
        }
    }

    // ── 33. TokenUsage is same type as UsageStats ─────────────────────────────
    #[test]
    fn test_token_usage_alias_is_usage_stats() {
        // If TokenUsage == UsageStats, we can use one where the other is expected.
        let usage: TokenUsage = UsageStats {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        };
        assert_eq!(usage.total_tokens, 15u32);
    }

    // ── 34. StreamChunk first/content/stop factory methods ────────────────────
    #[test]
    fn test_stream_chunk_factory_methods() {
        let id = "chatcmpl-xyz";
        let model = "gpt-4";
        let ts = 1_700_000_000u64;

        let first = StreamChunk::first_chunk(id, model, ts);
        assert_eq!(first.object, "chat.completion.chunk");
        assert!(matches!(
            first.choices[0].delta.role,
            Some(ChatRole::Assistant)
        ));

        let content = StreamChunk::content_chunk(id, model, ts, "token");
        assert_eq!(content.choices[0].delta.content.as_deref(), Some("token"));
        assert!(content.choices[0].delta.role.is_none());

        let stop = StreamChunk::stop_chunk(id, model, ts, FinishReason::Stop);
        assert_eq!(stop.choices[0].finish_reason, Some(FinishReason::Stop));
    }
}

mod openai_compat_tests;
