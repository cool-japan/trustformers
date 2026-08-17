//! Request routing for the OpenAI-compatible API.
//!
//! [`OpenAiApiRouter`] validates requests, enforces the model allow-list and
//! forwards the work to an [`OpenAiInferenceBackend`]. It owns no model of its
//! own: with no backend attached every generation and embedding request fails
//! with [`OpenAiCompatError::BackendUnavailable`] rather than returning
//! placeholder text or a zero vector.

use std::sync::Arc;

use super::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionPrompt,
    CompletionRequest, CompletionResponse, EmbeddingInput, EmbeddingRequest, EmbeddingResponse,
    ModelListResponse, OpenAiCompatError, OpenAiError, OpenAiResponseBuilder,
};

// ─── Inference backend ───────────────────────────────────────────────────────

/// A generation request handed to an [`OpenAiInferenceBackend`].
#[derive(Debug, Clone)]
pub struct BackendGenerationRequest {
    /// Model the client asked for.
    pub model: String,
    /// Flattened prompt text.
    pub prompt: String,
    /// Original chat messages, when the request came from `/v1/chat/completions`.
    pub messages: Vec<ChatMessage>,
    /// Maximum number of new tokens to generate, if the client set one.
    pub max_tokens: Option<u32>,
    /// Sampling temperature, if the client set one.
    pub temperature: Option<f32>,
    /// Nucleus sampling parameter, if the client set one.
    pub top_p: Option<f32>,
    /// Stop sequences, if the client set any.
    pub stop: Vec<String>,
}

/// What a backend produced for one generation request.
#[derive(Debug, Clone)]
pub struct BackendGenerationOutput {
    /// Generated text. Must be model output, never a placeholder.
    pub text: String,
    /// Real prompt token count, when the backend has a tokenizer.
    pub prompt_tokens: Option<u32>,
    /// Real completion token count, when the backend has a tokenizer.
    pub completion_tokens: Option<u32>,
    /// Why generation stopped: `"stop"` or `"length"`.
    pub finish_reason: String,
}

/// The inference path behind the OpenAI-compatible endpoints.
///
/// [`OpenAiApiRouter`] owns no model and cannot generate text on its own. A
/// router without a backend answers every generation request with
/// [`OpenAiCompatError::BackendUnavailable`] — it never substitutes placeholder
/// text or a zero embedding vector.
#[async_trait::async_trait]
pub trait OpenAiInferenceBackend: Send + Sync + std::fmt::Debug {
    /// Generate a completion.
    async fn generate(
        &self,
        request: &BackendGenerationRequest,
    ) -> Result<BackendGenerationOutput, OpenAiCompatError>;

    /// Embed `inputs` with `model`.
    ///
    /// Implementations that have no embedding model must return
    /// [`OpenAiCompatError::EmbeddingsUnavailable`]. Returning a zero vector is
    /// never acceptable: a caller cannot distinguish it from a real embedding.
    async fn embed(
        &self,
        model: &str,
        inputs: &[String],
    ) -> Result<Vec<Vec<f32>>, OpenAiCompatError>;

    /// Count tokens with the model's real tokenizer.
    ///
    /// Returning `None` means "no tokenizer available for this model", and the
    /// router then falls back to the documented `len / 4` approximation.
    fn count_tokens(&self, _model: &str, _text: &str) -> Option<u32> {
        None
    }

    /// Model identifiers this backend can serve.
    fn models(&self) -> Vec<String> {
        Vec::new()
    }
}

// ─── OpenAiApiRouter ─────────────────────────────────────────────────────────

/// Routes incoming OpenAI-compatible requests to the attached inference backend.
#[derive(Debug, Clone)]
pub struct OpenAiApiRouter {
    allowed_models: Vec<String>,
    backend: Option<Arc<dyn OpenAiInferenceBackend>>,
}

impl OpenAiApiRouter {
    /// Construct a router with a list of allowed model identifiers and **no**
    /// inference backend.
    ///
    /// An empty `allowed_models` list means all models are permitted. Until a
    /// backend is attached with [`Self::with_backend`], every generation and
    /// embedding request fails with [`OpenAiCompatError::BackendUnavailable`].
    pub fn new(allowed_models: Vec<String>) -> Self {
        Self {
            allowed_models,
            backend: None,
        }
    }

    /// Attach the inference backend that serves generation and embeddings.
    #[must_use]
    pub fn with_backend(mut self, backend: Arc<dyn OpenAiInferenceBackend>) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Whether an inference backend is attached.
    pub fn has_backend(&self) -> bool {
        self.backend.is_some()
    }

    /// Check whether a model name is allowed by this router.
    fn is_model_allowed(&self, model: &str) -> bool {
        self.allowed_models.is_empty() || self.allowed_models.iter().any(|m| m == model)
    }

    fn backend(&self) -> Result<&Arc<dyn OpenAiInferenceBackend>, OpenAiCompatError> {
        self.backend.as_ref().ok_or(OpenAiCompatError::BackendUnavailable)
    }

    /// Count tokens with the backend's tokenizer, falling back to the
    /// documented approximation when the backend has none.
    fn count_tokens(&self, model: &str, text: &str) -> u32 {
        self.backend
            .as_ref()
            .and_then(|backend| backend.count_tokens(model, text))
            .unwrap_or_else(|| OpenAiResponseBuilder::count_tokens(text))
    }

    /// Handle a `POST /v1/chat/completions` request.
    ///
    /// # Errors
    ///
    /// Returns a validation error for a malformed request,
    /// [`OpenAiCompatError::ModelNotAllowed`] for a disallowed model, or
    /// [`OpenAiCompatError::BackendUnavailable`] when no backend is attached.
    pub async fn route_chat_completion(
        &self,
        req: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, OpenAiCompatError> {
        OpenAiResponseBuilder::validate_chat_request(req)?;
        if !self.is_model_allowed(&req.model) {
            return Err(OpenAiCompatError::ModelNotAllowed(req.model.clone()));
        }
        let backend = self.backend()?;

        let prompt = OpenAiResponseBuilder::messages_to_prompt(&req.messages);
        let request = BackendGenerationRequest {
            model: req.model.clone(),
            prompt: prompt.clone(),
            messages: req.messages.clone(),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            stop: req.stop.clone().unwrap_or_default(),
        };
        let output = backend.generate(&request).await?;

        let prompt_tokens =
            output.prompt_tokens.unwrap_or_else(|| self.count_tokens(&req.model, &prompt));
        let completion_tokens = output
            .completion_tokens
            .unwrap_or_else(|| self.count_tokens(&req.model, &output.text));

        let mut response = OpenAiResponseBuilder::chat_completion(
            &req.model,
            &req.messages,
            &output.text,
            prompt_tokens,
            completion_tokens,
        );
        response.choices[0].finish_reason = output.finish_reason;
        Ok(response)
    }

    /// Handle a `POST /v1/completions` request.
    ///
    /// # Errors
    ///
    /// As for [`Self::route_chat_completion`].
    pub async fn route_completion(
        &self,
        req: &CompletionRequest,
    ) -> Result<CompletionResponse, OpenAiCompatError> {
        if req.model.trim().is_empty() {
            return Err(OpenAiCompatError::EmptyModel);
        }
        if !self.is_model_allowed(&req.model) {
            return Err(OpenAiCompatError::ModelNotAllowed(req.model.clone()));
        }
        let backend = self.backend()?;

        let prompt_text = match &req.prompt {
            CompletionPrompt::Single(s) => s.clone(),
            CompletionPrompt::Multiple(v) => v.join(" "),
        };
        let request = BackendGenerationRequest {
            model: req.model.clone(),
            prompt: prompt_text.clone(),
            messages: Vec::new(),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            stop: req.stop.clone().unwrap_or_default(),
        };
        let output = backend.generate(&request).await?;

        let prompt_tokens = output
            .prompt_tokens
            .unwrap_or_else(|| self.count_tokens(&req.model, &prompt_text));
        let completion_tokens = output
            .completion_tokens
            .unwrap_or_else(|| self.count_tokens(&req.model, &output.text));

        let mut response = OpenAiResponseBuilder::completion(
            &req.model,
            &prompt_text,
            &output.text,
            prompt_tokens,
            completion_tokens,
        );
        response.choices[0].finish_reason = output.finish_reason;
        Ok(response)
    }

    /// Handle a `POST /v1/embeddings` request.
    ///
    /// # Errors
    ///
    /// Returns [`OpenAiCompatError::EmbeddingsUnavailable`] when the backend has
    /// no embedding model. A zero vector is never returned.
    pub async fn route_embeddings(
        &self,
        req: &EmbeddingRequest,
    ) -> Result<EmbeddingResponse, OpenAiCompatError> {
        if req.model.trim().is_empty() {
            return Err(OpenAiCompatError::EmptyModel);
        }
        if !self.is_model_allowed(&req.model) {
            return Err(OpenAiCompatError::ModelNotAllowed(req.model.clone()));
        }
        let backend = self.backend()?;

        let texts: Vec<String> = match &req.input {
            EmbeddingInput::Single(s) => vec![s.clone()],
            EmbeddingInput::Multiple(v) => v.clone(),
            EmbeddingInput::Tokens(ids) => {
                vec![ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" ")]
            },
        };

        let embeddings = backend.embed(&req.model, &texts).await?;
        if embeddings.len() != texts.len() {
            return Err(OpenAiCompatError::BackendError(format!(
                "backend returned {} embeddings for {} inputs",
                embeddings.len(),
                texts.len()
            )));
        }
        if let Some(index) = embeddings.iter().position(|vector| vector.is_empty()) {
            return Err(OpenAiCompatError::BackendError(format!(
                "backend returned an empty embedding vector for input {index}"
            )));
        }

        let inputs_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        Ok(OpenAiResponseBuilder::embedding(
            &req.model,
            &inputs_ref,
            &embeddings,
        ))
    }

    /// Handle a `GET /v1/models` request.
    ///
    /// The list is the intersection of the router's allow-list and what the
    /// backend reports it can serve; with no backend the allow-list is returned
    /// as configured, and with neither the list is empty rather than invented.
    pub fn route_models(&self) -> ModelListResponse {
        let ids: Vec<String> = match &self.backend {
            Some(backend) => {
                let available = backend.models();
                if self.allowed_models.is_empty() {
                    available
                } else {
                    available.into_iter().filter(|model| self.is_model_allowed(model)).collect()
                }
            },
            None => self.allowed_models.clone(),
        };
        let refs: Vec<&str> = ids.iter().map(|id| id.as_str()).collect();
        OpenAiResponseBuilder::model_list(&refs)
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
            OpenAiCompatError::BackendUnavailable => {
                OpenAiError::service_unavailable(err.to_string())
            },
            OpenAiCompatError::EmbeddingsUnavailable(_) => {
                OpenAiError::service_unavailable(err.to_string())
            },
            OpenAiCompatError::BackendError(msg) => {
                OpenAiError::internal_error(format!("inference failed: {msg}"))
            },
        }
    }
}

// ─── Axum wiring ─────────────────────────────────────────────────────────────

/// Build the axum router exposing the OpenAI-compatible endpoints.
///
/// Mount it under the HTTP server with `router.merge(openai_compat_router(..))`.
/// Registered routes:
///
/// * `POST /v1/chat/completions`
/// * `POST /v1/completions`
/// * `POST /v1/embeddings`
/// * `GET  /v1/models`
///
/// With no backend attached to `router`, the three generation endpoints answer
/// `503 Service Unavailable` with an OpenAI-shaped error body.
pub fn openai_compat_router(router: OpenAiApiRouter) -> axum::Router {
    use axum::routing::{get, post};
    axum::Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/completions", post(completions_handler))
        .route("/v1/embeddings", post(embeddings_handler))
        .route("/v1/models", get(models_handler))
        .with_state(Arc::new(router))
}

fn error_response(err: &OpenAiCompatError) -> axum::response::Response {
    use axum::response::IntoResponse;
    let status = axum::http::StatusCode::from_u16(err.status_code())
        .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    (
        status,
        axum::Json(OpenAiApiRouter::format_error_response(err)),
    )
        .into_response()
}

async fn chat_completions_handler(
    axum::extract::State(router): axum::extract::State<Arc<OpenAiApiRouter>>,
    axum::Json(request): axum::Json<ChatCompletionRequest>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match router.route_chat_completion(&request).await {
        Ok(response) => axum::Json(response).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn completions_handler(
    axum::extract::State(router): axum::extract::State<Arc<OpenAiApiRouter>>,
    axum::Json(request): axum::Json<CompletionRequest>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match router.route_completion(&request).await {
        Ok(response) => axum::Json(response).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn embeddings_handler(
    axum::extract::State(router): axum::extract::State<Arc<OpenAiApiRouter>>,
    axum::Json(request): axum::Json<EmbeddingRequest>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match router.route_embeddings(&request).await {
        Ok(response) => axum::Json(response).into_response(),
        Err(e) => error_response(&e),
    }
}

async fn models_handler(
    axum::extract::State(router): axum::extract::State<Arc<OpenAiApiRouter>>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    axum::Json(router.route_models()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_compat::{ChatRole, UsageStats};

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

    #[test]
    fn test_usage_stats_are_reexported() {
        let usage = UsageStats {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
        };
        assert_eq!(usage.total_tokens, 3);
    }

    /// A deterministic backend double used to exercise the router. Declared
    /// under `#[cfg(test)]`: the production router ships with no backend and
    /// refuses requests rather than answering with placeholder text.
    #[derive(Debug)]
    struct RecordingBackend {
        reply: String,
        embedding_dim: usize,
        supports_embeddings: bool,
    }

    impl RecordingBackend {
        fn new(reply: &str) -> Self {
            Self {
                reply: reply.to_string(),
                embedding_dim: 8,
                supports_embeddings: true,
            }
        }

        fn without_embeddings(mut self) -> Self {
            self.supports_embeddings = false;
            self
        }
    }

    #[async_trait::async_trait]
    impl OpenAiInferenceBackend for RecordingBackend {
        async fn generate(
            &self,
            request: &BackendGenerationRequest,
        ) -> Result<BackendGenerationOutput, OpenAiCompatError> {
            Ok(BackendGenerationOutput {
                text: format!("{}|{}", self.reply, request.prompt.trim_end()),
                prompt_tokens: Some(request.prompt.split_whitespace().count() as u32),
                completion_tokens: Some(7),
                finish_reason: "stop".to_string(),
            })
        }

        async fn embed(
            &self,
            model: &str,
            inputs: &[String],
        ) -> Result<Vec<Vec<f32>>, OpenAiCompatError> {
            if !self.supports_embeddings {
                return Err(OpenAiCompatError::EmbeddingsUnavailable(model.to_string()));
            }
            Ok(inputs
                .iter()
                .map(|text| {
                    (0..self.embedding_dim).map(|i| (text.len() as f32 + i as f32) / 10.0).collect()
                })
                .collect())
        }

        fn count_tokens(&self, _model: &str, text: &str) -> Option<u32> {
            Some(text.split_whitespace().count() as u32)
        }

        fn models(&self) -> Vec<String> {
            vec!["gpt-4".to_string(), "ada".to_string()]
        }
    }

    // ── 27. OpenAiApiRouter::new stores allowed models ────────────────────────
    #[test]
    fn test_router_new_stores_models() {
        let router = OpenAiApiRouter::new(vec!["gpt-4".to_string(), "claude".to_string()]);
        assert!(router.is_model_allowed("gpt-4"));
        assert!(router.is_model_allowed("claude"));
        assert!(!router.is_model_allowed("unknown"));
        assert!(
            !router.has_backend(),
            "a fresh router has no inference backend"
        );
    }

    /// Regression test: `route_chat_completion` used to return
    /// `"[stub chat response]"` as if it were a model completion.
    #[tokio::test]
    async fn test_router_chat_completion_without_backend_is_rejected() {
        let router = OpenAiApiRouter::new(vec!["gpt-4".to_string()]);
        let err = router
            .route_chat_completion(&minimal_chat_req())
            .await
            .expect_err("no backend means no completion");
        assert!(matches!(err, OpenAiCompatError::BackendUnavailable));
        assert_eq!(err.status_code(), 503);
    }

    #[tokio::test]
    async fn test_router_route_chat_completion_uses_the_backend_output() {
        let router = OpenAiApiRouter::new(vec!["gpt-4".to_string()])
            .with_backend(Arc::new(RecordingBackend::new("REAL")));
        let resp = router
            .route_chat_completion(&minimal_chat_req())
            .await
            .expect("route_chat_completion");
        assert!(resp.id.starts_with("chatcmpl-"));
        assert_eq!(resp.model, "gpt-4");
        assert_eq!(resp.choices.len(), 1);
        let content = resp.choices[0].message.content.as_deref().unwrap_or_default();
        assert!(content.starts_with("REAL|"), "content = {content}");
        assert!(
            !content.contains("stub"),
            "no stub text may reach the client: {content}"
        );
        assert_eq!(
            resp.usage.completion_tokens, 7,
            "token counts must come from the backend"
        );
        // The timestamp must be a real clock reading, not a hash of the model.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        assert!(
            resp.created.abs_diff(now) < 60,
            "created = {} is not a current timestamp",
            resp.created
        );
    }

    // ── 29. OpenAiApiRouter route_chat_completion disallowed model ────────────
    #[tokio::test]
    async fn test_router_route_chat_completion_disallowed_model() {
        let router = OpenAiApiRouter::new(vec!["gpt-3".to_string()])
            .with_backend(Arc::new(RecordingBackend::new("REAL")));
        let req = minimal_chat_req(); // uses "gpt-4"
        let err = router.route_chat_completion(&req).await.unwrap_err();
        assert!(matches!(err, OpenAiCompatError::ModelNotAllowed(_)));
        assert_eq!(err.status_code(), 404);
    }

    // ── 30. OpenAiApiRouter route_completion happy path ───────────────────────
    #[tokio::test]
    async fn test_router_route_completion_happy() {
        let router = OpenAiApiRouter::new(vec!["davinci".to_string()])
            .with_backend(Arc::new(RecordingBackend::new("TEXT")));
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
        let resp = router.route_completion(&req).await.expect("route_completion");
        assert_eq!(resp.model, "davinci");
        assert_eq!(resp.object, "text_completion");
        assert!(resp.choices[0].text.starts_with("TEXT|"));
        assert!(!resp.choices[0].text.contains("stub"));
    }

    #[tokio::test]
    async fn test_router_route_completion_without_backend_is_rejected() {
        let router = OpenAiApiRouter::new(vec!["davinci".to_string()]);
        let req = CompletionRequest {
            model: "davinci".to_string(),
            prompt: CompletionPrompt::Single("Hello".to_string()),
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            echo: None,
        };
        let err = router.route_completion(&req).await.unwrap_err();
        assert!(matches!(err, OpenAiCompatError::BackendUnavailable));
    }

    // ── 31. OpenAiApiRouter route_embeddings happy path ───────────────────────
    #[tokio::test]
    async fn test_router_route_embeddings_returns_real_vectors() {
        let router = OpenAiApiRouter::new(vec!["ada".to_string()])
            .with_backend(Arc::new(RecordingBackend::new("x")));
        let req = EmbeddingRequest {
            model: "ada".to_string(),
            input: EmbeddingInput::Single("Hello world".to_string()),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        let resp = router.route_embeddings(&req).await.expect("route_embeddings");
        assert_eq!(resp.object, "list");
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].embedding.len(), 8);
        assert!(
            resp.data[0].embedding.iter().any(|v| *v != 0.0),
            "a zero vector is never an acceptable embedding: {:?}",
            resp.data[0].embedding
        );
    }

    /// Regression test: `route_embeddings` used to return `vec![0.0f32; 4]` for
    /// every input.
    #[tokio::test]
    async fn test_router_route_embeddings_without_backend_is_rejected() {
        let router = OpenAiApiRouter::new(vec!["ada".to_string()]);
        let req = EmbeddingRequest {
            model: "ada".to_string(),
            input: EmbeddingInput::Single("Hello world".to_string()),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        let err = router.route_embeddings(&req).await.unwrap_err();
        assert!(matches!(err, OpenAiCompatError::BackendUnavailable));
        assert_eq!(err.status_code(), 503);
    }

    #[tokio::test]
    async fn test_router_route_embeddings_reports_missing_embedding_model() {
        let router = OpenAiApiRouter::new(vec!["ada".to_string()])
            .with_backend(Arc::new(RecordingBackend::new("x").without_embeddings()));
        let req = EmbeddingRequest {
            model: "ada".to_string(),
            input: EmbeddingInput::Multiple(vec!["a".to_string(), "b".to_string()]),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        let err = router.route_embeddings(&req).await.unwrap_err();
        assert!(matches!(err, OpenAiCompatError::EmbeddingsUnavailable(_)));
    }

    #[tokio::test]
    async fn test_router_embeddings_multiple_inputs_get_distinct_vectors() {
        let router =
            OpenAiApiRouter::new(vec![]).with_backend(Arc::new(RecordingBackend::new("x")));
        let req = EmbeddingRequest {
            model: "ada".to_string(),
            input: EmbeddingInput::Multiple(vec!["a".to_string(), "abcd".to_string()]),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        let resp = router.route_embeddings(&req).await.expect("embeddings");
        assert_eq!(resp.data.len(), 2);
        assert_ne!(
            resp.data[0].embedding, resp.data[1].embedding,
            "different inputs must not share one embedding"
        );
    }

    #[test]
    fn test_router_models_without_backend_lists_the_allow_list() {
        let router = OpenAiApiRouter::new(vec!["gpt-4".to_string()]);
        let response = router.route_models();
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].id, "gpt-4");
    }

    #[test]
    fn test_router_models_intersects_backend_and_allow_list() {
        let router = OpenAiApiRouter::new(vec!["ada".to_string()])
            .with_backend(Arc::new(RecordingBackend::new("x")));
        let response = router.route_models();
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].id, "ada");
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
            OpenAiCompatError::BackendUnavailable,
            OpenAiCompatError::BackendError("cuda oom".to_string()),
            OpenAiCompatError::EmbeddingsUnavailable("ada".to_string()),
        ];
        for err in &cases {
            let oai_err = OpenAiApiRouter::format_error_response(err);
            // Every converted error should have a non-empty message.
            assert!(!oai_err.error.message.is_empty(), "empty message for {err}");
            assert!((400..=599).contains(&err.status_code()));
        }
    }

    // ── HTTP wiring ───────────────────────────────────────────────────────────

    async fn spawn_router(router: OpenAiApiRouter) -> String {
        let app = openai_compat_router(router);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{addr}")
    }

    /// Regression test: the OpenAI-compatible endpoints existed only as methods
    /// and were registered on no axum router at all.
    #[tokio::test]
    async fn test_http_chat_completions_endpoint_is_reachable() {
        let base = spawn_router(
            OpenAiApiRouter::new(vec!["gpt-4".to_string()])
                .with_backend(Arc::new(RecordingBackend::new("HTTP"))),
        )
        .await;

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}]
            }))
            .send()
            .await
            .expect("request");
        assert_eq!(response.status().as_u16(), 200);
        let body: serde_json::Value = response.json().await.expect("json");
        assert_eq!(body["object"], serde_json::json!("chat.completion"));
        let content = body["choices"][0]["message"]["content"].as_str().unwrap_or_default();
        assert!(content.starts_with("HTTP|"), "content = {content}");
    }

    #[tokio::test]
    async fn test_http_endpoints_return_503_without_a_backend() {
        let base = spawn_router(OpenAiApiRouter::new(vec![])).await;
        let client = reqwest::Client::new();

        for (path, payload) in [
            (
                "/v1/chat/completions",
                serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
            ),
            (
                "/v1/completions",
                serde_json::json!({"model": "m", "prompt": "hi"}),
            ),
            (
                "/v1/embeddings",
                serde_json::json!({"model": "m", "input": "hi"}),
            ),
        ] {
            let response = client
                .post(format!("{base}{path}"))
                .json(&payload)
                .send()
                .await
                .expect("request");
            assert_eq!(
                response.status().as_u16(),
                503,
                "{path} must report the missing backend"
            );
            let body: serde_json::Value = response.json().await.expect("json");
            assert_eq!(
                body["error"]["code"],
                serde_json::json!("service_unavailable")
            );
        }
    }

    #[tokio::test]
    async fn test_http_embeddings_endpoint_returns_real_vectors() {
        let base = spawn_router(
            OpenAiApiRouter::new(vec![]).with_backend(Arc::new(RecordingBackend::new("x"))),
        )
        .await;
        let client = reqwest::Client::new();
        let response = client
            .post(format!("{base}/v1/embeddings"))
            .json(&serde_json::json!({"model": "ada", "input": ["one", "two"]}))
            .send()
            .await
            .expect("request");
        assert_eq!(response.status().as_u16(), 200);
        let body: serde_json::Value = response.json().await.expect("json");
        let vectors = body["data"].as_array().expect("array");
        assert_eq!(vectors.len(), 2);
        for vector in vectors {
            let values = vector["embedding"].as_array().expect("embedding array");
            assert!(!values.is_empty());
            assert!(
                values.iter().any(|v| v.as_f64().unwrap_or(0.0) != 0.0),
                "no zero vectors may be served"
            );
        }
    }

    #[tokio::test]
    async fn test_http_models_endpoint_lists_backend_models() {
        let base = spawn_router(
            OpenAiApiRouter::new(vec![]).with_backend(Arc::new(RecordingBackend::new("x"))),
        )
        .await;
        let body: serde_json::Value = reqwest::get(format!("{base}/v1/models"))
            .await
            .expect("request")
            .json()
            .await
            .expect("json");
        let ids: Vec<&str> = body["data"]
            .as_array()
            .expect("array")
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        assert!(ids.contains(&"gpt-4"));
        assert!(ids.contains(&"ada"));
    }

    #[tokio::test]
    async fn test_http_validation_error_is_400() {
        let base = spawn_router(
            OpenAiApiRouter::new(vec![]).with_backend(Arc::new(RecordingBackend::new("x"))),
        )
        .await;
        let response = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({"model": "m", "messages": []}))
            .send()
            .await
            .expect("request");
        assert_eq!(response.status().as_u16(), 400);
    }
}
