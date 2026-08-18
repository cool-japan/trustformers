//! A real inference backend for the OpenAI-compatible endpoints.
//!
//! [`BatchExecutorBackend`] is the adapter that connects
//! [`OpenAiApiRouter`](super::OpenAiApiRouter) to the batching stack's genuine
//! model executor ([`ModelBatchExecutor`]). Every completion it returns is the
//! decoded output of a real forward pass, every token count comes from the
//! model's real tokenizer, and every embedding is a pooled hidden state of the
//! model itself.
//!
//! Nothing here has a "plausible output" path: a backend built without an
//! embedding-capable model answers `/v1/embeddings` with
//! [`OpenAiCompatError::EmbeddingsUnavailable`], and any failure in the model or
//! the tokenizer surfaces as [`OpenAiCompatError::BackendError`] rather than as
//! text that looks like a completion.

use std::sync::Arc;

use crate::batching::processor::{
    BatchModel, EmbeddingModel, ModelBatchExecutor, Tokenizer, DEFAULT_MAX_NEW_TOKENS,
};

use super::{
    BackendGenerationOutput, BackendGenerationRequest, OpenAiCompatError, OpenAiInferenceBackend,
};

/// Finish reason reported when the model itself ended the completion.
const FINISH_STOP: &str = "stop";
/// Finish reason reported when the decode budget ran out first.
const FINISH_LENGTH: &str = "length";

/// An [`OpenAiInferenceBackend`] backed by a real [`ModelBatchExecutor`].
///
/// Build one with [`BatchExecutorBackend::new`] and, when the model can produce
/// hidden states, attach it as the embedding model with
/// [`BatchExecutorBackend::with_embedding_model`].
pub struct BatchExecutorBackend {
    /// Model identifier this backend answers to, as clients spell it.
    model_id: String,
    /// The real executor: owns the model and drives the forward passes.
    executor: Arc<ModelBatchExecutor>,
    /// The model's own tokenizer, used for prompt encoding and token counts.
    tokenizer: Arc<dyn Tokenizer>,
    /// Optional hidden-state pooler behind `/v1/embeddings`.
    embedding_model: Option<Arc<dyn EmbeddingModel>>,
    /// Decode budget used when a request carries no `max_tokens`.
    default_max_tokens: usize,
}

impl std::fmt::Debug for BatchExecutorBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchExecutorBackend")
            .field("model_id", &self.model_id)
            .field("default_max_tokens", &self.default_max_tokens)
            .field("has_embedding_model", &self.embedding_model.is_some())
            .finish()
    }
}

impl BatchExecutorBackend {
    /// Build a backend around `model` and `tokenizer`, serving `model_id`.
    ///
    /// The resulting backend can generate completions immediately; it reports no
    /// embedding capability until [`Self::with_embedding_model`] is called.
    pub fn new(
        model_id: impl Into<String>,
        model: Arc<dyn BatchModel>,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Self {
        let executor = ModelBatchExecutor::new(model).with_tokenizer(Arc::clone(&tokenizer));
        Self {
            model_id: model_id.into(),
            executor: Arc::new(executor),
            tokenizer,
            embedding_model: None,
            default_max_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }

    /// Attach the model that answers `/v1/embeddings`.
    ///
    /// Without one, embedding requests fail with
    /// [`OpenAiCompatError::EmbeddingsUnavailable`] — a zero vector is never
    /// substituted.
    #[must_use]
    pub fn with_embedding_model(mut self, embedding_model: Arc<dyn EmbeddingModel>) -> Self {
        self.embedding_model = Some(embedding_model);
        self
    }

    /// Override the decode budget used when a request carries no `max_tokens`.
    #[must_use]
    pub fn with_default_max_tokens(mut self, max_tokens: usize) -> Self {
        self.default_max_tokens = max_tokens.max(1);
        self
    }

    /// The model identifier this backend serves.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Whether this backend can produce embeddings.
    pub fn has_embedding_model(&self) -> bool {
        self.embedding_model.is_some()
    }

    /// Reject a request naming a model this backend does not serve.
    fn check_model(&self, model: &str) -> Result<(), OpenAiCompatError> {
        if model == self.model_id {
            Ok(())
        } else {
            Err(OpenAiCompatError::ModelNotAllowed(model.to_string()))
        }
    }

    /// Run the CPU-bound forward passes off the async runtime's worker threads.
    async fn generate_tokens(
        &self,
        prompt_ids: Vec<u32>,
        budget: usize,
    ) -> Result<Vec<u32>, OpenAiCompatError> {
        let executor = Arc::clone(&self.executor);
        let rows = vec![prompt_ids];
        let generated = tokio::task::spawn_blocking(move || executor.generate(&rows, budget))
            .await
            .map_err(|e| {
                OpenAiCompatError::BackendError(format!("inference task failed to run: {e}"))
            })?
            .map_err(|e| OpenAiCompatError::BackendError(e.to_string()))?;

        generated.into_iter().next().ok_or_else(|| {
            OpenAiCompatError::BackendError(
                "the model executor returned no row for a single-row batch".to_string(),
            )
        })
    }

    /// Truncate `text` at the earliest stop sequence, if any.
    ///
    /// Returns the truncated text and whether a stop sequence was found. Empty
    /// stop strings are ignored: they would match at offset 0 and make every
    /// completion empty.
    fn apply_stop_sequences(text: &str, stop: &[String]) -> (String, bool) {
        let cut = stop
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| text.find(s.as_str()))
            .min();
        match cut {
            Some(index) => (text[..index].to_string(), true),
            None => (text.to_string(), false),
        }
    }
}

#[async_trait::async_trait]
impl OpenAiInferenceBackend for BatchExecutorBackend {
    /// Generate a completion with the real model.
    ///
    /// Token accounting is exact: `prompt_tokens` is the length of the encoded
    /// prompt and `completion_tokens` is the number of tokens the model actually
    /// emitted. When a stop sequence truncates the visible text, the returned
    /// text is re-encoded so the reported count matches what the client received
    /// rather than what the decoder happened to run past.
    async fn generate(
        &self,
        request: &BackendGenerationRequest,
    ) -> Result<BackendGenerationOutput, OpenAiCompatError> {
        self.check_model(&request.model)?;

        let prompt_ids = self.tokenizer.encode(&request.prompt);
        if prompt_ids.is_empty() {
            return Err(OpenAiCompatError::BackendError(
                "the prompt encoded to zero tokens; there is nothing to condition on".to_string(),
            ));
        }
        let prompt_tokens = prompt_ids.len() as u32;

        let budget =
            request.max_tokens.map(|n| n as usize).unwrap_or(self.default_max_tokens).max(1);

        let mut ids = self.generate_tokens(prompt_ids, budget).await?;

        // A trailing end-of-sequence id means the model stopped on its own; it is
        // a control token, not part of the completion the client asked for.
        let eos = self.executor.model().eos_token_id();
        let hit_eos = matches!((ids.last(), eos), (Some(last), Some(eos)) if *last == eos);
        if hit_eos {
            ids.pop();
        }
        let generated_tokens = ids.len() as u32;

        let decoded = self.tokenizer.decode(&ids);
        let (text, hit_stop) = Self::apply_stop_sequences(&decoded, &request.stop);

        let completion_tokens = if hit_stop {
            self.tokenizer.encode(&text).len() as u32
        } else {
            generated_tokens
        };

        let finish_reason = if hit_stop || hit_eos { FINISH_STOP } else { FINISH_LENGTH };

        Ok(BackendGenerationOutput {
            text,
            prompt_tokens: Some(prompt_tokens),
            completion_tokens: Some(completion_tokens),
            finish_reason: finish_reason.to_string(),
        })
    }

    /// Embed `inputs` by pooling the model's own hidden states.
    async fn embed(
        &self,
        model: &str,
        inputs: &[String],
    ) -> Result<Vec<Vec<f32>>, OpenAiCompatError> {
        // A model this backend does not serve is a 404 regardless of whether an
        // embedding model happens to be attached, so check it first.
        self.check_model(model)?;
        let embedding_model = self
            .embedding_model
            .as_ref()
            .ok_or_else(|| OpenAiCompatError::EmbeddingsUnavailable(model.to_string()))?;

        let mut rows: Vec<Vec<u32>> = Vec::with_capacity(inputs.len());
        for (index, text) in inputs.iter().enumerate() {
            let ids = self.tokenizer.encode(text);
            if ids.is_empty() {
                return Err(OpenAiCompatError::BackendError(format!(
                    "input {index} encoded to zero tokens; there is nothing to embed"
                )));
            }
            rows.push(ids);
        }

        let embedding_model = Arc::clone(embedding_model);
        let embeddings = tokio::task::spawn_blocking(move || {
            rows.iter()
                .map(|ids| embedding_model.embed_tokens(ids))
                .collect::<anyhow::Result<Vec<Vec<f32>>>>()
        })
        .await
        .map_err(|e| OpenAiCompatError::BackendError(format!("embedding task failed to run: {e}")))?
        .map_err(|e| OpenAiCompatError::BackendError(e.to_string()))?;

        Ok(embeddings)
    }

    /// Count tokens with the model's own tokenizer.
    fn count_tokens(&self, model: &str, text: &str) -> Option<u32> {
        if model == self.model_id {
            Some(self.tokenizer.encode(text).len() as u32)
        } else {
            None
        }
    }

    fn models(&self) -> Vec<String> {
        vec![self.model_id.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batching::model_executor::{ByteTokenizer, Gpt2BatchModel};
    use crate::openai_compat::{
        ChatCompletionRequest, ChatMessage, ChatRole, CompletionPrompt, CompletionRequest,
        EmbeddingInput, EmbeddingRequest, OpenAiApiRouter,
    };
    use trustformers_models::gpt2::Gpt2Config;

    /// A genuinely small GPT-2 over a byte vocabulary: real architecture, real
    /// (untrained) weights, real forward pass.
    pub(crate) fn tiny_gpt2() -> Arc<Gpt2BatchModel> {
        let config = Gpt2Config {
            vocab_size: ByteTokenizer::VOCAB_SIZE,
            n_positions: 64,
            n_embd: 16,
            n_layer: 1,
            n_head: 2,
            n_inner: Some(32),
            resid_pdrop: 0.0,
            embd_pdrop: 0.0,
            attn_pdrop: 0.0,
            bos_token_id: ByteTokenizer::EOT_ID,
            eos_token_id: ByteTokenizer::EOT_ID,
            ..Gpt2Config::default()
        };
        Arc::new(Gpt2BatchModel::untrained(config).expect("tiny GPT-2 must build"))
    }

    pub(crate) fn tiny_backend() -> BatchExecutorBackend {
        let model = tiny_gpt2();
        BatchExecutorBackend::new(
            "tiny-gpt2",
            Arc::clone(&model) as Arc<dyn BatchModel>,
            Arc::new(ByteTokenizer),
        )
        .with_embedding_model(model as Arc<dyn EmbeddingModel>)
        .with_default_max_tokens(4)
    }

    fn chat_request(prompt: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "tiny-gpt2".to_string(),
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: Some(prompt.to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            top_p: None,
            n: None,
            max_tokens: Some(4),
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            user: None,
        }
    }

    /// Regression: `/v1/chat/completions` had no inference path at all. It must
    /// now run a real forward pass and report real token counts.
    #[tokio::test]
    async fn chat_completion_runs_a_real_forward_pass() {
        let router = OpenAiApiRouter::new(vec!["tiny-gpt2".to_string()])
            .with_backend(Arc::new(tiny_backend()));
        let response = router
            .route_chat_completion(&chat_request("Hello"))
            .await
            .expect("must generate");

        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.model, "tiny-gpt2");
        // The flattened chat prompt is "user: Hello\n" — 12 bytes, hence 12
        // byte-level tokens. This is the real tokenizer's count; the `len / 4`
        // fallback the router uses without a tokenizer would have said 3.
        assert_eq!(response.usage.prompt_tokens, 12);
        assert!(response.usage.completion_tokens <= 4);
        assert!(matches!(
            response.choices[0].finish_reason.as_str(),
            FINISH_STOP | FINISH_LENGTH
        ));
    }

    /// Regression: the completion must be model output, never an echo of the
    /// prompt or a canned string.
    #[tokio::test]
    async fn completion_output_is_not_an_echo() {
        let backend = tiny_backend();
        let request = BackendGenerationRequest {
            model: "tiny-gpt2".to_string(),
            prompt: "The quick brown fox".to_string(),
            messages: Vec::new(),
            max_tokens: Some(4),
            temperature: None,
            top_p: None,
            stop: Vec::new(),
        };
        let output = backend.generate(&request).await.expect("must generate");

        assert!(!output.text.contains("The quick brown fox"));
        assert!(!output.text.starts_with("Processed: "));
        assert_eq!(output.prompt_tokens, Some(19));
    }

    /// The decode budget must be honoured: `max_tokens` bounds the completion.
    #[tokio::test]
    async fn max_tokens_bounds_the_completion() {
        let backend = tiny_backend();
        let request = CompletionRequest {
            model: "tiny-gpt2".to_string(),
            prompt: CompletionPrompt::Single("abc".to_string()),
            max_tokens: Some(2),
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            echo: None,
        };
        let router = OpenAiApiRouter::new(Vec::new()).with_backend(Arc::new(backend));
        let response = router.route_completion(&request).await.expect("must generate");
        assert!(response.usage.completion_tokens <= 2);
        assert_eq!(response.usage.prompt_tokens, 3);
    }

    /// Regression: `/v1/embeddings` must return pooled hidden states of the real
    /// model — vectors of the model's hidden size, distinct for distinct inputs.
    #[tokio::test]
    async fn embeddings_are_real_pooled_hidden_states() {
        let router = OpenAiApiRouter::new(Vec::new()).with_backend(Arc::new(tiny_backend()));
        let request = EmbeddingRequest {
            model: "tiny-gpt2".to_string(),
            input: EmbeddingInput::Multiple(vec!["alpha".to_string(), "omega".to_string()]),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        let response = router.route_embeddings(&request).await.expect("must embed");

        assert_eq!(response.data.len(), 2);
        for item in &response.data {
            assert_eq!(item.embedding.len(), 16, "hidden size of the tiny model");
            assert!(item.embedding.iter().all(|v| v.is_finite()));
            assert!(
                item.embedding.iter().any(|v| v.abs() > f32::EPSILON),
                "a real embedding must not be an all-zero vector"
            );
        }
        assert_ne!(response.data[0].embedding, response.data[1].embedding);
    }

    /// Regression: a backend with no embedding model must answer honestly rather
    /// than fabricating a vector.
    #[tokio::test]
    async fn embeddings_without_an_embedding_model_are_unavailable() {
        let model = tiny_gpt2();
        let backend = BatchExecutorBackend::new(
            "tiny-gpt2",
            model as Arc<dyn BatchModel>,
            Arc::new(ByteTokenizer),
        );
        assert!(!backend.has_embedding_model());

        let error = backend
            .embed("tiny-gpt2", &["hello".to_string()])
            .await
            .expect_err("must not fabricate an embedding");
        assert!(matches!(error, OpenAiCompatError::EmbeddingsUnavailable(_)));
        assert_eq!(error.status_code(), 503);
    }

    /// Stop sequences must truncate the completion and be reported as `"stop"`.
    #[test]
    fn stop_sequences_truncate_at_the_earliest_match() {
        let (text, hit) = BatchExecutorBackend::apply_stop_sequences(
            "one END two STOP",
            &["STOP".to_string(), "END".to_string()],
        );
        assert_eq!(text, "one ");
        assert!(hit);

        let (text, hit) =
            BatchExecutorBackend::apply_stop_sequences("nothing here", &["STOP".to_string()]);
        assert_eq!(text, "nothing here");
        assert!(!hit);

        // An empty stop string must not blank out every completion.
        let (text, hit) = BatchExecutorBackend::apply_stop_sequences("keep me", &[String::new()]);
        assert_eq!(text, "keep me");
        assert!(!hit);
    }

    /// A model the backend does not serve is a 404, not a silent substitution.
    #[tokio::test]
    async fn unknown_model_is_refused() {
        let backend = tiny_backend();
        let request = BackendGenerationRequest {
            model: "gpt-4".to_string(),
            prompt: "hello".to_string(),
            messages: Vec::new(),
            max_tokens: Some(2),
            temperature: None,
            top_p: None,
            stop: Vec::new(),
        };
        let error = backend.generate(&request).await.expect_err("must refuse");
        assert!(matches!(error, OpenAiCompatError::ModelNotAllowed(_)));
        assert_eq!(error.status_code(), 404);
    }

    /// `/v1/models` must report what the backend can actually serve.
    #[test]
    fn models_reports_the_served_model() {
        let router = OpenAiApiRouter::new(Vec::new()).with_backend(Arc::new(tiny_backend()));
        let list = router.route_models();
        assert_eq!(list.data.len(), 1);
        assert_eq!(list.data[0].id, "tiny-gpt2");
    }

    /// Token counts must come from the real tokenizer, not the `len / 4`
    /// approximation the router falls back to without one.
    #[test]
    fn token_counts_come_from_the_real_tokenizer() {
        let backend = tiny_backend();
        assert_eq!(backend.count_tokens("tiny-gpt2", "abcdefgh"), Some(8));
        assert_eq!(backend.count_tokens("other-model", "abcdefgh"), None);
    }
}
