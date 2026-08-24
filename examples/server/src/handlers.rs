//! Route handlers.
//!
//! Every response here comes from a real forward pass through the checkpoint
//! the caller loaded (`softmax`/`argmax` over genuine classifier-head logits,
//! a genuine autoregressive decode for generation) or from the loaded head's
//! own reported configuration. Where this server cannot compute something
//! honestly — character-level offsets, since the tokenizer stack behind
//! `AutoTokenizer` does not produce an offset mapping — the response reports
//! the honestly-available substitute (token indices, the tokenizer's own
//! subword text) instead of inventing a plausible-looking number.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    response::Json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::info;

use trustformers::core::tensor::Tensor;
use trustformers::core::traits::Model;
use trustformers::pipeline::text_generation::GenerationConfig;
use trustformers::{AutoModelForQuestionAnswering, AutoModelForTokenClassification};

use crate::error::{AppError, AppResult};
use crate::models::{LoadedModel, LoadedTask, ModelInfo, TaskKind};
use crate::AppState;

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    /// A local directory containing `config.json`, `model.safetensors`, and
    /// this checkpoint's tokenizer files. This server never downloads weights
    /// from the model hub — only `AutoConfig` (for the architecture shape)
    /// falls back to a hub lookup when no local `config.json` exists.
    pub model_name: String,
    pub task: TaskKind,
    /// Required for `text-classification` and `token-classification`; ignored
    /// otherwise.
    pub num_labels: Option<usize>,
    /// Optional friendly label names. Must have exactly `num_labels` entries
    /// or it is ignored in favour of `LABEL_0`, `LABEL_1`, ... — see
    /// [`crate::models::ModelInfo::labels`].
    pub labels: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct LoadModelResponse {
    pub model_id: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct ClassifyRequest {
    pub model_id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelScore {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f32,
    pub scores: Vec<LabelScore>,
}

#[derive(Debug, Deserialize)]
pub struct NerRequest {
    pub model_id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// The tokenizer's own text for this token (or merged run of tokens).
    /// May carry subword markers (e.g. a WordPiece `##` continuation prefix)
    /// since no character-offset mapping is available to reconstruct whole
    /// words safely.
    pub token: String,
    pub label: String,
    pub score: f32,
    /// Index of the (first, for a merged run) token in the tokenized
    /// sequence — not a character offset.
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct NerResponse {
    pub entities: Vec<Entity>,
}

#[derive(Debug, Deserialize)]
pub struct QaRequest {
    pub model_id: String,
    pub question: String,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaResult {
    pub answer: String,
    pub score: f32,
    /// Token indices into the tokenized `question [SEP] context` sequence —
    /// not character offsets into `context` (the tokenizer produces no
    /// offset mapping this server could use to compute those honestly).
    pub start_token: usize,
    pub end_token: usize,
}

/// Generation knobs that are genuinely wired through to
/// `AutoModelForCausalLM::generate`. Fields such as `do_sample`,
/// `repetition_penalty` or `num_beams` are deliberately absent: that method
/// always samples and never applies them, so accepting them here would mean
/// silently ignoring whatever a caller sent — exactly the kind of
/// accepted-but-dropped request this server must not allow.
#[derive(Debug, Default, Deserialize)]
pub struct GenerationOptions {
    /// Absolute target sequence length (prompt + generated tokens), not a
    /// new-tokens-only budget.
    pub max_length: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub model_id: String,
    pub prompt: String,
    #[serde(flatten)]
    pub options: GenerationOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub generated_text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    pub model_id: String,
    pub inputs: Vec<Value>,
}

#[derive(Debug, Serialize)]
pub struct BatchResponse {
    pub results: Vec<Value>,
    pub total_time_ms: u64,
}

// ---------------------------------------------------------------------------
// Shared numeric helpers
// ---------------------------------------------------------------------------

/// Numerically-stable softmax. Returns a uniform distribution (rather than
/// `NaN`s) on the degenerate all-equal / empty-sum input, and an empty vector
/// for empty input.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

fn tensor_to_vec(tensor: &Tensor, what: &str) -> AppResult<Vec<f32>> {
    tensor.data().map_err(|e| AppError::InferenceError(format!("failed to read {what}: {e}")))
}

// ---------------------------------------------------------------------------
// Task logic — reused by both the single-item endpoints and the batch endpoint
// ---------------------------------------------------------------------------

fn classify_text(loaded: &LoadedModel, text: &str) -> AppResult<ClassificationResult> {
    loaded.require_task(TaskKind::TextClassification)?;
    let LoadedTask::Classification { model, labels } = &loaded.task_data else {
        // `ModelManager::load_model` is the only place a `LoadedModel` is ever
        // constructed, and it always builds `task_data` from the same `task`
        // value stored alongside it, so `require_task` having just confirmed
        // `task == TextClassification` means this arm is unreachable today.
        // A graceful 500 here (instead of `unreachable!()`) means a future
        // change that breaks that invariant fails one request with a
        // diagnosable error instead of panicking the request task.
        return Err(AppError::InferenceError(
            "internal error: require_task confirmed a text-classification model but task_data \
             holds a different variant"
                .to_string(),
        ));
    };

    let inputs =
        loaded.tokenizer.encode(text).map_err(|e| AppError::InferenceError(format!("tokenization failed: {e}")))?;

    let logits = match model {
        trustformers::AutoModelForSequenceClassification::Bert(m) => m.forward(inputs).map(|o| o.logits),
        trustformers::AutoModelForSequenceClassification::Roberta(m) => m.forward(inputs).map(|o| o.logits),
        trustformers::AutoModelForSequenceClassification::Albert(m) => m.forward(inputs).map(|o| o.logits),
    }
    .map_err(|e| AppError::InferenceError(format!("forward pass failed: {e}")))?;

    let logits = tensor_to_vec(&logits, "classification logits")?;
    let probs = softmax(&logits);

    let mut scores: Vec<LabelScore> = labels
        .iter()
        .cloned()
        .zip(probs)
        .map(|(label, score)| LabelScore { label, score })
        .collect();
    scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let top = scores
        .first()
        .cloned()
        .ok_or_else(|| AppError::InferenceError("model produced no label scores".to_string()))?;

    Ok(ClassificationResult { label: top.label, score: top.score, scores })
}

/// Whether `merge_bio_entities` may safely merge/filter `predictions`: only
/// when the loaded label set actually asserts a BIO-style scheme. Labels the
/// caller made up freely (or left at the `LABEL_i` default) carry no such
/// promise, so every per-token prediction is reported as-is instead.
fn looks_like_bio_labels(labels: &[String]) -> bool {
    labels.iter().any(|l| l == "O") && labels.iter().any(|l| l.starts_with("B-") || l.starts_with("I-"))
}

fn merge_bio_entities(predictions: Vec<Entity>) -> Vec<Entity> {
    let mut merged: Vec<Entity> = Vec::new();
    for entity in predictions {
        if entity.label == "O" {
            continue;
        }
        let entity_type = entity.label.split_once('-').map_or(entity.label.as_str(), |(_, rest)| rest);

        // The mutation happens inside the same `is_some_and` closure that
        // decides whether to merge, rather than a separate `merged.last_mut()`
        // afterward: that avoids ever needing to re-assert (via `.expect()`)
        // that `merged` is still non-empty by the time of the mutation.
        let merged_into_previous = entity.label.starts_with("I-")
            && merged.last_mut().is_some_and(|prev| {
                let prev_type = prev.label.split_once('-').map_or(prev.label.as_str(), |(_, rest)| rest);
                let is_continuation = prev_type == entity_type && entity.index == prev.index + 1;
                if is_continuation {
                    prev.token = format!("{} {}", prev.token, entity.token);
                    prev.score = (prev.score + entity.score) / 2.0;
                }
                is_continuation
            });

        if !merged_into_previous {
            merged.push(entity);
        }
    }
    merged
}

fn tag_tokens(loaded: &LoadedModel, text: &str) -> AppResult<Vec<Entity>> {
    loaded.require_task(TaskKind::TokenClassification)?;
    let LoadedTask::TokenClassification { model, labels } = &loaded.task_data else {
        // See the identical comment in `classify_text` above: unreachable
        // today by construction, a graceful 500 rather than a panic if that
        // ever stops being true.
        return Err(AppError::InferenceError(
            "internal error: require_task confirmed a token-classification model but task_data \
             holds a different variant"
                .to_string(),
        ));
    };

    let inputs =
        loaded.tokenizer.encode(text).map_err(|e| AppError::InferenceError(format!("tokenization failed: {e}")))?;
    let input_ids = inputs.input_ids.clone();
    let special_tokens_mask = inputs.special_tokens_mask.clone();

    let logits = match model {
        AutoModelForTokenClassification::Bert(m) => m.forward(inputs).map(|o| o.logits),
        AutoModelForTokenClassification::Roberta(m) => m.forward(inputs).map(|o| o.logits),
        AutoModelForTokenClassification::Albert(m) => m.forward(inputs).map(|o| o.logits),
    }
    .map_err(|e| AppError::InferenceError(format!("forward pass failed: {e}")))?;

    let logits = tensor_to_vec(&logits, "token-classification logits")?;
    let num_labels = labels.len();
    let seq_len = input_ids.len();
    if num_labels == 0 || logits.len() != seq_len * num_labels {
        return Err(AppError::InferenceError(format!(
            "unexpected logits shape: got {} values for {seq_len} tokens x {num_labels} labels",
            logits.len()
        )));
    }

    let mut predictions = Vec::with_capacity(seq_len);
    for token_idx in 0..seq_len {
        let is_special =
            special_tokens_mask.as_ref().and_then(|mask| mask.get(token_idx)).copied().unwrap_or(0) == 1;
        if is_special {
            continue;
        }

        let token_logits = &logits[token_idx * num_labels..(token_idx + 1) * num_labels];
        let probs = softmax(token_logits);
        let (label_idx, score) = probs
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| AppError::InferenceError("token produced no label scores".to_string()))?;

        let label = labels.get(label_idx).cloned().unwrap_or_else(|| format!("LABEL_{label_idx}"));
        let token = loaded
            .tokenizer
            .id_to_token(input_ids[token_idx])
            .unwrap_or_else(|| format!("[id:{}]", input_ids[token_idx]));

        predictions.push(Entity { token, label, score, index: token_idx });
    }

    Ok(if looks_like_bio_labels(labels) { merge_bio_entities(predictions) } else { predictions })
}

fn answer_question(loaded: &LoadedModel, question: &str, context: &str) -> AppResult<QaResult> {
    loaded.require_task(TaskKind::QuestionAnswering)?;
    let LoadedTask::QuestionAnswering { model } = &loaded.task_data else {
        // See the identical comment in `classify_text` above: unreachable
        // today by construction, a graceful 500 rather than a panic if that
        // ever stops being true.
        return Err(AppError::InferenceError(
            "internal error: require_task confirmed a question-answering model but task_data \
             holds a different variant"
                .to_string(),
        ));
    };

    let inputs = loaded
        .tokenizer
        .encode_pair(question, context)
        .map_err(|e| AppError::InferenceError(format!("tokenization failed: {e}")))?;
    let input_ids = inputs.input_ids.clone();
    let token_type_ids = inputs.token_type_ids.clone().ok_or_else(|| {
        AppError::InferenceError(
            "tokenizer did not produce token_type_ids for this question/context pair; cannot tell where \
             the context segment begins"
                .to_string(),
        )
    })?;

    let (start_logits, end_logits) = match model {
        AutoModelForQuestionAnswering::Bert(m) => m.forward(inputs).map(|o| (o.start_logits, o.end_logits)),
        AutoModelForQuestionAnswering::Roberta(m) => m.forward(inputs).map(|o| (o.start_logits, o.end_logits)),
        AutoModelForQuestionAnswering::Albert(m) => m.forward(inputs).map(|o| (o.start_logits, o.end_logits)),
    }
    .map_err(|e| AppError::InferenceError(format!("forward pass failed: {e}")))?;

    let start_logits = tensor_to_vec(&start_logits, "start logits")?;
    let end_logits = tensor_to_vec(&end_logits, "end logits")?;

    let seq_len = input_ids.len();
    if start_logits.len() != seq_len || end_logits.len() != seq_len {
        return Err(AppError::InferenceError(format!(
            "unexpected QA logits shape: {} start / {} end values for {seq_len} tokens",
            start_logits.len(),
            end_logits.len()
        )));
    }

    // Only positions inside the context segment can ever be the answer: the
    // question tokens and the special tokens are never eligible.
    let context_positions: Vec<usize> =
        (0..seq_len).filter(|&i| token_type_ids.get(i).copied().unwrap_or(0) == 1).collect();
    if context_positions.is_empty() {
        return Err(AppError::InferenceError(
            "tokenizer produced no context-segment tokens for this pair".to_string(),
        ));
    }

    let start_probs = softmax(&start_logits);
    let end_probs = softmax(&end_logits);

    let start_pos = context_positions
        .iter()
        .copied()
        .max_by(|&a, &b| start_probs[a].partial_cmp(&start_probs[b]).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| {
            // Unreachable today: `context_positions.is_empty()` was already
            // checked above and returns early. A graceful 500 here (instead
            // of `.expect()`) means a future change to that check fails one
            // request with a diagnosable error instead of panicking it.
            AppError::InferenceError(
                "internal error: context_positions was checked non-empty above".to_string(),
            )
        })?;

    let end_pos = context_positions
        .iter()
        .copied()
        .filter(|&i| i >= start_pos)
        .max_by(|&a, &b| end_probs[a].partial_cmp(&end_probs[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(start_pos);

    // Geometric mean of the two independent argmax probabilities.
    let score = (start_probs[start_pos] * end_probs[end_pos]).sqrt();

    let answer = loaded
        .tokenizer
        .decode(&input_ids[start_pos..=end_pos])
        .map_err(|e| AppError::InferenceError(format!("failed to decode answer span: {e}")))?;

    Ok(QaResult { answer: answer.trim().to_string(), score, start_token: start_pos, end_token: end_pos })
}

fn generate_text(loaded: &LoadedModel, prompt: &str, options: &GenerationOptions) -> AppResult<GenerateResponse> {
    loaded.require_task(TaskKind::TextGeneration)?;
    let LoadedTask::Generation { model } = &loaded.task_data else {
        // See the identical comment in `classify_text` above: unreachable
        // today by construction, a graceful 500 rather than a panic if that
        // ever stops being true.
        return Err(AppError::InferenceError(
            "internal error: require_task confirmed a text-generation model but task_data holds \
             a different variant"
                .to_string(),
        ));
    };

    let inputs =
        loaded.tokenizer.encode(prompt).map_err(|e| AppError::InferenceError(format!("tokenization failed: {e}")))?;
    let prompt_len = inputs.input_ids.len();
    if prompt_len == 0 {
        return Err(AppError::BadRequest("prompt encoded to an empty token sequence".to_string()));
    }

    let mut config = GenerationConfig::default();
    if let Some(max_length) = options.max_length {
        if max_length <= prompt_len {
            return Err(AppError::BadRequest(format!(
                "`max_length` ({max_length}) must exceed the prompt's own token count ({prompt_len})"
            )));
        }
        config.max_length = max_length;
    }
    if let Some(temperature) = options.temperature {
        if !(temperature > 0.0 && temperature.is_finite()) {
            return Err(AppError::BadRequest("`temperature` must be a finite, positive number".to_string()));
        }
        config.temperature = temperature;
    }
    if let Some(top_k) = options.top_k {
        if top_k == 0 {
            return Err(AppError::BadRequest("`top_k` must be at least 1".to_string()));
        }
        config.top_k = Some(top_k);
    }
    if let Some(top_p) = options.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(AppError::BadRequest("`top_p` must lie in [0, 1]".to_string()));
        }
        config.top_p = Some(top_p);
    }

    let sequence = {
        let mut model = model
            .lock()
            .map_err(|_| AppError::InferenceError("generation lock was poisoned by an earlier panic".to_string()))?;
        model.generate(inputs, config).map_err(|e| AppError::InferenceError(format!("generation failed: {e}")))?
    };

    let completion_ids = &sequence[prompt_len.min(sequence.len())..];
    let generated_text = loaded
        .tokenizer
        .decode(completion_ids)
        .map_err(|e| AppError::InferenceError(format!("failed to decode generated tokens: {e}")))?;

    Ok(GenerateResponse {
        generated_text,
        prompt_tokens: prompt_len,
        completion_tokens: completion_ids.len(),
    })
}

// ---------------------------------------------------------------------------
// Model management handlers
// ---------------------------------------------------------------------------

pub async fn list_models(State(state): State<AppState>) -> Json<Vec<ModelInfo>> {
    Json(state.model_manager.list())
}

pub async fn get_model_info(
    Path(model_id): Path<String>,
    State(state): State<AppState>,
) -> AppResult<Json<ModelInfo>> {
    let loaded = state.model_manager.get(&model_id)?;
    Ok(Json(loaded.info()))
}

pub async fn load_model(
    State(state): State<AppState>,
    Json(request): Json<LoadModelRequest>,
) -> AppResult<Json<LoadModelResponse>> {
    info!(model_name = %request.model_name, task = request.task.as_str(), "loading model");

    let model_manager: Arc<crate::models::ModelManager> = Arc::clone(&state.model_manager);
    let LoadModelRequest { model_name, task, num_labels, labels } = request;

    // Loading reads and parses a whole checkpoint, so it runs on the blocking
    // pool instead of stalling the async reactor.
    let loaded = tokio::task::spawn_blocking(move || model_manager.load_model(&model_name, task, num_labels, labels))
        .await
        .map_err(|e| AppError::InferenceError(format!("model load task failed to run: {e}")))??;

    Ok(Json(LoadModelResponse {
        model_id: loaded.id.clone(),
        message: format!(
            "loaded `{}` for task `{}` in {:.3}s",
            loaded.model_name,
            loaded.task,
            loaded.load_duration.as_secs_f64()
        ),
    }))
}

pub async fn unload_model(
    Path(model_id): Path<String>,
    State(state): State<AppState>,
) -> AppResult<axum::http::StatusCode> {
    state.model_manager.unload_model(&model_id)?;
    Ok(axum::http::StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// Inference handlers
// ---------------------------------------------------------------------------

pub async fn text_classification(
    State(state): State<AppState>,
    Json(request): Json<ClassifyRequest>,
) -> AppResult<Json<ClassificationResult>> {
    let loaded = state.model_manager.get(&request.model_id)?;
    Ok(Json(classify_text(&loaded, &request.text)?))
}

pub async fn token_classification(
    State(state): State<AppState>,
    Json(request): Json<NerRequest>,
) -> AppResult<Json<NerResponse>> {
    let loaded = state.model_manager.get(&request.model_id)?;
    let entities = tag_tokens(&loaded, &request.text)?;
    Ok(Json(NerResponse { entities }))
}

pub async fn question_answering(
    State(state): State<AppState>,
    Json(request): Json<QaRequest>,
) -> AppResult<Json<QaResult>> {
    let loaded = state.model_manager.get(&request.model_id)?;
    Ok(Json(answer_question(&loaded, &request.question, &request.context)?))
}

pub async fn text_generation(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> AppResult<Json<GenerateResponse>> {
    let loaded = state.model_manager.get(&request.model_id)?;
    Ok(Json(generate_text(&loaded, &request.prompt, &request.options)?))
}

/// Runs every item in `request.inputs` through the real per-task logic above,
/// dispatched on whatever task the loaded model actually serves. There is no
/// batched kernel behind this — inputs are processed one after another — and
/// a single bad item fails the whole call rather than being written into the
/// response as if it were a prediction.
pub async fn batch_inference(
    State(state): State<AppState>,
    Json(request): Json<BatchRequest>,
) -> AppResult<Json<BatchResponse>> {
    if request.inputs.is_empty() {
        return Err(AppError::BadRequest("`inputs` was empty; nothing to process".to_string()));
    }

    let loaded = state.model_manager.get(&request.model_id)?;
    let start = std::time::Instant::now();

    let results: Vec<Value> = match loaded.task {
        TaskKind::TextClassification => request
            .inputs
            .iter()
            .map(|item| {
                let text = batch_field_str(item, "text", "text-classification")?;
                serde_json::to_value(classify_text(&loaded, text)?)
                    .map_err(|e| AppError::InferenceError(format!("failed to encode result: {e}")))
            })
            .collect::<AppResult<Vec<Value>>>()?,
        TaskKind::TokenClassification => request
            .inputs
            .iter()
            .map(|item| {
                let text = batch_field_str(item, "text", "token-classification")?;
                let entities = tag_tokens(&loaded, text)?;
                serde_json::to_value(NerResponse { entities })
                    .map_err(|e| AppError::InferenceError(format!("failed to encode result: {e}")))
            })
            .collect::<AppResult<Vec<Value>>>()?,
        TaskKind::QuestionAnswering => request
            .inputs
            .iter()
            .map(|item| {
                let question = batch_field_str(item, "question", "question-answering")?;
                let context = batch_field_str(item, "context", "question-answering")?;
                serde_json::to_value(answer_question(&loaded, question, context)?)
                    .map_err(|e| AppError::InferenceError(format!("failed to encode result: {e}")))
            })
            .collect::<AppResult<Vec<Value>>>()?,
        TaskKind::TextGeneration => request
            .inputs
            .iter()
            .map(|item| {
                let prompt = batch_field_str(item, "prompt", "text-generation")?;
                let options: GenerationOptions = serde_json::from_value(item.clone())
                    .map_err(|e| AppError::BadRequest(format!("invalid batch item for text-generation: {e}")))?;
                serde_json::to_value(generate_text(&loaded, prompt, &options)?)
                    .map_err(|e| AppError::InferenceError(format!("failed to encode result: {e}")))
            })
            .collect::<AppResult<Vec<Value>>>()?,
    };

    Ok(Json(BatchResponse { results, total_time_ms: start.elapsed().as_millis() as u64 }))
}

fn batch_field_str<'a>(item: &'a Value, field: &str, task: &str) -> AppResult<&'a str> {
    item.get(field).and_then(Value::as_str).ok_or_else(|| {
        AppError::BadRequest(format!("each batch item needs a string `{field}` field for task `{task}`"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_preserves_order() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn softmax_of_empty_is_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn softmax_never_produces_nan_on_degenerate_input() {
        let probs = softmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
        assert!(probs.iter().all(|p| p.is_finite()), "got {probs:?}");
    }

    #[test]
    fn bio_label_detection_requires_both_o_and_a_prefixed_label() {
        assert!(looks_like_bio_labels(&["O".to_string(), "B-PER".to_string(), "I-PER".to_string()]));
        assert!(!looks_like_bio_labels(&["LABEL_0".to_string(), "LABEL_1".to_string()]));
        assert!(!looks_like_bio_labels(&["O".to_string()]), "no B-/I- label present");
    }

    #[test]
    fn merge_bio_entities_joins_a_contiguous_i_run_and_drops_o() {
        let predictions = vec![
            Entity { token: "New".to_string(), label: "B-LOC".to_string(), score: 0.9, index: 0 },
            Entity { token: "York".to_string(), label: "I-LOC".to_string(), score: 0.8, index: 1 },
            Entity { token: "is".to_string(), label: "O".to_string(), score: 0.99, index: 2 },
            Entity { token: "big".to_string(), label: "O".to_string(), score: 0.9, index: 3 },
        ];
        let merged = merge_bio_entities(predictions);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].token, "New York");
        assert_eq!(merged[0].label, "B-LOC");
        assert!((merged[0].score - 0.85).abs() < 1e-6);
    }

    #[test]
    fn merge_bio_entities_does_not_merge_a_non_adjacent_repeat() {
        let predictions = vec![
            Entity { token: "Paris".to_string(), label: "B-LOC".to_string(), score: 0.9, index: 0 },
            Entity { token: "and".to_string(), label: "O".to_string(), score: 0.9, index: 1 },
            Entity { token: "Berlin".to_string(), label: "B-LOC".to_string(), score: 0.9, index: 2 },
        ];
        let merged = merge_bio_entities(predictions);
        assert_eq!(merged.len(), 2, "got {merged:?}");
        assert_eq!(merged[0].token, "Paris");
        assert_eq!(merged[1].token, "Berlin");
    }

    #[test]
    fn batch_field_str_names_the_missing_field_and_task() {
        let item = serde_json::json!({"other": "value"});
        let err = batch_field_str(&item, "text", "text-classification").expect_err("field is missing");
        let message = err.to_string();
        assert!(message.contains("text"), "message was: {message}");
        assert!(message.contains("text-classification"), "message was: {message}");
    }

    #[test]
    fn generation_options_flatten_alongside_prompt_and_model_id() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "model_id": "abc",
            "prompt": "hello",
            "max_length": 30,
            "temperature": 0.7
        }))
        .expect("valid request");
        assert_eq!(request.model_id, "abc");
        assert_eq!(request.options.max_length, Some(30));
        assert_eq!(request.options.top_k, None);
    }
}
