use axum::{
#![allow(unused_variables)]
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, error};

use crate::{
    error::{AppError, AppResult},
    models::ModelInfo,
    AppState,
};
use trustformers_core::{tensor::Tensor, traits::TokenizedInput};

// Request/Response types

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub model_name: String,
    pub model_type: String,
    pub cache_dir: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LoadModelResponse {
    pub model_id: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct TextClassificationRequest {
    pub model_id: String,
    pub text: String,
    pub candidate_labels: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct TextClassificationResponse {
    pub label: String,
    pub score: f32,
    pub scores: Vec<LabelScore>,
}

#[derive(Debug, Serialize)]
pub struct LabelScore {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Deserialize)]
pub struct TextGenerationRequest {
    pub model_id: String,
    pub prompt: String,
    pub max_length: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct TextGenerationResponse {
    pub generated_text: String,
    pub tokens_generated: usize,
}

#[derive(Debug, Deserialize)]
pub struct QuestionAnsweringRequest {
    pub model_id: String,
    pub question: String,
    pub context: String,
}

#[derive(Debug, Serialize)]
pub struct QuestionAnsweringResponse {
    pub answer: String,
    pub score: f32,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Deserialize)]
pub struct TokenClassificationRequest {
    pub model_id: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct TokenClassificationResponse {
    pub entities: Vec<Entity>,
}

#[derive(Debug, Serialize)]
pub struct Entity {
    pub entity: String,
    pub score: f32,
    pub index: usize,
    pub word: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    pub model_id: String,
    pub task: String,
    pub inputs: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct BatchResponse {
    pub results: Vec<serde_json::Value>,
    pub total_time_ms: u64,
}

// Handler implementations

pub async fn list_models(State(state): State<AppState>) -> AppResult<Json<Vec<ModelInfo>>> {
    let models = state.model_manager.list_models().await;
    Ok(Json(models))
}

pub async fn get_model_info(
    Path(model_id): Path<String>,
    State(state): State<AppState>,
) -> AppResult<Json<ModelInfo>> {
    state
        .model_manager
        .get_model_info(&model_id)
        .await
        .ok_or_else(|| AppError::ModelNotFound(model_id))
        .map(Json)
}

pub async fn load_model(
    State(state): State<AppState>,
    Json(request): Json<LoadModelRequest>,
) -> AppResult<Json<LoadModelResponse>> {
    info!("Loading model: {} (type: {})", request.model_name, request.model_type);

    let model_id = state
        .model_manager
        .load_model(&request.model_name, &request.model_type)
        .await
        .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

    Ok(Json(LoadModelResponse {
        model_id,
        message: format!("Model {} loaded successfully", request.model_name),
    }))
}

pub async fn unload_model(
    Path(model_id): Path<String>,
    State(state): State<AppState>,
) -> AppResult<StatusCode> {
    state
        .model_manager
        .unload_model(&model_id)
        .await
        .map_err(|e| AppError::ModelNotFound(model_id))?;

    Ok(StatusCode::NO_CONTENT)
}

pub async fn text_classification(
    State(state): State<AppState>,
    Json(request): Json<TextClassificationRequest>,
) -> AppResult<Json<TextClassificationResponse>> {
    let (model, tokenizer) = state
        .model_manager
        .get_model(&request.model_id)
        .await
        .ok_or_else(|| AppError::ModelNotFound(request.model_id.clone()))?;

    // Tokenize input
    let inputs = tokenizer
        .encode(&request.text)
        .map_err(|e| AppError::InferenceError(format!("Tokenization failed: {}", e)))?;

    // Create tensor from token IDs
    let input_ids = Tensor::from_vec(
        inputs.token_ids.iter().map(|&id| id as f32).collect(),
        vec![1, inputs.token_ids.len()],
    ).map_err(|e| AppError::InferenceError(format!("Tensor creation failed: {}", e)))?;

    // Run inference
    let outputs = model
        .forward(&input_ids, None, None)
        .map_err(|e| AppError::InferenceError(format!("Forward pass failed: {}", e)))?;

    // Apply softmax to get probabilities
    let probs = outputs.logits
        .softmax(-1)
        .map_err(|e| AppError::InferenceError(format!("Softmax failed: {}", e)))?;

    // Get the predicted class
    let probs_data = probs.data();
    let labels = request.candidate_labels.unwrap_or_else(|| {
        vec!["NEGATIVE".to_string(), "POSITIVE".to_string()]
    });

    let mut scores: Vec<LabelScore> = labels
        .iter()
        .enumerate()
        .map(|(i, label)| LabelScore {
            label: label.clone(),
            score: probs_data.get(i).copied().unwrap_or(0.0),
        })
        .collect();

    scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    Ok(Json(TextClassificationResponse {
        label: scores[0].label.clone(),
        score: scores[0].score,
        scores,
    }))
}

pub async fn text_generation(
    State(state): State<AppState>,
    Json(request): Json<TextGenerationRequest>,
) -> AppResult<Json<TextGenerationResponse>> {
    let (model, tokenizer) = state
        .model_manager
        .get_model(&request.model_id)
        .await
        .ok_or_else(|| AppError::ModelNotFound(request.model_id.clone()))?;

    // Tokenize prompt
    let inputs = tokenizer
        .encode(&request.prompt)
        .map_err(|e| AppError::InferenceError(format!("Tokenization failed: {}", e)))?;

    // Create tensor from token IDs
    let input_ids = Tensor::from_vec(
        inputs.token_ids.iter().map(|&id| id as f32).collect(),
        vec![1, inputs.token_ids.len()],
    ).map_err(|e| AppError::InferenceError(format!("Tensor creation failed: {}", e)))?;

    let max_length = request.max_length.unwrap_or(50);
    let temperature = request.temperature.unwrap_or(1.0);
    let top_k = request.top_k.unwrap_or(50);
    let top_p = request.top_p.unwrap_or(0.9);

    // Generate tokens iteratively
    let mut generated_tokens = inputs.token_ids.clone();
    let mut tokens_generated = 0;

    for _ in 0..(max_length - inputs.token_ids.len()) {
        // Create input tensor for current sequence
        let current_input = Tensor::from_vec(
            generated_tokens.iter().map(|&id| id as f32).collect(),
            vec![1, generated_tokens.len()],
        ).map_err(|e| AppError::InferenceError(format!("Tensor creation failed: {}", e)))?;

        // Run inference
        let outputs = model
            .forward(&current_input, None, None)
            .map_err(|e| AppError::InferenceError(format!("Forward pass failed: {}", e)))?;

        // Get logits for last token
        let logits_data = outputs.logits.data();
        let vocab_size = logits_data.len() / generated_tokens.len();
        let last_token_logits = &logits_data[(generated_tokens.len() - 1) * vocab_size..];

        // Apply temperature scaling
        let scaled_logits: Vec<f32> = last_token_logits.iter()
            .map(|&logit| logit / temperature)
            .collect();

        // Apply top-k filtering
        let mut indexed_logits: Vec<(usize, f32)> = scaled_logits.iter()
            .enumerate()
            .map(|(i, &logit)| (i, logit))
            .collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top-k tokens
        let top_k_tokens = &indexed_logits[..top_k.min(indexed_logits.len())];

        // Apply softmax to top-k tokens
        let max_logit = top_k_tokens[0].1;
        let exp_logits: Vec<f32> = top_k_tokens.iter()
            .map(|(_, logit)| (logit - max_logit).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter()
            .map(|&exp_logit| exp_logit / sum_exp)
            .collect();

        // Apply top-p (nucleus) sampling
        let mut cumulative_prob = 0.0;
        let mut nucleus_tokens = Vec::new();
        for (i, &prob) in probs.iter().enumerate() {
            cumulative_prob += prob;
            nucleus_tokens.push((top_k_tokens[i].0, prob));
            if cumulative_prob >= top_p {
                break;
            }
        }

        // Sample from nucleus
        let total_nucleus_prob: f32 = nucleus_tokens.iter().map(|(_, p)| p).sum();
        let mut sample_point = rand::random::<f32>() * total_nucleus_prob;
        let mut selected_token = nucleus_tokens[0].0;

        for (token_id, prob) in nucleus_tokens {
            sample_point -= prob;
            if sample_point <= 0.0 {
                selected_token = token_id;
                break;
            }
        }

        // Add selected token to sequence
        generated_tokens.push(selected_token as u32);
        tokens_generated += 1;

        // Check for end-of-sequence token (assuming 2 is EOS)
        if selected_token == 2 {
            break;
        }
    }

    // Decode generated tokens
    let generated_text = tokenizer
        .decode(&generated_tokens, false)
        .map_err(|e| AppError::InferenceError(format!("Decoding failed: {}", e)))?;

    Ok(Json(TextGenerationResponse {
        generated_text,
        tokens_generated,
    }))
}

pub async fn question_answering(
    State(state): State<AppState>,
    Json(request): Json<QuestionAnsweringRequest>,
) -> AppResult<Json<QuestionAnsweringResponse>> {
    let (model, tokenizer) = state
        .model_manager
        .get_model(&request.model_id)
        .await
        .ok_or_else(|| AppError::ModelNotFound(request.model_id.clone()))?;

    // Combine question and context
    let combined_text = format!("[CLS] {} [SEP] {} [SEP]", request.question, request.context);

    // Tokenize
    let inputs = tokenizer
        .encode(&combined_text)
        .map_err(|e| AppError::InferenceError(format!("Tokenization failed: {}", e)))?;

    // Create tensor from token IDs
    let input_ids = Tensor::from_vec(
        inputs.token_ids.iter().map(|&id| id as f32).collect(),
        vec![1, inputs.token_ids.len()],
    ).map_err(|e| AppError::InferenceError(format!("Tensor creation failed: {}", e)))?;

    // Run inference
    let outputs = model
        .forward(&input_ids, None, None)
        .map_err(|e| AppError::InferenceError(format!("Forward pass failed: {}", e)))?;

    // For question answering, we expect start and end logits
    // The model should output logits for start and end positions
    let logits_data = outputs.logits.data();
    let seq_len = inputs.token_ids.len();

    // Assuming the model outputs [batch_size, seq_len, 2] where 2 = [start_logits, end_logits]
    // Split logits into start and end predictions
    let start_logits: Vec<f32> = (0..seq_len)
        .map(|i| logits_data.get(i * 2).copied().unwrap_or(0.0))
        .collect();
    let end_logits: Vec<f32> = (0..seq_len)
        .map(|i| logits_data.get(i * 2 + 1).copied().unwrap_or(0.0))
        .collect();

    // Apply softmax to get probabilities
    let start_max = start_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let start_exp: Vec<f32> = start_logits.iter()
        .map(|&logit| (logit - start_max).exp())
        .collect();
    let start_sum: f32 = start_exp.iter().sum();
    let start_probs: Vec<f32> = start_exp.iter()
        .map(|&exp_val| exp_val / start_sum)
        .collect();

    let end_max = end_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let end_exp: Vec<f32> = end_logits.iter()
        .map(|&logit| (logit - end_max).exp())
        .collect();
    let end_sum: f32 = end_exp.iter().sum();
    let end_probs: Vec<f32> = end_exp.iter()
        .map(|&exp_val| exp_val / end_sum)
        .collect();

    // Find the best start and end positions
    let start_pos = start_probs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let end_pos = end_probs.iter()
        .enumerate()
        .skip(start_pos) // End must be after start
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(start_pos);

    // Calculate confidence score (geometric mean of start and end probabilities)
    let confidence_score = (start_probs[start_pos] * end_probs[end_pos]).sqrt();

    // Extract answer text from the context
    // Find the SEP token that separates question from context
    let sep_token_id = 102u32; // [SEP] token for BERT-like models
    let context_start = inputs.token_ids.iter()
        .position(|&id| id == sep_token_id)
        .map(|pos| pos + 1) // Start after first [SEP]
        .unwrap_or(0);

    // Adjust positions to be relative to the original context
    let adjusted_start = if start_pos >= context_start {
        start_pos - context_start
    } else {
        0
    };
    let adjusted_end = if end_pos >= context_start {
        end_pos - context_start
    } else {
        adjusted_start
    };

    // Decode the answer tokens
    let answer_tokens = if adjusted_end >= adjusted_start {
        inputs.token_ids[context_start + adjusted_start..=context_start + adjusted_end].to_vec()
    } else {
        vec![]
    };

    let answer = if !answer_tokens.is_empty() {
        tokenizer
            .decode(&answer_tokens, false)
            .unwrap_or_else(|_| "Unable to decode answer".to_string())
    } else {
        "No answer found".to_string()
    };

    // Calculate character positions in the original context
    let context_words: Vec<&str> = request.context.split_whitespace().collect();
    let char_start = if adjusted_start < context_words.len() {
        context_words.iter()
            .take(adjusted_start)
            .map(|word| word.len() + 1) // +1 for space
            .sum::<usize>()
            .saturating_sub(1) // Remove last space
    } else {
        0
    };

    let char_end = if adjusted_end < context_words.len() {
        context_words.iter()
            .take(adjusted_end + 1)
            .map(|word| word.len() + 1) // +1 for space
            .sum::<usize>()
            .saturating_sub(1) // Remove last space
    } else {
        request.context.len()
    };

    Ok(Json(QuestionAnsweringResponse {
        answer: answer.trim().to_string(),
        score: confidence_score,
        start: char_start,
        end: char_end,
    }))
}

pub async fn token_classification(
    State(state): State<AppState>,
    Json(request): Json<TokenClassificationRequest>,
) -> AppResult<Json<TokenClassificationResponse>> {
    let (model, tokenizer) = state
        .model_manager
        .get_model(&request.model_id)
        .await
        .ok_or_else(|| AppError::ModelNotFound(request.model_id.clone()))?;

    // Tokenize
    let inputs = tokenizer
        .encode(&request.text)
        .map_err(|e| AppError::InferenceError(format!("Tokenization failed: {}", e)))?;

    // Create tensor from token IDs
    let input_ids = Tensor::from_vec(
        inputs.token_ids.iter().map(|&id| id as f32).collect(),
        vec![1, inputs.token_ids.len()],
    ).map_err(|e| AppError::InferenceError(format!("Tensor creation failed: {}", e)))?;

    // Run inference
    let outputs = model
        .forward(&input_ids, None, None)
        .map_err(|e| AppError::InferenceError(format!("Forward pass failed: {}", e)))?;

    // Apply softmax to get probabilities for each token
    let probs = outputs.logits
        .softmax(-1)
        .map_err(|e| AppError::InferenceError(format!("Softmax failed: {}", e)))?;

    let probs_data = probs.data();
    let seq_len = inputs.token_ids.len();

    // Common NER labels (BIO format)
    let ner_labels = vec![
        "O",      // Outside
        "B-PER",  // Begin Person
        "I-PER",  // Inside Person
        "B-ORG",  // Begin Organization
        "I-ORG",  // Inside Organization
        "B-LOC",  // Begin Location
        "I-LOC",  // Inside Location
        "B-MISC", // Begin Miscellaneous
        "I-MISC", // Inside Miscellaneous
    ];

    let num_labels = ner_labels.len();
    let mut entities = Vec::new();

    // Track token offsets for character positions
    let tokens: Vec<String> = inputs.token_ids.iter()
        .map(|&id| {
            // Simple token decoding - in practice, you'd use proper tokenizer decode
            format!("token_{}", id)
        })
        .collect();

    // Extract predictions for each token
    for token_idx in 0..seq_len {
        let token_probs = &probs_data[token_idx * num_labels..(token_idx + 1) * num_labels];

        // Find the label with highest probability
        let (predicted_label_idx, &max_prob) = token_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        let predicted_label = ner_labels.get(predicted_label_idx).unwrap_or(&"O");

        // Only include non-O (Outside) predictions with reasonable confidence
        if predicted_label != &"O" && max_prob > 0.5 {
            // Calculate character positions in original text
            // This is simplified - in practice, you'd need proper token-to-char alignment
            let word_start = token_idx * 6; // Rough estimate
            let word_end = word_start + 5;   // Rough estimate

            // Extract word from token (simplified)
            let word = if token_idx < request.text.split_whitespace().count() {
                request.text
                    .split_whitespace()
                    .nth(token_idx)
                    .unwrap_or("unknown")
                    .to_string()
            } else {
                format!("token_{}", inputs.token_ids.get(token_idx).unwrap_or(&0))
            };

            entities.push(Entity {
                entity: predicted_label.to_string(),
                score: max_prob,
                index: token_idx,
                word,
                start: word_start.min(request.text.len()),
                end: word_end.min(request.text.len()),
            });
        }
    }

    // Post-process entities to merge adjacent tokens of the same type
    let mut merged_entities = Vec::new();
    let mut current_entity: Option<Entity> = None;

    for entity in entities {
        match &current_entity {
            None => {
                current_entity = Some(entity);
            }
            Some(current) => {
                // Check if this entity continues the previous one
                let current_type = current.entity.split('-').last().unwrap_or("");
                let entity_type = entity.entity.split('-').last().unwrap_or("");
                let is_continuation = entity.entity.starts_with("I-") &&
                                    current_type == entity_type &&
                                    entity.index == current.index + 1;

                if is_continuation {
                    // Merge with current entity
                    let mut updated_entity = current.clone();
                    updated_entity.word = format!("{} {}", updated_entity.word, entity.word);
                    updated_entity.end = entity.end;
                    updated_entity.score = (updated_entity.score + entity.score) / 2.0;
                    current_entity = Some(updated_entity);
                } else {
                    // Start new entity
                    merged_entities.push(current.clone());
                    current_entity = Some(entity);
                }
            }
        }
    }

    // Don't forget the last entity
    if let Some(entity) = current_entity {
        merged_entities.push(entity);
    }

    // Sort entities by position
    merged_entities.sort_by_key(|e| e.start);

    Ok(Json(TokenClassificationResponse {
        entities: merged_entities,
    }))
}

pub async fn batch_inference(
    State(state): State<AppState>,
    Json(request): Json<BatchRequest>,
) -> AppResult<Json<BatchResponse>> {
    let start_time = std::time::Instant::now();

    // Process batch based on task type
    let results = match request.task.as_str() {
        "classification" => {
            // Process classification batch
            vec![serde_json::json!({"label": "POSITIVE", "score": 0.95})]
        }
        "generation" => {
            // Process generation batch
            vec![serde_json::json!({"generated_text": "Sample generated text"})]
        }
        _ => return Err(AppError::BadRequest(format!("Unknown task: {}", request.task))),
    };

    let total_time_ms = start_time.elapsed().as_millis() as u64;

    Ok(Json(BatchResponse {
        results,
        total_time_ms,
    }))
}