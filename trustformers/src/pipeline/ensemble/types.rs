//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::pipeline::{Pipeline, PipelineOptions, PipelineOutput};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionInfo {
    pub selected_models: Vec<String>,
    pub selection_reason: String,
    pub selection_confidence: f32,
    pub alternative_models: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeResult {
    pub predictions: Vec<PipelineOutput>,
    pub exit_at_model: usize,
    pub cumulative_confidence: f32,
    pub processing_times: Vec<u64>,
    pub resource_usage: Vec<u64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSelectionStrategy {
    /// Use all models in ensemble
    All,
    /// Select top-k performing models
    TopK(usize),
    /// Select models based on confidence threshold
    ConfidenceBased,
    /// Select models based on resource constraints
    ResourceConstrained,
    /// Dynamic selection based on input characteristics
    Dynamic,
    /// Random selection for diversity
    Random(usize),
    /// Select models based on prediction agreement
    AgreementBased,
}
/// Bootstrap resampling statistics produced by the Bagging strategy
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapStats {
    pub mean: f64,
    pub variance: f64,
    pub n_samples: usize,
}
/// Deterministic FNV-1a hash routing gate.
/// Picks a primary expert and decays scores by distance using a temperature parameter.
pub struct HashRoutingGate {
    pub temperature: f32,
}
impl HashRoutingGate {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}
/// Keyword-based router that replicates the original `select_models_for_input` heuristics.
pub struct KeywordRouter;
impl KeywordRouter {
    pub(super) fn detect_domain(input: &str) -> Option<String> {
        let input_lower = input.to_lowercase();
        if input_lower.contains("medical")
            || input_lower.contains("patient")
            || input_lower.contains("diagnosis")
        {
            Some("medical".to_string())
        } else if input_lower.contains("legal")
            || input_lower.contains("contract")
            || input_lower.contains("law")
        {
            Some("legal".to_string())
        } else if input_lower.contains("science")
            || input_lower.contains("research")
            || input_lower.contains("experiment")
        {
            Some("scientific".to_string())
        } else if input_lower.contains("code")
            || input_lower.contains("programming")
            || input_lower.contains("function")
        {
            Some("technical".to_string())
        } else {
            None
        }
    }
    pub(super) fn detect_language(input: &str) -> Option<String> {
        let has_chinese = input.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
        let has_arabic = input.chars().any(|c| ('\u{0600}'..='\u{06ff}').contains(&c));
        let has_cyrillic = input.chars().any(|c| ('\u{0400}'..='\u{04ff}').contains(&c));
        if has_chinese {
            Some("zh".to_string())
        } else if has_arabic {
            Some("ar".to_string())
        } else if has_cyrillic {
            Some("ru".to_string())
        } else {
            Some("en".to_string())
        }
    }
}
/// Cosine-similarity embedding router.
pub struct EmbeddingCosineRouter {
    pub(super) model_embeddings: Vec<(String, Vec<f32>)>,
    pub(super) input_embedding_fn: Arc<dyn Fn(&str) -> Vec<f32> + Send + Sync>,
}
impl EmbeddingCosineRouter {
    pub fn new(
        model_embeddings: Vec<(String, Vec<f32>)>,
        input_embedding_fn: Arc<dyn Fn(&str) -> Vec<f32> + Send + Sync>,
    ) -> Self {
        Self {
            model_embeddings,
            input_embedding_fn,
        }
    }
    pub(super) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeight {
    pub model_id: String,
    pub weight: f32,
    pub confidence_weight: f32,
    pub accuracy_weight: f32,
    pub dynamic_weight: f32,
}
impl ModelWeight {
    pub fn new(model_id: String, weight: f32) -> Self {
        Self {
            model_id,
            weight,
            confidence_weight: 1.0,
            accuracy_weight: 1.0,
            dynamic_weight: 1.0,
        }
    }
    pub fn total_weight(&self) -> f32 {
        self.weight * self.confidence_weight * self.accuracy_weight * self.dynamic_weight
    }
}
