//! Speculative Decoding for Accelerated Inference
//!
//! This module implements speculative decoding techniques for 2024-2025:
//! - Draft-and-verify paradigm for faster generation
//! - Multiple draft models for different use cases
//! - Adaptive speculation depth based on acceptance rates
//! - Tree-based speculation for parallel verification
//! - Dynamic draft model selection

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{Result as TrustformersResult, TrustformersError};
use crate::pipeline::{Pipeline, PipelineInput, PipelineOutput};

/// Configuration for Speculative Decoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeDecodingConfig {
    /// Number of tokens to speculate ahead
    pub speculation_depth: usize,
    /// Minimum acceptance rate to continue speculation
    pub min_acceptance_rate: f32,
    /// Whether to use tree-based speculation
    pub tree_based_speculation: bool,
    /// Number of parallel speculation branches
    pub num_branches: usize,
    /// Whether to adapt speculation depth dynamically
    pub adaptive_depth: bool,
    /// Maximum speculation depth when adaptive
    pub max_speculation_depth: usize,
    /// Temperature for draft model sampling
    pub draft_temperature: f32,
    /// Temperature for target model verification
    pub target_temperature: f32,
    /// Strategy for draft model selection
    pub draft_selection_strategy: DraftSelectionStrategy,
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            speculation_depth: 4,
            min_acceptance_rate: 0.6,
            tree_based_speculation: false,
            num_branches: 3,
            adaptive_depth: true,
            max_speculation_depth: 8,
            draft_temperature: 1.0,
            target_temperature: 0.7,
            draft_selection_strategy: DraftSelectionStrategy::BestPerformance,
        }
    }
}

/// Strategy for selecting draft models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DraftSelectionStrategy {
    /// Use fastest draft model
    Fastest,
    /// Use most accurate draft model
    MostAccurate,
    /// Use best overall performance (speed + accuracy)
    BestPerformance,
    /// Adapt selection based on input characteristics
    InputAdaptive,
    /// Round-robin selection for load balancing
    RoundRobin,
}

/// Draft model characteristics
#[derive(Debug, Clone)]
pub struct DraftModelProfile {
    pub model_id: String,
    pub speed_score: f32,             // Tokens per second
    pub accuracy_score: f32,          // Historical acceptance rate
    pub memory_usage: usize,          // Memory usage in MB
    pub specialization: Vec<String>,  // Domain specializations
    pub recent_performance: Vec<f32>, // Recent acceptance rates
}

/// Speculation tree node for tree-based speculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationNode {
    pub token_id: u32,
    pub probability: f32,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub depth: usize,
    pub cumulative_probability: f32,
}

/// Tree-based speculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationTree {
    pub nodes: Vec<SpeculationNode>,
    pub root_index: usize,
    pub max_depth: usize,
    pub total_paths: usize,
}

/// Verification result for a single speculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub draft_tokens: Vec<u32>,
    pub accepted_tokens: Vec<u32>,
    pub rejection_index: Option<usize>,
    pub acceptance_rate: f32,
    pub verification_time_ms: u64,
    pub target_probabilities: Vec<f32>,
    pub draft_probabilities: Vec<f32>,
}

/// Complete speculative decoding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeDecodingResult {
    pub generated_tokens: Vec<u32>,
    pub generated_text: String,
    pub total_verifications: usize,
    pub total_accepted_tokens: usize,
    pub total_drafted_tokens: usize,
    pub overall_acceptance_rate: f32,
    pub speed_improvement: f32,
    pub verification_results: Vec<VerificationResult>,
    pub draft_model_used: String,
    pub speculation_tree: Option<SpeculationTree>,
}

/// Performance metrics for draft model evaluation
#[derive(Debug, Clone)]
pub struct DraftModelMetrics {
    pub model_id: String,
    pub total_tokens_drafted: usize,
    pub total_tokens_accepted: usize,
    pub average_latency_ms: f32,
    pub acceptance_rate_history: Vec<f32>,
    pub last_updated: std::time::SystemTime,
}

/// Trait for draft models
#[async_trait::async_trait]
pub trait DraftModel: Send + Sync {
    async fn generate_draft(
        &self,
        input_tokens: &[u32],
        num_tokens: usize,
        temperature: f32,
    ) -> TrustformersResult<Vec<(u32, f32)>>; // (token_id, probability)

    async fn generate_tree_draft(
        &self,
        input_tokens: &[u32],
        depth: usize,
        branches: usize,
        temperature: f32,
    ) -> TrustformersResult<SpeculationTree>;

    fn get_model_profile(&self) -> &DraftModelProfile;
}

/// Trait for target model verification
#[async_trait::async_trait]
pub trait TargetModel: Send + Sync {
    async fn verify_tokens(
        &self,
        input_tokens: &[u32],
        draft_tokens: &[u32],
        temperature: f32,
    ) -> TrustformersResult<VerificationResult>;

    async fn verify_tree(
        &self,
        input_tokens: &[u32],
        speculation_tree: &SpeculationTree,
        temperature: f32,
    ) -> TrustformersResult<Vec<VerificationResult>>;
}

/// Speculative Decoding Pipeline
pub struct SpeculativeDecodingPipeline {
    config: SpeculativeDecodingConfig,
    draft_models: Vec<Arc<dyn DraftModel>>,
    target_model: Arc<dyn TargetModel>,
    draft_metrics: Arc<RwLock<HashMap<String, DraftModelMetrics>>>,
    current_draft_index: Arc<RwLock<usize>>,
    tokenizer: Option<Arc<dyn crate::core::traits::Tokenizer>>,
}

impl SpeculativeDecodingPipeline {
    /// Create a new Speculative Decoding Pipeline
    pub fn new(
        config: SpeculativeDecodingConfig,
        draft_models: Vec<Arc<dyn DraftModel>>,
        target_model: Arc<dyn TargetModel>,
    ) -> Self {
        let draft_metrics = Arc::new(RwLock::new(HashMap::new()));

        // Initialize metrics for each draft model
        let mut metrics_map = HashMap::new();
        for draft_model in &draft_models {
            let profile = draft_model.get_model_profile();
            metrics_map.insert(
                profile.model_id.clone(),
                DraftModelMetrics {
                    model_id: profile.model_id.clone(),
                    total_tokens_drafted: 0,
                    total_tokens_accepted: 0,
                    average_latency_ms: 0.0,
                    acceptance_rate_history: Vec::new(),
                    last_updated: std::time::SystemTime::now(),
                },
            );
        }

        Self {
            config,
            draft_models,
            target_model,
            draft_metrics,
            current_draft_index: Arc::new(RwLock::new(0)),
            tokenizer: None,
        }
    }

    /// Set tokenizer for text processing
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn crate::core::traits::Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Perform speculative decoding
    async fn speculative_decode(
        &self,
        input_tokens: &[u32],
        max_new_tokens: usize,
    ) -> TrustformersResult<SpeculativeDecodingResult> {
        let mut generated_tokens = Vec::new();
        let mut verification_results = Vec::new();
        let mut total_drafted = 0;
        let mut total_accepted = 0;
        let mut current_tokens = input_tokens.to_vec();

        let start_time = std::time::Instant::now();

        while generated_tokens.len() < max_new_tokens {
            // Select draft model
            let draft_model = self.select_draft_model(&current_tokens).await?;
            let draft_model_id = draft_model.get_model_profile().model_id.clone();

            // Determine speculation depth
            let speculation_depth = self.determine_speculation_depth(&draft_model_id).await?;

            if speculation_depth == 0 {
                break; // No speculation beneficial
            }

            // Generate draft tokens
            let verification_result = if self.config.tree_based_speculation {
                self.speculative_decode_tree(&current_tokens, &*draft_model, speculation_depth)
                    .await?
            } else {
                self.speculative_decode_linear(&current_tokens, &*draft_model, speculation_depth)
                    .await?
            };

            // Update metrics
            self.update_draft_metrics(&draft_model_id, &verification_result).await?;

            // Add accepted tokens
            generated_tokens.extend_from_slice(&verification_result.accepted_tokens);
            current_tokens.extend_from_slice(&verification_result.accepted_tokens);

            total_drafted += verification_result.draft_tokens.len();
            total_accepted += verification_result.accepted_tokens.len();

            verification_results.push(verification_result);

            // Check if we should adjust speculation depth
            if self.config.adaptive_depth {
                self.adjust_speculation_depth(&draft_model_id).await?;
            }
        }

        let total_time = start_time.elapsed();
        let baseline_time = self.estimate_baseline_time(max_new_tokens).await?;
        let speed_improvement = baseline_time.as_millis() as f32 / total_time.as_millis() as f32;

        let generated_text = if let Some(tokenizer) = &self.tokenizer {
            tokenizer.decode(&generated_tokens)?
        } else {
            format!("Generated {} tokens", generated_tokens.len())
        };

        Ok(SpeculativeDecodingResult {
            generated_tokens,
            generated_text,
            total_verifications: verification_results.len(),
            total_accepted_tokens: total_accepted,
            total_drafted_tokens: total_drafted,
            overall_acceptance_rate: if total_drafted > 0 {
                total_accepted as f32 / total_drafted as f32
            } else {
                0.0
            },
            speed_improvement,
            verification_results,
            draft_model_used: self.get_current_draft_model_id().await?,
            speculation_tree: None, // Would be populated for tree-based speculation
        })
    }

    /// Linear speculative decoding
    async fn speculative_decode_linear(
        &self,
        input_tokens: &[u32],
        draft_model: &dyn DraftModel,
        speculation_depth: usize,
    ) -> TrustformersResult<VerificationResult> {
        // Generate draft tokens
        let draft_tokens_with_probs = draft_model
            .generate_draft(
                input_tokens,
                speculation_depth,
                self.config.draft_temperature,
            )
            .await?;

        let draft_tokens: Vec<u32> =
            draft_tokens_with_probs.iter().map(|(token, _)| *token).collect();
        let draft_probabilities: Vec<f32> =
            draft_tokens_with_probs.iter().map(|(_, prob)| *prob).collect();

        // Verify with target model
        let mut verification_result = self
            .target_model
            .verify_tokens(input_tokens, &draft_tokens, self.config.target_temperature)
            .await?;

        verification_result.draft_probabilities = draft_probabilities;

        Ok(verification_result)
    }

    /// Tree-based speculative decoding
    async fn speculative_decode_tree(
        &self,
        input_tokens: &[u32],
        draft_model: &dyn DraftModel,
        speculation_depth: usize,
    ) -> TrustformersResult<VerificationResult> {
        // Generate speculation tree
        let speculation_tree = draft_model
            .generate_tree_draft(
                input_tokens,
                speculation_depth,
                self.config.num_branches,
                self.config.draft_temperature,
            )
            .await?;

        // Verify tree with target model
        let tree_verification_results = self
            .target_model
            .verify_tree(
                input_tokens,
                &speculation_tree,
                self.config.target_temperature,
            )
            .await?;

        // Select best path from verification results
        let best_result = tree_verification_results
            .into_iter()
            .max_by(|a, b| a.accepted_tokens.len().cmp(&b.accepted_tokens.len()))
            .unwrap_or_else(|| VerificationResult {
                draft_tokens: Vec::new(),
                accepted_tokens: Vec::new(),
                rejection_index: Some(0),
                acceptance_rate: 0.0,
                verification_time_ms: 0,
                target_probabilities: Vec::new(),
                draft_probabilities: Vec::new(),
            });

        Ok(best_result)
    }

    /// Select draft model based on strategy
    async fn select_draft_model(
        &self,
        _input_tokens: &[u32],
    ) -> TrustformersResult<Arc<dyn DraftModel>> {
        let index = match self.config.draft_selection_strategy {
            DraftSelectionStrategy::Fastest => self.select_fastest_model().await?,
            DraftSelectionStrategy::MostAccurate => self.select_most_accurate_model().await?,
            DraftSelectionStrategy::BestPerformance => self.select_best_performance_model().await?,
            DraftSelectionStrategy::InputAdaptive => {
                // Could analyze input characteristics here
                self.select_best_performance_model().await?
            },
            DraftSelectionStrategy::RoundRobin => {
                let mut current_index = self.current_draft_index.write().await;
                let index = *current_index;
                *current_index = (index + 1) % self.draft_models.len();
                index
            },
        };

        Ok(self.draft_models[index].clone())
    }

    /// Select fastest draft model
    async fn select_fastest_model(&self) -> TrustformersResult<usize> {
        let mut best_index = 0;
        let mut best_speed = 0.0;

        for (i, model) in self.draft_models.iter().enumerate() {
            let speed = model.get_model_profile().speed_score;
            if speed > best_speed {
                best_speed = speed;
                best_index = i;
            }
        }

        Ok(best_index)
    }

    /// Select most accurate draft model
    async fn select_most_accurate_model(&self) -> TrustformersResult<usize> {
        let metrics = self.draft_metrics.read().await;
        let mut best_index = 0;
        let mut best_accuracy = 0.0;

        for (i, model) in self.draft_models.iter().enumerate() {
            let model_id = &model.get_model_profile().model_id;
            let accuracy = if let Some(model_metrics) = metrics.get(model_id) {
                if !model_metrics.acceptance_rate_history.is_empty() {
                    model_metrics.acceptance_rate_history.iter().sum::<f32>()
                        / model_metrics.acceptance_rate_history.len() as f32
                } else {
                    model.get_model_profile().accuracy_score
                }
            } else {
                model.get_model_profile().accuracy_score
            };

            if accuracy > best_accuracy {
                best_accuracy = accuracy;
                best_index = i;
            }
        }

        Ok(best_index)
    }

    /// Select model with best overall performance
    async fn select_best_performance_model(&self) -> TrustformersResult<usize> {
        let metrics = self.draft_metrics.read().await;
        let mut best_index = 0;
        let mut best_score = 0.0;

        for (i, model) in self.draft_models.iter().enumerate() {
            let profile = model.get_model_profile();
            let model_id = &profile.model_id;

            let accuracy = if let Some(model_metrics) = metrics.get(model_id) {
                if !model_metrics.acceptance_rate_history.is_empty() {
                    model_metrics.acceptance_rate_history.iter().sum::<f32>()
                        / model_metrics.acceptance_rate_history.len() as f32
                } else {
                    profile.accuracy_score
                }
            } else {
                profile.accuracy_score
            };

            // Combined score: speed * accuracy
            let score = profile.speed_score * accuracy;

            if score > best_score {
                best_score = score;
                best_index = i;
            }
        }

        Ok(best_index)
    }

    /// Determine optimal speculation depth
    async fn determine_speculation_depth(&self, model_id: &str) -> TrustformersResult<usize> {
        if !self.config.adaptive_depth {
            return Ok(self.config.speculation_depth);
        }

        let metrics = self.draft_metrics.read().await;
        if let Some(model_metrics) = metrics.get(model_id) {
            if model_metrics.acceptance_rate_history.len() >= 5 {
                let recent_rate =
                    model_metrics.acceptance_rate_history.iter().rev().take(5).sum::<f32>() / 5.0;

                if recent_rate > 0.8 {
                    Ok(self.config.max_speculation_depth)
                } else if recent_rate > 0.6 {
                    Ok(self.config.speculation_depth)
                } else if recent_rate > 0.4 {
                    Ok(self.config.speculation_depth / 2)
                } else {
                    Ok(1) // Minimal speculation
                }
            } else {
                Ok(self.config.speculation_depth)
            }
        } else {
            Ok(self.config.speculation_depth)
        }
    }

    /// Update draft model metrics
    async fn update_draft_metrics(
        &self,
        model_id: &str,
        verification_result: &VerificationResult,
    ) -> TrustformersResult<()> {
        let mut metrics = self.draft_metrics.write().await;

        if let Some(model_metrics) = metrics.get_mut(model_id) {
            model_metrics.total_tokens_drafted += verification_result.draft_tokens.len();
            model_metrics.total_tokens_accepted += verification_result.accepted_tokens.len();
            model_metrics.acceptance_rate_history.push(verification_result.acceptance_rate);

            // Keep only recent history
            if model_metrics.acceptance_rate_history.len() > 100 {
                model_metrics.acceptance_rate_history.remove(0);
            }

            model_metrics.last_updated = std::time::SystemTime::now();
        }

        Ok(())
    }

    /// Adjust speculation depth based on performance
    async fn adjust_speculation_depth(&self, _model_id: &str) -> TrustformersResult<()> {
        // Implementation would adjust speculation depth based on recent performance
        // For now, this is a placeholder
        Ok(())
    }

    /// Get current draft model ID
    async fn get_current_draft_model_id(&self) -> TrustformersResult<String> {
        let index = *self.current_draft_index.read().await;
        if index < self.draft_models.len() {
            Ok(self.draft_models[index].get_model_profile().model_id.clone())
        } else {
            Ok("unknown".to_string())
        }
    }

    /// Estimate baseline generation time without speculation
    async fn estimate_baseline_time(
        &self,
        num_tokens: usize,
    ) -> TrustformersResult<std::time::Duration> {
        // Mock estimation - in practice would benchmark target model
        let tokens_per_second = 10.0; // Conservative estimate
        let seconds = num_tokens as f32 / tokens_per_second;
        Ok(std::time::Duration::from_secs_f32(seconds))
    }
}

impl Pipeline for SpeculativeDecodingPipeline {
    type Input = PipelineInput;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> TrustformersResult<Self::Output> {
        let (input_tokens, max_new_tokens) = match input {
            PipelineInput::Text(text) => {
                let tokens = if let Some(tokenizer) = &self.tokenizer {
                    tokenizer
                        .encode(&text)
                        .map_err(|e| {
                            TrustformersError::runtime_error(format!("Tokenization failed: {}", e))
                        })?
                        .input_ids
                } else {
                    // Mock tokenization
                    text.split_whitespace().enumerate().map(|(i, _)| i as u32).collect()
                };
                (tokens, 50) // Default max tokens
            },
            PipelineInput::Tokens(tokens) => (tokens, 50),
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Speculative decoding requires text or token input".to_string(),
                ))
            },
        };

        // Use current runtime handle to avoid creating nested runtimes
        let result = if let Ok(handle) = tokio::runtime::Handle::try_current() {
            tokio::task::block_in_place(|| {
                handle.block_on(self.speculative_decode(&input_tokens, max_new_tokens))
            })
        } else {
            // Fallback for non-async contexts
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                TrustformersError::runtime_error(format!("Failed to create async runtime: {}", e))
            })?;
            rt.block_on(self.speculative_decode(&input_tokens, max_new_tokens))
        }
        .map_err(|e| {
            TrustformersError::runtime_error(format!("Speculative decoding failed: {}", e))
        })?;

        Ok(PipelineOutput::SpeculativeDecoding(result))
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl crate::pipeline::AsyncPipeline for SpeculativeDecodingPipeline {
    type Input = PipelineInput;
    type Output = PipelineOutput;

    async fn __call_async__(&self, input: Self::Input) -> TrustformersResult<Self::Output> {
        let (input_tokens, max_new_tokens) = match input {
            PipelineInput::Text(text) => {
                let tokens = if let Some(tokenizer) = &self.tokenizer {
                    tokenizer.encode(&text)?.input_ids
                } else {
                    // Mock tokenization
                    text.split_whitespace().enumerate().map(|(i, _)| i as u32).collect()
                };
                (tokens, 50) // Default max tokens
            },
            PipelineInput::Tokens(tokens) => (tokens, 50),
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Speculative decoding requires text or token input".to_string(),
                ))
            },
        };

        let result = self.speculative_decode(&input_tokens, max_new_tokens).await.map_err(|e| {
            TrustformersError::invalid_input(
                format!("Speculative decoding failed: {}", e),
                Some("input_tokens"),
                Some("valid tokenized input for speculative decoding"),
                None::<String>,
            )
        })?;
        Ok(PipelineOutput::SpeculativeDecoding(result))
    }
}

/// Mock implementations for testing and demonstration

/// Mock draft model
pub struct MockDraftModel {
    profile: DraftModelProfile,
}

impl MockDraftModel {
    pub fn new(model_id: String, speed: f32, accuracy: f32) -> Self {
        Self {
            profile: DraftModelProfile {
                model_id,
                speed_score: speed,
                accuracy_score: accuracy,
                memory_usage: 100,
                specialization: vec!["general".to_string()],
                recent_performance: Vec::new(),
            },
        }
    }
}

#[async_trait::async_trait]
impl DraftModel for MockDraftModel {
    async fn generate_draft(
        &self,
        _input_tokens: &[u32],
        num_tokens: usize,
        _temperature: f32,
    ) -> TrustformersResult<Vec<(u32, f32)>> {
        let draft_tokens =
            (0..num_tokens).map(|i| (1000 + i as u32, 0.8 - i as f32 * 0.1)).collect();
        Ok(draft_tokens)
    }

    async fn generate_tree_draft(
        &self,
        _input_tokens: &[u32],
        depth: usize,
        branches: usize,
        _temperature: f32,
    ) -> TrustformersResult<SpeculationTree> {
        let mut nodes = Vec::new();
        let mut node_id = 0;

        // Create root
        nodes.push(SpeculationNode {
            token_id: 0,
            probability: 1.0,
            parent: None,
            children: (1..=branches).collect(),
            depth: 0,
            cumulative_probability: 1.0,
        });
        node_id += 1;

        // Create tree structure
        for d in 1..=depth {
            for b in 0..branches {
                let parent_id = if d == 1 { 0 } else { (d - 2) * branches + b + 1 };
                nodes.push(SpeculationNode {
                    token_id: 1000 + (d * branches + b) as u32,
                    probability: 0.8 / branches as f32,
                    parent: Some(parent_id),
                    children: Vec::new(),
                    depth: d,
                    cumulative_probability: 0.8_f32.powi(d as i32) / branches as f32,
                });
                node_id += 1;
            }
        }

        Ok(SpeculationTree {
            nodes,
            root_index: 0,
            max_depth: depth,
            total_paths: branches.pow(depth as u32),
        })
    }

    fn get_model_profile(&self) -> &DraftModelProfile {
        &self.profile
    }
}

/// Mock target model
pub struct MockTargetModel;

#[async_trait::async_trait]
impl TargetModel for MockTargetModel {
    async fn verify_tokens(
        &self,
        _input_tokens: &[u32],
        draft_tokens: &[u32],
        _temperature: f32,
    ) -> TrustformersResult<VerificationResult> {
        // Mock verification - accept some tokens with decreasing probability
        let mut accepted_tokens = Vec::new();
        let mut rejection_index = None;

        for (i, &token) in draft_tokens.iter().enumerate() {
            let acceptance_prob = 0.9 - i as f32 * 0.2; // Decreasing acceptance
            if acceptance_prob > 0.5 {
                accepted_tokens.push(token);
            } else {
                rejection_index = Some(i);
                break;
            }
        }

        let acceptance_rate = accepted_tokens.len() as f32 / draft_tokens.len() as f32;

        Ok(VerificationResult {
            draft_tokens: draft_tokens.to_vec(),
            accepted_tokens,
            rejection_index,
            acceptance_rate,
            verification_time_ms: 10,
            target_probabilities: vec![0.8; draft_tokens.len()],
            draft_probabilities: vec![0.7; draft_tokens.len()],
        })
    }

    async fn verify_tree(
        &self,
        input_tokens: &[u32],
        speculation_tree: &SpeculationTree,
        temperature: f32,
    ) -> TrustformersResult<Vec<VerificationResult>> {
        // Mock tree verification - generate results for different paths
        let mut results = Vec::new();

        // For simplicity, verify a few representative paths
        for i in 0..3.min(speculation_tree.total_paths) {
            let draft_tokens = vec![1000 + i as u32, 1001 + i as u32];
            let result = self.verify_tokens(input_tokens, &draft_tokens, temperature).await?;
            results.push(result);
        }

        Ok(results)
    }
}

/// Factory functions for creating speculative decoding pipelines

/// Create a basic speculative decoding pipeline
pub fn create_speculative_decoding_pipeline(
    config: SpeculativeDecodingConfig,
) -> SpeculativeDecodingPipeline {
    let draft_models: Vec<Arc<dyn DraftModel>> = vec![
        Arc::new(MockDraftModel::new("fast_draft".to_string(), 100.0, 0.7)),
        Arc::new(MockDraftModel::new("accurate_draft".to_string(), 50.0, 0.9)),
    ];

    let target_model = Arc::new(MockTargetModel);

    SpeculativeDecodingPipeline::new(config, draft_models, target_model)
}

/// Create an efficiency-optimized speculative decoding pipeline
pub fn create_efficiency_optimized_speculative_pipeline() -> SpeculativeDecodingPipeline {
    let config = SpeculativeDecodingConfig {
        speculation_depth: 6,
        adaptive_depth: true,
        draft_selection_strategy: DraftSelectionStrategy::BestPerformance,
        tree_based_speculation: false,
        min_acceptance_rate: 0.7,
        ..Default::default()
    };

    create_speculative_decoding_pipeline(config)
}

/// Create a tree-based speculative decoding pipeline
pub fn create_tree_based_speculative_pipeline() -> SpeculativeDecodingPipeline {
    let config = SpeculativeDecodingConfig {
        tree_based_speculation: true,
        num_branches: 4,
        speculation_depth: 3,
        draft_selection_strategy: DraftSelectionStrategy::MostAccurate,
        ..Default::default()
    };

    create_speculative_decoding_pipeline(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_speculative_decoding_pipeline() {
        let config = SpeculativeDecodingConfig::default();
        let pipeline = create_speculative_decoding_pipeline(config);

        let input = PipelineInput::Text("Generate some text with speculative decoding".to_string());
        let result = pipeline.__call__(input);

        assert!(result.is_ok());
        if let Ok(PipelineOutput::SpeculativeDecoding(spec_result)) = result {
            assert!(!spec_result.generated_tokens.is_empty());
            assert!(spec_result.speed_improvement > 0.0);
            assert!(!spec_result.verification_results.is_empty());
        }
    }

    #[tokio::test]
    async fn test_draft_model_selection() {
        let config = SpeculativeDecodingConfig {
            draft_selection_strategy: DraftSelectionStrategy::Fastest,
            ..Default::default()
        };

        let pipeline = create_speculative_decoding_pipeline(config);
        let input_tokens = vec![1, 2, 3, 4, 5];

        let selected_model = pipeline.select_draft_model(&input_tokens).await.unwrap();
        assert_eq!(selected_model.get_model_profile().model_id, "fast_draft");
    }

    #[tokio::test]
    async fn test_adaptive_speculation_depth() {
        let config = SpeculativeDecodingConfig {
            adaptive_depth: true,
            max_speculation_depth: 8,
            ..Default::default()
        };

        let pipeline = create_speculative_decoding_pipeline(config);

        // Test with high acceptance rate
        let depth = pipeline.determine_speculation_depth("test_model").await.unwrap();
        assert!(depth > 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_tree_based_speculation() {
        let pipeline = create_tree_based_speculative_pipeline();

        let input = PipelineInput::Text("Test tree-based speculation".to_string());
        let result = pipeline.__call__(input);

        assert!(result.is_ok());
        if let Ok(PipelineOutput::SpeculativeDecoding(spec_result)) = result {
            assert!(!spec_result.verification_results.is_empty());
        }
    }

    #[tokio::test]
    async fn test_mock_draft_model() {
        let draft_model = MockDraftModel::new("test".to_string(), 50.0, 0.8);

        let draft_tokens = draft_model.generate_draft(&[1, 2, 3], 3, 0.8).await.unwrap();
        assert_eq!(draft_tokens.len(), 3);

        let tree = draft_model.generate_tree_draft(&[1, 2, 3], 2, 3, 0.8).await.unwrap();
        assert_eq!(tree.max_depth, 2);
        assert!(!tree.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_mock_target_model() {
        let target_model = MockTargetModel;

        let draft_tokens = vec![1000, 1001, 1002];
        let result = target_model.verify_tokens(&[1, 2, 3], &draft_tokens, 0.8).await.unwrap();

        assert!(!result.accepted_tokens.is_empty());
        assert!(result.acceptance_rate > 0.0);
    }
}
