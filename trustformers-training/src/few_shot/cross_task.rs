use anyhow::Result;
use scirs2_core::ndarray::{s, Array1, Array2}; // SciRS2 Integration Policy
use scirs2_core::random::*; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for cross-task generalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralizationConfig {
    /// Embedding dimension for task representations
    pub task_embedding_dim: usize,
    /// Number of shared layers
    pub num_shared_layers: usize,
    /// Number of task-specific layers
    pub num_task_specific_layers: usize,
    /// Method for computing task similarity
    pub similarity_method: SimilarityMethod,
    /// Threshold for task similarity
    pub similarity_threshold: f32,
    /// Strategy for knowledge transfer
    pub transfer_strategy: TransferStrategy,
    /// Regularization weight for cross-task consistency
    pub consistency_weight: f32,
    /// Whether to use attention-based task mixing
    pub use_attention_mixing: bool,
    /// Maximum number of source tasks to consider
    pub max_source_tasks: usize,
}

impl Default for GeneralizationConfig {
    fn default() -> Self {
        Self {
            task_embedding_dim: 128,
            num_shared_layers: 8,
            num_task_specific_layers: 2,
            similarity_method: SimilarityMethod::Cosine,
            similarity_threshold: 0.7,
            transfer_strategy: TransferStrategy::WeightedCombination,
            consistency_weight: 0.1,
            use_attention_mixing: true,
            max_source_tasks: 5,
        }
    }
}

/// Methods for computing task similarity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimilarityMethod {
    /// Cosine similarity between task embeddings
    Cosine,
    /// Euclidean distance (converted to similarity)
    Euclidean,
    /// Learned similarity function
    Learned,
    /// Task metadata-based similarity
    Metadata,
    /// Combined similarity metrics
    Combined(Vec<SimilarityMethod>),
}

/// Strategies for transferring knowledge between tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferStrategy {
    /// Weighted combination based on similarity
    WeightedCombination,
    /// Direct parameter transfer
    ParameterTransfer,
    /// Gradient-based transfer
    GradientTransfer,
    /// Attention-based knowledge distillation
    AttentionDistillation,
    /// Progressive transfer with fine-tuning
    ProgressiveTransfer,
}

/// Task embedding representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEmbedding {
    /// Dense representation of the task
    pub embedding: Array1<f32>,
    /// Task identifier
    pub task_id: String,
    /// Task metadata features
    pub metadata_features: HashMap<String, f32>,
    /// Performance statistics
    pub performance_stats: PerformanceStats,
    /// Domain information
    pub domain: String,
    /// Last update timestamp
    pub last_updated: std::time::SystemTime,
}

impl TaskEmbedding {
    pub fn new(task_id: String, embedding_dim: usize) -> Self {
        Self {
            embedding: Array1::zeros(embedding_dim),
            task_id,
            metadata_features: HashMap::new(),
            performance_stats: PerformanceStats::new(),
            domain: "unknown".to_string(),
            last_updated: std::time::SystemTime::now(),
        }
    }

    /// Update embedding with new data
    pub fn update_embedding(&mut self, new_embedding: Array1<f32>, learning_rate: f32) {
        self.embedding = (1.0 - learning_rate) * &self.embedding + learning_rate * &new_embedding;
        self.last_updated = std::time::SystemTime::now();
    }

    /// Compute similarity to another task
    pub fn similarity(&self, other: &TaskEmbedding, method: &SimilarityMethod) -> f32 {
        match method {
            SimilarityMethod::Cosine => self.cosine_similarity(other),
            SimilarityMethod::Euclidean => self.euclidean_similarity(other),
            SimilarityMethod::Learned => self.learned_similarity(other),
            SimilarityMethod::Metadata => self.metadata_similarity(other),
            SimilarityMethod::Combined(methods) => {
                let similarities: Vec<f32> =
                    methods.iter().map(|m| self.similarity(other, m)).collect();
                similarities.iter().sum::<f32>() / similarities.len() as f32
            },
        }
    }

    /// Cosine similarity between embeddings
    fn cosine_similarity(&self, other: &TaskEmbedding) -> f32 {
        let dot_product = self.embedding.dot(&other.embedding);
        let norm_self = self.embedding.dot(&self.embedding).sqrt();
        let norm_other = other.embedding.dot(&other.embedding).sqrt();
        dot_product / (norm_self * norm_other + 1e-8)
    }

    /// Euclidean similarity (converted from distance)
    fn euclidean_similarity(&self, other: &TaskEmbedding) -> f32 {
        let diff = &self.embedding - &other.embedding;
        let distance = diff.dot(&diff).sqrt();
        1.0 / (1.0 + distance)
    }

    /// Learned similarity (placeholder)
    fn learned_similarity(&self, _other: &TaskEmbedding) -> f32 {
        // In practice, would use a learned similarity function
        0.5
    }

    /// Metadata-based similarity
    fn metadata_similarity(&self, other: &TaskEmbedding) -> f32 {
        let mut common_features = 0;
        let mut total_features = 0;

        for (key, value) in &self.metadata_features {
            total_features += 1;
            if let Some(other_value) = other.metadata_features.get(key) {
                let feature_sim = 1.0 - (value - other_value).abs();
                if feature_sim > 0.8 {
                    common_features += 1;
                }
            }
        }

        if total_features == 0 {
            0.0
        } else {
            common_features as f32 / total_features as f32
        }
    }
}

/// Performance statistics for a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Best accuracy/performance achieved
    pub best_performance: f32,
    /// Number of training steps
    pub training_steps: usize,
    /// Convergence rate
    pub convergence_rate: f32,
    /// Sample efficiency (performance per sample)
    pub sample_efficiency: f32,
}

impl PerformanceStats {
    fn new() -> Self {
        Self {
            best_performance: 0.0,
            training_steps: 0,
            convergence_rate: 0.0,
            sample_efficiency: 0.0,
        }
    }

    /// Update performance statistics
    pub fn update(&mut self, performance: f32, steps: usize, num_samples: usize) {
        self.best_performance = self.best_performance.max(performance);
        self.training_steps = steps;
        self.convergence_rate = performance / (steps as f32 + 1.0);
        self.sample_efficiency = performance / (num_samples as f32 + 1.0);
    }
}

/// Knowledge representation for transfer
#[derive(Debug, Clone)]
pub struct TransferKnowledge {
    /// Source task embeddings
    pub source_embeddings: HashMap<String, TaskEmbedding>,
    /// Learned transformation matrices
    pub transformation_matrices: HashMap<String, Array2<f32>>,
    /// Transfer weights
    pub transfer_weights: HashMap<String, f32>,
    /// Shared representations
    pub shared_representations: Option<Array2<f32>>,
}

impl Default for TransferKnowledge {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferKnowledge {
    pub fn new() -> Self {
        Self {
            source_embeddings: HashMap::new(),
            transformation_matrices: HashMap::new(),
            transfer_weights: HashMap::new(),
            shared_representations: None,
        }
    }

    /// Add source task knowledge
    pub fn add_source_task(&mut self, task_embedding: TaskEmbedding, weight: f32) {
        let task_id = task_embedding.task_id.clone();
        self.transfer_weights.insert(task_id.clone(), weight);
        self.source_embeddings.insert(task_id, task_embedding);
    }

    /// Get relevant source tasks for a target task
    pub fn get_relevant_sources(
        &self,
        target_embedding: &TaskEmbedding,
        config: &GeneralizationConfig,
    ) -> Vec<(String, f32)> {
        let mut similarities: Vec<(String, f32)> = self
            .source_embeddings
            .iter()
            .map(|(task_id, source_embedding)| {
                let sim = target_embedding.similarity(source_embedding, &config.similarity_method);
                (task_id.clone(), sim)
            })
            .filter(|(_, sim)| *sim >= config.similarity_threshold)
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top-k sources
        similarities.into_iter().take(config.max_source_tasks).collect()
    }
}

/// Cross-task generalizer for managing knowledge transfer
pub struct CrossTaskGeneralizer {
    config: GeneralizationConfig,
    task_embeddings: HashMap<String, TaskEmbedding>,
    transfer_knowledge: TransferKnowledge,
    #[allow(dead_code)]
    attention_weights: Option<Array2<f32>>,
    shared_encoder: Option<SharedEncoder>,
}

impl CrossTaskGeneralizer {
    pub fn new(config: GeneralizationConfig) -> Self {
        let shared_encoder = Some(SharedEncoder::new(config.clone()));
        Self {
            config,
            task_embeddings: HashMap::new(),
            transfer_knowledge: TransferKnowledge::new(),
            attention_weights: None,
            shared_encoder,
        }
    }

    /// Register a new task
    pub fn register_task(&mut self, task_id: String, initial_data: &[Array1<f32>]) -> Result<()> {
        let mut task_embedding =
            TaskEmbedding::new(task_id.clone(), self.config.task_embedding_dim);

        // Compute initial embedding from data
        if !initial_data.is_empty() {
            let mean_features = self.compute_mean_features(initial_data);
            if let Some(shared_encoder) = &self.shared_encoder {
                let encoded = shared_encoder.encode(&mean_features)?;
                task_embedding.update_embedding(encoded, 1.0);
            }
        }

        self.task_embeddings.insert(task_id, task_embedding);
        Ok(())
    }

    /// Update task embedding based on new training data
    pub fn update_task_embedding(
        &mut self,
        task_id: &str,
        training_data: &[Array1<f32>],
        performance: f32,
        training_steps: usize,
    ) -> Result<()> {
        // Compute new embedding outside of borrow
        let new_embedding = if !training_data.is_empty() {
            let mean_features = self.compute_mean_features(training_data);
            if let Some(shared_encoder) = &self.shared_encoder {
                Some(shared_encoder.encode(&mean_features)?)
            } else {
                None
            }
        } else {
            None
        };

        // Now update the task embedding
        let task_embedding = self
            .task_embeddings
            .get_mut(task_id)
            .ok_or_else(|| anyhow::anyhow!("Task not found: {}", task_id))?;

        // Update performance stats
        task_embedding
            .performance_stats
            .update(performance, training_steps, training_data.len());

        // Apply new embedding if we computed one
        if let Some(embedding) = new_embedding {
            task_embedding.update_embedding(embedding, 0.1); // Learning rate 0.1
        }

        Ok(())
    }

    /// Transfer knowledge from source tasks to target task
    pub fn transfer_knowledge(
        &self,
        target_task_id: &str,
        target_data: &[Array1<f32>],
    ) -> Result<TransferredKnowledge> {
        let target_embedding = self
            .task_embeddings
            .get(target_task_id)
            .ok_or_else(|| anyhow::anyhow!("Target task not found: {}", target_task_id))?;

        // Find relevant source tasks
        let relevant_sources =
            self.transfer_knowledge.get_relevant_sources(target_embedding, &self.config);

        if relevant_sources.is_empty() {
            return Ok(TransferredKnowledge::empty());
        }

        match self.config.transfer_strategy {
            TransferStrategy::WeightedCombination => {
                self.weighted_combination_transfer(&relevant_sources, target_data)
            },
            TransferStrategy::ParameterTransfer => {
                self.parameter_transfer(&relevant_sources, target_data)
            },
            TransferStrategy::AttentionDistillation => {
                self.attention_distillation_transfer(&relevant_sources, target_data)
            },
            _ => {
                // Default to weighted combination
                self.weighted_combination_transfer(&relevant_sources, target_data)
            },
        }
    }

    /// Weighted combination transfer strategy
    fn weighted_combination_transfer(
        &self,
        relevant_sources: &[(String, f32)],
        _target_data: &[Array1<f32>],
    ) -> Result<TransferredKnowledge> {
        let mut combined_embedding = Array1::zeros(self.config.task_embedding_dim);
        let mut total_weight = 0.0;

        for (source_id, similarity) in relevant_sources {
            if let Some(source_embedding) = self.task_embeddings.get(source_id) {
                combined_embedding = combined_embedding + *similarity * &source_embedding.embedding;
                total_weight += similarity;
            }
        }

        if total_weight > 0.0 {
            combined_embedding /= total_weight;
        }

        Ok(TransferredKnowledge {
            transferred_embedding: Some(combined_embedding),
            source_weights: relevant_sources.iter().cloned().collect(),
            transfer_matrices: HashMap::new(),
            confidence: total_weight / relevant_sources.len() as f32,
        })
    }

    /// Parameter transfer strategy
    fn parameter_transfer(
        &self,
        relevant_sources: &[(String, f32)],
        _target_data: &[Array1<f32>],
    ) -> Result<TransferredKnowledge> {
        // In practice, would transfer actual model parameters
        // This is a simplified implementation
        let mut transfer_matrices = HashMap::new();

        for (source_id, _similarity) in relevant_sources {
            // Create identity matrix as placeholder
            let matrix = Array2::eye(self.config.task_embedding_dim);
            transfer_matrices.insert(source_id.clone(), matrix);
        }

        Ok(TransferredKnowledge {
            transferred_embedding: None,
            source_weights: relevant_sources.iter().cloned().collect(),
            transfer_matrices,
            confidence: 0.8,
        })
    }

    /// Attention-based distillation transfer
    fn attention_distillation_transfer(
        &self,
        relevant_sources: &[(String, f32)],
        target_data: &[Array1<f32>],
    ) -> Result<TransferredKnowledge> {
        if target_data.is_empty() {
            return self.weighted_combination_transfer(relevant_sources, target_data);
        }

        // Compute attention weights over source tasks
        let query = self.compute_mean_features(target_data);
        let mut attention_weights = Vec::new();
        let mut source_embeddings = Vec::new();

        for (source_id, _similarity) in relevant_sources {
            if let Some(source_embedding) = self.task_embeddings.get(source_id) {
                let attention = self.compute_attention(&query, &source_embedding.embedding);
                attention_weights.push(attention);
                source_embeddings.push(source_embedding.embedding.clone());
            }
        }

        // Normalize attention weights
        let total_attention: f32 = attention_weights.iter().sum();
        if total_attention > 0.0 {
            for weight in &mut attention_weights {
                *weight /= total_attention;
            }
        }

        // Compute attended combination
        let mut attended_embedding = Array1::zeros(self.config.task_embedding_dim);
        for (weight, embedding) in attention_weights.iter().zip(&source_embeddings) {
            attended_embedding = attended_embedding + (*weight * embedding);
        }

        Ok(TransferredKnowledge {
            transferred_embedding: Some(attended_embedding),
            source_weights: relevant_sources.iter().cloned().collect(),
            transfer_matrices: HashMap::new(),
            confidence: total_attention,
        })
    }

    /// Compute attention between query and key
    fn compute_attention(&self, query: &Array1<f32>, key: &Array1<f32>) -> f32 {
        let dot_product = query.dot(key);
        let norm_query = query.dot(query).sqrt();
        let norm_key = key.dot(key).sqrt();
        (dot_product / (norm_query * norm_key + 1e-8)).exp()
    }

    /// Compute mean features from data
    fn compute_mean_features(&self, data: &[Array1<f32>]) -> Array1<f32> {
        if data.is_empty() {
            return Array1::zeros(self.config.task_embedding_dim);
        }

        let mut mean = Array1::zeros(data[0].len());
        for sample in data {
            mean += sample;
        }
        mean / data.len() as f32
    }

    /// Get task embedding
    pub fn get_task_embedding(&self, task_id: &str) -> Option<&TaskEmbedding> {
        self.task_embeddings.get(task_id)
    }

    /// Get all task embeddings
    pub fn get_all_task_embeddings(&self) -> &HashMap<String, TaskEmbedding> {
        &self.task_embeddings
    }

    /// Compute task similarity matrix
    pub fn compute_task_similarity_matrix(&self) -> Array2<f32> {
        let task_ids: Vec<_> = self.task_embeddings.keys().cloned().collect();
        let n_tasks = task_ids.len();
        let mut similarity_matrix = Array2::zeros((n_tasks, n_tasks));

        for i in 0..n_tasks {
            for j in 0..n_tasks {
                if i == j {
                    similarity_matrix[[i, j]] = 1.0;
                } else {
                    let task1 = &self.task_embeddings[&task_ids[i]];
                    let task2 = &self.task_embeddings[&task_ids[j]];
                    let sim = task1.similarity(task2, &self.config.similarity_method);
                    similarity_matrix[[i, j]] = sim;
                }
            }
        }

        similarity_matrix
    }
}

/// Shared encoder for computing task embeddings
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SharedEncoder {
    /// Encoding layers
    layers: Vec<Array2<f32>>,
    /// Layer dimensions
    #[allow(dead_code)]
    layer_dims: Vec<usize>,
    config: GeneralizationConfig,
}

impl SharedEncoder {
    fn new(config: GeneralizationConfig) -> Self {
        let mut layers = Vec::new();
        let layer_dims = vec![config.task_embedding_dim; config.num_shared_layers + 1];

        // Initialize encoding layers
        for i in 0..config.num_shared_layers {
            let input_dim = layer_dims[i];
            let output_dim = layer_dims[i + 1];

            // Xavier initialization
            let bound = (6.0 / (input_dim + output_dim) as f32).sqrt();
            let mut rng = thread_rng();
            let uniform = Uniform::new(-bound, bound).expect("Invalid uniform distribution bounds");
            let layer =
                Array2::from_shape_fn((input_dim, output_dim), |_| uniform.sample(&mut rng));
            layers.push(layer);
        }

        Self {
            layers,
            layer_dims,
            config,
        }
    }

    /// Encode input to task embedding
    fn encode(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let mut current = input.clone();

        for layer in &self.layers {
            if current.len() != layer.nrows() {
                // Adjust input size if needed
                if current.len() < layer.nrows() {
                    let mut padded = Array1::zeros(layer.nrows());
                    padded.slice_mut(s![..current.len()]).assign(&current);
                    current = padded;
                } else {
                    current = current.slice(s![..layer.nrows()]).to_owned();
                }
            }

            current = current.dot(layer);
            // Apply activation (ReLU)
            current.mapv_inplace(|x| x.max(0.0));
        }

        Ok(current)
    }
}

/// Result of knowledge transfer
#[derive(Debug, Clone)]
pub struct TransferredKnowledge {
    /// Transferred task embedding
    pub transferred_embedding: Option<Array1<f32>>,
    /// Weights of source tasks
    pub source_weights: HashMap<String, f32>,
    /// Transfer transformation matrices
    pub transfer_matrices: HashMap<String, Array2<f32>>,
    /// Confidence in the transfer
    pub confidence: f32,
}

impl TransferredKnowledge {
    /// Create empty transferred knowledge
    pub fn empty() -> Self {
        Self {
            transferred_embedding: None,
            source_weights: HashMap::new(),
            transfer_matrices: HashMap::new(),
            confidence: 0.0,
        }
    }

    /// Check if transfer has useful knowledge
    pub fn has_knowledge(&self) -> bool {
        self.transferred_embedding.is_some() || !self.transfer_matrices.is_empty()
    }

    /// Get transfer confidence
    pub fn get_confidence(&self) -> f32 {
        self.confidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_task_embedding() {
        let mut embedding1 = TaskEmbedding::new("task1".to_string(), 10);
        let mut embedding2 = TaskEmbedding::new("task2".to_string(), 10);

        embedding1.embedding = Array1::ones(10);
        embedding2.embedding = Array1::ones(10);

        let similarity = embedding1.similarity(&embedding2, &SimilarityMethod::Cosine);
        assert_abs_diff_eq!(similarity, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cross_task_generalizer() {
        let config = GeneralizationConfig::default();
        let mut generalizer = CrossTaskGeneralizer::new(config);

        let data = vec![Array1::ones(128); 10];
        generalizer.register_task("task1".to_string(), &data).unwrap();

        assert!(generalizer.get_task_embedding("task1").is_some());
    }

    #[test]
    fn test_transfer_knowledge() {
        let mut knowledge = TransferKnowledge::new();
        let embedding = TaskEmbedding::new("source1".to_string(), 10);

        knowledge.add_source_task(embedding, 0.8);
        assert_eq!(knowledge.source_embeddings.len(), 1);
        assert_eq!(knowledge.transfer_weights.get("source1"), Some(&0.8));
    }

    #[test]
    fn test_similarity_methods() {
        let mut embedding1 = TaskEmbedding::new("task1".to_string(), 3);
        let mut embedding2 = TaskEmbedding::new("task2".to_string(), 3);

        embedding1.embedding = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        embedding2.embedding = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        let cosine_sim = embedding1.similarity(&embedding2, &SimilarityMethod::Cosine);
        let euclidean_sim = embedding1.similarity(&embedding2, &SimilarityMethod::Euclidean);

        assert_abs_diff_eq!(cosine_sim, 0.0, epsilon = 1e-6);
        assert!(euclidean_sim > 0.0 && euclidean_sim < 1.0);
    }

    #[test]
    fn test_shared_encoder() {
        let config = GeneralizationConfig {
            task_embedding_dim: 5,
            num_shared_layers: 2,
            ..Default::default()
        };
        let encoder = SharedEncoder::new(config);

        let input = Array1::ones(5);
        let encoded = encoder.encode(&input).unwrap();

        assert_eq!(encoded.len(), 5);
    }

    #[test]
    fn test_performance_stats() {
        let mut stats = PerformanceStats::new();
        stats.update(0.8, 100, 1000);

        assert_eq!(stats.best_performance, 0.8);
        assert_eq!(stats.training_steps, 100);
        assert_abs_diff_eq!(stats.sample_efficiency, 0.0008, epsilon = 1e-6);
    }

    #[test]
    fn test_task_similarity_matrix() {
        let config = GeneralizationConfig::default();
        let mut generalizer = CrossTaskGeneralizer::new(config);

        let data1 = vec![Array1::ones(10); 5];
        let data2 = vec![Array1::zeros(10); 5];

        generalizer.register_task("task1".to_string(), &data1).unwrap();
        generalizer.register_task("task2".to_string(), &data2).unwrap();

        let similarity_matrix = generalizer.compute_task_similarity_matrix();
        assert_eq!(similarity_matrix.shape(), &[2, 2]);
        assert_eq!(similarity_matrix[[0, 0]], 1.0);
        assert_eq!(similarity_matrix[[1, 1]], 1.0);
    }
}
