//! Configuration types for meta-learning

use serde::{Deserialize, Serialize};

/// Configuration for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig {
    /// Meta-learning algorithm to use
    pub algorithm: MetaAlgorithm,
    /// Inner loop learning rate (task-specific adaptation)
    pub inner_lr: f64,
    /// Meta learning rate (across-task learning)
    pub meta_lr: f64,
    /// Number of inner loop gradient steps
    pub inner_steps: usize,
    /// Number of support examples per task
    pub support_size: usize,
    /// Number of query examples per task
    pub query_size: usize,
    /// Number of ways (classes) per task
    pub num_ways: usize,
    /// Number of shots (examples per class) per task
    pub num_shots: usize,
    /// Whether to use first-order approximation
    pub first_order: bool,
    /// Temperature for softmax in prototypical networks
    pub temperature: f64,
    /// Dimension of learned embeddings
    pub embedding_dim: usize,
    /// Whether to normalize embeddings
    pub normalize_embeddings: bool,
    /// Memory size for memory-augmented networks
    pub memory_size: usize,
    /// Memory key dimension
    pub memory_key_dim: usize,
    /// Memory value dimension
    pub memory_value_dim: usize,
    /// Number of meta-training tasks per batch
    pub meta_batch_size: usize,
    /// Whether to use task-specific parameters
    pub task_specific_params: bool,
    /// L2 regularization for inner loop
    pub inner_l2_reg: f64,
    /// Gradient clipping threshold
    pub grad_clip_norm: f64,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            algorithm: MetaAlgorithm::MAML,
            inner_lr: 0.01,
            meta_lr: 0.001,
            inner_steps: 5,
            support_size: 5,
            query_size: 15,
            num_ways: 5,
            num_shots: 1,
            first_order: false,
            temperature: 1.0,
            embedding_dim: 512,
            normalize_embeddings: true,
            memory_size: 128,
            memory_key_dim: 64,
            memory_value_dim: 256,
            meta_batch_size: 32,
            task_specific_params: false,
            inner_l2_reg: 0.0001,
            grad_clip_norm: 10.0,
        }
    }
}

/// Meta-learning algorithms supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetaAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML,
    /// Reptile (first-order MAML)
    Reptile,
    /// Prototypical Networks
    ProtoNet,
    /// Matching Networks
    MatchingNet,
    /// Relation Networks
    RelationNet,
    /// Memory-Augmented Neural Networks
    MANN,
    /// Gradient-Based Meta-Learning
    GBML,
    /// Meta-SGD (learn learning rates)
    MetaSGD,
    /// Learning to Learn by Gradient Descent
    L2L,
}