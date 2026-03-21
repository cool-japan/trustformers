//! Optimization configuration and rules

use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::types::EnvironmentType;

/// Optimization rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRule {
    /// Rule name
    pub name: String,

    /// Rule condition
    pub condition: RuleCondition,

    /// Rule action
    pub action: RuleAction,

    /// Rule priority
    pub priority: u8,

    /// Rule enabled
    pub enabled: bool,
}

/// Rule condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleCondition {
    /// Environment type condition
    EnvironmentType(EnvironmentType),

    /// Resource condition
    Resource {
        resource: String,
        operator: String,
        value: f64,
    },

    /// Performance condition
    Performance {
        metric: String,
        operator: String,
        value: f64,
    },

    /// Time condition
    Time {
        period: String,
        operator: String,
        value: String,
    },

    /// Custom condition
    Custom(String),
}

/// Rule action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleAction {
    /// Set configuration value
    SetConfig {
        key: String,
        value: serde_json::Value,
    },

    /// Adjust parallelism
    AdjustParallelism(i32),

    /// Adjust resource allocation
    AdjustResource { resource: String, adjustment: f64 },

    /// Enable feature
    EnableFeature(String),

    /// Disable feature
    DisableFeature(String),

    /// Custom action
    Custom(String),
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization enabled
    pub enabled: bool,

    /// Optimization strategy
    pub strategy: OptimizationStrategy,

    /// Optimization targets
    pub targets: Vec<OptimizationTarget>,

    /// Optimization constraints
    pub constraints: Vec<OptimizationConstraint>,

    /// Learning configuration
    pub learning: LearningConfig,

    /// Optimization schedule
    pub schedule: OptimizationSchedule,

    /// Model persistence
    pub model_persistence: ModelPersistence,
}

/// Optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Greedy optimization
    Greedy,

    /// Genetic algorithm
    Genetic,

    /// Simulated annealing
    SimulatedAnnealing,

    /// Bayesian optimization
    Bayesian,

    /// Reinforcement learning
    ReinforcementLearning,

    /// Multi-objective optimization
    MultiObjective,

    /// Custom strategy
    Custom(String),
}

/// Optimization target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTarget {
    /// Target name
    pub name: String,

    /// Target metric
    pub metric: String,

    /// Target value
    pub target_value: f64,

    /// Optimization priority
    pub priority: OptimizationPriority,

    /// Weight in multi-objective optimization
    pub weight: f64,
}

/// Optimization priorities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraint {
    /// Constraint name
    pub name: String,

    /// Constraint type
    pub constraint_type: ConstraintType,

    /// Constraint value
    pub value: f64,

    /// Hard or soft constraint
    pub hard: bool,
}

/// Constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Maximum value constraint
    Maximum,

    /// Minimum value constraint
    Minimum,

    /// Range constraint
    Range { min: f64, max: f64 },

    /// Resource constraint
    Resource(String),

    /// Custom constraint
    Custom(String),
}

/// Learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Learning enabled
    pub enabled: bool,

    /// Learning algorithm
    pub algorithm: LearningAlgorithm,

    /// Learning rate
    pub learning_rate: f64,

    /// Exploration rate
    pub exploration_rate: f64,

    /// Training episodes
    pub training_episodes: usize,

    /// Memory size
    pub memory_size: usize,

    /// Batch size
    pub batch_size: usize,

    /// Update frequency
    pub update_frequency: usize,
}

/// Learning algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningAlgorithm {
    /// Q-Learning
    QLearning,

    /// Deep Q-Network
    DQN,

    /// Actor-Critic
    ActorCritic,

    /// Policy Gradient
    PolicyGradient,

    /// Proximal Policy Optimization
    PPO,

    /// Trust Region Policy Optimization
    TRPO,

    /// Soft Actor-Critic
    SAC,

    /// Twin Delayed Deep Deterministic Policy Gradient
    TD3,

    /// Custom algorithm
    Custom(String),
}

/// Model persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPersistence {
    /// Persistence enabled
    pub enabled: bool,

    /// Model save path
    pub save_path: String,

    /// Model format
    pub format: ModelFormat,

    /// Save frequency (episodes)
    pub save_frequency: usize,

    /// Keep best models only
    pub keep_best_only: bool,

    /// Maximum models to keep
    pub max_models: usize,

    /// Compression enabled
    pub compression: bool,
}

/// Model formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFormat {
    /// JSON format
    Json,

    /// Binary format
    Binary,

    /// ONNX format
    Onnx,

    /// TensorFlow format
    TensorFlow,

    /// PyTorch format
    PyTorch,

    /// Custom format
    Custom(String),
}

/// Optimization schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSchedule {
    /// Schedule enabled
    pub enabled: bool,

    /// Optimization interval
    pub interval: Duration,

    /// Minimum data points required
    pub min_data_points: usize,

    /// Cooldown period after optimization
    pub cooldown_period: Duration,

    /// Maximum optimization time
    pub max_optimization_time: Duration,
}
