//! # Neural Architecture Search (NAS) Framework
//!
//! This module provides comprehensive neural architecture search capabilities for automatically
//! discovering optimal transformer architectures. It supports various search strategies,
//! search spaces, and optimization objectives.
//!
//! ## Features
//!
//! - **Multiple Search Strategies**: Supports evolutionary search, reinforcement learning-based search,
//!   a REINFORCE controller, Bayesian optimization and random search
//! - **Flexible Search Space**: Define custom architecture search spaces with constraints
//! - **Multi-Objective Optimization**: Balance accuracy, efficiency, memory usage, and latency
//! - **Progressive Search**: Start with simple architectures and progressively increase complexity
//! - **Hardware-Aware Search**: Consider target hardware constraints during search
//! - **Architecture Encoding**: Efficient representation of neural architectures
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_models::neural_architecture_search::{
//!     NASConfig, NeuralArchitectureSearcher, SearchStrategy, SearchSpace, OptimizationObjective
//! };
//! use trustformers_core::Result;
//!
//! fn main() -> Result<()> {
//!     // Define search configuration
//!     let config = NASConfig {
//!         strategy: SearchStrategy::Evolutionary,
//!         search_space: SearchSpace::transformer_space(),
//!         objectives: vec![
//!             OptimizationObjective::Accuracy { weight: 0.7 },
//!             OptimizationObjective::Efficiency { weight: 0.3 },
//!         ],
//!         max_evaluations: 1000,
//!         ..Default::default()
//!     };
//!
//!     // Create and run searcher
//!     let mut searcher = NeuralArchitectureSearcher::new(config)?;
//!     let best_architecture = searcher.search()?;
//!
//!     let _ = best_architecture;
//!     Ok(())
//! }
//! ```

use scirs2_core::random::*; // SciRS2 Integration Policy (was: use rand::{Rng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use tracing::debug;
use trustformers_core::errors::{invalid_input, Result, TrustformersError};

pub mod proxy;
pub mod strategies;

pub use proxy::{ArchitectureEvaluator, MeasuredPerformance, ProxyTaskConfig, ProxyTaskEvaluator};
pub use strategies::{
    crowding_distances, encode_architecture, non_dominated_fronts, GaussianProcess,
    ReinforceController,
};

/// Configuration for Neural Architecture Search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NASConfig {
    /// Search strategy to use
    pub strategy: SearchStrategy,
    /// Architecture search space definition
    pub search_space: SearchSpace,
    /// Optimization objectives with weights
    pub objectives: Vec<OptimizationObjective>,
    /// Maximum number of architectures to evaluate
    pub max_evaluations: usize,
    /// Population size for evolutionary/population-based methods
    pub population_size: usize,
    /// Number of generations for evolutionary search
    pub generations: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Hardware constraints
    pub hardware_constraints: Option<HardwareConstraints>,
    /// Progressive search configuration
    pub progressive_search: bool,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for NASConfig {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::Evolutionary,
            search_space: SearchSpace::default(),
            objectives: vec![OptimizationObjective::Accuracy { weight: 1.0 }],
            max_evaluations: 1000,
            population_size: 50,
            generations: 20,
            patience: 5,
            hardware_constraints: None,
            progressive_search: true,
            seed: None,
        }
    }
}

/// Neural Architecture Search strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Random search baseline
    Random,
    /// Evolutionary algorithm-based search
    Evolutionary,
    /// Reinforcement learning-based search with a REINFORCE controller
    ReinforcementLearning,
    /// Progressive search with increasing complexity
    Progressive,
    /// Bayesian optimization
    BayesianOptimization,
    /// Multi-objective evolutionary algorithm
    NSGA2,
}

/// Architecture search space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    /// Dimension ranges for architecture components
    pub dimensions: HashMap<String, DimensionRange>,
    /// Component choices (e.g., activation functions, attention types)
    pub choices: HashMap<String, Vec<String>>,
    /// Architecture constraints
    pub constraints: Vec<ArchitectureConstraint>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self::transformer_space()
    }
}

impl SearchSpace {
    /// Create a transformer-specific search space
    pub fn transformer_space() -> Self {
        let mut dimensions = HashMap::new();
        let mut choices = HashMap::new();

        // Dimension ranges
        dimensions.insert("num_layers".to_string(), DimensionRange::new(6, 24, 1));
        dimensions.insert(
            "hidden_size".to_string(),
            DimensionRange::new(512, 4096, 64),
        );
        dimensions.insert("num_heads".to_string(), DimensionRange::new(8, 32, 4));
        dimensions.insert(
            "intermediate_size".to_string(),
            DimensionRange::new(2048, 16384, 256),
        );
        dimensions.insert(
            "max_position_embeddings".to_string(),
            DimensionRange::new(512, 8192, 512),
        );

        // Component choices
        choices.insert(
            "activation".to_string(),
            vec![
                "gelu".to_string(),
                "relu".to_string(),
                "swish".to_string(),
                "silu".to_string(),
                "gelu_new".to_string(),
            ],
        );

        choices.insert(
            "attention_type".to_string(),
            vec![
                "standard".to_string(),
                "grouped_query".to_string(),
                "multi_query".to_string(),
                "sparse".to_string(),
                "sliding_window".to_string(),
            ],
        );

        choices.insert(
            "normalization".to_string(),
            vec![
                "layer_norm".to_string(),
                "rms_norm".to_string(),
                "group_norm".to_string(),
            ],
        );

        choices.insert(
            "position_encoding".to_string(),
            vec![
                "absolute".to_string(),
                "relative".to_string(),
                "rotary".to_string(),
                "alibi".to_string(),
            ],
        );

        Self {
            dimensions,
            choices,
            constraints: vec![
                ArchitectureConstraint::DivisibilityConstraint {
                    dimension: "hidden_size".to_string(),
                    divisor: "num_heads".to_string(),
                },
                ArchitectureConstraint::RatioConstraint {
                    numerator: "intermediate_size".to_string(),
                    denominator: "hidden_size".to_string(),
                    min_ratio: 2.0,
                    max_ratio: 8.0,
                },
            ],
        }
    }

    /// Create a vision transformer search space
    pub fn vision_transformer_space() -> Self {
        let mut dimensions = HashMap::new();
        let mut choices = HashMap::new();

        dimensions.insert("num_layers".to_string(), DimensionRange::new(6, 24, 1));
        dimensions.insert(
            "hidden_size".to_string(),
            DimensionRange::new(384, 1536, 64),
        );
        dimensions.insert("num_heads".to_string(), DimensionRange::new(6, 24, 2));
        dimensions.insert("patch_size".to_string(), DimensionRange::new(8, 32, 4));
        dimensions.insert("image_size".to_string(), DimensionRange::new(224, 512, 32));

        choices.insert(
            "pooling".to_string(),
            vec![
                "cls_token".to_string(),
                "gap".to_string(),
                "map".to_string(),
            ],
        );

        Self {
            dimensions,
            choices,
            constraints: vec![
                ArchitectureConstraint::DivisibilityConstraint {
                    dimension: "hidden_size".to_string(),
                    divisor: "num_heads".to_string(),
                },
                ArchitectureConstraint::DivisibilityConstraint {
                    dimension: "image_size".to_string(),
                    divisor: "patch_size".to_string(),
                },
            ],
        }
    }

    /// Validate if an architecture satisfies all constraints
    pub fn validate_architecture(&self, architecture: &Architecture) -> Result<()> {
        for constraint in &self.constraints {
            constraint.validate(architecture)?;
        }
        Ok(())
    }
}

/// Range definition for continuous dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionRange {
    pub min: i32,
    pub max: i32,
    pub step: i32,
}

impl DimensionRange {
    pub fn new(min: i32, max: i32, step: i32) -> Self {
        Self { min, max, step }
    }

    pub fn sample(&self, rng: &mut impl Rng) -> i32 {
        let steps = (self.max - self.min) / self.step + 1;
        let step_idx = rng.random_range(0..steps);
        self.min + step_idx * self.step
    }

    pub fn validate(&self, value: i32) -> bool {
        value >= self.min && value <= self.max && (value - self.min) % self.step == 0
    }
}

/// Architecture constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitectureConstraint {
    /// Ensure one dimension is divisible by another
    DivisibilityConstraint { dimension: String, divisor: String },
    /// Ensure a ratio between two dimensions is within bounds
    RatioConstraint {
        numerator: String,
        denominator: String,
        min_ratio: f32,
        max_ratio: f32,
    },
    /// Ensure total parameters are within bounds
    ParameterConstraint {
        min_params: Option<usize>,
        max_params: Option<usize>,
    },
    /// Custom constraint function
    CustomConstraint { name: String, description: String },
}

impl ArchitectureConstraint {
    fn validate(&self, architecture: &Architecture) -> Result<()> {
        match self {
            ArchitectureConstraint::DivisibilityConstraint { dimension, divisor } => {
                let dim_val = architecture
                    .dimensions
                    .get(dimension)
                    .ok_or_else(|| invalid_input(format!("Missing dimension: {}", dimension)))?;
                let div_val = architecture
                    .dimensions
                    .get(divisor)
                    .ok_or_else(|| invalid_input(format!("Missing divisor: {}", divisor)))?;

                if dim_val % div_val != 0 {
                    return Err(invalid_input(format!(
                        "{} ({}) must be divisible by {} ({})",
                        dimension, dim_val, divisor, div_val
                    )));
                }
            },
            ArchitectureConstraint::RatioConstraint {
                numerator,
                denominator,
                min_ratio,
                max_ratio,
            } => {
                let num_val = *architecture
                    .dimensions
                    .get(numerator)
                    .ok_or_else(|| invalid_input(format!("Missing numerator: {}", numerator)))?
                    as f32;
                let den_val =
                    *architecture.dimensions.get(denominator).ok_or_else(|| {
                        invalid_input(format!("Missing denominator: {}", denominator))
                    })? as f32;

                let ratio = num_val / den_val;
                if ratio < *min_ratio || ratio > *max_ratio {
                    return Err(invalid_input(format!(
                        "Ratio {} / {} ({:.2}) must be between {:.2} and {:.2}",
                        numerator, denominator, ratio, min_ratio, max_ratio
                    )));
                }
            },
            ArchitectureConstraint::ParameterConstraint {
                min_params,
                max_params,
            } => {
                let params = architecture.estimate_parameters();
                if let Some(min) = min_params {
                    if params < *min {
                        return Err(invalid_input(format!(
                            "Architecture has {} parameters, minimum required: {}",
                            params, min
                        )));
                    }
                }
                if let Some(max) = max_params {
                    if params > *max {
                        return Err(invalid_input(format!(
                            "Architecture has {} parameters, maximum allowed: {}",
                            params, max
                        )));
                    }
                }
            },
            ArchitectureConstraint::CustomConstraint { name, .. } => {
                // Custom constraints would be implemented separately
                return Err(invalid_input(format!(
                    "Custom constraint '{}' not implemented",
                    name
                )));
            },
        }
        Ok(())
    }
}

/// Optimization objectives for NAS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Maximize model accuracy
    Accuracy { weight: f32 },
    /// Minimize inference latency
    Latency { weight: f32 },
    /// Minimize memory usage
    Memory { weight: f32 },
    /// Minimize energy consumption
    Energy { weight: f32 },
    /// Minimize model size (parameters)
    ModelSize { weight: f32 },
    /// Maximize efficiency (accuracy/flops)
    Efficiency { weight: f32 },
    /// Custom objective
    Custom { name: String, weight: f32 },
}

impl OptimizationObjective {
    pub fn weight(&self) -> f32 {
        match self {
            OptimizationObjective::Accuracy { weight }
            | OptimizationObjective::Latency { weight }
            | OptimizationObjective::Memory { weight }
            | OptimizationObjective::Energy { weight }
            | OptimizationObjective::ModelSize { weight }
            | OptimizationObjective::Efficiency { weight }
            | OptimizationObjective::Custom { weight, .. } => *weight,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            OptimizationObjective::Accuracy { .. } => "accuracy",
            OptimizationObjective::Latency { .. } => "latency",
            OptimizationObjective::Memory { .. } => "memory",
            OptimizationObjective::Energy { .. } => "energy",
            OptimizationObjective::ModelSize { .. } => "model_size",
            OptimizationObjective::Efficiency { .. } => "efficiency",
            OptimizationObjective::Custom { name, .. } => name,
        }
    }
}

/// Hardware constraints for architecture search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConstraints {
    /// Maximum memory in GB
    pub max_memory_gb: Option<f32>,
    /// Maximum latency in milliseconds
    pub max_latency_ms: Option<f32>,
    /// Target hardware platform
    pub platform: HardwarePlatform,
    /// Energy constraints
    pub max_energy_mj: Option<f32>,
    /// Throughput requirements
    pub min_throughput: Option<f32>,
}

/// Target hardware platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwarePlatform {
    CPU,
    GPU {
        memory_gb: f32,
    },
    TPU,
    Mobile,
    Edge,
    Custom {
        name: String,
        specs: HashMap<String, f32>,
    },
}

/// Neural architecture representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Architecture {
    /// Numerical dimensions (e.g., layer count, hidden size)
    pub dimensions: HashMap<String, i32>,
    /// Categorical choices (e.g., activation function, attention type)
    pub choices: HashMap<String, String>,
    /// Architecture metadata
    pub metadata: ArchitectureMetadata,
}

impl Default for Architecture {
    fn default() -> Self {
        Self::new()
    }
}

impl Architecture {
    pub fn new() -> Self {
        Self {
            dimensions: HashMap::new(),
            choices: HashMap::new(),
            metadata: ArchitectureMetadata::default(),
        }
    }

    /// Estimate the number of parameters for this architecture
    pub fn estimate_parameters(&self) -> usize {
        let hidden_size = *self.dimensions.get("hidden_size").unwrap_or(&768) as f64;
        let num_layers = *self.dimensions.get("num_layers").unwrap_or(&12) as f64;
        let vocab_size = *self.dimensions.get("vocab_size").unwrap_or(&32000) as f64;
        let intermediate_size =
            *self.dimensions.get("intermediate_size").unwrap_or(&(hidden_size as i32 * 4)) as f64;

        // Rough parameter estimation for transformer
        let embedding_params = vocab_size * hidden_size;
        let attention_params = num_layers * (4.0 * hidden_size * hidden_size);
        let ffn_params = num_layers * (2.0 * hidden_size * intermediate_size);
        let norm_params = num_layers * 2.0 * hidden_size;

        (embedding_params + attention_params + ffn_params + norm_params) as usize
    }

    /// Estimate memory usage in MB
    pub fn estimate_memory_mb(&self) -> f32 {
        let params = self.estimate_parameters() as f32;
        // Rough estimation: 4 bytes per parameter + activations overhead
        (params * 4.0 * 1.5) / (1024.0 * 1024.0)
    }

    /// Estimate inference latency (relative units)
    pub fn estimate_latency(&self) -> f32 {
        let num_layers = *self.dimensions.get("num_layers").unwrap_or(&12) as f32;
        let hidden_size = *self.dimensions.get("hidden_size").unwrap_or(&768) as f32;

        // Simple latency model based on architectural complexity
        num_layers * hidden_size.powf(1.5) / 1000000.0
    }

    /// Generate a random architecture within the search space
    pub fn random(search_space: &SearchSpace, rng: &mut impl Rng) -> Self {
        let mut architecture = Architecture::new();

        // Sample dimensions
        for (name, range) in &search_space.dimensions {
            architecture.dimensions.insert(name.clone(), range.sample(rng));
        }

        // Sample choices
        for (name, options) in &search_space.choices {
            if !options.is_empty() {
                let choice = options[rng.random_range(0..options.len())].clone();
                architecture.choices.insert(name.clone(), choice);
            }
        }

        architecture
    }

    /// Mutate the architecture for evolutionary search
    pub fn mutate(&mut self, search_space: &SearchSpace, mutation_rate: f32, rng: &mut impl Rng) {
        // Mutate dimensions
        for (name, value) in &mut self.dimensions {
            if rng.random::<f32>() < mutation_rate {
                if let Some(range) = search_space.dimensions.get(name) {
                    *value = range.sample(rng);
                }
            }
        }

        // Mutate choices
        for (name, value) in &mut self.choices {
            if rng.random::<f32>() < mutation_rate {
                if let Some(options) = search_space.choices.get(name) {
                    if !options.is_empty() {
                        *value = options[rng.random_range(0..options.len())].clone();
                    }
                }
            }
        }

        self.metadata.generation += 1;
    }

    /// Create a crossover between two architectures
    pub fn crossover(&self, other: &Architecture, rng: &mut impl Rng) -> Architecture {
        let mut child = Architecture::new();

        // Crossover dimensions
        for name in self.dimensions.keys() {
            let value = if rng.random::<f32>() < 0.5 {
                self.dimensions[name]
            } else {
                other.dimensions.get(name).copied().unwrap_or(self.dimensions[name])
            };
            child.dimensions.insert(name.clone(), value);
        }

        // Crossover choices
        for name in self.choices.keys() {
            let value = if rng.random::<f32>() < 0.5 {
                self.choices[name].clone()
            } else {
                other.choices.get(name).cloned().unwrap_or_else(|| self.choices[name].clone())
            };
            child.choices.insert(name.clone(), value);
        }

        child.metadata.generation =
            std::cmp::max(self.metadata.generation, other.metadata.generation) + 1;
        child
    }
}

/// Metadata associated with an architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureMetadata {
    /// Unique identifier
    pub id: String,
    /// Generation in evolutionary search
    pub generation: u32,
    /// Parent architectures (for tracking lineage)
    pub parents: Vec<String>,
    /// Creation timestamp
    pub created_at: std::time::SystemTime,
}

impl Default for ArchitectureMetadata {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            generation: 0,
            parents: Vec::new(),
            created_at: std::time::SystemTime::now(),
        }
    }
}

/// Evaluation results for an architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureEvaluation {
    /// The evaluated architecture
    pub architecture: Architecture,
    /// Performance metrics
    pub metrics: HashMap<String, f32>,
    /// Overall fitness score
    pub fitness: f32,
    /// Evaluation time
    pub evaluation_time: std::time::Duration,
    /// Additional information
    pub info: HashMap<String, String>,
}

impl ArchitectureEvaluation {
    pub fn new(architecture: Architecture) -> Self {
        Self {
            architecture,
            metrics: HashMap::new(),
            fitness: 0.0,
            evaluation_time: std::time::Duration::from_secs(0),
            info: HashMap::new(),
        }
    }
}

/// Main Neural Architecture Search engine
pub struct NeuralArchitectureSearcher {
    config: NASConfig,
    search_space: SearchSpace,
    population: Vec<ArchitectureEvaluation>,
    best_architecture: Option<ArchitectureEvaluation>,
    evaluation_history: Vec<ArchitectureEvaluation>,
    rng: StdRng,
    /// Performs the real train-and-evaluate loop for every candidate.
    evaluator: Box<dyn ArchitectureEvaluator>,
}

impl NeuralArchitectureSearcher {
    pub fn new(config: NASConfig) -> Result<Self> {
        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(random::<u64>())
        };

        Ok(Self {
            search_space: config.search_space.clone(),
            config,
            population: Vec::new(),
            best_architecture: None,
            evaluation_history: Vec::new(),
            rng,
            evaluator: Box::new(ProxyTaskEvaluator::new()),
        })
    }

    /// Create a searcher that evaluates candidates with a caller-supplied
    /// evaluator (for example a real training pipeline).
    pub fn with_evaluator(
        config: NASConfig,
        evaluator: Box<dyn ArchitectureEvaluator>,
    ) -> Result<Self> {
        let mut searcher = Self::new(config)?;
        searcher.evaluator = evaluator;
        Ok(searcher)
    }

    /// Description of what the fitness numbers were measured on.
    pub fn evaluator_description(&self) -> String {
        self.evaluator.description()
    }

    /// Run the architecture search
    pub fn search(&mut self) -> Result<ArchitectureEvaluation> {
        match self.config.strategy {
            SearchStrategy::Random => self.random_search(),
            SearchStrategy::Evolutionary => self.evolutionary_search(),
            SearchStrategy::ReinforcementLearning => self.rl_search(),
            SearchStrategy::Progressive => self.progressive_search(),
            SearchStrategy::BayesianOptimization => self.bayesian_search(),
            SearchStrategy::NSGA2 => self.nsga2_search(),
        }
    }

    fn random_search(&mut self) -> Result<ArchitectureEvaluation> {
        for i in 0..self.config.max_evaluations {
            let architecture = Architecture::random(&self.search_space, &mut self.rng);
            let evaluation = self.evaluate_architecture(architecture)?;

            self.update_best(&evaluation);
            self.evaluation_history.push(evaluation);

            if i % 100 == 0 {
                if let Some(ref best) = self.best_architecture {
                    debug!(iteration = i, best = best.fitness, "random search progress");
                }
            }
        }

        self.best_architecture.clone().ok_or_else(|| {
            TrustformersError::invalid_config("No architecture found during search".to_string())
        })
    }

    fn evolutionary_search(&mut self) -> Result<ArchitectureEvaluation> {
        // Initialize population
        self.initialize_population()?;

        for generation in 0..self.config.generations {
            // Select parents
            let parents = self.select_parents();

            // Create offspring through crossover and mutation
            let mut offspring = Vec::new();
            for _ in 0..self.config.population_size / 2 {
                let parent1_idx = self.rng.random_range(0..parents.len());
                let parent2_idx = self.rng.random_range(0..parents.len());
                let parent1 = &parents[parent1_idx];
                let parent2 = &parents[parent2_idx];

                let mut child1 =
                    parent1.architecture.crossover(&parent2.architecture, &mut self.rng);
                let mut child2 =
                    parent2.architecture.crossover(&parent1.architecture, &mut self.rng);

                child1.mutate(&self.search_space, 0.1, &mut self.rng);
                child2.mutate(&self.search_space, 0.1, &mut self.rng);

                offspring.push(self.evaluate_architecture(child1)?);
                offspring.push(self.evaluate_architecture(child2)?);
            }

            // Environmental selection
            self.environmental_selection(offspring)?;

            debug!(
                generation,
                best = self.best_architecture.as_ref().map_or(0.0, |a| a.fitness),
                "evolutionary search progress"
            );
        }

        self.best_architecture.clone().ok_or_else(|| {
            TrustformersError::invalid_config("No architecture found during search".to_string())
        })
    }

    /// Reinforcement-learning search driven by a REINFORCE controller.
    ///
    /// The controller keeps a categorical policy over every search-space
    /// dimension, samples architectures from it, and shifts probability mass
    /// towards the actions whose **measured** fitness beat the running baseline.
    fn rl_search(&mut self) -> Result<ArchitectureEvaluation> {
        let mut controller = ReinforceController::new(&self.search_space, 0.5);

        for i in 0..self.config.max_evaluations {
            let (architecture, actions) = controller.sample(&self.search_space, &mut self.rng);

            // A candidate the search space rejects is skipped; any *other*
            // failure (a broken objective, a diverging evaluator) is propagated
            // rather than hidden behind a shorter history.
            if self.search_space.validate_architecture(&architecture).is_err() {
                continue;
            }
            let evaluation = self.evaluate_architecture(architecture)?;

            controller.update(&actions, evaluation.fitness);
            self.update_best(&evaluation);
            self.evaluation_history.push(evaluation);

            if i % 100 == 0 {
                debug!(
                    iteration = i,
                    baseline = controller.baseline(),
                    best = self.best_architecture.as_ref().map_or(0.0, |a| a.fitness),
                    "REINFORCE search progress"
                );
            }
        }

        self.best_architecture.clone().ok_or_else(|| {
            TrustformersError::invalid_config("No architecture found during search".to_string())
        })
    }

    fn progressive_search(&mut self) -> Result<ArchitectureEvaluation> {
        // Start with small architectures and progressively increase complexity
        let complexity_stages = 5;
        let evaluations_per_stage = self.config.max_evaluations / complexity_stages;

        for stage in 0..complexity_stages {
            let complexity_factor = (stage + 1) as f32 / complexity_stages as f32;

            for i in 0..evaluations_per_stage {
                let mut architecture = Architecture::random(&self.search_space, &mut self.rng);

                // Scale down architecture based on complexity factor
                for (name, value) in &mut architecture.dimensions {
                    if let Some(range) = self.search_space.dimensions.get(name) {
                        let scaled =
                            range.min + (((*value - range.min) as f32 * complexity_factor) as i32);
                        *value = scaled.clamp(range.min, range.max);
                    }
                }

                let evaluation = self.evaluate_architecture(architecture)?;
                self.update_best(&evaluation);
                self.evaluation_history.push(evaluation);

                if i % 50 == 0 {
                    debug!(
                        stage,
                        iteration = i,
                        best = self.best_architecture.as_ref().map_or(0.0, |a| a.fitness),
                        "progressive search progress"
                    );
                }
            }
        }

        self.best_architecture.clone().ok_or_else(|| {
            TrustformersError::invalid_config("No architecture found during search".to_string())
        })
    }

    /// Bayesian optimization with a Gaussian-process surrogate.
    ///
    /// After an initial random design the search fits a GP to the **measured**
    /// fitness of everything evaluated so far and picks the candidate with the
    /// highest Expected Improvement out of a freshly sampled pool.
    fn bayesian_search(&mut self) -> Result<ArchitectureEvaluation> {
        const INITIAL_DESIGN: usize = 10;
        const CANDIDATE_POOL: usize = 32;

        let mut observations: Vec<Vec<f32>> = Vec::new();
        let mut targets: Vec<f32> = Vec::new();

        for i in 0..self.config.max_evaluations {
            let architecture = if i < INITIAL_DESIGN || observations.len() < 2 {
                Architecture::random(&self.search_space, &mut self.rng)
            } else {
                let surrogate =
                    GaussianProcess::fit(observations.clone(), targets.clone(), 0.5, 1e-4);

                let mut best_candidate: Option<(f32, Architecture)> = None;
                for _ in 0..CANDIDATE_POOL {
                    let candidate = Architecture::random(&self.search_space, &mut self.rng);
                    let encoding = encode_architecture(&candidate, &self.search_space);
                    let acquisition = surrogate.expected_improvement(&encoding);
                    let better = best_candidate
                        .as_ref()
                        .map(|(score, _)| acquisition > *score)
                        .unwrap_or(true);
                    if better {
                        best_candidate = Some((acquisition, candidate));
                    }
                }

                match best_candidate {
                    Some((_, candidate)) => candidate,
                    None => Architecture::random(&self.search_space, &mut self.rng),
                }
            };

            let encoding = encode_architecture(&architecture, &self.search_space);
            if self.search_space.validate_architecture(&architecture).is_err() {
                continue;
            }
            let evaluation = self.evaluate_architecture(architecture)?;

            observations.push(encoding);
            targets.push(evaluation.fitness);

            self.update_best(&evaluation);
            self.evaluation_history.push(evaluation);

            if i % 100 == 0 {
                debug!(
                    iteration = i,
                    observations = observations.len(),
                    best = self.best_architecture.as_ref().map_or(0.0, |a| a.fitness),
                    "Bayesian search progress"
                );
            }
        }

        self.best_architecture.clone().ok_or_else(|| {
            TrustformersError::invalid_config("No architecture found during search".to_string())
        })
    }

    fn nsga2_search(&mut self) -> Result<ArchitectureEvaluation> {
        // Simplified NSGA-II for multi-objective optimization
        self.initialize_population()?;

        for generation in 0..self.config.generations {
            let parents = self.select_parents();
            let mut offspring = Vec::new();

            for _ in 0..self.config.population_size {
                let parent1_idx = self.rng.random_range(0..parents.len());
                let parent2_idx = self.rng.random_range(0..parents.len());
                let parent1 = &parents[parent1_idx];
                let parent2 = &parents[parent2_idx];

                let mut child =
                    parent1.architecture.crossover(&parent2.architecture, &mut self.rng);
                child.mutate(&self.search_space, 0.1, &mut self.rng);

                offspring.push(self.evaluate_architecture(child)?);
            }

            // Multi-objective environmental selection
            self.nsga2_selection(offspring)?;

            debug!(
                generation,
                population = self.population.len(),
                "NSGA-II generation complete"
            );
        }

        // Return the best overall architecture
        self.best_architecture.clone().ok_or_else(|| {
            TrustformersError::invalid_config("No architecture found during search".to_string())
        })
    }

    fn initialize_population(&mut self) -> Result<()> {
        self.population.clear();

        for _ in 0..self.config.population_size {
            let architecture = Architecture::random(&self.search_space, &mut self.rng);
            let evaluation = self.evaluate_architecture(architecture)?;
            self.population.push(evaluation);
        }

        // Update best architecture
        if let Some(best) = self
            .population
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
        {
            self.best_architecture = Some(best.clone());
        }

        Ok(())
    }

    fn select_parents(&mut self) -> Vec<ArchitectureEvaluation> {
        // Tournament selection
        let tournament_size = 3;
        let mut parents = Vec::new();

        for _ in 0..self.config.population_size {
            let mut tournament = Vec::new();
            for _ in 0..tournament_size {
                let idx = self.rng.random_range(0..self.population.len());
                tournament.push(self.population[idx].clone());
            }

            let winner = tournament
                .into_iter()
                .max_by(|a, b| {
                    a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|| {
                    // Fallback: should never happen as tournament has fixed size
                    self.population[0].clone()
                });
            parents.push(winner);
        }

        parents
    }

    fn environmental_selection(&mut self, offspring: Vec<ArchitectureEvaluation>) -> Result<()> {
        // Combine population and offspring
        let mut combined = self.population.clone();
        combined.extend(offspring);

        // Sort by fitness
        combined
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top individuals
        self.population = combined.into_iter().take(self.config.population_size).collect();

        // Update best architecture
        if let Some(best) = self.population.first() {
            let should_update = self
                .best_architecture
                .as_ref()
                .is_none_or(|current| best.fitness > current.fitness);
            if should_update {
                self.best_architecture = Some(best.clone());
            }
        }

        Ok(())
    }

    /// NSGA-II environmental selection: fast non-dominated sorting followed by
    /// crowding-distance tie-breaking within the last accepted front.
    fn nsga2_selection(&mut self, offspring: Vec<ArchitectureEvaluation>) -> Result<()> {
        let mut combined = std::mem::take(&mut self.population);
        combined.extend(offspring);

        let objective_names: Vec<String> = self
            .config
            .objectives
            .iter()
            .map(|objective| objective.name().to_string())
            .collect();

        let fronts = non_dominated_fronts(&combined, &objective_names);
        let mut selected: Vec<usize> = Vec::with_capacity(self.config.population_size);

        for front in &fronts {
            if selected.len() + front.len() <= self.config.population_size {
                selected.extend(front.iter().copied());
                continue;
            }

            // Partially accept this front, preferring the least crowded members.
            let distances = crowding_distances(&combined, front, &objective_names);
            let mut ordered: Vec<(usize, f32)> = front.iter().copied().zip(distances).collect();
            ordered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (index, _) in ordered {
                if selected.len() >= self.config.population_size {
                    break;
                }
                selected.push(index);
            }
            break;
        }

        // Rebuild the population in the selected order.
        let mut keep = vec![false; combined.len()];
        for index in &selected {
            keep[*index] = true;
        }
        let mut new_population = Vec::with_capacity(selected.len());
        for (index, evaluation) in combined.into_iter().enumerate() {
            if keep[index] {
                new_population.push(evaluation);
            }
        }
        self.population = new_population;

        // Track the best scalarised individual for reporting.
        if let Some(best) = self
            .population
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
        {
            let should_update = self
                .best_architecture
                .as_ref()
                .is_none_or(|current| best.fitness > current.fitness);
            if should_update {
                self.best_architecture = Some(best.clone());
            }
        }

        Ok(())
    }

    /// Train and evaluate one candidate architecture.
    ///
    /// The accuracy comes from the evaluator's **measured** held-out accuracy
    /// after really training the candidate; latency is the evaluator's measured
    /// inference time. Size and memory objectives use the architecture's analytic
    /// cost model, which is recorded as such in `evaluation.info` so a reader can
    /// tell measurements from estimates.
    ///
    /// # Errors
    ///
    /// * The architecture violates the search space's constraints.
    /// * An objective has no measurement source (`Energy`, or a `Custom`
    ///   objective the evaluator does not report). Scoring those with a constant
    ///   would make the whole search meaningless.
    fn evaluate_architecture(
        &mut self,
        architecture: Architecture,
    ) -> Result<ArchitectureEvaluation> {
        let start_time = std::time::Instant::now();

        // Validate architecture
        self.search_space.validate_architecture(&architecture)?;

        let measured = self.evaluator.evaluate(&architecture)?;
        let mut evaluation = ArchitectureEvaluation::new(architecture);

        evaluation.info.insert("evaluator".to_string(), self.evaluator.description());
        evaluation.info.insert(
            "trained_parameters".to_string(),
            measured.trained_parameters.to_string(),
        );
        evaluation.info.insert(
            "train_loss".to_string(),
            format!("{:.6}", measured.train_loss),
        );

        // Compute metrics based on objectives
        for objective in &self.config.objectives {
            let (metric_name, metric_value, source) = match objective {
                OptimizationObjective::Accuracy { .. } => (
                    "accuracy",
                    measured.accuracy.clamp(0.0, 1.0),
                    "measured_holdout_accuracy",
                ),
                OptimizationObjective::Latency { .. } => {
                    // Measured inference time, inverted so that higher is better.
                    let latency_ms = (measured.inference_seconds * 1000.0) as f32;
                    (
                        "latency",
                        1.0 / (1.0 + latency_ms),
                        "measured_inference_time",
                    )
                },
                OptimizationObjective::Memory { .. } => {
                    let memory = evaluation.architecture.estimate_memory_mb();
                    (
                        "memory",
                        1.0 / (1.0 + memory / 1000.0),
                        "analytic_cost_model",
                    )
                },
                OptimizationObjective::ModelSize { .. } => {
                    let params = evaluation.architecture.estimate_parameters() as f32;
                    (
                        "model_size",
                        1.0 / (1.0 + params / 1_000_000.0),
                        "analytic_cost_model",
                    )
                },
                OptimizationObjective::Efficiency { .. } => {
                    // Measured accuracy per estimated million parameters.
                    let params = evaluation.architecture.estimate_parameters() as f32;
                    (
                        "efficiency",
                        measured.accuracy / (1.0 + params / 1_000_000.0),
                        "measured_accuracy_over_analytic_size",
                    )
                },
                OptimizationObjective::Energy { .. } => {
                    return Err(TrustformersError::invalid_config(
                        "the Energy objective has no measurement source: this crate cannot read \
                         an energy counter, and scoring it with a formula would fabricate the \
                         result. Remove the objective or supply an ArchitectureEvaluator that \
                         reports `energy` as a custom metric."
                            .to_string(),
                    ));
                },
                OptimizationObjective::Custom { name, .. } => {
                    let value = measured.custom_metrics.get(name).copied().ok_or_else(|| {
                        TrustformersError::invalid_config(format!(
                            "custom objective `{name}` was requested but the evaluator did not \
                             report a metric with that name"
                        ))
                    })?;
                    (name.as_str(), value, "evaluator_custom_metric")
                },
            };

            evaluation.metrics.insert(metric_name.to_string(), metric_value);
            evaluation.info.insert(format!("{metric_name}_source"), source.to_string());
        }

        // Compute overall fitness as weighted sum
        evaluation.fitness = self
            .config
            .objectives
            .iter()
            .map(|obj| {
                let metric_value = evaluation.metrics.get(obj.name()).unwrap_or(&0.0);
                obj.weight() * metric_value
            })
            .sum();

        evaluation.evaluation_time = start_time.elapsed();

        Ok(evaluation)
    }

    fn update_best(&mut self, evaluation: &ArchitectureEvaluation) {
        let should_update = self
            .best_architecture
            .as_ref()
            .is_none_or(|current| evaluation.fitness > current.fitness);
        if should_update {
            self.best_architecture = Some(evaluation.clone());
        }
    }

    /// Get the current best architecture
    pub fn best_architecture(&self) -> Option<&ArchitectureEvaluation> {
        self.best_architecture.as_ref()
    }

    /// Get evaluation history
    pub fn evaluation_history(&self) -> &[ArchitectureEvaluation] {
        &self.evaluation_history
    }

    /// Get search statistics
    pub fn get_statistics(&self) -> SearchStatistics {
        let mut stats = SearchStatistics::default();

        if !self.evaluation_history.is_empty() {
            let fitnesses: Vec<f32> = self.evaluation_history.iter().map(|e| e.fitness).collect();
            stats.num_evaluations = fitnesses.len();
            stats.best_fitness = fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            stats.average_fitness = fitnesses.iter().sum::<f32>() / fitnesses.len() as f32;
            stats.fitness_std = {
                let variance =
                    fitnesses.iter().map(|f| (f - stats.average_fitness).powi(2)).sum::<f32>()
                        / fitnesses.len() as f32;
                variance.sqrt()
            };
        }

        stats
    }
}

/// Search statistics
#[derive(Debug, Clone, Default)]
pub struct SearchStatistics {
    pub num_evaluations: usize,
    pub best_fitness: f32,
    pub average_fitness: f32,
    pub fitness_std: f32,
    pub convergence_generation: Option<usize>,
}

impl fmt::Display for SearchStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SearchStatistics {{ evaluations: {}, best: {:.4}, avg: {:.4}, std: {:.4} }}",
            self.num_evaluations, self.best_fitness, self.average_fitness, self.fitness_std
        )
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
