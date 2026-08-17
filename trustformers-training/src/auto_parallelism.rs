use crate::distributed::DistributedConfig;
use crate::expert_parallelism::ExpertParallelismConfig;
use crate::parallelism_3d::ParallelismConfig;
use crate::sequence_parallelism::SequenceParallelismConfig;
use crate::tensor_parallelism::TensorParallelismConfig;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use trustformers_core::Model;

/// Automatic Parallelism Selection Configuration
///
/// This system automatically chooses the optimal parallelism strategy based on:
/// - Model architecture and size
/// - Hardware configuration
/// - Memory constraints
/// - Communication bandwidth
/// - Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoParallelismConfig {
    /// Enable automatic parallelism selection
    pub enabled: bool,
    /// Strategy selection algorithm
    pub selection_algorithm: SelectionAlgorithm,
    /// Performance optimization objective
    pub optimization_objective: OptimizationObjective,
    /// Hardware constraints
    pub hardware_constraints: HardwareConstraints,
    /// Model constraints
    pub model_constraints: ModelConstraints,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
    /// Strategy evaluation method
    pub evaluation_method: EvaluationMethod,
    /// Whether to use dynamic adaptation during training
    pub dynamic_adaptation: bool,
    /// Adaptation frequency (number of steps)
    pub adaptation_frequency: usize,
}

impl Default for AutoParallelismConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            selection_algorithm: SelectionAlgorithm::CostBasedOptimization,
            optimization_objective: OptimizationObjective::MinimizeTime,
            hardware_constraints: HardwareConstraints::default(),
            model_constraints: ModelConstraints::default(),
            performance_requirements: PerformanceRequirements::default(),
            evaluation_method: EvaluationMethod::ModelBased,
            dynamic_adaptation: false,
            adaptation_frequency: 1000,
        }
    }
}

/// Selection algorithms for parallelism strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionAlgorithm {
    /// Rule-based selection using heuristics
    RuleBased,
    /// Cost-based optimization
    CostBasedOptimization,
    /// Machine learning-based selection
    MLBased,
    /// Genetic algorithm optimization
    GeneticAlgorithm,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Multi-objective optimization
    MultiObjective,
}

/// Optimization objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimize training time
    MinimizeTime,
    /// Minimize memory usage
    MinimizeMemory,
    /// Minimize communication overhead
    MinimizeCommunication,
    /// Maximize throughput
    MaximizeThroughput,
    /// Maximize efficiency (throughput/resources)
    MaximizeEfficiency,
    /// Multi-objective optimization
    MultiObjective(Vec<OptimizationObjective>),
}

/// Hardware constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConstraints {
    /// Number of available devices
    pub num_devices: usize,
    /// Memory per device (in bytes)
    pub memory_per_device: u64,
    /// Compute capability per device (FLOPS)
    pub compute_per_device: f64,
    /// Inter-device bandwidth (bytes/second)
    pub inter_device_bandwidth: u64,
    /// Intra-node bandwidth (bytes/second)
    pub intra_node_bandwidth: u64,
    /// Network latency (microseconds)
    pub network_latency: f64,
    /// Device types (GPU, TPU, CPU)
    pub device_types: Vec<DeviceType>,
    /// Topology information
    pub topology: NetworkTopology,
}

impl Default for HardwareConstraints {
    fn default() -> Self {
        Self {
            num_devices: 8,
            memory_per_device: 80 * 1024 * 1024 * 1024, // 80GB
            compute_per_device: 312e12,                 // 312 TFLOPS
            inter_device_bandwidth: 600 * 1024 * 1024 * 1024, // 600 GB/s
            intra_node_bandwidth: 900 * 1024 * 1024 * 1024, // 900 GB/s
            network_latency: 5.0,                       // 5 microseconds
            device_types: vec![DeviceType::GPU; 8],
            topology: NetworkTopology::FullyConnected,
        }
    }
}

/// Device types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    GPU,
    TPU,
    CPU,
    Custom(String),
}

/// Network topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkTopology {
    FullyConnected,
    Ring,
    Tree,
    Mesh2D,
    Mesh3D,
    Torus,
    Custom(String),
}

/// Model constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConstraints {
    /// Total number of parameters
    pub num_parameters: u64,
    /// Number of layers
    pub num_layers: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model architecture type
    pub architecture_type: ArchitectureType,
    /// Whether model uses MoE
    pub has_mixture_of_experts: bool,
    /// Number of experts (if MoE)
    pub num_experts: Option<usize>,
}

impl Default for ModelConstraints {
    fn default() -> Self {
        Self {
            num_parameters: 7_000_000_000, // 7B parameters
            num_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            max_sequence_length: 2048,
            vocab_size: 50257,
            architecture_type: ArchitectureType::Transformer,
            has_mixture_of_experts: false,
            num_experts: None,
        }
    }
}

/// Architecture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitectureType {
    Transformer,
    GPT,
    BERT,
    T5,
    MoE,
    ConvNet,
    RNN,
    Custom(String),
}

/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Maximum acceptable training time
    pub max_training_time: Option<Duration>,
    /// Minimum required throughput (samples/second)
    pub min_throughput: Option<f64>,
    /// Maximum memory usage per device
    pub max_memory_per_device: Option<u64>,
    /// Maximum communication overhead percentage
    pub max_communication_overhead: Option<f32>,
    /// Minimum efficiency requirement
    pub min_efficiency: Option<f32>,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_training_time: None,
            min_throughput: None,
            max_memory_per_device: None,
            max_communication_overhead: Some(0.3), // 30%
            min_efficiency: Some(0.7),             // 70%
        }
    }
}

/// Evaluation methods for parallelism strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvaluationMethod {
    /// Model-based evaluation using analytical models
    ModelBased,
    /// Simulation-based evaluation
    SimulationBased,
    /// Profiling-based evaluation (run small experiments)
    ProfilingBased,
    /// Hybrid approach
    Hybrid,
}

/// Parallelism strategy recommendation
#[derive(Debug, Clone)]
pub struct ParallelismStrategy {
    /// Strategy identifier
    pub strategy_id: String,
    /// Data parallelism configuration
    pub data_parallel: Option<DistributedConfig>,
    /// 3D parallelism configuration
    pub parallelism_3d: Option<ParallelismConfig>,
    /// Expert parallelism configuration
    pub expert_parallel: Option<ExpertParallelismConfig>,
    /// Sequence parallelism configuration
    pub sequence_parallel: Option<SequenceParallelismConfig>,
    /// Tensor parallelism configuration
    pub tensor_parallel: Option<TensorParallelismConfig>,
    /// Expected performance metrics
    pub expected_performance: PerformanceMetrics,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Rationale for this strategy
    pub rationale: String,
}

/// Performance metrics for evaluation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Expected training time per step
    pub time_per_step: Duration,
    /// Expected memory usage per device
    pub memory_per_device: u64,
    /// Expected communication overhead
    pub communication_overhead: f32,
    /// Expected throughput (samples/second)
    pub throughput: f64,
    /// Expected efficiency score
    pub efficiency: f32,
    /// Expected scalability factor
    pub scalability: f32,
}

/// Features extracted for ML-based strategy prediction
#[derive(Debug, Clone)]
pub struct MLFeatures {
    // Model features (log-transformed for better ML performance)
    pub log_num_parameters: f64,
    pub num_layers: f64,
    pub log_hidden_size: f64,
    pub num_attention_heads: f64,
    pub log_sequence_length: f64,
    pub log_vocab_size: f64,
    pub has_moe: f64, // 0.0 or 1.0

    // Hardware features (log-transformed)
    pub log_num_devices: f64,
    pub log_memory_per_device: f64,
    pub log_compute_per_device: f64,
    pub log_bandwidth: f64,
    pub network_latency: f64,

    // Derived features for better prediction
    pub memory_to_compute_ratio: f64,
    pub parameters_per_device: f64,
    pub communication_intensity: f64,
}

/// Individual in genetic algorithm population for strategy optimization
#[derive(Debug, Clone)]
pub struct GeneticIndividual {
    /// Parallelism strategy
    pub strategy: ParallelismStrategy,
    /// Fitness score (higher is better)
    pub fitness: f32,
    /// Data parallelism size
    pub dp_size: usize,
    /// Model parallelism size
    pub mp_size: usize,
    /// Pipeline parallelism size
    pub pp_size: usize,
}

/// Automatic parallelism selector
pub struct AutoParallelismSelector {
    config: AutoParallelismConfig,
    strategy_cache: HashMap<String, ParallelismStrategy>,
    performance_history: Vec<(ParallelismStrategy, PerformanceMetrics)>,
    current_strategy: Option<ParallelismStrategy>,
}

impl AutoParallelismSelector {
    /// Create a new automatic parallelism selector
    pub fn new(config: AutoParallelismConfig) -> Self {
        Self {
            config,
            strategy_cache: HashMap::new(),
            performance_history: Vec::new(),
            current_strategy: None,
        }
    }

    /// Select the optimal parallelism strategy
    pub fn select_strategy(&mut self) -> Result<ParallelismStrategy> {
        let strategies = self.generate_candidate_strategies()?;
        let evaluated_strategies = self.evaluate_strategies(strategies)?;
        let optimal_strategy = self.select_optimal_strategy(evaluated_strategies)?;

        self.current_strategy = Some(optimal_strategy.clone());
        Ok(optimal_strategy)
    }

    /// Generate candidate parallelism strategies
    fn generate_candidate_strategies(&self) -> Result<Vec<ParallelismStrategy>> {
        let mut strategies = Vec::new();

        // Generate strategies based on the selection algorithm
        match self.config.selection_algorithm {
            SelectionAlgorithm::RuleBased => {
                strategies.extend(self.generate_rule_based_strategies()?);
            },
            SelectionAlgorithm::CostBasedOptimization => {
                strategies.extend(self.generate_cost_based_strategies()?);
            },
            SelectionAlgorithm::MLBased => {
                strategies.extend(self.generate_ml_based_strategies()?);
            },
            SelectionAlgorithm::GeneticAlgorithm => {
                strategies.extend(self.generate_genetic_strategies()?);
            },
            SelectionAlgorithm::SimulatedAnnealing => {
                strategies.extend(self.generate_annealing_strategies()?);
            },
            SelectionAlgorithm::MultiObjective => {
                strategies.extend(self.generate_multi_objective_strategies()?);
            },
        }

        Ok(strategies)
    }

    /// Generate rule-based strategies using heuristics
    fn generate_rule_based_strategies(&self) -> Result<Vec<ParallelismStrategy>> {
        let mut strategies = Vec::new();
        let hardware = &self.config.hardware_constraints;
        let model = &self.config.model_constraints;

        // Rule 1: Small models -> Data parallelism only
        if model.num_parameters < 1_000_000_000 {
            // < 1B parameters
            strategies.push(self.create_data_parallel_strategy()?);
        }

        // Rule 2: Large models -> 3D parallelism
        if model.num_parameters > 10_000_000_000 {
            // > 10B parameters
            strategies.push(self.create_3d_parallel_strategy()?);
        }

        // Rule 3: MoE models -> Expert parallelism
        if model.has_mixture_of_experts {
            strategies.push(self.create_expert_parallel_strategy()?);
        }

        // Rule 4: Long sequences -> Sequence parallelism
        if model.max_sequence_length > 8192 {
            strategies.push(self.create_sequence_parallel_strategy()?);
        }

        // Rule 5: Wide models -> Tensor parallelism
        if model.hidden_size > 8192 {
            strategies.push(self.create_tensor_parallel_strategy()?);
        }

        // Rule 6: Many devices -> Hybrid parallelism
        if hardware.num_devices > 16 {
            strategies.push(self.create_hybrid_strategy()?);
        }

        // Fallback: If no rules matched, provide a sensible default
        if strategies.is_empty() {
            // For medium-sized models (1B-10B params), use data parallelism as default
            strategies.push(self.create_data_parallel_strategy()?);
        }

        Ok(strategies)
    }

    /// Generate cost-based optimization strategies
    fn generate_cost_based_strategies(&self) -> Result<Vec<ParallelismStrategy>> {
        let mut strategies = Vec::new();

        // Enumerate different parallelism combinations and estimate costs
        let dp_sizes = vec![1, 2, 4, 8];
        let mp_sizes = vec![1, 2, 4];
        let pp_sizes = vec![1, 2, 4];

        for dp in &dp_sizes {
            for mp in &mp_sizes {
                for pp in &pp_sizes {
                    if dp * mp * pp <= self.config.hardware_constraints.num_devices {
                        let strategy = self.create_3d_strategy_with_config(*dp, *mp, *pp)?;
                        strategies.push(strategy);
                    }
                }
            }
        }

        Ok(strategies)
    }

    /// Generate ML-based strategies using learned patterns
    fn generate_ml_based_strategies(&self) -> Result<Vec<ParallelismStrategy>> {
        // Extract features for ML model
        let features = self.extract_ml_features()?;

        // Use decision tree-based strategy prediction
        let predicted_strategies = self.predict_strategies_with_ml(&features)?;

        // If we have performance history, use it to refine predictions
        if !self.performance_history.is_empty() {
            return self.refine_strategies_with_history(predicted_strategies);
        }

        Ok(predicted_strategies)
    }

    /// Extract features for ML-based strategy prediction
    fn extract_ml_features(&self) -> Result<MLFeatures> {
        let hardware = &self.config.hardware_constraints;
        let model = &self.config.model_constraints;

        Ok(MLFeatures {
            // Model characteristics
            log_num_parameters: (model.num_parameters as f64).log10(),
            num_layers: model.num_layers as f64,
            log_hidden_size: (model.hidden_size as f64).log10(),
            num_attention_heads: model.num_attention_heads as f64,
            log_sequence_length: (model.max_sequence_length as f64).log10(),
            log_vocab_size: (model.vocab_size as f64).log10(),
            has_moe: if model.has_mixture_of_experts { 1.0 } else { 0.0 },

            // Hardware characteristics
            log_num_devices: (hardware.num_devices as f64).log10(),
            log_memory_per_device: (hardware.memory_per_device as f64).log10(),
            log_compute_per_device: hardware.compute_per_device.log10(),
            log_bandwidth: (hardware.inter_device_bandwidth as f64).log10(),
            network_latency: hardware.network_latency,

            // Derived features
            memory_to_compute_ratio: (hardware.memory_per_device as f64)
                / hardware.compute_per_device,
            parameters_per_device: (model.num_parameters as f64) / (hardware.num_devices as f64),
            communication_intensity: (model.hidden_size * model.num_attention_heads) as f64
                / (hardware.inter_device_bandwidth as f64 / 1e9), // GB/s
        })
    }

    /// Predict parallelism strategies using ML model (decision tree approach)
    fn predict_strategies_with_ml(
        &self,
        features: &MLFeatures,
    ) -> Result<Vec<ParallelismStrategy>> {
        let mut strategies = Vec::new();

        // Simple decision tree-based prediction
        // Node 1: Check model size
        if features.log_num_parameters < 9.0 {
            // < 1B parameters
            // Small model branch
            if features.log_num_devices < 1.0 {
                // < 10 devices
                strategies.push(self.create_data_parallel_strategy()?);
            } else {
                strategies.push(self.create_data_parallel_strategy()?);
                if features.log_hidden_size > 3.5 {
                    // > ~3000 hidden size
                    strategies.push(self.create_tensor_parallel_strategy()?);
                }
            }
        } else if features.log_num_parameters < 10.3 {
            // 1B-20B parameters
            // Medium model branch
            if features.log_num_devices < 0.9 {
                // < 8 devices
                strategies.push(self.create_data_parallel_strategy()?);
                if features.log_hidden_size > 3.6 {
                    strategies.push(self.create_tensor_parallel_strategy()?);
                }
            } else {
                strategies.push(self.create_3d_parallel_strategy()?);
                if features.has_moe > 0.5 {
                    strategies.push(self.create_expert_parallel_strategy()?);
                }
            }
        } else {
            // > 20B parameters
            // Large model branch
            strategies.push(self.create_3d_parallel_strategy()?);
            if features.log_num_devices > 1.2 {
                // > 15 devices
                strategies.push(self.create_hybrid_strategy()?);
            }
            if features.has_moe > 0.5 {
                strategies.push(self.create_expert_parallel_strategy()?);
            }
            if features.log_sequence_length > 3.9 {
                // > 8000 sequence length
                strategies.push(self.create_sequence_parallel_strategy()?);
            }
        }

        // Additional heuristics based on communication characteristics
        if features.communication_intensity > 0.1 {
            // High communication intensity -> prefer local parallelism
            if !strategies.iter().any(|s| s.strategy_id.contains("tensor_parallel")) {
                strategies.push(self.create_tensor_parallel_strategy()?);
            }
        }

        // Memory pressure heuristic
        if features.parameters_per_device > 10e9 {
            // > 10B parameters per device
            if !strategies.iter().any(|s| s.strategy_id.contains("3d_parallel")) {
                strategies.push(self.create_3d_parallel_strategy()?);
            }
        }

        Ok(strategies)
    }

    /// Refine strategy predictions using performance history
    fn refine_strategies_with_history(
        &self,
        mut strategies: Vec<ParallelismStrategy>,
    ) -> Result<Vec<ParallelismStrategy>> {
        // Analyze performance history to adjust strategy scores
        let mut strategy_performance_map: HashMap<String, Vec<f32>> = HashMap::new();

        for (historical_strategy, historical_performance) in &self.performance_history {
            let performance_score = self.calculate_performance_score(historical_performance);
            strategy_performance_map
                .entry(historical_strategy.strategy_id.clone())
                .or_default()
                .push(performance_score);
        }

        // Adjust confidence scores based on historical performance
        for strategy in &mut strategies {
            if let Some(historical_scores) = strategy_performance_map.get(&strategy.strategy_id) {
                let avg_score =
                    historical_scores.iter().sum::<f32>() / historical_scores.len() as f32;

                // Boost confidence for historically good strategies
                if avg_score > 0.8 {
                    strategy.confidence = (strategy.confidence + 0.2).min(1.0);
                } else if avg_score < 0.5 {
                    strategy.confidence = (strategy.confidence - 0.2).max(0.1);
                }
            }
        }

        // Sort by confidence and return top strategies
        strategies.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(strategies)
    }

    /// Calculate performance score from metrics (0.0 to 1.0)
    fn calculate_performance_score(&self, metrics: &PerformanceMetrics) -> f32 {
        let time_score = 1.0 / (metrics.time_per_step.as_secs_f32() + 1e-6);
        let memory_score = 1.0 / (metrics.memory_per_device as f32 / 1e9 + 1e-6);
        let comm_score = 1.0 - metrics.communication_overhead.clamp(0.0, 1.0);
        let throughput_score = (metrics.throughput as f32).min(10.0) / 10.0;
        let efficiency_score = metrics.efficiency;

        // Weighted average of scores
        (time_score * 0.25
            + memory_score * 0.15
            + comm_score * 0.2
            + throughput_score * 0.2
            + efficiency_score * 0.2)
            .clamp(0.0, 1.0)
    }

    /// Generate genetic algorithm strategies for parallelism optimization
    fn generate_genetic_strategies(&self) -> Result<Vec<ParallelismStrategy>> {
        let population_size = 20;
        let generations = 10;
        let mutation_rate = 0.2;
        let elite_size = 4;

        // Initialize population with random strategies
        let mut population = self.initialize_genetic_population(population_size)?;

        // Evolve population through generations
        for _generation in 0..generations {
            // Evaluate fitness for all individuals
            self.evaluate_genetic_fitness(&mut population)?;

            // Sort by fitness (higher is better)
            population.sort_by(|a, b| {
                b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Create new generation
            let mut new_population = Vec::new();

            // Keep elite individuals
            for i in 0..elite_size.min(population.len()) {
                new_population.push(population[i].clone());
            }

            // Generate offspring through crossover and mutation
            while new_population.len() < population_size {
                let parent1 = self.tournament_selection(&population, 3)?;
                let parent2 = self.tournament_selection(&population, 3)?;

                let mut offspring = self.crossover_genetic_individual(parent1, parent2)?;

                if fastrand::f32() < mutation_rate {
                    self.mutate_genetic_individual(&mut offspring)?;
                }

                new_population.push(offspring);
            }

            population = new_population;
        }

        // Return top strategies from final generation
        self.evaluate_genetic_fitness(&mut population)?;
        population
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        Ok(population.into_iter().take(5).map(|gi| gi.strategy).collect())
    }

    /// Initialize genetic algorithm population with random strategy configurations
    fn initialize_genetic_population(&self, size: usize) -> Result<Vec<GeneticIndividual>> {
        let mut population = Vec::new();
        let max_devices = self.config.hardware_constraints.num_devices;

        for _ in 0..size {
            // Generate random parallelism configuration
            let dp_size = 1 << fastrand::usize(0..4); // 1, 2, 4, 8
            let mp_size = 1 << fastrand::usize(0..3); // 1, 2, 4
            let pp_size = max_devices / (dp_size * mp_size).max(1);

            let strategy = if pp_size > 1 {
                self.create_3d_strategy_with_config(dp_size, mp_size, pp_size)?
            } else if mp_size > 1 {
                self.create_tensor_parallel_strategy()?
            } else {
                self.create_data_parallel_strategy()?
            };

            population.push(GeneticIndividual {
                strategy,
                fitness: 0.0,
                dp_size,
                mp_size,
                pp_size,
            });
        }

        Ok(population)
    }

    /// Evaluate fitness for genetic individuals
    fn evaluate_genetic_fitness(&self, population: &mut [GeneticIndividual]) -> Result<()> {
        for individual in population {
            individual.fitness = self.calculate_strategy_fitness(&individual.strategy);
        }
        Ok(())
    }

    /// Calculate fitness score for a strategy (higher is better)
    fn calculate_strategy_fitness(&self, strategy: &ParallelismStrategy) -> f32 {
        let metrics = &strategy.expected_performance;

        // Multi-objective fitness function
        let time_fitness = 1.0 / (metrics.time_per_step.as_secs_f32() + 1e-6);
        let memory_fitness = 1.0 / (metrics.memory_per_device as f32 / 1e9 + 1e-6);
        let comm_fitness = 1.0 - metrics.communication_overhead.clamp(0.0, 1.0);
        let throughput_fitness = (metrics.throughput as f32).min(10.0);
        let efficiency_fitness = metrics.efficiency;

        // Weighted combination based on optimization objective
        match &self.config.optimization_objective {
            OptimizationObjective::MinimizeTime => time_fitness,
            OptimizationObjective::MinimizeMemory => memory_fitness,
            OptimizationObjective::MinimizeCommunication => comm_fitness,
            OptimizationObjective::MaximizeThroughput => throughput_fitness,
            OptimizationObjective::MaximizeEfficiency => efficiency_fitness,
            OptimizationObjective::MultiObjective(_) => {
                (time_fitness
                    + memory_fitness
                    + comm_fitness
                    + throughput_fitness
                    + efficiency_fitness)
                    / 5.0
            },
        }
    }

    /// Tournament selection for genetic algorithm
    fn tournament_selection<'a>(
        &self,
        population: &'a [GeneticIndividual],
        tournament_size: usize,
    ) -> Result<&'a GeneticIndividual> {
        let mut best_individual = &population[fastrand::usize(0..population.len())];

        for _ in 1..tournament_size {
            let candidate = &population[fastrand::usize(0..population.len())];
            if candidate.fitness > best_individual.fitness {
                best_individual = candidate;
            }
        }

        Ok(best_individual)
    }

    /// Crossover operation for genetic individuals
    fn crossover_genetic_individual(
        &self,
        parent1: &GeneticIndividual,
        parent2: &GeneticIndividual,
    ) -> Result<GeneticIndividual> {
        // Single-point crossover on parallelism dimensions
        let dp_size = if fastrand::bool() { parent1.dp_size } else { parent2.dp_size };
        let mp_size = if fastrand::bool() { parent1.mp_size } else { parent2.mp_size };
        let pp_size = if fastrand::bool() { parent1.pp_size } else { parent2.pp_size };

        // Ensure valid configuration
        let total_devices = dp_size * mp_size * pp_size;
        let max_devices = self.config.hardware_constraints.num_devices;

        if total_devices <= max_devices {
            let strategy = self.create_3d_strategy_with_config(dp_size, mp_size, pp_size)?;
            Ok(GeneticIndividual {
                strategy,
                fitness: 0.0,
                dp_size,
                mp_size,
                pp_size,
            })
        } else {
            // If invalid, return a copy of the fitter parent
            Ok(if parent1.fitness > parent2.fitness {
                parent1.clone()
            } else {
                parent2.clone()
            })
        }
    }

    /// Mutation operation for genetic individuals
    fn mutate_genetic_individual(&self, individual: &mut GeneticIndividual) -> Result<()> {
        let max_devices = self.config.hardware_constraints.num_devices;

        // Randomly mutate one of the parallelism dimensions
        match fastrand::usize(0..3) {
            0 => {
                // Mutate data parallelism
                let new_dp = (individual.dp_size * 2).min(max_devices);
                if new_dp * individual.mp_size * individual.pp_size <= max_devices {
                    individual.dp_size = new_dp;
                }
            },
            1 => {
                // Mutate model parallelism
                let new_mp = (individual.mp_size * 2).min(8);
                if individual.dp_size * new_mp * individual.pp_size <= max_devices {
                    individual.mp_size = new_mp;
                }
            },
            2 => {
                // Mutate pipeline parallelism
                let new_pp = (individual.pp_size * 2).min(max_devices);
                if individual.dp_size * individual.mp_size * new_pp <= max_devices {
                    individual.pp_size = new_pp;
                }
            },
            _ => {},
        }

        // Recreate strategy with new configuration
        individual.strategy = self.create_3d_strategy_with_config(
            individual.dp_size,
            individual.mp_size,
            individual.pp_size,
        )?;
        individual.fitness = 0.0; // Reset fitness for re-evaluation

        Ok(())
    }

    /// Generate strategies with simulated annealing over the `(dp, mp, pp)` lattice.
    ///
    /// Starting from the smallest feasible configuration, the search repeatedly proposes a
    /// neighbour (one axis doubled or halved), accepts it outright when it compares better
    /// under [`AutoParallelismSelector::compare_strategies`], and otherwise accepts it with
    /// probability `exp(-Δ/T)`. The temperature decays geometrically, so the walk anneals from
    /// exploration to exploitation. The PRNG is seeded deterministically from the hardware and
    /// model constraints, which makes the search reproducible for a given problem.
    ///
    /// The returned vector is the set of distinct configurations visited — including the best
    /// one found — so the downstream evaluation step still sees a population to rank.
    fn generate_annealing_strategies(&self) -> Result<Vec<ParallelismStrategy>> {
        let max_devices = self.config.hardware_constraints.num_devices.max(1);

        // Deterministic seed derived from the problem definition.
        let mut rng_state = (max_devices as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (self.config.model_constraints.num_parameters as u64)
            ^ (self.config.model_constraints.num_layers as u64).wrapping_mul(0x1000_0000_01b3);
        let mut next_unit = || -> f64 {
            rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64) / ((1u64 << 53) as f64)
        };

        let mut current = (1usize, 1usize, 1usize);
        let mut current_strategy =
            self.create_3d_strategy_with_config(current.0, current.1, current.2)?;
        let mut best_strategy = current_strategy.clone();

        let mut visited: Vec<(usize, usize, usize)> = vec![current];
        let mut population = vec![current_strategy.clone()];

        let mut temperature = 1.0f64;
        const COOLING: f64 = 0.9;
        const ITERATIONS: usize = 64;

        for _ in 0..ITERATIONS {
            // Propose a neighbour: pick an axis, then double or halve it.
            let axis = (next_unit() * 3.0) as usize % 3;
            let grow = next_unit() < 0.5;
            let mut candidate = current;
            let slot = match axis {
                0 => &mut candidate.0,
                1 => &mut candidate.1,
                _ => &mut candidate.2,
            };
            if grow {
                *slot = (*slot).saturating_mul(2);
            } else {
                *slot = (*slot / 2).max(1);
            }

            if candidate.0 * candidate.1 * candidate.2 > max_devices {
                continue;
            }

            let candidate_strategy =
                self.create_3d_strategy_with_config(candidate.0, candidate.1, candidate.2)?;
            let better = self.compare_strategies(&candidate_strategy, &current_strategy)?
                == std::cmp::Ordering::Less;

            let accept = if better {
                true
            } else {
                // Metropolis acceptance on the normalised objective gap.
                let delta = (self.calculate_multi_objective_score(&current_strategy)
                    - self.calculate_multi_objective_score(&candidate_strategy))
                .abs() as f64;
                next_unit() < (-delta / temperature.max(1e-6)).exp()
            };

            if accept {
                current = candidate;
                current_strategy = candidate_strategy.clone();
                if !visited.contains(&candidate) {
                    visited.push(candidate);
                    population.push(candidate_strategy.clone());
                }
                if self.compare_strategies(&current_strategy, &best_strategy)?
                    == std::cmp::Ordering::Less
                {
                    best_strategy = current_strategy.clone();
                }
            }

            temperature *= COOLING;
        }

        // Guarantee the best configuration is present even if it was pruned above.
        if !population.iter().any(|s| s.strategy_id == best_strategy.strategy_id) {
            population.push(best_strategy);
        }

        Ok(population)
    }

    /// Generate the Pareto-optimal subset of the cost-based candidates.
    ///
    /// The three competing objectives are throughput (maximised), memory per device
    /// (minimised) and communication overhead (minimised). A candidate is kept unless another
    /// candidate is at least as good on all three and strictly better on one — the standard
    /// non-domination test. Unlike a weighted sum this makes no assumption about the relative
    /// importance of the objectives, which is the point of multi-objective search.
    fn generate_multi_objective_strategies(&self) -> Result<Vec<ParallelismStrategy>> {
        let candidates = self.generate_cost_based_strategies()?;
        if candidates.len() <= 1 {
            return Ok(candidates);
        }

        // Objective vector, all in "smaller is better" form.
        let objectives: Vec<[f64; 3]> = candidates
            .iter()
            .map(|s| {
                [
                    -s.expected_performance.throughput,
                    s.expected_performance.memory_per_device as f64,
                    s.expected_performance.communication_overhead as f64,
                ]
            })
            .collect();

        let dominates = |a: &[f64; 3], b: &[f64; 3]| -> bool {
            let no_worse = a.iter().zip(b.iter()).all(|(x, y)| x <= y);
            let strictly_better = a.iter().zip(b.iter()).any(|(x, y)| x < y);
            no_worse && strictly_better
        };

        let mut front = Vec::new();
        for (i, candidate) in candidates.iter().enumerate() {
            let dominated = objectives
                .iter()
                .enumerate()
                .any(|(j, other)| j != i && dominates(other, &objectives[i]));
            if !dominated {
                front.push(candidate.clone());
            }
        }

        if front.is_empty() {
            // Every candidate is mutually dominated only if the objective vectors are
            // degenerate; fall back to the full set rather than returning nothing.
            return Ok(candidates);
        }
        Ok(front)
    }

    /// Create data parallelism strategy
    fn create_data_parallel_strategy(&self) -> Result<ParallelismStrategy> {
        let data_parallel = Some(DistributedConfig {
            world_size: self.config.hardware_constraints.num_devices,
            rank: 0,
            backend: crate::distributed::DistributedBackend::NCCL,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            gradient_compression: false,
            bucket_size_mb: 25,
        });

        let expected_performance = self.estimate_performance_data_parallel()?;

        Ok(ParallelismStrategy {
            strategy_id: "data_parallel".to_string(),
            data_parallel,
            parallelism_3d: None,
            expert_parallel: None,
            sequence_parallel: None,
            tensor_parallel: None,
            expected_performance,
            confidence: 0.9,
            rationale: "Model size suitable for data parallelism".to_string(),
        })
    }

    /// Create 3D parallelism strategy
    fn create_3d_parallel_strategy(&self) -> Result<ParallelismStrategy> {
        let num_devices = self.config.hardware_constraints.num_devices;

        // Simple heuristic for 3D parallelism dimensions
        let dp_size = std::cmp::min(4, num_devices);
        let mp_size = std::cmp::min(2, num_devices / dp_size);
        let pp_size = num_devices / (dp_size * mp_size);

        self.create_3d_strategy_with_config(dp_size, mp_size, pp_size)
    }

    /// Create 3D parallelism strategy with specific configuration
    fn create_3d_strategy_with_config(
        &self,
        dp_size: usize,
        mp_size: usize,
        pp_size: usize,
    ) -> Result<ParallelismStrategy> {
        let parallelism_3d = Some(ParallelismConfig {
            dp_size,
            mp_size,
            pp_size,
            num_micro_batches: 4,
            gradient_accumulation: true,
            accumulation_steps: 1,
            activation_checkpointing: true,
            comm_backend: crate::parallelism_3d::CommBackend::NCCL,
            pipeline_schedule: crate::parallelism_3d::PipelineSchedule::GPipe,
            memory_optimization: crate::parallelism_3d::MemoryOptimization::Medium,
        });

        let expected_performance =
            self.estimate_performance_3d_parallel(dp_size, mp_size, pp_size)?;

        Ok(ParallelismStrategy {
            strategy_id: format!("3d_parallel_{}_{}_", dp_size, mp_size),
            data_parallel: None,
            parallelism_3d,
            expert_parallel: None,
            sequence_parallel: None,
            tensor_parallel: None,
            expected_performance,
            confidence: 0.8,
            rationale: format!(
                "Large model requiring 3D parallelism: DP={}, MP={}, PP={}",
                dp_size, mp_size, pp_size
            ),
        })
    }

    /// Create expert parallelism strategy
    fn create_expert_parallel_strategy(&self) -> Result<ParallelismStrategy> {
        let num_experts = self.config.model_constraints.num_experts.unwrap_or(8);
        let expert_parallel_size =
            std::cmp::min(num_experts, self.config.hardware_constraints.num_devices);

        let expert_parallel = Some(ExpertParallelismConfig {
            num_experts,
            experts_per_device: num_experts / expert_parallel_size,
            expert_parallel_size,
            top_k: 2,
            load_balancing: crate::expert_parallelism::LoadBalancingStrategy::TokenChoiceBased,
            routing_strategy: crate::expert_parallelism::ExpertRoutingStrategy::LearnedGating,
            capacity_factor: 1.25,
            drop_tokens: false,
            use_auxiliary_loss: true,
            auxiliary_loss_weight: 0.01,
            communication_pattern: crate::expert_parallelism::ExpertCommunicationPattern::AllToAll,
        });

        let expected_performance = self.estimate_performance_expert_parallel()?;

        Ok(ParallelismStrategy {
            strategy_id: "expert_parallel".to_string(),
            data_parallel: None,
            parallelism_3d: None,
            expert_parallel,
            sequence_parallel: None,
            tensor_parallel: None,
            expected_performance,
            confidence: 0.85,
            rationale: "MoE model requiring expert parallelism".to_string(),
        })
    }

    /// Create sequence parallelism strategy
    fn create_sequence_parallel_strategy(&self) -> Result<ParallelismStrategy> {
        let sequence_parallel_size = std::cmp::min(4, self.config.hardware_constraints.num_devices);
        let max_seq_per_device =
            self.config.model_constraints.max_sequence_length / sequence_parallel_size;

        let sequence_parallel = Some(SequenceParallelismConfig {
            sequence_parallel_size,
            max_sequence_length_per_device: max_seq_per_device,
            overlap_size: std::cmp::min(128, max_seq_per_device / 10),
            attention_communication_opt: true,
            communication_pattern:
                crate::sequence_parallelism::SequenceCommunicationPattern::RingAllReduce,
            splitting_strategy: crate::sequence_parallelism::SequenceSplittingStrategy::EqualChunks,
            sync_gradients: true,
            memory_optimization: crate::sequence_parallelism::SequenceMemoryOptimization::Medium,
            use_checkpointing: true,
        });

        let expected_performance = self.estimate_performance_sequence_parallel()?;

        Ok(ParallelismStrategy {
            strategy_id: "sequence_parallel".to_string(),
            data_parallel: None,
            parallelism_3d: None,
            expert_parallel: None,
            sequence_parallel,
            tensor_parallel: None,
            expected_performance,
            confidence: 0.8,
            rationale: "Long sequences requiring sequence parallelism".to_string(),
        })
    }

    /// Create tensor parallelism strategy
    fn create_tensor_parallel_strategy(&self) -> Result<ParallelismStrategy> {
        let tensor_parallel_size = std::cmp::min(4, self.config.hardware_constraints.num_devices);

        let tensor_parallel = Some(TensorParallelismConfig {
            tensor_parallel_size,
            partitioning_strategy:
                crate::tensor_parallelism::TensorPartitioningStrategy::ColumnWise,
            column_parallel: true,
            row_parallel: true,
            communication_pattern: crate::tensor_parallelism::TensorCommunicationPattern::AllReduce,
            async_communication: true,
            fusion_threshold_bytes: 1024 * 1024,
            gradient_accumulation: true,
            memory_optimization: crate::tensor_parallelism::TensorMemoryOptimization::Medium,
            mixed_precision: false,
        });

        let expected_performance = self.estimate_performance_tensor_parallel()?;

        Ok(ParallelismStrategy {
            strategy_id: "tensor_parallel".to_string(),
            data_parallel: None,
            parallelism_3d: None,
            expert_parallel: None,
            sequence_parallel: None,
            tensor_parallel,
            expected_performance,
            confidence: 0.85,
            rationale: "Wide model requiring tensor parallelism".to_string(),
        })
    }

    /// Create hybrid parallelism strategy
    fn create_hybrid_strategy(&self) -> Result<ParallelismStrategy> {
        let num_devices = self.config.hardware_constraints.num_devices;

        // Hybrid strategy combining multiple parallelism types
        let dp_size = 2;
        let mp_size = 2;
        let pp_size = num_devices / (dp_size * mp_size);

        let parallelism_3d = Some(ParallelismConfig {
            dp_size,
            mp_size,
            pp_size,
            num_micro_batches: 4,
            gradient_accumulation: true,
            accumulation_steps: 1,
            activation_checkpointing: true,
            comm_backend: crate::parallelism_3d::CommBackend::NCCL,
            pipeline_schedule: crate::parallelism_3d::PipelineSchedule::GPipe,
            memory_optimization: crate::parallelism_3d::MemoryOptimization::High,
        });

        let tensor_parallel = if self.config.model_constraints.hidden_size > 4096 {
            Some(TensorParallelismConfig {
                tensor_parallel_size: mp_size,
                ..Default::default()
            })
        } else {
            None
        };

        let expected_performance = self.estimate_performance_hybrid()?;

        Ok(ParallelismStrategy {
            strategy_id: "hybrid".to_string(),
            data_parallel: None,
            parallelism_3d,
            expert_parallel: None,
            sequence_parallel: None,
            tensor_parallel,
            expected_performance,
            confidence: 0.75,
            rationale: "Complex model and many devices requiring hybrid parallelism".to_string(),
        })
    }

    /// Evaluate parallelism strategies
    fn evaluate_strategies(
        &self,
        strategies: Vec<ParallelismStrategy>,
    ) -> Result<Vec<ParallelismStrategy>> {
        match self.config.evaluation_method {
            EvaluationMethod::ModelBased => self.evaluate_model_based(strategies),
            EvaluationMethod::SimulationBased => self.evaluate_simulation_based(strategies),
            EvaluationMethod::ProfilingBased => self.evaluate_profiling_based(strategies),
            EvaluationMethod::Hybrid => self.evaluate_hybrid(strategies),
        }
    }

    /// Model-based evaluation
    fn evaluate_model_based(
        &self,
        mut strategies: Vec<ParallelismStrategy>,
    ) -> Result<Vec<ParallelismStrategy>> {
        // Update performance estimates based on analytical models
        for strategy in &mut strategies {
            strategy.expected_performance = self.refine_performance_estimate(strategy)?;
            strategy.confidence = self.calculate_confidence(strategy);
        }
        Ok(strategies)
    }

    /// Simulation-based evaluation: roll the pipeline schedule forward micro-batch by
    /// micro-batch instead of using the closed-form estimate.
    ///
    /// For a `pp_size`-stage pipeline running `m` micro-batches, the analytical model charges
    /// only the steady-state cost; the simulation additionally counts the fill and drain
    /// bubbles that a real schedule pays:
    ///
    /// ```text
    /// GPipe      : total = (m + pp - 1) · t_stage          (fill and drain are serial)
    /// OneForwardOneBackward / Interleaved:
    ///              total = (m + pp - 1) · t_stage, with the bubble halved by overlapping
    ///              the backward of micro-batch i with the forward of micro-batch i+1
    /// ```
    ///
    /// The simulated `time_per_step` therefore differs from — and is never better than — the
    /// analytical one, and `throughput`, `efficiency` and `confidence` are recomputed from it.
    fn evaluate_simulation_based(
        &self,
        mut strategies: Vec<ParallelismStrategy>,
    ) -> Result<Vec<ParallelismStrategy>> {
        for strategy in &mut strategies {
            let base = self.refine_performance_estimate(strategy)?;
            let simulated = self.simulate_step_time(strategy, &base);

            let base_secs = base.time_per_step.as_secs_f64().max(1e-9);
            let ratio = simulated.as_secs_f64() / base_secs;

            strategy.expected_performance = PerformanceMetrics {
                time_per_step: simulated,
                throughput: base.throughput / ratio.max(1e-9),
                efficiency: (base.efficiency as f64 / ratio.max(1e-9)) as f32,
                ..base
            };
            // A timeline simulation is a stronger evidence source than the closed form, but
            // it is still a model: keep the analytical confidence.
            strategy.confidence = self.calculate_confidence(strategy);
            strategy.rationale = format!(
                "{} [simulated pipeline schedule: {:.1}% bubble overhead]",
                strategy.rationale,
                (ratio - 1.0).max(0.0) * 100.0
            );
        }
        Ok(strategies)
    }

    /// Simulate the wall-clock time of one optimizer step for `strategy`.
    ///
    /// Returns the analytical time unchanged for strategies without a pipeline dimension —
    /// there is no schedule to roll forward in that case.
    fn simulate_step_time(
        &self,
        strategy: &ParallelismStrategy,
        base: &PerformanceMetrics,
    ) -> Duration {
        let Some(config) = strategy.parallelism_3d.as_ref() else {
            return base.time_per_step;
        };
        let pp = config.pp_size.max(1);
        if pp == 1 {
            return base.time_per_step;
        }

        let micro_batches = config.num_micro_batches.max(1);
        // Per-stage cost implied by the analytical steady-state estimate.
        let stage_secs = base.time_per_step.as_secs_f64() / micro_batches as f64;

        // Fill + drain bubble, halved for schedules that interleave forward and backward.
        let bubble_stages = match config.pipeline_schedule {
            crate::parallelism_3d::PipelineSchedule::GPipe => (pp - 1) as f64,
            _ => (pp - 1) as f64 * 0.5,
        };

        let total_secs = stage_secs * (micro_batches as f64 + bubble_stages);
        Duration::from_secs_f64(total_secs.max(0.0))
    }

    /// Profiling-based evaluation.
    ///
    /// Genuine profiling means executing the candidate strategies on the target cluster and
    /// measuring them. This crate has no way to launch such experiments, and returning
    /// analytical estimates while claiming they came from profiling would be a fabricated
    /// result — so the mode reports that it is unavailable and names the alternatives.
    fn evaluate_profiling_based(
        &self,
        _strategies: Vec<ParallelismStrategy>,
    ) -> Result<Vec<ParallelismStrategy>> {
        Err(anyhow!(
            "EvaluationMethod::ProfilingBased requires executing candidate strategies on the \
             target hardware, which this selector cannot do. Run the candidates yourself and \
             feed the measurements back with `update_performance_history`, or choose \
             EvaluationMethod::ModelBased / SimulationBased / Hybrid."
        ))
    }

    /// Hybrid evaluation: blend the analytical model with the pipeline simulation.
    ///
    /// Each strategy is evaluated both ways and the two estimates are combined
    /// conservatively — the slower `time_per_step`, the lower `throughput` and `efficiency`
    /// — because the two models disagree exactly where one of them is missing a cost. The
    /// confidence is the mean of the two, reduced by the relative disagreement between them,
    /// so a strategy the two methods rank very differently is reported as less certain.
    fn evaluate_hybrid(
        &self,
        strategies: Vec<ParallelismStrategy>,
    ) -> Result<Vec<ParallelismStrategy>> {
        let model_based = self.evaluate_model_based(strategies.clone())?;
        let simulated = self.evaluate_simulation_based(strategies)?;

        let mut blended = Vec::with_capacity(model_based.len());
        for (analytic, sim) in model_based.into_iter().zip(simulated.into_iter()) {
            let analytic_secs = analytic.expected_performance.time_per_step.as_secs_f64();
            let sim_secs = sim.expected_performance.time_per_step.as_secs_f64();
            let disagreement = if analytic_secs > 0.0 {
                ((sim_secs - analytic_secs).abs() / analytic_secs).min(1.0) as f32
            } else {
                0.0
            };

            let performance = PerformanceMetrics {
                time_per_step: analytic
                    .expected_performance
                    .time_per_step
                    .max(sim.expected_performance.time_per_step),
                throughput: analytic
                    .expected_performance
                    .throughput
                    .min(sim.expected_performance.throughput),
                efficiency: analytic
                    .expected_performance
                    .efficiency
                    .min(sim.expected_performance.efficiency),
                memory_per_device: analytic
                    .expected_performance
                    .memory_per_device
                    .max(sim.expected_performance.memory_per_device),
                communication_overhead: analytic
                    .expected_performance
                    .communication_overhead
                    .max(sim.expected_performance.communication_overhead),
                scalability: analytic
                    .expected_performance
                    .scalability
                    .min(sim.expected_performance.scalability),
            };

            let confidence = ((analytic.confidence + sim.confidence) * 0.5 * (1.0 - disagreement))
                .clamp(0.0, 1.0);

            blended.push(ParallelismStrategy {
                expected_performance: performance,
                confidence,
                rationale: format!(
                    "{} [hybrid: analytical and simulated estimates disagree by {:.1}%]",
                    analytic.rationale,
                    disagreement * 100.0
                ),
                ..analytic
            });
        }

        Ok(blended)
    }

    /// Select the optimal strategy from evaluated strategies
    fn select_optimal_strategy(
        &self,
        mut strategies: Vec<ParallelismStrategy>,
    ) -> Result<ParallelismStrategy> {
        if strategies.is_empty() {
            return Err(anyhow!("No strategies available for selection"));
        }

        // Sort strategies based on optimization objective
        strategies
            .sort_by(|a, b| self.compare_strategies(a, b).unwrap_or(std::cmp::Ordering::Equal));

        strategies
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No strategies available for selection"))
    }

    /// Compare strategies based on optimization objective
    fn compare_strategies(
        &self,
        a: &ParallelismStrategy,
        b: &ParallelismStrategy,
    ) -> Result<std::cmp::Ordering> {
        match &self.config.optimization_objective {
            OptimizationObjective::MinimizeTime => {
                Ok(a.expected_performance.time_per_step.cmp(&b.expected_performance.time_per_step))
            },
            OptimizationObjective::MinimizeMemory => Ok(a
                .expected_performance
                .memory_per_device
                .cmp(&b.expected_performance.memory_per_device)),
            OptimizationObjective::MinimizeCommunication => Ok(a
                .expected_performance
                .communication_overhead
                .partial_cmp(&b.expected_performance.communication_overhead)
                .unwrap_or(std::cmp::Ordering::Equal)),
            OptimizationObjective::MaximizeThroughput => Ok(b
                .expected_performance
                .throughput
                .partial_cmp(&a.expected_performance.throughput)
                .unwrap_or(std::cmp::Ordering::Equal)),
            OptimizationObjective::MaximizeEfficiency => Ok(b
                .expected_performance
                .efficiency
                .partial_cmp(&a.expected_performance.efficiency)
                .unwrap_or(std::cmp::Ordering::Equal)),
            OptimizationObjective::MultiObjective(_objectives) => {
                // Simplified multi-objective comparison
                let score_a = self.calculate_multi_objective_score(a);
                let score_b = self.calculate_multi_objective_score(b);
                Ok(score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal))
            },
        }
    }

    /// Calculate multi-objective score
    fn calculate_multi_objective_score(&self, strategy: &ParallelismStrategy) -> f32 {
        // Simplified scoring function
        let time_score = 1.0 / (strategy.expected_performance.time_per_step.as_secs_f32() + 1e-6);
        let memory_score =
            1.0 / (strategy.expected_performance.memory_per_device as f32 / 1e9 + 1e-6);
        let comm_score = 1.0 / (strategy.expected_performance.communication_overhead + 1e-6);
        let throughput_score = strategy.expected_performance.throughput as f32;
        let efficiency_score = strategy.expected_performance.efficiency;

        (time_score + memory_score + comm_score + throughput_score + efficiency_score) / 5.0
    }

    /// Estimate performance for data parallelism
    fn estimate_performance_data_parallel(&self) -> Result<PerformanceMetrics> {
        let model = &self.config.model_constraints;
        let hardware = &self.config.hardware_constraints;

        // Simplified performance estimation
        let params_per_device = model.num_parameters * 4; // 4 bytes per param
        let memory_per_device = params_per_device + 2 * params_per_device; // gradients + optimizer states

        let compute_time = (model.num_parameters as f64 * 2.0) / hardware.compute_per_device; // 2 FLOPs per param
        let communication_time =
            (params_per_device as f64) / hardware.inter_device_bandwidth as f64;
        let total_time = compute_time + communication_time;

        Ok(PerformanceMetrics {
            time_per_step: Duration::from_secs_f64(total_time),
            memory_per_device,
            communication_overhead: communication_time as f32 / total_time as f32,
            throughput: 1.0 / total_time,
            efficiency: 0.8,
            scalability: 0.9,
        })
    }

    /// Estimate performance for 3D parallelism
    fn estimate_performance_3d_parallel(
        &self,
        dp_size: usize,
        mp_size: usize,
        _pp_size: usize,
    ) -> Result<PerformanceMetrics> {
        let model = &self.config.model_constraints;
        let hardware = &self.config.hardware_constraints;

        // Simplified performance estimation for 3D parallelism
        let params_per_device = model.num_parameters / (mp_size as u64);
        let memory_per_device = params_per_device * 4 + 2 * params_per_device;

        let compute_time = (params_per_device as f64 * 2.0) / hardware.compute_per_device;
        let pipeline_bubble = 0.1; // 10% pipeline bubble
        let communication_time = compute_time * 0.2; // 20% communication overhead
        let total_time = compute_time * (1.0 + pipeline_bubble) + communication_time;

        Ok(PerformanceMetrics {
            time_per_step: Duration::from_secs_f64(total_time),
            memory_per_device,
            communication_overhead: communication_time as f32 / total_time as f32,
            throughput: dp_size as f64 / total_time,
            efficiency: 0.85,
            scalability: 0.95,
        })
    }

    /// Estimate performance for expert parallelism
    fn estimate_performance_expert_parallel(&self) -> Result<PerformanceMetrics> {
        let model = &self.config.model_constraints;
        let hardware = &self.config.hardware_constraints;

        let experts_per_device = model.num_experts.unwrap_or(8) / hardware.num_devices;
        let params_per_expert = model.num_parameters / model.num_experts.unwrap_or(8) as u64;
        let memory_per_device = params_per_expert * experts_per_device as u64 * 4;

        let compute_time = (params_per_expert as f64 * 2.0) / hardware.compute_per_device;
        let routing_overhead = 0.1; // 10% routing overhead
        let communication_time = compute_time * 0.15; // 15% communication overhead
        let total_time = compute_time * (1.0 + routing_overhead) + communication_time;

        Ok(PerformanceMetrics {
            time_per_step: Duration::from_secs_f64(total_time),
            memory_per_device,
            communication_overhead: communication_time as f32 / total_time as f32,
            throughput: 1.0 / total_time,
            efficiency: 0.9,
            scalability: 0.95,
        })
    }

    /// Estimate performance for sequence parallelism
    fn estimate_performance_sequence_parallel(&self) -> Result<PerformanceMetrics> {
        let model = &self.config.model_constraints;
        let hardware = &self.config.hardware_constraints;

        let seq_per_device = model.max_sequence_length / hardware.num_devices;
        let memory_per_device = (seq_per_device * model.hidden_size * 4) as u64;

        let compute_time = (model.num_parameters as f64 * 2.0) / hardware.compute_per_device;
        let attention_comm_overhead = 0.2; // 20% attention communication overhead
        let total_time = compute_time * (1.0 + attention_comm_overhead);

        Ok(PerformanceMetrics {
            time_per_step: Duration::from_secs_f64(total_time),
            memory_per_device,
            communication_overhead: attention_comm_overhead as f32,
            throughput: 1.0 / total_time,
            efficiency: 0.8,
            scalability: 0.85,
        })
    }

    /// Estimate performance for tensor parallelism
    fn estimate_performance_tensor_parallel(&self) -> Result<PerformanceMetrics> {
        let model = &self.config.model_constraints;
        let hardware = &self.config.hardware_constraints;

        let params_per_device = model.num_parameters / hardware.num_devices as u64;
        let memory_per_device = params_per_device * 4;

        let compute_time = (params_per_device as f64 * 2.0) / hardware.compute_per_device;
        let tensor_comm_overhead = 0.25; // 25% tensor communication overhead
        let total_time = compute_time * (1.0 + tensor_comm_overhead);

        Ok(PerformanceMetrics {
            time_per_step: Duration::from_secs_f64(total_time),
            memory_per_device,
            communication_overhead: tensor_comm_overhead as f32,
            throughput: 1.0 / total_time,
            efficiency: 0.75,
            scalability: 0.8,
        })
    }

    /// Estimate performance for hybrid parallelism
    fn estimate_performance_hybrid(&self) -> Result<PerformanceMetrics> {
        // Simplified hybrid estimation - combines benefits and overheads
        let base_metrics = self.estimate_performance_3d_parallel(2, 2, 2)?;

        Ok(PerformanceMetrics {
            time_per_step: base_metrics.time_per_step,
            memory_per_device: base_metrics.memory_per_device / 2, // Better memory efficiency
            communication_overhead: base_metrics.communication_overhead * 1.1, // Slightly more overhead
            throughput: base_metrics.throughput * 0.95, // Slight throughput penalty
            efficiency: 0.9,
            scalability: 0.95,
        })
    }

    /// Refine performance estimate using detailed models
    fn refine_performance_estimate(
        &self,
        strategy: &ParallelismStrategy,
    ) -> Result<PerformanceMetrics> {
        // For now, return the existing estimate
        // In practice, would apply more sophisticated modeling
        Ok(strategy.expected_performance.clone())
    }

    /// Calculate confidence score for a strategy
    fn calculate_confidence(&self, strategy: &ParallelismStrategy) -> f32 {
        // Simplified confidence calculation
        let mut confidence: f32 = 0.5;

        // Increase confidence for well-known strategies
        if strategy.strategy_id.contains("data_parallel") {
            confidence += 0.3;
        }
        if strategy.strategy_id.contains("3d_parallel") {
            confidence += 0.2;
        }

        // Decrease confidence for very complex strategies
        if strategy.strategy_id.contains("hybrid") {
            confidence -= 0.1;
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Get current strategy
    pub fn current_strategy(&self) -> Option<&ParallelismStrategy> {
        self.current_strategy.as_ref()
    }

    /// Update performance history
    pub fn update_performance_history(&mut self, actual_performance: PerformanceMetrics) {
        if let Some(current_strategy) = &self.current_strategy {
            self.performance_history.push((current_strategy.clone(), actual_performance));

            // Keep only recent history
            if self.performance_history.len() > 100 {
                self.performance_history.remove(0);
            }
        }
    }

    /// Get configuration
    pub fn config(&self) -> &AutoParallelismConfig {
        &self.config
    }
}

/// Utilities for automatic parallelism selection
pub mod utils {
    use super::*;

    /// Estimate model memory requirements
    pub fn estimate_model_memory(constraints: &ModelConstraints) -> u64 {
        let param_memory = constraints.num_parameters * 4; // 4 bytes per float32
        let gradient_memory = param_memory; // Same size for gradients
        let optimizer_memory = param_memory * 2; // Typical optimizer state

        param_memory + gradient_memory + optimizer_memory
    }

    /// Check if strategy meets performance requirements
    pub fn meets_requirements(
        strategy: &ParallelismStrategy,
        requirements: &PerformanceRequirements,
    ) -> bool {
        if let Some(max_time) = requirements.max_training_time {
            if strategy.expected_performance.time_per_step > max_time {
                return false;
            }
        }

        if let Some(min_throughput) = requirements.min_throughput {
            if strategy.expected_performance.throughput < min_throughput {
                return false;
            }
        }

        if let Some(max_memory) = requirements.max_memory_per_device {
            if strategy.expected_performance.memory_per_device > max_memory {
                return false;
            }
        }

        if let Some(max_comm_overhead) = requirements.max_communication_overhead {
            if strategy.expected_performance.communication_overhead > max_comm_overhead {
                return false;
            }
        }

        if let Some(min_efficiency) = requirements.min_efficiency {
            if strategy.expected_performance.efficiency < min_efficiency {
                return false;
            }
        }

        true
    }

    /// Create hardware constraints from system information
    pub fn detect_hardware_constraints() -> Result<HardwareConstraints> {
        // In practice, would detect actual hardware configuration
        Ok(HardwareConstraints::default())
    }

    /// Create model constraints from model architecture
    pub fn analyze_model_constraints<M: Model>(_model: &M) -> Result<ModelConstraints> {
        // In practice, would analyze the actual model
        Ok(ModelConstraints::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Evaluation modes and strategy generation ─────────────────────────────

    #[test]
    fn test_profiling_based_evaluation_reports_that_it_is_unavailable() {
        // Regression: this mode silently returned analytical estimates while the caller had
        // explicitly asked for measurements.
        let config = AutoParallelismConfig {
            evaluation_method: EvaluationMethod::ProfilingBased,
            ..Default::default()
        };
        let selector = AutoParallelismSelector::new(config);
        let strategies =
            selector.generate_cost_based_strategies().expect("candidate generation failed");

        let err = selector
            .evaluate_profiling_based(strategies)
            .expect_err("profiling must not fabricate measurements");
        let message = err.to_string();
        assert!(
            message.contains("ProfilingBased"),
            "the error must name the mode: {message}"
        );
    }

    #[test]
    fn test_simulation_based_evaluation_differs_from_the_model() {
        // Regression: `evaluate_simulation_based` was `self.evaluate_model_based(..)`.
        let config = AutoParallelismConfig::default();
        let selector = AutoParallelismSelector::new(config);
        let strategy = selector.create_3d_strategy_with_config(1, 1, 4).expect("3d strategy");

        let analytic = selector.evaluate_model_based(vec![strategy.clone()]).expect("model based");
        let simulated =
            selector.evaluate_simulation_based(vec![strategy]).expect("simulation based");

        assert!(
            simulated[0].expected_performance.time_per_step
                > analytic[0].expected_performance.time_per_step,
            "a 4-stage pipeline must pay a bubble in simulation: {:?} vs {:?}",
            simulated[0].expected_performance.time_per_step,
            analytic[0].expected_performance.time_per_step
        );
        assert!(
            simulated[0].expected_performance.throughput
                < analytic[0].expected_performance.throughput,
            "the extra time must lower the throughput"
        );
        assert!(simulated[0].rationale.contains("simulated pipeline schedule"));
    }

    #[test]
    fn test_simulation_matches_the_model_without_a_pipeline_dimension() {
        let selector = AutoParallelismSelector::new(AutoParallelismConfig::default());
        let strategy = selector.create_3d_strategy_with_config(4, 1, 1).expect("3d strategy");

        let analytic = selector.evaluate_model_based(vec![strategy.clone()]).expect("model based");
        let simulated =
            selector.evaluate_simulation_based(vec![strategy]).expect("simulation based");
        assert_eq!(
            simulated[0].expected_performance.time_per_step,
            analytic[0].expected_performance.time_per_step,
            "with pp = 1 there is no schedule to simulate"
        );
    }

    #[test]
    fn test_hybrid_evaluation_is_conservative_and_lowers_confidence_on_disagreement() {
        let selector = AutoParallelismSelector::new(AutoParallelismConfig::default());
        let strategy = selector.create_3d_strategy_with_config(1, 1, 4).expect("3d strategy");

        let analytic = selector.evaluate_model_based(vec![strategy.clone()]).expect("model based");
        let hybrid = selector.evaluate_hybrid(vec![strategy]).expect("hybrid");

        assert!(
            hybrid[0].expected_performance.time_per_step
                >= analytic[0].expected_performance.time_per_step,
            "the hybrid estimate must take the slower of the two"
        );
        assert!(
            hybrid[0].confidence < analytic[0].confidence,
            "disagreement between the two models must reduce the confidence ({} vs {})",
            hybrid[0].confidence,
            analytic[0].confidence
        );
        assert!(hybrid[0].rationale.contains("hybrid"));
    }

    #[test]
    fn test_annealing_search_explores_more_than_one_configuration() {
        // Regression: this delegated verbatim to `generate_cost_based_strategies`.
        let selector = AutoParallelismSelector::new(AutoParallelismConfig::default());
        let annealed = selector.generate_annealing_strategies().expect("annealing failed");
        assert!(
            annealed.len() > 1,
            "the annealing walk must visit more than the initial configuration"
        );

        // Deterministic for a fixed problem definition.
        let again = selector.generate_annealing_strategies().expect("annealing failed");
        let ids: Vec<&str> = annealed.iter().map(|s| s.strategy_id.as_str()).collect();
        let ids_again: Vec<&str> = again.iter().map(|s| s.strategy_id.as_str()).collect();
        assert_eq!(ids, ids_again, "the seeded search must be reproducible");

        // Every configuration must respect the device budget.
        let max_devices = selector.config.hardware_constraints.num_devices;
        for strategy in &annealed {
            if let Some(cfg) = &strategy.parallelism_3d {
                assert!(
                    cfg.dp_size * cfg.mp_size * cfg.pp_size <= max_devices,
                    "annealing produced an infeasible configuration"
                );
            }
        }
    }

    #[test]
    fn test_multi_objective_returns_a_pareto_subset() {
        // Regression: this delegated verbatim to `generate_cost_based_strategies`.
        let selector = AutoParallelismSelector::new(AutoParallelismConfig::default());
        let all = selector.generate_cost_based_strategies().expect("cost based");
        let front = selector.generate_multi_objective_strategies().expect("multi objective");

        assert!(!front.is_empty());
        assert!(
            front.len() <= all.len(),
            "a Pareto front cannot be larger than the candidate set"
        );

        // No member of the front may be dominated by another candidate.
        let key = |s: &ParallelismStrategy| {
            [
                -s.expected_performance.throughput,
                s.expected_performance.memory_per_device as f64,
                s.expected_performance.communication_overhead as f64,
            ]
        };
        for member in &front {
            let m = key(member);
            for other in &all {
                let o = key(other);
                let dominates = o.iter().zip(m.iter()).all(|(x, y)| x <= y)
                    && o.iter().zip(m.iter()).any(|(x, y)| x < y);
                assert!(!dominates, "front member is dominated by another candidate");
            }
        }
    }

    #[test]
    fn test_auto_parallelism_config() {
        let config = AutoParallelismConfig::default();
        assert!(config.enabled);
        assert_eq!(config.hardware_constraints.num_devices, 8);
    }

    #[test]
    fn test_auto_parallelism_selector_creation() {
        let config = AutoParallelismConfig::default();
        let selector = AutoParallelismSelector::new(config);
        assert!(selector.current_strategy.is_none());
    }

    #[test]
    fn test_strategy_selection() {
        let config = AutoParallelismConfig::default();
        let mut selector = AutoParallelismSelector::new(config);

        let strategy = selector.select_strategy();
        assert!(strategy.is_ok());
        assert!(selector.current_strategy.is_some());
    }

    #[test]
    fn test_rule_based_strategy_generation() {
        let config = AutoParallelismConfig {
            selection_algorithm: SelectionAlgorithm::RuleBased,
            ..Default::default()
        };
        let selector = AutoParallelismSelector::new(config);

        let strategies = selector.generate_rule_based_strategies();
        assert!(strategies.is_ok());
        assert!(!strategies.expect("operation failed in test").is_empty());
    }

    #[test]
    fn test_performance_estimation() {
        let config = AutoParallelismConfig::default();
        let selector = AutoParallelismSelector::new(config);

        let metrics = selector.estimate_performance_data_parallel();
        assert!(metrics.is_ok());

        let metrics = metrics.expect("operation failed in test");
        assert!(metrics.time_per_step.as_secs_f64() > 0.0);
        assert!(metrics.memory_per_device > 0);
    }

    #[test]
    fn test_strategy_comparison() {
        let config = AutoParallelismConfig {
            optimization_objective: OptimizationObjective::MinimizeTime,
            ..Default::default()
        };
        let selector = AutoParallelismSelector::new(config);

        let strategy1 = ParallelismStrategy {
            strategy_id: "test1".to_string(),
            data_parallel: None,
            parallelism_3d: None,
            expert_parallel: None,
            sequence_parallel: None,
            tensor_parallel: None,
            expected_performance: PerformanceMetrics {
                time_per_step: Duration::from_secs(1),
                memory_per_device: 1000,
                communication_overhead: 0.1,
                throughput: 1.0,
                efficiency: 0.8,
                scalability: 0.9,
            },
            confidence: 0.8,
            rationale: "Test strategy 1".to_string(),
        };

        let strategy2 = ParallelismStrategy {
            strategy_id: "test2".to_string(),
            data_parallel: None,
            parallelism_3d: None,
            expert_parallel: None,
            sequence_parallel: None,
            tensor_parallel: None,
            expected_performance: PerformanceMetrics {
                time_per_step: Duration::from_secs(2),
                memory_per_device: 800,
                communication_overhead: 0.05,
                throughput: 0.5,
                efficiency: 0.9,
                scalability: 0.85,
            },
            confidence: 0.9,
            rationale: "Test strategy 2".to_string(),
        };

        let comparison = selector.compare_strategies(&strategy1, &strategy2);
        assert!(comparison.is_ok());
        assert_eq!(
            comparison.expect("operation failed in test"),
            std::cmp::Ordering::Less
        ); // strategy1 has less time
    }

    #[test]
    fn test_memory_estimation() {
        let constraints = ModelConstraints {
            num_parameters: 1_000_000,
            ..Default::default()
        };

        let memory = utils::estimate_model_memory(&constraints);
        assert_eq!(memory, 16_000_000); // 4 * 4 * 1M = 16MB
    }

    #[test]
    fn test_requirements_checking() {
        let strategy = ParallelismStrategy {
            strategy_id: "test".to_string(),
            data_parallel: None,
            parallelism_3d: None,
            expert_parallel: None,
            sequence_parallel: None,
            tensor_parallel: None,
            expected_performance: PerformanceMetrics {
                time_per_step: Duration::from_secs(1),
                memory_per_device: 1000,
                communication_overhead: 0.2,
                throughput: 2.0,
                efficiency: 0.8,
                scalability: 0.9,
            },
            confidence: 0.8,
            rationale: "Test strategy".to_string(),
        };

        let requirements = PerformanceRequirements {
            max_training_time: Some(Duration::from_secs(2)),
            min_throughput: Some(1.0),
            max_memory_per_device: Some(2000),
            max_communication_overhead: Some(0.3),
            min_efficiency: Some(0.7),
        };

        assert!(utils::meets_requirements(&strategy, &requirements));

        let strict_requirements = PerformanceRequirements {
            max_training_time: Some(Duration::from_millis(500)),
            ..requirements
        };

        assert!(!utils::meets_requirements(&strategy, &strict_requirements));
    }
}
