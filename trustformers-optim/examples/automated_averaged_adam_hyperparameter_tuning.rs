//! # Automated Hyperparameter Tuning for Averaged Adam Optimizer
//!
//! This example demonstrates advanced automated hyperparameter optimization
//! techniques specifically designed for the Averaged Adam optimizer, including
//! Bayesian optimization, population-based training, and adaptive tuning strategies.
//!
//! ## Features Demonstrated:
//! - Bayesian optimization for hyperparameter search
//! - Population-based training (PBT) for dynamic adaptation
//! - Multi-objective optimization (convergence speed vs stability)
//! - Automated learning rate scheduling with Averaged Adam
//! - Task-specific hyperparameter recommendations
//! - Performance-aware hyperparameter selection

use scirs2_core::random::*; // SciRS2 Integration Policy
use std::collections::HashMap;
use std::f32::consts::PI;
use std::time::Instant;
use trustformers_core::TrustformersError;
use trustformers_core::{traits::Optimizer, Tensor};
use trustformers_optim::*;

/// Hyperparameter search space for Averaged Adam
#[derive(Debug, Clone)]
pub struct AveragedAdamSearchSpace {
    /// Learning rate bounds
    pub learning_rate_bounds: (f32, f32),
    /// Beta1 bounds (momentum)
    pub beta1_bounds: (f32, f32),
    /// Beta2 bounds (second moment)
    pub beta2_bounds: (f32, f32),
    /// Weight decay bounds
    pub weight_decay_bounds: (f32, f32),
    /// Averaging coefficient bounds (key parameter for Averaged Adam)
    pub averaging_coeff_bounds: (f32, f32),
    /// Epsilon bounds
    pub epsilon_bounds: (f32, f32),
}

impl Default for AveragedAdamSearchSpace {
    fn default() -> Self {
        Self {
            learning_rate_bounds: (1e-5, 1e-2),
            beta1_bounds: (0.8, 0.95),
            beta2_bounds: (0.99, 0.9999),
            weight_decay_bounds: (0.0, 0.1),
            averaging_coeff_bounds: (0.99, 0.9999), // Critical for Averaged Adam
            epsilon_bounds: (1e-10, 1e-6),
        }
    }
}

impl AveragedAdamSearchSpace {
    /// Create search space for transformer training
    pub fn for_transformer_training() -> Self {
        Self {
            learning_rate_bounds: (1e-5, 5e-3),
            beta1_bounds: (0.85, 0.95),
            beta2_bounds: (0.995, 0.9999),
            weight_decay_bounds: (0.001, 0.01),
            averaging_coeff_bounds: (0.995, 0.9999),
            epsilon_bounds: (1e-8, 1e-7),
        }
    }

    /// Create search space for computer vision tasks
    pub fn for_computer_vision() -> Self {
        Self {
            learning_rate_bounds: (1e-4, 1e-2),
            beta1_bounds: (0.8, 0.9),
            beta2_bounds: (0.99, 0.999),
            weight_decay_bounds: (1e-4, 1e-2),
            averaging_coeff_bounds: (0.99, 0.999),
            epsilon_bounds: (1e-8, 1e-6),
        }
    }

    /// Create search space for reinforcement learning
    pub fn for_reinforcement_learning() -> Self {
        Self {
            learning_rate_bounds: (1e-5, 1e-2),
            beta1_bounds: (0.9, 0.99),
            beta2_bounds: (0.999, 0.9999),
            weight_decay_bounds: (0.0, 0.001),
            averaging_coeff_bounds: (0.999, 0.9999), // Higher averaging for RL stability
            epsilon_bounds: (1e-8, 1e-7),
        }
    }

    /// Sample random hyperparameters from the search space
    pub fn sample<R: Rng>(&self, rng: &mut R) -> AveragedAdamHyperparameters {
        AveragedAdamHyperparameters {
            learning_rate: rng
                .random_range(self.learning_rate_bounds.0..=self.learning_rate_bounds.1),
            beta1: rng.random_range(self.beta1_bounds.0..=self.beta1_bounds.1),
            beta2: rng.random_range(self.beta2_bounds.0..=self.beta2_bounds.1),
            weight_decay: rng.random_range(self.weight_decay_bounds.0..=self.weight_decay_bounds.1),
            averaging_coeff: rng
                .random_range(self.averaging_coeff_bounds.0..=self.averaging_coeff_bounds.1),
            epsilon: rng.random_range(self.epsilon_bounds.0..=self.epsilon_bounds.1),
        }
    }

    /// Sample using log-uniform distribution for parameters that benefit from it
    pub fn sample_log_uniform<R: Rng>(&self, rng: &mut R) -> AveragedAdamHyperparameters {
        AveragedAdamHyperparameters {
            learning_rate: sample_log_uniform(
                rng,
                self.learning_rate_bounds.0,
                self.learning_rate_bounds.1,
            ),
            beta1: rng.random_range(self.beta1_bounds.0..=self.beta1_bounds.1),
            beta2: rng.random_range(self.beta2_bounds.0..=self.beta2_bounds.1),
            weight_decay: sample_log_uniform(
                rng,
                self.weight_decay_bounds.0.max(1e-8),
                self.weight_decay_bounds.1,
            ),
            averaging_coeff: rng
                .random_range(self.averaging_coeff_bounds.0..=self.averaging_coeff_bounds.1),
            epsilon: sample_log_uniform(rng, self.epsilon_bounds.0, self.epsilon_bounds.1),
        }
    }
}

/// Sample from log-uniform distribution
fn sample_log_uniform<R: Rng>(rng: &mut R, min: f32, max: f32) -> f32 {
    let log_min = min.ln();
    let log_max = max.ln();
    (rng.random::<f32>() * (log_max - log_min) + log_min).exp()
}

/// Hyperparameter configuration for Averaged Adam
#[derive(Debug, Clone)]
pub struct AveragedAdamHyperparameters {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub averaging_coeff: f32,
    pub epsilon: f32,
}

impl AveragedAdamHyperparameters {
    /// Create optimizer with these hyperparameters
    pub fn create_optimizer(&self) -> AveragedAdam {
        AveragedAdam::new(
            self.learning_rate,
            (self.beta1, self.beta2),
            self.epsilon,
            self.weight_decay,
            self.averaging_coeff,
        )
    }

    /// Calculate distance to another hyperparameter configuration
    pub fn distance(&self, other: &Self) -> f32 {
        let lr_diff = (self.learning_rate.ln() - other.learning_rate.ln()).abs();
        let b1_diff = (self.beta1 - other.beta1).abs();
        let b2_diff = (self.beta2 - other.beta2).abs();
        let wd_diff = (self.weight_decay - other.weight_decay).abs();
        let ac_diff = (self.averaging_coeff - other.averaging_coeff).abs();
        let eps_diff = (self.epsilon.ln() - other.epsilon.ln()).abs();

        // Weighted distance with higher weight for critical Averaged Adam parameters
        lr_diff * 0.3
            + b1_diff * 0.15
            + b2_diff * 0.15
            + wd_diff * 0.1
            + ac_diff * 0.25
            + eps_diff * 0.05 // Higher weight for averaging_coeff
    }

    /// Apply mutation for evolutionary optimization
    pub fn mutate<R: Rng>(
        &mut self,
        rng: &mut R,
        mutation_strength: f32,
        search_space: &AveragedAdamSearchSpace,
    ) {
        let mut gaussian_noise = || -> f32 {
            // Box-Muller transformation for Gaussian noise
            let u1: f32 = rng.random();
            let u2: f32 = rng.random();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        };

        // Mutate learning rate (log-space)
        let lr_log = self.learning_rate.ln() + gaussian_noise() * mutation_strength;
        self.learning_rate = lr_log.exp().clamp(
            search_space.learning_rate_bounds.0,
            search_space.learning_rate_bounds.1,
        );

        // Mutate beta parameters
        self.beta1 = (self.beta1 + gaussian_noise() * mutation_strength * 0.05)
            .clamp(search_space.beta1_bounds.0, search_space.beta1_bounds.1);

        self.beta2 = (self.beta2 + gaussian_noise() * mutation_strength * 0.001)
            .clamp(search_space.beta2_bounds.0, search_space.beta2_bounds.1);

        // Mutate weight decay (log-space for small values)
        if self.weight_decay > 0.0 {
            let wd_log = self.weight_decay.ln() + gaussian_noise() * mutation_strength;
            self.weight_decay = wd_log.exp().clamp(
                search_space.weight_decay_bounds.0,
                search_space.weight_decay_bounds.1,
            );
        } else {
            self.weight_decay = (gaussian_noise() * mutation_strength * 0.01).abs().clamp(
                search_space.weight_decay_bounds.0,
                search_space.weight_decay_bounds.1,
            );
        }

        // Mutate averaging coefficient (critical parameter)
        self.averaging_coeff =
            (self.averaging_coeff + gaussian_noise() * mutation_strength * 0.001).clamp(
                search_space.averaging_coeff_bounds.0,
                search_space.averaging_coeff_bounds.1,
            );

        // Mutate epsilon (log-space)
        let eps_log = self.epsilon.ln() + gaussian_noise() * mutation_strength;
        self.epsilon = eps_log
            .exp()
            .clamp(search_space.epsilon_bounds.0, search_space.epsilon_bounds.1);
    }
}

/// Training task definition for hyperparameter optimization
#[derive(Debug, Clone)]
pub struct OptimizationTask {
    /// Task name
    pub name: String,
    /// Model parameter count
    pub parameter_count: usize,
    /// Training steps for evaluation
    pub training_steps: usize,
    /// Target loss for convergence
    pub target_loss: f32,
    /// Task type for specialized optimization
    pub task_type: TaskType,
    /// Evaluation metrics weights
    pub metric_weights: MetricWeights,
}

#[derive(Debug, Clone)]
pub enum TaskType {
    LanguageModeling,
    ImageClassification,
    ReinforcementLearning,
    FineTuning,
    PreTraining,
}

#[derive(Debug, Clone)]
pub struct MetricWeights {
    /// Weight for final loss
    pub loss_weight: f32,
    /// Weight for convergence speed
    pub speed_weight: f32,
    /// Weight for training stability
    pub stability_weight: f32,
    /// Weight for parameter norm change
    pub norm_change_weight: f32,
}

impl Default for MetricWeights {
    fn default() -> Self {
        Self {
            loss_weight: 0.4,
            speed_weight: 0.3,
            stability_weight: 0.2,
            norm_change_weight: 0.1,
        }
    }
}

/// Training results for hyperparameter evaluation
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub hyperparameters: AveragedAdamHyperparameters,
    pub final_loss: f32,
    pub convergence_step: Option<usize>,
    pub training_stability: f32, // Variance in loss trajectory
    pub param_norm_change: f32,
    pub training_time: std::time::Duration,
    pub composite_score: f32, // Multi-objective score
}

/// Bayesian optimization for hyperparameter tuning
pub struct BayesianOptimizer {
    /// Search space
    search_space: AveragedAdamSearchSpace,
    /// Evaluated configurations and their scores
    history: Vec<(AveragedAdamHyperparameters, f32)>,
    /// Current best configuration
    best_config: Option<AveragedAdamHyperparameters>,
    /// Best score achieved
    best_score: f32,
    /// Exploration vs exploitation trade-off
    exploration_factor: f32,
    /// Number of random initial samples
    initial_random_samples: usize,
}

impl BayesianOptimizer {
    /// Create new Bayesian optimizer
    pub fn new(search_space: AveragedAdamSearchSpace) -> Self {
        Self {
            search_space,
            history: Vec::new(),
            best_config: None,
            best_score: f32::NEG_INFINITY,
            exploration_factor: 0.1,
            initial_random_samples: 10,
        }
    }

    /// Suggest next hyperparameter configuration to try
    pub fn suggest_next<R: Rng>(&self, rng: &mut R) -> AveragedAdamHyperparameters {
        if self.history.len() < self.initial_random_samples {
            // Random exploration phase
            self.search_space.sample_log_uniform(rng)
        } else {
            // Bayesian optimization phase
            self.suggest_via_acquisition_function(rng)
        }
    }

    /// Suggest configuration using acquisition function (simplified GP-based)
    fn suggest_via_acquisition_function<R: Rng>(&self, rng: &mut R) -> AveragedAdamHyperparameters {
        let mut best_acquisition = f32::NEG_INFINITY;
        let mut best_candidate = self.search_space.sample_log_uniform(rng);

        // Generate multiple candidates and select best according to acquisition function
        for _ in 0..100 {
            let candidate = self.search_space.sample_log_uniform(rng);
            let acquisition_score = self.calculate_acquisition_score(&candidate);

            if acquisition_score > best_acquisition {
                best_acquisition = acquisition_score;
                best_candidate = candidate;
            }
        }

        best_candidate
    }

    /// Calculate acquisition function (Upper Confidence Bound approximation)
    fn calculate_acquisition_score(&self, candidate: &AveragedAdamHyperparameters) -> f32 {
        let predicted_mean = self.predict_performance(candidate);
        let predicted_uncertainty = self.estimate_uncertainty(candidate);

        // UCB: mean + exploration_factor * uncertainty
        predicted_mean + self.exploration_factor * predicted_uncertainty
    }

    /// Predict performance using k-nearest neighbors (simplified GP surrogate)
    fn predict_performance(&self, candidate: &AveragedAdamHyperparameters) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }

        // Find k nearest neighbors and weight by distance
        let k = 5.min(self.history.len());
        let mut distances: Vec<_> = self
            .history
            .iter()
            .map(|(config, score)| (candidate.distance(config), *score))
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Weighted average of nearest neighbors
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for i in 0..k {
            let (distance, score) = distances[i];
            let weight = 1.0 / (distance + 1e-6); // Inverse distance weighting
            weighted_sum += weight * score;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    /// Estimate uncertainty in prediction
    fn estimate_uncertainty(&self, candidate: &AveragedAdamHyperparameters) -> f32 {
        if self.history.is_empty() {
            return 1.0; // High uncertainty if no data
        }

        // Estimate uncertainty based on distance to nearest evaluated point
        let min_distance = self
            .history
            .iter()
            .map(|(config, _)| candidate.distance(config))
            .fold(f32::INFINITY, f32::min);

        // Higher uncertainty for points far from evaluated configurations
        (min_distance * 2.0).tanh()
    }

    /// Update optimizer with new evaluation result
    pub fn update(&mut self, hyperparams: AveragedAdamHyperparameters, score: f32) {
        self.history.push((hyperparams.clone(), score));

        if score > self.best_score {
            self.best_score = score;
            self.best_config = Some(hyperparams);
        }

        // Decay exploration factor as we gather more data
        self.exploration_factor *= 0.995;
    }

    /// Get current best configuration
    pub fn get_best_config(&self) -> Option<&AveragedAdamHyperparameters> {
        self.best_config.as_ref()
    }
}

/// Population-based training for dynamic hyperparameter adaptation
pub struct PopulationBasedTrainer {
    /// Population size
    population_size: usize,
    /// Current population
    population: Vec<PopulationMember>,
    /// Search space
    search_space: AveragedAdamSearchSpace,
    /// Exploitation threshold (top % to copy from)
    exploitation_threshold: f32,
    /// Mutation strength for exploration
    mutation_strength: f32,
}

#[derive(Debug, Clone)]
pub struct PopulationMember {
    pub hyperparams: AveragedAdamHyperparameters,
    pub performance_history: Vec<f32>,
    pub current_performance: f32,
    pub training_step: usize,
}

impl PopulationBasedTrainer {
    /// Create new population-based trainer
    pub fn new(population_size: usize, search_space: AveragedAdamSearchSpace) -> Self {
        let mut rng = thread_rng();
        let population = (0..population_size)
            .map(|_| PopulationMember {
                hyperparams: search_space.sample_log_uniform(&mut rng),
                performance_history: Vec::new(),
                current_performance: f32::NEG_INFINITY,
                training_step: 0,
            })
            .collect();

        Self {
            population_size,
            population,
            search_space,
            exploitation_threshold: 0.2, // Top 20%
            mutation_strength: 0.1,
        }
    }

    /// Evolve population based on performance
    pub fn evolve(&mut self) {
        // Sort population by performance
        self.population
            .sort_by(|a, b| b.current_performance.partial_cmp(&a.current_performance).unwrap());

        let top_k = ((self.population_size as f32 * self.exploitation_threshold) as usize).max(1);
        let mut rng = thread_rng();

        // Replace bottom performers with mutations of top performers
        for i in top_k..self.population_size {
            // Select a random top performer to copy from
            let source_idx = rng.random_range(0..top_k);
            let mut new_hyperparams = self.population[source_idx].hyperparams.clone();

            // Mutate the copied hyperparameters
            new_hyperparams.mutate(&mut rng, self.mutation_strength, &self.search_space);

            self.population[i].hyperparams = new_hyperparams;
            self.population[i].performance_history.clear();
            self.population[i].current_performance = f32::NEG_INFINITY;
            self.population[i].training_step = 0;
        }

        // Decay mutation strength over time
        self.mutation_strength *= 0.99;
    }

    /// Update performance for a population member
    pub fn update_performance(&mut self, member_idx: usize, performance: f32, step: usize) {
        if member_idx < self.population_size {
            self.population[member_idx].current_performance = performance;
            self.population[member_idx].performance_history.push(performance);
            self.population[member_idx].training_step = step;
        }
    }

    /// Get current population
    pub fn get_population(&self) -> &[PopulationMember] {
        &self.population
    }

    /// Get best member
    pub fn get_best_member(&self) -> Option<&PopulationMember> {
        self.population
            .iter()
            .max_by(|a, b| a.current_performance.partial_cmp(&b.current_performance).unwrap())
    }
}

/// Evaluate hyperparameter configuration on a training task
fn evaluate_hyperparameters(
    hyperparams: &AveragedAdamHyperparameters,
    task: &OptimizationTask,
) -> Result<TrainingResult, TrustformersError> {
    let mut optimizer = hyperparams.create_optimizer();

    // Create synthetic training data based on task
    let mut parameters = HashMap::new();
    let mut gradients = HashMap::new();

    let layer_count = (task.parameter_count as f32).log10() as usize;
    let params_per_layer = task.parameter_count / layer_count.max(1);

    for layer in 0..layer_count {
        let layer_params = vec![0.1f32; params_per_layer];
        let layer_grads = vec![0.0f32; params_per_layer];
        parameters.insert(format!("layer_{}", layer), Tensor::new(layer_params)?);
        gradients.insert(format!("layer_{}", layer), Tensor::new(layer_grads)?);
    }

    let start_time = Instant::now();
    let mut losses = Vec::new();
    let mut convergence_step = None;

    // Initial parameter norms
    let initial_norms: Vec<f32> = parameters.values().map(|p| p.norm().unwrap_or(0.0)).collect();

    // Training simulation
    let mut rng = thread_rng();
    for step in 0..task.training_steps {
        // Simulate forward pass and loss computation
        let base_loss = match task.task_type {
            TaskType::LanguageModeling => 5.0,
            TaskType::ImageClassification => 2.3,
            TaskType::ReinforcementLearning => 10.0,
            TaskType::FineTuning => 1.0,
            TaskType::PreTraining => 6.0,
        };

        let decay_rate = match task.task_type {
            TaskType::LanguageModeling => 0.001,
            TaskType::ImageClassification => 0.01,
            TaskType::ReinforcementLearning => 0.0005,
            TaskType::FineTuning => 0.02,
            TaskType::PreTraining => 0.0008,
        };

        let noise_factor = match task.task_type {
            TaskType::LanguageModeling => 0.1,
            TaskType::ImageClassification => 0.05,
            TaskType::ReinforcementLearning => 0.3,
            TaskType::FineTuning => 0.02,
            TaskType::PreTraining => 0.15,
        };

        let loss = base_loss * (-decay_rate * step as f32).exp()
            + (rng.random::<f32>() - 0.5) * noise_factor
            + 0.01; // Minimum loss

        losses.push(loss.max(0.001));

        // Check convergence
        if loss < task.target_loss && convergence_step.is_none() {
            convergence_step = Some(step);
        }

        // Simulate gradient computation
        for (name, gradient) in gradients.iter_mut() {
            let param_len = parameters[name].len();
            let grad_magnitude = match task.task_type {
                TaskType::LanguageModeling => 0.01,
                TaskType::ImageClassification => 0.02,
                TaskType::ReinforcementLearning => 0.005,
                TaskType::FineTuning => 0.001,
                TaskType::PreTraining => 0.015,
            };

            let grad_data: Vec<f32> =
                (0..param_len).map(|_| (rng.random::<f32>() - 0.5) * grad_magnitude).collect();
            *gradient = Tensor::new(grad_data)?;
        }

        // Update parameters
        for (name, param) in parameters.iter_mut() {
            if let Some(gradient) = gradients.get(name) {
                optimizer.update(param, gradient)?;
            }
        }
        optimizer.step();
    }

    let training_time = start_time.elapsed();

    // Calculate final metrics
    let final_loss = losses.last().copied().unwrap_or(f32::INFINITY);

    // Training stability (inverse of loss variance)
    let mean_loss = losses.iter().sum::<f32>() / losses.len() as f32;
    let loss_variance =
        losses.iter().map(|&loss| (loss - mean_loss).powi(2)).sum::<f32>() / losses.len() as f32;
    let training_stability = 1.0 / (1.0 + loss_variance);

    // Parameter norm change
    let final_norms: Vec<f32> = parameters.values().map(|p| p.norm().unwrap_or(0.0)).collect();
    let param_norm_change = initial_norms
        .iter()
        .zip(final_norms.iter())
        .map(|(initial, final_)| (final_ - initial).abs())
        .sum::<f32>()
        / initial_norms.len() as f32;

    // Composite score calculation
    let speed_score = match convergence_step {
        Some(step) => 1.0 - (step as f32 / task.training_steps as f32),
        None => 0.0,
    };

    let loss_score = 1.0 / (1.0 + final_loss);
    let norm_score = 1.0 / (1.0 + param_norm_change);

    let composite_score = task.metric_weights.loss_weight * loss_score
        + task.metric_weights.speed_weight * speed_score
        + task.metric_weights.stability_weight * training_stability
        + task.metric_weights.norm_change_weight * norm_score;

    Ok(TrainingResult {
        hyperparameters: hyperparams.clone(),
        final_loss,
        convergence_step,
        training_stability,
        param_norm_change,
        training_time,
        composite_score,
    })
}

/// Run Bayesian optimization for hyperparameter tuning
fn run_bayesian_optimization(
    task: &OptimizationTask,
    search_space: AveragedAdamSearchSpace,
    num_iterations: usize,
) -> Result<Vec<TrainingResult>, TrustformersError> {
    println!("🔬 Running Bayesian Optimization for {}", task.name);
    println!(
        "Parameters: {}, Steps: {}, Target Loss: {:.4}",
        task.parameter_count, task.training_steps, task.target_loss
    );

    let mut optimizer = BayesianOptimizer::new(search_space);
    let mut results = Vec::new();
    let mut rng = thread_rng();

    for iteration in 0..num_iterations {
        // Get next hyperparameter suggestion
        let hyperparams = optimizer.suggest_next(&mut rng);

        // Evaluate the suggested configuration
        let result = evaluate_hyperparameters(&hyperparams, task)?;

        if iteration % 5 == 0 || iteration == num_iterations - 1 {
            println!(
                "Iteration {}: Score = {:.4}, Loss = {:.4}, LR = {:.6}, γ = {:.5}",
                iteration,
                result.composite_score,
                result.final_loss,
                result.hyperparameters.learning_rate,
                result.hyperparameters.averaging_coeff
            );
        }

        // Update optimizer with result
        optimizer.update(hyperparams, result.composite_score);
        results.push(result);
    }

    if let Some(best_config) = optimizer.get_best_config() {
        println!("🏆 Best Configuration Found:");
        println!("   Learning Rate: {:.6}", best_config.learning_rate);
        println!(
            "   Beta1: {:.4}, Beta2: {:.4}",
            best_config.beta1, best_config.beta2
        );
        println!("   Weight Decay: {:.6}", best_config.weight_decay);
        println!(
            "   Averaging Coefficient: {:.5}",
            best_config.averaging_coeff
        );
        println!("   Epsilon: {:.2e}", best_config.epsilon);
    }

    Ok(results)
}

/// Run population-based training optimization
fn run_population_based_training(
    task: &OptimizationTask,
    search_space: AveragedAdamSearchSpace,
    population_size: usize,
    generations: usize,
) -> Result<Vec<TrainingResult>, TrustformersError> {
    println!("🧬 Running Population-Based Training for {}", task.name);
    println!(
        "Population: {}, Generations: {}",
        population_size, generations
    );

    let mut pbt = PopulationBasedTrainer::new(population_size, search_space);
    let mut all_results = Vec::new();

    for generation in 0..generations {
        println!("\n🔄 Generation {}/{}", generation + 1, generations);

        // Evaluate all members of current population
        let population = pbt.get_population();
        let mut generation_results = Vec::new();

        for (idx, member) in population.iter().enumerate() {
            let result = evaluate_hyperparameters(&member.hyperparams, task)?;
            generation_results.push((idx, result));
        }

        // Update performance after evaluation to avoid borrowing conflicts
        for (idx, result) in &generation_results {
            pbt.update_performance(*idx, result.composite_score, generation);
        }

        // Find best in this generation
        let best_gen_result = generation_results
            .iter()
            .max_by(|(_, a), (_, b)| a.composite_score.partial_cmp(&b.composite_score).unwrap());

        if let Some((_, best)) = best_gen_result {
            println!(
                "Best in Generation: Score = {:.4}, Loss = {:.4}, γ = {:.5}",
                best.composite_score, best.final_loss, best.hyperparameters.averaging_coeff
            );
        }

        all_results.extend(generation_results);

        // Evolve population for next generation
        if generation < generations - 1 {
            pbt.evolve();
        }
    }

    if let Some(best_member) = pbt.get_best_member() {
        println!("\n🏆 Overall Best Configuration:");
        println!(
            "   Learning Rate: {:.6}",
            best_member.hyperparams.learning_rate
        );
        println!(
            "   Beta1: {:.4}, Beta2: {:.4}",
            best_member.hyperparams.beta1, best_member.hyperparams.beta2
        );
        println!(
            "   Weight Decay: {:.6}",
            best_member.hyperparams.weight_decay
        );
        println!(
            "   Averaging Coefficient: {:.5}",
            best_member.hyperparams.averaging_coeff
        );
        println!("   Epsilon: {:.2e}", best_member.hyperparams.epsilon);
    }

    // Extract TrainingResults from tuples
    let results: Vec<TrainingResult> = all_results.into_iter().map(|(_, result)| result).collect();
    Ok(results)
}

/// Compare different optimization methods
fn compare_optimization_methods() -> Result<(), TrustformersError> {
    println!("🎯 Automated Hyperparameter Optimization Comparison");
    println!("=================================================");

    // Define optimization tasks
    let tasks = vec![
        OptimizationTask {
            name: "Transformer Pre-training".to_string(),
            parameter_count: 100_000,
            training_steps: 500,
            target_loss: 0.5,
            task_type: TaskType::PreTraining,
            metric_weights: MetricWeights {
                loss_weight: 0.5,
                speed_weight: 0.3,
                stability_weight: 0.15,
                norm_change_weight: 0.05,
            },
        },
        OptimizationTask {
            name: "Image Classification Fine-tuning".to_string(),
            parameter_count: 50_000,
            training_steps: 300,
            target_loss: 0.1,
            task_type: TaskType::FineTuning,
            metric_weights: MetricWeights {
                loss_weight: 0.6,
                speed_weight: 0.25,
                stability_weight: 0.1,
                norm_change_weight: 0.05,
            },
        },
    ];

    for task in tasks {
        println!("\n📊 Optimizing for: {}", task.name);
        println!("{}", "=".repeat(50));

        let search_space = match task.task_type {
            TaskType::PreTraining | TaskType::LanguageModeling => {
                AveragedAdamSearchSpace::for_transformer_training()
            },
            TaskType::ImageClassification | TaskType::FineTuning => {
                AveragedAdamSearchSpace::for_computer_vision()
            },
            TaskType::ReinforcementLearning => {
                AveragedAdamSearchSpace::for_reinforcement_learning()
            },
        };

        // Run Bayesian Optimization
        let bayesian_results = run_bayesian_optimization(&task, search_space.clone(), 25)?;
        let best_bayesian = bayesian_results
            .iter()
            .max_by(|a, b| a.composite_score.partial_cmp(&b.composite_score).unwrap())
            .unwrap();

        // Run Population-Based Training
        let pbt_results = run_population_based_training(&task, search_space, 8, 5)?;
        let best_pbt = pbt_results
            .iter()
            .max_by(|a, b| a.composite_score.partial_cmp(&b.composite_score).unwrap())
            .unwrap();

        // Compare results
        println!("\n📈 Optimization Results Comparison:");
        println!("Bayesian Optimization:");
        println!(
            "  Best Score: {:.4}, Final Loss: {:.4}",
            best_bayesian.composite_score, best_bayesian.final_loss
        );
        println!(
            "  Learning Rate: {:.6}, Averaging Coeff: {:.5}",
            best_bayesian.hyperparameters.learning_rate,
            best_bayesian.hyperparameters.averaging_coeff
        );

        println!("Population-Based Training:");
        println!(
            "  Best Score: {:.4}, Final Loss: {:.4}",
            best_pbt.composite_score, best_pbt.final_loss
        );
        println!(
            "  Learning Rate: {:.6}, Averaging Coeff: {:.5}",
            best_pbt.hyperparameters.learning_rate, best_pbt.hyperparameters.averaging_coeff
        );

        // Determine winner
        let winner = if best_bayesian.composite_score > best_pbt.composite_score {
            "Bayesian Optimization"
        } else {
            "Population-Based Training"
        };
        println!("🏆 Winner: {}", winner);
    }

    Ok(())
}

fn main() -> Result<(), TrustformersError> {
    println!("🎯 Automated Hyperparameter Optimization for Averaged Adam");
    println!("=========================================================");
    println!("This example demonstrates advanced automated hyperparameter");
    println!("optimization techniques specifically for Averaged Adam optimizer.\n");

    // Run optimization method comparison
    compare_optimization_methods()?;

    println!("\n🎉 Automated Hyperparameter Optimization Completed!");
    println!("==================================================");

    println!("\n📋 Key Insights:");
    println!("• Averaging coefficient (γ) is critical for Averaged Adam performance");
    println!("• Higher γ values (≥0.999) typically better for stable convergence");
    println!("• Learning rate bounds should be task-specific (transformers vs vision)");
    println!("• Bayesian optimization efficient for small hyperparameter spaces");
    println!("• Population-based training excels at dynamic adaptation during training");
    println!("• Multi-objective optimization balances convergence speed and stability");

    println!("\n💡 Optimization Recommendations:");
    println!("• For Transformer training: γ ∈ [0.995, 0.9999], LR ∈ [1e-5, 5e-3]");
    println!("• For Computer Vision: γ ∈ [0.99, 0.999], LR ∈ [1e-4, 1e-2]");
    println!("• For Reinforcement Learning: γ ∈ [0.999, 0.9999], LR ∈ [1e-5, 1e-2]");
    println!("• Use log-uniform sampling for learning rate and weight decay");
    println!("• Monitor parameter norm changes to detect optimization instabilities");
    println!("• Consider task-specific metric weights in multi-objective optimization");

    println!("\n🚀 Production Guidelines:");
    println!("• Start with task-specific search spaces for efficiency");
    println!("• Use Bayesian optimization for initial hyperparameter discovery");
    println!("• Apply population-based training for long-running experiments");
    println!("• Implement early stopping based on composite scores");
    println!("• Schedule hyperparameter re-optimization for changing data distributions");

    Ok(())
}
