//! Search strategies for hyperparameter optimization

use super::{
    Direction, ParameterValue, RandomSampler, Sampler, SearchSpace, TPESampler, Trial, TrialHistory,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Search strategy for hyperparameter optimization
pub trait SearchStrategy: Send + Sync {
    /// Suggest the next set of hyperparameters to try
    fn suggest(
        &mut self,
        search_space: &SearchSpace,
        history: &TrialHistory,
    ) -> Option<HashMap<String, ParameterValue>>;

    /// Check if the search should terminate
    fn should_terminate(&self, history: &TrialHistory) -> bool;

    /// Get the name of this strategy
    fn name(&self) -> &str;

    /// Update the strategy with a completed trial
    fn update(&mut self, _trial: &Trial) {}
}

/// Grid search strategy - exhaustively searches all combinations
pub struct GridSearch {
    /// All possible parameter combinations
    combinations: Vec<HashMap<String, ParameterValue>>,
    /// Current index in the combinations
    current_index: usize,
    /// Name of the strategy
    name: String,
}

impl GridSearch {
    /// Create a new grid search strategy
    pub fn new(search_space: &SearchSpace) -> Result<Self, String> {
        let combinations = Self::generate_combinations(search_space)?;
        Ok(Self {
            combinations,
            current_index: 0,
            name: "GridSearch".to_string(),
        })
    }

    /// Generate all possible combinations for the search space
    fn generate_combinations(
        search_space: &SearchSpace,
    ) -> Result<Vec<HashMap<String, ParameterValue>>, String> {
        if search_space.parameters.is_empty() {
            return Ok(vec![HashMap::new()]);
        }

        // Check if all parameters are discrete
        for param in &search_space.parameters {
            match param {
                super::search_space::HyperParameter::Continuous(_)
                | super::search_space::HyperParameter::Log(_) => {
                    return Err(
                        "Grid search only supports discrete and categorical parameters".to_string(),
                    );
                },
                _ => {},
            }
        }

        let mut combinations = vec![HashMap::new()];

        for param in &search_space.parameters {
            let param_name = param.name();
            let param_values = match param {
                super::search_space::HyperParameter::Categorical(p) => p
                    .choices
                    .iter()
                    .map(|choice| ParameterValue::String(choice.clone()))
                    .collect::<Vec<_>>(),
                super::search_space::HyperParameter::Discrete(p) => {
                    let mut values = Vec::new();
                    let mut current = p.low;
                    while current <= p.high {
                        values.push(ParameterValue::Int(current));
                        current += p.step;
                    }
                    values
                },
                _ => unreachable!(), // We checked above
            };

            // Cross product with existing combinations
            let mut new_combinations = Vec::new();
            for combination in combinations {
                for value in &param_values {
                    let mut new_combination = combination.clone();
                    new_combination.insert(param_name.to_string(), value.clone());
                    new_combinations.push(new_combination);
                }
            }
            combinations = new_combinations;
        }

        Ok(combinations)
    }

    /// Get the total number of combinations
    pub fn total_combinations(&self) -> usize {
        self.combinations.len()
    }

    /// Get the progress as a fraction [0.0, 1.0]
    pub fn progress(&self) -> f64 {
        if self.combinations.is_empty() {
            1.0
        } else {
            self.current_index as f64 / self.combinations.len() as f64
        }
    }
}

impl SearchStrategy for GridSearch {
    fn suggest(
        &mut self,
        _search_space: &SearchSpace,
        _history: &TrialHistory,
    ) -> Option<HashMap<String, ParameterValue>> {
        if self.current_index < self.combinations.len() {
            let suggestion = self.combinations[self.current_index].clone();
            self.current_index += 1;
            Some(suggestion)
        } else {
            None
        }
    }

    fn should_terminate(&self, _history: &TrialHistory) -> bool {
        self.current_index >= self.combinations.len()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Random search strategy
pub struct RandomSearch {
    /// Maximum number of trials
    max_trials: usize,
    /// Number of trials suggested so far
    trials_suggested: usize,
    /// Random sampler
    sampler: Box<dyn Sampler>,
    /// Name of the strategy
    name: String,
}

impl RandomSearch {
    /// Create a new random search strategy
    pub fn new(max_trials: usize) -> Self {
        Self {
            max_trials,
            trials_suggested: 0,
            sampler: Box::new(RandomSampler::new()),
            name: format!("RandomSearch(max_trials={})", max_trials),
        }
    }

    /// Create a random search with a specific seed
    pub fn with_seed(max_trials: usize, seed: u64) -> Self {
        Self {
            max_trials,
            trials_suggested: 0,
            sampler: Box::new(RandomSampler::with_seed(seed)),
            name: format!("RandomSearch(max_trials={}, seed={})", max_trials, seed),
        }
    }

    /// Get the progress as a fraction [0.0, 1.0]
    pub fn progress(&self) -> f64 {
        if self.max_trials == 0 {
            1.0
        } else {
            (self.trials_suggested as f64 / self.max_trials as f64).min(1.0)
        }
    }
}

impl SearchStrategy for RandomSearch {
    fn suggest(
        &mut self,
        search_space: &SearchSpace,
        history: &TrialHistory,
    ) -> Option<HashMap<String, ParameterValue>> {
        if self.trials_suggested < self.max_trials {
            let suggestion = self.sampler.sample(search_space, history);
            self.trials_suggested += 1;
            Some(suggestion)
        } else {
            None
        }
    }

    fn should_terminate(&self, _history: &TrialHistory) -> bool {
        self.trials_suggested >= self.max_trials
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Bayesian optimization strategy using TPE (Tree-structured Parzen Estimator)
pub struct BayesianOptimization {
    /// Maximum number of trials
    max_trials: usize,
    /// Number of trials suggested so far
    trials_suggested: usize,
    /// TPE sampler
    sampler: Box<dyn Sampler>,
    /// Name of the strategy
    name: String,
}

impl BayesianOptimization {
    /// Create a new Bayesian optimization strategy
    pub fn new(max_trials: usize) -> Self {
        Self {
            max_trials,
            trials_suggested: 0,
            sampler: Box::new(TPESampler::new()),
            name: format!("BayesianOptimization(max_trials={})", max_trials),
        }
    }

    /// Create Bayesian optimization with custom sampler configuration
    pub fn with_config(max_trials: usize, sampler_config: super::SamplerConfig) -> Self {
        Self {
            max_trials,
            trials_suggested: 0,
            sampler: Box::new(TPESampler::with_config(sampler_config)),
            name: format!("BayesianOptimization(max_trials={})", max_trials),
        }
    }

    /// Get the progress as a fraction [0.0, 1.0]
    pub fn progress(&self) -> f64 {
        if self.max_trials == 0 {
            1.0
        } else {
            (self.trials_suggested as f64 / self.max_trials as f64).min(1.0)
        }
    }
}

impl SearchStrategy for BayesianOptimization {
    fn suggest(
        &mut self,
        search_space: &SearchSpace,
        history: &TrialHistory,
    ) -> Option<HashMap<String, ParameterValue>> {
        if self.trials_suggested < self.max_trials {
            let suggestion = self.sampler.sample(search_space, history);
            self.trials_suggested += 1;
            Some(suggestion)
        } else {
            None
        }
    }

    fn should_terminate(&self, _history: &TrialHistory) -> bool {
        self.trials_suggested >= self.max_trials
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn update(&mut self, trial: &Trial) {
        self.sampler.update(trial);
    }
}

/// Halving strategy trait for reducing resource allocation
pub trait HalvingStrategy: Send + Sync {
    /// Determine the resource allocation for a trial at a given stage
    fn get_resource_allocation(&self, stage: usize, max_resource: f64) -> f64;

    /// Determine how many trials to keep for the next stage
    fn get_trials_to_keep(&self, current_trials: usize) -> usize;

    /// Get the maximum number of stages
    fn max_stages(&self) -> usize;
}

/// Standard successive halving
pub struct StandardHalving {
    /// Reduction factor (e.g., 3 means keep 1/3 of trials each stage)
    reduction_factor: usize,
}

impl StandardHalving {
    pub fn new(reduction_factor: usize) -> Self {
        assert!(reduction_factor >= 2, "Reduction factor must be at least 2");
        Self { reduction_factor }
    }
}

impl HalvingStrategy for StandardHalving {
    fn get_resource_allocation(&self, stage: usize, max_resource: f64) -> f64 {
        max_resource / (self.reduction_factor as f64).powi(stage as i32)
    }

    fn get_trials_to_keep(&self, current_trials: usize) -> usize {
        (current_trials / self.reduction_factor).max(1)
    }

    fn max_stages(&self) -> usize {
        10 // Arbitrary limit
    }
}

/// Successive Halving algorithm
pub struct SuccessiveHalving {
    /// Maximum resource budget
    max_resource: f64,
    /// Number of initial configurations
    initial_configs: usize,
    /// Halving strategy
    halving_strategy: Box<dyn HalvingStrategy>,
    /// Current stage
    current_stage: usize,
    /// Trials suggested so far
    trials_suggested: usize,
    /// Random sampler for initial suggestions
    sampler: Box<dyn Sampler>,
    /// Name of the strategy
    name: String,
}

impl SuccessiveHalving {
    /// Create a new successive halving strategy
    pub fn new(max_resource: f64, initial_configs: usize) -> Self {
        Self {
            max_resource,
            initial_configs,
            halving_strategy: Box::new(StandardHalving::new(3)),
            current_stage: 0,
            trials_suggested: 0,
            sampler: Box::new(RandomSampler::new()),
            name: format!(
                "SuccessiveHalving(max_resource={}, configs={})",
                max_resource, initial_configs
            ),
        }
    }

    /// Create successive halving with custom halving strategy
    pub fn with_halving_strategy(
        max_resource: f64,
        initial_configs: usize,
        halving_strategy: Box<dyn HalvingStrategy>,
    ) -> Self {
        Self {
            max_resource,
            initial_configs,
            halving_strategy,
            current_stage: 0,
            trials_suggested: 0,
            sampler: Box::new(RandomSampler::new()),
            name: format!(
                "SuccessiveHalving(max_resource={}, configs={})",
                max_resource, initial_configs
            ),
        }
    }

    /// Get the current resource allocation
    pub fn current_resource_allocation(&self) -> f64 {
        self.halving_strategy
            .get_resource_allocation(self.current_stage, self.max_resource)
    }
}

impl SearchStrategy for SuccessiveHalving {
    fn suggest(
        &mut self,
        search_space: &SearchSpace,
        history: &TrialHistory,
    ) -> Option<HashMap<String, ParameterValue>> {
        // In the first stage, suggest random configurations
        if self.current_stage == 0 && self.trials_suggested < self.initial_configs {
            let suggestion = self.sampler.sample(search_space, history);
            self.trials_suggested += 1;
            return Some(suggestion);
        }

        // For subsequent stages, we would need to implement the logic to:
        // 1. Evaluate partial results
        // 2. Select top performers
        // 3. Continue training them with more resources
        // This requires integration with the training loop

        None // For now, only support the initial stage
    }

    fn should_terminate(&self, _history: &TrialHistory) -> bool {
        self.current_stage >= self.halving_strategy.max_stages()
            || (self.current_stage == 0 && self.trials_suggested >= self.initial_configs)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Hyperband algorithm combining successive halving with different resource allocations
#[allow(dead_code)]
pub struct Hyperband {
    /// Maximum resource budget
    #[allow(dead_code)]
    max_resource: f64,
    /// Reduction factor
    reduction_factor: usize,
    /// Current bracket
    current_bracket: usize,
    /// Total brackets to run
    total_brackets: usize,
    /// Successive halving instances for each bracket
    successive_halving_instances: Vec<SuccessiveHalving>,
    /// Name of the strategy
    name: String,
}

impl Hyperband {
    /// Create a new Hyperband strategy
    pub fn new(max_resource: f64, reduction_factor: usize) -> Self {
        let total_brackets = (max_resource.log(reduction_factor as f64)).floor() as usize + 1;
        let mut instances = Vec::new();

        for bracket in 0..total_brackets {
            let s = total_brackets - 1 - bracket;
            let n = ((total_brackets * reduction_factor.pow(s as u32)) as f64 / (s + 1) as f64)
                .ceil() as usize;
            let r = max_resource / reduction_factor.pow(s as u32) as f64;

            instances.push(SuccessiveHalving::new(r, n));
        }

        Self {
            max_resource,
            reduction_factor,
            current_bracket: 0,
            total_brackets,
            successive_halving_instances: instances,
            name: format!(
                "Hyperband(max_resource={}, reduction_factor={})",
                max_resource, reduction_factor
            ),
        }
    }
}

impl SearchStrategy for Hyperband {
    fn suggest(
        &mut self,
        search_space: &SearchSpace,
        history: &TrialHistory,
    ) -> Option<HashMap<String, ParameterValue>> {
        if self.current_bracket >= self.total_brackets {
            return None;
        }

        let suggestion =
            self.successive_halving_instances[self.current_bracket].suggest(search_space, history);

        if suggestion.is_none() {
            // Current bracket is done, move to next
            self.current_bracket += 1;
            if self.current_bracket < self.total_brackets {
                return self.successive_halving_instances[self.current_bracket]
                    .suggest(search_space, history);
            }
        }

        suggestion
    }

    fn should_terminate(&self, _history: &TrialHistory) -> bool {
        self.current_bracket >= self.total_brackets
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Population-based Training (PBT) member
#[derive(Debug, Clone)]
pub struct PBTMember {
    /// Unique identifier for this member
    pub id: usize,
    /// Current hyperparameters
    pub hyperparameters: HashMap<String, ParameterValue>,
    /// Current performance score
    pub score: f64,
    /// Step/epoch when this member was last evaluated
    pub last_evaluated_step: usize,
    /// Training step when this member was created or last exploited
    pub creation_step: usize,
    /// Whether this member is still active
    pub is_active: bool,
    /// History of performance scores
    pub score_history: Vec<(usize, f64)>,
    /// Model state (in a real implementation, this would be the actual model weights)
    pub model_state: Option<Vec<u8>>,
}

impl PBTMember {
    pub fn new(id: usize, hyperparameters: HashMap<String, ParameterValue>) -> Self {
        Self {
            id,
            hyperparameters,
            score: f64::NEG_INFINITY,
            last_evaluated_step: 0,
            creation_step: 0,
            is_active: true,
            score_history: Vec::new(),
            model_state: None,
        }
    }

    pub fn update_score(&mut self, score: f64, step: usize) {
        self.score = score;
        self.last_evaluated_step = step;
        self.score_history.push((step, score));
    }
}

/// Configuration for Population-based Training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBTConfig {
    /// Size of the population
    pub population_size: usize,
    /// How often to perform exploit/explore (in training steps)
    pub exploit_interval: usize,
    /// Fraction of population to replace during exploitation
    pub exploit_fraction: f64,
    /// Standard deviation for parameter perturbation during exploration
    pub perturbation_std: f64,
    /// Minimum number of steps before exploitation can occur
    pub min_steps_before_exploit: usize,
    /// Whether to use tournament selection for exploitation
    pub use_tournament_selection: bool,
    /// Tournament size (if using tournament selection)
    pub tournament_size: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for PBTConfig {
    fn default() -> Self {
        Self {
            population_size: 10,
            exploit_interval: 1000,
            exploit_fraction: 0.2,
            perturbation_std: 0.1,
            min_steps_before_exploit: 500,
            use_tournament_selection: true,
            tournament_size: 3,
            seed: None,
        }
    }
}

/// Population-based Training strategy
pub struct PopulationBasedTraining {
    /// Configuration
    config: PBTConfig,
    /// Population of training members
    population: Vec<PBTMember>,
    /// Current training step
    current_step: usize,
    /// Next member ID to assign
    next_member_id: usize,
    /// Random sampler for parameter perturbation
    sampler: Box<dyn Sampler>,
    /// Members that need to be suggested
    pending_suggestions: Vec<HashMap<String, ParameterValue>>,
    /// Name of the strategy
    name: String,
    /// Shared state for population updates
    #[allow(dead_code)]
    shared_state: Arc<Mutex<PBTSharedState>>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct PBTSharedState {
    /// Performance updates from parallel training
    #[allow(dead_code)]
    performance_updates: Vec<(usize, f64, usize)>, // (member_id, score, step)
    /// Completed exploitations
    exploitations: Vec<(usize, usize)>, // (exploiter_id, exploited_id)
}

impl PopulationBasedTraining {
    /// Create a new Population-based Training strategy
    pub fn new(config: PBTConfig, search_space: &SearchSpace) -> Self {
        let mut population = Vec::new();
        let mut sampler: Box<dyn Sampler> = if let Some(seed) = config.seed {
            Box::new(RandomSampler::with_seed(seed))
        } else {
            Box::new(RandomSampler::new())
        };

        // Initialize population with random hyperparameters
        for i in 0..config.population_size {
            let hyperparameters =
                sampler.sample(search_space, &TrialHistory::new(Direction::Maximize));
            population.push(PBTMember::new(i, hyperparameters));
        }

        let pending_suggestions = population.iter().map(|m| m.hyperparameters.clone()).collect();

        Self {
            config: config.clone(),
            population,
            current_step: 0,
            next_member_id: config.population_size,
            sampler,
            pending_suggestions,
            name: format!(
                "PBT(pop_size={}, exploit_interval={})",
                config.population_size, config.exploit_interval
            ),
            shared_state: Arc::new(Mutex::new(PBTSharedState {
                performance_updates: Vec::new(),
                exploitations: Vec::new(),
            })),
        }
    }

    /// Update performance for a population member
    pub fn update_member_performance(&mut self, member_id: usize, score: f64, step: usize) {
        if let Some(member) = self.population.iter_mut().find(|m| m.id == member_id) {
            member.update_score(score, step);
        }
    }

    /// Perform exploitation: replace worst performers with copies of best performers
    fn exploit(&mut self, search_space: &SearchSpace) -> Vec<HashMap<String, ParameterValue>> {
        let mut suggestions = Vec::new();

        if self.current_step < self.config.min_steps_before_exploit {
            return suggestions;
        }

        // Sort population by performance
        let mut sorted_indices: Vec<usize> = (0..self.population.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            self.population[b]
                .score
                .partial_cmp(&self.population[a].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let num_to_replace = (self.config.exploit_fraction * self.population.len() as f64) as usize;
        let num_to_replace = num_to_replace.min(self.population.len() / 2);

        // Replace worst performers
        for i in 0..num_to_replace {
            let worst_idx = sorted_indices[sorted_indices.len() - 1 - i];
            let best_idx = if self.config.use_tournament_selection {
                self.tournament_selection(&sorted_indices)
            } else {
                sorted_indices[i % (sorted_indices.len() / 2)]
            };

            // Copy from best to worst
            let best_member = self.population[best_idx].clone();

            // Extract hyperparameters for perturbation first
            let params_to_perturb = {
                let worst_member = &mut self.population[worst_idx];

                // Exploit: copy hyperparameters and model state
                worst_member.hyperparameters = best_member.hyperparameters.clone();
                worst_member.model_state = best_member.model_state.clone();
                worst_member.creation_step = self.current_step;
                worst_member.score = f64::NEG_INFINITY; // Reset score
                worst_member.score_history.clear();

                // Extract hyperparameters for perturbation before calling perturb method
                worst_member.hyperparameters.clone()
            };

            // Explore: perturb hyperparameters (now we don't have a mutable borrow)
            let perturbed_params = self.perturb_hyperparameters(&params_to_perturb, search_space);

            // Update with perturbed parameters
            {
                let worst_member = &mut self.population[worst_idx];
                worst_member.hyperparameters = perturbed_params.clone();
            }

            suggestions.push(perturbed_params);
        }

        suggestions
    }

    /// Tournament selection for choosing a member to exploit from
    fn tournament_selection(&self, sorted_indices: &[usize]) -> usize {
        let mut best_idx = sorted_indices[0];
        let mut best_score = self.population[best_idx].score;

        for _ in 1..self.config.tournament_size {
            let candidate_idx = sorted_indices[fastrand::usize(0..sorted_indices.len())];
            let candidate_score = self.population[candidate_idx].score;

            if candidate_score > best_score {
                best_idx = candidate_idx;
                best_score = candidate_score;
            }
        }

        best_idx
    }

    /// Perturb hyperparameters for exploration
    fn perturb_hyperparameters(
        &self,
        hyperparameters: &HashMap<String, ParameterValue>,
        search_space: &SearchSpace,
    ) -> HashMap<String, ParameterValue> {
        let mut perturbed = hyperparameters.clone();

        for param in &search_space.parameters {
            let param_name = param.name();
            if let Some(value) = perturbed.get_mut(param_name) {
                match param {
                    super::search_space::HyperParameter::Continuous(p) => {
                        if let ParameterValue::Float(v) = value {
                            // Log-normal perturbation
                            let log_v = v.ln();
                            let noise = fastrand::f64() * 2.0 - 1.0; // [-1, 1]
                            let perturbed_log = log_v + noise * self.config.perturbation_std;
                            *v = perturbed_log.exp().clamp(p.low, p.high);
                        }
                    },
                    super::search_space::HyperParameter::Log(p) => {
                        if let ParameterValue::Float(v) = value {
                            let log_v = v.ln();
                            let noise = fastrand::f64() * 2.0 - 1.0;
                            let perturbed_log = log_v + noise * self.config.perturbation_std;
                            *v = perturbed_log.exp().clamp(p.low, p.high);
                        }
                    },
                    super::search_space::HyperParameter::Discrete(p) => {
                        if let ParameterValue::Int(v) = value {
                            let noise = ((fastrand::f64() * 2.0 - 1.0)
                                * self.config.perturbation_std
                                * 2.0) as i64;
                            *v = (*v + noise).clamp(p.low, p.high);
                        }
                    },
                    super::search_space::HyperParameter::Categorical(p) => {
                        // Randomly choose a different category with some probability
                        if fastrand::f64() < self.config.perturbation_std {
                            let new_choice = &p.choices[fastrand::usize(0..p.choices.len())];
                            *value = ParameterValue::String(new_choice.clone());
                        }
                    },
                }
            }
        }

        perturbed
    }

    /// Get the current population statistics
    pub fn get_population_stats(&self) -> PBTStats {
        let scores: Vec<f64> = self.population.iter().map(|m| m.score).collect();
        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_score = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        PBTStats {
            population_size: self.population.len(),
            current_step: self.current_step,
            mean_score,
            max_score,
            min_score,
            active_members: self.population.iter().filter(|m| m.is_active).count(),
        }
    }

    /// Get the best member from the population
    pub fn get_best_member(&self) -> Option<&PBTMember> {
        self.population
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    }
}

impl SearchStrategy for PopulationBasedTraining {
    fn suggest(
        &mut self,
        search_space: &SearchSpace,
        _history: &TrialHistory,
    ) -> Option<HashMap<String, ParameterValue>> {
        // Return pending suggestions first
        if !self.pending_suggestions.is_empty() {
            return Some(self.pending_suggestions.remove(0));
        }

        // Check if it's time to exploit
        if self.current_step % self.config.exploit_interval == 0 && self.current_step > 0 {
            let new_suggestions = self.exploit(search_space);
            self.pending_suggestions.extend(new_suggestions);

            if !self.pending_suggestions.is_empty() {
                return Some(self.pending_suggestions.remove(0));
            }
        }

        // Generate new random suggestion if needed
        if self.population.len() < self.config.population_size {
            let hyperparameters =
                self.sampler.sample(search_space, &TrialHistory::new(Direction::Maximize));
            self.population
                .push(PBTMember::new(self.next_member_id, hyperparameters.clone()));
            self.next_member_id += 1;
            return Some(hyperparameters);
        }

        None
    }

    fn should_terminate(&self, _history: &TrialHistory) -> bool {
        // PBT runs indefinitely until externally stopped
        false
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn update(&mut self, trial: &Trial) {
        // Update the corresponding population member
        if let Some(member_id) = trial.user_attrs.get("pbt_member_id") {
            if let Ok(id) = member_id.parse::<i64>() {
                let objective_value =
                    trial.result.as_ref().map(|r| r.metrics.objective_value).unwrap_or(0.0);
                self.update_member_performance(id as usize, objective_value, self.current_step);
            }
        }
        self.current_step += 1;
    }
}

/// Statistics for Population-based Training
#[derive(Debug, Clone)]
pub struct PBTStats {
    pub population_size: usize,
    pub current_step: usize,
    pub mean_score: f64,
    pub max_score: f64,
    pub min_score: f64,
    pub active_members: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperopt::{search_space::SearchSpaceBuilder, Direction, TrialHistory};

    fn create_discrete_search_space() -> SearchSpace {
        SearchSpaceBuilder::new()
            .discrete("batch_size", 8, 32, 8)
            .categorical("optimizer", vec!["adam", "sgd"])
            .build()
    }

    fn create_mixed_search_space() -> SearchSpace {
        SearchSpaceBuilder::new()
            .continuous("learning_rate", 1e-5, 1e-1)
            .discrete("batch_size", 8, 32, 8)
            .categorical("optimizer", vec!["adam", "sgd"])
            .build()
    }

    #[test]
    fn test_grid_search() {
        let search_space = create_discrete_search_space();
        let mut strategy = GridSearch::new(&search_space).unwrap();
        let history = TrialHistory::new(Direction::Maximize);

        // Should suggest all combinations
        let total_combinations = strategy.total_combinations();
        assert_eq!(total_combinations, 4 * 2); // 4 batch sizes * 2 optimizers

        let mut suggestions = Vec::new();
        while let Some(suggestion) = strategy.suggest(&search_space, &history) {
            suggestions.push(suggestion);
        }

        assert_eq!(suggestions.len(), total_combinations);
        assert!(strategy.should_terminate(&history));

        // All suggestions should be valid
        for suggestion in &suggestions {
            assert!(search_space.validate(suggestion).is_ok());
        }
    }

    #[test]
    fn test_grid_search_with_continuous_params() {
        let search_space = create_mixed_search_space();
        let result = GridSearch::new(&search_space);

        // Should fail because continuous parameters are not supported
        assert!(result.is_err());
    }

    #[test]
    fn test_random_search() {
        let search_space = create_mixed_search_space();
        let mut strategy = RandomSearch::with_seed(10, 42);
        let history = TrialHistory::new(Direction::Maximize);

        let mut suggestions = Vec::new();
        while let Some(suggestion) = strategy.suggest(&search_space, &history) {
            suggestions.push(suggestion);
            if strategy.should_terminate(&history) {
                break;
            }
        }

        assert_eq!(suggestions.len(), 10);
        assert!(strategy.should_terminate(&history));

        // All suggestions should be valid
        for suggestion in &suggestions {
            assert!(search_space.validate(suggestion).is_ok());
        }
    }

    #[test]
    fn test_bayesian_optimization() {
        let search_space = create_mixed_search_space();
        let mut strategy = BayesianOptimization::new(5);
        let history = TrialHistory::new(Direction::Maximize);

        let mut suggestions = Vec::new();
        while let Some(suggestion) = strategy.suggest(&search_space, &history) {
            suggestions.push(suggestion);
            if strategy.should_terminate(&history) {
                break;
            }
        }

        assert_eq!(suggestions.len(), 5);
        assert!(strategy.should_terminate(&history));

        // All suggestions should be valid
        for suggestion in &suggestions {
            assert!(search_space.validate(suggestion).is_ok());
        }
    }

    #[test]
    fn test_successive_halving() {
        let search_space = create_mixed_search_space();
        let mut strategy = SuccessiveHalving::new(100.0, 9);
        let history = TrialHistory::new(Direction::Maximize);

        let mut suggestions = Vec::new();
        while let Some(suggestion) = strategy.suggest(&search_space, &history) {
            suggestions.push(suggestion);
            if strategy.should_terminate(&history) {
                break;
            }
        }

        assert_eq!(suggestions.len(), 9); // Initial configurations
        assert!(strategy.should_terminate(&history));

        // Check resource allocation
        assert_eq!(
            strategy.current_resource_allocation(),
            100.0 / 3_f64.powi(0)
        );
    }

    #[test]
    fn test_hyperband() {
        let search_space = create_mixed_search_space();
        let mut strategy = Hyperband::new(81.0, 3);
        let history = TrialHistory::new(Direction::Maximize);

        let mut suggestions = Vec::new();
        let mut count = 0;
        while let Some(suggestion) = strategy.suggest(&search_space, &history) {
            suggestions.push(suggestion);
            count += 1;
            if strategy.should_terminate(&history) || count > 100 {
                break;
            }
        }

        assert!(!suggestions.is_empty());

        // All suggestions should be valid
        for suggestion in &suggestions {
            assert!(search_space.validate(suggestion).is_ok());
        }
    }

    #[test]
    fn test_standard_halving() {
        let halving = StandardHalving::new(3);

        assert_eq!(halving.get_resource_allocation(0, 100.0), 100.0);
        assert_eq!(halving.get_resource_allocation(1, 100.0), 100.0 / 3.0);
        assert_eq!(halving.get_resource_allocation(2, 100.0), 100.0 / 9.0);

        assert_eq!(halving.get_trials_to_keep(9), 3);
        assert_eq!(halving.get_trials_to_keep(3), 1);
        assert_eq!(halving.get_trials_to_keep(1), 1);
    }

    #[test]
    fn test_pbt_config_default() {
        let config = PBTConfig::default();
        assert_eq!(config.population_size, 10);
        assert_eq!(config.exploit_interval, 1000);
        assert_eq!(config.exploit_fraction, 0.2);
        assert_eq!(config.perturbation_std, 0.1);
        assert_eq!(config.min_steps_before_exploit, 500);
        assert!(config.use_tournament_selection);
        assert_eq!(config.tournament_size, 3);
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_pbt_member_creation() {
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("learning_rate".to_string(), ParameterValue::Float(0.01));
        hyperparameters.insert("batch_size".to_string(), ParameterValue::Int(32));

        let member = PBTMember::new(0, hyperparameters.clone());
        assert_eq!(member.id, 0);
        assert_eq!(member.hyperparameters, hyperparameters);
        assert_eq!(member.score, f64::NEG_INFINITY);
        assert_eq!(member.last_evaluated_step, 0);
        assert_eq!(member.creation_step, 0);
        assert!(member.is_active);
        assert!(member.score_history.is_empty());
        assert!(member.model_state.is_none());
    }

    #[test]
    fn test_pbt_member_update_score() {
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("learning_rate".to_string(), ParameterValue::Float(0.01));

        let mut member = PBTMember::new(0, hyperparameters);
        member.update_score(0.85, 100);

        assert_eq!(member.score, 0.85);
        assert_eq!(member.last_evaluated_step, 100);
        assert_eq!(member.score_history.len(), 1);
        assert_eq!(member.score_history[0], (100, 0.85));
    }

    #[test]
    fn test_pbt_strategy_creation() {
        let search_space = create_mixed_search_space();
        let config = PBTConfig::default();
        let strategy = PopulationBasedTraining::new(config.clone(), &search_space);

        assert_eq!(strategy.population.len(), config.population_size);
        assert_eq!(strategy.current_step, 0);
        assert_eq!(strategy.next_member_id, config.population_size);
        assert_eq!(strategy.pending_suggestions.len(), config.population_size);

        // Check that all population members have valid hyperparameters
        for member in &strategy.population {
            assert!(search_space.validate(&member.hyperparameters).is_ok());
        }
    }

    #[test]
    fn test_pbt_strategy_suggestions() {
        let search_space = create_mixed_search_space();
        let config = PBTConfig {
            population_size: 3,
            exploit_interval: 5,
            ..Default::default()
        };
        let mut strategy = PopulationBasedTraining::new(config, &search_space);
        let history = TrialHistory::new(Direction::Maximize);

        // Should return initial population suggestions
        for _i in 0..3 {
            let suggestion = strategy.suggest(&search_space, &history);
            assert!(suggestion.is_some());
            assert!(search_space.validate(&suggestion.unwrap()).is_ok());
        }

        // Should return None after initial population
        let suggestion = strategy.suggest(&search_space, &history);
        assert!(suggestion.is_none());
    }

    #[test]
    fn test_pbt_population_stats() {
        let search_space = create_mixed_search_space();
        let config = PBTConfig::default();
        let mut strategy = PopulationBasedTraining::new(config, &search_space);

        // Update some scores
        strategy.update_member_performance(0, 0.9, 100);
        strategy.update_member_performance(1, 0.8, 100);
        strategy.update_member_performance(2, 0.7, 100);

        let stats = strategy.get_population_stats();
        assert_eq!(stats.population_size, 10);
        assert_eq!(stats.current_step, 0);
        assert_eq!(stats.active_members, 10);
        assert!(stats.max_score > stats.min_score);
    }

    #[test]
    fn test_pbt_best_member() {
        let search_space = create_mixed_search_space();
        let config = PBTConfig::default();
        let mut strategy = PopulationBasedTraining::new(config, &search_space);

        // Update some scores
        strategy.update_member_performance(0, 0.9, 100);
        strategy.update_member_performance(1, 0.8, 100);
        strategy.update_member_performance(2, 0.95, 100);

        let best_member = strategy.get_best_member();
        assert!(best_member.is_some());
        assert_eq!(best_member.unwrap().id, 2);
        assert_eq!(best_member.unwrap().score, 0.95);
    }

    #[test]
    fn test_pbt_hyperparameter_perturbation() {
        let search_space = create_mixed_search_space();
        let config = PBTConfig {
            perturbation_std: 0.2,
            ..Default::default()
        };
        let strategy = PopulationBasedTraining::new(config, &search_space);

        let mut original_params = HashMap::new();
        original_params.insert("learning_rate".to_string(), ParameterValue::Float(0.01));
        original_params.insert("batch_size".to_string(), ParameterValue::Int(32));
        original_params.insert(
            "optimizer".to_string(),
            ParameterValue::String("adam".to_string()),
        );

        let perturbed_params = strategy.perturb_hyperparameters(&original_params, &search_space);

        // Parameters should be valid
        assert!(search_space.validate(&perturbed_params).is_ok());

        // Some parameters might be different (due to randomness)
        // But we can't guarantee they will be different in a single test
        assert_eq!(perturbed_params.len(), original_params.len());
    }
}
