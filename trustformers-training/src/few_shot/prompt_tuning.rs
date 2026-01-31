use anyhow::Result;
use scirs2_core::ndarray::{s, Array2, Array3}; // SciRS2 Integration Policy
use scirs2_core::random::*; // SciRS2 Integration Policy (for Normal, Uniform distributions)
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for prompt tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptConfig {
    /// Number of virtual tokens in the soft prompt
    pub prompt_length: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Learning rate for prompt parameters
    pub learning_rate: f32,
    /// Initialization strategy
    pub init_strategy: InitStrategy,
    /// Whether to freeze the main model
    pub freeze_model: bool,
    /// Regularization weight
    pub regularization_weight: f32,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            prompt_length: 20,
            embedding_dim: 768,
            learning_rate: 0.3,
            init_strategy: InitStrategy::Random,
            freeze_model: true,
            regularization_weight: 0.01,
            max_grad_norm: 1.0,
        }
    }
}

/// Initialization strategies for soft prompts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitStrategy {
    /// Random initialization from normal distribution
    Random,
    /// Initialize from existing vocabulary embeddings
    Vocabulary(Vec<String>),
    /// Initialize from task-specific examples
    TaskSpecific,
    /// Xavier/Glorot initialization
    Xavier,
    /// He initialization
    He,
}

/// Soft prompt implementation
#[derive(Debug, Clone)]
pub struct SoftPrompt {
    /// Prompt embeddings [prompt_length, embedding_dim]
    pub embeddings: Array2<f32>,
    /// Gradient accumulator
    gradients: Array2<f32>,
    /// Configuration
    config: PromptConfig,
    /// Task identifier
    task_id: String,
    /// Training step counter
    step: usize,
}

impl SoftPrompt {
    pub fn new(config: PromptConfig, task_id: String) -> Result<Self> {
        let embeddings = Self::initialize_embeddings(&config)?;
        let gradients = Array2::zeros((config.prompt_length, config.embedding_dim));

        Ok(Self {
            embeddings,
            gradients,
            config,
            task_id,
            step: 0,
        })
    }

    /// Initialize prompt embeddings based on strategy
    fn initialize_embeddings(config: &PromptConfig) -> Result<Array2<f32>> {
        let shape = (config.prompt_length, config.embedding_dim);

        let embeddings = match &config.init_strategy {
            InitStrategy::Random => {
                let std = 0.02; // Standard initialization for prompts
                let mut rng = thread_rng();
                let normal = Normal::new(0.0, std)?;
                Array2::from_shape_fn(shape, |_| normal.sample(&mut rng) as f32)
            },
            InitStrategy::Vocabulary(_tokens) => {
                // In practice, would use actual vocabulary embeddings
                let std = 0.02;
                let mut rng = thread_rng();
                let normal = Normal::new(0.0, std)?;
                Array2::from_shape_fn(shape, |_| normal.sample(&mut rng) as f32)
            },
            InitStrategy::TaskSpecific => {
                // Initialize based on task examples (simplified)
                let std = 0.01;
                let mut rng = thread_rng();
                let normal = Normal::new(0.0, std)?;
                Array2::from_shape_fn(shape, |_| normal.sample(&mut rng) as f32)
            },
            InitStrategy::Xavier => {
                let bound = (6.0 / (config.prompt_length + config.embedding_dim) as f32).sqrt();
                let mut rng = thread_rng();
                let uniform = Uniform::new(-bound, bound)?;
                Array2::from_shape_fn(shape, |_| uniform.sample(&mut rng))
            },
            InitStrategy::He => {
                let std = (2.0 / config.prompt_length as f32).sqrt();
                let mut rng = thread_rng();
                let normal = Normal::new(0.0, std)?;
                Array2::from_shape_fn(shape, |_| normal.sample(&mut rng) as f32)
            },
        };

        Ok(embeddings)
    }

    /// Get prompt embeddings for concatenation with input
    pub fn get_embeddings(&self) -> &Array2<f32> {
        &self.embeddings
    }

    /// Update prompt parameters with gradients
    pub fn update(&mut self, gradients: &Array2<f32>) -> Result<()> {
        if gradients.shape() != self.embeddings.shape() {
            return Err(anyhow::anyhow!("Gradient shape mismatch"));
        }

        // Gradient clipping
        let grad_norm = self.compute_gradient_norm(gradients);
        let clipped_gradients = if grad_norm > self.config.max_grad_norm {
            gradients * (self.config.max_grad_norm / grad_norm)
        } else {
            gradients.clone()
        };

        // Apply L2 regularization
        let regularized_gradients =
            &clipped_gradients + self.config.regularization_weight * &self.embeddings;

        // Update embeddings
        self.embeddings = &self.embeddings - self.config.learning_rate * &regularized_gradients;
        self.gradients = clipped_gradients;
        self.step += 1;

        Ok(())
    }

    /// Compute gradient norm for clipping
    fn compute_gradient_norm(&self, gradients: &Array2<f32>) -> f32 {
        gradients.mapv(|x| x * x).sum().sqrt()
    }

    /// Get current training step
    pub fn get_step(&self) -> usize {
        self.step
    }

    /// Reset gradients
    pub fn zero_grad(&mut self) {
        self.gradients.fill(0.0);
    }

    /// Get task ID
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// Save prompt to file
    pub fn save(&self, path: &str) -> Result<()> {
        let data_to_serialize = (
            self.embeddings.as_slice().expect("embeddings tensor should have valid slice"),
            self.embeddings.shape(),
            &self.config,
            &self.task_id,
            self.step,
        );
        let serialized = rmp_serde::to_vec(&data_to_serialize)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Load prompt from file
    pub fn load(path: &str) -> Result<Self> {
        let data = std::fs::read(path)?;
        let (embeddings_data, shape, config, task_id, step): (
            Vec<f32>,
            Vec<usize>,
            PromptConfig,
            String,
            usize,
        ) = rmp_serde::from_slice(&data)?;

        let embeddings = Array2::from_shape_vec((shape[0], shape[1]), embeddings_data)?;
        let gradients = Array2::zeros((config.prompt_length, config.embedding_dim));

        Ok(Self {
            embeddings,
            gradients,
            config,
            task_id,
            step,
        })
    }
}

/// Prompt tuning trainer
pub struct PromptTuner {
    /// Collection of soft prompts for different tasks
    prompts: HashMap<String, SoftPrompt>,
    /// Global configuration
    config: PromptConfig,
    /// Training statistics
    stats: TrainingStats,
}

impl PromptTuner {
    pub fn new(config: PromptConfig) -> Self {
        Self {
            prompts: HashMap::new(),
            config,
            stats: TrainingStats::new(),
        }
    }

    /// Create new prompt for a task
    pub fn create_prompt(&mut self, task_id: String) -> Result<()> {
        let prompt = SoftPrompt::new(self.config.clone(), task_id.clone())?;
        self.prompts.insert(task_id, prompt);
        Ok(())
    }

    /// Get prompt for a task
    pub fn get_prompt(&self, task_id: &str) -> Option<&SoftPrompt> {
        self.prompts.get(task_id)
    }

    /// Get mutable prompt for a task
    pub fn get_prompt_mut(&mut self, task_id: &str) -> Option<&mut SoftPrompt> {
        self.prompts.get_mut(task_id)
    }

    /// Train prompt with gradients
    pub fn train_step(&mut self, task_id: &str, gradients: &Array2<f32>, loss: f32) -> Result<()> {
        let prompt = self
            .prompts
            .get_mut(task_id)
            .ok_or_else(|| anyhow::anyhow!("Prompt not found for task: {}", task_id))?;

        prompt.update(gradients)?;
        self.stats.record_step(task_id, loss, gradients);

        Ok(())
    }

    /// Concatenate soft prompt with input embeddings
    pub fn apply_prompt(
        &self,
        task_id: &str,
        input_embeddings: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        let prompt = self
            .get_prompt(task_id)
            .ok_or_else(|| anyhow::anyhow!("Prompt not found for task: {}", task_id))?;

        // Concatenate prompt embeddings with input embeddings
        let prompt_embeddings = prompt.get_embeddings();
        let mut concatenated = Array2::zeros((
            prompt_embeddings.nrows() + input_embeddings.nrows(),
            prompt_embeddings.ncols(),
        ));
        concatenated
            .slice_mut(s![..prompt_embeddings.nrows(), ..])
            .assign(prompt_embeddings);
        concatenated
            .slice_mut(s![prompt_embeddings.nrows().., ..])
            .assign(input_embeddings);

        Ok(concatenated)
    }

    /// Apply prompt to batch of inputs
    pub fn apply_prompt_batch(
        &self,
        task_id: &str,
        input_batch: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        let prompt = self
            .get_prompt(task_id)
            .ok_or_else(|| anyhow::anyhow!("Prompt not found for task: {}", task_id))?;

        let (batch_size, seq_len, embed_dim) = input_batch.dim();
        let prompt_embeddings = prompt.get_embeddings();
        let prompt_len = prompt_embeddings.nrows();

        // Create batch of prompts
        let mut prompt_batch = Array3::zeros((batch_size, prompt_len, embed_dim));
        for i in 0..batch_size {
            prompt_batch.slice_mut(s![i, .., ..]).assign(prompt_embeddings);
        }

        // Concatenate with input batch
        let mut concatenated = Array3::zeros((batch_size, prompt_len + seq_len, embed_dim));
        concatenated.slice_mut(s![.., ..prompt_len, ..]).assign(&prompt_batch);
        concatenated.slice_mut(s![.., prompt_len.., ..]).assign(input_batch);

        Ok(concatenated)
    }

    /// Get training statistics
    pub fn get_stats(&self) -> &TrainingStats {
        &self.stats
    }

    /// Reset training statistics
    pub fn reset_stats(&mut self) {
        self.stats = TrainingStats::new();
    }

    /// Save all prompts
    pub fn save_prompts(&self, base_path: &str) -> Result<()> {
        std::fs::create_dir_all(base_path)?;

        for (task_id, prompt) in &self.prompts {
            let path = format!("{}/{}_prompt.bin", base_path, task_id);
            prompt.save(&path)?;
        }

        Ok(())
    }

    /// Load prompts from directory
    pub fn load_prompts(&mut self, base_path: &str) -> Result<()> {
        for entry in std::fs::read_dir(base_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Some(task_id) = stem.strip_suffix("_prompt") {
                        let prompt = SoftPrompt::load(path.to_str().unwrap())?;
                        self.prompts.insert(task_id.to_string(), prompt);
                    }
                }
            }
        }

        Ok(())
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Loss per task
    task_losses: HashMap<String, Vec<f32>>,
    /// Gradient norms per task
    gradient_norms: HashMap<String, Vec<f32>>,
    /// Training steps per task
    steps: HashMap<String, usize>,
}

impl TrainingStats {
    fn new() -> Self {
        Self {
            task_losses: HashMap::new(),
            gradient_norms: HashMap::new(),
            steps: HashMap::new(),
        }
    }

    fn record_step(&mut self, task_id: &str, loss: f32, gradients: &Array2<f32>) {
        let grad_norm = gradients.mapv(|x| x * x).sum().sqrt();

        self.task_losses.entry(task_id.to_string()).or_default().push(loss);

        self.gradient_norms.entry(task_id.to_string()).or_default().push(grad_norm);

        *self.steps.entry(task_id.to_string()).or_insert(0) += 1;
    }

    /// Get average loss for a task
    pub fn get_average_loss(&self, task_id: &str) -> Option<f32> {
        self.task_losses
            .get(task_id)
            .map(|losses| losses.iter().sum::<f32>() / losses.len() as f32)
    }

    /// Get latest loss for a task
    pub fn get_latest_loss(&self, task_id: &str) -> Option<f32> {
        self.task_losses.get(task_id).and_then(|losses| losses.last().copied())
    }

    /// Get training steps for a task
    pub fn get_steps(&self, task_id: &str) -> usize {
        self.steps.get(task_id).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_soft_prompt_creation() {
        let config = PromptConfig::default();
        let prompt = SoftPrompt::new(config.clone(), "test_task".to_string()).unwrap();

        assert_eq!(
            prompt.embeddings.shape(),
            &[config.prompt_length, config.embedding_dim]
        );
        assert_eq!(prompt.task_id(), "test_task");
        assert_eq!(prompt.get_step(), 0);
    }

    #[test]
    fn test_prompt_update() {
        let config = PromptConfig::default();
        let mut prompt = SoftPrompt::new(config.clone(), "test_task".to_string()).unwrap();

        let gradients = Array2::ones((config.prompt_length, config.embedding_dim));
        let initial_embeddings = prompt.embeddings.clone();

        prompt.update(&gradients).unwrap();

        assert_ne!(prompt.embeddings, initial_embeddings);
        assert_eq!(prompt.get_step(), 1);
    }

    #[test]
    fn test_prompt_tuner() {
        let config = PromptConfig::default();
        let mut tuner = PromptTuner::new(config.clone());

        tuner.create_prompt("task1".to_string()).unwrap();
        tuner.create_prompt("task2".to_string()).unwrap();

        assert!(tuner.get_prompt("task1").is_some());
        assert!(tuner.get_prompt("task2").is_some());
        assert!(tuner.get_prompt("task3").is_none());
    }

    #[test]
    fn test_apply_prompt() {
        let config = PromptConfig {
            prompt_length: 5,
            embedding_dim: 10,
            ..Default::default()
        };
        let mut tuner = PromptTuner::new(config.clone());
        tuner.create_prompt("test_task".to_string()).unwrap();

        let input_embeddings = Array2::ones((8, 10)); // 8 tokens, 10 dims
        let result = tuner.apply_prompt("test_task", &input_embeddings).unwrap();

        assert_eq!(result.shape(), &[13, 10]); // 5 prompt + 8 input tokens
    }

    #[test]
    fn test_apply_prompt_batch() {
        let config = PromptConfig {
            prompt_length: 3,
            embedding_dim: 5,
            ..Default::default()
        };
        let mut tuner = PromptTuner::new(config.clone());
        tuner.create_prompt("test_task".to_string()).unwrap();

        let input_batch = Array3::ones((2, 4, 5)); // batch_size=2, seq_len=4, embed_dim=5
        let result = tuner.apply_prompt_batch("test_task", &input_batch).unwrap();

        assert_eq!(result.shape(), &[2, 7, 5]); // 3 prompt + 4 input tokens per batch
    }

    #[test]
    fn test_training_stats() {
        let mut stats = TrainingStats::new();
        let gradients = Array2::ones((5, 10));

        stats.record_step("task1", 1.5, &gradients);
        stats.record_step("task1", 1.2, &gradients);

        assert_abs_diff_eq!(
            stats.get_average_loss("task1").unwrap(),
            1.35,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(stats.get_latest_loss("task1").unwrap(), 1.2, epsilon = 1e-6);
        assert_eq!(stats.get_steps("task1"), 2);
    }

    #[test]
    fn test_initialization_strategies() {
        let configs = vec![
            PromptConfig {
                init_strategy: InitStrategy::Random,
                ..Default::default()
            },
            PromptConfig {
                init_strategy: InitStrategy::Xavier,
                ..Default::default()
            },
            PromptConfig {
                init_strategy: InitStrategy::He,
                ..Default::default()
            },
        ];

        for config in configs {
            let prompt = SoftPrompt::new(config.clone(), "test".to_string()).unwrap();
            assert_eq!(
                prompt.embeddings.shape(),
                &[config.prompt_length, config.embedding_dim]
            );
        }
    }
}
