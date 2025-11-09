use anyhow::Result;
use ndarray_rand::RandomExt;
use scirs2_core::ndarray::{Array1, Array2, Axis}; // SciRS2 Integration Policy
use scirs2_core::random::*; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::result; // Required for bincode derives

/// Task descriptor containing metadata about a task
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct TaskDescriptor {
    /// Unique task identifier
    pub task_id: String,
    /// Task type (classification, regression, etc.)
    pub task_type: TaskType,
    /// Number of classes (for classification)
    pub num_classes: Option<usize>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Task-specific metadata
    pub metadata: HashMap<String, String>,
    /// Domain information
    pub domain: String,
    /// Language (for NLP tasks)
    pub language: Option<String>,
}

impl TaskDescriptor {
    pub fn new(task_id: String, task_type: TaskType, input_dim: usize, output_dim: usize) -> Self {
        Self {
            task_id,
            task_type,
            num_classes: None,
            input_dim,
            output_dim,
            metadata: HashMap::new(),
            domain: "general".to_string(),
            language: None,
        }
    }

    /// Create classification task descriptor
    pub fn classification(task_id: String, input_dim: usize, num_classes: usize) -> Self {
        Self {
            task_id,
            task_type: TaskType::Classification,
            num_classes: Some(num_classes),
            input_dim,
            output_dim: num_classes,
            metadata: HashMap::new(),
            domain: "general".to_string(),
            language: None,
        }
    }

    /// Create regression task descriptor
    pub fn regression(task_id: String, input_dim: usize, output_dim: usize) -> Self {
        Self {
            task_id,
            task_type: TaskType::Regression,
            num_classes: None,
            input_dim,
            output_dim,
            metadata: HashMap::new(),
            domain: "general".to_string(),
            language: None,
        }
    }
}

/// Types of tasks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bincode::Encode, bincode::Decode)]
pub enum TaskType {
    Classification,
    Regression,
    SequenceLabeling,
    Generation,
    Ranking,
    Clustering,
}

/// Configuration for task adaptation
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct AdaptationConfig {
    /// Adaptation strategy
    pub strategy: AdaptationStrategy,
    /// Learning rate for adaptation
    pub learning_rate: f32,
    /// Number of adaptation steps
    pub adaptation_steps: usize,
    /// Batch size for adaptation
    pub batch_size: usize,
    /// Regularization weight
    pub regularization_weight: f32,
    /// Whether to freeze base model parameters
    pub freeze_base: bool,
    /// Adapter configuration (if using adapters)
    pub adapter_config: Option<AdapterConfig>,
    /// Fine-tuning configuration
    pub fine_tune_config: Option<FineTuneConfig>,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            strategy: AdaptationStrategy::FineTuning,
            learning_rate: 0.001,
            adaptation_steps: 100,
            batch_size: 32,
            regularization_weight: 0.01,
            freeze_base: false,
            adapter_config: Some(AdapterConfig::default()),
            fine_tune_config: Some(FineTuneConfig::default()),
        }
    }
}

/// Different adaptation strategies
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub enum AdaptationStrategy {
    /// Full fine-tuning of the model
    FineTuning,
    /// Adapter layers
    Adapters,
    /// Linear probing (freeze features, train classifier)
    LinearProbing,
    /// Feature-based adaptation
    FeatureBased,
    /// Bias-only tuning
    BiasOnly,
    /// LoRA (Low-Rank Adaptation)
    LoRA,
    /// Prefix tuning
    PrefixTuning,
    /// Combined strategies
    Combined(Vec<AdaptationStrategy>),
}

/// Configuration for adapter layers
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct AdapterConfig {
    /// Adapter bottleneck dimension
    pub bottleneck_dim: usize,
    /// Dropout rate in adapters
    pub dropout_rate: f32,
    /// Activation function
    pub activation: ActivationFunction,
    /// Whether to use layer normalization
    pub use_layer_norm: bool,
    /// Adapter placement (which layers)
    pub placement: AdapterPlacement,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            bottleneck_dim: 64,
            dropout_rate: 0.1,
            activation: ActivationFunction::ReLU,
            use_layer_norm: true,
            placement: AdapterPlacement::AfterTransformer,
        }
    }
}

/// Where to place adapter layers
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub enum AdapterPlacement {
    AfterTransformer,
    AfterAttention,
    AfterFeedForward,
    Both,
    Custom(Vec<String>),
}

/// Activation functions for adapters
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub enum ActivationFunction {
    ReLU,
    GELU,
    Swish,
    Tanh,
    Sigmoid,
}

/// Configuration for fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct FineTuneConfig {
    /// Which layers to fine-tune
    pub layers_to_tune: LayerSelection,
    /// Learning rate decay strategy
    pub lr_decay: LearningRateDecay,
    /// Warmup steps
    pub warmup_steps: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
}

impl Default for FineTuneConfig {
    fn default() -> Self {
        Self {
            layers_to_tune: LayerSelection::All,
            lr_decay: LearningRateDecay::Cosine,
            warmup_steps: 100,
            gradient_accumulation_steps: 1,
            max_grad_norm: 1.0,
        }
    }
}

/// Which layers to fine-tune
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub enum LayerSelection {
    All,
    TopN(usize),
    BottomN(usize),
    Specific(Vec<String>),
    Classifier,
}

/// Learning rate decay strategies
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub enum LearningRateDecay {
    None,
    Linear,
    Cosine,
    Exponential(f32),
    Step(usize, f32),
}

/// Adapter layer implementation
#[derive(Debug, Clone)]
pub struct AdapterLayer {
    /// Down projection (hidden_dim -> bottleneck_dim)
    down_projection: Array2<f32>,
    /// Up projection (bottleneck_dim -> hidden_dim)
    up_projection: Array2<f32>,
    /// Bias terms
    down_bias: Array1<f32>,
    up_bias: Array1<f32>,
    /// Configuration
    config: AdapterConfig,
}

impl AdapterLayer {
    pub fn new(hidden_dim: usize, config: AdapterConfig) -> Result<Self> {
        let bottleneck_dim = config.bottleneck_dim;

        // Initialize weights (Xavier initialization)
        let down_bound = (6.0 / (hidden_dim + bottleneck_dim) as f32).sqrt();
        let up_bound = (6.0 / (bottleneck_dim + hidden_dim) as f32).sqrt();

        let down_projection = Array2::random(
            (hidden_dim, bottleneck_dim),
            ndarray_rand::rand_distr::Uniform::new(-down_bound, down_bound),
        );
        let up_projection = Array2::random(
            (bottleneck_dim, hidden_dim),
            ndarray_rand::rand_distr::Uniform::new(-up_bound, up_bound),
        );

        let down_bias = Array1::zeros(bottleneck_dim);
        let up_bias = Array1::zeros(hidden_dim);

        Ok(Self {
            down_projection,
            up_projection,
            down_bias,
            up_bias,
            config,
        })
    }

    /// Forward pass through adapter
    pub fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        // Down projection
        let down_output = input.dot(&self.down_projection) + &self.down_bias;

        // Apply activation
        let activated = self.apply_activation(&down_output);

        // Apply dropout (simplified - not implemented in this example)
        let dropped = activated; // In practice, would apply dropout

        // Up projection
        let up_output = dropped.dot(&self.up_projection) + &self.up_bias;

        // Residual connection
        let output = input + &up_output;

        Ok(output)
    }

    /// Apply activation function
    fn apply_activation(&self, input: &Array2<f32>) -> Array2<f32> {
        match self.config.activation {
            ActivationFunction::ReLU => input.mapv(|x| x.max(0.0)),
            ActivationFunction::GELU => input.mapv(|x| {
                0.5 * x
                    * (1.0
                        + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            }),
            ActivationFunction::Swish => input.mapv(|x| x / (1.0 + (-x).exp())),
            ActivationFunction::Tanh => input.mapv(|x| x.tanh()),
            ActivationFunction::Sigmoid => input.mapv(|x| 1.0 / (1.0 + (-x).exp())),
        }
    }

    /// Get trainable parameters
    pub fn get_parameters(&self) -> HashMap<String, Array2<f32>> {
        let mut params = HashMap::new();
        params.insert("down_projection".to_string(), self.down_projection.clone());
        params.insert("up_projection".to_string(), self.up_projection.clone());
        // Convert bias vectors to 2D for consistency
        params.insert(
            "down_bias".to_string(),
            self.down_bias.clone().insert_axis(Axis(0)),
        );
        params.insert(
            "up_bias".to_string(),
            self.up_bias.clone().insert_axis(Axis(0)),
        );
        params
    }

    /// Update parameters with gradients
    pub fn update_parameters(
        &mut self,
        gradients: &HashMap<String, Array2<f32>>,
        learning_rate: f32,
    ) -> Result<()> {
        if let Some(grad) = gradients.get("down_projection") {
            self.down_projection = &self.down_projection - learning_rate * grad;
        }
        if let Some(grad) = gradients.get("up_projection") {
            self.up_projection = &self.up_projection - learning_rate * grad;
        }
        if let Some(grad) = gradients.get("down_bias") {
            let bias_grad = grad.index_axis(Axis(0), 0);
            self.down_bias = &self.down_bias - learning_rate * &bias_grad;
        }
        if let Some(grad) = gradients.get("up_bias") {
            let bias_grad = grad.index_axis(Axis(0), 0);
            self.up_bias = &self.up_bias - learning_rate * &bias_grad;
        }
        Ok(())
    }
}

/// Task adapter for managing adaptation to specific tasks
pub struct TaskAdapter {
    /// Task descriptor
    task_descriptor: TaskDescriptor,
    /// Adaptation configuration
    config: AdaptationConfig,
    /// Adapter layers (if using adapter strategy)
    adapters: HashMap<String, AdapterLayer>,
    /// Task-specific classifier head
    classifier_head: Option<Array2<f32>>,
    /// Training statistics
    training_stats: AdaptationStats,
    /// Current adaptation step
    current_step: usize,
}

impl TaskAdapter {
    pub fn new(task_descriptor: TaskDescriptor, config: AdaptationConfig) -> Result<Self> {
        let mut adapter = Self {
            task_descriptor,
            config,
            adapters: HashMap::new(),
            classifier_head: None,
            training_stats: AdaptationStats::new(),
            current_step: 0,
        };

        adapter.initialize_components()?;
        Ok(adapter)
    }

    /// Initialize adaptation components based on strategy
    fn initialize_components(&mut self) -> Result<()> {
        match &self.config.strategy {
            AdaptationStrategy::Adapters => {
                if let Some(adapter_config) = &self.config.adapter_config {
                    // Create adapter layers (simplified - assuming one adapter)
                    let adapter =
                        AdapterLayer::new(self.task_descriptor.input_dim, adapter_config.clone())?;
                    self.adapters.insert("main_adapter".to_string(), adapter);
                }
            },
            AdaptationStrategy::LinearProbing => {
                // Initialize only classifier head
                self.initialize_classifier_head()?;
            },
            AdaptationStrategy::FineTuning => {
                // Initialize classifier head for task-specific output
                self.initialize_classifier_head()?;
            },
            _ => {
                // Other strategies can be implemented similarly
            },
        }

        Ok(())
    }

    /// Initialize task-specific classifier head
    fn initialize_classifier_head(&mut self) -> Result<()> {
        let input_dim = self.task_descriptor.input_dim;
        let output_dim = self.task_descriptor.output_dim;

        // Xavier initialization
        let bound = (6.0 / (input_dim + output_dim) as f32).sqrt();
        let classifier = Array2::random(
            (input_dim, output_dim),
            ndarray_rand::rand_distr::Uniform::new(-bound, bound),
        );

        self.classifier_head = Some(classifier);
        Ok(())
    }

    /// Adapt to task with given data
    pub fn adapt(&mut self, examples: &[(Array1<f32>, Array1<f32>)]) -> Result<()> {
        for step in 0..self.config.adaptation_steps {
            let mut total_loss = 0.0;
            let batch_size = self.config.batch_size.min(examples.len());

            // Sample batch
            let batch_indices = self.sample_batch_indices(examples.len(), batch_size);

            for &idx in &batch_indices {
                let (input, target) = &examples[idx];
                let loss = self.adaptation_step(input, target)?;
                total_loss += loss;
            }

            let avg_loss = total_loss / batch_indices.len() as f32;
            self.training_stats.record_step(step, avg_loss);
            self.current_step += 1;
        }

        Ok(())
    }

    /// Single adaptation step
    fn adaptation_step(&mut self, input: &Array1<f32>, target: &Array1<f32>) -> Result<f32> {
        // Forward pass
        let output = self.forward(input)?;

        // Compute loss
        let loss = self.compute_loss(&output, target)?;

        // Backward pass (simplified)
        self.backward_pass(input, target, &output)?;

        Ok(loss)
    }

    /// Forward pass through adaptation components
    pub fn forward(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let mut current_output = input.clone().insert_axis(Axis(0)); // Make 2D

        // Apply adapters if using adapter strategy
        if let AdaptationStrategy::Adapters = &self.config.strategy {
            for adapter in self.adapters.values() {
                current_output = adapter.forward(&current_output)?;
            }
        }

        // Apply classifier head
        if let Some(classifier) = &self.classifier_head {
            let output = current_output.dot(classifier);
            return Ok(output.index_axis(Axis(0), 0).to_owned());
        }

        Ok(current_output.index_axis(Axis(0), 0).to_owned())
    }

    /// Compute task-specific loss
    fn compute_loss(&self, prediction: &Array1<f32>, target: &Array1<f32>) -> Result<f32> {
        match self.task_descriptor.task_type {
            TaskType::Classification => {
                // Cross-entropy loss (simplified)
                let max_pred = prediction.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_pred: Array1<f32> = prediction.mapv(|x| (x - max_pred).exp());
                let sum_exp = exp_pred.sum();
                let log_probs = exp_pred.mapv(|x| (x / sum_exp).ln());

                // Negative log-likelihood (assuming target is one-hot)
                let loss = -(log_probs * target).sum();
                Ok(loss)
            },
            TaskType::Regression => {
                // Mean squared error
                let diff = prediction - target;
                let mse = (&diff * &diff).mean().unwrap();
                Ok(mse)
            },
            _ => {
                // Default to MSE for other types
                let diff = prediction - target;
                let mse = (&diff * &diff).mean().unwrap();
                Ok(mse)
            },
        }
    }

    /// Backward pass (simplified)
    fn backward_pass(
        &mut self,
        _input: &Array1<f32>,
        _target: &Array1<f32>,
        _output: &Array1<f32>,
    ) -> Result<()> {
        // In practice, would compute and apply gradients
        // This is a placeholder for the actual backward pass
        Ok(())
    }

    /// Sample batch indices
    fn sample_batch_indices(&self, total_size: usize, batch_size: usize) -> Vec<usize> {
        use rand::seq::SliceRandom;
        let mut indices: Vec<_> = (0..total_size).collect();
        indices.shuffle(&mut thread_rng().rng_mut());
        indices.into_iter().take(batch_size).collect()
    }

    /// Get task descriptor
    pub fn get_task_descriptor(&self) -> &TaskDescriptor {
        &self.task_descriptor
    }

    /// Get training statistics
    pub fn get_training_stats(&self) -> &AdaptationStats {
        &self.training_stats
    }

    /// Save adapter state
    pub fn save(&self, path: &str) -> Result<()> {
        let serialized = bincode::encode_to_vec(&(
            &self.task_descriptor,
            &self.config,
            &self.training_stats,
            self.current_step,
        ), bincode::config::standard())?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Load adapter state
    pub fn load(path: &str) -> Result<Self> {
        let data = std::fs::read(path)?;
        let ((task_descriptor, config, training_stats, current_step), _): ((
            TaskDescriptor,
            AdaptationConfig,
            AdaptationStats,
            usize,
        ), usize) = bincode::decode_from_slice(&data, bincode::config::standard())?;

        let mut adapter = Self {
            task_descriptor,
            config,
            adapters: HashMap::new(),
            classifier_head: None,
            training_stats,
            current_step,
        };

        adapter.initialize_components()?;
        Ok(adapter)
    }
}

/// Statistics for adaptation training
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct AdaptationStats {
    /// Loss per step
    pub losses: Vec<f32>,
    /// Best loss achieved
    pub best_loss: f32,
    /// Step when best loss was achieved
    pub best_step: usize,
    /// Training start time
    pub start_time: Option<std::time::SystemTime>,
    /// Total training time
    pub total_time: std::time::Duration,
}

impl AdaptationStats {
    fn new() -> Self {
        Self {
            losses: Vec::new(),
            best_loss: f32::INFINITY,
            best_step: 0,
            start_time: Some(std::time::SystemTime::now()),
            total_time: std::time::Duration::from_secs(0),
        }
    }

    fn record_step(&mut self, step: usize, loss: f32) {
        self.losses.push(loss);

        if loss < self.best_loss {
            self.best_loss = loss;
            self.best_step = step;
        }

        if let Some(start_time) = self.start_time {
            self.total_time =
                std::time::SystemTime::now().duration_since(start_time).unwrap_or_default();
        }
    }

    /// Get average loss over last N steps
    pub fn get_recent_average_loss(&self, n: usize) -> f32 {
        if self.losses.is_empty() {
            return f32::INFINITY;
        }

        let start_idx = self.losses.len().saturating_sub(n);
        let recent_losses = &self.losses[start_idx..];
        recent_losses.iter().sum::<f32>() / recent_losses.len() as f32
    }

    /// Check if training has converged
    pub fn has_converged(&self, patience: usize, tolerance: f32) -> bool {
        if self.losses.len() < patience {
            return false;
        }

        let recent_losses = &self.losses[self.losses.len() - patience..];
        let max_loss = recent_losses.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_loss = recent_losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        (max_loss - min_loss) < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_task_descriptor() {
        let desc = TaskDescriptor::classification("test_task".to_string(), 100, 10);
        assert_eq!(desc.task_type, TaskType::Classification);
        assert_eq!(desc.num_classes, Some(10));
        assert_eq!(desc.input_dim, 100);
        assert_eq!(desc.output_dim, 10);
    }

    #[test]
    fn test_adapter_layer() {
        let config = AdapterConfig::default();
        let adapter = AdapterLayer::new(128, config).unwrap();

        let input = Array2::ones((4, 128)); // batch_size=4, hidden_dim=128
        let output = adapter.forward(&input).unwrap();

        assert_eq!(output.shape(), &[4, 128]);
    }

    #[test]
    fn test_task_adapter_creation() {
        let desc = TaskDescriptor::classification("test".to_string(), 100, 5);
        let config = AdaptationConfig::default();
        let adapter = TaskAdapter::new(desc, config).unwrap();

        assert_eq!(adapter.task_descriptor.task_id, "test");
        assert_eq!(adapter.current_step, 0);
    }

    #[test]
    fn test_adaptation_stats() {
        let mut stats = AdaptationStats::new();

        stats.record_step(0, 1.5);
        stats.record_step(1, 1.2);
        stats.record_step(2, 1.0);

        assert_eq!(stats.best_loss, 1.0);
        assert_eq!(stats.best_step, 2);
        assert_abs_diff_eq!(stats.get_recent_average_loss(2), 1.1, epsilon = 1e-6);
    }

    #[test]
    fn test_forward_pass() {
        let desc = TaskDescriptor::classification("test".to_string(), 10, 3);
        let mut config = AdaptationConfig::default();
        config.strategy = AdaptationStrategy::LinearProbing;

        let adapter = TaskAdapter::new(desc, config).unwrap();
        let input = Array1::ones(10);
        let output = adapter.forward(&input).unwrap();

        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_convergence_detection() {
        let mut stats = AdaptationStats::new();

        // Add losses that don't converge
        for i in 0..10 {
            stats.record_step(i, (i as f32) * 0.1);
        }
        assert!(!stats.has_converged(5, 0.01));

        // Add losses that converge
        for i in 10..15 {
            stats.record_step(i, 1.0);
        }
        assert!(stats.has_converged(5, 0.01));
    }

    #[test]
    fn test_activation_functions() {
        let config = AdapterConfig {
            activation: ActivationFunction::ReLU,
            ..Default::default()
        };
        let adapter = AdapterLayer::new(10, config).unwrap();

        let input = Array2::from_shape_vec((1, 10), vec![-1.0; 10]).unwrap();
        let activated = adapter.apply_activation(&input);

        // ReLU should clip negative values to 0
        assert!(activated.iter().all(|&x| x >= 0.0));
    }
}
