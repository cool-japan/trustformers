use anyhow::Result;
use scirs2_core::ndarray::{s, Array2}; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Meta-learning algorithm trait
pub trait MetaLearningAlgorithm {
    fn meta_update(&mut self, task_batch: &TaskBatch) -> Result<MetaUpdateResult>;
    fn adapt(&self, support_set: &TaskData, adaptation_steps: usize) -> Result<ModelParameters>;
    fn evaluate(&self, params: &ModelParameters, query_set: &TaskData) -> Result<f32>;
}

/// Configuration for MAML (Model-Agnostic Meta-Learning)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MAMLConfig {
    /// Meta-learning rate (outer loop)
    pub meta_lr: f32,
    /// Task-specific learning rate (inner loop)
    pub inner_lr: f32,
    /// Number of gradient steps for adaptation
    pub adaptation_steps: usize,
    /// Number of tasks per meta-batch
    pub meta_batch_size: usize,
    /// Whether to use first-order approximation
    pub first_order: bool,
    /// Gradient clipping threshold
    pub grad_clip: f32,
    /// Whether to learn inner learning rates
    pub learn_inner_lrs: bool,
}

impl Default for MAMLConfig {
    fn default() -> Self {
        Self {
            meta_lr: 0.001,
            inner_lr: 0.01,
            adaptation_steps: 5,
            meta_batch_size: 16,
            first_order: false,
            grad_clip: 10.0,
            learn_inner_lrs: false,
        }
    }
}

/// Configuration for Reptile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReptileConfig {
    /// Meta-learning rate
    pub meta_lr: f32,
    /// Task-specific learning rate
    pub inner_lr: f32,
    /// Number of gradient steps for adaptation
    pub adaptation_steps: usize,
    /// Number of tasks per meta-batch
    pub meta_batch_size: usize,
    /// Gradient clipping threshold
    pub grad_clip: f32,
}

impl Default for ReptileConfig {
    fn default() -> Self {
        Self {
            meta_lr: 0.001,
            inner_lr: 0.01,
            adaptation_steps: 5,
            meta_batch_size: 16,
            grad_clip: 10.0,
        }
    }
}

/// Model parameters representation
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// Parameter tensors by layer name
    pub parameters: HashMap<String, Array2<f32>>,
    /// Parameter shapes
    pub shapes: HashMap<String, Vec<usize>>,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelParameters {
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            shapes: HashMap::new(),
        }
    }

    /// Add parameter tensor
    pub fn add_parameter(&mut self, name: String, tensor: Array2<f32>) {
        let shape = tensor.shape().to_vec();
        self.shapes.insert(name.clone(), shape);
        self.parameters.insert(name, tensor);
    }

    /// Get parameter by name
    pub fn get_parameter(&self, name: &str) -> Option<&Array2<f32>> {
        self.parameters.get(name)
    }

    /// Get mutable parameter by name
    pub fn get_parameter_mut(&mut self, name: &str) -> Option<&mut Array2<f32>> {
        self.parameters.get_mut(name)
    }

    /// Clone parameters
    pub fn clone_parameters(&self) -> Self {
        Self {
            parameters: self.parameters.clone(),
            shapes: self.shapes.clone(),
        }
    }

    /// Update parameters with gradients
    pub fn update_with_gradients(&mut self, gradients: &Self, learning_rate: f32) -> Result<()> {
        for (name, param) in &mut self.parameters {
            if let Some(grad) = gradients.get_parameter(name) {
                *param = param.clone() - learning_rate * grad;
            }
        }
        Ok(())
    }

    /// Compute parameter difference
    pub fn subtract(&self, other: &Self) -> Result<Self> {
        let mut result = Self::new();

        for (name, param) in &self.parameters {
            if let Some(other_param) = other.get_parameter(name) {
                let diff = param - other_param;
                result.add_parameter(name.clone(), diff);
            }
        }

        Ok(result)
    }

    /// Add parameters (element-wise)
    pub fn add(&self, other: &Self) -> Result<Self> {
        let mut result = Self::new();

        for (name, param) in &self.parameters {
            if let Some(other_param) = other.get_parameter(name) {
                let sum = param + other_param;
                result.add_parameter(name.clone(), sum);
            }
        }

        Ok(result)
    }

    /// Scale parameters by a scalar
    pub fn scale(&self, scalar: f32) -> Self {
        let mut result = Self::new();

        for (name, param) in &self.parameters {
            let scaled = param * scalar;
            result.add_parameter(name.clone(), scaled);
        }

        result
    }
}

/// Task data (support and query sets)
#[derive(Debug, Clone)]
pub struct TaskData {
    /// Input features
    pub inputs: Array2<f32>,
    /// Target outputs
    pub targets: Array2<f32>,
    /// Task identifier
    pub task_id: String,
}

impl TaskData {
    pub fn new(inputs: Array2<f32>, targets: Array2<f32>, task_id: String) -> Self {
        Self {
            inputs,
            targets,
            task_id,
        }
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.inputs.nrows()
    }

    /// Split into mini-batches
    pub fn split_batches(&self, batch_size: usize) -> Vec<TaskData> {
        let total_samples = self.batch_size();
        let mut batches = Vec::new();

        for start in (0..total_samples).step_by(batch_size) {
            let end = (start + batch_size).min(total_samples);
            let batch_inputs = self.inputs.slice(s![start..end, ..]).to_owned();
            let batch_targets = self.targets.slice(s![start..end, ..]).to_owned();

            batches.push(TaskData::new(
                batch_inputs,
                batch_targets,
                format!("{}_batch_{}", self.task_id, start / batch_size),
            ));
        }

        batches
    }
}

/// Batch of tasks for meta-learning
#[derive(Debug)]
pub struct TaskBatch {
    /// Support sets for each task
    pub support_sets: Vec<TaskData>,
    /// Query sets for each task
    pub query_sets: Vec<TaskData>,
}

impl TaskBatch {
    pub fn new(support_sets: Vec<TaskData>, query_sets: Vec<TaskData>) -> Result<Self> {
        if support_sets.len() != query_sets.len() {
            return Err(anyhow::anyhow!(
                "Support and query sets must have same length"
            ));
        }
        Ok(Self {
            support_sets,
            query_sets,
        })
    }

    /// Number of tasks in batch
    pub fn num_tasks(&self) -> usize {
        self.support_sets.len()
    }
}

/// Result of meta-update
#[derive(Debug)]
pub struct MetaUpdateResult {
    /// Meta-loss across all tasks
    pub meta_loss: f32,
    /// Per-task losses
    pub task_losses: Vec<f32>,
    /// Gradient norm
    pub grad_norm: f32,
    /// Updated parameters
    pub updated_parameters: ModelParameters,
}

/// MAML (Model-Agnostic Meta-Learning) implementation
pub struct MAMLTrainer {
    config: MAMLConfig,
    meta_parameters: Arc<RwLock<ModelParameters>>,
    #[allow(dead_code)]
    optimizer_state: HashMap<String, Array2<f32>>, // For Adam/RMSprop
    meta_step: usize,
}

impl MAMLTrainer {
    pub fn new(config: MAMLConfig, initial_parameters: ModelParameters) -> Self {
        Self {
            config,
            meta_parameters: Arc::new(RwLock::new(initial_parameters)),
            optimizer_state: HashMap::new(),
            meta_step: 0,
        }
    }

    /// Compute gradients for inner loop adaptation
    fn compute_inner_gradients(
        &self,
        parameters: &ModelParameters,
        task_data: &TaskData,
    ) -> Result<ModelParameters> {
        // Simplified gradient computation (in practice, would use automatic differentiation)
        let mut gradients = ModelParameters::new();

        for (name, param) in &parameters.parameters {
            // Compute loss and gradients (simplified)
            let grad = self.compute_parameter_gradient(param, task_data)?;
            gradients.add_parameter(name.clone(), grad);
        }

        Ok(gradients)
    }

    /// Compute gradients for a parameter using finite differences approximation
    fn compute_parameter_gradient(
        &self,
        param: &Array2<f32>,
        data: &TaskData,
    ) -> Result<Array2<f32>> {
        let eps = 1e-5f32;
        let mut gradients = Array2::zeros(param.raw_dim());
        let _original_loss = self.compute_loss_for_parameter(param, data)?;

        // Compute gradients using finite differences
        for ((i, j), param_val) in param.indexed_iter() {
            // Forward difference
            let mut param_plus = param.clone();
            param_plus[[i, j]] = param_val + eps;
            let loss_plus = self.compute_loss_for_parameter(&param_plus, data)?;

            // Backward difference
            let mut param_minus = param.clone();
            param_minus[[i, j]] = param_val - eps;
            let loss_minus = self.compute_loss_for_parameter(&param_minus, data)?;

            // Central difference for better accuracy
            gradients[[i, j]] = (loss_plus - loss_minus) / (2.0 * eps);
        }

        Ok(gradients)
    }

    /// Compute loss for a single parameter (helper for gradient computation)
    fn compute_loss_for_parameter(&self, param: &Array2<f32>, data: &TaskData) -> Result<f32> {
        // Simplified neural network forward pass
        // For this example, we'll assume a simple linear model: y = Wx + b
        let predictions = if param.ncols() == data.inputs.ncols() {
            // Weight matrix
            data.inputs.dot(param)
        } else if param.shape() == [1, data.targets.ncols()] {
            // Bias vector - broadcast across batch

            Array2::from_shape_fn((data.inputs.nrows(), param.ncols()), |(_, j)| param[[0, j]])
        } else {
            // Default: treat as identity-scaled inputs
            data.inputs.clone()
        };

        // Mean squared error loss
        let diff = &predictions - &data.targets;
        let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);

        Ok(mse)
    }

    /// Perform inner loop adaptation
    fn inner_loop_adaptation(
        &self,
        initial_params: &ModelParameters,
        support_set: &TaskData,
    ) -> Result<ModelParameters> {
        let mut adapted_params = initial_params.clone_parameters();

        for _step in 0..self.config.adaptation_steps {
            let gradients = self.compute_inner_gradients(&adapted_params, support_set)?;
            let lr = if self.config.learn_inner_lrs {
                // In practice, would have learned inner LRs per parameter
                self.config.inner_lr
            } else {
                self.config.inner_lr
            };

            adapted_params.update_with_gradients(&gradients, lr)?;
        }

        Ok(adapted_params)
    }

    /// Compute meta-gradients
    fn compute_meta_gradients(&self, task_batch: &TaskBatch) -> Result<(ModelParameters, f32)> {
        let meta_params = self.meta_parameters.read().expect("lock should not be poisoned");
        let mut meta_gradients = ModelParameters::new();
        let mut total_meta_loss = 0.0;

        // Initialize meta-gradients to zero
        for (name, param) in &meta_params.parameters {
            meta_gradients.add_parameter(name.clone(), Array2::zeros(param.raw_dim()));
        }

        // Accumulate gradients across tasks
        for (support_set, query_set) in task_batch.support_sets.iter().zip(&task_batch.query_sets) {
            // Inner loop adaptation
            let adapted_params = self.inner_loop_adaptation(&meta_params, support_set)?;

            // Compute query loss and gradients
            let query_loss = self.compute_query_loss(&adapted_params, query_set)?;
            total_meta_loss += query_loss;

            // Compute gradients w.r.t. meta-parameters
            let task_meta_grads = if self.config.first_order {
                // First-order approximation (Reptile-like)
                meta_params.subtract(&adapted_params)?
            } else {
                // Full second-order gradients (expensive)
                self.compute_second_order_gradients(&meta_params, &adapted_params, query_set)?
            };

            // Accumulate meta-gradients
            for (name, grad) in &task_meta_grads.parameters {
                if let Some(meta_grad) = meta_gradients.get_parameter_mut(name) {
                    *meta_grad = meta_grad.clone() + grad;
                }
            }
        }

        // Average gradients
        let num_tasks = task_batch.num_tasks() as f32;
        for grad in meta_gradients.parameters.values_mut() {
            *grad = grad.clone() / num_tasks;
        }

        total_meta_loss /= num_tasks;

        Ok((meta_gradients, total_meta_loss))
    }

    /// Compute query loss using current model parameters
    fn compute_query_loss(&self, params: &ModelParameters, query_set: &TaskData) -> Result<f32> {
        // Perform forward pass through the network
        let predictions = self.forward_pass(params, &query_set.inputs)?;

        // Compute loss based on task type
        let loss = if query_set.targets.ncols() == 1 {
            // Regression task - use MSE
            self.compute_mse_loss(&predictions, &query_set.targets)?
        } else {
            // Classification task - use cross-entropy
            self.compute_cross_entropy_loss(&predictions, &query_set.targets)?
        };

        Ok(loss)
    }

    /// Forward pass through a simple neural network
    fn forward_pass(&self, params: &ModelParameters, inputs: &Array2<f32>) -> Result<Array2<f32>> {
        let mut activations = inputs.clone();

        // Apply layers in sequence
        if let Some(layer1_weights) = params.get_parameter("layer1_weight") {
            activations = activations.dot(layer1_weights);

            // Add bias if present
            if let Some(layer1_bias) = params.get_parameter("layer1_bias") {
                for mut row in activations.rows_mut() {
                    for (i, &bias) in layer1_bias.row(0).iter().enumerate() {
                        if i < row.len() {
                            row[i] += bias;
                        }
                    }
                }
            }

            // Apply ReLU activation
            activations.mapv_inplace(|x| x.max(0.0));
        }

        // Output layer
        if let Some(output_weights) = params.get_parameter("output_weight") {
            activations = activations.dot(output_weights);

            if let Some(output_bias) = params.get_parameter("output_bias") {
                for mut row in activations.rows_mut() {
                    for (i, &bias) in output_bias.row(0).iter().enumerate() {
                        if i < row.len() {
                            row[i] += bias;
                        }
                    }
                }
            }
        }

        Ok(activations)
    }

    /// Compute mean squared error loss
    fn compute_mse_loss(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        let diff = predictions - targets;
        let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);
        Ok(mse)
    }

    /// Compute cross-entropy loss
    fn compute_cross_entropy_loss(
        &self,
        predictions: &Array2<f32>,
        targets: &Array2<f32>,
    ) -> Result<f32> {
        let batch_size = predictions.nrows();
        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let pred_row = predictions.row(i);
            let target_row = targets.row(i);

            // Apply softmax to predictions
            let max_pred = pred_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_preds: Vec<f32> = pred_row.iter().map(|&x| (x - max_pred).exp()).collect();
            let sum_exp: f32 = exp_preds.iter().sum();
            let softmax_preds: Vec<f32> = exp_preds.iter().map(|&x| x / sum_exp).collect();

            // Compute cross-entropy
            let mut row_loss = 0.0;
            for (&pred, &target) in softmax_preds.iter().zip(target_row.iter()) {
                if target > 0.0 {
                    row_loss -= target * pred.max(1e-15).ln();
                }
            }
            total_loss += row_loss;
        }

        Ok(total_loss / batch_size as f32)
    }

    /// Compute second-order gradients using finite differences of gradients
    fn compute_second_order_gradients(
        &self,
        meta_params: &ModelParameters,
        adapted_params: &ModelParameters,
        query_set: &TaskData,
    ) -> Result<ModelParameters> {
        let eps = 1e-4f32;
        let mut second_order_grads = ModelParameters::new();

        // For each parameter in meta_params, compute second-order gradients
        for (param_name, meta_param) in &meta_params.parameters {
            if adapted_params.get_parameter(param_name).is_none() {
                continue;
            }

            let mut param_grad = Array2::zeros(meta_param.raw_dim());

            // Compute Hessian-vector product using finite differences
            for ((i, j), _) in meta_param.indexed_iter() {
                // Perturb meta-parameter
                let mut meta_plus = meta_params.clone_parameters();
                let mut meta_minus = meta_params.clone_parameters();

                if let (Some(param_plus), Some(param_minus)) = (
                    meta_plus.get_parameter_mut(param_name),
                    meta_minus.get_parameter_mut(param_name),
                ) {
                    param_plus[[i, j]] += eps;
                    param_minus[[i, j]] -= eps;

                    // Compute gradients at perturbed points
                    let grad_plus =
                        self.compute_meta_gradient_at_point(&meta_plus, adapted_params, query_set)?;
                    let grad_minus = self.compute_meta_gradient_at_point(
                        &meta_minus,
                        adapted_params,
                        query_set,
                    )?;

                    // Second-order gradient via finite difference
                    if let (Some(g_plus), Some(g_minus)) = (
                        grad_plus.get_parameter(param_name),
                        grad_minus.get_parameter(param_name),
                    ) {
                        param_grad[[i, j]] = (g_plus[[i, j]] - g_minus[[i, j]]) / (2.0 * eps);
                    }
                }
            }

            second_order_grads.add_parameter(param_name.clone(), param_grad);
        }

        Ok(second_order_grads)
    }

    /// Compute meta-gradient at a specific parameter point
    fn compute_meta_gradient_at_point(
        &self,
        meta_params: &ModelParameters,
        adapted_params: &ModelParameters,
        query_set: &TaskData,
    ) -> Result<ModelParameters> {
        // Compute gradient of query loss w.r.t. adapted parameters
        let query_loss_grad = self.compute_query_loss_gradients(adapted_params, query_set)?;

        // Chain rule: grad w.r.t. meta_params = grad w.r.t. adapted_params * jacobian
        let jacobian = self.compute_adaptation_jacobian(meta_params, adapted_params)?;

        // Apply chain rule
        let mut meta_grad = ModelParameters::new();
        for (param_name, loss_grad) in &query_loss_grad.parameters {
            if let Some(jac) = jacobian.get_parameter(param_name) {
                // Simplified: element-wise multiplication (in practice would be matrix multiplication)
                let meta_gradient = loss_grad * jac;
                meta_grad.add_parameter(param_name.clone(), meta_gradient);
            }
        }

        Ok(meta_grad)
    }

    /// Compute gradients of query loss w.r.t. adapted parameters
    fn compute_query_loss_gradients(
        &self,
        params: &ModelParameters,
        query_set: &TaskData,
    ) -> Result<ModelParameters> {
        let mut gradients = ModelParameters::new();

        for (param_name, param) in &params.parameters {
            let grad = self.compute_parameter_gradient_for_query(param, query_set)?;
            gradients.add_parameter(param_name.clone(), grad);
        }

        Ok(gradients)
    }

    /// Compute parameter gradient for query loss
    fn compute_parameter_gradient_for_query(
        &self,
        param: &Array2<f32>,
        query_set: &TaskData,
    ) -> Result<Array2<f32>> {
        let eps = 1e-5f32;
        let mut gradients = Array2::zeros(param.raw_dim());

        for ((i, j), param_val) in param.indexed_iter() {
            // Forward difference
            let mut param_plus = param.clone();
            param_plus[[i, j]] = param_val + eps;

            let mut param_minus = param.clone();
            param_minus[[i, j]] = param_val - eps;

            // Create temporary parameter sets for loss computation
            let mut params_plus = ModelParameters::new();
            let mut params_minus = ModelParameters::new();
            params_plus.add_parameter("temp_param".to_string(), param_plus);
            params_minus.add_parameter("temp_param".to_string(), param_minus);

            let loss_plus = self.compute_query_loss(&params_plus, query_set)?;
            let loss_minus = self.compute_query_loss(&params_minus, query_set)?;

            gradients[[i, j]] = (loss_plus - loss_minus) / (2.0 * eps);
        }

        Ok(gradients)
    }

    /// Compute Jacobian of adaptation process (simplified)
    fn compute_adaptation_jacobian(
        &self,
        meta_params: &ModelParameters,
        adapted_params: &ModelParameters,
    ) -> Result<ModelParameters> {
        let mut jacobian = ModelParameters::new();

        // Simplified: assume identity Jacobian for first-order approximation
        // In practice, this would compute d(adapted_params)/d(meta_params)
        for (param_name, meta_param) in &meta_params.parameters {
            if adapted_params.get_parameter(param_name).is_some() {
                // Identity Jacobian (simplified)
                let identity_jac =
                    Array2::eye(meta_param.len()).into_shape_with_order(meta_param.raw_dim())?;
                jacobian.add_parameter(param_name.clone(), identity_jac);
            }
        }

        Ok(jacobian)
    }

    /// Clip gradients
    fn clip_gradients(&self, gradients: &mut ModelParameters) -> f32 {
        let mut total_norm = 0.0;

        // Compute total gradient norm
        for grad in gradients.parameters.values() {
            total_norm += grad.mapv(|x| x * x).sum();
        }
        total_norm = total_norm.sqrt();

        // Clip if necessary
        if total_norm > self.config.grad_clip {
            let clip_coef = self.config.grad_clip / total_norm;
            for grad in gradients.parameters.values_mut() {
                *grad = grad.clone() * clip_coef;
            }
        }

        total_norm
    }
}

impl MetaLearningAlgorithm for MAMLTrainer {
    fn meta_update(&mut self, task_batch: &TaskBatch) -> Result<MetaUpdateResult> {
        let (mut meta_gradients, meta_loss) = self.compute_meta_gradients(task_batch)?;
        let grad_norm = self.clip_gradients(&mut meta_gradients);

        // Update meta-parameters
        {
            let mut meta_params =
                self.meta_parameters.write().expect("lock should not be poisoned");
            meta_params.update_with_gradients(&meta_gradients, self.config.meta_lr)?;
        }

        self.meta_step += 1;

        Ok(MetaUpdateResult {
            meta_loss,
            task_losses: vec![meta_loss; task_batch.num_tasks()], // Simplified
            grad_norm,
            updated_parameters: self
                .meta_parameters
                .read()
                .expect("lock should not be poisoned")
                .clone_parameters(),
        })
    }

    fn adapt(&self, support_set: &TaskData, adaptation_steps: usize) -> Result<ModelParameters> {
        let meta_params = self.meta_parameters.read().expect("lock should not be poisoned");
        let mut adapted_params = meta_params.clone_parameters();

        for _ in 0..adaptation_steps {
            let gradients = self.compute_inner_gradients(&adapted_params, support_set)?;
            adapted_params.update_with_gradients(&gradients, self.config.inner_lr)?;
        }

        Ok(adapted_params)
    }

    fn evaluate(&self, params: &ModelParameters, query_set: &TaskData) -> Result<f32> {
        self.compute_query_loss(params, query_set)
    }
}

/// Reptile algorithm implementation
pub struct ReptileTrainer {
    config: ReptileConfig,
    meta_parameters: Arc<RwLock<ModelParameters>>,
    meta_step: usize,
}

impl ReptileTrainer {
    pub fn new(config: ReptileConfig, initial_parameters: ModelParameters) -> Self {
        Self {
            config,
            meta_parameters: Arc::new(RwLock::new(initial_parameters)),
            meta_step: 0,
        }
    }

    /// Perform SGD on a single task
    fn sgd_on_task(&self, task_data: &TaskData) -> Result<ModelParameters> {
        let meta_params = self.meta_parameters.read().expect("lock should not be poisoned");
        let mut task_params = meta_params.clone_parameters();

        for _ in 0..self.config.adaptation_steps {
            let gradients = self.compute_task_gradients(&task_params, task_data)?;
            task_params.update_with_gradients(&gradients, self.config.inner_lr)?;
        }

        Ok(task_params)
    }

    /// Compute gradients for task using proper gradient computation
    fn compute_task_gradients(
        &self,
        params: &ModelParameters,
        data: &TaskData,
    ) -> Result<ModelParameters> {
        let mut gradients = ModelParameters::new();

        // Compute gradients for each parameter
        for (param_name, param) in &params.parameters {
            let grad = self.compute_task_parameter_gradient(param, data, param_name)?;
            gradients.add_parameter(param_name.clone(), grad);
        }

        Ok(gradients)
    }

    /// Compute gradient for a specific parameter using finite differences
    fn compute_task_parameter_gradient(
        &self,
        param: &Array2<f32>,
        data: &TaskData,
        param_name: &str,
    ) -> Result<Array2<f32>> {
        let eps = 1e-5f32;
        let mut gradients = Array2::zeros(param.raw_dim());

        for ((i, j), param_val) in param.indexed_iter() {
            // Create perturbed parameters
            let mut param_plus = param.clone();
            let mut param_minus = param.clone();
            param_plus[[i, j]] = param_val + eps;
            param_minus[[i, j]] = param_val - eps;

            // Compute loss at perturbed points
            let loss_plus = self.compute_task_loss_for_param(&param_plus, data, param_name)?;
            let loss_minus = self.compute_task_loss_for_param(&param_minus, data, param_name)?;

            // Gradient via central difference
            gradients[[i, j]] = (loss_plus - loss_minus) / (2.0 * eps);
        }

        Ok(gradients)
    }

    /// Compute task loss for a specific parameter
    fn compute_task_loss_for_param(
        &self,
        param: &Array2<f32>,
        data: &TaskData,
        param_name: &str,
    ) -> Result<f32> {
        // Create a temporary parameter set with this parameter
        let mut temp_params = ModelParameters::new();
        temp_params.add_parameter(param_name.to_string(), param.clone());

        // Perform forward pass
        let predictions = if param_name.contains("weight") {
            // Weight matrix - matrix multiplication
            if param.ncols() == data.inputs.ncols() {
                data.inputs.dot(param)
            } else if param.nrows() == data.inputs.ncols() {
                data.inputs.dot(&param.t())
            } else {
                // Default behavior for mismatched dimensions
                data.inputs.clone()
            }
        } else if param_name.contains("bias") {
            // Bias vector - broadcast addition
            let mut result = data.inputs.clone();
            for mut row in result.rows_mut() {
                for (k, &bias) in param.row(0).iter().enumerate() {
                    if k < row.len() {
                        row[k] += bias;
                    }
                }
            }
            result
        } else {
            // Default: identity operation
            data.inputs.clone()
        };

        // Compute loss (MSE for simplicity)
        let diff = &predictions - &data.targets;
        let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);

        Ok(mse)
    }
}

impl MetaLearningAlgorithm for ReptileTrainer {
    fn meta_update(&mut self, task_batch: &TaskBatch) -> Result<MetaUpdateResult> {
        let mut total_update = ModelParameters::new();
        let mut total_loss = 0.0;

        // Initialize update to zero
        {
            let meta_params = self.meta_parameters.read().expect("lock should not be poisoned");
            for (name, param) in &meta_params.parameters {
                total_update.add_parameter(name.clone(), Array2::zeros(param.raw_dim()));
            }
        }

        // Accumulate updates from all tasks
        for (support_set, query_set) in task_batch.support_sets.iter().zip(&task_batch.query_sets) {
            // Train on support set
            let task_params = self.sgd_on_task(support_set)?;

            // Compute update direction
            let meta_params = self.meta_parameters.read().expect("lock should not be poisoned");
            let update = task_params.subtract(&meta_params)?;

            // Accumulate update
            for (name, param_update) in &update.parameters {
                if let Some(total_param_update) = total_update.get_parameter_mut(name) {
                    *total_param_update = total_param_update.clone() + param_update;
                }
            }

            // Evaluate on query set
            let loss = self.evaluate(&task_params, query_set)?;
            total_loss += loss;
        }

        // Average updates
        let num_tasks = task_batch.num_tasks() as f32;
        for param_update in total_update.parameters.values_mut() {
            *param_update = param_update.clone() / num_tasks;
        }
        total_loss /= num_tasks;

        // Apply meta-update
        {
            let mut meta_params =
                self.meta_parameters.write().expect("lock should not be poisoned");
            let scaled_update = total_update.scale(self.config.meta_lr);
            *meta_params = meta_params.add(&scaled_update)?;
        }

        self.meta_step += 1;

        Ok(MetaUpdateResult {
            meta_loss: total_loss,
            task_losses: vec![total_loss; task_batch.num_tasks()], // Simplified
            grad_norm: 0.0,                                        // Not computed for Reptile
            updated_parameters: self
                .meta_parameters
                .read()
                .expect("lock should not be poisoned")
                .clone_parameters(),
        })
    }

    fn adapt(&self, support_set: &TaskData, adaptation_steps: usize) -> Result<ModelParameters> {
        let meta_params = self.meta_parameters.read().expect("lock should not be poisoned");
        let mut adapted_params = meta_params.clone_parameters();

        for _ in 0..adaptation_steps {
            let gradients = self.compute_task_gradients(&adapted_params, support_set)?;
            adapted_params.update_with_gradients(&gradients, self.config.inner_lr)?;
        }

        Ok(adapted_params)
    }

    fn evaluate(&self, params: &ModelParameters, query_set: &TaskData) -> Result<f32> {
        // Perform forward pass through the model
        let mut predictions = query_set.inputs.clone();

        // Apply model parameters in sequence
        for (param_name, param) in &params.parameters {
            if param_name.contains("weight") {
                // Apply weight matrix
                if param.ncols() == predictions.ncols() {
                    predictions = predictions.dot(param);
                } else if param.nrows() == predictions.ncols() {
                    predictions = predictions.dot(&param.t());
                }

                // Apply ReLU activation after weight layers
                if !param_name.contains("output") {
                    predictions.mapv_inplace(|x| x.max(0.0));
                }
            } else if param_name.contains("bias") {
                // Apply bias
                for mut row in predictions.rows_mut() {
                    for (k, &bias) in param.row(0).iter().enumerate() {
                        if k < row.len() {
                            row[k] += bias;
                        }
                    }
                }
            }
        }

        // Compute loss
        let diff = &predictions - &query_set.targets;
        let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);

        Ok(mse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_parameters() {
        let mut params = ModelParameters::new();
        let tensor = Array2::ones((2, 3));
        params.add_parameter("layer1".to_string(), tensor.clone());

        assert_eq!(params.get_parameter("layer1").unwrap(), &tensor);
        assert_eq!(params.shapes.get("layer1").unwrap(), &vec![2, 3]);
    }

    #[test]
    fn test_task_data() {
        let inputs = Array2::ones((10, 5));
        let targets = Array2::zeros((10, 2));
        let task_data = TaskData::new(inputs, targets, "test_task".to_string());

        assert_eq!(task_data.batch_size(), 10);
        assert_eq!(task_data.task_id, "test_task");
    }

    #[test]
    fn test_task_batch() {
        let support = vec![
            TaskData::new(
                Array2::ones((5, 3)),
                Array2::zeros((5, 1)),
                "task1".to_string(),
            ),
            TaskData::new(
                Array2::ones((5, 3)),
                Array2::zeros((5, 1)),
                "task2".to_string(),
            ),
        ];
        let query = vec![
            TaskData::new(
                Array2::ones((3, 3)),
                Array2::zeros((3, 1)),
                "task1".to_string(),
            ),
            TaskData::new(
                Array2::ones((3, 3)),
                Array2::zeros((3, 1)),
                "task2".to_string(),
            ),
        ];

        let batch = TaskBatch::new(support, query).unwrap();
        assert_eq!(batch.num_tasks(), 2);
    }

    #[test]
    fn test_maml_trainer_creation() {
        let config = MAMLConfig::default();
        let mut params = ModelParameters::new();
        params.add_parameter("test".to_string(), Array2::<f32>::ones((2, 2)));

        let trainer = MAMLTrainer::new(config, params);
        assert_eq!(trainer.meta_step, 0);
    }

    #[test]
    fn test_reptile_trainer_creation() {
        let config = ReptileConfig::default();
        let mut params = ModelParameters::new();
        params.add_parameter("test".to_string(), Array2::<f32>::ones((2, 2)));

        let trainer = ReptileTrainer::new(config, params);
        assert_eq!(trainer.meta_step, 0);
    }

    #[test]
    fn test_parameter_operations() {
        let mut params1 = ModelParameters::new();
        let mut params2 = ModelParameters::new();

        params1.add_parameter("layer1".to_string(), Array2::<f32>::ones((2, 2)));
        params2.add_parameter("layer1".to_string(), Array2::<f32>::ones((2, 2)) * 2.0);

        let diff = params2.subtract(&params1).unwrap();
        let sum = params1.add(&params2).unwrap();
        let scaled = params1.scale(2.0);

        assert_eq!(
            diff.get_parameter("layer1").unwrap(),
            &Array2::<f32>::ones((2, 2))
        );
        assert_eq!(
            sum.get_parameter("layer1").unwrap(),
            &(Array2::<f32>::ones((2, 2)) * 3.0)
        );
        assert_eq!(
            scaled.get_parameter("layer1").unwrap(),
            &(Array2::<f32>::ones((2, 2)) * 2.0)
        );
    }
}
