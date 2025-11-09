use scirs2_core::ndarray::{Array1, Array2, ArrayD}; // SciRS2 Integration Policy
use trustformers_core::errors::{compute_error, Result};
use trustformers_core::Tensor;

/// Trait for loss functions
pub trait Loss: Send + Sync {
    /// Compute the loss given predictions and targets
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f32>;

    /// Compute the loss and return gradients
    fn compute_with_gradients(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<(f32, Tensor)>;

    /// Get the name of the loss function
    fn name(&self) -> &'static str;
}

/// Cross-entropy loss for classification tasks
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss {
    /// Whether to ignore a particular class index
    pub ignore_index: Option<usize>,
    /// Label smoothing parameter
    pub label_smoothing: f32,
    /// Reduction method: "mean", "sum", or "none"
    pub reduction: String,
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self {
            ignore_index: None,
            label_smoothing: 0.0,
            reduction: "mean".to_string(),
        }
    }
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_ignore_index(mut self, ignore_index: usize) -> Self {
        self.ignore_index = Some(ignore_index);
        self
    }

    pub fn with_label_smoothing(mut self, label_smoothing: f32) -> Self {
        self.label_smoothing = label_smoothing;
        self
    }

    /// Compute softmax with numerical stability
    fn softmax(logits: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let max_vals = logits.fold_axis(
            ndarray::Axis(logits.ndim() - 1),
            f32::NEG_INFINITY,
            |&a, &b| a.max(b),
        );

        // Subtract max for numerical stability
        let stable_logits = logits - &max_vals.insert_axis(ndarray::Axis(logits.ndim() - 1));

        // Compute exp
        let exp_logits = stable_logits.mapv(|x| x.exp());

        // Sum along last axis
        let sum_exp = exp_logits.sum_axis(ndarray::Axis(logits.ndim() - 1));

        // Divide by sum
        let probs = exp_logits / sum_exp.insert_axis(ndarray::Axis(logits.ndim() - 1));

        Ok(probs)
    }

    /// Apply label smoothing to targets
    fn smooth_labels(&self, targets: &Array1<usize>, num_classes: usize) -> Result<Array2<f32>> {
        let batch_size = targets.len();
        let mut smoothed = Array2::zeros((batch_size, num_classes));

        let smooth_value = self.label_smoothing / (num_classes as f32 - 1.0);
        let true_value = 1.0 - self.label_smoothing;

        for (i, &target) in targets.iter().enumerate() {
            // Fill with smoothing value
            for j in 0..num_classes {
                smoothed[[i, j]] = smooth_value;
            }
            // Set true class
            smoothed[[i, target]] = true_value;
        }

        Ok(smoothed)
    }
}

impl Loss for CrossEntropyLoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f32> {
        match (predictions, targets) {
            (Tensor::F32(pred_logits), Tensor::I64(target_labels)) => {
                // Convert targets to usize
                let targets_usize: Vec<usize> = target_labels.iter().map(|&x| x as usize).collect();
                let targets_arr = Array1::from_vec(targets_usize);

                // Compute softmax probabilities
                let probs = Self::softmax(pred_logits)?;

                let batch_size = targets_arr.len();
                let num_classes = pred_logits.shape()[pred_logits.ndim() - 1];

                let mut total_loss = 0.0;
                let mut valid_samples = 0;

                if self.label_smoothing > 0.0 {
                    // Use label smoothing
                    let smooth_targets = self.smooth_labels(&targets_arr, num_classes)?;

                    for i in 0..batch_size {
                        if let Some(ignore_idx) = self.ignore_index {
                            if targets_arr[i] == ignore_idx {
                                continue;
                            }
                        }

                        let mut sample_loss = 0.0;
                        for j in 0..num_classes {
                            let prob = probs[[i, j]].max(1e-8); // Avoid log(0)
                            sample_loss -= smooth_targets[[i, j]] * prob.ln();
                        }

                        total_loss += sample_loss;
                        valid_samples += 1;
                    }
                } else {
                    // Standard cross-entropy
                    for i in 0..batch_size {
                        let target_class = targets_arr[i];

                        if let Some(ignore_idx) = self.ignore_index {
                            if target_class == ignore_idx {
                                continue;
                            }
                        }

                        let prob = probs[[i, target_class]].max(1e-8); // Avoid log(0)
                        total_loss -= prob.ln();
                        valid_samples += 1;
                    }
                }

                match self.reduction.as_str() {
                    "mean" => {
                        Ok(if valid_samples > 0 { total_loss / valid_samples as f32 } else { 0.0 })
                    },
                    "sum" => Ok(total_loss),
                    "none" => Ok(total_loss), // Would need to return per-sample losses
                    _ => Err(compute_error(
                        "loss_computation",
                        format!("Unknown reduction: {}", self.reduction),
                    )),
                }
            },
            _ => Err(compute_error(
                "cross_entropy_loss",
                "CrossEntropyLoss expects F32 predictions and I64 targets",
            )),
        }
    }

    fn compute_with_gradients(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<(f32, Tensor)> {
        let loss = self.compute(predictions, targets)?;

        match (predictions, targets) {
            (Tensor::F32(pred_logits), Tensor::I64(target_labels)) => {
                let targets_usize: Vec<usize> = target_labels.iter().map(|&x| x as usize).collect();
                let targets_arr = Array1::from_vec(targets_usize);

                // Compute softmax probabilities
                let probs = Self::softmax(pred_logits)?;
                let mut gradients = probs.clone();

                let batch_size = targets_arr.len();
                let num_classes = pred_logits.shape()[pred_logits.ndim() - 1];

                if self.label_smoothing > 0.0 {
                    // Label smoothing gradients
                    let smooth_targets = self.smooth_labels(&targets_arr, num_classes)?;

                    for i in 0..batch_size {
                        if let Some(ignore_idx) = self.ignore_index {
                            if targets_arr[i] == ignore_idx {
                                // Zero out gradients for ignored samples
                                for j in 0..num_classes {
                                    gradients[[i, j]] = 0.0;
                                }
                                continue;
                            }
                        }

                        for j in 0..num_classes {
                            gradients[[i, j]] -= smooth_targets[[i, j]];
                        }
                    }
                } else {
                    // Standard cross-entropy gradients: p - y (where y is one-hot)
                    for i in 0..batch_size {
                        let target_class = targets_arr[i];

                        if let Some(ignore_idx) = self.ignore_index {
                            if target_class == ignore_idx {
                                // Zero out gradients for ignored samples
                                for j in 0..num_classes {
                                    gradients[[i, j]] = 0.0;
                                }
                                continue;
                            }
                        }

                        gradients[[i, target_class]] -= 1.0;
                    }
                }

                // Apply reduction to gradients
                if self.reduction == "mean" {
                    let valid_samples = if let Some(ignore_idx) = self.ignore_index {
                        targets_arr.iter().filter(|&&x| x != ignore_idx).count()
                    } else {
                        batch_size
                    };

                    if valid_samples > 0 {
                        gradients /= valid_samples as f32;
                    }
                }

                Ok((loss, Tensor::F32(gradients)))
            },
            _ => Err(compute_error(
                "cross_entropy_loss",
                "CrossEntropyLoss expects F32 predictions and I64 targets",
            )),
        }
    }

    fn name(&self) -> &'static str {
        "CrossEntropyLoss"
    }
}

/// Mean Squared Error loss for regression tasks
#[derive(Debug, Clone)]
pub struct MSELoss {
    /// Reduction method: "mean", "sum", or "none"
    pub reduction: String,
}

impl Default for MSELoss {
    fn default() -> Self {
        Self {
            reduction: "mean".to_string(),
        }
    }
}

impl MSELoss {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Loss for MSELoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f32> {
        match (predictions, targets) {
            (Tensor::F32(pred), Tensor::F32(target)) => {
                if pred.shape() != target.shape() {
                    return Err(compute_error(
                        "mse_loss",
                        "Predictions and targets must have the same shape",
                    ));
                }

                let diff = pred - target;
                let squared = diff.mapv(|x| x * x);

                let total_loss = squared.sum();

                match self.reduction.as_str() {
                    "mean" => Ok(total_loss / pred.len() as f32),
                    "sum" => Ok(total_loss),
                    "none" => Ok(total_loss), // Would need to return per-sample losses
                    _ => Err(compute_error(
                        "loss_computation",
                        format!("Unknown reduction: {}", self.reduction),
                    )),
                }
            },
            _ => Err(compute_error(
                "mse_loss",
                "MSELoss expects F32 predictions and F32 targets",
            )),
        }
    }

    fn compute_with_gradients(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<(f32, Tensor)> {
        let loss = self.compute(predictions, targets)?;

        match (predictions, targets) {
            (Tensor::F32(pred), Tensor::F32(target)) => {
                // Gradient of MSE: 2 * (pred - target)
                let mut gradients = 2.0 * (pred - target);

                // Apply reduction to gradients
                if self.reduction == "mean" {
                    gradients /= pred.len() as f32;
                }

                Ok((loss, Tensor::F32(gradients)))
            },
            _ => Err(compute_error(
                "mse_loss",
                "MSELoss expects F32 predictions and F32 targets",
            )),
        }
    }

    fn name(&self) -> &'static str {
        "MSELoss"
    }
}
