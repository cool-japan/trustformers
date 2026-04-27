use scirs2_core::ndarray::{Array1, Array2, ArrayD, Axis}; // SciRS2 Integration Policy
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
        let max_vals = logits.fold_axis(Axis(logits.ndim() - 1), f32::NEG_INFINITY, |&a, &b| {
            a.max(b)
        });

        // Subtract max for numerical stability
        let stable_logits = logits - &max_vals.insert_axis(Axis(logits.ndim() - 1));

        // Compute exp
        let exp_logits = stable_logits.mapv(|x| x.exp());

        // Sum along last axis
        let sum_exp = exp_logits.sum_axis(Axis(logits.ndim() - 1));

        // Divide by sum
        let probs = exp_logits / sum_exp.insert_axis(Axis(logits.ndim() - 1));

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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{ArrayD, IxDyn};
    use trustformers_core::Tensor;

    // Simple LCG for deterministic values without rand crate
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }

        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    fn make_f32_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
        let arr =
            ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape mismatch in test helper");
        Tensor::F32(arr)
    }

    fn make_i64_tensor(data: Vec<i64>, shape: &[usize]) -> Tensor {
        let arr =
            ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape mismatch in test helper");
        Tensor::I64(arr)
    }

    // ──────────────────── CrossEntropyLoss ────────────────────

    #[test]
    fn test_cross_entropy_name() {
        let loss = CrossEntropyLoss::new();
        assert_eq!(loss.name(), "CrossEntropyLoss");
    }

    #[test]
    fn test_cross_entropy_default_fields() {
        let loss = CrossEntropyLoss::default();
        assert!(loss.ignore_index.is_none());
        assert_eq!(loss.label_smoothing, 0.0);
        assert_eq!(loss.reduction, "mean");
    }

    #[test]
    fn test_cross_entropy_with_ignore_index() {
        let loss = CrossEntropyLoss::new().with_ignore_index(255);
        assert_eq!(loss.ignore_index, Some(255));
    }

    #[test]
    fn test_cross_entropy_with_label_smoothing() {
        let loss = CrossEntropyLoss::new().with_label_smoothing(0.1);
        assert!((loss.label_smoothing - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        // Perfect prediction: logits heavily favour correct class
        // batch=2, num_classes=3
        let logits = make_f32_tensor(vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0], &[2, 3]);
        let targets = make_i64_tensor(vec![0, 1], &[2]);
        let loss = CrossEntropyLoss::new();
        let result = loss.compute(&logits, &targets);
        assert!(result.is_ok(), "compute should succeed");
        let val = result.expect("loss value must be present");
        assert!(
            val < 0.01,
            "loss for perfect prediction should be near 0, got {}",
            val
        );
    }

    #[test]
    fn test_cross_entropy_uniform_prediction() {
        // Uniform logits → loss ≈ ln(num_classes)
        let logits = make_f32_tensor(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]);
        let targets = make_i64_tensor(vec![0, 1], &[2]);
        let loss = CrossEntropyLoss::new();
        let result = loss.compute(&logits, &targets);
        assert!(result.is_ok());
        let val = result.expect("loss value must be present");
        let expected = (3.0_f32).ln();
        assert!(
            (val - expected).abs() < 0.01,
            "uniform loss expected ~{:.3}, got {:.3}",
            expected,
            val
        );
    }

    #[test]
    fn test_cross_entropy_with_gradients_returns_same_loss() {
        let logits = make_f32_tensor(vec![2.0, 0.5, 0.5, 0.5, 2.0, 0.5], &[2, 3]);
        let targets = make_i64_tensor(vec![0, 1], &[2]);
        let loss = CrossEntropyLoss::new();
        let loss_only = loss.compute(&logits, &targets).expect("compute failed");
        let (loss_with_grad, _grad) = loss
            .compute_with_gradients(&logits, &targets)
            .expect("compute_with_gradients failed");
        assert!(
            (loss_only - loss_with_grad).abs() < 1e-5,
            "losses should match"
        );
    }

    #[test]
    fn test_cross_entropy_gradient_shape_matches_predictions() {
        let logits = make_f32_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let targets = make_i64_tensor(vec![2, 0], &[2]);
        let loss = CrossEntropyLoss::new();
        let (_, grad) = loss
            .compute_with_gradients(&logits, &targets)
            .expect("compute_with_gradients failed");
        match grad {
            Tensor::F32(arr) => {
                assert_eq!(arr.shape(), &[2, 3], "gradient shape must match logits");
            },
            _ => panic!("expected F32 gradient tensor"),
        }
    }

    #[test]
    fn test_cross_entropy_ignore_index_skips_sample() {
        // Only 1 valid sample out of 2; ignored sample has target == 99
        let logits = make_f32_tensor(vec![0.0, 10.0, 0.0, 0.0, 10.0, 0.0], &[2, 3]);
        // second target is ignored
        let targets = make_i64_tensor(vec![1, 99], &[2]);
        let loss_with_ignore = CrossEntropyLoss::new().with_ignore_index(99);
        let result = loss_with_ignore.compute(&logits, &targets);
        assert!(result.is_ok());
        let val = result.expect("loss value must be present");
        // With only the first sample (near-perfect), loss should be very small
        assert!(
            val < 0.1,
            "ignored sample should not contribute to loss, got {}",
            val
        );
    }

    #[test]
    fn test_cross_entropy_label_smoothing_increases_loss() {
        let logits = make_f32_tensor(vec![5.0, 0.0, 0.0, 5.0, 0.0, 0.0], &[2, 3]);
        let targets = make_i64_tensor(vec![0, 0], &[2]);
        let loss_no_smooth = CrossEntropyLoss::new();
        let loss_smooth = CrossEntropyLoss::new().with_label_smoothing(0.1);
        let val_no_smooth = loss_no_smooth.compute(&logits, &targets).expect("no-smooth failed");
        let val_smooth = loss_smooth.compute(&logits, &targets).expect("smooth failed");
        assert!(
            val_smooth > val_no_smooth,
            "label smoothing should increase loss for near-perfect predictions"
        );
    }

    #[test]
    fn test_cross_entropy_sum_reduction() {
        let logits = make_f32_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let targets = make_i64_tensor(vec![0, 0], &[2]);
        let loss_mean = CrossEntropyLoss::new();
        let loss_sum = CrossEntropyLoss {
            reduction: "sum".to_string(),
            ..CrossEntropyLoss::default()
        };
        let val_mean = loss_mean.compute(&logits, &targets).expect("mean failed");
        let val_sum = loss_sum.compute(&logits, &targets).expect("sum failed");
        // sum should be approximately 2 × mean for 2 equal samples
        assert!(
            (val_sum - val_mean * 2.0).abs() < 1e-4,
            "sum should be 2x mean for equal samples"
        );
    }

    #[test]
    fn test_cross_entropy_wrong_tensor_types_returns_err() {
        let pred = make_f32_tensor(vec![1.0, 0.0], &[2]);
        let target = make_f32_tensor(vec![0.0, 1.0], &[2]);
        let loss = CrossEntropyLoss::new();
        let result = loss.compute(&pred, &target);
        assert!(result.is_err(), "should fail for F32 targets");
    }

    #[test]
    fn test_cross_entropy_large_batch_stable() {
        let mut lcg = Lcg::new(42);
        let batch = 32;
        let classes = 10;
        let data: Vec<f32> = (0..batch * classes).map(|_| lcg.next_f32() * 4.0 - 2.0).collect();
        let logits = make_f32_tensor(data, &[batch, classes]);
        let labels: Vec<i64> = (0..batch).map(|i| (i % classes) as i64).collect();
        let targets = make_i64_tensor(labels, &[batch]);
        let loss = CrossEntropyLoss::new();
        let result = loss.compute(&logits, &targets);
        assert!(result.is_ok(), "large batch cross-entropy should succeed");
        let val = result.expect("loss value must be present");
        assert!(val.is_finite(), "loss must be finite");
        assert!(val > 0.0, "loss must be positive");
    }

    // ──────────────────── MSELoss ────────────────────

    #[test]
    fn test_mse_name() {
        let loss = MSELoss::new();
        assert_eq!(loss.name(), "MSELoss");
    }

    #[test]
    fn test_mse_default_reduction_is_mean() {
        let loss = MSELoss::default();
        assert_eq!(loss.reduction, "mean");
    }

    #[test]
    fn test_mse_zero_loss_identical_tensors() {
        let pred = make_f32_tensor(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let target = make_f32_tensor(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let loss = MSELoss::new();
        let result = loss.compute(&pred, &target);
        assert!(result.is_ok());
        let val = result.expect("loss value must be present");
        assert!(
            val.abs() < 1e-6,
            "identical tensors should have zero MSE, got {}",
            val
        );
    }

    #[test]
    fn test_mse_known_value() {
        // MSE([1,2], [3,4]) = ((1-3)^2 + (2-4)^2) / 2 = (4+4)/2 = 4.0
        let pred = make_f32_tensor(vec![1.0, 2.0], &[2]);
        let target = make_f32_tensor(vec![3.0, 4.0], &[2]);
        let loss = MSELoss::new();
        let val = loss.compute(&pred, &target).expect("MSE compute failed");
        assert!((val - 4.0).abs() < 1e-5, "expected 4.0, got {}", val);
    }

    #[test]
    fn test_mse_sum_reduction() {
        let pred = make_f32_tensor(vec![1.0, 2.0], &[2]);
        let target = make_f32_tensor(vec![3.0, 4.0], &[2]);
        let loss = MSELoss {
            reduction: "sum".to_string(),
        };
        let val = loss.compute(&pred, &target).expect("MSE sum failed");
        // sum = 4 + 4 = 8.0
        assert!(
            (val - 8.0).abs() < 1e-5,
            "expected 8.0 for sum reduction, got {}",
            val
        );
    }

    #[test]
    fn test_mse_shape_mismatch_returns_err() {
        let pred = make_f32_tensor(vec![1.0, 2.0, 3.0], &[3]);
        let target = make_f32_tensor(vec![1.0, 2.0], &[2]);
        let loss = MSELoss::new();
        let result = loss.compute(&pred, &target);
        assert!(result.is_err(), "shape mismatch should return an error");
    }

    #[test]
    fn test_mse_wrong_tensor_types_returns_err() {
        let pred = make_f32_tensor(vec![1.0, 2.0], &[2]);
        let target = make_i64_tensor(vec![1, 2], &[2]);
        let loss = MSELoss::new();
        let result = loss.compute(&pred, &target);
        assert!(result.is_err(), "MSE should fail for I64 target");
    }

    #[test]
    fn test_mse_gradient_matches_formula() {
        // grad = 2 * (pred - target) / n
        let pred = make_f32_tensor(vec![2.0, 4.0], &[2]);
        let target = make_f32_tensor(vec![0.0, 0.0], &[2]);
        let loss = MSELoss::new();
        let (_, grad) = loss.compute_with_gradients(&pred, &target).expect("mse grad failed");
        match grad {
            Tensor::F32(arr) => {
                // expected: [2*(2-0)/2, 2*(4-0)/2] = [2.0, 4.0]
                assert!(
                    (arr[IxDyn(&[0])] - 2.0).abs() < 1e-5,
                    "grad[0] expected 2.0"
                );
                assert!(
                    (arr[IxDyn(&[1])] - 4.0).abs() < 1e-5,
                    "grad[1] expected 4.0"
                );
            },
            _ => panic!("expected F32 gradient"),
        }
    }

    #[test]
    fn test_mse_gradient_shape_matches_input() {
        let pred = make_f32_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let target = make_f32_tensor(vec![0.0; 6], &[2, 3]);
        let loss = MSELoss::new();
        let (_, grad) = loss.compute_with_gradients(&pred, &target).expect("mse grad failed");
        match grad {
            Tensor::F32(arr) => {
                assert_eq!(arr.shape(), &[2, 3], "gradient shape must match input");
            },
            _ => panic!("expected F32 gradient"),
        }
    }

    #[test]
    fn test_mse_2d_batch() {
        let mut lcg = Lcg::new(17);
        let data_pred: Vec<f32> = (0..20).map(|_| lcg.next_f32()).collect();
        let data_tgt: Vec<f32> = (0..20).map(|_| lcg.next_f32()).collect();
        let pred = make_f32_tensor(data_pred.clone(), &[4, 5]);
        let target = make_f32_tensor(data_tgt.clone(), &[4, 5]);
        let loss = MSELoss::new();
        let result = loss.compute(&pred, &target);
        assert!(result.is_ok(), "2D batch MSE should succeed");
        let val = result.expect("loss value must be present");
        // Verify by hand: mean squared diff
        let manual: f32 = data_pred
            .iter()
            .zip(data_tgt.iter())
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>()
            / 20.0;
        assert!(
            (val - manual).abs() < 1e-4,
            "2D batch MSE mismatch: {} vs {}",
            val,
            manual
        );
    }
}
