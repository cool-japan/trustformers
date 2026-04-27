#[cfg(test)]
mod tests {
    use crate::losses::{CrossEntropyLoss, Loss, MSELoss};
    use scirs2_core::ndarray::{ArrayD, IxDyn};
    use trustformers_core::Tensor;

    fn make_logits(data: Vec<f32>, shape: &[usize]) -> Tensor {
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).expect("Failed to create array");
        Tensor::F32(arr)
    }

    fn make_targets_i64(data: Vec<i64>) -> Tensor {
        let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).expect("Failed to create targets");
        Tensor::I64(arr)
    }

    fn make_targets_f32(data: Vec<f32>, shape: &[usize]) -> Tensor {
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).expect("Failed to create targets");
        Tensor::F32(arr)
    }

    // --- CrossEntropyLoss Tests ---

    #[test]
    fn test_cross_entropy_loss_default() {
        let loss = CrossEntropyLoss::default();
        assert!(loss.ignore_index.is_none());
        assert!((loss.label_smoothing - 0.0).abs() < 1e-7);
        assert_eq!(loss.reduction, "mean");
    }

    #[test]
    fn test_cross_entropy_loss_new() {
        let loss = CrossEntropyLoss::new();
        assert_eq!(loss.name(), "CrossEntropyLoss");
    }

    #[test]
    fn test_cross_entropy_loss_with_ignore_index() {
        let loss = CrossEntropyLoss::new().with_ignore_index(0);
        assert_eq!(loss.ignore_index, Some(0));
    }

    #[test]
    fn test_cross_entropy_loss_with_label_smoothing() {
        let loss = CrossEntropyLoss::new().with_label_smoothing(0.1);
        assert!((loss.label_smoothing - 0.1).abs() < 1e-7);
    }

    #[test]
    fn test_cross_entropy_loss_compute_basic() {
        let loss = CrossEntropyLoss::new();
        // 3 samples, 4 classes
        let logits = make_logits(
            vec![
                2.0, 1.0, 0.1, 0.3, // sample 0: class 0 is highest
                0.1, 2.5, 0.2, 0.1, // sample 1: class 1 is highest
                0.3, 0.1, 3.0, 0.2, // sample 2: class 2 is highest
            ],
            &[3, 4],
        );
        let targets = make_targets_i64(vec![0, 1, 2]);
        let result = loss.compute(&logits, &targets);
        assert!(result.is_ok());
        let val = result.expect("Compute failed");
        // Since predictions match targets, loss should be relatively low
        assert!(val >= 0.0);
        assert!(val < 2.0);
    }

    #[test]
    fn test_cross_entropy_loss_compute_wrong_predictions() {
        let loss = CrossEntropyLoss::new();
        // Predictions are wrong
        let logits = make_logits(
            vec![
                0.1, 2.0, 0.1, 0.3, // sample 0: class 1 predicted, target is 0
                2.0, 0.1, 0.2, 0.1, // sample 1: class 0 predicted, target is 1
            ],
            &[2, 4],
        );
        let targets = make_targets_i64(vec![0, 1]);
        let result = loss.compute(&logits, &targets);
        assert!(result.is_ok());
        let val = result.expect("Compute failed");
        // Wrong predictions => higher loss
        assert!(val > 0.5);
    }

    #[test]
    fn test_cross_entropy_loss_with_ignore_index_computation() {
        let loss = CrossEntropyLoss::new().with_ignore_index(0);
        let logits = make_logits(
            vec![
                2.0, 1.0, 0.1, // sample 0: ignored
                0.1, 2.5, 0.2, // sample 1: computed
            ],
            &[2, 3],
        );
        let targets = make_targets_i64(vec![0, 1]);
        let result = loss.compute(&logits, &targets);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cross_entropy_loss_with_label_smoothing_computation() {
        let loss = CrossEntropyLoss::new().with_label_smoothing(0.1);
        let logits = make_logits(
            vec![2.0, 0.5, 0.3, 0.1, 3.0, 0.2],
            &[2, 3],
        );
        let targets = make_targets_i64(vec![0, 1]);
        let result = loss.compute(&logits, &targets);
        assert!(result.is_ok());
        let val = result.expect("Compute failed");
        assert!(val >= 0.0);
    }

    #[test]
    fn test_cross_entropy_loss_compute_with_gradients() {
        let loss = CrossEntropyLoss::new();
        let logits = make_logits(
            vec![2.0, 1.0, 0.1, 0.3, 0.1, 2.5, 0.2, 0.1],
            &[2, 4],
        );
        let targets = make_targets_i64(vec![0, 1]);
        let result = loss.compute_with_gradients(&logits, &targets);
        assert!(result.is_ok());
        let (loss_val, grads) = result.expect("Failed to compute with gradients");
        assert!(loss_val >= 0.0);
        match grads {
            Tensor::F32(arr) => {
                assert_eq!(arr.shape(), &[2, 4]);
            }
            _ => panic!("Expected F32 gradients"),
        }
    }

    #[test]
    fn test_cross_entropy_loss_compute_with_gradients_label_smoothing() {
        let loss = CrossEntropyLoss::new().with_label_smoothing(0.1);
        let logits = make_logits(
            vec![2.0, 1.0, 0.1, 0.3, 0.1, 2.5, 0.2, 0.1],
            &[2, 4],
        );
        let targets = make_targets_i64(vec![0, 1]);
        let result = loss.compute_with_gradients(&logits, &targets);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cross_entropy_loss_wrong_tensor_types() {
        let loss = CrossEntropyLoss::new();
        let predictions = make_targets_f32(vec![1.0, 2.0], &[2]);
        let targets = make_targets_f32(vec![0.0, 1.0], &[2]);
        let result = loss.compute(&predictions, &targets);
        assert!(result.is_err());
    }

    // --- MSELoss Tests ---

    #[test]
    fn test_mse_loss_default() {
        let loss = MSELoss::default();
        assert_eq!(loss.reduction, "mean");
    }

    #[test]
    fn test_mse_loss_new() {
        let loss = MSELoss::new();
        assert_eq!(loss.name(), "MSELoss");
    }

    #[test]
    fn test_mse_loss_compute_zero() {
        let loss = MSELoss::new();
        let preds = make_targets_f32(vec![1.0, 2.0, 3.0], &[3]);
        let targets = make_targets_f32(vec![1.0, 2.0, 3.0], &[3]);
        let result = loss.compute(&preds, &targets);
        assert!(result.is_ok());
        let val = result.expect("Compute failed");
        assert!(val.abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_compute_nonzero() {
        let loss = MSELoss::new();
        let preds = make_targets_f32(vec![1.0, 2.0, 3.0], &[3]);
        let targets = make_targets_f32(vec![2.0, 3.0, 4.0], &[3]);
        let result = loss.compute(&preds, &targets);
        assert!(result.is_ok());
        let val = result.expect("Compute failed");
        // MSE = mean((1-2)^2 + (2-3)^2 + (3-4)^2) = mean(1+1+1) = 1.0
        assert!((val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mse_loss_compute_shape_mismatch() {
        let loss = MSELoss::new();
        let preds = make_targets_f32(vec![1.0, 2.0], &[2]);
        let targets = make_targets_f32(vec![1.0, 2.0, 3.0], &[3]);
        let result = loss.compute(&preds, &targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_mse_loss_compute_with_gradients() {
        let loss = MSELoss::new();
        let preds = make_targets_f32(vec![1.0, 2.0, 3.0], &[3]);
        let targets = make_targets_f32(vec![2.0, 3.0, 4.0], &[3]);
        let result = loss.compute_with_gradients(&preds, &targets);
        assert!(result.is_ok());
        let (loss_val, grads) = result.expect("Compute with gradients failed");
        assert!(loss_val > 0.0);
        match grads {
            Tensor::F32(arr) => {
                assert_eq!(arr.shape(), &[3]);
                // Gradient = 2*(pred - target)/n = 2*(-1)/3 ~ -0.667 for each
                for &g in arr.iter() {
                    assert!((g - (-2.0 / 3.0)).abs() < 0.01);
                }
            }
            _ => panic!("Expected F32 gradients"),
        }
    }

    #[test]
    fn test_mse_loss_wrong_tensor_types() {
        let loss = MSELoss::new();
        let preds = make_targets_i64(vec![1, 2, 3]);
        let targets = make_targets_f32(vec![1.0, 2.0, 3.0], &[3]);
        let result = loss.compute(&preds, &targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_mse_loss_2d_tensor() {
        let loss = MSELoss::new();
        let preds = make_targets_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let targets = make_targets_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = loss.compute(&preds, &targets);
        assert!(result.is_ok());
        let val = result.expect("Compute failed");
        assert!(val.abs() < 1e-6);
    }

    #[test]
    fn test_loss_trait_name() {
        let ce = CrossEntropyLoss::new();
        assert_eq!(Loss::name(&ce), "CrossEntropyLoss");
        let mse = MSELoss::new();
        assert_eq!(Loss::name(&mse), "MSELoss");
    }
}
