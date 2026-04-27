// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for core traits and TokenizedInput.

#[cfg(test)]
mod tests {
    use crate::errors::Result;
    use crate::tensor::Tensor;
    use crate::traits::{Config, Layer, Optimizer, TokenizedInput, WeightReader};
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    // TokenizedInput tests

    #[test]
    fn test_tokenized_input_new() {
        let input = TokenizedInput::new(vec![101, 2023, 2003, 102], vec![1, 1, 1, 1]);
        assert_eq!(input.input_ids.len(), 4);
        assert_eq!(input.attention_mask.len(), 4);
        assert!(input.token_type_ids.is_none());
        assert!(input.special_tokens_mask.is_none());
        assert!(input.offset_mapping.is_none());
        assert!(input.overflowing_tokens.is_none());
    }

    #[test]
    fn test_tokenized_input_with_token_type_ids() {
        let input = TokenizedInput::with_token_type_ids(
            vec![101, 2023, 102, 3045, 102],
            vec![1, 1, 1, 1, 1],
            Some(vec![0, 0, 0, 1, 1]),
        );
        assert_eq!(input.input_ids.len(), 5);
        assert!(input.token_type_ids.is_some());
        if let Some(ref ttype) = input.token_type_ids {
            assert_eq!(ttype.len(), 5);
            assert_eq!(ttype[0], 0);
            assert_eq!(ttype[3], 1);
        }
    }

    #[test]
    fn test_tokenized_input_without_token_type_ids() {
        let input = TokenizedInput::with_token_type_ids(vec![101, 102], vec![1, 1], None);
        assert!(input.token_type_ids.is_none());
    }

    #[test]
    fn test_tokenized_input_default() {
        let input = TokenizedInput::default();
        assert!(input.input_ids.is_empty());
        assert!(input.attention_mask.is_empty());
        assert!(input.token_type_ids.is_none());
    }

    #[test]
    fn test_tokenized_input_clone() {
        let input = TokenizedInput::new(vec![1, 2, 3], vec![1, 1, 1]);
        let cloned = input.clone();
        assert_eq!(cloned.input_ids, input.input_ids);
        assert_eq!(cloned.attention_mask, input.attention_mask);
    }

    #[test]
    fn test_tokenized_input_debug() {
        let input = TokenizedInput::new(vec![1, 2], vec![1, 1]);
        let debug_str = format!("{:?}", input);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_tokenized_input_empty() {
        let input = TokenizedInput::new(vec![], vec![]);
        assert_eq!(input.input_ids.len(), 0);
        assert_eq!(input.attention_mask.len(), 0);
    }

    #[test]
    fn test_tokenized_input_large_ids() {
        let ids: Vec<u32> = (0..1000).collect();
        let mask: Vec<u8> = vec![1; 1000];
        let input = TokenizedInput::new(ids, mask);
        assert_eq!(input.input_ids.len(), 1000);
    }

    #[test]
    fn test_tokenized_input_with_special_fields() {
        let mut input = TokenizedInput::new(vec![101, 2023, 102], vec![1, 1, 1]);
        input.special_tokens_mask = Some(vec![1, 0, 1]);
        input.offset_mapping = Some(vec![(0, 0), (0, 4), (0, 0)]);
        input.overflowing_tokens = Some(vec![3045, 3046]);

        assert!(input.special_tokens_mask.is_some());
        assert!(input.offset_mapping.is_some());
        assert!(input.overflowing_tokens.is_some());
        if let Some(ref stm) = input.special_tokens_mask {
            assert_eq!(stm[0], 1);
            assert_eq!(stm[1], 0);
        }
    }

    // Mock implementations for trait testing

    #[derive(Debug, Deserialize, Serialize)]
    struct MockConfig {
        hidden_size: usize,
    }

    impl Config for MockConfig {
        fn architecture(&self) -> &'static str {
            "mock_model"
        }
    }

    #[test]
    fn test_config_validate_default() {
        let config = MockConfig { hidden_size: 768 };
        let result = config.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_architecture() {
        let config = MockConfig { hidden_size: 256 };
        assert_eq!(config.architecture(), "mock_model");
    }

    struct MockLayer {
        scale: f32,
    }

    impl Layer for MockLayer {
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Self::Input) -> Result<Self::Output> {
            input.mul_scalar(self.scale)
        }
    }

    // Safety: MockLayer is Send + Sync since f32 is Send + Sync
    unsafe impl Send for MockLayer {}
    unsafe impl Sync for MockLayer {}

    #[test]
    fn test_mock_layer_forward() {
        let layer = MockLayer { scale: 2.0 };
        if let Ok(input) = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]) {
            let result = layer.forward(input);
            assert!(result.is_ok());
        }
    }

    struct MockWeightReader {
        tensors: HashMap<String, Tensor>,
    }

    impl WeightReader for MockWeightReader {
        fn read_tensor(&mut self, name: &str) -> Result<Tensor> {
            match self.tensors.get(name) {
                Some(t) => Ok(t.clone()),
                None => Err(crate::errors::TrustformersError::tensor_op_error(
                    "Tensor not found",
                    "read_tensor",
                )),
            }
        }

        fn list_tensors(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }
    }

    #[test]
    fn test_weight_reader_read_existing() {
        let mut tensors = HashMap::new();
        if let Ok(t) = Tensor::zeros(&[4, 4]) {
            tensors.insert("weight".to_string(), t);
        }
        let mut reader = MockWeightReader { tensors };
        let result = reader.read_tensor("weight");
        assert!(result.is_ok());
    }

    #[test]
    fn test_weight_reader_read_nonexistent() {
        let mut reader = MockWeightReader {
            tensors: HashMap::new(),
        };
        let result = reader.read_tensor("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_reader_list_tensors() {
        let mut tensors = HashMap::new();
        if let Ok(t1) = Tensor::zeros(&[2]) {
            tensors.insert("a".to_string(), t1);
        }
        if let Ok(t2) = Tensor::zeros(&[3]) {
            tensors.insert("b".to_string(), t2);
        }
        let reader = MockWeightReader { tensors };
        let names = reader.list_tensors();
        assert_eq!(names.len(), 2);
    }

    struct MockOptimizer {
        lr: f32,
    }

    impl Optimizer for MockOptimizer {
        fn update(&mut self, _parameter: &mut Tensor, _grad: &Tensor) -> Result<()> {
            Ok(())
        }
        fn zero_grad(&mut self) {}
        fn step(&mut self) {}
        fn get_lr(&self) -> f32 {
            self.lr
        }
        fn set_lr(&mut self, lr: f32) {
            self.lr = lr;
        }
    }

    // Safety: MockOptimizer is Send + Sync since f32 is
    unsafe impl Send for MockOptimizer {}
    unsafe impl Sync for MockOptimizer {}

    #[test]
    fn test_optimizer_get_lr() {
        let opt = MockOptimizer { lr: 0.001 };
        assert!((opt.get_lr() - 0.001).abs() < 1e-7);
    }

    #[test]
    fn test_optimizer_set_lr() {
        let mut opt = MockOptimizer { lr: 0.001 };
        opt.set_lr(0.01);
        assert!((opt.get_lr() - 0.01).abs() < 1e-7);
    }

    #[test]
    fn test_optimizer_update() {
        let mut opt = MockOptimizer { lr: 0.001 };
        if let Ok(mut param) = Tensor::from_vec(vec![1.0, 2.0], &[2]) {
            if let Ok(grad) = Tensor::from_vec(vec![0.1, 0.2], &[2]) {
                let result = opt.update(&mut param, &grad);
                assert!(result.is_ok());
            }
        }
    }

    #[test]
    fn test_optimizer_zero_grad() {
        let mut opt = MockOptimizer { lr: 0.001 };
        opt.zero_grad(); // Should not panic
    }

    #[test]
    fn test_optimizer_step() {
        let mut opt = MockOptimizer { lr: 0.001 };
        opt.step(); // Should not panic
    }

    #[test]
    fn test_optimizer_accumulate_grad_default() {
        let mut opt = MockOptimizer { lr: 0.001 };
        if let Ok(mut param) = Tensor::from_vec(vec![1.0], &[1]) {
            if let Ok(grad) = Tensor::from_vec(vec![0.1], &[1]) {
                let result = opt.accumulate_grad(&mut param, &grad);
                assert!(result.is_ok());
            }
        }
    }
}
