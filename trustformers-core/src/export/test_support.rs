//! Shared test models for the exporter regression suite.
//!
//! Every exporter is checked with the same three fixtures:
//!
//! * two models with identical shapes but *different values* — the artifacts they
//!   produce must differ, which is exactly what the old dummy-weight exporters
//!   could not do;
//! * a model whose [`Model::named_tensors`] is empty — every exporter must refuse
//!   to write anything rather than invent weights.

use crate::errors::Result;
use crate::tensor::Tensor;
use crate::traits::{Config, Model};
use serde::{Deserialize, Serialize};
use std::io::Read;

/// Minimal configuration used by the exporter test models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4,
            num_layers: 2,
            vocab_size: 6,
        }
    }
}

impl Config for TestConfig {
    fn architecture(&self) -> &'static str {
        "test_transformer"
    }
}

/// A model that owns an explicit list of named tensors.
#[derive(Debug)]
pub struct TestModel {
    config: TestConfig,
    tensors: Vec<(String, Tensor)>,
}

impl TestModel {
    /// Build a model from an explicit `(name, tensor)` list.
    pub fn new(config: TestConfig, tensors: Vec<(String, Tensor)>) -> Self {
        Self { config, tensors }
    }

    /// A small model whose every weight is derived from `seed`, so two different
    /// seeds give the same shapes with different values.
    pub fn with_seed(seed: f32) -> Self {
        let config = TestConfig::default();
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;

        let embed: Vec<f32> = (0..vocab * hidden).map(|i| seed + (i as f32) * 0.25).collect();
        let proj: Vec<f32> = (0..hidden * hidden).map(|i| seed - (i as f32) * 0.5).collect();
        let norm: Vec<f32> = (0..hidden).map(|i| seed + (i as f32)).collect();

        let tensors = vec![
            (
                "token_embd.weight".to_string(),
                Tensor::from_vec(embed, &[vocab, hidden]).expect("valid embedding shape"),
            ),
            (
                "blk.0.attn_output.weight".to_string(),
                Tensor::from_vec(proj, &[hidden, hidden]).expect("valid projection shape"),
            ),
            (
                "output_norm.weight".to_string(),
                Tensor::from_vec(norm, &[hidden]).expect("valid norm shape"),
            ),
        ];

        Self::new(config, tensors)
    }

    /// A model that exposes no named tensors at all.
    pub fn empty() -> Self {
        Self::new(TestConfig::default(), Vec::new())
    }
}

impl Model for TestModel {
    type Config = TestConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        Ok(input)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.tensors.iter().map(|(_, t)| t.len()).sum()
    }

    fn named_tensors(&self) -> Vec<(String, &Tensor)> {
        self.tensors.iter().map(|(name, tensor)| (name.clone(), tensor)).collect()
    }

    fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        self.tensors.iter_mut().map(|(name, tensor)| (name.clone(), tensor)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_models_differ_in_values_but_not_shapes() {
        let a = TestModel::with_seed(0.0);
        let b = TestModel::with_seed(1.0);

        let a_named = a.named_tensors();
        let b_named = b.named_tensors();
        assert_eq!(a_named.len(), b_named.len());

        let mut any_difference = false;
        for ((a_name, a_tensor), (b_name, b_tensor)) in a_named.iter().zip(b_named.iter()) {
            assert_eq!(a_name, b_name);
            assert_eq!(a_tensor.shape(), b_tensor.shape());
            if a_tensor.to_vec_f32().expect("f32 data") != b_tensor.to_vec_f32().expect("f32 data")
            {
                any_difference = true;
            }
        }
        assert!(any_difference, "seeded models must differ in values");
    }

    #[test]
    fn empty_model_exposes_no_tensors() {
        assert!(TestModel::empty().named_tensors().is_empty());
    }
}
