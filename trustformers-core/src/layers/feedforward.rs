use crate::errors::Result;
use crate::layers::Linear;
use crate::ops::activations::gelu;
use crate::tensor::Tensor;
use crate::traits::Layer;

#[derive(Debug, Clone)]
pub struct FeedForward {
    dense: Linear,
    output: Linear,
    #[allow(dead_code)]
    dropout_prob: f32,
}

impl FeedForward {
    pub fn new(hidden_size: usize, intermediate_size: usize, dropout_prob: f32) -> Result<Self> {
        Ok(Self {
            dense: Linear::new(hidden_size, intermediate_size, true),
            output: Linear::new(intermediate_size, hidden_size, true),
            dropout_prob,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.dense.parameter_count() + self.output.parameter_count()
    }

    /// Set weights for the dense (first) layer
    pub fn set_dense_weight(&mut self, weight: Tensor) -> Result<()> {
        self.dense.set_weight(weight)
    }

    /// Set bias for the dense (first) layer
    pub fn set_dense_bias(&mut self, bias: Tensor) -> Result<()> {
        self.dense.set_bias(bias)
    }

    /// Set weights for the output (second) layer
    pub fn set_output_weight(&mut self, weight: Tensor) -> Result<()> {
        self.output.set_weight(weight)
    }

    /// Set bias for the output (second) layer
    pub fn set_output_bias(&mut self, bias: Tensor) -> Result<()> {
        self.output.set_bias(bias)
    }
}

impl Layer for FeedForward {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.dense.forward(input)?;
        let hidden_states = gelu(&hidden_states)?;
        self.output.forward(hidden_states)
    }
}
