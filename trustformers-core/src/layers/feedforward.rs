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
