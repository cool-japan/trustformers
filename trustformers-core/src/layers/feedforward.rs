use crate::device::Device;
use crate::errors::Result;
use crate::layers::attention::join_name;
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
    pub fn new_with_device(
        hidden_size: usize,
        intermediate_size: usize,
        dropout_prob: f32,
        device: Device,
    ) -> Self {
        Self {
            dense: Linear::new_with_device(hidden_size, intermediate_size, true, device),
            output: Linear::new_with_device(intermediate_size, hidden_size, true, device),
            dropout_prob,
        }
    }

    pub fn new(hidden_size: usize, intermediate_size: usize, dropout_prob: f32) -> Result<Self> {
        Ok(Self::new_with_device(
            hidden_size,
            intermediate_size,
            dropout_prob,
            Device::CPU,
        ))
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

    /// The dense (first, `hidden -> intermediate`) projection.
    ///
    /// Exposed so a model can publish this block's parameters through
    /// [`Model::named_tensors`](crate::traits::Model::named_tensors) without the
    /// feed-forward block having to know any checkpoint naming convention.
    pub fn dense(&self) -> &Linear {
        &self.dense
    }

    /// The output (second, `intermediate -> hidden`) projection.
    pub fn output(&self) -> &Linear {
        &self.output
    }

    /// Mutable access to the dense (first) projection.
    pub fn dense_mut(&mut self) -> &mut Linear {
        &mut self.dense
    }

    /// Mutable access to the output (second) projection.
    pub fn output_mut(&mut self) -> &mut Linear {
        &mut self.output
    }

    /// Append both projections' parameters to `into`.
    ///
    /// `dense_name` and `output_name` are the checkpoint sub-paths of the two
    /// projections, joined to `prefix` with a `.`; they differ per architecture
    /// (BERT's `intermediate.dense` / `output.dense` versus DistilBERT's
    /// `ffn.lin1` / `ffn.lin2`).
    pub fn collect_named_parameters<'a>(
        &'a self,
        prefix: &str,
        dense_name: &str,
        output_name: &str,
        into: &mut Vec<(String, &'a Tensor)>,
    ) {
        self.dense.collect_named_parameters(&join_name(prefix, dense_name), into);
        self.output.collect_named_parameters(&join_name(prefix, output_name), into);
    }

    /// Mutable counterpart of [`FeedForward::collect_named_parameters`].
    pub fn collect_named_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        dense_name: &str,
        output_name: &str,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        let dense_prefix = join_name(prefix, dense_name);
        let output_prefix = join_name(prefix, output_name);
        let (dense, output) = self.projections_mut();
        dense.collect_named_parameters_mut(&dense_prefix, into);
        output.collect_named_parameters_mut(&output_prefix, into);
    }

    /// Borrow both projections mutably at the same time.
    ///
    /// See [`crate::layers::Linear::parameters_mut`] for why successive `*_mut()`
    /// calls cannot do this.
    pub fn projections_mut(&mut self) -> (&mut Linear, &mut Linear) {
        (&mut self.dense, &mut self.output)
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
