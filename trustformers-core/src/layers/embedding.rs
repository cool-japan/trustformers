use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use crate::traits::Layer;
use ndarray::{Array2, Axis};

#[derive(Debug, Clone)]
pub struct Embedding {
    weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
    ) -> Result<Self> {
        let mut weight = Tensor::randn(&[num_embeddings, embedding_dim])?;

        // Zero out padding embedding if specified
        if let Some(padding_idx) = padding_idx {
            if padding_idx < num_embeddings {
                weight = weight.zero_padding_embedding(padding_idx)?;
            }
        }

        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
        })
    }

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    pub fn forward_ids(&self, input_ids: &[u32]) -> Result<Tensor> {
        self.forward(input_ids.to_vec())
    }

    /// Returns the number of parameters in this embedding layer.
    pub fn parameter_count(&self) -> usize {
        self.num_embeddings * self.embedding_dim
    }
}

impl Layer for Embedding {
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &self.weight {
            Tensor::F32(weight_arr) => {
                let batch_size = input.len();
                let mut output = Array2::<f32>::zeros((batch_size, self.embedding_dim));

                for (i, &idx) in input.iter().enumerate() {
                    if idx as usize >= self.num_embeddings {
                        return Err(TrustformersError::tensor_op_error(
                            &format!(
                                "Index {} out of range for embedding table of size {}",
                                idx, self.num_embeddings
                            ),
                            "Embedding::forward",
                        ));
                    }
                    let embedding = weight_arr.index_axis(Axis(0), idx as usize);
                    output.row_mut(i).assign(&embedding);
                }

                Ok(Tensor::F32(output.into_dyn()))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for embedding",
                "Embedding::forward",
            )),
        }
    }
}
