use super::config::{HierarchicalConfig, HierarchicalType};
use super::layers::{HierarchicalEncoder, NestedTransformerLayer, PyramidLayer, TreeAttention};
use super::utils::HierarchicalOutput;
use trustformers_core::{
    errors::{invalid_config, invalid_input, not_implemented, tensor_op_error, Result},
    layers::{Embedding, LayerNorm, Linear},
    tensor::Tensor,
    traits::{Layer, Model},
};

/// Embed a token sequence into a `[1, seq_len, hidden_size]` activation tensor.
///
/// [`Embedding`] returns a rank-2 `[seq_len, hidden_size]` lookup, but every
/// hierarchical component — the windowed pooling in
/// [`build_hierarchy`](super::utils::build_hierarchy), the level aggregation and
/// `MultiHeadAttention::split_heads` — is defined on the `[batch, seq, hidden]`
/// contract. The batch axis is therefore added exactly once, here, right after the
/// lookup.
///
/// Without it `HierarchicalTransformer`, `TreeTransformer` and `NestedTransformer`
/// failed on *every* input ("expected a 3-D [batch, seq, hidden] tensor" /
/// "split_heads … got 2"), and `PyramidTransformer` returned a rank-2 tensor whose
/// downstream "CLS token" selection (`select(1, 0)`) picked hidden channel 0 across
/// the whole sequence instead of the first token's vector.
fn embed_sequence(embeddings: &Embedding, input_ids: Vec<u32>) -> Result<Tensor> {
    if input_ids.is_empty() {
        return Err(invalid_input(
            "hierarchical models require at least one input token",
        ));
    }

    let embedded = embeddings.forward(input_ids)?;
    let shape = embedded.shape();
    match shape.len() {
        2 => embedded.reshape(&[1, shape[0], shape[1]]),
        3 => Ok(embedded),
        _ => Err(tensor_op_error(
            "hierarchical_embed",
            format!("embedding lookup produced an unusable shape {shape:?}"),
        )),
    }
}

/// The error every hierarchical checkpoint entry point returns.
///
/// No published checkpoint uses this family's parameter layout, and no naming
/// scheme has been agreed for it, so there is nothing to parse. Returning `Ok(())`
/// here would leave the caller holding a randomly initialised model while
/// believing a checkpoint had been applied — a previous revision did exactly
/// that, printing `Loaded <tensor>` lines for tensors it then discarded.
fn checkpoint_unsupported(model: &str) -> trustformers_core::errors::TrustformersError {
    not_implemented(format!(
        "{model}: no hierarchical checkpoint format is parsed yet. There is no agreed parameter \
         naming scheme for this model family, so loading would silently leave the randomly \
         initialised weights in place. Assign weights explicitly through the individual \
         Linear/LayerNorm/Embedding/MultiHeadAttention setters instead."
    ))
}

/// Main hierarchical transformer model
pub struct HierarchicalTransformer {
    config: HierarchicalConfig,
    embeddings: Embedding,
    encoder: HierarchicalEncoder,
    final_norm: LayerNorm,
}

impl HierarchicalTransformer {
    pub fn new(config: HierarchicalConfig, vocab_size: usize) -> Result<Self> {
        config.validate().map_err(|e| invalid_config("config_field", e.to_string()))?;

        let embeddings = Embedding::new(vocab_size, config.hidden_size, None)?;
        let encoder = HierarchicalEncoder::new(config.clone())?;
        let final_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        Ok(Self {
            config,
            embeddings,
            encoder,
            final_norm,
        })
    }

    /// Load weights for this model from a local checkpoint directory or file.
    ///
    /// # Errors
    ///
    /// Always fails: see [`checkpoint_unsupported`]. A previous revision walked a
    /// safetensors file, printed the shape of every tensor it found and then
    /// returned `Ok(())` without assigning any of them, so a caller ended up
    /// running the randomly initialised model believing it held pretrained
    /// weights.
    pub fn load_from_path(&mut self, _model_path: impl AsRef<std::path::Path>) -> Result<()> {
        Err(checkpoint_unsupported(
            "HierarchicalTransformer::load_from_path",
        ))
    }

    /// Load weights for this model from a HuggingFace Hub repository.
    ///
    /// # Errors
    ///
    /// Always fails: there is nothing that could consume a downloaded checkpoint,
    /// so no download is attempted either.
    pub fn load_from_huggingface(&mut self, _model_name: &str) -> Result<()> {
        Err(checkpoint_unsupported(
            "HierarchicalTransformer::load_from_huggingface",
        ))
    }
}

impl Model for HierarchicalTransformer {
    type Config = HierarchicalConfig;
    type Input = Vec<u32>;
    type Output = HierarchicalOutput;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        let embeddings = embed_sequence(&self.embeddings, input_ids)?;
        let encoder_output = self.encoder.forward(embeddings)?;
        let final_output = self.final_norm.forward(encoder_output.output)?;

        Ok(HierarchicalOutput {
            output: final_output,
            level_outputs: encoder_output.level_outputs,
            attention_weights: encoder_output.attention_weights,
            hierarchical_positions: encoder_output.hierarchical_positions,
        })
    }
    /// Loading a pretrained checkpoint is not implemented.
    ///
    /// # Errors
    ///
    /// Always fails: see [`checkpoint_unsupported`]. A previous revision spooled
    /// the reader to a temporary file, printed
    /// `Weight loading fallback - weights successfully processed`, deleted the
    /// file and returned `Ok(())` — the model kept every randomly initialised
    /// weight while the caller was told the load had succeeded.
    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
        Err(checkpoint_unsupported(
            "HierarchicalTransformer::load_pretrained",
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Embedding parameters
        total += self.embeddings.parameter_count();

        // Encoder parameters
        total += self.encoder.parameter_count();

        // Final norm parameters
        total += self.final_norm.parameter_count();

        total
    }
}

/// Pyramid transformer model
pub struct PyramidTransformer {
    config: HierarchicalConfig,
    embeddings: Embedding,
    pyramid_layers: Vec<PyramidLayer>,
    final_norm: LayerNorm,
}

impl PyramidTransformer {
    pub fn new(config: HierarchicalConfig, vocab_size: usize) -> Result<Self> {
        let embeddings = Embedding::new(vocab_size, config.hidden_size, None)?;

        let mut pyramid_layers = Vec::new();
        for _ in 0..config.num_layers_per_level {
            pyramid_layers.push(PyramidLayer::new(config.clone())?);
        }

        let final_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        Ok(Self {
            config,
            embeddings,
            pyramid_layers,
            final_norm,
        })
    }

    /// Load weights for this model from a local checkpoint.
    ///
    /// # Errors
    ///
    /// Always fails: see [`checkpoint_unsupported`]. A previous revision called
    /// three private helpers whose entire bodies were `Ok(())` with a comment
    /// saying an implementation "would" load the weights, so the model kept its
    /// random initialisation while reporting a successful load.
    pub fn load_from_path(&mut self, _model_path: &str) -> Result<()> {
        Err(checkpoint_unsupported("PyramidTransformer::load_from_path"))
    }
}

impl Model for PyramidTransformer {
    type Config = HierarchicalConfig;
    type Input = Vec<u32>;
    type Output = HierarchicalOutput;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        let mut hidden_states = embed_sequence(&self.embeddings, input_ids)?;
        let mut all_level_outputs = Vec::new();

        for layer in &self.pyramid_layers {
            let output = layer.forward(hidden_states)?;
            hidden_states = output.output;
            all_level_outputs.extend(output.level_outputs);
        }

        let final_output = self.final_norm.forward(hidden_states)?;

        Ok(HierarchicalOutput {
            output: final_output,
            level_outputs: all_level_outputs,
            attention_weights: None,
            hierarchical_positions: None,
        })
    }
    /// Loading a pretrained checkpoint is not implemented.
    ///
    /// # Errors
    ///
    /// Always fails: see [`checkpoint_unsupported`]. A previous revision spooled
    /// the reader to a temporary file, printed
    /// `Weight loading fallback - weights successfully processed`, deleted the
    /// file and returned `Ok(())` — the model kept every randomly initialised
    /// weight while the caller was told the load had succeeded.
    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
        Err(checkpoint_unsupported(
            "PyramidTransformer::load_pretrained",
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Embedding parameters
        total += self.embeddings.parameter_count();

        // Pyramid layer parameters
        for layer in &self.pyramid_layers {
            total += layer.parameter_count();
        }

        // Final norm parameters
        total += self.final_norm.parameter_count();

        total
    }
}

/// Tree transformer model
pub struct TreeTransformer {
    config: HierarchicalConfig,
    embeddings: Embedding,
    tree_layers: Vec<TreeAttention>,
    final_norm: LayerNorm,
}

impl TreeTransformer {
    pub fn new(config: HierarchicalConfig, vocab_size: usize) -> Result<Self> {
        let embeddings = Embedding::new(vocab_size, config.hidden_size, None)?;

        let mut tree_layers = Vec::new();
        for _ in 0..config.num_layers_per_level {
            tree_layers.push(TreeAttention::new(config.clone())?);
        }

        let final_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        Ok(Self {
            config,
            embeddings,
            tree_layers,
            final_norm,
        })
    }
}

impl Model for TreeTransformer {
    type Config = HierarchicalConfig;
    type Input = Vec<u32>;
    type Output = HierarchicalOutput;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        let mut hidden_states = embed_sequence(&self.embeddings, input_ids)?;

        for layer in &self.tree_layers {
            let output = layer.forward(hidden_states)?;
            hidden_states = output.output;
        }

        let final_output = self.final_norm.forward(hidden_states)?;

        Ok(HierarchicalOutput {
            output: final_output,
            level_outputs: vec![],
            attention_weights: None,
            hierarchical_positions: None,
        })
    }
    /// Loading a pretrained checkpoint is not implemented.
    ///
    /// # Errors
    ///
    /// Always fails: see [`checkpoint_unsupported`]. A previous revision spooled
    /// the reader to a temporary file, printed
    /// `Weight loading fallback - weights successfully processed`, deleted the
    /// file and returned `Ok(())` — the model kept every randomly initialised
    /// weight while the caller was told the load had succeeded.
    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
        Err(checkpoint_unsupported("TreeTransformer::load_pretrained"))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Embedding parameters
        total += self.embeddings.parameter_count();

        // Tree layer parameters
        for layer in &self.tree_layers {
            total += layer.parameter_count();
        }

        // Final norm parameters
        total += self.final_norm.parameter_count();

        total
    }
}

/// Nested transformer model
pub struct NestedTransformer {
    config: HierarchicalConfig,
    embeddings: Embedding,
    nested_layers: Vec<NestedTransformerLayer>,
    final_norm: LayerNorm,
}

impl NestedTransformer {
    pub fn new(config: HierarchicalConfig, vocab_size: usize) -> Result<Self> {
        let embeddings = Embedding::new(vocab_size, config.hidden_size, None)?;

        let mut nested_layers = Vec::new();
        for _ in 0..config.num_layers_per_level {
            nested_layers.push(NestedTransformerLayer::new(config.clone())?);
        }

        let final_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        Ok(Self {
            config,
            embeddings,
            nested_layers,
            final_norm,
        })
    }
}

impl Model for NestedTransformer {
    type Config = HierarchicalConfig;
    type Input = Vec<u32>;
    type Output = HierarchicalOutput;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        let mut hidden_states = embed_sequence(&self.embeddings, input_ids)?;
        let mut all_level_outputs = Vec::new();

        for layer in &self.nested_layers {
            let output = layer.forward(hidden_states)?;
            hidden_states = output.output;
            all_level_outputs.extend(output.level_outputs);
        }

        let final_output = self.final_norm.forward(hidden_states)?;

        Ok(HierarchicalOutput {
            output: final_output,
            level_outputs: all_level_outputs,
            attention_weights: None,
            hierarchical_positions: None,
        })
    }
    /// Loading a pretrained checkpoint is not implemented.
    ///
    /// # Errors
    ///
    /// Always fails: see [`checkpoint_unsupported`]. A previous revision spooled
    /// the reader to a temporary file, printed
    /// `Weight loading fallback - weights successfully processed`, deleted the
    /// file and returned `Ok(())` — the model kept every randomly initialised
    /// weight while the caller was told the load had succeeded.
    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
        Err(checkpoint_unsupported("NestedTransformer::load_pretrained"))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Embedding parameters
        total += self.embeddings.parameter_count();

        // Nested layer parameters
        for layer in &self.nested_layers {
            total += layer.parameter_count();
        }

        // Final norm parameters
        total += self.final_norm.parameter_count();

        total
    }
}

/// Hierarchical transformer for sequence classification
pub struct HierarchicalForSequenceClassification {
    base_model: HierarchicalTransformer,
    classifier: Linear,
    num_labels: usize,
}

impl HierarchicalForSequenceClassification {
    pub fn new(config: HierarchicalConfig, vocab_size: usize, num_labels: usize) -> Result<Self> {
        let base_model = HierarchicalTransformer::new(config.clone(), vocab_size)?;
        let classifier = Linear::new(config.hidden_size, num_labels, true);

        Ok(Self {
            base_model,
            classifier,
            num_labels,
        })
    }

    /// Number of classification labels this head produces.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }
}

impl Model for HierarchicalForSequenceClassification {
    type Config = HierarchicalConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        let model_output = self.base_model.forward(input_ids)?;

        // Use CLS token (first token) for classification
        let cls_output = model_output.output.select(1, 0)?;
        let logits = self.classifier.forward(cls_output)?;

        Ok(logits)
    }
    /// Loading a pretrained checkpoint is not implemented.
    ///
    /// # Errors
    ///
    /// Always fails: see [`checkpoint_unsupported`]. A previous revision spooled
    /// the reader to a temporary file, printed
    /// `Weight loading fallback - weights successfully processed`, deleted the
    /// file and returned `Ok(())` — the model kept every randomly initialised
    /// weight while the caller was told the load had succeeded.
    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
        Err(checkpoint_unsupported(
            "HierarchicalForSequenceClassification::load_pretrained",
        ))
    }

    fn get_config(&self) -> &Self::Config {
        self.base_model.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.base_model.num_parameters() + self.classifier.parameter_count()
    }
}

/// Hierarchical transformer for language modeling
pub struct HierarchicalForLanguageModeling {
    base_model: HierarchicalTransformer,
    lm_head: Linear,
    vocab_size: usize,
}

impl HierarchicalForLanguageModeling {
    pub fn new(config: HierarchicalConfig, vocab_size: usize) -> Result<Self> {
        let base_model = HierarchicalTransformer::new(config.clone(), vocab_size)?;
        let lm_head = Linear::new(config.hidden_size, vocab_size, false);

        Ok(Self {
            base_model,
            lm_head,
            vocab_size,
        })
    }

    /// Size of the vocabulary this head projects onto.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl Model for HierarchicalForLanguageModeling {
    type Config = HierarchicalConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        let model_output = self.base_model.forward(input_ids)?;
        let logits = self.lm_head.forward(model_output.output)?;

        Ok(logits)
    }
    /// Loading a pretrained checkpoint is not implemented.
    ///
    /// # Errors
    ///
    /// Always fails: see [`checkpoint_unsupported`]. A previous revision spooled
    /// the reader to a temporary file, printed
    /// `Weight loading fallback - weights successfully processed`, deleted the
    /// file and returned `Ok(())` — the model kept every randomly initialised
    /// weight while the caller was told the load had succeeded.
    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
        Err(checkpoint_unsupported(
            "HierarchicalForLanguageModeling::load_pretrained",
        ))
    }

    fn get_config(&self) -> &Self::Config {
        self.base_model.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.base_model.num_parameters() + self.lm_head.parameter_count()
    }
}

/// Factory function to create hierarchical transformers
pub fn create_hierarchical_transformer(
    config: HierarchicalConfig,
    vocab_size: usize,
) -> Result<
    Box<dyn Model<Config = HierarchicalConfig, Input = Vec<u32>, Output = HierarchicalOutput>>,
> {
    match config.hierarchical_type {
        HierarchicalType::Hierarchical => {
            let model = HierarchicalTransformer::new(config, vocab_size)?;
            Ok(Box::new(model))
        },
        HierarchicalType::Pyramid => {
            let model = PyramidTransformer::new(config, vocab_size)?;
            Ok(Box::new(model))
        },
        HierarchicalType::Tree => {
            let model = TreeTransformer::new(config, vocab_size)?;
            Ok(Box::new(model))
        },
        HierarchicalType::Nested => {
            let model = NestedTransformer::new(config, vocab_size)?;
            Ok(Box::new(model))
        },
        // A hybrid stack would have to interleave the pyramid, tree and nested
        // blocks, and no such composition is defined here. A previous revision
        // quietly built a plain `HierarchicalTransformer` instead, so the caller
        // received a different architecture than the one it asked for.
        HierarchicalType::Hybrid => Err(not_implemented(
            "HierarchicalType::Hybrid: no hybrid composition of the pyramid/tree/nested blocks is \
             defined; pick an explicit hierarchical_type",
        )),
    }
}
