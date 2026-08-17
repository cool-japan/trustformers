use crate::bert::config::BertConfig;
use crate::bert::layers::{BertEmbeddings, BertEncoder, BertLayerNames, BertPooler};
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, UnusedTensors};
use scirs2_core::ndarray::{ArrayD, IxDyn}; // SciRS2 Integration Policy
use std::io::Read;
use trustformers_core::device::Device;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Model, TokenizedInput};

#[derive(Debug, Clone)]
pub struct BertModel {
    config: BertConfig,
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: Option<BertPooler>,
    device: Device,
}

impl BertModel {
    pub fn new(config: BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: BertConfig, device: Device) -> Result<Self> {
        let embeddings = BertEmbeddings::new_with_device(&config, device)?;
        let encoder = BertEncoder::new_with_device(&config, device)?;
        let pooler = Some(BertPooler::new_with_device(&config, device)?);

        Ok(Self {
            config,
            embeddings,
            encoder,
            pooler,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward_with_embeddings(
        &self,
        input_ids: Vec<u32>,
        attention_mask: Option<Vec<u8>>,
        token_type_ids: Option<Vec<u32>>,
    ) -> Result<BertModelOutput> {
        let embeddings = self.embeddings.forward(input_ids.clone(), token_type_ids)?;

        // Add batch dimension: [seq_len, hidden_size] -> [1, seq_len, hidden_size]
        let batch_size = 1;
        let seq_len = input_ids.len();
        let hidden_size = self.config.hidden_size;

        let embeddings = match embeddings {
            trustformers_core::tensor::Tensor::F32(arr) => {
                let reshaped = arr
                    .to_shape(IxDyn(&[batch_size, seq_len, hidden_size]))
                    .map_err(|e| {
                        trustformers_core::errors::TrustformersError::shape_error(e.to_string())
                    })?
                    .to_owned();
                trustformers_core::tensor::Tensor::F32(reshaped)
            },
            _ => {
                return Err(
                    trustformers_core::errors::TrustformersError::tensor_op_error(
                        "Unsupported tensor type in embeddings",
                        "BertModel::forward_with_embeddings",
                    ),
                )
            },
        };

        let attention_mask_tensor = if let Some(mask) = attention_mask {
            let mask_f32: Vec<f32> = mask.iter().map(|&m| m as f32).collect();
            let shape = vec![1, 1, 1, mask_f32.len()];
            Some(Tensor::F32(
                ArrayD::from_shape_vec(IxDyn(&shape), mask_f32).map_err(|e| {
                    trustformers_core::errors::TrustformersError::shape_error(e.to_string())
                })?,
            ))
        } else {
            None
        };

        let encoder_output = self.encoder.forward(embeddings, attention_mask_tensor)?;

        // Temporarily disable pooler to test main tensor flow
        let pooler_output = None;

        Ok(BertModelOutput {
            last_hidden_state: encoder_output,
            pooler_output,
        })
    }
}

#[derive(Debug)]
pub struct BertModelOutput {
    pub last_hidden_state: Tensor,
    pub pooler_output: Option<Tensor>,
}

impl Model for BertModel {
    type Config = BertConfig;
    type Input = TokenizedInput;
    type Output = BertModelOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_with_embeddings(
            input.input_ids,
            Some(input.attention_mask),
            input.token_type_ids,
        )
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &<BertModel as Model>::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let embeddings_params = self.embeddings.parameter_count();
        let encoder_params = self.encoder.parameter_count();
        let pooler_params =
            if let Some(ref pooler) = self.pooler { pooler.parameter_count() } else { 0 };

        embeddings_params + encoder_params + pooler_params
    }
}

impl BertModel {
    /// Checkpoint namespaces a base BERT encoder legitimately does not consume.
    ///
    /// HuggingFace ships the pretraining and task heads inside the same
    /// checkpoint as the encoder, so `bert-base-uncased` carries
    /// `cls.predictions.*` next to `embeddings.*`. Loading the *encoder alone*
    /// therefore leaves them behind; the task wrappers in
    /// [`crate::bert::tasks`] bind their own head from the same checkpoint
    /// instead of relying on this list.
    pub(crate) const ALLOWED_UNUSED_PREFIXES: &'static [&'static str] = &[
        "cls.",
        "classifier.",
        "qa_outputs.",
        "lm_head.",
        "mlm_head.",
        "nsp_head.",
    ];

    /// Non-parameter buffers HuggingFace stores alongside BERT's weights.
    ///
    /// `position_ids` / `token_type_ids` are registered buffers, not learnable
    /// parameters, and they repeat under whatever task prefix the checkpoint
    /// uses — so they are matched by suffix rather than by namespace.
    pub(crate) const ALLOWED_UNUSED_SUFFIXES: &'static [&'static str] = &[
        "embeddings.position_ids",
        "embeddings.token_type_ids",
        "attention.self.position_embeddings.weight",
    ];

    /// The unused-tensor policy for a bare BERT encoder load.
    pub(crate) fn unused_tensor_policy() -> UnusedTensors<'static> {
        UnusedTensors::new(Self::ALLOWED_UNUSED_PREFIXES, Self::ALLOWED_UNUSED_SUFFIXES)
    }

    /// Load pretrained weights and report exactly what was bound.
    ///
    /// The stream may hold a safetensors file or a PyTorch `.bin` archive; the
    /// container is detected and parsed for real by
    /// [`Checkpoint::from_reader`]. Every parameter of the model is then bound by
    /// name, and the load fails — listing the offenders — if the checkpoint is
    /// missing any of them or carries tensors this architecture does not know.
    ///
    /// A previous revision scanned the raw bytes for a run of floats that "looked
    /// reasonable", fell back to `fastrand`-generated values when the scan found
    /// nothing, then discarded every tensor it had produced and returned `Ok(())`.
    /// Loading a checkpoint left the model randomly initialised while reporting
    /// success. None of that survives: weights are either bound or an error is
    /// returned.
    ///
    /// # Errors
    ///
    /// Fails when the container cannot be parsed, when the checkpoint does not
    /// look like a BERT checkpoint, when a tensor has the wrong shape, or when
    /// any parameter is missing.
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_from_checkpoint(&checkpoint)
    }

    /// Bind an already-parsed checkpoint into this model.
    ///
    /// # Errors
    ///
    /// See [`BertModel::load_pretrained_report`].
    pub fn load_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<LoadReport> {
        // `BertForSequenceClassification` and friends nest the encoder under
        // `bert.`; the plain `BertModel` export does not.
        let prefix =
            checkpoint.detect_prefix(&["", "bert."], "embeddings.word_embeddings.weight")?;
        let mut binder = checkpoint.binder(&prefix);
        let names = BertLayerNames::bert();

        self.embeddings.load_weights(&mut binder, &self.config, true)?;
        self.encoder.load_weights(&mut binder, &names, &self.config)?;

        // The pooler is optional in HuggingFace exports (`add_pooling_layer=False`).
        // When the checkpoint has no pooler, drop ours rather than leaving a
        // randomly-initialised projection in place pretending to be trained.
        if checkpoint.contains(&format!("{prefix}pooler.dense.weight")) {
            let pooler = match self.pooler.as_mut() {
                Some(pooler) => pooler,
                None => {
                    self.pooler = Some(BertPooler::new_with_device(&self.config, self.device)?);
                    self.pooler.as_mut().ok_or_else(|| {
                        TrustformersError::model_error(
                            "BERT pooler could not be created for the checkpoint".to_string(),
                        )
                    })?
                },
            };
            pooler.load_weights(&mut binder, &self.config)?;
        } else {
            self.pooler = None;
        }

        binder.finish(Self::unused_tensor_policy())
    }

    #[allow(dead_code)]
    fn get_config(&self) -> &BertConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::{Model, TokenizedInput};

    // --- LCG for reproducible sequences ---
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        *state
    }

    fn lcg_token(state: &mut u64, vocab: u32) -> u32 {
        (lcg_next(state) >> 33) as u32 % vocab
    }

    /// Build a tiny BertConfig suitable for fast unit tests.
    fn tiny_config() -> BertConfig {
        BertConfig {
            vocab_size: 256,
            hidden_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            intermediate_size: 128,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(false),
            classifier_dropout: None,
        }
    }

    // --- Construction ---

    #[test]
    fn test_bert_model_new_cpu() {
        let cfg = tiny_config();
        let model = BertModel::new(cfg).expect("BertModel::new must succeed");
        assert_eq!(model.device(), Device::CPU);
    }

    #[test]
    fn test_bert_model_new_with_device_cpu() {
        let cfg = tiny_config();
        let model = BertModel::new_with_device(cfg, Device::CPU)
            .expect("BertModel::new_with_device must succeed");
        assert_eq!(model.device(), Device::CPU);
    }

    // --- num_parameters ---

    #[test]
    fn test_bert_model_num_parameters_positive() {
        let cfg = tiny_config();
        let model = BertModel::new(cfg).expect("BertModel::new must succeed");
        assert!(
            model.num_parameters() > 0,
            "num_parameters must be positive"
        );
    }

    #[test]
    fn test_bert_model_larger_config_has_more_params() {
        let small = tiny_config();
        let big = BertConfig {
            vocab_size: 256,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            intermediate_size: 256,
            ..tiny_config()
        };
        let m_small = BertModel::new(small).expect("small model must succeed");
        let m_big = BertModel::new(big).expect("big model must succeed");
        assert!(
            m_big.num_parameters() > m_small.num_parameters(),
            "Larger config must have more parameters"
        );
    }

    // --- forward_with_embeddings ---

    #[test]
    fn test_bert_forward_last_hidden_state_shape() {
        let cfg = tiny_config();
        let model = BertModel::new(cfg.clone()).expect("BertModel::new must succeed");
        let seq_len = 5usize;
        let mut state: u64 = 42;
        let input_ids: Vec<u32> =
            (0..seq_len).map(|_| lcg_token(&mut state, cfg.vocab_size as u32)).collect();
        let output = model
            .forward_with_embeddings(input_ids, None, None)
            .expect("forward_with_embeddings must succeed");
        let shape = output.last_hidden_state.shape();
        // Expected: [batch=1, seq_len, hidden_size]
        assert_eq!(shape[0], 1, "batch dimension must be 1");
        assert_eq!(shape[1], seq_len, "second dim must equal seq_len");
        assert_eq!(
            shape[2], cfg.hidden_size,
            "third dim must equal hidden_size"
        );
    }

    #[test]
    fn test_bert_forward_with_attention_mask() {
        let cfg = tiny_config();
        let model = BertModel::new(cfg.clone()).expect("BertModel::new must succeed");
        let seq_len = 4usize;
        let input_ids: Vec<u32> = (0..seq_len as u32).collect();
        let attention_mask: Vec<u8> = vec![1, 1, 1, 0];
        let output = model
            .forward_with_embeddings(input_ids, Some(attention_mask), None)
            .expect("forward with attention mask must succeed");
        let shape = output.last_hidden_state.shape();
        assert_eq!(shape[1], seq_len);
        assert_eq!(shape[2], cfg.hidden_size);
    }

    #[test]
    fn test_bert_forward_with_token_type_ids() {
        let cfg = tiny_config();
        let model = BertModel::new(cfg.clone()).expect("BertModel::new must succeed");
        let seq_len = 6usize;
        let input_ids: Vec<u32> = (0..seq_len as u32).collect();
        let token_type_ids: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
        let output = model
            .forward_with_embeddings(input_ids, None, Some(token_type_ids))
            .expect("forward with token_type_ids must succeed");
        let shape = output.last_hidden_state.shape();
        assert_eq!(shape[1], seq_len);
        assert_eq!(shape[2], cfg.hidden_size);
    }

    // --- Model trait ---

    #[test]
    fn test_bert_model_trait_forward() {
        let cfg = tiny_config();
        let model = BertModel::new(cfg.clone()).expect("BertModel::new must succeed");
        let seq_len = 4usize;
        let input_ids: Vec<u32> = (0..seq_len as u32).collect();
        let attention_mask: Vec<u8> = vec![1u8; seq_len];
        let input = TokenizedInput::new(input_ids, attention_mask);
        let output = model.forward(input).expect("Model::forward must succeed");
        let shape = output.last_hidden_state.shape();
        assert_eq!(shape[1], seq_len);
        assert_eq!(shape[2], cfg.hidden_size);
    }

    #[test]
    fn test_bert_model_get_config() {
        let cfg = tiny_config();
        let model = BertModel::new(cfg.clone()).expect("BertModel::new must succeed");
        let returned = model.get_config();
        assert_eq!(returned.vocab_size, cfg.vocab_size);
        assert_eq!(returned.hidden_size, cfg.hidden_size);
    }

    // --- Bidirectional attention property: all tokens attend to each other ---
    // Note: with zero-initialized weights, LayerNorm may normalize outputs to similar values.
    // We verify the model produces consistent shapes for different inputs (structural test).

    #[test]
    fn test_bert_forward_outputs_are_finite() {
        let cfg = tiny_config();
        let model = BertModel::new(cfg.clone()).expect("BertModel::new must succeed");
        let input_ids: Vec<u32> = vec![1, 2, 3, 4];
        let output = model
            .forward_with_embeddings(input_ids, None, None)
            .expect("forward must succeed");
        if let trustformers_core::tensor::Tensor::F32(arr) = &output.last_hidden_state {
            for &v in arr.iter() {
                assert!(v.is_finite(), "BERT output must be finite, got {}", v);
            }
        }
    }

    #[test]
    fn test_bert_forward_same_input_produces_same_output() {
        let cfg = tiny_config();
        let model = BertModel::new(cfg.clone()).expect("BertModel::new must succeed");
        // The model must be deterministic: same input -> same output
        let out1 = model
            .forward_with_embeddings(vec![1, 2, 3, 4], None, None)
            .expect("first forward must succeed");
        let out2 = model
            .forward_with_embeddings(vec![1, 2, 3, 4], None, None)
            .expect("second forward must succeed");
        if let (
            trustformers_core::tensor::Tensor::F32(a),
            trustformers_core::tensor::Tensor::F32(b),
        ) = (&out1.last_hidden_state, &out2.last_hidden_state)
        {
            let all_equal = a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-6);
            assert!(
                all_equal,
                "BERT model must be deterministic: same input must produce same output"
            );
        }
    }

    // --- Single-token input ---

    #[test]
    fn test_bert_single_token_forward() {
        let cfg = tiny_config();
        let model = BertModel::new(cfg.clone()).expect("BertModel::new must succeed");
        let input_ids: Vec<u32> = vec![5];
        let output = model
            .forward_with_embeddings(input_ids, None, None)
            .expect("single-token forward must succeed");
        let shape = output.last_hidden_state.shape();
        assert_eq!(shape[1], 1, "single token input: seq_len must be 1");
        assert_eq!(shape[2], cfg.hidden_size);
    }
    // --- Real weight loading (regression for the fabricated-tensor loader) ---

    use crate::weight_loading::test_support::{build_safetensors, BertFixtureSpec, F32Tensor};

    fn loading_config() -> BertConfig {
        BertConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 8,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(false),
            classifier_dropout: None,
        }
    }

    fn fixture_spec(config: &BertConfig, prefix: &str) -> BertFixtureSpec {
        BertFixtureSpec {
            prefix: prefix.to_string(),
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            intermediate_size: config.intermediate_size,
            max_position_embeddings: config.max_position_embeddings,
            type_vocab_size: Some(config.type_vocab_size),
            include_pooler: true,
            names: BertLayerNames::bert(),
        }
    }

    fn hidden_values(model: &BertModel) -> Vec<f32> {
        let output = model
            .forward_with_embeddings(vec![1, 2, 3], Some(vec![1, 1, 1]), Some(vec![0, 0, 1]))
            .expect("forward must succeed after loading");
        match output.last_hidden_state {
            Tensor::F32(arr) => arr.iter().copied().collect(),
            other => panic!("expected an F32 hidden state, got {other:?}"),
        }
    }

    #[test]
    fn load_pretrained_makes_the_model_fully_determined_by_the_checkpoint() {
        // Two independently randomly-initialised models must become numerically
        // identical after loading the same checkpoint. That can only hold if every
        // parameter was really overwritten -- the previous implementation printed
        // shapes and discarded every tensor, so this assertion fails against it.
        let config = loading_config();
        let bytes = fixture_spec(&config, "").safetensors();

        let mut first = BertModel::new(config.clone()).expect("model must build");
        let mut second = BertModel::new(config.clone()).expect("model must build");

        let before_first = hidden_values(&first);
        let before_second = hidden_values(&second);
        assert_ne!(
            before_first, before_second,
            "two random initialisations must differ, otherwise the test proves nothing"
        );

        first
            .load_pretrained(&mut bytes.as_slice())
            .expect("checkpoint must load into the first model");
        second
            .load_pretrained(&mut bytes.as_slice())
            .expect("checkpoint must load into the second model");

        let after_first = hidden_values(&first);
        let after_second = hidden_values(&second);
        assert_eq!(
            after_first, after_second,
            "after loading, both models must be the checkpoint's model"
        );
        assert_ne!(
            after_first, before_first,
            "loading must change the model's behaviour"
        );
    }

    #[test]
    fn load_pretrained_recovers_exact_pooler_values() {
        let config = loading_config();
        let spec = fixture_spec(&config, "");
        let tensors = spec.tensors();
        let expected_weight = tensors
            .iter()
            .find(|t| t.name == "pooler.dense.weight")
            .expect("fixture must contain the pooler weight")
            .values
            .clone();
        let bytes = build_safetensors(&tensors);

        let mut model = BertModel::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("checkpoint must load");
        assert!(report.is_complete(), "report must show a complete load");
        assert!(report.unexpected.is_empty());

        let pooler = model.pooler.as_ref().expect("pooler must survive the load");
        match pooler.dense().weight() {
            Tensor::F32(arr) => {
                assert_eq!(arr.shape(), &[8, 8]);
                assert_eq!(arr.iter().copied().collect::<Vec<f32>>(), expected_weight);
            },
            other => panic!("expected an F32 weight, got {other:?}"),
        }
    }

    #[test]
    fn load_pretrained_reports_every_missing_tensor_instead_of_inventing_it() {
        let config = loading_config();
        let mut tensors = fixture_spec(&config, "").tensors();
        tensors.retain(|t| !t.name.contains("encoder.layer.1.attention.self.key"));
        let bytes = build_safetensors(&tensors);

        let mut model = BertModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an incomplete checkpoint must not load silently");
        let message = err.to_string();
        assert!(message.contains("missing"), "unexpected error: {message}");
        assert!(
            message.contains("encoder.layer.1.attention.self.key.weight"),
            "the error must name the missing parameter: {message}"
        );
    }

    #[test]
    fn load_pretrained_rejects_a_checkpoint_with_foreign_tensors() {
        let config = loading_config();
        let mut tensors = fixture_spec(&config, "").tensors();
        tensors.push(F32Tensor::ramp(
            "encoder.layer.9.mystery.weight",
            &[8, 8],
            42.0,
        ));
        let bytes = build_safetensors(&tensors);

        let mut model = BertModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an unrecognised tensor must fail the load");
        assert!(
            err.to_string().contains("encoder.layer.9.mystery.weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_pretrained_accepts_a_task_checkpoint_with_the_bert_prefix_and_a_head() {
        let config = loading_config();
        let mut tensors = fixture_spec(&config, "bert.").tensors();
        // A `BertForPreTraining` checkpoint carries the MLM head alongside the
        // encoder; those tensors are reported, not treated as a mismatch.
        tensors.push(F32Tensor::ramp("cls.predictions.bias", &[16], 7.0));
        let bytes = build_safetensors(&tensors);

        let mut model = BertModel::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a prefixed checkpoint must load");
        assert!(report.is_complete());
        assert_eq!(report.ignored, vec!["cls.predictions.bias".to_string()]);
    }

    #[test]
    fn load_pretrained_rejects_bytes_that_are_not_a_checkpoint() {
        // The old loader scanned arbitrary bytes for "reasonable-looking" floats
        // and, failing that, filled the model with `fastrand` values while
        // returning Ok. Garbage in must now mean an error out.
        let config = loading_config();
        let mut model = BertModel::new(config).expect("model must build");
        let garbage = vec![0xAB_u8; 4096];
        let err = model
            .load_pretrained(&mut garbage.as_slice())
            .expect_err("garbage must not be accepted as weights");
        assert!(
            err.to_string().contains("unrecognised checkpoint container"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_pretrained_rejects_a_checkpoint_for_a_different_configuration() {
        let config = loading_config();
        let wider = BertConfig {
            hidden_size: 16,
            num_attention_heads: 2,
            intermediate_size: 32,
            ..config.clone()
        };
        let bytes = fixture_spec(&wider, "").safetensors();

        let mut model = BertModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a mismatched checkpoint must not be reshaped into place");
        assert!(
            err.to_string().contains("expects"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_pretrained_drops_the_pooler_when_the_checkpoint_has_none() {
        let config = loading_config();
        let mut spec = fixture_spec(&config, "");
        spec.include_pooler = false;
        let bytes = spec.safetensors();

        let mut model = BertModel::new(config).expect("model must build");
        model
            .load_pretrained(&mut bytes.as_slice())
            .expect("a pooler-less checkpoint must load");
        assert!(
            model.pooler.is_none(),
            "a randomly-initialised pooler must not survive a checkpoint that has none"
        );
    }
    #[test]
    fn task_head_weights_present_in_the_checkpoint_reach_the_model() {
        // Regression: the task wrappers delegated to `BertModel::load_pretrained`,
        // which loads only the encoder. A fine-tuned checkpoint's `classifier.*`
        // tensors landed in `report.ignored` and inference ran through a
        // randomly-initialised classifier while `load_pretrained` returned Ok.
        use crate::bert::tasks::BertForSequenceClassification;

        let config = loading_config();
        let num_labels = 3usize;
        let mut tensors = fixture_spec(&config, "bert.").tensors();
        let classifier_weight =
            F32Tensor::ramp("classifier.weight", &[num_labels, config.hidden_size], 50.0);
        let classifier_bias = F32Tensor::ramp("classifier.bias", &[num_labels], 90.0);
        tensors.push(classifier_weight.clone());
        tensors.push(classifier_bias.clone());
        let bytes = build_safetensors(&tensors);

        let mut model =
            BertForSequenceClassification::new(config, num_labels).expect("task model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a fine-tuned checkpoint must load");

        assert!(
            report.is_complete(),
            "a checkpoint carrying the head must produce a complete load, missing: {:?}",
            report.missing
        );
        assert!(
            report.loaded.contains(&"classifier.weight".to_string()),
            "the classifier weight must be reported as loaded: {:?}",
            report.loaded
        );
        assert!(
            !report.ignored.contains(&"classifier.weight".to_string()),
            "the classifier weight must not be reported as ignored"
        );

        match model.classifier_weight() {
            Tensor::F32(arr) => {
                assert_eq!(
                    arr.iter().copied().collect::<Vec<f32>>(),
                    classifier_weight.values
                );
            },
            other => panic!("expected an F32 classifier weight, got {other:?}"),
        }
    }

    #[test]
    fn a_backbone_only_checkpoint_reports_the_absent_task_head() {
        // Loading a pretrained backbone into a task model is legitimate, but the
        // head keeps its random initialisation and that must be visible.
        use crate::bert::tasks::BertForSequenceClassification;

        let config = loading_config();
        let bytes = fixture_spec(&config, "bert.").safetensors();
        let mut model =
            BertForSequenceClassification::new(config, 3).expect("task model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a backbone-only checkpoint must still load");
        assert!(
            !report.is_complete(),
            "an absent head must be reported, not silently accepted"
        );
        assert!(
            report.missing.contains(&"classifier.weight".to_string()),
            "missing head must be named: {:?}",
            report.missing
        );
    }
}
