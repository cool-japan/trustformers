use crate::llama3_2::config::{Llama32Config, Llama32Error};
use crate::llama3_2::model::Llama32VisionModel;
use trustformers_core::{errors::Result, layers::Linear, tensor::Tensor, traits::Layer};

// ─────────────────────────────────────────────────────────────────────────────
// Llama32ForConditionalGeneration
// ─────────────────────────────────────────────────────────────────────────────

/// Llama-3.2 conditional generation model.
///
/// Wraps `Llama32VisionModel` with a language-modelling head, supporting both
/// text-only and image-conditioned generation.
pub struct Llama32ForConditionalGeneration {
    model: Llama32VisionModel,
    lm_head: Linear,
}

impl Llama32ForConditionalGeneration {
    /// Create a new model with the given configuration.
    pub fn new(config: Llama32Config) -> Result<Self> {
        let lm_head = Linear::new_with_device(
            config.hidden_size,
            config.vocab_size,
            false,
            trustformers_core::device::Device::CPU,
        );
        let model = Llama32VisionModel::new(config)?;
        Ok(Self { model, lm_head })
    }

    /// Access the underlying configuration.
    pub fn config(&self) -> &Llama32Config {
        self.model.config()
    }

    /// Total number of learnable parameters.
    pub fn parameter_count(&self) -> usize {
        self.model.parameter_count() + self.lm_head.parameter_count()
    }

    /// Encode pixel values into vision features.
    ///
    /// Wraps `Llama32VisionModel::encode_image` with typed error conversion.
    pub fn encode_image(
        &self,
        pixel_values: &[f32],
        height: usize,
        width: usize,
    ) -> std::result::Result<Vec<f32>, Llama32Error> {
        let expected = height * width * 3;
        if pixel_values.len() != expected {
            return Err(Llama32Error::PixelBufferSize {
                expected,
                got: pixel_values.len(),
            });
        }
        let tensor = self
            .model
            .encode_image(pixel_values, height, width)
            .map_err(|e| Llama32Error::TensorOp(e.to_string()))?;
        match tensor {
            Tensor::F32(arr) => Ok(arr.iter().copied().collect()),
            _ => Err(Llama32Error::TensorOp(
                "unexpected tensor dtype in vision encoder output".to_string(),
            )),
        }
    }

    /// Full multimodal forward pass, returning logits.
    ///
    /// * `input_ids`    — text token IDs
    /// * `pixel_values` — flat `[height * width * 3]` float buffer
    /// * `height`, `width` — image dimensions in pixels
    ///
    /// Returns logits of shape `[seq_len, vocab_size]`.
    pub fn forward_multimodal(
        &self,
        input_ids: Vec<u32>,
        pixel_values: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let hidden = self.model.forward_multimodal(input_ids, pixel_values, height, width)?;
        self.lm_head.forward(hidden)
    }

    /// Text-only forward pass, returning logits.
    pub fn forward_text_only(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let hidden = self.model.forward_text_only(input_ids)?;
        self.lm_head.forward(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama3_2::config::Llama32Config;

    fn small_config() -> Llama32Config {
        Llama32Config::small_test()
    }

    // ── 1. Llama32ForConditionalGeneration constructs ─────────────────────────

    #[test]
    fn test_construction_succeeds() {
        let result = Llama32ForConditionalGeneration::new(small_config());
        assert!(
            result.is_ok(),
            "Llama32ForConditionalGeneration must construct"
        );
    }

    // ── 2. config accessor returns correct vocab size ─────────────────────────

    #[test]
    fn test_config_accessor_vocab_size() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model =
            Llama32ForConditionalGeneration::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        assert_eq!(model.config().vocab_size, vocab);
    }

    // ── 3. parameter_count > 0 ───────────────────────────────────────────────

    #[test]
    fn test_parameter_count_nonzero() {
        let model = Llama32ForConditionalGeneration::new(small_config())
            .unwrap_or_else(|_| panic!("init failed"));
        assert!(model.parameter_count() > 0, "parameter_count must be > 0");
    }

    // ── 4. text-only forward succeeds ────────────────────────────────────────

    #[test]
    fn test_text_only_forward_succeeds() {
        let model = Llama32ForConditionalGeneration::new(small_config())
            .unwrap_or_else(|_| panic!("init failed"));
        let result = model.forward_text_only(vec![1u32, 2, 3]);
        assert!(result.is_ok(), "text-only forward must succeed");
    }

    // ── 5. text-only forward output is non-empty ──────────────────────────────

    #[test]
    fn test_text_only_forward_nonempty() {
        let model = Llama32ForConditionalGeneration::new(small_config())
            .unwrap_or_else(|_| panic!("init failed"));
        if let Ok(Tensor::F32(arr)) = model.forward_text_only(vec![0u32]).as_ref() {
            assert!(!arr.is_empty(), "output must be non-empty");
        }
    }

    // ── 6. text-only forward output is finite ────────────────────────────────

    #[test]
    fn test_text_only_forward_finite() {
        let model = Llama32ForConditionalGeneration::new(small_config())
            .unwrap_or_else(|_| panic!("init failed"));
        if let Ok(Tensor::F32(arr)) = model.forward_text_only(vec![1u32, 2]).as_ref() {
            for &v in arr.iter() {
                assert!(v.is_finite(), "logit {v} must be finite");
            }
        }
    }

    // ── 7. encode_image with correct pixel buffer size succeeds ──────────────

    #[test]
    fn test_encode_image_correct_size() {
        let model = Llama32ForConditionalGeneration::new(small_config())
            .unwrap_or_else(|_| panic!("init failed"));
        let h = 4;
        let w = 4;
        let pixels = vec![0.5f32; h * w * 3];
        let result = model.encode_image(&pixels, h, w);
        assert!(
            result.is_ok(),
            "encode_image with correct size must succeed"
        );
    }

    // ── 8. encode_image with wrong size returns error ─────────────────────────

    #[test]
    fn test_encode_image_wrong_size_error() {
        let model = Llama32ForConditionalGeneration::new(small_config())
            .unwrap_or_else(|_| panic!("init failed"));
        let wrong_pixels = vec![0.5f32; 10]; // wrong size
        let err = model.encode_image(&wrong_pixels, 4, 4);
        assert!(err.is_err(), "wrong pixel buffer size must return error");
    }

    // ── 9. Llama32Error display is non-empty ──────────────────────────────────

    #[test]
    fn test_error_invalid_config_display() {
        let msg = format!("{}", Llama32Error::InvalidConfig("bad".to_string()));
        assert!(!msg.is_empty(), "error display must be non-empty");
        assert!(msg.contains("bad"), "must mention error detail");
    }

    // ── 10. Llama32Error pixel buffer size display ────────────────────────────

    #[test]
    fn test_error_pixel_buffer_size_display() {
        let msg = format!(
            "{}",
            Llama32Error::PixelBufferSize {
                expected: 48,
                got: 12
            }
        );
        assert!(msg.contains("48"), "must mention expected");
        assert!(msg.contains("12"), "must mention actual");
    }

    // ── 11. small_test config has vision fields ───────────────────────────────

    #[test]
    fn test_small_test_config_has_vision_fields() {
        let cfg = small_config();
        assert!(cfg.vision_hidden_size > 0, "vision_hidden_size must be > 0");
        assert!(cfg.image_size > 0, "image_size must be > 0");
    }

    // ── 12. Llama32Config::num_patches ───────────────────────────────────────

    #[test]
    fn test_num_patches_calculation() {
        let patches = Llama32Config::num_patches(224, 14);
        assert_eq!(patches, 256, "224/14 rounded = 16, 16*16 = 256");
    }

    // ── 13. Llama32Config::default_cross_attention_layers ────────────────────

    #[test]
    fn test_default_cross_attention_layers() {
        let layers = Llama32Config::default_cross_attention_layers(8);
        assert!(
            !layers.is_empty(),
            "must return at least one cross-attention layer"
        );
        for &l in &layers {
            assert!(l < 8, "layer index {l} must be < num_layers 8");
        }
    }

    // ── 14. Llama32Error::VisionShapeMismatch display ─────────────────────────

    #[test]
    fn test_error_vision_shape_mismatch_display() {
        let msg = format!(
            "{}",
            Llama32Error::VisionShapeMismatch {
                expected: 512,
                got: 256
            }
        );
        assert!(msg.contains("512"), "must contain expected");
        assert!(msg.contains("256"), "must contain actual");
    }

    // ── 15. encode_image output is non-empty ─────────────────────────────────

    #[test]
    fn test_encode_image_output_nonempty() {
        let model = Llama32ForConditionalGeneration::new(small_config())
            .unwrap_or_else(|_| panic!("init failed"));
        let h = 28;
        let w = 28;
        let pixels = vec![0.1f32; h * w * 3];
        if let Ok(features) = model.encode_image(&pixels, h, w) {
            assert!(!features.is_empty(), "vision features must be non-empty");
        }
    }

    // ── 16. text-only forward is deterministic ────────────────────────────────

    #[test]
    fn test_text_only_forward_deterministic() {
        let model = Llama32ForConditionalGeneration::new(small_config())
            .unwrap_or_else(|_| panic!("init failed"));
        let ids = vec![1u32, 2, 3];
        let r1 = model.forward_text_only(ids.clone());
        let r2 = model.forward_text_only(ids);
        if let (Ok(a), Ok(b)) = (r1, r2) {
            if let (Tensor::F32(arr_a), Tensor::F32(arr_b)) = (&a, &b) {
                let v1: Vec<f32> = arr_a.iter().copied().collect();
                let v2: Vec<f32> = arr_b.iter().copied().collect();
                assert_eq!(v1, v2, "forward must be deterministic");
            }
        }
    }

    // ── 17. Llama32Error::TensorOp display ───────────────────────────────────

    #[test]
    fn test_error_tensor_op_display() {
        let msg = format!("{}", Llama32Error::TensorOp("matmul failed".to_string()));
        assert!(
            msg.contains("matmul"),
            "TensorOp display must include message"
        );
    }

    // ── 18. Llama32Error::NotImplemented display ──────────────────────────────

    #[test]
    fn test_error_not_implemented_display() {
        let msg = format!(
            "{}",
            Llama32Error::NotImplemented("beam search".to_string())
        );
        assert!(
            msg.contains("beam search"),
            "NotImplemented must include feature name"
        );
    }
}
