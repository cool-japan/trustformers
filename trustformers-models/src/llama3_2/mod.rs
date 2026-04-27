//! # Llama-3.2 Vision-Language Models
//!
//! Llama-3.2 introduced multimodal variants combining a ViT-style vision
//! encoder with the Llama-3 text backbone.  Cross-attention layers injected
//! at every 4th decoder layer allow the text decoder to attend to visual
//! patch features extracted by the vision encoder.
//!
//! ## Variants
//!
//! | Variant | Text Hidden | Q-heads | KV-heads | Vision Hidden | Layers |
//! |---------|------------|---------|----------|---------------|--------|
//! | 3B      | 3072       | 24      | 8        | 1280          | 28     |
//! | 11B     | 4096       | 32      | 8        | 1280          | 32     |
//!
//! Both variants share the same ViT-H vision encoder (1280-dim, 32 layers,
//! 16 attention heads) and use LongRoPE scaling (factor=32) to support
//! up to 131 072 text tokens.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::llama3_2::{Llama32Config, Llama32ForConditionalGeneration};
//!
//! let config = Llama32Config::default(); // 3B
//! let model = Llama32ForConditionalGeneration::new(config)?;
//!
//! // Text-only inference
//! let logits = model.forward_text_only(vec![1u32, 2, 3])?;
//! # Ok::<(), trustformers_core::errors::TrustformersError>(())
//! ```

pub mod config;
pub mod model;
pub mod tasks;

pub use config::{Llama32Config, Llama32Error};
pub use model::{
    CrossAttentionLayer, Llama32CrossAttentionDecoder, Llama32MLP, Llama32RmsNorm,
    Llama32SelfAttention, Llama32VisionModel, VisionAttention, VisionEncoder, VisionEncoderLayer,
    VisionLayerNorm, VisionMLP, VisionPatchEmbedding,
};
pub use tasks::Llama32ForConditionalGeneration;

#[cfg(test)]
mod tests {
    use super::config::Llama32Config;
    use super::model::{VisionEncoder, VisionPatchEmbedding};
    use trustformers_core::traits::Config;

    // ── Test 1: Config defaults (3B) ──────────────────────────────────────────

    #[test]
    fn test_llama32_config_defaults_3b() {
        let cfg = Llama32Config::default();
        assert_eq!(cfg.vocab_size, 128256);
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.max_position_embeddings, 131072);
        assert!((cfg.rope_theta - 500000.0_f64).abs() < 1e-3);
        assert!(cfg.use_scaled_rope);
        assert!((cfg.rope_scaling_factor - 32.0_f32).abs() < 1e-3);
    }

    // ── Test 2: num_patches calculation ───────────────────────────────────────

    #[test]
    fn test_llama32_num_patches_calculation() {
        // Default 3B: image_size=560, patch_size=14 → (560/14)² = 40² = 1600
        assert_eq!(Llama32Config::num_patches(560, 14), 1600);
        // Small test config
        let cfg = Llama32Config::small_test();
        let expected = (cfg.image_size / cfg.patch_size).pow(2);
        assert_eq!(cfg.num_patches, expected);
    }

    // ── Test 3: default_cross_attention_layers generation ────────────────────

    #[test]
    fn test_llama32_default_cross_attention_layers() {
        let layers = Llama32Config::default_cross_attention_layers(28);
        // Every 4th layer: indices 3, 7, 11, 15, 19, 23, 27
        assert_eq!(layers, vec![3, 7, 11, 15, 19, 23, 27]);

        let layers4 = Llama32Config::default_cross_attention_layers(4);
        assert_eq!(layers4, vec![3]);
    }

    // ── Test 4: vision hidden size ────────────────────────────────────────────

    #[test]
    fn test_llama32_vision_hidden_size() {
        let cfg = Llama32Config::default();
        assert_eq!(cfg.vision_hidden_size, 1280);
        assert_eq!(cfg.vision_num_attention_heads, 16);
        assert_eq!(cfg.vision_num_hidden_layers, 32);
        assert_eq!(cfg.vision_intermediate_size, 5120);
    }

    // ── Test 5: 11B config ────────────────────────────────────────────────────

    #[test]
    fn test_llama32_11b_config() {
        let cfg = Llama32Config::llama32_11b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.num_hidden_layers, 32);
        // vision output dim = 6 * 4096 = 24576
        assert_eq!(cfg.vision_output_dim, 6 * 4096);
    }

    // ── Test 6: validate passes for well-formed configs ───────────────────────

    #[test]
    fn test_llama32_validate_ok() {
        let cfg = Llama32Config::small_test();
        assert!(cfg.validate().is_ok());

        let cfg3b = Llama32Config::default();
        assert!(cfg3b.validate().is_ok());

        let cfg11b = Llama32Config::llama32_11b();
        assert!(cfg11b.validate().is_ok());
    }

    // ── Test 7: patch embedding output shape ──────────────────────────────────

    #[test]
    fn test_llama32_patch_embedding_shape() {
        let cfg = Llama32Config::small_test();
        // image_size=28, patch_size=14 → 2×2 = 4 patches
        let patch_emb = VisionPatchEmbedding::new(&cfg).expect("VisionPatchEmbedding::new");
        let height = cfg.image_size;
        let width = cfg.image_size;
        let pixel_values = vec![0.5_f32; height * width * 3];

        let output = patch_emb.embed_patches(&pixel_values, height, width).expect("embed_patches");

        let shape = output.shape();
        // Output shape should be [num_patches, vision_hidden_size]
        let total_elems: usize = shape.iter().product();
        assert_eq!(total_elems, cfg.num_patches * cfg.vision_hidden_size);
    }

    // ── Test 8: vision encoder output size ────────────────────────────────────

    #[test]
    fn test_llama32_vision_encoder_output_size() {
        let cfg = Llama32Config::small_test();
        let encoder = VisionEncoder::new(&cfg).expect("VisionEncoder::new");
        let height = cfg.image_size;
        let width = cfg.image_size;
        let pixel_values = vec![0.1_f32; height * width * 3];

        let output = encoder.encode(&pixel_values, height, width).expect("encode");

        let total_elems: usize = output.shape().iter().product();
        // [num_patches, vision_hidden_size]
        assert_eq!(total_elems, cfg.num_patches * cfg.vision_hidden_size);
    }

    // ── Test 9: Llama32Error display variants ─────────────────────────────────

    #[test]
    fn test_llama32_error_display_invalid_config() {
        let err = super::config::Llama32Error::InvalidConfig("bad value".to_string());
        let s = err.to_string();
        assert!(
            s.contains("invalid config") || s.contains("Llama32"),
            "got: {s}"
        );
        assert!(s.contains("bad value"), "got: {s}");
    }

    #[test]
    fn test_llama32_error_display_vision_shape_mismatch() {
        let err = super::config::Llama32Error::VisionShapeMismatch {
            expected: 1600,
            got: 400,
        };
        let s = err.to_string();
        assert!(s.contains("1600") && s.contains("400"), "got: {s}");
        assert!(
            s.to_lowercase().contains("mismatch") || s.to_lowercase().contains("vision"),
            "got: {s}"
        );
    }

    #[test]
    fn test_llama32_error_display_pixel_buffer_size() {
        let err = super::config::Llama32Error::PixelBufferSize {
            expected: 1000,
            got: 500,
        };
        let s = err.to_string();
        assert!(s.contains("1000") && s.contains("500"), "got: {s}");
    }

    #[test]
    fn test_llama32_error_display_cross_attention_oob() {
        let err = super::config::Llama32Error::CrossAttentionIndexOutOfRange {
            index: 30,
            num_layers: 28,
        };
        let s = err.to_string();
        assert!(s.contains("30") && s.contains("28"), "got: {s}");
    }

    // ── Test 13: Config clone/debug ───────────────────────────────────────────

    #[test]
    fn test_llama32_config_clone() {
        let cfg = Llama32Config::small_test();
        let cloned = cfg.clone();
        assert_eq!(cloned.hidden_size, cfg.hidden_size);
        assert_eq!(cloned.vocab_size, cfg.vocab_size);
        assert_eq!(cloned.cross_attention_layers, cfg.cross_attention_layers);
    }

    #[test]
    fn test_llama32_config_debug() {
        let cfg = Llama32Config::small_test();
        let s = format!("{:?}", cfg);
        assert!(s.contains("Llama32Config"), "got: {s}");
        assert!(s.contains("vocab_size"), "got: {s}");
        assert!(s.contains("vision_hidden_size"), "got: {s}");
    }

    // ── Test 15: validate fails when hidden_size not divisible by num_heads ───

    #[test]
    fn test_llama32_validate_fails_bad_hidden_divisibility() {
        let mut cfg = Llama32Config::small_test();
        // hidden=64, num_attention_heads=5 → not divisible
        cfg.num_attention_heads = 5;
        cfg.head_dim = 64 / 5; // rounded
        let result = cfg.validate();
        assert!(
            result.is_err(),
            "hidden_size not divisible by num_attention_heads should fail"
        );
    }

    // ── Test 16: validate fails when cross-attention layer index is OOB ───────

    #[test]
    fn test_llama32_validate_fails_cross_attention_oob() {
        let mut cfg = Llama32Config::small_test();
        // Add an out-of-range index
        cfg.cross_attention_layers.push(cfg.num_hidden_layers + 5);
        let result = cfg.validate();
        assert!(
            result.is_err(),
            "OOB cross-attention layer index should fail validation"
        );
    }

    // ── Test 17: 3B rope_scaling_factor is 32 ────────────────────────────────

    #[test]
    fn test_llama32_3b_rope_scaling_factor() {
        let cfg = Llama32Config::llama32_3b();
        assert!(
            (cfg.rope_scaling_factor - 32.0_f32).abs() < 1e-3,
            "3B rope_scaling_factor must be 32, got {}",
            cfg.rope_scaling_factor
        );
        assert!(cfg.use_scaled_rope, "3B must use scaled rope (LongRoPE)");
    }

    // ── Test 18: vision_output_dim = 6 × hidden_size for both variants ────────

    #[test]
    fn test_llama32_vision_output_dim_formula() {
        let cfg3b = Llama32Config::llama32_3b();
        assert_eq!(
            cfg3b.vision_output_dim,
            6 * cfg3b.hidden_size,
            "3B vision_output_dim must be 6 × hidden_size"
        );
        let cfg11b = Llama32Config::llama32_11b();
        assert_eq!(
            cfg11b.vision_output_dim,
            6 * cfg11b.hidden_size,
            "11B vision_output_dim must be 6 × hidden_size"
        );
    }

    // ── Test 19: num_global_layers field ─────────────────────────────────────

    #[test]
    fn test_llama32_num_global_layers() {
        let cfg = Llama32Config::default();
        assert_eq!(
            cfg.num_global_layers, 8,
            "default num_global_layers must be 8"
        );
        let small = Llama32Config::small_test();
        assert_eq!(small.num_global_layers, 1);
    }

    // ── Test 20: text-only forward logit shape ────────────────────────────────

    #[test]
    fn test_llama32_text_only_forward_shape() {
        use super::tasks::Llama32ForConditionalGeneration;
        let cfg = Llama32Config::small_test();
        let vocab_size = cfg.vocab_size;
        let model = Llama32ForConditionalGeneration::new(cfg).expect("model creation");
        let logits = model.forward_text_only(vec![1u32, 2, 3]).expect("forward_text_only");
        let total_elems: usize = logits.shape().iter().product();
        // [seq_len=3, vocab_size]
        assert_eq!(total_elems, 3 * vocab_size);
    }

    // ── Test 21: encode_image returns correctly-sized vision features ─────────

    #[test]
    fn test_llama32_encode_image_output_size() {
        use super::tasks::Llama32ForConditionalGeneration;
        let cfg = Llama32Config::small_test();
        let h = cfg.image_size;
        let w = cfg.image_size;
        let expected_patches = cfg.num_patches;
        let expected_vision_hidden = cfg.vision_hidden_size;
        let pixels = vec![0.3_f32; h * w * 3];
        let model = Llama32ForConditionalGeneration::new(cfg).expect("model creation");
        let features = model.encode_image(&pixels, h, w).expect("encode_image");
        // Features should be [num_patches × vision_hidden_size] elements
        assert_eq!(
            features.len(),
            expected_patches * expected_vision_hidden,
            "encode_image should return num_patches * vision_hidden_size floats"
        );
    }

    // ── Test 22: encode_image wrong buffer size error ─────────────────────────

    #[test]
    fn test_llama32_encode_image_wrong_buffer_size() {
        use super::tasks::Llama32ForConditionalGeneration;
        let cfg = Llama32Config::small_test();
        let model = Llama32ForConditionalGeneration::new(cfg.clone()).expect("model creation");
        // Provide a buffer that is too small
        let bad_pixels = vec![0.1_f32; 10];
        let result = model.encode_image(&bad_pixels, cfg.image_size, cfg.image_size);
        assert!(result.is_err(), "wrong buffer size should return Err");
    }

    // ── Test 23: cross_attention_layers count for 28-layer config ─────────────

    #[test]
    fn test_llama32_cross_attention_count_28_layers() {
        // Every 4th layer → indices 3,7,11,15,19,23,27 = 7 layers
        let layers = Llama32Config::default_cross_attention_layers(28);
        assert_eq!(layers.len(), 7);
    }

    // ── Test 24: cross_attention_layers count for 32-layer config ─────────────

    #[test]
    fn test_llama32_cross_attention_count_32_layers() {
        // Indices 3,7,11,15,19,23,27,31 = 8 layers
        let layers = Llama32Config::default_cross_attention_layers(32);
        assert_eq!(layers.len(), 8);
    }

    // ── Test 25: Llama32ForConditionalGeneration parameter_count > 0 ──────────

    #[test]
    fn test_llama32_conditional_gen_parameter_count() {
        use super::tasks::Llama32ForConditionalGeneration;
        let cfg = Llama32Config::small_test();
        let model = Llama32ForConditionalGeneration::new(cfg).expect("model creation");
        assert!(model.parameter_count() > 0, "model must have parameters");
    }

    // ── Test 26: architecture() returns "Llama-3.2" ───────────────────────────

    #[test]
    fn test_llama32_architecture_string() {
        use trustformers_core::traits::Config;
        let cfg = Llama32Config::default();
        assert_eq!(cfg.architecture(), "Llama-3.2");
    }

    // ── Test 27: small_test config vision num_patches=4 ──────────────────────

    #[test]
    fn test_llama32_small_test_num_patches() {
        let cfg = Llama32Config::small_test();
        // image_size=28, patch_size=14 → (28/14)² = 2² = 4
        assert_eq!(cfg.num_patches, 4, "small_test should have 4 patches");
    }
}
