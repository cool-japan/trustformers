//! Unit tests for [`super`] -- split out of `multimodal.rs` to keep it
//! under the workspace's 2000-line-per-file policy (see the `#[path =
//! "multimodal_tests.rs"] mod tests;` declaration at the bottom of that
//! file; the same convention is used by `pipeline/adaptive_inference.rs`,
//! `config_management.rs`, and `auto/optimizers/mod.rs`).

use super::*;

// ---- MultiModalConfig tests ----

#[test]
fn test_config_default_values() {
    let cfg = MultiModalConfig::default();
    assert_eq!(cfg.max_text_length, 512);
    assert_eq!(cfg.max_image_size, (224, 224));
    assert!((cfg.max_audio_duration - 30.0).abs() < 1e-6);
    assert!(cfg.normalize_inputs);
    assert!(cfg.cross_modal_attention);
    assert!((cfg.temperature - 1.0).abs() < 1e-6);
}

#[test]
fn test_config_clone() {
    let cfg = MultiModalConfig {
        max_text_length: 256,
        ..MultiModalConfig::default()
    };
    assert_eq!(cfg.clone().max_text_length, 256);
}

// ---- AttentionConfig tests ----

#[test]
fn test_attention_config_default() {
    let acfg = AttentionConfig::default();
    assert_eq!(acfg.num_heads, 8);
    assert_eq!(acfg.head_dim, 64);
    assert!((acfg.dropout - 0.1).abs() < 1e-6);
    assert!(acfg.use_relative_position);
    assert_eq!(acfg.max_relative_position, 128);
}

// ---- MultiModalInput tests ----

#[test]
fn test_input_text_only() {
    let input = MultiModalInput {
        text: Some("Hello world".to_string()),
        image: None,
        audio: None,
        video: None,
        metadata: HashMap::new(),
        modality_weights: None,
    };
    assert!(input.text.is_some());
    assert!(input.image.is_none());
}

#[test]
fn test_input_image_plus_text() {
    let input = MultiModalInput {
        text: Some("Describe this image".to_string()),
        image: Some(vec![0u8; 100]),
        audio: None,
        video: None,
        metadata: HashMap::new(),
        modality_weights: None,
    };
    assert!(input.text.is_some());
    assert!(input.image.is_some());
}

#[test]
fn test_input_multimodality_flags() {
    let input = MultiModalInput {
        text: Some("text".to_string()),
        image: Some(vec![1, 2, 3]),
        audio: Some(vec![4, 5, 6]),
        video: None,
        metadata: HashMap::new(),
        modality_weights: None,
    };
    let mut modalities = Vec::new();
    if input.text.is_some() {
        modalities.push("text");
    }
    if input.image.is_some() {
        modalities.push("image");
    }
    if input.audio.is_some() {
        modalities.push("audio");
    }
    if input.video.is_some() {
        modalities.push("video");
    }
    assert_eq!(modalities.len(), 3);
}

// -------------------------------------------------------------------
// TextProcessor: regression coverage for the bug where
// `TextProcessor::process` returned `sin((i*768+j) as f32) * 0.1` --
// entirely a function of position, never of the actual word. These
// tests fail against that old behavior because they assert real
// content-dependence.
// -------------------------------------------------------------------

#[test]
fn test_text_processor_produces_features() {
    let processor = TextProcessor::new();
    let cfg = MultiModalConfig::default();
    let features = processor.process("Hello world test", &cfg).expect("text processing succeeded");
    // 3 words -> 3 real per-word feature vectors.
    assert_eq!(features.len(), 3);
    assert_eq!(features[0].len(), COMMON_FEATURE_DIM);
}

#[test]
fn test_text_processor_respects_max_length() {
    let processor = TextProcessor::new();
    let cfg = MultiModalConfig {
        max_text_length: 2,
        ..MultiModalConfig::default()
    };
    let text = "one two three four five";
    let features = processor.process(text, &cfg).expect("text processing succeeded");
    assert_eq!(features.len(), 2);
}

#[test]
fn test_text_processor_empty_text() {
    let processor = TextProcessor::new();
    let cfg = MultiModalConfig::default();
    let features = processor.process("", &cfg).expect("empty text processing succeeded");
    assert!(features.is_empty());
}

#[test]
fn test_text_processor_same_word_gives_identical_vector() {
    // Real, deterministic content-derivation: the same word must
    // always hash to the same vector.
    let processor = TextProcessor::new();
    let cfg = MultiModalConfig::default();
    let features = processor.process("repeat repeat", &cfg).expect("ok");
    assert_eq!(features[0], features[1]);
}

#[test]
fn test_text_processor_different_words_give_different_vectors() {
    // Regression: the old sine-wave placeholder differed only by
    // *position*, so two different first words at the same position
    // across two calls would be identical -- this asserts real
    // content-dependence instead.
    let processor = TextProcessor::new();
    let cfg = MultiModalConfig::default();
    let a = processor.process("apple", &cfg).expect("ok");
    let b = processor.process("zebra", &cfg).expect("ok");
    assert_ne!(
        a[0], b[0],
        "different words at the same position must produce different feature vectors"
    );
}

#[test]
fn test_text_processor_vectors_are_l2_normalized() {
    let processor = TextProcessor::new();
    let cfg = MultiModalConfig::default();
    let features = processor.process("hello", &cfg).expect("ok");
    let norm: f32 = features[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected a unit-norm vector, got norm {norm}"
    );
}

// -------------------------------------------------------------------
// ImageProcessor: regression coverage for the bug where
// `ImageProcessor::process` returned `cos((i*768+j) as f32) * 0.1` for
// a patch grid derived purely from `config.max_image_size`, entirely
// ignoring the `_image: &[u8]` bytes (the parameter was even
// underscore-prefixed). The real path decodes the bytes and -- absent
// an attached vision encoder -- honestly reports
// `FeatureUnavailable` rather than a placeholder.
// -------------------------------------------------------------------

#[test]
fn test_image_processor_propagates_feature_unavailable_without_encoder() {
    let processor = ImageProcessor::new();
    let cfg = MultiModalConfig::default();
    // A tiny real, decodable Netpbm (P5, grayscale) image: header +
    // 2x2 8-bit pixels. Decoding must succeed; only the (nonexistent)
    // encoder step must fail.
    let ppm = b"P5\n2 2\n255\n\x00\x40\x80\xff".to_vec();
    let result = processor.process(&ppm, &cfg);
    match result {
        Err(TrustformersError::FeatureUnavailable { ref feature, .. }) => {
            assert!(feature.contains("vision"), "feature: {feature}");
        },
        other => panic!(
            "expected a structured FeatureUnavailable (real decode, no encoder attached), \
             got {other:?}"
        ),
    }
}

#[test]
fn test_image_processor_rejects_corrupt_bytes_before_claiming_success() {
    let processor = ImageProcessor::new();
    let cfg = MultiModalConfig::default();
    // Not a valid image container of any kind, and not empty either.
    let garbage = vec![1u8, 2, 3, 4, 5];
    let result = processor.process(&garbage, &cfg);
    assert!(
        result.is_err(),
        "undecodable bytes must error, never produce a fabricated vector"
    );
}

// -------------------------------------------------------------------
// AudioProcessor: regression coverage for the bug where
// `AudioProcessor::process` computed a frame count purely from
// `config.max_audio_duration` and filled every frame with
// `sin((i*128+j) as f32) * 0.2`, entirely ignoring the `_audio: &[u8]`
// bytes.
// -------------------------------------------------------------------

fn make_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    audio_dsp::encode_wav_pcm16(samples, sample_rate)
}

#[test]
fn test_audio_processor_rejects_non_wav_bytes() {
    let processor = AudioProcessor::new();
    let cfg = MultiModalConfig::default();
    let not_wav = vec![0u8; 64];
    let result = processor.process(&not_wav, &cfg);
    assert!(
        result.is_err(),
        "non-WAV bytes must be a structured error, not silent zeros"
    );
}

#[test]
fn test_audio_processor_produces_real_frames_from_real_wav() {
    let processor = AudioProcessor::new();
    let cfg = MultiModalConfig {
        max_audio_duration: 1.0,
        ..MultiModalConfig::default()
    };
    // 1 second of a real 440 Hz sine tone at 16 kHz -- genuine signal,
    // not silence, so the resulting spectral features are non-trivial.
    let sample_rate = 16000u32;
    let samples: Vec<f32> = (0..sample_rate)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect();
    let wav = make_wav(&samples, sample_rate);

    let features = processor.process(&wav, &cfg).expect("real WAV must process");
    assert!(
        !features.is_empty(),
        "a real 1-second tone must yield at least one frame"
    );
    assert_eq!(features[0].len(), AUDIO_FEATURE_DIM);
}

#[test]
fn test_audio_processor_silence_and_tone_produce_different_features() {
    // Regression: the old placeholder's output was a pure function of
    // frame/bin index, so silence and a real tone (same duration,
    // same sample rate) would have produced byte-identical "features".
    let processor = AudioProcessor::new();
    let cfg = MultiModalConfig {
        max_audio_duration: 0.5,
        ..MultiModalConfig::default()
    };
    let sample_rate = 16000u32;
    let n = sample_rate / 2;

    let silence = vec![0.0f32; n as usize];
    let tone: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect();

    let silence_features = processor
        .process(&make_wav(&silence, sample_rate), &cfg)
        .expect("silence must process");
    let tone_features = processor
        .process(&make_wav(&tone, sample_rate), &cfg)
        .expect("tone must process");

    assert_eq!(silence_features.len(), tone_features.len());
    assert_ne!(
        silence_features, tone_features,
        "real spectral features of silence and a real tone must differ"
    );
}

#[test]
fn test_audio_processor_respects_max_duration() {
    let processor = AudioProcessor::new();
    let short_cfg = MultiModalConfig {
        max_audio_duration: 0.25,
        ..MultiModalConfig::default()
    };
    let long_cfg = MultiModalConfig {
        max_audio_duration: 2.0,
        ..MultiModalConfig::default()
    };
    let sample_rate = 16000u32;
    let samples: Vec<f32> = (0..sample_rate * 2)
        .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin())
        .collect();
    let wav = make_wav(&samples, sample_rate);

    let short_features = processor.process(&wav, &short_cfg).expect("ok");
    let long_features = processor.process(&wav, &long_cfg).expect("ok");
    assert!(
        short_features.len() < long_features.len(),
        "a smaller max_audio_duration must truncate to fewer real frames: {} vs {}",
        short_features.len(),
        long_features.len()
    );
}

// -------------------------------------------------------------------
// VideoProcessor: regression coverage for the bug where
// `VideoProcessor::process` computed a frame count from
// `config.max_audio_duration` (not even a video-specific config
// field) and filled every frame with `cos((i*512+j) as f32) * 0.15`.
// No real video feature extraction path exists anywhere in this
// workspace, so every call must now fail structurally.
// -------------------------------------------------------------------

#[test]
fn test_video_processor_always_reports_unsupported_modality() {
    let processor = VideoProcessor::new();
    let cfg = MultiModalConfig::default();
    let result = processor.process(&[1, 2, 3, 4], &cfg);
    match result {
        Err(TrustformersError::FeatureUnavailable {
            ref feature,
            ref alternatives,
            ..
        }) => {
            assert!(feature.contains("video"), "feature: {feature}");
            assert!(
                alternatives.is_empty(),
                "no video backend exists to name as an alternative"
            );
        },
        other => panic!("expected a structured FeatureUnavailable for video, got {other:?}"),
    }
}

// ---- FusionLayer tests ----

#[test]
fn test_fusion_concatenation_non_empty() {
    let fusion = FusionLayer::new();
    let cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::Concatenation,
        ..MultiModalConfig::default()
    };
    let features = ModalityFeatures {
        text_features: Some(vec![vec![0.1; 768]; 3]),
        image_features: None,
        audio_features: None,
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };
    let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
    assert!(!fused.is_empty());
}

#[test]
fn test_fusion_addition_with_two_modalities() {
    let fusion = FusionLayer::new();
    let cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::Addition,
        ..MultiModalConfig::default()
    };
    let features = ModalityFeatures {
        text_features: Some(vec![vec![1.0; 768]]),
        image_features: Some(vec![vec![2.0; 768]]),
        audio_features: None,
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };
    let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
    assert_eq!(fused.len(), 1);
    // Average of 1.0 and 2.0 should be 1.5
    assert!(
        (fused[0][0] - 1.5).abs() < 1e-4,
        "expected 1.5, got {}",
        fused[0][0]
    );
}

#[test]
fn test_fusion_weighted_average() {
    let fusion = FusionLayer::new();
    let cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::WeightedAverage,
        ..MultiModalConfig::default()
    };
    let features = ModalityFeatures {
        text_features: Some(vec![vec![1.0; 768]]),
        image_features: Some(vec![vec![1.0; 768]]),
        audio_features: None,
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };
    let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
    assert!(!fused.is_empty());
}

#[test]
fn test_fusion_weighted_average_uses_audio_and_video_weights() {
    // Regression: `audio_weight`/`video_weight` used to be computed
    // and then never read -- only text/image were folded into the
    // weighted sum. This fails against that old behavior because a
    // pure-audio input (no text/image at all) would have produced no
    // fused output whatsoever.
    let fusion = FusionLayer::new();
    let cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::WeightedAverage,
        ..MultiModalConfig::default()
    };
    let features = ModalityFeatures {
        text_features: None,
        image_features: None,
        audio_features: Some(vec![vec![2.0; 768]]),
        video_features: Some(vec![vec![4.0; 768]]),
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };
    let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
    assert_eq!(
        fused.len(),
        1,
        "audio+video alone must still produce fused output"
    );
    // (2.0*0.3 + 4.0*0.2) / (0.3+0.2) = 1.4 / 0.5 = 2.8
    assert!((fused[0][0] - 2.8).abs() < 1e-4, "got {}", fused[0][0]);
}

// -------------------------------------------------------------------
// CrossAttention / GatedFusion / TransformerFusion: regression coverage
// for the bug where these three strategies were aliases for
// `Concatenation`/`WeightedAverage` -- `cross_attention_fusion` and
// `transformer_fusion` both called `self.concatenate_features(features)`
// verbatim, and `gated_fusion` called `self.weighted_average_features
// (features)` verbatim, despite each having a distinct real algorithm
// implemented elsewhere in this same file
// (`dot_product_attention_weights`, already used by
// `compute_cross_modal_attention`). Every test below fails against that
// old aliasing because it asserts an output shape or value the alias
// could never produce.
// -------------------------------------------------------------------

#[test]
fn test_cross_attention_fusion_differs_from_plain_concatenation() {
    // Old code: `cross_attention_fusion` === `concatenate_features`
    // exactly, so with two 768-wide modalities present the
    // concatenation alias would produce a 1536-wide row. Real
    // cross-attention output stays at `COMMON_FEATURE_DIM` (768) --
    // it is a weighted *combination*, not a concatenation.
    let fusion = FusionLayer::new();
    let cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::CrossAttention,
        ..MultiModalConfig::default()
    };
    let features = ModalityFeatures {
        text_features: Some(vec![vec![1.0; 768]]),
        image_features: Some(vec![vec![2.0; 768]]),
        audio_features: None,
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };
    let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
    assert_eq!(fused.len(), 1);
    assert_eq!(
        fused[0].len(),
        COMMON_FEATURE_DIM,
        "cross-attention output must stay at the common feature width, not concatenate to \
         {}",
        2 * COMMON_FEATURE_DIM
    );
}

#[test]
fn test_cross_attention_fusion_single_modality_returns_its_own_vector() {
    // Self-attention over a length-1 sequence has softmax weight
    // exactly 1.0 for the only element, so with a single modality
    // present the "attended" output must equal that modality's own
    // (uniform) feature vector -- there is nothing else to attend to.
    let fusion = FusionLayer::new();
    let cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::CrossAttention,
        ..MultiModalConfig::default()
    };
    let features = ModalityFeatures {
        text_features: Some(vec![vec![3.0; 768]]),
        image_features: None,
        audio_features: None,
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };
    let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
    assert_eq!(fused.len(), 1);
    assert!(
        fused[0].iter().all(|&v| (v - 3.0).abs() < 1e-3),
        "single-modality self-attention must return that modality's own vector: {:?}",
        &fused[0][..4]
    );
}

#[test]
fn test_cross_attention_fusion_differing_modalities_blend_between_them() {
    // Two *different* feature vectors: real cross-attention with
    // scaled dot-product scoring must produce a value strictly between
    // the two inputs (a weighted blend), not equal to either one alone
    // and not their simple unweighted concatenation/average by
    // construction (attention weights are data-dependent, not fixed
    // 50/50 -- see `dot_product_attention_weights`).
    let fusion = FusionLayer::new();
    let cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::CrossAttention,
        ..MultiModalConfig::default()
    };
    let mut text_row = vec![0.1f32; 768];
    text_row[0] = 5.0; // Give it a distinctive large component.
    let features = ModalityFeatures {
        text_features: Some(vec![text_row]),
        image_features: Some(vec![vec![0.1; 768]]),
        audio_features: None,
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };
    let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
    assert_eq!(fused.len(), 1);
    // Value must be finite and within the range spanned by the two
    // inputs at index 0 (0.1 for image, 5.0 for text) -- a genuine
    // weighted average property of softmax-weighted attention.
    assert!(fused[0][0].is_finite());
    assert!(
        (0.1..=5.0).contains(&fused[0][0]),
        "attended value at index 0 must lie within the span of its inputs: {}",
        fused[0][0]
    );
}

#[test]
fn test_gated_fusion_differs_from_fixed_weighted_average() {
    // Old code: `gated_fusion` === `weighted_average_features` exactly,
    // which uses fixed weights (text=0.4, image=0.6) regardless of
    // feature content. Real gated fusion's gate is
    // `sigmoid(mean(row))`, which for these inputs produces different
    // relative weighting than the fixed 0.4/0.6 split.
    let fusion = FusionLayer::new();
    let features = ModalityFeatures {
        text_features: Some(vec![vec![1.0; 768]]),
        image_features: Some(vec![vec![10.0; 768]]),
        audio_features: None,
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };

    let gated_cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::GatedFusion,
        ..MultiModalConfig::default()
    };
    let weighted_cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::WeightedAverage,
        ..MultiModalConfig::default()
    };

    let gated = fusion.fuse(&features, &gated_cfg).expect("gated fusion succeeded");
    let weighted = fusion
        .fuse(&features, &weighted_cfg)
        .expect("weighted average fusion succeeded");

    assert_eq!(gated.len(), 1);
    assert_eq!(weighted.len(), 1);
    assert!(
        (gated[0][0] - weighted[0][0]).abs() > 1e-3,
        "gated_fusion must diverge from the fixed-weight WeightedAverage strategy it used \
         to alias: gated={}, weighted={}",
        gated[0][0],
        weighted[0][0]
    );
}

#[test]
fn test_gated_fusion_gate_favours_the_higher_magnitude_modality() {
    // sigmoid(mean(row)) is monotonically increasing in the row's mean
    // activation, so a modality with a much larger mean activation
    // must receive a larger gate (closer to 1.0) than one near zero
    // (gate near 0.5), and therefore dominate the fused output more
    // than a naive unweighted average would.
    let fusion = FusionLayer::new();
    let cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::GatedFusion,
        ..MultiModalConfig::default()
    };
    let features = ModalityFeatures {
        text_features: Some(vec![vec![0.0; 768]]), // mean 0 -> gate 0.5
        image_features: Some(vec![vec![10.0; 768]]), // mean 10 -> gate ~1.0
        audio_features: None,
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };
    let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
    assert_eq!(fused.len(), 1);
    // Unweighted average would be 5.0; the high-gate image modality
    // must pull the result well above the unweighted midpoint.
    assert!(
        fused[0][0] > 6.0,
        "the higher-activation modality's larger gate must pull the fused value above the \
         unweighted average of 5.0: got {}",
        fused[0][0]
    );
}

#[test]
fn test_gated_fusion_single_modality_returns_its_own_vector() {
    // With only one modality present, its gate cancels out in the
    // normalization (contribution / total_gate == that modality's own
    // row), so the output must equal that modality's own vector.
    let fusion = FusionLayer::new();
    let cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::GatedFusion,
        ..MultiModalConfig::default()
    };
    let features = ModalityFeatures {
        text_features: Some(vec![vec![7.0; 768]]),
        image_features: None,
        audio_features: None,
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };
    let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
    assert_eq!(fused.len(), 1);
    assert!(
        fused[0].iter().all(|&v| (v - 7.0).abs() < 1e-3),
        "single-modality gated fusion must return that modality's own vector: {:?}",
        &fused[0][..4]
    );
}

#[test]
fn test_transformer_fusion_differs_from_plain_concatenation() {
    // Old code: `transformer_fusion` === `concatenate_features`
    // exactly (a 1536-wide row for two 768-wide modalities). Real
    // self-attention-with-residual output stays at COMMON_FEATURE_DIM.
    let fusion = FusionLayer::new();
    let cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::TransformerFusion,
        ..MultiModalConfig::default()
    };
    let features = ModalityFeatures {
        text_features: Some(vec![vec![1.0; 768]]),
        image_features: Some(vec![vec![2.0; 768]]),
        audio_features: None,
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };
    let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
    assert_eq!(fused.len(), 1);
    assert_eq!(
        fused[0].len(),
        COMMON_FEATURE_DIM,
        "transformer fusion output must stay at the common feature width, not concatenate"
    );
}

#[test]
fn test_transformer_fusion_residual_makes_it_differ_from_cross_attention() {
    // transformer_fusion adds the residual connection
    // (x + Attention(x)) that cross_attention_fusion does not, so for
    // the same input the two strategies must produce different
    // (residual-shifted) output magnitudes -- proving `transformer_fusion`
    // has its own real logic distinct from `cross_attention_fusion`,
    // not merely a second alias for the same underlying computation.
    let fusion = FusionLayer::new();
    let features = ModalityFeatures {
        text_features: Some(vec![vec![1.0; 768]]),
        image_features: Some(vec![vec![2.0; 768]]),
        audio_features: None,
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };

    let cross_attn_cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::CrossAttention,
        ..MultiModalConfig::default()
    };
    let transformer_cfg = MultiModalConfig {
        fusion_strategy: FusionStrategy::TransformerFusion,
        ..MultiModalConfig::default()
    };

    let cross_attn = fusion.fuse(&features, &cross_attn_cfg).expect("fusion succeeded");
    let transformer = fusion.fuse(&features, &transformer_cfg).expect("fusion succeeded");

    assert!(
        (cross_attn[0][0] - transformer[0][0]).abs() > 1e-3,
        "the residual connection must make transformer_fusion's output differ from \
         cross_attention_fusion's: cross_attn={}, transformer={}",
        cross_attn[0][0],
        transformer[0][0]
    );
}

#[test]
fn test_all_three_new_fusion_strategies_produce_finite_output() {
    let fusion = FusionLayer::new();
    let features = ModalityFeatures {
        text_features: Some(vec![vec![0.3; 768], vec![-0.2; 768]]),
        image_features: Some(vec![vec![1.1; 768]]),
        audio_features: Some(vec![vec![-0.7; 768]]),
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };
    for strategy in [
        FusionStrategy::CrossAttention,
        FusionStrategy::GatedFusion,
        FusionStrategy::TransformerFusion,
    ] {
        let cfg = MultiModalConfig {
            fusion_strategy: strategy,
            ..MultiModalConfig::default()
        };
        let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
        assert!(!fused.is_empty());
        for row in &fused {
            assert_eq!(row.len(), COMMON_FEATURE_DIM);
            assert!(
                row.iter().all(|v| v.is_finite()),
                "all values must be finite"
            );
        }
    }
}

// ---- Cross-attention weights tests ----

#[test]
fn test_attention_weights_normalised() {
    // compute_attention_weights normalises to sum 1 per query
    let query = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let key = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    // Reproduce the logic inline
    let mut attention_weights = Vec::new();
    for q in &query {
        let mut q_weights = Vec::new();
        for k in &key {
            let dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            let score = (dot / (q.len() as f32).sqrt()).exp();
            q_weights.push(score);
        }
        let sum: f32 = q_weights.iter().sum();
        if sum > 0.0 {
            q_weights.iter_mut().for_each(|w| *w /= sum);
        }
        attention_weights.push(q_weights);
    }

    for row in &attention_weights {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row sum = {}", sum);
    }
}

#[test]
fn test_dot_product_attention_weights_does_not_overflow_to_nan_at_realistic_scale() {
    // Regression test for a numerical-stability bug in the scaled
    // dot-product softmax: without subtracting the row max before
    // exponentiating, a full-width (768-dim) feature vector at a
    // realistic magnitude produces a scaled score in the hundreds
    // (768 * 3.0 * 3.0 / sqrt(768) ~= 249), and `f32::exp(249)` overflows
    // to `Infinity` well before that -- collapsing every softmax weight to
    // `Infinity / Infinity = NaN`. This is exactly the shape of input
    // `FusionLayer::cross_attention_fusion`/`transformer_fusion` feed
    // through this function on real `COMMON_FEATURE_DIM`-wide features.
    let query = vec![vec![3.0f32; 768]];
    let key = vec![vec![3.0f32; 768], vec![1.5f32; 768]];
    let weights = dot_product_attention_weights(&query, &key);

    assert_eq!(weights.len(), 1);
    assert_eq!(weights[0].len(), 2);
    for &w in &weights[0] {
        assert!(w.is_finite(), "attention weight must be finite, got {w}");
        assert!(
            (0.0..=1.0).contains(&w),
            "attention weight must be a probability, got {w}"
        );
    }
    let sum: f32 = weights[0].iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "weights must still sum to 1.0 after the numerically-stable rewrite, got {sum}"
    );
}

#[test]
fn test_dot_product_attention_weights_shift_invariance_matches_naive_formula_at_small_scale() {
    // At a small enough magnitude that the naive (non-max-subtracted)
    // formula does not overflow, the numerically-stable rewrite must
    // produce the same weights (softmax is mathematically shift-invariant)
    // -- this is not a new formula, only a safe way to evaluate the same
    // one.
    let query = vec![vec![0.5f32, -0.2, 0.1]];
    let key = vec![vec![0.4f32, 0.1, -0.3], vec![-0.1f32, 0.2, 0.5]];

    let stable = dot_product_attention_weights(&query, &key);

    let mut naive = Vec::new();
    for q in &query {
        let mut q_weights = Vec::new();
        for k in &key {
            let dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            let score = (dot / (q.len() as f32).sqrt()).exp();
            q_weights.push(score);
        }
        let sum: f32 = q_weights.iter().sum();
        if sum > 0.0 {
            q_weights.iter_mut().for_each(|w| *w /= sum);
        }
        naive.push(q_weights);
    }

    for (a, b) in stable[0].iter().zip(naive[0].iter()) {
        assert!((a - b).abs() < 1e-5, "stable={a}, naive={b}");
    }
}

#[test]
fn test_cross_modal_similarity_range() {
    // cosine similarity must be in [-1, 1]
    let a = [1.0_f32, 0.0, 0.0];
    let b = [0.0_f32, 1.0, 0.0];
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let sim = if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 };
    assert!((-1.0..=1.0).contains(&sim), "sim = {}", sim);
}

// ---- Output format tests ----

#[test]
fn test_classification_result_score_in_range() {
    let result = ClassificationResult {
        label: "positive".to_string(),
        score: 0.85,
        modality_contributions: HashMap::new(),
    };
    assert!(result.score >= 0.0 && result.score <= 1.0);
}

#[test]
fn test_processing_metadata_modalities_list() {
    let meta = ProcessingMetadata {
        processing_time_ms: 42,
        modalities_used: vec!["text".to_string(), "image".to_string()],
        fusion_strategy_used: "Concatenation".to_string(),
        model_confidence: None,
        feature_extraction_time_ms: HashMap::new(),
    };
    assert_eq!(meta.modalities_used.len(), 2);
    assert!(meta.modalities_used.contains(&"text".to_string()));
}

// -------------------------------------------------------------------
// insert_modality: regression coverage for the panic risk of indexing
// `text_features[0]` directly when a successfully-processed modality
// (e.g. empty text) produces zero feature vectors.
// -------------------------------------------------------------------

#[test]
fn test_insert_modality_empty_values_does_not_panic() {
    let mut features = ModalityFeatures {
        text_features: None,
        image_features: None,
        audio_features: None,
        video_features: None,
        feature_dims: HashMap::new(),
        attention_masks: HashMap::new(),
    };
    insert_modality(&mut features, "text", Vec::new(), |f, v| {
        f.text_features = v
    });
    assert_eq!(features.feature_dims["text"], 0);
    assert_eq!(features.text_features, Some(Vec::new()));
}

// -------------------------------------------------------------------
// chunk_features
// -------------------------------------------------------------------

#[test]
fn test_chunk_features_splits_evenly() {
    let flat: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let chunks = chunk_features(flat, 4);
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[1], vec![4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn test_chunk_features_drops_short_remainder() {
    let flat: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let chunks = chunk_features(flat, 4);
    // 10 / 4 = 2 full chunks; the trailing 2 values are dropped rather
    // than padded with fabricated zeros.
    assert_eq!(chunks.len(), 2);
}

// -------------------------------------------------------------------
// sniff_image_format
// -------------------------------------------------------------------

#[test]
fn test_sniff_image_format_recognises_png_signature() {
    let png_sig = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];
    assert!(matches!(sniff_image_format(&png_sig), ImageFormat::Png));
}

#[test]
fn test_sniff_image_format_recognises_jpeg_signature() {
    let jpeg_sig = [0xFF, 0xD8, 0xFF, 0xE0];
    assert!(matches!(sniff_image_format(&jpeg_sig), ImageFormat::Jpeg));
}
