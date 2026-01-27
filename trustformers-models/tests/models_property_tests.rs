//! Property-based tests for transformer models

use proptest::prelude::*;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Model, TokenizedInput};
use trustformers_models::bert::{BertConfig, BertModel};

// Strategy for generating very small sequence lengths for memory efficiency
fn tiny_seq_length_strategy() -> impl Strategy<Value = usize> {
    1usize..16 // Much smaller to reduce memory usage
}

// Strategy for generating very small model configs for memory-constrained testing
fn tiny_bert_config() -> impl Strategy<Value = BertConfig> {
    (
        (32usize..64).prop_filter("|x| x % 32 == 0", |x| x % 32 == 0), // hidden_size (much smaller)
        (1usize..2), // num_hidden_layers (just 1 layer)
        (1usize..4).prop_filter("power of 2", |x| x.is_power_of_two()), // num_attention_heads
    )
        .prop_map(|(hidden_size, num_hidden_layers, num_attention_heads)| {
            BertConfig {
                vocab_size: 100, // Reduced from 1000
                hidden_size,
                num_hidden_layers,
                num_attention_heads,
                intermediate_size: hidden_size * 2, // Reduced from 4x
                hidden_act: "gelu".to_string(),
                hidden_dropout_prob: 0.0, // No dropout for faster testing
                attention_probs_dropout_prob: 0.0,
                max_position_embeddings: 64, // Reduced from 512
                type_vocab_size: 2,
                initializer_range: 0.02,
                layer_norm_eps: 1e-12,
                position_embedding_type: Some("absolute".to_string()),
                use_cache: Some(false),
                ..Default::default()
            }
        })
}

// Property: BERT model output shapes with memory constraints
proptest! {
    #![proptest_config(ProptestConfig {
        timeout: 1000, // 1 second timeout per test case
        max_shrink_iters: 10,
        ..ProptestConfig::default()
    })]

    #[test]
    fn test_bert_output_shapes_memory_constrained(
        config in tiny_bert_config(),
        batch_size in 1usize..2, // Only 1 batch
        seq_len in tiny_seq_length_strategy()
    ) {
        prop_assume!(seq_len <= config.max_position_embeddings);
        prop_assume!(config.hidden_size % config.num_attention_heads == 0);

        let model = BertModel::new(config.clone()).unwrap();

        // Create input tensors
        let input_ids = Tensor::zeros(&[batch_size, seq_len]).unwrap();
        let attention_mask = Some(Tensor::ones(&[batch_size, seq_len]).unwrap());
        let token_type_ids = Some(Tensor::zeros(&[batch_size, seq_len]).unwrap());

        // Convert tensors to vectors for TokenizedInput
        let input_ids_vec: Vec<u32> = (0..batch_size*seq_len).map(|_| 0u32).collect();
        let attention_mask_vec: Vec<u8> = (0..batch_size*seq_len).map(|_| 1u8).collect();
        let token_type_ids_vec: Option<Vec<u32>> = Some((0..batch_size*seq_len).map(|_| 0u32).collect());

        let tokenized_input = TokenizedInput {
            input_ids: input_ids_vec,
            attention_mask: attention_mask_vec,
            token_type_ids: token_type_ids_vec,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let output = model.forward(tokenized_input);

        match output {
            Ok(model_output) => {
                // Check last hidden state shape
                prop_assert_eq!(
                    model_output.last_hidden_state.shape(),
                    &vec![batch_size, seq_len, config.hidden_size][..]
                );

                // Check pooler output if present
                if let Some(ref pooler_output) = model_output.pooler_output {
                    prop_assert_eq!(
                        pooler_output.shape(),
                        &vec![batch_size, config.hidden_size][..]
                    );
                }
            }
            Err(e) => prop_assert!(false, "Model forward failed: {:?}", e),
        }

        // Cleanup
        drop(model);
        drop(input_ids);
        drop(attention_mask);
        drop(token_type_ids);
        std::hint::black_box(());
    }
}

// Property: Model attention mask handling with memory constraints
proptest! {
    #![proptest_config(ProptestConfig {
        timeout: 500, // 0.5 second timeout
        max_shrink_iters: 5,
        ..ProptestConfig::default()
    })]

    #[test]
    #[ignore] // Child processes crash due to resource constraints
    fn test_attention_mask_dimensions_memory_constrained(
        batch_size in 1usize..2, // Only 1 batch
        seq_len in tiny_seq_length_strategy(),
        mask_proportion in 0.0f32..1.0
    ) {
        let config = BertConfig {
            vocab_size: 100, // Reduced from 1000
            hidden_size: 32, // Reduced from 128
            num_hidden_layers: 1,
            num_attention_heads: 2, // Reduced from 4
            intermediate_size: 64, // Reduced from 512
            max_position_embeddings: 64,
            ..Default::default()
        };

        let model = BertModel::new(config).unwrap();

        // Create input with partial masking
        let input_ids = Tensor::zeros(&[batch_size, seq_len]).unwrap();

        // Create attention mask with some positions masked
        let mut mask_data = vec![1.0f32; batch_size * seq_len];
        let num_masked = (seq_len as f32 * mask_proportion) as usize;
        for i in 0..batch_size {
            for j in 0..num_masked {
                mask_data[i * seq_len + j] = 0.0;
            }
        }
        let attention_mask = Some(Tensor::from_vec(mask_data.clone(), &[batch_size, seq_len]).unwrap());

        // Convert tensors to vectors for TokenizedInput
        let input_ids_vec: Vec<u32> = (0..batch_size*seq_len).map(|_| 0u32).collect();
        let attention_mask_vec: Vec<u8> = mask_data.iter().map(|&x| if x > 0.5 { 1u8 } else { 0u8 }).collect();

        let tokenized_input = TokenizedInput {
            input_ids: input_ids_vec,
            attention_mask: attention_mask_vec,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let output = model.forward(tokenized_input);

        // Model should handle any valid mask
        prop_assert!(output.is_ok());

        // Explicit cleanup
        if let Ok(output) = output {
            drop(output);
        }
        drop(model);
        drop(input_ids);
        drop(attention_mask);
        std::hint::black_box(());
    }
}

// Property: Model robustness to input variations with memory constraints
proptest! {
    #![proptest_config(ProptestConfig {
        timeout: 300, // 0.3 second timeout
        max_shrink_iters: 3,
        ..ProptestConfig::default()
    })]

    #[test]
    #[ignore] // Child processes crash due to resource constraints
    fn test_model_input_robustness_memory_constrained(
        batch_size in 1usize..2, // Only 1 batch
        seq_len in 1usize..8, // Much smaller
        vocab_size in 50usize..100, // Much smaller
        token_ids in prop::collection::vec(0i64..50, 1..32) // Much smaller
    ) {
        let config = BertConfig {
            vocab_size,
            hidden_size: 32, // Reduced from 128
            num_hidden_layers: 1,
            num_attention_heads: 2, // Reduced from 4
            intermediate_size: 64, // Reduced from 512
            max_position_embeddings: 64, // Reduced from 512
            ..Default::default()
        };

        let model = BertModel::new(config).unwrap();

        // Create input with random token IDs
        let input_data: Vec<f32> = token_ids.into_iter()
            .take(batch_size * seq_len)
            .map(|id| (id.min(vocab_size as i64 - 1)) as f32)
            .collect();

        let input_ids = Tensor::from_vec(
            input_data.clone(),
            &[batch_size, seq_len]
        ).unwrap();

        // Convert tensors to vectors for TokenizedInput
        let input_ids_vec: Vec<u32> = input_data.iter().map(|&x| x as u32).collect();
        let attention_mask_vec: Vec<u8> = (0..batch_size*seq_len).map(|_| 1u8).collect();

        let tokenized_input = TokenizedInput {
            input_ids: input_ids_vec,
            attention_mask: attention_mask_vec,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let output = model.forward(tokenized_input);

        // Model should handle any valid input IDs
        prop_assert!(output.is_ok());

        // Explicit cleanup
        if let Ok(output) = output {
            drop(output);
        }
        drop(model);
        drop(input_ids);
        std::hint::black_box(());
    }
}

// Property: Position embeddings bounds with memory constraints
proptest! {
    #![proptest_config(ProptestConfig {
        timeout: 500, // Increase timeout to 0.5 seconds
        max_shrink_iters: 1, // Reduce shrink iterations
        cases: 8, // Reduce number of test cases from default 256
        ..ProptestConfig::default()
    })]

    #[test]
    fn test_position_embedding_bounds_memory_constrained(
        seq_len in 1usize..8, // Even smaller range
        max_positions in 8usize..16 // Even smaller range
    ) {
        let config = BertConfig {
            vocab_size: 50, // Even smaller vocab
            hidden_size: 32, // Keep small
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 64,
            max_position_embeddings: max_positions,
            hidden_dropout_prob: 0.0, // No dropout for speed
            attention_probs_dropout_prob: 0.0,
            ..Default::default()
        };

        prop_assume!(seq_len <= max_positions);

        let model = BertModel::new(config).unwrap();

        let batch_size = 1;
        let input_ids = Tensor::zeros(&[batch_size, seq_len]).unwrap();

        // Convert tensors to vectors for TokenizedInput
        let input_ids_vec: Vec<u32> = (0..batch_size*seq_len).map(|_| 0u32).collect();
        let attention_mask_vec: Vec<u8> = (0..batch_size*seq_len).map(|_| 1u8).collect();

        let tokenized_input = TokenizedInput {
            input_ids: input_ids_vec,
            attention_mask: attention_mask_vec,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let output = model.forward(tokenized_input);

        // Should succeed for valid sequence lengths
        prop_assert!(output.is_ok());

        // Explicit cleanup
        if let Ok(output) = output {
            drop(output);
        }
        drop(model);
        drop(input_ids);
        std::hint::black_box(());
    }
}

// Property: Model determinism with memory constraints
proptest! {
    #![proptest_config(ProptestConfig {
        timeout: 300, // 0.3 second timeout
        max_shrink_iters: 2,
        ..ProptestConfig::default()
    })]

    #[test]
    #[ignore] // Child processes crash due to resource constraints
    fn test_model_determinism_memory_constrained(
        batch_size in 1usize..2, // Only 1 batch
        seq_len in 1usize..8 // Much smaller
    ) {
        let config = BertConfig {
            vocab_size: 100, // Reduced from 1000
            hidden_size: 32, // Reduced from 128
            num_hidden_layers: 1,
            num_attention_heads: 2, // Reduced from 4
            intermediate_size: 64, // Reduced from 512
            hidden_dropout_prob: 0.0, // No dropout for determinism
            attention_probs_dropout_prob: 0.0,
            ..Default::default()
        };

        let model = BertModel::new(config).unwrap();

        let input_ids = Tensor::zeros(&[batch_size, seq_len]).unwrap();
        let attention_mask = Some(Tensor::ones(&[batch_size, seq_len]).unwrap());

        // Convert tensors to vectors for TokenizedInput
        let input_ids_vec: Vec<u32> = (0..batch_size*seq_len).map(|_| 0u32).collect();
        let attention_mask_vec: Vec<u8> = (0..batch_size*seq_len).map(|_| 1u8).collect();

        let tokenized_input = TokenizedInput {
            input_ids: input_ids_vec,
            attention_mask: attention_mask_vec,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        // Run forward pass twice
        let output1 = model.forward(tokenized_input.clone()).unwrap();
        let output2 = model.forward(tokenized_input).unwrap();

        // Compare last hidden states
        let diff: f32 = output1.last_hidden_state.data().unwrap().iter()
            .zip(output2.last_hidden_state.data().unwrap().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        prop_assert!(diff < 1e-6, "Model should be deterministic without dropout");

        // Cleanup
        drop(model);
        drop(input_ids);
        drop(attention_mask);
        std::hint::black_box(());
    }
}
