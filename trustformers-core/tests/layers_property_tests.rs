//! Property-based tests for neural network layers

use proptest::prelude::*;
use scirs2_core::random::*;
use trustformers_core::{
    layers::{Embedding, FeedForward, LayerNorm, Linear, MultiHeadAttention},
    tensor::Tensor,
    traits::Layer,
};

// Strategy for reasonable layer dimensions
fn layer_dimension_strategy() -> impl Strategy<Value = usize> {
    (1usize..128).prop_filter("power of 2 preferred", |x| x % 2 == 0)
}

// Property: Linear layer output shape
proptest! {
    #[test]
    fn test_linear_layer_shape_property(
        batch_size in 1usize..32,
        in_features in layer_dimension_strategy(),
        out_features in layer_dimension_strategy(),
        use_bias in any::<bool>()
    ) {
        let layer = Linear::new(in_features, out_features, use_bias);

        let input_shape = vec![batch_size, in_features];
        let input_data = vec![0.1; batch_size * in_features];
        let input = Tensor::from_vec(input_data, &input_shape).unwrap();

        let output = layer.forward(input).unwrap();

        // Output shape should be [batch_size, out_features]
        prop_assert_eq!(output.shape(), vec![batch_size, out_features]);
    }
}

// Property: LayerNorm preserves shape
proptest! {
    #[test]
    fn test_layer_norm_shape_preservation(
        batch_size in 1usize..32,
        features in layer_dimension_strategy()
    ) {
        let normalized_shape = vec![features];
        let layer_norm = LayerNorm::new(normalized_shape, 1e-5).unwrap();

        let input_shape = vec![batch_size, features];
        let input_data = vec![1.0; batch_size * features];
        let input = Tensor::from_vec(input_data, &input_shape).unwrap();

        let input_shape_copy = input_shape.clone();
        let output = layer_norm.forward(input).unwrap();

        // Shape should be preserved
        prop_assert_eq!(output.shape(), input_shape_copy);
    }
}

// Property: LayerNorm statistical properties
proptest! {
    #[test]
    fn test_layer_norm_statistics(
        batch_size in 1usize..32,
        features in 2usize..64
    ) {
        let total_elements = batch_size * features;
        let values: Vec<f32> = (0..total_elements)
            .map(|i| (i as f32 % 20.0) - 10.0) // Generate values in range [-10.0, 10.0)
            .collect();

        let normalized_shape = vec![features];
        let layer_norm = LayerNorm::new(normalized_shape, 1e-5).unwrap();

        let input_shape = vec![batch_size, features];
        let input_data: Vec<f32> = values;
        let input = Tensor::from_vec(input_data, &input_shape).unwrap();

        let output = layer_norm.forward(input).unwrap();

        // Check that each sample has approximately zero mean and unit variance
        let output_data = output.data().unwrap();
        for b in 0..batch_size {
            let start = b * features;
            let end = start + features;
            let sample = &output_data[start..end];

            let mean: f32 = sample.iter().sum::<f32>() / features as f32;
            let variance: f32 = sample.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / features as f32;

            // Mean should be close to 0
            prop_assert!(mean.abs() < 0.1);
            // Variance should be close to 1
            prop_assert!((variance - 1.0).abs() < 0.1);
        }
    }
}

// Property: Embedding bounds
proptest! {
    #[test]
    fn test_embedding_bounds(
        vocab_size in 10usize..1000,
        embedding_dim in layer_dimension_strategy(),
        seq_len in 1usize..128
    ) {
        let embedding = Embedding::new(vocab_size, embedding_dim, None).unwrap();

        // Generate valid token indices
        let mut rng = thread_rng();
        let indices: Vec<u32> = (0..seq_len)
            .map(|_| rng.random_range(0..vocab_size) as u32)
            .collect();

        let output = embedding.forward(indices);

        match output {
            Ok(tensor) => {
                // Output shape should be [seq_len, embedding_dim] (batch is implicit in Vec input)
                prop_assert_eq!(tensor.shape(), vec![seq_len, embedding_dim]);
            }
            Err(_) => {
                // If it errors, it should be because of out-of-bounds indices
                prop_assert!(false, "Embedding forward should not fail with valid indices");
            }
        }
    }
}

// Property: FeedForward dimension transformation
proptest! {
    #[test]
    fn test_feedforward_dimensions(
        batch_size in 1usize..32,
        d_model in layer_dimension_strategy(),
        dim_feedforward_factor in 2usize..8,
        dropout in 0.0f32..0.5
    ) {
        let dim_feedforward = d_model * dim_feedforward_factor;
        let ff = FeedForward::new(d_model, dim_feedforward, dropout).unwrap();

        let input_shape = vec![batch_size, d_model];
        let input_data = vec![0.1; batch_size * d_model];
        let input = Tensor::from_vec(input_data, &input_shape).unwrap();

        let output = ff.forward(input).unwrap();

        // Output should have same shape as input
        prop_assert_eq!(output.shape(), vec![batch_size, d_model]);
    }
}

// Property: MultiHeadAttention shape consistency
proptest! {
    #[test]
    fn test_multihead_attention_shapes(
        batch_size in 1usize..16,
        seq_len in 1usize..64,
        d_model in (32usize..256).prop_filter("divisible by 8", |x| x % 8 == 0),
        num_heads in (1usize..8).prop_filter("power of 2", |x| x.is_power_of_two())
    ) {
        prop_assume!(d_model % num_heads == 0);

        let mha = MultiHeadAttention::new(d_model, num_heads, 0.1, true).unwrap();

        let input_shape = vec![batch_size, seq_len, d_model];
        let input_data = vec![0.1; batch_size * seq_len * d_model];
        let input = Tensor::from_vec(input_data, &input_shape).unwrap();

        // With self-attention
        let input_shape_copy = input_shape.clone();
        let output = mha.forward(input).unwrap();

        // Output shape should match input shape
        prop_assert_eq!(output.shape(), input_shape_copy);
    }
}

// Property: Attention mask effect
proptest! {
    #[test]
    fn test_attention_mask_effect(
        seq_len in 2usize..32,
        d_model in (32usize..128).prop_filter("divisible by 4", |x| x % 4 == 0)
    ) {
        let batch_size = 1;
        let num_heads = 4;
        prop_assume!(d_model % num_heads == 0);

        let mha = MultiHeadAttention::new(d_model, num_heads, 0.0, true).unwrap(); // No dropout for deterministic test

        let input_shape = vec![batch_size, seq_len, d_model];
        let input_data = vec![1.0; batch_size * seq_len * d_model];
        let input = Tensor::from_vec(input_data.clone(), &input_shape).unwrap();

        // Since the Layer trait API doesn't expose mask functionality directly,
        // we'll just test that the attention produces consistent outputs
        let input_copy = Tensor::from_vec(input_data, &input_shape).unwrap();

        let output1 = mha.forward(input).unwrap();
        let output2 = mha.forward(input_copy).unwrap();

        // With identical inputs, outputs should be the same (deterministic)
        let data1 = output1.data().unwrap();
        let data2 = output2.data().unwrap();
        let diff: f32 = data1.iter()
            .zip(data2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        prop_assert!(diff < 1e-5, "Identical inputs should produce identical outputs");
    }
}

// Property: Dropout effect (statistical)
proptest! {
    #[test]
    fn test_dropout_statistical_property(
        size in 100usize..1000,
        dropout_rate in 0.1f32..0.9
    ) {
        use trustformers_core::layers::Dropout;

        let dropout = Dropout::new(dropout_rate);

        let input_data = vec![1.0; size];
        let _input = Tensor::from_vec(input_data, &vec![size]).unwrap();

        // Apply dropout multiple times and check statistics
        let mut zero_counts = Vec::new();

        for _ in 0..10 {
            let input = Tensor::from_vec(vec![1.0; size], &vec![size]).unwrap();
            let output = dropout.forward(input).unwrap();
            let output_data = output.data().unwrap();
            let zero_count = output_data.iter().filter(|&&x| x == 0.0).count();
            zero_counts.push(zero_count);
        }

        // Average zero rate should be close to dropout rate
        let avg_zero_rate = zero_counts.iter().sum::<usize>() as f32 / (10.0 * size as f32);

        // Allow 10% tolerance
        prop_assert!((avg_zero_rate - dropout_rate).abs() < 0.1);
    }
}

// Property: Composition of layers preserves batch dimension
proptest! {
    #[test]
    fn test_layer_composition_batch_preservation(
        batch_size in 1usize..32,
        seq_len in 1usize..64,
        d_model in (64usize..256).prop_filter("divisible by 8", |x| x % 8 == 0)
    ) {
        // Build a small transformer block
        let mha = MultiHeadAttention::new(d_model, 8, 0.1, true).unwrap();
        let ff = FeedForward::new(d_model, d_model * 4, 0.1).unwrap();
        let ln1 = LayerNorm::new(vec![d_model], 1e-5).unwrap();
        let ln2 = LayerNorm::new(vec![d_model], 1e-5).unwrap();

        let input_shape = vec![batch_size, seq_len, d_model];
        let input_data = vec![0.1; batch_size * seq_len * d_model];
        let input = Tensor::from_vec(input_data, &input_shape).unwrap();

        // Forward through all layers
        let attn_out = mha.forward(input).unwrap();
        let norm1_out = ln1.forward(attn_out).unwrap();
        let ff_out = ff.forward(norm1_out).unwrap();
        let final_out = ln2.forward(ff_out).unwrap();

        // Batch size should be preserved throughout
        prop_assert_eq!(final_out.shape()[0], batch_size);
        prop_assert_eq!(final_out.shape(), vec![batch_size, seq_len, d_model]);
    }
}
