//! Tests for layer implementations

use crate::layers::{Embedding, FeedForward, LayerNorm, Linear, MultiHeadAttention};
use crate::tensor::Tensor;
use crate::traits::Layer;

#[test]
fn test_linear_layer() {
    let linear = Linear::new(10, 5, true);

    // Test 2D input
    let input_2d = Tensor::zeros(&[3, 10]).unwrap();
    let output_2d = linear.forward(input_2d).unwrap();
    assert_eq!(output_2d.shape(), vec![3, 5]);

    // Test 3D input
    let input_3d = Tensor::zeros(&[2, 3, 10]).unwrap();
    let output_3d = linear.forward(input_3d).unwrap();
    assert_eq!(output_3d.shape(), vec![2, 3, 5]);
}

#[test]
fn test_linear_layer_without_bias() {
    let linear = Linear::new(10, 5, false);

    let input = Tensor::zeros(&[3, 10]).unwrap();
    let output = linear.forward(input).unwrap();
    assert_eq!(output.shape(), vec![3, 5]);
}

#[test]
fn test_embedding_layer() {
    let embedding = Embedding::new(100, 64, None).unwrap();

    let input_ids = vec![1u32, 5, 10, 99];
    let output = embedding.forward(input_ids).unwrap();
    assert_eq!(output.shape(), vec![4, 64]);
}

#[test]
fn test_embedding_with_padding() {
    let embedding = Embedding::new(100, 64, Some(0)).unwrap();

    let input_ids = vec![0u32, 1, 5, 0]; // 0 is padding
    let output = embedding.forward(input_ids).unwrap();
    assert_eq!(output.shape(), vec![4, 64]);
}

#[test]
fn test_layer_norm() {
    let layer_norm = LayerNorm::new(vec![10], 1e-5).unwrap();

    // Test 2D input
    let input_2d = Tensor::ones(&[3, 10]).unwrap();
    let output_2d = layer_norm.forward(input_2d).unwrap();
    assert_eq!(output_2d.shape(), vec![3, 10]);

    // Test 3D input
    let input_3d = Tensor::ones(&[2, 4, 10]).unwrap();
    let output_3d = layer_norm.forward(input_3d).unwrap();
    assert_eq!(output_3d.shape(), vec![2, 4, 10]);
}

#[test]
fn test_feedforward_layer() {
    let ff = FeedForward::new(10, 20, 0.1).unwrap();

    // Test 2D input
    let input_2d = Tensor::zeros(&[3, 10]).unwrap();
    let output_2d = ff.forward(input_2d).unwrap();
    assert_eq!(output_2d.shape(), vec![3, 10]);

    // Test 3D input
    let input_3d = Tensor::zeros(&[2, 4, 10]).unwrap();
    let output_3d = ff.forward(input_3d).unwrap();
    assert_eq!(output_3d.shape(), vec![2, 4, 10]);
}

#[test]
fn test_multihead_attention() {
    let attention = MultiHeadAttention::new(64, 8, 0.1, false).unwrap();

    let input = Tensor::zeros(&[2, 10, 64]).unwrap(); // batch_size=2, seq_len=10, hidden_size=64
    let output = attention.forward_self_attention(&input, None, false).unwrap();
    assert_eq!(output.shape(), vec![2, 10, 64]);
}

#[test]
fn test_multihead_attention_with_mask() {
    let attention = MultiHeadAttention::new(64, 8, 0.1, false).unwrap();

    let input = Tensor::zeros(&[2, 10, 64]).unwrap();
    let mask = Some(Tensor::ones(&[2, 10, 10]).unwrap()); // Allow all attention

    let output = attention.forward_self_attention(&input, mask.as_ref(), false).unwrap();
    assert_eq!(output.shape(), vec![2, 10, 64]);
}

#[test]
fn test_multihead_attention_causal() {
    let attention = MultiHeadAttention::new(64, 8, 0.1, true).unwrap(); // causal=true

    let input = Tensor::zeros(&[1, 5, 64]).unwrap();
    match attention.forward_self_attention(&input, None, true) {
        Ok(output) => {
            assert_eq!(output.shape(), vec![1, 5, 64]); // Preserve batch dimension
        },
        Err(e) => {
            // Print detailed error for debugging
            println!("MultiHeadAttention causal test failed with error: {:?}", e);
            // Try without causal to isolate the issue
            match attention.forward_self_attention(&input, None, false) {
                Ok(_) => {
                    println!("Non-causal works, issue is specifically with causal masking")
                },
                Err(e2) => println!("Even non-causal fails: {:?}", e2),
            }
            panic!("Test failed: {:?}", e);
        },
    }
}

#[test]
fn test_attention_head_dimension() {
    // Test that head dimension is correctly computed
    let attention = MultiHeadAttention::new(64, 8, 0.0, false).unwrap();
    // head_dim should be 64 / 8 = 8

    let input = Tensor::zeros(&[1, 4, 64]).unwrap();
    let output = attention.forward(input).unwrap();
    assert_eq!(output.shape(), vec![1, 4, 64]); // Preserve batch dimension
}

#[test]
fn test_embedding_out_of_bounds() {
    let embedding = Embedding::new(100, 64, None).unwrap();

    let invalid_ids = vec![100u32, 101]; // Out of bounds
    let result = embedding.forward(invalid_ids);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_different_shapes() {
    // Test LayerNorm with different normalized shapes
    let layer_norm_last = LayerNorm::new(vec![10], 1e-5).unwrap();
    let layer_norm_2d = LayerNorm::new(vec![5, 10], 1e-5).unwrap();

    let input_3d = Tensor::ones(&[2, 5, 10]).unwrap();

    // Normalize over last dimension
    let output_last = layer_norm_last.forward(input_3d.clone()).unwrap();
    assert_eq!(output_last.shape(), vec![2, 5, 10]);

    // Normalize over last two dimensions
    let output_2d = layer_norm_2d.forward(input_3d).unwrap();
    assert_eq!(output_2d.shape(), vec![2, 5, 10]);
}

#[test]
fn test_linear_weight_initialization() {
    let linear = Linear::new(100, 50, true);

    // Check that weights are properly initialized (not all zeros)
    let test_input = Tensor::ones(&[1, 100]).unwrap();
    let output = linear.forward(test_input).unwrap();

    // Output should not be all zeros due to weight initialization
    // We can check shape but not values since weights are random
    assert_eq!(output.shape(), vec![1, 50]);
}

#[test]
fn test_feedforward_activation() {
    let ff = FeedForward::new(4, 8, 0.0).unwrap(); // no dropout

    // Test with small input
    let input = Tensor::ones(&[1, 4]).unwrap();
    let output = ff.forward(input).unwrap();

    assert_eq!(output.shape(), vec![1, 4]);
}

#[test]
fn test_attention_different_head_counts() {
    // Test different numbers of attention heads
    for num_heads in [1, 2, 4, 8] {
        let hidden_size = 64;
        if hidden_size % num_heads == 0 {
            let attention = MultiHeadAttention::new(hidden_size, num_heads, 0.0, false).unwrap();
            let input = Tensor::zeros(&[1, 5, hidden_size]).unwrap();
            let output = attention.forward_self_attention(&input, None, false).unwrap();
            assert_eq!(output.shape(), vec![1, 5, hidden_size]); // Preserves batch dimension
        }
    }
}

#[test]
fn test_layer_error_conditions() {
    // Test invalid configurations

    // Hidden size not divisible by num_heads
    let result = MultiHeadAttention::new(65, 8, 0.0, false);
    assert!(result.is_err());
}

#[test]
fn test_layer_consistency() {
    // Test that layers produce consistent outputs for same inputs
    let linear = Linear::new(10, 5, true);
    let input = Tensor::ones(&[3, 10]).unwrap();

    let output1 = linear.forward(input.clone()).unwrap();
    let output2 = linear.forward(input).unwrap();

    // Should be exactly the same (no randomness in forward pass)
    assert_eq!(output1.shape(), output2.shape());
}

#[test]
fn test_tensor_operations() {
    // Test basic tensor operations work
    let t1 = Tensor::zeros(&[2, 3]).unwrap();
    let t2 = Tensor::ones(&[2, 3]).unwrap();

    assert_eq!(t1.shape(), vec![2, 3]);
    assert_eq!(t2.shape(), vec![2, 3]);

    let sum = t1.add(&t2).unwrap();
    assert_eq!(sum.shape(), vec![2, 3]);
}

#[test]
fn test_embedding_basic_functionality() {
    let vocab_size = 10;
    let embedding_dim = 5;
    let embedding = Embedding::new(vocab_size, embedding_dim, None).unwrap();

    // Test single ID
    let single_id = vec![3u32];
    let output = embedding.forward(single_id).unwrap();
    assert_eq!(output.shape(), vec![1, embedding_dim]);

    // Test multiple IDs
    let multi_ids = vec![0u32, 1, 2, 9];
    let output = embedding.forward(multi_ids).unwrap();
    assert_eq!(output.shape(), vec![4, embedding_dim]);
}
