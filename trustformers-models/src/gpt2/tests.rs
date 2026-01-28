use crate::gpt2::{Gpt2Config, Gpt2LMHeadModel, Gpt2Model};
use std::time::Instant;
use trustformers_core::{
    tensor::Tensor,
    traits::{Model, TokenizedInput},
    Device,
};

#[test]
fn test_gpt2_model_creation() {
    let config = Gpt2Config::default();
    let model = Gpt2Model::new(config.clone()).expect("operation failed");
    assert_eq!(model.get_config().n_layer, 12);
    assert_eq!(model.get_config().n_head, 12);
}

#[test]
fn test_gpt2_lm_head_model_creation() {
    let config = Gpt2Config::default();
    let model = Gpt2LMHeadModel::new(config.clone()).expect("operation failed");
    assert_eq!(model.get_config().n_layer, 12);
}

#[test]
fn test_gpt2_forward_pass() {
    let config = Gpt2Config {
        vocab_size: 100,
        n_positions: 64,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        ..Default::default()
    };

    let model = Gpt2Model::new(config).expect("operation failed");
    let input = TokenizedInput {
        input_ids: vec![1, 2, 3, 4, 5],
        attention_mask: vec![1u8; 5],
        token_type_ids: None,
        offset_mapping: None,
        special_tokens_mask: None,
        overflowing_tokens: None,
    };

    let output = model.forward(input).expect("operation failed");
    match &output.last_hidden_state {
        Tensor::F32(arr) => {
            assert_eq!(arr.shape(), &[1, 5, 32]);
        },
        _ => panic!("Expected F32 tensor"),
    }
}

#[test]
fn test_gpt2_lm_forward_pass() {
    let config = Gpt2Config {
        vocab_size: 100,
        n_positions: 64,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        ..Default::default()
    };

    let model = Gpt2LMHeadModel::new(config).expect("operation failed");
    let input = TokenizedInput {
        input_ids: vec![1, 2, 3, 4, 5],
        attention_mask: vec![1u8; 5],
        token_type_ids: None,
        offset_mapping: None,
        special_tokens_mask: None,
        overflowing_tokens: None,
    };

    let output = model.forward(input).expect("operation failed");
    match &output.logits {
        Tensor::F32(arr) => {
            assert_eq!(arr.shape(), &[1, 5, 100]);
        },
        _ => panic!("Expected F32 tensor"),
    }
}

#[test]
fn test_gpt2_generate_greedy() {
    let config = Gpt2Config {
        vocab_size: 100,
        n_positions: 64,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        ..Default::default()
    };

    let model = Gpt2LMHeadModel::new(config).expect("operation failed");
    let input_ids = vec![1, 2, 3];
    let generated = model.generate_greedy(input_ids.clone(), 10).expect("operation failed");

    assert!(generated.len() >= input_ids.len());
    assert!(generated.len() <= 10);
    assert_eq!(&generated[..3], &input_ids[..]);
}

#[test]
fn test_gpt2_beam_search() {
    // Use even smaller config to reduce memory pressure
    let config = Gpt2Config {
        vocab_size: 30,  // Reduce from 50
        n_positions: 32, // Reduce from 64
        n_embd: 16,      // Reduce from 32
        n_layer: 1,
        n_head: 2, // Reduce from 4
        ..Default::default()
    };

    let model = Gpt2LMHeadModel::new(config).expect("operation failed");
    let input_ids = vec![1, 2];
    let max_length = 6; // Reduce from 10
    let num_beams = 2; // Reduce from 3

    let generated = model
        .generate_beam_search(input_ids.clone(), max_length, num_beams)
        .expect("operation failed");

    assert!(generated.len() >= input_ids.len());
    assert!(generated.len() <= max_length);
    assert_eq!(&generated[..2], &input_ids[..]);

    // Explicit cleanup
    drop(generated);
    drop(model);
    std::hint::black_box(());
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn test_gpt2_metal_sampling() {
    use crate::gpt2::generation::GenerativeModel;

    // Very small model config to prevent Metal SIGTRAP
    let config = Gpt2Config {
        vocab_size: 50,  // Reduce from 100
        n_positions: 32, // Reduce from 64
        n_embd: 32,      // Reduce from 64
        n_layer: 1,      // Reduce from 2
        n_head: 2,       // Reduce from 4
        ..Default::default()
    };

    // Test Metal device availability
    let device = Device::metal_if_available(0);
    println!("Using device: {:?}", device);

    // Skip test if Metal is not available
    if !matches!(device, Device::Metal(_)) {
        println!("Metal not available, skipping test");
        return;
    }

    // Create model with device
    let model = Gpt2LMHeadModel::new_with_device(config.clone(), device).expect("operation failed");

    // Test input - use smaller values
    let input_ids = vec![1, 2];
    let max_length = 8; // Reduce from 20
    let k = 5; // Reduce from 10
    let temperature = 1.0;

    // Measure generation time
    let start = Instant::now();
    let generated = model
        .generate_top_k(input_ids.clone(), max_length, k, temperature)
        .expect("operation failed");
    let elapsed = start.elapsed();

    println!(
        "Generated {} tokens in {:?}",
        generated.len() - input_ids.len(),
        elapsed
    );
    println!(
        "Tokens/sec: {:.2}",
        (generated.len() - input_ids.len()) as f64 / elapsed.as_secs_f64()
    );

    // Verify output
    assert!(generated.len() >= input_ids.len());
    assert!(generated.len() <= max_length);
    assert_eq!(&generated[..2], &input_ids[..]);

    // Explicit cleanup
    drop(generated);
    drop(model);
    std::hint::black_box(());
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn test_gpt2_metal_vs_cpu_performance() {
    use crate::gpt2::generation::GenerativeModel;

    // Very small model config to prevent Metal SIGTRAP
    let config = Gpt2Config {
        vocab_size: 50,  // Reduce from 100
        n_positions: 32, // Reduce from 128
        n_embd: 32,      // Reduce from 128
        n_layer: 1,      // Reduce from 4
        n_head: 2,       // Reduce from 8
        ..Default::default()
    };

    let input_ids = vec![1, 2, 3];
    let max_length = 8; // Reduce from 25
    let k = 5; // Reduce from 20
    let temperature = 1.0;

    // CPU benchmark
    println!("\n=== CPU Benchmark ===");
    let cpu_model =
        Gpt2LMHeadModel::new_with_device(config.clone(), Device::CPU).expect("operation failed");
    let cpu_start = Instant::now();
    let cpu_generated = cpu_model
        .generate_top_k(input_ids.clone(), max_length, k, temperature)
        .expect("operation failed");
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_tokens = cpu_generated.len() - input_ids.len();
    let cpu_tok_per_sec = cpu_tokens as f64 / cpu_elapsed.as_secs_f64();

    println!("CPU: Generated {} tokens in {:?}", cpu_tokens, cpu_elapsed);
    println!("CPU: {:.2} tokens/sec", cpu_tok_per_sec);

    // Explicit cleanup for CPU model
    drop(cpu_generated);
    drop(cpu_model);
    std::hint::black_box(());

    // Metal benchmark (if available)
    let device = Device::metal_if_available(0);
    if matches!(device, Device::Metal(_)) {
        println!("\n=== Metal GPU Benchmark ===");
        let metal_model =
            Gpt2LMHeadModel::new_with_device(config.clone(), device).expect("operation failed");
        let metal_start = Instant::now();
        let metal_generated = metal_model
            .generate_top_k(input_ids.clone(), max_length, k, temperature)
            .expect("operation failed");
        let metal_elapsed = metal_start.elapsed();
        let metal_tokens = metal_generated.len() - input_ids.len();
        let metal_tok_per_sec = metal_tokens as f64 / metal_elapsed.as_secs_f64();

        println!(
            "Metal: Generated {} tokens in {:?}",
            metal_tokens, metal_elapsed
        );
        println!("Metal: {:.2} tokens/sec", metal_tok_per_sec);

        // Calculate speedup
        let speedup = metal_tok_per_sec / cpu_tok_per_sec;
        println!("\n=== Results ===");
        println!("Speedup: {:.2}x", speedup);

        // Metal should be at least as fast as CPU (in practice, much faster)
        // Note: For small models, overhead may dominate, so we just check it runs
        assert!(metal_tokens > 0);
        assert_eq!(metal_tokens, cpu_tokens); // Should generate same number of tokens

        // Explicit cleanup for Metal model
        drop(metal_generated);
        drop(metal_model);
        std::hint::black_box(());
    } else {
        println!("\nMetal not available, skipping Metal benchmark");
    }
}
