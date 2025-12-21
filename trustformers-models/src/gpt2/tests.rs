use crate::gpt2::{Gpt2Config, Gpt2LMHeadModel, Gpt2Model};
use trustformers_core::{
    tensor::Tensor,
    traits::{Model, TokenizedInput},
};

#[test]
fn test_gpt2_model_creation() {
    let config = Gpt2Config::default();
    let model = Gpt2Model::new(config.clone()).unwrap();
    assert_eq!(model.get_config().n_layer, 12);
    assert_eq!(model.get_config().n_head, 12);
}

#[test]
fn test_gpt2_lm_head_model_creation() {
    let config = Gpt2Config::default();
    let model = Gpt2LMHeadModel::new(config.clone()).unwrap();
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

    let model = Gpt2Model::new(config).unwrap();
    let input = TokenizedInput {
        input_ids: vec![1, 2, 3, 4, 5],
        attention_mask: vec![1u8; 5],
        token_type_ids: None,
        offset_mapping: None,
        special_tokens_mask: None,
        overflowing_tokens: None,
    };

    let output = model.forward(input).unwrap();
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

    let model = Gpt2LMHeadModel::new(config).unwrap();
    let input = TokenizedInput {
        input_ids: vec![1, 2, 3, 4, 5],
        attention_mask: vec![1u8; 5],
        token_type_ids: None,
        offset_mapping: None,
        special_tokens_mask: None,
        overflowing_tokens: None,
    };

    let output = model.forward(input).unwrap();
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

    let model = Gpt2LMHeadModel::new(config).unwrap();
    let input_ids = vec![1, 2, 3];
    let generated = model.generate_greedy(input_ids.clone(), 10).unwrap();

    assert!(generated.len() >= input_ids.len());
    assert!(generated.len() <= 10);
    assert_eq!(&generated[..3], &input_ids[..]);
}

#[test]
fn test_gpt2_beam_search() {
    let config = Gpt2Config {
        vocab_size: 50,
        n_positions: 64,
        n_embd: 32,
        n_layer: 1,
        n_head: 4,
        ..Default::default()
    };

    let model = Gpt2LMHeadModel::new(config).unwrap();
    let input_ids = vec![1, 2];
    let generated = model.generate_beam_search(input_ids.clone(), 10, 3).unwrap();

    assert!(generated.len() >= input_ids.len());
    assert!(generated.len() <= 10);
    assert_eq!(&generated[..2], &input_ids[..]);
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn test_gpt2_metal_sampling() {
    use crate::gpt2::generation::GenerativeModel;

    // Small model config for testing
    let config = Gpt2Config {
        vocab_size: 100,
        n_positions: 64,
        n_embd: 64,
        n_layer: 2,
        n_head: 4,
        ..Default::default()
    };

    // Test Metal device availability
    let device = Device::metal_if_available(0);
    println!("Using device: {:?}", device);

    // Create model with device
    let model = Gpt2LMHeadModel::new_with_device(config.clone(), device).unwrap();

    // Test input
    let input_ids = vec![1, 2, 3];
    let max_length = 20;
    let k = 10;
    let temperature = 1.0;

    // Measure generation time
    let start = Instant::now();
    let generated = model.generate_top_k(input_ids.clone(), max_length, k, temperature).unwrap();
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
    assert_eq!(&generated[..3], &input_ids[..]);
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn test_gpt2_metal_vs_cpu_performance() {
    use crate::gpt2::generation::GenerativeModel;

    // Small model config for testing
    let config = Gpt2Config {
        vocab_size: 100,
        n_positions: 128,
        n_embd: 128,
        n_layer: 4,
        n_head: 8,
        ..Default::default()
    };

    let input_ids = vec![1, 2, 3, 4, 5];
    let max_length = 25; // Generate 20 new tokens
    let k = 20;
    let temperature = 1.0;

    // CPU benchmark
    println!("\n=== CPU Benchmark ===");
    let cpu_model = Gpt2LMHeadModel::new_with_device(config.clone(), Device::CPU).unwrap();
    let cpu_start = Instant::now();
    let cpu_generated =
        cpu_model.generate_top_k(input_ids.clone(), max_length, k, temperature).unwrap();
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_tokens = cpu_generated.len() - input_ids.len();
    let cpu_tok_per_sec = cpu_tokens as f64 / cpu_elapsed.as_secs_f64();

    println!("CPU: Generated {} tokens in {:?}", cpu_tokens, cpu_elapsed);
    println!("CPU: {:.2} tokens/sec", cpu_tok_per_sec);

    // Metal benchmark (if available)
    let device = Device::metal_if_available(0);
    if matches!(device, Device::Metal(_)) {
        println!("\n=== Metal GPU Benchmark ===");
        let metal_model = Gpt2LMHeadModel::new_with_device(config.clone(), device).unwrap();
        let metal_start = Instant::now();
        let metal_generated = metal_model
            .generate_top_k(input_ids.clone(), max_length, k, temperature)
            .unwrap();
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
    } else {
        println!("\nMetal not available, skipping Metal benchmark");
    }
}
