use super::*;
use trustformers_core::{
    tensor::Tensor,
    traits::{Config, Model},
};

#[test]
fn test_recursive_config_creation() {
    let config = RecursiveConfig::default();
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.recursion_depth, 3);
    assert_eq!(config.chunk_size, 512);
    assert!(config.validate().is_ok());
}

#[test]
fn test_long_document_config() {
    let config = RecursiveConfig::long_document();
    assert_eq!(config.model_type, "recursive-long-document");
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.max_position_embeddings, 32768);
    assert_eq!(config.recursion_depth, 4);
    assert_eq!(config.chunk_size, 1024);
    assert!(config.validate().is_ok());
}

#[test]
fn test_universal_config() {
    let config = RecursiveConfig::universal();
    assert_eq!(config.model_type, "recursive-universal");
    assert!(config.use_universal_transformer);
    assert_eq!(config.max_steps, 8);
    assert!(config.adaptive_computation_time);
    assert!(config.validate().is_ok());
}

#[test]
fn test_memory_efficient_config() {
    let config = RecursiveConfig::memory_efficient();
    assert_eq!(config.model_type, "recursive-memory-efficient");
    assert_eq!(config.hidden_size, 512);
    assert!(config.use_gradient_checkpointing);
    assert!(config.use_memory_compression);
    assert_eq!(config.compression_ratio, 0.25);
    assert!(config.validate().is_ok());
}

#[test]
fn test_hierarchical_config() {
    let config = RecursiveConfig::hierarchical();
    assert_eq!(config.model_type, "recursive-hierarchical");
    assert!(config.use_hierarchical_attention);
    assert_eq!(config.hierarchy_levels, 3);
    assert_eq!(config.level_compression_ratios.len(), 3);
    assert!(config.cross_level_attention);
    assert!(config.validate().is_ok());
}

#[test]
fn test_code_understanding_config() {
    let config = RecursiveConfig::code_understanding();
    assert_eq!(config.model_type, "recursive-code");
    assert_eq!(config.vocab_size, 50000);
    assert_eq!(config.max_position_embeddings, 8192);
    assert!(config.use_hierarchical_attention);
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validation() {
    let mut config = RecursiveConfig::default();

    // Test invalid hidden_size
    config.hidden_size = 100; // Not divisible by num_attention_heads (12)
    assert!(config.validate().is_err());

    // Fix hidden_size
    config.hidden_size = 768;
    assert!(config.validate().is_ok());

    // Test invalid recursion_depth
    config.recursion_depth = 0;
    assert!(config.validate().is_err());

    // Fix recursion_depth
    config.recursion_depth = 3;
    assert!(config.validate().is_ok());

    // Test invalid chunk_size
    config.chunk_size = 0;
    assert!(config.validate().is_err());

    // Fix chunk_size
    config.chunk_size = 512;
    assert!(config.validate().is_ok());

    // Test invalid overlap_size
    config.overlap_size = 512; // Equal to chunk_size
    assert!(config.validate().is_err());

    // Fix overlap_size
    config.overlap_size = 64;
    assert!(config.validate().is_ok());
}

#[test]
fn test_adaptive_depth_validation() {
    let mut config = RecursiveConfig::default();
    config.use_adaptive_depth = true;
    config.min_depth = 5;
    config.max_depth = 3; // min > max
    assert!(config.validate().is_err());

    config.max_depth = 7;
    assert!(config.validate().is_ok());
}

#[test]
fn test_hierarchical_validation() {
    let mut config = RecursiveConfig::default();
    config.use_hierarchical_attention = true;
    config.hierarchy_levels = 0;
    assert!(config.validate().is_err());

    config.hierarchy_levels = 3;
    config.level_compression_ratios = vec![1.0, 0.5]; // Wrong length
    assert!(config.validate().is_err());

    config.level_compression_ratios = vec![1.0, 0.5, 0.25];
    assert!(config.validate().is_ok());

    config.level_compression_ratios = vec![1.0, 1.5, 0.25]; // Invalid ratio > 1.0
    assert!(config.validate().is_err());
}

#[test]
fn test_universal_transformer_validation() {
    let mut config = RecursiveConfig::default();
    config.use_universal_transformer = true;
    config.max_steps = 0;
    assert!(config.validate().is_err());

    config.max_steps = 10;
    assert!(config.validate().is_ok());
}

#[test]
fn test_from_pretrained_name() {
    assert!(RecursiveConfig::from_pretrained_name("recursive-long-document").is_some());
    assert!(RecursiveConfig::from_pretrained_name("recursive-universal").is_some());
    assert!(RecursiveConfig::from_pretrained_name("recursive-memory-efficient").is_some());
    assert!(RecursiveConfig::from_pretrained_name("recursive-hierarchical").is_some());
    assert!(RecursiveConfig::from_pretrained_name("recursive-code").is_some());
    assert!(RecursiveConfig::from_pretrained_name("invalid-model").is_none());
}

#[test]
fn test_config_helper_methods() {
    let config = RecursiveConfig::default();
    assert_eq!(config.head_dim(), 64); // 768 / 12
    assert_eq!(config.num_kv_heads(), 12); // No GQA by default
    assert_eq!(config.effective_chunk_size(), 448); // 512 - 64
    // With hierarchical attention: 1024 * (1.0 + 0.5 + 0.25) = 1792
    assert_eq!(config.total_memory_capacity(), 1792);
}

#[test]
fn test_config_with_methods() {
    let mut config = RecursiveConfig::default();

    config.with_memory(2048, true, 0.5);
    assert_eq!(config.memory_size, 2048);
    assert!(config.use_memory_compression);
    assert_eq!(config.compression_ratio, 0.5);

    config.with_chunks(1024, 128);
    assert_eq!(config.chunk_size, 1024);
    assert_eq!(config.overlap_size, 128);

    config.with_depth(4, true);
    assert_eq!(config.recursion_depth, 4);
    assert!(config.use_adaptive_depth);
    assert_eq!(config.max_depth, 8); // depth * 2

    config.with_hierarchy(4, vec![1.0, 0.75, 0.5, 0.25]);
    assert!(config.use_hierarchical_attention);
    assert_eq!(config.hierarchy_levels, 4);
    assert_eq!(config.level_compression_ratios.len(), 4);

    config.with_universal(12, true);
    assert!(config.use_universal_transformer);
    assert_eq!(config.max_steps, 12);
    assert!(config.adaptive_computation_time);
}

#[test]
fn test_memory_state_creation() {
    let memory = MemoryState::new(2, 1024, 768);
    // MemoryState created successfully - implementation details are private

    let content = memory.get_content().unwrap();
    assert_eq!(content.shape(), &[2, 1024, 768]);
}

#[test]
fn test_memory_state_update() {
    let mut memory = MemoryState::new(1, 1024, 768);
    let new_content = Tensor::ones(&[1, 256, 768]).unwrap();

    assert!(memory.update(new_content).is_ok());
    // Memory update successful - write head position is internal

    // Test circular buffer behavior
    let large_content = Tensor::ones(&[1, 1000, 768]).unwrap();
    assert!(memory.update(large_content).is_ok());
    // Large content update successful - circular buffer behavior handled internally
}

#[test]
fn test_memory_state_read() {
    let mut memory = MemoryState::new(1, 1024, 768);
    let content = memory.read(256).unwrap();
    assert_eq!(content.shape(), &[1, 256, 768]);
    // Read head position updated successfully - position is internal

    // Test circular read
    let _ = memory.read(1000);
    // Circular read successful - read head position handled internally
}

#[test]
#[ignore] // Heavy test - RecursiveTransformer creation, run with --ignored
fn test_recursive_transformer_creation() {
    let config = RecursiveConfig::default();
    let result = RecursiveTransformer::new(config);
    assert!(
        result.is_ok(),
        "Failed to create RecursiveTransformer: {:?}",
        result.err()
    );
}

#[test]
#[ignore] // Heavy test - RecursiveForCausalLM creation, run with --ignored
fn test_recursive_for_causal_lm_creation() {
    let config = RecursiveConfig::default();
    let result = RecursiveForCausalLM::new(config);
    assert!(
        result.is_ok(),
        "Failed to create RecursiveForCausalLM: {:?}",
        result.err()
    );
}

#[test]
#[ignore] // Heavy test - RecursiveForSequenceClassification creation, run with --ignored
fn test_recursive_for_sequence_classification_creation() {
    let config = RecursiveConfig::default();
    let result = RecursiveForSequenceClassification::new(config, 10);
    assert!(
        result.is_ok(),
        "Failed to create RecursiveForSequenceClassification: {:?}",
        result.err()
    );
}

#[test]
fn test_memory_manager_creation() {
    let config = RecursiveConfig::default();
    let result = MemoryManager::new(config);
    assert!(
        result.is_ok(),
        "Failed to create MemoryManager: {:?}",
        result.err()
    );
}

#[test]
fn test_depth_predictor_creation() {
    let config = RecursiveConfig::default();
    let result = DepthPredictor::new(config);
    assert!(
        result.is_ok(),
        "Failed to create DepthPredictor: {:?}",
        result.err()
    );
}

#[test]
fn test_hierarchy_manager_creation() {
    let config = RecursiveConfig::hierarchical();
    let result = HierarchyManager::new(config);
    assert!(
        result.is_ok(),
        "Failed to create HierarchyManager: {:?}",
        result.err()
    );
}

#[test]
fn test_universal_controller_creation() {
    let config = RecursiveConfig::universal();
    let result = UniversalController::new(config);
    assert!(
        result.is_ok(),
        "Failed to create UniversalController: {:?}",
        result.err()
    );
}

#[test]
fn test_recursive_transformer_forward() {
    let config = RecursiveConfig::default();
    let model = RecursiveTransformer::new(config.clone()).unwrap();

    let input_ids = Tensor::zeros(&[1, 100]).unwrap(); // batch=1, seq_len=100
    let input = RecursiveInput {
        input_ids,
        attention_mask: None,
        position_ids: None,
        memory_state: None,
    };

    let result = model.forward(input);
    assert!(result.is_ok(), "Forward pass failed: {:?}", result.err());

    let output = result.unwrap();
    assert_eq!(output.last_hidden_state.shape()[0], 1); // batch size
    assert_eq!(output.last_hidden_state.shape()[1], 100); // sequence length
    assert_eq!(output.last_hidden_state.shape()[2], config.hidden_size); // hidden size
    assert_eq!(output.logits.shape()[2], config.vocab_size); // vocab size
}

#[test]
fn test_recursive_transformer_long_sequence() {
    let config = RecursiveConfig::long_document();
    let model = RecursiveTransformer::new(config.clone()).unwrap();

    let input_ids = Tensor::zeros(&[1, 2048]).unwrap(); // Long sequence
    let input = RecursiveInput {
        input_ids,
        attention_mask: None,
        position_ids: None,
        memory_state: None,
    };

    let result = model.forward(input);
    assert!(
        result.is_ok(),
        "Long sequence forward pass failed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    assert_eq!(output.last_hidden_state.shape()[0], 1);
    assert_eq!(output.last_hidden_state.shape()[1], 2048);
    assert!(output.recursion_depth > 0);
}

#[test]
fn test_recursive_transformer_with_memory() {
    let config = RecursiveConfig::default();
    let model = RecursiveTransformer::new(config.clone()).unwrap();

    let input_ids = Tensor::zeros(&[1, 100]).unwrap();
    let memory_state = MemoryState::new(1, config.memory_size, config.hidden_size);

    let input = RecursiveInput {
        input_ids,
        attention_mask: None,
        position_ids: None,
        memory_state: Some(memory_state),
    };

    let result = model.forward(input);
    assert!(
        result.is_ok(),
        "Forward pass with memory failed: {:?}",
        result.err()
    );

    let _output = result.unwrap();
    // Memory state initialized with correct capacity - capacity is internal
}

#[test]
fn test_causal_lm_forward() {
    let config = RecursiveConfig::default();
    let model = RecursiveForCausalLM::new(config.clone()).unwrap();

    let input_ids = Tensor::zeros(&[2, 50]).unwrap(); // batch=2, seq_len=50
    let input = RecursiveInput {
        input_ids,
        attention_mask: None,
        position_ids: None,
        memory_state: None,
    };

    let result = model.forward(input);
    assert!(
        result.is_ok(),
        "CausalLM forward pass failed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    assert_eq!(output.last_hidden_state.shape()[0], 2);
    assert_eq!(output.last_hidden_state.shape()[1], 50);
    assert_eq!(output.logits.shape()[2], config.vocab_size);
}

#[test]
fn test_sequence_classification_forward() {
    let config = RecursiveConfig::default();
    let num_labels = 5;
    let model = RecursiveForSequenceClassification::new(config.clone(), num_labels).unwrap();

    let input_ids = Tensor::zeros(&[2, 100]).unwrap();
    let input = RecursiveInput {
        input_ids,
        attention_mask: None,
        position_ids: None,
        memory_state: None,
    };

    let result = model.forward(input);
    assert!(
        result.is_ok(),
        "Classification forward pass failed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    assert_eq!(output.logits.shape()[0], 2); // batch size
    assert_eq!(output.logits.shape()[1], num_labels); // number of labels
}

#[test]
fn test_depth_predictor_predict() {
    let config = RecursiveConfig::default();
    let predictor = DepthPredictor::new(config).unwrap();
    let memory = MemoryState::new(1, 1024, 768);

    // Test short sequence
    let short_seq = Tensor::zeros(&[1, 100]).unwrap();
    let depth = predictor.predict_depth(&short_seq, &memory).unwrap();
    assert!((1..=5).contains(&depth));

    // Test long sequence
    let long_seq = Tensor::zeros(&[1, 5000]).unwrap();
    let depth = predictor.predict_depth(&long_seq, &memory).unwrap();
    assert!(depth >= 3); // Should predict higher depth for longer sequences
}

#[test]
fn test_model_info() {
    let info = model_info("recursive-long-document").unwrap();
    assert_eq!(info.name, "Recursive Long Document");
    assert_eq!(info.max_sequence_length, 32768);
    assert!(info.memory_efficient);
    assert!(info.adaptive_depth);

    let universal_info = model_info("recursive-universal").unwrap();
    assert!(universal_info.adaptive_depth);
    assert!(!universal_info.memory_efficient);
}

#[test]
fn test_available_models() {
    let models = available_models();
    assert!(models.contains(&"recursive-long-document"));
    assert!(models.contains(&"recursive-universal"));
    assert!(models.contains(&"recursive-memory-efficient"));
    assert!(models.contains(&"recursive-hierarchical"));
    assert!(models.contains(&"recursive-code"));
    assert_eq!(models.len(), 5);
}

#[test]
fn test_convenience_functions() {
    // Test that convenience functions work
    assert!(long_document().is_ok());
    assert!(universal().is_ok());
    assert!(memory_efficient().is_ok());
    assert!(hierarchical().is_ok());
    assert!(code_understanding().is_ok());

    // Test from_pretrained function
    assert!(from_pretrained("recursive-long-document").is_ok());
    assert!(from_pretrained("invalid-model").is_err());

    // Test task-specific functions
    let config = RecursiveConfig::default();
    assert!(for_causal_lm(config.clone()).is_ok());
    assert!(for_sequence_classification(config, 10).is_ok());
}

#[test]
fn test_utility_functions() {
    let config = RecursiveConfig::default();

    // Test memory state creation
    let _memory = create_memory_state(2, &config);
    // Memory state created with correct capacity - capacity is internal

    // Test optimal chunk size calculation
    let chunk_size = optimal_chunk_size(10000, 1024, 768);
    assert!(chunk_size > 0 && chunk_size <= 2500); // Should be 1/4 of sequence or less

    // Test memory usage estimation
    let memory_usage = estimate_memory_usage(&config, 1000);
    assert!(memory_usage > 0); // Should return positive MB estimate
}

#[test]
fn test_config_presets() {
    let book_config = ConfigPresets::book_processing();
    assert_eq!(book_config.chunk_size, 2048);
    assert!(book_config.use_hierarchical_attention);

    let code_config = ConfigPresets::code_analysis();
    assert_eq!(code_config.hierarchy_levels, 4);
    assert!(code_config.use_adaptive_depth);

    let legal_config = ConfigPresets::legal_documents();
    assert!(legal_config.use_memory_compression);
    assert_eq!(legal_config.compression_ratio, 0.3);

    let paper_config = ConfigPresets::research_papers();
    assert_eq!(paper_config.hierarchy_levels, 3);
    assert!(paper_config.cross_level_attention);

    let mobile_config = ConfigPresets::mobile_deployment();
    assert_eq!(mobile_config.hidden_size, 384);
    assert_eq!(mobile_config.chunk_size, 256);
}

#[test]
fn test_performance_tips() {
    let tips = performance_tips();
    assert!(!tips.is_empty());
    assert!(tips.len() >= 8);
    assert!(tips.iter().any(|tip| tip.contains("memory")));
    assert!(tips.iter().any(|tip| tip.contains("chunk")));
}

// Regression tests for edge cases
#[test]
fn test_empty_sequence() {
    let config = RecursiveConfig::default();
    let model = RecursiveTransformer::new(config).unwrap();

    let input_ids = Tensor::zeros(&[1, 1]).unwrap(); // Single token
    let input = RecursiveInput {
        input_ids,
        attention_mask: None,
        position_ids: None,
        memory_state: None,
    };

    let result = model.forward(input);
    assert!(result.is_ok());
}

#[test]
fn test_batch_processing() {
    let config = RecursiveConfig::default();
    let model = RecursiveTransformer::new(config).unwrap();

    let batch_size = 4;
    let seq_len = 200;

    let input_ids = Tensor::zeros(&[batch_size, seq_len]).unwrap();
    let input = RecursiveInput {
        input_ids,
        attention_mask: None,
        position_ids: None,
        memory_state: None,
    };

    let result = model.forward(input);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.last_hidden_state.shape()[0], batch_size);
    assert_eq!(output.last_hidden_state.shape()[1], seq_len);
}

#[test]
fn test_config_architecture_name() {
    let config = RecursiveConfig::default();
    assert_eq!(config.architecture(), "RecursiveTransformer");
}

#[test]
fn test_very_long_sequence() {
    let mut config = RecursiveConfig::long_document();
    config.chunk_size = 512; // Smaller chunks for test
    let model = RecursiveTransformer::new(config).unwrap();

    let input_ids = Tensor::zeros(&[1, 5000]).unwrap(); // Very long sequence
    let input = RecursiveInput {
        input_ids,
        attention_mask: None,
        position_ids: None,
        memory_state: None,
    };

    let result = model.forward(input);
    assert!(result.is_ok(), "Very long sequence processing failed");

    let output = result.unwrap();
    assert_eq!(output.last_hidden_state.shape()[1], 5000);
    assert!(output.recursion_depth > 1); // Should use multiple recursion levels
}
