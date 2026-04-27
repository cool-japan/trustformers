#[cfg(test)]
mod tests {
    use crate::biologically_inspired::config::{
        BiologicalArchitecture, BiologicalConfig, MemoryType,
    };
    use crate::biologically_inspired::neural_turing_machine::*;
    use trustformers_core::tensor::Tensor;

    fn small_ntm_config() -> BiologicalConfig {
        BiologicalConfig {
            architecture: BiologicalArchitecture::NeuralTuringMachine,
            d_model: 32,
            n_layer: 2,
            vocab_size: 100,
            max_position_embeddings: 64,
            memory_capacity: 16,
            memory_type: MemoryType::Working,
            use_bias: true,
            ..BiologicalConfig::default()
        }
    }

    // --- NTMLayer tests ---

    #[test]
    fn test_ntm_layer_creation() {
        let config = small_ntm_config();
        let layer = NTMLayer::new(&config);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_ntm_layer_config() {
        let config = small_ntm_config();
        if let Ok(layer) = NTMLayer::new(&config) {
            assert_eq!(layer.config.d_model, 32);
            assert_eq!(layer.num_read_heads, 1);
            assert_eq!(layer.num_write_heads, 1);
            assert_eq!(layer.memory_width, 32);
        }
    }

    #[test]
    fn test_ntm_layer_no_initial_memory() {
        let config = small_ntm_config();
        if let Ok(layer) = NTMLayer::new(&config) {
            assert!(layer.memory_bank.is_none());
        }
    }

    #[test]
    fn test_ntm_layer_controller_size() {
        let config = small_ntm_config();
        if let Ok(layer) = NTMLayer::new(&config) {
            assert_eq!(layer.read_head_controllers.len(), 1);
            assert_eq!(layer.write_head_controllers.len(), 1);
            assert_eq!(layer.erase_head_controllers.len(), 1);
            assert_eq!(layer.add_head_controllers.len(), 1);
        }
    }

    // --- NTMMemoryBank tests ---

    #[test]
    fn test_ntm_memory_bank_creation() {
        if let Ok(memory) = Tensor::zeros(&[16, 32]) {
            if let Ok(attn_weights) = Tensor::zeros(&[16]) {
                let head = NTMHead {
                    attention_weights: attn_weights.clone(),
                    prev_attention_weights: attn_weights.clone(),
                    key: Tensor::zeros(&[32]).unwrap_or_else(|_| attn_weights.clone()),
                    key_strength: 1.0,
                    interpolation_gate: 0.5,
                    shift_weights: Tensor::zeros(&[3]).unwrap_or_else(|_| attn_weights.clone()),
                    sharpening_factor: 1.0,
                };
                let bank = NTMMemoryBank {
                    memory,
                    read_heads: vec![head.clone()],
                    write_heads: vec![head],
                    memory_size: (16, 32),
                };
                assert_eq!(bank.memory_size, (16, 32));
                assert_eq!(bank.read_heads.len(), 1);
                assert_eq!(bank.write_heads.len(), 1);
            }
        }
    }

    // --- NTMHead tests ---

    #[test]
    fn test_ntm_head_creation() {
        if let Ok(attn_weights) = Tensor::zeros(&[8]) {
            if let Ok(key) = Tensor::zeros(&[16]) {
                if let Ok(shift) = Tensor::zeros(&[3]) {
                    let head = NTMHead {
                        attention_weights: attn_weights.clone(),
                        prev_attention_weights: attn_weights,
                        key,
                        key_strength: 1.5,
                        interpolation_gate: 0.7,
                        shift_weights: shift,
                        sharpening_factor: 2.0,
                    };
                    assert!((head.key_strength - 1.5).abs() < f32::EPSILON);
                    assert!((head.interpolation_gate - 0.7).abs() < f32::EPSILON);
                    assert!((head.sharpening_factor - 2.0).abs() < f32::EPSILON);
                }
            }
        }
    }

    #[test]
    fn test_ntm_head_clone() {
        if let Ok(attn_weights) = Tensor::zeros(&[4]) {
            if let Ok(key) = Tensor::zeros(&[8]) {
                if let Ok(shift) = Tensor::zeros(&[3]) {
                    let head = NTMHead {
                        attention_weights: attn_weights.clone(),
                        prev_attention_weights: attn_weights,
                        key,
                        key_strength: 1.0,
                        interpolation_gate: 0.5,
                        shift_weights: shift,
                        sharpening_factor: 1.0,
                    };
                    let cloned = head.clone();
                    assert!((cloned.key_strength - 1.0).abs() < f32::EPSILON);
                }
            }
        }
    }

    // --- NeuralTuringMachine tests ---

    #[test]
    fn test_neural_turing_machine_creation() {
        let config = small_ntm_config();
        let ntm = NeuralTuringMachine::new(&config);
        assert!(ntm.is_ok());
    }

    // --- Config tests for NTM ---

    #[test]
    fn test_ntm_config_builder() {
        let config = BiologicalConfig::neural_turing_machine();
        assert!(matches!(
            config.architecture,
            BiologicalArchitecture::NeuralTuringMachine
        ));
        assert_eq!(config.memory_capacity, 128);
        assert!(matches!(config.memory_type, MemoryType::Working));
    }

    #[test]
    fn test_ntm_config_default_d_model() {
        let config = BiologicalConfig::neural_turing_machine();
        assert_eq!(config.d_model, 768);
    }

    #[test]
    fn test_ntm_config_custom_memory_capacity() {
        let config = BiologicalConfig {
            memory_capacity: 256,
            ..BiologicalConfig::neural_turing_machine()
        };
        assert_eq!(config.memory_capacity, 256);
    }

    #[test]
    fn test_ntm_layer_different_configs() {
        let configs = vec![
            BiologicalConfig {
                d_model: 16,
                memory_capacity: 8,
                use_bias: true,
                ..BiologicalConfig::neural_turing_machine()
            },
            BiologicalConfig {
                d_model: 64,
                memory_capacity: 32,
                use_bias: false,
                ..BiologicalConfig::neural_turing_machine()
            },
        ];
        for config in configs {
            let layer = NTMLayer::new(&config);
            assert!(layer.is_ok());
        }
    }

    #[test]
    fn test_ntm_memory_bank_clone() {
        if let Ok(memory) = Tensor::zeros(&[4, 8]) {
            if let Ok(attn) = Tensor::zeros(&[4]) {
                if let Ok(key) = Tensor::zeros(&[8]) {
                    if let Ok(shift) = Tensor::zeros(&[3]) {
                        let head = NTMHead {
                            attention_weights: attn.clone(),
                            prev_attention_weights: attn,
                            key,
                            key_strength: 1.0,
                            interpolation_gate: 0.5,
                            shift_weights: shift,
                            sharpening_factor: 1.0,
                        };
                        let bank = NTMMemoryBank {
                            memory,
                            read_heads: vec![head.clone()],
                            write_heads: vec![head],
                            memory_size: (4, 8),
                        };
                        let cloned = bank.clone();
                        assert_eq!(cloned.memory_size, (4, 8));
                    }
                }
            }
        }
    }

    #[test]
    fn test_ntm_head_key_strength_range() {
        if let Ok(attn) = Tensor::zeros(&[4]) {
            if let Ok(key) = Tensor::zeros(&[8]) {
                if let Ok(shift) = Tensor::zeros(&[3]) {
                    for strength in [0.1_f32, 1.0, 5.0, 10.0] {
                        let head = NTMHead {
                            attention_weights: attn.clone(),
                            prev_attention_weights: attn.clone(),
                            key: key.clone(),
                            key_strength: strength,
                            interpolation_gate: 0.5,
                            shift_weights: shift.clone(),
                            sharpening_factor: 1.0,
                        };
                        assert!((head.key_strength - strength).abs() < f32::EPSILON);
                    }
                }
            }
        }
    }

    #[test]
    fn test_ntm_head_interpolation_gate_bounds() {
        if let Ok(attn) = Tensor::zeros(&[4]) {
            if let Ok(key) = Tensor::zeros(&[8]) {
                if let Ok(shift) = Tensor::zeros(&[3]) {
                    for gate in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
                        let head = NTMHead {
                            attention_weights: attn.clone(),
                            prev_attention_weights: attn.clone(),
                            key: key.clone(),
                            key_strength: 1.0,
                            interpolation_gate: gate,
                            shift_weights: shift.clone(),
                            sharpening_factor: 1.0,
                        };
                        assert!(head.interpolation_gate >= 0.0);
                        assert!(head.interpolation_gate <= 1.0);
                    }
                }
            }
        }
    }
}
