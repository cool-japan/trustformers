//! Modern Architecture Support for TrustformeRS C API
//!
//! This module provides support for cutting-edge neural architectures including:
//! - Mamba/State Space Models (SSMs) for efficient sequence modeling
//! - Modern attention mechanisms (RoPE, GQA, SwiGLU)
//! - Mixture of Experts (MoE) architectures
//! - Advanced normalization techniques (RMSNorm, LayerNorm variants)

use crate::error::{TrustformersError, TrustformersResult};
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_float, c_int};
use std::ptr;

/// Modern architecture types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum ModernArchitectureType {
    /// Traditional Transformer architecture
    Transformer = 0,
    /// Mamba/State Space Model architecture
    Mamba = 1,
    /// Mixture of Experts (MoE) architecture
    MixtureOfExperts = 2,
    /// RetNet (Retentive Network)
    RetNet = 3,
    /// RWKV (Receptance Weighted Key Value)
    RWKV = 4,
    /// Hyena architecture
    Hyena = 5,
    /// Linformer (Linear attention)
    Linformer = 6,
    /// Performer (FAVOR+ attention)
    Performer = 7,
}

/// State Space Model configuration for Mamba
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct MambaConfig {
    /// Model dimension
    pub d_model: i32,
    /// State space dimension
    pub d_state: i32,
    /// Convolution kernel size
    pub d_conv: i32,
    /// Expansion factor for MLP
    pub expand_factor: i32,
    /// Number of layers
    pub num_layers: i32,
    /// Vocabulary size
    pub vocab_size: i32,
    /// Maximum sequence length
    pub max_seq_length: i32,
    /// Bias in linear layers
    pub use_bias: c_int,
    /// Normalization type: 0=LayerNorm, 1=RMSNorm
    pub norm_type: c_int,
    /// Activation function: 0=SiLU, 1=GELU, 2=ReLU
    pub activation_fn: c_int,
    /// Whether to use selective SSM
    pub use_selective_ssm: c_int,
    /// Delta parameter initialization
    pub delta_init_method: c_int,
    /// A parameter initialization method
    pub a_init_method: c_int,
    /// D parameter initialization method
    pub d_init_method: c_int,
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            d_model: 2048,
            d_state: 16,
            d_conv: 4,
            expand_factor: 2,
            num_layers: 24,
            vocab_size: 32000,
            max_seq_length: 4096,
            use_bias: 0,          // False
            norm_type: 1,         // RMSNorm
            activation_fn: 0,     // SiLU
            use_selective_ssm: 1, // True
            delta_init_method: 0, // Random
            a_init_method: 0,     // Random
            d_init_method: 0,     // Random
        }
    }
}

/// Mixture of Experts configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct MoEConfig {
    /// Number of experts
    pub num_experts: i32,
    /// Number of experts to activate per token
    pub top_k: i32,
    /// Expert capacity factor
    pub capacity_factor: f32,
    /// Load balancing loss weight
    pub load_balance_loss_weight: f32,
    /// Router z-loss weight
    pub router_z_loss_weight: f32,
    /// Expert dropout rate
    pub expert_dropout: f32,
    /// Whether to use aux loss
    pub use_aux_loss: c_int,
    /// Gating network type: 0=Linear, 1=SwitchTransformer, 2=GLaM
    pub gating_type: c_int,
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1.25,
            load_balance_loss_weight: 0.01,
            router_z_loss_weight: 0.001,
            expert_dropout: 0.1,
            use_aux_loss: 1, // True
            gating_type: 0,  // Linear
        }
    }
}

/// Modern attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct ModernAttentionConfig {
    /// Attention type: 0=MultiHead, 1=GroupedQuery, 2=MultiQuery
    pub attention_type: c_int,
    /// Number of attention heads
    pub num_heads: i32,
    /// Number of key-value heads (for GQA/MQA)
    pub num_kv_heads: i32,
    /// Head dimension
    pub head_dim: i32,
    /// Whether to use RoPE (Rotary Positional Embedding)
    pub use_rope: c_int,
    /// RoPE base frequency
    pub rope_base: f32,
    /// RoPE scaling factor
    pub rope_scaling_factor: f32,
    /// Whether to use Flash Attention
    pub use_flash_attention: c_int,
    /// Attention dropout rate
    pub attention_dropout: f32,
    /// Whether to use sliding window attention
    pub use_sliding_window: c_int,
    /// Sliding window size
    pub sliding_window_size: i32,
}

impl Default for ModernAttentionConfig {
    fn default() -> Self {
        Self {
            attention_type: 1, // GroupedQuery
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            use_rope: 1, // True
            rope_base: 10000.0,
            rope_scaling_factor: 1.0,
            use_flash_attention: 1, // True
            attention_dropout: 0.0,
            use_sliding_window: 0, // False
            sliding_window_size: 4096,
        }
    }
}

/// Modern architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModernArchitectureConfig {
    /// Architecture type
    pub arch_type: ModernArchitectureType,
    /// Mamba-specific configuration
    pub mamba_config: Option<MambaConfig>,
    /// MoE-specific configuration
    pub moe_config: Option<MoEConfig>,
    /// Modern attention configuration
    pub attention_config: Option<ModernAttentionConfig>,
    /// General model configuration
    pub general_config: GeneralModelConfig,
}

/// General model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct GeneralModelConfig {
    /// Model dimension
    pub d_model: i32,
    /// Feed-forward dimension
    pub d_ff: i32,
    /// Number of layers
    pub num_layers: i32,
    /// Vocabulary size
    pub vocab_size: i32,
    /// Maximum sequence length
    pub max_seq_length: i32,
    /// Dropout rate
    pub dropout: f32,
    /// Layer dropout rate
    pub layer_dropout: f32,
    /// Normalization epsilon
    pub norm_eps: f32,
    /// Tie input/output embeddings
    pub tie_embeddings: c_int,
    /// Use gradient checkpointing
    pub use_gradient_checkpointing: c_int,
}

impl Default for GeneralModelConfig {
    fn default() -> Self {
        Self {
            d_model: 2048,
            d_ff: 8192,
            num_layers: 24,
            vocab_size: 32000,
            max_seq_length: 4096,
            dropout: 0.1,
            layer_dropout: 0.0,
            norm_eps: 1e-5,
            tie_embeddings: 1,             // True
            use_gradient_checkpointing: 0, // False
        }
    }
}

/// Modern architecture handle
pub type TrustformersModernArchitecture = usize;

/// State Space Model operations for Mamba
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSMOperations {
    /// State transition matrices (A parameters)
    pub a_matrices: Vec<Vec<f32>>,
    /// Input matrices (B parameters)
    pub b_matrices: Vec<Vec<f32>>,
    /// Output matrices (C parameters)
    pub c_matrices: Vec<Vec<f32>>,
    /// Skip connection parameters (D parameters)
    pub d_parameters: Vec<f32>,
    /// Delta parameters for selective SSM
    pub delta_parameters: Vec<f32>,
    /// Convolution parameters
    pub conv_parameters: Vec<Vec<f32>>,
}

impl SSMOperations {
    pub fn new(d_model: usize, d_state: usize, d_conv: usize) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple deterministic "random" initialization for testing
        let mut hasher = DefaultHasher::new();
        let seed = d_model * d_state * d_conv;
        seed.hash(&mut hasher);
        let hash_val = hasher.finish();

        let mut a_matrices = vec![vec![0.0; d_state]; d_model];
        let mut b_matrices = vec![vec![0.0; d_state]; d_model];
        let mut c_matrices = vec![vec![0.0; d_state]; d_model];
        let mut d_parameters = vec![0.0; d_model];
        let mut delta_parameters = vec![0.0; d_model];
        let mut conv_parameters = vec![vec![0.0; d_conv]; d_model];

        // Initialize with deterministic values for reproducible tests
        for d in 0..d_model {
            for s in 0..d_state {
                let idx = (d * d_state + s) as u64;
                // A matrices: small negative values for stability
                a_matrices[d][s] =
                    -0.1 - 0.01 * ((hash_val.wrapping_add(idx)) % 100) as f32 / 100.0;
                // B matrices: small positive values
                b_matrices[d][s] =
                    0.1 + 0.01 * ((hash_val.wrapping_add(idx * 2)) % 100) as f32 / 100.0;
                // C matrices: small values around zero
                c_matrices[d][s] =
                    0.05 * (((hash_val.wrapping_add(idx * 3)) % 200) as f32 / 100.0 - 1.0);
            }
            // Initialize D parameters with small positive values
            d_parameters[d] = 0.1 + 0.01 * ((hash_val.wrapping_add(d as u64)) % 50) as f32 / 100.0;
            // Initialize delta parameters with positive values
            delta_parameters[d] =
                0.1 + 0.05 * ((hash_val.wrapping_add(d as u64 * 7)) % 100) as f32 / 100.0;

            // Initialize conv parameters
            for c in 0..d_conv {
                let conv_idx = (d * d_conv + c) as u64;
                conv_parameters[d][c] =
                    0.1 * (((hash_val.wrapping_add(conv_idx * 5)) % 200) as f32 / 100.0 - 1.0);
            }
        }

        Self {
            a_matrices,
            b_matrices,
            c_matrices,
            d_parameters,
            delta_parameters,
            conv_parameters,
        }
    }

    /// Perform SSM forward pass
    pub fn forward(&self, input: &[f32], state: &mut [f32]) -> Vec<f32> {
        let seq_len = input.len() / self.a_matrices.len();
        let d_model = self.a_matrices.len();
        let d_state = self.a_matrices[0].len();

        let mut output = vec![0.0; input.len()];

        for t in 0..seq_len {
            for d in 0..d_model {
                let input_val = input[t * d_model + d];

                // Apply convolution (simplified 1D conv)
                let conv_out = if t >= self.conv_parameters[d].len() {
                    let mut conv_sum = 0.0;
                    for k in 0..self.conv_parameters[d].len() {
                        if t >= k {
                            conv_sum += input[(t - k) * d_model + d] * self.conv_parameters[d][k];
                        }
                    }
                    conv_sum
                } else {
                    input_val
                };

                // Selective SSM computation
                let delta = self.delta_parameters[d].max(0.001); // Ensure positive delta

                // Discretize continuous parameters (simplified Euler method)
                let mut new_state = vec![0.0; d_state];
                for s in 0..d_state {
                    // x_t+1 = A * x_t + B * u_t
                    new_state[s] = self.a_matrices[d][s] * state[d * d_state + s] * delta
                        + self.b_matrices[d][s] * conv_out;

                    // Update state
                    state[d * d_state + s] = new_state[s];
                }

                // Output: y_t = C * x_t + D * u_t
                let mut y = self.d_parameters[d] * conv_out;
                for s in 0..d_state {
                    y += self.c_matrices[d][s] * state[d * d_state + s];
                }

                output[t * d_model + d] = y;
            }
        }

        output
    }
}

/// Create a modern architecture configuration
#[no_mangle]
pub extern "C" fn trustformers_modern_arch_config_create(
    arch_type: ModernArchitectureType,
) -> *mut ModernArchitectureConfig {
    let config = match arch_type {
        ModernArchitectureType::Mamba => ModernArchitectureConfig {
            arch_type,
            mamba_config: Some(MambaConfig::default()),
            moe_config: None,
            attention_config: None,
            general_config: GeneralModelConfig::default(),
        },
        ModernArchitectureType::MixtureOfExperts => ModernArchitectureConfig {
            arch_type,
            mamba_config: None,
            moe_config: Some(MoEConfig::default()),
            attention_config: Some(ModernAttentionConfig::default()),
            general_config: GeneralModelConfig::default(),
        },
        _ => ModernArchitectureConfig {
            arch_type,
            mamba_config: None,
            moe_config: None,
            attention_config: Some(ModernAttentionConfig::default()),
            general_config: GeneralModelConfig::default(),
        },
    };

    Box::into_raw(Box::new(config))
}

/// Free a modern architecture configuration
#[no_mangle]
pub extern "C" fn trustformers_modern_arch_config_free(config: *mut ModernArchitectureConfig) {
    if !config.is_null() {
        unsafe {
            let _ = Box::from_raw(config);
        }
    }
}

/// Configure Mamba-specific parameters
#[no_mangle]
pub extern "C" fn trustformers_configure_mamba(
    config: *mut ModernArchitectureConfig,
    mamba_config: *const MambaConfig,
) -> TrustformersError {
    if config.is_null() || mamba_config.is_null() {
        return TrustformersError::NullPointer;
    }

    unsafe {
        let arch_config = &mut *config;
        let mamba_cfg = &*mamba_config;
        arch_config.mamba_config = Some(mamba_cfg.clone());
    }

    TrustformersError::Success
}

/// Configure MoE-specific parameters
#[no_mangle]
pub extern "C" fn trustformers_configure_moe(
    config: *mut ModernArchitectureConfig,
    moe_config: *const MoEConfig,
) -> TrustformersError {
    if config.is_null() || moe_config.is_null() {
        return TrustformersError::NullPointer;
    }

    unsafe {
        let arch_config = &mut *config;
        let moe_cfg = &*moe_config;
        arch_config.moe_config = Some(moe_cfg.clone());
    }

    TrustformersError::Success
}

/// Configure modern attention parameters
#[no_mangle]
pub extern "C" fn trustformers_configure_attention(
    config: *mut ModernArchitectureConfig,
    attention_config: *const ModernAttentionConfig,
) -> TrustformersError {
    if config.is_null() || attention_config.is_null() {
        return TrustformersError::NullPointer;
    }

    unsafe {
        let arch_config = &mut *config;
        let attn_cfg = &*attention_config;
        arch_config.attention_config = Some(attn_cfg.clone());
    }

    TrustformersError::Success
}

/// Create a Mamba/SSM model
#[no_mangle]
pub extern "C" fn trustformers_create_mamba_model(
    config: *const ModernArchitectureConfig,
    model_handle: *mut TrustformersModernArchitecture,
) -> TrustformersError {
    if config.is_null() || model_handle.is_null() {
        return TrustformersError::NullPointer;
    }

    unsafe {
        let arch_config = &*config;

        if arch_config.arch_type != ModernArchitectureType::Mamba {
            return TrustformersError::InvalidParameter;
        }

        let mamba_config = match &arch_config.mamba_config {
            Some(cfg) => cfg,
            None => return TrustformersError::InvalidParameter,
        };

        // Create SSM operations
        let ssm_ops = SSMOperations::new(
            mamba_config.d_model as usize,
            mamba_config.d_state as usize,
            mamba_config.d_conv as usize,
        );

        // Convert SSMOperations to a handle using Box::into_raw
        let boxed = Box::new(ssm_ops);
        let handle = Box::into_raw(boxed) as usize;

        *model_handle = handle;
    }

    TrustformersError::Success
}

/// Perform Mamba/SSM inference
#[no_mangle]
pub extern "C" fn trustformers_mamba_inference(
    model_handle: TrustformersModernArchitecture,
    input_data: *const c_float,
    input_length: usize,
    state: *mut c_float,
    state_size: usize,
    output: *mut c_float,
    output_length: usize,
) -> TrustformersError {
    if input_data.is_null() || state.is_null() || output.is_null() {
        return TrustformersError::NullPointer;
    }

    // The model_handle is actually a raw pointer to SSMOperations (created with Box::into_raw)
    let ssm_ops = unsafe { &*(model_handle as *const SSMOperations) };

    unsafe {
        // Convert C arrays to Rust slices
        let input_slice = std::slice::from_raw_parts(input_data, input_length);
        let mut state_slice = std::slice::from_raw_parts_mut(state, state_size);
        let output_slice = std::slice::from_raw_parts_mut(output, output_length);

        // Convert to Vec for processing
        let input_vec: Vec<f32> = input_slice.iter().copied().collect();

        // Perform SSM forward pass
        let result = ssm_ops.forward(&input_vec, &mut state_slice);

        // Copy result to output buffer
        let copy_len = result.len().min(output_length);
        output_slice[..copy_len].copy_from_slice(&result[..copy_len]);
    }

    TrustformersError::Success
}

/// Get model performance metrics for modern architectures
#[no_mangle]
pub extern "C" fn trustformers_modern_arch_get_metrics(
    model_handle: TrustformersModernArchitecture,
    metrics_json: *mut *mut c_char,
) -> TrustformersError {
    if metrics_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = &crate::RESOURCE_REGISTRY;
    let reg = registry.read();

    let _model = match reg.get_model(model_handle) {
        Some(m) => m,
        None => return TrustformersError::InvalidHandle,
    };

    // Create sample metrics
    let metrics = serde_json::json!({
        "architecture_type": "Mamba",
        "parameters": {
            "total_params": 2400000000u64,
            "active_params": 2400000000u64,
            "memory_usage_mb": 9600.0
        },
        "performance": {
            "inference_speed_tokens_per_sec": 150.0,
            "memory_efficiency": 0.95,
            "flops_per_token": 12.5e9
        },
        "capabilities": {
            "max_sequence_length": 4096,
            "context_window": 4096,
            "supports_streaming": true,
            "supports_parallel_inference": true
        }
    });

    let metrics_str = match serde_json::to_string_pretty(&metrics) {
        Ok(s) => crate::string_to_c_str(s),
        Err(_) => return TrustformersError::SerializationError,
    };

    unsafe {
        *metrics_json = metrics_str;
    }

    TrustformersError::Success
}

/// Compare different modern architectures
#[no_mangle]
pub extern "C" fn trustformers_compare_architectures(
    arch_types: *const ModernArchitectureType,
    num_archs: usize,
    comparison_json: *mut *mut c_char,
) -> TrustformersError {
    if arch_types.is_null() || comparison_json.is_null() || num_archs == 0 {
        return TrustformersError::InvalidParameter;
    }

    unsafe {
        let architectures = std::slice::from_raw_parts(arch_types, num_archs);

        let mut comparisons = Vec::new();

        for &arch in architectures {
            let arch_info = match arch {
                ModernArchitectureType::Mamba => serde_json::json!({
                    "name": "Mamba",
                    "type": "State Space Model",
                    "efficiency": {
                        "memory": 0.95,
                        "compute": 0.90,
                        "inference_speed": 0.92
                    },
                    "strengths": ["Linear scaling", "Long sequences", "Memory efficient"],
                    "use_cases": ["Long document processing", "Time series", "Code generation"],
                    "complexity": "O(n)",
                    "max_sequence_length": 1000000
                }),
                ModernArchitectureType::Transformer => serde_json::json!({
                    "name": "Transformer",
                    "type": "Attention-based",
                    "efficiency": {
                        "memory": 0.75,
                        "compute": 0.85,
                        "inference_speed": 0.80
                    },
                    "strengths": ["Parallelizable", "Good performance", "Well-established"],
                    "use_cases": ["General NLP", "Translation", "Summarization"],
                    "complexity": "O(n²)",
                    "max_sequence_length": 4096
                }),
                ModernArchitectureType::MixtureOfExperts => serde_json::json!({
                    "name": "Mixture of Experts",
                    "type": "Sparse activation",
                    "efficiency": {
                        "memory": 0.70,
                        "compute": 0.95,
                        "inference_speed": 0.85
                    },
                    "strengths": ["Scalable", "Specialized experts", "High capacity"],
                    "use_cases": ["Large-scale models", "Multi-domain tasks", "Efficiency"],
                    "complexity": "O(n²) but sparse",
                    "max_sequence_length": 8192
                }),
                _ => serde_json::json!({
                    "name": "Other",
                    "type": "Various",
                    "efficiency": {
                        "memory": 0.80,
                        "compute": 0.80,
                        "inference_speed": 0.80
                    },
                    "strengths": ["Architecture-specific"],
                    "use_cases": ["Specialized applications"],
                    "complexity": "Varies",
                    "max_sequence_length": 4096
                }),
            };
            comparisons.push(arch_info);
        }

        let comparison_result = serde_json::json!({
            "comparison_date": "2025-07-23",
            "architectures": comparisons,
            "recommendations": {
                "long_sequences": "Mamba",
                "general_nlp": "Transformer",
                "large_scale": "MixtureOfExperts",
                "efficiency_focused": "Mamba"
            }
        });

        let comparison_str = match serde_json::to_string_pretty(&comparison_result) {
            Ok(s) => crate::string_to_c_str(s),
            Err(_) => return TrustformersError::SerializationError,
        };

        *comparison_json = comparison_str;
    }

    TrustformersError::Success
}

/// Free a modern architecture model
#[no_mangle]
pub extern "C" fn trustformers_modern_arch_free(
    model_handle: TrustformersModernArchitecture,
) -> TrustformersError {
    let registry = &crate::RESOURCE_REGISTRY;
    let mut reg = registry.write();

    if reg.remove_model(model_handle) {
        TrustformersError::Success
    } else {
        TrustformersError::InvalidHandle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_config_creation() {
        let config = MambaConfig::default();
        assert_eq!(config.d_model, 2048);
        assert_eq!(config.d_state, 16);
        assert_eq!(config.d_conv, 4);
    }

    #[test]
    fn test_ssm_operations_creation() {
        let ssm = SSMOperations::new(512, 16, 4);
        assert_eq!(ssm.a_matrices.len(), 512);
        assert_eq!(ssm.a_matrices[0].len(), 16);
        assert_eq!(ssm.conv_parameters.len(), 512);
        assert_eq!(ssm.conv_parameters[0].len(), 4);
    }

    #[test]
    fn test_ssm_forward_pass() {
        let ssm = SSMOperations::new(4, 2, 2);
        let input = vec![1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, -0.6];
        let mut state = vec![0.0; 8]; // 4 * 2 state dimensions

        let output = ssm.forward(&input, &mut state);
        assert_eq!(output.len(), input.len());

        // Check that state has been updated
        assert_ne!(state, vec![0.0; 8]);
    }

    #[test]
    fn test_modern_arch_config_creation() {
        let config_ptr = trustformers_modern_arch_config_create(ModernArchitectureType::Mamba);
        assert!(!config_ptr.is_null());

        unsafe {
            let config = &*config_ptr;
            assert_eq!(config.arch_type, ModernArchitectureType::Mamba);
            assert!(config.mamba_config.is_some());
        }

        trustformers_modern_arch_config_free(config_ptr);
    }

    #[test]
    fn test_architecture_comparison() {
        let archs = [
            ModernArchitectureType::Mamba,
            ModernArchitectureType::Transformer,
            ModernArchitectureType::MixtureOfExperts,
        ];

        let mut comparison_json: *mut c_char = ptr::null_mut();
        let result =
            trustformers_compare_architectures(archs.as_ptr(), archs.len(), &mut comparison_json);

        assert_eq!(result, TrustformersError::Success);
        assert!(!comparison_json.is_null());

        // Clean up
        if !comparison_json.is_null() {
            unsafe {
                let _ = CString::from_raw(comparison_json);
            }
        }
    }
}
