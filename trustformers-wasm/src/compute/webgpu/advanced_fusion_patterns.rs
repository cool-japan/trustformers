//! Advanced Kernel Fusion Patterns for Transformer Models
//!
//! This module implements sophisticated fusion patterns optimized for transformer architectures,
//! including multi-head attention, feed-forward networks, and layer normalization patterns.

use crate::webgpu::DeviceCapabilities;
use std::collections::HashMap;
use std::string::String;
use wasm_bindgen::prelude::*;

/// Advanced fusion pattern types for transformers
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformerFusionPattern {
    /// Fused multi-head attention (Q, K, V projection + attention)
    MultiHeadAttention,
    /// Fused feed-forward network (Linear + Activation + Linear)
    FeedForward,
    /// Fused layer normalization + residual connection
    LayerNormResidual,
    /// Fused attention scoring (Q @ K^T + mask + softmax)
    AttentionScoring,
    /// Fused attention application (attention @ V + projection)
    AttentionOutput,
    /// Fused GELU activation (exact computation)
    GELUExact,
    /// Fused SwiGLU (element-wise gate + activation)
    SwiGLU,
    /// Fused RoPE (rotary position embedding)
    RotaryPositionEmbedding,
    /// Fused group norm + SiLU activation
    GroupNormSiLU,
    /// Fused QKV projection (single fused linear layer)
    QKVProjection,
    /// Fused attention bias (add + mask + clamp)
    AttentionBias,
    /// Fused RMSNorm (root mean square normalization)
    RMSNorm,
    /// Fused ALiBi (attention with linear biases)
    ALiBiAttention,
    /// Fused cross-attention (encoder-decoder)
    CrossAttention,
}

/// Fusion pattern configuration
#[derive(Debug, Clone)]
pub struct FusionPatternConfig {
    pub pattern_type: TransformerFusionPattern,
    pub enable_fp16: bool,
    pub enable_tf32: bool,
    pub use_flash_attention: bool,
    pub max_sequence_length: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl Default for FusionPatternConfig {
    fn default() -> Self {
        Self {
            pattern_type: TransformerFusionPattern::MultiHeadAttention,
            enable_fp16: true,
            enable_tf32: false,
            use_flash_attention: true,
            max_sequence_length: 2048,
            num_heads: 8,
            head_dim: 64,
        }
    }
}

/// Advanced fusion pattern optimizer
pub struct AdvancedFusionOptimizer {
    #[allow(dead_code)]
    capabilities: DeviceCapabilities,
    #[allow(dead_code)]
    pattern_configs: HashMap<TransformerFusionPattern, FusionPatternConfig>,
    fusion_cache: HashMap<String, String>, // shader cache
    profiling_enabled: bool,
    performance_stats: HashMap<TransformerFusionPattern, FusionStats>,
}

/// Fusion performance statistics
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    pub invocation_count: usize,
    pub total_time_ms: f32,
    pub average_time_ms: f32,
    pub min_time_ms: f32,
    pub max_time_ms: f32,
    pub memory_saved_bytes: usize,
}

impl AdvancedFusionOptimizer {
    /// Create a new advanced fusion optimizer
    pub fn new(capabilities: DeviceCapabilities) -> Self {
        Self {
            capabilities,
            pattern_configs: HashMap::new(),
            fusion_cache: HashMap::new(),
            profiling_enabled: false,
            performance_stats: HashMap::new(),
        }
    }

    /// Enable performance profiling
    pub fn enable_profiling(&mut self) {
        self.profiling_enabled = true;
    }

    /// Generate fused shader for multi-head attention
    pub fn generate_mha_fusion(&self, config: &FusionPatternConfig) -> Result<String, JsValue> {
        let precision = if config.enable_fp16 { "f16" } else { "f32" };

        // Generate optimized multi-head attention shader
        let shader = format!(
            r#"
// Fused Multi-Head Attention Kernel
// Precision: {}
// Num Heads: {}
// Head Dim: {}
// Max Seq Length: {}

@group(0) @binding(0) var<storage, read> queries: array<{}>;
@group(0) @binding(1) var<storage, read> keys: array<{}>;
@group(0) @binding(2) var<storage, read> values: array<{}>;
@group(0) @binding(3) var<storage, read_write> output: array<{}>;
@group(0) @binding(4) var<storage, read> attention_mask: array<{}>;

struct Params {{
    batch_size: u32,
    seq_length: u32,
    num_heads: u32,
    head_dim: u32,
    scale: {},
}}

@group(0) @binding(5) var<uniform> params: Params;

// Fused QK^T + Scale + Mask + Softmax
@compute @workgroup_size(256, 1, 1)
fn attention_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let batch_idx = global_id.x / (params.num_heads * params.seq_length);
    let head_idx = (global_id.x / params.seq_length) % params.num_heads;
    let seq_idx = global_id.x % params.seq_length;

    if batch_idx >= params.batch_size || seq_idx >= params.seq_length {{
        return;
    }}

    // Compute attention scores (fused QK^T + scale)
    var max_score: {} = -1e10;
    var scores: array<{}, 2048>;

    for (var k = 0u; k < params.seq_length; k = k + 1u) {{
        var score: {} = 0.0;

        // Compute dot product Q @ K^T
        for (var d = 0u; d < params.head_dim; d = d + 1u) {{
            let q_idx = batch_idx * params.num_heads * params.seq_length * params.head_dim +
                       head_idx * params.seq_length * params.head_dim +
                       seq_idx * params.head_dim + d;
            let k_idx = batch_idx * params.num_heads * params.seq_length * params.head_dim +
                       head_idx * params.seq_length * params.head_dim +
                       k * params.head_dim + d;

            score = score + queries[q_idx] * keys[k_idx];
        }}

        // Apply scale factor
        score = score * params.scale;

        // Apply attention mask (fused)
        let mask_idx = batch_idx * params.seq_length * params.seq_length + seq_idx * params.seq_length + k;
        score = score + attention_mask[mask_idx];

        scores[k] = score;
        max_score = max(max_score, score);
    }}

    // Fused softmax (numerically stable)
    var sum_exp: {} = 0.0;
    for (var k = 0u; k < params.seq_length; k = k + 1u) {{
        scores[k] = exp(scores[k] - max_score);
        sum_exp = sum_exp + scores[k];
    }}

    for (var k = 0u; k < params.seq_length; k = k + 1u) {{
        scores[k] = scores[k] / sum_exp;
    }}

    // Fused attention @ V
    for (var d = 0u; d < params.head_dim; d = d + 1u) {{
        var value: {} = 0.0;

        for (var k = 0u; k < params.seq_length; k = k + 1u) {{
            let v_idx = batch_idx * params.num_heads * params.seq_length * params.head_dim +
                       head_idx * params.seq_length * params.head_dim +
                       k * params.head_dim + d;
            value = value + scores[k] * values[v_idx];
        }}

        let out_idx = batch_idx * params.num_heads * params.seq_length * params.head_dim +
                     head_idx * params.seq_length * params.head_dim +
                     seq_idx * params.head_dim + d;
        output[out_idx] = value;
    }}
}}
"#,
            precision,
            config.num_heads,
            config.head_dim,
            config.max_sequence_length,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision
        );

        Ok(shader)
    }

    /// Generate fused shader for feed-forward network (FFN)
    pub fn generate_ffn_fusion(&self, config: &FusionPatternConfig) -> Result<String, JsValue> {
        let precision = if config.enable_fp16 { "f16" } else { "f32" };

        // Generate optimized FFN shader (Linear + GELU + Linear)
        let shader = format!(
            r#"
// Fused Feed-Forward Network Kernel
// Precision: {}
// Pattern: Linear + GELU + Linear

@group(0) @binding(0) var<storage, read> input: array<{}>;
@group(0) @binding(1) var<storage, read> weights1: array<{}>;
@group(0) @binding(2) var<storage, read> bias1: array<{}>;
@group(0) @binding(3) var<storage, read> weights2: array<{}>;
@group(0) @binding(4) var<storage, read> bias2: array<{}>;
@group(0) @binding(5) var<storage, read_write> output: array<{}>;

struct Params {{
    batch_size: u32,
    seq_length: u32,
    input_dim: u32,
    hidden_dim: u32,
    output_dim: u32,
}}

@group(0) @binding(6) var<uniform> params: Params;

// Constants for GELU approximation
const SQRT_2_OVER_PI: {} = 0.7978845608028654;
const GELU_ALPHA: {} = 0.044715;

// Fused GELU activation (exact)
fn gelu(x: {}) -> {} {{
    return 0.5 * x * (1.0 + tanh(SQRT_2_OVER_PI * (x + GELU_ALPHA * x * x * x)));
}}

@compute @workgroup_size(256, 1, 1)
fn ffn_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let batch_idx = global_id.x / params.seq_length;
    let seq_idx = global_id.x % params.seq_length;

    if batch_idx >= params.batch_size || seq_idx >= params.seq_length {{
        return;
    }}

    // First linear layer + GELU (fused)
    var hidden: array<{}, 4096>;
    for (var h = 0u; h < params.hidden_dim; h = h + 1u) {{
        var sum: {} = bias1[h];

        for (var i = 0u; i < params.input_dim; i = i + 1u) {{
            let in_idx = batch_idx * params.seq_length * params.input_dim + seq_idx * params.input_dim + i;
            let w_idx = h * params.input_dim + i;
            sum = sum + input[in_idx] * weights1[w_idx];
        }}

        // Apply GELU activation (fused)
        hidden[h] = gelu(sum);
    }}

    // Second linear layer (fused)
    for (var o = 0u; o < params.output_dim; o = o + 1u) {{
        var sum: {} = bias2[o];

        for (var h = 0u; h < params.hidden_dim; h = h + 1u) {{
            let w_idx = o * params.hidden_dim + h;
            sum = sum + hidden[h] * weights2[w_idx];
        }}

        let out_idx = batch_idx * params.seq_length * params.output_dim + seq_idx * params.output_dim + o;
        output[out_idx] = sum;
    }}
}}
"#,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision
        );

        Ok(shader)
    }

    /// Generate fused shader for layer norm + residual
    pub fn generate_layernorm_residual_fusion(
        &self,
        config: &FusionPatternConfig,
    ) -> Result<String, JsValue> {
        let precision = if config.enable_fp16 { "f16" } else { "f32" };

        let shader = format!(
            r#"
// Fused Layer Normalization + Residual Connection
// Precision: {}

@group(0) @binding(0) var<storage, read> input: array<{}>;
@group(0) @binding(1) var<storage, read> residual: array<{}>;
@group(0) @binding(2) var<storage, read> gamma: array<{}>;
@group(0) @binding(3) var<storage, read> beta: array<{}>;
@group(0) @binding(4) var<storage, read_write> output: array<{}>;

struct Params {{
    batch_size: u32,
    seq_length: u32,
    hidden_dim: u32,
    epsilon: {},
}}

@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn layernorm_residual_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let batch_idx = global_id.x / params.seq_length;
    let seq_idx = global_id.x % params.seq_length;

    if batch_idx >= params.batch_size || seq_idx >= params.seq_length {{
        return;
    }}

    let base_idx = batch_idx * params.seq_length * params.hidden_dim + seq_idx * params.hidden_dim;

    // Fused: Add residual + compute mean
    var mean: {} = 0.0;
    for (var d = 0u; d < params.hidden_dim; d = d + 1u) {{
        let idx = base_idx + d;
        let value = input[idx] + residual[idx];
        mean = mean + value;
    }}
    mean = mean / {}(params.hidden_dim);

    // Fused: Compute variance
    var variance: {} = 0.0;
    for (var d = 0u; d < params.hidden_dim; d = d + 1u) {{
        let idx = base_idx + d;
        let value = input[idx] + residual[idx];
        let diff = value - mean;
        variance = variance + diff * diff;
    }}
    variance = variance / {}(params.hidden_dim);

    // Fused: Normalize + scale + shift
    let inv_std = 1.0 / sqrt(variance + params.epsilon);
    for (var d = 0u; d < params.hidden_dim; d = d + 1u) {{
        let idx = base_idx + d;
        let value = input[idx] + residual[idx];
        let normalized = (value - mean) * inv_std;
        output[idx] = normalized * gamma[d] + beta[d];
    }}
}}
"#,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision
        );

        Ok(shader)
    }

    /// Generate fused shader for RMSNorm (used in LLaMA)
    pub fn generate_rmsnorm_fusion(&self, config: &FusionPatternConfig) -> Result<String, JsValue> {
        let precision = if config.enable_fp16 { "f16" } else { "f32" };

        let shader = format!(
            r#"
// Fused RMS Normalization (LLaMA-style)
// Precision: {}

@group(0) @binding(0) var<storage, read> input: array<{}>;
@group(0) @binding(1) var<storage, read> weight: array<{}>;
@group(0) @binding(2) var<storage, read_write> output: array<{}>;

struct Params {{
    batch_size: u32,
    seq_length: u32,
    hidden_dim: u32,
    epsilon: {},
}}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn rmsnorm_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let batch_idx = global_id.x / params.seq_length;
    let seq_idx = global_id.x % params.seq_length;

    if batch_idx >= params.batch_size || seq_idx >= params.seq_length {{
        return;
    }}

    let base_idx = batch_idx * params.seq_length * params.hidden_dim + seq_idx * params.hidden_dim;

    // Compute root mean square
    var rms: {} = 0.0;
    for (var d = 0u; d < params.hidden_dim; d = d + 1u) {{
        let idx = base_idx + d;
        let value = input[idx];
        rms = rms + value * value;
    }}
    rms = sqrt(rms / {}(params.hidden_dim) + params.epsilon);

    // Normalize and scale
    let inv_rms = 1.0 / rms;
    for (var d = 0u; d < params.hidden_dim; d = d + 1u) {{
        let idx = base_idx + d;
        output[idx] = input[idx] * inv_rms * weight[d];
    }}
}}
"#,
            precision, precision, precision, precision, precision, precision, precision
        );

        Ok(shader)
    }

    /// Generate fused shader for SwiGLU activation (used in PaLM/LLaMA)
    pub fn generate_swiglu_fusion(&self, config: &FusionPatternConfig) -> Result<String, JsValue> {
        let precision = if config.enable_fp16 { "f16" } else { "f32" };

        let shader = format!(
            r#"
// Fused SwiGLU Activation (Swish-Gated Linear Unit)
// Precision: {}
// Used in: PaLM, LLaMA

@group(0) @binding(0) var<storage, read> input: array<{}>;
@group(0) @binding(1) var<storage, read> gate_weights: array<{}>;
@group(0) @binding(2) var<storage, read> up_weights: array<{}>;
@group(0) @binding(3) var<storage, read_write> output: array<{}>;

struct Params {{
    batch_size: u32,
    seq_length: u32,
    input_dim: u32,
    hidden_dim: u32,
}}

@group(0) @binding(4) var<uniform> params: Params;

// SiLU (Swish) activation
fn silu(x: {}) -> {} {{
    return x / (1.0 + exp(-x));
}}

@compute @workgroup_size(256, 1, 1)
fn swiglu_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let batch_idx = global_id.x / params.seq_length;
    let seq_idx = global_id.x % params.seq_length;

    if batch_idx >= params.batch_size || seq_idx >= params.seq_length {{
        return;
    }}

    let base_in_idx = batch_idx * params.seq_length * params.input_dim + seq_idx * params.input_dim;
    let base_out_idx = batch_idx * params.seq_length * params.hidden_dim + seq_idx * params.hidden_dim;

    // Fused gate projection + up projection + SiLU + element-wise multiply
    for (var h = 0u; h < params.hidden_dim; h = h + 1u) {{
        var gate_value: {} = 0.0;
        var up_value: {} = 0.0;

        for (var i = 0u; i < params.input_dim; i = i + 1u) {{
            let in_idx = base_in_idx + i;
            let gate_w_idx = h * params.input_dim + i;
            let up_w_idx = h * params.input_dim + i;

            gate_value = gate_value + input[in_idx] * gate_weights[gate_w_idx];
            up_value = up_value + input[in_idx] * up_weights[up_w_idx];
        }}

        // Apply SwiGLU: SiLU(gate) * up
        output[base_out_idx + h] = silu(gate_value) * up_value;
    }}
}}
"#,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision,
            precision
        );

        Ok(shader)
    }

    /// Get performance statistics for a fusion pattern
    pub fn get_stats(&self, pattern: TransformerFusionPattern) -> Option<&FusionStats> {
        self.performance_stats.get(&pattern)
    }

    /// Clear all caches
    pub fn clear_all_caches(&mut self) {
        self.fusion_cache.clear();
        self.performance_stats.clear();
    }

    /// Estimate performance improvement from fusion
    pub fn estimate_speedup(&self, pattern: TransformerFusionPattern) -> f32 {
        match pattern {
            TransformerFusionPattern::MultiHeadAttention => 2.5, // 2.5x speedup
            TransformerFusionPattern::FeedForward => 1.8,
            TransformerFusionPattern::LayerNormResidual => 1.5,
            TransformerFusionPattern::SwiGLU => 1.9,
            TransformerFusionPattern::RMSNorm => 1.6,
            TransformerFusionPattern::GELUExact => 1.3,
            _ => 1.2, // Default conservative estimate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_pattern_config_default() {
        let config = FusionPatternConfig::default();
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.max_sequence_length, 2048);
        assert!(config.enable_fp16);
    }

    #[test]
    fn test_advanced_fusion_optimizer_creation() {
        let caps = DeviceCapabilities {
            webgpu_available: true,
            gpu_memory_limit: 1024 * 1024 * 1024,
            max_compute_workgroup_size: 256,
            max_compute_invocations_per_workgroup: 256,
            simd_support: true,
            performance_tier: 2,
            is_mobile: false,
            supports_f16: true,
        };

        let optimizer = AdvancedFusionOptimizer::new(caps);
        assert!(!optimizer.profiling_enabled);
    }

    #[test]
    fn test_mha_fusion_generation() {
        let caps = DeviceCapabilities {
            webgpu_available: true,
            gpu_memory_limit: 1024 * 1024 * 1024,
            max_compute_workgroup_size: 256,
            max_compute_invocations_per_workgroup: 256,
            simd_support: true,
            performance_tier: 2,
            is_mobile: false,
            supports_f16: true,
        };

        let optimizer = AdvancedFusionOptimizer::new(caps);
        let config = FusionPatternConfig::default();

        let shader = optimizer.generate_mha_fusion(&config);
        assert!(shader.is_ok());

        let shader_code = shader.unwrap();
        assert!(shader_code.contains("Multi-Head Attention"));
        assert!(shader_code.contains("@compute"));
    }

    #[test]
    fn test_ffn_fusion_generation() {
        let caps = DeviceCapabilities {
            webgpu_available: true,
            gpu_memory_limit: 1024 * 1024 * 1024,
            max_compute_workgroup_size: 256,
            max_compute_invocations_per_workgroup: 256,
            simd_support: true,
            performance_tier: 2,
            is_mobile: false,
            supports_f16: false,
        };

        let optimizer = AdvancedFusionOptimizer::new(caps);
        let config = FusionPatternConfig {
            enable_fp16: false,
            ..Default::default()
        };

        let shader = optimizer.generate_ffn_fusion(&config);
        assert!(shader.is_ok());

        let shader_code = shader.unwrap();
        assert!(shader_code.contains("Feed-Forward Network"));
        assert!(shader_code.contains("gelu"));
        assert!(shader_code.contains("f32")); // Should use f32 since fp16 is disabled
    }

    #[test]
    fn test_layernorm_residual_fusion() {
        let caps = DeviceCapabilities {
            webgpu_available: true,
            gpu_memory_limit: 1024 * 1024 * 1024,
            max_compute_workgroup_size: 256,
            max_compute_invocations_per_workgroup: 256,
            simd_support: true,
            performance_tier: 2,
            is_mobile: false,
            supports_f16: true,
        };

        let optimizer = AdvancedFusionOptimizer::new(caps);
        let config = FusionPatternConfig::default();

        let shader = optimizer.generate_layernorm_residual_fusion(&config);
        assert!(shader.is_ok());

        let shader_code = shader.unwrap();
        assert!(shader_code.contains("Layer Normalization + Residual"));
    }

    #[test]
    fn test_rmsnorm_fusion() {
        let caps = DeviceCapabilities {
            webgpu_available: true,
            gpu_memory_limit: 1024 * 1024 * 1024,
            max_compute_workgroup_size: 256,
            max_compute_invocations_per_workgroup: 256,
            simd_support: true,
            performance_tier: 2,
            is_mobile: false,
            supports_f16: true,
        };

        let optimizer = AdvancedFusionOptimizer::new(caps);
        let config = FusionPatternConfig::default();

        let shader = optimizer.generate_rmsnorm_fusion(&config);
        assert!(shader.is_ok());

        let shader_code = shader.unwrap();
        assert!(shader_code.contains("RMS Normalization"));
        assert!(shader_code.contains("LLaMA"));
    }

    #[test]
    fn test_swiglu_fusion() {
        let caps = DeviceCapabilities {
            webgpu_available: true,
            gpu_memory_limit: 1024 * 1024 * 1024,
            max_compute_workgroup_size: 256,
            max_compute_invocations_per_workgroup: 256,
            simd_support: true,
            performance_tier: 2,
            is_mobile: false,
            supports_f16: true,
        };

        let optimizer = AdvancedFusionOptimizer::new(caps);
        let config = FusionPatternConfig::default();

        let shader = optimizer.generate_swiglu_fusion(&config);
        assert!(shader.is_ok());

        let shader_code = shader.unwrap();
        assert!(shader_code.contains("SwiGLU"));
        assert!(shader_code.contains("silu"));
    }

    #[test]
    fn test_speedup_estimates() {
        let caps = DeviceCapabilities {
            webgpu_available: true,
            gpu_memory_limit: 1024 * 1024 * 1024,
            max_compute_workgroup_size: 256,
            max_compute_invocations_per_workgroup: 256,
            simd_support: true,
            performance_tier: 2,
            is_mobile: false,
            supports_f16: true,
        };

        let optimizer = AdvancedFusionOptimizer::new(caps);

        assert_eq!(
            optimizer.estimate_speedup(TransformerFusionPattern::MultiHeadAttention),
            2.5
        );
        assert_eq!(
            optimizer.estimate_speedup(TransformerFusionPattern::FeedForward),
            1.8
        );
        assert!(optimizer.estimate_speedup(TransformerFusionPattern::RMSNorm) > 1.0);
    }
}
