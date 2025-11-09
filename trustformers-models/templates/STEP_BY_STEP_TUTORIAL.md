# Step-by-Step New Model Implementation Tutorial

This comprehensive tutorial walks you through implementing a new transformer model in TrustformeRS from planning to production-ready code.

## Table of Contents

1. [Planning and Design Phase](#planning-and-design-phase)
2. [Project Structure Setup](#project-structure-setup)
3. [Configuration Implementation](#configuration-implementation)
4. [Core Architecture Implementation](#core-architecture-implementation)
5. [Model Integration](#model-integration)
6. [Task-Specific Heads](#task-specific-heads)
7. [Testing Strategy](#testing-strategy)
8. [Documentation and Examples](#documentation-and-examples)
9. [Advanced Features](#advanced-features)
10. [Deployment Considerations](#deployment-considerations)

## Planning and Design Phase

### Step 1: Research and Architecture Analysis

Before writing any code, thoroughly understand your target model:

```markdown
## Model Research Checklist

- [ ] Paper/documentation review
- [ ] Architecture diagram analysis  
- [ ] Parameter counting and sizing
- [ ] Reference implementation analysis
- [ ] Tokenization requirements
- [ ] Training procedure understanding
- [ ] Licensing considerations
```

**Example: Implementing Mistral 7B**

1. **Paper Analysis**: Read the Mistral 7B paper and understand:
   - Sliding window attention mechanism
   - Group Query Attention (GQA)
   - SwiGLU activation function
   - RMSNorm instead of LayerNorm

2. **Reference Implementation**: Study HuggingFace's implementation:
   ```bash
   # Clone and analyze reference implementation
   git clone https://github.com/huggingface/transformers
   cd transformers/src/transformers/models/mistral
   ```

3. **Architecture Mapping**:
   ```rust
   // Map paper concepts to Rust implementation
   struct MistralLayer {
       self_attn: MistralAttention,    // Sliding window + GQA
       mlp: MistralMLP,               // SwiGLU activation
       input_layernorm: RMSNorm,      // Pre-attention norm
       post_attention_layernorm: RMSNorm, // Pre-MLP norm
   }
   ```

### Step 2: Define Implementation Scope

Create a detailed implementation plan:

```markdown
## Implementation Scope

### Core Features (MVP)
- [ ] Basic forward pass
- [ ] Standard attention mechanism
- [ ] Text generation capability
- [ ] HuggingFace weight compatibility

### Advanced Features (Phase 2)
- [ ] Sliding window attention
- [ ] Group Query Attention
- [ ] Flash Attention integration
- [ ] Quantization support

### Task-Specific Features (Phase 3)
- [ ] Instruction tuning support
- [ ] Chat template integration
- [ ] Tool calling capabilities
```

## Project Structure Setup

### Step 3: Create Directory Structure

```bash
# Create model directory structure
mkdir -p src/mistral/{config,model,layers,attention,heads,tests}

# Directory layout:
src/mistral/
├── mod.rs           # Public interface
├── config.rs        # Configuration definitions
├── model.rs         # Main model implementation
├── layers.rs        # Layer implementations
├── attention.rs     # Attention mechanisms
├── heads.rs         # Task-specific heads
└── tests.rs         # Comprehensive tests
```

### Step 4: Module Declaration

**src/mistral/mod.rs**:
```rust
//! Mistral 7B Model Implementation
//! 
//! High-performance implementation of Mistral's transformer architecture
//! with sliding window attention and group query attention.

pub mod config;
pub mod model;
pub mod layers;
pub mod attention;
pub mod heads;

pub use config::*;
pub use model::*;
pub use heads::*;

#[cfg(test)]
mod tests;
```

## Configuration Implementation

### Step 5: Define Configuration Structure

**src/mistral/config.rs**:
```rust
use serde::{Deserialize, Serialize};
use crate::{ModelConfig, utils::activations::Activation};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralConfig {
    // Core architecture parameters
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,  // For Group Query Attention
    pub max_position_embeddings: usize,
    
    // Attention configuration
    pub sliding_window: Option<usize>,  // Mistral-specific
    pub attention_dropout: f32,
    
    // MLP configuration
    pub hidden_act: Activation,
    
    // Normalization
    pub rms_norm_eps: f32,
    
    // Initialization and training
    pub initializer_range: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub rope_theta: f32,
    
    // Generation parameters
    pub use_cache: bool,
    pub pad_token_id: Option<usize>,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    
    // Model metadata
    pub model_type: String,
    pub transformers_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub scaling_factor: f32,
}

impl Default for MistralConfig {
    fn default() -> Self {
        // Mistral 7B default configuration
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,  // GQA with 4:1 ratio
            max_position_embeddings: 32768,
            sliding_window: Some(4096),
            attention_dropout: 0.0,
            hidden_act: Activation::SiLU,
            rms_norm_eps: 1e-5,
            initializer_range: 0.02,
            rope_scaling: None,
            rope_theta: 10000.0,
            use_cache: true,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            model_type: "mistral".to_string(),
            transformers_version: None,
        }
    }
}

impl ModelConfig for MistralConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    
    fn num_labels(&self) -> Option<usize> {
        None
    }
    
    fn id2label(&self) -> Option<&HashMap<usize, String>> {
        None
    }
    
    fn label2id(&self) -> Option<&HashMap<String, usize>> {
        None
    }
    
    fn is_decoder(&self) -> bool {
        true
    }
    
    fn is_encoder_decoder(&self) -> bool {
        false
    }
}

// Configuration validation
impl MistralConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err("hidden_size must be divisible by num_attention_heads".to_string());
        }
        
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err("num_attention_heads must be divisible by num_key_value_heads".to_string());
        }
        
        if self.vocab_size == 0 {
            return Err("vocab_size must be greater than 0".to_string());
        }
        
        Ok(())
    }
    
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    
    pub fn num_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

// Preset configurations
impl MistralConfig {
    pub fn mistral_7b() -> Self {
        Self::default()
    }
    
    pub fn mistral_7b_instruct() -> Self {
        let mut config = Self::default();
        config.model_type = "mistral-instruct".to_string();
        config
    }
    
    pub fn mistral_7b_chat() -> Self {
        let mut config = Self::default();
        config.model_type = "mistral-chat".to_string();
        config
    }
}
```

### Step 6: Configuration Testing

**src/mistral/tests.rs** (Configuration section):
```rust
#[cfg(test)]
mod config_tests {
    use super::*;
    
    #[test]
    fn test_default_config_validity() {
        let config = MistralConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = MistralConfig::default();
        
        // Test invalid head dimensions
        config.num_attention_heads = 31; // Not divisible into hidden_size
        assert!(config.validate().is_err());
        
        // Test invalid GQA configuration
        config.num_attention_heads = 32;
        config.num_key_value_heads = 7; // 32 not divisible by 7
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_helper_methods() {
        let config = MistralConfig::default();
        assert_eq!(config.head_dim(), 128); // 4096 / 32
        assert_eq!(config.num_groups(), 4); // 32 / 8
    }
}
```

## Core Architecture Implementation

### Step 7: Implement RMSNorm

**src/mistral/layers.rs**:
```rust
use trustformers_core::{Result, Tensor, Module, Parameter};

pub struct RMSNorm {
    weight: Parameter,
    eps: f32,
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: Parameter::ones(&[hidden_size])?,
            eps,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_dtype = x.dtype();
        let variance = x.to_dtype(DataType::Float32)?
            .pow(&Tensor::scalar(2.0)?)?
            .mean_dim(&[-1], true)?;
        
        let hidden_states = x.div(&(variance + self.eps)?.sqrt()?)?;
        let weight = self.weight.unsqueeze(0)?;
        
        hidden_states.mul(&weight)?.to_dtype(input_dtype)
    }
}

impl Module for RMSNorm {
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight.data()]
    }
}
```

### Step 8: Implement Rotary Position Embedding

```rust
pub struct RotaryEmbedding {
    dim: usize,
    max_seq_len: usize,
    base: f32,
    cos_cached: Option<Tensor>,
    sin_cached: Option<Tensor>,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32) -> Result<Self> {
        Ok(Self {
            dim,
            max_seq_len,
            base,
            cos_cached: None,
            sin_cached: None,
        })
    }
    
    pub fn forward(&mut self, x: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        if self.cos_cached.is_none() || seq_len > self.max_seq_len {
            self.update_cache(seq_len.max(self.max_seq_len))?;
        }
        
        let cos = self.cos_cached.as_ref().unwrap()
            .slice(1, 0, seq_len, 1)?;
        let sin = self.sin_cached.as_ref().unwrap()
            .slice(1, 0, seq_len, 1)?;
            
        Ok((cos, sin))
    }
    
    fn update_cache(&mut self, seq_len: usize) -> Result<()> {
        let device = Device::Cpu; // Use appropriate device
        
        // Create frequency tensor
        let inv_freq = {
            let arange = Tensor::arange(0, self.dim as i64, 2, device)?
                .to_dtype(DataType::Float32)?;
            let inv_freq_base = Tensor::scalar(self.base)?
                .pow(&(arange.div(&Tensor::scalar(self.dim as f32)?)?)?)?;
            Tensor::scalar(1.0)?.div(&inv_freq_base)?
        };
        
        // Create position tensor
        let t = Tensor::arange(0, seq_len as i64, 1, device)?
            .to_dtype(DataType::Float32)?
            .unsqueeze(1)?;
        
        // Compute frequencies
        let freqs = t.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], -1)?;
        
        self.cos_cached = Some(emb.cos()?.unsqueeze(0)?.unsqueeze(0)?);
        self.sin_cached = Some(emb.sin()?.unsqueeze(0)?.unsqueeze(0)?);
        
        Ok(())
    }
}

pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // Apply rotary position embedding to query and key tensors
    let q_embed = (q.mul(cos)?)?.add(&rotate_half(q)?.mul(sin)?)?;
    let k_embed = (k.mul(cos)?)?.add(&rotate_half(k)?.mul(sin)?)?;
    
    Ok((q_embed, k_embed))
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let shape = x.shape();
    let last_dim = shape.len() - 1;
    let half_dim = shape[last_dim] / 2;
    
    let x1 = x.slice(last_dim, 0, half_dim, 1)?;
    let x2 = x.slice(last_dim, half_dim, shape[last_dim], 1)?;
    
    Tensor::cat(&[&x2.neg()?, &x1], last_dim)
}
```

### Step 9: Implement Group Query Attention

**src/mistral/attention.rs**:
```rust
use super::{config::MistralConfig, layers::{RMSNorm, RotaryEmbedding, apply_rotary_pos_emb}};
use trustformers_core::{Result, Tensor, Module, Linear};

pub struct MistralAttention {
    config: MistralConfig,
    hidden_size: usize,
    num_heads: usize,
    num_key_value_heads: usize,
    num_key_value_groups: usize,
    head_dim: usize,
    
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    
    rotary_emb: RotaryEmbedding,
}

impl MistralAttention {
    pub fn new(config: &MistralConfig, layer_idx: Option<usize>) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = hidden_size / num_heads;
        
        Ok(Self {
            config: config.clone(),
            hidden_size,
            num_heads,
            num_key_value_heads,
            num_key_value_groups: num_heads / num_key_value_heads,
            head_dim,
            
            q_proj: Linear::new(hidden_size, num_heads * head_dim, false)?,
            k_proj: Linear::new(hidden_size, num_key_value_heads * head_dim, false)?,
            v_proj: Linear::new(hidden_size, num_key_value_heads * head_dim, false)?,
            o_proj: Linear::new(num_heads * head_dim, hidden_size, false)?,
            
            rotary_emb: RotaryEmbedding::new(
                head_dim,
                config.max_position_embeddings,
                config.rope_theta,
            )?,
        })
    }
    
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_value: Option<(&Tensor, &Tensor)>,
        output_attentions: bool,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>, Option<Tensor>)> {
        let (bsz, q_len, _) = hidden_states.shape3()?;
        
        // Project to Q, K, V
        let query_states = self.q_proj.forward(hidden_states)?
            .view(&[bsz, q_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
            
        let key_states = self.k_proj.forward(hidden_states)?
            .view(&[bsz, q_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;
            
        let value_states = self.v_proj.forward(hidden_states)?
            .view(&[bsz, q_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;
        
        // Apply rotary position embedding
        let kv_seq_len = if let Some((past_k, _)) = past_key_value {
            past_k.shape()[2] + q_len
        } else {
            q_len
        };
        
        let (cos, sin) = self.rotary_emb.forward(&query_states, kv_seq_len)?;
        let (query_states, key_states) = apply_rotary_pos_emb(
            &query_states, &key_states, &cos, &sin
        )?;
        
        // Handle past key values for generation
        let (key_states, value_states) = if let Some((past_k, past_v)) = past_key_value {
            let key_states = Tensor::cat(&[past_k, &key_states], 2)?;
            let value_states = Tensor::cat(&[past_v, &value_states], 2)?;
            (key_states, value_states)
        } else {
            (key_states, value_states)
        };
        
        // Store for next iteration if using cache
        let present_key_value = if use_cache {
            Some((key_states.clone(), value_states.clone()))
        } else {
            None
        };
        
        // Repeat K,V for Group Query Attention
        let key_states = self.repeat_kv(&key_states)?;
        let value_states = self.repeat_kv(&value_states)?;
        
        // Compute attention
        let attn_output = self.compute_attention(
            &query_states,
            &key_states,
            &value_states,
            attention_mask,
        )?;
        
        // Project output
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .view(&[bsz, q_len, self.hidden_size])?;
        let attn_output = self.o_proj.forward(&attn_output)?;
        
        Ok((attn_output, present_key_value, None))
    }
    
    fn repeat_kv(&self, hidden_states: &Tensor) -> Result<Tensor> {
        if self.num_key_value_groups == 1 {
            return Ok(hidden_states.clone());
        }
        
        let (batch, num_key_value_heads, slen, head_dim) = hidden_states.shape4()?;
        
        hidden_states
            .unsqueeze(2)?
            .expand(&[batch, num_key_value_heads, self.num_key_value_groups, slen, head_dim])?
            .reshape(&[batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])
    }
    
    fn compute_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (bsz, num_heads, q_len, head_dim) = query.shape4()?;
        let kv_seq_len = key.shape()[2];
        
        // Compute attention scores
        let attn_weights = query.matmul(&key.transpose(-2, -1)?)?
            .div(&Tensor::scalar((head_dim as f32).sqrt())?)?;
        
        // Apply sliding window mask if configured
        let attn_weights = if let Some(window_size) = self.config.sliding_window {
            self.apply_sliding_window_mask(attn_weights, window_size, q_len, kv_seq_len)?
        } else {
            attn_weights
        };
        
        // Apply causal mask
        let attn_weights = self.apply_causal_mask(attn_weights, q_len, kv_seq_len)?;
        
        // Apply attention mask if provided
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.add(mask)?
        } else {
            attn_weights
        };
        
        // Softmax and apply to values
        let attn_weights = attn_weights.softmax(-1)?;
        let attn_output = attn_weights.matmul(value)?;
        
        Ok(attn_output)
    }
    
    fn apply_sliding_window_mask(
        &self,
        mut attn_weights: Tensor,
        window_size: usize,
        q_len: usize,
        kv_seq_len: usize,
    ) -> Result<Tensor> {
        // Create sliding window mask
        for i in 0..q_len {
            let start = if i + kv_seq_len >= window_size {
                i + kv_seq_len - window_size
            } else {
                0
            };
            let end = i + kv_seq_len;
            
            // Mask positions outside the sliding window
            if start > 0 {
                let mask_value = Tensor::scalar(f32::NEG_INFINITY)?;
                attn_weights = attn_weights.slice_assign(
                    &[-1, -1, i as i64, -1],
                    &[0, start],
                    &mask_value,
                )?;
            }
        }
        
        Ok(attn_weights)
    }
    
    fn apply_causal_mask(
        &self,
        attn_weights: Tensor,
        q_len: usize,
        kv_seq_len: usize,
    ) -> Result<Tensor> {
        // Create causal mask
        let causal_mask = Tensor::ones(&[q_len, kv_seq_len])?
            .tril(kv_seq_len as i64 - q_len as i64)?;
        let causal_mask = Tensor::where_scalar(
            &causal_mask,
            0.0,
            f32::NEG_INFINITY,
        )?;
        
        attn_weights.add(&causal_mask.unsqueeze(0)?.unsqueeze(0)?)
    }
}

impl Module for MistralAttention {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.o_proj.parameters());
        params
    }
}
```

### Step 10: Implement MLP with SwiGLU

```rust
pub struct MistralMLP {
    config: MistralConfig,
    hidden_size: usize,
    intermediate_size: usize,
    
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MistralMLP {
    pub fn new(config: &MistralConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            
            gate_proj: Linear::new(config.hidden_size, config.intermediate_size, false)?,
            up_proj: Linear::new(config.hidden_size, config.intermediate_size, false)?,
            down_proj: Linear::new(config.intermediate_size, config.hidden_size, false)?,
            act_fn: config.hidden_act.clone(),
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU activation: swish(gate) * up
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        
        let activated_gate = self.act_fn.forward(&gate)?;
        let gated = activated_gate.mul(&up)?;
        
        self.down_proj.forward(&gated)
    }
}

impl Module for MistralMLP {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.gate_proj.parameters());
        params.extend(self.up_proj.parameters());
        params.extend(self.down_proj.parameters());
        params
    }
}
```

## Model Integration

### Step 11: Implement Complete Model

**src/mistral/model.rs**:
```rust
use super::{
    config::MistralConfig,
    attention::MistralAttention,
    layers::{RMSNorm, MistralMLP},
};
use crate::{ModelBase, ModelOutput, PreTrainedModel};
use trustformers_core::{Result, Tensor, Module, ModuleList, Embedding};

pub struct MistralDecoderLayer {
    self_attn: MistralAttention,
    mlp: MistralMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl MistralDecoderLayer {
    pub fn new(config: &MistralConfig, layer_idx: usize) -> Result<Self> {
        Ok(Self {
            self_attn: MistralAttention::new(config, Some(layer_idx))?,
            mlp: MistralMLP::new(config)?,
            input_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps)?,
            post_attention_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps)?,
        })
    }
    
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_value: Option<(&Tensor, &Tensor)>,
        output_attentions: bool,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>, Option<Tensor>)> {
        let residual = hidden_states.clone();
        
        // Pre-attention normalization
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        
        // Self-attention
        let (hidden_states, present_key_value, attn_weights) = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
        )?;
        
        // Residual connection
        let hidden_states = hidden_states.add(&residual)?;
        
        // MLP
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = hidden_states.add(&residual)?;
        
        Ok((hidden_states, present_key_value, attn_weights))
    }
}

pub struct MistralModel {
    config: MistralConfig,
    padding_idx: Option<usize>,
    vocab_size: usize,
    
    embed_tokens: Embedding,
    layers: ModuleList<MistralDecoderLayer>,
    norm: RMSNorm,
}

impl MistralModel {
    pub fn new(config: MistralConfig) -> Result<Self> {
        config.validate()?;
        
        let mut layers = ModuleList::new();
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(MistralDecoderLayer::new(&config, layer_idx)?);
        }
        
        Ok(Self {
            padding_idx: config.pad_token_id,
            vocab_size: config.vocab_size,
            embed_tokens: Embedding::new(config.vocab_size, config.hidden_size)?,
            layers,
            norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps)?,
            config,
        })
    }
    
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_values: Option<Vec<(Tensor, Tensor)>>,
        inputs_embeds: Option<&Tensor>,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> Result<MistralModelOutput> {
        // Get input embeddings
        let hidden_states = if let Some(embeds) = inputs_embeds {
            embeds.clone()
        } else {
            self.embed_tokens.forward(input_ids)?
        };
        
        let (batch_size, seq_length, _) = hidden_states.shape3()?;
        
        // Prepare position IDs
        let position_ids = if let Some(pos_ids) = position_ids {
            pos_ids.clone()
        } else {
            let past_length = past_key_values.as_ref()
                .map(|cache| cache[0].0.shape()[2])
                .unwrap_or(0);
            
            Tensor::arange(
                past_length as i64,
                (seq_length + past_length) as i64,
                1,
            )?.unsqueeze(0)?.expand(&[batch_size, seq_length])?
        };
        
        // Prepare attention mask
        let attention_mask = if attention_mask.is_some() {
            self.prepare_decoder_attention_mask(
                attention_mask,
                &hidden_states,
                past_key_values.as_ref().map(|cache| cache[0].0.shape()[2]).unwrap_or(0),
            )?
        } else {
            None
        };
        
        // Forward through layers
        let mut hidden_states = hidden_states;
        let mut next_decoder_cache = if use_cache { Some(Vec::new()) } else { None };
        let mut all_hidden_states = if output_hidden_states { Some(Vec::new()) } else { None };
        let mut all_self_attns = if output_attentions { Some(Vec::new()) } else { None };
        
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            if let Some(ref mut hidden_states_vec) = all_hidden_states {
                hidden_states_vec.push(hidden_states.clone());
            }
            
            let past_key_value = past_key_values.as_ref()
                .and_then(|cache| cache.get(layer_idx))
                .map(|(k, v)| (k.as_ref(), v.as_ref()));
            
            let (layer_outputs, present_key_value, attn_weights) = layer.forward(
                &hidden_states,
                attention_mask.as_ref(),
                Some(&position_ids),
                past_key_value,
                output_attentions,
                use_cache,
            )?;
            
            hidden_states = layer_outputs;
            
            if let (Some(ref mut cache), Some(present)) = (&mut next_decoder_cache, present_key_value) {
                cache.push(present);
            }
            
            if let (Some(ref mut attns), Some(attn)) = (&mut all_self_attns, attn_weights) {
                attns.push(attn);
            }
        }
        
        // Final normalization
        let hidden_states = self.norm.forward(&hidden_states)?;
        
        if let Some(ref mut hidden_states_vec) = all_hidden_states {
            hidden_states_vec.push(hidden_states.clone());
        }
        
        Ok(MistralModelOutput {
            last_hidden_state: hidden_states,
            past_key_values: next_decoder_cache,
            hidden_states: all_hidden_states,
            attentions: all_self_attns,
        })
    }
    
    fn prepare_decoder_attention_mask(
        &self,
        attention_mask: Option<&Tensor>,
        hidden_states: &Tensor,
        past_key_values_length: usize,
    ) -> Result<Option<Tensor>> {
        // Implementation details for attention mask preparation
        // This is complex and involves creating causal masks, handling padding, etc.
        
        let (batch_size, seq_length) = (hidden_states.shape()[0], hidden_states.shape()[1]);
        
        // Create causal mask
        let causal_mask = self.make_causal_mask(
            batch_size,
            seq_length,
            past_key_values_length,
        )?;
        
        // Combine with attention mask if provided
        if let Some(attn_mask) = attention_mask {
            let expanded_mask = attn_mask.unsqueeze(1)?.unsqueeze(2)?;
            let combined_mask = causal_mask.add(&expanded_mask)?;
            Ok(Some(combined_mask))
        } else {
            Ok(Some(causal_mask))
        }
    }
    
    fn make_causal_mask(
        &self,
        batch_size: usize,
        target_length: usize,
        past_length: usize,
    ) -> Result<Tensor> {
        let total_length = target_length + past_length;
        
        // Create causal mask
        let mask = Tensor::ones(&[target_length, total_length])?;
        let mask = mask.tril(past_length as i64)?;
        
        // Convert to attention mask format (0 for attend, large negative for mask)
        let mask = Tensor::where_scalar(&mask, 0.0, f32::NEG_INFINITY)?;
        
        // Add batch dimension
        mask.unsqueeze(0)?.unsqueeze(0)?.expand(&[batch_size, 1, target_length, total_length])
    }
}

#[derive(Debug)]
pub struct MistralModelOutput {
    pub last_hidden_state: Tensor,
    pub past_key_values: Option<Vec<(Tensor, Tensor)>>,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}

impl ModelOutput for MistralModelOutput {
    fn logits(&self) -> Option<&Tensor> {
        None
    }
    
    fn loss(&self) -> Option<&Tensor> {
        None
    }
    
    fn hidden_states(&self) -> Option<&Vec<Tensor>> {
        self.hidden_states.as_ref()
    }
    
    fn attentions(&self) -> Option<&Vec<Tensor>> {
        self.attentions.as_ref()
    }
}

impl ModelBase for MistralModel {
    type Config = MistralConfig;
    type Output = MistralModelOutput;
    
    fn new(config: Self::Config) -> Result<Self> {
        Self::new(config)
    }
    
    fn config(&self) -> &Self::Config {
        &self.config
    }
    
    fn forward(&mut self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Self::Output> {
        self.forward(
            input_ids,
            attention_mask,
            None, // position_ids
            None, // past_key_values
            None, // inputs_embeds
            false, // use_cache
            false, // output_attentions
            false, // output_hidden_states
        )
    }
}

impl Module for MistralModel {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        
        params.extend(self.embed_tokens.parameters());
        
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        
        params.extend(self.norm.parameters());
        
        params
    }
}
```

This comprehensive tutorial provides a complete foundation for implementing new models in TrustformeRS. The remaining sections would cover task-specific heads, testing strategies, documentation, advanced features, and deployment considerations.

Would you like me to continue with the remaining sections of this tutorial?