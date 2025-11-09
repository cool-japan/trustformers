# Architecture Pattern Library

This document catalogs the common architectural patterns, components, and design principles used across TrustformeRS model implementations.

## Table of Contents

1. [Configuration Patterns](#configuration-patterns)
2. [Model Architecture Patterns](#model-architecture-patterns)
3. [Attention Mechanisms](#attention-mechanisms)
4. [Normalization Layers](#normalization-layers)
5. [Activation Functions](#activation-functions)
6. [Position Encodings](#position-encodings)
7. [Feed-Forward Networks](#feed-forward-networks)
8. [Generation Patterns](#generation-patterns)
9. [Training Patterns](#training-patterns)
10. [Testing Patterns](#testing-patterns)

## Configuration Patterns

### Standard Configuration Structure

All model configurations follow a common pattern:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Core architecture parameters
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    
    // Model-specific parameters
    // ...
    
    // Training and generation parameters
    pub initializer_range: f32,
    pub use_cache: bool,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        // Provide sensible defaults
    }
}

impl Config for ModelConfig {
    fn validate(&self) -> Result<()> {
        // Validate configuration consistency
    }
}
```

### Configuration Categories

#### 1. Encoder-Only Models (BERT family)
```rust
pub struct EncoderConfig {
    // Standard parameters
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    
    // Encoder-specific
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub position_embedding_type: String,
    pub use_position_embeddings: bool,
    pub use_token_type_embeddings: bool,
    
    // Normalization
    pub layer_norm_eps: f32,
    
    // Dropout
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
}
```

#### 2. Decoder-Only Models (GPT family)
```rust
pub struct DecoderConfig {
    // Standard parameters
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    
    // Decoder-specific
    pub max_position_embeddings: usize,
    pub context_length: usize,
    
    // Modern features
    pub num_key_value_heads: Option<usize>, // For GQA
    pub sliding_window: Option<usize>,      // For local attention
    pub rope_theta: f32,                    // For RoPE
    
    // Normalization (often RMSNorm for modern models)
    pub rms_norm_eps: f32,
    
    // Generation
    pub use_cache: bool,
}
```

#### 3. Encoder-Decoder Models (T5 family)
```rust
pub struct EncoderDecoderConfig {
    // Shared parameters
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    
    // Encoder-decoder specific
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub dropout_rate: f32,
    pub layer_norm_epsilon: f32,
    
    // T5-specific
    pub feed_forward_proj: String, // "relu" or "gated-gelu"
    pub is_encoder_decoder: bool,
    pub use_cache: bool,
}
```

### Configuration Validation Patterns

```rust
impl Config for ModelConfig {
    fn validate(&self) -> Result<()> {
        // Head dimension divisibility
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ConfigError::InvalidDimensions(
                "hidden_size must be divisible by num_attention_heads"
            ));
        }
        
        // Grouped Query Attention validation
        if let Some(kv_heads) = self.num_key_value_heads {
            if self.num_attention_heads % kv_heads != 0 {
                return Err(ConfigError::InvalidGQA(
                    "num_attention_heads must be divisible by num_key_value_heads"
                ));
            }
        }
        
        // Vocabulary size validation
        if self.vocab_size == 0 {
            return Err(ConfigError::InvalidVocabSize);
        }
        
        // Context length validation
        if self.max_position_embeddings == 0 {
            return Err(ConfigError::InvalidContextLength);
        }
        
        Ok(())
    }
}
```

## Model Architecture Patterns

### 1. Standard Transformer Layer

```rust
pub struct TransformerLayer {
    self_attention: MultiHeadAttention,
    cross_attention: Option<MultiHeadAttention>, // For encoder-decoder
    feed_forward: FeedForward,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    dropout: Dropout,
}

impl TransformerLayer {
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>, // For cross-attention
        past_key_value: Option<(&Tensor, &Tensor)>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        // Pre-norm (modern) or post-norm (traditional)
        let normed_input = self.input_layernorm.forward(hidden_states)?;
        
        // Self-attention
        let (attn_output, new_kv) = self.self_attention.forward(
            &normed_input,
            &normed_input,
            &normed_input,
            attention_mask,
            past_key_value,
            use_cache,
        )?;
        
        // Residual connection
        let hidden_states = hidden_states.add(&self.dropout.forward(&attn_output)?)?;
        
        // Cross-attention (if present)
        let hidden_states = if let (Some(cross_attn), Some(encoder_states)) = 
            (&self.cross_attention, encoder_hidden_states) {
            let normed = self.post_attention_layernorm.forward(&hidden_states)?;
            let (cross_output, _) = cross_attn.forward(
                &normed,
                encoder_states,
                encoder_states,
                None,
                None,
                false,
            )?;
            hidden_states.add(&self.dropout.forward(&cross_output)?)?
        } else {
            hidden_states
        };
        
        // Feed-forward
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let ff_output = self.feed_forward.forward(&normed)?;
        let output = hidden_states.add(&self.dropout.forward(&ff_output)?)?;
        
        Ok((output, new_kv))
    }
}
```

### 2. Modern Pre-Norm Architecture (LLaMA style)

```rust
pub struct ModernTransformerLayer {
    self_attn: GroupedQueryAttention,
    mlp: SwiGLUMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl ModernTransformerLayer {
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_value: Option<(&Tensor, &Tensor)>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let residual = hidden_states.clone();
        
        // Pre-attention normalization
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        
        // Self-attention with RoPE
        let (hidden_states, present_kv) = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
        )?;
        
        // Residual connection
        let hidden_states = hidden_states.add(&residual)?;
        
        // MLP block
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = hidden_states.add(&residual)?;
        
        Ok((hidden_states, present_kv))
    }
}
```

### 3. Parallel Architecture (PaLM style)

```rust
pub struct ParallelTransformerLayer {
    attention_norm: RMSNorm,
    ffn_norm: RMSNorm,
    self_attn: MultiHeadAttention,
    mlp: ParallelMLP,
}

impl ParallelTransformerLayer {
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Parallel attention and MLP computation
        let attn_input = self.attention_norm.forward(hidden_states)?;
        let ffn_input = self.ffn_norm.forward(hidden_states)?;
        
        let attn_output = self.self_attn.forward(&attn_input)?;
        let ffn_output = self.mlp.forward(&ffn_input)?;
        
        // Combine parallel outputs
        hidden_states.add(&attn_output)?.add(&ffn_output)
    }
}
```

## Attention Mechanisms

### 1. Multi-Head Attention (Standard)

```rust
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    
    attention_dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        past_key_value: Option<(&Tensor, &Tensor)>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (batch_size, seq_len, _) = query.shape3()?;
        
        // Project to Q, K, V
        let q = self.q_proj.forward(query)?
            .view(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = self.k_proj.forward(key)?
            .view(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = self.v_proj.forward(value)?
            .view(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        
        // Handle past key-values
        let (k, v) = if let Some((past_k, past_v)) = past_key_value {
            (Tensor::cat(&[past_k, &k], 2)?, Tensor::cat(&[past_v, &v], 2)?)
        } else {
            (k, v)
        };
        
        // Compute attention
        let attn_weights = q.matmul(&k.transpose(-2, -1)?)?
            .mul_scalar(self.scale)?;
        
        // Apply mask
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.add(mask)?
        } else {
            attn_weights
        };
        
        let attn_weights = attn_weights.softmax(-1)?;
        let attn_weights = self.attention_dropout.forward(&attn_weights)?;
        
        // Apply to values
        let attn_output = attn_weights.matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .view(&[batch_size, seq_len, self.num_heads * self.head_dim])?;
        
        let output = self.o_proj.forward(&attn_output)?;
        
        let present = if use_cache { Some((k, v)) } else { None };
        
        Ok((output, present))
    }
}
```

### 2. Grouped Query Attention (GQA)

```rust
pub struct GroupedQueryAttention {
    num_heads: usize,
    num_key_value_heads: usize,
    num_groups: usize,
    head_dim: usize,
    scale: f32,
    
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    
    rotary_emb: RotaryEmbedding,
}

impl GroupedQueryAttention {
    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_groups == 1 {
            return Ok(x.clone());
        }
        
        let (batch, num_kv_heads, seq_len, head_dim) = x.shape4()?;
        x.unsqueeze(2)?
            .expand(&[batch, num_kv_heads, self.num_groups, seq_len, head_dim])?
            .reshape(&[batch, num_kv_heads * self.num_groups, seq_len, head_dim])
    }
    
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_value: Option<(&Tensor, &Tensor)>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (batch_size, seq_len, _) = hidden_states.shape3()?;
        
        // Project with different dimensions for K, V
        let q = self.q_proj.forward(hidden_states)?
            .view(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = self.k_proj.forward(hidden_states)?
            .view(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = self.v_proj.forward(hidden_states)?
            .view(&[batch_size, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;
        
        // Apply rotary position embedding
        let (cos, sin) = self.rotary_emb.forward(&q, seq_len)?;
        let (q, k) = apply_rotary_pos_emb(&q, &k, &cos, &sin)?;
        
        // Handle cache
        let (k, v) = if let Some((past_k, past_v)) = past_key_value {
            (Tensor::cat(&[past_k, &k], 2)?, Tensor::cat(&[past_v, &v], 2)?)
        } else {
            (k, v)
        };
        
        // Repeat K, V for grouped attention
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;
        
        // Standard attention computation
        let attn_output = self.compute_attention(&q, &k, &v, attention_mask)?;
        
        let output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .view(&[batch_size, seq_len, self.num_heads * self.head_dim])?;
        let output = self.o_proj.forward(&output)?;
        
        let present = if use_cache { Some((k, v)) } else { None };
        
        Ok((output, present))
    }
}
```

### 3. Sliding Window Attention

```rust
pub struct SlidingWindowAttention {
    window_size: usize,
    base_attention: MultiHeadAttention,
}

impl SlidingWindowAttention {
    fn create_sliding_window_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mut mask = Tensor::full(&[seq_len, seq_len], f32::NEG_INFINITY)?;
        
        for i in 0..seq_len {
            let start = i.saturating_sub(self.window_size);
            let end = (i + self.window_size + 1).min(seq_len);
            
            for j in start..end {
                if j <= i { // Causal constraint
                    mask = mask.slice_assign(&[i as i64, j as i64], &Tensor::scalar(0.0)?)?;
                }
            }
        }
        
        Ok(mask)
    }
    
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq_len = hidden_states.shape()[1];
        let sliding_mask = self.create_sliding_window_mask(seq_len)?;
        
        let combined_mask = if let Some(mask) = attention_mask {
            sliding_mask.add(mask)?
        } else {
            sliding_mask
        };
        
        let (output, _) = self.base_attention.forward(
            hidden_states,
            hidden_states,
            hidden_states,
            Some(&combined_mask),
            None,
            false,
        )?;
        
        Ok(output)
    }
}
```

## Normalization Layers

### 1. Layer Normalization

```rust
pub struct LayerNorm {
    weight: Parameter,
    bias: Parameter,
    eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: &[usize], eps: f32) -> Result<Self> {
        Ok(Self {
            weight: Parameter::ones(normalized_shape)?,
            bias: Parameter::zeros(normalized_shape)?,
            eps,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mean = input.mean_dim(&[-1], true)?;
        let var = input.var_dim(&[-1], true, false)?;
        
        let normalized = input.sub(&mean)?
            .div(&(var + self.eps)?.sqrt()?)?;
        
        normalized.mul(&self.weight)?.add(&self.bias)
    }
}
```

### 2. RMS Normalization

```rust
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
    
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let input_dtype = hidden_states.dtype();
        let variance = hidden_states.to_dtype(DataType::Float32)?
            .pow(&Tensor::scalar(2.0)?)?
            .mean_dim(&[-1], true)?;
        
        let hidden_states = hidden_states.to_dtype(DataType::Float32)?
            .div(&(variance + self.eps)?.sqrt()?)?;
        
        self.weight.mul(&hidden_states)?.to_dtype(input_dtype)
    }
}
```

### 3. Group Normalization

```rust
pub struct GroupNorm {
    num_groups: usize,
    num_channels: usize,
    weight: Parameter,
    bias: Parameter,
    eps: f32,
}

impl GroupNorm {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch_size, channels, height, width) = input.shape4()?;
        let group_size = channels / self.num_groups;
        
        // Reshape for group normalization
        let input = input.view(&[batch_size, self.num_groups, group_size, height, width])?;
        
        // Compute statistics per group
        let mean = input.mean_dim(&[2, 3, 4], true)?;
        let var = input.var_dim(&[2, 3, 4], true, false)?;
        
        // Normalize
        let normalized = input.sub(&mean)?
            .div(&(var + self.eps)?.sqrt()?)?
            .view(&[batch_size, channels, height, width])?;
        
        // Apply learned parameters
        let weight = self.weight.view(&[1, channels, 1, 1])?;
        let bias = self.bias.view(&[1, channels, 1, 1])?;
        
        normalized.mul(&weight)?.add(&bias)
    }
}
```

## Activation Functions

### Common Activation Patterns

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    GELU,
    SiLU,      // Swish
    Tanh,
    Sigmoid,
    Mish,
    GLU,       // Gated Linear Unit
    SwiGLU,    // SiLU-gated GLU
    GeGLU,     // GELU-gated GLU
    NewGELU,   // Approximate GELU
}

impl ActivationFunction {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::ReLU => x.relu(),
            Self::GELU => x.gelu(),
            Self::SiLU => x.silu(),
            Self::Tanh => x.tanh(),
            Self::Sigmoid => x.sigmoid(),
            Self::Mish => x.mish(),
            Self::GLU => {
                let (a, b) = x.chunk(2, -1)?;
                a.mul(&b.sigmoid()?)
            },
            Self::SwiGLU => {
                let (gate, up) = x.chunk(2, -1)?;
                gate.silu()?.mul(&up)
            },
            Self::GeGLU => {
                let (gate, up) = x.chunk(2, -1)?;
                gate.gelu()?.mul(&up)
            },
            Self::NewGELU => {
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                let c1 = Tensor::scalar((2.0 / std::f32::consts::PI).sqrt())?;
                let c2 = Tensor::scalar(0.044715)?;
                let inner = x.add(&x.pow(&Tensor::scalar(3.0)?)?.mul(&c2)?)?
                    .mul(&c1)?
                    .tanh()?
                    .add(&Tensor::scalar(1.0)?)?;
                x.mul(&Tensor::scalar(0.5)?)?.mul(&inner)
            },
        }
    }
}
```

## Position Encodings

### 1. Sinusoidal Position Encoding

```rust
pub struct SinusoidalPositionEncoding {
    max_len: usize,
    d_model: usize,
    encoding: Tensor,
}

impl SinusoidalPositionEncoding {
    pub fn new(max_len: usize, d_model: usize) -> Result<Self> {
        let mut encoding = Tensor::zeros(&[max_len, d_model])?;
        
        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let angle = pos as f32 / 10000_f32.powf(2.0 * i as f32 / d_model as f32);
                encoding = encoding.slice_assign(
                    &[pos as i64, (2 * i) as i64],
                    &Tensor::scalar(angle.sin())?
                )?;
                encoding = encoding.slice_assign(
                    &[pos as i64, (2 * i + 1) as i64],
                    &Tensor::scalar(angle.cos())?
                )?;
            }
        }
        
        Ok(Self { max_len, d_model, encoding })
    }
    
    pub fn forward(&self, seq_len: usize) -> Result<Tensor> {
        self.encoding.slice(0, 0, seq_len.min(self.max_len) as i64, 1)
    }
}
```

### 2. Rotary Position Embedding (RoPE)

```rust
pub struct RotaryPositionEmbedding {
    dim: usize,
    max_position_embeddings: usize,
    base: f32,
    inv_freq: Tensor,
}

impl RotaryPositionEmbedding {
    pub fn new(dim: usize, max_position_embeddings: usize, base: f32) -> Result<Self> {
        let inv_freq = Tensor::arange(0, dim as i64, 2)?
            .to_dtype(DataType::Float32)?
            .div(&Tensor::scalar(dim as f32)?)?;
        let inv_freq = Tensor::scalar(base)?
            .pow(&inv_freq)?
            .reciprocal()?;
        
        Ok(Self {
            dim,
            max_position_embeddings,
            base,
            inv_freq,
        })
    }
    
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        // Create position indices
        let t = Tensor::arange(0, seq_len as i64, 1)?
            .to_dtype(DataType::Float32)?
            .unsqueeze(1)?;
        
        // Compute frequencies
        let freqs = t.matmul(&self.inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], -1)?;
        
        let cos = emb.cos()?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = emb.sin()?.unsqueeze(0)?.unsqueeze(0)?;
        
        Ok((cos, sin))
    }
}

pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let q_embed = q.mul(cos)?.add(&rotate_half(q)?.mul(sin)?)?;
    let k_embed = k.mul(cos)?.add(&rotate_half(k)?.mul(sin)?)?;
    Ok((q_embed, k_embed))
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let shape = x.shape();
    let last_dim = shape.len() - 1;
    let half_dim = shape[last_dim] / 2;
    
    let x1 = x.slice(last_dim, 0, half_dim as i64, 1)?;
    let x2 = x.slice(last_dim, half_dim as i64, shape[last_dim] as i64, 1)?;
    
    Tensor::cat(&[&x2.neg()?, &x1], last_dim)
}
```

### 3. Learned Position Embeddings

```rust
pub struct LearnedPositionEmbedding {
    embeddings: Embedding,
    max_position_embeddings: usize,
}

impl LearnedPositionEmbedding {
    pub fn new(max_position_embeddings: usize, hidden_size: usize) -> Result<Self> {
        Ok(Self {
            embeddings: Embedding::new(max_position_embeddings, hidden_size)?,
            max_position_embeddings,
        })
    }
    
    pub fn forward(&self, position_ids: &Tensor) -> Result<Tensor> {
        self.embeddings.forward(position_ids)
    }
    
    pub fn create_position_ids(&self, seq_len: usize, batch_size: usize) -> Result<Tensor> {
        Tensor::arange(0, seq_len as i64, 1)?
            .unsqueeze(0)?
            .expand(&[batch_size, seq_len])
    }
}
```

## Feed-Forward Networks

### 1. Standard FFN

```rust
pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    activation: ActivationFunction,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: ActivationFunction,
        dropout: f32,
    ) -> Result<Self> {
        Ok(Self {
            linear1: Linear::new(hidden_size, intermediate_size, true)?,
            linear2: Linear::new(intermediate_size, hidden_size, true)?,
            activation,
            dropout: Dropout::new(dropout),
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = self.activation.forward(&x)?;
        let x = self.dropout.forward(&x)?;
        self.linear2.forward(&x)
    }
}
```

### 2. SwiGLU FFN (Modern)

```rust
pub struct SwiGLUFFN {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGLUFFN {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::new(hidden_size, intermediate_size, false)?,
            up_proj: Linear::new(hidden_size, intermediate_size, false)?,
            down_proj: Linear::new(intermediate_size, hidden_size, false)?,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let gated = gate.mul(&up)?;
        self.down_proj.forward(&gated)
    }
}
```

### 3. Mixture of Experts (MoE) FFN

```rust
pub struct MixtureOfExpertsFFN {
    experts: Vec<FeedForward>,
    gate: Linear,
    num_experts: usize,
    top_k: usize,
}

impl MixtureOfExpertsFFN {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let hidden_size = x.shape()[2];
        
        // Reshape for expert routing
        let x_flat = x.view(&[batch_size * seq_len, hidden_size])?;
        
        // Compute gating weights
        let gate_logits = self.gate.forward(&x_flat)?;
        let (top_k_weights, top_k_indices) = gate_logits.topk(self.top_k, -1, false)?;
        let top_k_weights = top_k_weights.softmax(-1)?;
        
        // Route to experts
        let mut output = Tensor::zeros_like(&x_flat)?;
        
        for expert_idx in 0..self.num_experts {
            // Find tokens routed to this expert
            let mask = top_k_indices.eq(&Tensor::scalar(expert_idx as f32)?)?;
            let expert_tokens = x_flat.masked_select(&mask)?;
            
            if expert_tokens.numel() > 0 {
                let expert_output = self.experts[expert_idx].forward(&expert_tokens)?;
                let weights = top_k_weights.masked_select(&mask)?;
                let weighted_output = expert_output.mul(&weights.unsqueeze(-1)?)?;
                
                // Scatter back to original positions
                output = output.masked_scatter(&mask, &weighted_output)?;
            }
        }
        
        output.view(&[batch_size, seq_len, hidden_size])
    }
}
```

## Generation Patterns

### 1. Autoregressive Generation

```rust
pub trait GenerativeModel {
    fn generate(
        &mut self,
        input_ids: &Tensor,
        generation_config: &GenerationConfig,
    ) -> Result<GenerationOutput>;
    
    fn generate_stream(
        &mut self,
        input_ids: &Tensor,
        generation_config: &GenerationConfig,
    ) -> Result<impl Stream<Item = Result<Token>>>;
}

pub struct AutoregressiveGenerator {
    model: Box<dyn GenerativeModel>,
    tokenizer: Tokenizer,
}

impl AutoregressiveGenerator {
    pub fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt)?;
        let mut generated_ids = input_ids.clone();
        let mut past_key_values = None;
        
        for _ in 0..config.max_new_tokens {
            // Prepare input (only last token if using cache)
            let model_input = if config.use_cache && past_key_values.is_some() {
                generated_ids.slice(1, -1, generated_ids.shape()[1] as i64, 1)?
            } else {
                generated_ids.clone()
            };
            
            // Forward pass
            let output = self.model.forward(&model_input, None)?;
            
            // Update cache
            if config.use_cache {
                past_key_values = output.past_key_values;
            }
            
            // Sample next token
            let next_token = self.sample_next_token(&output.logits, config)?;
            
            // Check for EOS
            if let Some(eos_id) = config.eos_token_id {
                if next_token == eos_id {
                    break;
                }
            }
            
            // Append to sequence
            generated_ids = Tensor::cat(&[&generated_ids, &next_token.unsqueeze(0)?], 1)?;
        }
        
        let generated_text = self.tokenizer.decode(&generated_ids)?;
        Ok(generated_text)
    }
    
    fn sample_next_token(&self, logits: &Tensor, config: &GenerationConfig) -> Result<Tensor> {
        let mut logits = logits.slice(1, -1, logits.shape()[1] as i64, 1)?; // Last position
        
        // Apply temperature
        if config.temperature != 1.0 {
            logits = logits.div_scalar(config.temperature)?;
        }
        
        // Apply repetition penalty
        if config.repetition_penalty != 1.0 {
            // Implementation depends on tracking generated tokens
        }
        
        // Apply top-k filtering
        if let Some(top_k) = config.top_k {
            let (top_k_logits, top_k_indices) = logits.topk(top_k, -1, false)?;
            logits = Tensor::full_like(&logits, f32::NEG_INFINITY)?
                .scatter(-1, &top_k_indices, &top_k_logits)?;
        }
        
        // Apply top-p (nucleus) filtering
        if config.top_p < 1.0 {
            let sorted_logits = logits.sort(-1, true)?;
            let sorted_probs = sorted_logits.softmax(-1)?;
            let cumsum = sorted_probs.cumsum(-1)?;
            
            let mask = cumsum.le(&Tensor::scalar(config.top_p)?)?;
            // Keep at least one token
            let mask = mask.slice_assign(&[-1, 0], &Tensor::scalar(true)?)?;
            
            logits = Tensor::where_tensor(&mask, &sorted_logits, 
                &Tensor::scalar(f32::NEG_INFINITY)?)?;
        }
        
        // Sample
        if config.do_sample {
            let probs = logits.softmax(-1)?;
            probs.multinomial(1)
        } else {
            logits.argmax(-1, false)
        }
    }
}
```

### 2. Beam Search

```rust
pub struct BeamSearchGenerator {
    beam_size: usize,
    length_penalty: f32,
    early_stopping: bool,
}

impl BeamSearchGenerator {
    pub fn generate(
        &self,
        model: &mut dyn GenerativeModel,
        input_ids: &Tensor,
        max_length: usize,
    ) -> Result<Vec<Tensor>> {
        let batch_size = input_ids.shape()[0];
        let vocab_size = model.config().vocab_size();
        
        // Initialize beams
        let mut beams = vec![Beam::new(input_ids.clone(), 0.0)];
        let mut completed_beams = Vec::new();
        
        for step in 0..max_length {
            let mut all_candidates = Vec::new();
            
            for beam in &beams {
                if beam.is_complete() {
                    completed_beams.push(beam.clone());
                    continue;
                }
                
                // Generate from current beam
                let output = model.forward(&beam.sequence, None)?;
                let logits = output.logits.slice(1, -1, output.logits.shape()[1] as i64, 1)?;
                let log_probs = logits.log_softmax(-1)?;
                
                // Get top-k candidates
                let (top_scores, top_indices) = log_probs.topk(self.beam_size, -1, false)?;
                
                for i in 0..self.beam_size {
                    let token_score = top_scores.get_item(&[i])?;
                    let token_id = top_indices.get_item(&[i])? as u32;
                    
                    let new_sequence = Tensor::cat(&[
                        &beam.sequence,
                        &Tensor::scalar(token_id as f32)?.unsqueeze(0)?
                    ], 1)?;
                    
                    let new_score = beam.score + token_score;
                    let length_normalized_score = self.normalize_score(new_score, new_sequence.shape()[1]);
                    
                    all_candidates.push(Beam::new(new_sequence, length_normalized_score));
                }
            }
            
            // Select top beams
            all_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            beams = all_candidates.into_iter().take(self.beam_size).collect();
            
            // Early stopping check
            if self.early_stopping && beams.iter().all(|b| b.is_complete()) {
                completed_beams.extend(beams);
                break;
            }
        }
        
        // Return best completed sequences
        completed_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(completed_beams.into_iter().map(|b| b.sequence).collect())
    }
    
    fn normalize_score(&self, score: f32, length: usize) -> f32 {
        score / (length as f32).powf(self.length_penalty)
    }
}

#[derive(Clone)]
struct Beam {
    sequence: Tensor,
    score: f32,
}

impl Beam {
    fn new(sequence: Tensor, score: f32) -> Self {
        Self { sequence, score }
    }
    
    fn is_complete(&self) -> bool {
        // Check if sequence ends with EOS token
        false // Implementation depends on specific tokenizer
    }
}
```

## Testing Patterns

### 1. Model Configuration Tests

```rust
#[cfg(test)]
mod config_tests {
    use super::*;
    
    #[test]
    fn test_default_config_validity() {
        let config = ModelConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size % config.num_attention_heads, 0);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = ModelConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.hidden_size, deserialized.hidden_size);
    }
    
    #[test]
    fn test_invalid_configurations() {
        let mut config = ModelConfig::default();
        
        // Test invalid head dimensions
        config.num_attention_heads = config.hidden_size + 1;
        assert!(config.validate().is_err());
        
        // Test zero vocabulary
        config.vocab_size = 0;
        assert!(config.validate().is_err());
    }
}
```

### 2. Model Architecture Tests

```rust
#[cfg(test)]
mod model_tests {
    use super::*;
    
    #[test]
    fn test_model_creation() {
        let config = ModelConfig::default();
        let model = Model::new(config);
        assert!(model.is_ok());
    }
    
    #[test]
    fn test_forward_pass_shapes() {
        let config = ModelConfig {
            vocab_size: 1000,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            ..Default::default()
        };
        
        let mut model = Model::new(config).unwrap();
        
        let batch_size = 2;
        let seq_len = 10;
        let input_ids = Tensor::randint(0, 1000, &[batch_size, seq_len]).unwrap();
        
        let output = model.forward(&input_ids, None).unwrap();
        assert_eq!(output.last_hidden_state.shape(), &[batch_size, seq_len, 64]);
    }
    
    #[test]
    fn test_gradient_flow() {
        let config = ModelConfig::default();
        let mut model = Model::new(config).unwrap();
        
        let input_ids = Tensor::randint(0, 100, &[1, 5]).unwrap();
        let output = model.forward(&input_ids, None).unwrap();
        
        // Check that gradients can flow through the model
        let loss = output.last_hidden_state.mean()?;
        loss.backward()?;
        
        for param in model.parameters() {
            assert!(param.grad().is_some());
        }
    }
}
```

### 3. Numerical Accuracy Tests

```rust
#[cfg(test)]
mod accuracy_tests {
    use super::*;
    
    #[test]
    fn test_numerical_stability() {
        let config = ModelConfig::default();
        let mut model = Model::new(config).unwrap();
        
        // Test with various input ranges
        let test_cases = vec![
            Tensor::zeros(&[1, 10]).unwrap(),
            Tensor::ones(&[1, 10]).unwrap(),
            Tensor::randn(&[1, 10]).unwrap() * 100.0,
        ];
        
        for input in test_cases {
            let output = model.forward(&input, None).unwrap();
            
            // Check for NaN or Inf
            assert!(!output.last_hidden_state.isnan().any().unwrap());
            assert!(!output.last_hidden_state.isinf().any().unwrap());
            
            // Check reasonable output range
            let max_val = output.last_hidden_state.abs().max().unwrap();
            assert!(max_val < 1000.0); // Reasonable bound
        }
    }
    
    #[test]
    fn test_deterministic_output() {
        let config = ModelConfig::default();
        let mut model1 = Model::new(config.clone()).unwrap();
        let mut model2 = Model::new(config).unwrap();
        
        // Set same random seed
        torch::manual_seed(42);
        let input = Tensor::randint(0, 100, &[1, 5]).unwrap();
        
        let output1 = model1.forward(&input, None).unwrap();
        let output2 = model2.forward(&input, None).unwrap();
        
        let diff = (output1.last_hidden_state - output2.last_hidden_state).abs().max().unwrap();
        assert!(diff < 1e-6); // Should be identical
    }
}
```

This architecture pattern library provides comprehensive examples and patterns that can be reused across different model implementations in TrustformeRS. Each pattern includes implementation details, usage examples, and testing strategies.