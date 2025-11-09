# Model Implementation Tutorial

This tutorial guides you through implementing custom transformer models in TrustformeRS, from basic components to advanced architectures.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Basic Concepts](#basic-concepts)
3. [Implementing a Simple Transformer](#implementing-a-simple-transformer)
4. [Building Custom Layers](#building-custom-layers)
5. [Creating a BERT-like Model](#creating-a-bert-like-model)
6. [Implementing a GPT-like Model](#implementing-a-gpt-like-model)
7. [Advanced Techniques](#advanced-techniques)
8. [Testing and Validation](#testing-and-validation)
9. [Integration with TrustformeRS Ecosystem](#integration-with-trustformers-ecosystem)

## Prerequisites

Before starting, ensure you have:
- Basic understanding of transformer architecture
- Rust programming knowledge
- TrustformeRS installed

```toml
[dependencies]
trustformers = "0.1"
trustformers-core = "0.1"
trustformers-models = "0.1"
anyhow = "1.0"
```

## Basic Concepts

### Core Traits

Every model in TrustformeRS implements these core traits:

```rust
use trustformers_core::{Model, Config, Forward};

// Configuration trait
pub trait Config: Serialize + DeserializeOwned {
    fn validate(&self) -> Result<()>;
}

// Model trait
pub trait Model: Send + Sync {
    type Config: Config;
    type Input;
    type Output;
    
    fn new(config: Self::Config) -> Result<Self>
    where Self: Sized;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output>;
    
    fn from_pretrained(model_id: &str) -> Result<Self>
    where Self: Sized;
}
```

### Model Output Types

```rust
use trustformers_core::ModelOutput;
use std::collections::HashMap;

// Basic output
#[derive(Debug, Clone)]
pub struct BaseModelOutput {
    pub last_hidden_state: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}

// Classification output
#[derive(Debug, Clone)]
pub struct SequenceClassifierOutput {
    pub loss: Option<Tensor>,
    pub logits: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}
```

## Implementing a Simple Transformer

Let's build a basic transformer encoder from scratch:

### Step 1: Define Configuration

```rust
use serde::{Serialize, Deserialize};
use trustformers_core::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleTransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f32,
    pub pad_token_id: usize,
}

impl Default for SimpleTransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
        }
    }
}

impl Config for SimpleTransformerConfig {
    fn validate(&self) -> Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(anyhow::anyhow!(
                "hidden_size must be divisible by num_attention_heads"
            ));
        }
        Ok(())
    }
}
```

### Step 2: Implement Embeddings

```rust
use trustformers_core::{Tensor, Embedding, LayerNorm, Dropout, Layer};

pub struct SimpleEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl SimpleEmbeddings {
    pub fn new(config: &SimpleTransformerConfig) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(
                config.vocab_size,
                config.hidden_size,
            )?,
            position_embeddings: Embedding::new(
                config.max_position_embeddings,
                config.hidden_size,
            )?,
            layer_norm: LayerNorm::new(
                config.hidden_size,
                config.layer_norm_eps,
            ),
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }
    
    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq_length = input_ids.shape()[1];
        
        // Get word embeddings
        let inputs_embeds = self.word_embeddings.forward(input_ids)?;
        
        // Create position IDs if not provided
        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => Tensor::arange(0, seq_length as i64, 1)?
                .unsqueeze(0)?
                .expand_as(&input_ids)?,
        };
        
        // Get position embeddings
        let position_embeds = self.position_embeddings.forward(&position_ids)?;
        
        // Combine embeddings
        let embeddings = inputs_embeds.add(&position_embeds)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;
        
        Ok(embeddings)
    }
}
```

### Step 3: Implement Self-Attention

```rust
use trustformers_core::{Linear, Softmax};

pub struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl SelfAttention {
    pub fn new(config: &SimpleTransformerConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        
        Ok(Self {
            query: Linear::new(config.hidden_size, config.hidden_size)?,
            key: Linear::new(config.hidden_size, config.hidden_size)?,
            value: Linear::new(config.hidden_size, config.hidden_size)?,
            dropout: Dropout::new(config.attention_probs_dropout_prob),
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        })
    }
    
    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let mut new_shape = x.shape().to_vec();
        new_shape.pop(); // Remove hidden_size
        new_shape.push(self.num_attention_heads);
        new_shape.push(self.attention_head_size);
        
        x.view(&new_shape)?
            .permute(&[0, 2, 1, 3])
    }
    
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // Compute Q, K, V
        let query_layer = self.transpose_for_scores(
            &self.query.forward(hidden_states)?
        )?;
        let key_layer = self.transpose_for_scores(
            &self.key.forward(hidden_states)?
        )?;
        let value_layer = self.transpose_for_scores(
            &self.value.forward(hidden_states)?
        )?;
        
        // Compute attention scores
        let attention_scores = query_layer.matmul(&key_layer.transpose(-1, -2)?)?;
        let attention_scores = attention_scores.div_scalar(
            (self.attention_head_size as f32).sqrt()
        )?;
        
        // Apply attention mask if provided
        let attention_scores = if let Some(mask) = attention_mask {
            attention_scores.add(&mask.mul_scalar(-10000.0)?)?
        } else {
            attention_scores
        };
        
        // Normalize attention scores
        let attention_probs = attention_scores.softmax(-1)?;
        let attention_probs = self.dropout.forward(&attention_probs)?;
        
        // Compute context layer
        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.permute(&[0, 2, 1, 3])?
            .contiguous()?;
        
        let mut new_shape = context_layer.shape().to_vec();
        new_shape.pop();
        new_shape.pop();
        new_shape.push(self.num_attention_heads * self.attention_head_size);
        
        let context_layer = context_layer.view(&new_shape)?;
        
        Ok((context_layer, attention_probs))
    }
}
```

### Step 4: Implement Transformer Layer

```rust
pub struct TransformerLayer {
    attention: SelfAttention,
    attention_output: Linear,
    attention_dropout: Dropout,
    attention_layer_norm: LayerNorm,
    intermediate: Linear,
    output: Linear,
    output_dropout: Dropout,
    output_layer_norm: LayerNorm,
}

impl TransformerLayer {
    pub fn new(config: &SimpleTransformerConfig) -> Result<Self> {
        Ok(Self {
            attention: SelfAttention::new(config)?,
            attention_output: Linear::new(
                config.hidden_size,
                config.hidden_size,
            )?,
            attention_dropout: Dropout::new(config.hidden_dropout_prob),
            attention_layer_norm: LayerNorm::new(
                config.hidden_size,
                config.layer_norm_eps,
            ),
            intermediate: Linear::new(
                config.hidden_size,
                config.intermediate_size,
            )?,
            output: Linear::new(
                config.intermediate_size,
                config.hidden_size,
            )?,
            output_dropout: Dropout::new(config.hidden_dropout_prob),
            output_layer_norm: LayerNorm::new(
                config.hidden_size,
                config.layer_norm_eps,
            ),
        })
    }
    
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // Self-attention
        let (attention_output, attention_probs) = self.attention.forward(
            hidden_states,
            attention_mask,
        )?;
        
        // Add & Norm
        let attention_output = self.attention_output.forward(&attention_output)?;
        let attention_output = self.attention_dropout.forward(&attention_output)?;
        let attention_output = attention_output.add(hidden_states)?;
        let attention_output = self.attention_layer_norm.forward(&attention_output)?;
        
        // Feed-forward network
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let intermediate_output = intermediate_output.gelu()?; // GELU activation
        
        // Output projection
        let layer_output = self.output.forward(&intermediate_output)?;
        let layer_output = self.output_dropout.forward(&layer_output)?;
        let layer_output = layer_output.add(&attention_output)?;
        let layer_output = self.output_layer_norm.forward(&layer_output)?;
        
        Ok((layer_output, attention_probs))
    }
}
```

### Step 5: Complete Model

```rust
use trustformers_core::{Model, ModelOutput};

pub struct SimpleTransformer {
    config: SimpleTransformerConfig,
    embeddings: SimpleEmbeddings,
    encoder_layers: Vec<TransformerLayer>,
}

impl SimpleTransformer {
    pub fn new(config: SimpleTransformerConfig) -> Result<Self> {
        config.validate()?;
        
        let embeddings = SimpleEmbeddings::new(&config)?;
        let mut encoder_layers = Vec::with_capacity(config.num_hidden_layers);
        
        for _ in 0..config.num_hidden_layers {
            encoder_layers.push(TransformerLayer::new(&config)?);
        }
        
        Ok(Self {
            config,
            embeddings,
            encoder_layers,
        })
    }
}

impl Model for SimpleTransformer {
    type Config = SimpleTransformerConfig;
    type Input = Tensor; // input_ids
    type Output = BaseModelOutput;
    
    fn new(config: Self::Config) -> Result<Self> {
        Self::new(config)
    }
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Get embeddings
        let hidden_states = self.embeddings.forward(&input, None)?;
        
        // Create attention mask
        let attention_mask = create_attention_mask(&input, self.config.pad_token_id)?;
        
        let mut all_hidden_states = Vec::new();
        let mut all_attentions = Vec::new();
        
        let mut current_hidden_states = hidden_states;
        
        // Pass through encoder layers
        for layer in &self.encoder_layers {
            all_hidden_states.push(current_hidden_states.clone());
            
            let (layer_output, attention_probs) = layer.forward(
                &current_hidden_states,
                Some(&attention_mask),
            )?;
            
            current_hidden_states = layer_output;
            all_attentions.push(attention_probs);
        }
        
        all_hidden_states.push(current_hidden_states.clone());
        
        Ok(BaseModelOutput {
            last_hidden_state: current_hidden_states,
            hidden_states: Some(all_hidden_states),
            attentions: Some(all_attentions),
        })
    }
    
    fn from_pretrained(model_id: &str) -> Result<Self> {
        // Load configuration
        let config = SimpleTransformerConfig::from_pretrained(model_id)?;
        
        // Create model
        let mut model = Self::new(config)?;
        
        // Load weights
        model.load_pretrained_weights(model_id)?;
        
        Ok(model)
    }
}

// Helper function to create attention mask
fn create_attention_mask(input_ids: &Tensor, pad_token_id: usize) -> Result<Tensor> {
    // Create mask where 1.0 for real tokens and 0.0 for padding
    let mask = input_ids.ne_scalar(pad_token_id as f32)?
        .to_dtype(DataType::Float32)?;
    
    // Expand to attention shape [batch, 1, 1, seq_len]
    let mask = mask.unsqueeze(1)?.unsqueeze(2)?;
    
    Ok(mask)
}
```

## Building Custom Layers

### Custom Activation Function

```rust
use trustformers_core::{Layer, Tensor};

pub struct Swish {
    beta: f32,
}

impl Swish {
    pub fn new(beta: f32) -> Self {
        Self { beta }
    }
}

impl Layer for Swish {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // swish(x) = x * sigmoid(beta * x)
        let beta_x = input.mul_scalar(self.beta)?;
        let sigmoid = beta_x.sigmoid()?;
        input.mul(&sigmoid)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No trainable parameters
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}
```

### Custom Attention Mechanism

```rust
pub struct LocalAttention {
    window_size: usize,
    num_heads: usize,
    head_dim: usize,
    qkv_proj: Linear,
    out_proj: Linear,
    dropout: Dropout,
}

impl LocalAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        window_size: usize,
        dropout: f32,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        
        Ok(Self {
            window_size,
            num_heads,
            head_dim,
            qkv_proj: Linear::new(hidden_size, 3 * hidden_size)?,
            out_proj: Linear::new(hidden_size, hidden_size)?,
            dropout: Dropout::new(dropout),
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        
        // Project to Q, K, V
        let qkv = self.qkv_proj.forward(x)?;
        let qkv = qkv.reshape(&[
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        ])?;
        let qkv = qkv.permute(&[2, 0, 3, 1, 4])?;
        
        let q = qkv.select(0, 0)?;
        let k = qkv.select(0, 1)?;
        let v = qkv.select(0, 2)?;
        
        // Create local attention mask
        let mask = create_local_attention_mask(seq_len, self.window_size)?;
        
        // Compute attention
        let scores = q.matmul(&k.transpose(-2, -1)?)?
            .div_scalar((self.head_dim as f32).sqrt())?;
        let scores = scores.masked_fill(&mask.eq_scalar(0.0)?, f32::NEG_INFINITY)?;
        
        let attn_weights = scores.softmax(-1)?;
        let attn_weights = self.dropout.forward(&attn_weights)?;
        
        let attn_output = attn_weights.matmul(&v)?;
        
        // Reshape and project output
        let attn_output = attn_output.transpose(1, 2)?
            .reshape(&[batch_size, seq_len, -1])?;
        
        self.out_proj.forward(&attn_output)
    }
}

fn create_local_attention_mask(seq_len: usize, window_size: usize) -> Result<Tensor> {
    let mut mask = vec![0.0; seq_len * seq_len];
    
    for i in 0..seq_len {
        let start = (i as i32 - window_size as i32 / 2).max(0) as usize;
        let end = ((i + window_size / 2 + 1).min(seq_len)) as usize;
        
        for j in start..end {
            mask[i * seq_len + j] = 1.0;
        }
    }
    
    Tensor::from_vec(mask, &[seq_len, seq_len])
}
```

## Creating a BERT-like Model

Let's implement a complete BERT model with masked language modeling:

```rust
use trustformers_core::{Tensor, Config, Model};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: usize,
}

pub struct BertModel {
    config: BertConfig,
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: BertPooler,
}

pub struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertEmbeddings {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_shape = input_ids.shape();
        let seq_length = input_shape[1];
        
        // Token embeddings
        let inputs_embeds = self.word_embeddings.forward(input_ids)?;
        
        // Position embeddings
        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => Tensor::arange(0, seq_length as i64, 1)?
                .unsqueeze(0)?
                .expand(&[input_shape[0], seq_length])?,
        };
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        
        // Token type embeddings
        let token_type_ids = match token_type_ids {
            Some(ids) => ids.clone(),
            None => Tensor::zeros(&input_shape)?,
        };
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;
        
        // Combine all embeddings
        let embeddings = inputs_embeds
            .add(&position_embeddings)?
            .add(&token_type_embeddings)?;
        
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;
        
        Ok(embeddings)
    }
}

pub struct BertForMaskedLM {
    bert: BertModel,
    cls: BertMLMHead,
}

pub struct BertMLMHead {
    predictions: BertLMPredictionHead,
}

pub struct BertLMPredictionHead {
    transform: BertPredictionHeadTransform,
    decoder: Linear,
}

impl BertForMaskedLM {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        labels: Option<&Tensor>,
    ) -> Result<MaskedLMOutput> {
        let outputs = self.bert.forward(
            input_ids,
            attention_mask,
            token_type_ids,
        )?;
        
        let prediction_scores = self.cls.forward(&outputs.last_hidden_state)?;
        
        let loss = if let Some(labels) = labels {
            let loss_fct = CrossEntropyLoss::new();
            let masked_lm_loss = loss_fct.forward(
                &prediction_scores.view(&[-1, self.bert.config.vocab_size])?,
                &labels.view(&[-1])?,
            )?;
            Some(masked_lm_loss)
        } else {
            None
        };
        
        Ok(MaskedLMOutput {
            loss,
            logits: prediction_scores,
            hidden_states: outputs.hidden_states,
            attentions: outputs.attentions,
        })
    }
}
```

## Implementing a GPT-like Model

Now let's implement an autoregressive GPT model:

```rust
pub struct GPTConfig {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_inner: Option<usize>,
    pub activation_function: String,
    pub resid_pdrop: f32,
    pub embd_pdrop: f32,
    pub attn_pdrop: f32,
    pub layer_norm_epsilon: f32,
    pub initializer_range: f32,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
}

pub struct GPTModel {
    config: GPTConfig,
    wte: Embedding,      // Word token embeddings
    wpe: Embedding,      // Position embeddings
    drop: Dropout,
    h: Vec<GPTBlock>,    // Transformer blocks
    ln_f: LayerNorm,     // Final layer norm
}

pub struct GPTBlock {
    ln_1: LayerNorm,
    attn: GPTAttention,
    ln_2: LayerNorm,
    mlp: GPTMLP,
}

pub struct GPTAttention {
    n_head: usize,
    n_embd: usize,
    c_attn: Linear,      // Q, K, V projection
    c_proj: Linear,      // Output projection
    attn_dropout: Dropout,
    resid_dropout: Dropout,
}

impl GPTAttention {
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        use_cache: bool,
        past_key_value: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let batch_size = hidden_states.shape()[0];
        let seq_length = hidden_states.shape()[1];
        
        // Compute Q, K, V
        let qkv = self.c_attn.forward(hidden_states)?;
        let qkv = qkv.reshape(&[batch_size, seq_length, 3, self.n_head, -1])?;
        let qkv = qkv.permute(&[2, 0, 3, 1, 4])?;
        
        let query = qkv.select(0, 0)?;
        let key = qkv.select(0, 1)?;
        let value = qkv.select(0, 2)?;
        
        // Handle past key values for generation
        let (key, value) = if let Some((past_key, past_value)) = past_key_value {
            let key = Tensor::cat(&[past_key, &key], 2)?;
            let value = Tensor::cat(&[past_value, &value], 2)?;
            (key, value)
        } else {
            (key, value)
        };
        
        // Compute attention
        let attn_weights = query.matmul(&key.transpose(-2, -1)?)?;
        let attn_weights = attn_weights.div_scalar(
            (self.n_embd as f32 / self.n_head as f32).sqrt()
        )?;
        
        // Apply causal mask
        let causal_mask = create_causal_mask(seq_length)?;
        let attn_weights = attn_weights.masked_fill(
            &causal_mask.eq_scalar(0.0)?,
            f32::NEG_INFINITY,
        )?;
        
        // Apply attention mask if provided
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.add(&mask)?
        } else {
            attn_weights
        };
        
        let attn_weights = attn_weights.softmax(-1)?;
        let attn_weights = self.attn_dropout.forward(&attn_weights)?;
        
        let attn_output = attn_weights.matmul(&value)?;
        let attn_output = attn_output.transpose(1, 2)?
            .reshape(&[batch_size, seq_length, self.n_embd])?;
        
        let attn_output = self.c_proj.forward(&attn_output)?;
        let attn_output = self.resid_dropout.forward(&attn_output)?;
        
        let present = if use_cache {
            Some((key.clone(), value.clone()))
        } else {
            None
        };
        
        Ok((attn_output, present))
    }
}

pub struct GPTForCausalLM {
    transformer: GPTModel,
    lm_head: Linear,
}

impl GPTForCausalLM {
    pub fn generate(
        &self,
        input_ids: &Tensor,
        max_length: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> Result<Tensor> {
        let mut current_ids = input_ids.clone();
        let mut past_key_values = None;
        
        while current_ids.shape()[1] < max_length {
            // Forward pass
            let outputs = self.forward_with_past(
                &current_ids,
                past_key_values.as_ref(),
            )?;
            
            // Get next token logits
            let next_token_logits = outputs.logits
                .select(1, -1)?  // Last position
                .div_scalar(temperature)?;
            
            // Apply top-k/top-p filtering
            let filtered_logits = if let Some(k) = top_k {
                top_k_filtering(&next_token_logits, k)?
            } else {
                next_token_logits
            };
            
            let filtered_logits = if let Some(p) = top_p {
                top_p_filtering(&filtered_logits, p)?
            } else {
                filtered_logits
            };
            
            // Sample next token
            let probs = filtered_logits.softmax(-1)?;
            let next_token = probs.multinomial(1)?;
            
            // Append to sequence
            current_ids = Tensor::cat(&[&current_ids, &next_token], 1)?;
            
            // Update past key values
            past_key_values = outputs.past_key_values;
            
            // Check for EOS token
            if next_token.item::<i64>() == self.transformer.config.eos_token_id as i64 {
                break;
            }
        }
        
        Ok(current_ids)
    }
}

fn create_causal_mask(seq_length: usize) -> Result<Tensor> {
    let mut mask = vec![0.0; seq_length * seq_length];
    
    for i in 0..seq_length {
        for j in 0..=i {
            mask[i * seq_length + j] = 1.0;
        }
    }
    
    Tensor::from_vec(mask, &[seq_length, seq_length])
}
```

## Advanced Techniques

### Efficient Attention Implementations

```rust
pub struct FlashAttentionImpl {
    num_heads: usize,
    head_dim: usize,
    dropout: f32,
}

impl FlashAttentionImpl {
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
    ) -> Result<Tensor> {
        // Use optimized FlashAttention kernel
        use trustformers_kernels::flash_attention_forward;
        
        flash_attention_forward(
            q,
            k,
            v,
            self.dropout,
            causal,
            self.num_heads,
            self.head_dim,
        )
    }
}
```

### Mixture of Experts (MoE)

```rust
pub struct MoELayer {
    num_experts: usize,
    expert_capacity: f32,
    experts: Vec<FeedForward>,
    gate: Linear,
}

impl MoELayer {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        
        // Compute routing scores
        let router_logits = self.gate.forward(x)?;
        let routing_weights = router_logits.softmax(-1)?;
        
        // Get top-k experts per token
        let (expert_weights, expert_indices) = routing_weights.topk(2, -1)?;
        
        // Normalize expert weights
        let expert_weights = expert_weights.div(&expert_weights.sum_keepdim(-1)?)?;
        
        // Route tokens to experts
        let mut output = Tensor::zeros_like(x)?;
        
        for expert_idx in 0..self.num_experts {
            // Get tokens for this expert
            let expert_mask = expert_indices.eq_scalar(expert_idx as i64)?;
            let token_indices = expert_mask.nonzero()?;
            
            if token_indices.shape()[0] == 0 {
                continue;
            }
            
            // Apply expert
            let expert_input = x.index_select(0, &token_indices.select(1, 0)?)?
                .index_select(1, &token_indices.select(1, 1)?)?;
            
            let expert_output = self.experts[expert_idx].forward(&expert_input)?;
            
            // Weighted combine
            let weights = expert_weights.gather(2, &expert_indices)?
                .select(2, expert_mask.sum(-1)?.argmax(-1)?)?;
            
            output = output.scatter_add(
                0,
                &token_indices.select(1, 0)?,
                &(expert_output.mul(&weights.unsqueeze(-1)?)?),
            )?;
        }
        
        Ok(output)
    }
}
```

### Rotary Position Embeddings (RoPE)

```rust
pub struct RotaryEmbedding {
    dim: usize,
    max_seq_len: usize,
    base: f32,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32) -> Self {
        Self { dim, max_seq_len, base }
    }
    
    pub fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = q.shape()[2];
        let device = q.device();
        
        // Compute sinusoidal position encodings
        let inv_freq = (0..self.dim)
            .step_by(2)
            .map(|i| 1.0 / self.base.powf(i as f32 / self.dim as f32))
            .collect::<Vec<_>>();
        
        let inv_freq = Tensor::from_vec(inv_freq, &[self.dim / 2])?
            .to_device(device)?;
        
        let positions = Tensor::arange(0, seq_len as i64, 1)?
            .to_device(device)?;
        
        let freqs = positions.unsqueeze(-1)?
            .mul(&inv_freq.unsqueeze(0)?)?;
        
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        
        // Apply rotary embeddings
        let q_rot = apply_rotary_pos_emb(&q, &cos, &sin)?;
        let k_rot = apply_rotary_pos_emb(&k, &cos, &sin)?;
        
        Ok((q_rot, k_rot))
    }
}

fn apply_rotary_pos_emb(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    let x_shape = x.shape();
    let ndim = x_shape.len();
    
    // Reshape to separate even and odd dimensions
    let x = x.view(&[
        x_shape[0],
        x_shape[1],
        x_shape[2],
        x_shape[3] / 2,
        2,
    ])?;
    
    let x_even = x.select(-1, 0)?;
    let x_odd = x.select(-1, 1)?;
    
    // Apply rotation
    let cos = cos.unsqueeze(0)?.unsqueeze(1)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(1)?;
    
    let x_rot_even = x_even.mul(&cos)?.sub(&x_odd.mul(&sin)?)?;
    let x_rot_odd = x_even.mul(&sin)?.add(&x_odd.mul(&cos)?)?;
    
    // Combine back
    let x_rot = Tensor::stack(&[x_rot_even, x_rot_odd], -1)?
        .flatten(-2, -1)?;
    
    Ok(x_rot)
}
```

## Testing and Validation

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_attention_output_shape() {
        let config = SimpleTransformerConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            ..Default::default()
        };
        
        let attention = SelfAttention::new(&config).unwrap();
        let input = Tensor::randn(&[2, 10, 768]).unwrap();
        
        let (output, _) = attention.forward(&input, None).unwrap();
        
        assert_eq!(output.shape(), &[2, 10, 768]);
    }
    
    #[test]
    fn test_model_forward_pass() {
        let config = SimpleTransformerConfig::default();
        let model = SimpleTransformer::new(config).unwrap();
        
        let input_ids = Tensor::randint(0, 30522, &[2, 128]).unwrap();
        let output = model.forward(input_ids).unwrap();
        
        assert_eq!(output.last_hidden_state.shape(), &[2, 128, 768]);
        assert_eq!(output.hidden_states.as_ref().unwrap().len(), 13); // 12 layers + embeddings
    }
    
    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(4).unwrap();
        let expected = vec![
            1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
        ];
        
        assert_eq!(mask.to_vec::<f32>().unwrap(), expected);
    }
}
```

### Integration Tests

```rust
#[test]
fn test_model_save_load() {
    let config = SimpleTransformerConfig::default();
    let model = SimpleTransformer::new(config.clone()).unwrap();
    
    // Save model
    model.save_pretrained("/tmp/test_model").unwrap();
    
    // Load model
    let loaded_model = SimpleTransformer::from_pretrained("/tmp/test_model").unwrap();
    
    // Compare outputs
    let input = Tensor::randint(0, 30522, &[1, 64]).unwrap();
    let output1 = model.forward(input.clone()).unwrap();
    let output2 = loaded_model.forward(input).unwrap();
    
    assert_relative_eq!(
        output1.last_hidden_state.to_vec::<f32>().unwrap().as_slice(),
        output2.last_hidden_state.to_vec::<f32>().unwrap().as_slice(),
        epsilon = 1e-5
    );
}
```

### Benchmarking

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_attention(c: &mut Criterion) {
    let config = SimpleTransformerConfig::default();
    let attention = SelfAttention::new(&config).unwrap();
    let input = Tensor::randn(&[8, 512, 768]).unwrap();
    
    c.bench_function("self_attention_forward", |b| {
        b.iter(|| {
            let (output, _) = attention.forward(
                black_box(&input),
                None,
            ).unwrap();
            black_box(output);
        });
    });
}

criterion_group!(benches, benchmark_attention);
criterion_main!(benches);
```

## Integration with TrustformeRS Ecosystem

### Making Your Model Hub-Compatible

```rust
use trustformers::hub::{ModelCard, push_to_hub};

impl SimpleTransformer {
    pub fn push_to_hub(&self, repo_id: &str, token: Option<&str>) -> Result<()> {
        // Create model card
        let model_card = ModelCard {
            model_type: "transformer",
            task: "feature-extraction",
            language: "en",
            license: "apache-2.0",
            datasets: vec![],
            metrics: HashMap::new(),
            description: "Simple transformer model for feature extraction".to_string(),
        };
        
        // Save model files
        self.save_pretrained("./tmp_model")?;
        
        // Push to hub
        push_to_hub(
            repo_id,
            "./tmp_model",
            model_card,
            token,
        )?;
        
        Ok(())
    }
}
```

### Pipeline Integration

```rust
use trustformers::pipeline::{Pipeline, PipelineConfig};

pub struct SimpleTransformerPipeline {
    model: SimpleTransformer,
    tokenizer: Tokenizer,
}

impl Pipeline for SimpleTransformerPipeline {
    type Input = String;
    type Output = Vec<f32>;
    
    fn preprocess(&self, input: Self::Input) -> Result<Tensor> {
        let encoding = self.tokenizer.encode(input, None)?;
        Ok(encoding.input_ids)
    }
    
    fn forward(&self, input: Tensor) -> Result<BaseModelOutput> {
        self.model.forward(input)
    }
    
    fn postprocess(&self, output: BaseModelOutput) -> Result<Self::Output> {
        // Mean pooling over sequence dimension
        let pooled = output.last_hidden_state
            .mean_dim(&[1], true)?
            .squeeze(1)?;
        
        Ok(pooled.to_vec()?)
    }
}
```

## Best Practices

1. **Configuration Validation**: Always validate configuration in the constructor
2. **Error Handling**: Use `Result<T>` for all operations that can fail
3. **Memory Management**: Avoid unnecessary clones, use references when possible
4. **Documentation**: Document all public APIs with examples
5. **Testing**: Write comprehensive tests for all components
6. **Performance**: Profile your model and optimize bottlenecks

## Next Steps

- Explore [Advanced Architectures](./advanced_architectures.md)
- Learn about [Model Optimization](./model_optimization.md)
- Read the [Deployment Guide](./deployment.md)
- Check out [Example Models](../examples/models/)

Happy modeling! 🚀