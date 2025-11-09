use tch::{nn, Tensor, Kind, Device};
use crate::models::{TransformerConfig, TransformerDecoder, LayerNormTrait, LayerNorm, RMSNorm};
use crate::attention::SelfAttention;
use crate::layers::{FeedForward, ActivationType};
use crate::utils::{PositionalEncoding, PositionalEncodingType, create_causal_mask};
use std::collections::HashMap;

#[derive(Clone)]
pub struct GPTConfig {
    pub vocab_size: i64,
    pub d_model: i64,
    pub num_heads: i64,
    pub num_layers: i64,
    pub d_ff: i64,
    pub max_seq_len: i64,
    pub dropout: f64,
    pub layer_norm_eps: f64,
    pub use_rms_norm: bool,
    pub activation_type: ActivationType,
    pub positional_encoding: PositionalEncodingType,
    pub use_bias: bool,
    pub tie_weights: bool,
    pub pad_token_id: i64,
    pub eos_token_id: i64,
    pub bos_token_id: i64,
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            d_model: 768,
            num_heads: 12,
            num_layers: 12,
            d_ff: 3072,
            max_seq_len: 1024,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_rms_norm: false,
            activation_type: ActivationType::GELU,
            positional_encoding: PositionalEncodingType::Learned,
            use_bias: true,
            tie_weights: true,
            pad_token_id: 50256,
            eos_token_id: 50256,
            bos_token_id: 50256,
        }
    }
}

pub struct GPTModel {
    config: GPTConfig,
    wte: nn::Embedding, // token embeddings
    wpe: Option<nn::Embedding>, // position embeddings
    positional_encoding: Option<PositionalEncoding>,
    layers: Vec<GPTBlock>,
    ln_f: Box<dyn LayerNormTrait>, // final layer norm
    dropout: nn::Dropout,
}

pub struct GPTBlock {
    ln_1: Box<dyn LayerNormTrait>,
    attn: SelfAttention,
    ln_2: Box<dyn LayerNormTrait>,
    mlp: FeedForward,
    dropout: nn::Dropout,
}

impl GPTModel {
    pub fn new(vs: &nn::Path, config: GPTConfig) -> Self {
        let wte = nn::embedding(vs / "wte", config.vocab_size, config.d_model, Default::default());

        let (wpe, positional_encoding) = match config.positional_encoding {
            PositionalEncodingType::Learned => {
                let wpe = nn::embedding(vs / "wpe", config.max_seq_len, config.d_model, Default::default());
                (Some(wpe), None)
            },
            _ => {
                let pos_enc = PositionalEncoding::new(
                    &(vs / "pos_encoding"),
                    config.positional_encoding.clone(),
                    config.d_model,
                    config.max_seq_len,
                    0.0, // no dropout in positional encoding for GPT
                );
                (None, Some(pos_enc))
            },
        };

        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let layer_vs = vs / format!("h.{}", i);
            layers.push(GPTBlock::new(&layer_vs, &config));
        }

        let ln_f: Box<dyn LayerNormTrait> = if config.use_rms_norm {
            Box::new(RMSNorm::new(vs / "ln_f", vec![config.d_model], config.layer_norm_eps))
        } else {
            Box::new(LayerNorm::new(vs / "ln_f", vec![config.d_model], config.layer_norm_eps, true))
        };

        let dropout = nn::dropout(vs / "dropout", config.dropout);

        Self {
            config,
            wte,
            wpe,
            positional_encoding,
            layers,
            ln_f,
            dropout,
        }
    }

    pub fn forward(&self,
                   input_ids: &Tensor,
                   attention_mask: Option<&Tensor>,
                   past_key_values: Option<&mut Vec<HashMap<String, Tensor>>>) -> GPTOutput {
        let (batch_size, seq_len) = (input_ids.size()[0], input_ids.size()[1]);

        // Token embeddings
        let mut hidden_states = input_ids.apply(&self.wte);

        // Position embeddings
        if let Some(wpe) = &self.wpe {
            let past_length = past_key_values.as_ref().map_or(0, |cache| {
                cache.first().and_then(|layer_cache| {
                    layer_cache.get("key").map(|k| k.size()[2])
                }).unwrap_or(0)
            });

            let position_ids = Tensor::arange_start(past_length, past_length + seq_len, (Kind::Int64, input_ids.device()));
            let position_embeds = position_ids.apply(wpe);
            hidden_states = hidden_states + position_embeds.unsqueeze(0).expand([batch_size, seq_len, self.config.d_model]);
        } else if let Some(pos_enc) = &self.positional_encoding {
            hidden_states = pos_enc.forward(&hidden_states);
        }

        hidden_states = hidden_states.apply(&self.dropout);

        // Create causal mask if needed
        let causal_mask = if attention_mask.is_none() && past_key_values.is_none() {
            Some(create_causal_mask(seq_len, input_ids.device()))
        } else {
            None
        };

        let combined_mask = match (attention_mask, causal_mask.as_ref()) {
            (Some(attn_mask), Some(causal)) => {
                let expanded_attn_mask = attn_mask.unsqueeze(1).unsqueeze(1);
                let combined = expanded_attn_mask & causal.unsqueeze(0);
                Some(combined)
            },
            (Some(attn_mask), None) => Some(attn_mask.unsqueeze(1).unsqueeze(1)),
            (None, Some(causal)) => Some(causal.unsqueeze(0)),
            (None, None) => None,
        };

        // Initialize cache if needed
        if past_key_values.is_none() && self.config.num_layers > 0 {
            // Static allocation for inference
        }

        // Pass through transformer blocks
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = past_key_values.as_mut().map(|cache| &mut cache[i]);
            hidden_states = layer.forward(&hidden_states, combined_mask.as_ref(), layer_cache);
        }

        // Final layer norm
        hidden_states = self.ln_f.forward(&hidden_states);

        GPTOutput {
            last_hidden_state: hidden_states,
        }
    }

    pub fn generate(&self,
                   input_ids: &Tensor,
                   max_length: i64,
                   temperature: f64,
                   top_k: Option<i64>,
                   top_p: Option<f64>,
                   do_sample: bool,
                   pad_token_id: Option<i64>,
                   eos_token_id: Option<i64>) -> Tensor {
        let mut generated = input_ids.shallow_clone();
        let mut past_key_values: Vec<HashMap<String, Tensor>> = (0..self.config.num_layers)
            .map(|_| HashMap::new())
            .collect();

        let pad_id = pad_token_id.unwrap_or(self.config.pad_token_id);
        let eos_id = eos_token_id.unwrap_or(self.config.eos_token_id);

        for _ in input_ids.size()[1]..max_length {
            let current_input = if past_key_values.iter().any(|cache| !cache.is_empty()) {
                // Use only the last token for subsequent generations
                generated.narrow(1, -1, 1)
            } else {
                // Use full sequence for first generation
                generated.shallow_clone()
            };

            let output = self.forward(&current_input, None, Some(&mut past_key_values));
            let next_token_logits = output.last_hidden_state.narrow(1, -1, 1).squeeze_dim(1);

            let next_token = if do_sample {
                self.sample_next_token(&next_token_logits, temperature, top_k, top_p)
            } else {
                next_token_logits.argmax(-1, false)
            };

            generated = Tensor::cat(&[generated, next_token.unsqueeze(1)], 1);

            // Check for EOS token
            if next_token.int64_value(&[]) == eos_id {
                break;
            }
        }

        generated
    }

    fn sample_next_token(&self, logits: &Tensor, temperature: f64, top_k: Option<i64>, top_p: Option<f64>) -> Tensor {
        let mut logits = logits.shallow_clone();

        if temperature != 0.0 {
            logits = logits / temperature;
        }

        // Apply top-k filtering
        if let Some(k) = top_k {
            let (top_k_values, _) = logits.topk(k, -1, true, true);
            let threshold = top_k_values.narrow(-1, -1, 1);
            logits = logits.where_self(&logits.ge_tensor(&threshold), &Tensor::full_like(&logits, f64::NEG_INFINITY));
        }

        // Apply top-p filtering
        if let Some(p) = top_p {
            let probs = logits.softmax(-1, Kind::Float);
            let (sorted_probs, sorted_indices) = probs.sort(-1, true);
            let cumulative_probs = sorted_probs.cumsum(-1, Kind::Float);

            let mut sorted_indices_to_remove = cumulative_probs.gt(p);
            sorted_indices_to_remove = sorted_indices_to_remove.roll(&[1], &[-1]);
            sorted_indices_to_remove = sorted_indices_to_remove.narrow(-1, 0, sorted_indices_to_remove.size()[-1]);
            let first_element = Tensor::zeros(&[sorted_indices_to_remove.size()[0], 1], (Kind::Bool, sorted_indices_to_remove.device()));
            sorted_indices_to_remove = Tensor::cat(&[first_element, sorted_indices_to_remove.narrow(-1, 0, sorted_indices_to_remove.size()[-1] - 1)], -1);

            let indices_to_remove = sorted_indices_to_remove.scatter(-1, &sorted_indices, &sorted_indices_to_remove);
            logits = logits.masked_fill(&indices_to_remove, f64::NEG_INFINITY);
        }

        if temperature == 0.0 {
            logits.argmax(-1, false)
        } else {
            let probs = logits.softmax(-1, Kind::Float);
            probs.multinomial(1, false).squeeze_dim(-1)
        }
    }
}

impl GPTBlock {
    fn new(vs: &nn::Path, config: &GPTConfig) -> Self {
        let ln_1: Box<dyn LayerNormTrait> = if config.use_rms_norm {
            Box::new(RMSNorm::new(vs / "ln_1", vec![config.d_model], config.layer_norm_eps))
        } else {
            Box::new(LayerNorm::new(vs / "ln_1", vec![config.d_model], config.layer_norm_eps, true))
        };

        let attn = SelfAttention::new(&(vs / "attn"), config.d_model, config.num_heads, config.dropout);

        let ln_2: Box<dyn LayerNormTrait> = if config.use_rms_norm {
            Box::new(RMSNorm::new(vs / "ln_2", vec![config.d_model], config.layer_norm_eps))
        } else {
            Box::new(LayerNorm::new(vs / "ln_2", vec![config.d_model], config.layer_norm_eps, true))
        };

        let mlp = FeedForward::new(&(vs / "mlp"), config.d_model, config.d_ff, config.dropout, config.activation_type.clone());
        let dropout = nn::dropout(vs / "dropout", config.dropout);

        Self {
            ln_1,
            attn,
            ln_2,
            mlp,
            dropout,
        }
    }

    fn forward(&self,
               input: &Tensor,
               attention_mask: Option<&Tensor>,
               past_key_values: Option<&mut HashMap<String, Tensor>>) -> Tensor {
        // Pre-norm architecture
        let norm1_output = self.ln_1.forward(input);
        let attn_output = self.attn.forward_with_cache(&norm1_output, past_key_values, attention_mask);
        let residual1 = input + attn_output.apply(&self.dropout);

        let norm2_output = self.ln_2.forward(&residual1);
        let mlp_output = self.mlp.forward(&norm2_output);
        residual1 + mlp_output.apply(&self.dropout)
    }
}

pub struct GPTOutput {
    pub last_hidden_state: Tensor,
}

pub struct GPTForCausalLM {
    transformer: GPTModel,
    lm_head: nn::Linear,
}

impl GPTForCausalLM {
    pub fn new(vs: &nn::Path, config: GPTConfig) -> Self {
        let transformer = GPTModel::new(&(vs / "transformer"), config.clone());

        let lm_head = if config.tie_weights {
            // Share weights with token embeddings
            nn::linear_bias(vs / "lm_head", config.d_model, config.vocab_size, false, Default::default())
        } else {
            nn::linear(vs / "lm_head", config.d_model, config.vocab_size, Default::default())
        };

        Self {
            transformer,
            lm_head,
        }
    }

    pub fn forward(&self,
                   input_ids: &Tensor,
                   attention_mask: Option<&Tensor>,
                   labels: Option<&Tensor>) -> CausalLMOutput {
        let transformer_output = self.transformer.forward(input_ids, attention_mask, None);
        let lm_logits = transformer_output.last_hidden_state.apply(&self.lm_head);

        let loss = if let Some(labels) = labels {
            // Shift so that tokens < n predict n
            let shift_logits = lm_logits.narrow(1, 0, lm_logits.size()[1] - 1);
            let shift_labels = labels.narrow(1, 1, labels.size()[1] - 1);

            let loss_fct = nn::CrossEntropyLoss::new();
            let loss = shift_logits.view([-1, shift_logits.size()[-1]]).cross_entropy_for_logits(&shift_labels.view([-1]));
            Some(loss)
        } else {
            None
        };

        CausalLMOutput {
            loss,
            logits: lm_logits,
            hidden_states: transformer_output.last_hidden_state,
        }
    }

    pub fn generate(&self,
                   input_ids: &Tensor,
                   max_length: i64,
                   temperature: f64,
                   top_k: Option<i64>,
                   top_p: Option<f64>,
                   do_sample: bool,
                   pad_token_id: Option<i64>,
                   eos_token_id: Option<i64>) -> Tensor {
        self.transformer.generate(input_ids, max_length, temperature, top_k, top_p, do_sample, pad_token_id, eos_token_id)
    }
}

pub struct CausalLMOutput {
    pub loss: Option<Tensor>,
    pub logits: Tensor,
    pub hidden_states: Tensor,
}

// GPT-2 specific implementation
pub fn gpt2_config(model_size: GPT2ModelSize) -> GPTConfig {
    match model_size {
        GPT2ModelSize::Small => GPTConfig {
            vocab_size: 50257,
            d_model: 768,
            num_heads: 12,
            num_layers: 12,
            d_ff: 3072,
            max_seq_len: 1024,
            ..Default::default()
        },
        GPT2ModelSize::Medium => GPTConfig {
            vocab_size: 50257,
            d_model: 1024,
            num_heads: 16,
            num_layers: 24,
            d_ff: 4096,
            max_seq_len: 1024,
            ..Default::default()
        },
        GPT2ModelSize::Large => GPTConfig {
            vocab_size: 50257,
            d_model: 1280,
            num_heads: 20,
            num_layers: 36,
            d_ff: 5120,
            max_seq_len: 1024,
            ..Default::default()
        },
        GPT2ModelSize::XL => GPTConfig {
            vocab_size: 50257,
            d_model: 1600,
            num_heads: 25,
            num_layers: 48,
            d_ff: 6400,
            max_seq_len: 1024,
            ..Default::default()
        },
    }
}

pub enum GPT2ModelSize {
    Small,  // 117M parameters
    Medium, // 345M parameters
    Large,  // 762M parameters
    XL,     // 1.5B parameters
}

pub struct GPTTrainer {
    model: GPTForCausalLM,
    optimizer: nn::Optimizer,
    scheduler: Option<Box<dyn LRScheduler>>,
}

pub trait LRScheduler {
    fn step(&mut self, step: i64);
    fn get_lr(&self) -> f64;
}

impl GPTTrainer {
    pub fn new(vs: &nn::Path, config: GPTConfig, learning_rate: f64) -> Self {
        let model = GPTForCausalLM::new(vs, config);
        let optimizer = nn::Adam::default().build(vs, learning_rate).unwrap();

        Self {
            model,
            optimizer,
            scheduler: None,
        }
    }

    pub fn train_step(&mut self, input_ids: &Tensor, labels: &Tensor) -> f64 {
        let output = self.model.forward(input_ids, None, Some(labels));
        let loss = output.loss.expect("Loss should be computed during training");

        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        if let Some(scheduler) = &mut self.scheduler {
            scheduler.step(0); // You'd pass the actual step number
        }

        loss.double_value(&[])
    }
}