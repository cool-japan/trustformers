//! Fine-tuning helpers for parameter-efficient adaptation of pretrained models.
//!
//! This module provides two families of fine-tuning techniques:
//!
//! - **LoRA** ([`lora`]): Low-Rank Adaptation — injects trainable rank-decomposition
//!   matrices alongside frozen pretrained weights. Very parameter-efficient and
//!   widely used for large language models.
//!
//! - **Adapter** ([`adapter`]): Bottleneck adapters — inserts small trainable modules
//!   (down-project → activation → up-project + residual) after each transformer
//!   sub-layer. Houlsby et al. (2019) style.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use trustformers::finetuning::{LoraConfig, LoraLinear, AdapterConfig, BottleneckAdapter};
//!
//! // LoRA: 8-rank adapter for a 768 → 768 attention projection
//! let lora_cfg = LoraConfig::builder().rank(8).alpha(16.0).build().unwrap();
//! let lora_layer = LoraLinear::new(768, 768, 8, &lora_cfg).unwrap();
//!
//! // Bottleneck adapter for a BERT-base hidden layer
//! let adapter_cfg = AdapterConfig { hidden_size: 768, bottleneck_size: 64, ..Default::default() };
//! let adapter = BottleneckAdapter::new(adapter_cfg).unwrap();
//! ```

pub mod adapter;
pub mod lora;

pub use adapter::{AdapterActivation, AdapterConfig, BottleneckAdapter};
pub use lora::{LoraBias, LoraConfig, LoraConfigBuilder, LoraLinear};
