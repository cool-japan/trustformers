use dashmap::DashMap;
#![allow(unused_variables)]
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use trustformers::{
    AutoModel, AutoTokenizer,
    Model, Tokenizer,
};
use uuid::Uuid;
use std::mem;
use std::any::Any;

pub struct ModelManager {
    models: DashMap<String, LoadedModel>,
}

struct LoadedModel {
    id: String,
    name: String,
    model_type: String,
    model: Arc<dyn Model + Send + Sync>,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    loaded_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub model_type: String,
    pub loaded_at: String,
    pub memory_usage: Option<u64>,
}

impl ModelManager {
    pub async fn new() -> anyhow::Result<Self> {
        Ok(Self {
            models: DashMap::new(),
        })
    }

    /// Estimate memory usage for a loaded model
    /// This provides a reasonable approximation based on model type and parameters
    fn estimate_model_memory_usage(&self, model_type: &str) -> u64 {
        match model_type {
            "bert" => {
                // BERT base has ~110M parameters
                // Assuming fp32 (4 bytes per parameter) + overhead
                let parameter_memory = 110_000_000 * 4; // ~440MB for parameters
                let activation_memory = 50_000_000; // ~50MB for activations/intermediate states
                let tokenizer_memory = 10_000_000; // ~10MB for tokenizer vocab
                parameter_memory + activation_memory + tokenizer_memory
            }
            "gpt2" => {
                // GPT-2 base has ~117M parameters
                // Assuming fp32 (4 bytes per parameter) + overhead
                let parameter_memory = 117_000_000 * 4; // ~468MB for parameters
                let kv_cache_memory = 30_000_000; // ~30MB for KV cache
                let tokenizer_memory = 5_000_000; // ~5MB for BPE tokenizer
                parameter_memory + kv_cache_memory + tokenizer_memory
            }
            _ => {
                // Default estimate for unknown models
                100_000_000 // ~100MB default
            }
        }
    }

    pub async fn load_model(&self, model_name: &str, model_type: &str) -> anyhow::Result<String> {
        let id = Uuid::new_v4().to_string();

        // Load model and tokenizer
        let model = match model_type {
            "bert" => {
                let model = trustformers::models::bert::BertModel::new(
                    trustformers::models::bert::BertConfig::bert_base_uncased()
                );
                Arc::new(model) as Arc<dyn Model + Send + Sync>
            }
            "gpt2" => {
                let model = trustformers::models::gpt2::Gpt2Model::new(
                    trustformers::models::gpt2::Gpt2Config::gpt2()
                );
                Arc::new(model) as Arc<dyn Model + Send + Sync>
            }
            _ => return Err(anyhow::anyhow!("Unsupported model type: {}", model_type)),
        };

        // Create appropriate tokenizer
        let tokenizer = match model_type {
            "bert" => {
                Arc::new(trustformers::tokenizers::WordPieceTokenizer::default())
                    as Arc<dyn Tokenizer + Send + Sync>
            }
            "gpt2" => {
                Arc::new(trustformers::tokenizers::BPETokenizer::default())
                    as Arc<dyn Tokenizer + Send + Sync>
            }
            _ => return Err(anyhow::anyhow!("Unsupported tokenizer type")),
        };

        let loaded_model = LoadedModel {
            id: id.clone(),
            name: model_name.to_string(),
            model_type: model_type.to_string(),
            model,
            tokenizer,
            loaded_at: chrono::Utc::now(),
        };

        self.models.insert(id.clone(), loaded_model);

        Ok(id)
    }

    pub async fn unload_model(&self, model_id: &str) -> anyhow::Result<()> {
        self.models
            .remove(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_id))?;
        Ok(())
    }

    pub async fn get_model(&self, model_id: &str) -> Option<(Arc<dyn Model + Send + Sync>, Arc<dyn Tokenizer + Send + Sync>)> {
        self.models
            .get(model_id)
            .map(|entry| (entry.model.clone(), entry.tokenizer.clone()))
    }

    pub async fn list_models(&self) -> Vec<ModelInfo> {
        self.models
            .iter()
            .map(|entry| {
                let memory_usage = self.estimate_model_memory_usage(&entry.model_type);
                ModelInfo {
                    id: entry.id.clone(),
                    name: entry.name.clone(),
                    model_type: entry.model_type.clone(),
                    loaded_at: entry.loaded_at.to_rfc3339(),
                    memory_usage: Some(memory_usage),
                }
            })
            .collect()
    }

    pub async fn get_model_info(&self, model_id: &str) -> Option<ModelInfo> {
        self.models.get(model_id).map(|entry| {
            let memory_usage = self.estimate_model_memory_usage(&entry.model_type);
            ModelInfo {
                id: entry.id.clone(),
                name: entry.name.clone(),
                model_type: entry.model_type.clone(),
                loaded_at: entry.loaded_at.to_rfc3339(),
                memory_usage: Some(memory_usage),
            }
        })
    }
}