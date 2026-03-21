use std::sync::Arc;
#![allow(unused_variables)]
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::RwLock;
use tracing::{info, warn};

use trustformers::{AutoModel, AutoTokenizer, AutoConfig};

use crate::error::{ServiceError, ServiceResult};

pub struct LoadedModel {
    pub model: Arc<AutoModel>,
    pub tokenizer: Arc<AutoTokenizer>,
    pub config: AutoConfig,
    pub device: String,
    pub loaded_at: Instant,
    pub request_count: Arc<RwLock<u32>>,
    pub memory_used: u64,
}

pub struct ModelManager {
    models: DashMap<String, Arc<LoadedModel>>,
    max_models: usize,
    default_device: String,
}

impl ModelManager {
    pub fn new(max_models: usize, default_device: String) -> Self {
        Self {
            models: DashMap::new(),
            max_models,
            default_device,
        }
    }

    pub async fn load_model(
        &self,
        model_id: &str,
        model_path: Option<&str>,
        device: Option<&str>,
        use_fp16: bool,
        compile: bool,
    ) -> ServiceResult<Duration> {
        if self.models.contains_key(model_id) {
            return Err(ServiceError::InvalidInput(
                format!("Model {} is already loaded", model_id)
            ));
        }

        if self.models.len() >= self.max_models {
            return Err(ServiceError::ResourceExhausted(
                format!("Maximum number of models ({}) already loaded", self.max_models)
            ));
        }

        let start = Instant::now();
        let device = device.unwrap_or(&self.default_device);

        info!("Loading model {} on device {}", model_id, device);

        // Load model and tokenizer
        let model_path = model_path.unwrap_or(model_id);

        // Load config, model and tokenizer
        let config = trustformers::AutoConfig::from_pretrained(model_path)
            .map_err(|e| ServiceError::InferenceError(e.into()))?;

        let model = AutoModel::from_pretrained(model_path)
            .map_err(|e| ServiceError::InferenceError(e.into()))?;

        let tokenizer = AutoTokenizer::from_pretrained(model_path)
            .map_err(|e| ServiceError::InferenceError(e.into()))?;

        let loaded_model = Arc::new(LoadedModel {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            config,
            device: device.to_string(),
            loaded_at: Instant::now(),
            request_count: Arc::new(RwLock::new(0)),
            memory_used: estimate_model_memory(use_fp16),
        });

        self.models.insert(model_id.to_string(), loaded_model);

        let load_time = start.elapsed();
        info!("Model {} loaded in {:?}", model_id, load_time);

        Ok(load_time)
    }

    pub async fn unload_model(&self, model_id: &str) -> ServiceResult<()> {
        self.models
            .remove(model_id)
            .ok_or_else(|| ServiceError::ModelNotFound(model_id.to_string()))?;

        info!("Model {} unloaded", model_id);
        Ok(())
    }

    pub fn get_model(&self, model_id: &str) -> ServiceResult<Arc<LoadedModel>> {
        self.models
            .get(model_id)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| ServiceError::ModelNotLoaded(model_id.to_string()))
    }

    pub fn list_models(&self) -> Vec<(String, ModelInfo)> {
        self.models
            .iter()
            .map(|entry| {
                let model_id = entry.key().clone();
                let model = entry.value();
                let info = ModelInfo {
                    model_id: model_id.clone(),
                    device: model.device.clone(),
                    loaded_at: model.loaded_at,
                    request_count: 0, // Would need async to read
                    memory_used: model.memory_used,
                };
                (model_id, info)
            })
            .collect()
    }

    pub async fn increment_request_count(&self, model_id: &str) -> ServiceResult<()> {
        let model = self.get_model(model_id)?;
        let mut count = model.request_count.write().await;
        *count += 1;
        Ok(())
    }
}

pub struct ModelInfo {
    pub model_id: String,
    pub device: String,
    pub loaded_at: Instant,
    pub request_count: u32,
    pub memory_used: u64,
}

fn estimate_model_memory(use_fp16: bool) -> u64 {
    // Rough estimate for BERT-base
    if use_fp16 {
        110_000_000 * 2 // 110M params * 2 bytes
    } else {
        110_000_000 * 4 // 110M params * 4 bytes
    }
}