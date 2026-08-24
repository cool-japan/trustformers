//! Core data pipeline orchestration: the `DataPipeline` engine, its configuration and distributed-processing settings, and individual data samples.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use trustformers_core::tensor::Tensor;

use super::active_learning::{ActiveLearningConfig, ActiveLearningManager};
use super::augmentation::{DynamicAugmentationConfig, DynamicAugmentationManager};
use super::curriculum::{CurriculumLearningConfig, CurriculumLearningManager};
use super::multimodal::{MultiModalConfig, MultiModalHandler};
use super::streaming::{StreamingDataset, StreamingDatasetConfig};
use super::validation::{DataValidationConfig, DataValidator, ErrorHandling, ValidationResult};

/// Main data pipeline orchestrator
pub struct DataPipeline {
    /// Pipeline configuration
    pub(super) config: DataPipelineConfig,
    /// Active streaming datasets
    pub(super) streaming_datasets: Arc<Mutex<HashMap<String, StreamingDataset>>>,
    /// Dynamic augmentation manager
    pub(super) augmentation_manager: Arc<Mutex<DynamicAugmentationManager>>,
    /// Curriculum learning manager
    pub(super) curriculum_manager: Arc<Mutex<CurriculumLearningManager>>,
    /// Active learning manager
    pub(super) active_learning_manager: Arc<Mutex<ActiveLearningManager>>,
    /// Multi-modal data handler
    pub(super) multimodal_handler: Arc<Mutex<MultiModalHandler>>,
    /// Data validator
    pub(super) validator: Arc<Mutex<DataValidator>>,
}
impl DataPipeline {
    pub fn new(config: DataPipelineConfig) -> Self {
        Self {
            config,
            streaming_datasets: Arc::new(Mutex::new(HashMap::new())),
            augmentation_manager: Arc::new(Mutex::new(DynamicAugmentationManager::new())),
            curriculum_manager: Arc::new(Mutex::new(CurriculumLearningManager::new())),
            active_learning_manager: Arc::new(Mutex::new(ActiveLearningManager::new())),
            multimodal_handler: Arc::new(Mutex::new(MultiModalHandler::new())),
            validator: Arc::new(Mutex::new(DataValidator::new())),
        }
    }
    /// Register a streaming dataset under `dataset_id`.
    ///
    /// The dataset starts inactive; call [`DataPipeline::start_streaming`] to make it eligible
    /// for [`DataPipeline::get_batch`].
    pub fn register_dataset(&self, dataset_id: &str, config: StreamingDatasetConfig) -> Result<()> {
        let mut datasets = self
            .streaming_datasets
            .lock()
            .map_err(|_| anyhow::anyhow!("streaming dataset registry lock poisoned"))?;
        if datasets.contains_key(dataset_id) {
            return Err(anyhow::anyhow!(
                "dataset '{dataset_id}' is already registered"
            ));
        }
        datasets.insert(dataset_id.to_string(), StreamingDataset::new(config));
        Ok(())
    }
    /// Ids of every registered dataset.
    pub fn dataset_ids(&self) -> Result<Vec<String>> {
        let datasets = self
            .streaming_datasets
            .lock()
            .map_err(|_| anyhow::anyhow!("streaming dataset registry lock poisoned"))?;
        let mut ids: Vec<String> = datasets.keys().cloned().collect();
        ids.sort();
        Ok(ids)
    }
    /// Push samples into a registered dataset's buffer.
    ///
    /// This is how a producer (file reader, network stream, generator, …) hands data to the
    /// pipeline. Samples beyond `buffer_size` are rejected rather than silently dropped.
    pub fn push_samples(&self, dataset_id: &str, samples: Vec<DataSample>) -> Result<usize> {
        let mut datasets = self
            .streaming_datasets
            .lock()
            .map_err(|_| anyhow::anyhow!("streaming dataset registry lock poisoned"))?;
        let dataset = datasets.get_mut(dataset_id).ok_or_else(|| {
            anyhow::anyhow!("unknown dataset '{dataset_id}'; register it with register_dataset")
        })?;
        dataset.push_samples(samples)
    }
    /// Mark a registered dataset as streaming.
    ///
    /// # Errors
    ///
    /// Fails when `dataset_id` names a dataset that was never registered — previously this
    /// returned `Ok(())` for any string at all.
    pub async fn start_streaming(&self, dataset_id: &str) -> Result<()> {
        let mut datasets = self
            .streaming_datasets
            .lock()
            .map_err(|_| anyhow::anyhow!("streaming dataset registry lock poisoned"))?;
        let dataset = datasets.get_mut(dataset_id).ok_or_else(|| {
            anyhow::anyhow!(
                "cannot start streaming: dataset '{dataset_id}' is not registered \
                 (register it with DataPipeline::register_dataset first)"
            )
        })?;
        dataset.active = true;
        Ok(())
    }
    /// Stop streaming a dataset without discarding its buffered samples.
    pub async fn stop_streaming(&self, dataset_id: &str) -> Result<()> {
        let mut datasets = self
            .streaming_datasets
            .lock()
            .map_err(|_| anyhow::anyhow!("streaming dataset registry lock poisoned"))?;
        let dataset = datasets
            .get_mut(dataset_id)
            .ok_or_else(|| anyhow::anyhow!("unknown dataset '{dataset_id}'"))?;
        dataset.active = false;
        Ok(())
    }
    /// Draw up to `batch_size` samples from the active streaming datasets.
    ///
    /// Samples are taken round-robin from every active dataset (so no single source starves
    /// the others), validated with [`DataPipeline::validate_batch`], and filtered according to
    /// the validator's `error_handling` policy:
    ///
    /// * `Strict` — the first invalid sample aborts the batch with an error;
    /// * `Skip` / `Fix` — invalid samples are dropped from the batch;
    /// * `LogAndContinue` — invalid samples are kept and logged.
    ///
    /// # Errors
    ///
    /// Fails when `batch_size` is zero, when no dataset is streaming, or under a `Strict`
    /// validation policy when a sample is invalid.
    pub async fn get_batch(&self, batch_size: usize) -> Result<Vec<DataSample>> {
        if batch_size == 0 {
            return Err(anyhow::anyhow!("batch_size must be greater than zero"));
        }
        let collected = {
            let mut datasets = self
                .streaming_datasets
                .lock()
                .map_err(|_| anyhow::anyhow!("streaming dataset registry lock poisoned"))?;
            let mut active_ids: Vec<String> =
                datasets.iter().filter(|(_, d)| d.active).map(|(id, _)| id.clone()).collect();
            if active_ids.is_empty() {
                return Err(anyhow::anyhow!(
                    "no dataset is streaming; call DataPipeline::start_streaming first"
                ));
            }
            active_ids.sort();
            let mut collected = Vec::with_capacity(batch_size);
            let mut exhausted = 0usize;
            let mut cursor = 0usize;
            while collected.len() < batch_size && exhausted < active_ids.len() {
                let id = &active_ids[cursor % active_ids.len()];
                cursor += 1;
                let Some(dataset) = datasets.get_mut(id) else {
                    exhausted += 1;
                    continue;
                };
                match dataset.pop_sample() {
                    Some(sample) => {
                        exhausted = 0;
                        collected.push(sample);
                    },
                    None => exhausted += 1,
                }
            }
            collected
        };
        if collected.is_empty() {
            return Ok(Vec::new());
        }
        let results = self.validate_batch(&collected).await?;
        let error_handling = {
            let validator =
                self.validator.lock().map_err(|_| anyhow::anyhow!("validator lock poisoned"))?;
            validator.config.error_handling.clone()
        };
        let mut batch = Vec::with_capacity(collected.len());
        for (sample, result) in collected.into_iter().zip(results) {
            if result.is_valid {
                batch.push(sample);
                continue;
            }
            match error_handling {
                ErrorHandling::Strict => {
                    return Err(anyhow::anyhow!(
                        "sample '{}' failed validation: {}",
                        sample.id,
                        result
                            .errors
                            .iter()
                            .map(|e| e.message.clone())
                            .collect::<Vec<_>>()
                            .join("; ")
                    ));
                },
                ErrorHandling::Skip | ErrorHandling::Fix => {},
                ErrorHandling::LogAndContinue => {
                    log::warn!(
                        "data_pipeline: keeping invalid sample '{}': {}",
                        sample.id,
                        result
                            .errors
                            .iter()
                            .map(|e| e.message.clone())
                            .collect::<Vec<_>>()
                            .join("; ")
                    );
                    batch.push(sample);
                },
            }
        }
        Ok(batch)
    }
    /// Validate a batch of samples against the configured validator.
    ///
    /// Every sample is checked against the declarative rules in `DataValidationConfig.rules`
    /// **and** against every registered [`Validator`] trait object; the returned vector is
    /// aligned one-to-one with `samples`.
    pub async fn validate_batch(&self, samples: &[DataSample]) -> Result<Vec<ValidationResult>> {
        let mut validator =
            self.validator.lock().map_err(|_| anyhow::anyhow!("validator lock poisoned"))?;
        let mut results = Vec::with_capacity(samples.len());
        for sample in samples {
            results.push(validator.validate_sample(sample)?);
        }
        Ok(results)
    }
    /// Borrow the pipeline configuration.
    pub fn config(&self) -> &DataPipelineConfig {
        &self.config
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPipelineConfig {
    /// Streaming dataset configuration
    pub streaming: StreamingDatasetConfig,
    /// Dynamic augmentation configuration
    pub augmentation: DynamicAugmentationConfig,
    /// Curriculum learning configuration
    pub curriculum: CurriculumLearningConfig,
    /// Active learning configuration
    pub active_learning: ActiveLearningConfig,
    /// Multi-modal configuration
    pub multimodal: MultiModalConfig,
    /// Data validation configuration
    pub validation: DataValidationConfig,
    /// Distributed processing configuration
    pub distributed: DistributedProcessingConfig,
}
#[derive(Debug, Clone)]
pub struct DataSample {
    pub id: String,
    pub data: HashMap<String, Tensor>,
    pub metadata: HashMap<String, String>,
    pub timestamp: SystemTime,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedProcessingConfig {
    /// Number of worker processes
    pub num_workers: usize,
    /// Processing backend
    pub backend: ProcessingBackend,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin
    RoundRobin,
    /// Work-stealing
    WorkStealing,
    /// Dynamic load balancing
    Dynamic,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingBackend {
    /// Thread-based processing
    Threading,
    /// Process-based processing
    Multiprocessing,
    /// Ray distributed processing
    Ray { ray_config: HashMap<String, String> },
    /// Dask distributed processing
    Dask {
        dask_config: HashMap<String, String>,
    },
}
