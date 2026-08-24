//! In-memory registry of the checkpoints this server has loaded.
//!
//! Everything reported out of here is measured, not estimated: parameter counts
//! come from [`Model::num_parameters`], the architecture/shape numbers come from
//! the checkpoint's own [`AutoConfig`], and the request counter is a real
//! atomic. Capabilities the underlying library does not have (GPU placement,
//! fp16 weights, graph compilation) are rejected with a structured error rather
//! than accepted and silently ignored.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tracing::info;

use trustformers::core::traits::{Model, Tokenizer};
use trustformers::{AutoConfig, AutoModel, AutoTokenizer};

use crate::error::{ServiceError, ServiceResult};

/// The only device this example can actually run on.
///
/// `trustformers`' `AutoModel` loader has no device-placement parameter, so a
/// request for anything else is refused instead of being quietly downgraded to
/// CPU while the response still echoes the requested device back.
const SUPPORTED_DEVICE: &str = "cpu";

/// One checkpoint held in memory, together with the facts we can report about
/// it without guessing.
pub struct LoadedModel {
    /// The model, with its tokenizer attached so the generation entry points
    /// can encode prompts and decode completions.
    pub model: Arc<AutoModel>,
    /// The same tokenizer, kept separately so the service layer can decode
    /// token ids without going through the model.
    pub tokenizer: Arc<AutoTokenizer>,
    /// The checkpoint's own configuration; the source of every shape number
    /// this server reports.
    pub config: AutoConfig,
    /// Always [`SUPPORTED_DEVICE`] today, kept so the field means "where this
    /// model actually runs" rather than "what the client asked for".
    pub device: String,
    /// When the load finished.
    pub loaded_at: Instant,
    /// How long the load itself took.
    pub load_duration: Duration,
    /// Parameter count reported by the model itself.
    pub num_parameters: u64,
    /// Whether this architecture has a language-modelling head.
    pub is_generative: bool,
    /// Number of inference requests served since load.
    request_count: AtomicU64,
}

impl LoadedModel {
    /// Records one served request and returns the new total.
    pub fn record_request(&self) -> u64 {
        self.request_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// A consistent, lock-free view of everything the service layer reports.
    pub fn snapshot(&self, model_id: &str) -> ModelSnapshot {
        ModelSnapshot {
            model_id: model_id.to_string(),
            architecture: self.config.get_architecture_name(),
            num_parameters: self.num_parameters,
            is_generative: self.is_generative,
            hidden_size: self.config.get_hidden_size(),
            num_layers: self.config.get_num_layers(),
            num_heads: self.config.get_num_attention_heads(),
            vocab_size: self.config.get_vocab_size(),
            max_position_embeddings: self.config.get_max_sequence_length(),
            device: self.device.clone(),
            load_duration: self.load_duration,
            uptime: self.loaded_at.elapsed(),
            request_count: self.request_count.load(Ordering::Relaxed),
        }
    }
}

/// Immutable view of a loaded model, produced by [`LoadedModel::snapshot`].
#[derive(Debug, Clone)]
pub struct ModelSnapshot {
    pub model_id: String,
    pub architecture: &'static str,
    pub num_parameters: u64,
    pub is_generative: bool,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub device: String,
    pub load_duration: Duration,
    pub uptime: Duration,
    pub request_count: u64,
}

impl ModelSnapshot {
    /// Tasks this checkpoint can genuinely perform.
    ///
    /// Derived from the loaded architecture, not from a hardcoded list: every
    /// model exposes hidden states, and only checkpoints with a
    /// language-modelling head can generate text.
    pub fn supported_tasks(&self) -> Vec<String> {
        let mut tasks = vec!["feature-extraction".to_string()];
        if self.is_generative {
            tasks.push("text-generation".to_string());
        }
        tasks
    }
}

/// Registry of loaded checkpoints, keyed by the client-chosen model id.
pub struct ModelManager {
    models: DashMap<String, Arc<LoadedModel>>,
    max_models: usize,
    default_device: String,
}

impl ModelManager {
    /// Builds a registry, rejecting a configuration that could never serve a
    /// request.
    ///
    /// A `default_device` this build cannot run on is a startup error rather
    /// than something discovered later: otherwise every `LoadModel` would fail
    /// with a message naming a device the *client* never asked for.
    pub fn new(max_models: usize, default_device: String) -> ServiceResult<Self> {
        if max_models == 0 {
            return Err(ServiceError::InvalidInput(
                "at least one model slot is required".to_string(),
            ));
        }
        if !default_device.trim().eq_ignore_ascii_case(SUPPORTED_DEVICE) {
            return Err(ServiceError::Unsupported(format!(
                "default device `{default_device}` is not available: this server loads models                  through `trustformers::AutoModel`, which runs on `{SUPPORTED_DEVICE}` only"
            )));
        }
        Ok(Self {
            models: DashMap::new(),
            max_models,
            default_device,
        })
    }

    /// Normalises a requested device string, refusing anything this build
    /// cannot actually run on.
    fn resolve_device(&self, requested: Option<&str>) -> ServiceResult<String> {
        let requested = requested
            .map(str::trim)
            .filter(|device| !device.is_empty())
            .unwrap_or(&self.default_device);
        if requested.eq_ignore_ascii_case(SUPPORTED_DEVICE) {
            Ok(SUPPORTED_DEVICE.to_string())
        } else {
            Err(ServiceError::Unsupported(format!(
                "device `{requested}` is not available: this server loads models through \
                 `trustformers::AutoModel`, which runs on `{SUPPORTED_DEVICE}` only"
            )))
        }
    }

    /// Loads a checkpoint from `model_path` (or `model_id` when no path is
    /// given) and registers it under `model_id`.
    pub fn load_model(
        &self,
        model_id: &str,
        model_path: Option<&str>,
        device: Option<&str>,
        use_fp16: bool,
        compile: bool,
    ) -> ServiceResult<Duration> {
        if self.models.contains_key(model_id) {
            return Err(ServiceError::ModelAlreadyLoaded(model_id.to_string()));
        }

        if self.models.len() >= self.max_models {
            return Err(ServiceError::ResourceExhausted(format!(
                "maximum number of concurrently loaded models ({}) reached",
                self.max_models
            )));
        }

        let device = self.resolve_device(device)?;

        if use_fp16 {
            return Err(ServiceError::Unsupported(
                "`use_fp16` is not supported: this server loads whatever precision the \
                 checkpoint stores and performs no dtype conversion"
                    .to_string(),
            ));
        }
        if compile {
            return Err(ServiceError::Unsupported(
                "`compile` is not supported: this server has no graph-compilation backend"
                    .to_string(),
            ));
        }

        let start = Instant::now();
        let model_path = model_path
            .map(str::trim)
            .filter(|path| !path.is_empty())
            .unwrap_or(model_id);

        info!(model_id, model_path, device, "loading model");

        let config = AutoConfig::from_pretrained(model_path)
            .map_err(|e| ServiceError::Load(format!("config for `{model_path}`: {e}")))?;

        let model = AutoModel::from_pretrained(model_path)
            .map_err(|e| ServiceError::Load(format!("weights for `{model_path}`: {e}")))?;

        let tokenizer = Arc::new(
            AutoTokenizer::from_pretrained(model_path)
                .map_err(|e| ServiceError::Load(format!("tokenizer for `{model_path}`: {e}")))?,
        );

        // The generation entry points encode the prompt and decode the
        // completion through the model's own tokenizer, so it has to be
        // attached; without it every `generate_*` call fails.
        let shared_tokenizer: Arc<dyn Tokenizer> = Arc::clone(&tokenizer) as Arc<dyn Tokenizer>;
        let model = model.with_shared_tokenizer(shared_tokenizer);

        let num_parameters = model.num_parameters() as u64;
        let is_generative = model.is_generative();
        let load_duration = start.elapsed();

        let loaded_model = Arc::new(LoadedModel {
            model: Arc::new(model),
            tokenizer,
            config,
            device,
            loaded_at: Instant::now(),
            load_duration,
            num_parameters,
            is_generative,
            request_count: AtomicU64::new(0),
        });

        self.models.insert(model_id.to_string(), loaded_model);

        info!(
            model_id,
            num_parameters,
            elapsed_ms = load_duration.as_secs_f64() * 1000.0,
            "model loaded"
        );

        Ok(load_duration)
    }

    pub fn unload_model(&self, model_id: &str) -> ServiceResult<()> {
        self.models
            .remove(model_id)
            .ok_or_else(|| ServiceError::ModelNotFound(model_id.to_string()))?;

        info!(model_id, "model unloaded");
        Ok(())
    }

    pub fn get_model(&self, model_id: &str) -> ServiceResult<Arc<LoadedModel>> {
        self.models
            .get(model_id)
            .map(|entry| Arc::clone(entry.value()))
            .ok_or_else(|| ServiceError::ModelNotLoaded(model_id.to_string()))
    }

    pub fn list_models(&self) -> Vec<ModelSnapshot> {
        self.models
            .iter()
            .map(|entry| entry.value().snapshot(entry.key()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_device_is_rejected_rather_than_downgraded() {
        let manager = ModelManager::new(1, "cpu".to_string()).expect("cpu is supported");
        let err = manager
            .resolve_device(Some("cuda"))
            .expect_err("a CUDA request must not silently resolve to CPU");
        assert!(matches!(err, ServiceError::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn blank_device_falls_back_to_the_configured_default() {
        let manager = ModelManager::new(1, "cpu".to_string()).expect("cpu is supported");
        assert_eq!(
            manager.resolve_device(Some("  ")).expect("blank device"),
            "cpu"
        );
        assert_eq!(manager.resolve_device(None).expect("no device"), "cpu");
    }

    #[test]
    fn fp16_and_compile_requests_are_refused_before_any_io_happens() {
        let manager = ModelManager::new(1, "cpu".to_string()).expect("cpu is supported");
        // `/nonexistent` would fail the load anyway; the point is that the
        // unsupported-flag error wins, so the client is told the real reason.
        let err = manager
            .load_model("m", Some("/nonexistent"), Some("cpu"), true, false)
            .expect_err("fp16 must be refused");
        assert!(matches!(err, ServiceError::Unsupported(_)), "got {err:?}");

        let err = manager
            .load_model("m", Some("/nonexistent"), Some("cpu"), false, true)
            .expect_err("compile must be refused");
        assert!(matches!(err, ServiceError::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn a_default_device_the_build_cannot_run_on_is_a_startup_error() {
        // `ModelManager` has no `Debug`, so the Ok side is discarded before
        // `expect_err` (which needs one) is called.
        let err = ModelManager::new(1, "metal".to_string())
            .map(|_| ())
            .expect_err("a metal default must not produce a server that can load nothing");
        assert!(matches!(err, ServiceError::Unsupported(_)), "got {err:?}");

        let err = ModelManager::new(0, "cpu".to_string())
            .map(|_| ())
            .expect_err("a zero model budget must be rejected");
        assert!(matches!(err, ServiceError::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn supported_tasks_follow_the_architecture() {
        let generative = ModelSnapshot {
            model_id: "m".to_string(),
            architecture: "gpt2",
            num_parameters: 1,
            is_generative: true,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            vocab_size: 16,
            max_position_embeddings: 32,
            device: "cpu".to_string(),
            load_duration: Duration::from_millis(1),
            uptime: Duration::from_millis(1),
            request_count: 0,
        };
        assert_eq!(
            generative.supported_tasks(),
            vec![
                "feature-extraction".to_string(),
                "text-generation".to_string()
            ]
        );

        let encoder_only = ModelSnapshot {
            architecture: "bert",
            is_generative: false,
            ..generative
        };
        assert_eq!(
            encoder_only.supported_tasks(),
            vec!["feature-extraction".to_string()]
        );
    }
}
