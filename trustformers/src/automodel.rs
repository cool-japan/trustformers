use crate::core::traits::TokenizedInput;
use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read as IoRead;
use std::path::Path;
use std::sync::Arc;
use trustformers_core::errors::Result as CoreResult;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Config, Model, Tokenizer};
use trustformers_models::common_patterns::{DynConfig, GenerationConfig, GenerativeModel};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AutoConfig {
    #[cfg(feature = "bert")]
    Bert(crate::models::bert::BertConfig),
    #[cfg(feature = "roberta")]
    Roberta(crate::models::roberta::RobertaConfig),
    #[cfg(feature = "gpt2")]
    Gpt2(crate::models::gpt2::Gpt2Config),
    #[cfg(feature = "gpt_neo")]
    GptNeo(crate::models::gpt_neo::GptNeoConfig),
    #[cfg(feature = "gpt_j")]
    GptJ(crate::models::gpt_j::GptJConfig),
    #[cfg(feature = "t5")]
    T5(crate::models::t5::T5Config),
    #[cfg(feature = "albert")]
    Albert(crate::models::albert::AlbertConfig),
}

impl AutoConfig {
    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        Self::from_pretrained_with_revision(model_name_or_path, None)
    }

    /// Extract common model metadata from config variants
    pub fn get_vocab_size(&self) -> u32 {
        match self {
            #[cfg(feature = "bert")]
            AutoConfig::Bert(config) => config.vocab_size as u32,
            #[cfg(feature = "roberta")]
            AutoConfig::Roberta(config) => config.vocab_size as u32,
            #[cfg(feature = "gpt2")]
            AutoConfig::Gpt2(config) => config.vocab_size as u32,
            #[cfg(feature = "gpt_neo")]
            AutoConfig::GptNeo(config) => config.vocab_size as u32,
            #[cfg(feature = "gpt_j")]
            AutoConfig::GptJ(config) => config.vocab_size as u32,
            #[cfg(feature = "t5")]
            AutoConfig::T5(config) => config.vocab_size as u32,
            #[cfg(feature = "albert")]
            AutoConfig::Albert(config) => config.vocab_size as u32,
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    pub fn get_hidden_size(&self) -> u32 {
        match self {
            #[cfg(feature = "bert")]
            AutoConfig::Bert(config) => config.hidden_size as u32,
            #[cfg(feature = "roberta")]
            AutoConfig::Roberta(config) => config.hidden_size as u32,
            #[cfg(feature = "gpt2")]
            AutoConfig::Gpt2(config) => config.n_embd as u32,
            #[cfg(feature = "gpt_neo")]
            AutoConfig::GptNeo(config) => config.hidden_size as u32,
            #[cfg(feature = "gpt_j")]
            AutoConfig::GptJ(config) => config.n_embd as u32,
            #[cfg(feature = "t5")]
            AutoConfig::T5(config) => config.d_model as u32,
            #[cfg(feature = "albert")]
            AutoConfig::Albert(config) => config.hidden_size as u32,
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    pub fn get_num_layers(&self) -> u32 {
        match self {
            #[cfg(feature = "bert")]
            AutoConfig::Bert(config) => config.num_hidden_layers as u32,
            #[cfg(feature = "roberta")]
            AutoConfig::Roberta(config) => config.num_hidden_layers as u32,
            #[cfg(feature = "gpt2")]
            AutoConfig::Gpt2(config) => config.n_layer as u32,
            #[cfg(feature = "gpt_neo")]
            AutoConfig::GptNeo(config) => config.num_layers as u32,
            #[cfg(feature = "gpt_j")]
            AutoConfig::GptJ(config) => config.n_layer as u32,
            #[cfg(feature = "t5")]
            AutoConfig::T5(config) => config.num_layers as u32,
            #[cfg(feature = "albert")]
            AutoConfig::Albert(config) => config.num_hidden_layers as u32,
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    pub fn get_num_attention_heads(&self) -> u32 {
        match self {
            #[cfg(feature = "bert")]
            AutoConfig::Bert(config) => config.num_attention_heads as u32,
            #[cfg(feature = "roberta")]
            AutoConfig::Roberta(config) => config.num_attention_heads as u32,
            #[cfg(feature = "gpt2")]
            AutoConfig::Gpt2(config) => config.n_head as u32,
            #[cfg(feature = "gpt_neo")]
            AutoConfig::GptNeo(config) => config.num_heads as u32,
            #[cfg(feature = "gpt_j")]
            AutoConfig::GptJ(config) => config.n_head as u32,
            #[cfg(feature = "t5")]
            AutoConfig::T5(config) => config.num_heads as u32,
            #[cfg(feature = "albert")]
            AutoConfig::Albert(config) => config.num_attention_heads as u32,
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    pub fn get_max_sequence_length(&self) -> u32 {
        match self {
            #[cfg(feature = "bert")]
            AutoConfig::Bert(config) => config.max_position_embeddings as u32,
            #[cfg(feature = "roberta")]
            AutoConfig::Roberta(config) => config.max_position_embeddings as u32,
            #[cfg(feature = "gpt2")]
            AutoConfig::Gpt2(config) => config.n_positions as u32,
            #[cfg(feature = "gpt_neo")]
            AutoConfig::GptNeo(config) => config.max_position_embeddings as u32,
            #[cfg(feature = "gpt_j")]
            AutoConfig::GptJ(config) => config.n_positions as u32,
            // T5 uses relative-position attention buckets (see
            // `T5Config::relative_attention_max_distance`), not an absolute
            // position embedding table, so there is no config field bounding
            // sequence length the way `max_position_embeddings`/`n_positions`
            // do for the other architectures above; 512 is the commonly used
            // operating length for pretrained T5 checkpoints.
            #[cfg(feature = "t5")]
            AutoConfig::T5(_) => 512,
            #[cfg(feature = "albert")]
            AutoConfig::Albert(config) => config.max_position_embeddings as u32,
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    pub fn get_architecture_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "bert")]
            AutoConfig::Bert(_) => "bert",
            #[cfg(feature = "roberta")]
            AutoConfig::Roberta(_) => "roberta",
            #[cfg(feature = "gpt2")]
            AutoConfig::Gpt2(_) => "gpt2",
            #[cfg(feature = "gpt_neo")]
            AutoConfig::GptNeo(_) => "gpt_neo",
            #[cfg(feature = "gpt_j")]
            AutoConfig::GptJ(_) => "gpt_j",
            #[cfg(feature = "t5")]
            AutoConfig::T5(_) => "t5",
            #[cfg(feature = "albert")]
            AutoConfig::Albert(_) => "albert",
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    pub fn from_pretrained_with_revision(
        model_name_or_path: &str,
        revision: Option<&str>,
    ) -> Result<Self> {
        let config_path = Path::new(model_name_or_path).join("config.json");

        // Try to load config from local file first
        let config_str = if config_path.exists() {
            std::fs::read_to_string(&config_path)?
        } else {
            // Try to download from hub if not found locally
            let hub_options = crate::hub::HubOptions {
                revision: revision.map(|r| r.to_string()),
                cache_dir: None,
                force_download: false,
                token: None,
                parallel_downloads: true,
                max_concurrent_downloads: 4,
                enable_resumable_downloads: true,
                enable_delta_compression: true,
                chunk_size: 8192,
                timeout_seconds: 30,
                retry_attempts: 3,
                use_cdn: true,
                cdn_urls: vec![],
                smart_caching: true,
            };
            match crate::hub::download_file_from_hub(
                model_name_or_path,
                "config.json",
                Some(hub_options),
            ) {
                Ok(config_path) => std::fs::read_to_string(&config_path)?,
                Err(_) => {
                    // Fall back to name-based guessing
                    return Self::from_model_name(model_name_or_path);
                },
            }
        };

        let value: serde_json::Value = serde_json::from_str(&config_str)?;

        // Parse config based on model_type
        match value.get("model_type").and_then(|v| v.as_str()) {
            #[cfg(feature = "bert")]
            Some("bert") => {
                let config: crate::models::bert::BertConfig = serde_json::from_value(value)?;
                Ok(AutoConfig::Bert(config))
            },
            #[cfg(feature = "roberta")]
            Some("roberta") => {
                let config: crate::models::roberta::RobertaConfig = serde_json::from_value(value)?;
                Ok(AutoConfig::Roberta(config))
            },
            #[cfg(feature = "gpt2")]
            Some("gpt2") => {
                let config: crate::models::gpt2::Gpt2Config = serde_json::from_value(value)?;
                Ok(AutoConfig::Gpt2(config))
            },
            #[cfg(feature = "gpt_neo")]
            Some("gpt_neo") => {
                let config: crate::models::gpt_neo::GptNeoConfig = serde_json::from_value(value)?;
                Ok(AutoConfig::GptNeo(config))
            },
            #[cfg(feature = "gpt_j")]
            Some("gptj") => {
                let config: crate::models::gpt_j::GptJConfig = serde_json::from_value(value)?;
                Ok(AutoConfig::GptJ(config))
            },
            #[cfg(feature = "t5")]
            Some("t5") => {
                let config: crate::models::t5::T5Config = serde_json::from_value(value)?;
                Ok(AutoConfig::T5(config))
            },
            #[cfg(feature = "albert")]
            Some("albert") => {
                let config: crate::models::albert::AlbertConfig = serde_json::from_value(value)?;
                Ok(AutoConfig::Albert(config))
            },
            _ => Err(TrustformersError::invalid_input(
                format!(
                    "Unknown or unsupported model type in {}",
                    model_name_or_path
                ),
                Some("model_type"),
                Some("supported model type (bert, gpt2, t5, albert, etc.)"),
                None::<String>,
            )),
        }
    }

    /// Fallback method to guess model type from model name
    fn from_model_name(model_name_or_path: &str) -> Result<Self> {
        let model_name_lower = model_name_or_path.to_lowercase();

        if model_name_lower.contains("roberta") {
            #[cfg(feature = "roberta")]
            return Ok(AutoConfig::Roberta(
                crate::models::roberta::RobertaConfig::default(),
            ));
            #[cfg(not(feature = "roberta"))]
            return Err(TrustformersError::invalid_input(
                "RoBERTa feature not enabled".to_string(),
                Some("feature".to_string()),
                Some("roberta feature enabled".to_string()),
                Some("roberta feature disabled".to_string()),
            ));
        } else if model_name_lower.contains("bert") || model_name_lower.contains("distilbert") {
            #[cfg(feature = "bert")]
            return Ok(AutoConfig::Bert(crate::models::bert::BertConfig::default()));
            #[cfg(not(feature = "bert"))]
            return Err(TrustformersError::invalid_input_simple(
                "BERT feature not enabled".into(),
            ));
        } else if model_name_lower.contains("gpt-neo") || model_name_lower.contains("gpt_neo") {
            #[cfg(feature = "gpt_neo")]
            return Ok(AutoConfig::GptNeo(
                crate::models::gpt_neo::GptNeoConfig::from_pretrained_name(model_name_or_path),
            ));
            #[cfg(not(feature = "gpt_neo"))]
            return Err(TrustformersError::invalid_input_simple(
                "GPT-Neo feature not enabled",
            ));
        } else if model_name_lower.contains("gpt-j")
            || model_name_lower.contains("gpt_j")
            || model_name_lower.contains("gptj")
        {
            #[cfg(feature = "gpt_j")]
            return Ok(AutoConfig::GptJ(
                crate::models::gpt_j::GptJConfig::from_pretrained_name(model_name_or_path),
            ));
            #[cfg(not(feature = "gpt_j"))]
            return Err(TrustformersError::invalid_input_simple(
                "GPT-J feature not enabled",
            ));
        } else if model_name_lower.contains("gpt2") || model_name_lower.contains("gpt-2") {
            #[cfg(feature = "gpt2")]
            return Ok(AutoConfig::Gpt2(crate::models::gpt2::Gpt2Config::default()));
            #[cfg(not(feature = "gpt2"))]
            return Err(TrustformersError::invalid_input_simple(
                "GPT2 feature not enabled",
            ));
        } else if model_name_lower.contains("t5") {
            #[cfg(feature = "t5")]
            return Ok(AutoConfig::T5(
                crate::models::t5::T5Config::from_pretrained_name(model_name_or_path),
            ));
            #[cfg(not(feature = "t5"))]
            return Err(TrustformersError::invalid_input_simple(
                "T5 feature not enabled",
            ));
        } else if model_name_lower.contains("albert") {
            #[cfg(feature = "albert")]
            return Ok(AutoConfig::Albert(
                crate::models::albert::AlbertConfig::from_pretrained_name(model_name_or_path),
            ));
            #[cfg(not(feature = "albert"))]
            return Err(TrustformersError::invalid_input_simple(
                "ALBERT feature not enabled",
            ));
        } else {
            Err(TrustformersError::invalid_input(
                format!("Cannot determine model type from {}", model_name_or_path),
                Some("model_name_or_path"),
                Some("recognizable model name or valid config path"),
                Some(model_name_or_path),
            ))
        }
    }
}

impl Config for AutoConfig {
    fn validate(&self) -> CoreResult<()> {
        match self {
            #[cfg(feature = "bert")]
            AutoConfig::Bert(_config) => Ok(()),
            #[cfg(feature = "roberta")]
            AutoConfig::Roberta(_config) => Ok(()),
            #[cfg(feature = "gpt2")]
            AutoConfig::Gpt2(_config) => Ok(()),
            #[cfg(feature = "gpt_neo")]
            AutoConfig::GptNeo(_config) => Ok(()),
            #[cfg(feature = "gpt_j")]
            AutoConfig::GptJ(_config) => Ok(()),
            #[cfg(feature = "t5")]
            AutoConfig::T5(_config) => Ok(()),
            #[cfg(feature = "albert")]
            AutoConfig::Albert(_config) => Ok(()),
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    fn architecture(&self) -> &'static str {
        match self {
            #[cfg(feature = "bert")]
            AutoConfig::Bert(_) => "bert",
            #[cfg(feature = "roberta")]
            AutoConfig::Roberta(_) => "roberta",
            #[cfg(feature = "gpt2")]
            AutoConfig::Gpt2(_) => "gpt2",
            #[cfg(feature = "gpt_neo")]
            AutoConfig::GptNeo(_) => "gpt_neo",
            #[cfg(feature = "gpt_j")]
            AutoConfig::GptJ(_) => "gpt_j",
            #[cfg(feature = "t5")]
            AutoConfig::T5(_) => "t5",
            #[cfg(feature = "albert")]
            AutoConfig::Albert(_) => "albert",
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }
}

/// Weight file names [`AutoModel::from_pretrained`] recognises, in priority
/// order.
pub const WEIGHT_FILE_CANDIDATES: &[&str] = &[
    "model.safetensors",
    "pytorch_model.safetensors",
    "pytorch_model.bin",
    "model.bin",
];

/// Options controlling how [`AutoModel::from_pretrained_with_options`] loads a
/// checkpoint.
#[derive(Debug, Clone, Default)]
pub struct ModelLoadOptions {
    /// Hub revision (branch/tag/commit) to resolve, when applicable.
    pub revision: Option<String>,
    /// Accept a checkpoint that has **no** weight file and return a randomly
    /// initialised network instead.
    ///
    /// Off by default: a silent random init is indistinguishable from a real
    /// load at the call site, and every downstream number would be noise.
    pub allow_random_init: bool,
}

#[derive(Clone)]
pub struct AutoModel {
    pub config: AutoConfig,
    pub model_type: AutoModelType,
    /// Tokenizer that belongs to this checkpoint.
    ///
    /// Populated by [`AutoModel::from_pretrained`] when the checkpoint directory
    /// (or the Hub cache) carries a usable tokenizer. Text-level APIs such as
    /// [`GenerativeModel::generate`] require it: without a real vocabulary there
    /// is no honest way to turn a prompt into token ids, so those APIs return a
    /// structured error instead of guessing.
    tokenizer: Option<Arc<dyn Tokenizer>>,
}

#[derive(Clone)]
pub enum AutoModelType {
    #[cfg(feature = "bert")]
    Bert(crate::models::bert::BertModel),
    #[cfg(feature = "bert")]
    BertForMaskedLM(crate::models::bert::BertForMaskedLM),
    #[cfg(feature = "bert")]
    BertForSequenceClassification(crate::models::bert::BertForSequenceClassification),
    #[cfg(feature = "roberta")]
    Roberta(crate::models::roberta::RobertaModel),
    #[cfg(feature = "roberta")]
    RobertaForMaskedLM(crate::models::roberta::RobertaForMaskedLM),
    #[cfg(feature = "roberta")]
    RobertaForSequenceClassification(crate::models::roberta::RobertaForSequenceClassification),
    #[cfg(feature = "gpt2")]
    Gpt2(crate::models::gpt2::Gpt2Model),
    #[cfg(feature = "gpt2")]
    Gpt2LMHead(crate::models::gpt2::Gpt2LMHeadModel),
    #[cfg(feature = "gpt_neo")]
    GptNeo(crate::models::gpt_neo::GptNeoModel),
    #[cfg(feature = "gpt_neo")]
    GptNeoLMHead(crate::models::gpt_neo::GptNeoLMHeadModel),
    #[cfg(feature = "gpt_j")]
    GptJ(crate::models::gpt_j::GptJModel),
    #[cfg(feature = "gpt_j")]
    GptJLMHead(crate::models::gpt_j::GptJLMHeadModel),
    #[cfg(feature = "t5")]
    T5(crate::models::t5::T5Model),
    #[cfg(feature = "t5")]
    T5ForConditionalGeneration(crate::models::t5::T5ForConditionalGeneration),
    #[cfg(feature = "albert")]
    Albert(crate::models::albert::AlbertModel),
    #[cfg(feature = "albert")]
    AlbertForMaskedLM(crate::models::albert::AlbertForMaskedLM),
    #[cfg(feature = "albert")]
    AlbertForSequenceClassification(crate::models::albert::AlbertForSequenceClassification),
}

impl AutoModel {
    pub fn from_config(config: AutoConfig) -> Result<Self> {
        let model_type = match &config {
            #[cfg(feature = "bert")]
            AutoConfig::Bert(bert_config) => {
                AutoModelType::Bert(crate::models::bert::BertModel::new(bert_config.clone())?)
            },
            #[cfg(feature = "roberta")]
            AutoConfig::Roberta(roberta_config) => AutoModelType::Roberta(
                crate::models::roberta::RobertaModel::new(roberta_config.clone())?,
            ),
            #[cfg(feature = "gpt2")]
            AutoConfig::Gpt2(gpt2_config) => {
                AutoModelType::Gpt2(crate::models::gpt2::Gpt2Model::new(gpt2_config.clone())?)
            },
            #[cfg(feature = "gpt_neo")]
            AutoConfig::GptNeo(gpt_neo_config) => AutoModelType::GptNeo(
                crate::models::gpt_neo::GptNeoModel::new(gpt_neo_config.clone())?,
            ),
            #[cfg(feature = "gpt_j")]
            AutoConfig::GptJ(gpt_j_config) => {
                AutoModelType::GptJ(crate::models::gpt_j::GptJModel::new(gpt_j_config.clone())?)
            },
            #[cfg(feature = "t5")]
            AutoConfig::T5(t5_config) => {
                AutoModelType::T5(crate::models::t5::T5Model::new(t5_config.clone())?)
            },
            #[cfg(feature = "albert")]
            AutoConfig::Albert(albert_config) => AutoModelType::Albert(
                crate::models::albert::AlbertModel::new(albert_config.clone())?,
            ),
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        };

        Ok(AutoModel {
            config,
            model_type,
            tokenizer: None,
        })
    }

    /// Assemble a model from an already-built architecture instance.
    ///
    /// Useful when a caller has constructed a specific head (for example
    /// `BertForMaskedLM`) and wants to drive it through the `AutoModel`
    /// dispatch. No weights are loaded and no tokenizer is attached.
    pub fn from_parts(config: AutoConfig, model_type: AutoModelType) -> Self {
        Self {
            config,
            model_type,
            tokenizer: None,
        }
    }

    /// Attach the tokenizer that belongs to this checkpoint.
    ///
    /// Text-level generation ([`GenerativeModel::generate`]) needs a real
    /// vocabulary; use this when the model was built with
    /// [`AutoModel::from_config`] or when the tokenizer lives somewhere other
    /// than the model directory.
    pub fn with_tokenizer<T: Tokenizer + 'static>(mut self, tokenizer: T) -> Self {
        self.tokenizer = Some(Arc::new(tokenizer));
        self
    }

    /// Attach an already-shared tokenizer.
    pub fn with_shared_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Borrow the tokenizer attached to this model, if any.
    pub fn tokenizer(&self) -> Option<&Arc<dyn Tokenizer>> {
        self.tokenizer.as_ref()
    }

    /// Borrow the tokenizer, or fail with an actionable error.
    ///
    /// # Errors
    ///
    /// Returns a pipeline runtime error when no tokenizer was attached — the
    /// caller has to supply one via
    /// [`AutoModel::with_tokenizer`] or load the model through
    /// [`AutoModel::from_pretrained`] on a directory that contains one.
    pub fn require_tokenizer(&self) -> Result<&Arc<dyn Tokenizer>> {
        self.tokenizer.as_ref().ok_or_else(|| {
            TrustformersError::runtime_error(
                "no tokenizer is attached to this AutoModel: text generation needs the \
                 checkpoint's real vocabulary. Load the model with \
                 `AutoModel::from_pretrained(<dir containing tokenizer.json/vocab.txt>)` or \
                 attach one explicitly with `AutoModel::with_tokenizer(..)`."
                    .to_string(),
            )
        })
    }

    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        Self::from_pretrained_with_revision(model_name_or_path, None)
    }

    pub fn from_pretrained_with_revision(
        model_name_or_path: &str,
        revision: Option<&str>,
    ) -> Result<Self> {
        Self::from_pretrained_with_options(
            model_name_or_path,
            &ModelLoadOptions {
                revision: revision.map(str::to_string),
                ..ModelLoadOptions::default()
            },
        )
    }

    /// First recognised weight file inside `dir`, if any.
    ///
    /// `Checkpoint::from_bytes` auto-detects the container, so both
    /// safetensors and the pure-Rust torch `.bin` reader are covered.
    fn locate_weight_file(dir: &Path) -> Option<std::path::PathBuf> {
        WEIGHT_FILE_CANDIDATES
            .iter()
            .map(|name| dir.join(name))
            .find(|candidate| candidate.exists())
    }

    /// Load a checkpoint with explicit control over what counts as a
    /// successful load.
    ///
    /// By default the checkpoint **must** carry a weight file: returning a
    /// randomly initialised network from `from_pretrained` would look exactly
    /// like a successful load while producing meaningless outputs. Callers that
    /// genuinely want an untrained network (fresh training runs, architecture
    /// smoke tests) opt in via [`ModelLoadOptions::allow_random_init`].
    ///
    /// # Errors
    ///
    /// Fails when the configuration cannot be resolved, when no recognised
    /// weight file is present and random initialisation was not requested, or
    /// when the weight file itself cannot be parsed.
    pub fn from_pretrained_with_options(
        model_name_or_path: &str,
        options: &ModelLoadOptions,
    ) -> Result<Self> {
        let revision = options.revision.as_deref();
        let config = AutoConfig::from_pretrained_with_revision(model_name_or_path, revision)?;
        let mut model = Self::from_config(config)?;

        let weights_path = Self::locate_weight_file(Path::new(model_name_or_path));
        match (&weights_path, options.allow_random_init) {
            (None, false) => {
                return Err(TrustformersError::file_not_found(format!(
                    "no weight file found for `{model_name_or_path}` (looked for {}).                      Refusing to return a randomly initialised model from `from_pretrained`;                      pass `ModelLoadOptions {{ allow_random_init: true, .. }}` if an untrained                      network is genuinely what you want.",
                    WEIGHT_FILE_CANDIDATES.join(", ")
                )));
            },
            (None, true) => {
                tracing::warn!(
                    model = model_name_or_path,
                    "no weight file found; returning a randomly initialised model because                      `allow_random_init` was requested. Its outputs are not meaningful."
                );
            },
            (Some(_), _) => {},
        }

        if let Some(weights_path) = weights_path {
            match &mut model.model_type {
                #[cfg(feature = "bert")]
                AutoModelType::Bert(bert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    bert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "bert")]
                AutoModelType::BertForMaskedLM(bert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    bert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "bert")]
                AutoModelType::BertForSequenceClassification(bert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    bert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "roberta")]
                AutoModelType::Roberta(roberta) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    roberta.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "roberta")]
                AutoModelType::RobertaForMaskedLM(roberta) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    roberta.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "roberta")]
                AutoModelType::RobertaForSequenceClassification(roberta) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    roberta.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "gpt2")]
                AutoModelType::Gpt2(gpt2) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    gpt2.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "gpt2")]
                AutoModelType::Gpt2LMHead(gpt2) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    gpt2.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "gpt_neo")]
                AutoModelType::GptNeo(gpt_neo) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    gpt_neo.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "gpt_neo")]
                AutoModelType::GptNeoLMHead(gpt_neo) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    gpt_neo.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "gpt_j")]
                AutoModelType::GptJ(gpt_j) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    gpt_j.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "gpt_j")]
                AutoModelType::GptJLMHead(gpt_j) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    gpt_j.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "t5")]
                AutoModelType::T5(t5) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    t5.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "t5")]
                AutoModelType::T5ForConditionalGeneration(t5) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    t5.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "albert")]
                AutoModelType::Albert(albert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    albert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "albert")]
                AutoModelType::AlbertForMaskedLM(albert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    albert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "albert")]
                AutoModelType::AlbertForSequenceClassification(albert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    albert.load_pretrained(&mut reader)?;
                },
            }
        }

        // Attach the checkpoint's own tokenizer when it ships one. This is what
        // makes real (non-fabricated) text generation possible; when the
        // checkpoint has no tokenizer the field stays `None` and the text-level
        // APIs return a structured error rather than inventing token ids.
        match AutoTokenizer::from_pretrained_with_revision(model_name_or_path, revision) {
            Ok(tokenizer) => model.tokenizer = Some(Arc::new(tokenizer)),
            Err(err) => {
                tracing::debug!(
                    model = model_name_or_path,
                    error = %err,
                    "no tokenizer found next to the checkpoint; text-level APIs will error"
                );
            },
        }

        Ok(model)
    }
}

#[derive(Clone)]
pub enum AutoTokenizer {
    WordPiece(crate::tokenizers::WordPieceTokenizer),
    BPE(crate::tokenizers::BPETokenizer),
    SentencePiece(crate::tokenizers::SentencePieceTokenizer),
    HuggingFace(crate::tokenizers::TokenizerImpl),
}

impl AutoTokenizer {
    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        Self::from_pretrained_with_revision(model_name_or_path, None)
    }

    pub fn from_pretrained_with_revision(
        model_name_or_path: &str,
        revision: Option<&str>,
    ) -> Result<Self> {
        let base_path = Path::new(model_name_or_path);

        // Try to detect tokenizer type from available files. Each tokenizer
        // type below derives its own vocab/merges path (extensions differ:
        // vocab.txt for BERT-style WordPiece, vocab.json for GPT2/RoBERTa
        // BPE), so there is no single vocab/merges path to precompute here.
        let tokenizer_path = base_path.join("tokenizer.json");
        let tokenizer_config_path = base_path.join("tokenizer_config.json");

        // Check for tokenizer config to understand the tokenizer type
        if tokenizer_config_path.exists() {
            let config_str = std::fs::read_to_string(&tokenizer_config_path).ok();
            if let Some(config_str) = config_str {
                if let Ok(config_value) = serde_json::from_str::<serde_json::Value>(&config_str) {
                    if let Some(tokenizer_class) =
                        config_value.get("tokenizer_class").and_then(|v| v.as_str())
                    {
                        match tokenizer_class {
                            "BertTokenizer" | "DistilBertTokenizer" => {
                                // Try to use WordPiece tokenizer with vocab file
                                let vocab_path = format!("{}/vocab.txt", model_name_or_path);
                                if std::path::Path::new(&vocab_path).exists() {
                                    if let Ok(tokenizer) =
                                        crate::tokenizers::WordPieceTokenizer::from_vocab_file(
                                            &vocab_path,
                                            true,
                                        )
                                    {
                                        return Ok(AutoTokenizer::WordPiece(tokenizer));
                                    }
                                }
                            },
                            "GPT2Tokenizer" => {
                                // Try to use BPE tokenizer with vocab and merges files
                                let vocab_path = format!("{}/vocab.json", model_name_or_path);
                                let merges_path = format!("{}/merges.txt", model_name_or_path);
                                if std::path::Path::new(&vocab_path).exists()
                                    && std::path::Path::new(&merges_path).exists()
                                {
                                    if let Ok(tokenizer) =
                                        crate::tokenizers::BPETokenizer::from_files(
                                            &vocab_path,
                                            &merges_path,
                                        )
                                    {
                                        return Ok(AutoTokenizer::BPE(tokenizer));
                                    }
                                }
                            },
                            "RobertaTokenizer" => {
                                // Try to use RoBERTa-specific BPE tokenizer with vocab and merges files
                                let vocab_path = format!("{}/vocab.json", model_name_or_path);
                                let merges_path = format!("{}/merges.txt", model_name_or_path);
                                if std::path::Path::new(&vocab_path).exists()
                                    && std::path::Path::new(&merges_path).exists()
                                {
                                    if let Ok(tokenizer) =
                                        crate::tokenizers::BPETokenizer::from_roberta_files(
                                            &vocab_path,
                                            &merges_path,
                                        )
                                    {
                                        return Ok(AutoTokenizer::BPE(tokenizer));
                                    }
                                }
                            },
                            "T5Tokenizer" => {
                                // Use SentencePiece for T5
                                let tokenizer =
                                    crate::tokenizers::SentencePieceTokenizer::from_pretrained(
                                        model_name_or_path,
                                    )?;
                                return Ok(AutoTokenizer::SentencePiece(tokenizer));
                            },
                            _ => {},
                        }
                    }
                }
            }
        }

        // Also check model name patterns for T5
        let model_name_lower = model_name_or_path.to_lowercase();
        if model_name_lower.contains("t5") {
            let tokenizer =
                crate::tokenizers::SentencePieceTokenizer::from_pretrained(model_name_or_path)?;
            return Ok(AutoTokenizer::SentencePiece(tokenizer));
        }

        // Use standard HuggingFace tokenizer for now
        if tokenizer_path.exists() {
            let tokenizer = crate::tokenizers::TokenizerImpl::from_file(&tokenizer_path)?;
            Ok(AutoTokenizer::HuggingFace(tokenizer))
        } else {
            // Try to download tokenizer.json from the Hub with revision support,
            // mirroring AutoConfig::from_pretrained_with_revision's pattern above:
            // download_file_from_hub already early-returns without any network
            // access when the file is already present in the local Hub cache.
            let hub_options = crate::hub::HubOptions {
                revision: revision.map(|r| r.to_string()),
                cache_dir: None,
                force_download: false,
                token: None,
                parallel_downloads: true,
                max_concurrent_downloads: 4,
                enable_resumable_downloads: true,
                enable_delta_compression: true,
                chunk_size: 8192,
                timeout_seconds: 30,
                retry_attempts: 3,
                use_cdn: true,
                cdn_urls: vec![],
                smart_caching: true,
            };
            match crate::hub::download_file_from_hub(
                model_name_or_path,
                "tokenizer.json",
                Some(hub_options),
            ) {
                Ok(downloaded_path) => {
                    let tokenizer = crate::tokenizers::TokenizerImpl::from_file(&downloaded_path)?;
                    Ok(AutoTokenizer::HuggingFace(tokenizer))
                },
                Err(_) => {
                    // Fall back to the local-cache-path-only lookup (kept for
                    // compatibility with pre-populated HF_HOME/TRANSFORMERS_CACHE
                    // caches that TokenizerImpl checks directly).
                    let tokenizer =
                        crate::tokenizers::TokenizerImpl::from_pretrained_with_revision(
                            model_name_or_path,
                            revision,
                        )?;
                    Ok(AutoTokenizer::HuggingFace(tokenizer))
                },
            }
        }
    }
}

impl Tokenizer for AutoTokenizer {
    fn encode(&self, text: &str) -> CoreResult<crate::core::traits::TokenizedInput> {
        match self {
            AutoTokenizer::WordPiece(t) => t.encode(text),
            AutoTokenizer::BPE(t) => t.encode(text),
            AutoTokenizer::SentencePiece(t) => t.encode(text),
            AutoTokenizer::HuggingFace(t) => t.encode(text),
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    fn encode_pair(
        &self,
        text: &str,
        text2: &str,
    ) -> CoreResult<crate::core::traits::TokenizedInput> {
        match self {
            AutoTokenizer::WordPiece(t) => t.encode_pair(text, text2),
            AutoTokenizer::BPE(t) => t.encode_pair(text, text2),
            AutoTokenizer::SentencePiece(t) => t.encode_pair(text, text2),
            AutoTokenizer::HuggingFace(t) => t.encode_pair(text, text2),
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    fn decode(&self, ids: &[u32]) -> CoreResult<String> {
        match self {
            AutoTokenizer::WordPiece(t) => t.decode(ids),
            AutoTokenizer::BPE(t) => t.decode(ids),
            AutoTokenizer::SentencePiece(t) => t.decode(ids),
            AutoTokenizer::HuggingFace(t) => t.decode(ids),
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            AutoTokenizer::WordPiece(t) => t.vocab_size(),
            AutoTokenizer::BPE(t) => t.vocab_size(),
            AutoTokenizer::SentencePiece(t) => t.vocab_size(),
            AutoTokenizer::HuggingFace(t) => t.vocab_size(),
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        match self {
            AutoTokenizer::WordPiece(t) => t.get_vocab(),
            AutoTokenizer::BPE(t) => t.get_vocab(),
            AutoTokenizer::SentencePiece(t) => t.get_vocab(),
            AutoTokenizer::HuggingFace(t) => t.get_vocab(),
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            AutoTokenizer::WordPiece(t) => t.token_to_id(token),
            AutoTokenizer::BPE(t) => t.token_to_id(token),
            AutoTokenizer::SentencePiece(t) => t.token_to_id(token),
            AutoTokenizer::HuggingFace(t) => t.token_to_id(token),
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match self {
            AutoTokenizer::WordPiece(t) => t.id_to_token(id),
            AutoTokenizer::BPE(t) => t.id_to_token(id),
            AutoTokenizer::SentencePiece(t) => t.id_to_token(id),
            AutoTokenizer::HuggingFace(t) => t.id_to_token(id),
            // reason: catch-all is unreachable when model features are enabled, but
            // required so the match stays exhaustive across arbitrary feature subsets.
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }
}

impl Model for AutoModel {
    type Config = AutoConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        // Convert tensor input to TokenizedInput
        let token_input = {
            let data = input.data()?;
            let input_ids: Vec<u32> = data.iter().map(|&x| x as u32).collect();
            let attention_mask = vec![1u8; input_ids.len()];
            TokenizedInput {
                input_ids,
                attention_mask,
                token_type_ids: None,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            }
        };

        match &self.model_type {
            #[cfg(feature = "bert")]
            AutoModelType::Bert(model) => {
                let output = model.forward(token_input)?;
                Ok(output.last_hidden_state)
            },
            #[cfg(feature = "bert")]
            AutoModelType::BertForMaskedLM(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.logits)
            },
            #[cfg(feature = "bert")]
            AutoModelType::BertForSequenceClassification(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.logits)
            },
            #[cfg(feature = "roberta")]
            AutoModelType::Roberta(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.last_hidden_state)
            },
            #[cfg(feature = "roberta")]
            AutoModelType::RobertaForMaskedLM(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.logits)
            },
            #[cfg(feature = "roberta")]
            AutoModelType::RobertaForSequenceClassification(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.logits)
            },
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.last_hidden_state)
            },
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2LMHead(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.logits)
            },
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeo(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.last_hidden_state)
            },
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeoLMHead(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.logits)
            },
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJ(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.last_hidden_state)
            },
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJLMHead(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.logits)
            },
            #[cfg(feature = "t5")]
            AutoModelType::T5(model) => {
                let t5_input = trustformers_models::t5::T5Input {
                    input_ids: token_input.clone(),
                    decoder_input_ids: None,
                    encoder_outputs: None,
                };
                let output = model.forward(t5_input)?;
                Ok(output.last_hidden_state)
            },
            #[cfg(feature = "t5")]
            AutoModelType::T5ForConditionalGeneration(model) => {
                let t5_input = trustformers_models::t5::T5Input {
                    input_ids: token_input.clone(),
                    decoder_input_ids: None,
                    encoder_outputs: None,
                };
                let output = model.forward(t5_input)?;
                Ok(output.logits)
            },
            #[cfg(feature = "albert")]
            AutoModelType::Albert(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.last_hidden_state)
            },
            #[cfg(feature = "albert")]
            AutoModelType::AlbertForMaskedLM(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.logits)
            },
            #[cfg(feature = "albert")]
            AutoModelType::AlbertForSequenceClassification(model) => {
                let output = model.forward(token_input.clone())?;
                Ok(output.logits)
            },
        }
    }

    fn load_pretrained(&mut self, reader: &mut dyn IoRead) -> CoreResult<()> {
        match &mut self.model_type {
            #[cfg(feature = "bert")]
            AutoModelType::Bert(model) => model.load_pretrained(reader),
            #[cfg(feature = "bert")]
            AutoModelType::BertForMaskedLM(model) => model.load_pretrained(reader),
            #[cfg(feature = "bert")]
            AutoModelType::BertForSequenceClassification(model) => model.load_pretrained(reader),
            #[cfg(feature = "roberta")]
            AutoModelType::Roberta(model) => model.load_pretrained(reader),
            #[cfg(feature = "roberta")]
            AutoModelType::RobertaForMaskedLM(model) => model.load_pretrained(reader),
            #[cfg(feature = "roberta")]
            AutoModelType::RobertaForSequenceClassification(model) => model.load_pretrained(reader),
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2(model) => model.load_pretrained(reader),
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2LMHead(model) => model.load_pretrained(reader),
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeo(model) => model.load_pretrained(reader),
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeoLMHead(model) => model.load_pretrained(reader),
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJ(model) => model.load_pretrained(reader),
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJLMHead(model) => model.load_pretrained(reader),
            #[cfg(feature = "t5")]
            AutoModelType::T5(model) => model.load_pretrained(reader),
            #[cfg(feature = "t5")]
            AutoModelType::T5ForConditionalGeneration(model) => model.load_pretrained(reader),
            #[cfg(feature = "albert")]
            AutoModelType::Albert(model) => model.load_pretrained(reader),
            #[cfg(feature = "albert")]
            AutoModelType::AlbertForMaskedLM(model) => model.load_pretrained(reader),
            #[cfg(feature = "albert")]
            AutoModelType::AlbertForSequenceClassification(model) => model.load_pretrained(reader),
        }
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        match &self.model_type {
            #[cfg(feature = "bert")]
            AutoModelType::Bert(model) => model.num_parameters(),
            #[cfg(feature = "bert")]
            AutoModelType::BertForMaskedLM(model) => model.num_parameters(),
            #[cfg(feature = "bert")]
            AutoModelType::BertForSequenceClassification(model) => model.num_parameters(),
            #[cfg(feature = "roberta")]
            AutoModelType::Roberta(model) => model.num_parameters(),
            #[cfg(feature = "roberta")]
            AutoModelType::RobertaForMaskedLM(model) => model.num_parameters(),
            #[cfg(feature = "roberta")]
            AutoModelType::RobertaForSequenceClassification(model) => model.num_parameters(),
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2(model) => model.num_parameters(),
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2LMHead(model) => model.num_parameters(),
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeo(model) => model.num_parameters(),
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeoLMHead(model) => model.num_parameters(),
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJ(model) => model.num_parameters(),
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJLMHead(model) => model.num_parameters(),
            #[cfg(feature = "t5")]
            AutoModelType::T5(model) => model.num_parameters(),
            #[cfg(feature = "t5")]
            AutoModelType::T5ForConditionalGeneration(model) => model.num_parameters(),
            #[cfg(feature = "albert")]
            AutoModelType::Albert(model) => model.num_parameters(),
            #[cfg(feature = "albert")]
            AutoModelType::AlbertForMaskedLM(model) => model.num_parameters(),
            #[cfg(feature = "albert")]
            AutoModelType::AlbertForSequenceClassification(model) => model.num_parameters(),
            #[cfg(feature = "distilbert")]
            AutoModelType::DistilBert(model) => model.num_parameters(),
            #[cfg(feature = "distilbert")]
            AutoModelType::DistilBertForMaskedLM(model) => model.num_parameters(),
            #[cfg(feature = "distilbert")]
            AutoModelType::DistilBertForSequenceClassification(model) => model.num_parameters(),
            #[cfg(feature = "electra")]
            AutoModelType::Electra(model) => model.num_parameters(),
            #[cfg(feature = "electra")]
            AutoModelType::ElectraForMaskedLM(model) => model.num_parameters(),
            #[cfg(feature = "electra")]
            AutoModelType::ElectraForSequenceClassification(model) => model.num_parameters(),
        }
    }
}

impl GenerativeModel for AutoModel {
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> anyhow::Result<String> {
        let outcome = self.generate_token_ids(prompt, config)?;
        let tokenizer = self.require_tokenizer()?;
        Ok(tokenizer.decode(&outcome.sequence)?)
    }

    fn generate_batch(
        &self,
        prompts: &[&str],
        config: &GenerationConfig,
    ) -> anyhow::Result<Vec<String>> {
        prompts.iter().map(|prompt| self.generate(prompt, config)).collect()
    }

    fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> anyhow::Result<Box<dyn Iterator<Item = anyhow::Result<String>>>> {
        // Real incremental decoding: every `next()` runs exactly one more
        // decoding step and yields the newly decoded text delta. Time-to-first
        // token is therefore one forward pass, not a full generation.
        let stream = self.into_token_stream(prompt, config)?;
        Ok(Box::new(stream.map(|step| step.map(|s| s.text_delta))))
    }

    fn max_context_length(&self) -> usize {
        match &self.model_type {
            #[cfg(feature = "bert")]
            AutoModelType::Bert(_) => 512,
            #[cfg(feature = "bert")]
            AutoModelType::BertForMaskedLM(_) => 512,
            #[cfg(feature = "bert")]
            AutoModelType::BertForSequenceClassification(_) => 512,
            #[cfg(feature = "roberta")]
            AutoModelType::Roberta(_) => 512,
            #[cfg(feature = "roberta")]
            AutoModelType::RobertaForMaskedLM(_) => 512,
            #[cfg(feature = "roberta")]
            AutoModelType::RobertaForSequenceClassification(_) => 512,
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2(_) => 1024,
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2LMHead(_) => 1024,
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeo(_) => 2048,
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeoLMHead(_) => 2048,
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJ(_) => 2048,
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJLMHead(_) => 2048,
            #[cfg(feature = "t5")]
            AutoModelType::T5(_) => 512,
            #[cfg(feature = "t5")]
            AutoModelType::T5ForConditionalGeneration(_) => 512,
            #[cfg(feature = "albert")]
            AutoModelType::Albert(_) => 512,
            #[cfg(feature = "albert")]
            AutoModelType::AlbertForMaskedLM(_) => 512,
            #[cfg(feature = "albert")]
            AutoModelType::AlbertForSequenceClassification(_) => 512,
        }
    }

    fn config(&self) -> &dyn DynConfig {
        // Return the config as a DynConfig
        &self.config
    }

    fn supports_task(&self, task: &trustformers_models::common_patterns::TaskType) -> bool {
        match &self.model_type {
            #[cfg(feature = "bert")]
            AutoModelType::Bert(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextClassification
                    | trustformers_models::common_patterns::TaskType::QuestionAnswering
            ),
            #[cfg(feature = "bert")]
            AutoModelType::BertForMaskedLM(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextClassification
            ),
            #[cfg(feature = "bert")]
            AutoModelType::BertForSequenceClassification(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextClassification
            ),
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextGeneration
                    | trustformers_models::common_patterns::TaskType::CodeGeneration
            ),
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2LMHead(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextGeneration
                    | trustformers_models::common_patterns::TaskType::CodeGeneration
            ),
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeo(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextGeneration
                    | trustformers_models::common_patterns::TaskType::CodeGeneration
            ),
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeoLMHead(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextGeneration
                    | trustformers_models::common_patterns::TaskType::CodeGeneration
            ),
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJ(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextGeneration
                    | trustformers_models::common_patterns::TaskType::CodeGeneration
            ),
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJLMHead(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextGeneration
                    | trustformers_models::common_patterns::TaskType::CodeGeneration
            ),
            #[cfg(feature = "t5")]
            AutoModelType::T5(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextGeneration
                    | trustformers_models::common_patterns::TaskType::Summarization
                    | trustformers_models::common_patterns::TaskType::Translation
            ),
            #[cfg(feature = "t5")]
            AutoModelType::T5ForConditionalGeneration(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextGeneration
                    | trustformers_models::common_patterns::TaskType::Summarization
                    | trustformers_models::common_patterns::TaskType::Translation
            ),
            #[cfg(feature = "roberta")]
            AutoModelType::Roberta(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextClassification
                    | trustformers_models::common_patterns::TaskType::QuestionAnswering
            ),
            #[cfg(feature = "roberta")]
            AutoModelType::RobertaForMaskedLM(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextClassification
            ),
            #[cfg(feature = "roberta")]
            AutoModelType::RobertaForSequenceClassification(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextClassification
            ),
            #[cfg(feature = "albert")]
            AutoModelType::Albert(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextClassification
                    | trustformers_models::common_patterns::TaskType::QuestionAnswering
            ),
            #[cfg(feature = "albert")]
            AutoModelType::AlbertForMaskedLM(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextClassification
            ),
            #[cfg(feature = "albert")]
            AutoModelType::AlbertForSequenceClassification(_) => matches!(
                task,
                trustformers_models::common_patterns::TaskType::TextClassification
            ),
        }
    }
}

/// Real autoregressive generation for [`AutoModel`].
#[path = "automodel_generation.rs"]
mod generation;

pub use generation::{AutoModelTokenStream, GeneratedSequence, GenerationStep};

#[cfg(test)]
mod generation_honesty_tests {
    use super::*;
    use std::collections::HashMap;

    /// A real WordPiece tokenizer over a handful of words.
    #[cfg(feature = "bert")]
    fn tiny_tokenizer() -> crate::tokenizers::WordPieceTokenizer {
        let vocab: HashMap<String, u32> = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world",
        ]
        .iter()
        .enumerate()
        .map(|(i, w)| ((*w).to_string(), i as u32))
        .collect();
        crate::tokenizers::WordPieceTokenizer::new(vocab, true)
    }

    #[cfg(feature = "bert")]
    fn tiny_bert_config() -> crate::models::bert::BertConfig {
        crate::models::bert::BertConfig {
            vocab_size: 7,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 32,
            ..crate::models::bert::BertConfig::default()
        }
    }

    /// A model with no LM head must refuse to generate.
    ///
    /// The previous implementation returned
    /// `Ok("Model does not support text generation. Prompt: …")`, i.e. a
    /// successful-looking result carrying no model output at all.
    #[cfg(feature = "bert")]
    #[test]
    fn non_generative_model_errors_instead_of_returning_a_sentence() {
        let model = AutoModel::from_config(AutoConfig::Bert(tiny_bert_config()))
            .expect("tiny bert should build")
            .with_tokenizer(tiny_tokenizer());

        let Err(err) = model.generate("hello world", &GenerationConfig::default()) else {
            panic!("a headless BERT cannot generate text");
        };
        let message = err.to_string();
        assert!(
            !message.contains("Prompt:"),
            "the error must not echo the prompt back as if it were output: {message}"
        );
        assert!(
            message.contains("language-modelling head"),
            "the error should explain why generation is impossible: {message}"
        );
    }

    /// With none of `gpt2`/`gpt_neo`/`gpt_j`/`t5` compiled in (this crate's
    /// plain default features), `generate` must say so explicitly rather
    /// than reusing the generic "no language-modelling head" wording that
    /// implies the checkpoint's *architecture* is the problem. Before this
    /// existed, `generate_token_ids`'s `config`/`prompt_len` locals were
    /// unused dead code in exactly this build configuration because every
    /// match arm that read them was feature-gated out.
    #[cfg(not(any(
        feature = "gpt2",
        feature = "gpt_neo",
        feature = "gpt_j",
        feature = "t5"
    )))]
    #[cfg(feature = "bert")]
    #[test]
    fn generation_without_any_compiled_generative_feature_names_the_missing_features() {
        let model = AutoModel::from_config(AutoConfig::Bert(tiny_bert_config()))
            .expect("tiny bert should build")
            .with_tokenizer(tiny_tokenizer());

        let Err(err) = model.generate("hello world", &GenerationConfig::default()) else {
            panic!("no generative feature is compiled in, so generation must fail");
        };
        let message = err.to_string();
        assert!(
            message.contains("gpt2") && message.contains("t5"),
            "the error should name the generative features that would need enabling: {message}"
        );
        assert!(
            !message.contains("Prompt:"),
            "the error must not echo the prompt back as if it were output: {message}"
        );
    }

    /// Generation without a tokenizer must fail rather than fall back to
    /// character codes.
    #[cfg(feature = "bert")]
    #[test]
    fn generation_without_a_tokenizer_errors() {
        let model = AutoModel::from_config(AutoConfig::Bert(tiny_bert_config()))
            .expect("tiny bert should build");
        assert!(model.tokenizer().is_none());
        let Err(err) = model.generate("hello", &GenerationConfig::default()) else {
            panic!("no vocabulary means no honest tokenization");
        };
        assert!(
            err.to_string().contains("tokenizer"),
            "error should name the missing tokenizer: {err}"
        );
    }

    /// `from_pretrained` must not hand back an untrained network.
    #[cfg(feature = "bert")]
    #[test]
    fn from_pretrained_requires_weights_unless_opted_in() {
        let dir =
            std::env::temp_dir().join(format!("trustformers_weightless_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let config_json = serde_json::json!({
            "model_type": "bert",
            "vocab_size": 7,
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 32,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "pad_token_id": 0
        });
        std::fs::write(dir.join("config.json"), config_json.to_string()).expect("write config");
        let path = dir.to_string_lossy().to_string();

        let strict = AutoModel::from_pretrained(&path);
        let permissive = AutoModel::from_pretrained_with_options(
            &path,
            &ModelLoadOptions {
                allow_random_init: true,
                ..ModelLoadOptions::default()
            },
        );
        std::fs::remove_dir_all(&dir).ok();

        let Err(err) = strict else {
            panic!("a config-only directory carries no trained weights");
        };
        assert!(
            err.to_string().contains("no weight file"),
            "error should say which files were looked for: {err}"
        );
        assert!(
            permissive.is_ok(),
            "an explicit opt-in must still be able to build an untrained network"
        );
    }

    /// Encoder-decoder sequences cannot be scored by the decoder-only path.
    ///
    /// Feeding decoder ids into T5's encoder returns a plausible float that is
    /// not a log-probability of anything; refusing is the only honest answer.
    #[cfg(feature = "t5")]
    #[test]
    fn t5_sequence_scoring_is_refused() {
        let config = crate::models::t5::T5Config::default();
        let model = AutoModel::from_config(AutoConfig::T5(config)).expect("t5 should build");
        assert!(model.sequence_log_prob(&[1, 2, 3], 0).is_err());
    }
}

#[cfg(test)]
#[path = "automodel_tests.rs"]
mod automodel_tests;

#[cfg(test)]
mod hub_download_tests {
    use super::*;
    use std::fs;

    /// A minimal-but-valid `tokenizer.json` (HuggingFace `tokenizers` crate
    /// format: a `WordLevel` model needs only `vocab` + `unk_token`; every
    /// other top-level field is optional and defaults via `TokenizerBuilder`).
    /// Proven to parse successfully via the same `serde_json`-based
    /// `Tokenizer::from_file`/`from_str` path exercised by
    /// `trustformers-tokenizers/src/tokenizer.rs`'s own
    /// `test_tokenizer_from_json_string` test.
    const TOKENIZER_JSON_FIXTURE: &str = r#"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [
        { "id": 0, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
        { "id": 1, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true }
    ],
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": {
        "type": "WordLevel",
        "vocab": { "[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3 },
        "unk_token": "[UNK]"
    }
}"#;

    /// Cache-hit regression test for `AutoTokenizer::from_pretrained_with_revision`.
    ///
    /// Mirrors the early-return-on-cache-hit behavior already relied upon by
    /// `AutoConfig::from_pretrained_with_revision` (130 lines above, via
    /// `crate::hub::download_file_from_hub`): when `tokenizer.json` is already
    /// present at the exact path the Hub cache resolves to, loading must
    /// succeed purely from that cached file with **no network access** —
    /// `download_file_from_hub`'s own `!opts.force_download && file_path.exists()`
    /// check returns early, before the feature-gated `download_file` (the
    /// only place that ever touches the network) is ever called. This holds
    /// regardless of whether the `hub` feature is enabled, since neither
    /// `get_cache_dir` nor `download_file_from_hub` are feature-gated.
    ///
    /// The cache directory is redirected to a fresh `std::env::temp_dir()`
    /// subdirectory via `TRUSTFORMERS_CACHE` so this test never touches the
    /// real/shared model cache on the machine running it.
    #[test]
    fn test_auto_tokenizer_from_pretrained_with_revision_cache_hit_no_network() {
        let temp_cache = std::env::temp_dir().join(format!(
            "trustformers_test_tokenizer_cache_{}",
            std::process::id()
        ));
        let previous_cache_env = std::env::var("TRUSTFORMERS_CACHE").ok();
        std::env::set_var("TRUSTFORMERS_CACHE", &temp_cache);

        let model_id = "trustformers-test-org/cache-hit-tokenizer";
        let revision_dir = temp_cache.join("models").join(model_id.replace('/', "--")).join("main");
        fs::create_dir_all(&revision_dir).expect("should create fixture cache dir");
        fs::write(revision_dir.join("tokenizer.json"), TOKENIZER_JSON_FIXTURE)
            .expect("should write fixture tokenizer.json");

        let result = AutoTokenizer::from_pretrained_with_revision(model_id, None);

        // Restore the environment and clean up the temp cache before asserting,
        // so a failed assertion never leaks the override or the fixture files.
        match previous_cache_env {
            Some(v) => std::env::set_var("TRUSTFORMERS_CACHE", v),
            None => std::env::remove_var("TRUSTFORMERS_CACHE"),
        }
        fs::remove_dir_all(&temp_cache).ok();

        let tokenizer = result.expect(
            "from_pretrained_with_revision should succeed from the cache-hit path \
             without any network access",
        );
        assert!(
            matches!(tokenizer, AutoTokenizer::HuggingFace(_)),
            "cache-hit tokenizer.json should load as the HuggingFace variant"
        );
        assert_eq!(
            tokenizer.vocab_size(),
            4,
            "loaded tokenizer should reflect the 4-entry fixture vocab, proving it was \
             read from the pre-populated Hub cache rather than a fallback/mock path"
        );
    }
}
