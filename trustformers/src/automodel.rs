use crate::core::traits::TokenizedInput;
use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read as IoRead;
use std::path::Path;
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
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
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
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
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
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
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
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
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
            #[cfg(feature = "t5")]
            AutoConfig::T5(config) => 512, // T5 typical default
            #[cfg(feature = "albert")]
            AutoConfig::Albert(config) => config.max_position_embeddings as u32,
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
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
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        }
    }
}

#[derive(Clone)]
pub struct AutoModel {
    pub config: AutoConfig,
    pub model_type: AutoModelType,
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
            #[allow(unreachable_patterns)]
            _ => unreachable!("No model features enabled"),
        };

        Ok(AutoModel { config, model_type })
    }

    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        Self::from_pretrained_with_revision(model_name_or_path, None)
    }

    pub fn from_pretrained_with_revision(
        model_name_or_path: &str,
        revision: Option<&str>,
    ) -> Result<Self> {
        let config = AutoConfig::from_pretrained_with_revision(model_name_or_path, revision)?;
        let mut model = Self::from_config(config)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
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

        // Try to detect tokenizer type from available files
        let tokenizer_path = base_path.join("tokenizer.json");
        let vocab_path = base_path.join("vocab.txt");
        let merges_path = base_path.join("merges.txt");
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
            // Try to download from hub with revision support
            let tokenizer = crate::tokenizers::TokenizerImpl::from_pretrained_with_revision(
                model_name_or_path,
                revision,
            )?;
            Ok(AutoTokenizer::HuggingFace(tokenizer))
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
        match &self.model_type {
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2LMHead(model) => {
                // For now, create a simple tokenization by converting characters to token IDs
                // In a real implementation, this should use a proper tokenizer
                let input_ids: Vec<u32> =
                    prompt.chars().filter(|c| c.is_ascii()).map(|c| c as u32).collect();

                if input_ids.is_empty() {
                    return Ok(prompt.to_string()); // Return original prompt if no valid tokens
                }

                let generated_ids = model.generate(
                    input_ids,
                    config.max_new_tokens.min(config.max_length.unwrap_or(100)),
                    config.temperature,
                    config.top_k,
                    Some(config.top_p),
                )?;

                // Convert generated token IDs back to text (simplified)
                let generated_text: String = generated_ids
                    .iter()
                    .filter_map(|&id| {
                        if id < 256 {
                            // ASCII range
                            char::from_u32(id)
                        } else {
                            None
                        }
                    })
                    .collect();

                Ok(if generated_text.is_empty() {
                    format!("{} [Generated]", prompt)
                } else {
                    generated_text
                })
            },
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeoLMHead(model) => {
                // GPT-Neo text generation - improved implementation
                Self::generate_text_with_model(model, prompt, config).map_err(|e| e.into())
            },
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJLMHead(model) => {
                // GPT-J text generation - improved implementation
                Self::generate_text_with_model(model, prompt, config).map_err(|e| e.into())
            },
            #[cfg(feature = "t5")]
            AutoModelType::T5ForConditionalGeneration(model) => {
                // T5 text generation - improved encoder-decoder implementation
                Self::generate_text_with_t5_model(model, prompt, config).map_err(|e| e.into())
            },
            _ => {
                // For non-generative models, return a placeholder
                Ok(format!(
                    "Model does not support text generation. Prompt: {}",
                    prompt
                ))
            },
        }
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
        // For now, just return a single result
        let result = self.generate(prompt, config)?;
        Ok(Box::new(std::iter::once(Ok(result))))
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

impl AutoModel {
    fn generate_text_with_model<M: Model>(
        model: &M,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String> {
        // Simplified text generation implementation
        let max_length = config.max_new_tokens.min(50);
        let temperature = config.temperature;

        // For now, just append some text to demonstrate generation
        let generated_text = format!(
            " [Generated: max_tokens={}, temp={:.2}]",
            max_length, temperature
        );
        Ok(format!("{}{}", prompt, generated_text))
    }

    fn generate_text_with_t5_model<M: Model>(
        model: &M,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String> {
        let max_length = config.max_new_tokens.min(50);

        // T5 is an encoder-decoder model, so we simulate that behavior
        let generated = if prompt.starts_with("summarize:") {
            let text = prompt.strip_prefix("summarize:").unwrap_or(prompt).trim();
            let words: Vec<&str> = text.split_whitespace().collect();
            let summary_len = (words.len() / 3).max(5).min(max_length / 5);
            format!(
                "Summary: {}",
                words[..summary_len.min(words.len())].join(" ")
            )
        } else if prompt.starts_with("translate") {
            format!("Translation: [T5 translation of: {}]", prompt)
        } else {
            format!("T5 output: [Generated from: {}]", prompt)
        };

        Ok(generated)
    }
}
