use crate::core::traits::Model;
use crate::error::{Result, TrustformersError};
use std::path::Path;
use trustformers_core::errors::TrustformersError as CoreTrustformersError;

#[derive(Clone)]
pub enum AutoModelForSequenceClassification {
    #[cfg(feature = "bert")]
    Bert(crate::models::bert::BertForSequenceClassification),
    #[cfg(feature = "roberta")]
    Roberta(crate::models::roberta::RobertaForSequenceClassification),
    #[cfg(feature = "albert")]
    Albert(crate::models::albert::AlbertForSequenceClassification),
}

impl AutoModelForSequenceClassification {
    pub fn from_config(config: crate::automodel::AutoConfig, num_labels: usize) -> Result<Self> {
        match config {
            #[cfg(feature = "bert")]
            crate::automodel::AutoConfig::Bert(bert_config) => {
                Ok(AutoModelForSequenceClassification::Bert(
                    crate::models::bert::BertForSequenceClassification::new(
                        bert_config,
                        num_labels,
                    )?,
                ))
            },
            #[cfg(feature = "roberta")]
            crate::automodel::AutoConfig::Roberta(roberta_config) => {
                Ok(AutoModelForSequenceClassification::Roberta(
                    crate::models::roberta::RobertaForSequenceClassification::new(
                        roberta_config,
                        num_labels,
                    )?,
                ))
            },
            #[cfg(feature = "albert")]
            crate::automodel::AutoConfig::Albert(albert_config) => {
                Ok(AutoModelForSequenceClassification::Albert(
                    crate::models::albert::AlbertForSequenceClassification::new(
                        albert_config,
                        num_labels,
                    )?,
                ))
            },
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error(
                    "Model type does not support sequence classification".into(),
                ),
            )),
        }
    }

    pub fn from_pretrained(model_name_or_path: &str, num_labels: usize) -> Result<Self> {
        let config = crate::automodel::AutoConfig::from_pretrained(model_name_or_path)?;
        let mut model = Self::from_config(config, num_labels)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
            match &mut model {
                #[cfg(feature = "bert")]
                AutoModelForSequenceClassification::Bert(bert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    bert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "roberta")]
                AutoModelForSequenceClassification::Roberta(roberta) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    roberta.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "albert")]
                AutoModelForSequenceClassification::Albert(albert) => {
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
pub enum AutoModelForTokenClassification {
    #[cfg(feature = "bert")]
    Bert(crate::models::bert::BertForTokenClassification),
    #[cfg(feature = "roberta")]
    Roberta(crate::models::roberta::RobertaForTokenClassification),
    #[cfg(feature = "albert")]
    Albert(crate::models::albert::AlbertForTokenClassification),
}

impl AutoModelForTokenClassification {
    pub fn from_config(config: crate::automodel::AutoConfig, num_labels: usize) -> Result<Self> {
        match config {
            #[cfg(feature = "bert")]
            crate::automodel::AutoConfig::Bert(bert_config) => {
                Ok(AutoModelForTokenClassification::Bert(
                    crate::models::bert::BertForTokenClassification::new(bert_config, num_labels)?,
                ))
            },
            #[cfg(feature = "roberta")]
            crate::automodel::AutoConfig::Roberta(roberta_config) => {
                Ok(AutoModelForTokenClassification::Roberta(
                    crate::models::roberta::RobertaForTokenClassification::new(
                        roberta_config,
                        num_labels,
                    )?,
                ))
            },
            #[cfg(feature = "albert")]
            crate::automodel::AutoConfig::Albert(albert_config) => {
                Ok(AutoModelForTokenClassification::Albert(
                    crate::models::albert::AlbertForTokenClassification::new(
                        albert_config,
                        num_labels,
                    )?,
                ))
            },
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error(
                    "Model type does not support token classification".into(),
                ),
            )),
        }
    }

    #[cfg(feature = "bert")]
    pub fn from_pretrained(model_name_or_path: &str, num_labels: usize) -> Result<Self> {
        Self::from_pretrained_with_revision(model_name_or_path, num_labels, None)
    }

    #[cfg(feature = "bert")]
    pub fn from_pretrained_with_revision(
        model_name_or_path: &str,
        num_labels: usize,
        revision: Option<&str>,
    ) -> Result<Self> {
        let config = crate::automodel::AutoConfig::from_pretrained_with_revision(
            model_name_or_path,
            revision,
        )?;
        let mut model = Self::from_config(config, num_labels)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
            match &mut model {
                AutoModelForTokenClassification::Bert(bert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    bert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "roberta")]
                AutoModelForTokenClassification::Roberta(roberta) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    roberta.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "albert")]
                AutoModelForTokenClassification::Albert(albert) => {
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
pub enum AutoModelForQuestionAnswering {
    #[cfg(feature = "bert")]
    Bert(crate::models::bert::BertForQuestionAnswering),
    #[cfg(feature = "roberta")]
    Roberta(crate::models::roberta::RobertaForQuestionAnswering),
    #[cfg(feature = "albert")]
    Albert(crate::models::albert::AlbertForQuestionAnswering),
}

impl AutoModelForQuestionAnswering {
    pub fn from_config(config: crate::automodel::AutoConfig) -> Result<Self> {
        match config {
            #[cfg(feature = "bert")]
            crate::automodel::AutoConfig::Bert(bert_config) => {
                Ok(AutoModelForQuestionAnswering::Bert(
                    crate::models::bert::BertForQuestionAnswering::new(bert_config)?,
                ))
            },
            #[cfg(feature = "roberta")]
            crate::automodel::AutoConfig::Roberta(roberta_config) => {
                Ok(AutoModelForQuestionAnswering::Roberta(
                    crate::models::roberta::RobertaForQuestionAnswering::new(roberta_config)?,
                ))
            },
            #[cfg(feature = "albert")]
            crate::automodel::AutoConfig::Albert(albert_config) => {
                Ok(AutoModelForQuestionAnswering::Albert(
                    crate::models::albert::AlbertForQuestionAnswering::new(albert_config)?,
                ))
            },
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error(
                    "Model type does not support question answering".into(),
                ),
            )),
        }
    }

    #[cfg(feature = "bert")]
    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        Self::from_pretrained_with_revision(model_name_or_path, None)
    }

    #[cfg(feature = "bert")]
    pub fn from_pretrained_with_revision(
        model_name_or_path: &str,
        revision: Option<&str>,
    ) -> Result<Self> {
        let config = crate::automodel::AutoConfig::from_pretrained_with_revision(
            model_name_or_path,
            revision,
        )?;
        let mut model = Self::from_config(config)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
            match &mut model {
                AutoModelForQuestionAnswering::Bert(bert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    bert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "roberta")]
                AutoModelForQuestionAnswering::Roberta(roberta) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    roberta.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "albert")]
                AutoModelForQuestionAnswering::Albert(albert) => {
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
pub enum AutoModelForCausalLM {
    #[cfg(feature = "gpt2")]
    Gpt2(crate::models::gpt2::Gpt2LMHeadModel),
    #[cfg(feature = "gpt_neo")]
    GptNeo(crate::models::gpt_neo::GptNeoLMHeadModel),
    #[cfg(feature = "gpt_j")]
    GptJ(crate::models::gpt_j::GptJLMHeadModel),
}

impl AutoModelForCausalLM {
    pub fn from_config(config: crate::automodel::AutoConfig) -> Result<Self> {
        match config {
            #[cfg(feature = "gpt2")]
            crate::automodel::AutoConfig::Gpt2(gpt2_config) => Ok(AutoModelForCausalLM::Gpt2(
                crate::models::gpt2::Gpt2LMHeadModel::new(gpt2_config)?,
            )),
            #[cfg(feature = "gpt_neo")]
            crate::automodel::AutoConfig::GptNeo(gpt_neo_config) => {
                Ok(AutoModelForCausalLM::GptNeo(
                    crate::models::gpt_neo::GptNeoLMHeadModel::new(gpt_neo_config)?,
                ))
            },
            #[cfg(feature = "gpt_j")]
            crate::automodel::AutoConfig::GptJ(gpt_j_config) => Ok(AutoModelForCausalLM::GptJ(
                crate::models::gpt_j::GptJLMHeadModel::new(gpt_j_config)?,
            )),
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error(
                    "Model type does not support causal language modeling".into(),
                ),
            )),
        }
    }

    pub fn generate(
        &mut self,
        inputs: crate::core::traits::TokenizedInput,
        generation_config: crate::pipeline::text_generation::GenerationConfig,
    ) -> Result<Vec<u32>> {
        match self {
            #[cfg(feature = "gpt2")]
            AutoModelForCausalLM::Gpt2(gpt2) => gpt2
                .generate(
                    inputs.input_ids,
                    generation_config.max_length,
                    generation_config.temperature,
                    generation_config.top_k,
                    generation_config.top_p,
                )
                .map_err(Into::into),
            #[cfg(feature = "gpt_neo")]
            AutoModelForCausalLM::GptNeo(gpt_neo) => gpt_neo
                .generate(
                    inputs.input_ids,
                    generation_config.max_length,
                    generation_config.temperature,
                    generation_config.top_k,
                    generation_config.top_p,
                )
                .map_err(Into::into),
            #[cfg(feature = "gpt_j")]
            AutoModelForCausalLM::GptJ(gpt_j) => gpt_j
                .generate(
                    inputs.input_ids,
                    generation_config.max_length,
                    generation_config.temperature,
                    generation_config.top_k,
                    generation_config.top_p,
                )
                .map_err(Into::into),
            #[cfg(not(any(feature = "gpt2", feature = "gpt_neo", feature = "gpt_j")))]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error("No causal LM models available".into()),
            )),
        }
    }

    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        let config = crate::automodel::AutoConfig::from_pretrained(model_name_or_path)?;
        let mut model = Self::from_config(config)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
            match &mut model {
                #[cfg(feature = "gpt2")]
                AutoModelForCausalLM::Gpt2(gpt2) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    gpt2.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "gpt_neo")]
                AutoModelForCausalLM::GptNeo(gpt_neo) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    gpt_neo.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "gpt_j")]
                AutoModelForCausalLM::GptJ(gpt_j) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    gpt_j.load_pretrained(&mut reader)?;
                },
                #[cfg(not(any(feature = "gpt2", feature = "gpt_neo", feature = "gpt_j")))]
                _ => {},
            }
        }

        Ok(model)
    }
}

#[derive(Clone)]
pub enum AutoModelForSeq2SeqLM {
    #[cfg(feature = "t5")]
    T5(crate::models::t5::T5ForConditionalGeneration),
}

impl AutoModelForSeq2SeqLM {
    pub fn from_config(config: crate::automodel::AutoConfig) -> Result<Self> {
        match config {
            #[cfg(feature = "t5")]
            crate::automodel::AutoConfig::T5(t5_config) => Ok(AutoModelForSeq2SeqLM::T5(
                crate::models::t5::T5ForConditionalGeneration::new(t5_config)?,
            )),
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error(
                    "Model type does not support seq2seq language modeling".into(),
                ),
            )),
        }
    }

    #[cfg(feature = "t5")]
    pub fn generate(
        &mut self,
        inputs: crate::core::traits::TokenizedInput,
        generation_config: crate::pipeline::text_generation::GenerationConfig,
    ) -> Result<Vec<u32>> {
        match self {
            AutoModelForSeq2SeqLM::T5(t5) => t5
                .generate(
                    inputs.input_ids,
                    generation_config.max_length,
                    generation_config.num_beams,
                )
                .map_err(Into::into),
        }
    }

    #[cfg(feature = "t5")]
    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        let config = crate::automodel::AutoConfig::from_pretrained(model_name_or_path)?;
        let mut model = Self::from_config(config)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
            match &mut model {
                AutoModelForSeq2SeqLM::T5(t5) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    t5.load_pretrained(&mut reader)?;
                },
            }
        }

        Ok(model)
    }
}

#[derive(Clone)]
pub enum AutoModelForMaskedLM {
    #[cfg(feature = "bert")]
    Bert(crate::models::bert::BertForMaskedLM),
    #[cfg(feature = "roberta")]
    Roberta(crate::models::roberta::RobertaForMaskedLM),
    #[cfg(feature = "albert")]
    Albert(crate::models::albert::AlbertForMaskedLM),
}

impl AutoModelForMaskedLM {
    pub fn from_config(config: crate::automodel::AutoConfig) -> Result<Self> {
        match config {
            #[cfg(feature = "bert")]
            crate::automodel::AutoConfig::Bert(bert_config) => Ok(AutoModelForMaskedLM::Bert(
                crate::models::bert::BertForMaskedLM::new(bert_config)?,
            )),
            #[cfg(feature = "roberta")]
            crate::automodel::AutoConfig::Roberta(roberta_config) => {
                Ok(AutoModelForMaskedLM::Roberta(
                    crate::models::roberta::RobertaForMaskedLM::new(roberta_config)?,
                ))
            },
            #[cfg(feature = "albert")]
            crate::automodel::AutoConfig::Albert(albert_config) => {
                Ok(AutoModelForMaskedLM::Albert(
                    crate::models::albert::AlbertForMaskedLM::new(albert_config)?,
                ))
            },
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error(
                    "Model type does not support masked language modeling".into(),
                ),
            )),
        }
    }

    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        let config = crate::automodel::AutoConfig::from_pretrained(model_name_or_path)?;
        let mut model = Self::from_config(config)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
            match &mut model {
                #[cfg(feature = "bert")]
                AutoModelForMaskedLM::Bert(bert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    bert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "roberta")]
                AutoModelForMaskedLM::Roberta(roberta) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    roberta.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "albert")]
                AutoModelForMaskedLM::Albert(albert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    albert.load_pretrained(&mut reader)?;
                },
            }
        }

        Ok(model)
    }
}
