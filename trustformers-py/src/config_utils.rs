use pyo3::prelude::*;
use pyo3::types::PyDict;
use trustformers_models::{bert::BertConfig, gpt2::Gpt2Config, llama::LlamaConfig, t5::T5Config};

/// Parse GPT-2 config from Python dict
pub fn parse_gpt2_config(config_dict: &Bound<'_, PyAny>) -> PyResult<Gpt2Config> {
    let dict = config_dict.downcast::<PyDict>()?;
    let mut config = Gpt2Config::default();

    if let Ok(Some(val)) = dict.get_item("vocab_size") {
        config.vocab_size = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("n_embd") {
        config.n_embd = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("n_layer") {
        config.n_layer = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("n_head") {
        config.n_head = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("n_positions") {
        config.n_positions = val.extract()?;
    }

    Ok(config)
}

/// Convert GPT-2 config to Python dict
pub fn gpt2_config_to_dict<'py>(
    py: Python<'py>,
    config: &Gpt2Config,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("model_type", "gpt2")?;
    dict.set_item("vocab_size", config.vocab_size)?;
    dict.set_item("n_embd", config.n_embd)?;
    dict.set_item("n_layer", config.n_layer)?;
    dict.set_item("n_head", config.n_head)?;
    dict.set_item("n_positions", config.n_positions)?;
    dict.set_item("n_inner", config.n_inner.unwrap_or(config.n_embd * 4))?;
    dict.set_item("resid_pdrop", config.resid_pdrop)?;
    dict.set_item("embd_pdrop", config.embd_pdrop)?;
    dict.set_item("attn_pdrop", config.attn_pdrop)?;
    dict.set_item("layer_norm_epsilon", config.layer_norm_epsilon)?;
    Ok(dict)
}

/// Parse T5 config from Python dict
pub fn parse_t5_config(config_dict: &Bound<'_, PyAny>) -> PyResult<T5Config> {
    let dict = config_dict.downcast::<PyDict>()?;
    let mut config = T5Config::default();

    if let Ok(Some(val)) = dict.get_item("vocab_size") {
        config.vocab_size = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("d_model") {
        config.d_model = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("d_ff") {
        config.d_ff = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("num_layers") {
        config.num_layers = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("num_heads") {
        config.num_heads = val.extract()?;
    }

    Ok(config)
}

/// Convert T5 config to Python dict
pub fn t5_config_to_dict<'py>(py: Python<'py>, config: &T5Config) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("model_type", "t5")?;
    dict.set_item("vocab_size", config.vocab_size)?;
    dict.set_item("d_model", config.d_model)?;
    dict.set_item("d_ff", config.d_ff)?;
    dict.set_item("num_layers", config.num_layers)?;
    dict.set_item("num_decoder_layers", config.num_decoder_layers)?;
    dict.set_item("num_heads", config.num_heads)?;
    dict.set_item(
        "relative_attention_num_buckets",
        config.relative_attention_num_buckets,
    )?;
    dict.set_item(
        "relative_attention_max_distance",
        config.relative_attention_max_distance,
    )?;
    dict.set_item("dropout_rate", config.dropout_rate)?;
    dict.set_item("layer_norm_epsilon", config.layer_norm_epsilon)?;
    dict.set_item("initializer_factor", config.initializer_factor)?;
    Ok(dict)
}

/// Parse LLaMA config from Python dict
pub fn parse_llama_config(config_dict: &Bound<'_, PyAny>) -> PyResult<LlamaConfig> {
    let dict = config_dict.downcast::<PyDict>()?;
    let mut config = LlamaConfig::default();

    if let Ok(Some(val)) = dict.get_item("vocab_size") {
        config.vocab_size = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("hidden_size") {
        config.hidden_size = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("intermediate_size") {
        config.intermediate_size = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("num_hidden_layers") {
        config.num_hidden_layers = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("num_attention_heads") {
        config.num_attention_heads = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("num_key_value_heads") {
        config.num_key_value_heads = Some(val.extract()?);
    }

    Ok(config)
}

/// Convert LLaMA config to Python dict
pub fn llama_config_to_dict<'py>(
    py: Python<'py>,
    config: &LlamaConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("model_type", "llama")?;
    dict.set_item("vocab_size", config.vocab_size)?;
    dict.set_item("hidden_size", config.hidden_size)?;
    dict.set_item("intermediate_size", config.intermediate_size)?;
    dict.set_item("num_hidden_layers", config.num_hidden_layers)?;
    dict.set_item("num_attention_heads", config.num_attention_heads)?;
    if let Some(num_kv_heads) = config.num_key_value_heads {
        dict.set_item("num_key_value_heads", num_kv_heads)?;
    }
    dict.set_item("hidden_act", &config.hidden_act)?;
    dict.set_item("max_position_embeddings", config.max_position_embeddings)?;
    dict.set_item("rms_norm_eps", config.rms_norm_eps)?;
    dict.set_item("use_cache", config.use_cache)?;
    dict.set_item("pad_token_id", config.pad_token_id)?;
    dict.set_item("bos_token_id", config.bos_token_id)?;
    dict.set_item("eos_token_id", config.eos_token_id)?;
    Ok(dict)
}
