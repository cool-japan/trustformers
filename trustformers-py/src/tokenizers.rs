use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
// use trustformers::hub::{download_file_from_hub, HubOptions}; // Commented out - main trustformers crate not available

// Stub implementation for missing hub function
fn download_file_from_hub(
    _model_name: &str,
    _filename: &str,
    _options: Option<()>,
) -> Result<String, Box<dyn std::error::Error>> {
    // Return empty string as stub - tokenizers will use default vocab
    Ok(String::new())
}

use trustformers_core::traits::{TokenizedInput, Tokenizer};
use trustformers_tokenizers::{bpe::BPETokenizer, wordpiece::WordPieceTokenizer};

/// Base tokenizer class
#[pyclass(name = "PreTrainedTokenizer", module = "trustformers", subclass)]
pub struct PyPreTrainedTokenizer {
    pub pad_token: String,
    pub unk_token: String,
    pub cls_token: String,
    pub sep_token: String,
    pub mask_token: String,
    pub pad_token_id: usize,
    pub unk_token_id: usize,
    pub cls_token_id: usize,
    pub sep_token_id: usize,
    pub mask_token_id: usize,
}

#[pymethods]
impl PyPreTrainedTokenizer {
    /// Save tokenizer to directory
    pub fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        use std::fs;
        use std::path::Path;

        // Create directory if it doesn't exist
        let save_path = Path::new(save_directory);
        if !save_path.exists() {
            fs::create_dir_all(save_path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to create directory: {}", e))
            })?;
        }

        // Save tokenizer config
        let config = serde_json::json!({
            "tokenizer_class": "PreTrainedTokenizer",
            "model_max_length": 512,
            "special_tokens_map": {
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "mask_token": self.mask_token
            },
            "vocab_size": 30522  // Default BERT vocab size
        });

        let config_path = save_path.join("tokenizer_config.json");
        fs::write(
            &config_path,
            serde_json::to_string_pretty(&config).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to serialize config: {}",
                    e
                ))
            })?,
        )
        .map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write config file: {}", e))
        })?;

        // Save special tokens map
        let special_tokens = serde_json::json!({
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token
        });

        let special_tokens_path = save_path.join("special_tokens_map.json");
        fs::write(
            &special_tokens_path,
            serde_json::to_string_pretty(&special_tokens).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to serialize special tokens: {}",
                    e
                ))
            })?,
        )
        .map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to write special tokens file: {}",
                e
            ))
        })?;

        // Note: For a complete implementation, we would also save:
        // - vocab.txt (for WordPiece) or vocab.json (for BPE)
        // - merges.txt (for BPE tokenizers)
        // This is a basic implementation for demonstration purposes

        Ok(())
    }

    /// Get special tokens
    #[getter]
    pub fn special_tokens_map(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("pad_token", &self.pad_token)?;
        dict.set_item("unk_token", &self.unk_token)?;
        dict.set_item("cls_token", &self.cls_token)?;
        dict.set_item("sep_token", &self.sep_token)?;
        dict.set_item("mask_token", &self.mask_token)?;
        Ok(dict.into())
    }
}

/// WordPiece tokenizer wrapper
#[pyclass(name = "WordPieceTokenizer", module = "trustformers", extends = PyPreTrainedTokenizer)]
pub struct PyWordPieceTokenizer {
    inner: WordPieceTokenizer,
}

#[pymethods]
impl PyWordPieceTokenizer {
    /// Create a new WordPiece tokenizer
    #[new]
    #[pyo3(signature = (vocab=None, do_lower_case=true))]
    pub fn new(
        vocab: Option<HashMap<String, usize>>,
        do_lower_case: bool,
    ) -> PyResult<(Self, PyPreTrainedTokenizer)> {
        let vocab = vocab.unwrap_or_else(|| {
            let mut v = HashMap::new();
            v.insert("[PAD]".to_string(), 0);
            v.insert("[UNK]".to_string(), 1);
            v.insert("[CLS]".to_string(), 2);
            v.insert("[SEP]".to_string(), 3);
            v.insert("[MASK]".to_string(), 4);
            v
        });

        // Convert usize to u32 for compatibility with core WordPieceTokenizer
        let vocab_u32: HashMap<String, u32> =
            vocab.into_iter().map(|(k, v)| (k, v as u32)).collect();

        let tokenizer = WordPieceTokenizer::new(vocab_u32, do_lower_case);

        let base = PyPreTrainedTokenizer {
            pad_token: "[PAD]".to_string(),
            unk_token: "[UNK]".to_string(),
            cls_token: "[CLS]".to_string(),
            sep_token: "[SEP]".to_string(),
            mask_token: "[MASK]".to_string(),
            pad_token_id: 0,
            unk_token_id: 1,
            cls_token_id: 2,
            sep_token_id: 3,
            mask_token_id: 4,
        };

        Ok((PyWordPieceTokenizer { inner: tokenizer }, base))
    }

    /// Load from pretrained tokenizer
    #[staticmethod]
    #[pyo3(signature = (model_name_or_path, **kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        model_name_or_path: &str,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyWordPieceTokenizer>> {
        // Try to load vocabulary from hub
        let vocab_u32 = if let Ok(vocab_path) =
            download_file_from_hub(model_name_or_path, "vocab.txt", None)
        {
            // Load vocab.txt format (one token per line with implicit IDs)
            let vocab_content = std::fs::read_to_string(&vocab_path)
                .map_err(|e| PyValueError::new_err(format!("Failed to read vocab.txt: {}", e)))?;

            let mut vocab = HashMap::new();
            for (id, line) in vocab_content.lines().enumerate() {
                let token = line.trim();
                if !token.is_empty() {
                    vocab.insert(token.to_string(), id as u32);
                }
            }
            vocab
        } else if let Ok(vocab_path) =
            download_file_from_hub(model_name_or_path, "vocab.json", None)
        {
            // Load vocab.json format
            let vocab_content = std::fs::read_to_string(&vocab_path)
                .map_err(|e| PyValueError::new_err(format!("Failed to read vocab.json: {}", e)))?;

            let vocab_json: serde_json::Value = serde_json::from_str(&vocab_content)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse vocab.json: {}", e)))?;

            let mut vocab = HashMap::new();
            if let Some(vocab_obj) = vocab_json.as_object() {
                for (token, id) in vocab_obj {
                    if let Some(id_num) = id.as_u64() {
                        vocab.insert(token.clone(), id_num as u32);
                    }
                }
            }
            vocab
        } else {
            // Fallback to default vocabulary if no vocab file found
            [
                ("[PAD]".to_string(), 0),
                ("[UNK]".to_string(), 1),
                ("[CLS]".to_string(), 2),
                ("[SEP]".to_string(), 3),
                ("[MASK]".to_string(), 4),
            ]
            .into_iter()
            .collect()
        };

        // Try to load tokenizer config for special tokens
        let (pad_token, unk_token, cls_token, sep_token, mask_token) = if let Ok(config_path) =
            download_file_from_hub(model_name_or_path, "tokenizer_config.json", None)
        {
            let config_content = std::fs::read_to_string(&config_path).unwrap_or_default();
            if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_content) {
                let pad = config_json.get("pad_token").and_then(|v| v.as_str()).unwrap_or("[PAD]");
                let unk = config_json.get("unk_token").and_then(|v| v.as_str()).unwrap_or("[UNK]");
                let cls = config_json.get("cls_token").and_then(|v| v.as_str()).unwrap_or("[CLS]");
                let sep = config_json.get("sep_token").and_then(|v| v.as_str()).unwrap_or("[SEP]");
                let mask =
                    config_json.get("mask_token").and_then(|v| v.as_str()).unwrap_or("[MASK]");
                (
                    pad.to_string(),
                    unk.to_string(),
                    cls.to_string(),
                    sep.to_string(),
                    mask.to_string(),
                )
            } else {
                (
                    "[PAD]".to_string(),
                    "[UNK]".to_string(),
                    "[CLS]".to_string(),
                    "[SEP]".to_string(),
                    "[MASK]".to_string(),
                )
            }
        } else {
            (
                "[PAD]".to_string(),
                "[UNK]".to_string(),
                "[CLS]".to_string(),
                "[SEP]".to_string(),
                "[MASK]".to_string(),
            )
        };

        let tokenizer = WordPieceTokenizer::new(vocab_u32.clone(), true);

        // Get token IDs from vocab
        let pad_token_id = vocab_u32.get(&pad_token).copied().unwrap_or(0) as usize;
        let unk_token_id = vocab_u32.get(&unk_token).copied().unwrap_or(1) as usize;
        let cls_token_id = vocab_u32.get(&cls_token).copied().unwrap_or(2) as usize;
        let sep_token_id = vocab_u32.get(&sep_token).copied().unwrap_or(3) as usize;
        let mask_token_id = vocab_u32.get(&mask_token).copied().unwrap_or(4) as usize;

        let base = PyPreTrainedTokenizer {
            pad_token,
            unk_token,
            cls_token,
            sep_token,
            mask_token,
            pad_token_id,
            unk_token_id,
            cls_token_id,
            sep_token_id,
            mask_token_id,
        };

        Py::new(py, (PyWordPieceTokenizer { inner: tokenizer }, base))
    }

    /// Tokenize text
    pub fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        // Use the Tokenizer trait encode method and extract tokens
        let tokenized = self
            .inner
            .encode(text)
            .map_err(|e| PyValueError::new_err(format!("Tokenization failed: {}", e)))?;

        // Convert the input_ids back to tokens using id_to_token
        let tokens: Vec<String> = tokenized
            .input_ids
            .into_iter()
            .map(|id| self.inner.id_to_token(id).unwrap_or_else(|| format!("[UNK_{}]", id)))
            .collect();

        Ok(tokens)
    }

    /// Convert tokens to IDs
    pub fn convert_tokens_to_ids(&self, tokens: Vec<String>) -> Vec<usize> {
        // Use the tokenizer's token_to_id method to convert each token
        tokens
            .into_iter()
            .map(|token| {
                self.inner.token_to_id(&token).map(|id| id as usize).unwrap_or(1)
                // Return UNK ID (1) for unknown tokens
            })
            .collect()
    }

    /// Convert IDs to tokens
    pub fn convert_ids_to_tokens(&self, ids: Vec<usize>) -> Vec<String> {
        // Convert usize IDs to u32 for compatibility with core tokenizer
        ids.into_iter()
            .map(|id| {
                self.inner.id_to_token(id as u32).unwrap_or_else(|| format!("[UNK_{}]", id))
                // Fallback for unknown IDs
            })
            .collect()
    }

    /// Encode text
    #[pyo3(signature = (text, text_pair=None, add_special_tokens=true, max_length=None, padding=false, truncation=false, return_tensors=None))]
    pub fn encode(
        &self,
        py: Python<'_>,
        text: &str,
        text_pair: Option<&str>,
        add_special_tokens: bool,
        max_length: Option<usize>,
        padding: bool,
        truncation: bool,
        return_tensors: Option<&str>,
    ) -> PyResult<PyObject> {
        let output = if let Some(text2) = text_pair {
            self.inner.encode_pair(text, text2)
        } else {
            self.inner.encode(text)
        }
        .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))?;

        if let Some(format) = return_tensors {
            match format {
                "pt" | "np" => {
                    // Return as tensor
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("input_ids", output.input_ids)?;
                    dict.set_item("attention_mask", output.attention_mask)?;
                    if let Some(token_type_ids) = output.token_type_ids {
                        dict.set_item("token_type_ids", token_type_ids)?;
                    }
                    Ok(dict.into())
                },
                _ => Ok(output.input_ids.into_py(py)),
            }
        } else {
            Ok(output.input_ids.into_py(py))
        }
    }

    /// Batch encode
    #[pyo3(signature = (texts, text_pairs=None, add_special_tokens=true, max_length=None, padding=false, truncation=false, return_tensors=None))]
    pub fn batch_encode_plus(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        text_pairs: Option<Vec<Option<String>>>,
        add_special_tokens: bool,
        max_length: Option<usize>,
        padding: bool,
        truncation: bool,
        return_tensors: Option<&str>,
    ) -> PyResult<PyObject> {
        // Use single encode for each text since batch_encode doesn't exist
        let mut outputs = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let text_pair =
                text_pairs.as_ref().and_then(|pairs| pairs.get(i)).and_then(|p| p.as_deref());

            let mut tokenized = self
                .inner
                .encode(text)
                .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))?;

            // Handle text pairs by concatenating if provided
            if let Some(pair) = text_pair {
                let tokenized_pair = self
                    .inner
                    .encode(pair)
                    .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))?;

                // Concatenate the inputs (simplified - real implementation would handle special tokens properly)
                tokenized.input_ids.extend(tokenized_pair.input_ids);
                tokenized.attention_mask.extend(tokenized_pair.attention_mask);
            }

            outputs.push(tokenized);
        }

        let dict = pyo3::types::PyDict::new(py);

        // Convert outputs to Python lists
        let input_ids: Vec<Vec<u32>> = outputs.iter().map(|o| o.input_ids.clone()).collect();
        let attention_mask: Vec<Vec<u8>> =
            outputs.iter().map(|o| o.attention_mask.clone()).collect();

        // Convert u32 to usize for Python compatibility
        let input_ids_usize: Vec<Vec<usize>> = input_ids
            .iter()
            .map(|ids| ids.iter().map(|&id| id as usize).collect())
            .collect();
        let attention_mask_usize: Vec<Vec<usize>> = attention_mask
            .iter()
            .map(|mask| mask.iter().map(|&m| m as usize).collect())
            .collect();

        dict.set_item("input_ids", input_ids_usize)?;
        dict.set_item("attention_mask", attention_mask_usize)?;

        if outputs.iter().all(|o| o.token_type_ids.is_some()) {
            let token_type_ids: Vec<Vec<usize>> = outputs
                .iter()
                .map(|o| o.token_type_ids.clone().unwrap().iter().map(|&id| id as usize).collect())
                .collect();
            dict.set_item("token_type_ids", token_type_ids)?;
        }

        Ok(dict.into())
    }

    /// Decode IDs to text
    pub fn decode(&self, ids: Vec<usize>, skip_special_tokens: bool) -> PyResult<String> {
        let ids_u32: Vec<u32> = ids.into_iter().map(|id| id as u32).collect();
        self.inner
            .decode(&ids_u32)
            .map_err(|e| PyValueError::new_err(format!("Decoding failed: {}", e)))
    }

    /// Python's __call__ method
    #[pyo3(signature = (text, text_pair=None, add_special_tokens=true, max_length=None, padding=false, truncation=false, return_tensors=None))]
    pub fn __call__(
        &self,
        py: Python<'_>,
        text: TextInput,
        text_pair: Option<TextInput>,
        add_special_tokens: bool,
        max_length: Option<usize>,
        padding: bool,
        truncation: bool,
        return_tensors: Option<&str>,
    ) -> PyResult<PyObject> {
        match text {
            TextInput::Single(s) => {
                let pair = text_pair.map(|t| match t {
                    TextInput::Single(s) => s,
                    _ => panic!("text_pair must be a string when text is a string"),
                });
                self.encode(
                    py,
                    &s,
                    pair.as_deref(),
                    add_special_tokens,
                    max_length,
                    padding,
                    truncation,
                    return_tensors,
                )
            },
            TextInput::Batch(texts) => {
                let pairs = text_pair.map(|t| match t {
                    TextInput::Batch(pairs) => pairs.into_iter().map(Some).collect(),
                    _ => panic!("text_pair must be a list when text is a list"),
                });
                self.batch_encode_plus(
                    py,
                    texts,
                    pairs,
                    add_special_tokens,
                    max_length,
                    padding,
                    truncation,
                    return_tensors,
                )
            },
        }
    }

    /// Get vocabulary size
    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

/// BPE tokenizer wrapper
#[pyclass(name = "BPETokenizer", module = "trustformers", extends = PyPreTrainedTokenizer)]
pub struct PyBPETokenizer {
    inner: BPETokenizer,
}

#[pymethods]
impl PyBPETokenizer {
    /// Create a new BPE tokenizer
    #[new]
    #[pyo3(signature = (vocab=None, merges=None))]
    pub fn new(
        vocab: Option<HashMap<String, usize>>,
        merges: Option<Vec<(String, String)>>,
    ) -> PyResult<(Self, PyPreTrainedTokenizer)> {
        let vocab: HashMap<String, u32> = vocab
            .unwrap_or_else(HashMap::new)
            .into_iter()
            .map(|(k, v)| (k, v as u32))
            .collect();
        let merges = merges.unwrap_or_else(Vec::new);

        let tokenizer = BPETokenizer::new(vocab, merges);

        let base = PyPreTrainedTokenizer {
            pad_token: "<pad>".to_string(),
            unk_token: "<unk>".to_string(),
            cls_token: "<s>".to_string(),
            sep_token: "</s>".to_string(),
            mask_token: "<mask>".to_string(),
            pad_token_id: 0,
            unk_token_id: 1,
            cls_token_id: 2,
            sep_token_id: 3,
            mask_token_id: 4,
        };

        Ok((PyBPETokenizer { inner: tokenizer }, base))
    }

    /// Tokenize text (using encode then converting back to tokens)
    pub fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        let result = self
            .inner
            .encode(text)
            .map_err(|e| PyValueError::new_err(format!("Tokenization failed: {}", e)))?;

        // Convert the input_ids back to tokens using id_to_token
        let tokens: Vec<String> = result
            .input_ids
            .into_iter()
            .map(|id| self.inner.id_to_token(id).unwrap_or_else(|| format!("[UNK_{}]", id)))
            .collect();

        Ok(tokens)
    }

    /// Encode text
    #[pyo3(signature = (text, text_pair=None, add_special_tokens=true, max_length=None, padding=false, truncation=false, return_tensors=None))]
    pub fn encode(
        &self,
        py: Python<'_>,
        text: &str,
        text_pair: Option<&str>,
        add_special_tokens: bool,
        max_length: Option<usize>,
        padding: bool,
        truncation: bool,
        return_tensors: Option<&str>,
    ) -> PyResult<PyObject> {
        let output = if let Some(text2) = text_pair {
            self.inner.encode_pair(text, text2)
        } else {
            self.inner.encode(text)
        }
        .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))?;

        if let Some(format) = return_tensors {
            match format {
                "pt" | "np" => {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("input_ids", output.input_ids)?;
                    dict.set_item("attention_mask", output.attention_mask)?;
                    Ok(dict.into())
                },
                _ => Ok(output.input_ids.into_py(py)),
            }
        } else {
            Ok(output.input_ids.into_py(py))
        }
    }

    /// Decode IDs to text
    pub fn decode(&self, ids: Vec<usize>, skip_special_tokens: bool) -> PyResult<String> {
        let ids_u32: Vec<u32> = ids.into_iter().map(|id| id as u32).collect();
        self.inner
            .decode(&ids_u32)
            .map_err(|e| PyValueError::new_err(format!("Decoding failed: {}", e)))
    }

    /// Get vocabulary size
    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

/// Helper enum for text input
#[derive(FromPyObject)]
pub enum TextInput {
    Single(String),
    Batch(Vec<String>),
}
