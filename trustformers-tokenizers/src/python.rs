use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, PyResult};
use std::collections::HashMap;
use std::path::Path;

use crate::bpe::BPETokenizer;
use crate::char::CharTokenizer;
use crate::tokenizer::{TokenizedInputWithOffsets, TokenizerImpl};
use crate::unigram::UnigramTokenizer;
use crate::wordpiece::WordPieceTokenizer;
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Python wrapper for TokenizedInput
#[pyclass]
#[derive(Clone)]
pub struct PyTokenizedInput {
    #[pyo3(get)]
    pub input_ids: Vec<u32>,
    #[pyo3(get)]
    pub attention_mask: Vec<u8>,
    #[pyo3(get)]
    pub token_type_ids: Option<Vec<u32>>,
    #[pyo3(get)]
    pub offset_mapping: Option<Vec<(usize, usize)>>,
    #[pyo3(get)]
    pub special_tokens_mask: Option<Vec<u8>>,
}

#[pymethods]
impl PyTokenizedInput {
    #[new]
    fn new(
        input_ids: Vec<u32>,
        attention_mask: Vec<u8>,
        token_type_ids: Option<Vec<u32>>,
        offset_mapping: Option<Vec<(usize, usize)>>,
        special_tokens_mask: Option<Vec<u8>>,
    ) -> Self {
        Self {
            input_ids,
            attention_mask,
            token_type_ids,
            offset_mapping,
            special_tokens_mask,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTokenizedInput(input_ids={:?}, attention_mask={:?}, token_type_ids={:?})",
            self.input_ids, self.attention_mask, self.token_type_ids
        )
    }

    fn __len__(&self) -> usize {
        self.input_ids.len()
    }
}

impl From<TokenizedInputWithOffsets> for PyTokenizedInput {
    fn from(input: TokenizedInputWithOffsets) -> Self {
        Self {
            input_ids: input.input_ids,
            attention_mask: input.attention_mask,
            token_type_ids: input.token_type_ids,
            offset_mapping: input.offset_mapping,
            special_tokens_mask: input.special_tokens_mask,
        }
    }
}

impl From<TokenizedInput> for PyTokenizedInput {
    fn from(input: TokenizedInput) -> Self {
        Self {
            input_ids: input.input_ids,
            attention_mask: input.attention_mask,
            token_type_ids: input.token_type_ids,
            offset_mapping: None,
            special_tokens_mask: None,
        }
    }
}

/// Python wrapper for the main tokenizer implementation
#[pyclass]
pub struct PyTokenizer {
    inner: TokenizerImpl,
}

#[pymethods]
impl PyTokenizer {
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let tokenizer = TokenizerImpl::from_file(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to load tokenizer: {}",
                e
            ))
        })?;
        Ok(Self { inner: tokenizer })
    }

    #[staticmethod]
    fn from_pretrained(name: &str) -> PyResult<Self> {
        let tokenizer = TokenizerImpl::from_pretrained(name).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to load pretrained tokenizer: {}",
                e
            ))
        })?;
        Ok(Self { inner: tokenizer })
    }

    fn encode(&self, text: &str) -> PyResult<PyTokenizedInput> {
        let result = self.inner.encode(text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Encoding failed: {}", e))
        })?;
        Ok(PyTokenizedInput::from(result))
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> PyResult<PyTokenizedInput> {
        let result = self.inner.encode_pair(text_a, text_b).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Pair encoding failed: {}",
                e
            ))
        })?;
        Ok(PyTokenizedInput::from(result))
    }

    fn decode(&self, token_ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&token_ids).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Decoding failed: {}", e))
        })
    }

    fn get_vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

/// Python wrapper for BPE tokenizer
#[pyclass]
pub struct PyBPETokenizer {
    inner: BPETokenizer,
}

#[pymethods]
impl PyBPETokenizer {
    #[new]
    fn new(vocab: HashMap<String, u32>, merges: Vec<(String, String)>) -> Self {
        let tokenizer = BPETokenizer::new(vocab, merges);
        Self { inner: tokenizer }
    }

    fn encode(&self, text: &str) -> PyResult<PyTokenizedInput> {
        let result = self.inner.encode(text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("BPE encoding failed: {}", e))
        })?;
        Ok(PyTokenizedInput::from(result))
    }

    fn decode(&self, token_ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&token_ids).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("BPE decoding failed: {}", e))
        })
    }

    fn get_vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

/// Python wrapper for WordPiece tokenizer
#[pyclass]
pub struct PyWordPieceTokenizer {
    inner: WordPieceTokenizer,
}

#[pymethods]
impl PyWordPieceTokenizer {
    #[new]
    fn new(vocab: HashMap<String, u32>, do_lower_case: Option<bool>) -> Self {
        let tokenizer = WordPieceTokenizer::new(vocab, do_lower_case.unwrap_or(true));
        Self { inner: tokenizer }
    }

    fn encode(&self, text: &str) -> PyResult<PyTokenizedInput> {
        let result = self.inner.encode(text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "WordPiece encoding failed: {}",
                e
            ))
        })?;
        Ok(PyTokenizedInput::from(result))
    }

    fn decode(&self, token_ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&token_ids).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "WordPiece decoding failed: {}",
                e
            ))
        })
    }

    fn get_vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

/// Python wrapper for Unigram tokenizer
#[pyclass]
pub struct PyUnigramTokenizer {
    inner: UnigramTokenizer,
}

#[pymethods]
impl PyUnigramTokenizer {
    #[new]
    fn new(vocab: Vec<(String, f64)>) -> PyResult<Self> {
        let mut vocab_map = HashMap::new();
        let mut scores_map = HashMap::new();

        for (i, (token, score)) in vocab.iter().enumerate() {
            vocab_map.insert(token.clone(), i as u32);
            scores_map.insert(token.clone(), *score as f32);
        }

        let tokenizer = UnigramTokenizer::new(vocab_map, scores_map).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Unigram tokenizer creation failed: {}",
                e
            ))
        })?;
        Ok(Self { inner: tokenizer })
    }

    fn encode(&self, text: &str) -> PyResult<PyTokenizedInput> {
        let result = self.inner.encode(text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Unigram encoding failed: {}",
                e
            ))
        })?;
        Ok(PyTokenizedInput::from(result))
    }

    fn decode(&self, token_ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&token_ids).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Unigram decoding failed: {}",
                e
            ))
        })
    }

    fn get_vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

/// Python wrapper for character-level tokenizer
#[pyclass]
pub struct PyCharTokenizer {
    inner: CharTokenizer,
}

#[pymethods]
impl PyCharTokenizer {
    #[new]
    fn new(vocab: Option<HashMap<String, u32>>) -> Self {
        let default_vocab = if let Some(v) = vocab {
            v
        } else {
            // Create a basic character vocabulary for common characters
            let mut v = HashMap::new();
            v.insert("[PAD]".to_string(), 0);
            v.insert("[UNK]".to_string(), 1);
            v.insert("[CLS]".to_string(), 2);
            v.insert("[SEP]".to_string(), 3);

            // Add ASCII characters
            for i in 32..127 {
                let ch = char::from(i);
                v.insert(ch.to_string(), v.len() as u32);
            }
            v
        };

        let tokenizer = CharTokenizer::new(default_vocab);
        Self { inner: tokenizer }
    }

    fn encode(&self, text: &str) -> PyResult<PyTokenizedInput> {
        let result = self.inner.encode(text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Char encoding failed: {}",
                e
            ))
        })?;
        Ok(PyTokenizedInput::from(result))
    }

    fn decode(&self, token_ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&token_ids).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Char decoding failed: {}",
                e
            ))
        })
    }

    fn get_vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

/// Utility functions
#[pyfunction]
fn load_vocab_from_file(path: &str) -> PyResult<HashMap<String, u32>> {
    // Load vocabulary from a simple text file (one token per line)
    let content = std::fs::read_to_string(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read file: {}", e))
    })?;

    let mut vocab_map = HashMap::new();
    for (i, line) in content.lines().enumerate() {
        let token = line.trim();
        if !token.is_empty() {
            vocab_map.insert(token.to_string(), i as u32);
        }
    }
    Ok(vocab_map)
}

/// Load BPE merges from a file
fn load_merges_from_file(path: &str) -> PyResult<Vec<(String, String)>> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read merges file: {}", e))
    })?;

    let mut merges = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue; // Skip empty lines and comments
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            merges.push((parts[0].to_string(), parts[1].to_string()));
        }
    }

    Ok(merges)
}

#[pyfunction]
fn create_bpe_tokenizer(vocab_path: &str, merges_path: &str) -> PyResult<PyBPETokenizer> {
    // Load vocabulary
    let vocab = load_vocab_from_file(vocab_path)?;

    // Load merges from file
    let merges = load_merges_from_file(merges_path)?;

    Ok(PyBPETokenizer::new(vocab, merges))
}

#[pyfunction]
fn create_wordpiece_tokenizer(
    vocab_path: &str,
    do_lower_case: Option<bool>,
) -> PyResult<PyWordPieceTokenizer> {
    let vocab = load_vocab_from_file(vocab_path)?;
    Ok(PyWordPieceTokenizer::new(vocab, do_lower_case))
}

/// Training functionality
#[pyclass]
pub struct PyTokenizerTrainer {
    vocab_size: usize,
    special_tokens: Vec<String>,
}

#[pymethods]
impl PyTokenizerTrainer {
    #[new]
    fn new(vocab_size: usize, special_tokens: Option<Vec<String>>) -> Self {
        Self {
            vocab_size,
            special_tokens: special_tokens.unwrap_or_default(),
        }
    }

    fn train_from_files(&self, files: Vec<String>) -> PyResult<PyBPETokenizer> {
        // Use the actual BPE training infrastructure
        use crate::training::{BPETrainer, TrainingConfig};

        let config = TrainingConfig {
            vocab_size: self.vocab_size,
            special_tokens: self.special_tokens.clone(),
            ..Default::default()
        };

        let trainer = BPETrainer::new(config);

        // Read all files
        let mut texts = Vec::new();
        for file_path in &files {
            match std::fs::read_to_string(file_path) {
                Ok(content) => {
                    texts.push(content);
                },
                Err(e) => {
                    return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to read file {}: {}",
                        file_path, e
                    )));
                },
            }
        }

        let tokenizer = trainer.train(&texts).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Training failed: {}", e))
        })?;

        let vocab = tokenizer.get_vocab_map().clone();
        let merges = tokenizer.get_merge_rules().clone();

        Ok(PyBPETokenizer::new(vocab, merges))
    }

    fn train_from_iterator(&self, py: Python, iterator: Py<PyAny>) -> PyResult<PyBPETokenizer> {
        // Convert Python iterator to Rust iterator
        let texts: Vec<String> = iterator.extract(py)?;

        // Use the actual BPE training infrastructure
        use crate::training::{BPETrainer, TrainingConfig};

        let config = TrainingConfig {
            vocab_size: self.vocab_size,
            special_tokens: self.special_tokens.clone(),
            ..Default::default()
        };

        let trainer = BPETrainer::new(config);

        let tokenizer = trainer.train(&texts).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Training failed: {}", e))
        })?;

        let vocab = tokenizer.get_vocab_map().clone();
        let merges = tokenizer.get_merge_rules().clone();

        Ok(PyBPETokenizer::new(vocab, merges))
    }
}

/// Performance and analysis tools
#[pyfunction]
fn analyze_text_coverage(tokenizer: &PyTokenizer, texts: Vec<String>) -> PyResult<f64> {
    let mut total_chars = 0;
    let mut covered_chars = 0;

    for text in texts {
        total_chars += text.len();
        let encoded = tokenizer.encode(&text)?;
        let decoded = tokenizer.decode(encoded.input_ids)?;
        covered_chars += decoded.len();
    }

    Ok(covered_chars as f64 / total_chars as f64)
}

#[pyfunction]
fn benchmark_tokenizer(
    tokenizer: &PyTokenizer,
    texts: Vec<String>,
    iterations: usize,
) -> PyResult<f64> {
    use std::time::Instant;

    let start = Instant::now();
    for _ in 0..iterations {
        for text in &texts {
            let _ = tokenizer.encode(text)?;
        }
    }
    let duration = start.elapsed();

    let total_ops = texts.len() * iterations;
    Ok(total_ops as f64 / duration.as_secs_f64())
}

/// Python module definition
#[pymodule]
pub fn trustformers_tokenizers(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizedInput>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyBPETokenizer>()?;
    m.add_class::<PyWordPieceTokenizer>()?;
    m.add_class::<PyUnigramTokenizer>()?;
    m.add_class::<PyCharTokenizer>()?;
    m.add_class::<PyTokenizerTrainer>()?;

    m.add_function(wrap_pyfunction!(load_vocab_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(create_bpe_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(create_wordpiece_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_text_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_tokenizer, m)?)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "TrustformeRS Team")?;
    m.add(
        "__doc__",
        "High-performance tokenizers for transformer models",
    )?;

    Ok(())
}
