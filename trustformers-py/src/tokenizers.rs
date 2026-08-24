use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;

/// Write `files` (relative name -> contents) into `save_directory`, creating it
/// if needed.
fn write_tokenizer_files(save_directory: &str, files: &[(&str, String)]) -> PyResult<()> {
    let save_path = Path::new(save_directory);
    std::fs::create_dir_all(save_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "Failed to create {save_directory}: {e}"
        ))
    })?;
    for (name, contents) in files {
        let path = save_path.join(name);
        std::fs::write(&path, contents).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write {}: {e}", path.display()))
        })?;
    }
    Ok(())
}

/// Pretty-print a JSON value for a tokenizer asset file.
fn tokenizer_json(value: &serde_json::Value) -> PyResult<String> {
    serde_json::to_string_pretty(value)
        .map_err(|e| PyValueError::new_err(format!("Failed to serialize tokenizer file: {e}")))
}

/// Locate a tokenizer asset (`vocab.txt`, `vocab.json`, `tokenizer_config.json`)
/// belonging to a local model path.
///
/// This crate has no Hugging Face Hub downloader, so `model_name_or_path` must
/// already be local: a directory holding the asset, or a file (e.g. a
/// `config.json` or a checkpoint) sitting next to it. Returns `None` when the
/// asset is simply not there.
///
/// This replaces a `download_file_from_hub` stub that unconditionally returned
/// `Ok("")` -- the empty *path*, not an empty file. Every caller then did
/// `std::fs::read_to_string("")`, so `WordPieceTokenizer.from_pretrained` could
/// only ever fail, with "Failed to read vocab.txt: No such file or directory",
/// no matter how complete the local checkpoint directory was.
fn find_local_tokenizer_file(model_name_or_path: &str, filename: &str) -> Option<PathBuf> {
    let path = Path::new(model_name_or_path);
    let candidate = if path.is_dir() {
        path.join(filename)
    } else {
        path.parent()?.join(filename)
    };
    candidate.is_file().then_some(candidate)
}

/// Parse a WordPiece `vocab.txt`: one token per line, the id is the line index.
///
/// Blank lines still consume an id, exactly as HuggingFace's reader does --
/// skipping them would shift every subsequent token's id and silently
/// mis-tokenize the whole vocabulary.
fn parse_vocab_txt(content: &str) -> HashMap<String, u32> {
    content
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let token = line.trim_end_matches(['\r', '\n']);
            (!token.is_empty()).then(|| (token.to_string(), index as u32))
        })
        .collect()
}

/// Parse a `vocab.json` mapping of token -> id.
///
/// # Errors
///
/// Fails when the document is not JSON, is not an object, or holds an entry
/// whose value is not a vocabulary index -- all of which the previous
/// implementation dropped silently via `filter_map` / `unwrap_or_default`,
/// yielding a partial vocabulary that tokenizes to `[UNK]` without saying why.
fn parse_vocab_json(content: &str) -> Result<HashMap<String, u32>, String> {
    let value: serde_json::Value =
        serde_json::from_str(content).map_err(|e| format!("vocab.json is not valid JSON: {e}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "vocab.json must be a JSON object mapping tokens to ids".to_string())?;

    let mut vocab = HashMap::with_capacity(object.len());
    for (token, id) in object {
        let id = id
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
            .ok_or_else(|| format!("vocab.json entry '{token}' has a non-index value: {id}"))?;
        vocab.insert(token.clone(), id);
    }
    Ok(vocab)
}

/// The special-token names a `tokenizer_config.json` may override, with the
/// WordPiece defaults used when it does not.
fn special_tokens_from_config(config: Option<&serde_json::Value>) -> SpecialTokens {
    let read = |key: &str, fallback: &str| -> String {
        config
            .and_then(|config| config.get(key))
            .and_then(|value| value.as_str())
            .unwrap_or(fallback)
            .to_string()
    };
    SpecialTokens {
        pad: read("pad_token", "[PAD]"),
        unk: read("unk_token", "[UNK]"),
        cls: read("cls_token", "[CLS]"),
        sep: read("sep_token", "[SEP]"),
        mask: read("mask_token", "[MASK]"),
    }
}

/// Parse a BPE `merges.txt`: one space-separated symbol pair per line, in
/// merge-priority order.
///
/// The leading `#version:` comment HuggingFace writes is skipped; every other
/// line must be exactly two symbols, because merge *order* is the whole
/// content of the file -- silently dropping a malformed line would shift the
/// priority of every merge after it and change how the vocabulary tokenizes.
///
/// # Errors
///
/// Fails on any non-comment line that does not hold exactly two symbols.
fn parse_merges_txt(content: &str) -> Result<Vec<(String, String)>, String> {
    let mut merges = Vec::new();
    for (index, line) in content.lines().enumerate() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() || line.starts_with("#version:") {
            continue;
        }
        let mut symbols = line.split(' ');
        match (symbols.next(), symbols.next(), symbols.next()) {
            (Some(left), Some(right), None) if !left.is_empty() && !right.is_empty() => {
                merges.push((left.to_string(), right.to_string()));
            },
            _ => {
                return Err(format!(
                    "merges.txt line {} is not a pair of symbols: {line:?}",
                    index + 1
                ))
            },
        }
    }
    Ok(merges)
}

/// A token -> id vocabulary rendered as a `vocab.txt` body.
///
/// Ids are line numbers in this format, so an id with no token is written as a
/// blank line to keep every later token on its own id -- which is exactly what
/// [`parse_vocab_txt`] reads back, making the pair an exact round trip.
///
/// # Errors
///
/// Fails on an empty vocabulary, and on two tokens claiming the same id: the
/// format has one line per id, so one of them would have to be dropped
/// silently.
fn vocab_txt_body(vocab: &HashMap<String, u32>) -> Result<String, String> {
    if vocab.is_empty() {
        return Err("the tokenizer has an empty vocabulary".to_string());
    }
    let highest = vocab.values().copied().max().unwrap_or(0);
    let slots = usize::try_from(highest)
        .map_err(|_| format!("vocabulary id {highest} does not fit in this platform's usize"))?
        + 1;

    let mut by_id: Vec<Option<&str>> = vec![None; slots];
    for (token, &id) in vocab {
        let index = id as usize;
        if let Some(existing) = by_id[index] {
            return Err(format!(
                "tokens {existing:?} and {token:?} both claim id {id}; vocab.txt has one line per \
                 id, so one of them would be lost"
            ));
        }
        by_id[index] = Some(token);
    }

    let mut body = String::new();
    for token in by_id {
        body.push_str(token.unwrap_or(""));
        body.push('\n');
    }
    Ok(body)
}

/// Serialise a merge table back to the `merges.txt` format, header included.
fn merges_txt_body(merges: &[(String, String)]) -> String {
    let mut body = String::from("#version: 0.2\n");
    for (left, right) in merges {
        body.push_str(left);
        body.push(' ');
        body.push_str(right);
        body.push('\n');
    }
    body
}

/// The `special_tokens_map.json` body for a set of special tokens.
fn special_tokens_map_json(special: &SpecialTokens) -> serde_json::Value {
    serde_json::json!({
        "pad_token": special.pad,
        "unk_token": special.unk,
        "cls_token": special.cls,
        "sep_token": special.sep,
        "mask_token": special.mask,
    })
}

/// The five special tokens the Python tokenizer classes expose.
struct SpecialTokens {
    pad: String,
    unk: String,
    cls: String,
    sep: String,
    mask: String,
}

use std::path::{Path, PathBuf};

use trustformers_core::errors::TrustformersError;
use trustformers_core::traits::{TokenizedInput, Tokenizer};
use trustformers_tokenizers::{bpe::BPETokenizer, wordpiece::WordPieceTokenizer};

/// Tokenize one `(text, optional text_pair)` batch item, matching
/// `PyWordPieceTokenizer::encode`'s single-item behavior: `encode_pair` when a
/// pair is given (real `[CLS] A [SEP] B [SEP]` with 0/1 `token_type_ids`),
/// plain `encode` otherwise.
///
/// Extracted from `batch_encode_plus` so its per-item pair behavior is
/// unit-testable without a Python interpreter -- see `resolve_single_text_pair`
/// below for the same split.
///
/// This replaces `batch_encode_plus`'s previous per-item pair handling, which
/// called `encode(text)` and `encode(pair)` independently and concatenated
/// their `input_ids`/`attention_mask` -- carrying no `[SEP]` token between the
/// two segments, and never touching `token_type_ids` for the pair at all, so
/// the first call's `token_type_ids` (all zero, sized for `text` alone)
/// survived unmodified and shorter than the now-concatenated `input_ids`.
fn wordpiece_batch_item(
    tokenizer: &WordPieceTokenizer,
    text: &str,
    text_pair: Option<&str>,
) -> Result<TokenizedInput, TrustformersError> {
    match text_pair {
        Some(pair) => tokenizer.encode_pair(text, pair),
        None => tokenizer.encode(text),
    }
}

/// The `<bos> A <eos> <eos> B <eos>` pair-encoding convention this crate's
/// RoBERTa/GPT-2-family `BPETokenizer` uses for sequence pairs, matching
/// HuggingFace's `RobertaTokenizer.build_inputs_with_special_tokens`. BPE
/// models built this way do not use segment ids for pairs (matching
/// `RobertaTokenizer.create_token_type_ids_from_sequences`), so every
/// position gets `token_type_ids = 0` rather than a 0/1 split.
///
/// `bos_token`/`eos_token` are read from the tokenizer itself, so this
/// follows whichever preset is loaded: `<|endoftext|>` for the GPT-2 default
/// `BPETokenizer::new`, `<s>`/`</s>` for `BPETokenizer::from_roberta_files`.
///
/// # Offsets
///
/// `offset_mapping` is populated for real, built from the two independent
/// `BPETokenizer::encode` calls below -- each already returns real
/// per-sequence byte offsets (see that method's own doc). The four boundary
/// positions this function inserts (`bos`, the two `eos` separators, the
/// trailing `eos`) report `(0, 0)`, HuggingFace's convention for a special
/// token with no source span. The first sequence's spans index `text`; the
/// second sequence's index `text2` -- **not** a combined coordinate space --
/// matching `WordPieceTokenizer::encode_pair`'s per-sequence convention (see
/// `pipelines/span.rs`'s module doc for the byte-offset convention itself).
/// `special_tokens_mask` is populated the same way, flagging exactly those
/// four boundary positions, so a caller can tell a real token's `(0, 0)`
/// apart from a boundary token's (which cannot otherwise be distinguished
/// from the offsets alone).
///
/// This bypasses `BPETokenizer::encode_pair` entirely rather than calling it:
/// that method's own implementation is `format!("{} {}", text, text2)`,
/// re-encoded as a single string -- no separator token at all, its
/// `offset_mapping` indexes that joined string rather than either original
/// sequence, and its `token_type_ids` stays `None` -- which silently merges
/// the two sequences' boundary instead of marking it.
///
/// # Errors
///
/// Fails when either sequence fails to tokenize, when the tokenizer's
/// configured `bos_token`/`eos_token` is not in its own vocabulary (there is
/// then no id to place for it, and silently substituting `unk_token` would
/// fabricate a boundary token that was never actually there), or when an
/// `encode` call's `offset_mapping` does not have exactly one entry per
/// token. `BPETokenizer::encode` always keeps them in lockstep today, but
/// this is checked rather than assumed: a silent mismatch here would return
/// spans that look precise while no longer corresponding to the tokens they
/// claim to describe, which is exactly the kind of fabrication this crate's
/// honesty policy refuses to let through.
fn bpe_pair_encoding(
    tokenizer: &BPETokenizer,
    text: &str,
    text2: &str,
) -> Result<TokenizedInput, String> {
    let bos_id = tokenizer.token_to_id(tokenizer.bos_token()).ok_or_else(|| {
        format!(
            "the beginning-of-sequence token {:?} is not in this tokenizer's vocabulary",
            tokenizer.bos_token()
        )
    })?;
    let eos_id = tokenizer.token_to_id(tokenizer.eos_token()).ok_or_else(|| {
        format!(
            "the end-of-sequence token {:?} is not in this tokenizer's vocabulary",
            tokenizer.eos_token()
        )
    })?;

    let first = tokenizer.encode(text).map_err(|e| format!("Encoding failed: {e}"))?;
    let second = tokenizer.encode(text2).map_err(|e| format!("Encoding failed: {e}"))?;

    let first_offsets = first.offset_mapping.ok_or_else(|| {
        "BPETokenizer::encode did not return an offset mapping for the first sequence \
         (expected Some(..) unconditionally)"
            .to_string()
    })?;
    if first_offsets.len() != first.input_ids.len() {
        return Err(format!(
            "the first sequence's offset mapping has {} entries but {} tokens; refusing to \
             return spans that may not correspond to the right token",
            first_offsets.len(),
            first.input_ids.len()
        ));
    }
    let second_offsets = second.offset_mapping.ok_or_else(|| {
        "BPETokenizer::encode did not return an offset mapping for the second sequence \
         (expected Some(..) unconditionally)"
            .to_string()
    })?;
    if second_offsets.len() != second.input_ids.len() {
        return Err(format!(
            "the second sequence's offset mapping has {} entries but {} tokens; refusing to \
             return spans that may not correspond to the right token",
            second_offsets.len(),
            second.input_ids.len()
        ));
    }

    // HuggingFace's convention for a special token with no source span.
    const BOUNDARY_SPAN: (usize, usize) = (0, 0);

    let first_len = first.input_ids.len();
    let second_len = second.input_ids.len();
    let total_len = first_len + second_len + 4;

    let mut input_ids = Vec::with_capacity(total_len);
    let mut offset_mapping = Vec::with_capacity(total_len);
    let mut special_tokens_mask = Vec::with_capacity(total_len);

    input_ids.push(bos_id);
    offset_mapping.push(BOUNDARY_SPAN);
    special_tokens_mask.push(1u8);

    input_ids.extend(first.input_ids);
    offset_mapping.extend(first_offsets);
    special_tokens_mask.extend(std::iter::repeat_n(0u8, first_len));

    input_ids.push(eos_id);
    offset_mapping.push(BOUNDARY_SPAN);
    special_tokens_mask.push(1u8);
    input_ids.push(eos_id);
    offset_mapping.push(BOUNDARY_SPAN);
    special_tokens_mask.push(1u8);

    input_ids.extend(second.input_ids);
    offset_mapping.extend(second_offsets);
    special_tokens_mask.extend(std::iter::repeat_n(0u8, second_len));

    input_ids.push(eos_id);
    offset_mapping.push(BOUNDARY_SPAN);
    special_tokens_mask.push(1u8);

    let attention_mask = vec![1u8; input_ids.len()];
    let token_type_ids = vec![0u32; input_ids.len()];

    Ok(TokenizedInput {
        input_ids,
        attention_mask,
        token_type_ids: Some(token_type_ids),
        special_tokens_mask: Some(special_tokens_mask),
        offset_mapping: Some(offset_mapping),
        overflowing_tokens: None,
    })
}

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
    /// Save tokenizer to directory.
    ///
    /// `PreTrainedTokenizer` itself holds only the five special-token names and
    /// their ids, never a vocabulary -- every concrete tokenizer class
    /// (`WordPieceTokenizer`, `BPETokenizer`) overrides this with an
    /// implementation that also exports the vocabulary (and, for BPE, the merge
    /// table). Reaching this base implementation means there is no vocabulary to
    /// write, so it refuses instead of producing a directory that *looks* like a
    /// saved tokenizer.
    ///
    /// The previous implementation wrote a `tokenizer_config.json` containing
    /// `"tokenizer_class": "PreTrainedTokenizer"` and a hardcoded
    /// `"vocab_size": 30522` (BERT's, whatever the actual tokenizer was) plus a
    /// `special_tokens_map.json`, and stopped there -- a comment noted that
    /// `vocab.txt` / `merges.txt` were skipped as "a basic implementation for
    /// demonstration purposes". The result loaded back as an error at best and
    /// as a different tokenizer at worst.
    pub fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        let _ = save_directory;
        Err(PyValueError::new_err(
            "PreTrainedTokenizer.save_pretrained() has no vocabulary to save (this is the base \
             class): call save_pretrained on a concrete tokenizer subclass such as \
             WordPieceTokenizer or BPETokenizer instead.",
        ))
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

impl PyWordPieceTokenizer {
    /// The wrapped Rust tokenizer, for the task pipelines.
    pub(crate) fn tokenizer(&self) -> &WordPieceTokenizer {
        &self.inner
    }
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

    /// Load a WordPiece tokenizer from a local model directory.
    ///
    /// `model_name_or_path` must be local -- a directory holding `vocab.txt`
    /// (or `vocab.json`), or a file sitting next to one. This crate has no
    /// Hugging Face Hub downloader, so a bare model name is refused outright.
    ///
    /// Refusing is the point: the previous implementation could only ever
    /// raise "Failed to read vocab.txt: No such file or directory" (its
    /// `download_file_from_hub` stub returned the empty *path* for every
    /// request), and its unreachable else-branch would have fallen back to a
    /// five-entry `[PAD]/[UNK]/[CLS]/[SEP]/[MASK]` vocabulary -- under which
    /// every real word tokenizes to `[UNK]` and any downstream classification
    /// score is meaningless.
    #[staticmethod]
    #[pyo3(signature = (model_name_or_path, **_kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        model_name_or_path: &str,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyWordPieceTokenizer>> {
        let read = |path: &std::path::Path| -> PyResult<String> {
            std::fs::read_to_string(path).map_err(|e| {
                PyValueError::new_err(format!("Failed to read {}: {e}", path.display()))
            })
        };

        let vocab = if let Some(path) = find_local_tokenizer_file(model_name_or_path, "vocab.txt") {
            parse_vocab_txt(&read(&path)?)
        } else if let Some(path) = find_local_tokenizer_file(model_name_or_path, "vocab.json") {
            parse_vocab_json(&read(&path)?).map_err(PyValueError::new_err)?
        } else {
            return Err(PyValueError::new_err(format!(
                "no vocab.txt or vocab.json found for '{model_name_or_path}'. This crate has no \
                 Hugging Face Hub downloader, so the path must be a local directory holding the \
                 tokenizer files (or a file next to them)."
            )));
        };

        if vocab.is_empty() {
            return Err(PyValueError::new_err(format!(
                "the vocabulary file for '{model_name_or_path}' is empty"
            )));
        }

        // `tokenizer_config.json` is genuinely optional: its absence means the
        // WordPiece defaults apply, which are real defaults rather than
        // invented data.
        let config = match find_local_tokenizer_file(model_name_or_path, "tokenizer_config.json") {
            Some(path) => Some(
                serde_json::from_str::<serde_json::Value>(&read(&path)?).map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to parse {}: {e}",
                        path.display()
                    ))
                })?,
            ),
            None => None,
        };
        let special = special_tokens_from_config(config.as_ref());

        // Special-token ids come from the vocabulary itself. A checkpoint whose
        // vocabulary does not contain its own declared special tokens is
        // broken, and reporting an id that is not in the vocabulary (as the
        // previous `unwrap_or(0..4)` defaults did) would corrupt every encoding
        // built from it.
        let token_id = |token: &str| -> PyResult<usize> {
            vocab.get(token).map(|&id| id as usize).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "special token '{token}' is not present in the vocabulary of \
                     '{model_name_or_path}'"
                ))
            })
        };
        let base = PyPreTrainedTokenizer {
            pad_token_id: token_id(&special.pad)?,
            unk_token_id: token_id(&special.unk)?,
            cls_token_id: token_id(&special.cls)?,
            sep_token_id: token_id(&special.sep)?,
            mask_token_id: token_id(&special.mask)?,
            pad_token: special.pad,
            unk_token: special.unk,
            cls_token: special.cls,
            sep_token: special.sep,
            mask_token: special.mask,
        };

        let do_lower_case = config
            .as_ref()
            .and_then(|config| config.get("do_lower_case"))
            .and_then(|value| value.as_bool())
            .unwrap_or(true);

        let tokenizer = WordPieceTokenizer::new(vocab, do_lower_case);
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
        // Encoding options are accepted for HF API parity; the underlying tokenizer
        // applies its configured defaults.
        let _ = (add_special_tokens, max_length, padding, truncation);
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
                _ => output.input_ids.into_py_any(py),
            }
        } else {
            output.input_ids.into_py_any(py)
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
        // Encoding options are accepted for HF API parity; the underlying tokenizer
        // applies its configured defaults.
        let _ = (add_special_tokens, max_length, padding, truncation, return_tensors);
        // One call per text (there is no dedicated batch-encode primitive on
        // the wrapped tokenizer), routed per item through `wordpiece_batch_item`.
        let mut outputs = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let text_pair =
                text_pairs.as_ref().and_then(|pairs| pairs.get(i)).and_then(|p| p.as_deref());

            let tokenized = wordpiece_batch_item(&self.inner, text, text_pair)
                .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))?;

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
                .map(|o| {
                    o.token_type_ids
                        .clone()
                        .expect("token_type_ids is Some from all() check")
                        .iter()
                        .map(|&id| id as usize)
                        .collect()
                })
                .collect();
            dict.set_item("token_type_ids", token_type_ids)?;
        }

        Ok(dict.into())
    }

    /// Decode IDs to text
    pub fn decode(&self, ids: Vec<usize>, skip_special_tokens: bool) -> PyResult<String> {
        // Special-token filtering is handled by the underlying decoder configuration.
        let _ = skip_special_tokens;
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
        // A Rust `panic!` crossing the PyO3 boundary surfaces to Python as an
        // uncatchable `pyo3_runtime.PanicException` with a Rust backtrace, not a
        // `TypeError`. A mismatched `text_pair` is ordinary bad input from Python
        // callers, so it must be a normal, catchable `PyResult` error instead --
        // `resolve_single_text_pair`/`resolve_batch_text_pair` below are the pure
        // (interpreter-free) logic behind that check, unit-tested directly.
        match text {
            TextInput::Single(s) => {
                let pair = resolve_single_text_pair(text_pair).map_err(PyTypeError::new_err)?;
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
                let pairs = resolve_batch_text_pair(text_pair).map_err(PyTypeError::new_err)?;
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

    /// Save this tokenizer to `save_directory` in the format
    /// [`PyWordPieceTokenizer::from_pretrained`] reads back.
    ///
    /// Writes a real `vocab.txt` (one token per line, id = line number), a
    /// `tokenizer_config.json` carrying this tokenizer's *actual* vocabulary
    /// size and `do_lower_case` setting, and a `special_tokens_map.json`. The
    /// base-class implementation this overrides wrote no vocabulary at all and
    /// a hardcoded `"vocab_size": 30522`.
    pub fn save_pretrained(slf: PyRef<'_, Self>, save_directory: &str) -> PyResult<()> {
        let vocab = vocab_txt_body(&slf.inner.get_vocab()).map_err(PyValueError::new_err)?;
        let base = slf.as_super();
        let special = SpecialTokens {
            pad: base.pad_token.clone(),
            unk: base.unk_token.clone(),
            cls: base.cls_token.clone(),
            sep: base.sep_token.clone(),
            mask: base.mask_token.clone(),
        };
        let config = serde_json::json!({
            "tokenizer_class": "WordPieceTokenizer",
            "do_lower_case": slf.inner.do_lower_case(),
            "vocab_size": slf.inner.vocab_size(),
            "pad_token": special.pad,
            "unk_token": special.unk,
            "cls_token": special.cls,
            "sep_token": special.sep,
            "mask_token": special.mask,
        });

        write_tokenizer_files(
            save_directory,
            &[
                ("vocab.txt", vocab),
                ("tokenizer_config.json", tokenizer_json(&config)?),
                (
                    "special_tokens_map.json",
                    tokenizer_json(&special_tokens_map_json(&special))?,
                ),
            ],
        )
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

impl PyBPETokenizer {
    /// The wrapped Rust tokenizer, for the task pipelines.
    pub(crate) fn tokenizer(&self) -> &BPETokenizer {
        &self.inner
    }
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
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| (k, v as u32))
            .collect();
        let merges = merges.unwrap_or_default();

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

    /// Load a BPE tokenizer from a local model directory.
    ///
    /// Reads `vocab.json` and `merges.txt` -- both required, because a BPE
    /// tokenizer without its merge table cannot reproduce the vocabulary it
    /// was trained with. `model_name_or_path` must be local; this crate has no
    /// Hugging Face Hub downloader.
    ///
    /// This class previously had no `from_pretrained` at all, so
    /// `AutoTokenizer.from_pretrained("gpt2")` handed back a
    /// `BPETokenizer::new(empty_vocab, no_merges)` and reported success.
    #[staticmethod]
    #[pyo3(signature = (model_name_or_path, **_kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        model_name_or_path: &str,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyBPETokenizer>> {
        let read = |path: &std::path::Path| -> PyResult<String> {
            std::fs::read_to_string(path).map_err(|e| {
                PyValueError::new_err(format!("Failed to read {}: {e}", path.display()))
            })
        };
        let require = |filename: &str| -> PyResult<PathBuf> {
            find_local_tokenizer_file(model_name_or_path, filename).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "no {filename} found for '{model_name_or_path}'. This crate has no Hugging \
                     Face Hub downloader, so the path must be a local directory holding the \
                     tokenizer files (or a file next to them)."
                ))
            })
        };

        let vocab = parse_vocab_json(&read(&require("vocab.json")?)?)
            .map_err(PyValueError::new_err)?;
        if vocab.is_empty() {
            return Err(PyValueError::new_err(format!(
                "the vocab.json for '{model_name_or_path}' is empty"
            )));
        }
        let merges =
            parse_merges_txt(&read(&require("merges.txt")?)?).map_err(PyValueError::new_err)?;

        let config = match find_local_tokenizer_file(model_name_or_path, "tokenizer_config.json") {
            Some(path) => Some(
                serde_json::from_str::<serde_json::Value>(&read(&path)?).map_err(|e| {
                    PyValueError::new_err(format!("Failed to parse {}: {e}", path.display()))
                })?,
            ),
            None => None,
        };
        // GPT-2-family defaults, overridable by tokenizer_config.json.
        let read_token = |key: &str, fallback: &str| -> String {
            config
                .as_ref()
                .and_then(|config| config.get(key))
                .and_then(|value| value.as_str())
                .unwrap_or(fallback)
                .to_string()
        };
        let pad_token = read_token("pad_token", "<|endoftext|>");
        let unk_token = read_token("unk_token", "<|endoftext|>");
        let cls_token = read_token("cls_token", "<|endoftext|>");
        let sep_token = read_token("sep_token", "<|endoftext|>");
        let mask_token = read_token("mask_token", "<|endoftext|>");

        let token_id = |token: &str| -> PyResult<usize> {
            vocab.get(token).map(|&id| id as usize).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "special token '{token}' is not present in the vocabulary of \
                     '{model_name_or_path}'"
                ))
            })
        };
        let base = PyPreTrainedTokenizer {
            pad_token_id: token_id(&pad_token)?,
            unk_token_id: token_id(&unk_token)?,
            cls_token_id: token_id(&cls_token)?,
            sep_token_id: token_id(&sep_token)?,
            mask_token_id: token_id(&mask_token)?,
            pad_token,
            unk_token,
            cls_token,
            sep_token,
            mask_token,
        };

        Py::new(
            py,
            (
                PyBPETokenizer {
                    inner: BPETokenizer::new(vocab, merges),
                },
                base,
            ),
        )
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
        // Encoding options are accepted for HF API parity; the underlying tokenizer
        // applies its configured defaults.
        let _ = (add_special_tokens, max_length, padding, truncation);
        let output = match text_pair {
            Some(text2) => {
                bpe_pair_encoding(&self.inner, text, text2).map_err(PyValueError::new_err)?
            },
            None => self
                .inner
                .encode(text)
                .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))?,
        };

        if let Some(format) = return_tensors {
            match format {
                "pt" | "np" => {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("input_ids", output.input_ids)?;
                    dict.set_item("attention_mask", output.attention_mask)?;
                    if let Some(token_type_ids) = output.token_type_ids {
                        dict.set_item("token_type_ids", token_type_ids)?;
                    }
                    Ok(dict.into())
                },
                _ => output.input_ids.into_py_any(py),
            }
        } else {
            output.input_ids.into_py_any(py)
        }
    }

    /// Decode IDs to text
    pub fn decode(&self, ids: Vec<usize>, skip_special_tokens: bool) -> PyResult<String> {
        // Special-token filtering is handled by the underlying decoder configuration.
        let _ = skip_special_tokens;
        let ids_u32: Vec<u32> = ids.into_iter().map(|id| id as u32).collect();
        self.inner
            .decode(&ids_u32)
            .map_err(|e| PyValueError::new_err(format!("Decoding failed: {}", e)))
    }

    /// Save this tokenizer to `save_directory` in the format
    /// [`PyBPETokenizer::from_pretrained`] reads back.
    ///
    /// Writes a real `vocab.json` and `merges.txt` (merge order preserved,
    /// `#version: 0.2` header included) alongside the config files. Without the
    /// merge table a BPE tokenizer cannot reproduce its own tokenization, which
    /// is why the base-class implementation this overrides -- which wrote
    /// neither -- could not round-trip.
    pub fn save_pretrained(slf: PyRef<'_, Self>, save_directory: &str) -> PyResult<()> {
        let vocab = slf.inner.get_vocab_map();
        if vocab.is_empty() {
            return Err(PyValueError::new_err(
                "this BPETokenizer has an empty vocabulary, so there is nothing to save",
            ));
        }
        let base = slf.as_super();
        let special = SpecialTokens {
            pad: base.pad_token.clone(),
            unk: base.unk_token.clone(),
            cls: base.cls_token.clone(),
            sep: base.sep_token.clone(),
            mask: base.mask_token.clone(),
        };
        let vocab_json = serde_json::to_value(vocab)
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize vocab.json: {e}")))?;
        let config = serde_json::json!({
            "tokenizer_class": "BPETokenizer",
            "vocab_size": slf.inner.vocab_size(),
            "pad_token": special.pad,
            "unk_token": special.unk,
            "cls_token": special.cls,
            "sep_token": special.sep,
            "mask_token": special.mask,
        });

        write_tokenizer_files(
            save_directory,
            &[
                ("vocab.json", tokenizer_json(&vocab_json)?),
                ("merges.txt", merges_txt_body(slf.inner.get_merge_rules())),
                ("tokenizer_config.json", tokenizer_json(&config)?),
                (
                    "special_tokens_map.json",
                    tokenizer_json(&special_tokens_map_json(&special))?,
                ),
            ],
        )
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

/// Pure-Rust core of [`PyWordPieceTokenizer::__call__`]'s `text_pair` type
/// check for the `text: TextInput::Single` branch (`text` is one string, so
/// `text_pair`, if given, must also be one string).
///
/// Split out from `__call__` so it is unit-testable without an initialized
/// Python interpreter -- `TextInput` is a plain Rust enum (its
/// `#[derive(FromPyObject)]` only matters when extracting *from* a Python
/// object), so constructing values of it needs no GIL.
///
/// This used to be a `match` arm ending in `panic!("text_pair must be a
/// string when text is a string")`: a bare Rust panic crossing the PyO3
/// boundary surfaces to Python as an uncatchable `pyo3_runtime.PanicException`
/// (not a `TypeError`, not catchable by `except TypeError`), printing a Rust
/// backtrace. Returning `Err` here lets the PyO3 boundary turn it into a
/// normal, catchable `PyTypeError` instead.
fn resolve_single_text_pair(text_pair: Option<TextInput>) -> Result<Option<String>, &'static str> {
    match text_pair {
        Some(TextInput::Single(s)) => Ok(Some(s)),
        Some(TextInput::Batch(_)) => Err("text_pair must be a string when text is a string"),
        None => Ok(None),
    }
}

/// Pure-Rust core of [`PyWordPieceTokenizer::__call__`]'s `text_pair` type
/// check for the `text: TextInput::Batch` branch (`text` is a list of
/// strings, so `text_pair`, if given, must also be a list).
///
/// See [`resolve_single_text_pair`] for why this is split out and what it
/// replaces (the `TextInput::Single` arm's mirror-image `panic!`).
#[allow(clippy::type_complexity)]
fn resolve_batch_text_pair(
    text_pair: Option<TextInput>,
) -> Result<Option<Vec<Option<String>>>, &'static str> {
    match text_pair {
        Some(TextInput::Batch(pairs)) => Ok(Some(pairs.into_iter().map(Some).collect())),
        Some(TextInput::Single(_)) => Err("text_pair must be a list when text is a list"),
        None => Ok(None),
    }
}

#[cfg(test)]
mod local_asset_tests {
    use super::*;
    use std::fs;

    /// Unique temporary directory under `std::env::temp_dir()`, per the
    /// workspace's test-file policy.
    fn temp_dir(name: &str) -> PathBuf {
        let unique = format!(
            "trustformers-py-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let dir = std::env::temp_dir().join(unique);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir is creatable");
        dir
    }

    // ---- find_local_tokenizer_file ----

    /// The replaced `download_file_from_hub` stub returned `Ok("")` for every
    /// request, so no asset was ever found however complete the directory was.
    #[test]
    fn finds_an_asset_in_a_model_directory() {
        let dir = temp_dir("find-dir");
        fs::write(dir.join("vocab.txt"), "[PAD]\n").expect("write vocab");
        let found = find_local_tokenizer_file(
            dir.to_str().expect("utf-8 temp path"),
            "vocab.txt",
        );
        assert_eq!(found.as_deref(), Some(dir.join("vocab.txt").as_path()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn finds_an_asset_beside_a_file_path() {
        let dir = temp_dir("find-sibling");
        fs::write(dir.join("vocab.txt"), "[PAD]\n").expect("write vocab");
        fs::write(dir.join("config.json"), "{}").expect("write config");
        let config = dir.join("config.json");
        let found = find_local_tokenizer_file(
            config.to_str().expect("utf-8 temp path"),
            "vocab.txt",
        );
        assert_eq!(found.as_deref(), Some(dir.join("vocab.txt").as_path()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reports_a_missing_asset_as_absent() {
        let dir = temp_dir("find-missing");
        assert!(
            find_local_tokenizer_file(dir.to_str().expect("utf-8 temp path"), "vocab.txt")
                .is_none()
        );
        assert!(find_local_tokenizer_file("bert-base-uncased", "vocab.txt").is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    // ---- parse_vocab_txt ----

    #[test]
    fn vocab_txt_ids_are_line_indices() {
        let vocab = parse_vocab_txt("[PAD]\n[UNK]\nhello\nworld\n");
        assert_eq!(vocab.get("[PAD]"), Some(&0));
        assert_eq!(vocab.get("[UNK]"), Some(&1));
        assert_eq!(vocab.get("hello"), Some(&2));
        assert_eq!(vocab.get("world"), Some(&3));
    }

    /// A blank line still consumes an id in HuggingFace's reader. Skipping it
    /// would shift every later token by one and silently mis-tokenize the
    /// whole vocabulary.
    #[test]
    fn a_blank_line_still_consumes_an_id() {
        let vocab = parse_vocab_txt("[PAD]\n\nhello\n");
        assert_eq!(vocab.get("[PAD]"), Some(&0));
        assert_eq!(vocab.get("hello"), Some(&2));
        assert_eq!(vocab.len(), 2);
    }

    /// Tokens may legitimately contain leading whitespace (SentencePiece-style
    /// pieces); only the line terminator is stripped.
    #[test]
    fn only_line_endings_are_stripped() {
        let vocab = parse_vocab_txt("a\r\n b\n");
        assert_eq!(vocab.get("a"), Some(&0));
        assert_eq!(vocab.get(" b"), Some(&1));
    }

    // ---- parse_vocab_json ----

    #[test]
    fn parses_a_vocab_json_object() {
        let vocab = parse_vocab_json(r#"{"[PAD]": 0, "hello": 7}"#).expect("valid vocab.json");
        assert_eq!(vocab.get("[PAD]"), Some(&0));
        assert_eq!(vocab.get("hello"), Some(&7));
    }

    /// The replaced reader used `filter_map(..).unwrap_or_default()`, so a
    /// malformed document produced a silently partial (or empty) vocabulary
    /// instead of an error.
    #[test]
    fn rejects_malformed_vocab_json() {
        assert!(parse_vocab_json("not json").is_err());
        assert!(parse_vocab_json("[1, 2, 3]").is_err());
        assert!(parse_vocab_json(r#"{"hello": "seven"}"#).is_err());
        assert!(parse_vocab_json(r#"{"hello": -1}"#).is_err());
    }

    // ---- parse_merges_txt ----

    #[test]
    fn parses_merges_in_priority_order() {
        let merges = parse_merges_txt("#version: 0.2\nt h\nth e\n").expect("valid merges.txt");
        assert_eq!(
            merges,
            vec![
                ("t".to_string(), "h".to_string()),
                ("th".to_string(), "e".to_string())
            ]
        );
    }

    /// Merge *order* is the content of the file, so a malformed line cannot be
    /// skipped: doing so would shift the priority of every merge after it.
    #[test]
    fn rejects_a_malformed_merges_line() {
        assert!(parse_merges_txt("t h\nonlyone\n").is_err());
        assert!(parse_merges_txt("a b c\n").is_err());
    }

    #[test]
    fn blank_lines_and_the_version_header_are_skipped() {
        let merges = parse_merges_txt("#version: 0.2\n\na b\n\n").expect("valid merges.txt");
        assert_eq!(merges, vec![("a".to_string(), "b".to_string())]);
    }

    // ---- vocab_txt_body / merges_txt_body: the save -> load round trip ----

    /// `save_pretrained` must write exactly what `from_pretrained` reads back;
    /// the base-class implementation this replaces wrote no vocabulary at all.
    #[test]
    fn wordpiece_vocab_round_trips_through_the_file_format() {
        let vocab: HashMap<String, u32> = [
            ("[PAD]".to_string(), 0u32),
            ("[UNK]".to_string(), 1),
            ("hello".to_string(), 2),
            ("##world".to_string(), 3),
        ]
        .into_iter()
        .collect();
        let tokenizer = WordPieceTokenizer::new(vocab.clone(), true);

        let body = vocab_txt_body(&tokenizer.get_vocab()).expect("dense vocabulary exports");
        assert_eq!(parse_vocab_txt(&body), vocab);
    }

    /// Ids are line numbers, so an id with no token is a blank line. That
    /// keeps every later token on its own id, and round-trips exactly.
    #[test]
    fn a_sparse_vocabulary_round_trips_through_blank_lines() {
        let vocab: HashMap<String, u32> =
            [("[PAD]".to_string(), 0u32), ("hello".to_string(), 5)].into_iter().collect();
        let body = vocab_txt_body(&vocab).expect("sparse vocabulary exports");
        assert_eq!(body.lines().count(), 6);
        assert_eq!(parse_vocab_txt(&body), vocab);
    }

    /// Two tokens on one id cannot both survive a format with one line per id.
    #[test]
    fn duplicate_ids_are_refused() {
        let vocab: HashMap<String, u32> =
            [("a".to_string(), 0u32), ("b".to_string(), 0)].into_iter().collect();
        assert!(vocab_txt_body(&vocab).is_err());
    }

    #[test]
    fn an_empty_vocabulary_is_refused() {
        assert!(vocab_txt_body(&HashMap::new()).is_err());
    }

    #[test]
    fn merges_round_trip_through_the_file_format() {
        let merges = vec![
            ("t".to_string(), "h".to_string()),
            ("th".to_string(), "e".to_string()),
        ];
        let body = merges_txt_body(&merges);
        assert!(body.starts_with("#version: 0.2\n"));
        assert_eq!(parse_merges_txt(&body).expect("valid merges"), merges);
    }

    #[test]
    fn an_empty_merge_table_still_writes_a_valid_file() {
        let body = merges_txt_body(&[]);
        assert_eq!(parse_merges_txt(&body).expect("valid merges"), Vec::new());
    }

    /// The BPE save path serialises the vocabulary with `serde_json::to_value`
    /// and the merges with [`merges_txt_body`]; both must be readable back by
    /// the exact readers `BPETokenizer::from_pretrained` uses. Without the
    /// merge table a BPE tokenizer cannot reproduce its own tokenization, so
    /// the round trip has to cover both files, not just the vocabulary.
    #[test]
    fn bpe_vocab_and_merges_round_trip_through_the_file_format() {
        let vocab: HashMap<String, u32> = [
            ("<|endoftext|>".to_string(), 0u32),
            ("th".to_string(), 1),
            ("the".to_string(), 2),
            ("Ġthe".to_string(), 3),
        ]
        .into_iter()
        .collect();
        let merges = vec![
            ("t".to_string(), "h".to_string()),
            ("th".to_string(), "e".to_string()),
        ];
        let tokenizer = BPETokenizer::new(vocab.clone(), merges.clone());

        let vocab_body = serde_json::to_string_pretty(tokenizer.get_vocab_map())
            .expect("vocab map serializes");
        assert_eq!(
            parse_vocab_json(&vocab_body).expect("saved vocab.json is readable"),
            vocab
        );

        let merges_body = merges_txt_body(tokenizer.get_merge_rules());
        assert_eq!(
            parse_merges_txt(&merges_body).expect("saved merges.txt is readable"),
            merges
        );
    }

    #[test]
    fn special_tokens_map_carries_all_five_names() {
        let special = SpecialTokens {
            pad: "<pad>".to_string(),
            unk: "<unk>".to_string(),
            cls: "<s>".to_string(),
            sep: "</s>".to_string(),
            mask: "<mask>".to_string(),
        };
        let json = special_tokens_map_json(&special);
        assert_eq!(json.get("pad_token").and_then(|v| v.as_str()), Some("<pad>"));
        assert_eq!(json.get("mask_token").and_then(|v| v.as_str()), Some("<mask>"));
        // The saved map must be readable back by the config reader.
        let reloaded = special_tokens_from_config(Some(&json));
        assert_eq!(reloaded.pad, "<pad>");
        assert_eq!(reloaded.sep, "</s>");
    }

    // ---- special_tokens_from_config ----

    #[test]
    fn special_tokens_fall_back_to_wordpiece_defaults() {
        let special = special_tokens_from_config(None);
        assert_eq!(special.pad, "[PAD]");
        assert_eq!(special.unk, "[UNK]");
        assert_eq!(special.cls, "[CLS]");
        assert_eq!(special.sep, "[SEP]");
        assert_eq!(special.mask, "[MASK]");
    }

    #[test]
    fn special_tokens_follow_the_tokenizer_config() {
        let config = serde_json::json!({"pad_token": "<pad>", "unk_token": "<unk>"});
        let special = special_tokens_from_config(Some(&config));
        assert_eq!(special.pad, "<pad>");
        assert_eq!(special.unk, "<unk>");
        // Unset entries keep the defaults.
        assert_eq!(special.cls, "[CLS]");
    }
}

#[cfg(test)]
mod pair_encoding_tests {
    use super::*;

    fn wordpiece_fixture() -> WordPieceTokenizer {
        let vocab: HashMap<String, u32> = [
            ("[PAD]", 0u32),
            ("[UNK]", 1),
            ("[CLS]", 2),
            ("[SEP]", 3),
            ("[MASK]", 4),
            ("hello", 5),
            ("world", 6),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        WordPieceTokenizer::new(vocab, true)
    }

    // ---- wordpiece_batch_item ----

    #[test]
    fn wordpiece_batch_item_pair_matches_encode_pair() {
        let tokenizer = wordpiece_fixture();
        let via_helper =
            wordpiece_batch_item(&tokenizer, "hello", Some("world")).expect("pair encodes");
        let direct = tokenizer.encode_pair("hello", "world").expect("pair encodes");
        assert_eq!(via_helper.input_ids, direct.input_ids);
        assert_eq!(via_helper.token_type_ids, direct.token_type_ids);
    }

    #[test]
    fn wordpiece_batch_item_without_a_pair_matches_encode() {
        let tokenizer = wordpiece_fixture();
        let via_helper = wordpiece_batch_item(&tokenizer, "hello", None).expect("single encodes");
        let direct = tokenizer.encode("hello").expect("single encodes");
        assert_eq!(via_helper.input_ids, direct.input_ids);
    }

    /// The regression this helper exists for: the previous `batch_encode_plus`
    /// pair handling concatenated two independent `encode()` calls, so there
    /// was no `[SEP]` between segments, and `token_type_ids` stayed the
    /// length of (and all zero for) the first segment alone -- shorter than
    /// the now-concatenated `input_ids`.
    #[test]
    fn wordpiece_batch_item_pair_has_two_separators_and_full_length_token_type_ids() {
        let tokenizer = wordpiece_fixture();
        let output =
            wordpiece_batch_item(&tokenizer, "hello", Some("world")).expect("pair encodes");
        let token_type_ids = output.token_type_ids.expect("pair encoding carries token_type_ids");
        assert_eq!(
            token_type_ids.len(),
            output.input_ids.len(),
            "token_type_ids must cover every token, not just the first segment"
        );
        assert!(
            token_type_ids.contains(&0) && token_type_ids.contains(&1),
            "a real pair encoding must mark both segments, got {:?}",
            token_type_ids
        );
        let sep_id = tokenizer.token_to_id("[SEP]").expect("[SEP] is in the fixture vocab");
        let sep_count = output.input_ids.iter().filter(|&&id| id == sep_id).count();
        assert_eq!(sep_count, 2, "[CLS] A [SEP] B [SEP] has exactly two [SEP] tokens");
    }

    // ---- bpe_pair_encoding ----

    fn bpe_fixture_with_eot_in_vocab() -> BPETokenizer {
        let vocab: HashMap<String, u32> = [("<|endoftext|>", 0u32), ("h", 1), ("i", 2)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        BPETokenizer::new(vocab, vec![])
    }

    fn bpe_fixture_missing_eot_from_vocab() -> BPETokenizer {
        let vocab: HashMap<String, u32> =
            [("h", 0u32), ("i", 1)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        BPETokenizer::new(vocab, vec![])
    }

    #[test]
    fn bpe_pair_encoding_wraps_with_bos_and_double_eos() {
        let tokenizer = bpe_fixture_with_eot_in_vocab();
        let eot_id = tokenizer.token_to_id("<|endoftext|>").expect("in fixture vocab");
        let output = bpe_pair_encoding(&tokenizer, "hi", "hi").expect("pair encodes");

        // <bos> h i <eos> <eos> h i <eos>: 8 ids, the boundary token at
        // exactly 4 of them (positions 0, 3, 4, 7).
        assert_eq!(output.input_ids.len(), 8, "{:?}", output.input_ids);
        assert_eq!(output.input_ids.first(), Some(&eot_id), "must start with bos");
        assert_eq!(output.input_ids.last(), Some(&eot_id), "must end with eos");
        let boundary_count = output.input_ids.iter().filter(|&&id| id == eot_id).count();
        assert_eq!(
            boundary_count, 4,
            "expected bos + two eos separators + trailing eos, got {:?}",
            output.input_ids
        );
    }

    #[test]
    fn bpe_pair_encoding_uses_all_zero_token_type_ids() {
        let tokenizer = bpe_fixture_with_eot_in_vocab();
        let output = bpe_pair_encoding(&tokenizer, "h", "h").expect("pair encodes");
        let token_type_ids = output.token_type_ids.expect("pair encoding carries token_type_ids");
        assert_eq!(token_type_ids.len(), output.input_ids.len());
        assert!(
            token_type_ids.iter().all(|&t| t == 0),
            "BPE-family pairs use no segment ids, got {:?}",
            token_type_ids
        );
    }

    #[test]
    fn bpe_pair_encoding_attention_mask_covers_every_token() {
        let tokenizer = bpe_fixture_with_eot_in_vocab();
        let output = bpe_pair_encoding(&tokenizer, "hi", "hi").expect("pair encodes");
        assert_eq!(output.attention_mask.len(), output.input_ids.len());
        assert!(output.attention_mask.iter().all(|&m| m == 1));
    }

    /// A vocabulary missing only the boundary token (not the content tokens)
    /// must be a clear error, not a silent fallback to `unk_token`'s id --
    /// that would fabricate a boundary that was never actually placed there.
    #[test]
    fn bpe_pair_encoding_rejects_a_vocabulary_missing_the_boundary_token() {
        let tokenizer = bpe_fixture_missing_eot_from_vocab();
        let err = bpe_pair_encoding(&tokenizer, "h", "h")
            .expect_err("a vocabulary with no bos/eos token must error, not fabricate an id");
        assert!(err.contains("<|endoftext|>"), "error must name the missing token: {err}");
    }

    /// Different second sequences must produce different encodings -- proves
    /// this composes real per-sequence tokenization rather than, say, always
    /// wrapping the first sequence twice.
    #[test]
    fn bpe_pair_encoding_reflects_both_sequences() {
        let tokenizer = bpe_fixture_with_eot_in_vocab();
        let same = bpe_pair_encoding(&tokenizer, "h", "h").expect("pair encodes");
        let different = bpe_pair_encoding(&tokenizer, "h", "hi").expect("pair encodes");
        assert_ne!(same.input_ids, different.input_ids);
    }

    // ---- offset_mapping (previously hardcoded `None`; see the function's own doc) ----

    #[test]
    fn bpe_pair_encoding_offset_mapping_has_one_entry_per_token() {
        let tokenizer = bpe_fixture_with_eot_in_vocab();
        let output = bpe_pair_encoding(&tokenizer, "hi", "hi").expect("pair encodes");
        let offsets = output.offset_mapping.expect("offset_mapping must be populated");
        assert_eq!(
            offsets.len(),
            output.input_ids.len(),
            "offset_mapping must have exactly one entry per token, got {} offsets for {} tokens",
            offsets.len(),
            output.input_ids.len()
        );
    }

    /// Ground truth measured directly against `BPETokenizer::encode` (not
    /// assumed): with this fixture's empty merge table, byte-level BPE
    /// tokenizes one raw byte at a time, and `tokenize_with_offsets` widens
    /// each piece's span out to the enclosing character boundary -- so a
    /// multi-byte character's several byte-pieces all report the *same*
    /// widened span (see that method's own doc for why). "café" (5 bytes:
    /// c, a, f, then é's 2 bytes) therefore yields 5 tokens over 4 distinct
    /// spans, and "北京" (6 bytes: two 3-byte characters) yields 6 tokens
    /// over 2 distinct spans. Repeated consecutive spans here are correct,
    /// not a bug.
    ///
    /// This also locks the four boundary positions (`bos`, the two `eos`
    /// separators, the trailing `eos`) at `(0, 0)` -- HuggingFace's
    /// convention for a special token with no source span, matching
    /// `WordPieceTokenizer::encode_pair`'s `SPECIAL_TOKEN_SPAN`.
    #[test]
    fn bpe_pair_encoding_offsets_match_measured_ground_truth() {
        let tokenizer = bpe_fixture_with_eot_in_vocab();
        let output = bpe_pair_encoding(&tokenizer, "café", "北京").expect("pair encodes");

        // <bos> c a f é é <eos> <eos> 北 北 北 京 京 京 <eos> = 1+5+2+6+1 = 15
        assert_eq!(output.input_ids.len(), 15, "{:?}", output.input_ids);

        let offsets = output.offset_mapping.expect("offset_mapping must be populated");
        let expected = vec![
            (0, 0), // bos
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 5),
            (3, 5), // c a f é é -- indexes "café"
            (0, 0),
            (0, 0), // eos eos
            (0, 3),
            (0, 3),
            (0, 3),
            (3, 6),
            (3, 6),
            (3, 6), // 北 北 北 京 京 京 -- indexes "北京"
            (0, 0), // eos
        ];
        assert_eq!(offsets, expected, "{:?}", offsets);
    }

    /// Falsifies the specific bug this fix replaces: a naive implementation
    /// could shift the second sequence's offsets into a combined
    /// `text` + `text2` coordinate space (the way `BPETokenizer::encode_pair`
    /// -- deliberately not called here -- indexes its own joined string).
    /// Slicing `text2` directly at the second segment's reported spans must
    /// recover `text2`'s real characters; the value a `text.len()`-shifted
    /// bug would have reported is not even a valid slice of the (shorter)
    /// `text2`.
    #[test]
    fn bpe_pair_encoding_second_sequence_offsets_index_text2_not_a_combined_string() {
        let tokenizer = bpe_fixture_with_eot_in_vocab();
        let text = "café";
        let text2 = "北京";
        let output = bpe_pair_encoding(&tokenizer, text, text2).expect("pair encodes");
        let offsets = output.offset_mapping.expect("offset_mapping must be populated");

        // Positions 8..=13 are the second segment (bos=0, café=1..=5,
        // eos,eos=6,7, 北京=8..=13, eos=14).
        let second_segment = &offsets[8..14];
        assert_eq!(text2.get(second_segment[0].0..second_segment[0].1), Some("北"));
        assert_eq!(text2.get(second_segment[3].0..second_segment[3].1), Some("京"));

        let bogus_shift = text.len(); // what a combined-coordinate-space bug would add
        assert!(
            text2.get(bogus_shift..bogus_shift + 3).is_none(),
            "a text.len()-shifted offset must not even be a valid text2 slice"
        );
    }

    #[test]
    fn bpe_pair_encoding_special_tokens_mask_flags_exactly_the_four_boundary_positions() {
        let tokenizer = bpe_fixture_with_eot_in_vocab();
        let output = bpe_pair_encoding(&tokenizer, "hi", "hi").expect("pair encodes");
        let mask = output.special_tokens_mask.expect("special_tokens_mask must be populated");
        assert_eq!(mask, vec![1, 0, 0, 1, 1, 0, 0, 1], "{:?}", mask);
    }
}

#[cfg(test)]
mod text_pair_tests {
    use super::*;

    // ---- resolve_single_text_pair ----

    #[test]
    fn single_text_pair_none_resolves_to_none() {
        assert_eq!(resolve_single_text_pair(None), Ok(None));
    }

    #[test]
    fn single_text_pair_single_resolves_to_the_string() {
        let result = resolve_single_text_pair(Some(TextInput::Single("b".to_string())));
        assert_eq!(result, Ok(Some("b".to_string())));
    }

    #[test]
    fn single_text_with_batch_text_pair_is_a_type_error_not_a_panic() {
        // Regression test for the P1 finding: `tok(text="a", text_pair=["b"])`
        // used to reach `panic!("text_pair must be a string when text is a
        // string")` -- an uncatchable `pyo3_runtime.PanicException` crossing
        // the PyO3 boundary. Calling this pure function directly proves the
        // mismatch is now an ordinary `Err`, never a Rust panic, with
        // `std::panic::catch_unwind` as the independent proof of "no panic".
        let outcome = std::panic::catch_unwind(|| {
            resolve_single_text_pair(Some(TextInput::Batch(vec!["b".to_string()])))
        });
        match outcome {
            Ok(result) => {
                assert_eq!(result, Err("text_pair must be a string when text is a string"))
            },
            Err(_) => panic!("resolve_single_text_pair must return an Err, not unwind via panic!"),
        }
    }

    // ---- resolve_batch_text_pair ----

    #[test]
    fn batch_text_pair_none_resolves_to_none() {
        assert_eq!(resolve_batch_text_pair(None), Ok(None));
    }

    #[test]
    fn batch_text_pair_batch_resolves_to_wrapped_options() {
        let result = resolve_batch_text_pair(Some(TextInput::Batch(vec![
            "b".to_string(),
            "c".to_string(),
        ])));
        assert_eq!(result, Ok(Some(vec![Some("b".to_string()), Some("c".to_string())])));
    }

    #[test]
    fn batch_text_with_single_text_pair_is_a_type_error_not_a_panic() {
        // Regression test for the P1 finding's mirror-image case:
        // `tok(text=["a"], text_pair="b")` used to reach
        // `panic!("text_pair must be a list when text is a list")`.
        let outcome = std::panic::catch_unwind(|| {
            resolve_batch_text_pair(Some(TextInput::Single("b".to_string())))
        });
        match outcome {
            Ok(result) => assert_eq!(result, Err("text_pair must be a list when text is a list")),
            Err(_) => panic!("resolve_batch_text_pair must return an Err, not unwind via panic!"),
        }
    }
}
