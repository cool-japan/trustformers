use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use trustformers_core::errors::{Result, TrustformersError};

/// `serde(with = "json_value_map")` helper for `BinaryTokenizer::config`.
///
/// `serde_json::Value`'s `Deserialize` impl needs `deserialize_any` (it asks
/// the deserializer at runtime "what shape is this value"), which
/// non-self-describing binary formats like `oxicode`/bincode cannot support
/// (they need the shape known at compile time). Serializing each value as
/// its JSON text instead lets any wire format carry it losslessly as a plain
/// string, then reconstructs the real `Value` on the way back in.
mod json_value_map {
    use serde::de::Error as DeError;
    use serde::ser::Error as SerError;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S: Serializer>(
        map: &HashMap<String, serde_json::Value>,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut as_strings: HashMap<String, String> = HashMap::with_capacity(map.len());
        for (key, value) in map {
            let text = serde_json::to_string(value).map_err(SerError::custom)?;
            as_strings.insert(key.clone(), text);
        }
        as_strings.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<HashMap<String, serde_json::Value>, D::Error> {
        let as_strings: HashMap<String, String> = HashMap::deserialize(deserializer)?;
        let mut map = HashMap::with_capacity(as_strings.len());
        for (key, text) in as_strings {
            let value: serde_json::Value = serde_json::from_str(&text).map_err(DeError::custom)?;
            map.insert(key, value);
        }
        Ok(map)
    }
}

/// Binary format version for compatibility tracking
const BINARY_FORMAT_VERSION: u32 = 1;

/// Magic bytes to identify our binary format
const MAGIC_BYTES: &[u8] = b"TFMT"; // TrustForMers Tokenizer

/// Header information for the binary format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryHeader {
    /// Format version for backward compatibility
    pub version: u32,

    /// Tokenizer type identifier
    pub tokenizer_type: String,

    /// Compression level used (0 = none, 1-9 = zlib levels)
    pub compression_level: u8,

    /// Total size of the uncompressed data
    pub uncompressed_size: u64,

    /// Total size of the compressed data
    pub compressed_size: u64,

    /// Checksum of the uncompressed data
    pub checksum: u32,

    /// Metadata about the tokenizer
    pub metadata: HashMap<String, String>,

    /// Timestamp when this was created
    pub created_at: u64,
}

/// Configuration for binary serialization
#[derive(Debug, Clone)]
pub struct BinaryConfig {
    /// Compression level (0 = no compression, 1-9 = zlib compression levels)
    pub compression_level: u8,

    /// Whether to include metadata in the binary file
    pub include_metadata: bool,

    /// Whether to verify checksums on load
    pub verify_checksums: bool,

    /// Buffer size for I/O operations
    pub buffer_size: usize,

    /// Whether to use memory mapping for large files
    pub use_memory_mapping: bool,
}

impl Default for BinaryConfig {
    fn default() -> Self {
        Self {
            compression_level: 6,
            include_metadata: true,
            verify_checksums: true,
            buffer_size: 64 * 1024, // 64KB
            use_memory_mapping: false,
        }
    }
}

/// Binary tokenizer representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryTokenizer {
    /// Vocabulary mapping from tokens to IDs
    pub vocab: HashMap<String, u32>,

    /// Reverse mapping from IDs to tokens
    pub id_to_token: HashMap<u32, String>,

    /// Special tokens with their IDs
    pub special_tokens: HashMap<String, u32>,

    /// Token scores for ranking (if applicable)
    pub scores: Option<HashMap<u32, f32>>,

    /// Merges for BPE tokenizers (if applicable)
    pub merges: Option<Vec<(String, String)>>,

    /// Additional tokenizer-specific configuration.
    ///
    /// Encoded on the wire as JSON text per entry (see [`json_value_map`])
    /// rather than passed through serde generically: `serde_json::Value`'s
    /// `Deserialize` impl requires `deserialize_any` (runtime type
    /// introspection), which `oxicode`'s non-self-describing binary format
    /// does not implement, so round-tripping a `HashMap<String, Value>`
    /// through it directly fails with "deserialize_any not supported".
    #[serde(with = "json_value_map")]
    pub config: HashMap<String, serde_json::Value>,

    /// Normalization rules
    pub normalization_rules: Option<Vec<NormalizationRule>>,

    /// Pre-tokenization rules
    pub pre_tokenization_rules: Option<Vec<PreTokenizationRule>>,
}

/// Normalization rule for text preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationRule {
    pub rule_type: String,
    /// See [`json_value_map`] for why this needs a custom (de)serializer.
    #[serde(with = "json_value_map")]
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Pre-tokenization rule for splitting text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreTokenizationRule {
    pub rule_type: String,
    pub pattern: String,
    pub replacement: Option<String>,
}

/// Binary serializer/deserializer for tokenizers
pub struct BinarySerializer {
    config: BinaryConfig,
}

impl BinarySerializer {
    /// Create a new binary serializer with the given configuration
    pub fn new(config: BinaryConfig) -> Self {
        Self { config }
    }

    /// Serialize a tokenizer to binary format
    pub fn serialize<P: AsRef<Path>>(
        &self,
        tokenizer: &BinaryTokenizer,
        tokenizer_type: &str,
        path: P,
    ) -> Result<BinaryHeader> {
        let file = File::create(path.as_ref())
            .map_err(|e| TrustformersError::io_error(format!("Failed to create file: {}", e)))?;
        let mut writer = BufWriter::with_capacity(self.config.buffer_size, file);

        // Serialize the tokenizer data
        let data =
            oxicode::serde::encode_to_vec(tokenizer, oxicode::config::standard()).map_err(|e| {
                TrustformersError::serialization_error(format!(
                    "Failed to serialize tokenizer: {}",
                    e
                ))
            })?;

        // Calculate checksum
        let checksum = crc32fast::hash(&data);

        // Compress data if requested
        let (final_data, compressed_size) = if self.config.compression_level > 0 {
            let compressed = self.compress_data(&data)?;
            let size = compressed.len() as u64;
            (compressed, size)
        } else {
            let size = data.len() as u64;
            (data.clone(), size)
        };

        // Create header
        let mut metadata = HashMap::new();
        if self.config.include_metadata {
            metadata.insert("vocab_size".to_string(), tokenizer.vocab.len().to_string());
            metadata.insert(
                "has_scores".to_string(),
                tokenizer.scores.is_some().to_string(),
            );
            metadata.insert(
                "has_merges".to_string(),
                tokenizer.merges.is_some().to_string(),
            );
        }

        let header = BinaryHeader {
            version: BINARY_FORMAT_VERSION,
            tokenizer_type: tokenizer_type.to_string(),
            compression_level: self.config.compression_level,
            uncompressed_size: data.len() as u64,
            compressed_size,
            checksum,
            metadata,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        // Write magic bytes
        writer.write_all(MAGIC_BYTES).map_err(|e| {
            TrustformersError::io_error(format!("Failed to write magic bytes: {}", e))
        })?;

        // Write header
        let header_data = oxicode::serde::encode_to_vec(&header, oxicode::config::standard())
            .map_err(|e| {
                TrustformersError::serialization_error(format!("Failed to serialize header: {}", e))
            })?;
        let header_size = header_data.len() as u32;

        writer.write_all(&header_size.to_le_bytes()).map_err(|e| {
            TrustformersError::io_error(format!("Failed to write header size: {}", e))
        })?;
        writer
            .write_all(&header_data)
            .map_err(|e| TrustformersError::io_error(format!("Failed to write header: {}", e)))?;

        // Write tokenizer data
        writer.write_all(&final_data).map_err(|e| {
            TrustformersError::io_error(format!("Failed to write tokenizer data: {}", e))
        })?;

        writer
            .flush()
            .map_err(|e| TrustformersError::io_error(format!("Failed to flush writer: {}", e)))?;

        Ok(header)
    }

    /// Deserialize a tokenizer from binary format
    pub fn deserialize<P: AsRef<Path>>(&self, path: P) -> Result<(BinaryTokenizer, BinaryHeader)> {
        let file = File::open(path.as_ref())
            .map_err(|e| TrustformersError::io_error(format!("Failed to open file: {}", e)))?;
        let mut reader = BufReader::with_capacity(self.config.buffer_size, file);

        // Read and verify magic bytes
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| {
            TrustformersError::io_error(format!("Failed to read magic bytes: {}", e))
        })?;

        if magic != MAGIC_BYTES {
            return Err(trustformers_core::errors::invalid_format(
                "TFMT",
                String::from_utf8_lossy(&magic).to_string(),
            ));
        }

        // Read header size
        let mut header_size_bytes = [0u8; 4];
        reader.read_exact(&mut header_size_bytes).map_err(|e| {
            TrustformersError::io_error(format!("Failed to read header size: {}", e))
        })?;
        let header_size = u32::from_le_bytes(header_size_bytes) as usize;

        // Read header
        let mut header_data = vec![0u8; header_size];
        reader
            .read_exact(&mut header_data)
            .map_err(|e| TrustformersError::io_error(format!("Failed to read header: {}", e)))?;

        let (header, _): (BinaryHeader, usize) = oxicode::serde::decode_from_slice(
            &header_data,
            oxicode::config::standard(),
        )
        .map_err(|e| {
            TrustformersError::serialization_error(format!("Failed to deserialize header: {}", e))
        })?;

        // Verify version compatibility
        if header.version > BINARY_FORMAT_VERSION {
            return Err(trustformers_core::errors::invalid_format(
                BINARY_FORMAT_VERSION.to_string(),
                header.version.to_string(),
            ));
        }

        // Read tokenizer data
        let mut data = vec![0u8; header.compressed_size as usize];
        reader.read_exact(&mut data).map_err(|e| {
            TrustformersError::io_error(format!("Failed to read tokenizer data: {}", e))
        })?;

        // Decompress if needed
        let final_data = if header.compression_level > 0 {
            self.decompress_data(&data, header.uncompressed_size as usize)?
        } else {
            data
        };

        // Verify checksum if enabled
        if self.config.verify_checksums {
            let calculated_checksum = crc32fast::hash(&final_data);
            if calculated_checksum != header.checksum {
                return Err(trustformers_core::errors::invalid_format(
                    header.checksum.to_string(),
                    calculated_checksum.to_string(),
                ));
            }
        }

        // Deserialize tokenizer
        let (tokenizer, _): (BinaryTokenizer, usize) =
            oxicode::serde::decode_from_slice(&final_data, oxicode::config::standard()).map_err(
                |e| {
                    TrustformersError::serialization_error(format!(
                        "Failed to deserialize tokenizer: {}",
                        e
                    ))
                },
            )?;

        Ok((tokenizer, header))
    }

    /// Compress data using zlib
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use oxiarc_deflate::streaming::ZlibStreamEncoder;

        let mut encoder = ZlibStreamEncoder::new(Vec::new(), self.config.compression_level);
        encoder.write_all(data).map_err(|e| {
            TrustformersError::other(anyhow::anyhow!("Failed to compress data: {}", e).to_string())
        })?;
        encoder.finish().map_err(|e| {
            TrustformersError::other(
                anyhow::anyhow!("Failed to finish compression: {}", e).to_string(),
            )
        })
    }

    /// Decompress data using zlib
    fn decompress_data(&self, compressed_data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
        use oxiarc_deflate::streaming::ZlibStreamDecoder;

        let mut decoder = ZlibStreamDecoder::new(compressed_data);
        let mut decompressed = Vec::with_capacity(expected_size);
        decoder.read_to_end(&mut decompressed).map_err(|e| {
            TrustformersError::other(
                anyhow::anyhow!("Failed to decompress data: {}", e).to_string(),
            )
        })?;

        if decompressed.len() != expected_size {
            return Err(TrustformersError::other(
                anyhow::anyhow!(
                    "Decompressed size mismatch: expected {}, got {}",
                    expected_size,
                    decompressed.len()
                )
                .to_string(),
            ));
        }

        Ok(decompressed)
    }

    /// Get file info without fully loading the tokenizer
    pub fn get_file_info<P: AsRef<Path>>(&self, path: P) -> Result<BinaryHeader> {
        let file = File::open(path.as_ref())
            .map_err(|e| TrustformersError::io_error(format!("Failed to open file: {}", e)))?;
        let mut reader = BufReader::new(file);

        // Read and verify magic bytes
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| {
            TrustformersError::io_error(format!("Failed to read magic bytes: {}", e))
        })?;

        if magic != MAGIC_BYTES {
            return Err(trustformers_core::errors::invalid_format(
                "TFMT",
                String::from_utf8_lossy(&magic).to_string(),
            ));
        }

        // Read header size
        let mut header_size_bytes = [0u8; 4];
        reader.read_exact(&mut header_size_bytes).map_err(|e| {
            TrustformersError::io_error(format!("Failed to read header size: {}", e))
        })?;
        let header_size = u32::from_le_bytes(header_size_bytes) as usize;

        // Read header
        let mut header_data = vec![0u8; header_size];
        reader
            .read_exact(&mut header_data)
            .map_err(|e| TrustformersError::io_error(format!("Failed to read header: {}", e)))?;

        let (header, _): (BinaryHeader, usize) = oxicode::serde::decode_from_slice(
            &header_data,
            oxicode::config::standard(),
        )
        .map_err(|e| {
            TrustformersError::serialization_error(format!("Failed to deserialize header: {}", e))
        })?;

        Ok(header)
    }
}

/// Utilities for working with binary tokenizer files
pub struct BinaryUtils;

impl BinaryUtils {
    /// Validate a binary tokenizer file
    pub fn validate_file<P: AsRef<Path>>(path: P, config: &BinaryConfig) -> Result<bool> {
        let serializer = BinarySerializer::new(config.clone());
        let header = serializer.get_file_info(path.as_ref())?;

        // Basic validation checks
        if header.version > BINARY_FORMAT_VERSION {
            return Ok(false);
        }

        if header.compressed_size == 0 || header.uncompressed_size == 0 {
            return Ok(false);
        }

        Ok(true)
    }

    /// Compare two binary tokenizer files
    pub fn compare_files<P: AsRef<Path>>(
        path1: P,
        path2: P,
        config: &BinaryConfig,
    ) -> Result<bool> {
        let serializer = BinarySerializer::new(config.clone());

        let header1 = serializer.get_file_info(path1.as_ref())?;
        let header2 = serializer.get_file_info(path2.as_ref())?;

        // Compare checksums for quick comparison
        Ok(header1.checksum == header2.checksum)
    }

    /// Get compression ratio for a binary file
    pub fn get_compression_ratio<P: AsRef<Path>>(path: P, config: &BinaryConfig) -> Result<f64> {
        let serializer = BinarySerializer::new(config.clone());
        let header = serializer.get_file_info(path)?;

        if header.compression_level == 0 {
            return Ok(1.0);
        }

        Ok(header.uncompressed_size as f64 / header.compressed_size as f64)
    }

    /// Migrate an old format file to the current format
    pub fn migrate_format<P: AsRef<Path>>(
        old_path: P,
        new_path: P,
        config: &BinaryConfig,
    ) -> Result<BinaryHeader> {
        let serializer = BinarySerializer::new(config.clone());

        // Load the old format
        let (tokenizer, old_header) = serializer.deserialize(old_path)?;

        // Determine tokenizer type from old header or infer it
        let tokenizer_type = &old_header.tokenizer_type;

        // Save in new format
        serializer.serialize(&tokenizer, tokenizer_type, new_path)
    }
}

/// Converter for converting existing tokenizers to binary format
pub struct TokenizerConverter;

impl TokenizerConverter {
    /// Convert a HuggingFace tokenizer.json to binary format.
    ///
    /// Handles both vocabulary shapes HF's tokenizer.json uses (BPE/WordPiece
    /// store `model.vocab` as a `{token: id}` object; Unigram stores it as an
    /// array of `[piece, score]` pairs, with the array index as the id) and
    /// both merge-list encodings (`tokenizers` < 0.20 wrote `"a b"` strings;
    /// this workspace's `tokenizers` 0.23 writes `["a", "b"]` pairs). A
    /// genuinely malformed or unsupported shape is reported as an error
    /// rather than silently producing an empty or partial vocabulary/merge
    /// list. `normalizer`/`pre_tokenizer`/`post_processor`/`decoder`/
    /// `truncation`/`padding` are carried into `BinaryTokenizer::config`
    /// verbatim (as raw JSON) rather than dropped: this crate's
    /// [`NormalizationRule`]/[`PreTokenizationRule`] are a much simpler,
    /// crate-specific rule format, not a structural mirror of HF's
    /// normalizer/pre-tokenizer graph, so preserving the source JSON
    /// losslessly is more honest than a lossy best-effort translation.
    pub fn from_tokenizer_json<P: AsRef<Path>>(
        json_path: P,
        binary_path: P,
        config: &BinaryConfig,
    ) -> Result<BinaryHeader> {
        // Load the JSON tokenizer
        let json_content = std::fs::read_to_string(json_path.as_ref())
            .map_err(|e| TrustformersError::io_error(format!("Failed to read JSON file: {}", e)))?;

        let json_value: serde_json::Value = serde_json::from_str(&json_content).map_err(|e| {
            TrustformersError::serialization_error(format!("Failed to parse JSON: {}", e))
        })?;

        let model = json_value.get("model").ok_or_else(|| {
            TrustformersError::invalid_config(
                "tokenizer.json is missing the required \"model\" field".to_string(),
            )
        })?;

        // Extract vocabulary (and, for Unigram, per-piece scores).
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut scores: Option<HashMap<u32, f32>> = None;

        match model.get("vocab") {
            Some(serde_json::Value::Object(vocab_map)) => {
                // BPE / WordPiece: {token: id}.
                for (token, id) in vocab_map {
                    let id_u32 = id.as_u64().ok_or_else(|| {
                        TrustformersError::invalid_config(format!(
                            "tokenizer.json: vocab entry {:?} has a non-integer id",
                            token
                        ))
                    })? as u32;
                    vocab.insert(token.clone(), id_u32);
                    id_to_token.insert(id_u32, token.clone());
                }
            },
            Some(serde_json::Value::Array(entries)) => {
                // Unigram: [[piece, score], ...]; id = array index.
                let mut piece_scores = HashMap::with_capacity(entries.len());
                for (index, entry) in entries.iter().enumerate() {
                    let pair = entry.as_array().ok_or_else(|| {
                        TrustformersError::invalid_config(format!(
                            "tokenizer.json: Unigram vocab entry {} is not a [piece, score] pair",
                            index
                        ))
                    })?;
                    let piece = pair.first().and_then(|v| v.as_str()).ok_or_else(|| {
                        TrustformersError::invalid_config(format!(
                            "tokenizer.json: Unigram vocab entry {} is missing a string piece",
                            index
                        ))
                    })?;
                    let score = pair.get(1).and_then(|v| v.as_f64()).ok_or_else(|| {
                        TrustformersError::invalid_config(format!(
                            "tokenizer.json: Unigram vocab entry {} is missing a numeric score",
                            index
                        ))
                    })? as f32;

                    let id = index as u32;
                    vocab.insert(piece.to_string(), id);
                    id_to_token.insert(id, piece.to_string());
                    piece_scores.insert(id, score);
                }
                scores = Some(piece_scores);
            },
            Some(other) => {
                return Err(TrustformersError::invalid_config(format!(
                    "tokenizer.json: unsupported \"model.vocab\" shape: {}",
                    other
                )));
            },
            None => {
                return Err(TrustformersError::invalid_config(
                    "tokenizer.json: \"model.vocab\" is missing".to_string(),
                ));
            },
        }

        // Extract special tokens, preserving the full `added_tokens` entries
        // (including the lstrip/rstrip/single_word/normalized/special flags
        // the old code discarded) in `config` for lossless round-tripping.
        let mut special_tokens = HashMap::new();
        let mut added_tokens_meta = Vec::new();
        if let Some(added_tokens) = json_value.get("added_tokens").and_then(|v| v.as_array()) {
            for token_obj in added_tokens {
                let content = token_obj.get("content").and_then(|v| v.as_str());
                let id = token_obj.get("id").and_then(|v| v.as_u64());
                if let (Some(token_str), Some(id_num)) = (content, id) {
                    special_tokens.insert(token_str.to_string(), id_num as u32);
                }
                added_tokens_meta.push(token_obj.clone());
            }
        }

        // Extract merges for BPE, supporting both encodings tokenizer.json
        // has used across `tokenizers` versions.
        let merges = match model.get("merges") {
            Some(serde_json::Value::Array(merges_vec)) => {
                let mut extracted_merges = Vec::with_capacity(merges_vec.len());
                for (index, merge) in merges_vec.iter().enumerate() {
                    let pair = match merge {
                        serde_json::Value::String(merge_str) => {
                            let parts: Vec<&str> = merge_str.split(' ').collect();
                            if parts.len() != 2 {
                                return Err(TrustformersError::invalid_config(format!(
                                    "tokenizer.json: merge entry {} {:?} is not \"a b\"",
                                    index, merge_str
                                )));
                            }
                            (parts[0].to_string(), parts[1].to_string())
                        },
                        serde_json::Value::Array(pair_arr) if pair_arr.len() == 2 => {
                            let a = pair_arr[0].as_str().ok_or_else(|| {
                                TrustformersError::invalid_config(format!(
                                    "tokenizer.json: merge entry {} has a non-string element",
                                    index
                                ))
                            })?;
                            let b = pair_arr[1].as_str().ok_or_else(|| {
                                TrustformersError::invalid_config(format!(
                                    "tokenizer.json: merge entry {} has a non-string element",
                                    index
                                ))
                            })?;
                            (a.to_string(), b.to_string())
                        },
                        other => {
                            return Err(TrustformersError::invalid_config(format!(
                                "tokenizer.json: unsupported merge entry {} shape: {}",
                                index, other
                            )));
                        },
                    };
                    extracted_merges.push(pair);
                }
                Some(extracted_merges)
            },
            Some(other) => {
                return Err(TrustformersError::invalid_config(format!(
                    "tokenizer.json: unsupported \"model.merges\" shape: {}",
                    other
                )));
            },
            None => None,
        };

        // Preserve normalizer/pre_tokenizer/post_processor/decoder/
        // truncation/padding verbatim instead of silently dropping them.
        let mut config_map = HashMap::new();
        for key in [
            "normalizer",
            "pre_tokenizer",
            "post_processor",
            "decoder",
            "truncation",
            "padding",
        ] {
            if let Some(value) = json_value.get(key) {
                if !value.is_null() {
                    config_map.insert(key.to_string(), value.clone());
                }
            }
        }
        if !added_tokens_meta.is_empty() {
            config_map.insert(
                "added_tokens".to_string(),
                serde_json::Value::Array(added_tokens_meta),
            );
        }

        // Create binary tokenizer
        let binary_tokenizer = BinaryTokenizer {
            vocab,
            id_to_token,
            special_tokens,
            scores,
            merges,
            config: config_map,
            normalization_rules: None,
            pre_tokenization_rules: None,
        };

        // Determine tokenizer type
        let tokenizer_type = model.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");

        // Serialize to binary format
        let serializer = BinarySerializer::new(config.clone());
        serializer.serialize(&binary_tokenizer, tokenizer_type, binary_path)
    }

    /// Convert from SentencePiece model to binary format
    pub fn from_sentencepiece<P: AsRef<Path>>(
        sp_path: P,
        binary_path: P,
        config: &BinaryConfig,
    ) -> Result<BinaryHeader> {
        let sp_path = sp_path.as_ref();

        // Load SentencePiece model
        let (vocab, id_to_token, special_tokens, scores, sp_config) =
            Self::load_sentencepiece_model(sp_path)?;

        // Derive rules from whatever the model actually told us (protobuf
        // models carry `normalizer_spec`; plain-text `.vocab` files carry no
        // such metadata at all, so no rules are asserted for those instead
        // of assuming HF/SentencePiece defaults that may not hold).
        let normalization_rules = Self::extract_normalization_rules(&sp_config);
        let pre_tokenization_rules = Self::extract_pre_tokenization_rules(&sp_config);

        // Create binary tokenizer with loaded data
        let binary_tokenizer = BinaryTokenizer {
            vocab,
            id_to_token,
            special_tokens,
            scores: Some(scores),
            merges: None, // SentencePiece doesn't use BPE merges
            config: sp_config
                .into_iter()
                .map(|(k, v)| (k, serde_json::Value::String(v.to_string())))
                .collect(),
            normalization_rules: Some(normalization_rules),
            pre_tokenization_rules: Some(pre_tokenization_rules),
        };

        let serializer = BinarySerializer::new(config.clone());
        serializer.serialize(&binary_tokenizer, "sentencepiece", binary_path)
    }

    /// Load SentencePiece model from file
    fn load_sentencepiece_model<P: AsRef<Path>>(
        sp_path: P,
    ) -> Result<(
        HashMap<String, u32>,
        HashMap<u32, String>,
        HashMap<String, u32>,
        HashMap<u32, f32>,
        HashMap<String, String>,
    )> {
        let sp_path = sp_path.as_ref();

        // Check if it's a protobuf file (.model) or text file (.vocab)
        if sp_path.extension().and_then(|s| s.to_str()) == Some("model") {
            Self::load_sentencepiece_protobuf(sp_path)
        } else {
            Self::load_sentencepiece_vocab(sp_path)
        }
    }

    /// Load SentencePiece protobuf model file
    fn load_sentencepiece_protobuf<P: AsRef<Path>>(
        model_path: P,
    ) -> Result<(
        HashMap<String, u32>,
        HashMap<u32, String>,
        HashMap<String, u32>,
        HashMap<u32, f32>,
        HashMap<String, String>,
    )> {
        let mut file = File::open(model_path).map_err(|e| {
            TrustformersError::other(
                anyhow!("Failed to open SentencePiece model file: {}", e).to_string(),
            )
        })?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(|e| {
            TrustformersError::other(
                anyhow!("Failed to read SentencePiece model file: {}", e).to_string(),
            )
        })?;

        Self::parse_sentencepiece_protobuf(&buffer)
    }

    /// Parse a real SentencePiece `ModelProto` (see
    /// [`crate::sentencepiece::proto`]) into the binary format's flat
    /// vocab/score/config representation. Every piece, score, and type comes
    /// directly from the model file's wire-format bytes; nothing is
    /// synthesized, and a malformed or truncated file is rejected rather
    /// than heuristically scanned for UTF-8-looking runs.
    fn parse_sentencepiece_protobuf(
        data: &[u8],
    ) -> Result<(
        HashMap<String, u32>,
        HashMap<u32, String>,
        HashMap<String, u32>,
        HashMap<u32, f32>,
        HashMap<String, String>,
    )> {
        let model = crate::sentencepiece::proto::parse_model_proto(data)?;

        let mut vocab = HashMap::with_capacity(model.pieces.len());
        let mut id_to_token = HashMap::with_capacity(model.pieces.len());
        let mut special_tokens = HashMap::new();
        let mut scores = HashMap::with_capacity(model.pieces.len());

        // SentencePiece IDs are the piece's index within `ModelProto.pieces`.
        for (index, piece) in model.pieces.iter().enumerate() {
            let id = index as u32;
            vocab.insert(piece.piece.clone(), id);
            id_to_token.insert(id, piece.piece.clone());
            scores.insert(id, piece.score);

            if matches!(
                piece.piece_type,
                crate::sentencepiece::proto::PieceType::Unknown
                    | crate::sentencepiece::proto::PieceType::Control
                    | crate::sentencepiece::proto::PieceType::UserDefined
            ) {
                special_tokens.insert(piece.piece.clone(), id);
            }
        }

        let mut config = HashMap::new();
        config.insert(
            "model_type".to_string(),
            sentencepiece_model_type_name(model.trainer_spec.model_type).to_string(),
        );
        config.insert("vocab_size".to_string(), vocab.len().to_string());
        config.insert(
            "normalization".to_string(),
            if model.normalizer_spec.name.is_empty() {
                "identity".to_string()
            } else {
                model.normalizer_spec.name.clone()
            },
        );
        config.insert(
            "add_dummy_prefix".to_string(),
            model.normalizer_spec.add_dummy_prefix.to_string(),
        );
        config.insert(
            "remove_extra_whitespaces".to_string(),
            model.normalizer_spec.remove_extra_whitespaces.to_string(),
        );
        config.insert(
            "escape_whitespaces".to_string(),
            model.normalizer_spec.escape_whitespaces.to_string(),
        );
        config.insert(
            "byte_fallback".to_string(),
            model.trainer_spec.byte_fallback.to_string(),
        );
        config.insert(
            "treat_whitespace_as_suffix".to_string(),
            model.trainer_spec.treat_whitespace_as_suffix.to_string(),
        );

        Ok((vocab, id_to_token, special_tokens, scores, config))
    }

    /// Load a SentencePiece plain-text `.vocab` file (one `piece<TAB>score`
    /// entry per line, the format `spm_export_vocab` produces). Every score
    /// comes from the file itself; a line with a missing or unparsable score
    /// is rejected rather than silently defaulted to a heuristic estimate.
    fn load_sentencepiece_vocab<P: AsRef<Path>>(
        vocab_path: P,
    ) -> Result<(
        HashMap<String, u32>,
        HashMap<u32, String>,
        HashMap<String, u32>,
        HashMap<u32, f32>,
        HashMap<String, String>,
    )> {
        let file = File::open(vocab_path).map_err(|e| {
            TrustformersError::other(
                anyhow!("Failed to open SentencePiece vocab file: {}", e).to_string(),
            )
        })?;
        let reader = BufReader::new(file);

        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut special_tokens = HashMap::new();
        let mut scores = HashMap::new();
        let mut config = HashMap::new();
        // IDs are assigned by *vocabulary entry* order, not by source line
        // number: the old code used `line_num` directly, which silently
        // mis-numbered every entry after the first skipped blank/comment
        // line.
        let mut next_id: u32 = 0;

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                TrustformersError::other(
                    anyhow!("Failed to read line {}: {}", line_num + 1, e).to_string(),
                )
            })?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Format: "<piece>\t<score>" (spm_export_vocab's output); a few
            // hand-edited vocab files use plain whitespace instead of a tab.
            let parts: Vec<&str> = if line.contains('\t') {
                line.splitn(2, '\t').collect()
            } else {
                line.splitn(2, char::is_whitespace).collect()
            };

            let token = parts[0].to_string();
            let score = match parts.get(1) {
                Some(raw) => raw.trim().parse::<f32>().map_err(|e| {
                    TrustformersError::other(format!(
                        "SentencePiece vocab line {}: invalid score {:?} for piece {:?}: {}",
                        line_num + 1,
                        raw,
                        token,
                        e
                    ))
                })?,
                None => {
                    return Err(TrustformersError::other(format!(
                        "SentencePiece vocab line {}: missing score column for piece {:?}",
                        line_num + 1,
                        token
                    )));
                },
            };

            let id = next_id;
            next_id += 1;
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
            scores.insert(id, score);

            // Identify special tokens
            if token.starts_with('<') && token.ends_with('>') {
                special_tokens.insert(token, id);
            }
        }

        if vocab.is_empty() {
            return Err(TrustformersError::other(
                "SentencePiece vocab file contains no entries".to_string(),
            ));
        }

        // Plain-text `.vocab` files carry no normalizer/trainer metadata, so
        // (unlike the protobuf path) only the fields we can actually know
        // are populated; `extract_normalization_rules`/
        // `extract_pre_tokenization_rules` correctly emit no rules at all
        // for a config missing these keys rather than assuming defaults.
        config.insert("model_type".to_string(), "sentencepiece".to_string());
        config.insert("vocab_size".to_string(), vocab.len().to_string());

        Ok((vocab, id_to_token, special_tokens, scores, config))
    }

    /// Derive normalization rules from a SentencePiece model's own
    /// `normalizer_spec` config (as captured by [`Self::load_sentencepiece_model`]),
    /// instead of asserting a fixed NFKC/whitespace policy regardless of
    /// what the model actually specifies.
    fn extract_normalization_rules(config: &HashMap<String, String>) -> Vec<NormalizationRule> {
        let mut rules = Vec::new();

        if let Some(name) = config.get("normalization") {
            if name != "identity" {
                rules.push(NormalizationRule {
                    rule_type: name.to_uppercase(),
                    parameters: HashMap::new(),
                });
            }
        }

        if config.get("remove_extra_whitespaces").map(String::as_str) == Some("true") {
            rules.push(NormalizationRule {
                rule_type: "RemoveExtraSpaces".to_string(),
                parameters: {
                    let mut params = HashMap::new();
                    params.insert(
                        "pattern".to_string(),
                        serde_json::Value::String(r"\s+".to_string()),
                    );
                    params.insert(
                        "replacement".to_string(),
                        serde_json::Value::String(" ".to_string()),
                    );
                    params.insert("regex".to_string(), serde_json::Value::Bool(true));
                    params
                },
            });
        }

        rules
    }

    /// Derive pre-tokenization rules from a SentencePiece model's own
    /// `normalizer_spec` config, instead of asserting the dummy-prefix and
    /// whitespace-escaping rules unconditionally.
    fn extract_pre_tokenization_rules(
        config: &HashMap<String, String>,
    ) -> Vec<PreTokenizationRule> {
        let mut rules = Vec::new();

        if config.get("add_dummy_prefix").map(String::as_str) == Some("true") {
            rules.push(PreTokenizationRule {
                rule_type: "AddDummyPrefix".to_string(),
                pattern: "^".to_string(),
                replacement: Some("▁".to_string()),
            });
        }

        if config.get("escape_whitespaces").map(String::as_str) == Some("true") {
            rules.push(PreTokenizationRule {
                rule_type: "SpaceReplacement".to_string(),
                pattern: " ".to_string(),
                replacement: Some("▁".to_string()),
            });
        }

        rules
    }
}

/// Human-readable name for a SentencePiece `TrainerSpec.model_type` wire
/// value (`sentencepiece_model.proto`'s `ModelType` enum: `1 = UNIGRAM, 2 =
/// BPE, 3 = WORD, 4 = CHAR`).
fn sentencepiece_model_type_name(model_type: u64) -> &'static str {
    match model_type {
        1 => "unigram",
        2 => "bpe",
        3 => "word",
        4 => "char",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_tokenizer() -> BinaryTokenizer {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut special_tokens = HashMap::new();

        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("<pad>".to_string(), 2);

        id_to_token.insert(0, "hello".to_string());
        id_to_token.insert(1, "world".to_string());
        id_to_token.insert(2, "<pad>".to_string());

        special_tokens.insert("<pad>".to_string(), 2);

        BinaryTokenizer {
            vocab,
            id_to_token,
            special_tokens,
            scores: None,
            merges: None,
            config: HashMap::new(),
            normalization_rules: None,
            pre_tokenization_rules: None,
        }
    }

    #[test]
    fn test_serialize_deserialize() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let file_path = temp_dir.path().join("test_tokenizer.bin");

        let config = BinaryConfig::default();
        let serializer = BinarySerializer::new(config);

        let tokenizer = create_test_tokenizer();

        // Serialize
        let header = serializer
            .serialize(&tokenizer, "test", &file_path)
            .expect("Operation failed in test");
        assert_eq!(header.tokenizer_type, "test");
        assert_eq!(header.version, BINARY_FORMAT_VERSION);

        // Deserialize
        let (loaded_tokenizer, loaded_header) =
            serializer.deserialize(&file_path).expect("Operation failed in test");

        assert_eq!(loaded_tokenizer.vocab, tokenizer.vocab);
        assert_eq!(loaded_tokenizer.id_to_token, tokenizer.id_to_token);
        assert_eq!(loaded_header.tokenizer_type, "test");
    }

    #[test]
    fn test_compression() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let file_path = temp_dir.path().join("test_compressed.bin");

        let config = BinaryConfig {
            compression_level: 9,
            ..Default::default()
        };
        let serializer = BinarySerializer::new(config);

        let tokenizer = create_test_tokenizer();
        let header = serializer
            .serialize(&tokenizer, "test", &file_path)
            .expect("Operation failed in test");

        assert!(header.compressed_size < header.uncompressed_size);
        assert_eq!(header.compression_level, 9);

        // Should still deserialize correctly
        let (loaded_tokenizer, _) =
            serializer.deserialize(&file_path).expect("Operation failed in test");
        assert_eq!(loaded_tokenizer.vocab, tokenizer.vocab);
    }

    #[test]
    fn test_file_info() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let file_path = temp_dir.path().join("test_info.bin");

        let config = BinaryConfig::default();
        let serializer = BinarySerializer::new(config);

        let tokenizer = create_test_tokenizer();
        let original_header = serializer
            .serialize(&tokenizer, "test", &file_path)
            .expect("Operation failed in test");

        // Get file info without loading
        let info_header = serializer.get_file_info(&file_path).expect("Operation failed in test");

        assert_eq!(info_header.tokenizer_type, original_header.tokenizer_type);
        assert_eq!(info_header.checksum, original_header.checksum);
    }

    #[test]
    fn test_validation() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let file_path = temp_dir.path().join("test_validate.bin");

        let config = BinaryConfig::default();
        let serializer = BinarySerializer::new(config.clone());

        let tokenizer = create_test_tokenizer();
        serializer
            .serialize(&tokenizer, "test", &file_path)
            .expect("Operation failed in test");

        assert!(BinaryUtils::validate_file(&file_path, &config).expect("Operation failed in test"));
    }

    #[test]
    fn test_compression_ratio() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let file_path = temp_dir.path().join("test_ratio.bin");

        let config = BinaryConfig {
            compression_level: 6,
            ..Default::default()
        };
        let serializer = BinarySerializer::new(config.clone());

        let tokenizer = create_test_tokenizer();
        serializer
            .serialize(&tokenizer, "test", &file_path)
            .expect("Operation failed in test");

        let ratio = BinaryUtils::get_compression_ratio(&file_path, &config)
            .expect("Operation failed in test");
        assert!(ratio > 1.0); // Should have some compression
    }

    /// Regression test: a modern (`tokenizers` >= 0.20) BPE tokenizer.json
    /// encodes each merge as a 2-element array, not a space-joined string.
    /// The old parser only handled the string form, so `merges` silently
    /// came back empty for every current-format file.
    #[test]
    fn test_from_tokenizer_json_bpe_with_modern_array_merges() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let json_path = temp_dir.path().join("tokenizer.json");
        let binary_path = temp_dir.path().join("tokenizer.bin");

        std::fs::write(
            &json_path,
            serde_json::json!({
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1, "ab": 2},
                    "merges": [["a", "b"]]
                }
            })
            .to_string(),
        )
        .expect("Operation failed in test");

        let header = TokenizerConverter::from_tokenizer_json(
            &json_path,
            &binary_path,
            &BinaryConfig::default(),
        )
        .expect("Operation failed in test");
        assert_eq!(header.tokenizer_type, "BPE");

        let serializer = BinarySerializer::new(BinaryConfig::default());
        let (tokenizer, _) =
            serializer.deserialize(&binary_path).expect("Operation failed in test");
        assert_eq!(tokenizer.vocab.get("ab"), Some(&2));
        assert_eq!(
            tokenizer.merges,
            Some(vec![("a".to_string(), "b".to_string())]),
            "modern array-encoded merges must not be silently dropped"
        );
    }

    /// The legacy `tokenizers` < 0.20 space-joined-string merge encoding
    /// must still work.
    #[test]
    fn test_from_tokenizer_json_bpe_with_legacy_string_merges() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let json_path = temp_dir.path().join("tokenizer.json");
        let binary_path = temp_dir.path().join("tokenizer.bin");

        std::fs::write(
            &json_path,
            serde_json::json!({
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1, "ab": 2},
                    "merges": ["a b"]
                }
            })
            .to_string(),
        )
        .expect("Operation failed in test");

        TokenizerConverter::from_tokenizer_json(&json_path, &binary_path, &BinaryConfig::default())
            .expect("Operation failed in test");

        let serializer = BinarySerializer::new(BinaryConfig::default());
        let (tokenizer, _) =
            serializer.deserialize(&binary_path).expect("Operation failed in test");
        assert_eq!(
            tokenizer.merges,
            Some(vec![("a".to_string(), "b".to_string())])
        );
    }

    /// Regression test: Unigram tokenizer.json stores `model.vocab` as an
    /// array of `[piece, score]` pairs, not an object. The old parser only
    /// handled the object shape, so `vocab_obj.as_object()` returned `None`
    /// and the conversion silently produced an *empty* vocabulary while
    /// still returning `Ok`.
    #[test]
    fn test_from_tokenizer_json_unigram_array_vocab_is_not_empty() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let json_path = temp_dir.path().join("tokenizer.json");
        let binary_path = temp_dir.path().join("tokenizer.bin");

        std::fs::write(
            &json_path,
            serde_json::json!({
                "model": {
                    "type": "Unigram",
                    "vocab": [["<unk>", 0.0], ["▁the", -1.5], ["▁a", -2.5]]
                }
            })
            .to_string(),
        )
        .expect("Operation failed in test");

        TokenizerConverter::from_tokenizer_json(&json_path, &binary_path, &BinaryConfig::default())
            .expect("Operation failed in test");

        let serializer = BinarySerializer::new(BinaryConfig::default());
        let (tokenizer, _) =
            serializer.deserialize(&binary_path).expect("Operation failed in test");

        assert_eq!(
            tokenizer.vocab.len(),
            3,
            "Unigram array vocab must not come back empty"
        );
        assert_eq!(tokenizer.vocab.get("<unk>"), Some(&0));
        assert_eq!(tokenizer.vocab.get("\u{2581}the"), Some(&1));
        let scores = tokenizer.scores.expect("Unigram pieces must carry their real scores");
        assert!((scores[&1] - (-1.5)).abs() < 1e-6);
    }

    /// A `model.vocab` that is neither an object nor an array is a genuinely
    /// malformed file and must be rejected, not silently treated as empty.
    #[test]
    fn test_from_tokenizer_json_malformed_vocab_shape_is_error() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let json_path = temp_dir.path().join("tokenizer.json");
        let binary_path = temp_dir.path().join("tokenizer.bin");

        std::fs::write(
            &json_path,
            serde_json::json!({ "model": { "type": "BPE", "vocab": "not-a-vocab" } }).to_string(),
        )
        .expect("Operation failed in test");

        let result = TokenizerConverter::from_tokenizer_json(
            &json_path,
            &binary_path,
            &BinaryConfig::default(),
        );
        assert!(result.is_err());
    }

    /// `normalizer`/`pre_tokenizer` (and friends) must be preserved, not
    /// silently discarded, even though this crate's own
    /// `NormalizationRule`/`PreTokenizationRule` types don't structurally
    /// mirror HF's schema.
    #[test]
    fn test_from_tokenizer_json_preserves_normalizer_and_pre_tokenizer() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let json_path = temp_dir.path().join("tokenizer.json");
        let binary_path = temp_dir.path().join("tokenizer.bin");

        std::fs::write(
            &json_path,
            serde_json::json!({
                "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
                "normalizer": {"type": "NFKC"},
                "pre_tokenizer": {"type": "Whitespace"},
                "truncation": {"max_length": 512}
            })
            .to_string(),
        )
        .expect("Operation failed in test");

        TokenizerConverter::from_tokenizer_json(&json_path, &binary_path, &BinaryConfig::default())
            .expect("Operation failed in test");

        let serializer = BinarySerializer::new(BinaryConfig::default());
        let (tokenizer, _) =
            serializer.deserialize(&binary_path).expect("Operation failed in test");

        assert_eq!(
            tokenizer.config.get("normalizer"),
            Some(&serde_json::json!({"type": "NFKC"}))
        );
        assert_eq!(
            tokenizer.config.get("pre_tokenizer"),
            Some(&serde_json::json!({"type": "Whitespace"}))
        );
        assert_eq!(
            tokenizer.config.get("truncation"),
            Some(&serde_json::json!({"max_length": 512}))
        );
    }

    /// Regression test for the fabricated SentencePiece protobuf "parser":
    /// the old code never actually parsed protobuf -- it byte-scanned for
    /// UTF-8-looking runs and invented scores from a hardcoded
    /// length/prefix heuristic (`estimate_token_score`). A real
    /// `ModelProto` must now round-trip its exact pieces, scores, and
    /// types.
    #[test]
    fn test_from_sentencepiece_parses_real_protobuf_not_fabricated_scores() {
        use crate::sentencepiece::proto::encode::*;

        let mut model_bytes = Vec::new();
        model_bytes.extend(length_delimited(1, &piece("<unk>", -100.0, 1)));
        model_bytes.extend(length_delimited(1, &piece("\u{2581}hello", -3.5, 0)));
        model_bytes.extend(length_delimited(1, &piece("<s>", 0.0, 2)));

        let temp_dir = tempdir().expect("Operation failed in test");
        let model_path = temp_dir.path().join("spm.model");
        let binary_path = temp_dir.path().join("spm.bin");
        std::fs::write(&model_path, &model_bytes).expect("Operation failed in test");

        TokenizerConverter::from_sentencepiece(&model_path, &binary_path, &BinaryConfig::default())
            .expect("Operation failed in test");

        let serializer = BinarySerializer::new(BinaryConfig::default());
        let (tokenizer, _) =
            serializer.deserialize(&binary_path).expect("Operation failed in test");

        assert_eq!(tokenizer.vocab.len(), 3);
        // IDs are the piece's index in the model, not a scan-order counter
        // starting at 4.
        assert_eq!(tokenizer.vocab.get("<unk>"), Some(&0));
        assert_eq!(tokenizer.vocab.get("\u{2581}hello"), Some(&1));
        assert_eq!(tokenizer.vocab.get("<s>"), Some(&2));

        let scores = tokenizer.scores.expect("scores must be populated from the model");
        // Real scores from the model, not `estimate_token_score`'s invented
        // `-2.0` (single char) / `-5.0 + len*-0.1` heuristic.
        assert!((scores[&0] - (-100.0)).abs() < 1e-6);
        assert!((scores[&1] - (-3.5)).abs() < 1e-6);
        assert!((scores[&2] - 0.0).abs() < 1e-6);

        // PieceType::Unknown (<unk>) and Control (<s>) become special
        // tokens; the plain Normal piece does not.
        assert_eq!(tokenizer.special_tokens.get("<unk>"), Some(&0));
        assert_eq!(tokenizer.special_tokens.get("<s>"), Some(&2));
        assert!(!tokenizer.special_tokens.contains_key("\u{2581}hello"));
    }

    /// A `.model` file that is not valid protobuf at all (e.g. random or
    /// truncated bytes) must be rejected outright, matching the old
    /// heuristic scanner's replacement contract: no silent partial vocab.
    #[test]
    fn test_from_sentencepiece_rejects_garbage_bytes() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let model_path = temp_dir.path().join("garbage.model");
        let binary_path = temp_dir.path().join("garbage.bin");
        // Field number 0 (top 5 bits of the first byte) is not valid
        // protobuf; the old heuristic scanner would have happily "found"
        // fake tokens in this data instead of rejecting it.
        std::fs::write(&model_path, [0x01u8, 0x00, 0xFF, 0xFE]).expect("Operation failed in test");

        let result = TokenizerConverter::from_sentencepiece(
            &model_path,
            &binary_path,
            &BinaryConfig::default(),
        );
        assert!(result.is_err());
    }

    /// Regression test for the `.vocab` text-format loader: a missing or
    /// malformed score column must be an error, not silently defaulted to
    /// `0.0` (missing) or routed through the deleted `estimate_token_score`
    /// heuristic (missing column).
    #[test]
    fn test_from_sentencepiece_vocab_file_rejects_malformed_score() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let vocab_path = temp_dir.path().join("spm.vocab");
        let binary_path = temp_dir.path().join("spm.bin");
        std::fs::write(&vocab_path, "<unk>\t-100.0\nhello\tnot-a-number\n")
            .expect("Operation failed in test");

        let result = TokenizerConverter::from_sentencepiece(
            &vocab_path,
            &binary_path,
            &BinaryConfig::default(),
        );
        assert!(result.is_err());
    }

    /// A well-formed `.vocab` file parses real scores and assigns IDs by
    /// vocabulary-entry order (skipping comments/blank lines must not skew
    /// later IDs).
    #[test]
    fn test_from_sentencepiece_vocab_file_ids_follow_entry_order() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let vocab_path = temp_dir.path().join("spm.vocab");
        let binary_path = temp_dir.path().join("spm.bin");
        std::fs::write(
            &vocab_path,
            "# a comment line\n<unk>\t-100.0\n\nhello\t-3.5\nworld\t-4.0\n",
        )
        .expect("Operation failed in test");

        TokenizerConverter::from_sentencepiece(&vocab_path, &binary_path, &BinaryConfig::default())
            .expect("Operation failed in test");

        let serializer = BinarySerializer::new(BinaryConfig::default());
        let (tokenizer, _) =
            serializer.deserialize(&binary_path).expect("Operation failed in test");
        assert_eq!(tokenizer.vocab.get("<unk>"), Some(&0));
        assert_eq!(tokenizer.vocab.get("hello"), Some(&1));
        assert_eq!(tokenizer.vocab.get("world"), Some(&2));
    }
}
