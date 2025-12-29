use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use trustformers_core::errors::{Result, TrustformersError};

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

    /// Additional tokenizer-specific configuration
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
        use flate2::write::ZlibEncoder;
        use flate2::Compression;

        let mut encoder = ZlibEncoder::new(
            Vec::new(),
            Compression::new(self.config.compression_level as u32),
        );
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
        use flate2::read::ZlibDecoder;

        let mut decoder = ZlibDecoder::new(compressed_data);
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
    /// Convert a HuggingFace tokenizer.json to binary format
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

        // Extract vocabulary
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        if let Some(model) = json_value.get("model") {
            if let Some(vocab_obj) = model.get("vocab") {
                if let Some(vocab_map) = vocab_obj.as_object() {
                    for (token, id) in vocab_map {
                        if let Some(id_num) = id.as_u64() {
                            let id_u32 = id_num as u32;
                            vocab.insert(token.clone(), id_u32);
                            id_to_token.insert(id_u32, token.clone());
                        }
                    }
                }
            }
        }

        // Extract special tokens
        let mut special_tokens = HashMap::new();
        if let Some(added_tokens) = json_value.get("added_tokens") {
            if let Some(tokens_array) = added_tokens.as_array() {
                for token_obj in tokens_array {
                    if let Some(content) = token_obj.get("content") {
                        if let Some(id) = token_obj.get("id") {
                            if let (Some(token_str), Some(id_num)) = (content.as_str(), id.as_u64())
                            {
                                special_tokens.insert(token_str.to_string(), id_num as u32);
                            }
                        }
                    }
                }
            }
        }

        // Extract merges for BPE
        let merges = if let Some(model) = json_value.get("model") {
            if let Some(merges_array) = model.get("merges") {
                if let Some(merges_vec) = merges_array.as_array() {
                    let mut extracted_merges = Vec::new();
                    for merge in merges_vec {
                        if let Some(merge_str) = merge.as_str() {
                            let parts: Vec<&str> = merge_str.split(' ').collect();
                            if parts.len() == 2 {
                                extracted_merges.push((parts[0].to_string(), parts[1].to_string()));
                            }
                        }
                    }
                    Some(extracted_merges)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Create binary tokenizer
        let binary_tokenizer = BinaryTokenizer {
            vocab,
            id_to_token,
            special_tokens,
            scores: None, // JSON tokenizers typically don't have scores
            merges,
            config: HashMap::new(),
            normalization_rules: None,
            pre_tokenization_rules: None,
        };

        // Determine tokenizer type
        let tokenizer_type = if let Some(model) = json_value.get("model") {
            if let Some(type_str) = model.get("type") {
                type_str.as_str().unwrap_or("unknown").to_string()
            } else {
                "unknown".to_string()
            }
        } else {
            "unknown".to_string()
        };

        // Serialize to binary format
        let serializer = BinarySerializer::new(config.clone());
        serializer.serialize(&binary_tokenizer, &tokenizer_type, binary_path)
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
            normalization_rules: Some(Self::extract_normalization_rules()),
            pre_tokenization_rules: Some(Self::extract_pre_tokenization_rules()),
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

        // Parse protobuf data (simplified - would use actual protobuf parsing in production)
        Self::parse_sentencepiece_protobuf(&buffer)
    }

    /// Parse SentencePiece protobuf data
    fn parse_sentencepiece_protobuf(
        data: &[u8],
    ) -> Result<(
        HashMap<String, u32>,
        HashMap<u32, String>,
        HashMap<String, u32>,
        HashMap<u32, f32>,
        HashMap<String, String>,
    )> {
        // Simplified protobuf parsing - in production this would use proper protobuf library
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut special_tokens = HashMap::new();
        let mut scores = HashMap::new();
        let mut config = HashMap::new();

        // Add standard SentencePiece tokens
        let standard_tokens = vec![
            ("<unk>", 0, -100.0, true),
            ("<s>", 1, -1.0, true),
            ("</s>", 2, -1.0, true),
            ("<pad>", 3, -1.0, true),
        ];

        for (token, id, score, is_special) in standard_tokens {
            vocab.insert(token.to_string(), id);
            id_to_token.insert(id, token.to_string());
            scores.insert(id, score);
            if is_special {
                special_tokens.insert(token.to_string(), id);
            }
        }

        // Extract vocabulary from protobuf data
        let mut current_id = 4;
        let mut i = 0;

        while i < data.len() {
            // Look for token patterns in the binary data
            if let Some(token_data) = Self::extract_token_from_protobuf(data, &mut i) {
                let (token, score) = token_data;

                if !vocab.contains_key(&token) {
                    vocab.insert(token.clone(), current_id);
                    id_to_token.insert(current_id, token.clone());
                    scores.insert(current_id, score);
                    current_id += 1;
                }
            } else {
                i += 1;
            }
        }

        // Add configuration metadata
        config.insert("model_type".to_string(), "sentencepiece".to_string());
        config.insert("vocab_size".to_string(), vocab.len().to_string());
        config.insert("normalization".to_string(), "nfkc".to_string());
        config.insert("add_dummy_prefix".to_string(), "true".to_string());

        Ok((vocab, id_to_token, special_tokens, scores, config))
    }

    /// Extract token from SentencePiece protobuf data
    fn extract_token_from_protobuf(data: &[u8], pos: &mut usize) -> Option<(String, f32)> {
        if *pos >= data.len() {
            return None;
        }

        // Simplified extraction - look for UTF-8 sequences that could be tokens
        let start = *pos;
        let mut end = start;

        // Find potential token boundaries
        while end < data.len() && end < start + 50 {
            if data[end] == 0
                || (data[end] < 32 && data[end] != 9 && data[end] != 10 && data[end] != 13)
            {
                break;
            }
            end += 1;
        }

        if end > start {
            if let Ok(token) = String::from_utf8(data[start..end].to_vec()) {
                let clean_token = token.trim().to_string();
                if !clean_token.is_empty() && Self::is_valid_token(&clean_token) {
                    *pos = end + 1;
                    // Generate a score based on token characteristics
                    let score = Self::estimate_token_score(&clean_token);
                    return Some((clean_token, score));
                }
            }
        }

        *pos += 1;
        None
    }

    /// Load SentencePiece vocabulary file
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

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                TrustformersError::other(
                    anyhow!("Failed to read line {}: {}", line_num, e).to_string(),
                )
            })?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse line format: token\tscore or token score
            let parts: Vec<&str> = if line.contains('\t') {
                line.split('\t').collect()
            } else {
                line.split_whitespace().collect()
            };

            if parts.is_empty() {
                continue;
            }

            let token = parts[0].to_string();
            let score = if parts.len() > 1 {
                parts[1].parse::<f32>().unwrap_or(0.0)
            } else {
                Self::estimate_token_score(&token)
            };

            let id = line_num as u32;
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
            scores.insert(id, score);

            // Identify special tokens
            if token.starts_with('<') && token.ends_with('>') {
                special_tokens.insert(token, id);
            }
        }

        // Add configuration
        config.insert("model_type".to_string(), "sentencepiece".to_string());
        config.insert("vocab_size".to_string(), vocab.len().to_string());
        config.insert("normalization".to_string(), "nfkc".to_string());

        Ok((vocab, id_to_token, special_tokens, scores, config))
    }

    /// Check if a token is valid
    fn is_valid_token(token: &str) -> bool {
        // Token should not be too long, not be all whitespace, and contain printable characters
        token.len() <= 100
            && !token.trim().is_empty()
            && token.chars().any(|c| !c.is_whitespace())
            && token.chars().all(|c| c.is_ascii() || c as u32 > 127) // Allow ASCII and Unicode
    }

    /// Estimate token score based on characteristics
    fn estimate_token_score(token: &str) -> f32 {
        // Estimate score based on token frequency heuristics
        match token {
            "<unk>" => -100.0,
            "<s>" | "</s>" | "<pad>" => -1.0,
            _ if token.starts_with('<') && token.ends_with('>') => -10.0, // Special tokens
            _ if token.starts_with("▁") => -5.0 + (token.len() as f32 * -0.1), // SentencePiece prefix
            _ if token.len() == 1 => -2.0,                                     // Single characters
            _ if token.len() <= 3 => -3.0 + (token.len() as f32 * -0.2),
            _ => -5.0 + (token.len() as f32 * -0.1), // Longer subwords get lower scores
        }
    }

    /// Extract normalization rules for SentencePiece
    fn extract_normalization_rules() -> Vec<NormalizationRule> {
        vec![
            NormalizationRule {
                rule_type: "NFKC".to_string(),
                parameters: {
                    let mut params = HashMap::new();
                    params.insert(
                        "pattern".to_string(),
                        serde_json::Value::String(".*".to_string()),
                    );
                    params.insert(
                        "replacement".to_string(),
                        serde_json::Value::String("NFKC_NORMALIZED".to_string()),
                    );
                    params.insert("regex".to_string(), serde_json::Value::Bool(false));
                    params
                },
            },
            NormalizationRule {
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
            },
        ]
    }

    /// Extract pre-tokenization rules for SentencePiece
    fn extract_pre_tokenization_rules() -> Vec<PreTokenizationRule> {
        vec![
            PreTokenizationRule {
                rule_type: "AddDummyPrefix".to_string(),
                pattern: "^".to_string(),
                replacement: Some("▁".to_string()),
            },
            PreTokenizationRule {
                rule_type: "SpaceReplacement".to_string(),
                pattern: " ".to_string(),
                replacement: Some("▁".to_string()),
            },
        ]
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
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_tokenizer.bin");

        let config = BinaryConfig::default();
        let serializer = BinarySerializer::new(config);

        let tokenizer = create_test_tokenizer();

        // Serialize
        let header = serializer.serialize(&tokenizer, "test", &file_path).unwrap();
        assert_eq!(header.tokenizer_type, "test");
        assert_eq!(header.version, BINARY_FORMAT_VERSION);

        // Deserialize
        let (loaded_tokenizer, loaded_header) = serializer.deserialize(&file_path).unwrap();

        assert_eq!(loaded_tokenizer.vocab, tokenizer.vocab);
        assert_eq!(loaded_tokenizer.id_to_token, tokenizer.id_to_token);
        assert_eq!(loaded_header.tokenizer_type, "test");
    }

    #[test]
    fn test_compression() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_compressed.bin");

        let config = BinaryConfig {
            compression_level: 9,
            ..Default::default()
        };
        let serializer = BinarySerializer::new(config);

        let tokenizer = create_test_tokenizer();
        let header = serializer.serialize(&tokenizer, "test", &file_path).unwrap();

        assert!(header.compressed_size < header.uncompressed_size);
        assert_eq!(header.compression_level, 9);

        // Should still deserialize correctly
        let (loaded_tokenizer, _) = serializer.deserialize(&file_path).unwrap();
        assert_eq!(loaded_tokenizer.vocab, tokenizer.vocab);
    }

    #[test]
    fn test_file_info() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_info.bin");

        let config = BinaryConfig::default();
        let serializer = BinarySerializer::new(config);

        let tokenizer = create_test_tokenizer();
        let original_header = serializer.serialize(&tokenizer, "test", &file_path).unwrap();

        // Get file info without loading
        let info_header = serializer.get_file_info(&file_path).unwrap();

        assert_eq!(info_header.tokenizer_type, original_header.tokenizer_type);
        assert_eq!(info_header.checksum, original_header.checksum);
    }

    #[test]
    fn test_validation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_validate.bin");

        let config = BinaryConfig::default();
        let serializer = BinarySerializer::new(config.clone());

        let tokenizer = create_test_tokenizer();
        serializer.serialize(&tokenizer, "test", &file_path).unwrap();

        assert!(BinaryUtils::validate_file(&file_path, &config).unwrap());
    }

    #[test]
    fn test_compression_ratio() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_ratio.bin");

        let config = BinaryConfig {
            compression_level: 6,
            ..Default::default()
        };
        let serializer = BinarySerializer::new(config.clone());

        let tokenizer = create_test_tokenizer();
        serializer.serialize(&tokenizer, "test", &file_path).unwrap();

        let ratio = BinaryUtils::get_compression_ratio(&file_path, &config).unwrap();
        assert!(ratio > 1.0); // Should have some compression
    }
}
