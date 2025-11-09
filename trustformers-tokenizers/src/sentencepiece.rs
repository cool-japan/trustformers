use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_tokens: HashMap<String, u32>,
    scores: HashMap<u32, f32>,
    pad_token_id: Option<u32>,
    unk_token_id: Option<u32>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,

    // Tokenizer configuration
    model_type: ModelType,
    normalization: bool,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    treat_whitespace_as_suffix: bool,
    byte_fallback: bool,

    // Character normalization settings
    nfc_normalization: bool,
    nfkc_normalization: bool,
    escape_whitespaces: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    Unigram,
    Bpe,
    Word,
    Char,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct SentencePieceConfig {
    #[serde(default)]
    pad_token: Option<String>,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default)]
    extra_ids: i32,
}

/// SentencePiece model representation for protobuf parsing
#[derive(Debug, Default)]
struct SentencePieceModel {
    trainer_spec: TrainerSpec,
    normalizer_spec: NormalizerSpec,
    pieces: Vec<SentencePiece>,
}

/// Trainer specification from protobuf
#[derive(Debug, Default)]
struct TrainerSpec {
    model_type: usize,
    byte_fallback: bool,
}

/// Normalizer specification from protobuf
#[derive(Debug, Default)]
struct NormalizerSpec {
    normalization: bool,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    treat_whitespace_as_suffix: bool,
}

/// Individual sentence piece from protobuf
#[derive(Debug, Default)]
struct SentencePiece {
    piece: String,
    score: f32,
    piece_type: PieceType,
}

/// Type of sentence piece
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum PieceType {
    #[default]
    Normal = 0,
    Unknown = 1,
    Control = 2,
    UserDefined = 3,
    Unused = 4,
    Byte = 5,
}

impl SentencePieceTokenizer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashMap::new(),
            scores: HashMap::new(),
            pad_token_id: None,
            unk_token_id: None,
            bos_token_id: None,
            eos_token_id: None,

            model_type: ModelType::Unigram,
            normalization: true,
            add_dummy_prefix: true,
            remove_extra_whitespaces: true,
            treat_whitespace_as_suffix: false,
            byte_fallback: false,

            nfc_normalization: true,
            nfkc_normalization: false,
            escape_whitespaces: true,
        }
    }

    /// Load a SentencePiece model from a .model file (simplified implementation)
    pub fn from_model_file<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(TrustformersError::invalid_config(format!(
                "Model file not found: {:?}",
                path
            )));
        }

        // For now, try to parse as a text-based vocabulary file
        // In a full implementation, we would parse the protobuf format
        let mut tokenizer = Self::new();
        tokenizer.load_vocab_from_model_file(path)?;
        Ok(tokenizer)
    }

    /// Load vocabulary from various SentencePiece model formats
    pub fn load_vocab_from_model_file<P: AsRef<Path>>(&mut self, model_path: P) -> Result<()> {
        let path = model_path.as_ref();

        // Try to parse as protobuf first (standard SentencePiece format)
        if let Ok(()) = self.load_protobuf_model(path) {
            return Ok(());
        }

        // Fallback to text-based vocabulary loading
        self.load_text_vocab(path)
    }

    /// Load SentencePiece protobuf model file
    fn load_protobuf_model<P: AsRef<Path>>(&mut self, model_path: P) -> Result<()> {
        let mut file = File::open(model_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Parse the protobuf model
        let model = self.parse_sentencepiece_model(&buffer)?;

        // Extract model configuration
        self.model_type = match model.trainer_spec.model_type {
            1 => ModelType::Unigram,
            2 => ModelType::Bpe,
            3 => ModelType::Word,
            4 => ModelType::Char,
            _ => ModelType::Unigram,
        };

        self.normalization = model.normalizer_spec.normalization;
        self.add_dummy_prefix = model.normalizer_spec.add_dummy_prefix;
        self.remove_extra_whitespaces = model.normalizer_spec.remove_extra_whitespaces;
        self.treat_whitespace_as_suffix = model.normalizer_spec.treat_whitespace_as_suffix;
        self.byte_fallback = model.trainer_spec.byte_fallback;

        // Load vocabulary pieces
        for (id, piece) in model.pieces.iter().enumerate() {
            let token_id = id as u32;
            self.vocab.insert(piece.piece.clone(), token_id);
            self.id_to_token.insert(token_id, piece.piece.clone());
            self.scores.insert(token_id, piece.score);

            // Handle special tokens
            match piece.piece_type {
                PieceType::Unknown => {
                    self.unk_token_id = Some(token_id);
                    self.special_tokens.insert(piece.piece.clone(), token_id);
                },
                PieceType::Control => {
                    self.special_tokens.insert(piece.piece.clone(), token_id);

                    // Identify specific control tokens
                    if piece.piece == "<pad>" {
                        self.pad_token_id = Some(token_id);
                    } else if piece.piece == "<s>" {
                        self.bos_token_id = Some(token_id);
                    } else if piece.piece == "</s>" {
                        self.eos_token_id = Some(token_id);
                    }
                },
                PieceType::UserDefined => {
                    self.special_tokens.insert(piece.piece.clone(), token_id);
                },
                _ => {},
            }
        }

        Ok(())
    }

    /// Parse SentencePiece protobuf model (simplified implementation)
    fn parse_sentencepiece_model(&self, buffer: &[u8]) -> Result<SentencePieceModel> {
        // This is a simplified protobuf parser for SentencePiece models
        // In a full implementation, we would use a proper protobuf library
        let mut model = SentencePieceModel::default();

        // Parse the protobuf format
        let mut pos = 0;
        while pos < buffer.len() {
            // Read field tag and wire type
            let (tag, wire_type) = self.read_varint_pair(buffer, &mut pos)?;

            match tag {
                1 => {
                    // trainer_spec
                    let length = self.read_varint(buffer, &mut pos)?;
                    let trainer_data = &buffer[pos..pos + length];
                    model.trainer_spec = self.parse_trainer_spec(trainer_data)?;
                    pos += length;
                },
                2 => {
                    // normalizer_spec
                    let length = self.read_varint(buffer, &mut pos)?;
                    let normalizer_data = &buffer[pos..pos + length];
                    model.normalizer_spec = self.parse_normalizer_spec(normalizer_data)?;
                    pos += length;
                },
                3 => {
                    // pieces
                    let length = self.read_varint(buffer, &mut pos)?;
                    let piece_data = &buffer[pos..pos + length];
                    let piece = self.parse_sentence_piece(piece_data)?;
                    model.pieces.push(piece);
                    pos += length;
                },
                _ => {
                    // Skip unknown fields
                    match wire_type {
                        0 => {
                            self.read_varint(buffer, &mut pos)?;
                        },
                        2 => {
                            let length = self.read_varint(buffer, &mut pos)?;
                            pos += length;
                        },
                        _ => {
                            return Err(TrustformersError::invalid_config(
                                "Unsupported wire type in protobuf".to_string(),
                            ));
                        },
                    }
                },
            }
        }

        Ok(model)
    }

    /// Parse trainer spec from protobuf data
    fn parse_trainer_spec(&self, data: &[u8]) -> Result<TrainerSpec> {
        let mut spec = TrainerSpec::default();
        let mut pos = 0;

        while pos < data.len() {
            let (tag, _) = self.read_varint_pair(data, &mut pos)?;

            match tag {
                1 => {
                    // model_type
                    spec.model_type = self.read_varint(data, &mut pos)?;
                },
                7 => {
                    // byte_fallback
                    spec.byte_fallback = self.read_varint(data, &mut pos)? != 0;
                },
                _ => {
                    // Skip unknown fields
                    self.read_varint(data, &mut pos)?;
                },
            }
        }

        Ok(spec)
    }

    /// Parse normalizer spec from protobuf data
    fn parse_normalizer_spec(&self, data: &[u8]) -> Result<NormalizerSpec> {
        let mut spec = NormalizerSpec::default();
        let mut pos = 0;

        while pos < data.len() {
            let (tag, _) = self.read_varint_pair(data, &mut pos)?;

            match tag {
                1 => {
                    // normalization
                    spec.normalization = self.read_varint(data, &mut pos)? != 0;
                },
                2 => {
                    // add_dummy_prefix
                    spec.add_dummy_prefix = self.read_varint(data, &mut pos)? != 0;
                },
                3 => {
                    // remove_extra_whitespaces
                    spec.remove_extra_whitespaces = self.read_varint(data, &mut pos)? != 0;
                },
                4 => {
                    // treat_whitespace_as_suffix
                    spec.treat_whitespace_as_suffix = self.read_varint(data, &mut pos)? != 0;
                },
                _ => {
                    // Skip unknown fields
                    self.read_varint(data, &mut pos)?;
                },
            }
        }

        Ok(spec)
    }

    /// Parse sentence piece from protobuf data
    fn parse_sentence_piece(&self, data: &[u8]) -> Result<SentencePiece> {
        let mut piece = SentencePiece::default();
        let mut pos = 0;

        while pos < data.len() {
            let (tag, wire_type) = self.read_varint_pair(data, &mut pos)?;

            match tag {
                1 => {
                    // piece (string)
                    let length = self.read_varint(data, &mut pos)?;
                    let piece_bytes = &data[pos..pos + length];
                    piece.piece = String::from_utf8_lossy(piece_bytes).into_owned();
                    pos += length;
                },
                2 => {
                    // score (float)
                    if pos + 4 <= data.len() {
                        let score_bytes = &data[pos..pos + 4];
                        piece.score = f32::from_le_bytes([
                            score_bytes[0],
                            score_bytes[1],
                            score_bytes[2],
                            score_bytes[3],
                        ]);
                        pos += 4;
                    }
                },
                3 => {
                    // type (enum)
                    let type_value = self.read_varint(data, &mut pos)?;
                    piece.piece_type = match type_value {
                        0 => PieceType::Normal,
                        1 => PieceType::Unknown,
                        2 => PieceType::Control,
                        3 => PieceType::UserDefined,
                        4 => PieceType::Unused,
                        5 => PieceType::Byte,
                        _ => PieceType::Normal,
                    };
                },
                _ => {
                    // Skip unknown fields
                    match wire_type {
                        0 => {
                            self.read_varint(data, &mut pos)?;
                        },
                        2 => {
                            let length = self.read_varint(data, &mut pos)?;
                            pos += length;
                        },
                        5 => {
                            pos += 4;
                        },
                        _ => {
                            return Err(TrustformersError::invalid_config(
                                "Unsupported wire type in sentence piece".to_string(),
                            ));
                        },
                    }
                },
            }
        }

        Ok(piece)
    }

    /// Read varint from buffer
    fn read_varint(&self, buffer: &[u8], pos: &mut usize) -> Result<usize> {
        let mut result = 0;
        let mut shift = 0;

        while *pos < buffer.len() {
            let byte = buffer[*pos];
            *pos += 1;

            result |= ((byte & 0x7F) as usize) << shift;

            if (byte & 0x80) == 0 {
                return Ok(result);
            }

            shift += 7;
            if shift >= 64 {
                return Err(TrustformersError::invalid_config(
                    "Varint too large".to_string(),
                ));
            }
        }

        Err(TrustformersError::invalid_config(
            "Unexpected end of buffer".to_string(),
        ))
    }

    /// Read varint pair (tag and wire type)
    fn read_varint_pair(&self, buffer: &[u8], pos: &mut usize) -> Result<(usize, usize)> {
        let value = self.read_varint(buffer, pos)?;
        let wire_type = value & 0x7;
        let field_tag = value >> 3;
        Ok((field_tag, wire_type))
    }

    /// Load text-based vocabulary file
    fn load_text_vocab<P: AsRef<Path>>(&mut self, vocab_path: P) -> Result<()> {
        let path = vocab_path.as_ref();
        let file = File::open(path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to open vocab file: {}", e))
        })?;

        let reader = BufReader::new(file);
        self.load_text_vocab_from_reader(reader)
    }

    /// Load vocabulary from text format
    fn load_text_vocab_from_reader<R: BufRead>(&mut self, reader: R) -> Result<()> {
        for (id, line) in reader.lines().enumerate() {
            let line = line
                .map_err(|e| TrustformersError::io_error(format!("Failed to read line: {}", e)))?;
            let parts: Vec<&str> = line.split('\t').collect();

            if parts.is_empty() {
                continue;
            }

            let token = parts[0].to_string();
            let score = if parts.len() > 1 { parts[1].parse::<f32>().unwrap_or(0.0) } else { 0.0 };

            let token_id = id as u32;
            self.vocab.insert(token.clone(), token_id);
            self.id_to_token.insert(token_id, token.clone());
            self.scores.insert(token_id, score);

            // Detect special tokens
            if token == "<pad>" || token == "[PAD]" {
                self.pad_token_id = Some(token_id);
            } else if token == "<unk>" || token == "[UNK]" {
                self.unk_token_id = Some(token_id);
            } else if token == "<s>" || token == "[CLS]" {
                self.bos_token_id = Some(token_id);
            } else if token == "</s>" || token == "[SEP]" {
                self.eos_token_id = Some(token_id);
            }
        }

        Ok(())
    }

    /// Configure the tokenizer model type
    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Configure normalization settings
    pub fn with_normalization(mut self, enable: bool) -> Self {
        self.normalization = enable;
        self
    }

    /// Configure dummy prefix addition
    pub fn with_dummy_prefix(mut self, enable: bool) -> Self {
        self.add_dummy_prefix = enable;
        self
    }

    /// Configure byte fallback
    pub fn with_byte_fallback(mut self, enable: bool) -> Self {
        self.byte_fallback = enable;
        self
    }

    pub fn from_pretrained(_model_name_or_path: &str) -> Result<Self> {
        // This would typically load from a SentencePiece model file
        // For now, we'll create a basic T5 tokenizer
        let mut tokenizer = Self::new();

        // Add basic T5 vocabulary (simplified)
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Add special tokens
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("</s>".to_string(), 1);
        vocab.insert("<unk>".to_string(), 2);

        id_to_token.insert(0, "<pad>".to_string());
        id_to_token.insert(1, "</s>".to_string());
        id_to_token.insert(2, "<unk>".to_string());

        // Add extra IDs for T5 (used for sentinel tokens)
        for i in 0..100 {
            let extra_id = format!("<extra_id_{}>", i);
            let token_id = 32000 + i as u32;
            vocab.insert(extra_id.clone(), token_id);
            id_to_token.insert(token_id, extra_id);
        }

        // Add some basic vocabulary for testing
        let basic_words = [
            "▁Hello",
            "▁world",
            "▁test",
            "▁sample",
            "▁the",
            "▁a",
            "▁an",
            "▁and",
            "▁or",
            "▁but",
            "▁this",
            "▁that",
            "▁it",
            "▁is",
            "▁was",
            "▁are",
            "▁were",
            "▁be",
            "▁been",
            "▁have",
            "▁has",
            "▁had",
            "▁do",
            "▁does",
            "▁did",
            "▁will",
            "▁would",
            "▁could",
            "▁should",
            "▁can",
            "▁may",
            "▁might",
            "▁must",
            "ing",
            "ed",
            "er",
            "est",
            "ly",
            "s",
        ];

        for (i, word) in basic_words.iter().enumerate() {
            let token_id = 1000 + i as u32;
            vocab.insert(word.to_string(), token_id);
            id_to_token.insert(token_id, word.to_string());
        }

        tokenizer.vocab = vocab;
        tokenizer.id_to_token = id_to_token;
        tokenizer.pad_token_id = Some(0);
        tokenizer.eos_token_id = Some(1);
        tokenizer.unk_token_id = Some(2);

        Ok(tokenizer)
    }

    pub fn load_vocab_from_file(&mut self, vocab_file: &str) -> Result<()> {
        let content = std::fs::read_to_string(vocab_file)
            .map_err(|e| TrustformersError::other(format!("Failed to read vocab file: {}", e)))?;

        for (id, line) in content.lines().enumerate() {
            let token = line.trim().to_string();
            let token_id = id as u32;
            self.vocab.insert(token.clone(), token_id);
            self.id_to_token.insert(token_id, token);
        }

        Ok(())
    }

    /// Normalize input text according to SentencePiece standards
    fn normalize_text(&self, text: &str) -> String {
        if !self.normalization {
            return text.to_string();
        }

        let mut normalized = text.to_string();

        // Unicode normalization
        if self.nfc_normalization {
            normalized = normalized.nfc().collect();
        }
        if self.nfkc_normalization {
            normalized = normalized.nfkc().collect();
        }

        // Remove extra whitespaces
        if self.remove_extra_whitespaces {
            normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
        }

        // Escape whitespaces
        if self.escape_whitespaces {
            normalized = normalized.replace(' ', "▁");
        }

        // Add dummy prefix if needed
        if self.add_dummy_prefix && !normalized.starts_with('▁') {
            normalized = format!("▁{}", normalized);
        }

        normalized
    }

    /// Tokenize text using the appropriate algorithm based on model type
    fn tokenize_text(&self, text: &str) -> Vec<String> {
        let normalized_text = self.normalize_text(text);

        match self.model_type {
            ModelType::Unigram => self.tokenize_unigram(&normalized_text),
            ModelType::Bpe => self.tokenize_bpe(&normalized_text),
            ModelType::Word => self.tokenize_word(&normalized_text),
            ModelType::Char => self.tokenize_char(&normalized_text),
        }
    }

    /// Unigram tokenization using Viterbi algorithm
    fn tokenize_unigram(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return vec![];
        }

        // Simplified Unigram tokenization
        // In a full implementation, this would use the Viterbi algorithm
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;

        while start < chars.len() {
            let mut best_token = None;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_end = start + 1;

            // Try all possible substrings starting from current position
            for end in (start + 1)..=chars.len() {
                let candidate: String = chars[start..end].iter().collect();

                if let Some(&token_id) = self.vocab.get(&candidate) {
                    let score = self.scores.get(&token_id).copied().unwrap_or(0.0);
                    if score > best_score {
                        best_score = score;
                        best_token = Some(candidate);
                        best_end = end;
                    }
                }
            }

            if let Some(token) = best_token {
                tokens.push(token);
                start = best_end;
            } else {
                // Use single character if no match found
                let char_token: String = chars[start..start + 1].iter().collect();
                tokens.push(char_token);
                start += 1;
            }
        }

        tokens
    }

    /// BPE tokenization
    fn tokenize_bpe(&self, text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Apply BPE merges (simplified)
        loop {
            let mut best_pair = None;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_pos = 0;

            // Find best merge
            for i in 0..(tokens.len().saturating_sub(1)) {
                let merged = format!("{}{}", tokens[i], tokens[i + 1]);
                if let Some(&token_id) = self.vocab.get(&merged) {
                    let score = self.scores.get(&token_id).copied().unwrap_or(0.0);
                    if score > best_score {
                        best_score = score;
                        best_pair = Some(merged);
                        best_pos = i;
                    }
                }
            }

            if let Some(pair) = best_pair {
                // Apply merge
                tokens[best_pos] = pair;
                tokens.remove(best_pos + 1);
            } else {
                break;
            }
        }

        tokens
    }

    /// Word-level tokenization
    fn tokenize_word(&self, text: &str) -> Vec<String> {
        text.split('▁').filter(|s| !s.is_empty()).map(|s| format!("▁{}", s)).collect()
    }

    /// Character-level tokenization
    fn tokenize_char(&self, text: &str) -> Vec<String> {
        text.chars().map(|c| c.to_string()).collect()
    }

    fn convert_tokens_to_ids(&self, tokens: &[String]) -> Vec<u32> {
        tokens
            .iter()
            .map(|token| {
                self.vocab.get(token).copied().unwrap_or_else(|| {
                    // Try byte fallback if enabled
                    if self.byte_fallback {
                        self.handle_byte_fallback(token)
                    } else {
                        self.unk_token_id.unwrap_or(2)
                    }
                })
            })
            .collect()
    }

    fn convert_ids_to_tokens(&self, ids: &[u32]) -> Vec<String> {
        ids.iter()
            .map(|&id| self.id_to_token.get(&id).cloned().unwrap_or_else(|| "<unk>".to_string()))
            .collect()
    }

    /// Handle byte fallback for unknown tokens
    fn handle_byte_fallback(&self, _token: &str) -> u32 {
        // In a full implementation, this would convert to byte tokens
        // For now, just return UNK token
        self.unk_token_id.unwrap_or(2)
    }

    /// Enhanced decode method with proper text reconstruction
    fn decode_tokens(&self, tokens: &[String]) -> String {
        let mut result = String::new();

        for token in tokens {
            // Skip special tokens in output
            if self.is_special_token(token) {
                continue;
            }

            if token.starts_with('▁') {
                // SentencePiece word boundary marker
                if !result.is_empty() {
                    result.push(' ');
                }
                result.push_str(&token[3..]); // Remove the ▁ character (3 bytes in UTF-8)
            } else {
                result.push_str(token);
            }
        }

        // Post-process the result
        if self.escape_whitespaces {
            result = result.replace('▁', " ");
        }

        result.trim().to_string()
    }

    /// Check if a token is a special token
    fn is_special_token(&self, token: &str) -> bool {
        matches!(
            token,
            "<pad>"
                | "[PAD]"
                | "<unk>"
                | "[UNK]"
                | "<s>"
                | "[CLS]"
                | "</s>"
                | "[SEP]"
                | "<mask>"
                | "[MASK]"
        ) || token.starts_with("<extra_id_")
    }

    /// Get token score for ranking
    pub fn get_token_score(&self, token_id: u32) -> Option<f32> {
        self.scores.get(&token_id).copied()
    }

    /// Get all tokens sorted by score
    pub fn get_tokens_by_score(&self) -> Vec<(String, u32, f32)> {
        let mut tokens: Vec<_> = self
            .vocab
            .iter()
            .map(|(token, &id)| {
                let score = self.scores.get(&id).copied().unwrap_or(0.0);
                (token.clone(), id, score)
            })
            .collect();

        tokens.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        tokens
    }

    /// Get the score for a specific token ID
    pub fn get_score(&self, token_id: u32) -> Option<f32> {
        self.scores.get(&token_id).copied()
    }

    /// Check if a token is a special token (public version)
    pub fn is_special_token_public(&self, token: &str) -> bool {
        self.special_tokens.contains_key(token)
    }

    /// Get the UNK token string
    pub fn unk_token(&self) -> Option<&str> {
        self.unk_token_id.and_then(|id| self.id_to_token.get(&id).map(|s| s.as_str()))
    }

    /// Get BOS token ID
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Check if normalization is enabled
    pub fn uses_normalization(&self) -> bool {
        self.normalization
    }

    /// Check if extra whitespace removal is enabled
    pub fn removes_extra_whitespaces(&self) -> bool {
        self.remove_extra_whitespaces
    }

    /// Check if whitespace is treated as suffix
    pub fn treats_whitespace_as_suffix(&self) -> bool {
        self.treat_whitespace_as_suffix
    }

    /// Get model type as string
    pub fn model_type_string(&self) -> String {
        match self.model_type {
            ModelType::Unigram => "Unigram".to_string(),
            ModelType::Bpe => "BPE".to_string(),
            ModelType::Word => "Word".to_string(),
            ModelType::Char => "Char".to_string(),
        }
    }

    /// Check if byte fallback is enabled
    pub fn uses_byte_fallback(&self) -> bool {
        self.byte_fallback
    }
}

impl Default for SentencePieceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let tokens = self.tokenize_text(text);
        let input_ids = self.convert_tokens_to_ids(&tokens);

        // Create attention mask (all 1s for now)
        let attention_mask = vec![1u8; input_ids.len()];

        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        // For T5, we typically don't use paired inputs in the same way as BERT
        // But we can implement it by concatenating with a separator
        let combined_text = format!("{} </s> {}", text, text2);
        self.encode(&combined_text)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let tokens = self.convert_ids_to_tokens(ids);
        Ok(self.decode_tokens(&tokens))
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token.get(&id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentencepiece_tokenizer_creation() {
        let tokenizer = SentencePieceTokenizer::new();
        assert_eq!(tokenizer.vocab_size(), 0);
    }

    #[test]
    fn test_basic_tokenization() {
        let tokenizer = SentencePieceTokenizer::from_pretrained("t5-small").unwrap();
        let result = tokenizer.encode("Hello world").unwrap();
        assert!(!result.input_ids.is_empty());
        assert_eq!(result.input_ids.len(), result.attention_mask.len());
    }

    #[test]
    fn test_decode() {
        let tokenizer = SentencePieceTokenizer::from_pretrained("t5-small").unwrap();
        let encoded = tokenizer.encode("Hello world").unwrap();
        let decoded = tokenizer.decode(&encoded.input_ids).unwrap();
        // The decoded text should be similar to the original
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_enhanced_normalization() {
        let tokenizer =
            SentencePieceTokenizer::new().with_normalization(true).with_dummy_prefix(true);

        let normalized = tokenizer.normalize_text("Hello  world");
        assert!(normalized.starts_with('▁'));
        assert!(!normalized.contains("  ")); // Extra spaces should be removed
    }

    #[test]
    fn test_model_types() {
        let mut tokenizer = SentencePieceTokenizer::new().with_model_type(ModelType::Char);

        // Add some character vocabulary
        for ch in 'a'..='z' {
            let token_id = tokenizer.vocab.len() as u32;
            tokenizer.vocab.insert(ch.to_string(), token_id);
            tokenizer.id_to_token.insert(token_id, ch.to_string());
        }

        let tokens = tokenizer.tokenize_char("hello");
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], "h");
        assert_eq!(tokens[1], "e");
    }

    #[test]
    fn test_special_token_detection() {
        let tokenizer = SentencePieceTokenizer::new();

        assert!(tokenizer.is_special_token("<pad>"));
        assert!(tokenizer.is_special_token("[UNK]"));
        assert!(tokenizer.is_special_token("<extra_id_0>"));
        assert!(!tokenizer.is_special_token("hello"));
    }

    #[test]
    fn test_token_scores() {
        let mut tokenizer = SentencePieceTokenizer::new();

        // Add tokens with scores
        tokenizer.vocab.insert("test".to_string(), 0);
        tokenizer.scores.insert(0, 5.0);

        assert_eq!(tokenizer.get_token_score(0), Some(5.0));
        assert_eq!(tokenizer.get_token_score(999), None);

        let sorted_tokens = tokenizer.get_tokens_by_score();
        if !sorted_tokens.is_empty() {
            assert_eq!(sorted_tokens[0].0, "test");
            assert_eq!(sorted_tokens[0].2, 5.0);
        }
    }

    #[test]
    fn test_configuration_methods() {
        let tokenizer = SentencePieceTokenizer::new()
            .with_model_type(ModelType::Bpe)
            .with_normalization(false)
            .with_dummy_prefix(false)
            .with_byte_fallback(true);

        assert_eq!(tokenizer.model_type, ModelType::Bpe);
        assert!(!tokenizer.normalization);
        assert!(!tokenizer.add_dummy_prefix);
        assert!(tokenizer.byte_fallback);
    }
}
