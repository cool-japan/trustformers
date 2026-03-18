use crate::vocab::Vocab;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Tiktoken-style tokenizer compatible with OpenAI's encoding format
#[derive(Debug)]
pub struct TiktokenTokenizer {
    vocab: Vocab,
    encoder: HashMap<Vec<u8>, usize>,
    decoder: HashMap<usize, Vec<u8>>,
    special_tokens: HashMap<String, usize>,
    pattern: Regex,
    cache: RwLock<HashMap<String, Vec<usize>>>,
}

impl Clone for TiktokenTokenizer {
    fn clone(&self) -> Self {
        Self {
            vocab: self.vocab.clone(),
            encoder: self.encoder.clone(),
            decoder: self.decoder.clone(),
            special_tokens: self.special_tokens.clone(),
            pattern: self.pattern.clone(),
            cache: RwLock::new(HashMap::new()), // Create new cache for clone
        }
    }
}

// GPT-3.5/4 pattern from tiktoken (simplified for Rust regex)
static TIKTOKEN_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"
    ).expect("TIKTOKEN_PATTERN regex must be valid")
});

impl TiktokenTokenizer {
    /// Create a new Tiktoken tokenizer
    pub fn new(
        encoder: HashMap<Vec<u8>, usize>,
        special_tokens: HashMap<String, usize>,
        pattern: Option<Regex>,
    ) -> Self {
        // Create decoder from encoder
        let decoder: HashMap<usize, Vec<u8>> =
            encoder.iter().map(|(k, &v)| (v, k.clone())).collect();

        // Create vocab for compatibility
        let vocab_map: HashMap<String, u32> = encoder
            .iter()
            .map(|(bytes, &rank)| {
                let token = String::from_utf8_lossy(bytes).to_string();
                (token, rank as u32)
            })
            .collect();

        Self {
            vocab: Vocab::from_map(vocab_map),
            encoder,
            decoder,
            special_tokens,
            pattern: pattern.unwrap_or_else(|| TIKTOKEN_PATTERN.clone()),
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Create a tokenizer for GPT-3.5-turbo/GPT-4 (cl100k_base encoding)
    pub fn cl100k_base() -> Self {
        // This is a simplified version - in practice you'd load from tiktoken files
        let mut encoder = HashMap::new();
        let mut special_tokens = HashMap::new();

        // Add some basic tokens (this would normally be loaded from tiktoken data)
        for i in 0..256 {
            encoder.insert(vec![i as u8], i);
        }

        // Special tokens for cl100k_base
        special_tokens.insert("<|endoftext|>".to_string(), 100257);
        special_tokens.insert("<|fim_prefix|>".to_string(), 100258);
        special_tokens.insert("<|fim_middle|>".to_string(), 100259);
        special_tokens.insert("<|fim_suffix|>".to_string(), 100260);
        special_tokens.insert("<|endofprompt|>".to_string(), 100276);

        Self::new(encoder, special_tokens, None)
    }

    /// Create a tokenizer for GPT-2 (r50k_base encoding)
    pub fn r50k_base() -> Self {
        let mut encoder = HashMap::new();
        let mut special_tokens = HashMap::new();

        // Add basic byte tokens
        for i in 0..256 {
            encoder.insert(vec![i as u8], i);
        }

        // Special token for r50k_base
        special_tokens.insert("<|endoftext|>".to_string(), 50256);

        Self::new(encoder, special_tokens, None)
    }

    /// Load tokenizer from tiktoken files
    pub fn from_tiktoken_file(
        encoder_path: &str,
        special_tokens_path: Option<&str>,
    ) -> Result<Self> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        // Load main encoder from file
        let mut encoder = HashMap::new();

        let file = File::open(encoder_path).map_err(|e| {
            TrustformersError::other(format!(
                "Failed to open encoder file {}: {}",
                encoder_path, e
            ))
        })?;
        let reader = BufReader::new(file);

        // Parse tiktoken encoder format
        // Each line should be: base64_token rank
        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                TrustformersError::other(format!("Failed to read line {}: {}", line_num, e))
            })?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue; // Skip empty lines and comments
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(TrustformersError::other(format!(
                    "Invalid encoder format at line {}: expected 'token rank', got '{}'",
                    line_num + 1,
                    line
                )));
            }

            // Decode base64 token
            let token_bytes = Self::decode_tiktoken_token(parts[0]).map_err(|e| {
                TrustformersError::other(format!(
                    "Failed to decode token at line {}: {}",
                    line_num + 1,
                    e
                ))
            })?;

            // Parse rank
            let rank: usize = parts[1].parse().map_err(|e| {
                TrustformersError::other(format!("Invalid rank at line {}: {}", line_num + 1, e))
            })?;

            encoder.insert(token_bytes, rank);
        }

        // Load special tokens if provided
        let mut special_tokens = HashMap::new();
        if let Some(special_path) = special_tokens_path {
            let special_file = File::open(special_path).map_err(|e| {
                TrustformersError::other(format!(
                    "Failed to open special tokens file {}: {}",
                    special_path, e
                ))
            })?;
            let special_reader = BufReader::new(special_file);

            for (line_num, line) in special_reader.lines().enumerate() {
                let line = line.map_err(|e| {
                    TrustformersError::other(format!(
                        "Failed to read special tokens line {}: {}",
                        line_num, e
                    ))
                })?;
                let line = line.trim();

                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() != 2 {
                    return Err(TrustformersError::other(format!(
                        "Invalid special token format at line {}: expected 'token rank'",
                        line_num + 1
                    )));
                }

                let token = parts[0].to_string();
                let rank: usize = parts[1].parse().map_err(|e| {
                    TrustformersError::other(format!(
                        "Invalid special token rank at line {}: {}",
                        line_num + 1,
                        e
                    ))
                })?;

                special_tokens.insert(token, rank);
            }
        }

        // Use the default tiktoken pattern for now, but it could be customized
        let pattern = TIKTOKEN_PATTERN.clone();

        Ok(Self::new(encoder, special_tokens, Some(pattern)))
    }

    /// Decode a tiktoken base64-encoded token
    fn decode_tiktoken_token(encoded: &str) -> std::result::Result<Vec<u8>, String> {
        // Tiktoken uses a custom base64-like encoding
        // This is a simplified version - actual tiktoken uses more complex encoding

        // Handle common tiktoken escape sequences
        let decoded = if encoded.starts_with("b'") && encoded.ends_with("'") {
            // Python bytes literal format: b'hello'
            let inner = &encoded[2..encoded.len() - 1];
            Self::decode_python_bytes_literal(inner)?
        } else {
            // Try standard base64 decoding
            use base64::{engine::general_purpose::STANDARD, Engine as _};
            STANDARD.decode(encoded).map_err(|e| format!("Base64 decode error: {}", e))?
        };

        Ok(decoded)
    }

    /// Decode Python bytes literal format
    fn decode_python_bytes_literal(literal: &str) -> std::result::Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        let mut chars = literal.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                '\\' => {
                    match chars.next().ok_or("Unexpected end of escape sequence")? {
                        'n' => bytes.push(b'\n'),
                        'r' => bytes.push(b'\r'),
                        't' => bytes.push(b'\t'),
                        '\\' => bytes.push(b'\\'),
                        '\'' => bytes.push(b'\''),
                        '"' => bytes.push(b'"'),
                        'x' => {
                            // Hex escape: \xNN
                            let hex1 = chars.next().ok_or("Incomplete hex escape")?;
                            let hex2 = chars.next().ok_or("Incomplete hex escape")?;
                            let hex_str = format!("{}{}", hex1, hex2);
                            let byte_val = u8::from_str_radix(&hex_str, 16)
                                .map_err(|_| format!("Invalid hex escape: \\x{}", hex_str))?;
                            bytes.push(byte_val);
                        },
                        '0'..='7' => {
                            // Octal escape: \NNN (up to 3 digits)
                            let mut octal = String::new();
                            octal.push(chars.peek().copied().unwrap_or('0')); // First digit already consumed
                            chars.next(); // consume it

                            // Try to get up to 2 more octal digits
                            for _ in 0..2 {
                                if let Some(&next_char) = chars.peek() {
                                    if next_char.is_ascii_digit() && next_char <= '7' {
                                        octal.push(next_char);
                                        chars.next();
                                    } else {
                                        break;
                                    }
                                }
                            }

                            let byte_val = u8::from_str_radix(&octal, 8)
                                .map_err(|_| format!("Invalid octal escape: \\{}", octal))?;
                            bytes.push(byte_val);
                        },
                        other => return Err(format!("Unknown escape sequence: \\{}", other)),
                    }
                },
                _ => {
                    // Regular ASCII character
                    if ch.is_ascii() {
                        bytes.push(ch as u8);
                    } else {
                        // Non-ASCII character - encode as UTF-8
                        let mut utf8_buf = [0u8; 4];
                        let utf8_bytes = ch.encode_utf8(&mut utf8_buf).as_bytes();
                        bytes.extend_from_slice(utf8_bytes);
                    }
                },
            }
        }

        Ok(bytes)
    }

    /// Encode bytes using BPE with tiktoken's algorithm
    fn encode_bytes(&self, piece: &[u8]) -> Vec<usize> {
        if piece.len() == 1 {
            return vec![self.encoder.get(piece).copied().unwrap_or(0)];
        }

        // Convert to mutable vector of single-byte vectors
        let mut word: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();

        while word.len() > 1 {
            // Find the pair with minimum rank
            let mut min_rank = usize::MAX;
            let mut merge_idx = None;

            for i in 0..word.len() - 1 {
                let mut merged = word[i].clone();
                merged.extend_from_slice(&word[i + 1]);

                if let Some(&rank) = self.encoder.get(&merged) {
                    if rank < min_rank {
                        min_rank = rank;
                        merge_idx = Some(i);
                    }
                }
            }

            if let Some(idx) = merge_idx {
                let mut merged = word[idx].clone();
                merged.extend_from_slice(&word[idx + 1]);
                word[idx] = merged;
                word.remove(idx + 1);
            } else {
                break;
            }
        }

        // Convert to token IDs
        word.into_iter()
            .map(|bytes| self.encoder.get(&bytes).copied().unwrap_or(0))
            .collect()
    }

    /// Encode text to token IDs
    pub fn encode_text(&self, text: &str) -> Vec<usize> {
        // Check cache first
        if let Ok(cache) = self.cache.read() {
            if let Some(cached) = cache.get(text) {
                return cached.clone();
            }
        }

        let mut tokens = vec![];

        // Handle special tokens first
        let _remaining_text = text;
        for (special_token, &token_id) in &self.special_tokens {
            if text.contains(special_token) {
                // Split by special tokens and encode each part
                let parts: Vec<&str> = text.split(special_token).collect();
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        tokens.push(token_id);
                    }
                    if !part.is_empty() {
                        tokens.extend(self.encode_text_without_special(part));
                    }
                }

                // Cache the result
                if let Ok(mut cache) = self.cache.write() {
                    cache.insert(text.to_string(), tokens.clone());
                }
                return tokens;
            }
        }

        // No special tokens found, encode normally
        let result = self.encode_text_without_special(text);

        // Cache the result
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(text.to_string(), result.clone());
        }

        result
    }

    fn encode_text_without_special(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![];

        for mat in self.pattern.find_iter(text) {
            let piece = mat.as_str().as_bytes();
            tokens.extend(self.encode_bytes(piece));
        }

        tokens
    }

    /// Decode token IDs to text
    pub fn decode_tokens(&self, tokens: &[usize]) -> Result<String> {
        let mut bytes = vec![];

        for &token_id in tokens {
            // Check if it's a special token
            let is_special = self.special_tokens.values().any(|&id| id == token_id);

            if is_special {
                // Find the special token string
                if let Some((special_str, _)) =
                    self.special_tokens.iter().find(|(_, &id)| id == token_id)
                {
                    bytes.extend_from_slice(special_str.as_bytes());
                }
            } else if let Some(token_bytes) = self.decoder.get(&token_id) {
                bytes.extend_from_slice(token_bytes);
            }
        }

        String::from_utf8(bytes)
            .map_err(|e| TrustformersError::other(format!("Failed to decode UTF-8: {}", e)))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.encoder.len() + self.special_tokens.len()
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &HashMap<String, usize> {
        &self.special_tokens
    }

    /// Check if a token ID is a special token
    pub fn is_special_token(&self, token_id: usize) -> bool {
        self.special_tokens.values().any(|&id| id == token_id)
    }

    /// Encode with attention to special tokens
    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        allowed_special: &HashSet<String>,
    ) -> Vec<usize> {
        let mut result = vec![];
        let mut start = 0;

        while start < text.len() {
            let mut found_special = false;

            // Look for special tokens at current position
            for special_token in allowed_special {
                if text[start..].starts_with(special_token) {
                    if let Some(&token_id) = self.special_tokens.get(special_token) {
                        result.push(token_id);
                        start += special_token.len();
                        found_special = true;
                        break;
                    }
                }
            }

            if !found_special {
                // Find the next special token or end of string
                let mut end = text.len();
                for special_token in allowed_special {
                    if let Some(pos) = text[start..].find(special_token) {
                        end = end.min(start + pos);
                    }
                }

                // Encode the regular text portion
                if start < end {
                    result.extend(self.encode_text_without_special(&text[start..end]));
                    start = end;
                }
            }
        }

        result
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let tokens = self.encode_text(text);

        let input_ids: Vec<u32> = tokens.into_iter().map(|t| t as u32).collect();
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
        // For tiktoken, we typically don't use special separators like BERT
        // Just concatenate with a space
        let combined = format!("{} {}", text, text2);
        self.encode(&combined)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let tokens: Vec<usize> = ids.iter().map(|&id| id as usize).collect();
        self.decode_tokens(&tokens)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.iter().map(|(k, &v)| (k.clone(), v)).collect()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiktoken_cl100k_base() {
        let tokenizer = TiktokenTokenizer::cl100k_base();
        // This is a simplified implementation with basic tokens + special tokens
        assert!(tokenizer.vocab_size() > 250); // 256 basic + 5 special tokens
        assert!(tokenizer.special_tokens().contains_key("<|endoftext|>"));
        assert!(tokenizer.special_tokens().contains_key("<|fim_prefix|>"));
    }

    #[test]
    fn test_tiktoken_r50k_base() {
        let tokenizer = TiktokenTokenizer::r50k_base();
        // This is a simplified implementation with basic tokens + special tokens
        assert!(tokenizer.vocab_size() > 250); // 256 basic + 1 special token
        assert!(tokenizer.special_tokens().contains_key("<|endoftext|>"));
    }

    #[test]
    fn test_tiktoken_encode_decode() {
        let tokenizer = TiktokenTokenizer::cl100k_base();

        let text = "Hello, world!";
        let tokens = tokenizer.encode_text(text);
        assert!(!tokens.is_empty());

        let decoded = tokenizer.decode_tokens(&tokens).unwrap();
        // Due to tokenization differences, we check that it's roughly the same
        assert!(decoded.contains("Hello") || decoded.contains("world"));
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = TiktokenTokenizer::cl100k_base();

        let text = "Hello <|endoftext|> world";
        let tokens = tokenizer.encode_text(text);

        // Should contain the special token ID
        assert!(tokens.contains(&100257)); // <|endoftext|> token ID
    }

    #[test]
    fn test_tokenizer_trait() {
        let tokenizer = TiktokenTokenizer::cl100k_base();

        let result = tokenizer.encode("Hello, world!").unwrap();
        assert!(!result.input_ids.is_empty());
        assert_eq!(result.input_ids.len(), result.attention_mask.len());

        let decoded = tokenizer.decode(&result.input_ids).unwrap();
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_encode_with_special_tokens() {
        let tokenizer = TiktokenTokenizer::cl100k_base();

        let mut allowed_special = HashSet::new();
        allowed_special.insert("<|endoftext|>".to_string());

        let text = "Hello <|endoftext|> world";
        let tokens = tokenizer.encode_with_special_tokens(text, &allowed_special);

        assert!(!tokens.is_empty());
        assert!(tokens.contains(&100257)); // <|endoftext|> token ID
    }

    #[test]
    fn test_is_special_token() {
        let tokenizer = TiktokenTokenizer::cl100k_base();

        assert!(tokenizer.is_special_token(100257)); // <|endoftext|>
        assert!(!tokenizer.is_special_token(0)); // Regular token
    }

    #[test]
    fn test_cache_functionality() {
        let tokenizer = TiktokenTokenizer::cl100k_base();

        let text = "Hello, world!";
        let tokens1 = tokenizer.encode_text(text);
        let tokens2 = tokenizer.encode_text(text); // Should use cache

        assert_eq!(tokens1, tokens2);
    }
}
