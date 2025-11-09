use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Fairseq dictionary format tokenizer
///
/// Fairseq uses simple text-based dictionary files where each line contains:
/// token frequency_count
///
/// Special tokens:
/// - <pad> 0 (padding token)
/// - </s> 1 (end of sentence)
/// - <unk> 2 (unknown token)
/// - <s> 3 (start of sentence)
#[derive(Debug, Clone)]
pub struct FairseqTokenizer {
    /// Token to ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    /// Token frequencies from original dictionary
    token_frequencies: HashMap<String, u64>,
    /// Special tokens
    pad_token: String,
    eos_token: String,
    unk_token: String,
    bos_token: String,
    /// Maximum sequence length
    max_length: usize,
}

impl FairseqTokenizer {
    /// Create a new Fairseq tokenizer from a dictionary file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut token_frequencies = HashMap::new();

        // Add special tokens first
        let special_tokens = vec![("<pad>", 0), ("</s>", 1), ("<unk>", 2), ("<s>", 3)];

        for (token, id) in special_tokens {
            token_to_id.insert(token.to_string(), id);
            id_to_token.insert(id, token.to_string());
        }

        let mut next_id = 4;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() != 2 {
                return Err(TrustformersError::invalid_format(
                    "Valid Fairseq dictionary line format".to_string(),
                    format!("Invalid line: {}", line),
                ));
            }

            let token = parts[0].to_string();
            let frequency = parts[1]
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("Invalid frequency in line: {}", line))?;

            // Skip if already added as special token
            if token_to_id.contains_key(&token) {
                token_frequencies.insert(token, frequency);
                continue;
            }

            token_to_id.insert(token.clone(), next_id);
            id_to_token.insert(next_id, token.clone());
            token_frequencies.insert(token, frequency);
            next_id += 1;
        }

        Ok(Self {
            token_to_id,
            id_to_token,
            token_frequencies,
            pad_token: "<pad>".to_string(),
            eos_token: "</s>".to_string(),
            unk_token: "<unk>".to_string(),
            bos_token: "<s>".to_string(),
            max_length: 512,
        })
    }

    /// Create a new Fairseq tokenizer from token-frequency pairs
    pub fn from_tokens(tokens_with_freq: Vec<(String, u64)>) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut token_frequencies = HashMap::new();

        // Add special tokens first
        let special_tokens = vec![("<pad>", 0), ("</s>", 1), ("<unk>", 2), ("<s>", 3)];

        for (token, id) in special_tokens {
            token_to_id.insert(token.to_string(), id);
            id_to_token.insert(id, token.to_string());
        }

        let mut next_id = 4;

        for (token, frequency) in tokens_with_freq {
            if token_to_id.contains_key(&token) {
                token_frequencies.insert(token, frequency);
                continue;
            }

            token_to_id.insert(token.clone(), next_id);
            id_to_token.insert(next_id, token.clone());
            token_frequencies.insert(token, frequency);
            next_id += 1;
        }

        Self {
            token_to_id,
            id_to_token,
            token_frequencies,
            pad_token: "<pad>".to_string(),
            eos_token: "</s>".to_string(),
            unk_token: "<unk>".to_string(),
            bos_token: "<s>".to_string(),
            max_length: 512,
        }
    }

    /// Save dictionary to Fairseq format file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = File::create(path)?;

        // Create sorted list of tokens by ID
        let mut sorted_tokens: Vec<_> = self.id_to_token.iter().collect();
        sorted_tokens.sort_by_key(|(id, _)| *id);

        for (_, token) in sorted_tokens {
            let frequency = self.token_frequencies.get(token).unwrap_or(&1);
            writeln!(file, "{} {}", token, frequency)?;
        }

        Ok(())
    }

    /// Get token frequency
    pub fn get_frequency(&self, token: &str) -> Option<u64> {
        self.token_frequencies.get(token).copied()
    }

    /// Get all tokens with frequencies sorted by frequency (descending)
    pub fn get_tokens_by_frequency(&self) -> Vec<(String, u64)> {
        let mut tokens: Vec<_> = self
            .token_frequencies
            .iter()
            .map(|(token, freq)| (token.clone(), *freq))
            .collect();
        tokens.sort_by(|a, b| b.1.cmp(&a.1));
        tokens
    }

    /// Set maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Simple word-level tokenization (space-separated)
    fn tokenize_words(&self, text: &str) -> Vec<String> {
        text.split_whitespace().map(|s| s.to_lowercase()).collect()
    }
}

impl Tokenizer for FairseqTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let words = self.tokenize_words(text);
        let mut input_ids = Vec::new();

        // Add BOS token
        input_ids.push(self.token_to_id[&self.bos_token]);

        for word in words {
            let id = self
                .token_to_id
                .get(&word)
                .copied()
                .unwrap_or_else(|| self.token_to_id[&self.unk_token]);
            input_ids.push(id);
        }

        // Add EOS token
        input_ids.push(self.token_to_id[&self.eos_token]);

        // Truncate if necessary
        if input_ids.len() > self.max_length {
            input_ids.truncate(self.max_length - 1);
            input_ids.push(self.token_to_id[&self.eos_token]);
        }

        let attention_mask = vec![1; input_ids.len()];

        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let tokens: Vec<String> = token_ids
            .iter()
            .filter_map(|&id| self.id_to_token.get(&id))
            .filter(|token| {
                *token != &self.pad_token && *token != &self.bos_token && *token != &self.eos_token
            })
            .cloned()
            .collect();

        Ok(tokens.join(" "))
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        let words_a = self.tokenize_words(text_a);
        let words_b = self.tokenize_words(text_b);
        let mut input_ids = Vec::new();
        let mut token_type_ids = Vec::new();

        // Add BOS token
        input_ids.push(self.token_to_id[&self.bos_token]);
        token_type_ids.push(0);

        // Add first sequence
        for word in words_a {
            let id = self
                .token_to_id
                .get(&word)
                .copied()
                .unwrap_or_else(|| self.token_to_id[&self.unk_token]);
            input_ids.push(id);
            token_type_ids.push(0);
        }

        // Add EOS token
        input_ids.push(self.token_to_id[&self.eos_token]);
        token_type_ids.push(0);

        // Add second sequence
        for word in words_b {
            let id = self
                .token_to_id
                .get(&word)
                .copied()
                .unwrap_or_else(|| self.token_to_id[&self.unk_token]);
            input_ids.push(id);
            token_type_ids.push(1);
        }

        // Add final EOS token
        input_ids.push(self.token_to_id[&self.eos_token]);
        token_type_ids.push(1);

        // Truncate if necessary
        if input_ids.len() > self.max_length {
            input_ids.truncate(self.max_length - 1);
            token_type_ids.truncate(self.max_length - 1);
            input_ids.push(self.token_to_id[&self.eos_token]);
            token_type_ids.push(1);
        }

        let attention_mask = vec![1; input_ids.len()];

        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids: Some(token_type_ids),
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.token_to_id.clone()
    }

    fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token.get(&id).cloned()
    }
}

/// Fairseq dictionary builder
pub struct FairseqDictionaryBuilder {
    token_counts: HashMap<String, u64>,
    min_frequency: u64,
    max_vocab_size: Option<usize>,
}

impl FairseqDictionaryBuilder {
    /// Create a new dictionary builder
    pub fn new() -> Self {
        Self {
            token_counts: HashMap::new(),
            min_frequency: 1,
            max_vocab_size: None,
        }
    }

    /// Set minimum frequency threshold
    pub fn min_frequency(mut self, min_freq: u64) -> Self {
        self.min_frequency = min_freq;
        self
    }

    /// Set maximum vocabulary size
    pub fn max_vocab_size(mut self, max_size: usize) -> Self {
        self.max_vocab_size = Some(max_size);
        self
    }

    /// Add text to the dictionary
    pub fn add_text(&mut self, text: &str) {
        for word in text.split_whitespace() {
            let word = word.to_lowercase();
            *self.token_counts.entry(word).or_insert(0) += 1;
        }
    }

    /// Add multiple texts to the dictionary
    pub fn add_texts(&mut self, texts: &[String]) {
        for text in texts {
            self.add_text(text);
        }
    }

    /// Build the tokenizer
    pub fn build(self) -> FairseqTokenizer {
        let mut tokens: Vec<_> = self
            .token_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_frequency)
            .collect();

        // Sort by frequency (descending)
        tokens.sort_by(|a, b| b.1.cmp(&a.1));

        // Apply vocabulary size limit
        if let Some(max_size) = self.max_vocab_size {
            tokens.truncate(max_size.saturating_sub(4)); // Reserve space for special tokens
        }

        FairseqTokenizer::from_tokens(tokens)
    }
}

impl Default for FairseqDictionaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_fairseq_tokenizer_from_tokens() {
        let tokens = vec![
            ("hello".to_string(), 100),
            ("world".to_string(), 80),
            ("test".to_string(), 50),
        ];

        let tokenizer = FairseqTokenizer::from_tokens(tokens);

        assert_eq!(tokenizer.vocab_size(), 7); // 4 special + 3 regular tokens
        assert_eq!(tokenizer.token_to_id("hello"), Some(4));
        assert_eq!(tokenizer.token_to_id("world"), Some(5));
        assert_eq!(tokenizer.token_to_id("<unk>"), Some(2));
        assert_eq!(tokenizer.get_frequency("hello"), Some(100));
    }

    #[test]
    fn test_fairseq_tokenizer_encode() {
        let tokens = vec![("hello".to_string(), 100), ("world".to_string(), 80)];

        let tokenizer = FairseqTokenizer::from_tokens(tokens);
        let result = tokenizer.encode("hello world").unwrap();

        // Should be: <s> hello world </s>
        assert_eq!(result.input_ids, vec![3, 4, 5, 1]);
        assert_eq!(result.attention_mask, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_fairseq_tokenizer_decode() {
        let tokens = vec![("hello".to_string(), 100), ("world".to_string(), 80)];

        let tokenizer = FairseqTokenizer::from_tokens(tokens);
        let decoded = tokenizer.decode(&[3, 4, 5, 1]).unwrap();

        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_fairseq_tokenizer_unk_token() {
        let tokens = vec![("hello".to_string(), 100)];

        let tokenizer = FairseqTokenizer::from_tokens(tokens);
        let result = tokenizer.encode("hello unknown").unwrap();

        // Should be: <s> hello <unk> </s>
        assert_eq!(result.input_ids, vec![3, 4, 2, 1]);
    }

    #[test]
    fn test_fairseq_tokenizer_encode_pair() {
        let tokens = vec![
            ("hello".to_string(), 100),
            ("world".to_string(), 80),
            ("test".to_string(), 60),
        ];

        let tokenizer = FairseqTokenizer::from_tokens(tokens);
        let result = tokenizer.encode_pair("hello", "world test").unwrap();

        // Should be: <s> hello </s> world test </s>
        assert_eq!(result.input_ids, vec![3, 4, 1, 5, 6, 1]);
        assert_eq!(result.token_type_ids, Some(vec![0, 0, 0, 1, 1, 1]));
    }

    #[test]
    fn test_fairseq_tokenizer_max_length() {
        let tokens = vec![
            ("a".to_string(), 100),
            ("b".to_string(), 80),
            ("c".to_string(), 60),
        ];

        let tokenizer = FairseqTokenizer::from_tokens(tokens).with_max_length(5);
        let result = tokenizer.encode("a b c a b c").unwrap();

        assert_eq!(result.input_ids.len(), 5);
        assert_eq!(result.input_ids[result.input_ids.len() - 1], 1); // Should end with </s>
    }

    #[test]
    fn test_fairseq_tokenizer_file_io() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "<pad> 0")?;
        writeln!(temp_file, "</s> 1")?;
        writeln!(temp_file, "<unk> 2")?;
        writeln!(temp_file, "<s> 3")?;
        writeln!(temp_file, "hello 100")?;
        writeln!(temp_file, "world 80")?;
        temp_file.flush()?;

        let tokenizer = FairseqTokenizer::from_file(temp_file.path())?;

        assert_eq!(tokenizer.vocab_size(), 6);
        assert_eq!(tokenizer.token_to_id("hello"), Some(4));
        assert_eq!(tokenizer.get_frequency("hello"), Some(100));

        Ok(())
    }

    #[test]
    fn test_fairseq_dictionary_builder() {
        let mut builder = FairseqDictionaryBuilder::new().min_frequency(2).max_vocab_size(10);

        builder.add_text("hello world hello test");
        builder.add_text("hello again world");

        let tokenizer = builder.build();

        // Should have hello (3 times), world (2 times), but not test or again (1 time each)
        assert!(tokenizer.token_to_id("hello").is_some());
        assert!(tokenizer.token_to_id("world").is_some());
        assert!(tokenizer.token_to_id("test").is_none());
        assert!(tokenizer.token_to_id("again").is_none());
    }

    #[test]
    fn test_get_tokens_by_frequency() {
        let tokens = vec![
            ("world".to_string(), 80),
            ("hello".to_string(), 100),
            ("test".to_string(), 50),
        ];

        let tokenizer = FairseqTokenizer::from_tokens(tokens);
        let sorted_tokens = tokenizer.get_tokens_by_frequency();

        // Should be sorted by frequency descending
        assert_eq!(sorted_tokens[0].0, "hello");
        assert_eq!(sorted_tokens[0].1, 100);
        assert_eq!(sorted_tokens[1].0, "world");
        assert_eq!(sorted_tokens[1].1, 80);
    }
}
