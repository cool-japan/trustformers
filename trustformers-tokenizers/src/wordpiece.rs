use crate::vocab::Vocab;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    vocab: Vocab,
    unk_token: String,
    sep_token: String,
    pad_token: String,
    cls_token: String,
    #[allow(dead_code)]
    mask_token: String,
    do_lower_case: bool,
    max_input_chars_per_word: usize,
}

impl WordPieceTokenizer {
    pub fn new(vocab: HashMap<String, u32>, do_lower_case: bool) -> Self {
        Self {
            vocab: Vocab::from_map(vocab),
            unk_token: "[UNK]".to_string(),
            sep_token: "[SEP]".to_string(),
            pad_token: "[PAD]".to_string(),
            cls_token: "[CLS]".to_string(),
            mask_token: "[MASK]".to_string(),
            do_lower_case,
            max_input_chars_per_word: 100,
        }
    }

    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        // Implement loading from pretrained models
        // First try to load from a local vocab file, then fall back to basic vocab

        // Try to load from potential local paths
        let potential_paths = vec![
            format!("{}/vocab.txt", model_name),
            format!("{}-vocab.txt", model_name),
            format!("models/{}/vocab.txt", model_name),
            format!("./vocab/{}.txt", model_name),
        ];

        for path in potential_paths {
            if let Ok(vocab) = Self::load_vocab_from_file(&path) {
                return Ok(Self::new(vocab, true));
            }
        }

        // Fall back to model-specific default vocabularies
        let vocab = match model_name {
            "bert-base-uncased" | "bert-large-uncased" => Self::create_bert_base_vocab(),
            "bert-base-cased" | "bert-large-cased" => Self::create_bert_cased_vocab(),
            "distilbert-base-uncased" => Self::create_distilbert_vocab(),
            _ => Self::create_basic_vocab(),
        };

        Ok(Self::new(vocab, model_name.contains("uncased")))
    }

    pub fn from_vocab_file(vocab_path: &str, do_lower_case: bool) -> Result<Self> {
        let vocab = Self::load_vocab_from_file(vocab_path)?;
        Ok(Self::new(vocab, do_lower_case))
    }

    /// Load vocabulary from a file
    fn load_vocab_from_file(path: &str) -> Result<HashMap<String, u32>> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to open vocab file {}: {}", path, e))
        })?;
        let reader = BufReader::new(file);

        let mut vocab = HashMap::new();
        for (id, line) in reader.lines().enumerate() {
            let token = line
                .map_err(|e| TrustformersError::io_error(format!("Failed to read line: {}", e)))?;
            let token = token.trim().to_string();
            if !token.is_empty() {
                vocab.insert(token, id as u32);
            }
        }

        if vocab.is_empty() {
            return Err(TrustformersError::other(
                "Empty vocabulary file".to_string(),
            ));
        }

        Ok(vocab)
    }

    /// Create BERT base vocabulary with common tokens
    fn create_bert_base_vocab() -> HashMap<String, u32> {
        let mut vocab = HashMap::new();
        let special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];

        // Add special tokens first
        for (id, token) in special_tokens.iter().enumerate() {
            vocab.insert(token.to_string(), id as u32);
        }

        // Add common English tokens
        let common_tokens = vec![
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back",
            "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new",
            "want", "because", "any", "these", "give", "day", "most", "us",
        ];

        let mut id = special_tokens.len() as u32;
        for token in common_tokens {
            vocab.insert(token.to_string(), id);
            id += 1;
        }

        vocab
    }

    /// Create BERT cased vocabulary (similar to base but preserves case)
    fn create_bert_cased_vocab() -> HashMap<String, u32> {
        let mut vocab = Self::create_bert_base_vocab();
        let base_size = vocab.len() as u32;

        // Add some capitalized versions
        let capitalized = [
            "The", "Be", "To", "Of", "And", "A", "In", "That", "Have", "I",
        ];
        for (i, token) in capitalized.iter().enumerate() {
            vocab.insert(token.to_string(), base_size + i as u32);
        }

        vocab
    }

    /// Create DistilBERT vocabulary (subset of BERT)
    fn create_distilbert_vocab() -> HashMap<String, u32> {
        let bert_vocab = Self::create_bert_base_vocab();
        // DistilBERT uses a smaller vocabulary - take first 75% of BERT vocab
        let target_size = (bert_vocab.len() * 3) / 4;

        bert_vocab.into_iter().take(target_size).collect()
    }

    /// Create basic vocabulary for unknown models
    fn create_basic_vocab() -> HashMap<String, u32> {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("[CLS]".to_string(), 1);
        vocab.insert("[SEP]".to_string(), 2);
        vocab.insert("[PAD]".to_string(), 3);
        vocab.insert("[MASK]".to_string(), 4);

        // Add basic English alphabet and common punctuation
        let mut id = 5;
        for c in 'a'..='z' {
            vocab.insert(c.to_string(), id);
            id += 1;
        }

        for punct in [".", ",", "!", "?", ";", ":", "'", "\"", "-"] {
            vocab.insert(punct.to_string(), id);
            id += 1;
        }

        vocab
    }

    fn basic_tokenize(&self, text: &str) -> Vec<String> {
        let text = if self.do_lower_case { text.to_lowercase() } else { text.to_string() };

        let text = text.nfc().collect::<String>();

        text.split_whitespace().map(|s| s.to_string()).collect()
    }

    fn wordpiece_tokenize(&self, word: &str) -> Vec<String> {
        if word.chars().count() > self.max_input_chars_per_word {
            return vec![self.unk_token.clone()];
        }

        let mut output = Vec::new();
        let mut start = 0;
        let chars: Vec<char> = word.chars().collect();

        while start < chars.len() {
            let mut end = chars.len();
            let mut cur_substr = None;

            while start < end {
                let substr = if start > 0 {
                    format!("##{}", chars[start..end].iter().collect::<String>())
                } else {
                    chars[start..end].iter().collect::<String>()
                };

                if self.vocab.contains(&substr) {
                    cur_substr = Some(substr);
                    break;
                }

                end -= 1;
            }

            if let Some(substr) = cur_substr {
                output.push(substr);
                start = end;
            } else {
                output.push(self.unk_token.clone());
                break;
            }
        }

        output
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let mut tokens = vec![self.cls_token.clone()];

        for word in self.basic_tokenize(text) {
            tokens.extend(self.wordpiece_tokenize(&word));
        }

        tokens.push(self.sep_token.clone());

        let input_ids: Vec<u32> = tokens
            .iter()
            .map(|token| {
                self.vocab.get_id(token).unwrap_or(self.vocab.get_id(&self.unk_token).unwrap())
            })
            .collect();

        let attention_mask = vec![1u8; input_ids.len()];

        let input_ids_len = input_ids.len();
        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids: Some(vec![0u32; input_ids_len]),
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        let mut tokens = vec![self.cls_token.clone()];

        for word in self.basic_tokenize(text) {
            tokens.extend(self.wordpiece_tokenize(&word));
        }

        tokens.push(self.sep_token.clone());
        let first_seg_len = tokens.len();

        for word in self.basic_tokenize(text2) {
            tokens.extend(self.wordpiece_tokenize(&word));
        }

        tokens.push(self.sep_token.clone());

        let input_ids: Vec<u32> = tokens
            .iter()
            .map(|token| {
                self.vocab.get_id(token).unwrap_or(self.vocab.get_id(&self.unk_token).unwrap())
            })
            .collect();

        let attention_mask = vec![1u8; input_ids.len()];

        let mut token_type_ids = vec![0u32; first_seg_len];
        token_type_ids.extend(vec![1u32; input_ids.len() - first_seg_len]);

        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids: Some(token_type_ids),
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let tokens: Vec<String> = ids.iter().filter_map(|&id| self.vocab.get_token(id)).collect();

        let text = tokens
            .join(" ")
            .replace(" ##", "")
            .replace(&format!(" {} ", self.pad_token), " ")
            .replace(&format!(" {} ", self.cls_token), " ")
            .replace(&format!(" {} ", self.sep_token), " ")
            .trim()
            .to_string();

        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.get_token_to_id_map().clone()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get_token(id)
    }
}
