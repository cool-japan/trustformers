use crate::vocab::Vocab;
#[cfg(feature = "mecab")]
use mecab::Tagger;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
#[cfg(feature = "mecab")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "mecab")]
use trustformers_core::errors::ErrorKind;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Configuration for Japanese tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JapaneseTokenizerConfig {
    /// Whether to use MeCab for morphological analysis
    pub use_mecab: bool,
    /// MeCab configuration string
    pub mecab_config: Option<String>,
    /// Tokenization mode: "word", "morpheme", or "character"
    pub mode: JapaneseMode,
    /// Whether to keep punctuation as separate tokens
    pub keep_punctuation: bool,
    /// Whether to normalize katakana to hiragana
    pub normalize_katakana: bool,
    /// Whether to split compound words
    pub split_compounds: bool,
    /// Whether to include part-of-speech tags
    pub include_pos: bool,
    /// Maximum word length for character mode
    pub max_word_length: usize,
    /// Unknown token
    pub unk_token: String,
    /// Padding token
    pub pad_token: String,
    /// Special tokens
    pub special_tokens: Vec<String>,
}

/// Tokenization mode for Japanese text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JapaneseMode {
    /// Word-level tokenization
    Word,
    /// Morpheme-level tokenization with MeCab
    Morpheme,
    /// Character-level tokenization
    Character,
}

impl Default for JapaneseTokenizerConfig {
    fn default() -> Self {
        Self {
            use_mecab: cfg!(feature = "mecab"),
            mecab_config: None,
            mode: JapaneseMode::Morpheme,
            keep_punctuation: true,
            normalize_katakana: false,
            split_compounds: false,
            include_pos: false,
            max_word_length: 10,
            unk_token: "[UNK]".to_string(),
            pad_token: "[PAD]".to_string(),
            special_tokens: vec![
                "[PAD]".to_string(),
                "[UNK]".to_string(),
                "[CLS]".to_string(),
                "[SEP]".to_string(),
                "[MASK]".to_string(),
            ],
        }
    }
}

#[cfg(feature = "mecab")]
/// Thread-safe wrapper for MeCab Tagger
struct ThreadSafeTagger {
    tagger: Arc<Mutex<Tagger>>,
}

#[cfg(feature = "mecab")]
// MeCab's Tagger is not inherently thread-safe, but we protect it with a Mutex
// This is safe because:
// 1. We never access the raw pointers directly
// 2. All access is protected by the Mutex
// 3. We don't share the Tagger between threads without synchronization
unsafe impl Send for ThreadSafeTagger {}
#[cfg(feature = "mecab")]
unsafe impl Sync for ThreadSafeTagger {}

#[cfg(feature = "mecab")]
impl ThreadSafeTagger {
    fn new(config: &str) -> Result<Self> {
        let tagger = Tagger::new(config);
        Ok(Self {
            tagger: Arc::new(Mutex::new(tagger)),
        })
    }

    fn parse_to_node(&self, text: &str) -> Result<Vec<(String, String)>> {
        let mut tagger = self.tagger.lock().unwrap();
        let node = tagger.parse_to_node(text);
        let mut result = Vec::new();

        let mut current = node;
        loop {
            let surface = current.surface.clone();
            let feature = current.feature.clone();

            if !surface.is_empty() && feature != "BOS/EOS" {
                result.push((surface, feature));
            }

            if let Some(next) = current.next() {
                current = next;
            } else {
                break;
            }
        }

        Ok(result)
    }
}

#[cfg(feature = "mecab")]
impl std::fmt::Debug for ThreadSafeTagger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ThreadSafeTagger")
    }
}

/// Japanese tokenizer with MeCab integration
#[derive(Debug)]
pub struct JapaneseTokenizer {
    config: JapaneseTokenizerConfig,
    vocab: Vocab,
    #[cfg(feature = "mecab")]
    mecab: Option<ThreadSafeTagger>,
    /// Built-in dictionary of common Japanese words
    word_dict: HashSet<String>,
    /// Whether normalizer is enabled
    use_normalizer: bool,
}

impl JapaneseTokenizer {
    /// Create a new Japanese tokenizer
    pub fn new(config: JapaneseTokenizerConfig, vocab: Vocab) -> Result<Self> {
        #[cfg(feature = "mecab")]
        let mecab = if config.use_mecab {
            let mecab_config = config.mecab_config.as_deref().unwrap_or("");
            Some(ThreadSafeTagger::new(mecab_config)?)
        } else {
            None
        };

        let mut tokenizer = Self {
            config,
            vocab,
            #[cfg(feature = "mecab")]
            mecab,
            word_dict: HashSet::new(),
            use_normalizer: false,
        };

        tokenizer.init_builtin_dict();
        Ok(tokenizer)
    }

    /// Enable basic text normalization
    pub fn with_normalization(mut self) -> Self {
        self.use_normalizer = true;
        self
    }

    /// Initialize built-in Japanese word dictionary
    fn init_builtin_dict(&mut self) {
        // Common Japanese words
        let common_words = vec![
            "日本",
            "東京",
            "大阪",
            "名古屋",
            "福岡",
            "札幌",
            "私",
            "僕",
            "俺",
            "彼",
            "彼女",
            "あなた",
            "君",
            "今日",
            "明日",
            "昨日",
            "今",
            "先",
            "後",
            "時",
            "する",
            "した",
            "します",
            "できる",
            "ある",
            "いる",
            "なる",
            "とても",
            "少し",
            "たくさん",
            "全部",
            "一部",
            "何か",
            "誰か",
            "学校",
            "会社",
            "家",
            "店",
            "病院",
            "駅",
            "空港",
            "先生",
            "学生",
            "医者",
            "警察",
            "運転手",
            "料理人",
            "技術者",
            "好き",
            "嫌い",
            "愛",
            "恨み",
            "嬉しい",
            "悲しい",
            "怒り",
            "恐怖",
            "もの",
            "こと",
            "場所",
            "時間",
            "問題",
            "方法",
            "機会",
            "経験",
            "知識",
            "技術",
            "能力",
            "水準",
            "質",
            "量",
            "価格",
        ];

        for word in common_words {
            self.word_dict.insert(word.to_string());
        }
    }

    /// Check if a character is Hiragana
    pub fn is_hiragana(ch: char) -> bool {
        let code = ch as u32;
        (0x3040..=0x309F).contains(&code)
    }

    /// Check if a character is Katakana
    pub fn is_katakana(ch: char) -> bool {
        let code = ch as u32;
        (0x30A0..=0x30FF).contains(&code)
    }

    /// Check if a character is Kanji
    pub fn is_kanji(ch: char) -> bool {
        let code = ch as u32;
        // CJK Unified Ideographs
        (0x4E00..=0x9FFF).contains(&code) ||
        // CJK Extension A
        (0x3400..=0x4DBF).contains(&code) ||
        // CJK Extension B
        (0x20000..=0x2A6DF).contains(&code)
    }

    /// Check if a character is Japanese
    pub fn is_japanese_char(ch: char) -> bool {
        Self::is_hiragana(ch) || Self::is_katakana(ch) || Self::is_kanji(ch)
    }

    /// Check if a character is Japanese punctuation
    pub fn is_japanese_punctuation(ch: char) -> bool {
        matches!(
            ch,
            '。' | '、'
                | '？'
                | '！'
                | '：'
                | '；'
                | '「'
                | '」'
                | '『'
                | '』'
                | '（'
                | '）'
                | '｛'
                | '｝'
                | '【'
                | '】'
                | '・'
                | '…'
                | '〜'
                | '～'
                | 'ー'
                | '￥'
        )
    }

    /// Convert Katakana to Hiragana
    pub fn katakana_to_hiragana(text: &str) -> String {
        text.chars()
            .map(|ch| {
                if Self::is_katakana(ch) {
                    // Convert Katakana to Hiragana by subtracting 0x60
                    char::from_u32(ch as u32 - 0x60).unwrap_or(ch)
                } else {
                    ch
                }
            })
            .collect()
    }

    /// Preprocess text by normalizing and handling special cases
    fn preprocess_text(&self, text: &str) -> String {
        let mut processed = if self.use_normalizer {
            // Basic normalization: full-width to half-width and whitespace normalization
            text.chars()
                .map(|c| {
                    // Convert full-width ASCII to half-width
                    if (0xFF01..=0xFF5E).contains(&(c as u32)) {
                        char::from_u32(c as u32 - 0xFEE0).unwrap_or(c)
                    } else {
                        c
                    }
                })
                .filter(|c| !c.is_whitespace() || *c == ' ')
                .collect()
        } else {
            text.to_string()
        };

        // Normalize katakana to hiragana if requested
        if self.config.normalize_katakana {
            processed = Self::katakana_to_hiragana(&processed);
        }

        processed
    }

    /// Tokenize using MeCab morphological analyzer
    #[cfg(feature = "mecab")]
    fn tokenize_with_mecab(&self, text: &str) -> Result<Vec<String>> {
        let mecab = self.mecab.as_ref().ok_or_else(|| {
            TrustformersError::new(ErrorKind::TokenizationError {
                reason: "MeCab not initialized".to_string(),
            })
        })?;

        let mut tokens = Vec::new();
        let node_data = mecab.parse_to_node(text)?;

        for (surface, feature) in node_data {
            let token = if self.config.include_pos {
                format!("{}#{}", surface, feature.split(',').next().unwrap_or(""))
            } else {
                surface
            };
            tokens.push(token);
        }

        Ok(tokens)
    }

    /// Tokenize using simple character-based approach
    fn tokenize_characters(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut tokens = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];

            // Handle punctuation
            if Self::is_japanese_punctuation(ch) || ch.is_ascii_punctuation() {
                if self.config.keep_punctuation {
                    tokens.push(ch.to_string());
                }
                i += 1;
                continue;
            }

            // Handle special characters
            if ch.is_whitespace() {
                i += 1;
                continue;
            }

            // Regular character
            tokens.push(ch.to_string());
            i += 1;
        }

        tokens
    }

    /// Tokenize using word-based approach
    fn tokenize_words(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();
        let mut current_type = None;

        for ch in text.chars() {
            let char_type = if Self::is_hiragana(ch) {
                "hiragana"
            } else if Self::is_katakana(ch) {
                "katakana"
            } else if Self::is_kanji(ch) {
                "kanji"
            } else if ch.is_ascii_alphanumeric() {
                "ascii"
            } else if Self::is_japanese_punctuation(ch) || ch.is_ascii_punctuation() {
                "punct"
            } else {
                "other"
            };

            if char_type == "punct" {
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
                if self.config.keep_punctuation {
                    tokens.push(ch.to_string());
                }
                current_type = None;
            } else if char_type == "other" || ch.is_whitespace() {
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
                current_type = None;
            } else {
                // Check if we need to break the word
                if let Some(prev_type) = current_type {
                    if prev_type != char_type
                        || (char_type == "kanji"
                            && current_word.len() >= self.config.max_word_length)
                    {
                        tokens.push(current_word.clone());
                        current_word.clear();
                    }
                }
                current_word.push(ch);
                current_type = Some(char_type);
            }
        }

        if !current_word.is_empty() {
            tokens.push(current_word);
        }

        tokens
    }

    /// Tokenize text based on the configured mode
    pub fn tokenize_text(&self, text: &str) -> Result<Vec<String>> {
        let processed_text = self.preprocess_text(text);

        match self.config.mode {
            #[cfg(feature = "mecab")]
            JapaneseMode::Morpheme if self.config.use_mecab => {
                self.tokenize_with_mecab(&processed_text)
            },
            JapaneseMode::Word => Ok(self.tokenize_words(&processed_text)),
            JapaneseMode::Character => Ok(self.tokenize_characters(&processed_text)),
            JapaneseMode::Morpheme => {
                // Fallback to word mode if MeCab is not available
                Ok(self.tokenize_words(&processed_text))
            },
        }
    }

    /// Add word to dictionary
    pub fn add_word(&mut self, word: String) {
        self.word_dict.insert(word);
    }

    /// Remove word from dictionary
    pub fn remove_word(&mut self, word: &str) -> bool {
        self.word_dict.remove(word)
    }

    /// Load dictionary from text file
    pub fn load_dictionary(&mut self, words: Vec<String>) {
        for word in words {
            self.word_dict.insert(word);
        }
    }

    /// Get dictionary size
    pub fn dictionary_size(&self) -> usize {
        self.word_dict.len()
    }

    /// Check if word is in dictionary
    pub fn contains_word(&self, word: &str) -> bool {
        self.word_dict.contains(word)
    }

    /// Process tokens to handle special cases
    fn process_tokens(&self, tokens: Vec<String>) -> Vec<String> {
        let mut processed = Vec::new();

        for token in tokens {
            // Check if it's a special token
            if self.config.special_tokens.contains(&token) {
                processed.push(token);
                continue;
            }

            // Skip empty tokens
            if token.is_empty() {
                continue;
            }

            processed.push(token);
        }

        processed
    }
}

impl Tokenizer for JapaneseTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let tokens = self.tokenize_text(text)?;
        let processed_tokens = self.process_tokens(tokens);

        let mut token_ids = Vec::new();
        let mut attention_mask = Vec::new();

        for token in &processed_tokens {
            if let Some(id) = self.vocab.get_id(token) {
                token_ids.push(id);
                attention_mask.push(1);
            } else {
                // Use UNK token
                if let Some(unk_id) = self.vocab.get_id(&self.config.unk_token) {
                    token_ids.push(unk_id);
                    attention_mask.push(1);
                } else {
                    return Err(TrustformersError::other(
                        "UNK token not found in vocabulary".to_string(),
                    ));
                }
            }
        }

        Ok(TokenizedInput {
            input_ids: token_ids,
            attention_mask,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut result = String::new();

        for &token_id in token_ids {
            if let Some(token) = self.vocab.get_token(token_id) {
                // Skip padding tokens
                if *token == self.config.pad_token {
                    continue;
                }

                // Handle POS tags if included
                if self.config.include_pos && token.contains('#') {
                    let parts: Vec<&str> = token.split('#').collect();
                    if !parts.is_empty() {
                        result.push_str(parts[0]);
                    }
                } else {
                    result.push_str(&token);
                }
            } else {
                result.push_str(&self.config.unk_token);
            }
        }

        Ok(result)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        let tokens_a = self.tokenize_text(text_a)?;
        let tokens_b = self.tokenize_text(text_b)?;

        let processed_a = self.process_tokens(tokens_a);
        let processed_b = self.process_tokens(tokens_b);

        let mut all_tokens = processed_a;
        all_tokens.push("[SEP]".to_string());
        all_tokens.extend(processed_b);

        let mut token_ids = Vec::new();
        let mut attention_mask = Vec::new();
        let mut token_type_ids = Vec::new();

        let sep_pos = all_tokens.iter().position(|t| t == "[SEP]").unwrap_or(0);

        for (i, token) in all_tokens.iter().enumerate() {
            if let Some(id) = self.vocab.get_id(token) {
                token_ids.push(id);
                attention_mask.push(1);
                token_type_ids.push(if i <= sep_pos { 0 } else { 1 });
            } else if let Some(unk_id) = self.vocab.get_id(&self.config.unk_token) {
                token_ids.push(unk_id);
                attention_mask.push(1);
                token_type_ids.push(if i <= sep_pos { 0 } else { 1 });
            } else {
                return Err(TrustformersError::other(
                    "UNK token not found in vocabulary".to_string(),
                ));
            }
        }

        Ok(TokenizedInput {
            input_ids: token_ids,
            attention_mask,
            token_type_ids: Some(token_type_ids),
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.get_vocab().clone()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocab;
    use std::collections::HashMap;

    fn create_test_vocab() -> Vocab {
        let mut token_to_id = HashMap::new();

        let tokens = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "日",
            "本",
            "語",
            "私",
            "は",
            "です",
            "ます",
            "こんにちは",
            "ありがとう",
            "すみません",
            "はい",
            "いいえ",
            "今日",
            "明日",
            "昨日",
            "時間",
            "場所",
            "人",
            "あ",
            "い",
            "う",
            "え",
            "お",
            "か",
            "き",
            "く",
            "け",
            "こ",
            "ア",
            "イ",
            "ウ",
            "エ",
            "オ",
            "カ",
            "キ",
            "ク",
            "ケ",
            "コ",
            "。",
            "、",
            "？",
            "！",
            "：",
            "；",
            "「",
            "」",
        ];

        for (i, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.to_string(), i as u32);
        }

        Vocab::from_map(token_to_id)
    }

    #[test]
    fn test_character_type_detection() {
        assert!(JapaneseTokenizer::is_hiragana('あ'));
        assert!(JapaneseTokenizer::is_hiragana(
            "ひらがな".chars().next().unwrap()
        ));
        assert!(JapaneseTokenizer::is_katakana('ア'));
        assert!(JapaneseTokenizer::is_katakana(
            "カタカナ".chars().next().unwrap()
        ));
        assert!(JapaneseTokenizer::is_kanji('日'));
        assert!(JapaneseTokenizer::is_kanji('本'));
        assert!(!JapaneseTokenizer::is_japanese_char('a'));
        assert!(!JapaneseTokenizer::is_japanese_char('1'));
    }

    #[test]
    fn test_japanese_punctuation_detection() {
        assert!(JapaneseTokenizer::is_japanese_punctuation('。'));
        assert!(JapaneseTokenizer::is_japanese_punctuation('、'));
        assert!(JapaneseTokenizer::is_japanese_punctuation('？'));
        assert!(JapaneseTokenizer::is_japanese_punctuation('！'));
        assert!(!JapaneseTokenizer::is_japanese_punctuation('.'));
        assert!(!JapaneseTokenizer::is_japanese_punctuation(','));
    }

    #[test]
    fn test_katakana_to_hiragana() {
        let result = JapaneseTokenizer::katakana_to_hiragana("カタカナ");
        assert_eq!(result, "かたかな");

        let mixed = JapaneseTokenizer::katakana_to_hiragana("こんにちはカタカナ");
        assert_eq!(mixed, "こんにちはかたかな");
    }

    #[test]
    fn test_character_tokenization() {
        let config = JapaneseTokenizerConfig {
            mode: JapaneseMode::Character,
            use_mecab: false,
            ..Default::default()
        };
        let vocab = create_test_vocab();
        let tokenizer = JapaneseTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.tokenize_text("こんにちは").unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result, vec!["こ", "ん", "に", "ち", "は"]);
    }

    #[test]
    fn test_word_tokenization() {
        let config = JapaneseTokenizerConfig {
            mode: JapaneseMode::Word,
            use_mecab: false,
            ..Default::default()
        };
        let vocab = create_test_vocab();
        let tokenizer = JapaneseTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.tokenize_text("こんにちは世界").unwrap();
        assert!(result.len() > 0);
        // Should separate hiragana from kanji
        assert!(result.iter().any(|t| t.chars().all(JapaneseTokenizer::is_hiragana)));
        assert!(result.iter().any(|t| t.chars().all(JapaneseTokenizer::is_kanji)));
    }

    #[test]
    fn test_tokenization_encode_decode() {
        let config = JapaneseTokenizerConfig {
            mode: JapaneseMode::Character,
            use_mecab: false,
            ..Default::default()
        };
        let vocab = create_test_vocab();
        let tokenizer = JapaneseTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.encode("日本語").unwrap();
        assert!(!result.input_ids.is_empty());
        assert_eq!(result.input_ids.len(), result.attention_mask.len());

        let decoded = tokenizer.decode(&result.input_ids).unwrap();
        assert_eq!(decoded, "日本語");
    }

    #[test]
    fn test_pair_encoding() {
        let config = JapaneseTokenizerConfig {
            mode: JapaneseMode::Character,
            use_mecab: false,
            ..Default::default()
        };
        let vocab = create_test_vocab();
        let tokenizer = JapaneseTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.encode_pair("こんにちは", "世界").unwrap();
        assert!(!result.input_ids.is_empty());
        assert!(result.token_type_ids.is_some());

        let token_type_ids = result.token_type_ids.unwrap();
        assert!(token_type_ids.contains(&0)); // First sequence
        assert!(token_type_ids.contains(&1)); // Second sequence
    }

    #[test]
    fn test_dictionary_management() {
        let config = JapaneseTokenizerConfig::default();
        let vocab = create_test_vocab();
        let mut tokenizer = JapaneseTokenizer::new(config, vocab).unwrap();

        let initial_size = tokenizer.dictionary_size();

        tokenizer.add_word("テスト".to_string());
        assert_eq!(tokenizer.dictionary_size(), initial_size + 1);
        assert!(tokenizer.contains_word("テスト"));

        assert!(tokenizer.remove_word("テスト"));
        assert_eq!(tokenizer.dictionary_size(), initial_size);
        assert!(!tokenizer.contains_word("テスト"));
    }

    #[test]
    fn test_normalization() {
        let config = JapaneseTokenizerConfig {
            normalize_katakana: true,
            mode: JapaneseMode::Character,
            use_mecab: false,
            ..Default::default()
        };
        let vocab = create_test_vocab();
        let tokenizer = JapaneseTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.tokenize_text("カタカナ").unwrap();
        // Should be normalized to hiragana
        assert!(result.iter().all(|t| t.chars().all(JapaneseTokenizer::is_hiragana)));
    }
}
