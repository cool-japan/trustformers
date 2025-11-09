use crate::vocab::Vocab;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Configuration for Chinese tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChineseTokenizerConfig {
    /// Whether to enable word segmentation
    pub enable_word_segmentation: bool,
    /// Whether to keep punctuation as separate tokens
    pub keep_punctuation: bool,
    /// Whether to convert to simplified Chinese
    pub convert_to_simplified: bool,
    /// Whether to handle traditional Chinese
    pub handle_traditional: bool,
    /// Maximum word length for segmentation
    pub max_word_length: usize,
    /// Unknown token
    pub unk_token: String,
    /// Padding token
    pub pad_token: String,
    /// Special tokens
    pub special_tokens: Vec<String>,
}

impl Default for ChineseTokenizerConfig {
    fn default() -> Self {
        Self {
            enable_word_segmentation: true,
            keep_punctuation: true,
            convert_to_simplified: false,
            handle_traditional: true,
            max_word_length: 6,
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

/// Chinese tokenizer with word segmentation support
#[derive(Debug, Clone)]
pub struct ChineseTokenizer {
    config: ChineseTokenizerConfig,
    vocab: Vocab,
    /// Built-in dictionary of common Chinese words
    word_dict: HashSet<String>,
    /// Character frequency map for segmentation scoring
    char_freq: HashMap<char, u32>,
    /// Whether normalizer is enabled (simplified approach)
    use_normalizer: bool,
}

impl ChineseTokenizer {
    /// Create a new Chinese tokenizer
    pub fn new(config: ChineseTokenizerConfig, vocab: Vocab) -> Self {
        let mut tokenizer = Self {
            config,
            vocab,
            word_dict: HashSet::new(),
            char_freq: HashMap::new(),
            use_normalizer: false,
        };

        tokenizer.init_builtin_dict();
        tokenizer.init_char_freq();
        tokenizer
    }

    /// Enable basic text normalization
    pub fn with_normalization(mut self) -> Self {
        self.use_normalizer = true;
        self
    }

    /// Initialize built-in Chinese word dictionary
    fn init_builtin_dict(&mut self) {
        // Common Chinese words (this would be loaded from a file in practice)
        let common_words = vec![
            "中国",
            "人民",
            "共和国",
            "北京",
            "上海",
            "广州",
            "深圳",
            "我们",
            "你们",
            "他们",
            "什么",
            "怎么",
            "为什么",
            "在哪里",
            "今天",
            "明天",
            "昨天",
            "现在",
            "以前",
            "以后",
            "时候",
            "可以",
            "应该",
            "需要",
            "想要",
            "希望",
            "觉得",
            "认为",
            "工作",
            "学习",
            "生活",
            "家庭",
            "朋友",
            "同事",
            "老师",
            "学生",
            "医生",
            "警察",
            "司机",
            "厨师",
            "工程师",
            "律师",
            "喜欢",
            "讨厌",
            "爱",
            "恨",
            "高兴",
            "难过",
            "生气",
            "害怕",
            "东西",
            "地方",
            "时间",
            "问题",
            "方法",
            "机会",
            "经验",
            "知识",
            "技能",
            "能力",
            "水平",
            "质量",
            "数量",
            "价格",
        ];

        for word in common_words {
            self.word_dict.insert(word.to_string());
        }
    }

    /// Initialize character frequency map
    fn init_char_freq(&mut self) {
        // Initialize with common Chinese character frequencies
        let common_chars = vec![
            ('的', 1000),
            ('一', 900),
            ('是', 800),
            ('了', 700),
            ('我', 650),
            ('不', 600),
            ('人', 550),
            ('在', 500),
            ('他', 450),
            ('有', 400),
            ('这', 380),
            ('个', 360),
            ('上', 340),
            ('们', 320),
            ('来', 300),
            ('到', 280),
            ('时', 260),
            ('大', 240),
            ('地', 220),
            ('为', 200),
            ('子', 190),
            ('中', 180),
            ('你', 170),
            ('说', 160),
            ('生', 150),
            ('国', 145),
            ('年', 140),
            ('着', 135),
            ('就', 130),
            ('那', 125),
            ('和', 120),
            ('要', 115),
            ('她', 110),
            ('出', 105),
            ('也', 100),
        ];

        for (ch, freq) in common_chars {
            self.char_freq.insert(ch, freq);
        }
    }

    /// Check if a character is Chinese
    pub fn is_chinese_char(ch: char) -> bool {
        let code = ch as u32;
        // CJK Unified Ideographs
        (0x4E00..=0x9FFF).contains(&code) ||
        // CJK Extension A
        (0x3400..=0x4DBF).contains(&code) ||
        // CJK Extension B
        (0x20000..=0x2A6DF).contains(&code) ||
        // CJK Extension C
        (0x2A700..=0x2B73F).contains(&code) ||
        // CJK Extension D
        (0x2B740..=0x2B81F).contains(&code) ||
        // CJK Extension E
        (0x2B820..=0x2CEAF).contains(&code)
    }

    /// Check if a character is Chinese punctuation
    pub fn is_chinese_punctuation(ch: char) -> bool {
        matches!(
            ch,
            '，' | '。'
                | '？'
                | '！'
                | '；'
                | '：'
                | '、'
                | '\u{201C}' // "
                | '\u{201D}' // "
                | '\u{2018}' // '
                | '\u{2019}' // '
                | '《'
                | '》'
                | '（'
                | '）'
                | '【'
                | '】'
                | '—'
                | '…'
        )
    }

    /// Preprocess text by normalizing and handling special cases
    fn preprocess_text(&self, text: &str) -> String {
        let mut processed = if self.use_normalizer {
            // Basic normalization: lowercase and whitespace normalization
            text.to_lowercase()
                .chars()
                .filter(|c| !c.is_whitespace() || *c == ' ')
                .collect()
        } else {
            text.to_string()
        };

        // Convert to simplified Chinese if requested
        if self.config.convert_to_simplified {
            processed = self.traditional_to_simplified(&processed);
        }

        processed
    }

    /// Simple traditional to simplified Chinese conversion
    fn traditional_to_simplified(&self, text: &str) -> String {
        // This is a simplified mapping - in practice, you'd use a comprehensive dictionary
        let mapping = vec![
            ('東', '东'),
            ('習', '习'),
            ('國', '国'),
            ('學', '学'),
            ('長', '长'),
            ('開', '开'),
            ('關', '关'),
            ('時', '时'),
            ('間', '间'),
            ('問', '问'),
            ('題', '题'),
            ('會', '会'),
            ('來', '来'),
            ('說', '说'),
            ('話', '话'),
            ('見', '见'),
            ('覺', '觉'),
            ('經', '经'),
            ('過', '过'),
            ('對', '对'),
            ('現', '现'),
            ('發', '发'),
            ('車', '车'),
            ('門', '门'),
            ('們', '们'),
        ];

        let mut result = text.to_string();
        for (traditional, simplified) in mapping {
            result = result.replace(traditional, &simplified.to_string());
        }
        result
    }

    /// Segment Chinese text into words using dynamic programming
    pub fn segment_text(&self, text: &str) -> Vec<String> {
        if !self.config.enable_word_segmentation {
            return text.chars().map(|c| c.to_string()).collect();
        }

        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }

        let n = chars.len();
        let mut dp = vec![f64::NEG_INFINITY; n + 1];
        let mut path = vec![0; n + 1];
        dp[0] = 0.0;

        for i in 0..n {
            if dp[i] == f64::NEG_INFINITY {
                continue;
            }

            // Try all possible word lengths
            for j in 1..=self.config.max_word_length.min(n - i) {
                let word: String = chars[i..i + j].iter().collect();
                let score = self.calculate_word_score(&word);

                if dp[i] + score > dp[i + j] {
                    dp[i + j] = dp[i] + score;
                    path[i + j] = i;
                }
            }
        }

        // Reconstruct the segmentation
        let mut result = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let start = path[pos];
            let word: String = chars[start..pos].iter().collect();
            result.push(word);
            pos = start;
        }

        result.reverse();
        result
    }

    /// Calculate word score for segmentation
    fn calculate_word_score(&self, word: &str) -> f64 {
        if word.len() == 1 {
            let ch = word.chars().next().unwrap();
            if Self::is_chinese_char(ch) {
                // Single character Chinese words have lower score
                return self.char_freq.get(&ch).map(|&f| (f as f64).ln()).unwrap_or(-10.0);
            } else if Self::is_chinese_punctuation(ch) || ch.is_ascii_punctuation() {
                return 0.0; // Neutral score for punctuation
            } else {
                return -5.0; // Lower score for other single characters
            }
        }

        // Multi-character words
        if self.word_dict.contains(word) {
            // Known words get positive score based on length
            5.0 + word.len() as f64
        } else if word.chars().all(Self::is_chinese_char) {
            // Unknown Chinese words get moderate score
            2.0 + word.len() as f64 * 0.5
        } else if word.chars().all(|c| c.is_ascii_alphanumeric()) {
            // English words/numbers
            3.0 + word.len() as f64 * 0.3
        } else {
            // Mixed or special characters
            1.0
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

    /// Tokenize with proper handling of special tokens
    fn tokenize_segments(&self, segments: Vec<String>) -> Vec<String> {
        let mut tokens = Vec::new();

        for segment in segments {
            // Check if it's a special token
            if self.config.special_tokens.contains(&segment) {
                tokens.push(segment);
                continue;
            }

            // Handle punctuation
            if segment.len() == 1 {
                let ch = segment.chars().next().unwrap();
                if Self::is_chinese_punctuation(ch) || ch.is_ascii_punctuation() {
                    if self.config.keep_punctuation {
                        tokens.push(segment);
                    }
                    continue;
                }
            }

            // Regular token
            tokens.push(segment);
        }

        tokens
    }
}

impl Tokenizer for ChineseTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let processed_text = self.preprocess_text(text);
        let segments = self.segment_text(&processed_text);
        let tokens = self.tokenize_segments(segments);

        let mut token_ids = Vec::new();
        let mut attention_mask = Vec::new();

        for token in &tokens {
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
                result.push_str(&token);
            } else {
                result.push_str(&self.config.unk_token);
            }
        }

        Ok(result)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        let processed_a = self.preprocess_text(text_a);
        let processed_b = self.preprocess_text(text_b);

        let segments_a = self.segment_text(&processed_a);
        let segments_b = self.segment_text(&processed_b);

        let tokens_a = self.tokenize_segments(segments_a);
        let tokens_b = self.tokenize_segments(segments_b);

        let mut all_tokens = tokens_a;
        all_tokens.push("[SEP]".to_string()); // Add separator
        all_tokens.extend(tokens_b);

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
        self.vocab.get_token_to_id_map().clone()
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
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "中", "国", "人", "民", "的", "我", "你",
            "他", "今", "天", "明", "昨", "是", "不", "在", "中国", "人民", "今天", "明天", "昨天",
            "，", "。", "？", "！",
        ];

        for (i, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.to_string(), i as u32);
        }

        Vocab::from_map(token_to_id)
    }

    #[test]
    fn test_chinese_char_detection() {
        assert!(ChineseTokenizer::is_chinese_char('中'));
        assert!(ChineseTokenizer::is_chinese_char('国'));
        assert!(!ChineseTokenizer::is_chinese_char('a'));
        assert!(!ChineseTokenizer::is_chinese_char('1'));
    }

    #[test]
    fn test_chinese_punctuation_detection() {
        assert!(ChineseTokenizer::is_chinese_punctuation('，'));
        assert!(ChineseTokenizer::is_chinese_punctuation('。'));
        assert!(!ChineseTokenizer::is_chinese_punctuation(','));
        assert!(!ChineseTokenizer::is_chinese_punctuation('.'));
    }

    #[test]
    fn test_text_segmentation() {
        let config = ChineseTokenizerConfig::default();
        let vocab = create_test_vocab();
        let tokenizer = ChineseTokenizer::new(config, vocab);

        let segments = tokenizer.segment_text("中国人民");
        assert!(segments.len() > 0);
        assert!(segments.contains(&"中国".to_string()) || segments.contains(&"中".to_string()));
    }

    #[test]
    fn test_tokenization() {
        let config = ChineseTokenizerConfig::default();
        let vocab = create_test_vocab();
        let tokenizer = ChineseTokenizer::new(config, vocab);

        let result = tokenizer.encode("中国人民").unwrap();
        assert!(!result.input_ids.is_empty());
        assert_eq!(result.input_ids.len(), result.attention_mask.len());
    }

    #[test]
    fn test_decode() {
        let config = ChineseTokenizerConfig::default();
        let vocab = create_test_vocab();
        let tokenizer = ChineseTokenizer::new(config, vocab);

        let token_ids = vec![5, 6, 7]; // "中", "国", "人"
        let decoded = tokenizer.decode(&token_ids).unwrap();
        assert_eq!(decoded, "中国人");
    }

    #[test]
    fn test_pair_encoding() {
        let config = ChineseTokenizerConfig::default();
        let vocab = create_test_vocab();
        let tokenizer = ChineseTokenizer::new(config, vocab);

        let result = tokenizer.encode_pair("中国", "人民").unwrap();
        assert!(!result.input_ids.is_empty());
        assert!(result.token_type_ids.is_some());

        let token_type_ids = result.token_type_ids.unwrap();
        assert!(token_type_ids.contains(&0)); // First sequence
        assert!(token_type_ids.contains(&1)); // Second sequence
    }

    #[test]
    fn test_dictionary_management() {
        let config = ChineseTokenizerConfig::default();
        let vocab = create_test_vocab();
        let mut tokenizer = ChineseTokenizer::new(config, vocab);

        let initial_size = tokenizer.dictionary_size();

        tokenizer.add_word("测试".to_string());
        assert_eq!(tokenizer.dictionary_size(), initial_size + 1);
        assert!(tokenizer.contains_word("测试"));

        assert!(tokenizer.remove_word("测试"));
        assert_eq!(tokenizer.dictionary_size(), initial_size);
        assert!(!tokenizer.contains_word("测试"));
    }

    #[test]
    fn test_traditional_to_simplified() {
        let config = ChineseTokenizerConfig {
            convert_to_simplified: true,
            ..Default::default()
        };
        let vocab = create_test_vocab();
        let tokenizer = ChineseTokenizer::new(config, vocab);

        let simplified = tokenizer.traditional_to_simplified("東西");
        assert_eq!(simplified, "东西");
    }
}
