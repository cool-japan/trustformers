use crate::vocab::Vocab;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Configuration for Korean tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KoreanTokenizerConfig {
    /// Tokenization mode: "syllable", "jamo", or "word"
    pub mode: KoreanMode,
    /// Whether to decompose Hangul syllables into jamo
    pub decompose_hangul: bool,
    /// Whether to keep punctuation as separate tokens
    pub keep_punctuation: bool,
    /// Whether to normalize spacing
    pub normalize_spacing: bool,
    /// Whether to handle Hanja (Chinese characters in Korean)
    pub handle_hanja: bool,
    /// Maximum word length for word mode
    pub max_word_length: usize,
    /// Unknown token
    pub unk_token: String,
    /// Padding token
    pub pad_token: String,
    /// Special tokens
    pub special_tokens: Vec<String>,
}

/// Tokenization mode for Korean text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KoreanMode {
    /// Syllable-level tokenization (default for Korean)
    Syllable,
    /// Jamo-level tokenization (consonants and vowels)
    Jamo,
    /// Word-level tokenization
    Word,
}

impl Default for KoreanTokenizerConfig {
    fn default() -> Self {
        Self {
            mode: KoreanMode::Syllable,
            decompose_hangul: false,
            keep_punctuation: true,
            normalize_spacing: true,
            handle_hanja: true,
            max_word_length: 15,
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

/// Korean tokenizer with Hangul processing
#[derive(Debug)]
pub struct KoreanTokenizer {
    config: KoreanTokenizerConfig,
    vocab: Vocab,
    /// Built-in dictionary of common Korean words
    word_dict: HashSet<String>,
    /// Common Korean particles and endings
    particles: HashSet<String>,
    /// Whether normalizer is enabled
    use_normalizer: bool,
}

impl KoreanTokenizer {
    /// Create a new Korean tokenizer
    pub fn new(config: KoreanTokenizerConfig, vocab: Vocab) -> Result<Self> {
        let mut tokenizer = Self {
            config,
            vocab,
            word_dict: HashSet::new(),
            particles: HashSet::new(),
            use_normalizer: false,
        };

        tokenizer.init_builtin_dict();
        tokenizer.init_particles();
        Ok(tokenizer)
    }

    /// Enable basic text normalization
    pub fn with_normalization(mut self) -> Self {
        self.use_normalizer = true;
        self
    }

    /// Initialize built-in Korean word dictionary
    fn init_builtin_dict(&mut self) {
        // Common Korean words
        let common_words = vec![
            "한국",
            "서울",
            "부산",
            "대구",
            "인천",
            "광주",
            "대전",
            "울산",
            "나",
            "너",
            "그",
            "그녀",
            "우리",
            "당신",
            "이",
            "저",
            "오늘",
            "내일",
            "어제",
            "지금",
            "전",
            "후",
            "시간",
            "하다",
            "되다",
            "있다",
            "없다",
            "가다",
            "오다",
            "보다",
            "말하다",
            "매우",
            "조금",
            "많이",
            "전부",
            "일부",
            "무엇",
            "누구",
            "학교",
            "회사",
            "집",
            "가게",
            "병원",
            "역",
            "공항",
            "선생님",
            "학생",
            "의사",
            "경찰",
            "운전사",
            "요리사",
            "엔지니어",
            "좋다",
            "싫다",
            "사랑",
            "미움",
            "기쁘다",
            "슬프다",
            "화나다",
            "무섭다",
            "것",
            "일",
            "장소",
            "시간",
            "문제",
            "방법",
            "기회",
            "경험",
            "지식",
            "기술",
            "능력",
            "수준",
            "질",
            "양",
            "가격",
        ];

        for word in common_words {
            self.word_dict.insert(word.to_string());
        }
    }

    /// Initialize common Korean particles and endings
    fn init_particles(&mut self) {
        let particles = vec![
            "이",
            "가",
            "을",
            "를",
            "의",
            "에",
            "에서",
            "로",
            "으로",
            "와",
            "과",
            "하고",
            "도",
            "만",
            "까지",
            "부터",
            "처럼",
            "다",
            "아",
            "어",
            "여",
            "지",
            "고",
            "면",
            "서",
            "니",
            "요",
            "습니다",
            "입니다",
            "였습니다",
            "했습니다",
        ];

        for particle in particles {
            self.particles.insert(particle.to_string());
        }
    }

    /// Check if a character is Hangul syllable
    pub fn is_hangul_syllable(ch: char) -> bool {
        let code = ch as u32;
        // Hangul Syllables block: U+AC00-U+D7AF
        (0xAC00..=0xD7AF).contains(&code)
    }

    /// Check if a character is Hangul Jamo (consonant or vowel)
    pub fn is_hangul_jamo(ch: char) -> bool {
        let code = ch as u32;
        // Hangul Jamo block: U+1100-U+11FF
        // Hangul Compatibility Jamo: U+3130-U+318F
        (0x1100..=0x11FF).contains(&code) || (0x3130..=0x318F).contains(&code)
    }

    /// Check if a character is Korean (Hangul or Hanja)
    pub fn is_korean_char(ch: char) -> bool {
        Self::is_hangul_syllable(ch) || Self::is_hangul_jamo(ch) || Self::is_hanja(ch)
    }

    /// Check if a character is Hanja (Chinese characters used in Korean)
    pub fn is_hanja(ch: char) -> bool {
        let code = ch as u32;
        // CJK Unified Ideographs commonly used in Korean
        (0x4E00..=0x9FFF).contains(&code)
    }

    /// Check if a character is Korean punctuation
    pub fn is_korean_punctuation(ch: char) -> bool {
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
                | '·'
                | '…'
                | '〜'
                | '～'
                | 'ㆍ'
        )
    }

    /// Decompose a Hangul syllable into jamo (custom implementation)
    pub fn decompose_hangul(ch: char) -> Option<(char, char, Option<char>)> {
        if !Self::is_hangul_syllable(ch) {
            return None;
        }

        let syllable = ch as u32;
        let base = syllable - 0xAC00;

        // Constants for Hangul decomposition
        const JONGSEONG_COUNT: u32 = 28;
        const JUNGSEONG_COUNT: u32 = 21;

        let jongseong_index = base % JONGSEONG_COUNT;
        let jungseong_index = (base / JONGSEONG_COUNT) % JUNGSEONG_COUNT;
        let choseong_index = base / (JONGSEONG_COUNT * JUNGSEONG_COUNT);

        // Jamo arrays (simplified)
        let choseong = [
            'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ',
            'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
        ];
        let jungseong = [
            'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ',
            'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ',
        ];
        let jongseong = [
            None,
            Some('ㄱ'),
            Some('ㄲ'),
            Some('ㄳ'),
            Some('ㄴ'),
            Some('ㄵ'),
            Some('ㄶ'),
            Some('ㄷ'),
            Some('ㄹ'),
            Some('ㄺ'),
            Some('ㄻ'),
            Some('ㄼ'),
            Some('ㄽ'),
            Some('ㄾ'),
            Some('ㄿ'),
            Some('ㅀ'),
            Some('ㅁ'),
            Some('ㅂ'),
            Some('ㅄ'),
            Some('ㅅ'),
            Some('ㅆ'),
            Some('ㅇ'),
            Some('ㅈ'),
            Some('ㅊ'),
            Some('ㅋ'),
            Some('ㅌ'),
            Some('ㅍ'),
            Some('ㅎ'),
        ];

        let initial = choseong.get(choseong_index as usize).copied()?;
        let medial = jungseong.get(jungseong_index as usize).copied()?;
        let final_ = jongseong.get(jongseong_index as usize).copied().flatten();

        Some((initial, medial, final_))
    }

    /// Compose jamo into a Hangul syllable (custom implementation)
    pub fn compose_hangul(initial: char, medial: char, final_: Option<char>) -> Option<char> {
        // Hangul composition constants
        const HANGUL_BASE: u32 = 0xAC00;
        const JUNGSEONG_COUNT: u32 = 21;
        const JONGSEONG_COUNT: u32 = 28;

        // Initial consonants (choseong) - Unicode Jamo block
        const CHOSEONG: &[char] = &[
            'ᄀ', 'ᄁ', 'ᄂ', 'ᄃ', 'ᄄ', 'ᄅ', 'ᄆ', 'ᄇ', 'ᄈ', 'ᄉ', 'ᄊ', 'ᄋ', 'ᄌ', 'ᄍ',
            'ᄎ', 'ᄏ', 'ᄐ', 'ᄑ', 'ᄒ',
        ];

        // Medial vowels (jungseong)
        const JUNGSEONG: &[char] = &[
            'ᅡ', 'ᅢ', 'ᅣ', 'ᅤ', 'ᅥ', 'ᅦ', 'ᅧ', 'ᅨ', 'ᅩ', 'ᅪ', 'ᅫ', 'ᅬ', 'ᅭ', 'ᅮ', 'ᅯ', 'ᅰ', 'ᅱ', 'ᅲ', 'ᅳ', 'ᅴ', 'ᅵ',
        ];

        // Final consonants (jongseong) - None represented as first element
        const JONGSEONG: &[char] = &[
            '\0', 'ᆨ', 'ᆩ', 'ᆪ', 'ᆫ', 'ᆬ', 'ᆭ', 'ᆮ', 'ᆯ', 'ᆰ', 'ᆱ', 'ᆲ', 'ᆳ', 'ᆴ', 'ᆵ', 'ᆶ', 'ᆷ', 'ᆸ', 'ᆹ', 'ᆺ', 'ᆻ',
            'ᆼ', 'ᆽ', 'ᆾ', 'ᆿ', 'ᇀ', 'ᇁ', 'ᇂ',
        ];

        // Find indices for each component
        let initial_index = CHOSEONG.iter().position(|&c| c == initial)?;
        let medial_index = JUNGSEONG.iter().position(|&c| c == medial)?;
        let final_index = if let Some(f) = final_ {
            JONGSEONG.iter().position(|&c| c == f)?
        } else {
            0 // No final consonant
        };

        // Compose the syllable using the Hangul algorithm
        let syllable_code = HANGUL_BASE
            + (initial_index as u32 * JUNGSEONG_COUNT * JONGSEONG_COUNT)
            + (medial_index as u32 * JONGSEONG_COUNT)
            + final_index as u32;

        char::from_u32(syllable_code)
    }

    /// Preprocess text by normalizing and handling special cases
    fn preprocess_text(&self, text: &str) -> String {
        let mut processed = if self.use_normalizer {
            // Basic normalization: normalize spacing and remove extra whitespace
            text.chars()
                .map(|c| {
                    // Convert full-width characters to half-width
                    if (0xFF01..=0xFF5E).contains(&(c as u32)) {
                        char::from_u32(c as u32 - 0xFEE0).unwrap_or(c)
                    } else {
                        c
                    }
                })
                .collect::<String>()
        } else {
            text.to_string()
        };

        // Normalize spacing if requested
        if self.config.normalize_spacing {
            processed = processed
                .chars()
                .collect::<Vec<char>>()
                .windows(2)
                .map(|w| {
                    let curr = w[0];
                    let next = w[1];

                    // Add space between Korean and non-Korean characters
                    if (Self::is_korean_char(curr)
                        && !Self::is_korean_char(next)
                        && !next.is_whitespace())
                        || (!Self::is_korean_char(curr)
                            && Self::is_korean_char(next)
                            && !curr.is_whitespace())
                    {
                        format!("{} ", curr)
                    } else {
                        curr.to_string()
                    }
                })
                .collect::<String>();

            // Add the last character
            if let Some(last) = text.chars().last() {
                processed.push(last);
            }
        }

        processed
    }

    /// Tokenize using syllable-based approach
    fn tokenize_syllables(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                continue;
            }

            if Self::is_korean_punctuation(ch) || ch.is_ascii_punctuation() {
                if self.config.keep_punctuation {
                    tokens.push(ch.to_string());
                }
                continue;
            }

            if self.config.decompose_hangul && Self::is_hangul_syllable(ch) {
                if let Some((initial, medial, final_)) = Self::decompose_hangul(ch) {
                    tokens.push(initial.to_string());
                    tokens.push(medial.to_string());
                    if let Some(final_char) = final_ {
                        tokens.push(final_char.to_string());
                    }
                }
            } else {
                tokens.push(ch.to_string());
            }
        }

        tokens
    }

    /// Tokenize using jamo-based approach
    fn tokenize_jamos(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                continue;
            }

            if Self::is_korean_punctuation(ch) || ch.is_ascii_punctuation() {
                if self.config.keep_punctuation {
                    tokens.push(ch.to_string());
                }
                continue;
            }

            if Self::is_hangul_syllable(ch) {
                if let Some((initial, medial, final_)) = Self::decompose_hangul(ch) {
                    tokens.push(initial.to_string());
                    tokens.push(medial.to_string());
                    if let Some(final_char) = final_ {
                        tokens.push(final_char.to_string());
                    }
                }
            } else {
                tokens.push(ch.to_string());
            }
        }

        tokens
    }

    /// Tokenize using word-based approach with Korean morphology
    fn tokenize_words(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();
        let mut current_type = None;

        for ch in text.chars() {
            let char_type = if Self::is_hangul_syllable(ch) {
                "hangul"
            } else if Self::is_hangul_jamo(ch) {
                "jamo"
            } else if Self::is_hanja(ch) {
                "hanja"
            } else if ch.is_ascii_alphanumeric() {
                "ascii"
            } else if Self::is_korean_punctuation(ch) || ch.is_ascii_punctuation() {
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
                        || (char_type == "hangul"
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

        let tokens = match self.config.mode {
            KoreanMode::Syllable => self.tokenize_syllables(&processed_text),
            KoreanMode::Jamo => self.tokenize_jamos(&processed_text),
            KoreanMode::Word => self.tokenize_words(&processed_text),
        };

        Ok(tokens)
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

    /// Check if token is a Korean particle
    pub fn is_particle(&self, token: &str) -> bool {
        self.particles.contains(token)
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

impl Tokenizer for KoreanTokenizer {
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
                result.push_str(&token);
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
        self.vocab.get_token(id).map(|s| s.to_string())
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
            "한",
            "국",
            "어",
            "나",
            "는",
            "다",
            "요",
            "안녕하세요",
            "감사합니다",
            "죄송합니다",
            "네",
            "아니요",
            "오늘",
            "내일",
            "어제",
            "시간",
            "장소",
            "사람",
            "ㅏ",
            "ㅑ",
            "ㅓ",
            "ㅕ",
            "ㅗ",
            "ㅛ",
            "ㅜ",
            "ㅠ",
            "ㅡ",
            "ㅣ",
            "ㄱ",
            "ㄴ",
            "ㄷ",
            "ㄹ",
            "ㅁ",
            "ㅂ",
            "ㅅ",
            "ㅇ",
            "ㅈ",
            "ㅊ",
            "ㅋ",
            "ㅌ",
            "ㅍ",
            "ㅎ",
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
        assert!(KoreanTokenizer::is_hangul_syllable('한'));
        assert!(KoreanTokenizer::is_hangul_syllable('국'));
        assert!(KoreanTokenizer::is_hangul_jamo('ㅏ'));
        assert!(KoreanTokenizer::is_hangul_jamo('ㄱ'));
        assert!(KoreanTokenizer::is_hanja('中'));
        assert!(KoreanTokenizer::is_hanja('國'));
        assert!(!KoreanTokenizer::is_korean_char('a'));
        assert!(!KoreanTokenizer::is_korean_char('1'));
    }

    #[test]
    fn test_korean_punctuation_detection() {
        assert!(KoreanTokenizer::is_korean_punctuation('。'));
        assert!(KoreanTokenizer::is_korean_punctuation('、'));
        assert!(KoreanTokenizer::is_korean_punctuation('？'));
        assert!(KoreanTokenizer::is_korean_punctuation('！'));
        assert!(!KoreanTokenizer::is_korean_punctuation('.'));
        assert!(!KoreanTokenizer::is_korean_punctuation(','));
    }

    #[test]
    fn test_hangul_decomposition() {
        let result = KoreanTokenizer::decompose_hangul('한');
        assert!(result.is_some());
        let (initial, medial, final_) = result.unwrap();
        assert_eq!(initial, 'ㅎ');
        assert_eq!(medial, 'ㅏ');
        assert_eq!(final_, Some('ㄴ'));
    }

    #[test]
    fn test_hangul_composition() {
        // Test composition with various jamo combinations

        // Test basic composition: ㅎ(ᄒ) + ㅏ(ᅡ) + ㄴ(ᆫ) = 한
        let result = KoreanTokenizer::compose_hangul('ᄒ', 'ᅡ', Some('ᆫ'));
        assert_eq!(result, Some('한'));

        // Test composition without final consonant: ㄱ(ᄀ) + ㅏ(ᅡ) = 가
        let result = KoreanTokenizer::compose_hangul('ᄀ', 'ᅡ', None);
        assert_eq!(result, Some('가'));

        // Test another combination: ㄴ(ᄂ) + ㅜ(ᅮ) + ㄴ(ᆫ) = 눈
        let result = KoreanTokenizer::compose_hangul('ᄂ', 'ᅮ', Some('ᆫ'));
        assert_eq!(result, Some('눈'));

        // Test invalid jamo (should return None)
        let result = KoreanTokenizer::compose_hangul('a', 'ᅡ', None);
        assert_eq!(result, None);
    }

    #[test]
    fn test_syllable_tokenization() {
        let config = KoreanTokenizerConfig {
            mode: KoreanMode::Syllable,
            ..Default::default()
        };
        let vocab = create_test_vocab();
        let tokenizer = KoreanTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.tokenize_text("안녕하세요").unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result, vec!["안", "녕", "하", "세", "요"]);
    }

    #[test]
    fn test_jamo_tokenization() {
        let config = KoreanTokenizerConfig {
            mode: KoreanMode::Jamo,
            ..Default::default()
        };
        let vocab = create_test_vocab();
        let tokenizer = KoreanTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.tokenize_text("한").unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result, vec!["ㅎ", "ㅏ", "ㄴ"]);
    }

    #[test]
    fn test_word_tokenization() {
        let config = KoreanTokenizerConfig {
            mode: KoreanMode::Word,
            ..Default::default()
        };
        let vocab = create_test_vocab();
        let tokenizer = KoreanTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.tokenize_text("안녕하세요 한국어").unwrap();
        assert!(result.len() > 0);
        // Should keep multi-character words together
        assert!(result.contains(&"안녕하세요".to_string()) || result.len() > 1);
    }

    #[test]
    fn test_tokenization_encode_decode() {
        let config = KoreanTokenizerConfig {
            mode: KoreanMode::Syllable,
            ..Default::default()
        };
        let vocab = create_test_vocab();
        let tokenizer = KoreanTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.encode("한국어").unwrap();
        assert!(!result.input_ids.is_empty());
        assert_eq!(result.input_ids.len(), result.attention_mask.len());

        let decoded = tokenizer.decode(&result.input_ids).unwrap();
        assert_eq!(decoded, "한국어");
    }

    #[test]
    fn test_pair_encoding() {
        let config = KoreanTokenizerConfig {
            mode: KoreanMode::Syllable,
            ..Default::default()
        };
        let vocab = create_test_vocab();
        let tokenizer = KoreanTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.encode_pair("안녕하세요", "한국어").unwrap();
        assert!(!result.input_ids.is_empty());
        assert!(result.token_type_ids.is_some());

        let token_type_ids = result.token_type_ids.unwrap();
        assert!(token_type_ids.contains(&0)); // First sequence
        assert!(token_type_ids.contains(&1)); // Second sequence
    }

    #[test]
    fn test_dictionary_management() {
        let config = KoreanTokenizerConfig::default();
        let vocab = create_test_vocab();
        let mut tokenizer = KoreanTokenizer::new(config, vocab).unwrap();

        let initial_size = tokenizer.dictionary_size();

        tokenizer.add_word("테스트".to_string());
        assert_eq!(tokenizer.dictionary_size(), initial_size + 1);
        assert!(tokenizer.contains_word("테스트"));

        assert!(tokenizer.remove_word("테스트"));
        assert_eq!(tokenizer.dictionary_size(), initial_size);
        assert!(!tokenizer.contains_word("테스트"));
    }

    #[test]
    fn test_particle_detection() {
        let config = KoreanTokenizerConfig::default();
        let vocab = create_test_vocab();
        let tokenizer = KoreanTokenizer::new(config, vocab).unwrap();

        assert!(tokenizer.is_particle("이"));
        assert!(tokenizer.is_particle("가"));
        assert!(tokenizer.is_particle("을"));
        assert!(tokenizer.is_particle("를"));
        assert!(!tokenizer.is_particle("한국"));
    }

    #[test]
    fn test_hangul_decomposition_mode() {
        let config = KoreanTokenizerConfig {
            mode: KoreanMode::Syllable,
            decompose_hangul: true,
            ..Default::default()
        };
        let vocab = create_test_vocab();
        let tokenizer = KoreanTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.tokenize_text("한").unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result, vec!["ㅎ", "ㅏ", "ㄴ"]);
    }
}
