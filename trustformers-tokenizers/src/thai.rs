use crate::vocab::Vocab;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use trustformers_core::errors::Result;
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Configuration for Thai tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThaiTokenizerConfig {
    /// Tokenization mode: "word", "syllable", or "character"
    pub mode: ThaiMode,
    /// Whether to keep punctuation as separate tokens
    pub keep_punctuation: bool,
    /// Whether to normalize Thai numerals to Arabic numerals
    pub normalize_numerals: bool,
    /// Whether to normalize whitespace
    pub normalize_whitespace: bool,
    /// Whether to handle tone marks separately
    pub handle_tone_marks: bool,
    /// Maximum word length for word segmentation
    pub max_word_length: usize,
    /// Unknown token
    pub unk_token: String,
    /// Padding token
    pub pad_token: String,
    /// Special tokens
    pub special_tokens: Vec<String>,
}

/// Tokenization mode for Thai text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThaiMode {
    /// Word-level tokenization with segmentation
    Word,
    /// Syllable-level tokenization
    Syllable,
    /// Character-level tokenization
    Character,
}

impl Default for ThaiTokenizerConfig {
    fn default() -> Self {
        Self {
            mode: ThaiMode::Word,
            keep_punctuation: true,
            normalize_numerals: true,
            normalize_whitespace: true,
            handle_tone_marks: false,
            max_word_length: 20,
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

/// Thai tokenizer with word segmentation support
#[derive(Debug)]
pub struct ThaiTokenizer {
    config: ThaiTokenizerConfig,
    vocab: Vocab,
    /// Built-in dictionary of common Thai words
    word_dict: HashSet<String>,
    /// Thai syllable patterns
    syllable_patterns: HashMap<String, u32>,
    /// Thai character frequency for segmentation scoring
    char_freq: HashMap<char, u32>,
    /// Whether normalizer is enabled
    use_normalizer: bool,
}

impl ThaiTokenizer {
    /// Create a new Thai tokenizer
    pub fn new(config: ThaiTokenizerConfig, vocab: Vocab) -> Result<Self> {
        let mut tokenizer = Self {
            config,
            vocab,
            word_dict: HashSet::new(),
            syllable_patterns: HashMap::new(),
            char_freq: HashMap::new(),
            use_normalizer: false,
        };

        tokenizer.init_builtin_dict();
        tokenizer.init_syllable_patterns();
        tokenizer.init_char_freq();
        Ok(tokenizer)
    }

    /// Enable basic text normalization
    pub fn with_normalization(mut self) -> Self {
        self.use_normalizer = true;
        self
    }

    /// Initialize built-in Thai word dictionary
    fn init_builtin_dict(&mut self) {
        // Common Thai words
        let common_words = vec![
            // Pronouns
            "ฉัน",
            "เรา",
            "คุณ",
            "เขา",
            "เธอ",
            "มัน",
            "กัน",
            "นั่น",
            "นี่",
            "นั้น",
            // Common verbs
            "เป็น",
            "อยู่",
            "ทำ",
            "ไป",
            "มา",
            "ดู",
            "ฟัง",
            "กิน",
            "ดื่ม",
            "นอน",
            "ตื่น",
            "ใส่",
            "ถอด",
            "ซื้อ",
            "ขาย",
            "ให้",
            "รับ",
            "เอา",
            "ใช้",
            "ช่วย",
            // Common adjectives
            "ดี",
            "เก่ง",
            "สวย",
            "หล่อ",
            "ใหญ่",
            "เล็ก",
            "สูง",
            "เตี้ย",
            "ยาว",
            "สั้น",
            "หนา",
            "บาง",
            "หนัก",
            "เบา",
            "ร้อน",
            "เย็น",
            "อุ่น",
            "เปียก",
            "แห้ง",
            // Common nouns
            "บ้าน",
            "เมือง",
            "ประเทศ",
            "โรงเรียน",
            "มหาวิทยาลัย",
            "งาน",
            "เงิน",
            "เวลา",
            "วัน",
            "คืน",
            "เช้า",
            "เย็น",
            "อาหาร",
            "น้ำ",
            "รถ",
            "เครื่องบิน",
            // Numbers
            "หนึ่ง",
            "สอง",
            "สาม",
            "สี่",
            "ห้า",
            "หก",
            "เจ็ด",
            "แปด",
            "เก้า",
            "สิบ",
            // Particles and connectors
            "ที่",
            "แล้ว",
            "ได้",
            "จะ",
            "ก็",
            "ยัง",
            "อีก",
            "แค่",
            "เพียง",
            "มาก",
            "น้อย",
            "ทั้ง",
            "ทุก",
            "บาง",
            "หลาย",
            "เหมือน",
            "เท่า",
            "กว่า",
            "ต่อ",
            // Prepositions
            "ใน",
            "บน",
            "ล่าง",
            "ข้าง",
            "หน้า",
            "หลัง",
            "ซ้าย",
            "ขวา",
            "กลาง",
            "ระหว่าง",
            "ตาม",
            "ผ่าน",
            "ถึง",
            "จาก",
            "ไปยัง",
            "เพื่อ",
            "เพราะ",
            // Question words
            "อะไร",
            "ใคร",
            "ที่ไหน",
            "เมื่อไร",
            "ทำไม",
            "อย่างไร",
            "เท่าไร",
            // Common phrases
            "สวัสดี",
            "ขอบคุณ",
            "ขอโทษ",
            "ไม่เป็นไร",
            "ลาก่อน",
            "โชคดี",
            "สบายดี",
            "ยินดี",
            "เสียใจ",
            "น่าเสียดาย",
            "ไม่ต้อง",
            "ไม่ใช่",
            "ครับ",
            "ค่ะ",
            "คะ",
            // Time expressions
            "วันนี้",
            "เมื่อวาน",
            "พรุ่งนี้",
            "เดือนนี้",
            "ปีนี้",
            "สัปดาห์",
            "ชั่วโมง",
            "นาที",
            "วินาที",
            "ตอนเช้า",
            "ตอนเย็น",
            "ตอนกลางคืน",
            // Colors
            "สีแดง",
            "สีเหลือง",
            "สีเขียว",
            "สีน้ำเงิน",
            "สีม่วง",
            "สีส้ม",
            "สีชมพู",
            "สีน้ำตาล",
            "สีเทา",
            "สีดำ",
            "สีขาว",
            // Family
            "พ่อ",
            "แม่",
            "ลูก",
            "พี่",
            "น้อง",
            "ปู่",
            "ย่า",
            "ตา",
            "ยาย",
            "ลุง",
            "ป้า",
            "อา",
            "น้า",
            "สามี",
            "ภรรยา",
            "ครอบครัว",
        ];

        for word in common_words {
            self.word_dict.insert(word.to_string());
        }
    }

    /// Initialize Thai syllable patterns
    fn init_syllable_patterns(&mut self) {
        // Basic Thai syllable patterns with frequencies
        let patterns = vec![
            ("ก", 100),
            ("ข", 80),
            ("ค", 90),
            ("ง", 70),
            ("จ", 85),
            ("ฉ", 60),
            ("ช", 75),
            ("ซ", 50),
            ("ด", 95),
            ("ต", 90),
            ("ท", 85),
            ("น", 100),
            ("บ", 85),
            ("ป", 80),
            ("ผ", 70),
            ("ฝ", 40),
            ("พ", 75),
            ("ฟ", 60),
            ("ภ", 65),
            ("ม", 95),
            ("ย", 90),
            ("ร", 100),
            ("ล", 85),
            ("ว", 90),
            ("ศ", 70),
            ("ษ", 65),
            ("ส", 95),
            ("ห", 85),
            ("อ", 100),
            ("ฮ", 50),
            // Common syllables
            ("กา", 80),
            ("กิ", 70),
            ("กุ", 60),
            ("เก", 75),
            ("โก", 65),
            ("คา", 85),
            ("คิ", 60),
            ("คุ", 70),
            ("เค", 65),
            ("โค", 55),
            ("นา", 90),
            ("นิ", 80),
            ("นุ", 60),
            ("เน", 70),
            ("โน", 50),
            ("มา", 95),
            ("มิ", 70),
            ("มุ", 60),
            ("เม", 65),
            ("โม", 55),
            ("รา", 90),
            ("ริ", 80),
            ("รุ", 70),
            ("เร", 85),
            ("โร", 65),
            ("ลา", 85),
            ("ลิ", 70),
            ("ลุ", 60),
            ("เล", 75),
            ("โล", 60),
            ("วา", 90),
            ("วิ", 75),
            ("วุ", 50),
            ("เว", 70),
            ("โว", 55),
            ("สา", 95),
            ("สิ", 80),
            ("สุ", 70),
            ("เส", 75),
            ("โส", 60),
            ("หา", 85),
            ("หิ", 60),
            ("หุ", 50),
            ("เห", 70),
            ("โห", 55),
        ];

        for (pattern, freq) in patterns {
            self.syllable_patterns.insert(pattern.to_string(), freq);
        }
    }

    /// Initialize Thai character frequency
    fn init_char_freq(&mut self) {
        // Thai character frequencies (approximate)
        let char_frequencies = vec![
            ('ก', 1200),
            ('ข', 180),
            ('ค', 580),
            ('ง', 950),
            ('จ', 420),
            ('ฉ', 80),
            ('ช', 320),
            ('ซ', 90),
            ('ด', 650),
            ('ต', 980),
            ('ท', 450),
            ('น', 1800),
            ('บ', 320),
            ('ป', 450),
            ('ผ', 180),
            ('ฝ', 40),
            ('พ', 280),
            ('ฟ', 120),
            ('ภ', 90),
            ('ม', 950),
            ('ย', 720),
            ('ร', 1500),
            ('ล', 850),
            ('ว', 780),
            ('ศ', 120),
            ('ษ', 150),
            ('ส', 1200),
            ('ห', 450),
            ('อ', 1600),
            ('ฮ', 30),
            // Vowels
            ('า', 2000),
            ('ิ', 1500),
            ('ี', 1200),
            ('ึ', 300),
            ('ื', 400),
            ('ุ', 800),
            ('ู', 600),
            ('เ', 1800),
            ('แ', 400),
            ('โ', 300),
            ('ใ', 100),
            ('ไ', 200),
            ('ำ', 800),
            ('ฯ', 50),
            ('ๆ', 150),
            // Tone marks
            ('่', 1000),
            ('้', 800),
            ('๊', 200),
            ('๋', 150),
            // Numbers
            ('๐', 50),
            ('๑', 80),
            ('๒', 60),
            ('๓', 40),
            ('๔', 30),
            ('๕', 25),
            ('๖', 20),
            ('๗', 15),
            ('๘', 10),
            ('๙', 8),
        ];

        for (ch, freq) in char_frequencies {
            self.char_freq.insert(ch, freq);
        }
    }

    /// Check if a character is Thai
    pub fn is_thai_char(ch: char) -> bool {
        matches!(ch, '\u{0E00}'..='\u{0E7F}')
    }

    /// Check if a character is a Thai vowel
    pub fn is_thai_vowel(ch: char) -> bool {
        matches!(ch,
            '\u{0E30}'..='\u{0E3A}' | // Thai vowels
            '\u{0E40}'..='\u{0E44}' | // Thai vowels
            '\u{0E47}'..='\u{0E4E}'   // Thai vowels and tone marks
        )
    }

    /// Check if a character is a Thai tone mark
    pub fn is_thai_tone_mark(ch: char) -> bool {
        matches!(ch, '\u{0E48}'..='\u{0E4B}')
    }

    /// Check if a character is a Thai consonant
    pub fn is_thai_consonant(ch: char) -> bool {
        matches!(ch, '\u{0E01}'..='\u{0E2E}')
    }

    /// Normalize Thai text
    fn normalize_text(&self, text: &str) -> String {
        if !self.use_normalizer {
            return text.to_string();
        }

        let mut result = String::new();

        for ch in text.chars() {
            if self.config.normalize_numerals && Self::is_thai_numeral(ch) {
                result.push(Self::thai_numeral_to_arabic(ch));
            } else if self.config.normalize_whitespace && ch.is_whitespace() {
                result.push(' ');
            } else {
                result.push(ch);
            }
        }

        if self.config.normalize_whitespace {
            result = result.split_whitespace().collect::<Vec<_>>().join(" ");
        }

        result
    }

    /// Check if character is Thai numeral
    fn is_thai_numeral(ch: char) -> bool {
        matches!(ch, '\u{0E50}'..='\u{0E59}')
    }

    /// Convert Thai numeral to Arabic numeral
    fn thai_numeral_to_arabic(ch: char) -> char {
        match ch {
            '\u{0E50}' => '0',
            '\u{0E51}' => '1',
            '\u{0E52}' => '2',
            '\u{0E53}' => '3',
            '\u{0E54}' => '4',
            '\u{0E55}' => '5',
            '\u{0E56}' => '6',
            '\u{0E57}' => '7',
            '\u{0E58}' => '8',
            '\u{0E59}' => '9',
            _ => ch,
        }
    }

    /// Segment Thai text into words using maximum matching algorithm
    fn segment_words(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut words = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            let mut best_match = String::new();
            let mut best_len = 0;

            // Look for the longest matching word
            for len in 1..=std::cmp::min(self.config.max_word_length, chars.len() - i) {
                let candidate: String = chars[i..i + len].iter().collect();

                if self.word_dict.contains(&candidate) && len > best_len {
                    best_match = candidate;
                    best_len = len;
                }
            }

            if best_len > 0 {
                words.push(best_match);
                i += best_len;
            } else {
                // No match found, take single character
                words.push(chars[i].to_string());
                i += 1;
            }
        }

        words
    }

    /// Segment Thai text into syllables
    fn segment_syllables(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut syllables = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            let mut syllable = String::new();

            // Start with a consonant
            if i < chars.len() && Self::is_thai_consonant(chars[i]) {
                syllable.push(chars[i]);
                i += 1;
            }

            // Add vowels and tone marks
            while i < chars.len()
                && (Self::is_thai_vowel(chars[i]) || Self::is_thai_tone_mark(chars[i]))
            {
                syllable.push(chars[i]);
                i += 1;
            }

            // Add final consonant if present
            if i < chars.len()
                && Self::is_thai_consonant(chars[i])
                && (i + 1 >= chars.len() || !Self::is_thai_vowel(chars[i + 1]))
            {
                syllable.push(chars[i]);
                i += 1;
            }

            if syllable.is_empty() && i < chars.len() {
                syllable.push(chars[i]);
                i += 1;
            }

            if !syllable.is_empty() {
                syllables.push(syllable);
            }
        }

        syllables
    }

    /// Tokenize text based on the configured mode
    fn tokenize_text(&self, text: &str) -> Vec<String> {
        let normalized = self.normalize_text(text);
        let mut tokens = Vec::new();
        let mut current_text = String::new();

        for ch in normalized.chars() {
            if self.config.keep_punctuation && ch.is_ascii_punctuation() {
                if !current_text.is_empty() {
                    tokens.extend(self.process_text_segment(&current_text));
                    current_text.clear();
                }
                tokens.push(ch.to_string());
            } else if ch.is_whitespace() {
                if !current_text.is_empty() {
                    tokens.extend(self.process_text_segment(&current_text));
                    current_text.clear();
                }
            } else {
                current_text.push(ch);
            }
        }

        if !current_text.is_empty() {
            tokens.extend(self.process_text_segment(&current_text));
        }

        tokens
    }

    /// Process a text segment based on the tokenization mode
    fn process_text_segment(&self, text: &str) -> Vec<String> {
        match self.config.mode {
            ThaiMode::Word => self.segment_words(text),
            ThaiMode::Syllable => self.segment_syllables(text),
            ThaiMode::Character => text.chars().map(|c| c.to_string()).collect(),
        }
    }
}

impl Tokenizer for ThaiTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let tokens = self.tokenize_text(text);

        let mut token_ids = Vec::new();
        let mut attention_mask = Vec::new();

        // Add CLS token if available
        if let Some(cls_token) = self.config.special_tokens.get(2) {
            token_ids.push(self.vocab.get_id(cls_token).unwrap_or(0));
            attention_mask.push(1);
        }

        for token in tokens {
            token_ids.push(
                self.vocab
                    .get_id(&token)
                    .unwrap_or(self.vocab.get_id(&self.config.unk_token).unwrap_or(0)),
            );
            attention_mask.push(1);
        }

        // Add SEP token if available
        if let Some(sep_token) = self.config.special_tokens.get(3) {
            token_ids.push(self.vocab.get_id(sep_token).unwrap_or(0));
            attention_mask.push(1);
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
                if !self.config.special_tokens.contains(&token) {
                    result.push_str(&token);
                }
            }
        }

        Ok(result)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        let tokens_a = self.tokenize_text(text_a);
        let tokens_b = self.tokenize_text(text_b);

        let mut token_ids = Vec::new();
        let mut attention_mask = Vec::new();
        let mut token_type_ids = Vec::new();

        // Add CLS token if available
        if let Some(cls_token) = self.config.special_tokens.get(2) {
            token_ids.push(self.vocab.get_id(cls_token).unwrap_or(0));
            attention_mask.push(1);
            token_type_ids.push(0);
        }

        for token in tokens_a {
            token_ids.push(
                self.vocab
                    .get_id(&token)
                    .unwrap_or(self.vocab.get_id(&self.config.unk_token).unwrap_or(0)),
            );
            attention_mask.push(1);
            token_type_ids.push(0);
        }

        // Add SEP token if available
        if let Some(sep_token) = self.config.special_tokens.get(3) {
            token_ids.push(self.vocab.get_id(sep_token).unwrap_or(0));
            attention_mask.push(1);
            token_type_ids.push(0);
        }

        for token in tokens_b {
            token_ids.push(
                self.vocab
                    .get_id(&token)
                    .unwrap_or(self.vocab.get_id(&self.config.unk_token).unwrap_or(0)),
            );
            attention_mask.push(1);
            token_type_ids.push(1);
        }

        // Add SEP token if available
        if let Some(sep_token) = self.config.special_tokens.get(3) {
            token_ids.push(self.vocab.get_id(sep_token).unwrap_or(0));
            attention_mask.push(1);
            token_type_ids.push(1);
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

    #[test]
    fn test_thai_char_detection() {
        assert!(ThaiTokenizer::is_thai_char('ก'));
        assert!(ThaiTokenizer::is_thai_char('ข'));
        assert!(ThaiTokenizer::is_thai_char('า'));
        assert!(ThaiTokenizer::is_thai_char('่'));
        assert!(!ThaiTokenizer::is_thai_char('a'));
        assert!(!ThaiTokenizer::is_thai_char('1'));
    }

    #[test]
    fn test_thai_vowel_detection() {
        assert!(ThaiTokenizer::is_thai_vowel('า'));
        assert!(ThaiTokenizer::is_thai_vowel('ิ'));
        assert!(ThaiTokenizer::is_thai_vowel('เ'));
        assert!(ThaiTokenizer::is_thai_vowel('แ'));
        assert!(!ThaiTokenizer::is_thai_vowel('ก'));
        assert!(!ThaiTokenizer::is_thai_vowel('ข'));
    }

    #[test]
    fn test_thai_consonant_detection() {
        assert!(ThaiTokenizer::is_thai_consonant('ก'));
        assert!(ThaiTokenizer::is_thai_consonant('ข'));
        assert!(ThaiTokenizer::is_thai_consonant('ค'));
        assert!(!ThaiTokenizer::is_thai_consonant('า'));
        assert!(!ThaiTokenizer::is_thai_consonant('่'));
    }

    #[test]
    fn test_thai_tone_mark_detection() {
        assert!(ThaiTokenizer::is_thai_tone_mark('่'));
        assert!(ThaiTokenizer::is_thai_tone_mark('้'));
        assert!(ThaiTokenizer::is_thai_tone_mark('๊'));
        assert!(ThaiTokenizer::is_thai_tone_mark('๋'));
        assert!(!ThaiTokenizer::is_thai_tone_mark('ก'));
        assert!(!ThaiTokenizer::is_thai_tone_mark('า'));
    }

    #[test]
    fn test_thai_numeral_conversion() {
        assert_eq!(ThaiTokenizer::thai_numeral_to_arabic('๐'), '0');
        assert_eq!(ThaiTokenizer::thai_numeral_to_arabic('๑'), '1');
        assert_eq!(ThaiTokenizer::thai_numeral_to_arabic('๒'), '2');
        assert_eq!(ThaiTokenizer::thai_numeral_to_arabic('๙'), '9');
        assert_eq!(ThaiTokenizer::thai_numeral_to_arabic('a'), 'a');
    }

    #[test]
    fn test_thai_tokenizer_creation() {
        let config = ThaiTokenizerConfig::default();
        let mut token_to_id = HashMap::new();
        token_to_id.insert("[PAD]".to_string(), 0);
        token_to_id.insert("[UNK]".to_string(), 1);
        token_to_id.insert("[CLS]".to_string(), 2);
        token_to_id.insert("[SEP]".to_string(), 3);
        token_to_id.insert("[MASK]".to_string(), 4);
        token_to_id.insert("สวัสดี".to_string(), 5);
        token_to_id.insert("ครับ".to_string(), 6);

        let vocab = Vocab::from_map(token_to_id);
        let tokenizer = ThaiTokenizer::new(config, vocab).unwrap();

        assert_eq!(tokenizer.vocab_size(), 7);
    }

    #[test]
    fn test_thai_word_segmentation() {
        let config = ThaiTokenizerConfig::default();
        let mut token_to_id = HashMap::new();
        token_to_id.insert("[PAD]".to_string(), 0);
        token_to_id.insert("[UNK]".to_string(), 1);
        token_to_id.insert("[CLS]".to_string(), 2);
        token_to_id.insert("[SEP]".to_string(), 3);
        token_to_id.insert("[MASK]".to_string(), 4);
        token_to_id.insert("สวัสดี".to_string(), 5);
        token_to_id.insert("ครับ".to_string(), 6);

        let vocab = Vocab::from_map(token_to_id);
        let tokenizer = ThaiTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.encode("สวัสดีครับ").unwrap();
        assert_eq!(result.input_ids.len(), 4); // CLS + 2 tokens + SEP
        assert_eq!(result.input_ids[1], 5); // สวัสดี
        assert_eq!(result.input_ids[2], 6); // ครับ
    }

    #[test]
    fn test_thai_character_tokenization() {
        let mut config = ThaiTokenizerConfig::default();
        config.mode = ThaiMode::Character;

        let mut token_to_id = HashMap::new();
        token_to_id.insert("[PAD]".to_string(), 0);
        token_to_id.insert("[UNK]".to_string(), 1);
        token_to_id.insert("ก".to_string(), 2);
        token_to_id.insert("ข".to_string(), 3);
        token_to_id.insert("ค".to_string(), 4);

        let vocab = Vocab::from_map(token_to_id);
        let tokenizer = ThaiTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.encode("กขค").unwrap();
        assert_eq!(result.input_ids.len(), 5); // CLS + 3 tokens + SEP
        assert_eq!(result.input_ids[1], 2); // ก
        assert_eq!(result.input_ids[2], 3); // ข
        assert_eq!(result.input_ids[3], 4); // ค
    }

    #[test]
    fn test_thai_normalization() {
        let config = ThaiTokenizerConfig::default();
        let mut token_to_id = HashMap::new();
        token_to_id.insert("[PAD]".to_string(), 0);
        token_to_id.insert("[UNK]".to_string(), 1);
        token_to_id.insert("1".to_string(), 2);
        token_to_id.insert("2".to_string(), 3);
        token_to_id.insert("3".to_string(), 4);

        let vocab = Vocab::from_map(token_to_id);
        let tokenizer = ThaiTokenizer::new(config, vocab).unwrap().with_normalization();

        let result = tokenizer.encode("๑๒๓").unwrap();
        assert_eq!(result.input_ids.len(), 5); // CLS + 3 tokens + SEP
        assert_eq!(result.input_ids[1], 2); // 1
        assert_eq!(result.input_ids[2], 3); // 2
        assert_eq!(result.input_ids[3], 4); // 3
    }

    #[test]
    fn test_thai_encode_pair() {
        let config = ThaiTokenizerConfig::default();
        let mut token_to_id = HashMap::new();
        token_to_id.insert("[PAD]".to_string(), 0);
        token_to_id.insert("[UNK]".to_string(), 1);
        token_to_id.insert("[CLS]".to_string(), 2);
        token_to_id.insert("[SEP]".to_string(), 3);
        token_to_id.insert("สวัสดี".to_string(), 4);
        token_to_id.insert("ครับ".to_string(), 5);

        let vocab = Vocab::from_map(token_to_id);
        let tokenizer = ThaiTokenizer::new(config, vocab).unwrap();

        let result = tokenizer.encode_pair("สวัสดี", "ครับ").unwrap();
        assert!(result.input_ids.len() >= 4); // CLS + text_a + SEP + text_b + SEP
        assert_eq!(result.input_ids[0], 2); // CLS
        assert_eq!(result.input_ids[1], 4); // สวัสดี
        assert_eq!(result.input_ids[2], 3); // SEP
        assert_eq!(result.input_ids[3], 5); // ครับ
        assert_eq!(result.input_ids[4], 3); // SEP

        // Check token type IDs
        let token_type_ids = result.token_type_ids.unwrap();
        assert_eq!(token_type_ids[0], 0); // CLS
        assert_eq!(token_type_ids[1], 0); // First text
        assert_eq!(token_type_ids[3], 1); // Second text
    }

    #[test]
    fn test_thai_decode() {
        let config = ThaiTokenizerConfig::default();
        let mut token_to_id = HashMap::new();
        token_to_id.insert("[PAD]".to_string(), 0);
        token_to_id.insert("[UNK]".to_string(), 1);
        token_to_id.insert("[CLS]".to_string(), 2);
        token_to_id.insert("[SEP]".to_string(), 3);
        token_to_id.insert("สวัสดี".to_string(), 4);
        token_to_id.insert("ครับ".to_string(), 5);

        let vocab = Vocab::from_map(token_to_id);
        let tokenizer = ThaiTokenizer::new(config, vocab).unwrap();

        let token_ids = vec![4, 5];
        let result = tokenizer.decode(&token_ids).unwrap();
        assert_eq!(result, "สวัสดีครับ");
    }
}
