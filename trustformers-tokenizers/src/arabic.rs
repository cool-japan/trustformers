//! Arabic tokenization support for TrustformeRS
//!
//! This module provides comprehensive Arabic text tokenization with support for:
//! - Arabic script normalization
//! - Diacritical marks (tashkeel) handling
//! - Root-based morphological analysis
//! - Contextual letter forms
//! - Arabic-specific preprocessing
//!
//! # Features
//!
//! - **Script normalization**: Handles different Arabic Unicode forms
//! - **Diacritical marks**: Optional removal or preservation of tashkeel
//! - **Letter normalization**: Converts contextual forms to base forms
//! - **Word segmentation**: Basic Arabic word boundary detection
//! - **Morphological analysis**: Root extraction and pattern matching
//! - **RTL support**: Right-to-left text handling
//! - **Dialect support**: Support for different Arabic dialects
//!
//! # Example
//!
//! ```rust,no_run
//! use trustformers_tokenizers::arabic::{ArabicTokenizer, ArabicTokenizerConfig};
//! use trustformers_tokenizers::vocab::Vocab;
//! use trustformers_core::traits::Tokenizer;
//! use std::collections::HashMap;
//!
//! let config = ArabicTokenizerConfig::default();
//! let mut vocab_map = HashMap::new();
//! vocab_map.insert("مرحبا".to_string(), 1);
//! vocab_map.insert("بكم".to_string(), 2);
//! let vocab = Vocab::from_map(vocab_map);
//! let tokenizer = ArabicTokenizer::new(config, vocab);
//!
//! let text = "مرحبا بكم في عالم الذكاء الاصطناعي";
//! let result = tokenizer.encode(text).unwrap();
//! ```

use crate::vocab::Vocab;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Configuration for Arabic tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArabicTokenizerConfig {
    /// Whether to remove diacritical marks (tashkeel)
    pub remove_diacritics: bool,
    /// Whether to normalize different Arabic letter forms
    pub normalize_letters: bool,
    /// Whether to normalize Arabic numerals to ASCII
    pub normalize_numbers: bool,
    /// Whether to keep punctuation as separate tokens
    pub keep_punctuation: bool,
    /// Whether to handle different Arabic dialects
    pub handle_dialects: bool,
    /// Whether to perform basic morphological analysis
    pub morphological_analysis: bool,
    /// Maximum word length for processing
    pub max_word_length: usize,
    /// Arabic tokenization mode
    pub mode: ArabicMode,
    /// Unknown token
    pub unk_token: String,
    /// Padding token
    pub pad_token: String,
    /// Special tokens
    pub special_tokens: Vec<String>,
}

/// Arabic tokenization modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArabicMode {
    /// Character-level tokenization
    Character,
    /// Word-level tokenization
    Word,
    /// Morphological tokenization (root + patterns)
    Morphological,
    /// Subword tokenization (BPE-style)
    Subword,
}

impl Default for ArabicTokenizerConfig {
    fn default() -> Self {
        Self {
            remove_diacritics: true,
            normalize_letters: true,
            normalize_numbers: true,
            keep_punctuation: true,
            handle_dialects: true,
            morphological_analysis: false,
            max_word_length: 20,
            mode: ArabicMode::Word,
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

/// Arabic tokenizer with comprehensive Arabic text processing
#[derive(Debug, Clone)]
pub struct ArabicTokenizer {
    config: ArabicTokenizerConfig,
    vocab: Vocab,
    /// Common Arabic words dictionary
    word_dict: HashSet<String>,
    /// Arabic roots dictionary for morphological analysis
    roots_dict: HashSet<String>,
    /// Letter normalization map
    normalization_map: HashMap<char, char>,
    /// Diacritical marks set
    diacritics: HashSet<char>,
    /// Arabic letters set
    arabic_letters: HashSet<char>,
    /// Arabic punctuation set
    arabic_punctuation: HashSet<char>,
    /// Common Arabic prefixes and suffixes
    prefixes: Vec<String>,
    suffixes: Vec<String>,
}

impl ArabicTokenizer {
    /// Create a new Arabic tokenizer
    pub fn new(config: ArabicTokenizerConfig, vocab: Vocab) -> Self {
        let mut tokenizer = Self {
            config,
            vocab,
            word_dict: HashSet::new(),
            roots_dict: HashSet::new(),
            normalization_map: HashMap::new(),
            diacritics: HashSet::new(),
            arabic_letters: HashSet::new(),
            arabic_punctuation: HashSet::new(),
            prefixes: Vec::new(),
            suffixes: Vec::new(),
        };

        tokenizer.init_arabic_data();
        tokenizer.init_normalization_map();
        tokenizer.init_diacritics();
        tokenizer.init_common_words();
        tokenizer.init_morphological_data();
        tokenizer
    }

    /// Initialize Arabic character sets and linguistic data
    fn init_arabic_data(&mut self) {
        // Arabic letters (U+0600 to U+06FF)
        for c in '\u{0600}'..='\u{06FF}' {
            if c.is_alphabetic() {
                self.arabic_letters.insert(c);
            }
        }

        // Arabic punctuation
        let arabic_punct = [
            '،', '؍', '؎', '؏', 'ؐ', 'ؑ', 'ؒ', 'ؓ', 'ؔ', '؛', '؞', '؟', '٪', '٫', '٬', '٭', 'ٰ', 'ٱ',
            'ٲ', 'ٳ', 'ٴ', 'ٵ', 'ٶ', 'ٷ', 'ٸ', 'ٹ', 'ٺ', 'ٻ', 'ټ', 'ٽ', 'پ', 'ٿ', '۔', '۩', '۽',
            '۾',
        ];

        for &punct in &arabic_punct {
            self.arabic_punctuation.insert(punct);
        }
    }

    /// Initialize letter normalization map
    fn init_normalization_map(&mut self) {
        // Common Arabic letter normalizations
        let normalizations = [
            // Alef variations
            ('آ', 'ا'),
            ('أ', 'ا'),
            ('إ', 'ا'),
            ('ٱ', 'ا'),
            // Teh variations
            ('ة', 'ت'),
            ('ۃ', 'ت'),
            // Yeh variations
            ('ي', 'ى'),
            ('ئ', 'ى'),
            ('ې', 'ى'),
            // Waw variations
            ('ؤ', 'و'),
            // Heh variations
            ('ھ', 'ه'),
            ('ۂ', 'ه'),
            // Kaf variations
            ('ك', 'ک'),
            ('ڪ', 'ک'),
            // Gaf variations (Persian/Urdu)
            ('گ', 'ك'),
            ('ڬ', 'ك'),
            // Other common normalizations
            ('ڊ', 'د'),
            ('ڌ', 'د'),
            ('ڈ', 'د'),
            ('ڑ', 'ر'),
            ('ړ', 'ر'),
            ('ڕ', 'ر'),
            ('ښ', 'ش'),
            ('ڜ', 'ش'),
            ('ݩ', 'ف'),
            ('ڤ', 'ف'),
            ('ڙ', 'ز'),
            ('ږ', 'ز'),
        ];

        for (from, to) in normalizations {
            self.normalization_map.insert(from, to);
        }
    }

    /// Initialize diacritical marks (tashkeel)
    fn init_diacritics(&mut self) {
        let diacritics = [
            // Short vowels
            '\u{064B}', // Fathatan
            '\u{064C}', // Dammatan
            '\u{064D}', // Kasratan
            '\u{064E}', // Fatha
            '\u{064F}', // Damma
            '\u{0650}', // Kasra
            '\u{0651}', // Shadda
            '\u{0652}', // Sukun
            '\u{0653}', // Maddah
            '\u{0654}', // Hamza above
            '\u{0655}', // Hamza below
            '\u{0656}', // Subscript alef
            '\u{0657}', // Inverted damma
            '\u{0658}', // Mark noon ghunna
            '\u{0659}', // Zwarakay
            '\u{065A}', // Vowel sign small v
            '\u{065B}', // Vowel sign inverted small v
            '\u{065C}', // Vowel sign dot below
            '\u{065D}', // Reversed damma
            '\u{065E}', // Fatha with two dots
            '\u{065F}', // Wavy hamza below
            '\u{0660}', // Arabic-indic digit zero
            '\u{0670}', // Superscript alef
        ];

        for &diacritic in &diacritics {
            self.diacritics.insert(diacritic);
        }
    }

    /// Initialize common Arabic words
    fn init_common_words(&mut self) {
        let common_words = [
            // Pronouns
            "أنا",
            "أنت",
            "أنتم",
            "أنتن",
            "هو",
            "هي",
            "هم",
            "هن",
            "نحن",
            "إياي",
            "إياك",
            "إياه",
            "إياها",
            "إيانا",
            "إياكم",
            "إياكن",
            "إياهم",
            "إياهن",
            // Articles and particles
            "في",
            "من",
            "إلى",
            "على",
            "عن",
            "مع",
            "بعد",
            "قبل",
            "تحت",
            "فوق",
            "بين",
            "خلال",
            "ضد",
            "نحو",
            "حول",
            "دون",
            "سوى",
            "غير",
            "إلا",
            // Conjunctions
            "و",
            "أو",
            "لكن",
            "لكن",
            "إذا",
            "إذ",
            "حيث",
            "بينما",
            "كما",
            "مثل",
            // Common verbs
            "كان",
            "يكون",
            "أكون",
            "تكون",
            "نكون",
            "يكونون",
            "تكن",
            "يكن",
            "قال",
            "يقول",
            "أقول",
            "تقول",
            "نقول",
            "يقولون",
            "تقل",
            "يقلن",
            "فعل",
            "يفعل",
            "أفعل",
            "تفعل",
            "نفعل",
            "يفعلون",
            "تفعلين",
            "يفعلن",
            // Common nouns
            "بيت",
            "مدرسة",
            "جامعة",
            "مكتب",
            "مستشفى",
            "مطار",
            "محطة",
            "سوق",
            "رجل",
            "امرأة",
            "ولد",
            "بنت",
            "أب",
            "أم",
            "أخ",
            "أخت",
            "زوج",
            "زوجة",
            "صديق",
            "صديقة",
            "معلم",
            "معلمة",
            "طالب",
            "طالبة",
            "طبيب",
            "طبيبة",
            // Time expressions
            "اليوم",
            "غدا",
            "أمس",
            "الآن",
            "قبل",
            "بعد",
            "صباح",
            "مساء",
            "ليل",
            "سنة",
            "شهر",
            "أسبوع",
            "يوم",
            "ساعة",
            "دقيقة",
            "ثانية",
            // Numbers
            "واحد",
            "اثنان",
            "ثلاثة",
            "أربعة",
            "خمسة",
            "ستة",
            "سبعة",
            "ثمانية",
            "تسعة",
            "عشرة",
            "مائة",
            "ألف",
            "مليون",
            "مليار",
            // Adjectives
            "كبير",
            "صغير",
            "طويل",
            "قصير",
            "جميل",
            "قبيح",
            "جيد",
            "سيء",
            "سريع",
            "بطيء",
            "ساخن",
            "بارد",
            "جديد",
            "قديم",
            "صعب",
            "سهل",
            "مهم",
            "عادي",
            // Colors
            "أبيض",
            "أسود",
            "أحمر",
            "أزرق",
            "أخضر",
            "أصفر",
            "بني",
            "رمادي",
            "وردي",
            "برتقالي",
            // Question words
            "ما",
            "ماذا",
            "من",
            "متى",
            "أين",
            "كيف",
            "لماذا",
            "كم",
            "أي",
            "هل",
        ];

        for word in common_words {
            self.word_dict.insert(word.to_string());
        }
    }

    /// Initialize morphological data (prefixes, suffixes, roots)
    fn init_morphological_data(&mut self) {
        // Common Arabic prefixes
        self.prefixes = vec![
            "ال".to_string(),  // Definite article
            "و".to_string(),   // And
            "ف".to_string(),   // Then/so
            "ب".to_string(),   // In/with
            "ك".to_string(),   // Like/as
            "ل".to_string(),   // For/to
            "من".to_string(),  // From
            "في".to_string(),  // In
            "على".to_string(), // On
            "إلى".to_string(), // To
            "عن".to_string(),  // About
            "مع".to_string(),  // With
            "بال".to_string(), // With the
            "وال".to_string(), // And the
            "فال".to_string(), // Then the
            "كال".to_string(), // Like the
            "لل".to_string(),  // For the
        ];

        // Common Arabic suffixes
        self.suffixes = vec![
            "ة".to_string(),  // Teh marbuta (feminine)
            "ان".to_string(), // Dual masculine
            "ين".to_string(), // Dual feminine / plural masculine
            "ات".to_string(), // Plural feminine
            "ون".to_string(), // Plural masculine
            "ها".to_string(), // Her/its
            "هم".to_string(), // Their (masculine)
            "هن".to_string(), // Their (feminine)
            "ني".to_string(), // Me
            "ك".to_string(),  // You
            "كم".to_string(), // You (plural)
            "كن".to_string(), // You (feminine plural)
            "ية".to_string(), // Adjective suffix
            "ي".to_string(),  // My
            "نا".to_string(), // Us/our
        ];

        // Common Arabic roots (trilateral)
        let roots = [
            "كتب",
            "قرأ",
            "علم",
            "عمل",
            "ذهب",
            "جاء",
            "أكل",
            "شرب",
            "نام",
            "قام",
            "جلس",
            "وقف",
            "مشى",
            "ركض",
            "لعب",
            "درس",
            "فهم",
            "حفظ",
            "نسي",
            "تذكر",
            "حب",
            "كره",
            "خاف",
            "فرح",
            "حزن",
            "غضب",
            "ضحك",
            "بكى",
            "نظر",
            "سمع",
            "قال",
            "تكلم",
            "صمت",
            "فتح",
            "أغلق",
            "أخذ",
            "أعطى",
            "اشترى",
            "باع",
            "طبخ",
        ];

        for root in roots {
            self.roots_dict.insert(root.to_string());
        }
    }

    /// Normalize Arabic text
    pub fn normalize_text(&self, text: &str) -> String {
        let mut normalized = String::new();

        for ch in text.chars() {
            // Remove diacritics if configured
            if self.config.remove_diacritics && self.diacritics.contains(&ch) {
                continue;
            }

            // Normalize letters if configured
            if self.config.normalize_letters {
                if let Some(&normalized_ch) = self.normalization_map.get(&ch) {
                    normalized.push(normalized_ch);
                    continue;
                }
            }

            // Normalize Arabic-Indic numerals to ASCII if configured
            if self.config.normalize_numbers {
                match ch {
                    '٠' => normalized.push('0'),
                    '١' => normalized.push('1'),
                    '٢' => normalized.push('2'),
                    '٣' => normalized.push('3'),
                    '٤' => normalized.push('4'),
                    '٥' => normalized.push('5'),
                    '٦' => normalized.push('6'),
                    '٧' => normalized.push('7'),
                    '٨' => normalized.push('8'),
                    '٩' => normalized.push('9'),
                    // Extended Arabic-Indic digits
                    '۰' => normalized.push('0'),
                    '۱' => normalized.push('1'),
                    '۲' => normalized.push('2'),
                    '۳' => normalized.push('3'),
                    '۴' => normalized.push('4'),
                    '۵' => normalized.push('5'),
                    '۶' => normalized.push('6'),
                    '۷' => normalized.push('7'),
                    '۸' => normalized.push('8'),
                    '۹' => normalized.push('9'),
                    _ => normalized.push(ch),
                }
            } else {
                normalized.push(ch);
            }
        }

        normalized
    }

    /// Check if a character is Arabic
    pub fn is_arabic_char(&self, ch: char) -> bool {
        self.arabic_letters.contains(&ch) || self.arabic_punctuation.contains(&ch)
    }

    /// Check if a character is Arabic punctuation
    pub fn is_arabic_punctuation(&self, ch: char) -> bool {
        self.arabic_punctuation.contains(&ch)
    }

    /// Segment Arabic text into words
    pub fn segment_words(&self, text: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current_word = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current_word.is_empty() {
                    words.push(current_word.clone());
                    current_word.clear();
                }
            } else if self.is_arabic_punctuation(ch) && self.config.keep_punctuation {
                if !current_word.is_empty() {
                    words.push(current_word.clone());
                    current_word.clear();
                }
                words.push(ch.to_string());
            } else if self.is_arabic_char(ch) || ch.is_ascii_alphanumeric() {
                current_word.push(ch);
            } else {
                if !current_word.is_empty() {
                    words.push(current_word.clone());
                    current_word.clear();
                }
                if self.config.keep_punctuation {
                    words.push(ch.to_string());
                }
            }
        }

        if !current_word.is_empty() {
            words.push(current_word);
        }

        words
    }

    /// Perform basic morphological analysis
    pub fn analyze_morphology(&self, word: &str) -> MorphologicalAnalysis {
        let mut analysis = MorphologicalAnalysis {
            word: word.to_string(),
            root: None,
            prefix: None,
            suffix: None,
            pattern: None,
            pos_tags: Vec::new(),
        };

        if !self.config.morphological_analysis {
            return analysis;
        }

        let mut remaining = word.to_string();

        // Find prefix
        for prefix in &self.prefixes {
            if remaining.starts_with(prefix) {
                analysis.prefix = Some(prefix.clone());
                remaining = remaining[prefix.len()..].to_string();
                break;
            }
        }

        // Find suffix
        for suffix in &self.suffixes {
            if remaining.ends_with(suffix) {
                analysis.suffix = Some(suffix.clone());
                remaining = remaining[..remaining.len() - suffix.len()].to_string();
                break;
            }
        }

        // Check if remaining part is a known root
        if self.roots_dict.contains(&remaining) {
            analysis.root = Some(remaining);
        }

        // Basic POS tagging (very simplified)
        if analysis.prefix.as_ref().is_some_and(|p| p == "ال") {
            analysis.pos_tags.push("NOUN".to_string());
        }

        if analysis.suffix.as_ref().is_some_and(|s| s == "ة") {
            analysis.pos_tags.push("FEMININE".to_string());
        }

        analysis
    }

    /// Character-level tokenization
    fn tokenize_characters(&self, text: &str) -> Vec<String> {
        let normalized = self.normalize_text(text);
        normalized.chars().map(|c| c.to_string()).collect()
    }

    /// Word-level tokenization
    fn tokenize_words(&self, text: &str) -> Vec<String> {
        let normalized = self.normalize_text(text);
        self.segment_words(&normalized)
    }

    /// Morphological tokenization
    fn tokenize_morphological(&self, text: &str) -> Vec<String> {
        let words = self.tokenize_words(text);
        let mut tokens = Vec::new();

        for word in words {
            let analysis = self.analyze_morphology(&word);

            if let Some(prefix) = analysis.prefix {
                tokens.push(prefix);
            }

            if let Some(root) = analysis.root {
                tokens.push(root);
            } else {
                tokens.push(word);
            }

            if let Some(suffix) = analysis.suffix {
                tokens.push(suffix);
            }
        }

        tokens
    }

    /// Subword tokenization (simplified BPE-style)
    fn tokenize_subword(&self, text: &str) -> Vec<String> {
        let words = self.tokenize_words(text);
        let mut tokens = Vec::new();

        for word in words {
            if word.len() <= 3 {
                tokens.push(word);
            } else {
                // Simple subword splitting - in practice, this would use trained BPE
                let chars: Vec<char> = word.chars().collect();
                let mut i = 0;
                while i < chars.len() {
                    let end = (i + 3).min(chars.len());
                    let subword: String = chars[i..end].iter().collect();
                    tokens.push(subword);
                    i += 2; // Overlap for better coverage
                }
            }
        }

        tokens
    }

    /// Get token statistics
    pub fn get_token_stats(&self, text: &str) -> TokenizationStats {
        let normalized = self.normalize_text(text);
        let words = self.segment_words(&normalized);

        let mut stats = TokenizationStats {
            total_characters: text.len(),
            arabic_characters: 0,
            words: words.len(),
            arabic_words: 0,
            diacritics_removed: 0,
            normalized_letters: 0,
            oov_tokens: 0,
        };

        for ch in text.chars() {
            if self.is_arabic_char(ch) {
                stats.arabic_characters += 1;
            }
            if self.diacritics.contains(&ch) {
                stats.diacritics_removed += 1;
            }
            if self.normalization_map.contains_key(&ch) {
                stats.normalized_letters += 1;
            }
        }

        for word in &words {
            if word.chars().any(|c| self.is_arabic_char(c)) {
                stats.arabic_words += 1;
            }
            if !self.vocab.contains(word) {
                stats.oov_tokens += 1;
            }
        }

        stats
    }
}

/// Morphological analysis result
#[derive(Debug, Clone)]
pub struct MorphologicalAnalysis {
    pub word: String,
    pub root: Option<String>,
    pub prefix: Option<String>,
    pub suffix: Option<String>,
    pub pattern: Option<String>,
    pub pos_tags: Vec<String>,
}

/// Tokenization statistics
#[derive(Debug, Clone)]
pub struct TokenizationStats {
    pub total_characters: usize,
    pub arabic_characters: usize,
    pub words: usize,
    pub arabic_words: usize,
    pub diacritics_removed: usize,
    pub normalized_letters: usize,
    pub oov_tokens: usize,
}

impl ArabicTokenizer {
    /// Private helper method to tokenize text into string tokens
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let tokens = match self.config.mode {
            ArabicMode::Character => self.tokenize_characters(text),
            ArabicMode::Word => self.tokenize_words(text),
            ArabicMode::Morphological => self.tokenize_morphological(text),
            ArabicMode::Subword => self.tokenize_subword(text),
        };

        Ok(tokens)
    }
}

impl Tokenizer for ArabicTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let tokens = self.tokenize(text)?;
        let mut token_ids = Vec::new();

        for token in tokens {
            if let Some(id) = self.vocab.get_id(&token) {
                token_ids.push(id);
            } else {
                // Use UNK token for unknown tokens
                if let Some(unk_id) = self.vocab.get_id(&self.config.unk_token) {
                    token_ids.push(unk_id);
                } else {
                    return Err(TrustformersError::other(format!(
                        "Unknown token '{}' and no UNK token available",
                        token
                    )));
                }
            }
        }

        // Create attention mask (1 for real tokens, 0 for padding)
        let attention_mask = vec![1u8; token_ids.len()];

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
        let mut tokens = Vec::new();

        for &id in token_ids {
            if let Some(token) = self.vocab.get_token(id) {
                tokens.push(token);
            } else {
                return Err(TrustformersError::other(format!(
                    "Invalid token ID: {}",
                    id
                )));
            }
        }

        Ok(tokens.join(" "))
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        let tokens_a = self.tokenize(text_a)?;
        let tokens_b = self.tokenize(text_b)?;

        // Combine tokens with separator
        let mut combined_tokens = tokens_a.clone();
        combined_tokens
            .push(self.config.special_tokens.get(3).cloned().unwrap_or("[SEP]".to_string()));
        combined_tokens.extend(tokens_b.clone());

        let mut token_ids = Vec::new();

        // Encode tokens to IDs
        for token in &combined_tokens {
            if let Some(id) = self.vocab.get_id(token) {
                token_ids.push(id);
            } else {
                // Use UNK token for unknown tokens
                if let Some(unk_id) = self.vocab.get_id(&self.config.unk_token) {
                    token_ids.push(unk_id);
                } else {
                    return Err(TrustformersError::other(format!(
                        "Unknown token '{}' and no UNK token available",
                        token
                    )));
                }
            }
        }

        // Create attention mask (1 for real tokens, 0 for padding)
        let attention_mask = vec![1u8; token_ids.len()];

        // Create token type IDs (0 for first sequence, 1 for second sequence)
        let separator_pos = tokens_a.len();
        let mut token_type_ids = vec![0u32; separator_pos + 1]; // +1 for separator
        token_type_ids.extend(vec![1u32; tokens_b.len()]);

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

    fn get_vocab(&self) -> std::collections::HashMap<String, u32> {
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

    #[test]
    fn test_arabic_tokenizer_creation() {
        let config = ArabicTokenizerConfig::default();
        let vocab = Vocab::new();
        let tokenizer = ArabicTokenizer::new(config, vocab);

        assert!(tokenizer.arabic_letters.len() > 0);
        assert!(tokenizer.word_dict.len() > 0);
    }

    #[test]
    fn test_arabic_text_normalization() {
        let config = ArabicTokenizerConfig::default();
        let vocab = Vocab::new();
        let tokenizer = ArabicTokenizer::new(config, vocab);

        // Test diacritics removal
        let text_with_diacritics = "مَرْحَبًا";
        let normalized = tokenizer.normalize_text(text_with_diacritics);
        assert_eq!(normalized, "مرحبا");

        // Test number normalization
        let text_with_arabic_numbers = "١٢٣٤٥";
        let normalized = tokenizer.normalize_text(text_with_arabic_numbers);
        assert_eq!(normalized, "12345");
    }

    #[test]
    fn test_arabic_character_detection() {
        let config = ArabicTokenizerConfig::default();
        let vocab = Vocab::new();
        let tokenizer = ArabicTokenizer::new(config, vocab);

        assert!(tokenizer.is_arabic_char('م'));
        assert!(tokenizer.is_arabic_char('ر'));
        assert!(tokenizer.is_arabic_char('ح'));
        assert!(!tokenizer.is_arabic_char('a'));
        assert!(!tokenizer.is_arabic_char('1'));
    }

    #[test]
    fn test_word_segmentation() {
        let config = ArabicTokenizerConfig::default();
        let vocab = Vocab::new();
        let tokenizer = ArabicTokenizer::new(config, vocab);

        let text = "مرحبا بكم في عالم الذكاء الاصطناعي";
        let words = tokenizer.segment_words(text);

        assert_eq!(words.len(), 6);
        assert_eq!(words[0], "مرحبا");
        assert_eq!(words[1], "بكم");
        assert_eq!(words[2], "في");
        assert_eq!(words[3], "عالم");
        assert_eq!(words[4], "الذكاء");
        assert_eq!(words[5], "الاصطناعي");
    }

    #[test]
    fn test_morphological_analysis() {
        let config = ArabicTokenizerConfig {
            morphological_analysis: true,
            ..Default::default()
        };
        let vocab = Vocab::new();
        let tokenizer = ArabicTokenizer::new(config, vocab);

        let analysis = tokenizer.analyze_morphology("الكتاب");
        assert_eq!(analysis.word, "الكتاب");
        assert_eq!(analysis.prefix, Some("ال".to_string()));
    }

    #[test]
    fn test_tokenization_modes() -> Result<()> {
        let mut config = ArabicTokenizerConfig::default();
        let vocab = Vocab::new();

        let text = "مرحبا";

        // Test character mode
        config.mode = ArabicMode::Character;
        let tokenizer = ArabicTokenizer::new(config.clone(), vocab.clone());
        let tokens = tokenizer.tokenize(text)?;
        assert_eq!(tokens.len(), 5); // 5 characters

        // Test word mode
        config.mode = ArabicMode::Word;
        let tokenizer = ArabicTokenizer::new(config.clone(), vocab.clone());
        let tokens = tokenizer.tokenize(text)?;
        assert_eq!(tokens.len(), 1); // 1 word

        Ok(())
    }

    #[test]
    fn test_arabic_punctuation() {
        let config = ArabicTokenizerConfig::default();
        let vocab = Vocab::new();
        let tokenizer = ArabicTokenizer::new(config, vocab);

        assert!(tokenizer.is_arabic_punctuation('،'));
        assert!(tokenizer.is_arabic_punctuation('؟'));
        assert!(tokenizer.is_arabic_punctuation('؛'));
        assert!(!tokenizer.is_arabic_punctuation('.'));
        assert!(!tokenizer.is_arabic_punctuation('?'));
    }

    #[test]
    fn test_tokenization_stats() {
        let config = ArabicTokenizerConfig::default();
        let vocab = Vocab::new();
        let tokenizer = ArabicTokenizer::new(config, vocab);

        let text = "مَرْحَبًا بِكُمْ ١٢٣";
        let stats = tokenizer.get_token_stats(text);

        assert!(stats.arabic_characters > 0);
        assert!(stats.diacritics_removed > 0);
        assert_eq!(stats.words, 3);
    }

    #[test]
    fn test_empty_text_handling() -> Result<()> {
        let config = ArabicTokenizerConfig::default();
        let vocab = Vocab::new();
        let tokenizer = ArabicTokenizer::new(config, vocab);

        let tokens = tokenizer.tokenize("")?;
        assert!(tokens.is_empty());

        Ok(())
    }

    #[test]
    fn test_mixed_script_handling() -> Result<()> {
        let config = ArabicTokenizerConfig::default();
        let vocab = Vocab::new();
        let tokenizer = ArabicTokenizer::new(config, vocab);

        let text = "مرحبا Hello العالم World";
        let tokens = tokenizer.tokenize(text)?;

        assert!(tokens.len() > 0);
        assert!(tokens.contains(&"مرحبا".to_string()));
        assert!(tokens.contains(&"Hello".to_string()));
        assert!(tokens.contains(&"العالم".to_string()));
        assert!(tokens.contains(&"World".to_string()));

        Ok(())
    }
}
