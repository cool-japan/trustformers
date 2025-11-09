use crate::vocab::Vocab;
use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegexTokenizerConfig {
    pub case_insensitive: bool,
    pub unicode: bool,
    pub multiline: bool,
    pub dot_matches_new_line: bool,
    pub max_length: Option<usize>,
    pub padding_token: Option<String>,
    pub truncation: bool,
    pub custom_patterns: HashMap<String, String>,
    pub pattern_priorities: HashMap<String, u32>,
    pub fallback_to_char_split: bool,
}

impl Default for RegexTokenizerConfig {
    fn default() -> Self {
        Self {
            case_insensitive: false,
            unicode: true,
            multiline: false,
            dot_matches_new_line: false,
            max_length: None,
            padding_token: None,
            truncation: false,
            custom_patterns: HashMap::new(),
            pattern_priorities: HashMap::new(),
            fallback_to_char_split: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompiledPattern {
    pub name: String,
    pub regex: Regex,
    pub priority: u32,
    pub token_type: TokenType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenType {
    Word,
    Number,
    Punctuation,
    Whitespace,
    Email,
    Url,
    Hashtag,
    Mention,
    Emoji,
    DateTime,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct RegexTokenizer {
    vocab: Vocab,
    patterns: Vec<CompiledPattern>,
    pattern_set: RegexSet,
    config: RegexTokenizerConfig,
}

impl RegexTokenizer {
    pub fn new(vocab: Vocab, config: RegexTokenizerConfig) -> Result<Self> {
        let mut tokenizer = Self {
            vocab,
            patterns: Vec::new(),
            pattern_set: RegexSet::new(&[] as &[&str])
                .map_err(|e| TrustformersError::from(anyhow::anyhow!("Regex error: {}", e)))?,
            config,
        };

        tokenizer.setup_default_patterns()?;
        tokenizer.compile_patterns()?;
        Ok(tokenizer)
    }

    pub fn add_pattern(
        &mut self,
        name: String,
        pattern: String,
        priority: u32,
        token_type: TokenType,
    ) -> Result<()> {
        let flags = self.build_regex_flags();
        let regex = Regex::new(&format!("(?{}){}", flags, pattern))
            .map_err(|e| TrustformersError::from(anyhow::anyhow!("Regex error: {}", e)))?;

        self.patterns.push(CompiledPattern {
            name,
            regex,
            priority,
            token_type,
        });

        self.patterns.sort_by(|a, b| b.priority.cmp(&a.priority));
        self.compile_patterns()?;
        Ok(())
    }

    fn setup_default_patterns(&mut self) -> Result<()> {
        let patterns = vec![
            (
                "email",
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                100,
                TokenType::Email,
            ),
            (
                "url",
                r"https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?",
                95,
                TokenType::Url,
            ),
            ("hashtag", r"#\w+", 90, TokenType::Hashtag),
            ("mention", r"@\w+", 85, TokenType::Mention),
            (
                "emoji",
                r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF\U00002700-\U000027BF]",
                80,
                TokenType::Emoji,
            ),
            (
                "datetime",
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",
                75,
                TokenType::DateTime,
            ),
            (
                "date",
                r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
                70,
                TokenType::DateTime,
            ),
            (
                "time",
                r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?",
                65,
                TokenType::DateTime,
            ),
            (
                "number",
                r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?",
                60,
                TokenType::Number,
            ),
            ("word", r"\b\w+\b", 50, TokenType::Word),
            ("punctuation", r"[^\w\s]", 40, TokenType::Punctuation),
            ("whitespace", r"\s+", 30, TokenType::Whitespace),
        ];

        for (name, pattern, priority, token_type) in patterns {
            let flags = self.build_regex_flags();
            let regex = Regex::new(&format!("(?{}){}", flags, pattern))
                .map_err(|e| TrustformersError::from(anyhow::anyhow!("Regex error: {}", e)))?;

            self.patterns.push(CompiledPattern {
                name: name.to_string(),
                regex,
                priority,
                token_type,
            });
        }

        for (name, pattern) in &self.config.custom_patterns {
            let priority = self.config.pattern_priorities.get(name).copied().unwrap_or(50);
            let flags = self.build_regex_flags();
            let regex = Regex::new(&format!("(?{}){}", flags, pattern))
                .map_err(|e| TrustformersError::from(anyhow::anyhow!("Regex error: {}", e)))?;

            self.patterns.push(CompiledPattern {
                name: name.clone(),
                regex,
                priority,
                token_type: TokenType::Custom(name.clone()),
            });
        }

        self.patterns.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(())
    }

    fn build_regex_flags(&self) -> String {
        let mut flags = String::new();

        if self.config.case_insensitive {
            flags.push('i');
        }
        if self.config.multiline {
            flags.push('m');
        }
        if self.config.dot_matches_new_line {
            flags.push('s');
        }
        if self.config.unicode {
            flags.push('u');
        }

        flags
    }

    fn compile_patterns(&mut self) -> Result<()> {
        let pattern_strings: Vec<String> =
            self.patterns.iter().map(|p| p.regex.as_str().to_string()).collect();

        self.pattern_set = RegexSet::new(&pattern_strings)
            .map_err(|e| TrustformersError::from(anyhow::anyhow!("Regex set error: {}", e)))?;
        Ok(())
    }

    pub fn tokenize_with_metadata(&self, text: &str) -> Result<Vec<TokenWithMetadata>> {
        let normalized_text = text.to_string();

        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < normalized_text.len() {
            let remaining = &normalized_text[pos..];
            let mut matched = false;

            for pattern in self.patterns.iter() {
                if let Some(mat) = pattern.regex.find(remaining) {
                    if mat.start() == 0 {
                        let token_text = mat.as_str();
                        tokens.push(TokenWithMetadata {
                            text: token_text.to_string(),
                            token_type: pattern.token_type.clone(),
                            pattern_name: pattern.name.clone(),
                            start_pos: pos,
                            end_pos: pos + mat.end(),
                        });
                        pos += mat.end();
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                if self.config.fallback_to_char_split {
                    let ch = normalized_text.chars().nth(pos).unwrap_or(' ');
                    tokens.push(TokenWithMetadata {
                        text: ch.to_string(),
                        token_type: TokenType::Custom("char".to_string()),
                        pattern_name: "fallback".to_string(),
                        start_pos: pos,
                        end_pos: pos + ch.len_utf8(),
                    });
                    pos += ch.len_utf8();
                } else {
                    break;
                }
            }
        }

        Ok(tokens)
    }

    pub fn get_pattern_statistics(&self, text: &str) -> Result<PatternStatistics> {
        let tokens = self.tokenize_with_metadata(text)?;
        let mut stats = PatternStatistics::new();

        for token in tokens {
            stats.add_token(&token);
        }

        Ok(stats)
    }

    pub fn analyze_text_patterns(&self, text: &str) -> Result<TextAnalysis> {
        let tokens = self.tokenize_with_metadata(text)?;
        let mut analysis = TextAnalysis::new();

        for token in &tokens {
            analysis.add_token(token);
        }

        analysis.calculate_metrics(text.len());
        Ok(analysis)
    }

    pub fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let tokens = self.tokenize_with_metadata(text)?;
        let mut entities = Vec::new();

        for token in tokens {
            match token.token_type {
                TokenType::Email
                | TokenType::Url
                | TokenType::Hashtag
                | TokenType::Mention
                | TokenType::DateTime => {
                    entities.push(Entity {
                        text: token.text,
                        entity_type: token.token_type,
                        start_pos: token.start_pos,
                        end_pos: token.end_pos,
                        confidence: 1.0,
                    });
                },
                _ => {},
            }
        }

        Ok(entities)
    }
}

impl Tokenizer for RegexTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let tokens = self
            .tokenize_with_metadata(text)
            .map_err(|e| TrustformersError::other(e.to_string()))?;
        let mut token_ids = Vec::new();

        for token_meta in tokens {
            if let Some(id) = self.vocab.get_id(&token_meta.text) {
                token_ids.push(id);
            } else if let Some(unk_id) = self.vocab.get_id("[UNK]") {
                token_ids.push(unk_id);
            }
        }

        if let Some(max_len) = self.config.max_length {
            token_ids.truncate(max_len);
        }

        let attention_mask = vec![1u8; token_ids.len()];
        let token_type_ids = Some(vec![0u32; token_ids.len()]);

        Ok(TokenizedInput {
            input_ids: token_ids,
            attention_mask,
            token_type_ids,
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
            }
        }

        Ok(tokens.join(" "))
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        let result_a = self.encode(text_a)?;
        let result_b = self.encode(text_b)?;

        let mut input_ids = result_a.input_ids;
        let mut attention_mask = result_a.attention_mask;
        let mut token_type_ids =
            result_a.token_type_ids.unwrap_or_else(|| vec![0u32; input_ids.len()]);

        if let Some(sep_id) = self.vocab.get_id("[SEP]") {
            input_ids.push(sep_id);
            attention_mask.push(1);
            token_type_ids.push(0);
        }

        let b_ids_len = result_b.input_ids.len();
        input_ids.extend(result_b.input_ids);
        attention_mask.extend(result_b.attention_mask);

        let segment_b_type_ids = vec![1u32; b_ids_len];
        token_type_ids.extend(segment_b_type_ids);

        Ok(TokenizedInput {
            input_ids,
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

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get_token(id)
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.iter().map(|(token, id)| (token.clone(), *id)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct TokenWithMetadata {
    pub text: String,
    pub token_type: TokenType,
    pub pattern_name: String,
    pub start_pos: usize,
    pub end_pos: usize,
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub text: String,
    pub entity_type: TokenType,
    pub start_pos: usize,
    pub end_pos: usize,
    pub confidence: f32,
}

#[derive(Debug)]
pub struct PatternStatistics {
    pub pattern_counts: HashMap<String, u32>,
    pub token_type_counts: HashMap<String, u32>,
    pub total_tokens: u32,
    pub unique_patterns: u32,
}

impl PatternStatistics {
    fn new() -> Self {
        Self {
            pattern_counts: HashMap::new(),
            token_type_counts: HashMap::new(),
            total_tokens: 0,
            unique_patterns: 0,
        }
    }

    fn add_token(&mut self, token: &TokenWithMetadata) {
        *self.pattern_counts.entry(token.pattern_name.clone()).or_insert(0) += 1;
        let type_name = format!("{:?}", token.token_type);
        *self.token_type_counts.entry(type_name).or_insert(0) += 1;
        self.total_tokens += 1;
        self.unique_patterns = self.pattern_counts.len() as u32;
    }
}

#[derive(Debug)]
pub struct TextAnalysis {
    pub word_count: u32,
    pub number_count: u32,
    pub punctuation_count: u32,
    pub whitespace_count: u32,
    pub special_entity_count: u32,
    pub avg_word_length: f32,
    pub text_length: usize,
    pub token_density: f32,
}

impl TextAnalysis {
    fn new() -> Self {
        Self {
            word_count: 0,
            number_count: 0,
            punctuation_count: 0,
            whitespace_count: 0,
            special_entity_count: 0,
            avg_word_length: 0.0,
            text_length: 0,
            token_density: 0.0,
        }
    }

    fn add_token(&mut self, token: &TokenWithMetadata) {
        match token.token_type {
            TokenType::Word => self.word_count += 1,
            TokenType::Number => self.number_count += 1,
            TokenType::Punctuation => self.punctuation_count += 1,
            TokenType::Whitespace => self.whitespace_count += 1,
            TokenType::Email
            | TokenType::Url
            | TokenType::Hashtag
            | TokenType::Mention
            | TokenType::Emoji
            | TokenType::DateTime => {
                self.special_entity_count += 1;
            },
            _ => {},
        }
    }

    fn calculate_metrics(&mut self, text_length: usize) {
        self.text_length = text_length;

        if self.word_count > 0 {
            self.avg_word_length = text_length as f32 / self.word_count as f32;
        }

        let total_tokens = self.word_count
            + self.number_count
            + self.punctuation_count
            + self.whitespace_count
            + self.special_entity_count;

        if text_length > 0 {
            self.token_density = total_tokens as f32 / text_length as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_vocab() -> Vocab {
        let mut tokens = HashMap::new();
        tokens.insert("hello".to_string(), 1);
        tokens.insert("world".to_string(), 2);
        tokens.insert("test".to_string(), 3);
        tokens.insert("@user".to_string(), 4);
        tokens.insert("#hashtag".to_string(), 5);
        tokens.insert("[UNK]".to_string(), 0);
        tokens.insert("[SEP]".to_string(), 6);
        Vocab::from_map(tokens)
    }

    #[test]
    fn test_regex_tokenizer_creation() {
        let vocab = create_test_vocab();
        let config = RegexTokenizerConfig::default();
        let tokenizer = RegexTokenizer::new(vocab, config).unwrap();

        assert!(tokenizer.patterns.len() > 0);
        assert!(tokenizer.patterns.iter().any(|p| p.name == "word"));
        assert!(tokenizer.patterns.iter().any(|p| p.name == "email"));
    }

    #[test]
    fn test_basic_tokenization() {
        let vocab = create_test_vocab();
        let config = RegexTokenizerConfig::default();
        let tokenizer = RegexTokenizer::new(vocab, config).unwrap();

        let text = "Hello world! Test @user #hashtag";
        let tokens = tokenizer.tokenize_with_metadata(text).unwrap();

        assert!(tokens.len() > 0);
        assert!(tokens.iter().any(|t| matches!(t.token_type, TokenType::Word)));
        assert!(tokens.iter().any(|t| matches!(t.token_type, TokenType::Mention)));
        assert!(tokens.iter().any(|t| matches!(t.token_type, TokenType::Hashtag)));
    }

    #[test]
    fn test_email_detection() {
        let vocab = create_test_vocab();
        let config = RegexTokenizerConfig::default();
        let tokenizer = RegexTokenizer::new(vocab, config).unwrap();

        let text = "Contact me at user@example.com for details";
        let tokens = tokenizer.tokenize_with_metadata(text).unwrap();

        let email_token = tokens.iter().find(|t| matches!(t.token_type, TokenType::Email));
        assert!(email_token.is_some());
        assert_eq!(email_token.unwrap().text, "user@example.com");
    }

    #[test]
    fn test_url_detection() {
        let vocab = create_test_vocab();
        let config = RegexTokenizerConfig::default();
        let tokenizer = RegexTokenizer::new(vocab, config).unwrap();

        let text = "Visit https://example.com for more info";
        let tokens = tokenizer.tokenize_with_metadata(text).unwrap();

        let url_token = tokens.iter().find(|t| matches!(t.token_type, TokenType::Url));
        assert!(url_token.is_some());
        assert_eq!(url_token.unwrap().text, "https://example.com");
    }

    #[test]
    fn test_custom_pattern() {
        let vocab = create_test_vocab();
        let mut config = RegexTokenizerConfig::default();
        config
            .custom_patterns
            .insert("phone".to_string(), r"\d{3}-\d{3}-\d{4}".to_string());
        config.pattern_priorities.insert("phone".to_string(), 95);

        let tokenizer = RegexTokenizer::new(vocab, config).unwrap();

        let text = "Call me at 555-123-4567";
        let tokens = tokenizer.tokenize_with_metadata(text).unwrap();

        let phone_token = tokens.iter().find(|t| t.pattern_name == "phone");
        assert!(phone_token.is_some());
        assert_eq!(phone_token.unwrap().text, "555-123-4567");
    }

    #[test]
    fn test_entity_extraction() {
        let vocab = create_test_vocab();
        let config = RegexTokenizerConfig::default();
        let tokenizer = RegexTokenizer::new(vocab, config).unwrap();

        let text = "Email me at test@example.com or visit https://example.com #testing @user";
        let entities = tokenizer.extract_entities(text).unwrap();

        assert!(entities.len() >= 4);
        assert!(entities.iter().any(|e| matches!(e.entity_type, TokenType::Email)));
        assert!(entities.iter().any(|e| matches!(e.entity_type, TokenType::Url)));
        assert!(entities.iter().any(|e| matches!(e.entity_type, TokenType::Hashtag)));
        assert!(entities.iter().any(|e| matches!(e.entity_type, TokenType::Mention)));
    }

    #[test]
    fn test_pattern_statistics() {
        let vocab = create_test_vocab();
        let config = RegexTokenizerConfig::default();
        let tokenizer = RegexTokenizer::new(vocab, config).unwrap();

        let text = "Hello world! This is a test. @user #hashtag";
        let stats = tokenizer.get_pattern_statistics(text).unwrap();

        assert!(stats.total_tokens > 0);
        assert!(stats.pattern_counts.len() > 0);
        assert!(stats.token_type_counts.len() > 0);
    }

    #[test]
    fn test_text_analysis() {
        let vocab = create_test_vocab();
        let config = RegexTokenizerConfig::default();
        let tokenizer = RegexTokenizer::new(vocab, config).unwrap();

        let text = "Hello world! This has 123 numbers and @mentions #hashtags";
        let analysis = tokenizer.analyze_text_patterns(text).unwrap();

        assert!(analysis.word_count > 0);
        assert!(analysis.number_count > 0);
        assert!(analysis.special_entity_count > 0);
        assert!(analysis.text_length > 0);
    }

    #[test]
    fn test_tokenizer_trait_implementation() {
        let vocab = create_test_vocab();
        let config = RegexTokenizerConfig::default();
        let tokenizer = RegexTokenizer::new(vocab, config).unwrap();

        let text = "hello world";
        let result = tokenizer.encode(text).unwrap();
        assert!(result.input_ids.len() > 0);
        assert_eq!(result.input_ids.len(), result.attention_mask.len());

        let decoded = tokenizer.decode(&result.input_ids).unwrap();
        assert!(!decoded.is_empty());

        assert_eq!(tokenizer.vocab_size(), 7);
        assert!(tokenizer.token_to_id("hello").is_some());
        assert!(tokenizer.id_to_token(1).is_some());
    }

    #[test]
    fn test_case_insensitive_config() {
        let vocab = create_test_vocab();
        let mut config = RegexTokenizerConfig::default();
        config.case_insensitive = true;

        let tokenizer = RegexTokenizer::new(vocab, config).unwrap();

        let text = "HELLO World";
        let tokens = tokenizer.tokenize_with_metadata(text).unwrap();

        assert!(tokens.iter().any(|t| t.text.to_lowercase() == "hello"));
        assert!(tokens.iter().any(|t| t.text.to_lowercase() == "world"));
    }

    #[test]
    fn test_max_length_truncation() {
        let vocab = create_test_vocab();
        let mut config = RegexTokenizerConfig::default();
        config.max_length = Some(3);

        let tokenizer = RegexTokenizer::new(vocab, config).unwrap();

        let text = "hello world test more tokens";
        let result = tokenizer.encode(text).unwrap();

        assert!(result.input_ids.len() <= 3);
        assert_eq!(result.input_ids.len(), result.attention_mask.len());
    }
}
