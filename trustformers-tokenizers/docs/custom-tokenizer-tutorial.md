# Building Custom Tokenizers with TrustformeRS

This comprehensive tutorial will guide you through building custom tokenizers from scratch using TrustformeRS. We'll cover everything from simple rule-based tokenizers to advanced domain-specific tokenizers with training capabilities.

## Table of Contents

1. [Understanding Tokenization Fundamentals](#understanding-tokenization-fundamentals)
2. [Building Your First Custom Tokenizer](#building-your-first-custom-tokenizer)
3. [Advanced Custom Vocabulary](#advanced-custom-vocabulary)
4. [Rule-Based Tokenization](#rule-based-tokenization)
5. [Domain-Specific Tokenizers](#domain-specific-tokenizers)
6. [Training Custom Tokenizers](#training-custom-tokenizers)
7. [Optimizing Performance](#optimizing-performance)
8. [Testing and Validation](#testing-and-validation)
9. [Deployment Considerations](#deployment-considerations)

## Understanding Tokenization Fundamentals

Before building custom tokenizers, let's understand the core concepts and components that make up a tokenizer.

### Core Components

```rust
use trustformers_tokenizers::{Tokenizer, Vocab, TokenizedInput};
use std::collections::HashMap;

// Basic tokenizer trait that all tokenizers implement
pub trait CustomTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput, TokenizerError>;
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError>;
    fn get_vocab(&self) -> &Vocab;
    fn get_vocab_size(&self) -> usize;
}

// Core tokenization components
pub struct TokenizationComponents {
    // Pre-processing: normalize text before tokenization
    pub normalizer: Option<Box<dyn Normalizer>>,
    
    // Pre-tokenization: split text into chunks
    pub pre_tokenizer: Option<Box<dyn PreTokenizer>>,
    
    // Main tokenization: convert chunks to tokens
    pub model: Box<dyn TokenizationModel>,
    
    // Post-processing: add special tokens, create final output
    pub post_processor: Option<Box<dyn PostProcessor>>,
    
    // Vocabulary: mapping between tokens and IDs
    pub vocab: Vocab,
}
```

### Tokenization Pipeline

The tokenization process typically follows these steps:

1. **Normalization**: Clean and standardize the input text
2. **Pre-tokenization**: Split text into smaller units
3. **Model Application**: Apply the core tokenization algorithm
4. **Post-processing**: Add special tokens and finalize output

## Building Your First Custom Tokenizer

Let's start with a simple word-based tokenizer to understand the basics.

### Simple Word Tokenizer

```rust
use trustformers_tokenizers::{Tokenizer, Vocab, TokenizedInput, TokenizerError};
use std::collections::HashMap;
use regex::Regex;

pub struct SimpleWordTokenizer {
    vocab: Vocab,
    word_regex: Regex,
    unk_token: String,
    unk_id: u32,
}

impl SimpleWordTokenizer {
    pub fn new() -> Result<Self, TokenizerError> {
        let mut vocab = Vocab::new();
        
        // Add special tokens
        let unk_token = "[UNK]".to_string();
        let unk_id = vocab.add_token(unk_token.clone(), 0);
        
        let pad_token = "[PAD]".to_string();
        vocab.add_token(pad_token, 1);
        
        let cls_token = "[CLS]".to_string();
        vocab.add_token(cls_token, 2);
        
        let sep_token = "[SEP]".to_string();
        vocab.add_token(sep_token, 3);
        
        // Word boundary regex (simple version)
        let word_regex = Regex::new(r"\b\w+\b")?;
        
        Ok(Self {
            vocab,
            word_regex,
            unk_token,
            unk_id,
        })
    }
    
    pub fn build_vocab_from_texts(&mut self, texts: &[&str]) -> Result<(), TokenizerError> {
        let mut word_freq = HashMap::new();
        
        // Count word frequencies
        for text in texts {
            let words = self.extract_words(text);
            for word in words {
                *word_freq.entry(word.to_lowercase()).or_insert(0) += 1;
            }
        }
        
        // Add words to vocabulary (starting after special tokens)
        let mut token_id = 4;
        for (word, freq) in word_freq.iter() {
            if *freq >= 2 { // Minimum frequency threshold
                self.vocab.add_token(word.clone(), token_id);
                token_id += 1;
            }
        }
        
        println!("Built vocabulary with {} tokens", self.vocab.len());
        Ok(())
    }
    
    fn extract_words(&self, text: &str) -> Vec<&str> {
        self.word_regex.find_iter(text)
            .map(|m| m.as_str())
            .collect()
    }
}

impl Tokenizer for SimpleWordTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput, TokenizerError> {
        let words = self.extract_words(text);
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut attention_mask = Vec::new();
        
        for word in words {
            let normalized_word = word.to_lowercase();
            let token_id = self.vocab.get_id(&normalized_word)
                .unwrap_or(self.unk_id);
            
            ids.push(token_id);
            tokens.push(if token_id == self.unk_id {
                self.unk_token.clone()
            } else {
                normalized_word
            });
            attention_mask.push(1);
        }
        
        Ok(TokenizedInput {
            ids,
            tokens,
            attention_mask,
            ..Default::default()
        })
    }
    
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut tokens = Vec::new();
        
        for &id in ids {
            if let Some(token) = self.vocab.get_token(id) {
                if token != "[PAD]" {
                    tokens.push(token.clone());
                }
            }
        }
        
        Ok(tokens.join(" "))
    }
    
    fn get_vocab(&self) -> &Vocab {
        &self.vocab
    }
    
    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// Example usage
fn example_simple_tokenizer() -> Result<(), Box<dyn std::error::Error>> {
    let mut tokenizer = SimpleWordTokenizer::new()?;
    
    // Training data
    let training_texts = vec![
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog jumps over the lazy fox",
        "The lazy cat sleeps in the sun",
        "Quick movements catch the eye",
    ];
    
    // Build vocabulary from training data
    tokenizer.build_vocab_from_texts(&training_texts)?;
    
    // Test tokenization
    let test_text = "The quick cat jumps";
    let encoded = tokenizer.encode(test_text)?;
    
    println!("Text: {}", test_text);
    println!("Tokens: {:?}", encoded.tokens);
    println!("IDs: {:?}", encoded.ids);
    
    let decoded = tokenizer.decode(&encoded.ids)?;
    println!("Decoded: {}", decoded);
    
    Ok(())
}
```

## Advanced Custom Vocabulary

Let's build a more sophisticated tokenizer with advanced vocabulary management.

### Frequency-Based Vocabulary Builder

```rust
use trustformers_tokenizers::{Vocab, TokenizerError};
use std::collections::{HashMap, BTreeMap};

pub struct VocabularyBuilder {
    word_frequencies: HashMap<String, usize>,
    char_frequencies: HashMap<char, usize>,
    subword_frequencies: HashMap<String, usize>,
    min_frequency: usize,
    max_vocab_size: usize,
    special_tokens: Vec<String>,
}

impl VocabularyBuilder {
    pub fn new() -> Self {
        Self {
            word_frequencies: HashMap::new(),
            char_frequencies: HashMap::new(),
            subword_frequencies: HashMap::new(),
            min_frequency: 2,
            max_vocab_size: 30000,
            special_tokens: vec![
                "[UNK]".to_string(),
                "[PAD]".to_string(), 
                "[CLS]".to_string(),
                "[SEP]".to_string(),
                "[MASK]".to_string(),
            ],
        }
    }
    
    pub fn with_min_frequency(mut self, min_freq: usize) -> Self {
        self.min_frequency = min_freq;
        self
    }
    
    pub fn with_max_vocab_size(mut self, max_size: usize) -> Self {
        self.max_vocab_size = max_size;
        self
    }
    
    pub fn add_special_token(&mut self, token: String) {
        if !self.special_tokens.contains(&token) {
            self.special_tokens.push(token);
        }
    }
    
    pub fn analyze_corpus(&mut self, texts: &[&str]) -> Result<(), TokenizerError> {
        for text in texts {
            self.analyze_text(text)?;
        }
        Ok(())
    }
    
    fn analyze_text(&mut self, text: &str) -> Result<(), TokenizerError> {
        // Normalize text
        let normalized = text.to_lowercase();
        
        // Analyze characters
        for ch in normalized.chars() {
            if ch.is_alphabetic() || ch.is_numeric() {
                *self.char_frequencies.entry(ch).or_insert(0) += 1;
            }
        }
        
        // Analyze words
        let words: Vec<&str> = normalized.split_whitespace().collect();
        for word in words {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !clean_word.is_empty() {
                *self.word_frequencies.entry(clean_word.to_string()).or_insert(0) += 1;
                
                // Analyze subwords (character n-grams)
                self.analyze_subwords(clean_word);
            }
        }
        
        Ok(())
    }
    
    fn analyze_subwords(&mut self, word: &str) {
        // Character bigrams and trigrams
        let chars: Vec<char> = word.chars().collect();
        
        // Bigrams
        for window in chars.windows(2) {
            let bigram: String = window.iter().collect();
            *self.subword_frequencies.entry(bigram).or_insert(0) += 1;
        }
        
        // Trigrams
        for window in chars.windows(3) {
            let trigram: String = window.iter().collect();
            *self.subword_frequencies.entry(trigram).or_insert(0) += 1;
        }
        
        // Prefixes and suffixes
        if word.len() >= 3 {
            for i in 2..=std::cmp::min(6, word.len()) {
                let prefix = &word[..i];
                let suffix = &word[word.len()-i..];
                
                *self.subword_frequencies.entry(format!("###{}", prefix)).or_insert(0) += 1;
                *self.subword_frequencies.entry(format!("{}###", suffix)).or_insert(0) += 1;
            }
        }
    }
    
    pub fn build_vocabulary(&self) -> Result<Vocab, TokenizerError> {
        let mut vocab = Vocab::new();
        let mut token_id = 0;
        
        // Add special tokens first
        for special_token in &self.special_tokens {
            vocab.add_token(special_token.clone(), token_id);
            token_id += 1;
        }
        
        // Create a priority queue of tokens by frequency
        let mut all_tokens = BTreeMap::new();
        
        // Add characters (highest priority for coverage)
        for (ch, freq) in &self.char_frequencies {
            if *freq >= self.min_frequency {
                all_tokens.insert((*freq, format!("char_{}", ch)), ch.to_string());
            }
        }
        
        // Add frequent subwords
        for (subword, freq) in &self.subword_frequencies {
            if *freq >= self.min_frequency && subword.len() >= 2 {
                all_tokens.insert((*freq, format!("sub_{}", subword)), subword.clone());
            }
        }
        
        // Add frequent words
        for (word, freq) in &self.word_frequencies {
            if *freq >= self.min_frequency {
                all_tokens.insert((*freq, format!("word_{}", word)), word.clone());
            }
        }
        
        // Sort by frequency (descending) and add to vocabulary
        let mut sorted_tokens: Vec<_> = all_tokens.into_iter().collect();
        sorted_tokens.sort_by(|a, b| b.0.0.cmp(&a.0.0));
        
        for ((freq, _key), token) in sorted_tokens.into_iter() {
            if vocab.len() >= self.max_vocab_size {
                break;
            }
            
            if !vocab.contains_token(&token) {
                vocab.add_token(token, token_id);
                token_id += 1;
            }
        }
        
        println!("Built vocabulary with {} tokens", vocab.len());
        Ok(vocab)
    }
    
    pub fn get_vocabulary_stats(&self) -> VocabularyStats {
        VocabularyStats {
            total_words: self.word_frequencies.len(),
            total_chars: self.char_frequencies.len(),
            total_subwords: self.subword_frequencies.len(),
            most_frequent_word: self.word_frequencies.iter()
                .max_by_key(|(_, freq)| *freq)
                .map(|(word, freq)| (word.clone(), *freq)),
            vocabulary_richness: self.word_frequencies.len() as f64 / 
                                self.word_frequencies.values().sum::<usize>() as f64,
        }
    }
}

#[derive(Debug)]
pub struct VocabularyStats {
    pub total_words: usize,
    pub total_chars: usize,
    pub total_subwords: usize,
    pub most_frequent_word: Option<(String, usize)>,
    pub vocabulary_richness: f64,
}

// Example usage
fn example_advanced_vocabulary() -> Result<(), Box<dyn std::error::Error>> {
    let mut vocab_builder = VocabularyBuilder::new()
        .with_min_frequency(3)
        .with_max_vocab_size(10000);
    
    // Add domain-specific special tokens
    vocab_builder.add_special_token("[NUM]".to_string());
    vocab_builder.add_special_token("[URL]".to_string());
    vocab_builder.add_special_token("[EMAIL]".to_string());
    
    // Training corpus
    let training_corpus = vec![
        "The quick brown fox jumps over the lazy dog repeatedly",
        "Machine learning algorithms require large datasets for training",
        "Natural language processing involves tokenization and embedding",
        "Transformers have revolutionized the field of NLP completely",
        "Custom tokenizers can be built for specific domains easily",
        "Vocabulary building is crucial for effective tokenization performance",
    ];
    
    // Analyze corpus
    vocab_builder.analyze_corpus(&training_corpus)?;
    
    // Get statistics
    let stats = vocab_builder.get_vocabulary_stats();
    println!("Vocabulary Statistics:");
    println!("  Total unique words: {}", stats.total_words);
    println!("  Total unique characters: {}", stats.total_chars);
    println!("  Total unique subwords: {}", stats.total_subwords);
    println!("  Vocabulary richness: {:.4}", stats.vocabulary_richness);
    
    if let Some((word, freq)) = stats.most_frequent_word {
        println!("  Most frequent word: '{}' ({})", word, freq);
    }
    
    // Build vocabulary
    let vocab = vocab_builder.build_vocabulary()?;
    println!("Final vocabulary size: {}", vocab.len());
    
    Ok(())
}
```

## Rule-Based Tokenization

Now let's create a rule-based tokenizer that can handle complex tokenization patterns.

### Pattern-Based Tokenizer

```rust
use trustformers_tokenizers::{Tokenizer, Vocab, TokenizedInput, TokenizerError};
use regex::Regex;
use std::collections::HashMap;

pub struct PatternTokenizer {
    vocab: Vocab,
    patterns: Vec<TokenizationPattern>,
    fallback_pattern: Regex,
    unk_token_id: u32,
}

#[derive(Debug, Clone)]
pub struct TokenizationPattern {
    pub name: String,
    pub regex: Regex,
    pub replacement: Option<String>,
    pub preserve_original: bool,
    pub priority: usize,
}

impl PatternTokenizer {
    pub fn new() -> Result<Self, TokenizerError> {
        let mut vocab = Vocab::new();
        
        // Add special tokens
        let unk_token_id = vocab.add_token("[UNK]".to_string(), 0);
        vocab.add_token("[PAD]".to_string(), 1);
        vocab.add_token("[CLS]".to_string(), 2);
        vocab.add_token("[SEP]".to_string(), 3);
        
        // Default patterns
        let mut patterns = Vec::new();
        
        // URLs (highest priority)
        patterns.push(TokenizationPattern {
            name: "url".to_string(),
            regex: Regex::new(r"https?://[^\s]+")?,
            replacement: Some("[URL]".to_string()),
            preserve_original: false,
            priority: 100,
        });
        
        // Email addresses
        patterns.push(TokenizationPattern {
            name: "email".to_string(),
            regex: Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")?,
            replacement: Some("[EMAIL]".to_string()),
            preserve_original: false,
            priority: 90,
        });
        
        // Numbers with decimals
        patterns.push(TokenizationPattern {
            name: "decimal".to_string(),
            regex: Regex::new(r"\b\d+\.\d+\b")?,
            replacement: Some("[NUM]".to_string()),
            preserve_original: false,
            priority: 80,
        });
        
        // Integer numbers
        patterns.push(TokenizationPattern {
            name: "integer".to_string(),
            regex: Regex::new(r"\b\d+\b")?,
            replacement: Some("[NUM]".to_string()),
            preserve_original: false,
            priority: 70,
        });
        
        // Contractions (preserve but mark)
        patterns.push(TokenizationPattern {
            name: "contraction".to_string(),
            regex: Regex::new(r"\b\w+'\w+\b")?,
            replacement: None,
            preserve_original: true,
            priority: 60,
        });
        
        // Hashtags
        patterns.push(TokenizationPattern {
            name: "hashtag".to_string(),
            regex: Regex::new(r"#\w+")?,
            replacement: Some("[HASHTAG]".to_string()),
            preserve_original: false,
            priority: 50,
        });
        
        // Mentions
        patterns.push(TokenizationPattern {
            name: "mention".to_string(),
            regex: Regex::new(r"@\w+")?,
            replacement: Some("[MENTION]".to_string()),
            preserve_original: false,
            priority: 40,
        });
        
        // Sort patterns by priority (highest first)
        patterns.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        // Fallback pattern for words
        let fallback_pattern = Regex::new(r"\b\w+\b")?;
        
        Ok(Self {
            vocab,
            patterns,
            fallback_pattern,
            unk_token_id,
        })
    }
    
    pub fn add_pattern(&mut self, pattern: TokenizationPattern) -> Result<(), TokenizerError> {
        self.patterns.push(pattern);
        self.patterns.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(())
    }
    
    pub fn build_vocab_from_texts(&mut self, texts: &[&str]) -> Result<(), TokenizerError> {
        let mut token_freq = HashMap::new();
        
        for text in texts {
            let tokens = self.extract_tokens(text)?;
            for token in tokens {
                *token_freq.entry(token).or_insert(0) += 1;
            }
        }
        
        // Add frequent tokens to vocabulary
        let mut token_id = 4; // Start after special tokens
        for (token, freq) in token_freq.iter() {
            if *freq >= 2 && !self.vocab.contains_token(token) {
                self.vocab.add_token(token.clone(), token_id);
                token_id += 1;
            }
        }
        
        println!("Built vocabulary with {} tokens", self.vocab.len());
        Ok(())
    }
    
    fn extract_tokens(&self, text: &str) -> Result<Vec<String>, TokenizerError> {
        let mut tokens = Vec::new();
        let mut remaining_text = text.to_string();
        let mut processed_ranges = Vec::new();
        
        // Apply patterns in priority order
        for pattern in &self.patterns {
            let mut new_ranges = Vec::new();
            
            for mat in pattern.regex.find_iter(&remaining_text) {
                let start = mat.start();
                let end = mat.end();
                let matched_text = mat.as_str();
                
                // Check if this range overlaps with already processed ranges
                let overlaps = processed_ranges.iter().any(|(ps, pe)| {
                    (start < *pe && end > *ps)
                });
                
                if !overlaps {
                    if let Some(replacement) = &pattern.replacement {
                        tokens.push(replacement.clone());
                    } else if pattern.preserve_original {
                        tokens.push(matched_text.to_string());
                    }
                    
                    new_ranges.push((start, end));
                }
            }
            
            processed_ranges.extend(new_ranges);
        }
        
        // Sort processed ranges
        processed_ranges.sort_by_key(|(start, _)| *start);
        
        // Extract remaining text and apply fallback pattern
        let mut last_end = 0;
        for (start, end) in processed_ranges {
            // Process text before this match
            if start > last_end {
                let unmatched = &remaining_text[last_end..start];
                for word_match in self.fallback_pattern.find_iter(unmatched) {
                    tokens.push(word_match.as_str().to_lowercase());
                }
            }
            last_end = end;
        }
        
        // Process remaining text after last match
        if last_end < remaining_text.len() {
            let unmatched = &remaining_text[last_end..];
            for word_match in self.fallback_pattern.find_iter(unmatched) {
                tokens.push(word_match.as_str().to_lowercase());
            }
        }
        
        Ok(tokens)
    }
}

impl Tokenizer for PatternTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput, TokenizerError> {
        let extracted_tokens = self.extract_tokens(text)?;
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut attention_mask = Vec::new();
        
        for token in extracted_tokens {
            let token_id = self.vocab.get_id(&token).unwrap_or(self.unk_token_id);
            ids.push(token_id);
            tokens.push(token);
            attention_mask.push(1);
        }
        
        Ok(TokenizedInput {
            ids,
            tokens,
            attention_mask,
            ..Default::default()
        })
    }
    
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut tokens = Vec::new();
        
        for &id in ids {
            if let Some(token) = self.vocab.get_token(id) {
                if token != "[PAD]" {
                    tokens.push(token.clone());
                }
            }
        }
        
        Ok(tokens.join(" "))
    }
    
    fn get_vocab(&self) -> &Vocab {
        &self.vocab
    }
    
    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// Example usage
fn example_pattern_tokenizer() -> Result<(), Box<dyn std::error::Error>> {
    let mut tokenizer = PatternTokenizer::new()?;
    
    // Add custom patterns
    tokenizer.add_pattern(TokenizationPattern {
        name: "price".to_string(),
        regex: Regex::new(r"\$\d+(?:\.\d{2})?")?,
        replacement: Some("[PRICE]".to_string()),
        preserve_original: false,
        priority: 85,
    })?;
    
    tokenizer.add_pattern(TokenizationPattern {
        name: "time".to_string(),
        regex: Regex::new(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b")?,
        replacement: Some("[TIME]".to_string()),
        preserve_original: false,
        priority: 75,
    })?;
    
    // Training data
    let training_texts = vec![
        "Check out https://example.com for more info at user@example.com",
        "The price is $99.99 and meeting is at 3:30 PM tomorrow",
        "I can't believe #amazing news! @friend should see this",
        "Numbers like 42 and 3.14159 are important in calculations",
        "Contact me at john.doe@company.org for details about the $150.00 offer",
    ];
    
    // Build vocabulary
    tokenizer.build_vocab_from_texts(&training_texts)?;
    
    // Test tokenization
    let test_text = "Visit https://test.com, email admin@test.com, costs $29.99 at 2:15 PM #sale @everyone";
    let encoded = tokenizer.encode(test_text)?;
    
    println!("Text: {}", test_text);
    println!("Tokens: {:?}", encoded.tokens);
    println!("IDs: {:?}", encoded.ids);
    
    let decoded = tokenizer.decode(&encoded.ids)?;
    println!("Decoded: {}", decoded);
    
    Ok(())
}
```

## Domain-Specific Tokenizers

Let's create a tokenizer specifically designed for medical text, demonstrating how to build domain-specific tokenizers.

### Medical Text Tokenizer

```rust
use trustformers_tokenizers::{Tokenizer, Vocab, TokenizedInput, TokenizerError};
use regex::Regex;
use std::collections::{HashMap, HashSet};

pub struct MedicalTokenizer {
    vocab: Vocab,
    medical_terms: HashSet<String>,
    drug_names: HashSet<String>,
    measurement_regex: Regex,
    dosage_regex: Regex,
    medical_abbreviations: HashMap<String, String>,
    unk_token_id: u32,
}

impl MedicalTokenizer {
    pub fn new() -> Result<Self, TokenizerError> {
        let mut vocab = Vocab::new();
        
        // Add special tokens
        let unk_token_id = vocab.add_token("[UNK]".to_string(), 0);
        vocab.add_token("[PAD]".to_string(), 1);
        vocab.add_token("[CLS]".to_string(), 2);
        vocab.add_token("[SEP]".to_string(), 3);
        
        // Add medical-specific tokens
        vocab.add_token("[DRUG]".to_string(), 4);
        vocab.add_token("[DOSAGE]".to_string(), 5);
        vocab.add_token("[MEASUREMENT]".to_string(), 6);
        vocab.add_token("[CONDITION]".to_string(), 7);
        vocab.add_token("[PROCEDURE]".to_string(), 8);
        
        // Medical terminology
        let medical_terms = Self::load_medical_terms();
        let drug_names = Self::load_drug_names();
        let medical_abbreviations = Self::load_medical_abbreviations();
        
        // Regular expressions for medical patterns
        let measurement_regex = Regex::new(
            r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|mmHg|bpm|°C|°F|cm|mm|inch|feet|lbs|oz)\b"
        )?;
        
        let dosage_regex = Regex::new(
            r"\b\d+(?:\.\d+)?\s*(?:mg|g|ml|cc|units?)\s+(?:po|iv|im|sc|prn|bid|tid|qid|qd|q\d+h)\b"
        )?;
        
        Ok(Self {
            vocab,
            medical_terms,
            drug_names,
            measurement_regex,
            dosage_regex,
            medical_abbreviations,
            unk_token_id,
        })
    }
    
    fn load_medical_terms() -> HashSet<String> {
        // In a real implementation, this would load from a medical dictionary
        vec![
            "myocardial".to_string(),
            "infarction".to_string(),
            "hypertension".to_string(),
            "diabetes".to_string(),
            "pneumonia".to_string(),
            "cardiology".to_string(),
            "neurology".to_string(),
            "gastroenterology".to_string(),
            "endocrinology".to_string(),
            "dermatology".to_string(),
            "oncology".to_string(),
            "radiology".to_string(),
            "pathology".to_string(),
            "anesthesiology".to_string(),
            "emergency".to_string(),
            "intensive".to_string(),
            "surgical".to_string(),
            "medical".to_string(),
            "clinical".to_string(),
            "diagnostic".to_string(),
            "therapeutic".to_string(),
            "prophylactic".to_string(),
            "symptomatic".to_string(),
            "asymptomatic".to_string(),
            "acute".to_string(),
            "chronic".to_string(),
            "benign".to_string(),
            "malignant".to_string(),
            "primary".to_string(),
            "secondary".to_string(),
            "bilateral".to_string(),
            "unilateral".to_string(),
        ].into_iter().collect()
    }
    
    fn load_drug_names() -> HashSet<String> {
        // Common drug names
        vec![
            "acetaminophen".to_string(),
            "ibuprofen".to_string(),
            "aspirin".to_string(),
            "morphine".to_string(),
            "insulin".to_string(),
            "penicillin".to_string(),
            "amoxicillin".to_string(),
            "metformin".to_string(),
            "lisinopril".to_string(),
            "atorvastatin".to_string(),
            "levothyroxine".to_string(),
            "amlodipine".to_string(),
            "metoprolol".to_string(),
            "omeprazole".to_string(),
            "simvastatin".to_string(),
            "losartan".to_string(),
            "azithromycin".to_string(),
            "furosemide".to_string(),
            "prednisone".to_string(),
            "warfarin".to_string(),
        ].into_iter().collect()
    }
    
    fn load_medical_abbreviations() -> HashMap<String, String> {
        // Common medical abbreviations
        vec![
            ("bp".to_string(), "blood pressure".to_string()),
            ("hr".to_string(), "heart rate".to_string()),
            ("rr".to_string(), "respiratory rate".to_string()),
            ("temp".to_string(), "temperature".to_string()),
            ("wbc".to_string(), "white blood cell".to_string()),
            ("rbc".to_string(), "red blood cell".to_string()),
            ("hgb".to_string(), "hemoglobin".to_string()),
            ("hct".to_string(), "hematocrit".to_string()),
            ("bun".to_string(), "blood urea nitrogen".to_string()),
            ("creatinine".to_string(), "creatinine".to_string()),
            ("glucose".to_string(), "glucose".to_string()),
            ("na".to_string(), "sodium".to_string()),
            ("k".to_string(), "potassium".to_string()),
            ("cl".to_string(), "chloride".to_string()),
            ("co2".to_string(), "carbon dioxide".to_string()),
            ("bmi".to_string(), "body mass index".to_string()),
            ("ecg".to_string(), "electrocardiogram".to_string()),
            ("ekg".to_string(), "electrocardiogram".to_string()),
            ("mri".to_string(), "magnetic resonance imaging".to_string()),
            ("ct".to_string(), "computed tomography".to_string()),
            ("xray".to_string(), "x-ray".to_string()),
            ("ultrasound".to_string(), "ultrasound".to_string()),
            ("iv".to_string(), "intravenous".to_string()),
            ("po".to_string(), "by mouth".to_string()),
            ("im".to_string(), "intramuscular".to_string()),
            ("sc".to_string(), "subcutaneous".to_string()),
            ("prn".to_string(), "as needed".to_string()),
            ("bid".to_string(), "twice daily".to_string()),
            ("tid".to_string(), "three times daily".to_string()),
            ("qid".to_string(), "four times daily".to_string()),
        ].into_iter().collect()
    }
    
    pub fn build_medical_vocab(&mut self, medical_texts: &[&str]) -> Result<(), TokenizerError> {
        let mut token_freq = HashMap::new();
        
        for text in medical_texts {
            let processed_text = self.preprocess_medical_text(text);
            let tokens = self.extract_medical_tokens(&processed_text)?;
            
            for token in tokens {
                *token_freq.entry(token).or_insert(0) += 1;
            }
        }
        
        // Add frequent tokens to vocabulary
        let mut token_id = 9; // Start after special tokens
        for (token, freq) in token_freq.iter() {
            if *freq >= 1 && !self.vocab.contains_token(token) {
                self.vocab.add_token(token.clone(), token_id);
                token_id += 1;
            }
        }
        
        println!("Built medical vocabulary with {} tokens", self.vocab.len());
        Ok(())
    }
    
    fn preprocess_medical_text(&self, text: &str) -> String {
        let mut processed = text.to_lowercase();
        
        // Expand medical abbreviations
        for (abbrev, full_form) in &self.medical_abbreviations {
            let pattern = format!(r"\b{}\b", regex::escape(abbrev));
            if let Ok(regex) = Regex::new(&pattern) {
                processed = regex.replace_all(&processed, full_form).to_string();
            }
        }
        
        processed
    }
    
    fn extract_medical_tokens(&self, text: &str) -> Result<Vec<String>, TokenizerError> {
        let mut tokens = Vec::new();
        let mut remaining_text = text.to_string();
        let mut processed_ranges = Vec::new();
        
        // Extract dosages (highest priority)
        for mat in self.dosage_regex.find_iter(&remaining_text) {
            tokens.push("[DOSAGE]".to_string());
            processed_ranges.push((mat.start(), mat.end()));
        }
        
        // Extract measurements
        for mat in self.measurement_regex.find_iter(&remaining_text) {
            let overlaps = processed_ranges.iter().any(|(start, end)| {
                mat.start() < *end && mat.end() > *start
            });
            
            if !overlaps {
                tokens.push("[MEASUREMENT]".to_string());
                processed_ranges.push((mat.start(), mat.end()));
            }
        }
        
        // Extract words and classify them
        let word_regex = Regex::new(r"\b\w+\b")?;
        for word_match in word_regex.find_iter(&remaining_text) {
            let overlaps = processed_ranges.iter().any(|(start, end)| {
                word_match.start() < *end && word_match.end() > *start
            });
            
            if !overlaps {
                let word = word_match.as_str().to_lowercase();
                
                if self.drug_names.contains(&word) {
                    tokens.push("[DRUG]".to_string());
                } else if self.medical_terms.contains(&word) {
                    tokens.push(word);
                } else {
                    // Check if it's a compound medical term
                    if self.is_medical_compound(&word) {
                        tokens.push(word);
                    } else {
                        tokens.push(word);
                    }
                }
            }
        }
        
        Ok(tokens)
    }
    
    fn is_medical_compound(&self, word: &str) -> bool {
        // Check if word contains medical prefixes/suffixes
        let medical_prefixes = vec!["cardio", "neuro", "gastro", "hepato", "nephro", "pneumo", "dermato"];
        let medical_suffixes = vec!["ology", "itis", "osis", "pathy", "emia", "uria", "algia"];
        
        for prefix in &medical_prefixes {
            if word.starts_with(prefix) {
                return true;
            }
        }
        
        for suffix in &medical_suffixes {
            if word.ends_with(suffix) {
                return true;
            }
        }
        
        false
    }
    
    pub fn analyze_medical_text(&self, text: &str) -> Result<MedicalAnalysis, TokenizerError> {
        let processed_text = self.preprocess_medical_text(text);
        let tokens = self.extract_medical_tokens(&processed_text)?;
        
        let drug_count = tokens.iter().filter(|t| *t == "[DRUG]").count();
        let dosage_count = tokens.iter().filter(|t| *t == "[DOSAGE]").count();
        let measurement_count = tokens.iter().filter(|t| *t == "[MEASUREMENT]").count();
        
        let medical_term_count = tokens.iter().filter(|t| {
            self.medical_terms.contains(*t) || self.is_medical_compound(t)
        }).count();
        
        Ok(MedicalAnalysis {
            total_tokens: tokens.len(),
            drug_mentions: drug_count,
            dosage_mentions: dosage_count,
            measurement_mentions: measurement_count,
            medical_term_mentions: medical_term_count,
            medical_density: (drug_count + dosage_count + measurement_count + medical_term_count) as f64 / tokens.len() as f64,
        })
    }
}

#[derive(Debug)]
pub struct MedicalAnalysis {
    pub total_tokens: usize,
    pub drug_mentions: usize,
    pub dosage_mentions: usize,
    pub measurement_mentions: usize,
    pub medical_term_mentions: usize,
    pub medical_density: f64,
}

impl Tokenizer for MedicalTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput, TokenizerError> {
        let processed_text = self.preprocess_medical_text(text);
        let extracted_tokens = self.extract_medical_tokens(&processed_text)?;
        
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut attention_mask = Vec::new();
        
        for token in extracted_tokens {
            let token_id = self.vocab.get_id(&token).unwrap_or(self.unk_token_id);
            ids.push(token_id);
            tokens.push(token);
            attention_mask.push(1);
        }
        
        Ok(TokenizedInput {
            ids,
            tokens,
            attention_mask,
            ..Default::default()
        })
    }
    
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut tokens = Vec::new();
        
        for &id in ids {
            if let Some(token) = self.vocab.get_token(id) {
                if token != "[PAD]" {
                    tokens.push(token.clone());
                }
            }
        }
        
        Ok(tokens.join(" "))
    }
    
    fn get_vocab(&self) -> &Vocab {
        &self.vocab
    }
    
    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// Example usage
fn example_medical_tokenizer() -> Result<(), Box<dyn std::error::Error>> {
    let mut tokenizer = MedicalTokenizer::new()?;
    
    // Medical training texts
    let medical_texts = vec![
        "Patient presents with acute myocardial infarction. BP 180/100 mmHg, HR 110 bpm.",
        "Administered acetaminophen 650mg po q6h prn pain and fever.",
        "Blood glucose level 180 mg/dL. Continue metformin 500mg bid.",
        "Chest X-ray shows bilateral pneumonia. Start azithromycin 500mg daily.",
        "Post-operative patient, stable vitals. Temperature 98.6°F, WBC 12,000.",
        "Cardiology consultation recommended for ECG abnormalities.",
        "Patient has history of diabetes mellitus type 2 and hypertension.",
        "Prescribed lisinopril 10mg daily for blood pressure control.",
    ];
    
    // Build medical vocabulary
    tokenizer.build_medical_vocab(&medical_texts)?;
    
    // Test medical text analysis
    let test_text = "Patient received morphine 2mg IV q4h prn severe pain. BP improved to 140/90 mmHg after lisinopril.";
    
    println!("Medical Text: {}", test_text);
    
    // Analyze medical content
    let analysis = tokenizer.analyze_medical_text(test_text)?;
    println!("Medical Analysis:");
    println!("  Total tokens: {}", analysis.total_tokens);
    println!("  Drug mentions: {}", analysis.drug_mentions);
    println!("  Dosage mentions: {}", analysis.dosage_mentions);
    println!("  Measurement mentions: {}", analysis.measurement_mentions);
    println!("  Medical density: {:.2}%", analysis.medical_density * 100.0);
    
    // Tokenize
    let encoded = tokenizer.encode(test_text)?;
    println!("Tokens: {:?}", encoded.tokens);
    
    let decoded = tokenizer.decode(&encoded.ids)?;
    println!("Decoded: {}", decoded);
    
    Ok(())
}
```

## Training Custom Tokenizers

Now let's implement a trainable BPE tokenizer from scratch to understand the training process.

### Custom BPE Trainer

```rust
use trustformers_tokenizers::{Vocab, TokenizedInput, TokenizerError};
use std::collections::{HashMap, BTreeMap};

pub struct CustomBPETokenizer {
    vocab: Vocab,
    merges: Vec<(String, String)>,
    unk_token_id: u32,
}

pub struct BPETrainer {
    vocab_size: usize,
    min_frequency: usize,
    special_tokens: Vec<String>,
    word_frequencies: HashMap<String, usize>,
    character_frequencies: HashMap<char, usize>,
}

impl BPETrainer {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            min_frequency: 2,
            special_tokens: vec![
                "[UNK]".to_string(),
                "[PAD]".to_string(),
                "[CLS]".to_string(),
                "[SEP]".to_string(),
            ],
            word_frequencies: HashMap::new(),
            character_frequencies: HashMap::new(),
        }
    }
    
    pub fn with_min_frequency(mut self, min_freq: usize) -> Self {
        self.min_frequency = min_freq;
        self
    }
    
    pub fn add_special_token(&mut self, token: String) {
        if !self.special_tokens.contains(&token) {
            self.special_tokens.push(token);
        }
    }
    
    pub fn train(&mut self, texts: &[&str]) -> Result<CustomBPETokenizer, TokenizerError> {
        println!("Starting BPE training...");
        
        // Step 1: Analyze corpus and build initial vocabulary
        self.analyze_corpus(texts)?;
        
        // Step 2: Initialize vocabulary with characters
        let mut vocab = self.initialize_vocabulary()?;
        
        // Step 3: Initialize word representations
        let mut word_splits = self.initialize_word_splits()?;
        
        // Step 4: Learn BPE merges
        let merges = self.learn_bpe_merges(&mut word_splits, &mut vocab)?;
        
        // Step 5: Create final tokenizer
        let unk_token_id = vocab.get_id("[UNK]").unwrap();
        
        println!("BPE training completed!");
        println!("Final vocabulary size: {}", vocab.len());
        println!("Number of merges: {}", merges.len());
        
        Ok(CustomBPETokenizer {
            vocab,
            merges,
            unk_token_id,
        })
    }
    
    fn analyze_corpus(&mut self, texts: &[&str]) -> Result<(), TokenizerError> {
        println!("Analyzing corpus...");
        
        for text in texts {
            let normalized = text.to_lowercase();
            
            // Count character frequencies
            for ch in normalized.chars() {
                if ch.is_alphabetic() || ch.is_numeric() {
                    *self.character_frequencies.entry(ch).or_insert(0) += 1;
                }
            }
            
            // Count word frequencies
            let words: Vec<&str> = normalized.split_whitespace().collect();
            for word in words {
                let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
                if !clean_word.is_empty() {
                    *self.word_frequencies.entry(clean_word.to_string()).or_insert(0) += 1;
                }
            }
        }
        
        println!("Found {} unique words", self.word_frequencies.len());
        println!("Found {} unique characters", self.character_frequencies.len());
        
        Ok(())
    }
    
    fn initialize_vocabulary(&self) -> Result<Vocab, TokenizerError> {
        let mut vocab = Vocab::new();
        let mut token_id = 0;
        
        // Add special tokens
        for special_token in &self.special_tokens {
            vocab.add_token(special_token.clone(), token_id);
            token_id += 1;
        }
        
        // Add characters sorted by frequency
        let mut char_freq_vec: Vec<_> = self.character_frequencies.iter().collect();
        char_freq_vec.sort_by(|a, b| b.1.cmp(a.1));
        
        for (&ch, &freq) in char_freq_vec {
            if freq >= self.min_frequency {
                vocab.add_token(ch.to_string(), token_id);
                token_id += 1;
            }
        }
        
        println!("Initialized vocabulary with {} tokens", vocab.len());
        Ok(vocab)
    }
    
    fn initialize_word_splits(&self) -> Result<HashMap<String, Vec<String>>, TokenizerError> {
        let mut word_splits = HashMap::new();
        
        for (word, &freq) in &self.word_frequencies {
            if freq >= self.min_frequency {
                // Split word into characters, add </w> to mark word boundary
                let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                if let Some(last) = chars.last_mut() {
                    last.push_str("</w>");
                }
                word_splits.insert(word.clone(), chars);
            }
        }
        
        println!("Initialized {} word splits", word_splits.len());
        Ok(word_splits)
    }
    
    fn learn_bpe_merges(
        &self,
        word_splits: &mut HashMap<String, Vec<String>>,
        vocab: &mut Vocab,
    ) -> Result<Vec<(String, String)>, TokenizerError> {
        let mut merges = Vec::new();
        let mut token_id = vocab.len() as u32;
        
        let target_vocab_size = self.vocab_size;
        
        println!("Learning BPE merges...");
        
        while vocab.len() < target_vocab_size {
            // Find the most frequent pair
            let pair_frequencies = self.count_pair_frequencies(word_splits);
            
            if pair_frequencies.is_empty() {
                break;
            }
            
            let most_frequent_pair = pair_frequencies.iter()
                .max_by_key(|(_, &freq)| freq)
                .map(|((a, b), _)| (a.clone(), b.clone()));
            
            if let Some((left, right)) = most_frequent_pair {
                let merged = format!("{}{}", left, right);
                
                // Add merged token to vocabulary
                vocab.add_token(merged.clone(), token_id);
                token_id += 1;
                
                // Record merge
                merges.push((left.clone(), right.clone()));
                
                // Apply merge to all word splits
                self.apply_merge(word_splits, &left, &right, &merged);
                
                if merges.len() % 1000 == 0 {
                    println!("Learned {} merges, vocab size: {}", merges.len(), vocab.len());
                }
            } else {
                break;
            }
        }
        
        println!("Learned {} total merges", merges.len());
        Ok(merges)
    }
    
    fn count_pair_frequencies(
        &self,
        word_splits: &HashMap<String, Vec<String>>,
    ) -> HashMap<(String, String), usize> {
        let mut pair_frequencies = HashMap::new();
        
        for (word, splits) in word_splits {
            let word_freq = self.word_frequencies.get(word).unwrap_or(&1);
            
            for window in splits.windows(2) {
                if let [left, right] = window {
                    let pair = (left.clone(), right.clone());
                    *pair_frequencies.entry(pair).or_insert(0) += word_freq;
                }
            }
        }
        
        pair_frequencies
    }
    
    fn apply_merge(
        &self,
        word_splits: &mut HashMap<String, Vec<String>>,
        left: &str,
        right: &str,
        merged: &str,
    ) {
        for splits in word_splits.values_mut() {
            let mut new_splits = Vec::new();
            let mut i = 0;
            
            while i < splits.len() {
                if i < splits.len() - 1 && splits[i] == left && splits[i + 1] == right {
                    new_splits.push(merged.to_string());
                    i += 2; // Skip both tokens
                } else {
                    new_splits.push(splits[i].clone());
                    i += 1;
                }
            }
            
            *splits = new_splits;
        }
    }
}

impl CustomBPETokenizer {
    pub fn encode_word(&self, word: &str) -> Vec<String> {
        let mut word_chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if let Some(last) = word_chars.last_mut() {
            last.push_str("</w>");
        }
        
        // Apply merges in order
        for (left, right) in &self.merges {
            let mut new_chars = Vec::new();
            let mut i = 0;
            
            while i < word_chars.len() {
                if i < word_chars.len() - 1 && word_chars[i] == *left && word_chars[i + 1] == *right {
                    let merged = format!("{}{}", left, right);
                    new_chars.push(merged);
                    i += 2;
                } else {
                    new_chars.push(word_chars[i].clone());
                    i += 1;
                }
            }
            
            word_chars = new_chars;
        }
        
        word_chars
    }
}

impl Tokenizer for CustomBPETokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput, TokenizerError> {
        let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
        let mut all_tokens = Vec::new();
        
        for word in words {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !clean_word.is_empty() {
                let word_tokens = self.encode_word(clean_word);
                all_tokens.extend(word_tokens);
            }
        }
        
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut attention_mask = Vec::new();
        
        for token in all_tokens {
            let token_id = self.vocab.get_id(&token).unwrap_or(self.unk_token_id);
            ids.push(token_id);
            tokens.push(token);
            attention_mask.push(1);
        }
        
        Ok(TokenizedInput {
            ids,
            tokens,
            attention_mask,
            ..Default::default()
        })
    }
    
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut tokens = Vec::new();
        
        for &id in ids {
            if let Some(token) = self.vocab.get_token(id) {
                if token != "[PAD]" {
                    tokens.push(token.clone());
                }
            }
        }
        
        // Reconstruct words by joining tokens and handling </w> markers
        let text = tokens.join("")
            .replace("</w>", " ")
            .trim()
            .to_string();
        
        Ok(text)
    }
    
    fn get_vocab(&self) -> &Vocab {
        &self.vocab
    }
    
    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// Example usage
fn example_bpe_training() -> Result<(), Box<dyn std::error::Error>> {
    let mut trainer = BPETrainer::new(1000)
        .with_min_frequency(2);
    
    // Training corpus
    let training_texts = vec![
        "the quick brown fox jumps over the lazy dog",
        "machine learning algorithms require large datasets for training",
        "natural language processing involves tokenization and embedding",
        "transformers have revolutionized the field of nlp",
        "custom tokenizers can be built for specific domains",
        "byte pair encoding learns subword units from data",
        "frequent character pairs are merged to form new tokens",
        "this process continues until the desired vocabulary size",
        "the resulting tokenizer can handle out of vocabulary words",
        "subword tokenization improves handling of rare words",
    ];
    
    // Train BPE tokenizer
    let tokenizer = trainer.train(&training_texts)?;
    
    // Test the trained tokenizer
    let test_texts = vec![
        "the machine learns from data",
        "tokenization is important for nlp",
        "unknown words are handled gracefully",
    ];
    
    for test_text in test_texts {
        let encoded = tokenizer.encode(test_text)?;
        let decoded = tokenizer.decode(&encoded.ids)?;
        
        println!("Original: {}", test_text);
        println!("Tokens: {:?}", encoded.tokens);
        println!("Decoded: {}", decoded);
        println!();
    }
    
    Ok(())
}
```

This tutorial provides a comprehensive foundation for building custom tokenizers. The examples progress from simple word-based tokenizers to sophisticated domain-specific tokenizers with training capabilities. Each component can be extended and customized based on specific requirements.

The key takeaways are:
1. Understand the tokenization pipeline and its components
2. Design vocabulary and rules based on your domain
3. Implement proper training procedures for data-driven approaches
4. Test and validate your tokenizer thoroughly
5. Optimize for performance in production environments