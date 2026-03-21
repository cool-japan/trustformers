//! Corpus processing and streaming utilities.
//!
//! This module provides efficient corpus processing capabilities for tokenizer training,
//! including chunked processing for large files, streaming utilities for memory-efficient
//! training, and preprocessing pipelines for text normalization.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use trustformers_core::errors::Result;

/// Corpus processor for efficient large-scale training data handling.
pub struct CorpusProcessor {
    chunk_size: usize,
    max_line_length: usize,
    skip_empty_lines: bool,
    lowercase: bool,
}

impl CorpusProcessor {
    /// Create a new corpus processor with default settings.
    pub fn new() -> Self {
        Self {
            chunk_size: 10000,
            max_line_length: 1000,
            skip_empty_lines: true,
            lowercase: false,
        }
    }

    /// Set the chunk size for processing large files.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set the maximum line length filter.
    pub fn with_max_line_length(mut self, max_length: usize) -> Self {
        self.max_line_length = max_length;
        self
    }

    /// Enable or disable lowercase conversion.
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Enable or disable empty line skipping.
    pub fn with_skip_empty_lines(mut self, skip_empty: bool) -> Self {
        self.skip_empty_lines = skip_empty;
        self
    }

    /// Process a large corpus file in chunks to manage memory usage.
    ///
    /// This method reads a corpus file and returns processed text chunks,
    /// applying filtering and normalization as configured.
    pub fn process_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<String>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut texts = Vec::new();
        let mut current_chunk = Vec::new();

        for line in reader.lines() {
            let line = line?;

            if self.skip_empty_lines && line.trim().is_empty() {
                continue;
            }

            if line.len() > self.max_line_length {
                continue;
            }

            let processed_line = if self.lowercase { line.to_lowercase() } else { line };

            current_chunk.push(processed_line);

            if current_chunk.len() >= self.chunk_size {
                texts.append(&mut current_chunk);
            }
        }

        if !current_chunk.is_empty() {
            texts.extend(current_chunk);
        }

        Ok(texts)
    }

    /// Process multiple corpus files and combine the results.
    pub fn process_files<P: AsRef<Path>>(&self, paths: &[P]) -> Result<Vec<String>> {
        let mut all_texts = Vec::new();

        for path in paths {
            let texts = self.process_file(path)?;
            all_texts.extend(texts);
        }

        Ok(all_texts)
    }

    /// Stream process a large file without loading everything into memory.
    ///
    /// This method calls the provided callback function for each chunk of text,
    /// allowing for memory-efficient processing of very large corpora.
    pub fn stream_process_file<P, F>(&self, path: P, mut callback: F) -> Result<()>
    where
        P: AsRef<Path>,
        F: FnMut(&[String]) -> Result<()>,
    {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut current_chunk = Vec::new();

        for line in reader.lines() {
            let line = line?;

            if self.skip_empty_lines && line.trim().is_empty() {
                continue;
            }

            if line.len() > self.max_line_length {
                continue;
            }

            let processed_line = if self.lowercase { line.to_lowercase() } else { line };
            current_chunk.push(processed_line);

            if current_chunk.len() >= self.chunk_size {
                callback(&current_chunk)?;
                current_chunk.clear();
            }
        }

        // Process remaining lines
        if !current_chunk.is_empty() {
            callback(&current_chunk)?;
        }

        Ok(())
    }

    /// Calculate corpus statistics without loading all data into memory.
    pub fn analyze_corpus<P: AsRef<Path>>(&self, path: P) -> Result<CorpusStats> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut stats = CorpusStats::new();

        for line in reader.lines() {
            let line = line?;
            stats.total_lines += 1;

            if line.trim().is_empty() {
                stats.empty_lines += 1;
                continue;
            }

            if line.len() > self.max_line_length {
                stats.filtered_lines += 1;
                continue;
            }

            stats.processed_lines += 1;
            stats.total_chars += line.chars().count();
            stats.total_words += line.split_whitespace().count();

            let line_length = line.chars().count();
            stats.min_line_length = stats.min_line_length.min(line_length);
            stats.max_line_length = stats.max_line_length.max(line_length);
        }

        stats.avg_line_length = if stats.processed_lines > 0 {
            stats.total_chars as f64 / stats.processed_lines as f64
        } else {
            0.0
        };

        stats.avg_words_per_line = if stats.processed_lines > 0 {
            stats.total_words as f64 / stats.processed_lines as f64
        } else {
            0.0
        };

        Ok(stats)
    }

    /// Split a corpus file into training and validation sets.
    pub fn split_corpus<P: AsRef<Path>>(
        &self,
        input_path: P,
        train_path: P,
        val_path: P,
        validation_split: f64,
    ) -> Result<(usize, usize)> {
        let file = File::open(&input_path)?;
        let reader = BufReader::new(file);

        let train_file = File::create(&train_path)?;
        let val_file = File::create(&val_path)?;
        let mut train_writer = BufWriter::new(train_file);
        let mut val_writer = BufWriter::new(val_file);

        let mut train_count = 0;
        let mut val_count = 0;

        for line in reader.lines() {
            let line = line?;

            // Deterministic splitting based on line hash
            let line_hash = self.hash_line(&line);
            let is_validation = (line_hash as f64 / u32::MAX as f64) < validation_split;

            if is_validation {
                writeln!(val_writer, "{}", line)?;
                val_count += 1;
            } else {
                writeln!(train_writer, "{}", line)?;
                train_count += 1;
            }
        }

        train_writer.flush()?;
        val_writer.flush()?;

        Ok((train_count, val_count))
    }

    /// Simple hash function for deterministic splitting.
    fn hash_line(&self, line: &str) -> u32 {
        let mut hash = 5381u32;
        for byte in line.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u32);
        }
        hash
    }

    /// Get processor configuration.
    pub fn get_config(&self) -> CorpusProcessorConfig {
        CorpusProcessorConfig {
            chunk_size: self.chunk_size,
            max_line_length: self.max_line_length,
            skip_empty_lines: self.skip_empty_lines,
            lowercase: self.lowercase,
        }
    }
}

impl Default for CorpusProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for corpus processor.
#[derive(Debug, Clone)]
pub struct CorpusProcessorConfig {
    pub chunk_size: usize,
    pub max_line_length: usize,
    pub skip_empty_lines: bool,
    pub lowercase: bool,
}

/// Statistics about a processed corpus.
#[derive(Debug, Clone)]
pub struct CorpusStats {
    pub total_lines: usize,
    pub processed_lines: usize,
    pub empty_lines: usize,
    pub filtered_lines: usize,
    pub total_chars: usize,
    pub total_words: usize,
    pub min_line_length: usize,
    pub max_line_length: usize,
    pub avg_line_length: f64,
    pub avg_words_per_line: f64,
}

impl CorpusStats {
    /// Create new empty corpus statistics.
    pub fn new() -> Self {
        Self {
            total_lines: 0,
            processed_lines: 0,
            empty_lines: 0,
            filtered_lines: 0,
            total_chars: 0,
            total_words: 0,
            min_line_length: usize::MAX,
            max_line_length: 0,
            avg_line_length: 0.0,
            avg_words_per_line: 0.0,
        }
    }

    /// Calculate processing efficiency (fraction of lines kept).
    pub fn processing_efficiency(&self) -> f64 {
        if self.total_lines > 0 {
            self.processed_lines as f64 / self.total_lines as f64
        } else {
            0.0
        }
    }

    /// Generate a summary report of corpus statistics.
    pub fn summary(&self) -> String {
        format!(
            "Corpus Statistics:\n\
             - Total Lines: {}\n\
             - Processed Lines: {} ({:.1}%)\n\
             - Empty Lines: {}\n\
             - Filtered Lines: {}\n\
             - Total Characters: {}\n\
             - Total Words: {}\n\
             - Line Length: {} - {} (avg: {:.1})\n\
             - Words per Line: {:.1}\n\
             - Processing Efficiency: {:.1}%",
            self.total_lines,
            self.processed_lines,
            self.processing_efficiency() * 100.0,
            self.empty_lines,
            self.filtered_lines,
            self.total_chars,
            self.total_words,
            if self.min_line_length == usize::MAX { 0 } else { self.min_line_length },
            self.max_line_length,
            self.avg_line_length,
            self.avg_words_per_line,
            self.processing_efficiency() * 100.0
        )
    }
}

impl Default for CorpusStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility for preprocessing text before tokenizer training.
pub struct TextPreprocessor {
    remove_urls: bool,
    remove_emails: bool,
    normalize_whitespace: bool,
    remove_html_tags: bool,
}

impl TextPreprocessor {
    /// Create a new text preprocessor with default settings.
    pub fn new() -> Self {
        Self {
            remove_urls: false,
            remove_emails: false,
            normalize_whitespace: true,
            remove_html_tags: false,
        }
    }

    /// Enable URL removal.
    pub fn with_url_removal(mut self, remove: bool) -> Self {
        self.remove_urls = remove;
        self
    }

    /// Enable email removal.
    pub fn with_email_removal(mut self, remove: bool) -> Self {
        self.remove_emails = remove;
        self
    }

    /// Enable whitespace normalization.
    pub fn with_whitespace_normalization(mut self, normalize: bool) -> Self {
        self.normalize_whitespace = normalize;
        self
    }

    /// Enable HTML tag removal.
    pub fn with_html_tag_removal(mut self, remove: bool) -> Self {
        self.remove_html_tags = remove;
        self
    }

    /// Preprocess a single text according to configuration.
    pub fn preprocess(&self, text: &str) -> String {
        let mut result = text.to_string();

        if self.remove_html_tags {
            result = self.remove_html_tags_impl(&result);
        }

        if self.remove_urls {
            result = self.remove_urls_impl(&result);
        }

        if self.remove_emails {
            result = self.remove_emails_impl(&result);
        }

        if self.normalize_whitespace {
            result = self.normalize_whitespace_impl(&result);
        }

        result
    }

    /// Remove HTML tags from text.
    fn remove_html_tags_impl(&self, text: &str) -> String {
        // Simple HTML tag removal - could be enhanced with proper HTML parsing
        let mut result = String::new();
        let mut in_tag = false;

        for ch in text.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => result.push(ch),
                _ => {},
            }
        }

        result
    }

    /// Remove URLs from text.
    fn remove_urls_impl(&self, text: &str) -> String {
        // Simple URL removal - matches http(s):// patterns
        let words: Vec<&str> = text.split_whitespace().collect();
        let filtered_words: Vec<&str> = words
            .into_iter()
            .filter(|word| !word.starts_with("http://") && !word.starts_with("https://"))
            .collect();
        filtered_words.join(" ")
    }

    /// Remove email addresses from text.
    fn remove_emails_impl(&self, text: &str) -> String {
        // Simple email removal - matches patterns with @
        let words: Vec<&str> = text.split_whitespace().collect();
        let filtered_words: Vec<&str> = words
            .into_iter()
            .filter(|word| !word.contains('@') || !word.contains('.'))
            .collect();
        filtered_words.join(" ")
    }

    /// Normalize whitespace in text.
    fn normalize_whitespace_impl(&self, text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_processor_creation() {
        let processor = CorpusProcessor::new()
            .with_chunk_size(100)
            .with_max_line_length(50)
            .with_lowercase(true);

        let config = processor.get_config();
        assert_eq!(config.chunk_size, 100);
        assert_eq!(config.max_line_length, 50);
        assert!(config.lowercase);
        assert!(config.skip_empty_lines);
    }

    #[test]
    fn test_corpus_stats() {
        let mut stats = CorpusStats::new();
        assert_eq!(stats.total_lines, 0);
        assert_eq!(stats.processing_efficiency(), 0.0);

        stats.total_lines = 100;
        stats.processed_lines = 80;
        assert_eq!(stats.processing_efficiency(), 0.8);
    }

    #[test]
    fn test_text_preprocessor() {
        let preprocessor = TextPreprocessor::new()
            .with_url_removal(true)
            .with_email_removal(true)
            .with_whitespace_normalization(true)
            .with_html_tag_removal(true);

        let text =
            "Hello  world! Visit https://example.com or email test@example.com. <b>Bold text</b>";
        let processed = preprocessor.preprocess(text);

        assert!(!processed.contains("https://"));
        assert!(!processed.contains("test@example.com"));
        assert!(!processed.contains("<b>"));
        assert!(!processed.contains("</b>"));
        assert!(processed.contains("Hello world!"));
        assert!(processed.contains("Bold text"));
    }

    #[test]
    fn test_html_tag_removal() {
        let preprocessor = TextPreprocessor::new().with_html_tag_removal(true);
        let text = "<p>Hello <b>world</b>!</p>";
        let processed = preprocessor.preprocess(text);
        assert_eq!(processed, "Hello world!");
    }

    #[test]
    fn test_whitespace_normalization() {
        let preprocessor = TextPreprocessor::new().with_whitespace_normalization(true);
        let text = "Hello     world!\n\nHow   are you?";
        let processed = preprocessor.preprocess(text);
        assert_eq!(processed, "Hello world! How are you?");
    }

    #[test]
    fn test_corpus_processor_hash_consistency() {
        let processor = CorpusProcessor::new();
        let line = "test line for hashing";
        let hash1 = processor.hash_line(line);
        let hash2 = processor.hash_line(line);
        assert_eq!(hash1, hash2);

        let different_line = "different test line";
        let hash3 = processor.hash_line(different_line);
        assert_ne!(hash1, hash3);
    }
}
