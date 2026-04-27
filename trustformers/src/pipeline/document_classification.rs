//! # Document Classification Pipeline
//!
//! Classify long documents (reports, articles) by category, handling text longer than
//! 512 tokens via overlapping chunk aggregation.
//!
//! ## Supported models
//! - **BART** (facebook/bart-large-mnli) — zero-shot NLI-based classification
//! - Any sequence-classification model
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::document_classification::{
//!     DocumentClassificationConfig, DocumentClassificationPipeline,
//! };
//!
//! let config = DocumentClassificationConfig::default();
//! let pipeline = DocumentClassificationPipeline::new(config)?;
//! let result = pipeline.classify("Long article text...")?;
//! println!("Label: {} (score: {:.3})", result.label, result.score);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the document classification pipeline.
#[derive(Debug, Error)]
pub enum DocClassError {
    /// Input document was empty.
    #[error("Empty document")]
    EmptyDocument,
    /// No classification labels have been configured.
    #[error("No labels configured")]
    NoLabels,
    /// Underlying model error.
    #[error("Model error: {0}")]
    ModelError(String),
}

// ---------------------------------------------------------------------------
// ChunkAggregation
// ---------------------------------------------------------------------------

/// Strategy for combining per-chunk classification scores into a document-level result.
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkAggregation {
    /// Average logits/scores across all chunks.
    MeanScore,
    /// Use only the prediction of the first chunk.
    FirstChunk,
    /// Take the label with the most chunk-level wins.
    MajorityVote,
    /// Take the chunk that has the highest maximum confidence score.
    MaxConfidence,
}

// ---------------------------------------------------------------------------
// DocumentClassificationConfig
// ---------------------------------------------------------------------------

/// Configuration for [`DocumentClassificationPipeline`].
#[derive(Debug, Clone)]
pub struct DocumentClassificationConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Class labels.
    pub labels: Vec<String>,
    /// Words per chunk.
    pub chunk_size: usize,
    /// Overlapping words between consecutive chunks.
    pub chunk_overlap: usize,
    /// Strategy for merging chunk-level predictions.
    pub aggregation: ChunkAggregation,
    /// Maximum number of chunks to process per document.
    pub max_chunks: usize,
}

impl Default for DocumentClassificationConfig {
    fn default() -> Self {
        Self {
            model_name: "facebook/bart-large-mnli".to_string(),
            labels: vec![
                "technology".to_string(),
                "business".to_string(),
                "sports".to_string(),
                "politics".to_string(),
                "entertainment".to_string(),
            ],
            chunk_size: 512,
            chunk_overlap: 64,
            aggregation: ChunkAggregation::MeanScore,
            max_chunks: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// DocumentClassification
// ---------------------------------------------------------------------------

/// Classification result for a single document.
#[derive(Debug, Clone)]
pub struct DocumentClassification {
    /// Top predicted label.
    pub label: String,
    /// Score of the top label.
    pub score: f32,
    /// All (label, score) pairs sorted in descending order by score.
    pub all_scores: Vec<(String, f32)>,
    /// Number of chunks the document was split into.
    pub num_chunks: usize,
    /// Per-chunk top predicted label.
    pub chunk_predictions: Vec<String>,
}

impl DocumentClassification {
    /// Return the top-`k` (label, score) pairs.
    pub fn top_k(&self, k: usize) -> Vec<&(String, f32)> {
        self.all_scores.iter().take(k).collect()
    }
}

// ---------------------------------------------------------------------------
// DocumentClassificationPipeline
// ---------------------------------------------------------------------------

/// Long-document classification pipeline with chunk-based aggregation.
pub struct DocumentClassificationPipeline {
    config: DocumentClassificationConfig,
}

impl DocumentClassificationPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: DocumentClassificationConfig) -> Result<Self, DocClassError> {
        if config.labels.is_empty() {
            return Err(DocClassError::NoLabels);
        }
        Ok(Self { config })
    }

    /// Split `text` into overlapping word chunks according to the configuration.
    ///
    /// - Each chunk contains at most `chunk_size` words.
    /// - Consecutive chunks overlap by `chunk_overlap` words.
    /// - At most `max_chunks` chunks are produced.
    pub fn chunk_document(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return vec![];
        }

        let chunk_size = self.config.chunk_size.max(1);
        let overlap = self.config.chunk_overlap.min(chunk_size.saturating_sub(1));
        let step = chunk_size.saturating_sub(overlap).max(1);

        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < words.len() && chunks.len() < self.config.max_chunks {
            let end = (start + chunk_size).min(words.len());
            chunks.push(words[start..end].join(" "));
            start += step;
        }
        chunks
    }

    /// Classify a single text chunk, returning (label, score) pairs that sum to ~1.
    ///
    /// Mock: deterministic scores based on chunk hash and label index.
    fn classify_chunk(&self, chunk: &str) -> Result<Vec<(String, f32)>, DocClassError> {
        let num_labels = self.config.labels.len();
        let hash_val = simple_hash(chunk);

        // Generate raw unnormalised scores per label.
        let raw: Vec<f64> = (0..num_labels)
            .map(|i| ((hash_val as f64 + i as f64 * 137.0) * 0.001).sin().abs() + 0.1)
            .collect();

        // Softmax normalisation so scores sum to 1.
        let max_raw = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = raw.iter().map(|v| (v - max_raw).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();

        let mut scored: Vec<(String, f32)> = self
            .config
            .labels
            .iter()
            .zip(exps.iter())
            .map(|(label, &e)| (label.clone(), (e / sum_exp) as f32))
            .collect();

        // Sort descending.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }

    /// Classify a full document, chunking it internally.
    pub fn classify(&self, document: &str) -> Result<DocumentClassification, DocClassError> {
        if document.trim().is_empty() {
            return Err(DocClassError::EmptyDocument);
        }

        let chunks = self.chunk_document(document);
        let num_chunks = chunks.len();

        // Classify each chunk.
        let chunk_results: Vec<Vec<(String, f32)>> = chunks
            .iter()
            .map(|c| self.classify_chunk(c))
            .collect::<Result<Vec<_>, _>>()?;

        let chunk_predictions: Vec<String> = chunk_results
            .iter()
            .map(|scores| {
                scores
                    .first()
                    .map(|(l, _)| l.clone())
                    .unwrap_or_default()
            })
            .collect();

        let all_scores = self.aggregate_scores(&chunk_results);

        let (label, score) = all_scores
            .first()
            .map(|(l, s)| (l.clone(), *s))
            .unwrap_or_else(|| (String::new(), 0.0));

        Ok(DocumentClassification {
            label,
            score,
            all_scores,
            num_chunks,
            chunk_predictions,
        })
    }

    /// Batch classify multiple documents.
    pub fn classify_batch(
        &self,
        documents: &[&str],
    ) -> Result<Vec<DocumentClassification>, DocClassError> {
        documents.iter().map(|d| self.classify(d)).collect()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Aggregate per-chunk score vectors into a single ranking.
    fn aggregate_scores(
        &self,
        chunk_results: &[Vec<(String, f32)>],
    ) -> Vec<(String, f32)> {
        if chunk_results.is_empty() {
            return vec![];
        }

        match &self.config.aggregation {
            ChunkAggregation::FirstChunk => chunk_results[0].clone(),

            ChunkAggregation::MeanScore => {
                let num_labels = self.config.labels.len();
                let mut sums = vec![0.0f32; num_labels];
                // Use label index mapping from the first chunk (scores are sorted, so re-map).
                let label_order: Vec<String> = self.config.labels.clone();
                for chunk_scores in chunk_results {
                    let score_map: std::collections::HashMap<&str, f32> = chunk_scores
                        .iter()
                        .map(|(l, s)| (l.as_str(), *s))
                        .collect();
                    for (i, label) in label_order.iter().enumerate() {
                        sums[i] += score_map.get(label.as_str()).copied().unwrap_or(0.0);
                    }
                }
                let n = chunk_results.len() as f32;
                let mut averaged: Vec<(String, f32)> = label_order
                    .into_iter()
                    .enumerate()
                    .map(|(i, l)| (l, sums[i] / n))
                    .collect();
                averaged
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                averaged
            }

            ChunkAggregation::MajorityVote => {
                let mut vote_counts: std::collections::HashMap<String, usize> =
                    std::collections::HashMap::new();
                for chunk_scores in chunk_results {
                    if let Some((top_label, _)) = chunk_scores.first() {
                        *vote_counts.entry(top_label.clone()).or_insert(0) += 1;
                    }
                }
                // Convert vote counts to pseudo-probabilities.
                let total_votes = chunk_results.len() as f32;
                let mut scored: Vec<(String, f32)> = self
                    .config
                    .labels
                    .iter()
                    .map(|l| {
                        let votes = *vote_counts.get(l).unwrap_or(&0) as f32;
                        (l.clone(), votes / total_votes)
                    })
                    .collect();
                scored
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored
            }

            ChunkAggregation::MaxConfidence => {
                // Pick the chunk with the highest top score.
                let best_chunk = chunk_results
                    .iter()
                    .max_by(|a, b| {
                        let score_a = a.first().map(|(_, s)| *s).unwrap_or(0.0);
                        let score_b = b.first().map(|(_, s)| *s).unwrap_or(0.0);
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .cloned()
                    .unwrap_or_default();
                best_chunk
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DocumentLengthFeatures
// ---------------------------------------------------------------------------

/// Basic length-based feature set for a document.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentLengthFeatures {
    pub char_count: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub avg_word_length: f32,
}

// ---------------------------------------------------------------------------
// DocumentFeatureExtractor — text feature helpers
// ---------------------------------------------------------------------------

/// Stateless utility for extracting linguistic features from documents.
pub struct DocumentFeatureExtractor;

impl DocumentFeatureExtractor {
    /// Extract word n-grams of size `n` from `text`.
    ///
    /// Returns each n-gram as a space-joined string.
    pub fn extract_ngrams(text: &str, n: usize) -> Vec<String> {
        if n == 0 {
            return Vec::new();
        }
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < n {
            return Vec::new();
        }
        words.windows(n).map(|window| window.join(" ")).collect()
    }

    /// Compute the TF-IDF score for `term` in `doc` against a `corpus` of documents.
    ///
    /// TF = (occurrences of term in doc) / (total words in doc).
    /// IDF = ln((1 + N) / (1 + df)) + 1  where df = count of corpus docs containing term.
    pub fn tfidf_score(term: &str, doc: &str, corpus: &[String]) -> f32 {
        let doc_words: Vec<&str> = doc.split_whitespace().collect();
        let doc_len = doc_words.len();
        if doc_len == 0 {
            return 0.0;
        }
        let term_lower = term.to_lowercase();
        let tf_count = doc_words
            .iter()
            .filter(|w| w.to_lowercase() == term_lower)
            .count();
        if tf_count == 0 {
            return 0.0;
        }
        let tf = tf_count as f32 / doc_len as f32;

        let n = corpus.len() as f32;
        let df = corpus
            .iter()
            .filter(|d| d.to_lowercase().contains(&term_lower))
            .count() as f32;
        let idf = ((1.0 + n) / (1.0 + df)).ln() + 1.0;
        tf * idf
    }

    /// Compute length-based features for `text`.
    pub fn document_length_features(text: &str) -> DocumentLengthFeatures {
        let char_count = text.chars().count();
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();
        // Simple sentence split on `.`, `!`, `?`
        let sentence_count = DocumentHierarchy::split_into_sentences(text).len();
        let avg_word_length = if word_count == 0 {
            0.0
        } else {
            words.iter().map(|w| w.chars().count()).sum::<usize>() as f32 / word_count as f32
        };
        DocumentLengthFeatures { char_count, word_count, sentence_count, avg_word_length }
    }
}

// ---------------------------------------------------------------------------
// DocumentHierarchy — structural text splitting
// ---------------------------------------------------------------------------

/// Utilities for decomposing documents into structural units.
pub struct DocumentHierarchy;

impl DocumentHierarchy {
    /// Split `text` into sentences, splitting on `.`, `!`, `?` followed by whitespace.
    /// Empty results are discarded.
    pub fn split_into_sentences(text: &str) -> Vec<String> {
        // Split on sentence-ending punctuation followed by space or end-of-string.
        let mut sentences = Vec::new();
        let mut current = String::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut i = 0;
        while i < len {
            let c = chars[i];
            current.push(c);
            if matches!(c, '.' | '!' | '?') {
                // Peek ahead — if next char is whitespace or we're at end, treat as boundary
                let next_is_boundary = i + 1 >= len
                    || chars[i + 1].is_whitespace();
                if next_is_boundary {
                    let trimmed = current.trim().to_string();
                    if !trimmed.is_empty() {
                        sentences.push(trimmed);
                    }
                    current = String::new();
                    // skip following whitespace
                    i += 1;
                    while i < len && chars[i].is_whitespace() {
                        i += 1;
                    }
                    continue;
                }
            }
            i += 1;
        }
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }
        sentences
    }

    /// Split `text` into paragraphs using double newline (`\n\n`) as separator.
    /// Empty paragraphs are discarded.
    pub fn split_into_paragraphs(text: &str) -> Vec<String> {
        text.split("\n\n")
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect()
    }

    /// Produce overlapping chunks of `window_size` words with a step of `stride` words.
    ///
    /// `stride` must be ≥ 1; if `stride` == 0 it is treated as 1.
    pub fn sliding_window_chunks(text: &str, window_size: usize, stride: usize) -> Vec<String> {
        if window_size == 0 {
            return Vec::new();
        }
        let stride = stride.max(1);
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() || words.len() < window_size {
            if !words.is_empty() {
                return vec![words.join(" ")];
            }
            return Vec::new();
        }
        let mut chunks = Vec::new();
        let mut start = 0;
        while start + window_size <= words.len() {
            chunks.push(words[start..start + window_size].join(" "));
            start += stride;
        }
        // Include any remaining words that don't fill a full window
        if start < words.len() {
            chunks.push(words[start..].join(" "));
        }
        chunks
    }
}

// ---------------------------------------------------------------------------
// Specialised classifiers (wrappers around the generic pipeline)
// ---------------------------------------------------------------------------

/// A document classifier pre-configured for legal documents.
pub struct LegalDocumentClassifier {
    pipeline: DocumentClassificationPipeline,
}

impl LegalDocumentClassifier {
    pub fn new() -> Result<Self, DocClassError> {
        let config = DocumentClassificationConfig {
            model_name: "nlpaueb/legal-bert-base-uncased".to_string(),
            labels: vec![
                "contract".to_string(),
                "legislation".to_string(),
                "judgment".to_string(),
                "patent".to_string(),
                "regulation".to_string(),
            ],
            chunk_size: 256,
            chunk_overlap: 32,
            aggregation: ChunkAggregation::MeanScore,
            max_chunks: 20,
        };
        Ok(Self { pipeline: DocumentClassificationPipeline::new(config)? })
    }

    pub fn classify(&self, document: &str) -> Result<DocumentClassification, DocClassError> {
        self.pipeline.classify(document)
    }
}

/// A document classifier pre-configured for scientific papers.
pub struct ScientificDocumentClassifier {
    pipeline: DocumentClassificationPipeline,
}

impl ScientificDocumentClassifier {
    pub fn new() -> Result<Self, DocClassError> {
        let config = DocumentClassificationConfig {
            model_name: "allenai/scibert_scivocab_uncased".to_string(),
            labels: vec![
                "computer_science".to_string(),
                "medicine".to_string(),
                "physics".to_string(),
                "biology".to_string(),
                "chemistry".to_string(),
                "mathematics".to_string(),
            ],
            chunk_size: 512,
            chunk_overlap: 64,
            aggregation: ChunkAggregation::MeanScore,
            max_chunks: 10,
        };
        Ok(Self { pipeline: DocumentClassificationPipeline::new(config)? })
    }

    pub fn classify(&self, document: &str) -> Result<DocumentClassification, DocClassError> {
        self.pipeline.classify(document)
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Simple djb2-style hash for deterministic mock scoring.
fn simple_hash(text: &str) -> u64 {
    let mut h: u64 = 5381;
    for byte in text.bytes() {
        h = h.wrapping_mul(33).wrapping_add(byte as u64);
    }
    h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pipeline() -> DocumentClassificationPipeline {
        DocumentClassificationPipeline::new(DocumentClassificationConfig::default()).unwrap()
    }

    // Helper: build a text of `n` words.
    fn words(n: usize) -> String {
        (0..n).map(|i| format!("word{}", i)).collect::<Vec<_>>().join(" ")
    }

    // 1. chunk_document produces the expected number of chunks for a short text.
    #[test]
    fn chunk_document_basic() {
        let pipe = default_pipeline();
        let text = words(100);
        let chunks = pipe.chunk_document(&text);
        // 100 words, chunk_size=512, so we get 1 chunk.
        assert_eq!(chunks.len(), 1);
    }

    // 2. chunk_document overlap is honoured.
    #[test]
    fn chunk_document_overlap() {
        let config = DocumentClassificationConfig {
            chunk_size: 4,
            chunk_overlap: 2,
            max_chunks: 10,
            ..DocumentClassificationConfig::default()
        };
        let pipe = DocumentClassificationPipeline::new(config).unwrap();
        // 8 words, step=2: chunks starting at 0,2,4,6 → 4 chunks
        let text = "a b c d e f g h";
        let chunks = pipe.chunk_document(text);
        assert!(chunks.len() >= 2);
        // Second chunk should start at word index 2, overlapping with first.
        let second_words: Vec<&str> = chunks[1].split_whitespace().collect();
        let first_words: Vec<&str> = chunks[0].split_whitespace().collect();
        // Overlap: last 2 words of chunk[0] == first 2 words of chunk[1]
        assert_eq!(&first_words[2..], &second_words[..2]);
    }

    // 3. max_chunks is respected.
    #[test]
    fn max_chunks_limit() {
        let config = DocumentClassificationConfig {
            chunk_size: 2,
            chunk_overlap: 0,
            max_chunks: 3,
            ..DocumentClassificationConfig::default()
        };
        let pipe = DocumentClassificationPipeline::new(config).unwrap();
        let text = words(20);
        let chunks = pipe.chunk_document(&text);
        assert!(chunks.len() <= 3);
    }

    // 4. classify returns a valid label from the label list.
    #[test]
    fn classify_valid_label() {
        let pipe = default_pipeline();
        let result = pipe.classify("This is a test document about technology and AI.").unwrap();
        let labels = vec!["technology", "business", "sports", "politics", "entertainment"];
        assert!(labels.contains(&result.label.as_str()));
    }

    // 5. All scores sum to approximately 1.0.
    #[test]
    fn classify_scores_sum_to_one() {
        let pipe = default_pipeline();
        let result = pipe.classify("Document about the stock market and business.").unwrap();
        let total: f32 = result.all_scores.iter().map(|(_, s)| s).sum();
        assert!((total - 1.0).abs() < 0.01, "scores sum = {total}");
    }

    // 6. top_k returns at most k results.
    #[test]
    fn classify_top_k() {
        let pipe = default_pipeline();
        let result = pipe.classify("Sports match result football.").unwrap();
        let top2 = result.top_k(2);
        assert_eq!(top2.len(), 2);
    }

    // 7. chunk_predictions count matches num_chunks.
    #[test]
    fn chunk_predictions_count() {
        let pipe = default_pipeline();
        let result = pipe.classify("Simple text").unwrap();
        assert_eq!(result.chunk_predictions.len(), result.num_chunks);
    }

    // 8. MajorityVote aggregation produces a valid result.
    #[test]
    fn majority_vote_aggregation() {
        let config = DocumentClassificationConfig {
            aggregation: ChunkAggregation::MajorityVote,
            chunk_size: 2,
            chunk_overlap: 0,
            max_chunks: 5,
            ..DocumentClassificationConfig::default()
        };
        let pipe = DocumentClassificationPipeline::new(config).unwrap();
        let result = pipe.classify("politics government news election vote parliament").unwrap();
        let labels = vec!["technology", "business", "sports", "politics", "entertainment"];
        assert!(labels.contains(&result.label.as_str()));
    }

    // 9. FirstChunk aggregation is consistent with first chunk alone.
    #[test]
    fn first_chunk_aggregation() {
        let config = DocumentClassificationConfig {
            aggregation: ChunkAggregation::FirstChunk,
            ..DocumentClassificationConfig::default()
        };
        let pipe = DocumentClassificationPipeline::new(config).unwrap();
        let result = pipe.classify("Hello world from the first chunk test").unwrap();
        assert!(!result.label.is_empty());
    }

    // 10. Empty document returns DocClassError::EmptyDocument.
    #[test]
    fn classify_empty_document_error() {
        let pipe = default_pipeline();
        let result = pipe.classify("");
        assert!(matches!(result, Err(DocClassError::EmptyDocument)));
    }

    // 11. classify_batch returns one result per document.
    #[test]
    fn classify_batch_count() {
        let pipe = default_pipeline();
        let docs = vec!["First document.", "Second document.", "Third document."];
        let results = pipe.classify_batch(&docs).unwrap();
        assert_eq!(results.len(), 3);
    }

    // 12. No labels configured returns DocClassError::NoLabels.
    #[test]
    fn no_labels_error() {
        let config = DocumentClassificationConfig {
            labels: vec![],
            ..DocumentClassificationConfig::default()
        };
        let result = DocumentClassificationPipeline::new(config);
        assert!(matches!(result, Err(DocClassError::NoLabels)));
    }

    // 13. Long document (> max_chunks * chunk_size words) is capped at max_chunks.
    #[test]
    fn long_document_max_chunks() {
        let config = DocumentClassificationConfig {
            chunk_size: 10,
            chunk_overlap: 0,
            max_chunks: 3,
            ..DocumentClassificationConfig::default()
        };
        let pipe = DocumentClassificationPipeline::new(config).unwrap();
        // 200 words → would produce 20 chunks without cap
        let text = words(200);
        let result = pipe.classify(&text).unwrap();
        assert!(result.num_chunks <= 3);
    }

    // -----------------------------------------------------------------------
    // DocumentFeatureExtractor tests
    // -----------------------------------------------------------------------

    #[test]
    fn extract_ngrams_bigrams() {
        let ngrams = DocumentFeatureExtractor::extract_ngrams("the cat sat on", 2);
        assert_eq!(ngrams, vec!["the cat", "cat sat", "sat on"]);
    }

    #[test]
    fn extract_ngrams_unigrams() {
        let ngrams = DocumentFeatureExtractor::extract_ngrams("hello world", 1);
        assert_eq!(ngrams, vec!["hello", "world"]);
    }

    #[test]
    fn extract_ngrams_n_zero_empty() {
        let ngrams = DocumentFeatureExtractor::extract_ngrams("hello world", 0);
        assert!(ngrams.is_empty());
    }

    #[test]
    fn extract_ngrams_n_larger_than_text() {
        let ngrams = DocumentFeatureExtractor::extract_ngrams("one two", 5);
        assert!(ngrams.is_empty());
    }

    #[test]
    fn extract_ngrams_trigrams() {
        let ngrams = DocumentFeatureExtractor::extract_ngrams("a b c d", 3);
        assert_eq!(ngrams, vec!["a b c", "b c d"]);
    }

    #[test]
    fn tfidf_term_present() {
        let doc = "machine learning is great machine".to_string();
        let corpus = vec![
            "machine learning".to_string(),
            "deep learning".to_string(),
            "natural language processing".to_string(),
        ];
        let score = DocumentFeatureExtractor::tfidf_score("machine", &doc, &corpus);
        assert!(score > 0.0, "TF-IDF should be positive for present term");
    }

    #[test]
    fn tfidf_term_absent_returns_zero() {
        let doc = "the quick brown fox".to_string();
        let corpus = vec!["alpha beta".to_string()];
        let score = DocumentFeatureExtractor::tfidf_score("elephant", &doc, &corpus);
        assert!((score).abs() < 1e-8);
    }

    #[test]
    fn tfidf_rare_term_higher_than_common() {
        let doc = "the the the rare_word".to_string();
        let corpus: Vec<String> = (0..100).map(|i| {
            if i < 90 { "the common words".to_string() }
            else { "rare_word here".to_string() }
        }).collect();
        let tf_common = DocumentFeatureExtractor::tfidf_score("the", &doc, &corpus);
        let tf_rare = DocumentFeatureExtractor::tfidf_score("rare_word", &doc, &corpus);
        // rare_word appears in fewer docs → higher IDF → higher TF-IDF despite lower TF
        assert!(tf_rare > tf_common, "rare={tf_rare} common={tf_common}");
    }

    #[test]
    fn document_length_features_basic() {
        let text = "Hello world. This is a test! How are you?";
        let feats = DocumentFeatureExtractor::document_length_features(text);
        assert!(feats.word_count > 0);
        assert!(feats.char_count > 0);
        assert!(feats.sentence_count >= 1);
        assert!(feats.avg_word_length > 0.0);
    }

    #[test]
    fn document_length_features_empty() {
        let feats = DocumentFeatureExtractor::document_length_features("");
        assert_eq!(feats.word_count, 0);
        assert_eq!(feats.char_count, 0);
        assert!((feats.avg_word_length).abs() < 1e-8);
    }

    // -----------------------------------------------------------------------
    // DocumentHierarchy tests
    // -----------------------------------------------------------------------

    #[test]
    fn split_into_sentences_basic() {
        let text = "Hello world. This is a test. Goodbye!";
        let sentences = DocumentHierarchy::split_into_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("Hello"));
        assert!(sentences[1].contains("test"));
    }

    #[test]
    fn split_into_sentences_question_mark() {
        let text = "Are you there? Yes I am.";
        let sentences = DocumentHierarchy::split_into_sentences(text);
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn split_into_sentences_no_punctuation() {
        let text = "This has no sentence ending";
        let sentences = DocumentHierarchy::split_into_sentences(text);
        assert_eq!(sentences.len(), 1);
        assert!(sentences[0].contains("ending"));
    }

    #[test]
    fn split_into_paragraphs_double_newline() {
        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one.";
        let paras = DocumentHierarchy::split_into_paragraphs(text);
        assert_eq!(paras.len(), 3);
        assert!(paras[0].contains("First"));
        assert!(paras[1].contains("Second"));
    }

    #[test]
    fn split_into_paragraphs_single_paragraph() {
        let text = "Just one paragraph with no double newlines.";
        let paras = DocumentHierarchy::split_into_paragraphs(text);
        assert_eq!(paras.len(), 1);
    }

    #[test]
    fn split_into_paragraphs_empty_paragraphs_discarded() {
        let text = "First.\n\n\n\nSecond.";
        let paras = DocumentHierarchy::split_into_paragraphs(text);
        assert_eq!(paras.len(), 2);
    }

    #[test]
    fn sliding_window_chunks_basic() {
        let text = "a b c d e f";
        let chunks = DocumentHierarchy::sliding_window_chunks(text, 3, 2);
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0], "a b c");
    }

    #[test]
    fn sliding_window_chunks_stride_one() {
        let text = "w1 w2 w3 w4";
        let chunks = DocumentHierarchy::sliding_window_chunks(text, 2, 1);
        // windows: [w1 w2], [w2 w3], [w3 w4]
        assert!(chunks.len() >= 3);
    }

    #[test]
    fn sliding_window_chunks_window_larger_than_text() {
        let text = "hello world";
        let chunks = DocumentHierarchy::sliding_window_chunks(text, 10, 3);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn sliding_window_chunks_zero_window_empty() {
        let text = "hello world";
        let chunks = DocumentHierarchy::sliding_window_chunks(text, 0, 1);
        assert!(chunks.is_empty());
    }
}
