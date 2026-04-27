//! Multi-Document Summarization Pipeline
//!
//! Provides a production-quality pipeline for summarizing collections of
//! documents using multiple strategies: MapReduce, Hierarchical, Extractive,
//! and Fusion.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── SummarizationStrategy ─────────────────────────────────────────────────────

/// Strategy to use when summarizing multiple documents
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SummarizationStrategy {
    /// Summarize each document independently, then combine the summaries
    #[default]
    MapReduce,
    /// Multi-level tree reduction across the document set
    Hierarchical {
        /// Number of reduction levels (1 = single-pass, 2+ = tree)
        levels: usize,
    },
    /// Extract key sentences from all documents, then compose a summary
    Extractive,
    /// Treat all documents as a single fused context
    Fusion,
}

impl fmt::Display for SummarizationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SummarizationStrategy::MapReduce => write!(f, "map_reduce"),
            SummarizationStrategy::Hierarchical { levels } => {
                write!(f, "hierarchical(levels={})", levels)
            },
            SummarizationStrategy::Extractive => write!(f, "extractive"),
            SummarizationStrategy::Fusion => write!(f, "fusion"),
        }
    }
}

// ── CitationFormat ────────────────────────────────────────────────────────────

/// Citation style for cross-document references
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CitationFormat {
    /// \[1\], \[2\], … inline numeric references
    #[default]
    Numeric,
    /// Author (Year) APA style
    APA,
    /// Chicago style
    Chicago,
    /// Footnote style
    Footnote,
}

impl fmt::Display for CitationFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CitationFormat::Numeric => write!(f, "numeric"),
            CitationFormat::APA => write!(f, "apa"),
            CitationFormat::Chicago => write!(f, "chicago"),
            CitationFormat::Footnote => write!(f, "footnote"),
        }
    }
}

// ── KeyPointCategory ──────────────────────────────────────────────────────────

/// Semantic category of an extracted key point
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyPointCategory {
    /// A factual claim
    Fact,
    /// An opinion or assessment
    Opinion,
    /// A recommended or required action
    Action,
    /// A caution or risk notice
    Warning,
    /// A definition or explanation of a concept
    Definition,
}

impl fmt::Display for KeyPointCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyPointCategory::Fact => write!(f, "fact"),
            KeyPointCategory::Opinion => write!(f, "opinion"),
            KeyPointCategory::Action => write!(f, "action"),
            KeyPointCategory::Warning => write!(f, "warning"),
            KeyPointCategory::Definition => write!(f, "definition"),
        }
    }
}

// ── DocumentMetadata ──────────────────────────────────────────────────────────

/// Metadata associated with an input document
#[derive(Debug, Clone, Default)]
pub struct DocumentMetadata {
    /// Optional title of the document
    pub title: Option<String>,
    /// URL, file path, or other source identifier
    pub source: Option<String>,
    /// Publication or creation date
    pub date: Option<String>,
    /// Relevance score used for ranking (higher = more relevant)
    pub relevance_score: f32,
    /// Pre-computed word count (0 = compute on demand)
    pub word_count: usize,
}

// ── InputDocument ─────────────────────────────────────────────────────────────

/// A document to be summarized
#[derive(Debug, Clone)]
pub struct InputDocument {
    /// Raw text content
    pub content: String,
    /// Associated metadata
    pub metadata: DocumentMetadata,
}

impl InputDocument {
    /// Create a simple document with just content.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: DocumentMetadata::default(),
        }
    }

    /// Create a document with a title.
    pub fn with_title(content: impl Into<String>, title: impl Into<String>) -> Self {
        let content = content.into();
        let wc = content.split_whitespace().count();
        Self {
            content,
            metadata: DocumentMetadata {
                title: Some(title.into()),
                word_count: wc,
                ..Default::default()
            },
        }
    }
}

// ── MultiDocConfig ────────────────────────────────────────────────────────────

/// Configuration for the multi-document summarization pipeline
#[derive(Debug, Clone)]
pub struct MultiDocConfig {
    /// Algorithm to use for combining multiple documents
    pub strategy: SummarizationStrategy,
    /// Maximum word count of the final summary
    pub max_summary_length: usize,
    /// Minimum word count the final summary should reach
    pub min_summary_length: usize,
    /// Maximum number of documents the pipeline will accept
    pub max_documents: usize,
    /// Whether to extract and return key points
    pub extract_key_points: bool,
    /// Whether to deduplicate near-duplicate documents before processing
    pub deduplicate_content: bool,
    /// Jaccard similarity threshold above which two documents are considered duplicates
    pub similarity_threshold: f32,
    /// Whether to append citations to the summary
    pub include_citations: bool,
    /// Format to use for citations
    pub citation_format: CitationFormat,
}

impl Default for MultiDocConfig {
    fn default() -> Self {
        Self {
            strategy: SummarizationStrategy::default(),
            max_summary_length: 512,
            min_summary_length: 50,
            max_documents: 20,
            extract_key_points: true,
            deduplicate_content: true,
            similarity_threshold: 0.85,
            include_citations: false,
            citation_format: CitationFormat::default(),
        }
    }
}

// ── ExtractedKeyPoint ─────────────────────────────────────────────────────────

/// A key point extracted from one of the input documents
#[derive(Debug, Clone)]
pub struct ExtractedKeyPoint {
    /// The sentence or phrase
    pub text: String,
    /// Zero-based index of the source document in the original input slice
    pub source_doc_index: usize,
    /// Computed importance score in [0, 1]
    pub importance_score: f32,
    /// Semantic category of this key point
    pub category: KeyPointCategory,
}

// ── MultiDocSummaryOutput ─────────────────────────────────────────────────────

/// Output produced by the multi-document summarization pipeline
#[derive(Debug, Clone)]
pub struct MultiDocSummaryOutput {
    /// Final merged summary
    pub summary: String,
    /// Extracted key points across all documents
    pub key_points: Vec<ExtractedKeyPoint>,
    /// Per-document intermediate summaries (populated by MapReduce / Hierarchical)
    pub document_summaries: Vec<String>,
    /// Number of documents that were successfully processed
    pub num_docs_processed: usize,
    /// Number of documents skipped (too short, or duplicate)
    pub num_docs_skipped: usize,
    /// Total words processed across all retained documents
    pub total_words_processed: usize,
    /// Strategy that was actually used
    pub strategy_used: SummarizationStrategy,
    /// Formatted citation strings (empty when `include_citations` is false)
    pub citations: Vec<String>,
}

// ── SummarizationError ────────────────────────────────────────────────────────

/// Errors that can occur during multi-document summarization
#[derive(Debug, Clone)]
pub enum SummarizationError {
    /// No documents were provided
    NoDocuments,
    /// Input exceeded the configured document limit
    TooManyDocuments {
        /// Configured maximum
        max: usize,
        /// Actual document count
        got: usize,
    },
    /// A specific document is too short to summarize meaningfully
    DocumentTooShort {
        /// Zero-based document index
        index: usize,
        /// Minimum required word count
        min_words: usize,
    },
    /// The selected strategy encountered an internal error
    StrategyFailed(String),
}

impl fmt::Display for SummarizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SummarizationError::NoDocuments => write!(f, "no documents provided"),
            SummarizationError::TooManyDocuments { max, got } => {
                write!(f, "too many documents: max={}, got={}", max, got)
            },
            SummarizationError::DocumentTooShort { index, min_words } => write!(
                f,
                "document at index {} is too short (min {} words)",
                index, min_words
            ),
            SummarizationError::StrategyFailed(msg) => {
                write!(f, "summarization strategy failed: {}", msg)
            },
        }
    }
}

impl std::error::Error for SummarizationError {}

// ── MultiDocSummarizationPipeline ─────────────────────────────────────────────

/// Pipeline for summarizing multiple documents simultaneously
///
/// Supports four strategies:
/// - `MapReduce`: each document is summarized individually, then the
///   summaries are merged into a final summary
/// - `Hierarchical`: documents are grouped into levels; summaries at each
///   level feed the next, yielding a tree reduction
/// - `Extractive`: key sentences are scored across all documents and the
///   top sentences are assembled into a summary
/// - `Fusion`: all documents are concatenated into a single context for
///   holistic summarization
///
/// All algorithms are implemented in pure Rust without external LLM calls.
/// Actual inference is replaced with high-quality extractive summaries.
pub struct MultiDocSummarizationPipeline {
    config: MultiDocConfig,
}

impl MultiDocSummarizationPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: MultiDocConfig) -> Self {
        Self { config }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Summarize a collection of documents.
    pub fn summarize(
        &self,
        documents: Vec<InputDocument>,
    ) -> Result<MultiDocSummaryOutput, SummarizationError> {
        if documents.is_empty() {
            return Err(SummarizationError::NoDocuments);
        }
        if documents.len() > self.config.max_documents {
            return Err(SummarizationError::TooManyDocuments {
                max: self.config.max_documents,
                got: documents.len(),
            });
        }

        // Deduplicate if requested
        let (retained_docs, num_skipped) = if self.config.deduplicate_content {
            let kept_indices =
                Self::deduplicate_documents(&documents, self.config.similarity_threshold);
            let skipped = documents.len() - kept_indices.len();
            let retained: Vec<InputDocument> =
                kept_indices.iter().map(|&i| documents[i].clone()).collect();
            (retained, skipped)
        } else {
            let n = documents.len();
            (documents, 0_usize)
        };

        // Dispatch to the chosen strategy
        let strategy = self.config.strategy.clone();
        let mut output = match &strategy {
            SummarizationStrategy::MapReduce => self.map_reduce(&retained_docs)?,
            SummarizationStrategy::Hierarchical { levels } => {
                let lvls = *levels;
                self.hierarchical_summarize(&retained_docs, lvls)?
            },
            SummarizationStrategy::Extractive => self.extractive_summarize(&retained_docs)?,
            SummarizationStrategy::Fusion => self.fusion_summarize(&retained_docs)?,
        };

        output.num_docs_skipped = num_skipped;
        output.strategy_used = strategy;

        // Optionally append citations
        if self.config.include_citations {
            let citations: Vec<String> = retained_docs
                .iter()
                .enumerate()
                .map(|(i, doc)| Self::format_citation(doc, i, &self.config.citation_format))
                .collect();
            output.citations = citations;
        }

        Ok(output)
    }

    // ── Strategy implementations ──────────────────────────────────────────────

    /// Map-Reduce: summarize each doc, then merge.
    pub fn map_reduce(
        &self,
        docs: &[InputDocument],
    ) -> Result<MultiDocSummaryOutput, SummarizationError> {
        if docs.is_empty() {
            return Err(SummarizationError::NoDocuments);
        }

        let mut doc_summaries = Vec::with_capacity(docs.len());
        let mut total_words = 0usize;
        let mut key_points: Vec<ExtractedKeyPoint> = Vec::new();

        for (idx, doc) in docs.iter().enumerate() {
            let wc = Self::count_words(&doc.content);
            total_words += wc;

            let summary = Self::generate_per_doc_summary(
                doc,
                self.config.max_summary_length / docs.len().max(1),
            );
            doc_summaries.push(summary.clone());

            if self.config.extract_key_points {
                let sentences = Self::extract_key_sentences(&doc.content, 3);
                for (i, (sentence, score)) in sentences.into_iter().enumerate() {
                    let position_ratio = if wc > 0 { i as f32 / wc as f32 } else { 0.0 };
                    let importance = Self::score_key_point(&sentence, position_ratio);
                    let category = Self::classify_key_point(&sentence);
                    key_points.push(ExtractedKeyPoint {
                        text: sentence,
                        source_doc_index: idx,
                        importance_score: importance.max(score),
                        category,
                    });
                }
            }
        }

        let summary = Self::merge_summaries(&doc_summaries, self.config.max_summary_length);

        Ok(MultiDocSummaryOutput {
            summary,
            key_points,
            document_summaries: doc_summaries,
            num_docs_processed: docs.len(),
            num_docs_skipped: 0,
            total_words_processed: total_words,
            strategy_used: SummarizationStrategy::MapReduce,
            citations: Vec::new(),
        })
    }

    /// Hierarchical: multi-level tree reduction.
    pub fn hierarchical_summarize(
        &self,
        docs: &[InputDocument],
        levels: usize,
    ) -> Result<MultiDocSummaryOutput, SummarizationError> {
        if docs.is_empty() {
            return Err(SummarizationError::NoDocuments);
        }
        let levels = levels.max(1);

        let total_words: usize = docs.iter().map(|d| Self::count_words(&d.content)).sum();

        // Level 0: per-doc summaries
        let per_doc_len = self.config.max_summary_length / docs.len().max(1);
        let mut current_summaries: Vec<String> =
            docs.iter().map(|d| Self::generate_per_doc_summary(d, per_doc_len)).collect();

        let original_doc_summaries = current_summaries.clone();

        // Subsequent levels: group and merge
        for level in 1..levels {
            if current_summaries.len() <= 1 {
                break;
            }
            let group_size = (2_usize).pow(level as u32).min(current_summaries.len());
            let next_level_len =
                self.config.max_summary_length / (current_summaries.len() / group_size).max(1);

            let mut next_summaries = Vec::new();
            for chunk in current_summaries.chunks(group_size) {
                let merged = Self::merge_summaries(chunk, next_level_len);
                next_summaries.push(merged);
            }
            current_summaries = next_summaries;
        }

        let final_summary =
            Self::merge_summaries(&current_summaries, self.config.max_summary_length);

        let key_points = if self.config.extract_key_points {
            Self::extract_key_points_from_docs(docs, 2)
        } else {
            Vec::new()
        };

        Ok(MultiDocSummaryOutput {
            summary: final_summary,
            key_points,
            document_summaries: original_doc_summaries,
            num_docs_processed: docs.len(),
            num_docs_skipped: 0,
            total_words_processed: total_words,
            strategy_used: SummarizationStrategy::Hierarchical { levels },
            citations: Vec::new(),
        })
    }

    /// Extractive: score all sentences, pick top ones.
    pub fn extractive_summarize(
        &self,
        docs: &[InputDocument],
    ) -> Result<MultiDocSummaryOutput, SummarizationError> {
        if docs.is_empty() {
            return Err(SummarizationError::NoDocuments);
        }

        let total_words: usize = docs.iter().map(|d| Self::count_words(&d.content)).sum();

        // Collect all sentences with scores from all docs
        let mut scored_sentences: Vec<(String, f32, usize)> = Vec::new(); // (text, score, doc_idx)
        for (doc_idx, doc) in docs.iter().enumerate() {
            let wc = Self::count_words(&doc.content);
            let sentences = Self::extract_key_sentences(&doc.content, 5);
            for (rank, (sentence, raw_score)) in sentences.into_iter().enumerate() {
                let position_ratio = if wc > 0 { rank as f32 / wc as f32 } else { 0.5 };
                let score = Self::score_key_point(&sentence, position_ratio).max(raw_score);
                scored_sentences.push((sentence, score, doc_idx));
            }
        }

        // Sort by score descending
        scored_sentences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assemble summary from top sentences up to max_summary_length words
        let mut summary_words = 0usize;
        let mut summary_sentences: Vec<String> = Vec::new();
        let mut key_points: Vec<ExtractedKeyPoint> = Vec::new();

        for (sentence, score, doc_idx) in &scored_sentences {
            let wc = Self::count_words(sentence);
            if summary_words + wc > self.config.max_summary_length {
                continue;
            }
            summary_sentences.push(sentence.clone());
            summary_words += wc;

            if self.config.extract_key_points {
                let category = Self::classify_key_point(sentence);
                key_points.push(ExtractedKeyPoint {
                    text: sentence.clone(),
                    source_doc_index: *doc_idx,
                    importance_score: *score,
                    category,
                });
            }
        }

        let summary = summary_sentences.join(" ");
        let doc_summaries: Vec<String> = docs
            .iter()
            .map(|d| {
                Self::generate_per_doc_summary(
                    d,
                    self.config.max_summary_length / docs.len().max(1),
                )
            })
            .collect();

        Ok(MultiDocSummaryOutput {
            summary,
            key_points,
            document_summaries: doc_summaries,
            num_docs_processed: docs.len(),
            num_docs_skipped: 0,
            total_words_processed: total_words,
            strategy_used: SummarizationStrategy::Extractive,
            citations: Vec::new(),
        })
    }

    /// Fusion: concatenate all docs into a single context, then summarize.
    fn fusion_summarize(
        &self,
        docs: &[InputDocument],
    ) -> Result<MultiDocSummaryOutput, SummarizationError> {
        if docs.is_empty() {
            return Err(SummarizationError::NoDocuments);
        }

        let total_words: usize = docs.iter().map(|d| Self::count_words(&d.content)).sum();

        // Build a single fused document
        let fused_content: String = docs
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let fallback = format!("Document {}", i + 1);
                let title = doc.metadata.title.as_deref().unwrap_or(&fallback);
                format!("[{}]\n{}", title, doc.content)
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let fused_doc = InputDocument {
            content: fused_content,
            metadata: DocumentMetadata::default(),
        };

        let summary = Self::generate_per_doc_summary(&fused_doc, self.config.max_summary_length);

        let doc_summaries: Vec<String> = docs
            .iter()
            .map(|d| {
                Self::generate_per_doc_summary(
                    d,
                    self.config.max_summary_length / docs.len().max(1),
                )
            })
            .collect();

        let key_points = if self.config.extract_key_points {
            Self::extract_key_points_from_docs(docs, 2)
        } else {
            Vec::new()
        };

        Ok(MultiDocSummaryOutput {
            summary,
            key_points,
            document_summaries: doc_summaries,
            num_docs_processed: docs.len(),
            num_docs_skipped: 0,
            total_words_processed: total_words,
            strategy_used: SummarizationStrategy::Fusion,
            citations: Vec::new(),
        })
    }

    // ── Key sentence extraction ───────────────────────────────────────────────

    /// Extract the most important sentences from a text.
    ///
    /// Scoring factors:
    /// - Position: sentences near the beginning score higher
    /// - Length: medium-length sentences score higher than very short or very long
    /// - Keyword density: sentences containing frequent content words score higher
    ///
    /// Returns at most `max_count` `(sentence, score)` pairs.
    pub fn extract_key_sentences(text: &str, max_count: usize) -> Vec<(String, f32)> {
        if max_count == 0 {
            return Vec::new();
        }

        // Split into sentences on '.', '!', '?', ';' or newlines
        let sentences: Vec<String> = text
            .split(['.', '!', '?', ';', '\n'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.split_whitespace().count() >= 3)
            .collect();

        if sentences.is_empty() {
            return Vec::new();
        }

        let n = sentences.len();

        // Build word frequency table for keyword density scoring
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for sentence in &sentences {
            for word in sentence.split_whitespace() {
                let w = word.to_lowercase();
                let w = w.trim_matches(|c: char| !c.is_alphanumeric());
                if !w.is_empty() && w.len() > 3 {
                    *word_freq.entry(w.to_string()).or_insert(0) += 1;
                }
            }
        }

        let mut scored: Vec<(String, f32)> = sentences
            .into_iter()
            .enumerate()
            .map(|(i, sentence)| {
                let position_ratio = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
                let score = Self::score_sentence(&sentence, position_ratio, &word_freq, n);
                (sentence, score)
            })
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_count);
        scored
    }

    // ── Deduplication ─────────────────────────────────────────────────────────

    /// Return indices of documents to keep, removing near-duplicates.
    ///
    /// Documents are compared pairwise using word-level Jaccard similarity.
    /// When two documents exceed `threshold`, the one with the lower index is kept.
    pub fn deduplicate_documents(docs: &[InputDocument], threshold: f32) -> Vec<usize> {
        let n = docs.len();
        let mut removed: HashSet<usize> = HashSet::new();

        for i in 0..n {
            if removed.contains(&i) {
                continue;
            }
            for j in (i + 1)..n {
                if removed.contains(&j) {
                    continue;
                }
                let sim = Self::compute_jaccard(&docs[i].content, &docs[j].content);
                if sim >= threshold {
                    removed.insert(j);
                }
            }
        }

        (0..n).filter(|i| !removed.contains(i)).collect()
    }

    // ── Jaccard similarity ────────────────────────────────────────────────────

    /// Compute word-level Jaccard similarity between two texts.
    ///
    /// `J(A, B) = |A ∩ B| / |A ∪ B|`
    pub fn compute_jaccard(a: &str, b: &str) -> f32 {
        let words_a: HashSet<String> = a.split_whitespace().map(|w| w.to_lowercase()).collect();
        let words_b: HashSet<String> = b.split_whitespace().map(|w| w.to_lowercase()).collect();

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            1.0 // both empty → identical
        } else {
            intersection as f32 / union as f32
        }
    }

    // ── Word counting ─────────────────────────────────────────────────────────

    /// Count whitespace-separated words in a text.
    pub fn count_words(text: &str) -> usize {
        text.split_whitespace().count()
    }

    // ── Importance scoring ────────────────────────────────────────────────────

    /// Compute an importance score for a sentence given its relative position.
    ///
    /// Heuristics applied:
    /// - Sentences near the beginning of a document are more likely to be
    ///   important (journalistic inverted pyramid)
    /// - Medium length is preferred (neither telegraphic nor run-on)
    /// - Presence of numeric data is a signal of factual importance
    pub fn score_key_point(sentence: &str, position_ratio: f32) -> f32 {
        // Position score: decreasing linear from 1.0 (start) to 0.3 (end)
        let position_score = 1.0 - position_ratio * 0.7;

        // Length score: peak at ~15 words
        let wc = Self::count_words(sentence) as f32;
        let length_score = if wc < 3.0 {
            0.1
        } else if wc <= 20.0 {
            wc / 20.0
        } else {
            (40.0 - wc).max(0.0) / 20.0
        };

        // Numeric / factual signal
        let has_number = sentence.chars().any(|c| c.is_ascii_digit());
        let numeric_bonus = if has_number { 0.1 } else { 0.0 };

        // Keyword density bonus for common summary-worthy words
        let summary_keywords = [
            "important",
            "significant",
            "critical",
            "key",
            "major",
            "primary",
            "essential",
            "main",
            "result",
            "conclusion",
            "finding",
            "shows",
            "demonstrates",
            "indicates",
        ];
        let lower = sentence.to_lowercase();
        let keyword_bonus: f32 =
            summary_keywords.iter().filter(|&&kw| lower.contains(kw)).count() as f32 * 0.05;

        (position_score * 0.5 + length_score * 0.4 + numeric_bonus + keyword_bonus).min(1.0)
    }

    // ── Key point classification ──────────────────────────────────────────────

    /// Classify a sentence into a `KeyPointCategory` using keyword heuristics.
    pub fn classify_key_point(sentence: &str) -> KeyPointCategory {
        let lower = sentence.to_lowercase();

        // Warning signals
        if lower.contains("warning")
            || lower.contains("caution")
            || lower.contains("danger")
            || lower.contains("risk")
            || lower.contains("alert")
            || lower.contains("avoid")
            || lower.contains("never")
            || lower.contains("do not")
        {
            return KeyPointCategory::Warning;
        }

        // Action signals
        if lower.contains("should")
            || lower.contains("must")
            || lower.contains("need to")
            || lower.contains("recommend")
            || lower.contains("suggest")
            || lower.contains("require")
            || lower.starts_with("please")
            || lower.starts_with("ensure")
            || lower.starts_with("make sure")
            || lower.starts_with("use ")
            || lower.starts_with("run ")
            || lower.starts_with("check ")
        {
            return KeyPointCategory::Action;
        }

        // Definition signals
        if lower.contains(" is a ")
            || lower.contains(" are ")
            || lower.contains(" refers to")
            || lower.contains(" means ")
            || lower.contains(" defined as")
            || lower.contains(" is defined")
            || lower.contains(" is the")
            || lower.contains("is an ")
        {
            return KeyPointCategory::Definition;
        }

        // Opinion signals
        if lower.contains("believe")
            || lower.contains("think")
            || lower.contains("opinion")
            || lower.contains("argue")
            || lower.contains("claim")
            || lower.contains("suggest")
            || lower.starts_with("i ")
            || lower.starts_with("we ")
            || lower.contains("arguably")
            || lower.contains("perhaps")
            || lower.contains("seems")
        {
            return KeyPointCategory::Opinion;
        }

        // Default to Fact
        KeyPointCategory::Fact
    }

    // ── Citation formatting ───────────────────────────────────────────────────

    /// Format a citation for a document in the requested style.
    pub fn format_citation(doc: &InputDocument, index: usize, format: &CitationFormat) -> String {
        let title = doc.metadata.title.as_deref().unwrap_or("Untitled Document");
        let source = doc.metadata.source.as_deref().unwrap_or("Unknown Source");
        let date = doc.metadata.date.as_deref().unwrap_or("n.d.");

        match format {
            CitationFormat::Numeric => format!("[{}] {}", index + 1, title),
            CitationFormat::APA => format!("{} ({}). {}.", title, date, source),
            CitationFormat::Chicago => format!("{}. \"{}\". {}.", source, title, date),
            CitationFormat::Footnote => format!("{}. {} ({})", index + 1, title, date),
        }
    }

    // ── Per-document extractive summary ──────────────────────────────────────

    /// Generate a short extractive summary for a single document.
    ///
    /// In a production deployment this would call an LLM; here it uses the
    /// same extractive scoring logic that the pipeline uses internally.
    pub fn generate_per_doc_summary(doc: &InputDocument, max_len: usize) -> String {
        let max_len = max_len.max(10);
        let sentences = Self::extract_key_sentences(&doc.content, 5);

        let mut words_used = 0usize;
        let mut parts: Vec<String> = Vec::new();

        for (sentence, _) in sentences {
            let wc = Self::count_words(&sentence);
            if words_used + wc > max_len {
                break;
            }
            parts.push(sentence);
            words_used += wc;
        }

        if parts.is_empty() {
            // Fallback: take first `max_len` words of the raw content
            let words: Vec<&str> = doc.content.split_whitespace().take(max_len).collect();
            words.join(" ")
        } else {
            parts.join(". ")
        }
    }

    // ── Summary merging ───────────────────────────────────────────────────────

    /// Merge multiple summaries into a single cohesive summary.
    ///
    /// Selects sentences from each summary round-robin until `max_len` words
    /// are filled, preventing any single document from dominating.
    pub fn merge_summaries(summaries: &[String], max_len: usize) -> String {
        if summaries.is_empty() {
            return String::new();
        }
        if summaries.len() == 1 {
            let words: Vec<&str> = summaries[0].split_whitespace().take(max_len).collect();
            return words.join(" ");
        }

        // Collect sentences from each summary
        let per_summary_sentences: Vec<Vec<String>> = summaries
            .iter()
            .map(|s| {
                s.split(['.', '!', '?', ';'])
                    .map(|sent| sent.trim().to_string())
                    .filter(|sent| !sent.is_empty() && sent.split_whitespace().count() >= 3)
                    .collect()
            })
            .collect();

        // Round-robin interleaving
        let mut result_sentences: Vec<String> = Vec::new();
        let mut words_used = 0usize;
        let mut idx: Vec<usize> = vec![0; per_summary_sentences.len()];
        let mut progress = true;

        while progress && words_used < max_len {
            progress = false;
            for (i, sentences) in per_summary_sentences.iter().enumerate() {
                if idx[i] >= sentences.len() {
                    continue;
                }
                let sentence = &sentences[idx[i]];
                let wc = Self::count_words(sentence);
                if words_used + wc <= max_len {
                    result_sentences.push(sentence.clone());
                    words_used += wc;
                    idx[i] += 1;
                    progress = true;
                }
            }
        }

        if result_sentences.is_empty() {
            // Fallback: concatenate and truncate
            let combined = summaries.join(" ");
            let words: Vec<&str> = combined.split_whitespace().take(max_len).collect();
            words.join(" ")
        } else {
            result_sentences.join(". ")
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Score a sentence for key-sentence extraction.
    fn score_sentence(
        sentence: &str,
        position_ratio: f32,
        word_freq: &HashMap<String, usize>,
        total_sentences: usize,
    ) -> f32 {
        let position_score = if total_sentences > 1 {
            // First and last sentences often more important
            if !(0.15..=0.85).contains(&position_ratio) {
                0.9
            } else {
                1.0 - position_ratio * 0.6
            }
        } else {
            1.0
        };

        let words: Vec<&str> = sentence.split_whitespace().collect();
        let wc = words.len() as f32;

        // Length score: peak at ~12-18 words
        let length_score = if wc < 4.0 {
            0.2
        } else if wc <= 18.0 {
            wc / 18.0
        } else {
            (36.0 - wc).max(0.0) / 18.0
        };

        // Keyword density: average freq of content words
        let density_score = if words.is_empty() {
            0.0
        } else {
            let total_freq: usize = words
                .iter()
                .map(|w| {
                    let w = w.to_lowercase();
                    let w = w.trim_matches(|c: char| !c.is_alphanumeric());
                    word_freq.get(w).copied().unwrap_or(0)
                })
                .sum();
            let max_possible = words.len() * total_sentences;
            if max_possible > 0 {
                (total_freq as f32 / max_possible as f32).min(1.0)
            } else {
                0.0
            }
        };

        (position_score * 0.4 + length_score * 0.35 + density_score * 0.25).min(1.0)
    }

    /// Extract key points from multiple documents.
    fn extract_key_points_from_docs(
        docs: &[InputDocument],
        per_doc: usize,
    ) -> Vec<ExtractedKeyPoint> {
        let mut key_points = Vec::new();
        for (doc_idx, doc) in docs.iter().enumerate() {
            let wc = Self::count_words(&doc.content);
            let sentences = Self::extract_key_sentences(&doc.content, per_doc);
            for (rank, (sentence, raw_score)) in sentences.into_iter().enumerate() {
                let position_ratio = if wc > 0 { rank as f32 / wc as f32 } else { 0.0 };
                let importance = Self::score_key_point(&sentence, position_ratio).max(raw_score);
                let category = Self::classify_key_point(&sentence);
                key_points.push(ExtractedKeyPoint {
                    text: sentence,
                    source_doc_index: doc_idx,
                    importance_score: importance,
                    category,
                });
            }
        }
        key_points
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_doc(content: &str) -> InputDocument {
        InputDocument::new(content)
    }

    fn make_titled_doc(content: &str, title: &str) -> InputDocument {
        InputDocument::with_title(content, title)
    }

    // ── Word counting ─────────────────────────────────────────────────────────

    #[test]
    fn test_count_words_basic() {
        assert_eq!(MultiDocSummarizationPipeline::count_words("hello world"), 2);
        assert_eq!(
            MultiDocSummarizationPipeline::count_words("  leading and trailing  "),
            3
        );
        assert_eq!(MultiDocSummarizationPipeline::count_words(""), 0);
    }

    #[test]
    fn test_count_words_multiline() {
        let text = "line one\nline two\nline three";
        assert_eq!(MultiDocSummarizationPipeline::count_words(text), 6);
    }

    // ── Jaccard similarity ────────────────────────────────────────────────────

    #[test]
    fn test_jaccard_identical() {
        let sim = MultiDocSummarizationPipeline::compute_jaccard("hello world", "hello world");
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let sim = MultiDocSummarizationPipeline::compute_jaccard("apple banana", "cherry date");
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        // "the cat sat" vs "the dog sat" → intersection {the, sat}=2, union {the,cat,sat,dog}=4
        let sim = MultiDocSummarizationPipeline::compute_jaccard("the cat sat", "the dog sat");
        assert!((sim - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_both_empty() {
        let sim = MultiDocSummarizationPipeline::compute_jaccard("", "");
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_case_insensitive() {
        let sim = MultiDocSummarizationPipeline::compute_jaccard("Hello World", "hello world");
        assert!((sim - 1.0).abs() < 1e-6);
    }

    // ── Deduplication ─────────────────────────────────────────────────────────

    #[test]
    fn test_deduplicate_keeps_unique() {
        let docs = vec![
            make_doc("The quick brown fox jumps over the lazy dog"),
            make_doc("Machine learning is a subset of artificial intelligence"),
        ];
        let kept = MultiDocSummarizationPipeline::deduplicate_documents(&docs, 0.85);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_deduplicate_removes_near_duplicate() {
        let docs = vec![
            make_doc("The quick brown fox jumps over the lazy dog"),
            make_doc("The quick brown fox jumps over the lazy dog"),
        ];
        let kept = MultiDocSummarizationPipeline::deduplicate_documents(&docs, 0.85);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0], 0);
    }

    #[test]
    fn test_deduplicate_threshold_boundary() {
        // Slightly different documents should be kept at a strict threshold
        let docs = vec![
            make_doc("apple banana cherry date elderberry fig grape"),
            make_doc("apple banana cherry date elderberry fig grape honeydew"),
        ];
        let kept = MultiDocSummarizationPipeline::deduplicate_documents(&docs, 0.99);
        assert_eq!(kept.len(), 2); // Different enough at 0.99 threshold
    }

    // ── Key sentence extraction ───────────────────────────────────────────────

    #[test]
    fn test_extract_key_sentences_returns_expected_count() {
        let text = "This is the first sentence about important findings. The second sentence shows results. The third sentence provides context about the study. The fourth sentence gives background information. The fifth sentence concludes the analysis.";
        let sentences = MultiDocSummarizationPipeline::extract_key_sentences(text, 3);
        assert!(!sentences.is_empty());
        assert!(sentences.len() <= 3);
    }

    #[test]
    fn test_extract_key_sentences_empty_text() {
        let sentences = MultiDocSummarizationPipeline::extract_key_sentences("", 5);
        assert!(sentences.is_empty());
    }

    #[test]
    fn test_extract_key_sentences_max_zero() {
        let text = "First sentence here. Second sentence here.";
        let sentences = MultiDocSummarizationPipeline::extract_key_sentences(text, 0);
        assert!(sentences.is_empty());
    }

    #[test]
    fn test_extract_key_sentences_scores_non_negative() {
        let text = "The study demonstrates significant findings in the field. Results indicate critical improvements over baseline methods. This research shows major advances in the domain.";
        let sentences = MultiDocSummarizationPipeline::extract_key_sentences(text, 5);
        for (_, score) in &sentences {
            assert!(*score >= 0.0, "score should be non-negative");
            assert!(*score <= 1.0, "score should be at most 1.0");
        }
    }

    // ── Key point classification ──────────────────────────────────────────────

    #[test]
    fn test_classify_key_point_warning() {
        let cat = MultiDocSummarizationPipeline::classify_key_point(
            "Warning: this operation is dangerous",
        );
        assert_eq!(cat, KeyPointCategory::Warning);
    }

    #[test]
    fn test_classify_key_point_action() {
        let cat = MultiDocSummarizationPipeline::classify_key_point(
            "You should update your dependencies regularly",
        );
        assert_eq!(cat, KeyPointCategory::Action);
    }

    #[test]
    fn test_classify_key_point_definition() {
        let cat = MultiDocSummarizationPipeline::classify_key_point(
            "Machine learning is a subset of artificial intelligence",
        );
        assert_eq!(cat, KeyPointCategory::Definition);
    }

    #[test]
    fn test_classify_key_point_opinion() {
        let cat =
            MultiDocSummarizationPipeline::classify_key_point("I believe this approach is better");
        assert_eq!(cat, KeyPointCategory::Opinion);
    }

    #[test]
    fn test_classify_key_point_fact() {
        let cat = MultiDocSummarizationPipeline::classify_key_point(
            "The experiment ran for 42 hours and produced 1000 data points",
        );
        assert_eq!(cat, KeyPointCategory::Fact);
    }

    // ── Citation formatting ───────────────────────────────────────────────────

    #[test]
    fn test_format_citation_numeric() {
        let doc = make_titled_doc("content", "My Paper");
        let citation =
            MultiDocSummarizationPipeline::format_citation(&doc, 0, &CitationFormat::Numeric);
        assert!(citation.contains("[1]"));
        assert!(citation.contains("My Paper"));
    }

    #[test]
    fn test_format_citation_apa() {
        let mut doc = make_titled_doc("content", "Study Results");
        doc.metadata.date = Some("2024".to_string());
        doc.metadata.source = Some("Journal of Science".to_string());
        let citation =
            MultiDocSummarizationPipeline::format_citation(&doc, 0, &CitationFormat::APA);
        assert!(citation.contains("2024"));
        assert!(citation.contains("Study Results"));
    }

    #[test]
    fn test_format_citation_chicago() {
        let mut doc = make_titled_doc("content", "Historical Analysis");
        doc.metadata.source = Some("Oxford University Press".to_string());
        let citation =
            MultiDocSummarizationPipeline::format_citation(&doc, 2, &CitationFormat::Chicago);
        assert!(citation.contains("Historical Analysis"));
        assert!(citation.contains("Oxford University Press"));
    }

    // ── Merge summaries ───────────────────────────────────────────────────────

    #[test]
    fn test_merge_summaries_single() {
        let summaries = vec![
            "This is a single summary sentence about important research findings.".to_string(),
        ];
        let merged = MultiDocSummarizationPipeline::merge_summaries(&summaries, 100);
        assert!(!merged.is_empty());
    }

    #[test]
    fn test_merge_summaries_multiple() {
        let summaries = vec![
            "The first document discusses machine learning applications in healthcare. The results show improvements.".to_string(),
            "The second document covers natural language processing research. The methods are novel.".to_string(),
        ];
        let merged = MultiDocSummarizationPipeline::merge_summaries(&summaries, 50);
        assert!(!merged.is_empty());
        let word_count = MultiDocSummarizationPipeline::count_words(&merged);
        assert!(word_count <= 55, "merged should respect max length roughly");
    }

    #[test]
    fn test_merge_summaries_empty() {
        let merged = MultiDocSummarizationPipeline::merge_summaries(&[], 100);
        assert!(merged.is_empty());
    }

    // ── Strategies ────────────────────────────────────────────────────────────

    #[test]
    fn test_map_reduce_strategy() {
        let pipeline = MultiDocSummarizationPipeline::new(MultiDocConfig::default());
        let docs = vec![
            make_doc("Machine learning is a method of data analysis that automates analytical model building. It is based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."),
            make_doc("Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised."),
        ];
        let result = pipeline.summarize(docs);
        assert!(result.is_ok());
        let out = result.expect("output");
        assert!(!out.summary.is_empty());
        assert_eq!(out.num_docs_processed, 2);
    }

    #[test]
    fn test_hierarchical_strategy() {
        let mut cfg = MultiDocConfig::default();
        cfg.strategy = SummarizationStrategy::Hierarchical { levels: 2 };
        let pipeline = MultiDocSummarizationPipeline::new(cfg);
        let docs = vec![
            make_doc("Artificial intelligence encompasses machine learning and deep learning technologies that enable computers to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation."),
            make_doc("Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data."),
            make_doc("Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do."),
        ];
        let result = pipeline.summarize(docs);
        assert!(result.is_ok());
        let out = result.expect("output");
        assert!(!out.summary.is_empty());
        assert_eq!(out.num_docs_processed, 3);
    }

    #[test]
    fn test_extractive_strategy() {
        let mut cfg = MultiDocConfig::default();
        cfg.strategy = SummarizationStrategy::Extractive;
        let pipeline = MultiDocSummarizationPipeline::new(cfg);
        let docs = vec![
            make_doc("The new algorithm demonstrates significant improvements in performance. The results indicate a 25% speedup over previous methods. This represents an important finding for the research community."),
            make_doc("Experimental evaluation shows the method scales well to large datasets. Critical analysis reveals consistent gains across different benchmarks. The approach requires minimal hyperparameter tuning."),
        ];
        let result = pipeline.summarize(docs);
        assert!(result.is_ok());
        let out = result.expect("output");
        assert!(!out.summary.is_empty());
    }

    #[test]
    fn test_fusion_strategy() {
        let mut cfg = MultiDocConfig::default();
        cfg.strategy = SummarizationStrategy::Fusion;
        let pipeline = MultiDocSummarizationPipeline::new(cfg);
        let docs = vec![
            make_titled_doc("The primary study examines climate patterns over the past century. The data shows significant warming trends across all major regions.", "Climate Study"),
            make_titled_doc("Secondary analysis of ocean temperatures confirms the global trends. Marine ecosystems are responding to these temperature changes.", "Ocean Analysis"),
        ];
        let result = pipeline.summarize(docs);
        assert!(result.is_ok());
    }

    // ── Error conditions ──────────────────────────────────────────────────────

    #[test]
    fn test_error_no_documents() {
        let pipeline = MultiDocSummarizationPipeline::new(MultiDocConfig::default());
        let result = pipeline.summarize(vec![]);
        assert!(matches!(result, Err(SummarizationError::NoDocuments)));
    }

    #[test]
    fn test_error_too_many_documents() {
        let mut cfg = MultiDocConfig::default();
        cfg.max_documents = 2;
        let pipeline = MultiDocSummarizationPipeline::new(cfg);
        let docs = vec![
            make_doc("Doc one content here"),
            make_doc("Doc two content here"),
            make_doc("Doc three content here"),
        ];
        let result = pipeline.summarize(docs);
        assert!(matches!(
            result,
            Err(SummarizationError::TooManyDocuments { max: 2, got: 3 })
        ));
    }

    #[test]
    fn test_error_display() {
        assert!(SummarizationError::NoDocuments.to_string().contains("no documents"));
        assert!(SummarizationError::TooManyDocuments { max: 5, got: 10 }
            .to_string()
            .contains("10"));
        assert!(SummarizationError::DocumentTooShort {
            index: 2,
            min_words: 20
        }
        .to_string()
        .contains("index 2"));
        assert!(SummarizationError::StrategyFailed("test".to_string())
            .to_string()
            .contains("test"));
    }

    // ── Score key point ───────────────────────────────────────────────────────

    #[test]
    fn test_score_key_point_range() {
        let score = MultiDocSummarizationPipeline::score_key_point(
            "The experiment demonstrated significant results in 2024",
            0.0,
        );
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_score_key_point_beginning_vs_end() {
        let sentence = "The study shows important findings about the research topic";
        let score_start = MultiDocSummarizationPipeline::score_key_point(sentence, 0.0);
        let score_end = MultiDocSummarizationPipeline::score_key_point(sentence, 0.9);
        assert!(
            score_start > score_end,
            "sentences at start should score higher than at end"
        );
    }

    // ── Strategy display ──────────────────────────────────────────────────────

    #[test]
    fn test_strategy_display() {
        assert_eq!(SummarizationStrategy::MapReduce.to_string(), "map_reduce");
        assert_eq!(SummarizationStrategy::Extractive.to_string(), "extractive");
        assert_eq!(SummarizationStrategy::Fusion.to_string(), "fusion");
        assert_eq!(
            SummarizationStrategy::Hierarchical { levels: 3 }.to_string(),
            "hierarchical(levels=3)"
        );
    }

    // ── With deduplication enabled ────────────────────────────────────────────

    #[test]
    fn test_summarize_with_deduplication() {
        let mut cfg = MultiDocConfig::default();
        cfg.deduplicate_content = true;
        cfg.similarity_threshold = 0.85;
        let pipeline = MultiDocSummarizationPipeline::new(cfg);
        let docs = vec![
            make_doc("Artificial intelligence is transforming the healthcare industry through better diagnostics and treatment recommendations."),
            make_doc("Artificial intelligence is transforming the healthcare industry through better diagnostics and treatment recommendations."),
            make_doc("Quantum computing promises to solve complex optimization problems that are intractable for classical computers."),
        ];
        let result = pipeline.summarize(docs);
        assert!(result.is_ok());
        let out = result.expect("output");
        // One duplicate should be removed
        assert_eq!(out.num_docs_processed + out.num_docs_skipped, 3);
    }

    // ── Citation inclusion ────────────────────────────────────────────────────

    #[test]
    fn test_summarize_with_citations() {
        let mut cfg = MultiDocConfig::default();
        cfg.include_citations = true;
        cfg.citation_format = CitationFormat::Numeric;
        let pipeline = MultiDocSummarizationPipeline::new(cfg);
        let docs = vec![
            make_titled_doc("First document contains research findings about climate change and its global impact on ecosystems.", "Climate Report"),
            make_titled_doc("Second document presents economic analysis of renewable energy adoption rates worldwide.", "Energy Report"),
        ];
        let result = pipeline.summarize(docs);
        assert!(result.is_ok());
        let out = result.expect("output");
        assert!(!out.citations.is_empty());
        assert!(out.citations[0].contains("[1]"));
    }
}
