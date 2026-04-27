//! RAG (Retrieval Augmented Generation) pipeline
//!
//! Provides TF-IDF and BM25 based document retrieval with a full RAG pipeline
//! for augmenting generation with retrieved context.

use std::collections::HashMap;

// ── Document ─────────────────────────────────────────────────────────────────

/// A document in the knowledge base
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

impl Document {
    /// Create a new document with the given id and content.
    pub fn new(id: &str, content: &str) -> Self {
        Self {
            id: id.to_string(),
            content: content.to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Attach a metadata key-value pair (builder style).
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Number of whitespace-separated words in the document.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }

    /// Split the document into overlapping chunks of approximately `chunk_size` words.
    ///
    /// Each successive chunk starts `chunk_size - overlap` words after the previous one.
    /// The last chunk covers all remaining words even if it is shorter than `chunk_size`.
    pub fn chunk(&self, chunk_size: usize, overlap: usize) -> Vec<DocumentChunk> {
        let chunk_size = chunk_size.max(1);
        let overlap = overlap.min(chunk_size.saturating_sub(1));
        let step = chunk_size - overlap;

        let words: Vec<&str> = self.content.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut chunk_idx = 0usize;

        loop {
            let end = (start + chunk_size).min(words.len());
            let chunk_content = words[start..end].join(" ");
            chunks.push(DocumentChunk {
                doc_id: self.id.clone(),
                chunk_idx,
                content: chunk_content,
                start_word: start,
                end_word: end,
            });
            chunk_idx += 1;
            if end >= words.len() {
                break;
            }
            start += step;
        }

        chunks
    }
}

// ── DocumentChunk ─────────────────────────────────────────────────────────────

/// A chunk of a document suitable for indexing.
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub doc_id: String,
    pub chunk_idx: usize,
    pub content: String,
    pub start_word: usize,
    pub end_word: usize,
}

impl DocumentChunk {
    /// Number of whitespace-separated words in the chunk.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

// ── RetrievalResult ───────────────────────────────────────────────────────────

/// A retrieval result pairing a document chunk with its similarity score.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub chunk: DocumentChunk,
    pub score: f32,
    pub rank: usize,
}

// ── TfIdfRetriever ────────────────────────────────────────────────────────────

/// TF-IDF based document retriever with cosine similarity search.
///
/// Chunk vectors are L2-normalised so cosine similarity equals the dot product.
pub struct TfIdfRetriever {
    /// vocabulary: word → column index in the TF-IDF matrix
    vocab: HashMap<String, usize>,
    /// Smoothed IDF score per vocabulary term
    idf: Vec<f32>,
    /// Sparse unit-length TF-IDF vectors per indexed chunk
    chunk_vectors: Vec<HashMap<usize, f32>>,
    /// The indexed chunks (parallel to `chunk_vectors`)
    chunks: Vec<DocumentChunk>,
}

impl TfIdfRetriever {
    /// Create an empty retriever.
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            idf: Vec::new(),
            chunk_vectors: Vec::new(),
            chunks: Vec::new(),
        }
    }

    /// Index a batch of documents.
    ///
    /// Each document is split into overlapping chunks of `chunk_size` words with
    /// `overlap` words of overlap.  Returns the total number of chunks indexed.
    pub fn index(
        &mut self,
        documents: &[Document],
        chunk_size: usize,
        overlap: usize,
    ) -> Result<usize, RagError> {
        if documents.is_empty() {
            return Err(RagError::EmptyDocuments);
        }

        // 1. Chunk all documents
        let all_chunks: Vec<DocumentChunk> =
            documents.iter().flat_map(|doc| doc.chunk(chunk_size, overlap)).collect();

        if all_chunks.is_empty() {
            return Err(RagError::IndexingFailed(
                "no chunks produced from documents".to_string(),
            ));
        }

        let n = all_chunks.len();

        // 2. Build vocabulary and raw TF tables
        //    tf_raw[chunk_idx][word_idx] = raw count
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut tf_raw: Vec<HashMap<usize, u32>> = Vec::with_capacity(n);

        for chunk in &all_chunks {
            let tokens = Self::tokenize(&chunk.content);
            let mut counts: HashMap<usize, u32> = HashMap::new();
            for token in &tokens {
                // Assign a new index if the token is unseen
                let next_id = vocab.len();
                let idx = *vocab.entry(token.clone()).or_insert(next_id);
                *counts.entry(idx).or_insert(0) += 1;
            }
            tf_raw.push(counts);
        }

        let vocab_size = vocab.len();

        // 3. Document frequency per term
        let mut df: Vec<usize> = vec![0usize; vocab_size];
        for chunk_counts in &tf_raw {
            for &term_idx in chunk_counts.keys() {
                if term_idx < vocab_size {
                    df[term_idx] += 1;
                }
            }
        }

        // 4. Smoothed IDF: idf[t] = log((N + 1) / (df[t] + 1)) + 1
        let n_f = n as f32;
        let idf: Vec<f32> =
            df.iter().map(|&d| ((n_f + 1.0) / (d as f32 + 1.0)).ln() + 1.0).collect();

        // 5. Compute TF-IDF and L2-normalise each chunk vector
        let mut chunk_vectors: Vec<HashMap<usize, f32>> = Vec::with_capacity(n);

        for chunk_counts in &tf_raw {
            let total_words: u32 = chunk_counts.values().sum();
            let total_words_f = total_words.max(1) as f32;

            let mut vec: HashMap<usize, f32> = HashMap::new();
            for (&term_idx, &count) in chunk_counts {
                let tf = count as f32 / total_words_f;
                let tfidf = tf * idf[term_idx];
                if tfidf > 0.0 {
                    vec.insert(term_idx, tfidf);
                }
            }

            // L2-normalise
            let norm: f32 = vec.values().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in vec.values_mut() {
                    *v /= norm;
                }
            }
            chunk_vectors.push(vec);
        }

        self.vocab = vocab;
        self.idf = idf;
        self.chunk_vectors = chunk_vectors;
        self.chunks = all_chunks;

        Ok(n)
    }

    /// Retrieve the top-k most relevant chunks for `query`.
    pub fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<RetrievalResult>, RagError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(RagError::EmptyQuery);
        }
        if self.chunks.is_empty() {
            return Err(RagError::NotIndexed);
        }

        // 1. Tokenise and build sparse query TF-IDF vector (unit-length)
        let tokens = Self::tokenize(query);
        let total = tokens.len().max(1) as f32;
        let mut raw: HashMap<usize, f32> = HashMap::new();
        for token in &tokens {
            if let Some(&idx) = self.vocab.get(token) {
                *raw.entry(idx).or_insert(0.0) += 1.0 / total;
            }
        }
        // Apply IDF
        for (&idx, v) in raw.iter_mut() {
            *v *= self.idf[idx];
        }
        // L2-normalise
        let norm: f32 = raw.values().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in raw.values_mut() {
                *v /= norm;
            }
        }

        // 2. Score every chunk
        let mut scores: Vec<(usize, f32)> = self
            .chunk_vectors
            .iter()
            .enumerate()
            .map(|(i, cv)| (i, Self::cosine_similarity(&raw, cv)))
            .collect();

        // 3. Sort descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Take top-k
        let results = scores
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(rank, (idx, score))| RetrievalResult {
                chunk: self.chunks[idx].clone(),
                score,
                rank,
            })
            .collect();

        Ok(results)
    }

    /// Number of indexed chunks.
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Vocabulary size after indexing.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Tokenise `text` by lower-casing and splitting on non-alphanumeric characters.
    /// Single-character tokens are discarded.
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .map(|s| s.to_string())
            .collect()
    }

    /// Cosine similarity between two *already L2-normalised* sparse vectors.
    /// Since both vectors have unit norm, cosine similarity equals the dot product.
    fn cosine_similarity(a: &HashMap<usize, f32>, b: &HashMap<usize, f32>) -> f32 {
        // Iterate over the smaller map for efficiency
        let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        small.iter().filter_map(|(k, va)| large.get(k).map(|vb| va * vb)).sum()
    }
}

impl Default for TfIdfRetriever {
    fn default() -> Self {
        Self::new()
    }
}

// ── Bm25Retriever ─────────────────────────────────────────────────────────────

/// BM25 (Okapi BM25) probabilistic relevance retriever.
pub struct Bm25Retriever {
    /// Term saturation parameter (default 1.5)
    k1: f32,
    /// Length normalisation parameter (default 0.75)
    b: f32,
    vocab: HashMap<String, usize>,
    /// Document frequency per term index
    df: Vec<usize>,
    /// Term frequencies per chunk: chunk_id → (term_id → count)
    tf: Vec<HashMap<usize, u32>>,
    chunks: Vec<DocumentChunk>,
    avg_chunk_len: f32,
    num_chunks: usize,
}

impl Bm25Retriever {
    /// Create a BM25 retriever with explicit k1 and b parameters.
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            vocab: HashMap::new(),
            df: Vec::new(),
            tf: Vec::new(),
            chunks: Vec::new(),
            avg_chunk_len: 0.0,
            num_chunks: 0,
        }
    }

    /// Create a BM25 retriever with the standard default parameters (k1=1.5, b=0.75).
    pub fn default() -> Self {
        Self::new(1.5, 0.75)
    }

    /// Index `documents`, chunking each by `chunk_size` words with `overlap` overlap.
    pub fn index(
        &mut self,
        documents: &[Document],
        chunk_size: usize,
        overlap: usize,
    ) -> Result<usize, RagError> {
        if documents.is_empty() {
            return Err(RagError::EmptyDocuments);
        }

        let all_chunks: Vec<DocumentChunk> =
            documents.iter().flat_map(|doc| doc.chunk(chunk_size, overlap)).collect();

        if all_chunks.is_empty() {
            return Err(RagError::IndexingFailed(
                "no chunks produced from documents".to_string(),
            ));
        }

        let n = all_chunks.len();
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut tf_raw: Vec<HashMap<usize, u32>> = Vec::with_capacity(n);
        let mut total_words: usize = 0;

        for chunk in &all_chunks {
            let tokens = Self::tokenize(&chunk.content);
            total_words += tokens.len();
            let mut counts: HashMap<usize, u32> = HashMap::new();
            for token in &tokens {
                let next_id = vocab.len();
                let idx = *vocab.entry(token.clone()).or_insert(next_id);
                *counts.entry(idx).or_insert(0) += 1;
            }
            tf_raw.push(counts);
        }

        let vocab_size = vocab.len();
        let mut df: Vec<usize> = vec![0usize; vocab_size];
        for chunk_counts in &tf_raw {
            for &term_idx in chunk_counts.keys() {
                if term_idx < vocab_size {
                    df[term_idx] += 1;
                }
            }
        }

        self.avg_chunk_len = total_words as f32 / n as f32;
        self.vocab = vocab;
        self.df = df;
        self.tf = tf_raw;
        self.chunks = all_chunks;
        self.num_chunks = n;

        Ok(n)
    }

    /// Retrieve the top-k most relevant chunks for `query` using BM25 scoring.
    pub fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<RetrievalResult>, RagError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(RagError::EmptyQuery);
        }
        if self.num_chunks == 0 {
            return Err(RagError::NotIndexed);
        }

        let query_tokens = Self::tokenize(query);
        let n_f = self.num_chunks as f32;

        // Score each chunk
        let mut scores: Vec<(usize, f32)> = (0..self.num_chunks)
            .map(|chunk_idx| {
                let chunk_tf = &self.tf[chunk_idx];
                let chunk_len = chunk_tf.values().map(|&c| c as usize).sum::<usize>() as f32;

                let score: f32 = query_tokens
                    .iter()
                    .filter_map(|token| self.vocab.get(token))
                    .map(|&term_idx| {
                        let df_t = self.df.get(term_idx).copied().unwrap_or(0) as f32;
                        // Robertson–Spärck Jones IDF with smoothing
                        let idf = ((n_f - df_t + 0.5) / (df_t + 0.5) + 1.0).ln();
                        let tf_td = chunk_tf.get(&term_idx).copied().unwrap_or(0) as f32;
                        let numerator = tf_td * (self.k1 + 1.0);
                        let denominator = tf_td
                            + self.k1
                                * (1.0 - self.b + self.b * chunk_len / self.avg_chunk_len.max(1.0));
                        idf * numerator / denominator.max(f32::EPSILON)
                    })
                    .sum();

                (chunk_idx, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let results = scores
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(rank, (idx, score))| RetrievalResult {
                chunk: self.chunks[idx].clone(),
                score,
                rank,
            })
            .collect();

        Ok(results)
    }

    /// Number of indexed chunks.
    pub fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .map(|s| s.to_string())
            .collect()
    }
}

// ── RagConfig ─────────────────────────────────────────────────────────────────

/// Strategy used when retrieving relevant documents.
#[derive(Debug, Clone, PartialEq)]
pub enum RetrievalStrategy {
    TfIdf,
    Bm25,
}

/// Configuration for the RAG pipeline.
pub struct RagConfig {
    pub retrieval_strategy: RetrievalStrategy,
    pub top_k: usize,
    /// Number of words per document chunk
    pub chunk_size: usize,
    /// Word overlap between consecutive chunks
    pub chunk_overlap: usize,
    /// Maximum number of tokens in the augmented context
    pub max_context_length: usize,
    /// Template used to build the augmented prompt.
    /// `{context}` and `{query}` are replaced at runtime.
    pub context_template: String,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            retrieval_strategy: RetrievalStrategy::Bm25,
            top_k: 3,
            chunk_size: 200,
            chunk_overlap: 50,
            max_context_length: 2048,
            context_template: "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:".to_string(),
        }
    }
}

// ── RagPipeline ───────────────────────────────────────────────────────────────

/// RAG pipeline that combines retrieval with prompt augmentation.
pub struct RagPipeline {
    pub config: RagConfig,
    tfidf_retriever: Option<TfIdfRetriever>,
    bm25_retriever: Option<Bm25Retriever>,
    indexed: bool,
}

impl RagPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: RagConfig) -> Self {
        Self {
            config,
            tfidf_retriever: None,
            bm25_retriever: None,
            indexed: false,
        }
    }

    /// Index a knowledge base of documents.
    ///
    /// Returns the total number of document chunks indexed.
    pub fn index(&mut self, documents: &[Document]) -> Result<usize, RagError> {
        let chunk_size = self.config.chunk_size;
        let overlap = self.config.chunk_overlap;

        let n = match self.config.retrieval_strategy {
            RetrievalStrategy::TfIdf => {
                let mut retriever = TfIdfRetriever::new();
                let n = retriever.index(documents, chunk_size, overlap)?;
                self.tfidf_retriever = Some(retriever);
                n
            },
            RetrievalStrategy::Bm25 => {
                let mut retriever = Bm25Retriever::new(1.5, 0.75);
                let n = retriever.index(documents, chunk_size, overlap)?;
                self.bm25_retriever = Some(retriever);
                n
            },
        };

        self.indexed = true;
        Ok(n)
    }

    /// Retrieve the top-k most relevant chunks for `query`.
    pub fn retrieve(&self, query: &str) -> Result<Vec<RetrievalResult>, RagError> {
        if !self.indexed {
            return Err(RagError::NotIndexed);
        }
        match self.config.retrieval_strategy {
            RetrievalStrategy::TfIdf => {
                let retriever = self.tfidf_retriever.as_ref().ok_or(RagError::NotIndexed)?;
                retriever.retrieve(query, self.config.top_k)
            },
            RetrievalStrategy::Bm25 => {
                let retriever = self.bm25_retriever.as_ref().ok_or(RagError::NotIndexed)?;
                retriever.retrieve(query, self.config.top_k)
            },
        }
    }

    /// Build an augmented prompt from retrieved context chunks and the original query.
    pub fn build_prompt(&self, query: &str, results: &[RetrievalResult]) -> String {
        let context = results
            .iter()
            .map(|r| format!("[{}] {}", r.chunk.doc_id, r.chunk.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        self.config
            .context_template
            .replace("{context}", &context)
            .replace("{query}", query)
    }

    /// Run the full RAG pipeline: retrieve relevant chunks and build an augmented prompt.
    pub fn run(&self, query: &str) -> Result<RagResult, RagError> {
        let results = self.retrieve(query)?;
        let prompt = self.build_prompt(query, &results);
        Ok(RagResult {
            query: query.to_string(),
            retrieved_chunks: results,
            augmented_prompt: prompt,
        })
    }

    /// Whether the pipeline has been indexed.
    pub fn is_indexed(&self) -> bool {
        self.indexed
    }

    /// Total number of indexed document chunks.
    pub fn num_indexed_chunks(&self) -> usize {
        match self.config.retrieval_strategy {
            RetrievalStrategy::TfIdf => self.tfidf_retriever.as_ref().map_or(0, |r| r.num_chunks()),
            RetrievalStrategy::Bm25 => self.bm25_retriever.as_ref().map_or(0, |r| r.num_chunks()),
        }
    }
}

// ── RagResult ─────────────────────────────────────────────────────────────────

/// The output produced by a RAG pipeline run.
pub struct RagResult {
    pub query: String,
    pub retrieved_chunks: Vec<RetrievalResult>,
    pub augmented_prompt: String,
}

impl RagResult {
    /// Number of retrieved chunks.
    pub fn num_retrieved(&self) -> usize {
        self.retrieved_chunks.len()
    }

    /// Score of the top-ranked retrieved chunk (0.0 if none).
    pub fn top_score(&self) -> f32 {
        self.retrieved_chunks.first().map(|r| r.score).unwrap_or(0.0)
    }

    /// Whether any retrieved chunk has a score at least `min_score`.
    pub fn has_relevant_context(&self, min_score: f32) -> bool {
        self.retrieved_chunks.iter().any(|r| r.score >= min_score)
    }
}

// ── RagError ──────────────────────────────────────────────────────────────────

/// Errors that can occur in the RAG pipeline.
#[derive(Debug)]
pub enum RagError {
    /// Retrieval was requested before indexing documents.
    NotIndexed,
    /// The query string was empty or only whitespace.
    EmptyQuery,
    /// No documents were provided for indexing.
    EmptyDocuments,
    /// An error occurred during document indexing.
    IndexingFailed(String),
    /// An error occurred during retrieval.
    RetrievalFailed(String),
}

impl std::fmt::Display for RagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RagError::NotIndexed => write!(
                f,
                "RAG pipeline has not been indexed yet; call index() first"
            ),
            RagError::EmptyQuery => write!(f, "query must not be empty"),
            RagError::EmptyDocuments => write!(f, "no documents provided for indexing"),
            RagError::IndexingFailed(msg) => write!(f, "indexing failed: {}", msg),
            RagError::RetrievalFailed(msg) => write!(f, "retrieval failed: {}", msg),
        }
    }
}

impl std::error::Error for RagError {}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Document tests ────────────────────────────────────────────────────────

    #[test]
    fn test_document_new() {
        let doc = Document::new("d1", "hello world");
        assert_eq!(doc.id, "d1");
        assert_eq!(doc.content, "hello world");
        assert!(doc.metadata.is_empty());
        assert_eq!(doc.word_count(), 2);
    }

    #[test]
    fn test_document_chunk_basic() {
        let doc = Document::new("doc", "one two three four five six seven eight nine ten");
        // chunk_size=4, overlap=0 → steps of 4
        let chunks = doc.chunk(4, 0);
        assert_eq!(chunks.len(), 3); // [0..4], [4..8], [8..10]
        assert_eq!(chunks[0].content, "one two three four");
        assert_eq!(chunks[0].start_word, 0);
        assert_eq!(chunks[0].end_word, 4);
        assert_eq!(chunks[1].content, "five six seven eight");
        assert_eq!(chunks[2].content, "nine ten");
        assert_eq!(chunks[0].chunk_idx, 0);
        assert_eq!(chunks[1].chunk_idx, 1);
    }

    #[test]
    fn test_document_chunk_overlap() {
        let doc = Document::new("doc", "a b c d e f g");
        // chunk_size=4, overlap=2 → step=2
        // chunks: [0..4], [2..6], [4..7]
        let chunks = doc.chunk(4, 2);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].content, "a b c d");
        assert_eq!(chunks[1].content, "c d e f");
        assert_eq!(chunks[2].content, "e f g");
    }

    #[test]
    fn test_document_chunk_small_doc() {
        // Document with fewer words than chunk_size should yield exactly one chunk
        let doc = Document::new("doc", "hello world");
        let chunks = doc.chunk(10, 3);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "hello world");
    }

    // ── TfIdfRetriever tests ──────────────────────────────────────────────────

    #[test]
    fn test_tfidf_tokenize() {
        let tokens = TfIdfRetriever::tokenize("Hello, World! This is a test.");
        // single-char tokens like "a" are filtered; "is" is kept (length 2)
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(!tokens.contains(&"a".to_string())); // single char filtered
    }

    #[test]
    fn test_tfidf_index_single_doc() {
        let docs = vec![Document::new(
            "d1",
            "rust programming language systems performance memory safety",
        )];
        let mut retriever = TfIdfRetriever::new();
        let n = retriever.index(&docs, 50, 0).expect("index should succeed");
        assert_eq!(n, 1);
        assert_eq!(retriever.num_chunks(), 1);
        assert!(retriever.vocab_size() > 0);
    }

    #[test]
    fn test_tfidf_index_multiple_docs() {
        let docs = vec![
            Document::new("d1", "rust programming language"),
            Document::new("d2", "python machine learning data science"),
        ];
        let mut retriever = TfIdfRetriever::new();
        let n = retriever.index(&docs, 50, 0).expect("index should succeed");
        assert_eq!(n, 2);
        assert!(retriever.vocab_size() >= 5);
    }

    #[test]
    fn test_tfidf_retrieve_relevant() {
        let docs = vec![
            Document::new("rust", "rust programming language systems memory safety"),
            Document::new(
                "python",
                "python machine learning data science artificial intelligence",
            ),
        ];
        let mut retriever = TfIdfRetriever::new();
        retriever.index(&docs, 50, 0).expect("index");
        let results = retriever.retrieve("rust systems programming", 2).expect("retrieve");
        assert!(!results.is_empty());
        // The rust document should rank first
        assert_eq!(results[0].chunk.doc_id, "rust");
    }

    #[test]
    fn test_tfidf_retrieve_top_k() {
        let docs: Vec<Document> = (0..5)
            .map(|i| {
                Document::new(
                    &format!("d{}", i),
                    &format!("document {} content words here", i),
                )
            })
            .collect();
        let mut retriever = TfIdfRetriever::new();
        retriever.index(&docs, 50, 0).expect("index");
        let results = retriever.retrieve("document content", 3).expect("retrieve");
        assert!(results.len() <= 3);
        // Ranks should be contiguous starting at 0
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i);
        }
    }

    // ── Bm25Retriever tests ───────────────────────────────────────────────────

    #[test]
    fn test_bm25_index() {
        let docs = vec![
            Document::new("d1", "rust programming language"),
            Document::new("d2", "python data science"),
        ];
        let mut retriever = Bm25Retriever::new(1.5, 0.75);
        let n = retriever.index(&docs, 50, 0).expect("index");
        assert_eq!(n, 2);
        assert_eq!(retriever.num_chunks(), 2);
    }

    #[test]
    fn test_bm25_retrieve_relevant() {
        let docs = vec![
            Document::new(
                "rust",
                "rust programming language systems memory safety borrow checker",
            ),
            Document::new(
                "ml",
                "machine learning neural networks deep learning gradient descent",
            ),
        ];
        let mut retriever = Bm25Retriever::new(1.5, 0.75);
        retriever.index(&docs, 50, 0).expect("index");
        let results = retriever.retrieve("rust borrow checker", 2).expect("retrieve");
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk.doc_id, "rust");
    }

    #[test]
    fn test_bm25_retriever_default() {
        let retriever = Bm25Retriever::default();
        assert_eq!(retriever.k1, 1.5);
        assert_eq!(retriever.b, 0.75);
        assert_eq!(retriever.num_chunks(), 0);
    }

    // ── RagConfig tests ───────────────────────────────────────────────────────

    #[test]
    fn test_rag_config_default() {
        let cfg = RagConfig::default();
        assert_eq!(cfg.retrieval_strategy, RetrievalStrategy::Bm25);
        assert_eq!(cfg.top_k, 3);
        assert_eq!(cfg.chunk_size, 200);
        assert_eq!(cfg.chunk_overlap, 50);
        assert_eq!(cfg.max_context_length, 2048);
        assert!(cfg.context_template.contains("{context}"));
        assert!(cfg.context_template.contains("{query}"));
    }

    // ── RagPipeline tests ─────────────────────────────────────────────────────

    fn make_docs() -> Vec<Document> {
        vec![
            Document::new(
                "rust",
                "Rust is a systems programming language focused on safety performance and concurrency",
            ),
            Document::new(
                "python",
                "Python is a high level language popular for machine learning and data science",
            ),
            Document::new(
                "go",
                "Go is a compiled language designed by Google for cloud and networking applications",
            ),
        ]
    }

    #[test]
    fn test_rag_pipeline_index() {
        let mut pipeline = RagPipeline::new(RagConfig::default());
        assert!(!pipeline.is_indexed());
        let n = pipeline.index(&make_docs()).expect("index");
        assert!(n > 0);
        assert!(pipeline.is_indexed());
        assert_eq!(pipeline.num_indexed_chunks(), n);
    }

    #[test]
    fn test_rag_pipeline_retrieve() {
        let mut pipeline = RagPipeline::new(RagConfig {
            top_k: 2,
            ..RagConfig::default()
        });
        pipeline.index(&make_docs()).expect("index");
        let results = pipeline.retrieve("Rust safety systems").expect("retrieve");
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_rag_pipeline_build_prompt() {
        let mut pipeline = RagPipeline::new(RagConfig::default());
        pipeline.index(&make_docs()).expect("index");
        let results = pipeline.retrieve("rust programming").expect("retrieve");
        let prompt = pipeline.build_prompt("What is Rust?", &results);
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("Context:"));
    }

    #[test]
    fn test_rag_pipeline_run() {
        let mut pipeline = RagPipeline::new(RagConfig::default());
        pipeline.index(&make_docs()).expect("index");
        let result = pipeline.run("What is Rust?").expect("run");
        assert_eq!(result.query, "What is Rust?");
        assert!(!result.augmented_prompt.is_empty());
        assert!(result.num_retrieved() > 0);
    }

    #[test]
    fn test_rag_result_has_relevant() {
        let mut pipeline = RagPipeline::new(RagConfig::default());
        pipeline.index(&make_docs()).expect("index");
        let result = pipeline.run("rust programming language").expect("run");
        // BM25 score for a matching query should be positive
        assert!(result.top_score() > 0.0);
        assert!(result.has_relevant_context(0.0));
        // Nothing should exceed an absurdly high threshold
        assert!(!result.has_relevant_context(9999.0));
    }

    // ── RagError tests ────────────────────────────────────────────────────────

    #[test]
    fn test_rag_error_not_indexed() {
        let pipeline = RagPipeline::new(RagConfig::default());
        let err = pipeline.retrieve("hello").unwrap_err();
        matches!(err, RagError::NotIndexed);
    }

    #[test]
    fn test_rag_error_display() {
        let err_not_indexed = RagError::NotIndexed;
        let err_empty_query = RagError::EmptyQuery;
        let err_empty_docs = RagError::EmptyDocuments;
        let err_indexing = RagError::IndexingFailed("oops".to_string());
        let err_retrieval = RagError::RetrievalFailed("bad".to_string());

        assert!(format!("{}", err_not_indexed).contains("index"));
        assert!(format!("{}", err_empty_query).contains("empty"));
        assert!(format!("{}", err_empty_docs).contains("documents"));
        assert!(format!("{}", err_indexing).contains("oops"));
        assert!(format!("{}", err_retrieval).contains("bad"));
    }

    // ── TF-IDF pipeline variant test ──────────────────────────────────────────

    #[test]
    fn test_rag_pipeline_tfidf_variant() {
        let config = RagConfig {
            retrieval_strategy: RetrievalStrategy::TfIdf,
            top_k: 2,
            chunk_size: 50,
            chunk_overlap: 10,
            ..RagConfig::default()
        };
        let mut pipeline = RagPipeline::new(config);
        pipeline.index(&make_docs()).expect("index");
        let result = pipeline.run("Google cloud networking applications").expect("run");
        assert!(!result.augmented_prompt.is_empty());
    }
}
