//! # Feature Extraction Pipeline
//!
//! Extract dense vector embeddings from text for semantic search, clustering, and retrieval.
//!
//! ## Supported pooling strategies
//! - **CLS** — use the [CLS] token embedding
//! - **MeanPooling** — mean over all token embeddings
//! - **MaxPooling** — per-dimension max over all token embeddings
//! - **WeightedMean** — weighted mean using per-token weights
//!
//! ## Example
//!
//! ```rust,ignore
//! use trustformers::pipeline::feature_extraction::{
//!     FeatureExtractionConfig, FeatureExtractionPipeline,
//! };
//!
//! let config = FeatureExtractionConfig::default();
//! let pipeline = FeatureExtractionPipeline::new(config)?;
//! let embedding = pipeline.extract("Hello world")?;
//! println!("dim={}, norm={:.4}", embedding.dim, embedding.norm);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the feature extraction pipeline.
#[derive(Debug, Error)]
pub enum ExtractionError {
    /// Input text was empty.
    #[error("Empty text")]
    EmptyText,
    /// Batch size exceeds the configured maximum.
    #[error("Batch too large: {0} > max {1}")]
    BatchTooLarge(usize, usize),
    /// Number of clusters is invalid (0 or > number of texts).
    #[error("Invalid cluster count: {0}")]
    InvalidClusters(usize),
    /// Underlying model error.
    #[error("Model error: {0}")]
    ModelError(String),
}

// ---------------------------------------------------------------------------
// PoolingStrategy
// ---------------------------------------------------------------------------

/// Strategy used to convert per-token representations into a single embedding.
#[derive(Debug, Clone, PartialEq)]
pub enum PoolingStrategy {
    /// Use the [CLS] token embedding.
    Cls,
    /// Mean of all token embeddings.
    MeanPooling,
    /// Per-dimension max of all token embeddings.
    MaxPooling,
    /// Weighted mean using per-token weights.
    WeightedMean,
}

// ---------------------------------------------------------------------------
// FeatureExtractionConfig
// ---------------------------------------------------------------------------

/// Configuration for [`FeatureExtractionPipeline`].
#[derive(Debug, Clone)]
pub struct FeatureExtractionConfig {
    /// HuggingFace model identifier or local path.
    pub model_name: String,
    /// Dimensionality of the output embedding.
    pub embedding_dim: usize,
    /// Strategy for aggregating per-token embeddings.
    pub pooling_strategy: PoolingStrategy,
    /// Whether to L2-normalise the output embedding to a unit vector.
    pub normalize: bool,
    /// Maximum number of tokens to consider per input.
    pub max_length: usize,
    /// Maximum number of items in a single batch call.
    pub batch_size: usize,
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            embedding_dim: 384,
            pooling_strategy: PoolingStrategy::MeanPooling,
            normalize: true,
            max_length: 512,
            batch_size: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// A dense vector embedding.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Raw float values of the embedding.
    pub values: Vec<f32>,
    /// Number of dimensions.
    pub dim: usize,
    /// L2 norm of `values`.
    pub norm: f32,
}

impl Embedding {
    /// Construct a new `Embedding` from a `Vec<f32>`, computing `dim` and `norm`.
    pub fn new(values: Vec<f32>) -> Self {
        let dim = values.len();
        let norm = compute_l2_norm(&values);
        Self { values, dim, norm }
    }

    /// Return an L2-normalised copy of this embedding (unit vector).
    /// If the norm is zero the original values are returned unchanged.
    pub fn normalize(&self) -> Self {
        if self.norm < f32::EPSILON {
            return self.clone();
        }
        let normed: Vec<f32> = self.values.iter().map(|v| v / self.norm).collect();
        Self {
            dim: self.dim,
            norm: 1.0,
            values: normed,
        }
    }

    /// Cosine similarity with another embedding: dot(a,b) / (|a| * |b|).
    /// Returns 0.0 when either norm is zero.
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        let denom = self.norm * other.norm;
        if denom < f32::EPSILON {
            return 0.0;
        }
        self.dot_product(other) / denom
    }

    /// Dot product of two embeddings.
    pub fn dot_product(&self, other: &Embedding) -> f32 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Euclidean (L2) distance between two embeddings.
    pub fn euclidean_distance(&self, other: &Embedding) -> f32 {
        let sq_sum: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        sq_sum.sqrt()
    }

    /// Clone the raw values into a new `Vec<f32>`.
    pub fn to_vec(&self) -> Vec<f32> {
        self.values.clone()
    }
}

// ---------------------------------------------------------------------------
// SearchResult / ClusteringResult
// ---------------------------------------------------------------------------

/// A single result from semantic search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The text from the corpus.
    pub text: String,
    /// Original index in the corpus slice.
    pub index: usize,
    /// Cosine similarity score with the query.
    pub score: f32,
}

/// Output of k-means clustering.
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Cluster id for each input text.
    pub assignments: Vec<usize>,
    /// Centroid embedding for each cluster.
    pub centroids: Vec<Embedding>,
    /// Sum of squared distances from each point to its centroid (inertia).
    pub inertia: f64,
    /// Number of Lloyd iterations actually executed.
    pub iterations: usize,
}

// ---------------------------------------------------------------------------
// FeatureExtractionPipeline
// ---------------------------------------------------------------------------

/// Pipeline for extracting dense vector embeddings from text.
pub struct FeatureExtractionPipeline {
    config: FeatureExtractionConfig,
}

impl FeatureExtractionPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: FeatureExtractionConfig) -> Result<Self, ExtractionError> {
        if config.embedding_dim == 0 {
            return Err(ExtractionError::ModelError(
                "embedding_dim must be > 0".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Extract a single embedding for `text`.
    ///
    /// Uses a deterministic mock: embedding[i] = sin(hash * (i+1) * 0.01).
    pub fn extract(&self, text: &str) -> Result<Embedding, ExtractionError> {
        if text.is_empty() {
            return Err(ExtractionError::EmptyText);
        }
        let embedding = self.mock_embed(text);
        Ok(embedding)
    }

    /// Extract embeddings for a batch of texts.
    pub fn extract_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, ExtractionError> {
        if texts.len() > self.config.batch_size {
            return Err(ExtractionError::BatchTooLarge(
                texts.len(),
                self.config.batch_size,
            ));
        }
        texts.iter().map(|t| self.extract(t)).collect()
    }

    /// Return the top-`top_k` corpus entries most similar to `query`.
    ///
    /// Results are sorted by descending cosine similarity.
    pub fn semantic_search(
        &self,
        query: &str,
        corpus: &[&str],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, ExtractionError> {
        let query_emb = self.extract(query)?;
        let mut scored: Vec<SearchResult> = corpus
            .iter()
            .enumerate()
            .map(|(idx, text)| {
                let emb = self.mock_embed(text);
                let score = query_emb.cosine_similarity(&emb);
                SearchResult {
                    text: text.to_string(),
                    index: idx,
                    score,
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Cluster `texts` into `num_clusters` groups using Lloyd's k-means algorithm.
    pub fn cluster(
        &self,
        texts: &[&str],
        num_clusters: usize,
        max_iterations: usize,
    ) -> Result<ClusteringResult, ExtractionError> {
        if num_clusters == 0 || num_clusters > texts.len() {
            return Err(ExtractionError::InvalidClusters(num_clusters));
        }

        let embeddings: Vec<Embedding> = texts.iter().map(|t| self.mock_embed(t)).collect();
        let dim = self.config.embedding_dim;

        // Initialise centroids from first `num_clusters` embeddings.
        let mut centroids: Vec<Vec<f32>> = (0..num_clusters)
            .map(|i| embeddings[i].values.clone())
            .collect();

        let mut assignments = vec![0usize; texts.len()];
        let mut actual_iters = 0usize;

        for _iter in 0..max_iterations {
            actual_iters += 1;
            let mut changed = false;

            // Assignment step.
            for (idx, emb) in embeddings.iter().enumerate() {
                let best = (0..num_clusters)
                    .map(|c| {
                        let dist_sq: f32 = emb
                            .values
                            .iter()
                            .zip(centroids[c].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        (c, dist_sq)
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(c, _)| c)
                    .unwrap_or(0);

                if assignments[idx] != best {
                    assignments[idx] = best;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update step: recompute centroids as mean of assigned embeddings.
            let mut sums = vec![vec![0.0f32; dim]; num_clusters];
            let mut counts = vec![0usize; num_clusters];
            for (idx, &cluster) in assignments.iter().enumerate() {
                counts[cluster] += 1;
                for d in 0..dim {
                    sums[cluster][d] += embeddings[idx].values[d];
                }
            }
            for c in 0..num_clusters {
                if counts[c] > 0 {
                    for d in 0..dim {
                        centroids[c][d] = sums[c][d] / counts[c] as f32;
                    }
                }
            }
        }

        // Compute inertia.
        let inertia: f64 = embeddings
            .iter()
            .enumerate()
            .map(|(idx, emb)| {
                let c = assignments[idx];
                emb.values
                    .iter()
                    .zip(centroids[c].iter())
                    .map(|(a, b)| ((a - b) as f64).powi(2))
                    .sum::<f64>()
            })
            .sum();

        let centroid_embeddings: Vec<Embedding> =
            centroids.into_iter().map(Embedding::new).collect();

        Ok(ClusteringResult {
            assignments,
            centroids: centroid_embeddings,
            inertia,
            iterations: actual_iters,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Generate a deterministic embedding from `text` using a hash-based formula.
    fn mock_embed(&self, text: &str) -> Embedding {
        let hash_val = simple_hash(text);
        let dim = self.config.embedding_dim;
        let raw: Vec<f32> = (0..dim)
            .map(|i| (hash_val as f64 * (i as f64 + 1.0) * 0.01).sin() as f32)
            .collect();

        if self.config.normalize {
            Embedding::new(raw).normalize()
        } else {
            Embedding::new(raw)
        }
    }
}

// ---------------------------------------------------------------------------
// EmbeddingNormalizer — pooling and normalisation utilities
// ---------------------------------------------------------------------------

/// Stateless pooling and normalisation helpers for embedding pipelines.
pub struct EmbeddingNormalizer;

impl EmbeddingNormalizer {
    /// L2-normalise a single embedding vector to unit length.
    ///
    /// If the norm is ≤ ε the original vector is returned unchanged.
    pub fn l2_normalize(embedding: &[f32]) -> Vec<f32> {
        let norm = compute_l2_norm(embedding);
        if norm < f32::EPSILON {
            return embedding.to_vec();
        }
        embedding.iter().map(|&v| v / norm).collect()
    }

    /// Masked mean pooling: average token embeddings weighted by `attention_mask`.
    ///
    /// Only positions where `attention_mask[i] > 0` contribute.  If the sum of
    /// mask values is zero, the zero vector is returned.
    ///
    /// All embeddings must have the same dimensionality as `token_embeddings[0]`.
    pub fn mean_pooling(token_embeddings: &[Vec<f32>], attention_mask: &[u32]) -> Vec<f32> {
        if token_embeddings.is_empty() {
            return Vec::new();
        }
        let dim = token_embeddings[0].len();
        let mut sum = vec![0.0f32; dim];
        let mut total_weight = 0.0f32;
        for (emb, &mask) in token_embeddings.iter().zip(attention_mask.iter()) {
            if mask > 0 {
                let w = mask as f32;
                for (s, &e) in sum.iter_mut().zip(emb.iter()) {
                    *s += e * w;
                }
                total_weight += w;
            }
        }
        if total_weight < f32::EPSILON {
            return vec![0.0; dim];
        }
        sum.iter().map(|&s| s / total_weight).collect()
    }

    /// Max pooling: element-wise maximum over all token embeddings.
    pub fn max_pooling(token_embeddings: &[Vec<f32>]) -> Vec<f32> {
        if token_embeddings.is_empty() {
            return Vec::new();
        }
        let dim = token_embeddings[0].len();
        let mut result = vec![f32::NEG_INFINITY; dim];
        for emb in token_embeddings {
            for (r, &e) in result.iter_mut().zip(emb.iter()) {
                if e > *r {
                    *r = e;
                }
            }
        }
        // Replace any leftover NEG_INFINITY (shouldn't happen with non-empty input) with 0
        result.iter_mut().for_each(|v| {
            if v.is_infinite() {
                *v = 0.0;
            }
        });
        result
    }

    /// CLS pooling: return the first token's embedding.
    pub fn cls_pooling(token_embeddings: &[Vec<f32>]) -> Vec<f32> {
        token_embeddings.first().cloned().unwrap_or_default()
    }

    /// Weighted mean pooling with explicit per-token scalar weights.
    ///
    /// `weights` must have the same length as `token_embeddings`.
    pub fn weighted_mean_pooling(token_embeddings: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
        if token_embeddings.is_empty() {
            return Vec::new();
        }
        let dim = token_embeddings[0].len();
        let total_weight: f32 = weights.iter().sum();
        if total_weight < f32::EPSILON {
            return vec![0.0; dim];
        }
        let mut result = vec![0.0f32; dim];
        for (emb, &w) in token_embeddings.iter().zip(weights.iter()) {
            for (r, &e) in result.iter_mut().zip(emb.iter()) {
                *r += e * w;
            }
        }
        result.iter_mut().for_each(|r| *r /= total_weight);
        result
    }
}

// ---------------------------------------------------------------------------
// SimilarityMetrics — pairwise / pointwise distance measures
// ---------------------------------------------------------------------------

/// Collection of distance and similarity functions between embedding vectors.
pub struct SimilarityMetrics;

impl SimilarityMetrics {
    /// Cosine similarity: dot(a, b) / (|a| × |b|).
    ///
    /// Returns 0.0 if either norm is ≈ 0.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let norm_a = compute_l2_norm(a);
        let norm_b = compute_l2_norm(b);
        let denom = norm_a * norm_b;
        if denom < f32::EPSILON {
            return 0.0;
        }
        Self::dot_product(a, b) / denom
    }

    /// Euclidean (L2) distance.
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Dot product of two vectors.
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Manhattan (L1) distance.
    pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
    }

    /// Compute the N×N cosine similarity matrix for a set of embeddings.
    ///
    /// `matrix[i][j]` = cosine similarity between embeddings `i` and `j`.
    pub fn pairwise_cosine_matrix(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = embeddings.len();
        let mut matrix = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = Self::cosine_similarity(&embeddings[i], &embeddings[j]);
            }
        }
        matrix
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Simple djb2-style hash for deterministic mock embeddings.
fn simple_hash(text: &str) -> u64 {
    let mut h: u64 = 5381;
    for byte in text.bytes() {
        h = h.wrapping_mul(33).wrapping_add(byte as u64);
    }
    h
}

/// Compute the L2 (Euclidean) norm of a float slice.
fn compute_l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pipeline() -> FeatureExtractionPipeline {
        FeatureExtractionPipeline::new(FeatureExtractionConfig::default()).unwrap()
    }

    // 1. Embedding::new populates dim and norm correctly.
    #[test]
    fn embedding_new_dim_and_norm() {
        let values = vec![3.0f32, 4.0];
        let emb = Embedding::new(values);
        assert_eq!(emb.dim, 2);
        assert!((emb.norm - 5.0).abs() < 1e-5);
    }

    // 2. Embedding::normalize returns a unit vector.
    #[test]
    fn embedding_normalize_unit_vector() {
        let emb = Embedding::new(vec![3.0f32, 4.0]);
        let normed = emb.normalize();
        assert!((normed.norm - 1.0).abs() < 1e-5);
    }

    // 3. Cosine similarity of identical embeddings is 1.0.
    #[test]
    fn embedding_cosine_similarity_identical() {
        let emb = Embedding::new(vec![1.0f32, 0.0, 0.0]);
        let sim = emb.cosine_similarity(&emb);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    // 4. Cosine similarity of orthogonal embeddings is 0.0.
    #[test]
    fn embedding_cosine_similarity_orthogonal() {
        let a = Embedding::new(vec![1.0f32, 0.0]);
        let b = Embedding::new(vec![0.0f32, 1.0]);
        assert!((a.cosine_similarity(&b)).abs() < 1e-5);
    }

    // 5. Dot product.
    #[test]
    fn embedding_dot_product() {
        let a = Embedding::new(vec![1.0f32, 2.0, 3.0]);
        let b = Embedding::new(vec![4.0f32, 5.0, 6.0]);
        assert!((a.dot_product(&b) - 32.0).abs() < 1e-5);
    }

    // 6. Euclidean distance.
    #[test]
    fn embedding_euclidean_distance() {
        let a = Embedding::new(vec![0.0f32, 0.0]);
        let b = Embedding::new(vec![3.0f32, 4.0]);
        assert!((a.euclidean_distance(&b) - 5.0).abs() < 1e-5);
    }

    // 7. Basic extraction succeeds and returns correct dim.
    #[test]
    fn extract_basic() {
        let pipe = default_pipeline();
        let emb = pipe.extract("Hello world").unwrap();
        assert_eq!(emb.dim, 384);
        assert!(emb.norm > 0.0);
    }

    // 8. Empty text returns ExtractionError::EmptyText.
    #[test]
    fn extract_empty_error() {
        let pipe = default_pipeline();
        let result = pipe.extract("");
        assert!(matches!(result, Err(ExtractionError::EmptyText)));
    }

    // 9. extract_batch returns one embedding per text.
    #[test]
    fn extract_batch_count() {
        let pipe = default_pipeline();
        let texts = vec!["foo", "bar", "baz"];
        let embeddings = pipe.extract_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 3);
    }

    // 10. semantic_search returns results in descending score order.
    #[test]
    fn semantic_search_ordering() {
        let pipe = default_pipeline();
        let corpus = vec!["apple banana", "dog cat", "machine learning", "neural network"];
        let results = pipe.semantic_search("apple", &corpus, 4).unwrap();
        assert_eq!(results.len(), 4);
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    // 11. semantic_search top_k respects the limit even when corpus is larger.
    #[test]
    fn semantic_search_top_k_limit() {
        let pipe = default_pipeline();
        let corpus = vec!["a", "b", "c", "d", "e"];
        let results = pipe.semantic_search("query", &corpus, 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    // 12. cluster returns one assignment per text.
    #[test]
    fn cluster_assignment_count() {
        let pipe = default_pipeline();
        let texts = vec!["foo", "bar", "baz", "qux", "quux", "corge"];
        let result = pipe.cluster(&texts, 2, 10).unwrap();
        assert_eq!(result.assignments.len(), texts.len());
    }

    // 13. cluster returns one centroid per cluster.
    #[test]
    fn cluster_centroids_count() {
        let pipe = default_pipeline();
        let texts = vec!["alpha", "beta", "gamma", "delta"];
        let result = pipe.cluster(&texts, 2, 10).unwrap();
        assert_eq!(result.centroids.len(), 2);
    }

    // 14. Inertia is finite (not NaN or infinity).
    #[test]
    fn cluster_inertia_finite() {
        let pipe = default_pipeline();
        let texts = vec!["one", "two", "three", "four"];
        let result = pipe.cluster(&texts, 2, 10).unwrap();
        assert!(result.inertia.is_finite());
    }

    // 15. Pooling strategy variants are configurable.
    #[test]
    fn pooling_strategies_config() {
        for strategy in [
            PoolingStrategy::Cls,
            PoolingStrategy::MeanPooling,
            PoolingStrategy::MaxPooling,
            PoolingStrategy::WeightedMean,
        ] {
            let config = FeatureExtractionConfig {
                pooling_strategy: strategy.clone(),
                ..FeatureExtractionConfig::default()
            };
            assert_eq!(config.pooling_strategy, strategy);
        }
    }

    // 16. normalize flag: when false, norm is not necessarily 1.0.
    #[test]
    fn normalize_flag_false() {
        let config = FeatureExtractionConfig {
            normalize: false,
            ..FeatureExtractionConfig::default()
        };
        let pipe = FeatureExtractionPipeline::new(config).unwrap();
        let emb = pipe.extract("test text").unwrap();
        // Norm should be > 0 but not constrained to 1.0.
        assert!(emb.norm > 0.0);
    }

    // -----------------------------------------------------------------------
    // EmbeddingNormalizer::l2_normalize
    // -----------------------------------------------------------------------

    #[test]
    fn l2_normalize_unit_vector() {
        let v = vec![3.0f32, 4.0];
        let normed = EmbeddingNormalizer::l2_normalize(&v);
        let norm: f32 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_zero_vector_unchanged() {
        let v = vec![0.0f32, 0.0, 0.0];
        let normed = EmbeddingNormalizer::l2_normalize(&v);
        assert_eq!(normed, v);
    }

    #[test]
    fn l2_normalize_already_unit() {
        let v = vec![1.0f32, 0.0, 0.0];
        let normed = EmbeddingNormalizer::l2_normalize(&v);
        assert!((normed[0] - 1.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // EmbeddingNormalizer::mean_pooling
    // -----------------------------------------------------------------------

    #[test]
    fn mean_pooling_all_masked() {
        let embeddings = vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mask = vec![1u32, 1, 1];
        let pooled = EmbeddingNormalizer::mean_pooling(&embeddings, &mask);
        assert!((pooled[0] - 3.0).abs() < 1e-5);
        assert!((pooled[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn mean_pooling_partial_mask() {
        let embeddings = vec![vec![2.0f32, 4.0], vec![10.0, 20.0]];
        // Only first token is unmasked
        let mask = vec![1u32, 0];
        let pooled = EmbeddingNormalizer::mean_pooling(&embeddings, &mask);
        assert!((pooled[0] - 2.0).abs() < 1e-5);
        assert!((pooled[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn mean_pooling_zero_mask_returns_zero() {
        let embeddings = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
        let mask = vec![0u32, 0];
        let pooled = EmbeddingNormalizer::mean_pooling(&embeddings, &mask);
        for &v in &pooled {
            assert!(v.abs() < 1e-8);
        }
    }

    #[test]
    fn mean_pooling_empty_returns_empty() {
        let pooled = EmbeddingNormalizer::mean_pooling(&[], &[]);
        assert!(pooled.is_empty());
    }

    // -----------------------------------------------------------------------
    // EmbeddingNormalizer::max_pooling
    // -----------------------------------------------------------------------

    #[test]
    fn max_pooling_basic() {
        let embeddings = vec![vec![1.0f32, 5.0], vec![3.0, 2.0], vec![2.0, 4.0]];
        let pooled = EmbeddingNormalizer::max_pooling(&embeddings);
        assert!((pooled[0] - 3.0).abs() < 1e-5);
        assert!((pooled[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn max_pooling_single_token() {
        let embeddings = vec![vec![7.0f32, 8.0]];
        let pooled = EmbeddingNormalizer::max_pooling(&embeddings);
        assert!((pooled[0] - 7.0).abs() < 1e-5);
        assert!((pooled[1] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn max_pooling_empty_returns_empty() {
        let pooled = EmbeddingNormalizer::max_pooling(&[]);
        assert!(pooled.is_empty());
    }

    // -----------------------------------------------------------------------
    // EmbeddingNormalizer::cls_pooling
    // -----------------------------------------------------------------------

    #[test]
    fn cls_pooling_returns_first() {
        let embeddings = vec![
            vec![1.0f32, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let pooled = EmbeddingNormalizer::cls_pooling(&embeddings);
        assert_eq!(pooled, vec![1.0, 2.0]);
    }

    #[test]
    fn cls_pooling_empty_returns_empty() {
        let pooled = EmbeddingNormalizer::cls_pooling(&[]);
        assert!(pooled.is_empty());
    }

    // -----------------------------------------------------------------------
    // EmbeddingNormalizer::weighted_mean_pooling
    // -----------------------------------------------------------------------

    #[test]
    fn weighted_mean_pooling_basic() {
        let embeddings = vec![vec![0.0f32, 0.0], vec![4.0, 8.0]];
        let weights = vec![0.0f32, 1.0]; // only second token
        let pooled = EmbeddingNormalizer::weighted_mean_pooling(&embeddings, &weights);
        assert!((pooled[0] - 4.0).abs() < 1e-5);
        assert!((pooled[1] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn weighted_mean_pooling_equal_weights_same_as_mean() {
        let embeddings = vec![vec![2.0f32, 6.0], vec![4.0, 2.0]];
        let weights = vec![1.0f32, 1.0];
        let pooled = EmbeddingNormalizer::weighted_mean_pooling(&embeddings, &weights);
        assert!((pooled[0] - 3.0).abs() < 1e-5);
        assert!((pooled[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn weighted_mean_pooling_zero_weights_zero_output() {
        let embeddings = vec![vec![5.0f32, 10.0]];
        let weights = vec![0.0f32];
        let pooled = EmbeddingNormalizer::weighted_mean_pooling(&embeddings, &weights);
        for &v in &pooled {
            assert!(v.abs() < 1e-8);
        }
    }

    // -----------------------------------------------------------------------
    // SimilarityMetrics
    // -----------------------------------------------------------------------

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0f32, 0.0, 0.0];
        let sim = SimilarityMetrics::cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(SimilarityMetrics::cosine_similarity(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        assert!((SimilarityMetrics::cosine_similarity(&a, &b) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn euclidean_distance_known() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        assert!((SimilarityMetrics::euclidean_distance(&a, &b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn euclidean_distance_same_vector_zero() {
        let a = vec![1.0f32, 2.0, 3.0];
        assert!(SimilarityMetrics::euclidean_distance(&a, &a) < 1e-8);
    }

    #[test]
    fn dot_product_known() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        assert!((SimilarityMetrics::dot_product(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn manhattan_distance_known() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 6.0, 3.0];
        // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
        assert!((SimilarityMetrics::manhattan_distance(&a, &b) - 7.0).abs() < 1e-5);
    }

    #[test]
    fn manhattan_distance_same_vector_zero() {
        let a = vec![5.0f32, -3.0, 2.0];
        assert!(SimilarityMetrics::manhattan_distance(&a, &a) < 1e-8);
    }

    #[test]
    fn pairwise_cosine_matrix_diagonal_is_one() {
        let embeddings = vec![
            vec![1.0f32, 0.0],
            vec![0.0f32, 1.0],
            vec![1.0f32, 1.0],
        ];
        let matrix = SimilarityMetrics::pairwise_cosine_matrix(&embeddings);
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        for i in 0..3 {
            assert!((matrix[i][i] - 1.0).abs() < 1e-4, "diagonal[{i}]={}", matrix[i][i]);
        }
    }

    #[test]
    fn pairwise_cosine_matrix_symmetric() {
        let embeddings = vec![
            vec![1.0f32, 2.0],
            vec![3.0f32, 4.0],
        ];
        let matrix = SimilarityMetrics::pairwise_cosine_matrix(&embeddings);
        assert!((matrix[0][1] - matrix[1][0]).abs() < 1e-5);
    }

    #[test]
    fn pairwise_cosine_matrix_orthogonal_vectors() {
        let embeddings = vec![
            vec![1.0f32, 0.0],
            vec![0.0f32, 1.0],
        ];
        let matrix = SimilarityMetrics::pairwise_cosine_matrix(&embeddings);
        assert!(matrix[0][1].abs() < 1e-5);
        assert!(matrix[1][0].abs() < 1e-5);
    }
}
