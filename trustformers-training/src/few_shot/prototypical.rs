//! Prototypical Networks, Matching Networks, and ICL episode builder
//! for few-shot learning.
//!
//! ## Key types
//!
//! - [`PrototypicalNetworks`]: Nearest-prototype classifier.
//! - [`MatchingNetworks`]: Soft attention-based few-shot classifier.
//! - [`IclEpisodeBuilder`]: In-context learning prompt builder.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned by few-shot learning operations.
#[derive(Debug, Clone, PartialEq)]
pub struct FewShotError {
    pub message: String,
}

impl FewShotError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

impl fmt::Display for FewShotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FewShotError: {}", self.message)
    }
}

impl std::error::Error for FewShotError {}

// ---------------------------------------------------------------------------
// Prototypical Networks
// ---------------------------------------------------------------------------

/// Configuration for prototypical networks.
#[derive(Debug, Clone)]
pub struct PrototypicalConfig {
    /// Dimension of the embedding space.
    pub embedding_dim: usize,
    /// Number of classes per episode (N-way).
    pub n_way: usize,
    /// Number of support examples per class (K-shot).
    pub k_shot: usize,
    /// Number of query examples per class.
    pub n_query: usize,
}

impl PrototypicalConfig {
    pub fn new(embedding_dim: usize, n_way: usize, k_shot: usize, n_query: usize) -> Self {
        Self {
            embedding_dim,
            n_way,
            k_shot,
            n_query,
        }
    }
}

/// Prototypical Networks for few-shot classification.
///
/// The class prototype is the mean of support embeddings for that class.
/// Classification uses nearest Euclidean prototype.
pub struct PrototypicalNetworks {
    config: PrototypicalConfig,
}

impl PrototypicalNetworks {
    /// Create a new instance with the given config.
    pub fn new(config: PrototypicalConfig) -> Self {
        Self { config }
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &PrototypicalConfig {
        &self.config
    }

    /// Compute class prototypes as the mean of support embeddings.
    ///
    /// `support_embeddings`: flattened as `[n_way * k_shot, embedding_dim]`.
    /// `labels`: class index for each support embedding (length `n_way * k_shot`).
    ///
    /// Returns a vec of `n_way` prototype vectors, each of dimension `embedding_dim`.
    pub fn compute_prototypes(
        &self,
        support_embeddings: &[Vec<f32>],
        labels: &[usize],
    ) -> Result<Vec<Vec<f32>>, FewShotError> {
        if support_embeddings.is_empty() {
            return Err(FewShotError::new("support_embeddings must not be empty"));
        }
        if support_embeddings.len() != labels.len() {
            return Err(FewShotError::new(
                "support_embeddings and labels must have the same length",
            ));
        }
        let dim = self.config.embedding_dim;
        for (i, emb) in support_embeddings.iter().enumerate() {
            if emb.len() != dim {
                return Err(FewShotError::new(format!(
                    "support_embedding[{}] has dim {}, expected {}",
                    i,
                    emb.len(),
                    dim
                )));
            }
        }

        let n_way = self.config.n_way;
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0_f32; dim]; n_way];
        let mut counts: Vec<usize> = vec![0; n_way];

        for (emb, &label) in support_embeddings.iter().zip(labels.iter()) {
            if label >= n_way {
                return Err(FewShotError::new(format!(
                    "label {} out of range [0, {})",
                    label, n_way
                )));
            }
            for (j, &v) in emb.iter().enumerate() {
                sums[label][j] += v;
            }
            counts[label] += 1;
        }

        let mut prototypes = Vec::with_capacity(n_way);
        for (class, count) in counts.iter().enumerate().take(n_way) {
            if *count == 0 {
                return Err(FewShotError::new(format!(
                    "class {} has no support examples",
                    class
                )));
            }
            let proto: Vec<f32> = sums[class].iter().map(|s| s / *count as f32).collect();
            prototypes.push(proto);
        }

        Ok(prototypes)
    }

    /// Classify query embeddings using nearest prototype (Euclidean distance).
    ///
    /// Returns predicted class indices for each query.
    pub fn classify_queries(
        &self,
        query_embeddings: &[Vec<f32>],
        prototypes: &[Vec<f32>],
    ) -> Result<Vec<usize>, FewShotError> {
        if query_embeddings.is_empty() {
            return Err(FewShotError::new("query_embeddings must not be empty"));
        }
        if prototypes.is_empty() {
            return Err(FewShotError::new("prototypes must not be empty"));
        }
        let dim = self.config.embedding_dim;
        for (i, proto) in prototypes.iter().enumerate() {
            if proto.len() != dim {
                return Err(FewShotError::new(format!(
                    "prototype[{}] has dim {}, expected {}",
                    i,
                    proto.len(),
                    dim
                )));
            }
        }

        let mut predictions = Vec::with_capacity(query_embeddings.len());
        for (qi, query) in query_embeddings.iter().enumerate() {
            if query.len() != dim {
                return Err(FewShotError::new(format!(
                    "query[{}] has dim {}, expected {}",
                    qi,
                    query.len(),
                    dim
                )));
            }
            let mut best_class = 0;
            let mut best_dist = f32::MAX;
            for (ci, proto) in prototypes.iter().enumerate() {
                let d = Self::euclidean_distance(query, proto);
                if d < best_dist {
                    best_dist = d;
                    best_class = ci;
                }
            }
            predictions.push(best_class);
        }

        Ok(predictions)
    }

    /// Compute the episode loss: mean negative log probability over queries.
    ///
    /// P(y = c | x) ∝ exp(-||x - prototype_c||²)
    pub fn episode_loss(
        &self,
        query_embeddings: &[Vec<f32>],
        query_labels: &[usize],
        prototypes: &[Vec<f32>],
    ) -> Result<f32, FewShotError> {
        if query_embeddings.len() != query_labels.len() {
            return Err(FewShotError::new(
                "query_embeddings and query_labels must have the same length",
            ));
        }
        if query_embeddings.is_empty() {
            return Err(FewShotError::new("query set must not be empty"));
        }
        if prototypes.is_empty() {
            return Err(FewShotError::new("prototypes must not be empty"));
        }

        let n_classes = prototypes.len();
        let dim = self.config.embedding_dim;
        let mut total_loss = 0.0_f32;

        for (qi, (query, &true_class)) in
            query_embeddings.iter().zip(query_labels.iter()).enumerate()
        {
            if query.len() != dim {
                return Err(FewShotError::new(format!(
                    "query[{}] has dim {}, expected {}",
                    qi,
                    query.len(),
                    dim
                )));
            }
            if true_class >= n_classes {
                return Err(FewShotError::new(format!(
                    "query label {} out of range [0, {})",
                    true_class, n_classes
                )));
            }

            // Compute negative squared distances.
            let neg_sq_dists: Vec<f32> = prototypes
                .iter()
                .map(|proto| {
                    let d = Self::euclidean_distance(query, proto);
                    -d * d
                })
                .collect();

            // Numerically stable softmax.
            let max_val = neg_sq_dists.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = neg_sq_dists.iter().map(|&v| (v - max_val).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let log_prob = exps[true_class].ln() - sum_exp.ln();
            total_loss += -log_prob;
        }

        Ok(total_loss / query_embeddings.len() as f32)
    }

    /// Euclidean distance between two vectors.
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
    }
}

// ---------------------------------------------------------------------------
// Matching Networks
// ---------------------------------------------------------------------------

/// Matching Networks for few-shot learning using soft attention.
pub struct MatchingNetworks {
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Number of classes per episode.
    pub n_way: usize,
    /// Number of support examples per class.
    pub k_shot: usize,
}

impl MatchingNetworks {
    /// Compute the soft attention kernel over support embeddings.
    ///
    /// k(x̂, x_i) = softmax( -||x̂ - x_i||² )
    ///
    /// Returns a probability distribution over the support set.
    pub fn attention_kernel(query: &[f32], support: &[Vec<f32>]) -> Vec<f32> {
        if support.is_empty() {
            return Vec::new();
        }
        // Compute negative squared Euclidean distances.
        let neg_sq_dists: Vec<f32> = support
            .iter()
            .map(|s| {
                let d = PrototypicalNetworks::euclidean_distance(query, s);
                -d * d
            })
            .collect();

        // Numerically stable softmax.
        let max_val = neg_sq_dists.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = neg_sq_dists.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        if sum_exp == 0.0 {
            let n = support.len();
            return vec![1.0 / n as f32; n];
        }
        exps.iter().map(|e| e / sum_exp).collect()
    }

    /// Classify a query using attention-weighted sum over support labels.
    ///
    /// Returns a probability distribution over classes (length = `n_way`).
    pub fn classify(
        &self,
        query: &[f32],
        support_embeddings: &[Vec<f32>],
        support_labels: &[usize],
    ) -> Result<Vec<f32>, FewShotError> {
        if support_embeddings.len() != support_labels.len() {
            return Err(FewShotError::new(
                "support_embeddings and support_labels must have the same length",
            ));
        }
        if support_embeddings.is_empty() {
            return Err(FewShotError::new("support_embeddings must not be empty"));
        }

        let attn = Self::attention_kernel(query, support_embeddings);
        let mut class_probs = vec![0.0_f32; self.n_way];

        for (weight, &label) in attn.iter().zip(support_labels.iter()) {
            if label >= self.n_way {
                return Err(FewShotError::new(format!(
                    "support label {} out of range [0, {})",
                    label, self.n_way
                )));
            }
            class_probs[label] += weight;
        }

        Ok(class_probs)
    }
}

// ---------------------------------------------------------------------------
// In-context learning episode builder
// ---------------------------------------------------------------------------

/// Template for building in-context learning prompts.
#[derive(Debug, Clone)]
pub enum IclTemplate {
    /// Question-Answer format.
    QA {
        question_prefix: String,
        answer_prefix: String,
    },
    /// Sentiment analysis format.
    SentimentAnalysis {
        label_positive: String,
        label_negative: String,
    },
    /// Custom format using `{input}` and `{label}` placeholders.
    Custom {
        /// Format string with `{input}` and `{label}` placeholders.
        format_fn: String,
    },
}

/// Builder for in-context learning (ICL) few-shot prompts.
pub struct IclEpisodeBuilder {
    /// Number of demonstration examples (shots).
    pub n_shot: usize,
    /// Template used to format demonstrations.
    pub template: IclTemplate,
}

impl IclEpisodeBuilder {
    /// Create a new builder.
    pub fn new(n_shot: usize, template: IclTemplate) -> Self {
        Self { n_shot, template }
    }

    /// Build a full ICL prompt from demonstration pairs and a query.
    ///
    /// `demonstrations`: vec of `(input, label)` pairs used as few-shot examples.
    /// `query`: the test input to predict.
    pub fn build_prompt(&self, demonstrations: &[(String, String)], query: &str) -> String {
        let demos_to_use = demonstrations.len().min(self.n_shot);
        let mut prompt = String::new();

        for (input, label) in demonstrations.iter().take(demos_to_use) {
            let formatted = self.format_example(input, label);
            prompt.push_str(&formatted);
            prompt.push('\n');
        }

        // Append the query (without the answer).
        let query_line = self.format_query(query);
        prompt.push_str(&query_line);
        prompt
    }

    fn format_example(&self, input: &str, label: &str) -> String {
        match &self.template {
            IclTemplate::QA {
                question_prefix,
                answer_prefix,
            } => {
                format!("{}{}\n{}{}", question_prefix, input, answer_prefix, label)
            },
            IclTemplate::SentimentAnalysis {
                label_positive,
                label_negative,
            } => {
                let sentiment = if label == "positive" || label == label_positive {
                    label_positive.as_str()
                } else {
                    label_negative.as_str()
                };
                format!("Review: {}\nSentiment: {}", input, sentiment)
            },
            IclTemplate::Custom { format_fn } => {
                format_fn.replace("{input}", input).replace("{label}", label)
            },
        }
    }

    fn format_query(&self, query: &str) -> String {
        match &self.template {
            IclTemplate::QA {
                question_prefix,
                answer_prefix,
            } => {
                format!("{}{}\n{}", question_prefix, query, answer_prefix)
            },
            IclTemplate::SentimentAnalysis { .. } => {
                format!("Review: {}\nSentiment:", query)
            },
            IclTemplate::Custom { format_fn } => {
                // Only replace {input}; leave {label} blank for the query.
                format_fn.replace("{input}", query).replace("{label}", "")
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(dim: usize, n_way: usize, k_shot: usize, n_query: usize) -> PrototypicalConfig {
        PrototypicalConfig::new(dim, n_way, k_shot, n_query)
    }

    // --- euclidean_distance -------------------------------------------------

    #[test]
    fn test_euclidean_distance_zero() {
        let a = vec![1.0_f32, 2.0, 3.0];
        assert!((PrototypicalNetworks::euclidean_distance(&a, &a)).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_known() {
        // ||[0,0] - [3,4]|| = 5.0
        let a = vec![0.0_f32, 0.0];
        let b = vec![3.0_f32, 4.0];
        let d = PrototypicalNetworks::euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-5, "Expected 5.0, got {}", d);
    }

    // --- compute_prototypes -------------------------------------------------

    #[test]
    fn test_compute_prototypes_mean() {
        let net = PrototypicalNetworks::new(make_config(2, 2, 2, 1));
        // Class 0: [1,0] + [3,0] → mean [2,0]
        // Class 1: [0,1] + [0,3] → mean [0,2]
        let embeddings = vec![
            vec![1.0_f32, 0.0],
            vec![3.0_f32, 0.0],
            vec![0.0_f32, 1.0],
            vec![0.0_f32, 3.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let protos = net.compute_prototypes(&embeddings, &labels).expect("should succeed");
        assert_eq!(protos.len(), 2);
        assert!((protos[0][0] - 2.0).abs() < 1e-5);
        assert!((protos[0][1] - 0.0).abs() < 1e-5);
        assert!((protos[1][0] - 0.0).abs() < 1e-5);
        assert!((protos[1][1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_prototypes_label_out_of_range() {
        let net = PrototypicalNetworks::new(make_config(2, 2, 1, 1));
        let embeddings = vec![vec![1.0_f32, 0.0]];
        let labels = vec![5]; // out of range
        assert!(net.compute_prototypes(&embeddings, &labels).is_err());
    }

    #[test]
    fn test_compute_prototypes_missing_class() {
        let net = PrototypicalNetworks::new(make_config(2, 2, 1, 1));
        // Only class 0, no class 1 support.
        let embeddings = vec![vec![1.0_f32, 0.0]];
        let labels = vec![0];
        assert!(net.compute_prototypes(&embeddings, &labels).is_err());
    }

    // --- classify_queries ---------------------------------------------------

    #[test]
    fn test_classify_queries_nearest_prototype() {
        let net = PrototypicalNetworks::new(make_config(2, 2, 2, 2));
        // Prototype 0 at [10, 0], prototype 1 at [-10, 0].
        let prototypes = vec![vec![10.0_f32, 0.0], vec![-10.0_f32, 0.0]];
        let queries = vec![
            vec![9.0_f32, 0.0],  // nearest to class 0
            vec![-9.0_f32, 0.0], // nearest to class 1
        ];
        let preds = net.classify_queries(&queries, &prototypes).expect("should succeed");
        assert_eq!(preds, vec![0, 1]);
    }

    #[test]
    fn test_classify_queries_empty_queries() {
        let net = PrototypicalNetworks::new(make_config(2, 2, 2, 1));
        let protos = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
        assert!(net.classify_queries(&[], &protos).is_err());
    }

    // --- episode_loss -------------------------------------------------------

    #[test]
    fn test_episode_loss_perfect_predictions() {
        let net = PrototypicalNetworks::new(make_config(1, 2, 1, 2));
        // Two prototypes very far apart.
        let prototypes = vec![vec![1000.0_f32], vec![-1000.0_f32]];
        // Queries: one near class 0, one near class 1.
        let queries = vec![vec![999.0_f32], vec![-999.0_f32]];
        let labels = vec![0, 1];
        let loss = net.episode_loss(&queries, &labels, &prototypes).expect("should succeed");
        // With perfect separation, loss should be very close to 0.
        assert!(
            loss < 0.01,
            "Loss should be near 0 for perfect separation, got {}",
            loss
        );
    }

    #[test]
    fn test_episode_loss_mismatch_lengths() {
        let net = PrototypicalNetworks::new(make_config(1, 2, 1, 2));
        let protos = vec![vec![0.0_f32], vec![1.0]];
        let queries = vec![vec![0.0_f32], vec![1.0]];
        let labels = vec![0]; // length mismatch
        assert!(net.episode_loss(&queries, &labels, &protos).is_err());
    }

    // --- MatchingNetworks ---------------------------------------------------

    #[test]
    fn test_attention_kernel_sums_to_one() {
        let support = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0], vec![1.0_f32, 1.0]];
        let query = vec![0.5_f32, 0.5];
        let attn = MatchingNetworks::attention_kernel(&query, &support);
        let sum: f32 = attn.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Attention weights should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_attention_kernel_nearest_gets_most_weight() {
        // Query at [0,0], support[0] at [0,0], support[1] at [100,100].
        let support = vec![vec![0.0_f32, 0.0], vec![100.0_f32, 100.0]];
        let query = vec![0.0_f32, 0.0];
        let attn = MatchingNetworks::attention_kernel(&query, &support);
        assert!(attn[0] > attn[1], "Nearest support should get more weight");
    }

    #[test]
    fn test_matching_classify_returns_class_probs() {
        let mn = MatchingNetworks {
            embedding_dim: 2,
            n_way: 2,
            k_shot: 2,
        };
        let support = vec![
            vec![1.0_f32, 0.0],
            vec![1.1_f32, 0.1],
            vec![0.0_f32, 1.0],
            vec![0.1_f32, 1.1],
        ];
        let labels = vec![0, 0, 1, 1];
        let query = vec![1.0_f32, 0.0];
        let probs = mn.classify(&query, &support, &labels).expect("should succeed");
        assert_eq!(probs.len(), 2);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Class probs should sum to 1, got {}",
            sum
        );
        assert!(
            probs[0] > probs[1],
            "Class 0 should have higher probability"
        );
    }

    #[test]
    fn test_matching_classify_empty_support() {
        let mn = MatchingNetworks {
            embedding_dim: 2,
            n_way: 2,
            k_shot: 1,
        };
        let result = mn.classify(&[0.0, 0.0], &[], &[]);
        assert!(result.is_err());
    }

    // --- IclEpisodeBuilder --------------------------------------------------

    #[test]
    fn test_icl_qa_prompt_structure() {
        let builder = IclEpisodeBuilder::new(
            2,
            IclTemplate::QA {
                question_prefix: "Q: ".to_string(),
                answer_prefix: "A: ".to_string(),
            },
        );
        let demos = vec![
            ("What is 2+2?".to_string(), "4".to_string()),
            ("What is 3+3?".to_string(), "6".to_string()),
        ];
        let prompt = builder.build_prompt(&demos, "What is 4+4?");
        assert!(prompt.contains("Q: What is 2+2?"));
        assert!(prompt.contains("A: 4"));
        assert!(prompt.contains("Q: What is 4+4?"));
        // Query should NOT contain the answer.
        assert!(!prompt.ends_with("8"));
    }

    #[test]
    fn test_icl_custom_template() {
        let builder = IclEpisodeBuilder::new(
            1,
            IclTemplate::Custom {
                format_fn: "Input: {input} -> {label}".to_string(),
            },
        );
        let demos = vec![("hello".to_string(), "greeting".to_string())];
        let prompt = builder.build_prompt(&demos, "bye");
        assert!(prompt.contains("Input: hello -> greeting"));
        assert!(prompt.contains("Input: bye ->"));
    }

    #[test]
    fn test_icl_limits_to_n_shot() {
        let builder = IclEpisodeBuilder::new(
            1, // only 1 shot
            IclTemplate::QA {
                question_prefix: "Q: ".to_string(),
                answer_prefix: "A: ".to_string(),
            },
        );
        let demos = vec![
            ("q1".to_string(), "a1".to_string()),
            ("q2".to_string(), "a2".to_string()),
            ("q3".to_string(), "a3".to_string()),
        ];
        let prompt = builder.build_prompt(&demos, "q4");
        // Only first demo should appear.
        assert!(prompt.contains("q1"));
        assert!(!prompt.contains("q2"));
    }

    #[test]
    fn test_icl_sentiment_template() {
        let builder = IclEpisodeBuilder::new(
            2,
            IclTemplate::SentimentAnalysis {
                label_positive: "POSITIVE".to_string(),
                label_negative: "NEGATIVE".to_string(),
            },
        );
        let demos = vec![
            ("Great movie!".to_string(), "positive".to_string()),
            ("Terrible film.".to_string(), "NEGATIVE".to_string()),
        ];
        let prompt = builder.build_prompt(&demos, "It was okay.");
        assert!(prompt.contains("Sentiment: POSITIVE"));
        assert!(prompt.contains("Sentiment: NEGATIVE"));
        assert!(prompt.contains("Sentiment:"));
    }

    // ── Prototypical: cosine distance variant ────────────────────────────────

    /// Cosine distance = 1 - cosine_similarity. Nearest cosine prototype
    /// should agree with nearest Euclidean prototype on unit-length vectors.
    #[test]
    fn test_cosine_distance_on_unit_vectors() {
        // For unit vectors ||a||=||b||=1: ||a-b||^2 = 2*(1 - a·b).
        // So argmin Euclidean dist == argmax cosine similarity on unit vectors.
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        // Normalise.
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let ua: Vec<f32> = a.iter().map(|x| x / norm_a).collect();
        let ub: Vec<f32> = b.iter().map(|x| x / norm_b).collect();

        let query = vec![0.9_f32, 0.1]; // closer to a

        let cos_sim_ua: f32 = query.iter().zip(ua.iter()).map(|(x, y)| x * y).sum();
        let cos_sim_ub: f32 = query.iter().zip(ub.iter()).map(|(x, y)| x * y).sum();
        let dist_ua = PrototypicalNetworks::euclidean_distance(&query, &ua);
        let dist_ub = PrototypicalNetworks::euclidean_distance(&query, &ub);

        // Both metrics should agree: a is closer.
        assert!(cos_sim_ua > cos_sim_ub, "cos: a should be closer to query");
        assert!(dist_ua < dist_ub, "euclid: a should be closer to query");
    }

    // ── Prototypical: episode accuracy ───────────────────────────────────────

    /// Episode accuracy = fraction of queries correctly classified.
    #[test]
    fn test_episode_accuracy_perfect() {
        let net = PrototypicalNetworks::new(make_config(2, 3, 1, 3));
        // 3-way 1-shot: one support per class.
        let support = vec![
            vec![10.0_f32, 0.0],
            vec![0.0_f32, 10.0],
            vec![-10.0_f32, 0.0],
        ];
        let support_labels = vec![0, 1, 2];
        let protos = net.compute_prototypes(&support, &support_labels).expect("compute_prototypes");

        // Queries clearly near their respective prototypes.
        let queries = vec![vec![9.0_f32, 0.0], vec![0.0_f32, 9.0], vec![-9.0_f32, 0.0]];
        let true_labels = [0usize, 1, 2];
        let preds = net.classify_queries(&queries, &protos).expect("classify_queries");
        let correct = preds.iter().zip(true_labels.iter()).filter(|(p, t)| p == t).count();
        let accuracy = correct as f32 / queries.len() as f32;
        assert!(
            (accuracy - 1.0).abs() < 1e-6,
            "perfect episode should give accuracy=1.0, got {accuracy}"
        );
    }

    // ── Prototypical: 5-way 5-shot episode ──────────────────────────────────

    #[test]
    fn test_5_way_5_shot_prototype_shapes() {
        let n_way = 5;
        let k_shot = 5;
        let dim = 4;
        let net = PrototypicalNetworks::new(make_config(dim, n_way, k_shot, 2));

        // LCG to generate deterministic embeddings.
        let mut state = 42u64;
        let mut lcg = || -> f32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state as i64 as f32) / i64::MAX as f32
        };

        let mut support = Vec::new();
        let mut labels = Vec::new();
        for class in 0..n_way {
            for _ in 0..k_shot {
                support.push((0..dim).map(|_| lcg()).collect::<Vec<f32>>());
                labels.push(class);
            }
        }

        let protos = net.compute_prototypes(&support, &labels).expect("should succeed");
        assert_eq!(protos.len(), n_way, "should have one prototype per class");
        for p in &protos {
            assert_eq!(
                p.len(),
                dim,
                "each prototype should have embedding_dim dimensions"
            );
        }
    }

    // ── Prototypical: empty support set error ────────────────────────────────

    #[test]
    fn test_compute_prototypes_empty_support() {
        let net = PrototypicalNetworks::new(make_config(2, 2, 1, 1));
        let result = net.compute_prototypes(&[], &[]);
        assert!(result.is_err(), "empty support should return error");
    }

    // ── Prototypical: dim mismatch in support ────────────────────────────────

    #[test]
    fn test_compute_prototypes_dim_mismatch() {
        let net = PrototypicalNetworks::new(make_config(3, 2, 1, 1));
        // Support embedding has wrong dim (2 instead of 3).
        let support = vec![vec![1.0_f32, 0.0]]; // dim=2, expected 3
        let labels = vec![0];
        assert!(
            net.compute_prototypes(&support, &labels).is_err(),
            "dim mismatch should be rejected"
        );
    }

    // ── Prototypical: episode_loss label out of range ────────────────────────

    #[test]
    fn test_episode_loss_label_out_of_range() {
        let net = PrototypicalNetworks::new(make_config(2, 2, 1, 1));
        let protos = vec![vec![0.0_f32, 0.0], vec![1.0, 1.0]];
        let queries = vec![vec![0.0_f32, 0.0]];
        let labels = vec![5usize]; // out of range for 2-class
        assert!(net.episode_loss(&queries, &labels, &protos).is_err());
    }

    // ── Matching Networks: empty support ────────────────────────────────────

    #[test]
    fn test_attention_kernel_empty_support() {
        let attn = MatchingNetworks::attention_kernel(&[1.0_f32, 0.0], &[]);
        assert!(attn.is_empty(), "empty support should give empty attention");
    }

    // ── Matching Networks: uniform attention for equidistant support ─────────

    #[test]
    fn test_attention_kernel_equidistant_uniform() {
        // Four support points equally distant from the origin query.
        let support = vec![
            vec![1.0_f32, 0.0],
            vec![-1.0_f32, 0.0],
            vec![0.0_f32, 1.0],
            vec![0.0_f32, -1.0],
        ];
        let query = vec![0.0_f32, 0.0]; // equidistant from all
        let attn = MatchingNetworks::attention_kernel(&query, &support);
        assert_eq!(attn.len(), 4);
        for (i, &w) in attn.iter().enumerate() {
            assert!(
                (w - 0.25).abs() < 1e-5,
                "equidistant support should give uniform attention, attn[{i}]={w}"
            );
        }
    }

    // ── Matching Networks: label out of range ────────────────────────────────

    #[test]
    fn test_matching_classify_label_out_of_range() {
        let mn = MatchingNetworks {
            embedding_dim: 2,
            n_way: 2,
            k_shot: 1,
        };
        let support = vec![vec![1.0_f32, 0.0]];
        let labels = vec![5usize]; // out of range for n_way=2
        let result = mn.classify(&[1.0, 0.0], &support, &labels);
        assert!(result.is_err(), "label out of range should error");
    }

    // ── Matching Networks: length mismatch ───────────────────────────────────

    #[test]
    fn test_matching_classify_length_mismatch() {
        let mn = MatchingNetworks {
            embedding_dim: 2,
            n_way: 2,
            k_shot: 1,
        };
        let support = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let labels = vec![0usize]; // length 1, but support has 2
        let result = mn.classify(&[1.0, 0.0], &support, &labels);
        assert!(result.is_err(), "length mismatch should error");
    }

    // ── ICL: zero-shot (no demonstrations) ───────────────────────────────────

    #[test]
    fn test_icl_zero_shot_prompt() {
        let builder = IclEpisodeBuilder::new(
            0, // zero-shot: no demonstrations used
            IclTemplate::QA {
                question_prefix: "Q: ".to_string(),
                answer_prefix: "A: ".to_string(),
            },
        );
        let prompt = builder.build_prompt(
            &[("ignored".to_string(), "ignored".to_string())],
            "What is 1+1?",
        );
        // Zero-shot: no demonstration should appear.
        assert!(
            !prompt.contains("ignored"),
            "zero-shot prompt should not contain demos"
        );
        assert!(
            prompt.contains("What is 1+1?"),
            "query should appear in prompt"
        );
    }

    // ── ICL: custom template with no label placeholder in query ──────────────

    #[test]
    fn test_icl_custom_query_strips_label() {
        let builder = IclEpisodeBuilder::new(
            1,
            IclTemplate::Custom {
                format_fn: "{input} => {label}".to_string(),
            },
        );
        let demos = vec![("hello".to_string(), "world".to_string())];
        let prompt = builder.build_prompt(&demos, "test_input");
        // The query line should have {label} replaced with "" (empty).
        assert!(
            prompt.contains("test_input =>"),
            "query part should strip label placeholder"
        );
        assert!(
            prompt.contains("hello => world"),
            "demo should be formatted correctly"
        );
    }
}
