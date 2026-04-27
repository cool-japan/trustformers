//! Evaluation metrics for NLP tasks.
//!
//! This module provides production-quality implementations of:
//!
//! - [`metrics::bleu_score`] — BLEU (Papineni et al., 2002)
//! - [`metrics::ngram_precision`] — individual n-gram precision component
//! - [`metrics::rouge_n_score`] — ROUGE-N F1
//! - [`metrics::rouge_l_score`] — ROUGE-L (LCS-based)
//! - [`metrics::token_f1_score`] — token-level F1 for QA / NER
//! - [`metrics::exact_match`] / [`metrics::exact_match_score`] — exact string match
//! - [`metrics::perplexity`] — exponentiated cross-entropy
//!
//! All functions return `Result<_, MetricError>` — no panics.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use trustformers::evaluation::{bleu_score, exact_match, perplexity, NlpBenchmarkResult};
//!
//! // BLEU
//! let hyp = &["the", "cat", "sat", "on", "the", "mat"];
//! let refs: &[&[&str]] = &[&["the", "cat", "is", "on", "the", "mat"]];
//! let bleu = bleu_score(hyp, refs, 4, true).unwrap();
//!
//! // Exact match
//! assert!(exact_match("Paris", "paris"));
//!
//! // Perplexity
//! let ppl = perplexity(&vec![-2.3_f64; 10]).unwrap();
//!
//! // Aggregated result
//! let mut bench = NlpBenchmarkResult::new();
//! bench.bleu_4 = Some(bleu);
//! bench.perplexity = Some(ppl);
//! println!("{}", bench.display_summary());
//! ```

pub mod bridge;
pub mod metrics;

pub use metrics::{
    bleu_score, exact_match, exact_match_score, ngram_precision, perplexity, rouge_l_score,
    rouge_n_score, token_f1_score, F1Score, MetricError, NlpBenchmarkResult, RougeScore,
};
