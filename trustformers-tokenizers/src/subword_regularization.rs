use scirs2_core::random::*; // SciRS2 Integration Policy - Replaces rand
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Configuration for subword regularization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubwordRegularizationConfig {
    /// Alpha parameter for controlling randomness (0.0 = no randomness, 1.0 = maximum randomness).
    /// For [`SubwordRegularizer`] this is the probability, per word-internal
    /// character boundary, that the word is split there before re-encoding.
    /// For [`UnigramSubwordRegularizer`] this is a smoothing temperature
    /// applied to piece log-scores before sampling (SentencePiece's own
    /// `--alpha`): values near `0.0` concentrate sampling on the
    /// highest-scoring (Viterbi) segmentation, larger values flatten the
    /// distribution.
    pub alpha: f32,
    /// Number of alternative segmentations to sample
    pub num_samples: usize,
    /// Seed for reproducible randomness
    pub seed: Option<u64>,
    /// Enable debugging output
    pub debug: bool,
}

impl Default for SubwordRegularizationConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            num_samples: 1,
            seed: None,
            debug: false,
        }
    }
}

/// Subword regularization wrapper that samples alternative *segmentations*
/// of the same text through an arbitrary [`Tokenizer`].
///
/// True Kudo (2018) subword regularization samples from a unigram
/// language-model lattice, which needs per-piece scores; the [`Tokenizer`]
/// trait this type wraps is intentionally generic (BPE, WordPiece,
/// character-level, ...) and exposes no such scores. Instead, this samples
/// random *word-internal split points* (with per-boundary probability
/// `alpha`) and encodes each resulting fragment independently through the
/// real tokenizer, then stitches the fragment encodings back together. This
/// changes the resulting token IDs whenever the wrapped tokenizer's
/// segmentation is sensitive to where a word is split (true of any merge- or
/// longest-match-based subword tokenizer) while the source *text* is never
/// altered: the fragments always concatenate back to the original input.
///
/// For a tokenizer that does expose per-piece scores, prefer
/// [`UnigramSubwordRegularizer`], which samples directly from the unigram
/// lattice instead of relying on split-point sensitivity.
pub struct SubwordRegularizer<T: Tokenizer> {
    tokenizer: T,
    config: SubwordRegularizationConfig,
    rng: StdRng,
}

impl<T: Tokenizer> SubwordRegularizer<T> {
    pub fn new(tokenizer: T, config: SubwordRegularizationConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            // Generate random seed from thread_rng
            let seed = thread_rng().random();
            StdRng::seed_from_u64(seed)
        };

        Self {
            tokenizer,
            config,
            rng,
        }
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.config.alpha = alpha;
        self
    }

    pub fn with_num_samples(mut self, num_samples: usize) -> Self {
        self.config.num_samples = num_samples;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Generate multiple tokenizations with regularization.
    ///
    /// Each sample re-splits `text` at random word-internal boundaries (see
    /// the type-level docs) and re-encodes the resulting fragments through
    /// the wrapped tokenizer; the source text itself is never modified.
    pub fn encode_with_regularization(&mut self, text: &str) -> Result<Vec<TokenizedInput>> {
        // The token(s) this tokenizer always wraps around content (e.g. a
        // leading BOS/CLS, a trailing EOS/SEP), measured directly from the
        // tokenizer's own behavior rather than assumed, so stitching
        // fragments back together does not duplicate them at every join.
        let (prefix, suffix) = self.detect_envelope(text)?;

        let mut results = Vec::with_capacity(self.config.num_samples);
        for _ in 0..self.config.num_samples {
            let fragments = self.sample_fragments(text);
            results.push(self.encode_fragments(&fragments, &prefix, &suffix)?);
        }

        Ok(results)
    }

    /// Encode a list of text fragments (whose concatenation must equal the
    /// original text) through the wrapped tokenizer, stripping the
    /// tokenizer's own envelope tokens from every fragment join so they
    /// appear exactly once each, at the very start and end of the result.
    fn encode_fragments(
        &self,
        fragments: &[String],
        prefix: &[u32],
        suffix: &[u32],
    ) -> Result<TokenizedInput> {
        let mut input_ids = Vec::new();
        let last = fragments.len().saturating_sub(1);

        for (i, fragment) in fragments.iter().enumerate() {
            if fragment.is_empty() {
                continue;
            }
            let mut ids = self.tokenizer.encode(fragment)?.input_ids;
            if i != 0 {
                strip_prefix_ids(&mut ids, prefix);
            }
            if i != last {
                strip_suffix_ids(&mut ids, suffix);
            }
            input_ids.extend(ids);
        }

        // No padding is ever introduced here, so the attention mask is
        // simply "all real tokens".
        let attention_mask = vec![1u8; input_ids.len()];

        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    /// Detect the token IDs this tokenizer always wraps around any content,
    /// by comparing `encode("")` (envelope only) against `encode(text)`
    /// (envelope + content): their common prefix/suffix is the envelope.
    fn detect_envelope(&self, text: &str) -> Result<(Vec<u32>, Vec<u32>)> {
        let empty = self.tokenizer.encode("")?.input_ids;
        if empty.is_empty() || text.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let whole = self.tokenizer.encode(text)?.input_ids;

        let prefix_len = empty.iter().zip(whole.iter()).take_while(|(a, b)| a == b).count();
        let remaining = empty.len() - prefix_len;
        let suffix_len = empty
            .iter()
            .rev()
            .zip(whole.iter().rev())
            .take_while(|(a, b)| a == b)
            .count()
            // Don't let a short envelope be double-counted as both prefix
            // and suffix.
            .min(remaining);

        Ok((
            empty[..prefix_len].to_vec(),
            empty[empty.len() - suffix_len..].to_vec(),
        ))
    }

    /// Split `text` into fragments whose concatenation is exactly `text`,
    /// sampling random word-internal split points with per-boundary
    /// probability `self.config.alpha`. Whitespace runs are always kept
    /// intact as their own fragment (splitting inside whitespace has no
    /// bearing on subword segmentation).
    fn sample_fragments(&mut self, text: &str) -> Vec<String> {
        if self.config.alpha <= 0.0 {
            return vec![text.to_string()];
        }

        let mut fragments = Vec::new();
        for word in split_keeping_whitespace(text) {
            let char_count = word.chars().count();
            if char_count < 2 || word.chars().all(char::is_whitespace) {
                fragments.push(word);
                continue;
            }

            let chars: Vec<char> = word.chars().collect();
            let mut start = 0;
            for i in 1..chars.len() {
                if self.rng.random::<f32>() < self.config.alpha {
                    fragments.push(chars[start..i].iter().collect());
                    start = i;
                }
            }
            fragments.push(chars[start..].iter().collect());
        }
        fragments
    }

    /// Get the underlying tokenizer
    pub fn inner(&self) -> &T {
        &self.tokenizer
    }

    /// Get the configuration
    pub fn config(&self) -> &SubwordRegularizationConfig {
        &self.config
    }
}

/// Remove `prefix` from the front of `ids` if it appears there exactly.
fn strip_prefix_ids(ids: &mut Vec<u32>, prefix: &[u32]) {
    if !prefix.is_empty() && ids.len() >= prefix.len() && ids[..prefix.len()] == *prefix {
        ids.drain(..prefix.len());
    }
}

/// Remove `suffix` from the back of `ids` if it appears there exactly.
fn strip_suffix_ids(ids: &mut Vec<u32>, suffix: &[u32]) {
    if !suffix.is_empty() && ids.len() >= suffix.len() && ids[ids.len() - suffix.len()..] == *suffix
    {
        ids.truncate(ids.len() - suffix.len());
    }
}

/// Split `text` into maximal runs of whitespace / non-whitespace characters.
/// The concatenation of the returned fragments is always exactly `text`.
fn split_keeping_whitespace(text: &str) -> Vec<String> {
    let mut fragments = Vec::new();
    let mut current = String::new();
    let mut current_is_ws: Option<bool> = None;

    for ch in text.chars() {
        let is_ws = ch.is_whitespace();
        if current_is_ws == Some(is_ws) {
            current.push(ch);
        } else {
            if !current.is_empty() {
                fragments.push(std::mem::take(&mut current));
            }
            current.push(ch);
            current_is_ws = Some(is_ws);
        }
    }
    if !current.is_empty() {
        fragments.push(current);
    }
    fragments
}

impl<T: Tokenizer> Tokenizer for SubwordRegularizer<T> {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        // For the basic interface, just use the underlying tokenizer
        self.tokenizer.encode(text)
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        self.tokenizer.encode_pair(text, text2)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids)
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.tokenizer.get_vocab()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}

/// Unigram-specific subword regularization implementation.
///
/// Implements Kudo (2018) subword regularization via forward-filtering,
/// backward-sampling (FFBS) over the unigram segmentation lattice:
///
/// 1. **Forward filtering**: for every prefix length `e`, compute the
///    log-sum-exp ("soft-max") over every valid piece ending at `e` of
///    `log_alpha[start] + score(piece) / temperature` — the log of the total
///    probability mass of every segmentation of `text[0..e]`.
/// 2. **Backward sampling**: starting at the end of the text, repeatedly
///    sample a preceding split point with probability proportional to its
///    share of the current position's log-sum-exp mass, walking back to
///    position 0.
///
/// This samples a genuinely different, always text-preserving segmentation
/// on each call (when `temperature > 0`), weighted by the real piece
/// scores — as opposed to perturbing the text itself.
pub struct UnigramSubwordRegularizer {
    vocab: HashMap<String, f32>,
    config: SubwordRegularizationConfig,
    rng: StdRng,
}

impl UnigramSubwordRegularizer {
    pub fn new(vocab: HashMap<String, f32>, config: SubwordRegularizationConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            // Generate random seed from thread_rng
            let seed = thread_rng().random();
            StdRng::seed_from_u64(seed)
        };

        Self { vocab, config, rng }
    }

    /// Sample one segmentation of `text` from the unigram lattice via
    /// forward-filtering backward-sampling. Returns an error if some
    /// substring of `text` has no vocabulary coverage at all (no valid
    /// segmentation reaches the end of the text), rather than silently
    /// returning a partial or empty result.
    pub fn sample_segmentation(&mut self, text: &str) -> Result<Vec<String>> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();

        if n == 0 {
            return Ok(vec![]);
        }

        // A smoothing temperature: values near 0 concentrate the forward
        // pass (and therefore sampling) on the single highest-scoring
        // (Viterbi) segmentation; larger values flatten the distribution.
        // This mirrors SentencePiece's own `--alpha` semantics.
        let temperature = self.config.alpha.max(1e-4);

        // log_alpha[e] = logsumexp over every valid piece ending at `e` of
        // (log_alpha[start] + score(piece) / temperature). `NEG_INFINITY`
        // marks a position with no valid segmentation reaching it yet; every
        // value this algorithm produces is otherwise finite (never NaN or
        // +infinity), so `is_finite()` is an exact "is reachable" test.
        let mut log_alpha = vec![f32::NEG_INFINITY; n + 1];
        log_alpha[0] = 0.0;
        // For each end position, the `(start, weighted_score)` pairs that
        // contributed to it, needed again during backward sampling.
        let mut incoming: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n + 1];

        for end in 1..=n {
            for start in 0..end {
                if !log_alpha[start].is_finite() {
                    continue;
                }
                let piece: String = chars[start..end].iter().collect();
                if let Some(&score) = self.vocab.get(&piece) {
                    let weighted = log_alpha[start] + score / temperature;
                    incoming[end].push((start, weighted));
                    log_alpha[end] = log_sum_exp(log_alpha[end], weighted);
                }
            }
        }

        if !log_alpha[n].is_finite() {
            return Err(TrustformersError::invalid_input(format!(
                "No valid unigram segmentation exists for {:?}: some substring has no \
                 vocabulary coverage (consider adding single-character fallback entries)",
                text
            )));
        }

        let mut pieces = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let candidates = &incoming[pos];
            let start = self.sample_start(candidates, log_alpha[pos]);
            pieces.push(chars[start..pos].iter().collect::<String>());
            pos = start;
        }
        pieces.reverse();
        Ok(pieces)
    }

    /// Sample one `start` position from `candidates` (each a `(start,
    /// weighted_score)` pair for a piece ending at the same position), with
    /// probability proportional to `exp(weighted_score - log_norm)`.
    fn sample_start(&mut self, candidates: &[(usize, f32)], log_norm: f32) -> usize {
        if candidates.len() == 1 {
            return candidates[0].0;
        }

        let weights: Vec<f32> = candidates.iter().map(|&(_, w)| (w - log_norm).exp()).collect();
        let total: f32 = weights.iter().sum();
        let mut threshold = self.rng.random::<f32>() * total;
        for (i, &w) in weights.iter().enumerate() {
            if threshold < w {
                return candidates[i].0;
            }
            threshold -= w;
        }
        // Floating-point rounding at the boundary; fall back to the last
        // candidate rather than panicking.
        candidates[candidates.len() - 1].0
    }
}

/// Numerically stable `ln(exp(a) + exp(b))`. Treats a non-finite input as
/// "contributes zero probability mass" (this module only ever produces
/// `NEG_INFINITY` as a non-finite value, never `NaN` or `+INFINITY`).
fn log_sum_exp(a: f32, b: f32) -> f32 {
    if !a.is_finite() {
        return b;
    }
    if !b.is_finite() {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::char::CharTokenizer;

    /// Minimal greedy longest-match tokenizer, defined only for these tests.
    /// Unlike `CharTokenizer` (which always tokenizes one character at a
    /// time, so its output can never depend on where a word is split), this
    /// tokenizer's segmentation genuinely depends on split points, letting
    /// the tests demonstrate that fragment-boundary sampling changes the
    /// resulting token IDs.
    #[derive(Clone)]
    struct GreedyTestTokenizer {
        vocab: HashMap<String, u32>,
    }

    impl GreedyTestTokenizer {
        fn new(pairs: &[(&str, u32)]) -> Self {
            Self {
                vocab: pairs.iter().map(|&(s, id)| (s.to_string(), id)).collect(),
            }
        }

        fn greedy_ids(&self, text: &str) -> Vec<u32> {
            let chars: Vec<char> = text.chars().collect();
            let mut ids = Vec::new();
            let mut i = 0;
            while i < chars.len() {
                let mut matched = None;
                for end in (i + 1..=chars.len()).rev() {
                    let candidate: String = chars[i..end].iter().collect();
                    if let Some(&id) = self.vocab.get(&candidate) {
                        matched = Some((id, end));
                        break;
                    }
                }
                match matched {
                    Some((id, end)) => {
                        ids.push(id);
                        i = end;
                    },
                    None => {
                        // Never panic on uncovered input; use an
                        // out-of-band id derived from the character.
                        ids.push(1_000_000 + chars[i] as u32);
                        i += 1;
                    },
                }
            }
            ids
        }
    }

    impl Tokenizer for GreedyTestTokenizer {
        fn encode(&self, text: &str) -> Result<TokenizedInput> {
            let input_ids = self.greedy_ids(text);
            let attention_mask = vec![1u8; input_ids.len()];
            Ok(TokenizedInput {
                input_ids,
                attention_mask,
                token_type_ids: None,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            })
        }

        fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
            let mut combined = self.encode(text)?;
            let second = self.encode(text2)?;
            combined.input_ids.extend(second.input_ids);
            combined.attention_mask.extend(second.attention_mask);
            Ok(combined)
        }

        fn decode(&self, _ids: &[u32]) -> Result<String> {
            Ok(String::new())
        }

        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }

        fn get_vocab(&self) -> HashMap<String, u32> {
            self.vocab.clone()
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.vocab.get(token).copied()
        }

        fn id_to_token(&self, _id: u32) -> Option<String> {
            None
        }
    }

    fn greedy_tokenizer() -> GreedyTestTokenizer {
        GreedyTestTokenizer::new(&[
            ("hello", 1),
            ("he", 2),
            ("llo", 3),
            ("l", 4),
            ("o", 5),
            ("hel", 6),
            ("lo", 7),
            ("h", 8),
            ("e", 9),
        ])
    }

    #[test]
    fn test_subword_regularization_config() {
        let config = SubwordRegularizationConfig::default();
        assert_eq!(config.alpha, 0.1);
        assert_eq!(config.num_samples, 1);
        assert_eq!(config.seed, None);
        assert!(!config.debug);
    }

    #[test]
    fn test_subword_regularizer_creation() {
        let tokenizer = CharTokenizer::from_text("hello world", 1000);
        let config = SubwordRegularizationConfig::default();
        let regularizer = SubwordRegularizer::new(tokenizer, config);

        assert_eq!(regularizer.config().alpha, 0.1);
        assert_eq!(regularizer.config().num_samples, 1);
    }

    #[test]
    fn test_subword_regularizer_encode() {
        let tokenizer = CharTokenizer::from_text("hello world", 1000);
        let config = SubwordRegularizationConfig::default();
        let regularizer = SubwordRegularizer::new(tokenizer, config);

        let result = regularizer.encode("hello");
        assert!(result.is_ok());

        let tokenized = result.expect("Operation failed in test");
        assert!(!tokenized.input_ids.is_empty());
    }

    #[test]
    fn test_subword_regularizer_with_seed() {
        let tokenizer = CharTokenizer::from_text("hello world", 1000);
        let config = SubwordRegularizationConfig::default();
        let mut regularizer = SubwordRegularizer::new(tokenizer, config).with_seed(42);

        let result1 = regularizer.encode_with_regularization("hello world");
        assert!(result1.is_ok());

        // Reset with same seed
        let tokenizer2 = CharTokenizer::from_text("hello world", 1000);
        let config2 = SubwordRegularizationConfig::default();
        let mut regularizer2 = SubwordRegularizer::new(tokenizer2, config2).with_seed(42);

        let result2 = regularizer2.encode_with_regularization("hello world");
        assert!(result2.is_ok());
    }

    #[test]
    fn test_subword_regularizer_multiple_samples() {
        let tokenizer = CharTokenizer::from_text("hello world", 1000);
        let config = SubwordRegularizationConfig::default();
        let mut regularizer =
            SubwordRegularizer::new(tokenizer, config).with_num_samples(3).with_alpha(0.2);

        let results = regularizer.encode_with_regularization("hello world");
        assert!(results.is_ok());

        let tokenized_results = results.expect("Operation failed in test");
        assert_eq!(tokenized_results.len(), 3);

        for result in tokenized_results {
            assert!(!result.input_ids.is_empty());
        }
    }

    #[test]
    fn test_unigram_subword_regularizer() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 1.0);
        vocab.insert("world".to_string(), 1.0);
        vocab.insert("h".to_string(), 0.5);
        vocab.insert("e".to_string(), 0.5);
        vocab.insert("l".to_string(), 0.5);
        vocab.insert("o".to_string(), 0.5);

        let config = SubwordRegularizationConfig::default();
        let mut regularizer = UnigramSubwordRegularizer::new(vocab, config);

        let result = regularizer.sample_segmentation("hello");
        assert!(result.is_ok());

        let segmentation = result.expect("Operation failed in test");
        assert!(!segmentation.is_empty());
    }

    #[test]
    fn test_unigram_regularizer_with_alpha() {
        let mut vocab = HashMap::new();
        vocab.insert("test".to_string(), 1.0);
        vocab.insert("t".to_string(), 0.3);
        vocab.insert("e".to_string(), 0.3);
        vocab.insert("s".to_string(), 0.3);

        let config = SubwordRegularizationConfig {
            alpha: 0.5,
            num_samples: 1,
            seed: Some(123),
            debug: false,
        };

        let mut regularizer = UnigramSubwordRegularizer::new(vocab, config);

        let result1 = regularizer.sample_segmentation("test");
        assert!(result1.is_ok());

        // Results should be different due to regularization
        let result2 = regularizer.sample_segmentation("test");
        assert!(result2.is_ok());
    }

    #[test]
    fn test_regularization_config_serialization() {
        let config = SubwordRegularizationConfig {
            alpha: 0.3,
            num_samples: 5,
            seed: Some(42),
            debug: true,
        };

        let serialized = serde_json::to_string(&config).expect("Serialization failed");
        let deserialized: SubwordRegularizationConfig =
            serde_json::from_str(&serialized).expect("Deserialization failed");

        assert_eq!(config.alpha, deserialized.alpha);
        assert_eq!(config.num_samples, deserialized.num_samples);
        assert_eq!(config.seed, deserialized.seed);
        assert_eq!(config.debug, deserialized.debug);
    }

    /// Regression test for the core P0 bug: `SubwordRegularizer` used to
    /// randomly delete and duplicate *characters* of the input text. At
    /// `alpha = 1.0` roughly 10% of characters were deleted and 5%
    /// duplicated per sample, so across several samples the decoded text
    /// would essentially never match the source. The fix must never alter
    /// the text: only where it gets *split* before re-encoding.
    #[test]
    fn test_regularization_never_corrupts_the_source_text() {
        let text = "the quick brown fox jumps over the lazy dog";
        let tokenizer = CharTokenizer::from_text(text, 1000);
        let config = SubwordRegularizationConfig {
            alpha: 1.0, // maximum split probability: old code -> maximum corruption
            num_samples: 10,
            seed: Some(7),
            debug: false,
        };
        let mut regularizer = SubwordRegularizer::new(tokenizer, config);

        let results =
            regularizer.encode_with_regularization(text).expect("Operation failed in test");
        assert_eq!(results.len(), 10);

        for tokenized in &results {
            let decoded = regularizer
                .inner()
                .decode(&tokenized.input_ids)
                .expect("Operation failed in test");
            // CharTokenizer::decode strips [PAD]/[CLS]/[SEP]; every other
            // character must reconstruct the source text exactly, regardless
            // of how it was internally split and re-stitched.
            assert_eq!(decoded, text);
        }
    }

    /// `sample_fragments` must always produce fragments whose concatenation
    /// is exactly the original text -- this is the property that lets
    /// re-encoding fragments independently stand in for "sampling a
    /// segmentation" without ever corrupting the source text.
    #[test]
    fn test_sample_fragments_concatenate_to_original_text() {
        let tokenizer = greedy_tokenizer();
        let config = SubwordRegularizationConfig {
            alpha: 0.7,
            num_samples: 1,
            seed: Some(1),
            debug: false,
        };
        let mut regularizer = SubwordRegularizer::new(tokenizer, config);

        let text = "hello world hello";
        for _ in 0..25 {
            let fragments = regularizer.sample_fragments(text);
            assert_eq!(fragments.concat(), text);
        }
    }

    /// At `alpha = 1.0`, every word-internal boundary is split, so a
    /// merge-sensitive tokenizer must produce a *different* token-ID
    /// sequence than encoding the whole word at once. This is deterministic
    /// (does not depend on a particular RNG draw succeeding), so it is not
    /// a flaky test.
    #[test]
    fn test_full_split_changes_segmentation_for_merge_sensitive_tokenizer() {
        let tokenizer = greedy_tokenizer();
        let whole_word_ids = tokenizer.encode("hello").expect("Operation failed in test").input_ids;
        assert_eq!(whole_word_ids, vec![1]); // greedy match on "hello" itself

        let config = SubwordRegularizationConfig {
            alpha: 1.0,
            num_samples: 1,
            seed: Some(3),
            debug: false,
        };
        let mut regularizer = SubwordRegularizer::new(tokenizer, config);

        let results = regularizer
            .encode_with_regularization("hello")
            .expect("Operation failed in test");
        assert_eq!(results.len(), 1);
        // Fully character-split: h, e, l, l, o -> ids 8, 9, 4, 4, 5.
        assert_eq!(results[0].input_ids, vec![8, 9, 4, 4, 5]);
        assert_ne!(results[0].input_ids, whole_word_ids);
    }

    /// Repeated sampling at a moderate alpha must, across enough draws,
    /// produce more than one distinct segmentation of the same text -- the
    /// defining property of subword regularization (as opposed to a
    /// deterministic, single-output "regularizer").
    #[test]
    fn test_repeated_sampling_yields_varied_segmentations() {
        let tokenizer = greedy_tokenizer();
        let config = SubwordRegularizationConfig {
            alpha: 0.5,
            num_samples: 1,
            seed: Some(99),
            debug: false,
        };
        let mut regularizer = SubwordRegularizer::new(tokenizer, config);

        let mut distinct = std::collections::HashSet::new();
        for _ in 0..40 {
            let results = regularizer
                .encode_with_regularization("hello")
                .expect("Operation failed in test");
            distinct.insert(results[0].input_ids.clone());
        }

        assert!(
            distinct.len() > 1,
            "expected multiple distinct segmentations across 40 samples, got {:?}",
            distinct
        );
    }

    /// `UnigramSubwordRegularizer::sample_segmentation` must always return
    /// pieces that reconstruct the original text.
    #[test]
    fn test_unigram_ffbs_pieces_concatenate_to_original_text() {
        let mut vocab = HashMap::new();
        vocab.insert("ab".to_string(), -0.1);
        vocab.insert("a".to_string(), -1.0);
        vocab.insert("b".to_string(), -1.0);

        let config = SubwordRegularizationConfig {
            alpha: 1.0,
            num_samples: 1,
            seed: Some(5),
            debug: false,
        };
        let mut regularizer = UnigramSubwordRegularizer::new(vocab, config);

        for _ in 0..25 {
            let pieces = regularizer.sample_segmentation("ab").expect("Operation failed in test");
            assert_eq!(pieces.concat(), "ab");
        }
    }

    /// With an ambiguous vocabulary and a non-trivial temperature, repeated
    /// FFBS sampling must visit more than one segmentation -- proving this
    /// is real probabilistic *sampling* over the lattice, not a fixed
    /// argmax dressed up as "regularization".
    #[test]
    fn test_unigram_ffbs_produces_varied_segmentations() {
        let mut vocab = HashMap::new();
        vocab.insert("ab".to_string(), -0.1);
        vocab.insert("a".to_string(), -1.0);
        vocab.insert("b".to_string(), -1.0);

        let config = SubwordRegularizationConfig {
            alpha: 1.0,
            num_samples: 1,
            seed: Some(11),
            debug: false,
        };
        let mut regularizer = UnigramSubwordRegularizer::new(vocab, config);

        let mut distinct = std::collections::HashSet::new();
        for _ in 0..60 {
            let pieces = regularizer.sample_segmentation("ab").expect("Operation failed in test");
            distinct.insert(pieces);
        }

        assert!(
            distinct.len() > 1,
            "expected both the ['ab'] and ['a','b'] segmentations to appear, got {:?}",
            distinct
        );
    }

    /// A very small temperature (alpha near 0) must concentrate sampling on
    /// the single highest-scoring (Viterbi) segmentation.
    #[test]
    fn test_unigram_ffbs_near_zero_alpha_is_near_deterministic() {
        let mut vocab = HashMap::new();
        vocab.insert("ab".to_string(), -0.1); // clearly the best segmentation
        vocab.insert("a".to_string(), -5.0);
        vocab.insert("b".to_string(), -5.0);

        let config = SubwordRegularizationConfig {
            alpha: 1e-3,
            num_samples: 1,
            seed: Some(17),
            debug: false,
        };
        let mut regularizer = UnigramSubwordRegularizer::new(vocab, config);

        for _ in 0..20 {
            let pieces = regularizer.sample_segmentation("ab").expect("Operation failed in test");
            assert_eq!(pieces, vec!["ab".to_string()]);
        }
    }

    /// A character with zero vocabulary coverage anywhere makes the text
    /// unsegmentable; this must be a structured error, not a silently
    /// empty or partial result.
    #[test]
    fn test_unigram_ffbs_errors_on_uncoverable_text() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), -1.0);
        // No entry covers "b" at all, alone or combined.

        let config = SubwordRegularizationConfig::default();
        let mut regularizer = UnigramSubwordRegularizer::new(vocab, config);

        let result = regularizer.sample_segmentation("ab");
        assert!(result.is_err());
    }
}
