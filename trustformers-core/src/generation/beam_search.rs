//! Beam search decoding for sequence generation.
//!
//! Maintains B beams simultaneously, scoring and pruning at each step.
//! Also supports diverse beam search via group-based diversity penalties.

use std::fmt;

use super::logits_processing::{apply_repetition_penalty_indexed, forbidden_ngram_tokens};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during beam search decoding.
#[derive(Debug, Clone, PartialEq)]
pub enum BeamError {
    /// log_probs slice was empty
    EmptyLogProbs,
    /// Number of logit entries does not match vocab_size
    VocabSizeMismatch,
    /// Configuration is invalid
    InvalidConfig(String),
    /// The scoring function returned an error
    ScoreFunctionError(String),
}

impl fmt::Display for BeamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BeamError::EmptyLogProbs => write!(f, "log_probs slice is empty"),
            BeamError::VocabSizeMismatch => {
                write!(f, "vocab size of log_probs does not match config")
            },
            BeamError::InvalidConfig(msg) => write!(f, "invalid beam search config: {msg}"),
            BeamError::ScoreFunctionError(msg) => {
                write!(f, "score function returned an error: {msg}")
            },
        }
    }
}

impl std::error::Error for BeamError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for beam search decoding.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams to maintain simultaneously.
    pub num_beams: usize,
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Minimum token length before EOS is allowed.
    pub min_length: usize,
    /// Length penalty: >1 favours longer sequences, <1 favours shorter.
    pub length_penalty: f32,
    /// Stop when all beams have hit EOS.
    pub early_stopping: bool,
    /// Prevent repeating n-grams of this size (0 = disabled).
    pub no_repeat_ngram_size: usize,
    /// Penalty factor for tokens already present in the sequence (1.0 = off).
    pub repetition_penalty: f32,
    /// Diversity penalty applied across beam groups.
    pub diversity_penalty: f32,
    /// Number of beam groups for diverse beam search.
    pub num_beam_groups: usize,
    /// Token id that signals end-of-sequence.
    pub eos_token_id: Option<u32>,
    /// Token id used for padding.
    pub pad_token_id: Option<u32>,
    /// Vocabulary size.
    pub vocab_size: usize,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            num_beams: 4,
            max_new_tokens: 50,
            min_length: 0,
            length_penalty: 1.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            repetition_penalty: 1.0,
            diversity_penalty: 0.0,
            num_beam_groups: 1,
            eos_token_id: None,
            pad_token_id: None,
            vocab_size: 32000,
        }
    }
}

impl BeamSearchConfig {
    /// Validate that the configuration is internally consistent.
    pub fn validate(&self) -> Result<(), BeamError> {
        if self.num_beams == 0 {
            return Err(BeamError::InvalidConfig(
                "num_beams must be at least 1".to_string(),
            ));
        }
        if self.num_beam_groups == 0 {
            return Err(BeamError::InvalidConfig(
                "num_beam_groups must be at least 1".to_string(),
            ));
        }
        if self.num_beam_groups > self.num_beams {
            return Err(BeamError::InvalidConfig(
                "num_beam_groups must not exceed num_beams".to_string(),
            ));
        }
        if !self.num_beams.is_multiple_of(self.num_beam_groups) {
            return Err(BeamError::InvalidConfig(
                "num_beams must be divisible by num_beam_groups".to_string(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(BeamError::InvalidConfig(
                "vocab_size must be at least 1".to_string(),
            ));
        }
        if self.repetition_penalty <= 0.0 {
            return Err(BeamError::InvalidConfig(
                "repetition_penalty must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Beam hypothesis
// ---------------------------------------------------------------------------

/// A single hypothesis (partially-generated sequence) maintained in a beam.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Generated token ids (does *not* include the prompt).
    pub tokens: Vec<u32>,
    /// Cumulative log-probability of the sequence so far.
    pub score: f32,
}

impl BeamHypothesis {
    /// Create a new hypothesis with an empty token list and zero score.
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            score: 0.0,
        }
    }

    /// Length-normalised score used for ranking completed hypotheses.
    ///
    /// Formula: `score / len^length_penalty`
    pub fn length_normalized_score(&self, length_penalty: f32) -> f32 {
        let len = self.tokens.len().max(1) as f32;
        self.score / len.powf(length_penalty)
    }
}

impl Default for BeamHypothesis {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Beam state
// ---------------------------------------------------------------------------

/// Tracks the set of active and completed beams during decoding.
#[derive(Debug, Clone)]
pub struct BeamState {
    /// Currently active (incomplete) beam hypotheses.
    pub hypotheses: Vec<BeamHypothesis>,
    /// Hypotheses that have finished (hit EOS or max length).
    pub completed: Vec<BeamHypothesis>,
    /// Tokens from the original prompt (shared by all beams).
    pub prompt_tokens: Vec<u32>,
}

impl BeamState {
    /// Create a new beam state from prompt tokens and an initial set of beams.
    pub fn new(prompt_tokens: Vec<u32>, hypotheses: Vec<BeamHypothesis>) -> Self {
        Self {
            hypotheses,
            completed: Vec::new(),
            prompt_tokens,
        }
    }

    /// Return the best hypothesis found so far.
    ///
    /// Prefers completed hypotheses ranked by length-normalised score;
    /// falls back to the best active hypothesis if none have completed.
    pub fn best_hypothesis(&self) -> Option<&BeamHypothesis> {
        // Default length_penalty=1.0 for ranking – callers should pass the
        // real penalty themselves, but we expose the simpler API here and use
        // a helper for internal ranking.
        self.best_hypothesis_with_penalty(1.0)
    }

    /// Return the best hypothesis with an explicit length penalty.
    pub fn best_hypothesis_with_penalty(&self, length_penalty: f32) -> Option<&BeamHypothesis> {
        let best_completed = self.completed.iter().max_by(|a, b| {
            a.length_normalized_score(length_penalty)
                .partial_cmp(&b.length_normalized_score(length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if best_completed.is_some() {
            return best_completed;
        }

        // Fall back to active beams
        self.hypotheses.iter().max_by(|a, b| {
            a.length_normalized_score(length_penalty)
                .partial_cmp(&b.length_normalized_score(length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Check whether decoding is complete.
    pub fn is_done(&self, num_beams: usize, early_stopping: bool) -> bool {
        if self.hypotheses.is_empty() {
            return true;
        }
        if early_stopping && self.completed.len() >= num_beams {
            return true;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// N-gram blocking helper
// ---------------------------------------------------------------------------

/// Return the set of tokens whose addition to `tokens` would complete an
/// n-gram that already appears earlier in `tokens`.
///
/// Returns an empty vec when `ngram_size` is 0 or fewer than `ngram_size-1`
/// tokens have been generated.
pub fn get_forbidden_tokens_for_ngram(tokens: &[u32], ngram_size: usize) -> Vec<u32> {
    forbidden_ngram_tokens(tokens, ngram_size)
}

// ---------------------------------------------------------------------------
// Single beam-search step
// ---------------------------------------------------------------------------

/// Advance the beam state by one token.
///
/// `log_probs` has shape `[num_active_beams][vocab_size]`.
///
/// The function:
/// 1. Collapses parents that carry the *same* token sequence.  All beams start
///    out identical, so without this the top-`num_beams` candidates would be
///    the same token proposed by every parent and the whole beam would follow a
///    single greedy path forever.
/// 2. Applies repetition penalty, n-gram blocking and minimum-length EOS
///    suppression to the per-beam log-probabilities.
/// 3. Computes `new_score = beam.score + log_prob` for every (beam, token) pair
///    and keeps the best `2 * num_beams` of them, so that beams which finish on
///    EOS can be replaced instead of shrinking the active set.
/// 4. Moves beams that emitted EOS into `beam_state.completed`, keeping only
///    the `num_beams` best finished hypotheses.
pub fn beam_search_step(
    beam_state: &mut BeamState,
    log_probs: &[Vec<f32>],
    config: &BeamSearchConfig,
) -> Result<(), BeamError> {
    if log_probs.is_empty() {
        return Err(BeamError::EmptyLogProbs);
    }

    let num_active = beam_state.hypotheses.len();
    if log_probs.len() != num_active {
        return Err(BeamError::VocabSizeMismatch);
    }

    for beam_log_probs in log_probs.iter() {
        if beam_log_probs.len() != config.vocab_size {
            return Err(BeamError::VocabSizeMismatch);
        }
    }

    // Step 1: unique parents only.
    let mut unique_parents: Vec<usize> = Vec::with_capacity(num_active);
    for beam_idx in 0..num_active {
        let is_duplicate = unique_parents.iter().any(|&kept| {
            beam_state.hypotheses[kept].tokens == beam_state.hypotheses[beam_idx].tokens
        });
        if !is_duplicate {
            unique_parents.push(beam_idx);
        }
    }

    // Number of candidates kept per parent: 2x the beam width lets EOS-finished
    // candidates be replaced without starving the active set.
    let per_parent = config.num_beams.saturating_mul(2).max(1).min(config.vocab_size);

    let mut candidates: Vec<(f32, usize, u32)> =
        Vec::with_capacity(unique_parents.len() * per_parent);

    for &beam_idx in &unique_parents {
        let hyp = &beam_state.hypotheses[beam_idx];
        let mut scores = log_probs[beam_idx].clone();

        // All tokens in this beam (prompt + generated)
        let all_tokens: Vec<u32> =
            beam_state.prompt_tokens.iter().chain(hyp.tokens.iter()).copied().collect();

        apply_repetition_penalty_indexed(
            &mut scores,
            all_tokens.iter().map(|&token| token as usize),
            config.repetition_penalty,
        );

        if config.no_repeat_ngram_size > 0 {
            for token in forbidden_ngram_tokens(&all_tokens, config.no_repeat_ngram_size) {
                if let Some(slot) = scores.get_mut(token as usize) {
                    *slot = f32::NEG_INFINITY;
                }
            }
        }

        // Suppress EOS while the hypothesis is shorter than min_length.
        let below_min_length = hyp.tokens.len() < config.min_length;
        let eos_slot = config
            .eos_token_id
            .filter(|_| below_min_length)
            .and_then(|eos| scores.get_mut(eos as usize));
        if let Some(slot) = eos_slot {
            *slot = f32::NEG_INFINITY;
        }

        // Keep this parent's best `per_parent` tokens in O(vocab).
        let mut order: Vec<u32> = (0..scores.len() as u32).collect();
        if per_parent < order.len() {
            let view: &[f32] = &scores;
            order.select_nth_unstable_by(per_parent - 1, |&a, &b| {
                view[b as usize].total_cmp(&view[a as usize]).then_with(|| a.cmp(&b))
            });
        }
        for &token_id in order.iter().take(per_parent) {
            let log_prob = scores[token_id as usize];
            if !log_prob.is_finite() {
                continue; // blocked token
            }
            candidates.push((hyp.score + log_prob, beam_idx, token_id));
        }
    }

    // Step 3: global ranking of the pooled candidates.
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.2.cmp(&b.2)));

    let old_hypotheses = std::mem::take(&mut beam_state.hypotheses);
    let mut new_hypotheses: Vec<BeamHypothesis> = Vec::with_capacity(config.num_beams);

    for (new_score, beam_idx, token_id) in candidates {
        let is_eos = config.eos_token_id == Some(token_id);
        if !is_eos && new_hypotheses.len() >= config.num_beams {
            continue;
        }

        let parent = &old_hypotheses[beam_idx];
        let mut new_tokens = Vec::with_capacity(parent.tokens.len() + 1);
        new_tokens.extend_from_slice(&parent.tokens);
        new_tokens.push(token_id);

        let new_hyp = BeamHypothesis {
            tokens: new_tokens,
            score: new_score,
        };

        if is_eos {
            beam_state.completed.push(new_hyp);
        } else {
            new_hypotheses.push(new_hyp);
        }
    }

    // Step 4: keep only the best finished hypotheses so the list cannot grow
    // without bound over long generations.
    if beam_state.completed.len() > config.num_beams {
        let penalty = config.length_penalty;
        beam_state.completed.sort_by(|a, b| {
            b.length_normalized_score(penalty)
                .total_cmp(&a.length_normalized_score(penalty))
        });
        beam_state.completed.truncate(config.num_beams);
    }

    beam_state.hypotheses = new_hypotheses;
    Ok(())
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// High-level beam search decoder.
pub struct BeamSearchDecoder {
    pub config: BeamSearchConfig,
}

impl BeamSearchDecoder {
    /// Create a new decoder with the given configuration.
    pub fn new(config: BeamSearchConfig) -> Result<Self, BeamError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Create `num_beams` identical initial beams from the prompt.
    pub fn initialize_beams(&self, prompt_tokens: &[u32]) -> BeamState {
        let hypotheses: Vec<BeamHypothesis> =
            (0..self.config.num_beams).map(|_| BeamHypothesis::new()).collect();
        BeamState::new(prompt_tokens.to_vec(), hypotheses)
    }

    /// Run full beam search decoding.
    ///
    /// `score_fn` receives the current set of beam sequences (prompt + generated)
    /// and must return `[num_active_beams][vocab_size]` log-probabilities.
    ///
    /// Returns the token ids of the best hypothesis (excluding the prompt).
    pub fn decode(
        &self,
        prompt_tokens: &[u32],
        score_fn: impl Fn(&[Vec<u32>]) -> Result<Vec<Vec<f32>>, BeamError>,
    ) -> Result<Vec<u32>, BeamError> {
        let mut best = self.decode_top_n(prompt_tokens, 1, score_fn)?;
        if best.is_empty() {
            return Err(BeamError::EmptyLogProbs);
        }
        Ok(best.remove(0).tokens)
    }

    /// Run full beam search decoding and return the `n` best hypotheses,
    /// ranked by length-normalised score.
    ///
    /// `score_fn` receives the current set of beam sequences (prompt + generated)
    /// and must return `[num_active_beams][vocab_size]` log-probabilities.
    ///
    /// Fewer than `n` hypotheses are returned only when the search itself
    /// produced fewer; `n` is capped at `num_beams`.
    pub fn decode_top_n(
        &self,
        prompt_tokens: &[u32],
        n: usize,
        score_fn: impl Fn(&[Vec<u32>]) -> Result<Vec<Vec<f32>>, BeamError>,
    ) -> Result<Vec<BeamHypothesis>, BeamError> {
        if n == 0 {
            return Err(BeamError::InvalidConfig(
                "requested zero hypotheses".to_string(),
            ));
        }
        let mut beam_state = self.initialize_beams(prompt_tokens);

        for _step in 0..self.config.max_new_tokens {
            if beam_state.is_done(self.config.num_beams, self.config.early_stopping) {
                break;
            }
            if beam_state.hypotheses.is_empty() {
                break;
            }

            // Build full sequences for scoring
            let sequences: Vec<Vec<u32>> = beam_state
                .hypotheses
                .iter()
                .map(|hyp| {
                    let mut seq = beam_state.prompt_tokens.clone();
                    seq.extend_from_slice(&hyp.tokens);
                    seq
                })
                .collect();

            let log_probs = score_fn(&sequences)?;

            beam_search_step(&mut beam_state, &log_probs, &self.config)?;
        }

        // Move remaining active beams to completed
        let active: Vec<BeamHypothesis> = beam_state.hypotheses.drain(..).collect();
        beam_state.completed.extend(active);

        if beam_state.completed.is_empty() {
            return Err(BeamError::EmptyLogProbs);
        }

        let penalty = self.config.length_penalty;
        beam_state.completed.sort_by(|a, b| {
            b.length_normalized_score(penalty)
                .total_cmp(&a.length_normalized_score(penalty))
        });
        beam_state.completed.truncate(n.min(self.config.num_beams).max(1));

        Ok(beam_state.completed)
    }

    /// Diverse beam search step that applies a diversity penalty for tokens
    /// already generated by beams in previous groups.
    ///
    /// `group_idx` identifies the current group (0-indexed).
    /// `previous_group_tokens` contains all tokens emitted by earlier groups
    /// in the current decoding step.
    pub fn diverse_beam_search_step(
        beam_state: &mut BeamState,
        log_probs: &[Vec<f32>],
        group_idx: usize,
        previous_group_tokens: &[u32],
        config: &BeamSearchConfig,
    ) -> Result<(), BeamError> {
        if log_probs.is_empty() {
            return Err(BeamError::EmptyLogProbs);
        }

        let num_active = beam_state.hypotheses.len();
        if log_probs.len() != num_active {
            return Err(BeamError::VocabSizeMismatch);
        }

        for beam_log_probs in log_probs.iter() {
            if beam_log_probs.len() != config.vocab_size {
                return Err(BeamError::VocabSizeMismatch);
            }
        }

        // Apply diversity penalty: subtract diversity_penalty * group_idx for
        // tokens that appeared in previous groups.
        let diversity_discount = config.diversity_penalty * (group_idx as f32);

        let mut penalized: Vec<Vec<f32>> = log_probs.to_vec();
        if diversity_discount > 0.0 {
            for beam_lp in penalized.iter_mut() {
                for &tok in previous_group_tokens {
                    if (tok as usize) < beam_lp.len() {
                        beam_lp[tok as usize] -= diversity_discount;
                    }
                }
            }
        }

        beam_search_step(beam_state, &penalized, config)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // 1. Config defaults
    // ------------------------------------------------------------------
    #[test]
    fn test_config_defaults() {
        let cfg = BeamSearchConfig::default();
        assert_eq!(cfg.num_beams, 4);
        assert_eq!(cfg.max_new_tokens, 50);
        assert_eq!(cfg.min_length, 0);
        assert!((cfg.length_penalty - 1.0).abs() < f32::EPSILON);
        assert!(cfg.early_stopping);
        assert_eq!(cfg.no_repeat_ngram_size, 0);
        assert!((cfg.repetition_penalty - 1.0).abs() < f32::EPSILON);
        assert!((cfg.diversity_penalty - 0.0).abs() < f32::EPSILON);
        assert_eq!(cfg.num_beam_groups, 1);
        assert_eq!(cfg.eos_token_id, None);
        assert_eq!(cfg.pad_token_id, None);
        assert_eq!(cfg.vocab_size, 32000);
    }

    // ------------------------------------------------------------------
    // 2. Length normalisation
    // ------------------------------------------------------------------
    #[test]
    fn test_length_normalized_score_length_penalty_lt_1() {
        // When length_penalty < 1.0, shorter sequences get HIGHER normalised scores.
        let short_hyp = BeamHypothesis {
            tokens: vec![1, 2],
            score: -2.0,
        };
        let long_hyp = BeamHypothesis {
            tokens: vec![1, 2, 3, 4],
            score: -4.0,
        };
        // Same score-per-token, but with length_penalty = 0.5 shorter wins
        let length_penalty = 0.5_f32;
        let short_norm = short_hyp.length_normalized_score(length_penalty);
        let long_norm = long_hyp.length_normalized_score(length_penalty);
        assert!(
            short_norm > long_norm,
            "short={short_norm}, long={long_norm}"
        );
    }

    #[test]
    fn test_length_normalized_score_length_penalty_gt_1() {
        // When length_penalty > 1.0, longer sequences are preferred.
        let short_hyp = BeamHypothesis {
            tokens: vec![1, 2],
            score: -2.0,
        };
        let long_hyp = BeamHypothesis {
            tokens: vec![1, 2, 3, 4],
            score: -4.0,
        };
        let length_penalty = 2.0_f32;
        let short_norm = short_hyp.length_normalized_score(length_penalty);
        let long_norm = long_hyp.length_normalized_score(length_penalty);
        assert!(
            long_norm > short_norm,
            "short={short_norm}, long={long_norm}"
        );
    }

    // ------------------------------------------------------------------
    // 3. Single beam reduces to greedy
    // ------------------------------------------------------------------
    #[test]
    fn test_single_beam_is_greedy() {
        let cfg = BeamSearchConfig {
            num_beams: 1,
            max_new_tokens: 3,
            vocab_size: 5,
            eos_token_id: Some(4),
            ..Default::default()
        };
        let decoder = BeamSearchDecoder::new(cfg).expect("valid config");

        let result = decoder.decode(&[0], |seqs| {
            // Always return token 2 as the best
            let lp: Vec<Vec<f32>> = seqs
                .iter()
                .map(|_| {
                    let mut v = vec![-10.0_f32; 5];
                    v[2] = 0.0;
                    v
                })
                .collect();
            Ok(lp)
        });

        let tokens = result.expect("decoding succeeded");
        // All steps should pick token 2
        assert!(tokens.iter().all(|&t| t == 2), "tokens: {tokens:?}");
    }

    // ------------------------------------------------------------------
    // 4. N-gram blocking (get_forbidden_tokens_for_ngram)
    // ------------------------------------------------------------------
    #[test]
    fn test_forbidden_tokens_ngram_zero() {
        // n-gram size 0 → no forbidden tokens
        let forbidden = get_forbidden_tokens_for_ngram(&[1, 2, 1], 0);
        assert!(forbidden.is_empty());
    }

    #[test]
    fn test_forbidden_tokens_ngram_bigram() {
        // Sequence: [1, 2, 1], ngram_size=2.
        // Suffix of length 1 is [1].  Token 2 follows [1] at position 0.
        let forbidden = get_forbidden_tokens_for_ngram(&[1, 2, 1], 2);
        assert!(forbidden.contains(&2), "expected 2 in {forbidden:?}");
    }

    #[test]
    fn test_forbidden_tokens_ngram_trigram() {
        // Sequence: [1, 2, 3, 1, 2], ngram_size=3.
        // Suffix of length 2 is [1, 2].  Token 3 follows [1, 2] at position 0.
        let forbidden = get_forbidden_tokens_for_ngram(&[1, 2, 3, 1, 2], 3);
        assert!(forbidden.contains(&3), "expected 3 in {forbidden:?}");
    }

    #[test]
    fn test_forbidden_tokens_no_repeat_yet() {
        // No repetition exists → no forbidden tokens.
        let forbidden = get_forbidden_tokens_for_ngram(&[1, 2, 3], 2);
        assert!(!forbidden.contains(&2), "unexpected 2 in {forbidden:?}");
    }

    // ------------------------------------------------------------------
    // 5. Repetition penalty
    // ------------------------------------------------------------------
    #[test]
    fn test_repetition_penalty_demotes_seen_tokens() {
        // Regression: the previous implementation divided *every* logit by the
        // penalty.  Beam scores are log-probabilities, i.e. negative, so
        // dividing made an already-seen token MORE attractive - the exact
        // opposite of a repetition penalty.
        let cfg = BeamSearchConfig {
            num_beams: 1,
            vocab_size: 4,
            repetition_penalty: 10.0,
            max_new_tokens: 1,
            ..Default::default()
        };

        // Token 0 is in the prompt and is also the unpenalised argmax (-0.1).
        // Sign-aware penalty: -0.1 * 10 = -1.0, which loses against -0.5.
        let mut state = BeamState::new(vec![0], vec![BeamHypothesis::new()]);
        let log_probs = vec![vec![-0.1_f32, -0.5, -0.6, -0.7]];
        beam_search_step(&mut state, &log_probs, &cfg).expect("step ok");
        assert_eq!(
            state.hypotheses[0].tokens[0], 1,
            "the repeated token must be demoted, got {:?}",
            state.hypotheses[0].tokens
        );
    }

    #[test]
    fn test_repetition_penalty_one_is_identity() {
        let cfg = BeamSearchConfig {
            num_beams: 1,
            vocab_size: 4,
            repetition_penalty: 1.0,
            max_new_tokens: 1,
            ..Default::default()
        };
        let mut state = BeamState::new(vec![0], vec![BeamHypothesis::new()]);
        let log_probs = vec![vec![-0.1_f32, -0.5, -0.6, -0.7]];
        beam_search_step(&mut state, &log_probs, &cfg).expect("step ok");
        assert_eq!(state.hypotheses[0].tokens[0], 0);
    }

    #[test]
    fn test_identical_parents_are_collapsed_so_beams_stay_distinct() {
        // Regression: every beam starts identical.  Expanding all of them made
        // the top-`num_beams` candidates the SAME token from N parents, so the
        // beam collapsed onto one greedy path.
        let cfg = BeamSearchConfig {
            num_beams: 3,
            vocab_size: 4,
            ..Default::default()
        };
        let decoder = BeamSearchDecoder::new(cfg.clone()).expect("valid config");
        let mut state = decoder.initialize_beams(&[]);

        let log_probs = vec![vec![-0.1_f32, -0.2, -0.3, -0.4]; 3];
        beam_search_step(&mut state, &log_probs, &cfg).expect("step ok");

        assert_eq!(state.hypotheses.len(), 3);
        let tokens: Vec<u32> = state.hypotheses.iter().map(|hyp| hyp.tokens[0]).collect();
        assert_eq!(tokens, vec![0, 1, 2], "beams must explore distinct tokens");
    }

    // ------------------------------------------------------------------
    // 6. Beam initialization
    // ------------------------------------------------------------------
    #[test]
    fn test_beam_initialization() {
        let cfg = BeamSearchConfig {
            num_beams: 4,
            ..Default::default()
        };
        let decoder = BeamSearchDecoder::new(cfg).expect("valid config");
        let state = decoder.initialize_beams(&[10, 20, 30]);

        assert_eq!(state.hypotheses.len(), 4);
        assert_eq!(state.prompt_tokens, vec![10, 20, 30]);
        for hyp in &state.hypotheses {
            assert!(hyp.tokens.is_empty());
            assert!((hyp.score - 0.0).abs() < f32::EPSILON);
        }
        assert!(state.completed.is_empty());
    }

    // ------------------------------------------------------------------
    // 7. beam_search_step: top-k selection
    // ------------------------------------------------------------------
    #[test]
    fn test_beam_search_step_top_k_selection() {
        // 2 beams, vocab=4.  beam 0 best token=3 (score 0.0), beam 1 best token=1 (score -0.1).
        let cfg = BeamSearchConfig {
            num_beams: 2,
            vocab_size: 4,
            ..Default::default()
        };
        let hypotheses = vec![BeamHypothesis::new(), BeamHypothesis::new()];
        let mut state = BeamState::new(vec![], hypotheses);

        let log_probs = vec![
            vec![-1.0_f32, -1.0, -1.0, 0.0], // beam 0: best is token 3
            vec![-0.1_f32, 0.0, -1.0, -1.0], // wait: beam 1: token 1 = 0.0, token 0 = -0.1
        ];
        // Actually beam1 best = token 1 (score 0.0), beam0 best = token 3 (score 0.0).
        // Both tied at 0.0 — we just verify the step doesn't error and produces 2 active beams.
        beam_search_step(&mut state, &log_probs, &cfg).expect("step ok");
        assert_eq!(state.hypotheses.len(), 2);
    }

    // ------------------------------------------------------------------
    // 8. Completed hypothesis detection (EOS)
    // ------------------------------------------------------------------
    #[test]
    fn test_completed_hypothesis_on_eos() {
        let cfg = BeamSearchConfig {
            num_beams: 2,
            vocab_size: 3,
            eos_token_id: Some(2),
            ..Default::default()
        };
        let hypotheses = vec![BeamHypothesis::new(), BeamHypothesis::new()];
        let mut state = BeamState::new(vec![], hypotheses);

        // Both beams are identical, so they collapse into one parent whose best
        // candidate is EOS (token 2); the remaining candidates stay active.
        let log_probs = vec![vec![-1.0_f32, -1.0, 0.0], vec![-1.0_f32, -1.0, 0.0]];
        beam_search_step(&mut state, &log_probs, &cfg).expect("step ok");

        assert_eq!(state.completed.len(), 1);
        assert_eq!(state.completed[0].tokens, vec![2]);
        assert_eq!(state.hypotheses.len(), 2, "active beams get refilled");
    }

    #[test]
    fn test_eos_is_suppressed_below_min_length() {
        let cfg = BeamSearchConfig {
            num_beams: 1,
            vocab_size: 3,
            eos_token_id: Some(2),
            min_length: 2,
            ..Default::default()
        };
        let mut state = BeamState::new(vec![], vec![BeamHypothesis::new()]);
        // EOS is by far the best token, but the hypothesis has length 0 < 2.
        let log_probs = vec![vec![-1.0_f32, -2.0, 0.0]];
        beam_search_step(&mut state, &log_probs, &cfg).expect("step ok");
        assert!(state.completed.is_empty(), "EOS must be blocked");
        assert_eq!(state.hypotheses[0].tokens, vec![0]);
    }

    #[test]
    fn test_decode_top_n_returns_ranked_hypotheses() {
        let cfg = BeamSearchConfig {
            num_beams: 3,
            max_new_tokens: 2,
            vocab_size: 4,
            ..Default::default()
        };
        let decoder = BeamSearchDecoder::new(cfg).expect("valid config");
        let hypotheses = decoder
            .decode_top_n(&[0], 3, |seqs| {
                Ok(seqs.iter().map(|_| vec![-0.1_f32, -0.2, -0.3, -0.4]).collect())
            })
            .expect("decode");

        assert_eq!(hypotheses.len(), 3);
        for pair in hypotheses.windows(2) {
            assert!(
                pair[0].length_normalized_score(1.0) >= pair[1].length_normalized_score(1.0),
                "hypotheses must be ranked best-first"
            );
        }
        assert_eq!(hypotheses[0].tokens, vec![0, 0], "best path is 0 -> 0");
    }

    // ------------------------------------------------------------------
    // 9. Best hypothesis selection
    // ------------------------------------------------------------------
    #[test]
    fn test_best_hypothesis_prefers_completed() {
        let mut state = BeamState::new(
            vec![],
            vec![BeamHypothesis {
                tokens: vec![1, 2, 3],
                score: -3.0,
            }],
        );
        state.completed.push(BeamHypothesis {
            tokens: vec![5, 6],
            score: -1.0,
        });

        let best = state.best_hypothesis().expect("has a best");
        // Completed has higher length-normalised score: -1.0/2^1 = -0.5 vs -3.0/3^1 = -1.0
        assert_eq!(best.tokens, vec![5, 6]);
    }

    // ------------------------------------------------------------------
    // 10. Diverse beam penalty
    // ------------------------------------------------------------------
    #[test]
    fn test_diverse_beam_penalty_applied() {
        let cfg = BeamSearchConfig {
            num_beams: 1,
            vocab_size: 4,
            diversity_penalty: 1.0,
            num_beam_groups: 1,
            ..Default::default()
        };
        let mut state = BeamState::new(vec![], vec![BeamHypothesis::new()]);

        // Without diversity: token 0 has highest logit (0.0)
        let log_probs = vec![vec![0.0_f32, -0.5, -1.0, -1.0]];
        // Previous group emitted token 0.  diversity penalty for group_idx=1: 1.0 * 1 = 1.0
        // Token 0 new logit = 0.0 - 1.0 = -1.0; token 1 = -0.5.  Token 1 wins.
        BeamSearchDecoder::diverse_beam_search_step(
            &mut state,
            &log_probs,
            1,    // group_idx
            &[0], // previous group emitted token 0
            &cfg,
        )
        .expect("step ok");

        assert_eq!(
            state.hypotheses[0].tokens[0], 1,
            "token 1 should win after diversity penalty"
        );
    }

    // ------------------------------------------------------------------
    // 11. is_done
    // ------------------------------------------------------------------
    #[test]
    fn test_is_done_early_stopping() {
        let mut state = BeamState::new(vec![], vec![BeamHypothesis::new(), BeamHypothesis::new()]);
        assert!(!state.is_done(2, true));

        state.completed.push(BeamHypothesis::new());
        state.completed.push(BeamHypothesis::new());
        assert!(state.is_done(2, true));
    }

    #[test]
    fn test_is_done_no_early_stopping() {
        let mut state = BeamState::new(vec![], vec![BeamHypothesis::new()]);
        state.completed.push(BeamHypothesis::new());
        // Without early stopping, still not done while active beams exist
        assert!(!state.is_done(1, false));
    }

    // ------------------------------------------------------------------
    // 12. Error cases
    // ------------------------------------------------------------------
    #[test]
    fn test_error_empty_log_probs() {
        let cfg = BeamSearchConfig {
            num_beams: 1,
            vocab_size: 4,
            ..Default::default()
        };
        let mut state = BeamState::new(vec![], vec![BeamHypothesis::new()]);
        let result = beam_search_step(&mut state, &[], &cfg);
        assert_eq!(result, Err(BeamError::EmptyLogProbs));
    }

    #[test]
    fn test_error_vocab_size_mismatch() {
        let cfg = BeamSearchConfig {
            num_beams: 1,
            vocab_size: 4,
            ..Default::default()
        };
        let mut state = BeamState::new(vec![], vec![BeamHypothesis::new()]);
        // Wrong vocab size (3 instead of 4)
        let log_probs = vec![vec![0.0_f32, 0.0, 0.0]];
        let result = beam_search_step(&mut state, &log_probs, &cfg);
        assert_eq!(result, Err(BeamError::VocabSizeMismatch));
    }

    #[test]
    fn test_error_invalid_config_zero_beams() {
        let cfg = BeamSearchConfig {
            num_beams: 0,
            ..Default::default()
        };
        let result = BeamSearchDecoder::new(cfg);
        assert!(matches!(result, Err(BeamError::InvalidConfig(_))));
    }

    #[test]
    fn test_error_display() {
        assert!(!BeamError::EmptyLogProbs.to_string().is_empty());
        assert!(!BeamError::VocabSizeMismatch.to_string().is_empty());
        assert!(!BeamError::InvalidConfig("x".to_string()).to_string().is_empty());
        assert!(!BeamError::ScoreFunctionError("y".to_string()).to_string().is_empty());
    }
}
