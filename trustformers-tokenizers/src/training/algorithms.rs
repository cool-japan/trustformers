//! Core tokenizer training algorithms.
//!
//! This module implements the fundamental tokenizer training algorithms:
//! - BPE (Byte Pair Encoding)
//! - WordPiece
//! - Unigram
//!
//! Each algorithm provides comprehensive training capabilities with configurable
//! parameters and normalization support.

use crate::bpe::BPETokenizer;
use crate::normalizer::Normalizer;
use crate::unigram::UnigramTokenizer;
use crate::wordpiece::WordPieceTokenizer;
use std::collections::HashMap;
use trustformers_core::errors::Result;

use super::config::TrainingConfig;

/// BPE (Byte Pair Encoding) trainer implementation.
///
/// BPE works by iteratively merging the most frequent pairs of characters
/// or character sequences to build a vocabulary of subwords.
pub struct BPETrainer {
    config: TrainingConfig,
    normalizer: Option<Box<dyn Normalizer>>,
}

impl BPETrainer {
    /// Create a new BPE trainer with the given configuration.
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            normalizer: None,
        }
    }

    /// Set a text normalizer for preprocessing.
    pub fn with_normalizer(mut self, normalizer: Box<dyn Normalizer>) -> Self {
        self.normalizer = Some(normalizer);
        self
    }

    /// Train a BPE tokenizer on the provided texts.
    ///
    /// # Arguments
    ///
    /// * `texts` - Training corpus as a slice of strings
    ///
    /// # Returns
    ///
    /// A trained `BPETokenizer` ready for encoding/decoding
    pub fn train(&self, texts: &[String]) -> Result<BPETokenizer> {
        // Step 1: Collect and count word frequencies
        let mut word_freqs = HashMap::new();

        for text in texts {
            let processed_text = if let Some(ref normalizer) = self.normalizer {
                normalizer.normalize(text)
            } else {
                text.clone()
            };

            for word in processed_text.split_whitespace() {
                *word_freqs.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Step 2: Initialize vocabulary with characters
        let mut vocab = HashMap::new();
        let mut merge_rules = Vec::new();

        // Add special tokens first
        for (i, token) in self.config.special_tokens.iter().enumerate() {
            vocab.insert(token.clone(), i as u32);
        }

        let mut next_id = self.config.special_tokens.len() as u32;

        // Collect all characters and their frequencies
        let mut char_freqs = HashMap::new();
        for (word, freq) in &word_freqs {
            for ch in word.chars() {
                *char_freqs.entry(ch.to_string()).or_insert(0) += freq;
            }
        }

        // Add frequent characters to vocabulary
        for (ch, freq) in char_freqs {
            if freq >= self.config.min_frequency {
                vocab.insert(ch, next_id);
                next_id += 1;
            }
        }

        // Step 3: BPE algorithm - iteratively merge most frequent pairs
        let mut splits = HashMap::new();
        for (word, freq) in word_freqs {
            if word.chars().count() <= self.config.max_input_chars_per_word {
                let split: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                splits.insert(word, (split, freq));
            }
        }

        while vocab.len() < self.config.vocab_size {
            let mut pair_freqs = HashMap::new();

            // Count pair frequencies across all word splits
            for (split, freq) in splits.values() {
                for i in 0..split.len().saturating_sub(1) {
                    let pair = (split[i].clone(), split[i + 1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            if pair_freqs.is_empty() {
                break;
            }

            // Find most frequent pair (guaranteed present by the is_empty check above)
            let Some(best_pair) =
                pair_freqs.iter().max_by_key(|(_, &freq)| freq).map(|(pair, _)| pair.clone())
            else {
                break;
            };

            // Add merged token to vocabulary
            let merged_token = format!("{}{}", best_pair.0, best_pair.1);
            vocab.insert(merged_token, next_id);
            next_id += 1;

            // Record merge rule
            merge_rules.push(best_pair.clone());

            // Update splits by applying the new merge rule
            let mut new_splits = HashMap::new();
            for (word, (split, freq)) in splits {
                let new_split = self.merge_word(&split, &best_pair);
                new_splits.insert(word, (new_split, freq));
            }
            splits = new_splits;
        }

        Ok(BPETokenizer::new(vocab, merge_rules))
    }

    /// Merge adjacent pairs in a word split according to a merge rule.
    fn merge_word(&self, word: &[String], pair: &(String, String)) -> Vec<String> {
        let mut new_word = Vec::new();
        let mut i = 0;

        while i < word.len() {
            if i < word.len() - 1 && word[i] == pair.0 && word[i + 1] == pair.1 {
                new_word.push(format!("{}{}", pair.0, pair.1));
                i += 2;
            } else {
                new_word.push(word[i].clone());
                i += 1;
            }
        }

        new_word
    }
}

/// WordPiece trainer implementation.
///
/// WordPiece builds vocabulary by selecting subwords that maximize likelihood
/// of the training corpus when segmented using the vocabulary.
pub struct WordPieceTrainer {
    config: TrainingConfig,
    normalizer: Option<Box<dyn Normalizer>>,
}

impl WordPieceTrainer {
    /// Create a new WordPiece trainer with the given configuration.
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            normalizer: None,
        }
    }

    /// Set a text normalizer for preprocessing.
    pub fn with_normalizer(mut self, normalizer: Box<dyn Normalizer>) -> Self {
        self.normalizer = Some(normalizer);
        self
    }

    /// Train a WordPiece tokenizer on the provided texts.
    ///
    /// # Arguments
    ///
    /// * `texts` - Training corpus as a slice of strings
    ///
    /// # Returns
    ///
    /// A trained `WordPieceTokenizer` ready for encoding/decoding
    pub fn train(&self, texts: &[String]) -> Result<WordPieceTokenizer> {
        // Step 1: Collect word frequencies
        let mut word_freqs = HashMap::new();

        for text in texts {
            let processed_text = if let Some(ref normalizer) = self.normalizer {
                normalizer.normalize(text)
            } else {
                text.clone()
            };

            for word in processed_text.split_whitespace() {
                *word_freqs.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Step 2: Initialize vocabulary with special tokens and characters
        let mut vocab = HashMap::new();

        // Add special tokens
        for (i, token) in self.config.special_tokens.iter().enumerate() {
            vocab.insert(token.clone(), i as u32);
        }

        let mut next_id = self.config.special_tokens.len() as u32;

        // Add single characters from the corpus
        let mut char_set = std::collections::HashSet::new();
        for word in word_freqs.keys() {
            for ch in word.chars() {
                char_set.insert(ch);
            }
        }

        for ch in char_set {
            vocab.insert(ch.to_string(), next_id);
            next_id += 1;
        }

        // Step 3: WordPiece algorithm - iteratively add best subwords
        while vocab.len() < self.config.vocab_size {
            let mut subword_scores = HashMap::new();

            // Generate candidate subwords and score them
            for (word, freq) in &word_freqs {
                let subwords = self.generate_subwords(word, &vocab);
                for subword in subwords {
                    if !vocab.contains_key(&subword) {
                        let score = self.score_subword(&subword, &word_freqs, &vocab);
                        *subword_scores.entry(subword).or_insert(0.0) += score * (*freq as f64);
                    }
                }
            }

            if subword_scores.is_empty() {
                break;
            }

            // Add best scoring subword to vocabulary (guaranteed by the check above)
            let Some(best_subword) = subword_scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(subword, _)| subword.clone())
            else {
                break;
            };

            vocab.insert(best_subword, next_id);
            next_id += 1;
        }

        Ok(WordPieceTokenizer::new(vocab, false))
    }

    /// Generate candidate subwords from a word.
    fn generate_subwords(&self, word: &str, vocab: &HashMap<String, u32>) -> Vec<String> {
        let mut subwords = Vec::new();
        let chars: Vec<char> = word.chars().collect();

        for start in 0..chars.len() {
            for end in (start + 1)..=chars.len() {
                let subword = if start > 0 {
                    format!(
                        "{}{}",
                        self.config.end_of_word_suffix,
                        chars[start..end].iter().collect::<String>()
                    )
                } else {
                    chars[start..end].iter().collect::<String>()
                };

                if subword.len() > 1 && subword.len() <= 10 && !vocab.contains_key(&subword) {
                    subwords.push(subword);
                }
            }
        }

        subwords
    }

    /// Score a subword candidate based on its utility for corpus segmentation.
    fn score_subword(
        &self,
        subword: &str,
        word_freqs: &HashMap<String, usize>,
        _vocab: &HashMap<String, u32>,
    ) -> f64 {
        let mut score = 0.0;

        // Count how many words contain this subword
        for word in word_freqs.keys() {
            if word.contains(subword.trim_start_matches("##")) {
                score += 1.0;
            }
        }

        // Prefer longer subwords (they tend to be more meaningful)
        score * (subword.len() as f64).sqrt()
    }
}

/// Unigram trainer implementation.
///
/// Unigram uses the Expectation-Maximization algorithm to learn a vocabulary
/// that maximizes the likelihood of the training corpus.
pub struct UnigramTrainer {
    config: TrainingConfig,
    normalizer: Option<Box<dyn Normalizer>>,
    shrinking_factor: f64,
    num_iterations: usize,
}

impl UnigramTrainer {
    /// Create a new Unigram trainer with the given configuration.
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            normalizer: None,
            shrinking_factor: 0.75, // Remove 25% of vocabulary each iteration
            num_iterations: 8,
        }
    }

    /// Set a text normalizer for preprocessing.
    pub fn with_normalizer(mut self, normalizer: Box<dyn Normalizer>) -> Self {
        self.normalizer = Some(normalizer);
        self
    }

    /// Set the shrinking factor for vocabulary pruning iterations.
    pub fn with_shrinking_factor(mut self, factor: f64) -> Self {
        self.shrinking_factor = factor;
        self
    }

    /// Set the number of EM iterations for training.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.num_iterations = iterations;
        self
    }

    /// Train a Unigram tokenizer on the provided texts.
    ///
    /// # Arguments
    ///
    /// * `texts` - Training corpus as a slice of strings
    ///
    /// # Returns
    ///
    /// A trained `UnigramTokenizer` ready for encoding/decoding
    pub fn train(&self, texts: &[String]) -> Result<UnigramTokenizer> {
        // Step 1: Collect word frequencies
        let mut word_freqs = HashMap::new();

        for text in texts {
            let processed_text = if let Some(ref normalizer) = self.normalizer {
                normalizer.normalize(text)
            } else {
                text.clone()
            };

            for word in processed_text.split_whitespace() {
                *word_freqs.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Step 2: Create initial vocabulary with characters and common substrings
        let mut vocab = self.create_initial_vocabulary(&word_freqs)?;

        // Step 3: Iterative pruning using EM algorithm
        for _ in 0..self.num_iterations {
            vocab = self.prune_vocabulary(vocab, &word_freqs)?;
            if vocab.len() <= self.config.vocab_size {
                break;
            }
        }

        // Step 4: Final vocabulary adjustment to target size
        while vocab.len() > self.config.vocab_size {
            vocab = self.prune_vocabulary(vocab, &word_freqs)?;
        }

        // Convert to the format expected by UnigramTokenizer
        let mut vocab_map = HashMap::new();
        let mut scores_map = HashMap::new();

        for (i, (token, score)) in vocab.iter().enumerate() {
            vocab_map.insert(token.clone(), i as u32);
            scores_map.insert(token.clone(), *score as f32);
        }

        UnigramTokenizer::new(vocab_map, scores_map)
    }

    /// Create initial vocabulary with characters and frequent substrings.
    fn create_initial_vocabulary(
        &self,
        word_freqs: &HashMap<String, usize>,
    ) -> Result<HashMap<String, f64>> {
        let mut vocab = HashMap::new();

        // Add special tokens with high scores (log probability = 0)
        for token in &self.config.special_tokens {
            vocab.insert(token.clone(), 0.0);
        }

        // Add all characters with their log frequencies
        let mut char_freqs = HashMap::new();
        for (word, freq) in word_freqs {
            for ch in word.chars() {
                *char_freqs.entry(ch.to_string()).or_insert(0) += freq;
            }
        }

        // Add characters to vocabulary
        for (ch, freq) in char_freqs {
            if freq >= self.config.min_frequency {
                vocab.insert(ch, (freq as f64).ln());
            }
        }

        // Generate and add subword candidates
        let subword_candidates = self.generate_subword_candidates(word_freqs);
        for (subword, score) in subword_candidates {
            if vocab.len() >= self.config.vocab_size * 4 {
                break; // Start with 4x target vocabulary size
            }
            vocab.insert(subword, score);
        }

        Ok(vocab)
    }

    /// Generate candidate subwords with frequency-based scoring.
    fn generate_subword_candidates(
        &self,
        word_freqs: &HashMap<String, usize>,
    ) -> Vec<(String, f64)> {
        let mut subword_counts = HashMap::new();

        // Extract all possible substrings
        for (word, freq) in word_freqs {
            let chars: Vec<char> = word.chars().collect();
            for start in 0..chars.len() {
                for end in (start + 1)..=chars.len() {
                    if end - start > 1 && end - start <= 10 {
                        // Max subword length
                        let subword = chars[start..end].iter().collect::<String>();
                        *subword_counts.entry(subword).or_insert(0) += freq;
                    }
                }
            }
        }

        // Score and sort subwords
        let mut scored_subwords: Vec<_> = subword_counts
            .into_iter()
            .filter(|(_, freq)| *freq >= self.config.min_frequency)
            .map(|(subword, freq)| {
                // Score based on frequency but penalize length
                let score = (freq as f64).ln() - (subword.len() as f64) * 0.1;
                (subword, score)
            })
            .collect();

        scored_subwords.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_subwords
    }

    /// Prune vocabulary using EM algorithm to remove least useful tokens.
    ///
    /// This is a hard-EM (Viterbi) approximation of SentencePiece's unigram
    /// pruning: an **E-step** re-segments every corpus word under the
    /// current vocabulary via the same maximum-log-probability lattice
    /// search a trained [`UnigramTokenizer`] performs, then an **M-step**
    /// scores each removable piece by the *actual* total corpus
    /// log-likelihood drop that results from re-segmenting, without it,
    /// only the words whose current best segmentation uses it. Pieces are
    /// removed lowest-loss-first until the vocabulary shrinks to
    /// `target_size`.
    fn prune_vocabulary(
        &self,
        mut vocab: HashMap<String, f64>,
        word_freqs: &HashMap<String, usize>,
    ) -> Result<HashMap<String, f64>> {
        if vocab.len() <= self.config.vocab_size {
            return Ok(vocab);
        }

        let unk_score = Self::unigram_unk_score(&vocab);

        // E-step: Viterbi-segment every corpus word under the *current*
        // vocabulary once, caching both the chosen pieces and their total
        // log-probability, and build an inverted index (piece -> words
        // whose current segmentation actually uses it) so the M-step only
        // ever re-segments words a candidate piece can possibly affect.
        let mut segmentations: HashMap<&str, (Vec<String>, f64)> =
            HashMap::with_capacity(word_freqs.len());
        for word in word_freqs.keys() {
            let (pieces, score) = Self::viterbi_segment(&vocab, unk_score, None, word);
            segmentations.insert(word.as_str(), (pieces, score));
        }
        // Second pass: build the inverted index from the now-stable
        // `segmentations` map. Doing this in the same loop that inserts into
        // `segmentations` does not borrow-check: `piece_to_words` would hold
        // `&str` borrows into each word's local `pieces: Vec<String>` right
        // before that same `Vec<String>` is moved into `segmentations`.
        let mut piece_to_words: HashMap<&str, Vec<&str>> = HashMap::new();
        for (word, (pieces, _)) in &segmentations {
            for piece in pieces {
                piece_to_words.entry(piece.as_str()).or_default().push(*word);
            }
        }

        // M-step: score every removable piece by its real removal loss.
        let mut loss_scores = Vec::with_capacity(vocab.len());
        for token in vocab.keys() {
            // Skip special tokens
            if self.config.special_tokens.contains(token) {
                continue;
            }
            // The base 1-character alphabet is always kept: it is the
            // fallback that keeps every future input encodable without
            // collapsing into `<unk>`, exactly as real SentencePiece
            // unigram training protects the alphabet from pruning.
            if token.chars().count() <= 1 {
                continue;
            }

            let loss = Self::calculate_removal_loss(
                token,
                &vocab,
                unk_score,
                &segmentations,
                &piece_to_words,
                word_freqs,
            );
            loss_scores.push((token.clone(), loss));
        }

        // Sort by loss (ascending - remove tokens with least loss first)
        loss_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Remove tokens with lowest loss
        let target_size = ((vocab.len() as f64) * self.shrinking_factor)
            .max(self.config.vocab_size as f64) as usize;
        let tokens_to_remove = vocab.len().saturating_sub(target_size);

        for (token, _) in loss_scores.iter().take(tokens_to_remove) {
            vocab.remove(token);
        }

        Ok(vocab)
    }

    /// Score assigned to a single out-of-vocabulary character in the
    /// Viterbi lattice, mirroring `UnigramTokenizer`'s own `unk_score`
    /// (`min_score - UNK_PENALTY`). Keeping every position in the lattice
    /// reachable via this heavily-penalized edge is what guarantees
    /// [`Self::viterbi_segment`] always finds a genuine segmentation.
    fn unigram_unk_score(vocab: &HashMap<String, f64>) -> f64 {
        const UNK_PENALTY: f64 = 10.0;
        let min_score =
            vocab.values().copied().filter(|s| s.is_finite()).fold(f64::INFINITY, f64::min);
        if min_score.is_finite() {
            min_score - UNK_PENALTY
        } else {
            -UNK_PENALTY
        }
    }

    /// Viterbi search for the maximum-log-probability segmentation of
    /// `word` under `vocab`, optionally pretending `exclude` is not in the
    /// vocabulary at all (used to score a removal candidate). This is the
    /// same lattice search [`UnigramTokenizer::encode`] performs
    /// (reimplemented here, operating directly on the trainer's own `f64`
    /// working scores with an `exclude` parameter, rather than
    /// constructing a full `UnigramTokenizer` per removal candidate).
    /// Returns the chosen piece sequence and its total log-probability.
    fn viterbi_segment(
        vocab: &HashMap<String, f64>,
        unk_score: f64,
        exclude: Option<&str>,
        word: &str,
    ) -> (Vec<String>, f64) {
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();
        if len == 0 {
            return (Vec::new(), 0.0);
        }

        // best[i] = (best log-prob of chars[..i], start index of its last piece)
        let mut best = vec![(f64::NEG_INFINITY, 0usize); len + 1];
        best[0] = (0.0, 0);

        for end in 1..=len {
            for start in 0..end {
                if best[start].0 == f64::NEG_INFINITY {
                    continue;
                }

                let piece: String = chars[start..end].iter().collect();
                let excluded = exclude == Some(piece.as_str());
                let score = if !excluded && vocab.contains_key(&piece) {
                    vocab[&piece]
                } else if end - start == 1 {
                    // Unknown single character: always reachable.
                    unk_score
                } else {
                    continue;
                };

                let candidate = best[start].0 + score;
                if candidate > best[end].0 {
                    best[end] = (candidate, start);
                }
            }
        }

        // Backtrack. Every position is reachable via the unknown edges, so
        // this always terminates at `pos == 0`.
        let mut pieces = Vec::new();
        let mut pos = len;
        while pos > 0 {
            let (score_here, start) = best[pos];
            debug_assert!(
                score_here.is_finite(),
                "every position is reachable via unknown edges"
            );
            pieces.push(chars[start..pos].iter().collect::<String>());
            pos = start;
        }
        pieces.reverse();

        (pieces, best[len].0)
    }

    /// Real EM-style removal loss for `token`: the corpus log-likelihood
    /// drop from re-segmenting, without `token`, only the words whose
    /// current Viterbi segmentation (`segmentations` / `piece_to_words`,
    /// from the E-step) actually uses it. Words that never chose `token`
    /// are entirely unaffected by its removal and are never re-segmented.
    ///
    /// Replaces the old `word.contains(token)` substring-count heuristic,
    /// which counted a token as "used" by any word containing it as a
    /// substring -- even words whose actual best segmentation never chose
    /// it -- and invented a length-based penalty rather than measuring any
    /// real likelihood change.
    #[allow(clippy::too_many_arguments)]
    fn calculate_removal_loss(
        token: &str,
        vocab: &HashMap<String, f64>,
        unk_score: f64,
        segmentations: &HashMap<&str, (Vec<String>, f64)>,
        piece_to_words: &HashMap<&str, Vec<&str>>,
        word_freqs: &HashMap<String, usize>,
    ) -> f64 {
        let Some(words) = piece_to_words.get(token) else {
            // No word's current best segmentation uses this piece at all:
            // removing it changes nothing.
            return 0.0;
        };

        let mut total_loss = 0.0f64;
        for &word in words {
            let freq = *word_freqs.get(word).unwrap_or(&0) as f64;
            if freq == 0.0 {
                continue;
            }
            let (_, old_score) = &segmentations[word];
            let (_, new_score) = Self::viterbi_segment(vocab, unk_score, Some(token), word);
            // Removing a piece can only leave the best achievable score the
            // same or strictly worse (the search space shrinks), so this
            // is >= 0 up to floating-point noise.
            total_loss += freq * (old_score - new_score);
        }
        total_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizer::LowercaseNormalizer;
    use trustformers_core::traits::Tokenizer;

    #[test]
    fn test_bpe_training() {
        let config = TrainingConfig {
            vocab_size: 100,
            min_frequency: 1,
            special_tokens: vec!["[UNK]".to_string()],
            ..Default::default()
        };

        let trainer = BPETrainer::new(config).with_normalizer(Box::new(LowercaseNormalizer));

        let texts = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "world peace".to_string(),
        ];

        let tokenizer = trainer.train(&texts).expect("Operation failed in test");
        assert!(tokenizer.vocab_size() > 0);
        assert!(tokenizer.vocab_size() <= 100);

        // Test that it can encode the training texts
        let encoded = tokenizer.encode("hello world").expect("Encoding failed");
        assert!(!encoded.input_ids.is_empty());
    }

    #[test]
    fn test_wordpiece_training() {
        let config = TrainingConfig {
            vocab_size: 100,
            min_frequency: 1,
            special_tokens: vec![
                "[UNK]".to_string(),
                "[CLS]".to_string(),
                "[SEP]".to_string(),
            ],
            ..Default::default()
        };

        let trainer = WordPieceTrainer::new(config);

        let texts = vec!["hello world".to_string(), "hello there".to_string()];

        let tokenizer = trainer.train(&texts).expect("Operation failed in test");
        assert!(tokenizer.vocab_size() > 0);
        assert!(tokenizer.vocab_size() <= 100);
    }

    #[test]
    fn test_unigram_training() {
        let config = TrainingConfig {
            vocab_size: 50,
            min_frequency: 1,
            special_tokens: vec![
                "<unk>".to_string(),
                "<s>".to_string(),
                "</s>".to_string(),
                "<pad>".to_string(),
            ],
            ..Default::default()
        };

        let trainer = UnigramTrainer::new(config).with_shrinking_factor(0.8).with_iterations(5);

        let texts = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "world peace".to_string(),
            "hello hello world".to_string(),
        ];

        let tokenizer = trainer.train(&texts).expect("Operation failed in test");
        assert!(tokenizer.vocab_size() > 0);
        assert!(tokenizer.vocab_size() <= 50);

        // Test that it can encode the training texts
        let encoded = tokenizer.encode("hello world").expect("Encoding failed");
        assert!(!encoded.input_ids.is_empty());
    }

    #[test]
    fn test_bpe_merge_word() {
        let config = TrainingConfig::default();
        let trainer = BPETrainer::new(config);

        let word = vec![
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "l".to_string(),
            "o".to_string(),
        ];
        let pair = ("l".to_string(), "l".to_string());
        let merged = trainer.merge_word(&word, &pair);

        assert_eq!(merged, vec!["h", "e", "ll", "o"]);
    }

    #[test]
    fn test_wordpiece_subword_generation() {
        let config = TrainingConfig::default();
        let trainer = WordPieceTrainer::new(config);
        let vocab = HashMap::new();

        let subwords = trainer.generate_subwords("hello", &vocab);
        assert!(!subwords.is_empty());

        // Check that some expected subwords are generated
        assert!(subwords.iter().any(|s| s == "he" || s == "##ell" || s == "hello"));
    }

    #[test]
    fn test_trainer_with_normalizer() {
        let config = TrainingConfig {
            vocab_size: 50,
            min_frequency: 1,
            ..Default::default()
        };

        let trainer = BPETrainer::new(config).with_normalizer(Box::new(LowercaseNormalizer));

        let texts = vec!["Hello World".to_string(), "HELLO WORLD".to_string()];
        let tokenizer = trainer.train(&texts).expect("Operation failed in test");

        // Both inputs should be normalized to lowercase
        let encoded1 = tokenizer.encode("Hello World").expect("Encoding failed");
        let encoded2 = tokenizer.encode("hello world").expect("Encoding failed");

        // They should have similar tokenization (the normalizer should handle case)
        assert!(!encoded1.input_ids.is_empty());
        assert!(!encoded2.input_ids.is_empty());
    }
}
