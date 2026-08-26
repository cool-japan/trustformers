//! Unigram language-model tokenizer (SentencePiece `--model_type=unigram`).
//!
//! Segmentation is a Viterbi search over a lattice of character positions: the
//! best path maximizes the sum of piece log-probabilities. Every position is
//! reachable because a single-character *unknown* edge (scored `unk_score`) is
//! always available, which is what keeps one out-of-vocabulary character from
//! collapsing an entire word into a single `<unk>`.

use crate::vocab::Vocab;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// SentencePiece word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK).
pub const WHITESPACE_MARKER: char = '▁';

/// Penalty applied below the least likely known piece for an unknown character,
/// mirroring SentencePiece's `kUnkPenalty`.
const UNK_PENALTY: f32 = 10.0;

#[derive(Debug, Clone)]
pub struct UnigramTokenizer {
    vocab: Vocab,
    scores: HashMap<String, f32>,
    unk_token: String,
    bos_token: String,
    eos_token: String,
    pad_token: String,
    unk_id: u32,
    unk_score: f32,
    escape_whitespace: bool,
}

impl UnigramTokenizer {
    pub fn new(vocab: HashMap<String, u32>, scores: HashMap<String, f32>) -> Result<Self> {
        let vocab_obj = Vocab::from_map(vocab);
        let unk_token = "<unk>".to_string();

        let unk_id = vocab_obj.get_id(&unk_token).ok_or_else(|| {
            TrustformersError::other("UNK token not found in vocabulary".to_string())
        })?;

        let unk_score = Self::compute_unk_score(&scores);

        Ok(Self {
            vocab: vocab_obj,
            scores,
            unk_token,
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            pad_token: "<pad>".to_string(),
            unk_id,
            unk_score,
            escape_whitespace: false,
        })
    }

    /// Treat whitespace as the SentencePiece `▁` marker during tokenization.
    ///
    /// Required for vocabularies loaded from real `.model` files, whose pieces
    /// carry a leading `▁`; off by default so plain word vocabularies keep
    /// working.
    pub fn with_whitespace_escaping(mut self, enable: bool) -> Self {
        self.escape_whitespace = enable;
        self
    }

    /// Whether whitespace is escaped to `▁` before segmentation.
    pub fn escapes_whitespace(&self) -> bool {
        self.escape_whitespace
    }

    /// Piece log-probabilities (token -> score).
    pub fn scores(&self) -> &HashMap<String, f32> {
        &self.scores
    }

    /// The unknown-token string.
    pub fn unk_token(&self) -> &str {
        &self.unk_token
    }

    /// Score assigned to a single-character unknown edge in the lattice.
    pub fn unk_score(&self) -> f32 {
        self.unk_score
    }

    fn compute_unk_score(scores: &HashMap<String, f32>) -> f32 {
        let min_score =
            scores.values().copied().filter(|s| s.is_finite()).fold(f32::INFINITY, f32::min);
        if min_score.is_finite() {
            min_score - UNK_PENALTY
        } else {
            -UNK_PENALTY
        }
    }

    /// Sum of the piece scores of a segmentation (unknown pieces use `unk_score`).
    ///
    /// Exposed so callers (and tests) can compare candidate segmentations on the
    /// same objective the Viterbi search maximizes.
    pub fn segmentation_score(&self, pieces: &[String]) -> f32 {
        pieces
            .iter()
            .map(|piece| self.scores.get(piece).copied().unwrap_or(self.unk_score))
            .sum()
    }

    /// Viterbi search for the maximum-log-probability segmentation.
    fn encode_word(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return vec![];
        }

        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();

        // best[i] = (best score of chars[..i], start index of the last piece)
        let mut best = vec![(f32::NEG_INFINITY, 0usize); len + 1];
        best[0] = (0.0, 0);

        for end in 1..=len {
            for start in 0..end {
                if best[start].0 == f32::NEG_INFINITY {
                    continue;
                }

                let token: String = chars[start..end].iter().collect();
                let score = if self.vocab.contains(&token) {
                    self.scores.get(&token).copied().unwrap_or(self.unk_score)
                } else if end - start == 1 {
                    // Unknown single character: always reachable, heavily penalized.
                    self.unk_score
                } else {
                    continue;
                };

                if !score.is_finite() {
                    continue;
                }

                let candidate = best[start].0 + score;
                if candidate > best[end].0 {
                    best[end] = (candidate, start);
                }
            }
        }

        // Backtrack. Every position is reachable thanks to the unknown edges, so
        // the path below is always a genuine segmentation of the input.
        let mut tokens = Vec::new();
        let mut pos = len;
        while pos > 0 {
            let start = best[pos].1;
            let token: String = chars[start..pos].iter().collect();

            if self.vocab.contains(&token) {
                tokens.push(token);
            } else {
                tokens.push(self.unk_token.clone());
            }

            debug_assert!(start < pos, "Viterbi backtrack must make progress");
            if start >= pos {
                break;
            }
            pos = start;
        }

        tokens.reverse();
        tokens
    }

    /// Replace whitespace runs with the SentencePiece marker.
    fn escape_whitespace_marker(text: &str) -> String {
        let mut escaped = String::with_capacity(text.len() + 3);
        escaped.push(WHITESPACE_MARKER);
        let mut previous_was_space = true;
        for ch in text.trim().chars() {
            if ch.is_whitespace() {
                if !previous_was_space {
                    escaped.push(WHITESPACE_MARKER);
                }
                previous_was_space = true;
            } else {
                escaped.push(ch);
                previous_was_space = false;
            }
        }
        escaped
    }

    fn tokenize_text(&self, text: &str) -> Vec<String> {
        if self.escape_whitespace {
            let escaped = Self::escape_whitespace_marker(text);
            if escaped.chars().all(|c| c == WHITESPACE_MARKER) {
                return Vec::new();
            }
            return self.encode_word(&escaped);
        }

        let mut tokens = Vec::new();
        for word in text.split_whitespace() {
            tokens.extend(self.encode_word(word));
        }
        tokens
    }
}

impl Tokenizer for UnigramTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let tokens = self.tokenize_text(text);

        let input_ids: Vec<u32> = tokens
            .iter()
            .map(|token| self.vocab.get_id(token).unwrap_or(self.unk_id))
            .collect();

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
        let tokens1 = self.tokenize_text(text);
        let tokens2 = self.tokenize_text(text2);

        let mut all_tokens = tokens1;
        all_tokens.push(self.eos_token.clone());
        all_tokens.extend(tokens2);

        let input_ids: Vec<u32> = all_tokens
            .iter()
            .map(|token| self.vocab.get_id(token).unwrap_or(self.unk_id))
            .collect();

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

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let special: [&str; 3] = [
            self.pad_token.as_str(),
            self.bos_token.as_str(),
            self.eos_token.as_str(),
        ];

        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| self.vocab.get_token(id))
            .filter(|token| !special.contains(&token.as_str()))
            .collect();

        let text = if self.escape_whitespace {
            tokens.concat().replace(WHITESPACE_MARKER, " ").trim().to_string()
        } else {
            tokens.join(" ").trim().to_string()
        };

        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.get_token_to_id_map().clone()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tokenizer_with(pieces: &[(&str, f32)]) -> UnigramTokenizer {
        let mut vocab = HashMap::new();
        let mut scores = HashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        scores.insert("<unk>".to_string(), -100.0);
        for (index, (piece, score)) in pieces.iter().enumerate() {
            vocab.insert((*piece).to_string(), (index + 1) as u32);
            scores.insert((*piece).to_string(), *score);
        }
        UnigramTokenizer::new(vocab, scores).expect("construction must succeed")
    }

    #[test]
    fn test_unigram_tokenizer() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("<unk>".to_string(), 2);
        vocab.insert("he".to_string(), 3);
        vocab.insert("llo".to_string(), 4);

        let mut scores = HashMap::new();
        scores.insert("hello".to_string(), -1.0);
        scores.insert("world".to_string(), -1.0);
        scores.insert("<unk>".to_string(), -10.0);
        scores.insert("he".to_string(), -2.0);
        scores.insert("llo".to_string(), -2.0);

        let tokenizer = UnigramTokenizer::new(vocab, scores).expect("Construction failed");
        let result = tokenizer.encode("hello world").expect("Encoding failed");

        assert_eq!(result.input_ids, vec![0, 1]);
        assert_eq!(result.attention_mask, vec![1, 1]);
    }

    /// Regression: one OOV character used to collapse the whole word to `<unk>`
    /// because the unreachable end position kept its default back-pointer of 0.
    #[test]
    fn test_out_of_vocabulary_character_does_not_collapse_word() {
        let tokenizer = tokenizer_with(&[("ab", -1.0), ("cd", -1.0)]);

        // 'X' is not in the vocabulary at all.
        let tokens = tokenizer.encode_word("abXcd");
        assert_eq!(
            tokens,
            vec!["ab".to_string(), "<unk>".to_string(), "cd".to_string()],
            "the valid prefix/suffix segmentation must survive an unknown character"
        );
    }

    /// Viterbi must beat greedy longest/highest-score-at-position matching.
    #[test]
    fn test_viterbi_beats_greedy_segmentation() {
        // Greedy from position 0 picks "a" (-1.0 > -1.2) and is then forced into
        // the very expensive "bcd" (-9.0). Viterbi finds "ab" + "cd" (-2.2).
        let tokenizer = tokenizer_with(&[("a", -1.0), ("ab", -1.2), ("cd", -1.0), ("bcd", -9.0)]);

        let tokens = tokenizer.encode_word("abcd");
        assert_eq!(tokens, vec!["ab".to_string(), "cd".to_string()]);

        let greedy = vec!["a".to_string(), "bcd".to_string()];
        assert!(
            tokenizer.segmentation_score(&tokens) > tokenizer.segmentation_score(&greedy),
            "Viterbi path {:?} must score above the greedy path {:?}",
            tokens,
            greedy
        );
        assert!((tokenizer.segmentation_score(&tokens) - (-2.2)).abs() < 1e-5);
        assert!((tokenizer.segmentation_score(&greedy) - (-10.0)).abs() < 1e-5);
    }

    #[test]
    fn test_whitespace_marker_escaping() {
        let tokenizer =
            tokenizer_with(&[("▁hello", -1.0), ("▁world", -1.0)]).with_whitespace_escaping(true);

        let tokens = tokenizer.tokenize_text("hello world");
        assert_eq!(tokens, vec!["▁hello".to_string(), "▁world".to_string()]);

        let encoded = tokenizer.encode("hello world").expect("encoding must succeed");
        let decoded = tokenizer.decode(&encoded.input_ids).expect("decoding must succeed");
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_unk_score_is_below_every_known_piece() {
        let tokenizer = tokenizer_with(&[("a", -1.0), ("b", -3.5)]);
        assert!(tokenizer.unk_score() < -3.5);
    }
}
