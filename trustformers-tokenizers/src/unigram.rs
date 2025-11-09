use crate::vocab::Vocab;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

#[derive(Debug, Clone)]
pub struct UnigramTokenizer {
    vocab: Vocab,
    scores: HashMap<String, f32>,
    unk_token: String,
    bos_token: String,
    eos_token: String,
    pad_token: String,
    unk_id: u32,
}

impl UnigramTokenizer {
    pub fn new(vocab: HashMap<String, u32>, scores: HashMap<String, f32>) -> Result<Self> {
        let vocab_obj = Vocab::from_map(vocab);
        let unk_token = "<unk>".to_string();

        let unk_id = vocab_obj.get_id(&unk_token).ok_or_else(|| {
            TrustformersError::other("UNK token not found in vocabulary".to_string())
        })?;

        Ok(Self {
            vocab: vocab_obj,
            scores,
            unk_token,
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            pad_token: "<pad>".to_string(),
            unk_id,
        })
    }

    /// Viterbi algorithm to find the best segmentation
    fn encode_word(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return vec![];
        }

        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();

        // dp[i] = (best_score, best_last_token_start)
        let mut dp = vec![(-f32::INFINITY, 0usize); len + 1];
        dp[0] = (0.0, 0);

        for end in 1..=len {
            for start in 0..end {
                let token: String = chars[start..end].iter().collect();
                let score = self.scores.get(&token).copied().unwrap_or(-f32::INFINITY);

                if score != -f32::INFINITY {
                    let new_score = dp[start].0 + score;
                    if new_score > dp[end].0 {
                        dp[end] = (new_score, start);
                    }
                }
            }
        }

        // Backtrack to find the segmentation
        let mut tokens = Vec::new();
        let mut pos = len;

        while pos > 0 {
            let start = dp[pos].1;
            let token: String = chars[start..pos].iter().collect();

            if self.vocab.contains(&token) {
                tokens.push(token);
            } else {
                // Fall back to UNK if token not in vocab
                tokens.push(self.unk_token.clone());
            }

            pos = start;
        }

        tokens.reverse();
        tokens
    }

    fn tokenize_text(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            let word_tokens = self.encode_word(word);
            tokens.extend(word_tokens);
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
        let tokens: Vec<String> = ids.iter().filter_map(|&id| self.vocab.get_token(id)).collect();

        let text = tokens
            .join(" ")
            .replace(&format!(" {} ", self.pad_token), " ")
            .replace(&format!(" {} ", self.bos_token), " ")
            .replace(&format!(" {} ", self.eos_token), " ")
            .trim()
            .to_string();

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

        let tokenizer = UnigramTokenizer::new(vocab, scores).unwrap();
        let result = tokenizer.encode("hello world").unwrap();

        assert_eq!(result.input_ids, vec![0, 1]);
        assert_eq!(result.attention_mask, vec![1, 1]);
    }
}
