use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// Simplified stream trait for our use case
pub trait GenerationStreamTrait {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}

/// Streaming generation result
#[derive(Debug, Clone)]
pub struct GenerationToken {
    pub token_id: usize,
    pub token_str: String,
    pub logprobs: Option<f32>,
    pub is_finished: bool,
    pub finish_reason: Option<FinishReason>,
}

impl GenerationToken {
    pub fn new(
        token_id: usize,
        token_str: String,
        logprobs: Option<f32>,
        is_finished: bool,
    ) -> Self {
        Self {
            token_id,
            token_str,
            logprobs,
            is_finished,
            finish_reason: None,
        }
    }

    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = Some(reason);
        self.is_finished = true;
        self
    }
}

/// Reasons why generation finished
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    MaxLength,
    EosToken,
    StopSequence,
    UserStopped,
    ConstraintViolation,
    Error,
}

/// Streaming generation iterator
pub struct GenerationStream {
    tokens: VecDeque<GenerationToken>,
    finished: bool,
}

impl GenerationStream {
    pub fn new() -> Self {
        Self {
            tokens: VecDeque::new(),
            finished: false,
        }
    }

    pub fn push_token(&mut self, token: GenerationToken) {
        self.finished = token.is_finished;
        self.tokens.push_back(token);
    }

    pub fn finish(&mut self, reason: FinishReason) {
        self.finished = true;
        if let Some(last_token) = self.tokens.back_mut() {
            last_token.is_finished = true;
            last_token.finish_reason = Some(reason);
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

impl Default for GenerationStream {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationStreamTrait for GenerationStream {
    type Item = GenerationToken;

    fn next(&mut self) -> Option<Self::Item> {
        // Would be pending in async context if !self.finished
        self.tokens.pop_front()
    }
}
