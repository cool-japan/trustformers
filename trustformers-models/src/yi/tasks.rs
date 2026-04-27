//! # Yi-1.5 Task-Specific Implementations
//!
//! This module provides task-specific utilities for Yi-1.5 language models,
//! including:
//! - ChatML prompt formatting (Yi / Qwen-2 style)
//! - Causal LM head utilities (greedy, top-k, top-p)
//! - Sequence classification head
//! - Token classification head
//! - RMSNorm and SwiGLU helpers
//! - GQA attention coverage utilities (Yi uses 4 / 8 KV heads)

use std::fmt;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors produced by Yi task-specific operations.
#[derive(Debug)]
pub enum YiTaskError {
    /// Empty input sequence.
    EmptyInput,
    /// Empty ChatML turn content.
    EmptyChatContent { turn: usize },
    /// Top-k value exceeds vocabulary size.
    TopKTooLarge { k: usize, vocab_size: usize },
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Forward pass failed.
    ForwardError(String),
    /// Invalid nucleus probability.
    InvalidNucleus(f32),
}

impl fmt::Display for YiTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            YiTaskError::EmptyInput => write!(f, "Yi task error: empty input"),
            YiTaskError::EmptyChatContent { turn } => {
                write!(f, "Yi task error: empty content in turn {turn}")
            },
            YiTaskError::TopKTooLarge { k, vocab_size } => {
                write!(
                    f,
                    "Yi task error: top_k={k} exceeds vocab_size={vocab_size}"
                )
            },
            YiTaskError::InvalidConfig(msg) => {
                write!(f, "Yi task error: invalid config: {msg}")
            },
            YiTaskError::ForwardError(msg) => {
                write!(f, "Yi task error: forward error: {msg}")
            },
            YiTaskError::InvalidNucleus(p) => {
                write!(f, "Yi task error: nucleus probability {p} out of (0,1]")
            },
        }
    }
}

impl std::error::Error for YiTaskError {}

// ─── ChatML constants ────────────────────────────────────────────────────────

/// ChatML turn-start token.
pub const IM_START: &str = "<|im_start|>";
/// ChatML turn-end token.
pub const IM_END: &str = "<|im_end|>";
/// System role identifier.
pub const ROLE_SYSTEM: &str = "system";
/// User role identifier.
pub const ROLE_USER: &str = "user";
/// Assistant role identifier.
pub const ROLE_ASSISTANT: &str = "assistant";

// ─── ChatML prompt builder ────────────────────────────────────────────────────

/// A single ChatML message.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role: `"system"`, `"user"`, or `"assistant"`.
    pub role: String,
    /// Text content of the turn.
    pub content: String,
}

impl ChatMessage {
    /// Convenience constructor.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Format a ChatML prompt from a sequence of `(role, content)` messages.
///
/// Appends an open `<|im_start|>assistant\n` at the end to prime generation.
///
/// # Errors
///
/// Returns [`YiTaskError::EmptyChatContent`] if any turn has empty content.
pub fn format_chatml_prompt(messages: &[ChatMessage]) -> Result<String, YiTaskError> {
    let mut buf = String::new();
    for (idx, msg) in messages.iter().enumerate() {
        if msg.content.trim().is_empty() {
            return Err(YiTaskError::EmptyChatContent { turn: idx });
        }
        buf.push_str(IM_START);
        buf.push_str(&msg.role);
        buf.push('\n');
        buf.push_str(&msg.content);
        buf.push('\n');
        buf.push_str(IM_END);
        buf.push('\n');
    }
    buf.push_str(IM_START);
    buf.push_str(ROLE_ASSISTANT);
    buf.push('\n');
    Ok(buf)
}

/// Format a simple single-turn ChatML prompt.
///
/// Optionally includes a system instruction.
///
/// # Errors
///
/// Returns [`YiTaskError::EmptyInput`] if `user_message` is empty.
pub fn format_simple_prompt(
    system: Option<&str>,
    user_message: &str,
) -> Result<String, YiTaskError> {
    if user_message.trim().is_empty() {
        return Err(YiTaskError::EmptyInput);
    }
    let mut msgs = Vec::new();
    if let Some(sys) = system {
        msgs.push(ChatMessage::new(ROLE_SYSTEM, sys));
    }
    msgs.push(ChatMessage::new(ROLE_USER, user_message));
    format_chatml_prompt(&msgs)
}

// ─── RMSNorm ──────────────────────────────────────────────────────────────────

/// RMSNorm with unit weight (no bias).
pub fn rms_norm(input: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
    let rms = (mean_sq + eps).sqrt();
    input.iter().map(|x| x / rms).collect()
}

// ─── SwiGLU ──────────────────────────────────────────────────────────────────

/// SiLU: `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU: `silu(gate) * up`.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect()
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// Numerically stable softmax.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return exps;
    }
    exps.iter().map(|&e| e / sum).collect()
}

// ─── Greedy decode ────────────────────────────────────────────────────────────

/// Greedy decode: return the index of the maximum logit.
pub fn greedy_decode(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
}

// ─── Top-k filter ─────────────────────────────────────────────────────────────

/// Mask logits below the top-k threshold with `f32::NEG_INFINITY`.
pub fn top_k_filter(logits: &[f32], k: usize) -> Result<Vec<f32>, YiTaskError> {
    let vocab = logits.len();
    if k > vocab {
        return Err(YiTaskError::TopKTooLarge {
            k,
            vocab_size: vocab,
        });
    }
    if k == 0 {
        return Ok(vec![f32::NEG_INFINITY; vocab]);
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = indexed[k - 1].1;
    Ok(logits
        .iter()
        .map(|&l| if l >= threshold { l } else { f32::NEG_INFINITY })
        .collect())
}

// ─── GQA utilities ────────────────────────────────────────────────────────────

/// Number of query groups for GQA given the head counts.
///
/// Returns `num_query_heads / num_kv_heads`.
///
/// # Errors
///
/// Returns [`YiTaskError::InvalidConfig`] if `num_kv_heads == 0` or if the
/// division is not exact.
pub fn gqa_groups(num_query_heads: usize, num_kv_heads: usize) -> Result<usize, YiTaskError> {
    if num_kv_heads == 0 {
        return Err(YiTaskError::InvalidConfig(
            "num_kv_heads must be > 0".to_string(),
        ));
    }
    if !num_query_heads.is_multiple_of(num_kv_heads) {
        return Err(YiTaskError::InvalidConfig(format!(
            "num_query_heads={num_query_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )));
    }
    Ok(num_query_heads / num_kv_heads)
}

// ─── Causal LM head ───────────────────────────────────────────────────────────

/// Yi causal LM head.
///
/// All weights zero-initialised for test use.
pub struct YiForCausalLM {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Whether lm_head is tied to the embedding weights.
    pub tie_word_embeddings: bool,
    /// LM head weight `[vocab_size × hidden_size]`.
    lm_weight: Vec<Vec<f32>>,
}

impl YiForCausalLM {
    /// Create a new causal LM head.
    pub fn new(
        hidden_size: usize,
        vocab_size: usize,
        tie_word_embeddings: bool,
    ) -> Result<Self, YiTaskError> {
        if hidden_size == 0 {
            return Err(YiTaskError::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if vocab_size == 0 {
            return Err(YiTaskError::InvalidConfig(
                "vocab_size must be > 0".to_string(),
            ));
        }
        Ok(Self {
            vocab_size,
            hidden_size,
            tie_word_embeddings,
            lm_weight: vec![vec![0.0f32; hidden_size]; vocab_size],
        })
    }

    /// Compute vocabulary logits from the last hidden state.
    pub fn compute_logits(&self, last_hidden: &[f32]) -> Result<Vec<f32>, YiTaskError> {
        if last_hidden.len() != self.hidden_size {
            return Err(YiTaskError::ForwardError(format!(
                "expected hidden_size={}, got {}",
                self.hidden_size,
                last_hidden.len()
            )));
        }
        let logits: Vec<f32> = self
            .lm_weight
            .iter()
            .map(|row| row.iter().zip(last_hidden.iter()).map(|(w, x)| w * x).sum())
            .collect();
        Ok(logits)
    }

    /// Greedy forward: argmax token from the last hidden state.
    pub fn forward_greedy(&self, last_hidden: &[f32]) -> Result<u32, YiTaskError> {
        let logits = self.compute_logits(last_hidden)?;
        greedy_decode(&logits).ok_or_else(|| YiTaskError::ForwardError("argmax failed".into()))
    }
}

// ─── Sequence classification head ────────────────────────────────────────────

/// Yi sequence classification head.
pub struct YiForSequenceClassification {
    /// Number of output labels.
    pub num_labels: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Weight matrix `[num_labels × hidden_size]`.
    weight: Vec<Vec<f32>>,
    /// Bias `[num_labels]`.
    bias: Vec<f32>,
}

impl YiForSequenceClassification {
    /// Create a new sequence classification head.
    pub fn new(hidden_size: usize, num_labels: usize) -> Result<Self, YiTaskError> {
        if hidden_size == 0 {
            return Err(YiTaskError::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if num_labels == 0 {
            return Err(YiTaskError::InvalidConfig(
                "num_labels must be > 0".to_string(),
            ));
        }
        Ok(Self {
            num_labels,
            hidden_size,
            weight: vec![vec![0.0f32; hidden_size]; num_labels],
            bias: vec![0.0f32; num_labels],
        })
    }

    /// Pool last-token hidden state → label logits.
    pub fn forward(&self, hidden_states: &[f32]) -> Result<Vec<f32>, YiTaskError> {
        if hidden_states.is_empty() {
            return Err(YiTaskError::EmptyInput);
        }
        let seq_len = hidden_states.len() / self.hidden_size;
        if seq_len == 0 {
            return Err(YiTaskError::EmptyInput);
        }
        let start = (seq_len - 1) * self.hidden_size;
        let last = &hidden_states[start..start + self.hidden_size];
        let logits: Vec<f32> = self
            .weight
            .iter()
            .zip(self.bias.iter())
            .map(|(row, &b)| row.iter().zip(last.iter()).map(|(w, x)| w * x).sum::<f32>() + b)
            .collect();
        Ok(logits)
    }
}

// ─── Token classification head ────────────────────────────────────────────────

/// Yi token-level classification head.
pub struct YiForTokenClassification {
    /// Number of token-level labels.
    pub num_labels: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Weight matrix `[num_labels × hidden_size]`.
    weight: Vec<Vec<f32>>,
    /// Bias `[num_labels]`.
    bias: Vec<f32>,
}

impl YiForTokenClassification {
    /// Create a new token classification head.
    pub fn new(hidden_size: usize, num_labels: usize) -> Result<Self, YiTaskError> {
        if hidden_size == 0 {
            return Err(YiTaskError::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if num_labels == 0 {
            return Err(YiTaskError::InvalidConfig(
                "num_labels must be > 0".to_string(),
            ));
        }
        Ok(Self {
            num_labels,
            hidden_size,
            weight: vec![vec![0.0f32; hidden_size]; num_labels],
            bias: vec![0.0f32; num_labels],
        })
    }

    /// Project every token's hidden state to label logits.
    ///
    /// Returns `seq_len × num_labels`.
    pub fn forward(&self, hidden_states: &[f32]) -> Result<Vec<Vec<f32>>, YiTaskError> {
        if hidden_states.is_empty() {
            return Err(YiTaskError::EmptyInput);
        }
        let seq_len = hidden_states.len() / self.hidden_size;
        if seq_len == 0 {
            return Err(YiTaskError::EmptyInput);
        }
        let result = (0..seq_len)
            .map(|t| {
                let start = t * self.hidden_size;
                let tok = &hidden_states[start..start + self.hidden_size];
                self.weight
                    .iter()
                    .zip(self.bias.iter())
                    .map(|(row, &b)| {
                        row.iter().zip(tok.iter()).map(|(w, x)| w * x).sum::<f32>() + b
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();
        Ok(result)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── test_yi_config_6b ─────────────────────────────────────────────────

    #[test]
    fn test_yi_config_6b() {
        use crate::yi::config::YiConfig;
        use trustformers_core::traits::Config;
        let cfg = YiConfig::yi_6b();
        assert_eq!(cfg.vocab_size, 64000);
        assert_eq!(cfg.hidden_size, 4096);
        assert!(cfg.validate().is_ok());
    }

    // ── test_yi_config_34b ────────────────────────────────────────────────

    #[test]
    fn test_yi_config_34b() {
        use crate::yi::config::YiConfig;
        let cfg = YiConfig::yi_34b();
        assert_eq!(cfg.hidden_size, 7168);
        assert_eq!(cfg.num_hidden_layers, 60);
        assert_eq!(cfg.head_dim(), 128);
    }

    // ── test_yi_config_200k_context ───────────────────────────────────────

    #[test]
    fn test_yi_config_200k_context() {
        use crate::yi::config::YiConfig;
        let cfg = YiConfig::yi_6b_200k();
        assert_eq!(cfg.max_position_embeddings, 200000);
        assert!((cfg.rope_theta - 5_000_000.0).abs() < 1.0);
    }

    // ── test_yi_chatml_simple ─────────────────────────────────────────────

    #[test]
    fn test_yi_chatml_simple() {
        let prompt = format_simple_prompt(None, "Hello!").expect("prompt");
        assert!(prompt.contains(IM_START));
        assert!(prompt.contains(ROLE_USER));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.ends_with(&format!("{IM_START}{ROLE_ASSISTANT}\n")));
    }

    // ── test_yi_chatml_with_system ────────────────────────────────────────

    #[test]
    fn test_yi_chatml_with_system() {
        let prompt = format_simple_prompt(Some("Be helpful."), "What is 2+2?").expect("prompt");
        assert!(prompt.contains(ROLE_SYSTEM));
        assert!(prompt.contains("Be helpful."));
        assert!(prompt.contains(ROLE_USER));
        assert!(prompt.contains("What is 2+2?"));
    }

    // ── test_yi_chatml_multi_turn ─────────────────────────────────────────

    #[test]
    fn test_yi_chatml_multi_turn() {
        let msgs = vec![
            ChatMessage::new(ROLE_SYSTEM, "You are helpful."),
            ChatMessage::new(ROLE_USER, "Hi there!"),
            ChatMessage::new(ROLE_ASSISTANT, "Hello!"),
            ChatMessage::new(ROLE_USER, "How are you?"),
        ];
        let prompt = format_chatml_prompt(&msgs).expect("multi-turn prompt");
        assert!(prompt.contains("Hi there!"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.ends_with(&format!("{IM_START}{ROLE_ASSISTANT}\n")));
    }

    // ── test_yi_chatml_empty_content ──────────────────────────────────────

    #[test]
    fn test_yi_chatml_empty_content() {
        let msgs = vec![ChatMessage::new(ROLE_USER, "   ")];
        let err = format_chatml_prompt(&msgs);
        assert!(matches!(
            err,
            Err(YiTaskError::EmptyChatContent { turn: 0 })
        ));
    }

    // ── test_yi_simple_prompt_empty ───────────────────────────────────────

    #[test]
    fn test_yi_simple_prompt_empty() {
        assert!(matches!(
            format_simple_prompt(None, "  "),
            Err(YiTaskError::EmptyInput)
        ));
    }

    // ── test_yi_rms_norm ──────────────────────────────────────────────────

    #[test]
    fn test_yi_rms_norm() {
        let x = vec![3.0f32, 4.0];
        let out = rms_norm(&x, 1e-5);
        let rms = (12.5f32 + 1e-5).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / rms).abs() < 1e-5);
    }

    // ── test_yi_rms_norm_constant ─────────────────────────────────────────

    #[test]
    fn test_yi_rms_norm_constant() {
        let x = vec![5.0f32; 8];
        let out = rms_norm(&x, 1e-8);
        for &v in &out {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "constant input must normalize to 1, got {v}"
            );
        }
    }

    // ── test_yi_swiglu ────────────────────────────────────────────────────

    #[test]
    fn test_yi_swiglu() {
        let gate = vec![1.0f32, -1.0, 0.0];
        let up = vec![2.0f32, 2.0, 2.0];
        let out = swiglu(&gate, &up);
        assert_eq!(out.len(), 3);
        assert!(out[0] > 0.0);
        assert!(out[1] < 0.0);
        assert!(out[2].abs() < 1e-5, "silu(0)*2 == 0, got {}", out[2]);
    }

    // ── test_yi_gqa_groups ────────────────────────────────────────────────

    #[test]
    fn test_yi_gqa_groups_6b() {
        // Yi-6B: 32 query heads, 4 KV heads → 8 groups
        let g = gqa_groups(32, 4).expect("gqa_groups");
        assert_eq!(g, 8);
    }

    // ── test_yi_gqa_groups_34b ────────────────────────────────────────────

    #[test]
    fn test_yi_gqa_groups_34b() {
        // Yi-34B: 56 query heads, 8 KV heads → 7 groups
        let g = gqa_groups(56, 8).expect("gqa_groups");
        assert_eq!(g, 7);
    }

    // ── test_yi_gqa_invalid ───────────────────────────────────────────────

    #[test]
    fn test_yi_gqa_invalid() {
        assert!(matches!(
            gqa_groups(32, 0),
            Err(YiTaskError::InvalidConfig(_))
        ));
        assert!(matches!(
            gqa_groups(32, 5),
            Err(YiTaskError::InvalidConfig(_))
        ));
    }

    // ── test_yi_causal_lm_construction ───────────────────────────────────

    #[test]
    fn test_yi_causal_lm_construction() {
        let head = YiForCausalLM::new(64, 1000, true);
        assert!(head.is_ok());
        let h = head.expect("causal lm");
        assert_eq!(h.vocab_size, 1000);
        assert!(h.tie_word_embeddings);
    }

    // ── test_yi_causal_lm_forward_greedy ──────────────────────────────────

    #[test]
    fn test_yi_causal_lm_forward_greedy() {
        let head = YiForCausalLM::new(4, 10, true).expect("causal lm");
        let token = head.forward_greedy(&[0.0f32; 4]).expect("greedy");
        // zero weights → all 0 → max_by returns last equal-max index
        assert!(token < 10u32, "token {token} must be within vocab_size=10");
    }

    // ── test_yi_causal_lm_dim_mismatch ────────────────────────────────────

    #[test]
    fn test_yi_causal_lm_dim_mismatch() {
        let head = YiForCausalLM::new(8, 10, false).expect("causal lm");
        assert!(matches!(
            head.compute_logits(&[0.0f32; 4]),
            Err(YiTaskError::ForwardError(_))
        ));
    }

    // ── test_yi_seq_cls_forward ───────────────────────────────────────────

    #[test]
    fn test_yi_seq_cls_forward() {
        let head = YiForSequenceClassification::new(8, 3).expect("seq cls");
        let hidden = vec![0.5f32; 24]; // seq_len=3
        let logits = head.forward(&hidden).expect("forward");
        assert_eq!(logits.len(), 3);
    }

    // ── test_yi_seq_cls_empty ─────────────────────────────────────────────

    #[test]
    fn test_yi_seq_cls_empty() {
        let head = YiForSequenceClassification::new(4, 2).expect("seq cls");
        assert!(matches!(head.forward(&[]), Err(YiTaskError::EmptyInput)));
    }

    // ── test_yi_token_cls_forward ─────────────────────────────────────────

    #[test]
    fn test_yi_token_cls_forward() {
        let head = YiForTokenClassification::new(16, 5).expect("token cls");
        let hidden = vec![0.1f32; 48]; // seq_len=3, hidden=16
        let logits = head.forward(&hidden).expect("forward");
        assert_eq!(logits.len(), 3, "one row per token");
        for row in &logits {
            assert_eq!(row.len(), 5, "num_labels per token");
        }
    }

    // ── test_yi_token_cls_single ──────────────────────────────────────────

    #[test]
    fn test_yi_token_cls_single() {
        let head = YiForTokenClassification::new(4, 2).expect("token cls");
        let hidden = vec![1.0f32, 2.0, 3.0, 4.0]; // seq_len=1
        let logits = head.forward(&hidden).expect("forward");
        assert_eq!(logits.len(), 1);
        assert_eq!(logits[0].len(), 2);
    }

    // ── test_yi_softmax ───────────────────────────────────────────────────

    #[test]
    fn test_yi_softmax() {
        let logits = vec![0.0f32, 1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[3] > probs[2]);
    }

    // ── test_yi_top_k_filter ──────────────────────────────────────────────

    #[test]
    fn test_yi_top_k_filter() {
        let logits = vec![1.0f32, 5.0, 3.0, 2.0];
        let f = top_k_filter(&logits, 2).expect("top_k");
        assert!((f[1] - 5.0).abs() < 1e-6);
        assert!((f[2] - 3.0).abs() < 1e-6);
        assert!(f[0].is_infinite() && f[0] < 0.0);
        assert!(f[3].is_infinite() && f[3] < 0.0);
    }

    // ── test_yi_error_display ─────────────────────────────────────────────

    #[test]
    fn test_yi_error_display() {
        let e1 = YiTaskError::EmptyInput;
        assert!(e1.to_string().contains("empty"));

        let e2 = YiTaskError::EmptyChatContent { turn: 2 };
        assert!(e2.to_string().contains("2"));

        let e3 = YiTaskError::TopKTooLarge {
            k: 5,
            vocab_size: 3,
        };
        assert!(e3.to_string().contains("5") && e3.to_string().contains("3"));

        let e4 = YiTaskError::InvalidConfig("oops".to_string());
        assert!(e4.to_string().contains("oops"));
    }

    // ── test_yi_lcg_token_cls ─────────────────────────────────────────────

    #[test]
    fn test_yi_lcg_token_cls() {
        let mut state = 23u64;
        for _ in 0..6 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let hidden = ((state % 4) + 2) as usize * 8;
            let labels = ((state >> 4) % 4 + 2) as usize;
            let seq_len = ((state >> 8) % 3 + 2) as usize;
            let head = YiForTokenClassification::new(hidden, labels).expect("token cls head");
            let hs: Vec<f32> = (0..hidden * seq_len).map(|i| i as f32 * 0.01).collect();
            let out = head.forward(&hs).expect("forward");
            assert_eq!(out.len(), seq_len);
            for row in &out {
                assert_eq!(row.len(), labels);
            }
        }
    }
}
