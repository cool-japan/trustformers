//! Command-R task-specific utilities: RAG prompt formatting, tool use formatting,
//! and extended CausalLM interfaces for Cohere Command-R models.

use thiserror::Error;

/// Errors specific to Command-R task operations.
#[derive(Debug, Error)]
pub enum CommandRTaskError {
    /// An empty query was provided.
    #[error("query must not be empty")]
    EmptyQuery,
    /// Tool list was empty when at least one tool is required.
    #[error("tool list must not be empty")]
    NoTools,
    /// A document title or snippet was empty.
    #[error("document entry at index {0} has empty {1}")]
    EmptyDocument(usize, &'static str),
}

// ─── Special token constants ───────────────────────────────────────────────

/// Start-of-turn delimiter used in Command-R's chat format.
pub const START_OF_TURN: &str = "<|START_OF_TURN_TOKEN|>";
/// End-of-turn delimiter used in Command-R's chat format.
pub const END_OF_TURN: &str = "<|END_OF_TURN_TOKEN|>";
/// Role token for the user turn.
pub const USER_TOKEN: &str = "<|USER_TOKEN|>";
/// Role token for the assistant (chatbot) turn.
pub const CHATBOT_TOKEN: &str = "<|CHATBOT_TOKEN|>";
/// Role token for a system-level instruction.
pub const SYSTEM_TOKEN: &str = "<|SYSTEM_TOKEN|>";

// ─── Chat format ───────────────────────────────────────────────────────────

/// Format a simple user message for Command-R's chat format.
///
/// The resulting string is ready to be tokenised and fed to the model.
/// The model should then generate the assistant reply starting from
/// the open `<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>` sentinel that is
/// appended at the end.
///
/// # Errors
///
/// Returns [`CommandRTaskError::EmptyQuery`] when `user_message` is empty.
pub fn format_chat_prompt(
    system: Option<&str>,
    user_message: &str,
) -> Result<String, CommandRTaskError> {
    if user_message.trim().is_empty() {
        return Err(CommandRTaskError::EmptyQuery);
    }

    let mut buf = String::new();

    if let Some(sys) = system {
        buf.push_str(START_OF_TURN);
        buf.push_str(SYSTEM_TOKEN);
        buf.push_str(sys);
        buf.push_str(END_OF_TURN);
        buf.push('\n');
    }

    buf.push_str(START_OF_TURN);
    buf.push_str(USER_TOKEN);
    buf.push_str(user_message);
    buf.push_str(END_OF_TURN);
    buf.push('\n');

    // Open the assistant turn; the model generates text from this point.
    buf.push_str(START_OF_TURN);
    buf.push_str(CHATBOT_TOKEN);

    Ok(buf)
}

// ─── RAG prompt ────────────────────────────────────────────────────────────

/// A document passed to the RAG prompt formatter.
///
/// `title` is shown as a heading, `content` is the passage body.
#[derive(Debug, Clone)]
pub struct RagDocument<'a> {
    pub title: &'a str,
    pub content: &'a str,
}

/// Format a Retrieval-Augmented Generation (RAG) prompt for Command-R.
///
/// Constructs a system prompt that lists the supplied grounding documents and
/// instructs the model to answer only from those documents, followed by a user
/// turn containing the query.
///
/// # Arguments
///
/// * `system_preamble` – Optional extra instruction prepended to the system turn
///   (e.g. persona or safety guidance).
/// * `documents` – Slice of `(title, content)` pairs representing retrieved passages.
///   Accepts an empty slice; in that case the system turn simply instructs the
///   model that no supporting documents were retrieved.
/// * `query` – The user's question.
///
/// # Errors
///
/// * [`CommandRTaskError::EmptyQuery`] if `query` is empty.
/// * [`CommandRTaskError::EmptyDocument`] if any document has an empty title or content.
pub fn format_rag_prompt(
    system_preamble: Option<&str>,
    documents: &[RagDocument<'_>],
    query: &str,
) -> Result<String, CommandRTaskError> {
    if query.trim().is_empty() {
        return Err(CommandRTaskError::EmptyQuery);
    }

    for (idx, doc) in documents.iter().enumerate() {
        if doc.title.trim().is_empty() {
            return Err(CommandRTaskError::EmptyDocument(idx, "title"));
        }
        if doc.content.trim().is_empty() {
            return Err(CommandRTaskError::EmptyDocument(idx, "content"));
        }
    }

    let mut buf = String::new();

    // System turn
    buf.push_str(START_OF_TURN);
    buf.push_str(SYSTEM_TOKEN);

    if let Some(preamble) = system_preamble {
        buf.push_str(preamble);
        buf.push('\n');
    }

    if documents.is_empty() {
        buf.push_str("No supporting documents were retrieved. Answer based on your knowledge.");
    } else {
        buf.push_str(
            "Use the documents below to answer the user's question. \
             Cite the relevant document title(s) in your response.\n\n",
        );
        for (idx, doc) in documents.iter().enumerate() {
            buf.push_str(&format!("Document [{}] (Title: {})\n", idx + 1, doc.title));
            buf.push_str(doc.content);
            buf.push_str("\n\n");
        }
    }

    buf.push_str(END_OF_TURN);
    buf.push('\n');

    // User turn
    buf.push_str(START_OF_TURN);
    buf.push_str(USER_TOKEN);
    buf.push_str(query);
    buf.push_str(END_OF_TURN);
    buf.push('\n');

    // Open assistant turn
    buf.push_str(START_OF_TURN);
    buf.push_str(CHATBOT_TOKEN);

    Ok(buf)
}

// ─── Tool-use prompt ───────────────────────────────────────────────────────

/// Format a tool-use prompt for Command-R.
///
/// The model will be shown the available tool signatures and then asked the
/// user's query so it can decide which (if any) tool to call.
///
/// # Arguments
///
/// * `tools` – Slice of tool descriptions (e.g. `"search(query: str) -> str"`).
///   Must not be empty.
/// * `query` – The user's request.
///
/// # Errors
///
/// * [`CommandRTaskError::NoTools`] if the slice is empty.
/// * [`CommandRTaskError::EmptyQuery`] if `query` is empty.
pub fn format_tool_use_prompt(tools: &[&str], query: &str) -> Result<String, CommandRTaskError> {
    if tools.is_empty() {
        return Err(CommandRTaskError::NoTools);
    }
    if query.trim().is_empty() {
        return Err(CommandRTaskError::EmptyQuery);
    }

    let mut buf = String::new();

    // System turn: list available tools
    buf.push_str(START_OF_TURN);
    buf.push_str(SYSTEM_TOKEN);
    buf.push_str(
        "You have access to the following tools. \
         When appropriate, call a tool by emitting a JSON object \
         with \"tool\" and \"parameters\" keys.\n\n",
    );
    buf.push_str("Available tools:\n");
    for (idx, tool) in tools.iter().enumerate() {
        buf.push_str(&format!("  {}. {}\n", idx + 1, tool));
    }
    buf.push_str(END_OF_TURN);
    buf.push('\n');

    // User turn
    buf.push_str(START_OF_TURN);
    buf.push_str(USER_TOKEN);
    buf.push_str(query);
    buf.push_str(END_OF_TURN);
    buf.push('\n');

    // Open assistant turn
    buf.push_str(START_OF_TURN);
    buf.push_str(CHATBOT_TOKEN);

    Ok(buf)
}

// ─── Logit-scale helpers ───────────────────────────────────────────────────

/// Apply a scalar logit-scale factor to raw model logits.
///
/// Command-R multiplies final logits by `logit_scale` (default 0.0625 = 1/16)
/// to keep them in a numerically stable range for the softmax.
///
/// # Arguments
///
/// * `logits` – A flat `[seq_len * vocab_size]` or `[vocab_size]` slice produced
///   by the LM head.
/// * `scale` – The `logit_scale` value from [`super::CommandRConfig`].
///
/// # Returns
///
/// A new `Vec<f32>` with each element multiplied by `scale`.
pub fn apply_logit_scale(logits: &[f32], scale: f32) -> Vec<f32> {
    logits.iter().map(|&x| x * scale).collect()
}

/// Greedy-decode one token from a logit slice of shape `[vocab_size]`.
///
/// Returns the index of the maximum logit.  This is a deterministic, zero-copy
/// scan with no `unwrap`.
pub fn greedy_token(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config tests ──────────────────────────────────────────────────────

    #[test]
    fn test_command_r_config_default() {
        use crate::command_r::CommandRConfig;
        let cfg = CommandRConfig::default();
        assert_eq!(cfg.vocab_size, 256000);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_command_r_config_base() {
        use crate::command_r::CommandRConfig;
        let cfg = CommandRConfig::command_r();
        assert_eq!(cfg.hidden_size, 8192);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, 64);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_command_r_config_plus() {
        use crate::command_r::CommandRConfig;
        let cfg = CommandRConfig::command_r_plus();
        assert_eq!(cfg.hidden_size, 12288);
        assert_eq!(cfg.num_hidden_layers, 64);
        assert_eq!(cfg.num_attention_heads, 96);
        assert_eq!(cfg.num_key_value_heads, 96);
        assert_eq!(cfg.intermediate_size, 33792);
        assert!(cfg.validate().is_ok());
    }

    // ── LayerNorm tests ───────────────────────────────────────────────────

    #[test]
    fn test_command_r_layer_norm_basic() {
        // LayerNorm: mean-subtraction + divide by std
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32; 4];
        let bias = vec![0.0_f32; 4];
        let eps = 1e-5_f64;

        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n; // 2.5
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std = (var + eps as f32).sqrt();
        let expected: Vec<f32> = x
            .iter()
            .zip(weight.iter())
            .zip(bias.iter())
            .map(|((xi, wi), bi)| (xi - mean) / std * wi + bi)
            .collect();

        assert!((expected[0] - (-1.3416_f32)).abs() < 1e-3);
        assert!((expected[3] - 1.3416_f32).abs() < 1e-3);
    }

    #[test]
    fn test_command_r_layer_norm_zero_mean() {
        // After LayerNorm the output should have near-zero mean (with unit weights, zero bias).
        let x = vec![1.0_f32, 3.0, 5.0, 7.0];
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std = (var + 1e-5_f32).sqrt();
        let normed: Vec<f32> = x.iter().map(|xi| (xi - mean) / std).collect();
        let normed_mean: f32 = normed.iter().sum::<f32>() / n;
        assert!(
            normed_mean.abs() < 1e-4,
            "mean should be ~0, got {}",
            normed_mean
        );
    }

    #[test]
    fn test_command_r_layernorm_vs_rmsnorm_differ() {
        // RMSNorm does NOT subtract the mean; LayerNorm does.
        let x = vec![2.0_f32, 4.0, 6.0];
        let n = x.len() as f32;
        let eps = 1e-5_f32;

        // LayerNorm
        let mean_ln = x.iter().sum::<f32>() / n;
        let var_ln = x.iter().map(|v| (v - mean_ln).powi(2)).sum::<f32>() / n;
        let ln_out0 = (x[0] - mean_ln) / (var_ln + eps).sqrt();

        // RMSNorm
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / n + eps).sqrt();
        let rms_out0 = x[0] / rms;

        // They should produce different values for a non-zero-mean input.
        assert!((ln_out0 - rms_out0).abs() > 1e-3);
    }

    // ── RoPE test ─────────────────────────────────────────────────────────

    #[test]
    fn test_command_r_rope_standard() {
        // Verify that freq[0] = 1.0 and freq decreases with increasing dimension index.
        let dim = 8usize;
        let base = 10000.0_f32;
        let inv_freqs: Vec<f32> =
            (0..dim / 2).map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32)).collect();
        assert!((inv_freqs[0] - 1.0_f32).abs() < 1e-6);
        for i in 1..inv_freqs.len() {
            assert!(inv_freqs[i] < inv_freqs[i - 1]);
        }
    }

    // ── Attention / GQA tests ─────────────────────────────────────────────

    #[test]
    fn test_command_r_attention_gqa() {
        use crate::command_r::config::CommandRConfig;
        // Build a custom GQA config: 32 query heads, 8 KV heads → 4 groups
        let cfg = CommandRConfig {
            num_attention_heads: 32,
            num_key_value_heads: 8,
            hidden_size: 4096,
            ..CommandRConfig::command_r()
        };
        assert_eq!(cfg.num_query_groups(), 4);
        assert!(cfg.is_gqa());
    }

    #[test]
    fn test_command_r_gqa_head_ratio() {
        use crate::command_r::config::CommandRConfig;
        // 56 query heads, 8 KV heads → 7 queries per KV head
        let cfg = CommandRConfig {
            num_attention_heads: 56,
            num_key_value_heads: 8,
            hidden_size: 7168,
            ..CommandRConfig::command_r()
        };
        assert_eq!(cfg.num_query_groups(), 7);
    }

    #[test]
    fn test_command_r_non_gqa_base() {
        use crate::command_r::config::CommandRConfig;
        // Base command_r has equal query and KV heads (non-GQA)
        let cfg = CommandRConfig::command_r();
        assert!(!cfg.is_gqa());
        assert_eq!(cfg.num_query_groups(), 1);
    }

    // ── MLP / SwiGLU tests ────────────────────────────────────────────────

    #[test]
    fn test_command_r_mlp_swiglu() {
        // SwiGLU: output = gate * silu(gate) * up (element-wise)
        // Here we verify the SiLU component: silu(x) = x * sigmoid(x)
        let x = 1.0_f32;
        let sigmoid_x = 1.0 / (1.0 + (-x).exp());
        let silu_x = x * sigmoid_x;
        assert!((silu_x - 0.7311_f32).abs() < 1e-4);

        // SwiGLU for a small vector
        let gate = vec![1.0_f32, -1.0, 0.5];
        let up = vec![2.0_f32, 2.0, 2.0];
        let swiglu: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| {
                let s = 1.0 / (1.0 + (-g).exp());
                g * s * u
            })
            .collect();
        assert!(swiglu[0] > 0.0); // positive input → positive output
        assert!(swiglu[1] < 0.0); // negative input → negative output
    }

    // ── Model forward / logit-scale tests ─────────────────────────────────

    #[test]
    fn test_command_r_logit_scale() {
        let raw = vec![10.0_f32, 5.0, 0.0, -5.0];
        let scaled = apply_logit_scale(&raw, 0.0625);
        assert!((scaled[0] - 0.625_f32).abs() < 1e-6);
        assert!((scaled[1] - 0.3125_f32).abs() < 1e-6);
        assert!((scaled[2] - 0.0_f32).abs() < 1e-6);
        assert!((scaled[3] - (-0.3125_f32)).abs() < 1e-6);
    }

    #[test]
    fn test_command_r_model_forward() {
        // Minimal smoke-test: greedy decode on a tiny logit vector.
        let logits = vec![0.1_f32, 0.9, 0.3, 0.7];
        let token = greedy_token(&logits);
        assert_eq!(token, Some(1u32)); // index 1 has the highest value
    }

    #[test]
    fn test_command_r_generate() {
        // Greedy generation from a sequence of per-step logit sets.
        let steps: Vec<Vec<f32>> = vec![
            vec![0.1, 0.9, 0.2], // step 0 → token 1
            vec![0.8, 0.1, 0.3], // step 1 → token 0
            vec![0.2, 0.3, 0.7], // step 2 → token 2
        ];
        let generated: Vec<u32> = steps.iter().filter_map(|logits| greedy_token(logits)).collect();
        assert_eq!(generated, vec![1u32, 0, 2]);
    }

    // ── RAG prompt tests ──────────────────────────────────────────────────

    #[test]
    fn test_command_r_rag_prompt_no_docs() {
        let prompt = format_rag_prompt(None, &[], "What is the capital of France?")
            .expect("should succeed with no docs");
        assert!(prompt.contains(START_OF_TURN));
        assert!(prompt.contains(SYSTEM_TOKEN));
        assert!(prompt.contains("No supporting documents"));
        assert!(prompt.contains("What is the capital of France?"));
        assert!(prompt.ends_with(CHATBOT_TOKEN));
    }

    #[test]
    fn test_command_r_rag_prompt_with_docs() {
        let docs = vec![
            RagDocument {
                title: "Paris Guide",
                content: "Paris is the capital of France.",
            },
            RagDocument {
                title: "Europe Facts",
                content: "France is a country in Western Europe.",
            },
        ];
        let prompt = format_rag_prompt(None, &docs, "What is the capital of France?")
            .expect("should succeed");
        assert!(prompt.contains("Paris Guide"));
        assert!(prompt.contains("Europe Facts"));
        assert!(prompt.contains("Document [1]"));
        assert!(prompt.contains("Document [2]"));
        assert!(prompt.ends_with(CHATBOT_TOKEN));
    }

    // ── Tool-use prompt tests ─────────────────────────────────────────────

    #[test]
    fn test_command_r_tool_use_prompt() {
        let tools = &[
            "search(query: str) -> List[str]",
            "calculator(expr: str) -> float",
        ];
        let prompt = format_tool_use_prompt(tools, "What is 2 + 2?").expect("should succeed");
        assert!(prompt.contains("search(query: str)"));
        assert!(prompt.contains("calculator(expr: str)"));
        assert!(prompt.contains("What is 2 + 2?"));
        assert!(prompt.ends_with(CHATBOT_TOKEN));
    }

    #[test]
    fn test_command_r_tool_use_prompt_empty_tools() {
        let err = format_tool_use_prompt(&[], "hello");
        assert!(matches!(err, Err(CommandRTaskError::NoTools)));
    }

    // ── Chat format test ──────────────────────────────────────────────────

    #[test]
    fn test_command_r_chat_format() {
        let prompt =
            format_chat_prompt(Some("You are helpful."), "Hello!").expect("should succeed");
        assert!(prompt.contains(SYSTEM_TOKEN));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains(USER_TOKEN));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.ends_with(CHATBOT_TOKEN));
    }

    #[test]
    fn test_command_r_chat_format_no_system() {
        let prompt =
            format_chat_prompt(None, "Hello!").expect("should succeed without system prompt");
        assert!(!prompt.contains(SYSTEM_TOKEN));
        assert!(prompt.contains(USER_TOKEN));
        assert!(prompt.ends_with(CHATBOT_TOKEN));
    }

    #[test]
    fn test_command_r_chat_format_empty_query() {
        let err = format_chat_prompt(None, "  ");
        assert!(matches!(err, Err(CommandRTaskError::EmptyQuery)));
    }

    // ── Error display tests ───────────────────────────────────────────────

    #[test]
    fn test_command_r_error_display() {
        let e1 = CommandRTaskError::EmptyQuery;
        assert!(e1.to_string().contains("empty"));

        let e2 = CommandRTaskError::NoTools;
        assert!(e2.to_string().contains("empty"));

        let e3 = CommandRTaskError::EmptyDocument(2, "title");
        assert!(e3.to_string().contains("2"));
        assert!(e3.to_string().contains("title"));
    }
}
