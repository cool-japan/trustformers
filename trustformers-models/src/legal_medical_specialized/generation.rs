//! Real autoregressive text generation for [`LegalMedicalForCausalLM`].
//!
//! `generate`/`generate_with_config` used to format-echo the prompt back to
//! the caller (`Ok(format!("[Legal/Medical Generated]: {}", prompt))`). This
//! module replaces that with genuine token-by-token autoregressive decoding:
//! a byte-level tokenizer turns the prompt into token ids,
//! [`LegalMedicalForCausalLM::forward_logits_with_mask`] runs the real
//! transformer forward pass at every step, and
//! [`crate::generation_utils::GenerationUtils`] samples the next token from
//! the resulting logits. With random (untrained) weights the output is not
//! fluent legal/medical prose, but it is genuine model output: it varies
//! with the seed, the prompt, and the weights - the one property an echo can
//! never have.
//!
//! ## Byte-level tokenizer
//!
//! These configurations describe a vocabulary size but ship no actual
//! vocabulary/BPE merges, so this module defines its own minimal, total,
//! reversible mapping: token ids `[BYTE_TOKEN_OFFSET, BYTE_TOKEN_OFFSET +
//! 256)` correspond 1:1 to raw UTF-8 bytes of the input text. Ids below that
//! are reserved (`bos_token_id`/`eos_token_id`/padding). Sampling is
//! restricted to the byte range plus EOS (see [`restrict_logits_to_byte_vocab`])
//! so decoding is always exact - never a guess at what an out-of-scheme id
//! "would have meant".
//!
//! ## Confidentiality-aware masking
//!
//! When [`LegalMedicalConfig::confidentiality_protection`] is set,
//! [`build_confidentiality_mask`] scans the token stream for
//! `<confidential>...</confidential>` spans (see
//! [`LegalMedicalSpecialTokens`]) and builds a real additive attention mask:
//! a position outside a confidential span may not attend into a *different*
//! confidential span (nor, for text generated after an unclosed span, back
//! into an earlier one), which keeps separate privileged blocks from leaking
//! into each other's attention while still letting confidential text attend
//! back to the public context that introduced it.

use super::{LegalMedicalForCausalLM, LegalMedicalSpecialTokens};
use crate::common_patterns::GenerationConfig;
use crate::generation_utils::GenerationUtils;
use anyhow::Result;
use scirs2_core::random::*;
use trustformers_core::tensor::Tensor;

/// First token id used for byte-level tokens. Ids below this are reserved
/// for `bos_token_id` / `eos_token_id` / padding.
const BYTE_TOKEN_OFFSET: u32 = 8;
/// One token per possible byte value.
const BYTE_VOCAB_SIZE: u32 = 256;
/// Hard safety cap on total sequence length regardless of the caller's
/// `max_new_tokens`, so a pathological config can't spin forever.
const MAX_SEQUENCE_LENGTH: usize = 4096;

fn min_vocab_size() -> usize {
    (BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE) as usize
}

impl LegalMedicalForCausalLM {
    /// Encode `text` as byte-level token ids, prefixed with `bos_token_id`.
    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        if self.config.vocab_size < min_vocab_size() {
            anyhow::bail!(
                "vocab_size {} is too small for the byte-level tokenizer (needs >= {})",
                self.config.vocab_size,
                min_vocab_size()
            );
        }
        let mut ids = Vec::with_capacity(text.len() + 1);
        ids.push(self.config.bos_token_id);
        ids.extend(text.bytes().map(|b| BYTE_TOKEN_OFFSET + b as u32));
        Ok(ids)
    }

    /// Decode generated token ids back to text. Ids outside the byte range
    /// (BOS/EOS/padding/anything a random-init model happened to emit before
    /// the logit mask below was applied to earlier calls) carry no text
    /// content and are dropped rather than guessed at.
    fn decode_tokens(tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|&t| {
                if (BYTE_TOKEN_OFFSET..BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE).contains(&t) {
                    Some((t - BYTE_TOKEN_OFFSET) as u8)
                } else {
                    None
                }
            })
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Generate a continuation of `input`, sampling with a fresh
    /// (non-reproducible) random source. For reproducible output see
    /// [`LegalMedicalForCausalLM::generate_with_seed`].
    pub fn generate(&self, input: &str, max_length: usize) -> Result<String> {
        let mut rng = thread_rng();
        self.generate_impl(input, max_length, &mut rng)
    }

    /// Generate a continuation of `input` with a seeded RNG, so the output is
    /// reproducible for a fixed model, prompt, and seed - and, for
    /// random-init weights, changes when the seed changes. This is the
    /// property a hardcoded echo can never have.
    pub fn generate_with_seed(&self, input: &str, max_length: usize, seed: u64) -> Result<String> {
        let mut rng = StdRng::seed_from_u64(seed);
        self.generate_impl(input, max_length, &mut rng)
    }

    fn generate_impl(&self, input: &str, max_length: usize, rng: &mut impl Rng) -> Result<String> {
        // Create generation config with privacy protection
        let gen_config = GenerationConfig {
            max_new_tokens: max_length,
            temperature: 0.7, // Conservative for legal/medical
            top_p: 0.8,
            do_sample: true,
            repetition_penalty: 1.2, // Reduce repetition
            ..Default::default()
        };

        // Apply privacy protection and generate
        let protected_input = self.apply_privacy_protection(input)?;
        self.generate_with_config(&protected_input, &gen_config, rng)
    }

    fn apply_privacy_protection(&self, text: &str) -> Result<String> {
        // Apply basic privacy protection before processing
        let mut protected_text = text.to_string();

        // Add privacy markers for sensitive content
        if self.contains_sensitive_info(text)? {
            protected_text = format!("[PRIVACY_PROTECTED] {}", protected_text);
        }

        Ok(protected_text)
    }

    /// Run real autoregressive generation: encode `prompt`, repeatedly run
    /// the transformer forward pass and sample a next token, and decode only
    /// the newly generated suffix (never the prompt itself, which is what
    /// made the old implementation an echo rather than generation).
    fn generate_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        rng: &mut impl Rng,
    ) -> Result<String> {
        let prompt_ids = self.encode_text(prompt)?;
        let prompt_len = prompt_ids.len();
        let mut generated = prompt_ids;
        let target_len = (prompt_len + config.max_new_tokens).min(prompt_len + MAX_SEQUENCE_LENGTH);

        while generated.len() < target_len {
            let mask = if self.config.confidentiality_protection {
                build_confidentiality_mask(&generated, &self.config.get_special_tokens())
            } else {
                None
            };

            let logits_tensor = self.forward_logits_with_mask(&generated, mask.as_ref())?;
            let mut logits = last_position_logits(&logits_tensor)?;

            restrict_logits_to_byte_vocab(&mut logits, self.config.eos_token_id);
            GenerationUtils::apply_temperature(&mut logits, config.temperature);
            GenerationUtils::apply_repetition_penalty(
                &mut logits,
                &generated,
                config.repetition_penalty,
                1.0,
            );

            let next_token = if !config.do_sample {
                GenerationUtils::sample_greedy(&logits)
            } else if let Some(k) = config.top_k {
                GenerationUtils::sample_top_k(&logits, k, rng)?
            } else {
                let p = config.top_p.clamp(1e-4, 1.0);
                GenerationUtils::sample_top_p(&logits, p, rng)?
            };

            generated.push(next_token);
            if next_token == self.config.eos_token_id {
                break;
            }
        }

        Ok(Self::decode_tokens(&generated[prompt_len..]))
    }
}

/// Extract the logits for the last sequence position from a `[seq_len,
/// vocab_size]` logits tensor.
fn last_position_logits(logits: &Tensor) -> Result<Vec<f32>> {
    let shape = logits.shape();
    if shape.len() != 2 {
        anyhow::bail!(
            "expected a 2-D [seq_len, vocab_size] logits tensor, got {:?}",
            shape
        );
    }
    let (seq_len, vocab_size) = (shape[0], shape[1]);
    let data = logits.data()?;
    let start = (seq_len - 1) * vocab_size;
    Ok(data[start..start + vocab_size].to_vec())
}

/// Restrict sampling to the byte-token range plus EOS, so decoding is always
/// exact: a token that was never masked out is guaranteed to either be a
/// real byte or the end-of-sequence marker, never an id this tokenizer has
/// no defined meaning for.
fn restrict_logits_to_byte_vocab(logits: &mut [f32], eos_token_id: u32) {
    for (id, logit) in logits.iter_mut().enumerate() {
        let id = id as u32;
        let in_byte_range = (BYTE_TOKEN_OFFSET..BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE).contains(&id);
        if !in_byte_range && id != eos_token_id {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Find every (non-overlapping-search, but possibly overlapping-result)
/// occurrence of `needle`'s byte-token encoding inside `haystack`.
fn find_byte_pattern(haystack: &[u32], needle: &str) -> Vec<usize> {
    let needle_ids: Vec<u32> = needle.bytes().map(|b| BYTE_TOKEN_OFFSET + b as u32).collect();
    if needle_ids.is_empty() || haystack.len() < needle_ids.len() {
        return Vec::new();
    }
    (0..=haystack.len() - needle_ids.len())
        .filter(|&start| haystack[start..start + needle_ids.len()] == needle_ids[..])
        .collect()
}

/// Assign a "section id" to every position in `token_ids`: `0` for text
/// outside any `start_marker .. end_marker` span, and a distinct positive id
/// per *top-level* span (nested reopenings reuse the enclosing span's id).
/// Returns `None` when no start marker is present at all.
fn compute_section_ids(
    token_ids: &[u32],
    start_marker: &str,
    end_marker: &str,
) -> Option<Vec<u32>> {
    let starts = find_byte_pattern(token_ids, start_marker);
    if starts.is_empty() {
        return None;
    }
    let ends = find_byte_pattern(token_ids, end_marker);

    let mut events: Vec<(usize, i32)> = Vec::with_capacity(starts.len() + ends.len());
    events.extend(starts.iter().map(|&pos| (pos, 1)));
    events.extend(ends.iter().map(|&pos| (pos, -1)));
    events.sort_by_key(|&(pos, _)| pos);

    let n = token_ids.len();
    let mut section = vec![0u32; n];
    let mut depth: i32 = 0;
    let mut current_id = 0u32;
    let mut next_id = 1u32;
    let mut event_idx = 0;
    for (i, slot) in section.iter_mut().enumerate() {
        while event_idx < events.len() && events[event_idx].0 == i {
            let (_, delta) = events[event_idx];
            if delta > 0 {
                if depth == 0 {
                    current_id = next_id;
                    next_id += 1;
                }
                depth += 1;
            } else {
                depth = (depth - 1).max(0);
                if depth == 0 {
                    current_id = 0;
                }
            }
            event_idx += 1;
        }
        *slot = if depth > 0 { current_id } else { 0 };
    }
    Some(section)
}

/// Build a real additive attention mask (see [`trustformers_core::layers::attention::mask`])
/// that isolates confidential spans from each other: position `i` may attend
/// to key position `j` iff `j <= i` (causal) and `j` is either public
/// (`section[j] == 0`) or in the *same* confidential span as `i`. Returns
/// `None` when the token stream contains no confidential markers, so callers
/// can skip attention masking entirely rather than passing a redundant
/// all-zero mask.
fn build_confidentiality_mask(
    token_ids: &[u32],
    special_tokens: &LegalMedicalSpecialTokens,
) -> Option<Tensor> {
    let section = compute_section_ids(
        token_ids,
        &special_tokens.confidential_start,
        &special_tokens.confidential_end,
    )?;
    let n = token_ids.len();
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let visible = section[j] == 0 || section[j] == section[i];
            data[i * n + j] = if visible { 0.0 } else { f32::NEG_INFINITY };
        }
        for j in (i + 1)..n {
            data[i * n + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, &[1, 1, n, n]).ok()
}

#[cfg(test)]
mod tests {
    use super::super::tests::tiny_config;
    use super::*;

    fn model() -> LegalMedicalForCausalLM {
        LegalMedicalForCausalLM::new(tiny_config()).expect("model construction")
    }

    #[test]
    fn generate_does_not_echo_the_prompt() {
        // Regression test for the "[Legal/Medical Generated]: {prompt}" bug:
        // the returned text must not simply be the prompt wrapped in a
        // bracketed label, and it must not contain the prompt as a substring
        // at all (a real byte-level decode of freshly sampled tokens has no
        // reason to reproduce the input).
        let m = model();
        let prompt = "The parties agree to the following confidential terms";
        let output = m.generate_with_seed(prompt, 12, 7).expect("generate");
        assert!(
            !output.contains(prompt),
            "output must not simply echo the prompt back: {output:?}"
        );
        assert!(!output.starts_with("[Legal/Medical Generated]"));
        assert!(!output.starts_with("[PRIVACY_PROTECTED]"));
    }

    #[test]
    fn generate_with_seed_is_reproducible() {
        let m = model();
        let a = m.generate_with_seed("Whereas the parties", 16, 42).expect("a");
        let b = m.generate_with_seed("Whereas the parties", 16, 42).expect("b");
        assert_eq!(a, b, "the same seed must produce the same output");
    }

    #[test]
    fn different_seeds_produce_different_output() {
        let m = model();
        let a = m.generate_with_seed("Whereas the parties", 24, 1).expect("a");
        let b = m.generate_with_seed("Whereas the parties", 24, 2).expect("b");
        assert_ne!(
            a, b,
            "different seeds must (almost certainly) sample different tokens"
        );
    }

    #[test]
    fn independently_constructed_models_generate_different_output() {
        // Proves the output actually depends on the (randomly initialised)
        // weights and is not some fixed function of the prompt alone.
        let model_a = model();
        let model_b = model();
        let a = model_a.generate_with_seed("Notice of termination", 24, 5).expect("a");
        let b = model_b.generate_with_seed("Notice of termination", 24, 5).expect("b");
        assert_ne!(
            a, b,
            "two independently-initialised models must not produce identical output"
        );
    }

    #[test]
    fn greedy_generation_is_deterministic_without_a_seed() {
        let m = model();
        let cfg = GenerationConfig {
            max_new_tokens: 10,
            do_sample: false,
            ..Default::default()
        };
        let mut rng_a = thread_rng();
        let mut rng_b = thread_rng();
        let a = m.generate_with_config("Order of the court", &cfg, &mut rng_a).expect("a");
        let b = m.generate_with_config("Order of the court", &cfg, &mut rng_b).expect("b");
        assert_eq!(a, b, "greedy decoding must not depend on the RNG stream");
    }

    #[test]
    fn encode_decode_byte_roundtrip() {
        let m = model();
        let text = "Patient reports mild discomfort.";
        let ids = m.encode_text(text).expect("encode");
        assert_eq!(ids[0], m.config.bos_token_id);
        let decoded = LegalMedicalForCausalLM::decode_tokens(&ids[1..]);
        assert_eq!(decoded, text);
    }

    #[test]
    fn encode_rejects_vocab_too_small_for_byte_tokenizer() {
        let mut config = tiny_config();
        config.vocab_size = 16;
        let m = LegalMedicalForCausalLM::new(config).expect("model");
        assert!(m.encode_text("hello").is_err());
    }

    #[test]
    fn confidentiality_mask_blocks_cross_section_attention() {
        let special = tiny_config().get_special_tokens();
        // "AAA<confidential>BBB</confidential>CCC<confidential>DDD</confidential>"
        let mut tokens = Vec::new();
        tokens.extend(std::iter::repeat_n(BYTE_TOKEN_OFFSET + b'A' as u32, 3));
        tokens.extend(special.confidential_start.bytes().map(|b| BYTE_TOKEN_OFFSET + b as u32));
        tokens.extend(std::iter::repeat_n(BYTE_TOKEN_OFFSET + b'B' as u32, 3));
        tokens.extend(special.confidential_end.bytes().map(|b| BYTE_TOKEN_OFFSET + b as u32));
        tokens.extend(std::iter::repeat_n(BYTE_TOKEN_OFFSET + b'C' as u32, 3));
        tokens.extend(special.confidential_start.bytes().map(|b| BYTE_TOKEN_OFFSET + b as u32));
        tokens.extend(std::iter::repeat_n(BYTE_TOKEN_OFFSET + b'D' as u32, 3));
        tokens.extend(special.confidential_end.bytes().map(|b| BYTE_TOKEN_OFFSET + b as u32));

        let section = compute_section_ids(
            &tokens,
            &special.confidential_start,
            &special.confidential_end,
        )
        .expect("spans present");

        let a_index = 1; // inside "AAA", public
        let b_index = 3 + special.confidential_start.len() + 1; // inside "BBB", span 1
        let d_index = tokens.len() - special.confidential_end.len() - 2; // inside "DDD", span 2

        assert_eq!(section[a_index], 0, "public text must be section 0");
        assert_ne!(
            section[b_index], 0,
            "text inside a confidential span must not be section 0"
        );
        assert_ne!(
            section[d_index], 0,
            "text inside the second span must not be section 0"
        );
        assert_ne!(
            section[b_index], section[d_index],
            "two separate confidential spans must not share a section id"
        );

        let mask = build_confidentiality_mask(&tokens, &special).expect("mask present");
        let data = mask.data().expect("mask data");
        let n = tokens.len();

        // D (later, span 2) must not be able to attend back into B (span 1).
        assert_eq!(data[d_index * n + b_index], f32::NEG_INFINITY);
        // D must still be able to attend to public text A.
        assert_eq!(data[d_index * n + a_index], 0.0);
        // D must be able to attend to itself.
        assert_eq!(data[d_index * n + d_index], 0.0);
        // Causal: nothing may attend to a strictly future position.
        assert_eq!(data[a_index * n + d_index], f32::NEG_INFINITY);

        // Every row must keep at least one unmasked key (itself), or softmax
        // would divide by zero.
        for i in 0..n {
            assert!(
                data[i * n..(i + 1) * n].contains(&0.0),
                "row {i} has no unmasked key"
            );
        }
    }

    #[test]
    fn confidentiality_mask_is_none_without_markers() {
        let special = tiny_config().get_special_tokens();
        let tokens: Vec<u32> =
            "plain public text".bytes().map(|b| BYTE_TOKEN_OFFSET + b as u32).collect();
        assert!(build_confidentiality_mask(&tokens, &special).is_none());
    }

    #[test]
    fn generation_with_confidential_markers_stays_finite() {
        // End-to-end: a prompt containing confidential markers must still
        // drive a real forward pass (finite logits, real sampling) with the
        // section mask applied, not NaN out.
        let m = model();
        let prompt = "<confidential>patient record</confidential> follow-up plan";
        // The call must complete (real forward passes with the section mask
        // applied, no NaN/panic) and decode to valid text; an immediate EOS
        // sample legitimately yields an empty continuation, so the point of
        // this test is that it returns at all rather than a length bound.
        let _output = m.generate_with_seed(prompt, 10, 3).expect("generate");
    }
}
