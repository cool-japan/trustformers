//! Pure-Rust decoding core shared by `GPT2LMHeadModel.generate()` and the
//! `text-generation` pipeline.
//!
//! Both entry points drive the *same* real decoder --
//! [`trustformers_core::generation::TextGenerator`] -- so there is exactly one
//! place where sampling controls are translated into a
//! [`GenerationConfig`], and no way for one of them to quietly ignore an
//! argument the other honours. (`TextGenerator` itself refuses configurations
//! it cannot honour rather than returning text that silently drops them.)
//!
//! Free of the Python C API, so the decoding path is unit-testable with a
//! plain `cargo test` against a real -- if tiny -- GPT-2 model (the test
//! binary does not link `libpython`; see `weights.rs`, `losses.rs` and
//! `inputs.rs` for the same split).

use super::inputs::tokenized_input_from_ids;
use trustformers_core::errors::{runtime_error, TrustformersError};
use trustformers_core::generation::{GenerationConfig, GenerationStrategy, TextGenerator};
use trustformers_core::traits::Model;
use trustformers_models::gpt2::Gpt2LMHeadModel;

/// The HuggingFace-shaped generation controls this crate exposes to Python.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SamplingOptions {
    /// Maximum *total* sequence length (prompt + generated), as in HF.
    pub max_length: usize,
    /// Minimum *total* length (prompt + generated) before an EOS token is
    /// allowed to stop decoding.
    ///
    /// `0` means "no minimum", matching HuggingFace's own default: the core
    /// decoder compares `sequence.len() >= min_length.unwrap_or(0)`, so
    /// `Some(0)` and `None` are the same condition there. It is deliberately
    /// not [`GenerationConfig::default`]'s `Some(1)`, which is a core-level
    /// default rather than a pipeline one.
    pub min_length: usize,
    /// Sample from the distribution instead of taking the argmax.
    pub do_sample: bool,
    /// Softmax temperature; only meaningful when `do_sample` is set.
    pub temperature: f32,
    /// Top-k truncation; only meaningful when `do_sample` is set.
    pub top_k: Option<usize>,
    /// Nucleus (top-p) truncation; takes precedence over `top_k`, as in HF.
    pub top_p: Option<f32>,
    /// How many independent sequences to return.
    pub num_return_sequences: usize,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            max_length: 50,
            min_length: 0,
            do_sample: true,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            num_return_sequences: 1,
        }
    }
}

/// Pick the decoding strategy the options describe.
///
/// Mirrors HuggingFace's precedence: `do_sample=False` is greedy regardless of
/// the sampling knobs, and `top_p` wins over `top_k` when both are set.
pub(crate) fn generation_strategy(options: &SamplingOptions) -> GenerationStrategy {
    if !options.do_sample {
        GenerationStrategy::Greedy
    } else if let Some(p) = options.top_p {
        GenerationStrategy::TopP {
            p,
            temperature: options.temperature,
        }
    } else if let Some(k) = options.top_k {
        GenerationStrategy::TopK {
            k,
            temperature: options.temperature,
        }
    } else {
        GenerationStrategy::Sampling {
            temperature: options.temperature,
        }
    }
}

/// Build the [`GenerationConfig`] for `options` against a model whose EOS token
/// is `eos_token_id`.
///
/// `use_cache` is deliberately `false`: GPT-2's KV cache is not yet threaded
/// through the Rust core forward pass, so every step recomputes the full
/// prefix. Claiming `use_cache: true` here would advertise an optimisation
/// that does not happen.
pub(crate) fn generation_config(
    options: &SamplingOptions,
    eos_token_id: usize,
) -> GenerationConfig {
    GenerationConfig {
        strategy: generation_strategy(options),
        max_length: Some(options.max_length),
        min_length: Some(options.min_length),
        do_sample: options.do_sample,
        num_return_sequences: options.num_return_sequences,
        use_cache: false,
        eos_token_id: Some(eos_token_id),
        ..GenerationConfig::default()
    }
}

/// Decode continuations of `prompt` with a real GPT-2 language-model head.
///
/// Returns one token sequence per requested return sequence, each including
/// the prompt tokens (the shape HuggingFace's `generate` returns).
///
/// # Errors
///
/// Fails when `prompt` is empty, when the model's forward pass fails, or when
/// the decoder refuses the configuration (for example
/// `num_return_sequences > 1` with greedy decoding, which cannot produce more
/// than one distinct sequence).
pub(crate) fn generate_with_gpt2(
    model: &Gpt2LMHeadModel,
    prompt: &[usize],
    options: &SamplingOptions,
) -> Result<Vec<Vec<usize>>, TrustformersError> {
    if prompt.is_empty() {
        return Err(runtime_error(
            "generation requires at least one input token".to_string(),
        ));
    }

    let config = model.get_config();
    let generator = TextGenerator::new(
        generation_config(options, config.eos_token_id as usize),
        config.vocab_size,
    );

    let sequences = generator.generate(prompt, |sequence, _cache| {
        let token_ids: Vec<u32> = sequence.iter().map(|&token| token as u32).collect();
        let output = model.forward(tokenized_input_from_ids(&token_ids))?;
        Ok((output.logits, None))
    })?;

    if sequences.is_empty() {
        return Err(runtime_error(
            "the decoder returned no sequences".to_string(),
        ));
    }
    Ok(sequences)
}

/// The tokens `generate_with_gpt2` appended to `prompt` in `sequence`.
///
/// Returns an error rather than a silently empty continuation when the decoder
/// hands back a sequence that is not an extension of the prompt.
pub(crate) fn continuation_tokens<'a>(
    prompt: &[usize],
    sequence: &'a [usize],
) -> Result<&'a [usize], TrustformersError> {
    if sequence.len() < prompt.len() || &sequence[..prompt.len()] != prompt {
        return Err(runtime_error(format!(
            "the generated sequence ({} tokens) does not extend the {}-token prompt",
            sequence.len(),
            prompt.len()
        )));
    }
    Ok(&sequence[prompt.len()..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_models::gpt2::Gpt2Config;

    /// A GPT-2 small enough to run a real forward pass per decoding step in a
    /// unit test. Randomly initialised, which is all that is needed: these
    /// tests assert on the *decoding contract*, not on what a trained model
    /// would say.
    fn tiny_gpt2_with_eos(eos_token_id: u32) -> Gpt2LMHeadModel {
        let config = Gpt2Config {
            vocab_size: 32,
            n_positions: 16,
            n_embd: 16,
            n_layer: 1,
            n_head: 2,
            eos_token_id,
            bos_token_id: eos_token_id,
            ..Gpt2Config::default()
        };
        Gpt2LMHeadModel::new(config).expect("tiny GPT-2 config is valid")
    }

    /// A tiny GPT-2 whose EOS id (`63`) lies outside its 32-token vocabulary,
    /// so it can never be produced.
    ///
    /// These models are *randomly* initialised (`trustformers-models` asserts
    /// as much in its own tests), so with an in-vocabulary EOS the decoder
    /// could stop at any step and any assertion on the exact output length
    /// would be flaky. Making EOS unreachable leaves `max_length` as the only
    /// stop condition, which is what the length tests are actually about;
    /// `generation_stops_at_eos_or_max_length` covers the EOS path separately,
    /// with an assertion that holds either way.
    fn tiny_gpt2() -> Gpt2LMHeadModel {
        tiny_gpt2_with_eos(63)
    }

    fn greedy(max_length: usize) -> SamplingOptions {
        SamplingOptions {
            max_length,
            do_sample: false,
            ..SamplingOptions::default()
        }
    }

    #[test]
    fn strategy_follows_huggingface_precedence() {
        let base = SamplingOptions::default();
        assert!(matches!(
            generation_strategy(&SamplingOptions {
                do_sample: false,
                top_p: Some(0.9),
                top_k: Some(5),
                ..base
            }),
            GenerationStrategy::Greedy
        ));
        assert!(matches!(
            generation_strategy(&SamplingOptions {
                top_p: Some(0.9),
                top_k: Some(5),
                ..base
            }),
            GenerationStrategy::TopP { .. }
        ));
        assert!(matches!(
            generation_strategy(&SamplingOptions {
                top_k: Some(5),
                ..base
            }),
            GenerationStrategy::TopK { .. }
        ));
        assert!(matches!(
            generation_strategy(&base),
            GenerationStrategy::Sampling { .. }
        ));
    }

    /// The pipeline's `max_length` / `min_length` / `num_return_sequences`
    /// arguments used to be swallowed by `let _ = (...)`; they must reach the
    /// decoder.
    #[test]
    fn options_reach_the_generation_config() {
        let config = generation_config(
            &SamplingOptions {
                max_length: 12,
                min_length: 4,
                num_return_sequences: 3,
                ..SamplingOptions::default()
            },
            31,
        );
        assert_eq!(config.max_length, Some(12));
        assert_eq!(config.min_length, Some(4));
        assert_eq!(config.num_return_sequences, 3);
        assert_eq!(config.eos_token_id, Some(31));
        assert!(!config.use_cache, "the KV cache is not wired up yet");
    }

    /// The replaced pipeline returned `format!("{text} [Generated continuation]")`
    /// without ever touching a model. This runs the real decoder over a real
    /// forward pass and checks the decoding contract: the prompt is preserved
    /// and the sequence grows to `max_length`.
    #[test]
    fn greedy_decoding_extends_the_prompt_to_max_length() {
        let model = tiny_gpt2();
        let prompt = vec![1usize, 2, 3];
        let sequences =
            generate_with_gpt2(&model, &prompt, &greedy(8)).expect("greedy decoding succeeds");
        assert_eq!(sequences.len(), 1);
        let sequence = &sequences[0];
        assert_eq!(&sequence[..prompt.len()], prompt.as_slice());
        assert_eq!(sequence.len(), 8);
        assert!(sequence.iter().all(|&token| token < 32));
    }

    /// The real stop contract with a reachable EOS: never longer than
    /// `max_length`, always longer than the prompt, and it stops either at
    /// `max_length` or on an EOS token. Holds whatever the random weights make
    /// the model say.
    #[test]
    fn generation_stops_at_eos_or_max_length() {
        let eos = 31u32;
        let model = tiny_gpt2_with_eos(eos);
        let prompt = vec![1usize, 2, 3];
        let sequences =
            generate_with_gpt2(&model, &prompt, &greedy(8)).expect("greedy decoding succeeds");
        let sequence = &sequences[0];
        assert_eq!(&sequence[..prompt.len()], prompt.as_slice());
        assert!(sequence.len() > prompt.len(), "at least one token is generated");
        assert!(sequence.len() <= 8, "max_length is never exceeded");
        assert!(
            sequence.len() == 8 || sequence.last() == Some(&(eos as usize)),
            "a short sequence must have stopped on EOS"
        );
    }

    /// `min_length` is a *total* length and `0` means "no minimum" -- the same
    /// condition as `None` in the core decoder. It must not silently become
    /// `GenerationConfig::default()`'s `Some(1)`.
    #[test]
    fn a_zero_min_length_is_passed_through_as_no_minimum() {
        let config = generation_config(&SamplingOptions::default(), 31);
        assert_eq!(config.min_length, Some(0));
    }

    /// Greedy decoding is deterministic, so the same prompt must give the same
    /// continuation -- impossible to observe when the "generation" was a
    /// format string.
    #[test]
    fn greedy_decoding_is_deterministic() {
        let model = tiny_gpt2();
        let prompt = vec![4usize, 5];
        let first = generate_with_gpt2(&model, &prompt, &greedy(7)).expect("first run");
        let second = generate_with_gpt2(&model, &prompt, &greedy(7)).expect("second run");
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_an_empty_prompt() {
        let model = tiny_gpt2();
        assert!(generate_with_gpt2(&model, &[], &greedy(4)).is_err());
    }

    /// Greedy decoding cannot produce several distinct sequences, and the core
    /// decoder says so instead of returning duplicates.
    #[test]
    fn rejects_multiple_returns_from_greedy_decoding() {
        let model = tiny_gpt2();
        let options = SamplingOptions {
            max_length: 6,
            do_sample: false,
            num_return_sequences: 3,
            ..SamplingOptions::default()
        };
        assert!(generate_with_gpt2(&model, &[1, 2], &options).is_err());
    }

    #[test]
    fn sampling_returns_the_requested_number_of_sequences() {
        let model = tiny_gpt2();
        let options = SamplingOptions {
            max_length: 6,
            do_sample: true,
            temperature: 1.0,
            num_return_sequences: 3,
            ..SamplingOptions::default()
        };
        let sequences = generate_with_gpt2(&model, &[1, 2], &options).expect("sampling succeeds");
        assert_eq!(sequences.len(), 3);
        for sequence in &sequences {
            assert_eq!(&sequence[..2], &[1, 2]);
        }
    }

    #[test]
    fn continuation_tokens_strips_exactly_the_prompt() {
        assert_eq!(
            continuation_tokens(&[1, 2], &[1, 2, 3, 4]).expect("valid extension"),
            &[3, 4]
        );
        assert_eq!(
            continuation_tokens(&[1, 2], &[1, 2]).expect("empty continuation"),
            &[] as &[usize]
        );
        assert!(continuation_tokens(&[1, 2], &[9, 9, 3]).is_err());
        assert!(continuation_tokens(&[1, 2], &[1]).is_err());
    }
}
