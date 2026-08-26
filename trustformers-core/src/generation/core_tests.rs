//! Behavioural tests for [`crate::generation::core::TextGenerator`].
//!
//! Every strategy is checked against a hand-built toy model whose exact
//! distribution is known, so that a strategy which silently degrades into
//! greedy decoding fails these tests.

#[cfg(test)]
mod tests {
    use crate::errors::Result;
    use crate::generation::cache::KVCache;
    use crate::generation::config::{GenerationConfig, GenerationStrategy};
    use crate::generation::core::TextGenerator;
    use crate::tensor::Tensor;

    // ------------------------------------------------------------------
    // Toy models
    // ------------------------------------------------------------------

    fn tensor_1d(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), &[values.len()]).expect("logits tensor")
    }

    fn log_of(probs: &[f32]) -> Vec<f32> {
        probs.iter().map(|p| p.ln()).collect()
    }

    /// A model whose next-token distribution never changes.
    fn constant_model(
        logits: Vec<f32>,
    ) -> impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)> {
        move |_tokens: &[usize], _cache: Option<&KVCache>| Ok((tensor_1d(&logits), None))
    }

    /// A model whose distribution depends only on the last emitted token.
    fn markov_model(
        table: Vec<Vec<f32>>,
    ) -> impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)> {
        move |tokens: &[usize], _cache: Option<&KVCache>| {
            let last = tokens.last().copied().unwrap_or(0);
            let row = table.get(last).unwrap_or(&table[0]);
            Ok((tensor_1d(&log_of(row)), None))
        }
    }

    /// Two-step model where the locally best first token leads to a globally
    /// worse continuation.  Greedy picks `1`, beam search must find `2`.
    ///
    /// | state       | distribution                          |
    /// |-------------|---------------------------------------|
    /// | after `0`   | `[.02, .50, .45, .02, .01]`           |
    /// | after `1`   | `[.15, .15, .15, .40, .15]`           |
    /// | after `2`   | `[.005, .005, .005, .98, .005]`       |
    ///
    /// `log .50 + log .40 = -1.609` versus `log .45 + log .98 = -0.819`.
    fn beam_trap_model() -> impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>
    {
        markov_model(vec![
            vec![0.02, 0.50, 0.45, 0.02, 0.01],
            vec![0.15, 0.15, 0.15, 0.40, 0.15],
            vec![0.005, 0.005, 0.005, 0.98, 0.005],
            vec![0.2, 0.2, 0.2, 0.2, 0.2],
            vec![0.2, 0.2, 0.2, 0.2, 0.2],
        ])
    }

    fn config(strategy: GenerationStrategy, max_new_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            strategy,
            max_length: None,
            max_new_tokens: Some(max_new_tokens),
            min_length: None,
            ..Default::default()
        }
    }

    // ------------------------------------------------------------------
    // 1. Greedy baseline
    // ------------------------------------------------------------------

    #[test]
    fn test_greedy_returns_prompt_plus_generated_tokens() {
        let generator = TextGenerator::new(config(GenerationStrategy::Greedy, 3), 4);
        let sequences = generator
            .generate(&[0], constant_model(vec![0.1, 0.9, 0.4, 0.2]))
            .expect("generate");
        assert_eq!(sequences.len(), 1);
        assert_eq!(sequences[0], vec![0, 1, 1, 1]);
    }

    #[test]
    fn test_max_new_tokens_is_a_budget_not_a_total() {
        // Regression: the old loop ran `max_length` times regardless of the
        // prompt length, emitting far more tokens than requested.
        let generator = TextGenerator::new(config(GenerationStrategy::Greedy, 2), 4);
        let sequences = generator
            .generate(&[0, 0, 0], constant_model(vec![0.1, 0.9, 0.4, 0.2]))
            .expect("generate");
        assert_eq!(sequences[0].len(), 5, "3 prompt + 2 generated");
    }

    // ------------------------------------------------------------------
    // 2. Top-k
    // ------------------------------------------------------------------

    #[test]
    fn test_top_k_one_equals_greedy() {
        let logits = vec![0.3_f32, 1.7, 0.9, -2.0, 1.6];
        let greedy = TextGenerator::new(config(GenerationStrategy::Greedy, 4), 5)
            .generate(&[0], constant_model(logits.clone()))
            .expect("greedy");
        let top_k = TextGenerator::new(
            config(
                GenerationStrategy::TopK {
                    k: 1,
                    temperature: 1.0,
                },
                4,
            ),
            5,
        )
        .with_seed(11)
        .generate(&[0], constant_model(logits))
        .expect("top-k");

        assert_eq!(greedy, top_k, "top_k = 1 must reduce to argmax decoding");
    }

    #[test]
    fn test_top_k_restricts_the_support_but_is_not_greedy() {
        // Regression: top-k used to fall through to greedy, so every draw was
        // the argmax.  With k = 2 exactly tokens 1 and 4 may appear.
        let logits = vec![0.3_f32, 1.7, 0.9, -2.0, 1.6];
        let generator = TextGenerator::new(
            config(
                GenerationStrategy::TopK {
                    k: 2,
                    temperature: 1.0,
                },
                200,
            ),
            5,
        )
        .with_seed(2024);
        let sequences = generator.generate(&[0], constant_model(logits)).expect("top-k");

        let generated = &sequences[0][1..];
        assert!(
            generated.iter().all(|&token| token == 1 || token == 4),
            "sampled outside the top-2 set: {generated:?}"
        );
        assert!(
            generated.contains(&4),
            "top-k never explored the second-best token"
        );
    }

    // ------------------------------------------------------------------
    // 3. Top-p
    // ------------------------------------------------------------------

    #[test]
    fn test_top_p_respects_the_probability_mass_cutoff() {
        // Probabilities [.05, .60, .25, .07, .03]; sorted descending the 0.8
        // nucleus is {.60, .25} -> tokens 1 and 2 only.
        let probs = vec![0.05_f32, 0.60, 0.25, 0.07, 0.03];
        let generator = TextGenerator::new(
            config(
                GenerationStrategy::TopP {
                    p: 0.8,
                    temperature: 1.0,
                },
                300,
            ),
            5,
        )
        .with_seed(99);
        let sequences = generator.generate(&[0], constant_model(log_of(&probs))).expect("top-p");

        let generated = &sequences[0][1..];
        assert!(
            generated.iter().all(|&token| token == 1 || token == 2),
            "top-p sampled outside the nucleus: {generated:?}"
        );
        assert!(generated.contains(&2), "top-p collapsed to the argmax");
    }

    #[test]
    fn test_top_p_tiny_mass_is_deterministic() {
        let probs = vec![0.05_f32, 0.60, 0.25, 0.07, 0.03];
        let generator = TextGenerator::new(
            config(
                GenerationStrategy::TopP {
                    p: 0.1,
                    temperature: 1.0,
                },
                20,
            ),
            5,
        )
        .with_seed(5);
        let sequences = generator.generate(&[0], constant_model(log_of(&probs))).expect("top-p");
        assert!(sequences[0][1..].iter().all(|&token| token == 1));
    }

    // ------------------------------------------------------------------
    // 4. Plain sampling
    // ------------------------------------------------------------------

    #[test]
    fn test_sampling_is_stochastic_and_seed_reproducible() {
        // Regression: sampling used to be argmax, producing one repeated token.
        let logits = vec![0.0_f32, 0.0, 0.0, 0.0];
        let make = |seed: u64| {
            TextGenerator::new(
                config(GenerationStrategy::Sampling { temperature: 1.0 }, 40),
                4,
            )
            .with_seed(seed)
            .generate(&[0], constant_model(logits.clone()))
            .expect("sampling")
        };

        let first = make(1234);
        let same = make(1234);
        let other = make(4321);

        assert_eq!(
            first, same,
            "the same seed must reproduce the same sequence"
        );
        assert!(
            first[0][1..].windows(2).any(|pair| pair[0] != pair[1]),
            "a uniform distribution must not decode to a constant: {:?}",
            first[0]
        );
        assert_ne!(first, other, "different seeds should diverge");
    }

    #[test]
    fn test_low_temperature_sampling_converges_to_greedy() {
        let logits = vec![0.0_f32, 1.0, 0.5, 0.2];
        let greedy = TextGenerator::new(config(GenerationStrategy::Greedy, 12), 4)
            .generate(&[0], constant_model(logits.clone()))
            .expect("greedy");
        let cold = TextGenerator::new(
            config(GenerationStrategy::Sampling { temperature: 0.001 }, 12),
            4,
        )
        .with_seed(77)
        .generate(&[0], constant_model(logits))
        .expect("sampling");
        assert_eq!(greedy, cold);
    }

    #[test]
    fn test_invalid_temperature_is_an_error() {
        let generator = TextGenerator::new(
            config(GenerationStrategy::Sampling { temperature: 0.0 }, 4),
            4,
        );
        assert!(generator.generate(&[0], constant_model(vec![0.0, 1.0, 0.0, 0.0])).is_err());
    }

    // ------------------------------------------------------------------
    // 5. Beam search
    // ------------------------------------------------------------------

    #[test]
    fn test_beam_size_one_matches_greedy() {
        let greedy = TextGenerator::new(config(GenerationStrategy::Greedy, 2), 5)
            .generate(&[0], beam_trap_model())
            .expect("greedy");
        let beam = TextGenerator::new(
            config(GenerationStrategy::BeamSearch { num_beams: 1 }, 2),
            5,
        )
        .generate(&[0], beam_trap_model())
        .expect("beam");

        assert_eq!(greedy, beam, "beam width 1 is greedy decoding");
    }

    #[test]
    fn test_beam_search_finds_the_globally_better_path() {
        // Regression: beam search used to delegate to greedy, which walks into
        // the locally-best-but-globally-worse token 1.
        let greedy = TextGenerator::new(config(GenerationStrategy::Greedy, 2), 5)
            .generate(&[0], beam_trap_model())
            .expect("greedy");
        assert_eq!(greedy[0], vec![0, 1, 3], "greedy takes the trap");

        let beam = TextGenerator::new(
            config(GenerationStrategy::BeamSearch { num_beams: 2 }, 2),
            5,
        )
        .generate(&[0], beam_trap_model())
        .expect("beam");
        assert_eq!(
            beam[0],
            vec![0, 2, 3],
            "beam search must recover the higher-probability path"
        );
    }

    #[test]
    fn test_beam_search_returns_multiple_ranked_sequences() {
        let mut cfg = config(GenerationStrategy::BeamSearch { num_beams: 3 }, 2);
        cfg.num_return_sequences = 3;
        let sequences = TextGenerator::new(cfg, 5).generate(&[0], beam_trap_model()).expect("beam");

        assert_eq!(sequences.len(), 3);
        assert_eq!(sequences[0], vec![0, 2, 3], "best hypothesis comes first");
        for sequence in &sequences {
            assert_eq!(sequence[0], 0, "the prompt must be preserved");
        }
        assert!(
            sequences[1] != sequences[0] && sequences[2] != sequences[0],
            "returned hypotheses must differ: {sequences:?}"
        );
    }

    #[test]
    fn test_beam_search_rejects_more_returns_than_beams() {
        let mut cfg = config(GenerationStrategy::BeamSearch { num_beams: 2 }, 2);
        cfg.num_return_sequences = 5;
        assert!(TextGenerator::new(cfg, 5).generate(&[0], beam_trap_model()).is_err());
    }

    // ------------------------------------------------------------------
    // 6. Contrastive search
    // ------------------------------------------------------------------

    /// Model where token 1 leads back to the state the context is already in
    /// and token 2 leads somewhere new.
    fn degeneration_model(
    ) -> impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)> {
        markov_model(vec![
            vec![0.05, 0.50, 0.40, 0.05],
            vec![0.05, 0.50, 0.40, 0.05], // identical to the state after `0`
            vec![0.25, 0.25, 0.25, 0.25], // a genuinely new state
            vec![0.25, 0.25, 0.25, 0.25],
        ])
    }

    #[test]
    fn test_contrastive_search_avoids_the_degenerate_token() {
        // Regression: contrastive search used to fall through to greedy, which
        // always picks token 1 here.
        let greedy = TextGenerator::new(config(GenerationStrategy::Greedy, 1), 4)
            .generate(&[0], degeneration_model())
            .expect("greedy");
        assert_eq!(greedy[0], vec![0, 1]);

        let contrastive = TextGenerator::new(
            config(
                GenerationStrategy::ContrastiveSearch {
                    penalty_alpha: 0.6,
                    top_k: 2,
                },
                1,
            ),
            4,
        )
        .generate(&[0], degeneration_model())
        .expect("contrastive");

        assert_eq!(
            contrastive[0],
            vec![0, 2],
            "the degeneration penalty must demote the repeating token"
        );
    }

    #[test]
    fn test_contrastive_search_alpha_zero_is_greedy() {
        let greedy = TextGenerator::new(config(GenerationStrategy::Greedy, 3), 4)
            .generate(&[0], degeneration_model())
            .expect("greedy");
        let contrastive = TextGenerator::new(
            config(
                GenerationStrategy::ContrastiveSearch {
                    penalty_alpha: 0.0,
                    top_k: 4,
                },
                3,
            ),
            4,
        )
        .generate(&[0], degeneration_model())
        .expect("contrastive");
        assert_eq!(greedy, contrastive);
    }

    #[test]
    fn test_contrastive_search_with_hidden_states() {
        // Hidden states: state after `1` repeats the prompt state exactly,
        // state after `2` is orthogonal to it.
        let generator = TextGenerator::new(
            config(
                GenerationStrategy::ContrastiveSearch {
                    penalty_alpha: 0.5,
                    top_k: 2,
                },
                1,
            ),
            4,
        );

        let sequences = generator
            .generate_contrastive_search_with_hidden_states(&[0], 0.5, 2, |tokens| {
                let last = tokens.last().copied().unwrap_or(0);
                let (probs, hidden) = match last {
                    2 => (vec![0.25_f32, 0.25, 0.25, 0.25], vec![0.0_f32, 1.0]),
                    _ => (vec![0.05_f32, 0.50, 0.40, 0.05], vec![1.0_f32, 0.0]),
                };
                Ok((tensor_1d(&log_of(&probs)), tensor_1d(&hidden)))
            })
            .expect("contrastive with hidden states");

        assert_eq!(sequences[0], vec![0, 2]);
    }

    // ------------------------------------------------------------------
    // 7. Stopping criteria and logit processors
    // ------------------------------------------------------------------

    #[test]
    fn test_eos_token_stops_generation() {
        let mut cfg = config(GenerationStrategy::Greedy, 10);
        cfg.eos_token_id = Some(3);
        let sequences = TextGenerator::new(cfg, 4)
            .generate(&[0], constant_model(vec![0.0, 0.1, 0.2, 5.0]))
            .expect("generate");
        assert_eq!(sequences[0], vec![0, 3]);
    }

    #[test]
    fn test_min_length_suppresses_eos() {
        let mut cfg = config(GenerationStrategy::Greedy, 10);
        cfg.eos_token_id = Some(3);
        cfg.min_length = Some(4);
        let sequences = TextGenerator::new(cfg, 4)
            .generate(&[0], constant_model(vec![0.0, 0.0, 0.0, 5.0]))
            .expect("generate");
        assert_eq!(
            sequences[0],
            vec![0, 0, 0, 0, 3],
            "EOS must be blocked until the sequence reaches min_length"
        );
    }

    #[test]
    fn test_repetition_penalty_changes_the_decoded_token() {
        let logits = vec![0.0_f32, 3.0, 2.9, 0.0];

        let plain = TextGenerator::new(config(GenerationStrategy::Greedy, 1), 4)
            .generate(&[1], constant_model(logits.clone()))
            .expect("plain");
        assert_eq!(plain[0], vec![1, 1]);

        let mut cfg = config(GenerationStrategy::Greedy, 1);
        cfg.repetition_penalty = 2.0;
        let penalised = TextGenerator::new(cfg, 4)
            .generate(&[1], constant_model(logits))
            .expect("penalised");
        assert_eq!(penalised[0], vec![1, 2], "token 1 must be demoted");
    }

    #[test]
    fn test_no_repeat_ngram_blocks_previous_tokens() {
        let mut cfg = config(GenerationStrategy::Greedy, 3);
        cfg.no_repeat_ngram_size = Some(1);
        let sequences = TextGenerator::new(cfg, 4)
            .generate(&[0], constant_model(vec![0.9, 0.5, 0.4, 0.3]))
            .expect("generate");
        assert_eq!(sequences[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_exhausted_vocabulary_errors_instead_of_emitting_a_blocked_token() {
        // Regression: greedy decoding used to argmax over an all `-inf` vector
        // and return token 0 - a token the n-gram filter had just forbidden -
        // while every sampling strategy errored on the same input.
        let mut cfg = config(GenerationStrategy::Greedy, 4);
        cfg.no_repeat_ngram_size = Some(1);
        let error = TextGenerator::new(cfg, 4)
            .generate(&[0], constant_model(vec![0.9, 0.5, 0.4, 0.3]))
            .expect_err("the 4th step has no legal token left");
        assert!(
            error.to_string().contains("masked"),
            "unexpected error: {error}"
        );
    }

    // ------------------------------------------------------------------
    // 7b. Length configuration through `max_length` (the crate default)
    // ------------------------------------------------------------------

    #[test]
    fn test_max_length_bounds_the_total_sequence() {
        // `GenerationConfig::default()` uses `max_length` with `max_new_tokens`
        // unset, which is the other branch of `get_max_length`.
        let cfg = GenerationConfig {
            strategy: GenerationStrategy::Greedy,
            max_length: Some(5),
            max_new_tokens: None,
            ..Default::default()
        };
        let generator = TextGenerator::new(cfg, 4);
        assert_eq!(generator.get_max_length(2), 5);
        assert_eq!(generator.max_new_tokens(2), 3);

        let sequences = generator
            .generate(&[0, 1], constant_model(vec![0.1, 0.2, 0.9, 0.3]))
            .expect("generate");
        assert_eq!(
            sequences[0],
            vec![0, 1, 2, 2, 2],
            "max_length counts prompt + generated"
        );
    }

    #[test]
    fn test_max_length_shorter_than_prompt_generates_nothing() {
        let cfg = GenerationConfig {
            strategy: GenerationStrategy::Greedy,
            max_length: Some(2),
            max_new_tokens: None,
            ..Default::default()
        };
        let sequences = TextGenerator::new(cfg, 4)
            .generate(&[0, 1, 2], constant_model(vec![0.1, 0.2, 0.9, 0.3]))
            .expect("generate");
        assert_eq!(sequences[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_default_min_length_allows_immediate_eos() {
        // The crate default is `min_length: Some(1)`, so a one-token prompt has
        // already met the minimum and EOS must not be suppressed.
        let cfg = GenerationConfig {
            strategy: GenerationStrategy::Greedy,
            max_length: Some(10),
            max_new_tokens: None,
            eos_token_id: Some(3),
            ..Default::default()
        };
        assert_eq!(cfg.min_length, Some(1));
        let sequences = TextGenerator::new(cfg, 4)
            .generate(&[0], constant_model(vec![0.0, 0.1, 0.2, 5.0]))
            .expect("generate");
        assert_eq!(sequences[0], vec![0, 3]);
    }

    // ------------------------------------------------------------------
    // 7c. min_length across the beam-search unit conversion
    // ------------------------------------------------------------------

    #[test]
    fn test_beam_search_maps_total_min_length_onto_generated_tokens() {
        // `GenerationConfig::min_length` counts the whole sequence while
        // `BeamSearchConfig::min_length` counts generated tokens; this checks
        // the `saturating_sub(prompt_len)` bridge between the two.
        let mut cfg = config(GenerationStrategy::BeamSearch { num_beams: 2 }, 5);
        cfg.eos_token_id = Some(3);
        cfg.min_length = Some(4); // prompt is 2 tokens -> at least 2 generated

        // EOS (token 3) is the runaway argmax at every step.
        let sequences = TextGenerator::new(cfg, 4)
            .generate(&[0, 1], constant_model(log_of(&[0.01, 0.01, 0.01, 0.97])))
            .expect("beam");

        let best = &sequences[0];
        assert_eq!(best[0..2], [0, 1], "the prompt is preserved");
        let generated = &best[2..];
        assert_eq!(
            generated.len(),
            3,
            "EOS must be blocked for exactly two steps, then win: {best:?}"
        );
        assert_eq!(generated[2], 3, "the third generated token is EOS");
        assert!(
            !generated[..2].contains(&3),
            "EOS escaped the minimum length: {best:?}"
        );
    }

    #[test]
    fn test_beam_search_min_length_zero_lets_eos_win_immediately() {
        let mut cfg = config(GenerationStrategy::BeamSearch { num_beams: 2 }, 5);
        cfg.eos_token_id = Some(3);
        cfg.min_length = None;

        let sequences = TextGenerator::new(cfg, 4)
            .generate(&[0, 1], constant_model(log_of(&[0.01, 0.01, 0.01, 0.97])))
            .expect("beam");
        assert_eq!(sequences[0], vec![0, 1, 3]);
    }

    // ------------------------------------------------------------------
    // 8. Plumbing
    // ------------------------------------------------------------------

    #[test]
    fn test_next_token_logits_takes_the_last_position() {
        let generator = TextGenerator::new(config(GenerationStrategy::Greedy, 1), 3);
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 9.0, 8.0, 7.0], &[2, 3]).expect("tensor");
        let logits = generator.next_token_logits(&tensor).expect("logits");
        assert_eq!(logits, vec![9.0, 8.0, 7.0]);
    }

    #[test]
    fn test_next_token_logits_rejects_ragged_buffers() {
        let generator = TextGenerator::new(config(GenerationStrategy::Greedy, 1), 4);
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor");
        assert!(generator.next_token_logits(&tensor).is_err());
    }

    #[test]
    fn test_num_return_sequences_for_sampling() {
        let mut cfg = config(GenerationStrategy::Sampling { temperature: 1.0 }, 6);
        cfg.num_return_sequences = 3;
        let sequences = TextGenerator::new(cfg, 4)
            .with_seed(31337)
            .generate(&[0], constant_model(vec![0.0, 0.0, 0.0, 0.0]))
            .expect("generate");
        assert_eq!(sequences.len(), 3);
        assert!(
            sequences[0] != sequences[1] || sequences[1] != sequences[2],
            "independent draws should not all coincide: {sequences:?}"
        );
    }

    #[test]
    fn test_num_return_sequences_rejected_for_greedy() {
        let mut cfg = config(GenerationStrategy::Greedy, 4);
        cfg.num_return_sequences = 3;
        assert!(TextGenerator::new(cfg, 4)
            .generate(&[0], constant_model(vec![0.0, 1.0, 0.0, 0.0]))
            .is_err());
    }

    // ------------------------------------------------------------------
    // 8b. Configuration fields this decoder cannot honour
    // ------------------------------------------------------------------

    #[test]
    fn test_watermarking_request_is_refused_instead_of_ignored() {
        // Regression: `watermarking` was accepted and then silently dropped, so
        // callers received plain, unwatermarked text that looked successful.
        use crate::generation::config::{WatermarkingAlgorithm, WatermarkingConfig};

        let mut cfg = config(GenerationStrategy::Greedy, 2);
        cfg.watermarking = Some(WatermarkingConfig {
            algorithm: WatermarkingAlgorithm::GreenList,
            ..Default::default()
        });
        let error = TextGenerator::new(cfg, 4)
            .generate(&[0], constant_model(vec![0.0, 1.0, 0.0, 0.0]))
            .expect_err("watermarking cannot be honoured");
        assert!(
            error.to_string().contains("watermarking"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_text_constraints_are_refused_instead_of_ignored() {
        // Regression: a regex / grammar / schema / choice-list constraint was
        // accepted and then never consulted, so guided generation produced
        // completely unconstrained text.
        use crate::generation::config::GuidedGenerationConfig;

        for guided in [
            GuidedGenerationConfig {
                regex_pattern: Some(r"\d+".to_string()),
                ..Default::default()
            },
            GuidedGenerationConfig {
                grammar: Some("start ::= 'a'".to_string()),
                ..Default::default()
            },
            GuidedGenerationConfig {
                json_schema: Some("{}".to_string()),
                ..Default::default()
            },
            GuidedGenerationConfig {
                choice_list: Some(vec!["yes".to_string()]),
                ..Default::default()
            },
        ] {
            let mut cfg = config(GenerationStrategy::Greedy, 2);
            cfg.guided_generation = Some(guided);
            let error = TextGenerator::new(cfg, 4)
                .generate(&[0], constant_model(vec![0.0, 1.0, 0.0, 0.0]))
                .expect_err("text constraints cannot be honoured here");
            assert!(
                error.to_string().contains("guided_generation"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn test_guided_generation_carrying_only_cfg_still_generates() {
        // The classifier-free guidance sub-config *is* honoured, by
        // `CFGGenerator`; it must not block plain decoding.
        use crate::generation::config::{CFGConfig, GuidedGenerationConfig};

        let mut cfg = config(GenerationStrategy::Greedy, 2);
        cfg.guided_generation = Some(GuidedGenerationConfig {
            regex_pattern: None,
            grammar: None,
            json_schema: None,
            choice_list: None,
            cfg: Some(CFGConfig::default()),
            ..Default::default()
        });
        let sequences = TextGenerator::new(cfg, 4)
            .generate(&[0], constant_model(vec![0.0, 1.0, 0.0, 0.0]))
            .expect("generate");
        assert_eq!(sequences[0], vec![0, 1, 1]);
    }

    #[test]
    fn test_empty_prompt_is_rejected() {
        let generator = TextGenerator::new(config(GenerationStrategy::Greedy, 4), 4);
        assert!(generator.generate(&[], constant_model(vec![0.0, 1.0, 0.0, 0.0])).is_err());
    }

    #[test]
    fn test_should_stop_honours_the_length_budget() {
        let generator = TextGenerator::new(config(GenerationStrategy::Greedy, 2), 4);
        assert!(!generator.should_stop(&[0, 1], 1, 1));
        assert!(generator.should_stop(&[0, 1, 2], 2, 2));
    }
}
