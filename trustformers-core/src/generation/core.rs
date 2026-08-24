//! Strategy-aware autoregressive decoding.
//!
//! [`TextGenerator`] implements every strategy declared by
//! [`GenerationStrategy`] for real: greedy argmax, temperature sampling,
//! top-k truncation, nucleus (top-p) truncation, beam search with per-beam
//! log-probability accumulation and length penalty, and contrastive search
//! with a degeneration penalty.  No strategy silently degrades into another
//! one; anything that cannot be honoured returns a structured error.

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use scirs2_core::random::{RngExt, SeedableRng, StdRng};

use super::beam_search::{BeamError, BeamSearchConfig, BeamSearchDecoder};
use super::cache::KVCache;
use super::config::{GenerationConfig, GenerationStrategy};
use super::logits_processing as lp;

/// Core text generator with various generation strategies
pub struct TextGenerator {
    pub config: GenerationConfig,
    pub vocab_size: usize,
    seed: Option<u64>,
}

impl TextGenerator {
    pub fn new(config: GenerationConfig, vocab_size: usize) -> Self {
        Self {
            config,
            vocab_size,
            seed: None,
        }
    }

    /// Pin the sampling RNG to a fixed seed, making every stochastic strategy
    /// reproducible.  Without a seed the generator draws its seed from the
    /// thread RNG on each call to [`TextGenerator::generate`].
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// The configured RNG seed, if any.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Build a fresh RNG, seeded from [`TextGenerator::with_seed`] when set and
    /// from thread entropy otherwise.
    pub fn new_rng(&self) -> StdRng {
        match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut entropy = scirs2_core::random::rng();
                let seed: u64 = entropy.random();
                StdRng::seed_from_u64(seed)
            },
        }
    }

    /// Generate text using the configured strategy.
    ///
    /// The returned sequences always contain the prompt followed by the newly
    /// decoded tokens, for every strategy.
    pub fn generate(
        &self,
        input_ids: &[usize],
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        if input_ids.is_empty() {
            return Err(TrustformersError::invalid_input(
                "generation requires at least one input token".to_string(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(TrustformersError::invalid_input(
                "generation requires a non-zero vocabulary size".to_string(),
            ));
        }
        self.reject_unhonoured_config()?;

        match self.config.strategy {
            GenerationStrategy::Greedy => self.generate_greedy(input_ids, logits_fn),
            GenerationStrategy::BeamSearch { num_beams } => {
                self.generate_beam_search(input_ids, num_beams, logits_fn)
            },
            GenerationStrategy::Sampling { temperature } => {
                self.generate_sampling(input_ids, temperature, logits_fn)
            },
            GenerationStrategy::TopK { k, temperature } => {
                self.generate_top_k(input_ids, k, temperature, logits_fn)
            },
            GenerationStrategy::TopP { p, temperature } => {
                self.generate_top_p(input_ids, p, temperature, logits_fn)
            },
            GenerationStrategy::ContrastiveSearch {
                penalty_alpha,
                top_k,
            } => self.generate_contrastive_search(input_ids, penalty_alpha, top_k, logits_fn),
        }
    }

    /// Refuse configurations whose fields this decoder cannot honour, instead
    /// of returning text that silently ignores them.
    ///
    /// Two [`GenerationConfig`] fields are outside what the `logits_fn`
    /// interface can support:
    ///
    /// * `watermarking`: no watermarking algorithm is implemented in this
    ///   crate, so a request for one cannot be met.
    /// * `guided_generation` carrying a regex, grammar, JSON schema or choice
    ///   list: [`super::constraints::ConstraintValidator`] decides over *text*,
    ///   while `logits_fn` only ever hands this decoder token ids, so without a
    ///   detokenizer the constraint cannot be evaluated.  Drive the validator
    ///   directly, or use [`super::cfg::CFGGenerator`] for the classifier-free
    ///   guidance sub-config, which this decoder *can* honour.
    fn reject_unhonoured_config(&self) -> Result<()> {
        if self.config.watermarking.is_some() {
            return Err(TrustformersError::not_implemented(
                "generation_config.watermarking was set but no watermarking algorithm is \
                 implemented; unset it rather than receiving unwatermarked text"
                    .to_string(),
            ));
        }

        if let Some(guided) = self.config.guided_generation.as_ref() {
            let unsupported = [
                ("regex_pattern", guided.regex_pattern.is_some()),
                ("grammar", guided.grammar.is_some()),
                ("json_schema", guided.json_schema.is_some()),
                ("choice_list", guided.choice_list.is_some()),
            ];
            let requested: Vec<&str> = unsupported
                .iter()
                .filter(|(_, is_set)| *is_set)
                .map(|(name, _)| *name)
                .collect();
            if !requested.is_empty() {
                return Err(TrustformersError::not_implemented(format!(
                    "guided_generation.{} cannot be enforced by TextGenerator: constraints are \
                     evaluated on decoded text and the logits callback only supplies token ids, \
                     so no detokenizer is available; drive ConstraintValidator yourself, or use \
                     CFGGenerator for guided_generation.cfg",
                    requested.join(" / ")
                )));
            }
        }

        Ok(())
    }

    /// Maximum *total* sequence length (prompt + generated).
    pub fn get_max_length(&self, input_length: usize) -> usize {
        if let Some(max_new_tokens) = self.config.max_new_tokens {
            input_length + max_new_tokens
        } else if let Some(max_length) = self.config.max_length {
            max_length
        } else {
            input_length + 100 // Default fallback
        }
    }

    /// Number of tokens that may still be generated for a prompt of
    /// `input_length` tokens.
    pub fn max_new_tokens(&self, input_length: usize) -> usize {
        self.get_max_length(input_length).saturating_sub(input_length)
    }

    /// Check if generation should stop.
    ///
    /// `new_tokens_generated` counts the tokens produced by the decoder so far
    /// (i.e. `sequence.len() - prompt.len()`).  Generation stops when the total
    /// sequence reaches [`TextGenerator::get_max_length`], or when the last
    /// token is the EOS token *and* the configured minimum total length has
    /// been reached.
    pub fn should_stop(
        &self,
        sequence: &[usize],
        last_token: usize,
        new_tokens_generated: usize,
    ) -> bool {
        let prompt_length = sequence.len().saturating_sub(new_tokens_generated);

        if sequence.len() >= self.get_max_length(prompt_length) {
            return true;
        }

        if Some(last_token) == self.config.eos_token_id
            && sequence.len() >= self.config.min_length.unwrap_or(0)
        {
            return true;
        }

        false
    }

    /// Greedy (argmax) selection from a logits tensor.
    ///
    /// This deliberately ignores the configured sampling strategy; use
    /// [`TextGenerator::generate`] for strategy-aware decoding, or
    /// [`TextGenerator::select_next_token`] to apply the configured strategy to
    /// a single logits tensor.
    pub fn argmax_token(&self, logits: &Tensor) -> Result<usize> {
        let data = self.next_token_logits(logits)?;
        lp::argmax(&data)
    }

    /// Apply the configured strategy to a single logits tensor.
    ///
    /// `sequence` is the full token history (prompt + generated) and is used by
    /// the repetition penalty, n-gram blocking and minimum-length constraints.
    /// Beam search cannot be expressed as a single-token decision and therefore
    /// returns a `NotImplemented` error here — call
    /// [`TextGenerator::generate`] instead.
    pub fn select_next_token(
        &self,
        logits: &Tensor,
        rng: &mut StdRng,
        sequence: &[usize],
    ) -> Result<usize> {
        let mut data = self.next_token_logits(logits)?;
        self.apply_logit_processors(&mut data, sequence);

        match self.config.strategy {
            GenerationStrategy::Greedy => lp::argmax(&data),
            GenerationStrategy::Sampling { temperature } => {
                lp::apply_temperature(&mut data, temperature)?;
                let probs = lp::softmax(&data)?;
                lp::multinomial_sample(&probs, rng)
            },
            GenerationStrategy::TopK { k, temperature } => {
                lp::apply_temperature(&mut data, temperature)?;
                lp::top_k_filter(&mut data, k)?;
                let probs = lp::softmax(&data)?;
                lp::multinomial_sample(&probs, rng)
            },
            GenerationStrategy::TopP { p, temperature } => {
                lp::apply_temperature(&mut data, temperature)?;
                lp::top_p_filter(&mut data, p)?;
                let probs = lp::softmax(&data)?;
                lp::multinomial_sample(&probs, rng)
            },
            GenerationStrategy::ContrastiveSearch { .. }
            | GenerationStrategy::BeamSearch { .. } => Err(TrustformersError::not_implemented(
                "single-step token selection is undefined for beam search and contrastive \
                 search; both need multi-step lookahead - use TextGenerator::generate"
                    .to_string(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Logits plumbing
    // -----------------------------------------------------------------------

    /// Flatten a tensor to `f32` in logical row-major order.
    fn tensor_to_f32(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        match tensor {
            Tensor::F32(arr) => Ok(arr.iter().copied().collect()),
            Tensor::F64(arr) => Ok(arr.iter().map(|&x| x as f32).collect()),
            Tensor::F16(arr) => Ok(arr.iter().map(|&x| x.to_f32()).collect()),
            Tensor::BF16(arr) => Ok(arr.iter().map(|&x| x.to_f32()).collect()),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Tensor::Metal(metal_data) => {
                use crate::gpu_ops::metal::get_metal_backend;
                let backend = get_metal_backend()?;
                backend.download_buffer_to_vec(&metal_data.buffer_id())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "unsupported tensor dtype for generation logits",
                "tensor_to_f32",
            )),
        }
    }

    /// Extract the logits of the *last* position.
    ///
    /// Accepts `[vocab]`, `[seq, vocab]` and `[batch, seq, vocab]` layouts: the
    /// element order of an `ArrayD` is row-major, so the final `vocab_size`
    /// entries are always the newest position.  A length that is not a multiple
    /// of `vocab_size` is rejected rather than silently mis-read.
    pub fn next_token_logits(&self, logits: &Tensor) -> Result<Vec<f32>> {
        let data = self.tensor_to_f32(logits)?;
        last_chunk(&data, self.vocab_size).map_err(|_| {
            TrustformersError::tensor_op_error(
                &format!(
                    "logits tensor of {} elements is not a multiple of vocab_size {}",
                    data.len(),
                    self.vocab_size
                ),
                "next_token_logits",
            )
        })
    }

    /// Apply repetition penalty, n-gram blocking and minimum-length EOS
    /// suppression to a next-token logits vector, in place.
    fn apply_logit_processors(&self, logits: &mut [f32], sequence: &[usize]) {
        lp::apply_repetition_penalty(logits, sequence, self.config.repetition_penalty);

        if let Some(ngram_size) = self.config.no_repeat_ngram_size {
            for token in lp::forbidden_ngram_tokens(sequence, ngram_size) {
                if let Some(slot) = logits.get_mut(token) {
                    *slot = f32::NEG_INFINITY;
                }
            }
        }

        // Suppress EOS until the configured minimum total length is reached, so
        // that a short sequence cannot terminate early.
        let below_min_length = sequence.len() < self.config.min_length.unwrap_or(0);
        let eos_slot = self
            .config
            .eos_token_id
            .filter(|_| below_min_length)
            .and_then(|eos| logits.get_mut(eos));
        if let Some(slot) = eos_slot {
            *slot = f32::NEG_INFINITY;
        }
    }

    // -----------------------------------------------------------------------
    // Sequential decoding driver
    // -----------------------------------------------------------------------

    /// Run one autoregressive rollout, delegating the per-step token decision
    /// to `select`.
    fn run_sequential(
        &self,
        input_ids: &[usize],
        logits_fn: &impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
        select: &mut dyn FnMut(&mut [f32]) -> Result<usize>,
    ) -> Result<Vec<usize>> {
        let mut sequence = input_ids.to_vec();
        let mut cache = if self.config.use_cache { Some(KVCache::new()) } else { None };

        for _ in 0..self.max_new_tokens(input_ids.len()) {
            let (logits_tensor, new_cache) = logits_fn(&sequence, cache.as_ref())?;
            cache = new_cache;

            let mut logits = self.next_token_logits(&logits_tensor)?;
            self.apply_logit_processors(&mut logits, &sequence);

            let next_token = select(&mut logits)?;
            if next_token >= self.vocab_size {
                return Err(TrustformersError::invalid_input(format!(
                    "selected token {next_token} is outside the vocabulary of size {}",
                    self.vocab_size
                )));
            }
            sequence.push(next_token);

            let generated = sequence.len() - input_ids.len();
            if self.should_stop(&sequence, next_token, generated) {
                break;
            }
        }

        Ok(sequence)
    }

    /// Number of sequences the caller asked for (at least one).
    fn num_return_sequences(&self) -> usize {
        self.config.num_return_sequences.max(1)
    }

    /// Reject `num_return_sequences > 1` for deterministic strategies, which
    /// would otherwise return N identical copies and pretend to be diverse.
    fn reject_multi_return(&self, strategy: &str) -> Result<()> {
        if self.num_return_sequences() > 1 {
            return Err(TrustformersError::invalid_input(format!(
                "{strategy} is deterministic and can only return one sequence; \
                 num_return_sequences = {} was requested",
                self.config.num_return_sequences
            )));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Strategies
    // -----------------------------------------------------------------------

    /// Greedy decoding: always take the highest-scoring token.
    fn generate_greedy(
        &self,
        input_ids: &[usize],
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        self.reject_multi_return("greedy decoding")?;
        let mut select = |logits: &mut [f32]| lp::argmax(logits);
        Ok(vec![self.run_sequential(
            input_ids,
            &logits_fn,
            &mut select,
        )?])
    }

    /// Ancestral sampling from the temperature-scaled softmax.
    fn generate_sampling(
        &self,
        input_ids: &[usize],
        temperature: f32,
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        let mut rng = self.new_rng();
        let mut select = |logits: &mut [f32]| {
            lp::apply_temperature(logits, temperature)?;
            let probs = lp::softmax(logits)?;
            lp::multinomial_sample(&probs, &mut rng)
        };
        self.run_repeated(input_ids, &logits_fn, &mut select)
    }

    /// Top-k sampling: renormalise over the `k` most probable tokens.
    fn generate_top_k(
        &self,
        input_ids: &[usize],
        k: usize,
        temperature: f32,
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        if k == 0 {
            return Err(TrustformersError::invalid_input(
                "top-k sampling requires k >= 1".to_string(),
            ));
        }
        let mut rng = self.new_rng();
        let mut select = |logits: &mut [f32]| {
            lp::apply_temperature(logits, temperature)?;
            lp::top_k_filter(logits, k)?;
            let probs = lp::softmax(logits)?;
            lp::multinomial_sample(&probs, &mut rng)
        };
        self.run_repeated(input_ids, &logits_fn, &mut select)
    }

    /// Nucleus sampling: renormalise over the smallest set of tokens whose
    /// cumulative probability reaches `p`.
    fn generate_top_p(
        &self,
        input_ids: &[usize],
        p: f32,
        temperature: f32,
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        let mut rng = self.new_rng();
        let mut select = |logits: &mut [f32]| {
            lp::apply_temperature(logits, temperature)?;
            lp::top_p_filter(logits, p)?;
            let probs = lp::softmax(logits)?;
            lp::multinomial_sample(&probs, &mut rng)
        };
        self.run_repeated(input_ids, &logits_fn, &mut select)
    }

    /// Run `num_return_sequences` independent stochastic rollouts.
    fn run_repeated(
        &self,
        input_ids: &[usize],
        logits_fn: &impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
        select: &mut dyn FnMut(&mut [f32]) -> Result<usize>,
    ) -> Result<Vec<Vec<usize>>> {
        let mut sequences = Vec::with_capacity(self.num_return_sequences());
        for _ in 0..self.num_return_sequences() {
            sequences.push(self.run_sequential(input_ids, logits_fn, &mut *select)?);
        }
        Ok(sequences)
    }

    /// Beam search with per-beam cumulative log-probabilities, length penalty,
    /// n-gram blocking and EOS handling.
    ///
    /// Beam decoding re-runs the model once per active beam per step and
    /// therefore cannot reuse a per-beam key/value cache through the
    /// single-sequence `logits_fn` interface: `config.use_cache` has no effect
    /// on this path and `None` is passed as the cache on every call.
    ///
    /// Exactly `num_return_sequences` sequences are returned, ranked best
    /// first; if the search cannot produce that many hypotheses the call fails
    /// rather than silently returning a shorter list than the sampling paths
    /// would for the same configuration field.
    fn generate_beam_search(
        &self,
        input_ids: &[usize],
        num_beams: usize,
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        if num_beams == 0 {
            return Err(TrustformersError::invalid_input(
                "beam search requires num_beams >= 1".to_string(),
            ));
        }
        if self.num_return_sequences() > num_beams {
            return Err(TrustformersError::invalid_input(format!(
                "num_return_sequences ({}) cannot exceed num_beams ({num_beams})",
                self.config.num_return_sequences
            )));
        }
        if self.vocab_size > u32::MAX as usize {
            return Err(TrustformersError::invalid_input(
                "beam search supports vocabularies up to u32::MAX entries".to_string(),
            ));
        }
        let prompt: Vec<u32> = input_ids
            .iter()
            .map(|&token| {
                u32::try_from(token).map_err(|_| {
                    TrustformersError::invalid_input(format!(
                        "token id {token} does not fit in u32"
                    ))
                })
            })
            .collect::<Result<Vec<u32>>>()?;

        let eos_token_id = optional_token_as_u32(self.config.eos_token_id, "eos_token_id")?;
        let pad_token_id = optional_token_as_u32(self.config.pad_token_id, "pad_token_id")?;

        let beam_config = BeamSearchConfig {
            num_beams,
            max_new_tokens: self.max_new_tokens(input_ids.len()),
            // `BeamSearchConfig::min_length` counts generated tokens, while
            // `GenerationConfig::min_length` counts the total sequence.
            min_length: self.config.min_length.unwrap_or(0).saturating_sub(input_ids.len()),
            length_penalty: self.config.length_penalty,
            early_stopping: self.config.early_stopping,
            no_repeat_ngram_size: self.config.no_repeat_ngram_size.unwrap_or(0),
            repetition_penalty: self.config.repetition_penalty,
            diversity_penalty: 0.0,
            num_beam_groups: 1,
            eos_token_id,
            pad_token_id,
            vocab_size: self.vocab_size,
        };

        let decoder = BeamSearchDecoder::new(beam_config).map_err(beam_error)?;
        let hypotheses = decoder.decode_top_n(&prompt, self.num_return_sequences(), |sequences| {
            let mut scores = Vec::with_capacity(sequences.len());
            for sequence in sequences {
                let tokens: Vec<usize> = sequence.iter().map(|&token| token as usize).collect();
                let (tensor, _) = logits_fn(&tokens, None)
                    .map_err(|e| BeamError::ScoreFunctionError(e.to_string()))?;
                let logits = self
                    .next_token_logits(&tensor)
                    .map_err(|e| BeamError::ScoreFunctionError(e.to_string()))?;
                let log_probs = lp::log_softmax(&logits)
                    .map_err(|e| BeamError::ScoreFunctionError(e.to_string()))?;
                scores.push(log_probs);
            }
            Ok(scores)
        });

        let hypotheses = hypotheses.map_err(beam_error)?;
        if hypotheses.len() < self.num_return_sequences() {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "beam search produced only {} hypotheses but {} sequences were requested",
                    hypotheses.len(),
                    self.num_return_sequences()
                ),
                "beam_search",
            ));
        }
        Ok(hypotheses
            .into_iter()
            .map(|hypothesis| {
                let mut sequence = input_ids.to_vec();
                sequence.extend(hypothesis.tokens.iter().map(|&token| token as usize));
                sequence
            })
            .collect())
    }

    /// Contrastive search (Su et al., 2022, "A Contrastive Framework for
    /// Neural Text Generation").
    ///
    /// At every step the `top_k` most probable continuations are scored with
    ///
    /// ```text
    /// score(v) = (1 - penalty_alpha) * p(v | prefix)
    ///          - penalty_alpha * max_j cos(rep(prefix + v), rep(prefix_j))
    /// ```
    ///
    /// and the highest-scoring candidate is emitted, so a token that steers the
    /// model back into an already-visited state is demoted even when it is the
    /// argmax.
    ///
    /// **Representation used.** The reference formulation compares the *hidden
    /// states* of the candidate and of every previous position.  The
    /// `logits_fn` interface of this generator exposes only the model's output
    /// layer, so this method uses the next-token probability vector
    /// `softmax(f(prefix))` as the representation of the state reached after a
    /// prefix.  That vector is a deterministic function of the final hidden
    /// state through the (frozen) language-model head, so it is a genuine -
    /// though not identical - state descriptor: no similarity value is
    /// invented.  Call
    /// [`TextGenerator::generate_contrastive_search_with_hidden_states`] to run
    /// the exact hidden-state formulation when the model can expose them.
    ///
    /// Cost: `1 + top_k` forward passes per generated token.
    fn generate_contrastive_search(
        &self,
        input_ids: &[usize],
        penalty_alpha: f32,
        top_k: usize,
        logits_fn: impl Fn(&[usize], Option<&KVCache>) -> Result<(Tensor, Option<KVCache>)>,
    ) -> Result<Vec<Vec<usize>>> {
        self.reject_multi_return("contrastive search")?;
        self.contrastive_search_impl(input_ids, penalty_alpha, top_k, |tokens| {
            let (tensor, _) = logits_fn(tokens, None)?;
            let logits = self.next_token_logits(&tensor)?;
            let probs = lp::softmax(&logits)?;
            Ok((logits, probs))
        })
        .map(|sequence| vec![sequence])
    }

    /// Exact contrastive search using the model's hidden states.
    ///
    /// `model_fn` must return `(logits, hidden_state)` for the *last* position
    /// of the supplied token sequence.  The hidden state may have any shape;
    /// its trailing dimension is taken as the representation vector.
    pub fn generate_contrastive_search_with_hidden_states(
        &self,
        input_ids: &[usize],
        penalty_alpha: f32,
        top_k: usize,
        model_fn: impl Fn(&[usize]) -> Result<(Tensor, Tensor)>,
    ) -> Result<Vec<Vec<usize>>> {
        self.reject_multi_return("contrastive search")?;
        self.contrastive_search_impl(input_ids, penalty_alpha, top_k, |tokens| {
            let (logits_tensor, hidden_tensor) = model_fn(tokens)?;
            let logits = self.next_token_logits(&logits_tensor)?;
            let hidden_width = *hidden_tensor.shape().last().ok_or_else(|| {
                TrustformersError::invalid_input(
                    "hidden state tensor has no dimensions".to_string(),
                )
            })?;
            let hidden = self.tensor_to_f32(&hidden_tensor)?;
            let representation = last_chunk(&hidden, hidden_width)?;
            Ok((logits, representation))
        })
        .map(|sequence| vec![sequence])
    }

    /// Shared contrastive-search driver.
    ///
    /// `evaluate` maps a token prefix to `(next_token_logits, representation)`.
    fn contrastive_search_impl(
        &self,
        input_ids: &[usize],
        penalty_alpha: f32,
        top_k: usize,
        evaluate: impl Fn(&[usize]) -> Result<(Vec<f32>, Vec<f32>)>,
    ) -> Result<Vec<usize>> {
        if top_k == 0 {
            return Err(TrustformersError::invalid_input(
                "contrastive search requires top_k >= 1".to_string(),
            ));
        }
        if !penalty_alpha.is_finite() || !(0.0..=1.0).contains(&penalty_alpha) {
            return Err(TrustformersError::invalid_input(format!(
                "contrastive search requires penalty_alpha in [0, 1], got {penalty_alpha}"
            )));
        }

        let mut sequence = input_ids.to_vec();
        let (initial_logits, initial_representation) = evaluate(&sequence)?;
        let mut pending_logits = initial_logits;
        let mut context_representations = vec![initial_representation];

        for _ in 0..self.max_new_tokens(input_ids.len()) {
            let mut logits = pending_logits;
            self.apply_logit_processors(&mut logits, &sequence);
            let probs = lp::softmax(&logits)?;

            let mut masked = logits.clone();
            lp::top_k_filter(&mut masked, top_k.min(self.vocab_size))?;
            let candidates: Vec<usize> = masked
                .iter()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .map(|(index, _)| index)
                .collect();
            if candidates.is_empty() {
                return Err(TrustformersError::tensor_op_error(
                    "contrastive search found no viable candidate tokens",
                    "contrastive_search",
                ));
            }

            let mut best: Option<(f32, usize, Vec<f32>, Vec<f32>)> = None;
            for &candidate in &candidates {
                let mut lookahead = sequence.clone();
                lookahead.push(candidate);
                let (candidate_logits, representation) = evaluate(&lookahead)?;

                let similarity = context_representations
                    .iter()
                    .map(|previous| cosine_similarity(&representation, previous))
                    .fold(f32::NEG_INFINITY, f32::max);

                let confidence = probs.get(candidate).copied().unwrap_or(0.0);
                let score = (1.0 - penalty_alpha) * confidence - penalty_alpha * similarity;

                // `candidates` is ascending, so a strict `>` keeps the lowest
                // token id on ties - matching the greedy tie-break rule.
                let is_better = match &best {
                    None => true,
                    Some((best_score, _, _, _)) => score > *best_score,
                };
                if is_better {
                    best = Some((score, candidate, representation, candidate_logits));
                }
            }

            let (_, next_token, representation, next_logits) = best.ok_or_else(|| {
                TrustformersError::tensor_op_error(
                    "contrastive search failed to score any candidate",
                    "contrastive_search",
                )
            })?;

            sequence.push(next_token);
            context_representations.push(representation);
            pending_logits = next_logits;

            let generated = sequence.len() - input_ids.len();
            if self.should_stop(&sequence, next_token, generated) {
                break;
            }
        }

        Ok(sequence)
    }

    /// Apply softmax to logits.
    pub fn softmax(&self, logits: &[f32]) -> Result<Vec<f32>> {
        lp::softmax(logits)
    }

    /// Sample an index from an explicit probability distribution.
    pub fn sample_from_probs(&self, probs: &[f32], rng: &mut StdRng) -> Result<usize> {
        lp::multinomial_sample(probs, rng)
    }
}

/// Take the final `width`-sized chunk of a row-major buffer.
fn last_chunk(data: &[f32], width: usize) -> Result<Vec<f32>> {
    if width == 0 || data.len() < width || !data.len().is_multiple_of(width) {
        return Err(TrustformersError::invalid_input(format!(
            "buffer of {} elements cannot be split into rows of {width}",
            data.len()
        )));
    }
    Ok(data[data.len() - width..].to_vec())
}

/// Cosine similarity between two equally-sized vectors; `0.0` when either side
/// has no magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denominator = norm_a.sqrt() * norm_b.sqrt();
    if denominator <= 0.0 || !denominator.is_finite() {
        return 0.0;
    }
    dot / denominator
}

/// Lift a beam-search error into the crate error type.
fn beam_error(error: BeamError) -> TrustformersError {
    TrustformersError::tensor_op_error(&error.to_string(), "beam_search")
}

/// Narrow an optional `usize` token id to `u32`, with a descriptive error.
fn optional_token_as_u32(token: Option<usize>, field: &str) -> Result<Option<u32>> {
    match token {
        Some(value) => u32::try_from(value).map(Some).map_err(|_| {
            TrustformersError::invalid_input(format!("{field} = {value} does not fit in u32"))
        }),
        None => Ok(None),
    }
}
