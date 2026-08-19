//! Real generation engine behind [`AutoModel`].
//!
//! Split out of `automodel.rs` to keep both files under the workspace's
//! 2000-line limit. Everything here operates on the same `AutoModel` type; as a
//! child module it can reach the parent's private fields.

use super::{AutoModel, AutoModelType};
use crate::core::traits::{Config, Model, TokenizedInput};
use crate::error::{Result, TrustformersError};
use std::borrow::Cow;
use trustformers_core::tensor::Tensor;
use trustformers_models::common_patterns::{GenerationConfig, GenerativeModel};

impl AutoModel {
    /// Absolute sequence length (prompt + newly generated tokens) at which a
    /// decoder-only run must stop.
    ///
    /// Honours `max_new_tokens`, the optional absolute `max_length`, and the
    /// model's own context window, and always leaves room for at least one new
    /// token so a caller can never ask for a zero-step "generation".
    fn resolve_decoder_target_len(&self, prompt_len: usize, config: &GenerationConfig) -> usize {
        let by_new_tokens = prompt_len.saturating_add(config.max_new_tokens.max(1));
        let by_absolute = config.max_length.unwrap_or(usize::MAX);
        by_new_tokens
            .min(by_absolute)
            .min(self.max_context_length())
            .max(prompt_len.saturating_add(1))
    }

    /// Number of decoder tokens a T5 run should produce.
    fn resolve_encoder_decoder_target_len(&self, config: &GenerationConfig) -> usize {
        config
            .max_new_tokens
            .max(1)
            .min(config.max_length.unwrap_or(usize::MAX))
            .min(self.max_context_length())
    }

    /// Encode `prompt` with this checkpoint's real tokenizer.
    ///
    /// # Errors
    ///
    /// Fails when no tokenizer is attached, or when the prompt encodes to an
    /// empty token sequence (there is nothing to condition on).
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let tokenizer = self.require_tokenizer()?;
        let encoded = tokenizer.encode(prompt)?;
        if encoded.input_ids.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "prompt encoded to an empty token sequence; nothing to generate from".to_string(),
            ));
        }
        Ok(encoded.input_ids)
    }

    /// Run a real autoregressive decode and return the token ids the model
    /// produced.
    ///
    /// For decoder-only models (GPT-2 / GPT-Neo / GPT-J) the returned sequence
    /// starts with the prompt tokens; `prompt_len` records how many. For T5 the
    /// sequence contains decoder tokens only and `prompt_len` is `0`.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::FeatureUnavailable`] for architectures that
    /// have no language-modelling head — those models genuinely cannot generate
    /// text and this method will never invent a string for them.
    pub fn generate_token_ids(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GeneratedSequence> {
        // Tokenizer validity is checked unconditionally (and first) so an
        // absent tokenizer is always the reported cause, regardless of which
        // generative-model features this build has compiled in.
        let input_ids = self.encode_prompt(prompt)?;
        let prompt_len = input_ids.len();

        #[cfg(not(any(
            feature = "gpt2",
            feature = "gpt_neo",
            feature = "gpt_j",
            feature = "t5"
        )))]
        {
            Err(Self::no_generative_backend_error(
                self.architecture_name(),
                config,
                format!("{prompt_len}-token prompt"),
            ))
        }

        #[cfg(any(
            feature = "gpt2",
            feature = "gpt_neo",
            feature = "gpt_j",
            feature = "t5"
        ))]
        match &self.model_type {
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2LMHead(model) => {
                let target = self.resolve_decoder_target_len(prompt_len, config);
                let sequence = if config.do_sample {
                    model.generate(
                        input_ids,
                        target,
                        config.temperature,
                        config.top_k,
                        Some(config.top_p),
                    )?
                } else {
                    model.generate_greedy(input_ids, target)?
                };
                Ok(GeneratedSequence::decoder_only(
                    sequence, prompt_len, config,
                ))
            },
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeoLMHead(model) => {
                let target = self.resolve_decoder_target_len(prompt_len, config);
                let sequence = if config.do_sample {
                    model.generate(
                        input_ids,
                        target,
                        config.temperature,
                        config.top_k,
                        Some(config.top_p),
                    )?
                } else {
                    model.generate_greedy(input_ids, target)?
                };
                Ok(GeneratedSequence::decoder_only(
                    sequence, prompt_len, config,
                ))
            },
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJLMHead(model) => {
                let target = self.resolve_decoder_target_len(prompt_len, config);
                let sequence = if config.do_sample {
                    model.generate(
                        input_ids,
                        target,
                        config.temperature,
                        config.top_k,
                        Some(config.top_p),
                    )?
                } else {
                    model.generate_greedy(input_ids, target)?
                };
                Ok(GeneratedSequence::decoder_only(
                    sequence, prompt_len, config,
                ))
            },
            #[cfg(feature = "t5")]
            AutoModelType::T5ForConditionalGeneration(model) => {
                let target = self.resolve_encoder_decoder_target_len(config);
                let num_beams = config.num_beams.unwrap_or(1).max(1);
                let sequence = model.generate(input_ids, target, num_beams)?;
                let finished_by_eos = sequence.len() < target;
                Ok(GeneratedSequence {
                    sequence,
                    prompt_len: 0,
                    finished_by_eos,
                })
            },
            _ => Err(Self::not_generative_error(self.architecture_name())),
        }
    }

    /// Build a real incremental decoder over `prompt`.
    ///
    /// Every [`Iterator::next`] call performs exactly one additional decoding
    /// step and yields the token id plus the newly decoded text.
    ///
    /// # Errors
    ///
    /// Same conditions as [`AutoModel::generate_token_ids`].
    pub fn token_stream<'a>(
        &'a self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<AutoModelTokenStream<'a>> {
        AutoModelTokenStream::new(Cow::Borrowed(self), prompt, config)
    }

    /// Same as [`AutoModel::token_stream`] but the stream owns its model clone,
    /// so it satisfies the `'static` bound of
    /// [`GenerativeModel::generate_stream`].
    ///
    /// # Errors
    ///
    /// Same conditions as [`AutoModel::generate_token_ids`].
    pub fn into_token_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<AutoModelTokenStream<'static>> {
        AutoModelTokenStream::new(Cow::Owned(self.clone()), prompt, config)
    }

    /// Human-readable architecture name, used in error messages.
    fn architecture_name(&self) -> &'static str {
        Config::architecture(&self.config)
    }

    fn not_generative_error(architecture: &str) -> TrustformersError {
        TrustformersError::feature_unavailable(
            format!(
                "model architecture `{architecture}` has no language-modelling head, so it \
                 cannot generate text. Load a *LMHead / *ForConditionalGeneration checkpoint \
                 instead."
            ),
            "text-generation",
        )
    }

    /// Error returned by the generation entry points when this build has
    /// none of the `gpt2`/`gpt_neo`/`gpt_j`/`t5` features compiled in, so no
    /// architecture-specific branch in the dispatch below could ever run —
    /// distinct from [`Self::not_generative_error`], which fires when
    /// generative features *are* compiled in but this particular checkpoint
    /// isn't one of the generative architectures.
    #[cfg(not(any(
        feature = "gpt2",
        feature = "gpt_neo",
        feature = "gpt_j",
        feature = "t5"
    )))]
    fn no_generative_backend_error(
        architecture: &str,
        config: &GenerationConfig,
        context: impl std::fmt::Display,
    ) -> TrustformersError {
        TrustformersError::feature_unavailable(
            format!(
                "no generative model feature (gpt2/gpt_neo/gpt_j/t5) is compiled into this \
                 build, so architecture `{architecture}` cannot generate up to {} new tokens \
                 ({context}). Checkpoints with a language-modelling head (GPT-2/GPT-Neo/GPT-J \
                 *LMHead, T5 ForConditionalGeneration) need the matching feature enabled; other \
                 architectures cannot generate regardless of features.",
                config.max_new_tokens.max(1),
            ),
            "text-generation",
        )
    }

    /// Produce exactly one more token given the tokens decoded so far.
    ///
    /// `prompt_ids` are the encoder/prompt tokens; `decoded` are the tokens the
    /// decoder has already emitted. Returns `None` when the model stopped.
    fn step_once(
        &self,
        prompt_ids: &[u32],
        decoded: &[u32],
        config: &GenerationConfig,
    ) -> Result<Option<u32>> {
        // In practice this is unreachable without a generative feature: the
        // only caller (`AutoModelTokenStream::new`) refuses to construct a
        // stream unless `is_generative()` is true, which is unconditionally
        // false in that configuration. Still handled explicitly (rather than
        // cfg'd away) because `Iterator::next` calls this method
        // unconditionally, so it must exist and type-check in every feature
        // configuration.
        #[cfg(not(any(
            feature = "gpt2",
            feature = "gpt_neo",
            feature = "gpt_j",
            feature = "t5"
        )))]
        {
            Err(Self::no_generative_backend_error(
                self.architecture_name(),
                config,
                format!(
                    "{} prompt + {} decoded tokens",
                    prompt_ids.len(),
                    decoded.len()
                ),
            ))
        }

        #[cfg(any(
            feature = "gpt2",
            feature = "gpt_neo",
            feature = "gpt_j",
            feature = "t5"
        ))]
        match &self.model_type {
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2LMHead(model) => {
                let mut context = prompt_ids.to_vec();
                context.extend_from_slice(decoded);
                let target = context.len() + 1;
                let next = if config.do_sample {
                    model.generate(
                        context.clone(),
                        target,
                        config.temperature,
                        config.top_k,
                        Some(config.top_p),
                    )?
                } else {
                    model.generate_greedy(context.clone(), target)?
                };
                Ok(next.get(context.len()).copied())
            },
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeoLMHead(model) => {
                let mut context = prompt_ids.to_vec();
                context.extend_from_slice(decoded);
                let target = context.len() + 1;
                let next = if config.do_sample {
                    model.generate(
                        context.clone(),
                        target,
                        config.temperature,
                        config.top_k,
                        Some(config.top_p),
                    )?
                } else {
                    model.generate_greedy(context.clone(), target)?
                };
                Ok(next.get(context.len()).copied())
            },
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJLMHead(model) => {
                let mut context = prompt_ids.to_vec();
                context.extend_from_slice(decoded);
                let target = context.len() + 1;
                let next = if config.do_sample {
                    model.generate(
                        context.clone(),
                        target,
                        config.temperature,
                        config.top_k,
                        Some(config.top_p),
                    )?
                } else {
                    model.generate_greedy(context.clone(), target)?
                };
                Ok(next.get(context.len()).copied())
            },
            #[cfg(feature = "t5")]
            AutoModelType::T5ForConditionalGeneration(model) => {
                // Only greedy T5 decoding is prefix-stable, so only greedy can be
                // replayed one token at a time and still agree with `generate`.
                if config.do_sample || config.num_beams.unwrap_or(1) > 1 {
                    return Err(TrustformersError::feature_unavailable(
                        "incremental T5 decoding is only available for greedy search \
                         (`do_sample = false`, `num_beams <= 1`): sampled and beam-searched \
                         prefixes are not stable, so streamed tokens would not match the \
                         non-streamed result"
                            .to_string(),
                        "encoder-decoder-streaming",
                    ));
                }
                let want = decoded.len() + 1;
                let produced = model.generate(prompt_ids.to_vec(), want, 1)?;
                Ok(produced.get(decoded.len()).copied())
            },
            _ => Err(Self::not_generative_error(self.architecture_name())),
        }
    }

    /// Run the encoder and return its real per-token hidden states.
    ///
    /// Only base (headless) architectures answer this: a task head replaces the
    /// hidden states with logits, and pooling logits would not be an embedding.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::FeatureUnavailable`] when the loaded
    /// checkpoint carries a task head instead of exposing hidden states.
    pub fn hidden_states(&self, input: TokenizedInput) -> Result<Tensor> {
        match &self.model_type {
            #[cfg(feature = "bert")]
            AutoModelType::Bert(model) => Ok(model.forward(input)?.last_hidden_state),
            #[cfg(feature = "roberta")]
            AutoModelType::Roberta(model) => Ok(model.forward(input)?.last_hidden_state),
            #[cfg(feature = "albert")]
            AutoModelType::Albert(model) => Ok(model.forward(input)?.last_hidden_state),
            #[cfg(feature = "gpt2")]
            AutoModelType::Gpt2(model) => Ok(model.forward(input)?.last_hidden_state),
            #[cfg(feature = "gpt_neo")]
            AutoModelType::GptNeo(model) => Ok(model.forward(input)?.last_hidden_state),
            #[cfg(feature = "gpt_j")]
            AutoModelType::GptJ(model) => Ok(model.forward(input)?.last_hidden_state),
            #[cfg(feature = "t5")]
            AutoModelType::T5(model) => {
                let t5_input = trustformers_models::t5::T5Input {
                    input_ids: input,
                    decoder_input_ids: None,
                    encoder_outputs: None,
                };
                Ok(model.forward(t5_input)?.last_hidden_state)
            },
            _ => Err(TrustformersError::feature_unavailable(
                format!(
                    "checkpoint `{}` carries a task head, so it produces logits rather than \
                     hidden states. Load the headless base encoder to extract embeddings.",
                    Config::architecture(&self.config)
                ),
                "feature-extraction",
            )),
        }
    }

    /// Hidden size this checkpoint produces per token.
    pub fn hidden_size(&self) -> usize {
        self.config.get_hidden_size() as usize
    }

    /// Sum of the model's own log-probabilities for `ids[skip_prefix..]`.
    ///
    /// Runs one real teacher-forced forward pass over `ids` and reads the
    /// log-softmax of the produced logits at each target position. This is the
    /// standard sequence score reported by generation APIs; it is measured from
    /// the model, never estimated.
    ///
    /// # Errors
    ///
    /// Fails when `ids` is too short to contain a predicted token, when the
    /// model has no LM head (its `forward` then returns hidden states whose last
    /// dimension is not the vocabulary), or when the forward pass itself fails.
    pub fn sequence_log_prob(&self, ids: &[u32], skip_prefix: usize) -> Result<f32> {
        if !self.is_decoder_only() {
            return Err(TrustformersError::feature_unavailable(
                "scoring an encoder-decoder sequence needs the encoder input alongside \
                 teacher-forced decoder ids; `sequence_log_prob` only supports decoder-only \
                 architectures"
                    .to_string(),
                "sequence-scoring",
            ));
        }
        let first_target = skip_prefix.max(1);
        if ids.len() <= first_target {
            return Err(TrustformersError::invalid_input_simple(format!(
                "sequence of {} tokens has no scored positions after skipping {}",
                ids.len(),
                skip_prefix
            )));
        }

        let input = Tensor::from_vec(
            ids.iter().map(|&id| id as f32).collect::<Vec<f32>>(),
            &[ids.len()],
        )?;
        let logits = self.forward(input)?;
        let shape = logits.shape();
        let vocab = *shape.last().ok_or_else(|| {
            TrustformersError::runtime_error("model returned a rank-0 logits tensor".to_string())
        })?;
        let data = logits.data()?;
        if data.len() < ids.len() * vocab {
            return Err(TrustformersError::runtime_error(format!(
                "logits tensor holds {} values, expected at least {} for {} positions",
                data.len(),
                ids.len() * vocab,
                ids.len()
            )));
        }

        let mut total = 0.0f32;
        for position in first_target..ids.len() {
            let row = &data[(position - 1) * vocab..position * vocab];
            let target = ids[position] as usize;
            if target >= vocab {
                return Err(TrustformersError::runtime_error(format!(
                    "token id {target} is outside the model's {vocab}-entry output space"
                )));
            }
            let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp =
                max_logit + row.iter().map(|&x| (x - max_logit).exp()).sum::<f32>().ln();
            total += row[target] - log_sum_exp;
        }
        Ok(total)
    }

    /// Whether this checkpoint carries a language-modelling head and can
    /// therefore generate text at all.
    pub fn is_generative(&self) -> bool {
        #[allow(unused_mut)]
        let mut generative = false;
        #[cfg(feature = "gpt2")]
        {
            generative |= matches!(self.model_type, AutoModelType::Gpt2LMHead(_));
        }
        #[cfg(feature = "gpt_neo")]
        {
            generative |= matches!(self.model_type, AutoModelType::GptNeoLMHead(_));
        }
        #[cfg(feature = "gpt_j")]
        {
            generative |= matches!(self.model_type, AutoModelType::GptJLMHead(_));
        }
        #[cfg(feature = "t5")]
        {
            generative |= matches!(
                self.model_type,
                AutoModelType::T5ForConditionalGeneration(_)
            );
        }
        generative
    }

    /// Whether this architecture decodes autoregressively from the prompt
    /// tokens (`true`) or from a separate decoder stream (`false`).
    fn is_decoder_only(&self) -> bool {
        match &self.model_type {
            #[cfg(feature = "t5")]
            AutoModelType::T5ForConditionalGeneration(_) => false,
            #[cfg(feature = "t5")]
            AutoModelType::T5(_) => false,
            _ => true,
        }
    }
}

/// Token ids produced by a real generation run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedSequence {
    /// Every token id the run yielded.
    ///
    /// Decoder-only models return `prompt || completion`; encoder-decoder
    /// models return the decoder tokens only.
    pub sequence: Vec<u32>,
    /// How many leading entries of `sequence` are prompt tokens.
    pub prompt_len: usize,
    /// `true` when decoding stopped because the model emitted an end-of-sequence
    /// token rather than because the length budget ran out.
    pub finished_by_eos: bool,
}

impl GeneratedSequence {
    /// Only called from the decoder-only (GPT-2 / GPT-Neo / GPT-J) arms of
    /// `generate_token_ids_with_backend`; T5 builds `Self` directly instead.
    #[cfg(any(feature = "gpt2", feature = "gpt_neo", feature = "gpt_j"))]
    fn decoder_only(sequence: Vec<u32>, prompt_len: usize, config: &GenerationConfig) -> Self {
        let finished_by_eos = match config.eos_token_id {
            Some(eos) => sequence.last().is_some_and(|&last| last == eos),
            // GPT-2 / GPT-Neo / GPT-J all use 50256 as `<|endoftext|>`.
            None => sequence.last().is_some_and(|&last| last == GPT_ENDOFTEXT_TOKEN_ID),
        };
        Self {
            sequence,
            prompt_len,
            finished_by_eos,
        }
    }

    /// The newly generated token ids, excluding the prompt.
    pub fn completion_ids(&self) -> &[u32] {
        &self.sequence[self.prompt_len.min(self.sequence.len())..]
    }
}

/// `<|endoftext|>` in the GPT-2 BPE vocabulary, shared by GPT-Neo and GPT-J.
const GPT_ENDOFTEXT_TOKEN_ID: u32 = 50256;

/// One real decoding step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationStep {
    /// The token the model emitted at this step.
    pub token_id: u32,
    /// Text that became visible because of this token.
    pub text_delta: String,
    /// Zero-based index of this step within the run.
    pub step_index: usize,
    /// Whether this token is the model's end-of-sequence marker.
    pub is_eos: bool,
}

/// Real incremental decoder returned by [`AutoModel::token_stream`].
///
/// Each `next()` performs one forward pass and yields the token that came out
/// of it, so the first chunk is available after a single step instead of after
/// the whole generation.
pub struct AutoModelTokenStream<'a> {
    model: Cow<'a, AutoModel>,
    prompt_ids: Vec<u32>,
    decoded: Vec<u32>,
    decoded_text: String,
    remaining_steps: usize,
    eos_token_id: u32,
    config: GenerationConfig,
    finished: bool,
}

impl<'a> AutoModelTokenStream<'a> {
    fn new(model: Cow<'a, AutoModel>, prompt: &str, config: &GenerationConfig) -> Result<Self> {
        let prompt_ids = model.encode_prompt(prompt)?;
        if !model.is_generative() {
            return Err(AutoModel::not_generative_error(model.architecture_name()));
        }
        let remaining_steps = if model.is_decoder_only() {
            model
                .resolve_decoder_target_len(prompt_ids.len(), config)
                .saturating_sub(prompt_ids.len())
        } else {
            model.resolve_encoder_decoder_target_len(config)
        };
        let eos_token_id = config.eos_token_id.unwrap_or(if model.is_decoder_only() {
            GPT_ENDOFTEXT_TOKEN_ID
        } else {
            T5_EOS_TOKEN_ID
        });
        Ok(Self {
            model,
            prompt_ids,
            decoded: Vec::new(),
            decoded_text: String::new(),
            remaining_steps,
            eos_token_id,
            config: config.clone(),
            finished: false,
        })
    }

    /// Token ids emitted so far (excluding the prompt).
    pub fn decoded_ids(&self) -> &[u32] {
        &self.decoded
    }

    fn advance(&mut self) -> Result<Option<GenerationStep>> {
        if self.finished || self.remaining_steps == 0 {
            self.finished = true;
            return Ok(None);
        }

        let next = self.model.step_once(&self.prompt_ids, &self.decoded, &self.config)?;
        let Some(token_id) = next else {
            self.finished = true;
            return Ok(None);
        };

        let step_index = self.decoded.len();
        self.decoded.push(token_id);
        self.remaining_steps -= 1;

        let tokenizer = self.model.require_tokenizer()?;
        let full_text = tokenizer.decode(&self.decoded)?;
        let text_delta = full_text
            .strip_prefix(self.decoded_text.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| full_text.clone());
        self.decoded_text = full_text;

        let is_eos = token_id == self.eos_token_id;
        if is_eos {
            self.finished = true;
        }

        Ok(Some(GenerationStep {
            token_id,
            text_delta,
            step_index,
            is_eos,
        }))
    }
}

/// `</s>` in the T5 SentencePiece vocabulary.
const T5_EOS_TOKEN_ID: u32 = 1;

impl Iterator for AutoModelTokenStream<'_> {
    type Item = anyhow::Result<GenerationStep>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.advance() {
            Ok(Some(step)) => Some(Ok(step)),
            Ok(None) => None,
            Err(err) => {
                self.finished = true;
                Some(Err(err.into()))
            },
        }
    }
}
