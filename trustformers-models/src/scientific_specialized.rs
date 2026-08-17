//! # Scientific Domain-Specialized Models
//!
//! This module provides specialized model configurations and implementations
//! optimized for scientific research, academic writing, and technical literature.
//!
//! ## Features
//!
//! - **Extended Context**: Support for long scientific papers (32K-128K tokens)
//! - **Scientific Vocabularies**: Enhanced tokenization for technical terminology
//! - **Domain Specialization**: Optimized for physics, chemistry, biology, mathematics
//! - **Citation Support**: Understanding of academic citation patterns
//! - **Formula Integration**: Enhanced handling of mathematical expressions and LaTeX
//! - **Multi-disciplinary**: Cross-domain scientific reasoning capabilities
//!
//! ## Supported Domains
//!
//! ### Physics
//! - Theoretical and experimental physics
//! - Quantum mechanics and relativity
//! - Statistical mechanics and thermodynamics
//!
//! ### Chemistry
//! - Organic and inorganic chemistry
//! - Chemical reactions and molecular structures
//! - Biochemistry and materials science
//!
//! ### Biology
//! - Molecular biology and genetics
//! - Ecology and evolutionary biology
//! - Biomedical research and pharmacology
//!
//! ### Mathematics
//! - Pure and applied mathematics
//! - Statistics and probability theory
//! - Computational mathematics
//!
//! ## Architecture
//!
//! [`ScientificModel`] embeds token ids with a domain-sized vocabulary and
//! [`ScientificAttention`] performs real scaled dot-product attention (see
//! [`trustformers_core::layers::GroupedQueryAttention`]), followed by a
//! SiLU-gated [`ScientificMLP`]. [`ScientificForCausalLM::generate`] drives
//! real autoregressive decoding through a byte-level tokenizer (see the
//! `generate_with_config` doc below) rather than echoing the prompt.
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use trustformers_models::scientific_specialized::{ScientificConfig, ScientificForCausalLM};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a scientific model for physics research
//! // (`physics_7b()` is a genuine ~7B-parameter preset; this example is marked
//! // `no_run` since constructing it allocates full-size weights.)
//! let config = ScientificConfig::physics_7b();
//! let model = ScientificForCausalLM::new(config)?;
//!
//! // For scientific text generation
//! let input = "The quantum mechanical behavior of electrons in superconductors";
//! let response = model.generate(input, 200)?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```

use crate::common_patterns::GenerationConfig;
use crate::generation_utils::GenerationUtils;
use anyhow::Result;
use scirs2_core::random::*;
use serde::{Deserialize, Serialize};
use trustformers_core::errors::{not_implemented, tensor_op_error, Result as CoreResult};
use trustformers_core::layers::{
    Embedding, FlashAttentionInput, GroupedQueryAttention, Linear, RMSNorm,
};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Config, Layer, Model};

/// First token id used for byte-level tokens; ids below this are reserved
/// (`bos_token_id`/`eos_token_id`/padding). See [`ScientificForCausalLM::generate`].
const BYTE_TOKEN_OFFSET: u32 = 8;
/// One token per possible byte value.
const BYTE_VOCAB_SIZE: u32 = 256;
/// Hard safety cap on total sequence length regardless of the caller's
/// `max_new_tokens`.
const MAX_SEQUENCE_LENGTH: usize = 4096;

fn min_vocab_size() -> usize {
    (BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE) as usize
}

/// Scientific domain specialization types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScientificDomain {
    /// General scientific research across disciplines
    General,
    /// Physics and astronomy
    Physics,
    /// Chemistry and materials science
    Chemistry,
    /// Biology and life sciences
    Biology,
    /// Mathematics and statistics
    Mathematics,
    /// Computer science and informatics
    ComputerScience,
    /// Environmental and earth sciences
    Environmental,
    /// Medical and health sciences
    Medical,
    /// Engineering disciplines
    Engineering,
    /// Social sciences and psychology
    SocialSciences,
}

/// Citation style preferences for scientific writing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CitationStyle {
    /// APA style (American Psychological Association)
    APA,
    /// MLA style (Modern Language Association)
    MLA,
    /// Chicago style
    Chicago,
    /// Harvard style
    Harvard,
    /// IEEE style (Institute of Electrical and Electronics Engineers)
    IEEE,
    /// Nature journal style
    Nature,
    /// Science journal style
    Science,
    /// Custom or mixed styles
    Custom,
}

/// Scientific model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScientificConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub rms_norm_eps: f32,
    pub use_cache: bool,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub model_type: String,

    // Scientific-specific fields
    pub scientific_domain: ScientificDomain,
    pub domain: ScientificDomain,
    pub citation_style: CitationStyle,
    pub latex_support: bool,
    pub formula_understanding: bool,
    pub scientific_notation: bool,
    pub cross_references: bool,
    pub experimental_data_analysis: bool,
    pub hypothesis_generation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub scaling_factor: f32,
}

/// Special tokens for scientific text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScientificSpecialTokens {
    pub equation_start: String,
    pub equation_end: String,
    pub citation_start: String,
    pub citation_end: String,
    pub reference_start: String,
    pub reference_end: String,
    pub figure_start: String,
    pub figure_end: String,
    pub table_start: String,
    pub table_end: String,
    pub abstract_start: String,
    pub abstract_end: String,
    pub hypothesis_start: String,
    pub hypothesis_end: String,
    pub conclusion_start: String,
    pub conclusion_end: String,
}

impl Default for ScientificConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50000, // Expanded for scientific terminology
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            hidden_act: "silu".to_string(),
            max_position_embeddings: 32768, // Long context for papers
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            rope_theta: 500000.0,
            rope_scaling: None,
            attention_bias: false,
            mlp_bias: false,
            model_type: "scientific".to_string(),
            scientific_domain: ScientificDomain::General,
            domain: ScientificDomain::General,
            citation_style: CitationStyle::APA,
            latex_support: true,
            formula_understanding: true,
            scientific_notation: true,
            cross_references: true,
            experimental_data_analysis: true,
            hypothesis_generation: true,
        }
    }
}

impl Config for ScientificConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(trustformers_core::errors::TrustformersError::config_error(
                "hidden_size must be divisible by num_attention_heads",
                "config_validation",
            ));
        }

        if let Some(num_kv_heads) = self.num_key_value_heads {
            if !self.num_attention_heads.is_multiple_of(num_kv_heads) {
                return Err(trustformers_core::errors::TrustformersError::config_error(
                    "num_attention_heads must be divisible by num_key_value_heads",
                    "config_validation",
                ));
            }
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Scientific"
    }
}

impl ScientificConfig {
    /// General scientific research model (7B parameters)
    pub fn scientific_7b() -> Self {
        Self {
            vocab_size: 50000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 32768,
            scientific_domain: ScientificDomain::General,
            domain: ScientificDomain::General,
            model_type: "scientific-general".to_string(),
            ..Self::default()
        }
    }

    /// Physics-specialized model (7B parameters)
    pub fn physics_7b() -> Self {
        Self {
            vocab_size: 45000, // Physics-focused vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 32768,
            scientific_domain: ScientificDomain::Physics,
            citation_style: CitationStyle::Nature,
            model_type: "scientific-physics".to_string(),
            ..Self::default()
        }
    }

    /// Chemistry-specialized model (7B parameters)
    pub fn chemistry_7b() -> Self {
        Self {
            vocab_size: 48000, // Chemistry-focused vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 32768,
            scientific_domain: ScientificDomain::Chemistry,
            citation_style: CitationStyle::APA,
            model_type: "scientific-chemistry".to_string(),
            ..Self::default()
        }
    }

    /// Biology-specialized model (7B parameters)
    pub fn biology_7b() -> Self {
        Self {
            vocab_size: 52000, // Biology-focused vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 32768,
            scientific_domain: ScientificDomain::Biology,
            citation_style: CitationStyle::Nature,
            model_type: "scientific-biology".to_string(),
            ..Self::default()
        }
    }

    /// Mathematics-specialized model (7B parameters)
    pub fn mathematics_7b() -> Self {
        Self {
            vocab_size: 40000, // Math-focused vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 16384, // Shorter context for mathematical proofs
            scientific_domain: ScientificDomain::Mathematics,
            citation_style: CitationStyle::APA,
            latex_support: true,
            formula_understanding: true,
            model_type: "scientific-mathematics".to_string(),
            ..Self::default()
        }
    }

    /// Computer Science-specialized model (7B parameters)
    pub fn computer_science_7b() -> Self {
        Self {
            vocab_size: 55000, // CS-focused vocabulary including code
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 16384,
            scientific_domain: ScientificDomain::ComputerScience,
            citation_style: CitationStyle::IEEE,
            model_type: "scientific-cs".to_string(),
            ..Self::default()
        }
    }

    /// Environmental Science model (7B parameters)
    pub fn environmental_7b() -> Self {
        Self {
            vocab_size: 46000, // Environmental science vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 32768,
            scientific_domain: ScientificDomain::Environmental,
            citation_style: CitationStyle::APA,
            experimental_data_analysis: true,
            model_type: "scientific-environmental".to_string(),
            ..Self::default()
        }
    }

    /// Large scientific model for comprehensive research (13B parameters)
    pub fn scientific_13b() -> Self {
        Self {
            vocab_size: 60000, // Very large vocabulary
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: Some(8),
            max_position_embeddings: 65536, // Very long context
            scientific_domain: ScientificDomain::General,
            domain: ScientificDomain::General,
            model_type: "scientific-large".to_string(),
            ..Self::default()
        }
    }

    /// Get special tokens for the scientific model
    pub fn get_special_tokens(&self) -> ScientificSpecialTokens {
        ScientificSpecialTokens {
            equation_start: "<eq>".to_string(),
            equation_end: "</eq>".to_string(),
            citation_start: "<cite>".to_string(),
            citation_end: "</cite>".to_string(),
            reference_start: "<ref>".to_string(),
            reference_end: "</ref>".to_string(),
            figure_start: "<fig>".to_string(),
            figure_end: "</fig>".to_string(),
            table_start: "<tab>".to_string(),
            table_end: "</tab>".to_string(),
            abstract_start: "<abstract>".to_string(),
            abstract_end: "</abstract>".to_string(),
            hypothesis_start: "<hypothesis>".to_string(),
            hypothesis_end: "</hypothesis>".to_string(),
            conclusion_start: "<conclusion>".to_string(),
            conclusion_end: "</conclusion>".to_string(),
        }
    }

    /// Create configuration from domain and size
    pub fn from_domain_and_size(domain: ScientificDomain, size: &str) -> Option<Self> {
        match (domain, size) {
            (ScientificDomain::General, "7b") => Some(Self::scientific_7b()),
            (ScientificDomain::General, "13b") => Some(Self::scientific_13b()),
            (ScientificDomain::Physics, "7b") => Some(Self::physics_7b()),
            (ScientificDomain::Chemistry, "7b") => Some(Self::chemistry_7b()),
            (ScientificDomain::Biology, "7b") => Some(Self::biology_7b()),
            (ScientificDomain::Mathematics, "7b") => Some(Self::mathematics_7b()),
            (ScientificDomain::ComputerScience, "7b") => Some(Self::computer_science_7b()),
            (ScientificDomain::Environmental, "7b") => Some(Self::environmental_7b()),
            _ => None,
        }
    }
}

/// Scientific model implementation
pub struct ScientificModel {
    config: ScientificConfig,
    embed_tokens: Embedding,
    layers: Vec<ScientificLayer>,
    norm: RMSNorm,
}

impl Model for ScientificModel {
    type Config = ScientificConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        // Convert input to token IDs if needed
        let token_ids: Vec<u32> = input.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        Layer::forward(self, token_ids)
    }

    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> CoreResult<()> {
        checked_unimplemented_load(reader, "ScientificModel")
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let embed_params = self.embed_tokens.parameter_count();
        let layers_params: usize = self.layers.iter().map(|layer| layer.parameter_count()).sum();
        let norm_params = self.norm.parameter_count();

        embed_params + layers_params + norm_params
    }
}

/// Scientific transformer layer with domain-specific optimizations
pub struct ScientificLayer {
    self_attention: ScientificAttention,
    feed_forward: ScientificMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

/// Scientific attention mechanism.
///
/// Backed by [`GroupedQueryAttention`], which performs real scaled
/// dot-product attention (`softmax(QK^T / sqrt(head_dim)) V`) with causal
/// masking and, when `num_key_value_heads < num_attention_heads`,
/// grouped-query key/value sharing. The previous implementation computed
/// `q_proj(x) + v_proj(x)` and discarded the key projection entirely; this
/// one actually uses it to compute attention scores.
pub struct ScientificAttention {
    attn: GroupedQueryAttention,
}

/// Scientific MLP with domain-aware processing
pub struct ScientificMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

/// Scientific model for causal language modeling
pub struct ScientificForCausalLM {
    model: ScientificModel,
    lm_head: Linear,
    config: ScientificConfig,
}

impl ScientificForCausalLM {
    pub fn new(config: ScientificConfig) -> Result<Self> {
        config.validate()?;

        // Create the base model
        let model = ScientificModel::new(config.clone())?;

        // Create the language modeling head
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);

        Ok(Self {
            model,
            lm_head,
            config,
        })
    }

    /// Generate a continuation of `input`, sampling with a fresh
    /// (non-reproducible) random source. For reproducible output see
    /// [`ScientificForCausalLM::generate_with_seed`].
    ///
    /// This used to return `Ok(format!("[Scientific Generated]: {}", prompt))`,
    /// a formatted echo of the input. It now runs real autoregressive
    /// decoding: a byte-level tokenizer (see the module-level docs) encodes
    /// the prompt, the real transformer forward pass produces logits at each
    /// step, and [`GenerationUtils`] samples the next token. With random
    /// (untrained) weights the output is not fluent prose, but it is genuine
    /// model output that varies with the seed, prompt, and weights.
    pub fn generate(&self, input: &str, max_length: usize) -> Result<String> {
        let mut rng = thread_rng();
        self.generate_impl(input, max_length, &mut rng)
    }

    /// Generate a continuation of `input` with a seeded RNG, so the output is
    /// reproducible for a fixed model, prompt, and seed - and changes when
    /// the seed changes.
    pub fn generate_with_seed(&self, input: &str, max_length: usize, seed: u64) -> Result<String> {
        let mut rng = StdRng::seed_from_u64(seed);
        self.generate_impl(input, max_length, &mut rng)
    }

    fn generate_impl(&self, input: &str, max_length: usize, rng: &mut impl Rng) -> Result<String> {
        // Create generation config optimized for scientific text
        let gen_config = GenerationConfig {
            max_new_tokens: max_length,
            temperature: 0.7, // Balanced for scientific accuracy
            top_p: 0.9,
            do_sample: true,
            repetition_penalty: 1.1,
            ..Default::default()
        };

        // Enhance prompt for scientific context
        let enhanced_prompt = self.enhance_scientific_prompt(input)?;
        self.generate_with_config(&enhanced_prompt, &gen_config, rng)
    }

    pub fn analyze_scientific_text(&self, text: &str) -> Result<ScientificAnalysis> {
        // Analyze scientific text for domain-specific patterns
        let domain_classification = self.classify_scientific_domain(text)?;
        let citation_count = self.count_citations(text)?;
        let equation_count = self.count_equations(text)?;
        let figure_references = self.count_figure_references(text)?;
        let hypothesis_statements = self.extract_hypothesis_statements(text)?;
        let key_findings = self.extract_key_findings(text)?;
        let methodology_description = self.extract_methodology(text)?;
        let statistical_significance = self.assess_statistical_significance(text)?;
        let reproducibility_score = self.assess_reproducibility(text)?;

        Ok(ScientificAnalysis {
            domain_classification,
            citation_count,
            equation_count,
            figure_references,
            hypothesis_statements,
            key_findings,
            methodology_description,
            statistical_significance,
            reproducibility_score,
        })
    }

    pub fn generate_hypothesis(&self, context: &str) -> Result<String> {
        // Generate scientific hypotheses based on context
        let domain = self.classify_scientific_domain(context)?;
        let hypothesis_prompt = self.create_hypothesis_prompt(context, &domain)?;

        let gen_config = GenerationConfig {
            max_new_tokens: 200,
            temperature: 0.8, // Slightly more creative for hypothesis generation
            top_p: 0.9,
            do_sample: true,
            repetition_penalty: 1.2,
            ..Default::default()
        };

        let mut rng = thread_rng();
        self.generate_with_config(&hypothesis_prompt, &gen_config, &mut rng)
    }

    pub fn summarize_paper(&self, paper_text: &str) -> Result<String> {
        // Create scientific paper summaries
        let analysis = self.analyze_scientific_text(paper_text)?;
        let summary_prompt = self.create_summary_prompt(paper_text, &analysis)?;

        let gen_config = GenerationConfig {
            max_new_tokens: 500,
            temperature: 0.6, // Conservative for accurate summaries
            top_p: 0.85,
            do_sample: true,
            repetition_penalty: 1.1,
            ..Default::default()
        };

        let mut rng = thread_rng();
        self.generate_with_config(&summary_prompt, &gen_config, &mut rng)
    }

    /// Real forward pass from token ids straight through to logits. Used by
    /// generation to drive autoregressive decoding without going through the
    /// `Layer`/`Model` trait objects.
    fn forward_logits(&self, input_ids: &[u32]) -> CoreResult<Tensor> {
        let hidden_states = Layer::forward(&self.model, input_ids.to_vec())?;
        self.lm_head.forward(hidden_states)
    }

    /// Encode `text` as byte-level token ids, prefixed with `bos_token_id`.
    /// See the module-level docs for why this crate defines its own
    /// tokenizer rather than assuming a BPE vocabulary these configs do not
    /// actually ship.
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

    /// Decode generated token ids back to text; ids outside the byte range
    /// carry no text content and are dropped.
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

    /// Run real autoregressive generation and decode only the newly
    /// generated suffix - never the prompt itself, which is what made the
    /// old implementation an echo rather than generation.
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
            let logits_tensor = self.forward_logits(&generated)?;
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
/// exact.
fn restrict_logits_to_byte_vocab(logits: &mut [f32], eos_token_id: u32) {
    for (id, logit) in logits.iter_mut().enumerate() {
        let id = id as u32;
        let in_byte_range = (BYTE_TOKEN_OFFSET..BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE).contains(&id);
        if !in_byte_range && id != eos_token_id {
            *logit = f32::NEG_INFINITY;
        }
    }
}

// Implementation of ScientificModel
impl ScientificModel {
    pub fn new(config: ScientificConfig) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(ScientificLayer::new(&config)?);
        }

        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }
}

// Implementation of ScientificLayer
impl ScientificLayer {
    pub fn new(config: &ScientificConfig) -> Result<Self> {
        let self_attention = ScientificAttention::new(config)?;
        let feed_forward = ScientificMLP::new(config)?;
        let input_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            self_attention,
            feed_forward,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// Implementation of ScientificAttention
impl ScientificAttention {
    pub fn new(config: &ScientificConfig) -> Result<Self> {
        let num_kv_heads = config.num_key_value_heads.unwrap_or(config.num_attention_heads);
        let mut attn = GroupedQueryAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            num_kv_heads,
            // ScientificConfig does not expose an attention-dropout knob;
            // 0.0 also keeps inference deterministic.
            0.0,
            config.attention_bias,
        )?;
        // Decoder-only causal language model: generation must not see future
        // tokens.
        attn.set_causal(true);
        Ok(Self { attn })
    }

    /// Replace the query/key/value/output projections. Only used by tests to
    /// prove the key projection actually participates in the output.
    #[cfg(test)]
    fn set_projections(&mut self, query: Linear, key: Linear, value: Linear, out_proj: Linear) {
        self.attn.set_projections(query, key, value, out_proj);
    }
}

// Implementation of ScientificMLP
impl ScientificMLP {
    pub fn new(config: &ScientificConfig) -> Result<Self> {
        let gate_proj = Linear::new(
            config.hidden_size,
            config.intermediate_size,
            config.mlp_bias,
        );
        let up_proj = Linear::new(
            config.hidden_size,
            config.intermediate_size,
            config.mlp_bias,
        );
        let down_proj = Linear::new(
            config.intermediate_size,
            config.hidden_size,
            config.mlp_bias,
        );

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

// Layer trait implementations
impl Layer for ScientificModel {
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        // Convert token IDs to embeddings
        let mut hidden_states = self.embed_tokens.forward(input)?;

        // Pass through all layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        // Apply final normalization
        let output = self.norm.forward(hidden_states)?;
        Ok(output)
    }
}

impl Layer for ScientificLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        // Pre-norm architecture
        let normalized_input = self.input_layernorm.forward(input.clone())?;
        let attn_output = self.self_attention.forward(normalized_input)?;
        let residual1 = input.add(&attn_output)?;

        let normalized_residual = self.post_attention_layernorm.forward(residual1.clone())?;
        let mlp_output = self.feed_forward.forward(normalized_residual)?;
        let residual2 = residual1.add(&mlp_output)?;

        Ok(residual2)
    }
}

impl ScientificLayer {
    pub fn parameter_count(&self) -> usize {
        self.self_attention.parameter_count()
            + self.feed_forward.parameter_count()
            + self.input_layernorm.parameter_count()
            + self.post_attention_layernorm.parameter_count()
    }
}

impl Layer for ScientificAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        self.attn.forward(FlashAttentionInput {
            hidden_states: input,
            attention_mask: None,
        })
    }
}

impl ScientificAttention {
    pub fn parameter_count(&self) -> usize {
        self.attn.parameter_count()
    }
}

impl Layer for ScientificMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        // SiLU activation MLP with scientific processing
        let gate_output = self.gate_proj.forward(input.clone())?;
        let up_output = self.up_proj.forward(input)?;

        // Apply SiLU activation
        let gate_activated = match &gate_output {
            Tensor::F32(arr) => {
                let activated = arr.mapv(|x| x / (1.0 + (-x).exp()));
                Tensor::F32(activated)
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type for SiLU activation",
                ))
            },
        };

        // Element-wise multiply
        let combined = match (&gate_activated, &up_output) {
            (Tensor::F32(gate_arr), Tensor::F32(up_arr)) => {
                let result = gate_arr * up_arr;
                Tensor::F32(result)
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor types for element-wise multiplication",
                ))
            },
        };

        self.down_proj.forward(combined)
    }
}

impl ScientificMLP {
    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
    }
}

// Model trait implementation for ScientificForCausalLM
impl Model for ScientificForCausalLM {
    type Config = ScientificConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        self.forward_logits(&input)
    }

    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> CoreResult<()> {
        checked_unimplemented_load(reader, "ScientificForCausalLM")
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.model.num_parameters() + self.lm_head.parameter_count()
    }
}

/// Shared `load_pretrained` body for this family: reads the stream so I/O
/// failures are reported honestly, then refuses instead of pretending the
/// (entirely discarded) bytes became model weights. The previous
/// implementation wrote the buffer to a temp file, `println!`ed a success
/// message, deleted the file, and returned `Ok(())` without touching a
/// single layer's weights.
fn checked_unimplemented_load(
    reader: &mut dyn std::io::Read,
    architecture: &str,
) -> CoreResult<()> {
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).map_err(|e| {
        trustformers_core::errors::TrustformersError::io_error(format!(
            "Failed to read weight data: {}",
            e
        ))
    })?;
    if buffer.is_empty() {
        return Err(trustformers_core::errors::TrustformersError::io_error(
            "Weight data is empty".to_string(),
        ));
    }
    Err(not_implemented(format!(
        "{architecture}::load_pretrained: binding a checkpoint's tensors into this \
         architecture's layers is not implemented (read {} bytes but could not bind them). \
         Construct the model with default/random weights, or use \
         trustformers_models::weight_loading for architectures with an implemented binder.",
        buffer.len()
    )))
}

// Helper methods for ScientificForCausalLM
impl ScientificForCausalLM {
    fn enhance_scientific_prompt(&self, prompt: &str) -> Result<String> {
        // Enhance prompt with scientific context markers
        let domain_context = match self.config.domain {
            ScientificDomain::Physics => "[PHYSICS]",
            ScientificDomain::Chemistry => "[CHEMISTRY]",
            ScientificDomain::Biology => "[BIOLOGY]",
            ScientificDomain::Mathematics => "[MATH]",
            ScientificDomain::ComputerScience => "[CS]",
            ScientificDomain::Environmental => "[ENV_SCI]",
            _ => "[SCIENCE]",
        };

        let enhanced = format!("{} {}", domain_context, prompt);
        Ok(enhanced)
    }

    fn classify_scientific_domain(&self, text: &str) -> Result<ScientificDomain> {
        let text_lower = text.to_lowercase();

        // Physics keywords
        if text_lower.contains("physics")
            || text_lower.contains("quantum")
            || text_lower.contains("energy")
            || text_lower.contains("force")
            || text_lower.contains("particle")
            || text_lower.contains("wave")
        {
            Ok(ScientificDomain::Physics)
        }
        // Chemistry keywords
        else if text_lower.contains("chemistry")
            || text_lower.contains("molecule")
            || text_lower.contains("reaction")
            || text_lower.contains("compound")
            || text_lower.contains("element")
            || text_lower.contains("bond")
        {
            Ok(ScientificDomain::Chemistry)
        }
        // Biology keywords
        else if text_lower.contains("biology")
            || text_lower.contains("cell")
            || text_lower.contains("gene")
            || text_lower.contains("protein")
            || text_lower.contains("organism")
            || text_lower.contains("dna")
        {
            Ok(ScientificDomain::Biology)
        }
        // Mathematics keywords
        else if text_lower.contains("mathematics")
            || text_lower.contains("equation")
            || text_lower.contains("theorem")
            || text_lower.contains("proof")
            || text_lower.contains("formula")
            || text_lower.contains("function")
        {
            Ok(ScientificDomain::Mathematics)
        }
        // Computer Science keywords
        else if text_lower.contains("computer")
            || text_lower.contains("algorithm")
            || text_lower.contains("software")
            || text_lower.contains("programming")
            || text_lower.contains("data")
            || text_lower.contains("network")
        {
            Ok(ScientificDomain::ComputerScience)
        }
        // Environmental Science keywords
        else if text_lower.contains("environment")
            || text_lower.contains("climate")
            || text_lower.contains("ecosystem")
            || text_lower.contains("pollution")
            || text_lower.contains("sustainability")
            || text_lower.contains("carbon")
        {
            Ok(ScientificDomain::Environmental)
        } else {
            Ok(ScientificDomain::General)
        }
    }

    fn count_citations(&self, text: &str) -> Result<usize> {
        // Count scientific citations
        let mut count = 0;

        // Count references like [1], [2], etc.
        count += text.matches(char::is_numeric).count() / 10; // Rough estimate

        // Count DOI references
        count += text.matches("DOI:").count();
        count += text.matches("doi:").count();

        // Count et al. references
        count += text.matches("et al.").count();

        // Count year references like (2023), (2022), etc.
        count += text.matches("(202").count();
        count += text.matches("(201").count();

        Ok(count)
    }

    fn count_equations(&self, text: &str) -> Result<usize> {
        // Count mathematical equations
        let mut count = 0;

        // Count LaTeX-style equations
        count += text.matches("\\\\begin{equation}").count();
        count += text.matches("\\\\[").count();
        count += text.matches("$$").count() / 2; // Paired delimiters

        // Count inline math
        count += text.matches('$').count() / 2; // Paired delimiters

        // Count equals signs as rough equation indicator
        count += text.matches(" = ").count();

        Ok(count)
    }

    fn count_figure_references(&self, text: &str) -> Result<usize> {
        // Count figure references
        let mut count = 0;

        count += text.matches("Figure").count();
        count += text.matches("Fig.").count();
        count += text.matches("figure").count();
        count += text.matches("Table").count();
        count += text.matches("table").count();

        Ok(count)
    }

    fn extract_hypothesis_statements(&self, text: &str) -> Result<Vec<String>> {
        let mut hypotheses = Vec::new();

        // Look for hypothesis indicators
        let lines: Vec<&str> = text.lines().collect();
        for line in lines {
            let line_lower = line.to_lowercase();
            if line_lower.contains("hypothesis")
                || line_lower.contains("we hypothesize")
                || line_lower.contains("we propose")
                || line_lower.contains("we suggest")
            {
                hypotheses.push(line.to_string());
            }
        }

        Ok(hypotheses)
    }

    fn extract_key_findings(&self, text: &str) -> Result<Vec<String>> {
        let mut findings = Vec::new();

        // Look for finding indicators
        let lines: Vec<&str> = text.lines().collect();
        for line in lines {
            let line_lower = line.to_lowercase();
            if line_lower.contains("we found")
                || line_lower.contains("results show")
                || line_lower.contains("we observed")
                || line_lower.contains("conclusion")
                || line_lower.contains("significant")
                || line_lower.contains("demonstrates")
            {
                findings.push(line.to_string());
            }
        }

        Ok(findings)
    }

    fn extract_methodology(&self, text: &str) -> Result<Option<String>> {
        // Extract methodology section
        let text_lower = text.to_lowercase();

        if text_lower.contains("methodology")
            || text_lower.contains("methods")
            || text_lower.contains("experimental")
            || text_lower.contains("procedure")
        {
            // Find methodology section (simplified)
            let lines: Vec<&str> = text.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                let line_lower = line.to_lowercase();
                if line_lower.contains("method") || line_lower.contains("procedure") {
                    // Return next few lines as methodology
                    let end_idx = (i + 5).min(lines.len());
                    let methodology = lines[i..end_idx].join(" ");
                    return Ok(Some(methodology));
                }
            }
        }

        Ok(None)
    }

    fn assess_statistical_significance(&self, text: &str) -> Result<bool> {
        // Check for statistical significance indicators
        let text_lower = text.to_lowercase();

        Ok(text_lower.contains("p <")
            || text_lower.contains("p-value")
            || text_lower.contains("significant")
            || text_lower.contains("confidence interval")
            || text_lower.contains("alpha")
            || text_lower.contains("statistical"))
    }

    fn assess_reproducibility(&self, text: &str) -> Result<f32> {
        // Assess reproducibility score based on various factors
        let mut score = 0.0;
        let text_lower = text.to_lowercase();

        // Check for data availability
        if text_lower.contains("data available") || text_lower.contains("supplementary") {
            score += 0.2;
        }

        // Check for code availability
        if text_lower.contains("code")
            || text_lower.contains("software")
            || text_lower.contains("github")
            || text_lower.contains("repository")
        {
            score += 0.2;
        }

        // Check for detailed methodology
        if text_lower.contains("methodology")
            || text_lower.contains("procedure")
            || text_lower.contains("protocol")
        {
            score += 0.2;
        }

        // Check for statistical information
        if text_lower.contains("sample size")
            || text_lower.contains("n =")
            || text_lower.contains("participants")
        {
            score += 0.2;
        }

        // Check for replication information
        if text_lower.contains("replicated")
            || text_lower.contains("repeated")
            || text_lower.contains("validation")
        {
            score += 0.2;
        }

        Ok(score)
    }

    fn create_hypothesis_prompt(&self, context: &str, domain: &ScientificDomain) -> Result<String> {
        let domain_instruction = match domain {
            ScientificDomain::Physics => {
                "Generate a physics hypothesis based on the following context:"
            },
            ScientificDomain::Chemistry => {
                "Generate a chemistry hypothesis based on the following context:"
            },
            ScientificDomain::Biology => {
                "Generate a biology hypothesis based on the following context:"
            },
            ScientificDomain::Mathematics => {
                "Generate a mathematical hypothesis based on the following context:"
            },
            ScientificDomain::ComputerScience => {
                "Generate a computer science hypothesis based on the following context:"
            },
            ScientificDomain::Environmental => {
                "Generate an environmental science hypothesis based on the following context:"
            },
            _ => "Generate a scientific hypothesis based on the following context:",
        };

        let prompt = format!("{} {}", domain_instruction, context);
        Ok(prompt)
    }

    fn create_summary_prompt(
        &self,
        paper_text: &str,
        analysis: &ScientificAnalysis,
    ) -> Result<String> {
        let domain_context = match analysis.domain_classification {
            ScientificDomain::Physics => "[PHYSICS PAPER]",
            ScientificDomain::Chemistry => "[CHEMISTRY PAPER]",
            ScientificDomain::Biology => "[BIOLOGY PAPER]",
            ScientificDomain::Mathematics => "[MATH PAPER]",
            ScientificDomain::ComputerScience => "[CS PAPER]",
            ScientificDomain::Environmental => "[ENV SCI PAPER]",
            _ => "[SCIENTIFIC PAPER]",
        };

        let summary_info = format!(
            "Citations: {}, Equations: {}, Figures: {}",
            analysis.citation_count, analysis.equation_count, analysis.figure_references
        );

        let prompt = format!(
            "{} Summarize the following scientific paper. {}\n\n{}",
            domain_context, summary_info, paper_text
        );

        Ok(prompt)
    }
}

/// Scientific text analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScientificAnalysis {
    pub domain_classification: ScientificDomain,
    pub citation_count: usize,
    pub equation_count: usize,
    pub figure_references: usize,
    pub hypothesis_statements: Vec<String>,
    pub key_findings: Vec<String>,
    pub methodology_description: Option<String>,
    pub statistical_significance: bool,
    pub reproducibility_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scientific_config_creation() {
        let config = ScientificConfig::scientific_7b();
        assert_eq!(config.scientific_domain, ScientificDomain::General);
        assert_eq!(config.vocab_size, 50000);
        assert_eq!(config.max_position_embeddings, 32768);
    }

    #[test]
    fn test_physics_config() {
        let config = ScientificConfig::physics_7b();
        assert_eq!(config.scientific_domain, ScientificDomain::Physics);
        assert_eq!(config.citation_style, CitationStyle::Nature);
        assert!(config.latex_support);
    }

    #[test]
    fn test_special_tokens() {
        let config = ScientificConfig::scientific_7b();
        let tokens = config.get_special_tokens();
        assert_eq!(tokens.equation_start, "<eq>");
        assert_eq!(tokens.citation_start, "<cite>");
    }

    #[test]
    fn test_domain_and_size_creation() {
        let config = ScientificConfig::from_domain_and_size(ScientificDomain::Chemistry, "7b");
        assert!(config.is_some());
        let config = config.expect("operation failed");
        assert_eq!(config.scientific_domain, ScientificDomain::Chemistry);
    }

    #[test]
    fn test_config_validation() {
        let config = ScientificConfig::scientific_7b();
        assert!(config.validate().is_ok());
    }

    /// A tiny configuration for fast tests: real compute, not a 7B allocation.
    fn tiny_config() -> ScientificConfig {
        ScientificConfig {
            vocab_size: 512,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 128,
            rope_scaling: None,
            ..ScientificConfig::default()
        }
    }

    /// Deterministic pseudo-random data so assertions are reproducible.
    fn deterministic_data(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
        let mut data = Vec::with_capacity(count);
        for _ in 0..count {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            data.push(unit * 2.0 - 1.0);
        }
        data
    }

    fn deterministic_linear(in_features: usize, out_features: usize, seed: u32) -> Linear {
        let data = deterministic_data(out_features * in_features, seed);
        let mut layer = Linear::new(in_features, out_features, false);
        layer
            .set_weight(Tensor::from_vec(data, &[out_features, in_features]).expect("weight shape"))
            .expect("set weight");
        layer
    }

    #[test]
    fn attention_is_real_scaled_dot_product_not_q_plus_v() {
        // Regression test for the `q + v` bug: two attention layers that are
        // identical except for the key projection must produce different
        // outputs.
        let config = tiny_config();
        let head_dim = config.hidden_size / config.num_attention_heads;
        let kv_heads = config.num_key_value_heads.expect("configured");
        let kv_hidden = kv_heads * head_dim;

        let mut attn_a = ScientificAttention::new(&config).expect("attention a");
        attn_a.set_projections(
            deterministic_linear(config.hidden_size, config.hidden_size, 1),
            deterministic_linear(config.hidden_size, kv_hidden, 2),
            deterministic_linear(config.hidden_size, kv_hidden, 3),
            deterministic_linear(config.hidden_size, config.hidden_size, 4),
        );

        let mut attn_b = ScientificAttention::new(&config).expect("attention b");
        attn_b.set_projections(
            deterministic_linear(config.hidden_size, config.hidden_size, 1),
            deterministic_linear(config.hidden_size, kv_hidden, 99),
            deterministic_linear(config.hidden_size, kv_hidden, 3),
            deterministic_linear(config.hidden_size, config.hidden_size, 4),
        );

        // Deliberately *not* a uniform vector repeated across positions: if
        // every position held the same value, causal softmax over identical
        // scores would be uniform regardless of K (an average of identical
        // V rows equals that row no matter how it was weighted), which
        // would make this test pass even with the old q+v code path by
        // accident rather than by actually exercising K.
        let input = Tensor::from_vec(
            deterministic_data(5 * config.hidden_size, 50),
            &[5, config.hidden_size],
        )
        .expect("input tensor");

        let out_a = attn_a.forward(input.clone()).expect("forward a");
        let out_b = attn_b.forward(input).expect("forward b");

        let data_a = out_a.data().expect("data a");
        let data_b = out_b.data().expect("data b");
        let max_diff = data_a
            .iter()
            .zip(data_b.iter())
            .fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()));
        assert!(
            max_diff > 1e-6,
            "changing only the key projection must change the attention output \
             (max diff was {max_diff}); the old q+v implementation ignored K entirely"
        );
    }

    #[test]
    fn causal_attention_does_not_leak_future_tokens() {
        let config = tiny_config();
        let attn = ScientificAttention::new(&config).expect("attention");
        let seq_len = 6;
        let base = vec![0.05_f32; seq_len * config.hidden_size];
        let mut perturbed = base.clone();
        for value in perturbed[4 * config.hidden_size..].iter_mut() {
            *value += 3.0;
        }

        let base_out = attn
            .forward(Tensor::from_vec(base, &[seq_len, config.hidden_size]).expect("base"))
            .expect("base forward");
        let perturbed_out = attn
            .forward(
                Tensor::from_vec(perturbed, &[seq_len, config.hidden_size]).expect("perturbed"),
            )
            .expect("perturbed forward");

        let prefix = 4 * config.hidden_size;
        let base_data = base_out.data().expect("base data");
        let perturbed_data = perturbed_out.data().expect("perturbed data");
        let max_diff = base_data[..prefix]
            .iter()
            .zip(perturbed_data[..prefix].iter())
            .fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()));
        assert!(
            max_diff < 1e-4,
            "causal attention leaked future tokens into the past"
        );
    }

    fn model() -> ScientificForCausalLM {
        ScientificForCausalLM::new(tiny_config()).expect("model construction")
    }

    #[test]
    fn generate_does_not_echo_the_prompt() {
        let m = model();
        let prompt = "The quantum mechanical behavior of superconductors";
        let output = m.generate_with_seed(prompt, 12, 7).expect("generate");
        assert!(
            !output.contains(prompt),
            "output must not echo the prompt back: {output:?}"
        );
        assert!(!output.starts_with("[Scientific Generated]"));
    }

    #[test]
    fn generate_with_seed_is_reproducible() {
        let m = model();
        let a = m.generate_with_seed("Consider the following hypothesis", 16, 42).expect("a");
        let b = m.generate_with_seed("Consider the following hypothesis", 16, 42).expect("b");
        assert_eq!(a, b, "the same seed must produce the same output");
    }

    #[test]
    fn different_seeds_produce_different_output() {
        let m = model();
        let a = m.generate_with_seed("Consider the following hypothesis", 24, 1).expect("a");
        let b = m.generate_with_seed("Consider the following hypothesis", 24, 2).expect("b");
        assert_ne!(
            a, b,
            "different seeds must (almost certainly) sample different tokens"
        );
    }

    #[test]
    fn encode_decode_byte_roundtrip() {
        let m = model();
        let text = "E = mc^2 describes mass-energy equivalence.";
        let ids = m.encode_text(text).expect("encode");
        assert_eq!(ids[0], m.config.bos_token_id);
        let decoded = ScientificForCausalLM::decode_tokens(&ids[1..]);
        assert_eq!(decoded, text);
    }

    #[test]
    fn load_pretrained_reports_an_honest_error_instead_of_fake_success() {
        let config = tiny_config();
        let mut model = ScientificForCausalLM::new(config).expect("model");
        let mut reader = std::io::Cursor::new(vec![0u8; 4096]);
        let result = Model::load_pretrained(&mut model, &mut reader);
        assert!(
            result.is_err(),
            "load_pretrained must not report success when no weights were actually bound"
        );
    }
}
