//! # Creative Writing Domain-Specialized Models
//!
//! This module provides specialized model configurations and implementations
//! optimized for creative writing tasks, storytelling, and literary content generation.
//!
//! ## Features
//!
//! - **Literary Styles**: Support for various writing styles and genres
//! - **Character Development**: Enhanced understanding of character arcs and dialogue
//! - **Narrative Structure**: Awareness of story structure and pacing
//! - **Genre Specialization**: Optimized for specific literary genres
//! - **Stylistic Analysis**: Understanding of literary devices and techniques
//! - **Creative Constraints**: Support for formal constraints (poetry, etc.)
//!
//! ## Supported Genres
//!
//! ### Fiction
//! - Literary fiction
//! - Science fiction and fantasy
//! - Mystery and thriller
//! - Romance and drama
//! - Historical fiction
//!
//! ### Poetry
//! - Free verse and formal poetry
//! - Haiku and other traditional forms
//! - Song lyrics and ballads
//! - Experimental poetry
//!
//! ### Screenwriting
//! - Film and television scripts
//! - Stage plays and dialogues
//! - Interactive narratives
//!
//! ### Non-fiction Creative
//! - Creative non-fiction
//! - Memoirs and autobiographies
//! - Travel writing
//! - Food and lifestyle writing
//!
//! ## Architecture
//!
//! [`CreativeWritingModel`] embeds token ids with a domain-sized vocabulary
//! and [`CreativeWritingAttention`] performs real scaled dot-product
//! attention (see [`trustformers_core::layers::GroupedQueryAttention`]),
//! followed by a SiLU-gated [`CreativeWritingMLP`]. `generate`/`generate_story`/
//! `generate_dialogue`/`generate_poetry` drive real autoregressive decoding
//! through a byte-level tokenizer rather than echoing the prompt.
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use trustformers_models::creative_writing_specialized::{CreativeWritingConfig, CreativeWritingForCausalLM};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a fantasy writing model
//! // (`fantasy_7b()` is a genuine ~7B-parameter preset; this example is marked
//! // `no_run` since constructing it allocates full-size weights.)
//! let config = CreativeWritingConfig::fantasy_7b();
//! let model = CreativeWritingForCausalLM::new(config)?;
//!
//! // Generate a story beginning
//! let prompt = "In a world where magic has been forgotten, a young librarian discovers";
//! let story = model.generate(prompt, 500)?;
//! # let _ = story;
//! # Ok(())
//! # }
//! ```

use crate::common_patterns::GenerationConfig;
use crate::generation_utils::GenerationUtils;
use anyhow::Result;
use scirs2_core::random::*;
use serde::{Deserialize, Serialize};
use trustformers_core::errors::{not_implemented, tensor_op_error, Result as CoreResult};
use trustformers_core::layers::{Embedding, FlashAttentionInput, GroupedQueryAttention, Linear};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Config, Layer, Model};

// Import actual implementations from trustformers_core
use trustformers_core::layers::RMSNorm;

/// First token id used for byte-level tokens; ids below this are reserved
/// (`bos_token_id`/`eos_token_id`/padding). See [`CreativeWritingForCausalLM::generate`].
const BYTE_TOKEN_OFFSET: u32 = 8;
/// One token per possible byte value.
const BYTE_VOCAB_SIZE: u32 = 256;
/// Hard safety cap on total sequence length regardless of the caller's
/// `max_new_tokens`.
const MAX_SEQUENCE_LENGTH: usize = 4096;

fn min_vocab_size() -> usize {
    (BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE) as usize
}

/// Creative writing genre specialization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WritingGenre {
    /// General creative writing
    General,
    /// Literary fiction
    LiteraryFiction,
    /// Science fiction
    ScienceFiction,
    /// Fantasy
    Fantasy,
    /// Mystery and thriller
    Mystery,
    /// Romance
    Romance,
    /// Historical fiction
    Historical,
    /// Horror
    Horror,
    /// Poetry
    Poetry,
    /// Screenwriting
    Screenwriting,
    /// Creative non-fiction
    CreativeNonfiction,
    /// Children's literature
    Childrens,
    /// Young adult fiction
    YoungAdult,
}

/// Writing style preferences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WritingStyle {
    /// Descriptive and atmospheric
    Descriptive,
    /// Dialogue-heavy
    DialogueDriven,
    /// Action-oriented
    ActionPacked,
    /// Introspective and psychological
    Psychological,
    /// Minimalist style
    Minimalist,
    /// Ornate and elaborate
    Ornate,
    /// Stream of consciousness
    StreamOfConsciousness,
    /// Experimental
    Experimental,
}

/// Narrative perspective
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NarrativePerspective {
    /// First person singular (I)
    FirstPerson,
    /// Second person (You)
    SecondPerson,
    /// Third person limited
    ThirdPersonLimited,
    /// Third person omniscient
    ThirdPersonOmniscient,
    /// Multiple perspectives
    MultipleViewpoints,
}

/// Literary devices and techniques
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LiteraryDevice {
    /// Metaphor and simile
    Metaphor,
    /// Symbolism
    Symbolism,
    /// Foreshadowing
    Foreshadowing,
    /// Irony
    Irony,
    /// Alliteration
    Alliteration,
    /// Imagery
    Imagery,
    /// Dialogue
    Dialogue,
    /// Flashback
    Flashback,
}

/// Creative writing model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeWritingConfig {
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

    // Creative writing specific fields
    pub genre: WritingGenre,
    pub writing_style: WritingStyle,
    pub narrative_perspective: NarrativePerspective,
    pub literary_devices: Vec<LiteraryDevice>,
    pub character_development: bool,
    pub dialogue_enhancement: bool,
    pub world_building: bool,
    pub plot_structure_awareness: bool,
    pub creative_constraints: bool,
    pub style_adaptation: bool,
    pub emotional_intelligence: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub scaling_factor: f32,
}

/// Special tokens for creative writing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeWritingSpecialTokens {
    pub character_start: String,
    pub character_end: String,
    pub dialogue_start: String,
    pub dialogue_end: String,
    pub setting_start: String,
    pub setting_end: String,
    pub action_start: String,
    pub action_end: String,
    pub thought_start: String,
    pub thought_end: String,
    pub flashback_start: String,
    pub flashback_end: String,
    pub scene_break: String,
    pub chapter_break: String,
    pub narrator_voice: String,
    pub author_note: String,
}

impl Default for CreativeWritingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 35000, // Creative vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            hidden_act: "silu".to_string(),
            max_position_embeddings: 16384, // Medium context for stories
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
            model_type: "creative-writing".to_string(),
            genre: WritingGenre::General,
            writing_style: WritingStyle::Descriptive,
            narrative_perspective: NarrativePerspective::ThirdPersonLimited,
            literary_devices: vec![
                LiteraryDevice::Metaphor,
                LiteraryDevice::Imagery,
                LiteraryDevice::Dialogue,
            ],
            character_development: true,
            dialogue_enhancement: true,
            world_building: true,
            plot_structure_awareness: true,
            creative_constraints: false,
            style_adaptation: true,
            emotional_intelligence: true,
        }
    }
}

impl Config for CreativeWritingConfig {
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
        "CreativeWriting"
    }
}

impl CreativeWritingConfig {
    /// General creative writing model (7B parameters)
    pub fn creative_writing_7b() -> Self {
        Self {
            vocab_size: 35000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 16384,
            genre: WritingGenre::General,
            writing_style: WritingStyle::Descriptive,
            model_type: "creative-general".to_string(),
            ..Self::default()
        }
    }

    /// Fantasy writing model (7B parameters)
    pub fn fantasy_7b() -> Self {
        Self {
            vocab_size: 40000, // Expanded for fantasy terminology
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 20480, // Longer for epic fantasy
            genre: WritingGenre::Fantasy,
            writing_style: WritingStyle::Descriptive,
            world_building: true,
            character_development: true,
            model_type: "creative-fantasy".to_string(),
            ..Self::default()
        }
    }

    /// Science fiction writing model (7B parameters)
    pub fn scifi_7b() -> Self {
        Self {
            vocab_size: 38000, // Sci-fi terminology
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 16384,
            genre: WritingGenre::ScienceFiction,
            writing_style: WritingStyle::ActionPacked,
            world_building: true,
            plot_structure_awareness: true,
            model_type: "creative-scifi".to_string(),
            ..Self::default()
        }
    }

    /// Mystery and thriller writing model (7B parameters)
    pub fn mystery_7b() -> Self {
        Self {
            vocab_size: 32000, // Mystery-focused vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 16384,
            genre: WritingGenre::Mystery,
            writing_style: WritingStyle::Psychological,
            literary_devices: vec![
                LiteraryDevice::Foreshadowing,
                LiteraryDevice::Irony,
                LiteraryDevice::Dialogue,
            ],
            plot_structure_awareness: true,
            model_type: "creative-mystery".to_string(),
            ..Self::default()
        }
    }

    /// Romance writing model (7B parameters)
    pub fn romance_7b() -> Self {
        Self {
            vocab_size: 30000, // Romance-focused vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 16384,
            genre: WritingGenre::Romance,
            writing_style: WritingStyle::DialogueDriven,
            character_development: true,
            dialogue_enhancement: true,
            emotional_intelligence: true,
            model_type: "creative-romance".to_string(),
            ..Self::default()
        }
    }

    /// Poetry writing model (7B parameters)
    pub fn poetry_7b() -> Self {
        Self {
            vocab_size: 25000, // Poetic vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 4096, // Shorter for poems
            genre: WritingGenre::Poetry,
            writing_style: WritingStyle::Ornate,
            literary_devices: vec![
                LiteraryDevice::Metaphor,
                LiteraryDevice::Symbolism,
                LiteraryDevice::Alliteration,
                LiteraryDevice::Imagery,
            ],
            creative_constraints: true,
            style_adaptation: true,
            model_type: "creative-poetry".to_string(),
            ..Self::default()
        }
    }

    /// Screenwriting model (7B parameters)
    pub fn screenwriting_7b() -> Self {
        Self {
            vocab_size: 28000, // Screenplay vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 8192, // Script length
            genre: WritingGenre::Screenwriting,
            writing_style: WritingStyle::DialogueDriven,
            dialogue_enhancement: true,
            plot_structure_awareness: true,
            creative_constraints: true,
            model_type: "creative-screenplay".to_string(),
            ..Self::default()
        }
    }

    /// Children's literature model (7B parameters)
    pub fn childrens_7b() -> Self {
        Self {
            vocab_size: 20000, // Simplified vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 8192, // Shorter stories
            genre: WritingGenre::Childrens,
            writing_style: WritingStyle::Descriptive,
            character_development: true,
            world_building: true,
            emotional_intelligence: true,
            model_type: "creative-childrens".to_string(),
            ..Self::default()
        }
    }

    /// Literary fiction model (7B parameters)
    pub fn literary_7b() -> Self {
        Self {
            vocab_size: 45000, // Rich literary vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 20480, // Long literary works
            genre: WritingGenre::LiteraryFiction,
            writing_style: WritingStyle::Ornate,
            literary_devices: vec![
                LiteraryDevice::Symbolism,
                LiteraryDevice::Metaphor,
                LiteraryDevice::Imagery,
                LiteraryDevice::Irony,
            ],
            character_development: true,
            style_adaptation: true,
            emotional_intelligence: true,
            model_type: "creative-literary".to_string(),
            ..Self::default()
        }
    }

    /// Large creative writing model (13B parameters)
    pub fn creative_writing_13b() -> Self {
        Self {
            vocab_size: 50000, // Very large vocabulary
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: Some(8),
            max_position_embeddings: 32768, // Long context
            genre: WritingGenre::General,
            model_type: "creative-large".to_string(),
            ..Self::default()
        }
    }

    /// Get special tokens for creative writing
    pub fn get_special_tokens(&self) -> CreativeWritingSpecialTokens {
        CreativeWritingSpecialTokens {
            character_start: "<character>".to_string(),
            character_end: "</character>".to_string(),
            dialogue_start: "<dialogue>".to_string(),
            dialogue_end: "</dialogue>".to_string(),
            setting_start: "<setting>".to_string(),
            setting_end: "</setting>".to_string(),
            action_start: "<action>".to_string(),
            action_end: "</action>".to_string(),
            thought_start: "<thought>".to_string(),
            thought_end: "</thought>".to_string(),
            flashback_start: "<flashback>".to_string(),
            flashback_end: "</flashback>".to_string(),
            scene_break: "---".to_string(),
            chapter_break: "***".to_string(),
            narrator_voice: "<narrator>".to_string(),
            author_note: "<note>".to_string(),
        }
    }

    /// Create configuration from genre and size
    pub fn from_genre_and_size(genre: WritingGenre, size: &str) -> Option<Self> {
        match (genre, size) {
            (WritingGenre::General, "7b") => Some(Self::creative_writing_7b()),
            (WritingGenre::General, "13b") => Some(Self::creative_writing_13b()),
            (WritingGenre::Fantasy, "7b") => Some(Self::fantasy_7b()),
            (WritingGenre::ScienceFiction, "7b") => Some(Self::scifi_7b()),
            (WritingGenre::Mystery, "7b") => Some(Self::mystery_7b()),
            (WritingGenre::Romance, "7b") => Some(Self::romance_7b()),
            (WritingGenre::Poetry, "7b") => Some(Self::poetry_7b()),
            (WritingGenre::Screenwriting, "7b") => Some(Self::screenwriting_7b()),
            (WritingGenre::Childrens, "7b") => Some(Self::childrens_7b()),
            (WritingGenre::LiteraryFiction, "7b") => Some(Self::literary_7b()),
            _ => None,
        }
    }
}

/// Creative writing model implementation
pub struct CreativeWritingModel {
    config: CreativeWritingConfig,
    embeddings: Embedding,
    layers: Vec<CreativeWritingLayer>,
    norm: RMSNorm,
}

impl Model for CreativeWritingModel {
    type Config = CreativeWritingConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        // Convert input to token IDs if needed
        let token_ids: Vec<u32> = input.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        Layer::forward(self, token_ids)
    }

    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> CoreResult<()> {
        checked_unimplemented_load(reader, "CreativeWritingModel")
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let embed_params = self.embeddings.parameter_count();
        let layers_params: usize = self.layers.iter().map(|layer| layer.parameter_count()).sum();
        let norm_params = self.norm.parameter_count();

        embed_params + layers_params + norm_params
    }
}

/// Creative writing transformer layer
pub struct CreativeWritingLayer {
    self_attention: CreativeWritingAttention,
    feed_forward: CreativeWritingMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

/// Creative writing attention mechanism.
///
/// Backed by [`GroupedQueryAttention`], which performs real scaled
/// dot-product attention (`softmax(QK^T / sqrt(head_dim)) V`) with causal
/// masking and, when `num_key_value_heads < num_attention_heads`,
/// grouped-query key/value sharing. The previous implementation computed
/// `q_proj(x) + v_proj(x)` and discarded the key projection entirely; this
/// one actually uses it to compute attention scores.
pub struct CreativeWritingAttention {
    attn: GroupedQueryAttention,
}

/// Creative writing MLP
pub struct CreativeWritingMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

/// Creative writing model for causal language modeling
pub struct CreativeWritingForCausalLM {
    model: CreativeWritingModel,
    lm_head: Linear,
    config: CreativeWritingConfig,
}

impl CreativeWritingForCausalLM {
    pub fn new(config: CreativeWritingConfig) -> Result<Self> {
        config.validate()?;

        // Create the base model
        let model = CreativeWritingModel::new(config.clone())?;

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
    /// [`CreativeWritingForCausalLM::generate_with_seed`].
    ///
    /// This used to return `Ok(format!("[Creative Generation] {}", enhanced_prompt))`,
    /// a formatted echo of the (lightly wrapped) input, built from a
    /// `GenerationConfig` that was constructed and then never used. It now
    /// runs real autoregressive decoding: a byte-level tokenizer (see the
    /// module-level docs) encodes the prompt, the real transformer forward
    /// pass produces logits at each step, and [`GenerationUtils`] samples the
    /// next token. With random (untrained) weights the output is not fluent
    /// prose, but it is genuine model output that varies with the seed,
    /// prompt, and weights.
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
        let gen_config = GenerationConfig {
            max_new_tokens: max_length,
            temperature: 0.8, // Slightly creative
            top_p: 0.9,
            do_sample: true,
            ..Default::default()
        };

        let enhanced_prompt = self.enhance_prompt_for_creativity(input)?;
        self.generate_with_config(&enhanced_prompt, &gen_config, rng)
    }

    pub fn generate_story(&self, prompt: &str, max_length: usize) -> Result<String> {
        // Enhance the prompt for story generation
        let story_prompt = self.format_story_prompt(prompt)?;

        // Generate with story-specific parameters
        let gen_config = GenerationConfig {
            max_new_tokens: max_length,
            temperature: 0.9, // More creative for stories
            top_p: 0.95,
            do_sample: true,
            repetition_penalty: 1.1,
            ..Default::default()
        };

        // Generate the story
        let mut rng = thread_rng();
        self.generate_with_config(&story_prompt, &gen_config, &mut rng)
    }

    pub fn continue_story(&self, story_beginning: &str, target_length: usize) -> Result<String> {
        // Analyze the existing story context
        let _context = self.analyze_story_context(story_beginning)?;

        // Generate continuation with consistent style
        let continuation_prompt = format!("{} [CONTINUE]", story_beginning);
        let gen_config = GenerationConfig {
            max_new_tokens: target_length,
            temperature: 0.8,
            top_p: 0.9,
            do_sample: true,
            repetition_penalty: 1.2, // Avoid repetition
            ..Default::default()
        };

        let mut rng = thread_rng();
        self.generate_with_config(&continuation_prompt, &gen_config, &mut rng)
    }

    pub fn generate_dialogue(&self, context: &str, character_names: &[&str]) -> Result<String> {
        // Format dialogue prompt with character names
        let dialogue_prompt = self.format_dialogue_prompt(context, character_names)?;

        // Generate with dialogue-specific parameters
        let gen_config = GenerationConfig {
            max_new_tokens: 500,
            temperature: 0.85,
            top_p: 0.92,
            do_sample: true,
            repetition_penalty: 1.15,
            ..Default::default()
        };

        let mut rng = thread_rng();
        self.generate_with_config(&dialogue_prompt, &gen_config, &mut rng)
    }

    pub fn analyze_writing_style(&self, text: &str) -> Result<StyleAnalysis> {
        // Analyze various aspects of the writing style
        let word_count = text.split_whitespace().count();
        let sentence_count = text.split(['.', '!', '?']).count();
        let avg_sentence_length =
            if sentence_count > 0 { word_count as f32 / sentence_count as f32 } else { 0.0 };

        // Detect genre based on keywords and style
        let detected_genre = self.detect_genre(text)?;

        // Analyze other style features
        let style_analysis = StyleAnalysis {
            detected_genre,
            writing_style: self.detect_writing_style(text)?,
            narrative_perspective: self.detect_narrative_perspective(text)?,
            literary_devices_used: self.detect_literary_devices(text)?,
            readability_score: self.calculate_readability_score(text)?,
            vocabulary_richness: self.calculate_vocabulary_richness(text)?,
            sentence_complexity: avg_sentence_length,
            emotional_tone: self.detect_emotional_tone(text)?,
            character_development_score: self.analyze_character_development(text)?,
            dialogue_quality: self.analyze_dialogue_quality(text)?,
        };

        Ok(style_analysis)
    }

    pub fn suggest_improvements(&self, text: &str) -> Result<Vec<WritingImprovement>> {
        let mut improvements = Vec::new();

        // Analyze the text for common improvement areas
        let style_analysis = self.analyze_writing_style(text)?;

        // Suggest improvements based on analysis
        if style_analysis.readability_score < 0.5 {
            improvements.push(WritingImprovement {
                suggestion_type: ImprovementType::SentenceStructure,
                location: "Throughout text".to_string(),
                original_text: "Complex sentence structures".to_string(),
                suggested_text: "Consider breaking down complex sentences for better readability"
                    .to_string(),
                explanation: "Shorter sentences improve readability and flow".to_string(),
                confidence: 0.8,
            });
        }

        if style_analysis.vocabulary_richness < 0.6 {
            improvements.push(WritingImprovement {
                suggestion_type: ImprovementType::WordChoice,
                location: "Throughout text".to_string(),
                original_text: "Limited vocabulary".to_string(),
                suggested_text: "Consider using more varied and descriptive vocabulary".to_string(),
                explanation: "Rich vocabulary enhances reader engagement".to_string(),
                confidence: 0.7,
            });
        }

        Ok(improvements)
    }

    pub fn generate_poetry(&self, style: PoetryStyle, topic: &str) -> Result<String> {
        // Create poetry-specific prompt
        let poetry_prompt = self.format_poetry_prompt(style.clone(), topic)?;

        // Generate with poetry-specific parameters
        let gen_config = GenerationConfig {
            max_new_tokens: match style {
                PoetryStyle::Haiku => 30,
                PoetryStyle::Limerick => 80,
                PoetryStyle::Sonnet => 200,
                _ => 150,
            },
            temperature: 0.9, // High creativity for poetry
            top_p: 0.95,
            do_sample: true,
            repetition_penalty: 1.3, // Avoid repetition in poetry
            ..Default::default()
        };

        let mut rng = thread_rng();
        self.generate_with_config(&poetry_prompt, &gen_config, &mut rng)
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

// Implementation of CreativeWritingModel
impl CreativeWritingModel {
    pub fn new(config: CreativeWritingConfig) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(CreativeWritingLayer::new(&config)?);
        }

        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            config,
            embeddings: embed_tokens,
            layers,
            norm,
        })
    }
}

// Implementation of CreativeWritingLayer
impl CreativeWritingLayer {
    pub fn new(config: &CreativeWritingConfig) -> Result<Self> {
        let self_attention = CreativeWritingAttention::new(config)?;
        let feed_forward = CreativeWritingMLP::new(config)?;
        let input_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            self_attention,
            feed_forward,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.self_attention.parameter_count()
            + self.feed_forward.parameter_count()
            + self.input_layernorm.parameter_count()
            + self.post_attention_layernorm.parameter_count()
    }
}

// Implementation of CreativeWritingAttention
impl CreativeWritingAttention {
    pub fn new(config: &CreativeWritingConfig) -> Result<Self> {
        let num_kv_heads = config.num_key_value_heads.unwrap_or(config.num_attention_heads);
        let mut attn = GroupedQueryAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            num_kv_heads,
            // CreativeWritingConfig does not expose an attention-dropout
            // knob; 0.0 also keeps inference deterministic.
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

// Implementation of CreativeWritingMLP
impl CreativeWritingMLP {
    pub fn new(config: &CreativeWritingConfig) -> Result<Self> {
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

    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
    }
}

// Layer trait implementations
impl Layer for CreativeWritingModel {
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        // Convert token IDs to embeddings
        let mut hidden_states = self.embeddings.forward(input)?;

        // Pass through all layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        // Apply final normalization
        let output = self.norm.forward(hidden_states)?;
        Ok(output)
    }
}

impl Layer for CreativeWritingLayer {
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

impl CreativeWritingAttention {
    pub fn parameter_count(&self) -> usize {
        self.attn.parameter_count()
    }
}

impl Layer for CreativeWritingAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        self.attn.forward(FlashAttentionInput {
            hidden_states: input,
            attention_mask: None,
        })
    }
}

impl Layer for CreativeWritingMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        // SiLU activation MLP
        let gate_output = self.gate_proj.forward(input.clone())?;
        let up_output = self.up_proj.forward(input)?;

        // Apply SiLU activation to gate output
        let gate_activated = match &gate_output {
            Tensor::F32(arr) => {
                let activated = arr.mapv(|x| x / (1.0 + (-x).exp())); // SiLU activation
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

// Model trait implementation for CreativeWritingForCausalLM
impl Model for CreativeWritingForCausalLM {
    type Config = CreativeWritingConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        self.forward_logits(&input)
    }

    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> CoreResult<()> {
        checked_unimplemented_load(reader, "CreativeWritingForCausalLM")
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

// Helper methods for CreativeWritingForCausalLM
impl CreativeWritingForCausalLM {
    fn enhance_prompt_for_creativity(&self, prompt: &str) -> Result<String> {
        let special_tokens = self.config.get_special_tokens();
        let enhanced = format!(
            "{}{}{}",
            special_tokens.character_start, prompt, special_tokens.character_end
        );
        Ok(enhanced)
    }

    fn format_story_prompt(&self, prompt: &str) -> Result<String> {
        let special_tokens = self.config.get_special_tokens();
        let formatted = format!(
            "{}{}{} {}",
            special_tokens.setting_start, prompt, special_tokens.setting_end, "Once upon a time"
        );
        Ok(formatted)
    }

    fn format_dialogue_prompt(&self, context: &str, character_names: &[&str]) -> Result<String> {
        let special_tokens = self.config.get_special_tokens();
        let characters = character_names.join(", ");
        let formatted = format!(
            "{}{}{} {}Characters: {}{}",
            special_tokens.setting_start,
            context,
            special_tokens.setting_end,
            special_tokens.dialogue_start,
            characters,
            special_tokens.dialogue_end
        );
        Ok(formatted)
    }

    fn format_poetry_prompt(&self, style: PoetryStyle, topic: &str) -> Result<String> {
        let style_instruction = match style {
            PoetryStyle::Haiku => "Write a haiku (5-7-5 syllables)",
            PoetryStyle::Sonnet => "Write a sonnet (14 lines, ABAB CDCD EFEF GG)",
            PoetryStyle::Limerick => "Write a limerick (AABBA rhyme scheme)",
            PoetryStyle::FreeVerse => "Write a free verse poem",
            _ => "Write a poem",
        };

        let formatted = format!("{} about: {}", style_instruction, topic);
        Ok(formatted)
    }

    fn analyze_story_context(&self, story: &str) -> Result<String> {
        // Analyze the story context for continuation
        let word_count = story.split_whitespace().count();
        let context = if word_count > 50 {
            "Long narrative context"
        } else {
            "Short narrative context"
        };
        Ok(context.to_string())
    }

    fn detect_genre(&self, text: &str) -> Result<WritingGenre> {
        // Simple genre detection based on keywords
        let text_lower = text.to_lowercase();
        if text_lower.contains("magic")
            || text_lower.contains("dragon")
            || text_lower.contains("wizard")
        {
            Ok(WritingGenre::Fantasy)
        } else if text_lower.contains("space")
            || text_lower.contains("robot")
            || text_lower.contains("future")
        {
            Ok(WritingGenre::ScienceFiction)
        } else if text_lower.contains("love")
            || text_lower.contains("heart")
            || text_lower.contains("romance")
        {
            Ok(WritingGenre::Romance)
        } else {
            Ok(WritingGenre::General)
        }
    }

    fn detect_writing_style(&self, text: &str) -> Result<WritingStyle> {
        let sentences = text.split(['.', '!', '?']).collect::<Vec<_>>();
        let avg_sentence_length = if !sentences.is_empty() {
            text.len() as f32 / sentences.len() as f32
        } else {
            0.0
        };

        if avg_sentence_length > 100.0 {
            Ok(WritingStyle::Ornate)
        } else if text.contains('"') {
            Ok(WritingStyle::DialogueDriven)
        } else if avg_sentence_length < 50.0 {
            Ok(WritingStyle::Minimalist)
        } else {
            Ok(WritingStyle::Descriptive)
        }
    }

    fn detect_narrative_perspective(&self, text: &str) -> Result<NarrativePerspective> {
        let text_lower = text.to_lowercase();
        if text_lower.contains(" i ") || text_lower.starts_with("i ") {
            Ok(NarrativePerspective::FirstPerson)
        } else if text_lower.contains(" you ") || text_lower.starts_with("you ") {
            Ok(NarrativePerspective::SecondPerson)
        } else {
            Ok(NarrativePerspective::ThirdPersonLimited)
        }
    }

    fn detect_literary_devices(&self, text: &str) -> Result<Vec<LiteraryDevice>> {
        let mut devices = Vec::new();

        if text.contains('"') {
            devices.push(LiteraryDevice::Dialogue);
        }
        if text.contains(" like ") || text.contains(" as ") {
            devices.push(LiteraryDevice::Metaphor);
        }
        if text.contains("seemed") || text.contains("appeared") {
            devices.push(LiteraryDevice::Imagery);
        }

        Ok(devices)
    }

    fn calculate_readability_score(&self, text: &str) -> Result<f32> {
        // Simple readability score based on sentence and word length
        let words = text.split_whitespace().count();
        let sentences = text.split(['.', '!', '?']).count();

        if sentences == 0 {
            return Ok(0.0);
        }

        let avg_sentence_length = words as f32 / sentences as f32;
        let score = 1.0 - (avg_sentence_length / 50.0).min(1.0);
        Ok(score.max(0.0))
    }

    fn calculate_vocabulary_richness(&self, text: &str) -> Result<f32> {
        // Calculate vocabulary richness (unique words / total words)
        let words: Vec<&str> = text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();

        if words.is_empty() {
            return Ok(0.0);
        }

        let richness = unique_words.len() as f32 / words.len() as f32;
        Ok(richness)
    }

    fn detect_emotional_tone(&self, text: &str) -> Result<EmotionalTone> {
        let text_lower = text.to_lowercase();
        if text_lower.contains("happy")
            || text_lower.contains("joy")
            || text_lower.contains("laugh")
        {
            Ok(EmotionalTone::Joyful)
        } else if text_lower.contains("sad")
            || text_lower.contains("cry")
            || text_lower.contains("tear")
        {
            Ok(EmotionalTone::Melancholic)
        } else if text_lower.contains("dark")
            || text_lower.contains("fear")
            || text_lower.contains("death")
        {
            Ok(EmotionalTone::Dark)
        } else if text_lower.contains("love")
            || text_lower.contains("heart")
            || text_lower.contains("kiss")
        {
            Ok(EmotionalTone::Romantic)
        } else {
            Ok(EmotionalTone::Neutral)
        }
    }

    fn analyze_character_development(&self, text: &str) -> Result<f32> {
        // Simple character development analysis
        let character_indicators = ["he", "she", "they", "character", "person"];
        let mut score = 0.0;

        for indicator in &character_indicators {
            if text.to_lowercase().contains(indicator) {
                score += 0.2;
            }
        }

        Ok(f32::min(score, 1.0))
    }

    fn analyze_dialogue_quality(&self, text: &str) -> Result<f32> {
        // Analyze dialogue quality based on presence and variety
        let quote_count = text.matches('"').count();
        let dialogue_score = if quote_count > 0 {
            (quote_count as f32 / text.len() as f32 * 100.0).min(1.0)
        } else {
            0.0
        };

        Ok(dialogue_score)
    }
}

/// Writing style analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleAnalysis {
    pub detected_genre: WritingGenre,
    pub writing_style: WritingStyle,
    pub narrative_perspective: NarrativePerspective,
    pub literary_devices_used: Vec<LiteraryDevice>,
    pub readability_score: f32,
    pub vocabulary_richness: f32,
    pub sentence_complexity: f32,
    pub emotional_tone: EmotionalTone,
    pub character_development_score: f32,
    pub dialogue_quality: f32,
}

/// Emotional tone analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmotionalTone {
    Joyful,
    Melancholic,
    Suspenseful,
    Romantic,
    Dark,
    Humorous,
    Nostalgic,
    Hopeful,
    Neutral,
}

/// Writing improvement suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WritingImprovement {
    pub suggestion_type: ImprovementType,
    pub location: String,
    pub original_text: String,
    pub suggested_text: String,
    pub explanation: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImprovementType {
    WordChoice,
    SentenceStructure,
    Dialogue,
    Pacing,
    CharacterDevelopment,
    PlotStructure,
    Imagery,
    Consistency,
}

/// Poetry generation styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoetryStyle {
    FreeVerse,
    Sonnet,
    Haiku,
    Limerick,
    Ballad,
    Acrostic,
    BlankVerse,
    Villanelle,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creative_writing_config() {
        let config = CreativeWritingConfig::creative_writing_7b();
        assert_eq!(config.genre, WritingGenre::General);
        assert_eq!(config.vocab_size, 35000);
        assert!(config.character_development);
    }

    #[test]
    fn test_fantasy_config() {
        let config = CreativeWritingConfig::fantasy_7b();
        assert_eq!(config.genre, WritingGenre::Fantasy);
        assert!(config.world_building);
        assert_eq!(config.max_position_embeddings, 20480);
    }

    #[test]
    fn test_poetry_config() {
        let config = CreativeWritingConfig::poetry_7b();
        assert_eq!(config.genre, WritingGenre::Poetry);
        assert!(config.creative_constraints);
        assert!(config.literary_devices.contains(&LiteraryDevice::Metaphor));
    }

    #[test]
    fn test_special_tokens() {
        let config = CreativeWritingConfig::creative_writing_7b();
        let tokens = config.get_special_tokens();
        assert_eq!(tokens.dialogue_start, "<dialogue>");
        assert_eq!(tokens.scene_break, "---");
    }

    #[test]
    fn test_genre_and_size_creation() {
        let config = CreativeWritingConfig::from_genre_and_size(WritingGenre::Mystery, "7b");
        assert!(config.is_some());
        let config = config.expect("operation failed");
        assert_eq!(config.genre, WritingGenre::Mystery);
    }

    #[test]
    fn test_config_validation() {
        let config = CreativeWritingConfig::romance_7b();
        assert!(config.validate().is_ok());
    }

    /// A tiny configuration for fast tests: real compute, not a 7B allocation.
    fn tiny_config() -> CreativeWritingConfig {
        CreativeWritingConfig {
            vocab_size: 512,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 128,
            rope_scaling: None,
            ..CreativeWritingConfig::default()
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

        let mut attn_a = CreativeWritingAttention::new(&config).expect("attention a");
        attn_a.set_projections(
            deterministic_linear(config.hidden_size, config.hidden_size, 1),
            deterministic_linear(config.hidden_size, kv_hidden, 2),
            deterministic_linear(config.hidden_size, kv_hidden, 3),
            deterministic_linear(config.hidden_size, config.hidden_size, 4),
        );

        let mut attn_b = CreativeWritingAttention::new(&config).expect("attention b");
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
        let attn = CreativeWritingAttention::new(&config).expect("attention");
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

    fn model() -> CreativeWritingForCausalLM {
        CreativeWritingForCausalLM::new(tiny_config()).expect("model construction")
    }

    #[test]
    fn generate_does_not_echo_the_prompt() {
        let m = model();
        let prompt = "In a world where magic has been forgotten";
        let output = m.generate_with_seed(prompt, 12, 7).expect("generate");
        assert!(
            !output.contains(prompt),
            "output must not echo the prompt back: {output:?}"
        );
        assert!(!output.starts_with("[Creative Generation]"));
    }

    #[test]
    fn generate_story_does_not_echo_the_prompt() {
        let m = model();
        let prompt = "a young librarian discovers a hidden door";
        let output = m.generate_story(prompt, 12).expect("generate_story");
        assert!(
            !output.contains(prompt),
            "output must not echo the prompt back: {output:?}"
        );
        assert!(!output.starts_with("[Generated]"));
    }

    #[test]
    fn generate_with_seed_is_reproducible() {
        let m = model();
        let a = m.generate_with_seed("Once upon a time", 16, 42).expect("a");
        let b = m.generate_with_seed("Once upon a time", 16, 42).expect("b");
        assert_eq!(a, b, "the same seed must produce the same output");
    }

    #[test]
    fn different_seeds_produce_different_output() {
        let m = model();
        let a = m.generate_with_seed("Once upon a time", 24, 1).expect("a");
        let b = m.generate_with_seed("Once upon a time", 24, 2).expect("b");
        assert_ne!(
            a, b,
            "different seeds must (almost certainly) sample different tokens"
        );
    }

    #[test]
    fn encode_decode_byte_roundtrip() {
        let m = model();
        let text = "The dragon circled the tower twice.";
        let ids = m.encode_text(text).expect("encode");
        assert_eq!(ids[0], m.config.bos_token_id);
        let decoded = CreativeWritingForCausalLM::decode_tokens(&ids[1..]);
        assert_eq!(decoded, text);
    }

    #[test]
    fn load_pretrained_reports_an_honest_error_instead_of_fake_success() {
        let config = tiny_config();
        let mut model = CreativeWritingForCausalLM::new(config).expect("model");
        let mut reader = std::io::Cursor::new(vec![0u8; 4096]);
        let result = Model::load_pretrained(&mut model, &mut reader);
        assert!(
            result.is_err(),
            "load_pretrained must not report success when no weights were actually bound"
        );
    }
}
