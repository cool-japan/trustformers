//! # Legal and Medical Domain-Specialized Models
//!
//! This module provides specialized model configurations and implementations
//! optimized for legal documents, medical literature, and healthcare applications.
//!
//! ## Features
//!
//! - **Extended Context**: Support for very long documents (64K-128K tokens)
//! - **Domain Vocabularies**: Specialized terminology for legal and medical fields
//! - **Document Structure**: Understanding of legal/medical document formats
//! - **Regulatory Compliance**: Awareness of privacy and regulatory requirements
//! - **Citation Support**: Legal case citations and medical reference handling
//! - **Multi-jurisdictional**: Support for different legal systems and medical standards
//!
//! ## Legal Domain Features
//!
//! ### Document Types
//! - Contracts and agreements
//! - Court filings and pleadings
//! - Legal briefs and opinions
//! - Regulatory documents
//! - Patent applications
//!
//! ### Legal Systems
//! - Common law (US, UK, etc.)
//! - Civil law (Continental Europe)
//! - International law
//! - Regulatory frameworks
//!
//! ## Medical Domain Features
//!
//! ### Document Types
//! - Medical records and charts
//! - Research papers and studies
//! - Clinical trial reports
//! - Drug information and prescriptions
//! - Medical imaging reports
//!
//! ### Medical Specialties
//! - Internal medicine
//! - Surgery and procedures
//! - Radiology and imaging
//! - Pharmacology
//! - Public health
//!
//! ## Architecture
//!
//! The transformer backbone lives in this file: [`LegalMedicalModel`] embeds
//! token ids with a domain-sized vocabulary, [`LegalMedicalAttention`] performs
//! real scaled dot-product attention (see [`trustformers_core::layers::GroupedQueryAttention`]),
//! and [`LegalMedicalMLP`] applies a SiLU-gated feed-forward. The rest of the
//! domain-specific surface lives in submodules:
//!
//! - [`generation`]: byte-level tokenizer, autoregressive `generate`, and the
//!   confidentiality-aware attention mask.
//! - [`redaction`]: pattern-based PII redaction (`redact_sensitive_info`).
//! - [`analysis`]: document/citation/compliance heuristics.
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use trustformers_models::legal_medical_specialized::{LegalMedicalConfig, LegalMedicalForCausalLM};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a legal model
//! // (`legal_contract_7b()` / `medical_clinical_7b()` are genuine ~7B-parameter
//! // presets; this example is marked `no_run` since constructing them allocates
//! // full-size weights, twice over.)
//! let config = LegalMedicalConfig::legal_contract_7b();
//! let model = LegalMedicalForCausalLM::new(config)?;
//! # let _ = model;
//!
//! // Create a medical model
//! let config = LegalMedicalConfig::medical_clinical_7b();
//! let model = LegalMedicalForCausalLM::new(config)?;
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

mod analysis;
mod generation;
mod redaction;

pub use analysis::{
    Citation, CitationType, ComplianceReport, ComplianceViolation, DocumentAnalysis,
};
pub use redaction::RedactionReport;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use trustformers_core::errors::{not_implemented, Result as CoreResult};
use trustformers_core::layers::{
    Embedding, FlashAttentionInput, GroupedQueryAttention, Linear, RMSNorm,
};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Config, Layer, Model};

/// Domain specialization types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LegalMedicalDomain {
    /// General legal documents
    Legal,
    /// Contract law and commercial agreements
    LegalContract,
    /// Litigation and court documents
    LegalLitigation,
    /// Regulatory and compliance documents
    LegalRegulatory,
    /// Intellectual property law
    LegalIP,
    /// Criminal law
    LegalCriminal,
    /// General medical documents
    Medical,
    /// Clinical medicine and patient care
    MedicalClinical,
    /// Medical research and studies
    MedicalResearch,
    /// Pharmacology and drug information
    MedicalPharmacology,
    /// Medical imaging and radiology
    MedicalRadiology,
    /// Public health and epidemiology
    MedicalPublicHealth,
}

/// Legal system types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LegalSystem {
    /// Common law system (US, UK, etc.)
    CommonLaw,
    /// Civil law system (Continental Europe)
    CivilLaw,
    /// International law
    International,
    /// Administrative/regulatory law
    Administrative,
    /// Custom or mixed systems
    Mixed,
}

/// Medical standards and regulations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MedicalStandard {
    /// HIPAA (Health Insurance Portability and Accountability Act)
    HIPAA,
    /// FDA (Food and Drug Administration) standards
    FDA,
    /// European Medicines Agency standards
    EMA,
    /// World Health Organization standards
    WHO,
    /// International Conference on Harmonisation
    ICH,
    /// Good Clinical Practice
    GCP,
}

/// Privacy and compliance requirements
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PrivacyRequirement {
    /// HIPAA compliance for medical data
    HIPAA,
    /// GDPR compliance for EU data
    GDPR,
    /// Attorney-client privilege protection
    AttorneyClient,
    /// Medical confidentiality
    MedicalConfidentiality,
    /// Custom privacy requirements
    Custom(String),
}

/// Legal/Medical model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalMedicalConfig {
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

    // Domain-specific fields
    pub domain: LegalMedicalDomain,
    pub legal_system: Option<LegalSystem>,
    pub medical_standard: Option<MedicalStandard>,
    pub privacy_requirements: Vec<PrivacyRequirement>,
    pub citation_support: bool,
    pub case_law_understanding: bool,
    pub medical_terminology: bool,
    pub regulatory_compliance: bool,
    pub document_structure_awareness: bool,
    pub confidentiality_protection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub scaling_factor: f32,
}

/// Special tokens for legal and medical text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalMedicalSpecialTokens {
    pub citation_start: String,
    pub citation_end: String,
    pub case_reference: String,
    pub statute_reference: String,
    pub regulation_reference: String,
    pub patient_id_start: String,
    pub patient_id_end: String,
    pub medical_code_start: String,
    pub medical_code_end: String,
    pub prescription_start: String,
    pub prescription_end: String,
    pub confidential_start: String,
    pub confidential_end: String,
    pub redacted_placeholder: String,
}

impl Default for LegalMedicalConfig {
    fn default() -> Self {
        Self {
            vocab_size: 45000, // Large vocabulary for specialized terms
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            hidden_act: "silu".to_string(),
            max_position_embeddings: 65536, // Very long context for documents
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                scaling_type: "linear".to_string(),
                scaling_factor: 8.0,
            }),
            attention_bias: false,
            mlp_bias: false,
            model_type: "legal-medical".to_string(),
            domain: LegalMedicalDomain::Legal,
            legal_system: Some(LegalSystem::CommonLaw),
            medical_standard: None,
            privacy_requirements: vec![PrivacyRequirement::AttorneyClient],
            citation_support: true,
            case_law_understanding: true,
            medical_terminology: false,
            regulatory_compliance: true,
            document_structure_awareness: true,
            confidentiality_protection: true,
        }
    }
}

impl Config for LegalMedicalConfig {
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
        "LegalMedical"
    }
}

impl LegalMedicalConfig {
    /// General legal document model (7B parameters)
    pub fn legal_7b() -> Self {
        Self {
            vocab_size: 40000, // Legal terminology focused
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 65536, // Very long legal documents
            domain: LegalMedicalDomain::Legal,
            legal_system: Some(LegalSystem::CommonLaw),
            privacy_requirements: vec![PrivacyRequirement::AttorneyClient],
            case_law_understanding: true,
            model_type: "legal-general".to_string(),
            ..Self::default()
        }
    }

    /// Contract law specialized model (7B parameters)
    pub fn legal_contract_7b() -> Self {
        Self {
            vocab_size: 38000, // Contract-focused vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 32768, // Medium-long contracts
            domain: LegalMedicalDomain::LegalContract,
            legal_system: Some(LegalSystem::CommonLaw),
            privacy_requirements: vec![PrivacyRequirement::AttorneyClient],
            case_law_understanding: true,
            document_structure_awareness: true,
            model_type: "legal-contract".to_string(),
            ..Self::default()
        }
    }

    /// Litigation and court documents model (7B parameters)
    pub fn legal_litigation_7b() -> Self {
        Self {
            vocab_size: 42000, // Litigation vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 65536, // Long court documents
            domain: LegalMedicalDomain::LegalLitigation,
            legal_system: Some(LegalSystem::CommonLaw),
            privacy_requirements: vec![PrivacyRequirement::AttorneyClient],
            case_law_understanding: true,
            citation_support: true,
            model_type: "legal-litigation".to_string(),
            ..Self::default()
        }
    }

    /// Regulatory compliance model (7B parameters)
    pub fn legal_regulatory_7b() -> Self {
        Self {
            vocab_size: 45000, // Regulatory vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 65536, // Very long regulations
            domain: LegalMedicalDomain::LegalRegulatory,
            legal_system: Some(LegalSystem::Administrative),
            privacy_requirements: vec![PrivacyRequirement::GDPR],
            regulatory_compliance: true,
            document_structure_awareness: true,
            model_type: "legal-regulatory".to_string(),
            ..Self::default()
        }
    }

    /// General medical model (7B parameters)
    pub fn medical_7b() -> Self {
        Self {
            vocab_size: 45000, // Medical terminology
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 32768, // Long medical documents
            domain: LegalMedicalDomain::Medical,
            legal_system: None,
            medical_standard: Some(MedicalStandard::HIPAA),
            privacy_requirements: vec![
                PrivacyRequirement::HIPAA,
                PrivacyRequirement::MedicalConfidentiality,
            ],
            medical_terminology: true,
            regulatory_compliance: true,
            confidentiality_protection: true,
            model_type: "medical-general".to_string(),
            ..Self::default()
        }
    }

    /// Clinical medicine model (7B parameters)
    pub fn medical_clinical_7b() -> Self {
        Self {
            vocab_size: 48000, // Clinical vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 16384, // Medical records length
            domain: LegalMedicalDomain::MedicalClinical,
            medical_standard: Some(MedicalStandard::HIPAA),
            privacy_requirements: vec![
                PrivacyRequirement::HIPAA,
                PrivacyRequirement::MedicalConfidentiality,
            ],
            medical_terminology: true,
            regulatory_compliance: true,
            confidentiality_protection: true,
            document_structure_awareness: true,
            model_type: "medical-clinical".to_string(),
            ..Self::default()
        }
    }

    /// Medical research model (7B parameters)
    pub fn medical_research_7b() -> Self {
        Self {
            vocab_size: 50000, // Research vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 32768, // Research papers
            domain: LegalMedicalDomain::MedicalResearch,
            medical_standard: Some(MedicalStandard::GCP),
            privacy_requirements: vec![PrivacyRequirement::HIPAA],
            medical_terminology: true,
            citation_support: true,
            regulatory_compliance: true,
            model_type: "medical-research".to_string(),
            ..Self::default()
        }
    }

    /// Pharmacology model (7B parameters)
    pub fn medical_pharmacology_7b() -> Self {
        Self {
            vocab_size: 42000, // Drug and pharmacology vocabulary
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 16384, // Drug information length
            domain: LegalMedicalDomain::MedicalPharmacology,
            medical_standard: Some(MedicalStandard::FDA),
            privacy_requirements: vec![PrivacyRequirement::MedicalConfidentiality],
            medical_terminology: true,
            regulatory_compliance: true,
            model_type: "medical-pharmacology".to_string(),
            ..Self::default()
        }
    }

    /// Large legal/medical model (13B parameters)
    pub fn legal_medical_13b() -> Self {
        Self {
            vocab_size: 60000, // Very large vocabulary
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: Some(8),
            max_position_embeddings: 131072, // 128K context
            rope_scaling: Some(RopeScaling {
                scaling_type: "linear".to_string(),
                scaling_factor: 16.0,
            }),
            domain: LegalMedicalDomain::Legal,
            model_type: "legal-medical-large".to_string(),
            ..Self::default()
        }
    }

    /// Get special tokens for the model
    pub fn get_special_tokens(&self) -> LegalMedicalSpecialTokens {
        LegalMedicalSpecialTokens {
            citation_start: "<cite>".to_string(),
            citation_end: "</cite>".to_string(),
            case_reference: "<case>".to_string(),
            statute_reference: "<statute>".to_string(),
            regulation_reference: "<regulation>".to_string(),
            patient_id_start: "<patient>".to_string(),
            patient_id_end: "</patient>".to_string(),
            medical_code_start: "<medcode>".to_string(),
            medical_code_end: "</medcode>".to_string(),
            prescription_start: "<rx>".to_string(),
            prescription_end: "</rx>".to_string(),
            confidential_start: "<confidential>".to_string(),
            confidential_end: "</confidential>".to_string(),
            redacted_placeholder: "[REDACTED]".to_string(),
        }
    }

    /// Create configuration from domain and size
    pub fn from_domain_and_size(domain: LegalMedicalDomain, size: &str) -> Option<Self> {
        match (domain, size) {
            (LegalMedicalDomain::Legal, "7b") => Some(Self::legal_7b()),
            (LegalMedicalDomain::Legal, "13b") => Some(Self::legal_medical_13b()),
            (LegalMedicalDomain::LegalContract, "7b") => Some(Self::legal_contract_7b()),
            (LegalMedicalDomain::LegalLitigation, "7b") => Some(Self::legal_litigation_7b()),
            (LegalMedicalDomain::LegalRegulatory, "7b") => Some(Self::legal_regulatory_7b()),
            (LegalMedicalDomain::Medical, "7b") => Some(Self::medical_7b()),
            (LegalMedicalDomain::Medical, "13b") => Some(Self::legal_medical_13b()),
            (LegalMedicalDomain::MedicalClinical, "7b") => Some(Self::medical_clinical_7b()),
            (LegalMedicalDomain::MedicalResearch, "7b") => Some(Self::medical_research_7b()),
            (LegalMedicalDomain::MedicalPharmacology, "7b") => {
                Some(Self::medical_pharmacology_7b())
            },
            _ => None,
        }
    }
}

/// Legal/Medical model implementation
pub struct LegalMedicalModel {
    config: LegalMedicalConfig,
    embed_tokens: Embedding,
    layers: Vec<LegalMedicalLayer>,
    norm: RMSNorm,
}

impl LegalMedicalModel {
    pub fn new(config: LegalMedicalConfig) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LegalMedicalLayer::new(&config)?);
        }

        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass with an optional pre-softmax attention mask (see
    /// [`super::generation`]'s confidentiality mask). `Layer::forward`/`Model::forward`
    /// delegate here with `mask = None`.
    fn forward_with_mask(&self, input: Vec<u32>, mask: Option<&Tensor>) -> CoreResult<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input)?;
        for layer in &self.layers {
            hidden_states = layer.forward_with_mask(hidden_states, mask)?;
        }
        self.norm.forward(hidden_states)
    }
}

impl Model for LegalMedicalModel {
    type Config = LegalMedicalConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        let token_ids: Vec<u32> = input.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        self.forward_with_mask(token_ids, None)
    }

    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> CoreResult<()> {
        checked_unimplemented_load(reader, "LegalMedicalModel")
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

/// Legal/Medical transformer layer with domain-specific optimizations
pub struct LegalMedicalLayer {
    self_attention: LegalMedicalAttention,
    feed_forward: LegalMedicalMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

/// Legal/Medical attention mechanism with privacy protection.
///
/// Backed by [`GroupedQueryAttention`], which performs real scaled
/// dot-product attention (`softmax(QK^T / sqrt(head_dim)) V`) with causal
/// masking and, when `num_key_value_heads < num_attention_heads`,
/// grouped-query key/value sharing. Setting `num_key_value_heads ==
/// num_attention_heads` degenerates to standard multi-head attention, so
/// this one type serves both cases - unlike the previous `q + v` stand-in,
/// the key projection is not just allocated but actually determines the
/// output.
pub struct LegalMedicalAttention {
    attn: GroupedQueryAttention,
}

impl LegalMedicalAttention {
    pub fn new(config: &LegalMedicalConfig) -> Result<Self> {
        let num_kv_heads = config.num_key_value_heads.unwrap_or(config.num_attention_heads);
        let mut attn = GroupedQueryAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            num_kv_heads,
            // LegalMedicalConfig does not expose an attention-dropout knob;
            // 0.0 also keeps inference deterministic.
            0.0,
            config.attention_bias,
        )?;
        // These are decoder-only causal language models: generation must not
        // see future tokens.
        attn.set_causal(true);
        Ok(Self { attn })
    }

    pub fn parameter_count(&self) -> usize {
        self.attn.parameter_count()
    }

    /// Replace the query/key/value/output projections. Only used by tests to
    /// prove the key projection actually participates in the output (the
    /// `q + v` bug this replaces silently discarded it).
    #[cfg(test)]
    fn set_projections(&mut self, query: Linear, key: Linear, value: Linear, out_proj: Linear) {
        self.attn.set_projections(query, key, value, out_proj);
    }

    fn forward_with_mask(&self, input: Tensor, mask: Option<&Tensor>) -> CoreResult<Tensor> {
        self.attn.forward(FlashAttentionInput {
            hidden_states: input,
            attention_mask: mask.cloned(),
        })
    }
}

/// Legal/Medical MLP with compliance features
pub struct LegalMedicalMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

/// Legal/Medical model for causal language modeling
pub struct LegalMedicalForCausalLM {
    model: LegalMedicalModel,
    lm_head: Linear,
    config: LegalMedicalConfig,
}

impl LegalMedicalForCausalLM {
    pub fn new(config: LegalMedicalConfig) -> Result<Self> {
        config.validate()?;

        // Create the base model
        let model = LegalMedicalModel::new(config.clone())?;

        // Create the language modeling head
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);

        Ok(Self {
            model,
            lm_head,
            config,
        })
    }

    /// Real forward pass from token ids straight through to logits, with an
    /// optional pre-softmax attention mask. Used by [`generation`] to drive
    /// autoregressive decoding without going through the `Layer`/`Model`
    /// trait objects (which have no room for a mask parameter).
    fn forward_logits_with_mask(
        &self,
        input_ids: &[u32],
        mask: Option<&Tensor>,
    ) -> CoreResult<Tensor> {
        let hidden_states = self.model.forward_with_mask(input_ids.to_vec(), mask)?;
        self.lm_head.forward(hidden_states)
    }
}

// Implementation of LegalMedicalLayer
impl LegalMedicalLayer {
    pub fn new(config: &LegalMedicalConfig) -> Result<Self> {
        let self_attention = LegalMedicalAttention::new(config)?;
        let feed_forward = LegalMedicalMLP::new(config)?;
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

    fn forward_with_mask(&self, input: Tensor, mask: Option<&Tensor>) -> CoreResult<Tensor> {
        // Pre-norm architecture
        let normalized_input = self.input_layernorm.forward(input.clone())?;
        let attn_output = self.self_attention.forward_with_mask(normalized_input, mask)?;
        let residual1 = input.add(&attn_output)?;

        let normalized_residual = self.post_attention_layernorm.forward(residual1.clone())?;
        let mlp_output = self.feed_forward.forward(normalized_residual)?;
        residual1.add(&mlp_output)
    }
}

// Implementation of LegalMedicalMLP
impl LegalMedicalMLP {
    pub fn new(config: &LegalMedicalConfig) -> Result<Self> {
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
impl Layer for LegalMedicalModel {
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        self.forward_with_mask(input, None)
    }
}

impl Layer for LegalMedicalLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        self.forward_with_mask(input, None)
    }
}

impl Layer for LegalMedicalAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        self.forward_with_mask(input, None)
    }
}

impl Layer for LegalMedicalMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        // SiLU activation MLP with compliance tracking
        let gate_output = self.gate_proj.forward(input.clone())?;
        let up_output = self.up_proj.forward(input)?;

        // Apply SiLU activation
        let gate_activated = match &gate_output {
            Tensor::F32(arr) => {
                let activated = arr.mapv(|x| x / (1.0 + (-x).exp()));
                Tensor::F32(activated)
            },
            _ => {
                return Err(trustformers_core::errors::tensor_op_error(
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
                return Err(trustformers_core::errors::tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor types for element-wise multiplication",
                ))
            },
        };

        self.down_proj.forward(combined)
    }
}

// Model trait implementation for LegalMedicalForCausalLM
impl Model for LegalMedicalForCausalLM {
    type Config = LegalMedicalConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> CoreResult<Self::Output> {
        self.forward_logits_with_mask(&input, None)
    }

    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> CoreResult<()> {
        checked_unimplemented_load(reader, "LegalMedicalForCausalLM")
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
/// (entirely discarded) bytes became model weights.
///
/// The previous implementation wrote the buffer to a temp file, `println!`ed
/// a success message, deleted the file, and returned `Ok(())` without ever
/// touching a single layer's weights - a fabricated success. Real checkpoint
/// binding (matching tensor names to this architecture's layers) is not
/// implemented for this family; callers that need it should use
/// [`crate::weight_loading`]'s checkpoint utilities directly against a model
/// whose layer names are known, rather than this trait method.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legal_config_creation() {
        let config = LegalMedicalConfig::legal_7b();
        assert_eq!(config.domain, LegalMedicalDomain::Legal);
        assert_eq!(config.vocab_size, 40000);
        assert!(config.case_law_understanding);
    }

    #[test]
    fn test_medical_config_creation() {
        let config = LegalMedicalConfig::medical_7b();
        assert_eq!(config.domain, LegalMedicalDomain::Medical);
        assert!(config.medical_terminology);
        assert!(config.confidentiality_protection);
    }

    #[test]
    fn test_privacy_requirements() {
        let config = LegalMedicalConfig::medical_clinical_7b();
        assert!(config.privacy_requirements.contains(&PrivacyRequirement::HIPAA));
        assert!(config
            .privacy_requirements
            .contains(&PrivacyRequirement::MedicalConfidentiality));
    }

    #[test]
    fn test_special_tokens() {
        let config = LegalMedicalConfig::legal_7b();
        let tokens = config.get_special_tokens();
        assert_eq!(tokens.case_reference, "<case>");
        assert_eq!(tokens.confidential_start, "<confidential>");
    }

    #[test]
    fn test_config_validation() {
        let config = LegalMedicalConfig::legal_contract_7b();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_domain_and_size_creation() {
        let config =
            LegalMedicalConfig::from_domain_and_size(LegalMedicalDomain::LegalContract, "7b");
        assert!(config.is_some());
        let config = config.expect("operation failed");
        assert_eq!(config.domain, LegalMedicalDomain::LegalContract);
    }

    /// A tiny configuration for fast tests: real compute, not a 7B allocation.
    pub(crate) fn tiny_config() -> LegalMedicalConfig {
        LegalMedicalConfig {
            vocab_size: 512,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 128,
            rope_scaling: None,
            ..LegalMedicalConfig::default()
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
        // outputs, because a real key projection participates in the
        // attention scores. Under the old `let combined = q_arr + v_arr;`
        // code, `_k` was computed and discarded, so this test would have
        // failed (identical outputs) against the old implementation.
        let config = tiny_config();
        let head_dim = config.hidden_size / config.num_attention_heads;
        let kv_heads = config.num_key_value_heads.expect("configured");
        let kv_hidden = kv_heads * head_dim;

        let mut attn_a = LegalMedicalAttention::new(&config).expect("attention a");
        attn_a.set_projections(
            deterministic_linear(config.hidden_size, config.hidden_size, 1),
            deterministic_linear(config.hidden_size, kv_hidden, 2),
            deterministic_linear(config.hidden_size, kv_hidden, 3),
            deterministic_linear(config.hidden_size, config.hidden_size, 4),
        );

        let mut attn_b = LegalMedicalAttention::new(&config).expect("attention b");
        attn_b.set_projections(
            deterministic_linear(config.hidden_size, config.hidden_size, 1),
            // Different key projection only.
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
        assert_eq!(data_a.len(), data_b.len());
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
    fn attention_output_is_finite_and_shape_preserving() {
        let config = tiny_config();
        let attn = LegalMedicalAttention::new(&config).expect("attention");
        let seq_len = 6;
        let input = Tensor::randn(&[seq_len, config.hidden_size]).expect("input");
        let output = attn.forward(input).expect("forward");
        assert_eq!(output.shape(), vec![seq_len, config.hidden_size]);
        assert!(output.data().expect("data").iter().all(|x| x.is_finite()));
    }

    #[test]
    fn causal_attention_does_not_leak_future_tokens() {
        // With real causal SDPA, changing a later token must not change the
        // hidden state of an earlier position.
        let config = tiny_config();
        let attn = LegalMedicalAttention::new(&config).expect("attention");
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

    #[test]
    fn load_pretrained_reports_an_honest_error_instead_of_fake_success() {
        // Regression test for the fabricated-success bug: the old
        // implementation wrote the input to a temp file, printed a "success"
        // message, and returned `Ok(())` without loading a single weight.
        let config = tiny_config();
        let mut model = LegalMedicalForCausalLM::new(config).expect("model");
        let bytes = vec![0u8; 4096];
        let mut reader = std::io::Cursor::new(bytes);
        let result = Model::load_pretrained(&mut model, &mut reader);
        assert!(
            result.is_err(),
            "load_pretrained must not report success when no weights were actually bound"
        );
    }

    #[test]
    fn load_pretrained_rejects_empty_input() {
        let config = tiny_config();
        let mut model = LegalMedicalForCausalLM::new(config).expect("model");
        let mut reader = std::io::Cursor::new(Vec::<u8>::new());
        assert!(Model::load_pretrained(&mut model, &mut reader).is_err());
    }
}
