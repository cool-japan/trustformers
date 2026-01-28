//! # Mathematics-Specialized Models
//!
//! This module provides specialized model configurations and implementations
//! optimized for mathematical reasoning, symbolic computation, and formula understanding.
//!
//! ## Features
//!
//! - **Mathematical Notation**: Enhanced tokenization for mathematical symbols and LaTeX
//! - **Step-by-Step Reasoning**: Structured generation for mathematical proofs and solutions
//! - **Formula Understanding**: Specialized attention patterns for mathematical expressions
//! - **Multi-Modal Math**: Support for text, equations, and mathematical diagrams
//! - **Theorem Proving**: Integration with formal verification systems
//! - **Scientific Computing**: Optimized for scientific and engineering calculations
//!
//! ## Supported Model Families
//!
//! ### MathLlama Family
//! - Mathematical reasoning versions of LLaMA models
//! - Enhanced mathematical vocabulary and training
//! - Step-by-step solution generation
//!
//! ### PaLM-Math Family
//! - Google's PaLM models fine-tuned for mathematics
//! - Chain-of-thought reasoning capabilities
//! - Multi-step problem solving
//!
//! ### DeepSeek-Math Family
//! - DeepSeek models specialized for mathematical reasoning
//! - Formal proof generation capabilities
//! - Scientific computation optimization
//!
//! ### MAmmoTH Family
//! - Large-scale mathematical reasoning models
//! - Multi-domain mathematical understanding
//! - Competition-level problem solving
//!
//! ## Example Usage
//!
//! ```rust
//! use trustformers_models::math_specialized::{MathSpecializedConfig, MathSpecializedForCausalLM};
//!
//! // Create a mathematical reasoning model
//! let config = MathSpecializedConfig::math_llama_7b();
//! let model = MathSpecializedForCausalLM::new(config)?;
//!
//! // Solve a mathematical problem
//! let problem = "Find the derivative of f(x) = x^2 + 3x + 1";
//! let solution = model.solve_step_by_step(problem)?;
//! ```

use serde::{Deserialize, Serialize};
use trustformers_core::errors::{invalid_config, Result};
use trustformers_core::tensor::Tensor;
use trustformers_core::{Config, Layer, Model};

#[cfg(feature = "llama")]
use crate::llama::{LlamaConfig, LlamaModel};

/// Configuration for mathematics-specialized models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathSpecializedConfig {
    /// Base model configuration
    pub base_config: LlamaConfig,
    /// Mathematical vocabulary size (including symbols)
    pub math_vocab_size: Option<usize>,
    /// Whether to use step-by-step reasoning
    pub step_by_step_reasoning: bool,
    /// Support for mathematical notation
    pub math_notation_support: bool,
    /// LaTeX rendering capabilities
    pub latex_support: bool,
    /// Formula parsing and understanding
    pub formula_parsing: bool,
    /// Symbolic computation capabilities
    pub symbolic_computation: bool,
    /// Mathematical domains supported
    pub supported_domains: Vec<MathDomain>,
    /// Reasoning strategies
    pub reasoning_strategies: Vec<ReasoningStrategy>,
    /// Mathematical special tokens
    pub math_tokens: MathSpecialTokens,
    /// Context length optimized for math problems
    pub math_context_length: usize,
    /// Whether to use mathematical attention patterns
    pub math_attention_patterns: bool,
    /// Model variant type
    pub model_variant: MathModelVariant,
    /// Chain-of-thought configuration
    pub chain_of_thought: ChainOfThoughtConfig,
}

/// Mathematical domains supported by the model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MathDomain {
    /// Elementary mathematics
    Elementary,
    /// Algebra and pre-calculus
    Algebra,
    /// Calculus and analysis
    Calculus,
    /// Linear algebra
    LinearAlgebra,
    /// Discrete mathematics
    DiscreteMath,
    /// Statistics and probability
    Statistics,
    /// Geometry and topology
    Geometry,
    /// Number theory
    NumberTheory,
    /// Mathematical logic
    Logic,
    /// Applied mathematics
    Applied,
    /// Physics and engineering math
    Physics,
    /// Computer science mathematics
    ComputerScience,
    /// Competition mathematics
    Competition,
}

/// Reasoning strategies for mathematical problem solving
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReasoningStrategy {
    /// Step-by-step analytical reasoning
    StepByStep,
    /// Chain-of-thought reasoning
    ChainOfThought,
    /// Working backwards from conclusion
    BackwardReasoning,
    /// Case-by-case analysis
    CaseAnalysis,
    /// Proof by contradiction
    ProofByContradiction,
    /// Mathematical induction
    Induction,
    /// Constructive proof
    Constructive,
    /// Analogical reasoning
    Analogical,
    /// Visual/geometric reasoning
    Visual,
    /// Symbolic manipulation
    Symbolic,
}

/// Mathematical special tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathSpecialTokens {
    /// Step separator token
    pub step_separator: String,
    /// Solution start token
    pub solution_start: String,
    /// Solution end token
    pub solution_end: String,
    /// Equation start token
    pub equation_start: String,
    /// Equation end token
    pub equation_end: String,
    /// Proof start token
    pub proof_start: String,
    /// Proof end token
    pub proof_end: String,
    /// Therefore symbol
    pub therefore: String,
    /// Because symbol
    pub because: String,
    /// QED symbol
    pub qed: String,
}

/// Chain-of-thought configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainOfThoughtConfig {
    /// Enable chain-of-thought reasoning
    pub enabled: bool,
    /// Maximum reasoning steps
    pub max_steps: usize,
    /// Step verification enabled
    pub step_verification: bool,
    /// Confidence scoring for each step
    pub confidence_scoring: bool,
    /// Backtracking on errors
    pub backtrack_on_error: bool,
}

/// Mathematical model variant types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MathModelVariant {
    /// Mathematical reasoning version of LLaMA
    MathLlama,
    /// DeepSeek-Math variant
    DeepSeekMath,
    /// PaLM-Math variant
    PalmMath,
    /// MAmmoTH variant
    Mammoth,
    /// Minerva variant (Google)
    Minerva,
    /// Mathematical Qwen variant
    QwenMath,
    /// CodeT5-Math variant
    CodeT5Math,
}

impl Default for MathSpecialTokens {
    fn default() -> Self {
        Self {
            step_separator: "<step>".to_string(),
            solution_start: "<solution>".to_string(),
            solution_end: "</solution>".to_string(),
            equation_start: "<eq>".to_string(),
            equation_end: "</eq>".to_string(),
            proof_start: "<proof>".to_string(),
            proof_end: "</proof>".to_string(),
            therefore: "∴".to_string(),
            because: "∵".to_string(),
            qed: "□".to_string(),
        }
    }
}

impl Default for ChainOfThoughtConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_steps: 20,
            step_verification: true,
            confidence_scoring: true,
            backtrack_on_error: true,
        }
    }
}

impl Default for MathSpecializedConfig {
    fn default() -> Self {
        Self {
            base_config: LlamaConfig::default(),
            math_vocab_size: None,
            step_by_step_reasoning: true,
            math_notation_support: true,
            latex_support: true,
            formula_parsing: true,
            symbolic_computation: true,
            supported_domains: vec![
                MathDomain::Elementary,
                MathDomain::Algebra,
                MathDomain::Calculus,
                MathDomain::LinearAlgebra,
                MathDomain::Statistics,
                MathDomain::Geometry,
            ],
            reasoning_strategies: vec![
                ReasoningStrategy::StepByStep,
                ReasoningStrategy::ChainOfThought,
                ReasoningStrategy::CaseAnalysis,
                ReasoningStrategy::Symbolic,
            ],
            math_tokens: MathSpecialTokens::default(),
            math_context_length: 8192,
            math_attention_patterns: true,
            model_variant: MathModelVariant::MathLlama,
            chain_of_thought: ChainOfThoughtConfig::default(),
        }
    }
}

impl MathSpecializedConfig {
    /// Math LLaMA 7B configuration
    pub fn math_llama_7b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32000,
                hidden_size: 4096,
                intermediate_size: 11008,
                num_hidden_layers: 32,
                num_attention_heads: 32,
                max_position_embeddings: 8192,
                ..LlamaConfig::default()
            },
            math_vocab_size: Some(35000), // Extended vocabulary for math symbols
            math_context_length: 8192,
            model_variant: MathModelVariant::MathLlama,
            ..Self::default()
        }
    }

    /// Math LLaMA 13B configuration
    pub fn math_llama_13b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32000,
                hidden_size: 5120,
                intermediate_size: 13824,
                num_hidden_layers: 40,
                num_attention_heads: 40,
                max_position_embeddings: 8192,
                ..LlamaConfig::default()
            },
            math_vocab_size: Some(35000),
            math_context_length: 8192,
            model_variant: MathModelVariant::MathLlama,
            ..Self::default()
        }
    }

    /// Math LLaMA 70B configuration
    pub fn math_llama_70b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32000,
                hidden_size: 8192,
                intermediate_size: 28672,
                num_hidden_layers: 80,
                num_attention_heads: 64,
                num_key_value_heads: Some(8), // Grouped-query attention
                max_position_embeddings: 8192,
                ..LlamaConfig::default()
            },
            math_vocab_size: Some(35000),
            math_context_length: 8192,
            model_variant: MathModelVariant::MathLlama,
            ..Self::default()
        }
    }

    /// DeepSeek-Math 7B configuration
    pub fn deepseek_math_7b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32000,
                hidden_size: 4096,
                intermediate_size: 11008,
                num_hidden_layers: 32,
                num_attention_heads: 32,
                max_position_embeddings: 16384,
                ..LlamaConfig::default()
            },
            math_vocab_size: Some(34000),
            math_context_length: 16384,
            model_variant: MathModelVariant::DeepSeekMath,
            supported_domains: vec![
                MathDomain::Elementary,
                MathDomain::Algebra,
                MathDomain::Calculus,
                MathDomain::LinearAlgebra,
                MathDomain::Logic,
                MathDomain::Competition,
            ],
            ..Self::default()
        }
    }

    /// DeepSeek-Math 67B configuration
    pub fn deepseek_math_67b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32000,
                hidden_size: 8192,
                intermediate_size: 22016,
                num_hidden_layers: 95,
                num_attention_heads: 64,
                num_key_value_heads: Some(8),
                max_position_embeddings: 16384,
                ..LlamaConfig::default()
            },
            math_vocab_size: Some(34000),
            math_context_length: 16384,
            model_variant: MathModelVariant::DeepSeekMath,
            ..Self::default()
        }
    }

    /// Minerva 8B configuration (Google's mathematical reasoning model)
    pub fn minerva_8b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32000,
                hidden_size: 4096,
                intermediate_size: 16384,
                num_hidden_layers: 32,
                num_attention_heads: 32,
                max_position_embeddings: 2048,
                ..LlamaConfig::default()
            },
            math_vocab_size: Some(33000),
            math_context_length: 2048,
            model_variant: MathModelVariant::Minerva,
            latex_support: true,
            formula_parsing: true,
            supported_domains: vec![
                MathDomain::Elementary,
                MathDomain::Algebra,
                MathDomain::Calculus,
                MathDomain::LinearAlgebra,
                MathDomain::Statistics,
                MathDomain::Physics,
                MathDomain::Competition,
            ],
            ..Self::default()
        }
    }

    /// Minerva 62B configuration
    pub fn minerva_62b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32000,
                hidden_size: 8192,
                intermediate_size: 32768,
                num_hidden_layers: 64,
                num_attention_heads: 64,
                num_key_value_heads: Some(8),
                max_position_embeddings: 2048,
                ..LlamaConfig::default()
            },
            math_vocab_size: Some(33000),
            math_context_length: 2048,
            model_variant: MathModelVariant::Minerva,
            ..Self::default()
        }
    }

    /// MAmmoTH 7B configuration
    pub fn mammoth_7b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32000,
                hidden_size: 4096,
                intermediate_size: 11008,
                num_hidden_layers: 32,
                num_attention_heads: 32,
                max_position_embeddings: 8192,
                ..LlamaConfig::default()
            },
            math_vocab_size: Some(36000), // Larger vocabulary for diverse domains
            math_context_length: 8192,
            model_variant: MathModelVariant::Mammoth,
            supported_domains: vec![
                MathDomain::Elementary,
                MathDomain::Algebra,
                MathDomain::Calculus,
                MathDomain::LinearAlgebra,
                MathDomain::DiscreteMath,
                MathDomain::Statistics,
                MathDomain::Geometry,
                MathDomain::NumberTheory,
                MathDomain::Applied,
                MathDomain::Physics,
                MathDomain::ComputerScience,
                MathDomain::Competition,
            ],
            reasoning_strategies: vec![
                ReasoningStrategy::StepByStep,
                ReasoningStrategy::ChainOfThought,
                ReasoningStrategy::BackwardReasoning,
                ReasoningStrategy::CaseAnalysis,
                ReasoningStrategy::Induction,
                ReasoningStrategy::ProofByContradiction,
                ReasoningStrategy::Constructive,
                ReasoningStrategy::Analogical,
                ReasoningStrategy::Symbolic,
            ],
            ..Self::default()
        }
    }

    /// MAmmoTH 13B configuration
    pub fn mammoth_13b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32000,
                hidden_size: 5120,
                intermediate_size: 13824,
                num_hidden_layers: 40,
                num_attention_heads: 40,
                max_position_embeddings: 8192,
                ..LlamaConfig::default()
            },
            math_vocab_size: Some(36000),
            math_context_length: 8192,
            model_variant: MathModelVariant::Mammoth,
            ..Self::mammoth_7b()
        }
    }

    /// Qwen-Math 7B configuration
    pub fn qwen_math_7b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 151936,
                hidden_size: 3584,
                intermediate_size: 18944,
                num_hidden_layers: 28,
                num_attention_heads: 28,
                num_key_value_heads: Some(4),
                max_position_embeddings: 32768,
                ..LlamaConfig::default()
            },
            math_vocab_size: Some(155000), // Extended for math symbols
            math_context_length: 32768,    // Long context for complex problems
            model_variant: MathModelVariant::QwenMath,
            ..Self::default()
        }
    }

    /// Qwen-Math 72B configuration
    pub fn qwen_math_72b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 151936,
                hidden_size: 8192,
                intermediate_size: 24576,
                num_hidden_layers: 80,
                num_attention_heads: 64,
                num_key_value_heads: Some(8),
                max_position_embeddings: 32768,
                ..LlamaConfig::default()
            },
            math_vocab_size: Some(155000),
            math_context_length: 32768,
            model_variant: MathModelVariant::QwenMath,
            ..Self::default()
        }
    }

    /// CodeT5-Math configuration (for code-math hybrid reasoning)
    pub fn codet5_math() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32100,
                hidden_size: 768,
                intermediate_size: 3072,
                num_hidden_layers: 12,
                num_attention_heads: 12,
                max_position_embeddings: 1024,
                ..LlamaConfig::default()
            },
            math_vocab_size: Some(35000),
            math_context_length: 1024,
            model_variant: MathModelVariant::CodeT5Math,
            symbolic_computation: true,
            supported_domains: vec![
                MathDomain::ComputerScience,
                MathDomain::Applied,
                MathDomain::DiscreteMath,
                MathDomain::Statistics,
            ],
            ..Self::default()
        }
    }

    /// Create configuration from model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            // Math LLaMA variants
            "math-llama-7b" | "mathllama-7b" => Some(Self::math_llama_7b()),
            "math-llama-13b" | "mathllama-13b" => Some(Self::math_llama_13b()),
            "math-llama-70b" | "mathllama-70b" => Some(Self::math_llama_70b()),

            // DeepSeek-Math variants
            "deepseek-math-7b" => Some(Self::deepseek_math_7b()),
            "deepseek-math-67b" => Some(Self::deepseek_math_67b()),

            // Minerva variants
            "minerva-8b" | "google/minerva-8b" => Some(Self::minerva_8b()),
            "minerva-62b" | "google/minerva-62b" => Some(Self::minerva_62b()),

            // MAmmoTH variants
            "mammoth-7b" | "mammoth-math-7b" => Some(Self::mammoth_7b()),
            "mammoth-13b" | "mammoth-math-13b" => Some(Self::mammoth_13b()),

            // Qwen-Math variants
            "qwen-math-7b" | "Qwen/Qwen2-Math-7B" => Some(Self::qwen_math_7b()),
            "qwen-math-72b" | "Qwen/Qwen2-Math-72B" => Some(Self::qwen_math_72b()),

            // CodeT5-Math
            "codet5-math" | "Salesforce/codet5-math" => Some(Self::codet5_math()),

            _ => None,
        }
    }

    /// Get all available model names
    pub fn available_models() -> Vec<&'static str> {
        vec![
            // Math LLaMA
            "math-llama-7b",
            "math-llama-13b",
            "math-llama-70b",
            // DeepSeek-Math
            "deepseek-math-7b",
            "deepseek-math-67b",
            // Minerva
            "minerva-8b",
            "minerva-62b",
            // MAmmoTH
            "mammoth-7b",
            "mammoth-13b",
            // Qwen-Math
            "qwen-math-7b",
            "qwen-math-72b",
            // CodeT5-Math
            "codet5-math",
        ]
    }

    /// Check if configuration is valid
    pub fn validate(&self) -> Result<()> {
        self.base_config.validate()?;

        if self.math_context_length == 0 {
            return Err(invalid_config(
                "config_field",
                "Math context length must be greater than 0".to_string(),
            ));
        }

        if self.supported_domains.is_empty() {
            return Err(invalid_config(
                "config_field",
                "At least one mathematical domain must be supported".to_string(),
            ));
        }

        if self.reasoning_strategies.is_empty() {
            return Err(invalid_config(
                "config_field",
                "At least one reasoning strategy must be supported".to_string(),
            ));
        }

        if self.chain_of_thought.enabled && self.chain_of_thought.max_steps == 0 {
            return Err(invalid_config(
                "config_field",
                "Chain-of-thought max steps must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Get the effective vocabulary size
    pub fn effective_vocab_size(&self) -> usize {
        self.math_vocab_size.unwrap_or(self.base_config.vocab_size)
    }

    /// Check if model supports a specific mathematical domain
    pub fn supports_domain(&self, domain: &MathDomain) -> bool {
        self.supported_domains.contains(domain)
    }

    /// Check if model supports a specific reasoning strategy
    pub fn supports_strategy(&self, strategy: &ReasoningStrategy) -> bool {
        self.reasoning_strategies.contains(strategy)
    }

    /// Check if model supports LaTeX
    pub fn supports_latex(&self) -> bool {
        self.latex_support
    }

    /// Check if model supports symbolic computation
    pub fn supports_symbolic_computation(&self) -> bool {
        self.symbolic_computation
    }

    /// Get model architecture name
    pub fn architecture(&self) -> &'static str {
        match self.model_variant {
            MathModelVariant::MathLlama => "MathLlama",
            MathModelVariant::DeepSeekMath => "DeepSeekMath",
            MathModelVariant::PalmMath => "PaLMMath",
            MathModelVariant::Mammoth => "MAmmoTH",
            MathModelVariant::Minerva => "Minerva",
            MathModelVariant::QwenMath => "QwenMath",
            MathModelVariant::CodeT5Math => "CodeT5Math",
        }
    }
}

/// Mathematics-specialized model implementation
pub struct MathSpecializedModel {
    base_model: LlamaModel,
    config: MathSpecializedConfig,
}

impl MathSpecializedModel {
    /// Create a new mathematics-specialized model
    pub fn new(config: MathSpecializedConfig) -> Result<Self> {
        config.validate()?;
        let base_model = LlamaModel::new(config.base_config.clone())?;

        Ok(Self { base_model, config })
    }

    /// Get the configuration
    pub fn config(&self) -> &MathSpecializedConfig {
        &self.config
    }

    /// Check if model supports a specific mathematical domain
    pub fn supports_domain(&self, domain: &MathDomain) -> bool {
        self.config.supports_domain(domain)
    }

    /// Check if model supports a specific reasoning strategy
    pub fn supports_strategy(&self, strategy: &ReasoningStrategy) -> bool {
        self.config.supports_strategy(strategy)
    }

    /// Get supported mathematical domains
    pub fn supported_domains(&self) -> &[MathDomain] {
        &self.config.supported_domains
    }

    /// Get supported reasoning strategies
    pub fn supported_strategies(&self) -> &[ReasoningStrategy] {
        &self.config.reasoning_strategies
    }

    /// Create model from pretrained name
    pub fn from_pretrained_name(name: &str) -> Result<Self> {
        let config = MathSpecializedConfig::from_pretrained_name(name)
            .ok_or_else(|| invalid_config("model_name", format!("Unknown math model: {}", name)))?;
        Self::new(config)
    }
}

impl Layer for MathSpecializedModel {
    type Input = Vec<u32>; // Token IDs
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.base_model.forward(input)
    }
}

/// Mathematics-specialized model with language modeling head
pub struct MathSpecializedForCausalLM {
    model: MathSpecializedModel,
    lm_head: trustformers_core::layers::Linear,
}

impl MathSpecializedForCausalLM {
    /// Create a new mathematics-specialized model for causal language modeling
    pub fn new(config: MathSpecializedConfig) -> Result<Self> {
        let vocab_size = config.effective_vocab_size();
        let hidden_size = config.base_config.hidden_size;

        let model = MathSpecializedModel::new(config)?;
        let lm_head = trustformers_core::layers::Linear::new(hidden_size, vocab_size, false);

        Ok(Self { model, lm_head })
    }

    /// Get the configuration
    pub fn config(&self) -> &MathSpecializedConfig {
        self.model.config()
    }

    /// Create model from pretrained name
    pub fn from_pretrained_name(name: &str) -> Result<Self> {
        let config = MathSpecializedConfig::from_pretrained_name(name)
            .ok_or_else(|| invalid_config("model_name", format!("Unknown math model: {}", name)))?;
        Self::new(config)
    }

    /// Solve a mathematical problem step-by-step (placeholder implementation)
    pub fn solve_step_by_step(&mut self, _problem: &str) -> Result<String> {
        // This is a placeholder implementation
        // In a real implementation, this would:
        // 1. Parse the mathematical problem
        // 2. Apply appropriate reasoning strategies
        // 3. Generate step-by-step solution
        // 4. Verify each step
        // 5. Format the final solution

        Ok("Step-by-step solution would be generated here".to_string())
    }

    /// Generate mathematical proof (placeholder implementation)
    pub fn generate_proof(&mut self, _theorem: &str) -> Result<String> {
        // Placeholder for proof generation
        Ok("Mathematical proof would be generated here".to_string())
    }

    /// Evaluate mathematical expression (placeholder implementation)
    pub fn evaluate_expression(&mut self, _expression: &str) -> Result<String> {
        // Placeholder for expression evaluation
        Ok("Expression evaluation would be performed here".to_string())
    }
}

impl Layer for MathSpecializedForCausalLM {
    type Input = Vec<u32>; // Token IDs
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.model.forward(input)?;
        self.lm_head.forward(hidden_states)
    }
}

/// Mathematical problem types for specialized handling
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MathProblemType {
    /// Algebraic equations and manipulations
    Algebraic,
    /// Calculus problems (derivatives, integrals)
    Calculus,
    /// Geometry problems
    Geometry,
    /// Probability and statistics
    Statistics,
    /// Number theory problems
    NumberTheory,
    /// Linear algebra problems
    LinearAlgebra,
    /// Optimization problems
    Optimization,
    /// Differential equations
    DifferentialEquations,
    /// Mathematical proofs
    Proof,
    /// Word problems
    WordProblem,
}

/// Mathematical reasoning output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathReasoningOutput {
    /// The original problem
    pub problem: String,
    /// Problem type classification
    pub problem_type: MathProblemType,
    /// Step-by-step solution
    pub steps: Vec<ReasoningStep>,
    /// Final answer
    pub answer: String,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Reasoning strategy used
    pub strategy: ReasoningStrategy,
}

/// Individual reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Step number
    pub step_number: usize,
    /// Step description
    pub description: String,
    /// Mathematical expression or equation
    pub expression: Option<String>,
    /// Justification for this step
    pub justification: String,
    /// Confidence in this step
    pub confidence: f32,
}

// Convenience type aliases for common math models
pub type MathLlamaConfig = MathSpecializedConfig;
pub type MathLlamaModel = MathSpecializedModel;
pub type MathLlamaForCausalLM = MathSpecializedForCausalLM;

pub type DeepSeekMathConfig = MathSpecializedConfig;
pub type DeepSeekMathModel = MathSpecializedModel;
pub type DeepSeekMathForCausalLM = MathSpecializedForCausalLM;

pub type MinervaConfig = MathSpecializedConfig;
pub type MinervaModel = MathSpecializedModel;
pub type MinervaForCausalLM = MathSpecializedForCausalLM;

pub type MammothConfig = MathSpecializedConfig;
pub type MammothModel = MathSpecializedModel;
pub type MammothForCausalLM = MathSpecializedForCausalLM;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_specialized_config_creation() {
        let config = MathSpecializedConfig::math_llama_7b();
        assert_eq!(config.base_config.hidden_size, 4096);
        assert_eq!(config.math_context_length, 8192);
        assert_eq!(config.model_variant, MathModelVariant::MathLlama);
        assert!(config.step_by_step_reasoning);
        assert!(config.math_notation_support);
    }

    #[test]
    fn test_deepseek_math_config() {
        let config = MathSpecializedConfig::deepseek_math_7b();
        assert_eq!(config.base_config.hidden_size, 4096);
        assert_eq!(config.math_context_length, 16384);
        assert_eq!(config.model_variant, MathModelVariant::DeepSeekMath);
        assert!(config.supports_domain(&MathDomain::Competition));
    }

    #[test]
    fn test_minerva_config() {
        let config = MathSpecializedConfig::minerva_8b();
        assert_eq!(config.base_config.hidden_size, 4096);
        assert_eq!(config.model_variant, MathModelVariant::Minerva);
        assert!(config.latex_support);
        assert!(config.formula_parsing);
    }

    #[test]
    fn test_mammoth_config() {
        let config = MathSpecializedConfig::mammoth_7b();
        assert_eq!(config.model_variant, MathModelVariant::Mammoth);
        assert_eq!(config.supported_domains.len(), 12); // All domains
        assert_eq!(config.reasoning_strategies.len(), 9); // All strategies including ProofByContradiction
    }

    #[test]
    fn test_from_pretrained_name() {
        let config = MathSpecializedConfig::from_pretrained_name("math-llama-7b");
        assert!(config.is_some());
        let config = config.expect("operation failed");
        assert_eq!(config.model_variant, MathModelVariant::MathLlama);

        let config = MathSpecializedConfig::from_pretrained_name("minerva-8b");
        assert!(config.is_some());
        let config = config.expect("operation failed");
        assert_eq!(config.model_variant, MathModelVariant::Minerva);

        let config = MathSpecializedConfig::from_pretrained_name("unknown-model");
        assert!(config.is_none());
    }

    #[test]
    fn test_available_models() {
        let models = MathSpecializedConfig::available_models();
        assert!(models.contains(&"math-llama-7b"));
        assert!(models.contains(&"deepseek-math-7b"));
        assert!(models.contains(&"minerva-8b"));
        assert!(models.contains(&"mammoth-7b"));
        assert!(models.contains(&"qwen-math-7b"));
        assert!(models.len() >= 11); // Should have at least 11 models
    }

    #[test]
    fn test_domain_support() {
        let config = MathSpecializedConfig::default();
        assert!(config.supports_domain(&MathDomain::Algebra));
        assert!(config.supports_domain(&MathDomain::Calculus));
        assert!(!config.supports_domain(&MathDomain::Competition));

        let mammoth_config = MathSpecializedConfig::mammoth_7b();
        assert!(mammoth_config.supports_domain(&MathDomain::Competition));
        assert!(mammoth_config.supports_domain(&MathDomain::NumberTheory));
    }

    #[test]
    fn test_strategy_support() {
        let config = MathSpecializedConfig::default();
        assert!(config.supports_strategy(&ReasoningStrategy::StepByStep));
        assert!(config.supports_strategy(&ReasoningStrategy::ChainOfThought));
        assert!(!config.supports_strategy(&ReasoningStrategy::Induction));

        let mammoth_config = MathSpecializedConfig::mammoth_7b();
        assert!(mammoth_config.supports_strategy(&ReasoningStrategy::Induction));
        assert!(mammoth_config.supports_strategy(&ReasoningStrategy::ProofByContradiction));
    }

    #[test]
    fn test_config_validation() {
        let config = MathSpecializedConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = MathSpecializedConfig::default();
        invalid_config.math_context_length = 0;
        assert!(invalid_config.validate().is_err());

        let mut invalid_config = MathSpecializedConfig::default();
        invalid_config.supported_domains.clear();
        assert!(invalid_config.validate().is_err());

        let mut invalid_config = MathSpecializedConfig::default();
        invalid_config.reasoning_strategies.clear();
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_chain_of_thought_config() {
        let config = ChainOfThoughtConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_steps, 20);
        assert!(config.step_verification);
        assert!(config.confidence_scoring);
        assert!(config.backtrack_on_error);
    }

    #[test]
    fn test_math_special_tokens() {
        let tokens = MathSpecialTokens::default();
        assert_eq!(tokens.step_separator, "<step>");
        assert_eq!(tokens.solution_start, "<solution>");
        assert_eq!(tokens.therefore, "∴");
        assert_eq!(tokens.because, "∵");
        assert_eq!(tokens.qed, "□");
    }

    #[test]
    fn test_effective_vocab_size() {
        let config = MathSpecializedConfig::math_llama_7b();
        assert_eq!(config.effective_vocab_size(), 35000);

        let mut config = MathSpecializedConfig::default();
        config.math_vocab_size = None;
        config.base_config.vocab_size = 50000;
        assert_eq!(config.effective_vocab_size(), 50000);
    }

    #[test]
    fn test_architecture_names() {
        let config = MathSpecializedConfig::math_llama_7b();
        assert_eq!(config.architecture(), "MathLlama");

        let config = MathSpecializedConfig::minerva_8b();
        assert_eq!(config.architecture(), "Minerva");

        let config = MathSpecializedConfig::deepseek_math_7b();
        assert_eq!(config.architecture(), "DeepSeekMath");
    }

    #[test]
    fn test_model_creation() {
        let config = MathSpecializedConfig {
            base_config: LlamaConfig {
                vocab_size: 1000,
                hidden_size: 64,
                intermediate_size: 256,
                num_hidden_layers: 2,
                num_attention_heads: 4,
                max_position_embeddings: 512,
                ..LlamaConfig::default()
            },
            math_context_length: 512,
            ..MathSpecializedConfig::default()
        };

        let model = MathSpecializedModel::new(config.clone());
        assert!(model.is_ok());
        let model = model.expect("operation failed");
        assert!(model.supports_domain(&MathDomain::Algebra));
        assert!(model.supports_strategy(&ReasoningStrategy::StepByStep));

        let causal_lm = MathSpecializedForCausalLM::new(config);
        assert!(causal_lm.is_ok());
    }

    #[test]
    fn test_math_problem_types() {
        // Test that all math problem types are properly defined
        let problem_types = [
            MathProblemType::Algebraic,
            MathProblemType::Calculus,
            MathProblemType::Geometry,
            MathProblemType::Statistics,
            MathProblemType::NumberTheory,
            MathProblemType::LinearAlgebra,
            MathProblemType::Optimization,
            MathProblemType::DifferentialEquations,
            MathProblemType::Proof,
            MathProblemType::WordProblem,
        ];
        assert_eq!(problem_types.len(), 10);
    }

    #[test]
    fn test_reasoning_step() {
        let step = ReasoningStep {
            step_number: 1,
            description: "Expand the expression".to_string(),
            expression: Some("(x + 1)^2 = x^2 + 2x + 1".to_string()),
            justification: "Using the binomial theorem".to_string(),
            confidence: 0.95,
        };
        assert_eq!(step.step_number, 1);
        assert!(step.expression.is_some());
        assert_eq!(step.confidence, 0.95);
    }

    #[test]
    fn test_qwen_math_long_context() {
        let config = MathSpecializedConfig::qwen_math_7b();
        assert_eq!(config.math_context_length, 32768); // Long context
        assert_eq!(config.model_variant, MathModelVariant::QwenMath);
    }

    #[test]
    fn test_codet5_math_config() {
        let config = MathSpecializedConfig::codet5_math();
        assert_eq!(config.model_variant, MathModelVariant::CodeT5Math);
        assert!(config.symbolic_computation);
        assert!(config.supports_domain(&MathDomain::ComputerScience));
    }
}
