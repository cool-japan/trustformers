//! Input Validation Module
//!
//! Provides comprehensive input validation and sanitization for inference requests
//! to prevent security vulnerabilities and ensure safe processing.

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for input validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Maximum text input length
    pub max_text_length: usize,
    /// Maximum number of tokens
    pub max_tokens: usize,
    /// Maximum temperature value
    pub max_temperature: f32,
    /// Minimum temperature value
    pub min_temperature: f32,
    /// Maximum top_p value
    pub max_top_p: f32,
    /// Minimum top_p value
    pub min_top_p: f32,
    /// Maximum number of sequences to generate
    pub max_num_sequences: usize,
    /// Enable profanity filtering
    pub enable_profanity_filter: bool,
    /// Enable PII detection
    pub enable_pii_detection: bool,
    /// Allowed model IDs (empty = allow all)
    pub allowed_models: Vec<String>,
    /// Blocked keywords
    pub blocked_keywords: Vec<String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_text_length: 10000,
            max_tokens: 2048,
            max_temperature: 2.0,
            min_temperature: 0.0,
            max_top_p: 1.0,
            min_top_p: 0.0,
            max_num_sequences: 10,
            enable_profanity_filter: true,
            enable_pii_detection: true,
            allowed_models: vec![],
            blocked_keywords: vec![],
        }
    }
}

/// Validation errors
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Text input too long: {0} > {1} characters")]
    TextTooLong(usize, usize),

    #[error("Invalid token count: {0} > {1}")]
    InvalidTokenCount(usize, usize),

    #[error("Invalid temperature: {0} not in range [{1}, {2}]")]
    InvalidTemperature(f32, f32, f32),

    #[error("Invalid top_p: {0} not in range [{1}, {2}]")]
    InvalidTopP(f32, f32, f32),

    #[error("Too many sequences requested: {0} > {1}")]
    TooManySequences(usize, usize),

    #[error("Model not allowed: {0}")]
    ModelNotAllowed(String),

    #[error("Content blocked: contains restricted keywords")]
    ContentBlocked,

    #[error("Profanity detected in input")]
    ProfanityDetected,

    #[error("PII detected in input: {0}")]
    PIIDetected(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Input validation service
#[derive(Debug, Clone)]
pub struct ValidationService {
    config: ValidationConfig,
    profanity_regex: Option<Regex>,
    pii_regexes: HashMap<String, Regex>,
}

impl ValidationService {
    /// Create new validation service with configuration
    pub fn new(config: ValidationConfig) -> Result<Self> {
        let mut service = Self {
            config,
            profanity_regex: None,
            pii_regexes: HashMap::new(),
        };

        service.init_regexes()?;
        Ok(service)
    }

    /// Initialize regex patterns for content filtering
    fn init_regexes(&mut self) -> Result<()> {
        // Basic profanity filter (expand as needed)
        if self.config.enable_profanity_filter {
            let profanity_pattern = r"(?i)\b(badword1|badword2|profanity)\b";
            self.profanity_regex = Some(Regex::new(profanity_pattern)?);
        }

        // PII detection patterns
        if self.config.enable_pii_detection {
            // Email addresses
            self.pii_regexes.insert(
                "email".to_string(),
                Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")?,
            );

            // Phone numbers (US format)
            self.pii_regexes.insert(
                "phone".to_string(),
                Regex::new(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b")?,
            );

            // Social Security Numbers (US format)
            self.pii_regexes
                .insert("ssn".to_string(), Regex::new(r"\b\d{3}-\d{2}-\d{4}\b")?);

            // Credit card numbers (basic pattern)
            self.pii_regexes.insert(
                "credit_card".to_string(),
                Regex::new(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")?,
            );
        }

        Ok(())
    }

    /// Validate text input content
    pub fn validate_text(&self, text: &str) -> Result<(), ValidationError> {
        // Check length
        if text.len() > self.config.max_text_length {
            return Err(ValidationError::TextTooLong(
                text.len(),
                self.config.max_text_length,
            ));
        }

        // Check blocked keywords
        for keyword in &self.config.blocked_keywords {
            if text.to_lowercase().contains(&keyword.to_lowercase()) {
                return Err(ValidationError::ContentBlocked);
            }
        }

        // Profanity check
        if let Some(ref regex) = self.profanity_regex {
            if regex.is_match(text) {
                return Err(ValidationError::ProfanityDetected);
            }
        }

        // PII detection
        for (pii_type, regex) in &self.pii_regexes {
            if regex.is_match(text) {
                return Err(ValidationError::PIIDetected(pii_type.clone()));
            }
        }

        Ok(())
    }

    /// Validate model ID
    pub fn validate_model(&self, model_id: &str) -> Result<(), ValidationError> {
        if !self.config.allowed_models.is_empty()
            && !self.config.allowed_models.contains(&model_id.to_string())
        {
            return Err(ValidationError::ModelNotAllowed(model_id.to_string()));
        }
        Ok(())
    }

    /// Validate generation parameters
    pub fn validate_generation_params(
        &self,
        params: &GenerationParams,
    ) -> Result<(), ValidationError> {
        // Validate max_tokens
        if params.max_tokens > self.config.max_tokens {
            return Err(ValidationError::InvalidTokenCount(
                params.max_tokens,
                self.config.max_tokens,
            ));
        }

        // Validate temperature
        if let Some(temp) = params.temperature {
            if temp < self.config.min_temperature || temp > self.config.max_temperature {
                return Err(ValidationError::InvalidTemperature(
                    temp,
                    self.config.min_temperature,
                    self.config.max_temperature,
                ));
            }
        }

        // Validate top_p
        if let Some(top_p) = params.top_p {
            if top_p < self.config.min_top_p || top_p > self.config.max_top_p {
                return Err(ValidationError::InvalidTopP(
                    top_p,
                    self.config.min_top_p,
                    self.config.max_top_p,
                ));
            }
        }

        // Validate number of sequences
        if let Some(n) = params.n {
            if n > self.config.max_num_sequences {
                return Err(ValidationError::TooManySequences(
                    n,
                    self.config.max_num_sequences,
                ));
            }
        }

        Ok(())
    }

    /// Validate complete inference request
    pub fn validate_request(&self, request: &InferenceRequest) -> Result<(), ValidationError> {
        // Validate model
        self.validate_model(&request.model)?;

        // Validate input text
        self.validate_text(&request.prompt)?;

        // Validate generation parameters
        self.validate_generation_params(&request.params)?;

        Ok(())
    }

    /// Sanitize output text
    pub fn sanitize_output(&self, text: &str) -> String {
        let mut sanitized = text.to_string();

        // Remove potential PII from output
        for regex in self.pii_regexes.values() {
            sanitized = regex.replace_all(&sanitized, "[REDACTED]").to_string();
        }

        // Filter profanity
        if let Some(ref regex) = self.profanity_regex {
            sanitized = regex.replace_all(&sanitized, "[FILTERED]").to_string();
        }

        sanitized
    }
}

/// Generation parameters for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub n: Option<usize>,
    pub stop: Option<Vec<String>>,
}

/// Inference request structure for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model: String,
    pub prompt: String,
    pub params: GenerationParams,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_length_validation() {
        let config = ValidationConfig {
            max_text_length: 10,
            ..Default::default()
        };
        let service = ValidationService::new(config).unwrap();

        assert!(service.validate_text("short").is_ok());
        assert!(service.validate_text("this is too long").is_err());
    }

    #[test]
    fn test_model_validation() {
        let config = ValidationConfig {
            allowed_models: vec!["gpt-3.5".to_string(), "gpt-4".to_string()],
            ..Default::default()
        };
        let service = ValidationService::new(config).unwrap();

        assert!(service.validate_model("gpt-3.5").is_ok());
        assert!(service.validate_model("gpt-4").is_ok());
        assert!(service.validate_model("claude").is_err());
    }

    #[test]
    fn test_generation_params_validation() {
        let config = ValidationConfig::default();
        let service = ValidationService::new(config).unwrap();

        let valid_params = GenerationParams {
            max_tokens: 100,
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: None,
            n: Some(1),
            stop: None,
        };
        assert!(service.validate_generation_params(&valid_params).is_ok());

        let invalid_params = GenerationParams {
            max_tokens: 10000,      // Too many
            temperature: Some(3.0), // Too high
            top_p: Some(1.5),       // Too high
            top_k: None,
            n: Some(20), // Too many
            stop: None,
        };
        assert!(service.validate_generation_params(&invalid_params).is_err());
    }
}
