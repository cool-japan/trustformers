//! # AntiPatternClassifier - Trait Implementations
//!
//! This module contains trait implementations for `AntiPatternClassifier`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `PatternClassifier`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::test_characterization::types::{DetectedPattern, PatternType};
use anyhow::Result;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::time::Instant;

use super::functions::PatternClassifier;
use super::types::{
    AntiPatternClassifier, ClassificationResult, ClassifierMetadata, PatternCategory,
};

impl Default for AntiPatternClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternClassifier for AntiPatternClassifier {
    fn classify(
        &self,
        _pattern: &DetectedPattern,
    ) -> Pin<Box<dyn Future<Output = Result<ClassificationResult>> + Send + '_>> {
        Box::pin(async move {
            Ok(ClassificationResult {
                category: PatternCategory::AntiPattern,
                confidence: 0.9,
                metadata: HashMap::new(),
                alternatives: Vec::new(),
                classified_at: Instant::now(),
            })
        })
    }
    fn metadata(&self) -> ClassifierMetadata {
        ClassifierMetadata::default()
    }
    fn can_classify(&self, pattern_type: PatternType) -> bool {
        matches!(pattern_type, PatternType::Behavioral)
    }
    fn get_confidence(&self, _pattern: &DetectedPattern) -> f64 {
        0.9
    }
}
