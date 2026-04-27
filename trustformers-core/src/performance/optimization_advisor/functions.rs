//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::errors::Result;

use super::types::{AnalysisContext, OptimizationSuggestion};

/// Trait for optimization rules
pub trait OptimizationRule: Send + Sync {
    /// Analyze context and provide suggestion if applicable
    fn analyze(&self, context: &AnalysisContext) -> Result<Option<OptimizationSuggestion>>;
}
