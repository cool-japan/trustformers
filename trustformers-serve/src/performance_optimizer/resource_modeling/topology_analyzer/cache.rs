//! Cache hierarchy analysis

use super::types::*;

/// Cache hierarchy analyzer
pub struct CacheHierarchyAnalyzer {
    /// Cache analysis configuration
    pub config: CacheAnalysisConfig,
}

/// Cache analysis configuration
pub struct CacheAnalysisConfig {
    /// Enable detailed cache analysis
    pub enable_detailed_analysis: bool,
}

/// Advanced cache analysis results
pub struct CacheAnalysisAdvanced {
    /// Cache levels with detailed information
    pub cache_levels: Vec<CacheLevelInfoAdvanced>,
}

/// Advanced cache level information
pub struct CacheLevelInfoAdvanced {
    /// Cache level
    pub level: u32,

    /// Cache size
    pub size: usize,

    /// Cache type
    pub cache_type: CacheType,

    /// Replacement policy
    pub replacement_policy: CacheReplacementPolicy,

    /// Write policy
    pub write_policy: CacheWritePolicy,
}
