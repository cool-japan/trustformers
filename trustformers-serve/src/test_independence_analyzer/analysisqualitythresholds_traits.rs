//! # AnalysisQualityThresholds - Trait Implementations
//!
//! This module contains trait implementations for `AnalysisQualityThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

pub use super::types::*;

use std::time::Duration;

impl Default for AnalysisQualityThresholds {
    fn default() -> Self {
        Self {
            min_dependency_accuracy: 0.8,
            min_conflict_accuracy: 0.85,
            min_grouping_quality: 0.7,
            max_analysis_time: Duration::from_secs(120),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_default_dependency_accuracy() {
        let t = AnalysisQualityThresholds::default();
        assert!((t.min_dependency_accuracy - 0.8_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_conflict_accuracy() {
        let t = AnalysisQualityThresholds::default();
        assert!((t.min_conflict_accuracy - 0.85_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_grouping_quality() {
        let t = AnalysisQualityThresholds::default();
        assert!((t.min_grouping_quality - 0.7_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_max_analysis_time() {
        let t = AnalysisQualityThresholds::default();
        assert_eq!(t.max_analysis_time, Duration::from_secs(120));
    }

    #[test]
    fn test_all_accuracy_values_in_range() {
        let t = AnalysisQualityThresholds::default();
        assert!(t.min_dependency_accuracy >= 0.0 && t.min_dependency_accuracy <= 1.0);
        assert!(t.min_conflict_accuracy >= 0.0 && t.min_conflict_accuracy <= 1.0);
        assert!(t.min_grouping_quality >= 0.0 && t.min_grouping_quality <= 1.0);
    }

    #[test]
    fn test_conflict_accuracy_higher_than_dependency() {
        let t = AnalysisQualityThresholds::default();
        assert!(t.min_conflict_accuracy >= t.min_dependency_accuracy);
    }

    #[test]
    fn test_max_analysis_time_positive() {
        let t = AnalysisQualityThresholds::default();
        assert!(t.max_analysis_time > Duration::from_secs(0));
    }

    #[test]
    fn test_max_analysis_time_reasonable() {
        let t = AnalysisQualityThresholds::default();
        // Should be less than 10 minutes
        assert!(t.max_analysis_time < Duration::from_secs(600));
    }

    #[test]
    fn test_custom_thresholds_construction() {
        let t = AnalysisQualityThresholds {
            min_dependency_accuracy: 0.95,
            min_conflict_accuracy: 0.99,
            min_grouping_quality: 0.90,
            max_analysis_time: Duration::from_secs(60),
        };
        assert!(t.min_dependency_accuracy > 0.9);
        assert!(t.max_analysis_time < Duration::from_secs(120));
    }

    #[test]
    fn test_lenient_thresholds_construction() {
        let t = AnalysisQualityThresholds {
            min_dependency_accuracy: 0.5,
            min_conflict_accuracy: 0.5,
            min_grouping_quality: 0.5,
            max_analysis_time: Duration::from_secs(300),
        };
        assert!(t.min_dependency_accuracy < 0.8);
        assert!(t.max_analysis_time > Duration::from_secs(120));
    }

    #[test]
    fn test_thresholds_debug_format() {
        let t = AnalysisQualityThresholds::default();
        let debug_str = format!("{:?}", t);
        assert!(
            debug_str.contains("min_dependency_accuracy")
                || debug_str.contains("AnalysisQualityThresholds")
        );
    }

    #[test]
    fn test_grouping_quality_lower_than_conflict() {
        let t = AnalysisQualityThresholds::default();
        assert!(t.min_grouping_quality <= t.min_conflict_accuracy);
    }

    #[test]
    fn test_two_defaults_equal_accuracy() {
        let t1 = AnalysisQualityThresholds::default();
        let t2 = AnalysisQualityThresholds::default();
        assert!((t1.min_dependency_accuracy - t2.min_dependency_accuracy).abs() < f32::EPSILON);
        assert!((t1.min_conflict_accuracy - t2.min_conflict_accuracy).abs() < f32::EPSILON);
        assert!((t1.min_grouping_quality - t2.min_grouping_quality).abs() < f32::EPSILON);
        assert_eq!(t1.max_analysis_time, t2.max_analysis_time);
    }

    #[test]
    fn test_strict_thresholds() {
        let strict = AnalysisQualityThresholds {
            min_dependency_accuracy: 1.0,
            min_conflict_accuracy: 1.0,
            min_grouping_quality: 1.0,
            max_analysis_time: Duration::from_secs(10),
        };
        assert!((strict.min_dependency_accuracy - 1.0_f32).abs() < f32::EPSILON);
        assert!((strict.min_conflict_accuracy - 1.0_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_analysis_time_comparison() {
        let fast = AnalysisQualityThresholds {
            min_dependency_accuracy: 0.7,
            min_conflict_accuracy: 0.75,
            min_grouping_quality: 0.6,
            max_analysis_time: Duration::from_secs(30),
        };
        let slow = AnalysisQualityThresholds {
            min_dependency_accuracy: 0.95,
            min_conflict_accuracy: 0.98,
            min_grouping_quality: 0.9,
            max_analysis_time: Duration::from_secs(300),
        };
        assert!(fast.max_analysis_time < slow.max_analysis_time);
        assert!(fast.min_dependency_accuracy < slow.min_dependency_accuracy);
    }

    #[test]
    fn test_min_analysis_time_nonzero() {
        let t = AnalysisQualityThresholds {
            min_dependency_accuracy: 0.8,
            min_conflict_accuracy: 0.85,
            min_grouping_quality: 0.7,
            max_analysis_time: Duration::from_millis(500),
        };
        assert!(t.max_analysis_time.as_millis() >= 500);
    }

    #[test]
    fn test_default_vs_custom_comparison() {
        let default = AnalysisQualityThresholds::default();
        let custom = AnalysisQualityThresholds {
            min_dependency_accuracy: 0.9,
            min_conflict_accuracy: 0.92,
            min_grouping_quality: 0.88,
            max_analysis_time: Duration::from_secs(60),
        };
        // custom should be stricter on accuracy but faster on time
        assert!(custom.min_dependency_accuracy > default.min_dependency_accuracy);
        assert!(custom.max_analysis_time < default.max_analysis_time);
    }

    #[test]
    fn test_all_thresholds_finite() {
        let t = AnalysisQualityThresholds::default();
        assert!(t.min_dependency_accuracy.is_finite());
        assert!(t.min_conflict_accuracy.is_finite());
        assert!(t.min_grouping_quality.is_finite());
    }

    #[test]
    fn test_zero_thresholds_are_valid_lower_bound() {
        let t = AnalysisQualityThresholds {
            min_dependency_accuracy: 0.0,
            min_conflict_accuracy: 0.0,
            min_grouping_quality: 0.0,
            max_analysis_time: Duration::from_secs(1),
        };
        assert_eq!(t.min_dependency_accuracy, 0.0);
        assert_eq!(t.min_conflict_accuracy, 0.0);
        assert_eq!(t.min_grouping_quality, 0.0);
    }

    #[test]
    fn test_dependency_accuracy_not_nan() {
        let t = AnalysisQualityThresholds::default();
        assert!(!t.min_dependency_accuracy.is_nan());
        assert!(!t.min_conflict_accuracy.is_nan());
        assert!(!t.min_grouping_quality.is_nan());
    }

    #[test]
    fn test_max_analysis_time_two_minutes() {
        let t = AnalysisQualityThresholds::default();
        assert_eq!(t.max_analysis_time.as_secs(), 120);
    }
}
