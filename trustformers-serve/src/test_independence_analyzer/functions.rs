//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

pub trait DurationExt {
    fn from_hours(hours: u64) -> Duration;
}

impl DurationExt for Duration {
    fn from_hours(hours: u64) -> Duration {
        Duration::from_secs(hours * 3600)
    }
}

#[cfg(test)]
mod tests {

    use crate::test_independence_analyzer::AnalysisConfig;
    use crate::test_timeout_optimization::{
        TestCategory, TestComplexityHints, TestExecutionContext,
    };
    use crate::TestIndependenceAnalyzer;
    use std::time::Duration;

    fn create_test_context(name: &str, category: TestCategory) -> TestExecutionContext {
        TestExecutionContext {
            test_name: name.to_string(),
            category,
            environment: "test".to_string(),
            complexity_hints: TestComplexityHints::default(),
            expected_duration: Some(Duration::from_secs(10)),
            timeout_override: None,
        }
    }

    #[tokio::test]
    async fn test_analyzer_creation() {
        let analyzer = TestIndependenceAnalyzer::new();
        let stats = analyzer.get_analysis_statistics();
        assert_eq!(stats.total_analyses, 0);
    }

    #[tokio::test]
    async fn test_basic_analysis() {
        let analyzer = TestIndependenceAnalyzer::new();
        let tests = vec![
            create_test_context("test1", TestCategory::Unit),
            create_test_context("test2", TestCategory::Integration),
        ];
        let analysis = analyzer.analyze_test_independence(&tests).await.unwrap();
        assert_eq!(analysis.tests.len(), 2);
        assert!(!analysis.groups.is_empty());
        assert!(analysis.analysis_metadata.analysis_quality >= 0.0);
    }

    #[tokio::test]
    async fn test_configuration_update() {
        let analyzer = TestIndependenceAnalyzer::new();
        let mut config = AnalysisConfig::default();
        config.enable_ml_conflict_prediction = true;
        analyzer.update_config(config.clone());
        let retrieved_config = analyzer.get_config();
        assert!(retrieved_config.enable_ml_conflict_prediction);
    }

    #[tokio::test]
    async fn test_quality_assessment() {
        let analyzer = TestIndependenceAnalyzer::new();
        let tests = vec![create_test_context("test1", TestCategory::Unit)];
        let analysis = analyzer.analyze_test_independence(&tests).await.unwrap();
        assert!(analysis.quality_assessment.overall_score >= 0.0);
        assert!(analysis.quality_assessment.overall_score <= 1.0);
    }
}
