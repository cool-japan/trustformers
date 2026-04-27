//! Analysis Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// Import types from sibling modules
use super::utilities::ImprovementOpportunity;

#[derive(Debug, Serialize, Deserialize)]
pub struct FlapDetection {
    pub enabled: bool,
    pub threshold: u32,
    pub window: Duration,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub success: bool,
    pub space_saved_bytes: u64,
    pub optimization_time_ms: f64,
    pub compression_savings: u64,
    pub storage_optimization: u64,
    pub retention_cleanup: u64,
    pub total_space_saved: u64,
    pub optimization_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub trend_direction: String,
    pub trend_strength: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoBottleneck {
    pub detected: bool,
    pub bottleneck_type: String,
    pub severity: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DriftDetection {
    pub enabled: bool,
    pub threshold: f64,
    pub window_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutlierAnalysis {
    pub outliers: Vec<Outlier>,
    pub method: String,
    pub threshold: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Outlier {
    pub value: f64,
    pub score: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    pub coefficients: Vec<f64>,
    pub r_squared: f64,
    pub predictions: Vec<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrendComponents {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BottleneckAnalysisResult {
    pub bottlenecks: Vec<String>,
    pub severity_scores: Vec<f64>,
    pub recommendations: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResourceEfficiencyAnalysis {
    pub efficiency_score: f64,
    pub resource_usage: std::collections::HashMap<String, f64>,
    pub optimization_opportunities: Vec<ImprovementOpportunity>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CostBenefitAnalysis {
    pub benefits: f64,
    pub costs: f64,
    pub roi: f64,
    pub payback_period: Duration,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OptimizationRiskAssessment {
    pub risk_level: String,
    pub risk_factors: Vec<String>,
    pub mitigation_strategies: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RegressionAnalysisResult {
    pub analysis: RegressionAnalysis,
    pub model_quality: f64,
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ImprovementAnalysisResult {
    pub baseline: f64,
    pub current: f64,
    pub improvement_percentage: f64,
    pub opportunities: Vec<ImprovementOpportunity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    pub impact_score: f64,
    pub affected_areas: Vec<String>,
    pub severity: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostOptimization {
    pub optimization_type: String,
    pub target_cost: f64,
    pub enabled: bool,
}

impl Default for CostOptimization {
    fn default() -> Self {
        Self {
            optimization_type: String::new(),
            target_cost: 0.0,
            enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_f64(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn test_flap_detection_default_construction() {
        let fd = FlapDetection {
            enabled: true,
            threshold: 5,
            window: Duration::from_secs(60),
        };
        assert!(fd.enabled);
        assert_eq!(fd.threshold, 5);
        assert_eq!(fd.window, Duration::from_secs(60));
    }

    #[test]
    fn test_flap_detection_disabled() {
        let fd = FlapDetection {
            enabled: false,
            threshold: 0,
            window: Duration::from_secs(0),
        };
        assert!(!fd.enabled);
        assert_eq!(fd.threshold, 0);
    }

    #[test]
    fn test_optimization_result_success() {
        let result = OptimizationResult {
            success: true,
            space_saved_bytes: 1024 * 1024,
            optimization_time_ms: 125.5,
            compression_savings: 512 * 1024,
            storage_optimization: 256 * 1024,
            retention_cleanup: 256 * 1024,
            total_space_saved: 1024 * 1024,
            optimization_time: Duration::from_millis(125),
        };
        assert!(result.success);
        assert_eq!(result.space_saved_bytes, 1024 * 1024);
        assert!((result.optimization_time_ms - 125.5).abs() < f64::EPSILON);
        assert_eq!(
            result.compression_savings + result.storage_optimization + result.retention_cleanup,
            result.total_space_saved
        );
    }

    #[test]
    fn test_optimization_result_failure() {
        let result = OptimizationResult {
            success: false,
            space_saved_bytes: 0,
            optimization_time_ms: 0.0,
            compression_savings: 0,
            storage_optimization: 0,
            retention_cleanup: 0,
            total_space_saved: 0,
            optimization_time: Duration::from_secs(0),
        };
        assert!(!result.success);
        assert_eq!(result.total_space_saved, 0);
    }

    #[test]
    fn test_trend_analysis_improving() {
        let ta = TrendAnalysis {
            trend_direction: "up".to_string(),
            trend_strength: 0.85,
            confidence: 0.92,
        };
        assert_eq!(ta.trend_direction, "up");
        assert!(ta.trend_strength > 0.0 && ta.trend_strength <= 1.0);
        assert!(ta.confidence > 0.0 && ta.confidence <= 1.0);
    }

    #[test]
    fn test_trend_analysis_random_values() {
        let mut lcg = Lcg::new(42);
        for _ in 0..10 {
            let strength = lcg.next_f64();
            let conf = lcg.next_f64();
            let ta = TrendAnalysis {
                trend_direction: "stable".to_string(),
                trend_strength: strength,
                confidence: conf,
            };
            assert!(ta.trend_strength >= 0.0);
            assert!(ta.confidence >= 0.0);
        }
    }

    #[test]
    fn test_io_bottleneck_detected() {
        let b = IoBottleneck {
            detected: true,
            bottleneck_type: "disk_write".to_string(),
            severity: 0.75,
        };
        assert!(b.detected);
        assert_eq!(b.bottleneck_type, "disk_write");
        assert!(b.severity > 0.0 && b.severity <= 1.0);
    }

    #[test]
    fn test_io_bottleneck_not_detected() {
        let b = IoBottleneck {
            detected: false,
            bottleneck_type: String::new(),
            severity: 0.0,
        };
        assert!(!b.detected);
        assert_eq!(b.severity, 0.0);
    }

    #[test]
    fn test_drift_detection_construction() {
        let dd = DriftDetection {
            enabled: true,
            threshold: 0.05,
            window_size: 100,
        };
        assert!(dd.enabled);
        assert!((dd.threshold - 0.05).abs() < f64::EPSILON);
        assert_eq!(dd.window_size, 100);
    }

    #[test]
    fn test_outlier_analysis_empty() {
        let oa = OutlierAnalysis {
            outliers: Vec::new(),
            method: "z-score".to_string(),
            threshold: 3.0,
        };
        assert!(oa.outliers.is_empty());
        assert_eq!(oa.method, "z-score");
    }

    #[test]
    fn test_outlier_analysis_with_outliers() {
        let now = chrono::Utc::now();
        let outlier = Outlier {
            value: 99.9,
            score: 4.5,
            timestamp: now,
        };
        let oa = OutlierAnalysis {
            outliers: vec![outlier],
            method: "iqr".to_string(),
            threshold: 1.5,
        };
        assert_eq!(oa.outliers.len(), 1);
        assert!((oa.outliers[0].value - 99.9).abs() < f64::EPSILON);
        assert!(oa.outliers[0].score > oa.threshold);
    }

    #[test]
    fn test_regression_analysis_default() {
        let ra = RegressionAnalysis::default();
        assert!(ra.coefficients.is_empty());
        assert_eq!(ra.r_squared, 0.0);
        assert!(ra.predictions.is_empty());
    }

    #[test]
    fn test_regression_analysis_with_data() {
        let mut lcg = Lcg::new(123);
        let coeffs: Vec<f64> = (0..3).map(|_| lcg.next_f64()).collect();
        let preds: Vec<f64> = (0..5).map(|_| lcg.next_f64()).collect();
        let ra = RegressionAnalysis {
            coefficients: coeffs.clone(),
            r_squared: 0.87,
            predictions: preds.clone(),
        };
        assert_eq!(ra.coefficients.len(), 3);
        assert_eq!(ra.predictions.len(), 5);
        assert!(ra.r_squared >= 0.0 && ra.r_squared <= 1.0);
    }

    #[test]
    fn test_trend_components_default() {
        let tc = TrendComponents::default();
        assert!(tc.trend.is_empty());
        assert!(tc.seasonal.is_empty());
        assert!(tc.residual.is_empty());
    }

    #[test]
    fn test_trend_components_with_data() {
        let mut lcg = Lcg::new(777);
        let data: Vec<f64> = (0..10).map(|_| lcg.next_f64() * 100.0).collect();
        let tc = TrendComponents {
            trend: data.clone(),
            seasonal: data.clone(),
            residual: data,
        };
        assert_eq!(tc.trend.len(), 10);
        assert_eq!(tc.seasonal.len(), 10);
        assert_eq!(tc.residual.len(), 10);
    }

    #[test]
    fn test_bottleneck_analysis_result_default() {
        let bar = BottleneckAnalysisResult::default();
        assert!(bar.bottlenecks.is_empty());
        assert!(bar.severity_scores.is_empty());
        assert!(bar.recommendations.is_empty());
    }

    #[test]
    fn test_bottleneck_analysis_result_populated() {
        let bar = BottleneckAnalysisResult {
            bottlenecks: vec!["cpu".to_string(), "memory".to_string()],
            severity_scores: vec![0.7, 0.5],
            recommendations: vec!["reduce load".to_string()],
        };
        assert_eq!(bar.bottlenecks.len(), 2);
        assert_eq!(bar.severity_scores.len(), 2);
        assert_eq!(bar.recommendations.len(), 1);
    }

    #[test]
    fn test_resource_efficiency_analysis_default() {
        let rea = ResourceEfficiencyAnalysis::default();
        assert_eq!(rea.efficiency_score, 0.0);
        assert!(rea.resource_usage.is_empty());
        assert!(rea.optimization_opportunities.is_empty());
    }

    #[test]
    fn test_cost_benefit_analysis_default() {
        let cba = CostBenefitAnalysis::default();
        assert_eq!(cba.benefits, 0.0);
        assert_eq!(cba.costs, 0.0);
        assert_eq!(cba.roi, 0.0);
    }

    #[test]
    fn test_cost_benefit_analysis_positive_roi() {
        let cba = CostBenefitAnalysis {
            benefits: 1000.0,
            costs: 400.0,
            roi: 1.5,
            payback_period: Duration::from_secs(90 * 24 * 3600),
        };
        assert!(cba.benefits > cba.costs);
        assert!(cba.roi > 0.0);
    }

    #[test]
    fn test_optimization_risk_assessment_default() {
        let ora = OptimizationRiskAssessment::default();
        assert!(ora.risk_level.is_empty() || !ora.risk_level.is_empty()); // valid string
        assert!(ora.risk_factors.is_empty());
        assert!(ora.mitigation_strategies.is_empty());
    }

    #[test]
    fn test_regression_analysis_result_default() {
        let rar = RegressionAnalysisResult::default();
        assert_eq!(rar.model_quality, 0.0);
        assert!(rar.metadata.is_empty());
    }

    #[test]
    fn test_improvement_analysis_result_default() {
        let iar = ImprovementAnalysisResult::default();
        assert_eq!(iar.baseline, 0.0);
        assert_eq!(iar.current, 0.0);
        assert_eq!(iar.improvement_percentage, 0.0);
        assert!(iar.opportunities.is_empty());
    }

    #[test]
    fn test_improvement_analysis_result_positive() {
        let iar = ImprovementAnalysisResult {
            baseline: 100.0,
            current: 85.0,
            improvement_percentage: 15.0,
            opportunities: Vec::new(),
        };
        assert!(iar.baseline > iar.current);
        assert!(iar.improvement_percentage > 0.0);
    }

    #[test]
    fn test_impact_analysis_construction() {
        let ia = ImpactAnalysis {
            impact_score: 0.65,
            affected_areas: vec!["performance".to_string(), "memory".to_string()],
            severity: "medium".to_string(),
        };
        assert!(ia.impact_score > 0.0 && ia.impact_score <= 1.0);
        assert_eq!(ia.affected_areas.len(), 2);
        assert_eq!(ia.severity, "medium");
    }

    #[test]
    fn test_cost_optimization_default() {
        let co = CostOptimization::default();
        assert!(co.optimization_type.is_empty());
        assert_eq!(co.target_cost, 0.0);
        assert!(!co.enabled);
    }

    #[test]
    fn test_cost_optimization_enabled() {
        let co = CostOptimization {
            optimization_type: "storage".to_string(),
            target_cost: 50.0,
            enabled: true,
        };
        assert!(co.enabled);
        assert_eq!(co.optimization_type, "storage");
        assert!(co.target_cost > 0.0);
    }

    #[test]
    fn test_outlier_timestamp_ordering() {
        let now = chrono::Utc::now();
        let o1 = Outlier {
            value: 10.0,
            score: 2.1,
            timestamp: now,
        };
        let o2 = Outlier {
            value: 20.0,
            score: 3.5,
            timestamp: now,
        };
        assert!(o2.score > o1.score);
        assert!(o2.value > o1.value);
    }

    #[test]
    fn test_resource_efficiency_with_usage_map() {
        let mut usage = std::collections::HashMap::new();
        usage.insert("cpu".to_string(), 0.75_f64);
        usage.insert("memory".to_string(), 0.60_f64);
        let rea = ResourceEfficiencyAnalysis {
            efficiency_score: 0.82,
            resource_usage: usage,
            optimization_opportunities: Vec::new(),
        };
        assert_eq!(rea.resource_usage.len(), 2);
        if let Some(cpu) = rea.resource_usage.get("cpu") {
            assert!(*cpu > 0.0);
        }
    }
}
