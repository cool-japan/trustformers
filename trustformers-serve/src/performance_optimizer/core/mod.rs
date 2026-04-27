//! Auto-generated module structure

pub mod simplelinearregression_traits;
pub mod types;
pub mod functions;

// Re-export all types
pub use simplelinearregression_traits::*;
pub use types::*;
pub use functions::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Duration;

    // ── OptimizationResult ────────────────────────────────────────────────

    #[test]
    fn test_optimization_result_successful() {
        let result = OptimizationResult {
            applied_optimizations: vec!["cache_warmup".to_string(), "thread_pool_resize".to_string()],
            performance_improvement: 0.35_f32,
            application_duration: Duration::from_millis(120),
            success: true,
            details: HashMap::new(),
        };
        assert!(result.success);
        assert_eq!(result.applied_optimizations.len(), 2);
        assert!(result.performance_improvement > 0.0);
    }

    #[test]
    fn test_optimization_result_failed() {
        let result = OptimizationResult {
            applied_optimizations: vec![],
            performance_improvement: 0.0_f32,
            application_duration: Duration::from_millis(5),
            success: false,
            details: {
                let mut m = HashMap::new();
                m.insert("error".to_string(), "resource unavailable".to_string());
                m
            },
        };
        assert!(!result.success);
        assert!(result.applied_optimizations.is_empty());
        assert_eq!(result.performance_improvement, 0.0);
        assert!(result.details.contains_key("error"));
    }

    #[test]
    fn test_optimization_result_details_map() {
        let mut details = HashMap::new();
        details.insert("cpu_cores_used".to_string(), "8".to_string());
        details.insert("memory_mb".to_string(), "2048".to_string());
        let result = OptimizationResult {
            applied_optimizations: vec!["numa_binding".to_string()],
            performance_improvement: 0.12_f32,
            application_duration: Duration::from_millis(45),
            success: true,
            details,
        };
        assert_eq!(result.details.len(), 2);
        assert_eq!(result.details.get("cpu_cores_used").map(|s| s.as_str()), Some("8"));
    }

    #[test]
    fn test_optimization_result_zero_duration() {
        let result = OptimizationResult {
            applied_optimizations: vec!["noop".to_string()],
            performance_improvement: 0.001_f32,
            application_duration: Duration::ZERO,
            success: true,
            details: HashMap::new(),
        };
        assert_eq!(result.application_duration, Duration::ZERO);
    }

    #[test]
    fn test_optimization_result_many_optimizations() {
        let names: Vec<String> = (0..12).map(|i| format!("opt_{}", i)).collect();
        let count = names.len();
        let result = OptimizationResult {
            applied_optimizations: names,
            performance_improvement: 0.85_f32,
            application_duration: Duration::from_secs(1),
            success: true,
            details: HashMap::new(),
        };
        assert_eq!(result.applied_optimizations.len(), count);
    }

    #[test]
    fn test_optimization_result_high_improvement() {
        let result = OptimizationResult {
            applied_optimizations: vec!["full_pipeline_rewrite".to_string()],
            performance_improvement: 3.5_f32,
            application_duration: Duration::from_secs(30),
            success: true,
            details: HashMap::new(),
        };
        // 3.5x improvement is possible in ideal conditions
        assert!(result.performance_improvement > 1.0);
    }

    #[test]
    fn test_optimization_result_clone() {
        let result = OptimizationResult {
            applied_optimizations: vec!["batching".to_string()],
            performance_improvement: 0.2_f32,
            application_duration: Duration::from_millis(80),
            success: true,
            details: HashMap::new(),
        };
        let cloned = result.clone();
        assert_eq!(cloned.applied_optimizations, result.applied_optimizations);
        assert_eq!(cloned.success, result.success);
    }

    #[test]
    fn test_optimization_result_debug() {
        let result = OptimizationResult {
            applied_optimizations: vec!["debug_test".to_string()],
            performance_improvement: 0.1_f32,
            application_duration: Duration::from_millis(10),
            success: true,
            details: HashMap::new(),
        };
        let dbg = format!("{:?}", result);
        assert!(dbg.contains("OptimizationResult"));
        assert!(dbg.contains("debug_test"));
    }

    #[test]
    fn test_optimization_result_partial_success() {
        // success = false but some optimizations were attempted
        let result = OptimizationResult {
            applied_optimizations: vec!["partial_opt".to_string()],
            performance_improvement: 0.05_f32,
            application_duration: Duration::from_millis(200),
            success: false,
            details: {
                let mut m = HashMap::new();
                m.insert("partial".to_string(), "true".to_string());
                m
            },
        };
        assert!(!result.success);
        assert!(!result.applied_optimizations.is_empty());
        assert!(result.performance_improvement > 0.0);
    }

    #[test]
    fn test_optimization_result_duration_precision() {
        let dur = Duration::from_nanos(123_456_789);
        let result = OptimizationResult {
            applied_optimizations: vec![],
            performance_improvement: 0.0,
            application_duration: dur,
            success: false,
            details: HashMap::new(),
        };
        assert_eq!(result.application_duration.subsec_nanos(), 123_456_789);
    }

    #[test]
    fn test_optimization_result_multiple_details() {
        let mut details = HashMap::new();
        for i in 0..10_usize {
            details.insert(format!("key_{}", i), format!("val_{}", i));
        }
        let result = OptimizationResult {
            applied_optimizations: vec!["stress_test".to_string()],
            performance_improvement: 0.5,
            application_duration: Duration::from_millis(300),
            success: true,
            details,
        };
        assert_eq!(result.details.len(), 10);
    }

    #[test]
    fn test_optimization_result_applied_list_is_ordered() {
        // Verify that insertion order is preserved via Vec
        let optimizations = vec![
            "step_1".to_string(),
            "step_2".to_string(),
            "step_3".to_string(),
        ];
        let result = OptimizationResult {
            applied_optimizations: optimizations.clone(),
            performance_improvement: 0.25,
            application_duration: Duration::from_millis(60),
            success: true,
            details: HashMap::new(),
        };
        assert_eq!(result.applied_optimizations[0], "step_1");
        assert_eq!(result.applied_optimizations[1], "step_2");
        assert_eq!(result.applied_optimizations[2], "step_3");
    }
}
