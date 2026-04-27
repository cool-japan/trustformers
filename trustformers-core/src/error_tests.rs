// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for error types and error context.

#[cfg(test)]
mod tests {
    use crate::error::{ErrorCode, ErrorContext, PerformanceInfo};
    use std::time::Duration;

    #[test]
    fn test_error_code_e1001() {
        assert_eq!(ErrorCode::E1001.code(), "E1001");
    }

    #[test]
    fn test_error_code_e2001() {
        assert_eq!(ErrorCode::E2001.code(), "E2001");
    }

    #[test]
    fn test_error_code_e3001() {
        assert_eq!(ErrorCode::E3001.code(), "E3001");
    }

    #[test]
    fn test_error_code_e4001() {
        assert_eq!(ErrorCode::E4001.code(), "E4001");
    }

    #[test]
    fn test_error_code_e5001() {
        assert_eq!(ErrorCode::E5001.code(), "E5001");
    }

    #[test]
    fn test_error_code_e6001() {
        assert_eq!(ErrorCode::E6001.code(), "E6001");
    }

    #[test]
    fn test_error_code_e9001() {
        assert_eq!(ErrorCode::E9001.code(), "E9001");
    }

    #[test]
    fn test_error_code_suggestions_e1001() {
        let suggestions = ErrorCode::E1001.suggestions();
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("dimensions")));
    }

    #[test]
    fn test_error_code_suggestions_e3001() {
        let suggestions = ErrorCode::E3001.suggestions();
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("batch size") || s.contains("memory")));
    }

    #[test]
    fn test_error_code_suggestions_e5001() {
        let suggestions = ErrorCode::E5001.suggestions();
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("GPU") || s.contains("driver")));
    }

    #[test]
    fn test_error_code_suggestions_default() {
        let suggestions = ErrorCode::E9003.suggestions();
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_error_context_new() {
        let ctx = ErrorContext::new(ErrorCode::E1001, "test_op".to_string());
        assert_eq!(ctx.code, ErrorCode::E1001);
        assert_eq!(ctx.operation, "test_op");
        assert!(ctx.context.is_empty());
        assert!(ctx.performance_info.is_none());
        assert!(!ctx.recovery_suggestions.is_empty());
    }

    #[test]
    fn test_error_context_with_context() {
        let ctx = ErrorContext::new(ErrorCode::E2001, "op".to_string())
            .with_context("additional info".to_string());
        assert_eq!(ctx.context.len(), 1);
        assert_eq!(ctx.context[0], "additional info");
    }

    #[test]
    fn test_error_context_multiple_contexts() {
        let ctx = ErrorContext::new(ErrorCode::E2001, "op".to_string())
            .with_context("first".to_string())
            .with_context("second".to_string());
        assert_eq!(ctx.context.len(), 2);
    }

    #[test]
    fn test_error_context_with_performance() {
        let perf = PerformanceInfo {
            memory_usage_mb: Some(1024),
            operation_duration: Some(Duration::from_millis(500)),
            gpu_utilization: Some(75.0),
            cpu_usage: Some(50.0),
        };
        let ctx =
            ErrorContext::new(ErrorCode::E6001, "heavy_op".to_string()).with_performance(perf);
        assert!(ctx.performance_info.is_some());
        if let Some(ref p) = ctx.performance_info {
            assert_eq!(p.memory_usage_mb, Some(1024));
            assert_eq!(p.gpu_utilization, Some(75.0));
        }
    }

    #[test]
    fn test_error_context_add_recovery_suggestion() {
        let ctx = ErrorContext::new(ErrorCode::E1001, "op".to_string())
            .add_recovery_suggestion("Try reducing batch size".to_string());
        assert!(ctx.recovery_suggestions.iter().any(|s| s.contains("batch size")));
    }

    #[test]
    fn test_performance_info_partial_eq() {
        let p1 = PerformanceInfo {
            memory_usage_mb: Some(512),
            operation_duration: None,
            gpu_utilization: None,
            cpu_usage: Some(25.0),
        };
        let p2 = PerformanceInfo {
            memory_usage_mb: Some(512),
            operation_duration: None,
            gpu_utilization: None,
            cpu_usage: Some(25.0),
        };
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_performance_info_not_equal() {
        let p1 = PerformanceInfo {
            memory_usage_mb: Some(512),
            operation_duration: None,
            gpu_utilization: None,
            cpu_usage: None,
        };
        let p2 = PerformanceInfo {
            memory_usage_mb: Some(1024),
            operation_duration: None,
            gpu_utilization: None,
            cpu_usage: None,
        };
        assert_ne!(p1, p2);
    }

    #[test]
    fn test_performance_info_clone() {
        let perf = PerformanceInfo {
            memory_usage_mb: Some(256),
            operation_duration: Some(Duration::from_secs(1)),
            gpu_utilization: Some(90.0),
            cpu_usage: Some(100.0),
        };
        let cloned = perf.clone();
        assert_eq!(cloned, perf);
    }

    #[test]
    fn test_error_context_partial_eq() {
        let ctx1 = ErrorContext::new(ErrorCode::E1001, "op".to_string());
        let ctx2 = ErrorContext::new(ErrorCode::E1001, "op".to_string());
        assert_eq!(ctx1, ctx2);
    }

    #[test]
    fn test_error_context_not_equal_different_code() {
        let ctx1 = ErrorContext::new(ErrorCode::E1001, "op".to_string());
        let ctx2 = ErrorContext::new(ErrorCode::E2001, "op".to_string());
        assert_ne!(ctx1, ctx2);
    }

    #[test]
    fn test_error_context_not_equal_different_op() {
        let ctx1 = ErrorContext::new(ErrorCode::E1001, "op1".to_string());
        let ctx2 = ErrorContext::new(ErrorCode::E1001, "op2".to_string());
        assert_ne!(ctx1, ctx2);
    }

    #[test]
    fn test_error_code_all_codes_unique() {
        let codes = [
            ErrorCode::E1001,
            ErrorCode::E1002,
            ErrorCode::E1003,
            ErrorCode::E2001,
            ErrorCode::E2002,
            ErrorCode::E2003,
            ErrorCode::E3001,
            ErrorCode::E3002,
            ErrorCode::E3003,
            ErrorCode::E4001,
            ErrorCode::E4002,
            ErrorCode::E4003,
            ErrorCode::E5001,
            ErrorCode::E5002,
            ErrorCode::E5003,
            ErrorCode::E6001,
            ErrorCode::E6002,
            ErrorCode::E6003,
            ErrorCode::E9001,
            ErrorCode::E9002,
            ErrorCode::E9003,
        ];
        let mut code_strings: Vec<&str> = codes.iter().map(|c| c.code()).collect();
        let len_before = code_strings.len();
        code_strings.sort();
        code_strings.dedup();
        assert_eq!(code_strings.len(), len_before);
    }

    #[test]
    fn test_error_code_debug() {
        let code = ErrorCode::E1001;
        let debug_str = format!("{:?}", code);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_error_code_clone() {
        let code = ErrorCode::E2001;
        let cloned = code;
        assert_eq!(code, cloned);
    }

    #[test]
    fn test_error_code_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ErrorCode::E1001);
        set.insert(ErrorCode::E2001);
        set.insert(ErrorCode::E1001);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_performance_info_all_none() {
        let perf = PerformanceInfo {
            memory_usage_mb: None,
            operation_duration: None,
            gpu_utilization: None,
            cpu_usage: None,
        };
        assert!(perf.memory_usage_mb.is_none());
        assert!(perf.operation_duration.is_none());
    }

    #[test]
    fn test_error_context_debug() {
        let ctx = ErrorContext::new(ErrorCode::E3001, "alloc".to_string());
        let debug_str = format!("{:?}", ctx);
        assert!(!debug_str.is_empty());
    }
}
