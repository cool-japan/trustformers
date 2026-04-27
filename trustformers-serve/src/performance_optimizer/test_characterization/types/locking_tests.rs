    use super::*;
    use std::collections::HashMap;
    use std::time::Duration;

    #[test]
    fn test_conflict_severity_variants() {
        let severities = vec![
            ConflictSeverity::Minor,
            ConflictSeverity::Moderate,
            ConflictSeverity::Major,
            ConflictSeverity::Severe,
            ConflictSeverity::Critical,
            ConflictSeverity::Blocking,
            ConflictSeverity::Fatal,
            ConflictSeverity::Low,
            ConflictSeverity::Medium,
            ConflictSeverity::High,
        ];
        assert_eq!(severities.len(), 10);
        assert_ne!(ConflictSeverity::Minor, ConflictSeverity::Fatal);
    }

    #[test]
    fn test_conflict_type_variants() {
        let types = vec![
            ConflictType::ResourceAccess,
            ConflictType::Data,
            ConflictType::Lock,
            ConflictType::Timing,
            ConflictType::Configuration,
            ConflictType::Memory,
            ConflictType::Io,
            ConflictType::Network,
            ConflictType::Database,
            ConflictType::Process,
            ConflictType::ReadWrite,
        ];
        assert_eq!(types.len(), 11);
    }

    #[test]
    fn test_contention_impact_variants() {
        let impacts = vec![
            ContentionImpact::Low,
            ContentionImpact::Medium,
            ContentionImpact::High,
            ContentionImpact::Critical,
        ];
        assert_eq!(impacts.len(), 4);
        assert_ne!(ContentionImpact::Low, ContentionImpact::Critical);
    }

    #[test]
    fn test_contention_severity_variants() {
        let sevs = vec![
            ContentionSeverity::Low,
            ContentionSeverity::Medium,
            ContentionSeverity::High,
            ContentionSeverity::Critical,
            ContentionSeverity::Severe,
        ];
        assert_eq!(sevs.len(), 5);
    }

    #[test]
    fn test_conflict_severity_equality() {
        let s1 = ConflictSeverity::Critical;
        let s2 = ConflictSeverity::Critical;
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_conflict_type_hash() {
        let mut map = HashMap::new();
        map.insert(ConflictType::Lock, "lock_handler");
        map.insert(ConflictType::Data, "data_handler");
        assert_eq!(map.len(), 2);
        if let Some(val) = map.get(&ConflictType::Lock) {
            assert_eq!(*val, "lock_handler");
        }
    }

    #[test]
    fn test_contention_impact_hash() {
        let mut map = HashMap::new();
        map.insert(ContentionImpact::High, 0.8);
        map.insert(ContentionImpact::Low, 0.2);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_conflict_severity_clone() {
        let severity = ConflictSeverity::Major;
        let cloned = severity;
        assert_eq!(severity, cloned);
    }

    #[test]
    fn test_conflict_type_clone() {
        let ct = ConflictType::Database;
        let cloned = ct;
        assert_eq!(ct, cloned);
    }

    #[test]
    fn test_conflict_type_debug() {
        let ct = ConflictType::Network;
        let debug_str = format!("{:?}", ct);
        assert!(debug_str.contains("Network"));
    }

    #[test]
    fn test_conflict_severity_debug() {
        let cs = ConflictSeverity::Blocking;
        let debug_str = format!("{:?}", cs);
        assert!(debug_str.contains("Blocking"));
    }

    #[test]
    fn test_contention_severity_debug() {
        let cs = ContentionSeverity::Severe;
        let debug_str = format!("{:?}", cs);
        assert!(debug_str.contains("Severe"));
    }

    #[test]
    fn test_lock_analysis_config_default() {
        let config = LockAnalysisConfig::default();
        assert!(config.enable_contention_analysis);
        assert!(config.enable_dependency_analysis);
        assert_eq!(config.max_analysis_duration, Duration::from_secs(30));
    }

    #[test]
    fn test_multiple_conflict_types_in_collection() {
        let types: Vec<ConflictType> = vec![
            ConflictType::ResourceAccess,
            ConflictType::Lock,
            ConflictType::Memory,
            ConflictType::Io,
        ];
        let unique: std::collections::HashSet<ConflictType> = types.into_iter().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn test_multiple_severities_in_collection() {
        let sevs: Vec<ConflictSeverity> = vec![
            ConflictSeverity::Minor,
            ConflictSeverity::Minor,
            ConflictSeverity::Major,
        ];
        let unique: std::collections::HashSet<ConflictSeverity> = sevs.into_iter().collect();
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn test_contention_impact_equality() {
        assert_eq!(ContentionImpact::Medium, ContentionImpact::Medium);
        assert_ne!(ContentionImpact::Low, ContentionImpact::High);
    }

    #[test]
    fn test_conflict_severity_all_differ() {
        let all = vec![
            ConflictSeverity::Minor,
            ConflictSeverity::Moderate,
            ConflictSeverity::Major,
            ConflictSeverity::Severe,
            ConflictSeverity::Critical,
            ConflictSeverity::Blocking,
            ConflictSeverity::Fatal,
        ];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j]);
            }
        }
    }

    #[test]
    fn test_lock_analysis_config_custom() {
        let config = LockAnalysisConfig {
            enable_contention_analysis: false,
            enable_dependency_analysis: true,
            max_analysis_duration: Duration::from_secs(60),
        };
        assert!(!config.enable_contention_analysis);
        assert!(config.enable_dependency_analysis);
        assert_eq!(config.max_analysis_duration.as_secs(), 60);
    }

    #[test]
    fn test_conflict_type_memory_vs_io() {
        let mem = ConflictType::Memory;
        let io = ConflictType::Io;
        assert_ne!(mem, io);
    }

    #[test]
    fn test_contention_severity_ordering_conceptual() {
        // Verify that all severity levels exist and are distinct
        let low = ContentionSeverity::Low;
        let med = ContentionSeverity::Medium;
        let high = ContentionSeverity::High;
        let critical = ContentionSeverity::Critical;
        let severe = ContentionSeverity::Severe;
        assert_ne!(low, med);
        assert_ne!(med, high);
        assert_ne!(high, critical);
        assert_ne!(critical, severe);
    }

    #[test]
    fn test_conflict_type_process() {
        let ct = ConflictType::Process;
        let debug = format!("{:?}", ct);
        assert_eq!(debug, "Process");
    }

    #[test]
    fn test_conflict_type_readwrite() {
        let ct = ConflictType::ReadWrite;
        let debug = format!("{:?}", ct);
        assert_eq!(debug, "ReadWrite");
    }

    #[test]
    fn test_conflict_severity_medium_alias() {
        // Medium is an alias for Moderate - they should be distinct enum variants
        assert_ne!(ConflictSeverity::Medium, ConflictSeverity::Moderate);
    }

    #[test]
    fn test_conflict_severity_low_alias() {
        // Low is an alias for Minor - they should be distinct enum variants
        assert_ne!(ConflictSeverity::Low, ConflictSeverity::Minor);
    }

    #[test]
    fn test_conflict_severity_high_alias() {
        // High is an alias for Major - they should be distinct enum variants
        assert_ne!(ConflictSeverity::High, ConflictSeverity::Major);
    }

    #[test]
    fn test_instant_now_helper() {
        let i1 = instant_now();
        let i2 = instant_now();
        // i2 should be same or after i1
        assert!(i2 >= i1);
    }

    #[test]
    fn test_duration_zero_helper() {
        let d = duration_zero();
        assert_eq!(d.as_secs(), 0);
        assert_eq!(d.as_nanos(), 0);
    }
