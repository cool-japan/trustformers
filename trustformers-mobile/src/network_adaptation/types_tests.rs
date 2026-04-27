#[cfg(test)]
mod tests {
    use crate::network_adaptation::types::*;
    use std::collections::HashMap;
    use std::time::Instant;

    // --- NetworkQuality Tests ---

    #[test]
    fn test_network_quality_ordering() {
        assert!(NetworkQuality::Poor < NetworkQuality::Fair);
        assert!(NetworkQuality::Fair < NetworkQuality::Good);
        assert!(NetworkQuality::Good < NetworkQuality::Excellent);
    }

    #[test]
    fn test_network_quality_equality() {
        assert_eq!(NetworkQuality::Good, NetworkQuality::Good);
        assert_ne!(NetworkQuality::Poor, NetworkQuality::Excellent);
    }

    #[test]
    fn test_network_quality_serialization() {
        let quality = NetworkQuality::Excellent;
        let json = serde_json::to_string(&quality).expect("Failed to serialize");
        let deserialized: NetworkQuality =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, NetworkQuality::Excellent);
    }

    // --- TrendDirection Tests ---

    #[test]
    fn test_trend_direction_variants() {
        assert_eq!(TrendDirection::Improving, TrendDirection::Improving);
        assert_ne!(TrendDirection::Improving, TrendDirection::Degrading);
        let _ = TrendDirection::Stable;
        let _ = TrendDirection::Volatile;
    }

    #[test]
    fn test_trend_direction_serialization() {
        let trend = TrendDirection::Volatile;
        let json = serde_json::to_string(&trend).expect("Failed to serialize");
        let deserialized: TrendDirection =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, TrendDirection::Volatile);
    }

    // --- TimePeriod Tests ---

    #[test]
    fn test_time_period_variants() {
        assert_eq!(TimePeriod::Hourly, TimePeriod::Hourly);
        assert_ne!(TimePeriod::Daily, TimePeriod::Weekly);
        let _ = TimePeriod::Monthly;
        let _ = TimePeriod::Seasonal;
    }

    #[test]
    fn test_time_period_serialization() {
        let period = TimePeriod::Weekly;
        let json = serde_json::to_string(&period).expect("Failed to serialize");
        let deserialized: TimePeriod = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, TimePeriod::Weekly);
    }

    // --- NetworkConditions Tests ---

    #[test]
    fn test_network_conditions_creation() {
        let conditions = NetworkConditions {
            timestamp: Instant::now(),
            connection_type: crate::profiler::NetworkConnectionType::WiFi,
            bandwidth_mbps: 50.0,
            latency_ms: 20.0,
            packet_loss_percent: 0.1,
            signal_strength_dbm: Some(-50),
            jitter_ms: 2.0,
            stability_score: 0.95,
            quality_assessment: NetworkQuality::Excellent,
            available_data_mb: Some(5000),
        };
        assert!((conditions.bandwidth_mbps - 50.0).abs() < 1e-5);
        assert_eq!(conditions.quality_assessment, NetworkQuality::Excellent);
        assert_eq!(conditions.signal_strength_dbm, Some(-50));
    }

    #[test]
    fn test_network_conditions_clone() {
        let conditions = NetworkConditions {
            timestamp: Instant::now(),
            connection_type: crate::profiler::NetworkConnectionType::Cellular4G,
            bandwidth_mbps: 30.0,
            latency_ms: 40.0,
            packet_loss_percent: 0.5,
            signal_strength_dbm: None,
            jitter_ms: 5.0,
            stability_score: 0.8,
            quality_assessment: NetworkQuality::Good,
            available_data_mb: None,
        };
        let cloned = conditions.clone();
        assert!((cloned.bandwidth_mbps - 30.0).abs() < 1e-5);
        assert_eq!(cloned.quality_assessment, NetworkQuality::Good);
    }

    // --- NetworkPattern Tests ---

    #[test]
    fn test_network_pattern_creation() {
        let pattern = NetworkPattern {
            time_period: TimePeriod::Daily,
            expected_bandwidth_mbps: 100.0,
            expected_latency_ms: 10.0,
            expected_stability: 0.9,
            confidence: 0.85,
        };
        assert!((pattern.expected_bandwidth_mbps - 100.0).abs() < 1e-5);
        assert!((pattern.confidence - 0.85).abs() < 1e-5);
    }

    #[test]
    fn test_network_pattern_serialization() {
        let pattern = NetworkPattern {
            time_period: TimePeriod::Hourly,
            expected_bandwidth_mbps: 50.0,
            expected_latency_ms: 25.0,
            expected_stability: 0.75,
            confidence: 0.6,
        };
        let json = serde_json::to_string(&pattern).expect("Failed to serialize");
        let deserialized: NetworkPattern =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert!((deserialized.expected_bandwidth_mbps - 50.0).abs() < 1e-5);
    }

    // --- ConflictResolutionStrategy Tests ---

    #[test]
    fn test_conflict_resolution_strategy_variants() {
        let _ = ConflictResolutionStrategy::LastWriterWins;
        let _ = ConflictResolutionStrategy::MergeConflicts;
        let _ = ConflictResolutionStrategy::UserDecision;
        let _ = ConflictResolutionStrategy::ServerDecision;
        let _ = ConflictResolutionStrategy::VersionVector;
    }

    // --- MergeAlgorithm Tests ---

    #[test]
    fn test_merge_algorithm_variants() {
        let _ = MergeAlgorithm::AverageMerge;
        let _ = MergeAlgorithm::WeightedMerge;
        let _ = MergeAlgorithm::SelectiveMerge;
        let _ = MergeAlgorithm::AdaptiveMerge;
    }

    // --- ChecksumAlgorithm Tests ---

    #[test]
    fn test_checksum_algorithm_variants() {
        let _ = ChecksumAlgorithm::CRC32;
        let _ = ChecksumAlgorithm::SHA256;
        let _ = ChecksumAlgorithm::MD5;
        let _ = ChecksumAlgorithm::Custom;
    }

    // --- SyncStrategy Tests ---

    #[test]
    fn test_sync_strategy_variants() {
        let _ = SyncStrategy::Adaptive;
        let _ = format!("{:?}", SyncStrategy::Adaptive);
    }

    // --- NetworkPredictionConfig Tests ---

    #[test]
    fn test_network_prediction_config_default() {
        let config = NetworkPredictionConfig::default();
        let _ = format!("{:?}", config);
    }
}
