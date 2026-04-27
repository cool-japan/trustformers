//! Network optimization and adaptation tests for trustformers-mobile
//!
//! Tests bandwidth tiers, download config, compression algorithms,
//! and offline-first strategies.

use trustformers_mobile::{
    BandwidthThresholds, BandwidthTier, CacheEvictionStrategy, CompressionAlgorithm,
    DownloadCompressionConfig, DownloadRetryConfig, OfflineFirstConfig, OfflineSyncStrategy,
    QualityAdaptationStrategy,
};

#[test]
fn test_compression_algorithm_variants_exist() {
    let _zstd = CompressionAlgorithm::Zstd;
    let _lz4 = CompressionAlgorithm::LZ4;
    let _gzip = CompressionAlgorithm::Gzip;
    let _brotli = CompressionAlgorithm::Brotli;
    let _none = CompressionAlgorithm::None;
}

#[test]
fn test_bandwidth_tier_variants_exist() {
    let _very_low = BandwidthTier::VeryLow;
    let _low = BandwidthTier::Low;
    let _medium = BandwidthTier::Medium;
    let _high = BandwidthTier::High;
    let _ultra_high = BandwidthTier::UltraHigh;
}

#[test]
fn test_bandwidth_thresholds_ordering() {
    let thresholds = BandwidthThresholds {
        low_bandwidth_kbps: 100.0,
        medium_bandwidth_kbps: 1000.0,
        high_bandwidth_kbps: 10000.0,
        ultra_high_bandwidth_kbps: 100000.0,
    };
    assert!(thresholds.low_bandwidth_kbps < thresholds.medium_bandwidth_kbps);
    assert!(thresholds.medium_bandwidth_kbps < thresholds.high_bandwidth_kbps);
    assert!(thresholds.high_bandwidth_kbps < thresholds.ultra_high_bandwidth_kbps);
}

#[test]
fn test_quality_adaptation_strategy_variants_exist() {
    let _conservative = QualityAdaptationStrategy::Conservative;
    let _aggressive = QualityAdaptationStrategy::Aggressive;
    let _balanced = QualityAdaptationStrategy::Balanced;
    let _manual = QualityAdaptationStrategy::Manual;
}

#[test]
fn test_offline_sync_strategy_variants_exist() {
    let _immediate = OfflineSyncStrategy::Immediate;
    let _opportunistic = OfflineSyncStrategy::Opportunistic;
    let _manual = OfflineSyncStrategy::Manual;
    let _background = OfflineSyncStrategy::Background;
    let _adaptive = OfflineSyncStrategy::Adaptive;
}

#[test]
fn test_download_retry_config_backoff_multiplier_positive() {
    let config = DownloadRetryConfig {
        max_retries: 3,
        initial_delay_ms: 1000.0,
        max_delay_ms: 60000.0,
        backoff_multiplier: 2.0,
        jitter_factor: 0.1,
    };
    assert!(
        config.backoff_multiplier > 1.0,
        "backoff multiplier should be > 1.0 for exponential backoff"
    );
}

#[test]
fn test_download_retry_config_max_delay_exceeds_initial() {
    let config = DownloadRetryConfig {
        max_retries: 3,
        initial_delay_ms: 1000.0,
        max_delay_ms: 60000.0,
        backoff_multiplier: 2.0,
        jitter_factor: 0.1,
    };
    assert!(config.max_delay_ms > config.initial_delay_ms);
}

#[test]
fn test_download_compression_config_creation() {
    let config = DownloadCompressionConfig {
        enable_compression: true,
        preferred_algorithms: vec![CompressionAlgorithm::Zstd, CompressionAlgorithm::Gzip],
        min_size_for_compression: 10240,
        enable_streaming_decompression: true,
    };
    assert!(config.enable_compression);
    assert!(!config.preferred_algorithms.is_empty());
    assert!(config.min_size_for_compression > 0);
}

#[test]
fn test_offline_first_config_creation() {
    let config = OfflineFirstConfig {
        enable_offline_mode: true,
        offline_cache_size_mb: 256,
        fallback_models: vec!["bert-tiny".to_string()],
        sync_strategy: OfflineSyncStrategy::Opportunistic,
        offline_retention: trustformers_mobile::OfflineRetentionPolicy {
            model_retention_days: 7,
            cache_retention_hours: 24,
            auto_cleanup_on_low_storage: true,
            min_storage_threshold_mb: 100,
        },
    };
    assert!(config.enable_offline_mode);
    assert!(config.offline_cache_size_mb > 0);
    assert!(!config.fallback_models.is_empty());
}

#[test]
fn test_cache_eviction_strategy_variants_exist() {
    let _lru = CacheEvictionStrategy::LRU;
    let _lfu = CacheEvictionStrategy::LFU;
    let _fifo = CacheEvictionStrategy::FIFO;
    let _ttl = CacheEvictionStrategy::TTL;
    let _size = CacheEvictionStrategy::SizeBased;
}

#[test]
fn test_bandwidth_threshold_serialization_roundtrip() {
    let thresholds = BandwidthThresholds {
        low_bandwidth_kbps: 100.0,
        medium_bandwidth_kbps: 1000.0,
        high_bandwidth_kbps: 10000.0,
        ultra_high_bandwidth_kbps: 100000.0,
    };
    let json = serde_json::to_string(&thresholds).expect("serialization should succeed");
    let restored: BandwidthThresholds =
        serde_json::from_str(&json).expect("deserialization should succeed");
    assert!((restored.low_bandwidth_kbps - thresholds.low_bandwidth_kbps).abs() < 1e-6);
    assert!(
        (restored.ultra_high_bandwidth_kbps - thresholds.ultra_high_bandwidth_kbps).abs() < 1e-6
    );
}

#[test]
fn test_bandwidth_tier_hash_usable_as_map_key() {
    use std::collections::HashMap;
    let mut map: HashMap<BandwidthTier, &str> = HashMap::new();
    map.insert(BandwidthTier::Low, "low");
    map.insert(BandwidthTier::High, "high");
    assert_eq!(map.len(), 2);
    assert_eq!(map[&BandwidthTier::Low], "low");
}
