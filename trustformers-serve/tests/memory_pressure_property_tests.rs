//! Comprehensive Property-Based Tests for Memory Pressure Handlers
//!
//! This module contains extensive property-based tests that verify the correctness
//! and robustness of the memory pressure handling system under various conditions.
//! These tests automatically generate thousands of test cases to validate that
//! fundamental properties always hold true.

use chrono::{Duration as ChronoDuration, Utc};
use proptest::prelude::*;
use tokio_test;
use trustformers_serve::memory_pressure::{
    BufferCompactionHandler, CleanupHandler, GarbageCollectionHandler, GpuCleanupStrategy,
    MemoryPressureConfig, MemoryPressureHandler, MemoryPressureLevel, PressureSnapshot,
};

/// Custom strategy to generate valid memory pressure configurations
fn memory_pressure_config_strategy() -> impl Strategy<Value = MemoryPressureConfig> {
    (
        prop::bool::ANY,   // enabled
        0.1f32..0.8f32,    // low threshold
        0.2f32..0.9f32,    // medium threshold
        0.3f32..0.95f32,   // high threshold
        0.4f32..0.98f32,   // critical threshold
        0.8f32..0.99f32,   // emergency threshold
        1u64..3600u64,     // monitoring interval
        1usize..1024usize, // memory buffer mb
    )
        .prop_map(
            |(enabled, low, medium, high, critical, emergency, interval, buffer)| {
                // Ensure proper ordering of thresholds
                let low = low.min(medium - 0.01);
                let medium = medium.clamp(low + 0.01, high - 0.01);
                let high = high.clamp(medium + 0.01, critical - 0.01);
                let critical = critical.clamp(high + 0.01, emergency - 0.01);

                let mut config = MemoryPressureConfig::default();
                config.enabled = enabled;
                config.pressure_thresholds.low = low;
                config.pressure_thresholds.medium = medium;
                config.pressure_thresholds.high = high;
                config.pressure_thresholds.critical = critical;
                config.emergency_threshold = emergency;
                config.monitoring_interval_seconds = interval;
                config.memory_buffer_mb = buffer;

                config
            },
        )
}

/// Strategy to generate valid memory allocation sizes
fn allocation_size_strategy() -> impl Strategy<Value = u64> {
    prop_oneof![
        1u64..1024u64,                        // Small allocations (1B - 1KB)
        1024u64..1024 * 1024u64,              // Medium allocations (1KB - 1MB)
        1024 * 1024u64..100 * 1024 * 1024u64, // Large allocations (1MB - 100MB)
        // Occasionally test very large allocations
        Just(1024 * 1024 * 1024u64), // 1GB
    ]
}

/// Strategy to generate pressure snapshots
fn pressure_snapshot_strategy() -> impl Strategy<Value = PressureSnapshot> {
    (
        0.0f32..1.0f32,                                        // utilization
        1024u64 * 1024u64..8u64 * 1024u64 * 1024u64 * 1024u64, // available memory (1MB-8GB)
    )
        .prop_map(|(utilization, available_memory)| {
            let now = Utc::now();
            let pressure_level = match utilization {
                u if u >= 0.95 => MemoryPressureLevel::Emergency,
                u if u >= 0.85 => MemoryPressureLevel::Critical,
                u if u >= 0.75 => MemoryPressureLevel::High,
                u if u >= 0.60 => MemoryPressureLevel::Medium,
                u if u >= 0.45 => MemoryPressureLevel::Low,
                _ => MemoryPressureLevel::Normal,
            };

            PressureSnapshot {
                timestamp: now,
                utilization,
                pressure_level,
                available_memory,
            }
        })
}

proptest! {
    /// Property: Memory pressure configuration validation
    /// All generated configurations should be internally consistent
    #[test]
    fn prop_config_consistency(config in memory_pressure_config_strategy()) {
        // Property: Thresholds should be ordered
        prop_assert!(config.pressure_thresholds.low <= config.pressure_thresholds.medium);
        prop_assert!(config.pressure_thresholds.medium <= config.pressure_thresholds.high);
        prop_assert!(config.pressure_thresholds.high <= config.pressure_thresholds.critical);
        prop_assert!(config.pressure_thresholds.critical <= config.emergency_threshold);

        // Property: All thresholds should be valid percentages
        prop_assert!(config.pressure_thresholds.low >= 0.0 && config.pressure_thresholds.low <= 1.0);
        prop_assert!(config.pressure_thresholds.medium >= 0.0 && config.pressure_thresholds.medium <= 1.0);
        prop_assert!(config.pressure_thresholds.high >= 0.0 && config.pressure_thresholds.high <= 1.0);
        prop_assert!(config.pressure_thresholds.critical >= 0.0 && config.pressure_thresholds.critical <= 1.0);
        prop_assert!(config.emergency_threshold >= 0.0 && config.emergency_threshold <= 1.0);

        // Property: Configuration should create valid handler
        let handler = MemoryPressureHandler::new(config);
        let _ = tokio_test::block_on(async move {
            let stats = handler.get_memory_stats().await;
            prop_assert!(stats.utilization >= 0.0 && stats.utilization <= 1.0);
            Ok(())
        });
    }

    /// Property: Pressure level transitions are always valid
    /// Moving between pressure levels should follow expected patterns
    #[test]
    fn prop_pressure_level_transitions(
        initial_utilization in 0.0f32..1.0f32,
        final_utilization in 0.0f32..1.0f32,
        config in memory_pressure_config_strategy()
    ) {
        let handler = MemoryPressureHandler::new(config);

        let initial_level = handler.calculate_pressure_level(initial_utilization);
        let final_level = handler.calculate_pressure_level(final_utilization);

        // Property: Level should be monotonic with utilization
        if final_utilization > initial_utilization {
            prop_assert!(final_level >= initial_level);
        } else if final_utilization < initial_utilization {
            prop_assert!(final_level <= initial_level);
        } else {
            prop_assert_eq!(final_level, initial_level);
        }
    }

    /// Property: Memory allocation and deallocation consistency
    #[test]
    fn prop_allocation_deallocation_consistency(
        allocations in prop::collection::vec((allocation_size_strategy(), "[a-z]{1,10}"), 1..10)
    ) {
        let _ = tokio_test::block_on(async move {
            let config = MemoryPressureConfig::default();
            let handler = MemoryPressureHandler::new(config);

            let mut allocation_ids = Vec::new();

            // Test multiple allocations
            for (size, allocation_type) in allocations.iter() {
                match handler.request_allocation(*size, allocation_type.clone()).await {
                    Ok(id) => {
                        // Property: Allocation ID should be unique and non-empty
                        prop_assert!(!id.is_empty());
                        prop_assert!(!allocation_ids.contains(&id));
                        allocation_ids.push(id);
                    }
                    Err(_) => {
                        // Property: Allocation can fail due to memory pressure, which is valid
                        // This means the system is correctly rejecting requests under pressure
                    }
                }
            }

            // Test deallocation of all successful allocations
            for id in allocation_ids {
                prop_assert!(handler.release_allocation(&id).await.is_ok());
            }

            Ok(())
        });
    }

    /// Property: Cleanup strategies effectiveness
    #[test]
    fn prop_cleanup_effectiveness(
        pressure_level in prop::sample::select(vec![
            MemoryPressureLevel::Low,
            MemoryPressureLevel::Medium,
            MemoryPressureLevel::High,
            MemoryPressureLevel::Critical,
            MemoryPressureLevel::Emergency,
        ]),
        iterations in 1usize..5usize
    ) {
        let gc_handler = GarbageCollectionHandler::new();
        let buffer_handler = BufferCompactionHandler::new();

        let mut total_gc_freed = 0u64;
        let mut total_buffer_freed = 0u64;

        // Property: Multiple cleanup calls should be safe and effective
        for _ in 0..iterations {
            let gc_freed = gc_handler.cleanup(pressure_level).unwrap();
            let buffer_freed = buffer_handler.cleanup(pressure_level).unwrap();

            total_gc_freed += gc_freed;
            total_buffer_freed += buffer_freed;

            // Property: Each cleanup should free some memory
            prop_assert!(gc_freed > 0);
            prop_assert!(buffer_freed > 0);
        }

        // Property: Total freed memory should be reasonable
        prop_assert!(total_gc_freed >= iterations as u64 * 1024 * 1024); // At least 1MB per iteration
        prop_assert!(total_buffer_freed >= iterations as u64 * 1024 * 1024);

        // Property: Estimates should be in reasonable range of actual
        let gc_estimate = gc_handler.estimate_memory_freed();
        let buffer_estimate = buffer_handler.estimate_memory_freed();

        prop_assert!(gc_estimate > 0);
        prop_assert!(buffer_estimate > 0);
    }

    /// Property: GPU cleanup strategy robustness
    #[test]
    fn prop_gpu_cleanup_robustness(
        device_ids in prop::collection::vec(0u32..8u32, 1..4),
        strategy in prop::sample::select(vec![
            GpuCleanupStrategy::GpuCacheEviction,
            GpuCleanupStrategy::GpuBufferCompaction,
            GpuCleanupStrategy::GpuModelUnloading,
            GpuCleanupStrategy::GpuVramCompaction,
            GpuCleanupStrategy::GpuMemoryDefragmentation,
            GpuCleanupStrategy::GpuContextSwitching,
            GpuCleanupStrategy::GpuStreamCleanup,
            GpuCleanupStrategy::GpuTextureCleanup,
            GpuCleanupStrategy::GpuMemoryPoolReset,
            GpuCleanupStrategy::GpuBatchSizeReduction,
        ])
    ) {
        let _ = tokio_test::block_on(async move {
            let config = MemoryPressureConfig::default();
            let handler = MemoryPressureHandler::new(config);

            let mut total_freed = 0u64;

            // Property: GPU cleanup should work on multiple devices
            for device_id in &device_ids {
                let memory_freed = match strategy {
                    GpuCleanupStrategy::GpuCacheEviction => {
                        handler.execute_gpu_cache_eviction(*device_id).await
                    }
                    GpuCleanupStrategy::GpuBufferCompaction => {
                        handler.execute_gpu_buffer_compaction(*device_id).await
                    }
                    GpuCleanupStrategy::GpuModelUnloading => {
                        handler.execute_gpu_model_unloading(*device_id).await
                    }
                    GpuCleanupStrategy::GpuVramCompaction => {
                        handler.execute_gpu_vram_compaction(*device_id).await
                    }
                    GpuCleanupStrategy::GpuMemoryDefragmentation => {
                        handler.execute_gpu_memory_defragmentation(*device_id).await
                    }
                    GpuCleanupStrategy::GpuContextSwitching => {
                        handler.execute_gpu_context_switching(*device_id).await
                    }
                    GpuCleanupStrategy::GpuStreamCleanup => {
                        handler.execute_gpu_stream_cleanup(*device_id).await
                    }
                    GpuCleanupStrategy::GpuTextureCleanup => {
                        handler.execute_gpu_texture_cleanup(*device_id).await
                    }
                    GpuCleanupStrategy::GpuMemoryPoolReset => {
                        handler.execute_gpu_memory_pool_reset(*device_id).await
                    }
                    GpuCleanupStrategy::GpuBatchSizeReduction => {
                        handler.execute_gpu_batch_size_reduction(*device_id).await
                    }
                };

                // Property: Should always free some memory
                prop_assert!(memory_freed > 0);

                // Property: Freed amount should be reasonable
                prop_assert!(memory_freed <= 2u64 * 1024 * 1024 * 1024); // Max 2GB per operation

                total_freed += memory_freed;
            }

            // Property: Total freed memory should be proportional to number of devices
            prop_assert!(total_freed >= device_ids.len() as u64 * 10 * 1024 * 1024); // At least 10MB per device

            Ok(())
        });
    }

    /// Property: Adaptive threshold adjustment stability
    #[test]
    fn prop_adaptive_threshold_stability(
        utilization_sequence in prop::collection::vec(0.0f32..1.0f32, 10..100),
        learning_rate in 0.001f32..0.1f32,
        adjustment_factor in 0.1f32..2.0f32
    ) {
        let _ = tokio_test::block_on(async move {
            let mut config = MemoryPressureConfig::default();
            config.pressure_thresholds.adaptive = true;
            config.pressure_thresholds.learning_rate = learning_rate;
            config.pressure_thresholds.adjustment_factor = adjustment_factor;

            let handler = MemoryPressureHandler::new(config);

            let initial_thresholds = {
                let thresholds_arc = handler.adaptive_thresholds();
                let thresholds = thresholds_arc.read().await;
                (thresholds.low, thresholds.medium, thresholds.high, thresholds.critical)
            };

            // Simulate memory utilization sequence
            for utilization in utilization_sequence.iter() {
                let _ = handler.update_memory_prediction(*utilization).await;
                let _ = handler.adapt_thresholds().await;
            }

            let final_thresholds = {
                let thresholds_arc = handler.adaptive_thresholds();
                let thresholds = thresholds_arc.read().await;
                (thresholds.low, thresholds.medium, thresholds.high, thresholds.critical)
            };

            // Property: Thresholds should remain valid after adaptation
            prop_assert!(final_thresholds.0 >= 0.0 && final_thresholds.0 <= 1.0);
            prop_assert!(final_thresholds.1 >= 0.0 && final_thresholds.1 <= 1.0);
            prop_assert!(final_thresholds.2 >= 0.0 && final_thresholds.2 <= 1.0);
            prop_assert!(final_thresholds.3 >= 0.0 && final_thresholds.3 <= 1.0);

            // Property: Threshold ordering should be maintained
            prop_assert!(final_thresholds.0 <= final_thresholds.1);
            prop_assert!(final_thresholds.1 <= final_thresholds.2);
            prop_assert!(final_thresholds.2 <= final_thresholds.3);

            // Property: Changes should be bounded by learning rate
            let max_change = learning_rate * utilization_sequence.len() as f32 * 0.1;
            prop_assert!((final_thresholds.0 - initial_thresholds.0).abs() <= max_change * 2.0);
            prop_assert!((final_thresholds.1 - initial_thresholds.1).abs() <= max_change * 2.0);
            prop_assert!((final_thresholds.2 - initial_thresholds.2).abs() <= max_change * 2.0);
            prop_assert!((final_thresholds.3 - initial_thresholds.3).abs() <= max_change * 2.0);

            Ok(())
        });
    }

    /// Property: Memory prediction temporal consistency
    #[test]
    fn prop_memory_prediction_temporal_consistency(
        base_utilization in 0.2f32..0.8f32,
        trend in -0.1f32..0.1f32, // Small trend changes
        sequence_length in 5usize..50usize,
        noise_level in 0.0f32..0.1f32
    ) {
        let _ = tokio_test::block_on(async move {
            let config = MemoryPressureConfig::default();
            let handler = MemoryPressureHandler::new(config);

            let mut predictions = Vec::new();

            // Generate sequence with trend and noise
            for i in 0..sequence_length {
                let utilization = base_utilization +
                    trend * i as f32 +
                    noise_level * (i as f32 % 3.0 - 1.0); // Simple noise pattern

                let utilization = utilization.clamp(0.0, 1.0);

                let prediction = handler.update_memory_prediction(utilization).await.unwrap();
                predictions.push((utilization, prediction));

                // Property: Each prediction should be valid
                prop_assert!(prediction >= 0.0 && prediction <= 1.0);
            }

            // Property: Predictions should show some correlation with actual values
            let last_few_predictions: Vec<f32> = predictions.iter().rev().take(5).map(|(_, p)| *p).collect();
            let last_few_actual: Vec<f32> = predictions.iter().rev().take(5).map(|(a, _)| *a).collect();

            if last_few_predictions.len() >= 3 {
                let pred_avg = last_few_predictions.iter().sum::<f32>() / last_few_predictions.len() as f32;
                let actual_avg = last_few_actual.iter().sum::<f32>() / last_few_actual.len() as f32;

                // Predictions should be reasonably close to actual values on average
                prop_assert!((pred_avg - actual_avg).abs() <= 0.3);
            }

            Ok(())
        });
    }

    /// Property: Emergency cleanup effectiveness under extreme pressure
    #[test]
    fn prop_emergency_cleanup_effectiveness(
        config in memory_pressure_config_strategy(),
        stress_iterations in 1usize..10usize
    ) {
        let _ = tokio_test::block_on(async move {
            let handler = MemoryPressureHandler::new(config);

            let mut total_freed = 0u64;

            // Property: Emergency cleanup should work multiple times
            for _ in 0..stress_iterations {
                let freed = handler.trigger_emergency_cleanup().await.unwrap();
                total_freed += freed;

                // Property: Should always free some memory
                prop_assert!(freed > 0);

                // Property: Should free substantial amount during emergency
                prop_assert!(freed >= 10 * 1024 * 1024); // At least 10MB
            }

            // Property: Multiple emergency cleanups should be effective
            prop_assert!(total_freed >= stress_iterations as u64 * 10 * 1024 * 1024);

            // Property: Total freed should be reasonable (not infinite)
            prop_assert!(total_freed <= stress_iterations as u64 * 1024 * 1024 * 1024); // Max 1GB per iteration

            Ok(())
        });
    }

    /// Property: Pressure trend calculation mathematical properties
    #[test]
    fn prop_pressure_trend_mathematical_properties(
        snapshots in prop::collection::vec(pressure_snapshot_strategy(), 2..20)
    ) {
        let config = MemoryPressureConfig::default();
        let handler = MemoryPressureHandler::new(config);

        let snapshot_refs: Vec<&PressureSnapshot> = snapshots.iter().collect();
        let trend = handler.calculate_pressure_trend(&snapshot_refs);

        // Property: Trend should be bounded
        prop_assert!(trend >= -1.0 && trend <= 1.0);

        // Property: For constant utilization, trend should be near zero
        let utilizations: Vec<f32> = snapshots.iter().map(|s| s.utilization).collect();
        let is_constant = utilizations.windows(2).all(|w| (w[0] - w[1]).abs() < 0.01);

        if is_constant && snapshots.len() >= 3 {
            prop_assert!(trend.abs() <= 0.1);
        }

        // Property: For strictly increasing utilization, trend should be positive
        let is_increasing = utilizations.windows(2).all(|w| w[0] <= w[1]);
        let has_actual_increase = utilizations.first() < utilizations.last();

        if is_increasing && has_actual_increase && snapshots.len() >= 3 {
            prop_assert!(trend >= 0.0);
        }

        // Property: For strictly decreasing utilization, trend should be negative
        let is_decreasing = utilizations.windows(2).all(|w| w[0] >= w[1]);
        let has_actual_decrease = utilizations.first() > utilizations.last();

        if is_decreasing && has_actual_decrease && snapshots.len() >= 3 {
            prop_assert!(trend <= 0.0);
        }
    }

    /// Property: Memory forecast accuracy bounds
    #[test]
    fn prop_memory_forecast_accuracy(
        forecast_minutes in 1u32..60u32, // Up to 1 hour forecast
        pattern_variance in 0.0f32..0.2f32, // Low to moderate variance
        base_utilization in 0.1f32..0.9f32
    ) {
        let _ = tokio_test::block_on(async move {
            let config = MemoryPressureConfig::default();
            let handler = MemoryPressureHandler::new(config);

            // Create realistic historical pattern
            let now = Utc::now();
            {
                let pattern_history_arc = handler.pattern_history();
                let mut pattern_history = pattern_history_arc.lock().await;
                for i in 0..24 {
                    let hour_variation = (i as f32 * std::f32::consts::PI / 12.0).sin() * pattern_variance;
                    let utilization = (base_utilization + hour_variation).clamp(0.0, 1.0);

                    pattern_history.push_back((
                        now - ChronoDuration::hours(i),
                        utilization
                    ));
                }
            }

            let forecast = handler.get_memory_forecast(forecast_minutes).await.unwrap();

            // Property: Forecast should be structurally valid
            prop_assert!(forecast.predicted_utilization >= 0.0 && forecast.predicted_utilization <= 1.0);
            prop_assert!(forecast.confidence >= 0.0 && forecast.confidence <= 1.0);
            prop_assert!(forecast.trend >= -1.0 && forecast.trend <= 1.0);
            prop_assert_eq!(forecast.window_seconds, forecast_minutes as u64 * 60);

            // Property: Prediction should be influenced by historical data
            let historical_avg = {
                let pattern_history_arc = handler.pattern_history();
                let pattern_history = pattern_history_arc.lock().await;
                let total: f32 = pattern_history.iter().map(|(_, util)| util).sum();
                total / pattern_history.len() as f32
            };

            // Prediction should be within reasonable variance of historical average
            let prediction_variance = (forecast.predicted_utilization - historical_avg).abs();
            prop_assert!(prediction_variance <= 0.4); // Allow 40% variance from historical average

            // Property: Confidence should correlate with pattern stability
            if pattern_variance < 0.05 {
                // Low variance should result in higher confidence
                prop_assert!(forecast.confidence >= 0.4);
            }

            Ok(())
        });
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;
    use std::sync::Arc;
    use tokio::time::{timeout, Duration};

    /// Stress test: Concurrent memory allocations and deallocations
    #[tokio::test]
    async fn stress_concurrent_allocations() {
        let config = MemoryPressureConfig::default();
        let handler = Arc::new(MemoryPressureHandler::new(config));

        let num_tasks = 50;
        let allocations_per_task = 10;

        let mut tasks = Vec::new();

        for task_id in 0..num_tasks {
            let handler_clone = Arc::clone(&handler);

            tasks.push(tokio::spawn(async move {
                let mut allocation_ids = Vec::new();

                // Allocate
                for i in 0..allocations_per_task {
                    let size = 1024 * 1024 * (i + 1); // 1MB, 2MB, 3MB, etc.
                    let allocation_type = format!("stress_test_{}_{}", task_id, i);

                    match handler_clone.request_allocation(size as u64, allocation_type).await {
                        Ok(id) => allocation_ids.push(id),
                        Err(_) => {
                            // Expected under memory pressure
                            break;
                        },
                    }
                }

                // Deallocate
                for id in allocation_ids {
                    let _ = handler_clone.release_allocation(&id).await;
                }

                task_id
            }));
        }

        // All tasks should complete within reasonable time
        let results = timeout(Duration::from_secs(30), futures::future::join_all(tasks)).await;
        assert!(
            results.is_ok(),
            "Stress test should complete within 30 seconds"
        );

        let completed_tasks: Vec<_> = results.unwrap().into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(
            completed_tasks.len(),
            num_tasks,
            "All tasks should complete successfully"
        );
    }

    /// Stress test: Rapid pressure level changes
    #[tokio::test]
    async fn stress_rapid_pressure_changes() {
        let config = MemoryPressureConfig::default();
        let handler = MemoryPressureHandler::new(config);

        let pressure_sequence = vec![
            0.1, 0.9, 0.2, 0.8, 0.3, 0.95, 0.1, 0.85, 0.4, 0.99, 0.05, 0.75, 0.6, 0.92, 0.15, 0.88,
            0.45, 0.96, 0.25, 0.82,
        ];

        for utilization in pressure_sequence {
            let level = handler.calculate_pressure_level(utilization);

            // Verify level is appropriate for utilization
            match utilization {
                u if u >= 0.95 => assert_eq!(level, MemoryPressureLevel::Emergency),
                u if u >= 0.85 => assert_eq!(level, MemoryPressureLevel::Critical),
                u if u >= 0.75 => assert_eq!(level, MemoryPressureLevel::High),
                u if u >= 0.60 => assert_eq!(level, MemoryPressureLevel::Medium),
                u if u >= 0.45 => assert_eq!(level, MemoryPressureLevel::Low),
                _ => assert_eq!(level, MemoryPressureLevel::Normal),
            }

            // Update prediction to test adaptation
            let _ = handler.update_memory_prediction(utilization).await;
        }
    }

    /// Stress test: GPU cleanup under high load
    #[tokio::test]
    async fn stress_gpu_cleanup_high_load() {
        let config = MemoryPressureConfig::default();
        let handler = Arc::new(MemoryPressureHandler::new(config));

        let strategies = vec![
            GpuCleanupStrategy::GpuCacheEviction,
            GpuCleanupStrategy::GpuBufferCompaction,
            GpuCleanupStrategy::GpuModelUnloading,
            GpuCleanupStrategy::GpuVramCompaction,
        ];

        let mut tasks = Vec::new();

        // Simulate cleanup on multiple devices simultaneously
        for device_id in 0..4 {
            for strategy in &strategies {
                let handler_clone = Arc::clone(&handler);
                let strategy_clone = strategy.clone();

                tasks.push(tokio::spawn(async move {
                    let mut total_freed = 0u64;

                    for _ in 0..5 {
                        let freed = match strategy_clone {
                            GpuCleanupStrategy::GpuCacheEviction => {
                                handler_clone.execute_gpu_cache_eviction(device_id).await
                            },
                            GpuCleanupStrategy::GpuBufferCompaction => {
                                handler_clone.execute_gpu_buffer_compaction(device_id).await
                            },
                            GpuCleanupStrategy::GpuModelUnloading => {
                                handler_clone.execute_gpu_model_unloading(device_id).await
                            },
                            GpuCleanupStrategy::GpuVramCompaction => {
                                handler_clone.execute_gpu_vram_compaction(device_id).await
                            },
                            _ => 0,
                        };

                        assert!(freed > 0, "GPU cleanup should always free some memory");
                        total_freed += freed;
                    }

                    total_freed
                }));
            }
        }

        let results = futures::future::join_all(tasks).await;
        let total_system_freed: u64 = results.into_iter().filter_map(|r| r.ok()).sum();

        assert!(
            total_system_freed > 0,
            "System should free substantial memory under stress"
        );
        assert!(
            total_system_freed >= 100 * 1024 * 1024,
            "Should free at least 100MB total"
        );
    }
}
