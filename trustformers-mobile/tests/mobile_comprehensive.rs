// trustformers-mobile comprehensive test suite
// Tests the mobile deployment infrastructure, quantization,
// memory management, federated learning, and model management

#![allow(unused_imports)]
#![allow(clippy::float_cmp)]

// ============================================================
// Module 1: platform_detection
// ============================================================
mod platform_detection {
    use trustformers_mobile::{
        MemoryOptimization, MobileBackend, MobileConfig, MobilePlatform, MobileQuantizationConfig,
        MobileQuantizationScheme,
    };

    #[test]
    fn test_mobile_config_default_is_generic_cpu() {
        let config = MobileConfig::default();
        assert_eq!(config.platform, MobilePlatform::Generic);
        assert_eq!(config.backend, MobileBackend::CPU);
    }

    #[test]
    fn test_mobile_config_default_memory_optimization_is_balanced() {
        let config = MobileConfig::default();
        assert_eq!(config.memory_optimization, MemoryOptimization::Balanced);
    }

    #[test]
    fn test_mobile_config_default_memory_mb_reasonable() {
        let config = MobileConfig::default();
        // Should be a reasonable mobile default (between 64MB and 4GB)
        assert!(config.max_memory_mb >= 64);
        assert!(config.max_memory_mb <= 4096);
    }

    #[test]
    fn test_mobile_platform_variants_cover_ios_android_generic() {
        let ios = MobilePlatform::Ios;
        let android = MobilePlatform::Android;
        let generic = MobilePlatform::Generic;
        assert_ne!(ios, android);
        assert_ne!(android, generic);
        assert_ne!(ios, generic);
    }

    #[test]
    fn test_mobile_config_validate_passes_for_default() {
        let config = MobileConfig::default();
        assert!(config.validate().is_ok(), "default config should be valid");
    }

    #[test]
    fn test_mobile_config_validate_rejects_too_little_memory() {
        let config = MobileConfig {
            max_memory_mb: 10,
            ..Default::default()
        }; // below 64 MB minimum
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mobile_config_validate_rejects_too_many_threads() {
        let config = MobileConfig {
            num_threads: 17,
            ..Default::default()
        }; // exceeds 16-thread limit
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mobile_config_ios_platform_is_ios() {
        let config = MobileConfig::ios_optimized();
        assert_eq!(config.platform, MobilePlatform::Ios);
    }

    #[test]
    fn test_mobile_config_android_platform_is_android() {
        let config = MobileConfig::android_optimized();
        assert_eq!(config.platform, MobilePlatform::Android);
    }

    #[test]
    fn test_mobile_config_validate_rejects_nnapi_on_ios() {
        let mut config = MobileConfig::ios_optimized();
        config.backend = MobileBackend::NNAPI;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mobile_config_validate_rejects_coreml_on_android() {
        let mut config = MobileConfig::android_optimized();
        config.backend = MobileBackend::CoreML;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mobile_config_get_thread_count_nonzero() {
        let config = MobileConfig::default();
        let threads = config.get_thread_count();
        assert!(threads > 0, "thread count must be at least 1");
    }

    #[test]
    fn test_memory_optimization_eq() {
        assert_eq!(MemoryOptimization::Minimal, MemoryOptimization::Minimal);
        assert_ne!(MemoryOptimization::Minimal, MemoryOptimization::Maximum);
    }
}

// ============================================================
// Module 2: quantization_mobile
// ============================================================
mod quantization_mobile {
    use trustformers_mobile::{
        MemoryOptimization, MobileConfig, MobileQuantizationConfig, MobileQuantizationScheme,
    };

    fn make_int8_quant() -> MobileQuantizationConfig {
        MobileQuantizationConfig {
            scheme: MobileQuantizationScheme::Int8,
            dynamic: false,
            per_channel: true,
        }
    }

    fn make_int4_quant() -> MobileQuantizationConfig {
        MobileQuantizationConfig {
            scheme: MobileQuantizationScheme::Int4,
            dynamic: true,
            per_channel: true,
        }
    }

    fn make_fp16_quant() -> MobileQuantizationConfig {
        MobileQuantizationConfig {
            scheme: MobileQuantizationScheme::FP16,
            dynamic: false,
            per_channel: false,
        }
    }

    #[test]
    fn test_int8_quantization_config_creation() {
        let q = make_int8_quant();
        assert_eq!(q.scheme, MobileQuantizationScheme::Int8);
        assert!(!q.dynamic);
        assert!(q.per_channel);
    }

    #[test]
    fn test_int4_quantization_config_creation() {
        let q = make_int4_quant();
        assert_eq!(q.scheme, MobileQuantizationScheme::Int4);
        assert!(q.dynamic);
        assert!(q.per_channel);
    }

    #[test]
    fn test_fp16_quantization_config_creation() {
        let q = make_fp16_quant();
        assert_eq!(q.scheme, MobileQuantizationScheme::FP16);
        assert!(!q.dynamic);
        assert!(!q.per_channel);
    }

    #[test]
    fn test_dynamic_quantization_variant_exists() {
        let q = MobileQuantizationConfig {
            scheme: MobileQuantizationScheme::Dynamic,
            dynamic: true,
            per_channel: false,
        };
        assert_eq!(q.scheme, MobileQuantizationScheme::Dynamic);
    }

    #[test]
    fn test_quantization_scheme_ne() {
        assert_ne!(
            MobileQuantizationScheme::Int8,
            MobileQuantizationScheme::Int4
        );
        assert_ne!(
            MobileQuantizationScheme::FP16,
            MobileQuantizationScheme::Dynamic
        );
    }

    #[test]
    fn test_int4_memory_reduction_greater_than_int8() {
        // estimate_memory_usage compresses int4 more than int8
        let mut config = MobileConfig {
            quantization: Some(MobileQuantizationConfig {
                scheme: MobileQuantizationScheme::Int4,
                dynamic: true,
                per_channel: true,
            }),
            ..Default::default()
        };
        let int4_memory = config.estimate_memory_usage(1000);

        config.quantization = Some(MobileQuantizationConfig {
            scheme: MobileQuantizationScheme::Int8,
            dynamic: false,
            per_channel: false,
        });
        let int8_memory = config.estimate_memory_usage(1000);

        assert!(
            int4_memory < int8_memory,
            "int4 ({int4_memory}) should use less memory than int8 ({int8_memory})"
        );
    }

    #[test]
    fn test_fp16_memory_reduction_less_than_int8() {
        let mut config = MobileConfig {
            quantization: Some(MobileQuantizationConfig {
                scheme: MobileQuantizationScheme::FP16,
                dynamic: false,
                per_channel: false,
            }),
            ..Default::default()
        };
        let fp16_memory = config.estimate_memory_usage(1000);

        config.quantization = Some(MobileQuantizationConfig {
            scheme: MobileQuantizationScheme::Int8,
            dynamic: false,
            per_channel: false,
        });
        let int8_memory = config.estimate_memory_usage(1000);

        // FP16 = 1/2 of FP32, Int8 = 1/4 of FP32, so fp16 > int8 in memory
        assert!(
            fp16_memory > int8_memory,
            "fp16 ({fp16_memory}) should use more memory than int8 ({int8_memory})"
        );
    }

    #[test]
    fn test_quantization_reduces_memory_vs_no_quantization() {
        let base_size = 1000;

        let mut config = MobileConfig {
            use_fp16: false,
            quantization: None,
            ..Default::default()
        };
        let no_quant = config.estimate_memory_usage(base_size);

        config.quantization = Some(MobileQuantizationConfig {
            scheme: MobileQuantizationScheme::Int8,
            dynamic: true,
            per_channel: false,
        });
        let with_quant = config.estimate_memory_usage(base_size);

        assert!(
            with_quant < no_quant,
            "quantization should reduce memory: no_quant={no_quant}, with_quant={with_quant}"
        );
    }

    #[test]
    fn test_quantization_config_serialization() {
        let q = make_int8_quant();
        let json = serde_json::to_string(&q).expect("serialization should succeed");
        let restored: MobileQuantizationConfig =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(restored.scheme, q.scheme);
        assert_eq!(restored.dynamic, q.dynamic);
        assert_eq!(restored.per_channel, q.per_channel);
    }

    #[test]
    fn test_ultra_low_memory_uses_int4() {
        let config = MobileConfig::ultra_low_memory();
        let q = config.quantization.expect("ultra_low_memory should have quantization");
        assert_eq!(q.scheme, MobileQuantizationScheme::Int4);
    }
}

// ============================================================
// Module 3: memory_management
// ============================================================
mod memory_management {
    use trustformers_mobile::{
        lifecycle::{LifecycleConfig, MemoryPressureLevel, MemoryWarningConfig},
        MemoryOptimization, MobileConfig,
    };

    #[test]
    fn test_memory_optimization_maximum_gives_lowest_overhead() {
        let mut config = MobileConfig {
            quantization: None,
            use_fp16: false,
            ..Default::default()
        };

        config.memory_optimization = MemoryOptimization::Maximum;
        let max_mem = config.estimate_memory_usage(1000);

        config.memory_optimization = MemoryOptimization::Minimal;
        let min_mem = config.estimate_memory_usage(1000);

        assert!(
            max_mem < min_mem,
            "Maximum optimization ({max_mem}) should have lower overhead than Minimal ({min_mem})"
        );
    }

    #[test]
    fn test_memory_optimization_balanced_between_extremes() {
        let mut config = MobileConfig {
            quantization: None,
            use_fp16: false,
            ..Default::default()
        };

        config.memory_optimization = MemoryOptimization::Maximum;
        let max_mem = config.estimate_memory_usage(1000);

        config.memory_optimization = MemoryOptimization::Balanced;
        let balanced_mem = config.estimate_memory_usage(1000);

        config.memory_optimization = MemoryOptimization::Minimal;
        let min_mem = config.estimate_memory_usage(1000);

        assert!(
            max_mem <= balanced_mem && balanced_mem <= min_mem,
            "Balanced should be between Maximum and Minimal: max={max_mem}, balanced={balanced_mem}, min={min_mem}"
        );
    }

    #[test]
    fn test_ultra_low_memory_max_is_256mb() {
        let config = MobileConfig::ultra_low_memory();
        assert!(config.max_memory_mb <= 256);
    }

    #[test]
    fn test_ultra_low_memory_optimization_is_maximum() {
        let config = MobileConfig::ultra_low_memory();
        assert_eq!(config.memory_optimization, MemoryOptimization::Maximum);
    }

    #[test]
    fn test_ultra_low_memory_single_threaded() {
        let config = MobileConfig::ultra_low_memory();
        assert_eq!(config.num_threads, 1);
    }

    #[test]
    fn test_memory_pressure_level_variants_exist() {
        let _normal = MemoryPressureLevel::Normal;
        let _warning = MemoryPressureLevel::Warning;
        let _critical = MemoryPressureLevel::Critical;
        let _emergency = MemoryPressureLevel::Emergency;
    }

    #[test]
    fn test_lifecycle_config_default_has_background_enabled() {
        let config = LifecycleConfig::default();
        assert!(config.enable_background_execution);
    }

    #[test]
    fn test_lifecycle_config_default_has_state_persistence() {
        let config = LifecycleConfig::default();
        assert!(config.enable_state_persistence);
    }

    #[test]
    fn test_lifecycle_config_serialization_roundtrip() {
        let config = LifecycleConfig::default();
        let json = serde_json::to_string(&config).expect("serialization should succeed");
        let restored: LifecycleConfig =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(
            restored.enable_background_execution,
            config.enable_background_execution
        );
        assert_eq!(
            restored.background_execution_limit_seconds,
            config.background_execution_limit_seconds
        );
    }

    #[test]
    fn test_memory_estimate_scales_with_model_size() {
        let config = MobileConfig::default();
        let small = config.estimate_memory_usage(100);
        let large = config.estimate_memory_usage(10_000);
        assert!(large > small, "larger model should use more memory");
    }

    #[test]
    fn test_mobile_config_ios_has_more_memory_than_ultra_low() {
        let ios = MobileConfig::ios_optimized();
        let low = MobileConfig::ultra_low_memory();
        assert!(ios.max_memory_mb > low.max_memory_mb);
    }
}

// ============================================================
// Module 4: model_management
// ============================================================
mod model_management {
    use std::env::temp_dir;
    use trustformers_mobile::model_management::{
        DownloadStatus, ModelCompatibility, ModelManagerConfig, ModelMetadata, ModelUpdate,
        UpdatePriority, UpdateType,
    };
    use trustformers_mobile::MobileConfig;

    fn make_config() -> ModelManagerConfig {
        let storage_dir = temp_dir().join("trustformers_comprehensive_model_test");
        ModelManagerConfig {
            update_server_url: "https://models.example.com".to_string(),
            api_key: Some("test-api-key".to_string()),
            storage_directory: storage_dir,
            max_storage_size_mb: 1024,
            enable_auto_updates: true,
            update_check_interval_seconds: 3600,
            enable_differential_updates: true,
            require_signature_verification: true,
            download_timeout_seconds: 300,
            max_concurrent_downloads: 2,
            enable_compression: true,
            download_retry_attempts: 3,
        }
    }

    fn make_metadata(id: &str, version: &str) -> ModelMetadata {
        ModelMetadata {
            model_id: id.to_string(),
            version: version.to_string(),
            model_type: "bert-base".to_string(),
            size_bytes: 256 * 1024 * 1024,
            checksum: "deadbeef12345678".to_string(),
            signature: None,
            download_url: format!("https://models.example.com/{id}/{version}/model.bin"),
            differential_url: None,
            description: "Test BERT model".to_string(),
            required_config: MobileConfig::default(),
            compatibility: ModelCompatibility::default(),
            release_timestamp: 1_700_000_000,
            deprecation_timestamp: None,
            tags: vec!["nlp".to_string(), "bert".to_string()],
        }
    }

    #[test]
    fn test_model_manager_config_default_is_valid() {
        let config = ModelManagerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_model_manager_config_empty_url_is_invalid() {
        let mut config = make_config();
        config.update_server_url = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_manager_config_zero_storage_is_invalid() {
        let mut config = make_config();
        config.max_storage_size_mb = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_manager_config_short_timeout_is_invalid() {
        let mut config = make_config();
        config.download_timeout_seconds = 10; // less than 30
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_manager_config_zero_downloads_is_invalid() {
        let mut config = make_config();
        config.max_concurrent_downloads = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_metadata_id_and_version() {
        let m = make_metadata("bert-base-uncased", "1.0.0");
        assert_eq!(m.model_id, "bert-base-uncased");
        assert_eq!(m.version, "1.0.0");
    }

    #[test]
    fn test_model_metadata_serialization_roundtrip() {
        let m = make_metadata("gpt2", "2.1.3");
        let json = serde_json::to_string(&m).expect("serialization should succeed");
        let restored: ModelMetadata =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(restored.model_id, m.model_id);
        assert_eq!(restored.version, m.version);
        assert_eq!(restored.size_bytes, m.size_bytes);
    }

    #[test]
    fn test_model_compatibility_default_has_min_memory() {
        let compat = ModelCompatibility::default();
        assert!(compat.min_memory_mb > 0);
    }

    #[test]
    fn test_model_compatibility_supports_arm64() {
        let compat = ModelCompatibility::default();
        assert!(compat.supported_architectures.contains(&"arm64".to_string()));
    }

    #[test]
    fn test_update_priority_variants_exist() {
        let _critical = UpdatePriority::Critical;
        let _high = UpdatePriority::High;
        let _normal = UpdatePriority::Normal;
        let _low = UpdatePriority::Low;
    }

    #[test]
    fn test_download_status_variants_exist() {
        let _pending = DownloadStatus::Pending;
        let _downloading = DownloadStatus::Downloading;
        let _verifying = DownloadStatus::Verifying;
        let _installing = DownloadStatus::Installing;
        let _completed = DownloadStatus::Completed;
        let _failed = DownloadStatus::Failed("test error".to_string());
        let _cancelled = DownloadStatus::Cancelled;
    }

    #[test]
    fn test_update_type_variants_exist() {
        let _full = UpdateType::Full;
        let _differential = UpdateType::Differential;
        let _config_only = UpdateType::ConfigOnly;
    }

    #[test]
    fn test_temp_dir_storage_path_is_usable() {
        let config = make_config();
        // The storage directory path should be constructible (not require actual creation)
        assert!(!config.storage_directory.as_os_str().is_empty());
    }
}

// ============================================================
// Module 5: inference_config
// ============================================================
mod inference_config {
    use trustformers_mobile::{
        inference::{ExecutionPlan, ExecutionStrategy, MobileInferenceBuilder, ModelFormat},
        MobileBackend, MobileConfig, MobilePlatform,
    };

    #[test]
    fn test_execution_plan_new_sequential() {
        let plan = ExecutionPlan::new(ExecutionStrategy::Sequential, 12);
        assert_eq!(plan.strategy, ExecutionStrategy::Sequential);
        assert_eq!(plan.num_layers, 12);
    }

    #[test]
    fn test_execution_plan_batch_size_default_one() {
        let plan = ExecutionPlan::new(ExecutionStrategy::Sequential, 6);
        assert_eq!(plan.batch_size, 1);
    }

    #[test]
    fn test_execution_strategy_variants_exist() {
        let _seq = ExecutionStrategy::Sequential;
        let _layer = ExecutionStrategy::LayerParallel;
        let _full = ExecutionStrategy::FullParallel;
    }

    #[test]
    fn test_model_format_variants_exist() {
        let _safe = ModelFormat::SafeTensors;
        let _torch = ModelFormat::PyTorch;
        let _onnx = ModelFormat::ONNX;
        let _tf = ModelFormat::TensorFlow;
        let _unknown = ModelFormat::Unknown;
    }

    #[test]
    fn test_model_format_ne() {
        assert_ne!(ModelFormat::SafeTensors, ModelFormat::PyTorch);
        assert_ne!(ModelFormat::ONNX, ModelFormat::TensorFlow);
    }

    #[test]
    fn test_mobile_inference_builder_builds_successfully() {
        let engine = MobileInferenceBuilder::default().build();
        assert!(engine.is_ok(), "default builder should succeed");
    }

    #[test]
    fn test_mobile_config_batching_disabled_by_default() {
        let config = MobileConfig::default();
        assert!(!config.enable_batching);
    }

    #[test]
    fn test_mobile_config_ios_batching_enabled() {
        let config = MobileConfig::ios_optimized();
        assert!(config.enable_batching);
    }

    #[test]
    fn test_mobile_config_android_batching_disabled() {
        let config = MobileConfig::android_optimized();
        assert!(!config.enable_batching);
    }

    #[test]
    fn test_execution_plan_serialization() {
        let plan = ExecutionPlan::new(ExecutionStrategy::LayerParallel, 24);
        let json = serde_json::to_string(&plan).expect("serialization should succeed");
        let restored: ExecutionPlan =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(restored.strategy, plan.strategy);
        assert_eq!(restored.num_layers, plan.num_layers);
    }

    #[test]
    fn test_mobile_inference_engine_creation_with_default_config() {
        use trustformers_mobile::inference::MobileInferenceEngine;
        let config = MobileConfig::default();
        let engine = MobileInferenceEngine::new(config);
        assert!(engine.is_ok(), "should create engine with default config");
    }

    #[test]
    fn test_mobile_config_num_threads_auto_detects_for_ios() {
        let mut config = MobileConfig::ios_optimized();
        config.num_threads = 0; // auto-detect
        let threads = config.get_thread_count();
        assert!(threads >= 1);
    }
}

// ============================================================
// Module 6: battery_thermal
// ============================================================
mod battery_thermal {
    use trustformers_mobile::{
        battery::{BatteryConfig, BatteryLevel, BatteryThresholds, PowerUsageLimits},
        thermal_power::{ThermalPowerConfig, ThermalThresholds, ThrottleLevel, ThrottlingStrategy},
        ThermalState,
    };

    fn make_battery_thresholds() -> BatteryThresholds {
        BatteryThresholds::default()
    }

    fn make_power_limits() -> PowerUsageLimits {
        PowerUsageLimits::default()
    }

    #[test]
    fn test_battery_level_critical_threshold_below_low() {
        let t = make_battery_thresholds();
        // Critical threshold must be strictly below low threshold
        assert!(t.critical_percent < t.low_percent);
    }

    #[test]
    fn test_battery_level_thresholds_ordering() {
        let t = make_battery_thresholds();
        assert!(t.critical_percent < t.low_percent);
        assert!(t.low_percent < t.medium_percent);
        assert!(t.medium_percent < t.high_percent);
    }

    #[test]
    fn test_battery_level_enum_variants_exist() {
        let _critical = BatteryLevel::Critical;
        let _low = BatteryLevel::Low;
        let _medium = BatteryLevel::Medium;
        let _high = BatteryLevel::High;
        let _full = BatteryLevel::Full;
        let _charging = BatteryLevel::Charging;
    }

    #[test]
    fn test_power_limits_charging_exceeds_battery_limit() {
        let limits = make_power_limits();
        assert!(limits.max_power_when_charging_mw > limits.max_power_on_battery_mw);
    }

    #[test]
    fn test_power_limits_background_lower_than_foreground() {
        let limits = make_power_limits();
        assert!(limits.max_background_power_mw < limits.max_power_on_battery_mw);
    }

    #[test]
    fn test_throttle_level_ordering() {
        assert!(ThrottleLevel::None < ThrottleLevel::Light);
        assert!(ThrottleLevel::Light < ThrottleLevel::Moderate);
        assert!(ThrottleLevel::Moderate < ThrottleLevel::Aggressive);
        assert!(ThrottleLevel::Aggressive < ThrottleLevel::Emergency);
    }

    #[test]
    fn test_thermal_state_nominal_ne_critical() {
        assert_ne!(ThermalState::Nominal, ThermalState::Critical);
    }

    #[test]
    fn test_thermal_state_variants_all_exist() {
        let _n = ThermalState::Nominal;
        let _f = ThermalState::Fair;
        let _s = ThermalState::Serious;
        let _c = ThermalState::Critical;
        let _e = ThermalState::Emergency;
        let _shutdown = ThermalState::Shutdown;
    }

    #[test]
    fn test_throttling_strategy_variants_exist() {
        let _c = ThrottlingStrategy::Conservative;
        let _b = ThrottlingStrategy::Balanced;
        let _a = ThrottlingStrategy::Aggressive;
        let _x = ThrottlingStrategy::Custom;
    }

    #[test]
    fn test_battery_config_monitoring_enabled_by_default() {
        let config = BatteryConfig::default();
        assert!(config.enable_monitoring);
    }

    #[test]
    fn test_battery_config_monitoring_interval_positive() {
        let config = BatteryConfig::default();
        assert!(config.monitoring_interval_ms > 0);
    }

    #[test]
    fn test_thermal_power_config_monitoring_enabled_by_default() {
        let config = ThermalPowerConfig::default();
        assert!(config.enable_thermal_monitoring);
    }

    #[test]
    fn test_thermal_thresholds_emergency_above_aggressive_throttle() {
        let config = ThermalPowerConfig::default();
        let t = &config.thermal_thresholds;
        assert!(t.emergency_celsius > t.aggressive_throttle_celsius);
    }
}

// ============================================================
// Module 7: federated_learning (privacy-preserving inference)
// ============================================================
mod federated_learning {
    // We test the always-available privacy_preserving_inference module types.

    use trustformers_mobile::privacy_preserving_inference::{
        AggregationMethod, InferenceBudgetConfig, InferencePrivacyConfig, InputPerturbationMethod,
        InputPrivacyConfig, OutputPrivacyConfig, OutputPrivacyMethod, PrivacyLevel,
        SecureAggregationConfig,
    };

    fn make_inference_privacy_config() -> InferencePrivacyConfig {
        InferencePrivacyConfig {
            enabled: true,
            privacy_level: PrivacyLevel::Medium,
            input_privacy: InputPrivacyConfig {
                enabled: true,
                method: InputPerturbationMethod::Gaussian,
                noise_scale: 0.01,
                adaptive_noise: false,
                max_perturbation: 1.0,
            },
            output_privacy: OutputPrivacyConfig {
                enabled: true,
                method: OutputPrivacyMethod::PredictionNoise,
                noise_scale: 0.001,
                post_processing_privacy: false,
                calibrated_outputs: false,
            },
            inference_budget: InferenceBudgetConfig {
                total_epsilon: 10.0,
                total_delta: 1e-5,
                epsilon_per_request: 0.1,
                reset_period_secs: 3600,
                track_budget: true,
            },
            secure_aggregation: SecureAggregationConfig {
                enabled: false,
                method: AggregationMethod::SecureSum,
                min_participants: 5,
                security_threshold: 0.8,
            },
        }
    }

    #[test]
    fn test_inference_privacy_config_creation() {
        let config = make_inference_privacy_config();
        assert!(config.enabled);
        assert!(config.input_privacy.enabled);
    }

    #[test]
    fn test_input_perturbation_method_variants_exist() {
        let _gaussian = InputPerturbationMethod::Gaussian;
        let _laplace = InputPerturbationMethod::Laplacian;
        let _rand = InputPerturbationMethod::RandomizedResponse;
        let _local = InputPerturbationMethod::LocalSensitivity;
        let _feature = InputPerturbationMethod::FeatureSpecific;
    }

    #[test]
    fn test_output_privacy_method_variants_exist() {
        let _pred = OutputPrivacyMethod::PredictionNoise;
        let _smooth = OutputPrivacyMethod::ProbabilitySmoothing;
        let _trunc = OutputPrivacyMethod::ConfidenceTruncation;
        let _report = OutputPrivacyMethod::ReportMechanism;
        let _exp = OutputPrivacyMethod::ExponentialMechanism;
    }

    #[test]
    fn test_budget_config_epsilon_positive() {
        let config = make_inference_privacy_config();
        assert!(config.inference_budget.total_epsilon > 0.0);
    }

    #[test]
    fn test_budget_config_delta_small_positive() {
        let config = make_inference_privacy_config();
        assert!(config.inference_budget.total_delta > 0.0);
        assert!(config.inference_budget.total_delta < 1.0);
    }

    #[test]
    fn test_secure_aggregation_threshold_positive() {
        let config = make_inference_privacy_config();
        assert!(config.secure_aggregation.min_participants > 0);
    }

    #[test]
    fn test_aggregation_method_variants_exist() {
        let _sum = AggregationMethod::SecureSum;
        let _fedavg = AggregationMethod::FederatedAverage;
        let _mpc = AggregationMethod::MultiPartyComputation;
        let _thresh = AggregationMethod::ThresholdAggregation;
    }

    #[test]
    fn test_inference_privacy_config_serialization_roundtrip() {
        let config = make_inference_privacy_config();
        let json = serde_json::to_string(&config).expect("serialization should succeed");
        let restored: InferencePrivacyConfig =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(restored.enabled, config.enabled);
        assert_eq!(
            restored.inference_budget.total_epsilon,
            config.inference_budget.total_epsilon
        );
    }

    #[test]
    fn test_privacy_level_variants_exist() {
        let _none = PrivacyLevel::None;
        let _low = PrivacyLevel::Low;
        let _med = PrivacyLevel::Medium;
        let _high = PrivacyLevel::High;
        let _very_high = PrivacyLevel::VeryHigh;
        let _max = PrivacyLevel::Maximum;
        let _custom = PrivacyLevel::Custom;
    }

    #[test]
    fn test_budget_per_request_less_than_total() {
        let config = make_inference_privacy_config();
        assert!(
            config.inference_budget.epsilon_per_request <= config.inference_budget.total_epsilon
        );
    }
}

// ============================================================
// Module 8: cross_platform
// ============================================================
mod cross_platform {
    use trustformers_mobile::{
        wasm_simd::{
            SimdInstructionSet, SimdLaneWidth, SimdOperationType, SimdPerformanceMetrics,
            WasmSimdConfig,
        },
        webnn_integration::{
            WebNNBackend, WebNNCapabilities, WebNNDataType, WebNNDevice, WebNNGraphConfig,
            WebNNOperation, WebNNPowerPreference, WebNNSupportLevel, WebNNTensorDescriptor,
            WebNNUtils,
        },
    };

    #[test]
    fn test_webnn_graph_config_default_device_is_auto() {
        let config = WebNNGraphConfig::default();
        assert_eq!(config.device, WebNNDevice::Auto);
    }

    #[test]
    fn test_webnn_graph_config_default_power_preference_is_default() {
        let config = WebNNGraphConfig::default();
        assert_eq!(config.power_preference, WebNNPowerPreference::Default);
    }

    #[test]
    fn test_webnn_graph_config_default_dtype_is_float32() {
        let config = WebNNGraphConfig::default();
        assert_eq!(config.default_dtype, WebNNDataType::Float32);
    }

    #[test]
    fn test_webnn_capabilities_default_is_unavailable() {
        let caps = WebNNCapabilities::default();
        assert!(!caps.available);
    }

    #[test]
    fn test_webnn_capabilities_default_supports_cpu() {
        let caps = WebNNCapabilities::default();
        assert!(caps.supported_devices.contains(&WebNNDevice::CPU));
    }

    #[test]
    fn test_webnn_data_type_size_bytes_float32() {
        assert_eq!(WebNNDataType::Float32.size_bytes(), 4);
    }

    #[test]
    fn test_webnn_data_type_size_bytes_float16() {
        assert_eq!(WebNNDataType::Float16.size_bytes(), 2);
    }

    #[test]
    fn test_webnn_data_type_size_bytes_int8() {
        assert_eq!(WebNNDataType::Int8.size_bytes(), 1);
    }

    #[test]
    fn test_webnn_device_display_cpu() {
        let s = format!("{}", WebNNDevice::CPU);
        assert_eq!(s, "CPU");
    }

    #[test]
    fn test_webnn_device_display_gpu() {
        let s = format!("{}", WebNNDevice::GPU);
        assert_eq!(s, "GPU");
    }

    #[test]
    fn test_webnn_graph_config_serialization_roundtrip() {
        let config = WebNNGraphConfig::default();
        let json = serde_json::to_string(&config).expect("serialization should succeed");
        let restored: WebNNGraphConfig =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(restored.device, config.device);
        assert_eq!(restored.enable_fusion, config.enable_fusion);
    }

    #[test]
    fn test_wasm_simd_config_default_enables_simd() {
        let config = WasmSimdConfig::default();
        assert!(config.enable_simd);
    }

    #[test]
    fn test_wasm_simd_config_default_lane_is_32() {
        let config = WasmSimdConfig::default();
        assert_eq!(config.lane_width, SimdLaneWidth::Lane32);
    }

    #[test]
    fn test_wasm_simd_config_default_instruction_set_is_wasm128() {
        let config = WasmSimdConfig::default();
        assert_eq!(config.instruction_set, SimdInstructionSet::WASM128);
    }

    #[test]
    fn test_simd_operation_type_variants_exist() {
        let _matmul = SimdOperationType::MatMul;
        let _conv = SimdOperationType::Conv2D;
        let _add = SimdOperationType::Add;
        let _mul = SimdOperationType::Mul;
        let _act = SimdOperationType::Activation;
        let _bn = SimdOperationType::BatchNorm;
        let _attn = SimdOperationType::Attention;
        let _pool = SimdOperationType::Pooling;
    }

    #[test]
    fn test_simd_lane_width_variants_exist() {
        let _l8 = SimdLaneWidth::Lane8;
        let _l16 = SimdLaneWidth::Lane16;
        let _l32 = SimdLaneWidth::Lane32;
        let _l64 = SimdLaneWidth::Lane64;
        let _mixed = SimdLaneWidth::Mixed;
    }

    #[test]
    fn test_wasm_simd_config_serialization_roundtrip() {
        let config = WasmSimdConfig::default();
        let json = serde_json::to_string(&config).expect("serialization should succeed");
        let restored: WasmSimdConfig =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(restored.enable_simd, config.enable_simd);
        assert_eq!(restored.lane_width, config.lane_width);
        assert_eq!(restored.batch_size, config.batch_size);
    }

    #[test]
    fn test_webnn_support_level_variants_exist() {
        let _full = WebNNSupportLevel::Full;
        let _partial = WebNNSupportLevel::Partial;
        let _not_available = WebNNSupportLevel::NotAvailable;
    }

    #[test]
    fn test_webnn_power_preference_variants_exist() {
        let _low = WebNNPowerPreference::LowPower;
        let _high = WebNNPowerPreference::HighPerformance;
        let _def = WebNNPowerPreference::Default;
    }
}
