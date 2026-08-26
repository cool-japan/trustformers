//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use trustformers_core::tensor::Tensor;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    fn sample_with(id: &str, field: &str, values: Vec<f32>, shape: &[usize]) -> DataSample {
        let mut data = HashMap::new();
        data.insert(
            field.to_string(),
            Tensor::from_vec(values, shape).expect("tensor creation failed"),
        );
        DataSample {
            id: id.to_string(),
            data,
            metadata: HashMap::new(),
            timestamp: SystemTime::now(),
        }
    }

    fn minimal_streaming_config(buffer_size: usize) -> StreamingDatasetConfig {
        StreamingDatasetConfig {
            sources: vec![],
            buffer_size,
            prefetch_size: 0,
            shuffle: ShuffleConfig {
                enabled: false,
                buffer_size: 0,
                seed: None,
                strategy: ShuffleStrategy::Random,
            },
            batching: BatchingConfig {
                batch_size: 2,
                dynamic: false,
                max_batch_size: 2,
                strategy: BatchingStrategy::Fixed,
                drop_last: false,
            },
            caching: CachingConfig {
                enabled: false,
                cache_type: CacheType::Memory,
                max_size_gb: 0.0,
                eviction_policy: EvictionPolicy::LRU,
                compression: CompressionConfig {
                    enabled: false,
                    algorithm: CompressionAlgorithm::Zstd,
                    level: 0,
                },
            },
        }
    }

    fn pipeline_with_dataset(buffer_size: usize) -> DataPipeline {
        let config = DataPipelineConfig {
            streaming: minimal_streaming_config(buffer_size),
            augmentation: DynamicAugmentationConfig {
                strategies: vec![],
                adaptive: AdaptiveAugmentationConfig {
                    enabled: false,
                    strategy: AdaptationStrategy::PerformanceBased {
                        target_metric: "loss".to_string(),
                        threshold: 0.0,
                    },
                    update_frequency: 0,
                    metrics: vec![],
                },
                scheduling: AugmentationScheduling {
                    schedule_type: ScheduleType::Fixed,
                    parameters: HashMap::new(),
                },
            },
            curriculum: CurriculumLearningConfig {
                strategy: CurriculumStrategy::Manual { stages: vec![] },
                difficulty_assessment: DifficultyAssessment::Static {
                    score_field: "difficulty".to_string(),
                },
                pacing: PacingFunction {
                    pacing_type: PacingType::Linear,
                    parameters: HashMap::new(),
                },
                scheduling: CurriculumScheduling {
                    strategy: CurriculumSchedulingStrategy::EpochBased,
                    update_frequency: 1,
                },
            },
            active_learning: ActiveLearningConfig {
                query_strategy: QueryStrategy::UncertaintySampling {
                    uncertainty_measure: UncertaintyMeasure::Entropy,
                },
                sampling: SamplingConfig {
                    batch_size: 1,
                    budget: 1,
                    diversity_constraint: None,
                },
                annotation: AnnotationConfig {
                    source: AnnotationSource::Human {
                        annotator_pool: vec![],
                    },
                    quality_control: QualityControl {
                        multi_annotation: false,
                        agreement_threshold: 0.0,
                        assessment_method: QualityAssessmentMethod::InterAnnotatorAgreement,
                    },
                },
                integration: ActiveLearningIntegration {
                    update_frequency: 1,
                    min_new_samples: 1,
                    retrain_from_scratch: false,
                },
            },
            multimodal: MultiModalConfig {
                modalities: vec![],
                fusion_strategy: FusionStrategy::EarlyFusion,
                alignment: AlignmentConfig {
                    method: AlignmentMethod::Timestamp,
                    temporal_alignment: false,
                },
                preprocessing: MultiModalPreprocessing {
                    synchronization: SynchronizationConfig {
                        require_all: false,
                        sync_window: Duration::from_secs(1),
                    },
                    missing_modality_handling: MissingModalityHandling::Skip,
                },
            },
            validation: DataValidationConfig {
                rules: vec![],
                strategy: ValidationStrategy::All,
                error_handling: ErrorHandling::Skip,
            },
            distributed: DistributedProcessingConfig {
                num_workers: 1,
                backend: ProcessingBackend::Threading,
                load_balancing: LoadBalancingStrategy::RoundRobin,
            },
        };
        let pipeline = DataPipeline::new(config);
        pipeline
            .register_dataset("train", minimal_streaming_config(buffer_size))
            .expect("register failed");
        pipeline
    }

    // ── Streaming / batching / validation ────────────────────────────────────

    #[tokio::test]
    async fn test_start_streaming_rejects_an_unknown_dataset() {
        // Regression: `start_streaming` used to return Ok(()) for any string.
        let pipeline = pipeline_with_dataset(8);
        assert!(
            pipeline.start_streaming("does-not-exist").await.is_err(),
            "an unknown dataset id must be an error"
        );
        assert!(pipeline.start_streaming("train").await.is_ok());
    }

    #[tokio::test]
    async fn test_get_batch_returns_the_pushed_samples() {
        // Regression: `get_batch` returned an empty vector no matter what.
        let pipeline = pipeline_with_dataset(8);
        pipeline
            .push_samples(
                "train",
                vec![
                    sample_with("a", "x", vec![1.0, 2.0], &[2]),
                    sample_with("b", "x", vec![3.0, 4.0], &[2]),
                    sample_with("c", "x", vec![5.0, 6.0], &[2]),
                ],
            )
            .expect("push failed");
        pipeline.start_streaming("train").await.expect("start failed");

        let batch = pipeline.get_batch(2).await.expect("get_batch failed");
        assert_eq!(batch.len(), 2, "the batch must contain the pushed samples");
        assert_eq!(batch[0].id, "a");
        assert_eq!(batch[1].id, "b");

        let rest = pipeline.get_batch(2).await.expect("get_batch failed");
        assert_eq!(rest.len(), 1, "only one sample is left");
        assert_eq!(rest[0].id, "c");

        let empty = pipeline.get_batch(2).await.expect("get_batch failed");
        assert!(empty.is_empty(), "a drained stream yields an empty batch");
    }

    #[tokio::test]
    async fn test_get_batch_errors_when_nothing_is_streaming() {
        let pipeline = pipeline_with_dataset(8);
        pipeline
            .push_samples("train", vec![sample_with("a", "x", vec![1.0], &[1])])
            .expect("push failed");
        assert!(
            pipeline.get_batch(1).await.is_err(),
            "get_batch must not silently return an empty stream"
        );
    }

    #[tokio::test]
    async fn test_get_batch_rejects_zero_batch_size() {
        let pipeline = pipeline_with_dataset(8);
        pipeline.start_streaming("train").await.expect("start failed");
        assert!(pipeline.get_batch(0).await.is_err());
    }

    #[tokio::test]
    async fn test_validate_batch_runs_the_configured_rules() {
        // Regression: `validate_batch` returned an empty vector, so nothing was ever checked.
        let pipeline = pipeline_with_dataset(8);
        {
            let mut validator = pipeline.validator.lock().expect("validator lock");
            validator.set_config(DataValidationConfig {
                rules: vec![
                    ValidationRule {
                        name: "x_present".to_string(),
                        rule_type: ValidationRuleType::Schema,
                        severity: ValidationSeverity::Error,
                        parameters: HashMap::from([("field".to_string(), "x".to_string())]),
                    },
                    ValidationRule {
                        name: "x_range".to_string(),
                        rule_type: ValidationRuleType::Range,
                        severity: ValidationSeverity::Error,
                        parameters: HashMap::from([
                            ("field".to_string(), "x".to_string()),
                            ("min".to_string(), "0.0".to_string()),
                            ("max".to_string(), "1.0".to_string()),
                        ]),
                    },
                ],
                strategy: ValidationStrategy::All,
                error_handling: ErrorHandling::Skip,
            });
        }

        let samples = vec![
            sample_with("good", "x", vec![0.25, 0.75], &[2]),
            sample_with("out_of_range", "x", vec![0.25, 9.0], &[2]),
            sample_with("wrong_field", "y", vec![0.25], &[1]),
        ];
        let results = pipeline.validate_batch(&samples).await.expect("validate_batch failed");

        assert_eq!(results.len(), 3, "one result per sample");
        assert!(results[0].is_valid, "in-range sample must pass");
        assert!(!results[1].is_valid, "out-of-range sample must fail");
        assert!(!results[2].is_valid, "missing field must fail");
        assert!(results[1].errors.iter().any(|e| e.rule_name == "x_range"));
        assert!(results[2].errors.iter().any(|e| e.rule_name == "x_present"));
    }

    #[tokio::test]
    async fn test_get_batch_skips_invalid_samples_under_the_skip_policy() {
        let pipeline = pipeline_with_dataset(8);
        {
            let mut validator = pipeline.validator.lock().expect("validator lock");
            validator.set_config(DataValidationConfig {
                rules: vec![ValidationRule {
                    name: "finite".to_string(),
                    rule_type: ValidationRuleType::Quality,
                    severity: ValidationSeverity::Error,
                    parameters: HashMap::from([("field".to_string(), "x".to_string())]),
                }],
                strategy: ValidationStrategy::All,
                error_handling: ErrorHandling::Skip,
            });
        }
        pipeline
            .push_samples(
                "train",
                vec![
                    sample_with("ok", "x", vec![1.0], &[1]),
                    sample_with("nan", "x", vec![f32::NAN], &[1]),
                ],
            )
            .expect("push failed");
        pipeline.start_streaming("train").await.expect("start failed");

        let batch = pipeline.get_batch(2).await.expect("get_batch failed");
        assert_eq!(batch.len(), 1, "the NaN sample must be skipped");
        assert_eq!(batch[0].id, "ok");
    }

    #[tokio::test]
    async fn test_get_batch_aborts_under_the_strict_policy() {
        let pipeline = pipeline_with_dataset(8);
        {
            let mut validator = pipeline.validator.lock().expect("validator lock");
            validator.set_config(DataValidationConfig {
                rules: vec![ValidationRule {
                    name: "finite".to_string(),
                    rule_type: ValidationRuleType::Quality,
                    severity: ValidationSeverity::Error,
                    parameters: HashMap::from([("field".to_string(), "x".to_string())]),
                }],
                strategy: ValidationStrategy::All,
                error_handling: ErrorHandling::Strict,
            });
        }
        pipeline
            .push_samples("train", vec![sample_with("nan", "x", vec![f32::NAN], &[1])])
            .expect("push failed");
        pipeline.start_streaming("train").await.expect("start failed");
        assert!(pipeline.get_batch(1).await.is_err());
    }

    #[test]
    fn test_push_samples_respects_the_buffer_size() {
        let pipeline = pipeline_with_dataset(2);
        assert!(pipeline
            .push_samples(
                "train",
                vec![
                    sample_with("a", "x", vec![1.0], &[1]),
                    sample_with("b", "x", vec![1.0], &[1]),
                    sample_with("c", "x", vec![1.0], &[1]),
                ]
            )
            .is_err());
        assert!(pipeline
            .push_samples("train", vec![sample_with("a", "x", vec![1.0], &[1])])
            .is_ok());
    }

    #[test]
    fn test_push_samples_rejects_unknown_dataset() {
        let pipeline = pipeline_with_dataset(4);
        assert!(pipeline
            .push_samples("nope", vec![sample_with("a", "x", vec![1.0], &[1])])
            .is_err());
    }

    #[test]
    fn test_data_pipeline_creation() {
        let config = DataPipelineConfig {
            streaming: StreamingDatasetConfig {
                sources: vec![],
                buffer_size: 1000,
                prefetch_size: 100,
                shuffle: ShuffleConfig {
                    enabled: true,
                    buffer_size: 1000,
                    strategy: ShuffleStrategy::Random,
                    seed: Some(42),
                },
                batching: BatchingConfig {
                    batch_size: 32,
                    dynamic: false,
                    max_batch_size: 64,
                    strategy: BatchingStrategy::Fixed,
                    drop_last: false,
                },
                caching: CachingConfig {
                    enabled: false,
                    cache_type: CacheType::Memory,
                    max_size_gb: 1.0,
                    eviction_policy: EvictionPolicy::LRU,
                    compression: CompressionConfig {
                        enabled: false,
                        algorithm: CompressionAlgorithm::Gzip,
                        level: 6,
                    },
                },
            },
            augmentation: DynamicAugmentationConfig {
                strategies: vec![],
                adaptive: AdaptiveAugmentationConfig {
                    enabled: false,
                    strategy: AdaptationStrategy::PerformanceBased {
                        target_metric: "accuracy".to_string(),
                        threshold: 0.8,
                    },
                    update_frequency: 100,
                    metrics: vec![],
                },
                scheduling: AugmentationScheduling {
                    schedule_type: ScheduleType::Fixed,
                    parameters: HashMap::new(),
                },
            },
            curriculum: CurriculumLearningConfig {
                strategy: CurriculumStrategy::Manual { stages: vec![] },
                difficulty_assessment: DifficultyAssessment::Static {
                    score_field: "difficulty".to_string(),
                },
                pacing: PacingFunction {
                    pacing_type: PacingType::Linear,
                    parameters: HashMap::new(),
                },
                scheduling: CurriculumScheduling {
                    strategy: CurriculumSchedulingStrategy::EpochBased,
                    update_frequency: 1,
                },
            },
            active_learning: ActiveLearningConfig {
                query_strategy: QueryStrategy::UncertaintySampling {
                    uncertainty_measure: UncertaintyMeasure::Entropy,
                },
                sampling: SamplingConfig {
                    batch_size: 10,
                    budget: 1000,
                    diversity_constraint: None,
                },
                annotation: AnnotationConfig {
                    source: AnnotationSource::Human {
                        annotator_pool: vec![],
                    },
                    quality_control: QualityControl {
                        multi_annotation: false,
                        agreement_threshold: 0.8,
                        assessment_method: QualityAssessmentMethod::InterAnnotatorAgreement,
                    },
                },
                integration: ActiveLearningIntegration {
                    update_frequency: 100,
                    min_new_samples: 10,
                    retrain_from_scratch: false,
                },
            },
            multimodal: MultiModalConfig {
                modalities: vec![],
                fusion_strategy: FusionStrategy::EarlyFusion,
                alignment: AlignmentConfig {
                    method: AlignmentMethod::Timestamp,
                    temporal_alignment: false,
                },
                preprocessing: MultiModalPreprocessing {
                    synchronization: SynchronizationConfig {
                        require_all: true,
                        sync_window: Duration::from_secs(1),
                    },
                    missing_modality_handling: MissingModalityHandling::Skip,
                },
            },
            validation: DataValidationConfig {
                rules: vec![],
                strategy: ValidationStrategy::All,
                error_handling: ErrorHandling::LogAndContinue,
            },
            distributed: DistributedProcessingConfig {
                num_workers: 4,
                backend: ProcessingBackend::Threading,
                load_balancing: LoadBalancingStrategy::RoundRobin,
            },
        };

        let pipeline = DataPipeline::new(config);
        assert!(pipeline
            .streaming_datasets
            .lock()
            .expect("lock should not be poisoned")
            .is_empty());
    }

    #[test]
    fn test_augmentation_manager() {
        let manager = DynamicAugmentationManager::new();
        assert!(manager.strategies.is_empty());
        assert_eq!(manager.stats.augmentations_applied.len(), 0);
    }

    #[test]
    fn test_curriculum_manager() {
        let manager = CurriculumLearningManager::new();
        assert_eq!(manager.current_stage, 0);
        assert_eq!(manager.stats.current_difficulty, 0.0);
    }

    #[test]
    fn test_active_learning_manager() {
        let manager = ActiveLearningManager::new();
        assert_eq!(manager.stats.queries_made, 0);
        assert_eq!(manager.stats.annotations_received, 0);
    }

    #[test]
    fn test_multimodal_handler() {
        let handler = MultiModalHandler::new();
        assert!(handler.modality_processors.is_empty());
        assert_eq!(handler.stats.fusion_efficiency, 0.0);
    }

    #[test]
    fn test_data_validator() {
        let validator = DataValidator::new();
        assert!(validator.validators.is_empty());
        assert_eq!(validator.stats.samples_validated, 0);
    }
}
