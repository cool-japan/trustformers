#[cfg(test)]
mod tests {
    use crate::trainer::{
        EarlyStoppingCallback, LogEntry, ModelOutput, TaskType, TrainerCallback, TrainingState,
    };
    use crate::training_args::TrainingArguments;
    use std::collections::HashMap;
    use trustformers_core::Tensor;

    // --- TaskType Tests ---

    #[test]
    fn test_task_type_variants() {
        let lm = TaskType::LanguageModeling;
        let cls = TaskType::Classification;
        let repr = TaskType::Representation;
        assert_eq!(lm, TaskType::LanguageModeling);
        assert_ne!(cls, repr);
    }

    #[test]
    fn test_task_type_clone() {
        let orig = TaskType::Classification;
        let cloned = orig;
        assert_eq!(orig, cloned);
    }

    // --- TrainingState Tests ---

    #[test]
    fn test_training_state_creation() {
        let state = TrainingState {
            epoch: 0.0,
            global_step: 0,
            best_metric: None,
            best_model_checkpoint: None,
            log_history: Vec::new(),
            trial_name: None,
            trial_params: None,
        };
        assert_eq!(state.epoch, 0.0);
        assert_eq!(state.global_step, 0);
        assert!(state.best_metric.is_none());
    }

    #[test]
    fn test_training_state_with_progress() {
        let state = TrainingState {
            epoch: 2.5,
            global_step: 1000,
            best_metric: Some(0.95),
            best_model_checkpoint: Some(std::path::PathBuf::from("/tmp/ckpt")),
            log_history: vec![LogEntry {
                step: 100,
                epoch: 1.0,
                learning_rate: 5e-5,
                loss: 0.5,
                eval_metrics: None,
                train_metrics: None,
            }],
            trial_name: Some("trial_1".to_string()),
            trial_params: None,
        };
        assert_eq!(state.epoch, 2.5);
        assert_eq!(state.global_step, 1000);
        assert_eq!(state.best_metric, Some(0.95));
        assert_eq!(state.log_history.len(), 1);
    }

    #[test]
    fn test_training_state_clone() {
        let state = TrainingState {
            epoch: 1.0,
            global_step: 500,
            best_metric: Some(0.9),
            best_model_checkpoint: None,
            log_history: Vec::new(),
            trial_name: None,
            trial_params: None,
        };
        let cloned = state.clone();
        assert_eq!(cloned.epoch, 1.0);
        assert_eq!(cloned.global_step, 500);
    }

    #[test]
    fn test_training_state_serialization() {
        let state = TrainingState {
            epoch: 1.5,
            global_step: 100,
            best_metric: Some(0.88),
            best_model_checkpoint: None,
            log_history: Vec::new(),
            trial_name: None,
            trial_params: None,
        };
        let json = serde_json::to_string(&state).expect("Failed to serialize");
        let deserialized: TrainingState =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.epoch, 1.5);
        assert_eq!(deserialized.global_step, 100);
    }

    // --- LogEntry Tests ---

    #[test]
    fn test_log_entry_creation() {
        let entry = LogEntry {
            step: 50,
            epoch: 0.5,
            learning_rate: 1e-4,
            loss: 2.3,
            eval_metrics: None,
            train_metrics: None,
        };
        assert_eq!(entry.step, 50);
        assert!((entry.loss - 2.3).abs() < 1e-6);
    }

    #[test]
    fn test_log_entry_with_metrics() {
        let mut eval_metrics = HashMap::new();
        eval_metrics.insert("accuracy".to_string(), 0.92);
        eval_metrics.insert("f1".to_string(), 0.89);
        let entry = LogEntry {
            step: 100,
            epoch: 1.0,
            learning_rate: 5e-5,
            loss: 0.5,
            eval_metrics: Some(eval_metrics),
            train_metrics: None,
        };
        let eval = entry.eval_metrics.as_ref().expect("expected eval metrics");
        assert_eq!(eval.len(), 2);
        assert!(eval.contains_key("accuracy"));
    }

    #[test]
    fn test_log_entry_serialization() {
        let entry = LogEntry {
            step: 10,
            epoch: 0.1,
            learning_rate: 3e-4,
            loss: 1.5,
            eval_metrics: None,
            train_metrics: None,
        };
        let json = serde_json::to_string(&entry).expect("Failed to serialize");
        let deserialized: LogEntry = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.step, 10);
    }

    // --- EarlyStoppingCallback Tests ---

    #[test]
    fn test_early_stopping_creation() {
        let cb = EarlyStoppingCallback::new(5, 0.001, "loss".to_string(), false);
        assert!(!cb.should_stop());
    }

    #[test]
    fn test_early_stopping_improves() {
        let mut cb = EarlyStoppingCallback::new(3, 0.0, "accuracy".to_string(), true);
        let args = TrainingArguments::default();
        let state = TrainingState {
            epoch: 1.0,
            global_step: 100,
            best_metric: None,
            best_model_checkpoint: None,
            log_history: Vec::new(),
            trial_name: None,
            trial_params: None,
        };

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.5);
        cb.on_evaluate(&args, &state, &metrics);
        assert!(!cb.should_stop());

        metrics.insert("accuracy".to_string(), 0.6);
        cb.on_evaluate(&args, &state, &metrics);
        assert!(!cb.should_stop());

        metrics.insert("accuracy".to_string(), 0.7);
        cb.on_evaluate(&args, &state, &metrics);
        assert!(!cb.should_stop());
    }

    #[test]
    fn test_early_stopping_triggers() {
        let mut cb = EarlyStoppingCallback::new(2, 0.0, "loss".to_string(), false);
        let args = TrainingArguments::default();
        let state = TrainingState {
            epoch: 1.0,
            global_step: 100,
            best_metric: None,
            best_model_checkpoint: None,
            log_history: Vec::new(),
            trial_name: None,
            trial_params: None,
        };

        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 1.0);
        cb.on_evaluate(&args, &state, &metrics);
        assert!(!cb.should_stop());

        // Loss does not improve (higher)
        metrics.insert("loss".to_string(), 1.1);
        cb.on_evaluate(&args, &state, &metrics);
        assert!(!cb.should_stop());

        // Still no improvement
        metrics.insert("loss".to_string(), 1.2);
        cb.on_evaluate(&args, &state, &metrics);
        // patience=2, waited 2 times without improvement
        assert!(cb.should_stop());
    }

    #[test]
    fn test_early_stopping_resets_on_improvement() {
        let mut cb = EarlyStoppingCallback::new(2, 0.0, "loss".to_string(), false);
        let args = TrainingArguments::default();
        let state = TrainingState {
            epoch: 1.0,
            global_step: 100,
            best_metric: None,
            best_model_checkpoint: None,
            log_history: Vec::new(),
            trial_name: None,
            trial_params: None,
        };

        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 1.0);
        cb.on_evaluate(&args, &state, &metrics);

        metrics.insert("loss".to_string(), 1.1); // no improvement
        cb.on_evaluate(&args, &state, &metrics);

        metrics.insert("loss".to_string(), 0.5); // improvement!
        cb.on_evaluate(&args, &state, &metrics);
        assert!(!cb.should_stop());

        metrics.insert("loss".to_string(), 0.6); // no improvement
        cb.on_evaluate(&args, &state, &metrics);
        assert!(!cb.should_stop());
    }

    #[test]
    fn test_early_stopping_missing_metric() {
        let mut cb = EarlyStoppingCallback::new(2, 0.0, "loss".to_string(), false);
        let args = TrainingArguments::default();
        let state = TrainingState {
            epoch: 1.0,
            global_step: 100,
            best_metric: None,
            best_model_checkpoint: None,
            log_history: Vec::new(),
            trial_name: None,
            trial_params: None,
        };

        let metrics = HashMap::new(); // empty
        cb.on_evaluate(&args, &state, &metrics);
        assert!(!cb.should_stop()); // no crash, no stop
    }

    #[test]
    fn test_early_stopping_with_threshold() {
        let mut cb = EarlyStoppingCallback::new(2, 0.1, "accuracy".to_string(), true);
        let args = TrainingArguments::default();
        let state = TrainingState {
            epoch: 1.0,
            global_step: 100,
            best_metric: None,
            best_model_checkpoint: None,
            log_history: Vec::new(),
            trial_name: None,
            trial_params: None,
        };

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.8);
        cb.on_evaluate(&args, &state, &metrics);

        // Small improvement, below threshold of 0.1
        metrics.insert("accuracy".to_string(), 0.85);
        cb.on_evaluate(&args, &state, &metrics);
        // 0.85 > 0.8 + 0.1? No (0.85 < 0.9), so no improvement
        // wait_count = 1

        metrics.insert("accuracy".to_string(), 0.86);
        cb.on_evaluate(&args, &state, &metrics);
        // 0.86 > 0.8 + 0.1? No, so wait_count = 2 = patience
        assert!(cb.should_stop());
    }

    // --- ModelOutput for Tensor ---

    #[test]
    fn test_tensor_model_output_primary() {
        let tensor = Tensor::zeros(&[2, 3]).expect("Failed to create tensor");
        let primary = tensor.primary_output();
        assert_eq!(primary.shape(), &[2, 3]);
    }

    #[test]
    fn test_tensor_model_output_logits() {
        let tensor = Tensor::zeros(&[2, 3]).expect("Failed to create tensor");
        assert!(tensor.logits().is_some());
    }

    #[test]
    fn test_tensor_model_output_hidden_states() {
        let tensor = Tensor::zeros(&[2, 3]).expect("Failed to create tensor");
        assert!(tensor.hidden_states().is_some());
    }

    #[test]
    fn test_tensor_model_output_pooled_output_is_none() {
        let tensor = Tensor::zeros(&[2, 3]).expect("Failed to create tensor");
        // Default pooled_output returns None for Tensor
        // Actually Tensor impl doesn't override pooled_output
        assert!(tensor.pooled_output().is_none());
    }
}
