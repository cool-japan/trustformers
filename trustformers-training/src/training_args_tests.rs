#[cfg(test)]
mod tests {
    use crate::training_args::{EvaluationStrategy, SaveStrategy, TrainingArguments};
    use std::path::PathBuf;

    #[test]
    fn test_training_args_default() {
        let args = TrainingArguments::default();
        assert_eq!(args.output_dir, PathBuf::from("./results"));
        assert!(!args.overwrite_output_dir);
        assert!(!args.do_eval);
        assert!(!args.do_predict);
        assert!(args.do_train);
        assert_eq!(args.learning_rate, 5e-5);
        assert_eq!(args.num_train_epochs, 3.0);
        assert_eq!(args.per_device_train_batch_size, 8);
        assert_eq!(args.per_device_eval_batch_size, 8);
        assert_eq!(args.gradient_accumulation_steps, 1);
        assert_eq!(args.seed, 42);
        assert!(!args.fp16);
        assert!(!args.bf16);
    }

    #[test]
    fn test_training_args_new() {
        let args = TrainingArguments::new("/tmp/output");
        assert_eq!(args.output_dir, PathBuf::from("/tmp/output"));
        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_training_args_validate_success() {
        let args = TrainingArguments::default();
        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_training_args_validate_zero_train_batch() {
        let args = TrainingArguments {
            per_device_train_batch_size: 0,
            ..Default::default()
        };
        assert!(args.validate().is_err());
    }

    #[test]
    fn test_training_args_validate_zero_eval_batch() {
        let args = TrainingArguments {
            per_device_eval_batch_size: 0,
            ..Default::default()
        };
        assert!(args.validate().is_err());
    }

    #[test]
    fn test_training_args_validate_zero_gradient_accumulation() {
        let args = TrainingArguments {
            gradient_accumulation_steps: 0,
            ..Default::default()
        };
        assert!(args.validate().is_err());
    }

    #[test]
    fn test_training_args_validate_negative_lr() {
        let args = TrainingArguments {
            learning_rate: -0.001,
            ..Default::default()
        };
        assert!(args.validate().is_err());
    }

    #[test]
    fn test_training_args_validate_zero_lr() {
        let args = TrainingArguments {
            learning_rate: 0.0,
            ..Default::default()
        };
        assert!(args.validate().is_err());
    }

    #[test]
    fn test_training_args_validate_no_steps_no_epochs() {
        let args = TrainingArguments {
            num_train_epochs: 0.0,
            max_steps: None,
            ..Default::default()
        };
        assert!(args.validate().is_err());
    }

    #[test]
    fn test_training_args_validate_max_steps_overrides_epochs() {
        let args = TrainingArguments {
            num_train_epochs: 0.0,
            max_steps: Some(100),
            ..Default::default()
        };
        // max_steps set, so num_train_epochs doesn't matter
        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_get_total_steps_with_max_steps() {
        let args = TrainingArguments {
            max_steps: Some(500),
            ..Default::default()
        };
        assert_eq!(args.get_total_steps(1000), 500);
    }

    #[test]
    fn test_get_total_steps_from_epochs() {
        let args = TrainingArguments {
            num_train_epochs: 3.0,
            per_device_train_batch_size: 10,
            max_steps: None,
            ..Default::default()
        };
        // 100 examples, batch 10 => 10 steps/epoch * 3 epochs = 30
        assert_eq!(args.get_total_steps(100), 30);
    }

    #[test]
    fn test_get_total_steps_with_remainder() {
        let args = TrainingArguments {
            num_train_epochs: 1.0,
            per_device_train_batch_size: 3,
            max_steps: None,
            ..Default::default()
        };
        // 10 examples, batch 3 => ceil(10/3) = 4 steps/epoch * 1 epoch = 4
        assert_eq!(args.get_total_steps(10), 4);
    }

    #[test]
    fn test_get_effective_batch_size() {
        let args = TrainingArguments {
            per_device_train_batch_size: 8,
            gradient_accumulation_steps: 4,
            ..Default::default()
        };
        assert_eq!(args.get_effective_batch_size(), 32);
    }

    #[test]
    fn test_get_warmup_steps_from_steps() {
        let args = TrainingArguments {
            warmup_steps: 100,
            warmup_ratio: 0.0,
            ..Default::default()
        };
        assert_eq!(args.get_warmup_steps(1000), 100);
    }

    #[test]
    fn test_get_warmup_steps_from_ratio() {
        let args = TrainingArguments {
            warmup_steps: 0,
            warmup_ratio: 0.1,
            ..Default::default()
        };
        assert_eq!(args.get_warmup_steps(1000), 100);
    }

    #[test]
    fn test_get_warmup_steps_steps_override_ratio() {
        let args = TrainingArguments {
            warmup_steps: 50,
            warmup_ratio: 0.5, // would be 500, but steps overrides
            ..Default::default()
        };
        assert_eq!(args.get_warmup_steps(1000), 50);
    }

    #[test]
    fn test_evaluation_strategy_variants() {
        let _no = EvaluationStrategy::No;
        let _steps = EvaluationStrategy::Steps;
        let _epoch = EvaluationStrategy::Epoch;
        assert_eq!(EvaluationStrategy::No, EvaluationStrategy::No);
        assert_ne!(EvaluationStrategy::No, EvaluationStrategy::Steps);
    }

    #[test]
    fn test_save_strategy_variants() {
        let _no = SaveStrategy::No;
        let _steps = SaveStrategy::Steps;
        let _epoch = SaveStrategy::Epoch;
        assert_eq!(SaveStrategy::Steps, SaveStrategy::Steps);
        assert_ne!(SaveStrategy::No, SaveStrategy::Epoch);
    }

    #[test]
    fn test_training_args_with_eval() {
        let args = TrainingArguments {
            do_eval: true,
            evaluation_strategy: EvaluationStrategy::Steps,
            eval_steps: 100,
            ..Default::default()
        };
        assert!(args.do_eval);
        assert_eq!(args.eval_steps, 100);
    }

    #[test]
    fn test_training_args_with_mixed_precision() {
        let args = TrainingArguments {
            fp16: true,
            ..Default::default()
        };
        assert!(args.fp16);
        assert!(!args.bf16);
    }

    #[test]
    fn test_training_args_with_early_stopping() {
        let args = TrainingArguments {
            early_stopping_patience: Some(5),
            early_stopping_threshold: Some(0.01),
            ..Default::default()
        };
        assert_eq!(args.early_stopping_patience, Some(5));
        assert_eq!(args.early_stopping_threshold, Some(0.01));
    }

    #[test]
    fn test_training_args_with_checkpoint() {
        let args = TrainingArguments {
            save_strategy: SaveStrategy::Steps,
            save_steps: 200,
            save_total_limit: Some(3),
            ..Default::default()
        };
        assert_eq!(args.save_steps, 200);
        assert_eq!(args.save_total_limit, Some(3));
    }

    #[test]
    fn test_training_args_serialization_roundtrip() {
        let args = TrainingArguments::default();
        let json = serde_json::to_string(&args).expect("Failed to serialize");
        let deserialized: TrainingArguments =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.learning_rate, args.learning_rate);
        assert_eq!(deserialized.seed, args.seed);
    }
}
