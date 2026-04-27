#[cfg(test)]
mod tests {
    use crate::metrics::{Accuracy, F1Score, Metric, MetricCollection, Perplexity};
    use scirs2_core::ndarray::{ArrayD, IxDyn};
    use trustformers_core::Tensor;

    fn make_logits(data: Vec<f32>, shape: &[usize]) -> Tensor {
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).expect("Failed to create logits");
        Tensor::F32(arr)
    }

    fn make_labels_i64(data: Vec<i64>) -> Tensor {
        let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).expect("Failed to create labels");
        Tensor::I64(arr)
    }

    fn make_preds_i64(data: Vec<i64>) -> Tensor {
        let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).expect("Failed to create preds");
        Tensor::I64(arr)
    }

    // --- Accuracy Tests ---

    #[test]
    fn test_accuracy_name() {
        let acc = Accuracy;
        assert_eq!(acc.name(), "accuracy");
    }

    #[test]
    fn test_accuracy_higher_is_better() {
        let acc = Accuracy;
        assert!(acc.higher_is_better());
    }

    #[test]
    fn test_accuracy_perfect_from_logits() {
        let acc = Accuracy;
        let logits = make_logits(
            vec![
                5.0, 1.0, 0.1, // class 0
                0.1, 5.0, 0.1, // class 1
                0.1, 0.1, 5.0, // class 2
            ],
            &[3, 3],
        );
        let targets = make_labels_i64(vec![0, 1, 2]);
        let result = acc.compute(&logits, &targets);
        assert!(result.is_ok());
        let val = result.expect("Accuracy computation failed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_accuracy_zero_from_logits() {
        let acc = Accuracy;
        let logits = make_logits(
            vec![
                0.1, 5.0, 0.1, // class 1 predicted, target 0
                5.0, 0.1, 0.1, // class 0 predicted, target 1
            ],
            &[2, 3],
        );
        let targets = make_labels_i64(vec![0, 1]);
        let result = acc.compute(&logits, &targets);
        assert!(result.is_ok());
        let val = result.expect("Accuracy computation failed");
        assert!(val.abs() < 1e-6);
    }

    #[test]
    fn test_accuracy_partial() {
        let acc = Accuracy;
        let logits = make_logits(
            vec![
                5.0, 1.0, 0.1, // class 0 - correct
                0.1, 5.0, 0.1, // class 1 - correct
                5.0, 0.1, 0.1, // class 0 - wrong (target is 2)
                0.1, 0.1, 5.0, // class 2 - correct
            ],
            &[4, 3],
        );
        let targets = make_labels_i64(vec![0, 1, 2, 2]);
        let result = acc.compute(&logits, &targets);
        assert!(result.is_ok());
        let val = result.expect("Accuracy computation failed");
        assert!((val - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_accuracy_from_class_predictions() {
        let acc = Accuracy;
        let preds = make_preds_i64(vec![0, 1, 2, 2]);
        let targets = make_labels_i64(vec![0, 1, 1, 2]);
        let result = acc.compute(&preds, &targets);
        assert!(result.is_ok());
        let val = result.expect("Accuracy computation failed");
        assert!((val - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_accuracy_wrong_types() {
        let acc = Accuracy;
        let preds = make_logits(vec![1.0, 2.0], &[2]);
        let targets = make_logits(vec![0.0, 1.0], &[2]);
        let result = acc.compute(&preds, &targets);
        assert!(result.is_err());
    }

    // --- F1Score Tests ---

    #[test]
    fn test_f1_score_default() {
        let f1 = F1Score::default();
        assert_eq!(f1.average, "binary");
        assert_eq!(f1.pos_label, Some(1));
    }

    #[test]
    fn test_f1_score_name() {
        let f1 = F1Score::new();
        assert_eq!(f1.name(), "f1");
    }

    #[test]
    fn test_f1_score_higher_is_better() {
        let f1 = F1Score::new();
        assert!(f1.higher_is_better());
    }

    #[test]
    fn test_f1_score_macro() {
        let f1 = F1Score::macro_averaged();
        assert_eq!(f1.average, "macro");
    }

    #[test]
    fn test_f1_score_micro() {
        let f1 = F1Score::micro();
        assert_eq!(f1.average, "micro");
    }

    #[test]
    fn test_f1_score_weighted() {
        let f1 = F1Score::weighted();
        assert_eq!(f1.average, "weighted");
    }

    #[test]
    fn test_f1_binary_perfect() {
        let f1 = F1Score::new();
        let preds = make_preds_i64(vec![0, 1, 1, 0]);
        let targets = make_labels_i64(vec![0, 1, 1, 0]);
        let result = f1.compute(&preds, &targets);
        assert!(result.is_ok());
        let val = result.expect("F1 computation failed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f1_binary_zero() {
        let f1 = F1Score::new();
        // All predictions are wrong for class 1
        let preds = make_preds_i64(vec![0, 0, 0, 0]);
        let targets = make_labels_i64(vec![1, 1, 1, 1]);
        let result = f1.compute(&preds, &targets);
        assert!(result.is_ok());
        let val = result.expect("F1 computation failed");
        assert!(val.abs() < 1e-6);
    }

    #[test]
    fn test_f1_macro_multiclass() {
        let f1 = F1Score::macro_averaged();
        let preds = make_preds_i64(vec![0, 1, 2, 0, 1, 2]);
        let targets = make_labels_i64(vec![0, 1, 2, 0, 1, 2]);
        let result = f1.compute(&preds, &targets);
        assert!(result.is_ok());
        let val = result.expect("F1 computation failed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f1_micro() {
        let f1 = F1Score::micro();
        let preds = make_preds_i64(vec![0, 1, 2, 0, 1, 2]);
        let targets = make_labels_i64(vec![0, 1, 2, 0, 1, 2]);
        let result = f1.compute(&preds, &targets);
        assert!(result.is_ok());
        let val = result.expect("F1 computation failed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f1_weighted() {
        let f1 = F1Score::weighted();
        let preds = make_preds_i64(vec![0, 1, 2]);
        let targets = make_labels_i64(vec![0, 1, 2]);
        let result = f1.compute(&preds, &targets);
        assert!(result.is_ok());
    }

    #[test]
    fn test_f1_from_logits() {
        let f1 = F1Score::new();
        let logits = make_logits(
            vec![0.1, 5.0, 5.0, 0.1],
            &[2, 2],
        );
        let targets = make_labels_i64(vec![1, 0]);
        let result = f1.compute(&logits, &targets);
        assert!(result.is_ok());
    }

    // --- Perplexity Tests ---

    #[test]
    fn test_perplexity_name() {
        let ppl = Perplexity;
        assert_eq!(ppl.name(), "perplexity");
    }

    #[test]
    fn test_perplexity_lower_is_better() {
        let ppl = Perplexity;
        assert!(!ppl.higher_is_better());
    }

    #[test]
    fn test_perplexity_compute() {
        let ppl = Perplexity;
        let logits = make_logits(
            vec![5.0, 0.1, 0.1, 0.1, 5.0, 0.1],
            &[2, 3],
        );
        let targets = make_labels_i64(vec![0, 1]);
        let result = ppl.compute(&logits, &targets);
        assert!(result.is_ok());
        let val = result.expect("Perplexity computation failed");
        // Perplexity = exp(loss), should be > 1
        assert!(val >= 1.0);
    }

    #[test]
    fn test_perplexity_wrong_types() {
        let ppl = Perplexity;
        let preds = make_preds_i64(vec![1, 2]);
        let targets = make_labels_i64(vec![1, 2]);
        let result = ppl.compute(&preds, &targets);
        assert!(result.is_err());
    }

    // --- MetricCollection Tests ---

    #[test]
    fn test_metric_collection_new() {
        let collection = MetricCollection::new();
        // Should be empty
        let logits = make_logits(vec![5.0, 0.1], &[1, 2]);
        let targets = make_labels_i64(vec![0]);
        let results = collection.compute_all(&logits, &targets);
        assert!(results.is_ok());
        let results = results.expect("Compute all failed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_metric_collection_add_metric() {
        let collection = MetricCollection::new()
            .add_metric(Box::new(Accuracy));
        let logits = make_logits(
            vec![5.0, 0.1, 0.1, 5.0],
            &[2, 2],
        );
        let targets = make_labels_i64(vec![0, 1]);
        let results = collection
            .compute_all(&logits, &targets)
            .expect("Compute all failed");
        assert!(results.contains_key("accuracy"));
    }

    #[test]
    fn test_metric_collection_add_metric_mut() {
        let mut collection = MetricCollection::new();
        collection.add_metric_mut(Box::new(Accuracy));
        collection.add_metric_mut(Box::new(F1Score::new()));
        let logits = make_logits(
            vec![5.0, 0.1, 0.1, 5.0],
            &[2, 2],
        );
        let targets = make_labels_i64(vec![0, 1]);
        let results = collection
            .compute_all(&logits, &targets)
            .expect("Compute all failed");
        assert_eq!(results.len(), 2);
        assert!(results.contains_key("accuracy"));
        assert!(results.contains_key("f1"));
    }

    #[test]
    fn test_metric_collection_default() {
        let collection = MetricCollection::default();
        let logits = make_logits(vec![5.0, 0.1], &[1, 2]);
        let targets = make_labels_i64(vec![0]);
        let results = collection
            .compute_all(&logits, &targets)
            .expect("Compute all failed");
        assert!(results.is_empty());
    }
}
