// Evaluation harness for running benchmarks
use crate::evaluation::benchmarks::{
    GLUEEvaluator, GLUETask, HellaSwagEvaluator, HumanEvalEvaluator, MMLUEvaluator,
    SuperGLUEEvaluator, SuperGLUETask,
};
use crate::evaluation::EvaluationModel;
use crate::evaluation::{EvaluationConfig, EvaluationSuite, Evaluator};
use anyhow::Result;
use serde_json;
use std::collections::HashMap;

/// Main evaluation harness
pub struct EvaluationHarness {
    evaluators: HashMap<String, Box<dyn Evaluator>>,
    config: EvaluationConfig,
}

impl EvaluationHarness {
    pub fn new() -> Self {
        let mut harness = Self {
            evaluators: HashMap::new(),
            config: EvaluationConfig::default(),
        };

        // Register default evaluators
        harness.register_glue_evaluator();
        harness.register_superglue_evaluator();
        harness.register_mmlu_evaluator();
        harness.register_hellaswag_evaluator();
        harness.register_humaneval_evaluator();

        harness
    }

    pub fn with_config(mut self, config: EvaluationConfig) -> Self {
        self.config = config;
        self
    }

    /// Register GLUE evaluator with all tasks
    pub fn register_glue_evaluator(&mut self) {
        let evaluator = GLUEEvaluator::new();
        self.evaluators.insert("glue".to_string(), Box::new(evaluator));
    }

    /// Register GLUE evaluator with specific tasks
    pub fn register_glue_tasks(&mut self, tasks: Vec<GLUETask>) {
        let evaluator = GLUEEvaluator::new().with_tasks(tasks);
        self.evaluators.insert("glue".to_string(), Box::new(evaluator));
    }

    /// Register SuperGLUE evaluator with all tasks
    pub fn register_superglue_evaluator(&mut self) {
        let evaluator = SuperGLUEEvaluator::new();
        self.evaluators.insert("superglue".to_string(), Box::new(evaluator));
    }

    /// Register SuperGLUE evaluator with specific tasks
    pub fn register_superglue_tasks(&mut self, tasks: Vec<SuperGLUETask>) {
        let evaluator = SuperGLUEEvaluator::new().with_tasks(tasks);
        self.evaluators.insert("superglue".to_string(), Box::new(evaluator));
    }

    /// Register MMLU evaluator with all subjects
    pub fn register_mmlu_evaluator(&mut self) {
        let evaluator = MMLUEvaluator::new();
        self.evaluators.insert("mmlu".to_string(), Box::new(evaluator));
    }

    /// Register MMLU evaluator with specific subjects
    pub fn register_mmlu_subjects(&mut self, subjects: Vec<String>) {
        let evaluator = MMLUEvaluator::new().with_subjects(subjects);
        self.evaluators.insert("mmlu".to_string(), Box::new(evaluator));
    }

    /// Register HellaSwag evaluator
    pub fn register_hellaswag_evaluator(&mut self) {
        let evaluator = HellaSwagEvaluator::new();
        self.evaluators.insert("hellaswag".to_string(), Box::new(evaluator));
    }

    /// Register HumanEval evaluator
    pub fn register_humaneval_evaluator(&mut self) {
        let evaluator = HumanEvalEvaluator::new();
        self.evaluators.insert("humaneval".to_string(), Box::new(evaluator));
    }

    /// Register a custom evaluator
    pub fn register_evaluator(&mut self, name: String, evaluator: Box<dyn Evaluator>) {
        self.evaluators.insert(name, evaluator);
    }

    /// Run evaluation on all registered evaluators
    pub fn evaluate_all(&self, model: &dyn EvaluationModel) -> Result<EvaluationSuite> {
        let mut combined_suite = EvaluationSuite::new();

        for (evaluator_name, evaluator) in &self.evaluators {
            println!("Running evaluator: {}", evaluator_name);
            let suite = evaluator.evaluate(model, &self.config)?;

            // Add all results to combined suite
            for result in suite.results {
                combined_suite.add_result(result);
            }
        }

        Ok(combined_suite)
    }

    /// Run evaluation on a specific evaluator
    pub fn evaluate_with(
        &self,
        model: &dyn EvaluationModel,
        evaluator_name: &str,
    ) -> Result<EvaluationSuite> {
        let evaluator = self
            .evaluators
            .get(evaluator_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown evaluator: {}", evaluator_name))?;

        evaluator.evaluate(model, &self.config)
    }

    /// Run evaluation on specific tasks
    pub fn evaluate_tasks(
        &self,
        model: &dyn EvaluationModel,
        task_names: &[String],
    ) -> Result<EvaluationSuite> {
        let mut combined_suite = EvaluationSuite::new();

        for task_name in task_names {
            let result = self.evaluate_single_task(model, task_name)?;
            combined_suite.add_result(result);
        }

        Ok(combined_suite)
    }

    /// Run evaluation on a single task
    pub fn evaluate_single_task(
        &self,
        model: &dyn EvaluationModel,
        task_name: &str,
    ) -> Result<crate::evaluation::EvaluationResult> {
        // Determine which evaluator handles this task
        for evaluator in self.evaluators.values() {
            let supported_tasks = evaluator.supported_tasks();
            if supported_tasks.contains(&task_name.to_string()) {
                return evaluator.evaluate_single_task(model, task_name, &self.config);
            }
        }

        Err(anyhow::anyhow!(
            "No evaluator found for task: {}",
            task_name
        ))
    }

    /// List all available tasks
    pub fn list_tasks(&self) -> Vec<String> {
        let mut all_tasks = Vec::new();

        for evaluator in self.evaluators.values() {
            all_tasks.extend(evaluator.supported_tasks());
        }

        all_tasks.sort();
        all_tasks.dedup();
        all_tasks
    }

    /// List tasks by evaluator
    pub fn list_tasks_by_evaluator(&self) -> HashMap<String, Vec<String>> {
        let mut tasks_by_evaluator = HashMap::new();

        for (evaluator_name, evaluator) in &self.evaluators {
            tasks_by_evaluator.insert(evaluator_name.clone(), evaluator.supported_tasks());
        }

        tasks_by_evaluator
    }

    /// Save evaluation results to file
    pub fn save_results(&self, suite: &EvaluationSuite, output_path: &str) -> Result<()> {
        // Create summary data structure for JSON serialization
        let mut summary_data = serde_json::Map::new();

        // Add timestamp
        summary_data.insert(
            "timestamp".to_string(),
            serde_json::Value::String(suite.timestamp.to_rfc3339()),
        );

        // Add overall summary
        let mut summary_metrics = serde_json::Map::new();
        for (metric, value) in &suite.summary {
            summary_metrics.insert(
                metric.clone(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(*value)
                        .unwrap_or_else(|| serde_json::Number::from(0)),
                ),
            );
        }
        summary_data.insert(
            "summary".to_string(),
            serde_json::Value::Object(summary_metrics),
        );

        // Add task results
        let mut task_results = Vec::new();
        for result in &suite.results {
            let mut task_data = serde_json::Map::new();
            task_data.insert(
                "task_name".to_string(),
                serde_json::Value::String(result.task_name.clone()),
            );

            // Add metrics
            let mut metrics_data = serde_json::Map::new();
            for (metric, value) in &result.metrics {
                metrics_data.insert(
                    metric.clone(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(*value)
                            .unwrap_or_else(|| serde_json::Number::from(0)),
                    ),
                );
            }
            task_data.insert(
                "metrics".to_string(),
                serde_json::Value::Object(metrics_data),
            );

            // Add metadata
            task_data.insert(
                "metadata".to_string(),
                serde_json::Value::Object(result.metadata.clone().into_iter().collect()),
            );

            // Optionally add predictions and targets
            if self.config.output_predictions {
                task_data.insert(
                    "predictions".to_string(),
                    serde_json::Value::Array(
                        result
                            .predictions
                            .iter()
                            .map(|p| serde_json::Value::String(p.clone()))
                            .collect(),
                    ),
                );
                task_data.insert(
                    "targets".to_string(),
                    serde_json::Value::Array(
                        result
                            .targets
                            .iter()
                            .map(|t| serde_json::Value::String(t.clone()))
                            .collect(),
                    ),
                );
            }

            task_results.push(serde_json::Value::Object(task_data));
        }
        summary_data.insert(
            "results".to_string(),
            serde_json::Value::Array(task_results),
        );

        // Write to file
        let json_string = serde_json::to_string_pretty(&summary_data)?;
        std::fs::write(output_path, json_string)?;

        println!("Results saved to: {}", output_path);
        Ok(())
    }

    /// Load evaluation results from file
    pub fn load_results(&self, input_path: &str) -> Result<EvaluationSuite> {
        let content = std::fs::read_to_string(input_path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;

        let mut suite = EvaluationSuite::new();

        // Parse timestamp
        if let Some(timestamp_str) = data.get("timestamp").and_then(|v| v.as_str()) {
            suite.timestamp =
                chrono::DateTime::parse_from_rfc3339(timestamp_str)?.with_timezone(&chrono::Utc);
        }

        // Parse results
        if let Some(results_array) = data.get("results").and_then(|v| v.as_array()) {
            for result_value in results_array {
                if let Some(result_obj) = result_value.as_object() {
                    let task_name = result_obj
                        .get("task_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let mut metrics = HashMap::new();
                    if let Some(metrics_obj) = result_obj.get("metrics").and_then(|v| v.as_object())
                    {
                        for (metric_name, metric_value) in metrics_obj {
                            if let Some(value) = metric_value.as_f64() {
                                metrics.insert(metric_name.clone(), value);
                            }
                        }
                    }

                    let mut metadata = HashMap::new();
                    if let Some(metadata_obj) =
                        result_obj.get("metadata").and_then(|v| v.as_object())
                    {
                        for (key, value) in metadata_obj {
                            metadata.insert(key.clone(), value.clone());
                        }
                    }

                    let predictions = result_obj
                        .get("predictions")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
                        })
                        .unwrap_or_default();

                    let targets = result_obj
                        .get("targets")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
                        })
                        .unwrap_or_default();

                    let result = crate::evaluation::EvaluationResult {
                        task_name,
                        metrics,
                        predictions,
                        targets,
                        metadata,
                    };

                    suite.add_result(result);
                }
            }
        }

        Ok(suite)
    }

    /// Compare two evaluation results
    pub fn compare_results(
        &self,
        baseline_path: &str,
        current_path: &str,
        output_path: Option<&str>,
    ) -> Result<HashMap<String, f64>> {
        let baseline_suite = self.load_results(baseline_path)?;
        let current_suite = self.load_results(current_path)?;

        let mut comparisons = HashMap::new();

        // Group results by task name
        let mut baseline_by_task = HashMap::new();
        for result in &baseline_suite.results {
            baseline_by_task.insert(result.task_name.clone(), result);
        }

        let mut current_by_task = HashMap::new();
        for result in &current_suite.results {
            current_by_task.insert(result.task_name.clone(), result);
        }

        // Compare metrics for each task
        for (task_name, current_result) in &current_by_task {
            if let Some(baseline_result) = baseline_by_task.get(task_name) {
                for (metric_name, current_value) in &current_result.metrics {
                    if let Some(baseline_value) = baseline_result.metrics.get(metric_name) {
                        let diff = current_value - baseline_value;
                        let relative_diff =
                            if baseline_value.abs() > 1e-10 { diff / baseline_value } else { diff };

                        let comparison_key = format!("{}_{}", task_name, metric_name);
                        comparisons.insert(format!("{}_absolute_diff", comparison_key), diff);
                        comparisons
                            .insert(format!("{}_relative_diff", comparison_key), relative_diff);
                    }
                }
            }
        }

        // Save comparison if output path provided
        if let Some(output_path) = output_path {
            let mut comparison_data = serde_json::Map::new();
            comparison_data.insert(
                "baseline_file".to_string(),
                serde_json::Value::String(baseline_path.to_string()),
            );
            comparison_data.insert(
                "current_file".to_string(),
                serde_json::Value::String(current_path.to_string()),
            );
            comparison_data.insert(
                "comparison_timestamp".to_string(),
                serde_json::Value::String(chrono::Utc::now().to_rfc3339()),
            );

            let mut comparisons_obj = serde_json::Map::new();
            for (key, value) in &comparisons {
                comparisons_obj.insert(
                    key.clone(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(*value)
                            .unwrap_or_else(|| serde_json::Number::from(0)),
                    ),
                );
            }
            comparison_data.insert(
                "comparisons".to_string(),
                serde_json::Value::Object(comparisons_obj),
            );

            let json_string = serde_json::to_string_pretty(&comparison_data)?;
            std::fs::write(output_path, json_string)?;

            println!("Comparison saved to: {}", output_path);
        }

        Ok(comparisons)
    }

    /// Generate a quick evaluation report
    pub fn quick_eval(
        &self,
        model: &dyn EvaluationModel,
        tasks: Option<Vec<String>>,
    ) -> Result<()> {
        let suite = if let Some(task_names) = tasks {
            self.evaluate_tasks(model, &task_names)?
        } else {
            self.evaluate_all(model)?
        };

        suite.print_summary();

        // Save results if configured
        if self.config.save_results {
            let output_dir = self.config.output_dir.as_deref().unwrap_or("./eval_results");
            std::fs::create_dir_all(output_dir)?;

            let timestamp = suite.timestamp.format("%Y%m%d_%H%M%S");
            let output_path = format!("{}/evaluation_{}.json", output_dir, timestamp);
            self.save_results(&suite, &output_path)?;
        }

        Ok(())
    }
}

impl Default for EvaluationHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::benchmarks::GLUETask;

    #[test]
    fn test_harness_creation() {
        let harness = EvaluationHarness::new();
        assert!(harness.evaluators.contains_key("glue"));
        assert!(harness.evaluators.contains_key("superglue"));
        assert!(harness.evaluators.contains_key("mmlu"));
        assert!(harness.evaluators.contains_key("hellaswag"));
        assert!(harness.evaluators.contains_key("humaneval"));
    }

    #[test]
    fn test_harness_with_config() {
        let config = EvaluationConfig {
            batch_size: 16,
            max_length: 1024,
            num_samples: Some(50),
            ..Default::default()
        };

        let harness = EvaluationHarness::new().with_config(config);
        assert_eq!(harness.config.batch_size, 16);
        assert_eq!(harness.config.max_length, 1024);
        assert_eq!(harness.config.num_samples, Some(50));
    }

    #[test]
    fn test_task_listing() {
        let harness = EvaluationHarness::new();
        let all_tasks = harness.list_tasks();

        // Should have GLUE, SuperGLUE, MMLU, HellaSwag, and HumanEval tasks
        assert!(!all_tasks.is_empty());
        assert!(all_tasks.iter().any(|task| task.starts_with("glue_")));
        assert!(all_tasks.iter().any(|task| task.starts_with("superglue_")));
        assert!(all_tasks.iter().any(|task| task.starts_with("mmlu_")));
        assert!(all_tasks.iter().any(|task| task == "hellaswag"));
        assert!(all_tasks.iter().any(|task| task == "humaneval"));
    }

    #[test]
    fn test_tasks_by_evaluator() {
        let harness = EvaluationHarness::new();
        let tasks_by_evaluator = harness.list_tasks_by_evaluator();

        assert!(tasks_by_evaluator.contains_key("glue"));
        assert!(tasks_by_evaluator.contains_key("superglue"));
        assert!(tasks_by_evaluator.contains_key("mmlu"));
        assert!(tasks_by_evaluator.contains_key("hellaswag"));
        assert!(tasks_by_evaluator.contains_key("humaneval"));

        let glue_tasks = &tasks_by_evaluator["glue"];
        assert!(!glue_tasks.is_empty());
        assert!(glue_tasks.iter().all(|task| task.starts_with("glue_")));
    }

    #[test]
    fn test_custom_glue_tasks() {
        let mut harness = EvaluationHarness::new();
        harness.register_glue_tasks(vec![GLUETask::SST2, GLUETask::MRPC]);

        let tasks_by_evaluator = harness.list_tasks_by_evaluator();
        let glue_tasks = &tasks_by_evaluator["glue"];

        assert_eq!(glue_tasks.len(), 2);
        assert!(glue_tasks.contains(&"glue_sst2".to_string()));
        assert!(glue_tasks.contains(&"glue_mrpc".to_string()));
    }

    #[test]
    fn test_save_and_load_results() {
        let harness = EvaluationHarness::new();

        // Create a dummy suite
        let mut suite = EvaluationSuite::new();
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.85);

        let result = crate::evaluation::EvaluationResult {
            task_name: "test_task".to_string(),
            metrics,
            predictions: vec!["pos".to_string()],
            targets: vec!["pos".to_string()],
            metadata: HashMap::new(),
        };
        suite.add_result(result);

        // Save and load
        let temp_path = "/tmp/test_results.json";
        harness.save_results(&suite, temp_path).unwrap();
        let loaded_suite = harness.load_results(temp_path).unwrap();

        assert_eq!(loaded_suite.results.len(), 1);
        assert_eq!(loaded_suite.results[0].task_name, "test_task");
        assert_eq!(
            loaded_suite.results[0].metrics.get("accuracy").unwrap(),
            &0.85
        );

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }
}
