//! Standard benchmarks (GLUE, SuperGLUE, MMLU, HellaSwag, HumanEval).
//!
//! Every evaluator here reads its examples from the real benchmark
//! distribution through [`crate::evaluation::dataset_files`]. None of them
//! synthesises inputs or labels: without
//! [`EvaluationConfig::dataset_dir`](crate::evaluation::EvaluationConfig) they
//! return an error, because a number labelled "GLUE CoLA accuracy" that was
//! measured on generated sentences is not a GLUE score.

use crate::evaluation::dataset_files::{self, DatasetSchema};
use crate::evaluation::metrics::{F1Average, MetricCollection};
use crate::evaluation::EvaluationModel;
use crate::evaluation::{EvaluationConfig, EvaluationResult, Evaluator};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// GLUE benchmark tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GLUETask {
    CoLA, // Corpus of Linguistic Acceptability
    SST2, // Stanford Sentiment Treebank
    MRPC, // Microsoft Research Paraphrase Corpus
    STSB, // Semantic Textual Similarity Benchmark
    QQP,  // Quora Question Pairs
    MNLI, // Multi-Genre Natural Language Inference
    QNLI, // Question-answering Natural Language Inference
    RTE,  // Recognizing Textual Entailment
    WNLI, // Winograd Natural Language Inference
}

impl GLUETask {
    pub fn all_tasks() -> Vec<GLUETask> {
        vec![
            GLUETask::CoLA,
            GLUETask::SST2,
            GLUETask::MRPC,
            GLUETask::STSB,
            GLUETask::QQP,
            GLUETask::MNLI,
            GLUETask::QNLI,
            GLUETask::RTE,
            GLUETask::WNLI,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            GLUETask::CoLA => "cola",
            GLUETask::SST2 => "sst2",
            GLUETask::MRPC => "mrpc",
            GLUETask::STSB => "stsb",
            GLUETask::QQP => "qqp",
            GLUETask::MNLI => "mnli",
            GLUETask::QNLI => "qnli",
            GLUETask::RTE => "rte",
            GLUETask::WNLI => "wnli",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            GLUETask::CoLA => "Corpus of Linguistic Acceptability - grammaticality classification",
            GLUETask::SST2 => "Stanford Sentiment Treebank - binary sentiment classification",
            GLUETask::MRPC => "Microsoft Research Paraphrase Corpus - paraphrase detection",
            GLUETask::STSB => "Semantic Textual Similarity Benchmark - similarity scoring",
            GLUETask::QQP => "Quora Question Pairs - duplicate question detection",
            GLUETask::MNLI => "Multi-Genre Natural Language Inference - textual entailment",
            GLUETask::QNLI => "Question-answering Natural Language Inference - QA entailment",
            GLUETask::RTE => "Recognizing Textual Entailment - textual entailment",
            GLUETask::WNLI => "Winograd Natural Language Inference - coreference resolution",
        }
    }

    pub fn is_classification(&self) -> bool {
        match self {
            GLUETask::STSB => false, // Regression task
            _ => true,
        }
    }

    pub fn num_labels(&self) -> usize {
        match self {
            GLUETask::STSB => 1, // Regression
            GLUETask::CoLA
            | GLUETask::SST2
            | GLUETask::MRPC
            | GLUETask::QQP
            | GLUETask::QNLI
            | GLUETask::RTE
            | GLUETask::WNLI => 2, // Binary classification
            GLUETask::MNLI => 3, // 3-way classification (entailment, contradiction, neutral)
        }
    }

    pub fn primary_metric(&self) -> &'static str {
        match self {
            GLUETask::CoLA => "mcc", // Matthews correlation coefficient
            GLUETask::SST2 => "accuracy",
            GLUETask::MRPC | GLUETask::QQP => "f1",
            GLUETask::STSB => "pearson_and_spearman", // Average of Pearson and Spearman correlation
            GLUETask::MNLI | GLUETask::QNLI | GLUETask::RTE | GLUETask::WNLI => "accuracy",
        }
    }
}

/// SuperGLUE benchmark tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SuperGLUETask {
    BoolQ,   // Boolean Questions
    CB,      // CommitmentBank
    COPA,    // Choice of Plausible Alternatives
    MultiRC, // Multi-Sentence Reading Comprehension
    ReCoRD,  // Reading Comprehension with Commonsense Reasoning
    RTE,     // Recognizing Textual Entailment
    WiC,     // Words in Context
    WSC,     // Winograd Schema Challenge
}

impl SuperGLUETask {
    pub fn all_tasks() -> Vec<SuperGLUETask> {
        vec![
            SuperGLUETask::BoolQ,
            SuperGLUETask::CB,
            SuperGLUETask::COPA,
            SuperGLUETask::MultiRC,
            SuperGLUETask::ReCoRD,
            SuperGLUETask::RTE,
            SuperGLUETask::WiC,
            SuperGLUETask::WSC,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            SuperGLUETask::BoolQ => "boolq",
            SuperGLUETask::CB => "cb",
            SuperGLUETask::COPA => "copa",
            SuperGLUETask::MultiRC => "multirc",
            SuperGLUETask::ReCoRD => "record",
            SuperGLUETask::RTE => "rte",
            SuperGLUETask::WiC => "wic",
            SuperGLUETask::WSC => "wsc",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            SuperGLUETask::BoolQ => "Boolean Questions - yes/no question answering",
            SuperGLUETask::CB => "CommitmentBank - 3-way textual entailment",
            SuperGLUETask::COPA => "Choice of Plausible Alternatives - causal reasoning",
            SuperGLUETask::MultiRC => "Multi-Sentence Reading Comprehension - paragraph QA",
            SuperGLUETask::ReCoRD => "Reading Comprehension with Commonsense Reasoning",
            SuperGLUETask::RTE => "Recognizing Textual Entailment",
            SuperGLUETask::WiC => "Words in Context - word sense disambiguation",
            SuperGLUETask::WSC => "Winograd Schema Challenge - coreference resolution",
        }
    }

    pub fn primary_metric(&self) -> &'static str {
        match self {
            SuperGLUETask::BoolQ
            | SuperGLUETask::COPA
            | SuperGLUETask::RTE
            | SuperGLUETask::WiC
            | SuperGLUETask::WSC => "accuracy",
            SuperGLUETask::CB => "f1_macro",
            SuperGLUETask::MultiRC => "f1_and_em", // F1 over all answer-options and exact match
            SuperGLUETask::ReCoRD => "f1_and_em",  // F1 and exact match
        }
    }
}

/// Other popular benchmarks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OtherBenchmark {
    MMLU,       // Massive Multitask Language Understanding
    HellaSwag,  // HellaSwag commonsense reasoning
    HumanEval,  // HumanEval code generation
    TruthfulQA, // TruthfulQA factual accuracy
    GSM8K,      // GSM8K math word problems
    ARC,        // AI2 Reasoning Challenge
}

impl OtherBenchmark {
    pub fn all_benchmarks() -> Vec<OtherBenchmark> {
        vec![
            OtherBenchmark::MMLU,
            OtherBenchmark::HellaSwag,
            OtherBenchmark::HumanEval,
            OtherBenchmark::TruthfulQA,
            OtherBenchmark::GSM8K,
            OtherBenchmark::ARC,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            OtherBenchmark::MMLU => "mmlu",
            OtherBenchmark::HellaSwag => "hellaswag",
            OtherBenchmark::HumanEval => "humaneval",
            OtherBenchmark::TruthfulQA => "truthfulqa",
            OtherBenchmark::GSM8K => "gsm8k",
            OtherBenchmark::ARC => "arc",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            OtherBenchmark::MMLU => {
                "Massive Multitask Language Understanding - 57 academic subjects"
            },
            OtherBenchmark::HellaSwag => {
                "HellaSwag - commonsense reasoning about physical situations"
            },
            OtherBenchmark::HumanEval => "HumanEval - Python code generation from docstrings",
            OtherBenchmark::TruthfulQA => "TruthfulQA - truthfulness in language generation",
            OtherBenchmark::GSM8K => "GSM8K - grade school math word problems",
            OtherBenchmark::ARC => "AI2 Reasoning Challenge - science exam questions",
        }
    }

    pub fn primary_metric(&self) -> &'static str {
        match self {
            OtherBenchmark::MMLU | OtherBenchmark::HellaSwag | OtherBenchmark::ARC => "accuracy",
            OtherBenchmark::HumanEval => "pass_at_1",
            OtherBenchmark::TruthfulQA => "truthfulness",
            OtherBenchmark::GSM8K => "accuracy",
        }
    }
}

/// GLUE benchmark evaluator
pub struct GLUEEvaluator {
    tasks: Vec<GLUETask>,
}

impl Default for GLUEEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl GLUEEvaluator {
    pub fn new() -> Self {
        Self {
            tasks: GLUETask::all_tasks(),
        }
    }

    pub fn with_tasks(mut self, tasks: Vec<GLUETask>) -> Self {
        self.tasks = tasks;
        self
    }

    fn get_metrics_for_task(&self, task: GLUETask) -> MetricCollection {
        match task {
            GLUETask::CoLA => {
                // CoLA uses Matthews Correlation Coefficient as primary metric
                MetricCollection::new().add_accuracy().add_f1(F1Average::Binary)
            },
            GLUETask::SST2 | GLUETask::QNLI | GLUETask::RTE | GLUETask::WNLI => {
                MetricCollection::new().add_accuracy().add_f1(F1Average::Binary)
            },
            GLUETask::MRPC | GLUETask::QQP => {
                MetricCollection::new().add_accuracy().add_f1(F1Average::Binary)
            },
            GLUETask::STSB => {
                // STS-B is a regression task scored by Pearson and Spearman
                // correlation, not accuracy.
                MetricCollection::new().add_pearson().add_spearman()
            },
            GLUETask::MNLI => MetricCollection::new().add_accuracy().add_f1(F1Average::Macro),
        }
    }

    fn evaluate_task(
        &self,
        model: &dyn EvaluationModel,
        task: GLUETask,
        config: &EvaluationConfig,
    ) -> Result<EvaluationResult> {
        // Load dataset (placeholder - would load actual GLUE data)
        let (inputs, targets) = self.load_task_data(task, config)?;

        // Run inference
        let predictions = self.run_inference(model, task, &inputs, config)?;

        // Compute metrics
        let metrics_collection = self.get_metrics_for_task(task);
        let metrics = metrics_collection.compute_all(&predictions, &targets)?;

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert(
            "task_type".to_string(),
            serde_json::Value::String("classification".to_string()),
        );
        metadata.insert(
            "num_labels".to_string(),
            serde_json::Value::Number(task.num_labels().into()),
        );
        metadata.insert(
            "primary_metric".to_string(),
            serde_json::Value::String(task.primary_metric().to_string()),
        );
        metadata.insert(
            "description".to_string(),
            serde_json::Value::String(task.description().to_string()),
        );

        Ok(EvaluationResult {
            task_name: format!("glue_{}", task.name()),
            metrics,
            predictions: if config.output_predictions { predictions } else { Vec::new() },
            targets: if config.output_predictions { targets } else { Vec::new() },
            metadata,
        })
    }

    /// The on-disk layout of one GLUE task.
    ///
    /// Column names follow the official GLUE distribution; CoLA ships without a
    /// header row, so its schema also carries positional indices.
    fn task_schema(task: GLUETask) -> DatasetSchema<'static> {
        const DEFAULT_FILES: &[&str] =
            &["dev.tsv", "validation.tsv", "dev.jsonl", "validation.jsonl"];

        match task {
            GLUETask::CoLA => DatasetSchema {
                subdirectory: Some("cola"),
                file_candidates: DEFAULT_FILES,
                // Headerless layout: source, label, notes, sentence.
                input_fields: &[&["sentence", "3"]],
                label_fields: &["label", "1"],
            },
            GLUETask::SST2 => DatasetSchema {
                subdirectory: Some("sst2"),
                file_candidates: DEFAULT_FILES,
                input_fields: &[&["sentence"]],
                label_fields: &["label"],
            },
            GLUETask::MRPC => DatasetSchema {
                subdirectory: Some("mrpc"),
                file_candidates: DEFAULT_FILES,
                input_fields: &[&["#1 String", "sentence1"], &["#2 String", "sentence2"]],
                label_fields: &["Quality", "label"],
            },
            GLUETask::STSB => DatasetSchema {
                subdirectory: Some("stsb"),
                file_candidates: DEFAULT_FILES,
                input_fields: &[&["sentence1"], &["sentence2"]],
                label_fields: &["score", "label"],
            },
            GLUETask::QQP => DatasetSchema {
                subdirectory: Some("qqp"),
                file_candidates: DEFAULT_FILES,
                input_fields: &[&["question1"], &["question2"]],
                label_fields: &["is_duplicate", "label"],
            },
            GLUETask::MNLI => DatasetSchema {
                subdirectory: Some("mnli"),
                file_candidates: &[
                    "dev_matched.tsv",
                    "dev.tsv",
                    "validation_matched.jsonl",
                    "validation.jsonl",
                ],
                input_fields: &[&["sentence1", "premise"], &["sentence2", "hypothesis"]],
                label_fields: &["gold_label", "label"],
            },
            GLUETask::QNLI => DatasetSchema {
                subdirectory: Some("qnli"),
                file_candidates: DEFAULT_FILES,
                input_fields: &[&["question"], &["sentence"]],
                label_fields: &["label"],
            },
            GLUETask::RTE => DatasetSchema {
                subdirectory: Some("rte"),
                file_candidates: DEFAULT_FILES,
                input_fields: &[&["sentence1"], &["sentence2"]],
                label_fields: &["label"],
            },
            GLUETask::WNLI => DatasetSchema {
                subdirectory: Some("wnli"),
                file_candidates: DEFAULT_FILES,
                input_fields: &[&["sentence1"], &["sentence2"]],
                label_fields: &["label"],
            },
        }
    }

    /// Load a GLUE task's real dev split.
    ///
    /// Errors when `config.dataset_dir` is unset or the task's files are
    /// missing. Nothing is generated.
    fn load_task_data(
        &self,
        task: GLUETask,
        config: &EvaluationConfig,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let root = config.require_dataset_dir(&format!("GLUE {}", task.name()))?;
        let schema = Self::task_schema(task);
        let examples = dataset_files::load_examples(root, &schema, config.num_samples)?;

        let mut inputs = Vec::with_capacity(examples.len());
        let mut targets = Vec::with_capacity(examples.len());
        for example in examples {
            inputs.push(example.input);
            targets.push(example.target);
        }

        Ok((inputs, targets))
    }

    fn run_inference(
        &self,
        model: &dyn EvaluationModel,
        _task: GLUETask,
        inputs: &[String],
        _config: &EvaluationConfig,
    ) -> Result<Vec<String>> {
        // Run inference through the model for each input
        let mut predictions = Vec::new();
        for input in inputs {
            let prediction = model.forward(input)?;
            predictions.push(prediction);
        }
        Ok(predictions)
    }
}

impl Evaluator for GLUEEvaluator {
    fn evaluate(
        &self,
        model: &dyn EvaluationModel,
        config: &EvaluationConfig,
    ) -> Result<crate::evaluation::EvaluationSuite> {
        let mut suite = crate::evaluation::EvaluationSuite::new();

        for &task in &self.tasks {
            tracing::info!(
                "Evaluating GLUE task: {} - {}",
                task.name(),
                task.description()
            );
            let result = self.evaluate_task(model, task, config)?;
            suite.add_result(result);
        }

        Ok(suite)
    }

    fn supported_tasks(&self) -> Vec<String> {
        self.tasks.iter().map(|task| format!("glue_{}", task.name())).collect()
    }

    fn evaluate_single_task(
        &self,
        model: &dyn EvaluationModel,
        task_name: &str,
        config: &EvaluationConfig,
    ) -> Result<EvaluationResult> {
        let task_suffix = task_name
            .strip_prefix("glue_")
            .ok_or_else(|| anyhow::anyhow!("Task name must start with 'glue_'"))?;

        let task = GLUETask::all_tasks()
            .into_iter()
            .find(|t| t.name() == task_suffix)
            .ok_or_else(|| anyhow::anyhow!("Unknown GLUE task: {}", task_suffix))?;

        self.evaluate_task(model, task, config)
    }
}

/// SuperGLUE benchmark evaluator
pub struct SuperGLUEEvaluator {
    tasks: Vec<SuperGLUETask>,
}

impl Default for SuperGLUEEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl SuperGLUEEvaluator {
    pub fn new() -> Self {
        Self {
            tasks: SuperGLUETask::all_tasks(),
        }
    }

    pub fn with_tasks(mut self, tasks: Vec<SuperGLUETask>) -> Self {
        self.tasks = tasks;
        self
    }
}

impl SuperGLUEEvaluator {
    /// The on-disk layout of one SuperGLUE task (the official distribution
    /// ships `val.jsonl` per task directory).
    fn task_schema(task: SuperGLUETask) -> DatasetSchema<'static> {
        const FILES: &[&str] = &["val.jsonl", "validation.jsonl", "dev.jsonl"];

        match task {
            SuperGLUETask::BoolQ => DatasetSchema {
                subdirectory: Some("BoolQ"),
                file_candidates: FILES,
                input_fields: &[&["passage"], &["question"]],
                label_fields: &["label"],
            },
            SuperGLUETask::CB => DatasetSchema {
                subdirectory: Some("CB"),
                file_candidates: FILES,
                input_fields: &[&["premise"], &["hypothesis"]],
                label_fields: &["label"],
            },
            SuperGLUETask::COPA => DatasetSchema {
                subdirectory: Some("COPA"),
                file_candidates: FILES,
                input_fields: &[&["premise"], &["choice1"], &["choice2"], &["question"]],
                label_fields: &["label"],
            },
            SuperGLUETask::MultiRC => DatasetSchema {
                subdirectory: Some("MultiRC"),
                file_candidates: FILES,
                input_fields: &[&["passage", "paragraph"], &["question"]],
                label_fields: &["label"],
            },
            SuperGLUETask::ReCoRD => DatasetSchema {
                subdirectory: Some("ReCoRD"),
                file_candidates: FILES,
                input_fields: &[&["passage"], &["qas", "query"]],
                label_fields: &["label", "answers"],
            },
            SuperGLUETask::RTE => DatasetSchema {
                subdirectory: Some("RTE"),
                file_candidates: FILES,
                input_fields: &[&["premise"], &["hypothesis"]],
                label_fields: &["label"],
            },
            SuperGLUETask::WiC => DatasetSchema {
                subdirectory: Some("WiC"),
                file_candidates: FILES,
                input_fields: &[&["word"], &["sentence1"], &["sentence2"]],
                label_fields: &["label"],
            },
            SuperGLUETask::WSC => DatasetSchema {
                subdirectory: Some("WSC"),
                file_candidates: FILES,
                input_fields: &[&["text"], &["target"]],
                label_fields: &["label"],
            },
        }
    }

    fn evaluate_task(
        &self,
        model: &dyn EvaluationModel,
        task: SuperGLUETask,
        config: &EvaluationConfig,
    ) -> Result<EvaluationResult> {
        let root = config.require_dataset_dir(&format!("SuperGLUE {}", task.name()))?;
        let schema = Self::task_schema(task);
        let examples = dataset_files::load_examples(root, &schema, config.num_samples)?;

        let mut predictions = Vec::with_capacity(examples.len());
        let mut targets = Vec::with_capacity(examples.len());
        for example in &examples {
            predictions.push(model.forward(&example.input)?);
            targets.push(example.target.clone());
        }

        let metrics = MetricCollection::new()
            .add_accuracy()
            .add_f1(F1Average::Macro)
            .compute_all(&predictions, &targets)?;

        let mut metadata = HashMap::new();
        metadata.insert(
            "description".to_string(),
            serde_json::Value::String(task.description().to_string()),
        );
        metadata.insert(
            "num_examples".to_string(),
            serde_json::Value::Number(examples.len().into()),
        );

        Ok(EvaluationResult {
            task_name: format!("superglue_{}", task.name()),
            metrics,
            predictions: if config.output_predictions { predictions } else { Vec::new() },
            targets: if config.output_predictions { targets } else { Vec::new() },
            metadata,
        })
    }
}

impl Evaluator for SuperGLUEEvaluator {
    fn evaluate(
        &self,
        model: &dyn EvaluationModel,
        config: &EvaluationConfig,
    ) -> Result<crate::evaluation::EvaluationSuite> {
        let mut suite = crate::evaluation::EvaluationSuite::new();

        for &task in &self.tasks {
            tracing::info!(task = task.name(), "evaluating SuperGLUE task");
            let result = self.evaluate_task(model, task, config)?;
            suite.add_result(result);
        }

        Ok(suite)
    }

    fn supported_tasks(&self) -> Vec<String> {
        self.tasks.iter().map(|task| format!("superglue_{}", task.name())).collect()
    }

    fn evaluate_single_task(
        &self,
        model: &dyn EvaluationModel,
        task_name: &str,
        config: &EvaluationConfig,
    ) -> Result<EvaluationResult> {
        let suffix = task_name
            .strip_prefix("superglue_")
            .ok_or_else(|| anyhow!("Task name must start with 'superglue_'"))?;
        let task = SuperGLUETask::all_tasks()
            .into_iter()
            .find(|candidate| candidate.name() == suffix)
            .ok_or_else(|| anyhow!("Unknown SuperGLUE task: {}", suffix))?;

        self.evaluate_task(model, task, config)
    }
}

/// MMLU (Massive Multitask Language Understanding) benchmark evaluator
pub struct MMLUEvaluator {
    subjects: Vec<String>,
}

impl Default for MMLUEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl MMLUEvaluator {
    pub fn new() -> Self {
        Self {
            subjects: Self::all_subjects(),
        }
    }

    pub fn with_subjects(mut self, subjects: Vec<String>) -> Self {
        self.subjects = subjects;
        self
    }

    fn all_subjects() -> Vec<String> {
        vec![
            // STEM
            "abstract_algebra".to_string(),
            "anatomy".to_string(),
            "astronomy".to_string(),
            "college_biology".to_string(),
            "college_chemistry".to_string(),
            "college_computer_science".to_string(),
            "college_mathematics".to_string(),
            "college_physics".to_string(),
            "computer_security".to_string(),
            "conceptual_physics".to_string(),
            "electrical_engineering".to_string(),
            "elementary_mathematics".to_string(),
            "high_school_biology".to_string(),
            "high_school_chemistry".to_string(),
            "high_school_computer_science".to_string(),
            "high_school_mathematics".to_string(),
            "high_school_physics".to_string(),
            "high_school_statistics".to_string(),
            "machine_learning".to_string(),
            // Humanities
            "formal_logic".to_string(),
            "high_school_european_history".to_string(),
            "high_school_us_history".to_string(),
            "high_school_world_history".to_string(),
            "international_law".to_string(),
            "jurisprudence".to_string(),
            "logical_fallacies".to_string(),
            "moral_disputes".to_string(),
            "moral_scenarios".to_string(),
            "philosophy".to_string(),
            "prehistory".to_string(),
            "professional_law".to_string(),
            "world_religions".to_string(),
            // Social Sciences
            "econometrics".to_string(),
            "high_school_geography".to_string(),
            "high_school_government_and_politics".to_string(),
            "high_school_macroeconomics".to_string(),
            "high_school_microeconomics".to_string(),
            "high_school_psychology".to_string(),
            "human_sexuality".to_string(),
            "professional_psychology".to_string(),
            "public_relations".to_string(),
            "security_studies".to_string(),
            "sociology".to_string(),
            "us_foreign_policy".to_string(),
            // Other
            "business_ethics".to_string(),
            "clinical_knowledge".to_string(),
            "college_medicine".to_string(),
            "global_facts".to_string(),
            "human_aging".to_string(),
            "management".to_string(),
            "marketing".to_string(),
            "medical_genetics".to_string(),
            "miscellaneous".to_string(),
            "nutrition".to_string(),
            "professional_accounting".to_string(),
            "professional_medicine".to_string(),
            "virology".to_string(),
        ]
    }

    /// Load one MMLU subject from the real distribution.
    ///
    /// MMLU ships headerless CSV files (`test/<subject>_test.csv`) with the
    /// columns `question, A, B, C, D, answer`. Errors when the dataset
    /// directory is unset or the subject file is missing.
    fn load_subject_data(
        &self,
        subject: &str,
        config: &EvaluationConfig,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let root = config.require_dataset_dir(&format!("MMLU {}", subject))?;
        let test_file = format!("{}_test.csv", subject);
        let validation_file = format!("{}_val.csv", subject);
        let candidates = [test_file.as_str(), validation_file.as_str()];

        // Try the conventional `test/` and `val/` subdirectories, then the root.
        let mut last_error = None;
        for subdirectory in [Some("test"), Some("val"), None] {
            let schema = DatasetSchema {
                subdirectory,
                file_candidates: &candidates,
                // Headerless: question, A, B, C, D, answer.
                input_fields: &[&["0"], &["1"], &["2"], &["3"], &["4"]],
                label_fields: &["5"],
            };
            match dataset_files::load_examples(root, &schema, config.num_samples) {
                Ok(examples) => {
                    let mut questions = Vec::with_capacity(examples.len());
                    let mut answers = Vec::with_capacity(examples.len());
                    for example in examples {
                        // Rebuild the standard multiple-choice prompt from the
                        // real question and its four real options.
                        let parts: Vec<&str> = example.input.split(" [SEP] ").collect();
                        let prompt = match parts.as_slice() {
                            [question, a, b, c, d] => {
                                format!("{}\nA) {}\nB) {}\nC) {}\nD) {}", question, a, b, c, d)
                            },
                            _ => example.input.clone(),
                        };
                        questions.push(prompt);
                        answers.push(example.target);
                    }
                    return Ok((questions, answers));
                },
                Err(error) => last_error = Some(error),
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("no MMLU data found for subject {}", subject)))
    }

    fn evaluate_subject(
        &self,
        model: &dyn EvaluationModel,
        subject: &str,
        config: &EvaluationConfig,
    ) -> Result<EvaluationResult> {
        let (questions, targets) = self.load_subject_data(subject, config)?;

        let mut predictions = Vec::new();
        for question in &questions {
            let prediction = model.forward(question)?;
            // Extract first letter (A, B, C, or D) from prediction
            let answer = prediction
                .chars()
                .find(|c| matches!(c, 'A' | 'B' | 'C' | 'D'))
                .map(|c| c.to_string())
                .unwrap_or_else(|| "A".to_string());
            predictions.push(answer);
        }

        let metrics_collection = MetricCollection::new().add_accuracy();
        let metrics = metrics_collection.compute_all(&predictions, &targets)?;

        let mut metadata = HashMap::new();
        metadata.insert(
            "subject".to_string(),
            serde_json::Value::String(subject.to_string()),
        );
        metadata.insert(
            "task_type".to_string(),
            serde_json::Value::String("multiple_choice".to_string()),
        );
        metadata.insert(
            "num_choices".to_string(),
            serde_json::Value::Number(4.into()),
        );

        Ok(EvaluationResult {
            task_name: format!("mmlu_{}", subject),
            metrics,
            predictions: if config.output_predictions { predictions } else { Vec::new() },
            targets: if config.output_predictions { targets } else { Vec::new() },
            metadata,
        })
    }
}

impl Evaluator for MMLUEvaluator {
    fn evaluate(
        &self,
        model: &dyn EvaluationModel,
        config: &EvaluationConfig,
    ) -> Result<crate::evaluation::EvaluationSuite> {
        let mut suite = crate::evaluation::EvaluationSuite::new();

        for subject in &self.subjects {
            tracing::info!("Evaluating MMLU subject: {}", subject.replace("_", " "));
            let result = self.evaluate_subject(model, subject, config)?;
            suite.add_result(result);
        }

        Ok(suite)
    }

    fn supported_tasks(&self) -> Vec<String> {
        self.subjects.iter().map(|subject| format!("mmlu_{}", subject)).collect()
    }

    fn evaluate_single_task(
        &self,
        model: &dyn EvaluationModel,
        task_name: &str,
        config: &EvaluationConfig,
    ) -> Result<EvaluationResult> {
        let subject = task_name
            .strip_prefix("mmlu_")
            .ok_or_else(|| anyhow::anyhow!("Invalid MMLU task name: {}", task_name))?;

        self.evaluate_subject(model, subject, config)
    }
}

/// HellaSwag benchmark evaluator
pub struct HellaSwagEvaluator {}

impl Default for HellaSwagEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl HellaSwagEvaluator {
    pub fn new() -> Self {
        Self {}
    }

    /// Load the real HellaSwag validation split.
    ///
    /// HellaSwag ships JSONL with `ctx` (or `ctx_a`/`ctx_b`), an `endings`
    /// array and a `label` index. Errors when the dataset directory is unset or
    /// the file is missing.
    fn load_data(&self, config: &EvaluationConfig) -> Result<(Vec<String>, Vec<String>)> {
        let root = config.require_dataset_dir("HellaSwag")?;
        let schema = DatasetSchema {
            subdirectory: None,
            file_candidates: &[
                "hellaswag_val.jsonl",
                "validation.jsonl",
                "val.jsonl",
                "hellaswag/hellaswag_val.jsonl",
            ],
            input_fields: &[&["ctx", "ctx_a", "context"], &["endings"]],
            label_fields: &["label"],
        };

        let examples = dataset_files::load_examples(root, &schema, config.num_samples)?;

        let mut questions = Vec::with_capacity(examples.len());
        let mut answers = Vec::with_capacity(examples.len());
        for example in examples {
            let (context, endings) = example
                .input
                .split_once(" [SEP] ")
                .map(|(context, endings)| (context.to_string(), endings.to_string()))
                .unwrap_or((example.input.clone(), String::new()));

            let mut prompt = context;
            for (index, ending) in endings.split(" | ").enumerate().take(4) {
                let letter = [b'A', b'B', b'C', b'D'][index] as char;
                prompt.push_str(&format!("\n{}) {}", letter, ending));
            }
            questions.push(prompt);

            // HellaSwag labels are 0-based indices; report them as letters so
            // they line up with the model's multiple-choice answer.
            let letter = example
                .target
                .trim()
                .parse::<usize>()
                .ok()
                .and_then(|index| ["A", "B", "C", "D"].get(index).copied())
                .unwrap_or(example.target.trim());
            answers.push(letter.to_string());
        }

        Ok((questions, answers))
    }
}

impl Evaluator for HellaSwagEvaluator {
    fn evaluate(
        &self,
        model: &dyn EvaluationModel,
        config: &EvaluationConfig,
    ) -> Result<crate::evaluation::EvaluationSuite> {
        let mut suite = crate::evaluation::EvaluationSuite::new();

        tracing::info!("Evaluating HellaSwag commonsense reasoning");
        let result = self.evaluate_single_task(model, "hellaswag", config)?;
        suite.add_result(result);

        Ok(suite)
    }

    fn supported_tasks(&self) -> Vec<String> {
        vec!["hellaswag".to_string()]
    }

    fn evaluate_single_task(
        &self,
        model: &dyn EvaluationModel,
        _task_name: &str,
        config: &EvaluationConfig,
    ) -> Result<EvaluationResult> {
        let (questions, targets) = self.load_data(config)?;

        let mut predictions = Vec::new();
        for question in &questions {
            let prediction = model.forward(question)?;
            let answer = prediction
                .chars()
                .find(|c| matches!(c, 'A' | 'B' | 'C' | 'D'))
                .map(|c| c.to_string())
                .unwrap_or_else(|| "A".to_string());
            predictions.push(answer);
        }

        let metrics_collection = MetricCollection::new().add_accuracy();
        let metrics = metrics_collection.compute_all(&predictions, &targets)?;

        let mut metadata = HashMap::new();
        metadata.insert(
            "task_type".to_string(),
            serde_json::Value::String("commonsense_reasoning".to_string()),
        );
        metadata.insert(
            "num_choices".to_string(),
            serde_json::Value::Number(4.into()),
        );
        metadata.insert(
            "description".to_string(),
            serde_json::Value::String(
                "Commonsense reasoning about physical situations".to_string(),
            ),
        );

        Ok(EvaluationResult {
            task_name: "hellaswag".to_string(),
            metrics,
            predictions: if config.output_predictions { predictions } else { Vec::new() },
            targets: if config.output_predictions { targets } else { Vec::new() },
            metadata,
        })
    }
}

/// Runs a candidate program against a problem's unit tests.
///
/// HumanEval's pass@k is defined by *executing* the generated program against
/// the problem's tests. `trustformers-core` does not ship a Python sandbox, so
/// a caller who wants a real pass@k must supply one of these.
pub trait CodeExecutor: Send + Sync {
    /// Run `program` (the prompt plus the model's completion, followed by the
    /// problem's test code) and report whether every test passed.
    fn run(&self, program: &str, entry_point: &str) -> Result<bool>;
}

/// One HumanEval problem, as it appears in `HumanEval.jsonl`.
#[derive(Debug, Clone)]
pub struct HumanEvalProblem {
    /// Problem identifier, e.g. `HumanEval/0`.
    pub task_id: String,
    /// The prompt shown to the model (signature plus docstring).
    pub prompt: String,
    /// The unit-test code appended after the completion.
    pub test: String,
    /// The function the tests call.
    pub entry_point: String,
    /// The reference solution shipped with the benchmark.
    pub canonical_solution: String,
}

/// HumanEval benchmark evaluator for code generation
pub struct HumanEvalEvaluator {
    executor: Option<Box<dyn CodeExecutor>>,
}

impl Default for HumanEvalEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl HumanEvalEvaluator {
    /// Create an evaluator with no code executor.
    ///
    /// Without an executor, [`Evaluator::evaluate_single_task`] errors rather
    /// than approximating pass@k: the previous substring heuristic scored any
    /// syntactically plausible but semantically wrong function as a pass.
    pub fn new() -> Self {
        Self { executor: None }
    }

    /// Supply the sandbox that runs candidate programs against the tests.
    pub fn with_executor(mut self, executor: Box<dyn CodeExecutor>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Load the real `HumanEval.jsonl` problem set.
    fn load_problems(&self, config: &EvaluationConfig) -> Result<Vec<HumanEvalProblem>> {
        let root = config.require_dataset_dir("HumanEval")?;
        let schema = DatasetSchema {
            subdirectory: None,
            file_candidates: &["HumanEval.jsonl", "humaneval.jsonl", "test.jsonl"],
            input_fields: &[&["prompt"]],
            label_fields: &["canonical_solution"],
        };
        let path = dataset_files::resolve_file(root, &schema)?;
        let records = dataset_files::load_jsonl_records(&path, config.num_samples)?;

        let field = |record: &serde_json::Value, name: &str| -> Result<String> {
            record
                .get(name)
                .and_then(|value| value.as_str())
                .map(|value| value.to_string())
                .ok_or_else(|| anyhow!("HumanEval record is missing the '{}' field", name))
        };

        records
            .iter()
            .map(|record| {
                Ok(HumanEvalProblem {
                    task_id: field(record, "task_id").unwrap_or_default(),
                    prompt: field(record, "prompt")?,
                    test: field(record, "test")?,
                    entry_point: field(record, "entry_point")?,
                    canonical_solution: field(record, "canonical_solution").unwrap_or_default(),
                })
            })
            .collect()
    }

    /// Assemble the program that the executor runs for one candidate.
    fn assemble_program(problem: &HumanEvalProblem, completion: &str) -> String {
        format!(
            "{}\n{}\n\n{}\ncheck({})\n",
            problem.prompt, completion, problem.test, problem.entry_point
        )
    }
}

impl Evaluator for HumanEvalEvaluator {
    fn evaluate(
        &self,
        model: &dyn EvaluationModel,
        config: &EvaluationConfig,
    ) -> Result<crate::evaluation::EvaluationSuite> {
        let mut suite = crate::evaluation::EvaluationSuite::new();

        tracing::info!("Evaluating HumanEval code generation");
        let result = self.evaluate_single_task(model, "humaneval", config)?;
        suite.add_result(result);

        Ok(suite)
    }

    fn supported_tasks(&self) -> Vec<String> {
        vec!["humaneval".to_string()]
    }

    fn evaluate_single_task(
        &self,
        model: &dyn EvaluationModel,
        _task_name: &str,
        config: &EvaluationConfig,
    ) -> Result<EvaluationResult> {
        let executor = self.executor.as_ref().ok_or_else(|| {
            anyhow!(
                "HumanEval pass@k requires executing the generated code against the problem's \
                 unit tests. trustformers-core ships no Python sandbox; supply one with \
                 HumanEvalEvaluator::with_executor. No score is produced from a substring \
                 heuristic."
            )
        })?;

        let problems = self.load_problems(config)?;

        let mut predictions = Vec::with_capacity(problems.len());
        let mut targets = Vec::with_capacity(problems.len());
        let mut pass_count = 0usize;

        for problem in &problems {
            let completion = model.forward(&problem.prompt)?;
            let program = Self::assemble_program(problem, &completion);
            if executor.run(&program, &problem.entry_point)? {
                pass_count += 1;
            }
            predictions.push(completion);
            targets.push(problem.canonical_solution.clone());
        }

        let pass_at_1 = if problems.is_empty() {
            return Err(anyhow!("HumanEval dataset contained no problems"));
        } else {
            pass_count as f64 / problems.len() as f64
        };

        let mut metrics = HashMap::new();
        metrics.insert("pass_at_1".to_string(), pass_at_1);
        metrics.insert("total_problems".to_string(), problems.len() as f64);
        metrics.insert("passed_problems".to_string(), pass_count as f64);

        let mut metadata = HashMap::new();
        metadata.insert(
            "task_type".to_string(),
            serde_json::Value::String("code_generation".to_string()),
        );
        metadata.insert(
            "language".to_string(),
            serde_json::Value::String("python".to_string()),
        );
        metadata.insert(
            "description".to_string(),
            serde_json::Value::String("Python code generation from docstrings".to_string()),
        );

        Ok(EvaluationResult {
            task_name: "humaneval".to_string(),
            metrics,
            predictions: if config.output_predictions { predictions } else { Vec::new() },
            targets: if config.output_predictions { targets } else { Vec::new() },
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A model that answers with the label it is told to answer with, so tests
    /// can distinguish "scored on the real data" from "scored on templates".
    struct EchoModel {
        answer: String,
    }

    impl EvaluationModel for EchoModel {
        fn forward(&self, _input: &str) -> Result<String> {
            Ok(self.answer.clone())
        }
    }

    fn dataset_root(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "trustformers_benchmarks_{}_{}",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create_dir_all failed");
        path
    }

    /// Regression test: `load_task_data` used to generate
    /// `"The sentence {i} is grammatically correct."` and alternating `1`/`0`
    /// labels, and `evaluate_task` reported the resulting number as a GLUE
    /// score. Without real data there must now be no score at all.
    #[test]
    fn test_glue_refuses_to_score_without_a_dataset() {
        let evaluator = GLUEEvaluator::new().with_tasks(vec![GLUETask::SST2]);
        let model = EchoModel {
            answer: "1".to_string(),
        };
        let error = evaluator
            .evaluate(&model, &EvaluationConfig::default())
            .expect_err("no GLUE score may be produced from generated sentences");
        assert!(
            error.to_string().contains("dataset_dir"),
            "unexpected error: {error}"
        );
    }

    /// With a real GLUE dev.tsv the evaluator must score the model on that file.
    #[test]
    fn test_glue_scores_the_real_dev_split() {
        let root = dataset_root("glue");
        std::fs::create_dir_all(root.join("sst2")).expect("mkdir failed");
        std::fs::write(
            root.join("sst2/dev.tsv"),
            "sentence	label
first review	1
second review	1
third review	0
",
        )
        .expect("write failed");

        let evaluator = GLUEEvaluator::new().with_tasks(vec![GLUETask::SST2]);
        let model = EchoModel {
            answer: "1".to_string(),
        };
        let config = EvaluationConfig {
            output_predictions: true,
            ..EvaluationConfig::default()
        }
        .with_dataset_dir(&root);

        let suite = evaluator.evaluate(&model, &config).expect("evaluation failed");
        let result = suite
            .results
            .iter()
            .find(|result| result.task_name == "glue_sst2")
            .expect("sst2 result");

        // Two of the three real labels are "1", which the model always answers.
        let accuracy = result.metrics.get("accuracy").copied().expect("accuracy");
        assert!(
            (accuracy - 2.0 / 3.0).abs() < 1e-6,
            "accuracy {accuracy} must come from the three real rows"
        );
        assert_eq!(result.targets, vec!["1", "1", "0"]);

        std::fs::remove_dir_all(&root).ok();
    }

    /// Regression test: STS-B is a correlation task; it used to be scored with
    /// accuracy.
    #[test]
    fn test_stsb_is_scored_with_correlations() {
        let root = dataset_root("stsb");
        std::fs::create_dir_all(root.join("stsb")).expect("mkdir failed");
        std::fs::write(
            root.join("stsb/dev.tsv"),
            "sentence1	sentence2	score
a	b	1.0
c	d	2.0
e	f	3.0
",
        )
        .expect("write failed");

        let evaluator = GLUEEvaluator::new().with_tasks(vec![GLUETask::STSB]);
        let model = EchoModel {
            answer: "2.0".to_string(),
        };
        let config = EvaluationConfig::default().with_dataset_dir(&root);

        // A constant prediction has zero variance, so correlation is undefined
        // and must be reported as an error rather than as an accuracy number.
        let error = evaluator
            .evaluate(&model, &config)
            .expect_err("correlation of a constant prediction is undefined");
        assert!(
            error.to_string().contains("variance"),
            "unexpected: {error}"
        );

        std::fs::remove_dir_all(&root).ok();
    }

    /// Regression test: `SuperGLUEEvaluator::evaluate_single_task` returned a
    /// hardcoded `accuracy: 0.5` for every task without invoking the model.
    #[test]
    fn test_superglue_no_longer_returns_a_constant() {
        let evaluator = SuperGLUEEvaluator::new().with_tasks(vec![SuperGLUETask::RTE]);
        let model = EchoModel {
            answer: "entailment".to_string(),
        };

        let error = evaluator
            .evaluate(&model, &EvaluationConfig::default())
            .expect_err("no SuperGLUE score without SuperGLUE data");
        assert!(error.to_string().contains("dataset_dir"));

        let root = dataset_root("superglue");
        std::fs::create_dir_all(root.join("RTE")).expect("mkdir failed");
        std::fs::write(
            root.join("RTE/val.jsonl"),
            "{\"premise\":\"a\",\"hypothesis\":\"b\",\"label\":\"entailment\"}\n             {\"premise\":\"c\",\"hypothesis\":\"d\",\"label\":\"not_entailment\"}\n",
        )
        .expect("write failed");

        let config = EvaluationConfig::default().with_dataset_dir(&root);
        let result = evaluator
            .evaluate_single_task(&model, "superglue_rte", &config)
            .expect("evaluation failed");
        let accuracy = result.metrics.get("accuracy").copied().expect("accuracy");
        assert!(
            (accuracy - 0.5).abs() < 1e-9,
            "the model got exactly one of two real examples right, got {accuracy}"
        );
        // ... but it is now a measured 0.5, backed by real predictions.
        assert_eq!(
            result.metadata.get("num_examples"),
            Some(&serde_json::Value::Number(2.into()))
        );

        std::fs::remove_dir_all(&root).ok();
    }

    /// Regression test: MMLU/HellaSwag used to score models on generated
    /// multiple-choice questions with a fixed answer key.
    #[test]
    fn test_mmlu_and_hellaswag_require_real_data() {
        let model = EchoModel {
            answer: "A".to_string(),
        };
        let config = EvaluationConfig::default();

        assert!(MMLUEvaluator::new()
            .with_subjects(vec!["anatomy".to_string()])
            .evaluate(&model, &config)
            .is_err());
        assert!(HellaSwagEvaluator::new().evaluate(&model, &config).is_err());
    }

    #[test]
    fn test_mmlu_scores_the_real_csv() {
        let root = dataset_root("mmlu");
        std::fs::create_dir_all(root.join("test")).expect("mkdir failed");
        std::fs::write(
            root.join("test/anatomy_test.csv"),
            "\"Largest organ?\",Skin,Liver,Heart,Lung,A\n\"Pumps blood?\",Skin,Liver,Heart,Lung,C\n",
        )
        .expect("write failed");

        let evaluator = MMLUEvaluator::new().with_subjects(vec!["anatomy".to_string()]);
        let model = EchoModel {
            answer: "A".to_string(),
        };
        let config = EvaluationConfig::default().with_dataset_dir(&root);

        let result = evaluator
            .evaluate_single_task(&model, "mmlu_anatomy", &config)
            .expect("evaluation failed");
        let accuracy = result.metrics.get("accuracy").copied().expect("accuracy");
        assert!(
            (accuracy - 0.5).abs() < 1e-9,
            "one of the two real answers is A, got {accuracy}"
        );

        std::fs::remove_dir_all(&root).ok();
    }

    /// Regression test: HumanEval pass@k used to be a substring heuristic that
    /// scored any plausible-looking function as a pass.
    #[test]
    fn test_humaneval_requires_a_code_executor() {
        let root = dataset_root("humaneval");
        std::fs::write(
            root.join("HumanEval.jsonl"),
            "{\"task_id\":\"HumanEval/0\",\"prompt\":\"def add(a, b):\\n\",             \"test\":\"def check(f):\\n    assert f(1,2)==3\\n\",             \"entry_point\":\"add\",\"canonical_solution\":\"    return a + b\\n\"}\n",
        )
        .expect("write failed");

        let model = EchoModel {
            answer: "    return a + b\n".to_string(),
        };
        let config = EvaluationConfig::default().with_dataset_dir(&root);

        let error = HumanEvalEvaluator::new()
            .evaluate_single_task(&model, "humaneval", &config)
            .expect_err("pass@k needs real execution");
        assert!(
            error.to_string().contains("with_executor"),
            "unexpected: {error}"
        );

        // With an executor the score comes from the executor's verdict.
        struct AlwaysFails;
        impl CodeExecutor for AlwaysFails {
            fn run(&self, program: &str, entry_point: &str) -> Result<bool> {
                assert!(
                    program.contains("return a + b"),
                    "the completion must be assembled in"
                );
                assert_eq!(entry_point, "add");
                Ok(false)
            }
        }

        let result = HumanEvalEvaluator::new()
            .with_executor(Box::new(AlwaysFails))
            .evaluate_single_task(&model, "humaneval", &config)
            .expect("evaluation failed");
        assert_eq!(result.metrics.get("pass_at_1").copied(), Some(0.0));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_glue_tasks() {
        let tasks = GLUETask::all_tasks();
        assert_eq!(tasks.len(), 9);

        assert_eq!(GLUETask::CoLA.name(), "cola");
        assert_eq!(GLUETask::SST2.name(), "sst2");
        assert_eq!(GLUETask::MNLI.name(), "mnli");

        assert!(GLUETask::CoLA.is_classification());
        assert!(!GLUETask::STSB.is_classification());

        assert_eq!(GLUETask::MNLI.num_labels(), 3);
        assert_eq!(GLUETask::SST2.num_labels(), 2);
        assert_eq!(GLUETask::STSB.num_labels(), 1);
    }

    #[test]
    fn test_superglue_tasks() {
        let tasks = SuperGLUETask::all_tasks();
        assert_eq!(tasks.len(), 8);

        assert_eq!(SuperGLUETask::BoolQ.name(), "boolq");
        assert_eq!(SuperGLUETask::CB.name(), "cb");
        assert_eq!(SuperGLUETask::COPA.name(), "copa");

        assert_eq!(SuperGLUETask::BoolQ.primary_metric(), "accuracy");
        assert_eq!(SuperGLUETask::CB.primary_metric(), "f1_macro");
    }

    #[test]
    fn test_other_benchmarks() {
        let benchmarks = OtherBenchmark::all_benchmarks();
        assert_eq!(benchmarks.len(), 6);

        assert_eq!(OtherBenchmark::MMLU.name(), "mmlu");
        assert_eq!(OtherBenchmark::HumanEval.name(), "humaneval");

        assert_eq!(OtherBenchmark::MMLU.primary_metric(), "accuracy");
        assert_eq!(OtherBenchmark::HumanEval.primary_metric(), "pass_at_1");
    }

    #[test]
    fn test_glue_evaluator_creation() {
        let evaluator = GLUEEvaluator::new();
        assert_eq!(evaluator.tasks.len(), 9);

        let custom_evaluator =
            GLUEEvaluator::new().with_tasks(vec![GLUETask::SST2, GLUETask::MRPC]);
        assert_eq!(custom_evaluator.tasks.len(), 2);
    }

    #[test]
    fn test_supported_tasks() {
        let glue_evaluator = GLUEEvaluator::new().with_tasks(vec![GLUETask::SST2, GLUETask::MRPC]);
        let supported = glue_evaluator.supported_tasks();

        assert_eq!(supported.len(), 2);
        assert!(supported.contains(&"glue_sst2".to_string()));
        assert!(supported.contains(&"glue_mrpc".to_string()));
    }

    #[test]
    fn test_mmlu_evaluator_creation() {
        let evaluator = MMLUEvaluator::new();
        let all_subjects = MMLUEvaluator::all_subjects();
        assert_eq!(evaluator.subjects.len(), all_subjects.len());
        assert!(all_subjects.len() > 50); // Should have 57 subjects

        let custom_evaluator =
            evaluator.with_subjects(vec!["abstract_algebra".to_string(), "anatomy".to_string()]);
        assert_eq!(custom_evaluator.subjects.len(), 2);
    }

    #[test]
    fn test_mmlu_supported_tasks() {
        let evaluator = MMLUEvaluator::new().with_subjects(vec!["abstract_algebra".to_string()]);
        let supported = evaluator.supported_tasks();

        assert_eq!(supported.len(), 1);
        assert!(supported.contains(&"mmlu_abstract_algebra".to_string()));
    }

    #[test]
    fn test_hellaswag_evaluator_creation() {
        let evaluator = HellaSwagEvaluator::new();
        let supported = evaluator.supported_tasks();

        assert_eq!(supported.len(), 1);
        assert!(supported.contains(&"hellaswag".to_string()));
    }

    #[test]
    fn test_humaneval_evaluator_creation() {
        let evaluator = HumanEvalEvaluator::new();
        let supported = evaluator.supported_tasks();

        assert_eq!(supported.len(), 1);
        assert!(supported.contains(&"humaneval".to_string()));
    }
}
