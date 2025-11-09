// Standard benchmarks (GLUE, SuperGLUE, etc.)
use crate::evaluation::metrics::{F1Average, MetricCollection};
use crate::evaluation::EvaluationModel;
use crate::evaluation::{EvaluationConfig, EvaluationResult, Evaluator};
use anyhow::Result;
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
                // STSB is a regression task, so we use different metrics
                MetricCollection::new().add_accuracy() // Placeholder - would need correlation metrics
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

    fn load_task_data(
        &self,
        task: GLUETask,
        config: &EvaluationConfig,
    ) -> Result<(Vec<String>, Vec<String>)> {
        // Placeholder implementation - in practice, would load from actual GLUE datasets
        let num_samples = config.num_samples.unwrap_or(100);

        let inputs = match task {
            GLUETask::CoLA => {
                // Single sentence acceptability
                (0..num_samples)
                    .map(|i| format!("The sentence {} is grammatically correct.", i))
                    .collect()
            },
            GLUETask::SST2 => {
                // Sentiment classification
                (0..num_samples)
                    .map(|i| {
                        if i % 2 == 0 {
                            "This movie is absolutely wonderful and entertaining.".to_string()
                        } else {
                            "This movie is terrible and boring.".to_string()
                        }
                    })
                    .collect()
            },
            GLUETask::MRPC => {
                // Paraphrase detection (sentence pairs)
                (0..num_samples)
                    .map(|i| {
                        if i % 2 == 0 {
                            "The cat sat on the mat. [SEP] A cat was sitting on the mat."
                                .to_string()
                        } else {
                            "The dog ran fast. [SEP] The car was red.".to_string()
                        }
                    })
                    .collect()
            },
            GLUETask::MNLI => {
                // Natural language inference
                (0..num_samples)
                    .map(|i| match i % 3 {
                        0 => "A man is eating pizza. [SEP] A person is consuming food.".to_string(),
                        1 => "A man is eating pizza. [SEP] A person is sleeping.".to_string(),
                        _ => "A man is eating pizza. [SEP] A person might be hungry.".to_string(),
                    })
                    .collect()
            },
            _ => {
                // Generic placeholder for other tasks
                (0..num_samples)
                    .map(|i| format!("Input sentence {} for task {}.", i, task.name()))
                    .collect()
            },
        };

        let targets = match task {
            GLUETask::CoLA
            | GLUETask::SST2
            | GLUETask::MRPC
            | GLUETask::QQP
            | GLUETask::QNLI
            | GLUETask::RTE
            | GLUETask::WNLI => {
                // Binary classification
                (0..num_samples)
                    .map(|i| if i % 2 == 0 { "1".to_string() } else { "0".to_string() })
                    .collect()
            },
            GLUETask::MNLI => {
                // 3-way classification
                (0..num_samples)
                    .map(|i| match i % 3 {
                        0 => "entailment".to_string(),
                        1 => "contradiction".to_string(),
                        _ => "neutral".to_string(),
                    })
                    .collect()
            },
            GLUETask::STSB => {
                // Regression (similarity scores)
                (0..num_samples).map(|i| format!("{:.2}", (i % 5) as f64 / 4.0 * 5.0)).collect()
            },
        };

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
            println!(
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

impl Evaluator for SuperGLUEEvaluator {
    fn evaluate(
        &self,
        model: &dyn EvaluationModel,
        config: &EvaluationConfig,
    ) -> Result<crate::evaluation::EvaluationSuite> {
        let mut suite = crate::evaluation::EvaluationSuite::new();

        for &task in &self.tasks {
            println!(
                "Evaluating SuperGLUE task: {} - {}",
                task.name(),
                task.description()
            );
            let result =
                self.evaluate_single_task(model, &format!("superglue_{}", task.name()), config)?;
            suite.add_result(result);
        }

        Ok(suite)
    }

    fn supported_tasks(&self) -> Vec<String> {
        self.tasks.iter().map(|task| format!("superglue_{}", task.name())).collect()
    }

    fn evaluate_single_task(
        &self,
        _model: &dyn EvaluationModel,
        task_name: &str,
        _config: &EvaluationConfig,
    ) -> Result<EvaluationResult> {
        // Placeholder implementation
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.5);

        Ok(EvaluationResult {
            task_name: task_name.to_string(),
            metrics,
            predictions: Vec::new(),
            targets: Vec::new(),
            metadata: HashMap::new(),
        })
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

    fn load_subject_data(
        &self,
        subject: &str,
        config: &EvaluationConfig,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let num_samples = config.num_samples.unwrap_or(50);

        let questions: Vec<String> = (0..num_samples).map(|i| {
            format!(
                "Question {}: What is the primary concept in {}?\nA) Option A\nB) Option B\nC) Option C\nD) Option D",
                i + 1, subject.replace("_", " ")
            )
        }).collect();

        let answers: Vec<String> = (0..num_samples)
            .map(|i| match i % 4 {
                0 => "A".to_string(),
                1 => "B".to_string(),
                2 => "C".to_string(),
                _ => "D".to_string(),
            })
            .collect();

        Ok((questions, answers))
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
            println!("Evaluating MMLU subject: {}", subject.replace("_", " "));
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

    fn load_data(&self, config: &EvaluationConfig) -> Result<(Vec<String>, Vec<String>)> {
        let num_samples = config.num_samples.unwrap_or(100);

        let contexts_and_choices: Vec<String> = (0..num_samples)
            .map(|i| {
                let contexts = [
                    "A person is washing dishes. They pick up a sponge and",
                    "Someone is walking down the street. They see a red light and",
                    "A chef is preparing dinner. They grab a knife and",
                    "A student is studying for exams. They open a book and",
                ];

                let choices = [
                    vec![
                        "start scrubbing the plates clean.",
                        "throw it at the wall.",
                        "eat it like cake.",
                        "use it as a hat.",
                    ],
                    vec![
                        "stop at the crosswalk.",
                        "start dancing wildly.",
                        "climb a tree.",
                        "begin singing opera.",
                    ],
                    vec![
                        "carefully slice the vegetables.",
                        "start juggling with it.",
                        "use it as a comb.",
                        "plant it in soil.",
                    ],
                    vec![
                        "begin reading the first chapter.",
                        "use it as a pillow.",
                        "start eating the pages.",
                        "throw it out the window.",
                    ],
                ];

                let context_idx = i % contexts.len();
                let context = contexts[context_idx];
                let choice_set = &choices[context_idx];

                format!(
                    "{}\nA) {}\nB) {}\nC) {}\nD) {}",
                    context, choice_set[0], choice_set[1], choice_set[2], choice_set[3]
                )
            })
            .collect();

        let answers: Vec<String> = (0..num_samples).map(|_| "A".to_string()).collect();

        Ok((contexts_and_choices, answers))
    }
}

impl Evaluator for HellaSwagEvaluator {
    fn evaluate(
        &self,
        model: &dyn EvaluationModel,
        config: &EvaluationConfig,
    ) -> Result<crate::evaluation::EvaluationSuite> {
        let mut suite = crate::evaluation::EvaluationSuite::new();

        println!("Evaluating HellaSwag commonsense reasoning");
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

/// HumanEval benchmark evaluator for code generation
pub struct HumanEvalEvaluator {}

impl Default for HumanEvalEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl HumanEvalEvaluator {
    pub fn new() -> Self {
        Self {}
    }

    fn load_data(&self, config: &EvaluationConfig) -> Result<(Vec<String>, Vec<String>)> {
        let num_samples = config.num_samples.unwrap_or(20);

        let problems_and_solutions: Vec<(String, String)> = vec![
            (
                "def add(a, b):\n    \"\"\"\n    Add two numbers.\n    \"\"\"\n    # TODO: implement this function\n    pass".to_string(),
                "def add(a, b):\n    \"\"\"\n    Add two numbers.\n    \"\"\"\n    return a + b".to_string()
            ),
            (
                "def multiply(a, b):\n    \"\"\"\n    Multiply two numbers.\n    \"\"\"\n    # TODO: implement this function\n    pass".to_string(),
                "def multiply(a, b):\n    \"\"\"\n    Multiply two numbers.\n    \"\"\"\n    return a * b".to_string()
            ),
            (
                "def is_even(n):\n    \"\"\"\n    Check if a number is even.\n    \"\"\"\n    # TODO: implement this function\n    pass".to_string(),
                "def is_even(n):\n    \"\"\"\n    Check if a number is even.\n    \"\"\"\n    return n % 2 == 0".to_string()
            ),
            (
                "def reverse_string(s):\n    \"\"\"\n    Reverse a string.\n    \"\"\"\n    # TODO: implement this function\n    pass".to_string(),
                "def reverse_string(s):\n    \"\"\"\n    Reverse a string.\n    \"\"\"\n    return s[::-1]".to_string()
            ),
            (
                "def factorial(n):\n    \"\"\"\n    Calculate factorial of n.\n    \"\"\"\n    # TODO: implement this function\n    pass".to_string(),
                "def factorial(n):\n    \"\"\"\n    Calculate factorial of n.\n    \"\"\"\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)".to_string()
            ),
        ];

        let problems: Vec<String> = (0..num_samples)
            .map(|i| {
                let (problem, _) = &problems_and_solutions[i % problems_and_solutions.len()];
                problem.clone()
            })
            .collect();

        let solutions: Vec<String> = (0..num_samples)
            .map(|i| {
                let (_, solution) = &problems_and_solutions[i % problems_and_solutions.len()];
                solution.clone()
            })
            .collect();

        Ok((problems, solutions))
    }

    fn evaluate_code(&self, generated_code: &str, reference_solution: &str) -> bool {
        // Simplified evaluation - in practice, you'd run the code and test it
        // For now, we'll do a simple check if the generated code contains key elements

        // Extract function name from reference
        let func_name = if let Some(start) = reference_solution.find("def ") {
            if let Some(end) = reference_solution[start + 4..].find('(') {
                &reference_solution[start + 4..start + 4 + end]
            } else {
                return false;
            }
        } else {
            return false;
        };

        // Check if generated code defines the function and has reasonable content
        generated_code.contains(&format!("def {}", func_name)) &&
        generated_code.contains("return") &&
        generated_code.len() > 20 && // Must be non-trivial
        !generated_code.contains("pass") && // Should not contain placeholder
        !generated_code.contains("TODO") // Should not contain TODO
    }
}

impl Evaluator for HumanEvalEvaluator {
    fn evaluate(
        &self,
        model: &dyn EvaluationModel,
        config: &EvaluationConfig,
    ) -> Result<crate::evaluation::EvaluationSuite> {
        let mut suite = crate::evaluation::EvaluationSuite::new();

        println!("Evaluating HumanEval code generation");
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
        let (problems, reference_solutions) = self.load_data(config)?;

        let mut predictions = Vec::new();
        let mut pass_count = 0;

        for (problem, reference) in problems.iter().zip(reference_solutions.iter()) {
            let generated_code = model.forward(problem)?;
            let passes = self.evaluate_code(&generated_code, reference);

            if passes {
                pass_count += 1;
            }

            predictions.push(generated_code);
        }

        let pass_at_1 = pass_count as f64 / problems.len() as f64;

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
            targets: if config.output_predictions { reference_solutions } else { Vec::new() },
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_humaneval_code_evaluation() {
        let evaluator = HumanEvalEvaluator::new();

        let reference = "def add(a, b):\n    return a + b";
        let good_code = "def add(a, b):\n    return a + b";
        let bad_code = "def add(a, b):\n    pass";

        assert!(evaluator.evaluate_code(good_code, reference));
        assert!(!evaluator.evaluate_code(bad_code, reference));
    }
}
