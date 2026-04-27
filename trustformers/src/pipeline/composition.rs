use crate::error::{Result, TrustformersError};
use crate::pipeline::{Pipeline, PipelineOutput};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "async")]
use crate::pipeline::AsyncPipeline;

/// Error handling strategy for pipeline composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandling {
    /// Stop execution on first error
    StopOnError,
    /// Continue with default values on error
    ContinueWithDefault,
    /// Skip failed steps and continue
    SkipOnError,
}

/// Strategy for pipeline composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionStrategy {
    /// Sequential execution
    Sequential,
    /// Parallel execution (where possible)
    Parallel,
    /// Conditional execution based on outputs
    Conditional,
}

/// Configuration for pipeline composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionConfig {
    /// Error handling strategy
    pub error_handling: ErrorHandling,
    /// Composition strategy
    pub strategy: CompositionStrategy,
    /// Maximum execution time (in seconds)
    pub timeout: Option<f64>,
}

impl Default for CompositionConfig {
    fn default() -> Self {
        Self {
            error_handling: ErrorHandling::StopOnError,
            strategy: CompositionStrategy::Sequential,
            timeout: None,
        }
    }
}

/// Alias for backward compatibility
pub type PipelineComposition = PipelineComposer;

/// Trait for converting between pipeline outputs and inputs
pub trait OutputConverter<T>: Send + Sync {
    fn convert(&self, output: PipelineOutput) -> Result<T>;
}

/// Default converter that attempts to extract text from pipeline outputs
pub struct TextConverter;

impl OutputConverter<String> for TextConverter {
    fn convert(&self, output: PipelineOutput) -> Result<String> {
        match output {
            PipelineOutput::Generation(gen) => Ok(gen.generated_text),
            PipelineOutput::Summarization(text) => Ok(text),
            PipelineOutput::Translation(text) => Ok(text),
            PipelineOutput::Classification(results) => {
                if let Some(first) = results.first() {
                    Ok(first.label.clone())
                } else {
                    Err(TrustformersError::invalid_input_simple(
                        "No classification results to convert".to_string(),
                    ))
                }
            },
            PipelineOutput::QuestionAnswering(qa) => Ok(qa.answer),
            PipelineOutput::FillMask(results) => {
                if let Some(first) = results.first() {
                    Ok(first.sequence.clone())
                } else {
                    Err(TrustformersError::invalid_input_simple(
                        "No fill mask results to convert".to_string(),
                    ))
                }
            },
            PipelineOutput::TokenClassification(tokens) => {
                // Concatenate all token words
                let text = tokens.iter().map(|t| &t.word).cloned().collect::<Vec<_>>().join(" ");
                Ok(text)
            },
            #[cfg(feature = "vision")]
            PipelineOutput::ImageToText(result) => Ok(result.generated_text),
            #[cfg(feature = "vision")]
            PipelineOutput::VisualQuestionAnswering(result) => Ok(result.answer),
            #[cfg(feature = "audio")]
            PipelineOutput::SpeechToText(result) => Ok(result.text),
            #[cfg(feature = "audio")]
            PipelineOutput::TextToSpeech(_result) => Err(TrustformersError::invalid_input_simple(
                "Cannot convert TextToSpeech output to text".to_string(),
            )),
            PipelineOutput::DocumentUnderstanding(result) => Ok(result.text.unwrap_or_default()),
            PipelineOutput::MultiModal(result) => Ok(result.text.unwrap_or_default()),
            #[cfg(feature = "async")]
            PipelineOutput::Conversational(result) => Ok(result.response),
            PipelineOutput::AdvancedRAG(result) => Ok(result.final_answer),
            PipelineOutput::MixtureOfDepths(result) => Ok(format!(
                "Processed with efficiency: {}",
                result.efficiency_score
            )),
            PipelineOutput::SpeculativeDecoding(result) => Ok(result.generated_text),
            PipelineOutput::Mamba2(result) => Ok(result.text),
            PipelineOutput::Text(text) => Ok(text),
        }
    }
}

/// A pipeline that composes two pipelines sequentially
pub struct ComposedPipeline<P1, P2> {
    first: Arc<P1>,
    second: Arc<P2>,
    converter: Arc<TextConverter>,
}

impl<P1, P2> ComposedPipeline<P1, P2>
where
    P1: Pipeline<Input = String, Output = PipelineOutput>,
    P2: Pipeline<Input = String, Output = PipelineOutput>,
{
    pub fn new(first: P1, second: P2) -> Self {
        Self {
            first: Arc::new(first),
            second: Arc::new(second),
            converter: Arc::new(TextConverter),
        }
    }

    /// Chain another pipeline to this composed pipeline
    pub fn chain<P3>(self, third: P3) -> ComposedPipeline<Self, P3>
    where
        P3: Pipeline<Input = String, Output = PipelineOutput>,
    {
        ComposedPipeline::new(self, third)
    }
}

impl<P1, P2> Pipeline for ComposedPipeline<P1, P2>
where
    P1: Pipeline<Input = String, Output = PipelineOutput>,
    P2: Pipeline<Input = String, Output = PipelineOutput>,
{
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        // Process with first pipeline
        let first_output = self.first.__call__(input)?;

        // Convert output to input for second pipeline
        let second_input = self.converter.convert(first_output)?;

        // Process with second pipeline
        self.second.__call__(second_input)
    }

    fn batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        // Process all inputs through first pipeline
        let first_outputs = self.first.batch(inputs)?;

        // Convert all outputs to inputs for second pipeline
        let second_inputs: Result<Vec<_>> =
            first_outputs.into_iter().map(|output| self.converter.convert(output)).collect();
        let second_inputs = second_inputs?;

        // Process through second pipeline
        self.second.batch(second_inputs)
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl<P1, P2> AsyncPipeline for ComposedPipeline<P1, P2>
where
    P1: AsyncPipeline<Input = String, Output = PipelineOutput> + Sync,
    P2: AsyncPipeline<Input = String, Output = PipelineOutput> + Sync,
{
    type Input = String;
    type Output = PipelineOutput;

    async fn __call_async__(&self, input: Self::Input) -> Result<Self::Output> {
        // Process with first pipeline
        let first_output = self.first.__call_async__(input).await?;

        // Convert output to input for second pipeline
        let second_input = self.converter.convert(first_output)?;

        // Process with second pipeline
        self.second.__call_async__(second_input).await
    }

    async fn batch_async(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        // Process all inputs through first pipeline
        let first_outputs = self.first.batch_async(inputs).await?;

        // Convert all outputs to inputs for second pipeline
        let second_inputs: Result<Vec<_>> =
            first_outputs.into_iter().map(|output| self.converter.convert(output)).collect();
        let second_inputs = second_inputs?;

        // Process through second pipeline
        self.second.batch_async(second_inputs).await
    }
}

/// A flexible pipeline chain that can handle multiple pipelines
pub struct PipelineChain {
    stages: Vec<Box<dyn Pipeline<Input = String, Output = PipelineOutput>>>,
}

impl Default for PipelineChain {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineChain {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a pipeline stage to the chain
    pub fn add_stage<P>(mut self, pipeline: P) -> Self
    where
        P: Pipeline<Input = String, Output = PipelineOutput> + 'static,
    {
        self.stages.push(Box::new(pipeline));
        self
    }

    /// Create a chain from a vector of pipelines
    pub fn from_pipelines(
        pipelines: Vec<Box<dyn Pipeline<Input = String, Output = PipelineOutput>>>,
    ) -> Self {
        Self { stages: pipelines }
    }
}

impl Pipeline for PipelineChain {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        if self.stages.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "Pipeline chain is empty".to_string(),
            ));
        }

        let mut current_input = input;
        let mut current_output = None;

        for (i, stage) in self.stages.iter().enumerate() {
            let output = stage.__call__(current_input.clone())?;

            if i == self.stages.len() - 1 {
                // Last stage, return the output
                current_output = Some(output);
            } else {
                // Convert output to string for next stage
                let converter = TextConverter;
                current_input = converter.convert(output)?;
            }
        }

        current_output.ok_or_else(|| {
            TrustformersError::invalid_input_simple("Pipeline chain produced no output".to_string())
        })
    }

    fn batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        inputs.into_iter().map(|input| self.__call__(input)).collect()
    }
}

/// Builder for creating pipeline compositions
pub struct PipelineComposer {
    stages: Vec<Box<dyn Pipeline<Input = String, Output = PipelineOutput>>>,
}

impl PipelineComposer {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Start the composition with a pipeline
    pub fn start<P>(mut self, pipeline: P) -> Self
    where
        P: Pipeline<Input = String, Output = PipelineOutput> + 'static,
    {
        self.stages.push(Box::new(pipeline));
        self
    }

    /// Add another pipeline to the composition
    pub fn then<P>(mut self, pipeline: P) -> Self
    where
        P: Pipeline<Input = String, Output = PipelineOutput> + 'static,
    {
        self.stages.push(Box::new(pipeline));
        self
    }

    /// Build the final composed pipeline; returns Err if no stages were added
    pub fn build(self) -> Result<Box<dyn Pipeline<Input = String, Output = PipelineOutput>>> {
        if self.stages.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "No pipelines added to composer".to_string(),
            ));
        }
        Ok(Box::new(PipelineChain::from_pipelines(self.stages)))
    }
}

impl Default for PipelineComposer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to create a simple two-pipeline composition
pub fn compose_pipelines<P1, P2>(first: P1, second: P2) -> ComposedPipeline<P1, P2>
where
    P1: Pipeline<Input = String, Output = PipelineOutput>,
    P2: Pipeline<Input = String, Output = PipelineOutput>,
{
    ComposedPipeline::new(first, second)
}

/// Macro for easy pipeline chaining
#[macro_export]
macro_rules! chain_pipelines {
    ($first:expr) => {
        $crate::pipeline::composition::PipelineComposer::new().start($first).build()
    };
    ($first:expr, $($rest:expr),+ $(,)?) => {
        {
            let mut composer = $crate::pipeline::composition::PipelineComposer::new().start($first);
            $(
                composer = composer.then($rest);
            )+
            composer.build()
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{GenerationOutput, PipelineOutput};

    // ── Shared mock helpers ───────────────────────────────────────────────────

    struct MockPipeline {
        name: String,
    }

    impl MockPipeline {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    impl Pipeline for MockPipeline {
        type Input = String;
        type Output = PipelineOutput;

        fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
            Ok(PipelineOutput::Generation(GenerationOutput {
                generated_text: format!("{}({})", self.name, input),
                sequences: None,
                scores: None,
            }))
        }
    }

    struct FailingPipeline;

    impl Pipeline for FailingPipeline {
        type Input = String;
        type Output = PipelineOutput;

        fn __call__(&self, _input: Self::Input) -> Result<Self::Output> {
            Err(TrustformersError::invalid_input_simple(
                "simulated failure".to_string(),
            ))
        }
    }

    // ── ComposedPipeline (A → B) ──────────────────────────────────────────────

    #[test]
    fn test_composed_pipeline() {
        let composed =
            ComposedPipeline::new(MockPipeline::new("first"), MockPipeline::new("second"));
        let result = composed
            .__call__("input".to_string())
            .expect("composed pipeline should succeed");
        if let PipelineOutput::Generation(gen) = result {
            assert_eq!(gen.generated_text, "second(first(input))");
        } else {
            panic!("Expected generation output");
        }
    }

    #[test]
    fn test_composed_pipeline_error_propagates() {
        let composed = ComposedPipeline::new(FailingPipeline, MockPipeline::new("second"));
        let result = composed.__call__("input".to_string());
        assert!(result.is_err(), "error in first stage should propagate");
    }

    #[test]
    fn test_composed_pipeline_second_stage_error() {
        let composed = ComposedPipeline::new(MockPipeline::new("first"), FailingPipeline);
        let result = composed.__call__("input".to_string());
        assert!(result.is_err(), "error in second stage should propagate");
    }

    // ── Pipeline chaining (A → B → C) ────────────────────────────────────────

    #[test]
    fn test_composed_pipeline_chain_three() {
        let ab = ComposedPipeline::new(MockPipeline::new("A"), MockPipeline::new("B"));
        let abc = ab.chain(MockPipeline::new("C"));
        let result = abc.__call__("x".to_string()).expect("3-stage chain should succeed");
        if let PipelineOutput::Generation(gen) = result {
            assert!(
                gen.generated_text.contains("C"),
                "final stage name should appear in output"
            );
            assert!(
                gen.generated_text.contains("x"),
                "original input should appear in output"
            );
        } else {
            panic!("Expected generation output");
        }
    }

    // ── PipelineChain (vec of stages) ─────────────────────────────────────────

    #[test]
    fn test_pipeline_chain() {
        let chain = PipelineChain::new()
            .add_stage(MockPipeline::new("stage1"))
            .add_stage(MockPipeline::new("stage2"))
            .add_stage(MockPipeline::new("stage3"));
        let result = chain.__call__("input".to_string()).expect("operation failed in test");
        if let PipelineOutput::Generation(gen) = result {
            assert_eq!(gen.generated_text, "stage3(stage2(stage1(input)))");
        } else {
            panic!("Expected generation output");
        }
    }

    #[test]
    fn test_pipeline_chain_empty_errors() {
        let chain = PipelineChain::new();
        let result = chain.__call__("input".to_string());
        assert!(result.is_err(), "empty chain should return an error");
    }

    #[test]
    fn test_pipeline_chain_single_stage() {
        let chain = PipelineChain::new().add_stage(MockPipeline::new("only"));
        let result = chain.__call__("val".to_string()).expect("single-stage chain should succeed");
        if let PipelineOutput::Generation(gen) = result {
            assert_eq!(gen.generated_text, "only(val)");
        } else {
            panic!("Expected generation output");
        }
    }

    #[test]
    fn test_pipeline_chain_error_propagates() {
        let chain = PipelineChain::new()
            .add_stage(MockPipeline::new("s1"))
            .add_stage(FailingPipeline);
        let result = chain.__call__("input".to_string());
        assert!(result.is_err(), "error in chain stage should propagate");
    }

    #[test]
    fn test_pipeline_chain_from_pipelines() {
        let stages: Vec<Box<dyn Pipeline<Input = String, Output = PipelineOutput>>> = vec![
            Box::new(MockPipeline::new("a")),
            Box::new(MockPipeline::new("b")),
        ];
        let chain = PipelineChain::from_pipelines(stages);
        let result = chain.__call__("x".to_string()).expect("from_pipelines chain should succeed");
        if let PipelineOutput::Generation(gen) = result {
            assert!(gen.generated_text.contains("b"));
        } else {
            panic!("Expected generation output");
        }
    }

    // ── PipelineChain batch ───────────────────────────────────────────────────

    #[test]
    fn test_pipeline_chain_batch() {
        let chain = PipelineChain::new()
            .add_stage(MockPipeline::new("p1"))
            .add_stage(MockPipeline::new("p2"));
        let results = chain
            .batch(vec!["a".to_string(), "b".to_string()])
            .expect("batch should succeed");
        assert_eq!(results.len(), 2);
    }

    // ── PipelineComposer ──────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_composer() {
        let composed = PipelineComposer::new()
            .start(MockPipeline::new("first"))
            .then(MockPipeline::new("second"))
            .then(MockPipeline::new("third"))
            .build()
            .expect("operation failed in test");
        let result = composed.__call__("input".to_string()).expect("operation failed in test");
        if let PipelineOutput::Generation(gen) = result {
            // All three stages must participate: third(second(first(input)))
            assert_eq!(gen.generated_text, "third(second(first(input)))");
        } else {
            panic!("Expected generation output");
        }
    }

    #[test]
    fn test_pipeline_composer_three_stages() {
        let composed = PipelineComposer::new()
            .start(MockPipeline::new("A"))
            .then(MockPipeline::new("B"))
            .then(MockPipeline::new("C"))
            .build()
            .expect("three-stage composer should build");
        let result = composed.__call__("x".to_string()).expect("three-stage call should succeed");
        if let PipelineOutput::Generation(gen) = result {
            assert_eq!(
                gen.generated_text, "C(B(A(x)))",
                "all three stages must run in order"
            );
        } else {
            panic!("Expected generation output");
        }
    }

    #[test]
    fn test_pipeline_composer_empty_build() {
        let result = PipelineComposer::new().build();
        assert!(result.is_err(), "build() with zero stages must return Err");
    }

    #[test]
    fn test_pipeline_composer_empty_build_errors() {
        let result = PipelineComposer::new().build();
        assert!(result.is_err(), "building an empty composer should fail");
    }

    #[test]
    fn test_pipeline_composer_single_pipeline() {
        let composed = PipelineComposer::new()
            .start(MockPipeline::new("solo"))
            .build()
            .expect("single-pipeline composer should succeed");
        let result = composed.__call__("in".to_string()).expect("solo pipeline should succeed");
        if let PipelineOutput::Generation(gen) = result {
            assert!(gen.generated_text.contains("solo"));
        } else {
            panic!("Expected generation output");
        }
    }

    // ── TextConverter ────────────────────────────────────────────────────────

    #[test]
    fn test_text_converter_generation() {
        let converter = TextConverter;
        let output = PipelineOutput::Generation(GenerationOutput {
            generated_text: "hello world".to_string(),
            sequences: None,
            scores: None,
        });
        let text = converter.convert(output).expect("TextConverter should convert generation");
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_text_converter_summarization() {
        let converter = TextConverter;
        let output = PipelineOutput::Summarization("summary text".to_string());
        let text = converter.convert(output).expect("TextConverter should convert summarization");
        assert_eq!(text, "summary text");
    }

    #[test]
    fn test_text_converter_text_variant() {
        let converter = TextConverter;
        let output = PipelineOutput::Text("raw text".to_string());
        let text = converter.convert(output).expect("TextConverter should convert Text");
        assert_eq!(text, "raw text");
    }

    // ── Pipeline composition function ─────────────────────────────────────────

    #[test]
    fn test_compose_pipelines_function() {
        let composed = compose_pipelines(MockPipeline::new("first"), MockPipeline::new("second"));
        let result =
            composed.__call__("test".to_string()).expect("compose_pipelines should succeed");
        if let PipelineOutput::Generation(gen) = result {
            assert_eq!(gen.generated_text, "second(first(test))");
        } else {
            panic!("Expected generation output");
        }
    }

    // ── Intermediate output capture in chain ─────────────────────────────────

    #[test]
    fn test_intermediate_output_propagates_through_chain() {
        // A → B → C: verify B transforms A's output before C sees it
        let chain = PipelineChain::new()
            .add_stage(MockPipeline::new("A"))
            .add_stage(MockPipeline::new("B"))
            .add_stage(MockPipeline::new("C"));
        let result = chain.__call__("x".to_string()).expect("chain should succeed");
        if let PipelineOutput::Generation(gen) = result {
            // C sees B(A(x)) as its input
            assert_eq!(gen.generated_text, "C(B(A(x)))");
        } else {
            panic!("Expected generation output");
        }
    }

    // ── Macro test ────────────────────────────────────────────────────────────

    #[test]
    fn test_chain_pipelines_macro() {
        let result = chain_pipelines!(
            MockPipeline::new("p1"),
            MockPipeline::new("p2"),
            MockPipeline::new("p3")
        )
        .expect("operation failed in test");
        let output = result.__call__("test".to_string()).expect("operation failed in test");
        if let PipelineOutput::Generation(gen) = output {
            assert!(gen.generated_text.contains("test"));
        } else {
            panic!("Expected generation output");
        }
    }

    // ── Parallel fork/join conceptual test ───────────────────────────────────

    #[test]
    fn test_two_pipelines_independent_execution() {
        // Verify A and B can each be run independently (simulates parallel fork)
        let a = MockPipeline::new("fork_a");
        let b = MockPipeline::new("fork_b");
        let res_a = a.__call__("x".to_string()).expect("fork_a should succeed");
        let res_b = b.__call__("x".to_string()).expect("fork_b should succeed");
        if let (PipelineOutput::Generation(g_a), PipelineOutput::Generation(g_b)) = (res_a, res_b) {
            assert_ne!(
                g_a.generated_text, g_b.generated_text,
                "forked pipelines should produce different outputs"
            );
        } else {
            panic!("Expected generation outputs from both forks");
        }
    }

    // ── Pipeline metadata accumulation ────────────────────────────────────────

    #[test]
    fn test_pipeline_chain_accumulates_all_stage_names() {
        // The chain output should reflect all stage transformations
        let chain = PipelineChain::new()
            .add_stage(MockPipeline::new("alpha"))
            .add_stage(MockPipeline::new("beta"))
            .add_stage(MockPipeline::new("gamma"));
        let result = chain.__call__("seed".to_string()).expect("chain should succeed");
        if let PipelineOutput::Generation(gen) = result {
            assert!(
                gen.generated_text.contains("gamma"),
                "last stage must appear in output"
            );
            assert!(
                gen.generated_text.contains("seed"),
                "original input must appear in output"
            );
        }
    }
}
