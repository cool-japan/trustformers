//! TrustformeRS Tokenizers - Tokenization library for transformer models

// Allow large error types in Result (TrustformersError is large by design)
#![allow(clippy::result_large_err)]
// Allow common patterns in tokenization code
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::excessive_nesting)]

pub mod advanced_vocab_intelligence;
pub mod alignment;
pub mod arabic;
pub mod async_tokenizer;
pub mod benchmark_utils;
pub mod binary_format;
pub mod bio;
pub mod bpe;
pub mod canine;
pub mod char;
pub mod chemical;
pub mod chinese;
pub mod code_tokenizer;
pub mod compressed_vocab;
pub mod coverage;
pub mod custom;
pub mod custom_format;
pub mod fairseq;
#[cfg(feature = "gpu")]
pub mod gpu_tokenization;
pub mod japanese;
#[cfg(feature = "jax")]
pub mod jax;
pub mod korean;
pub mod math_tokenizer;
pub mod messagepack_serialization;
pub mod minimal_perfect_hash;
pub mod mmap_vocab;
pub mod multimodal;
pub mod music;
pub mod normalizer;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod parallel;
pub mod performance_profiler;
pub mod protobuf_serialization;
#[cfg(feature = "pytorch")]
pub mod pytorch;
pub mod regex_tokenizer;
pub mod sentencepiece;
pub mod sequence_packing;
pub mod shared_vocab_pool;
pub mod simd;
pub mod special_tokens;
pub mod streaming;
pub mod subword_regularization;
#[cfg(feature = "tensorflow")]
pub mod tensorflow;
pub mod test_infrastructure;
pub mod thai;
pub mod tiktoken;
pub mod tokenization_debugger;
pub mod tokenizer;
pub mod training;
pub mod unigram;
pub mod visualization;
pub mod vocab;
pub mod vocab_analyzer;
pub mod wordpiece;
pub mod zero_copy;

#[cfg(feature = "python")]
pub mod python;

pub use advanced_vocab_intelligence::{
    ActionableRecommendation, CompressionAnalysis, CompressionOpportunity,
    CompressionOpportunityType, CrossLingualAnalysis, DeclineToken, DomainAnalysis,
    DomainDistribution, EvolutionAnalysis, EvolutionPrediction, ImplementationDifficulty,
    LanguageCoverage, MergeRisk, MultilingualOpportunity, RecommendationCategory,
    RecommendationPriority, RedundantTokenGroup, RiskAssessment, RiskLevel, SemanticAnalysis,
    SemanticCluster, TrendingToken, VocabIntelligenceAnalyzer, VocabIntelligenceConfig,
    VocabIntelligenceResult,
};
pub use alignment::{
    AlignedSpan, AlignmentConfig, AlignmentEngine, AlignmentStats, TokenAlignment, Word,
};
pub use arabic::{
    ArabicMode, ArabicTokenizer, ArabicTokenizerConfig, MorphologicalAnalysis, TokenizationStats,
};
pub use async_tokenizer::{
    AsyncTokenizer, AsyncTokenizerConfig, AsyncTokenizerWrapper, ConfigurableAsyncTokenizer,
};
pub use benchmark_utils::{
    BenchmarkConfig, BenchmarkResult as TokenizerBenchmarkResult, TokenizerBenchmark,
};
pub use binary_format::{
    BinaryConfig, BinaryHeader, BinarySerializer, BinaryTokenizer, BinaryUtils,
    NormalizationRule as BinaryNormalizationRule, PreTokenizationRule as BinaryPreTokenizationRule,
    TokenizerConverter,
};
pub use bio::{
    BioAnalysis, BioToken, BioTokenMetadata, BioTokenType, BioTokenizer, BioTokenizerConfig,
};
pub use bpe::BPETokenizer;
pub use canine::CanineTokenizer;
pub use char::CharTokenizer;
pub use chemical::{
    ChemicalAnalysis, ChemicalToken, ChemicalTokenMetadata, ChemicalTokenType, ChemicalTokenizer,
    ChemicalTokenizerConfig,
};
pub use chinese::{ChineseTokenizer, ChineseTokenizerConfig};
pub use code_tokenizer::{
    CodeToken, CodeTokenType, CodeTokenizer, CodeTokenizerConfig, CommentPatterns, Language,
    LiteralType, TokenPosition,
};
pub use compressed_vocab::{CompressedVocab, CompressedVocabStats, PrefixTrie};
pub use coverage::{
    CharacterCoverage, CoverageAnalyzer, CoverageConfig, CoverageExample, CoverageReport,
    CoverageReportExporter, CoverageThresholds, CoverageWarning, PerformanceMetrics,
    QualityMetrics, ReportFormat, TokenDistribution, VocabularyCoverage,
};
pub use custom::{CustomVocabTokenizer, CustomVocabTokenizerBuilder};
pub use custom_format::{
    CustomFormatConverter, CustomFormatTokenizer, CustomSpecialToken, CustomToken,
    CustomTokenizerFormat, CustomVocabulary, NormalizationRule as CustomNormalizationRule,
    NormalizationType, PostProcessingRule, PostProcessingType,
    PreTokenizationRule as CustomPreTokenizationRule, PreTokenizationType, SpecialTokenType,
    VocabularyType,
};
pub use fairseq::{FairseqDictionaryBuilder, FairseqTokenizer};
#[cfg(feature = "gpu")]
pub use gpu_tokenization::{
    BatchProcessingConfig, BenchmarkResult as GpuBenchmarkResult, GpuTokenizationBenchmark,
    GpuTokenizationResult, GpuTokenizationStats, GpuTokenizer, GpuTokenizerConfig,
    GpuTokenizerError, KernelOptimization, MemoryOptimization,
    PaddingStrategy as GpuPaddingStrategy,
};
pub use japanese::{JapaneseMode, JapaneseTokenizer, JapaneseTokenizerConfig};
#[cfg(feature = "jax")]
pub use jax::{
    JaxArray, JaxBatch, JaxCompiledTokenizer, JaxConfig, JaxDType, JaxDataIterator, JaxDataset,
    JaxDevice, JaxMesh, JaxPaddingStrategy, JaxSharding, JaxTokenizer, JaxTruncationStrategy,
    JaxUtils,
};
pub use korean::{KoreanMode, KoreanTokenizer, KoreanTokenizerConfig};
pub use math_tokenizer::{
    MathAnalysis, MathToken, MathTokenType, MathTokenizer, MathTokenizerConfig,
};
pub use messagepack_serialization::{
    MessagePackConfig, MessagePackMergeRule, MessagePackNormalizationRule, MessagePackSerializer,
    MessagePackTokenizedInput, MessagePackTokenizerConfig, MessagePackTokenizerMetadata,
    MessagePackUtils, MessagePackVocabEntry,
};
pub use minimal_perfect_hash::{
    EfficiencyComparison, MemoryUsage, MinimalPerfectHash, MinimalPerfectHashConfig,
    MinimalPerfectHashVocab,
};
pub use mmap_vocab::{MemoryStats, MmapVocab, TokenIterator};
pub use multimodal::{
    AudioFrame, FusionStrategy, GraphData, ImagePatch, ModalityType, MultimodalConfig,
    MultimodalInput, MultimodalToken, MultimodalTokenMetadata, MultimodalTokenizedInput,
    MultimodalTokenizer, MultimodalUtils, TableData, VideoFrame,
};
pub use music::{
    MusicAnalysis, MusicToken, MusicTokenMetadata, MusicTokenType, MusicTokenizer,
    MusicTokenizerConfig,
};
#[cfg(feature = "onnx")]
pub use onnx::{
    OnnxAttribute, OnnxDataType, OnnxExportConfig, OnnxModel, OnnxModelMetadata, OnnxNode,
    OnnxOptimizationLevel, OnnxSessionOptions, OnnxTensorData, OnnxTensorInfo,
    OnnxTokenizerExporter, OnnxTokenizerRuntime, OnnxUtils,
};
pub use parallel::{BatchTokenizer, BatchedTokenizedInput, ParallelTokenizer};
pub use performance_profiler::{
    BenchmarkResult, ExportFormat, MemoryStats as ProfilerMemoryStats, PerformanceProfiler,
    ProfilerConfig, ProfilingReport, ProfilingSummary, ThroughputStats, TimingStats,
    TokenizerComparison as ProfilerComparison,
};
pub use protobuf_serialization::{
    ProtobufConvertible, ProtobufExportConfig, ProtobufExporter, ProtobufFormat, ProtobufMergeRule,
    ProtobufNormalizationRule, ProtobufSerializer, ProtobufTokenizedInput,
    ProtobufTokenizerMetadata, ProtobufTokenizerModel, ProtobufVocabEntry,
};
#[cfg(feature = "pytorch")]
pub use pytorch::{
    BatchIterator, PaddingStrategy as PyTorchPaddingStrategy, PyTorchBatch, PyTorchConfig,
    PyTorchDataset, PyTorchTensor, PyTorchTokenizer, PyTorchUtils, TensorDType, TruncationStrategy,
};
pub use sentencepiece::SentencePieceTokenizer;
pub use sequence_packing::{
    AdvancedSequencePacker, PackedSequence, PackingConfig, PackingInfo, PackingStats,
    PackingStrategy, SequencePacker,
};
pub use shared_vocab_pool::{PooledVocab, SharedVocabPool, VocabPoolConfig, VocabPoolStats};
pub use simd::SimdTokenizer;
pub use special_tokens::{
    AdvancedTemplate, ConversationMessage, PlaceholderProcessor, PlaceholderToken, PlaceholderType,
    SpecialTokenConfig, SpecialTokenManager,
};
pub use streaming::{BatchedStreamingTokenizer, StreamingTokenizer, TextFileIterator};
pub use subword_regularization::{
    SubwordRegularizationConfig, SubwordRegularizer, UnigramSubwordRegularizer,
};
#[cfg(feature = "tensorflow")]
pub use tensorflow::{
    RaggedTensor, TensorFlowBatch, TensorFlowConfig, TensorFlowDataset, TensorFlowTensor,
    TensorFlowTokenizer, TensorFlowUtils, TensorOrRagged, TfDType, TfDataIterator,
    TfPaddingStrategy, TfTruncationStrategy,
};
pub use test_infrastructure::{
    BenchmarkResults, CrossValidationResults, CrossValidationRunner, FuzzingResults,
    InconsistencySeverity, RegressionResults, TestCaseGenerator, TestConfig, TestReportUtils,
    TestResult, TestRunner, TestSuiteResult,
};
pub use thai::{ThaiMode, ThaiTokenizer, ThaiTokenizerConfig};
pub use tiktoken::TiktokenTokenizer;
pub use tokenization_debugger::{
    CharacterAnalysis, CompressionStats, DebugAnalysis, DebugSession, DebuggerConfig,
    DetectedIssue, IssueSeverity, IssueType, PatternAnalysis, PerformanceStats,
    TokenizationDebugger, TokenizationResult,
};
pub use tokenizer::{
    TokenizedInputWithAlignment, TokenizedInputWithOffsets, TokenizerImpl, TokenizerWrapper,
};
pub use training::{
    CoverageAnalysis, DistributedTrainingCoordinator, LanguageDetectionResult, LanguageDetector,
    StreamingTrainer, TokenDistributionAnalyzer, TokenDistributionResult, TrainingCheckpoint,
};
pub use unigram::UnigramTokenizer;
pub use visualization::{
    ComparisonStats, TokenInfo, TokenVisualization, TokenVisualizer,
    TokenizationStats as VisualizationTokenizationStats, TokenizerComparison, VisualizationConfig,
};
pub use vocab::{FlexibleVocab, LazyVocab, MergeStrategy, Vocab};
pub use vocab_analyzer::{
    CharacterPattern, CoverageAnalysis as VocabCoverageAnalysis, FrequencyAnalysis,
    IssueSeverity as VocabIssueSeverity, LanguageDistribution, SubwordPattern, VocabAnalysisConfig,
    VocabAnalysisResult, VocabAnalyzer, VocabBasicStats, VocabDebugUtils, VocabIssue,
    VocabIssueType,
};
pub use wordpiece::WordPieceTokenizer;
pub use zero_copy::{
    ZeroCopyBuilder, ZeroCopyHeader, ZeroCopyMemoryStats, ZeroCopyTokenizer, ZeroCopyUtils,
    ZeroCopyVocabEntry,
};

use trustformers_core::traits::{TokenizedInput, Tokenizer};

// Python module export

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
