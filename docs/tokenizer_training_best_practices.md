# TrustformeRS Tokenizer Training Best Practices

## Overview

Training high-quality tokenizers is crucial for optimal model performance. This guide provides comprehensive best practices for training tokenizers using TrustformeRS, covering data preparation, algorithm selection, optimization strategies, and evaluation methods.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Algorithm Selection](#algorithm-selection)
3. [Training Configuration](#training-configuration)
4. [Vocabulary Optimization](#vocabulary-optimization)
5. [Quality Assessment](#quality-assessment)
6. [Performance Optimization](#performance-optimization)
7. [Domain-Specific Training](#domain-specific-training)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Techniques](#advanced-techniques)

## Data Preparation

### Corpus Collection

#### Size Guidelines
```rust
// Recommended corpus sizes for different tokenizer types
let corpus_size_recommendations = vec![
    ("BPE", "10M-1B tokens"),
    ("WordPiece", "50M-500M tokens"),
    ("SentencePiece", "100M-1B tokens"),
    ("Unigram", "100M-1B tokens"),
];
```

#### Quality Standards
- **Cleanliness**: Remove corrupted, duplicate, or irrelevant text
- **Diversity**: Include varied genres, styles, and domains
- **Balance**: Ensure representative coverage of target use cases
- **Recency**: Include up-to-date content for current language patterns

#### Data Preprocessing

```rust
use trustformers_tokenizers::training::{CorpusProcessor, PreprocessingConfig};

let preprocessing_config = PreprocessingConfig::new()
    .with_deduplication(true)
    .with_language_detection(true)
    .with_length_filtering(10, 10000)
    .with_quality_filtering(true)
    .with_encoding_normalization(true);

let processor = CorpusProcessor::new(preprocessing_config);
let clean_corpus = processor.process_corpus(&raw_corpus)?;
```

### Text Normalization

#### Unicode Normalization
```rust
use trustformers_tokenizers::training::TextNormalizer;

let normalizer = TextNormalizer::new()
    .with_unicode_form(UnicodeForm::NFC)  // Canonical composition
    .with_case_handling(CaseHandling::Preserve)
    .with_accent_handling(AccentHandling::Preserve)
    .with_whitespace_normalization(true);
```

#### Language-Specific Normalization
```rust
// For multilingual corpora
let multilingual_normalizer = TextNormalizer::new()
    .with_language_detection(true)
    .with_per_language_rules(vec![
        ("ja", JapaneseNormalization::new().with_katakana_normalization(true)),
        ("ar", ArabicNormalization::new().with_diacritic_handling(true)),
        ("zh", ChineseNormalization::new().with_traditional_simplified_mapping(true)),
    ]);
```

## Algorithm Selection

### BPE Training

#### Basic Configuration
```rust
use trustformers_tokenizers::training::BPETrainer;

let bpe_trainer = BPETrainer::new()
    .with_vocab_size(32000)
    .with_min_frequency(2)
    .with_special_tokens(vec!["<pad>", "<unk>", "<s>", "</s>"])
    .with_end_of_word_suffix("</w>")
    .with_dropout(Some(0.1));  // Subword regularization
```

#### Advanced BPE Configuration
```rust
let advanced_bpe = BPETrainer::new()
    .with_vocab_size(50000)
    .with_character_coverage(0.9995)
    .with_byte_fallback(true)  // Handle unknown bytes
    .with_regex_splitting(r"\w+|[^\w\s]")  // Better word boundaries
    .with_continuing_subword_prefix("##")
    .with_show_progress(true);
```

### WordPiece Training

#### Standard Configuration
```rust
use trustformers_tokenizers::training::WordPieceTrainer;

let wordpiece_trainer = WordPieceTrainer::new()
    .with_vocab_size(30000)
    .with_min_frequency(2)
    .with_limit_alphabet(1000)  // Character vocabulary limit
    .with_continuing_subword_prefix("##")
    .with_unk_token("[UNK]");
```

#### BERT-Style Configuration
```rust
let bert_wordpiece = WordPieceTrainer::new()
    .with_vocab_size(30522)  // BERT vocabulary size
    .with_special_tokens(vec![
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
    ])
    .with_lowercase(true)
    .with_strip_accents(true)
    .with_clean_text(true);
```

### SentencePiece Training

#### Multilingual Configuration
```rust
use trustformers_tokenizers::training::SentencePieceTrainer;

let sp_trainer = SentencePieceTrainer::new()
    .with_vocab_size(32000)
    .with_character_coverage(0.9995)
    .with_model_type(SentencePieceModel::BPE)
    .with_split_by_whitespace(false)  // Language-agnostic
    .with_split_by_number(false)
    .with_split_by_unicode_script(true)
    .with_split_digits(true);
```

#### Unigram Configuration
```rust
let unigram_trainer = SentencePieceTrainer::new()
    .with_vocab_size(32000)
    .with_model_type(SentencePieceModel::Unigram)
    .with_seed_sentencepiece_size(1000000)
    .with_shrinking_factor(0.75)
    .with_num_threads(8)
    .with_max_sentence_length(4192);
```

### Unigram Training

#### High-Quality Configuration
```rust
use trustformers_tokenizers::training::UnigramTrainer;

let unigram_trainer = UnigramTrainer::new()
    .with_vocab_size(32000)
    .with_n_sub_iterations(2)
    .with_shrinking_factor(0.75)
    .with_max_piece_length(16)
    .with_seed_size(1000000)
    .with_alpha(0.1);  // Smoothing parameter
```

## Training Configuration

### Memory Management

#### Large-Scale Training
```rust
use trustformers_tokenizers::training::{TrainingConfig, MemoryConfig};

let memory_config = MemoryConfig::new()
    .with_chunk_size(1_000_000)  // Process 1M tokens at a time
    .with_max_memory_usage(8_000_000_000)  // 8GB limit
    .with_streaming_mode(true)
    .with_temp_directory("/tmp/tokenizer_training");

let training_config = TrainingConfig::new()
    .with_memory_config(memory_config)
    .with_progress_reporting(true)
    .with_checkpoint_interval(100_000);
```

#### Distributed Training
```rust
use trustformers_tokenizers::training::DistributedTrainingCoordinator;

let distributed_config = DistributedTrainingCoordinator::new()
    .with_num_workers(4)
    .with_corpus_partitioning(PartitioningStrategy::Round Robin)
    .with_vocabulary_merging(MergingStrategy::Union)
    .with_synchronization_interval(1000);
```

### Parallel Processing

```rust
let parallel_trainer = BPETrainer::new()
    .with_num_threads(num_cpus::get())
    .with_parallel_processing(true)
    .with_batch_size(10000)
    .with_prefetch_factor(2);
```

## Vocabulary Optimization

### Size Selection

#### Automatic Optimization
```rust
use trustformers_tokenizers::training::{VocabSizeOptimizer, OptimizationTarget};

let optimizer = VocabSizeOptimizer::new()
    .with_target_coverage(0.95)
    .with_optimization_target(OptimizationTarget::Balanced)
    .with_size_range(8000..100000)
    .with_step_size(1000);

let optimal_size = optimizer.find_optimal_size(&corpus)?;
println!("Optimal vocabulary size: {}", optimal_size);
```

#### Coverage-Based Selection
```rust
use trustformers_tokenizers::training::CoverageAnalyzer;

let analyzer = CoverageAnalyzer::new();
let mut best_size = 30000;
let mut best_coverage = 0.0;

for size in (10000..=50000).step_by(5000) {
    let tokenizer = BPETrainer::new()
        .with_vocab_size(size)
        .train(&corpus)?;
    
    let coverage = analyzer.calculate_coverage(&tokenizer, &validation_corpus)?;
    if coverage.character_coverage > best_coverage {
        best_coverage = coverage.character_coverage;
        best_size = size;
    }
}
```

### Special Token Management

#### Task-Specific Tokens
```rust
// For different tasks
let classification_tokens = vec!["[CLS]", "[SEP]", "[PAD]", "[MASK]"];
let generation_tokens = vec!["<bos>", "<eos>", "<pad>", "<unk>"];
let conversation_tokens = vec!["<user>", "<assistant>", "<system>", "<end_turn>"];

let trainer = BPETrainer::new()
    .with_special_tokens(classification_tokens)
    .with_special_token_frequency(1000000);  // High frequency for special tokens
```

#### Domain-Specific Tokens
```rust
// For code
let code_tokens = vec!["<CODE>", "<COMMENT>", "<STRING>", "<NUMBER>", "<INDENT>", "<DEDENT>"];

// For math
let math_tokens = vec!["<EQUATION>", "<FORMULA>", "<VARIABLE>", "<CONSTANT>"];

// For chemistry
let chem_tokens = vec!["<MOLECULE>", "<REACTION>", "<BOND>", "<CHARGE>"];
```

## Quality Assessment

### Coverage Analysis

```rust
use trustformers_tokenizers::training::{QualityMetrics, CoverageReport};

let quality_metrics = QualityMetrics::new();
let report = quality_metrics.comprehensive_analysis(
    &tokenizer,
    &validation_corpus,
    &test_corpus
)?;

println!("Character coverage: {:.2}%", report.character_coverage * 100.0);
println!("Word coverage: {:.2}%", report.word_coverage * 100.0);
println!("OOV rate: {:.2}%", report.oov_rate * 100.0);
println!("Compression ratio: {:.2}", report.compression_ratio);
```

### Distribution Analysis

```rust
use trustformers_tokenizers::training::TokenDistributionAnalyzer;

let distribution_analyzer = TokenDistributionAnalyzer::new();
let analysis = distribution_analyzer.analyze(&tokenizer, &corpus)?;

// Check for healthy distribution
assert!(analysis.zipf_coefficient > 0.8, "Token distribution not Zipfian");
assert!(analysis.vocab_utilization > 0.9, "Low vocabulary utilization");
assert!(analysis.avg_token_length > 2.0, "Tokens too short");
assert!(analysis.avg_token_length < 8.0, "Tokens too long");
```

### Language Detection

```rust
use trustformers_tokenizers::training::LanguageDetector;

let detector = LanguageDetector::new();
let language_dist = detector.analyze_corpus_languages(&corpus)?;

for (lang, proportion) in language_dist {
    println!("Language: {}, Proportion: {:.2}%", lang, proportion * 100.0);
}
```

## Performance Optimization

### Training Speed

#### CPU Optimization
```rust
let fast_trainer = BPETrainer::new()
    .with_simd_optimization(true)
    .with_parallel_processing(true)
    .with_memory_mapping(true)
    .with_batch_processing(true)
    .with_cache_frequency_counts(true);
```

#### Memory Efficiency
```rust
let memory_efficient = BPETrainer::new()
    .with_streaming_training(true)
    .with_checkpoint_frequency(10000)
    .with_compression(CompressionType::LZ4)
    .with_vocabulary_pruning(true);
```

### Incremental Training

```rust
use trustformers_tokenizers::training::IncrementalTrainer;

let incremental_trainer = IncrementalTrainer::new(base_tokenizer)
    .with_adaptation_rate(0.1)
    .with_vocabulary_expansion(true)
    .with_frequency_decay(0.99);

// Add new domain data
let updated_tokenizer = incremental_trainer.update(&new_domain_corpus)?;
```

## Domain-Specific Training

### Code Tokenization

```rust
use trustformers_tokenizers::training::CodeTokenizerTrainer;

let code_trainer = CodeTokenizerTrainer::new()
    .with_languages(vec!["rust", "python", "javascript", "cpp"])
    .with_keyword_preservation(true)
    .with_identifier_tokenization(IdentifierTokenization::CamelCase)
    .with_string_literal_handling(true)
    .with_comment_handling(CommentHandling::Separate);

let code_tokenizer = code_trainer.train(&code_corpus)?;
```

### Mathematical Text

```rust
use trustformers_tokenizers::training::MathTokenizerTrainer;

let math_trainer = MathTokenizerTrainer::new()
    .with_latex_support(true)
    .with_unicode_math(true)
    .with_equation_detection(true)
    .with_symbol_preservation(true)
    .with_number_tokenization(NumberTokenization::Scientific);

let math_tokenizer = math_trainer.train(&math_corpus)?;
```

### Multilingual Training

```rust
use trustformers_tokenizers::training::MultilingualTrainer;

let multilingual_trainer = MultilingualTrainer::new()
    .with_languages(vec!["en", "zh", "ar", "hi", "es", "fr"])
    .with_script_mixing(ScriptMixing::Balanced)
    .with_cross_lingual_consistency(true)
    .with_language_identification(true);

let multilingual_tokenizer = multilingual_trainer.train(&multilingual_corpus)?;
```

## Troubleshooting

### Common Issues

#### Low Coverage
```rust
// Diagnosis
let analyzer = CoverageAnalyzer::new();
let coverage = analyzer.analyze_coverage(&tokenizer, &test_corpus)?;

if coverage.character_coverage < 0.95 {
    // Solutions:
    // 1. Increase vocabulary size
    // 2. Improve corpus quality
    // 3. Adjust character coverage parameter
    // 4. Use more diverse training data
}
```

#### Poor Compression
```rust
if coverage.compression_ratio < 3.0 {
    // Solutions:
    // 1. Decrease minimum frequency
    // 2. Increase vocabulary size
    // 3. Use BPE instead of WordPiece
    // 4. Improve subword boundaries
}
```

#### Slow Training
```rust
// Performance optimization
let optimized_trainer = BPETrainer::new()
    .with_num_threads(num_cpus::get())
    .with_batch_size(50000)  // Larger batches
    .with_memory_mapping(true)
    .with_streaming_mode(true);  // For very large corpora
```

### Memory Issues

```rust
// For large corpora
let memory_safe_trainer = BPETrainer::new()
    .with_chunk_size(100000)  // Smaller chunks
    .with_max_memory_mb(4000)  // Memory limit
    .with_streaming_mode(true)
    .with_temp_directory("/tmp/large_training");
```

## Advanced Techniques

### Curriculum Learning

```rust
use trustformers_tokenizers::training::CurriculumTrainer;

let curriculum_trainer = CurriculumTrainer::new()
    .with_difficulty_metric(DifficultyMetric::Perplexity)
    .with_pacing_function(PacingFunction::Linear)
    .with_stages(vec![
        TrainingStage::new("easy", 0.3),    // 30% easy data
        TrainingStage::new("medium", 0.5),  // 50% medium data
        TrainingStage::new("hard", 0.2),    // 20% hard data
    ]);
```

### Multi-Corpus Training

```rust
use trustformers_tokenizers::training::MultiCorpusTrainer;

let multi_trainer = MultiCorpusTrainer::new()
    .add_corpus("general", general_corpus, 0.6)      // 60% weight
    .add_corpus("domain", domain_corpus, 0.3)        // 30% weight
    .add_corpus("recent", recent_corpus, 0.1)        // 10% weight
    .with_balancing_strategy(BalancingStrategy::Weighted);
```

### Adaptive Training

```rust
use trustformers_tokenizers::training::AdaptiveTrainer;

let adaptive_trainer = AdaptiveTrainer::new()
    .with_performance_monitoring(true)
    .with_automatic_adjustment(true)
    .with_convergence_detection(true)
    .with_early_stopping(EarlyStoppingConfig::new()
        .with_patience(10)
        .with_min_delta(0.001)
    );
```

### Subword Regularization

```rust
use trustformers_tokenizers::training::SubwordRegularization;

let regularized_trainer = BPETrainer::new()
    .with_subword_regularization(SubwordRegularization::new()
        .with_alpha(0.1)
        .with_sampling_strategy(SamplingStrategy::Unigram)
        .with_enable_sampling(true)
    );
```

## Validation and Testing

### Cross-Validation

```rust
use trustformers_tokenizers::training::CrossValidator;

let validator = CrossValidator::new()
    .with_k_folds(5)
    .with_stratification(true)
    .with_metrics(vec![
        ValidationMetric::Coverage,
        ValidationMetric::Compression,
        ValidationMetric::Perplexity,
    ]);

let cv_results = validator.validate(&corpus, &training_config)?;
```

### A/B Testing

```rust
use trustformers_tokenizers::training::TokenizerComparator;

let comparator = TokenizerComparator::new();
let comparison = comparator.compare_tokenizers(
    vec![tokenizer_a, tokenizer_b],
    &test_corpus,
    &comparison_metrics
)?;

println!("Best tokenizer: {}", comparison.winner);
```

## Production Considerations

### Model Versioning

```rust
use trustformers_tokenizers::training::TokenizerVersioning;

let versioning = TokenizerVersioning::new()
    .with_version("v1.0.0")
    .with_training_metadata(training_metadata)
    .with_performance_metrics(performance_metrics)
    .with_compatibility_info(compatibility_info);

versioning.save_model(&tokenizer, "tokenizer_v1.0.0")?;
```

### Deployment Pipeline

```rust
use trustformers_tokenizers::training::DeploymentPipeline;

let pipeline = DeploymentPipeline::new()
    .add_stage("validation", validation_tests)
    .add_stage("benchmark", performance_tests)
    .add_stage("compatibility", compatibility_tests)
    .add_stage("security", security_scans)
    .with_rollback_capability(true);

pipeline.deploy(&tokenizer)?;
```

## Conclusion

Training high-quality tokenizers requires careful attention to data preparation, algorithm selection, and validation. Follow these best practices to achieve optimal results:

1. **Start with clean, diverse data**
2. **Choose the right algorithm for your use case**
3. **Optimize vocabulary size based on coverage analysis**
4. **Use appropriate validation metrics**
5. **Consider domain-specific requirements**
6. **Plan for production deployment**

For additional guidance, refer to:
- [Tokenizer Selection Guide](tokenizer_selection_guide.md)
- [Performance Tuning Guide](tokenizer_performance_tuning.md)
- [Troubleshooting Guide](tokenizer_troubleshooting.md)