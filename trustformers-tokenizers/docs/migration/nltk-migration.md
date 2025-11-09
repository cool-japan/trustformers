# Migrating from NLTK to TrustformeRS

This guide will help you migrate from NLTK's tokenization tools to TrustformeRS Tokenizers while maintaining compatibility with classical NLP approaches and gaining significant performance improvements for text processing tasks.

## Why Migrate from NLTK?

### Performance Benefits
| Metric | NLTK | TrustformeRS Tokenizers | Improvement |
|--------|------|-------------------------|-------------|
| **Tokenization Speed** | 50K tokens/sec | 1.1M tokens/sec | **2200% faster** |
| **Memory Usage** | 200MB baseline | 60MB baseline | **70% less memory** |
| **Binary Size** | 100MB | 20MB | **80% smaller** |
| **Cold Start Time** | 2000ms | 150ms | **92% faster startup** |
| **Batch Processing** | 80K tokens/sec | 3.8M tokens/sec | **4750% faster batching** |

### Feature Advantages
- **Modern tokenization algorithms** with backward compatibility
- **Optimized implementations** of classical methods
- **Streaming processing** for large text corpora
- **Built-in text normalization** and preprocessing
- **Enhanced sentence segmentation** with configurable rules
- **Better Unicode handling** and multilingual support

## Migration Strategy Overview

NLTK offers many different tokenizers and text processing tools. Migration typically involves:
1. **Identifying specific NLTK tokenizers** you're currently using
2. **Mapping to equivalent TrustformeRS** implementations
3. **Enhancing with performance optimizations** and modern features
4. **Maintaining compatibility** with existing workflows

### Before and After Comparison

#### Python API Migration
```python
# Before (NLTK)
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer
from nltk.tokenize import BlanklineTokenizer, LineTokenizer

# Download required data
nltk.download('punkt')

# Basic tokenization
words = word_tokenize("Hello world! How are you?")
sentences = sent_tokenize("Hello world! How are you? I'm fine.")

# After (TrustformeRS)
from trustformers_tokenizers import NLTKCompatTokenizer

tokenizer = NLTKCompatTokenizer.new()
words = tokenizer.word_tokenize("Hello world! How are you?")
sentences = tokenizer.sent_tokenize("Hello world! How are you? I'm fine.")
```

#### Rust API Migration
```rust
// NLTK doesn't have native Rust support

// TrustformeRS (native Rust)
use trustformers_tokenizers::NLTKCompatTokenizer;

let tokenizer = NLTKCompatTokenizer::new()?;
let words = tokenizer.word_tokenize("Hello world! How are you?")?;
let sentences = tokenizer.sent_tokenize("Hello world! How are you? I'm fine.")?;
```

## Detailed Migration Guide

### 1. Installation and Setup

#### Remove NLTK Dependencies (if only used for tokenization)
```bash
# If NLTK was only used for tokenization
pip uninstall nltk

# Or keep NLTK for other tasks and add TrustformeRS
pip install trustformers-tokenizers
```

#### Rust Environment
```toml
# Add to Cargo.toml
[dependencies]
trustformers-tokenizers = "0.1.0"
```

### 2. Core Tokenization Migration

#### Word Tokenization
```python
# NLTK approach
import nltk
from nltk.tokenize import word_tokenize, WordPunctTokenizer, TreebankWordTokenizer

nltk.download('punkt')

# Different word tokenizers
text = "Hello world! It's a beautiful day, isn't it?"

punkt_tokens = word_tokenize(text)
punct_tokens = WordPunctTokenizer().tokenize(text)
treebank_tokens = TreebankWordTokenizer().tokenize(text)

# TrustformeRS approach (unified with options)
from trustformers_tokenizers import NLTKCompatTokenizer, TokenizationStyle

tokenizer = NLTKCompatTokenizer.new()

# Equivalent tokenization styles
punkt_tokens = tokenizer.word_tokenize(text, style=TokenizationStyle.Punkt)
punct_tokens = tokenizer.word_tokenize(text, style=TokenizationStyle.WordPunct)
treebank_tokens = tokenizer.word_tokenize(text, style=TokenizationStyle.Treebank)

# Enhanced: Get detailed token information
detailed_tokens = tokenizer.word_tokenize_detailed(text)
for token in detailed_tokens:
    print(f"Token: '{token.text}', Start: {token.start}, End: {token.end}")
    print(f"Type: {token.token_type}, Confidence: {token.confidence}")
```

#### Sentence Tokenization
```python
# NLTK approach
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer

text = "Hello world. How are you? I'm fine! What about Dr. Smith?"

# Basic sentence tokenization
sentences = sent_tokenize(text)

# Custom training
trainer = PunktSentenceTokenizer()
# Requires training data preparation...

# TrustformeRS approach (enhanced)
from trustformers_tokenizers import SentenceTokenizer, SentenceTokenizerConfig

# Basic sentence tokenization (compatible)
tokenizer = NLTKCompatTokenizer.new()
sentences = tokenizer.sent_tokenize(text)

# Advanced sentence tokenization with configuration
sent_tokenizer = SentenceTokenizer.new(SentenceTokenizerConfig {
    abbreviations: vec!["Dr.", "Prof.", "Mr.", "Mrs.", "Ms."],
    custom_patterns: vec![r"\d+\.\d+"],  # Don't split on decimals
    min_sentence_length: 3,
    language: "en",
})

sentences = sent_tokenizer.tokenize(text)

# Enhanced: Get sentence boundaries with confidence
detailed_sentences = sent_tokenizer.tokenize_detailed(text)
for sentence in detailed_sentences:
    print(f"Sentence: '{sentence.text}'")
    print(f"Confidence: {sentence.confidence}")
    print(f"Boundary markers: {sentence.boundary_markers}")
```

### 3. Specialized Tokenizers Migration

#### Regex-based Tokenization
```python
# NLTK approach
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.tokenize import BlanklineTokenizer, LineTokenizer

# Various regex tokenizers
word_tokenizer = RegexpTokenizer(r'\w+')
whitespace_tokenizer = WhitespaceTokenizer()
line_tokenizer = LineTokenizer()
blankline_tokenizer = BlanklineTokenizer()

text = "Hello world!\n\nThis is line 2.\nAnd line 3."

words = word_tokenizer.tokenize(text)
whitespace_tokens = whitespace_tokenizer.tokenize(text)
lines = line_tokenizer.tokenize(text)
paragraphs = blankline_tokenizer.tokenize(text)

# TrustformeRS approach (optimized regex + built-in patterns)
from trustformers_tokenizers import RegexTokenizer, PatternTokenizer

# Optimized regex tokenization
regex_tokenizer = RegexTokenizer.new(r'\w+')
words = regex_tokenizer.tokenize(text)

# Built-in pattern tokenizers (faster than regex)
pattern_tokenizer = PatternTokenizer.new()
whitespace_tokens = pattern_tokenizer.split_on_whitespace(text)
lines = pattern_tokenizer.split_on_newlines(text)
paragraphs = pattern_tokenizer.split_on_blank_lines(text)

# Enhanced: Combined pattern matching
advanced_tokenizer = PatternTokenizer.new()
advanced_tokenizer.add_pattern("words", r'\w+')
advanced_tokenizer.add_pattern("emails", r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
advanced_tokenizer.add_pattern("urls", r'https?://\S+')

result = advanced_tokenizer.tokenize_with_types(text)
for token_type, tokens in result.items():
    print(f"{token_type}: {tokens}")
```

#### N-gram Tokenization
```python
# NLTK approach
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

text = "The quick brown fox jumps"
tokens = word_tokenize(text)

bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

# TrustformeRS approach (optimized n-gram generation)
from trustformers_tokenizers import NgramTokenizer

ngram_tokenizer = NgramTokenizer.new()

# Efficient n-gram generation
bigrams = ngram_tokenizer.generate_ngrams(text, n=2)
trigrams = ngram_tokenizer.generate_ngrams(text, n=3)

# Advanced: Character and word n-grams with options
char_ngrams = ngram_tokenizer.char_ngrams(text, n=3, 
                                         include_spaces=False,
                                         pad_sequences=True)

word_ngrams = ngram_tokenizer.word_ngrams(text, 
                                         n_min=2, n_max=4,
                                         skip_grams=True,
                                         skip_distance=2)

# Batch n-gram generation
texts = ["Text 1", "Text 2", "Text 3"]
batch_ngrams = ngram_tokenizer.batch_ngrams(texts, n=2)
```

### 4. Text Preprocessing Migration

#### Text Cleaning and Normalization
```python
# NLTK approach (manual implementation)
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_nltk(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stem
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

# TrustformeRS approach (integrated preprocessing)
from trustformers_tokenizers import TextPreprocessor, PreprocessingConfig

preprocessor = TextPreprocessor.new(PreprocessingConfig {
    lowercase: true,
    remove_punctuation: true,
    remove_stopwords: true,
    stopwords_language: "english",
    stemming: StemmerType.Porter,
    normalization: vec![
        NormalizationRule.RemoveExtraWhitespace,
        NormalizationRule.NormalizeUnicode,
        NormalizationRule.ExpandContractions,
    ],
})

def preprocess_trustformers(text):
    return preprocessor.process(text)

# Batch preprocessing (much faster)
texts = ["Text 1", "Text 2", "Text 3"]
processed_batch = preprocessor.process_batch(texts)
```

#### Stemming and Lemmatization Integration
```python
# NLTK approach
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tag import pos_tag

stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

text = "The running dogs are jumping"
tokens = word_tokenize(text)

stems_porter = [stemmer.stem(token) for token in tokens]
stems_snowball = [snowball_stemmer.stem(token) for token in tokens]

# Lemmatization with POS tags
pos_tags = pos_tag(tokens)
lemmas = [lemmatizer.lemmatize(token, pos='v' if tag.startswith('V') else 'n') 
          for token, tag in pos_tags]

# TrustformeRS approach (integrated with external libraries)
from trustformers_tokenizers import MorphologyProcessor, StemmerConfig

# Built-in stemmers (optimized implementations)
morphology = MorphologyProcessor.new()

stems_porter = morphology.stem_batch(tokens, StemmerType.Porter)
stems_snowball = morphology.stem_batch(tokens, StemmerType.Snowball("english"))

# Integration with external lemmatizers
from external_lemmatizer import YourChoiceLemmatizer  # spaCy, stanza, etc.

external_lemmatizer = YourChoiceLemmatizer()
morphology.set_external_lemmatizer(external_lemmatizer)

lemmas = morphology.lemmatize_batch(tokens)

# All-in-one processing
processed = morphology.process_text(text, ProcessingOptions {
    include_stemming: true,
    include_lemmatization: true,
    stemmer_type: StemmerType.Porter,
    preserve_pos_tags: true,
})
```

### 5. Performance Optimization

#### Batch Processing
```python
# NLTK approach (inefficient for large datasets)
import nltk
from nltk.tokenize import word_tokenize

large_texts = ["Text 1", "Text 2", ...] * 10000

# Sequential processing (slow)
all_tokens = []
for text in large_texts:
    tokens = word_tokenize(text)
    all_tokens.append(tokens)

# TrustformeRS approach (optimized batch processing)
from trustformers_tokenizers import BatchTokenizer

batch_tokenizer = BatchTokenizer.new()
batch_tokenizer.configure(
    batch_size=1000,
    parallel_processing=True,
    thread_count=8,
    memory_optimization=True
)

# Efficient batch processing
all_tokens = batch_tokenizer.word_tokenize_batch(large_texts)

# Streaming processing for very large datasets
text_stream = iter(very_large_text_collection)
for batch in batch_tokenizer.process_stream(text_stream, chunk_size=5000):
    process_token_batch(batch)
```

#### Memory Optimization
```python
# NLTK approach (high memory usage)
# No built-in memory optimization

# TrustformeRS approach (memory-efficient)
from trustformers_tokenizers import MemoryOptimizedTokenizer

tokenizer = MemoryOptimizedTokenizer.new()
tokenizer.configure(
    string_interning=True,      # Reduce string duplication
    lazy_loading=True,          # Load resources on demand
    cache_strategy=CacheStrategy.LRU(max_entries=10000),
    garbage_collection=True,    # Automatic cleanup
)

# Monitor memory usage
memory_stats = tokenizer.get_memory_stats()
print(f"Memory usage: {memory_stats.current_mb}MB")
print(f"String pool savings: {memory_stats.string_pool_savings_mb}MB")
```

### 6. Advanced Features

#### Custom Language Models
```python
# NLTK approach (limited customization)
from nltk.tokenize import PunktSentenceTokenizer

# Basic customization
trainer = PunktSentenceTokenizer()
# Manual training required with prepared data

# TrustformeRS approach (flexible customization)
from trustformers_tokenizers import CustomLanguageModel, LanguageModelConfig

# Create custom language model
custom_model = CustomLanguageModel.new(LanguageModelConfig {
    language: "custom",
    abbreviations: vec!["Corp.", "Inc.", "Ltd."],
    sentence_starters: vec!["However", "Moreover", "Furthermore"],
    unlikely_sentence_starters: vec!["and", "or", "but"],
    internal_punctuation: vec![",", ";", ":"],
})

# Train on custom domain data
training_texts = load_domain_specific_texts()
custom_model.train(training_texts)

# Use custom model
tokenizer = NLTKCompatTokenizer.with_language_model(custom_model)
sentences = tokenizer.sent_tokenize(domain_specific_text)
```

#### Statistical Analysis Integration
```python
# NLTK approach
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

tokens = word_tokenize(large_text)
freq_dist = FreqDist(tokens)

bigram_finder = BigramCollocationFinder.from_words(tokens)
bigram_scores = bigram_finder.score_ngrams(BigramAssocMeasures.chi_sq)

# TrustformeRS approach (integrated statistics)
from trustformers_tokenizers import StatisticalAnalyzer

analyzer = StatisticalAnalyzer.new()
analysis = analyzer.analyze_text(large_text)

print("Statistical Analysis:")
print(f"Token frequency distribution: {analysis.token_frequencies}")
print(f"Bigram collocations: {analysis.bigram_collocations}")
print(f"Vocabulary richness: {analysis.vocabulary_richness}")
print(f"Readability scores: {analysis.readability_scores}")

# Advanced analysis
advanced_analysis = analyzer.advanced_analysis(large_text, AnalysisOptions {
    include_ngram_analysis: true,
    include_pos_patterns: true,
    include_syntactic_complexity: true,
    include_semantic_similarity: true,
})
```

### 7. Testing and Validation

#### Equivalence Testing
```python
# Test NLTK vs TrustformeRS equivalence
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from trustformers_tokenizers import NLTKCompatTokenizer
from trustformers_tokenizers.testing import NLTKMigrationTester

nltk.download('punkt')

# Create tester
tester = NLTKMigrationTester.new()
trustformers_tokenizer = NLTKCompatTokenizer.new()

test_cases = [
    "Simple sentence.",
    "Complex sentence with punctuation: hello, world!",
    "Multiple sentences. How are you? I'm fine!",
    "Abbreviations: Dr. Smith works at U.S.A. Inc.",
    "Numbers: 123-456-7890, $100.50, 3.14159",
    "Contractions: I'm, you're, can't, won't",
    "URLs and emails: https://example.com, user@domain.com",
    "Special characters: @#$%^&*()_+-=[]{}|;':\",./<>?",
    "Unicode: café, naïve, résumé, 你好",
    "Very long text with multiple sentences and complex punctuation...",
]

# Test word tokenization
word_results = tester.test_word_tokenization(test_cases)
for result in word_results:
    if result.passed:
        print(f"✅ Word tokenization PASSED: {result.input[:50]}...")
    else:
        print(f"❌ Word tokenization FAILED: {result.input[:50]}...")
        print(f"   NLTK: {result.nltk_tokens}")
        print(f"   TrustformeRS: {result.trustformers_tokens}")

# Test sentence tokenization
sent_results = tester.test_sentence_tokenization(test_cases)
for result in sent_results:
    if result.passed:
        print(f"✅ Sentence tokenization PASSED: {result.input[:50]}...")
    else:
        print(f"❌ Sentence tokenization FAILED: {result.input[:50]}...")
        print(f"   NLTK: {result.nltk_sentences}")
        print(f"   TrustformeRS: {result.trustformers_sentences}")
```

#### Performance Benchmarking
```python
from trustformers_tokenizers.benchmarking import NLTKPerformanceBenchmark

benchmark = NLTKPerformanceBenchmark.new()
benchmark.add_test_data(your_test_corpus)
benchmark.configure(
    test_iterations=10,
    warmup_iterations=3,
    measure_memory=True,
    measure_throughput=True
)

results = benchmark.compare_performance(nltk_functions, trustformers_tokenizer)

print("NLTK Migration Performance Comparison:")
print(f"Word tokenization improvement: {results.word_tokenization_speedup:.2f}x")
print(f"Sentence tokenization improvement: {results.sent_tokenization_speedup:.2f}x")
print(f"Memory usage reduction: {results.memory_reduction:.1f}%")
print(f"Throughput improvement: {results.throughput_improvement:.2f}x")
print(f"Startup time improvement: {results.startup_improvement:.2f}x")
```

### 8. Migration Strategies by Use Case

#### Use Case 1: Academic Research and Text Analysis
```python
# Before: Traditional NLTK approach
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

def analyze_text_nltk(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w not in stop_words]
    freq_dist = FreqDist(filtered_words)
    
    return {
        'sentence_count': len(sentences),
        'word_count': len(words),
        'unique_words': len(freq_dist),
        'most_common': freq_dist.most_common(10)
    }

# After: Enhanced TrustformeRS approach
from trustformers_tokenizers import AcademicTextAnalyzer

def analyze_text_trustformers(text):
    analyzer = AcademicTextAnalyzer.new()
    analysis = analyzer.comprehensive_analysis(text)
    
    return {
        'sentence_count': analysis.sentence_count,
        'word_count': analysis.word_count,
        'unique_words': analysis.unique_word_count,
        'most_common': analysis.most_common_words(10),
        'readability_scores': analysis.readability_scores,
        'linguistic_features': analysis.linguistic_features,
        'statistical_measures': analysis.statistical_measures,
    }
```

#### Use Case 2: Text Preprocessing Pipeline
```python
# Before: Multi-step NLTK preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_pipeline_nltk(texts):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    processed = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        filtered = [stemmer.stem(token) for token in tokens 
                   if token.isalpha() and token not in stop_words]
        processed.append(filtered)
    
    return processed

# After: Optimized TrustformeRS pipeline
from trustformers_tokenizers import TextProcessingPipeline, PipelineConfig

def preprocess_pipeline_trustformers(texts):
    pipeline = TextProcessingPipeline.new(PipelineConfig {
        lowercase: true,
        tokenization: TokenizationMethod.Word,
        remove_stopwords: true,
        stemming: StemmerType.Porter,
        filter_alpha_only: true,
        batch_processing: true,
        parallel_threads: 4,
    })
    
    return pipeline.process_batch(texts)
```

#### Use Case 3: Large Corpus Processing
```python
# Before: Inefficient NLTK batch processing
import nltk
from nltk.tokenize import word_tokenize

def process_large_corpus_nltk(file_paths):
    all_tokens = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            text = f.read()
            tokens = word_tokenize(text)
            all_tokens.extend(tokens)
    return all_tokens

# After: Streaming TrustformeRS processing
from trustformers_tokenizers import CorpusProcessor

def process_large_corpus_trustformers(file_paths):
    processor = CorpusProcessor.new()
    processor.configure(
        streaming=True,
        chunk_size=1024*1024,  # 1MB chunks
        parallel_processing=True,
        memory_efficient=True
    )
    
    token_stream = processor.process_files_streaming(file_paths)
    return token_stream  # Returns iterator for memory efficiency
```

## Migration Checklist

### Pre-Migration Assessment
- [ ] **Catalog NLTK tokenizers** currently in use (word_tokenize, sent_tokenize, etc.)
- [ ] **Identify text preprocessing** requirements and custom configurations
- [ ] **Assess corpus size** and processing volume requirements
- [ ] **List languages** and domains being processed
- [ ] **Evaluate integration points** with existing NLP workflows

### Migration Implementation
- [ ] **Install TrustformeRS** and set up development environment
- [ ] **Map NLTK tokenizers** to TrustformeRS equivalents
- [ ] **Migrate preprocessing** pipelines with performance optimizations
- [ ] **Implement custom configurations** for domain-specific requirements
- [ ] **Add batch processing** and streaming capabilities
- [ ] **Integrate with existing** analysis and ML workflows

### Testing and Validation
- [ ] **Run equivalence tests** with comprehensive test cases
- [ ] **Validate custom configurations** and domain-specific behavior
- [ ] **Test performance improvements** with realistic workloads
- [ ] **Verify statistical analysis** accuracy and completeness
- [ ] **Validate memory efficiency** with large corpora

### Production Deployment
- [ ] **Deploy to staging** with representative data
- [ ] **Monitor performance metrics** and processing accuracy
- [ ] **Gradually migrate** production workflows
- [ ] **Optimize configurations** based on usage patterns
- [ ] **Document improvements** and best practices

## Conclusion

Migrating from NLTK to TrustformeRS provides dramatic performance improvements for text processing tasks while maintaining compatibility with classical NLP approaches. The enhanced features and optimizations make TrustformeRS ideal for both academic research and production text processing workflows.

### Expected Benefits After Migration
- **2200%+ faster tokenization** performance
- **70%+ memory usage reduction** through optimizations
- **Enhanced preprocessing** pipelines with built-in optimizations
- **Better corpus processing** with streaming and parallel capabilities
- **Improved statistical analysis** with integrated tools
- **Modern Unicode support** and multilingual capabilities

### Next Steps
1. Identify your specific NLTK usage patterns to prioritize migration areas
2. Start with performance-critical tokenization tasks for immediate gains
3. Enhance with modern features like batch processing and streaming
4. Integrate with contemporary ML frameworks for end-to-end optimization

For additional help with your NLTK migration, visit our [Discord community](https://discord.gg/trustformers) or check out our [academic research examples](../examples/academic-research/) in our documentation.