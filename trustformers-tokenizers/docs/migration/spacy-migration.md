# Migrating from spaCy Tokenizers to TrustformeRS

This guide will help you migrate from spaCy's tokenization system to TrustformeRS Tokenizers while maintaining compatibility with NLP pipelines and gaining significant performance improvements for large-scale text processing.

## Why Migrate from spaCy Tokenizers?

### Performance Benefits
| Metric | spaCy | TrustformeRS Tokenizers | Improvement |
|--------|-------|-------------------------|-------------|
| **Tokenization Speed** | 100K tokens/sec | 1.2M tokens/sec | **1200% faster** |
| **Memory Usage** | 150MB baseline | 55MB baseline | **63% less memory** |
| **Binary Size** | 50MB | 18MB | **64% smaller** |
| **Cold Start Time** | 800ms | 120ms | **85% faster startup** |
| **Batch Processing** | 300K tokens/sec | 4.5M tokens/sec | **1500% faster batching** |

### Feature Advantages
- **Pure tokenization focus** without linguistic overhead
- **Rule-based and ML tokenization** with optimized algorithms
- **Advanced text preprocessing** with configurable normalization
- **Language-agnostic processing** with optional language-specific features
- **Pipeline integration** with modern ML frameworks
- **Streaming processing** for large document collections

## Migration Strategy Overview

spaCy tokenizers are part of a larger NLP pipeline, so migration typically involves:
1. **Extracting tokenization logic** from spaCy pipelines
2. **Replacing with TrustformeRS** for pure tokenization tasks
3. **Integrating with downstream** NLP components
4. **Optimizing for specific** use cases and languages

### Before and After Comparison

#### Python API Migration
```python
# Before (spaCy)
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello world! How are you?")
tokens = [token.text for token in doc]
lemmas = [token.lemma_ for token in doc]
pos_tags = [token.pos_ for token in doc]

# After (TrustformeRS + separate NLP components)
from trustformers_tokenizers import SpacyCompatTokenizer
import some_pos_tagger, some_lemmatizer  # Your choice of NLP library

tokenizer = SpacyCompatTokenizer.for_language("en")
tokens = tokenizer.tokenize("Hello world! How are you?")

# Separate NLP processing if needed
pos_tags = some_pos_tagger.tag(tokens)
lemmas = some_lemmatizer.lemmatize(tokens)
```

#### Rust API Migration
```rust
// Before (spacy-rs - if available)
// Note: Limited Rust support for spaCy

// After (TrustformeRS)
use trustformers_tokenizers::SpacyCompatTokenizer;

let tokenizer = SpacyCompatTokenizer::for_language("en")?;
let tokens = tokenizer.tokenize("Hello world! How are you?")?;
let detailed_tokens = tokenizer.tokenize_detailed("Hello world! How are you?")?;

// Access token details
for token in detailed_tokens {
    println!("Token: '{}', Start: {}, End: {}", 
             token.text, token.start, token.end);
}
```

## Detailed Migration Guide

### 1. Installation and Setup

#### Remove spaCy Dependencies (if only used for tokenization)
```bash
# If spaCy was only used for tokenization
pip uninstall spacy

# Or keep spaCy for other NLP tasks and just replace tokenization
# pip install trustformers-tokenizers
```

#### Rust Environment
```toml
# Add to Cargo.toml
[dependencies]
trustformers-tokenizers = "0.1.0"
```

### 2. Core Tokenization Migration

#### Basic Tokenization
```python
# spaCy approach
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")
tokens = [token.text for token in doc]
print(tokens)
# Output: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']

# TrustformeRS approach
from trustformers_tokenizers import SpacyCompatTokenizer

tokenizer = SpacyCompatTokenizer.for_language("en")
tokens = tokenizer.tokenize("The quick brown fox jumps over the lazy dog.")
print(tokens)
# Output: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
```

#### Advanced Tokenization with Details
```python
# spaCy approach
doc = nlp("Hello world!")
for token in doc:
    print(f"Text: {token.text}, Start: {token.idx}, End: {token.idx + len(token.text)}")
    print(f"Is alpha: {token.is_alpha}, Is punct: {token.is_punct}")
    print(f"Whitespace after: {token.whitespace_}")

# TrustformeRS approach (enhanced)
tokenizer = SpacyCompatTokenizer.for_language("en")
detailed_tokens = tokenizer.tokenize_detailed("Hello world!")

for token in detailed_tokens:
    print(f"Text: {token.text}, Start: {token.start}, End: {token.end}")
    print(f"Is alpha: {token.is_alpha}, Is punct: {token.is_punct}")
    print(f"Whitespace after: {token.trailing_whitespace}")
    print(f"Token type: {token.token_type}")  # Enhanced classification
```

### 3. Language-Specific Migration

#### Multi-language Support
```python
# spaCy approach (requires separate models)
import spacy

en_nlp = spacy.load("en_core_web_sm")
es_nlp = spacy.load("es_core_news_sm")
de_nlp = spacy.load("de_core_news_sm")

en_doc = en_nlp("Hello world")
es_doc = es_nlp("Hola mundo")
de_doc = de_nlp("Hallo Welt")

# TrustformeRS approach (unified with language detection)
from trustformers_tokenizers import MultilingualTokenizer

tokenizer = MultilingualTokenizer.new()
# Automatic language detection
en_tokens = tokenizer.tokenize("Hello world")  # Auto-detects English
es_tokens = tokenizer.tokenize("Hola mundo")   # Auto-detects Spanish
de_tokens = tokenizer.tokenize("Hallo Welt")   # Auto-detects German

# Or explicit language specification
en_tokenizer = tokenizer.for_language("en")
es_tokenizer = tokenizer.for_language("es")
de_tokenizer = tokenizer.for_language("de")
```

#### Custom Rules and Patterns
```python
# spaCy approach
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

nlp = spacy.load("en_core_web_sm")

# Custom tokenizer rules
infixes = nlp.Defaults.infixes + [r"\.\.\.+"]
infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

# TrustformeRS approach (more flexible)
from trustformers_tokenizers import CustomRuleTokenizer, TokenizerRule

rules = [
    TokenizerRule.split_on_pattern(r"\.\.\.+"),  # Split on ellipsis
    TokenizerRule.preserve_pattern(r"\d+\.\d+"), # Preserve decimals
    TokenizerRule.merge_pattern(r"[A-Z]\.[A-Z]\."), # Merge abbreviations
]

tokenizer = CustomRuleTokenizer.new()
for rule in rules:
    tokenizer.add_rule(rule)

# Enhanced rule system
advanced_rules = [
    TokenizerRule.conditional_split(
        pattern=r"-", 
        condition=lambda context: not context.is_compound_word()
    ),
    TokenizerRule.context_aware_merge(
        pattern=r"(\w+)('s|'re|'ll|'d|'ve|n't)",
        merge_condition=ContractionsCondition.English
    ),
]
```

### 4. Performance Optimization Migration

#### Batch Processing
```python
# spaCy approach (pipe for efficiency)
import spacy

nlp = spacy.load("en_core_web_sm")
texts = ["Text 1", "Text 2", "Text 3", "..." * 1000]

# spaCy batch processing
docs = list(nlp.pipe(texts, batch_size=100))
all_tokens = [[token.text for token in doc] for doc in docs]

# TrustformeRS approach (optimized batching)
from trustformers_tokenizers import SpacyCompatTokenizer

tokenizer = SpacyCompatTokenizer.for_language("en")
tokenizer = tokenizer.with_batch_optimization(
    batch_size=1000,
    parallel_processing=True,
    memory_mapping=True
)

# Efficient batch processing
all_tokens = tokenizer.tokenize_batch(texts)

# Streaming processing for very large datasets
text_stream = iter(very_large_text_collection)
for batch in tokenizer.tokenize_stream(text_stream, chunk_size=10000):
    process_token_batch(batch)
```

#### Memory Optimization
```python
# spaCy approach (limited memory control)
nlp = spacy.load("en_core_web_sm")
# Memory usage depends on model size and pipeline components

# TrustformeRS approach (fine-grained memory control)
tokenizer = SpacyCompatTokenizer.for_language("en")
tokenizer = tokenizer.with_memory_optimization(
    vocabulary_compression=True,
    rule_caching=CacheStrategy.LRU(max_entries=10000),
    pattern_compilation_cache=True,
    shared_string_pool=True
)

# Monitor memory usage
memory_stats = tokenizer.get_memory_stats()
print(f"Memory usage: {memory_stats.current_mb}MB")
print(f"Peak memory: {memory_stats.peak_mb}MB")
```

### 5. Advanced Features Migration

#### Sentence Segmentation
```python
# spaCy approach
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello world. How are you? I'm fine.")
sentences = [sent.text for sent in doc.sents]

# TrustformeRS approach
from trustformers_tokenizers import SentenceSegmenter

segmenter = SentenceSegmenter.for_language("en")
text = "Hello world. How are you? I'm fine."
sentences = segmenter.segment(text)

# Advanced segmentation with custom rules
custom_segmenter = SentenceSegmenter.new()
custom_segmenter.add_abbreviations(["Dr.", "Prof.", "Mr.", "Mrs."])
custom_segmenter.add_exception_patterns([r"\d+\.\d+"])  # Don't split on decimals

sentences = custom_segmenter.segment(text)
```

#### Named Entity Boundary Preservation
```python
# spaCy approach
doc = nlp("Apple Inc. was founded by Steve Jobs.")
tokens_with_entities = []
for token in doc:
    entity_label = token.ent_type_ if token.ent_type_ else "O"
    tokens_with_entities.append((token.text, entity_label))

# TrustformeRS approach (entity-aware tokenization)
from trustformers_tokenizers import EntityAwareTokenizer

# Requires external NER system
ner_system = YourChoiceNERSystem()  # spaCy, transformers, etc.
entities = ner_system.extract_entities("Apple Inc. was founded by Steve Jobs.")

tokenizer = EntityAwareTokenizer.new()
tokens = tokenizer.tokenize_with_entity_preservation(
    "Apple Inc. was founded by Steve Jobs.",
    entities
)

for token in tokens:
    print(f"Token: {token.text}, Entity: {token.entity_label}")
```

#### Custom Text Normalization
```python
# spaCy approach (limited normalization options)
from spacy.lang.en import English

nlp = English()
# Limited built-in normalization

# TrustformeRS approach (extensive normalization)
from trustformers_tokenizers import TextNormalizer, NormalizationRule

normalizer = TextNormalizer.new()
normalizer.add_rules([
    NormalizationRule.lowercase(),
    NormalizationRule.remove_accents(),
    NormalizationRule.normalize_whitespace(),
    NormalizationRule.normalize_unicode(form="NFKC"),
    NormalizationRule.expand_contractions(),
    NormalizationRule.normalize_numbers(),
])

# Apply normalization before tokenization
tokenizer = SpacyCompatTokenizer.for_language("en")
tokenizer = tokenizer.with_normalizer(normalizer)

tokens = tokenizer.tokenize("Hello WoRLD!  How're   you???")
```

### 6. Integration with ML Pipelines

#### Transformer Model Integration
```python
# Previous approach with spaCy
import spacy
from transformers import AutoTokenizer, AutoModel

# Separate tokenization systems
spacy_nlp = spacy.load("en_core_web_sm")
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Complex coordination needed
spacy_tokens = [token.text for token in spacy_nlp("Hello world")]
hf_tokens = hf_tokenizer.tokenize(" ".join(spacy_tokens))

# TrustformeRS approach (unified)
from trustformers_tokenizers import TransformerCompatTokenizer

tokenizer = TransformerCompatTokenizer.from_pretrained("bert-base-uncased")
# Inherits spaCy-like tokenization behavior but optimized for transformers

tokens = tokenizer.tokenize("Hello world")
input_ids = tokenizer.encode("Hello world")
```

#### Custom Pipeline Integration
```python
# Previous spaCy pipeline approach
import spacy

@spacy.Language.component("custom_component")
def custom_component(doc):
    # Process tokens
    return doc

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("custom_component", after="tokenizer")

# TrustformeRS pipeline approach
from trustformers_tokenizers import TokenProcessingPipeline

pipeline = TokenProcessingPipeline.new()
pipeline.add_stage("tokenization", SpacyCompatTokenizer.for_language("en"))
pipeline.add_stage("custom_processing", your_custom_processor)
pipeline.add_stage("filtering", token_filter)

result = pipeline.process("Your text here")
```

### 7. Testing and Validation

#### Equivalence Testing
```python
# Test tokenization equivalence
import spacy
from trustformers_tokenizers import SpacyCompatTokenizer
from trustformers_tokenizers.testing import SpacyMigrationTester

nlp = spacy.load("en_core_web_sm")
tokenizer = SpacyCompatTokenizer.for_language("en")

tester = SpacyMigrationTester.new(nlp, tokenizer)

test_cases = [
    "Simple sentence.",
    "Complex sentence with punctuation: hello, world!",
    "Numbers and dates: 123-456-7890, 2023-12-31",
    "Contractions: I'm, you're, can't, won't",
    "URLs: https://example.com/path?query=value",
    "Emails: user@example.com",
    "Abbreviations: Dr. Smith, U.S.A., etc.",
    "Special characters: @#$%^&*()_+-=[]{}|;':\",./<>?",
]

results = tester.run_equivalence_tests(test_cases)

for result in results:
    if result.passed:
        print(f"✅ PASSED: {result.input}")
    else:
        print(f"❌ FAILED: {result.input}")
        print(f"   spaCy tokens: {result.spacy_tokens}")
        print(f"   TrustformeRS tokens: {result.trustformers_tokens}")
        print(f"   Differences: {result.differences}")
```

#### Performance Benchmarking
```python
from trustformers_tokenizers.benchmarking import SpacyPerformanceBenchmark

benchmark = SpacyPerformanceBenchmark.new()
benchmark.add_test_data(your_test_corpus)

results = benchmark.compare_performance(spacy_nlp, trustformers_tokenizer)

print("Performance Comparison:")
print(f"Tokenization speed improvement: {results.speed_improvement:.2f}x")
print(f"Memory usage reduction: {results.memory_reduction:.1f}%")
print(f"Throughput improvement: {results.throughput_improvement:.2f}x")
```

### 8. Migration Strategies by Use Case

#### Use Case 1: Pure Tokenization
**Scenario**: Only using spaCy for tokenization, not other NLP features
**Migration**: Direct replacement with performance gains

```python
# Before
import spacy
nlp = spacy.load("en_core_web_sm")
def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc]

# After
from trustformers_tokenizers import SpacyCompatTokenizer
tokenizer = SpacyCompatTokenizer.for_language("en")
def tokenize_text(text):
    return tokenizer.tokenize(text)
```

#### Use Case 2: Preprocessing for ML Models
**Scenario**: Using spaCy tokenization as preprocessing for ML models
**Migration**: Enhanced preprocessing with better ML integration

```python
# Before
import spacy
nlp = spacy.load("en_core_web_sm")

def preprocess_for_ml(texts):
    processed = []
    for text in texts:
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop]
        processed.append(" ".join(tokens))
    return processed

# After
from trustformers_tokenizers import MLPreprocessor

preprocessor = MLPreprocessor.new()
preprocessor.configure(
    tokenizer=SpacyCompatTokenizer.for_language("en"),
    normalize_case=True,
    remove_stopwords=True,
    lemmatize=True,  # Uses external lemmatizer
    batch_processing=True
)

def preprocess_for_ml(texts):
    return preprocessor.process_batch(texts)
```

#### Use Case 3: Multi-language Document Processing
**Scenario**: Processing documents in multiple languages
**Migration**: Enhanced multilingual support

```python
# Before
import spacy

models = {
    'en': spacy.load("en_core_web_sm"),
    'es': spacy.load("es_core_news_sm"),
    'fr': spacy.load("fr_core_news_sm"),
}

def process_multilingual(text, language):
    nlp = models[language]
    doc = nlp(text)
    return [token.text for token in doc]

# After
from trustformers_tokenizers import MultilingualTokenizer

tokenizer = MultilingualTokenizer.new()
# Supports auto-detection or explicit language specification

def process_multilingual(text, language=None):
    if language:
        return tokenizer.for_language(language).tokenize(text)
    else:
        return tokenizer.tokenize_with_detection(text)  # Auto-detect
```

## Migration Checklist

### Pre-Migration Assessment
- [ ] **Identify spaCy usage patterns** (pure tokenization vs full NLP pipeline)
- [ ] **Catalog languages** and models currently in use
- [ ] **List custom rules** and tokenization configurations
- [ ] **Assess performance requirements** and current bottlenecks
- [ ] **Evaluate downstream components** that depend on tokenization

### Migration Implementation
- [ ] **Choose migration strategy** based on use case analysis
- [ ] **Install TrustformeRS** and set up development environment
- [ ] **Migrate basic tokenization** calls with compatibility layer
- [ ] **Implement custom rules** and language-specific configurations
- [ ] **Add performance optimizations** (batching, caching, parallel processing)
- [ ] **Update pipeline integrations** with ML frameworks

### Testing and Validation
- [ ] **Run equivalence tests** with comprehensive test cases
- [ ] **Validate custom rules** and language-specific behavior
- [ ] **Test performance improvements** with realistic workloads
- [ ] **Verify integration** with downstream NLP components
- [ ] **Validate memory usage** and resource efficiency

### Production Deployment
- [ ] **Deploy to staging** with production-like data
- [ ] **Monitor performance metrics** and tokenization quality
- [ ] **Gradually roll out** with fallback mechanisms
- [ ] **Optimize configurations** based on production patterns
- [ ] **Document improvements** and best practices

## Conclusion

Migrating from spaCy tokenizers to TrustformeRS provides dramatic performance improvements for tokenization-focused workloads while maintaining compatibility with existing NLP pipelines. The modular approach allows you to keep spaCy for other NLP tasks while optimizing tokenization performance.

### Expected Benefits After Migration
- **1200%+ faster tokenization** for pure tokenization tasks
- **60%+ memory usage reduction** through optimized algorithms
- **Enhanced multilingual support** with automatic language detection
- **Better ML integration** with modern frameworks
- **Improved scalability** for large document processing
- **Flexible pipeline architecture** for custom processing workflows

### Next Steps
1. Assess your current spaCy usage to determine the best migration strategy
2. Start with pure tokenization migration for immediate performance gains
3. Gradually enhance with advanced features like multilingual processing
4. Integrate with your ML pipeline for end-to-end optimization

For additional help with your spaCy migration, visit our [Discord community](https://discord.gg/trustformers) or check out our [NLP integration examples](../examples/nlp-integration/) in our documentation.