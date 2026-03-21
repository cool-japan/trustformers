# TrustformeRS Tokenizers - Common Use Cases and Examples

This document provides practical examples for common tokenization scenarios and use cases.

## Table of Contents

- [Basic Tokenization](#basic-tokenization)
- [Batch Processing](#batch-processing)
- [Model Fine-tuning Preparation](#model-fine-tuning-preparation)
- [Text Preprocessing Pipelines](#text-preprocessing-pipelines)
- [Multilingual Processing](#multilingual-processing)
- [Domain-Specific Tokenization](#domain-specific-tokenization)
- [Performance Optimization](#performance-optimization)
- [Custom Tokenization Rules](#custom-tokenization-rules)
- [Integration with ML Pipelines](#integration-with-ml-pipelines)
- [Advanced Analysis and Debugging](#advanced-analysis-and-debugging)

## Basic Tokenization

### Loading and Using Pre-trained Tokenizers

```rust
use trustformers_tokenizers::TokenizerImpl;

fn basic_tokenization() -> Result<(), Box<dyn std::error::Error>> {
    // Load a pre-trained tokenizer
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    
    // Basic encoding
    let text = "Hello, world! How are you today?";
    let encoded = tokenizer.encode(text)?;
    
    println!("Original text: {}", text);
    println!("Token IDs: {:?}", encoded.ids());
    println!("Tokens: {:?}", encoded.tokens());
    println!("Attention mask: {:?}", encoded.attention_mask());
    
    // Decoding
    let decoded = tokenizer.decode(encoded.ids())?;
    println!("Decoded text: {}", decoded);
    
    Ok(())
}
```

### Working with Different Tokenizer Types

```rust
use trustformers_tokenizers::{BPETokenizer, WordPieceTokenizer, UnigramTokenizer};

fn different_tokenizer_types() -> Result<(), Box<dyn std::error::Error>> {
    let text = "The quick brown fox jumps over the lazy dog.";
    
    // BPE Tokenizer
    let bpe_tokenizer = BPETokenizer::from_files("vocab.json", "merges.txt")?;
    let bpe_tokens = bpe_tokenizer.encode(text)?;
    println!("BPE tokens: {:?}", bpe_tokens.tokens());
    
    // WordPiece Tokenizer  
    let wp_tokenizer = WordPieceTokenizer::from_vocab(vocab, "[UNK]")?;
    let wp_tokens = wp_tokenizer.encode(text)?;
    println!("WordPiece tokens: {:?}", wp_tokens.tokens());
    
    // Unigram Tokenizer
    let unigram_tokenizer = UnigramTokenizer::from_vocab_and_scores(vocab, scores)?;
    let unigram_tokens = unigram_tokenizer.encode(text)?;
    println!("Unigram tokens: {:?}", unigram_tokens.tokens());
    
    Ok(())
}
```

## Batch Processing

### Efficient Batch Tokenization

```rust
use trustformers_tokenizers::{TokenizerImpl, ParallelTokenizer};

fn batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    
    let texts = vec![
        "First example sentence.",
        "Second example with different length.",
        "A much longer third example sentence with more words to tokenize.",
        "Short one.",
        "Another example that demonstrates batch processing capabilities.",
    ];
    
    // Basic batch processing
    let batch_encoded = tokenizer.encode_batch(&texts)?;
    
    for (i, encoded) in batch_encoded.iter().enumerate() {
        println!("Text {}: {} tokens", i, encoded.tokens().len());
    }
    
    // Parallel batch processing for large datasets
    let parallel_tokenizer = ParallelTokenizer::new(tokenizer)
        .with_thread_count(4)
        .with_chunk_size(1000);
    
    let large_batch: Vec<String> = (0..10000)
        .map(|i| format!("Example text number {}", i))
        .collect();
    
    let start = std::time::Instant::now();
    let parallel_results = parallel_tokenizer.encode_batch(&large_batch)?;
    let duration = start.elapsed();
    
    println!("Processed {} texts in {:?}", large_batch.len(), duration);
    println!("Throughput: {:.0} texts/sec", large_batch.len() as f64 / duration.as_secs_f64());
    
    Ok(())
}
```

### Streaming Large Files

```rust
use trustformers_tokenizers::{StreamingTokenizer, TextFileIterator};

fn streaming_processing() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    let streaming_tokenizer = StreamingTokenizer::new(tokenizer);
    
    // Process very large files without loading into memory
    let file_iterator = TextFileIterator::new("large_corpus.txt")?
        .with_chunk_size(1024 * 1024) // 1MB chunks
        .with_overlap(100); // 100 character overlap
    
    let mut total_tokens = 0;
    let mut processed_chunks = 0;
    
    for chunk_result in streaming_tokenizer.process_iterator(file_iterator) {
        let tokens = chunk_result?;
        total_tokens += tokens.len();
        processed_chunks += 1;
        
        if processed_chunks % 100 == 0 {
            println!("Processed {} chunks, {} total tokens", processed_chunks, total_tokens);
        }
    }
    
    println!("Final: {} chunks, {} total tokens", processed_chunks, total_tokens);
    
    Ok(())
}
```

## Model Fine-tuning Preparation

### Preparing Data for BERT Fine-tuning

```rust
use trustformers_tokenizers::{TokenizerImpl, SpecialTokenManager};

fn prepare_bert_finetuning_data() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    
    // Classification task preparation
    let classification_examples = vec![
        ("This movie is great!", "positive"),
        ("I didn't like this film.", "negative"),
        ("The acting was superb!", "positive"),
        ("Boring and predictable plot.", "negative"),
    ];
    
    let mut prepared_data = Vec::new();
    
    for (text, label) in classification_examples {
        let encoded = tokenizer.encode(text)?;
        
        // Ensure sequences fit within BERT's max length
        let max_length = 512;
        let input_ids = if encoded.ids().len() > max_length {
            &encoded.ids()[..max_length]
        } else {
            encoded.ids()
        };
        
        // Create attention mask
        let attention_mask: Vec<u32> = input_ids.iter().map(|&id| {
            if id == tokenizer.get_vocab().get_pad_id() { 0 } else { 1 }
        }).collect();
        
        prepared_data.push((input_ids.to_vec(), attention_mask, label));
    }
    
    // Save prepared data
    for (i, (input_ids, attention_mask, label)) in prepared_data.iter().enumerate() {
        println!("Example {}: {}", i, label);
        println!("  Input IDs: {:?}", input_ids);
        println!("  Attention mask: {:?}", attention_mask);
        println!();
    }
    
    Ok(())
}
```

### Question Answering Data Preparation

```rust
use trustformers_tokenizers::TokenizerImpl;

fn prepare_qa_data() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    
    let qa_examples = vec![
        (
            "The quick brown fox jumps over the lazy dog.",
            "What color is the fox?",
            "brown",
            21, 26  // character positions of answer
        ),
    ];
    
    for (context, question, answer, answer_start, answer_end) in qa_examples {
        // Encode question and context separately
        let question_encoded = tokenizer.encode(question)?;
        let context_encoded = tokenizer.encode(context)?;
        
        // Combine with [SEP] token
        let sep_id = tokenizer.get_vocab().get_token_id("[SEP]").unwrap();
        let cls_id = tokenizer.get_vocab().get_token_id("[CLS]").unwrap();
        
        let mut input_ids = vec![cls_id];
        input_ids.extend(question_encoded.ids());
        input_ids.push(sep_id);
        input_ids.extend(context_encoded.ids());
        input_ids.push(sep_id);
        
        // Create token type IDs (0 for question, 1 for context)
        let mut token_type_ids = vec![0]; // [CLS]
        token_type_ids.extend(vec![0; question_encoded.ids().len()]); // question
        token_type_ids.push(0); // [SEP]
        token_type_ids.extend(vec![1; context_encoded.ids().len()]); // context
        token_type_ids.push(1); // [SEP]
        
        println!("QA Example:");
        println!("  Input IDs: {:?}", input_ids);
        println!("  Token type IDs: {:?}", token_type_ids);
        println!("  Answer: {}", answer);
        println!();
    }
    
    Ok(())
}
```

## Text Preprocessing Pipelines

### News Article Processing Pipeline

```rust
use trustformers_tokenizers::{TokenizerImpl, SpecialTokenManager};
use regex::Regex;

fn news_processing_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    
    // Sample news articles
    let articles = vec![
        "BREAKING: Major earthquake hits California. Scientists report magnitude 7.2 quake centered near Los Angeles at 3:47 PM PST.",
        "SPORTS: Local team wins championship after overtime victory. Coach says 'This is the result of months of hard work.'",
        "TECH: New AI model achieves state-of-the-art results on language understanding tasks. Researchers from Stanford published their findings.",
    ];
    
    // Preprocessing patterns
    let url_regex = Regex::new(r"https?://\S+")?;
    let mention_regex = Regex::new(r"@\w+")?;
    let time_regex = Regex::new(r"\d{1,2}:\d{2}\s*(AM|PM|am|pm)")?;
    
    for (i, article) in articles.iter().enumerate() {
        println!("=== Article {} ===", i + 1);
        println!("Original: {}", article);
        
        // Clean the text
        let mut cleaned = article.to_string();
        
        // Replace URLs with special token
        cleaned = url_regex.replace_all(&cleaned, "[URL]").to_string();
        
        // Replace mentions with special token
        cleaned = mention_regex.replace_all(&cleaned, "[MENTION]").to_string();
        
        // Normalize time expressions
        cleaned = time_regex.replace_all(&cleaned, "[TIME]").to_string();
        
        println!("Cleaned: {}", cleaned);
        
        // Tokenize
        let encoded = tokenizer.encode(&cleaned)?;
        println!("Tokens: {:?}", encoded.tokens());
        println!("Token count: {}", encoded.tokens().len());
        
        // Extract key information
        if cleaned.contains("BREAKING:") {
            println!("Category: Breaking News");
        } else if cleaned.contains("SPORTS:") {
            println!("Category: Sports");
        } else if cleaned.contains("TECH:") {
            println!("Category: Technology");
        }
        
        println!();
    }
    
    Ok(())
}
```

### Social Media Text Processing

```rust
use trustformers_tokenizers::{TokenizerImpl, SpecialTokenManager};
use regex::Regex;

fn social_media_processing() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    
    let posts = vec![
        "Just had the best coffee ☕ at @starbucks! #coffee #morning https://example.com/photo",
        "Can't believe what happened today 😱 @friend1 @friend2 you need to see this!",
        "Beautiful sunset 🌅 #photography #nature #blessed",
    ];
    
    // Social media specific patterns
    let hashtag_regex = Regex::new(r"#\w+")?;
    let mention_regex = Regex::new(r"@\w+")?;
    let url_regex = Regex::new(r"https?://\S+")?;
    let emoji_regex = Regex::new(r"[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]")?;
    
    for (i, post) in posts.iter().enumerate() {
        println!("=== Post {} ===", i + 1);
        println!("Original: {}", post);
        
        // Extract hashtags, mentions, URLs before cleaning
        let hashtags: Vec<&str> = hashtag_regex.find_iter(post).map(|m| m.as_str()).collect();
        let mentions: Vec<&str> = mention_regex.find_iter(post).map(|m| m.as_str()).collect();
        let urls: Vec<&str> = url_regex.find_iter(post).map(|m| m.as_str()).collect();
        
        // Clean for tokenization
        let mut cleaned = post.to_string();
        cleaned = hashtag_regex.replace_all(&cleaned, "[HASHTAG]").to_string();
        cleaned = mention_regex.replace_all(&cleaned, "[MENTION]").to_string();
        cleaned = url_regex.replace_all(&cleaned, "[URL]").to_string();
        cleaned = emoji_regex.replace_all(&cleaned, "[EMOJI]").to_string();
        
        println!("Cleaned: {}", cleaned);
        
        // Tokenize
        let encoded = tokenizer.encode(&cleaned)?;
        println!("Tokens: {:?}", encoded.tokens());
        
        // Display extracted entities
        if !hashtags.is_empty() {
            println!("Hashtags: {:?}", hashtags);
        }
        if !mentions.is_empty() {
            println!("Mentions: {:?}", mentions);
        }
        if !urls.is_empty() {
            println!("URLs: {:?}", urls);
        }
        
        println!();
    }
    
    Ok(())
}
```

## Multilingual Processing

### Cross-lingual Document Processing

```rust
use trustformers_tokenizers::{TokenizerImpl, ChineseTokenizer, JapaneseTokenizer, ArabicTokenizer};

fn multilingual_processing() -> Result<(), Box<dyn std::error::Error>> {
    // Load multilingual tokenizer
    let multilingual_tokenizer = TokenizerImpl::from_pretrained("bert-base-multilingual-cased")?;
    
    // Language-specific tokenizers
    let chinese_tokenizer = ChineseTokenizer::new(Default::default());
    let japanese_tokenizer = JapaneseTokenizer::new(JapaneseMode::Word);
    let arabic_tokenizer = ArabicTokenizer::new(ArabicMode::Word);
    
    let multilingual_texts = vec![
        ("en", "Hello, how are you today?"),
        ("zh", "你好，今天过得怎么样？"),
        ("ja", "こんにちは、今日はいかがですか？"),
        ("ar", "مرحبا، كيف حالك اليوم؟"),
        ("es", "Hola, ¿cómo estás hoy?"),
        ("fr", "Bonjour, comment allez-vous aujourd'hui?"),
    ];
    
    for (lang, text) in multilingual_texts {
        println!("=== {} ({}) ===", lang.to_uppercase(), text);
        
        // General multilingual tokenization
        let multilingual_encoded = multilingual_tokenizer.encode(text)?;
        println!("Multilingual tokens: {:?}", multilingual_encoded.tokens());
        
        // Language-specific tokenization where available
        match lang {
            "zh" => {
                let chinese_tokens = chinese_tokenizer.tokenize(text)?;
                println!("Chinese-specific tokens: {:?}", chinese_tokens);
            },
            "ja" => {
                let japanese_tokens = japanese_tokenizer.tokenize(text)?;
                println!("Japanese-specific tokens: {:?}", japanese_tokens);
            },
            "ar" => {
                let arabic_tokens = arabic_tokenizer.tokenize(text)?;
                println!("Arabic-specific tokens: {:?}", arabic_tokens);
            },
            _ => {
                println!("Using general multilingual tokenization");
            }
        }
        
        println!();
    }
    
    Ok(())
}
```

### Translation Data Preparation

```rust
use trustformers_tokenizers::TokenizerImpl;

fn translation_data_preparation() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-multilingual-cased")?;
    
    // Translation pairs (English -> Spanish)
    let translation_pairs = vec![
        ("Hello, world!", "¡Hola, mundo!"),
        ("How are you?", "¿Cómo estás?"),
        ("Good morning", "Buenos días"),
        ("Thank you very much", "Muchas gracias"),
    ];
    
    for (en_text, es_text) in translation_pairs {
        println!("EN: {}", en_text);
        println!("ES: {}", es_text);
        
        // Encode source and target
        let en_encoded = tokenizer.encode(en_text)?;
        let es_encoded = tokenizer.encode(es_text)?;
        
        println!("EN tokens: {:?}", en_encoded.tokens());
        println!("ES tokens: {:?}", es_encoded.tokens());
        
        // Calculate length ratio for training
        let length_ratio = es_encoded.tokens().len() as f32 / en_encoded.tokens().len() as f32;
        println!("Length ratio (ES/EN): {:.2}", length_ratio);
        
        println!();
    }
    
    Ok(())
}
```

## Domain-Specific Tokenization

### Scientific Text Processing

```rust
use trustformers_tokenizers::{TokenizerImpl, MathTokenizer, ChemicalTokenizer, BioTokenizer};

fn scientific_text_processing() -> Result<(), Box<dyn std::error::Error>> {
    let general_tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    let math_tokenizer = MathTokenizer::new();
    let chemical_tokenizer = ChemicalTokenizer::new();
    let bio_tokenizer = BioTokenizer::new();
    
    // Mathematical text
    let math_text = "The derivative of sin(x) is cos(x), and ∫cos(x)dx = sin(x) + C";
    println!("=== Mathematical Text ===");
    println!("Text: {}", math_text);
    
    let general_math = general_tokenizer.encode(math_text)?;
    println!("General tokens: {:?}", general_math.tokens());
    
    let math_tokens = math_tokenizer.tokenize(math_text)?;
    println!("Math-specific tokens: {:?}", math_tokens);
    
    let math_analysis = math_tokenizer.analyze(math_text)?;
    println!("Complexity score: {:.2}", math_analysis.complexity_score);
    
    // Chemical text
    let chemical_text = "The reaction H2SO4 + 2NaOH → Na2SO4 + 2H2O is acid-base neutralization";
    println!("\n=== Chemical Text ===");
    println!("Text: {}", chemical_text);
    
    let general_chem = general_tokenizer.encode(chemical_text)?;
    println!("General tokens: {:?}", general_chem.tokens());
    
    let chem_tokens = chemical_tokenizer.tokenize_formula("H2SO4")?;
    println!("Chemical formula tokens: {:?}", chem_tokens);
    
    // Biological sequence
    let dna_sequence = "ATCGATCGTAGCTAGC";
    println!("\n=== Biological Sequence ===");
    println!("DNA: {}", dna_sequence);
    
    let dna_tokens = bio_tokenizer.tokenize_dna(dna_sequence)?;
    println!("DNA tokens: {:?}", dna_tokens);
    
    let bio_analysis = bio_tokenizer.analyze_sequence(dna_sequence)?;
    println!("GC content: {:.2}%", bio_analysis.gc_content * 100.0);
    
    Ok(())
}
```

### Code Tokenization

```rust
use trustformers_tokenizers::{CodeTokenizer, Language};

fn code_tokenization() -> Result<(), Box<dyn std::error::Error>> {
    let rust_tokenizer = CodeTokenizer::new(Language::Rust);
    let python_tokenizer = CodeTokenizer::new(Language::Python);
    
    // Rust code
    let rust_code = r#"
    fn main() {
        let message = "Hello, world!";
        println!("{}", message);
    }
    "#;
    
    println!("=== Rust Code ===");
    println!("Code: {}", rust_code);
    
    let rust_tokens = rust_tokenizer.tokenize_detailed(rust_code)?;
    for token in rust_tokens {
        println!("{:?}: '{}'", token.token_type, token.text);
    }
    
    // Python code
    let python_code = r#"
    def greet(name):
        message = f"Hello, {name}!"
        print(message)
        return message
    "#;
    
    println!("\n=== Python Code ===");
    println!("Code: {}", python_code);
    
    let python_tokens = python_tokenizer.tokenize_detailed(python_code)?;
    for token in python_tokens {
        println!("{:?}: '{}'", token.token_type, token.text);
    }
    
    Ok(())
}
```

## Performance Optimization

### Memory-Efficient Processing

```rust
use trustformers_tokenizers::{
    TokenizerImpl, CompressedVocab, SharedVocabPool, MemoryOptimizedTokenizer
};
use std::sync::Arc;

fn memory_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // Load base tokenizer
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    
    // Create compressed vocabulary
    let vocab = tokenizer.get_vocab();
    let compressed_vocab = CompressedVocab::from_vocab(vocab)?;
    let compressed_stats = compressed_vocab.memory_stats();
    
    println!("Vocabulary compression:");
    println!("  Original size: {} bytes", compressed_stats.original_size);
    println!("  Compressed size: {} bytes", compressed_stats.compressed_size);
    println!("  Compression ratio: {:.2}x", compressed_stats.compression_ratio);
    
    // Use shared vocabulary pool for multiple tokenizers
    let vocab_pool = SharedVocabPool::new(VocabPoolConfig {
        max_size: 5,
        cleanup_interval: Duration::from_secs(300),
    });
    
    let tokenizer1 = Arc::new(tokenizer.with_shared_vocab_pool(vocab_pool.clone()));
    let tokenizer2 = Arc::new(TokenizerImpl::from_pretrained("distilbert-base-uncased")?
        .with_shared_vocab_pool(vocab_pool.clone()));
    
    // Monitor memory usage
    let pool_stats = vocab_pool.stats();
    println!("\nVocabulary pool statistics:");
    println!("  Active vocabularies: {}", pool_stats.active_count);
    println!("  Memory saved: {} MB", pool_stats.memory_saved_mb);
    println!("  Hit rate: {:.1}%", pool_stats.hit_rate * 100.0);
    
    // Memory-optimized tokenizer
    let optimized_tokenizer = MemoryOptimizedTokenizer::new(tokenizer1.clone())
        .with_string_interning(true)
        .with_lazy_loading(true)
        .with_garbage_collection(true);
    
    let memory_stats = optimized_tokenizer.get_memory_stats();
    println!("\nMemory optimization:");
    println!("  Current usage: {} MB", memory_stats.current_mb);
    println!("  Peak usage: {} MB", memory_stats.peak_mb);
    println!("  String pool savings: {} MB", memory_stats.string_pool_savings_mb);
    
    Ok(())
}
```

### High-Throughput Processing

```rust
use trustformers_tokenizers::{TokenizerImpl, ParallelTokenizer, AsyncTokenizer};
use std::time::Instant;

async fn high_throughput_processing() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    
    // Generate test data
    let test_data: Vec<String> = (0..100_000)
        .map(|i| format!("This is test sentence number {} with some additional text to make it longer.", i))
        .collect();
    
    println!("Processing {} texts...", test_data.len());
    
    // Sequential processing baseline
    let start = Instant::now();
    let mut sequential_results = Vec::new();
    for text in &test_data[..1000] { // Process subset for comparison
        sequential_results.push(tokenizer.encode(text)?);
    }
    let sequential_time = start.elapsed();
    
    println!("Sequential (1K texts): {:?}", sequential_time);
    println!("Sequential rate: {:.0} texts/sec", 1000.0 / sequential_time.as_secs_f64());
    
    // Parallel processing
    let parallel_tokenizer = ParallelTokenizer::new(tokenizer.clone())
        .with_thread_count(8)
        .with_chunk_size(1000);
    
    let start = Instant::now();
    let parallel_results = parallel_tokenizer.encode_batch(&test_data)?;
    let parallel_time = start.elapsed();
    
    println!("Parallel ({}K texts): {:?}", test_data.len() / 1000, parallel_time);
    println!("Parallel rate: {:.0} texts/sec", test_data.len() as f64 / parallel_time.as_secs_f64());
    
    // Async processing
    let async_tokenizer = AsyncTokenizer::from_tokenizer(tokenizer);
    
    let start = Instant::now();
    let futures: Vec<_> = test_data[..10000].iter()
        .map(|text| async_tokenizer.encode_async(text))
        .collect();
    
    let async_results = futures_util::future::join_all(futures).await;
    let async_time = start.elapsed();
    
    println!("Async (10K texts): {:?}", async_time);
    println!("Async rate: {:.0} texts/sec", 10000.0 / async_time.as_secs_f64());
    
    // Calculate speedup
    let parallel_speedup = (sequential_time.as_secs_f64() * test_data.len() as f64) / 
                          (parallel_time.as_secs_f64() * 1000.0);
    println!("Parallel speedup: {:.1}x", parallel_speedup);
    
    Ok(())
}
```

## Custom Tokenization Rules

### Building Custom Rule-Based Tokenizer

```rust
use trustformers_tokenizers::{CustomVocabTokenizer, SpecialTokenManager};
use std::collections::HashMap;

fn custom_rule_tokenizer() -> Result<(), Box<dyn std::error::Error>> {
    // Create custom vocabulary
    let mut vocab = HashMap::new();
    
    // Add basic tokens
    let basic_tokens = vec![
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "hello", "world", "the", "a", "is", "and", "to", "of", "in",
        "good", "bad", "great", "terrible", "awesome", "horrible",
        "!", "?", ".", ",", ";", ":", "(", ")", "[", "]"
    ];
    
    for (i, token) in basic_tokens.iter().enumerate() {
        vocab.insert(token.to_string(), i as u32);
    }
    
    // Build custom tokenizer
    let tokenizer = CustomVocabTokenizer::from_vocab(vocab)?
        .with_unk_token("[UNK]")
        .with_special_tokens(vec!["[PAD]", "[CLS]", "[SEP]", "[MASK]"])
        .with_max_length(128);
    
    // Set up special token manager
    let special_token_manager = SpecialTokenManager::new()
        .add_token("[SENTIMENT_POS]", SpecialTokenType::Custom)
        .add_token("[SENTIMENT_NEG]", SpecialTokenType::Custom)
        .add_template("review", "[CLS] {sentiment} {text} [SEP]");
    
    let enhanced_tokenizer = tokenizer.with_special_token_manager(special_token_manager);
    
    // Test custom tokenization
    let test_texts = vec![
        "Hello world!",
        "This is great and awesome!",
        "That was terrible and horrible.",
        "Unknown words will become [UNK] tokens.",
    ];
    
    for text in test_texts {
        println!("Text: {}", text);
        
        let encoded = enhanced_tokenizer.encode(text)?;
        println!("Tokens: {:?}", encoded.tokens());
        println!("IDs: {:?}", encoded.ids());
        
        // Add sentiment based on content
        let sentiment = if text.contains("great") || text.contains("awesome") {
            "[SENTIMENT_POS]"
        } else if text.contains("terrible") || text.contains("horrible") {
            "[SENTIMENT_NEG]"
        } else {
            ""
        };
        
        if !sentiment.is_empty() {
            let template_vars = vec![
                ("sentiment".to_string(), sentiment.to_string()),
                ("text".to_string(), text.to_string()),
            ];
            
            let formatted = enhanced_tokenizer.format_template("review", &template_vars)?;
            println!("Formatted: {}", formatted);
        }
        
        println!();
    }
    
    Ok(())
}
```

### Domain-Specific Vocabulary Creation

```rust
use trustformers_tokenizers::{TokenizerImpl, VocabAnalyzer, CoverageAnalyzer};
use std::collections::HashMap;

fn domain_vocabulary_creation() -> Result<(), Box<dyn std::error::Error>> {
    // Analyze existing vocabulary for domain coverage
    let base_tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    let vocab_analyzer = VocabAnalyzer::new();
    
    // Medical domain texts for analysis
    let medical_texts = vec![
        "Patient presents with acute myocardial infarction",
        "Administered acetaminophen 500mg every 6 hours",
        "Blood pressure reading: 120/80 mmHg",
        "Diagnosed with hypertension and diabetes mellitus type 2",
        "Recommended follow-up with cardiologist in 2 weeks",
    ];
    
    // Analyze coverage
    let coverage_analyzer = CoverageAnalyzer::new(Default::default());
    let coverage_report = coverage_analyzer.analyze_corpus(&base_tokenizer, &medical_texts)?;
    
    println!("Domain Coverage Analysis:");
    println!("Vocabulary coverage: {:.1}%", coverage_report.vocabulary_coverage.overall * 100.0);
    println!("OOV rate: {:.1}%", coverage_report.vocabulary_coverage.oov_rate * 100.0);
    
    // Identify domain-specific terms that are being split
    for warning in &coverage_report.warnings {
        if warning.severity == CoverageSeverity::High {
            println!("Issue: {}", warning.message);
            for example in &warning.examples {
                println!("  Example: '{}' -> {:?}", example.text, example.tokens);
            }
        }
    }
    
    // Create enhanced vocabulary with domain terms
    let mut enhanced_vocab = base_tokenizer.get_vocab().clone();
    let domain_terms = vec![
        "myocardial", "infarction", "acetaminophen", "mmHg", 
        "hypertension", "diabetes", "mellitus", "cardiologist"
    ];
    
    let mut next_id = enhanced_vocab.len() as u32;
    for term in domain_terms {
        if !enhanced_vocab.contains_token(term) {
            enhanced_vocab.add_token(term.to_string(), next_id);
            next_id += 1;
        }
    }
    
    println!("\nEnhanced vocabulary size: {}", enhanced_vocab.len());
    
    // Test improved tokenization
    for text in &medical_texts {
        let original_encoded = base_tokenizer.encode(text)?;
        println!("\nText: {}", text);
        println!("Original tokens: {:?}", original_encoded.tokens());
        println!("Token count: {}", original_encoded.tokens().len());
    }
    
    Ok(())
}
```

## Integration with ML Pipelines

### PyTorch Integration Example

```rust
#[cfg(feature = "pytorch")]
use trustformers_tokenizers::{PyTorchTokenizer, PyTorchConfig, TensorDType};

#[cfg(feature = "pytorch")]
fn pytorch_integration() -> Result<(), Box<dyn std::error::Error>> {
    let config = PyTorchConfig {
        device: "cuda:0".to_string(),
        dtype: TensorDType::Long,
        return_attention_mask: true,
        return_token_type_ids: true,
        max_length: Some(512),
        padding: Some(PaddingStrategy::MaxLength),
        truncation: Some(TruncationStrategy::LongestFirst),
    };
    
    let tokenizer = PyTorchTokenizer::from_pretrained("bert-base-uncased")?
        .with_config(config);
    
    // Prepare training batch
    let training_texts = vec![
        "This is a positive example.",
        "This is negative.",
        "Another positive sentence with more words.",
        "Short negative.",
    ];
    
    let labels = vec![1, 0, 1, 0]; // 1 = positive, 0 = negative
    
    // Tokenize to PyTorch tensors
    let batch = tokenizer.encode_batch_to_tensors(&training_texts)?;
    
    println!("Batch shape: {:?}", batch.input_ids.shape());
    println!("Attention mask shape: {:?}", batch.attention_mask.shape());
    
    // Example training loop preparation
    for epoch in 0..3 {
        println!("Epoch {}", epoch);
        
        // In real training, you'd iterate through your dataloader
        let input_ids = &batch.input_ids;
        let attention_mask = &batch.attention_mask;
        
        println!("  Processing batch with {} sequences", input_ids.shape()[0]);
        
        // Here you would:
        // 1. Forward pass through model
        // 2. Calculate loss
        // 3. Backward pass
        // 4. Update weights
    }
    
    Ok(())
}
```

### TensorFlow Integration Example

```rust
#[cfg(feature = "tensorflow")]
use trustformers_tokenizers::{TensorFlowTokenizer, TensorFlowConfig};

#[cfg(feature = "tensorflow")]
fn tensorflow_integration() -> Result<(), Box<dyn std::error::Error>> {
    let config = TensorFlowConfig {
        dtype: TfDType::Int32,
        padding_strategy: TfPaddingStrategy::Longest,
        truncation_strategy: TfTruncationStrategy::LongestFirst,
        max_length: Some(256),
        return_ragged: false,
    };
    
    let tokenizer = TensorFlowTokenizer::from_pretrained("bert-base-uncased")?
        .with_config(config);
    
    // Prepare data for tf.data pipeline
    let texts = vec![
        "First example sentence for TensorFlow.",
        "Second example with different length.",
        "Third example that is much longer and contains more words to demonstrate padding behavior.",
    ];
    
    // Convert to TensorFlow tensors
    let tf_dataset = tokenizer.create_tf_dataset(&texts)?;
    
    println!("Created TensorFlow dataset");
    println!("Dataset element spec: {:?}", tf_dataset.element_spec());
    
    // Example tf.data pipeline
    let processed_dataset = tf_dataset
        .batch(2)?
        .prefetch(1)?;
    
    println!("Configured TensorFlow data pipeline");
    
    // Export for TF Serving
    let export_config = TfServingExportConfig {
        model_name: "tokenizer_model".to_string(),
        version: 1,
        signature_name: "serving_default".to_string(),
    };
    
    tokenizer.export_for_serving("./exported_model", export_config)?;
    println!("Exported model for TensorFlow Serving");
    
    Ok(())
}
```

## Advanced Analysis and Debugging

### Tokenization Quality Analysis

```rust
use trustformers_tokenizers::{
    TokenizerImpl, TokenizationDebugger, VocabAnalyzer, PerformanceProfiler
};

fn tokenization_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    
    // Set up debugging and analysis tools
    let debugger = TokenizationDebugger::new();
    let vocab_analyzer = VocabAnalyzer::new();
    let profiler = PerformanceProfiler::new();
    
    let test_texts = vec![
        "Normal sentence for testing.",
        "This sentence has some really-long-hyphenated-words that might cause issues.",
        "Unicode test: café, naïve, résumé, 你好, مرحبا",
        "Numbers and symbols: 123-456-7890, $100.50, 50% discount!",
        "Very long sentence that might exceed typical token limits and could potentially cause truncation issues in downstream applications when processed by transformer models.",
    ];
    
    println!("=== Tokenization Quality Analysis ===\n");
    
    // Debug each text
    for (i, text) in test_texts.iter().enumerate() {
        println!("Text {}: {}", i + 1, text);
        
        // Basic tokenization
        let encoded = tokenizer.encode(text)?;
        println!("Tokens ({}): {:?}", encoded.tokens().len(), encoded.tokens());
        
        // Debug analysis
        let debug_result = debugger.debug_tokenization(&tokenizer, text)?;
        
        if !debug_result.issues.is_empty() {
            println!("Issues detected:");
            for issue in &debug_result.issues {
                println!("  {:?}: {}", issue.issue_type, issue.description);
            }
        }
        
        // Character analysis
        let char_analysis = debug_result.character_analysis;
        if char_analysis.unicode_chars > 0 {
            println!("Unicode characters: {}", char_analysis.unicode_chars);
        }
        if char_analysis.special_chars > 0 {
            println!("Special characters: {}", char_analysis.special_chars);
        }
        
        // Compression statistics
        let compression = debug_result.compression_stats;
        println!("Compression ratio: {:.2}", compression.compression_ratio);
        println!("Character efficiency: {:.2}%", compression.character_efficiency * 100.0);
        
        println!();
    }
    
    // Overall vocabulary analysis
    println!("=== Vocabulary Analysis ===");
    let vocab_analysis = vocab_analyzer.analyze_vocabulary(tokenizer.get_vocab())?;
    
    println!("Total tokens: {}", vocab_analysis.basic_stats.total_tokens);
    println!("Average token length: {:.2}", vocab_analysis.basic_stats.avg_token_length);
    println!("Longest token: {}", vocab_analysis.basic_stats.longest_token);
    
    if !vocab_analysis.issues.is_empty() {
        println!("\nVocabulary issues:");
        for issue in &vocab_analysis.issues {
            println!("  {:?}: {}", issue.issue_type, issue.description);
        }
    }
    
    // Performance profiling
    println!("\n=== Performance Analysis ===");
    let profile_result = profiler.profile_tokenizer(&tokenizer, &test_texts)?;
    
    println!("Average tokenization time: {:.2}ms", profile_result.timing.avg_time_ms);
    println!("Throughput: {:.0} tokens/sec", profile_result.throughput.tokens_per_second);
    println!("Memory usage: {:.1}MB", profile_result.memory.peak_mb);
    
    Ok(())
}
```

### Cross-Tokenizer Comparison

```rust
use trustformers_tokenizers::{TokenizerImpl, BPETokenizer, WordPieceTokenizer, TokenVisualization};

fn cross_tokenizer_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Load different tokenizers
    let bert_tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    let gpt2_tokenizer = TokenizerImpl::from_pretrained("gpt2")?;
    let roberta_tokenizer = TokenizerImpl::from_pretrained("roberta-base")?;
    
    let test_sentences = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Tokenization is the process of breaking text into tokens.",
        "Machine learning models require numerical input representations.",
    ];
    
    println!("=== Cross-Tokenizer Comparison ===\n");
    
    for (i, sentence) in test_sentences.iter().enumerate() {
        println!("Sentence {}: {}", i + 1, sentence);
        println!("{}", "=".repeat(sentence.len() + 15));
        
        // Tokenize with each tokenizer
        let bert_encoded = bert_tokenizer.encode(sentence)?;
        let gpt2_encoded = gpt2_tokenizer.encode(sentence)?;
        let roberta_encoded = roberta_tokenizer.encode(sentence)?;
        
        // Compare results
        println!("BERT    ({}): {:?}", bert_encoded.tokens().len(), bert_encoded.tokens());
        println!("GPT-2   ({}): {:?}", gpt2_encoded.tokens().len(), gpt2_encoded.tokens());
        println!("RoBERTa ({}): {:?}", roberta_encoded.tokens().len(), roberta_encoded.tokens());
        
        // Calculate compression ratios
        let char_count = sentence.len();
        println!("\nCompression ratios (chars/tokens):");
        println!("BERT:    {:.2}", char_count as f32 / bert_encoded.tokens().len() as f32);
        println!("GPT-2:   {:.2}", char_count as f32 / gpt2_encoded.tokens().len() as f32);
        println!("RoBERTa: {:.2}", char_count as f32 / roberta_encoded.tokens().len() as f32);
        
        // Identify differences
        let bert_tokens = bert_encoded.tokens();
        let gpt2_tokens = gpt2_encoded.tokens();
        let roberta_tokens = roberta_encoded.tokens();
        
        if bert_tokens != gpt2_tokens || bert_tokens != roberta_tokens {
            println!("\nDifferences detected:");
            
            // Find unique tokens
            let bert_unique: Vec<&String> = bert_tokens.iter()
                .filter(|t| !gpt2_tokens.contains(t) || !roberta_tokens.contains(t))
                .collect();
            let gpt2_unique: Vec<&String> = gpt2_tokens.iter()
                .filter(|t| !bert_tokens.contains(t) || !roberta_tokens.contains(t))
                .collect();
            let roberta_unique: Vec<&String> = roberta_tokens.iter()
                .filter(|t| !bert_tokens.contains(t) || !gpt2_tokens.contains(t))
                .collect();
            
            if !bert_unique.is_empty() {
                println!("BERT unique: {:?}", bert_unique);
            }
            if !gpt2_unique.is_empty() {
                println!("GPT-2 unique: {:?}", gpt2_unique);
            }
            if !roberta_unique.is_empty() {
                println!("RoBERTa unique: {:?}", roberta_unique);
            }
        }
        
        println!("\n");
    }
    
    Ok(())
}
```

This completes the comprehensive examples documentation covering the most common use cases and patterns for the TrustformeRS Tokenizers library. Each example is practical and can be adapted for specific requirements.