//! Basic Pipeline Usage Example
#![allow(unused_variables)]
//!
//! This example demonstrates the fundamental usage of TrustformeRS pipelines,
//! including text classification, generation, and question answering.

use trustformers::{pipeline, Result};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 TrustformeRS Basic Pipeline Examples\n");

    // Text Classification Example
    text_classification_example().await?;

    // Text Generation Example
    text_generation_example().await?;

    // Question Answering Example
    question_answering_example().await?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Demonstrate text classification pipeline
async fn text_classification_example() -> Result<()> {
    println!("📝 Text Classification Example");
    println!("==============================");

    // Create a text classification pipeline
    let classifier = pipeline(
        "text-classification",
        None, // Use default configuration
        None, // Use default options
    )?;

    // Sample texts to classify
    let texts = vec![
        "I love this new transformer library!",
        "This is the worst software I've ever used.",
        "The weather is nice today.",
        "TrustformeRS makes ML so much easier!",
    ];

    println!("Classifying texts:");
    for text in &texts {
        println!("  Input: \"{}\"", text);
        let result = classifier.__call__(text.to_string())?;
        println!("  Result: {:?}", result);
        println!();
    }

    // Batch processing example
    println!("Batch processing example:");
    let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
    let batch_results = classifier.batch(texts_owned.clone())?;
    for (text, result) in texts_owned.iter().zip(batch_results.iter()) {
        println!("  \"{}\" -> {:?}", text, result);
    }

    println!();
    Ok(())
}

/// Demonstrate text generation pipeline
async fn text_generation_example() -> Result<()> {
    println!("✍️  Text Generation Example");
    println!("===========================");

    // Create a text generation pipeline
    let generator = pipeline("text-generation", None, None)?;

    // Sample prompts
    let prompts = vec![
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "The benefits of renewable energy include",
    ];

    println!("Generating text:");
    for prompt in &prompts {
        println!("  Prompt: \"{}\"", prompt);
        let result = generator.__call__(prompt.to_string())?;
        println!("  Generated: {:?}", result);
        println!();
    }

    Ok(())
}

/// Demonstrate question answering pipeline
async fn question_answering_example() -> Result<()> {
    println!("❓ Question Answering Example");
    println!("=============================");

    // Create a question answering pipeline
    let qa_pipeline = pipeline("question-answering", None, None)?;

    // Context and questions
    let context = "TrustformeRS is a high-performance machine learning library written in Rust. \
                   It provides state-of-the-art transformer models with excellent performance \
                   and memory efficiency. The library supports multiple model architectures \
                   including BERT, GPT, T5, and many others.";

    let questions = vec![
        "What is TrustformeRS?",
        "What programming language is TrustformeRS written in?",
        "What model architectures does it support?",
    ];

    println!("Context: {}", context);
    println!("\nAnswering questions:");

    for question in &questions {
        println!("  Question: \"{}\"", question);

        // Create QA input (implementation depends on the actual QA pipeline structure)
        let qa_input = format!("Question: {} Context: {}", question, context);
        let result = qa_pipeline.__call__(qa_input)?;
        println!("  Answer: {:?}", result);
        println!();
    }

    Ok(())
}

/// Additional utility functions for interactive examples
pub fn print_model_info(model_name: &str) {
    println!("📊 Model Information");
    println!("Model: {}", model_name);
    println!("Framework: TrustformeRS");
    println!("Language: Rust");
    println!();
}

/// Interactive example runner with user input
#[allow(dead_code)]
pub async fn interactive_example() -> Result<()> {
    use std::io::{self, Write};

    println!("🎯 Interactive TrustformeRS Example");
    println!("===================================");

    // Create a text classification pipeline
    let classifier = pipeline("text-classification", None, None)?;

    loop {
        print!("Enter text to classify (or 'quit' to exit): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input == "quit" {
            break;
        }

        if input.is_empty() {
            continue;
        }

        match classifier.__call__(input.to_string()) {
            Ok(result) => println!("Classification result: {:?}\n", result),
            Err(e) => println!("Error: {}\n", e),
        }
    }

    Ok(())
}

/// Performance comparison example
#[allow(dead_code)]
pub async fn performance_comparison_example() -> Result<()> {
    use std::time::Instant;

    println!("⚡ Performance Comparison Example");
    println!("=================================");

    let classifier = pipeline("text-classification", None, None)?;

    let test_texts: Vec<String> =
        (0..100).map(|i| format!("This is test sentence number {}", i)).collect();

    // Single inference benchmark
    println!("Single inference benchmark:");
    let start = Instant::now();
    for text in &test_texts[..10] {
        let _ = classifier.__call__(text.clone())?;
    }
    let single_time = start.elapsed();
    println!("  10 single inferences: {:?}", single_time);

    // Batch inference benchmark
    println!("Batch inference benchmark:");
    let start = Instant::now();
    let _ = classifier.batch(test_texts[..10].to_vec())?;
    let batch_time = start.elapsed();
    println!("  1 batch of 10: {:?}", batch_time);

    println!(
        "  Speedup: {:.2}x",
        single_time.as_nanos() as f64 / batch_time.as_nanos() as f64
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_examples() {
        // Note: These tests would need mock models in a real scenario
        println!("Testing basic pipeline examples...");
        // In a real implementation, we'd test with smaller models or mocks
    }

    #[test]
    fn test_model_info() {
        print_model_info("test-model");
        // Verify the function doesn't panic
    }
}
