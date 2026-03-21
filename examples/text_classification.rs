//! Example: Text Classification with BERT
#![allow(unused_variables)]
//!
//! This example demonstrates how to use a pre-trained BERT model
//! for sentiment analysis using the pipeline API.

use trustformers::prelude::*;
use trustformers::pipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("TrustformeRS Text Classification Example");
    println!("========================================");

    // Create a text classification pipeline
    // This would normally download the model from Hugging Face Hub
    let classifier = pipeline("text-classification", Some("bert-base-uncased"), None)?;

    // Example texts to classify
    let texts = vec![
        "I love this movie! It's absolutely fantastic.",
        "This is terrible. I hate it.",
        "It's okay, nothing special but not bad either.",
        "The weather is nice today.",
    ];

    // Classify each text
    for text in texts {
        println!("\nText: \"{}\"", text);

        match classifier.__call__(text.to_string()) {
            Ok(results) => {
                println!("Results:");
                for result in results {
                    println!("  - Label: {}, Score: {:.4}", result.label, result.score);
                }
            }
            Err(e) => {
                println!("  Error: {}", e);
            }
        }
    }

    Ok(())
}

// Note: This example currently returns an error because the pipeline
// implementation is not complete. Once the models can be loaded from
// the Hugging Face Hub, this will work properly.