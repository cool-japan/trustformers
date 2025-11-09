//! Basic TrustformeRS Client Example
#![allow(unused_variables)]
//!
//! This example demonstrates the basic usage of the TrustformeRS Rust client library,
//! including authentication, health checks, model information, and inference operations.

use std::collections::HashMap;
use std::time::Duration;
use trustformers_client::{
    TrustformersClient, InferenceRequest, InferenceOptions, BatchInferenceRequest, Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("🤖 TrustformeRS Rust Client Demo");
    println!("===============================\n");

    // Configure and build the client
    let client = TrustformersClient::builder("https://api.trustformers.com")
        .api_key("your-api-key-here") // Replace with your actual API key
        .timeout(Duration::from_secs(60))
        .debug(true)
        .max_retries(3)
        .build()?;

    // Alternative authentication methods:
    // .jwt_token("your-jwt-token")
    // .oauth2("client-id", "client-secret", "https://auth.provider.com/token")?

    println!("✅ Client configured successfully\n");

    // 1. Health Check
    println!("🔍 Checking server health...");
    match client.health().await {
        Ok(health) => {
            println!("   Status: {}", health.status);
            println!("   Version: {}", health.version);
            println!("   Uptime: {:.1} seconds", health.uptime);
        }
        Err(e) => {
            println!("   ❌ Health check failed: {}", e);
            // Continue with other operations
        }
    }

    // 2. List Available Models
    println!("\n📚 Listing available models...");
    match client.list_models().await {
        Ok(models) => {
            println!("   Found {} models:", models.len());
            for model in &models[..3.min(models.len())] {
                println!("   • {} - {} ({} parameters)",
                    model.id, model.name, format_number(model.parameters));
            }
            if models.len() > 3 {
                println!("   ... and {} more", models.len() - 3);
            }
        }
        Err(e) => {
            println!("   ❌ Failed to list models: {}", e);
        }
    }

    // 3. Single Inference Request
    println!("\n🧠 Performing single inference...");
    let request = InferenceRequest::new("What are the benefits of using Rust for machine learning?")
        .model_id("llama-3-8b-instruct")
        .options(
            InferenceOptions::new()
                .max_tokens(150)
                .temperature(0.7)
                .top_p(0.9)
        );

    match client.inference(request).await {
        Ok(response) => {
            println!("   ✅ Inference successful!");
            println!("   Model: {}", response.model);
            if let Some(choice) = response.choices.first() {
                println!("   Response: {}", choice.text);
                if let Some(confidence) = choice.confidence {
                    println!("   Confidence: {:.1}%", confidence * 100.0);
                }
            }
            println!("   Tokens used: {} (prompt: {}, completion: {})",
                response.usage.total_tokens,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            );
            println!("   Processing time: {:.1}ms", response.processing_time_ms);
        }
        Err(e) => {
            println!("   ❌ Inference failed: {}", e);
        }
    }

    // 4. Batch Inference
    println!("\n📦 Performing batch inference...");
    let batch_requests = vec![
        InferenceRequest::new("Explain machine learning in one sentence.")
            .model_id("llama-3-8b-instruct")
            .parameter("max_tokens".to_string(), serde_json::Value::Number(50.into())),
        InferenceRequest::new("What is the capital of France?")
            .model_id("llama-3-8b-instruct")
            .parameter("max_tokens".to_string(), serde_json::Value::Number(20.into())),
        InferenceRequest::new("Write a haiku about technology.")
            .model_id("llama-3-8b-instruct")
            .parameter("max_tokens".to_string(), serde_json::Value::Number(100.into())),
    ];

    let batch_request = BatchInferenceRequest::new(batch_requests);

    match client.batch_inference(batch_request).await {
        Ok(batch_response) => {
            println!("   ✅ Batch inference successful!");
            println!("   Processed {} requests", batch_response.batch_size);
            for (i, response) in batch_response.responses.iter().enumerate() {
                if let Some(choice) = response.choices.first() {
                    println!("   Request {}: {}", i + 1, choice.text.chars().take(80).collect::<String>());
                }
            }
        }
        Err(e) => {
            println!("   ❌ Batch inference failed: {}", e);
        }
    }

    // 5. Streaming Inference (if supported)
    println!("\n🌊 Performing streaming inference...");
    let streaming_request = InferenceRequest::new("Tell me a short story about artificial intelligence.")
        .model_id("llama-3-8b-instruct")
        .options(
            InferenceOptions::new()
                .max_tokens(200)
                .temperature(0.8)
                .stream(true)
        );

    match client.stream_inference(streaming_request).await {
        Ok(mut stream) => {
            println!("   🌊 Streaming response:");
            print!("   ");

            use futures_util::StreamExt;
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                print!("{}", content);
                                use std::io::{self, Write};
                                io::stdout().flush().unwrap();
                            }
                        }
                    }
                    Err(e) => {
                        println!("\n   ❌ Streaming error: {}", e);
                        break;
                    }
                }
            }
            println!("\n   ✅ Streaming completed!");
        }
        Err(e) => {
            println!("   ❌ Streaming failed: {}", e);
        }
    }

    // 6. Get Server Metrics
    println!("\n📊 Fetching server metrics...");
    match client.get_metrics().await {
        Ok(metrics) => {
            println!("   ✅ Metrics retrieved:");
            for (key, value) in metrics.iter().take(5) {
                println!("   • {}: {}", key, value);
            }
            if metrics.len() > 5 {
                println!("   ... and {} more metrics", metrics.len() - 5);
            }
        }
        Err(e) => {
            println!("   ❌ Failed to get metrics: {}", e);
        }
    }

    // 7. Model Information
    println!("\n🔍 Getting model information...");
    match client.get_model("llama-3-8b-instruct").await {
        Ok(model) => {
            println!("   ✅ Model information:");
            println!("   • Name: {}", model.name);
            println!("   • Version: {}", model.version);
            println!("   • Architecture: {}", model.architecture);
            println!("   • Parameters: {}", format_number(model.parameters));
            println!("   • Status: {}", model.status);
            println!("   • Description: {}", model.description);
        }
        Err(e) => {
            println!("   ❌ Failed to get model info: {}", e);
        }
    }

    println!("\n🎉 Demo completed successfully!");
    println!("   The TrustformeRS Rust client provides:");
    println!("   • Type-safe API with comprehensive error handling");
    println!("   • Async/await support for high-performance operations");
    println!("   • Multiple authentication methods (API key, JWT, OAuth2)");
    println!("   • Streaming support for real-time inference");
    println!("   • Batch processing for efficient multi-request handling");
    println!("   • Retry logic with exponential backoff");
    println!("   • Comprehensive request/response logging and debugging");

    Ok(())
}

/// Format large numbers with appropriate suffixes
fn format_number(num: u64) -> String {
    if num >= 1_000_000_000 {
        format!("{:.1}B", num as f64 / 1_000_000_000.0)
    } else if num >= 1_000_000 {
        format!("{:.1}M", num as f64 / 1_000_000.0)
    } else if num >= 1_000 {
        format!("{:.1}K", num as f64 / 1_000.0)
    } else {
        num.to_string()
    }
}