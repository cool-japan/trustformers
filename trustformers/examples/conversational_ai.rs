//! Conversational AI Example
//!
//! Demonstrates the conversational pipeline against a **real** checkpoint:
//! multi-turn dialogue, persona configuration, memory/summarization settings,
//! safety filtering and token-level streaming.
//!
//! The pipeline runs an actual model, so this example needs one. Point it at a
//! local checkpoint directory that contains `config.json`, a weight file
//! (`model.safetensors` or `pytorch_model.bin`) and a tokenizer:
//!
//! ```text
//! TRUSTFORMERS_EXAMPLE_MODEL=/path/to/gpt2 cargo run --example conversational_ai --features gpt2
//! # or
//! cargo run --example conversational_ai --features gpt2 -- /path/to/gpt2
//! ```
//!
//! Without a checkpoint the example explains what it needs and exits — it does
//! not fabricate a conversation.

use std::env;

use futures::StreamExt;
use trustformers::pipeline::conversational::{
    ConversationMode, ConversationalConfig, ConversationalInput, ConversationalPipeline,
    MemoryConfig, PersonaConfig, RepairConfig, StreamingConfig, SummarizationConfig,
};
use trustformers::{AutoModel, AutoTokenizer};
use trustformers_core::generation::GenerationConfig;

/// Environment variable naming the checkpoint directory.
const MODEL_ENV: &str = "TRUSTFORMERS_EXAMPLE_MODEL";

/// Boxed-error result, so the example never returns a large error by value.
type ExampleResult = std::result::Result<(), Box<dyn std::error::Error>>;

#[tokio::main]
async fn main() -> ExampleResult {
    println!("TrustformeRS Conversational AI Example\n");

    let Some(model_path) = resolve_model_path() else {
        println!("No checkpoint was provided, so there is nothing to converse with.\n");
        println!("Set {MODEL_ENV} (or pass a path as the first argument) to a directory that");
        println!("contains config.json, a weight file and a tokenizer, for example:\n");
        println!("  {MODEL_ENV}=/path/to/gpt2 cargo run --example conversational_ai\n");
        return Ok(());
    };

    println!("Loading checkpoint from {model_path}");
    let model = AutoModel::from_pretrained(&model_path)?;
    let tokenizer = AutoTokenizer::from_pretrained(&model_path)?;
    let pipeline = ConversationalPipeline::new(model, tokenizer)?.with_config(chat_config());
    println!("Loaded.\n");

    multi_turn_dialogue(&pipeline).await?;
    persona_conversation(&model_path).await?;
    safety_filtering(&model_path).await?;
    streaming_response(&pipeline).await?;

    println!("\nAll conversational examples completed.");
    Ok(())
}

/// Resolve the checkpoint directory from the environment or the command line.
fn resolve_model_path() -> Option<String> {
    if let Ok(path) = env::var(MODEL_ENV) {
        if !path.trim().is_empty() {
            return Some(path);
        }
    }
    env::args().nth(1)
}

/// A plain chat configuration.
fn chat_config() -> ConversationalConfig {
    ConversationalConfig {
        max_history_turns: 10,
        max_context_tokens: 1024,
        enable_summarization: true,
        temperature: 0.7,
        top_p: 0.9,
        top_k: Some(50),
        max_response_tokens: 64,
        system_prompt: Some("You are a helpful AI assistant.".to_string()),
        enable_safety_filter: true,
        conversation_mode: ConversationMode::Chat,
        enable_persistence: false,
        persona: None,
        summarization_config: SummarizationConfig::default(),
        memory_config: MemoryConfig::default(),
        generation_config: GenerationConfig::default(),
        repair_config: RepairConfig::default(),
        streaming_config: StreamingConfig::default(),
    }
}

/// Several turns sharing one conversation id.
async fn multi_turn_dialogue(
    pipeline: &ConversationalPipeline<AutoModel, AutoTokenizer>,
) -> ExampleResult {
    println!("Multi-turn dialogue");
    println!("-------------------");

    let mut conversation_id: Option<String> = None;
    for message in [
        "Hello! What can you help me with?",
        "Tell me about state space models.",
        "How do they compare to attention?",
    ] {
        let output = pipeline
            .process_conversation(ConversationalInput {
                message: message.to_string(),
                conversation_id: conversation_id.clone(),
                context: None,
                config_override: None,
            })
            .await?;

        println!("  user      : {message}");
        println!("  assistant : {}", output.response.trim());
        println!(
            "  turns={} tokens={}",
            output.conversation_state.turns.len(),
            output.generation_stats.tokens_generated
        );
        conversation_id = Some(output.conversation_id);
    }
    println!();
    Ok(())
}

/// The same model driven with a persona.
async fn persona_conversation(model_path: &str) -> ExampleResult {
    println!("Persona-based conversation");
    println!("-------------------------");

    let mut config = chat_config();
    config.persona = Some(PersonaConfig {
        name: "Ada".to_string(),
        personality: "Concise and technically precise".to_string(),
        background: "An assistant focused on mathematics and computing".to_string(),
        speaking_style: "direct".to_string(),
        expertise: vec!["mathematics".to_string(), "computing".to_string()],
        constraints: vec!["Prefer short answers".to_string()],
    });

    let model = AutoModel::from_pretrained(model_path)?;
    let tokenizer = AutoTokenizer::from_pretrained(model_path)?;
    let pipeline = ConversationalPipeline::new(model, tokenizer)?.with_config(config);

    let output = pipeline
        .process_conversation(ConversationalInput {
            message: "Summarise what a transformer block does.".to_string(),
            conversation_id: None,
            context: None,
            config_override: None,
        })
        .await?;
    println!("  assistant : {}", output.response.trim());
    println!();
    Ok(())
}

/// Safety filtering applied to the generated response.
async fn safety_filtering(model_path: &str) -> ExampleResult {
    println!("Safety filtering");
    println!("----------------");

    let model = AutoModel::from_pretrained(model_path)?;
    let tokenizer = AutoTokenizer::from_pretrained(model_path)?;
    let pipeline = ConversationalPipeline::new(model, tokenizer)?
        .with_config(chat_config())
        .with_safety_filter(true);

    let output = pipeline
        .process_conversation(ConversationalInput {
            message: "Give me a friendly greeting.".to_string(),
            conversation_id: None,
            context: None,
            config_override: None,
        })
        .await?;

    println!("  assistant : {}", output.response.trim());
    println!("  safety filter: enabled");
    println!();
    Ok(())
}

/// Token-level streaming: chunks arrive as the model produces them.
async fn streaming_response(
    pipeline: &ConversationalPipeline<AutoModel, AutoTokenizer>,
) -> ExampleResult {
    println!("Streaming response");
    println!("------------------");

    let mut stream = pipeline
        .generate_streaming_response(ConversationalInput {
            message: "Count to five.".to_string(),
            conversation_id: None,
            context: None,
            config_override: None,
        })
        .await?;

    print!("  assistant : ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => print!("{text}"),
            Err(error) => {
                println!("\n  streaming failed: {error}");
                break;
            },
        }
    }
    println!("\n");
    Ok(())
}
