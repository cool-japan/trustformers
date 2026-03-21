use trustformers::prelude::*;
#[allow(unused_variables)]
use trustformers::{BertConfig, BertModel, TokenizedInput};

fn main() -> Result<()> {
    println!("TrustformeRS Basic BERT Example");
    println!("================================\n");

    // Create a BERT configuration
    let config = BertConfig {
        vocab_size: 30522,
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        ..Default::default()
    };

    println!("Creating BERT model with configuration:");
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Layers: {}", config.num_hidden_layers);
    println!("  Attention heads: {}", config.num_attention_heads);
    println!();

    // Create the model
    let model = BertModel::new(config)?;
    println!("✓ Model created successfully");

    // Create dummy input
    let input = TokenizedInput {
        input_ids: vec![101, 7592, 1010, 2088, 999, 102], // [CLS] Hello, world! [SEP]
        attention_mask: vec![1, 1, 1, 1, 1, 1],
        token_type_ids: Some(vec![0, 0, 0, 0, 0, 0]),
        special_tokens_mask: Some(vec![1, 0, 0, 0, 0, 1]), // [CLS] and [SEP] are special tokens
        offset_mapping: None,
        overflowing_tokens: None,
    };

    println!("\nInput tokens: {:?}", input.input_ids);
    println!("Attention mask: {:?}", input.attention_mask);

    // Run forward pass
    println!("\nRunning forward pass...");
    let output = model.forward(input)?;

    println!("✓ Forward pass completed");
    println!("\nOutput shape: {:?}", output.last_hidden_state.shape());

    if let Some(pooler_output) = output.pooler_output {
        println!("Pooler output shape: {:?}", pooler_output.shape());
    }

    Ok(())
}
