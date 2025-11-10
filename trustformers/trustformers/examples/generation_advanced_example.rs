// SPDX-License-Identifier: Apache-2.0

//! # Advanced Text Generation Example
//!
//! This example demonstrates advanced text generation capabilities with TrustformeRS.
//! It showcases various sampling strategies, configuration options, and real-world use cases.
//!
//! ## Features Demonstrated
//!
//! - Greedy decoding for deterministic generation
//! - Beam search for high-quality outputs
//! - Top-k sampling for controlled randomness
//! - Top-p (nucleus) sampling for diverse outputs
//! - Min-p sampling for probability filtering
//! - Temperature-based sampling
//! - Combined sampling strategies
//! - Repetition penalty and frequency/presence penalties
//! - Bad words filtering
//! - Early stopping strategies
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example generation_advanced_example --features "gpt2"
//! ```

use anyhow::{Context, Result};

/// Print a nicely formatted section header
fn print_section(title: &str) {
    println!("\n{}", "█".repeat(80));
    println!("{}", title);
    println!("{}", "█".repeat(80));
}

/// Print a subsection header
fn print_subsection(title: &str) {
    println!("\n{}", "-".repeat(80));
    println!("{}", title);
    println!("{}", "-".repeat(80));
}

/// Example 1: Greedy Decoding
fn example_greedy_generation() -> Result<()> {
    print_section("Example 1: Greedy Decoding");

    println!("\nGreedy decoding always selects the most likely next token.");
    println!("This produces deterministic, focused outputs but may lack creativity.");

    let prompt = "Once upon a time in a land far away";
    let max_length = 50;

    print_subsection("Configuration");
    println!("Prompt:      {}", prompt);
    println!("Max length:  {}", max_length);
    println!("Strategy:    Greedy");

    print_subsection("Generated Text");
    // In a real implementation, you would:
    // let config = GenerationConfig::greedy();
    // config.max_length = max_length;
    // let output = model.generate_with_config(tokenize(prompt), config)?;
    
    println!("{}", prompt);
    println!("... [generated text would appear here]");

    Ok(())
}

/// Example 2: Beam Search
fn example_beam_search_generation() -> Result<()> {
    print_section("Example 2: Beam Search");

    println!("\nBeam search maintains multiple hypotheses and selects the best overall sequence.");
    println!("This produces high-quality, coherent outputs.");

    let prompt = "The future of artificial intelligence includes";
    let max_length = 60;
    let num_beams = 5;
    let num_return_sequences = 3;

    print_subsection("Configuration");
    println!("Prompt:               {}", prompt);
    println!("Max length:           {}", max_length);
    println!("Number of beams:      {}", num_beams);
    println!("Return sequences:     {}", num_return_sequences);
    println!("Length penalty:       1.0 (no penalty)");

    print_subsection("Generated Sequences");
    // In a real implementation:
    // let config = GenerationConfig::beam_search(num_beams);
    // config.max_length = max_length;
    // config.num_return_sequences = num_return_sequences;
    // let outputs = model.generate_with_config(tokenize(prompt), config)?;

    for i in 1..=num_return_sequences {
        println!("\nSequence {}:", i);
        println!("{}", prompt);
        println!("... [beam search output {} would appear here]", i);
    }

    Ok(())
}

/// Example 3: Top-k Sampling
fn example_top_k_sampling() -> Result<()> {
    print_section("Example 3: Top-k Sampling");

    println!("\nTop-k sampling selects from the k most likely next tokens.");
    println!("This adds controlled randomness while avoiding low-probability tokens.");

    let prompt = "In a surprising turn of events";
    let max_length = 40;
    let k = 50;
    let temperature = 0.8;
    let num_samples = 3;

    print_subsection("Configuration");
    println!("Prompt:       {}", prompt);
    println!("Max length:   {}", max_length);
    println!("Top-k:        {}", k);
    println!("Temperature:  {}", temperature);
    println!("Samples:      {}", num_samples);

    print_subsection("Generated Samples");
    // In a real implementation:
    // let config = GenerationConfig::top_k(k);
    // config.max_length = max_length;
    // config.temperature = temperature;

    for i in 1..=num_samples {
        println!("\nSample {}:", i);
        println!("{}", prompt);
        println!("... [top-k sample {} would appear here]", i);
    }

    Ok(())
}

/// Example 4: Top-p (Nucleus) Sampling
fn example_top_p_sampling() -> Result<()> {
    print_section("Example 4: Top-p (Nucleus) Sampling");

    println!("\nTop-p sampling selects from the smallest set of tokens whose cumulative");
    println!("probability exceeds p. This adapts the sampling pool size dynamically.");

    let prompt = "The scientific breakthrough revealed";
    let max_length = 50;
    let p = 0.9;
    let temperature = 0.9;
    let num_samples = 3;

    print_subsection("Configuration");
    println!("Prompt:       {}", prompt);
    println!("Max length:   {}", max_length);
    println!("Top-p:        {}", p);
    println!("Temperature:  {}", temperature);
    println!("Samples:      {}", num_samples);

    print_subsection("Generated Samples");
    for i in 1..=num_samples {
        println!("\nSample {}:", i);
        println!("{}", prompt);
        println!("... [nucleus sample {} would appear here]", i);
    }

    Ok(())
}

/// Example 5: Temperature Scaling
fn example_temperature_effects() -> Result<()> {
    print_section("Example 5: Temperature Effects");

    println!("\nTemperature controls the randomness of predictions:");
    println!("  - Low (0.1-0.5):  More focused and deterministic");
    println!("  - Medium (0.7-0.9): Balanced creativity");
    println!("  - High (1.0-2.0):   More diverse and creative");

    let prompt = "To solve this problem, we should";
    let max_length = 40;
    let temperatures = vec![0.3, 0.7, 1.2];

    for temp in temperatures {
        print_subsection(&format!("Temperature: {}", temp));
        println!("Prompt:       {}", prompt);
        println!("Max length:   {}", max_length);
        println!("Temperature:  {}", temp);
        println!("\nGenerated:");
        println!("{}", prompt);
        println!("... [output with temperature {} would appear here]", temp);
    }

    Ok(())
}

/// Example 6: Repetition Penalty
fn example_repetition_penalty() -> Result<()> {
    print_section("Example 6: Repetition Penalty");

    println!("\nRepetition penalty discourages repeating tokens, making outputs more diverse.");
    println!("Values > 1.0 penalize repetition, values < 1.0 encourage it.");

    let prompt = "The main advantages are";
    let max_length = 60;
    let repetition_penalties = vec![1.0, 1.2, 1.5];

    for penalty in repetition_penalties {
        print_subsection(&format!("Repetition Penalty: {}", penalty));
        println!("Prompt:               {}", prompt);
        println!("Max length:           {}", max_length);
        println!("Repetition penalty:   {}", penalty);
        println!("\nGenerated:");
        println!("{}", prompt);
        println!("... [output with penalty {} would appear here]", penalty);
    }

    Ok(())
}

/// Example 7: Frequency and Presence Penalties
fn example_frequency_presence_penalties() -> Result<()> {
    print_section("Example 7: Frequency and Presence Penalties");

    println!("\nFrequency penalty: Reduces probability based on how often a token has appeared.");
    println!("Presence penalty: Reduces probability if a token has appeared at all.");

    let prompt = "Key factors to consider include";
    let max_length = 50;

    print_subsection("No Penalties");
    println!("Prompt:       {}", prompt);
    println!("\nGenerated:");
    println!("{}", prompt);
    println!("... [baseline output]");

    print_subsection("With Frequency Penalty");
    println!("Frequency penalty: 0.3");
    println!("\nGenerated:");
    println!("{}", prompt);
    println!("... [output with frequency penalty]");

    print_subsection("With Presence Penalty");
    println!("Presence penalty: 0.3");
    println!("\nGenerated:");
    println!("{}", prompt);
    println!("... [output with presence penalty]");

    Ok(())
}

/// Example 8: Combined Sampling Strategies
fn example_combined_sampling() -> Result<()> {
    print_section("Example 8: Combined Sampling Strategies");

    println!("\nCombining top-k and top-p sampling provides fine-grained control:");
    println!("First apply top-k to limit choices, then top-p for dynamic thresholding.");

    let prompt = "In the near future, technology will enable us to";
    let max_length = 50;
    let k = 100;
    let p = 0.9;
    let temperature = 0.8;

    print_subsection("Configuration");
    println!("Prompt:       {}", prompt);
    println!("Max length:   {}", max_length);
    println!("Top-k:        {}", k);
    println!("Top-p:        {}", p);
    println!("Temperature:  {}", temperature);

    print_subsection("Generated Text");
    println!("{}", prompt);
    println!("... [combined sampling output would appear here]");

    Ok(())
}

/// Example 9: Bad Words Filtering
fn example_bad_words_filtering() -> Result<()> {
    print_section("Example 9: Bad Words Filtering");

    println!("\nBad words filtering prevents specific tokens from being generated.");
    println!("Useful for content moderation and controlling output vocabulary.");

    let prompt = "The controversial topic of";
    let max_length = 40;
    let bad_words = vec!["controversial", "debate", "argument"];

    print_subsection("Configuration");
    println!("Prompt:       {}", prompt);
    println!("Max length:   {}", max_length);
    println!("Bad words:    {:?}", bad_words);

    print_subsection("Generated Text");
    println!("{}", prompt);
    println!("... [output without bad words would appear here]");

    Ok(())
}

/// Example 10: Early Stopping Strategies
fn example_early_stopping() -> Result<()> {
    print_section("Example 10: Early Stopping Strategies");

    println!("\nEarly stopping ends generation when specific conditions are met:");
    println!("  - EOS token encountered");
    println!("  - Maximum length reached");
    println!("  - Minimum quality threshold not met (beam search)");

    let prompt = "The story concludes";
    let max_length = 100;

    print_subsection("Configuration");
    println!("Prompt:         {}", prompt);
    println!("Max length:     {}", max_length);
    println!("Early stopping: Enabled");
    println!("EOS tokens:     ['.', '!', '?', '<|endoftext|>']");

    print_subsection("Generated Text");
    println!("{}", prompt);
    println!("... [output with early stopping would appear here]");
    println!("\n[Generation stopped at EOS token]");

    Ok(())
}

/// Example 11: Min-p Sampling
fn example_min_p_sampling() -> Result<()> {
    print_section("Example 11: Min-p Sampling");

    println!("\nMin-p sampling filters out tokens below a probability threshold.");
    println!("This provides more stable sampling than top-p for some distributions.");

    let prompt = "Recent developments in quantum computing";
    let max_length = 45;
    let min_p = 0.05;
    let temperature = 0.85;
    let num_samples = 3;

    print_subsection("Configuration");
    println!("Prompt:       {}", prompt);
    println!("Max length:   {}", max_length);
    println!("Min-p:        {}", min_p);
    println!("Temperature:  {}", temperature);
    println!("Samples:      {}", num_samples);

    print_subsection("Generated Samples");
    for i in 1..=num_samples {
        println!("\nSample {}:", i);
        println!("{}", prompt);
        println!("... [min-p sample {} would appear here]", i);
    }

    Ok(())
}

/// Example 12: Long-form Generation with KV Cache
fn example_long_form_generation() -> Result<()> {
    print_section("Example 12: Long-form Generation with KV Cache");

    println!("\nKV (key-value) caching significantly speeds up long-form generation");
    println!("by avoiding redundant computations for previously processed tokens.");

    let prompt = "Chapter 1: The Beginning\n\nIt was a dark and stormy night when";
    let max_length = 200;

    print_subsection("Configuration");
    println!("Prompt:       {} [...]", &prompt[..50]);
    println!("Max length:   {}", max_length);
    println!("KV cache:     Enabled");
    println!("Temperature:  0.8");
    println!("Top-p:        0.9");

    print_subsection("Generation Progress");
    println!("Starting generation...");
    
    // Simulate progress updates
    let milestones = vec![25, 50, 75, 100, 150, 200];
    for milestone in milestones {
        println!("  {} tokens generated...", milestone);
    }

    print_subsection("Generated Text");
    println!("{}", prompt);
    println!("... [long-form output would appear here]");

    Ok(())
}

fn main() -> Result<()> {
    println!("\n{}", "▓".repeat(80));
    println!("TrustformeRS - Advanced Text Generation Examples");
    println!("{}", "▓".repeat(80));
    println!("\nThis example demonstrates various text generation strategies and configurations.");
    println!("Note: Actual model inference requires loading pre-trained weights.");

    // Run all examples
    example_greedy_generation().context("Greedy generation example failed")?;
    example_beam_search_generation().context("Beam search example failed")?;
    example_top_k_sampling().context("Top-k sampling example failed")?;
    example_top_p_sampling().context("Top-p sampling example failed")?;
    example_temperature_effects().context("Temperature example failed")?;
    example_repetition_penalty().context("Repetition penalty example failed")?;
    example_frequency_presence_penalties()
        .context("Frequency/presence penalty example failed")?;
    example_combined_sampling().context("Combined sampling example failed")?;
    example_bad_words_filtering().context("Bad words filtering example failed")?;
    example_early_stopping().context("Early stopping example failed")?;
    example_min_p_sampling().context("Min-p sampling example failed")?;
    example_long_form_generation().context("Long-form generation example failed")?;

    println!("\n{}", "▓".repeat(80));
    println!("All advanced generation examples completed successfully!");
    println!("{}", "▓".repeat(80));
    println!("\nFor production usage:");
    println!("  1. Load a pre-trained GPT-2 model using AutoModel");
    println!("  2. Use trustformers_tokenizers to tokenize input text");
    println!("  3. Configure GenerationConfig with desired parameters");
    println!("  4. Call model.generate_with_config() for inference");
    println!("\nSee trustformers-models/src/gpt2/generation.rs for implementation details.");
    println!();

    Ok(())
}
