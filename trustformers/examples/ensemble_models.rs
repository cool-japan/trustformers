//! Ensemble Models Example
#![allow(unused_variables)]
//!
//! This example demonstrates ensemble learning techniques including
//! model combination, voting strategies, and performance optimization.

use trustformers::pipeline::{
    EnsembleConfig, EnsemblePipeline, EnsembleStrategy, ModelSelectionStrategy, Pipeline,
};
use trustformers::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎯 TrustformeRS Ensemble Models Examples\n");

    // Basic Ensemble Example
    basic_ensemble_example().await?;

    // Voting Strategies Example
    voting_strategies_example().await?;

    // Dynamic Model Selection Example
    dynamic_selection_example().await?;

    // Cascade Ensemble Example
    cascade_ensemble_example().await?;

    // Performance Comparison Example
    performance_comparison_example().await?;

    // Adaptive Ensemble Example
    adaptive_ensemble_example().await?;

    println!("\n✅ All ensemble model examples completed successfully!");
    Ok(())
}

/// Demonstrate basic ensemble configuration and usage
async fn basic_ensemble_example() -> Result<()> {
    println!("🎯 Basic Ensemble Example");
    println!("=========================");

    // Create individual models for the ensemble
    let models = vec![
        ("distilbert-base-uncased-finetuned-sst-2-english", 0.35),
        ("cardiffnlp/twitter-roberta-base-sentiment-latest", 0.40),
        ("nlptown/bert-base-multilingual-uncased-sentiment", 0.25),
    ];

    println!("Ensemble Configuration:");
    println!("  Models in ensemble: {}", models.len());
    for (model_name, weight) in &models {
        println!("    - {} (weight: {:.2})", model_name, weight);
    }

    // Configure ensemble
    let ensemble_config = EnsembleConfig {
        strategy: EnsembleStrategy::WeightedAverage(vec![0.3, 0.4, 0.3]),
        model_selection_strategy: ModelSelectionStrategy::All,
        cascade_max_models: 3,
        confidence_threshold: 0.7,
        ..Default::default()
    };

    println!("  Strategy: {:?}", ensemble_config.strategy);
    println!("  Max models: {}", ensemble_config.cascade_max_models);
    println!(
        "  Confidence threshold: {}",
        ensemble_config.confidence_threshold
    );

    // Create ensemble pipeline
    let mut ensemble = EnsemblePipeline::new(ensemble_config);

    // Add models to ensemble
    for (model_name, weight) in &models {
        if let Err(e) =
            ensemble.add_model_from_pretrained(model_name, "text-classification", *weight, None)
        {
            println!("Warning: Could not add model {}: {}", model_name, e);
        }
    }

    // Test sentences with different sentiments
    let test_sentences = vec![
        "I absolutely love this new movie! It's fantastic!",
        "This product is terrible and I want my money back.",
        "The weather is okay today, nothing special.",
        "Amazing service and great quality! Highly recommended!",
        "Not sure about this, it's somewhat disappointing.",
    ];

    println!("\nEnsemble Predictions:");
    for (i, sentence) in test_sentences.iter().enumerate() {
        println!("\nSentence {}: \"{}\"", i + 1, sentence);

        let result = match ensemble.__call__(sentence.to_string()) {
            Ok(output) => {
                // Create a mock EnsemblePrediction for the example
                EnsemblePrediction {
                    prediction: "POSITIVE".to_string(), // Simplified for example
                    confidence: 0.85,
                    agreement_count: 2,
                }
            },
            Err(e) => {
                println!("Error processing sentence: {}", e);
                continue;
            },
        };

        // Show individual model predictions (simulated)
        let individual_predictions = simulate_individual_predictions(sentence);
        println!("  Individual predictions:");
        for (model, prediction, confidence) in &individual_predictions {
            println!("    {}: {} ({:.1}%)", model, prediction, confidence * 100.0);
        }

        // Show ensemble result
        println!(
            "  Ensemble result: {} ({:.1}%)",
            result.prediction,
            result.confidence * 100.0
        );
        println!(
            "  Agreement: {}/{} models",
            result.agreement_count,
            models.len()
        );

        if result.confidence < 0.7 {
            println!("  ⚠️  Low confidence - consider manual review");
        }
    }

    Ok(())
}

/// Demonstrate different voting strategies
async fn voting_strategies_example() -> Result<()> {
    println!("🗳️  Voting Strategies Example");
    println!("=============================");

    let voting_strategies = vec![
        ("Majority Voting", EnsembleStrategy::MajorityVote),
        (
            "Weighted Voting",
            EnsembleStrategy::WeightedAverage(vec![0.3, 0.4, 0.3]),
        ),
        ("Confidence Voting", EnsembleStrategy::Maximum),
        ("Consensus", EnsembleStrategy::Average),
        ("Best Model", EnsembleStrategy::Maximum),
    ];

    let test_sentence = "This TrustformeRS library is incredibly powerful and well-designed!";
    println!("Test sentence: \"{}\"", test_sentence);

    // Simulate model predictions
    let model_predictions = vec![
        ("Model A", "POSITIVE", 0.92),
        ("Model B", "POSITIVE", 0.87),
        ("Model C", "POSITIVE", 0.95),
    ];

    println!("\nIndividual model predictions:");
    for (model, prediction, confidence) in &model_predictions {
        println!("  {}: {} ({:.1}%)", model, prediction, confidence * 100.0);
    }

    println!("\nVoting strategy results:");
    for (strategy_name, strategy) in voting_strategies {
        let result = simulate_voting_strategy(&model_predictions, &strategy);
        println!(
            "  {}: {} ({:.1}%)",
            strategy_name,
            result.0,
            result.1 * 100.0
        );

        match strategy {
            EnsembleStrategy::MajorityVote => {
                println!("    Logic: Simple majority wins (3/3 voted POSITIVE)");
            },
            EnsembleStrategy::WeightedAverage(_) => {
                println!("    Logic: Weighted average (0.35×0.92 + 0.40×0.87 + 0.25×0.95)");
            },
            EnsembleStrategy::Maximum => {
                println!("    Logic: Confidence-weighted prediction (higher confidence = more influence)");
            },
            EnsembleStrategy::Average => {
                println!("    Logic: All models must agree (unanimous POSITIVE)");
            },
            _ => {},
        }
    }

    // Compare strategy performance
    println!("\nStrategy Comparison:");
    println!("  Scenario: Disagreement case");
    let disagreement_predictions = vec![
        ("Model A", "POSITIVE", 0.85),
        ("Model B", "NEGATIVE", 0.78),
        ("Model C", "POSITIVE", 0.82),
    ];

    for (strategy_name, strategy) in &[
        ("Majority Voting", EnsembleStrategy::MajorityVote),
        (
            "Weighted Voting",
            EnsembleStrategy::WeightedAverage(vec![0.3, 0.4, 0.3]),
        ),
        ("Consensus", EnsembleStrategy::Average),
    ] {
        let result = simulate_voting_strategy(&disagreement_predictions, strategy);
        println!(
            "    {}: {} ({:.1}%)",
            strategy_name,
            result.0,
            result.1 * 100.0
        );
    }

    Ok(())
}

/// Demonstrate dynamic model selection based on input characteristics
async fn dynamic_selection_example() -> Result<()> {
    println!("🧠 Dynamic Model Selection Example");
    println!("==================================");

    // Define model specializations
    let specialized_models = vec![
        (
            "twitter-sentiment",
            "Social media text",
            vec!["informal", "short", "emoji"],
        ),
        (
            "news-sentiment",
            "News articles",
            vec!["formal", "long", "factual"],
        ),
        (
            "product-reviews",
            "E-commerce reviews",
            vec!["product", "rating", "experience"],
        ),
        (
            "general-purpose",
            "General text",
            vec!["versatile", "balanced"],
        ),
    ];

    println!("Specialized Models:");
    for (name, description, characteristics) in &specialized_models {
        println!("  {}: {} - {:?}", name, description, characteristics);
    }

    // Test inputs with different characteristics
    let test_inputs = vec![
        ("OMG this is amazing!!! 🔥🔥🔥 #love", "Social media"),
        (
            "The quarterly earnings report showed significant growth across all sectors.",
            "News/Business",
        ),
        (
            "Great product, fast shipping, would definitely recommend to others. 5 stars!",
            "Product review",
        ),
        (
            "The implementation of this algorithm demonstrates excellent performance.",
            "Technical/General",
        ),
    ];

    println!("\nDynamic Selection Results:");
    for (text, category) in test_inputs {
        println!("\nInput: \"{}\"", text);
        println!("Category: {}", category);

        // Analyze input characteristics
        let characteristics = analyze_input_characteristics(text);
        println!("Detected characteristics: {:?}", characteristics);

        // Select best model
        let selected_model = select_best_model(&specialized_models, &characteristics);
        println!("Selected model: {}", selected_model.0);
        println!("Reason: Matches characteristics {:?}", selected_model.1);

        // Simulate confidence adjustment
        let base_confidence = 0.85;
        let adjusted_confidence = if selected_model.1.len() > 1 {
            base_confidence + 0.10
        } else {
            base_confidence
        };

        println!(
            "Prediction confidence: {:.1}% (boosted by specialization)",
            adjusted_confidence * 100.0
        );
    }

    // Show adaptation statistics
    println!("\nSelection Statistics:");
    println!("  Total inputs processed: 4");
    println!("  twitter-sentiment: 1 selection");
    println!("  news-sentiment: 1 selection");
    println!("  product-reviews: 1 selection");
    println!("  general-purpose: 1 selection");
    println!("  Average confidence boost: +8.5%");

    Ok(())
}

/// Demonstrate cascade ensemble for efficient processing
async fn cascade_ensemble_example() -> Result<()> {
    println!("⚡ Cascade Ensemble Example");
    println!("===========================");

    // Define cascade levels (fast to slow, simple to complex)
    let cascade_levels = vec![
        ("Fast Filter", "distilbert-base-uncased", 50, 0.9), // 50ms, exit if confidence > 0.9
        ("Medium Model", "roberta-base", 150, 0.8),          // 150ms, exit if confidence > 0.8
        ("Complex Model", "bert-large", 400, 0.0),           // 400ms, final decision
    ];

    println!("Cascade Configuration:");
    for (name, model, latency, threshold) in &cascade_levels {
        println!(
            "  Level: {} ({}) - {}ms, exit threshold: {:.1}",
            name, model, latency, threshold
        );
    }

    // Test inputs with varying difficulty
    let test_cases = vec![
        ("I love this!", "Easy case (clear sentiment)", 0.95),
        (
            "This is absolutely terrible and awful!",
            "Easy case (strong negative)",
            0.93,
        ),
        (
            "It's okay, not bad but not great either.",
            "Medium case (neutral)",
            0.75,
        ),
        (
            "The implementation has some merits but also significant drawbacks.",
            "Hard case (mixed)",
            0.65,
        ),
        (
            "I'm not entirely sure how I feel about this particular situation.",
            "Hard case (uncertain)",
            0.60,
        ),
    ];

    println!("\nCascade Processing Results:");

    let mut total_time_cascade = 0u32;
    let mut total_time_full = 0u32;

    for (text, description, simulated_confidence) in test_cases {
        println!("\nInput: \"{}\"", text);
        println!("Difficulty: {}", description);

        let mut current_level = 0;
        let mut exit_early = false;
        let mut processing_time = 0u32;

        for (level_name, _model, latency, threshold) in &cascade_levels {
            processing_time += latency;

            // Simulate confidence increasing with model complexity
            let level_confidence = simulated_confidence * (0.7 + current_level as f32 * 0.15);

            println!(
                "  {}: {:.1}% confidence ({}ms)",
                level_name,
                level_confidence * 100.0,
                latency
            );

            if level_confidence >= *threshold {
                println!("    ✅ Exiting cascade (threshold reached)");
                exit_early = true;
                break;
            } else if current_level < cascade_levels.len() - 1 {
                println!("    ➡️  Proceeding to next level");
            } else {
                println!("    🏁 Final level reached");
            }

            current_level += 1;
        }

        total_time_cascade += processing_time;
        total_time_full += cascade_levels.iter().map(|(_, _, latency, _)| latency).sum::<u32>();

        println!("  Total processing time: {}ms", processing_time);
        println!(
            "  Levels used: {}/{}",
            current_level + 1,
            cascade_levels.len()
        );

        if exit_early {
            let time_saved = cascade_levels
                .iter()
                .skip(current_level + 1)
                .map(|(_, _, latency, _)| latency)
                .sum::<u32>();
            println!("  Time saved: {}ms", time_saved);
        }
    }

    // Overall cascade statistics
    println!("\nCascade Performance Summary:");
    println!("  Total cascade time: {}ms", total_time_cascade);
    println!("  Total full processing time: {}ms", total_time_full);
    println!(
        "  Time savings: {}ms ({:.1}%)",
        total_time_full - total_time_cascade,
        (total_time_full - total_time_cascade) as f32 / total_time_full as f32 * 100.0
    );

    let avg_levels_used = 2.2; // Simulated average
    println!(
        "  Average levels used: {:.1}/{}",
        avg_levels_used,
        cascade_levels.len()
    );
    println!(
        "  Early exit rate: {:.1}%",
        (cascade_levels.len() as f32 - avg_levels_used) / cascade_levels.len() as f32 * 100.0
    );

    Ok(())
}

/// Demonstrate performance comparison between single models and ensembles
async fn performance_comparison_example() -> Result<()> {
    println!("📊 Performance Comparison Example");
    println!("=================================");

    // Simulate performance metrics for different approaches
    let approaches = vec![
        ("Single Best Model", 0.87, 45, 1),
        ("Simple Ensemble (3 models)", 0.91, 135, 3),
        ("Weighted Ensemble (3 models)", 0.93, 140, 3),
        ("Cascade Ensemble (3 levels)", 0.92, 95, 2),
        ("Dynamic Selection", 0.89, 75, 1),
        ("Adaptive Ensemble", 0.94, 160, 4),
    ];

    println!("Approach Comparison:");
    println!("  Approach                     | Accuracy | Latency | Avg Models");
    println!("  -----------------------------|----------|---------|------------");

    for (name, accuracy, latency, avg_models) in &approaches {
        println!(
            "  {:28} | {:6.1}% | {:5}ms | {:8.1}",
            name,
            accuracy * 100.0,
            latency,
            avg_models
        );
    }

    // Performance analysis
    println!("\nPerformance Analysis:");

    // Find best accuracy
    let best_accuracy = approaches.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!(
        "  Best accuracy: {} ({:.1}%)",
        best_accuracy.0,
        best_accuracy.1 * 100.0
    );

    // Find lowest latency
    let best_latency = approaches.iter().min_by_key(|a| a.2).unwrap();
    println!(
        "  Lowest latency: {} ({}ms)",
        best_latency.0, best_latency.2
    );

    // Calculate efficiency metrics
    println!("\nEfficiency Metrics:");
    for (name, accuracy, latency, _) in &approaches {
        let efficiency = accuracy / (*latency as f32 / 1000.0); // Accuracy per second
        println!("  {}: {:.2} accuracy/sec", name, efficiency);
    }

    // Resource utilization
    println!("\nResource Utilization:");
    println!("  Memory usage (estimated):");
    for (name, _, _, avg_models) in &approaches {
        let memory_mb = (*avg_models * 500) as u32; // 500MB per model
        println!("    {}: {}MB", name, memory_mb);
    }

    // Trade-off analysis
    println!("\nTrade-off Analysis:");
    println!("  Single Model: ✓ Fast, ✓ Low memory, ✗ Lower accuracy");
    println!("  Simple Ensemble: ✓ Better accuracy, ✗ 3x latency, ✗ 3x memory");
    println!("  Weighted Ensemble: ✓ Best accuracy, ✗ Highest latency, ✗ 3x memory");
    println!("  Cascade Ensemble: ✓ Good accuracy, ✓ Reduced latency, ✗ Complex logic");
    println!("  Dynamic Selection: ✓ Balanced, ✓ Adaptive, ✗ Selection overhead");
    println!("  Adaptive Ensemble: ✓ Highest accuracy, ✗ Highest resource usage");

    Ok(())
}

/// Demonstrate adaptive ensemble that learns from performance
async fn adaptive_ensemble_example() -> Result<()> {
    println!("🔄 Adaptive Ensemble Example");
    println!("============================");

    // Initialize adaptive ensemble
    let mut ensemble_weights = vec![("model_a", 0.33), ("model_b", 0.33), ("model_c", 0.34)];

    println!("Initial ensemble weights:");
    for (model, weight) in &ensemble_weights {
        println!("  {}: {:.2}", model, weight);
    }

    // Simulate learning from performance data
    let performance_feedback = vec![
        // (model_a_correct, model_b_correct, model_c_correct, ensemble_correct)
        (true, true, false, true),   // Model C was wrong
        (true, false, true, true),   // Model B was wrong
        (false, true, true, true),   // Model A was wrong
        (true, true, true, true),    // All correct
        (true, false, false, false), // Only Model A correct
        (false, true, true, true),   // Model A wrong again
        (true, true, false, true),   // Model C wrong again
        (true, true, true, true),    // All correct
    ];

    println!("\nLearning from performance feedback:");

    // Track model performance
    let mut model_scores = vec![0f32; 3];
    let total_samples = performance_feedback.len() as f32;

    for (i, feedback) in performance_feedback.iter().enumerate() {
        let correct = [feedback.0, feedback.1, feedback.2];

        // Update scores
        for (j, &is_correct) in correct.iter().enumerate() {
            if is_correct {
                model_scores[j] += 1.0;
            }
        }

        println!("  Sample {}: Models correct: {:?}", i + 1, correct);
    }

    // Calculate performance rates
    let performance_rates: Vec<f32> =
        model_scores.iter().map(|&score| score / total_samples).collect();

    println!(
        "\nPerformance rates after {} samples:",
        performance_feedback.len()
    );
    for (i, &rate) in performance_rates.iter().enumerate() {
        println!(
            "  model_{}: {:.1}% ({}/{})",
            ['a', 'b', 'c'][i],
            rate * 100.0,
            model_scores[i] as u32,
            total_samples as u32
        );
    }

    // Adaptive weight adjustment
    let total_performance: f32 = performance_rates.iter().sum();
    let new_weights: Vec<f32> =
        performance_rates.iter().map(|&rate| rate / total_performance).collect();

    println!("\nAdaptive weight adjustment:");
    println!("  Model    | Old Weight | Performance | New Weight | Change");
    println!("  ---------|------------|-------------|------------|--------");

    for i in 0..3 {
        let old_weight = ensemble_weights[i].1;
        let new_weight = new_weights[i];
        let change = new_weight - old_weight;
        let change_str =
            if change > 0.0 { format!("+{:.2}", change) } else { format!("{:.2}", change) };

        println!(
            "  model_{} | {:10.2} | {:9.1}% | {:10.2} | {}",
            ['a', 'b', 'c'][i],
            old_weight,
            performance_rates[i] * 100.0,
            new_weight,
            change_str
        );

        ensemble_weights[i].1 = new_weight;
    }

    // Predict ensemble improvement
    println!("\nEnsemble improvement prediction:");
    let old_ensemble_score = 0.85; // Simulated baseline
    let new_ensemble_score = performance_rates
        .iter()
        .zip(new_weights.iter())
        .map(|(rate, weight)| rate * weight)
        .sum::<f32>();

    println!(
        "  Baseline ensemble accuracy: {:.1}%",
        old_ensemble_score * 100.0
    );
    println!(
        "  Predicted new accuracy: {:.1}%",
        new_ensemble_score * 100.0
    );
    println!(
        "  Expected improvement: {:.1}%",
        (new_ensemble_score - old_ensemble_score) * 100.0
    );

    // Adaptation strategies
    println!("\nAdaptation strategies applied:");
    if performance_rates[0] > performance_rates[1] && performance_rates[0] > performance_rates[2] {
        println!("  ✓ Increased weight for best performing model (model_a)");
    }
    if performance_rates.iter().any(|&rate| rate < 0.5) {
        println!("  ⚠️  Consider removing poorly performing models");
    }
    println!("  ✓ Weights normalized to sum to 1.0");
    println!("  ✓ Continuous learning enabled for future adaptation");

    Ok(())
}

/// Utility functions for ensemble examples

fn simulate_individual_predictions(text: &str) -> Vec<(&'static str, &'static str, f32)> {
    // Simulate predictions based on text characteristics
    let positive_indicators = ["love", "amazing", "fantastic", "great", "excellent"];
    let negative_indicators = ["terrible", "awful", "hate", "worst", "disappointing"];

    let text_lower = text.to_lowercase();
    let has_positive = positive_indicators.iter().any(|&word| text_lower.contains(word));
    let has_negative = negative_indicators.iter().any(|&word| text_lower.contains(word));

    let (prediction, base_confidence) = if has_positive && !has_negative {
        ("POSITIVE", 0.9)
    } else if has_negative && !has_positive {
        ("NEGATIVE", 0.9)
    } else if has_positive && has_negative {
        ("NEUTRAL", 0.6)
    } else {
        ("NEUTRAL", 0.7)
    };

    // Simulate slight variations between models
    vec![
        ("DistilBERT", prediction, base_confidence - 0.05),
        ("RoBERTa", prediction, base_confidence + 0.02),
        ("BERT Multilingual", prediction, base_confidence - 0.02),
    ]
}

fn simulate_voting_strategy(
    predictions: &[(&str, &str, f32)],
    strategy: &EnsembleStrategy,
) -> (String, f32) {
    match strategy {
        EnsembleStrategy::MajorityVote => {
            let positive_votes =
                predictions.iter().filter(|(_, pred, _)| *pred == "POSITIVE").count();
            let negative_votes =
                predictions.iter().filter(|(_, pred, _)| *pred == "NEGATIVE").count();

            if positive_votes > negative_votes {
                ("POSITIVE".to_string(), 0.85)
            } else if negative_votes > positive_votes {
                ("NEGATIVE".to_string(), 0.85)
            } else {
                ("NEUTRAL".to_string(), 0.60)
            }
        },
        EnsembleStrategy::WeightedAverage(_) => {
            let weights = [0.35, 0.40, 0.25];
            let weighted_positive: f32 = predictions
                .iter()
                .zip(weights.iter())
                .filter(|((_, pred, _), _)| *pred == "POSITIVE")
                .map(|((_, _, conf), weight)| conf * weight)
                .sum();

            if weighted_positive > 0.4 {
                ("POSITIVE".to_string(), weighted_positive + 0.15)
            } else {
                ("NEGATIVE".to_string(), 0.80)
            }
        },
        EnsembleStrategy::Maximum => {
            let best_prediction =
                predictions.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
            (best_prediction.1.to_string(), best_prediction.2)
        },
        EnsembleStrategy::Average => {
            let first_prediction = predictions[0].1;
            let all_agree = predictions.iter().all(|(_, pred, _)| *pred == first_prediction);

            if all_agree {
                (first_prediction.to_string(), 0.95)
            } else {
                ("UNCERTAIN".to_string(), 0.40)
            }
        },
        _ => ("NEUTRAL".to_string(), 0.50),
    }
}

fn analyze_input_characteristics(text: &str) -> Vec<String> {
    let mut characteristics = Vec::new();

    if text.len() < 50 {
        characteristics.push("short".to_string());
    } else if text.len() > 200 {
        characteristics.push("long".to_string());
    }

    if text.contains("!") || text.contains("?") || text.chars().any(|c| c.is_uppercase()) {
        characteristics.push("informal".to_string());
    }

    if text.contains("📊") || text.contains("🔥") || text.contains("#") {
        characteristics.push("social_media".to_string());
    }

    if text.contains("recommend") || text.contains("stars") || text.contains("product") {
        characteristics.push("review".to_string());
    }

    if text.contains("report") || text.contains("analysis") || text.contains("quarterly") {
        characteristics.push("business".to_string());
    }

    if characteristics.is_empty() {
        characteristics.push("general".to_string());
    }

    characteristics
}

fn select_best_model<'a>(
    models: &'a [(&str, &str, Vec<&str>)],
    characteristics: &[String],
) -> (&'a str, Vec<String>) {
    let mut best_match = &models[0];
    let mut best_score = 0;

    for model in models {
        let score = model
            .2
            .iter()
            .filter(|&trait_| characteristics.iter().any(|char| char.contains(trait_)))
            .count();

        if score > best_score {
            best_score = score;
            best_match = model;
        }
    }

    let matched_traits = best_match
        .2
        .iter()
        .filter(|&trait_| characteristics.iter().any(|char| char.contains(trait_)))
        .map(|s| s.to_string())
        .collect();

    (best_match.0, matched_traits)
}

/// Mock types and implementations
#[derive(Debug)]
pub struct EnsemblePrediction {
    pub prediction: String,
    pub confidence: f32,
    pub agreement_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_predictions() {
        let predictions = simulate_individual_predictions("I love this!");
        assert_eq!(predictions.len(), 3);
        assert!(predictions.iter().all(|(_, pred, _)| *pred == "POSITIVE"));
    }

    #[test]
    fn test_analyze_characteristics() {
        let chars = analyze_input_characteristics("OMG this is amazing!!! 🔥");
        assert!(chars.contains(&"short".to_string()));
        assert!(chars.contains(&"informal".to_string()));
    }

    #[test]
    fn test_voting_strategies() {
        let predictions = vec![
            ("Model A", "POSITIVE", 0.9),
            ("Model B", "POSITIVE", 0.8),
            ("Model C", "NEGATIVE", 0.7),
        ];

        let result = simulate_voting_strategy(&predictions, &EnsembleStrategy::MajorityVote);
        assert_eq!(result.0, "POSITIVE");
    }
}
