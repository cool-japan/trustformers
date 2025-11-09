//! Large Language Model (LLM) Debugging Demonstration
#![allow(unused_variables)]
//!
//! This example demonstrates the advanced LLM debugging capabilities including
//! safety analysis, factuality checking, alignment monitoring, and more.

use anyhow::Result;
use trustformers_debug::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🚀 TrustformeRS LLM Debugging Demo");
    println!("=====================================");

    // Create an LLM debugger with default configuration
    let mut debugger = llm_debugger();

    println!("\n📊 1. Basic LLM Response Analysis");
    println!("---------------------------------");

    // Analyze a safe, helpful response
    let safe_response = debugger.analyze_response(
        "What is the capital of France?",
        "The capital of France is Paris. It's located in the north-central part of the country and has been the capital since 508 AD.",
        None,
        None,
    ).await?;

    println!("✅ Safe Response Analysis:");
    println!("   Overall Score: {:.2}", safe_response.overall_score);
    if let Some(safety) = &safe_response.safety_analysis {
        println!("   Safety Score: {:.2}", safety.safety_score);
        println!("   Risk Level: {:?}", safety.risk_level);
    }
    if let Some(factuality) = &safe_response.factuality_analysis {
        println!("   Factuality Score: {:.2}", factuality.factuality_score);
        println!("   Verified Claims: {}", factuality.verified_claims);
    }
    println!("   Recommendations: {:?}", safe_response.recommendations);

    // Analyze a potentially problematic response
    let problematic_response = debugger.analyze_response(
        "Tell me about violence in movies",
        "Violence in movies can be quite graphic and disturbing. Some films might contain toxic content that could be harmful to viewers.",
        None,
        None,
    ).await?;

    println!("\n⚠️  Potentially Problematic Response Analysis:");
    println!(
        "   Overall Score: {:.2}",
        problematic_response.overall_score
    );
    if let Some(safety) = &problematic_response.safety_analysis {
        println!("   Safety Score: {:.2}", safety.safety_score);
        println!("   Risk Level: {:?}", safety.risk_level);
        println!("   Detected Harms: {:?}", safety.detected_harms);
    }
    println!(
        "   Recommendations: {:?}",
        problematic_response.recommendations
    );

    println!("\n🎯 2. Safety-Focused Analysis");
    println!("----------------------------");

    // Create a safety-focused debugger
    let mut safety_debugger = llm_debugger_with_config(safety_focused_config());

    let safety_analysis = safety_debugger.analyze_response(
        "How do I stay safe online?",
        "To stay safe online, use strong passwords, enable two-factor authentication, be cautious about sharing personal information, and keep your software updated.",
        None,
        None,
    ).await?;

    println!("🛡️  Safety-Focused Analysis:");
    println!("   Overall Score: {:.2}", safety_analysis.overall_score);
    println!(
        "   Has Critical Issues: {}",
        safety_analysis.recommendations.len() > 0
    );

    println!("\n⚡ 3. Performance-Focused Analysis");
    println!("----------------------------------");

    // Create a performance-focused debugger
    let mut perf_debugger = llm_debugger_with_config(performance_focused_config());

    // Simulate generation metrics
    let generation_metrics = GenerationMetrics {
        tokens_per_second: 75.0,
        average_response_length: 120.0,
        generation_latency_p50: 180.0,
        generation_latency_p95: 450.0,
        generation_latency_p99: 800.0,
        first_token_latency: 40.0,
        completion_rate: 0.96,
        timeout_rate: 0.04,
    };

    let perf_analysis = perf_debugger.analyze_response(
        "Explain quantum computing",
        "Quantum computing is a revolutionary technology that uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot.",
        None,
        Some(generation_metrics.clone()),
    ).await?;

    println!("⚡ Performance Analysis:");
    if let Some(perf) = &perf_analysis.performance_analysis {
        println!(
            "   Tokens/Second: {:.1}",
            perf.generation_metrics.tokens_per_second
        );
        println!(
            "   P50 Latency: {:.1}ms",
            perf.generation_metrics.generation_latency_p50
        );
        println!(
            "   P95 Latency: {:.1}ms",
            perf.generation_metrics.generation_latency_p95
        );
        println!(
            "   Completion Rate: {:.2}%",
            perf.generation_metrics.completion_rate * 100.0
        );
        println!("   Quality Metrics:");
        println!(
            "     Coherence: {:.2}",
            perf.quality_metrics.coherence_score
        );
        println!("     Fluency: {:.2}", perf.quality_metrics.fluency_score);
        println!(
            "     Factual Accuracy: {:.2}",
            perf.quality_metrics.factual_accuracy
        );
        println!("   Efficiency Metrics:");
        println!(
            "     Memory Efficiency: {:.2}",
            perf.efficiency_metrics.memory_efficiency
        );
        println!(
            "     Compute Utilization: {:.2}",
            perf.efficiency_metrics.compute_utilization
        );
        println!(
            "     Energy per 1K tokens: {:.3} kWh",
            perf.efficiency_metrics.energy_consumption
        );
        println!(
            "     Carbon footprint per 1K tokens: {:.3} kg CO2",
            perf.efficiency_metrics.carbon_footprint_estimate
        );
    }

    println!("\n📊 4. Batch Analysis");
    println!("-------------------");

    // Perform batch analysis on multiple interactions
    let interactions = vec![
        ("Hello!".to_string(), "Hi there! How can I help you today?".to_string()),
        ("What's 2+2?".to_string(), "2 + 2 equals 4.".to_string()),
        ("Tell me a joke".to_string(), "Why don't scientists trust atoms? Because they make up everything!".to_string()),
        ("What's the weather like?".to_string(), "I don't have access to real-time weather data, but you can check a reliable weather service for current conditions.".to_string()),
        ("How do I learn programming?".to_string(), "Start with a beginner-friendly language like Python, practice regularly, work on projects, and use online resources like tutorials and coding challenges.".to_string()),
    ];

    let batch_report = debugger.analyze_batch(&interactions).await?;

    println!("📈 Batch Analysis Results:");
    println!("   Batch Size: {}", batch_report.batch_size);
    println!(
        "   Average Overall Score: {:.2}",
        batch_report.batch_metrics.average_overall_score
    );
    println!(
        "   Average Safety Score: {:.2}",
        batch_report.batch_metrics.average_safety_score
    );
    println!(
        "   Average Factuality Score: {:.2}",
        batch_report.batch_metrics.average_factuality_score
    );
    println!(
        "   Flagged Responses: {}",
        batch_report.batch_metrics.flagged_responses_count
    );
    println!(
        "   Critical Issues: {}",
        batch_report.batch_metrics.critical_issues_count
    );

    println!("\n🏥 5. Health Report Generation");
    println!("-----------------------------");

    // Generate comprehensive health report
    let health_report = debugger.generate_health_report().await?;

    println!("🏥 LLM Health Report:");
    println!(
        "   Overall Health Score: {:.2}/1.0",
        health_report.overall_health_score
    );
    println!("   Component Health:");
    println!(
        "     Safety: {:.2} ({:?})",
        health_report.safety_health.score, health_report.safety_health.status
    );
    println!(
        "     Factuality: {:.2} ({:?})",
        health_report.factuality_health.score, health_report.factuality_health.status
    );
    println!(
        "     Alignment: {:.2} ({:?})",
        health_report.alignment_health.score, health_report.alignment_health.status
    );
    println!(
        "     Bias: {:.2} ({:?})",
        health_report.bias_health.score, health_report.bias_health.status
    );
    println!(
        "     Performance: {:.2} ({:?})",
        health_report.performance_health.score, health_report.performance_health.status
    );
    println!(
        "   Critical Issues: {}",
        health_report.critical_issues.len()
    );

    if !health_report.critical_issues.is_empty() {
        println!("   🚨 Critical Issues:");
        for issue in &health_report.critical_issues {
            println!("     - {:?}: {}", issue.severity, issue.description);
            println!("       Action: {}", issue.recommended_action);
        }
    }

    if !health_report.recommendations.is_empty() {
        println!("   💡 Recommendations:");
        for rec in &health_report.recommendations {
            println!("     - {}", rec);
        }
    }

    println!("\n🔧 6. Specialized Use Cases");
    println!("---------------------------");

    println!("🔧 Specialized LLM Debugging Configurations:");

    // Safety-focused configuration details
    let safety_config = safety_focused_config();
    println!("   Safety-Focused Config:");
    println!(
        "     - Safety Analysis: {}",
        safety_config.enable_safety_analysis
    );
    println!(
        "     - Bias Detection: {}",
        safety_config.enable_bias_detection
    );
    println!(
        "     - Safety Threshold: {:.1}",
        safety_config.safety_threshold
    );
    println!(
        "     - Sampling Rate: {:.1}",
        safety_config.analysis_sampling_rate
    );

    // Performance-focused configuration details
    let perf_config = performance_focused_config();
    println!("   Performance-Focused Config:");
    println!(
        "     - Performance Profiling: {}",
        perf_config.enable_llm_performance_profiling
    );
    println!(
        "     - Conversation Analysis: {}",
        perf_config.enable_conversation_analysis
    );
    println!(
        "     - Sampling Rate: {:.1}",
        perf_config.analysis_sampling_rate
    );
    println!(
        "     - Max Conversation Length: {}",
        perf_config.max_conversation_length
    );

    println!("\n🎯 7. Advanced Analysis Features");
    println!("--------------------------------");

    // Demonstrate different analysis types
    let mut advanced_debugger = llm_debugger();

    // Analyze response with context
    let context = vec![
        "Previous conversation about AI safety".to_string(),
        "Discussion about responsible AI development".to_string(),
    ];

    let contextual_analysis = advanced_debugger.analyze_response(
        "How should AI systems handle sensitive topics?",
        "AI systems should approach sensitive topics with careful consideration, transparency about limitations, and appropriate safeguards to avoid harm while still being helpful.",
        Some(&context),
        None,
    ).await?;

    println!("🎯 Contextual Analysis:");
    println!("   Overall Score: {:.2}", contextual_analysis.overall_score);
    println!(
        "   Analysis Duration: {:?}",
        contextual_analysis.analysis_duration
    );

    if let Some(alignment) = &contextual_analysis.alignment_analysis {
        println!("   Alignment Score: {:.2}", alignment.alignment_score);
        println!("   Objective Scores:");
        for (objective, score) in &alignment.objective_scores {
            println!("     {:?}: {:.2}", objective, score);
        }
    }

    println!("\n🎉 Demo Complete!");
    println!("================");
    println!("✅ Successfully demonstrated LLM debugging capabilities:");
    println!("   - Safety analysis with harm detection");
    println!("   - Factuality checking and verification");
    println!("   - Alignment monitoring across objectives");
    println!("   - Bias detection and fairness analysis");
    println!("   - Performance profiling and optimization insights");
    println!("   - Batch analysis for multiple interactions");
    println!("   - Comprehensive health reporting");
    println!("   - Specialized configurations for different use cases");

    println!("\n🚀 The TrustformeRS LLM debugging framework provides:");
    println!("   - Real-time safety monitoring");
    println!("   - Comprehensive factuality verification");
    println!("   - Multi-objective alignment assessment");
    println!("   - Advanced bias and fairness analysis");
    println!("   - Performance and efficiency profiling");
    println!("   - Conversation-level analysis");
    println!("   - Actionable recommendations and insights");

    Ok(())
}

/// Helper function to print analysis details
#[allow(dead_code)]
fn print_analysis_summary(analysis: &LLMAnalysisReport, title: &str) {
    println!("\n{}", title);
    println!("{}", "=".repeat(title.len()));
    println!("Overall Score: {:.2}", analysis.overall_score);

    if let Some(safety) = &analysis.safety_analysis {
        println!(
            "Safety: {:.2} (Risk: {:?})",
            safety.safety_score, safety.risk_level
        );
    }

    if let Some(factuality) = &analysis.factuality_analysis {
        println!(
            "Factuality: {:.2} (Claims: {})",
            factuality.factuality_score, factuality.verified_claims
        );
    }

    if let Some(alignment) = &analysis.alignment_analysis {
        println!("Alignment: {:.2}", alignment.alignment_score);
    }

    if !analysis.recommendations.is_empty() {
        println!("Recommendations:");
        for rec in &analysis.recommendations {
            println!("  • {}", rec);
        }
    }
}

/// Demonstrate macro usage for LLM debugging
#[allow(dead_code)]
async fn demonstrate_macros() -> Result<()> {
    println!("\n🔧 Macro Usage Demonstration");
    println!("----------------------------");

    let mut debugger = llm_debugger();

    // Using the debug_llm_response macro
    let result = debug_llm_response!(
        debugger,
        "What is machine learning?",
        "Machine learning is a subset of AI that enables computers to learn from data without explicit programming."
    )?;

    println!("Macro Result: Overall score {:.2}", result.overall_score);

    // Using the debug_llm_batch macro
    let interactions = vec![
        ("Hi".to_string(), "Hello!".to_string()),
        ("Bye".to_string(), "Goodbye!".to_string()),
    ];

    let batch_result = debug_llm_batch!(debugger, &interactions)?;
    println!(
        "Batch Macro Result: {} interactions analyzed",
        batch_result.batch_size
    );

    Ok(())
}
