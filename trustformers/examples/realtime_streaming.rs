//! Real-time Streaming Demo
#![allow(unused_variables)]
//!
//! This example demonstrates TrustformeRS's real-time streaming capabilities,
//! showing how to process continuous text streams, handle backpressure,
//! and provide real-time analytics.

use clap::Parser;
use colored::*;
use fastrand; // Using fastrand from dependencies
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::{broadcast, mpsc};
use tokio::time::{interval, Duration, Instant};
use trustformers::pipeline::StreamConfig;
use trustformers::Result;

#[derive(Parser)]
#[command(name = "trustformers-streaming")]
#[command(about = "Real-time streaming processing demo")]
struct Args {
    #[arg(long, default_value = "text-classification")]
    task: String,

    #[arg(long, default_value = "100")]
    rate: u64, // messages per second

    #[arg(long, default_value = "30")]
    duration: u64, // seconds

    #[arg(long)]
    interactive: bool,

    #[arg(long)]
    analytics: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StreamMessage {
    id: u64,
    text: String,
    timestamp: u64,
    source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProcessedMessage {
    id: u64,
    original: StreamMessage,
    result: serde_json::Value,
    processing_time_ms: u64,
    processed_at: u64,
}

#[derive(Debug, Clone)]
struct StreamingStats {
    messages_received: u64,
    messages_processed: u64,
    messages_failed: u64,
    total_processing_time: u64,
    average_latency: f64,
    #[allow(dead_code)]
    throughput: f64,
    errors: Vec<String>,
}

#[derive(Clone)]
struct StreamProcessor {
    pipeline: Arc<Mutex<MockStreamingPipeline>>,
    stats: Arc<Mutex<StreamingStats>>,
    config: StreamConfig,
}

struct MockStreamingPipeline {
    task: String,
    #[allow(dead_code)]
    model_name: String,
}

// Remove local StreamingConfig as we'll use StreamConfig from the library

impl MockStreamingPipeline {
    fn new(task: String) -> Self {
        let model_name = match task.as_str() {
            "text-classification" => "distilbert-base-uncased-finetuned-sst-2-english",
            "text-generation" => "gpt2",
            _ => "default-model",
        };

        Self {
            task,
            model_name: model_name.to_string(),
        }
    }

    async fn process(&self, message: &StreamMessage) -> Result<serde_json::Value> {
        // Simulate processing time
        let processing_delay = match self.task.as_str() {
            "text-classification" => Duration::from_millis(20 + (message.text.len() as u64 / 10)),
            "text-generation" => Duration::from_millis(50 + (message.text.len() as u64 / 5)),
            _ => Duration::from_millis(30),
        };

        tokio::time::sleep(processing_delay).await;

        // Simulate processing logic
        match self.task.as_str() {
            "text-classification" => {
                let sentiment = if message.text.to_lowercase().contains("good")
                    || message.text.to_lowercase().contains("great")
                    || message.text.to_lowercase().contains("love")
                {
                    "POSITIVE"
                } else if message.text.to_lowercase().contains("bad")
                    || message.text.to_lowercase().contains("hate")
                    || message.text.to_lowercase().contains("terrible")
                {
                    "NEGATIVE"
                } else {
                    "NEUTRAL"
                };

                let score = 0.85 + (fastrand::f64() * 0.15);

                Ok(serde_json::json!({
                    "label": sentiment,
                    "score": score
                }))
            },
            "text-generation" => {
                let generated = format!("{} continues with amazing possibilities...", message.text);
                Ok(serde_json::json!({
                    "generated_text": generated
                }))
            },
            _ => Ok(serde_json::json!({"result": "processed"})),
        }
    }

    async fn process_batch(&self, messages: &[StreamMessage]) -> Vec<Result<serde_json::Value>> {
        // Simulate batch processing with some efficiency gains
        let batch_delay = Duration::from_millis(10 + (messages.len() as u64 * 15));
        tokio::time::sleep(batch_delay).await;

        // Process each message in the batch
        let mut results = Vec::new();
        for message in messages {
            results.push(self.process(message).await);
        }
        results
    }
}

impl StreamProcessor {
    fn new(task: String, config: StreamConfig) -> Self {
        Self {
            pipeline: Arc::new(Mutex::new(MockStreamingPipeline::new(task))),
            stats: Arc::new(Mutex::new(StreamingStats {
                messages_received: 0,
                messages_processed: 0,
                messages_failed: 0,
                total_processing_time: 0,
                average_latency: 0.0,
                throughput: 0.0,
                errors: Vec::new(),
            })),
            config,
        }
    }

    async fn start_processing(
        &self,
        mut message_rx: mpsc::Receiver<StreamMessage>,
        result_tx: broadcast::Sender<ProcessedMessage>,
    ) -> Result<()> {
        let mut message_buffer = VecDeque::new();
        let mut batch_timer = interval(Duration::from_millis(self.config.flush_interval_ms));
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.max_concurrent));

        println!("🌊 Starting stream processor...");
        println!("  Max batch size: {}", self.config.batch_size.unwrap_or(16));
        println!("  Flush interval: {}ms", self.config.flush_interval_ms);
        println!("  Buffer size: {}", self.config.buffer_size);
        println!("  Concurrent requests: {}", self.config.max_concurrent);

        loop {
            tokio::select! {
                // Receive new messages
                message = message_rx.recv() => {
                    match message {
                        Some(msg) => {
                            {
                                let mut stats = self.stats.lock().await;
                                stats.messages_received += 1;
                            }

                            message_buffer.push_back(msg);

                            // Check if buffer is full or batch is ready
                            let max_batch_size = self.config.batch_size.unwrap_or(16);
                            if message_buffer.len() >= max_batch_size {
                                self.process_batch(&mut message_buffer, &result_tx, &semaphore).await;
                            }

                            // Handle backpressure
                            let backpressure_threshold = (self.config.buffer_size as f64 * self.config.backpressure_threshold) as usize;
                            if message_buffer.len() > backpressure_threshold {
                                println!("⚠️ Backpressure detected, dropping oldest messages");
                                while message_buffer.len() > self.config.buffer_size / 2 {
                                    message_buffer.pop_front();
                                }
                            }
                        },
                        None => {
                            println!("📪 Message channel closed, processing remaining messages...");
                            if !message_buffer.is_empty() {
                                self.process_batch(&mut message_buffer, &result_tx, &semaphore).await;
                            }
                            break;
                        }
                    }
                },

                // Process batch on timer
                _ = batch_timer.tick() => {
                    if !message_buffer.is_empty() {
                        self.process_batch(&mut message_buffer, &result_tx, &semaphore).await;
                    }
                }
            }
        }

        Ok(())
    }

    async fn process_batch(
        &self,
        message_buffer: &mut VecDeque<StreamMessage>,
        result_tx: &broadcast::Sender<ProcessedMessage>,
        semaphore: &Arc<tokio::sync::Semaphore>,
    ) {
        if message_buffer.is_empty() {
            return;
        }

        // Extract batch from buffer
        let max_batch_size = self.config.batch_size.unwrap_or(16);
        let batch_size = std::cmp::min(message_buffer.len(), max_batch_size);
        let batch: Vec<StreamMessage> = message_buffer.drain(..batch_size).collect();

        // Acquire semaphore permit for concurrent processing
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let pipeline = self.pipeline.clone();
        let stats = self.stats.clone();
        let result_tx_clone = result_tx.clone();

        // Process batch in background task
        tokio::spawn(async move {
            let start_time = Instant::now();

            let pipeline_guard = pipeline.lock().await;
            let results = pipeline_guard.process_batch(&batch).await;
            drop(pipeline_guard);

            let processing_time = start_time.elapsed().as_millis() as u64;

            // Send results and update stats
            let mut stats_guard = stats.lock().await;
            for (message, result) in batch.iter().zip(results.iter()) {
                match result {
                    Ok(value) => {
                        stats_guard.messages_processed += 1;
                        stats_guard.total_processing_time += processing_time / batch.len() as u64;

                        let processed = ProcessedMessage {
                            id: message.id,
                            original: message.clone(),
                            result: value.clone(),
                            processing_time_ms: processing_time / batch.len() as u64,
                            processed_at: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64,
                        };

                        let _ = result_tx_clone.send(processed);
                    },
                    Err(e) => {
                        stats_guard.messages_failed += 1;
                        stats_guard.errors.push(format!(
                            "Processing failed for message {}: {}",
                            message.id, e
                        ));
                    },
                }
            }

            // Update derived stats
            if stats_guard.messages_processed > 0 {
                stats_guard.average_latency = stats_guard.total_processing_time as f64
                    / stats_guard.messages_processed as f64;
            }

            drop(permit);
        });
    }

    async fn get_stats(&self) -> StreamingStats {
        self.stats.lock().await.clone()
    }
}

struct MessageGenerator {
    message_counter: u64,
    sources: Vec<String>,
}

impl MessageGenerator {
    fn new() -> Self {
        Self {
            message_counter: 0,
            sources: vec![
                "social_media".to_string(),
                "customer_support".to_string(),
                "news_feed".to_string(),
                "user_reviews".to_string(),
                "chat_messages".to_string(),
            ],
        }
    }

    fn generate_message(&mut self) -> StreamMessage {
        let templates = vec![
            "I love this new product!",
            "This service is terrible",
            "The customer support was amazing",
            "Having issues with the application",
            "Great experience overall",
            "Could be better but acceptable",
            "Fantastic work by the team",
            "Not what I expected",
            "Highly recommend to others",
            "Needs significant improvement",
        ];

        self.message_counter += 1;

        StreamMessage {
            id: self.message_counter,
            text: templates[fastrand::usize(0..templates.len())].to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            source: self.sources[fastrand::usize(0..self.sources.len())].clone(),
        }
    }

    async fn start_generating(
        &mut self,
        message_tx: mpsc::Sender<StreamMessage>,
        rate: u64,
        duration: u64,
    ) -> Result<()> {
        let interval_duration = Duration::from_millis(1000 / rate);
        let mut interval = interval(interval_duration);
        let end_time = Instant::now() + Duration::from_secs(duration);

        println!("📡 Starting message generation...");
        println!("  Rate: {} messages/second", rate);
        println!("  Duration: {} seconds", duration);

        while Instant::now() < end_time {
            interval.tick().await;

            let message = self.generate_message();

            match message_tx.send(message).await {
                Ok(_) => {},
                Err(_) => {
                    println!("📪 Message channel closed, stopping generation");
                    break;
                },
            }
        }

        println!("✅ Message generation completed");
        Ok(())
    }
}

struct RealTimeAnalytics {
    window_size: Duration,
    data_points: VecDeque<(Instant, StreamingStats)>,
}

impl RealTimeAnalytics {
    fn new(window_size: Duration) -> Self {
        Self {
            window_size,
            data_points: VecDeque::new(),
        }
    }

    fn update(&mut self, stats: StreamingStats) {
        let now = Instant::now();
        self.data_points.push_back((now, stats));

        // Remove old data points outside the window
        while let Some((timestamp, _)) = self.data_points.front() {
            if now.duration_since(*timestamp) > self.window_size {
                self.data_points.pop_front();
            } else {
                break;
            }
        }
    }

    fn get_analytics(&self) -> HashMap<String, f64> {
        if self.data_points.is_empty() {
            return HashMap::new();
        }

        let mut analytics = HashMap::new();

        if let (Some((_, first)), Some((_, last))) =
            (self.data_points.front(), self.data_points.back())
        {
            let processed_diff = last.messages_processed - first.messages_processed;
            let time_diff = self.window_size.as_secs_f64();

            analytics.insert(
                "throughput_per_sec".to_string(),
                processed_diff as f64 / time_diff,
            );
            analytics.insert(
                "error_rate".to_string(),
                last.messages_failed as f64 / (last.messages_received as f64).max(1.0),
            );
            analytics.insert("average_latency_ms".to_string(), last.average_latency);
            analytics.insert(
                "total_processed".to_string(),
                last.messages_processed as f64,
            );
            analytics.insert("total_failed".to_string(), last.messages_failed as f64);
        }

        analytics
    }

    fn print_dashboard(&self) {
        let analytics = self.get_analytics();

        println!("\n{}", "📊 Real-Time Analytics Dashboard".bold().cyan());
        println!("{}", "═".repeat(50).cyan());

        if analytics.is_empty() {
            println!("No data available yet...");
            return;
        }

        println!(
            "🚀 Throughput: {:.1} msg/sec",
            analytics.get("throughput_per_sec").unwrap_or(&0.0)
        );
        println!(
            "⚡ Avg Latency: {:.1} ms",
            analytics.get("average_latency_ms").unwrap_or(&0.0)
        );
        println!(
            "✅ Total Processed: {:.0}",
            analytics.get("total_processed").unwrap_or(&0.0)
        );
        println!(
            "❌ Total Failed: {:.0}",
            analytics.get("total_failed").unwrap_or(&0.0)
        );
        println!(
            "📈 Error Rate: {:.3}%",
            analytics.get("error_rate").unwrap_or(&0.0) * 100.0
        );

        // Simple ASCII chart for throughput
        let throughput = analytics.get("throughput_per_sec").unwrap_or(&0.0);
        let bar_length = (*throughput / 10.0).min(50.0) as usize;
        let bar = "█".repeat(bar_length);
        println!(
            "📊 Throughput: {}{}",
            bar.green(),
            " ".repeat(50 - bar_length)
        );
    }
}

async fn interactive_mode() -> Result<()> {
    println!("{}", "🎮 Interactive Streaming Mode".bold().cyan());
    println!("{}", "═".repeat(40).cyan());

    use std::io::{self, Write};

    let config = StreamConfig::default();
    let processor = StreamProcessor::new("text-classification".to_string(), config);

    let (message_tx, message_rx) = mpsc::channel(1000);
    let (result_tx, mut result_rx) = broadcast::channel(1000);

    // Start processor
    let processor_clone = processor.clone();
    let processor_handle =
        tokio::spawn(async move { processor_clone.start_processing(message_rx, result_tx).await });

    // Start result handler
    let result_handle = tokio::spawn(async move {
        while let Ok(result) = result_rx.recv().await {
            println!("✅ Processed message {}: {:?}", result.id, result.result);
        }
    });

    println!("Enter messages to process (type 'quit' to exit):");

    let mut message_id = 0;
    loop {
        print!("📝 Enter message: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input == "quit" {
            break;
        }

        if !input.is_empty() {
            message_id += 1;
            let message = StreamMessage {
                id: message_id,
                text: input.to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                source: "interactive".to_string(),
            };

            if let Err(_) = message_tx.send(message).await {
                println!("❌ Failed to send message");
                break;
            }
        }
    }

    drop(message_tx);
    let _ = processor_handle.await;
    result_handle.abort();

    // Print final stats
    let stats = processor.get_stats().await;
    println!("\n📊 Final Statistics:");
    println!("  Messages processed: {}", stats.messages_processed);
    println!("  Messages failed: {}", stats.messages_failed);
    println!("  Average latency: {:.2}ms", stats.average_latency);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.interactive {
        return interactive_mode().await;
    }

    println!(
        "{}",
        "🌊 TrustformeRS Real-Time Streaming Demo".bold().cyan()
    );
    println!("{}", "═".repeat(50).cyan());

    // Set up channels
    let (message_tx, message_rx) = mpsc::channel(1000);
    let (result_tx, mut result_rx) = broadcast::channel(1000);

    // Create processor
    let config = StreamConfig::default();
    let processor = StreamProcessor::new(args.task.clone(), config);

    // Create analytics
    let mut analytics = RealTimeAnalytics::new(Duration::from_secs(10));

    // Start processor
    let processor_clone = processor.clone();
    let processor_handle =
        tokio::spawn(async move { processor_clone.start_processing(message_rx, result_tx).await });

    // Start message generator
    let mut generator = MessageGenerator::new();
    let generator_handle = tokio::spawn(async move {
        generator.start_generating(message_tx, args.rate, args.duration).await
    });

    // Start analytics updater
    let processor_stats = processor.clone();
    let analytics_handle = tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            let stats = processor_stats.get_stats().await;
            analytics.update(stats);

            if args.analytics {
                analytics.print_dashboard();
            }
        }
    });

    // Handle results
    let result_handle = tokio::spawn(async move {
        let mut count = 0;
        while let Ok(result) = result_rx.recv().await {
            count += 1;
            if count % 10 == 0 {
                println!(
                    "📦 Processed {} messages (latest: {}ms)",
                    count, result.processing_time_ms
                );
            }
        }
    });

    // Wait for completion
    let _ = generator_handle.await;
    let _ = processor_handle.await;
    analytics_handle.abort();
    result_handle.abort();

    // Print final statistics
    let final_stats = processor.get_stats().await;
    println!("\n{}", "🏁 Final Results".bold().green());
    println!("{}", "═".repeat(30).green());
    println!("📊 Messages received: {}", final_stats.messages_received);
    println!("✅ Messages processed: {}", final_stats.messages_processed);
    println!("❌ Messages failed: {}", final_stats.messages_failed);
    println!("⚡ Average latency: {:.2}ms", final_stats.average_latency);
    println!(
        "🚀 Success rate: {:.1}%",
        (final_stats.messages_processed as f64 / final_stats.messages_received as f64) * 100.0
    );

    if !final_stats.errors.is_empty() {
        println!("\n⚠️ Errors encountered:");
        for error in &final_stats.errors[..std::cmp::min(5, final_stats.errors.len())] {
            println!("  - {}", error);
        }
        if final_stats.errors.len() > 5 {
            println!("  ... and {} more", final_stats.errors.len() - 5);
        }
    }

    println!("\n✨ Streaming demo completed successfully!");

    Ok(())
}
