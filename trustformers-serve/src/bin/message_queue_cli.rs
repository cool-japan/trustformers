use anyhow::Result;
use chrono::Utc;
use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::time::Duration;
use tokio::time::sleep;
use trustformers_serve::message_queue::{
    CompressionAlgorithm, Message, MessageBatch, MessageQueueBackend, MessageQueueConfig,
    MessageQueueManager, PerformanceConfig, RetryPolicy, SecurityConfig, SerializationFormat,
};
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "message_queue_cli")]
#[command(about = "A CLI tool for interacting with message queues")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Send a message to a queue
    Send {
        /// Message queue backend (kafka, rabbitmq, redis, nats, sqs, inmemory)
        #[arg(short, long, default_value = "inmemory")]
        backend: String,

        /// Connection string
        #[arg(short, long, default_value = "localhost")]
        connection: String,

        /// Topic to send to
        #[arg(short, long, default_value = "test-topic")]
        topic: String,

        /// Message key
        #[arg(short, long)]
        key: Option<String>,

        /// Message payload (or path to file with @)
        #[arg(short, long)]
        payload: String,

        /// JSON headers (e.g., '{"user-id": "123"}')
        #[arg(long)]
        headers: Option<String>,

        /// Correlation ID
        #[arg(long)]
        correlation_id: Option<String>,

        /// Reply-to topic
        #[arg(long)]
        reply_to: Option<String>,
    },

    /// Send a batch of messages
    Batch {
        /// Message queue backend
        #[arg(short, long, default_value = "inmemory")]
        backend: String,

        /// Connection string
        #[arg(short, long, default_value = "localhost")]
        connection: String,

        /// Topic to send to
        #[arg(short, long, default_value = "test-topic")]
        topic: String,

        /// Number of messages to send
        #[arg(short, long, default_value = "10")]
        count: usize,

        /// Message prefix
        #[arg(short, long, default_value = "Message")]
        prefix: String,
    },

    /// Consume messages from a queue
    Consume {
        /// Message queue backend
        #[arg(short, long, default_value = "inmemory")]
        backend: String,

        /// Connection string
        #[arg(short, long, default_value = "localhost")]
        connection: String,

        /// Topics to consume from
        #[arg(short, long, default_value = "test-topic")]
        topics: Vec<String>,

        /// Consumer group
        #[arg(short, long, default_value = "cli-consumer")]
        group: String,

        /// Timeout in milliseconds
        #[arg(long, default_value = "5000")]
        timeout: u64,

        /// Maximum number of messages to consume
        #[arg(short, long)]
        max_messages: Option<usize>,

        /// Auto-commit messages
        #[arg(long, default_value = "true")]
        auto_commit: bool,
    },

    /// Monitor queue health and statistics
    Monitor {
        /// Message queue backend
        #[arg(short, long, default_value = "inmemory")]
        backend: String,

        /// Connection string
        #[arg(short, long, default_value = "localhost")]
        connection: String,

        /// Monitoring interval in seconds
        #[arg(short, long, default_value = "5")]
        interval: u64,

        /// Show detailed stats
        #[arg(long)]
        detailed: bool,
    },

    /// Test queue performance
    Performance {
        /// Message queue backend
        #[arg(short, long, default_value = "inmemory")]
        backend: String,

        /// Connection string
        #[arg(short, long, default_value = "localhost")]
        connection: String,

        /// Topic to test
        #[arg(short, long, default_value = "perf-test")]
        topic: String,

        /// Number of messages to send
        #[arg(short, long, default_value = "1000")]
        count: usize,

        /// Message size in bytes
        #[arg(short, long, default_value = "1024")]
        size: usize,

        /// Number of concurrent producers
        #[arg(long, default_value = "1")]
        producers: usize,

        /// Number of concurrent consumers
        #[arg(long, default_value = "1")]
        consumers: usize,

        /// Test duration in seconds
        #[arg(short, long, default_value = "30")]
        duration: u64,
    },

    /// Load configuration from file
    Config {
        /// Configuration file path
        #[arg(short, long)]
        file: String,

        /// Validate configuration only
        #[arg(long)]
        validate: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Send {
            backend,
            connection,
            topic,
            key,
            payload,
            headers,
            correlation_id,
            reply_to,
        } => {
            send_message(SendMessageParams {
                backend,
                connection,
                topic,
                key,
                payload,
                headers,
                correlation_id,
                reply_to,
            })
            .await?;
        },
        Commands::Batch {
            backend,
            connection,
            topic,
            count,
            prefix,
        } => {
            send_batch(backend, connection, topic, count, prefix).await?;
        },
        Commands::Consume {
            backend,
            connection,
            topics,
            group,
            timeout,
            max_messages,
            auto_commit,
        } => {
            consume_messages(
                backend,
                connection,
                topics,
                group,
                timeout,
                max_messages,
                auto_commit,
            )
            .await?;
        },
        Commands::Monitor {
            backend,
            connection,
            interval,
            detailed,
        } => {
            monitor_queue(backend, connection, interval, detailed).await?;
        },
        Commands::Performance {
            backend,
            connection,
            topic,
            count,
            size,
            producers,
            consumers,
            duration,
        } => {
            performance_test(PerformanceTestParams {
                backend,
                connection,
                topic,
                count,
                size,
                producers,
                consumers,
                duration,
            })
            .await?;
        },
        Commands::Config { file, validate } => {
            handle_config(file, validate).await?;
        },
    }

    Ok(())
}

/// Send message parameters
struct SendMessageParams {
    backend: String,
    connection: String,
    topic: String,
    key: Option<String>,
    payload: String,
    headers: Option<String>,
    correlation_id: Option<String>,
    reply_to: Option<String>,
}

async fn send_message(params: SendMessageParams) -> Result<()> {
    let SendMessageParams {
        backend,
        connection,
        topic,
        key,
        payload,
        headers,
        correlation_id,
        reply_to,
    } = params;
    let config = create_config(&backend, &connection, vec![topic.clone()])?;
    let manager = MessageQueueManager::new(config).await?;

    // Parse payload (check if it's a file path starting with @)
    let payload_bytes = if let Some(file_path) = payload.strip_prefix('@') {
        fs::read(file_path)?
    } else {
        payload.into_bytes()
    };

    // Parse headers if provided
    let headers_map = if let Some(headers_str) = headers {
        serde_json::from_str::<HashMap<String, String>>(&headers_str)?
    } else {
        HashMap::new()
    };

    let message = Message {
        id: Uuid::new_v4(),
        topic,
        key,
        payload: payload_bytes,
        headers: headers_map,
        timestamp: Utc::now(),
        partition: None,
        offset: None,
        delivery_count: 1,
        correlation_id,
        reply_to,
    };

    let result = manager.send_message(message).await?;

    println!("✅ Message sent successfully!");
    println!("📧 Message ID: {}", result.message_id);
    println!("📨 Topic: {}", result.topic);
    println!("📍 Partition: {}", result.partition);
    println!("📊 Offset: {}", result.offset);
    println!("📏 Size: {} bytes", result.size);
    println!("⏰ Timestamp: {}", result.timestamp);

    manager.close().await?;
    Ok(())
}

async fn send_batch(
    backend: String,
    connection: String,
    topic: String,
    count: usize,
    prefix: String,
) -> Result<()> {
    let config = create_config(&backend, &connection, vec![topic.clone()])?;
    let manager = MessageQueueManager::new(config).await?;

    let messages: Vec<Message> = (0..count)
        .map(|i| {
            let mut headers = HashMap::new();
            headers.insert("batch-index".to_string(), i.to_string());
            headers.insert("batch-total".to_string(), count.to_string());

            Message {
                id: Uuid::new_v4(),
                topic: topic.clone(),
                key: Some(format!("batch-{}", i)),
                payload: format!("{} {}", prefix, i).into_bytes(),
                headers,
                timestamp: Utc::now(),
                partition: None,
                offset: None,
                delivery_count: 1,
                correlation_id: Some(Uuid::new_v4().to_string()),
                reply_to: None,
            }
        })
        .collect();

    let batch = MessageBatch {
        messages,
        topic,
        batch_id: Uuid::new_v4(),
        created_at: Utc::now(),
    };

    let result = manager.send_batch(batch).await?;

    println!("✅ Batch sent successfully!");
    println!("📦 Batch ID: {}", result.batch_id);
    println!("✅ Success Count: {}", result.success_count);
    println!("❌ Failure Count: {}", result.failure_count);
    println!("📏 Total Size: {} bytes", result.total_size);

    manager.close().await?;
    Ok(())
}

async fn consume_messages(
    backend: String,
    connection: String,
    topics: Vec<String>,
    group: String,
    timeout: u64,
    max_messages: Option<usize>,
    auto_commit: bool,
) -> Result<()> {
    let mut config = create_config(&backend, &connection, topics.clone())?;
    config.consumer_group = Some(group);

    let manager = MessageQueueManager::new(config).await?;

    // Subscribe to topics
    manager.subscribe(&topics).await?;

    println!("🔄 Consuming messages from topics: {:?}", topics);
    println!(
        "👥 Consumer group: {}",
        manager.consumer_group().unwrap_or(&"default".to_string())
    );
    println!("⏱️  Timeout: {}ms", timeout);
    if let Some(max) = max_messages {
        println!("🔢 Max messages: {}", max);
    }
    println!("🚀 Auto-commit: {}", auto_commit);
    println!("Press Ctrl+C to stop...\n");

    let mut message_count = 0;

    loop {
        let messages = manager.consume_messages(timeout).await?;

        if messages.is_empty() {
            print!(".");
            io::stdout().flush()?;
            continue;
        }

        for message in messages {
            message_count += 1;

            println!("\n📥 Message {}: {}", message_count, message.id);
            println!("📨 Topic: {}", message.topic);
            if let Some(key) = &message.key {
                println!("🔑 Key: {}", key);
            }
            println!("📏 Payload size: {} bytes", message.payload.len());
            println!("📄 Payload: {}", String::from_utf8_lossy(&message.payload));

            if !message.headers.is_empty() {
                println!("📋 Headers:");
                for (k, v) in &message.headers {
                    println!("   {}: {}", k, v);
                }
            }

            if let Some(correlation_id) = &message.correlation_id {
                println!("🔗 Correlation ID: {}", correlation_id);
            }

            if let Some(reply_to) = &message.reply_to {
                println!("↩️  Reply To: {}", reply_to);
            }

            println!("⏰ Timestamp: {}", message.timestamp);

            if auto_commit {
                manager.commit_message(&message).await?;
                println!("✅ Message committed");
            }

            if let Some(max) = max_messages {
                if message_count >= max {
                    println!("\n🎯 Reached maximum message count ({})", max);
                    manager.close().await?;
                    return Ok(());
                }
            }
        }
    }
}

async fn monitor_queue(
    backend: String,
    connection: String,
    interval: u64,
    detailed: bool,
) -> Result<()> {
    let config = create_config(&backend, &connection, vec!["monitoring".to_string()])?;
    let manager = MessageQueueManager::new(config).await?;

    println!("📊 Monitoring queue statistics (interval: {}s)", interval);
    println!("Press Ctrl+C to stop...\n");

    let mut iteration = 0;

    loop {
        iteration += 1;

        // Get health status
        let health = manager.health_check().await?;

        // Get statistics
        let stats = manager.get_stats().await;

        println!("=== Iteration {} ===", iteration);
        println!("🏥 Health Status: {:?}", health.status);
        println!("🔗 Connections: {}", health.connection_count);
        println!("📊 Queue Size: {}", health.message_queue_size);
        println!("⚠️  Errors: {}", health.error_count);

        if let Some(timestamp) = health.last_message_timestamp {
            println!("⏰ Last Message: {}", timestamp);
        }

        println!("📈 Statistics:");
        println!("  Messages Produced: {}", stats.messages_produced);
        println!("  Messages Consumed: {}", stats.messages_consumed);
        println!("  Bytes Produced: {}", stats.bytes_produced);
        println!("  Bytes Consumed: {}", stats.bytes_consumed);
        println!("  Errors: {}", stats.errors);
        println!("  Latency: {:.2}ms", stats.latency_ms);
        println!("  Throughput: {:.2} msg/s", stats.throughput_msg_per_sec);

        if detailed {
            println!("  Active Connections: {}", stats.connection_count);
            println!("  Active Transactions: {}", stats.active_transactions);
        }

        println!();

        sleep(Duration::from_secs(interval)).await;
    }
}

/// Performance test parameters
struct PerformanceTestParams {
    backend: String,
    connection: String,
    topic: String,
    count: usize,
    size: usize,
    producers: usize,
    consumers: usize,
    duration: u64,
}

async fn performance_test(params: PerformanceTestParams) -> Result<()> {
    let PerformanceTestParams {
        backend,
        connection,
        topic,
        count,
        size,
        producers,
        consumers,
        duration,
    } = params;
    let config = create_config(&backend, &connection, vec![topic.clone()])?;

    println!("🚀 Starting performance test");
    println!("📊 Configuration:");
    println!("  Backend: {}", backend);
    println!("  Topic: {}", topic);
    println!("  Messages: {}", count);
    println!("  Message Size: {} bytes", size);
    println!("  Producers: {}", producers);
    println!("  Consumers: {}", consumers);
    println!("  Duration: {}s", duration);
    println!();

    let manager = MessageQueueManager::new(config).await?;

    let start_time = std::time::Instant::now();

    // Create test payload
    let payload = vec![b'X'; size];

    // Send messages
    let mut sent_count = 0;
    for i in 0..count {
        let message = Message {
            id: Uuid::new_v4(),
            topic: topic.clone(),
            key: Some(format!("perf-{}", i)),
            payload: payload.clone(),
            headers: {
                let mut headers = HashMap::new();
                headers.insert("test-id".to_string(), "performance".to_string());
                headers.insert("sequence".to_string(), i.to_string());
                headers
            },
            timestamp: Utc::now(),
            partition: None,
            offset: None,
            delivery_count: 1,
            correlation_id: None,
            reply_to: None,
        };

        manager.send_message(message).await?;
        sent_count += 1;

        if i % 100 == 0 {
            print!("📤 Sent: {} messages\r", sent_count);
            io::stdout().flush()?;
        }
    }

    let send_duration = start_time.elapsed();
    let send_rate = sent_count as f64 / send_duration.as_secs_f64();

    println!("\n✅ Sending completed!");
    println!("📊 Send Performance:");
    println!("  Messages sent: {}", sent_count);
    println!("  Time taken: {:.2}s", send_duration.as_secs_f64());
    println!("  Send rate: {:.2} msg/s", send_rate);
    println!(
        "  Throughput: {:.2} MB/s",
        (send_rate * size as f64) / (1024.0 * 1024.0)
    );

    // Get final statistics
    let stats = manager.get_stats().await;
    println!("📈 Final Statistics:");
    println!("  Messages Produced: {}", stats.messages_produced);
    println!("  Messages Consumed: {}", stats.messages_consumed);
    println!("  Bytes Produced: {}", stats.bytes_produced);
    println!("  Bytes Consumed: {}", stats.bytes_consumed);
    println!("  Errors: {}", stats.errors);
    println!("  Latency: {:.2}ms", stats.latency_ms);
    println!("  Throughput: {:.2} msg/s", stats.throughput_msg_per_sec);

    manager.close().await?;
    println!("✅ Performance test completed!");
    Ok(())
}

async fn handle_config(file: String, validate: bool) -> Result<()> {
    let config_content = fs::read_to_string(&file)?;
    let config: MessageQueueConfig = serde_json::from_str(&config_content)?;

    if validate {
        println!("✅ Configuration is valid!");
        println!("📋 Configuration details:");
        println!("  Backend: {:?}", config.backend);
        println!("  Connection: {}", config.connection_string);
        println!("  Topics: {:?}", config.topics);
        println!("  Consumer Group: {:?}", config.consumer_group);
        println!("  Batch Size: {}", config.batch_size);
        println!("  Serialization: {:?}", config.serialization);
        println!("  Compression: {:?}", config.compression);
        return Ok(());
    }

    // Create manager with config
    let manager = MessageQueueManager::new(config).await?;

    // Test connection
    let health = manager.health_check().await?;
    println!("✅ Connection test successful!");
    println!("🏥 Health Status: {:?}", health);

    manager.close().await?;
    Ok(())
}

fn create_config(
    backend: &str,
    connection: &str,
    topics: Vec<String>,
) -> Result<MessageQueueConfig> {
    let backend_enum = match backend.to_lowercase().as_str() {
        "kafka" => MessageQueueBackend::Kafka,
        "rabbitmq" => MessageQueueBackend::RabbitMQ,
        "redis" => MessageQueueBackend::RedisStreams,
        "nats" => MessageQueueBackend::Nats,
        "sqs" => MessageQueueBackend::AmazonSqs,
        "inmemory" => MessageQueueBackend::InMemory,
        _ => return Err(anyhow::anyhow!("Unsupported backend: {}", backend)),
    };

    let connection_string = match backend_enum {
        MessageQueueBackend::Kafka => {
            if connection == "localhost" {
                "localhost:9092".to_string()
            } else {
                connection.to_string()
            }
        },
        MessageQueueBackend::RabbitMQ => {
            if connection == "localhost" {
                "amqp://localhost:5672".to_string()
            } else {
                connection.to_string()
            }
        },
        MessageQueueBackend::RedisStreams => {
            if connection == "localhost" {
                "redis://localhost:6379".to_string()
            } else {
                connection.to_string()
            }
        },
        MessageQueueBackend::Nats => {
            if connection == "localhost" {
                "nats://localhost:4222".to_string()
            } else {
                connection.to_string()
            }
        },
        MessageQueueBackend::AmazonSqs => {
            if connection == "localhost" {
                "https://sqs.us-east-1.amazonaws.com".to_string()
            } else {
                connection.to_string()
            }
        },
        MessageQueueBackend::InMemory => "memory://localhost".to_string(),
    };

    Ok(MessageQueueConfig {
        backend: backend_enum,
        connection_string,
        topics,
        consumer_group: Some("cli-consumer".to_string()),
        batch_size: 100,
        retry_policy: RetryPolicy {
            max_retries: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            exponential_backoff: true,
            jitter: true,
        },
        serialization: SerializationFormat::Json,
        compression: Some(CompressionAlgorithm::Snappy),
        security: SecurityConfig {
            tls_enabled: false,
            ca_cert_path: None,
            client_cert_path: None,
            client_key_path: None,
            sasl_mechanism: None,
            username: None,
            password: None,
        },
        performance: PerformanceConfig {
            connection_pool_size: 5,
            buffer_size: 1048576,
            flush_interval_ms: 1000,
            compression_threshold: 1024,
            prefetch_count: 100,
        },
    })
}
