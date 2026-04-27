#![allow(unused_variables)]
use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
#[cfg(feature = "kafka")]
use trustformers_serve::message_queue::{
    CompressionAlgorithm, PerformanceConfig, RetryPolicy, SecurityConfig, SerializationFormat,
};
use trustformers_serve::message_queue::{
    EventHandler, Message, MessageBatch, MessageQueueBackend, MessageQueueConfig,
    MessageQueueEvent, MessageQueueManager,
};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Run different message queue backend examples
    println!("🚀 Starting Message Queue Demo");
    println!("==============================");

    // Example 1: In-Memory Queue
    println!("\n📦 Example 1: In-Memory Message Queue");
    run_inmemory_example().await?;

    // Example 2: Kafka Queue (requires Kafka server)
    #[cfg(feature = "kafka")]
    {
        println!("\n📡 Example 2: Kafka Message Queue");
        run_kafka_example().await?;
    }

    // Example 3: Redis Streams (requires Redis server)
    println!("\n🔴 Example 3: Redis Streams Message Queue");
    run_redis_example().await?;

    // Example 4: Batch Processing
    println!("\n📊 Example 4: Batch Message Processing");
    run_batch_example().await?;

    // Example 5: Event Handling
    println!("\n🎯 Example 5: Event Handling");
    run_event_handling_example().await?;

    // Example 6: Transactional Processing
    println!("\n🔄 Example 6: Transactional Processing");
    run_transactional_example().await?;

    println!("\n✅ Message Queue Demo Completed Successfully!");
    Ok(())
}

async fn run_inmemory_example() -> Result<()> {
    let config = MessageQueueConfig {
        backend: MessageQueueBackend::InMemory,
        connection_string: "memory://localhost".to_string(),
        topics: vec!["inference-requests".to_string()],
        consumer_group: Some("demo-group".to_string()),
        ..Default::default()
    };

    let manager = MessageQueueManager::new(config).await?;

    // Subscribe to topics
    manager.subscribe(&["inference-requests".to_string()]).await?;

    // Create a sample message
    let message = Message {
        id: Uuid::new_v4(),
        topic: "inference-requests".to_string(),
        key: Some("user-123".to_string()),
        payload: b"Hello, World!".to_vec(),
        headers: {
            let mut headers = HashMap::new();
            headers.insert("user-id".to_string(), "123".to_string());
            headers.insert("request-type".to_string(), "inference".to_string());
            headers
        },
        timestamp: Utc::now(),
        partition: None,
        offset: None,
        delivery_count: 1,
        correlation_id: Some(Uuid::new_v4().to_string()),
        reply_to: Some("response-topic".to_string()),
    };

    // Send message
    let result = manager.send_message(message).await?;
    println!("✅ Message sent successfully: {:?}", result);

    // Consume messages
    let messages = manager.consume_messages(1000).await?;
    println!("📥 Received {} messages", messages.len());

    for msg in messages {
        println!(
            "Message: {} - {}",
            msg.id,
            String::from_utf8_lossy(&msg.payload)
        );
        manager.commit_message(&msg).await?;
    }

    // Get stats
    let stats = manager.get_stats().await;
    println!("📊 Queue Stats: {:?}", stats);

    // Health check
    let health = manager.health_check().await?;
    println!("🏥 Health Status: {:?}", health);

    manager.close().await?;
    println!("✅ In-Memory example completed");
    Ok(())
}

#[cfg(feature = "kafka")]
async fn run_kafka_example() -> Result<()> {
    let config = MessageQueueConfig {
        backend: MessageQueueBackend::Kafka,
        connection_string: "localhost:9092".to_string(),
        topics: vec!["inference-requests".to_string()],
        consumer_group: Some("demo-group".to_string()),
        batch_size: 50,
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
    };

    // For demo purposes, we'll just create the manager
    // In a real scenario, you would need a running Kafka server
    let manager = MessageQueueManager::new(config).await;

    match manager {
        Ok(manager) => {
            println!("✅ Kafka manager created successfully");
            let health = manager.health_check().await?;
            println!("🏥 Kafka Health Status: {:?}", health);
            manager.close().await?;
        },
        Err(e) => {
            println!("⚠️  Kafka not available (expected in demo): {}", e);
        },
    }

    println!("✅ Kafka example completed");
    Ok(())
}

async fn run_redis_example() -> Result<()> {
    let config = MessageQueueConfig {
        backend: MessageQueueBackend::RedisStreams,
        connection_string: "redis://localhost:6379".to_string(),
        topics: vec!["inference-stream".to_string()],
        consumer_group: Some("demo-group".to_string()),
        ..Default::default()
    };

    // For demo purposes, we'll just create the manager
    // In a real scenario, you would need a running Redis server
    let manager = MessageQueueManager::new(config).await;

    match manager {
        Ok(manager) => {
            println!("✅ Redis manager created successfully");
            let health = manager.health_check().await?;
            println!("🏥 Redis Health Status: {:?}", health);
            manager.close().await?;
        },
        Err(e) => {
            println!("⚠️  Redis not available (expected in demo): {}", e);
        },
    }

    println!("✅ Redis example completed");
    Ok(())
}

async fn run_batch_example() -> Result<()> {
    let config = MessageQueueConfig {
        backend: MessageQueueBackend::InMemory,
        connection_string: "memory://localhost".to_string(),
        topics: vec!["batch-processing".to_string()],
        batch_size: 10,
        ..Default::default()
    };

    let manager = MessageQueueManager::new(config).await?;

    // Create a batch of messages
    let messages = (0..5)
        .map(|i| Message {
            id: Uuid::new_v4(),
            topic: "batch-processing".to_string(),
            key: Some(format!("key-{}", i)),
            payload: format!("Batch message {}", i).into_bytes(),
            headers: {
                let mut headers = HashMap::new();
                headers.insert("batch-id".to_string(), "batch-001".to_string());
                headers.insert("sequence".to_string(), i.to_string());
                headers
            },
            timestamp: Utc::now(),
            partition: None,
            offset: None,
            delivery_count: 1,
            correlation_id: Some(Uuid::new_v4().to_string()),
            reply_to: None,
        })
        .collect();

    let batch = MessageBatch {
        messages,
        topic: "batch-processing".to_string(),
        batch_id: Uuid::new_v4(),
        created_at: Utc::now(),
    };

    // Send batch
    let result = manager.send_batch(batch).await?;
    println!("✅ Batch sent successfully: {:?}", result);

    // Get stats
    let stats = manager.get_stats().await;
    println!("📊 Batch Stats: {:?}", stats);

    manager.close().await?;
    println!("✅ Batch example completed");
    Ok(())
}

async fn run_event_handling_example() -> Result<()> {
    let config = MessageQueueConfig {
        backend: MessageQueueBackend::InMemory,
        connection_string: "memory://localhost".to_string(),
        topics: vec!["events".to_string()],
        ..Default::default()
    };

    let manager = MessageQueueManager::new(config).await?;

    // Register event handlers
    let message_handler: EventHandler = Box::new(|event| match event {
        MessageQueueEvent::MessageProduced(result) => {
            println!(
                "🎉 Message produced: {} on topic {}",
                result.message_id, result.topic
            );
        },
        MessageQueueEvent::MessageConsumed(message) => {
            println!(
                "📥 Message consumed: {} with payload length {}",
                message.id,
                message.payload.len()
            );
        },
        MessageQueueEvent::BatchProduced(result) => {
            println!(
                "📦 Batch produced: {} with {} messages",
                result.batch_id, result.success_count
            );
        },
        MessageQueueEvent::ConnectionLost => {
            println!("⚠️  Connection lost!");
        },
        MessageQueueEvent::ConnectionRestored => {
            println!("✅ Connection restored!");
        },
        MessageQueueEvent::Error(error) => {
            println!("❌ Error: {}", error);
        },
        _ => {},
    });

    manager
        .register_event_handler("message_handler".to_string(), message_handler)
        .await;

    // Send some messages to trigger events
    for i in 0..3 {
        let message = Message {
            id: Uuid::new_v4(),
            topic: "events".to_string(),
            key: Some(format!("event-{}", i)),
            payload: format!("Event message {}", i).into_bytes(),
            headers: HashMap::new(),
            timestamp: Utc::now(),
            partition: None,
            offset: None,
            delivery_count: 1,
            correlation_id: None,
            reply_to: None,
        };

        manager.send_message(message).await?;
        sleep(Duration::from_millis(100)).await;
    }

    // Subscribe and consume
    manager.subscribe(&["events".to_string()]).await?;
    let messages = manager.consume_messages(1000).await?;

    for msg in messages {
        manager.commit_message(&msg).await?;
    }

    manager.close().await?;
    println!("✅ Event handling example completed");
    Ok(())
}

async fn run_transactional_example() -> Result<()> {
    let config = MessageQueueConfig {
        backend: MessageQueueBackend::InMemory,
        connection_string: "memory://localhost".to_string(),
        topics: vec!["transactions".to_string()],
        ..Default::default()
    };

    let manager = MessageQueueManager::new(config).await?;

    // Begin transaction
    let transaction_id = manager.begin_transaction().await?;
    println!("🔄 Transaction started: {}", transaction_id);

    // Send messages within transaction
    for i in 0..3 {
        let message = Message {
            id: Uuid::new_v4(),
            topic: "transactions".to_string(),
            key: Some(format!("tx-{}", i)),
            payload: format!("Transactional message {}", i).into_bytes(),
            headers: {
                let mut headers = HashMap::new();
                headers.insert("transaction-id".to_string(), transaction_id.clone());
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
    }

    // Commit transaction
    manager.commit_transaction(transaction_id.clone()).await?;
    println!("✅ Transaction committed: {}", transaction_id);

    // Get stats
    let stats = manager.get_stats().await;
    println!("📊 Transaction Stats: {:?}", stats);

    manager.close().await?;
    println!("✅ Transactional example completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_inmemory_queue() {
        let result = run_inmemory_example().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let result = run_batch_example().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_event_handling() {
        let result = run_event_handling_example().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_transactional_processing() {
        let result = run_transactional_example().await;
        assert!(result.is_ok());
    }
}
