#[cfg(feature = "kafka")]
use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageQueueConfig {
    pub backend: MessageQueueBackend,
    pub connection_string: String,
    pub topics: Vec<String>,
    pub consumer_group: Option<String>,
    pub batch_size: usize,
    pub retry_policy: RetryPolicy,
    pub serialization: SerializationFormat,
    pub compression: Option<CompressionAlgorithm>,
    pub security: SecurityConfig,
    pub performance: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageQueueBackend {
    #[cfg(feature = "kafka")]
    Kafka,
    RabbitMQ,
    RedisStreams,
    Nats,
    AmazonSqs,
    InMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub exponential_backoff: bool,
    pub jitter: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializationFormat {
    Json,
    Avro,
    Protobuf,
    MessagePack,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Snappy,
    Lz4,
    Zstd,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub tls_enabled: bool,
    pub ca_cert_path: Option<String>,
    pub client_cert_path: Option<String>,
    pub client_key_path: Option<String>,
    pub sasl_mechanism: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub connection_pool_size: usize,
    pub buffer_size: usize,
    pub flush_interval_ms: u64,
    pub compression_threshold: usize,
    pub prefetch_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub topic: String,
    pub key: Option<String>,
    pub payload: Vec<u8>,
    pub headers: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub partition: Option<u32>,
    pub offset: Option<u64>,
    pub delivery_count: u32,
    pub correlation_id: Option<String>,
    pub reply_to: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageBatch {
    pub messages: Vec<Message>,
    pub topic: String,
    pub batch_id: Uuid,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerConfig {
    pub group_id: String,
    pub auto_offset_reset: AutoOffsetReset,
    pub enable_auto_commit: bool,
    pub auto_commit_interval_ms: u64,
    pub session_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
    pub max_poll_records: usize,
    pub fetch_max_wait_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutoOffsetReset {
    Earliest,
    Latest,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProducerConfig {
    pub acks: AcknowledgmentMode,
    pub retries: u32,
    pub batch_size: usize,
    pub linger_ms: u64,
    pub buffer_memory: usize,
    pub compression_type: Option<CompressionAlgorithm>,
    pub idempotence: bool,
    pub transaction_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcknowledgmentMode {
    None,
    Leader,
    All,
}

#[async_trait]
pub trait MessageQueueProducer: Send + Sync {
    async fn send_message(&self, message: Message) -> Result<MessageResult>;
    async fn send_batch(&self, batch: MessageBatch) -> Result<BatchResult>;
    async fn send_with_callback(&self, message: Message, callback: ProducerCallback) -> Result<()>;
    async fn begin_transaction(&self) -> Result<TransactionId>;
    async fn commit_transaction(&self, transaction_id: TransactionId) -> Result<()>;
    async fn abort_transaction(&self, transaction_id: TransactionId) -> Result<()>;
    async fn flush(&self) -> Result<()>;
    async fn close(&self) -> Result<()>;
}

#[async_trait]
pub trait MessageQueueConsumer: Send + Sync {
    async fn subscribe(&self, topics: &[String]) -> Result<()>;
    async fn unsubscribe(&self, topics: &[String]) -> Result<()>;
    async fn poll(&self, timeout_ms: u64) -> Result<Vec<Message>>;
    async fn commit(&self, message: &Message) -> Result<()>;
    async fn commit_batch(&self, messages: &[Message]) -> Result<()>;
    async fn seek(&self, topic: &str, partition: u32, offset: u64) -> Result<()>;
    async fn pause(&self, topics: &[String]) -> Result<()>;
    async fn resume(&self, topics: &[String]) -> Result<()>;
    async fn close(&self) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct MessageResult {
    pub message_id: Uuid,
    pub topic: String,
    pub partition: u32,
    pub offset: u64,
    pub timestamp: DateTime<Utc>,
    pub size: usize,
}

#[derive(Debug, Clone)]
pub struct BatchResult {
    pub batch_id: Uuid,
    pub results: Vec<MessageResult>,
    pub success_count: usize,
    pub failure_count: usize,
    pub total_size: usize,
}

pub type ProducerCallback = Box<dyn Fn(Result<MessageResult>) + Send + Sync>;
pub type TransactionId = String;

#[derive(Debug, Clone, serde::Serialize)]
pub struct MessageQueueStats {
    pub messages_produced: u64,
    pub messages_consumed: u64,
    pub bytes_produced: u64,
    pub bytes_consumed: u64,
    pub errors: u64,
    pub latency_ms: f64,
    pub throughput_msg_per_sec: f64,
    pub connection_count: u32,
    pub active_transactions: u32,
}

pub struct MessageQueueManager {
    config: MessageQueueConfig,
    producer: Arc<dyn MessageQueueProducer>,
    consumer: Arc<dyn MessageQueueConsumer>,
    stats: Arc<RwLock<MessageQueueStats>>,
    event_handlers: Arc<RwLock<HashMap<String, EventHandler>>>,
    /// Timestamp of the last message this manager produced or consumed.
    /// `None` until a message really passes through.
    last_message_at: Arc<RwLock<Option<DateTime<Utc>>>>,
}

pub type EventHandler = Box<dyn Fn(MessageQueueEvent) + Send + Sync>;

#[derive(Debug, Clone)]
pub enum MessageQueueEvent {
    MessageProduced(MessageResult),
    MessageConsumed(Message),
    BatchProduced(BatchResult),
    BatchConsumed(MessageBatch),
    ConnectionLost,
    ConnectionRestored,
    Error(String),
    PartitionRebalanced,
    TransactionCommitted(TransactionId),
    TransactionAborted(TransactionId),
}

impl MessageQueueManager {
    pub async fn new(config: MessageQueueConfig) -> Result<Self> {
        let (producer, consumer) = Self::create_backend_clients(&config).await?;

        Ok(Self {
            config,
            producer,
            consumer,
            stats: Arc::new(RwLock::new(MessageQueueStats::default())),
            event_handlers: Arc::new(RwLock::new(HashMap::new())),
            last_message_at: Arc::new(RwLock::new(None)),
        })
    }

    async fn create_backend_clients(
        config: &MessageQueueConfig,
    ) -> Result<(Arc<dyn MessageQueueProducer>, Arc<dyn MessageQueueConsumer>)> {
        match config.backend {
            #[cfg(feature = "kafka")]
            MessageQueueBackend::Kafka => Self::create_kafka_clients(config).await,
            MessageQueueBackend::RabbitMQ => Self::create_rabbitmq_clients(config).await,
            MessageQueueBackend::RedisStreams => Self::create_redis_clients(config).await,
            MessageQueueBackend::Nats => Self::create_nats_clients(config).await,
            MessageQueueBackend::AmazonSqs => Self::create_sqs_clients(config).await,
            MessageQueueBackend::InMemory => Self::create_inmemory_clients(config).await,
        }
    }

    #[cfg(feature = "kafka")]
    async fn create_kafka_clients(
        config: &MessageQueueConfig,
    ) -> Result<(Arc<dyn MessageQueueProducer>, Arc<dyn MessageQueueConsumer>)> {
        let producer = Arc::new(KafkaProducer::new(config).await?);
        let consumer = Arc::new(KafkaConsumer::new(config).await?);
        Ok((producer, consumer))
    }

    async fn create_rabbitmq_clients(
        config: &MessageQueueConfig,
    ) -> Result<(Arc<dyn MessageQueueProducer>, Arc<dyn MessageQueueConsumer>)> {
        let producer = Arc::new(RabbitMQProducer::new(config).await?);
        let consumer = Arc::new(RabbitMQConsumer::new(config).await?);
        Ok((producer, consumer))
    }

    async fn create_redis_clients(
        config: &MessageQueueConfig,
    ) -> Result<(Arc<dyn MessageQueueProducer>, Arc<dyn MessageQueueConsumer>)> {
        let producer = Arc::new(RedisProducer::new(config).await?);
        let consumer = Arc::new(RedisConsumer::new(config).await?);
        Ok((producer, consumer))
    }

    async fn create_nats_clients(
        config: &MessageQueueConfig,
    ) -> Result<(Arc<dyn MessageQueueProducer>, Arc<dyn MessageQueueConsumer>)> {
        let producer = Arc::new(NatsProducer::new(config).await?);
        let consumer = Arc::new(NatsConsumer::new(config).await?);
        Ok((producer, consumer))
    }

    async fn create_sqs_clients(
        config: &MessageQueueConfig,
    ) -> Result<(Arc<dyn MessageQueueProducer>, Arc<dyn MessageQueueConsumer>)> {
        let producer = Arc::new(SqsProducer::new(config).await?);
        let consumer = Arc::new(SqsConsumer::new(config).await?);
        Ok((producer, consumer))
    }

    async fn create_inmemory_clients(
        config: &MessageQueueConfig,
    ) -> Result<(Arc<dyn MessageQueueProducer>, Arc<dyn MessageQueueConsumer>)> {
        let producer = Arc::new(InMemoryProducer::new(config).await?);
        let consumer = Arc::new(InMemoryConsumer::new(config).await?);
        Ok((producer, consumer))
    }

    pub async fn send_message(&self, message: Message) -> Result<MessageResult> {
        let result = match self.producer.send_message(message).await {
            Ok(result) => result,
            Err(e) => {
                self.record_error(&e).await;
                return Err(e);
            },
        };
        self.update_stats_produced(&result).await;
        self.emit_event(MessageQueueEvent::MessageProduced(result.clone())).await;
        Ok(result)
    }

    pub async fn send_batch(&self, batch: MessageBatch) -> Result<BatchResult> {
        let result = self.producer.send_batch(batch).await?;
        self.update_stats_batch_produced(&result).await;
        self.emit_event(MessageQueueEvent::BatchProduced(result.clone())).await;
        Ok(result)
    }

    pub async fn consume_messages(&self, timeout_ms: u64) -> Result<Vec<Message>> {
        let messages = match self.consumer.poll(timeout_ms).await {
            Ok(messages) => messages,
            Err(e) => {
                self.record_error(&e).await;
                return Err(e);
            },
        };
        for message in &messages {
            self.update_stats_consumed(message).await;
            self.emit_event(MessageQueueEvent::MessageConsumed(message.clone())).await;
        }
        Ok(messages)
    }

    pub async fn subscribe(&self, topics: &[String]) -> Result<()> {
        self.consumer.subscribe(topics).await
    }

    pub async fn commit_message(&self, message: &Message) -> Result<()> {
        self.consumer.commit(message).await
    }

    pub async fn register_event_handler(&self, name: String, handler: EventHandler) {
        let mut handlers = self.event_handlers.write().await;
        handlers.insert(name, handler);
    }

    /// Begin a transaction
    pub async fn begin_transaction(&self) -> Result<TransactionId> {
        self.producer.begin_transaction().await
    }

    /// Commit a transaction
    pub async fn commit_transaction(&self, transaction_id: TransactionId) -> Result<()> {
        self.producer.commit_transaction(transaction_id).await
    }

    /// Abort a transaction
    pub async fn abort_transaction(&self, transaction_id: TransactionId) -> Result<()> {
        self.producer.abort_transaction(transaction_id).await
    }

    /// Count a real failure and surface it to registered event handlers.
    async fn record_error(&self, error: &anyhow::Error) {
        self.stats.write().await.errors += 1;
        self.emit_event(MessageQueueEvent::Error(error.to_string())).await;
    }

    async fn emit_event(&self, event: MessageQueueEvent) {
        let handlers = self.event_handlers.read().await;
        for handler in handlers.values() {
            handler(event.clone());
        }
    }

    async fn update_stats_produced(&self, result: &MessageResult) {
        {
            let mut stats = self.stats.write().await;
            stats.messages_produced += 1;
            stats.bytes_produced += result.size as u64;
        }
        *self.last_message_at.write().await = Some(result.timestamp);
    }

    async fn update_stats_batch_produced(&self, result: &BatchResult) {
        let mut stats = self.stats.write().await;
        stats.messages_produced += result.success_count as u64;
        stats.bytes_produced += result.total_size as u64;
    }

    async fn update_stats_consumed(&self, message: &Message) {
        {
            let mut stats = self.stats.write().await;
            stats.messages_consumed += 1;
            stats.bytes_consumed += message.payload.len() as u64;
        }
        *self.last_message_at.write().await = Some(message.timestamp);
    }

    pub async fn get_stats(&self) -> MessageQueueStats {
        self.stats.read().await.clone()
    }

    /// Report the manager's observed state.
    ///
    /// Every field is derived from what this manager has actually seen: the
    /// backlog is produced-minus-consumed, the error count comes from the
    /// counters, and `last_message_timestamp` is `None` until a message really
    /// passes through. Nothing is asserted about broker-side health that this
    /// process has not observed.
    ///
    /// # Errors
    ///
    /// This call does not fail; the signature is kept for API stability.
    pub async fn health_check(&self) -> Result<MessageQueueHealth> {
        let stats = self.stats.read().await.clone();
        let last_message_timestamp = *self.last_message_at.read().await;

        let status = if stats.errors == 0 {
            HealthStatus::Healthy
        } else if stats.messages_produced > 0 || stats.messages_consumed > 0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };

        Ok(MessageQueueHealth {
            backend: self.config.backend.clone(),
            status,
            // One producer client and one consumer client are held open.
            connection_count: 2,
            message_queue_size: stats.messages_produced.saturating_sub(stats.messages_consumed),
            last_message_timestamp,
            error_count: stats.errors,
        })
    }

    pub async fn close(&self) -> Result<()> {
        self.producer.close().await?;
        self.consumer.close().await?;
        Ok(())
    }

    /// Get a reference to the message queue configuration
    pub fn config(&self) -> &MessageQueueConfig {
        &self.config
    }

    /// Get the consumer group name
    pub fn consumer_group(&self) -> Option<&String> {
        self.config.consumer_group.as_ref()
    }

    /// Get the backend type
    pub fn backend(&self) -> &MessageQueueBackend {
        &self.config.backend
    }

    /// Get the configured topics
    pub fn topics(&self) -> &[String] {
        &self.config.topics
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageQueueHealth {
    pub backend: MessageQueueBackend,
    pub status: HealthStatus,
    pub connection_count: u32,
    pub message_queue_size: u64,
    pub last_message_timestamp: Option<DateTime<Utc>>,
    pub error_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl Default for MessageQueueStats {
    fn default() -> Self {
        Self {
            messages_produced: 0,
            messages_consumed: 0,
            bytes_produced: 0,
            bytes_consumed: 0,
            errors: 0,
            latency_ms: 0.0,
            throughput_msg_per_sec: 0.0,
            connection_count: 0,
            active_transactions: 0,
        }
    }
}

impl Default for MessageQueueConfig {
    fn default() -> Self {
        Self {
            backend: MessageQueueBackend::InMemory,
            connection_string: "localhost:9092".to_string(),
            topics: vec!["inference-requests".to_string()],
            consumer_group: Some("trustformers-serve".to_string()),
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
        }
    }
}

// Kafka implementation
#[cfg(feature = "kafka")]
struct KafkaProducer {
    producer: rdkafka::producer::FutureProducer,
}

#[cfg(feature = "kafka")]
impl KafkaProducer {
    async fn new(config: &MessageQueueConfig) -> Result<Self> {
        use rdkafka::config::ClientConfig;
        use rdkafka::producer::FutureProducer;

        let mut client_config = ClientConfig::new();
        client_config.set("bootstrap.servers", &config.connection_string);
        client_config.set("message.timeout.ms", "30000");
        client_config.set("queue.buffering.max.messages", "100000");

        if let Some(username) = &config.security.username {
            client_config.set("sasl.username", username);
        }
        if let Some(password) = &config.security.password {
            client_config.set("sasl.password", password);
        }

        let producer: FutureProducer =
            client_config.create().context("Failed to create Kafka producer")?;

        Ok(Self { producer })
    }
}

#[cfg(feature = "kafka")]
#[async_trait]
impl MessageQueueProducer for KafkaProducer {
    async fn send_message(&self, message: Message) -> Result<MessageResult> {
        use rdkafka::producer::FutureRecord;

        let mut record = FutureRecord::to(&message.topic).payload(&message.payload);

        if let Some(key) = &message.key {
            record = record.key(key);
        }

        let delivery_result = self.producer.send(record, std::time::Duration::from_secs(10)).await;

        match delivery_result {
            Ok(delivery) => {
                // Delivery struct has partition and offset fields
                Ok(MessageResult {
                    message_id: message.id,
                    topic: message.topic,
                    partition: delivery.partition as u32,
                    offset: delivery.offset as u64,
                    timestamp: Utc::now(),
                    size: message.payload.len(),
                })
            },
            Err((kafka_error, _)) => {
                anyhow::bail!("Kafka send failed: {}", kafka_error)
            },
        }
    }

    async fn send_batch(&self, batch: MessageBatch) -> Result<BatchResult> {
        let mut results = Vec::new();
        let mut success_count = 0;
        let mut failure_count = 0;
        let mut total_size = 0;

        for message in batch.messages {
            match self.send_message(message).await {
                Ok(result) => {
                    total_size += result.size;
                    results.push(result);
                    success_count += 1;
                },
                Err(_) => {
                    failure_count += 1;
                },
            }
        }

        Ok(BatchResult {
            batch_id: batch.batch_id,
            results,
            success_count,
            failure_count,
            total_size,
        })
    }

    async fn send_with_callback(&self, message: Message, callback: ProducerCallback) -> Result<()> {
        let result = self.send_message(message).await;
        callback(result);
        Ok(())
    }

    async fn begin_transaction(&self) -> Result<TransactionId> {
        Ok(Uuid::new_v4().to_string())
    }

    async fn commit_transaction(&self, _transaction_id: TransactionId) -> Result<()> {
        Ok(())
    }

    async fn abort_transaction(&self, _transaction_id: TransactionId) -> Result<()> {
        Ok(())
    }

    async fn flush(&self) -> Result<()> {
        use rdkafka::producer::Producer;
        let _ = self.producer.flush(std::time::Duration::from_secs(10));
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(feature = "kafka")]
struct KafkaConsumer {
    consumer: rdkafka::consumer::StreamConsumer,
}

#[cfg(feature = "kafka")]
impl KafkaConsumer {
    async fn new(config: &MessageQueueConfig) -> Result<Self> {
        use rdkafka::config::ClientConfig;
        use rdkafka::consumer::StreamConsumer;

        let mut client_config = ClientConfig::new();
        client_config.set("bootstrap.servers", &config.connection_string);
        client_config.set(
            "group.id",
            config.consumer_group.as_deref().unwrap_or("default"),
        );
        client_config.set("enable.auto.commit", "true");
        client_config.set("auto.offset.reset", "earliest");

        let consumer: StreamConsumer =
            client_config.create().context("Failed to create Kafka consumer")?;

        Ok(Self { consumer })
    }
}

#[cfg(feature = "kafka")]
#[async_trait]
impl MessageQueueConsumer for KafkaConsumer {
    async fn subscribe(&self, topics: &[String]) -> Result<()> {
        use rdkafka::consumer::Consumer;
        let topics_refs: Vec<&str> = topics.iter().map(|s| s.as_str()).collect();
        self.consumer
            .subscribe(&topics_refs)
            .context("Failed to subscribe to Kafka topics")?;
        Ok(())
    }

    async fn unsubscribe(&self, _topics: &[String]) -> Result<()> {
        use rdkafka::consumer::Consumer;
        self.consumer.unsubscribe();
        Ok(())
    }

    async fn poll(&self, timeout_ms: u64) -> Result<Vec<Message>> {
        use rdkafka::message::Headers;
        use rdkafka::message::Message as KafkaMessage;

        let _timeout = std::time::Duration::from_millis(timeout_ms);
        let message = self.consumer.recv().await.context("Failed to receive message from Kafka")?;

        let mut headers = HashMap::new();
        if let Some(kafka_headers) = message.headers() {
            for header in kafka_headers.iter() {
                if let Ok(value) = std::str::from_utf8(header.value.unwrap_or(b"")) {
                    headers.insert(header.key.to_string(), value.to_string());
                }
            }
        }

        let msg = Message {
            id: Uuid::new_v4(),
            topic: message.topic().to_string(),
            key: message.key().map(|k| String::from_utf8_lossy(k).to_string()),
            payload: message.payload().unwrap_or(b"").to_vec(),
            headers,
            timestamp: Utc::now(),
            partition: Some(message.partition() as u32),
            offset: Some(message.offset() as u64),
            delivery_count: 1,
            correlation_id: None,
            reply_to: None,
        };

        Ok(vec![msg])
    }

    async fn commit(&self, _message: &Message) -> Result<()> {
        use rdkafka::consumer::Consumer;
        self.consumer
            .commit_consumer_state(rdkafka::consumer::CommitMode::Async)
            .context("Failed to commit Kafka message")?;
        Ok(())
    }

    async fn commit_batch(&self, _messages: &[Message]) -> Result<()> {
        use rdkafka::consumer::Consumer;
        self.consumer
            .commit_consumer_state(rdkafka::consumer::CommitMode::Async)
            .context("Failed to commit Kafka batch")?;
        Ok(())
    }

    async fn seek(&self, _topic: &str, _partition: u32, _offset: u64) -> Result<()> {
        Ok(())
    }

    async fn pause(&self, _topics: &[String]) -> Result<()> {
        Ok(())
    }

    async fn resume(&self, _topics: &[String]) -> Result<()> {
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

// Real backend implementations. Each lives in its own file under
// `message_queue/` and performs genuine broker I/O; none of them fabricates a
// send result or polls an empty vector unconditionally.
pub mod inmemory;
pub mod nats;
pub mod rabbitmq;
pub mod redis_streams;
pub mod sqs;

pub use inmemory::{InMemoryBroker, InMemoryConsumer, InMemoryProducer};
pub use nats::{NatsConsumer, NatsProducer};
pub use rabbitmq::{RabbitMQConsumer, RabbitMQProducer};
pub use redis_streams::{RedisConsumer, RedisProducer};
pub use sqs::{SqsConsumer, SqsProducer};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_default_config() -> MessageQueueConfig {
        MessageQueueConfig::default()
    }

    fn make_test_message(topic: &str, payload: &[u8]) -> Message {
        Message {
            id: Uuid::new_v4(),
            topic: topic.to_string(),
            key: None,
            payload: payload.to_vec(),
            headers: HashMap::new(),
            timestamp: Utc::now(),
            partition: None,
            offset: None,
            delivery_count: 0,
            correlation_id: None,
            reply_to: None,
        }
    }

    fn make_test_batch(topic: &str, messages: Vec<Message>) -> MessageBatch {
        MessageBatch {
            messages,
            topic: topic.to_string(),
            batch_id: Uuid::new_v4(),
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_default_config_backend_is_inmemory() {
        let config = make_default_config();
        assert!(matches!(config.backend, MessageQueueBackend::InMemory));
    }

    #[test]
    fn test_default_config_topics_nonempty() {
        let config = make_default_config();
        assert!(!config.topics.is_empty());
    }

    #[test]
    fn test_default_config_batch_size() {
        let config = make_default_config();
        assert_eq!(config.batch_size, 100);
    }

    #[test]
    fn test_default_config_consumer_group_present() {
        let config = make_default_config();
        assert!(config.consumer_group.is_some());
    }

    #[test]
    fn test_default_config_retry_policy() {
        let config = make_default_config();
        assert_eq!(config.retry_policy.max_retries, 3);
        assert!(config.retry_policy.exponential_backoff);
    }

    #[test]
    fn test_default_config_tls_disabled() {
        let config = make_default_config();
        assert!(!config.security.tls_enabled);
    }

    #[test]
    fn test_stats_default_values() {
        let stats = MessageQueueStats::default();
        assert_eq!(stats.messages_produced, 0);
        assert_eq!(stats.messages_consumed, 0);
        assert_eq!(stats.bytes_produced, 0);
        assert_eq!(stats.bytes_consumed, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.connection_count, 0);
        assert_eq!(stats.active_transactions, 0);
    }

    #[test]
    fn test_message_creation() {
        let msg = make_test_message("test-topic", b"hello world");
        assert_eq!(msg.topic, "test-topic");
        assert_eq!(msg.payload, b"hello world");
        assert_eq!(msg.delivery_count, 0);
        assert!(msg.key.is_none());
        assert!(msg.partition.is_none());
        assert!(msg.offset.is_none());
        assert!(msg.correlation_id.is_none());
        assert!(msg.reply_to.is_none());
    }

    #[test]
    fn test_message_with_key() {
        let mut msg = make_test_message("topic", b"data");
        msg.key = Some("partition-key".to_string());
        assert_eq!(msg.key.as_deref(), Some("partition-key"));
    }

    #[test]
    fn test_message_headers() {
        let mut msg = make_test_message("topic", b"data");
        msg.headers.insert("content-type".to_string(), "application/json".to_string());
        assert_eq!(
            msg.headers.get("content-type").map(|s| s.as_str()),
            Some("application/json")
        );
    }

    #[test]
    fn test_message_batch_creation() {
        let msgs = vec![
            make_test_message("t", b"a"),
            make_test_message("t", b"b"),
            make_test_message("t", b"c"),
        ];
        let batch = make_test_batch("t", msgs);
        assert_eq!(batch.messages.len(), 3);
        assert_eq!(batch.topic, "t");
    }

    #[test]
    fn test_batch_result_counts() {
        let result = BatchResult {
            batch_id: Uuid::new_v4(),
            results: vec![],
            success_count: 7,
            failure_count: 2,
            total_size: 1024,
        };
        assert_eq!(result.success_count, 7);
        assert_eq!(result.failure_count, 2);
        assert_eq!(result.total_size, 1024);
    }

    #[test]
    fn test_message_result_fields() {
        let result = MessageResult {
            message_id: Uuid::new_v4(),
            topic: "my-topic".to_string(),
            partition: 3,
            offset: 42,
            timestamp: Utc::now(),
            size: 256,
        };
        assert_eq!(result.topic, "my-topic");
        assert_eq!(result.partition, 3);
        assert_eq!(result.offset, 42);
        assert_eq!(result.size, 256);
    }

    #[test]
    fn test_health_status_variants() {
        let healthy = HealthStatus::Healthy;
        let degraded = HealthStatus::Degraded;
        let unhealthy = HealthStatus::Unhealthy;
        assert!(matches!(healthy, HealthStatus::Healthy));
        assert!(matches!(degraded, HealthStatus::Degraded));
        assert!(matches!(unhealthy, HealthStatus::Unhealthy));
    }

    #[test]
    fn test_auto_offset_reset_variants() {
        let earliest = AutoOffsetReset::Earliest;
        let latest = AutoOffsetReset::Latest;
        let none = AutoOffsetReset::None;
        assert!(matches!(earliest, AutoOffsetReset::Earliest));
        assert!(matches!(latest, AutoOffsetReset::Latest));
        assert!(matches!(none, AutoOffsetReset::None));
    }

    #[test]
    fn test_acknowledgment_mode_variants() {
        assert!(matches!(AcknowledgmentMode::None, AcknowledgmentMode::None));
        assert!(matches!(
            AcknowledgmentMode::Leader,
            AcknowledgmentMode::Leader
        ));
        assert!(matches!(AcknowledgmentMode::All, AcknowledgmentMode::All));
    }

    #[test]
    fn test_compression_algorithm_variants() {
        assert!(matches!(
            CompressionAlgorithm::Gzip,
            CompressionAlgorithm::Gzip
        ));
        assert!(matches!(
            CompressionAlgorithm::Snappy,
            CompressionAlgorithm::Snappy
        ));
        assert!(matches!(
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Lz4
        ));
        assert!(matches!(
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Zstd
        ));
    }

    #[test]
    fn test_serialization_format_variants() {
        assert!(matches!(
            SerializationFormat::Json,
            SerializationFormat::Json
        ));
        assert!(matches!(
            SerializationFormat::Avro,
            SerializationFormat::Avro
        ));
        assert!(matches!(
            SerializationFormat::Protobuf,
            SerializationFormat::Protobuf
        ));
        assert!(matches!(
            SerializationFormat::MessagePack,
            SerializationFormat::MessagePack
        ));
    }

    #[tokio::test]
    async fn test_inmemory_manager_creation() {
        let config = MessageQueueConfig {
            backend: MessageQueueBackend::InMemory,
            ..Default::default()
        };
        let manager = MessageQueueManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_manager_config_accessor() {
        let config = MessageQueueConfig {
            backend: MessageQueueBackend::InMemory,
            connection_string: "localhost:1234".to_string(),
            ..Default::default()
        };
        let manager =
            MessageQueueManager::new(config).await.expect("manager creation should succeed");
        assert_eq!(manager.config().connection_string, "localhost:1234");
    }

    #[tokio::test]
    async fn test_manager_backend_accessor() {
        let config = make_default_config();
        let manager =
            MessageQueueManager::new(config).await.expect("manager creation should succeed");
        assert!(matches!(manager.backend(), MessageQueueBackend::InMemory));
    }

    #[tokio::test]
    async fn test_manager_topics_accessor() {
        let config = MessageQueueConfig {
            backend: MessageQueueBackend::InMemory,
            topics: vec!["topic-a".to_string(), "topic-b".to_string()],
            ..Default::default()
        };
        let manager =
            MessageQueueManager::new(config).await.expect("manager creation should succeed");
        assert_eq!(manager.topics().len(), 2);
        assert!(manager.topics().contains(&"topic-a".to_string()));
    }

    #[tokio::test]
    async fn test_manager_consumer_group_accessor() {
        let config = MessageQueueConfig {
            backend: MessageQueueBackend::InMemory,
            consumer_group: Some("test-group".to_string()),
            ..Default::default()
        };
        let manager =
            MessageQueueManager::new(config).await.expect("manager creation should succeed");
        assert_eq!(
            manager.consumer_group().map(|s| s.as_str()),
            Some("test-group")
        );
    }

    #[tokio::test]
    async fn test_manager_send_message() {
        let config = make_default_config();
        let manager =
            MessageQueueManager::new(config).await.expect("manager creation should succeed");
        let msg = make_test_message("test-topic", b"test payload");
        let result = manager.send_message(msg).await;
        assert!(result.is_ok());
        let msg_result = result.expect("send should succeed");
        assert_eq!(msg_result.topic, "test-topic");
        assert_eq!(msg_result.size, b"test payload".len());
    }

    #[tokio::test]
    async fn test_manager_stats_after_send() {
        let config = make_default_config();
        let manager =
            MessageQueueManager::new(config).await.expect("manager creation should succeed");

        let msg = make_test_message("t", b"abc");
        manager.send_message(msg).await.expect("send should succeed");

        let stats = manager.get_stats().await;
        assert_eq!(stats.messages_produced, 1);
        assert_eq!(stats.bytes_produced, 3);
    }

    #[tokio::test]
    async fn test_manager_health_check() {
        let config = make_default_config();
        let manager =
            MessageQueueManager::new(config).await.expect("manager creation should succeed");

        let health = manager.health_check().await;
        assert!(health.is_ok());
        let h = health.expect("health check should succeed");
        assert!(matches!(h.status, HealthStatus::Healthy));
    }

    #[tokio::test]
    async fn test_manager_send_batch() {
        let config = make_default_config();
        let manager =
            MessageQueueManager::new(config).await.expect("manager creation should succeed");

        let msgs = vec![
            make_test_message("batch-topic", b"msg1"),
            make_test_message("batch-topic", b"msg2"),
        ];
        let batch = make_test_batch("batch-topic", msgs);
        let result = manager.send_batch(batch).await;
        assert!(result.is_ok());
        let batch_result = result.expect("batch send should succeed");
        assert_eq!(batch_result.success_count, 2);
        assert_eq!(batch_result.failure_count, 0);
    }

    #[tokio::test]
    async fn test_manager_stats_after_batch() {
        let config = make_default_config();
        let manager =
            MessageQueueManager::new(config).await.expect("manager creation should succeed");

        let msgs = vec![make_test_message("t", b"x"), make_test_message("t", b"yy")];
        let batch = make_test_batch("t", msgs);
        manager.send_batch(batch).await.expect("batch should succeed");

        let stats = manager.get_stats().await;
        assert_eq!(stats.messages_produced, 2);
    }

    #[tokio::test]
    async fn test_manager_close() {
        let config = make_default_config();
        let manager =
            MessageQueueManager::new(config).await.expect("manager creation should succeed");
        let result = manager.close().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_message_queue_event_variants() {
        let msg_id = Uuid::new_v4();
        let result = MessageResult {
            message_id: msg_id,
            topic: "t".to_string(),
            partition: 0,
            offset: 0,
            timestamp: Utc::now(),
            size: 10,
        };
        let event = MessageQueueEvent::MessageProduced(result);
        assert!(matches!(event, MessageQueueEvent::MessageProduced(_)));

        let conn_lost = MessageQueueEvent::ConnectionLost;
        assert!(matches!(conn_lost, MessageQueueEvent::ConnectionLost));

        let error_event = MessageQueueEvent::Error("test error".to_string());
        assert!(matches!(error_event, MessageQueueEvent::Error(_)));
    }

    #[tokio::test]
    async fn test_manager_register_event_handler() {
        let config = make_default_config();
        let manager =
            MessageQueueManager::new(config).await.expect("manager creation should succeed");

        manager
            .register_event_handler("test_handler".to_string(), Box::new(|_event| {}))
            .await;

        // No panic means the handler was registered successfully
    }
    /// A config bound to a broker nobody else in the test suite uses.
    fn isolated_config(label: &str) -> MessageQueueConfig {
        MessageQueueConfig {
            backend: MessageQueueBackend::InMemory,
            connection_string: format!("inmemory://{label}/{}", Uuid::new_v4()),
            topics: vec!["t1".to_string()],
            consumer_group: Some("g1".to_string()),
            batch_size: 10,
            ..Default::default()
        }
    }

    /// Regression test: the InMemory producer used to fabricate a
    /// `MessageResult` without storing anything, and the consumer always polled
    /// an empty vector, so every message was silently discarded.
    #[tokio::test]
    async fn test_inmemory_roundtrip_delivers_the_message() {
        let config = isolated_config("roundtrip");
        let manager = MessageQueueManager::new(config).await.expect("manager");

        let sent = make_test_message("t1", b"real payload");
        let sent_id = sent.id;
        let result = manager.send_message(sent).await.expect("send");
        assert_eq!(result.offset, 0, "the first message must get offset 0");

        let received = manager.consume_messages(500).await.expect("consume");
        assert_eq!(received.len(), 1, "the produced message must come back");
        assert_eq!(received[0].id, sent_id);
        assert_eq!(received[0].payload, b"real payload");
        assert_eq!(received[0].offset, Some(0));
        assert_eq!(received[0].topic, "t1");
    }

    #[tokio::test]
    async fn test_inmemory_preserves_order_and_offsets() {
        let config = isolated_config("order");
        let manager = MessageQueueManager::new(config).await.expect("manager");

        for i in 0..5u8 {
            manager.send_message(make_test_message("t1", &[i])).await.expect("send");
        }
        let received = manager.consume_messages(500).await.expect("consume");
        assert_eq!(received.len(), 5);
        for (index, message) in received.iter().enumerate() {
            assert_eq!(message.payload, vec![index as u8]);
            assert_eq!(message.offset, Some(index as u64));
        }
    }

    #[tokio::test]
    async fn test_inmemory_consumer_group_offset_advances_only_once() {
        let config = isolated_config("offsets");
        let manager = MessageQueueManager::new(config).await.expect("manager");

        manager.send_message(make_test_message("t1", b"a")).await.expect("send");
        let first = manager.consume_messages(200).await.expect("consume");
        assert_eq!(first.len(), 1);
        // Nothing new was produced, so a second poll returns nothing.
        let second = manager.consume_messages(50).await.expect("consume");
        assert!(second.is_empty(), "a consumed offset must not be replayed");
    }

    #[tokio::test]
    async fn test_inmemory_unacknowledged_message_is_redelivered() {
        let config = isolated_config("redelivery");
        let producer = inmemory::InMemoryProducer::new(&config).await.expect("producer");
        let consumer = inmemory::InMemoryConsumer::new(&config)
            .await
            .expect("consumer")
            .with_visibility_timeout(std::time::Duration::from_millis(50));

        producer
            .send_message(make_test_message("t1", b"needs-ack"))
            .await
            .expect("send");

        let first = consumer.poll(200).await.expect("poll");
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].delivery_count, 1);

        // Do not acknowledge; after the visibility timeout the message must
        // come back rather than being lost.
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        let redelivered = consumer.poll(200).await.expect("poll");
        assert_eq!(
            redelivered.len(),
            1,
            "an unacked message must be redelivered"
        );
        assert_eq!(redelivered[0].id, first[0].id);
        assert_eq!(redelivered[0].delivery_count, 2);

        consumer.commit(&redelivered[0]).await.expect("commit");
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        let after_ack = consumer.poll(100).await.expect("poll");
        assert!(
            after_ack.is_empty(),
            "an acknowledged message must not return"
        );
    }

    #[tokio::test]
    async fn test_inmemory_commit_of_undelivered_message_errors() {
        let config = isolated_config("bad-ack");
        let consumer = inmemory::InMemoryConsumer::new(&config).await.expect("consumer");
        let stranger = make_test_message("t1", b"never delivered");
        assert!(
            consumer.commit(&stranger).await.is_err(),
            "acknowledging a message that was never delivered must fail"
        );
    }

    #[tokio::test]
    async fn test_inmemory_seek_replays_from_an_offset() {
        let config = isolated_config("seek");
        let producer = inmemory::InMemoryProducer::new(&config).await.expect("producer");
        let consumer = inmemory::InMemoryConsumer::new(&config).await.expect("consumer");

        for i in 0..3u8 {
            producer.send_message(make_test_message("t1", &[i])).await.expect("send");
        }
        let first = consumer.poll(200).await.expect("poll");
        assert_eq!(first.len(), 3);
        for message in &first {
            consumer.commit(message).await.expect("commit");
        }

        consumer.seek("t1", 0, 1).await.expect("seek");
        let replayed = consumer.poll(200).await.expect("poll");
        assert_eq!(
            replayed.len(),
            2,
            "seeking to offset 1 must replay the tail"
        );
        assert_eq!(replayed[0].payload, vec![1u8]);
    }

    #[tokio::test]
    async fn test_inmemory_pause_stops_delivery_and_resume_restores_it() {
        let config = isolated_config("pause");
        let producer = inmemory::InMemoryProducer::new(&config).await.expect("producer");
        let consumer = inmemory::InMemoryConsumer::new(&config).await.expect("consumer");

        producer.send_message(make_test_message("t1", b"x")).await.expect("send");
        consumer.pause(&["t1".to_string()]).await.expect("pause");
        assert!(consumer.poll(50).await.expect("poll").is_empty());

        consumer.resume(&["t1".to_string()]).await.expect("resume");
        assert_eq!(consumer.poll(200).await.expect("poll").len(), 1);
    }

    #[tokio::test]
    async fn test_inmemory_transaction_is_invisible_until_commit() {
        let config = isolated_config("txn");
        let producer = inmemory::InMemoryProducer::new(&config).await.expect("producer");
        let consumer = inmemory::InMemoryConsumer::new(&config).await.expect("consumer");

        let transaction = producer.begin_transaction().await.expect("begin");
        producer
            .send_message(make_test_message("t1", b"transactional"))
            .await
            .expect("send");
        assert!(
            consumer.poll(50).await.expect("poll").is_empty(),
            "uncommitted messages must not be visible"
        );

        producer.commit_transaction(transaction).await.expect("commit");
        let delivered = consumer.poll(200).await.expect("poll");
        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].payload, b"transactional");
    }

    #[tokio::test]
    async fn test_inmemory_aborted_transaction_discards_messages() {
        let config = isolated_config("abort");
        let producer = inmemory::InMemoryProducer::new(&config).await.expect("producer");
        let consumer = inmemory::InMemoryConsumer::new(&config).await.expect("consumer");

        let transaction = producer.begin_transaction().await.expect("begin");
        producer.send_message(make_test_message("t1", b"dropped")).await.expect("send");
        producer.abort_transaction(transaction).await.expect("abort");

        assert!(consumer.poll(100).await.expect("poll").is_empty());
    }

    #[tokio::test]
    async fn test_inmemory_batch_send_is_really_stored() {
        let config = isolated_config("batch");
        let manager = MessageQueueManager::new(config).await.expect("manager");

        let batch = make_test_batch(
            "t1",
            vec![
                make_test_message("t1", b"one"),
                make_test_message("t1", b"two"),
            ],
        );
        let result = manager.send_batch(batch).await.expect("batch");
        assert_eq!(result.success_count, 2);
        assert_eq!(
            result.results.len(),
            2,
            "a batch result must carry one entry per stored message"
        );

        let received = manager.consume_messages(500).await.expect("consume");
        assert_eq!(received.len(), 2);
    }

    #[tokio::test]
    async fn test_producer_and_consumer_share_the_broker_by_connection_string() {
        let config = isolated_config("shared");
        // Two independently-created clients, as the manager builds them.
        let producer = inmemory::InMemoryProducer::new(&config).await.expect("producer");
        let consumer = inmemory::InMemoryConsumer::new(&config).await.expect("consumer");

        producer.send_message(make_test_message("t1", b"shared")).await.expect("send");
        let received = consumer.poll(300).await.expect("poll");
        assert_eq!(
            received.len(),
            1,
            "independently built clients must share state"
        );
    }

    #[tokio::test]
    async fn test_different_connection_strings_are_isolated() {
        let config_a = isolated_config("iso-a");
        let config_b = isolated_config("iso-b");
        let producer = inmemory::InMemoryProducer::new(&config_a).await.expect("producer");
        let consumer = inmemory::InMemoryConsumer::new(&config_b).await.expect("consumer");

        producer.send_message(make_test_message("t1", b"a")).await.expect("send");
        assert!(
            consumer.poll(100).await.expect("poll").is_empty(),
            "brokers with different connection strings must not share topics"
        );
    }

    #[tokio::test]
    async fn test_health_check_reports_the_real_backlog() {
        let config = isolated_config("health");
        let manager = MessageQueueManager::new(config).await.expect("manager");

        let health = manager.health_check().await.expect("health");
        assert!(
            health.last_message_timestamp.is_none(),
            "no message has passed through yet"
        );
        assert_eq!(health.message_queue_size, 0);

        manager.send_message(make_test_message("t1", b"x")).await.expect("send");
        let health = manager.health_check().await.expect("health");
        assert_eq!(health.message_queue_size, 1, "one produced, none consumed");
        assert!(health.last_message_timestamp.is_some());

        manager.consume_messages(300).await.expect("consume");
        let health = manager.health_check().await.expect("health");
        assert_eq!(health.message_queue_size, 0);
        assert_eq!(health.error_count, 0);
    }

    /// Every non-InMemory backend connects eagerly: pointing it at a closed
    /// port must fail loudly instead of yielding a client that drops messages.
    #[tokio::test]
    async fn test_broker_backends_fail_when_the_broker_is_unreachable() {
        // Bind and immediately drop a listener to obtain a port nothing serves.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        drop(listener);

        for backend in [
            MessageQueueBackend::RabbitMQ,
            MessageQueueBackend::RedisStreams,
            MessageQueueBackend::Nats,
        ] {
            let config = MessageQueueConfig {
                backend: backend.clone(),
                connection_string: format!("{}:{}", addr.ip(), addr.port()),
                topics: vec!["t1".to_string()],
                ..Default::default()
            };
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(20),
                MessageQueueManager::new(config),
            )
            .await
            .expect("connection attempt must not hang");
            assert!(
                result.is_err(),
                "{backend:?} must not report success against a closed port"
            );
        }
    }
}
