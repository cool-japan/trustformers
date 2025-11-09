use anyhow::{Context, Result};
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
        })
    }

    async fn create_backend_clients(
        config: &MessageQueueConfig,
    ) -> Result<(Arc<dyn MessageQueueProducer>, Arc<dyn MessageQueueConsumer>)> {
        match config.backend {
            MessageQueueBackend::Kafka => Self::create_kafka_clients(config).await,
            MessageQueueBackend::RabbitMQ => Self::create_rabbitmq_clients(config).await,
            MessageQueueBackend::RedisStreams => Self::create_redis_clients(config).await,
            MessageQueueBackend::Nats => Self::create_nats_clients(config).await,
            MessageQueueBackend::AmazonSqs => Self::create_sqs_clients(config).await,
            MessageQueueBackend::InMemory => Self::create_inmemory_clients(config).await,
        }
    }

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
        let result = self.producer.send_message(message).await?;
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
        let messages = self.consumer.poll(timeout_ms).await?;
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

    async fn emit_event(&self, event: MessageQueueEvent) {
        let handlers = self.event_handlers.read().await;
        for handler in handlers.values() {
            handler(event.clone());
        }
    }

    async fn update_stats_produced(&self, result: &MessageResult) {
        let mut stats = self.stats.write().await;
        stats.messages_produced += 1;
        stats.bytes_produced += result.size as u64;
    }

    async fn update_stats_batch_produced(&self, result: &BatchResult) {
        let mut stats = self.stats.write().await;
        stats.messages_produced += result.success_count as u64;
        stats.bytes_produced += result.total_size as u64;
    }

    async fn update_stats_consumed(&self, message: &Message) {
        let mut stats = self.stats.write().await;
        stats.messages_consumed += 1;
        stats.bytes_consumed += message.payload.len() as u64;
    }

    pub async fn get_stats(&self) -> MessageQueueStats {
        self.stats.read().await.clone()
    }

    pub async fn health_check(&self) -> Result<MessageQueueHealth> {
        Ok(MessageQueueHealth {
            backend: self.config.backend.clone(),
            status: HealthStatus::Healthy,
            connection_count: self.config.performance.connection_pool_size as u32,
            message_queue_size: 0,
            last_message_timestamp: Some(Utc::now()),
            error_count: 0,
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
struct KafkaProducer {
    producer: rdkafka::producer::FutureProducer,
}

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
            Ok((partition, offset)) => Ok(MessageResult {
                message_id: message.id,
                topic: message.topic,
                partition: partition as u32,
                offset: offset as u64,
                timestamp: Utc::now(),
                size: message.payload.len(),
            }),
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

struct KafkaConsumer {
    consumer: rdkafka::consumer::StreamConsumer,
}

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

// Placeholder implementations for other backends
struct RabbitMQProducer;
struct RabbitMQConsumer;
struct RedisProducer;
struct RedisConsumer;
struct NatsProducer;
struct NatsConsumer;
struct SqsProducer;
struct SqsConsumer;
struct InMemoryProducer;
struct InMemoryConsumer;

macro_rules! impl_placeholder_backend {
    ($producer:ident, $consumer:ident) => {
        impl $producer {
            async fn new(_config: &MessageQueueConfig) -> Result<Self> {
                Ok(Self)
            }
        }

        impl $consumer {
            async fn new(_config: &MessageQueueConfig) -> Result<Self> {
                Ok(Self)
            }
        }

        #[async_trait]
        impl MessageQueueProducer for $producer {
            async fn send_message(&self, message: Message) -> Result<MessageResult> {
                Ok(MessageResult {
                    message_id: message.id,
                    topic: message.topic,
                    partition: 0,
                    offset: 0,
                    timestamp: Utc::now(),
                    size: message.payload.len(),
                })
            }

            async fn send_batch(&self, batch: MessageBatch) -> Result<BatchResult> {
                Ok(BatchResult {
                    batch_id: batch.batch_id,
                    results: vec![],
                    success_count: batch.messages.len(),
                    failure_count: 0,
                    total_size: batch.messages.iter().map(|m| m.payload.len()).sum(),
                })
            }

            async fn send_with_callback(
                &self,
                message: Message,
                callback: ProducerCallback,
            ) -> Result<()> {
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
                Ok(())
            }

            async fn close(&self) -> Result<()> {
                Ok(())
            }
        }

        #[async_trait]
        impl MessageQueueConsumer for $consumer {
            async fn subscribe(&self, _topics: &[String]) -> Result<()> {
                Ok(())
            }

            async fn unsubscribe(&self, _topics: &[String]) -> Result<()> {
                Ok(())
            }

            async fn poll(&self, _timeout_ms: u64) -> Result<Vec<Message>> {
                Ok(vec![])
            }

            async fn commit(&self, _message: &Message) -> Result<()> {
                Ok(())
            }

            async fn commit_batch(&self, _messages: &[Message]) -> Result<()> {
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
    };
}

impl_placeholder_backend!(RabbitMQProducer, RabbitMQConsumer);
impl_placeholder_backend!(RedisProducer, RedisConsumer);
impl_placeholder_backend!(NatsProducer, NatsConsumer);
impl_placeholder_backend!(SqsProducer, SqsConsumer);
impl_placeholder_backend!(InMemoryProducer, InMemoryConsumer);
