//! RabbitMQ (AMQP 0-9-1) backend built on [`lapin`].
//!
//! The producer publishes to the default exchange with the topic as the routing
//! key, waiting for a publisher confirmation. The consumer polls with
//! `basic_get` and acknowledges explicitly, so an unacknowledged message is
//! redelivered by the broker rather than silently dropped.
//!
//! Connecting is eager: if the broker is unreachable, construction fails with a
//! real error instead of producing a client that pretends to work.

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use lapin::options::{
    BasicAckOptions, BasicGetOptions, BasicNackOptions, BasicPublishOptions, QueueDeclareOptions,
};
use lapin::types::{AMQPValue, FieldTable, ShortString};
use lapin::{BasicProperties, Channel, Connection, ConnectionProperties};
use tokio::sync::Mutex;
use uuid::Uuid;

use super::inmemory::message_from_parts;
use super::{
    BatchResult, Message, MessageBatch, MessageQueueConfig, MessageQueueConsumer,
    MessageQueueProducer, MessageResult, ProducerCallback, TransactionId,
};

/// Open a connection and channel to the configured broker.
async fn connect(config: &MessageQueueConfig) -> Result<(Connection, Channel)> {
    let uri = if config.connection_string.contains("://") {
        config.connection_string.clone()
    } else {
        format!("amqp://{}/%2f", config.connection_string)
    };
    let connection = Connection::connect(&uri, ConnectionProperties::default())
        .await
        .with_context(|| format!("cannot connect to RabbitMQ at {uri}"))?;
    let channel = connection.create_channel().await.context("cannot open a RabbitMQ channel")?;
    Ok((connection, channel))
}

async fn declare_topics(channel: &Channel, topics: &[String]) -> Result<()> {
    for topic in topics {
        channel
            .queue_declare(
                ShortString::from(topic.as_str()),
                QueueDeclareOptions::durable(),
                FieldTable::default(),
            )
            .await
            .with_context(|| format!("cannot declare RabbitMQ queue {topic}"))?;
    }
    Ok(())
}

fn headers_to_field_table(headers: &HashMap<String, String>) -> FieldTable {
    let mut table = FieldTable::default();
    for (key, value) in headers {
        table.insert(
            ShortString::from(key.as_str()),
            AMQPValue::LongString(value.as_str().into()),
        );
    }
    table
}

fn field_table_to_headers(table: Option<&FieldTable>) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    if let Some(table) = table {
        for (key, value) in table.inner() {
            if let AMQPValue::LongString(text) = value {
                headers.insert(key.to_string(), text.to_string());
            }
        }
    }
    headers
}

/// RabbitMQ producer.
pub struct RabbitMQProducer {
    _connection: Connection,
    channel: Channel,
    transactions: Mutex<HashMap<TransactionId, Vec<Message>>>,
    active_transaction: Mutex<Option<TransactionId>>,
}

impl std::fmt::Debug for RabbitMQProducer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RabbitMQProducer").finish_non_exhaustive()
    }
}

impl RabbitMQProducer {
    /// Connect to RabbitMQ and declare the configured topics.
    ///
    /// # Errors
    ///
    /// Returns an error if the broker is unreachable or a queue cannot be
    /// declared.
    pub async fn new(config: &MessageQueueConfig) -> Result<Self> {
        let (connection, channel) = connect(config).await?;
        declare_topics(&channel, &config.topics).await?;
        Ok(Self {
            _connection: connection,
            channel,
            transactions: Mutex::new(HashMap::new()),
            active_transaction: Mutex::new(None),
        })
    }

    async fn publish(&self, message: &Message) -> Result<MessageResult> {
        let mut properties = BasicProperties::default()
            .with_message_id(ShortString::from(message.id.to_string()))
            .with_headers(headers_to_field_table(&message.headers));
        if let Some(correlation_id) = &message.correlation_id {
            properties = properties.with_correlation_id(ShortString::from(correlation_id.as_str()));
        }
        if let Some(reply_to) = &message.reply_to {
            properties = properties.with_reply_to(ShortString::from(reply_to.as_str()));
        }

        self.channel
            .basic_publish(
                ShortString::from(""),
                ShortString::from(message.topic.as_str()),
                BasicPublishOptions::default(),
                &message.payload,
                properties,
            )
            .await
            .with_context(|| format!("RabbitMQ publish to {} failed", message.topic))?
            .await
            .with_context(|| format!("RabbitMQ did not confirm publish to {}", message.topic))?;

        Ok(MessageResult {
            message_id: message.id,
            topic: message.topic.clone(),
            // AMQP has no partitions or offsets; report the queue's single
            // logical partition and no offset rather than inventing one.
            partition: 0,
            offset: 0,
            timestamp: message.timestamp,
            size: message.payload.len(),
        })
    }
}

#[async_trait]
impl MessageQueueProducer for RabbitMQProducer {
    async fn send_message(&self, message: Message) -> Result<MessageResult> {
        let active = self.active_transaction.lock().await.clone();
        if let Some(transaction_id) = active {
            let result = MessageResult {
                message_id: message.id,
                topic: message.topic.clone(),
                partition: 0,
                offset: 0,
                timestamp: message.timestamp,
                size: message.payload.len(),
            };
            self.transactions.lock().await.entry(transaction_id).or_default().push(message);
            return Ok(result);
        }
        self.publish(&message).await
    }

    async fn send_batch(&self, batch: MessageBatch) -> Result<BatchResult> {
        let batch_id = batch.batch_id;
        let mut results = Vec::new();
        let mut failure_count = 0usize;
        let mut total_size = 0usize;
        for message in batch.messages {
            match self.send_message(message).await {
                Ok(result) => {
                    total_size += result.size;
                    results.push(result);
                },
                Err(_) => failure_count += 1,
            }
        }
        Ok(BatchResult {
            batch_id,
            success_count: results.len(),
            failure_count,
            total_size,
            results,
        })
    }

    async fn send_with_callback(&self, message: Message, callback: ProducerCallback) -> Result<()> {
        let result = self.send_message(message).await;
        callback(result);
        Ok(())
    }

    async fn begin_transaction(&self) -> Result<TransactionId> {
        let id = Uuid::new_v4().to_string();
        self.transactions.lock().await.insert(id.clone(), Vec::new());
        *self.active_transaction.lock().await = Some(id.clone());
        Ok(id)
    }

    async fn commit_transaction(&self, transaction_id: TransactionId) -> Result<()> {
        let buffered = self
            .transactions
            .lock()
            .await
            .remove(&transaction_id)
            .ok_or_else(|| anyhow::anyhow!("unknown transaction: {transaction_id}"))?;
        {
            let mut active = self.active_transaction.lock().await;
            if active.as_deref() == Some(transaction_id.as_str()) {
                *active = None;
            }
        }
        for message in buffered {
            self.publish(&message).await?;
        }
        Ok(())
    }

    async fn abort_transaction(&self, transaction_id: TransactionId) -> Result<()> {
        self.transactions
            .lock()
            .await
            .remove(&transaction_id)
            .ok_or_else(|| anyhow::anyhow!("unknown transaction: {transaction_id}"))?;
        let mut active = self.active_transaction.lock().await;
        if active.as_deref() == Some(transaction_id.as_str()) {
            *active = None;
        }
        Ok(())
    }

    async fn flush(&self) -> Result<()> {
        // Every publish already awaits its broker confirmation.
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.channel
            .close(200, ShortString::from("closing"))
            .await
            .context("RabbitMQ channel close failed")?;
        Ok(())
    }
}

/// RabbitMQ consumer.
pub struct RabbitMQConsumer {
    _connection: Connection,
    channel: Channel,
    subscriptions: Mutex<Vec<String>>,
    paused: Mutex<Vec<String>>,
    /// Delivery tags of messages handed out but not yet acknowledged.
    in_flight: Mutex<HashMap<Uuid, u64>>,
    max_poll_records: usize,
}

impl std::fmt::Debug for RabbitMQConsumer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RabbitMQConsumer").finish_non_exhaustive()
    }
}

impl RabbitMQConsumer {
    /// Connect to RabbitMQ and declare the configured topics.
    ///
    /// # Errors
    ///
    /// Returns an error if the broker is unreachable or a queue cannot be
    /// declared.
    pub async fn new(config: &MessageQueueConfig) -> Result<Self> {
        let (connection, channel) = connect(config).await?;
        declare_topics(&channel, &config.topics).await?;
        Ok(Self {
            _connection: connection,
            channel,
            subscriptions: Mutex::new(config.topics.clone()),
            paused: Mutex::new(Vec::new()),
            in_flight: Mutex::new(HashMap::new()),
            max_poll_records: config.batch_size.max(1),
        })
    }
}

#[async_trait]
impl MessageQueueConsumer for RabbitMQConsumer {
    async fn subscribe(&self, topics: &[String]) -> Result<()> {
        declare_topics(&self.channel, topics).await?;
        let mut subscriptions = self.subscriptions.lock().await;
        for topic in topics {
            if !subscriptions.contains(topic) {
                subscriptions.push(topic.clone());
            }
        }
        Ok(())
    }

    async fn unsubscribe(&self, topics: &[String]) -> Result<()> {
        self.subscriptions.lock().await.retain(|topic| !topics.contains(topic));
        Ok(())
    }

    async fn poll(&self, timeout_ms: u64) -> Result<Vec<Message>> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        loop {
            let topics = self.subscriptions.lock().await.clone();
            let paused = self.paused.lock().await.clone();
            let mut collected = Vec::new();

            for topic in topics.iter().filter(|topic| !paused.contains(topic)) {
                while collected.len() < self.max_poll_records {
                    let delivery = self
                        .channel
                        .basic_get(
                            ShortString::from(topic.as_str()),
                            BasicGetOptions::default(),
                        )
                        .await
                        .with_context(|| format!("RabbitMQ basic_get on {topic} failed"))?;
                    let Some(delivery) = delivery else {
                        break;
                    };
                    let properties = delivery.properties.clone();
                    let mut message = message_from_parts(
                        topic,
                        delivery.data.clone(),
                        None,
                        field_table_to_headers(properties.headers().as_ref()),
                    );
                    if let Some(id) = properties
                        .message_id()
                        .as_ref()
                        .and_then(|id| Uuid::parse_str(id.as_str()).ok())
                    {
                        message.id = id;
                    }
                    message.correlation_id =
                        properties.correlation_id().as_ref().map(|id| id.to_string());
                    message.reply_to = properties.reply_to().as_ref().map(|to| to.to_string());
                    message.offset = Some(delivery.delivery_tag);

                    self.in_flight.lock().await.insert(message.id, delivery.delivery_tag);
                    collected.push(message);
                }
            }

            if !collected.is_empty() {
                return Ok(collected);
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return Ok(Vec::new());
            }
            tokio::time::sleep(std::cmp::min(
                std::time::Duration::from_millis(25),
                deadline - now,
            ))
            .await;
        }
    }

    async fn commit(&self, message: &Message) -> Result<()> {
        let tag = self.in_flight.lock().await.remove(&message.id).ok_or_else(|| {
            anyhow::anyhow!("message {} was not delivered by this consumer", message.id)
        })?;
        self.channel
            .basic_ack(tag, BasicAckOptions::default())
            .await
            .context("RabbitMQ basic_ack failed")?;
        Ok(())
    }

    async fn commit_batch(&self, messages: &[Message]) -> Result<()> {
        for message in messages {
            self.commit(message).await?;
        }
        Ok(())
    }

    async fn seek(&self, _topic: &str, _partition: u32, _offset: u64) -> Result<()> {
        Err(anyhow::anyhow!(
            "AMQP queues have no seekable offsets; requeue unacknowledged messages instead"
        ))
    }

    async fn pause(&self, topics: &[String]) -> Result<()> {
        let mut paused = self.paused.lock().await;
        for topic in topics {
            if !paused.contains(topic) {
                paused.push(topic.clone());
            }
        }
        Ok(())
    }

    async fn resume(&self, topics: &[String]) -> Result<()> {
        self.paused.lock().await.retain(|topic| !topics.contains(topic));
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        // Return anything still unacknowledged to the queue instead of losing it.
        let in_flight: Vec<u64> = self.in_flight.lock().await.values().copied().collect();
        for tag in in_flight {
            let _ = self
                .channel
                .basic_nack(
                    tag,
                    BasicNackOptions {
                        requeue: true,
                        ..BasicNackOptions::default()
                    },
                )
                .await;
        }
        self.channel
            .close(200, ShortString::from("closing"))
            .await
            .context("RabbitMQ channel close failed")?;
        Ok(())
    }
}
