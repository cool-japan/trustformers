//! Redis Streams backend.
//!
//! Messages are appended with `XADD` and consumed with `XREADGROUP`, so
//! delivery is tracked by Redis' pending entries list (PEL) and an
//! unacknowledged entry stays pending until `XACK`. The consumer group named by
//! [`MessageQueueConfig::consumer_group`](super::MessageQueueConfig) is created
//! on demand with `XGROUP CREATE … MKSTREAM`.
//!
//! Connecting is eager: an unreachable Redis fails construction with a real
//! error rather than yielding a client that silently discards messages.

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use redis::aio::MultiplexedConnection;
use redis::AsyncCommands;
use tokio::sync::Mutex;
use uuid::Uuid;

use super::inmemory::message_from_parts;
use super::{
    BatchResult, Message, MessageBatch, MessageQueueConfig, MessageQueueConsumer,
    MessageQueueProducer, MessageResult, ProducerCallback, TransactionId,
};

/// Stream field holding the raw payload.
const FIELD_PAYLOAD: &str = "payload";
/// Stream field holding the message id.
const FIELD_MESSAGE_ID: &str = "message_id";
/// Stream field holding the partition key.
const FIELD_KEY: &str = "key";
/// Prefix for user headers stored as stream fields.
const HEADER_PREFIX: &str = "h:";

fn redis_url(config: &MessageQueueConfig) -> String {
    if config.connection_string.contains("://") {
        config.connection_string.clone()
    } else {
        format!("redis://{}", config.connection_string)
    }
}

async fn connect(config: &MessageQueueConfig) -> Result<MultiplexedConnection> {
    let url = redis_url(config);
    let client =
        redis::Client::open(url.clone()).with_context(|| format!("invalid Redis URL: {url}"))?;
    client
        .get_multiplexed_async_connection()
        .await
        .with_context(|| format!("cannot connect to Redis at {url}"))
}

/// Redis Streams producer.
#[derive(Debug)]
pub struct RedisProducer {
    connection: Mutex<MultiplexedConnection>,
    transactions: Mutex<HashMap<TransactionId, Vec<Message>>>,
    active_transaction: Mutex<Option<TransactionId>>,
}

impl RedisProducer {
    /// Connect to Redis.
    ///
    /// # Errors
    ///
    /// Returns an error if the URL is invalid or Redis is unreachable.
    pub async fn new(config: &MessageQueueConfig) -> Result<Self> {
        Ok(Self {
            connection: Mutex::new(connect(config).await?),
            transactions: Mutex::new(HashMap::new()),
            active_transaction: Mutex::new(None),
        })
    }

    async fn xadd(&self, message: &Message) -> Result<MessageResult> {
        let mut fields: Vec<(String, Vec<u8>)> = vec![
            (FIELD_PAYLOAD.to_string(), message.payload.clone()),
            (
                FIELD_MESSAGE_ID.to_string(),
                message.id.to_string().into_bytes(),
            ),
        ];
        if let Some(key) = &message.key {
            fields.push((FIELD_KEY.to_string(), key.clone().into_bytes()));
        }
        for (name, value) in &message.headers {
            fields.push((format!("{HEADER_PREFIX}{name}"), value.clone().into_bytes()));
        }

        let mut connection = self.connection.lock().await;
        let entry_id: String = connection
            .xadd(&message.topic, "*", &fields)
            .await
            .with_context(|| format!("Redis XADD to stream {} failed", message.topic))?;

        // Redis entry ids are `<millis>-<seq>`; use the millisecond part as the
        // offset so it is a real, monotonically increasing broker value.
        let offset = entry_id
            .split('-')
            .next()
            .and_then(|part| part.parse::<u64>().ok())
            .unwrap_or_default();

        Ok(MessageResult {
            message_id: message.id,
            topic: message.topic.clone(),
            partition: 0,
            offset,
            timestamp: message.timestamp,
            size: message.payload.len(),
        })
    }
}

#[async_trait]
impl MessageQueueProducer for RedisProducer {
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
        self.xadd(&message).await
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
            self.xadd(&message).await?;
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
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

/// Redis Streams consumer, reading through a consumer group.
#[derive(Debug)]
pub struct RedisConsumer {
    connection: Mutex<MultiplexedConnection>,
    group: String,
    consumer_name: String,
    max_poll_records: usize,
    subscriptions: Mutex<Vec<String>>,
    paused: Mutex<Vec<String>>,
    /// Stream entry ids of delivered-but-unacknowledged messages.
    in_flight: Mutex<HashMap<Uuid, (String, String)>>,
}

impl RedisConsumer {
    /// Connect to Redis and prepare the consumer group for each topic.
    ///
    /// # Errors
    ///
    /// Returns an error if Redis is unreachable.
    pub async fn new(config: &MessageQueueConfig) -> Result<Self> {
        let connection = connect(config).await?;
        let consumer = Self {
            connection: Mutex::new(connection),
            group: config.consumer_group.clone().unwrap_or_else(|| "default".to_string()),
            consumer_name: format!("trustformers-{}", Uuid::new_v4()),
            max_poll_records: config.batch_size.max(1),
            subscriptions: Mutex::new(config.topics.clone()),
            paused: Mutex::new(Vec::new()),
            in_flight: Mutex::new(HashMap::new()),
        };
        consumer.ensure_groups(&config.topics).await?;
        Ok(consumer)
    }

    async fn ensure_groups(&self, topics: &[String]) -> Result<()> {
        let mut connection = self.connection.lock().await;
        for topic in topics {
            // BUSYGROUP means the group already exists, which is not an error.
            let created: redis::RedisResult<String> = redis::cmd("XGROUP")
                .arg("CREATE")
                .arg(topic)
                .arg(&self.group)
                .arg("$")
                .arg("MKSTREAM")
                .query_async(&mut *connection)
                .await;
            if let Err(e) = created {
                let message = e.to_string();
                if !message.contains("BUSYGROUP") {
                    return Err(anyhow::Error::new(e).context(format!(
                        "cannot create Redis consumer group {} on stream {topic}",
                        self.group
                    )));
                }
            }
        }
        Ok(())
    }

    fn decode(topic: &str, entry: &redis::streams::StreamId) -> Message {
        let mut payload = Vec::new();
        let mut headers = HashMap::new();
        let mut key = None;
        let mut message_id = None;

        for (field, value) in &entry.map {
            let bytes: Vec<u8> = match value {
                redis::Value::BulkString(data) => data.clone(),
                redis::Value::SimpleString(text) => text.clone().into_bytes(),
                other => format!("{other:?}").into_bytes(),
            };
            match field.as_str() {
                FIELD_PAYLOAD => payload = bytes,
                FIELD_MESSAGE_ID => {
                    message_id =
                        String::from_utf8(bytes).ok().and_then(|s| Uuid::parse_str(&s).ok())
                },
                FIELD_KEY => key = String::from_utf8(bytes).ok(),
                name if name.starts_with(HEADER_PREFIX) => {
                    if let Ok(text) = String::from_utf8(bytes) {
                        headers.insert(name[HEADER_PREFIX.len()..].to_string(), text);
                    }
                },
                _ => {},
            }
        }

        let mut message = message_from_parts(topic, payload, key, headers);
        if let Some(id) = message_id {
            message.id = id;
        }
        if let Some(count) = entry.delivered_count {
            message.delivery_count = count as u32;
        }
        message.offset = entry.id.split('-').next().and_then(|part| part.parse::<u64>().ok());
        message
    }
}

#[async_trait]
impl MessageQueueConsumer for RedisConsumer {
    async fn subscribe(&self, topics: &[String]) -> Result<()> {
        self.ensure_groups(topics).await?;
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
        let topics: Vec<String> = {
            let subscriptions = self.subscriptions.lock().await;
            let paused = self.paused.lock().await;
            subscriptions.iter().filter(|t| !paused.contains(t)).cloned().collect()
        };
        if topics.is_empty() {
            return Ok(Vec::new());
        }

        let mut command = redis::cmd("XREADGROUP");
        command
            .arg("GROUP")
            .arg(&self.group)
            .arg(&self.consumer_name)
            .arg("COUNT")
            .arg(self.max_poll_records)
            .arg("BLOCK")
            .arg(timeout_ms.max(1))
            .arg("STREAMS");
        for topic in &topics {
            command.arg(topic);
        }
        for _ in &topics {
            command.arg(">");
        }

        let mut connection = self.connection.lock().await;
        let reply: Option<redis::streams::StreamReadReply> =
            command.query_async(&mut *connection).await.context("Redis XREADGROUP failed")?;
        drop(connection);

        let Some(reply) = reply else {
            return Ok(Vec::new());
        };

        let mut messages = Vec::new();
        let mut in_flight = self.in_flight.lock().await;
        for stream in reply.keys {
            for entry in &stream.ids {
                let message = Self::decode(&stream.key, entry);
                in_flight.insert(message.id, (stream.key.clone(), entry.id.clone()));
                messages.push(message);
            }
        }
        Ok(messages)
    }

    async fn commit(&self, message: &Message) -> Result<()> {
        let (stream, entry_id) =
            self.in_flight.lock().await.remove(&message.id).ok_or_else(|| {
                anyhow::anyhow!("message {} was not delivered by this consumer", message.id)
            })?;
        let mut connection = self.connection.lock().await;
        let acked: i64 = redis::cmd("XACK")
            .arg(&stream)
            .arg(&self.group)
            .arg(&entry_id)
            .query_async(&mut *connection)
            .await
            .context("Redis XACK failed")?;
        if acked == 0 {
            return Err(anyhow::anyhow!(
                "Redis did not acknowledge entry {entry_id} of stream {stream}"
            ));
        }
        Ok(())
    }

    async fn commit_batch(&self, messages: &[Message]) -> Result<()> {
        for message in messages {
            self.commit(message).await?;
        }
        Ok(())
    }

    async fn seek(&self, topic: &str, _partition: u32, offset: u64) -> Result<()> {
        let mut connection = self.connection.lock().await;
        let _: String = redis::cmd("XGROUP")
            .arg("SETID")
            .arg(topic)
            .arg(&self.group)
            .arg(format!("{offset}-0"))
            .query_async(&mut *connection)
            .await
            .context("Redis XGROUP SETID failed")?;
        Ok(())
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
        // Unacknowledged entries stay in the group's pending list and are
        // reclaimed by another consumer; nothing is dropped here.
        self.subscriptions.lock().await.clear();
        Ok(())
    }
}
