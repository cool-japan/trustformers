//! NATS backend built on [`async_nats`].
//!
//! The producer publishes with headers carrying the message id, key and user
//! headers; the consumer holds one core-NATS subscription per topic and drains
//! it on `poll`. Core NATS is fire-and-forget, so `commit` is a local no-op that
//! only checks the message really came from this consumer — it never claims a
//! broker acknowledgement that does not exist.
//!
//! Connecting is eager: an unreachable server fails construction with a real
//! error rather than yielding a client that discards messages.

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_nats::{Client, HeaderMap as NatsHeaders, Subscriber};
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use tokio::sync::Mutex;
use uuid::Uuid;

use super::inmemory::message_from_parts;
use super::{
    BatchResult, Message, MessageBatch, MessageQueueConfig, MessageQueueConsumer,
    MessageQueueProducer, MessageResult, ProducerCallback, TransactionId,
};

/// Header carrying the TrustformeRS message id.
const HEADER_MESSAGE_ID: &str = "Trustformers-Message-Id";
/// Header carrying the partition key.
const HEADER_KEY: &str = "Trustformers-Key";
/// Prefix for user headers.
const HEADER_PREFIX: &str = "Trustformers-H-";

async fn connect(config: &MessageQueueConfig) -> Result<Client> {
    let address = if config.connection_string.contains("://") {
        config.connection_string.clone()
    } else {
        format!("nats://{}", config.connection_string)
    };
    async_nats::connect(address.clone())
        .await
        .with_context(|| format!("cannot connect to NATS at {address}"))
}

/// NATS producer.
#[derive(Debug)]
pub struct NatsProducer {
    client: Client,
    transactions: Mutex<HashMap<TransactionId, Vec<Message>>>,
    active_transaction: Mutex<Option<TransactionId>>,
}

impl NatsProducer {
    /// Connect to NATS.
    ///
    /// # Errors
    ///
    /// Returns an error if the server is unreachable.
    pub async fn new(config: &MessageQueueConfig) -> Result<Self> {
        Ok(Self {
            client: connect(config).await?,
            transactions: Mutex::new(HashMap::new()),
            active_transaction: Mutex::new(None),
        })
    }

    async fn publish(&self, message: &Message) -> Result<MessageResult> {
        let mut headers = NatsHeaders::new();
        headers.insert(HEADER_MESSAGE_ID, message.id.to_string().as_str());
        if let Some(key) = &message.key {
            headers.insert(HEADER_KEY, key.as_str());
        }
        for (name, value) in &message.headers {
            headers.insert(format!("{HEADER_PREFIX}{name}").as_str(), value.as_str());
        }

        self.client
            .publish_with_headers(
                message.topic.clone(),
                headers,
                Bytes::from(message.payload.clone()),
            )
            .await
            .with_context(|| format!("NATS publish to {} failed", message.topic))?;
        // Core NATS acknowledges only at flush time; make the send durable
        // before reporting success.
        self.client.flush().await.context("NATS flush failed")?;

        Ok(MessageResult {
            message_id: message.id,
            topic: message.topic.clone(),
            partition: 0,
            offset: 0,
            timestamp: message.timestamp,
            size: message.payload.len(),
        })
    }
}

#[async_trait]
impl MessageQueueProducer for NatsProducer {
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
        self.client.flush().await.context("NATS flush failed")?;
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.client.flush().await.context("NATS flush failed")?;
        Ok(())
    }
}

/// NATS consumer holding one subscription per subscribed subject.
#[derive(Debug)]
pub struct NatsConsumer {
    client: Client,
    subscriptions: Mutex<HashMap<String, Subscriber>>,
    paused: Mutex<Vec<String>>,
    delivered: Mutex<Vec<Uuid>>,
    max_poll_records: usize,
}

impl NatsConsumer {
    /// Connect to NATS and subscribe to the configured topics.
    ///
    /// # Errors
    ///
    /// Returns an error if the server is unreachable or a subject is invalid.
    pub async fn new(config: &MessageQueueConfig) -> Result<Self> {
        let client = connect(config).await?;
        let consumer = Self {
            client,
            subscriptions: Mutex::new(HashMap::new()),
            paused: Mutex::new(Vec::new()),
            delivered: Mutex::new(Vec::new()),
            max_poll_records: config.batch_size.max(1),
        };
        consumer.subscribe(&config.topics).await?;
        Ok(consumer)
    }

    fn decode(subject: &str, nats_message: async_nats::Message) -> Message {
        let mut headers = HashMap::new();
        let mut key = None;
        let mut message_id = None;
        if let Some(nats_headers) = &nats_message.headers {
            for (name, values) in nats_headers.iter() {
                let Some(value) = values.first() else {
                    continue;
                };
                let name = name.to_string();
                if name == HEADER_MESSAGE_ID {
                    message_id = Uuid::parse_str(value.as_str()).ok();
                } else if name == HEADER_KEY {
                    key = Some(value.to_string());
                } else if let Some(stripped) = name.strip_prefix(HEADER_PREFIX) {
                    headers.insert(stripped.to_string(), value.to_string());
                }
            }
        }
        let mut message = message_from_parts(subject, nats_message.payload.to_vec(), key, headers);
        if let Some(id) = message_id {
            message.id = id;
        }
        message.reply_to = nats_message.reply.map(|subject| subject.to_string());
        message
    }
}

#[async_trait]
impl MessageQueueConsumer for NatsConsumer {
    async fn subscribe(&self, topics: &[String]) -> Result<()> {
        let mut subscriptions = self.subscriptions.lock().await;
        for topic in topics {
            if subscriptions.contains_key(topic) {
                continue;
            }
            let subscriber = self
                .client
                .subscribe(topic.clone())
                .await
                .with_context(|| format!("NATS subscribe to {topic} failed"))?;
            subscriptions.insert(topic.clone(), subscriber);
        }
        Ok(())
    }

    async fn unsubscribe(&self, topics: &[String]) -> Result<()> {
        let mut subscriptions = self.subscriptions.lock().await;
        for topic in topics {
            if let Some(mut subscriber) = subscriptions.remove(topic) {
                subscriber
                    .unsubscribe()
                    .await
                    .with_context(|| format!("NATS unsubscribe from {topic} failed"))?;
            }
        }
        Ok(())
    }

    async fn poll(&self, timeout_ms: u64) -> Result<Vec<Message>> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        let mut collected = Vec::new();
        let paused = self.paused.lock().await.clone();
        let mut subscriptions = self.subscriptions.lock().await;

        for (subject, subscriber) in subscriptions.iter_mut() {
            if paused.contains(subject) {
                continue;
            }
            while collected.len() < self.max_poll_records {
                let now = std::time::Instant::now();
                let remaining = if now >= deadline {
                    std::time::Duration::from_millis(0)
                } else {
                    deadline - now
                };
                match tokio::time::timeout(remaining, subscriber.next()).await {
                    Ok(Some(nats_message)) => {
                        collected.push(Self::decode(subject, nats_message));
                    },
                    // Subscription closed or the poll window expired.
                    Ok(None) | Err(_) => break,
                }
            }
        }
        drop(subscriptions);

        let mut delivered = self.delivered.lock().await;
        for message in &collected {
            delivered.push(message.id);
        }
        Ok(collected)
    }

    async fn commit(&self, message: &Message) -> Result<()> {
        let mut delivered = self.delivered.lock().await;
        match delivered.iter().position(|id| id == &message.id) {
            Some(index) => {
                delivered.remove(index);
                Ok(())
            },
            // Core NATS has no server-side acknowledgement, so the only real
            // check available is that this consumer delivered the message.
            None => Err(anyhow::anyhow!(
                "message {} was not delivered by this consumer",
                message.id
            )),
        }
    }

    async fn commit_batch(&self, messages: &[Message]) -> Result<()> {
        for message in messages {
            self.commit(message).await?;
        }
        Ok(())
    }

    async fn seek(&self, _topic: &str, _partition: u32, _offset: u64) -> Result<()> {
        Err(anyhow::anyhow!(
            "core NATS subjects have no retained log to seek in; use JetStream for replay"
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
        let topics: Vec<String> = self.subscriptions.lock().await.keys().cloned().collect();
        self.unsubscribe(&topics).await
    }
}
