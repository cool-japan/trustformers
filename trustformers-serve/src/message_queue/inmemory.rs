//! A real in-process message queue.
//!
//! This backend is not a placeholder: it is an append-only per-topic log with
//! consumer-group offsets, at-least-once delivery, explicit acknowledgement and
//! redelivery of messages that were never acknowledged. Producers and consumers
//! created independently (as [`MessageQueueManager`](super::MessageQueueManager)
//! does) find each other through a process-wide broker registry keyed by the
//! configuration's `connection_string`, so a message sent by the producer really
//! is returned by the consumer.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use tokio::sync::{Mutex, Notify};
use uuid::Uuid;

use super::{
    BatchResult, Message, MessageBatch, MessageQueueConfig, MessageQueueConsumer,
    MessageQueueProducer, MessageResult, ProducerCallback, TransactionId,
};

/// How long an unacknowledged message stays invisible before redelivery.
const DEFAULT_VISIBILITY_TIMEOUT: Duration = Duration::from_secs(30);

/// Process-wide registry of in-memory brokers, keyed by connection string.
static BROKERS: Lazy<DashMap<String, Arc<InMemoryBroker>>> = Lazy::new(DashMap::new);

/// Look up (or create) the broker for `connection_string`.
fn broker_for(connection_string: &str) -> Arc<InMemoryBroker> {
    BROKERS
        .entry(connection_string.to_string())
        .or_insert_with(|| Arc::new(InMemoryBroker::default()))
        .clone()
}

/// Remove a broker from the registry. Used by tests to isolate state.
pub fn reset_broker(connection_string: &str) {
    BROKERS.remove(connection_string);
}

/// One in-flight (delivered but unacknowledged) message.
#[derive(Debug)]
struct InFlight {
    message: Message,
    delivered_at: Instant,
}

/// The append-only log and delivery bookkeeping for one topic.
#[derive(Debug, Default)]
struct TopicState {
    /// Every message ever produced to the topic, in publication order. The index
    /// is the message's offset.
    log: Mutex<Vec<Message>>,
    /// Next offset to hand out, per consumer group.
    next_offset: Mutex<HashMap<String, u64>>,
    /// Delivered-but-unacknowledged messages, keyed by message id.
    in_flight: Mutex<HashMap<Uuid, InFlight>>,
    /// Woken whenever a message is appended.
    notify: Notify,
}

/// An in-process broker: a set of topics plus paused-topic state.
#[derive(Debug, Default)]
pub struct InMemoryBroker {
    topics: DashMap<String, Arc<TopicState>>,
    paused: DashMap<String, bool>,
}

impl InMemoryBroker {
    fn topic(&self, name: &str) -> Arc<TopicState> {
        self.topics
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(TopicState::default()))
            .clone()
    }

    /// Number of messages currently held in a topic's log.
    pub async fn log_len(&self, topic: &str) -> usize {
        self.topic(topic).log.lock().await.len()
    }

    /// Number of delivered-but-unacknowledged messages across all topics.
    pub async fn in_flight_count(&self) -> usize {
        let mut total = 0;
        for entry in self.topics.iter() {
            total += entry.value().in_flight.lock().await.len();
        }
        total
    }

    async fn append(&self, mut message: Message) -> MessageResult {
        let topic = self.topic(&message.topic);
        let mut log = topic.log.lock().await;
        let offset = log.len() as u64;
        message.partition = Some(0);
        message.offset = Some(offset);
        let result = MessageResult {
            message_id: message.id,
            topic: message.topic.clone(),
            partition: 0,
            offset,
            timestamp: message.timestamp,
            size: message.payload.len(),
        };
        log.push(message);
        drop(log);
        topic.notify.notify_waiters();
        result
    }

    /// Reclaim at most `max` messages whose visibility timeout expired, so an
    /// unacknowledged message is delivered again instead of being lost.
    ///
    /// Only the messages actually returned leave `in_flight`; the rest stay
    /// pending and are reclaimed by a later poll. Removing more than the caller
    /// can take would drop them permanently, because their offsets are already
    /// behind the consumer-group cursor.
    async fn reclaim_expired(
        &self,
        topic_name: &str,
        timeout: Duration,
        max: usize,
    ) -> Vec<Message> {
        if max == 0 {
            return Vec::new();
        }
        let topic = self.topic(topic_name);
        let mut in_flight = topic.in_flight.lock().await;
        let expired: Vec<Uuid> = in_flight
            .iter()
            .filter(|(_, entry)| entry.delivered_at.elapsed() >= timeout)
            .map(|(id, _)| *id)
            .take(max)
            .collect();
        let mut reclaimed = Vec::with_capacity(expired.len());
        for id in expired {
            if let Some(mut entry) = in_flight.remove(&id) {
                entry.message.delivery_count += 1;
                reclaimed.push(entry.message);
            }
        }
        reclaimed
    }

    async fn fetch(
        &self,
        topic_name: &str,
        group: &str,
        max: usize,
        visibility_timeout: Duration,
    ) -> Vec<Message> {
        if self.paused.get(topic_name).map(|flag| *flag).unwrap_or(false) {
            return Vec::new();
        }

        let mut batch = self.reclaim_expired(topic_name, visibility_timeout, max).await;

        if batch.len() < max {
            let topic = self.topic(topic_name);
            let log = topic.log.lock().await;
            let mut offsets = topic.next_offset.lock().await;
            let cursor = offsets.entry(group.to_string()).or_insert(0);
            while batch.len() < max && (*cursor as usize) < log.len() {
                let mut message = log[*cursor as usize].clone();
                message.delivery_count += 1;
                *cursor += 1;
                batch.push(message);
            }
        }

        if !batch.is_empty() {
            let topic = self.topic(topic_name);
            let mut in_flight = topic.in_flight.lock().await;
            for message in &batch {
                in_flight.insert(
                    message.id,
                    InFlight {
                        message: message.clone(),
                        delivered_at: Instant::now(),
                    },
                );
            }
        }

        batch
    }

    async fn acknowledge(&self, message: &Message) -> bool {
        let topic = self.topic(&message.topic);
        let mut in_flight = topic.in_flight.lock().await;
        in_flight.remove(&message.id).is_some()
    }

    async fn seek(&self, topic_name: &str, group: &str, offset: u64) {
        let topic = self.topic(topic_name);
        topic.next_offset.lock().await.insert(group.to_string(), offset);
    }
}

/// In-memory producer.
#[derive(Debug)]
pub struct InMemoryProducer {
    broker: Arc<InMemoryBroker>,
    /// Messages buffered inside an open transaction, keyed by transaction id.
    transactions: Mutex<HashMap<TransactionId, Vec<Message>>>,
    /// Transaction that new sends are buffered into, if any.
    active_transaction: Mutex<Option<TransactionId>>,
}

impl InMemoryProducer {
    /// Create a producer bound to the broker named by `config.connection_string`.
    ///
    /// # Errors
    ///
    /// Never fails; the signature matches the other backends.
    pub async fn new(config: &MessageQueueConfig) -> Result<Self> {
        Ok(Self {
            broker: broker_for(&config.connection_string),
            transactions: Mutex::new(HashMap::new()),
            active_transaction: Mutex::new(None),
        })
    }

    /// The broker this producer publishes to (used by tests for assertions).
    pub fn broker(&self) -> Arc<InMemoryBroker> {
        Arc::clone(&self.broker)
    }
}

#[async_trait]
impl MessageQueueProducer for InMemoryProducer {
    async fn send_message(&self, message: Message) -> Result<MessageResult> {
        let active = self.active_transaction.lock().await.clone();
        if let Some(transaction_id) = active {
            let result = MessageResult {
                message_id: message.id,
                topic: message.topic.clone(),
                partition: 0,
                // Buffered: the real offset is assigned at commit time.
                offset: u64::MAX,
                timestamp: message.timestamp,
                size: message.payload.len(),
            };
            self.transactions.lock().await.entry(transaction_id).or_default().push(message);
            return Ok(result);
        }
        Ok(self.broker.append(message).await)
    }

    async fn send_batch(&self, batch: MessageBatch) -> Result<BatchResult> {
        let batch_id = batch.batch_id;
        let mut results = Vec::with_capacity(batch.messages.len());
        let mut total_size = 0usize;
        for message in batch.messages {
            let result = self.send_message(message).await?;
            total_size += result.size;
            results.push(result);
        }
        Ok(BatchResult {
            batch_id,
            success_count: results.len(),
            failure_count: 0,
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
            self.broker.append(message).await;
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
        // Appends are synchronous, so nothing can still be buffered outside of
        // an explicitly opened transaction.
        let active = self.active_transaction.lock().await.clone();
        if let Some(id) = active {
            return Err(anyhow::anyhow!(
                "cannot flush while transaction {id} is open: commit or abort it first"
            ));
        }
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        let open: Vec<TransactionId> = self.transactions.lock().await.keys().cloned().collect();
        for id in open {
            self.abort_transaction(id).await?;
        }
        Ok(())
    }
}

/// In-memory consumer with consumer-group offsets and explicit acknowledgement.
#[derive(Debug)]
pub struct InMemoryConsumer {
    broker: Arc<InMemoryBroker>,
    group: String,
    max_poll_records: usize,
    visibility_timeout: Duration,
    subscriptions: Mutex<Vec<String>>,
}

impl InMemoryConsumer {
    /// Create a consumer bound to the broker named by `config.connection_string`.
    ///
    /// The consumer starts subscribed to `config.topics`, matching the
    /// behaviour of the broker-backed consumers.
    ///
    /// # Errors
    ///
    /// Never fails; the signature matches the other backends.
    pub async fn new(config: &MessageQueueConfig) -> Result<Self> {
        Ok(Self {
            broker: broker_for(&config.connection_string),
            group: config.consumer_group.clone().unwrap_or_else(|| "default".to_string()),
            max_poll_records: config.batch_size.max(1),
            visibility_timeout: DEFAULT_VISIBILITY_TIMEOUT,
            subscriptions: Mutex::new(config.topics.clone()),
        })
    }

    /// Override the redelivery timeout for unacknowledged messages.
    #[must_use]
    pub fn with_visibility_timeout(mut self, timeout: Duration) -> Self {
        self.visibility_timeout = timeout;
        self
    }

    /// The broker this consumer reads from (used by tests for assertions).
    pub fn broker(&self) -> Arc<InMemoryBroker> {
        Arc::clone(&self.broker)
    }
}

#[async_trait]
impl MessageQueueConsumer for InMemoryConsumer {
    async fn subscribe(&self, topics: &[String]) -> Result<()> {
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
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            let topics = self.subscriptions.lock().await.clone();
            let mut collected = Vec::new();
            for topic in &topics {
                if collected.len() >= self.max_poll_records {
                    break;
                }
                let remaining = self.max_poll_records - collected.len();
                collected.extend(
                    self.broker.fetch(topic, &self.group, remaining, self.visibility_timeout).await,
                );
            }
            if !collected.is_empty() {
                return Ok(collected);
            }

            let now = Instant::now();
            if now >= deadline {
                return Ok(Vec::new());
            }

            // With nothing subscribed there is no publication to wait for, and
            // `select_all` panics on an empty iterator — wait out the deadline.
            if topics.is_empty() {
                tokio::time::sleep(deadline - now).await;
                return Ok(Vec::new());
            }

            // Wait for a publication on any subscribed topic, or the deadline.
            let waiters: Vec<_> = topics.iter().map(|topic| self.broker.topic(topic)).collect();
            let notified = futures::future::select_all(
                waiters.iter().map(|state| Box::pin(state.notify.notified())),
            );
            let _ = tokio::time::timeout(deadline - now, notified).await;
        }
    }

    async fn commit(&self, message: &Message) -> Result<()> {
        if !self.broker.acknowledge(message).await {
            return Err(anyhow::anyhow!(
                "message {} of topic {} is not in flight: it was never delivered, \
                 was already acknowledged, or its visibility timeout expired",
                message.id,
                message.topic
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
        self.broker.seek(topic, &self.group, offset).await;
        Ok(())
    }

    async fn pause(&self, topics: &[String]) -> Result<()> {
        for topic in topics {
            self.broker.paused.insert(topic.clone(), true);
        }
        Ok(())
    }

    async fn resume(&self, topics: &[String]) -> Result<()> {
        for topic in topics {
            self.broker.paused.insert(topic.clone(), false);
        }
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.subscriptions.lock().await.clear();
        Ok(())
    }
}

/// Build a `Message` with sensible defaults. Shared by the broker backends.
pub(super) fn message_from_parts(
    topic: &str,
    payload: Vec<u8>,
    key: Option<String>,
    headers: HashMap<String, String>,
) -> Message {
    Message {
        id: Uuid::new_v4(),
        topic: topic.to_string(),
        key,
        payload,
        headers,
        timestamp: Utc::now(),
        partition: None,
        offset: None,
        delivery_count: 1,
        correlation_id: None,
        reply_to: None,
    }
}
