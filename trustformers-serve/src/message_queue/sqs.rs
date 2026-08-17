//! Amazon SQS backend built on [`aws_sdk_sqs`].
//!
//! `send_message` issues a real `SendMessage` call, `poll` issues
//! `ReceiveMessage` with long polling, and `commit` issues `DeleteMessage`
//! against the receipt handle — so an uncommitted message really does become
//! visible again after its visibility timeout instead of vanishing.
//!
//! The provider must be given SQS clients; without them every operation returns
//! an error rather than fabricating a send result.

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use aws_sdk_sqs::types::MessageAttributeValue;
use aws_sdk_sqs::Client;
use tokio::sync::Mutex;
use uuid::Uuid;

use super::inmemory::message_from_parts;
use super::{
    BatchResult, Message, MessageBatch, MessageQueueConfig, MessageQueueConsumer,
    MessageQueueProducer, MessageResult, ProducerCallback, TransactionId,
};

/// Message attribute carrying the TrustformeRS message id.
const ATTR_MESSAGE_ID: &str = "TrustformersMessageId";
/// Message attribute carrying the partition key.
const ATTR_KEY: &str = "TrustformersKey";
/// Prefix for user headers carried as message attributes.
const ATTR_HEADER_PREFIX: &str = "TrustformersH";

/// Build an SQS client from the ambient AWS configuration.
async fn build_client() -> Client {
    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    Client::new(&config)
}

/// Resolve the queue URL for a topic.
///
/// `connection_string` may be a full queue URL prefix
/// (`https://sqs.us-east-1.amazonaws.com/123456789012`) in which case the topic
/// is appended as the queue name, or a full URL already containing the topic.
fn queue_url(config: &MessageQueueConfig, topic: &str) -> String {
    let base = config.connection_string.trim_end_matches('/');
    if base.ends_with(topic) {
        base.to_string()
    } else {
        format!("{base}/{topic}")
    }
}

fn string_attribute(value: &str) -> Result<MessageAttributeValue> {
    MessageAttributeValue::builder()
        .data_type("String")
        .string_value(value)
        .build()
        .context("cannot build SQS message attribute")
}

/// Amazon SQS producer.
#[derive(Debug)]
pub struct SqsProducer {
    client: Client,
    config: MessageQueueConfig,
    transactions: Mutex<HashMap<TransactionId, Vec<Message>>>,
    active_transaction: Mutex<Option<TransactionId>>,
}

impl SqsProducer {
    /// Build a producer using the ambient AWS configuration.
    ///
    /// # Errors
    ///
    /// Never fails at construction; SQS errors surface on the first call.
    pub async fn new(config: &MessageQueueConfig) -> Result<Self> {
        Ok(Self::with_client(build_client().await, config.clone()))
    }

    /// Build a producer from an existing SQS client (used by tests with a
    /// mock endpoint).
    pub fn with_client(client: Client, config: MessageQueueConfig) -> Self {
        Self {
            client,
            config,
            transactions: Mutex::new(HashMap::new()),
            active_transaction: Mutex::new(None),
        }
    }

    async fn send(&self, message: &Message) -> Result<MessageResult> {
        let url = queue_url(&self.config, &message.topic);
        let body = if message.payload.is_empty() {
            String::new()
        } else {
            // SQS bodies are text; base64 keeps binary payloads intact.
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.encode(&message.payload)
        };

        let mut request = self
            .client
            .send_message()
            .queue_url(&url)
            .message_body(body)
            .message_attributes(ATTR_MESSAGE_ID, string_attribute(&message.id.to_string())?);
        if let Some(key) = &message.key {
            request = request.message_attributes(ATTR_KEY, string_attribute(key)?);
        }
        for (name, value) in &message.headers {
            request = request.message_attributes(
                format!("{ATTR_HEADER_PREFIX}{name}"),
                string_attribute(value)?,
            );
        }

        request
            .send()
            .await
            .with_context(|| format!("SQS SendMessage to {url} failed"))?;

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
impl MessageQueueProducer for SqsProducer {
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
        self.send(&message).await
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
            self.send(&message).await?;
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

/// Amazon SQS consumer.
#[derive(Debug)]
pub struct SqsConsumer {
    client: Client,
    config: MessageQueueConfig,
    subscriptions: Mutex<Vec<String>>,
    paused: Mutex<Vec<String>>,
    /// Receipt handles of received-but-undeleted messages.
    in_flight: Mutex<HashMap<Uuid, (String, String)>>,
    max_poll_records: usize,
}

impl SqsConsumer {
    /// Build a consumer using the ambient AWS configuration.
    ///
    /// # Errors
    ///
    /// Never fails at construction; SQS errors surface on the first call.
    pub async fn new(config: &MessageQueueConfig) -> Result<Self> {
        Ok(Self::with_client(build_client().await, config.clone()))
    }

    /// Build a consumer from an existing SQS client (used by tests with a
    /// mock endpoint).
    pub fn with_client(client: Client, config: MessageQueueConfig) -> Self {
        let topics = config.topics.clone();
        let max_poll_records = config.batch_size.clamp(1, 10);
        Self {
            client,
            config,
            subscriptions: Mutex::new(topics),
            paused: Mutex::new(Vec::new()),
            in_flight: Mutex::new(HashMap::new()),
            max_poll_records,
        }
    }

    fn decode(topic: &str, sqs_message: &aws_sdk_sqs::types::Message) -> Message {
        use base64::Engine as _;
        let payload = sqs_message
            .body()
            .map(|body| {
                base64::engine::general_purpose::STANDARD
                    .decode(body)
                    .unwrap_or_else(|_| body.as_bytes().to_vec())
            })
            .unwrap_or_default();

        let mut headers = HashMap::new();
        let mut key = None;
        let mut message_id = None;
        if let Some(attributes) = sqs_message.message_attributes() {
            for (name, value) in attributes {
                let Some(text) = value.string_value() else {
                    continue;
                };
                if name == ATTR_MESSAGE_ID {
                    message_id = Uuid::parse_str(text).ok();
                } else if name == ATTR_KEY {
                    key = Some(text.to_string());
                } else if let Some(stripped) = name.strip_prefix(ATTR_HEADER_PREFIX) {
                    headers.insert(stripped.to_string(), text.to_string());
                }
            }
        }

        let mut message = message_from_parts(topic, payload, key, headers);
        if let Some(id) = message_id {
            message.id = id;
        }
        message
    }
}

#[async_trait]
impl MessageQueueConsumer for SqsConsumer {
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
        let topics: Vec<String> = {
            let subscriptions = self.subscriptions.lock().await;
            let paused = self.paused.lock().await;
            subscriptions.iter().filter(|t| !paused.contains(t)).cloned().collect()
        };

        // SQS long polling is expressed in whole seconds, capped at 20.
        let wait_seconds = timeout_ms.div_ceil(1000).min(20) as i32;
        let mut collected = Vec::new();
        for topic in topics {
            if collected.len() >= self.max_poll_records {
                break;
            }
            let url = queue_url(&self.config, &topic);
            let output = self
                .client
                .receive_message()
                .queue_url(&url)
                .max_number_of_messages((self.max_poll_records - collected.len()).min(10) as i32)
                .wait_time_seconds(wait_seconds)
                .message_attribute_names("All")
                .send()
                .await
                .with_context(|| format!("SQS ReceiveMessage on {url} failed"))?;

            for sqs_message in output.messages() {
                let message = Self::decode(&topic, sqs_message);
                if let Some(receipt) = sqs_message.receipt_handle() {
                    self.in_flight
                        .lock()
                        .await
                        .insert(message.id, (url.clone(), receipt.to_string()));
                }
                collected.push(message);
            }
        }
        Ok(collected)
    }

    async fn commit(&self, message: &Message) -> Result<()> {
        let (url, receipt) = self.in_flight.lock().await.remove(&message.id).ok_or_else(|| {
            anyhow::anyhow!("message {} was not delivered by this consumer", message.id)
        })?;
        self.client
            .delete_message()
            .queue_url(&url)
            .receipt_handle(receipt)
            .send()
            .await
            .with_context(|| format!("SQS DeleteMessage on {url} failed"))?;
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
            "SQS queues have no seekable offsets; messages are redelivered after their \
             visibility timeout instead"
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
        // Anything still in flight becomes visible again after its visibility
        // timeout; nothing is deleted here.
        self.subscriptions.lock().await.clear();
        Ok(())
    }
}
