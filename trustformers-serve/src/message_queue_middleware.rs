use anyhow::Result;
use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
    routing::{get, post},
    Router,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::error;
use uuid::Uuid;

use crate::message_queue::{
    EventHandler, Message, MessageBatch, MessageQueueConfig, MessageQueueEvent, MessageQueueHealth,
    MessageQueueManager, MessageQueueStats,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageQueueMiddlewareConfig {
    pub enabled: bool,
    pub async_inference: bool,
    pub event_publishing: bool,
    pub request_logging: bool,
    pub metrics_collection: bool,
    pub dead_letter_queue: bool,
    pub async_topics: AsyncTopicConfig,
    pub event_topics: EventTopicConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncTopicConfig {
    pub inference_requests: String,
    pub inference_responses: String,
    pub batch_requests: String,
    pub batch_responses: String,
    pub health_checks: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventTopicConfig {
    pub model_events: String,
    pub system_events: String,
    pub user_events: String,
    pub error_events: String,
    pub audit_events: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncInferenceRequest {
    pub request_id: String,
    pub model_name: String,
    pub input: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub callback_url: Option<String>,
    pub priority: Option<i32>,
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncInferenceResponse {
    pub request_id: String,
    pub status: String,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub processing_time_ms: Option<u64>,
    pub model_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageQueueRequest {
    pub topic: String,
    pub key: Option<String>,
    pub payload: serde_json::Value,
    pub headers: Option<HashMap<String, String>>,
    pub correlation_id: Option<String>,
    pub reply_to: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageQueueResponse {
    pub message_id: String,
    pub topic: String,
    pub partition: u32,
    pub offset: u64,
    pub timestamp: chrono::DateTime<Utc>,
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMessageRequest {
    pub topic: String,
    pub messages: Vec<BatchMessageItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMessageItem {
    pub key: Option<String>,
    pub payload: serde_json::Value,
    pub headers: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMessageResponse {
    pub batch_id: String,
    pub success_count: usize,
    pub failure_count: usize,
    pub total_size: usize,
    pub results: Vec<MessageQueueResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumeRequest {
    pub topics: Vec<String>,
    pub timeout_ms: Option<u64>,
    pub max_messages: Option<usize>,
    pub auto_commit: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumeResponse {
    pub messages: Vec<ConsumedMessage>,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumedMessage {
    pub id: String,
    pub topic: String,
    pub key: Option<String>,
    pub payload: serde_json::Value,
    pub headers: HashMap<String, String>,
    pub timestamp: chrono::DateTime<Utc>,
    pub partition: Option<u32>,
    pub offset: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionRequest {
    pub topics: Vec<String>,
    pub consumer_group: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRequest {
    pub message_ids: Vec<String>,
}

pub struct MessageQueueMiddleware {
    pub config: MessageQueueMiddlewareConfig,
    pub manager: Arc<MessageQueueManager>,
    pub message_cache: Arc<RwLock<HashMap<String, Message>>>,
    pub stats: Arc<RwLock<MessageQueueStats>>,
}

impl MessageQueueMiddleware {
    pub async fn new(
        mq_config: MessageQueueConfig,
        middleware_config: MessageQueueMiddlewareConfig,
    ) -> Result<Self> {
        let manager = Arc::new(MessageQueueManager::new(mq_config).await?);
        let message_cache = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(RwLock::new(MessageQueueStats::default()));

        let middleware = Self {
            config: middleware_config,
            manager,
            message_cache,
            stats,
        };

        // Register event handlers
        middleware.setup_event_handlers().await?;

        Ok(middleware)
    }

    async fn setup_event_handlers(&self) -> Result<()> {
        if self.config.event_publishing {
            let stats = self.stats.clone();
            let event_handler: EventHandler = Box::new(move |event| {
                let stats = stats.clone();
                tokio::spawn(async move {
                    let mut stats = stats.write().await;
                    match event {
                        MessageQueueEvent::MessageProduced(_) => {
                            stats.messages_produced += 1;
                        },
                        MessageQueueEvent::MessageConsumed(_) => {
                            stats.messages_consumed += 1;
                        },
                        MessageQueueEvent::Error(_) => {
                            stats.errors += 1;
                        },
                        _ => {},
                    }
                });
            });

            self.manager
                .register_event_handler("middleware_stats".to_string(), event_handler)
                .await;
        }

        Ok(())
    }

    pub fn routes(&self) -> Router {
        Router::new()
            .route("/message-queue/send", post(send_message))
            .route("/message-queue/batch", post(send_batch))
            .route("/message-queue/consume", post(consume_messages))
            .route("/message-queue/subscribe", post(subscribe_topics))
            .route("/message-queue/commit", post(commit_messages))
            .route("/message-queue/stats", get(get_stats))
            .route("/message-queue/health", get(get_health))
            .route("/message-queue/async/inference", post(async_inference))
            .route(
                "/message-queue/async/inference/:request_id",
                get(get_async_result),
            )
            .with_state(Arc::new(self.clone()))
    }

    pub async fn publish_inference_request(&self, request: &AsyncInferenceRequest) -> Result<()> {
        if !self.config.async_inference {
            return Ok(());
        }

        let message = Message {
            id: Uuid::new_v4(),
            topic: self.config.async_topics.inference_requests.clone(),
            key: Some(request.request_id.clone()),
            payload: serde_json::to_vec(request)?,
            headers: {
                let mut headers = HashMap::new();
                headers.insert("model_name".to_string(), request.model_name.clone());
                headers.insert("request_type".to_string(), "inference".to_string());
                if let Some(priority) = request.priority {
                    headers.insert("priority".to_string(), priority.to_string());
                }
                headers
            },
            timestamp: Utc::now(),
            partition: None,
            offset: None,
            delivery_count: 1,
            correlation_id: Some(request.request_id.clone()),
            reply_to: Some(self.config.async_topics.inference_responses.clone()),
        };

        self.manager.send_message(message).await?;
        Ok(())
    }

    pub async fn publish_inference_response(
        &self,
        response: &AsyncInferenceResponse,
    ) -> Result<()> {
        if !self.config.async_inference {
            return Ok(());
        }

        let message = Message {
            id: Uuid::new_v4(),
            topic: self.config.async_topics.inference_responses.clone(),
            key: Some(response.request_id.clone()),
            payload: serde_json::to_vec(response)?,
            headers: {
                let mut headers = HashMap::new();
                headers.insert("status".to_string(), response.status.clone());
                headers.insert("request_id".to_string(), response.request_id.clone());
                headers
            },
            timestamp: Utc::now(),
            partition: None,
            offset: None,
            delivery_count: 1,
            correlation_id: Some(response.request_id.clone()),
            reply_to: None,
        };

        self.manager.send_message(message).await?;
        Ok(())
    }

    pub async fn publish_system_event(
        &self,
        event_type: &str,
        data: serde_json::Value,
    ) -> Result<()> {
        if !self.config.event_publishing {
            return Ok(());
        }

        let message = Message {
            id: Uuid::new_v4(),
            topic: self.config.event_topics.system_events.clone(),
            key: Some(event_type.to_string()),
            payload: serde_json::to_vec(&data)?,
            headers: {
                let mut headers = HashMap::new();
                headers.insert("event_type".to_string(), event_type.to_string());
                headers.insert("source".to_string(), "trustformers-serve".to_string());
                headers
            },
            timestamp: Utc::now(),
            partition: None,
            offset: None,
            delivery_count: 1,
            correlation_id: None,
            reply_to: None,
        };

        self.manager.send_message(message).await?;
        Ok(())
    }

    pub async fn publish_audit_event(
        &self,
        event_type: &str,
        user_id: &str,
        data: serde_json::Value,
    ) -> Result<()> {
        if !self.config.event_publishing {
            return Ok(());
        }

        let message = Message {
            id: Uuid::new_v4(),
            topic: self.config.event_topics.audit_events.clone(),
            key: Some(user_id.to_string()),
            payload: serde_json::to_vec(&data)?,
            headers: {
                let mut headers = HashMap::new();
                headers.insert("event_type".to_string(), event_type.to_string());
                headers.insert("user_id".to_string(), user_id.to_string());
                headers.insert("source".to_string(), "trustformers-serve".to_string());
                headers
            },
            timestamp: Utc::now(),
            partition: None,
            offset: None,
            delivery_count: 1,
            correlation_id: None,
            reply_to: None,
        };

        self.manager.send_message(message).await?;
        Ok(())
    }

    pub async fn request_middleware(
        State(middleware): State<Arc<MessageQueueMiddleware>>,
        request: axum::extract::Request,
        next: Next,
    ) -> Result<Response, StatusCode> {
        if !middleware.config.request_logging {
            return Ok(next.run(request).await);
        }

        let uri = request.uri().clone();
        let method = request.method().clone();
        let headers = request.headers().clone();

        let start_time = std::time::Instant::now();
        let response = next.run(request).await;
        let duration = start_time.elapsed();

        // Log request to message queue
        let log_data = serde_json::json!({
            "method": method.to_string(),
            "uri": uri.to_string(),
            "status": response.status().as_u16(),
            "duration_ms": duration.as_millis(),
            "timestamp": Utc::now(),
            "headers": headers.iter().map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string())).collect::<HashMap<String, String>>()
        });

        if let Err(e) = middleware.publish_system_event("http_request", log_data).await {
            error!("Failed to publish request log: {}", e);
        }

        Ok(response)
    }
}

impl Clone for MessageQueueMiddleware {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            manager: self.manager.clone(),
            message_cache: self.message_cache.clone(),
            stats: self.stats.clone(),
        }
    }
}

// HTTP handlers
async fn send_message(
    State(middleware): State<Arc<MessageQueueMiddleware>>,
    Json(request): Json<MessageQueueRequest>,
) -> Result<Json<MessageQueueResponse>, StatusCode> {
    let message = Message {
        id: Uuid::new_v4(),
        topic: request.topic,
        key: request.key,
        payload: serde_json::to_vec(&request.payload).map_err(|_| StatusCode::BAD_REQUEST)?,
        headers: request.headers.unwrap_or_default(),
        timestamp: Utc::now(),
        partition: None,
        offset: None,
        delivery_count: 1,
        correlation_id: request.correlation_id,
        reply_to: request.reply_to,
    };

    let result = middleware
        .manager
        .send_message(message)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(MessageQueueResponse {
        message_id: result.message_id.to_string(),
        topic: result.topic,
        partition: result.partition,
        offset: result.offset,
        timestamp: result.timestamp,
        size: result.size,
    }))
}

async fn send_batch(
    State(middleware): State<Arc<MessageQueueMiddleware>>,
    Json(request): Json<BatchMessageRequest>,
) -> Result<Json<BatchMessageResponse>, StatusCode> {
    let messages: Result<Vec<Message>, serde_json::Error> = request
        .messages
        .into_iter()
        .map(|item| {
            Ok(Message {
                id: Uuid::new_v4(),
                topic: request.topic.clone(),
                key: item.key,
                payload: serde_json::to_vec(&item.payload)?,
                headers: item.headers.unwrap_or_default(),
                timestamp: Utc::now(),
                partition: None,
                offset: None,
                delivery_count: 1,
                correlation_id: None,
                reply_to: None,
            })
        })
        .collect();

    let messages = messages.map_err(|_| StatusCode::BAD_REQUEST)?;

    let batch = MessageBatch {
        messages,
        topic: request.topic,
        batch_id: Uuid::new_v4(),
        created_at: Utc::now(),
    };

    let result = middleware
        .manager
        .send_batch(batch)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let responses = result
        .results
        .into_iter()
        .map(|r| MessageQueueResponse {
            message_id: r.message_id.to_string(),
            topic: r.topic,
            partition: r.partition,
            offset: r.offset,
            timestamp: r.timestamp,
            size: r.size,
        })
        .collect();

    Ok(Json(BatchMessageResponse {
        batch_id: result.batch_id.to_string(),
        success_count: result.success_count,
        failure_count: result.failure_count,
        total_size: result.total_size,
        results: responses,
    }))
}

async fn consume_messages(
    State(middleware): State<Arc<MessageQueueMiddleware>>,
    Json(request): Json<ConsumeRequest>,
) -> Result<Json<ConsumeResponse>, StatusCode> {
    let timeout = request.timeout_ms.unwrap_or(5000);
    let messages = middleware
        .manager
        .consume_messages(timeout)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut cache = middleware.message_cache.write().await;
    let consumed_messages: Vec<ConsumedMessage> = messages
        .into_iter()
        .map(|msg| {
            let consumed = ConsumedMessage {
                id: msg.id.to_string(),
                topic: msg.topic.clone(),
                key: msg.key.clone(),
                payload: serde_json::from_slice(&msg.payload).unwrap_or(serde_json::Value::Null),
                headers: msg.headers.clone(),
                timestamp: msg.timestamp,
                partition: msg.partition,
                offset: msg.offset,
            };
            cache.insert(msg.id.to_string(), msg);
            consumed
        })
        .collect();

    let count = consumed_messages.len();

    Ok(Json(ConsumeResponse {
        messages: consumed_messages,
        count,
    }))
}

async fn subscribe_topics(
    State(middleware): State<Arc<MessageQueueMiddleware>>,
    Json(request): Json<SubscriptionRequest>,
) -> Result<StatusCode, StatusCode> {
    middleware
        .manager
        .subscribe(&request.topics)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(StatusCode::OK)
}

async fn commit_messages(
    State(middleware): State<Arc<MessageQueueMiddleware>>,
    Json(request): Json<CommitRequest>,
) -> Result<StatusCode, StatusCode> {
    let cache = middleware.message_cache.read().await;

    for message_id in request.message_ids {
        if let Some(message) = cache.get(&message_id) {
            middleware
                .manager
                .commit_message(message)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
    }

    Ok(StatusCode::OK)
}

#[axum::debug_handler]
async fn get_stats(
    State(middleware): State<Arc<MessageQueueMiddleware>>,
) -> Result<Json<MessageQueueStats>, StatusCode> {
    let stats = middleware.manager.get_stats().await;
    Ok(Json(stats))
}

async fn get_health(
    State(middleware): State<Arc<MessageQueueMiddleware>>,
) -> Result<Json<MessageQueueHealth>, StatusCode> {
    let health = middleware
        .manager
        .health_check()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(health))
}

async fn async_inference(
    State(middleware): State<Arc<MessageQueueMiddleware>>,
    Json(request): Json<AsyncInferenceRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    middleware
        .publish_inference_request(&request)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(serde_json::json!({
        "request_id": request.request_id,
        "status": "queued",
        "message": "Inference request queued for processing"
    })))
}

async fn get_async_result(
    State(_middleware): State<Arc<MessageQueueMiddleware>>,
    Path(request_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // In a real implementation, this would check the results queue/cache
    // For now, return a placeholder response
    Ok(Json(serde_json::json!({
        "request_id": request_id,
        "status": "processing",
        "message": "Result not yet available"
    })))
}

impl Default for MessageQueueMiddlewareConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            async_inference: true,
            event_publishing: true,
            request_logging: true,
            metrics_collection: true,
            dead_letter_queue: true,
            async_topics: AsyncTopicConfig {
                inference_requests: "inference-requests".to_string(),
                inference_responses: "inference-responses".to_string(),
                batch_requests: "batch-requests".to_string(),
                batch_responses: "batch-responses".to_string(),
                health_checks: "health-checks".to_string(),
            },
            event_topics: EventTopicConfig {
                model_events: "model-events".to_string(),
                system_events: "system-events".to_string(),
                user_events: "user-events".to_string(),
                error_events: "error-events".to_string(),
                audit_events: "audit-events".to_string(),
            },
        }
    }
}
