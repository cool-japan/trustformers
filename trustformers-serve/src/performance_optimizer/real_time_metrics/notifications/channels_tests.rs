//! Delivery tests for [`super::channels`].
//!
//! The HTTP channels are exercised against a loopback `axum` server bound to
//! port 0, so nothing here touches the network. Every assertion is about what
//! the channel actually did: the bytes the server received, the status code it
//! returned, and the counters the channel kept.

use super::channels::{
    ChannelCounters, EmailNotificationChannel, HttpChannelConfig, LogNotificationChannel,
    NotificationChannel, PagerDutyNotificationChannel, SlackNotificationChannel,
    SmsNotificationChannel, WebhookNotificationChannel,
};
use super::types::{DeliveryGuarantee, Notification, NotificationPriority};
use crate::performance_optimizer::real_time_metrics::types::SeverityLevel;
use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    routing::{any, post},
    Router,
};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type Recorder = Arc<Mutex<Vec<serde_json::Value>>>;

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{addr}")
}

/// Endpoint that records every JSON body it receives and answers 200.
async fn accepting_endpoint() -> (String, Recorder) {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let router = Router::new()
        .route(
            "/notify",
            post(|State(rec): State<Recorder>, body: Bytes| async move {
                let value = serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null);
                rec.lock().expect("recorder lock").push(value);
                (StatusCode::OK, "ok")
            }),
        )
        .with_state(Arc::clone(&recorder));
    let base = spawn(router).await;
    (format!("{base}/notify"), recorder)
}

/// Endpoint that always fails with 500 and a diagnostic body.
async fn failing_endpoint() -> String {
    let router = Router::new().route(
        "/notify",
        any(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "downstream exploded") }),
    );
    let base = spawn(router).await;
    format!("{base}/notify")
}

fn notification(subject: &str) -> Notification {
    Notification {
        id: "notif-1".to_string(),
        alert_id: "alert-1".to_string(),
        channels: vec!["webhook".to_string()],
        recipients: vec!["ops@example.invalid".to_string()],
        subject: subject.to_string(),
        content: "disk usage above threshold".to_string(),
        priority: NotificationPriority::High,
        severity: SeverityLevel::Warning,
        delivery_guarantee: DeliveryGuarantee::BestEffort,
        created_at: Utc::now(),
        deadline: None,
        template: None,
        template_vars: HashMap::new(),
        metadata: HashMap::new(),
        tags: Vec::new(),
        escalation_policy: None,
        correlation_id: None,
    }
}

#[test]
fn channel_counters_report_no_success_rate_before_any_attempt() {
    let counters = ChannelCounters::default();
    assert_eq!(counters.attempts(), 0);
    assert_eq!(
        counters.success_rate(),
        None,
        "a channel that has never delivered anything has no measured success rate"
    );
}

#[test]
fn channel_counters_success_rate_is_measured_not_assumed() {
    let counters = ChannelCounters::default();
    counters.record(true);
    counters.record(false);
    assert_eq!(counters.attempts(), 2);
    assert_eq!(counters.successes(), 1);
    assert_eq!(counters.consecutive_failures(), 1);
    assert_eq!(counters.success_rate(), Some(0.5));

    counters.record(true);
    assert_eq!(counters.consecutive_failures(), 0);
}

#[tokio::test]
async fn webhook_channel_without_endpoint_refuses_to_send() {
    let channel = WebhookNotificationChannel::new().await.expect("construct channel");
    let error = channel
        .send_notification(&notification("no endpoint"))
        .await
        .expect_err("an unconfigured webhook channel must not report a delivery");
    assert!(
        error.to_string().contains("no endpoint configured"),
        "unexpected error: {error}"
    );

    let health = channel.health_check().await.expect("health check");
    assert!(!health.healthy);
    assert_eq!(health.success_rate, 0.0);
}

#[tokio::test]
async fn webhook_channel_posts_the_notification_and_reports_the_real_status() {
    let (endpoint, recorder) = accepting_endpoint().await;
    let channel = WebhookNotificationChannel::with_endpoint(endpoint);

    let result = channel.send_notification(&notification("disk pressure")).await.expect("send");

    assert!(result.success, "unexpected failure: {:?}", result.error);
    assert_eq!(
        result.response_data.get("status_code").map(String::as_str),
        Some("200")
    );
    assert!(result.delivered_at.is_some());

    let captured = recorder.lock().expect("recorder lock").clone();
    assert_eq!(captured.len(), 1, "endpoint must receive exactly one POST");
    assert_eq!(captured[0]["subject"], "disk pressure");
    assert_eq!(captured[0]["alert_id"], "alert-1");
}

#[tokio::test]
async fn webhook_channel_reports_a_server_error_instead_of_success() {
    let endpoint = failing_endpoint().await;
    let channel = WebhookNotificationChannel::with_endpoint(endpoint);

    let result = channel
        .send_notification(&notification("failing"))
        .await
        .expect("send returns a delivery result even when the endpoint fails");

    assert!(!result.success);
    assert!(result.delivered_at.is_none());
    let error = result.error.expect("failed delivery carries an error");
    assert!(error.contains("500"), "unexpected error: {error}");
    assert!(
        error.contains("downstream exploded"),
        "the endpoint's own body must survive: {error}"
    );
    assert_eq!(
        result.response_data.get("status_code").map(String::as_str),
        Some("500")
    );
}

#[tokio::test]
async fn webhook_channel_reports_transport_failure_for_an_unreachable_endpoint() {
    // Port 0 is never listening, so this exercises the transport-error branch
    // without leaving the loopback interface.
    let channel = WebhookNotificationChannel::with_config(
        HttpChannelConfig::new("http://127.0.0.1:0/notify")
            .with_timeout(std::time::Duration::from_millis(500)),
    );

    let result = channel.send_notification(&notification("unreachable")).await.expect("send");

    assert!(!result.success);
    assert!(result.error.is_some());
    assert!(!result.response_data.contains_key("status_code"));
}

#[tokio::test]
async fn slack_channel_posts_an_incoming_webhook_text_payload() {
    let (endpoint, recorder) = accepting_endpoint().await;
    let channel = SlackNotificationChannel::with_endpoint(endpoint);

    let result = channel.send_notification(&notification("queue backlog")).await.expect("send");

    assert!(result.success, "unexpected failure: {:?}", result.error);
    let captured = recorder.lock().expect("recorder lock").clone();
    assert_eq!(captured.len(), 1);
    let text = captured[0]["text"].as_str().expect("slack payload text");
    assert!(text.contains("queue backlog"), "unexpected text: {text}");
    assert!(text.contains("[HIGH]"), "unexpected text: {text}");
}

#[tokio::test]
async fn slack_channel_without_endpoint_refuses_to_send() {
    let channel = SlackNotificationChannel::new().await.expect("construct channel");
    let error = channel
        .send_notification(&notification("no endpoint"))
        .await
        .expect_err("an unconfigured slack channel must not report a delivery");
    assert!(
        error.to_string().contains("no endpoint configured"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn pagerduty_channel_posts_an_events_v2_trigger() {
    let (endpoint, recorder) = accepting_endpoint().await;
    let channel = PagerDutyNotificationChannel::with_routing_key_and_config(
        "routing-key-1",
        HttpChannelConfig::new(endpoint),
    );

    let result = channel.send_notification(&notification("node down")).await.expect("send");

    assert!(result.success, "unexpected failure: {:?}", result.error);
    let captured = recorder.lock().expect("recorder lock").clone();
    assert_eq!(captured[0]["routing_key"], "routing-key-1");
    assert_eq!(captured[0]["event_action"], "trigger");
    assert_eq!(captured[0]["dedup_key"], "alert-1");
    assert_eq!(captured[0]["payload"]["summary"], "node down");
    assert_eq!(captured[0]["payload"]["severity"], "error");
}

#[tokio::test]
async fn pagerduty_channel_without_routing_key_refuses_to_send() {
    let channel = PagerDutyNotificationChannel::new().await.expect("construct channel");
    let error = channel
        .send_notification(&notification("no key"))
        .await
        .expect_err("an unconfigured pagerduty channel must not report a delivery");
    assert!(
        error.to_string().contains("no endpoint configured"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn email_channel_reports_the_missing_transport() {
    let channel = EmailNotificationChannel::new().await.expect("construct channel");
    let error = channel
        .send_notification(&notification("mail"))
        .await
        .expect_err("email delivery has no transport and must not report success");
    assert!(
        error.to_string().contains("not implemented"),
        "unexpected error: {error}"
    );

    let health = channel.health_check().await.expect("health check");
    assert!(!health.healthy);
    assert_eq!(health.success_rate, 0.0);
}

#[tokio::test]
async fn sms_channel_reports_the_missing_transport() {
    let channel = SmsNotificationChannel::new().await.expect("construct channel");
    let error = channel
        .send_notification(&notification("sms"))
        .await
        .expect_err("SMS delivery has no transport and must not report success");
    assert!(
        error.to_string().contains("not implemented"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn log_channel_health_follows_its_own_counters() {
    let channel = LogNotificationChannel::new().await.expect("construct");

    let before = channel.health_check().await.expect("health check");
    assert!(before.healthy);
    assert_eq!(before.consecutive_failures, 0);

    channel
        .send_notification(&notification("logged"))
        .await
        .expect("log delivery always succeeds");

    let after = channel.health_check().await.expect("health check");
    assert!(after.healthy);
    assert_eq!(after.success_rate, 1.0);
}

#[tokio::test]
async fn webhook_health_reflects_observed_failures() {
    let endpoint = failing_endpoint().await;
    let channel = WebhookNotificationChannel::with_endpoint(endpoint);

    let _ = channel.send_notification(&notification("first")).await;
    let health = channel.health_check().await.expect("health check");

    assert!(
        !health.healthy,
        "a channel whose last delivery failed cannot report itself healthy"
    );
    assert_eq!(health.consecutive_failures, 1);
    assert_eq!(health.success_rate, 0.0);
}
