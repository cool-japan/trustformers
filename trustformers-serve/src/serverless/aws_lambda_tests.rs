//! Integration tests for [`AwsLambdaProvider`] against a local mock AWS endpoint.
//!
//! A real `aws_sdk_lambda::Client` / `aws_sdk_cloudwatch::Client` is built with
//! `endpoint_url` pointing at an in-process axum server, so the request shape
//! the SDK actually puts on the wire is asserted, and the response the provider
//! parses is a real AWS-format payload. No network access and no credentials
//! are involved.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{delete, post, put};
use axum::Router;
use tokio::sync::Mutex;

use super::errors::ServerlessError;
use super::functions::ServerlessProviderTrait;
use super::types::{AwsLambdaProvider, PackageType, ServerlessConfig, Trigger, TriggerType};

/// Every request the mock endpoint received, in arrival order.
#[derive(Debug, Clone, Default)]
struct RecordedRequest {
    method: String,
    path: String,
    body: String,
}

type Recorder = Arc<Mutex<Vec<RecordedRequest>>>;

fn config(function_name: &str, package: PackageType, source: &str) -> ServerlessConfig {
    let mut config = super::functions::tests::aws_config(function_name);
    config.deployment_package.package_type = package;
    config.deployment_package.source_location = source.to_string();
    config
}

async fn record(recorder: &Recorder, method: &str, path: &str, body: &Bytes) {
    recorder.lock().await.push(RecordedRequest {
        method: method.to_string(),
        path: path.to_string(),
        body: String::from_utf8_lossy(body).to_string(),
    });
}

/// Spawn `router` on an ephemeral loopback port and return its base URL.
async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{addr}")
}

fn lambda_client(endpoint: &str) -> aws_sdk_lambda::Client {
    let config = aws_sdk_lambda::Config::builder()
        .behavior_version(aws_sdk_lambda::config::BehaviorVersion::latest())
        .region(aws_sdk_lambda::config::Region::new("us-east-1"))
        .credentials_provider(aws_sdk_lambda::config::Credentials::new(
            "AKIDTEST", "secret", None, None, "mock",
        ))
        .endpoint_url(endpoint)
        .retry_config(aws_sdk_lambda::config::retry::RetryConfig::disabled())
        .build();
    aws_sdk_lambda::Client::from_conf(config)
}

fn cloudwatch_client(endpoint: &str) -> aws_sdk_cloudwatch::Client {
    let config = aws_sdk_cloudwatch::Config::builder()
        .behavior_version(aws_sdk_cloudwatch::config::BehaviorVersion::latest())
        .region(aws_sdk_cloudwatch::config::Region::new("us-east-1"))
        .credentials_provider(aws_sdk_cloudwatch::config::Credentials::new(
            "AKIDTEST", "secret", None, None, "mock",
        ))
        .endpoint_url(endpoint)
        .retry_config(aws_sdk_cloudwatch::config::retry::RetryConfig::disabled())
        .build();
    aws_sdk_cloudwatch::Client::from_conf(config)
}

// ── Lambda control-plane mock ────────────────────────────────────────────────

async fn create_function(
    State(recorder): State<Recorder>,
    body: Bytes,
) -> (StatusCode, HeaderMap, String) {
    record(&recorder, "POST", "/2015-03-31/functions", &body).await;
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().expect("header"));
    (
        StatusCode::CREATED,
        headers,
        serde_json::json!({
            "FunctionName": "test-function",
            "FunctionArn": "arn:aws:lambda:us-east-1:000000000000:function:test-function",
            "Runtime": "provided.al2",
            "Version": "1",
            "State": "Active",
            "MemorySize": 512,
            "Timeout": 30
        })
        .to_string(),
    )
}

async fn create_function_url(
    State(recorder): State<Recorder>,
    Path(name): Path<String>,
    body: Bytes,
) -> (StatusCode, HeaderMap, String) {
    record(&recorder, "POST", &format!("/url/{name}"), &body).await;
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().expect("header"));
    (
        StatusCode::CREATED,
        headers,
        serde_json::json!({
            "FunctionUrl": "https://mock-url-id.lambda-url.us-east-1.on.aws/",
            "FunctionArn": "arn:aws:lambda:us-east-1:000000000000:function:test-function",
            "AuthType": "AWS_IAM",
            "CreationTime": "2024-01-01T00:00:00.000+0000"
        })
        .to_string(),
    )
}

async fn invoke_function(
    State(recorder): State<Recorder>,
    Path(name): Path<String>,
    body: Bytes,
) -> (StatusCode, HeaderMap, String) {
    record(&recorder, "POST", &format!("/invoke/{name}"), &body).await;
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().expect("header"));
    // A genuinely different payload from the request, so an echo cannot pass.
    (
        StatusCode::OK,
        headers,
        serde_json::json!({"statusCode": 200, "body": "{\"generated\":\"from-lambda\"}"})
            .to_string(),
    )
}

async fn invoke_function_error(
    Path(_name): Path<String>,
    _body: Bytes,
) -> (StatusCode, HeaderMap, String) {
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().expect("header"));
    headers.insert("X-Amz-Function-Error", "Unhandled".parse().expect("header"));
    (
        StatusCode::OK,
        headers,
        serde_json::json!({"errorMessage": "boom", "errorType": "RuntimeError"}).to_string(),
    )
}

async fn update_code(
    State(recorder): State<Recorder>,
    Path(name): Path<String>,
    body: Bytes,
) -> (StatusCode, HeaderMap, String) {
    record(&recorder, "PUT", &format!("/code/{name}"), &body).await;
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().expect("header"));
    (
        StatusCode::OK,
        headers,
        serde_json::json!({
            "FunctionName": name,
            "FunctionArn": format!("arn:aws:lambda:us-east-1:000000000000:function:{name}"),
            "Version": "2"
        })
        .to_string(),
    )
}

async fn update_configuration(
    State(recorder): State<Recorder>,
    Path(name): Path<String>,
    body: Bytes,
) -> (StatusCode, HeaderMap, String) {
    record(&recorder, "PUT", &format!("/configuration/{name}"), &body).await;
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().expect("header"));
    (
        StatusCode::OK,
        headers,
        serde_json::json!({"FunctionName": name, "MemorySize": 512}).to_string(),
    )
}

async fn delete_function(State(recorder): State<Recorder>, Path(name): Path<String>) -> StatusCode {
    record(
        &recorder,
        "DELETE",
        &format!("/delete/{name}"),
        &Bytes::new(),
    )
    .await;
    StatusCode::NO_CONTENT
}

async fn put_concurrency(
    State(recorder): State<Recorder>,
    Path(name): Path<String>,
    body: Bytes,
) -> (StatusCode, HeaderMap, String) {
    record(&recorder, "PUT", &format!("/concurrency/{name}"), &body).await;
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().expect("header"));
    (
        StatusCode::OK,
        headers,
        serde_json::json!({"ReservedConcurrentExecutions": 10}).to_string(),
    )
}

fn lambda_router(recorder: Recorder) -> Router {
    Router::new()
        .route("/2015-03-31/functions", post(create_function))
        .route(
            "/2015-03-31/functions/{name}/invocations",
            post(invoke_function),
        )
        .route("/2015-03-31/functions/{name}/code", put(update_code))
        .route(
            "/2015-03-31/functions/{name}/configuration",
            put(update_configuration),
        )
        .route("/2015-03-31/functions/{name}", delete(delete_function))
        .route(
            "/2021-10-31/functions/{name}/url",
            post(create_function_url),
        )
        .route(
            "/2017-10-31/functions/{name}/concurrency",
            put(put_concurrency),
        )
        .with_state(recorder)
}

// ── CloudWatch mock (Smithy RPC v2 CBOR) ─────────────────────────────────────
//
// `aws-sdk-cloudwatch` 1.123 speaks `smithy-protocol: rpc-v2-cbor`: it POSTs a
// CBOR body to
// `/service/GraniteServiceVersion20100801/operation/GetMetricStatistics` and
// expects a CBOR `GetMetricStatisticsOutput` back. The tiny encoder below emits
// exactly that, so the provider's real deserializer is exercised.

/// Write a CBOR header for `major` type with argument `value`.
fn cbor_header(out: &mut Vec<u8>, major: u8, value: u64) {
    let major = major << 5;
    if value < 24 {
        out.push(major | value as u8);
    } else if value < 0x100 {
        out.push(major | 24);
        out.push(value as u8);
    } else if value < 0x1_0000 {
        out.push(major | 25);
        out.extend_from_slice(&(value as u16).to_be_bytes());
    } else {
        out.push(major | 26);
        out.extend_from_slice(&(value as u32).to_be_bytes());
    }
}

fn cbor_text(out: &mut Vec<u8>, text: &str) {
    cbor_header(out, 3, text.len() as u64);
    out.extend_from_slice(text.as_bytes());
}

fn cbor_f64(out: &mut Vec<u8>, value: f64) {
    out.push(0xFB);
    out.extend_from_slice(&value.to_be_bytes());
}

/// A `Datapoint` carrying one named statistic.
fn cbor_datapoint(statistic: &str, value: f64) -> Vec<u8> {
    let mut out = Vec::new();
    cbor_header(&mut out, 5, 2); // map(2)
    cbor_text(&mut out, statistic);
    cbor_f64(&mut out, value);
    cbor_text(&mut out, "Unit");
    cbor_text(&mut out, "Count");
    out
}

/// A `Datapoint` carrying an `ExtendedStatistics` percentile map.
fn cbor_percentile_datapoint(percentile: &str, value: f64) -> Vec<u8> {
    let mut out = Vec::new();
    cbor_header(&mut out, 5, 1); // map(1)
    cbor_text(&mut out, "ExtendedStatistics");
    cbor_header(&mut out, 5, 1); // map(1)
    cbor_text(&mut out, percentile);
    cbor_f64(&mut out, value);
    out
}

/// `GetMetricStatisticsOutput { Label, Datapoints }`.
fn cbor_metric_response(datapoints: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out = Vec::new();
    cbor_header(&mut out, 5, 2); // map(2)
    cbor_text(&mut out, "Label");
    cbor_text(&mut out, "mock");
    cbor_text(&mut out, "Datapoints");
    cbor_header(&mut out, 4, datapoints.len() as u64); // array(n)
    for datapoint in datapoints {
        out.extend_from_slice(&datapoint);
    }
    out
}

async fn get_metric_statistics(
    State(recorder): State<Recorder>,
    body: Bytes,
) -> (StatusCode, HeaderMap, Vec<u8>) {
    // CBOR text strings embed their content as plain ASCII, so the requested
    // metric name is findable without decoding the whole body.
    let raw = String::from_utf8_lossy(&body).to_string();
    record(&recorder, "POST", "/cloudwatch", &body).await;

    // Longest names first: "Invocations" is a suffix of
    // "ProvisionedConcurrencyInvocations".
    let datapoints = if raw.contains("ProvisionedConcurrencyInvocations")
        || raw.contains("ProvisionedConcurrencySpilloverInvocations")
        || raw.contains("DeadLetterErrors")
        || raw.contains("IteratorAge")
    {
        // CloudWatch published nothing for these in the window.
        Vec::new()
    } else if raw.contains("Invocations") {
        vec![cbor_datapoint("Sum", 120.0)]
    } else if raw.contains("Errors") {
        vec![cbor_datapoint("Sum", 3.0)]
    } else if raw.contains("Throttles") {
        vec![cbor_datapoint("Sum", 1.0)]
    } else if raw.contains("ConcurrentExecutions") {
        vec![cbor_datapoint("Maximum", 7.0)]
    } else if raw.contains("p95") {
        vec![cbor_percentile_datapoint("p95", 480.0)]
    } else if raw.contains("p50") || raw.contains("p99") {
        // Deliberately unmeasured, so the provider must flag them.
        Vec::new()
    } else if raw.contains("Duration") {
        vec![cbor_datapoint("Average", 210.5)]
    } else {
        Vec::new()
    };

    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/cbor".parse().expect("header"));
    headers.insert("smithy-protocol", "rpc-v2-cbor".parse().expect("header"));
    (StatusCode::OK, headers, cbor_metric_response(datapoints))
}

fn cloudwatch_router(recorder: Recorder) -> Router {
    // RPC v2 CBOR uses an operation-specific path; `fallback` accepts whatever
    // the SDK version in use produces.
    Router::new().fallback(post(get_metric_statistics)).with_state(recorder)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_deploy_calls_create_function_with_the_real_request_shape() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let endpoint = spawn(lambda_router(Arc::clone(&recorder))).await;
    let provider =
        AwsLambdaProvider::with_clients(lambda_client(&endpoint), None, "us-east-1".to_string())
            .with_execution_role("arn:aws:iam::000000000000:role/lambda-exec");

    let result = provider
        .deploy(&config(
            "test-function",
            PackageType::Zip,
            "s3://my-bucket/artifacts/function.zip",
        ))
        .await
        .expect("deploy against the mock endpoint must succeed");

    assert_eq!(
        result.function_arn.as_deref(),
        Some("arn:aws:lambda:us-east-1:000000000000:function:test-function"),
        "the ARN must come from the CreateFunction response, not be synthesized"
    );
    assert_eq!(
        result.function_url.as_deref(),
        Some("https://mock-url-id.lambda-url.us-east-1.on.aws/")
    );

    let requests = recorder.lock().await.clone();
    let create = requests
        .iter()
        .find(|r| r.path == "/2015-03-31/functions")
        .expect("CreateFunction must have been called");
    assert_eq!(create.method, "POST");
    let payload: serde_json::Value =
        serde_json::from_str(&create.body).expect("CreateFunction body must be JSON");
    assert_eq!(payload["FunctionName"], serde_json::json!("test-function"));
    assert_eq!(
        payload["Role"],
        serde_json::json!("arn:aws:iam::000000000000:role/lambda-exec")
    );
    assert_eq!(payload["MemorySize"], serde_json::json!(512));
    assert_eq!(payload["Timeout"], serde_json::json!(30));
    assert_eq!(payload["Handler"], serde_json::json!("main"));
    assert_eq!(payload["Code"]["S3Bucket"], serde_json::json!("my-bucket"));
    assert_eq!(
        payload["Code"]["S3Key"],
        serde_json::json!("artifacts/function.zip")
    );
    assert_eq!(
        payload["Environment"]["Variables"]["LOG_LEVEL"],
        serde_json::json!("INFO")
    );
    assert_eq!(
        payload["TracingConfig"]["Mode"],
        serde_json::json!("Active")
    );

    assert!(
        requests.iter().any(|r| r.path == "/url/test-function"),
        "an HTTP trigger must create a function URL"
    );
}

#[tokio::test]
async fn test_deploy_without_http_trigger_creates_no_function_url() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let endpoint = spawn(lambda_router(Arc::clone(&recorder))).await;
    let provider =
        AwsLambdaProvider::with_clients(lambda_client(&endpoint), None, "us-east-1".to_string())
            .with_execution_role("arn:aws:iam::000000000000:role/lambda-exec");

    let mut cfg = config("test-function", PackageType::Zip, "s3://b/k.zip");
    cfg.triggers = vec![Trigger {
        trigger_type: TriggerType::SQS,
        source_arn: None,
        event_source_mapping: None,
    }];

    let result = provider.deploy(&cfg).await.expect("deploy");
    assert!(
        result.function_url.is_none(),
        "no URL may be reported when no HTTP trigger was requested"
    );
    let requests = recorder.lock().await.clone();
    assert!(!requests.iter().any(|r| r.path.starts_with("/url/")));
}

#[tokio::test]
async fn test_deploy_without_execution_role_is_rejected_before_any_call() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let endpoint = spawn(lambda_router(Arc::clone(&recorder))).await;
    let provider =
        AwsLambdaProvider::with_clients(lambda_client(&endpoint), None, "us-east-1".to_string());

    let err = provider
        .deploy(&config("test-function", PackageType::Zip, "s3://b/k.zip"))
        .await
        .expect_err("CreateFunction needs an IAM role");
    assert!(matches!(
        err.downcast_ref::<ServerlessError>(),
        Some(ServerlessError::MissingConfiguration { .. })
    ));
    assert!(
        recorder.lock().await.is_empty(),
        "no AWS call may be made when configuration is incomplete"
    );
}

#[tokio::test]
async fn test_invoke_returns_the_lambda_response_not_the_request() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let endpoint = spawn(lambda_router(Arc::clone(&recorder))).await;
    let provider =
        AwsLambdaProvider::with_clients(lambda_client(&endpoint), None, "us-east-1".to_string());

    let request_payload = serde_json::json!({"prompt": "unique-request-marker"});
    let response = provider
        .invoke(
            &config("test-function", PackageType::Zip, "s3://b/k.zip"),
            request_payload.clone(),
        )
        .await
        .expect("invoke");

    assert_ne!(
        response, request_payload,
        "the provider must not echo the request payload"
    );
    assert_eq!(response["statusCode"], serde_json::json!(200));
    assert!(
        response["body"].as_str().unwrap_or_default().contains("from-lambda"),
        "response must come from the Lambda payload: {response}"
    );

    let requests = recorder.lock().await.clone();
    let invoke = requests
        .iter()
        .find(|r| r.path == "/invoke/test-function")
        .expect("Invoke must have been called");
    assert!(
        invoke.body.contains("unique-request-marker"),
        "the request payload must be sent to Lambda: {}",
        invoke.body
    );
}

#[tokio::test]
async fn test_invoke_surfaces_a_function_error() {
    let router = Router::new().route(
        "/2015-03-31/functions/{name}/invocations",
        post(invoke_function_error),
    );
    let endpoint = spawn(router).await;
    let provider =
        AwsLambdaProvider::with_clients(lambda_client(&endpoint), None, "us-east-1".to_string());

    let err = provider
        .invoke(
            &config("test-function", PackageType::Zip, "s3://b/k.zip"),
            serde_json::json!({}),
        )
        .await
        .expect_err("a Lambda function error must not be reported as success");
    let message = err.to_string();
    assert!(message.contains("Unhandled"), "{message}");
}

#[tokio::test]
async fn test_update_calls_both_code_and_configuration() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let endpoint = spawn(lambda_router(Arc::clone(&recorder))).await;
    let provider =
        AwsLambdaProvider::with_clients(lambda_client(&endpoint), None, "us-east-1".to_string());

    let result = provider
        .update(&config(
            "test-function",
            PackageType::Zip,
            "s3://my-bucket/v2.zip",
        ))
        .await
        .expect("update");
    assert_eq!(
        result.function_arn.as_deref(),
        Some("arn:aws:lambda:us-east-1:000000000000:function:test-function")
    );

    let requests = recorder.lock().await.clone();
    let code = requests
        .iter()
        .find(|r| r.path == "/code/test-function")
        .expect("UpdateFunctionCode must be called");
    let code_payload: serde_json::Value = serde_json::from_str(&code.body).expect("json body");
    assert_eq!(code_payload["S3Bucket"], serde_json::json!("my-bucket"));
    assert_eq!(code_payload["S3Key"], serde_json::json!("v2.zip"));
    assert!(
        requests.iter().any(|r| r.path == "/configuration/test-function"),
        "UpdateFunctionConfiguration must be called too"
    );
}

#[tokio::test]
async fn test_delete_calls_delete_function() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let endpoint = spawn(lambda_router(Arc::clone(&recorder))).await;
    let provider =
        AwsLambdaProvider::with_clients(lambda_client(&endpoint), None, "us-east-1".to_string());

    provider
        .delete(&config("test-function", PackageType::Zip, "s3://b/k.zip"))
        .await
        .expect("delete");
    let requests = recorder.lock().await.clone();
    assert!(requests
        .iter()
        .any(|r| r.path == "/delete/test-function" && r.method == "DELETE"));
}

#[tokio::test]
async fn test_configure_auto_scaling_puts_reserved_concurrency() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let endpoint = spawn(lambda_router(Arc::clone(&recorder))).await;
    let provider =
        AwsLambdaProvider::with_clients(lambda_client(&endpoint), None, "us-east-1".to_string());

    let cfg = config("test-function", PackageType::Zip, "s3://b/k.zip");
    provider.configure_auto_scaling(&cfg, &cfg.scaling).await.expect("auto scaling");

    let requests = recorder.lock().await.clone();
    let put = requests
        .iter()
        .find(|r| r.path == "/concurrency/test-function")
        .expect("PutFunctionConcurrency must be called");
    let payload: serde_json::Value = serde_json::from_str(&put.body).expect("json");
    assert_eq!(
        payload["ReservedConcurrentExecutions"],
        serde_json::json!(10)
    );
}

#[tokio::test]
async fn test_get_metrics_parses_real_cloudwatch_datapoints() {
    let lambda_recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let cw_recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let lambda_endpoint = spawn(lambda_router(Arc::clone(&lambda_recorder))).await;
    let cw_endpoint = spawn(cloudwatch_router(Arc::clone(&cw_recorder))).await;

    let provider = AwsLambdaProvider::with_clients(
        lambda_client(&lambda_endpoint),
        Some(cloudwatch_client(&cw_endpoint)),
        "us-east-1".to_string(),
    );

    let metrics = provider
        .get_metrics(&config("test-function", PackageType::Zip, "s3://b/k.zip"))
        .await
        .expect("metrics");

    // Values must come from the mock CloudWatch response, not the old constants
    // (1000 invocations, 5 errors, 250 ms, 99.5 % success).
    assert_eq!(metrics.invocations, 120);
    assert_eq!(metrics.errors, 3);
    assert_eq!(metrics.throttles, 1);
    assert_eq!(metrics.concurrent_executions, 7);
    assert!((metrics.duration_ms - 210.5).abs() < 1e-6);
    assert!((metrics.success_rate - 97.5).abs() < 1e-6);

    // The extended-statistics (percentile) parsing path: the mock publishes a
    // p95 datapoint and none for p50/p99, so exactly one percentile is measured.
    assert!((metrics.p95_duration_ms - 480.0).abs() < 1e-6);
    assert!(metrics.is_measured("p95_duration_ms"));
    assert!(!metrics.is_measured("p50_duration_ms"));
    assert!(!metrics.is_measured("p99_duration_ms"));

    // Fields CloudWatch metrics cannot supply are flagged, never claimed.
    assert!(!metrics.is_measured("cost_usd"));
    assert!(!metrics.is_measured("cold_starts"));
    assert!(!metrics.is_measured("memory_utilization"));
    assert_eq!(metrics.cost_usd, 0.0);
    assert_eq!(metrics.cold_starts, 0);
    assert!(metrics.is_measured("invocations"));

    let requests = cw_recorder.lock().await.clone();
    assert!(
        requests.iter().any(|r| r.body.contains("Invocations")),
        "the Invocations metric must actually be queried"
    );
    assert!(
        requests.iter().any(|r| r.body.contains("AWS/Lambda")),
        "the AWS/Lambda namespace must be queried"
    );
    assert!(
        requests.iter().any(|r| r.body.contains("test-function")),
        "the FunctionName dimension must be sent"
    );
}

#[tokio::test]
async fn test_get_metrics_without_cloudwatch_client_fails_before_any_call() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let endpoint = spawn(lambda_router(Arc::clone(&recorder))).await;
    let provider =
        AwsLambdaProvider::with_clients(lambda_client(&endpoint), None, "us-east-1".to_string());

    let err = provider
        .get_metrics(&config("test-function", PackageType::Zip, "s3://b/k.zip"))
        .await
        .expect_err("no CloudWatch client means no metrics");
    assert!(matches!(
        err.downcast_ref::<ServerlessError>(),
        Some(ServerlessError::MissingCredentials { .. })
    ));
    assert!(recorder.lock().await.is_empty());
}

#[tokio::test]
async fn test_deploy_with_local_zip_uploads_the_bytes() {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let endpoint = spawn(lambda_router(Arc::clone(&recorder))).await;
    let provider =
        AwsLambdaProvider::with_clients(lambda_client(&endpoint), None, "us-east-1".to_string())
            .with_execution_role("arn:aws:iam::000000000000:role/lambda-exec");

    let path =
        std::env::temp_dir().join(format!("trustformers-lambda-{}.zip", uuid::Uuid::new_v4()));
    tokio::fs::write(&path, b"PK\x03\x04 mock zip payload")
        .await
        .expect("write zip");

    provider
        .deploy(&config(
            "test-function",
            PackageType::Zip,
            &path.to_string_lossy(),
        ))
        .await
        .expect("deploy");

    let requests = recorder.lock().await.clone();
    let create = requests
        .iter()
        .find(|r| r.path == "/2015-03-31/functions")
        .expect("CreateFunction");
    let payload: serde_json::Value = serde_json::from_str(&create.body).expect("json");
    assert!(
        payload["Code"]["ZipFile"].is_string(),
        "a local zip must be uploaded inline: {payload}"
    );

    let _ = tokio::fs::remove_file(&path).await;
}

#[tokio::test]
async fn test_deploy_with_unreadable_package_reports_configuration_error() {
    let endpoint = spawn(lambda_router(Arc::new(Mutex::new(Vec::new())))).await;
    let provider =
        AwsLambdaProvider::with_clients(lambda_client(&endpoint), None, "us-east-1".to_string())
            .with_execution_role("arn:aws:iam::000000000000:role/lambda-exec");

    let missing = std::env::temp_dir().join(format!("absent-{}.zip", uuid::Uuid::new_v4()));
    let err = provider
        .deploy(&config(
            "test-function",
            PackageType::Zip,
            &missing.to_string_lossy(),
        ))
        .await
        .expect_err("an unreadable deployment package must fail");
    assert!(matches!(
        err.downcast_ref::<ServerlessError>(),
        Some(ServerlessError::MissingConfiguration { .. })
    ));
}
