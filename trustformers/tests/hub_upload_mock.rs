//! Integration tests asserting `HubUploader`'s real HTTP request shape
//! against a local mock server (never the real Hugging Face Hub).
//!
//! Requires the `hub` feature (real `reqwest` networking) — the whole file
//! is a no-op without it. Run with:
//! `cargo nextest run -p trustformers --features hub --test hub_upload_mock`

#![cfg(feature = "hub")]

use axum::body::Bytes;
use axum::extract::{Request, State};
use axum::http::{header, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Router;
use std::sync::{Arc, Mutex};
use trustformers::hub_upload::{HubUploader, RepoType, UploadConfig, UploadFile};

/// One request the mock server observed, captured verbatim for assertions.
#[derive(Debug, Clone)]
struct CapturedRequest {
    method: Method,
    path: String,
    auth_header: Option<String>,
    content_type: Option<String>,
    body: Vec<u8>,
}

#[derive(Default)]
struct MockState {
    requests: Mutex<Vec<CapturedRequest>>,
    /// HTTP status the mock returns for a repo-existence `GET`, configurable
    /// per test (default 0 means "use the test's explicit setting").
    repo_exists_status: Mutex<u16>,
}

impl MockState {
    fn requests(&self) -> Vec<CapturedRequest> {
        self.requests.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn find(&self, method: Method, path_contains: &str) -> Option<CapturedRequest> {
        self.requests()
            .into_iter()
            .find(|r| r.method == method && r.path.contains(path_contains))
    }
}

/// Start the mock server on a background thread with its own Tokio runtime,
/// and return its base URL (`http://127.0.0.1:PORT`).
///
/// Deliberately *not* an `async fn`/`#[tokio::test]`: `HubUploader`'s public
/// API is synchronous and runs its own single-threaded runtime internally
/// (mirroring `hub.rs`'s `download_file`), which panics if called from a
/// thread that already has an ambient Tokio runtime. Running the mock server
/// on a separate thread, and calling `HubUploader` from the test's own plain
/// (non-async) thread, keeps the two runtimes fully independent.
fn start_mock_server(state: Arc<MockState>) -> String {
    let (addr_tx, addr_rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build mock server runtime");
        rt.block_on(async move {
            let app: Router = Router::new().fallback(handle_any).with_state(state);
            let listener =
                tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind mock server");
            let addr = listener.local_addr().expect("mock server local_addr");
            addr_tx.send(addr).expect("send mock server addr");
            axum::serve(listener, app).await.expect("serve mock server");
        });
    });

    let addr = addr_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("mock server start");
    format!("http://{addr}")
}

/// A single fallback handler for every route, dispatching on method + raw
/// path rather than axum's path-parameter routing — simplest way to handle
/// repo ids containing a `/` (`owner/name`) without fighting axum's
/// catch-all-must-be-last-segment rule.
async fn handle_any(State(state): State<Arc<MockState>>, req: Request) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let auth_header = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    let body: Bytes = axum::body::to_bytes(req.into_body(), 64 * 1024 * 1024)
        .await
        .unwrap_or_default();

    state.requests.lock().unwrap_or_else(|e| e.into_inner()).push(CapturedRequest {
        method: method.clone(),
        path: path.clone(),
        auth_header,
        content_type,
        body: body.to_vec(),
    });

    if method == Method::POST && path == "/api/repos/create" {
        return axum::Json(serde_json::json!({
            "url": format!("http://mock-hub.test/{}", "testuser/test-model")
        }))
        .into_response();
    }

    if method == Method::GET && path.starts_with("/api/models/") {
        let status = *state.repo_exists_status.lock().unwrap_or_else(|e| e.into_inner());
        let status = StatusCode::from_u16(status).unwrap_or(StatusCode::NOT_FOUND);
        return status.into_response();
    }

    if method == Method::POST && path.contains("/commit/") {
        return axum::Json(serde_json::json!({
            "commitUrl": "http://mock-hub.test/testuser/test-model/commit/abcdef0123456789",
            "commitOid": "abcdef0123456789"
        }))
        .into_response();
    }

    (
        StatusCode::NOT_FOUND,
        format!("unhandled mock route: {method} {path}"),
    )
        .into_response()
}

fn base_config(base_url: String) -> UploadConfig {
    UploadConfig {
        token: "hf_test_token_123".to_string(),
        repo_id: "testuser/test-model".to_string(),
        repo_type: RepoType::Model,
        revision: "main".to_string(),
        commit_message: "Integration test commit".to_string(),
        create_if_missing: false,
        private: false,
        base_url,
        dry_run: false,
    }
}

fn write_temp_file(name: &str, content: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("trustformers_hub_upload_mock_test");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(name);
    std::fs::write(&path, content).expect("write temp file");
    path
}

#[test]
fn test_create_repo_sends_real_request_with_correct_shape() {
    let state = Arc::new(MockState::default());
    let base_url = start_mock_server(state.clone());

    let uploader = HubUploader::new(base_config(base_url));
    let url = uploader
        .create_repo()
        .expect("create_repo should succeed against the mock server");
    assert!(url.contains("testuser/test-model"));

    let req = state
        .find(Method::POST, "/api/repos/create")
        .expect("mock server must have received a POST to /api/repos/create");

    assert_eq!(req.auth_header.as_deref(), Some("Bearer hf_test_token_123"));

    let body: serde_json::Value = serde_json::from_slice(&req.body).expect("valid JSON body");
    assert_eq!(body["name"], "test-model");
    assert_eq!(body["organization"], "testuser");
    assert_eq!(body["type"], "model");
    assert_eq!(body["private"], false);
}

#[test]
fn test_repo_exists_true_on_http_200() {
    let state = Arc::new(MockState::default());
    *state.repo_exists_status.lock().unwrap() = 200;
    let base_url = start_mock_server(state.clone());

    let uploader = HubUploader::new(base_config(base_url));
    assert!(uploader.repo_exists().expect("repo_exists"));

    let req = state
        .find(Method::GET, "/api/models/testuser/test-model")
        .expect("mock server must have received the repo-existence GET");
    assert_eq!(req.auth_header.as_deref(), Some("Bearer hf_test_token_123"));
}

#[test]
fn test_repo_exists_false_on_http_404() {
    let state = Arc::new(MockState::default());
    *state.repo_exists_status.lock().unwrap() = 404;
    let base_url = start_mock_server(state.clone());

    let uploader = HubUploader::new(base_config(base_url));
    assert!(!uploader.repo_exists().expect("repo_exists"));
}

#[test]
fn test_upload_file_sends_ndjson_commit_with_base64_content() {
    use base64::Engine as _;

    let state = Arc::new(MockState::default());
    let base_url = start_mock_server(state.clone());
    let file_content = b"{\"hidden_size\": 768, \"model_type\": \"bert\"}";
    let path = write_temp_file("config.json", file_content);

    let uploader = HubUploader::new(base_config(base_url));
    let result = uploader
        .upload_file(&UploadFile::new(&path, "config.json"))
        .expect("upload_file should succeed against the mock server");

    assert!(!result.dry_run);
    assert_eq!(
        result.commit_url.as_deref(),
        Some("http://mock-hub.test/testuser/test-model/commit/abcdef0123456789")
    );
    assert_eq!(result.commit_oid.as_deref(), Some("abcdef0123456789"));

    let req = state
        .find(Method::POST, "/commit/main")
        .expect("mock server must have received the commit POST");
    assert_eq!(req.content_type.as_deref(), Some("application/x-ndjson"));
    assert_eq!(req.auth_header.as_deref(), Some("Bearer hf_test_token_123"));

    let body_text = String::from_utf8(req.body).expect("ndjson body must be UTF-8");
    let lines: Vec<&str> = body_text.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        lines.len(),
        2,
        "expected exactly one header line and one file line"
    );

    let header_line: serde_json::Value = serde_json::from_str(lines[0]).expect("header line json");
    assert_eq!(header_line["key"], "header");
    assert_eq!(header_line["value"]["summary"], "Integration test commit");

    let file_line: serde_json::Value = serde_json::from_str(lines[1]).expect("file line json");
    assert_eq!(file_line["key"], "file");
    assert_eq!(file_line["value"]["path"], "config.json");
    assert_eq!(file_line["value"]["encoding"], "base64");
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(file_line["value"]["content"].as_str().expect("content is a string"))
        .expect("valid base64");
    assert_eq!(decoded, file_content);
}

#[test]
fn test_delete_file_sends_deleted_file_commit_op() {
    let state = Arc::new(MockState::default());
    let base_url = start_mock_server(state.clone());

    let uploader = HubUploader::new(base_config(base_url));
    uploader.delete_file("old_weights.bin").expect("delete_file should succeed");

    let req = state.find(Method::POST, "/commit/main").expect("commit POST for delete");
    let body_text = String::from_utf8(req.body).expect("utf8 body");
    let lines: Vec<&str> = body_text.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 2);

    let delete_line: serde_json::Value = serde_json::from_str(lines[1]).expect("delete line json");
    assert_eq!(delete_line["key"], "deletedFile");
    assert_eq!(delete_line["value"]["path"], "old_weights.bin");
}

#[test]
fn test_create_if_missing_creates_repo_before_committing() {
    let state = Arc::new(MockState::default());
    *state.repo_exists_status.lock().unwrap() = 404; // repo does not exist yet
    let base_url = start_mock_server(state.clone());
    let path = write_temp_file("weights.safetensors", b"pretend tensor bytes");

    let mut config = base_config(base_url);
    config.create_if_missing = true;
    let uploader = HubUploader::new(config);

    uploader
        .upload_file(&UploadFile::new(&path, "weights.safetensors"))
        .expect("upload should create the repo then commit");

    let requests = state.requests();
    assert!(
        requests
            .iter()
            .any(|r| r.method == Method::GET && r.path.starts_with("/api/models/")),
        "expected a repo-existence check"
    );
    assert!(
        requests
            .iter()
            .any(|r| r.method == Method::POST && r.path == "/api/repos/create"),
        "expected a repo-create call since the repo did not exist"
    );
    assert!(
        requests.iter().any(|r| r.method == Method::POST && r.path.contains("/commit/")),
        "expected the commit to still happen after repo creation"
    );
}

/// Regression test: a missing token must be rejected *before* any request is
/// sent — proving `validate()`'s `MissingCredentials` check runs client-side,
/// not merely relying on the (mock, or real) server to reject an
/// unauthenticated request.
#[test]
fn test_upload_file_without_token_never_contacts_the_server() {
    let state = Arc::new(MockState::default());
    let base_url = start_mock_server(state.clone());
    let path = write_temp_file("config.json", b"{}");

    let mut config = base_config(base_url);
    config.token = String::new();
    let uploader = HubUploader::new(config);

    let result = uploader.upload_file(&UploadFile::new(&path, "config.json"));
    assert!(result.is_err(), "an empty token must be rejected");
    assert!(
        state.requests().is_empty(),
        "no request should reach the server when credentials are missing"
    );
}

/// Regression test: a file at/above the inline-upload threshold must be
/// refused locally, without ever reaching the commit endpoint.
#[test]
fn test_large_file_never_reaches_the_commit_endpoint() {
    let state = Arc::new(MockState::default());
    let base_url = start_mock_server(state.clone());
    // 10 MiB, at the threshold.
    let big_content = vec![0u8; 10 * 1024 * 1024];
    let path = write_temp_file("huge.safetensors", &big_content);

    let uploader = HubUploader::new(base_config(base_url));
    let result = uploader.upload_file(&UploadFile::new(&path, "huge.safetensors"));
    assert!(result.is_err());
    assert!(
        state.requests().is_empty(),
        "an oversized file must be refused before any network request"
    );
}
