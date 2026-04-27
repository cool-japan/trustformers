#![cfg(test)]
/// Extended tests for the WASM serving backend.
use super::*;

fn handler_with_models(models: Vec<&str>) -> WasmServingHandler {
    WasmServingHandler::new(WasmServingConfig {
        allowed_models: models.into_iter().map(String::from).collect(),
        ..Default::default()
    })
}

fn make_request(id: &str, model: &str) -> WasmInferenceRequest {
    WasmInferenceRequest {
        request_id: id.to_string(),
        model_id: model.to_string(),
        task: WasmTask::TextGeneration,
        inputs: serde_json::json!({"text": "Hello"}),
        parameters: WasmInferenceParameters::default(),
    }
}

// ── 27. WasmServingConfig::default — max_batch_size is 8 ─────────────────
#[test]
fn test_wasm_serving_config_defaults() {
    let cfg = WasmServingConfig::default();
    assert_eq!(cfg.max_batch_size, 8);
    assert_eq!(cfg.max_sequence_length, 2048);
    assert!(cfg.cache_responses);
}

// ── 28. WasmInferenceParameters::default — correct defaults ──────────────
#[test]
fn test_wasm_inference_params_defaults() {
    let p = WasmInferenceParameters::default();
    assert_eq!(p.max_new_tokens, Some(128));
    assert!(p.truncation);
    assert!(!p.padding);
    assert!(!p.return_full_text);
}

// ── 29. validate_request — empty request_id returns error ─────────────────
#[test]
fn test_validate_empty_request_id_returns_error() {
    let handler = handler_with_models(vec!["default"]);
    let mut req = make_request("", "default");
    req.request_id = String::new();
    let err = handler.validate_request(&req).unwrap_err();
    assert!(matches!(err, ServeError::InvalidRequest(_)));
}

// ── 30. validate_request — disallowed model returns ModelNotAllowed ───────
#[test]
fn test_validate_disallowed_model_returns_error() {
    let handler = handler_with_models(vec!["bert"]);
    let req = make_request("req-1", "gpt-4");
    let err = handler.validate_request(&req).unwrap_err();
    assert!(matches!(err, ServeError::ModelNotAllowed(_)));
}

// ── 31. validate_request — sequence too long returns SequenceTooLong ──────
#[test]
fn test_validate_sequence_too_long_returns_error() {
    let handler = WasmServingHandler::new(WasmServingConfig {
        max_sequence_length: 512,
        allowed_models: vec!["default".to_string()],
        ..Default::default()
    });
    let mut req = make_request("req-1", "default");
    req.parameters.max_length = Some(1024); // exceeds 512
    let err = handler.validate_request(&req).unwrap_err();
    assert!(matches!(err, ServeError::SequenceTooLong { .. }));
}

// ── 32. validate_request — max_length == max_sequence_length is ok ────────
#[test]
fn test_validate_max_length_at_limit_is_ok() {
    let handler = WasmServingHandler::new(WasmServingConfig {
        max_sequence_length: 512,
        allowed_models: vec!["default".to_string()],
        ..Default::default()
    });
    let mut req = make_request("req-1", "default");
    req.parameters.max_length = Some(512);
    assert!(handler.validate_request(&req).is_ok());
}

// ── 33. handle — valid request returns response with request_id ───────────
#[test]
fn test_handle_valid_request_returns_correct_id() {
    let handler = handler_with_models(vec!["default"]);
    let req = make_request("my-req-123", "default");
    let resp = handler.handle(req);
    assert_eq!(resp.request_id, "my-req-123");
    assert!(resp.error.is_none());
}

// ── 34. handle — disallowed model sets error field ─────────────────────────
#[test]
fn test_handle_disallowed_model_sets_error() {
    let handler = handler_with_models(vec!["bert"]);
    let req = make_request("req-1", "gpt-99");
    let resp = handler.handle(req);
    assert!(resp.error.is_some());
}

// ── 35. handle — tokens_generated from max_new_tokens ────────────────────
#[test]
fn test_handle_tokens_generated_from_params() {
    let handler = handler_with_models(vec!["default"]);
    let mut req = make_request("req-1", "default");
    req.parameters.max_new_tokens = Some(42);
    let resp = handler.handle(req);
    assert_eq!(resp.tokens_generated, Some(42));
}

// ── 36. handle_batch — respects max_batch_size cap ────────────────────────
#[test]
fn test_handle_batch_capped_at_max_batch_size() {
    let handler = WasmServingHandler::new(WasmServingConfig {
        max_batch_size: 3,
        allowed_models: vec!["default".to_string()],
        ..Default::default()
    });
    let requests: Vec<WasmInferenceRequest> =
        (0u32..10).map(|i| make_request(&i.to_string(), "default")).collect();
    let responses = handler.handle_batch(requests);
    assert_eq!(
        responses.len(),
        3,
        "batch must be capped at max_batch_size=3"
    );
}

// ── 37. handle_batch — empty input returns empty output ──────────────────
#[test]
fn test_handle_batch_empty_returns_empty() {
    let handler = handler_with_models(vec!["default"]);
    let resps = handler.handle_batch(vec![]);
    assert!(resps.is_empty());
}

// ── 38. error_response — sets error field ────────────────────────────────
#[test]
fn test_error_response_sets_error_field() {
    let resp = WasmServingHandler::error_response("req-err", "INTERNAL", "something broke");
    assert!(resp.error.is_some());
    let e = resp.error.unwrap();
    assert_eq!(e.code, "INTERNAL");
    assert!(e.message.contains("something broke"));
}

// ── 39. error_response — model_id is empty ────────────────────────────────
#[test]
fn test_error_response_empty_model_id() {
    let resp = WasmServingHandler::error_response("r", "E", "m");
    assert!(resp.model_id.is_empty());
}

// ── 40. error_response — latency_ms is 0.0 ───────────────────────────────
#[test]
fn test_error_response_latency_zero() {
    let resp = WasmServingHandler::error_response("r", "E", "m");
    assert_eq!(resp.latency_ms, 0.0);
}

// ── 41. health_check — status is "ok" ────────────────────────────────────
#[test]
fn test_health_check_status_ok() {
    let handler = handler_with_models(vec!["default"]);
    let hc = handler.health_check();
    assert_eq!(hc["status"].as_str(), Some("ok"));
}

// ── 42. to_config_json — produces valid JSON ──────────────────────────────
#[test]
fn test_to_config_json_produces_valid_json() {
    let handler = handler_with_models(vec!["default"]);
    let json = handler.to_config_json().expect("to_config_json");
    let _: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
}

// ── 43. build_cors_headers — wildcard allows any origin ──────────────────
#[test]
fn test_cors_headers_wildcard_allows_any_origin() {
    let handler = WasmServingHandler::new(WasmServingConfig {
        cors_origins: vec!["*".to_string()],
        allowed_models: vec!["default".to_string()],
        ..Default::default()
    });
    let headers = handler.build_cors_headers("https://example.com");
    let acao = headers
        .iter()
        .find(|(k, _)| k == "Access-Control-Allow-Origin")
        .map(|(_, v)| v.as_str())
        .unwrap_or("");
    assert_eq!(acao, "https://example.com");
}

// ── 44. build_cors_headers — specific origin matching ────────────────────
#[test]
fn test_cors_headers_specific_origin_matching() {
    let handler = WasmServingHandler::new(WasmServingConfig {
        cors_origins: vec!["https://trusted.com".to_string()],
        allowed_models: vec!["default".to_string()],
        ..Default::default()
    });
    let headers = handler.build_cors_headers("https://trusted.com");
    let acao = headers
        .iter()
        .find(|(k, _)| k == "Access-Control-Allow-Origin")
        .map(|(_, v)| v.as_str())
        .unwrap_or("");
    assert_eq!(acao, "https://trusted.com");
}

// ── 45. build_cors_headers — exactly 4 headers ───────────────────────────
#[test]
fn test_cors_headers_count() {
    let handler = handler_with_models(vec!["default"]);
    let headers = handler.build_cors_headers("*");
    assert_eq!(headers.len(), 4);
}

// ── 46. EdgePlatform — CloudflareWorkers != Generic ───────────────────────
#[test]
fn test_edge_platform_variants_differ() {
    assert_ne!(EdgePlatform::CloudflareWorkers, EdgePlatform::Generic);
    assert_ne!(EdgePlatform::DenoDeployAPI, EdgePlatform::VercelEdge);
}

// ── 47. EdgeDeploymentConfig::to_json — produces valid JSON ──────────────
#[test]
fn test_edge_deployment_config_to_json() {
    let cfg = EdgeDeploymentConfig::new(
        EdgePlatform::CloudflareWorkers,
        WasmServingConfig::default(),
    );
    let json = cfg.to_json().expect("to_json");
    let v: serde_json::Value = serde_json::from_str(&json).expect("parse JSON");
    assert_eq!(v["platform"].as_str(), Some("cloudflare_workers"));
}

// ── 48. EdgeDeploymentConfig::platform_specific_notes — non-empty ─────────
#[test]
fn test_platform_specific_notes_non_empty() {
    let platforms = [
        EdgePlatform::CloudflareWorkers,
        EdgePlatform::DenoDeployAPI,
        EdgePlatform::VercelEdge,
        EdgePlatform::AwsLambdaEdge,
        EdgePlatform::Generic,
    ];
    for p in platforms {
        let cfg = EdgeDeploymentConfig::new(p, WasmServingConfig::default());
        assert!(
            !cfg.platform_specific_notes().is_empty(),
            "platform notes must be non-empty"
        );
    }
}

// ── 49. WasmTask variants can be cloned ───────────────────────────────────
#[test]
fn test_wasm_task_can_be_cloned() {
    let tasks = [
        WasmTask::TextGeneration,
        WasmTask::TextClassification,
        WasmTask::TokenClassification,
        WasmTask::QuestionAnswering,
        WasmTask::Summarization,
        WasmTask::FeatureExtraction,
        WasmTask::ImageClassification,
        WasmTask::AudioClassification,
    ];
    for task in &tasks {
        let _cloned = task.clone();
    }
}

// ── 50. handle — cached field is false (stub doesn't cache) ───────────────
#[test]
fn test_handle_cached_is_false() {
    let handler = handler_with_models(vec!["default"]);
    let req = make_request("req-1", "default");
    let resp = handler.handle(req);
    assert!(!resp.cached, "stub backend should not report cached=true");
}
