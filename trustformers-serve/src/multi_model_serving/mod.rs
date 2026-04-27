//! Auto-generated module structure

pub mod functions;
pub mod multimodelconfig_traits;
pub mod performancestats_traits;
pub mod resourceusage_traits;
pub mod types;

// Re-export all types
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_model_info(id: &str, status: ModelStatus) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            name: format!("{}-name", id),
            version: "1.0".to_string(),
            characteristics: vec![ModelCharacteristic::Size(ModelSize::Small)],
            capabilities: vec!["text-generation".to_string()],
            status,
            performance_stats: PerformanceStats::default(),
            resource_usage: ResourceUsage::default(),
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_multi_model_server_empty_on_creation() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);
        let models = server.get_models().await;
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_register_single_model() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);
        let result =
            server.register_model(make_model_info("model-a", ModelStatus::Available)).await;
        assert!(result.is_ok());
        let models = server.get_models().await;
        assert_eq!(models.len(), 1);
        assert!(models.contains_key("model-a"));
    }

    #[tokio::test]
    async fn test_register_multiple_models() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("m1", ModelStatus::Available))
            .await
            .unwrap_or_default();
        server
            .register_model(make_model_info("m2", ModelStatus::Available))
            .await
            .unwrap_or_default();
        server
            .register_model(make_model_info("m3", ModelStatus::Loading))
            .await
            .unwrap_or_default();
        let models = server.get_models().await;
        assert_eq!(models.len(), 3);
    }

    #[tokio::test]
    async fn test_unregister_existing_model() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("remove-me", ModelStatus::Available))
            .await
            .unwrap_or_default();
        let result = server.unregister_model("remove-me").await;
        assert!(result.is_ok());
        let models = server.get_models().await;
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_unregister_nonexistent_model_returns_error() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);
        let result = server.unregister_model("does-not-exist").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_route_request_single_model() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("route-model", ModelStatus::Available))
            .await
            .unwrap_or_default();
        let request = InferenceRequest {
            input_text: "Hello".to_string(),
            path: "/v1/infer".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        let result = server.route_request(&request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_route_request_returns_single_model_routing() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);
        server
            .register_model(make_model_info("only-model", ModelStatus::Available))
            .await
            .unwrap_or_default();
        let request = InferenceRequest {
            input_text: "test".to_string(),
            path: "/infer".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        let result = server.route_request(&request).await;
        if let Ok(RoutingResult::SingleModel { model_id }) = result {
            assert_eq!(model_id, "only-model");
        }
    }

    #[test]
    fn test_model_status_variants_debug() {
        let statuses = [
            ModelStatus::Available,
            ModelStatus::Loading,
            ModelStatus::Unavailable,
            ModelStatus::Maintenance,
            ModelStatus::Deprecated,
        ];
        for status in &statuses {
            let s = format!("{:?}", status);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_model_size_variants() {
        let sizes = [
            ModelSize::Small,
            ModelSize::Medium,
            ModelSize::Large,
            ModelSize::XLarge,
        ];
        for size in &sizes {
            let s = format!("{:?}", size);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_performance_stats_default() {
        let stats = PerformanceStats::default();
        assert_eq!(stats.request_count, 0);
        assert_eq!(stats.error_rate, 0.0);
    }

    #[test]
    fn test_resource_usage_default() {
        let usage = ResourceUsage::default();
        assert_eq!(usage.cpu_usage, 0.0);
        assert_eq!(usage.memory_usage, 0);
    }

    #[test]
    fn test_inference_request_fields() {
        let mut headers = HashMap::new();
        headers.insert("content-type".to_string(), "application/json".to_string());
        let req = InferenceRequest {
            input_text: "test input".to_string(),
            path: "/v1/models/my-model/infer".to_string(),
            headers: headers.clone(),
            user_id: Some("user-123".to_string()),
            metadata: HashMap::new(),
        };
        assert_eq!(req.input_text, "test input");
        assert_eq!(req.user_id, Some("user-123".to_string()));
        assert_eq!(req.headers.len(), 1);
    }

    #[test]
    fn test_routing_result_variants() {
        let single = RoutingResult::SingleModel {
            model_id: "m1".to_string(),
        };
        let ensemble = RoutingResult::Ensemble {
            method: EnsembleMethod::Bagging {
                models: vec!["m1".to_string(), "m2".to_string()],
                sample_size: 2.0,
            },
            models: vec!["m1".to_string(), "m2".to_string()],
        };
        match single {
            RoutingResult::SingleModel { model_id } => assert_eq!(model_id, "m1"),
            _ => panic!("unexpected variant"),
        }
        match ensemble {
            RoutingResult::Ensemble { models, .. } => assert_eq!(models.len(), 2),
            _ => panic!("unexpected variant"),
        }
    }

    #[test]
    fn test_multi_model_config_default() {
        let config = MultiModelConfig::default();
        // Just verify default can be created without panic
        let _ = format!("{:?}", config.routing.default_strategy);
    }
}
