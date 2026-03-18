//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {

    use crate::multi_model_serving::{
        ModelCharacteristic, ModelInfo, ModelSize, ModelStatus, PerformanceStats, ResourceUsage,
    };
    use crate::{InferenceRequest, MultiModelConfig, MultiModelServer, RoutingResult};
    use std::collections::HashMap;
    #[tokio::test]
    async fn test_multi_model_server_creation() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);
        let models = server.get_models().await;
        assert!(models.is_empty());
    }
    #[tokio::test]
    async fn test_model_registration() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);
        let model_info = ModelInfo {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            version: "1.0".to_string(),
            characteristics: vec![ModelCharacteristic::Size(ModelSize::Small)],
            capabilities: vec!["text-generation".to_string()],
            status: ModelStatus::Available,
            performance_stats: PerformanceStats::default(),
            resource_usage: ResourceUsage::default(),
            metadata: HashMap::new(),
        };
        let result = server.register_model(model_info).await;
        assert!(result.is_ok());
        let models = server.get_models().await;
        assert_eq!(models.len(), 1);
    }
    #[tokio::test]
    async fn test_request_routing() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);
        let model_info = ModelInfo {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            version: "1.0".to_string(),
            characteristics: vec![ModelCharacteristic::Size(ModelSize::Small)],
            capabilities: vec!["text-generation".to_string()],
            status: ModelStatus::Available,
            performance_stats: PerformanceStats::default(),
            resource_usage: ResourceUsage::default(),
            metadata: HashMap::new(),
        };
        server.register_model(model_info).await.unwrap();
        let request = InferenceRequest {
            input_text: "Hello world".to_string(),
            path: "/v1/inference".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };
        let result = server.route_request(&request).await;
        assert!(result.is_ok());
        match result.unwrap() {
            RoutingResult::SingleModel { model_id } => {
                assert_eq!(model_id, "test-model");
            },
            other => {
                assert!(
                    matches!(other, RoutingResult::SingleModel { .. }),
                    "Expected single model routing, got: {:?}",
                    other
                );
            },
        }
    }
}
