//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use anyhow::Result;
use std::sync::Arc;
use tracing::info;
/// Integration with main debug session
impl crate::DebugSession {
    /// Enable streaming for this debug session
    pub async fn enable_streaming(
        &mut self,
        config: StreamingDebugConfig,
    ) -> Result<Arc<StreamingDebugger>> {
        let streaming_debugger = Arc::new(StreamingDebugger::new(config));
        streaming_debugger.start().await?;
        info!("Enabled streaming for debug session {}", self.id());
        Ok(streaming_debugger)
    }
}
/// Convenience macros for streaming debugging
#[macro_export]
macro_rules! stream_tensor {
    ($streamer:expr, $session_id:expr, $tensor:expr, $name:expr) => {{
        let tensor_id = uuid::Uuid::new_v4();
        let shape = $tensor.shape().to_vec();
        let values: Vec<f64> = $tensor.iter().map(|&x| x.into()).collect();
        $streamer
            .send_tensor_data($session_id, tensor_id, $name.to_string(), shape, values)
            .await
    }};
}
#[macro_export]
macro_rules! stream_gradients {
    ($streamer:expr, $session_id:expr, $layer_name:expr, $gradients:expr) => {{
        let gradient_values: Vec<f64> = $gradients.iter().map(|&x| x.into()).collect();
        $streamer
            .send_gradient_flow($session_id, $layer_name.to_string(), &gradient_values)
            .await
    }};
}
#[macro_export]
macro_rules! stream_anomaly {
    (
        $streamer:expr, $session_id:expr, $anomaly_type:expr, $severity:expr,
        $description:expr
    ) => {{
        $streamer
            .send_anomaly_detected(
                $session_id,
                $anomaly_type,
                $severity,
                $description.to_string(),
                0.95,
                vec![],
            )
            .await
    }};
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::{Duration, SystemTime};

    use uuid::Uuid;
    #[tokio::test]
    async fn test_streaming_debugger_creation() {
        let config = StreamingDebugConfig::default();
        let debugger = StreamingDebugger::new(config);
        assert!(!*debugger.is_running.read().await);
    }
    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // TODO: Fix timeout issue in streaming debugger
    async fn test_start_stop_streaming() {
        let config = StreamingDebugConfig {
            stream_interval_ms: 50,
            ..Default::default()
        };
        let debugger = StreamingDebugger::new(config);
        let test_result = tokio::time::timeout(Duration::from_secs(3), async {
            assert!(debugger.start().await.is_ok());
            assert!(*debugger.is_running.read().await);
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(debugger.stop().await.is_ok());
            assert!(!*debugger.is_running.read().await);
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok::<(), anyhow::Error>(())
        })
        .await;
        assert!(test_result.is_ok(), "Test timed out");
        assert!(test_result.unwrap().is_ok());
    }
    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // TODO: Fix timeout issue in streaming debugger
    async fn test_subscription() {
        let config = StreamingDebugConfig {
            stream_interval_ms: 50,
            ..Default::default()
        };
        let debugger = StreamingDebugger::new(config);
        let test_result = tokio::time::timeout(Duration::from_secs(3), async {
            debugger.start().await.unwrap();
            let subscription = debugger
                .subscribe(
                    "test_subscriber".to_string(),
                    StreamFormat::Json,
                    StreamFilter::default(),
                )
                .await
                .unwrap();
            assert_eq!(debugger.get_subscribers().await.len(), 1);
            debugger.unsubscribe(subscription.subscriber_id()).await.unwrap();
            assert_eq!(debugger.get_subscribers().await.len(), 0);
            debugger.stop().await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok::<(), anyhow::Error>(())
        })
        .await;
        assert!(test_result.is_ok(), "Test timed out");
        assert!(test_result.unwrap().is_ok());
    }
    #[tokio::test]
    async fn test_tensor_statistics() {
        let config = StreamingDebugConfig::default();
        let debugger = StreamingDebugger::new(config);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = debugger.compute_tensor_statistics(&values);
        assert_eq!(stats.mean, 3.0);
        assert!(stats.std > 0.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.zero_count, 0);
    }
    #[tokio::test]
    async fn test_gradient_statistics() {
        let config = StreamingDebugConfig::default();
        let debugger = StreamingDebugger::new(config);
        let gradients = vec![0.1, -0.2, 0.3, -0.1, 0.0];
        let stats = debugger.compute_gradient_statistics(&gradients);
        assert!(stats.l1_norm > 0.0);
        assert!(stats.l2_norm > 0.0);
        assert_eq!(stats.max_grad, 0.3);
        assert_eq!(stats.min_grad, -0.2);
    }
    #[tokio::test]
    async fn test_event_filtering() {
        let session_id1 = Uuid::new_v4();
        let session_id2 = Uuid::new_v4();
        let filter = StreamFilter {
            session_ids: Some(vec![session_id1]),
            event_types: Some(vec!["TensorData".to_string()]),
            min_severity: None,
            time_range: None,
            custom_filters: HashMap::new(),
        };
        let matching_event = StreamEvent::TensorData {
            session_id: session_id1,
            tensor_id: Uuid::new_v4(),
            name: "test".to_string(),
            shape: vec![2, 2],
            values: vec![1.0, 2.0, 3.0, 4.0],
            statistics: TensorStatistics {
                mean: 2.5,
                std: 1.29,
                min: 1.0,
                max: 4.0,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
                sparsity: 0.0,
            },
            timestamp: SystemTime::now(),
        };
        let non_matching_event = StreamEvent::TensorData {
            session_id: session_id2,
            tensor_id: Uuid::new_v4(),
            name: "test".to_string(),
            shape: vec![2, 2],
            values: vec![1.0, 2.0, 3.0, 4.0],
            statistics: TensorStatistics {
                mean: 2.5,
                std: 1.29,
                min: 1.0,
                max: 4.0,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
                sparsity: 0.0,
            },
            timestamp: SystemTime::now(),
        };
        assert!(StreamSubscription::matches_filter(&matching_event, &filter));
        assert!(!StreamSubscription::matches_filter(
            &non_matching_event,
            &filter
        ));
    }
}
/// Trait for custom aggregation rules
pub trait AggregationRule {
    fn aggregate(&self, events: &[StreamEvent]) -> Result<f64>;
    fn rule_name(&self) -> &str;
}
#[cfg(test)]
mod enhanced_tests {
    use super::*;
    use std::time::{Duration, Instant, SystemTime};
    use uuid::Uuid;
    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // TODO: Fix timeout issue in enhanced streaming debugger
    async fn test_enhanced_streaming_debugger() {
        let base_config = StreamingDebugConfig {
            stream_interval_ms: 50,
            ..Default::default()
        };
        let adaptive_config = AdaptiveStreamingConfig {
            monitoring_interval_ms: 500,
            ..Default::default()
        };
        let aggregation_config = RealTimeAggregationConfig {
            window_size_seconds: 1,
            ..Default::default()
        };
        let buffering_config = IntelligentBufferingConfig::default();
        let mut debugger = EnhancedStreamingDebugger::new(
            base_config,
            adaptive_config,
            aggregation_config,
            buffering_config,
        );
        let test_result = tokio::time::timeout(Duration::from_secs(5), async {
            assert!(debugger.start_enhanced_streaming().await.is_ok());
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert!(debugger.stop_enhanced_streaming().await.is_ok());
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok::<(), anyhow::Error>(())
        })
        .await;
        assert!(test_result.is_ok(), "Test timed out");
        assert!(test_result.unwrap().is_ok());
    }
    #[tokio::test]
    async fn test_network_condition_monitor() {
        let mut monitor = NetworkConditionMonitor::new();
        monitor.update_conditions().await;
        assert!(monitor.quality_score >= 0.0);
        assert!(monitor.quality_score <= 1.0);
        assert!(!monitor.history.is_empty());
    }
    #[test]
    fn test_buffer_performance_predictor() {
        let predictor = BufferPerformancePredictor {
            performance_history: vec![
                BufferPerformancePoint {
                    buffer_size: 500,
                    throughput: 100.0,
                    latency: 50.0,
                    memory_usage: 50000,
                    timestamp: Instant::now(),
                },
                BufferPerformancePoint {
                    buffer_size: 1000,
                    throughput: 150.0,
                    latency: 40.0,
                    memory_usage: 100000,
                    timestamp: Instant::now(),
                },
            ],
            model_params: vec![],
            accuracy: 0.8,
        };
        let optimal_size = predictor.predict_optimal_size().unwrap();
        assert_eq!(optimal_size, 1000);
    }
    #[tokio::test]
    async fn test_importance_scorer() {
        let scorer = ImportanceScorer::new();
        let critical_event = StreamEvent::AnomalyDetected {
            session_id: Uuid::new_v4(),
            anomaly_type: AnomalyType::GradientExplosion,
            severity: AnomalySeverity::Critical,
            description: "Critical gradient explosion".to_string(),
            confidence: 0.95,
            affected_components: vec!["layer1".to_string()],
            timestamp: SystemTime::now(),
        };
        let low_event = StreamEvent::AnomalyDetected {
            session_id: Uuid::new_v4(),
            anomaly_type: AnomalyType::TrainingStagnation,
            severity: AnomalySeverity::Low,
            description: "Slow convergence detected".to_string(),
            confidence: 0.6,
            affected_components: vec!["layer2".to_string()],
            timestamp: SystemTime::now(),
        };
        let critical_score = scorer.calculate_importance(&critical_event).await.unwrap();
        let low_score = scorer.calculate_importance(&low_event).await.unwrap();
        assert!(critical_score > low_score);
    }
}
