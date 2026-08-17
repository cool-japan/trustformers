//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::{
    DeploymentResult, DetailedMetrics, ScalingConfig, ServerlessConfig, ServerlessMetrics,
};

#[async_trait::async_trait]
pub trait ServerlessProviderTrait {
    async fn deploy(&self, config: &ServerlessConfig) -> Result<DeploymentResult>;
    async fn update(&self, config: &ServerlessConfig) -> Result<DeploymentResult>;
    async fn delete(&self, config: &ServerlessConfig) -> Result<()>;
    async fn invoke(
        &self,
        config: &ServerlessConfig,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value>;
    async fn get_metrics(&self, config: &ServerlessConfig) -> Result<ServerlessMetrics>;
    async fn configure_provisioned_concurrency(
        &self,
        config: &ServerlessConfig,
        concurrency: u32,
    ) -> Result<()>;
    async fn configure_warmup_schedule(
        &self,
        config: &ServerlessConfig,
        schedule: &str,
    ) -> Result<()>;
    async fn configure_keep_warm(
        &self,
        config: &ServerlessConfig,
        requests_per_minute: u32,
    ) -> Result<()>;
    async fn get_detailed_metrics(&self, config: &ServerlessConfig) -> Result<DetailedMetrics>;
    async fn configure_auto_scaling(
        &self,
        config: &ServerlessConfig,
        scaling_config: &ScalingConfig,
    ) -> Result<()>;
}
#[cfg(test)]
pub(crate) mod tests {
    use super::super::errors::ServerlessError;
    use super::super::*;
    use std::collections::HashMap;

    /// Build a config that names a provider but supplies no credentials.
    pub(crate) fn aws_config(function_name: &str) -> ServerlessConfig {
        ServerlessConfig {
            provider: ServerlessProvider::AwsLambda,
            function_name: function_name.to_string(),
            runtime: "provided.al2".to_string(),
            memory_mb: 512,
            timeout_seconds: 30,
            environment_variables: HashMap::from([("LOG_LEVEL".to_string(), "INFO".to_string())]),
            vpc_config: None,
            deployment_package: DeploymentPackage {
                package_type: PackageType::Zip,
                source_location: "s3://bucket/function.zip".to_string(),
                handler: "main".to_string(),
                layers: vec![],
            },
            triggers: vec![Trigger {
                trigger_type: TriggerType::Http,
                source_arn: None,
                event_source_mapping: None,
            }],
            scaling: ScalingConfig {
                min_instances: 0,
                max_instances: 100,
                target_utilization: 0.7,
                scale_down_delay_seconds: 300,
                scale_up_delay_seconds: 60,
                concurrency_limit: Some(10),
            },
            monitoring: MonitoringConfig {
                enable_logging: true,
                log_level: "INFO".to_string(),
                enable_tracing: true,
                enable_metrics: true,
                custom_metrics: vec!["inference_duration".to_string()],
                enable_xray: true,
                enable_insights: true,
                log_retention_days: Some(30),
            },
            cold_start: None,
            cost_optimization: None,
            region: Some("us-east-1".to_string()),
            tags: HashMap::from([("environment".to_string(), "test".to_string())]),
        }
    }

    #[tokio::test]
    async fn test_serverless_orchestrator_creation() {
        let orchestrator = ServerlessOrchestrator::new();
        let deployments = orchestrator.list_deployments().await;
        assert!(deployments.is_empty());
    }

    #[tokio::test]
    async fn test_provider_registration() {
        let orchestrator = ServerlessOrchestrator::new();
        let provider = Box::new(AwsLambdaProvider::new("us-east-1".to_string()));
        let result = orchestrator.register_provider(ServerlessProvider::AwsLambda, provider).await;
        assert!(result.is_ok());
    }

    /// Regression test: `deploy()` used to return
    /// `arn:aws:lambda:us-east-1:123456789012:function:{name}` and a
    /// `lambda-url.us-east-1.on.aws` URL without ever calling AWS.
    #[tokio::test]
    async fn test_deployment_without_credentials_is_rejected() {
        let orchestrator = ServerlessOrchestrator::new();
        orchestrator
            .register_provider(
                ServerlessProvider::AwsLambda,
                Box::new(AwsLambdaProvider::new("us-east-1".to_string())),
            )
            .await
            .expect("provider registration");

        let err = orchestrator
            .deploy_function(aws_config("test-function"))
            .await
            .expect_err("deploying without AWS credentials must fail");
        assert!(
            matches!(
                err.downcast_ref::<ServerlessError>(),
                Some(ServerlessError::MissingCredentials { .. })
            ),
            "expected MissingCredentials, got: {err}"
        );
        assert!(
            !err.to_string().contains("123456789012"),
            "no fabricated account id may appear anywhere"
        );
    }

    /// Regression test: `invoke()` used to echo the request payload back as
    /// `{"statusCode": 200, "body": <payload>}`.
    #[tokio::test]
    async fn test_invocation_without_credentials_is_rejected() {
        let provider = AwsLambdaProvider::new("us-east-1".to_string());
        let payload = serde_json::json!({ "message": "Hello, World!" });
        let err = provider
            .invoke(&aws_config("test-function"), payload.clone())
            .await
            .expect_err("invoking without AWS credentials must fail");
        assert!(matches!(
            err.downcast_ref::<ServerlessError>(),
            Some(ServerlessError::MissingCredentials { .. })
        ));
        assert!(
            !err.to_string().contains("Hello, World!"),
            "the request payload must not be echoed back as a result"
        );
    }

    /// Regression test: `get_metrics()` used to return invocations = 1000,
    /// errors = 5, cold_starts = 50 and cost_usd = 2.50 for a function that had
    /// never been deployed.
    #[tokio::test]
    async fn test_metrics_without_credentials_is_rejected() {
        let provider = AwsLambdaProvider::new("us-east-1".to_string());
        let err = provider
            .get_metrics(&aws_config("test-function"))
            .await
            .expect_err("metrics without CloudWatch credentials must fail");
        assert!(matches!(
            err.downcast_ref::<ServerlessError>(),
            Some(ServerlessError::MissingCredentials { .. })
        ));
    }

    #[tokio::test]
    async fn test_collect_metrics_reports_failures_instead_of_swallowing_them() {
        let orchestrator = ServerlessOrchestrator::new();
        orchestrator
            .register_provider(
                ServerlessProvider::AwsLambda,
                Box::new(AwsLambdaProvider::new("us-east-1".to_string())),
            )
            .await
            .expect("provider registration");

        // No deployments yet: nothing to collect, nothing to fail.
        let report = orchestrator.collect_metrics().await.expect("collect");
        assert_eq!(report.collected, 0);
        assert_eq!(report.failed, 0);
        assert!(report.errors.is_empty());
    }

    /// Azure and GCP have no cloud client wired up at all.
    #[tokio::test]
    async fn test_other_providers_report_not_implemented() {
        let azure = AzureFunctionsProvider::new("sub".to_string(), "rg".to_string());
        let mut config = aws_config("azure-fn");
        config.provider = ServerlessProvider::AzureFunctions;
        let err = azure.deploy(&config).await.expect_err("azure deploy must fail");
        assert!(matches!(
            err.downcast_ref::<ServerlessError>(),
            Some(ServerlessError::NotImplemented { .. })
        ));
        assert!(
            !err.to_string().contains("azurewebsites.net"),
            "no fabricated endpoint may be produced"
        );

        let gcp = GoogleCloudFunctionsProvider::new("project".to_string());
        let err = gcp.get_metrics(&config).await.expect_err("gcp metrics must fail");
        assert!(matches!(
            err.downcast_ref::<ServerlessError>(),
            Some(ServerlessError::NotImplemented { .. })
        ));
    }

    #[test]
    fn test_unmeasured_metrics_snapshot_is_all_zero() {
        let metrics = ServerlessMetrics::unmeasured();
        assert_eq!(metrics.invocations, 0);
        assert_eq!(metrics.cost_usd, 0.0);
        assert!(metrics.unavailable_fields.is_empty());
        assert!(metrics.is_measured("invocations"));
    }

    #[test]
    fn test_is_measured_tracks_unavailable_fields() {
        let mut metrics = ServerlessMetrics::unmeasured();
        metrics.unavailable_fields.push("cost_usd".to_string());
        assert!(!metrics.is_measured("cost_usd"));
        assert!(metrics.is_measured("invocations"));
    }
}
