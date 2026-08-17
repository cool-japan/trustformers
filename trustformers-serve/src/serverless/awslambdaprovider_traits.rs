//! # AwsLambdaProvider - Trait Implementations
//!
//! Real AWS Lambda and CloudWatch integration for [`AwsLambdaProvider`].
//!
//! Every operation here issues an actual SDK call:
//!
//! | Operation | AWS API |
//! |---|---|
//! | [`deploy`](ServerlessProviderTrait::deploy) | `CreateFunction` (+ `CreateFunctionUrlConfig` for HTTP triggers) |
//! | [`update`](ServerlessProviderTrait::update) | `UpdateFunctionCode` + `UpdateFunctionConfiguration` |
//! | [`delete`](ServerlessProviderTrait::delete) | `DeleteFunction` |
//! | [`invoke`](ServerlessProviderTrait::invoke) | `Invoke` |
//! | [`get_metrics`](ServerlessProviderTrait::get_metrics) | `GetMetricStatistics` on the `AWS/Lambda` namespace |
//! | [`configure_provisioned_concurrency`](ServerlessProviderTrait::configure_provisioned_concurrency) | `PutProvisionedConcurrencyConfig` |
//! | [`configure_auto_scaling`](ServerlessProviderTrait::configure_auto_scaling) | `PutFunctionConcurrency` |
//!
//! Without SDK clients none of these can run, so the provider returns
//! [`ServerlessError::MissingCredentials`] instead of a synthesized ARN, an
//! echoed payload or a hardcoded metric snapshot.

use anyhow::Result;
use aws_sdk_lambda::primitives::Blob;
use aws_sdk_lambda::types::{
    Architecture as LambdaArchitecture, Environment, FunctionCode, FunctionUrlAuthType,
    PackageType as LambdaPackageType, Runtime as LambdaRuntime, TracingConfig, TracingMode,
    VpcConfig as LambdaVpcConfig,
};

use super::errors::ServerlessError;
use super::functions::ServerlessProviderTrait;
use super::types::{
    Architecture, AwsLambdaProvider, CostBreakdown, DeploymentResult, DetailedMetrics, PackageType,
    PerformanceBreakdown, ResourceUtilization, ScalingConfig, ServerlessConfig, ServerlessMetrics,
    TriggerType,
};

const PROVIDER: &str = "AwsLambdaProvider";

/// Fields of [`ServerlessMetrics`] that the `AWS/Lambda` CloudWatch **metric**
/// namespace does not publish. They are only recoverable from CloudWatch Logs
/// `REPORT` lines, so `get_metrics` reports them as unmeasured.
const UNAVAILABLE_FROM_CLOUDWATCH_METRICS: [&str; 6] = [
    "cold_starts",
    "init_duration_ms",
    "max_memory_used_mb",
    "billed_duration_ms",
    "memory_utilization",
    "cost_usd",
];

impl AwsLambdaProvider {
    fn lambda(&self, operation: &'static str) -> Result<&aws_sdk_lambda::Client> {
        self.lambda_client
            .as_ref()
            .ok_or_else(|| ServerlessError::missing_credentials(PROVIDER, operation).into())
    }

    fn cloudwatch(&self, operation: &'static str) -> Result<&aws_sdk_cloudwatch::Client> {
        self.cloudwatch_client
            .as_ref()
            .ok_or_else(|| ServerlessError::missing_credentials(PROVIDER, operation).into())
    }

    /// Translate the deployment package into the SDK's `FunctionCode`.
    ///
    /// `source_location` may be an `s3://bucket/key` URI (Zip packages), a
    /// container image URI (Image packages) or a local `.zip` path, which is
    /// read from disk and uploaded inline.
    async fn function_code(&self, config: &ServerlessConfig) -> Result<FunctionCode> {
        let location = config.deployment_package.source_location.as_str();
        match config.deployment_package.package_type {
            PackageType::Image => Ok(FunctionCode::builder().image_uri(location).build()),
            PackageType::Zip | PackageType::Source => {
                if let Some(rest) = location.strip_prefix("s3://") {
                    let (bucket, key) = rest.split_once('/').ok_or_else(|| {
                        anyhow::Error::from(ServerlessError::MissingConfiguration {
                            provider: PROVIDER,
                            operation: "deploy",
                            detail: format!(
                                "deployment_package.source_location `{location}` is not a \
                                 valid s3://bucket/key URI"
                            ),
                        })
                    })?;
                    Ok(FunctionCode::builder().s3_bucket(bucket).s3_key(key).build())
                } else {
                    let bytes = tokio::fs::read(location).await.map_err(|e| {
                        anyhow::Error::from(ServerlessError::MissingConfiguration {
                            provider: PROVIDER,
                            operation: "deploy",
                            detail: format!(
                                "cannot read deployment package `{location}`: {e}; supply an \
                                 s3://bucket/key URI or a readable local zip"
                            ),
                        })
                    })?;
                    Ok(FunctionCode::builder().zip_file(Blob::new(bytes)).build())
                }
            },
        }
    }

    fn architecture(config: &ServerlessConfig) -> LambdaArchitecture {
        match config.cost_optimization.as_ref().map(|c| &c.architecture) {
            Some(Architecture::ARM64) => LambdaArchitecture::Arm64,
            _ => LambdaArchitecture::X8664,
        }
    }

    /// Sum a CloudWatch metric over the configured window.
    ///
    /// Returns `None` when CloudWatch answered but published no datapoints,
    /// which is a real "nothing happened in this window" reading rather than a
    /// guess.
    async fn metric_statistic(
        &self,
        function_name: &str,
        metric_name: &str,
        statistic: aws_sdk_cloudwatch::types::Statistic,
    ) -> Result<Option<f64>> {
        let client = self.cloudwatch("get_metrics")?;
        let now = std::time::SystemTime::now();
        let start = now - std::time::Duration::from_secs(self.metrics_window_secs.max(60) as u64);

        let response = client
            .get_metric_statistics()
            .namespace("AWS/Lambda")
            .metric_name(metric_name)
            .dimensions(
                aws_sdk_cloudwatch::types::Dimension::builder()
                    .name("FunctionName")
                    .value(function_name)
                    .build(),
            )
            .start_time(aws_sdk_cloudwatch::primitives::DateTime::from(start))
            .end_time(aws_sdk_cloudwatch::primitives::DateTime::from(now))
            .period(self.metrics_window_secs.max(60) as i32)
            .statistics(statistic.clone())
            .send()
            .await
            .map_err(|e| ServerlessError::api(PROVIDER, "get_metrics", format!("{e:?}")))?;

        let datapoints = response.datapoints();
        if datapoints.is_empty() {
            return Ok(None);
        }

        let values: Vec<f64> = datapoints
            .iter()
            .filter_map(|point| match statistic {
                aws_sdk_cloudwatch::types::Statistic::Sum => point.sum(),
                aws_sdk_cloudwatch::types::Statistic::Average => point.average(),
                aws_sdk_cloudwatch::types::Statistic::Maximum => point.maximum(),
                aws_sdk_cloudwatch::types::Statistic::Minimum => point.minimum(),
                aws_sdk_cloudwatch::types::Statistic::SampleCount => point.sample_count(),
                _ => None,
            })
            .collect();

        if values.is_empty() {
            return Ok(None);
        }

        Ok(Some(match statistic {
            aws_sdk_cloudwatch::types::Statistic::Sum
            | aws_sdk_cloudwatch::types::Statistic::SampleCount => values.iter().sum(),
            aws_sdk_cloudwatch::types::Statistic::Maximum => {
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            },
            aws_sdk_cloudwatch::types::Statistic::Minimum => {
                values.iter().copied().fold(f64::INFINITY, f64::min)
            },
            _ => values.iter().sum::<f64>() / values.len() as f64,
        }))
    }

    /// Query a percentile of the `Duration` metric via extended statistics.
    async fn duration_percentile(
        &self,
        function_name: &str,
        percentile: &str,
    ) -> Result<Option<f64>> {
        let client = self.cloudwatch("get_metrics")?;
        let now = std::time::SystemTime::now();
        let start = now - std::time::Duration::from_secs(self.metrics_window_secs.max(60) as u64);

        let response = client
            .get_metric_statistics()
            .namespace("AWS/Lambda")
            .metric_name("Duration")
            .dimensions(
                aws_sdk_cloudwatch::types::Dimension::builder()
                    .name("FunctionName")
                    .value(function_name)
                    .build(),
            )
            .start_time(aws_sdk_cloudwatch::primitives::DateTime::from(start))
            .end_time(aws_sdk_cloudwatch::primitives::DateTime::from(now))
            .period(self.metrics_window_secs.max(60) as i32)
            .extended_statistics(percentile)
            .send()
            .await
            .map_err(|e| ServerlessError::api(PROVIDER, "get_metrics", format!("{e:?}")))?;

        let values: Vec<f64> = response
            .datapoints()
            .iter()
            .filter_map(|point| {
                point.extended_statistics().and_then(|map| map.get(percentile).copied())
            })
            .collect();

        if values.is_empty() {
            Ok(None)
        } else {
            Ok(Some(values.iter().sum::<f64>() / values.len() as f64))
        }
    }
}

#[async_trait::async_trait]
impl ServerlessProviderTrait for AwsLambdaProvider {
    async fn deploy(&self, config: &ServerlessConfig) -> Result<DeploymentResult> {
        let client = self.lambda("deploy")?;
        let role = self.execution_role_arn.as_deref().ok_or_else(|| {
            anyhow::Error::from(ServerlessError::MissingConfiguration {
                provider: PROVIDER,
                operation: "deploy",
                detail: "CreateFunction requires an IAM execution role: build the provider \
                         with `with_execution_role(\"arn:aws:iam::…:role/…\")`"
                    .to_string(),
            })
        })?;

        let code = self.function_code(config).await?;

        let mut request = client
            .create_function()
            .function_name(&config.function_name)
            .role(role)
            .code(code)
            .memory_size(config.memory_mb as i32)
            .timeout(config.timeout_seconds as i32)
            .architectures(Self::architecture(config))
            .publish(true);

        match config.deployment_package.package_type {
            PackageType::Image => {
                request = request.package_type(LambdaPackageType::Image);
            },
            PackageType::Zip | PackageType::Source => {
                request = request
                    .package_type(LambdaPackageType::Zip)
                    .runtime(LambdaRuntime::from(config.runtime.as_str()))
                    .handler(&config.deployment_package.handler);
            },
        }

        if !config.environment_variables.is_empty() {
            let mut environment = Environment::builder();
            for (key, value) in &config.environment_variables {
                environment = environment.variables(key, value);
            }
            request = request.environment(environment.build());
        }

        if let Some(vpc) = &config.vpc_config {
            let mut vpc_builder = LambdaVpcConfig::builder();
            for subnet in &vpc.subnet_ids {
                vpc_builder = vpc_builder.subnet_ids(subnet);
            }
            for group in &vpc.security_group_ids {
                vpc_builder = vpc_builder.security_group_ids(group);
            }
            request = request.vpc_config(vpc_builder.build());
        }

        if config.monitoring.enable_xray {
            request =
                request.tracing_config(TracingConfig::builder().mode(TracingMode::Active).build());
        }

        for layer in &config.deployment_package.layers {
            request = request.layers(layer);
        }
        for (key, value) in &config.tags {
            request = request.tags(key, value);
        }

        let output = request
            .send()
            .await
            .map_err(|e| ServerlessError::api(PROVIDER, "deploy", format!("{e:?}")))?;

        let function_arn = output.function_arn().map(str::to_string);
        if function_arn.is_none() {
            return Err(ServerlessError::InvalidResponse {
                provider: PROVIDER,
                operation: "deploy",
                detail: "CreateFunction returned no FunctionArn".to_string(),
            }
            .into());
        }

        // Only create a function URL when an HTTP trigger was actually asked
        // for; never synthesize a URL that does not exist.
        let wants_http = config
            .triggers
            .iter()
            .any(|trigger| matches!(trigger.trigger_type, TriggerType::Http));
        let function_url = if wants_http {
            let url_output = client
                .create_function_url_config()
                .function_name(&config.function_name)
                .auth_type(FunctionUrlAuthType::AwsIam)
                .send()
                .await
                .map_err(|e| {
                    ServerlessError::api(PROVIDER, "create_function_url_config", format!("{e:?}"))
                })?;
            Some(url_output.function_url().to_string())
        } else {
            None
        };

        Ok(DeploymentResult {
            function_arn,
            function_url,
        })
    }

    async fn update(&self, config: &ServerlessConfig) -> Result<DeploymentResult> {
        let client = self.lambda("update")?;
        let code = self.function_code(config).await?;

        let mut code_request =
            client.update_function_code().function_name(&config.function_name).publish(true);
        if let Some(image_uri) = code.image_uri() {
            code_request = code_request.image_uri(image_uri);
        } else if let (Some(bucket), Some(key)) = (code.s3_bucket(), code.s3_key()) {
            code_request = code_request.s3_bucket(bucket).s3_key(key);
        } else if let Some(zip) = code.zip_file() {
            code_request = code_request.zip_file(zip.clone());
        }

        let code_output = code_request.send().await.map_err(|e| {
            ServerlessError::api(PROVIDER, "update_function_code", format!("{e:?}"))
        })?;

        let mut config_request = client
            .update_function_configuration()
            .function_name(&config.function_name)
            .memory_size(config.memory_mb as i32)
            .timeout(config.timeout_seconds as i32);
        if matches!(
            config.deployment_package.package_type,
            PackageType::Zip | PackageType::Source
        ) {
            config_request = config_request
                .runtime(LambdaRuntime::from(config.runtime.as_str()))
                .handler(&config.deployment_package.handler);
        }
        if !config.environment_variables.is_empty() {
            let mut environment = Environment::builder();
            for (key, value) in &config.environment_variables {
                environment = environment.variables(key, value);
            }
            config_request = config_request.environment(environment.build());
        }

        config_request.send().await.map_err(|e| {
            ServerlessError::api(PROVIDER, "update_function_configuration", format!("{e:?}"))
        })?;

        Ok(DeploymentResult {
            function_arn: code_output.function_arn().map(str::to_string),
            function_url: None,
        })
    }

    async fn delete(&self, config: &ServerlessConfig) -> Result<()> {
        let client = self.lambda("delete")?;
        client
            .delete_function()
            .function_name(&config.function_name)
            .send()
            .await
            .map_err(|e| ServerlessError::api(PROVIDER, "delete", format!("{e:?}")))?;
        Ok(())
    }

    async fn invoke(
        &self,
        config: &ServerlessConfig,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let client = self.lambda("invoke")?;
        let body = serde_json::to_vec(&payload)?;

        let output = client
            .invoke()
            .function_name(&config.function_name)
            .payload(Blob::new(body))
            .send()
            .await
            .map_err(|e| ServerlessError::api(PROVIDER, "invoke", format!("{e:?}")))?;

        if let Some(function_error) = output.function_error() {
            let detail = output
                .payload()
                .and_then(|blob| String::from_utf8(blob.as_ref().to_vec()).ok())
                .unwrap_or_default();
            return Err(ServerlessError::api(
                PROVIDER,
                "invoke",
                format!("Lambda reported {function_error}: {detail}"),
            )
            .into());
        }

        let Some(blob) = output.payload() else {
            return Err(ServerlessError::InvalidResponse {
                provider: PROVIDER,
                operation: "invoke",
                detail: "Lambda returned no response payload".to_string(),
            }
            .into());
        };

        serde_json::from_slice(blob.as_ref()).map_err(|e| {
            ServerlessError::InvalidResponse {
                provider: PROVIDER,
                operation: "invoke",
                detail: format!("Lambda response payload is not JSON: {e}"),
            }
            .into()
        })
    }

    async fn get_metrics(&self, config: &ServerlessConfig) -> Result<ServerlessMetrics> {
        use aws_sdk_cloudwatch::types::Statistic;

        // Fail before issuing any request if CloudWatch is not configured.
        self.cloudwatch("get_metrics")?;
        let name = config.function_name.as_str();

        let invocations = self.metric_statistic(name, "Invocations", Statistic::Sum).await?;
        let errors = self.metric_statistic(name, "Errors", Statistic::Sum).await?;
        let throttles = self.metric_statistic(name, "Throttles", Statistic::Sum).await?;
        let duration = self.metric_statistic(name, "Duration", Statistic::Average).await?;
        let concurrent =
            self.metric_statistic(name, "ConcurrentExecutions", Statistic::Maximum).await?;
        let provisioned_invocations = self
            .metric_statistic(name, "ProvisionedConcurrencyInvocations", Statistic::Sum)
            .await?;
        let provisioned_spillover = self
            .metric_statistic(
                name,
                "ProvisionedConcurrencySpilloverInvocations",
                Statistic::Sum,
            )
            .await?;
        let dead_letter = self.metric_statistic(name, "DeadLetterErrors", Statistic::Sum).await?;
        let iterator_age = self.metric_statistic(name, "IteratorAge", Statistic::Average).await?;
        let p50 = self.duration_percentile(name, "p50").await?;
        let p95 = self.duration_percentile(name, "p95").await?;
        let p99 = self.duration_percentile(name, "p99").await?;

        let mut metrics = ServerlessMetrics::unmeasured();
        metrics.unavailable_fields = UNAVAILABLE_FROM_CLOUDWATCH_METRICS
            .iter()
            .map(|name| (*name).to_string())
            .collect();

        let invocations_value = invocations.unwrap_or(0.0);
        let errors_value = errors.unwrap_or(0.0);

        metrics.invocations = invocations_value.max(0.0) as u64;
        metrics.errors = errors_value.max(0.0) as u64;
        metrics.throttles = throttles.unwrap_or(0.0).max(0.0) as u64;
        metrics.concurrent_executions = concurrent.unwrap_or(0.0).max(0.0) as u64;
        metrics.provisioned_concurrency_invocations =
            provisioned_invocations.unwrap_or(0.0).max(0.0) as u64;
        metrics.provisioned_concurrency_spillover =
            provisioned_spillover.unwrap_or(0.0).max(0.0) as u64;
        metrics.dead_letter_errors = dead_letter.unwrap_or(0.0).max(0.0) as u64;
        metrics.iterator_age_ms = iterator_age;

        match duration {
            Some(value) => metrics.duration_ms = value,
            None => metrics.unavailable_fields.push("duration_ms".to_string()),
        }
        match p50 {
            Some(value) => metrics.p50_duration_ms = value,
            None => metrics.unavailable_fields.push("p50_duration_ms".to_string()),
        }
        match p95 {
            Some(value) => metrics.p95_duration_ms = value,
            None => metrics.unavailable_fields.push("p95_duration_ms".to_string()),
        }
        match p99 {
            Some(value) => metrics.p99_duration_ms = value,
            None => metrics.unavailable_fields.push("p99_duration_ms".to_string()),
        }

        // A success rate is only meaningful once at least one invocation was
        // recorded; otherwise it is unmeasured, not 100 %.
        if invocations_value > 0.0 {
            metrics.success_rate =
                ((invocations_value - errors_value).max(0.0) / invocations_value) * 100.0;
        } else {
            metrics.unavailable_fields.push("success_rate".to_string());
        }

        Ok(metrics)
    }

    async fn configure_provisioned_concurrency(
        &self,
        config: &ServerlessConfig,
        concurrency: u32,
    ) -> Result<()> {
        let client = self.lambda("configure_provisioned_concurrency")?;
        let qualifier = config
            .cold_start
            .as_ref()
            .and_then(|cold| cold.pre_initialization_handler.clone())
            .unwrap_or_else(|| "$LATEST".to_string());

        client
            .put_provisioned_concurrency_config()
            .function_name(&config.function_name)
            .qualifier(qualifier)
            .provisioned_concurrent_executions(concurrency as i32)
            .send()
            .await
            .map_err(|e| {
                ServerlessError::api(
                    PROVIDER,
                    "configure_provisioned_concurrency",
                    format!("{e:?}"),
                )
            })?;
        Ok(())
    }

    async fn configure_warmup_schedule(
        &self,
        _config: &ServerlessConfig,
        _schedule: &str,
    ) -> Result<()> {
        // A warm-up schedule is an EventBridge rule plus a Lambda permission,
        // neither of which this provider holds a client for. Refuse instead of
        // logging a line and returning Ok.
        Err(ServerlessError::not_implemented(PROVIDER, "configure_warmup_schedule").into())
    }

    async fn configure_keep_warm(
        &self,
        _config: &ServerlessConfig,
        _requests_per_minute: u32,
    ) -> Result<()> {
        Err(ServerlessError::not_implemented(PROVIDER, "configure_keep_warm").into())
    }

    async fn get_detailed_metrics(&self, config: &ServerlessConfig) -> Result<DetailedMetrics> {
        let basic_metrics = self.get_metrics(config).await?;

        // The breakdowns below are not published by CloudWatch metrics. They are
        // reported as zero and the corresponding names are recorded in
        // `basic_metrics.unavailable_fields`, so no caller mistakes them for
        // measurements.
        Ok(DetailedMetrics {
            basic_metrics,
            performance_breakdown: PerformanceBreakdown {
                initialization_ms: 0.0,
                execution_ms: 0.0,
                overhead_ms: 0.0,
                network_latency_ms: 0.0,
                queue_time_ms: 0.0,
            },
            cost_breakdown: CostBreakdown {
                compute_cost: 0.0,
                request_cost: 0.0,
                data_transfer_cost: 0.0,
                storage_cost: 0.0,
                additional_services_cost: 0.0,
            },
            resource_utilization: ResourceUtilization {
                cpu_usage_percent: 0.0,
                memory_usage_mb: 0,
                network_in_mb: 0.0,
                network_out_mb: 0.0,
                disk_usage_mb: 0.0,
            },
        })
    }

    async fn configure_auto_scaling(
        &self,
        config: &ServerlessConfig,
        scaling_config: &ScalingConfig,
    ) -> Result<()> {
        let client = self.lambda("configure_auto_scaling")?;
        let Some(limit) = scaling_config.concurrency_limit else {
            return Err(ServerlessError::MissingConfiguration {
                provider: PROVIDER,
                operation: "configure_auto_scaling",
                detail: "Lambda scales automatically; the only knob is a reserved concurrency \
                         limit, so `scaling.concurrency_limit` must be set"
                    .to_string(),
            }
            .into());
        };

        client
            .put_function_concurrency()
            .function_name(&config.function_name)
            .reserved_concurrent_executions(limit as i32)
            .send()
            .await
            .map_err(|e| {
                ServerlessError::api(PROVIDER, "configure_auto_scaling", format!("{e:?}"))
            })?;
        Ok(())
    }
}

impl Default for AwsLambdaProvider {
    fn default() -> Self {
        Self::new("us-east-1".to_string())
    }
}
