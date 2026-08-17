//! # AzureFunctionsProvider - Trait Implementations
//!
//! AzureFunctionsProvider has no real Azure Functions client wired up, so every operation returns
//! [`ServerlessError::NotImplemented`] rather than a fabricated resource id,
//! an echoed payload or an invented metric snapshot. A deployment pipeline that
//! targets Azure Functions therefore fails loudly instead of believing it shipped a
//! function that does not exist.

use anyhow::Result;

use super::errors::ServerlessError;
use super::functions::ServerlessProviderTrait;
use super::types::{
    AzureFunctionsProvider, DeploymentResult, DetailedMetrics, ScalingConfig, ServerlessConfig,
    ServerlessMetrics,
};

const PROVIDER: &str = "AzureFunctionsProvider";

#[async_trait::async_trait]
impl ServerlessProviderTrait for AzureFunctionsProvider {
    async fn deploy(&self, _config: &ServerlessConfig) -> Result<DeploymentResult> {
        Err(ServerlessError::not_implemented(PROVIDER, "deploy").into())
    }
    async fn update(&self, _config: &ServerlessConfig) -> Result<DeploymentResult> {
        Err(ServerlessError::not_implemented(PROVIDER, "update").into())
    }
    async fn delete(&self, _config: &ServerlessConfig) -> Result<()> {
        Err(ServerlessError::not_implemented(PROVIDER, "delete").into())
    }
    async fn invoke(
        &self,
        _config: &ServerlessConfig,
        _payload: serde_json::Value,
    ) -> Result<serde_json::Value> {
        Err(ServerlessError::not_implemented(PROVIDER, "invoke").into())
    }
    async fn get_metrics(&self, _config: &ServerlessConfig) -> Result<ServerlessMetrics> {
        Err(ServerlessError::not_implemented(PROVIDER, "get_metrics").into())
    }
    async fn configure_provisioned_concurrency(
        &self,
        _config: &ServerlessConfig,
        _concurrency: u32,
    ) -> Result<()> {
        Err(ServerlessError::not_implemented(PROVIDER, "configure_provisioned_concurrency").into())
    }
    async fn configure_warmup_schedule(
        &self,
        _config: &ServerlessConfig,
        _schedule: &str,
    ) -> Result<()> {
        Err(ServerlessError::not_implemented(PROVIDER, "configure_warmup_schedule").into())
    }
    async fn configure_keep_warm(
        &self,
        _config: &ServerlessConfig,
        _requests_per_minute: u32,
    ) -> Result<()> {
        Err(ServerlessError::not_implemented(PROVIDER, "configure_keep_warm").into())
    }
    async fn get_detailed_metrics(&self, _config: &ServerlessConfig) -> Result<DetailedMetrics> {
        Err(ServerlessError::not_implemented(PROVIDER, "get_detailed_metrics").into())
    }
    async fn configure_auto_scaling(
        &self,
        _config: &ServerlessConfig,
        _scaling_config: &ScalingConfig,
    ) -> Result<()> {
        Err(ServerlessError::not_implemented(PROVIDER, "configure_auto_scaling").into())
    }
}
