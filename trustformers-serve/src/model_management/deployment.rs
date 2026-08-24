//! Deployment Strategies
//!
//! Handles canary deployments, blue-green deployments, and A/B testing for model rollouts.

use crate::model_management::{
    config::{ABTestConfig, BlueGreenConfig, CanaryConfig},
    ModelError, ModelMetrics, ModelResult,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use uuid::Uuid;

/// Deployment status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeploymentStatus {
    /// Deployment is being planned
    Planning,
    /// Deployment is in progress
    InProgress,
    /// Deployment completed successfully
    Completed,
    /// Deployment failed
    Failed { error: String },
    /// Deployment was aborted
    Aborted,
    /// Deployment is being rolled back
    RollingBack,
    /// Rollback completed
    RolledBack,
}

/// Canary deployment state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryDeployment {
    /// Deployment ID
    pub id: String,
    /// Model being deployed
    pub model_id: String,
    /// Current traffic percentage
    pub current_percentage: f32,
    /// Target traffic percentage
    pub target_percentage: f32,
    /// Current status
    pub status: DeploymentStatus,
    /// Start time
    pub started_at: DateTime<Utc>,
    /// Last update time
    pub updated_at: DateTime<Utc>,
    /// Step history
    pub steps: Vec<CanaryStep>,
    /// Metrics collected during deployment
    pub metrics: CanaryMetrics,
    /// Configuration used
    pub config: CanaryConfig,
}

/// Individual canary step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryStep {
    /// Step number
    pub step: u32,
    /// Traffic percentage for this step
    pub percentage: f32,
    /// Step start time
    pub started_at: DateTime<Utc>,
    /// Step completion time
    pub completed_at: Option<DateTime<Utc>>,
    /// Whether step was successful
    pub success: bool,
    /// Error message if step failed
    pub error: Option<String>,
    /// Metrics during this step
    pub metrics: ModelMetrics,
}

/// Canary deployment metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CanaryMetrics {
    /// Success rate comparison (new vs old)
    pub success_rate_new: f32,
    pub success_rate_old: f32,
    /// Latency comparison
    pub avg_latency_new: f32,
    pub avg_latency_old: f32,
    /// Error rate comparison
    pub error_rate_new: f32,
    pub error_rate_old: f32,
    /// Total requests served by new version
    pub requests_new: u64,
    /// Total requests served by old version
    pub requests_old: u64,
}

/// Blue-green deployment state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueGreenDeployment {
    /// Deployment ID
    pub id: String,
    /// Blue environment model ID (current)
    pub blue_model_id: Option<String>,
    /// Green environment model ID (new)
    pub green_model_id: String,
    /// Current active environment
    pub active_environment: Environment,
    /// Deployment status
    pub status: DeploymentStatus,
    /// Start time
    pub started_at: DateTime<Utc>,
    /// Switch time (when traffic was switched)
    pub switched_at: Option<DateTime<Utc>>,
    /// Validation results
    pub validation_results: Vec<ValidationResult>,
    /// Configuration used
    pub config: BlueGreenConfig,
}

/// Environment identifier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Environment {
    Blue,
    Green,
}

/// Validation result for blue-green deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Validation check name
    pub check_name: String,
    /// Whether validation passed
    pub passed: bool,
    /// Validation message
    pub message: String,
    /// Execution time
    pub executed_at: DateTime<Utc>,
    /// Metrics collected during validation
    pub metrics: HashMap<String, f64>,
}

/// A/B test deployment state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestDeployment {
    /// Test ID
    pub id: String,
    /// Test name
    pub name: String,
    /// Model variants being tested
    pub variants: Vec<ABTestVariant>,
    /// Current status
    pub status: DeploymentStatus,
    /// Start time
    pub started_at: DateTime<Utc>,
    /// Planned end time
    pub ends_at: DateTime<Utc>,
    /// Current results
    pub results: ABTestResults,
    /// Configuration used
    pub config: ABTestConfig,
}

/// A/B test variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestVariant {
    /// Variant ID
    pub id: String,
    /// Variant name
    pub name: String,
    /// Model ID for this variant
    pub model_id: String,
    /// Traffic allocation percentage
    pub traffic_percentage: f32,
    /// Metrics for this variant
    pub metrics: ModelMetrics,
}

/// A/B test results and analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ABTestResults {
    /// Statistical significance achieved
    pub significant: bool,
    /// Confidence level
    pub confidence_level: f32,
    /// Sample size per variant
    pub sample_sizes: HashMap<String, u64>,
    /// Conversion rates per variant
    pub conversion_rates: HashMap<String, f32>,
    /// Performance metrics per variant
    pub performance_metrics: HashMap<String, HashMap<String, f64>>,
    /// Winning variant (if any)
    pub winner: Option<String>,
    /// Recommendation
    pub recommendation: String,
}

/// Deployment manager handles all deployment strategies
pub struct DeploymentManager {
    /// Active canary deployments
    canary_deployments: Arc<RwLock<HashMap<String, CanaryDeployment>>>,
    /// Active blue-green deployments
    blue_green_deployments: Arc<RwLock<HashMap<String, BlueGreenDeployment>>>,
    /// Active A/B tests
    ab_tests: Arc<RwLock<HashMap<String, ABTestDeployment>>>,
    /// Traffic router for directing requests
    traffic_router: Arc<TrafficRouter>,
    /// Real per-model request counters, the sole source of canary metrics.
    metrics: Arc<DeploymentMetricsRegistry>,
    /// Validators available to `run_validation_check`, keyed by check name.
    validators: Arc<RwLock<HashMap<String, Arc<dyn DeploymentValidator>>>>,
}

/// Live per-model request counters recorded by the serving path.
///
/// Canary decisions read exclusively from here, so a promotion can only happen
/// on traffic that was genuinely served.
#[derive(Debug, Default)]
pub struct DeploymentMetricsRegistry {
    counters: RwLock<HashMap<String, ModelRequestCounters>>,
}

/// Accumulated outcomes for one model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelRequestCounters {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_latency_ms: f64,
}

impl ModelRequestCounters {
    /// Fraction of requests that succeeded, or `None` when nothing was served.
    pub fn success_rate(&self) -> Option<f32> {
        if self.total_requests == 0 {
            None
        } else {
            Some(self.successful_requests as f32 / self.total_requests as f32)
        }
    }

    /// Fraction of requests that failed, or `None` when nothing was served.
    pub fn error_rate(&self) -> Option<f32> {
        if self.total_requests == 0 {
            None
        } else {
            Some(self.failed_requests as f32 / self.total_requests as f32)
        }
    }

    /// Mean latency, or `None` when nothing was served.
    pub fn avg_latency_ms(&self) -> Option<f32> {
        if self.total_requests == 0 {
            None
        } else {
            Some((self.total_latency_ms / self.total_requests as f64) as f32)
        }
    }
}

impl DeploymentMetricsRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one served request against `model_id`.
    pub fn record_request(&self, model_id: &str, success: bool, latency_ms: f64) {
        let mut counters = self.counters.write().unwrap_or_else(|p| p.into_inner());
        let entry = counters.entry(model_id.to_string()).or_default();
        entry.total_requests += 1;
        if success {
            entry.successful_requests += 1;
        } else {
            entry.failed_requests += 1;
        }
        entry.total_latency_ms += latency_ms;
    }

    /// Counters for `model_id`, or `None` when it has served nothing.
    pub fn counters_for(&self, model_id: &str) -> Option<ModelRequestCounters> {
        let counters = self.counters.read().unwrap_or_else(|p| p.into_inner());
        counters.get(model_id).cloned()
    }
}

/// Runs one named pre-promotion validation check against a model.
#[async_trait::async_trait]
pub trait DeploymentValidator: Send + Sync + std::fmt::Debug {
    /// Execute the check. Returning `Err` marks the check as failed.
    async fn validate(&self, model_id: &str) -> ModelResult<ValidationResult>;
}

impl DeploymentManager {
    /// Create a new deployment manager
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(DeploymentMetricsRegistry::new()),
            validators: Arc::new(RwLock::new(HashMap::new())),
            canary_deployments: Arc::new(RwLock::new(HashMap::new())),
            blue_green_deployments: Arc::new(RwLock::new(HashMap::new())),
            ab_tests: Arc::new(RwLock::new(HashMap::new())),
            traffic_router: Arc::new(TrafficRouter::new()),
        }
    }

    /// Start a canary deployment
    pub async fn start_canary_deployment(
        &self,
        model_id: String,
        target_percentage: f32,
        config: CanaryConfig,
    ) -> ModelResult<String> {
        let deployment_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let deployment = CanaryDeployment {
            id: deployment_id.clone(),
            model_id: model_id.clone(),
            current_percentage: 0.0,
            target_percentage,
            status: DeploymentStatus::Planning,
            started_at: now,
            updated_at: now,
            steps: Vec::new(),
            metrics: CanaryMetrics::default(),
            config: config.clone(),
        };

        // Add to active deployments
        {
            let mut deployments =
                self.canary_deployments.write().unwrap_or_else(|p| p.into_inner());
            deployments.insert(deployment_id.clone(), deployment);
        }

        // Start the canary progression
        self.progress_canary_deployment(&deployment_id).await?;

        Ok(deployment_id)
    }

    /// Progress a canary deployment to next step
    async fn progress_canary_deployment(&self, deployment_id: &str) -> ModelResult<()> {
        let mut deployment = {
            let deployments = self.canary_deployments.read().unwrap_or_else(|p| p.into_inner());
            deployments
                .get(deployment_id)
                .cloned()
                .ok_or_else(|| ModelError::DeploymentFailed {
                    error: format!("Canary deployment {} not found", deployment_id),
                })?
        };

        if deployment.status != DeploymentStatus::Planning
            && deployment.status != DeploymentStatus::InProgress
        {
            return Ok(());
        }

        // Calculate next percentage
        let next_percentage = if deployment.current_percentage == 0.0 {
            deployment.config.default_percentage
        } else {
            (deployment.current_percentage + deployment.config.step_size)
                .min(deployment.target_percentage)
        };

        // Update traffic routing
        self.traffic_router.set_canary_traffic(&deployment.model_id, next_percentage)?;

        // Create new step
        let step = CanaryStep {
            step: deployment.steps.len() as u32 + 1,
            percentage: next_percentage,
            started_at: Utc::now(),
            completed_at: None,
            success: false,
            error: None,
            metrics: ModelMetrics::default(),
        };

        deployment.steps.push(step);
        deployment.current_percentage = next_percentage;
        deployment.status = DeploymentStatus::InProgress;
        deployment.updated_at = Utc::now();

        // Update deployment
        {
            let mut deployments =
                self.canary_deployments.write().unwrap_or_else(|p| p.into_inner());
            deployments.insert(deployment_id.to_string(), deployment.clone());
        }

        // Note: In a real implementation, a background scheduler would handle
        // timing and evaluation of canary steps. For now, we just log the step creation.
        tracing::info!(
            "Started canary step {} for deployment {}",
            next_percentage,
            deployment_id
        );

        Ok(())
    }

    /// Evaluate current canary step and decide whether to continue
    async fn evaluate_canary_step(&self, deployment_id: &str) -> ModelResult<()> {
        let mut deployment = {
            let deployments = self.canary_deployments.read().unwrap_or_else(|p| p.into_inner());
            deployments
                .get(deployment_id)
                .cloned()
                .ok_or_else(|| ModelError::DeploymentFailed {
                    error: format!("Canary deployment {} not found", deployment_id),
                })?
        };

        // Metrics come from the real request counters for the canary and its
        // baseline; if either has served nothing, evaluation fails loudly.
        let baseline_id = self
            .traffic_router
            .baseline_for(&deployment.model_id)
            .unwrap_or_else(|| format!("{}-baseline", deployment.model_id));
        let metrics = self.collect_canary_metrics(&deployment.model_id, &baseline_id).await?;
        deployment.metrics = metrics;

        // Check success criteria
        let success_rate = deployment.metrics.success_rate_new;
        let error_rate = deployment.metrics.error_rate_new;

        let step_success = success_rate >= deployment.config.success_threshold
            && error_rate <= deployment.config.error_threshold;

        // Update current step
        if let Some(current_step) = deployment.steps.last_mut() {
            current_step.completed_at = Some(Utc::now());
            current_step.success = step_success;
            current_step.metrics = ModelMetrics {
                successful_requests: (deployment.metrics.requests_new as f32 * success_rate) as u64,
                failed_requests: (deployment.metrics.requests_new as f32 * error_rate) as u64,
                total_requests: deployment.metrics.requests_new,
                avg_latency_ms: deployment.metrics.avg_latency_new,
                ..Default::default()
            };
        }

        if !step_success && deployment.config.auto_rollback {
            // Rollback
            deployment.status = DeploymentStatus::RollingBack;
            self.rollback_canary_deployment(deployment_id).await?;
        } else if deployment.current_percentage >= deployment.target_percentage {
            // Deployment complete
            deployment.status = DeploymentStatus::Completed;
        } else if step_success {
            // Step succeeded, but don't automatically progress here
            // The deployment manager would handle progression through a separate scheduler
            tracing::info!("Canary step succeeded for deployment {}", deployment_id);
        } else {
            // Step failed but no auto-rollback
            deployment.status = DeploymentStatus::Failed {
                error: "Step failed to meet success criteria".to_string(),
            };
        }

        // Update deployment
        {
            let mut deployments =
                self.canary_deployments.write().unwrap_or_else(|p| p.into_inner());
            deployments.insert(deployment_id.to_string(), deployment);
        }

        Ok(())
    }

    /// Rollback a canary deployment
    async fn rollback_canary_deployment(&self, deployment_id: &str) -> ModelResult<()> {
        let mut deployment = {
            let deployments = self.canary_deployments.read().unwrap_or_else(|p| p.into_inner());
            deployments
                .get(deployment_id)
                .cloned()
                .ok_or_else(|| ModelError::DeploymentFailed {
                    error: format!("Canary deployment {} not found", deployment_id),
                })?
        };

        // Reset traffic to 0%
        self.traffic_router.set_canary_traffic(&deployment.model_id, 0.0)?;

        deployment.status = DeploymentStatus::RolledBack;
        deployment.updated_at = Utc::now();

        // Update deployment
        {
            let mut deployments =
                self.canary_deployments.write().unwrap_or_else(|p| p.into_inner());
            deployments.insert(deployment_id.to_string(), deployment);
        }

        Ok(())
    }

    /// Start a blue-green deployment
    pub async fn start_blue_green_deployment(
        &self,
        blue_model_id: Option<String>,
        green_model_id: String,
        config: BlueGreenConfig,
    ) -> ModelResult<String> {
        let deployment_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let deployment = BlueGreenDeployment {
            id: deployment_id.clone(),
            blue_model_id,
            green_model_id: green_model_id.clone(),
            active_environment: Environment::Blue,
            status: DeploymentStatus::Planning,
            started_at: now,
            switched_at: None,
            validation_results: Vec::new(),
            config: config.clone(),
        };

        // Add to active deployments
        {
            let mut deployments =
                self.blue_green_deployments.write().unwrap_or_else(|p| p.into_inner());
            deployments.insert(deployment_id.clone(), deployment);
        }

        // Start validation process
        self.validate_green_environment(&deployment_id).await?;

        Ok(deployment_id)
    }

    /// Validate green environment before switching
    async fn validate_green_environment(&self, deployment_id: &str) -> ModelResult<()> {
        let mut deployment = {
            let deployments = self.blue_green_deployments.read().unwrap_or_else(|p| p.into_inner());
            deployments
                .get(deployment_id)
                .cloned()
                .ok_or_else(|| ModelError::DeploymentFailed {
                    error: format!("Blue-green deployment {} not found", deployment_id),
                })?
        };

        deployment.status = DeploymentStatus::InProgress;

        // Run validation checks
        for check_name in &deployment.config.validation_checks {
            let result = self.run_validation_check(check_name, &deployment.green_model_id).await?;
            deployment.validation_results.push(result);
        }

        // Check if all validations passed
        let all_passed = deployment.validation_results.iter().all(|r| r.passed);

        if all_passed {
            // Switch to green environment
            self.switch_to_green_environment(deployment_id).await?;
        } else if deployment.config.auto_rollback {
            // Rollback (keep blue active)
            deployment.status = DeploymentStatus::RolledBack;
        } else {
            deployment.status = DeploymentStatus::Failed {
                error: "Validation checks failed".to_string(),
            };
        }

        // Update deployment
        {
            let mut deployments =
                self.blue_green_deployments.write().unwrap_or_else(|p| p.into_inner());
            deployments.insert(deployment_id.to_string(), deployment);
        }

        Ok(())
    }

    /// Switch traffic to green environment
    async fn switch_to_green_environment(&self, deployment_id: &str) -> ModelResult<()> {
        let mut deployment = {
            let deployments = self.blue_green_deployments.read().unwrap_or_else(|p| p.into_inner());
            deployments
                .get(deployment_id)
                .cloned()
                .ok_or_else(|| ModelError::DeploymentFailed {
                    error: format!("Blue-green deployment {} not found", deployment_id),
                })?
        };

        // Switch traffic to green
        self.traffic_router.switch_to_green(&deployment.green_model_id)?;

        deployment.active_environment = Environment::Green;
        deployment.switched_at = Some(Utc::now());
        deployment.status = DeploymentStatus::Completed;

        // Update deployment
        {
            let mut deployments =
                self.blue_green_deployments.write().unwrap_or_else(|p| p.into_inner());
            deployments.insert(deployment_id.to_string(), deployment);
        }

        Ok(())
    }

    /// Start an A/B test
    pub async fn start_ab_test(
        &self,
        name: String,
        variants: Vec<(String, String, f32)>, // (name, model_id, traffic_percentage)
        config: ABTestConfig,
    ) -> ModelResult<String> {
        let test_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let test_variants: Vec<ABTestVariant> = variants
            .into_iter()
            .map(|(name, model_id, traffic_percentage)| ABTestVariant {
                id: Uuid::new_v4().to_string(),
                name,
                model_id,
                traffic_percentage,
                metrics: ModelMetrics::default(),
            })
            .collect();

        let test = ABTestDeployment {
            id: test_id.clone(),
            name,
            variants: test_variants,
            status: DeploymentStatus::InProgress,
            started_at: now,
            ends_at: now
                + chrono::Duration::from_std(config.test_duration)
                    .unwrap_or_else(|_| chrono::Duration::zero()),
            results: ABTestResults::default(),
            config,
        };

        // Configure traffic splitting
        for variant in &test.variants {
            self.traffic_router.set_ab_test_traffic(
                &test.id,
                &variant.model_id,
                variant.traffic_percentage,
            )?;
        }

        // Add to active tests
        {
            let mut tests = self.ab_tests.write().unwrap_or_else(|p| p.into_inner());
            tests.insert(test_id.clone(), test);
        }

        Ok(test_id)
    }

    /// Collect canary metrics from the real request counters.
    ///
    /// `model_id` is the candidate; `baseline_id` is the model it is replacing.
    ///
    /// # Errors
    ///
    /// Fails when either side has served no traffic. Promotion therefore cannot
    /// happen on absent data — the previous implementation returned constants in
    /// which the new model always looked better.
    pub async fn collect_canary_metrics(
        &self,
        model_id: &str,
        baseline_id: &str,
    ) -> ModelResult<CanaryMetrics> {
        let new =
            self.metrics
                .counters_for(model_id)
                .ok_or_else(|| ModelError::DeploymentFailed {
                    error: format!(
                        "canary model {} has served no requests; there is nothing to evaluate",
                        model_id
                    ),
                })?;
        let old =
            self.metrics
                .counters_for(baseline_id)
                .ok_or_else(|| ModelError::DeploymentFailed {
                    error: format!(
                    "baseline model {} has served no requests; there is nothing to compare against",
                    baseline_id
                ),
                })?;

        let missing = |what: &str, which: &str| ModelError::DeploymentFailed {
            error: format!("{} for {} is unavailable", what, which),
        };

        Ok(CanaryMetrics {
            success_rate_new: new
                .success_rate()
                .ok_or_else(|| missing("success rate", model_id))?,
            success_rate_old: old
                .success_rate()
                .ok_or_else(|| missing("success rate", baseline_id))?,
            avg_latency_new: new
                .avg_latency_ms()
                .ok_or_else(|| missing("average latency", model_id))?,
            avg_latency_old: old
                .avg_latency_ms()
                .ok_or_else(|| missing("average latency", baseline_id))?,
            error_rate_new: new.error_rate().ok_or_else(|| missing("error rate", model_id))?,
            error_rate_old: old.error_rate().ok_or_else(|| missing("error rate", baseline_id))?,
            requests_new: new.total_requests,
            requests_old: old.total_requests,
        })
    }

    /// Run a named validation check by dispatching to its registered validator.
    ///
    /// An unregistered check name produces a *failed* result: an unrunnable
    /// check must never read as a pass.
    async fn run_validation_check(
        &self,
        check_name: &str,
        model_id: &str,
    ) -> ModelResult<ValidationResult> {
        let validator = {
            let validators = self.validators.read().unwrap_or_else(|p| p.into_inner());
            validators.get(check_name).cloned()
        };

        let Some(validator) = validator else {
            return Ok(ValidationResult {
                check_name: check_name.to_string(),
                passed: false,
                message: format!(
                    "no validator is registered for check '{}'; register one with \
                     DeploymentManager::register_validator",
                    check_name
                ),
                executed_at: Utc::now(),
                metrics: HashMap::new(),
            });
        };

        match validator.validate(model_id).await {
            Ok(result) => Ok(result),
            Err(e) => Ok(ValidationResult {
                check_name: check_name.to_string(),
                passed: false,
                message: format!("validator failed: {}", e),
                executed_at: Utc::now(),
                metrics: HashMap::new(),
            }),
        }
    }

    /// Clone for background tasks
    fn clone_for_background(&self) -> DeploymentManager {
        DeploymentManager {
            canary_deployments: Arc::clone(&self.canary_deployments),
            blue_green_deployments: Arc::clone(&self.blue_green_deployments),
            ab_tests: Arc::clone(&self.ab_tests),
            traffic_router: Arc::clone(&self.traffic_router),
            metrics: Arc::clone(&self.metrics),
            validators: Arc::clone(&self.validators),
        }
    }

    /// The live request-metrics registry backing canary decisions.
    pub fn metrics(&self) -> &Arc<DeploymentMetricsRegistry> {
        &self.metrics
    }

    /// The traffic router used to place requests.
    pub fn traffic_router(&self) -> &Arc<TrafficRouter> {
        &self.traffic_router
    }

    /// Register a validator for a named pre-promotion check.
    pub fn register_validator(&self, check_name: &str, validator: Arc<dyn DeploymentValidator>) {
        let mut validators = self.validators.write().unwrap_or_else(|p| p.into_inner());
        validators.insert(check_name.to_string(), validator);
    }
}

impl Default for DeploymentManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Traffic router for managing request routing
pub struct TrafficRouter {
    /// Canary traffic percentages
    canary_traffic: Arc<RwLock<HashMap<String, f32>>>,
    /// A/B test traffic routing
    ab_test_traffic: Arc<RwLock<HashMap<String, HashMap<String, f32>>>>,
    /// Blue-green active models
    blue_green_active: Arc<RwLock<HashMap<String, String>>>,
    /// Baseline model each canary is compared against
    canary_baselines: Arc<RwLock<HashMap<String, String>>>,
}

/// Slot key used when a blue/green deployment does not name one.
const BLUE_GREEN_DEFAULT_SLOT: &str = "default";

impl TrafficRouter {
    /// Create a new traffic router
    pub fn new() -> Self {
        Self {
            canary_traffic: Arc::new(RwLock::new(HashMap::new())),
            ab_test_traffic: Arc::new(RwLock::new(HashMap::new())),
            blue_green_active: Arc::new(RwLock::new(HashMap::new())),
            canary_baselines: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set canary traffic percentage
    pub fn set_canary_traffic(&self, model_id: &str, percentage: f32) -> ModelResult<()> {
        let mut canary_traffic = self.canary_traffic.write().unwrap_or_else(|p| p.into_inner());
        canary_traffic.insert(model_id.to_string(), percentage);
        Ok(())
    }

    /// Set A/B test traffic
    pub fn set_ab_test_traffic(
        &self,
        test_id: &str,
        model_id: &str,
        percentage: f32,
    ) -> ModelResult<()> {
        let mut ab_test_traffic = self.ab_test_traffic.write().unwrap_or_else(|p| p.into_inner());
        ab_test_traffic
            .entry(test_id.to_string())
            .or_default()
            .insert(model_id.to_string(), percentage);
        Ok(())
    }

    /// Record the baseline a canary is being compared against.
    pub fn set_canary_baseline(&self, canary_model_id: &str, baseline_model_id: &str) {
        let mut baselines = self.canary_baselines.write().unwrap_or_else(|p| p.into_inner());
        baselines.insert(canary_model_id.to_string(), baseline_model_id.to_string());
    }

    /// Baseline registered for a canary model, if any.
    pub fn baseline_for(&self, canary_model_id: &str) -> Option<String> {
        let baselines = self.canary_baselines.read().unwrap_or_else(|p| p.into_inner());
        baselines.get(canary_model_id).cloned()
    }

    /// Make `green_model_id` the active model for its deployment.
    ///
    /// Subsequent `route_request` calls resolve to it.
    pub fn switch_to_green(&self, green_model_id: &str) -> ModelResult<()> {
        let mut active = self.blue_green_active.write().unwrap_or_else(|p| p.into_inner());
        active.insert(
            BLUE_GREEN_DEFAULT_SLOT.to_string(),
            green_model_id.to_string(),
        );
        tracing::info!("Switched traffic to green model: {}", green_model_id);
        Ok(())
    }

    /// The model currently serving the default blue/green slot, if one is set.
    pub fn active_model(&self) -> Option<String> {
        let active = self.blue_green_active.read().unwrap_or_else(|p| p.into_inner());
        active.get(BLUE_GREEN_DEFAULT_SLOT).cloned()
    }

    /// Route a request to a model.
    ///
    /// Routing is deterministic in `request_id`, so the same request always lands
    /// on the same model and the traffic split is reproducible:
    ///
    /// 1. an active A/B test wins, splitting by cumulative variant weight;
    /// 2. otherwise a configured canary takes its configured share;
    /// 3. otherwise the active blue/green model serves.
    ///
    /// # Errors
    ///
    /// Returns `None` when nothing is configured, rather than naming a
    /// "default-model" that may not exist.
    pub fn route_request(&self, request_id: &str) -> Option<String> {
        let bucket = Self::bucket_of(request_id);

        // A/B tests take precedence when one is configured.
        {
            let ab = self.ab_test_traffic.read().unwrap_or_else(|p| p.into_inner());
            if let Some((_test_id, variants)) = ab.iter().next() {
                let mut ordered: Vec<(&String, &f32)> = variants.iter().collect();
                ordered.sort_by(|a, b| a.0.cmp(b.0));
                let mut cumulative = 0.0f32;
                for (model_id, percentage) in ordered {
                    cumulative += *percentage;
                    if bucket < cumulative {
                        return Some(model_id.clone());
                    }
                }
            }
        }

        // Canary split.
        {
            let canary = self.canary_traffic.read().unwrap_or_else(|p| p.into_inner());
            let mut ordered: Vec<(&String, &f32)> = canary.iter().collect();
            ordered.sort_by(|a, b| a.0.cmp(b.0));
            let mut cumulative = 0.0f32;
            for (model_id, percentage) in ordered {
                cumulative += *percentage;
                if bucket < cumulative {
                    return Some(model_id.clone());
                }
            }
        }

        self.active_model()
    }

    /// Map a request id onto a stable bucket in `[0, 100)`.
    fn bucket_of(request_id: &str) -> f32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        request_id.hash(&mut hasher);
        (hasher.finish() % 10_000) as f32 / 100.0
    }
}

impl Default for TrafficRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: canary metrics used to be a constant in which the new model
    /// always looked better. They must now come from the real counters.
    #[tokio::test]
    async fn canary_metrics_come_from_real_counters() {
        let manager = DeploymentManager::new();
        let metrics = manager.metrics();

        // Candidate: 3 requests, 1 failure.
        metrics.record_request("candidate", true, 100.0);
        metrics.record_request("candidate", true, 200.0);
        metrics.record_request("candidate", false, 300.0);
        // Baseline: 2 requests, both successful.
        metrics.record_request("baseline", true, 50.0);
        metrics.record_request("baseline", true, 150.0);

        let canary = manager
            .collect_canary_metrics("candidate", "baseline")
            .await
            .expect("both sides have traffic");

        assert_eq!(canary.requests_new, 3);
        assert_eq!(canary.requests_old, 2);
        assert!((canary.success_rate_new - 2.0 / 3.0).abs() < 1e-6);
        assert!((canary.success_rate_old - 1.0).abs() < 1e-6);
        assert!((canary.error_rate_new - 1.0 / 3.0).abs() < 1e-6);
        assert!((canary.avg_latency_new - 200.0).abs() < 1e-3);
        assert!((canary.avg_latency_old - 100.0).abs() < 1e-3);

        // The old constants must not reappear.
        assert_ne!(canary.success_rate_new, 0.95);
        assert_ne!(canary.requests_new, 1000);
    }

    /// Regression: a canary with no traffic must not be evaluated at all.
    #[tokio::test]
    async fn canary_without_traffic_cannot_be_evaluated() {
        let manager = DeploymentManager::new();
        manager.metrics().record_request("baseline", true, 10.0);

        let error = manager
            .collect_canary_metrics("candidate", "baseline")
            .await
            .expect_err("a candidate with no traffic must not be promotable");
        assert!(error.to_string().contains("served no requests"));
    }

    /// Regression: an unregistered validation check must fail, not pass.
    #[tokio::test]
    async fn unregistered_validation_check_fails() {
        let manager = DeploymentManager::new();
        let result = manager
            .run_validation_check("smoke", "model-a")
            .await
            .expect("the check must produce a result");
        assert!(!result.passed);
        assert!(result.message.contains("no validator is registered"));
        assert_ne!(result.message, "Validation passed");
    }

    #[derive(Debug)]
    struct AlwaysPasses;

    #[async_trait::async_trait]
    impl DeploymentValidator for AlwaysPasses {
        async fn validate(&self, model_id: &str) -> ModelResult<ValidationResult> {
            Ok(ValidationResult {
                check_name: "smoke".to_string(),
                passed: true,
                message: format!("{} passed the smoke check", model_id),
                executed_at: Utc::now(),
                metrics: HashMap::new(),
            })
        }
    }

    #[tokio::test]
    async fn registered_validation_check_is_dispatched() {
        let manager = DeploymentManager::new();
        manager.register_validator("smoke", Arc::new(AlwaysPasses));

        let result = manager
            .run_validation_check("smoke", "model-a")
            .await
            .expect("the check must produce a result");
        assert!(result.passed);
        assert!(result.message.contains("model-a"));
    }

    /// Regression: routing used to return the literal "default-model".
    #[test]
    fn routing_honours_the_configured_split() {
        let router = TrafficRouter::new();
        assert_eq!(router.route_request("req-1"), None);

        router.switch_to_green("green-model").expect("switch succeeds");
        assert_eq!(
            router.route_request("req-1"),
            Some("green-model".to_string())
        );
        assert_eq!(router.active_model(), Some("green-model".to_string()));

        // A canary taking 100% of traffic must serve every request.
        router.set_canary_traffic("canary-model", 100.0).expect("set succeeds");
        for id in ["a", "b", "c", "d", "e"] {
            assert_eq!(
                router.route_request(id),
                Some("canary-model".to_string()),
                "request {id} must reach the canary"
            );
        }

        // Routing is deterministic in the request id.
        let first = router.route_request("stable-id");
        assert_eq!(first, router.route_request("stable-id"));
    }

    /// A partial canary split must actually send some traffic each way.
    #[test]
    fn partial_canary_split_divides_traffic() {
        let router = TrafficRouter::new();
        router.switch_to_green("stable-model").expect("switch succeeds");
        router.set_canary_traffic("canary-model", 50.0).expect("set succeeds");

        let mut canary = 0usize;
        let mut stable = 0usize;
        for i in 0..500 {
            match router.route_request(&format!("request-{i}")) {
                Some(model) if model == "canary-model" => canary += 1,
                Some(model) if model == "stable-model" => stable += 1,
                other => panic!("unexpected routing target: {other:?}"),
            }
        }
        assert!(canary > 100, "canary received too little traffic: {canary}");
        assert!(stable > 100, "stable received too little traffic: {stable}");
    }

    #[test]
    fn counters_report_none_without_traffic() {
        let counters = ModelRequestCounters::default();
        assert!(counters.success_rate().is_none());
        assert!(counters.error_rate().is_none());
        assert!(counters.avg_latency_ms().is_none());
    }
}
