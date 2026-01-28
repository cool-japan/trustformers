use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEntry {
    pub entry_id: String,
    pub job_id: String,
    pub resource_type: String,
    pub resource_amount: f64,
    pub cost_per_unit: f64,
    pub total_cost: f64,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub duration: Option<Duration>,
    pub billing_model: BillingModel,
    pub tags: HashMap<String, String>,
    pub region: String,
    pub provider: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BillingModel {
    PayPerUse,
    Subscription,
    Spot,
    Reserved,
    Preemptible,
    Custom { name: String, rate: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub budget_id: String,
    pub name: String,
    pub total_amount: f64,
    pub spent_amount: f64,
    pub remaining_amount: f64,
    pub period: BudgetPeriod,
    pub start_date: u64,
    pub end_date: u64,
    pub alert_thresholds: Vec<AlertThreshold>,
    pub filters: BudgetFilters,
    pub status: BudgetStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetPeriod {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
    Custom { days: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThreshold {
    pub threshold_id: String,
    pub percentage: f32,
    pub notification_type: NotificationType,
    pub recipients: Vec<String>,
    pub triggered: bool,
    pub last_triggered: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    Email,
    Slack,
    Webhook,
    SMS,
    Dashboard,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetFilters {
    pub job_ids: Vec<String>,
    pub resource_types: Vec<String>,
    pub tags: HashMap<String, String>,
    pub regions: Vec<String>,
    pub providers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetStatus {
    Active,
    Exceeded,
    Warning,
    Inactive,
}

#[derive(Debug, Clone)]
pub struct CostReport {
    pub report_id: String,
    pub report_type: ReportType,
    pub time_range: TimeRange,
    pub total_cost: f64,
    pub cost_breakdown: CostBreakdown,
    pub cost_trends: Vec<CostTrend>,
    pub top_cost_drivers: Vec<CostDriver>,
    pub efficiency_metrics: EfficiencyMetrics,
    pub recommendations: Vec<CostRecommendation>,
}

#[derive(Debug, Clone)]
pub enum ReportType {
    Daily,
    Weekly,
    Monthly,
    Custom,
    JobSpecific,
    ResourceSpecific,
}

#[derive(Debug, Clone)]
pub struct TimeRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone)]
pub struct CostBreakdown {
    pub by_resource_type: HashMap<String, f64>,
    pub by_job: HashMap<String, f64>,
    pub by_region: HashMap<String, f64>,
    pub by_provider: HashMap<String, f64>,
    pub by_billing_model: HashMap<String, f64>,
    pub by_tag: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct CostTrend {
    pub timestamp: u64,
    pub cost: f64,
    pub cumulative_cost: f64,
}

#[derive(Debug, Clone)]
pub struct CostDriver {
    pub name: String,
    pub cost: f64,
    pub percentage: f32,
    pub category: String,
}

#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    pub cost_per_hour: f64,
    pub resource_utilization: f64,
    pub idle_cost_percentage: f32,
    pub spot_instance_savings: f64,
    pub efficiency_score: f64,
}

#[derive(Debug, Clone)]
pub struct CostRecommendation {
    pub recommendation_id: String,
    pub title: String,
    pub description: String,
    pub potential_savings: f64,
    pub confidence: f32,
    pub implementation_effort: ImplementationEffort,
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
}

#[derive(Debug, Clone)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone)]
pub enum RecommendationCategory {
    ResourceRightsizing,
    SpotInstances,
    ReservedInstances,
    SchedulingOptimization,
    IdleResourceElimination,
    RegionOptimization,
}

#[derive(Debug, Clone)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

pub struct CostTracker {
    cost_entries: Arc<RwLock<Vec<CostEntry>>>,
    budgets: Arc<RwLock<HashMap<String, Budget>>>,
    billing_rates: Arc<RwLock<BillingRates>>,
    statistics: Arc<RwLock<CostStatistics>>,
    forecasting_model: Arc<RwLock<CostForecastingModel>>,
}

#[derive(Debug, Clone)]
pub struct BillingRates {
    pub cpu_rates: HashMap<String, f64>, // region -> rate per core-hour
    pub memory_rates: HashMap<String, f64>, // region -> rate per GB-hour
    pub gpu_rates: HashMap<String, HashMap<String, f64>>, // region -> gpu_type -> rate per hour
    pub storage_rates: HashMap<String, f64>, // region -> rate per GB-month
    pub network_rates: HashMap<String, f64>, // region -> rate per GB
    pub spot_discounts: HashMap<String, f64>, // region -> discount percentage
}

#[derive(Debug, Default, Clone)]
pub struct CostStatistics {
    pub total_cost: f64,
    pub monthly_cost: f64,
    pub daily_cost: f64,
    pub cost_by_resource_type: HashMap<String, f64>,
    pub cost_by_job: HashMap<String, f64>,
    pub average_cost_per_job: f64,
    pub peak_cost_period: Option<TimeRange>,
    pub cost_growth_rate: f64,
    pub forecasted_monthly_cost: f64,
}

pub struct CostForecastingModel {
    historical_data: VecDeque<CostDataPoint>,
    model_parameters: ForecastingParameters,
    #[allow(dead_code)]
    accuracy_metrics: ForecastingAccuracy,
}

#[derive(Debug, Clone)]
pub struct CostDataPoint {
    pub timestamp: u64,
    pub cost: f64,
    pub job_count: u32,
    pub resource_usage: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct ForecastingParameters {
    pub trend_weight: f64,
    pub seasonal_weight: f64,
    pub noise_filter: f64,
    pub confidence_interval: f64,
}

#[derive(Debug, Clone)]
pub struct ForecastingAccuracy {
    pub mean_absolute_error: f64,
    pub mean_percentage_error: f64,
    pub confidence_score: f64,
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CostTracker {
    pub fn new() -> Self {
        Self {
            cost_entries: Arc::new(RwLock::new(Vec::new())),
            budgets: Arc::new(RwLock::new(HashMap::new())),
            billing_rates: Arc::new(RwLock::new(BillingRates::default())),
            statistics: Arc::new(RwLock::new(CostStatistics::default())),
            forecasting_model: Arc::new(RwLock::new(CostForecastingModel::new())),
        }
    }

    pub fn record_cost(
        &self,
        job_id: String,
        resource_type: String,
        resource_amount: f64,
        duration: Duration,
        billing_model: BillingModel,
        region: String,
        provider: String,
        tags: HashMap<String, String>,
    ) -> Result<String> {
        let entry_id = uuid::Uuid::new_v4().to_string();
        let start_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        // Calculate cost based on billing model and rates
        let cost_per_unit =
            self.calculate_cost_per_unit(&resource_type, &billing_model, &region)?;
        let total_cost =
            self.calculate_total_cost(resource_amount, duration, cost_per_unit, &billing_model)?;

        let entry = CostEntry {
            entry_id: entry_id.clone(),
            job_id,
            resource_type: resource_type.clone(),
            resource_amount,
            cost_per_unit,
            total_cost,
            start_time,
            end_time: Some(start_time + duration.as_secs()),
            duration: Some(duration),
            billing_model,
            tags,
            region,
            provider,
        };

        // Add to cost entries
        {
            let mut entries = self
                .cost_entries
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on cost entries"))?;
            entries.push(entry.clone());
        }

        // Update statistics
        self.update_statistics(&entry)?;

        // Check budget alerts
        self.check_budget_alerts()?;

        // Update forecasting model
        self.update_forecasting_model(&entry)?;

        Ok(entry_id)
    }

    fn calculate_cost_per_unit(
        &self,
        resource_type: &str,
        billing_model: &BillingModel,
        region: &str,
    ) -> Result<f64> {
        let rates = self
            .billing_rates
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on billing rates"))?;

        let base_rate = match resource_type {
            "cpu" => rates.cpu_rates.get(region).unwrap_or(&0.1),
            "memory" => rates.memory_rates.get(region).unwrap_or(&0.05),
            "gpu" => rates
                .gpu_rates
                .get(region)
                .and_then(|gpu_rates| gpu_rates.get("default"))
                .unwrap_or(&1.0),
            "storage" => rates.storage_rates.get(region).unwrap_or(&0.01),
            "network" => rates.network_rates.get(region).unwrap_or(&0.02),
            _ => &0.0,
        };

        let adjusted_rate = match billing_model {
            BillingModel::Spot => {
                let discount = rates.spot_discounts.get(region).unwrap_or(&0.7);
                base_rate * discount
            },
            BillingModel::Reserved => base_rate * 0.6, // 40% discount
            BillingModel::Preemptible => base_rate * 0.3, // 70% discount
            BillingModel::Custom { rate, .. } => *rate,
            _ => *base_rate,
        };

        Ok(adjusted_rate)
    }

    fn calculate_total_cost(
        &self,
        resource_amount: f64,
        duration: Duration,
        cost_per_unit: f64,
        billing_model: &BillingModel,
    ) -> Result<f64> {
        let hours = duration.as_secs_f64() / 3600.0;

        let total_cost = match billing_model {
            BillingModel::PayPerUse | BillingModel::Spot | BillingModel::Preemptible => {
                resource_amount * cost_per_unit * hours
            },
            BillingModel::Subscription => {
                // For subscription, cost is fixed regardless of usage
                cost_per_unit * hours
            },
            BillingModel::Reserved => {
                // Reserved instances have upfront cost amortized over time
                resource_amount * cost_per_unit * hours
            },
            BillingModel::Custom { rate, .. } => resource_amount * rate * hours,
        };

        Ok(total_cost)
    }

    fn update_statistics(&self, entry: &CostEntry) -> Result<()> {
        let mut stats = self
            .statistics
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on statistics"))?;

        stats.total_cost += entry.total_cost;

        // Update cost by resource type
        *stats.cost_by_resource_type.entry(entry.resource_type.clone()).or_insert(0.0) +=
            entry.total_cost;

        // Update cost by job
        *stats.cost_by_job.entry(entry.job_id.clone()).or_insert(0.0) += entry.total_cost;

        // Calculate average cost per job
        if !stats.cost_by_job.is_empty() {
            stats.average_cost_per_job = stats.total_cost / stats.cost_by_job.len() as f64;
        }

        // Update daily and monthly costs (simplified)
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        let day_start = current_time - (current_time % 86400);
        let month_start = current_time - (current_time % (86400 * 30));

        if entry.start_time >= day_start {
            stats.daily_cost += entry.total_cost;
        }

        if entry.start_time >= month_start {
            stats.monthly_cost += entry.total_cost;
        }

        Ok(())
    }

    pub fn create_budget(&self, budget: Budget) -> Result<String> {
        let mut budgets = self
            .budgets
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on budgets"))?;

        budgets.insert(budget.budget_id.clone(), budget.clone());
        Ok(budget.budget_id)
    }

    pub fn update_budget(&self, budget_id: &str, spent_amount: f64) -> Result<()> {
        let mut budgets = self
            .budgets
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on budgets"))?;

        if let Some(budget) = budgets.get_mut(budget_id) {
            budget.spent_amount += spent_amount;
            budget.remaining_amount = budget.total_amount - budget.spent_amount;

            // Update status based on spending
            let usage_percentage = (budget.spent_amount / budget.total_amount) * 100.0;
            budget.status = if usage_percentage >= 100.0 {
                BudgetStatus::Exceeded
            } else if usage_percentage >= 80.0 {
                BudgetStatus::Warning
            } else {
                BudgetStatus::Active
            };
        } else {
            return Err(anyhow::anyhow!("Budget not found: {}", budget_id));
        }

        Ok(())
    }

    fn check_budget_alerts(&self) -> Result<()> {
        let mut budgets = self
            .budgets
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on budgets"))?;

        let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        for budget in budgets.values_mut() {
            let usage_percentage = (budget.spent_amount / budget.total_amount) * 100.0;

            for threshold in &mut budget.alert_thresholds {
                if usage_percentage >= threshold.percentage.into() && !threshold.triggered {
                    threshold.triggered = true;
                    threshold.last_triggered = Some(current_time);

                    // In a real implementation, would send notifications here
                    println!(
                        "Budget alert: {} has reached {}% of budget",
                        budget.name, threshold.percentage
                    );
                }
            }
        }

        Ok(())
    }

    pub fn generate_cost_report(
        &self,
        report_type: ReportType,
        time_range: TimeRange,
    ) -> Result<CostReport> {
        let entries = self
            .cost_entries
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on cost entries"))?;

        // Filter entries by time range
        let filtered_entries: Vec<_> = entries
            .iter()
            .filter(|entry| {
                entry.start_time >= time_range.start && entry.start_time <= time_range.end
            })
            .collect();

        let total_cost: f64 = filtered_entries.iter().map(|e| e.total_cost).sum();

        // Generate cost breakdown
        let cost_breakdown = self.generate_cost_breakdown(&filtered_entries);

        // Generate cost trends
        let cost_trends = self.generate_cost_trends(&filtered_entries, &time_range);

        // Generate top cost drivers
        let top_cost_drivers = self.generate_top_cost_drivers(&filtered_entries);

        // Calculate efficiency metrics
        let efficiency_metrics = self.calculate_efficiency_metrics(&filtered_entries);

        // Generate recommendations
        let recommendations =
            self.generate_cost_recommendations(&filtered_entries, &efficiency_metrics);

        Ok(CostReport {
            report_id: uuid::Uuid::new_v4().to_string(),
            report_type,
            time_range,
            total_cost,
            cost_breakdown,
            cost_trends,
            top_cost_drivers,
            efficiency_metrics,
            recommendations,
        })
    }

    fn generate_cost_breakdown(&self, entries: &[&CostEntry]) -> CostBreakdown {
        let mut by_resource_type = HashMap::new();
        let mut by_job = HashMap::new();
        let mut by_region = HashMap::new();
        let mut by_provider = HashMap::new();
        let mut by_billing_model = HashMap::new();
        let mut by_tag = HashMap::new();

        for entry in entries {
            *by_resource_type.entry(entry.resource_type.clone()).or_insert(0.0) += entry.total_cost;
            *by_job.entry(entry.job_id.clone()).or_insert(0.0) += entry.total_cost;
            *by_region.entry(entry.region.clone()).or_insert(0.0) += entry.total_cost;
            *by_provider.entry(entry.provider.clone()).or_insert(0.0) += entry.total_cost;

            let billing_key = format!("{:?}", entry.billing_model);
            *by_billing_model.entry(billing_key).or_insert(0.0) += entry.total_cost;

            for (tag_key, tag_value) in &entry.tags {
                let tag_entry = format!("{}:{}", tag_key, tag_value);
                *by_tag.entry(tag_entry).or_insert(0.0) += entry.total_cost;
            }
        }

        CostBreakdown {
            by_resource_type,
            by_job,
            by_region,
            by_provider,
            by_billing_model,
            by_tag,
        }
    }

    fn generate_cost_trends(
        &self,
        entries: &[&CostEntry],
        _time_range: &TimeRange,
    ) -> Vec<CostTrend> {
        let mut trends = Vec::new();
        let mut cumulative_cost = 0.0;

        // Group entries by day
        let mut daily_costs = BTreeMap::new();
        for entry in entries {
            let day = entry.start_time - (entry.start_time % 86400);
            *daily_costs.entry(day).or_insert(0.0) += entry.total_cost;
        }

        for (timestamp, cost) in daily_costs {
            cumulative_cost += cost;
            trends.push(CostTrend {
                timestamp,
                cost,
                cumulative_cost,
            });
        }

        trends
    }

    fn generate_top_cost_drivers(&self, entries: &[&CostEntry]) -> Vec<CostDriver> {
        let mut drivers = HashMap::new();
        let total_cost: f64 = entries.iter().map(|e| e.total_cost).sum();

        for entry in entries {
            let key = format!("{}-{}", entry.resource_type, entry.job_id);
            *drivers.entry(key.clone()).or_insert(0.0) += entry.total_cost;
        }

        let mut sorted_drivers: Vec<_> = drivers
            .into_iter()
            .map(|(name, cost)| CostDriver {
                name: name.clone(),
                cost,
                percentage: if total_cost > 0.0 { (cost / total_cost * 100.0) as f32 } else { 0.0 },
                category: "Resource Usage".to_string(),
            })
            .collect();

        sorted_drivers.sort_by(|a, b| b.cost.partial_cmp(&a.cost).unwrap());
        sorted_drivers.truncate(10); // Top 10 cost drivers

        sorted_drivers
    }

    fn calculate_efficiency_metrics(&self, entries: &[&CostEntry]) -> EfficiencyMetrics {
        let total_cost: f64 = entries.iter().map(|e| e.total_cost).sum();
        let total_hours: f64 = entries
            .iter()
            .filter_map(|e| e.duration)
            .map(|d| d.as_secs_f64() / 3600.0)
            .sum();

        let cost_per_hour = if total_hours > 0.0 { total_cost / total_hours } else { 0.0 };

        // Simplified efficiency calculations
        let resource_utilization = 0.75; // Would be calculated from actual usage data
        let idle_cost_percentage = 15.0; // Would be calculated from idle resources
        let spot_instance_savings = entries.iter()
            .filter(|e| matches!(e.billing_model, BillingModel::Spot))
            .map(|e| e.total_cost * 0.3) // Estimated 30% savings
            .sum();

        let efficiency_score = (resource_utilization * 100.0 - idle_cost_percentage as f64) / 100.0;

        EfficiencyMetrics {
            cost_per_hour,
            resource_utilization,
            idle_cost_percentage,
            spot_instance_savings,
            efficiency_score,
        }
    }

    fn generate_cost_recommendations(
        &self,
        entries: &[&CostEntry],
        efficiency_metrics: &EfficiencyMetrics,
    ) -> Vec<CostRecommendation> {
        let mut recommendations = Vec::new();

        // Recommend spot instances if not used
        let spot_usage =
            entries.iter().filter(|e| matches!(e.billing_model, BillingModel::Spot)).count() as f32
                / entries.len() as f32;

        if spot_usage < 0.3 {
            recommendations.push(CostRecommendation {
                recommendation_id: uuid::Uuid::new_v4().to_string(),
                title: "Increase Spot Instance Usage".to_string(),
                description: "Consider using more spot instances for non-critical workloads to reduce costs by up to 70%".to_string(),
                potential_savings: efficiency_metrics.spot_instance_savings,
                confidence: 0.8,
                implementation_effort: ImplementationEffort::Medium,
                category: RecommendationCategory::SpotInstances,
                priority: RecommendationPriority::High,
            });
        }

        // Recommend resource rightsizing if utilization is low
        if efficiency_metrics.resource_utilization < 0.6 {
            recommendations.push(CostRecommendation {
                recommendation_id: uuid::Uuid::new_v4().to_string(),
                title: "Resource Rightsizing".to_string(),
                description: "Current resource utilization is low. Consider rightsizing instances to match actual workload requirements.".to_string(),
                potential_savings: efficiency_metrics.cost_per_hour * 24.0 * 30.0 * 0.3, // 30% of monthly cost
                confidence: 0.7,
                implementation_effort: ImplementationEffort::High,
                category: RecommendationCategory::ResourceRightsizing,
                priority: RecommendationPriority::Medium,
            });
        }

        // Recommend idle resource elimination
        if efficiency_metrics.idle_cost_percentage > 20.0 {
            recommendations.push(CostRecommendation {
                recommendation_id: uuid::Uuid::new_v4().to_string(),
                title: "Eliminate Idle Resources".to_string(),
                description: "High percentage of idle resources detected. Implement automatic shutdown policies for unused resources.".to_string(),
                potential_savings: efficiency_metrics.cost_per_hour * 24.0 * 30.0 * (efficiency_metrics.idle_cost_percentage / 100.0) as f64,
                confidence: 0.9,
                implementation_effort: ImplementationEffort::Low,
                category: RecommendationCategory::IdleResourceElimination,
                priority: RecommendationPriority::Critical,
            });
        }

        // Always provide a general cost optimization recommendation
        if recommendations.is_empty() {
            recommendations.push(CostRecommendation {
                recommendation_id: uuid::Uuid::new_v4().to_string(),
                title: "Regular Cost Review".to_string(),
                description: "Current cost metrics look healthy. Continue regular cost reviews and consider implementing cost allocation tags for better tracking.".to_string(),
                potential_savings: 0.0,
                confidence: 0.5,
                implementation_effort: ImplementationEffort::Low,
                category: RecommendationCategory::SchedulingOptimization,
                priority: RecommendationPriority::Low,
            });
        }

        recommendations
    }

    fn update_forecasting_model(&self, entry: &CostEntry) -> Result<()> {
        let mut model = self
            .forecasting_model
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on forecasting model"))?;

        let data_point = CostDataPoint {
            timestamp: entry.start_time,
            cost: entry.total_cost,
            job_count: 1,
            resource_usage: {
                let mut usage = HashMap::new();
                usage.insert(entry.resource_type.clone(), entry.resource_amount);
                usage
            },
        };

        model.add_data_point(data_point);
        Ok(())
    }

    pub fn forecast_costs(&self, days_ahead: u32) -> Result<Vec<CostDataPoint>> {
        let model = self
            .forecasting_model
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on forecasting model"))?;

        model.forecast(days_ahead)
    }

    pub fn get_statistics(&self) -> Result<CostStatistics> {
        let stats = self
            .statistics
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on statistics"))?;
        Ok((*stats).clone())
    }

    pub fn get_budget_status(&self, budget_id: &str) -> Result<Budget> {
        let budgets = self
            .budgets
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on budgets"))?;

        budgets
            .get(budget_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Budget not found: {}", budget_id))
    }

    pub fn update_billing_rates(&self, rates: BillingRates) -> Result<()> {
        let mut billing_rates = self
            .billing_rates
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on billing rates"))?;
        *billing_rates = rates;
        Ok(())
    }
}

impl BillingRates {
    fn default() -> Self {
        let mut cpu_rates = HashMap::new();
        cpu_rates.insert("us-east-1".to_string(), 0.1);
        cpu_rates.insert("us-west-2".to_string(), 0.12);
        cpu_rates.insert("eu-west-1".to_string(), 0.11);

        let mut memory_rates = HashMap::new();
        memory_rates.insert("us-east-1".to_string(), 0.05);
        memory_rates.insert("us-west-2".to_string(), 0.06);
        memory_rates.insert("eu-west-1".to_string(), 0.055);

        let mut gpu_rates = HashMap::new();
        let mut us_east_gpu = HashMap::new();
        us_east_gpu.insert("default".to_string(), 1.0);
        us_east_gpu.insert("v100".to_string(), 2.5);
        us_east_gpu.insert("a100".to_string(), 4.0);
        gpu_rates.insert("us-east-1".to_string(), us_east_gpu);

        let mut storage_rates = HashMap::new();
        storage_rates.insert("us-east-1".to_string(), 0.01);
        storage_rates.insert("us-west-2".to_string(), 0.012);

        let mut network_rates = HashMap::new();
        network_rates.insert("us-east-1".to_string(), 0.02);
        network_rates.insert("us-west-2".to_string(), 0.025);

        let mut spot_discounts = HashMap::new();
        spot_discounts.insert("us-east-1".to_string(), 0.3);
        spot_discounts.insert("us-west-2".to_string(), 0.35);

        Self {
            cpu_rates,
            memory_rates,
            gpu_rates,
            storage_rates,
            network_rates,
            spot_discounts,
        }
    }
}

impl CostForecastingModel {
    fn new() -> Self {
        Self {
            historical_data: VecDeque::with_capacity(1000),
            model_parameters: ForecastingParameters {
                trend_weight: 0.3,
                seasonal_weight: 0.2,
                noise_filter: 0.1,
                confidence_interval: 0.95,
            },
            accuracy_metrics: ForecastingAccuracy {
                mean_absolute_error: 0.0,
                mean_percentage_error: 0.0,
                confidence_score: 0.0,
            },
        }
    }

    fn add_data_point(&mut self, data_point: CostDataPoint) {
        if self.historical_data.len() >= 1000 {
            self.historical_data.pop_front();
        }
        self.historical_data.push_back(data_point);
    }

    fn forecast(&self, days_ahead: u32) -> Result<Vec<CostDataPoint>> {
        if self.historical_data.len() < 7 {
            return Err(anyhow::anyhow!(
                "Insufficient historical data for forecasting"
            ));
        }

        let mut forecasted_points = Vec::new();
        let last_point = self.historical_data.back().unwrap();

        // Simple linear trend forecasting (in practice, would use more sophisticated models)
        let recent_costs: Vec<f64> =
            self.historical_data.iter().rev().take(7).map(|p| p.cost).collect();

        let average_daily_cost = recent_costs.iter().sum::<f64>() / recent_costs.len() as f64;

        // Calculate trend
        let trend = if recent_costs.len() > 1 {
            (recent_costs[0] - recent_costs[recent_costs.len() - 1])
                / (recent_costs.len() - 1) as f64
        } else {
            0.0
        };

        for day in 1..=days_ahead {
            let forecasted_cost =
                average_daily_cost + (trend * day as f64 * self.model_parameters.trend_weight);

            forecasted_points.push(CostDataPoint {
                timestamp: last_point.timestamp + (day as u64 * 86400),
                cost: forecasted_cost.max(0.0),
                job_count: last_point.job_count,
                resource_usage: last_point.resource_usage.clone(),
            });
        }

        Ok(forecasted_points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_tracker_creation() {
        let tracker = CostTracker::new();
        let stats = tracker.get_statistics().unwrap();
        assert_eq!(stats.total_cost, 0.0);
    }

    #[test]
    fn test_cost_recording() {
        let tracker = CostTracker::new();

        let entry_id = tracker
            .record_cost(
                "job-1".to_string(),
                "cpu".to_string(),
                4.0,
                Duration::from_secs(3600),
                BillingModel::PayPerUse,
                "us-east-1".to_string(),
                "aws".to_string(),
                HashMap::new(),
            )
            .unwrap();

        assert!(!entry_id.is_empty());

        let stats = tracker.get_statistics().unwrap();
        assert!(stats.total_cost > 0.0);
        assert_eq!(stats.cost_by_job.get("job-1"), Some(&stats.total_cost));
    }

    #[test]
    fn test_budget_creation() {
        let tracker = CostTracker::new();

        let budget = Budget {
            budget_id: "budget-1".to_string(),
            name: "Monthly Budget".to_string(),
            total_amount: 1000.0,
            spent_amount: 0.0,
            remaining_amount: 1000.0,
            period: BudgetPeriod::Monthly,
            start_date: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            end_date: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + 2592000, // 30 days
            alert_thresholds: vec![AlertThreshold {
                threshold_id: "alert-1".to_string(),
                percentage: 80.0,
                notification_type: NotificationType::Email,
                recipients: vec!["admin@example.com".to_string()],
                triggered: false,
                last_triggered: None,
            }],
            filters: BudgetFilters {
                job_ids: vec![],
                resource_types: vec![],
                tags: HashMap::new(),
                regions: vec![],
                providers: vec![],
            },
            status: BudgetStatus::Active,
        };

        let budget_id = tracker.create_budget(budget).unwrap();
        assert_eq!(budget_id, "budget-1");

        let retrieved_budget = tracker.get_budget_status(&budget_id).unwrap();
        assert_eq!(retrieved_budget.name, "Monthly Budget");
    }

    #[test]
    fn test_cost_report_generation() {
        let tracker = CostTracker::new();

        // Record some costs
        tracker
            .record_cost(
                "job-1".to_string(),
                "cpu".to_string(),
                4.0,
                Duration::from_secs(3600),
                BillingModel::PayPerUse,
                "us-east-1".to_string(),
                "aws".to_string(),
                HashMap::new(),
            )
            .unwrap();

        tracker
            .record_cost(
                "job-2".to_string(),
                "gpu".to_string(),
                1.0,
                Duration::from_secs(1800),
                BillingModel::Spot,
                "us-west-2".to_string(),
                "aws".to_string(),
                HashMap::new(),
            )
            .unwrap();

        let time_range = TimeRange {
            start: 0,
            end: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + 86400,
        };

        let report = tracker.generate_cost_report(ReportType::Daily, time_range).unwrap();

        assert!(report.total_cost > 0.0);
        assert!(!report.cost_breakdown.by_resource_type.is_empty());
        assert!(!report.cost_breakdown.by_job.is_empty());
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_spot_instance_cost_calculation() {
        let tracker = CostTracker::new();

        // Test spot instance pricing
        let entry_id = tracker
            .record_cost(
                "job-spot".to_string(),
                "cpu".to_string(),
                4.0,
                Duration::from_secs(3600),
                BillingModel::Spot,
                "us-east-1".to_string(),
                "aws".to_string(),
                HashMap::new(),
            )
            .unwrap();

        let spot_cost = {
            let entries = tracker.cost_entries.read().unwrap();
            entries.iter().find(|e| e.entry_id == entry_id).unwrap().total_cost
        };

        // Record regular instance for comparison
        let regular_entry_id = tracker
            .record_cost(
                "job-regular".to_string(),
                "cpu".to_string(),
                4.0,
                Duration::from_secs(3600),
                BillingModel::PayPerUse,
                "us-east-1".to_string(),
                "aws".to_string(),
                HashMap::new(),
            )
            .unwrap();

        let regular_cost = {
            let entries = tracker.cost_entries.read().unwrap();
            entries.iter().find(|e| e.entry_id == regular_entry_id).unwrap().total_cost
        };

        // Spot instance should be significantly cheaper
        assert!(spot_cost < regular_cost * 0.5);
    }

    #[test]
    fn test_cost_forecasting() {
        let tracker = CostTracker::new();

        // Add some historical data
        for i in 0..10 {
            tracker
                .record_cost(
                    format!("job-{}", i),
                    "cpu".to_string(),
                    4.0,
                    Duration::from_secs(3600),
                    BillingModel::PayPerUse,
                    "us-east-1".to_string(),
                    "aws".to_string(),
                    HashMap::new(),
                )
                .unwrap();
        }

        let forecast = tracker.forecast_costs(7).unwrap();
        assert_eq!(forecast.len(), 7);

        for point in forecast {
            assert!(point.cost >= 0.0);
        }
    }
}
