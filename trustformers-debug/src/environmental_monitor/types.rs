//! Common types and enums for environmental monitoring

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of measurement being performed
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MeasurementType {
    Training,
    Inference,
    DataPreprocessing,
    ModelEvaluation,
    Development,
}

/// Device type for energy monitoring
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum DeviceType {
    GPU,
    CPU,
    Memory,
    Storage,
    Network,
    Cooling,
    Other(String),
}

/// Power measurement method
#[derive(Debug, Clone)]
pub enum PowerMeasurementMethod {
    NVML,       // NVIDIA Management Library
    RAPL,       // Running Average Power Limit
    PowerMeter, // External power meter
    Estimated,  // Model-based estimation
}

/// Types of efficiency improvements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EfficiencyType {
    ModelArchitecture,
    TrainingOptimization,
    HardwareUtilization,
    SchedulingOptimization,
    CoolingOptimization,
    RegionalOptimization,
    BatchSizeOptimization,
    PrecisionOptimization,
}

/// Implementation effort required
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Types of energy waste
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WasteType {
    IdleResources,
    InefficientAlgorithm,
    PoorScheduling,
    OverProvisioning,
    ThermalThrottling,
    MemoryThrashing,
}

/// Goal types for sustainability tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GoalType {
    CarbonReduction,
    EnergyEfficiency,
    RenewableEnergy,
    WasteReduction,
    CustomGoal(String),
}

/// Impact categories for best practices
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ImpactCategory {
    High,
    Medium,
    Low,
}

/// Trend direction for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Fluctuating,
}

/// Report frequency options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFrequency {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annual,
}

/// Visualization types for reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationType {
    LineChart,
    BarChart,
    PieChart,
    Heatmap,
    Gauge,
    Table,
}

/// Workload priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkloadPriority {
    Critical,
    High,
    Medium,
    Low,
    Background,
}

/// Environmental report types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReportType {
    Summary,
    Detailed,
    Technical,
    Executive,
    Compliance,
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecommendationCategory {
    Energy,
    Carbon,
    Cost,
    Performance,
    Sustainability,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Schedule types for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleType {
    LowCarbon,
    LowCost,
    Balanced,
    HighPerformance,
}

/// Carbon measurement data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonMeasurement {
    pub timestamp: std::time::SystemTime,
    pub energy_consumed_kwh: f64,
    pub carbon_intensity_gco2_kwh: f64,
    pub co2_emissions_kg: f64,
    pub scope2_emissions_kg: f64,
    pub scope3_emissions_kg: Option<f64>,
    pub region: String,
    pub measurement_type: MeasurementType,
}

/// Total carbon emissions breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonEmissions {
    pub total_co2_kg: f64,
    pub scope1_emissions_kg: f64, // Direct emissions
    pub scope2_emissions_kg: f64, // Electricity
    pub scope3_emissions_kg: f64, // Infrastructure, manufacturing
    pub training_emissions_kg: f64,
    pub inference_emissions_kg: f64,
    pub equivalent_metrics: EquivalentMetrics,
}

/// Equivalent metrics for carbon impact visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalentMetrics {
    pub car_miles_equivalent: f64,
    pub tree_months_to_offset: f64,
    pub coal_pounds_equivalent: f64,
    pub households_daily_energy: f64,
}

/// Energy measurement data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyMeasurement {
    pub timestamp: std::time::SystemTime,
    pub device_id: String,
    pub power_watts: f64,
    pub energy_kwh: f64,
    pub utilization: f64,
    pub temperature: Option<f64>,
    pub efficiency_ratio: f64,
}

/// Energy efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyEfficiencyMetrics {
    pub operations_per_kwh: f64,
    pub flops_per_watt: f64,
    pub model_energy_efficiency: f64, // Operations per joule
    pub training_energy_efficiency: f64,
    pub inference_energy_efficiency: f64,
    pub comparative_efficiency: ComparativeEfficiency,
}

/// Comparative efficiency benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeEfficiency {
    pub vs_cpu_only: f64,
    pub vs_previous_generation: f64,
    pub vs_cloud_baseline: f64,
    pub efficiency_percentile: f64, // Where this system ranks
}

/// Efficiency improvement opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyOpportunity {
    pub opportunity_type: EfficiencyType,
    pub description: String,
    pub potential_energy_savings_kwh: f64,
    pub potential_cost_savings_usd: f64,
    pub potential_carbon_reduction_kg: f64,
    pub implementation_effort: ImplementationEffort,
    pub confidence: f64,
    pub recommendation: String,
}

/// Energy waste measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasteMeasurement {
    pub timestamp: std::time::SystemTime,
    pub waste_type: WasteType,
    pub wasted_energy_kwh: f64,
    pub wasted_cost_usd: f64,
    pub efficiency_lost_percentage: f64,
    pub description: String,
}

/// Optimal scheduling recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalSchedule {
    pub schedule_type: ScheduleType,
    pub start_time: std::time::SystemTime,
    pub duration_hours: f64,
    pub projected_savings: ProjectedSavings,
    pub carbon_intensity_forecast: Vec<f64>,
    pub confidence: f64,
}

/// Projected savings from optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectedSavings {
    pub energy_savings_kwh: f64,
    pub cost_savings_usd: f64,
    pub carbon_reduction_kg: f64,
    pub efficiency_improvement_percent: f64,
}

/// Model energy profile for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEnergyProfile {
    pub model_name: String,
    pub parameter_count: u64,
    pub training_energy_kwh: f64,
    pub inference_energy_per_sample: f64,
    pub memory_requirements_gb: f64,
    pub compute_requirements_flops: u64,
    pub efficiency_score: f64,
}

/// Model optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOptimizationRecommendation {
    pub recommendation_type: String,
    pub description: String,
    pub potential_savings: ProjectedSavings,
    pub implementation_complexity: ImplementationEffort,
}

/// Sustainability goal definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SustainabilityGoal {
    pub goal_type: GoalType,
    pub target_value: f64,
    pub current_value: f64,
    pub target_date: std::time::SystemTime,
    pub description: String,
}

/// Progress measurement for sustainability goals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressMeasurement {
    pub timestamp: std::time::SystemTime,
    pub goal_type: GoalType,
    pub current_value: f64,
    pub progress_percentage: f64,
    pub trend: TrendDirection,
    pub period: String,
}

/// Sustainability best practice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPractice {
    pub title: String,
    pub description: String,
    pub impact_category: ImpactCategory,
    pub implementation_effort: ImplementationEffort,
    pub estimated_savings: Option<ProjectedSavings>,
}

/// Environmental dashboard metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalDashboardMetrics {
    pub total_energy_consumed_kwh: f64,
    pub total_co2_emissions_kg: f64,
    pub current_power_usage_watts: f64,
    pub energy_efficiency_score: f64,
    pub carbon_intensity_gco2_kwh: f64,
    pub cost_per_hour_usd: f64,
    pub trend: TrendDirection,
}

/// Session impact information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub start_time: std::time::SystemTime,
    pub duration_hours: f64,
    pub workload_description: String,
    pub region: String,
    pub session_type: MeasurementType,
    pub estimated_energy_kwh: f64,
}

/// Session impact report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionImpactReport {
    pub session_info: SessionInfo,
    pub carbon_emissions: CarbonEmissions,
    pub energy_consumption: f64,
    pub cost_usd: f64,
    pub efficiency_metrics: EnergyEfficiencyMetrics,
    pub recommendations: Vec<String>,
    pub energy_measurement: EnergyMeasurement,
    pub carbon_measurement: CarbonMeasurement,
    pub efficiency_analysis: SessionEfficiencyAnalysis,
    pub cost_analysis: CostAnalysis,
}

/// Session efficiency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEfficiencyAnalysis {
    pub efficiency_score: f64,
    pub waste_percentage: f64,
    pub optimization_opportunities: Vec<EfficiencyOpportunity>,
    pub comparative_analysis: ComparativeEfficiency,
}

/// Cost analysis breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysis {
    pub energy_cost_usd: f64,
    pub carbon_cost_usd: Option<f64>,
    pub infrastructure_cost_usd: f64,
    pub total_cost_usd: f64,
    pub cost_per_operation: f64,
}

/// Real-time environmental metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeEnvironmentalMetrics {
    pub timestamp: std::time::SystemTime,
    pub current_power_watts: f64,
    pub energy_consumed_kwh: f64,
    pub co2_emissions_kg: f64,
    pub efficiency_ratio: f64,
    pub temperature_celsius: Option<f64>,
}

/// Workload description for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadDescription {
    pub workload_name: String,
    pub workload_type: String,
    pub priority: WorkloadPriority,
    pub estimated_duration_hours: f64,
    pub resource_requirements: HashMap<String, f64>,
    pub estimated_energy_kwh: f64,
}

/// Environmental report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalReport {
    pub report_id: String,
    pub report_type: ReportType,
    pub generated_at: std::time::SystemTime,
    pub period_start: std::time::SystemTime,
    pub period_end: std::time::SystemTime,
    pub summary: String,
    pub metrics: EnvironmentalDashboardMetrics,
    pub detailed_analysis: String,
    pub recommendations: Vec<SustainabilityRecommendation>,
    pub charts: Vec<ChartData>,
}

/// Chart data for visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    pub chart_type: VisualizationType,
    pub title: String,
    pub data_points: Vec<(String, f64)>,
    pub labels: Vec<String>,
}

/// Sustainability recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SustainabilityRecommendation {
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub potential_impact: String,
    pub implementation_steps: Vec<String>,
}
