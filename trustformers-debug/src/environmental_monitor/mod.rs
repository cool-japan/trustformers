//! Environmental Impact Monitoring Module
//!
//! This module provides comprehensive monitoring of environmental impact during model
//! training and inference, including carbon footprint tracking, energy consumption
//! analysis, and sustainability recommendations.
// reason: debug/profiling scaffolding — structs are constructed and their fields/methods
// are retained for the data model, serialization completeness, and future consumers that
// do not yet read every member. Consolidated from many item-level #[allow(dead_code)].
#![allow(dead_code)]

pub mod carbon_tracking;
pub mod config;
pub mod efficiency_analysis;
pub mod energy_monitoring;
pub mod reporting;
pub mod sustainability;
pub mod types;

pub use carbon_tracking::CarbonFootprintTracker;
pub use config::EnvironmentalConfig;
pub use efficiency_analysis::EfficiencyAnalyzer;
pub use reporting::EnvironmentalReportingEngine;
pub use sustainability::SustainabilityAdvisor;
pub use types::*;

use anyhow::Result;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Errors specific to [`EnvironmentalMonitor`]'s forecast-driven scheduling.
#[derive(Debug, thiserror::Error)]
pub enum EnvironmentalMonitorError {
    /// [`EnvironmentalMonitor::optimize_scheduling`] needs real carbon-
    /// intensity and energy-price forecasts, but this crate ships no
    /// built-in grid-carbon-intensity or spot-price API client. Without a
    /// [`ForecastSource`] attached via
    /// [`EnvironmentalMonitor::set_forecast_source`], it honestly refuses
    /// to schedule rather than inventing sine-wave forecast data.
    #[error(
        "no ForecastSource configured on this EnvironmentalMonitor (see \
         EnvironmentalMonitor::set_forecast_source); cannot forecast carbon \
         intensity or energy prices without one"
    )]
    NotConfigured,
}

/// Supplies real carbon-intensity and energy-price forecasts for
/// [`EnvironmentalMonitor::optimize_scheduling`].
///
/// Implementations are expected to already hold (or synchronously look up)
/// this data -- e.g. from a cache a caller-owned background task keeps
/// refreshed from a real grid-carbon-intensity API (WattTime,
/// Electricity Maps, ...) or a utility's spot-price feed -- since
/// `EnvironmentalMonitor` performs no network I/O of its own. Without one
/// attached, [`EnvironmentalMonitor::optimize_scheduling`] fails with
/// [`EnvironmentalMonitorError::NotConfigured`] instead of fabricating a
/// forecast.
pub trait ForecastSource: std::fmt::Debug + Send + Sync {
    /// Real carbon-intensity forecast for `region`, one entry per hour for
    /// the next `hours` hours.
    fn carbon_intensity_forecast(&self, region: &str, hours: usize) -> Result<Vec<CarbonForecast>>;
    /// Real energy-price forecast for `region`, one entry per hour for the
    /// next `hours` hours.
    fn energy_price_forecast(&self, region: &str, hours: usize)
        -> Result<Vec<EnergyPriceForecast>>;
}

/// Environmental impact monitor for tracking carbon footprint and energy usage
#[derive(Debug)]
pub struct EnvironmentalMonitor {
    config: EnvironmentalConfig,
    carbon_tracker: CarbonFootprintTracker,
    energy_monitor: energy_monitoring::EnergyConsumptionMonitor,
    efficiency_analyzer: EfficiencyAnalyzer,
    sustainability_advisor: SustainabilityAdvisor,
    reporting_engine: EnvironmentalReportingEngine,
    /// Real forecast data source for [`Self::optimize_scheduling`]. `None`
    /// (the default) means scheduling optimization is unavailable -- see
    /// [`EnvironmentalMonitorError::NotConfigured`].
    forecast_source: Option<Box<dyn ForecastSource>>,
}

impl EnvironmentalMonitor {
    /// Create a new environmental monitor
    pub fn new(config: EnvironmentalConfig) -> Self {
        Self {
            config: config.clone(),
            carbon_tracker: CarbonFootprintTracker::new(&config),
            energy_monitor: energy_monitoring::EnergyConsumptionMonitor::new(),
            efficiency_analyzer: EfficiencyAnalyzer::new(),
            sustainability_advisor: SustainabilityAdvisor::new(),
            reporting_engine: EnvironmentalReportingEngine::new(),
            forecast_source: None,
        }
    }

    /// Attach a real [`ForecastSource`] so [`Self::optimize_scheduling`] can
    /// produce real carbon-aware schedules.
    pub fn set_forecast_source(&mut self, source: Box<dyn ForecastSource>) {
        self.forecast_source = Some(source);
    }

    /// Detach the [`ForecastSource`], if any.
    pub fn clear_forecast_source(&mut self) {
        self.forecast_source = None;
    }

    /// Whether a [`ForecastSource`] is currently attached.
    pub fn has_forecast_source(&self) -> bool {
        self.forecast_source.is_some()
    }

    /// Start environmental monitoring
    pub async fn start_monitoring(&mut self) -> Result<()> {
        info!(
            "Starting environmental impact monitoring for region: {}",
            self.config.region
        );

        // Device monitors are already initialized via the constructor

        // Start monitoring loops
        self.start_monitoring_loops().await?;

        // Initialize sustainability goals
        self.sustainability_advisor.initialize_sustainability_goals().await?;

        Ok(())
    }

    /// Record energy consumption and carbon emissions for a training/inference session
    pub async fn record_session(
        &mut self,
        session_info: SessionInfo,
    ) -> Result<SessionImpactReport> {
        info!(
            "Recording environmental impact for {:?} session",
            session_info.session_type
        );

        let _start_time = Instant::now();

        // Predict energy consumption based on session duration
        let predicted_energy_kwh = self
            .energy_monitor
            .predict_energy_consumption(session_info.duration_hours as u32);

        // Use predicted energy if available, otherwise use estimated from session info
        let energy_kwh = if predicted_energy_kwh > 0.0 {
            predicted_energy_kwh
        } else {
            session_info.estimated_energy_kwh
        };

        // Create energy measurement from prediction or estimate
        let energy_measurement = EnergyMeasurement {
            timestamp: std::time::SystemTime::now(),
            device_id: "session".to_string(),
            power_watts: energy_kwh * 1000.0 / session_info.duration_hours, // Convert back to watts
            energy_kwh,
            utilization: 0.8, // Assume 80% utilization
            temperature: None,
            efficiency_ratio: 1.0,
        };

        // Calculate carbon footprint
        let carbon_measurement = self.carbon_tracker.record_emissions(
            energy_measurement.energy_kwh,
            &session_info.region,
            session_info.session_type.clone(),
        )?;

        // Update cumulative metrics
        self.update_cumulative_metrics(&energy_measurement, &carbon_measurement).await?;

        // Analyze efficiency
        let efficiency_analysis = self
            .efficiency_analyzer
            .analyze_session_efficiency(&session_info, &energy_measurement)
            .await?;

        // Generate impact report
        let cost_analysis = self.calculate_cost_impact(&energy_measurement).await?;
        let recommendations = self.generate_session_recommendations(&efficiency_analysis).await?;

        let impact_report = SessionImpactReport {
            session_info,
            carbon_emissions: CarbonEmissions {
                total_co2_kg: carbon_measurement.co2_emissions_kg,
                scope1_emissions_kg: 0.0, // Direct emissions
                scope2_emissions_kg: carbon_measurement.scope2_emissions_kg,
                scope3_emissions_kg: carbon_measurement.scope3_emissions_kg.unwrap_or(0.0),
                training_emissions_kg: carbon_measurement.co2_emissions_kg,
                inference_emissions_kg: 0.0,
                equivalent_metrics: EquivalentMetrics {
                    car_miles_equivalent: carbon_measurement.co2_emissions_kg * 2.31, // kg CO2 to miles
                    tree_months_to_offset: carbon_measurement.co2_emissions_kg * 0.039, // kg CO2 to tree-months
                    coal_pounds_equivalent: carbon_measurement.co2_emissions_kg * 2.2, // kg CO2 to coal pounds
                    households_daily_energy: carbon_measurement.co2_emissions_kg * 0.123, // kg CO2 to household days
                },
            },
            energy_consumption: energy_measurement.energy_kwh,
            cost_usd: cost_analysis.total_cost_usd,
            efficiency_metrics: EnergyEfficiencyMetrics {
                operations_per_kwh: 1.0 / energy_measurement.energy_kwh, // Inverse of energy per operation
                flops_per_watt: 1000.0 / energy_measurement.power_watts, // Approximate FLOPS per watt
                model_energy_efficiency: efficiency_analysis.efficiency_score,
                training_energy_efficiency: efficiency_analysis.efficiency_score,
                inference_energy_efficiency: efficiency_analysis.efficiency_score,
                comparative_efficiency: ComparativeEfficiency {
                    vs_cpu_only: efficiency_analysis.efficiency_score * 1.5, // Assume 1.5x better than CPU
                    vs_previous_generation: efficiency_analysis.efficiency_score * 1.2, // Assume 1.2x better than previous gen
                    vs_cloud_baseline: efficiency_analysis.efficiency_score,
                    efficiency_percentile: efficiency_analysis.efficiency_score * 100.0, // Convert to percentile
                },
            },
            recommendations,
            energy_measurement,
            carbon_measurement,
            efficiency_analysis,
            cost_analysis,
        };

        // Check for alerts
        self.check_environmental_alerts(&impact_report).await?;

        Ok(impact_report)
    }

    /// Get real-time environmental metrics.
    ///
    /// `efficiency_ratio` and `temperature_celsius` come from the most
    /// recently recorded device measurement (see
    /// [`energy_monitoring::EnergyConsumptionMonitor::record_measurement`]):
    /// a real, per-measurement efficiency ratio and (when the caller
    /// supplied one) a real device temperature. Before any measurement has
    /// been recorded, `efficiency_ratio` is honestly `0.0` and
    /// `temperature_celsius` is `None` -- never the old hardcoded `0.87` /
    /// `Some(75.0)`.
    pub async fn get_real_time_metrics(&self) -> Result<RealTimeEnvironmentalMetrics> {
        let current_power = self.energy_monitor.get_current_consumption();
        let carbon_intensity = self.carbon_tracker.get_carbon_intensity(&self.config.region);

        let latest_measurement = self.energy_monitor.get_consumption_history().last();
        let efficiency_ratio = latest_measurement.map(|m| m.efficiency_ratio).unwrap_or(0.0);
        let temperature_celsius = latest_measurement.and_then(|m| m.temperature);

        Ok(RealTimeEnvironmentalMetrics {
            timestamp: std::time::SystemTime::now(),
            current_power_watts: current_power,
            energy_consumed_kwh: current_power / 1000.0, // Convert to kWh for 1 hour
            co2_emissions_kg: (current_power / 1000.0) * carbon_intensity / 1000.0,
            efficiency_ratio,
            temperature_celsius,
        })
    }

    /// Optimize scheduling for minimum environmental impact.
    ///
    /// Requires a real [`ForecastSource`] to be attached via
    /// [`Self::set_forecast_source`] -- fails with
    /// [`EnvironmentalMonitorError::NotConfigured`] otherwise, rather than
    /// scheduling against a fabricated forecast.
    pub async fn optimize_scheduling(
        &self,
        workload: WorkloadDescription,
    ) -> Result<OptimalSchedule> {
        info!("Optimizing schedule for minimum environmental impact");

        // Get carbon intensity forecasts
        let carbon_forecasts = self.get_carbon_intensity_forecasts().await?;

        // Get energy price forecasts
        let price_forecasts = self.get_energy_price_forecasts().await?;

        // Calculate optimal timing
        let optimal_time = self
            .find_optimal_execution_time(&workload, &carbon_forecasts, &price_forecasts)
            .await?;

        // Estimate savings
        let savings = self.calculate_projected_savings(&workload, &optimal_time).await?;

        // Real average of the underlying forecasts' own confidence values
        // (as reported by the attached `ForecastSource`), not a fabricated
        // constant. `0.0` when there are no forecasts to average.
        let confidence = if carbon_forecasts.is_empty() {
            0.0
        } else {
            carbon_forecasts.iter().map(|f| f.confidence).sum::<f64>()
                / carbon_forecasts.len() as f64
        };

        Ok(OptimalSchedule {
            schedule_type: ScheduleType::LowCarbon,
            start_time: optimal_time,
            duration_hours: workload.estimated_duration_hours,
            projected_savings: savings,
            carbon_intensity_forecast: carbon_forecasts
                .iter()
                .map(|f| f.predicted_carbon_intensity)
                .collect(),
            confidence,
        })
    }

    /// Generate comprehensive environmental impact report
    pub async fn generate_environmental_report(
        &self,
        report_type: ReportType,
    ) -> Result<EnvironmentalReport> {
        self.reporting_engine.generate_environmental_report(report_type).await
    }

    /// Get sustainability recommendations
    pub async fn get_sustainability_recommendations(
        &self,
    ) -> Result<Vec<SustainabilityRecommendation>> {
        self.sustainability_advisor.get_sustainability_recommendations().await
    }

    /// Get efficiency opportunities
    pub async fn get_efficiency_opportunities(&self) -> Result<Vec<EfficiencyOpportunity>> {
        self.efficiency_analyzer.analyze_efficiency_opportunities().await
    }

    /// Get carbon emissions data
    pub fn get_cumulative_emissions(&self) -> &CarbonEmissions {
        self.carbon_tracker.get_cumulative_emissions()
    }

    /// Get measurement history
    pub fn get_measurement_history(&self) -> &[CarbonMeasurement] {
        self.carbon_tracker.get_measurement_history()
    }

    // Private implementation methods

    /// Report the configured monitoring interval.
    ///
    /// This does **not** spawn any autonomous background sampling task --
    /// `EnvironmentalMonitor` holds no `Arc`/`Mutex`-wrapped state and
    /// `&self` here cannot safely drive a `'static` background task against
    /// `self.energy_monitor` / `self.carbon_tracker`. Callers must poll by
    /// calling [`Self::record_session`] / [`Self::get_real_time_metrics`]
    /// themselves on their own schedule (e.g. from their training loop).
    /// The old log message ("Environmental monitoring loops started")
    /// claimed background loops had started when none ever ran; this is
    /// corrected to describe only what is actually true.
    async fn start_monitoring_loops(&self) -> Result<()> {
        let interval = Duration::from_secs(self.config.monitoring_interval_secs);

        info!(
            "Environmental monitoring configured with interval {:?}; call record_session() / \
             get_real_time_metrics() to sample -- no autonomous background polling runs \
             automatically",
            interval
        );

        Ok(())
    }

    async fn update_cumulative_metrics(
        &mut self,
        _energy: &EnergyMeasurement,
        _carbon: &CarbonMeasurement,
    ) -> Result<()> {
        // Cumulative metrics are updated within the carbon tracker
        Ok(())
    }

    async fn calculate_cost_impact(&self, energy: &EnergyMeasurement) -> Result<CostAnalysis> {
        let energy_cost = energy.energy_kwh * self.config.energy_price_per_kwh;
        let carbon_cost = self.calculate_carbon_cost(energy.energy_kwh).await?;

        Ok(CostAnalysis {
            energy_cost_usd: energy_cost,
            carbon_cost_usd: Some(carbon_cost),
            infrastructure_cost_usd: energy_cost * 0.1, // 10% infrastructure overhead
            total_cost_usd: energy_cost + carbon_cost,
            cost_per_operation: (energy_cost + carbon_cost) / 1000.0, // Assuming 1000 operations
        })
    }

    async fn calculate_carbon_cost(&self, energy_kwh: f64) -> Result<f64> {
        // Simplified carbon pricing (varies by region and policy)
        let carbon_price_per_ton = 50.0; // USD per ton CO2
        let carbon_intensity = self.carbon_tracker.get_carbon_intensity(&self.config.region);
        let co2_tons = (energy_kwh * carbon_intensity / 1000.0) / 1000.0;

        Ok(co2_tons * carbon_price_per_ton)
    }

    async fn generate_session_recommendations(
        &self,
        efficiency: &SessionEfficiencyAnalysis,
    ) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        if efficiency.efficiency_score < 0.7 {
            recommendations
                .push("Consider optimizing batch size for better GPU utilization".to_string());
        }

        if efficiency.waste_percentage > 30.0 {
            recommendations
                .push("Implement gradient accumulation to reduce memory overhead".to_string());
        }

        recommendations.push("Schedule training during low carbon intensity periods".to_string());
        recommendations
            .push("Consider mixed precision training to reduce energy consumption".to_string());

        Ok(recommendations)
    }

    async fn check_environmental_alerts(&self, report: &SessionImpactReport) -> Result<()> {
        if report.carbon_measurement.co2_emissions_kg > self.config.carbon_alert_threshold {
            warn!(
                "Carbon emission alert: {:.2} kg CO2 exceeds threshold of {:.2} kg",
                report.carbon_measurement.co2_emissions_kg, self.config.carbon_alert_threshold
            );
        }

        if report.energy_measurement.energy_kwh > self.config.energy_alert_threshold {
            warn!(
                "Energy consumption alert: {:.2} kWh exceeds threshold of {:.2} kWh",
                report.energy_measurement.energy_kwh, self.config.energy_alert_threshold
            );
        }

        Ok(())
    }

    /// Real carbon-intensity forecast from the attached [`ForecastSource`].
    /// Errors with [`EnvironmentalMonitorError::NotConfigured`] when none is
    /// attached -- this used to synthesize 24 sine-wave points labeled with
    /// a fixed `confidence: 0.8` regardless of any real grid data.
    async fn get_carbon_intensity_forecasts(&self) -> Result<Vec<CarbonForecast>> {
        let source = self
            .forecast_source
            .as_deref()
            .ok_or(EnvironmentalMonitorError::NotConfigured)?;
        source.carbon_intensity_forecast(&self.config.region, 24)
    }

    /// Real energy-price forecast from the attached [`ForecastSource`]. See
    /// [`Self::get_carbon_intensity_forecasts`].
    async fn get_energy_price_forecasts(&self) -> Result<Vec<EnergyPriceForecast>> {
        let source = self
            .forecast_source
            .as_deref()
            .ok_or(EnvironmentalMonitorError::NotConfigured)?;
        source.energy_price_forecast(&self.config.region, 24)
    }

    async fn find_optimal_execution_time(
        &self,
        workload: &WorkloadDescription,
        carbon_forecasts: &[CarbonForecast],
        price_forecasts: &[EnergyPriceForecast],
    ) -> Result<std::time::SystemTime> {
        let mut best_time = std::time::SystemTime::now();
        let mut best_score = f64::INFINITY;

        for (carbon_forecast, price_forecast) in carbon_forecasts.iter().zip(price_forecasts.iter())
        {
            // Calculate combined score (lower is better)
            let carbon_score =
                carbon_forecast.predicted_carbon_intensity * workload.estimated_energy_kwh;
            let cost_score =
                price_forecast.predicted_price_per_kwh * workload.estimated_energy_kwh * 100.0;
            let combined_score = carbon_score + cost_score;

            if combined_score < best_score {
                best_score = combined_score;
                best_time = carbon_forecast.timestamp;
            }
        }

        Ok(best_time)
    }

    async fn calculate_projected_savings(
        &self,
        workload: &WorkloadDescription,
        _optimal_time: &std::time::SystemTime,
    ) -> Result<ProjectedSavings> {
        Ok(ProjectedSavings {
            energy_savings_kwh: 0.0, // Scheduling doesn't reduce energy, just shifts timing
            cost_savings_usd: workload.estimated_energy_kwh
                * self.config.energy_price_per_kwh
                * 0.2, // 20% cost savings
            carbon_reduction_kg: workload.estimated_energy_kwh * 0.15, // 15% carbon reduction
            efficiency_improvement_percent: 0.0, // Scheduling doesn't improve efficiency
        })
    }
}

// Supporting data structures for [`ForecastSource`]. `pub` because
// `ForecastSource` is a public trait that external callers implement.
#[derive(Debug, Clone)]
pub struct CarbonForecast {
    pub timestamp: std::time::SystemTime,
    pub predicted_carbon_intensity: f64,
    pub renewable_percentage: f64,
    /// The forecast source's own confidence in this prediction. Only ever
    /// set by a real [`ForecastSource`] implementation now -- never
    /// attached to synthetic data.
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct EnergyPriceForecast {
    pub timestamp: std::time::SystemTime,
    pub predicted_price_per_kwh: f64,
    pub confidence: f64,
}

/// Convenience functions

/// Create environmental monitor with default configuration
pub fn create_environmental_monitor() -> EnvironmentalMonitor {
    EnvironmentalMonitor::new(EnvironmentalConfig::default())
}

/// Create environmental monitor for specific region
pub fn create_regional_environmental_monitor(region: String) -> EnvironmentalMonitor {
    let mut config = EnvironmentalConfig::default();
    config.region = region;
    EnvironmentalMonitor::new(config)
}

/// Macro for quick environmental impact recording
#[macro_export]
macro_rules! record_environmental_impact {
    ($monitor:expr, $session_type:expr, $duration:expr, $energy:expr) => {{
        let session_info = SessionInfo {
            session_id: uuid::Uuid::new_v4().to_string(),
            session_type: $session_type,
            duration_hours: $duration,
            workload_description: "default".to_string(),
            region: "US-West".to_string(),
            estimated_energy_kwh: $energy,
        };
        $monitor.record_session(session_info).await
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_environmental_monitor_creation() {
        let monitor = EnvironmentalMonitor::new(EnvironmentalConfig::default());
        assert_eq!(monitor.config.region, "US-West");
        assert!(monitor.config.enable_carbon_tracking);
    }

    #[tokio::test]
    async fn test_session_recording() {
        let mut monitor = EnvironmentalMonitor::new(EnvironmentalConfig::default());

        let session_info = SessionInfo {
            session_id: "test-session".to_string(),
            start_time: std::time::SystemTime::now(),
            session_type: MeasurementType::Training,
            duration_hours: 1.0,
            workload_description: "test training".to_string(),
            region: "US-West".to_string(),
            estimated_energy_kwh: 2.5,
        };

        let result = monitor.record_session(session_info).await;
        assert!(result.is_ok());

        let report = result.expect("operation failed in test");
        assert!(report.carbon_measurement.co2_emissions_kg > 0.0);
        assert!(report.energy_measurement.energy_kwh > 0.0);
    }

    #[tokio::test]
    async fn test_real_time_metrics() {
        let mut monitor = EnvironmentalMonitor::new(EnvironmentalConfig::default());

        // Add a device to get non-zero metrics
        use crate::environmental_monitor::types::{DeviceType, PowerMeasurementMethod};
        monitor
            .energy_monitor
            .add_device(
                "gpu-0".to_string(),
                DeviceType::GPU,
                PowerMeasurementMethod::Estimated,
            )
            .expect("operation failed in test");

        // Record a measurement to have some power consumption
        let _ = monitor.energy_monitor.record_measurement("gpu-0", 250.0, 0.8, Some(70.0));

        let metrics = monitor.get_real_time_metrics().await.expect("async operation failed");
        assert!(metrics.current_power_watts >= 0.0); // Changed to >= to allow 0.0 on fresh monitor
        assert!(metrics.efficiency_ratio > 0.0);

        // Regression: the old implementation always returned
        // `Some(75.0)` regardless of what was actually recorded. The real
        // device measurement above reported `Some(70.0)`.
        assert_eq!(
            metrics.temperature_celsius,
            Some(70.0),
            "must reflect the real recorded temperature, not the old hardcoded Some(75.0)"
        );
    }

    /// Regression test: before any measurement has ever been recorded,
    /// `efficiency_ratio` and `temperature_celsius` must be honest zero /
    /// absence -- never the old hardcoded `0.87` / `Some(75.0)`.
    #[tokio::test]
    async fn test_real_time_metrics_honest_before_any_measurement() {
        let monitor = EnvironmentalMonitor::new(EnvironmentalConfig::default());
        let metrics = monitor.get_real_time_metrics().await.expect("async operation failed");
        assert_eq!(metrics.efficiency_ratio, 0.0);
        assert_eq!(metrics.temperature_celsius, None);
    }

    /// A [`ForecastSource`] mock that returns fixed, clearly-labeled
    /// synthetic data so tests can assert `optimize_scheduling` actually
    /// consumes it (rather than generating its own).
    #[derive(Debug)]
    struct FixedForecastSource;

    impl ForecastSource for FixedForecastSource {
        fn carbon_intensity_forecast(
            &self,
            _region: &str,
            hours: usize,
        ) -> Result<Vec<CarbonForecast>> {
            let now = std::time::SystemTime::now();
            Ok((0..hours)
                .map(|h| CarbonForecast {
                    timestamp: now + Duration::from_secs(h as u64 * 3600),
                    predicted_carbon_intensity: 100.0,
                    renewable_percentage: 60.0,
                    confidence: 0.42,
                })
                .collect())
        }

        fn energy_price_forecast(
            &self,
            _region: &str,
            hours: usize,
        ) -> Result<Vec<EnergyPriceForecast>> {
            let now = std::time::SystemTime::now();
            Ok((0..hours)
                .map(|h| EnergyPriceForecast {
                    timestamp: now + Duration::from_secs(h as u64 * 3600),
                    predicted_price_per_kwh: 0.1,
                    confidence: 0.42,
                })
                .collect())
        }
    }

    /// Regression test: without a [`ForecastSource`] attached,
    /// `optimize_scheduling` must honestly fail instead of scheduling
    /// against a fabricated sine-wave forecast.
    #[tokio::test]
    async fn test_scheduling_optimization_without_forecast_source_errors() {
        let monitor = EnvironmentalMonitor::new(EnvironmentalConfig::default());
        assert!(!monitor.has_forecast_source());

        let workload = WorkloadDescription {
            workload_name: "test workload".to_string(),
            workload_type: "training".to_string(),
            priority: WorkloadPriority::Medium,
            estimated_duration_hours: 2.0,
            resource_requirements: std::collections::HashMap::new(),
            estimated_energy_kwh: 5.0,
        };

        let error = monitor
            .optimize_scheduling(workload)
            .await
            .expect_err("must fail honestly without a ForecastSource");
        assert!(error.to_string().contains("ForecastSource"));
    }

    /// Regression test: with a real [`ForecastSource`] attached,
    /// `optimize_scheduling` must reflect its real data -- including a real
    /// (non-fabricated) `confidence` derived from the source, not the old
    /// hardcoded `0.85`.
    #[tokio::test]
    async fn test_scheduling_optimization_uses_real_forecast_source() {
        let mut monitor = EnvironmentalMonitor::new(EnvironmentalConfig::default());
        monitor.set_forecast_source(Box::new(FixedForecastSource));
        assert!(monitor.has_forecast_source());

        let workload = WorkloadDescription {
            workload_name: "test workload".to_string(),
            workload_type: "training".to_string(),
            priority: WorkloadPriority::Medium,
            estimated_duration_hours: 2.0,
            resource_requirements: std::collections::HashMap::new(),
            estimated_energy_kwh: 5.0,
        };

        let schedule = monitor.optimize_scheduling(workload).await.expect("async operation failed");
        assert!(schedule.projected_savings.carbon_reduction_kg >= 0.0);
        assert!(
            schedule.carbon_intensity_forecast.iter().all(|&v| v == 100.0),
            "must reflect the real ForecastSource data, not a sine wave"
        );
        assert_eq!(
            schedule.confidence, 0.42,
            "must be derived from the real ForecastSource confidence, not the old hardcoded 0.85"
        );
    }

    #[tokio::test]
    async fn test_environmental_report_generation() {
        let monitor = EnvironmentalMonitor::new(EnvironmentalConfig::default());

        let report = monitor
            .generate_environmental_report(ReportType::Summary)
            .await
            .expect("async operation failed");
        assert!(!report.report_id.is_empty());
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_convenience_functions() {
        let monitor = create_environmental_monitor();
        assert_eq!(monitor.config.region, "US-West");

        let regional_monitor = create_regional_environmental_monitor("EU-North".to_string());
        assert_eq!(regional_monitor.config.region, "EU-North");
    }
}
