//! Predictive thermal throttling prevention.
//!
//! Uses a linear regression model on recent temperature history to predict
//! the thermal trajectory and proactively reduce workload before throttling
//! occurs on the device.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use trustformers_core::errors::{Result, TrustformersError};

// ─── Core data types ──────────────────────────────────────────────────────────

/// A single thermal measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalSample {
    /// Milliseconds since some epoch (e.g. process start or Unix time).
    pub timestamp_ms: u64,
    /// Device junction / skin temperature in degrees Celsius.
    pub temperature_celsius: f32,
    /// Current OS-reported thermal state.
    pub thermal_state: ThermalState,
    /// Normalised workload intensity: 0.0 = idle, 1.0 = maximum.
    pub workload_intensity: f32,
}

/// Coarse thermal state classification.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum ThermalState {
    Cool,     // < 40 °C
    Warm,     // 40–60 °C
    Hot,      // 60–75 °C
    Critical, // > 75 °C
}

impl ThermalState {
    /// Classify from a Celsius reading.
    pub fn from_celsius(temp: f32) -> Self {
        if temp < 40.0 {
            ThermalState::Cool
        } else if temp < 60.0 {
            ThermalState::Warm
        } else if temp < 75.0 {
            ThermalState::Hot
        } else {
            ThermalState::Critical
        }
    }

    /// Maximum recommended workload intensity for this thermal state.
    pub fn max_workload_intensity(self) -> f32 {
        match self {
            ThermalState::Cool => 1.0,
            ThermalState::Warm => 0.8,
            ThermalState::Hot => 0.5,
            ThermalState::Critical => 0.2,
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            ThermalState::Cool => "cool",
            ThermalState::Warm => "warm",
            ThermalState::Hot => "hot",
            ThermalState::Critical => "critical",
        }
    }
}

// ─── Trend ────────────────────────────────────────────────────────────────────

/// Directional thermal trend computed from recent history.
#[derive(Debug, Clone, PartialEq)]
pub enum ThermalTrend {
    /// Temperature is rising at `rate_celsius_per_sec` degrees per second.
    Rising { rate_celsius_per_sec: f32 },
    /// Temperature is stable (rate < threshold).
    Stable,
    /// Temperature is cooling at `rate_celsius_per_sec` degrees per second.
    Cooling { rate_celsius_per_sec: f32 },
}

impl ThermalTrend {
    /// Signed rate in °C/sec (positive = rising, negative = cooling, 0 = stable).
    pub fn rate_celsius_per_sec(&self) -> f32 {
        match self {
            ThermalTrend::Rising { rate_celsius_per_sec } => *rate_celsius_per_sec,
            ThermalTrend::Stable => 0.0,
            ThermalTrend::Cooling { rate_celsius_per_sec } => -rate_celsius_per_sec,
        }
    }
}

// ─── Actions ─────────────────────────────────────────────────────────────────

/// Recommended corrective action from the thermal manager.
#[derive(Debug, Clone, PartialEq)]
pub enum ThermalAction {
    /// No change needed.
    NoChange,
    /// Switch from FP32 to FP16 to reduce compute heat.
    ReducePrecision,
    /// Use a smaller batch size.
    ReduceBatchSize,
    /// Temporarily suspend inference.
    PauseInference,
    /// Move workload from GPU/NPU to CPU (lower clock speed).
    SwitchToCpu,
    /// Emergency: maximum throttle, all non-essential work suspended.
    EmergencyThrottle,
}

// ─── Prediction result ────────────────────────────────────────────────────────

/// Output of `PredictiveThermalManager::predict`.
#[derive(Debug, Clone)]
pub struct ThermalPrediction {
    /// Most recently recorded temperature.
    pub current_temp: f32,
    /// Predicted temperature 30 seconds from now.
    pub predicted_temp_30s: f32,
    /// Predicted temperature 60 seconds from now.
    pub predicted_temp_60s: f32,
    /// Current thermal trend.
    pub trend: ThermalTrend,
    /// Seconds until the throttle threshold is breached, given current trend.
    pub time_to_throttle_secs: Option<f32>,
    /// Suggested workload intensity (0.0–1.0).
    pub recommended_intensity: f32,
    /// Recommended system action.
    pub action: ThermalAction,
}

// ─── Summary ─────────────────────────────────────────────────────────────────

/// Aggregate statistics over recent thermal history.
#[derive(Debug, Clone)]
pub struct ThermalSummary {
    pub current_state: ThermalState,
    pub avg_temp_last_60s: f32,
    pub max_temp_last_60s: f32,
    pub trend: ThermalTrend,
    /// Number of times the current session has hit a Hot/Critical state.
    pub throttle_events: u32,
    pub total_samples: usize,
}

// ─── Manager ──────────────────────────────────────────────────────────────────

/// Predictive thermal manager.
///
/// Records temperature samples, fits a linear model to recent history, and
/// predicts future thermal trajectories to guide proactive throttling.
pub struct PredictiveThermalManager {
    history: VecDeque<ThermalSample>,
    max_history: usize,
    throttle_threshold_celsius: f32,
    prediction_horizon_secs: f32,
    throttle_events: u32,
}

impl Default for PredictiveThermalManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictiveThermalManager {
    /// Create a manager with sensible defaults.
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(120),
            max_history: 120,
            throttle_threshold_celsius: 75.0,
            prediction_horizon_secs: 60.0,
        throttle_events: 0,
        }
    }

    /// Override the throttle threshold.
    pub fn with_throttle_threshold(mut self, celsius: f32) -> Self {
        self.throttle_threshold_celsius = celsius;
        self
    }

    /// Override the maximum history buffer depth.
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Record a new thermal sample.
    pub fn record(&mut self, sample: ThermalSample) {
        if sample.thermal_state >= ThermalState::Hot {
            self.throttle_events += 1;
        }
        if self.history.len() == self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(sample);
    }

    /// Predict future thermal state using linear regression on recent samples.
    pub fn predict(&self) -> Result<ThermalPrediction> {
        if self.history.is_empty() {
            return Err(TrustformersError::invalid_input(
                "no thermal history — record at least one sample before predicting".to_string(),
            ));
        }

        let current_temp = self
            .history
            .back()
            .map(|s| s.temperature_celsius)
            .unwrap_or(25.0);

        let trend = self.compute_trend();
        let rate = trend.rate_celsius_per_sec();

        let predicted_temp_30s = (current_temp + rate * 30.0).max(0.0);
        let predicted_temp_60s = (current_temp + rate * 60.0).max(0.0);

        let time_to_throttle_secs = self.time_to_threshold(current_temp, &trend);

        let recommended_intensity = self.recommended_intensity();

        let action = Self::choose_action(
            current_temp,
            &trend,
            time_to_throttle_secs,
            recommended_intensity,
        );

        Ok(ThermalPrediction {
            current_temp,
            predicted_temp_30s,
            predicted_temp_60s,
            trend,
            time_to_throttle_secs,
            recommended_intensity,
            action,
        })
    }

    /// Compute linear regression slope (°C/sec) from recent history.
    ///
    /// Timestamps are in milliseconds; slope is converted to °C/sec.
    /// Returns `ThermalTrend::Stable` if fewer than 2 samples or slope is negligible.
    fn compute_trend(&self) -> ThermalTrend {
        if self.history.len() < 2 {
            return ThermalTrend::Stable;
        }

        // Use the most recent 20 samples for the regression.
        let samples: Vec<&ThermalSample> = self.history.iter().rev().take(20).collect();
        let n = samples.len() as f64;

        // x = time in seconds relative to the oldest sample in the window
        let t0_ms = samples.last().map(|s| s.timestamp_ms).unwrap_or(0) as f64;

        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_xy = 0.0_f64;

        for s in &samples {
            let x = (s.timestamp_ms as f64 - t0_ms) / 1000.0; // seconds
            let y = s.temperature_celsius as f64;
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return ThermalTrend::Stable;
        }
        let slope = (n * sum_xy - sum_x * sum_y) / denom; // °C / sec

        const STABILITY_THRESHOLD: f64 = 0.05; // °C/sec
        if slope.abs() < STABILITY_THRESHOLD {
            ThermalTrend::Stable
        } else if slope > 0.0 {
            ThermalTrend::Rising {
                rate_celsius_per_sec: slope as f32,
            }
        } else {
            ThermalTrend::Cooling {
                rate_celsius_per_sec: (-slope) as f32,
            }
        }
    }

    /// Seconds until `current_temp` reaches `throttle_threshold_celsius` at the current rate.
    fn time_to_threshold(&self, current_temp: f32, trend: &ThermalTrend) -> Option<f32> {
        let rate = trend.rate_celsius_per_sec();
        if rate <= 0.0 {
            return None; // not heating up
        }
        let headroom = self.throttle_threshold_celsius - current_temp;
        if headroom <= 0.0 {
            return Some(0.0); // already throttled
        }
        Some(headroom / rate)
    }

    /// Recommended workload intensity based on current and near-future thermal state.
    pub fn recommended_intensity(&self) -> f32 {
        let current_temp = self
            .history
            .back()
            .map(|s| s.temperature_celsius)
            .unwrap_or(25.0);

        let current_state = ThermalState::from_celsius(current_temp);
        let base_intensity = current_state.max_workload_intensity();

        // If we're approaching the threshold quickly, pre-emptively reduce.
        let trend = self.compute_trend();
        let rate = trend.rate_celsius_per_sec();
        if rate > 0.5 {
            // More than 0.5 °C/sec: back off faster
            (base_intensity * 0.75).max(0.1)
        } else if rate > 0.2 {
            (base_intensity * 0.9).max(0.1)
        } else {
            base_intensity
        }
    }

    /// Choose the corrective action given current conditions.
    fn choose_action(
        current_temp: f32,
        trend: &ThermalTrend,
        time_to_throttle: Option<f32>,
        recommended_intensity: f32,
    ) -> ThermalAction {
        let state = ThermalState::from_celsius(current_temp);
        match state {
            ThermalState::Critical => ThermalAction::EmergencyThrottle,
            ThermalState::Hot => {
                if matches!(trend, ThermalTrend::Rising { .. }) {
                    ThermalAction::PauseInference
                } else {
                    ThermalAction::SwitchToCpu
                }
            }
            ThermalState::Warm => {
                // Proactive: if we'll hit threshold within 30s, act now
                if matches!(time_to_throttle, Some(t) if t < 30.0) {
                    ThermalAction::ReduceBatchSize
                } else if recommended_intensity < 0.7 {
                    ThermalAction::ReducePrecision
                } else {
                    ThermalAction::NoChange
                }
            }
            ThermalState::Cool => ThermalAction::NoChange,
        }
    }

    /// Recent thermal history (oldest first).
    pub fn history(&self) -> &VecDeque<ThermalSample> {
        &self.history
    }

    /// Snapshot summary over the last 60 seconds of history.
    pub fn summary(&self) -> ThermalSummary {
        let current_temp = self
            .history
            .back()
            .map(|s| s.temperature_celsius)
            .unwrap_or(25.0);
        let current_state = ThermalState::from_celsius(current_temp);

        let recent: Vec<f32> = if self.history.is_empty() {
            vec![]
        } else {
            let newest_ts = self.history.back().map(|s| s.timestamp_ms).unwrap_or(0);
            self.history
                .iter()
                .filter(|s| newest_ts.saturating_sub(s.timestamp_ms) <= 60_000)
                .map(|s| s.temperature_celsius)
                .collect()
        };

        let avg_temp_last_60s = if recent.is_empty() {
            current_temp
        } else {
            recent.iter().sum::<f32>() / recent.len() as f32
        };
        let max_temp_last_60s = recent
            .iter()
            .cloned()
            .fold(current_temp, f32::max);

        ThermalSummary {
            current_state,
            avg_temp_last_60s,
            max_temp_last_60s,
            trend: self.compute_trend(),
            throttle_events: self.throttle_events,
            total_samples: self.history.len(),
        }
    }
}

// ─── Physics-based predictor ──────────────────────────────────────────────────

/// Physics-based thermal predictor using Newton's law of cooling.
///
/// Models the device as a lumped thermal mass with:
/// - A thermal time constant τ (analogous to an RC circuit)
/// - An ambient temperature T_∞
/// - A linear workload → power model
pub struct ThermalPredictor {
    /// Linear regression coefficients: [workload_coeff, time_coeff, prev_temp_coeff].
    pub model_coefficients: Vec<f32>,
    /// Thermal time constant τ in seconds (RC-style cooling rate).
    pub thermal_time_constant: f32,
    /// Ambient / environment temperature in °C.
    pub ambient_temp: f32,
}

impl ThermalPredictor {
    /// Create a new predictor with sensible defaults.
    ///
    /// - `time_constant`: τ in seconds (e.g. 30.0 for a typical smartphone)
    /// - `ambient_temp`: room / environment temperature in °C
    pub fn new(time_constant: f32, ambient_temp: f32) -> Self {
        Self {
            model_coefficients: vec![0.5, 0.1, 0.4],
            thermal_time_constant: time_constant.max(f32::EPSILON),
            ambient_temp,
        }
    }

    /// Newton's law of cooling:
    /// T(t) = T_∞ + (T_0 − T_∞) · exp(−t / τ)
    pub fn predict_cooling(&self, start_temp: f32, time_secs: f32) -> f32 {
        let tau = self.thermal_time_constant;
        let t_inf = self.ambient_temp;
        t_inf + (start_temp - t_inf) * (-time_secs / tau).exp()
    }

    /// Estimate temperature increase from a constant workload.
    ///
    /// Uses a simplified thermal mass model:
    ///   ΔT = (power_watts × time_secs) / (τ × 2)
    /// where `τ × 2` is treated as a conceptual thermal capacity.
    pub fn predict_heating(&self, start_temp: f32, power_watts: f32, time_secs: f32) -> f32 {
        let thermal_mass = self.thermal_time_constant * 2.0;
        start_temp + (power_watts * time_secs) / thermal_mass
    }

    /// Simulate the combined heating and cooling trajectory through a workload schedule.
    ///
    /// For each step `(power_watts, duration_secs)`:
    /// 1. Apply Newton's cooling over `duration_secs` from the current temperature.
    /// 2. Add the heating delta from `power_watts` over the same duration.
    ///
    /// Returns the temperature **after** each step.
    pub fn predict_trajectory(
        &self,
        start_temp: f32,
        workload_schedule: &[(f32, f32)],
    ) -> Vec<f32> {
        let mut temps = Vec::with_capacity(workload_schedule.len());
        let mut current = start_temp;

        for &(power_watts, duration_secs) in workload_schedule {
            // Newton cooling component
            let cooled = self.predict_cooling(current, duration_secs);
            // Heating delta
            let thermal_mass = self.thermal_time_constant * 2.0;
            let delta_heat = (power_watts * duration_secs) / thermal_mass;
            current = cooled + delta_heat;
            temps.push(current);
        }

        temps
    }

    /// Check whether any temperature in the trajectory exceeds `limit_celsius`.
    pub fn will_throttle(&self, trajectory: &[f32], limit_celsius: f32) -> bool {
        trajectory.iter().any(|&t| t >= limit_celsius)
    }

    /// Compute the maximum safe inference throughput (tokens/sec or abstract units)
    /// such that the device stays below `limit_celsius` at thermal steady state.
    ///
    /// Steady-state heat balance:
    ///   power_in = power_out
    ///   power_out ≈ (T_steady − T_∞) / (τ × 0.1)  [simplified thermal resistance]
    ///
    /// Max safe power = (limit_celsius − ambient_temp) / (τ × 0.1)
    /// Max throughput = max_power / power_per_token
    ///
    /// Returns `0.0` if `current_temp >= limit_celsius` or `power_per_token <= 0`.
    pub fn max_safe_throughput(
        &self,
        current_temp: f32,
        limit_celsius: f32,
        power_per_token: f32,
    ) -> f32 {
        if current_temp >= limit_celsius || power_per_token <= 0.0 {
            return 0.0;
        }
        let thermal_resistance = self.thermal_time_constant * 0.1;
        let max_power = (limit_celsius - self.ambient_temp) / thermal_resistance;
        if max_power <= 0.0 {
            return 0.0;
        }
        max_power / power_per_token
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(ts_ms: u64, temp: f32) -> ThermalSample {
        ThermalSample {
            timestamp_ms: ts_ms,
            temperature_celsius: temp,
            thermal_state: ThermalState::from_celsius(temp),
            workload_intensity: 0.8,
        }
    }

    #[test]
    fn test_thermal_state_classification() {
        assert_eq!(ThermalState::from_celsius(35.0), ThermalState::Cool);
        assert_eq!(ThermalState::from_celsius(50.0), ThermalState::Warm);
        assert_eq!(ThermalState::from_celsius(70.0), ThermalState::Hot);
        assert_eq!(ThermalState::from_celsius(80.0), ThermalState::Critical);
    }

    #[test]
    fn test_max_workload_intensity() {
        assert_eq!(ThermalState::Cool.max_workload_intensity(), 1.0);
        assert_eq!(ThermalState::Warm.max_workload_intensity(), 0.8);
        assert_eq!(ThermalState::Hot.max_workload_intensity(), 0.5);
        assert_eq!(ThermalState::Critical.max_workload_intensity(), 0.2);
    }

    #[test]
    fn test_predict_requires_history() {
        let manager = PredictiveThermalManager::new();
        assert!(manager.predict().is_err());
    }

    #[test]
    fn test_predict_single_sample() {
        let mut manager = PredictiveThermalManager::new();
        manager.record(make_sample(0, 35.0));
        let pred = manager.predict().expect("should predict");
        assert_eq!(pred.current_temp, 35.0);
        assert!(matches!(pred.trend, ThermalTrend::Stable));
        assert_eq!(pred.action, ThermalAction::NoChange);
    }

    #[test]
    fn test_rising_trend_detection() {
        let mut manager = PredictiveThermalManager::new();
        // +1 °C per second over 10 samples spaced 1 s apart
        for i in 0..10u64 {
            manager.record(make_sample(i * 1000, 50.0 + i as f32));
        }
        let pred = manager.predict().expect("should predict");
        assert!(
            matches!(pred.trend, ThermalTrend::Rising { rate_celsius_per_sec } if rate_celsius_per_sec > 0.5),
            "expected Rising trend, got {:?}",
            pred.trend
        );
        assert!(pred.predicted_temp_30s > pred.current_temp);
    }

    #[test]
    fn test_cooling_trend_detection() {
        let mut manager = PredictiveThermalManager::new();
        // Cooling from 70 to 60 over 10 seconds
        for i in 0..10u64 {
            manager.record(make_sample(i * 1000, 70.0 - i as f32));
        }
        let pred = manager.predict().expect("should predict");
        assert!(
            matches!(pred.trend, ThermalTrend::Cooling { .. }),
            "expected Cooling trend, got {:?}",
            pred.trend
        );
        assert!(pred.predicted_temp_30s < pred.current_temp);
    }

    #[test]
    fn test_time_to_throttle_calculated_when_rising() {
        let mut manager =
            PredictiveThermalManager::new().with_throttle_threshold(75.0);
        // Start at 70 °C, rising 1 °C/s → should throttle in ~5 s
        for i in 0..5u64 {
            manager.record(make_sample(i * 1000, 70.0 + i as f32));
        }
        let pred = manager.predict().expect("should predict");
        assert!(
            pred.time_to_throttle_secs.is_some(),
            "expected a time-to-throttle estimate"
        );
        let ttl = pred.time_to_throttle_secs.unwrap();
        assert!(ttl < 30.0, "expected throttle within 30 s, got {ttl}");
    }

    #[test]
    fn test_emergency_throttle_when_critical() {
        let mut manager = PredictiveThermalManager::new();
        manager.record(make_sample(0, 80.0));
        let pred = manager.predict().expect("should predict");
        assert_eq!(pred.action, ThermalAction::EmergencyThrottle);
    }

    #[test]
    fn test_throttle_event_counter() {
        let mut manager = PredictiveThermalManager::new();
        manager.record(make_sample(0, 35.0)); // cool
        manager.record(make_sample(1000, 65.0)); // hot
        manager.record(make_sample(2000, 65.0)); // hot
        let summary = manager.summary();
        assert_eq!(summary.throttle_events, 2);
    }

    #[test]
    fn test_summary_statistics() {
        let mut manager = PredictiveThermalManager::new();
        let temps = [30.0f32, 40.0, 50.0, 60.0];
        for (i, &t) in temps.iter().enumerate() {
            manager.record(make_sample(i as u64 * 1000, t));
        }
        let summary = manager.summary();
        assert_eq!(summary.total_samples, 4);
        assert!(summary.max_temp_last_60s >= 60.0);
        assert!(summary.avg_temp_last_60s > 30.0);
    }

    // ── ThermalState boundary tests ─────────────────────────────────────────

    #[test]
    fn test_thermal_state_boundary_at_40() {
        // Exactly 40 °C is Warm (< 40 is Cool)
        assert_eq!(ThermalState::from_celsius(39.99), ThermalState::Cool);
        assert_eq!(ThermalState::from_celsius(40.0), ThermalState::Warm);
    }

    #[test]
    fn test_thermal_state_boundary_at_60() {
        assert_eq!(ThermalState::from_celsius(59.99), ThermalState::Warm);
        assert_eq!(ThermalState::from_celsius(60.0), ThermalState::Hot);
    }

    #[test]
    fn test_thermal_state_boundary_at_75() {
        assert_eq!(ThermalState::from_celsius(74.99), ThermalState::Hot);
        assert_eq!(ThermalState::from_celsius(75.0), ThermalState::Critical);
    }

    // ── ThermalState labels ─────────────────────────────────────────────────

    #[test]
    fn test_thermal_state_labels() {
        assert_eq!(ThermalState::Cool.label(), "cool");
        assert_eq!(ThermalState::Warm.label(), "warm");
        assert_eq!(ThermalState::Hot.label(), "hot");
        assert_eq!(ThermalState::Critical.label(), "critical");
    }

    // ── ThermalTrend rate_celsius_per_sec ───────────────────────────────────

    #[test]
    fn test_thermal_trend_rate_rising() {
        let trend = ThermalTrend::Rising { rate_celsius_per_sec: 2.5 };
        let rate = trend.rate_celsius_per_sec();
        assert!((rate - 2.5).abs() < 1e-6, "Rising rate should be positive, got {rate}");
    }

    #[test]
    fn test_thermal_trend_rate_stable() {
        let trend = ThermalTrend::Stable;
        assert!((trend.rate_celsius_per_sec()).abs() < 1e-6, "Stable rate should be 0");
    }

    #[test]
    fn test_thermal_trend_rate_cooling() {
        let trend = ThermalTrend::Cooling { rate_celsius_per_sec: 1.5 };
        let rate = trend.rate_celsius_per_sec();
        assert!(rate < 0.0, "Cooling rate should be negative, got {rate}");
        assert!((rate - (-1.5)).abs() < 1e-6, "Expected -1.5, got {rate}");
    }

    // ── PredictiveThermalManager with_throttle_threshold ────────────────────

    #[test]
    fn test_custom_throttle_threshold_affects_time_to_throttle() {
        // With a higher throttle threshold, time-to-throttle should be longer.
        // Use rising samples so that time_to_throttle_secs is computed.
        let mut manager_low =
            PredictiveThermalManager::new().with_throttle_threshold(65.0);
        let mut manager_high =
            PredictiveThermalManager::new().with_throttle_threshold(90.0);

        for i in 0..5u64 {
            let sample = make_sample(i * 1000, 60.0 + i as f32);
            manager_low.record(sample.clone());
            manager_high.record(sample);
        }

        let pred_low = manager_low.predict().expect("predict (low threshold)");
        let pred_high = manager_high.predict().expect("predict (high threshold)");

        // Both have rising trend; high threshold gives more headroom → later throttle
        let ttl_low = pred_low.time_to_throttle_secs.unwrap_or(f32::INFINITY);
        let ttl_high = pred_high.time_to_throttle_secs.unwrap_or(f32::INFINITY);
        assert!(
            ttl_high > ttl_low,
            "Higher threshold should give more time before throttle: low={ttl_low} high={ttl_high}"
        );
    }

    // ── Stable prediction scenario ─────────────────────────────────────────

    #[test]
    fn test_stable_prediction_no_change_action() {
        let mut manager = PredictiveThermalManager::new();
        // Constant cool temperature
        for i in 0..5u64 {
            manager.record(make_sample(i * 1000, 35.0));
        }
        let pred = manager.predict().expect("should predict");
        assert_eq!(pred.action, ThermalAction::NoChange);
        assert!(matches!(pred.trend, ThermalTrend::Stable));
    }
}
