//! Thermal management modules for mobile deployment.
//!
//! - `predictive`: time-series predictive thermal throttle prevention.

pub mod predictive;

pub use predictive::{
    PredictiveThermalManager, ThermalAction, ThermalPrediction, ThermalPredictor, ThermalSample,
    ThermalState,
};

// ─── Module-level tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ThermalPredictor construction ─────────────────────────────────────────

    #[test]
    fn test_thermal_predictor_new_fields() {
        let p = ThermalPredictor::new(30.0, 25.0);
        assert!((p.thermal_time_constant - 30.0).abs() < 1e-6);
        assert!((p.ambient_temp - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_thermal_predictor_model_coefficients_length() {
        let p = ThermalPredictor::new(20.0, 22.0);
        assert_eq!(p.model_coefficients.len(), 3);
    }

    // ── predict_cooling ───────────────────────────────────────────────────────

    #[test]
    fn test_predict_cooling_at_t0_equals_start_temp() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let result = p.predict_cooling(70.0, 0.0);
        assert!((result - 70.0).abs() < 1e-4, "at t=0 should return start_temp, got {result}");
    }

    #[test]
    fn test_predict_cooling_approaches_ambient_at_large_t() {
        let p = ThermalPredictor::new(30.0, 25.0);
        // After 10 time constants, exp(-10) ≈ 4.5e-5 ≈ 0
        let result = p.predict_cooling(70.0, 300.0);
        assert!((result - 25.0).abs() < 1.0, "should approach ambient 25°C, got {result}");
    }

    #[test]
    fn test_predict_cooling_start_equals_ambient_stays_ambient() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let result = p.predict_cooling(25.0, 60.0);
        assert!((result - 25.0).abs() < 1e-5, "should stay at ambient, got {result}");
    }

    #[test]
    fn test_predict_cooling_monotonically_decreasing_toward_ambient() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let t1 = p.predict_cooling(70.0, 10.0);
        let t2 = p.predict_cooling(70.0, 20.0);
        let t3 = p.predict_cooling(70.0, 60.0);
        assert!(t1 > t2 && t2 > t3, "should be monotonically decreasing");
        assert!(t3 > 25.0, "should still be above ambient after 60s");
    }

    #[test]
    fn test_predict_cooling_result_between_ambient_and_start() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let result = p.predict_cooling(70.0, 15.0);
        assert!(result > 25.0 && result < 70.0, "result {result} should be between ambient and start");
    }

    // ── predict_heating ───────────────────────────────────────────────────────

    #[test]
    fn test_predict_heating_increases_temp() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let result = p.predict_heating(30.0, 5.0, 10.0);
        assert!(result > 30.0, "heating should increase temperature, got {result}");
    }

    #[test]
    fn test_predict_heating_zero_power_unchanged() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let result = p.predict_heating(50.0, 0.0, 60.0);
        assert!((result - 50.0).abs() < 1e-6, "zero power should not change temp, got {result}");
    }

    #[test]
    fn test_predict_heating_proportional_to_power() {
        let p = ThermalPredictor::new(20.0, 25.0);
        let delta1 = p.predict_heating(30.0, 2.0, 10.0) - 30.0;
        let delta2 = p.predict_heating(30.0, 4.0, 10.0) - 30.0;
        assert!((delta2 - 2.0 * delta1).abs() < 1e-4, "heating should be proportional to power");
    }

    // ── predict_trajectory ────────────────────────────────────────────────────

    #[test]
    fn test_predict_trajectory_empty_schedule() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let traj = p.predict_trajectory(40.0, &[]);
        assert!(traj.is_empty(), "empty schedule should yield empty trajectory");
    }

    #[test]
    fn test_predict_trajectory_length_matches_schedule() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let schedule = vec![(1.0f32, 5.0f32), (2.0, 10.0), (0.5, 3.0)];
        let traj = p.predict_trajectory(40.0, &schedule);
        assert_eq!(traj.len(), 3, "trajectory length should match schedule");
    }

    #[test]
    fn test_predict_trajectory_zero_power_step_cools() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let schedule = vec![(0.0f32, 30.0f32)]; // no workload, 30s
        let traj = p.predict_trajectory(70.0, &schedule);
        assert_eq!(traj.len(), 1);
        assert!(traj[0] < 70.0, "should cool from 70°C with no power, got {}", traj[0]);
    }

    #[test]
    fn test_predict_trajectory_high_power_heats() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let schedule = vec![(100.0f32, 5.0f32)]; // very high power
        let traj = p.predict_trajectory(30.0, &schedule);
        assert!(traj[0] > 30.0, "high power should heat device, got {}", traj[0]);
    }

    // ── will_throttle ─────────────────────────────────────────────────────────

    #[test]
    fn test_will_throttle_false_when_all_below() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let traj = vec![50.0, 55.0, 60.0, 70.0];
        assert!(!p.will_throttle(&traj, 75.0));
    }

    #[test]
    fn test_will_throttle_true_when_one_above() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let traj = vec![50.0, 55.0, 76.0, 70.0];
        assert!(p.will_throttle(&traj, 75.0));
    }

    #[test]
    fn test_will_throttle_false_on_empty_trajectory() {
        let p = ThermalPredictor::new(30.0, 25.0);
        assert!(!p.will_throttle(&[], 75.0));
    }

    #[test]
    fn test_will_throttle_at_exact_limit() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let traj = vec![75.0]; // equal to limit
        assert!(p.will_throttle(&traj, 75.0), "equal to limit should trigger");
    }

    // ── max_safe_throughput ───────────────────────────────────────────────────

    #[test]
    fn test_max_safe_throughput_zero_when_at_limit() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let result = p.max_safe_throughput(75.0, 75.0, 1.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_max_safe_throughput_zero_when_above_limit() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let result = p.max_safe_throughput(80.0, 75.0, 1.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_max_safe_throughput_zero_when_power_zero() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let result = p.max_safe_throughput(30.0, 75.0, 0.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_max_safe_throughput_zero_when_power_negative() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let result = p.max_safe_throughput(30.0, 75.0, -1.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_max_safe_throughput_positive_below_limit() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let result = p.max_safe_throughput(30.0, 75.0, 0.5);
        assert!(result > 0.0, "should return positive throughput when below limit, got {result}");
    }

    #[test]
    fn test_max_safe_throughput_inversely_proportional_to_power() {
        let p = ThermalPredictor::new(30.0, 25.0);
        let t1 = p.max_safe_throughput(30.0, 75.0, 1.0);
        let t2 = p.max_safe_throughput(30.0, 75.0, 2.0);
        assert!((t1 - 2.0 * t2).abs() < 1e-3, "throughput should be inversely proportional to power_per_token");
    }
}
