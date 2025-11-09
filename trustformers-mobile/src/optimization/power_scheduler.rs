//! Power-Aware Scheduling Module
//!
//! Provides intelligent scheduling of operations based on power constraints,
//! thermal state, and battery level for optimal mobile inference.

use super::{ComputationGraph, ExecutionSchedule, GraphOperator, KernelType, PowerHint, PowerMode};
use crate::MobileConfig;
use std::collections::{HashMap, VecDeque};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

/// Scheduling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// Minimize latency
    Performance,
    /// Balance performance and power
    Balanced,
    /// Minimize power consumption
    PowerSaving,
    /// Adaptive based on device state
    Adaptive,
}

/// Power profile for operations
#[derive(Debug, Clone)]
pub struct PowerProfile {
    /// Operation type to power consumption mapping
    pub operation_power: HashMap<String, f32>,
    /// Thermal coefficients
    pub thermal_coefficients: ThermalCoefficients,
    /// Battery discharge model
    pub battery_model: BatteryModel,
}

/// Thermal constraints
#[derive(Debug, Clone)]
pub struct ThermalConstraints {
    /// Maximum temperature (Celsius)
    pub max_temperature: f32,
    /// Thermal throttling thresholds
    pub throttle_thresholds: Vec<(f32, f32)>, // (temperature, performance_ratio)
    /// Current temperature
    pub current_temperature: f32,
}

/// Battery constraints
#[derive(Debug, Clone)]
pub struct BatteryConstraints {
    /// Minimum battery level to maintain
    pub min_battery_level: f32,
    /// Current battery level (0-100)
    pub current_battery_level: f32,
    /// Is device charging
    pub is_charging: bool,
    /// Power budget (Watts)
    pub power_budget: f32,
}

/// Thermal model coefficients
#[derive(Debug, Clone)]
pub struct ThermalCoefficients {
    /// Heat generation per watt
    pub heat_per_watt: f32,
    /// Cooling rate
    pub cooling_rate: f32,
    /// Ambient temperature
    pub ambient_temp: f32,
}

/// Battery discharge model
#[derive(Debug, Clone)]
pub struct BatteryModel {
    /// Battery capacity (mAh)
    pub capacity_mah: f32,
    /// Voltage
    pub voltage: f32,
    /// Discharge curve
    pub discharge_curve: Vec<(f32, f32)>, // (power, discharge_rate)
}

/// Scheduling decision
#[derive(Debug, Clone)]
pub struct SchedulingDecision {
    /// Operator execution order
    pub execution_order: Vec<usize>,
    /// Power mode for each operator
    pub power_modes: HashMap<usize, PowerMode>,
    /// Estimated execution time
    pub estimated_time_ms: f32,
    /// Estimated power consumption
    pub estimated_power_watts: f32,
    /// Estimated temperature rise
    pub estimated_temp_rise: f32,
}

/// Power-aware scheduler
pub struct PowerAwareScheduler {
    config: MobileConfig,
    policy: SchedulingPolicy,
    power_profile: PowerProfile,
    thermal_constraints: ThermalConstraints,
    battery_constraints: BatteryConstraints,
}

impl PowerAwareScheduler {
    /// Create new power-aware scheduler
    pub fn new(config: MobileConfig) -> Self {
        Self {
            config,
            policy: SchedulingPolicy::Balanced,
            power_profile: Self::create_default_power_profile(),
            thermal_constraints: Self::create_default_thermal_constraints(),
            battery_constraints: Self::create_default_battery_constraints(),
        }
    }

    /// Set scheduling policy
    pub fn set_policy(&mut self, policy: SchedulingPolicy) {
        self.policy = policy;
    }

    /// Update thermal state
    pub fn update_thermal_state(&mut self, temperature: f32) {
        self.thermal_constraints.current_temperature = temperature;
    }

    /// Update battery state
    pub fn update_battery_state(&mut self, level: f32, is_charging: bool) {
        self.battery_constraints.current_battery_level = level;
        self.battery_constraints.is_charging = is_charging;
    }

    /// Create execution schedule
    pub fn create_schedule(&self, graph: &ComputationGraph) -> Result<ExecutionSchedule> {
        let decision = match self.policy {
            SchedulingPolicy::Performance => self.schedule_for_performance(graph)?,
            SchedulingPolicy::PowerSaving => self.schedule_for_power_saving(graph)?,
            SchedulingPolicy::Balanced => self.schedule_balanced(graph)?,
            SchedulingPolicy::Adaptive => self.schedule_adaptive(graph)?,
        };

        Ok(self.decision_to_schedule(decision))
    }

    /// Schedule for maximum performance
    fn schedule_for_performance(&self, graph: &ComputationGraph) -> Result<SchedulingDecision> {
        // Topological sort for dependency ordering
        let execution_order = self.topological_sort(graph)?;

        // All operators run at high performance
        let mut power_modes = HashMap::new();
        for &op_id in &execution_order {
            power_modes.insert(op_id, PowerMode::HighPerformance);
        }

        // Estimate metrics
        let (time_ms, power_watts, temp_rise) =
            self.estimate_execution_metrics(graph, &execution_order, &power_modes)?;

        Ok(SchedulingDecision {
            execution_order,
            power_modes,
            estimated_time_ms: time_ms,
            estimated_power_watts: power_watts,
            estimated_temp_rise: temp_rise,
        })
    }

    /// Schedule for minimum power consumption
    fn schedule_for_power_saving(&self, graph: &ComputationGraph) -> Result<SchedulingDecision> {
        let execution_order = self.topological_sort(graph)?;

        // All operators run in power saving mode
        let mut power_modes = HashMap::new();
        for &op_id in &execution_order {
            power_modes.insert(op_id, PowerMode::PowerSaving);
        }

        // Reorder operations to minimize peak power
        let reordered = self.reorder_for_power(&execution_order, graph)?;

        let (time_ms, power_watts, temp_rise) =
            self.estimate_execution_metrics(graph, &reordered, &power_modes)?;

        Ok(SchedulingDecision {
            execution_order: reordered,
            power_modes,
            estimated_time_ms: time_ms,
            estimated_power_watts: power_watts,
            estimated_temp_rise: temp_rise,
        })
    }

    /// Schedule with balanced approach
    fn schedule_balanced(&self, graph: &ComputationGraph) -> Result<SchedulingDecision> {
        let execution_order = self.topological_sort(graph)?;
        let mut power_modes = HashMap::new();

        // Assign power modes based on operation characteristics
        for &op_id in &execution_order {
            let op = &graph.operators[op_id];
            let mode = self.select_power_mode_for_op(op)?;
            power_modes.insert(op_id, mode);
        }

        // Optimize order for thermal management
        let optimized_order = self.optimize_for_thermal(&execution_order, graph, &power_modes)?;

        let (time_ms, power_watts, temp_rise) =
            self.estimate_execution_metrics(graph, &optimized_order, &power_modes)?;

        Ok(SchedulingDecision {
            execution_order: optimized_order,
            power_modes,
            estimated_time_ms: time_ms,
            estimated_power_watts: power_watts,
            estimated_temp_rise: temp_rise,
        })
    }

    /// Adaptive scheduling based on device state
    fn schedule_adaptive(&self, graph: &ComputationGraph) -> Result<SchedulingDecision> {
        // Choose policy based on current conditions
        let effective_policy = self.determine_adaptive_policy();

        match effective_policy {
            SchedulingPolicy::Performance => self.schedule_for_performance(graph),
            SchedulingPolicy::PowerSaving => self.schedule_for_power_saving(graph),
            SchedulingPolicy::Balanced => self.schedule_balanced(graph),
            SchedulingPolicy::Adaptive => {
                // Fallback to balanced scheduling for recursive adaptive case
                self.schedule_balanced(graph)
            },
        }
    }

    /// Determine adaptive policy based on device state
    fn determine_adaptive_policy(&self) -> SchedulingPolicy {
        let temp = self.thermal_constraints.current_temperature;
        let battery = self.battery_constraints.current_battery_level;
        let charging = self.battery_constraints.is_charging;

        // High temperature - prioritize cooling
        if temp > self.thermal_constraints.max_temperature * 0.9 {
            return SchedulingPolicy::PowerSaving;
        }

        // Low battery and not charging - save power
        if battery < 20.0 && !charging {
            return SchedulingPolicy::PowerSaving;
        }

        // Charging or high battery - performance mode
        if charging || battery > 80.0 {
            return SchedulingPolicy::Performance;
        }

        // Default to balanced
        SchedulingPolicy::Balanced
    }

    /// Select power mode for specific operation
    fn select_power_mode_for_op(&self, op: &GraphOperator) -> Result<PowerMode> {
        let op_name = format!("{:?}", op.kernel);
        let power_consumption = self.power_profile.operation_power.get(&op_name).unwrap_or(&1.0);

        // High power operations get throttled if needed
        if *power_consumption > 2.0 && self.should_throttle() {
            Ok(PowerMode::PowerSaving)
        } else if *power_consumption < 0.5 {
            // Low power operations can run at full speed
            Ok(PowerMode::HighPerformance)
        } else {
            Ok(PowerMode::Balanced)
        }
    }

    /// Check if throttling is needed
    fn should_throttle(&self) -> bool {
        let temp_ratio =
            self.thermal_constraints.current_temperature / self.thermal_constraints.max_temperature;
        let battery_low = self.battery_constraints.current_battery_level < 30.0;

        temp_ratio > 0.8 || (battery_low && !self.battery_constraints.is_charging)
    }

    /// Topological sort of graph
    fn topological_sort(&self, graph: &ComputationGraph) -> Result<Vec<usize>> {
        let n = graph.operators.len();
        let mut in_degree = vec![0; n];
        let mut adj_list: Vec<Vec<usize>> = vec![vec![]; n];

        // Build adjacency list
        for edge in &graph.edges {
            adj_list[edge.from].push(edge.to);
            in_degree[edge.to] += 1;
        }

        // Find nodes with no dependencies
        let mut queue = VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        let mut sorted = Vec::new();

        while let Some(node) = queue.pop_front() {
            sorted.push(node);

            for &neighbor in &adj_list[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        if sorted.len() != n {
            return Err(TrustformersError::runtime_error("Graph contains cycles".into()).into());
        }

        Ok(sorted)
    }

    /// Reorder operations to minimize peak power
    fn reorder_for_power(&self, order: &[usize], graph: &ComputationGraph) -> Result<Vec<usize>> {
        // Simple heuristic: interleave high and low power operations
        let mut high_power = Vec::new();
        let mut low_power = Vec::new();

        for &op_id in order {
            let op = &graph.operators[op_id];
            let op_name = format!("{:?}", op.kernel);
            let power = self.power_profile.operation_power.get(&op_name).unwrap_or(&1.0);

            if *power > 1.5 {
                high_power.push(op_id);
            } else {
                low_power.push(op_id);
            }
        }

        // Interleave high and low power operations
        let mut reordered = Vec::new();
        let mut high_idx = 0;
        let mut low_idx = 0;

        while high_idx < high_power.len() || low_idx < low_power.len() {
            if low_idx < low_power.len() {
                reordered.push(low_power[low_idx]);
                low_idx += 1;
            }

            if high_idx < high_power.len() {
                reordered.push(high_power[high_idx]);
                high_idx += 1;
            }
        }

        Ok(reordered)
    }

    /// Optimize order for thermal management
    fn optimize_for_thermal(
        &self,
        order: &[usize],
        graph: &ComputationGraph,
        power_modes: &HashMap<usize, PowerMode>,
    ) -> Result<Vec<usize>> {
        // Insert cooling periods between high-power operations
        let mut optimized = Vec::new();
        let mut accumulated_heat = 0.0;

        for &op_id in order {
            let op = &graph.operators[op_id];
            let heat_generated =
                self.estimate_heat_generation(op, power_modes.get(&op_id).unwrap())?;

            // Check if we need a cooling period
            if accumulated_heat + heat_generated > 5.0 {
                // Would benefit from a break - but we can't insert delays in inference
                // Instead, we might run the next operation in power-saving mode
                accumulated_heat *= self.power_profile.thermal_coefficients.cooling_rate;
            }

            optimized.push(op_id);
            accumulated_heat += heat_generated;
        }

        Ok(optimized)
    }

    /// Estimate heat generation for an operation
    fn estimate_heat_generation(&self, op: &GraphOperator, mode: &PowerMode) -> Result<f32> {
        let op_name = format!("{:?}", op.kernel);
        let base_power = self.power_profile.operation_power.get(&op_name).unwrap_or(&1.0);

        let power_multiplier = match mode {
            PowerMode::HighPerformance => 1.5,
            PowerMode::Balanced => 1.0,
            PowerMode::PowerSaving => 0.6,
        };

        let power = base_power * power_multiplier;
        let heat = power * self.power_profile.thermal_coefficients.heat_per_watt;

        Ok(heat)
    }

    /// Estimate execution metrics
    fn estimate_execution_metrics(
        &self,
        graph: &ComputationGraph,
        order: &[usize],
        power_modes: &HashMap<usize, PowerMode>,
    ) -> Result<(f32, f32, f32)> {
        let mut total_time = 0.0;
        let mut total_energy = 0.0;
        let mut peak_power: f32 = 0.0;
        let mut total_heat = 0.0;

        for &op_id in order {
            let op = &graph.operators[op_id];
            let mode = power_modes.get(&op_id).unwrap();

            // Estimate operation time
            let base_time = self.estimate_op_time(op)?;
            let time_multiplier = match mode {
                PowerMode::HighPerformance => 0.7,
                PowerMode::Balanced => 1.0,
                PowerMode::PowerSaving => 1.5,
            };
            let op_time = base_time * time_multiplier;

            // Estimate power consumption
            let op_name = format!("{:?}", op.kernel);
            let base_power = self.power_profile.operation_power.get(&op_name).unwrap_or(&1.0);
            let power_multiplier = match mode {
                PowerMode::HighPerformance => 1.5,
                PowerMode::Balanced => 1.0,
                PowerMode::PowerSaving => 0.6,
            };
            let op_power = base_power * power_multiplier;

            total_time += op_time;
            total_energy += op_power * op_time;
            peak_power = peak_power.max(op_power);
            total_heat += self.estimate_heat_generation(op, mode)?;
        }

        let avg_power = total_energy / total_time;
        let temp_rise =
            total_heat - (total_heat * self.power_profile.thermal_coefficients.cooling_rate);

        Ok((total_time * 1000.0, avg_power, temp_rise))
    }

    /// Estimate operation execution time
    fn estimate_op_time(&self, op: &GraphOperator) -> Result<f32> {
        // Simplified estimation based on operation type and size
        let total_elements: usize = op.output_shape.iter().product();
        let flops_per_element = match op.kernel {
            KernelType::Conv2d => 100.0,
            KernelType::Linear => 10.0,
            KernelType::Attention => 50.0,
            KernelType::Activation => 1.0,
            _ => 5.0,
        };

        let total_flops = total_elements as f32 * flops_per_element;
        let gflops_per_sec = 10.0; // Typical mobile GPU performance

        Ok(total_flops / (gflops_per_sec * 1e9))
    }

    /// Convert decision to execution schedule
    fn decision_to_schedule(&self, decision: SchedulingDecision) -> ExecutionSchedule {
        let power_hints: Vec<PowerHint> = decision
            .power_modes
            .into_iter()
            .map(|(op_id, mode)| PowerHint {
                operator_id: op_id,
                power_mode: mode,
            })
            .collect();

        ExecutionSchedule {
            operator_order: decision.execution_order,
            power_hints,
        }
    }

    /// Create default power profile
    fn create_default_power_profile() -> PowerProfile {
        let mut operation_power = HashMap::new();

        // Typical power consumption in Watts
        operation_power.insert("Conv2d".to_string(), 2.0);
        operation_power.insert("Linear".to_string(), 1.5);
        operation_power.insert("BatchNorm".to_string(), 0.5);
        operation_power.insert("Activation".to_string(), 0.3);
        operation_power.insert("Attention".to_string(), 2.5);
        operation_power.insert("Pooling".to_string(), 0.4);

        PowerProfile {
            operation_power,
            thermal_coefficients: ThermalCoefficients {
                heat_per_watt: 1.2,
                cooling_rate: 0.9,
                ambient_temp: 25.0,
            },
            battery_model: BatteryModel {
                capacity_mah: 3000.0,
                voltage: 3.7,
                discharge_curve: vec![(0.5, 1.0), (1.0, 1.1), (2.0, 1.3), (3.0, 1.5)],
            },
        }
    }

    /// Create default thermal constraints
    fn create_default_thermal_constraints() -> ThermalConstraints {
        ThermalConstraints {
            max_temperature: 60.0,
            throttle_thresholds: vec![(45.0, 1.0), (50.0, 0.8), (55.0, 0.6), (58.0, 0.4)],
            current_temperature: 35.0,
        }
    }

    /// Create default battery constraints
    fn create_default_battery_constraints() -> BatteryConstraints {
        BatteryConstraints {
            min_battery_level: 10.0,
            current_battery_level: 50.0,
            is_charging: false,
            power_budget: 3.0, // 3W typical for mobile
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::{ComputationGraph, Edge, GraphOperator, KernelType};

    #[test]
    fn test_scheduler_creation() {
        let config = crate::MobileConfig::default();
        let scheduler = PowerAwareScheduler::new(config);

        assert_eq!(scheduler.policy, SchedulingPolicy::Balanced);
    }

    #[test]
    fn test_topological_sort() {
        let graph = ComputationGraph {
            operators: vec![
                GraphOperator {
                    id: 0,
                    kernel: KernelType::Linear,
                    inputs: vec!["input".to_string()],
                    outputs: vec!["linear_out".to_string()],
                    input_shapes: vec![vec![10]],
                    output_shape: vec![10],
                    cache_hints: None,
                },
                GraphOperator {
                    id: 1,
                    kernel: KernelType::Activation,
                    inputs: vec!["linear_out".to_string()],
                    outputs: vec!["output".to_string()],
                    input_shapes: vec![vec![10]],
                    output_shape: vec![10],
                    cache_hints: None,
                },
            ],
            edges: vec![Edge {
                from: 0,
                to: 1,
                tensor_name: "linear_out".to_string(),
            }],
        };

        let config = crate::MobileConfig::default();
        let scheduler = PowerAwareScheduler::new(config);

        let sorted = scheduler.topological_sort(&graph).unwrap();
        assert_eq!(sorted, vec![0, 1]);
    }

    #[test]
    fn test_adaptive_policy() {
        let config = crate::MobileConfig::default();
        let mut scheduler = PowerAwareScheduler::new(config);

        // High temperature
        scheduler.update_thermal_state(55.0);
        let policy = scheduler.determine_adaptive_policy();
        assert_eq!(policy, SchedulingPolicy::PowerSaving);

        // Low battery
        scheduler.update_thermal_state(30.0);
        scheduler.update_battery_state(15.0, false);
        let policy = scheduler.determine_adaptive_policy();
        assert_eq!(policy, SchedulingPolicy::PowerSaving);

        // Charging
        scheduler.update_battery_state(50.0, true);
        let policy = scheduler.determine_adaptive_policy();
        assert_eq!(policy, SchedulingPolicy::Performance);
    }
}
