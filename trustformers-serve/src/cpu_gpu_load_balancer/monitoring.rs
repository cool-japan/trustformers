//! CPU/GPU Load Balancer Monitoring and Metrics
//!
//! This module provides comprehensive monitoring, statistics collection,
//! and performance metrics for the CPU/GPU load balancing system.

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

use super::power::PowerEfficiencyMetrics;
use super::types::ProcessorType;

/// Load balancer statistics
///
/// Comprehensive statistics for load balancer performance and efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerStats {
    /// Total tasks processed
    pub total_tasks: u64,

    /// Tasks assigned to CPU
    pub cpu_tasks: u64,

    /// Tasks assigned to GPU
    pub gpu_tasks: u64,

    /// Average CPU utilization
    pub avg_cpu_utilization: f32,

    /// Average GPU utilization
    pub avg_gpu_utilization: f32,

    /// Total energy consumed
    pub total_energy_consumed: f32,

    /// Average task latency
    pub avg_task_latency: Duration,

    /// Throughput (tasks/sec)
    pub throughput: f64,

    /// Load balancing efficiency
    pub load_balance_efficiency: f32,

    /// Task success rate
    pub success_rate: f32,

    /// Processor statistics
    pub processor_stats: HashMap<ProcessorType, ProcessorStats>,

    /// Power efficiency metrics
    pub power_efficiency_metrics: PowerEfficiencyMetrics,
}

/// Processor-specific statistics
///
/// Detailed performance metrics for individual processor types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorStats {
    /// Tasks executed
    pub tasks_executed: u64,

    /// Average execution time
    pub avg_execution_time: Duration,

    /// Average utilization
    pub avg_utilization: f32,

    /// Energy efficiency
    pub energy_efficiency: f32,

    /// Success rate
    pub success_rate: f32,

    /// Throughput
    pub throughput: f64,
}

/// Power efficiency report
///
/// Comprehensive power efficiency analysis and recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerEfficiencyReport {
    /// Overall power efficiency score (0-1)
    pub overall_score: f32,

    /// CPU power efficiency
    pub cpu_efficiency: f32,

    /// GPU power efficiency
    pub gpu_efficiency: f32,

    /// Total power savings achieved
    pub power_savings: f32,

    /// Energy cost reduction percentage
    pub cost_reduction: f32,

    /// Recommended optimizations
    pub recommendations: Vec<String>,

    /// Power usage breakdown
    pub power_breakdown: HashMap<String, f32>,

    /// Efficiency trends
    pub efficiency_trends: Vec<EfficiencyTrendPoint>,
}

/// Efficiency trend data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyTrendPoint {
    /// Timestamp (minutes since start)
    pub timestamp: u64,

    /// Efficiency score at this point
    pub efficiency: f32,

    /// Power consumption at this point
    pub power_consumption: f32,

    /// Throughput at this point
    pub throughput: f64,
}

/// Performance monitoring interface
pub trait PerformanceMonitor {
    /// Update statistics with new task execution data
    fn update_task_stats(
        &mut self,
        processor_type: ProcessorType,
        execution_time: Duration,
        success: bool,
    );

    /// Update resource utilization metrics
    fn update_utilization(&mut self, processor_type: ProcessorType, utilization: f32);

    /// Update power consumption metrics
    fn update_power_consumption(&mut self, processor_type: ProcessorType, power: f32);

    /// Get current statistics
    fn get_stats(&self) -> &LoadBalancerStats;

    /// Generate efficiency report
    fn generate_efficiency_report(&self) -> PowerEfficiencyReport;

    /// Reset statistics
    fn reset_stats(&mut self);
}

/// Default monitoring implementation
pub struct DefaultPerformanceMonitor {
    stats: LoadBalancerStats,
    start_time: std::time::Instant,
    efficiency_history: Vec<EfficiencyTrendPoint>,
}

impl DefaultPerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            stats: LoadBalancerStats::default(),
            start_time: std::time::Instant::now(),
            efficiency_history: Vec::new(),
        }
    }

    /// Calculate load balancing efficiency
    fn calculate_load_balance_efficiency(&self) -> f32 {
        if self.stats.total_tasks == 0 {
            return 1.0;
        }

        let cpu_ratio = self.stats.cpu_tasks as f32 / self.stats.total_tasks as f32;
        let gpu_ratio = self.stats.gpu_tasks as f32 / self.stats.total_tasks as f32;

        // Ideal balance is task-dependent, but we aim for reasonable distribution
        let ideal_cpu_ratio = 0.4;
        let ideal_gpu_ratio = 0.6;

        let cpu_efficiency = 1.0 - (cpu_ratio - ideal_cpu_ratio).abs();
        let gpu_efficiency = 1.0 - (gpu_ratio - ideal_gpu_ratio).abs();

        (cpu_efficiency + gpu_efficiency) / 2.0
    }

    /// Update efficiency trends
    fn update_efficiency_trends(&mut self) {
        let elapsed_minutes = self.start_time.elapsed().as_secs() / 60;
        let current_efficiency = self.calculate_load_balance_efficiency();
        let current_power = self.stats.power_efficiency_metrics.avg_power_consumption;

        self.efficiency_history.push(EfficiencyTrendPoint {
            timestamp: elapsed_minutes,
            efficiency: current_efficiency,
            power_consumption: current_power,
            throughput: self.stats.throughput,
        });

        // Keep only last 24 hours of data (1440 minutes)
        if self.efficiency_history.len() > 1440 {
            self.efficiency_history.remove(0);
        }
    }
}

impl PerformanceMonitor for DefaultPerformanceMonitor {
    fn update_task_stats(
        &mut self,
        processor_type: ProcessorType,
        execution_time: Duration,
        success: bool,
    ) {
        self.stats.total_tasks += 1;

        match processor_type {
            ProcessorType::CPU => self.stats.cpu_tasks += 1,
            ProcessorType::GPU => self.stats.gpu_tasks += 1,
        }

        // Update processor-specific stats
        let proc_stats = self
            .stats
            .processor_stats
            .entry(processor_type)
            .or_insert_with(ProcessorStats::default);

        proc_stats.tasks_executed += 1;

        // Update running averages
        let task_count = proc_stats.tasks_executed as f32;
        proc_stats.avg_execution_time = Duration::from_secs_f32(
            (proc_stats.avg_execution_time.as_secs_f32() * (task_count - 1.0)
                + execution_time.as_secs_f32())
                / task_count,
        );

        if success {
            proc_stats.success_rate =
                (proc_stats.success_rate * (task_count - 1.0) + 1.0) / task_count;
        } else {
            proc_stats.success_rate = (proc_stats.success_rate * (task_count - 1.0)) / task_count;
        }

        // Update global success rate
        let total_tasks = self.stats.total_tasks as f32;
        if success {
            self.stats.success_rate =
                (self.stats.success_rate * (total_tasks - 1.0) + 1.0) / total_tasks;
        } else {
            self.stats.success_rate = (self.stats.success_rate * (total_tasks - 1.0)) / total_tasks;
        }

        // Update global latency
        self.stats.avg_task_latency = Duration::from_secs_f32(
            (self.stats.avg_task_latency.as_secs_f32() * (total_tasks - 1.0)
                + execution_time.as_secs_f32())
                / total_tasks,
        );

        // Update load balancing efficiency
        self.stats.load_balance_efficiency = self.calculate_load_balance_efficiency();

        // Update efficiency trends every 5 tasks
        if self.stats.total_tasks % 5 == 0 {
            self.update_efficiency_trends();
        }
    }

    fn update_utilization(&mut self, processor_type: ProcessorType, utilization: f32) {
        match processor_type {
            ProcessorType::CPU => {
                self.stats.avg_cpu_utilization =
                    (self.stats.avg_cpu_utilization * 0.9 + utilization * 0.1).min(1.0);
            },
            ProcessorType::GPU => {
                self.stats.avg_gpu_utilization =
                    (self.stats.avg_gpu_utilization * 0.9 + utilization * 0.1).min(1.0);
            },
        }

        // Update processor-specific utilization
        if let Some(proc_stats) = self.stats.processor_stats.get_mut(&processor_type) {
            proc_stats.avg_utilization =
                (proc_stats.avg_utilization * 0.9 + utilization * 0.1).min(1.0);
        }
    }

    fn update_power_consumption(&mut self, _processor_type: ProcessorType, power: f32) {
        // Update power efficiency metrics
        let power_metrics = &mut self.stats.power_efficiency_metrics;
        power_metrics.avg_power_consumption =
            (power_metrics.avg_power_consumption * 0.9 + power * 0.1).max(0.0);

        if power > power_metrics.peak_power_consumption {
            power_metrics.peak_power_consumption = power;
        }

        // Update total energy consumed (approximate)
        power_metrics.total_energy_consumed += power * 0.001; // Assuming 1ms updates

        // Calculate energy efficiency
        if power > 0.0 {
            let ops_per_second = self.stats.throughput;
            power_metrics.energy_efficiency = ops_per_second as f32 / power;
        }
    }

    fn get_stats(&self) -> &LoadBalancerStats {
        &self.stats
    }

    fn generate_efficiency_report(&self) -> PowerEfficiencyReport {
        let power_metrics = &self.stats.power_efficiency_metrics;
        let cpu_stats = self.stats.processor_stats.get(&ProcessorType::CPU);
        let gpu_stats = self.stats.processor_stats.get(&ProcessorType::GPU);

        let cpu_efficiency = cpu_stats.map(|s| s.energy_efficiency).unwrap_or(0.0);
        let gpu_efficiency = gpu_stats.map(|s| s.energy_efficiency).unwrap_or(0.0);

        let overall_score = if cpu_efficiency + gpu_efficiency > 0.0 {
            (cpu_efficiency + gpu_efficiency) / 2.0 / 100.0 // Normalize to 0-1
        } else {
            0.0
        };

        let mut recommendations = Vec::new();
        if self.stats.avg_cpu_utilization > 0.9 {
            recommendations.push("Consider offloading more tasks to GPU".to_string());
        }
        if self.stats.avg_gpu_utilization < 0.3 {
            recommendations
                .push("GPU is underutilized, consider optimizing task assignment".to_string());
        }
        if power_metrics.avg_power_consumption > 800.0 {
            recommendations
                .push("High power consumption detected, enable power efficiency mode".to_string());
        }

        let mut power_breakdown = HashMap::new();
        power_breakdown.insert("CPU".to_string(), self.stats.avg_cpu_utilization * 100.0);
        power_breakdown.insert("GPU".to_string(), self.stats.avg_gpu_utilization * 200.0); // GPUs typically use more power

        PowerEfficiencyReport {
            overall_score: overall_score.min(1.0),
            cpu_efficiency,
            gpu_efficiency,
            power_savings: 0.0,  // Would need baseline to calculate
            cost_reduction: 0.0, // Would need cost model
            recommendations,
            power_breakdown,
            efficiency_trends: self.efficiency_history.clone(),
        }
    }

    fn reset_stats(&mut self) {
        self.stats = LoadBalancerStats::default();
        self.start_time = std::time::Instant::now();
        self.efficiency_history.clear();
    }
}

/// Default implementations
impl Default for LoadBalancerStats {
    fn default() -> Self {
        Self {
            total_tasks: 0,
            cpu_tasks: 0,
            gpu_tasks: 0,
            avg_cpu_utilization: 0.0,
            avg_gpu_utilization: 0.0,
            total_energy_consumed: 0.0,
            avg_task_latency: Duration::from_millis(0),
            throughput: 0.0,
            load_balance_efficiency: 1.0,
            success_rate: 1.0,
            processor_stats: HashMap::new(),
            power_efficiency_metrics: PowerEfficiencyMetrics::default(),
        }
    }
}

impl Default for ProcessorStats {
    fn default() -> Self {
        Self {
            tasks_executed: 0,
            avg_execution_time: Duration::from_millis(0),
            avg_utilization: 0.0,
            energy_efficiency: 0.0,
            success_rate: 1.0,
            throughput: 0.0,
        }
    }
}

impl Default for PowerEfficiencyReport {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            cpu_efficiency: 0.0,
            gpu_efficiency: 0.0,
            power_savings: 0.0,
            cost_reduction: 0.0,
            recommendations: Vec::new(),
            power_breakdown: HashMap::new(),
            efficiency_trends: Vec::new(),
        }
    }
}

impl Default for DefaultPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}
