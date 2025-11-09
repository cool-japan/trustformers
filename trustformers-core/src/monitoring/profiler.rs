// Performance profiling utilities
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Model profiler for tracking performance metrics
#[derive(Debug, Clone)]
pub struct ModelProfiler {
    config: ProfilerConfig,
    active_sessions: HashMap<String, ProfilingSession>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    pub enabled: bool,
    pub track_layer_times: bool,
    pub track_memory_usage: bool,
    pub track_compute_utilization: bool,
    pub sample_interval_ms: u64,
    pub max_samples: usize,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            track_layer_times: true,
            track_memory_usage: true,
            track_compute_utilization: false, // Expensive
            sample_interval_ms: 10,
            max_samples: 10000,
        }
    }
}

#[derive(Debug, Clone)]
struct ProfilingSession {
    id: String,
    start_time: Instant,
    layer_timings: HashMap<String, Vec<Duration>>,
    operation_timings: HashMap<String, Vec<Duration>>,
    memory_samples: Vec<MemorySample>,
    compute_samples: Vec<ComputeSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySample {
    pub timestamp: Duration,
    pub cpu_usage_mb: f64,
    pub gpu_usage_mb: f64,
    pub peak_usage_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeSample {
    pub timestamp: Duration,
    pub cpu_utilization: f64,
    pub gpu_utilization: f64,
    pub memory_bandwidth: f64,
    pub flops: f64,
}

/// Complete profiling report for a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingReport {
    pub session_id: String,
    pub total_duration: Duration,
    pub layer_performance: LayerPerformanceReport,
    pub operation_performance: OperationPerformanceReport,
    pub memory_profile: MemoryProfile,
    pub compute_profile: ComputeProfile,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPerformanceReport {
    pub layer_timings: HashMap<String, LayerTiming>,
    pub total_layer_time: Duration,
    pub slowest_layers: Vec<(String, Duration)>,
    pub layer_efficiency: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTiming {
    pub layer_name: String,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub std_deviation: Duration,
    pub call_count: usize,
    pub total_time: Duration,
    pub percentage_of_total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationPerformanceReport {
    pub operation_timings: HashMap<String, OperationTiming>,
    pub total_operation_time: Duration,
    pub slowest_operations: Vec<(String, Duration)>,
    pub operation_efficiency: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationTiming {
    pub operation_name: String,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub std_deviation: Duration,
    pub call_count: usize,
    pub total_time: Duration,
    pub percentage_of_total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    pub peak_memory_usage: f64,
    pub average_memory_usage: f64,
    pub memory_efficiency: f64,
    pub memory_fragmentation: f64,
    pub memory_timeline: Vec<MemorySample>,
    pub allocation_patterns: Vec<AllocationPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPattern {
    pub pattern_type: String,
    pub frequency: usize,
    pub average_size_mb: f64,
    pub total_size_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeProfile {
    pub average_cpu_utilization: f64,
    pub average_gpu_utilization: f64,
    pub peak_flops: f64,
    pub average_flops: f64,
    pub compute_efficiency: f64,
    pub utilization_timeline: Vec<ComputeSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub primary_bottleneck: BottleneckType,
    pub bottleneck_severity: f64,
    pub affected_operations: Vec<String>,
    pub bottleneck_timeline: Vec<BottleneckEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    Memory,
    Compute,
    IO,
    Network,
    Synchronization,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckEvent {
    pub timestamp: Duration,
    pub bottleneck_type: BottleneckType,
    pub severity: f64,
    pub duration: Duration,
    pub affected_operation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub suggestion_type: OptimizationType,
    pub priority: OptimizationPriority,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_complexity: ComplexityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    MemoryOptimization,
    ComputeOptimization,
    ArchitecturalChange,
    AlgorithmicImprovement,
    HardwareUtilization,
    DataLayout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

impl Default for ProfilingReport {
    fn default() -> Self {
        Self {
            session_id: String::new(),
            total_duration: Duration::from_secs(0),
            layer_performance: LayerPerformanceReport::default(),
            operation_performance: OperationPerformanceReport::default(),
            memory_profile: MemoryProfile::default(),
            compute_profile: ComputeProfile::default(),
            bottleneck_analysis: BottleneckAnalysis::default(),
            optimization_suggestions: Vec::new(),
        }
    }
}

impl Default for LayerPerformanceReport {
    fn default() -> Self {
        Self {
            layer_timings: HashMap::new(),
            total_layer_time: Duration::from_secs(0),
            slowest_layers: Vec::new(),
            layer_efficiency: HashMap::new(),
        }
    }
}

impl Default for OperationPerformanceReport {
    fn default() -> Self {
        Self {
            operation_timings: HashMap::new(),
            total_operation_time: Duration::from_secs(0),
            slowest_operations: Vec::new(),
            operation_efficiency: HashMap::new(),
        }
    }
}

impl Default for MemoryProfile {
    fn default() -> Self {
        Self {
            peak_memory_usage: 0.0,
            average_memory_usage: 0.0,
            memory_efficiency: 0.0,
            memory_fragmentation: 0.0,
            memory_timeline: Vec::new(),
            allocation_patterns: Vec::new(),
        }
    }
}

impl Default for ComputeProfile {
    fn default() -> Self {
        Self {
            average_cpu_utilization: 0.0,
            average_gpu_utilization: 0.0,
            peak_flops: 0.0,
            average_flops: 0.0,
            compute_efficiency: 0.0,
            utilization_timeline: Vec::new(),
        }
    }
}

impl Default for BottleneckAnalysis {
    fn default() -> Self {
        Self {
            primary_bottleneck: BottleneckType::None,
            bottleneck_severity: 0.0,
            affected_operations: Vec::new(),
            bottleneck_timeline: Vec::new(),
        }
    }
}

impl Default for ModelProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelProfiler {
    pub fn new() -> Self {
        Self {
            config: ProfilerConfig::default(),
            active_sessions: HashMap::new(),
        }
    }

    pub fn with_config(config: ProfilerConfig) -> Self {
        Self {
            config,
            active_sessions: HashMap::new(),
        }
    }

    /// Start profiling a session
    pub fn start_profiling(&mut self, session_id: &str) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let session = ProfilingSession {
            id: session_id.to_string(),
            start_time: Instant::now(),
            layer_timings: HashMap::new(),
            operation_timings: HashMap::new(),
            memory_samples: Vec::new(),
            compute_samples: Vec::new(),
        };

        self.active_sessions.insert(session_id.to_string(), session);
        Ok(())
    }

    /// Profile a layer execution
    pub fn profile_layer<T, F>(
        &mut self,
        session_id: &str,
        layer_name: &str,
        operation: F,
    ) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        if !self.config.enabled {
            return operation();
        }

        let start_time = Instant::now();
        let result = operation()?;
        let duration = start_time.elapsed();

        if let Some(session) = self.active_sessions.get_mut(session_id) {
            session.layer_timings.entry(layer_name.to_string()).or_default().push(duration);
        }

        Ok(result)
    }

    /// Profile an operation execution
    pub fn profile_operation<T, F>(
        &mut self,
        session_id: &str,
        operation_name: &str,
        operation: F,
    ) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        if !self.config.enabled {
            return operation();
        }

        let start_time = Instant::now();
        let result = operation()?;
        let duration = start_time.elapsed();

        if let Some(session) = self.active_sessions.get_mut(session_id) {
            session
                .operation_timings
                .entry(operation_name.to_string())
                .or_default()
                .push(duration);
        }

        Ok(result)
    }

    /// Take a memory sample
    pub fn sample_memory(&mut self, session_id: &str) -> Result<()> {
        if !self.config.enabled || !self.config.track_memory_usage {
            return Ok(());
        }

        let timestamp = if let Some(session) = self.active_sessions.get(session_id) {
            session.start_time.elapsed()
        } else {
            return Ok(());
        };

        let sample = self.get_memory_sample(timestamp)?;

        if let Some(session) = self.active_sessions.get_mut(session_id) {
            if session.memory_samples.len() < self.config.max_samples {
                session.memory_samples.push(sample);
            }
        }

        Ok(())
    }

    /// Take a compute sample
    pub fn sample_compute(&mut self, session_id: &str) -> Result<()> {
        if !self.config.enabled || !self.config.track_compute_utilization {
            return Ok(());
        }

        let timestamp = if let Some(session) = self.active_sessions.get(session_id) {
            session.start_time.elapsed()
        } else {
            return Ok(());
        };

        let sample = self.get_compute_sample(timestamp)?;

        if let Some(session) = self.active_sessions.get_mut(session_id) {
            if session.compute_samples.len() < self.config.max_samples {
                session.compute_samples.push(sample);
            }
        }

        Ok(())
    }

    /// End profiling and generate report
    pub fn end_profiling(&mut self, session_id: &str) -> Result<ProfilingReport> {
        let session = self
            .active_sessions
            .remove(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        let total_duration = session.start_time.elapsed();

        let layer_performance = self.analyze_layer_performance(&session, total_duration)?;
        let operation_performance = self.analyze_operation_performance(&session, total_duration)?;
        let memory_profile = self.analyze_memory_profile(&session)?;
        let compute_profile = self.analyze_compute_profile(&session)?;
        let bottleneck_analysis =
            self.analyze_bottlenecks(&session, &layer_performance, &operation_performance)?;
        let optimization_suggestions = self.generate_optimization_suggestions(
            &bottleneck_analysis,
            &memory_profile,
            &compute_profile,
        )?;

        Ok(ProfilingReport {
            session_id: session.id,
            total_duration,
            layer_performance,
            operation_performance,
            memory_profile,
            compute_profile,
            bottleneck_analysis,
            optimization_suggestions,
        })
    }

    /// Clear all profiling data
    pub fn clear(&mut self) -> Result<()> {
        self.active_sessions.clear();
        Ok(())
    }

    /// Get current memory sample
    fn get_memory_sample(&self, timestamp: Duration) -> Result<MemorySample> {
        // Simplified implementation - would use actual system monitoring
        Ok(MemorySample {
            timestamp,
            cpu_usage_mb: 1024.0 + (timestamp.as_millis() as f64 * 0.1) % 512.0,
            gpu_usage_mb: 2048.0 + (timestamp.as_millis() as f64 * 0.05) % 1024.0,
            peak_usage_mb: 3072.0,
        })
    }

    /// Get current compute sample
    fn get_compute_sample(&self, timestamp: Duration) -> Result<ComputeSample> {
        // Simplified implementation - would use actual system monitoring
        let phase = (timestamp.as_millis() as f64 * 0.01) % (2.0 * std::f64::consts::PI);

        Ok(ComputeSample {
            timestamp,
            cpu_utilization: 0.6 + 0.3 * phase.sin(),
            gpu_utilization: 0.8 + 0.2 * phase.cos(),
            memory_bandwidth: 200.0 + 50.0 * phase.sin(),
            flops: 1000.0 + 200.0 * phase.cos(),
        })
    }

    /// Analyze layer performance
    fn analyze_layer_performance(
        &self,
        session: &ProfilingSession,
        total_duration: Duration,
    ) -> Result<LayerPerformanceReport> {
        let mut layer_timings = HashMap::new();
        let mut total_layer_time = Duration::from_secs(0);
        let mut slowest_layers = Vec::new();
        let mut layer_efficiency = HashMap::new();

        for (layer_name, timings) in &session.layer_timings {
            let total_time: Duration = timings.iter().sum();
            let average_time = total_time / timings.len() as u32;
            let min_time = *timings.iter().min().unwrap_or(&Duration::from_secs(0));
            let max_time = *timings.iter().max().unwrap_or(&Duration::from_secs(0));

            // Calculate standard deviation
            let mean_nanos = average_time.as_nanos() as f64;
            let variance = timings
                .iter()
                .map(|t| {
                    let diff = t.as_nanos() as f64 - mean_nanos;
                    diff * diff
                })
                .sum::<f64>()
                / timings.len() as f64;
            let std_dev_nanos = variance.sqrt() as u64;
            let std_deviation = Duration::from_nanos(std_dev_nanos);

            let percentage_of_total = if total_duration.as_nanos() > 0 {
                (total_time.as_nanos() as f64 / total_duration.as_nanos() as f64) * 100.0
            } else {
                0.0
            };

            layer_timings.insert(
                layer_name.clone(),
                LayerTiming {
                    layer_name: layer_name.clone(),
                    average_time,
                    min_time,
                    max_time,
                    std_deviation,
                    call_count: timings.len(),
                    total_time,
                    percentage_of_total,
                },
            );

            total_layer_time += total_time;
            slowest_layers.push((layer_name.clone(), total_time));

            // Calculate efficiency (inverse of coefficient of variation)
            let efficiency = if std_dev_nanos > 0 && mean_nanos > 0.0 {
                1.0 / (std_dev_nanos as f64 / mean_nanos)
            } else {
                1.0
            };
            layer_efficiency.insert(layer_name.clone(), efficiency);
        }

        // Sort slowest layers by total time
        slowest_layers.sort_by(|a, b| b.1.cmp(&a.1));
        slowest_layers.truncate(10); // Keep top 10

        Ok(LayerPerformanceReport {
            layer_timings,
            total_layer_time,
            slowest_layers,
            layer_efficiency,
        })
    }

    /// Analyze operation performance
    fn analyze_operation_performance(
        &self,
        session: &ProfilingSession,
        total_duration: Duration,
    ) -> Result<OperationPerformanceReport> {
        let mut operation_timings = HashMap::new();
        let mut total_operation_time = Duration::from_secs(0);
        let mut slowest_operations = Vec::new();
        let mut operation_efficiency = HashMap::new();

        for (operation_name, timings) in &session.operation_timings {
            let total_time: Duration = timings.iter().sum();
            let average_time = total_time / timings.len() as u32;
            let min_time = *timings.iter().min().unwrap_or(&Duration::from_secs(0));
            let max_time = *timings.iter().max().unwrap_or(&Duration::from_secs(0));

            // Calculate standard deviation
            let mean_nanos = average_time.as_nanos() as f64;
            let variance = timings
                .iter()
                .map(|t| {
                    let diff = t.as_nanos() as f64 - mean_nanos;
                    diff * diff
                })
                .sum::<f64>()
                / timings.len() as f64;
            let std_dev_nanos = variance.sqrt() as u64;
            let std_deviation = Duration::from_nanos(std_dev_nanos);

            let percentage_of_total = if total_duration.as_nanos() > 0 {
                (total_time.as_nanos() as f64 / total_duration.as_nanos() as f64) * 100.0
            } else {
                0.0
            };

            operation_timings.insert(
                operation_name.clone(),
                OperationTiming {
                    operation_name: operation_name.clone(),
                    average_time,
                    min_time,
                    max_time,
                    std_deviation,
                    call_count: timings.len(),
                    total_time,
                    percentage_of_total,
                },
            );

            total_operation_time += total_time;
            slowest_operations.push((operation_name.clone(), total_time));

            // Calculate efficiency
            let efficiency = if std_dev_nanos > 0 && mean_nanos > 0.0 {
                1.0 / (std_dev_nanos as f64 / mean_nanos)
            } else {
                1.0
            };
            operation_efficiency.insert(operation_name.clone(), efficiency);
        }

        // Sort slowest operations
        slowest_operations.sort_by(|a, b| b.1.cmp(&a.1));
        slowest_operations.truncate(10);

        Ok(OperationPerformanceReport {
            operation_timings,
            total_operation_time,
            slowest_operations,
            operation_efficiency,
        })
    }

    /// Analyze memory profile
    fn analyze_memory_profile(&self, session: &ProfilingSession) -> Result<MemoryProfile> {
        if session.memory_samples.is_empty() {
            return Ok(MemoryProfile::default());
        }

        let peak_memory_usage = session
            .memory_samples
            .iter()
            .map(|s| s.cpu_usage_mb.max(s.gpu_usage_mb))
            .fold(0.0, f64::max);

        let average_memory_usage = session
            .memory_samples
            .iter()
            .map(|s| s.cpu_usage_mb + s.gpu_usage_mb)
            .sum::<f64>()
            / session.memory_samples.len() as f64;

        let memory_efficiency = if peak_memory_usage > 0.0 {
            average_memory_usage / peak_memory_usage
        } else {
            0.0
        };

        let memory_fragmentation = 0.1; // Simplified calculation

        let allocation_patterns = vec![
            AllocationPattern {
                pattern_type: "Tensor".to_string(),
                frequency: 100,
                average_size_mb: 10.0,
                total_size_mb: 1000.0,
            },
            AllocationPattern {
                pattern_type: "Weight".to_string(),
                frequency: 50,
                average_size_mb: 20.0,
                total_size_mb: 1000.0,
            },
        ];

        Ok(MemoryProfile {
            peak_memory_usage,
            average_memory_usage,
            memory_efficiency,
            memory_fragmentation,
            memory_timeline: session.memory_samples.clone(),
            allocation_patterns,
        })
    }

    /// Analyze compute profile
    fn analyze_compute_profile(&self, session: &ProfilingSession) -> Result<ComputeProfile> {
        if session.compute_samples.is_empty() {
            return Ok(ComputeProfile::default());
        }

        let average_cpu_utilization =
            session.compute_samples.iter().map(|s| s.cpu_utilization).sum::<f64>()
                / session.compute_samples.len() as f64;

        let average_gpu_utilization =
            session.compute_samples.iter().map(|s| s.gpu_utilization).sum::<f64>()
                / session.compute_samples.len() as f64;

        let peak_flops = session.compute_samples.iter().map(|s| s.flops).fold(0.0, f64::max);

        let average_flops = session.compute_samples.iter().map(|s| s.flops).sum::<f64>()
            / session.compute_samples.len() as f64;

        let compute_efficiency = if peak_flops > 0.0 { average_flops / peak_flops } else { 0.0 };

        Ok(ComputeProfile {
            average_cpu_utilization,
            average_gpu_utilization,
            peak_flops,
            average_flops,
            compute_efficiency,
            utilization_timeline: session.compute_samples.clone(),
        })
    }

    /// Analyze bottlenecks
    fn analyze_bottlenecks(
        &self,
        _session: &ProfilingSession,
        layer_performance: &LayerPerformanceReport,
        _operation_performance: &OperationPerformanceReport,
    ) -> Result<BottleneckAnalysis> {
        let mut primary_bottleneck = BottleneckType::None;
        let mut bottleneck_severity = 0.0;
        let mut affected_operations = Vec::new();

        // Find the slowest layers as potential bottlenecks
        if let Some((slowest_layer, duration)) = layer_performance.slowest_layers.first() {
            let total_time = layer_performance.total_layer_time;
            if total_time.as_nanos() > 0 {
                let percentage =
                    (duration.as_nanos() as f64 / total_time.as_nanos() as f64) * 100.0;
                if percentage > 30.0 {
                    primary_bottleneck = BottleneckType::Compute;
                    bottleneck_severity = percentage / 100.0;
                    affected_operations.push(slowest_layer.clone());
                }
            }
        }

        Ok(BottleneckAnalysis {
            primary_bottleneck,
            bottleneck_severity,
            affected_operations,
            bottleneck_timeline: Vec::new(),
        })
    }

    /// Generate optimization suggestions
    fn generate_optimization_suggestions(
        &self,
        bottleneck_analysis: &BottleneckAnalysis,
        memory_profile: &MemoryProfile,
        compute_profile: &ComputeProfile,
    ) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        // Memory optimization suggestions
        if memory_profile.memory_efficiency < 0.7 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::MemoryOptimization,
                priority: OptimizationPriority::High,
                description: "Consider implementing gradient checkpointing to reduce memory usage"
                    .to_string(),
                expected_improvement: 0.3,
                implementation_complexity: ComplexityLevel::Medium,
            });
        }

        // Compute optimization suggestions
        if compute_profile.compute_efficiency < 0.6 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::ComputeOptimization,
                priority: OptimizationPriority::High,
                description:
                    "Improve compute efficiency with kernel fusion and better parallelization"
                        .to_string(),
                expected_improvement: 0.4,
                implementation_complexity: ComplexityLevel::High,
            });
        }

        // Bottleneck-specific suggestions
        match bottleneck_analysis.primary_bottleneck {
            BottleneckType::Memory => {
                suggestions.push(OptimizationSuggestion {
                    suggestion_type: OptimizationType::MemoryOptimization,
                    priority: OptimizationPriority::Critical,
                    description: "Memory bottleneck detected. Consider reducing batch size or using gradient accumulation".to_string(),
                    expected_improvement: 0.5,
                    implementation_complexity: ComplexityLevel::Low,
                });
            },
            BottleneckType::Compute => {
                suggestions.push(OptimizationSuggestion {
                    suggestion_type: OptimizationType::ComputeOptimization,
                    priority: OptimizationPriority::Critical,
                    description: "Compute bottleneck detected. Consider using mixed precision training or model parallelism".to_string(),
                    expected_improvement: 0.4,
                    implementation_complexity: ComplexityLevel::Medium,
                });
            },
            _ => {},
        }

        Ok(suggestions)
    }
}

impl ProfilingReport {
    /// Print a summary of the profiling report
    pub fn print_summary(&self) {
        println!("Profiling Report Summary");
        println!("=======================");
        println!("Total Duration: {:.2}ms", self.total_duration.as_millis());
        println!("Layer Performance:");
        println!(
            "  Total Layer Time: {:.2}ms",
            self.layer_performance.total_layer_time.as_millis()
        );
        println!(
            "  Slowest Layers: {}",
            self.layer_performance.slowest_layers.len()
        );

        if let Some((slowest_layer, duration)) = self.layer_performance.slowest_layers.first() {
            println!(
                "  Slowest Layer: {} ({:.2}ms)",
                slowest_layer,
                duration.as_millis()
            );
        }

        println!("Memory Profile:");
        println!(
            "  Peak Usage: {:.1} MB",
            self.memory_profile.peak_memory_usage
        );
        println!(
            "  Average Usage: {:.1} MB",
            self.memory_profile.average_memory_usage
        );
        println!(
            "  Memory Efficiency: {:.1}%",
            self.memory_profile.memory_efficiency * 100.0
        );

        println!("Compute Profile:");
        println!(
            "  Average CPU Utilization: {:.1}%",
            self.compute_profile.average_cpu_utilization * 100.0
        );
        println!(
            "  Average GPU Utilization: {:.1}%",
            self.compute_profile.average_gpu_utilization * 100.0
        );
        println!(
            "  Compute Efficiency: {:.1}%",
            self.compute_profile.compute_efficiency * 100.0
        );

        println!("Bottleneck Analysis:");
        println!(
            "  Primary Bottleneck: {:?}",
            self.bottleneck_analysis.primary_bottleneck
        );
        println!(
            "  Severity: {:.1}%",
            self.bottleneck_analysis.bottleneck_severity * 100.0
        );

        if !self.optimization_suggestions.is_empty() {
            println!(
                "Optimization Suggestions: {}",
                self.optimization_suggestions.len()
            );
            for (i, suggestion) in self.optimization_suggestions.iter().take(3).enumerate() {
                println!(
                    "  {}. [{:?}] {}",
                    i + 1,
                    suggestion.priority,
                    suggestion.description
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = ModelProfiler::new();
        assert!(profiler.config.enabled);
        assert!(profiler.config.track_layer_times);
    }

    #[test]
    fn test_profiler_with_config() {
        let config = ProfilerConfig {
            enabled: true,
            track_layer_times: true,
            track_memory_usage: false,
            track_compute_utilization: true,
            sample_interval_ms: 50,
            max_samples: 5000,
        };

        let profiler = ModelProfiler::with_config(config.clone());
        assert!(!profiler.config.track_memory_usage);
        assert!(profiler.config.track_compute_utilization);
        assert_eq!(profiler.config.max_samples, 5000);
    }

    #[test]
    fn test_profiling_session() -> Result<()> {
        let mut profiler = ModelProfiler::new();
        let session_id = "test_session";

        profiler.start_profiling(session_id)?;

        // Profile a layer
        let _result = profiler.profile_layer(session_id, "attention", || {
            std::thread::sleep(Duration::from_millis(10));
            Ok(42)
        })?;

        // Profile an operation
        let _result = profiler.profile_operation(session_id, "matmul", || {
            std::thread::sleep(Duration::from_millis(5));
            Ok("done".to_string())
        })?;

        let report = profiler.end_profiling(session_id)?;

        assert_eq!(report.session_id, session_id);
        assert!(report.total_duration > Duration::from_millis(10));
        assert!(report.layer_performance.layer_timings.contains_key("attention"));
        assert!(report.operation_performance.operation_timings.contains_key("matmul"));

        Ok(())
    }

    #[test]
    fn test_memory_sampling() -> Result<()> {
        let mut profiler = ModelProfiler::new();
        let session_id = "test_session";

        profiler.start_profiling(session_id)?;
        profiler.sample_memory(session_id)?;
        profiler.sample_memory(session_id)?;

        let report = profiler.end_profiling(session_id)?;

        assert!(report.memory_profile.memory_timeline.len() >= 2);
        assert!(report.memory_profile.peak_memory_usage > 0.0);

        Ok(())
    }

    #[test]
    fn test_compute_sampling() -> Result<()> {
        let mut profiler = ModelProfiler::with_config(ProfilerConfig {
            track_compute_utilization: true,
            ..Default::default()
        });

        let session_id = "test_session";

        profiler.start_profiling(session_id)?;
        profiler.sample_compute(session_id)?;
        profiler.sample_compute(session_id)?;

        let report = profiler.end_profiling(session_id)?;

        assert!(report.compute_profile.utilization_timeline.len() >= 2);
        assert!(report.compute_profile.average_cpu_utilization > 0.0);

        Ok(())
    }

    #[test]
    fn test_optimization_suggestions() {
        let suggestion = OptimizationSuggestion {
            suggestion_type: OptimizationType::MemoryOptimization,
            priority: OptimizationPriority::High,
            description: "Test suggestion".to_string(),
            expected_improvement: 0.3,
            implementation_complexity: ComplexityLevel::Medium,
        };

        assert_eq!(suggestion.expected_improvement, 0.3);
        assert!(matches!(suggestion.priority, OptimizationPriority::High));
        assert!(matches!(
            suggestion.implementation_complexity,
            ComplexityLevel::Medium
        ));
    }

    #[test]
    fn test_bottleneck_analysis() {
        let analysis = BottleneckAnalysis {
            primary_bottleneck: BottleneckType::Memory,
            bottleneck_severity: 0.8,
            affected_operations: vec!["attention".to_string()],
            bottleneck_timeline: Vec::new(),
        };

        assert!(matches!(
            analysis.primary_bottleneck,
            BottleneckType::Memory
        ));
        assert_eq!(analysis.bottleneck_severity, 0.8);
        assert_eq!(analysis.affected_operations.len(), 1);
    }

    #[test]
    fn test_layer_timing_calculation() -> Result<()> {
        let mut profiler = ModelProfiler::new();
        let session_id = "test_session";

        profiler.start_profiling(session_id)?;

        // Profile the same layer multiple times
        for _ in 0..5 {
            profiler.profile_layer(session_id, "test_layer", || {
                std::thread::sleep(Duration::from_millis(10));
                Ok(())
            })?;
        }

        let report = profiler.end_profiling(session_id)?;

        if let Some(timing) = report.layer_performance.layer_timings.get("test_layer") {
            assert_eq!(timing.call_count, 5);
            assert!(timing.average_time >= Duration::from_millis(8)); // Allow some variance
            assert!(timing.total_time >= Duration::from_millis(40));
        } else {
            assert!(false, "Layer timing not found");
        }

        Ok(())
    }
}
