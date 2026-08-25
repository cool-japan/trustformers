//! Core debugging session and configuration management
//!
//! This module contains the fundamental components for TrustformeRS debugging including
//! the main DebugSession coordinator, configuration structures, and session lifecycle management.
// reason: debug/profiling scaffolding — structs are constructed and their fields/methods
// are retained for the data model, serialization completeness, and future consumers that
// do not yet read every member. Consolidated from many item-level #[allow(dead_code)].
#![allow(dead_code)]

use crate::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Configuration for debugging session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    /// Enable tensor inspection
    pub enable_tensor_inspection: bool,
    /// Enable gradient debugging
    pub enable_gradient_debugging: bool,
    /// Enable model diagnostics
    pub enable_model_diagnostics: bool,
    /// Enable visual debugging (requires display)
    pub enable_visualization: bool,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable computation graph analysis
    pub enable_computation_graph_analysis: bool,
    /// Maximum number of tensors to track
    pub max_tracked_tensors: usize,
    /// Maximum history length for gradients
    pub max_gradient_history: usize,
    /// Output directory for debug artifacts
    pub output_dir: Option<String>,
    /// Sampling rate for expensive operations (0.0 to 1.0)
    pub sampling_rate: f32,
    /// Memory profiling configuration
    pub memory_profiling_config: MemoryProfilingConfig,
    /// Computation graph analysis configuration
    pub graph_analysis_config: GraphAnalysisConfig,
    /// Architecture analysis configuration
    pub architecture_analysis_config: architecture_analysis::ArchitectureAnalysisConfig,
    /// Behavior analysis configuration
    pub behavior_analysis_config: BehaviorAnalysisConfig,
    /// Training dynamics analysis configuration
    pub training_dynamics_config: TrainingDynamicsConfig,
    /// Differential debugging configuration
    pub differential_debugging_config: DifferentialDebuggingConfig,
    /// Interpretability tools configuration
    pub interpretability_config: InterpretabilityConfig,
    /// Neural network debugging configuration
    pub neural_network_debugging_config: Option<neural_network_debugging::TransformerDebugConfig>,
    /// Advanced ML debugging configuration
    pub advanced_ml_debugging_config: AdvancedMLDebuggingConfig,
    /// Advanced GPU profiling configuration
    pub advanced_gpu_profiling_config: AdvancedGpuProfilingConfig,
    /// Kernel optimization configuration
    pub kernel_optimization_config: KernelOptimizationConfig,
    /// AI code analysis configuration
    pub ai_code_analysis_config: AIAnalysisConfig,
    /// Distributed debugging configuration
    pub distributed_debugging_config: Option<DistributedDebugConfig>,
    /// Environmental monitoring configuration
    pub environmental_monitoring_config: EnvironmentalConfig,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enable_tensor_inspection: true,
            enable_gradient_debugging: true,
            enable_model_diagnostics: true,
            enable_visualization: false,
            enable_memory_profiling: true,
            enable_computation_graph_analysis: true,
            max_tracked_tensors: 1000,
            max_gradient_history: 100,
            output_dir: None,
            sampling_rate: 1.0,
            memory_profiling_config: MemoryProfilingConfig::default(),
            graph_analysis_config: GraphAnalysisConfig::default(),
            architecture_analysis_config:
                architecture_analysis::ArchitectureAnalysisConfig::default(),
            behavior_analysis_config: BehaviorAnalysisConfig::default(),
            training_dynamics_config: TrainingDynamicsConfig::default(),
            differential_debugging_config: DifferentialDebuggingConfig::default(),
            interpretability_config: InterpretabilityConfig::default(),
            neural_network_debugging_config: None,
            advanced_ml_debugging_config: AdvancedMLDebuggingConfig::default(),
            advanced_gpu_profiling_config: AdvancedGpuProfilingConfig::default(),
            kernel_optimization_config: KernelOptimizationConfig::default(),
            ai_code_analysis_config: AIAnalysisConfig::default(),
            distributed_debugging_config: None,
            environmental_monitoring_config: EnvironmentalConfig::default(),
        }
    }
}

/// Main debugging session that coordinates all debugging tools
#[derive(Debug)]
pub struct DebugSession {
    id: Uuid,
    config: DebugConfig,
    tensor_inspector: TensorInspector,
    gradient_debugger: GradientDebugger,
    model_diagnostics: ModelDiagnostics,
    hooks: HookManager,
    profiler: Profiler,
    memory_profiler: Option<MemoryProfiler>,
    interactive_debugger: InteractiveDebugger,
    anomaly_detector: AnomalyDetector,
    computation_graph_analyzer: ComputationGraphAnalyzer,
    architecture_analyzer: architecture_analysis::ArchitectureAnalyzer,
    behavior_analyzer: BehaviorAnalyzer,
    training_dynamics_analyzer: TrainingDynamicsAnalyzer,
    differential_debugger: DifferentialDebugger,
    interpretability_analyzer: InterpretabilityAnalyzer,
    health_checker: crate::health_checker::HealthChecker,
    transformer_debugger: Option<neural_network_debugging::TransformerDebugger>,
    advanced_ml_debugger: AdvancedMLDebugger,
    advanced_gpu_profiler: Option<AdvancedGpuMemoryProfiler>,
    kernel_optimizer: KernelOptimizationAnalyzer,
    ai_code_analyzer: Option<AICodeAnalyzer>,
    distributed_debugger: Option<DistributedDebugger>,
    environmental_monitor: Option<EnvironmentalMonitor>,
}

impl DebugSession {
    /// Create a new debugging session
    pub fn new(config: DebugConfig) -> Self {
        let id = Uuid::new_v4();

        let memory_profiler = if config.enable_memory_profiling {
            Some(MemoryProfiler::new(config.memory_profiling_config.clone()))
        } else {
            None
        };

        let transformer_debugger =
            config.neural_network_debugging_config.as_ref().map(|neural_config| {
                neural_network_debugging::TransformerDebugger::new(neural_config.clone())
            });

        let advanced_gpu_profiler = if config.advanced_gpu_profiling_config.enable_gpu_profiling {
            AdvancedGpuMemoryProfiler::new(config.advanced_gpu_profiling_config.device_count).ok()
        } else {
            None
        };

        let ai_code_analyzer = if config.ai_code_analysis_config.enable_deep_analysis {
            Some(AICodeAnalyzer::new(config.ai_code_analysis_config.clone()))
        } else {
            None
        };

        let distributed_debugger =
            if let Some(ref dist_config) = config.distributed_debugging_config {
                let node_id = NodeId::new(0, "debug-node".to_string());
                Some(DistributedDebugger::new(dist_config.clone(), node_id))
            } else {
                None
            };

        let environmental_monitor = if config.environmental_monitoring_config.enable_carbon_tracking
        {
            Some(EnvironmentalMonitor::new(
                config.environmental_monitoring_config.clone(),
            ))
        } else {
            None
        };

        Self {
            id,
            tensor_inspector: TensorInspector::new(&config),
            gradient_debugger: GradientDebugger::new(config.clone()),
            model_diagnostics: ModelDiagnostics::new(&config),
            hooks: HookManager::new(),
            profiler: Profiler::new(&config),
            memory_profiler,
            interactive_debugger: InteractiveDebugger::new(&config),
            anomaly_detector: AnomalyDetector::new(&config),
            computation_graph_analyzer: ComputationGraphAnalyzer::new(
                config.graph_analysis_config.clone(),
            ),
            architecture_analyzer: architecture_analysis::ArchitectureAnalyzer::new(
                config.architecture_analysis_config.clone(),
            ),
            behavior_analyzer: BehaviorAnalyzer::new(config.behavior_analysis_config.clone()),
            training_dynamics_analyzer: TrainingDynamicsAnalyzer::new(),
            differential_debugger: DifferentialDebugger::new(
                config.differential_debugging_config.clone(),
            ),
            interpretability_analyzer: InterpretabilityAnalyzer::new(
                config.interpretability_config.clone(),
            ),
            health_checker: crate::health_checker::HealthChecker::new(&config),
            transformer_debugger,
            advanced_ml_debugger: AdvancedMLDebugger::new(
                config.advanced_ml_debugging_config.clone(),
            ),
            advanced_gpu_profiler,
            kernel_optimizer: match KernelOptimizationAnalyzer::new() {
                Ok(analyzer) => analyzer,
                Err(e) => {
                    tracing::warn!(
                        "Failed to initialize kernel optimizer sub-analyzers: {}; \
                         continuing with an empty analyzer (fully functional, no prior history)",
                        e
                    );
                    KernelOptimizationAnalyzer::new_empty()
                },
            },
            ai_code_analyzer,
            distributed_debugger,
            environmental_monitor,
            config,
        }
    }

    /// Get session ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get debug configuration
    pub fn config(&self) -> &DebugConfig {
        &self.config
    }

    /// Get tensor inspector
    pub fn tensor_inspector(&self) -> &TensorInspector {
        &self.tensor_inspector
    }

    /// Get mutable tensor inspector
    pub fn tensor_inspector_mut(&mut self) -> &mut TensorInspector {
        &mut self.tensor_inspector
    }

    /// Get gradient debugger
    pub fn gradient_debugger(&self) -> &GradientDebugger {
        &self.gradient_debugger
    }

    /// Get mutable gradient debugger
    pub fn gradient_debugger_mut(&mut self) -> &mut GradientDebugger {
        &mut self.gradient_debugger
    }

    /// Get model diagnostics
    pub fn model_diagnostics(&self) -> &ModelDiagnostics {
        &self.model_diagnostics
    }

    /// Get mutable model diagnostics
    pub fn model_diagnostics_mut(&mut self) -> &mut ModelDiagnostics {
        &mut self.model_diagnostics
    }

    /// Get hook manager
    pub fn hooks(&self) -> &HookManager {
        &self.hooks
    }

    /// Get mutable hook manager
    pub fn hooks_mut(&mut self) -> &mut HookManager {
        &mut self.hooks
    }

    /// Get profiler
    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }

    /// Get mutable profiler
    pub fn profiler_mut(&mut self) -> &mut Profiler {
        &mut self.profiler
    }

    /// Get memory profiler
    pub fn memory_profiler(&self) -> Option<&MemoryProfiler> {
        self.memory_profiler.as_ref()
    }

    /// Get mutable memory profiler
    pub fn memory_profiler_mut(&mut self) -> Option<&mut MemoryProfiler> {
        self.memory_profiler.as_mut()
    }

    /// Get interactive debugger
    pub fn interactive_debugger(&self) -> &InteractiveDebugger {
        &self.interactive_debugger
    }

    /// Get mutable interactive debugger
    pub fn interactive_debugger_mut(&mut self) -> &mut InteractiveDebugger {
        &mut self.interactive_debugger
    }

    /// Get anomaly detector
    pub fn anomaly_detector(&self) -> &AnomalyDetector {
        &self.anomaly_detector
    }

    /// Get mutable anomaly detector
    pub fn anomaly_detector_mut(&mut self) -> &mut AnomalyDetector {
        &mut self.anomaly_detector
    }

    /// Get computation graph analyzer
    pub fn computation_graph_analyzer(&self) -> &ComputationGraphAnalyzer {
        &self.computation_graph_analyzer
    }

    /// Get mutable computation graph analyzer
    pub fn computation_graph_analyzer_mut(&mut self) -> &mut ComputationGraphAnalyzer {
        &mut self.computation_graph_analyzer
    }

    /// Get architecture analyzer
    pub fn architecture_analyzer(&self) -> &architecture_analysis::ArchitectureAnalyzer {
        &self.architecture_analyzer
    }

    /// Get mutable architecture analyzer
    pub fn architecture_analyzer_mut(
        &mut self,
    ) -> &mut architecture_analysis::ArchitectureAnalyzer {
        &mut self.architecture_analyzer
    }

    /// Get behavior analyzer
    pub fn behavior_analyzer(&self) -> &BehaviorAnalyzer {
        &self.behavior_analyzer
    }

    /// Get mutable behavior analyzer
    pub fn behavior_analyzer_mut(&mut self) -> &mut BehaviorAnalyzer {
        &mut self.behavior_analyzer
    }

    /// Get training dynamics analyzer
    pub fn training_dynamics_analyzer(&self) -> &TrainingDynamicsAnalyzer {
        &self.training_dynamics_analyzer
    }

    /// Get mutable training dynamics analyzer
    pub fn training_dynamics_analyzer_mut(&mut self) -> &mut TrainingDynamicsAnalyzer {
        &mut self.training_dynamics_analyzer
    }

    /// Get differential debugger
    pub fn differential_debugger(&self) -> &DifferentialDebugger {
        &self.differential_debugger
    }

    /// Get mutable differential debugger
    pub fn differential_debugger_mut(&mut self) -> &mut DifferentialDebugger {
        &mut self.differential_debugger
    }

    /// Get interpretability analyzer
    pub fn interpretability_analyzer(&self) -> &InterpretabilityAnalyzer {
        &self.interpretability_analyzer
    }

    /// Get mutable interpretability analyzer
    pub fn interpretability_analyzer_mut(&mut self) -> &mut InterpretabilityAnalyzer {
        &mut self.interpretability_analyzer
    }

    /// Get health checker
    pub fn health_checker(&self) -> &crate::health_checker::HealthChecker {
        &self.health_checker
    }

    /// Get mutable health checker
    pub fn health_checker_mut(&mut self) -> &mut crate::health_checker::HealthChecker {
        &mut self.health_checker
    }

    /// Get transformer debugger
    pub fn transformer_debugger(&self) -> Option<&neural_network_debugging::TransformerDebugger> {
        self.transformer_debugger.as_ref()
    }

    /// Get mutable transformer debugger
    pub fn transformer_debugger_mut(
        &mut self,
    ) -> Option<&mut neural_network_debugging::TransformerDebugger> {
        self.transformer_debugger.as_mut()
    }

    /// Get advanced ML debugger
    pub fn advanced_ml_debugger(&self) -> &AdvancedMLDebugger {
        &self.advanced_ml_debugger
    }

    /// Get mutable advanced ML debugger
    pub fn advanced_ml_debugger_mut(&mut self) -> &mut AdvancedMLDebugger {
        &mut self.advanced_ml_debugger
    }

    /// Get AI code analyzer
    pub fn ai_code_analyzer(&self) -> Option<&AICodeAnalyzer> {
        self.ai_code_analyzer.as_ref()
    }

    /// Get mutable AI code analyzer
    pub fn ai_code_analyzer_mut(&mut self) -> Option<&mut AICodeAnalyzer> {
        self.ai_code_analyzer.as_mut()
    }

    /// Get distributed debugger
    pub fn distributed_debugger(&self) -> Option<&DistributedDebugger> {
        self.distributed_debugger.as_ref()
    }

    /// Get mutable distributed debugger
    pub fn distributed_debugger_mut(&mut self) -> Option<&mut DistributedDebugger> {
        self.distributed_debugger.as_mut()
    }

    /// Get environmental monitor
    pub fn environmental_monitor(&self) -> Option<&EnvironmentalMonitor> {
        self.environmental_monitor.as_ref()
    }

    /// Get mutable environmental monitor
    pub fn environmental_monitor_mut(&mut self) -> Option<&mut EnvironmentalMonitor> {
        self.environmental_monitor.as_mut()
    }

    /// Start debugging session
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting debug session {}", self.id);

        if self.config.enable_tensor_inspection {
            self.tensor_inspector.start().await?;
        }

        if self.config.enable_gradient_debugging {
            self.gradient_debugger.start().await?;
        }

        if self.config.enable_model_diagnostics {
            self.model_diagnostics.start().await?;
        }

        self.profiler.start().await?;

        if let Some(ref mut memory_profiler) = self.memory_profiler {
            memory_profiler.start().await?;
        }

        self.interactive_debugger.start().await?;
        self.anomaly_detector.start().await?;

        Ok(())
    }

    /// Stop debugging session and generate report
    pub async fn stop(&mut self) -> Result<DebugReport> {
        tracing::info!("Stopping debug session {}", self.id);

        let tensor_report = if self.config.enable_tensor_inspection {
            Some(self.tensor_inspector.generate_report().await?)
        } else {
            None
        };

        let gradient_report = if self.config.enable_gradient_debugging {
            Some(self.gradient_debugger.generate_report().await?)
        } else {
            None
        };

        let diagnostics_report = if self.config.enable_model_diagnostics {
            Some(self.model_diagnostics.generate_report().await?)
        } else {
            None
        };

        let profiler_report = self.profiler.generate_report().await?;

        let memory_profiler_report = if let Some(ref mut memory_profiler) = self.memory_profiler {
            Some(memory_profiler.stop().await?)
        } else {
            None
        };

        let interactive_debugger_report = self.interactive_debugger.generate_report().await?;
        let anomaly_report = self.anomaly_detector.generate_report().await?;

        // `DebugSession` never drives `ComputationGraphAnalyzer` itself: graph
        // analysis is an explicit, caller-initiated step
        // (`crate::computation_graph`), so a session report has no graph to
        // summarise. Honestly absent rather than an empty-looking report.
        let computation_graph_report = None;

        // Get new analyzer reports
        let architecture_analysis_report =
            Some(self.architecture_analyzer.generate_report().await?);
        let behavior_analysis_report = Some(self.behavior_analyzer.generate_report().await?);
        let training_dynamics_report =
            Some(self.training_dynamics_analyzer.generate_report().await?);
        let differential_debugging_report =
            Some(self.differential_debugger.generate_report().await?);
        let interpretability_report = Some(self.interpretability_analyzer.generate_report().await?);
        let advanced_ml_debugging_report = Some(self.advanced_ml_debugger.generate_report().await?);

        // Generate GPU profiling reports
        let advanced_gpu_profiling_report = self
            .advanced_gpu_profiler
            .as_ref()
            .map(|profiler| profiler.get_memory_analysis_report());

        let kernel_optimization_report =
            Some(self.generate_kernel_optimization_summary_report().await?);

        Ok(DebugReport {
            session_id: self.id,
            tensor_report,
            gradient_report,
            diagnostics_report,
            profiler_report,
            memory_profiler_report,
            interactive_debugger_report,
            anomaly_report,
            computation_graph_report,
            architecture_analysis_report,
            behavior_analysis_report,
            training_dynamics_report,
            differential_debugging_report,
            interpretability_report,
            advanced_ml_debugging_report,
            advanced_gpu_profiling_report,
            kernel_optimization_report,
            config: self.config.clone(),
        })
    }

    /// Export debug session to file
    pub async fn export(&self, path: &str) -> Result<()> {
        let report = self.generate_snapshot().await?;
        let json = serde_json::to_string_pretty(&report)?;
        tokio::fs::write(path, json).await?;
        Ok(())
    }

    /// Generate a snapshot of current state
    pub async fn generate_snapshot(&self) -> Result<DebugReport> {
        let tensor_report = if self.config.enable_tensor_inspection {
            Some(self.tensor_inspector.generate_report().await?)
        } else {
            None
        };

        let gradient_report = if self.config.enable_gradient_debugging {
            Some(self.gradient_debugger.generate_report().await?)
        } else {
            None
        };

        let diagnostics_report = if self.config.enable_model_diagnostics {
            Some(self.model_diagnostics.generate_report().await?)
        } else {
            None
        };

        let profiler_report = self.profiler.generate_report().await?;

        // A snapshot must not stop the profiler, and `MemoryProfiler` only
        // produces a report at `stop()` -- there is no borrow-a-report-mid-run
        // API -- so a snapshot carries no memory report. `generate_report`
        // (which does stop the profiler) is the call that returns one.
        let memory_profiler_report = None;

        let interactive_debugger_report = self.interactive_debugger.generate_report().await?;
        let anomaly_report = self.anomaly_detector.generate_report().await?;

        // `DebugSession` never drives `ComputationGraphAnalyzer` itself: graph
        // analysis is an explicit, caller-initiated step
        // (`crate::computation_graph`), so a session report has no graph to
        // summarise. Honestly absent rather than an empty-looking report.
        let computation_graph_report = None;

        // Get new analyzer reports
        let architecture_analysis_report =
            Some(self.architecture_analyzer.generate_report().await?);
        let behavior_analysis_report = Some(self.behavior_analyzer.generate_report().await?);
        let training_dynamics_report =
            Some(self.training_dynamics_analyzer.generate_report().await?);
        let differential_debugging_report =
            Some(self.differential_debugger.generate_report().await?);
        let interpretability_report = Some(self.interpretability_analyzer.generate_report().await?);
        let advanced_ml_debugging_report = Some(self.advanced_ml_debugger.generate_report().await?);

        // Generate GPU profiling reports for snapshot
        let advanced_gpu_profiling_report = self
            .advanced_gpu_profiler
            .as_ref()
            .map(|profiler| profiler.get_memory_analysis_report());

        let kernel_optimization_report =
            Some(self.generate_kernel_optimization_summary_report().await?);

        Ok(DebugReport {
            session_id: self.id,
            tensor_report,
            gradient_report,
            diagnostics_report,
            profiler_report,
            memory_profiler_report,
            interactive_debugger_report,
            anomaly_report,
            computation_graph_report,
            architecture_analysis_report,
            behavior_analysis_report,
            training_dynamics_report,
            differential_debugging_report,
            interpretability_report,
            advanced_ml_debugging_report,
            advanced_gpu_profiling_report,
            kernel_optimization_report,
            config: self.config.clone(),
        })
    }

    /// Convenience method for debugging tensors (used by debug_tensor! macro)
    pub fn debug_tensor<T>(&mut self, tensor: &ArrayD<T>, name: &str) -> Result<Uuid>
    where
        T: Clone + Into<f64> + fmt::Debug + 'static,
    {
        self.tensor_inspector.inspect_tensor(tensor, name, None, None)
    }

    /// Summarise every kernel this session's optimizer has really analysed.
    ///
    /// All counts come from the analyzer's own recorded profiles and
    /// suggestions; `overall_optimization_score` is `None` when no kernel has
    /// been analysed yet. The previous version returned
    /// `total_kernels_analyzed: 0` alongside `overall_optimization_score: 85.0`
    /// -- an 85-out-of-100 verdict on zero kernels.
    async fn generate_kernel_optimization_summary_report(
        &self,
    ) -> Result<KernelOptimizationSummaryReport> {
        /// A suggestion promising at least this much performance gain counts
        /// as high impact.
        const HIGH_IMPACT_GAIN_PERCENT: f64 = 10.0;

        let kernel_names: Vec<String> = self
            .kernel_optimizer
            .analyzed_kernel_names()
            .into_iter()
            .map(str::to_string)
            .collect();
        let total_kernels_analyzed = kernel_names.len();
        let suggestions = self.kernel_optimizer.optimization_suggestions();

        let optimization_opportunities_found = suggestions.values().map(Vec::len).sum::<usize>();

        let mut high_impact_optimizations: Vec<HighImpactOptimization> = suggestions
            .iter()
            .flat_map(|(kernel_name, opts)| {
                opts.iter().filter_map(move |opt| {
                    let gain = opt.expected_improvement.performance_gain_percentage;
                    (gain >= HIGH_IMPACT_GAIN_PERCENT).then(|| HighImpactOptimization {
                        kernel_name: kernel_name.clone(),
                        optimization_type: format!("{:?}", opt.optimization_type),
                        // Gain percentage expressed as a speedup factor.
                        expected_speedup: 1.0 + gain / 100.0,
                        implementation_difficulty: format!("{:?}", opt.implementation_difficulty),
                        description: opt.explanation.clone(),
                    })
                })
            })
            .collect();
        high_impact_optimizations.sort_by(|a, b| {
            b.expected_speedup
                .partial_cmp(&a.expected_speedup)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Fusion opportunities and regression alerts are surfaced per kernel by
        // `KernelOptimizationAnalyzer::get_optimization_report`, which needs a
        // kernel name; count them across the kernels really analysed.
        let mut fusion_opportunities = 0usize;
        let mut regression_alerts = 0usize;
        for name in &kernel_names {
            if let Ok(report) = self.kernel_optimizer.get_optimization_report(name) {
                fusion_opportunities += report.fusion_opportunities.len();
                if report.regression_status.as_ref().is_some_and(|status| status.has_regression) {
                    regression_alerts += 1;
                }
            }
        }

        // Score: start at 100 and dock the mean promised gain still on the
        // table, so a kernel set with nothing left to improve scores 100.
        let overall_optimization_score = if total_kernels_analyzed == 0 {
            None
        } else {
            let total_gain: f64 = suggestions
                .values()
                .flat_map(|opts| opts.iter())
                .map(|opt| opt.expected_improvement.performance_gain_percentage)
                .sum();
            Some((100.0 - total_gain / total_kernels_analyzed as f64).clamp(0.0, 100.0))
        };

        let top_recommendations: Vec<String> = high_impact_optimizations
            .iter()
            .take(5)
            .map(|opt| format!("{}: {}", opt.kernel_name, opt.description))
            .collect();

        Ok(KernelOptimizationSummaryReport {
            total_kernels_analyzed,
            optimization_opportunities_found,
            high_impact_optimizations,
            fusion_opportunities,
            regression_alerts,
            overall_optimization_score,
            top_recommendations,
        })
    }

    /// Convenience method for debugging gradients (used by debug_gradient! macro)
    pub fn debug_gradients<T>(&mut self, layer_name: &str, gradients: &[T]) -> Result<()>
    where
        T: Clone + Into<f64> + fmt::Debug + 'static,
    {
        // Convert gradients vector to ndarray
        use scirs2_core::ndarray::Array; // SciRS2 Integration Policy
        let gradient_array = Array::from_vec(gradients.to_vec()).into_dyn();

        // Record the gradients as a real tracked tensor of their own, named
        // after the layer, so their statistics survive the call.
        //
        // Previously this minted a fresh `Uuid::new_v4()` and passed it to
        // `inspect_gradients`, which looks the id up in `tracked_tensors`. A
        // freshly generated v4 uuid is never in that map, so the statistics
        // were computed and then dropped on the floor while the call reported
        // `Ok(())`, and `_layer_name` was ignored entirely.
        let gradient_tensor_name = format!("{layer_name}.grad");
        self.tensor_inspector.inspect_tensor(
            &gradient_array,
            &gradient_tensor_name,
            Some(layer_name),
            Some("gradient"),
        )?;

        // Additionally attach them as this layer's `gradient_stats` when the
        // layer's own tensor has been inspected before.
        if let Some(tensor_id) = self.tensor_inspector.tensor_id_for_name(layer_name) {
            self.tensor_inspector.inspect_gradients(tensor_id, &gradient_array)?;
        } else {
            tracing::debug!(
                layer = layer_name,
                "no previously inspected tensor for this layer; gradients recorded standalone \
                 as {gradient_tensor_name}"
            );
        }

        Ok(())
    }
}

/// Comprehensive debug report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugReport {
    pub session_id: Uuid,
    pub tensor_report: Option<TensorInspectionReport>,
    pub gradient_report: Option<GradientDebugReport>,
    pub diagnostics_report: Option<ModelDiagnosticsReport>,
    pub profiler_report: ProfilerReport,
    pub memory_profiler_report: Option<MemoryProfilingReport>,
    pub interactive_debugger_report: InteractiveDebuggerReport,
    pub anomaly_report: AnomalyDetectorReport,
    pub computation_graph_report: Option<GraphAnalysisResult>,
    pub architecture_analysis_report: Option<ArchitectureAnalysisReport>,
    pub behavior_analysis_report: Option<BehaviorAnalysisReport>,
    pub training_dynamics_report: Option<model_diagnostics::training::TrainingDynamicsReport>,
    pub differential_debugging_report: Option<DifferentialDebuggingReport>,
    pub interpretability_report: Option<InterpretabilityReport>,
    pub advanced_ml_debugging_report: Option<AdvancedMLDebuggingReport>,
    pub advanced_gpu_profiling_report: Option<MemoryAnalysisReport>,
    pub kernel_optimization_report: Option<KernelOptimizationSummaryReport>,
    pub config: DebugConfig,
}

impl DebugReport {
    /// Get summary of key findings
    pub fn summary(&self) -> DebugSummary {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // Analyze tensor issues
        if let Some(ref tensor_report) = self.tensor_report {
            if tensor_report.has_nan_values() {
                issues.push("NaN values detected in tensors".to_string());
                recommendations.push("Check input data and model initialization".to_string());
            }

            if tensor_report.has_inf_values() {
                issues.push("Infinite values detected in tensors".to_string());
                recommendations.push("Reduce learning rate or add gradient clipping".to_string());
            }
        }

        // Analyze gradient issues
        if let Some(ref gradient_report) = self.gradient_report {
            if gradient_report.has_vanishing_gradients() {
                issues.push("Vanishing gradients detected".to_string());
                recommendations
                    .push("Consider residual connections or gradient scaling".to_string());
            }

            if gradient_report.has_exploding_gradients() {
                issues.push("Exploding gradients detected".to_string());
                recommendations.push("Add gradient clipping".to_string());
            }
        }

        DebugSummary {
            session_id: self.session_id,
            total_issues: issues.len(),
            critical_issues: issues
                .iter()
                .filter(|i| i.contains("NaN") || i.contains("exploding"))
                .count(),
            issues,
            recommendations,
        }
    }
}

/// High-level summary of debug findings
#[derive(Debug, Serialize, Deserialize)]
pub struct DebugSummary {
    pub session_id: Uuid,
    pub total_issues: usize,
    pub critical_issues: usize,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Convenience function to create a debug session with default config
pub fn debug_session() -> DebugSession {
    DebugSession::new(DebugConfig::default())
}

/// Convenience function to create a debug session with custom config
pub fn debug_session_with_config(config: DebugConfig) -> DebugSession {
    DebugSession::new(config)
}

/// Convenience function to create a debug session with transformer debugging enabled
pub fn debug_session_with_transformer() -> DebugSession {
    let config = DebugConfig {
        neural_network_debugging_config: Some(
            neural_network_debugging::TransformerDebugConfig::default(),
        ),
        ..Default::default()
    };
    DebugSession::new(config)
}
