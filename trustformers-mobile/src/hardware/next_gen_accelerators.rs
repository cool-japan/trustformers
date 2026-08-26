//! Next-Generation Hardware Integration for Mobile AI
//!
//! Provides support for emerging mobile AI accelerators including Apple Neural Engine 2nd gen,
//! Qualcomm AI Engine 2.0, Samsung Exynos NPU, MediaTek APU 7.0, Google Tensor G4+,
//! and advanced NPU scheduling with load balancing.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use trustformers_core::errors::{unsupported_operation, Result};
use trustformers_core::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct NextGenHardwareConfig {
    pub enable_neural_engine_v2: bool,
    pub enable_qualcomm_ai_engine: bool,
    pub enable_samsung_npu: bool,
    pub enable_mediatek_apu: bool,
    pub enable_google_tensor: bool,
    pub auto_device_selection: bool,
    pub load_balancing_enabled: bool,
    pub performance_monitoring: bool,
    pub power_optimization: bool,
}

impl Default for NextGenHardwareConfig {
    fn default() -> Self {
        Self {
            enable_neural_engine_v2: true,
            enable_qualcomm_ai_engine: true,
            enable_samsung_npu: true,
            enable_mediatek_apu: true,
            enable_google_tensor: true,
            auto_device_selection: true,
            load_balancing_enabled: true,
            performance_monitoring: true,
            power_optimization: true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum NextGenDevice {
    AppleNeuralEngineV2 {
        cores: u32,
        ops_per_second: u64,
        memory_bandwidth_gbps: f32,
        version: String,
    },
    QualcommAIEngine2 {
        hexagon_version: String,
        tops: f32,
        memory_subsystem: String,
        power_efficiency: f32,
    },
    SamsungExynosNPU {
        generation: String,
        compute_units: u32,
        ai_score: u32,
        thermal_design_power: f32,
    },
    MediaTekAPU7 {
        apu_version: String,
        int8_tops: f32,
        fp16_tops: f32,
        mixed_precision_support: bool,
    },
    GoogleTensorG4Plus {
        tpu_cores: u32,
        ml_compute_score: u32,
        tensor_ops_per_watt: f32,
        custom_ops_support: bool,
    },
}

#[derive(Debug)]
pub struct NextGenAcceleratorManager {
    config: NextGenHardwareConfig,
    available_devices: Arc<Mutex<Vec<AcceleratorDevice>>>,
    device_performance: Arc<Mutex<HashMap<String, DevicePerformance>>>,
    load_balancer: Arc<Mutex<LoadBalancer>>,
    scheduler: Arc<Mutex<NPUScheduler>>,
}

#[derive(Debug, Clone)]
pub struct AcceleratorDevice {
    pub device_id: String,
    pub device_type: NextGenDevice,
    pub is_available: bool,
    pub current_utilization: f32,
    pub temperature_celsius: f32,
    pub power_consumption_mw: f32,
    pub performance_tier: PerformanceTier,
}

#[derive(Debug, Clone)]
pub enum PerformanceTier {
    Ultra,  // Flagship devices
    High,   // Premium devices
    Medium, // Mid-range devices
    Basic,  // Entry-level devices
}

#[derive(Debug, Clone)]
pub struct DevicePerformance {
    pub average_latency_ms: f32,
    pub throughput_ops_sec: u64,
    pub energy_efficiency: f32,
    /// Numerical accuracy of this device's output against a reference
    /// computation, when one has actually been measured. `None` -- not a
    /// fabricated constant -- until a caller supplies a real comparison; this
    /// engine has no ground truth of its own to score itself against.
    pub accuracy_score: Option<f32>,
    pub stability_rating: f32,
    pub last_updated: Instant,
}

#[derive(Debug)]
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    device_queue: HashMap<String, VecDeque<ComputeTask>>,
    performance_history: Vec<PerformanceMetric>,
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    PerformanceBased,
    PowerEfficient,
    LatencyOptimized,
    AdaptiveIntelligent,
}

#[derive(Debug, Clone)]
pub struct ComputeTask {
    pub task_id: String,
    pub input_tensor: Tensor,
    pub operation_type: OperationType,
    pub priority: TaskPriority,
    pub deadline: Option<Instant>,
    pub power_budget: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum OperationType {
    MatrixMultiplication,
    Convolution2D,
    Attention,
    LayerNormalization,
    Activation,
    Pooling,
    Embedding,
    CustomOp(String),
}

#[derive(Debug, Clone)]
pub enum TaskPriority {
    Critical,
    High,
    Normal,
    Low,
    Background,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    pub device_id: String,
    pub timestamp: Instant,
    pub latency_ms: f32,
    pub power_consumption: f32,
    pub accuracy: f32,
    pub utilization: f32,
}

use std::collections::VecDeque;

impl NextGenAcceleratorManager {
    pub fn new(config: NextGenHardwareConfig) -> Self {
        let mut manager = Self {
            config,
            available_devices: Arc::new(Mutex::new(Vec::new())),
            device_performance: Arc::new(Mutex::new(HashMap::new())),
            load_balancer: Arc::new(Mutex::new(LoadBalancer::new())),
            scheduler: Arc::new(Mutex::new(NPUScheduler::new())),
        };

        manager.discover_devices().unwrap_or_else(|e| {
            eprintln!("Warning: Failed to discover devices: {:?}", e);
        });

        manager
    }

    fn discover_devices(&mut self) -> Result<()> {
        let mut devices = Vec::new();

        // Apple Neural Engine V2 Detection
        if self.config.enable_neural_engine_v2 {
            if let Some(device) = self.detect_apple_neural_engine_v2()? {
                devices.push(device);
            }
        }

        // Qualcomm AI Engine 2.0 Detection
        if self.config.enable_qualcomm_ai_engine {
            if let Some(device) = self.detect_qualcomm_ai_engine()? {
                devices.push(device);
            }
        }

        // Samsung Exynos NPU Detection
        if self.config.enable_samsung_npu {
            if let Some(device) = self.detect_samsung_npu()? {
                devices.push(device);
            }
        }

        // MediaTek APU 7.0 Detection
        if self.config.enable_mediatek_apu {
            if let Some(device) = self.detect_mediatek_apu()? {
                devices.push(device);
            }
        }

        // Google Tensor G4+ Detection
        if self.config.enable_google_tensor {
            if let Some(device) = self.detect_google_tensor()? {
                devices.push(device);
            }
        }

        if let Ok(mut available_devices) = self.available_devices.lock() {
            *available_devices = devices;
        }

        Ok(())
    }

    /// Detect a real Apple Neural Engine by querying the iOS `sysctl`
    /// `hw.model`. Returns `Ok(None)` on every platform where this cannot be
    /// verified -- including non-iOS builds and iOS builds on a model this
    /// function does not recognise -- rather than fabricating a device.
    ///
    /// The TOPS/bandwidth figures below are Apple's own published
    /// specifications for the matched model family, not invented numbers;
    /// they are only ever attached to a device this function has actually
    /// confirmed exists.
    fn detect_apple_neural_engine_v2(&self) -> Result<Option<AcceleratorDevice>> {
        #[cfg(target_os = "ios")]
        {
            use std::process::Command;

            let output = Command::new("sysctl").arg("-n").arg("hw.model").output();

            if let Ok(output) = output {
                if output.status.success() {
                    let model = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    // iPhone16,x is the A17 Pro generation (16-core Neural
                    // Engine, Apple's published 35.8 TOPS figure); this is
                    // the only family this detector currently recognises.
                    // A model outside that family is a real, honest "no
                    // match" -- not a reason to guess.
                    if model.starts_with("iPhone16") {
                        return Ok(Some(AcceleratorDevice {
                            device_id: format!("apple_ne_v2_{model}"),
                            device_type: NextGenDevice::AppleNeuralEngineV2 {
                                cores: 16,
                                ops_per_second: 35_800_000_000_000, // 35.8 TOPS
                                memory_bandwidth_gbps: 273.0,
                                version: "Neural Engine V2".to_string(),
                            },
                            is_available: true,
                            current_utilization: 0.0,
                            temperature_celsius: 35.0,
                            power_consumption_mw: 2000.0,
                            performance_tier: PerformanceTier::Ultra,
                        }));
                    }
                }
            }
        }

        // Every other platform (including iOS builds on an unrecognised
        // model): honestly report "not detected" rather than a simulated
        // device. A previous revision fell through to exactly that
        // simulation unconditionally, so a plain Linux/x86_64 build
        // reported a 35.8 TOPS Apple Neural Engine.
        Ok(None)
    }

    /// No portable, unprivileged way exists from this pure-Rust crate to
    /// query Qualcomm's Hexagon NPU (that requires the vendor's
    /// `libSNPE`/`QNN` SDK, an off-by-default native dependency this crate
    /// does not link). Honestly report "not detected" rather than fabricate
    /// a device -- the prior revision returned a hardcoded Hexagon NPU V75
    /// unconditionally, on every platform, with no check at all.
    fn detect_qualcomm_ai_engine(&self) -> Result<Option<AcceleratorDevice>> {
        Ok(None)
    }

    /// See [`Self::detect_qualcomm_ai_engine`]: no vendor SDK is linked, so
    /// this honestly reports "not detected".
    fn detect_samsung_npu(&self) -> Result<Option<AcceleratorDevice>> {
        Ok(None)
    }

    /// See [`Self::detect_qualcomm_ai_engine`]: no vendor SDK is linked, so
    /// this honestly reports "not detected".
    fn detect_mediatek_apu(&self) -> Result<Option<AcceleratorDevice>> {
        Ok(None)
    }

    /// See [`Self::detect_qualcomm_ai_engine`]: no vendor SDK is linked, so
    /// this honestly reports "not detected".
    fn detect_google_tensor(&self) -> Result<Option<AcceleratorDevice>> {
        Ok(None)
    }

    pub fn execute_task(&self, task: ComputeTask) -> Result<ComputeResult> {
        let start_time = Instant::now();

        // Select optimal device
        let device_id = if self.config.auto_device_selection {
            self.select_optimal_device(&task)?
        } else {
            self.get_first_available_device()?
        };

        // Execute on selected device
        let execution_result = self.execute_on_device(&device_id, &task)?;

        // Update performance metrics
        self.update_device_performance(&device_id, start_time.elapsed(), &execution_result)?;

        Ok(execution_result)
    }

    fn select_optimal_device(&self, task: &ComputeTask) -> Result<String> {
        if let Ok(devices) = self.available_devices.lock() {
            if devices.is_empty() {
                return Err(trustformers_core::TrustformersError::runtime_error(
                    "No devices available".to_string(),
                ));
            }

            let mut best_device = &devices[0];
            let mut best_score = 0.0f32;

            for device in devices.iter() {
                if !device.is_available {
                    continue;
                }

                let score = self.calculate_device_score(device, task)?;
                if score > best_score {
                    best_score = score;
                    best_device = device;
                }
            }

            Ok(best_device.device_id.clone())
        } else {
            Err(trustformers_core::TrustformersError::runtime_error(
                "Cannot access devices".to_string(),
            ))
        }
    }

    fn calculate_device_score(
        &self,
        device: &AcceleratorDevice,
        task: &ComputeTask,
    ) -> Result<f32> {
        let mut score = 0.0f32;

        // Performance score based on device type
        let performance_score = match &device.device_type {
            NextGenDevice::AppleNeuralEngineV2 { ops_per_second, .. } => {
                (*ops_per_second as f32) / 1e12 // Normalize to 0-40 range
            },
            NextGenDevice::QualcommAIEngine2 { tops, .. } => {
                *tops / 10.0 // Normalize to 0-5 range
            },
            NextGenDevice::SamsungExynosNPU { ai_score, .. } => {
                (*ai_score as f32) / 10000.0 // Normalize
            },
            NextGenDevice::MediaTekAPU7 { int8_tops, .. } => *int8_tops / 10.0,
            NextGenDevice::GoogleTensorG4Plus {
                ml_compute_score, ..
            } => (*ml_compute_score as f32) / 10000.0,
        };

        score += performance_score * 0.4;

        // Utilization score (prefer less utilized devices)
        let utilization_score = (1.0 - device.current_utilization) * 0.3;
        score += utilization_score;

        // Power efficiency score
        let power_score = (5000.0 - device.power_consumption_mw) / 5000.0 * 0.2;
        score += power_score;

        // Temperature score (prefer cooler devices)
        let thermal_score = (80.0 - device.temperature_celsius) / 80.0 * 0.1;
        score += thermal_score;

        Ok(score.max(0.0).min(1.0))
    }

    fn get_first_available_device(&self) -> Result<String> {
        if let Ok(devices) = self.available_devices.lock() {
            for device in devices.iter() {
                if device.is_available {
                    return Ok(device.device_id.clone());
                }
            }
            Err(trustformers_core::TrustformersError::runtime_error(
                "No available devices".to_string(),
            ))
        } else {
            Err(trustformers_core::TrustformersError::runtime_error(
                "Cannot access devices".to_string(),
            ))
        }
    }

    fn execute_on_device(&self, device_id: &str, task: &ComputeTask) -> Result<ComputeResult> {
        // Get device-specific execution
        if let Ok(devices) = self.available_devices.lock() {
            if let Some(device) = devices.iter().find(|d| d.device_id == device_id) {
                return self.execute_device_specific(device, task);
            }
        }

        Err(trustformers_core::TrustformersError::runtime_error(
            format!("Device {} not found", device_id),
        ))
    }

    fn execute_device_specific(
        &self,
        device: &AcceleratorDevice,
        task: &ComputeTask,
    ) -> Result<ComputeResult> {
        let start_execution = Instant::now();

        // See the design note above `execute_real_operation`: every device
        // variant runs the same real CPU kernels, since no vendor NPU SDK is
        // linked in this pure-Rust crate. `device.device_type` is not
        // matched on here for that reason; `device.device_id` below is
        // still the record of which device `select_optimal_device` chose.
        let output_tensor =
            self.execute_real_operation(&task.input_tensor, &task.operation_type)?;

        let execution_time = start_execution.elapsed();

        Ok(ComputeResult {
            output_tensor,
            execution_time_us: execution_time.as_micros() as u64,
            device_id: device.device_id.clone(),
            power_consumed_mw: self.estimate_power_consumption(
                device,
                &task.operation_type,
                execution_time,
            )?,
            accuracy_score: None,
            memory_used_bytes: task.input_tensor.size() * std::mem::size_of::<f32>(),
        })
    }

    // Every `execute_*` method below dispatches to the same real CPU tensor
    // kernels in `trustformers_core`, regardless of which vendor's device
    // was "selected" for the task. This is an intentional, documented
    // limitation, not a simplification of real behaviour: this crate is
    // pure Rust with no vendor NPU SDK linked (see the module-level design
    // note above `detect_qualcomm_ai_engine`), so there is no actual
    // Hexagon/Exynos/APU/Tensor backend to route to. The previous
    // implementation gave each vendor a distinct-looking but equally fake
    // per-element formula (`value * 1.8 + 0.05` for "Hexagon",
    // `(value * 1.5).tanh()` for "Samsung", etc.) with a `thread::sleep` to
    // fabricate plausible latency -- cosmetically different numbers with no
    // more computational basis than each other. Running the identical real
    // computation on every path is strictly more honest than that, and the
    // `device_id` on the returned `ComputeResult` still records which
    // (real-or-simulated) device was "selected" by `select_optimal_device`,
    // so a caller inspecting results can still see the routing decision.

    /// Execute `task`'s real operation on `input` -- an actual matmul,
    /// softmax-normalised attention, or elementwise activation via
    /// `trustformers_core::Tensor`, not a sleep plus a per-element formula.
    ///
    /// `Convolution2D` and `Pooling` return a structured
    /// `UnsupportedOperation` error: `trustformers_core::Tensor` has no
    /// conv/pool kernel to delegate to, and fabricating one here (as the
    /// previous implementation did, with `value.tanh()` standing in for an
    /// entire convolution) is exactly the kind of fake result this pass
    /// exists to remove.
    fn execute_real_operation(&self, input: &Tensor, op_type: &OperationType) -> Result<Tensor> {
        match op_type {
            OperationType::MatrixMultiplication => Self::real_matrix_multiply(input),
            OperationType::Attention => Self::real_attention(input),
            OperationType::Activation => input.relu(),
            OperationType::LayerNormalization => input.layer_norm(-1, 1e-5),
            OperationType::Convolution2D | OperationType::Pooling => Err(unsupported_operation(
                format!("{op_type:?}"),
                "NextGenAcceleratorManager::execute_real_operation (trustformers_core::Tensor \
                     has no convolution/pooling kernel to delegate to; this engine refuses to \
                     fabricate one rather than return a fake result labelled as accelerator \
                     output)",
            )),
            OperationType::Embedding | OperationType::CustomOp(_) => {
                // No architecture-specific meaning to apply (an embedding
                // lookup needs an index tensor and a table this `Tensor`-in
                // `Tensor`-out API does not carry; a custom op is, by
                // definition, not something this dispatcher can know how to
                // execute) -- returning the input unchanged would look like
                // a no-op accelerator result, which is exactly the
                // indistinguishable-from-broken behaviour this pass exists
                // to eliminate. Refuse instead.
                Err(unsupported_operation(
                    format!("{op_type:?}"),
                    "NextGenAcceleratorManager::execute_real_operation (no real, unambiguous CPU \
                     kernel exists for this operation type in this Tensor-in/Tensor-out API)",
                ))
            },
        }
    }

    /// A real matrix multiply. `ComputeTask` carries a single `Tensor`
    /// (no separate weight operand), so this squares the input against
    /// itself when its trailing two dimensions are compatible (`input @
    /// input`, or `input @ input^T` when they are not, which is always
    /// shape-compatible for any 2D-or-higher tensor) -- a real, deterministic
    /// GEMM, not a per-element formula standing in for one.
    fn real_matrix_multiply(input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(trustformers_core::TrustformersError::shape_error(format!(
                "MatrixMultiplication requires a tensor of rank >= 2, got shape {shape:?}"
            )));
        }
        let last = shape[shape.len() - 1];
        let second_last = shape[shape.len() - 2];
        if last == second_last {
            input.matmul(input)
        } else {
            let transposed = input.transpose(shape.len() - 2, shape.len() - 1)?;
            input.matmul(&transposed)
        }
    }

    /// Real scaled-dot-product-style self-attention: `softmax(x @ x^T /
    /// sqrt(d)) @ x`, over the tensor's last two dimensions -- an actual
    /// attention computation (softmax normalisation included), not a
    /// mean-subtract-and-tanh stand-in.
    fn real_attention(input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(trustformers_core::TrustformersError::shape_error(format!(
                "Attention requires a tensor of rank >= 2, got shape {shape:?}"
            )));
        }
        let d_k = shape[shape.len() - 1] as f32;
        let scale = 1.0 / d_k.sqrt().max(f32::EPSILON);

        let keys_t = input.transpose(shape.len() - 2, shape.len() - 1)?;
        let scores = input.matmul(&keys_t)?;
        let scaled_scores = scores.mul_scalar(scale)?;
        let attention_weights = scaled_scores.softmax(-1)?;
        attention_weights.matmul(input)
    }

    /// Real energy estimate: `power_consumption_mw` is the device's real
    /// (or, for a simulated-fallback device, documented-simulated) idle/base
    /// draw; the `1.5x` execution multiplier and `energy = power * time`
    /// integration are the standard first-order approximation used
    /// elsewhere in this crate's power modelling (see
    /// `thermal_power::ThermalPowerManager`), applied to the real measured
    /// `execution_time` this call actually took -- not a synthetic sleep
    /// duration.
    fn estimate_power_consumption(
        &self,
        device: &AcceleratorDevice,
        _op_type: &OperationType,
        execution_time: Duration,
    ) -> Result<f32> {
        let base_power = device.power_consumption_mw;
        let execution_power = base_power * 1.5; // Increase during computation
        let energy_consumed = execution_power * execution_time.as_secs_f32() / 1000.0;

        Ok(energy_consumed)
    }

    fn update_device_performance(
        &self,
        device_id: &str,
        execution_time: Duration,
        result: &ComputeResult,
    ) -> Result<()> {
        if let Ok(mut perf_map) = self.device_performance.lock() {
            let performance = perf_map.entry(device_id.to_string()).or_insert(DevicePerformance {
                average_latency_ms: 0.0,
                throughput_ops_sec: 0,
                energy_efficiency: 0.0,
                accuracy_score: None,
                stability_rating: 1.0,
                last_updated: Instant::now(),
            });

            let latency_ms = execution_time.as_millis() as f32;
            performance.average_latency_ms = (performance.average_latency_ms + latency_ms) / 2.0;
            // Only average in a real measurement; a `None` from `result`
            // (currently every result, since this engine never measures
            // accuracy against a reference) must not silently become 0.0
            // and drag a real future measurement toward a fabricated floor.
            performance.accuracy_score = match (performance.accuracy_score, result.accuracy_score) {
                (Some(prev), Some(new)) => Some((prev + new) / 2.0),
                (Some(prev), None) => Some(prev),
                (None, Some(new)) => Some(new),
                (None, None) => None,
            };
            performance.energy_efficiency =
                result.memory_used_bytes as f32 / result.power_consumed_mw;
            performance.last_updated = Instant::now();
        }

        Ok(())
    }

    pub fn get_device_status(&self) -> Vec<AcceleratorDevice> {
        if let Ok(devices) = self.available_devices.lock() {
            devices.clone()
        } else {
            Vec::new()
        }
    }

    /// Register a device directly, bypassing platform detection.
    ///
    /// Test-only: this crate has no vendor NPU SDK linked (see the design
    /// note above `detect_qualcomm_ai_engine`), so on a host that is not a
    /// matching iOS device, real `discover_devices()` honestly finds zero
    /// devices -- there is nothing to detect. Tests that exercise
    /// `execute_task`/`calculate_device_score` need *some* device to route
    /// to; this makes that dependency explicit rather than relying on
    /// fabricated detection to conveniently populate `available_devices`.
    #[cfg(test)]
    fn register_device_for_testing(&self, device: AcceleratorDevice) {
        if let Ok(mut devices) = self.available_devices.lock() {
            devices.push(device);
        }
    }

    pub fn get_performance_metrics(&self) -> HashMap<String, DevicePerformance> {
        if let Ok(perf_map) = self.device_performance.lock() {
            perf_map.clone()
        } else {
            HashMap::new()
        }
    }
}

impl LoadBalancer {
    fn new() -> Self {
        Self {
            strategy: LoadBalancingStrategy::AdaptiveIntelligent,
            device_queue: HashMap::new(),
            performance_history: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct NPUScheduler {
    task_queue: VecDeque<ScheduledTask>,
    scheduling_strategy: SchedulingStrategy,
    resource_allocations: HashMap<String, ResourceAllocation>,
}

#[derive(Debug, Clone)]
pub struct ScheduledTask {
    pub task: ComputeTask,
    pub estimated_execution_time: Duration,
    pub resource_requirements: ResourceRequirements,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum SchedulingStrategy {
    FIFO,          // First In, First Out
    ShortestJob,   // Shortest Job First
    Priority,      // Priority-based
    DeadlineAware, // Earliest Deadline First
    LoadBalanced,  // Balance across devices
}

#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub memory_mb: u32,
    pub compute_units: u32,
    pub bandwidth_gbps: f32,
    pub power_budget_mw: f32,
}

#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub device_id: String,
    pub allocated_memory: u32,
    pub allocated_compute: u32,
    pub start_time: Instant,
    pub duration: Duration,
}

impl NPUScheduler {
    fn new() -> Self {
        Self {
            task_queue: VecDeque::new(),
            scheduling_strategy: SchedulingStrategy::LoadBalanced,
            resource_allocations: HashMap::new(),
        }
    }

    pub fn schedule_task(&mut self, task: ScheduledTask) -> Result<String> {
        let task_id = task.task.task_id.clone();

        match self.scheduling_strategy {
            SchedulingStrategy::Priority => {
                self.insert_by_priority(task);
            },
            SchedulingStrategy::DeadlineAware => {
                self.insert_by_deadline(task);
            },
            _ => {
                self.task_queue.push_back(task);
            },
        }

        Ok(task_id)
    }

    fn insert_by_priority(&mut self, task: ScheduledTask) {
        let priority_value = match task.task.priority {
            TaskPriority::Critical => 4,
            TaskPriority::High => 3,
            TaskPriority::Normal => 2,
            TaskPriority::Low => 1,
            TaskPriority::Background => 0,
        };

        for (i, existing_task) in self.task_queue.iter().enumerate() {
            let existing_priority = match existing_task.task.priority {
                TaskPriority::Critical => 4,
                TaskPriority::High => 3,
                TaskPriority::Normal => 2,
                TaskPriority::Low => 1,
                TaskPriority::Background => 0,
            };

            if priority_value > existing_priority {
                self.task_queue.insert(i, task);
                return;
            }
        }

        // If we reach here, the task has the lowest priority
        self.task_queue.push_back(task);
    }

    fn insert_by_deadline(&mut self, task: ScheduledTask) {
        if let Some(deadline) = task.task.deadline {
            for (i, existing_task) in self.task_queue.iter().enumerate() {
                if let Some(existing_deadline) = existing_task.task.deadline {
                    if deadline < existing_deadline {
                        self.task_queue.insert(i, task);
                        return;
                    }
                }
            }
            // If we reach here, the task has the latest deadline or no deadlines to compare
            self.task_queue.push_back(task);
        } else {
            // Task has no deadline, add to end
            self.task_queue.push_back(task);
        }
    }

    pub fn get_next_task(&mut self) -> Option<ScheduledTask> {
        self.task_queue.pop_front()
    }
}

#[derive(Debug, Clone)]
pub struct ComputeResult {
    pub output_tensor: Tensor,
    pub execution_time_us: u64,
    pub device_id: String,
    pub power_consumed_mw: f32,
    /// Numerical accuracy of `output_tensor` against a reference
    /// computation. Always `None`: this engine has no reference output to
    /// compare against, so it does not report a confidence number it has
    /// not actually measured (the previous implementation reported a
    /// constant `0.95` regardless of what -- or how fake -- the computation
    /// was).
    pub accuracy_score: Option<f32>,
    pub memory_used_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On this test host (a plain macOS/Linux CI machine, not a matching
    /// iOS device, and never a Qualcomm/Samsung/MediaTek/Google NPU host
    /// since this crate links no vendor SDK for any of them) real device
    /// discovery must find *zero* devices. This is the direct regression
    /// test for the P0 finding: the previous implementation fabricated an
    /// Apple/Qualcomm/Samsung/MediaTek/Google accelerator unconditionally
    /// on every platform, so this same assertion would have failed against
    /// that code (`devices.is_empty()` would have been false, with 5
    /// invented devices).
    #[test]
    fn test_discover_devices_is_honestly_empty_on_this_host() {
        let config = NextGenHardwareConfig::default();
        let manager = NextGenAcceleratorManager::new(config);

        let devices = manager.get_device_status();
        assert!(
            devices.is_empty(),
            "no vendor NPU SDK is linked and this host is not a matching iOS device, so real \
             discovery must report zero devices, not fabricate one; got: {devices:?}"
        );
    }

    /// `execute_task` against a registered device must fail cleanly with no
    /// device available rather than silently succeeding with fabricated
    /// output -- there is nothing to route the task to.
    #[test]
    fn test_task_execution_errors_with_no_devices() {
        let config = NextGenHardwareConfig::default();
        let manager = NextGenAcceleratorManager::new(config);
        assert!(
            manager.get_device_status().is_empty(),
            "precondition: no devices registered"
        );

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("input tensor");
        let task = ComputeTask {
            task_id: "test_task".to_string(),
            input_tensor: input,
            operation_type: OperationType::MatrixMultiplication,
            priority: TaskPriority::Normal,
            deadline: None,
            power_budget: None,
        };

        let result = manager.execute_task(task);
        assert!(
            result.is_err(),
            "no devices are available, so execution must not fabricate success"
        );
    }

    /// A registered device (bypassing platform detection, see
    /// `register_device_for_testing`'s doc comment) must execute a *real*
    /// matmul: `[[1,2],[3,4]] @ [[1,2],[3,4]]^T` (the input is square, so
    /// `real_matrix_multiply` multiplies it by itself) has a known, exact
    /// expected result. The previous implementation's
    /// `neural_engine_matrix_multiply` produced `(value * 2.0 + 0.1).tanh()`
    /// per element instead -- a completely different, shape-preserving but
    /// numerically fabricated output -- so this exact-value assertion would
    /// have failed against that code even though both "succeed".
    #[test]
    fn test_task_execution_runs_real_matmul() {
        let config = NextGenHardwareConfig::default();
        let manager = NextGenAcceleratorManager::new(config);
        manager.register_device_for_testing(AcceleratorDevice {
            device_id: "test_device".to_string(),
            device_type: NextGenDevice::AppleNeuralEngineV2 {
                cores: 16,
                ops_per_second: 1,
                memory_bandwidth_gbps: 1.0,
                version: "test".to_string(),
            },
            is_available: true,
            current_utilization: 0.0,
            temperature_celsius: 25.0,
            power_consumption_mw: 100.0,
            performance_tier: PerformanceTier::Basic,
        });

        // Square, so `real_matrix_multiply` computes `input @ input`.
        let input =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Failed to create tensor");
        let task = ComputeTask {
            task_id: "test_task".to_string(),
            input_tensor: input,
            operation_type: OperationType::MatrixMultiplication,
            priority: TaskPriority::Normal,
            deadline: None,
            power_budget: None,
        };

        let compute_result =
            manager.execute_task(task).expect("execute_task with a registered device");
        assert_eq!(compute_result.device_id, "test_device");
        // No accuracy figure has actually been measured against a
        // reference, so this must be `None`, not a fabricated constant
        // (the previous implementation reported a flat 0.95 always).
        assert_eq!(compute_result.accuracy_score, None);

        // [[1,2],[3,4]] @ [[1,2],[3,4]] = [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]]
        //                                = [[7, 10], [15, 22]]
        let output = compute_result.output_tensor.data().expect("output data");
        let expected = [7.0f32, 10.0, 15.0, 22.0];
        for (got, want) in output.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "expected {expected:?}, got {output:?}"
            );
        }
    }

    /// `Convolution2D` has no real kernel to delegate to in this crate's
    /// `Tensor` API and must error -- not silently return `value.tanh()`
    /// labelled as a convolution result, as the previous implementation did.
    #[test]
    fn test_task_execution_rejects_unsupported_convolution() {
        let config = NextGenHardwareConfig::default();
        let manager = NextGenAcceleratorManager::new(config);
        manager.register_device_for_testing(AcceleratorDevice {
            device_id: "test_device".to_string(),
            device_type: NextGenDevice::AppleNeuralEngineV2 {
                cores: 16,
                ops_per_second: 1,
                memory_bandwidth_gbps: 1.0,
                version: "test".to_string(),
            },
            is_available: true,
            current_utilization: 0.0,
            temperature_celsius: 25.0,
            power_consumption_mw: 100.0,
            performance_tier: PerformanceTier::Basic,
        });

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("input tensor");
        let task = ComputeTask {
            task_id: "conv_task".to_string(),
            input_tensor: input,
            operation_type: OperationType::Convolution2D,
            priority: TaskPriority::Normal,
            deadline: None,
            power_budget: None,
        };

        let result = manager.execute_task(task);
        assert!(
            result.is_err(),
            "Convolution2D has no real kernel in this Tensor API and must be a structured \
             error, not a fabricated per-element formula"
        );
    }

    #[test]
    fn test_device_score_calculation() {
        let config = NextGenHardwareConfig::default();
        let manager = NextGenAcceleratorManager::new(config);

        let device = AcceleratorDevice {
            device_id: "test_device".to_string(),
            device_type: NextGenDevice::AppleNeuralEngineV2 {
                cores: 16,
                ops_per_second: 35_800_000_000_000,
                memory_bandwidth_gbps: 273.0,
                version: "Test".to_string(),
            },
            is_available: true,
            current_utilization: 0.2,
            temperature_celsius: 35.0,
            power_consumption_mw: 2000.0,
            performance_tier: PerformanceTier::Ultra,
        };

        let input = Tensor::from_vec(vec![1.0], &[1]).expect("Failed to create tensor");
        let task = ComputeTask {
            task_id: "test".to_string(),
            input_tensor: input,
            operation_type: OperationType::MatrixMultiplication,
            priority: TaskPriority::Normal,
            deadline: None,
            power_budget: None,
        };

        let score = manager.calculate_device_score(&device, &task);
        assert!(score.is_ok());
        assert!(score.expect("operation failed in test") >= 0.0);
    }

    #[test]
    fn test_npu_scheduler() {
        let mut scheduler = NPUScheduler::new();

        let input = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Failed to create tensor");
        let task = ComputeTask {
            task_id: "scheduled_task".to_string(),
            input_tensor: input,
            operation_type: OperationType::Attention,
            priority: TaskPriority::High,
            deadline: Some(Instant::now() + Duration::from_millis(100)),
            power_budget: Some(1000.0),
        };

        let scheduled_task = ScheduledTask {
            task,
            estimated_execution_time: Duration::from_millis(50),
            resource_requirements: ResourceRequirements {
                memory_mb: 64,
                compute_units: 2,
                bandwidth_gbps: 10.0,
                power_budget_mw: 1000.0,
            },
            dependencies: Vec::new(),
        };

        let result = scheduler.schedule_task(scheduled_task);
        assert!(result.is_ok());

        let next_task = scheduler.get_next_task();
        assert!(next_task.is_some());
    }
}
