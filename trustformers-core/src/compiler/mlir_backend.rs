/*!
# MLIR Backend Module

This module drives the external [`mlir-opt`](https://mlir.llvm.org/) command-line tool
as a subprocess to apply MLIR (Multi-Level Intermediate Representation) optimizations:

- **MLIR Integration**: Shells out to a real `mlir-opt` binary for optimization/lowering/
  validation/version queries. No MLIR toolchain is vendored or linked into this crate —
  `mlir-opt` is a separate C++ program this crate never builds.
- **Dialect Support**: Emits Tensor/Linalg/Arith/TOSA and a custom `ml.*` dialect as MLIR
  text; [`DialectRegistry`] documents which operations this crate knows how to emit.
- **Pass Pipeline**: Configurable optimization pass pipeline ([`MlirPassPipeline`]).
- **Target Lowering**: Lowering to different hardware targets via `mlir-opt` passes.

## Honesty contract
`mlir-opt` is usually **not installed** — this backend is opt-in
(`CompilerConfig::enable_mlir`, default `false`) and experimental. [`MlirBackend::new`]
therefore does not silently pretend to compile anything when no binary can be found: it
returns a structured [`TrustformersError`] naming `mlir-opt`, the paths that were searched,
and how to point at a custom build (the `TRUSTFORMERS_MLIR_OPT_PATH` environment variable).
A caller who wants to exercise the pass-pipeline/cache-key plumbing without an MLIR
toolchain installed must explicitly opt into [`MlirBackend::new_simulated`], whose output is
a labeled, non-functional passthrough — never a value returned by `new`. [`MlirBackend::is_real`]
tells a caller which mode a given instance is in. [`MlirBackend::validate_mlir`] and
[`MlirBackend::get_mlir_version`] likewise never fabricate a semantic-validation result or a
tool version; see their doc comments. [`MlirBackend::generate_stats`] only reports quantities
this backend actually measures (pass names, wall-clock time, before/after byte counts);
op/loop/memory-access counts are not computed by anything in this module and are `None`.

Invoking `mlir-opt` when it is present is ordinary subprocess use (`std::process::Command`),
not FFI — it does not link against MLIR/LLVM.
*/

use crate::compiler::{CompilerConfig, IntermediateRepresentation};
use crate::errors::TrustformersError;
use crate::errors::{compute_error, runtime_error, unsupported_operation};
use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};
use tempfile::NamedTempFile;

/// MLIR backend for advanced optimizations.
///
/// `mlir_opt_path` is `Some(path)` when a real `mlir-opt` binary was located and responds
/// successfully to `--version` (a "real" backend, [`MlirBackend::is_real`] returns `true`);
/// it is `None` only for an instance built via [`MlirBackend::new_simulated`], which is the
/// sole, explicitly-named way to get a non-functional backend — [`MlirBackend::new`] never
/// produces one silently.
pub struct MlirBackend {
    config: CompilerConfig,
    cache: HashMap<String, Vec<u8>>,
    mlir_opt_path: Option<String>,
    dialect_support: DialectSupport,
}

impl MlirBackend {
    /// Create a new MLIR backend backed by a real `mlir-opt` binary.
    ///
    /// Returns a structured error naming `mlir-opt` and the searched locations when no
    /// working binary can be found, rather than silently falling back to a mock. See
    /// [`MlirBackend::new_simulated`] for an explicit, labeled non-functional alternative.
    pub fn new(config: &CompilerConfig) -> Result<Self, TrustformersError> {
        let mlir_opt_path = Self::find_mlir_opt().ok_or_else(Self::missing_binary_error)?;

        Ok(Self {
            config: config.clone(),
            cache: HashMap::new(),
            mlir_opt_path: Some(mlir_opt_path),
            dialect_support: DialectSupport::default(),
        })
    }

    /// Create an MLIR backend in explicit simulation mode: no `mlir-opt` binary is invoked,
    /// and every "compiled"/"optimized" output is a clearly labeled passthrough of the input
    /// MLIR text, never real machine code. Intended for exercising the pass-pipeline and
    /// cache-key plumbing in this module without an MLIR toolchain installed (e.g. this
    /// crate's own tests). `is_real()` returns `false` on the result. This constructor is the
    /// *only* way to obtain a non-real `MlirBackend` — [`MlirBackend::new`] errors instead of
    /// reaching this state on its own.
    pub fn new_simulated(config: &CompilerConfig) -> Self {
        Self {
            config: config.clone(),
            cache: HashMap::new(),
            mlir_opt_path: None,
            dialect_support: DialectSupport::default(),
        }
    }

    /// `true` if this backend is driving a real `mlir-opt` binary; `false` if it was built via
    /// [`MlirBackend::new_simulated`] and every compile/optimize/validate call below returns
    /// clearly-labeled placeholder output instead of real results.
    pub fn is_real(&self) -> bool {
        self.mlir_opt_path.is_some()
    }

    /// Build the structured error `new()` returns when no `mlir-opt` binary is available.
    fn missing_binary_error() -> TrustformersError {
        let searched = Self::search_paths().join(", ");
        let mut error = runtime_error(format!(
            "MlirBackend requires the external `mlir-opt` binary, but none was found. \
             Searched (in order): {searched}{}.",
            match std::env::var("TRUSTFORMERS_MLIR_OPT_PATH") {
                Ok(custom) => format!(
                    ", and TRUSTFORMERS_MLIR_OPT_PATH={custom:?} (did not respond successfully to `--version`)"
                ),
                Err(_) => String::new(),
            }
        ))
        .with_suggestion("Install an MLIR toolchain that provides `mlir-opt` and put it on PATH")
        .with_suggestion(
            "Or set the TRUSTFORMERS_MLIR_OPT_PATH environment variable to a working mlir-opt binary",
        );
        error = error.with_suggestion(
            "Or call MlirBackend::new_simulated(&config) to exercise the pass-pipeline/cache-key \
             logic without a real MLIR toolchain (its output is a labeled non-functional \
             placeholder; MlirBackend::is_real() reports false)",
        );
        error
    }

    /// Create MLIR backend with advanced features, backed by a real `mlir-opt` binary. See
    /// [`MlirBackend::new`] for the honesty contract this delegates to.
    pub fn new_with_features(
        config: &CompilerConfig,
        features: MlirAdvancedFeatures,
    ) -> Result<Self, TrustformersError> {
        let mut backend = Self::new(config)?;

        // Configure advanced features
        if features.ml_dialect_enabled {
            backend.dialect_support.ml_custom = true;
        }

        Ok(backend)
    }

    /// Compile with custom pass pipeline.
    ///
    /// If this backend is in explicit simulation mode ([`Self::is_real`] is `false`), the
    /// returned bytes are a labeled non-functional passthrough of the generated MLIR text, not
    /// a real compiled artifact — check `is_real()` before treating the result as executable.
    pub fn compile_with_pipeline(
        &mut self,
        ir: IntermediateRepresentation,
        pipeline: MlirPassPipeline,
    ) -> Result<Vec<u8>, TrustformersError> {
        // Generate cache key including pipeline
        let pipeline_hash = self.hash_pipeline(&pipeline)?;
        let cache_key = format!("{}-{}", self.generate_cache_key(&ir)?, pipeline_hash);

        // Check cache
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Convert IR to MLIR with custom operations
        let mlir_code = self.ir_to_mlir_advanced(&ir)?;

        // Apply custom pass pipeline
        let optimized_mlir = self.apply_custom_pipeline(&mlir_code, &pipeline)?;

        // Lower to target code
        let target_code = self.lower_to_target(&optimized_mlir)?;

        // Cache result
        self.cache.insert(cache_key, target_code.clone());

        Ok(target_code)
    }

    /// Hash pass pipeline for caching
    fn hash_pipeline(&self, pipeline: &MlirPassPipeline) -> Result<String, TrustformersError> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        pipeline.to_pass_string().hash(&mut hasher);
        Ok(format!("{:x}", hasher.finish()))
    }

    /// Convert IR to MLIR with advanced ML operations
    fn ir_to_mlir_advanced(
        &self,
        ir: &IntermediateRepresentation,
    ) -> Result<String, TrustformersError> {
        let mut mlir_code = String::new();

        // MLIR module header with custom dialects
        mlir_code.push_str(
            "module {
",
        );

        // Add tensor type definitions
        for instruction in &ir.instructions {
            if let Some(tensor_def) = self.generate_tensor_definitions(instruction)? {
                mlir_code.push_str(&format!("  {}\n", tensor_def));
            }
        }

        // Convert instructions to advanced MLIR operations
        for instruction in &ir.instructions {
            let mlir_op = self.instruction_to_mlir_advanced(instruction)?;
            mlir_code.push_str(&format!("  {}\n", mlir_op));
        }

        mlir_code.push_str("}\n");

        Ok(mlir_code)
    }

    /// Generate tensor type definitions
    fn generate_tensor_definitions(
        &self,
        instruction: &crate::compiler::jit_compiler::IRInstruction,
    ) -> Result<Option<String>, TrustformersError> {
        // Generate type definitions based on instruction requirements
        match instruction.opcode {
            crate::compiler::jit_compiler::IROpcode::MatMul => Ok(Some(
                "%tensor_type = !ml.tensor<dynamic x dynamic x f32>".to_string(),
            )),
            _ => Ok(None),
        }
    }

    /// Convert instruction to advanced MLIR operation
    fn instruction_to_mlir_advanced(
        &self,
        instruction: &crate::compiler::jit_compiler::IRInstruction,
    ) -> Result<String, TrustformersError> {
        use crate::compiler::jit_compiler::IROpcode;

        let mlir_op = match instruction.opcode {
            IROpcode::MatMul => {
                // Use custom ML dialect for better optimization
                format!(
                    "%{} = ml.batch_matmul %input0, %input1 {{
        transpose_a = false,
        transpose_b = false,
        fused_activation = \"none\"
    }} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>",
                    instruction.id
                )
            },
            IROpcode::LayerNorm => {
                format!(
                    "%{} = ml.layer_norm %input0, %gamma, %beta {{
        axis = [-1],
        epsilon = 1.0e-5 : f64,
        fused_add_bias = false
    }} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>",
                    instruction.id
                )
            },
            IROpcode::Softmax => {
                format!(
                    "%{} = ml.softmax %input0 {{
        axis = -1 : i64,
        temperature = 1.0 : f32
    }} : (tensor<*xf32>) -> tensor<*xf32>",
                    instruction.id
                )
            },
            _ => {
                // Fall back to basic conversion
                self.instruction_to_mlir(instruction)?
            },
        };

        Ok(mlir_op)
    }

    /// Apply custom pass pipeline
    fn apply_custom_pipeline(
        &self,
        mlir_code: &str,
        pipeline: &MlirPassPipeline,
    ) -> Result<String, TrustformersError> {
        let Some(mlir_opt_path) = &self.mlir_opt_path else {
            // Explicit simulation mode (MlirBackend::new_simulated): no mlir-opt was invoked,
            // and this text is not actually optimized. The banner makes that unmistakable to
            // anything printing or logging the result.
            return Ok(format!(
                "// [trustformers simulated MLIR backend -- no mlir-opt binary; text \
                 passthrough only, NOT optimized] pipeline: {}\n{}",
                pipeline.to_pass_string(),
                mlir_code
            ));
        };

        // Create temporary file for MLIR code
        let mut temp_file = NamedTempFile::new()
            .map_err(|e| runtime_error(format!("Failed to create temp file: {}", e)))?;

        temp_file
            .write_all(mlir_code.as_bytes())
            .map_err(|e| runtime_error(format!("Failed to write MLIR code: {}", e)))?;

        // Build command with custom pipeline
        let mut cmd = Command::new(mlir_opt_path);
        cmd.arg(temp_file.path());

        // Add pipeline passes
        let pass_string = pipeline.to_pass_string();
        for pass_arg in pass_string.split_whitespace() {
            cmd.arg(pass_arg);
        }

        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let output = cmd
            .output()
            .map_err(|e| runtime_error(format!("Failed to run mlir-opt: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(compute_error(
                "mlir_compilation",
                format!("MLIR optimization failed: {}", stderr),
            ));
        }

        let optimized_code = String::from_utf8(output.stdout)
            .map_err(|e| runtime_error(format!("Invalid UTF-8 in MLIR output: {}", e)))?;

        Ok(optimized_code)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &CompilerConfig) -> Result<(), TrustformersError> {
        self.config = config.clone();
        Ok(())
    }

    /// Fixed candidate locations searched for `mlir-opt`, in order. Does not include the
    /// `TRUSTFORMERS_MLIR_OPT_PATH` override, which is checked first by [`Self::find_mlir_opt`].
    fn search_paths() -> Vec<&'static str> {
        vec![
            "mlir-opt",
            "/usr/local/bin/mlir-opt",
            "/opt/mlir/bin/mlir-opt",
            "/usr/bin/mlir-opt",
        ]
    }

    /// `true` if `path` runs and exits successfully on `--version`. This is a liveness probe,
    /// not proof the binary is actually `mlir-opt` — same contract as the pre-existing check.
    fn probe_binary(path: &str) -> bool {
        Command::new(path)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Locate a working `mlir-opt` binary: the `TRUSTFORMERS_MLIR_OPT_PATH` environment
    /// variable is tried first (letting a caller point at a non-standard build), then the
    /// fixed candidate locations. Returns `None` — never a sentinel string — when nothing
    /// responds successfully to `--version`.
    fn find_mlir_opt() -> Option<String> {
        if let Ok(custom) = std::env::var("TRUSTFORMERS_MLIR_OPT_PATH") {
            if Self::probe_binary(&custom) {
                return Some(custom);
            }
        }

        for path in Self::search_paths() {
            if Self::probe_binary(path) {
                return Some(path.to_string());
            }
        }

        None
    }

    /// Compile IR using MLIR.
    ///
    /// If this backend is in explicit simulation mode ([`Self::is_real`] is `false`), the
    /// returned bytes are a labeled non-functional passthrough of the generated MLIR text, not
    /// a real compiled artifact — check `is_real()` before treating the result as executable.
    pub fn compile_ir(
        &mut self,
        ir: IntermediateRepresentation,
    ) -> Result<Vec<u8>, TrustformersError> {
        // Generate cache key
        let cache_key = self.generate_cache_key(&ir)?;

        // Check cache
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Convert IR to MLIR format
        let mlir_code = self.ir_to_mlir(&ir)?;

        // Apply MLIR optimizations
        let optimized_mlir = self.apply_mlir_passes(&mlir_code)?;

        // Lower to target code
        let target_code = self.lower_to_target(&optimized_mlir)?;

        // Cache result
        self.cache.insert(cache_key, target_code.clone());

        Ok(target_code)
    }

    /// Convert IR to MLIR format
    fn ir_to_mlir(&self, ir: &IntermediateRepresentation) -> Result<String, TrustformersError> {
        let mut mlir_code = String::new();

        // MLIR module header
        mlir_code.push_str("module {\n");

        // Convert instructions to MLIR operations
        for instruction in &ir.instructions {
            let mlir_op = self.instruction_to_mlir(instruction)?;
            mlir_code.push_str(&format!("  {}\n", mlir_op));
        }

        mlir_code.push_str("}\n");

        Ok(mlir_code)
    }

    /// Convert IR instruction to MLIR operation
    fn instruction_to_mlir(
        &self,
        instruction: &crate::compiler::jit_compiler::IRInstruction,
    ) -> Result<String, TrustformersError> {
        use crate::compiler::jit_compiler::IROpcode;

        let mlir_op = match instruction.opcode {
            IROpcode::Add => {
                format!(
                    "%{} = arith.addf %input0, %input1 : tensor<*xf32>",
                    instruction.id
                )
            },
            IROpcode::Mul => {
                format!(
                    "%{} = arith.mulf %input0, %input1 : tensor<*xf32>",
                    instruction.id
                )
            },
            IROpcode::MatMul => {
                format!("%{} = linalg.matmul ins(%input0, %input1 : tensor<*xf32>, tensor<*xf32>) outs(%output : tensor<*xf32>) -> tensor<*xf32>", instruction.id)
            },
            IROpcode::ReLU => {
                format!(
                    "%{} = math.max %input0, %c0 : tensor<*xf32>",
                    instruction.id
                )
            },
            IROpcode::Sigmoid => {
                // sigmoid(x) = 1 / (1 + exp(-x))
                format!("%{} = math.sigmoid %input0 : tensor<*xf32>", instruction.id)
            },
            IROpcode::Softmax => {
                format!("%{} = tosa.softmax %input0 {{axis = -1 : i64}} : (tensor<*xf32>) -> tensor<*xf32>", instruction.id)
            },
            IROpcode::LayerNorm => {
                format!("%{} = tosa.layer_norm %input0, %gamma, %beta {{axis = -1 : i64, epsilon = 1.0e-5 : f64}} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>", instruction.id)
            },
            IROpcode::Reshape => {
                format!(
                    "%{} = tensor.reshape %input0 : tensor<*xf32> to tensor<*xf32>",
                    instruction.id
                )
            },
            IROpcode::Transpose => {
                format!("%{} = linalg.transpose ins(%input0 : tensor<*xf32>) outs(%output : tensor<*xf32>) permutation = [1, 0]", instruction.id)
            },
            _ => {
                return Err(unsupported_operation(
                    "mlir_dialect",
                    format!(
                        "MLIR conversion not implemented for {:?}",
                        instruction.opcode
                    ),
                ));
            },
        };

        Ok(mlir_op)
    }

    /// Apply MLIR optimization passes
    fn apply_mlir_passes(&self, mlir_code: &str) -> Result<String, TrustformersError> {
        let Some(mlir_opt_path) = &self.mlir_opt_path else {
            // Explicit simulation mode (MlirBackend::new_simulated) -- see apply_custom_pipeline.
            return Ok(format!(
                "// [trustformers simulated MLIR backend -- no mlir-opt binary; text \
                 passthrough only, NOT optimized]\n{}",
                mlir_code
            ));
        };

        // Create temporary file for MLIR code
        let mut temp_file = NamedTempFile::new()
            .map_err(|e| runtime_error(format!("Failed to create temp file: {}", e)))?;

        temp_file
            .write_all(mlir_code.as_bytes())
            .map_err(|e| runtime_error(format!("Failed to write MLIR code: {}", e)))?;

        // Build optimization passes based on config
        let passes = self.build_optimization_passes();

        // Run mlir-opt with passes
        let mut cmd = Command::new(mlir_opt_path);
        cmd.arg(temp_file.path());

        for pass in passes {
            cmd.arg(format!("--{}", pass));
        }

        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let output = cmd
            .output()
            .map_err(|e| runtime_error(format!("Failed to run mlir-opt: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(compute_error(
                "mlir_compilation",
                format!("MLIR optimization failed: {}", stderr),
            ));
        }

        let optimized_code = String::from_utf8(output.stdout)
            .map_err(|e| runtime_error(format!("Invalid UTF-8 in MLIR output: {}", e)))?;

        Ok(optimized_code)
    }

    /// Build list of optimization passes based on configuration
    fn build_optimization_passes(&self) -> Vec<String> {
        use crate::compiler::OptimizationLevel;

        let mut passes = Vec::new();

        // Always include basic passes
        passes.push("canonicalize".to_string());
        passes.push("cse".to_string()); // Common subexpression elimination

        match self.config.optimization_level {
            OptimizationLevel::None => {
                // No additional passes
            },
            OptimizationLevel::Basic => {
                passes.push("sccp".to_string()); // Sparse conditional constant propagation
            },
            OptimizationLevel::Standard => {
                passes.push("sccp".to_string());
                passes.push("loop-invariant-code-motion".to_string());
                passes.push("inline".to_string());
            },
            OptimizationLevel::Aggressive => {
                passes.push("sccp".to_string());
                passes.push("loop-invariant-code-motion".to_string());
                passes.push("inline".to_string());
                passes.push("loop-unroll".to_string());
                passes.push("vectorization".to_string());
            },
            OptimizationLevel::Maximum => {
                passes.push("sccp".to_string());
                passes.push("loop-invariant-code-motion".to_string());
                passes.push("inline".to_string());
                passes.push("loop-unroll".to_string());
                passes.push("vectorization".to_string());
                passes.push("loop-tile".to_string());
                passes.push("affine-parallelize".to_string());
            },
        }

        // Add hardware-specific passes
        match self.config.target_hardware.device_type {
            crate::compiler::DeviceType::CPU => {
                passes.push("convert-vector-to-scf".to_string());
                passes.push("convert-linalg-to-llvm".to_string());
            },
            crate::compiler::DeviceType::GPU => {
                passes.push("gpu-kernel-outlining".to_string());
                passes.push("convert-gpu-to-spirv".to_string());
            },
            _ => {
                // Generic lowering
                passes.push("convert-arith-to-llvm".to_string());
            },
        }

        passes
    }

    /// Lower optimized MLIR to target code
    fn lower_to_target(&self, mlir_code: &str) -> Result<Vec<u8>, TrustformersError> {
        let Some(mlir_opt_path) = &self.mlir_opt_path else {
            // Explicit simulation mode (MlirBackend::new_simulated): there is no lowering to
            // perform, so the "compiled artifact" is just the input MLIR text re-encoded as
            // bytes -- not machine code. Callers must check is_real() before treating this as
            // a real compilation result.
            return Ok(mlir_code.as_bytes().to_vec());
        };

        // Create temporary file for optimized MLIR
        let mut temp_file = NamedTempFile::new()
            .map_err(|e| runtime_error(format!("Failed to create temp file: {}", e)))?;

        temp_file
            .write_all(mlir_code.as_bytes())
            .map_err(|e| runtime_error(format!("Failed to write MLIR code: {}", e)))?;

        // Convert to LLVM IR
        let mut cmd = Command::new(mlir_opt_path);
        cmd.arg(temp_file.path())
            .arg("--convert-to-llvm")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = cmd
            .output()
            .map_err(|e| runtime_error(format!("Failed to lower to LLVM: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(compute_error(
                "mlir_compilation",
                format!("MLIR lowering failed: {}", stderr),
            ));
        }

        Ok(output.stdout)
    }

    /// Generate cache key for IR
    fn generate_cache_key(
        &self,
        ir: &IntermediateRepresentation,
    ) -> Result<String, TrustformersError> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash IR content
        ir.instructions.len().hash(&mut hasher);
        ir.dependencies.len().hash(&mut hasher);

        // Hash configuration
        format!("{:?}", self.config.optimization_level).hash(&mut hasher);
        self.config.target_hardware.device_type.hash(&mut hasher);

        Ok(format!("mlir_{:x}", hasher.finish()))
    }

    /// Clear compilation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get supported dialects
    pub fn supported_dialects(&self) -> &DialectSupport {
        &self.dialect_support
    }

    /// Generate optimization statistics from a completed compilation.
    ///
    /// Only quantities this backend actually measures are populated: the pass names the
    /// caller applied, elapsed wall-clock time, and the byte length of the MLIR text before
    /// and after optimization (`code_before`/`code_after` are the actual strings/bytes that
    /// went into and came out of `Self::apply_mlir_passes` or `Self::apply_custom_pipeline`
    /// -- pass the real values, not placeholders). Fused-operation, loop, and
    /// memory-access-optimization counts would require parsing and diffing MLIR IR structure,
    /// which this module does not implement; those fields are `None` rather than a fabricated
    /// number.
    pub fn generate_stats(
        &self,
        passes_applied: Vec<String>,
        start_time: std::time::Instant,
        code_before: &str,
        code_after: &[u8],
    ) -> MlirStats {
        let optimization_time_ms = start_time.elapsed().as_millis() as u64;

        MlirStats {
            passes_applied,
            optimization_time_ms,
            code_size_before: code_before.len(),
            code_size_after: code_after.len(),
            // Not computed: would need structural MLIR analysis this module does not
            // implement. `None` means "not measured", never a fabricated value.
            operations_fused: None,
            loops_optimized: None,
            memory_accesses_optimized: None,
        }
    }

    /// Validate MLIR code.
    ///
    /// When `is_real()` is `true`, this runs `mlir-opt --verify-diagnostics` and reports full
    /// semantic validation. When the backend is in explicit simulation mode
    /// ([`Self::new_simulated`]), there is no semantic validator to call, so this performs a
    /// real (not fabricated) structural sanity check instead -- non-empty, contains the
    /// `module` keyword, and balanced `{}` -- and labels the result as
    /// [`MlirValidationKind::Structural`] so a caller cannot mistake it for semantic
    /// validation. Never silently returns one kind while claiming the other.
    pub fn validate_mlir(&self, mlir_code: &str) -> Result<MlirValidation, TrustformersError> {
        let Some(mlir_opt_path) = &self.mlir_opt_path else {
            return Ok(MlirValidation {
                valid: Self::structural_check(mlir_code),
                kind: MlirValidationKind::Structural,
            });
        };

        // Use mlir-opt to validate
        let mut temp_file = NamedTempFile::new()
            .map_err(|e| runtime_error(format!("Failed to create temp file: {}", e)))?;

        temp_file
            .write_all(mlir_code.as_bytes())
            .map_err(|e| runtime_error(format!("Failed to write MLIR code: {}", e)))?;

        let output = Command::new(mlir_opt_path)
            .arg(temp_file.path())
            .arg("--verify-diagnostics")
            .output()
            .map_err(|e| runtime_error(format!("Failed to validate MLIR: {}", e)))?;

        Ok(MlirValidation {
            valid: output.status.success(),
            kind: MlirValidationKind::Semantic,
        })
    }

    /// Real structural sanity check used only when no `mlir-opt` binary is available: the text
    /// is non-empty, mentions the `module` keyword, and has balanced `{`/`}`. This is a
    /// genuine (if shallow) check on the actual text -- not a fabricated constant -- but it
    /// cannot catch type errors, dialect misuse, or malformed operations the way
    /// `mlir-opt --verify-diagnostics` does.
    fn structural_check(mlir_code: &str) -> bool {
        if mlir_code.trim().is_empty() || !mlir_code.contains("module") {
            return false;
        }

        let mut depth: i64 = 0;
        for ch in mlir_code.chars() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth < 0 {
                        return false;
                    }
                },
                _ => {},
            }
        }

        depth == 0
    }

    /// Get MLIR toolchain version information.
    ///
    /// Requires a real `mlir-opt` binary: there is no meaningful version to report for the
    /// explicit simulation backend ([`Self::new_simulated`]), so this returns a structured
    /// error there instead of a fabricated version string.
    pub fn get_mlir_version(&self) -> Result<String, TrustformersError> {
        let Some(mlir_opt_path) = &self.mlir_opt_path else {
            return Err(runtime_error(
                "MlirBackend is in explicit simulation mode (MlirBackend::new_simulated): there \
                 is no real mlir-opt toolchain to report a version for. Use MlirBackend::new() \
                 with a real mlir-opt binary on PATH or TRUSTFORMERS_MLIR_OPT_PATH instead.",
            ));
        };

        let output = Command::new(mlir_opt_path)
            .arg("--version")
            .output()
            .map_err(|e| runtime_error(format!("Failed to get MLIR version: {}", e)))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Err(runtime_error("Failed to get MLIR version"))
        }
    }
}

/// MLIR dialect support information
#[derive(Debug, Clone)]
pub struct DialectSupport {
    pub tensor: bool,
    pub linalg: bool,
    pub arith: bool,
    pub math: bool,
    pub gpu: bool,
    pub spirv: bool,
    pub llvm: bool,
    pub tosa: bool,
    pub ml_custom: bool, // Custom ML dialect
}

impl Default for DialectSupport {
    fn default() -> Self {
        Self {
            tensor: true,
            linalg: true,
            arith: true,
            math: true,
            gpu: true,
            spirv: false, // Requires SPIRV support
            llvm: true,
            tosa: true,
            ml_custom: true, // Custom ML dialect
        }
    }
}

/// MLIR optimization statistics. See [`MlirBackend::generate_stats`] for which fields are
/// genuinely measured versus `None` because this module does not compute them.
#[derive(Debug, Clone)]
pub struct MlirStats {
    pub passes_applied: Vec<String>,
    pub optimization_time_ms: u64,
    /// Byte length of the MLIR text before optimization.
    pub code_size_before: usize,
    /// Byte length of the (real or simulated) optimized artifact.
    pub code_size_after: usize,
    /// Not computed by this backend (would require MLIR IR structural analysis).
    pub operations_fused: Option<usize>,
    /// Not computed by this backend (would require MLIR IR structural analysis).
    pub loops_optimized: Option<usize>,
    /// Not computed by this backend (would require MLIR IR structural analysis).
    pub memory_accesses_optimized: Option<usize>,
}

/// Which kind of check produced an [`MlirValidation`] result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlirValidationKind {
    /// Full semantic validation performed by the real `mlir-opt --verify-diagnostics`.
    Semantic,
    /// Structural-only sanity check (non-empty, `module` keyword present, balanced braces).
    /// This is NOT semantic validation: it cannot catch type errors, dialect misuse, or
    /// malformed operations. Only produced when no `mlir-opt` binary is available
    /// (`MlirBackend::is_real() == false`).
    Structural,
}

/// Outcome of [`MlirBackend::validate_mlir`], labeled with which kind of check ran so a
/// caller can never mistake a structural sanity check for full semantic validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlirValidation {
    pub valid: bool,
    pub kind: MlirValidationKind,
}

/// Advanced MLIR features for ML workloads
pub struct MlirAdvancedFeatures {
    /// Custom ML dialect support
    pub ml_dialect_enabled: bool,
    /// Automatic batching optimization
    pub auto_batching: bool,
    /// Memory layout optimization
    pub memory_layout_opt: bool,
    /// Kernel fusion strategies
    pub kernel_fusion: KernelFusionStrategy,
    /// Loop tiling configurations
    pub loop_tiling: LoopTilingConfig,
}

/// Kernel fusion strategies
#[derive(Debug, Clone)]
pub enum KernelFusionStrategy {
    /// No fusion
    None,
    /// Basic element-wise fusion
    ElementWise,
    /// Advanced producer-consumer fusion
    ProducerConsumer,
    /// Aggressive fusion with memory analysis
    Aggressive,
}

/// Loop tiling configuration
#[derive(Debug, Clone)]
pub struct LoopTilingConfig {
    pub tile_sizes: Vec<usize>,
    pub enable_cache_optimization: bool,
    pub vectorization_factor: usize,
    pub parallel_dimension: Option<usize>,
}

/// Custom ML dialect operations
pub struct MlDialect;

impl MlDialect {
    /// Generate MLIR for transformer attention
    pub fn attention_operation(batch_size: usize, seq_len: usize, hidden_dim: usize) -> String {
        format!(
            "%attention_out = ml.attention(%query, %key, %value) {{
                batch_size = {},
                seq_len = {},
                hidden_dim = {},
                scale = true
            }} : (tensor<{}x{}x{}xf32>, tensor<{}x{}x{}xf32>, tensor<{}x{}x{}xf32>) -> tensor<{}x{}x{}xf32>",
            batch_size, seq_len, hidden_dim,
            batch_size, seq_len, hidden_dim,
            batch_size, seq_len, hidden_dim,
            batch_size, seq_len, hidden_dim,
            batch_size, seq_len, hidden_dim
        )
    }

    /// Generate MLIR for flash attention
    pub fn flash_attention_operation(
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> String {
        format!(
            "%flash_out = ml.flash_attention(%query, %key, %value) {{
                batch_size = {},
                num_heads = {},
                seq_len = {},
                head_dim = {},
                block_size = 64,
                causal = false
            }} : (tensor<{}x{}x{}x{}xf32>, tensor<{}x{}x{}x{}xf32>, tensor<{}x{}x{}x{}xf32>) -> tensor<{}x{}x{}x{}xf32>",
            batch_size, num_heads, seq_len, head_dim,
            batch_size, num_heads, seq_len, head_dim,
            batch_size, num_heads, seq_len, head_dim,
            batch_size, num_heads, seq_len, head_dim,
            batch_size, num_heads, seq_len, head_dim
        )
    }

    /// Generate MLIR for quantized operations
    pub fn quantized_matmul(m: usize, n: usize, k: usize, bits: u8) -> String {
        format!(
            "%qmatmul_out = ml.quantized_matmul(%a, %b, %scale_a, %zero_a, %scale_b, %zero_b) {{
                bits = {},
                symmetric = false
            }} : (tensor<{}x{}xi{}, tensor<{}x{}xi{}, f32, i{}, f32, i{}) -> tensor<{}x{}xf32>",
            bits, m, k, bits, k, n, bits, bits, bits, m, n
        )
    }
}

/// MLIR pass pipeline builder
pub struct MlirPassPipeline {
    passes: Vec<MlirPass>,
    #[allow(dead_code)]
    target_features: TargetFeatures,
}

/// Individual MLIR pass configuration
#[derive(Debug, Clone)]
pub struct MlirPass {
    pub name: String,
    pub options: HashMap<String, String>,
    pub dependencies: Vec<String>,
}

/// Target-specific features
#[derive(Debug, Clone)]
pub struct TargetFeatures {
    pub vector_width: usize,
    pub cache_line_size: usize,
    pub has_tensor_cores: bool,
    pub memory_hierarchy: MemoryHierarchy,
}

/// Memory hierarchy information
#[derive(Debug, Clone)]
pub struct MemoryHierarchy {
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
    pub memory_bandwidth: f32, // GB/s
}

impl MlirPassPipeline {
    /// Create optimized pass pipeline for transformers
    pub fn transformer_optimized() -> Self {
        let mut passes = Vec::new();

        // Early optimization passes
        passes.push(MlirPass {
            name: "canonicalize".to_string(),
            options: HashMap::new(),
            dependencies: vec![],
        });

        passes.push(MlirPass {
            name: "cse".to_string(),
            options: HashMap::new(),
            dependencies: vec!["canonicalize".to_string()],
        });

        // ML-specific passes
        passes.push(MlirPass {
            name: "ml-attention-fusion".to_string(),
            options: {
                {
                    let mut opts = HashMap::new();
                    opts.insert("enable-flash-attention".to_string(), "true".to_string());
                    opts.insert("block-size".to_string(), "64".to_string());
                    opts
                }
            },
            dependencies: vec!["cse".to_string()],
        });

        passes.push(MlirPass {
            name: "ml-quantization-aware".to_string(),
            options: {
                {
                    let mut opts = HashMap::new();
                    opts.insert("target-bits".to_string(), "8".to_string());
                    opts.insert("symmetric".to_string(), "false".to_string());
                    opts
                }
            },
            dependencies: vec!["ml-attention-fusion".to_string()],
        });

        // Loop optimization passes
        passes.push(MlirPass {
            name: "loop-tile".to_string(),
            options: {
                {
                    let mut opts = HashMap::new();
                    opts.insert("tile-sizes".to_string(), "64,64,32".to_string());
                    opts
                }
            },
            dependencies: vec!["ml-quantization-aware".to_string()],
        });

        passes.push(MlirPass {
            name: "affine-parallelize".to_string(),
            options: HashMap::new(),
            dependencies: vec!["loop-tile".to_string()],
        });

        // Vectorization passes
        passes.push(MlirPass {
            name: "vector-transfer-permutation-lowering".to_string(),
            options: HashMap::new(),
            dependencies: vec!["affine-parallelize".to_string()],
        });

        Self {
            passes,
            target_features: TargetFeatures {
                vector_width: 512, // AVX-512
                cache_line_size: 64,
                has_tensor_cores: false,
                memory_hierarchy: MemoryHierarchy {
                    l1_cache_size: 32 * 1024,       // 32KB
                    l2_cache_size: 256 * 1024,      // 256KB
                    l3_cache_size: 8 * 1024 * 1024, // 8MB
                    memory_bandwidth: 100.0,        // 100 GB/s
                },
            },
        }
    }

    /// Create GPU-optimized pass pipeline
    pub fn gpu_optimized() -> Self {
        // GPU-specific passes
        let passes = vec![
            MlirPass {
                name: "gpu-kernel-outlining".to_string(),
                options: HashMap::new(),
                dependencies: vec![],
            },
            MlirPass {
                name: "gpu-async-region".to_string(),
                options: HashMap::new(),
                dependencies: vec!["gpu-kernel-outlining".to_string()],
            },
            MlirPass {
                name: "convert-gpu-to-spirv".to_string(),
                options: HashMap::new(),
                dependencies: vec!["gpu-async-region".to_string()],
            },
        ];

        Self {
            passes,
            target_features: TargetFeatures {
                vector_width: 1024, // GPU wide vectors
                cache_line_size: 128,
                has_tensor_cores: true,
                memory_hierarchy: MemoryHierarchy {
                    l1_cache_size: 128 * 1024,      // 128KB
                    l2_cache_size: 6 * 1024 * 1024, // 6MB
                    l3_cache_size: 0,               // No L3 on GPU
                    memory_bandwidth: 1000.0,       // 1000 GB/s HBM
                },
            },
        }
    }

    /// Generate pass pipeline string for mlir-opt
    pub fn to_pass_string(&self) -> String {
        let mut pass_args = Vec::new();

        for pass in &self.passes {
            if pass.options.is_empty() {
                pass_args.push(format!("--{}", pass.name));
            } else {
                let options: Vec<String> =
                    pass.options.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
                pass_args.push(format!("--{}={{{}}}", pass.name, options.join(",")));
            }
        }

        pass_args.join(" ")
    }
}

/// MLIR dialect registry
pub struct DialectRegistry {
    registered_dialects: HashMap<String, DialectInfo>,
}

/// Information about a registered dialect
#[derive(Debug, Clone)]
pub struct DialectInfo {
    pub name: String,
    pub version: String,
    pub operations: Vec<String>,
    pub types: Vec<String>,
    pub attributes: Vec<String>,
}

impl Default for DialectRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DialectRegistry {
    /// Create new dialect registry with standard dialects
    pub fn new() -> Self {
        let mut registry = Self {
            registered_dialects: HashMap::new(),
        };

        // Register standard ML dialects
        registry.register_standard_dialects();
        registry
    }

    /// Register standard MLIR dialects
    fn register_standard_dialects(&mut self) {
        // Tensor dialect
        self.registered_dialects.insert(
            "tensor".to_string(),
            DialectInfo {
                name: "tensor".to_string(),
                version: "1.0".to_string(),
                operations: vec![
                    "tensor.empty".to_string(),
                    "tensor.extract".to_string(),
                    "tensor.insert".to_string(),
                    "tensor.reshape".to_string(),
                ],
                types: vec!["tensor".to_string()],
                attributes: vec!["shape".to_string(), "element_type".to_string()],
            },
        );

        // Linalg dialect
        self.registered_dialects.insert(
            "linalg".to_string(),
            DialectInfo {
                name: "linalg".to_string(),
                version: "1.0".to_string(),
                operations: vec![
                    "linalg.matmul".to_string(),
                    "linalg.generic".to_string(),
                    "linalg.conv_2d".to_string(),
                    "linalg.batch_matmul".to_string(),
                ],
                types: vec!["memref".to_string()],
                attributes: vec!["indexing_maps".to_string(), "iterator_types".to_string()],
            },
        );

        // Custom ML dialect
        self.registered_dialects.insert(
            "ml".to_string(),
            DialectInfo {
                name: "ml".to_string(),
                version: "1.0".to_string(),
                operations: vec![
                    "ml.attention".to_string(),
                    "ml.flash_attention".to_string(),
                    "ml.layer_norm".to_string(),
                    "ml.quantized_matmul".to_string(),
                    "ml.embedding_lookup".to_string(),
                ],
                types: vec!["ml.quantized_tensor".to_string()],
                attributes: vec![
                    "scale".to_string(),
                    "zero_point".to_string(),
                    "bits".to_string(),
                ],
            },
        );
    }

    /// Check if dialect is supported
    pub fn is_dialect_supported(&self, dialect: &str) -> bool {
        self.registered_dialects.contains_key(dialect)
    }

    /// Get operations for a dialect
    pub fn get_dialect_operations(&self, dialect: &str) -> Option<&Vec<String>> {
        self.registered_dialects.get(dialect).map(|info| &info.operations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::CompilerConfig;

    /// `MlirBackend::new` must be honest about a missing `mlir-opt`, not silently mock it: on
    /// a machine without a real binary it returns a structured error naming `mlir-opt` and how
    /// to point at one; on a machine that happens to have one, it must genuinely be "real".
    /// Written to pass either way so it tests the actual contract instead of assuming this
    /// machine's toolchain state.
    #[test]
    fn test_mlir_backend_creation_is_honest_about_missing_binary() {
        std::env::remove_var("TRUSTFORMERS_MLIR_OPT_PATH");
        let config = CompilerConfig::default();
        match MlirBackend::new(&config) {
            Ok(backend) => assert!(backend.is_real(), "a constructed backend must be real"),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("mlir-opt"),
                    "error must name the missing binary: {msg}"
                );
                assert!(
                    msg.contains("TRUSTFORMERS_MLIR_OPT_PATH"),
                    "error must say how to point at a custom binary: {msg}"
                );
            },
        }
    }

    #[test]
    fn test_new_simulated_is_explicit_and_labeled_not_real() {
        let config = CompilerConfig::default();
        let backend = MlirBackend::new_simulated(&config);
        assert!(!backend.is_real());
    }

    /// Point the override env var at a program that always exits 0 on `--version` (a
    /// stand-in for a real mlir-opt build in a non-standard location) and verify
    /// `MlirBackend::new` actually uses it rather than falling back silently.
    #[test]
    fn test_mlir_opt_path_env_override_is_used() {
        std::env::set_var("TRUSTFORMERS_MLIR_OPT_PATH", "echo");
        let config = CompilerConfig::default();
        let result = MlirBackend::new(&config);
        std::env::remove_var("TRUSTFORMERS_MLIR_OPT_PATH");
        let backend = result.expect("echo always exits 0, so --version probing must succeed");
        assert!(backend.is_real());
    }

    /// Simulated-mode `apply_mlir_passes`/`apply_custom_pipeline`/`lower_to_target` must be
    /// unmistakably labeled as non-functional output, never presented as real compilation.
    #[test]
    fn test_simulated_backend_output_is_labeled() {
        let config = CompilerConfig::default();
        let backend = MlirBackend::new_simulated(&config);
        let mlir_code = "module {\n  %0 = arith.addf %input0, %input1 : tensor<*xf32>\n}\n";

        let optimized = backend.apply_mlir_passes(mlir_code).expect("simulated apply_mlir_passes");
        assert!(optimized.contains("simulated"));
        assert!(optimized.contains(mlir_code));

        let target = backend.lower_to_target(&optimized).expect("simulated lower_to_target");
        assert_eq!(target, optimized.as_bytes());
    }

    /// In simulated mode, `validate_mlir` performs a real structural check (not a fabricated
    /// constant) and labels it as `Structural` rather than pretending to be semantic.
    #[test]
    fn test_simulated_validate_mlir_is_real_structural_check() {
        let config = CompilerConfig::default();
        let backend = MlirBackend::new_simulated(&config);

        let balanced = backend
            .validate_mlir("module {\n  %0 = arith.addf %a, %b : f32\n}\n")
            .expect("validate_mlir");
        assert!(balanced.valid);
        assert_eq!(balanced.kind, MlirValidationKind::Structural);

        let unbalanced = backend
            .validate_mlir("module { %0 = arith.addf %a, %b : f32")
            .expect("validate_mlir");
        assert!(
            !unbalanced.valid,
            "unbalanced braces must fail the structural check"
        );
        assert_eq!(unbalanced.kind, MlirValidationKind::Structural);

        let no_module_keyword = backend.validate_mlir("{ }").expect("validate_mlir");
        assert!(
            !no_module_keyword.valid,
            "missing `module` keyword must fail"
        );

        let empty = backend.validate_mlir("").expect("validate_mlir");
        assert!(!empty.valid);
    }

    /// There is no real MLIR toolchain behind a simulated backend, so `get_mlir_version` must
    /// error rather than fabricate a version string.
    #[test]
    fn test_simulated_get_mlir_version_errors_instead_of_fabricating() {
        let config = CompilerConfig::default();
        let backend = MlirBackend::new_simulated(&config);
        let error = backend.get_mlir_version().expect_err("simulated backend has no real version");
        assert!(error.to_string().contains("simulation"));
    }

    /// `generate_stats` must report the real byte sizes of what was actually passed in/out,
    /// and must not fabricate the fields it cannot compute.
    #[test]
    fn test_generate_stats_reports_real_byte_sizes_and_options_the_rest() {
        let config = CompilerConfig::default();
        let backend = MlirBackend::new_simulated(&config);
        let before = "module {\n  %0 = arith.addf %a, %b : f32\n}\n";
        let after = backend.apply_mlir_passes(before).expect("simulated apply_mlir_passes");
        let start = std::time::Instant::now();

        let stats = backend.generate_stats(
            vec!["canonicalize".to_string()],
            start,
            before,
            after.as_bytes(),
        );

        assert_eq!(stats.code_size_before, before.len());
        assert_eq!(stats.code_size_after, after.len());
        assert_ne!(
            stats.code_size_before, stats.code_size_after,
            "the simulated banner changes the byte size, so these must differ for this input"
        );
        assert_eq!(stats.operations_fused, None);
        assert_eq!(stats.loops_optimized, None);
        assert_eq!(stats.memory_accesses_optimized, None);
    }

    #[test]
    fn test_dialect_support() {
        let support = DialectSupport::default();
        assert!(support.tensor);
        assert!(support.linalg);
        assert!(support.arith);
        assert!(support.ml_custom);
    }

    #[test]
    fn test_pass_building() {
        // build_optimization_passes only reads config, not the mlir-opt binary; use the
        // always-available simulated constructor so this test does not depend on whether
        // this machine has a real MLIR toolchain installed.
        let config = CompilerConfig::default();
        let backend = MlirBackend::new_simulated(&config);
        let passes = backend.build_optimization_passes();
        assert!(!passes.is_empty());
        assert!(passes.contains(&"canonicalize".to_string()));
    }

    #[test]
    fn test_cache_key_generation() {
        let config = CompilerConfig::default();
        let backend = MlirBackend::new_simulated(&config);
        let ir = IntermediateRepresentation::new();
        let key = backend.generate_cache_key(&ir);
        assert!(key.is_ok());
        assert!(key.expect("operation failed in test").starts_with("mlir_"));
    }

    #[test]
    fn test_advanced_features() {
        let features = MlirAdvancedFeatures {
            ml_dialect_enabled: true,
            auto_batching: true,
            memory_layout_opt: true,
            kernel_fusion: KernelFusionStrategy::Aggressive,
            loop_tiling: LoopTilingConfig {
                tile_sizes: vec![64, 64, 32],
                enable_cache_optimization: true,
                vectorization_factor: 8,
                parallel_dimension: Some(0),
            },
        };

        assert!(features.ml_dialect_enabled);
        assert!(features.auto_batching);
        assert!(matches!(
            features.kernel_fusion,
            KernelFusionStrategy::Aggressive
        ));
    }

    #[test]
    fn test_ml_dialect_operations() {
        let attention_op = MlDialect::attention_operation(4, 512, 768);
        assert!(attention_op.contains("ml.attention"));
        assert!(attention_op.contains("batch_size = 4"));
        assert!(attention_op.contains("seq_len = 512"));
        assert!(attention_op.contains("hidden_dim = 768"));

        let flash_attention_op = MlDialect::flash_attention_operation(2, 8, 1024, 64);
        assert!(flash_attention_op.contains("ml.flash_attention"));
        assert!(flash_attention_op.contains("num_heads = 8"));
        assert!(flash_attention_op.contains("block_size = 64"));

        let quantized_op = MlDialect::quantized_matmul(256, 256, 256, 8);
        assert!(quantized_op.contains("ml.quantized_matmul"));
        assert!(quantized_op.contains("bits = 8"));
    }

    #[test]
    fn test_transformer_pipeline() {
        let pipeline = MlirPassPipeline::transformer_optimized();
        assert!(!pipeline.passes.is_empty());

        // Check for ML-specific passes
        let pass_names: Vec<&String> = pipeline.passes.iter().map(|p| &p.name).collect();
        assert!(pass_names.contains(&&"ml-attention-fusion".to_string()));
        assert!(pass_names.contains(&&"ml-quantization-aware".to_string()));
        assert!(pass_names.contains(&&"loop-tile".to_string()));

        // Test pass string generation
        let pass_string = pipeline.to_pass_string();
        assert!(pass_string.contains("--canonicalize"));
        assert!(pass_string.contains("--ml-attention-fusion"));
    }

    #[test]
    fn test_gpu_pipeline() {
        let pipeline = MlirPassPipeline::gpu_optimized();
        assert!(!pipeline.passes.is_empty());

        let pass_names: Vec<&String> = pipeline.passes.iter().map(|p| &p.name).collect();
        assert!(pass_names.contains(&&"gpu-kernel-outlining".to_string()));
        assert!(pass_names.contains(&&"convert-gpu-to-spirv".to_string()));

        // Check target features for GPU
        assert!(pipeline.target_features.has_tensor_cores);
        assert_eq!(pipeline.target_features.vector_width, 1024);
        assert_eq!(pipeline.target_features.memory_hierarchy.l3_cache_size, 0);
    }

    #[test]
    fn test_dialect_registry() {
        let registry = DialectRegistry::new();

        assert!(registry.is_dialect_supported("tensor"));
        assert!(registry.is_dialect_supported("linalg"));
        assert!(registry.is_dialect_supported("ml"));
        assert!(!registry.is_dialect_supported("unknown"));

        let ml_ops = registry.get_dialect_operations("ml").expect("operation failed in test");
        assert!(ml_ops.contains(&"ml.attention".to_string()));
        assert!(ml_ops.contains(&"ml.flash_attention".to_string()));
        assert!(ml_ops.contains(&"ml.quantized_matmul".to_string()));
    }

    #[test]
    fn test_target_features() {
        let cpu_features = TargetFeatures {
            vector_width: 512,
            cache_line_size: 64,
            has_tensor_cores: false,
            memory_hierarchy: MemoryHierarchy {
                l1_cache_size: 32 * 1024,
                l2_cache_size: 256 * 1024,
                l3_cache_size: 8 * 1024 * 1024,
                memory_bandwidth: 100.0,
            },
        };

        assert_eq!(cpu_features.vector_width, 512);
        assert!(!cpu_features.has_tensor_cores);
        assert_eq!(cpu_features.memory_hierarchy.l1_cache_size, 32 * 1024);

        let gpu_features = TargetFeatures {
            vector_width: 1024,
            cache_line_size: 128,
            has_tensor_cores: true,
            memory_hierarchy: MemoryHierarchy {
                l1_cache_size: 128 * 1024,
                l2_cache_size: 6 * 1024 * 1024,
                l3_cache_size: 0,
                memory_bandwidth: 1000.0,
            },
        };

        assert!(gpu_features.has_tensor_cores);
        assert_eq!(gpu_features.memory_hierarchy.memory_bandwidth, 1000.0);
    }

    #[test]
    fn test_kernel_fusion_strategies() {
        let none = KernelFusionStrategy::None;
        let elementwise = KernelFusionStrategy::ElementWise;
        let producer_consumer = KernelFusionStrategy::ProducerConsumer;
        let aggressive = KernelFusionStrategy::Aggressive;

        assert!(matches!(none, KernelFusionStrategy::None));
        assert!(matches!(elementwise, KernelFusionStrategy::ElementWise));
        assert!(matches!(
            producer_consumer,
            KernelFusionStrategy::ProducerConsumer
        ));
        assert!(matches!(aggressive, KernelFusionStrategy::Aggressive));
    }

    #[test]
    fn test_loop_tiling_config() {
        let config = LoopTilingConfig {
            tile_sizes: vec![64, 64, 32],
            enable_cache_optimization: true,
            vectorization_factor: 8,
            parallel_dimension: Some(0),
        };

        assert_eq!(config.tile_sizes.len(), 3);
        assert_eq!(config.tile_sizes[0], 64);
        assert!(config.enable_cache_optimization);
        assert_eq!(config.vectorization_factor, 8);
        assert_eq!(config.parallel_dimension, Some(0));
    }

    #[test]
    fn test_pass_dependencies() {
        let pipeline = MlirPassPipeline::transformer_optimized();

        // Find ml-attention-fusion pass
        let fusion_pass = pipeline
            .passes
            .iter()
            .find(|p| p.name == "ml-attention-fusion")
            .expect("operation failed in test");

        assert!(fusion_pass.dependencies.contains(&"cse".to_string()));

        // Check options
        assert!(fusion_pass.options.contains_key("enable-flash-attention"));
        assert_eq!(fusion_pass.options["enable-flash-attention"], "true");
    }

    #[test]
    fn test_mlir_stats() {
        // operations_fused/loops_optimized/memory_accesses_optimized are Option<usize>
        // because this backend does not compute them (see MlirBackend::generate_stats) --
        // None here represents "not measured", not zero.
        let stats = MlirStats {
            passes_applied: vec!["canonicalize".to_string(), "cse".to_string()],
            optimization_time_ms: 150,
            code_size_before: 1000,
            code_size_after: 800,
            operations_fused: None,
            loops_optimized: None,
            memory_accesses_optimized: None,
        };

        assert_eq!(stats.passes_applied.len(), 2);
        assert_eq!(stats.optimization_time_ms, 150);
        assert_eq!(stats.operations_fused, None);
        assert_eq!(stats.loops_optimized, None);
        assert_eq!(stats.memory_accesses_optimized, None);

        // Test compression ratio
        let compression_ratio = stats.code_size_after as f32 / stats.code_size_before as f32;
        assert!((compression_ratio - 0.8).abs() < 0.01);
    }
}
