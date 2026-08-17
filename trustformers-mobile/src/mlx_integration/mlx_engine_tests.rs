//! Unit tests for the MLX-style engine.
//!
//! Split out of `mlx_engine.rs` to keep every source file under the 2000-line limit;
//! included with `#[path]` so the tests keep their `super::` access to the engine's
//! private helpers (`execute_convolution`, `execute_attention`, ...).

use crate::mlx_integration::mlx_types::*;
use crate::mlx_integration::MlxEngine;
use std::collections::HashMap;
use trustformers_core::error::Result;
use trustformers_core::Tensor;

use super::*;

#[test]
fn test_mlx_engine_creation() {
    let mut config = MlxConfig::default();

    // Use conservative settings that should work on most systems
    config.compute_units.cpu_config.performance_cores = 4;
    config.compute_units.cpu_config.efficiency_cores = 2;
    config.compute_units.gpu_config.gpu_cores = 8;
    config.memory_config.max_memory_gb = 8.0;

    let engine = MlxEngine::new(config);

    // Print error for debugging on non-Apple Silicon platforms
    if let Err(ref e) = engine {
        println!("MLX Engine creation failed: {:?}", e);
    }

    // Only assert success on Apple Silicon
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    assert!(engine.is_ok());

    // Allow failure on non-Apple Silicon platforms
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        // Just ensure the function returns some result (Ok or Err)
        let _ = engine;
    }
}

/// Regression: `detect_device_capabilities` used to take the chip enum from the
/// caller and echo a hardcoded table back, so asking for an M4 on an M1 returned
/// M4 specs. It now probes the running machine, so the reported figures must agree
/// with what the OS itself says.
#[test]
fn test_device_capabilities_detection() {
    let caps = match MlxEngine::detect_device_capabilities() {
        Ok(caps) => caps,
        Err(error) => {
            // Non-Apple platforms legitimately cannot probe; that is an error, not
            // an excuse to invent numbers.
            println!("hardware probing unavailable on this platform: {error}");
            return;
        },
    };
    assert!(!caps.cpu_brand.is_empty(), "cpu brand must be probed");
    assert!(caps.logical_cores > 0, "logical core count must be probed");
    assert!(
        caps.unified_memory_gb > 0.5,
        "installed memory must be probed, got {} GiB",
        caps.unified_memory_gb
    );
    // Quantities Apple exposes no API for must stay absent, never guessed.
    assert!(caps.gpu_cores.is_none(), "GPU core count is not queryable");
    assert!(
        caps.neural_engine_tops.is_none(),
        "Neural Engine TOPS is not queryable"
    );
    assert!(
        caps.memory_bandwidth_gbps.is_none(),
        "memory bandwidth is not queryable"
    );
    if let (Some(perf), Some(eff)) = (caps.performance_cores, caps.efficiency_cores) {
        assert_eq!(
            perf + eff,
            caps.logical_cores,
            "probed core clusters must sum to the logical core count"
        );
    }
}

/// The spec table still exists, but under a name that says what it is.
#[test]
fn published_specs_are_labelled_as_specs_not_measurements() {
    let m1 = MlxEngine::published_specs_for(&AppleSiliconDevice::M1);
    let m4 = MlxEngine::published_specs_for(&AppleSiliconDevice::M4);
    let a18 = MlxEngine::published_specs_for(&AppleSiliconDevice::A18Pro);
    assert!(m4.neural_engine_tops > m1.neural_engine_tops);
    assert!(m4.memory_bandwidth_gbps > m1.memory_bandwidth_gbps);
    assert!(!a18.amx_support);
    assert!(m4.amx_support);
    assert_eq!(m4.device, AppleSiliconDevice::M4);
}

/// Regression: `MlxEngine::new(MlxConfig::default())` could never succeed - the
/// default config asked for 8 performance cores and 20 GPU cores while the default
/// device's own table listed 4 and 10.
#[test]
fn default_config_constructs_successfully() {
    match MlxEngine::new(MlxConfig::default()) {
        Ok(engine) => {
            assert!(
                !engine.get_device_capabilities().cpu_brand.is_empty(),
                "a constructed engine must carry probed capabilities"
            );
        },
        Err(error) => {
            // Only acceptable where hardware probing itself is impossible.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            panic!("default config must construct on Apple platforms, got: {error}");
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                let _ = error;
            }
        },
    }
}

#[test]
fn test_model_compilation() {
    let mut config = MlxConfig::default();

    // Use conservative settings
    config.compute_units.cpu_config.performance_cores = 4;
    config.compute_units.cpu_config.efficiency_cores = 2;
    config.compute_units.gpu_config.gpu_cores = 8;
    config.memory_config.max_memory_gb = 8.0;

    let engine_result = MlxEngine::new(config);

    // Skip test if engine creation fails (non-Apple Silicon)
    if engine_result.is_err() {
        println!("Skipping model compilation test - MLX not available");
        return;
    }

    let mut engine = engine_result.expect("Operation failed");

    let model_graph = vec![
        (MlxOperation::MatMul, vec![0, 1], HashMap::new()),
        (MlxOperation::Activation, vec![1], {
            let mut params = HashMap::new();
            params.insert("type".to_string(), 0.0); // ReLU
            params
        }),
    ];

    let result = engine.compile_model("test_model".to_string(), model_graph, OptimizationLevel::O2);

    assert!(result.is_ok());
    assert_eq!(result.expect("Operation failed"), "test_model");
    assert!(engine.compiled_models.contains_key("test_model"));
}

#[test]
fn test_model_execution() {
    let mut config = MlxConfig::default();

    // Use conservative settings
    config.compute_units.cpu_config.performance_cores = 4;
    config.compute_units.cpu_config.efficiency_cores = 2;
    config.compute_units.gpu_config.gpu_cores = 8;
    config.memory_config.max_memory_gb = 8.0;

    let engine_result = MlxEngine::new(config);

    // Skip test if engine creation fails (non-Apple Silicon)
    if engine_result.is_err() {
        println!("Skipping model execution test - MLX not available");
        return;
    }

    let mut engine = engine_result.expect("Operation failed");

    // Compile a simple model
    let model_graph = vec![(MlxOperation::MatMul, vec![0, 0], HashMap::new())];

    engine
        .compile_model("test_model".to_string(), model_graph, OptimizationLevel::O1)
        .expect("Operation failed");

    // Execute the model
    let input1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Operation failed");
    let input2 = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("Operation failed");

    let result = engine.execute_model("test_model", &[input1, input2]);
    assert!(result.is_ok());

    let outputs = result.expect("Operation failed");
    assert!(!outputs.is_empty());
}

#[test]
/// Regression: this used to assert only `is_err()`, which passed for the wrong
/// reason - the *default* config already failed on core counts, so the memory
/// check could have been deleted without the test noticing. Assert the message.
fn test_config_validation() {
    let mut config = MlxConfig::default();
    config.memory_config.max_memory_gb = 1_000_000.0; // more than any Mac has

    let rendered = match MlxEngine::new(config) {
        Ok(_) => panic!("an over-large memory request must fail"),
        Err(error) => format!("{error}"),
    };
    assert!(
        rendered.contains("exceeds the") && rendered.contains("installed on this machine"),
        "the failure must be the memory check specifically, got: {rendered}"
    );

    // And an unsatisfiable core request must fail on the core check, not memory.
    if let Ok(caps) = MlxEngine::detect_device_capabilities() {
        if let Some(available) = caps.performance_cores {
            let mut config = MlxConfig::default();
            config.compute_units.cpu_config.performance_cores =
                u8::try_from(available + 1).unwrap_or(u8::MAX);
            let rendered = match MlxEngine::new(config) {
                Ok(_) => panic!("an over-large core request must fail"),
                Err(error) => format!("{error}"),
            };
            assert!(
                rendered.contains("performance cores"),
                "expected the core-count check, got: {rendered}"
            );
        }
    }
}

/// Regression: `execute_convolution` used to be `Ok(vec![input.clone()])`.
#[test]
fn convolution_actually_convolves() -> Result<()> {
    let engine = match MlxEngine::new(MlxConfig::default()) {
        Ok(engine) => engine,
        Err(_) => return Ok(()),
    };
    // 1x1x3x3 input, 1x1x2x2 kernel of ones, stride 1, no padding -> 2x2 output
    // whose entries are the sums of the 2x2 windows.
    let input = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
    )?;
    let kernel = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2])?;
    let out = engine.execute_convolution(&[input.clone(), kernel], &HashMap::new())?;
    assert_eq!(out.len(), 1);
    let result = &out[0];
    assert_eq!(
        result.shape(),
        vec![1, 1, 2, 2],
        "a real convolution changes the spatial shape; returning the input would keep [1,1,3,3]"
    );
    assert_eq!(result.data()?, vec![12.0, 16.0, 24.0, 28.0]);
    assert_ne!(
        result.data()?,
        input.data()?,
        "the output must not be the unchanged input"
    );
    Ok(())
}

/// Convolution with padding and stride, checked against a hand-computed result.
#[test]
fn convolution_honours_padding_and_stride() -> Result<()> {
    let engine = match MlxEngine::new(MlxConfig::default()) {
        Ok(engine) => engine,
        Err(_) => return Ok(()),
    };
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
    let kernel = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2])?;
    let mut params = HashMap::new();
    params.insert("padding".to_string(), 1.0);
    params.insert("stride".to_string(), 2.0);
    // Padded input is 4x4; a 2x2 kernel with stride 2 yields a 2x2 output whose
    // windows are the four corners of the padded image.
    let out = engine.execute_convolution(&[input, kernel], &params)?;
    assert_eq!(out[0].shape(), vec![1, 1, 2, 2]);
    assert_eq!(out[0].data()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

/// Malformed convolution input must error rather than pass the tensor through.
#[test]
fn convolution_rejects_wrong_rank() {
    let engine = match MlxEngine::new(MlxConfig::default()) {
        Ok(engine) => engine,
        Err(_) => return,
    };
    let input = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).expect("tensor");
    let kernel = Tensor::from_vec(vec![1.0], &[1]).expect("tensor");
    let error = engine
        .execute_convolution(&[input, kernel], &HashMap::new())
        .expect_err("2-D input must be rejected");
    assert!(format!("{error}").contains("4-D"));
}

/// Regression: `execute_attention` used to be `result[i] = input[i] * scale`.
#[test]
fn attention_is_real_softmax_attention() -> Result<()> {
    let engine = match MlxEngine::new(MlxConfig::default()) {
        Ok(engine) => engine,
        Err(_) => return Ok(()),
    };
    // Single head, head_dim 2, seq_len 2, causal.
    let q = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
    let k = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
    let v = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2])?;
    let out = engine.execute_attention(&[q, k.clone(), v], &HashMap::new())?;
    let data = out[0].data()?;
    assert_eq!(out[0].shape(), vec![2, 2]);

    // Row 0 is causally restricted to key 0, so it must be exactly v[0].
    assert!((data[0] - 10.0).abs() < 1e-5, "row0 = {:?}", &data[..2]);
    assert!((data[1] - 20.0).abs() < 1e-5, "row0 = {:?}", &data[..2]);

    // Row 1 attends to both keys with scores 0 and 1/sqrt(2); compute the exact
    // softmax mixture by hand.
    let scale = 1.0f32 / 2.0f32.sqrt();
    let (s0, s1) = (0.0f32, 1.0f32 * scale);
    let (e0, e1) = ((s0 - s1).exp(), 0.0f32.exp());
    let total = e0 + e1;
    let expect_x = (e0 * 10.0 + e1 * 30.0) / total;
    let expect_y = (e0 * 20.0 + e1 * 40.0) / total;
    assert!(
        (data[2] - expect_x).abs() < 1e-3 && (data[3] - expect_y).abs() < 1e-3,
        "row1 {:?} != expected [{expect_x}, {expect_y}]",
        &data[2..]
    );
    Ok(())
}

#[test]
fn attention_requires_query_key_and_value() {
    let engine = match MlxEngine::new(MlxConfig::default()) {
        Ok(engine) => engine,
        Err(_) => return,
    };
    let only_q = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).expect("tensor");
    let error = engine
        .execute_attention(&[only_q], &HashMap::new())
        .expect_err("a single input is not attention");
    assert!(format!("{error}").contains("three input tensors"));
}

/// Regression: the report used to print a fixed 75/85/90% utilisation and 15 W as
/// if measured. It must now contain neither those constants nor any MLX claim.
#[test]
fn performance_report_reports_only_measured_values() {
    let engine = match MlxEngine::new(MlxConfig::default()) {
        Ok(engine) => engine,
        Err(_) => return,
    };
    let report = engine.export_performance_report();
    for fabricated in [
        "- CPU utilization: 75.0%",
        "- GPU utilization: 85.0%",
        "- Neural Engine utilization: 90.0%",
        "- Power consumption: 15.0 W",
    ] {
        assert!(
            !report.contains(fabricated),
            "report still contains the fabricated line {fabricated:?}"
        );
    }
    assert!(
        report.contains("does NOT"),
        "report must state that Apple's MLX is not linked"
    );
    assert!(
        report.contains("Not measured"),
        "report must name what it cannot measure"
    );
    assert!(
        report.contains("Probed hardware"),
        "report must show the probed hardware section"
    );
}

/// After a real run the metrics must reflect that run.
#[test]
fn metrics_reflect_a_real_execution() -> Result<()> {
    let mut engine = match MlxEngine::new(MlxConfig::default()) {
        Ok(engine) => engine,
        Err(_) => return Ok(()),
    };
    // Node inputs index the graph's own node list, so a single node references
    // node 0 twice (mirroring `test_model_execution`); the two runtime tensors are
    // supplied to `execute_model`.
    let graph = vec![(MlxOperation::MatMul, vec![0, 0], HashMap::new())];
    let model_id =
        engine.compile_model("metrics-probe".to_string(), graph, OptimizationLevel::O1)?;
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
    let outputs = engine.execute_model(&model_id, &[a.clone(), b])?;
    assert_eq!(outputs[0].data()?, a.data()?, "A @ I must equal A");

    let metrics = engine.get_performance_metrics();
    assert!(
        metrics.last_execution_ms > 0.0,
        "a real run must record a positive wall-clock duration"
    );
    assert!(
        metrics.last_execution_nodes > 0,
        "a real run must record the nodes it executed"
    );
    assert!(
        metrics.compilation_time_ms > 0.0,
        "compilation must record its own duration"
    );
    Ok(())
}

#[test]
fn test_performance_metrics() {
    let mut config = MlxConfig::default();

    // Use conservative settings
    config.compute_units.cpu_config.performance_cores = 4;
    config.compute_units.cpu_config.efficiency_cores = 2;
    config.compute_units.gpu_config.gpu_cores = 8;
    config.memory_config.max_memory_gb = 8.0;

    let engine_result = MlxEngine::new(config);

    // Skip test if engine creation fails (non-Apple Silicon)
    if engine_result.is_err() {
        println!("Skipping performance metrics test - MLX not available");
        return;
    }

    let engine = engine_result.expect("Operation failed");

    let metrics = engine.get_performance_metrics();
    assert_eq!(metrics.ops_per_second, 0.0);
    assert_eq!(metrics.last_execution_nodes, 0);
    assert_eq!(metrics.pool_memory_gb, 0.0);
    // Before any measurement, unmeasured quantities are absent - not 75/85/90%.
    assert!(metrics.cpu_utilization.is_none());
    assert!(metrics.process_memory_gb.is_none());
}

#[test]
fn test_apple_silicon_variants() {
    let m1 = MlxEngine::published_specs_for(&AppleSiliconDevice::M1);
    let m4 = MlxEngine::published_specs_for(&AppleSiliconDevice::M4);
    let a18 = MlxEngine::published_specs_for(&AppleSiliconDevice::A18Pro);
    assert!(m4.neural_engine_tops > m1.neural_engine_tops);
    assert!(m4.memory_bandwidth_gbps > m1.memory_bandwidth_gbps);
    assert!(!a18.amx_support);
    assert!(m4.amx_support);
}

#[test]
fn test_performance_report() {
    let mut config = MlxConfig::default();

    // Use conservative settings
    config.compute_units.cpu_config.performance_cores = 4;
    config.compute_units.cpu_config.efficiency_cores = 2;
    config.compute_units.gpu_config.gpu_cores = 8;
    config.memory_config.max_memory_gb = 8.0;

    let engine_result = MlxEngine::new(config);

    // Skip test if engine creation fails (non-Apple Silicon)
    if engine_result.is_err() {
        println!("Skipping performance report test - MLX not available");
        return;
    }

    let engine = engine_result.expect("Operation failed");

    let report = engine.export_performance_report();
    assert!(report.contains("MLX-style Engine Report"));
    assert!(report.contains("Probed hardware (sysctlbyname)"));
    assert!(report.contains("Measured execution"));
    assert!(report.contains("Not measured"));
}
