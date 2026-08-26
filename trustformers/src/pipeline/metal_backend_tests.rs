//! Unit tests for the pipeline Metal backend.
//!
//! Split out of `metal_backend.rs` to keep every source file under the 2000-line
//! limit; included with `#[path]` so the tests keep `super::` access to the module's
//! private helpers.

use super::*;

#[test]
fn config_defaults_are_fp32_shared() {
    let config = MetalBackendConfig::default();
    assert_eq!(config.precision_mode, MetalPrecisionMode::Fp32);
    assert_eq!(config.memory_strategy, MetalMemoryStrategy::Shared);
    assert_eq!(config.device_type, MetalDeviceType::SystemDefault);
}

#[test]
fn apple_silicon_config_requires_unified_memory() {
    let config = MetalBackendConfig::for_apple_silicon();
    assert_eq!(config.device_type, MetalDeviceType::RequireUnifiedMemory);
}

/// Regression: the old backend accepted every precision and then produced the
/// same constant f32 logits regardless.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn unimplemented_precision_is_rejected_not_faked() {
    let config = MetalBackendConfig {
        precision_mode: MetalPrecisionMode::Fp16,
        ..Default::default()
    };
    let rendered = match MetalBackend::new(config) {
        Ok(_) => panic!("fp16 must be rejected, not silently downgraded"),
        Err(error) => format!("{error}"),
    };
    assert!(
        rendered.contains("f32 only"),
        "error must name the real limitation, got: {rendered}"
    );
}

/// Regression: `get_device_capabilities` used to return the same hardcoded struct
/// on every machine. It now reflects the live device, so the reported name is a
/// real GPU name and the buffer limit is the device's own.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn device_capabilities_come_from_the_live_device() -> Result<()> {
    let backend = MetalBackend::new(MetalBackendConfig::default())?;
    let caps = backend.device_capabilities()?;
    assert!(
        !caps.device_name.is_empty(),
        "device name must be read from MTLDevice"
    );
    assert!(
        caps.registry_id != 0,
        "registryID must be a real identifier"
    );
    assert!(
        caps.max_buffer_size > 0,
        "maxBufferLength must be a real device limit"
    );
    assert_ne!(
        caps.max_buffer_size,
        4 * 1024 * 1024 * 1024,
        "4 GiB was the hardcoded constant the mock always reported"
    );
    assert!(
        caps.max_threads_per_threadgroup > 0,
        "maxThreadsPerThreadgroup must be real"
    );
    println!(
        "Metal device: {} (registry {}, unified_memory={}, apple_family={:?}, \
         metal3={}, max_buffer={} MiB, oxicuda={})",
        caps.device_name,
        caps.registry_id,
        caps.unified_memory,
        caps.apple_gpu_family,
        caps.supports_metal3,
        caps.max_buffer_size / (1024 * 1024),
        caps.oxicuda_metal_available
    );
    Ok(())
}

/// Regression: `run_inference` used to return `vec![0.5; 512]` for any input.
/// With no graph and no weights it must now refuse.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn run_inference_without_a_model_errors_instead_of_fabricating() -> Result<()> {
    let backend = MetalBackend::new(MetalBackendConfig::default())?;
    let error = backend
        .run_inference(HashMap::new())
        .expect_err("inference without a graph must fail");
    assert!(
        format!("{error}").contains("no graph declared"),
        "unexpected error: {error}"
    );
    Ok(())
}

/// Regression: `compile_model` used to be `Ok(())` for any path, including
/// nonexistent ones.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn compile_model_rejects_a_missing_checkpoint() -> Result<()> {
    let mut backend = MetalBackend::new(MetalBackendConfig::default())?;
    let missing = std::env::temp_dir().join("trustformers-metal-no-such-checkpoint.safetensors");
    let error = backend
        .compile_model(&missing)
        .expect_err("a missing checkpoint must not compile successfully");
    assert!(
        format!("{error}").contains("not a readable file"),
        "unexpected error: {error}"
    );
    Ok(())
}

/// End-to-end: a real safetensors checkpoint, real GPU matmul + bias + GELU,
/// compared against a CPU reference.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn real_graph_execution_matches_cpu_reference() -> Result<()> {
    use std::io::Write;

    let (m, k, n) = (3usize, 4usize, 5usize);
    let weight: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.25 - 0.75).collect();
    let bias: Vec<f32> = (0..n).map(|i| ((i % 3) as f32) * 0.5 - 0.5).collect();
    let input: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.5 - 1.0).collect();

    // Write a genuine safetensors file into the OS temp dir.
    let path = std::env::temp_dir().join("trustformers-metal-graph-test.safetensors");
    let mut tensors: HashMap<String, (safetensors::Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
    tensors.insert(
        "head.weight".to_string(),
        (
            safetensors::Dtype::F32,
            vec![k, n],
            weight.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    );
    tensors.insert(
        "head.bias".to_string(),
        (
            safetensors::Dtype::F32,
            vec![n],
            bias.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    );
    let views: Vec<(String, TestTensorView)> = tensors
        .into_iter()
        .map(|(name, (dtype, shape, data))| (name, TestTensorView { dtype, shape, data }))
        .collect();
    let serialized = safetensors::serialize(views, None).map_err(|e| {
        TrustformersError::runtime_error(format!("failed to serialize fixture: {e}"))
    })?;
    let mut file = std::fs::File::create(&path)
        .map_err(|e| TrustformersError::runtime_error(format!("failed to create fixture: {e}")))?;
    file.write_all(&serialized)
        .map_err(|e| TrustformersError::runtime_error(format!("failed to write fixture: {e}")))?;
    drop(file);

    let mut backend = MetalBackend::new(MetalBackendConfig::default())?;
    backend.compile_model(&path)?;
    assert_eq!(
        backend.loaded_weight_names(),
        vec!["head.bias", "head.weight"],
        "compile_model must upload the real tensors"
    );

    backend.set_graph(MetalGraph::new(vec![
        MetalOp::MatMul {
            weight: "head.weight".to_string(),
        },
        MetalOp::AddBias {
            bias: "head.bias".to_string(),
        },
        MetalOp::Gelu,
    ]));

    let mut inputs = HashMap::new();
    inputs.insert(
        "input".to_string(),
        Tensor::from_vec(input.clone(), &[m, k]).map_err(map_core)?,
    );
    let outputs = backend.run_inference(inputs)?;
    let logits = outputs
        .get("logits")
        .ok_or_else(|| TrustformersError::runtime_error("missing logits".to_string()))?;
    assert_eq!(logits.shape(), vec![m, n]);
    let got = logits.data().map_err(map_core)?;

    // CPU reference: matmul, bias, exact GELU.
    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += input[row * k + inner] * weight[inner * n + col];
            }
            acc += bias[col];
            // The Metal `gelu` kernel uses the tanh approximation, so the CPU
            // reference must too - matching it against exact erf-GELU would
            // compare two different functions.
            expected[row * n + col] = tanh_gelu(acc);
        }
    }

    // The old mock returned 0.5 for every element; assert we are not that.
    assert!(
        got.iter().any(|v| (v - 0.5).abs() > 1e-6),
        "output must not be the constant 0.5 the mock produced: {got:?}"
    );
    for (index, (actual, want)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 2e-3,
            "element {index}: GPU {actual} vs CPU reference {want}"
        );
    }

    let _ = std::fs::remove_file(&path);
    Ok(())
}

/// Minimal `safetensors::View` implementation so the test can write a real
/// checkpoint without pulling in another dependency.
#[cfg(all(target_os = "macos", feature = "metal"))]
struct TestTensorView {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl safetensors::View for TestTensorView {
    fn dtype(&self) -> safetensors::Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Tanh-approximation GELU, mirroring the MSL `gelu` kernel exactly (including
/// its saturation clamps at +/-10).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn tanh_gelu(x: f32) -> f32 {
    if x > 10.0 {
        return x;
    }
    if x < -10.0 {
        return 0.0;
    }
    // sqrt(2/pi) to f32 precision, matching the MSL kernel's literal.
    let inner = 0.797_884_6_f32 * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}
