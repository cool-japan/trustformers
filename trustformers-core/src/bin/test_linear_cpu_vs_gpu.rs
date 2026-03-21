/// Test a single Linear layer CPU vs GPU to isolate the bug
use anyhow::Result;
#[cfg(all(target_os = "macos", feature = "metal"))]
use trustformers_core::device::Device;
#[cfg(all(target_os = "macos", feature = "metal"))]
use trustformers_core::layers::Linear;
#[cfg(all(target_os = "macos", feature = "metal"))]
use trustformers_core::tensor::Tensor;
#[cfg(all(target_os = "macos", feature = "metal"))]
use trustformers_core::traits::Layer;

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() -> Result<()> {
    println!("================================================================================");
    println!("🔬 Linear Layer CPU vs GPU Test");
    println!("================================================================================\n");

    // Create a Linear layer with realistic dimensions (like rinna model)
    let in_features = 2048; // hidden_size
    let out_features = 44928; // vocab_size
    let has_bias = true;

    println!(
        "Creating Linear layer: {}x{} (bias={})",
        in_features, out_features, has_bias
    );

    // Create CPU version
    let cpu_layer = Linear::new(in_features, out_features, has_bias);

    // Create GPU version with same weights
    let device = Device::metal_if_available(0);
    let mut gpu_layer = Linear::new(in_features, out_features, has_bias);

    // Copy CPU weights to GPU layer
    let cpu_weight = cpu_layer.weight().clone();
    gpu_layer.set_weight(cpu_weight)?;

    if let Some(cpu_bias) = cpu_layer.bias() {
        gpu_layer.set_bias(cpu_bias.clone())?;
    }

    // Upload GPU layer weights to GPU
    println!("Uploading weights to GPU...");
    gpu_layer.weights_to_gpu(&device)?;
    println!("✅ Weights uploaded\n");

    // Create test input (small batch)
    let batch_size = 1;
    let input_data: Vec<f32> = (0..in_features).map(|i| (i as f32) * 0.001).collect();

    println!("Test input (first 10): {:?}\n", &input_data[..10]);

    // CPU forward pass
    println!("🔵 CPU Forward Pass:");
    println!("====================");
    let cpu_input = Tensor::from_slice(&input_data, &[batch_size, in_features])?;
    let cpu_output = cpu_layer.forward(cpu_input)?;

    let cpu_output_data = match &cpu_output {
        Tensor::F32(arr) => {
            let data: Vec<f32> = arr.iter().cloned().collect();
            data
        },
        _ => panic!("Expected F32 tensor"),
    };

    println!("CPU Output (first 10): {:?}", &cpu_output_data[..10]);
    println!(
        "CPU Output (last 10): {:?}",
        &cpu_output_data[cpu_output_data.len() - 10..]
    );

    let cpu_min = cpu_output_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let cpu_max = cpu_output_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let cpu_mean = cpu_output_data.iter().sum::<f32>() / cpu_output_data.len() as f32;
    println!(
        "CPU Stats: min={:.6}, max={:.6}, mean={:.6}\n",
        cpu_min, cpu_max, cpu_mean
    );

    // GPU forward pass
    println!("🟢 GPU Forward Pass:");
    println!("====================");
    let gpu_input = Tensor::from_slice(&input_data, &[batch_size, in_features])?;
    let gpu_input = gpu_input.to_device_enum(&device)?;
    let gpu_output = gpu_layer.forward(gpu_input)?;

    let gpu_output_data = match &gpu_output {
        Tensor::Metal(metal_data) => {
            use trustformers_core::gpu_ops::metal::get_metal_backend;
            let backend = get_metal_backend()?;
            backend.download_buffer_to_vec(&metal_data.buffer_id)?
        },
        Tensor::F32(arr) => {
            let data: Vec<f32> = arr.iter().cloned().collect();
            data
        },
        _ => panic!("Expected Metal or F32 tensor"),
    };

    println!("GPU Output (first 10): {:?}", &gpu_output_data[..10]);
    println!(
        "GPU Output (last 10): {:?}",
        &gpu_output_data[gpu_output_data.len() - 10..]
    );

    let gpu_min = gpu_output_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let gpu_max = gpu_output_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let gpu_mean = gpu_output_data.iter().sum::<f32>() / gpu_output_data.len() as f32;
    println!(
        "GPU Stats: min={:.6}, max={:.6}, mean={:.6}\n",
        gpu_min, gpu_max, gpu_mean
    );

    // Compare
    println!("📊 Comparison:");
    println!("==============");
    println!(
        "Mean difference: {:.6} ({:.2}%)",
        gpu_mean - cpu_mean,
        (gpu_mean - cpu_mean) / cpu_mean.abs() * 100.0
    );
    println!("Max difference: {:.6}", gpu_max - cpu_max);
    println!("Min difference: {:.6}", gpu_min - cpu_min);

    // Element-wise comparison
    let mut max_abs_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut num_large_diffs = 0;

    for i in 0..cpu_output_data.len() {
        let diff = (gpu_output_data[i] - cpu_output_data[i]).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            max_diff_idx = i;
        }
        if diff > 0.01 {
            num_large_diffs += 1;
        }
    }

    println!(
        "\nMax element-wise diff: {:.6} at index {}",
        max_abs_diff, max_diff_idx
    );
    println!(
        "   CPU[{}] = {:.6}",
        max_diff_idx, cpu_output_data[max_diff_idx]
    );
    println!(
        "   GPU[{}] = {:.6}",
        max_diff_idx, gpu_output_data[max_diff_idx]
    );
    println!(
        "   Relative error: {:.2}%",
        max_abs_diff / cpu_output_data[max_diff_idx].abs() * 100.0
    );

    println!("\nElements with >0.01 difference: {}", num_large_diffs);

    // Verdict
    println!("\n================================================================================");
    if max_abs_diff < 0.001 {
        println!("✅ CPU and GPU results MATCH (max diff < 0.001)");
    } else if max_abs_diff < 0.01 {
        println!("⚠️  CPU and GPU results have SMALL differences (max diff < 0.01)");
    } else {
        println!("🔴 CPU and GPU results DIFFER SIGNIFICANTLY (max diff >= 0.01)");
    }
    println!("================================================================================");

    Ok(())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() -> Result<()> {
    println!("This test requires macOS and the 'metal' feature to be enabled.");
    Ok(())
}
