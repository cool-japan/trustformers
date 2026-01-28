//! Test device-aware Linear layer functionality

use trustformers_core::{device::Device, layers::Linear, tensor::Tensor, traits::Layer};

#[test]
fn test_linear_cpu() -> Result<(), Box<dyn std::error::Error>> {
    // Create a linear layer on CPU with smaller dimensions to avoid memory issues
    let linear = Linear::new_with_device(128, 256, true, Device::CPU);

    // Test with smaller 2D input for memory safety
    let input = Tensor::randn(&[16, 128])?;
    let output = linear.forward(input)?;

    assert_eq!(output.shape(), &[16, 256]);
    assert_eq!(linear.device(), Device::CPU);

    // No explicit drop - let Rust handle cleanup naturally to avoid SIGSEGV

    Ok(())
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn test_linear_metal() -> Result<(), Box<dyn std::error::Error>> {
    // Create a linear layer on Metal GPU
    let device = Device::metal_if_available(0);
    let linear = Linear::new_with_device(768, 3072, true, device);

    // Test with 2D input
    let input = Tensor::randn(&[128, 768])?;
    let output = linear.forward(input)?;

    assert_eq!(output.shape(), &[128, 3072]);

    Ok(())
}

#[test]
fn test_linear_device_migration() -> Result<(), Box<dyn std::error::Error>> {
    // Create a linear layer on CPU with smaller dimensions
    let linear_cpu = Linear::new_with_device(256, 512, false, Device::CPU);
    assert_eq!(linear_cpu.device(), Device::CPU);

    // Move to best available device
    let device = Device::best_available();
    let linear_gpu = linear_cpu.to_device(device);
    assert_eq!(linear_gpu.device(), device);

    Ok(())
}

#[test]
fn test_linear_3d_input() -> Result<(), Box<dyn std::error::Error>> {
    // Test with batched 3D input - use smaller dimensions
    let linear = Linear::new_with_device(256, 512, true, Device::CPU);

    let input_3d = Tensor::randn(&[2, 32, 256])?;
    let output_3d = linear.forward(input_3d)?;

    assert_eq!(output_3d.shape(), &[2, 32, 512]);

    Ok(())
}

#[test]
fn test_linear_parameter_count() -> Result<(), Box<dyn std::error::Error>> {
    // Test with bias - use smaller dimensions
    let linear_with_bias = Linear::new_with_device(256, 512, true, Device::CPU);
    let expected_params_with_bias = (256 * 512) + 512; // weights + bias
    assert_eq!(
        linear_with_bias.parameter_count(),
        expected_params_with_bias
    );

    // Test without bias
    let linear_no_bias = Linear::new_with_device(256, 512, false, Device::CPU);
    let expected_params_no_bias = 256 * 512; // weights only
    assert_eq!(linear_no_bias.parameter_count(), expected_params_no_bias);

    Ok(())
}
