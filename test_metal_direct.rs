#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    use trustformers_core::gpu_ops::metal::get_metal_backend;
    
    println!("Testing direct Metal backend...");
    
    match get_metal_backend() {
        Ok(backend) => {
            println!("✅ Metal backend initialized successfully!");
            println!("   Metal GPU is available and working");
            
            // Try a simple matmul
            let a = vec![1.0f32; 100];
            let b = vec![2.0f32; 100];
            
            match backend.matmul_f32(&a, &b, 10, 10, 10) {
                Ok(result) => {
                    println!("✅ Metal matmul successful!");
                    println!("   Result size: {}", result.len());
                    println!("   First value: {}", result[0]);
                }
                Err(e) => println!("❌ Metal matmul failed: {}", e),
            }
        }
        Err(e) => {
            println!("❌ Metal backend initialization failed: {}", e);
            println!("   This could mean:");
            println!("   1. Not running on macOS");
            println!("   2. Metal not available on this system");
            println!("   3. Metal feature not enabled");
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() {
    println!("❌ Metal feature not enabled or not on macOS");
    println!("   Compile with: cargo run --features metal");
}
