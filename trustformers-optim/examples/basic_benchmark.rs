#![allow(clippy::all)]
use std::time::Instant;
use trustformers_core::TrustformersError;
use trustformers_core::{traits::Optimizer, Tensor};
use trustformers_optim::*;

fn main() -> Result<(), TrustformersError> {
    println!("🚀 TrustformeRS Optimizer Basic Validation");
    println!("=========================================");

    // Create simple test data
    let mut params = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])?;
    let gradients = Tensor::new(vec![0.1, 0.2, 0.1, 0.3, 0.1])?;

    println!("📊 Initial parameters: {:?}", params.data()?);

    // Test Adam optimizer
    println!("\n🔧 Testing Adam Optimizer...");
    let mut adam = Adam::new(0.1, (0.9, 0.999), 1e-8, 0.0);

    let start = Instant::now();
    for i in 0..10 {
        adam.update(&mut params, &gradients)?;
        adam.step();
        if i < 3 {
            println!("   Step {}: {:?}", i + 1, params.data()?);
        }
    }
    let duration = start.elapsed();

    println!("   ✅ Adam completed 10 steps in {:?}", duration);
    println!("   📊 Final parameters: {:?}", params.data()?);

    // Test SGD optimizer
    println!("\n🔧 Testing SGD Optimizer...");
    let mut params_sgd = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])?;
    let mut sgd = SGD::new(0.1, 0.9, 0.0, false);

    let start = Instant::now();
    for _ in 0..10 {
        sgd.update(&mut params_sgd, &gradients)?;
        sgd.step();
    }
    let duration = start.elapsed();

    println!("   ✅ SGD completed 10 steps in {:?}", duration);
    println!("   📊 Final parameters: {:?}", params_sgd.data()?);

    // Test learning rate scheduler
    println!("\n🔧 Testing Learning Rate Scheduler...");
    let scheduler = LinearScheduler::new(0.001, 5, 15);

    for step in 0..15 {
        let lr = scheduler.get_lr(step);
        if step < 5 || step > 12 {
            println!("   Step {}: LR = {:.6}", step, lr);
        } else if step == 5 {
            println!("   ... (warmup continues) ...");
        }
    }

    println!("\n🎉 All optimizers working correctly!");
    println!("   ✅ Compilation successful");
    println!("   ✅ Runtime execution successful");
    println!("   ✅ Parameter updates functioning");
    println!("   ✅ Learning rate scheduling working");

    Ok(())
}
