//! Benchmarks for optimization algorithms

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::hint::black_box;
use trustformers_optim::{
    Adam, AdamConfig,
    SGD, SGDConfig,
    Optimizer,
    LearningRateScheduler,
    SchedulerType,
};
use trustformers_core::tensor::Tensor;
use std::collections::HashMap;

fn create_model_parameters(num_params: usize, param_size: usize) -> HashMap<String, Tensor> {
    let mut params = HashMap::new();

    for i in 0..num_params {
        let param_name = format!("param_{}", i);
        let param = Tensor::randn(&[param_size, param_size]).expect("Failed to create parameter");
        params.insert(param_name, param);
    }

    params
}

fn create_gradients(params: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let mut grads = HashMap::new();

    for (name, param) in params {
        let grad = Tensor::randn(param.shape()).expect("Failed to create gradient");
        grads.insert(name.clone(), grad);
    }

    grads
}

fn adam_optimizer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("adam_optimizer");

    let configs = vec![
        ("default", AdamConfig::default()),
        ("low_eps", AdamConfig {
            eps: 1e-10,
            ..Default::default()
        }),
        ("high_lr", AdamConfig {
            lr: 0.001,
            ..Default::default()
        }),
        ("adamw", AdamConfig {
            weight_decay: 0.01,
            ..Default::default()
        }),
    ];

    for (name, config) in configs {
        for param_size in [128, 512, 1024].iter() {
            let num_params = 100;
            let params = create_model_parameters(num_params, *param_size);
            let mut optimizer = Adam::new(params.clone(), config.clone());

            let total_elements = num_params * param_size * param_size;
            group.throughput(Throughput::Elements(total_elements as u64));

            group.bench_with_input(
                BenchmarkId::new(name, param_size),
                &params,
                |b, params| {
                    b.iter(|| {
                        let grads = create_gradients(params);
                        optimizer.step(&grads);
                        black_box(&optimizer);
                    })
                },
            );
        }
    }

    group.finish();
}

fn sgd_optimizer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgd_optimizer");

    let configs = vec![
        ("vanilla", SGDConfig {
            lr: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }),
        ("momentum", SGDConfig {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
            nesterov: false,
        }),
        ("nesterov", SGDConfig {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
            nesterov: true,
        }),
        ("weight_decay", SGDConfig {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
            nesterov: false,
        }),
    ];

    for (name, config) in configs {
        for param_size in [256, 512, 1024].iter() {
            let num_params = 50;
            let params = create_model_parameters(num_params, *param_size);
            let mut optimizer = SGD::new(params.clone(), config.clone());

            let total_elements = num_params * param_size * param_size;
            group.throughput(Throughput::Elements(total_elements as u64));

            group.bench_with_input(
                BenchmarkId::new(name, param_size),
                &params,
                |b, params| {
                    b.iter(|| {
                        let grads = create_gradients(params);
                        optimizer.step(&grads);
                        black_box(&optimizer);
                    })
                },
            );
        }
    }

    group.finish();
}

fn scheduler_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("lr_scheduler");

    let schedulers = vec![
        ("constant", SchedulerType::Constant),
        ("linear", SchedulerType::Linear {
            num_warmup_steps: 1000,
            num_training_steps: 10000,
        }),
        ("cosine", SchedulerType::Cosine {
            num_warmup_steps: 1000,
            num_training_steps: 10000,
            num_cycles: 0.5,
        }),
        ("polynomial", SchedulerType::Polynomial {
            num_warmup_steps: 1000,
            num_training_steps: 10000,
            power: 1.0,
            lr_end: 0.0,
        }),
    ];

    for (name, scheduler_type) in schedulers {
        let mut scheduler = LearningRateScheduler::new(0.001, scheduler_type);

        group.bench_function(name, |b| {
            let mut step = 0;
            b.iter(|| {
                let lr = scheduler.get_lr(step);
                step += 1;
                black_box(lr)
            })
        });
    }

    group.finish();
}

#[cfg(feature = "distributed")]
fn distributed_optimizer_benchmarks(c: &mut Criterion) {
    use trustformers_optim::zero::{ZeroOptimizer, ZeroConfig, ZeroStage};

    let mut group = c.benchmark_group("distributed_optimizer");

    let zero_configs = vec![
        ("zero_stage1", ZeroConfig {
            stage: ZeroStage::Stage1,
            ..Default::default()
        }),
        ("zero_stage2", ZeroConfig {
            stage: ZeroStage::Stage2,
            ..Default::default()
        }),
        ("zero_stage3", ZeroConfig {
            stage: ZeroStage::Stage3,
            ..Default::default()
        }),
    ];

    for (name, zero_config) in zero_configs {
        let param_size = 512;
        let num_params = 100;
        let params = create_model_parameters(num_params, param_size);

        let adam_config = AdamConfig::default();
        let mut optimizer = ZeroOptimizer::new(
            Box::new(Adam::new(params.clone(), adam_config)),
            zero_config,
        );

        let total_elements = num_params * param_size * param_size;
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::new(name, param_size),
            &params,
            |b, params| {
                b.iter(|| {
                    let grads = create_gradients(params);
                    optimizer.step(&grads);
                    black_box(&optimizer);
                })
            },
        );
    }

    group.finish();
}

fn memory_efficiency_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_memory");

    // Measure peak memory usage for different optimizers
    let param_sizes = vec![256, 512, 1024, 2048];

    for param_size in param_sizes {
        let num_params = 20; // Fewer params for memory tests
        let params = create_model_parameters(num_params, param_size);

        // Adam (requires momentum buffers)
        group.bench_with_input(
            BenchmarkId::new("adam_memory", param_size),
            &params,
            |b, params| {
                b.iter(|| {
                    let optimizer = Adam::new(params.clone(), AdamConfig::default());
                    black_box(optimizer);
                })
            },
        );

        // SGD (minimal memory)
        group.bench_with_input(
            BenchmarkId::new("sgd_memory", param_size),
            &params,
            |b, params| {
                b.iter(|| {
                    let optimizer = SGD::new(params.clone(), SGDConfig::default());
                    black_box(optimizer);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    adam_optimizer_benchmarks,
    sgd_optimizer_benchmarks,
    scheduler_benchmarks,
    memory_efficiency_benchmarks
);

#[cfg(feature = "distributed")]
criterion_group!(distributed_benches, distributed_optimizer_benchmarks);

#[cfg(not(feature = "distributed"))]
criterion_main!(benches);

#[cfg(feature = "distributed")]
criterion_main!(benches, distributed_benches);