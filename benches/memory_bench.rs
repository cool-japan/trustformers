//! Memory usage benchmarks

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use trustformers_core::{
    tensor::Tensor,
    cache::{InferenceCache, CacheConfig},
    memory::{MemoryPool, AllocationStrategy},
};
use trustformers_models::{
    bert::{BertConfig, BertModel},
    gpt2::{GPT2Config, GPT2Model},
    llama::{LlamaConfig, LlamaModel},
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

// Custom allocator to track memory usage
struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn get_allocated_bytes() -> usize {
    ALLOCATED.load(Ordering::SeqCst)
}

fn reset_memory_tracking() {
    ALLOCATED.store(0, Ordering::SeqCst);
}

fn tensor_memory_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_memory");

    let sizes = vec![
        ("small", vec![128, 128]),
        ("medium", vec![512, 512]),
        ("large", vec![1024, 1024]),
        ("xlarge", vec![2048, 2048]),
        ("3d_small", vec![32, 128, 768]),
        ("3d_large", vec![64, 512, 1024]),
    ];

    for (name, shape) in sizes {
        group.bench_with_input(
            BenchmarkId::new("allocation", name),
            &shape,
            |b, shape| {
                b.iter(|| {
                    reset_memory_tracking();
                    let tensor = Tensor::zeros(shape);
                    let bytes_allocated = get_allocated_bytes();
                    black_box((tensor, bytes_allocated))
                })
            },
        );

        // Test memory usage with operations
        group.bench_with_input(
            BenchmarkId::new("operations", name),
            &shape,
            |b, shape| {
                b.iter(|| {
                    reset_memory_tracking();
                    let a = Tensor::randn(shape).expect("Failed to create tensor");
                    let b = Tensor::randn(shape).expect("Failed to create tensor");
                    let c = a.add(&b);
                    let d = c.mul(&b);
                    let bytes_allocated = get_allocated_bytes();
                    black_box((d, bytes_allocated))
                })
            },
        );
    }

    group.finish();
}

fn model_memory_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_memory");

    // BERT configurations
    let bert_configs = vec![
        ("bert-tiny", BertConfig {
            vocab_size: 30522,
            hidden_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 512,
            ..Default::default()
        }),
        ("bert-small", BertConfig {
            vocab_size: 30522,
            hidden_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            intermediate_size: 1024,
            ..Default::default()
        }),
        ("bert-base", BertConfig {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            ..Default::default()
        }),
    ];

    for (name, config) in bert_configs {
        group.bench_with_input(
            BenchmarkId::new("bert_init", name),
            &config,
            |b, config| {
                b.iter(|| {
                    reset_memory_tracking();
                    let model = BertModel::new(config.clone());
                    let bytes_allocated = get_allocated_bytes();
                    black_box((model, bytes_allocated))
                })
            },
        );
    }

    // GPT-2 configurations
    let gpt2_configs = vec![
        ("gpt2-small", GPT2Config {
            vocab_size: 50257,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            ..Default::default()
        }),
    ];

    for (name, config) in gpt2_configs {
        group.bench_with_input(
            BenchmarkId::new("gpt2_init", name),
            &config,
            |b, config| {
                b.iter(|| {
                    reset_memory_tracking();
                    let model = GPT2Model::new(config.clone());
                    let bytes_allocated = get_allocated_bytes();
                    black_box((model, bytes_allocated))
                })
            },
        );
    }

    group.finish();
}

fn cache_memory_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_memory");

    let cache_configs = vec![
        ("small_cache", CacheConfig {
            max_size_bytes: 100 * 1024 * 1024, // 100MB
            eviction_policy: EvictionPolicy::LRU,
            ttl_seconds: Some(300),
        }),
        ("medium_cache", CacheConfig {
            max_size_bytes: 500 * 1024 * 1024, // 500MB
            eviction_policy: EvictionPolicy::LFU,
            ttl_seconds: Some(600),
        }),
        ("large_cache", CacheConfig {
            max_size_bytes: 1024 * 1024 * 1024, // 1GB
            eviction_policy: EvictionPolicy::FIFO,
            ttl_seconds: None,
        }),
    ];

    for (name, config) in cache_configs {
        group.bench_with_input(
            BenchmarkId::new("cache_init", name),
            &config,
            |b, config| {
                b.iter(|| {
                    reset_memory_tracking();
                    let cache = InferenceCache::new(config.clone());
                    let bytes_allocated = get_allocated_bytes();
                    black_box((cache, bytes_allocated))
                })
            },
        );

        // Test cache with entries
        let cache = InferenceCache::new(config.clone());
        let num_entries = 100;

        group.bench_with_input(
            BenchmarkId::new("cache_fill", name),
            &(cache, num_entries),
            |b, (cache, num_entries)| {
                b.iter(|| {
                    reset_memory_tracking();

                    for i in 0..*num_entries {
                        let key = format!("key_{}", i);
                        let value = Tensor::randn(&[128, 768]).expect("Failed to create tensor");
                        cache.insert(key, value);
                    }

                    let bytes_allocated = get_allocated_bytes();
                    black_box(bytes_allocated)
                })
            },
        );
    }

    group.finish();
}

fn memory_pool_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool");

    let pool_sizes = vec![
        ("small", 10 * 1024 * 1024),    // 10MB
        ("medium", 100 * 1024 * 1024),  // 100MB
        ("large", 500 * 1024 * 1024),   // 500MB
    ];

    let allocation_strategies = vec![
        ("first_fit", AllocationStrategy::FirstFit),
        ("best_fit", AllocationStrategy::BestFit),
        ("buddy", AllocationStrategy::BuddyAllocator),
    ];

    for (size_name, pool_size) in &pool_sizes {
        for (strategy_name, strategy) in &allocation_strategies {
            let pool = MemoryPool::new(*pool_size, strategy.clone());

            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}", size_name, strategy_name), "alloc"),
                &pool,
                |b, pool| {
                    b.iter(|| {
                        // Allocate various sizes
                        let sizes = vec![1024, 4096, 16384, 65536, 262144];
                        let mut allocations = Vec::new();

                        for size in &sizes {
                            if let Ok(alloc) = pool.allocate(*size) {
                                allocations.push(alloc);
                            }
                        }

                        // Deallocate half
                        for (i, alloc) in allocations.iter().enumerate() {
                            if i % 2 == 0 {
                                pool.deallocate(alloc.clone());
                            }
                        }

                        // Try to allocate again
                        for size in &sizes {
                            let _ = pool.allocate(*size);
                        }

                        black_box(pool.get_usage_stats())
                    })
                },
            );
        }
    }

    group.finish();
}

fn gradient_accumulation_memory_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_accumulation");

    let model_size = 768 * 768; // Approximate parameter count
    let batch_sizes = vec![1, 4, 8, 16, 32];
    let accumulation_steps = vec![1, 2, 4, 8];

    for batch_size in &batch_sizes {
        for acc_steps in &accumulation_steps {
            let effective_batch = batch_size * acc_steps;

            group.bench_with_input(
                BenchmarkId::new(
                    format!("batch_{}_acc_{}", batch_size, acc_steps),
                    "memory"
                ),
                &(batch_size, acc_steps),
                |b, (batch_size, acc_steps)| {
                    b.iter(|| {
                        reset_memory_tracking();

                        // Simulate gradient accumulation
                        let mut accumulated_grads = Vec::new();

                        for _ in 0..*acc_steps {
                            // Create gradients for one batch
                            let grads = Tensor::randn(&[*batch_size, model_size])
                                .expect("Failed to create gradients");
                            accumulated_grads.push(grads);
                        }

                        // Sum gradients
                        let total_grad = accumulated_grads.iter()
                            .fold(
                                Tensor::zeros(&[*batch_size, model_size]),
                                |acc, grad| acc.add(grad)
                            );

                        let bytes_allocated = get_allocated_bytes();
                        black_box((total_grad, bytes_allocated))
                    })
                },
            );
        }
    }

    group.finish();
}

fn activation_checkpointing_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_checkpointing");

    let num_layers = 12;
    let hidden_size = 768;
    let seq_length = 512;
    let batch_size = 8;

    // Without checkpointing - store all activations
    group.bench_function("no_checkpointing", |b| {
        b.iter(|| {
            reset_memory_tracking();

            let mut activations = Vec::new();
            let mut current = Tensor::randn(&[batch_size, seq_length, hidden_size])
                .expect("Failed to create tensor");

            for _ in 0..num_layers {
                // Simulate layer forward pass
                let output = current.add(&current); // Simplified operation
                activations.push(current.clone());
                current = output;
            }

            let bytes_allocated = get_allocated_bytes();
            black_box((activations, current, bytes_allocated))
        })
    });

    // With checkpointing - store only checkpoints
    group.bench_function("with_checkpointing", |b| {
        b.iter(|| {
            reset_memory_tracking();

            let checkpoint_interval = 4;
            let mut checkpoints = Vec::new();
            let mut current = Tensor::randn(&[batch_size, seq_length, hidden_size])
                .expect("Failed to create tensor");

            for i in 0..num_layers {
                // Simulate layer forward pass
                let output = current.add(&current); // Simplified operation

                if i % checkpoint_interval == 0 {
                    checkpoints.push(current.clone());
                }

                current = output;
            }

            let bytes_allocated = get_allocated_bytes();
            black_box((checkpoints, current, bytes_allocated))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    tensor_memory_benchmarks,
    model_memory_benchmarks,
    cache_memory_benchmarks,
    memory_pool_benchmarks,
    gradient_accumulation_memory_benchmarks,
    activation_checkpointing_benchmarks
);
criterion_main!(benches);