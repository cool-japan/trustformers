//! Benchmark builder for creating complex benchmarks

use super::{BenchmarkConfig, BenchmarkIteration, BenchmarkMetrics, CustomBenchmark};
use anyhow::Result;
use parking_lot::Mutex;
use std::sync::Arc;

/// Builder for creating custom benchmarks
pub struct BenchmarkBuilder {
    name: String,
    description: String,
    stages: Vec<BenchmarkStage>,
    config: BenchmarkConfig,
    tags: Vec<String>,
}

/// A stage in a multi-stage benchmark
pub struct BenchmarkStage {
    pub name: String,
    pub setup: Option<Box<dyn Fn() -> Result<()> + Send + Sync>>,
    pub run: Box<dyn Fn() -> Result<BenchmarkIteration> + Send + Sync>,
    pub teardown: Option<Box<dyn Fn() -> Result<()> + Send + Sync>>,
    pub weight: f64,
}

/// Specification for a benchmark
#[derive(Clone)]
pub struct BenchmarkSpec {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub config: BenchmarkConfig,
}

impl BenchmarkBuilder {
    /// Create a new benchmark builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            stages: Vec::new(),
            config: BenchmarkConfig::default(),
            tags: Vec::new(),
        }
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add tags
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set configuration
    pub fn config(mut self, config: BenchmarkConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a simple stage
    pub fn add_stage<F>(mut self, name: impl Into<String>, run: F) -> Self
    where
        F: Fn() -> Result<BenchmarkIteration> + Send + Sync + 'static,
    {
        self.stages.push(BenchmarkStage {
            name: name.into(),
            setup: None,
            run: Box::new(run),
            teardown: None,
            weight: 1.0,
        });
        self
    }

    /// Add a stage with setup and teardown
    pub fn add_stage_with_lifecycle<S, R, T>(
        mut self,
        name: impl Into<String>,
        setup: S,
        run: R,
        teardown: T,
    ) -> Self
    where
        S: Fn() -> Result<()> + Send + Sync + 'static,
        R: Fn() -> Result<BenchmarkIteration> + Send + Sync + 'static,
        T: Fn() -> Result<()> + Send + Sync + 'static,
    {
        self.stages.push(BenchmarkStage {
            name: name.into(),
            setup: Some(Box::new(setup)),
            run: Box::new(run),
            teardown: Some(Box::new(teardown)),
            weight: 1.0,
        });
        self
    }

    /// Add a weighted stage
    pub fn add_weighted_stage<F>(mut self, name: impl Into<String>, weight: f64, run: F) -> Self
    where
        F: Fn() -> Result<BenchmarkIteration> + Send + Sync + 'static,
    {
        self.stages.push(BenchmarkStage {
            name: name.into(),
            setup: None,
            run: Box::new(run),
            teardown: None,
            weight,
        });
        self
    }

    /// Build the benchmark
    pub fn build(self) -> Result<BuiltBenchmark> {
        if self.stages.is_empty() {
            anyhow::bail!("Benchmark must have at least one stage");
        }

        Ok(BuiltBenchmark {
            name: self.name,
            description: self.description,
            stages: Arc::new(self.stages),
            config: self.config,
            tags: self.tags,
            current_stage: Arc::new(Mutex::new(0)),
        })
    }
}

/// A built custom benchmark
pub struct BuiltBenchmark {
    name: String,
    description: String,
    stages: Arc<Vec<BenchmarkStage>>,
    config: BenchmarkConfig,
    tags: Vec<String>,
    current_stage: Arc<Mutex<usize>>,
}

impl CustomBenchmark for BuiltBenchmark {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn tags(&self) -> Vec<String> {
        self.tags.clone()
    }

    fn setup(&mut self) -> Result<()> {
        // Run setup for all stages
        for stage in self.stages.iter() {
            if let Some(setup) = &stage.setup {
                setup()?;
            }
        }
        Ok(())
    }

    fn run_iteration(&mut self) -> Result<BenchmarkIteration> {
        let total_weight: f64 = self.stages.iter().map(|s| s.weight).sum();
        let random_value = rand::random::<f64>() * total_weight;

        let mut cumulative_weight = 0.0;
        for (i, stage) in self.stages.iter().enumerate() {
            cumulative_weight += stage.weight;
            if random_value <= cumulative_weight {
                *self.current_stage.lock() = i;
                return (stage.run)();
            }
        }

        // Fallback to first stage
        (self.stages[0].run)()
    }

    fn teardown(&mut self) -> Result<()> {
        // Run teardown for all stages
        for stage in self.stages.iter() {
            if let Some(teardown) = &stage.teardown {
                teardown()?;
            }
        }
        Ok(())
    }

    fn config(&self) -> BenchmarkConfig {
        self.config.clone()
    }
}

/// Fluent API for building benchmarks
pub struct BenchmarkDSL;

impl BenchmarkDSL {
    /// Start building a latency benchmark
    pub fn latency_benchmark(name: impl Into<String>) -> LatencyBenchmarkBuilder {
        LatencyBenchmarkBuilder::new(name)
    }

    /// Start building a throughput benchmark
    pub fn throughput_benchmark(name: impl Into<String>) -> ThroughputBenchmarkBuilder {
        ThroughputBenchmarkBuilder::new(name)
    }

    /// Start building a memory benchmark
    pub fn memory_benchmark(name: impl Into<String>) -> MemoryBenchmarkBuilder {
        MemoryBenchmarkBuilder::new(name)
    }
}

/// Builder for latency benchmarks
pub struct LatencyBenchmarkBuilder {
    builder: BenchmarkBuilder,
    percentiles: Vec<f64>,
}

impl LatencyBenchmarkBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            builder: BenchmarkBuilder::new(name),
            percentiles: vec![0.5, 0.9, 0.95, 0.99, 0.999],
        }
    }

    pub fn percentiles(mut self, percentiles: Vec<f64>) -> Self {
        self.percentiles = percentiles;
        self
    }

    pub fn measure<F>(self, name: impl Into<String>, f: F) -> Self
    where
        F: Fn() -> Result<std::time::Duration> + Send + Sync + 'static,
    {
        let stage_name = name.into();
        let builder = self.builder.add_stage(stage_name, move || {
            let duration = f()?;

            let mut metrics = BenchmarkMetrics::default();
            metrics.custom.insert("latency_ms".to_string(), duration.as_secs_f64() * 1000.0);

            Ok(BenchmarkIteration {
                duration,
                metrics,
                validation_passed: None,
                metadata: None,
            })
        });

        Self {
            builder,
            percentiles: self.percentiles,
        }
    }

    pub fn build(self) -> Result<BuiltBenchmark> {
        self.builder.tags(vec!["latency".to_string()]).build()
    }
}

/// Builder for throughput benchmarks
pub struct ThroughputBenchmarkBuilder {
    builder: BenchmarkBuilder,
    batch_size: usize,
}

impl ThroughputBenchmarkBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            builder: BenchmarkBuilder::new(name),
            batch_size: 1,
        }
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn measure<F>(self, name: impl Into<String>, items: usize, f: F) -> Self
    where
        F: Fn() -> Result<std::time::Duration> + Send + Sync + 'static,
    {
        let stage_name = name.into();
        let builder = self.builder.add_stage(stage_name, move || {
            let duration = f()?;
            let throughput = items as f64 / duration.as_secs_f64();

            let metrics = BenchmarkMetrics {
                throughput: Some(throughput),
                ..Default::default()
            };

            Ok(BenchmarkIteration {
                duration,
                metrics,
                validation_passed: None,
                metadata: None,
            })
        });

        Self {
            builder,
            batch_size: self.batch_size,
        }
    }

    pub fn build(self) -> Result<BuiltBenchmark> {
        self.builder.tags(vec!["throughput".to_string()]).build()
    }
}

/// Builder for memory benchmarks
pub struct MemoryBenchmarkBuilder {
    builder: BenchmarkBuilder,
}

impl MemoryBenchmarkBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            builder: BenchmarkBuilder::new(name),
        }
    }

    pub fn measure<F>(self, name: impl Into<String>, f: F) -> Self
    where
        F: Fn() -> Result<(std::time::Duration, usize)> + Send + Sync + 'static,
    {
        let stage_name = name.into();
        let builder = self.builder.add_stage(stage_name, move || {
            let (duration, memory_bytes) = f()?;

            let metrics = BenchmarkMetrics {
                memory_bytes: Some(memory_bytes),
                ..Default::default()
            };

            Ok(BenchmarkIteration {
                duration,
                metrics,
                validation_passed: None,
                metadata: None,
            })
        });

        Self { builder }
    }

    pub fn build(self) -> Result<BuiltBenchmark> {
        self.builder.tags(vec!["memory".to_string()]).build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_benchmark_builder() {
        let benchmark = BenchmarkBuilder::new("test_benchmark")
            .description("Test benchmark")
            .tags(vec!["test".to_string()])
            .add_stage("stage1", || {
                Ok(BenchmarkIteration {
                    duration: Duration::from_millis(10),
                    metrics: BenchmarkMetrics::default(),
                    validation_passed: Some(true),
                    metadata: None,
                })
            })
            .build()
            .unwrap();

        assert_eq!(benchmark.name(), "test_benchmark");
        assert_eq!(benchmark.description(), "Test benchmark");
        assert_eq!(benchmark.tags(), vec!["test"]);
    }

    #[test]
    fn test_latency_benchmark_builder() {
        let benchmark = BenchmarkDSL::latency_benchmark("latency_test")
            .measure("operation", || Ok(Duration::from_millis(50)))
            .build()
            .unwrap();

        assert!(benchmark.tags().contains(&"latency".to_string()));
    }

    #[test]
    fn test_throughput_benchmark_builder() {
        let benchmark = BenchmarkDSL::throughput_benchmark("throughput_test")
            .batch_size(32)
            .measure("process_batch", 32, || Ok(Duration::from_millis(100)))
            .build()
            .unwrap();

        assert!(benchmark.tags().contains(&"throughput".to_string()));
    }
}
