//! Benchmark registry for managing and discovering benchmarks

use super::CustomBenchmark;
use anyhow::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Type alias for benchmark factory function
pub type BenchmarkFactory = Box<dyn Fn() -> Box<dyn CustomBenchmark> + Send + Sync>;

/// Metadata about a registered benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    /// Unique name
    pub name: String,
    /// Description
    pub description: String,
    /// Category
    pub category: BenchmarkCategory,
    /// Tags for filtering
    pub tags: Vec<String>,
    /// Author
    pub author: Option<String>,
    /// Version
    pub version: Option<String>,
    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Benchmark categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BenchmarkCategory {
    /// Model inference benchmarks
    Inference,
    /// Training benchmarks
    Training,
    /// Memory benchmarks
    Memory,
    /// I/O benchmarks
    IO,
    /// Tokenization benchmarks
    Tokenization,
    /// Custom category
    Custom(String),
}

/// Global benchmark registry
pub struct BenchmarkRegistry {
    benchmarks: Arc<RwLock<HashMap<String, RegisteredBenchmark>>>,
    categories: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

/// A registered benchmark
struct RegisteredBenchmark {
    metadata: BenchmarkMetadata,
    factory: BenchmarkFactory,
}

impl Default for BenchmarkRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            benchmarks: Arc::new(RwLock::new(HashMap::new())),
            categories: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the global registry instance
    pub fn global() -> &'static Self {
        static REGISTRY: once_cell::sync::Lazy<BenchmarkRegistry> =
            once_cell::sync::Lazy::new(BenchmarkRegistry::new);
        &REGISTRY
    }

    /// Register a benchmark
    pub fn register<F>(&self, metadata: BenchmarkMetadata, factory: F) -> Result<()>
    where
        F: Fn() -> Box<dyn CustomBenchmark> + Send + Sync + 'static,
    {
        let name = metadata.name.clone();
        let category = format!("{:?}", metadata.category);

        if self.benchmarks.read().contains_key(&name) {
            anyhow::bail!("Benchmark '{}' already registered", name);
        }

        let registered = RegisteredBenchmark {
            metadata: metadata.clone(),
            factory: Box::new(factory),
        };

        self.benchmarks.write().insert(name.clone(), registered);

        // Update category index
        self.categories.write().entry(category).or_default().push(name);

        Ok(())
    }

    /// Register a benchmark with builder pattern
    pub fn register_with_builder(&self) -> RegistrationBuilder<'_> {
        RegistrationBuilder::new(self)
    }

    /// Create a benchmark instance by name
    pub fn create(&self, name: &str) -> Result<Box<dyn CustomBenchmark>> {
        let benchmarks = self.benchmarks.read();
        let registered = benchmarks
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Benchmark '{}' not found", name))?;

        Ok((registered.factory)())
    }

    /// List all registered benchmarks
    pub fn list(&self) -> Vec<BenchmarkMetadata> {
        self.benchmarks.read().values().map(|r| r.metadata.clone()).collect()
    }

    /// List benchmarks by category
    pub fn list_by_category(&self, category: BenchmarkCategory) -> Vec<BenchmarkMetadata> {
        let category_str = format!("{:?}", category);
        let categories = self.categories.read();

        if let Some(names) = categories.get(&category_str) {
            let benchmarks = self.benchmarks.read();
            names
                .iter()
                .filter_map(|name| benchmarks.get(name).map(|r| r.metadata.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Search benchmarks by tags
    pub fn search_by_tags(&self, tags: &[String]) -> Vec<BenchmarkMetadata> {
        self.benchmarks
            .read()
            .values()
            .filter(|r| tags.iter().any(|tag| r.metadata.tags.contains(tag)))
            .map(|r| r.metadata.clone())
            .collect()
    }

    /// Get benchmark metadata
    pub fn get_metadata(&self, name: &str) -> Option<BenchmarkMetadata> {
        self.benchmarks.read().get(name).map(|r| r.metadata.clone())
    }

    /// Remove a benchmark
    pub fn unregister(&self, name: &str) -> Result<()> {
        let mut benchmarks = self.benchmarks.write();
        let registered = benchmarks
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("Benchmark '{}' not found", name))?;

        // Remove from category index
        let category = format!("{:?}", registered.metadata.category);
        if let Some(names) = self.categories.write().get_mut(&category) {
            names.retain(|n| n != name);
        }

        Ok(())
    }

    /// Clear all registrations
    pub fn clear(&self) {
        self.benchmarks.write().clear();
        self.categories.write().clear();
    }
}

/// Builder for registering benchmarks
pub struct RegistrationBuilder<'a> {
    registry: &'a BenchmarkRegistry,
    name: Option<String>,
    description: Option<String>,
    category: Option<BenchmarkCategory>,
    tags: Vec<String>,
    author: Option<String>,
    version: Option<String>,
    dependencies: Vec<String>,
}

impl<'a> RegistrationBuilder<'a> {
    fn new(registry: &'a BenchmarkRegistry) -> Self {
        Self {
            registry,
            name: None,
            description: None,
            category: None,
            tags: Vec::new(),
            author: None,
            version: None,
            dependencies: Vec::new(),
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn category(mut self, category: BenchmarkCategory) -> Self {
        self.category = Some(category);
        self
    }

    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    pub fn dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    pub fn register<F>(self, factory: F) -> Result<()>
    where
        F: Fn() -> Box<dyn CustomBenchmark> + Send + Sync + 'static,
    {
        let metadata = BenchmarkMetadata {
            name: self.name.ok_or_else(|| anyhow::anyhow!("Name is required"))?,
            description: self.description.unwrap_or_default(),
            category: self
                .category
                .unwrap_or(BenchmarkCategory::Custom("uncategorized".to_string())),
            tags: self.tags,
            author: self.author,
            version: self.version,
            dependencies: self.dependencies,
        };

        self.registry.register(metadata, factory)
    }
}

/// Macro for easy benchmark registration
#[macro_export]
macro_rules! register_benchmark {
    ($benchmark_type:ty) => {
        $crate::performance::custom_benchmarks::BenchmarkRegistry::global()
            .register_with_builder()
            .name(stringify!($benchmark_type))
            .description(concat!("Benchmark: ", stringify!($benchmark_type)))
            .register(|| Box::new(<$benchmark_type>::new()))
            .expect(concat!("Failed to register ", stringify!($benchmark_type)));
    };

    ($benchmark_type:ty, $category:expr) => {
        $crate::performance::custom_benchmarks::BenchmarkRegistry::global()
            .register_with_builder()
            .name(stringify!($benchmark_type))
            .description(concat!("Benchmark: ", stringify!($benchmark_type)))
            .category($category)
            .register(|| Box::new(<$benchmark_type>::new()))
            .expect(concat!("Failed to register ", stringify!($benchmark_type)));
    };
}

/// Decorator for automatic registration (requires inventory crate in practice)
pub fn auto_register(_metadata: BenchmarkMetadata) -> impl Fn() {
    || {
        // In practice, this would use the inventory crate for automatic registration
        println!("Auto-registration placeholder");
    }
}

/// Benchmark suite for grouping related benchmarks
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    pub name: String,
    pub description: String,
    pub benchmarks: Vec<String>,
}

impl BenchmarkSuite {
    /// Create a new suite
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            benchmarks: Vec::new(),
        }
    }

    /// Add a benchmark to the suite
    pub fn add_benchmark(mut self, name: impl Into<String>) -> Self {
        self.benchmarks.push(name.into());
        self
    }

    /// Run all benchmarks in the suite
    pub fn run(&self, registry: &BenchmarkRegistry) -> Result<Vec<super::BenchmarkReport>> {
        use crate::performance::custom_benchmarks::{BenchmarkRunner, RunConfig};

        let mut runner = BenchmarkRunner::new(RunConfig::default());

        for name in &self.benchmarks {
            let benchmark = registry.create(name)?;
            runner = runner.add_benchmark(benchmark);
        }

        runner.run()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::custom_benchmarks::ExampleBenchmark;

    #[test]
    fn test_registry() {
        let registry = BenchmarkRegistry::new();

        // Register a benchmark
        let metadata = BenchmarkMetadata {
            name: "test_bench".to_string(),
            description: "Test benchmark".to_string(),
            category: BenchmarkCategory::Inference,
            tags: vec!["test".to_string()],
            author: None,
            version: None,
            dependencies: vec![],
        };

        registry
            .register(metadata, || {
                Box::new(ExampleBenchmark::new("test".to_string(), 32, 128))
            })
            .unwrap();

        // List benchmarks
        let all = registry.list();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].name, "test_bench");

        // Create instance
        let benchmark = registry.create("test_bench").unwrap();
        assert_eq!(benchmark.name(), "example_benchmark");

        // Search by tags
        let found = registry.search_by_tags(&["test".to_string()]);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_registration_builder() {
        let registry = BenchmarkRegistry::new();

        registry
            .register_with_builder()
            .name("builder_test")
            .description("Test with builder")
            .category(BenchmarkCategory::Memory)
            .tags(vec!["builder".to_string(), "test".to_string()])
            .author("Test Author")
            .version("1.0.0")
            .register(|| Box::new(ExampleBenchmark::new("test".to_string(), 16, 64)))
            .unwrap();

        let metadata = registry.get_metadata("builder_test").unwrap();
        assert_eq!(metadata.author, Some("Test Author".to_string()));
        assert_eq!(metadata.version, Some("1.0.0".to_string()));
    }

    #[test]
    fn test_benchmark_suite() {
        let registry = BenchmarkRegistry::new();

        // Register some benchmarks
        for i in 0..3 {
            let name = format!("suite_bench_{}", i);
            registry
                .register_with_builder()
                .name(name.clone())
                .category(BenchmarkCategory::Inference)
                .register(move || Box::new(ExampleBenchmark::new(format!("model_{}", i), 32, 128)))
                .unwrap();
        }

        // Create suite
        let suite = BenchmarkSuite::new("test_suite", "Test benchmark suite")
            .add_benchmark("suite_bench_0")
            .add_benchmark("suite_bench_1")
            .add_benchmark("suite_bench_2");

        assert_eq!(suite.benchmarks.len(), 3);
    }
}
