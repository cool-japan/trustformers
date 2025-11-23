// Dataset loading and management for evaluation
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Dataset sample for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSample {
    pub input: String,
    pub target: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Dataset collection for evaluation tasks
#[derive(Debug, Clone)]
pub struct EvaluationDataset {
    pub name: String,
    pub samples: Vec<DatasetSample>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl EvaluationDataset {
    pub fn new(name: String) -> Self {
        Self {
            name,
            samples: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_sample(&mut self, sample: DatasetSample) {
        self.samples.push(sample);
    }

    pub fn add_samples(&mut self, samples: Vec<DatasetSample>) {
        self.samples.extend(samples);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn get_inputs(&self) -> Vec<String> {
        self.samples.iter().map(|s| s.input.clone()).collect()
    }

    pub fn get_targets(&self) -> Vec<String> {
        self.samples.iter().map(|s| s.target.clone()).collect()
    }

    pub fn sample(&self, n: usize) -> EvaluationDataset {
        let mut sampled = self.clone();
        if n < self.samples.len() {
            sampled.samples.truncate(n);
        }
        sampled
    }

    pub fn shuffle(&mut self, seed: Option<u64>) {
        use scirs2_core::random::*;

        if let Some(seed) = seed {
            let mut rng = StdRng::seed_from_u64(seed);
            self.samples.shuffle(&mut rng);
        } else {
            let mut rng = thread_rng();
            self.samples.shuffle(&mut rng);
        }
    }

    pub fn split(&self, train_ratio: f64) -> (EvaluationDataset, EvaluationDataset) {
        let split_idx = (self.samples.len() as f64 * train_ratio) as usize;

        let mut train_dataset = EvaluationDataset::new(format!("{}_train", self.name));
        train_dataset.samples = self.samples[..split_idx].to_vec();
        train_dataset.metadata = self.metadata.clone();

        let mut test_dataset = EvaluationDataset::new(format!("{}_test", self.name));
        test_dataset.samples = self.samples[split_idx..].to_vec();
        test_dataset.metadata = self.metadata.clone();

        (train_dataset, test_dataset)
    }
}

/// Dataset loader trait
pub trait DatasetLoader {
    fn load(&self, dataset_name: &str, split: &str) -> Result<EvaluationDataset>;
    fn available_datasets(&self) -> Vec<String>;
    fn available_splits(&self, dataset_name: &str) -> Vec<String>;
}

/// File-based dataset loader
pub struct FileDatasetLoader {
    data_dir: String,
}

impl FileDatasetLoader {
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_string_lossy().to_string(),
        }
    }

    fn get_dataset_path(&self, dataset_name: &str, split: &str) -> String {
        format!("{}/{}/{}.jsonl", self.data_dir, dataset_name, split)
    }

    fn load_jsonl(&self, path: &str) -> Result<Vec<DatasetSample>> {
        let content = std::fs::read_to_string(path)?;
        let mut samples = Vec::new();

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let json_value: serde_json::Value = serde_json::from_str(line)?;

            let input = json_value
                .get("input")
                .or_else(|| json_value.get("text"))
                .or_else(|| json_value.get("sentence"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let target = json_value
                .get("target")
                .or_else(|| json_value.get("label"))
                .or_else(|| json_value.get("output"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let mut metadata = HashMap::new();
            if let Some(obj) = json_value.as_object() {
                for (key, value) in obj {
                    if key != "input"
                        && key != "text"
                        && key != "sentence"
                        && key != "target"
                        && key != "label"
                        && key != "output"
                    {
                        metadata.insert(key.clone(), value.clone());
                    }
                }
            }

            samples.push(DatasetSample {
                input,
                target,
                metadata,
            });
        }

        Ok(samples)
    }
}

impl DatasetLoader for FileDatasetLoader {
    fn load(&self, dataset_name: &str, split: &str) -> Result<EvaluationDataset> {
        let path = self.get_dataset_path(dataset_name, split);
        let samples = self.load_jsonl(&path)?;

        let mut dataset = EvaluationDataset::new(format!("{}_{}", dataset_name, split));
        dataset.add_samples(samples);

        // Add dataset metadata
        dataset.metadata.insert(
            "source".to_string(),
            serde_json::Value::String("file".to_string()),
        );
        dataset.metadata.insert("path".to_string(), serde_json::Value::String(path));
        dataset.metadata.insert(
            "dataset_name".to_string(),
            serde_json::Value::String(dataset_name.to_string()),
        );
        dataset.metadata.insert(
            "split".to_string(),
            serde_json::Value::String(split.to_string()),
        );

        Ok(dataset)
    }

    fn available_datasets(&self) -> Vec<String> {
        let data_path = Path::new(&self.data_dir);
        if !data_path.exists() {
            return Vec::new();
        }

        let mut datasets = Vec::new();
        if let Ok(entries) = std::fs::read_dir(data_path) {
            for entry in entries.flatten() {
                if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                    if let Some(name) = entry.file_name().to_str() {
                        datasets.push(name.to_string());
                    }
                }
            }
        }

        datasets.sort();
        datasets
    }

    fn available_splits(&self, dataset_name: &str) -> Vec<String> {
        let dataset_path = Path::new(&self.data_dir).join(dataset_name);
        if !dataset_path.exists() {
            return Vec::new();
        }

        let mut splits = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dataset_path) {
            for entry in entries.flatten() {
                if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                    if let Some(name) = entry.file_name().to_str() {
                        if name.ends_with(".jsonl") {
                            let split_name = name.strip_suffix(".jsonl").unwrap_or(name);
                            splits.push(split_name.to_string());
                        }
                    }
                }
            }
        }

        splits.sort();
        splits
    }
}

/// In-memory dataset loader for testing
pub struct MemoryDatasetLoader {
    datasets: HashMap<String, HashMap<String, EvaluationDataset>>,
}

impl MemoryDatasetLoader {
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
        }
    }

    pub fn add_dataset(&mut self, dataset: EvaluationDataset, dataset_name: &str, split: &str) {
        self.datasets
            .entry(dataset_name.to_string())
            .or_default()
            .insert(split.to_string(), dataset);
    }

    pub fn create_dummy_glue_datasets(&mut self) {
        // Create dummy GLUE datasets for testing
        self.create_dummy_classification_dataset("cola", "train", 1000, vec!["0", "1"]);
        self.create_dummy_classification_dataset("cola", "validation", 200, vec!["0", "1"]);

        self.create_dummy_classification_dataset(
            "sst2",
            "train",
            2000,
            vec!["negative", "positive"],
        );
        self.create_dummy_classification_dataset(
            "sst2",
            "validation",
            400,
            vec!["negative", "positive"],
        );

        self.create_dummy_classification_dataset(
            "mrpc",
            "train",
            1500,
            vec!["not_equivalent", "equivalent"],
        );
        self.create_dummy_classification_dataset(
            "mrpc",
            "validation",
            300,
            vec!["not_equivalent", "equivalent"],
        );

        self.create_dummy_classification_dataset(
            "mnli",
            "train",
            10000,
            vec!["entailment", "neutral", "contradiction"],
        );
        self.create_dummy_classification_dataset(
            "mnli",
            "validation_matched",
            2000,
            vec!["entailment", "neutral", "contradiction"],
        );
        self.create_dummy_classification_dataset(
            "mnli",
            "validation_mismatched",
            2000,
            vec!["entailment", "neutral", "contradiction"],
        );
    }

    fn create_dummy_classification_dataset(
        &mut self,
        name: &str,
        split: &str,
        size: usize,
        labels: Vec<&str>,
    ) {
        let mut samples = Vec::new();

        for i in 0..size {
            let input = match name {
                "cola" => format!("This is sentence number {} for acceptability.", i),
                "sst2" => {
                    if i % 2 == 0 {
                        format!("This is a positive movie review {}.", i)
                    } else {
                        format!("This is a negative movie review {}.", i)
                    }
                },
                "mrpc" => format!("Sentence A {}. [SEP] Sentence B {}.", i, i + 1),
                "mnli" => format!("Premise sentence {}. [SEP] Hypothesis sentence {}.", i, i),
                _ => format!("Input text {} for task {}.", i, name),
            };

            let target = labels[i % labels.len()].to_string();

            let mut metadata = HashMap::new();
            metadata.insert("idx".to_string(), serde_json::Value::Number(i.into()));
            metadata.insert(
                "task".to_string(),
                serde_json::Value::String(name.to_string()),
            );

            samples.push(DatasetSample {
                input,
                target,
                metadata,
            });
        }

        let mut dataset = EvaluationDataset::new(format!("{}_{}", name, split));
        dataset.add_samples(samples);
        dataset.metadata.insert(
            "source".to_string(),
            serde_json::Value::String("memory".to_string()),
        );
        dataset.metadata.insert(
            "task_type".to_string(),
            serde_json::Value::String("classification".to_string()),
        );
        dataset.metadata.insert(
            "num_labels".to_string(),
            serde_json::Value::Number(labels.len().into()),
        );

        self.add_dataset(dataset, name, split);
    }
}

impl DatasetLoader for MemoryDatasetLoader {
    fn load(&self, dataset_name: &str, split: &str) -> Result<EvaluationDataset> {
        self.datasets
            .get(dataset_name)
            .and_then(|splits| splits.get(split))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Dataset {}:{} not found", dataset_name, split))
    }

    fn available_datasets(&self) -> Vec<String> {
        let mut datasets: Vec<String> = self.datasets.keys().cloned().collect();
        datasets.sort();
        datasets
    }

    fn available_splits(&self, dataset_name: &str) -> Vec<String> {
        self.datasets
            .get(dataset_name)
            .map(|splits| {
                let mut split_names: Vec<String> = splits.keys().cloned().collect();
                split_names.sort();
                split_names
            })
            .unwrap_or_default()
    }
}

impl Default for MemoryDatasetLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Dataset manager for coordinating multiple loaders
pub struct DatasetManager {
    loaders: HashMap<String, Box<dyn DatasetLoader>>,
    default_loader: String,
}

impl DatasetManager {
    pub fn new() -> Self {
        let mut manager = Self {
            loaders: HashMap::new(),
            default_loader: "memory".to_string(),
        };

        // Register default memory loader
        manager.register_loader("memory".to_string(), Box::new(MemoryDatasetLoader::new()));

        manager
    }

    pub fn register_loader(&mut self, name: String, loader: Box<dyn DatasetLoader>) {
        self.loaders.insert(name, loader);
    }

    pub fn register_file_loader<P: AsRef<Path>>(&mut self, name: String, data_dir: P) {
        let loader = FileDatasetLoader::new(data_dir);
        self.loaders.insert(name, Box::new(loader));
    }

    pub fn set_default_loader(&mut self, name: String) {
        if self.loaders.contains_key(&name) {
            self.default_loader = name;
        }
    }

    pub fn load_dataset(
        &self,
        dataset_name: &str,
        split: &str,
        loader_name: Option<&str>,
    ) -> Result<EvaluationDataset> {
        let loader_name = loader_name.unwrap_or(&self.default_loader);
        let loader = self
            .loaders
            .get(loader_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown loader: {}", loader_name))?;

        loader.load(dataset_name, split)
    }

    pub fn list_datasets(&self, loader_name: Option<&str>) -> Vec<String> {
        let loader_name = loader_name.unwrap_or(&self.default_loader);
        self.loaders
            .get(loader_name)
            .map(|loader| loader.available_datasets())
            .unwrap_or_default()
    }

    pub fn list_splits(&self, dataset_name: &str, loader_name: Option<&str>) -> Vec<String> {
        let loader_name = loader_name.unwrap_or(&self.default_loader);
        self.loaders
            .get(loader_name)
            .map(|loader| loader.available_splits(dataset_name))
            .unwrap_or_default()
    }
}

impl Default for DatasetManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_dataset_sample() {
        let mut metadata = HashMap::new();
        metadata.insert("idx".to_string(), serde_json::Value::Number(0.into()));

        let sample = DatasetSample {
            input: "Test input".to_string(),
            target: "Test target".to_string(),
            metadata,
        };

        assert_eq!(sample.input, "Test input");
        assert_eq!(sample.target, "Test target");
        assert_eq!(sample.metadata.len(), 1);
    }

    #[test]
    fn test_evaluation_dataset() {
        let mut dataset = EvaluationDataset::new("test_dataset".to_string());
        assert_eq!(dataset.name, "test_dataset");
        assert_eq!(dataset.len(), 0);
        assert!(dataset.is_empty());

        let sample = DatasetSample {
            input: "Input 1".to_string(),
            target: "Target 1".to_string(),
            metadata: HashMap::new(),
        };
        dataset.add_sample(sample);

        assert_eq!(dataset.len(), 1);
        assert!(!dataset.is_empty());

        let inputs = dataset.get_inputs();
        let targets = dataset.get_targets();
        assert_eq!(inputs, vec!["Input 1"]);
        assert_eq!(targets, vec!["Target 1"]);
    }

    #[test]
    fn test_dataset_sampling() {
        let mut dataset = EvaluationDataset::new("test".to_string());

        for i in 0..10 {
            dataset.add_sample(DatasetSample {
                input: format!("Input {}", i),
                target: format!("Target {}", i),
                metadata: HashMap::new(),
            });
        }

        let sampled = dataset.sample(5);
        assert_eq!(sampled.len(), 5);
        assert_eq!(sampled.name, "test");
    }

    #[test]
    fn test_dataset_split() {
        let mut dataset = EvaluationDataset::new("test".to_string());

        for i in 0..10 {
            dataset.add_sample(DatasetSample {
                input: format!("Input {}", i),
                target: format!("Target {}", i),
                metadata: HashMap::new(),
            });
        }

        let (train, test) = dataset.split(0.7);
        assert_eq!(train.len(), 7);
        assert_eq!(test.len(), 3);
        assert_eq!(train.name, "test_train");
        assert_eq!(test.name, "test_test");
    }

    #[test]
    fn test_memory_dataset_loader() {
        let mut loader = MemoryDatasetLoader::new();

        let mut dataset = EvaluationDataset::new("test_train".to_string());
        dataset.add_sample(DatasetSample {
            input: "Test input".to_string(),
            target: "Test target".to_string(),
            metadata: HashMap::new(),
        });

        loader.add_dataset(dataset, "test", "train");

        let available_datasets = loader.available_datasets();
        assert_eq!(available_datasets, vec!["test"]);

        let available_splits = loader.available_splits("test");
        assert_eq!(available_splits, vec!["train"]);

        let loaded_dataset = loader.load("test", "train").unwrap();
        assert_eq!(loaded_dataset.len(), 1);
        assert_eq!(loaded_dataset.name, "test_train");
    }

    #[test]
    fn test_dummy_glue_datasets() {
        let mut loader = MemoryDatasetLoader::new();
        loader.create_dummy_glue_datasets();

        let datasets = loader.available_datasets();
        assert!(datasets.contains(&"cola".to_string()));
        assert!(datasets.contains(&"sst2".to_string()));
        assert!(datasets.contains(&"mrpc".to_string()));
        assert!(datasets.contains(&"mnli".to_string()));

        let cola_splits = loader.available_splits("cola");
        assert!(cola_splits.contains(&"train".to_string()));
        assert!(cola_splits.contains(&"validation".to_string()));

        let cola_train = loader.load("cola", "train").unwrap();
        assert_eq!(cola_train.len(), 1000);
    }

    #[test]
    fn test_dataset_manager() {
        let mut manager = DatasetManager::new();

        // Test with memory loader
        let datasets = manager.list_datasets(None);
        assert_eq!(datasets.len(), 0); // Empty by default

        // Add a file loader
        manager.register_file_loader("file".to_string(), "/tmp");

        // Test loading (will fail since /tmp doesn't have datasets, but tests the interface)
        let result = manager.load_dataset("nonexistent", "train", Some("file"));
        assert!(result.is_err());
    }
}
