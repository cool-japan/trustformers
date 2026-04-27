pub mod cross_task;
pub mod in_context;
pub mod meta_learning;
pub mod prompt_tuning;
pub mod prototypical;
pub mod task_adaptation;

pub use cross_task::{CrossTaskGeneralizer, GeneralizationConfig, TaskEmbedding};
pub use in_context::{ICLExample, InContextConfig, InContextLearner};
pub use meta_learning::{
    MAMLConfig, MAMLTrainer, MetaLearningAlgorithm, ReptileConfig, ReptileTrainer, TaskBatch,
};
pub use prompt_tuning::{PromptConfig, PromptTuner, SoftPrompt};
pub use prototypical::{
    FewShotError, IclEpisodeBuilder, IclTemplate, MatchingNetworks, PrototypicalConfig,
    PrototypicalNetworks,
};
pub use task_adaptation::{AdaptationConfig, TaskAdapter, TaskDescriptor};

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Configuration for few-shot and zero-shot learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotConfig {
    /// Number of examples per class (K in K-shot)
    pub k_shot: usize,
    /// Method for few-shot learning
    pub method: FewShotMethod,
    /// In-context learning configuration
    pub in_context: Option<InContextConfig>,
    /// Prompt tuning configuration
    pub prompt_tuning: Option<PromptConfig>,
    /// Meta-learning configuration
    pub meta_learning: Option<MetaLearningConfig>,
    /// Task adaptation configuration
    pub task_adaptation: Option<AdaptationConfig>,
    /// Whether to use cross-task generalization
    pub enable_cross_task: bool,
}

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            k_shot: 5,
            method: FewShotMethod::InContext,
            in_context: Some(InContextConfig::default()),
            prompt_tuning: None,
            meta_learning: None,
            task_adaptation: None,
            enable_cross_task: false,
        }
    }
}

/// Methods for few-shot learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FewShotMethod {
    /// In-context learning (like GPT-3)
    InContext,
    /// Prompt tuning with soft prompts
    PromptTuning,
    /// Meta-learning (MAML, Reptile)
    MetaLearning,
    /// Task-specific adaptation
    TaskAdaptation,
    /// Combined approach
    Combined(Vec<FewShotMethod>),
}

/// Meta-learning configuration wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaLearningConfig {
    MAML(MAMLConfig),
    Reptile(ReptileConfig),
}

/// Few-shot learning example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotExample {
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    pub task_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Support set for few-shot learning
#[derive(Debug, Clone)]
pub struct SupportSet {
    pub examples: Vec<FewShotExample>,
    pub task_id: String,
    pub k_shot: usize,
    pub num_classes: Option<usize>,
}

impl SupportSet {
    pub fn new(task_id: String, k_shot: usize) -> Self {
        Self {
            examples: Vec::new(),
            task_id,
            k_shot,
            num_classes: None,
        }
    }

    pub fn add_example(&mut self, example: FewShotExample) -> Result<()> {
        if self.examples.len() >= self.k_shot * self.num_classes.unwrap_or(usize::MAX) {
            return Err(anyhow::anyhow!("Support set is full"));
        }
        self.examples.push(example);
        Ok(())
    }

    pub fn is_complete(&self) -> bool {
        if let Some(num_classes) = self.num_classes {
            self.examples.len() == self.k_shot * num_classes
        } else {
            false
        }
    }
}

/// Query set for evaluation
#[derive(Debug, Clone)]
pub struct QuerySet {
    pub examples: Vec<FewShotExample>,
    pub task_id: String,
}

/// Few-shot learning manager
pub struct FewShotLearningManager {
    config: FewShotConfig,
    support_sets: std::collections::HashMap<String, SupportSet>,
    query_sets: std::collections::HashMap<String, QuerySet>,
}

impl FewShotLearningManager {
    pub fn new(config: FewShotConfig) -> Self {
        Self {
            config,
            support_sets: std::collections::HashMap::new(),
            query_sets: std::collections::HashMap::new(),
        }
    }

    pub fn create_support_set(&mut self, task_id: String, num_classes: usize) -> Result<()> {
        let mut support_set = SupportSet::new(task_id.clone(), self.config.k_shot);
        support_set.num_classes = Some(num_classes);
        self.support_sets.insert(task_id, support_set);
        Ok(())
    }

    pub fn add_support_example(&mut self, task_id: &str, example: FewShotExample) -> Result<()> {
        let support_set = self
            .support_sets
            .get_mut(task_id)
            .ok_or_else(|| anyhow::anyhow!("Support set not found for task: {}", task_id))?;
        support_set.add_example(example)?;
        Ok(())
    }

    pub fn create_query_set(&mut self, task_id: String) -> Result<()> {
        let query_set = QuerySet {
            examples: Vec::new(),
            task_id: task_id.clone(),
        };
        self.query_sets.insert(task_id, query_set);
        Ok(())
    }

    pub fn add_query_example(&mut self, task_id: &str, example: FewShotExample) -> Result<()> {
        let query_set = self
            .query_sets
            .get_mut(task_id)
            .ok_or_else(|| anyhow::anyhow!("Query set not found for task: {}", task_id))?;
        query_set.examples.push(example);
        Ok(())
    }

    pub fn get_support_set(&self, task_id: &str) -> Option<&SupportSet> {
        self.support_sets.get(task_id)
    }

    pub fn get_query_set(&self, task_id: &str) -> Option<&QuerySet> {
        self.query_sets.get(task_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_support_set() {
        let mut support_set = SupportSet::new("task1".to_string(), 5);
        support_set.num_classes = Some(2);

        for i in 0..10 {
            let example = FewShotExample {
                input: vec![i as f32],
                output: vec![(i % 2) as f32],
                task_id: Some("task1".to_string()),
                metadata: None,
            };
            support_set.add_example(example).expect("add operation failed");
        }

        assert!(support_set.is_complete());
        assert_eq!(support_set.examples.len(), 10);
    }

    #[test]
    fn test_few_shot_manager() {
        let config = FewShotConfig::default();
        let mut manager = FewShotLearningManager::new(config);

        manager
            .create_support_set("task1".to_string(), 2)
            .expect("operation failed in test");
        manager.create_query_set("task1".to_string()).expect("operation failed in test");

        let example = FewShotExample {
            input: vec![1.0, 2.0],
            output: vec![0.0],
            task_id: Some("task1".to_string()),
            metadata: None,
        };

        manager
            .add_support_example("task1", example.clone())
            .expect("add operation failed");
        manager.add_query_example("task1", example).expect("add operation failed");

        assert!(manager.get_support_set("task1").is_some());
        assert!(manager.get_query_set("task1").is_some());
    }

    #[test]
    fn test_few_shot_config_default() {
        let cfg = FewShotConfig::default();
        assert_eq!(cfg.k_shot, 5);
        assert!(cfg.in_context.is_some());
        assert!(cfg.prompt_tuning.is_none());
        assert!(cfg.meta_learning.is_none());
        assert!(!cfg.enable_cross_task);
    }

    #[test]
    fn test_support_set_is_not_complete_initially() {
        let ss = SupportSet::new("t".to_string(), 3);
        // num_classes is None → is_complete returns false
        assert!(!ss.is_complete());
    }

    #[test]
    fn test_support_set_incomplete_before_full() {
        let mut ss = SupportSet::new("t".to_string(), 3);
        ss.num_classes = Some(2); // need 6 examples
        for i in 0..5 {
            ss.add_example(FewShotExample {
                input: vec![i as f32],
                output: vec![0.0],
                task_id: None,
                metadata: None,
            })
            .expect("add should succeed for first 5 examples");
        }
        assert!(
            !ss.is_complete(),
            "should not be complete with only 5 of 6 examples"
        );
    }

    #[test]
    fn test_support_set_add_over_capacity_errors() {
        let mut ss = SupportSet::new("t".to_string(), 2);
        ss.num_classes = Some(1); // capacity = k_shot * num_classes = 2

        for i in 0..2 {
            ss.add_example(FewShotExample {
                input: vec![i as f32],
                output: vec![0.0],
                task_id: None,
                metadata: None,
            })
            .expect("first 2 adds should succeed");
        }

        // Third add should fail (exceeds k_shot * num_classes = 2)
        let result = ss.add_example(FewShotExample {
            input: vec![99.0],
            output: vec![0.0],
            task_id: None,
            metadata: None,
        });
        assert!(result.is_err(), "adding beyond capacity should return Err");
    }

    #[test]
    fn test_few_shot_manager_get_missing_support_set() {
        let manager = FewShotLearningManager::new(FewShotConfig::default());
        assert!(
            manager.get_support_set("nonexistent").is_none(),
            "missing support set should return None"
        );
    }

    #[test]
    fn test_few_shot_manager_get_missing_query_set() {
        let manager = FewShotLearningManager::new(FewShotConfig::default());
        assert!(
            manager.get_query_set("nonexistent").is_none(),
            "missing query set should return None"
        );
    }

    #[test]
    fn test_few_shot_manager_add_support_missing_task_errors() {
        let mut manager = FewShotLearningManager::new(FewShotConfig::default());
        let result = manager.add_support_example(
            "ghost",
            FewShotExample {
                input: vec![1.0],
                output: vec![0.0],
                task_id: None,
                metadata: None,
            },
        );
        assert!(
            result.is_err(),
            "adding to missing support set should return Err"
        );
    }

    #[test]
    fn test_few_shot_manager_add_query_missing_task_errors() {
        let mut manager = FewShotLearningManager::new(FewShotConfig::default());
        let result = manager.add_query_example(
            "phantom",
            FewShotExample {
                input: vec![2.0],
                output: vec![1.0],
                task_id: None,
                metadata: None,
            },
        );
        assert!(
            result.is_err(),
            "adding to missing query set should return Err"
        );
    }

    #[test]
    fn test_few_shot_manager_multiple_tasks() {
        let config = FewShotConfig {
            k_shot: 3,
            ..FewShotConfig::default()
        };
        let mut manager = FewShotLearningManager::new(config);

        manager
            .create_support_set("task_a".to_string(), 2)
            .expect("create task_a support");
        manager
            .create_support_set("task_b".to_string(), 4)
            .expect("create task_b support");

        assert!(manager.get_support_set("task_a").is_some());
        assert!(manager.get_support_set("task_b").is_some());
        assert!(manager.get_support_set("task_c").is_none());
    }

    #[test]
    fn test_few_shot_example_metadata() {
        let ex = FewShotExample {
            input: vec![0.1, 0.2],
            output: vec![1.0],
            task_id: Some("task_meta".to_string()),
            metadata: Some(serde_json::json!({"source": "wikipedia"})),
        };
        assert!(ex.metadata.is_some());
        assert_eq!(
            ex.task_id.as_deref().expect("task_id must be set"),
            "task_meta"
        );
    }

    #[test]
    fn test_query_set_add_multiple_examples() {
        let config = FewShotConfig::default();
        let mut manager = FewShotLearningManager::new(config);
        manager.create_query_set("q1".to_string()).expect("create query set");

        for i in 0..5 {
            manager
                .add_query_example(
                    "q1",
                    FewShotExample {
                        input: vec![i as f32],
                        output: vec![(i % 2) as f32],
                        task_id: None,
                        metadata: None,
                    },
                )
                .expect("add query example should succeed");
        }

        let qs = manager.get_query_set("q1").expect("query set must exist");
        assert_eq!(qs.examples.len(), 5);
    }

    #[test]
    fn test_few_shot_method_variant_in_context() {
        let cfg = FewShotConfig {
            method: FewShotMethod::InContext,
            ..FewShotConfig::default()
        };
        assert!(matches!(cfg.method, FewShotMethod::InContext));
    }

    #[test]
    fn test_few_shot_method_variant_prompt_tuning() {
        let cfg = FewShotConfig {
            method: FewShotMethod::PromptTuning,
            ..FewShotConfig::default()
        };
        assert!(matches!(cfg.method, FewShotMethod::PromptTuning));
    }

    #[test]
    fn test_support_set_k_shot_and_task_id() {
        let ss = SupportSet::new("classification".to_string(), 10);
        assert_eq!(ss.k_shot, 10);
        assert_eq!(ss.task_id, "classification");
        assert!(ss.examples.is_empty());
    }
}
