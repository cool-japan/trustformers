pub mod catastrophic_prevention;
pub mod ewc;
pub mod memory_replay;
pub mod progressive_networks;
pub mod replay_buffer;
pub mod task_boundary;

pub use catastrophic_prevention::{CatastrophicPreventionStrategy, RegularizationMethod};
pub use ewc::{EWCConfig, EWCTrainer, FisherInformation};
pub use memory_replay::{ExperienceBuffer, MemoryReplay, MemoryReplayConfig};
pub use progressive_networks::{ProgressiveConfig, ProgressiveNetwork, TaskModule};
pub use replay_buffer::{
    compute_bwt, compute_fwt, compute_intransigence, LateralAdapter, PnnConfig, ReplayBuffer,
    ReplaySample, ReplayStrategy,
};
pub use task_boundary::{BoundaryDetectionConfig, TaskBoundaryDetector, TaskTransition};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for continual learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualLearningConfig {
    /// Method for preventing catastrophic forgetting
    pub prevention_method: CatastrophicPreventionStrategy,
    /// Task boundary detection configuration
    pub boundary_detection: BoundaryDetectionConfig,
    /// Memory replay configuration
    pub memory_replay: Option<MemoryReplayConfig>,
    /// EWC configuration
    pub ewc: Option<EWCConfig>,
    /// Progressive networks configuration
    pub progressive: Option<ProgressiveConfig>,
    /// Maximum number of tasks to remember
    pub max_tasks: usize,
    /// Whether to use online or offline learning
    pub online_learning: bool,
}

impl Default for ContinualLearningConfig {
    fn default() -> Self {
        Self {
            prevention_method: CatastrophicPreventionStrategy::EWC,
            boundary_detection: BoundaryDetectionConfig::default(),
            memory_replay: None,
            ewc: Some(EWCConfig::default()),
            progressive: None,
            max_tasks: 10,
            online_learning: true,
        }
    }
}

/// Task information for continual learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    pub task_id: String,
    pub name: String,
    pub description: Option<String>,
    pub data_size: usize,
    pub num_classes: Option<usize>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Continual learning manager
pub struct ContinualLearningManager {
    config: ContinualLearningConfig,
    tasks: Vec<TaskInfo>,
    current_task: Option<String>,
    task_transitions: Vec<TaskTransition>,
    #[allow(dead_code)]
    prevention_strategies: HashMap<String, Box<dyn RegularizationMethod>>,
}

impl ContinualLearningManager {
    pub fn new(config: ContinualLearningConfig) -> Self {
        Self {
            config,
            tasks: Vec::new(),
            current_task: None,
            task_transitions: Vec::new(),
            prevention_strategies: HashMap::new(),
        }
    }

    pub fn add_task(&mut self, task: TaskInfo) -> anyhow::Result<()> {
        if self.tasks.len() >= self.config.max_tasks {
            return Err(anyhow::anyhow!("Maximum number of tasks reached"));
        }

        self.tasks.push(task);
        Ok(())
    }

    pub fn set_current_task(&mut self, task_id: String) -> anyhow::Result<()> {
        if !self.tasks.iter().any(|t| t.task_id == task_id) {
            return Err(anyhow::anyhow!("Task not found: {}", task_id));
        }

        if let Some(prev_task) = &self.current_task {
            let transition = TaskTransition {
                from_task: prev_task.clone(),
                to_task: task_id.clone(),
                timestamp: chrono::Utc::now(),
                boundary_score: 1.0, // This would be computed by boundary detector
            };
            self.task_transitions.push(transition);
        }

        self.current_task = Some(task_id);
        Ok(())
    }

    pub fn get_current_task(&self) -> Option<&TaskInfo> {
        self.current_task
            .as_ref()
            .and_then(|id| self.tasks.iter().find(|t| &t.task_id == id))
    }

    pub fn get_task_count(&self) -> usize {
        self.tasks.len()
    }

    pub fn get_task_transitions(&self) -> &[TaskTransition] {
        &self.task_transitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continual_learning_manager() {
        let config = ContinualLearningConfig::default();
        let mut manager = ContinualLearningManager::new(config);

        let task1 = TaskInfo {
            task_id: "task1".to_string(),
            name: "Classification Task 1".to_string(),
            description: Some("First classification task".to_string()),
            data_size: 1000,
            num_classes: Some(10),
            created_at: chrono::Utc::now(),
        };

        manager.add_task(task1).expect("add operation failed");
        assert_eq!(manager.get_task_count(), 1);

        manager.set_current_task("task1".to_string()).expect("operation failed in test");
        assert!(manager.get_current_task().is_some());
    }

    #[test]
    fn test_max_tasks_limit() {
        let config = ContinualLearningConfig {
            max_tasks: 2,
            ..ContinualLearningConfig::default()
        };
        let mut manager = ContinualLearningManager::new(config);

        for i in 0..3 {
            let task = TaskInfo {
                task_id: format!("task{}", i),
                name: format!("Task {}", i),
                description: None,
                data_size: 100,
                num_classes: Some(5),
                created_at: chrono::Utc::now(),
            };

            if i < 2 {
                assert!(manager.add_task(task).is_ok());
            } else {
                assert!(manager.add_task(task).is_err());
            }
        }
    }

    #[test]
    fn test_get_current_task_none_initially() {
        let manager = ContinualLearningManager::new(ContinualLearningConfig::default());
        assert!(manager.get_current_task().is_none());
    }

    #[test]
    fn test_set_current_task_not_found_errors() {
        let mut manager = ContinualLearningManager::new(ContinualLearningConfig::default());
        let result = manager.set_current_task("nonexistent".to_string());
        assert!(
            result.is_err(),
            "setting a non-existent task should return Err"
        );
    }

    #[test]
    fn test_task_transitions_recorded() {
        let config = ContinualLearningConfig::default();
        let mut manager = ContinualLearningManager::new(config);

        for i in 0..3 {
            manager
                .add_task(TaskInfo {
                    task_id: format!("t{}", i),
                    name: format!("Task {}", i),
                    description: None,
                    data_size: 50,
                    num_classes: Some(3),
                    created_at: chrono::Utc::now(),
                })
                .expect("add_task should succeed");
        }

        manager.set_current_task("t0".to_string()).expect("set t0 should succeed");
        manager.set_current_task("t1".to_string()).expect("set t1 should succeed");
        manager.set_current_task("t2".to_string()).expect("set t2 should succeed");

        let transitions = manager.get_task_transitions();
        assert_eq!(transitions.len(), 2, "2 transitions: t0→t1 and t1→t2");
        assert_eq!(transitions[0].from_task, "t0");
        assert_eq!(transitions[0].to_task, "t1");
        assert_eq!(transitions[1].from_task, "t1");
        assert_eq!(transitions[1].to_task, "t2");
    }

    #[test]
    fn test_get_task_count_empty() {
        let manager = ContinualLearningManager::new(ContinualLearningConfig::default());
        assert_eq!(manager.get_task_count(), 0);
    }

    #[test]
    fn test_config_default_fields() {
        let cfg = ContinualLearningConfig::default();
        assert_eq!(cfg.max_tasks, 10);
        assert!(cfg.online_learning);
        assert!(cfg.ewc.is_some());
        assert!(cfg.memory_replay.is_none());
        assert!(cfg.progressive.is_none());
    }

    #[test]
    fn test_config_online_learning_false() {
        let cfg = ContinualLearningConfig {
            online_learning: false,
            ..ContinualLearningConfig::default()
        };
        assert!(!cfg.online_learning);
    }

    #[test]
    fn test_task_info_fields() {
        let now = chrono::Utc::now();
        let task = TaskInfo {
            task_id: "t42".to_string(),
            name: "Sentiment Analysis".to_string(),
            description: Some("Classify sentiment".to_string()),
            data_size: 50_000,
            num_classes: Some(3),
            created_at: now,
        };
        assert_eq!(task.task_id, "t42");
        assert_eq!(task.data_size, 50_000);
        assert_eq!(task.num_classes, Some(3));
        assert_eq!(
            task.description.as_deref().expect("description must be set"),
            "Classify sentiment"
        );
    }

    #[test]
    fn test_get_task_transitions_empty() {
        let manager = ContinualLearningManager::new(ContinualLearningConfig::default());
        assert!(manager.get_task_transitions().is_empty());
    }

    #[test]
    fn test_add_multiple_tasks_and_get_count() {
        let mut manager = ContinualLearningManager::new(ContinualLearningConfig::default());
        for i in 0..5 {
            manager
                .add_task(TaskInfo {
                    task_id: format!("task_{}", i),
                    name: format!("T{}", i),
                    description: None,
                    data_size: 100 * (i + 1),
                    num_classes: None,
                    created_at: chrono::Utc::now(),
                })
                .expect("add should succeed for 5 tasks under limit 10");
        }
        assert_eq!(manager.get_task_count(), 5);
    }

    #[test]
    fn test_current_task_id_updated() {
        let mut manager = ContinualLearningManager::new(ContinualLearningConfig::default());
        manager
            .add_task(TaskInfo {
                task_id: "alpha".to_string(),
                name: "Alpha".to_string(),
                description: None,
                data_size: 10,
                num_classes: Some(2),
                created_at: chrono::Utc::now(),
            })
            .expect("add alpha should succeed");

        manager.set_current_task("alpha".to_string()).expect("set alpha should succeed");
        let current = manager.get_current_task().expect("current task should be Some");
        assert_eq!(current.task_id, "alpha");
        assert_eq!(current.name, "Alpha");
    }

    #[test]
    fn test_task_info_no_num_classes() {
        let task = TaskInfo {
            task_id: "gen".to_string(),
            name: "Generation".to_string(),
            description: None,
            data_size: 1000,
            num_classes: None,
            created_at: chrono::Utc::now(),
        };
        assert!(task.num_classes.is_none());
    }
}
