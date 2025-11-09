// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! # Cleanup Strategy Selection and Processing
//!
//! This module handles the selection and execution of cleanup strategies
//! based on memory pressure levels and system state. It provides intelligent
//! strategy selection, execution ordering, and queue management.
//!
//! ## Key Components
//!
//! - **Strategy Selection**: Chooses appropriate cleanup strategies based on pressure level
//! - **Execution Planning**: Orders cleanup actions by priority and effectiveness
//! - **Queue Management**: Manages cleanup action queue with proper prioritization
//! - **Progress Tracking**: Monitors cleanup effectiveness and adjusts strategies
//!
//! ## Strategy Selection Logic
//!
//! The system uses a multi-criteria approach for strategy selection:
//!
//! 1. **Pressure Level**: Higher pressure triggers more aggressive strategies
//! 2. **Effectiveness**: Prioritizes strategies that free the most memory
//! 3. **Speed**: Considers execution time for time-critical situations
//! 4. **Risk**: Balances memory freed against potential service impact
//! 5. **System State**: Adapts to current memory usage patterns

use super::{CleanupContext, CleanupHandler};
use crate::memory_pressure::config::*;
use anyhow::Result;
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    sync::Arc,
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

// =============================================================================
// Strategy Selection Engine
// =============================================================================

/// Cleanup strategy selection and execution engine
///
/// This component is responsible for intelligently selecting and executing
/// cleanup strategies based on current system state and memory pressure levels.
/// It maintains a registry of available cleanup handlers and orchestrates
/// their execution for optimal memory recovery.
#[derive(Debug)]
pub struct CleanupStrategyEngine {
    /// Registry of available cleanup handlers
    handlers: Arc<RwLock<HashMap<CleanupStrategy, Arc<dyn CleanupHandler>>>>,

    /// Cleanup action queue with priority ordering
    action_queue: Arc<Mutex<BinaryHeap<PrioritizedCleanupAction>>>,

    /// Strategy effectiveness tracking
    strategy_stats: Arc<RwLock<HashMap<CleanupStrategy, StrategyStats>>>,

    /// Engine configuration
    config: CleanupEngineConfig,
}

impl CleanupStrategyEngine {
    /// Create a new cleanup strategy engine
    pub fn new(config: CleanupEngineConfig) -> Self {
        Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
            action_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            strategy_stats: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Register a cleanup handler for a specific strategy
    pub async fn register_handler(
        &self,
        strategy: CleanupStrategy,
        handler: Arc<dyn CleanupHandler>,
    ) {
        let mut handlers = self.handlers.write().await;
        handlers.insert(strategy.clone(), handler);

        // Initialize statistics for this strategy
        let mut stats = self.strategy_stats.write().await;
        stats.insert(strategy, StrategyStats::new());
    }

    /// Select appropriate cleanup strategies for the given pressure level
    pub async fn select_strategies(
        &self,
        pressure_level: MemoryPressureLevel,
        target_memory_freed: Option<u64>,
    ) -> Vec<CleanupStrategy> {
        let handlers = self.handlers.read().await;
        let stats = self.strategy_stats.read().await;

        let mut strategies = Vec::new();

        // Base strategies based on pressure level
        let base_strategies = self.get_base_strategies_for_pressure(pressure_level);

        for strategy in base_strategies {
            if let Some(handler) = handlers.get(&strategy) {
                // Check if handler should execute at this pressure level
                if handler.should_execute(pressure_level) {
                    // Consider strategy effectiveness from historical data
                    if let Some(strategy_stats) = stats.get(&strategy) {
                        if strategy_stats.should_include_in_selection() {
                            strategies.push(strategy);
                        }
                    } else {
                        // Include new strategies without historical data
                        strategies.push(strategy);
                    }
                }
            }
        }

        // Sort strategies by effectiveness and priority
        self.sort_strategies_by_effectiveness(&mut strategies, &handlers, &stats).await;

        // Limit strategies based on target memory if specified
        if let Some(target) = target_memory_freed {
            strategies = self.limit_strategies_by_target(strategies, target, &handlers).await;
        }

        strategies
    }

    /// Queue cleanup actions for execution
    pub async fn queue_cleanup_actions(
        &self,
        strategies: Vec<CleanupStrategy>,
        context: CleanupContext,
    ) -> Result<usize> {
        let handlers = self.handlers.read().await;
        let mut queue = self.action_queue.lock().await;

        let mut actions_queued = 0;

        for strategy in strategies {
            if let Some(handler) = handlers.get(&strategy) {
                let estimated_memory = handler.estimate_memory_freed();
                let priority = self
                    .calculate_action_priority(&strategy, handler.get_priority(), &context)
                    .await;

                let action = PrioritizedCleanupAction {
                    action: CleanupAction {
                        strategy: strategy.clone(),
                        priority: handler.get_priority(),
                        estimated_memory_freed: estimated_memory,
                        gpu_device_id: None, // Set by caller if needed
                        queued_at: chrono::Utc::now(),
                    },
                    calculated_priority: priority,
                    urgency_score: context.get_urgency_score(),
                };

                queue.push(action);
                actions_queued += 1;
            }
        }

        debug!("Queued {} cleanup actions", actions_queued);
        Ok(actions_queued)
    }

    /// Execute the next cleanup action from the queue
    pub async fn execute_next_action(&self) -> Result<Option<CleanupResult>> {
        let action = {
            let mut queue = self.action_queue.lock().await;
            queue.pop()
        };

        if let Some(prioritized_action) = action {
            self.execute_cleanup_action(prioritized_action.action).await
        } else {
            Ok(None)
        }
    }

    /// Execute all queued cleanup actions
    pub async fn execute_all_actions(&self) -> Result<Vec<CleanupResult>> {
        let mut results = Vec::new();

        while let Some(result) = self.execute_next_action().await? {
            results.push(result);

            // Check if we should stop early due to memory target reached
            if self.should_stop_cleanup(&results).await {
                break;
            }
        }

        Ok(results)
    }

    /// Get cleanup queue status
    pub async fn get_queue_status(&self) -> CleanupQueueStatus {
        let queue = self.action_queue.lock().await;

        CleanupQueueStatus {
            pending_actions: queue.len(),
            estimated_memory_freed: queue
                .iter()
                .map(|action| action.action.estimated_memory_freed)
                .sum(),
            highest_priority: queue.peek().map(|action| action.calculated_priority),
        }
    }

    // =============================================================================
    // Private Implementation Methods
    // =============================================================================

    /// Get base cleanup strategies for a given pressure level
    fn get_base_strategies_for_pressure(
        &self,
        pressure_level: MemoryPressureLevel,
    ) -> Vec<CleanupStrategy> {
        match pressure_level {
            MemoryPressureLevel::Normal => vec![],
            MemoryPressureLevel::Low => vec![CleanupStrategy::CacheEviction],
            MemoryPressureLevel::Medium => vec![
                CleanupStrategy::CacheEviction,
                CleanupStrategy::GarbageCollection,
            ],
            MemoryPressureLevel::High => vec![
                CleanupStrategy::CacheEviction,
                CleanupStrategy::BufferCompaction,
                CleanupStrategy::GarbageCollection,
                CleanupStrategy::ModelUnloading,
            ],
            MemoryPressureLevel::Critical => vec![
                CleanupStrategy::BufferCompaction,
                CleanupStrategy::CacheEviction,
                CleanupStrategy::ModelUnloading,
                CleanupStrategy::GarbageCollection,
                CleanupStrategy::MemoryDefragmentation,
            ],
            MemoryPressureLevel::Emergency => vec![
                CleanupStrategy::BufferCompaction,
                CleanupStrategy::ModelUnloading,
                CleanupStrategy::CacheEviction,
                CleanupStrategy::MemoryDefragmentation,
                CleanupStrategy::GarbageCollection,
                CleanupStrategy::RequestRejection,
            ],
        }
    }

    /// Sort strategies by effectiveness and priority
    async fn sort_strategies_by_effectiveness(
        &self,
        strategies: &mut Vec<CleanupStrategy>,
        handlers: &HashMap<CleanupStrategy, Arc<dyn CleanupHandler>>,
        stats: &HashMap<CleanupStrategy, StrategyStats>,
    ) {
        strategies.sort_by(|a, b| {
            let score_a = self.calculate_strategy_score(a, handlers, stats);
            let score_b = self.calculate_strategy_score(b, handlers, stats);
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });
    }

    /// Calculate effectiveness score for a strategy
    fn calculate_strategy_score(
        &self,
        strategy: &CleanupStrategy,
        handlers: &HashMap<CleanupStrategy, Arc<dyn CleanupHandler>>,
        stats: &HashMap<CleanupStrategy, StrategyStats>,
    ) -> f64 {
        let mut score = 0.0;

        // Base score from handler priority
        if let Some(handler) = handlers.get(strategy) {
            score += handler.get_priority() as f64;
        }

        // Effectiveness multiplier from historical data
        if let Some(strategy_stats) = stats.get(strategy) {
            score *= strategy_stats.get_effectiveness_multiplier();
        }

        // Memory freed estimate
        if let Some(handler) = handlers.get(strategy) {
            let memory_mb = handler.estimate_memory_freed() as f64 / (1024.0 * 1024.0);
            score += memory_mb * 10.0; // 10 points per MB
        }

        score
    }

    /// Limit strategies to meet target memory freed
    async fn limit_strategies_by_target(
        &self,
        strategies: Vec<CleanupStrategy>,
        target_memory: u64,
        handlers: &HashMap<CleanupStrategy, Arc<dyn CleanupHandler>>,
    ) -> Vec<CleanupStrategy> {
        let mut accumulated_memory = 0u64;
        let mut limited_strategies = Vec::new();

        for strategy in strategies {
            if let Some(handler) = handlers.get(&strategy) {
                let estimated_memory = handler.estimate_memory_freed();
                accumulated_memory += estimated_memory;
                limited_strategies.push(strategy);

                // Stop if we've reached the target
                if accumulated_memory >= target_memory {
                    break;
                }
            }
        }

        limited_strategies
    }

    /// Calculate priority for a cleanup action
    async fn calculate_action_priority(
        &self,
        strategy: &CleanupStrategy,
        base_priority: u32,
        context: &CleanupContext,
    ) -> u32 {
        let mut priority = base_priority;

        // Urgency multiplier
        let urgency_multiplier = 1.0 + context.get_urgency_score() as f64;
        priority = (priority as f64 * urgency_multiplier) as u32;

        // Emergency boost
        if context.is_emergency {
            priority += 50;
        }

        // Strategy-specific adjustments
        match strategy {
            CleanupStrategy::RequestRejection => {
                // Lower priority unless emergency
                if !context.is_emergency {
                    priority = priority.saturating_sub(30);
                }
            },
            CleanupStrategy::ModelUnloading => {
                // Higher priority for very high memory usage
                if context.utilization > 0.9 {
                    priority += 20;
                }
            },
            _ => {},
        }

        priority.min(255) // Cap at maximum priority
    }

    /// Execute a cleanup action
    async fn execute_cleanup_action(&self, action: CleanupAction) -> Result<Option<CleanupResult>> {
        let handlers = self.handlers.read().await;

        if let Some(handler) = handlers.get(&action.strategy) {
            let start_time = std::time::Instant::now();

            match handler.cleanup(MemoryPressureLevel::Normal) {
                // Will be updated with actual pressure
                Ok(memory_freed) => {
                    let duration = start_time.elapsed();

                    // Update strategy statistics
                    self.update_strategy_stats(
                        &action.strategy,
                        memory_freed,
                        action.estimated_memory_freed,
                        duration,
                        true,
                    )
                    .await;

                    let result = CleanupResult {
                        strategy: action.strategy,
                        memory_freed,
                        estimated_memory_freed: action.estimated_memory_freed,
                        execution_time: duration,
                        success: true,
                        error_message: None,
                    };

                    info!(
                        "Cleanup action '{}' freed {} bytes in {:?}",
                        handler.name(),
                        memory_freed,
                        duration
                    );

                    Ok(Some(result))
                },
                Err(e) => {
                    let duration = start_time.elapsed();

                    // Update strategy statistics for failure
                    self.update_strategy_stats(
                        &action.strategy,
                        0,
                        action.estimated_memory_freed,
                        duration,
                        false,
                    )
                    .await;

                    warn!("Cleanup action '{}' failed: {}", handler.name(), e);

                    let result = CleanupResult {
                        strategy: action.strategy,
                        memory_freed: 0,
                        estimated_memory_freed: action.estimated_memory_freed,
                        execution_time: duration,
                        success: false,
                        error_message: Some(e.to_string()),
                    };

                    Ok(Some(result))
                },
            }
        } else {
            warn!(
                "No handler found for cleanup strategy: {:?}",
                action.strategy
            );
            Ok(None)
        }
    }

    /// Update strategy statistics
    async fn update_strategy_stats(
        &self,
        strategy: &CleanupStrategy,
        memory_freed: u64,
        estimated_memory: u64,
        duration: std::time::Duration,
        success: bool,
    ) {
        let mut stats = self.strategy_stats.write().await;

        if let Some(strategy_stats) = stats.get_mut(strategy) {
            strategy_stats.record_execution(memory_freed, estimated_memory, duration, success);
        }
    }

    /// Check if cleanup should stop early
    async fn should_stop_cleanup(&self, results: &[CleanupResult]) -> bool {
        if results.is_empty() {
            return false;
        }

        // Stop if we've freed enough memory for our target
        if let Some(target) = self.config.target_memory_freed {
            let total_freed: u64 = results.iter().map(|r| r.memory_freed).sum();
            if total_freed >= target {
                return true;
            }
        }

        // Stop if recent actions are not effective
        if results.len() >= 3 {
            let recent_results = &results[results.len() - 3..];
            let avg_freed: u64 = recent_results.iter().map(|r| r.memory_freed).sum::<u64>() / 3;

            if avg_freed < self.config.min_effective_memory_freed {
                debug!("Stopping cleanup due to low effectiveness");
                return true;
            }
        }

        false
    }
}

// =============================================================================
// Supporting Data Structures
// =============================================================================

/// Configuration for the cleanup strategy engine
#[derive(Debug, Clone)]
pub struct CleanupEngineConfig {
    /// Target amount of memory to free (optional)
    pub target_memory_freed: Option<u64>,

    /// Minimum effective memory freed per action
    pub min_effective_memory_freed: u64,

    /// Maximum actions to execute in one cleanup cycle
    pub max_actions_per_cycle: usize,

    /// Timeout for individual cleanup actions
    pub action_timeout_ms: u64,
}

impl Default for CleanupEngineConfig {
    fn default() -> Self {
        Self {
            target_memory_freed: None,
            min_effective_memory_freed: 1024 * 1024, // 1MB minimum
            max_actions_per_cycle: 10,
            action_timeout_ms: 5000, // 5 seconds
        }
    }
}

/// Prioritized cleanup action for queue ordering
#[derive(Debug, Clone)]
struct PrioritizedCleanupAction {
    action: CleanupAction,
    calculated_priority: u32,
    urgency_score: f32,
}

impl PartialEq for PrioritizedCleanupAction {
    fn eq(&self, other: &Self) -> bool {
        self.calculated_priority == other.calculated_priority
    }
}

impl Eq for PrioritizedCleanupAction {}

impl PartialOrd for PrioritizedCleanupAction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedCleanupAction {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority actions come first
        self.calculated_priority.cmp(&other.calculated_priority)
    }
}

/// Statistics for cleanup strategy effectiveness
#[derive(Debug, Clone)]
struct StrategyStats {
    total_executions: u64,
    successful_executions: u64,
    total_memory_freed: u64,
    total_estimated_memory: u64,
    avg_execution_time: std::time::Duration,
    last_execution: Option<chrono::DateTime<chrono::Utc>>,
}

impl StrategyStats {
    fn new() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            total_memory_freed: 0,
            total_estimated_memory: 0,
            avg_execution_time: std::time::Duration::from_millis(0),
            last_execution: None,
        }
    }

    fn record_execution(
        &mut self,
        memory_freed: u64,
        estimated_memory: u64,
        duration: std::time::Duration,
        success: bool,
    ) {
        self.total_executions += 1;
        if success {
            self.successful_executions += 1;
        }

        self.total_memory_freed += memory_freed;
        self.total_estimated_memory += estimated_memory;

        // Update average execution time
        let total_time = self.avg_execution_time.as_millis() as u64 * (self.total_executions - 1)
            + duration.as_millis() as u64;
        self.avg_execution_time =
            std::time::Duration::from_millis(total_time / self.total_executions);

        self.last_execution = Some(chrono::Utc::now());
    }

    fn get_success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            1.0
        } else {
            self.successful_executions as f64 / self.total_executions as f64
        }
    }

    fn get_effectiveness_ratio(&self) -> f64 {
        if self.total_estimated_memory == 0 {
            1.0
        } else {
            self.total_memory_freed as f64 / self.total_estimated_memory as f64
        }
    }

    fn get_effectiveness_multiplier(&self) -> f64 {
        let success_rate = self.get_success_rate();
        let effectiveness_ratio = self.get_effectiveness_ratio();

        // Combine success rate and effectiveness ratio
        (success_rate * 0.4 + effectiveness_ratio * 0.6).clamp(0.1, 2.0)
    }

    fn should_include_in_selection(&self) -> bool {
        // Include if success rate is reasonable and it's not completely ineffective
        self.get_success_rate() > 0.3 && self.get_effectiveness_ratio() > 0.1
    }
}

/// Result of a cleanup action execution
#[derive(Debug, Clone)]
pub struct CleanupResult {
    pub strategy: CleanupStrategy,
    pub memory_freed: u64,
    pub estimated_memory_freed: u64,
    pub execution_time: std::time::Duration,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Status of the cleanup action queue
#[derive(Debug, Clone)]
pub struct CleanupQueueStatus {
    pub pending_actions: usize,
    pub estimated_memory_freed: u64,
    pub highest_priority: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_pressure::cleanup::handlers::GarbageCollectionHandler;

    #[tokio::test]
    async fn test_strategy_engine_creation() {
        let config = CleanupEngineConfig::default();
        let engine = CleanupStrategyEngine::new(config);

        let status = engine.get_queue_status().await;
        assert_eq!(status.pending_actions, 0);
    }

    #[tokio::test]
    async fn test_handler_registration() {
        let config = CleanupEngineConfig::default();
        let engine = CleanupStrategyEngine::new(config);

        let handler = Arc::new(GarbageCollectionHandler::new());
        engine.register_handler(CleanupStrategy::GarbageCollection, handler).await;

        let strategies = engine.select_strategies(MemoryPressureLevel::Medium, None).await;
        assert!(strategies.contains(&CleanupStrategy::GarbageCollection));
    }

    #[tokio::test]
    async fn test_strategy_selection_by_pressure() {
        let config = CleanupEngineConfig::default();
        let engine = CleanupStrategyEngine::new(config);

        // Register a handler
        let handler = Arc::new(GarbageCollectionHandler::new());
        engine.register_handler(CleanupStrategy::GarbageCollection, handler).await;

        // Low pressure should have fewer strategies
        let low_strategies = engine.select_strategies(MemoryPressureLevel::Low, None).await;
        let high_strategies = engine.select_strategies(MemoryPressureLevel::High, None).await;

        assert!(high_strategies.len() >= low_strategies.len());
    }

    #[tokio::test]
    async fn test_cleanup_action_queue() {
        let config = CleanupEngineConfig::default();
        let engine = CleanupStrategyEngine::new(config);

        // Register a handler first
        let handler = Arc::new(GarbageCollectionHandler::new());
        engine.register_handler(CleanupStrategy::GarbageCollection, handler).await;

        let context = CleanupContext::new(MemoryPressureLevel::Medium, 0.7, 1024 * 1024 * 1024);

        let strategies = vec![CleanupStrategy::GarbageCollection];
        let actions_queued = engine.queue_cleanup_actions(strategies, context).await.unwrap();

        assert_eq!(actions_queued, 1);

        let status = engine.get_queue_status().await;
        assert_eq!(status.pending_actions, 1);
    }

    #[test]
    fn test_strategy_stats() {
        let mut stats = StrategyStats::new();

        // Record some executions
        stats.record_execution(
            100 * 1024 * 1024,
            90 * 1024 * 1024,
            std::time::Duration::from_millis(50),
            true,
        );

        stats.record_execution(
            80 * 1024 * 1024,
            90 * 1024 * 1024,
            std::time::Duration::from_millis(60),
            true,
        );

        assert_eq!(stats.get_success_rate(), 1.0);
        assert!(stats.get_effectiveness_ratio() > 0.9);
        assert!(stats.should_include_in_selection());
    }
}
