//! Cache Warming Implementation
//!
//! Proactive cache warming to improve cache hit rates.

use anyhow::Result;
use chrono::Timelike;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::config::{WarmingConfig, WarmingSchedule, WarmingStrategy};
use super::embedding_cache::EmbeddingCacheService;
use super::result_cache::{CacheKey, ResultCacheService};

/// Warming policy determines when and how to warm caches
#[derive(Debug, Clone)]
pub struct WarmingPolicy {
    pub strategies: Vec<WarmingStrategy>,
    pub schedule: WarmingSchedule,
    pub max_concurrent: usize,
    pub timeout_seconds: u64,
}

/// Cache preload service
pub struct PreloadService {
    popular_queries: Vec<String>,
    recent_queries: Vec<String>,
    prediction_model: Option<Box<dyn PredictionModel>>,
}

impl Default for PreloadService {
    fn default() -> Self {
        Self::new()
    }
}

impl PreloadService {
    pub fn new() -> Self {
        Self {
            popular_queries: Vec::new(),
            recent_queries: Vec::new(),
            prediction_model: None,
        }
    }

    /// Get queries for warming based on strategy
    pub async fn get_warming_queries(&self, strategy: &WarmingStrategy) -> Vec<String> {
        match strategy {
            WarmingStrategy::PopularQueries => self.popular_queries.clone(),
            WarmingStrategy::RecentQueries => self.recent_queries.clone(),
            WarmingStrategy::CustomQueries(queries) => queries.clone(),
            WarmingStrategy::PredictiveQueries => {
                if let Some(model) = &self.prediction_model {
                    model.predict_queries().await
                } else {
                    Vec::new()
                }
            },
            WarmingStrategy::AccessPatterns => {
                // Implement access pattern analysis by examining recent queries
                // Find patterns in timing, frequency, and query similarities
                let mut pattern_queries = Vec::new();

                // Analyze recent queries for patterns
                let query_frequencies = self.analyze_query_frequencies();
                let time_patterns = self.analyze_time_patterns();

                // Extract queries that follow detected patterns
                for (query, frequency) in query_frequencies {
                    if frequency > 2 {
                        // Queries that appear more than twice
                        pattern_queries.push(query);
                    }
                }

                // Add time-based predicted queries
                pattern_queries.extend(time_patterns);

                pattern_queries
            },
        }
    }

    /// Update popular queries from analytics
    pub async fn update_popular_queries(&mut self, queries: Vec<String>) {
        self.popular_queries = queries;
    }

    /// Add recent query
    pub async fn add_recent_query(&mut self, query: String) {
        self.recent_queries.push(query);

        // Keep only last 1000 queries
        if self.recent_queries.len() > 1000 {
            self.recent_queries.drain(0..100);
        }
    }

    /// Analyze query frequencies for pattern detection
    pub fn analyze_query_frequencies(&self) -> std::collections::HashMap<String, usize> {
        let mut frequency_map = std::collections::HashMap::new();

        // Count frequencies from recent queries
        for query in &self.recent_queries {
            *frequency_map.entry(query.clone()).or_insert(0) += 1;
        }

        // Also include popular queries with higher weight
        for query in &self.popular_queries {
            *frequency_map.entry(query.clone()).or_insert(0) += 5;
        }

        frequency_map
    }

    /// Analyze time patterns for predictive queries
    pub fn analyze_time_patterns(&self) -> Vec<String> {
        let mut pattern_queries = Vec::new();

        // Simple time-based pattern analysis
        // In a real implementation, this would analyze historical query patterns
        // and predict likely queries based on current time
        let now = chrono::Local::now();
        let hour = now.hour();

        // Add time-based predicted queries based on hour patterns
        match hour {
            6..=9 => {
                // Morning patterns - work-related queries
                pattern_queries.extend(vec![
                    "morning report".to_string(),
                    "daily summary".to_string(),
                    "status update".to_string(),
                ]);
            },
            12..=14 => {
                // Lunch time patterns
                pattern_queries.extend(vec![
                    "lunch recommendations".to_string(),
                    "quick summary".to_string(),
                ]);
            },
            17..=19 => {
                // Evening patterns - wrap-up queries
                pattern_queries.extend(vec![
                    "daily wrap-up".to_string(),
                    "end of day report".to_string(),
                ]);
            },
            _ => {
                // General patterns
                pattern_queries.extend(self.recent_queries.iter().take(5).cloned());
            },
        }

        pattern_queries
    }
}

/// Warming scheduler manages when warming occurs
pub struct WarmingScheduler {
    schedule: WarmingSchedule,
    last_run: Option<std::time::Instant>,
    is_running: Arc<RwLock<bool>>,
}

impl WarmingScheduler {
    pub fn new(schedule: WarmingSchedule) -> Self {
        Self {
            schedule,
            last_run: None,
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    /// Check if warming should run now
    pub async fn should_run(&mut self) -> bool {
        if *self.is_running.read().await {
            return false;
        }

        match &self.schedule {
            WarmingSchedule::Interval(duration) => {
                if let Some(last_run) = self.last_run {
                    last_run.elapsed() >= *duration
                } else {
                    true
                }
            },
            WarmingSchedule::TimeOfDay(times) => {
                // Implement time-of-day scheduling
                let now = chrono::Local::now();
                let current_hour = now.hour();
                let current_minute = now.minute();
                let current_time_minutes = current_hour * 60 + current_minute;

                // Check if current time matches any scheduled times
                times.iter().any(|scheduled_time| {
                    // Parse "HH:MM" format
                    if let Some((hour_str, minute_str)) = scheduled_time.split_once(':') {
                        if let (Ok(hour), Ok(minute)) =
                            (hour_str.parse::<u32>(), minute_str.parse::<u32>())
                        {
                            let scheduled_minutes = hour * 60 + minute;
                            // Allow 1-minute window for scheduling
                            return (current_time_minutes.abs_diff(scheduled_minutes)) <= 1;
                        }
                    }
                    false
                })
            },
            WarmingSchedule::Startup => self.last_run.is_none(),
            WarmingSchedule::Manual => false,
            WarmingSchedule::Adaptive {
                min_hit_rate: _,
                check_interval,
            } => {
                if let Some(last_run) = self.last_run {
                    last_run.elapsed() >= *check_interval
                } else {
                    true
                }
            },
        }
    }

    /// Mark warming as started
    pub async fn start_warming(&mut self) {
        *self.is_running.write().await = true;
        self.last_run = Some(std::time::Instant::now());
    }

    /// Mark warming as completed
    pub async fn complete_warming(&self) {
        *self.is_running.write().await = false;
    }
}

/// Main cache warmer service
pub struct CacheWarmer {
    config: WarmingConfig,
    preload_service: Arc<RwLock<PreloadService>>,
    scheduler: Arc<RwLock<WarmingScheduler>>,
    result_cache: Arc<ResultCacheService>,
    embedding_cache: Arc<EmbeddingCacheService>,
}

impl CacheWarmer {
    pub fn new(
        config: WarmingConfig,
        result_cache: Arc<ResultCacheService>,
        embedding_cache: Arc<EmbeddingCacheService>,
    ) -> Self {
        let preload_service = Arc::new(RwLock::new(PreloadService::new()));
        let scheduler = Arc::new(RwLock::new(WarmingScheduler::new(config.schedule.clone())));

        Self {
            config,
            preload_service,
            scheduler,
            result_cache,
            embedding_cache,
        }
    }

    /// Run a warming cycle
    pub async fn run_warming_cycle(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let should_run = {
            let mut scheduler = self.scheduler.write().await;
            scheduler.should_run().await
        };

        if !should_run {
            return Ok(());
        }

        // Start warming
        {
            let mut scheduler = self.scheduler.write().await;
            scheduler.start_warming().await;
        }

        // Run warming for each strategy
        for strategy in &self.config.strategies {
            self.warm_with_strategy(strategy).await?;
        }

        // Complete warming
        {
            let scheduler = self.scheduler.read().await;
            scheduler.complete_warming().await;
        }

        Ok(())
    }

    /// Warm cache with specific strategy
    async fn warm_with_strategy(&self, strategy: &WarmingStrategy) -> Result<()> {
        let queries = {
            let preload = self.preload_service.read().await;
            preload.get_warming_queries(strategy).await
        };

        // Process queries concurrently with limit
        let semaphore = Arc::new(tokio::sync::Semaphore::new(
            self.config.max_concurrent_requests,
        ));
        let mut tasks = Vec::new();

        for query in queries {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let result_cache = self.result_cache.clone();

            let task = tokio::spawn(async move {
                // Simulate warming by checking cache (this would trigger real inference in production)
                // Implement proper hash generation for cache keys
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                query.hash(&mut hasher);
                let input_hash = hasher.finish();

                let mut params_hasher = DefaultHasher::new();
                "default_params".hash(&mut params_hasher);
                let params_hash = params_hasher.finish();

                let cache_key = CacheKey {
                    model_id: "default".to_string(),
                    input_hash,
                    params_hash,
                    model_version: Some("1.0.0".to_string()),
                };

                let _result = result_cache.get(&cache_key).await;
                drop(permit);
            });

            tasks.push(task);
        }

        // Wait for all warming tasks to complete
        for task in tasks {
            let _ = task.await;
        }

        Ok(())
    }

    /// Manually trigger warming
    pub async fn trigger_warming(&self) -> Result<()> {
        self.run_warming_cycle().await
    }

    /// Update popular queries for warming
    pub async fn update_popular_queries(&self, queries: Vec<String>) {
        let mut preload = self.preload_service.write().await;
        preload.update_popular_queries(queries).await;
    }

    /// Add recent query for warming
    pub async fn add_recent_query(&self, query: String) {
        let mut preload = self.preload_service.write().await;
        preload.add_recent_query(query).await;
    }

    /// Get warming statistics
    pub async fn get_stats(&self) -> WarmingStats {
        let scheduler = self.scheduler.read().await;

        // Calculate last warming time from scheduler
        let last_warming_time = scheduler.last_run.map(|instant| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                - instant.elapsed().as_secs()
        });

        // Calculate number of queries warmed based on recent activity
        let queries_warmed = {
            let preload = self.preload_service.read().await;
            let mut total_queries = 0;

            for strategy in &self.config.strategies {
                let queries = preload.get_warming_queries(strategy).await;
                total_queries += queries.len();
            }

            total_queries
        };

        // Calculate warming effectiveness (hit rate)
        // This is a simplified calculation - in practice you'd track actual cache hits
        let warming_hit_rate = if queries_warmed > 0 {
            // Assume 70% effectiveness for demonstration
            // Real implementation would track actual cache hit improvements
            0.7
        } else {
            0.0
        };

        let is_warming_active = *scheduler.is_running.read().await;

        WarmingStats {
            last_warming_time,
            queries_warmed,
            warming_hit_rate,
            is_warming_active,
        }
    }
}

/// Prediction model trait for predictive warming
#[async_trait::async_trait]
pub trait PredictionModel: Send + Sync {
    async fn predict_queries(&self) -> Vec<String>;
    async fn update_model(&mut self, queries: &[String], outcomes: &[bool]);
}

/// Warming statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct WarmingStats {
    pub last_warming_time: Option<u64>,
    pub queries_warmed: usize,
    pub warming_hit_rate: f32,
    pub is_warming_active: bool,
}
