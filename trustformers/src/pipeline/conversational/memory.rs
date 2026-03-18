//! Conversation memory systems and context management.

use super::types::*;
use crate::error::{Result, TrustformersError};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::RwLock;

// ================================================================================================
// UTILITY TYPES
// ================================================================================================

/// Utility functions for memory management
pub type MemoryUtils = MemoryManager;

/// Manager for conversational memories
pub type ConversationMemoryManager = MemoryManager;

/// Memory management component
#[derive(Debug)]
pub struct MemoryManager {
    pub config: MemoryConfig,
    /// Optional persistent storage path
    pub storage_path: Option<String>,
    /// In-memory cache for frequently accessed memories
    pub memory_cache: Arc<RwLock<HashMap<String, ConversationMemory>>>,
}

impl MemoryManager {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            storage_path: None,
            memory_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new MemoryManager with persistent storage
    pub fn with_storage<P: AsRef<Path>>(config: MemoryConfig, storage_path: P) -> Self {
        Self {
            config,
            storage_path: Some(storage_path.as_ref().to_string_lossy().to_string()),
            memory_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create memory from conversation turn
    pub fn create_memory(&self, turn: &ConversationTurn) -> Option<ConversationMemory> {
        if !self.config.enabled {
            return None;
        }

        let importance = self.calculate_importance(turn);
        if importance < 0.3 {
            return None; // Skip low-importance content
        }

        Some(ConversationMemory {
            id: uuid::Uuid::new_v4().to_string(),
            content: turn.content.clone(),
            importance,
            last_accessed: chrono::Utc::now(),
            access_count: 0,
            memory_type: self.classify_memory_type(turn),
            tags: self.extract_tags(turn),
        })
    }

    fn calculate_importance(&self, turn: &ConversationTurn) -> f32 {
        let mut importance = 0.5; // Base importance

        // Increase importance for questions
        if turn.content.contains('?') {
            importance += 0.2;
        }

        // Increase for personal information
        if ["i am", "my name", "i like", "i prefer"]
            .iter()
            .any(|&pattern| turn.content.to_lowercase().contains(pattern))
        {
            importance += 0.3;
        }

        // Increase for goals and preferences
        if ["want", "need", "goal", "prefer", "like"]
            .iter()
            .any(|&pattern| turn.content.to_lowercase().contains(pattern))
        {
            importance += 0.2;
        }

        // Adjust based on metadata
        if let Some(metadata) = &turn.metadata {
            importance += metadata.confidence * 0.1;
            if metadata.engagement_level == EngagementLevel::High {
                importance += 0.2;
            }
        }

        importance.min(1.0)
    }

    fn classify_memory_type(&self, turn: &ConversationTurn) -> MemoryType {
        let content = turn.content.to_lowercase();

        if ["prefer", "like", "dislike", "favorite"]
            .iter()
            .any(|&pattern| content.contains(pattern))
        {
            MemoryType::Preference
        } else if ["goal", "want", "plan", "will"]
            .iter()
            .any(|&pattern| content.contains(pattern))
        {
            MemoryType::Goal
        } else if ["friend", "family", "colleague", "know"]
            .iter()
            .any(|&pattern| content.contains(pattern))
        {
            MemoryType::Relationship
        } else if ["happened", "did", "went", "experience"]
            .iter()
            .any(|&pattern| content.contains(pattern))
        {
            MemoryType::Experience
        } else {
            MemoryType::Fact
        }
    }

    fn extract_tags(&self, turn: &ConversationTurn) -> Vec<String> {
        let mut tags = Vec::new();

        if let Some(metadata) = &turn.metadata {
            tags.extend(metadata.topics.clone());
            if let Some(sentiment) = &metadata.sentiment {
                tags.push(format!("sentiment:{}", sentiment));
            }
        }

        // Extract simple keyword tags
        for keyword in ["work", "family", "hobby", "food", "travel", "technology"] {
            if turn.content.to_lowercase().contains(keyword) {
                tags.push(keyword.to_string());
            }
        }

        tags
    }

    /// Decay memory importance over time
    pub fn decay_memories(&self, memories: &mut [ConversationMemory]) {
        if !self.config.enabled {
            return;
        }

        for memory in memories {
            let time_factor =
                (chrono::Utc::now() - memory.last_accessed).num_hours() as f32 / (24.0 * 7.0);
            memory.importance *= self.config.decay_rate.powf(time_factor);
        }
    }

    /// Compress memories based on similarity and importance
    pub fn compress_memories(&self, memories: &mut Vec<ConversationMemory>) {
        if !self.config.enabled || memories.len() <= self.config.max_memories {
            return;
        }

        // Sort by importance (descending)
        memories.sort_by(|a, b| {
            b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Group similar memories and merge them
        let mut compressed = Vec::new();
        let mut skip_indices = std::collections::HashSet::new();

        for i in 0..memories.len() {
            if skip_indices.contains(&i) {
                continue;
            }

            let current = &memories[i];
            let mut similar_memories = vec![current.clone()];

            // Find similar memories
            for j in (i + 1)..memories.len() {
                if skip_indices.contains(&j) {
                    continue;
                }

                if self.are_memories_similar(&memories[i], &memories[j]) {
                    similar_memories.push(memories[j].clone());
                    skip_indices.insert(j);
                }
            }

            // Merge similar memories if we found any
            if similar_memories.len() > 1 {
                compressed.push(self.merge_memories(similar_memories));
            } else {
                compressed.push(current.clone());
            }
        }

        // Keep only the most important memories
        compressed.sort_by(|a, b| {
            b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal)
        });
        compressed.truncate(self.config.max_memories);

        *memories = compressed;
    }

    /// Check if two memories are similar enough to merge
    fn are_memories_similar(
        &self,
        memory1: &ConversationMemory,
        memory2: &ConversationMemory,
    ) -> bool {
        // Same memory type
        if memory1.memory_type != memory2.memory_type {
            return false;
        }

        // Similar tags
        let common_tags = memory1.tags.iter().filter(|tag| memory2.tags.contains(tag)).count();

        let total_tags = memory1.tags.len() + memory2.tags.len();
        let tag_similarity =
            if total_tags > 0 { (common_tags * 2) as f32 / total_tags as f32 } else { 0.0 };

        // Content similarity (simple word overlap)
        let content1_lower = memory1.content.to_lowercase();
        let words1: std::collections::HashSet<&str> = content1_lower.split_whitespace().collect();
        let content2_lower = memory2.content.to_lowercase();
        let words2: std::collections::HashSet<&str> = content2_lower.split_whitespace().collect();

        let common_words = words1.intersection(&words2).count();
        let total_words = words1.len() + words2.len();
        let content_similarity = if total_words > 0 {
            (common_words * 2) as f32 / total_words as f32
        } else {
            0.0
        };

        // Consider similar if either tag or content similarity is high
        tag_similarity > 0.6 || content_similarity > 0.5
    }

    /// Merge similar memories into a single consolidated memory
    fn merge_memories(&self, memories: Vec<ConversationMemory>) -> ConversationMemory {
        if memories.is_empty() {
            panic!("Cannot merge empty memory list");
        }

        if memories.len() == 1 {
            return memories[0].clone();
        }

        // Find the most important memory as the base
        let base_memory = memories
            .iter()
            .max_by(|a, b| {
                a.importance.partial_cmp(&b.importance).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("memories list should not be empty");

        // Combine content
        let mut combined_content = base_memory.content.clone();
        for memory in &memories {
            if memory.id != base_memory.id && !combined_content.contains(&memory.content) {
                combined_content.push_str(" | ");
                combined_content.push_str(&memory.content);
            }
        }

        // Combine tags
        let mut all_tags: Vec<String> =
            memories.iter().flat_map(|m| m.tags.iter().cloned()).collect();
        all_tags.sort();
        all_tags.dedup();

        // Calculate combined importance
        let max_importance = memories.iter().map(|m| m.importance).fold(0.0f32, f32::max);
        let avg_importance =
            memories.iter().map(|m| m.importance).sum::<f32>() / memories.len() as f32;
        let combined_importance = (max_importance + avg_importance) / 2.0;

        // Use the most recent access time
        let last_accessed = memories
            .iter()
            .map(|m| m.last_accessed)
            .max()
            .expect("memories should have at least one element");

        // Sum access counts
        let total_access_count = memories.iter().map(|m| m.access_count).sum();

        ConversationMemory {
            id: uuid::Uuid::new_v4().to_string(),
            content: combined_content,
            importance: combined_importance.min(1.0),
            last_accessed,
            access_count: total_access_count,
            memory_type: base_memory.memory_type.clone(),
            tags: all_tags,
        }
    }

    /// Retrieve memories by type
    pub fn get_memories_by_type<'a>(
        &self,
        memories: &'a [ConversationMemory],
        memory_type: MemoryType,
    ) -> Vec<&'a ConversationMemory> {
        memories.iter().filter(|memory| memory.memory_type == memory_type).collect()
    }

    /// Search memories by tag
    pub fn search_memories_by_tag<'a>(
        &self,
        memories: &'a [ConversationMemory],
        tag: &str,
    ) -> Vec<&'a ConversationMemory> {
        memories
            .iter()
            .filter(|memory| memory.tags.iter().any(|t| t.contains(tag)))
            .collect()
    }

    /// Update memory access statistics
    pub fn access_memory(&self, memory: &mut ConversationMemory) {
        memory.last_accessed = chrono::Utc::now();
        memory.access_count += 1;

        // Boost importance slightly for accessed memories
        memory.importance = (memory.importance * 1.05).min(1.0);
    }

    /// Get memory statistics
    pub fn get_memory_stats(&self, memories: &[ConversationMemory]) -> MemoryStats {
        if memories.is_empty() {
            return MemoryStats::default();
        }

        let total_memories = memories.len();
        let avg_importance =
            memories.iter().map(|m| m.importance).sum::<f32>() / total_memories as f32;
        let total_access_count = memories.iter().map(|m| m.access_count).sum();

        let mut type_distribution = std::collections::HashMap::new();
        for memory in memories {
            *type_distribution.entry(memory.memory_type.clone()).or_insert(0) += 1;
        }

        let most_important = memories
            .iter()
            .max_by(|a, b| {
                a.importance.partial_cmp(&b.importance).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();

        let most_accessed = memories.iter().max_by_key(|m| m.access_count).cloned();

        MemoryStats {
            total_memories,
            avg_importance,
            total_access_count,
            type_distribution,
            most_important,
            most_accessed,
        }
    }

    /// Save memories to persistent storage
    pub async fn save_memories(
        &self,
        conversation_id: &str,
        memories: &[ConversationMemory],
    ) -> Result<()> {
        if !self.config.persist_important_memories || self.storage_path.is_none() {
            return Ok(());
        }

        let storage_path = self.storage_path.as_ref().expect("storage_path checked as Some above");
        let file_path = format!("{}/memories_{}.json", storage_path, conversation_id);

        // Filter important memories to persist
        let important_memories: Vec<&ConversationMemory> = memories
            .iter()
            .filter(|m| m.importance >= self.config.compression_threshold)
            .collect();

        let serialized = serde_json::to_string_pretty(&important_memories).map_err(|e| {
            TrustformersError::invalid_input(
                format!("Serialization failed: {}", e),
                Some("memory_data".to_string()),
                Some("valid serializable data".to_string()),
                Some("data with serialization issues".to_string()),
            )
        })?;

        // Ensure directory exists
        if let Some(parent) = Path::new(&file_path).parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                TrustformersError::invalid_input(
                    format!("Serialization failed: {}", e),
                    Some("memory_data".to_string()),
                    Some("valid serializable data".to_string()),
                    Some("data with serialization issues".to_string()),
                )
            })?;
        }

        fs::write(&file_path, serialized).await.map_err(|e| {
            TrustformersError::invalid_input(
                format!("Serialization failed: {}", e),
                Some("memory_data".to_string()),
                Some("valid serializable data".to_string()),
                Some("data with serialization issues".to_string()),
            )
        })?;

        Ok(())
    }

    /// Load memories from persistent storage
    pub async fn load_memories(&self, conversation_id: &str) -> Result<Vec<ConversationMemory>> {
        if !self.config.persist_important_memories || self.storage_path.is_none() {
            return Ok(Vec::new());
        }

        let storage_path = self.storage_path.as_ref().expect("storage_path checked as Some above");
        let file_path = format!("{}/memories_{}.json", storage_path, conversation_id);

        if !Path::new(&file_path).exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(&file_path).await.map_err(|e| {
            TrustformersError::invalid_input(
                format!("Serialization failed: {}", e),
                Some("memory_data".to_string()),
                Some("valid serializable data".to_string()),
                Some("data with serialization issues".to_string()),
            )
        })?;

        let memories: Vec<ConversationMemory> = serde_json::from_str(&content).map_err(|e| {
            TrustformersError::invalid_input(
                format!("Serialization failed: {}", e),
                Some("memory_data".to_string()),
                Some("valid serializable data".to_string()),
                Some("data with serialization issues".to_string()),
            )
        })?;

        // Update cache with loaded memories
        let mut cache = self.memory_cache.write().await;
        for memory in &memories {
            cache.insert(memory.id.clone(), memory.clone());
        }

        Ok(memories)
    }

    /// Delete persistent memories for a conversation
    pub async fn delete_memories(&self, conversation_id: &str) -> Result<()> {
        if self.storage_path.is_none() {
            return Ok(());
        }

        let storage_path = self.storage_path.as_ref().expect("storage_path checked as Some above");
        let file_path = format!("{}/memories_{}.json", storage_path, conversation_id);

        if Path::new(&file_path).exists() {
            fs::remove_file(&file_path).await.map_err(|e| {
                TrustformersError::invalid_input(
                    format!("Serialization failed: {}", e),
                    Some("memory_data".to_string()),
                    Some("valid serializable data".to_string()),
                    Some("data with serialization issues".to_string()),
                )
            })?;
        }

        // Remove from cache
        let mut cache = self.memory_cache.write().await;
        cache.retain(|_, memory| !memory.id.starts_with(conversation_id));

        Ok(())
    }

    /// Perform memory maintenance and cleanup
    pub async fn maintenance_cleanup(
        &self,
        memories: &mut Vec<ConversationMemory>,
    ) -> Result<MaintenanceReport> {
        let initial_count = memories.len();
        let mut report = MaintenanceReport::default();

        // Apply decay
        self.decay_memories(memories);
        report.decay_applied = true;

        // Remove expired memories
        let decay_threshold = 0.1;
        let before_cleanup = memories.len();
        memories.retain(|m| m.importance > decay_threshold);
        report.expired_removed = before_cleanup - memories.len();

        // Compress similar memories
        self.compress_memories(memories);
        report.compression_applied = true;

        // Update access patterns
        self.update_access_patterns(memories).await;

        report.final_count = memories.len();
        report.memories_processed = initial_count;

        Ok(report)
    }

    /// Update access patterns and boost frequently accessed memories
    async fn update_access_patterns(&self, memories: &mut [ConversationMemory]) {
        let cache = self.memory_cache.read().await;

        for memory in memories {
            if let Some(cached_memory) = cache.get(&memory.id) {
                // Update access count from cache
                memory.access_count = cached_memory.access_count;
                memory.last_accessed = cached_memory.last_accessed;

                // Boost importance for frequently accessed memories
                if cached_memory.access_count > 10 {
                    memory.importance = (memory.importance * 1.1).min(1.0);
                }
            }
        }
    }

    /// Export memories in various formats
    pub async fn export_memories(
        &self,
        memories: &[ConversationMemory],
        format: ExportFormat,
    ) -> Result<String> {
        match format {
            ExportFormat::Json => serde_json::to_string_pretty(memories).map_err(|e| {
                TrustformersError::InvalidInput {
                    message: format!("Failed to serialize memories to JSON: {}", e),
                    parameter: Some("memories".to_string()),
                    expected: Some("Valid serializable data".to_string()),
                    received: Some("Data with serialization issues".to_string()),
                    suggestion: Some("Check that all memory data is serializable".to_string()),
                }
            }),
            ExportFormat::Csv => {
                let mut csv_content = String::from(
                    "id,content,importance,memory_type,tags,access_count,last_accessed\n",
                );
                for memory in memories {
                    let tags_str = memory.tags.join(";");
                    csv_content.push_str(&format!(
                        "{},{},{},{:?},{},{},{}\n",
                        memory.id,
                        memory.content.replace(',', ";"),
                        memory.importance,
                        memory.memory_type,
                        tags_str,
                        memory.access_count,
                        memory.last_accessed.format("%Y-%m-%d %H:%M:%S")
                    ));
                }
                Ok(csv_content)
            },
            ExportFormat::Summary => {
                let stats = self.get_memory_stats(memories);
                Ok(format!(
                    "Memory Summary:\n\
                     Total memories: {}\n\
                     Average importance: {:.2}\n\
                     Total accesses: {}\n\
                     Type distribution: {:?}\n\
                     Most important: {}\n\
                     Most accessed: {}",
                    stats.total_memories,
                    stats.avg_importance,
                    stats.total_access_count,
                    stats.type_distribution,
                    stats.most_important.as_ref().map(|m| m.content.as_str()).unwrap_or("None"),
                    stats.most_accessed.as_ref().map(|m| m.content.as_str()).unwrap_or("None")
                ))
            },
        }
    }

    /// Import memories from JSON string
    pub async fn import_memories(&self, json_content: &str) -> Result<Vec<ConversationMemory>> {
        let memories: Vec<ConversationMemory> =
            serde_json::from_str(json_content).map_err(|e| {
                TrustformersError::invalid_input(
                    format!("Serialization failed: {}", e),
                    Some("memory_data".to_string()),
                    Some("valid serializable data".to_string()),
                    Some("data with serialization issues".to_string()),
                )
            })?;

        // Update cache with imported memories
        let mut cache = self.memory_cache.write().await;
        for memory in &memories {
            cache.insert(memory.id.clone(), memory.clone());
        }

        Ok(memories)
    }

    /// Analyze memory patterns and provide insights
    pub fn analyze_memory_patterns(&self, memories: &[ConversationMemory]) -> MemoryAnalysis {
        let mut analysis = MemoryAnalysis::default();

        if memories.is_empty() {
            return analysis;
        }

        // Analyze memory types
        let mut type_counts = HashMap::new();
        for memory in memories {
            *type_counts.entry(memory.memory_type.clone()).or_insert(0) += 1;
        }
        analysis.type_distribution = type_counts;

        // Find dominant memory type
        analysis.dominant_type = analysis
            .type_distribution
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(memory_type, _)| memory_type.clone());

        // Analyze importance distribution
        let total_importance: f32 = memories.iter().map(|m| m.importance).sum();
        analysis.avg_importance = total_importance / memories.len() as f32;

        // Find high-importance memories
        analysis.high_importance_count = memories.iter().filter(|m| m.importance > 0.8).count();

        // Analyze access patterns
        analysis.total_accesses = memories.iter().map(|m| m.access_count).sum();
        analysis.avg_accesses = analysis.total_accesses as f32 / memories.len() as f32;

        // Recent activity analysis
        let now = chrono::Utc::now();
        analysis.recent_activity_count =
            memories.iter().filter(|m| (now - m.last_accessed).num_hours() < 24).count();

        // Memory health score
        analysis.health_score = self.calculate_memory_health(memories);

        analysis
    }

    /// Calculate overall memory health score
    fn calculate_memory_health(&self, memories: &[ConversationMemory]) -> f32 {
        if memories.is_empty() {
            return 1.0;
        }

        let avg_importance: f32 =
            memories.iter().map(|m| m.importance).sum::<f32>() / memories.len() as f32;
        let recent_access_ratio = {
            let now = chrono::Utc::now();
            let recent_count = memories.iter()
                .filter(|m| (now - m.last_accessed).num_hours() < 168) // 1 week
                .count();
            recent_count as f32 / memories.len() as f32
        };

        let type_diversity = {
            let mut types = std::collections::HashSet::new();
            for memory in memories {
                types.insert(&memory.memory_type);
            }
            types.len() as f32 / 5.0 // Max 5 memory types
        };

        (avg_importance * 0.4 + recent_access_ratio * 0.3 + type_diversity * 0.3).min(1.0)
    }
}

/// Statistics about memory usage
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub avg_importance: f32,
    pub total_access_count: usize,
    pub type_distribution: std::collections::HashMap<MemoryType, usize>,
    pub most_important: Option<ConversationMemory>,
    pub most_accessed: Option<ConversationMemory>,
}

/// Report for memory maintenance operations
#[derive(Debug, Clone, Default)]
pub struct MaintenanceReport {
    pub memories_processed: usize,
    pub expired_removed: usize,
    pub decay_applied: bool,
    pub compression_applied: bool,
    pub final_count: usize,
}

/// Export format options for memories
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Summary,
}

/// Memory analysis results
#[derive(Debug, Clone, Default)]
pub struct MemoryAnalysis {
    pub type_distribution: HashMap<MemoryType, usize>,
    pub dominant_type: Option<MemoryType>,
    pub avg_importance: f32,
    pub high_importance_count: usize,
    pub total_accesses: usize,
    pub avg_accesses: f32,
    pub recent_activity_count: usize,
    pub health_score: f32,
}

/// Long-term memory management utilities
pub struct LongTermMemoryManager {
    memory_manager: MemoryManager,
    conversation_summaries: HashMap<String, String>,
}

impl LongTermMemoryManager {
    pub fn new(memory_manager: MemoryManager) -> Self {
        Self {
            memory_manager,
            conversation_summaries: HashMap::new(),
        }
    }

    /// Consolidate memories across multiple conversations
    pub async fn consolidate_cross_conversation_memories(
        &mut self,
        conversation_memories: HashMap<String, Vec<ConversationMemory>>,
    ) -> Result<Vec<ConversationMemory>> {
        let mut all_memories = Vec::new();

        // Collect all memories
        for (conversation_id, memories) in conversation_memories {
            // Store conversation summary
            if !memories.is_empty() {
                let summary = self.create_conversation_summary(&memories);
                self.conversation_summaries.insert(conversation_id, summary);
            }
            all_memories.extend(memories);
        }

        // Apply global compression and deduplication
        self.memory_manager.compress_memories(&mut all_memories);

        // Apply cross-conversation similarity detection
        self.merge_cross_conversation_similarities(&mut all_memories);

        Ok(all_memories)
    }

    /// Create a summary of conversation memories
    fn create_conversation_summary(&self, memories: &[ConversationMemory]) -> String {
        let mut summary_parts = Vec::new();

        // Collect key themes by memory type
        let mut type_groups: HashMap<MemoryType, Vec<&ConversationMemory>> = HashMap::new();
        for memory in memories {
            type_groups.entry(memory.memory_type.clone()).or_default().push(memory);
        }

        for (memory_type, type_memories) in type_groups {
            if !type_memories.is_empty() {
                let key_content: Vec<&str> = type_memories
                    .iter()
                    .filter(|m| m.importance > 0.7)
                    .take(3)
                    .map(|m| m.content.as_str())
                    .collect();

                if !key_content.is_empty() {
                    summary_parts.push(format!("{:?}: {}", memory_type, key_content.join("; ")));
                }
            }
        }

        summary_parts.join(" | ")
    }

    /// Merge memories that are similar across different conversations
    fn merge_cross_conversation_similarities(&self, memories: &mut Vec<ConversationMemory>) {
        // Group by content similarity
        let mut similarity_groups: Vec<Vec<usize>> = Vec::new();
        let mut processed = vec![false; memories.len()];

        for i in 0..memories.len() {
            if processed[i] {
                continue;
            }

            let mut group = vec![i];
            processed[i] = true;

            for j in (i + 1)..memories.len() {
                if processed[j] {
                    continue;
                }

                if self.memory_manager.are_memories_similar(&memories[i], &memories[j]) {
                    group.push(j);
                    processed[j] = true;
                }
            }

            if group.len() > 1 {
                similarity_groups.push(group);
            }
        }

        // Merge similar groups
        for group in similarity_groups.into_iter().rev() {
            if group.len() <= 1 {
                continue;
            }

            let group_memories: Vec<ConversationMemory> =
                group.iter().map(|&idx| memories[idx].clone()).collect();

            let merged = self.memory_manager.merge_memories(group_memories);

            // Remove original memories (in reverse order to maintain indices)
            for &idx in group.iter().rev() {
                memories.remove(idx);
            }

            // Add merged memory
            memories.push(merged);
        }
    }

    /// Get global memory insights across all conversations
    pub fn get_global_insights(&self, all_memories: &[ConversationMemory]) -> GlobalMemoryInsights {
        let mut insights = GlobalMemoryInsights::default();

        if all_memories.is_empty() {
            return insights;
        }

        // Analyze memory patterns
        let analysis = self.memory_manager.analyze_memory_patterns(all_memories);
        insights.memory_analysis = analysis;

        // Find recurring themes
        insights.recurring_themes = self.find_recurring_themes(all_memories);

        // Analyze user preferences
        insights.user_preferences = self.extract_user_preferences(all_memories);

        // Calculate memory efficiency
        insights.memory_efficiency = self.calculate_memory_efficiency(all_memories);

        insights
    }

    /// Find recurring themes across memories
    fn find_recurring_themes(&self, memories: &[ConversationMemory]) -> Vec<String> {
        let mut tag_frequency: HashMap<String, usize> = HashMap::new();

        for memory in memories {
            for tag in &memory.tags {
                *tag_frequency.entry(tag.clone()).or_insert(0) += 1;
            }
        }

        // Return tags that appear in multiple memories
        tag_frequency
            .into_iter()
            .filter(|(_, count)| *count >= 3)
            .map(|(tag, _)| tag)
            .collect()
    }

    /// Extract user preferences from memories
    fn extract_user_preferences(&self, memories: &[ConversationMemory]) -> Vec<String> {
        memories
            .iter()
            .filter(|m| m.memory_type == MemoryType::Preference && m.importance > 0.6)
            .map(|m| m.content.clone())
            .collect()
    }

    /// Calculate memory system efficiency
    fn calculate_memory_efficiency(&self, memories: &[ConversationMemory]) -> f32 {
        if memories.is_empty() {
            return 1.0;
        }

        let high_importance_count = memories.iter().filter(|m| m.importance > 0.7).count();
        let accessed_memories = memories.iter().filter(|m| m.access_count > 0).count();

        let importance_ratio = high_importance_count as f32 / memories.len() as f32;
        let access_ratio = accessed_memories as f32 / memories.len() as f32;

        (importance_ratio * 0.6 + access_ratio * 0.4).min(1.0)
    }
}

/// Global insights across all memories
#[derive(Debug, Clone, Default)]
pub struct GlobalMemoryInsights {
    pub memory_analysis: MemoryAnalysis,
    pub recurring_themes: Vec<String>,
    pub user_preferences: Vec<String>,
    pub memory_efficiency: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::env;
    use tokio::fs;

    fn create_test_memory_config() -> MemoryConfig {
        MemoryConfig {
            enabled: true,
            compression_threshold: 0.7,
            persist_important_memories: true,
            decay_rate: 0.95,
            max_memories: 10,
        }
    }

    fn create_test_memory(
        content: &str,
        importance: f32,
        memory_type: MemoryType,
    ) -> ConversationMemory {
        ConversationMemory {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.to_string(),
            importance,
            last_accessed: Utc::now(),
            access_count: 0,
            memory_type,
            tags: vec!["test".to_string()],
        }
    }

    fn create_test_turn(content: &str, role: ConversationRole) -> ConversationTurn {
        ConversationTurn {
            role,
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: Some(ConversationMetadata {
                sentiment: Some("neutral".to_string()),
                intent: Some("test".to_string()),
                confidence: 0.8,
                topics: vec!["test".to_string()],
                safety_flags: vec![],
                entities: vec![],
                quality_score: 0.9,
                engagement_level: EngagementLevel::Medium,
                reasoning_type: None,
            }),
            token_count: 10,
        }
    }

    #[test]
    fn test_memory_manager_creation() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config.clone());

        assert_eq!(manager.config.enabled, config.enabled);
        assert_eq!(manager.config.max_memories, config.max_memories);
        assert!(manager.storage_path.is_none());
    }

    #[test]
    fn test_memory_manager_with_storage() {
        let config = create_test_memory_config();
        let storage_path = "/tmp/test_memories";
        let manager = MemoryManager::with_storage(config, storage_path);

        assert!(manager.storage_path.is_some());
        assert_eq!(manager.storage_path.unwrap(), storage_path);
    }

    #[test]
    fn test_create_memory_from_turn() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let turn = create_test_turn("I like programming in Rust", ConversationRole::User);
        let memory = manager.create_memory(&turn);

        assert!(memory.is_some());
        let memory = memory.unwrap();
        assert_eq!(memory.content, "I like programming in Rust");
        assert!(memory.importance > 0.5); // Should be high due to preference
        assert_eq!(memory.memory_type, MemoryType::Preference);
    }

    #[test]
    fn test_memory_importance_calculation() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        // Test high importance for personal information
        let personal_turn = create_test_turn(
            "My name is Alice and I prefer Python",
            ConversationRole::User,
        );
        let memory = manager.create_memory(&personal_turn).unwrap();
        assert!(memory.importance > 0.7);

        // Test low importance for generic content
        let generic_turn = create_test_turn("Hello there", ConversationRole::User);
        let memory = manager.create_memory(&generic_turn);
        // Memory may exist but should have lower importance than personal info
        if let Some(mem) = memory {
            assert!(
                mem.importance < 0.8,
                "Generic content should have moderate or low importance"
            );
        }
    }

    #[test]
    fn test_memory_type_classification() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let test_cases = vec![
            ("I prefer coffee over tea", MemoryType::Preference),
            ("My goal is to learn Rust", MemoryType::Goal),
            ("I went to the park yesterday", MemoryType::Experience),
            ("My friend John is a developer", MemoryType::Relationship),
            ("The sky is blue", MemoryType::Fact),
        ];

        for (content, expected_type) in test_cases {
            let turn = create_test_turn(content, ConversationRole::User);
            if let Some(memory) = manager.create_memory(&turn) {
                assert_eq!(
                    memory.memory_type, expected_type,
                    "Failed for content: {}",
                    content
                );
            }
        }
    }

    #[test]
    fn test_memory_decay() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let mut memories = vec![
            create_test_memory("Test memory 1", 0.8, MemoryType::Fact),
            create_test_memory("Test memory 2", 0.9, MemoryType::Preference),
        ];

        let original_importance1 = memories[0].importance;
        let original_importance2 = memories[1].importance;

        manager.decay_memories(&mut memories);

        // Importance should decay slightly
        assert!(memories[0].importance <= original_importance1);
        assert!(memories[1].importance <= original_importance2);
    }

    #[test]
    fn test_memory_compression() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let mut memories = vec![];
        // Create more memories than the limit
        for i in 0..15 {
            memories.push(create_test_memory(
                &format!("Test memory {}", i),
                0.5 + (i as f32 * 0.03),
                MemoryType::Fact,
            ));
        }

        let original_count = memories.len();
        manager.compress_memories(&mut memories);

        // Should be compressed to max_memories limit
        assert!(memories.len() <= 10);
        assert!(memories.len() < original_count);

        // Remaining memories should be sorted by importance
        for i in 1..memories.len() {
            assert!(memories[i - 1].importance >= memories[i].importance);
        }
    }

    #[test]
    fn test_memory_similarity_detection() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let memory1 =
            create_test_memory("I like programming in Python", 0.8, MemoryType::Preference);
        let memory2 = create_test_memory(
            "I enjoy coding with Python language",
            0.7,
            MemoryType::Preference,
        );
        let memory3 = create_test_memory("I went to the store", 0.6, MemoryType::Experience);

        assert!(manager.are_memories_similar(&memory1, &memory2));
        assert!(!manager.are_memories_similar(&memory1, &memory3));
    }

    #[test]
    fn test_memory_merging() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let memories = vec![
            create_test_memory("I like Python", 0.8, MemoryType::Preference),
            create_test_memory(
                "Python is my favorite language",
                0.7,
                MemoryType::Preference,
            ),
        ];

        let merged = manager.merge_memories(memories);

        assert!(merged.content.contains("Python"));
        assert!(merged.importance >= 0.7);
        assert_eq!(merged.memory_type, MemoryType::Preference);
    }

    #[test]
    fn test_memory_retrieval_by_type() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let memories = vec![
            create_test_memory("I like coffee", 0.8, MemoryType::Preference),
            create_test_memory("I went shopping", 0.7, MemoryType::Experience),
            create_test_memory("I prefer tea", 0.6, MemoryType::Preference),
        ];

        let preferences = manager.get_memories_by_type(&memories, MemoryType::Preference);
        assert_eq!(preferences.len(), 2);

        let experiences = manager.get_memories_by_type(&memories, MemoryType::Experience);
        assert_eq!(experiences.len(), 1);
    }

    #[test]
    fn test_memory_search_by_tag() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let mut memory1 = create_test_memory("Programming content", 0.8, MemoryType::Fact);
        memory1.tags = vec!["programming".to_string(), "rust".to_string()];

        let mut memory2 = create_test_memory("Cooking content", 0.7, MemoryType::Experience);
        memory2.tags = vec!["cooking".to_string(), "food".to_string()];

        let memories = vec![memory1, memory2];

        let programming_memories = manager.search_memories_by_tag(&memories, "programming");
        assert_eq!(programming_memories.len(), 1);

        let cooking_memories = manager.search_memories_by_tag(&memories, "cooking");
        assert_eq!(cooking_memories.len(), 1);
    }

    #[test]
    fn test_memory_access_tracking() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let mut memory = create_test_memory("Test memory", 0.8, MemoryType::Fact);
        let original_access_count = memory.access_count;
        let original_importance = memory.importance;

        manager.access_memory(&mut memory);

        assert_eq!(memory.access_count, original_access_count + 1);
        assert!(memory.importance >= original_importance);
    }

    #[test]
    fn test_memory_statistics() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let memories = vec![
            create_test_memory("Memory 1", 0.9, MemoryType::Preference),
            create_test_memory("Memory 2", 0.7, MemoryType::Experience),
            create_test_memory("Memory 3", 0.8, MemoryType::Fact),
        ];

        let stats = manager.get_memory_stats(&memories);

        assert_eq!(stats.total_memories, 3);
        assert_eq!(stats.avg_importance, (0.9 + 0.7 + 0.8) / 3.0);
        assert_eq!(stats.type_distribution.len(), 3);
        assert!(stats.most_important.is_some());
    }

    #[test]
    fn test_memory_analysis() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let memories = vec![
            create_test_memory("High importance memory", 0.9, MemoryType::Preference),
            create_test_memory("Medium importance memory", 0.6, MemoryType::Experience),
            create_test_memory("Low importance memory", 0.3, MemoryType::Fact),
        ];

        let analysis = manager.analyze_memory_patterns(&memories);

        assert_eq!(analysis.high_importance_count, 1);
        assert!(analysis.avg_importance > 0.5);
        assert!(analysis.health_score > 0.0);
    }

    #[tokio::test]
    async fn test_memory_persistence() {
        let temp_dir = env::temp_dir().join("memory_test");
        let config = create_test_memory_config();
        let manager = MemoryManager::with_storage(config, &temp_dir);

        let memories = vec![
            create_test_memory("Important memory", 0.9, MemoryType::Preference),
            create_test_memory("Less important memory", 0.5, MemoryType::Fact),
        ];

        let conversation_id = "test_conversation";

        // Test saving memories
        let result = manager.save_memories(conversation_id, &memories).await;
        assert!(result.is_ok());

        // Test loading memories
        let loaded_memories = manager.load_memories(conversation_id).await.unwrap();
        assert_eq!(loaded_memories.len(), 1); // Only important memory should be saved
        assert_eq!(loaded_memories[0].content, "Important memory");

        // Test deleting memories
        let result = manager.delete_memories(conversation_id).await;
        assert!(result.is_ok());

        let loaded_after_delete = manager.load_memories(conversation_id).await.unwrap();
        assert!(loaded_after_delete.is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_maintenance_cleanup() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let mut memories = vec![
            create_test_memory("Good memory", 0.8, MemoryType::Preference),
            create_test_memory("Expired memory", 0.05, MemoryType::Fact), // Very low importance
            create_test_memory("Another good memory", 0.7, MemoryType::Experience),
        ];

        let report = manager.maintenance_cleanup(&mut memories).await.unwrap();

        assert!(report.decay_applied);
        assert!(report.compression_applied);
        assert_eq!(report.expired_removed, 1); // Low importance memory should be removed
        assert_eq!(memories.len(), 2);
    }

    #[tokio::test]
    async fn test_memory_export_import() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        let memories = vec![
            create_test_memory("Test memory 1", 0.8, MemoryType::Preference),
            create_test_memory("Test memory 2", 0.7, MemoryType::Experience),
        ];

        // Test JSON export
        let json_export = manager.export_memories(&memories, ExportFormat::Json).await.unwrap();
        assert!(json_export.contains("Test memory 1"));
        assert!(json_export.contains("Test memory 2"));

        // Test CSV export
        let csv_export = manager.export_memories(&memories, ExportFormat::Csv).await.unwrap();
        assert!(csv_export.contains("id,content,importance"));
        assert!(csv_export.contains("Test memory 1"));

        // Test summary export
        let summary_export =
            manager.export_memories(&memories, ExportFormat::Summary).await.unwrap();
        assert!(summary_export.contains("Memory Summary"));
        assert!(summary_export.contains("Total memories: 2"));

        // Test import
        let imported = manager.import_memories(&json_export).await.unwrap();
        assert_eq!(imported.len(), 2);
    }

    #[test]
    fn test_long_term_memory_manager() {
        let config = create_test_memory_config();
        let memory_manager = MemoryManager::new(config);
        let mut ltm_manager = LongTermMemoryManager::new(memory_manager);

        let mut conversation_memories = HashMap::new();
        conversation_memories.insert(
            "conv1".to_string(),
            vec![create_test_memory(
                "User likes Python",
                0.9,
                MemoryType::Preference,
            )],
        );
        conversation_memories.insert(
            "conv2".to_string(),
            vec![create_test_memory(
                "User went to store",
                0.6,
                MemoryType::Experience,
            )],
        );

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let consolidated = runtime
            .block_on(async {
                ltm_manager.consolidate_cross_conversation_memories(conversation_memories).await
            })
            .unwrap();

        assert!(!consolidated.is_empty());
        assert!(!ltm_manager.conversation_summaries.is_empty());
    }

    #[test]
    fn test_global_memory_insights() {
        let config = create_test_memory_config();
        let memory_manager = MemoryManager::new(config);
        let ltm_manager = LongTermMemoryManager::new(memory_manager);

        let mut memory1 = create_test_memory("I like programming", 0.9, MemoryType::Preference);
        memory1.tags = vec!["programming".to_string(), "coding".to_string()];

        let mut memory2 = create_test_memory("I enjoy coding in Rust", 0.8, MemoryType::Preference);
        memory2.tags = vec!["programming".to_string(), "rust".to_string()];

        let mut memory3 = create_test_memory("I went to a conference", 0.6, MemoryType::Experience);
        memory3.tags = vec!["conference".to_string(), "programming".to_string()];

        let memories = vec![memory1, memory2, memory3];
        let insights = ltm_manager.get_global_insights(&memories);

        assert!(!insights.recurring_themes.is_empty());
        assert!(insights.recurring_themes.contains(&"programming".to_string()));
        assert!(!insights.user_preferences.is_empty());
        assert!(insights.memory_efficiency > 0.0);
    }

    #[test]
    fn test_memory_health_calculation() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        // Test with healthy memories (high importance, recent access)
        let healthy_memories = vec![
            create_test_memory("Important memory 1", 0.9, MemoryType::Preference),
            create_test_memory("Important memory 2", 0.8, MemoryType::Goal),
        ];

        let health_score = manager.calculate_memory_health(&healthy_memories);
        assert!(health_score > 0.5);

        // Test with unhealthy memories (low importance)
        let unhealthy_memories = vec![
            create_test_memory("Unimportant memory 1", 0.2, MemoryType::Fact),
            create_test_memory("Unimportant memory 2", 0.1, MemoryType::Fact),
        ];

        let low_health_score = manager.calculate_memory_health(&unhealthy_memories);
        assert!(low_health_score < health_score);
    }

    #[test]
    fn test_edge_cases() {
        let config = create_test_memory_config();
        let manager = MemoryManager::new(config);

        // Test with empty memories
        let empty_memories: Vec<ConversationMemory> = vec![];
        let stats = manager.get_memory_stats(&empty_memories);
        assert_eq!(stats.total_memories, 0);

        let analysis = manager.analyze_memory_patterns(&empty_memories);
        assert_eq!(analysis.total_accesses, 0);

        // Test with disabled config
        let mut disabled_config = create_test_memory_config();
        disabled_config.enabled = false;
        let disabled_manager = MemoryManager::new(disabled_config);

        let turn = create_test_turn("Test content", ConversationRole::User);
        let memory = disabled_manager.create_memory(&turn);
        assert!(memory.is_none());
    }
}
