//! Database connection management for test resource allocation.
//!
//! This module provides database connection pool management for parallel test execution,
//! including connection allocation, monitoring, and usage statistics.

use anyhow::Result;
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::{debug, info, warn};

use super::types::{DatabasePoolConfig, DatabaseUsageStatistics};

/// Database connection management system
pub struct DatabaseConnectionManager {
    /// Configuration
    config: Arc<Mutex<DatabasePoolConfig>>,
    /// Available connections
    available_connections: Arc<Mutex<Vec<String>>>,
    /// Allocated connections
    allocated_connections: Arc<Mutex<HashMap<String, DatabaseConnection>>>,
    /// Usage statistics
    usage_stats: Arc<Mutex<DatabaseUsageStatistics>>,
}

/// Database connection information
#[derive(Debug, Clone)]
pub struct DatabaseConnection {
    /// Connection ID
    pub connection_id: String,
    /// Test ID that allocated the connection
    pub test_id: String,
    /// Connection URL
    pub connection_url: String,
    /// Allocated timestamp
    pub allocated_at: chrono::DateTime<chrono::Utc>,
    /// Connection type
    pub connection_type: DatabaseType,
}

/// Database connection types
#[derive(Debug, Clone)]
pub enum DatabaseType {
    /// PostgreSQL
    PostgreSQL,
    /// MySQL
    MySQL,
    /// SQLite
    SQLite,
    /// MongoDB
    MongoDB,
    /// Redis
    Redis,
    /// Custom database
    Custom(String),
}

impl DatabaseConnectionManager {
    /// Create new database connection manager
    pub async fn new(config: DatabasePoolConfig) -> Result<Self> {
        let mut available_connections = Vec::new();

        // Initialize connection pool
        for i in 0..config.max_connections {
            let connection_id = format!("conn_{}", i);
            available_connections.push(connection_id);
        }

        info!(
            "Initialized database connection manager with {} connections",
            available_connections.len()
        );

        Ok(Self {
            config: Arc::new(Mutex::new(config)),
            available_connections: Arc::new(Mutex::new(available_connections)),
            allocated_connections: Arc::new(Mutex::new(HashMap::new())),
            usage_stats: Arc::new(Mutex::new(DatabaseUsageStatistics::default())),
        })
    }

    /// Allocate database connections for a test
    pub async fn allocate_connections(&self, count: usize, test_id: &str) -> Result<Vec<String>> {
        if count == 0 {
            return Ok(vec![]);
        }

        let mut available_connections = self.available_connections.lock();
        let mut allocated_connections = self.allocated_connections.lock();
        let mut usage_stats = self.usage_stats.lock();

        if available_connections.len() < count {
            return Err(anyhow::anyhow!(
                "Insufficient available database connections: requested {}, available {}",
                count,
                available_connections.len()
            ));
        }

        let mut allocated = Vec::new();
        let now = chrono::Utc::now();

        // Allocate connections
        for _ in 0..count {
            if let Some(connection_id) = available_connections.pop() {
                let connection = DatabaseConnection {
                    connection_id: connection_id.clone(),
                    test_id: test_id.to_string(),
                    connection_url: format!("postgresql://localhost:5432/test_{}", test_id),
                    allocated_at: now,
                    connection_type: DatabaseType::PostgreSQL,
                };

                allocated_connections.insert(connection_id.clone(), connection);
                allocated.push(connection_id);
            } else {
                // Rollback partial allocation
                for conn_id in &allocated {
                    if let Some(connection) = allocated_connections.remove(conn_id) {
                        available_connections.push(connection.connection_id);
                    }
                }
                return Err(anyhow::anyhow!("Failed to allocate database connections"));
            }
        }

        // Update statistics
        usage_stats.total_connections += count as u64;
        usage_stats.active_connections = allocated_connections.len();
        usage_stats.peak_connections =
            usage_stats.peak_connections.max(allocated_connections.len());

        info!(
            "Allocated {} database connections for test {}: {:?}",
            allocated.len(),
            test_id,
            allocated
        );

        Ok(allocated)
    }

    /// Deallocate a specific connection
    pub async fn deallocate_connection(&self, conn_id: &str) -> Result<()> {
        let mut available_connections = self.available_connections.lock();
        let mut allocated_connections = self.allocated_connections.lock();
        let mut usage_stats = self.usage_stats.lock();

        if let Some(connection) = allocated_connections.remove(conn_id) {
            available_connections.push(connection.connection_id.clone());
            usage_stats.active_connections = allocated_connections.len();

            // Update average duration statistics
            let duration = connection.allocated_at.signed_duration_since(chrono::Utc::now()).abs();
            let duration_std = Duration::from_secs(duration.num_seconds().max(0) as u64);

            if usage_stats.total_connections > 0 {
                let total_duration = usage_stats.average_duration.as_secs() as f64
                    * (usage_stats.total_connections - 1) as f64;
                let new_average = (total_duration + duration_std.as_secs() as f64)
                    / usage_stats.total_connections as f64;
                usage_stats.average_duration = Duration::from_secs(new_average as u64);
            }

            info!(
                "Deallocated database connection {} for test {}",
                conn_id, connection.test_id
            );
            Ok(())
        } else {
            warn!(
                "Attempted to deallocate database connection {} that was not allocated",
                conn_id
            );
            Err(anyhow::anyhow!(
                "Database connection {} was not allocated",
                conn_id
            ))
        }
    }

    /// Deallocate all connections for a specific test
    pub async fn deallocate_connections_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating database connections for test: {}", test_id);

        let mut available_connections = self.available_connections.lock();
        let mut allocated_connections = self.allocated_connections.lock();
        let mut usage_stats = self.usage_stats.lock();

        let mut deallocated_connections = Vec::new();

        // Find and collect connections to deallocate
        allocated_connections.retain(|conn_id, connection| {
            if connection.test_id == test_id {
                available_connections.push(connection.connection_id.clone());
                deallocated_connections.push(conn_id.clone());
                false // Remove from allocated_connections
            } else {
                true // Keep in allocated_connections
            }
        });

        usage_stats.active_connections = allocated_connections.len();

        if !deallocated_connections.is_empty() {
            info!(
                "Released {} database connections for test {}: {:?}",
                deallocated_connections.len(),
                test_id,
                deallocated_connections
            );
        }

        Ok(())
    }

    /// Check if requested number of connections are available
    pub async fn check_availability(&self, count: usize) -> Result<bool> {
        let available_connections = self.available_connections.lock();
        Ok(available_connections.len() >= count)
    }

    /// Get current database usage statistics
    pub async fn get_statistics(&self) -> Result<DatabaseUsageStatistics> {
        let stats = self.usage_stats.lock();
        Ok(stats.clone())
    }

    /// Get available connection count
    pub async fn get_available_connection_count(&self) -> usize {
        let available_connections = self.available_connections.lock();
        available_connections.len()
    }

    /// Get active connection count
    pub async fn get_active_connection_count(&self) -> usize {
        let allocated_connections = self.allocated_connections.lock();
        allocated_connections.len()
    }

    /// Get utilization percentage
    pub async fn get_utilization(&self) -> f32 {
        let config = self.config.lock();
        let active_count = self.get_active_connection_count().await;
        let max_connections = config.max_connections;

        if max_connections == 0 {
            0.0
        } else {
            active_count as f32 / max_connections as f32
        }
    }

    /// Generate connection report
    pub async fn generate_connection_report(&self) -> String {
        let stats = self.get_statistics().await.unwrap_or_default();
        let available_count = self.get_available_connection_count().await;
        let active_count = self.get_active_connection_count().await;
        let utilization = self.get_utilization().await;

        format!(
            "Database Connection Report:\n\
             - Available connections: {}\n\
             - Active connections: {}\n\
             - Total connections allocated: {}\n\
             - Peak connections: {}\n\
             - Current utilization: {:.1}%\n\
             - Average connection duration: {}s\n\
             - Query throughput: {:.2} queries/sec",
            available_count,
            active_count,
            stats.total_connections,
            stats.peak_connections,
            utilization * 100.0,
            stats.average_duration.as_secs(),
            stats.query_throughput
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_management::types::DatabasePoolConfig;

    #[tokio::test]
    async fn test_database_connection_manager_creation() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config).await;
        assert!(mgr.is_ok());
    }

    #[tokio::test]
    async fn test_initial_available_connections() {
        let config = DatabasePoolConfig::default();
        let max = config.max_connections;
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        assert_eq!(mgr.get_available_connection_count().await, max);
    }

    #[tokio::test]
    async fn test_allocate_connections() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        let conns = mgr.allocate_connections(3, "test-001").await;
        assert!(conns.is_ok());
        assert_eq!(conns.unwrap_or_default().len(), 3);
    }

    #[tokio::test]
    async fn test_allocate_zero_connections() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        let conns = mgr.allocate_connections(0, "test-zero").await;
        assert!(conns.is_ok());
        assert!(conns.unwrap_or_default().is_empty());
    }

    #[tokio::test]
    async fn test_active_count_after_allocation() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        mgr.allocate_connections(4, "test-active").await.unwrap_or_default();
        assert_eq!(mgr.get_active_connection_count().await, 4);
    }

    #[tokio::test]
    async fn test_deallocate_connection() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        let conns = mgr.allocate_connections(1, "test-dealloc").await.unwrap_or_default();
        if let Some(conn_id) = conns.first() {
            let result = mgr.deallocate_connection(conn_id).await;
            assert!(result.is_ok());
            assert_eq!(mgr.get_active_connection_count().await, 0);
        }
    }

    #[tokio::test]
    async fn test_deallocate_nonexistent() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        let result = mgr.deallocate_connection("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_deallocate_connections_for_test() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        mgr.allocate_connections(3, "my-test").await.unwrap_or_default();
        let result = mgr.deallocate_connections_for_test("my-test").await;
        assert!(result.is_ok());
        assert_eq!(mgr.get_active_connection_count().await, 0);
    }

    #[tokio::test]
    async fn test_check_availability() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        assert!(mgr.check_availability(5).await.unwrap_or(false));
    }

    #[tokio::test]
    async fn test_utilization_zero_initially() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        assert_eq!(mgr.get_utilization().await, 0.0);
    }

    #[tokio::test]
    async fn test_utilization_after_allocation() {
        let mut config = DatabasePoolConfig::default();
        config.max_connections = 10;
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        mgr.allocate_connections(5, "test-util").await.unwrap_or_default();
        assert_eq!(mgr.get_utilization().await, 0.5);
    }

    #[tokio::test]
    async fn test_generate_connection_report() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        let report = mgr.generate_connection_report().await;
        assert!(report.contains("Database Connection Report"));
    }

    #[test]
    fn test_database_type_variants() {
        assert_eq!(format!("{:?}", DatabaseType::PostgreSQL), "PostgreSQL");
        assert_eq!(format!("{:?}", DatabaseType::MySQL), "MySQL");
        assert_eq!(format!("{:?}", DatabaseType::SQLite), "SQLite");
        assert_eq!(format!("{:?}", DatabaseType::MongoDB), "MongoDB");
        assert_eq!(format!("{:?}", DatabaseType::Redis), "Redis");
        let custom = DatabaseType::Custom("cassandra".to_string());
        match custom {
            DatabaseType::Custom(name) => assert_eq!(name, "cassandra"),
            _ => panic!("wrong variant"),
        }
    }
}
