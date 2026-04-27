//! Database connection pool management for test parallelization.

// Re-export types for external access
pub use super::types::DatabaseUsageStatistics;
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::{debug, info};

use crate::test_parallelization::DatabasePoolConfig;

/// Database connection manager
pub struct DatabaseConnectionManager {
    /// Configuration
    config: Arc<RwLock<DatabasePoolConfig>>,
    /// Active connections
    active_connections: Arc<Mutex<HashMap<String, DatabaseConnection>>>,
    /// Connection pool
    connection_pool: Arc<Mutex<Vec<DatabaseConnection>>>,
    /// Usage statistics
    usage_stats: Arc<Mutex<DatabaseUsageStatistics>>,
}

/// Database connection information
#[derive(Debug, Clone)]
pub struct DatabaseConnection {
    /// Connection ID
    pub connection_id: String,
    /// Database type
    pub database_type: DatabaseType,
    /// Connection string
    pub connection_string: String,
    /// Test ID using this connection
    pub test_id: Option<String>,
    /// Connection timestamp
    pub connected_at: DateTime<Utc>,
    /// Last used timestamp
    pub last_used: DateTime<Utc>,
    /// Connection status
    pub status: ConnectionStatus,
    /// Connection metadata
    pub metadata: HashMap<String, String>,
}

/// Database types
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

/// Connection status
#[derive(Debug, Clone)]
pub enum ConnectionStatus {
    /// Active and ready
    Active,
    /// In use by test
    InUse,
    /// Connection error
    Error(String),
    /// Closed
    Closed,
    /// Maintenance mode
    Maintenance,
}

impl DatabaseConnectionManager {
    /// Create new database connection manager
    pub async fn new(config: DatabasePoolConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            active_connections: Arc::new(Mutex::new(HashMap::new())),
            connection_pool: Arc::new(Mutex::new(Vec::new())),
            usage_stats: Arc::new(Mutex::new(DatabaseUsageStatistics::default())),
        })
    }

    /// Allocate database connections for a test
    pub async fn allocate_connections(&self, count: usize, test_id: &str) -> Result<Vec<String>> {
        info!(
            "Allocating {} database connections for test: {}",
            count, test_id
        );

        // For now, return placeholder connection IDs
        // In a real implementation, this would:
        // 1. Check available connections in pool
        // 2. Create new connections if needed
        // 3. Associate connections with test ID
        // 4. Track allocation

        let connections: Vec<String> =
            (0..count).map(|i| format!("conn-{}-{}", test_id, i)).collect();

        // Create connection records
        for conn_id in &connections {
            let connection = DatabaseConnection {
                connection_id: conn_id.clone(),
                database_type: DatabaseType::PostgreSQL,
                connection_string: format!("postgres://localhost:5432/test_{}", test_id),
                test_id: Some(test_id.to_string()),
                connected_at: Utc::now(),
                last_used: Utc::now(),
                status: ConnectionStatus::InUse,
                metadata: HashMap::new(),
            };

            let mut active_connections = self.active_connections.lock();
            active_connections.insert(conn_id.clone(), connection);
        }

        // Update statistics
        let mut stats = self.usage_stats.lock();
        stats.total_allocated += connections.len() as u64;
        stats.currently_active += connections.len();

        info!(
            "Allocated database connections {:?} for test: {}",
            connections, test_id
        );
        Ok(connections)
    }

    /// Deallocate a specific connection
    pub async fn deallocate_connection(&self, conn_id: &str) -> Result<()> {
        debug!("Deallocating database connection: {}", conn_id);

        let mut active_connections = self.active_connections.lock();
        if let Some(connection) = active_connections.remove(conn_id) {
            // Return connection to pool if still valid
            if matches!(
                connection.status,
                ConnectionStatus::InUse | ConnectionStatus::Active
            ) {
                let mut returned_connection = connection;
                returned_connection.test_id = None;
                returned_connection.status = ConnectionStatus::Active;

                let mut connection_pool = self.connection_pool.lock();
                connection_pool.push(returned_connection);
            }

            // Update statistics
            let mut stats = self.usage_stats.lock();
            stats.currently_active = stats.currently_active.saturating_sub(1);

            info!("Successfully deallocated database connection: {}", conn_id);
        } else {
            debug!(
                "Database connection {} was not found or already deallocated",
                conn_id
            );
        }

        Ok(())
    }

    /// Deallocate connections for a test
    pub async fn deallocate_connections_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating database connections for test: {}", test_id);

        let mut active_connections = self.active_connections.lock();
        let connections_to_remove: Vec<String> = active_connections
            .iter()
            .filter(|(_, connection)| {
                connection.test_id.as_ref().map(|id| id == test_id).unwrap_or(false)
            })
            .map(|(conn_id, _)| conn_id.clone())
            .collect();

        let mut connection_pool = self.connection_pool.lock();
        for conn_id in &connections_to_remove {
            if let Some(connection) = active_connections.remove(conn_id) {
                // Return to pool if connection is still valid
                if matches!(
                    connection.status,
                    ConnectionStatus::InUse | ConnectionStatus::Active
                ) {
                    let mut returned_connection = connection;
                    returned_connection.test_id = None;
                    returned_connection.status = ConnectionStatus::Active;
                    connection_pool.push(returned_connection);
                }
            }
        }

        // Update statistics
        let mut stats = self.usage_stats.lock();
        stats.currently_active = stats.currently_active.saturating_sub(connections_to_remove.len());

        info!(
            "Released {} database connections for test: {}",
            connections_to_remove.len(),
            test_id
        );
        Ok(())
    }

    /// Check if requested number of connections are available
    pub async fn check_availability(&self, count: usize) -> Result<bool> {
        let config = self.config.read();
        let active_count = self.active_connections.lock().len();
        let _pool_count = self.connection_pool.lock().len();

        // Check against configured maximum connections
        Ok(active_count + count <= config.max_connections)
    }

    /// Get database usage statistics
    pub async fn get_statistics(&self) -> Result<DatabaseUsageStatistics> {
        let stats = self.usage_stats.lock();
        // MutexGuard doesn't implement Clone, dereference to clone the inner value
        Ok((*stats).clone())
    }

    /// Get current active connections
    pub async fn get_active_connections(&self) -> Result<Vec<DatabaseConnection>> {
        let active_connections = self.active_connections.lock();
        Ok(active_connections.values().cloned().collect())
    }

    /// Get connections for a specific test
    pub async fn get_connections_for_test(&self, test_id: &str) -> Result<Vec<DatabaseConnection>> {
        let active_connections = self.active_connections.lock();
        Ok(active_connections
            .values()
            .filter(|connection| {
                connection.test_id.as_ref().map(|id| id == test_id).unwrap_or(false)
            })
            .cloned()
            .collect())
    }

    /// Test connection health
    pub async fn test_connection_health(&self, conn_id: &str) -> Result<bool> {
        debug!("Testing health of database connection: {}", conn_id);

        let active_connections = self.active_connections.lock();
        if let Some(connection) = active_connections.get(conn_id) {
            // In a real implementation, this would:
            // 1. Execute a simple query (e.g., SELECT 1)
            // 2. Check connection latency
            // 3. Verify connection state

            match connection.status {
                ConnectionStatus::Active | ConnectionStatus::InUse => Ok(true),
                _ => Ok(false),
            }
        } else {
            Ok(false)
        }
    }

    /// Cleanup idle connections
    pub async fn cleanup_idle_connections(&self, idle_timeout: Duration) -> Result<usize> {
        let mut cleaned_count = 0;
        let cutoff_time = Utc::now() - chrono::Duration::from_std(idle_timeout)?;

        let mut active_connections = self.active_connections.lock();
        let connections_to_remove: Vec<String> = active_connections
            .iter()
            .filter(|(_, connection)| {
                connection.last_used < cutoff_time
                    && matches!(connection.status, ConnectionStatus::Active)
            })
            .map(|(conn_id, _)| conn_id.clone())
            .collect();

        for conn_id in connections_to_remove {
            active_connections.remove(&conn_id);
            cleaned_count += 1;
        }

        // Update statistics
        let mut stats = self.usage_stats.lock();
        stats.currently_active = stats.currently_active.saturating_sub(cleaned_count);

        if cleaned_count > 0 {
            info!("Cleaned up {} idle database connections", cleaned_count);
        }

        Ok(cleaned_count)
    }

    /// Force close all connections
    pub async fn force_close_all(&self) -> Result<usize> {
        let mut active_connections = self.active_connections.lock();
        let count = active_connections.len();
        active_connections.clear();

        let mut connection_pool = self.connection_pool.lock();
        connection_pool.clear();

        // Reset statistics
        let mut stats = self.usage_stats.lock();
        stats.currently_active = 0;

        info!("Force closed all {} database connections", count);
        Ok(count)
    }

    /// Update configuration
    pub async fn update_config(&self, config: DatabasePoolConfig) -> Result<()> {
        let mut current_config = self.config.write();
        *current_config = config;
        info!("Updated database connection manager configuration");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_parallelization::DatabasePoolConfig;

    #[tokio::test]
    async fn test_database_connection_manager_new() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config).await;
        assert!(mgr.is_ok());
    }

    #[tokio::test]
    async fn test_allocate_connections_returns_ids() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        let conns = mgr.allocate_connections(2, "test-db-001").await.unwrap_or_default();
        assert_eq!(conns.len(), 2);
    }

    #[tokio::test]
    async fn test_allocate_updates_stats() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        mgr.allocate_connections(3, "test-db-002").await.unwrap_or_default();
        let stats = mgr.get_statistics().await.unwrap_or_default();
        assert_eq!(stats.total_allocated, 3);
        assert_eq!(stats.currently_active, 3);
    }

    #[tokio::test]
    async fn test_deallocate_connection() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        let conns = mgr.allocate_connections(1, "test-dealloc").await.unwrap_or_default();
        if let Some(conn_id) = conns.first() {
            let r = mgr.deallocate_connection(conn_id).await;
            assert!(r.is_ok());
        }
    }

    #[tokio::test]
    async fn test_deallocate_nonexistent_connection() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        let r = mgr.deallocate_connection("nonexistent-conn").await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn test_deallocate_connections_for_test() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        mgr.allocate_connections(2, "test-batch-dealloc").await.unwrap_or_default();
        let r = mgr.deallocate_connections_for_test("test-batch-dealloc").await;
        assert!(r.is_ok());
        let stats = mgr.get_statistics().await.unwrap_or_default();
        assert_eq!(stats.currently_active, 0);
    }

    #[tokio::test]
    async fn test_check_availability_empty() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        let avail = mgr.check_availability(5).await.unwrap_or(false);
        assert!(avail);
    }

    #[tokio::test]
    async fn test_check_availability_over_limit() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        let avail = mgr.check_availability(1000).await.unwrap_or(true);
        assert!(!avail);
    }

    #[tokio::test]
    async fn test_get_active_connections_empty() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        let conns = mgr.get_active_connections().await.unwrap_or_default();
        assert!(conns.is_empty());
    }

    #[tokio::test]
    async fn test_get_active_connections_after_alloc() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        mgr.allocate_connections(2, "t-active").await.unwrap_or_default();
        let conns = mgr.get_active_connections().await.unwrap_or_default();
        assert_eq!(conns.len(), 2);
    }

    #[tokio::test]
    async fn test_get_connections_for_test() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        mgr.allocate_connections(2, "t-get-by-id").await.unwrap_or_default();
        let conns = mgr.get_connections_for_test("t-get-by-id").await.unwrap_or_default();
        assert_eq!(conns.len(), 2);
    }

    #[tokio::test]
    async fn test_test_connection_health_true() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        let ids = mgr.allocate_connections(1, "t-health").await.unwrap_or_default();
        if let Some(id) = ids.first() {
            let healthy = mgr.test_connection_health(id).await.unwrap_or(false);
            assert!(healthy);
        }
    }

    #[tokio::test]
    async fn test_test_connection_health_nonexistent() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        let healthy = mgr.test_connection_health("no-such-conn").await.unwrap_or(true);
        assert!(!healthy);
    }

    #[tokio::test]
    async fn test_force_close_all() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        mgr.allocate_connections(3, "t-force-close").await.unwrap_or_default();
        let count = mgr.force_close_all().await.unwrap_or(0);
        assert!(count >= 3);
        let stats = mgr.get_statistics().await.unwrap_or_default();
        assert_eq!(stats.currently_active, 0);
    }

    #[tokio::test]
    async fn test_cleanup_idle_connections_none() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseConnectionManager::new(config)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        let count = mgr
            .cleanup_idle_connections(std::time::Duration::from_secs(0))
            .await
            .unwrap_or(99);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_database_type_variants() {
        let types = [
            format!("{:?}", DatabaseType::PostgreSQL),
            format!("{:?}", DatabaseType::MySQL),
            format!("{:?}", DatabaseType::SQLite),
            format!("{:?}", DatabaseType::MongoDB),
            format!("{:?}", DatabaseType::Redis),
            format!("{:?}", DatabaseType::Custom("custom-db".to_string())),
        ];
        assert_eq!(types[0], "PostgreSQL");
        assert_eq!(types[4], "Redis");
    }

    #[test]
    fn test_connection_status_variants() {
        let s1 = format!("{:?}", ConnectionStatus::Active);
        let s2 = format!("{:?}", ConnectionStatus::InUse);
        let s3 = format!("{:?}", ConnectionStatus::Closed);
        let s4 = format!("{:?}", ConnectionStatus::Maintenance);
        assert_eq!(s1, "Active");
        assert_eq!(s2, "InUse");
        assert_eq!(s3, "Closed");
        assert_eq!(s4, "Maintenance");
    }

    #[test]
    fn test_database_connection_creation() {
        let conn = DatabaseConnection {
            connection_id: "c-1".to_string(),
            database_type: DatabaseType::PostgreSQL,
            connection_string: "postgres://localhost:5432/test".to_string(),
            test_id: Some("t-1".to_string()),
            connected_at: chrono::Utc::now(),
            last_used: chrono::Utc::now(),
            status: ConnectionStatus::Active,
            metadata: HashMap::new(),
        };
        assert_eq!(conn.connection_id, "c-1");
        assert!(conn.test_id.is_some());
    }
}
