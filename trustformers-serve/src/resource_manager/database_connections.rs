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
