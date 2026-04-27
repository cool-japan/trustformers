#[cfg(test)]
mod tests {
    use crate::network_adaptation::sync::*;
    use crate::network_adaptation::types::*;
    use std::collections::HashMap;
    use std::time::Instant;

    // --- SyncStatus Tests ---

    #[test]
    fn test_sync_status_variants() {
        assert_eq!(SyncStatus::Pending, SyncStatus::Pending);
        assert_ne!(SyncStatus::Pending, SyncStatus::InProgress);
        let _ = SyncStatus::Completed;
        let _ = SyncStatus::Failed;
        let _ = SyncStatus::Cancelled;
    }

    #[test]
    fn test_sync_status_clone() {
        let status = SyncStatus::InProgress;
        let cloned = status.clone();
        assert_eq!(cloned, SyncStatus::InProgress);
    }

    // --- SyncRequest Tests ---

    #[test]
    fn test_sync_request_creation() {
        let req = SyncRequest {
            sync_id: "sync-001".to_string(),
            source_version: "v1.0".to_string(),
            target_version: "v1.1".to_string(),
            model_data: vec![0u8; 64],
            priority: 5,
            timestamp: Instant::now(),
        };
        assert_eq!(req.sync_id, "sync-001");
        assert_eq!(req.source_version, "v1.0");
        assert_eq!(req.target_version, "v1.1");
        assert_eq!(req.model_data.len(), 64);
        assert_eq!(req.priority, 5);
    }

    #[test]
    fn test_sync_request_clone() {
        let req = SyncRequest {
            sync_id: "sync-002".to_string(),
            source_version: "v2.0".to_string(),
            target_version: "v2.1".to_string(),
            model_data: vec![1, 2, 3, 4],
            priority: 10,
            timestamp: Instant::now(),
        };
        let cloned = req.clone();
        assert_eq!(cloned.sync_id, "sync-002");
        assert_eq!(cloned.model_data.len(), 4);
    }

    // --- ModelVersion Tests ---

    #[test]
    fn test_model_version_creation() {
        let version = ModelVersion {
            version: "v1.0.0".to_string(),
            timestamp: Instant::now(),
            checksum: "abc123def456".to_string(),
            size_bytes: 1024 * 1024,
            parent_version: None,
            metadata: HashMap::new(),
        };
        assert_eq!(version.version, "v1.0.0");
        assert_eq!(version.size_bytes, 1024 * 1024);
        assert!(version.parent_version.is_none());
    }

    #[test]
    fn test_model_version_with_parent() {
        let version = ModelVersion {
            version: "v1.1.0".to_string(),
            timestamp: Instant::now(),
            checksum: "xyz789".to_string(),
            size_bytes: 2 * 1024 * 1024,
            parent_version: Some("v1.0.0".to_string()),
            metadata: {
                let mut m = HashMap::new();
                m.insert("author".to_string(), "test".to_string());
                m
            },
        };
        assert_eq!(version.parent_version, Some("v1.0.0".to_string()));
        assert_eq!(version.metadata.len(), 1);
    }

    #[test]
    fn test_model_version_clone() {
        let version = ModelVersion {
            version: "v2.0".to_string(),
            timestamp: Instant::now(),
            checksum: "hash123".to_string(),
            size_bytes: 512,
            parent_version: Some("v1.0".to_string()),
            metadata: HashMap::new(),
        };
        let cloned = version.clone();
        assert_eq!(cloned.version, "v2.0");
        assert_eq!(cloned.checksum, "hash123");
    }

    // --- SyncResponse Tests ---

    #[test]
    fn test_sync_response_success() {
        let resp = SyncResponse {
            sync_id: "sync-001".to_string(),
            status: SyncStatus::Completed,
            timestamp: Instant::now(),
            result_data: vec![10, 20, 30],
            error_message: None,
            metadata: HashMap::new(),
        };
        assert_eq!(resp.status, SyncStatus::Completed);
        assert!(resp.error_message.is_none());
        assert_eq!(resp.result_data.len(), 3);
    }

    #[test]
    fn test_sync_response_failure() {
        let resp = SyncResponse {
            sync_id: "sync-002".to_string(),
            status: SyncStatus::Failed,
            timestamp: Instant::now(),
            result_data: vec![],
            error_message: Some("Network timeout".to_string()),
            metadata: HashMap::new(),
        };
        assert_eq!(resp.status, SyncStatus::Failed);
        assert!(resp.error_message.is_some());
        assert_eq!(
            resp.error_message.as_ref().expect("expected error"),
            "Network timeout"
        );
    }

    #[test]
    fn test_sync_response_clone() {
        let resp = SyncResponse {
            sync_id: "sync-003".to_string(),
            status: SyncStatus::InProgress,
            timestamp: Instant::now(),
            result_data: vec![],
            error_message: None,
            metadata: {
                let mut m = HashMap::new();
                m.insert("key".to_string(), "value".to_string());
                m
            },
        };
        let cloned = resp.clone();
        assert_eq!(cloned.sync_id, "sync-003");
        assert_eq!(cloned.metadata.len(), 1);
    }

    // --- ConflictMetadata Tests ---

    #[test]
    fn test_conflict_metadata_creation() {
        let conflict = ConflictMetadata {
            conflict_id: "conflict-001".to_string(),
            source_version: "v1.0".to_string(),
            target_version: "v1.1".to_string(),
            conflict_type: "parameter_divergence".to_string(),
            timestamp: Instant::now(),
            resolution_strategy: ConflictResolutionStrategy::ServerDecision,
            metadata: HashMap::new(),
        };
        assert_eq!(conflict.conflict_id, "conflict-001");
        assert_eq!(conflict.conflict_type, "parameter_divergence");
    }

    #[test]
    fn test_conflict_metadata_clone() {
        let conflict = ConflictMetadata {
            conflict_id: "conflict-002".to_string(),
            source_version: "v2.0".to_string(),
            target_version: "v2.1".to_string(),
            conflict_type: "version_conflict".to_string(),
            timestamp: Instant::now(),
            resolution_strategy: ConflictResolutionStrategy::UserDecision,
            metadata: {
                let mut m = HashMap::new();
                m.insert("severity".to_string(), "high".to_string());
                m
            },
        };
        let cloned = conflict.clone();
        assert_eq!(cloned.conflict_id, "conflict-002");
        assert_eq!(cloned.metadata.len(), 1);
    }

    // --- Integration Tests ---

    #[test]
    fn test_sync_workflow_pending_to_completed() {
        let req = SyncRequest {
            sync_id: "workflow-001".to_string(),
            source_version: "v1.0".to_string(),
            target_version: "v1.1".to_string(),
            model_data: vec![0u8; 32],
            priority: 5,
            timestamp: Instant::now(),
        };

        // Simulate status transitions
        let pending = SyncStatus::Pending;
        assert_eq!(pending, SyncStatus::Pending);

        let in_progress = SyncStatus::InProgress;
        assert_eq!(in_progress, SyncStatus::InProgress);

        let completed = SyncStatus::Completed;
        assert_eq!(completed, SyncStatus::Completed);

        let resp = SyncResponse {
            sync_id: req.sync_id.clone(),
            status: completed,
            timestamp: Instant::now(),
            result_data: vec![1, 2, 3],
            error_message: None,
            metadata: HashMap::new(),
        };
        assert_eq!(resp.sync_id, "workflow-001");
        assert_eq!(resp.status, SyncStatus::Completed);
    }

    #[test]
    fn test_version_chain() {
        let v1 = ModelVersion {
            version: "v1.0".to_string(),
            timestamp: Instant::now(),
            checksum: "aaa".to_string(),
            size_bytes: 100,
            parent_version: None,
            metadata: HashMap::new(),
        };

        let v2 = ModelVersion {
            version: "v2.0".to_string(),
            timestamp: Instant::now(),
            checksum: "bbb".to_string(),
            size_bytes: 200,
            parent_version: Some(v1.version.clone()),
            metadata: HashMap::new(),
        };

        let v3 = ModelVersion {
            version: "v3.0".to_string(),
            timestamp: Instant::now(),
            checksum: "ccc".to_string(),
            size_bytes: 300,
            parent_version: Some(v2.version.clone()),
            metadata: HashMap::new(),
        };

        assert_eq!(v2.parent_version, Some("v1.0".to_string()));
        assert_eq!(v3.parent_version, Some("v2.0".to_string()));
        assert!(v1.parent_version.is_none());
    }

    #[test]
    fn test_multiple_sync_requests_ordering() {
        let reqs: Vec<SyncRequest> = (0..5)
            .map(|i| SyncRequest {
                sync_id: format!("sync-{:03}", i),
                source_version: format!("v{}.0", i),
                target_version: format!("v{}.1", i),
                model_data: vec![i as u8; 16],
                priority: (5 - i) as u8,
                timestamp: Instant::now(),
            })
            .collect();

        assert_eq!(reqs.len(), 5);
        assert_eq!(reqs[0].priority, 5);
        assert_eq!(reqs[4].priority, 1);
    }
}
