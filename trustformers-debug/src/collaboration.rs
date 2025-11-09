//! Collaboration features for sharing and annotating debugging reports
//!
//! This module provides tools for team collaboration on debugging sessions,
//! including report sharing, annotation systems, and collaborative analysis.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Collaboration system for debugging reports and analysis
#[derive(Debug, Clone)]
pub struct CollaborationManager {
    /// Shared reports
    shared_reports: HashMap<Uuid, SharedReport>,
    /// Active annotation sessions
    annotation_sessions: HashMap<Uuid, AnnotationSession>,
    /// Team members
    team_members: HashMap<Uuid, TeamMember>,
    /// Comment threads
    comment_threads: HashMap<Uuid, CommentThread>,
}

/// A shared debugging report with collaboration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedReport {
    /// Unique identifier
    pub id: Uuid,
    /// Report title
    pub title: String,
    /// Report content (JSON or Markdown)
    pub content: String,
    /// Content format
    pub format: ReportFormat,
    /// Author information
    pub author: TeamMember,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub last_modified: DateTime<Utc>,
    /// Access permissions
    pub permissions: ReportPermissions,
    /// Associated tags
    pub tags: Vec<String>,
    /// Report version
    pub version: u32,
    /// Sharing status
    pub sharing_status: SharingStatus,
    /// Associated annotations
    pub annotations: Vec<Uuid>,
    /// View count
    pub view_count: u64,
    /// Collaborators
    pub collaborators: Vec<Uuid>,
}

/// Team member information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamMember {
    /// Unique identifier
    pub id: Uuid,
    /// Display name
    pub name: String,
    /// Email address
    pub email: String,
    /// Avatar URL
    pub avatar_url: Option<String>,
    /// Role in the team
    pub role: TeamRole,
    /// Timezone
    pub timezone: String,
    /// Last active timestamp
    pub last_active: DateTime<Utc>,
    /// Notification preferences
    pub notification_preferences: NotificationPreferences,
}

/// Annotation session for collaborative analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationSession {
    /// Session identifier
    pub id: Uuid,
    /// Report being annotated
    pub report_id: Uuid,
    /// Session participants
    pub participants: Vec<Uuid>,
    /// Session creator
    pub creator: Uuid,
    /// Session start time
    pub started_at: DateTime<Utc>,
    /// Session end time (if ended)
    pub ended_at: Option<DateTime<Utc>>,
    /// Session status
    pub status: SessionStatus,
    /// Annotations in this session
    pub annotations: Vec<Annotation>,
    /// Session type
    pub session_type: AnnotationType,
}

/// Individual annotation within a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// Annotation identifier
    pub id: Uuid,
    /// Author of the annotation
    pub author: Uuid,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Annotation type
    pub annotation_type: AnnotationType,
    /// Target location in the report
    pub target: AnnotationTarget,
    /// Annotation content
    pub content: String,
    /// Importance level
    pub importance: ImportanceLevel,
    /// Resolution status
    pub status: AnnotationStatus,
    /// Related annotations
    pub related_annotations: Vec<Uuid>,
    /// Attachments
    pub attachments: Vec<Attachment>,
}

/// Comment thread for discussions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommentThread {
    /// Thread identifier
    pub id: Uuid,
    /// Associated report
    pub report_id: Uuid,
    /// Thread subject
    pub subject: String,
    /// Thread creator
    pub creator: Uuid,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Thread status
    pub status: ThreadStatus,
    /// Comments in the thread
    pub comments: Vec<Comment>,
    /// Thread participants
    pub participants: Vec<Uuid>,
    /// Thread tags
    pub tags: Vec<String>,
}

/// Individual comment in a thread
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    /// Comment identifier
    pub id: Uuid,
    /// Comment author
    pub author: Uuid,
    /// Comment content
    pub content: String,
    /// Creation timestamp
    pub timestamp: DateTime<Utc>,
    /// Edit timestamp (if edited)
    pub edited_at: Option<DateTime<Utc>>,
    /// Parent comment (for replies)
    pub parent_id: Option<Uuid>,
    /// Comment reactions
    pub reactions: HashMap<String, Vec<Uuid>>,
    /// Attachments
    pub attachments: Vec<Attachment>,
}

/// File or media attachment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    /// Attachment identifier
    pub id: Uuid,
    /// File name
    pub filename: String,
    /// Content type
    pub content_type: String,
    /// File size in bytes
    pub size: u64,
    /// Storage URL or path
    pub url: String,
    /// Upload timestamp
    pub uploaded_at: DateTime<Utc>,
    /// Uploader
    pub uploader: Uuid,
}

/// Report format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Markdown,
    Html,
    Pdf,
    Custom(String),
}

/// Access permissions for shared reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportPermissions {
    /// Public visibility
    pub is_public: bool,
    /// Team visibility
    pub team_access: bool,
    /// Specific user access
    pub user_access: Vec<Uuid>,
    /// Edit permissions
    pub edit_permissions: Vec<Uuid>,
    /// Comment permissions
    pub comment_permissions: Vec<Uuid>,
}

/// Sharing status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SharingStatus {
    Private,
    TeamShared,
    PublicShared,
    LinkShared(String),
}

/// Team roles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TeamRole {
    Owner,
    Admin,
    Developer,
    Reviewer,
    Viewer,
}

/// Notification preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPreferences {
    /// Email notifications
    pub email_enabled: bool,
    /// New comment notifications
    pub comment_notifications: bool,
    /// New annotation notifications
    pub annotation_notifications: bool,
    /// Report share notifications
    pub share_notifications: bool,
    /// Digest frequency
    pub digest_frequency: DigestFrequency,
}

/// Notification digest frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DigestFrequency {
    Immediate,
    Hourly,
    Daily,
    Weekly,
    Disabled,
}

/// Annotation session status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStatus {
    Active,
    Paused,
    Completed,
    Cancelled,
}

/// Annotation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationType {
    Note,
    Issue,
    Question,
    Improvement,
    Bug,
    Performance,
    Security,
    Documentation,
}

/// Target location for annotations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationTarget {
    /// Line number in report
    Line(u32),
    /// Range of lines
    LineRange(u32, u32),
    /// Section identifier
    Section(String),
    /// Specific element
    Element(String),
    /// Custom location
    Custom(HashMap<String, String>),
}

/// Importance levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportanceLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Annotation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationStatus {
    Open,
    InProgress,
    Resolved,
    Dismissed,
}

/// Comment thread status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreadStatus {
    Open,
    Resolved,
    Locked,
    Archived,
}

impl CollaborationManager {
    /// Create a new collaboration manager
    pub fn new() -> Self {
        Self {
            shared_reports: HashMap::new(),
            annotation_sessions: HashMap::new(),
            team_members: HashMap::new(),
            comment_threads: HashMap::new(),
        }
    }

    /// Share a debugging report
    pub fn share_report(
        &mut self,
        title: String,
        content: String,
        format: ReportFormat,
        author: TeamMember,
        permissions: ReportPermissions,
        tags: Vec<String>,
    ) -> Result<Uuid> {
        let report_id = Uuid::new_v4();
        let now = Utc::now();

        let shared_report = SharedReport {
            id: report_id,
            title,
            content,
            format,
            author: author.clone(),
            created_at: now,
            last_modified: now,
            permissions,
            tags,
            version: 1,
            sharing_status: SharingStatus::Private,
            annotations: Vec::new(),
            view_count: 0,
            collaborators: vec![author.id],
        };

        self.shared_reports.insert(report_id, shared_report);
        self.team_members.insert(author.id, author);

        Ok(report_id)
    }

    /// Get a shared report
    pub fn get_report(&mut self, report_id: Uuid) -> Option<&mut SharedReport> {
        if let Some(report) = self.shared_reports.get_mut(&report_id) {
            report.view_count += 1;
            Some(report)
        } else {
            None
        }
    }

    /// Start an annotation session
    pub fn start_annotation_session(
        &mut self,
        report_id: Uuid,
        creator: Uuid,
        participants: Vec<Uuid>,
        session_type: AnnotationType,
    ) -> Result<Uuid> {
        let session_id = Uuid::new_v4();
        let now = Utc::now();

        let session = AnnotationSession {
            id: session_id,
            report_id,
            participants,
            creator,
            started_at: now,
            ended_at: None,
            status: SessionStatus::Active,
            annotations: Vec::new(),
            session_type,
        };

        self.annotation_sessions.insert(session_id, session);
        Ok(session_id)
    }

    /// Add an annotation to a session
    pub fn add_annotation(
        &mut self,
        session_id: Uuid,
        author: Uuid,
        annotation_type: AnnotationType,
        target: AnnotationTarget,
        content: String,
        importance: ImportanceLevel,
    ) -> Result<Uuid> {
        let annotation_id = Uuid::new_v4();
        let now = Utc::now();

        let annotation = Annotation {
            id: annotation_id,
            author,
            timestamp: now,
            annotation_type,
            target,
            content,
            importance,
            status: AnnotationStatus::Open,
            related_annotations: Vec::new(),
            attachments: Vec::new(),
        };

        if let Some(session) = self.annotation_sessions.get_mut(&session_id) {
            session.annotations.push(annotation);

            // Add annotation to the report as well
            if let Some(report) = self.shared_reports.get_mut(&session.report_id) {
                report.annotations.push(annotation_id);
                report.last_modified = now;
            }

            Ok(annotation_id)
        } else {
            Err(anyhow::anyhow!("Annotation session not found"))
        }
    }

    /// Create a comment thread
    pub fn create_comment_thread(
        &mut self,
        report_id: Uuid,
        subject: String,
        creator: Uuid,
        tags: Vec<String>,
    ) -> Result<Uuid> {
        let thread_id = Uuid::new_v4();
        let now = Utc::now();

        let thread = CommentThread {
            id: thread_id,
            report_id,
            subject,
            creator,
            created_at: now,
            status: ThreadStatus::Open,
            comments: Vec::new(),
            participants: vec![creator],
            tags,
        };

        self.comment_threads.insert(thread_id, thread);
        Ok(thread_id)
    }

    /// Add a comment to a thread
    pub fn add_comment(
        &mut self,
        thread_id: Uuid,
        author: Uuid,
        content: String,
        parent_id: Option<Uuid>,
    ) -> Result<Uuid> {
        let comment_id = Uuid::new_v4();
        let now = Utc::now();

        let comment = Comment {
            id: comment_id,
            author,
            content,
            timestamp: now,
            edited_at: None,
            parent_id,
            reactions: HashMap::new(),
            attachments: Vec::new(),
        };

        if let Some(thread) = self.comment_threads.get_mut(&thread_id) {
            thread.comments.push(comment);

            // Add author as participant if not already present
            if !thread.participants.contains(&author) {
                thread.participants.push(author);
            }

            Ok(comment_id)
        } else {
            Err(anyhow::anyhow!("Comment thread not found"))
        }
    }

    /// Add a team member
    pub fn add_team_member(&mut self, member: TeamMember) -> Uuid {
        let member_id = member.id;
        self.team_members.insert(member_id, member);
        member_id
    }

    /// Update report sharing status
    pub fn update_sharing_status(&mut self, report_id: Uuid, status: SharingStatus) -> Result<()> {
        if let Some(report) = self.shared_reports.get_mut(&report_id) {
            report.sharing_status = status;
            report.last_modified = Utc::now();
            Ok(())
        } else {
            Err(anyhow::anyhow!("Report not found"))
        }
    }

    /// Get reports accessible to a user
    pub fn get_user_reports(&self, user_id: Uuid) -> Vec<&SharedReport> {
        self.shared_reports
            .values()
            .filter(|report| {
                // Check if user has access
                report.author.id == user_id
                    || report.collaborators.contains(&user_id)
                    || report.permissions.user_access.contains(&user_id)
                    || report.permissions.is_public
                    || (report.permissions.team_access && self.team_members.contains_key(&user_id))
            })
            .collect()
    }

    /// Get annotations for a report
    pub fn get_report_annotations(&self, report_id: Uuid) -> Vec<&Annotation> {
        let mut annotations = Vec::new();

        for session in self.annotation_sessions.values() {
            if session.report_id == report_id {
                annotations.extend(session.annotations.iter());
            }
        }

        annotations
    }

    /// Get comment threads for a report
    pub fn get_report_threads(&self, report_id: Uuid) -> Vec<&CommentThread> {
        self.comment_threads
            .values()
            .filter(|thread| thread.report_id == report_id)
            .collect()
    }

    /// Export collaboration data
    pub fn export_collaboration_data(&self) -> Result<String> {
        #[derive(Serialize)]
        struct CollaborationExport<'a> {
            reports: Vec<&'a SharedReport>,
            sessions: Vec<&'a AnnotationSession>,
            threads: Vec<&'a CommentThread>,
            members: Vec<&'a TeamMember>,
            export_timestamp: DateTime<Utc>,
        }

        let export_data = CollaborationExport {
            reports: self.shared_reports.values().collect(),
            sessions: self.annotation_sessions.values().collect(),
            threads: self.comment_threads.values().collect(),
            members: self.team_members.values().collect(),
            export_timestamp: Utc::now(),
        };

        serde_json::to_string_pretty(&export_data)
            .map_err(|e| anyhow::anyhow!("Failed to export collaboration data: {}", e))
    }

    /// Generate collaboration statistics
    pub fn get_collaboration_stats(&self) -> CollaborationStats {
        let total_reports = self.shared_reports.len();
        let total_annotations =
            self.annotation_sessions.values().map(|s| s.annotations.len()).sum();
        let total_comments = self.comment_threads.values().map(|t| t.comments.len()).sum();
        let active_sessions = self
            .annotation_sessions
            .values()
            .filter(|s| matches!(s.status, SessionStatus::Active))
            .count();
        let team_size = self.team_members.len();

        CollaborationStats {
            total_reports,
            total_annotations,
            total_comments,
            active_sessions,
            team_size,
            reports_per_member: if team_size > 0 {
                total_reports as f64 / team_size as f64
            } else {
                0.0
            },
            annotations_per_report: if total_reports > 0 {
                total_annotations as f64 / total_reports as f64
            } else {
                0.0
            },
        }
    }
}

/// Collaboration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationStats {
    pub total_reports: usize,
    pub total_annotations: usize,
    pub total_comments: usize,
    pub active_sessions: usize,
    pub team_size: usize,
    pub reports_per_member: f64,
    pub annotations_per_report: f64,
}

impl Default for CollaborationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_member() -> TeamMember {
        TeamMember {
            id: Uuid::new_v4(),
            name: "Test User".to_string(),
            email: "test@example.com".to_string(),
            avatar_url: None,
            role: TeamRole::Developer,
            timezone: "UTC".to_string(),
            last_active: Utc::now(),
            notification_preferences: NotificationPreferences {
                email_enabled: true,
                comment_notifications: true,
                annotation_notifications: true,
                share_notifications: true,
                digest_frequency: DigestFrequency::Daily,
            },
        }
    }

    #[test]
    fn test_share_report() {
        let mut manager = CollaborationManager::new();
        let author = create_test_member();

        let permissions = ReportPermissions {
            is_public: false,
            team_access: true,
            user_access: vec![],
            edit_permissions: vec![author.id],
            comment_permissions: vec![author.id],
        };

        let report_id = manager
            .share_report(
                "Test Report".to_string(),
                "Report content".to_string(),
                ReportFormat::Markdown,
                author,
                permissions,
                vec!["test".to_string()],
            )
            .unwrap();

        assert!(manager.shared_reports.contains_key(&report_id));
    }

    #[test]
    fn test_annotation_session() {
        let mut manager = CollaborationManager::new();
        let author = create_test_member();

        // First create a report
        let permissions = ReportPermissions {
            is_public: false,
            team_access: true,
            user_access: vec![],
            edit_permissions: vec![author.id],
            comment_permissions: vec![author.id],
        };

        let report_id = manager
            .share_report(
                "Test Report".to_string(),
                "Report content".to_string(),
                ReportFormat::Markdown,
                author.clone(),
                permissions,
                vec![],
            )
            .unwrap();

        // Start annotation session
        let session_id = manager
            .start_annotation_session(report_id, author.id, vec![author.id], AnnotationType::Note)
            .unwrap();

        // Add annotation
        let _annotation_id = manager
            .add_annotation(
                session_id,
                author.id,
                AnnotationType::Issue,
                AnnotationTarget::Line(10),
                "This looks like a bug".to_string(),
                ImportanceLevel::High,
            )
            .unwrap();

        assert!(manager.annotation_sessions.contains_key(&session_id));

        // Verify annotation was added to report
        let report = manager.shared_reports.get(&report_id).unwrap();
        assert!(!report.annotations.is_empty());
    }

    #[test]
    fn test_comment_thread() {
        let mut manager = CollaborationManager::new();
        let author = create_test_member();

        // Create a report first
        let permissions = ReportPermissions {
            is_public: false,
            team_access: true,
            user_access: vec![],
            edit_permissions: vec![author.id],
            comment_permissions: vec![author.id],
        };

        let report_id = manager
            .share_report(
                "Test Report".to_string(),
                "Report content".to_string(),
                ReportFormat::Markdown,
                author.clone(),
                permissions,
                vec![],
            )
            .unwrap();

        // Create comment thread
        let thread_id = manager
            .create_comment_thread(
                report_id,
                "Discussion about results".to_string(),
                author.id,
                vec!["discussion".to_string()],
            )
            .unwrap();

        // Add comment
        let _comment_id = manager
            .add_comment(thread_id, author.id, "Great analysis!".to_string(), None)
            .unwrap();

        assert!(manager.comment_threads.contains_key(&thread_id));

        let thread = manager.comment_threads.get(&thread_id).unwrap();
        assert_eq!(thread.comments.len(), 1);
    }

    #[test]
    fn test_collaboration_stats() {
        let mut manager = CollaborationManager::new();
        let author = create_test_member();

        // Add some test data
        manager.add_team_member(author.clone());

        let permissions = ReportPermissions {
            is_public: false,
            team_access: true,
            user_access: vec![],
            edit_permissions: vec![author.id],
            comment_permissions: vec![author.id],
        };

        manager
            .share_report(
                "Test Report".to_string(),
                "Content".to_string(),
                ReportFormat::Markdown,
                author,
                permissions,
                vec![],
            )
            .unwrap();

        let stats = manager.get_collaboration_stats();
        assert_eq!(stats.total_reports, 1);
        assert_eq!(stats.team_size, 1);
        assert_eq!(stats.reports_per_member, 1.0);
    }
}
