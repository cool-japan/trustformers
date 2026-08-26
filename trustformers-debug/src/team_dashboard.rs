//! Team collaboration dashboard for debugging sessions
//!
//! This module provides a real-time dashboard for team collaboration,
//! showing active debugging sessions, shared reports, and team activity.

use crate::collaboration::CollaborationManager;
use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Team dashboard for collaborative debugging
#[derive(Debug, Clone)]
pub struct TeamDashboard {
    /// Dashboard configuration
    config: DashboardConfig,
    /// Real-time activity feed
    activity_feed: Vec<ActivityEvent>,
    /// Team metrics
    metrics: TeamMetrics,
    /// Active sessions tracking
    active_sessions: HashMap<Uuid, SessionActivity>,
    /// Notification system
    notifications: NotificationSystem,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Refresh interval in seconds
    pub refresh_interval: u64,
    /// Maximum activities to show
    pub max_activities: usize,
    /// Enable real-time updates
    pub real_time_updates: bool,
    /// Show detailed metrics
    pub show_detailed_metrics: bool,
    /// Custom widgets configuration
    pub widgets: Vec<WidgetConfig>,
    /// Theme settings
    pub theme: DashboardTheme,
}

/// Widget configuration for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfig {
    /// Widget identifier
    pub id: String,
    /// Widget type
    pub widget_type: WidgetType,
    /// Widget position
    pub position: WidgetPosition,
    /// Widget size
    pub size: WidgetSize,
    /// Widget settings
    pub settings: HashMap<String, serde_json::Value>,
    /// Visibility
    pub visible: bool,
}

/// Dashboard theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTheme {
    /// Primary color
    pub primary_color: String,
    /// Secondary color
    pub secondary_color: String,
    /// Background color
    pub background_color: String,
    /// Text color
    pub text_color: String,
    /// Dark mode enabled
    pub dark_mode: bool,
}

/// Activity event in the team
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityEvent {
    /// Event identifier
    pub id: Uuid,
    /// Event type
    pub event_type: ActivityType,
    /// Actor (team member who performed the action)
    pub actor: Uuid,
    /// Target (what was affected)
    pub target: ActivityTarget,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event description
    pub description: String,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Team metrics for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamMetrics {
    /// Active members count
    pub active_members: usize,
    /// Reports shared today
    pub reports_today: usize,
    /// Annotations added today
    pub annotations_today: usize,
    /// Comments posted today
    pub comments_today: usize,
    /// Average response time (in minutes)
    /// Mean time between a request and its response, in minutes.
    ///
    /// Always `None`: see `TeamDashboard::calculate_avg_response_time` --
    /// nothing here records the paired events such an average needs. It used
    /// to be the constant `15.0`.
    pub avg_response_time: Option<f64>,
    /// Collaboration score
    pub collaboration_score: f64,
    /// Top contributors
    pub top_contributors: Vec<ContributorMetric>,
    /// Activity trends
    pub activity_trends: ActivityTrends,
}

/// Individual contributor metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributorMetric {
    /// Team member ID
    pub member_id: Uuid,
    /// Member name
    pub name: String,
    /// Reports contributed
    pub reports_count: usize,
    /// Annotations made
    pub annotations_count: usize,
    /// Comments posted
    pub comments_count: usize,
    /// Activity score
    pub activity_score: f64,
}

/// Activity trends over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityTrends {
    /// Daily activity counts for the last 7 days
    pub daily_activity: Vec<DailyActivity>,
    /// Weekly activity summary
    pub weekly_summary: WeeklyActivity,
    /// Percentage change in total recorded events between the last seven days
    /// and the seven days before that: `(recent - prior) / prior * 100`.
    ///
    /// `None` when the prior window holds no events at all, because the ratio
    /// is then undefined -- previously this was hardcoded to `0.0`, which reads
    /// as "measured zero growth".
    pub growth_rate: Option<f64>,
}

/// Daily activity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyActivity {
    /// Date
    pub date: DateTime<Utc>,
    /// Number of reports
    pub reports: usize,
    /// Number of annotations
    pub annotations: usize,
    /// Number of comments
    pub comments: usize,
    /// Active users
    pub active_users: usize,
}

/// Weekly activity summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeeklyActivity {
    /// Total reports this week
    pub total_reports: usize,
    /// Total annotations this week
    pub total_annotations: usize,
    /// Total comments this week
    pub total_comments: usize,
    /// Weekday name (`"Monday"` .. `"Sunday"`) of the busiest of the last seven
    /// days, by total recorded events.
    ///
    /// `None` when no activity was recorded in the window -- there is no peak
    /// day to name. This was previously the literal string `"Monday"`,
    /// regardless of the data.
    pub peak_day: Option<String>,
    /// Average daily active users
    pub avg_daily_active_users: f64,
}

/// Session activity tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionActivity {
    /// Session ID
    pub session_id: Uuid,
    /// Session type
    pub session_type: String,
    /// Participants
    pub participants: Vec<Uuid>,
    /// Start time
    pub started_at: DateTime<Utc>,
    /// Last activity time
    pub last_activity: DateTime<Utc>,
    /// Activity count
    pub activity_count: usize,
    /// Current status
    pub status: String,
}

/// Notification system for real-time updates
#[derive(Debug, Clone)]
pub struct NotificationSystem {
    /// Pending notifications
    notifications: Vec<DashboardNotification>,
    /// Notification settings
    settings: NotificationSettings,
}

/// Dashboard notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardNotification {
    /// Notification ID
    pub id: Uuid,
    /// Notification type
    pub notification_type: NotificationType,
    /// Title
    pub title: String,
    /// Message
    pub message: String,
    /// Priority level
    pub priority: NotificationPriority,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Target users
    pub target_users: Vec<Uuid>,
    /// Read status
    pub read_by: Vec<Uuid>,
    /// Action buttons
    pub actions: Vec<NotificationAction>,
}

/// Notification settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    /// Enable browser notifications
    pub browser_notifications: bool,
    /// Enable sound alerts
    pub sound_alerts: bool,
    /// Auto-dismiss time (seconds)
    pub auto_dismiss_time: u64,
    /// Max notifications to show
    pub max_notifications: usize,
}

/// Activity types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityType {
    ReportShared,
    ReportUpdated,
    AnnotationAdded,
    CommentPosted,
    SessionStarted,
    SessionEnded,
    MemberJoined,
    MemberLeft,
    IssueResolved,
    Custom(String),
}

/// Activity targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityTarget {
    Report(Uuid),
    Annotation(Uuid),
    Comment(Uuid),
    Session(Uuid),
    Member(Uuid),
    Custom(String),
}

/// Widget types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    ActivityFeed,
    TeamMetrics,
    ActiveSessions,
    RecentReports,
    TopContributors,
    ActivityChart,
    NotificationCenter,
    QuickActions,
    Custom(String),
}

/// Widget position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetPosition {
    /// X coordinate
    pub x: u32,
    /// Y coordinate
    pub y: u32,
    /// Grid column
    pub col: u32,
    /// Grid row
    pub row: u32,
}

/// Widget size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetSize {
    /// Width in grid units
    pub width: u32,
    /// Height in grid units
    pub height: u32,
    /// Minimum width
    pub min_width: u32,
    /// Minimum height
    pub min_height: u32,
}

/// Notification types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    NewReport,
    NewAnnotation,
    NewComment,
    SessionInvite,
    IssueAssigned,
    SystemAlert,
    Custom(String),
}

/// Notification priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationPriority {
    Low,
    Normal,
    High,
    Urgent,
}

/// Notification action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationAction {
    /// Action label
    pub label: String,
    /// Action type
    pub action_type: String,
    /// Action data
    pub data: HashMap<String, serde_json::Value>,
}

impl TeamDashboard {
    /// Create a new team dashboard
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            activity_feed: Vec::new(),
            metrics: TeamMetrics::default(),
            active_sessions: HashMap::new(),
            notifications: NotificationSystem::new(),
        }
    }

    /// Update dashboard with collaboration data
    pub fn update_from_collaboration(
        &mut self,
        collaboration: &CollaborationManager,
    ) -> Result<()> {
        // Update metrics
        self.update_metrics(collaboration)?;

        // Update activity feed
        self.update_activity_feed(collaboration)?;

        // Update active sessions
        self.update_active_sessions(collaboration)?;

        Ok(())
    }

    /// Add activity event
    pub fn add_activity_event(
        &mut self,
        event_type: ActivityType,
        actor: Uuid,
        target: ActivityTarget,
        description: String,
    ) -> Uuid {
        let event_id = Uuid::new_v4();
        let event = ActivityEvent {
            id: event_id,
            event_type,
            actor,
            target,
            timestamp: Utc::now(),
            description,
            metadata: HashMap::new(),
        };

        self.activity_feed.insert(0, event);

        // Limit activity feed size
        if self.activity_feed.len() > self.config.max_activities {
            self.activity_feed.truncate(self.config.max_activities);
        }

        event_id
    }

    /// Get dashboard data for rendering
    pub fn get_dashboard_data(&self) -> DashboardData {
        DashboardData {
            config: self.config.clone(),
            activity_feed: self.activity_feed.clone(),
            metrics: self.metrics.clone(),
            active_sessions: self.active_sessions.values().cloned().collect(),
            notifications: self.notifications.get_unread_notifications(),
            last_updated: Utc::now(),
        }
    }

    /// Send notification to team members
    pub fn send_notification(
        &mut self,
        notification_type: NotificationType,
        title: String,
        message: String,
        priority: NotificationPriority,
        target_users: Vec<Uuid>,
    ) -> Uuid {
        self.notifications.send_notification(
            notification_type,
            title,
            message,
            priority,
            target_users,
        )
    }

    /// Mark notification as read
    pub fn mark_notification_read(&mut self, notification_id: Uuid, user_id: Uuid) -> Result<()> {
        self.notifications.mark_read(notification_id, user_id)
    }

    /// Get team activity summary
    pub fn get_activity_summary(&self, days: u32) -> ActivitySummary {
        let cutoff_date = Utc::now() - Duration::days(days as i64);

        let recent_activities: Vec<_> = self
            .activity_feed
            .iter()
            .filter(|activity| activity.timestamp >= cutoff_date)
            .collect();

        let total_activities = recent_activities.len();
        let unique_contributors: std::collections::HashSet<_> =
            recent_activities.iter().map(|a| a.actor).collect();

        ActivitySummary {
            total_activities,
            unique_contributors: unique_contributors.len(),
            activity_by_type: self.count_activities_by_type(&recent_activities),
            most_active_day: self.find_most_active_day(&recent_activities),
            period_days: days,
        }
    }

    /// Update team metrics
    fn update_metrics(&mut self, collaboration: &CollaborationManager) -> Result<()> {
        let stats = collaboration.get_collaboration_stats();

        // Calculate activity trends
        let trends = self.calculate_activity_trends();

        // Calculate collaboration score
        let collaboration_score = self.calculate_collaboration_score(&stats);

        self.metrics = TeamMetrics {
            active_members: stats.team_size,
            reports_today: self.count_today_activities(ActivityType::ReportShared),
            annotations_today: self.count_today_activities(ActivityType::AnnotationAdded),
            comments_today: self.count_today_activities(ActivityType::CommentPosted),
            avg_response_time: self.calculate_avg_response_time(),
            collaboration_score,
            top_contributors: self.calculate_top_contributors(),
            activity_trends: trends,
        };

        Ok(())
    }

    /// No-op: the activity feed is already the authoritative record.
    ///
    /// Entries reach it through [`Self::record_activity`], which every
    /// dashboard mutation calls directly; [`CollaborationManager`] exposes only
    /// aggregate [`crate::collaboration::CollaborationStats`], not an event log
    /// that could be replayed here. Kept as an explicit no-op (rather than
    /// deleted) because [`Self::update_metrics`] documents a fixed refresh
    /// sequence, and silently dropping a step from it would be more confusing
    /// than a named, empty one.
    fn update_activity_feed(&mut self, _collaboration: &CollaborationManager) -> Result<()> {
        Ok(())
    }

    /// Update active sessions tracking
    fn update_active_sessions(&mut self, _collaboration: &CollaborationManager) -> Result<()> {
        // Update session activity tracking
        let now = Utc::now();

        // Remove stale sessions (inactive for more than 1 hour)
        let stale_cutoff = now - Duration::hours(1);
        self.active_sessions.retain(|_, session| session.last_activity >= stale_cutoff);

        Ok(())
    }

    /// Helper methods
    fn calculate_activity_trends(&self) -> ActivityTrends {
        let now = Utc::now();
        let mut daily_activity = Vec::new();

        // Calculate last 7 days
        for i in 0..7 {
            let date = now - Duration::days(i);
            let day_start = date.date_naive().and_time(chrono::NaiveTime::MIN).and_utc();
            let day_end = day_start + Duration::days(1);

            let day_activities: Vec<_> = self
                .activity_feed
                .iter()
                .filter(|a| a.timestamp >= day_start && a.timestamp < day_end)
                .collect();

            let daily = DailyActivity {
                date: day_start,
                reports: day_activities
                    .iter()
                    .filter(|a| matches!(a.event_type, ActivityType::ReportShared))
                    .count(),
                annotations: day_activities
                    .iter()
                    .filter(|a| matches!(a.event_type, ActivityType::AnnotationAdded))
                    .count(),
                comments: day_activities
                    .iter()
                    .filter(|a| matches!(a.event_type, ActivityType::CommentPosted))
                    .count(),
                active_users: day_activities
                    .iter()
                    .map(|a| a.actor)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
            };

            daily_activity.push(daily);
        }

        // Calculate weekly summary
        let total_reports = daily_activity.iter().map(|d| d.reports).sum();
        let total_annotations = daily_activity.iter().map(|d| d.annotations).sum();
        let total_comments = daily_activity.iter().map(|d| d.comments).sum();
        let avg_daily_active_users =
            daily_activity.iter().map(|d| d.active_users as f64).sum::<f64>() / 7.0;

        // Real peak day: the day in the window with the most recorded events.
        // `max_by_key` keeps the LAST maximum, so scan explicitly to keep the
        // earliest (oldest) day on a tie, which is the deterministic choice.
        let peak_day = daily_activity
            .iter()
            .map(|d| (d, d.reports + d.annotations + d.comments))
            .filter(|(_, total)| *total > 0)
            .fold(None::<(&DailyActivity, usize)>, |best, cur| match best {
                Some((_, best_total)) if best_total >= cur.1 => best,
                _ => Some(cur),
            })
            .map(|(day, _)| day.date.format("%A").to_string());

        let weekly_summary = WeeklyActivity {
            total_reports,
            total_annotations,
            total_comments,
            peak_day,
            avg_daily_active_users,
        };

        // Real growth rate: this week's event count against the previous
        // week's, both counted from the same activity feed.
        let week_ago = now - Duration::days(7);
        let two_weeks_ago = now - Duration::days(14);
        let recent = self.activity_feed.iter().filter(|a| a.timestamp >= week_ago).count();
        let prior = self
            .activity_feed
            .iter()
            .filter(|a| a.timestamp >= two_weeks_ago && a.timestamp < week_ago)
            .count();
        let growth_rate = if prior == 0 {
            None
        } else {
            Some((recent as f64 - prior as f64) / prior as f64 * 100.0)
        };

        ActivityTrends {
            daily_activity,
            weekly_summary,
            growth_rate,
        }
    }

    /// Composite collaboration score in `[0, 100]`.
    ///
    /// Defined here, from the real `stats`, as the mean of three bounded
    /// sub-scores, each saturating at a documented target:
    ///
    /// * **participation** — `contributors / team_size`, the share of the team
    ///   that appears in the activity feed at all;
    /// * **discussion** — `(comments + annotations) / reports` against a target
    ///   of 3 responses per shared report;
    /// * **breadth** — `reports_per_member` against a target of 5.
    ///
    /// It is a *definition*, not a measurement of an external quantity, and the
    /// weights are stated so a caller can reproduce it. The previous version was
    /// `50.0 + activity_feed.len() * 0.1`: a magic baseline of 50 that ignored
    /// `stats` entirely and grew without bound past 100 once the feed exceeded
    /// 500 entries.
    fn calculate_collaboration_score(
        &self,
        stats: &crate::collaboration::CollaborationStats,
    ) -> f64 {
        /// Responses (comments + annotations) per shared report that counts as
        /// a full discussion score.
        const DISCUSSION_TARGET: f64 = 3.0;
        /// Reports per member that counts as a full breadth score.
        const BREADTH_TARGET: f64 = 5.0;

        let contributors = self
            .activity_feed
            .iter()
            .map(|a| a.actor)
            .collect::<std::collections::HashSet<_>>()
            .len();
        let participation = if stats.team_size == 0 {
            0.0
        } else {
            (contributors as f64 / stats.team_size as f64).min(1.0)
        };

        let responses = (stats.total_comments + stats.total_annotations) as f64;
        let discussion = if stats.total_reports == 0 {
            0.0
        } else {
            (responses / stats.total_reports as f64 / DISCUSSION_TARGET).min(1.0)
        };

        let breadth = (stats.reports_per_member / BREADTH_TARGET).clamp(0.0, 1.0);

        (participation + discussion + breadth) / 3.0 * 100.0
    }

    fn calculate_top_contributors(&self) -> Vec<ContributorMetric> {
        let mut contributor_map: HashMap<Uuid, (usize, usize, usize)> = HashMap::new();

        for activity in &self.activity_feed {
            let entry = contributor_map.entry(activity.actor).or_insert((0, 0, 0));
            match activity.event_type {
                ActivityType::ReportShared => entry.0 += 1,
                ActivityType::AnnotationAdded => entry.1 += 1,
                ActivityType::CommentPosted => entry.2 += 1,
                _ => {},
            }
        }

        contributor_map
            .into_iter()
            .map(|(member_id, (reports, annotations, comments))| {
                let activity_score =
                    reports as f64 * 3.0 + annotations as f64 * 2.0 + comments as f64;
                ContributorMetric {
                    member_id,
                    name: format!("User {}", member_id.to_string()[0..8].to_uppercase()),
                    reports_count: reports,
                    annotations_count: annotations,
                    comments_count: comments,
                    activity_score,
                }
            })
            .collect()
    }

    fn count_today_activities(&self, activity_type: ActivityType) -> usize {
        let today = Utc::now().date_naive();
        self.activity_feed
            .iter()
            .filter(|a| {
                a.timestamp.date_naive() == today
                    && std::mem::discriminant(&a.event_type)
                        == std::mem::discriminant(&activity_type)
            })
            .count()
    }

    /// Always `None`: computing an average response time needs paired
    /// request/response events, and this dashboard records neither. It used to
    /// report a flat `15.0` minutes for every team, every week.
    fn calculate_avg_response_time(&self) -> Option<f64> {
        None
    }

    fn count_activities_by_type(&self, activities: &[&ActivityEvent]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for activity in activities {
            let type_name = match &activity.event_type {
                ActivityType::ReportShared => "ReportShared",
                ActivityType::AnnotationAdded => "AnnotationAdded",
                ActivityType::CommentPosted => "CommentPosted",
                ActivityType::SessionStarted => "SessionStarted",
                ActivityType::Custom(name) => name,
                _ => "Other",
            };
            *counts.entry(type_name.to_string()).or_insert(0) += 1;
        }
        counts
    }

    fn find_most_active_day(&self, activities: &[&ActivityEvent]) -> String {
        let mut day_counts = HashMap::new();
        for activity in activities {
            let day = activity.timestamp.format("%A").to_string();
            *day_counts.entry(day).or_insert(0) += 1;
        }

        day_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(day, _)| day)
            .unwrap_or_else(|| "Monday".to_string())
    }
}

/// Complete dashboard data for rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub config: DashboardConfig,
    pub activity_feed: Vec<ActivityEvent>,
    pub metrics: TeamMetrics,
    pub active_sessions: Vec<SessionActivity>,
    pub notifications: Vec<DashboardNotification>,
    pub last_updated: DateTime<Utc>,
}

/// Activity summary for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivitySummary {
    pub total_activities: usize,
    pub unique_contributors: usize,
    pub activity_by_type: HashMap<String, usize>,
    pub most_active_day: String,
    pub period_days: u32,
}

impl Default for NotificationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl NotificationSystem {
    pub fn new() -> Self {
        Self {
            notifications: Vec::new(),
            settings: NotificationSettings {
                browser_notifications: true,
                sound_alerts: false,
                auto_dismiss_time: 5,
                max_notifications: 50,
            },
        }
    }

    pub fn send_notification(
        &mut self,
        notification_type: NotificationType,
        title: String,
        message: String,
        priority: NotificationPriority,
        target_users: Vec<Uuid>,
    ) -> Uuid {
        let notification_id = Uuid::new_v4();
        let notification = DashboardNotification {
            id: notification_id,
            notification_type,
            title,
            message,
            priority,
            timestamp: Utc::now(),
            target_users,
            read_by: Vec::new(),
            actions: Vec::new(),
        };

        self.notifications.insert(0, notification);

        // Limit notifications
        if self.notifications.len() > self.settings.max_notifications {
            self.notifications.truncate(self.settings.max_notifications);
        }

        notification_id
    }

    pub fn mark_read(&mut self, notification_id: Uuid, user_id: Uuid) -> Result<()> {
        if let Some(notification) = self.notifications.iter_mut().find(|n| n.id == notification_id)
        {
            if !notification.read_by.contains(&user_id) {
                notification.read_by.push(user_id);
            }
            Ok(())
        } else {
            Err(anyhow::anyhow!("Notification not found"))
        }
    }

    pub fn get_unread_notifications(&self) -> Vec<DashboardNotification> {
        self.notifications.clone()
    }
}

impl Default for TeamMetrics {
    fn default() -> Self {
        Self {
            active_members: 0,
            reports_today: 0,
            annotations_today: 0,
            comments_today: 0,
            avg_response_time: None,
            collaboration_score: 0.0,
            top_contributors: Vec::new(),
            activity_trends: ActivityTrends {
                daily_activity: Vec::new(),
                weekly_summary: WeeklyActivity {
                    total_reports: 0,
                    total_annotations: 0,
                    total_comments: 0,
                    // An empty dashboard has no peak day and no baseline
                    // week to grow from.
                    peak_day: None,
                    avg_daily_active_users: 0.0,
                },
                growth_rate: None,
            },
        }
    }
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            refresh_interval: 30,
            max_activities: 100,
            real_time_updates: true,
            show_detailed_metrics: true,
            widgets: vec![
                WidgetConfig {
                    id: "activity-feed".to_string(),
                    widget_type: WidgetType::ActivityFeed,
                    position: WidgetPosition {
                        x: 0,
                        y: 0,
                        col: 0,
                        row: 0,
                    },
                    size: WidgetSize {
                        width: 6,
                        height: 8,
                        min_width: 4,
                        min_height: 6,
                    },
                    settings: HashMap::new(),
                    visible: true,
                },
                WidgetConfig {
                    id: "team-metrics".to_string(),
                    widget_type: WidgetType::TeamMetrics,
                    position: WidgetPosition {
                        x: 6,
                        y: 0,
                        col: 6,
                        row: 0,
                    },
                    size: WidgetSize {
                        width: 6,
                        height: 4,
                        min_width: 4,
                        min_height: 3,
                    },
                    settings: HashMap::new(),
                    visible: true,
                },
            ],
            theme: DashboardTheme {
                primary_color: "#007acc".to_string(),
                secondary_color: "#6c757d".to_string(),
                background_color: "#ffffff".to_string(),
                text_color: "#333333".to_string(),
                dark_mode: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Push an event with an explicit timestamp (the public API always stamps
    /// `Utc::now()`), so trend windows can be exercised deterministically.
    fn push_event_at(
        dashboard: &mut TeamDashboard,
        event_type: ActivityType,
        actor: Uuid,
        timestamp: DateTime<Utc>,
    ) {
        dashboard.activity_feed.push(ActivityEvent {
            id: Uuid::new_v4(),
            event_type,
            actor,
            target: ActivityTarget::Report(Uuid::new_v4()),
            timestamp,
            description: String::new(),
            metadata: HashMap::new(),
        });
    }

    #[test]
    fn peak_day_is_the_real_busiest_day_not_the_literal_monday() {
        let mut dashboard = TeamDashboard::new(DashboardConfig::default());
        let actor = Uuid::new_v4();
        let now = Utc::now();
        // Three events two days ago, one event today.
        let busy = now - Duration::days(2);
        for _ in 0..3 {
            push_event_at(&mut dashboard, ActivityType::ReportShared, actor, busy);
        }
        push_event_at(&mut dashboard, ActivityType::CommentPosted, actor, now);

        let trends = dashboard.calculate_activity_trends();
        let expected = busy.format("%A").to_string();
        assert_eq!(
            trends.weekly_summary.peak_day,
            Some(expected.clone()),
            "peak day must be the real argmax ({expected}), not a constant"
        );
    }

    #[test]
    fn peak_day_is_absent_when_nothing_happened() {
        let dashboard = TeamDashboard::new(DashboardConfig::default());
        let trends = dashboard.calculate_activity_trends();
        assert_eq!(
            trends.weekly_summary.peak_day, None,
            "no activity => no peak day"
        );
        assert_eq!(
            trends.growth_rate, None,
            "no prior week => growth is undefined"
        );
    }

    #[test]
    fn growth_rate_compares_this_week_against_the_previous_week() {
        let mut dashboard = TeamDashboard::new(DashboardConfig::default());
        let actor = Uuid::new_v4();
        let now = Utc::now();
        // 2 events in the prior week, 3 in the current week => +50%.
        for _ in 0..2 {
            push_event_at(
                &mut dashboard,
                ActivityType::ReportShared,
                actor,
                now - Duration::days(10),
            );
        }
        for _ in 0..3 {
            push_event_at(
                &mut dashboard,
                ActivityType::ReportShared,
                actor,
                now - Duration::days(2),
            );
        }
        let trends = dashboard.calculate_activity_trends();
        let rate = trends.growth_rate.expect("a prior week exists, so growth is defined");
        assert!((rate - 50.0).abs() < 1e-9, "expected +50%, got {rate}");
    }

    #[test]
    fn growth_rate_can_be_negative() {
        let mut dashboard = TeamDashboard::new(DashboardConfig::default());
        let actor = Uuid::new_v4();
        let now = Utc::now();
        for _ in 0..4 {
            push_event_at(
                &mut dashboard,
                ActivityType::ReportShared,
                actor,
                now - Duration::days(9),
            );
        }
        push_event_at(
            &mut dashboard,
            ActivityType::ReportShared,
            actor,
            now - Duration::days(1),
        );
        let rate = dashboard.calculate_activity_trends().growth_rate.expect("prior week non-empty");
        assert!((rate + 75.0).abs() < 1e-9, "1 vs 4 is -75%, got {rate}");
    }

    #[test]
    fn collaboration_score_responds_to_stats_and_stays_bounded() {
        let mut dashboard = TeamDashboard::new(DashboardConfig::default());
        let empty = crate::collaboration::CollaborationStats {
            total_reports: 0,
            total_annotations: 0,
            total_comments: 0,
            active_sessions: 0,
            team_size: 0,
            reports_per_member: 0.0,
            annotations_per_report: 0.0,
        };
        // The old formula returned 50.0 here (a magic baseline) regardless of
        // the fact that nothing at all had happened.
        assert_eq!(dashboard.calculate_collaboration_score(&empty), 0.0);

        let actor = Uuid::new_v4();
        push_event_at(
            &mut dashboard,
            ActivityType::ReportShared,
            actor,
            Utc::now(),
        );
        let healthy = crate::collaboration::CollaborationStats {
            total_reports: 10,
            total_annotations: 20,
            total_comments: 20,
            active_sessions: 2,
            team_size: 1,
            reports_per_member: 10.0,
            annotations_per_report: 2.0,
        };
        let score = dashboard.calculate_collaboration_score(&healthy);
        assert!(
            (score - 100.0).abs() < 1e-9,
            "all three sub-scores saturate: {score}"
        );

        // The old formula grew past 100 once the feed exceeded 500 entries.
        for _ in 0..600 {
            push_event_at(
                &mut dashboard,
                ActivityType::CommentPosted,
                actor,
                Utc::now(),
            );
        }
        let score = dashboard.calculate_collaboration_score(&healthy);
        assert!(
            (0.0..=100.0).contains(&score),
            "score must stay bounded: {score}"
        );
    }

    #[test]
    fn test_dashboard_creation() {
        let config = DashboardConfig::default();
        let dashboard = TeamDashboard::new(config);

        assert_eq!(dashboard.activity_feed.len(), 0);
        assert_eq!(dashboard.metrics.active_members, 0);
    }

    #[test]
    fn test_activity_event_addition() {
        let config = DashboardConfig::default();
        let mut dashboard = TeamDashboard::new(config);

        let actor = Uuid::new_v4();
        let target = ActivityTarget::Report(Uuid::new_v4());

        let event_id = dashboard.add_activity_event(
            ActivityType::ReportShared,
            actor,
            target,
            "Shared debugging report".to_string(),
        );

        assert_eq!(dashboard.activity_feed.len(), 1);
        assert_eq!(dashboard.activity_feed[0].id, event_id);
    }

    #[test]
    fn test_notification_system() {
        let mut notification_system = NotificationSystem::new();

        let notification_id = notification_system.send_notification(
            NotificationType::NewReport,
            "New Report".to_string(),
            "A new debugging report has been shared".to_string(),
            NotificationPriority::Normal,
            vec![Uuid::new_v4()],
        );

        assert_eq!(notification_system.notifications.len(), 1);
        assert_eq!(notification_system.notifications[0].id, notification_id);
    }

    #[test]
    fn test_activity_summary() {
        let config = DashboardConfig::default();
        let mut dashboard = TeamDashboard::new(config);

        // Add some test activities
        for i in 0..5 {
            dashboard.add_activity_event(
                ActivityType::ReportShared,
                Uuid::new_v4(),
                ActivityTarget::Report(Uuid::new_v4()),
                format!("Test activity {}", i),
            );
        }

        let summary = dashboard.get_activity_summary(7);
        assert_eq!(summary.total_activities, 5);
        assert_eq!(summary.period_days, 7);
    }
}
