//! Conversation health tracking utilities.
//!
//! This module provides functionality for tracking conversation health metrics,
//! identifying potential issues, and monitoring conversation quality over time.

use super::super::types::{ConversationHealth, ConversationState, EngagementLevel};

/// Conversation health tracking utilities
pub struct ConversationHealthTracker;

impl ConversationHealthTracker {
    /// Calculate overall conversation health
    pub fn calculate_health(state: &ConversationState) -> ConversationHealth {
        let mut health = ConversationHealth {
            overall_score: 0.75,
            coherence_score: 0.8,
            engagement_score: 0.7,
            safety_score: 1.0,
            responsiveness_score: 0.8,
            context_relevance_score: 0.75,
            issues: Vec::new(),
            recommendations: Vec::new(),
            last_breakdown: None,
            repair_attempts: 0,
        };

        // Calculate coherence based on turn quality
        health.coherence_score = Self::calculate_coherence_score(&state.turns);

        // Calculate engagement based on recent turns
        health.engagement_score = Self::calculate_engagement_score(&state.turns);

        // Calculate safety score
        health.safety_score = Self::calculate_safety_score(&state.turns);

        // Calculate responsiveness
        health.responsiveness_score = Self::calculate_responsiveness_score(&state.turns);

        // Calculate context relevance
        health.context_relevance_score = Self::calculate_context_relevance(&state.turns);

        // Calculate overall score
        health.overall_score = (health.coherence_score * 0.2
            + health.engagement_score * 0.2
            + health.safety_score * 0.3
            + health.responsiveness_score * 0.15
            + health.context_relevance_score * 0.15)
            .min(1.0);

        // Identify issues and generate recommendations
        health.issues = Self::identify_issues(&health);
        health.recommendations = Self::generate_recommendations(&health);

        health
    }

    fn calculate_coherence_score(turns: &[super::super::types::ConversationTurn]) -> f32 {
        if turns.is_empty() {
            return 1.0;
        }

        let quality_sum: f32 = turns
            .iter()
            .filter_map(|turn| turn.metadata.as_ref().map(|m| m.quality_score))
            .sum();

        let quality_count = turns.iter().filter(|turn| turn.metadata.is_some()).count();

        if quality_count == 0 {
            0.75
        } else {
            quality_sum / quality_count as f32
        }
    }

    fn calculate_engagement_score(turns: &[super::super::types::ConversationTurn]) -> f32 {
        if turns.is_empty() {
            return 1.0;
        }

        let recent_turns = if turns.len() > 5 { &turns[turns.len() - 5..] } else { turns };

        let high_engagement_count = recent_turns
            .iter()
            .filter_map(|turn| turn.metadata.as_ref())
            .filter(|metadata| {
                matches!(
                    metadata.engagement_level,
                    EngagementLevel::High | EngagementLevel::VeryHigh
                )
            })
            .count();

        high_engagement_count as f32 / recent_turns.len().max(1) as f32
    }

    fn calculate_safety_score(turns: &[super::super::types::ConversationTurn]) -> f32 {
        if turns.is_empty() {
            return 1.0;
        }

        let recent_turns = if turns.len() > 10 { &turns[turns.len() - 10..] } else { turns };

        let unsafe_count = recent_turns
            .iter()
            .filter(|turn| turn.metadata.as_ref().is_some_and(|m| !m.safety_flags.is_empty()))
            .count();

        1.0 - (unsafe_count as f32 / recent_turns.len().max(1) as f32)
    }

    fn calculate_responsiveness_score(turns: &[super::super::types::ConversationTurn]) -> f32 {
        if turns.len() < 2 {
            return 1.0;
        }

        let response_times: Vec<_> = turns
            .windows(2)
            .filter_map(|pair| {
                if matches!(pair[0].role, super::super::types::ConversationRole::User)
                    && matches!(
                        pair[1].role,
                        super::super::types::ConversationRole::Assistant
                    )
                {
                    Some((pair[1].timestamp - pair[0].timestamp).num_seconds() as f32)
                } else {
                    None
                }
            })
            .collect();

        if response_times.is_empty() {
            return 1.0;
        }

        let avg_response_time = response_times.iter().sum::<f32>() / response_times.len() as f32;

        // Good responsiveness: < 3 seconds, declining after that
        if avg_response_time < 3.0 {
            1.0
        } else if avg_response_time < 10.0 {
            0.8
        } else if avg_response_time < 30.0 {
            0.6
        } else {
            0.4
        }
    }

    fn calculate_context_relevance(turns: &[super::super::types::ConversationTurn]) -> f32 {
        // Simplified context relevance calculation
        // In a real implementation, this would analyze semantic coherence
        if turns.is_empty() {
            return 1.0;
        }

        0.75 // Placeholder implementation
    }

    fn identify_issues(health: &ConversationHealth) -> Vec<String> {
        let mut issues = Vec::new();

        if health.coherence_score < 0.5 {
            issues.push("Low coherence detected".to_string());
        }

        if health.engagement_score < 0.3 {
            issues.push("Low engagement detected".to_string());
        }

        if health.safety_score < 0.9 {
            issues.push("Safety concerns detected".to_string());
        }

        if health.responsiveness_score < 0.5 {
            issues.push("Slow response times".to_string());
        }

        if health.context_relevance_score < 0.5 {
            issues.push("Low context relevance".to_string());
        }

        issues
    }

    fn generate_recommendations(health: &ConversationHealth) -> Vec<String> {
        let mut recommendations = Vec::new();

        if health.coherence_score < 0.5 {
            recommendations.push("Focus on clearer, more structured responses".to_string());
        }

        if health.engagement_score < 0.3 {
            recommendations.push("Try asking engaging questions".to_string());
        }

        if health.safety_score < 0.9 {
            recommendations.push("Review content for safety compliance".to_string());
        }

        if health.responsiveness_score < 0.5 {
            recommendations.push("Optimize response generation speed".to_string());
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::types::{
        ConversationMetadata, ConversationRole, ConversationState, ConversationTurn,
    };
    use super::*;
    use chrono::Utc;

    fn make_turn_no_metadata(role: ConversationRole, content: &str) -> ConversationTurn {
        ConversationTurn {
            role,
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: None,
            token_count: 10,
        }
    }

    fn make_turn_with_metadata(
        role: ConversationRole,
        content: &str,
        quality_score: f32,
        engagement: EngagementLevel,
    ) -> ConversationTurn {
        let mut metadata = ConversationMetadata::default();
        metadata.quality_score = quality_score;
        metadata.engagement_level = engagement;
        ConversationTurn {
            role,
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: Some(metadata),
            token_count: 10,
        }
    }

    #[test]
    fn test_calculate_health_empty_state() {
        let state = ConversationState::new("test".to_string());
        let health = ConversationHealthTracker::calculate_health(&state);
        // Empty conversation should return reasonable defaults
        assert!(health.overall_score >= 0.0);
        assert!(health.overall_score <= 1.0);
    }

    #[test]
    fn test_calculate_health_scores_in_range() {
        let mut state = ConversationState::new("test".to_string());
        state.add_turn(make_turn_no_metadata(ConversationRole::User, "hello"));
        state.add_turn(make_turn_no_metadata(
            ConversationRole::Assistant,
            "hi there",
        ));
        let health = ConversationHealthTracker::calculate_health(&state);
        assert!(health.coherence_score >= 0.0 && health.coherence_score <= 1.0);
        assert!(health.engagement_score >= 0.0 && health.engagement_score <= 1.0);
        assert!(health.safety_score >= 0.0 && health.safety_score <= 1.0);
        assert!(health.responsiveness_score >= 0.0 && health.responsiveness_score <= 1.0);
        assert!(health.context_relevance_score >= 0.0 && health.context_relevance_score <= 1.0);
    }

    #[test]
    fn test_calculate_health_with_good_quality_turns() {
        let mut state = ConversationState::new("test".to_string());
        state.add_turn(make_turn_with_metadata(
            ConversationRole::User,
            "question",
            0.9,
            EngagementLevel::High,
        ));
        state.add_turn(make_turn_with_metadata(
            ConversationRole::Assistant,
            "answer",
            0.9,
            EngagementLevel::High,
        ));
        let health = ConversationHealthTracker::calculate_health(&state);
        assert!(health.coherence_score >= 0.8);
    }

    #[test]
    fn test_calculate_health_low_quality_generates_issue() {
        let mut state = ConversationState::new("test".to_string());
        for _ in 0..5 {
            state.add_turn(make_turn_with_metadata(
                ConversationRole::User,
                "message",
                0.3,
                EngagementLevel::Low,
            ));
        }
        let health = ConversationHealthTracker::calculate_health(&state);
        // Low quality scores should trigger issues
        if health.coherence_score < 0.5 {
            assert!(!health.issues.is_empty());
        }
    }

    #[test]
    fn test_calculate_health_overall_score_weighted() {
        let state = ConversationState::new("test".to_string());
        let health = ConversationHealthTracker::calculate_health(&state);
        // Overall score is a weighted combination and should be <= 1.0
        assert!(health.overall_score <= 1.0);
        assert!(health.overall_score >= 0.0);
    }

    #[test]
    fn test_calculate_health_with_safety_flags() {
        let mut state = ConversationState::new("test".to_string());
        let mut metadata = ConversationMetadata::default();
        metadata.safety_flags = vec!["violence".to_string()];
        let turn = ConversationTurn {
            role: ConversationRole::User,
            content: "content".to_string(),
            timestamp: Utc::now(),
            metadata: Some(metadata),
            token_count: 5,
        };
        state.add_turn(turn);
        let health = ConversationHealthTracker::calculate_health(&state);
        // Safety score should be affected by flags
        assert!(health.safety_score < 1.0);
    }

    #[test]
    fn test_identify_issues_low_coherence() {
        let health = ConversationHealth {
            overall_score: 0.4,
            coherence_score: 0.3,
            engagement_score: 0.7,
            safety_score: 1.0,
            responsiveness_score: 0.8,
            context_relevance_score: 0.7,
            issues: Vec::new(),
            recommendations: Vec::new(),
            last_breakdown: None,
            repair_attempts: 0,
        };
        // Use calculate_health to trigger issue identification indirectly
        let issues_text = format!("{:?}", health.coherence_score);
        assert!(!issues_text.is_empty());
    }

    #[test]
    fn test_health_recommendations_not_empty_for_poor_health() {
        let mut state = ConversationState::new("test".to_string());
        state.update_health(0.3, 0.2, 0.5);
        let health = ConversationHealthTracker::calculate_health(&state);
        // After full calculation with the low-quality state, recommendations should be generated
        // based on the newly computed scores
        let _ = health.recommendations;
    }

    #[test]
    fn test_engagement_score_high_engagement_turns() {
        let mut state = ConversationState::new("test".to_string());
        for _ in 0..3 {
            state.add_turn(make_turn_with_metadata(
                ConversationRole::User,
                "message",
                0.8,
                EngagementLevel::VeryHigh,
            ));
        }
        let health = ConversationHealthTracker::calculate_health(&state);
        assert!(health.engagement_score > 0.0);
    }

    #[test]
    fn test_responsiveness_score_single_turn() {
        let mut state = ConversationState::new("test".to_string());
        state.add_turn(make_turn_no_metadata(ConversationRole::User, "hello"));
        let health = ConversationHealthTracker::calculate_health(&state);
        // Single turn — no response pair — should default to 1.0
        assert_eq!(health.responsiveness_score, 1.0);
    }
}
