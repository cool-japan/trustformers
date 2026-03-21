//! Security and attack detection for federated learning
//!
//! This module implements security monitoring, attack detection, and defense
//! mechanisms for federated learning systems, including Byzantine fault tolerance,
//! poisoning attack detection, and adversarial defense strategies.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::federated_learning_v2_backup::types::*;
use trustformers_core::{Result, CoreError, Tensor};

/// Security configuration for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Attack detection enabled
    pub attack_detection_enabled: bool,
    /// Byzantine fault tolerance enabled
    pub byzantine_fault_tolerance: bool,
    /// Maximum fraction of Byzantine participants
    pub max_byzantine_fraction: f64,
    /// Anomaly detection threshold
    pub anomaly_threshold: f64,
    /// Trust score threshold
    pub trust_threshold: f64,
    /// Reputation system enabled
    pub reputation_system: bool,
    /// Differential privacy for defense
    pub defensive_dp: bool,
    /// Model validation enabled
    pub model_validation: bool,
    /// Gradient clipping enabled
    pub gradient_clipping: bool,
    /// Outlier detection methods
    pub outlier_detection_methods: Vec<OutlierDetectionMethod>,
}

/// Outlier detection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutlierDetectionMethod {
    /// Statistical outlier detection
    Statistical,
    /// Clustering-based detection
    Clustering,
    /// Distance-based detection
    DistanceBased,
    /// Ensemble methods
    Ensemble,
    /// Machine learning-based detection
    MachineLearning,
}

/// Attack detector for federated learning security
#[derive(Debug)]
pub struct AttackDetector {
    config: SecurityConfig,
    participant_history: HashMap<String, ParticipantSecurityProfile>,
    detection_history: Vec<AttackDetectionEvent>,
    trust_scores: HashMap<String, f64>,
    baseline_statistics: BaselineStatistics,
}

/// Participant security profile
#[derive(Debug, Clone)]
pub struct ParticipantSecurityProfile {
    /// Participant ID
    pub participant_id: String,
    /// Historical update statistics
    pub update_statistics: UpdateStatistics,
    /// Anomaly scores over time
    pub anomaly_scores: Vec<f64>,
    /// Trust evolution
    pub trust_evolution: Vec<f64>,
    /// Detected attack types
    pub detected_attacks: Vec<AttackType>,
    /// Last update timestamp
    pub last_update: u64,
}

/// Update statistics for security analysis
#[derive(Debug, Clone)]
pub struct UpdateStatistics {
    /// Average gradient norm
    pub avg_gradient_norm: f64,
    /// Gradient norm variance
    pub gradient_norm_variance: f64,
    /// Update frequency
    pub update_frequency: f64,
    /// Average update size
    pub avg_update_size: usize,
    /// Model performance contributions
    pub performance_contributions: Vec<f64>,
}

/// Baseline statistics for anomaly detection
#[derive(Debug, Clone)]
pub struct BaselineStatistics {
    /// Normal gradient norm range
    pub normal_gradient_norm_range: (f64, f64),
    /// Normal update size range
    pub normal_update_size_range: (usize, usize),
    /// Normal performance range
    pub normal_performance_range: (f64, f64),
    /// Update frequency baseline
    pub baseline_update_frequency: f64,
}

/// Reputation system for participant trust management
#[derive(Debug)]
pub struct ReputationSystem {
    /// Participant reputations
    pub reputations: HashMap<String, Reputation>,
    /// Reputation decay factor
    pub decay_factor: f64,
    /// Reward factor for good behavior
    pub reward_factor: f64,
    /// Penalty factor for bad behavior
    pub penalty_factor: f64,
}

/// Reputation information
#[derive(Debug, Clone)]
pub struct Reputation {
    /// Current reputation score (0.0 to 1.0)
    pub score: f64,
    /// Number of positive interactions
    pub positive_interactions: u32,
    /// Number of negative interactions
    pub negative_interactions: u32,
    /// Last update timestamp
    pub last_update: u64,
    /// Reputation history
    pub history: Vec<(u64, f64)>, // (timestamp, score)
}

impl AttackDetector {
    /// Create a new attack detector
    pub fn new(config: &SecurityConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            participant_history: HashMap::new(),
            detection_history: Vec::new(),
            trust_scores: HashMap::new(),
            baseline_statistics: BaselineStatistics::default(),
        })
    }

    /// Analyze participant update for security threats
    pub fn analyze_update(&mut self, participant_id: &str, update: &Tensor) -> Result<()> {
        if !self.config.attack_detection_enabled {
            return Ok(());
        }

        // Get or create participant profile
        let mut profile = self.participant_history
            .get(participant_id)
            .cloned()
            .unwrap_or_else(|| ParticipantSecurityProfile::new(participant_id.to_string()));

        // Compute update statistics
        let update_stats = self.compute_update_statistics(update)?;

        // Detect anomalies
        let anomaly_score = self.detect_anomalies(&update_stats, &profile)?;
        profile.anomaly_scores.push(anomaly_score);

        // Update participant profile
        profile.update_statistics = update_stats;
        profile.last_update = self.get_current_timestamp();

        // Check for specific attack types
        self.check_model_poisoning(&profile, update)?;
        self.check_byzantine_behavior(&profile)?;
        self.check_gradient_inversion(&profile, update)?;

        // Update trust score
        let trust_score = self.compute_trust_score(&profile);
        self.trust_scores.insert(participant_id.to_string(), trust_score);
        profile.trust_evolution.push(trust_score);

        // Store updated profile
        self.participant_history.insert(participant_id.to_string(), profile);

        Ok(())
    }

    /// Compute statistics for an update
    fn compute_update_statistics(&self, update: &Tensor) -> Result<UpdateStatistics> {
        let data = update.data()?;

        // Compute gradient norm
        let gradient_norm: f64 = data.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();

        // Simplified statistics computation
        Ok(UpdateStatistics {
            avg_gradient_norm: gradient_norm,
            gradient_norm_variance: 0.0, // Simplified
            update_frequency: 1.0, // Simplified
            avg_update_size: data.len(),
            performance_contributions: vec![0.0], // Simplified
        })
    }

    /// Detect anomalies in update statistics
    fn detect_anomalies(&self, stats: &UpdateStatistics, profile: &ParticipantSecurityProfile) -> Result<f64> {
        let mut anomaly_score = 0.0;

        // Check gradient norm anomaly
        if stats.avg_gradient_norm > self.baseline_statistics.normal_gradient_norm_range.1 * 2.0 {
            anomaly_score += 0.5;
        }

        // Check update size anomaly
        if stats.avg_update_size > self.baseline_statistics.normal_update_size_range.1 * 2 {
            anomaly_score += 0.3;
        }

        // Check consistency with participant's history
        if !profile.anomaly_scores.is_empty() {
            let avg_historical_score: f64 = profile.anomaly_scores.iter().sum::<f64>() / profile.anomaly_scores.len() as f64;
            if stats.avg_gradient_norm > avg_historical_score * 3.0 {
                anomaly_score += 0.2;
            }
        }

        Ok(anomaly_score.min(1.0))
    }

    /// Check for model poisoning attacks
    fn check_model_poisoning(&mut self, profile: &ParticipantSecurityProfile, update: &Tensor) -> Result<()> {
        let data = update.data()?;
        let gradient_norm: f64 = data.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();

        // Simple heuristic: very large gradients might indicate poisoning
        if gradient_norm > self.config.anomaly_threshold * 10.0 {
            let event = AttackDetectionEvent {
                timestamp: self.get_current_timestamp(),
                participant_id: profile.participant_id.clone(),
                attack_type: AttackType::ModelPoisoning,
                confidence_score: 0.8,
                countermeasures: vec![Countermeasure::UpdateRejection],
                details: {
                    let mut details = HashMap::new();
                    details.insert("gradient_norm".to_string(), gradient_norm.to_string());
                    details
                },
            };
            self.detection_history.push(event);
        }

        Ok(())
    }

    /// Check for Byzantine behavior
    fn check_byzantine_behavior(&mut self, profile: &ParticipantSecurityProfile) -> Result<()> {
        // Check if participant shows consistent anomalous behavior
        if profile.anomaly_scores.len() >= 5 {
            let recent_scores: Vec<f64> = profile.anomaly_scores.iter().rev().take(5).copied().collect();
            let avg_recent_anomaly: f64 = recent_scores.iter().sum::<f64>() / recent_scores.len() as f64;

            if avg_recent_anomaly > self.config.anomaly_threshold {
                let event = AttackDetectionEvent {
                    timestamp: self.get_current_timestamp(),
                    participant_id: profile.participant_id.clone(),
                    attack_type: AttackType::Byzantine,
                    confidence_score: avg_recent_anomaly,
                    countermeasures: vec![Countermeasure::WeightReduction],
                    details: {
                        let mut details = HashMap::new();
                        details.insert("avg_anomaly_score".to_string(), avg_recent_anomaly.to_string());
                        details
                    },
                };
                self.detection_history.push(event);
            }
        }

        Ok(())
    }

    /// Check for gradient inversion attacks
    fn check_gradient_inversion(&mut self, profile: &ParticipantSecurityProfile, update: &Tensor) -> Result<()> {
        // Simplified gradient inversion detection
        let data = update.data()?;

        // Check for unusual patterns that might indicate gradient inversion
        let mut pattern_score = 0.0;

        // Check for repeated values (simplified heuristic)
        let unique_values: std::collections::HashSet<_> = data.iter().map(|&x| (x * 1000.0) as i32).collect();
        if unique_values.len() < data.len() / 10 {
            pattern_score += 0.5;
        }

        if pattern_score > 0.3 {
            let event = AttackDetectionEvent {
                timestamp: self.get_current_timestamp(),
                participant_id: profile.participant_id.clone(),
                attack_type: AttackType::GradientInversion,
                confidence_score: pattern_score,
                countermeasures: vec![Countermeasure::AdditionalNoise],
                details: {
                    let mut details = HashMap::new();
                    details.insert("pattern_score".to_string(), pattern_score.to_string());
                    details
                },
            };
            self.detection_history.push(event);
        }

        Ok(())
    }

    /// Compute trust score for a participant
    fn compute_trust_score(&self, profile: &ParticipantSecurityProfile) -> f64 {
        let mut trust_score = 1.0;

        // Reduce trust based on anomaly scores
        if !profile.anomaly_scores.is_empty() {
            let avg_anomaly: f64 = profile.anomaly_scores.iter().sum::<f64>() / profile.anomaly_scores.len() as f64;
            trust_score *= (1.0 - avg_anomaly).max(0.0);
        }

        // Reduce trust based on detected attacks
        let attack_penalty = profile.detected_attacks.len() as f64 * 0.1;
        trust_score *= (1.0 - attack_penalty).max(0.0);

        trust_score.max(0.0).min(1.0)
    }

    /// Get trust score for a participant
    pub fn get_trust_score(&self, participant_id: &str) -> f64 {
        self.trust_scores.get(participant_id).copied().unwrap_or(1.0)
    }

    /// Get detection history
    pub fn get_detection_history(&self) -> &[AttackDetectionEvent] {
        &self.detection_history
    }

    /// Update baseline statistics
    pub fn update_baseline_statistics(&mut self, updates: &[&Tensor]) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        let mut gradient_norms = Vec::new();
        let mut update_sizes = Vec::new();

        for update in updates {
            let data = update.data()?;
            let gradient_norm: f64 = data.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            gradient_norms.push(gradient_norm);
            update_sizes.push(data.len());
        }

        // Compute statistics
        let min_norm = gradient_norms.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_norm = gradient_norms.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_size = *update_sizes.iter().min().unwrap_or(&0);
        let max_size = *update_sizes.iter().max().unwrap_or(&0);

        self.baseline_statistics = BaselineStatistics {
            normal_gradient_norm_range: (min_norm, max_norm),
            normal_update_size_range: (min_size, max_size),
            normal_performance_range: (0.0, 1.0), // Simplified
            baseline_update_frequency: 1.0, // Simplified
        };

        Ok(())
    }

    /// Apply countermeasures
    pub fn apply_countermeasures(&mut self, participant_id: &str, countermeasures: &[Countermeasure]) -> Result<()> {
        for countermeasure in countermeasures {
            match countermeasure {
                Countermeasure::UpdateRejection => {
                    // Mark update as rejected
                    self.trust_scores.insert(participant_id.to_string(), 0.0);
                }
                Countermeasure::AdditionalNoise => {
                    // Request additional noise for privacy
                }
                Countermeasure::WeightReduction => {
                    // Reduce participant weight in aggregation
                    if let Some(trust_score) = self.trust_scores.get_mut(participant_id) {
                        *trust_score *= 0.5;
                    }
                }
                Countermeasure::TemporaryExclusion => {
                    // Temporarily exclude participant
                    self.trust_scores.insert(participant_id.to_string(), 0.0);
                }
                Countermeasure::PermanentBan => {
                    // Permanently ban participant
                    self.trust_scores.remove(participant_id);
                    self.participant_history.remove(participant_id);
                }
            }
        }
        Ok(())
    }

    /// Get current timestamp (simplified)
    fn get_current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

impl ParticipantSecurityProfile {
    /// Create a new participant security profile
    pub fn new(participant_id: String) -> Self {
        Self {
            participant_id,
            update_statistics: UpdateStatistics::default(),
            anomaly_scores: Vec::new(),
            trust_evolution: Vec::new(),
            detected_attacks: Vec::new(),
            last_update: 0,
        }
    }
}

impl ReputationSystem {
    /// Create a new reputation system
    pub fn new(decay_factor: f64, reward_factor: f64, penalty_factor: f64) -> Self {
        Self {
            reputations: HashMap::new(),
            decay_factor,
            reward_factor,
            penalty_factor,
        }
    }

    /// Update reputation for a participant
    pub fn update_reputation(&mut self, participant_id: &str, positive: bool) {
        let reputation = self.reputations
            .entry(participant_id.to_string())
            .or_insert_with(|| Reputation::new());

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Apply time decay
        let time_diff = current_time - reputation.last_update;
        let decay = (-self.decay_factor * time_diff as f64).exp();
        reputation.score *= decay;

        // Update based on interaction
        if positive {
            reputation.positive_interactions += 1;
            reputation.score = (reputation.score + self.reward_factor).min(1.0);
        } else {
            reputation.negative_interactions += 1;
            reputation.score = (reputation.score - self.penalty_factor).max(0.0);
        }

        reputation.last_update = current_time;
        reputation.history.push((current_time, reputation.score));

        // Keep only recent history
        if reputation.history.len() > 1000 {
            reputation.history.drain(0..500);
        }
    }

    /// Get reputation score for a participant
    pub fn get_reputation(&self, participant_id: &str) -> f64 {
        self.reputations.get(participant_id)
            .map(|rep| rep.score)
            .unwrap_or(0.5) // Default neutral reputation
    }

    /// Get all reputations
    pub fn get_all_reputations(&self) -> &HashMap<String, Reputation> {
        &self.reputations
    }
}

impl Reputation {
    /// Create a new reputation with default values
    pub fn new() -> Self {
        Self {
            score: 0.5, // Start with neutral reputation
            positive_interactions: 0,
            negative_interactions: 0,
            last_update: 0,
            history: Vec::new(),
        }
    }

    /// Get total interactions
    pub fn get_total_interactions(&self) -> u32 {
        self.positive_interactions + self.negative_interactions
    }

    /// Get positive interaction ratio
    pub fn get_positive_ratio(&self) -> f64 {
        let total = self.get_total_interactions();
        if total == 0 {
            0.5
        } else {
            self.positive_interactions as f64 / total as f64
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            attack_detection_enabled: true,
            byzantine_fault_tolerance: true,
            max_byzantine_fraction: 0.3,
            anomaly_threshold: 0.5,
            trust_threshold: 0.7,
            reputation_system: true,
            defensive_dp: true,
            model_validation: true,
            gradient_clipping: true,
            outlier_detection_methods: vec![
                OutlierDetectionMethod::Statistical,
                OutlierDetectionMethod::DistanceBased,
            ],
        }
    }
}

impl Default for UpdateStatistics {
    fn default() -> Self {
        Self {
            avg_gradient_norm: 0.0,
            gradient_norm_variance: 0.0,
            update_frequency: 1.0,
            avg_update_size: 0,
            performance_contributions: Vec::new(),
        }
    }
}

impl Default for BaselineStatistics {
    fn default() -> Self {
        Self {
            normal_gradient_norm_range: (0.0, 10.0),
            normal_update_size_range: (1000, 100000),
            normal_performance_range: (0.0, 1.0),
            baseline_update_frequency: 1.0,
        }
    }
}