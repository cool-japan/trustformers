//! Network monitoring and condition assessment.

use super::types::*;
use std::collections::VecDeque;
use std::time::Instant;
use trustformers_core::error::Result;

/// Network monitoring system
pub struct NetworkMonitor {
    config: NetworkAdaptationConfig,
    current_conditions: NetworkConditions,
    history: VecDeque<NetworkConditions>,
    quality_analyzer: NetworkQualityAnalyzer,
    last_update: Instant,
}

/// Network quality analyzer
pub struct NetworkQualityAnalyzer {
    thresholds: NetworkQualityThresholds,
    quality_history: VecDeque<NetworkQuality>,
    trend_analyzer: NetworkTrendAnalyzer,
}

/// Network trend analyzer for predictive optimization
pub struct NetworkTrendAnalyzer {
    bandwidth_trend: TrendDirection,
    latency_trend: TrendDirection,
    stability_trend: TrendDirection,
    prediction_confidence: f32,
}

impl NetworkMonitor {
    /// Create new network monitor
    pub fn new(config: NetworkAdaptationConfig) -> Result<Self> {
        Ok(Self {
            config,
            current_conditions: NetworkConditions::default(),
            history: VecDeque::new(),
            quality_analyzer: NetworkQualityAnalyzer::new(),
            last_update: Instant::now(),
        })
    }

    /// Start network monitoring
    pub fn start(&mut self) -> Result<()> {
        // Initialize monitoring subsystem
        // In a real implementation, this would start background monitoring threads
        Ok(())
    }

    /// Stop network monitoring
    pub fn stop(&mut self) -> Result<()> {
        // Stop monitoring subsystem
        Ok(())
    }

    /// Get current network conditions
    pub fn get_current_conditions(&self) -> NetworkConditions {
        self.current_conditions.clone()
    }

    /// Update network conditions with new measurement
    pub fn update_conditions(&mut self, conditions: NetworkConditions) -> Result<()> {
        // Update quality assessment
        let quality = self.quality_analyzer.assess_quality(&conditions);
        let mut updated_conditions = conditions;
        updated_conditions.quality_assessment = quality;

        // Add to history
        self.history.push_back(updated_conditions.clone());
        if self.history.len() > 100 {
            self.history.pop_front();
        }

        // Update current conditions
        self.current_conditions = updated_conditions;
        self.last_update = Instant::now();

        // Update quality analyzer
        self.quality_analyzer.update(quality);

        Ok(())
    }

    /// Get network quality history
    pub fn get_quality_history(&self) -> &VecDeque<NetworkConditions> {
        &self.history
    }

    /// Get network trend analysis
    pub fn get_trend_analysis(&self) -> &NetworkTrendAnalyzer {
        &self.quality_analyzer.trend_analyzer
    }

    /// Check if network meets requirements for task
    pub fn meets_requirements(&self, requirements: &super::types::NetworkRequirements) -> bool {
        let conditions = &self.current_conditions;

        conditions.bandwidth_mbps >= requirements.min_bandwidth_mbps
            && conditions.latency_ms <= requirements.max_latency_ms
            && requirements.preferred_connections.contains(&conditions.connection_type)
    }

    /// Get network stability score (0.0-1.0)
    pub fn get_stability_score(&self) -> f32 {
        self.current_conditions.stability_score
    }

    /// Update monitoring configuration
    pub fn update_config(&mut self, config: NetworkAdaptationConfig) -> Result<()> {
        self.config = config.clone();
        self.quality_analyzer.update_thresholds(config.quality_thresholds);
        Ok(())
    }

    /// Check if network is suitable for federated learning tasks
    pub fn is_suitable_for_federated_learning(&self) -> bool {
        matches!(
            self.current_conditions.quality_assessment,
            NetworkQuality::Good | NetworkQuality::Excellent
        )
    }

    /// Get time since last update
    pub fn time_since_last_update(&self) -> std::time::Duration {
        self.last_update.elapsed()
    }

    /// Predict network conditions for the next interval
    pub fn predict_conditions(&self, duration: std::time::Duration) -> NetworkConditions {
        // Simple prediction based on current trends
        let mut predicted = self.current_conditions.clone();
        predicted.timestamp = Instant::now() + duration;

        // Apply trend-based adjustments
        match self.quality_analyzer.trend_analyzer.bandwidth_trend {
            TrendDirection::Improving => predicted.bandwidth_mbps *= 1.1,
            TrendDirection::Degrading => predicted.bandwidth_mbps *= 0.9,
            _ => {},
        }

        match self.quality_analyzer.trend_analyzer.latency_trend {
            TrendDirection::Improving => predicted.latency_ms *= 0.9,
            TrendDirection::Degrading => predicted.latency_ms *= 1.1,
            _ => {},
        }

        predicted
    }
}

impl NetworkQualityAnalyzer {
    /// Create new quality analyzer
    pub fn new() -> Self {
        Self {
            thresholds: NetworkQualityThresholds::default(),
            quality_history: VecDeque::new(),
            trend_analyzer: NetworkTrendAnalyzer::new(),
        }
    }

    /// Assess network quality based on conditions
    pub fn assess_quality(&self, conditions: &NetworkConditions) -> NetworkQuality {
        let mut score = 0;

        // Bandwidth assessment
        if conditions.bandwidth_mbps >= self.thresholds.min_bandwidth_full_sync_mbps {
            score += 3;
        } else if conditions.bandwidth_mbps >= self.thresholds.min_bandwidth_incremental_sync_mbps {
            score += 2;
        } else {
            score += 1;
        }

        // Latency assessment
        if conditions.latency_ms <= self.thresholds.max_latency_realtime_ms {
            score += 3;
        } else if conditions.latency_ms <= self.thresholds.max_latency_realtime_ms * 2.0 {
            score += 2;
        } else {
            score += 1;
        }

        // Packet loss assessment
        if conditions.packet_loss_percent <= self.thresholds.max_packet_loss_percent {
            score += 3;
        } else if conditions.packet_loss_percent <= self.thresholds.max_packet_loss_percent * 2.0 {
            score += 2;
        } else {
            score += 1;
        }

        // Jitter assessment
        if conditions.jitter_ms <= self.thresholds.max_jitter_ms {
            score += 3;
        } else if conditions.jitter_ms <= self.thresholds.max_jitter_ms * 2.0 {
            score += 2;
        } else {
            score += 1;
        }

        // Convert score to quality
        match score {
            10..=12 => NetworkQuality::Excellent,
            7..=9 => NetworkQuality::Good,
            5..=6 => NetworkQuality::Fair,
            _ => NetworkQuality::Poor,
        }
    }

    /// Update quality history and trends
    pub fn update(&mut self, quality: NetworkQuality) {
        self.quality_history.push_back(quality);
        if self.quality_history.len() > 50 {
            self.quality_history.pop_front();
        }

        // Update trend analysis
        self.trend_analyzer.update_trends(&self.quality_history);
    }

    /// Update quality thresholds
    pub fn update_thresholds(&mut self, thresholds: NetworkQualityThresholds) {
        self.thresholds = thresholds;
    }

    /// Get average quality over recent history
    pub fn get_average_quality(&self) -> NetworkQuality {
        if self.quality_history.is_empty() {
            return NetworkQuality::Fair;
        }

        let sum: u32 = self
            .quality_history
            .iter()
            .map(|q| match q {
                NetworkQuality::Poor => 1,
                NetworkQuality::Fair => 2,
                NetworkQuality::Good => 3,
                NetworkQuality::Excellent => 4,
            })
            .sum();

        let avg = sum as f32 / self.quality_history.len() as f32;

        match avg as u32 {
            4 => NetworkQuality::Excellent,
            3 => NetworkQuality::Good,
            2 => NetworkQuality::Fair,
            _ => NetworkQuality::Poor,
        }
    }

    /// Check if network quality is stable
    pub fn is_quality_stable(&self) -> bool {
        if self.quality_history.len() < 10 {
            return false;
        }

        let recent: Vec<_> = self.quality_history.iter().rev().take(10).collect();
        let first_quality = match recent.last() {
            Some(q) => q,
            None => return false,
        };

        recent.iter().all(|&q| q == *first_quality)
    }
}

impl NetworkTrendAnalyzer {
    /// Create new trend analyzer
    pub fn new() -> Self {
        Self {
            bandwidth_trend: TrendDirection::Stable,
            latency_trend: TrendDirection::Stable,
            stability_trend: TrendDirection::Stable,
            prediction_confidence: 0.5,
        }
    }

    /// Update trend analysis based on quality history
    pub fn update_trends(&mut self, quality_history: &VecDeque<NetworkQuality>) {
        if quality_history.len() < 5 {
            return;
        }

        let recent: Vec<_> = quality_history.iter().rev().take(5).collect();
        let quality_values: Vec<u32> = recent
            .iter()
            .map(|&q| match q {
                NetworkQuality::Poor => 1,
                NetworkQuality::Fair => 2,
                NetworkQuality::Good => 3,
                NetworkQuality::Excellent => 4,
            })
            .collect();

        // Simple trend detection
        let trend = self.detect_trend(&quality_values);
        self.bandwidth_trend = trend;
        self.latency_trend = trend;
        self.stability_trend = trend;

        // Update confidence based on consistency
        self.prediction_confidence = self.calculate_confidence(&quality_values);
    }

    /// Detect trend direction from values
    fn detect_trend(&self, values: &[u32]) -> TrendDirection {
        if values.len() < 3 {
            return TrendDirection::Stable;
        }

        let mut improving = 0;
        let mut degrading = 0;

        for window in values.windows(2) {
            if window[1] > window[0] {
                improving += 1;
            } else if window[1] < window[0] {
                degrading += 1;
            }
        }

        if improving > degrading * 2 {
            TrendDirection::Improving
        } else if degrading > improving * 2 {
            TrendDirection::Degrading
        } else if improving + degrading > values.len() / 2 {
            TrendDirection::Volatile
        } else {
            TrendDirection::Stable
        }
    }

    /// Calculate prediction confidence
    fn calculate_confidence(&self, values: &[u32]) -> f32 {
        if values.len() < 2 {
            return 0.5;
        }

        let variance = self.calculate_variance(values);

        // Lower variance = higher confidence
        (1.0 / (1.0 + variance)).clamp(0.0, 1.0)
    }

    /// Calculate variance of values
    fn calculate_variance(&self, values: &[u32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<u32>() as f32 / values.len() as f32;
        let variance = values
            .iter()
            .map(|&x| {
                let diff = x as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / values.len() as f32;

        variance
    }

    /// Get bandwidth trend
    pub fn get_bandwidth_trend(&self) -> TrendDirection {
        self.bandwidth_trend
    }

    /// Get latency trend
    pub fn get_latency_trend(&self) -> TrendDirection {
        self.latency_trend
    }

    /// Get stability trend
    pub fn get_stability_trend(&self) -> TrendDirection {
        self.stability_trend
    }

    /// Get prediction confidence
    pub fn get_prediction_confidence(&self) -> f32 {
        self.prediction_confidence
    }
}

impl Default for NetworkQualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for NetworkTrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
