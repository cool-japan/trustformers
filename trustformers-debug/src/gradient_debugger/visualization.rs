//! Gradient Flow Visualization Data Generation
//!
//! This module provides comprehensive visualization data generation for gradient flow
//! analysis, including network topology, temporal flows, and critical path identification.

use super::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete gradient flow visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlowVisualization {
    pub layer_flows: HashMap<String, GradientLayerFlow>,
    pub temporal_flows: Vec<TemporalGradientFlow>,
    pub flow_network: GradientFlowNetwork,
    pub critical_paths: Vec<CriticalGradientPath>,
    pub vanishing_regions: Vec<VanishingRegion>,
    pub exploding_regions: Vec<ExplodingRegion>,
    pub dead_zones: Vec<GradientDeadZone>,
    pub visualization_config: GradientVisualizationConfig,
}

/// Gradient flow data for a specific layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientLayerFlow {
    pub layer_name: String,
    pub gradient_magnitudes: Vec<f64>,
    pub gradient_directions: Vec<GradientDirection>,
    pub flow_consistency: f64,
    pub bottleneck_score: f64,
    pub information_flow_rate: f64,
}

/// Direction and characteristics of gradient flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientDirection {
    pub step: usize,
    pub direction_vector: Vec<f64>,
    pub magnitude: f64,
    pub consistency_score: f64,
}

/// Temporal gradient flow at a specific timestep
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalGradientFlow {
    pub step: usize,
    pub layer_name: String,
    pub gradient_magnitude: f64,
    pub flow_direction: FlowDirection,
    pub stability_score: f64,
}

/// Flow direction classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowDirection {
    Forward,
    Backward,
    Oscillating,
    Stagnant,
}

/// Network representation of gradient flows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlowNetwork {
    pub nodes: Vec<FlowNode>,
    pub edges: Vec<FlowEdge>,
    pub network_metrics: NetworkMetrics,
}

/// Node in the gradient flow network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowNode {
    pub layer_name: String,
    pub node_type: NodeType,
    pub gradient_strength: f64,
    pub connectivity: usize,
    pub influence_score: f64,
}

/// Type of node in the flow network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Source,
    Sink,
    Bottleneck,
    Amplifier,
    Normal,
}

/// Edge in the gradient flow network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowEdge {
    pub from_layer: String,
    pub to_layer: String,
    pub flow_strength: f64,
    pub flow_consistency: f64,
    pub edge_type: EdgeType,
}

/// Type of edge in the flow network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeType {
    Strong,
    Weak,
    Intermittent,
    Blocked,
}

/// Network-level gradient flow metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub overall_flow_efficiency: f64,
    pub network_connectivity: f64,
    pub bottleneck_density: f64,
    pub flow_stability: f64,
    pub information_propagation_speed: f64,
}

/// Critical path in gradient flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalGradientPath {
    pub path_id: String,
    pub layers: Vec<String>,
    pub path_length: usize,
    pub total_flow_strength: f64,
    pub bottleneck_layers: Vec<String>,
    pub criticality_score: f64,
    pub optimization_potential: f64,
}

/// Region where gradients are vanishing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VanishingRegion {
    pub region_id: String,
    pub affected_layers: Vec<String>,
    pub severity_level: VanishingSeverity,
    pub extent: RegionExtent,
    pub mitigation_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VanishingSeverity {
    Mild,
    Moderate,
    Severe,
    Critical,
}

/// Region where gradients are exploding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplodingRegion {
    pub region_id: String,
    pub affected_layers: Vec<String>,
    pub severity_level: ExplodingSeverity,
    pub extent: RegionExtent,
    pub mitigation_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplodingSeverity {
    Mild,
    Moderate,
    Severe,
    Critical,
}

/// Spatial extent of a gradient region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionExtent {
    pub start_layer: String,
    pub end_layer: String,
    pub affected_parameters: usize,
    pub duration_steps: usize,
}

/// Zone where gradient flow is effectively dead
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientDeadZone {
    pub zone_id: String,
    pub affected_layers: Vec<String>,
    pub dead_duration: usize,
    pub recovery_potential: RecoveryPotential,
    pub intervention_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryPotential {
    High,
    Medium,
    Low,
    None,
}

/// Configuration for gradient visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientVisualizationConfig {
    pub show_temporal_flows: bool,
    pub show_critical_paths: bool,
    pub show_problem_regions: bool,
    pub color_scheme: ColorScheme,
    pub temporal_window: usize,
    pub flow_threshold: f64,
}

impl Default for GradientVisualizationConfig {
    fn default() -> Self {
        Self {
            show_temporal_flows: true,
            show_critical_paths: true,
            show_problem_regions: true,
            color_scheme: ColorScheme::Default,
            temporal_window: 50,
            flow_threshold: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    Default,
    HighContrast,
    ColorBlind,
    Monochrome,
}

/// Gradient flow visualization generator
#[derive(Debug, Default)]
pub struct GradientFlowVisualizer {
    config: GradientVisualizationConfig,
}

impl GradientFlowVisualizer {
    pub fn new(config: GradientVisualizationConfig) -> Self {
        Self { config }
    }

    pub fn generate_visualization(
        &self,
        gradient_histories: &HashMap<String, GradientHistory>,
        current_step: usize,
    ) -> GradientFlowVisualization {
        let layer_flows = self.generate_layer_flows(gradient_histories);
        let temporal_flows = self.generate_temporal_flows(gradient_histories, current_step);
        let flow_network = self.build_gradient_flow_network(&layer_flows);
        let critical_paths = self.identify_critical_gradient_paths(&flow_network);
        let vanishing_regions = self.identify_vanishing_regions(gradient_histories);
        let exploding_regions = self.identify_exploding_regions(gradient_histories);
        let dead_zones = self.identify_gradient_dead_zones(gradient_histories);

        GradientFlowVisualization {
            layer_flows,
            temporal_flows,
            flow_network,
            critical_paths,
            vanishing_regions,
            exploding_regions,
            dead_zones,
            visualization_config: self.config.clone(),
        }
    }

    fn generate_layer_flows(
        &self,
        gradient_histories: &HashMap<String, GradientHistory>,
    ) -> HashMap<String, GradientLayerFlow> {
        let mut layer_flows = HashMap::new();

        for (layer_name, history) in gradient_histories {
            let gradient_magnitudes: Vec<f64> = history.gradient_norms.iter().cloned().collect();
            let gradient_directions = self.compute_gradient_directions(history);
            let flow_consistency = self.compute_flow_consistency(history);
            let bottleneck_score = self.compute_bottleneck_score(history);
            let information_flow_rate = self.compute_information_flow_rate(history);

            let flow_data = GradientLayerFlow {
                layer_name: layer_name.clone(),
                gradient_magnitudes,
                gradient_directions,
                flow_consistency,
                bottleneck_score,
                information_flow_rate,
            };

            layer_flows.insert(layer_name.clone(), flow_data);
        }

        layer_flows
    }

    fn compute_gradient_directions(&self, history: &GradientHistory) -> Vec<GradientDirection> {
        let mut directions = Vec::new();

        for (i, (&norm, &step)) in
            history.gradient_norms.iter().zip(history.step_numbers.iter()).enumerate()
        {
            // Simplified direction computation - in practice, this would use actual gradient vectors
            let direction_vector = vec![norm]; // Placeholder
            let magnitude = norm;
            let consistency_score = if i > 0 {
                let prev_norm = history.gradient_norms[i - 1];
                1.0 - ((norm - prev_norm).abs() / (norm + prev_norm + 1e-8))
            } else {
                1.0
            };

            directions.push(GradientDirection {
                step,
                direction_vector,
                magnitude,
                consistency_score,
            });
        }

        directions
    }

    fn compute_flow_consistency(&self, history: &GradientHistory) -> f64 {
        if history.gradient_norms.len() < 2 {
            return 1.0;
        }

        let variations: Vec<f64> = history
            .gradient_norms
            .iter()
            .collect::<Vec<&f64>>()
            .windows(2)
            .map(|pair| (*pair[1] - *pair[0]).abs() / (*pair[0] + 1e-8))
            .collect();

        let avg_variation = variations.iter().sum::<f64>() / variations.len() as f64;
        (1.0_f64 / (1.0 + avg_variation)).min(1.0)
    }

    fn compute_bottleneck_score(&self, history: &GradientHistory) -> f64 {
        if history.gradient_norms.is_empty() {
            return 0.0;
        }

        let mean = history.gradient_norms.iter().sum::<f64>() / history.gradient_norms.len() as f64;
        let min_val = history.gradient_norms.iter().cloned().fold(f64::INFINITY, f64::min);

        if mean == 0.0 {
            return 1.0;
        }

        1.0 - (min_val / mean).min(1.0)
    }

    fn compute_information_flow_rate(&self, history: &GradientHistory) -> f64 {
        if history.gradient_norms.len() < 2 {
            return 0.0;
        }

        let total_change: f64 = history
            .gradient_norms
            .iter()
            .collect::<Vec<&f64>>()
            .windows(2)
            .map(|pair| (*pair[1] - *pair[0]).abs())
            .sum();

        let time_span = history.gradient_norms.len() as f64;
        total_change / time_span
    }

    fn generate_temporal_flows(
        &self,
        gradient_histories: &HashMap<String, GradientHistory>,
        current_step: usize,
    ) -> Vec<TemporalGradientFlow> {
        let mut temporal_flows = Vec::new();

        for (layer_name, history) in gradient_histories {
            if let Some(latest_norm) = history.gradient_norms.back() {
                let flow_direction = self.get_latest_flow_direction(history);
                let stability_score = self.compute_stability_score(history);

                temporal_flows.push(TemporalGradientFlow {
                    step: current_step,
                    layer_name: layer_name.clone(),
                    gradient_magnitude: *latest_norm,
                    flow_direction,
                    stability_score,
                });
            }
        }

        temporal_flows
    }

    fn get_latest_flow_direction(&self, history: &GradientHistory) -> FlowDirection {
        if history.gradient_norms.len() < 3 {
            return FlowDirection::Forward;
        }

        let recent: Vec<f64> = history.gradient_norms.iter().rev().take(3).cloned().collect();
        let trend = recent[0] - recent[2]; // Latest - oldest in recent window

        if trend.abs() < 1e-6 {
            FlowDirection::Stagnant
        } else if trend > 0.0 {
            FlowDirection::Forward
        } else {
            // Check for oscillation
            let changes: Vec<f64> = recent.windows(2).map(|pair| pair[0] - pair[1]).collect();
            let sign_changes = changes.windows(2).filter(|pair| pair[0] * pair[1] < 0.0).count();

            if sign_changes > 0 {
                FlowDirection::Oscillating
            } else {
                FlowDirection::Backward
            }
        }
    }

    fn compute_stability_score(&self, history: &GradientHistory) -> f64 {
        if history.gradient_norms.len() < 3 {
            return 1.0;
        }

        let recent: Vec<f64> = history.gradient_norms.iter().rev().take(5).cloned().collect();
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance =
            recent.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;

        1.0 / (1.0 + variance)
    }

    fn build_gradient_flow_network(
        &self,
        layer_flows: &HashMap<String, GradientLayerFlow>,
    ) -> GradientFlowNetwork {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Create nodes
        for (layer_name, flow) in layer_flows {
            let node_type = self.classify_node_type(flow);
            let gradient_strength = flow.gradient_magnitudes.iter().sum::<f64>()
                / flow.gradient_magnitudes.len() as f64;
            let connectivity = layer_flows.len(); // Simplified
            let influence_score = gradient_strength * flow.flow_consistency;

            nodes.push(FlowNode {
                layer_name: layer_name.clone(),
                node_type,
                gradient_strength,
                connectivity,
                influence_score,
            });
        }

        // Create edges (simplified - would need actual layer connectivity information)
        let layer_names: Vec<String> = layer_flows.keys().cloned().collect();
        for i in 0..layer_names.len().saturating_sub(1) {
            let from_layer = &layer_names[i];
            let to_layer = &layer_names[i + 1];

            if let (Some(from_flow), Some(to_flow)) =
                (layer_flows.get(from_layer), layer_flows.get(to_layer))
            {
                let flow_strength =
                    (from_flow.information_flow_rate + to_flow.information_flow_rate) / 2.0;
                let flow_consistency =
                    (from_flow.flow_consistency + to_flow.flow_consistency) / 2.0;
                let edge_type = self.classify_edge_type(flow_strength, flow_consistency);

                edges.push(FlowEdge {
                    from_layer: from_layer.clone(),
                    to_layer: to_layer.clone(),
                    flow_strength,
                    flow_consistency,
                    edge_type,
                });
            }
        }

        let network_metrics = self.compute_network_metrics(&nodes, &edges);

        GradientFlowNetwork {
            nodes,
            edges,
            network_metrics,
        }
    }

    fn classify_node_type(&self, flow: &GradientLayerFlow) -> NodeType {
        if flow.bottleneck_score > 0.8 {
            NodeType::Bottleneck
        } else if flow.information_flow_rate > 1.0 {
            NodeType::Amplifier
        } else if flow.gradient_magnitudes.iter().sum::<f64>() < 0.01 {
            NodeType::Sink
        } else if flow.gradient_magnitudes.iter().any(|&x| x > 10.0) {
            NodeType::Source
        } else {
            NodeType::Normal
        }
    }

    fn classify_edge_type(&self, flow_strength: f64, flow_consistency: f64) -> EdgeType {
        if flow_strength > 1.0 && flow_consistency > 0.8 {
            EdgeType::Strong
        } else if flow_strength < 0.1 || flow_consistency < 0.3 {
            EdgeType::Weak
        } else if flow_consistency < 0.6 {
            EdgeType::Intermittent
        } else {
            EdgeType::Blocked
        }
    }

    fn compute_network_metrics(&self, nodes: &[FlowNode], edges: &[FlowEdge]) -> NetworkMetrics {
        let overall_flow_efficiency =
            edges.iter().map(|e| e.flow_strength).sum::<f64>() / edges.len().max(1) as f64;
        let network_connectivity = edges.len() as f64
            / (nodes.len().max(1) * (nodes.len().saturating_sub(1)).max(1)) as f64;
        let bottleneck_density =
            nodes.iter().filter(|n| matches!(n.node_type, NodeType::Bottleneck)).count() as f64
                / nodes.len() as f64;
        let flow_stability =
            edges.iter().map(|e| e.flow_consistency).sum::<f64>() / edges.len().max(1) as f64;
        let information_propagation_speed = overall_flow_efficiency * network_connectivity;

        NetworkMetrics {
            overall_flow_efficiency,
            network_connectivity,
            bottleneck_density,
            flow_stability,
            information_propagation_speed,
        }
    }

    fn identify_critical_gradient_paths(
        &self,
        network: &GradientFlowNetwork,
    ) -> Vec<CriticalGradientPath> {
        let mut paths = Vec::new();

        // Simplified path identification - would use graph algorithms in practice
        if network.nodes.len() < 2 {
            return paths;
        }

        let path_layers: Vec<String> = network.nodes.iter().map(|n| n.layer_name.clone()).collect();
        let total_flow_strength = network.edges.iter().map(|e| e.flow_strength).sum();
        let bottleneck_layers: Vec<String> = network
            .nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Bottleneck))
            .map(|n| n.layer_name.clone())
            .collect();

        paths.push(CriticalGradientPath {
            path_id: "main_path".to_string(),
            path_length: path_layers.len(),
            layers: path_layers,
            total_flow_strength,
            bottleneck_layers,
            criticality_score: 0.8, // Simplified
            optimization_potential: 0.6,
        });

        paths
    }

    fn identify_vanishing_regions(
        &self,
        gradient_histories: &HashMap<String, GradientHistory>,
    ) -> Vec<VanishingRegion> {
        let mut regions = Vec::new();
        let mut region_id = 0;

        for (layer_name, history) in gradient_histories {
            let avg_gradient =
                history.gradient_norms.iter().sum::<f64>() / history.gradient_norms.len() as f64;
            if avg_gradient < 1e-5 {
                region_id += 1;
                regions.push(VanishingRegion {
                    region_id: format!("vanishing_{}", region_id),
                    affected_layers: vec![layer_name.clone()],
                    severity_level: if avg_gradient < 1e-7 {
                        VanishingSeverity::Critical
                    } else {
                        VanishingSeverity::Moderate
                    },
                    extent: RegionExtent {
                        start_layer: layer_name.clone(),
                        end_layer: layer_name.clone(),
                        affected_parameters: 1000, // Placeholder
                        duration_steps: history.gradient_norms.len(),
                    },
                    mitigation_suggestions: vec![
                        "Consider better weight initialization".to_string(),
                        "Add skip connections".to_string(),
                        "Use gradient clipping".to_string(),
                    ],
                });
            }
        }

        regions
    }

    fn identify_exploding_regions(
        &self,
        gradient_histories: &HashMap<String, GradientHistory>,
    ) -> Vec<ExplodingRegion> {
        let mut regions = Vec::new();
        let mut region_id = 0;

        for (layer_name, history) in gradient_histories {
            let max_gradient = history.gradient_norms.iter().cloned().fold(0.0, f64::max);
            if max_gradient > 100.0 {
                region_id += 1;
                regions.push(ExplodingRegion {
                    region_id: format!("exploding_{}", region_id),
                    affected_layers: vec![layer_name.clone()],
                    severity_level: if max_gradient > 1000.0 {
                        ExplodingSeverity::Critical
                    } else {
                        ExplodingSeverity::Moderate
                    },
                    extent: RegionExtent {
                        start_layer: layer_name.clone(),
                        end_layer: layer_name.clone(),
                        affected_parameters: 1000, // Placeholder
                        duration_steps: history.gradient_norms.len(),
                    },
                    mitigation_suggestions: vec![
                        "Apply gradient clipping".to_string(),
                        "Reduce learning rate".to_string(),
                        "Check weight initialization".to_string(),
                    ],
                });
            }
        }

        regions
    }

    fn identify_gradient_dead_zones(
        &self,
        gradient_histories: &HashMap<String, GradientHistory>,
    ) -> Vec<GradientDeadZone> {
        let mut dead_zones = Vec::new();
        let mut zone_id = 0;

        for (layer_name, history) in gradient_histories {
            let zero_gradients = history.gradient_norms.iter().filter(|&&x| x < 1e-8).count();
            let dead_ratio = zero_gradients as f64 / history.gradient_norms.len() as f64;

            if dead_ratio > 0.5 {
                zone_id += 1;
                dead_zones.push(GradientDeadZone {
                    zone_id: format!("dead_zone_{}", zone_id),
                    affected_layers: vec![layer_name.clone()],
                    dead_duration: zero_gradients,
                    recovery_potential: if dead_ratio > 0.9 {
                        RecoveryPotential::Low
                    } else {
                        RecoveryPotential::Medium
                    },
                    intervention_required: dead_ratio > 0.8,
                });
            }
        }

        dead_zones
    }

    /// Create comprehensive visualization data for gradient flows
    pub fn create_visualization(
        &self,
        gradient_histories: &HashMap<String, GradientHistory>,
    ) -> GradientFlowVisualization {
        // Use existing methods to generate the visualization
        self.generate_visualization(gradient_histories, 0)
    }
}
