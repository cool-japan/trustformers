//! Grafana dashboard integration.
//!
//! Generates Grafana dashboard JSON configs and provides a Prometheus-compatible
//! metrics exporter with Grafana-specific annotations.

use std::path::Path;

use thiserror::Error;

/// Errors from the Grafana integration layer
#[derive(Debug, Error)]
pub enum GrafanaError {
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Panel visualization type
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GrafanaPanelType {
    #[serde(rename = "timeseries")]
    TimeSeries,
    #[serde(rename = "gauge")]
    Gauge,
    #[serde(rename = "stat")]
    Stat,
    #[serde(rename = "table")]
    Table,
    #[serde(rename = "heatmap")]
    Heatmap,
    #[serde(rename = "text")]
    Text,
}

/// PromQL target for a Grafana panel
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GrafanaTarget {
    pub expr: String,
    pub legend_format: String,
    pub ref_id: String,
}

/// Panel grid position
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GrafanaGridPos {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// A single Grafana panel
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GrafanaPanel {
    pub id: u32,
    pub title: String,
    #[serde(rename = "type")]
    pub panel_type: GrafanaPanelType,
    pub datasource: String,
    pub targets: Vec<GrafanaTarget>,
    #[serde(rename = "gridPos")]
    pub grid_pos: GrafanaGridPos,
    pub options: serde_json::Value,
}

impl GrafanaPanel {
    fn new_timeseries(
        id: u32,
        title: &str,
        datasource: &str,
        targets: Vec<GrafanaTarget>,
        grid_pos: GrafanaGridPos,
    ) -> Self {
        Self {
            id,
            title: title.to_string(),
            panel_type: GrafanaPanelType::TimeSeries,
            datasource: datasource.to_string(),
            targets,
            grid_pos,
            options: serde_json::json!({}),
        }
    }

    fn new_stat(
        id: u32,
        title: &str,
        datasource: &str,
        targets: Vec<GrafanaTarget>,
        grid_pos: GrafanaGridPos,
    ) -> Self {
        Self {
            id,
            title: title.to_string(),
            panel_type: GrafanaPanelType::Stat,
            datasource: datasource.to_string(),
            targets,
            grid_pos,
            options: serde_json::json!({"reduceOptions": {"calcs": ["lastNotNull"]}}),
        }
    }

    fn new_gauge(
        id: u32,
        title: &str,
        datasource: &str,
        targets: Vec<GrafanaTarget>,
        grid_pos: GrafanaGridPos,
    ) -> Self {
        Self {
            id,
            title: title.to_string(),
            panel_type: GrafanaPanelType::Gauge,
            datasource: datasource.to_string(),
            targets,
            grid_pos,
            options: serde_json::json!({"reduceOptions": {"calcs": ["lastNotNull"]}, "minVizWidth": 75}),
        }
    }
}

/// Time range for a Grafana dashboard
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GrafanaTimeRange {
    pub from: String,
    pub to: String,
}

impl Default for GrafanaTimeRange {
    fn default() -> Self {
        Self {
            from: "now-1h".to_string(),
            to: "now".to_string(),
        }
    }
}

/// A complete Grafana dashboard definition
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GrafanaDashboard {
    pub uid: String,
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub panels: Vec<GrafanaPanel>,
    pub time: GrafanaTimeRange,
    pub refresh: String,
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    pub version: u32,
}

/// TrustformeRS Grafana dashboard generator
pub struct TrustformersDashboardGenerator;

impl TrustformersDashboardGenerator {
    /// Generate the standard TrustformeRS serving dashboard.
    ///
    /// Includes panels for:
    /// - Request throughput (req/sec)
    /// - P50/P95/P99 latency
    /// - GPU memory utilization
    /// - Cache hit rate
    /// - Active connections
    /// - Error rate
    /// - Tokens per second
    /// - Queue depth
    pub fn serving_dashboard(datasource: &str) -> GrafanaDashboard {
        let panels = vec![
            // Row 0 — throughput + latency
            GrafanaPanel::new_timeseries(
                1,
                "Request Throughput (req/sec)",
                datasource,
                vec![GrafanaTarget {
                    expr: "rate(inference_requests_total[1m])".to_string(),
                    legend_format: "req/sec".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos { x: 0, y: 0, w: 12, h: 8 },
            ),
            GrafanaPanel::new_timeseries(
                2,
                "Request Latency (P50/P95/P99)",
                datasource,
                vec![
                    GrafanaTarget {
                        expr: "histogram_quantile(0.50, rate(inference_request_duration_seconds_bucket[5m]))".to_string(),
                        legend_format: "p50".to_string(),
                        ref_id: "A".to_string(),
                    },
                    GrafanaTarget {
                        expr: "histogram_quantile(0.95, rate(inference_request_duration_seconds_bucket[5m]))".to_string(),
                        legend_format: "p95".to_string(),
                        ref_id: "B".to_string(),
                    },
                    GrafanaTarget {
                        expr: "histogram_quantile(0.99, rate(inference_request_duration_seconds_bucket[5m]))".to_string(),
                        legend_format: "p99".to_string(),
                        ref_id: "C".to_string(),
                    },
                ],
                GrafanaGridPos { x: 12, y: 0, w: 12, h: 8 },
            ),
            // Row 1 — GPU memory + cache hit rate
            GrafanaPanel::new_gauge(
                3,
                "GPU Memory Utilization (%)",
                datasource,
                vec![GrafanaTarget {
                    expr: "100 * gpu_memory_used_bytes / gpu_memory_total_bytes".to_string(),
                    legend_format: "GPU {{gpu}}".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos { x: 0, y: 8, w: 6, h: 8 },
            ),
            GrafanaPanel::new_stat(
                4,
                "Cache Hit Rate (%)",
                datasource,
                vec![GrafanaTarget {
                    expr: "100 * rate(inference_cache_hits_total[5m]) / (rate(inference_cache_hits_total[5m]) + rate(inference_cache_misses_total[5m]) + 0.001)".to_string(),
                    legend_format: "hit rate".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos { x: 6, y: 8, w: 6, h: 8 },
            ),
            // Row 1 — active connections + error rate
            GrafanaPanel::new_stat(
                5,
                "Active Connections",
                datasource,
                vec![GrafanaTarget {
                    expr: "inference_active_requests".to_string(),
                    legend_format: "active".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos { x: 12, y: 8, w: 6, h: 8 },
            ),
            GrafanaPanel::new_timeseries(
                6,
                "Error Rate (errors/sec)",
                datasource,
                vec![GrafanaTarget {
                    expr: "rate(inference_errors_total[1m])".to_string(),
                    legend_format: "{{error_type}}".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos { x: 18, y: 8, w: 6, h: 8 },
            ),
            // Row 2 — tokens/sec + queue depth
            GrafanaPanel::new_timeseries(
                7,
                "Tokens Per Second",
                datasource,
                vec![GrafanaTarget {
                    expr: "rate(inference_tokens_generated_total[1m])".to_string(),
                    legend_format: "tokens/sec".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos { x: 0, y: 16, w: 12, h: 8 },
            ),
            GrafanaPanel::new_timeseries(
                8,
                "Request Queue Depth",
                datasource,
                vec![
                    GrafanaTarget {
                        expr: "inference_queue_depth{priority=\"critical\"}".to_string(),
                        legend_format: "critical".to_string(),
                        ref_id: "A".to_string(),
                    },
                    GrafanaTarget {
                        expr: "inference_queue_depth{priority=\"high\"}".to_string(),
                        legend_format: "high".to_string(),
                        ref_id: "B".to_string(),
                    },
                    GrafanaTarget {
                        expr: "inference_queue_depth{priority=\"normal\"}".to_string(),
                        legend_format: "normal".to_string(),
                        ref_id: "C".to_string(),
                    },
                    GrafanaTarget {
                        expr: "inference_queue_depth{priority=\"low\"}".to_string(),
                        legend_format: "low".to_string(),
                        ref_id: "D".to_string(),
                    },
                ],
                GrafanaGridPos { x: 12, y: 16, w: 12, h: 8 },
            ),
        ];

        GrafanaDashboard {
            uid: "trustformers-serving".to_string(),
            title: "TrustformeRS — Serving Overview".to_string(),
            description: "Real-time serving metrics for TrustformeRS inference endpoints"
                .to_string(),
            tags: vec![
                "trustformers".to_string(),
                "serving".to_string(),
                "ml".to_string(),
            ],
            panels,
            time: GrafanaTimeRange::default(),
            refresh: "10s".to_string(),
            schema_version: 38,
            version: 1,
        }
    }

    /// Generate a training metrics dashboard
    pub fn training_dashboard(datasource: &str) -> GrafanaDashboard {
        let panels = vec![
            GrafanaPanel::new_timeseries(
                1,
                "Training Loss",
                datasource,
                vec![GrafanaTarget {
                    expr: "training_loss".to_string(),
                    legend_format: "loss".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos {
                    x: 0,
                    y: 0,
                    w: 12,
                    h: 8,
                },
            ),
            GrafanaPanel::new_timeseries(
                2,
                "Validation Loss",
                datasource,
                vec![GrafanaTarget {
                    expr: "validation_loss".to_string(),
                    legend_format: "val loss".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos {
                    x: 12,
                    y: 0,
                    w: 12,
                    h: 8,
                },
            ),
            GrafanaPanel::new_timeseries(
                3,
                "Learning Rate",
                datasource,
                vec![GrafanaTarget {
                    expr: "training_learning_rate".to_string(),
                    legend_format: "lr".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos {
                    x: 0,
                    y: 8,
                    w: 8,
                    h: 8,
                },
            ),
            GrafanaPanel::new_stat(
                4,
                "Global Step",
                datasource,
                vec![GrafanaTarget {
                    expr: "training_global_step".to_string(),
                    legend_format: "step".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos {
                    x: 8,
                    y: 8,
                    w: 8,
                    h: 8,
                },
            ),
            GrafanaPanel::new_gauge(
                5,
                "GPU Utilization (%)",
                datasource,
                vec![GrafanaTarget {
                    expr: "100 * gpu_compute_utilization".to_string(),
                    legend_format: "GPU {{gpu}}".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos {
                    x: 16,
                    y: 8,
                    w: 8,
                    h: 8,
                },
            ),
            GrafanaPanel::new_timeseries(
                6,
                "Gradient Norm",
                datasource,
                vec![GrafanaTarget {
                    expr: "training_gradient_norm".to_string(),
                    legend_format: "grad norm".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos {
                    x: 0,
                    y: 16,
                    w: 12,
                    h: 8,
                },
            ),
            GrafanaPanel::new_timeseries(
                7,
                "Samples Per Second",
                datasource,
                vec![GrafanaTarget {
                    expr: "rate(training_samples_total[1m])".to_string(),
                    legend_format: "samples/sec".to_string(),
                    ref_id: "A".to_string(),
                }],
                GrafanaGridPos {
                    x: 12,
                    y: 16,
                    w: 12,
                    h: 8,
                },
            ),
        ];

        GrafanaDashboard {
            uid: "trustformers-training".to_string(),
            title: "TrustformeRS — Training Metrics".to_string(),
            description: "Training progress and performance for TrustformeRS models".to_string(),
            tags: vec![
                "trustformers".to_string(),
                "training".to_string(),
                "ml".to_string(),
            ],
            panels,
            time: GrafanaTimeRange {
                from: "now-6h".to_string(),
                to: "now".to_string(),
            },
            refresh: "30s".to_string(),
            schema_version: 38,
            version: 1,
        }
    }

    /// Export dashboard to JSON file (for Grafana provisioning)
    pub fn export_to_file(dashboard: &GrafanaDashboard, path: &Path) -> Result<(), GrafanaError> {
        let json = serde_json::to_string_pretty(dashboard)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load dashboard from JSON file
    pub fn load_from_file(path: &Path) -> Result<GrafanaDashboard, GrafanaError> {
        let data = std::fs::read_to_string(path)?;
        let dashboard = serde_json::from_str(&data)?;
        Ok(dashboard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serving_dashboard_has_8_panels() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        assert_eq!(
            dash.panels.len(),
            8,
            "serving dashboard should have 8 panels"
        );
    }

    #[test]
    fn test_serving_dashboard_uid() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        assert_eq!(dash.uid, "trustformers-serving");
    }

    #[test]
    fn test_training_dashboard_panels_count() {
        let dash = TrustformersDashboardGenerator::training_dashboard("prometheus");
        assert!(!dash.panels.is_empty());
    }

    #[test]
    fn test_serving_dashboard_serialization() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        let json_str = serde_json::to_string(&dash).expect("serialization failed");
        let reparsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("reparsing failed");
        assert_eq!(reparsed["uid"], "trustformers-serving");
        assert_eq!(reparsed["schemaVersion"], 38);
    }

    #[test]
    fn test_export_and_load_roundtrip() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        let dir = std::env::temp_dir();
        let path = dir.join("trustformers_grafana_test_dashboard.json");

        TrustformersDashboardGenerator::export_to_file(&dash, &path).expect("export failed");

        let loaded = TrustformersDashboardGenerator::load_from_file(&path).expect("load failed");

        assert_eq!(loaded.uid, dash.uid);
        assert_eq!(loaded.panels.len(), dash.panels.len());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_panel_types_in_serving_dashboard() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        // Must include timeseries, stat, and gauge panels
        let has_timeseries =
            dash.panels.iter().any(|p| matches!(p.panel_type, GrafanaPanelType::TimeSeries));
        let has_stat = dash.panels.iter().any(|p| matches!(p.panel_type, GrafanaPanelType::Stat));
        let has_gauge = dash.panels.iter().any(|p| matches!(p.panel_type, GrafanaPanelType::Gauge));
        assert!(has_timeseries, "should have timeseries panels");
        assert!(has_stat, "should have stat panels");
        assert!(has_gauge, "should have gauge panels");
    }

    #[test]
    fn test_serving_dashboard_panel_targets_have_expr() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        for panel in &dash.panels {
            for target in &panel.targets {
                assert!(
                    !target.expr.is_empty(),
                    "target expr must not be empty in panel {}",
                    panel.title
                );
            }
        }
    }

    #[test]
    fn test_training_dashboard_uid() {
        let dash = TrustformersDashboardGenerator::training_dashboard("prometheus");
        assert_eq!(dash.uid, "trustformers-training");
    }

    #[test]
    fn test_dashboard_refresh_interval() {
        let serving = TrustformersDashboardGenerator::serving_dashboard("prom");
        assert_eq!(serving.refresh, "10s");

        let training = TrustformersDashboardGenerator::training_dashboard("prom");
        assert_eq!(training.refresh, "30s");
    }

    #[test]
    fn test_grafana_error_display() {
        let io_err = GrafanaError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert!(io_err.to_string().contains("IO error"));
    }

    // ── Additional comprehensive tests ──────────────────────────────────────

    #[test]
    fn test_serving_dashboard_panel_ids_are_unique() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        let mut ids = std::collections::HashSet::new();
        for panel in &dash.panels {
            assert!(
                ids.insert(panel.id),
                "panel id {} appears more than once",
                panel.id
            );
        }
    }

    #[test]
    fn test_serving_dashboard_panel_titles_are_nonempty() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        for panel in &dash.panels {
            assert!(
                !panel.title.is_empty(),
                "panel id {} has empty title",
                panel.id
            );
        }
    }

    #[test]
    fn test_serving_dashboard_grid_positions_have_positive_dimensions() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        for panel in &dash.panels {
            let gp = &panel.grid_pos;
            assert!(gp.w > 0, "panel {} has zero width", panel.id);
            assert!(gp.h > 0, "panel {} has zero height", panel.id);
        }
    }

    #[test]
    fn test_serving_dashboard_schema_version() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        assert_eq!(dash.schema_version, 38);
    }

    #[test]
    fn test_serving_dashboard_tags_include_trustformers_and_serving() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        assert!(
            dash.tags.contains(&"trustformers".to_string()),
            "tags should include 'trustformers'"
        );
        assert!(
            dash.tags.contains(&"serving".to_string()),
            "tags should include 'serving'"
        );
    }

    #[test]
    fn test_training_dashboard_tags_include_training() {
        let dash = TrustformersDashboardGenerator::training_dashboard("prometheus");
        assert!(
            dash.tags.contains(&"trustformers".to_string()),
            "tags should include 'trustformers'"
        );
        assert!(
            dash.tags.contains(&"training".to_string()),
            "tags should include 'training'"
        );
    }

    #[test]
    fn test_serving_dashboard_has_throughput_panel() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        let found = dash.panels.iter().any(|p| p.title.contains("Throughput"));
        assert!(found, "should have a throughput panel");
    }

    #[test]
    fn test_serving_dashboard_has_latency_panel() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        let found = dash.panels.iter().any(|p| p.title.contains("Latency"));
        assert!(found, "should have a latency panel");
    }

    #[test]
    fn test_serving_dashboard_has_gpu_memory_panel() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prometheus");
        let found = dash.panels.iter().any(|p| p.title.contains("GPU"));
        assert!(found, "should have a GPU memory panel");
    }

    #[test]
    fn test_serving_dashboard_datasource_propagated() {
        let datasource = "my-prom-ds";
        let dash = TrustformersDashboardGenerator::serving_dashboard(datasource);
        for panel in &dash.panels {
            assert_eq!(
                panel.datasource, datasource,
                "panel {} should use datasource '{datasource}'",
                panel.id
            );
        }
    }

    #[test]
    fn test_training_dashboard_datasource_propagated() {
        let datasource = "train-prom";
        let dash = TrustformersDashboardGenerator::training_dashboard(datasource);
        for panel in &dash.panels {
            assert_eq!(
                panel.datasource, datasource,
                "panel {} should use datasource '{datasource}'",
                panel.id
            );
        }
    }

    #[test]
    fn test_serving_dashboard_time_range_default() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prom");
        assert_eq!(dash.time.from, "now-1h");
        assert_eq!(dash.time.to, "now");
    }

    #[test]
    fn test_training_dashboard_time_range_is_set() {
        let dash = TrustformersDashboardGenerator::training_dashboard("prom");
        assert!(!dash.time.from.is_empty(), "time.from should not be empty");
        assert!(!dash.time.to.is_empty(), "time.to should not be empty");
    }

    #[test]
    fn test_export_to_nonexistent_dir_returns_error() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prom");
        let bad_path = std::path::PathBuf::from("/nonexistent_dir_xyz/dashboard.json");
        let result = TrustformersDashboardGenerator::export_to_file(&dash, &bad_path);
        assert!(result.is_err(), "writing to nonexistent dir should fail");
    }

    #[test]
    fn test_load_from_nonexistent_file_returns_error() {
        let bad_path = std::path::PathBuf::from("/nonexistent_xyz/dashboard.json");
        let result = TrustformersDashboardGenerator::load_from_file(&bad_path);
        assert!(result.is_err(), "loading nonexistent file should fail");
    }

    #[test]
    fn test_serving_dashboard_panel_target_ref_ids_start_with_letter() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prom");
        for panel in &dash.panels {
            for target in &panel.targets {
                let first_char = target.ref_id.chars().next();
                assert!(
                    first_char.is_some_and(|c| c.is_alphabetic()),
                    "ref_id '{}' in panel '{}' should start with a letter",
                    target.ref_id,
                    panel.title
                );
            }
        }
    }

    #[test]
    fn test_grafana_time_range_default_values() {
        let tr = GrafanaTimeRange::default();
        assert_eq!(tr.from, "now-1h");
        assert_eq!(tr.to, "now");
    }

    #[test]
    fn test_serving_dashboard_version_is_one() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prom");
        assert_eq!(dash.version, 1);
    }

    #[test]
    fn test_roundtrip_preserves_panel_titles() {
        let dash = TrustformersDashboardGenerator::training_dashboard("prom");
        let dir = std::env::temp_dir();
        let path = dir.join("trustformers_grafana_training_titles_test.json");

        TrustformersDashboardGenerator::export_to_file(&dash, &path).expect("export failed");

        let loaded = TrustformersDashboardGenerator::load_from_file(&path).expect("load failed");

        let orig_titles: Vec<&str> = dash.panels.iter().map(|p| p.title.as_str()).collect();
        let loaded_titles: Vec<&str> = loaded.panels.iter().map(|p| p.title.as_str()).collect();
        assert_eq!(
            orig_titles, loaded_titles,
            "panel titles should be preserved"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_serving_dashboard_description_nonempty() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prom");
        assert!(
            !dash.description.is_empty(),
            "description should not be empty"
        );
    }

    #[test]
    fn test_training_dashboard_has_loss_panel() {
        let dash = TrustformersDashboardGenerator::training_dashboard("prom");
        let found = dash.panels.iter().any(|p| p.title.contains("Loss"));
        assert!(found, "training dashboard should have a loss panel");
    }

    #[test]
    fn test_serving_dashboard_panel_legend_formats_nonempty() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prom");
        for panel in &dash.panels {
            for target in &panel.targets {
                assert!(
                    !target.legend_format.is_empty(),
                    "legend_format in panel '{}' target '{}' should not be empty",
                    panel.title,
                    target.ref_id
                );
            }
        }
    }

    #[test]
    fn test_serialization_preserves_panel_exprs() {
        let dash = TrustformersDashboardGenerator::serving_dashboard("prom");
        let json_str = serde_json::to_string(&dash).expect("serialization failed");
        // Check that prometheus expressions survive serialization
        assert!(
            json_str.contains("rate(inference_requests_total"),
            "serialized JSON should contain rate() expr"
        );
    }
}
