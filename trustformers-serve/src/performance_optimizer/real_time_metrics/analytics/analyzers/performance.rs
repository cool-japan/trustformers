//! Performance analysis of an observed metrics window.
//!
//! Replaces the `create_analyzer_placeholder!`-generated `PerformanceAnalyzer`,
//! which ignored its input and returned an invented report: `p95: 80.0`,
//! `p99: 100.0`, `current_availability: 0.999`, `mttr: 300s`, `mtbf: 86400s`,
//! `compliance_rate: 0.95` with ten violations, and four resource utilisation
//! blocks of made-up numbers. Every value below is derived from the samples;
//! availability and SLA compliance are now `Option` and reported as `None`
//! because a metrics window carries neither uptime incidents nor an SLA target.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};

use super::super::super::types::data_structures::TimestampedMetrics;
use super::super::super::types::enums::{InsightType, TrendDirection};
use super::super::super::types::metrics::{ImpactAssessment, PerformanceInsight};
use super::super::types::{
    AnomalousPeriod, EfficiencyAnalysis, EfficiencyComponents, EfficiencyTrend, LatencyAnalysis,
    LatencyStatistics, PerformanceAnalysisResult, PerformanceMetricsAnalysis,
    PerformanceThresholds, SeasonalEffect, TrendComponents, UtilizationMetrics,
};
use super::super::types::{
    BottleneckAnalysis, BottleneckType, ErrorRateAnalysis, ForecastPoint, LatencyDistribution,
    LatencyTrend, OutlierAnalysis, PerformanceBottleneck, PerformanceTrendAnalysis, TailAnalysis,
    ThroughputAnalysis, TrendForecast,
};
use super::distribution::{best_family, quartiles};
use super::series::{
    autocorrelation, extract_series, linear_fit, mean, pearson, percentile_sorted, sample_std_dev,
    sample_variance, slope_p_value, sorted_finite, varies_materially, MetricSeries,
};
use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

/// Minimum samples before a percentile or trend claim is defensible.
const MIN_SAMPLES: usize = 5;

/// Number of forecast points emitted per trended metric.
const FORECAST_STEPS: usize = 5;

/// Measures throughput, latency, resource use and error rate over a window.
#[derive(Clone, Debug)]
pub struct PerformanceAnalyzer {
    thresholds: PerformanceThresholds,
    shutdown: Arc<AtomicBool>,
}

impl PerformanceAnalyzer {
    /// Create a performance analyzer bound to the caller's thresholds.
    pub async fn new(thresholds: PerformanceThresholds) -> Result<Self> {
        Ok(Self {
            thresholds,
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Analyse the performance characteristics of `data`.
    pub async fn analyze(&self, data: &[TimestampedMetrics]) -> Result<PerformanceAnalysisResult> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(anyhow!("Performance analyzer is shut down"));
        }
        if data.len() < MIN_SAMPLES {
            return Err(anyhow!(
                "Performance analysis needs at least {} samples, got {}",
                MIN_SAMPLES,
                data.len()
            ));
        }
        let series = extract_series(data);
        let timestamps: Vec<DateTime<Utc>> = data.iter().map(|s| s.timestamp).collect();
        let step = median_step(&timestamps);

        let throughput = self.analyse_throughput(&series)?;
        let latency = self.analyse_latency(&series, step)?;
        let resource_utilization = self.analyse_resources(&series, &timestamps);
        let error_rate = self.analyse_error_rate(&series)?;
        let bottleneck_analysis = self.analyse_bottlenecks(&series);
        let efficiency_analysis = self.analyse_efficiency(&series);
        let performance_trends = self.analyse_trends(&series, &timestamps, step);
        let insights = self.derive_insights(&series, &latency, &bottleneck_analysis);

        Ok(PerformanceAnalysisResult {
            metrics_analysis: PerformanceMetricsAnalysis {
                throughput,
                latency,
                resource_utilization,
                error_rate,
                // Availability needs uptime/incident records; a metrics window
                // carries none, so no availability figure is reported.
                availability: None,
            },
            bottleneck_analysis,
            efficiency_analysis,
            performance_trends,
            insights,
            // Estimating ROI, implementation effort and risk is a planning
            // activity with no basis in the sampled metrics. The measured
            // headroom is reported through `efficiency_analysis` instead.
            optimization_opportunities: Vec::new(),
        })
    }

    /// Stop accepting analyses.
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// True once `shutdown` has been called.
    pub fn is_shut_down(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    fn analyse_throughput(&self, series: &[MetricSeries]) -> Result<ThroughputAnalysis> {
        let values = pick(series, "throughput")
            .ok_or_else(|| anyhow!("Metric series `throughput` carried no finite samples"))?;
        let average = mean(values).ok_or_else(|| anyhow!("Empty `throughput` series"))?;
        let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let current = values.last().copied().unwrap_or(average);
        let std_dev = sample_std_dev(values).unwrap_or(0.0);
        Ok(ThroughputAnalysis {
            current_throughput: current,
            peak_throughput: peak,
            average_throughput: average,
            trend: trend_direction(values),
            // Coefficient of variation: spread relative to the mean level.
            variability: if average.abs() > f64::EPSILON { std_dev / average.abs() } else { 0.0 },
            // Reported against the window's own observed peak; no absolute
            // capacity figure is available to this analyzer.
            capacity_utilization: if peak > 0.0 { (average / peak).clamp(0.0, 1.0) } else { 0.0 },
        })
    }

    fn analyse_latency(&self, series: &[MetricSeries], step: Duration) -> Result<LatencyAnalysis> {
        let values = pick(series, "latency_seconds")
            .ok_or_else(|| anyhow!("Metric series `latency_seconds` carried no finite samples"))?;
        let sorted = sorted_finite(values);
        if sorted.is_empty() {
            return Err(anyhow!("No finite latency samples in the window"));
        }
        let current_stats = LatencyStatistics {
            mean: mean(&sorted).unwrap_or(0.0),
            median: percentile_sorted(&sorted, 0.50).unwrap_or(0.0),
            p95: percentile_sorted(&sorted, 0.95).unwrap_or(0.0),
            p99: percentile_sorted(&sorted, 0.99).unwrap_or(0.0),
            p999: percentile_sorted(&sorted, 0.999).unwrap_or(0.0),
            max: sorted.last().copied().unwrap_or(0.0),
            std_dev: sample_std_dev(&sorted).unwrap_or(0.0),
        };
        let (family, parameters, _score) =
            best_family(&sorted).unwrap_or_else(|| ("unfitted".to_string(), HashMap::new(), 0.0));
        let distribution = LatencyDistribution {
            distribution_type: family,
            parameters,
            tail_analysis: tail_analysis(&sorted, &current_stats),
            outlier_analysis: outlier_analysis(&sorted),
        };
        let trends = latency_trends(values, step);
        Ok(LatencyAnalysis {
            current_stats,
            distribution,
            trends,
            // No latency SLA target is configured anywhere this analyzer can
            // read, so compliance against one cannot be computed.
            sla_compliance: None,
        })
    }

    fn analyse_resources(
        &self,
        series: &[MetricSeries],
        timestamps: &[DateTime<Utc>],
    ) -> super::super::types::ResourceUtilizationAnalysis {
        let cpu = utilization(pick(series, "cpu_utilization"), timestamps);
        let memory = utilization(pick(series, "memory_utilization"), timestamps);
        let io = utilization(pick(series, "io_usage"), timestamps);
        let network = utilization(pick(series, "network_usage"), timestamps);
        let ratios: Vec<f64> = [&cpu, &memory, &io, &network]
            .into_iter()
            .filter(|m| m.peak > 0.0)
            .map(|m| (m.average / m.peak).clamp(0.0, 1.0))
            .collect();
        super::super::types::ResourceUtilizationAnalysis {
            cpu_utilization: cpu,
            memory_utilization: memory,
            io_utilization: io,
            network_utilization: network,
            // Mean-to-peak ratio across the resources that reported readings:
            // 1.0 means every resource ran flat at its observed peak.
            efficiency_score: mean(&ratios).unwrap_or(0.0),
        }
    }

    fn analyse_error_rate(&self, series: &[MetricSeries]) -> Result<ErrorRateAnalysis> {
        let values = pick(series, "error_rate")
            .ok_or_else(|| anyhow!("Metric series `error_rate` carried no finite samples"))?;
        let current = values.last().copied().unwrap_or(0.0);
        let average = mean(values).unwrap_or(0.0);
        Ok(ErrorRateAnalysis {
            current_rate: current,
            trend: trend_direction(values),
            // The metric stream carries a scalar rate with no type breakdown,
            // so no per-type analysis can be produced.
            error_types: HashMap::new(),
            // Likewise there is no error taxonomy to mine for patterns.
            patterns: Vec::new(),
            // Impact is reported as the window's mean error rate: the share of
            // work that failed over the window.
            impact_assessment: average,
        })
    }

    fn analyse_bottlenecks(&self, series: &[MetricSeries]) -> BottleneckAnalysis {
        let Some(latency) = pick(series, "latency_seconds") else {
            return BottleneckAnalysis {
                bottlenecks: Vec::new(),
                severity_ranking: Vec::new(),
                impact_assessment: HashMap::new(),
                mitigation_strategies: Vec::new(),
            };
        };
        let candidates: [(&str, BottleneckType); 4] = [
            ("cpu_utilization", BottleneckType::Cpu),
            ("memory_utilization", BottleneckType::Memory),
            ("io_usage", BottleneckType::Io),
            ("network_usage", BottleneckType::Network),
        ];
        let mut bottlenecks = Vec::new();
        let mut impact_assessment = HashMap::new();
        for (name, kind) in candidates {
            let Some(values) = pick(series, name) else {
                continue;
            };
            let Some(r) = pearson(values, latency) else {
                continue;
            };
            if r <= 0.0 {
                // Only a resource whose rise tracks rising latency is evidence
                // of a bottleneck.
                continue;
            }
            let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let near_peak = if peak > 0.0 {
                values.iter().filter(|v| **v >= peak * 0.95).count() as f64 / values.len() as f64
            } else {
                0.0
            };
            // Seconds of latency added per unit of this resource, from the OLS
            // fit of latency on the resource reading.
            let degradation = regression_slope(values, latency).unwrap_or(0.0);
            impact_assessment.insert(name.to_string(), r);
            bottlenecks.push(PerformanceBottleneck {
                name: name.to_string(),
                bottleneck_type: kind,
                impact_score: r,
                frequency: near_peak,
                contributing_factors: vec![format!(
                    "pearson(latency, {}) = {:.3} over {} samples",
                    name,
                    r,
                    values.len()
                )],
                performance_degradation: degradation,
            });
        }
        bottlenecks.sort_by(|a, b| {
            b.impact_score.partial_cmp(&a.impact_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        let severity_ranking = bottlenecks.iter().map(|b| b.name.clone()).collect();
        BottleneckAnalysis {
            bottlenecks,
            severity_ranking,
            impact_assessment,
            // Choosing a mitigation is a planning decision with an engineering
            // cost this analyzer cannot observe; it reports evidence only.
            mitigation_strategies: Vec::new(),
        }
    }

    fn analyse_efficiency(&self, series: &[MetricSeries]) -> EfficiencyAnalysis {
        let throughput = pick(series, "throughput");
        let cpu = pick(series, "cpu_utilization");
        let latency = pick(series, "latency_seconds");

        // Throughput delivered per unit of CPU, normalised by the best such
        // ratio the window actually achieved.
        let resource_efficiency = match (throughput, cpu) {
            (Some(t), Some(c)) if t.len() == c.len() => {
                let ratios: Vec<f64> = t
                    .iter()
                    .zip(c.iter())
                    .filter(|(_, cpu)| **cpu > 0.0)
                    .map(|(throughput, cpu)| throughput / cpu)
                    .collect();
                let best = ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                match (mean(&ratios), best) {
                    (Some(avg), best) if best > 0.0 => (avg / best).clamp(0.0, 1.0),
                    _ => 0.0,
                }
            },
            _ => 0.0,
        };
        // How close the window ran to its own observed peak throughput.
        let computational_efficiency = match throughput {
            Some(values) => {
                let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                match (mean(values), peak) {
                    (Some(avg), peak) if peak > 0.0 => (avg / peak).clamp(0.0, 1.0),
                    _ => 0.0,
                }
            },
            None => 0.0,
        };
        // How close the average request ran to the fastest one observed.
        let time_efficiency = match latency {
            Some(values) => {
                let fastest = values.iter().copied().fold(f64::INFINITY, f64::min);
                match mean(values) {
                    Some(avg) if avg > 0.0 && fastest.is_finite() => {
                        (fastest / avg).clamp(0.0, 1.0)
                    },
                    _ => 0.0,
                }
            },
            None => 0.0,
        };
        let components = EfficiencyComponents {
            resource_efficiency,
            computational_efficiency,
            // No power telemetry reaches this analyzer.
            energy_efficiency: None,
            // No cost telemetry reaches this analyzer.
            cost_efficiency: None,
            time_efficiency,
        };
        let measured = [
            resource_efficiency,
            computational_efficiency,
            time_efficiency,
        ];
        let overall_efficiency = mean(&measured).unwrap_or(0.0);
        let mut benchmarks = HashMap::new();
        if let Some(values) = throughput {
            benchmarks.insert(
                "peak_throughput".to_string(),
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            );
        }
        if let Some(values) = latency {
            benchmarks.insert(
                "fastest_latency_seconds".to_string(),
                values.iter().copied().fold(f64::INFINITY, f64::min),
            );
        }
        benchmarks.insert(
            "max_cpu_threshold".to_string(),
            self.thresholds.max_cpu_usage,
        );
        EfficiencyAnalysis {
            overall_efficiency,
            components,
            trends: efficiency_trends(series),
            benchmarks,
            // Headroom to a window that ran at its own observed best throughout.
            improvement_potential: (1.0 - overall_efficiency).clamp(0.0, 1.0),
        }
    }

    fn analyse_trends(
        &self,
        series: &[MetricSeries],
        timestamps: &[DateTime<Utc>],
        step: Duration,
    ) -> Vec<PerformanceTrendAnalysis> {
        series
            .iter()
            .filter(|s| s.values.len() >= MIN_SAMPLES)
            .filter_map(|entry| {
                let fit = linear_fit(&entry.values)?;
                let spread = sample_std_dev(&entry.values).unwrap_or(0.0);
                let quadratic = quadratic_coefficient(&entry.values).unwrap_or(0.0);
                let seasonal = (2..=(entry.values.len() / 3).min(32))
                    .filter_map(|lag| autocorrelation(&entry.values, lag))
                    .fold(0.0f64, f64::max);
                let last = entry.values.last().copied().unwrap_or(0.0);
                let last_timestamp = timestamps.last().copied().unwrap_or_else(Utc::now);
                let predicted_values: Vec<ForecastPoint> = (1..=FORECAST_STEPS)
                    .map(|k| {
                        let value = fit.intercept + fit.slope * (entry.values.len() + k - 1) as f64;
                        let margin = 1.96 * fit.residual_std_dev * (k as f64).sqrt();
                        ForecastPoint {
                            timestamp: last_timestamp
                                + chrono::Duration::from_std(step * k as u32)
                                    .unwrap_or_else(|_| chrono::Duration::zero()),
                            value,
                            lower_bound: value - margin,
                            upper_bound: value + margin,
                            confidence: fit.r_squared,
                        }
                    })
                    .collect();
                Some(PerformanceTrendAnalysis {
                    metric: entry.name.to_string(),
                    trend_components: TrendComponents {
                        linear_trend: fit.slope,
                        quadratic_trend: quadratic,
                        seasonal_component: seasonal,
                        noise_component: if spread > 0.0 {
                            (fit.residual_std_dev / spread).clamp(0.0, 1.0)
                        } else {
                            0.0
                        },
                        explained_variance: fit.r_squared,
                    },
                    seasonal_effects: if seasonal >= 0.5 {
                        vec![SeasonalEffect {
                            season: format!("{}_autocorrelated_cycle", entry.name),
                            magnitude: seasonal * spread,
                            duration: step,
                            confidence: seasonal,
                        }]
                    } else {
                        Vec::new()
                    },
                    anomalous_periods: anomalous_periods(entry, timestamps),
                    forecast: TrendForecast {
                        horizon: step * FORECAST_STEPS as u32,
                        predicted_values,
                        confidence: fit.r_squared,
                        method: "ordinary_least_squares".to_string(),
                    },
                })
                .filter(|_| last.is_finite())
            })
            .collect()
    }

    fn derive_insights(
        &self,
        series: &[MetricSeries],
        latency: &LatencyAnalysis,
        bottlenecks: &BottleneckAnalysis,
    ) -> Vec<PerformanceInsight> {
        let mut out = Vec::new();
        if let Some(top) = bottlenecks.bottlenecks.first() {
            let mut insight = PerformanceInsight::new(
                InsightType::ResourceBottleneck,
                format!(
                    "`{}` tracks latency most closely (r = {:.3}); it sat within 5% of its \
                     observed peak in {:.1}% of samples",
                    top.name,
                    top.impact_score,
                    top.frequency * 100.0
                ),
                severity_from(top.impact_score),
            );
            insight.add_data("correlation_with_latency".to_string(), top.impact_score);
            insight.add_data("near_peak_fraction".to_string(), top.frequency);
            insight.impact = measured_impact(top.impact_score as f32, top.frequency as f32);
            insight.confidence = top.impact_score as f32;
            out.push(insight);
        }
        let stats = &latency.current_stats;
        if stats.median > 0.0 && stats.p99 > stats.median * 5.0 {
            let ratio = stats.p99 / stats.median;
            let mut insight = PerformanceInsight::new(
                InsightType::PerformanceDegradation,
                format!(
                    "Latency tail is {:.1}x the median (p99 {:.6}s vs median {:.6}s)",
                    ratio, stats.p99, stats.median
                ),
                severity_from((ratio / 20.0).clamp(0.0, 1.0)),
            );
            insight.add_data("p99_over_median".to_string(), ratio);
            insight.add_data("p99_seconds".to_string(), stats.p99);
            insight.impact = measured_impact((ratio / 20.0).clamp(0.0, 1.0) as f32, 0.0);
            insight.confidence = (ratio / 20.0).clamp(0.0, 1.0) as f32;
            out.push(insight);
        }
        if let Some(values) = pick(series, "error_rate") {
            let average = mean(values).unwrap_or(0.0);
            if average > 0.0 {
                let mut insight = PerformanceInsight::new(
                    InsightType::ThresholdViolation,
                    format!(
                        "Mean error rate over the window is {:.4} across {} samples",
                        average,
                        values.len()
                    ),
                    severity_from(average),
                );
                insight.add_data("mean_error_rate".to_string(), average);
                insight.impact = measured_impact(average as f32, 0.0);
                insight.confidence = 1.0;
                out.push(insight);
            }
        }
        if let Some(values) = pick(series, "throughput") {
            if matches!(trend_direction(values), TrendDirection::Decreasing) {
                if let Some(fit) = linear_fit(values) {
                    let mut insight = PerformanceInsight::new(
                        InsightType::TrendChange,
                        format!(
                            "Throughput is falling at {:.6} per sample (R^2 = {:.3})",
                            fit.slope, fit.r_squared
                        ),
                        severity_from(fit.r_squared),
                    );
                    insight.add_data("slope_per_sample".to_string(), fit.slope);
                    insight.add_data("r_squared".to_string(), fit.r_squared);
                    insight.impact = measured_impact(fit.r_squared as f32, 0.0);
                    insight.confidence = fit.r_squared as f32;
                    out.push(insight);
                }
            }
        }
        out
    }
}

/// An impact record carrying only the two fields this analyzer measures.
///
/// `complexity`, `risk_level`, `estimated_benefit` and `implementation_time`
/// describe a remediation this analyzer never proposes; they are left at zero
/// and must not be read as an assessment.
fn measured_impact(performance_impact: f32, resource_impact: f32) -> ImpactAssessment {
    ImpactAssessment {
        performance_impact,
        resource_impact,
        complexity: 0.0,
        risk_level: 0.0,
        estimated_benefit: 0.0,
        implementation_time: Duration::ZERO,
    }
}

fn severity_from(score: f64) -> SeverityLevel {
    match score {
        s if s >= 0.8 => SeverityLevel::Critical,
        s if s >= 0.6 => SeverityLevel::High,
        s if s >= 0.3 => SeverityLevel::Medium,
        s if s > 0.0 => SeverityLevel::Low,
        _ => SeverityLevel::Info,
    }
}

fn pick<'a>(series: &'a [MetricSeries], name: &str) -> Option<&'a [f64]> {
    series
        .iter()
        .find(|s| s.name == name && !s.values.is_empty())
        .map(|s| s.values.as_slice())
}

fn median_step(timestamps: &[DateTime<Utc>]) -> Duration {
    let mut gaps: Vec<i64> = timestamps
        .windows(2)
        .filter_map(|pair| {
            let (Some(first), Some(second)) = (pair.first(), pair.get(1)) else {
                return None;
            };
            let delta = second.signed_duration_since(*first).num_milliseconds();
            if delta > 0 {
                Some(delta)
            } else {
                None
            }
        })
        .collect();
    if gaps.is_empty() {
        return Duration::from_secs(1);
    }
    gaps.sort_unstable();
    Duration::from_millis(gaps.get(gaps.len() / 2).copied().unwrap_or(1000).max(1) as u64)
}

/// Direction of a series, called only when the OLS slope is significant at 5%.
fn trend_direction(values: &[f64]) -> TrendDirection {
    let Some(fit) = linear_fit(values) else {
        return TrendDirection::Unknown;
    };
    let p_value = slope_p_value(&fit, values.len());
    if p_value >= 0.05 {
        // No significant drift: distinguish a steady series from a noisy one by
        // its coefficient of variation.
        let (Some(m), Some(sd)) = (mean(values), sample_std_dev(values)) else {
            return TrendDirection::Stable;
        };
        return if m.abs() > f64::EPSILON && sd / m.abs() > 0.5 {
            TrendDirection::Volatile
        } else {
            TrendDirection::Stable
        };
    }
    if fit.slope > 0.0 {
        TrendDirection::Increasing
    } else {
        TrendDirection::Decreasing
    }
}

fn regression_slope(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 3 {
        return None;
    }
    let mx = mean(x)?;
    let my = mean(y)?;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (a, b) in x.iter().zip(y.iter()) {
        let dx = a - mx;
        sxx += dx * dx;
        sxy += dx * (b - my);
    }
    if sxx <= 0.0 {
        return None;
    }
    Some(sxy / sxx)
}

/// Quadratic coefficient of a degree-2 least-squares fit against the index.
fn quadratic_coefficient(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n < 4 {
        return None;
    }
    // Fit against centred, squared index so the quadratic term is orthogonal to
    // the mean and to the linear term for an evenly spaced index.
    let centre = (n as f64 - 1.0) / 2.0;
    let x2: Vec<f64> = (0..n)
        .map(|i| {
            let d = i as f64 - centre;
            d * d
        })
        .collect();
    let adjusted_mean = mean(&x2)?;
    let centred: Vec<f64> = x2.iter().map(|v| v - adjusted_mean).collect();
    let my = mean(values)?;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (x, y) in centred.iter().zip(values.iter()) {
        sxx += x * x;
        sxy += x * (y - my);
    }
    if sxx <= 0.0 {
        return None;
    }
    Some(sxy / sxx)
}

fn utilization(values: Option<&[f64]>, timestamps: &[DateTime<Utc>]) -> UtilizationMetrics {
    let Some(values) = values else {
        // No readings: every statistic is zero and `saturation_points` empty,
        // which is distinguishable from a real all-zero series only by the
        // absence of the series from the analytics input.
        return UtilizationMetrics {
            current: 0.0,
            peak: 0.0,
            average: 0.0,
            variance: 0.0,
            saturation_points: Vec::new(),
        };
    };
    let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let saturation_points = if peak > 0.0 {
        values
            .iter()
            .enumerate()
            .filter(|(_, v)| **v >= peak * 0.95)
            .filter_map(|(i, _)| timestamps.get(i).copied())
            .collect()
    } else {
        Vec::new()
    };
    UtilizationMetrics {
        current: values.last().copied().unwrap_or(0.0),
        peak: if peak.is_finite() { peak } else { 0.0 },
        average: mean(values).unwrap_or(0.0),
        variance: sample_variance(values).unwrap_or(0.0),
        saturation_points,
    }
}

fn tail_analysis(sorted: &[f64], stats: &LatencyStatistics) -> TailAnalysis {
    let mut extreme_value_stats = HashMap::new();
    extreme_value_stats.insert("p99".to_string(), stats.p99);
    extreme_value_stats.insert("p999".to_string(), stats.p999);
    extreme_value_stats.insert("max".to_string(), stats.max);
    let tail_index = hill_estimator(sorted).unwrap_or(f64::NAN);
    TailAnalysis {
        // A Hill tail index below two means the variance of the tail does not
        // converge: the classic heavy-tail signature.
        has_heavy_tails: tail_index.is_finite() && tail_index < 2.0,
        tail_index,
        extreme_value_stats,
    }
}

/// Hill estimator of the tail index over the top decile.
fn hill_estimator(sorted: &[f64]) -> Option<f64> {
    let n = sorted.len();
    let k = (n / 10).max(2);
    if n <= k {
        return None;
    }
    let threshold = *sorted.get(n - k - 1)?;
    if threshold <= 0.0 {
        return None;
    }
    let mut total = 0.0;
    for value in sorted.get(n - k..)? {
        if *value <= 0.0 {
            return None;
        }
        total += (value / threshold).ln();
    }
    if total <= 0.0 {
        return None;
    }
    Some(k as f64 / total)
}

fn outlier_analysis(sorted: &[f64]) -> OutlierAnalysis {
    let Some((q1, _q2, q3)) = quartiles(sorted) else {
        return OutlierAnalysis {
            outlier_count: 0,
            outlier_percentage: 0.0,
            characteristics: HashMap::new(),
            impact_assessment: 0.0,
        };
    };
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    let outliers: Vec<f64> = sorted.iter().copied().filter(|v| *v < lower || *v > upper).collect();
    let total: f64 = sorted.iter().sum();
    let outlier_total: f64 = outliers.iter().sum();
    let mut characteristics = HashMap::new();
    characteristics.insert("q1".to_string(), q1);
    characteristics.insert("q3".to_string(), q3);
    characteristics.insert("iqr".to_string(), iqr);
    characteristics.insert("lower_fence".to_string(), lower);
    characteristics.insert("upper_fence".to_string(), upper);
    OutlierAnalysis {
        outlier_count: outliers.len() as u64,
        outlier_percentage: outliers.len() as f64 / sorted.len() as f64,
        characteristics,
        // Share of the summed latency contributed by samples outside the fences.
        impact_assessment: if total > 0.0 { outlier_total / total } else { 0.0 },
    }
}

fn latency_trends(values: &[f64], step: Duration) -> Vec<LatencyTrend> {
    let Some(fit) = linear_fit(values) else {
        return Vec::new();
    };
    let p_value = slope_p_value(&fit, values.len());
    if p_value >= 0.05 {
        return Vec::new();
    }
    vec![LatencyTrend {
        metric: "latency_seconds".to_string(),
        direction: if fit.slope > 0.0 {
            TrendDirection::Increasing
        } else {
            TrendDirection::Decreasing
        },
        magnitude: fit.slope,
        duration: step * values.len() as u32,
        confidence: 1.0 - p_value,
    }]
}

fn efficiency_trends(series: &[MetricSeries]) -> Vec<EfficiencyTrend> {
    series
        .iter()
        .filter(|s| s.values.len() >= MIN_SAMPLES)
        .filter_map(|entry| {
            let fit = linear_fit(&entry.values)?;
            let p_value = slope_p_value(&fit, entry.values.len());
            if p_value >= 0.05 {
                return None;
            }
            Some(EfficiencyTrend {
                component: entry.name.to_string(),
                direction: if fit.slope > 0.0 {
                    TrendDirection::Increasing
                } else {
                    TrendDirection::Decreasing
                },
                improvement_rate: fit.slope,
                // The trend covers the whole window; the caller knows its span.
                duration: Duration::ZERO,
                confidence: 1.0 - p_value,
            })
        })
        .collect()
}

fn anomalous_periods(entry: &MetricSeries, timestamps: &[DateTime<Utc>]) -> Vec<AnomalousPeriod> {
    let (Some(m), Some(sd)) = (mean(&entry.values), sample_std_dev(&entry.values)) else {
        return Vec::new();
    };
    if !varies_materially(&entry.values) {
        return Vec::new();
    }
    entry
        .values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            let z = (value - m).abs() / sd;
            if z <= 3.0 {
                return None;
            }
            let timestamp = timestamps.get(index).copied()?;
            Some(AnomalousPeriod {
                start_time: timestamp,
                end_time: timestamp,
                severity: ((z - 3.0) / 3.0).clamp(0.0, 1.0),
                description: format!(
                    "`{}` sample {} sits {:.2} standard deviations from the window mean",
                    entry.name, index, z
                ),
                factors: vec![format!("mean={:.6}", m), format!("std_dev={:.6}", sd)],
            })
        })
        .collect()
}
