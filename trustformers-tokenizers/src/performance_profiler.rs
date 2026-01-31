use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Configuration for performance profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    pub warmup_iterations: usize,
    pub benchmark_iterations: usize,
    pub measure_memory: bool,
    pub measure_throughput: bool,
    pub concurrent_threads: Option<usize>,
    pub text_lengths: Vec<usize>,
    pub batch_sizes: Vec<usize>,
    pub detailed_timing: bool,
    pub export_format: ExportFormat,
}

/// Export format for profiling results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
    Html,
    Markdown,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            benchmark_iterations: 10,
            measure_memory: true,
            measure_throughput: true,
            concurrent_threads: Some(num_cpus::get()),
            text_lengths: vec![50, 100, 500, 1000, 5000],
            batch_sizes: vec![1, 8, 16, 32, 64],
            detailed_timing: true,
            export_format: ExportFormat::Json,
        }
    }
}

/// Timing measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    pub mean: Duration,
    pub median: Duration,
    pub min: Duration,
    pub max: Duration,
    pub std_dev: Duration,
    pub percentile_95: Duration,
    pub percentile_99: Duration,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub memory_growth_mb: f64,
    pub allocations_count: Option<usize>,
    pub deallocations_count: Option<usize>,
}

/// Throughput measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputStats {
    pub tokens_per_second: f64,
    pub characters_per_second: f64,
    pub batches_per_second: f64,
    pub peak_throughput: f64,
    pub average_throughput: f64,
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub tokenizer_name: String,
    pub text_length: usize,
    pub batch_size: usize,
    pub thread_count: usize,
    pub timing: TimingStats,
    pub memory: Option<MemoryStats>,
    pub throughput: Option<ThroughputStats>,
    pub error_rate: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Complete profiling session results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingReport {
    pub config: ProfilerConfig,
    pub benchmarks: Vec<BenchmarkResult>,
    pub summary: ProfilingSummary,
    pub comparisons: Vec<TokenizerComparison>,
    pub recommendations: Vec<String>,
    pub timestamp: String,
}

/// Summary statistics across all benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingSummary {
    pub total_benchmarks: usize,
    pub fastest_tokenizer: String,
    pub most_memory_efficient: String,
    pub highest_throughput: String,
    pub most_consistent: String,
    pub overall_stats: HashMap<String, f64>,
}

/// Comparison between tokenizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerComparison {
    pub scenario: String,
    pub results: HashMap<String, BenchmarkResult>,
    pub winner: String,
    pub performance_gap: f64,
}

/// Performance profiler implementation
pub struct PerformanceProfiler {
    config: ProfilerConfig,
    results: Vec<BenchmarkResult>,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Create profiler with default configuration
    pub fn default() -> Self {
        Self::new(ProfilerConfig::default())
    }

    /// Profile a single tokenizer
    pub fn profile_tokenizer<T: Tokenizer + Sync>(
        &mut self,
        name: &str,
        tokenizer: &T,
        test_texts: &[String],
    ) -> Result<Vec<BenchmarkResult>> {
        let mut tokenizer_results = Vec::new();

        for &text_length in &self.config.text_lengths {
            for &batch_size in &self.config.batch_sizes {
                // Prepare test data
                let texts = self.prepare_test_texts(test_texts, text_length, batch_size);

                // Run benchmark
                let result =
                    self.benchmark_scenario(name, tokenizer, &texts, text_length, batch_size)?;

                tokenizer_results.push(result.clone());
                self.results.push(result);
            }
        }

        Ok(tokenizer_results)
    }

    /// Profile multiple tokenizers
    pub fn profile_multiple<T: Tokenizer + Sync>(
        &mut self,
        tokenizers: HashMap<String, &T>,
        test_texts: &[String],
    ) -> Result<ProfilingReport> {
        for (name, tokenizer) in tokenizers {
            self.profile_tokenizer(&name, tokenizer, test_texts)?;
        }

        self.generate_report()
    }

    /// Benchmark a specific scenario
    fn benchmark_scenario<T: Tokenizer + Sync>(
        &self,
        name: &str,
        tokenizer: &T,
        texts: &[String],
        text_length: usize,
        batch_size: usize,
    ) -> Result<BenchmarkResult> {
        let thread_count = self.config.concurrent_threads.unwrap_or(1);

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = self.run_tokenization(tokenizer, texts)?;
        }

        // Collect timing measurements
        let mut timings = Vec::new();
        let mut error_count = 0;
        let start_memory = self.get_memory_usage();

        for _ in 0..self.config.benchmark_iterations {
            let start = Instant::now();
            match self.run_tokenization(tokenizer, texts) {
                Ok(_) => {
                    let duration = start.elapsed();
                    timings.push(duration);
                },
                Err(_) => {
                    error_count += 1;
                    timings.push(Duration::from_millis(u64::MAX)); // Mark as failed
                },
            }
        }

        let end_memory = self.get_memory_usage();
        let error_rate = error_count as f64 / self.config.benchmark_iterations as f64;

        // Calculate statistics
        let timing = self.calculate_timing_stats(&timings);
        let memory = if self.config.measure_memory {
            Some(MemoryStats {
                peak_memory_mb: end_memory,
                average_memory_mb: (start_memory + end_memory) / 2.0,
                memory_growth_mb: end_memory - start_memory,
                allocations_count: None,
                deallocations_count: None,
            })
        } else {
            None
        };

        let throughput = if self.config.measure_throughput {
            Some(self.calculate_throughput_stats(texts, &timings, batch_size))
        } else {
            None
        };

        Ok(BenchmarkResult {
            tokenizer_name: name.to_string(),
            text_length,
            batch_size,
            thread_count,
            timing,
            memory,
            throughput,
            error_rate,
            metadata: HashMap::new(),
        })
    }

    /// Run tokenization on texts
    fn run_tokenization<T: Tokenizer>(
        &self,
        tokenizer: &T,
        texts: &[String],
    ) -> Result<Vec<TokenizedInput>> {
        let mut results = Vec::new();
        for text in texts {
            let result = tokenizer.encode(text)?;
            results.push(result);
        }
        Ok(results)
    }

    /// Prepare test texts for benchmarking
    fn prepare_test_texts(
        &self,
        source_texts: &[String],
        target_length: usize,
        count: usize,
    ) -> Vec<String> {
        let mut texts = Vec::new();
        let mut text_pool = source_texts.iter().cycle();

        for _ in 0..count {
            let mut combined_text = String::new();

            while combined_text.len() < target_length {
                if let Some(text) = text_pool.next() {
                    combined_text.push_str(text);
                    combined_text.push(' ');
                } else {
                    break;
                }
            }

            // Truncate to exact length
            if combined_text.len() > target_length {
                combined_text.truncate(target_length);
            }

            texts.push(combined_text);
        }

        texts
    }

    /// Calculate timing statistics
    fn calculate_timing_stats(&self, timings: &[Duration]) -> TimingStats {
        let mut valid_timings: Vec<Duration> = timings
            .iter()
            .filter(|&&t| t != Duration::from_millis(u64::MAX))
            .copied()
            .collect();

        valid_timings.sort();

        if valid_timings.is_empty() {
            return TimingStats {
                mean: Duration::ZERO,
                median: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                std_dev: Duration::ZERO,
                percentile_95: Duration::ZERO,
                percentile_99: Duration::ZERO,
            };
        }

        let sum: Duration = valid_timings.iter().sum();
        let mean = sum / valid_timings.len() as u32;

        let median = valid_timings[valid_timings.len() / 2];
        let min = valid_timings[0];
        let max = valid_timings[valid_timings.len() - 1];

        // Calculate standard deviation
        let variance: f64 = valid_timings
            .iter()
            .map(|&t| {
                let diff = t.as_nanos() as f64 - mean.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>()
            / valid_timings.len() as f64;

        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        let p95_idx = (valid_timings.len() as f64 * 0.95) as usize;
        let p99_idx = (valid_timings.len() as f64 * 0.99) as usize;

        let percentile_95 = valid_timings.get(p95_idx).copied().unwrap_or(max);
        let percentile_99 = valid_timings.get(p99_idx).copied().unwrap_or(max);

        TimingStats {
            mean,
            median,
            min,
            max,
            std_dev,
            percentile_95,
            percentile_99,
        }
    }

    /// Calculate throughput statistics
    fn calculate_throughput_stats(
        &self,
        texts: &[String],
        timings: &[Duration],
        batch_size: usize,
    ) -> ThroughputStats {
        let total_chars: usize = texts.iter().map(|t| t.len()).sum();
        let total_tokens = texts.len() * batch_size; // Approximate

        let valid_timings: Vec<Duration> = timings
            .iter()
            .filter(|&&t| t != Duration::from_millis(u64::MAX))
            .copied()
            .collect();

        if valid_timings.is_empty() {
            return ThroughputStats {
                tokens_per_second: 0.0,
                characters_per_second: 0.0,
                batches_per_second: 0.0,
                peak_throughput: 0.0,
                average_throughput: 0.0,
            };
        }

        let throughputs: Vec<f64> = valid_timings
            .iter()
            .map(|&duration| {
                if duration.as_secs_f64() > 0.0 {
                    total_tokens as f64 / duration.as_secs_f64()
                } else {
                    0.0
                }
            })
            .collect();

        let average_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let peak_throughput = throughputs
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(0.0);

        let avg_duration = valid_timings.iter().sum::<Duration>() / valid_timings.len() as u32;
        let tokens_per_second = if avg_duration.as_secs_f64() > 0.0 {
            total_tokens as f64 / avg_duration.as_secs_f64()
        } else {
            0.0
        };

        let characters_per_second = if avg_duration.as_secs_f64() > 0.0 {
            total_chars as f64 / avg_duration.as_secs_f64()
        } else {
            0.0
        };

        let batches_per_second = if avg_duration.as_secs_f64() > 0.0 {
            1.0 / avg_duration.as_secs_f64()
        } else {
            0.0
        };

        ThroughputStats {
            tokens_per_second,
            characters_per_second,
            batches_per_second,
            peak_throughput,
            average_throughput,
        }
    }

    /// Get current memory usage (simplified)
    fn get_memory_usage(&self) -> f64 {
        // This is a simplified implementation
        // In a real implementation, you'd use platform-specific APIs
        // or libraries like `memory-stats` for accurate memory measurement
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
                for line in contents.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<f64>() {
                                return kb / 1024.0; // Convert to MB
                            }
                        }
                    }
                }
            }
        }

        // Fallback: return 0 if we can't measure memory
        0.0
    }

    /// Generate profiling report
    fn generate_report(&self) -> Result<ProfilingReport> {
        let summary = self.generate_summary();
        let comparisons = self.generate_comparisons();
        let recommendations = self.generate_recommendations();

        Ok(ProfilingReport {
            config: self.config.clone(),
            benchmarks: self.results.clone(),
            summary,
            comparisons,
            recommendations,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Generate summary statistics
    fn generate_summary(&self) -> ProfilingSummary {
        if self.results.is_empty() {
            return ProfilingSummary {
                total_benchmarks: 0,
                fastest_tokenizer: "N/A".to_string(),
                most_memory_efficient: "N/A".to_string(),
                highest_throughput: "N/A".to_string(),
                most_consistent: "N/A".to_string(),
                overall_stats: HashMap::new(),
            };
        }

        // Find fastest tokenizer (lowest mean time)
        let fastest = self
            .results
            .iter()
            .min_by(|a, b| {
                a.timing.mean.partial_cmp(&b.timing.mean).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.tokenizer_name.clone())
            .unwrap_or_else(|| "N/A".to_string());

        // Find most memory efficient
        let most_memory_efficient = self
            .results
            .iter()
            .filter_map(|r| r.memory.as_ref().map(|m| (r, m.peak_memory_mb)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(r, _)| r.tokenizer_name.clone())
            .unwrap_or_else(|| "N/A".to_string());

        // Find highest throughput
        let highest_throughput = self
            .results
            .iter()
            .filter_map(|r| r.throughput.as_ref().map(|t| (r, t.peak_throughput)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(r, _)| r.tokenizer_name.clone())
            .unwrap_or_else(|| "N/A".to_string());

        // Find most consistent (lowest std deviation)
        let most_consistent = self
            .results
            .iter()
            .min_by(|a, b| {
                a.timing
                    .std_dev
                    .partial_cmp(&b.timing.std_dev)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.tokenizer_name.clone())
            .unwrap_or_else(|| "N/A".to_string());

        // Calculate overall statistics
        let mut overall_stats = HashMap::new();
        let total_time: Duration = self.results.iter().map(|r| r.timing.mean).sum();
        overall_stats.insert(
            "total_benchmark_time_ms".to_string(),
            total_time.as_millis() as f64,
        );

        let avg_throughput = self
            .results
            .iter()
            .filter_map(|r| r.throughput.as_ref())
            .map(|t| t.average_throughput)
            .sum::<f64>()
            / self.results.len() as f64;
        overall_stats.insert("average_throughput".to_string(), avg_throughput);

        ProfilingSummary {
            total_benchmarks: self.results.len(),
            fastest_tokenizer: fastest,
            most_memory_efficient,
            highest_throughput,
            most_consistent,
            overall_stats,
        }
    }

    /// Generate tokenizer comparisons
    fn generate_comparisons(&self) -> Vec<TokenizerComparison> {
        let mut comparisons = Vec::new();

        // Group results by scenario (text_length + batch_size)
        let mut scenarios: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in &self.results {
            let scenario = format!("length_{}_batch_{}", result.text_length, result.batch_size);
            scenarios.entry(scenario).or_default().push(result);
        }

        for (scenario, results) in scenarios {
            if results.len() > 1 {
                let mut scenario_results = HashMap::new();
                for result in &results {
                    scenario_results.insert(result.tokenizer_name.clone(), (*result).clone());
                }

                // Find winner (fastest)
                let winner = results
                    .iter()
                    .min_by(|a, b| {
                        a.timing
                            .mean
                            .partial_cmp(&b.timing.mean)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|r| r.tokenizer_name.clone())
                    .unwrap_or_else(|| "N/A".to_string());

                // Calculate performance gap
                let fastest_time =
                    results.iter().map(|r| r.timing.mean.as_millis()).min().unwrap_or(0);
                let slowest_time =
                    results.iter().map(|r| r.timing.mean.as_millis()).max().unwrap_or(0);

                let performance_gap = if fastest_time > 0 {
                    (slowest_time as f64 / fastest_time as f64) - 1.0
                } else {
                    0.0
                };

                comparisons.push(TokenizerComparison {
                    scenario,
                    results: scenario_results,
                    winner,
                    performance_gap,
                });
            }
        }

        comparisons
    }

    /// Generate recommendations based on results
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.results.is_empty() {
            return recommendations;
        }

        // Analyze error rates
        let high_error_rate = self.results.iter().any(|r| r.error_rate > 0.1);
        if high_error_rate {
            recommendations
                .push("Consider investigating tokenizers with high error rates (>10%)".to_string());
        }

        // Analyze memory usage
        if let Some(max_memory) = self
            .results
            .iter()
            .filter_map(|r| r.memory.as_ref())
            .map(|m| m.peak_memory_mb)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            if max_memory > 1000.0 {
                recommendations.push(
                    "Consider using memory-efficient tokenizers for large-scale processing"
                        .to_string(),
                );
            }
        }

        // Analyze consistency
        let high_variance = self
            .results
            .iter()
            .any(|r| r.timing.std_dev.as_millis() > r.timing.mean.as_millis() / 2);
        if high_variance {
            recommendations.push(
                "Some tokenizers show high timing variance - consider warmup strategies"
                    .to_string(),
            );
        }

        // Analyze throughput
        let throughputs: Vec<f64> = self
            .results
            .iter()
            .filter_map(|r| r.throughput.as_ref())
            .map(|t| t.average_throughput)
            .collect();
        if !throughputs.is_empty() {
            let max_throughput = throughputs
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
                .unwrap_or(0.0);
            let min_throughput = throughputs
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
                .unwrap_or(0.0);

            if max_throughput > min_throughput * 2.0 {
                recommendations.push("Significant throughput differences detected - choose tokenizer based on use case".to_string());
            }
        }

        recommendations
    }

    /// Export report to different formats
    pub fn export_report(&self, report: &ProfilingReport, format: ExportFormat) -> Result<String> {
        match format {
            ExportFormat::Json => self.export_json(report),
            ExportFormat::Csv => self.export_csv(report),
            ExportFormat::Html => self.export_html(report),
            ExportFormat::Markdown => self.export_markdown(report),
        }
    }

    /// Export to JSON
    fn export_json(&self, report: &ProfilingReport) -> Result<String> {
        serde_json::to_string_pretty(report).map_err(|e| {
            TrustformersError::other(
                anyhow::anyhow!("Failed to serialize to JSON: {}", e).to_string(),
            )
        })
    }

    /// Export to CSV
    fn export_csv(&self, report: &ProfilingReport) -> Result<String> {
        let mut csv = String::new();
        csv.push_str(
            "tokenizer_name,text_length,batch_size,mean_time_ms,memory_mb,throughput,error_rate\n",
        );

        for benchmark in &report.benchmarks {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                benchmark.tokenizer_name,
                benchmark.text_length,
                benchmark.batch_size,
                benchmark.timing.mean.as_millis(),
                benchmark.memory.as_ref().map(|m| m.peak_memory_mb).unwrap_or(0.0),
                benchmark.throughput.as_ref().map(|t| t.average_throughput).unwrap_or(0.0),
                benchmark.error_rate
            ));
        }

        Ok(csv)
    }

    /// Export to HTML
    fn export_html(&self, report: &ProfilingReport) -> Result<String> {
        let mut html = String::new();
        html.push_str(
            "<!DOCTYPE html>\n<html>\n<head>\n<title>Tokenizer Performance Report</title>\n",
        );
        html.push_str("<style>body{font-family:Arial,sans-serif;margin:40px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background-color:#f2f2f2;}</style>\n");
        html.push_str("</head>\n<body>\n");
        html.push_str("<h1>Tokenizer Performance Report</h1>\n");

        html.push_str("<h2>Summary</h2>\n");
        html.push_str("<table>\n");
        html.push_str(&format!(
            "<tr><td>Total Benchmarks</td><td>{}</td></tr>\n",
            report.summary.total_benchmarks
        ));
        html.push_str(&format!(
            "<tr><td>Fastest Tokenizer</td><td>{}</td></tr>\n",
            report.summary.fastest_tokenizer
        ));
        html.push_str(&format!(
            "<tr><td>Most Memory Efficient</td><td>{}</td></tr>\n",
            report.summary.most_memory_efficient
        ));
        html.push_str(&format!(
            "<tr><td>Highest Throughput</td><td>{}</td></tr>\n",
            report.summary.highest_throughput
        ));
        html.push_str("</table>\n");

        html.push_str("<h2>Detailed Results</h2>\n");
        html.push_str("<table>\n");
        html.push_str("<tr><th>Tokenizer</th><th>Text Length</th><th>Batch Size</th><th>Mean Time (ms)</th><th>Memory (MB)</th><th>Throughput</th></tr>\n");

        for benchmark in &report.benchmarks {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.1}</td><td>{:.1}</td></tr>\n",
                benchmark.tokenizer_name,
                benchmark.text_length,
                benchmark.batch_size,
                benchmark.timing.mean.as_millis(),
                benchmark.memory.as_ref().map(|m| m.peak_memory_mb).unwrap_or(0.0),
                benchmark.throughput.as_ref().map(|t| t.average_throughput).unwrap_or(0.0)
            ));
        }

        html.push_str("</table>\n</body>\n</html>");
        Ok(html)
    }

    /// Export to Markdown
    fn export_markdown(&self, report: &ProfilingReport) -> Result<String> {
        let mut md = String::new();
        md.push_str("# Tokenizer Performance Report\n\n");

        md.push_str("## Summary\n\n");
        md.push_str(&format!(
            "- **Total Benchmarks**: {}\n",
            report.summary.total_benchmarks
        ));
        md.push_str(&format!(
            "- **Fastest Tokenizer**: {}\n",
            report.summary.fastest_tokenizer
        ));
        md.push_str(&format!(
            "- **Most Memory Efficient**: {}\n",
            report.summary.most_memory_efficient
        ));
        md.push_str(&format!(
            "- **Highest Throughput**: {}\n\n",
            report.summary.highest_throughput
        ));

        md.push_str("## Detailed Results\n\n");
        md.push_str("| Tokenizer | Text Length | Batch Size | Mean Time (ms) | Memory (MB) | Throughput |\n");
        md.push_str("|-----------|-------------|------------|----------------|-------------|------------|\n");

        for benchmark in &report.benchmarks {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {:.1} | {:.1} |\n",
                benchmark.tokenizer_name,
                benchmark.text_length,
                benchmark.batch_size,
                benchmark.timing.mean.as_millis(),
                benchmark.memory.as_ref().map(|m| m.peak_memory_mb).unwrap_or(0.0),
                benchmark.throughput.as_ref().map(|t| t.average_throughput).unwrap_or(0.0)
            ));
        }

        if !report.recommendations.is_empty() {
            md.push_str("\n## Recommendations\n\n");
            for (i, rec) in report.recommendations.iter().enumerate() {
                md.push_str(&format!("{}. {}\n", i + 1, rec));
            }
        }

        Ok(md)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::char::CharTokenizer;
    use std::collections::HashMap;

    fn create_test_char_tokenizer() -> CharTokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[CLS]".to_string(), 2);
        vocab.insert("[SEP]".to_string(), 3);
        vocab.insert("h".to_string(), 4);
        vocab.insert("e".to_string(), 5);
        vocab.insert("l".to_string(), 6);
        vocab.insert("o".to_string(), 7);
        vocab.insert("w".to_string(), 8);
        vocab.insert("r".to_string(), 9);
        vocab.insert("d".to_string(), 10);
        vocab.insert(" ".to_string(), 11);
        vocab.insert("t".to_string(), 12);
        vocab.insert("s".to_string(), 13);
        CharTokenizer::new(vocab)
    }

    #[test]
    fn test_profiler_creation() {
        let config = ProfilerConfig::default();
        let profiler = PerformanceProfiler::new(config);
        assert_eq!(profiler.results.len(), 0);
    }

    #[test]
    fn test_single_tokenizer_profiling() {
        let mut profiler = PerformanceProfiler::new(ProfilerConfig {
            warmup_iterations: 1,
            benchmark_iterations: 2,
            text_lengths: vec![10],
            batch_sizes: vec![1],
            ..Default::default()
        });

        let tokenizer = create_test_char_tokenizer();
        let test_texts = vec!["Hello world!".to_string()];

        let results = profiler.profile_tokenizer("char", &tokenizer, &test_texts).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tokenizer_name, "char");
    }

    #[test]
    fn test_timing_stats_calculation() {
        let profiler = PerformanceProfiler::default();
        let timings = vec![
            Duration::from_millis(100),
            Duration::from_millis(110),
            Duration::from_millis(90),
            Duration::from_millis(105),
        ];

        let stats = profiler.calculate_timing_stats(&timings);
        assert!(stats.mean.as_millis() > 0);
        assert!(stats.min <= stats.median);
        assert!(stats.median <= stats.max);
    }

    #[test]
    fn test_report_generation() {
        let mut profiler = PerformanceProfiler::new(ProfilerConfig {
            warmup_iterations: 1,
            benchmark_iterations: 1,
            text_lengths: vec![5],
            batch_sizes: vec![1],
            ..Default::default()
        });

        let tokenizer = create_test_char_tokenizer();
        let test_texts = vec!["Hi".to_string()];

        profiler.profile_tokenizer("test", &tokenizer, &test_texts).unwrap();
        let report = profiler.generate_report().unwrap();

        assert_eq!(report.benchmarks.len(), 1);
        assert_eq!(report.summary.total_benchmarks, 1);
    }

    #[test]
    fn test_export_formats() {
        let profiler = PerformanceProfiler::default();
        let report = ProfilingReport {
            config: ProfilerConfig::default(),
            benchmarks: vec![],
            summary: ProfilingSummary {
                total_benchmarks: 0,
                fastest_tokenizer: "test".to_string(),
                most_memory_efficient: "test".to_string(),
                highest_throughput: "test".to_string(),
                most_consistent: "test".to_string(),
                overall_stats: HashMap::new(),
            },
            comparisons: vec![],
            recommendations: vec![],
            timestamp: "2023-01-01T00:00:00Z".to_string(),
        };

        let json = profiler.export_report(&report, ExportFormat::Json).unwrap();
        assert!(json.contains("fastest_tokenizer"));

        let csv = profiler.export_report(&report, ExportFormat::Csv).unwrap();
        assert!(csv.contains("tokenizer_name"));

        let html = profiler.export_report(&report, ExportFormat::Html).unwrap();
        assert!(html.contains("<html>"));

        let md = profiler.export_report(&report, ExportFormat::Markdown).unwrap();
        assert!(md.contains("# Tokenizer Performance Report"));
    }
}
