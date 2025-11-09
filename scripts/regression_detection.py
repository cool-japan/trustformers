#!/usr/bin/env python3
"""
TrustformeRS Performance Regression Detection System

This script analyzes benchmark results and detects performance regressions
by comparing against historical baselines and thresholds.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics


class RegressionDetector:
    """Detects performance regressions in benchmark results."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.results_dir = Path(self.config.get("results_dir", "benchmark_results"))
        self.thresholds = self.config.get("thresholds", {})
        
    def load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "results_dir": "benchmark_results",
            "thresholds": {
                "default": {
                    "warning": 0.05,  # 5% regression triggers warning
                    "error": 0.10,    # 10% regression triggers error
                },
                "tensor_ops": {
                    "warning": 0.03,
                    "error": 0.05,
                },
                "model_inference": {
                    "warning": 0.05,
                    "error": 0.10,
                },
                "memory": {
                    "warning": 0.10,
                    "error": 0.20,
                }
            },
            "historical_window": 10,  # Number of historical runs to consider
            "min_samples": 3,         # Minimum samples for statistical significance
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def parse_criterion_output(self, json_path: str) -> Dict:
        """Parse Criterion benchmark JSON output."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        results = {}
        for benchmark in data.get("benchmarks", []):
            name = benchmark["group"]
            if name not in results:
                results[name] = {}
            
            # Extract timing information
            bench_name = benchmark["benchmark"]
            mean_time = benchmark["mean"]["point_estimate"]
            std_dev = benchmark["std_dev"]["point_estimate"]
            
            results[name][bench_name] = {
                "mean": mean_time,
                "std_dev": std_dev,
                "unit": benchmark["unit"],
            }
            
        return results
    
    def calculate_regression(self, current: float, baseline: float) -> float:
        """Calculate percentage regression (positive means slower)."""
        if baseline == 0:
            return 0.0
        return (current - baseline) / baseline
    
    def get_threshold(self, benchmark_name: str) -> Tuple[float, float]:
        """Get warning and error thresholds for a benchmark."""
        # Check specific thresholds first
        for key, thresholds in self.thresholds.items():
            if key in benchmark_name:
                return thresholds["warning"], thresholds["error"]
        
        # Fall back to default
        default = self.thresholds.get("default", {"warning": 0.05, "error": 0.10})
        return default["warning"], default["error"]
    
    def analyze_regression(self, current_results: Dict, baseline_results: Dict) -> Dict:
        """Analyze regression between current and baseline results."""
        regressions = {
            "errors": [],
            "warnings": [],
            "improvements": [],
            "summary": {}
        }
        
        for group, benchmarks in current_results.items():
            if group not in baseline_results:
                continue
                
            for bench_name, current_data in benchmarks.items():
                if bench_name not in baseline_results[group]:
                    continue
                
                baseline_data = baseline_results[group][bench_name]
                current_mean = current_data["mean"]
                baseline_mean = baseline_data["mean"]
                
                regression = self.calculate_regression(current_mean, baseline_mean)
                warning_threshold, error_threshold = self.get_threshold(f"{group}/{bench_name}")
                
                result = {
                    "benchmark": f"{group}/{bench_name}",
                    "current": current_mean,
                    "baseline": baseline_mean,
                    "regression": regression,
                    "unit": current_data["unit"]
                }
                
                if regression > error_threshold:
                    regressions["errors"].append(result)
                elif regression > warning_threshold:
                    regressions["warnings"].append(result)
                elif regression < -warning_threshold:  # Significant improvement
                    regressions["improvements"].append(result)
                
        # Calculate summary statistics
        all_regressions = [r["regression"] for r in 
                          regressions["errors"] + regressions["warnings"]]
        
        if all_regressions:
            regressions["summary"] = {
                "total_benchmarks": len(current_results),
                "regressions_found": len(all_regressions),
                "max_regression": max(all_regressions),
                "mean_regression": statistics.mean(all_regressions),
                "improvements_found": len(regressions["improvements"])
            }
        
        return regressions
    
    def load_historical_data(self, benchmark_name: str, limit: int = 10) -> List[Dict]:
        """Load historical benchmark data."""
        historical_data = []
        
        # Look for historical results in results directory
        history_file = self.results_dir / f"history_{benchmark_name}.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                data = json.load(f)
                historical_data = data[-limit:]  # Get last N results
        
        return historical_data
    
    def detect_trend(self, historical_data: List[float]) -> Dict:
        """Detect performance trends in historical data."""
        if len(historical_data) < 3:
            return {"trend": "insufficient_data"}
        
        # Simple linear regression to detect trend
        n = len(historical_data)
        x = list(range(n))
        y = historical_data
        
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        den = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if den == 0:
            slope = 0
        else:
            slope = num / den
        
        # Determine trend based on slope
        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0.001:
            trend = "degrading"
        else:
            trend = "improving"
        
        return {
            "trend": trend,
            "slope": slope,
            "samples": n
        }
    
    def generate_report(self, analysis: Dict, output_format: str = "markdown") -> str:
        """Generate regression report in specified format."""
        if output_format == "markdown":
            return self._generate_markdown_report(analysis)
        elif output_format == "json":
            return json.dumps(analysis, indent=2)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    
    def _generate_markdown_report(self, analysis: Dict) -> str:
        """Generate Markdown regression report."""
        report = ["# Performance Regression Report\n"]
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary
        if "summary" in analysis and analysis["summary"]:
            summary = analysis["summary"]
            report.append("## Summary\n")
            report.append(f"- Total benchmarks analyzed: {summary.get('total_benchmarks', 0)}")
            report.append(f"- Regressions found: {summary.get('regressions_found', 0)}")
            report.append(f"- Maximum regression: {summary.get('max_regression', 0):.1%}")
            report.append(f"- Mean regression: {summary.get('mean_regression', 0):.1%}")
            report.append(f"- Improvements found: {summary.get('improvements_found', 0)}\n")
        
        # Errors
        if analysis["errors"]:
            report.append("## 🚨 Critical Regressions\n")
            report.append("| Benchmark | Current | Baseline | Regression | Unit |")
            report.append("|-----------|---------|----------|------------|------|")
            
            for error in analysis["errors"]:
                report.append(
                    f"| {error['benchmark']} | "
                    f"{error['current']:.3f} | "
                    f"{error['baseline']:.3f} | "
                    f"**{error['regression']:.1%}** | "
                    f"{error['unit']} |"
                )
            report.append("")
        
        # Warnings
        if analysis["warnings"]:
            report.append("## ⚠️  Performance Warnings\n")
            report.append("| Benchmark | Current | Baseline | Regression | Unit |")
            report.append("|-----------|---------|----------|------------|------|")
            
            for warning in analysis["warnings"]:
                report.append(
                    f"| {warning['benchmark']} | "
                    f"{warning['current']:.3f} | "
                    f"{warning['baseline']:.3f} | "
                    f"{warning['regression']:.1%} | "
                    f"{warning['unit']} |"
                )
            report.append("")
        
        # Improvements
        if analysis["improvements"]:
            report.append("## ✅ Performance Improvements\n")
            report.append("| Benchmark | Current | Baseline | Improvement | Unit |")
            report.append("|-----------|---------|----------|-------------|------|")
            
            for improvement in analysis["improvements"]:
                report.append(
                    f"| {improvement['benchmark']} | "
                    f"{improvement['current']:.3f} | "
                    f"{improvement['baseline']:.3f} | "
                    f"{abs(improvement['regression']):.1%} | "
                    f"{improvement['unit']} |"
                )
            report.append("")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Detect performance regressions in TrustformeRS benchmarks"
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current benchmark results (JSON)"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline benchmark results (JSON)"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format for report"
    )
    parser.add_argument(
        "--output-file",
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with non-zero status if regressions found"
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RegressionDetector(args.config)
    
    # Load benchmark results
    try:
        current_results = detector.parse_criterion_output(args.current)
        baseline_results = detector.parse_criterion_output(args.baseline)
    except Exception as e:
        print(f"Error loading benchmark results: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Analyze regressions
    analysis = detector.analyze_regression(current_results, baseline_results)
    
    # Generate report
    report = detector.generate_report(analysis, args.output)
    
    # Output report
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(report)
    else:
        print(report)
    
    # Exit with appropriate status
    if args.fail_on_regression and analysis["errors"]:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()