#!/usr/bin/env python3
"""
Benchmark comparison script for TrustformeRS CI/CD pipeline.

This script compares benchmark results between a baseline (main branch) and 
current PR to detect performance regressions.
"""

import json
import sys
import argparse
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import statistics

class BenchmarkComparison:
    """Compare benchmark results and detect regressions."""
    
    def __init__(self, regression_threshold: float = 0.05):
        """
        Initialize benchmark comparison.
        
        Args:
            regression_threshold: Threshold for considering a regression (5% by default)
        """
        self.regression_threshold = regression_threshold
        
    def load_benchmark_data(self, filepath: str) -> Dict:
        """Load benchmark data from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading benchmark data from {filepath}: {e}")
            return {}
    
    def extract_benchmark_metrics(self, data: Dict) -> Dict[str, float]:
        """Extract relevant metrics from benchmark data."""
        metrics = {}
        
        # Handle different benchmark formats
        if 'benchmarks' in data:
            # Criterion format
            for bench in data['benchmarks']:
                name = bench.get('name', 'unknown')
                if 'mean' in bench:
                    # Extract mean execution time
                    if isinstance(bench['mean'], dict):
                        metrics[name] = bench['mean'].get('point_estimate', 0.0)
                    else:
                        metrics[name] = bench['mean']
                elif 'value' in bench:
                    metrics[name] = bench['value']
        
        elif isinstance(data, dict):
            # Simple key-value format
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
                elif isinstance(value, dict) and 'time' in value:
                    metrics[key] = float(value['time'])
        
        return metrics
    
    def compare_benchmarks(
        self, 
        baseline_metrics: Dict[str, float], 
        current_metrics: Dict[str, float]
    ) -> Dict[str, Dict]:
        """
        Compare benchmark metrics between baseline and current.
        
        Returns:
            Dictionary containing comparison results for each benchmark
        """
        results = {}
        
        for bench_name in current_metrics.keys():
            if bench_name not in baseline_metrics:
                results[bench_name] = {
                    'status': 'new',
                    'baseline': None,
                    'current': current_metrics[bench_name],
                    'change_ratio': None,
                    'change_percent': None
                }
                continue
            
            baseline_value = baseline_metrics[bench_name]
            current_value = current_metrics[bench_name]
            
            if baseline_value == 0:
                change_ratio = float('inf') if current_value > 0 else 0
                change_percent = float('inf') if current_value > 0 else 0
            else:
                change_ratio = current_value / baseline_value
                change_percent = ((current_value - baseline_value) / baseline_value) * 100
            
            # Determine status
            if abs(change_percent) <= self.regression_threshold * 100:
                status = 'stable'
            elif change_percent > 0:
                status = 'regression'  # Slower is worse for execution time
            else:
                status = 'improvement'  # Faster is better
            
            results[bench_name] = {
                'status': status,
                'baseline': baseline_value,
                'current': current_value,
                'change_ratio': change_ratio,
                'change_percent': change_percent
            }
        
        # Check for removed benchmarks
        for bench_name in baseline_metrics.keys():
            if bench_name not in current_metrics:
                results[bench_name] = {
                    'status': 'removed',
                    'baseline': baseline_metrics[bench_name],
                    'current': None,
                    'change_ratio': None,
                    'change_percent': None
                }
        
        return results
    
    def format_time(self, nanoseconds: float) -> str:
        """Format time in a human-readable way."""
        if nanoseconds < 1_000:
            return f"{nanoseconds:.2f} ns"
        elif nanoseconds < 1_000_000:
            return f"{nanoseconds / 1_000:.2f} μs"
        elif nanoseconds < 1_000_000_000:
            return f"{nanoseconds / 1_000_000:.2f} ms"
        else:
            return f"{nanoseconds / 1_000_000_000:.2f} s"
    
    def generate_report(self, comparison_results: Dict[str, Dict]) -> str:
        """Generate a human-readable report."""
        report = []
        report.append("# Benchmark Comparison Report")
        report.append("")
        
        # Count results by status
        status_counts = {}
        for result in comparison_results.values():
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        report.append("## Summary")
        report.append("")
        for status, count in sorted(status_counts.items()):
            emoji = {
                'stable': '✅',
                'improvement': '🚀',
                'regression': '⚠️',
                'new': '🆕',
                'removed': '❌'
            }.get(status, '❓')
            report.append(f"- {emoji} **{status.title()}**: {count} benchmarks")
        
        report.append("")
        
        # Detailed results
        if any(r['status'] == 'regression' for r in comparison_results.values()):
            report.append("## ⚠️ Performance Regressions")
            report.append("")
            report.append("| Benchmark | Baseline | Current | Change |")
            report.append("|-----------|----------|---------|---------|")
            
            for name, result in comparison_results.items():
                if result['status'] == 'regression':
                    baseline_str = self.format_time(result['baseline']) if result['baseline'] else 'N/A'
                    current_str = self.format_time(result['current']) if result['current'] else 'N/A'
                    change_str = f"+{result['change_percent']:.1f}%" if result['change_percent'] else 'N/A'
                    report.append(f"| `{name}` | {baseline_str} | {current_str} | {change_str} |")
            
            report.append("")
        
        if any(r['status'] == 'improvement' for r in comparison_results.values()):
            report.append("## 🚀 Performance Improvements")
            report.append("")
            report.append("| Benchmark | Baseline | Current | Change |")
            report.append("|-----------|----------|---------|---------|")
            
            for name, result in comparison_results.items():
                if result['status'] == 'improvement':
                    baseline_str = self.format_time(result['baseline']) if result['baseline'] else 'N/A'
                    current_str = self.format_time(result['current']) if result['current'] else 'N/A'
                    change_str = f"{result['change_percent']:.1f}%" if result['change_percent'] else 'N/A'
                    report.append(f"| `{name}` | {baseline_str} | {current_str} | {change_str} |")
            
            report.append("")
        
        if any(r['status'] == 'new' for r in comparison_results.values()):
            report.append("## 🆕 New Benchmarks")
            report.append("")
            for name, result in comparison_results.items():
                if result['status'] == 'new':
                    current_str = self.format_time(result['current']) if result['current'] else 'N/A'
                    report.append(f"- `{name}`: {current_str}")
            
            report.append("")
        
        if any(r['status'] == 'removed' for r in comparison_results.values()):
            report.append("## ❌ Removed Benchmarks")
            report.append("")
            for name, result in comparison_results.items():
                if result['status'] == 'removed':
                    baseline_str = self.format_time(result['baseline']) if result['baseline'] else 'N/A'
                    report.append(f"- `{name}`: was {baseline_str}")
            
            report.append("")
        
        return "\n".join(report)
    
    def check_regressions(self, comparison_results: Dict[str, Dict]) -> bool:
        """Check if there are any significant regressions."""
        return any(r['status'] == 'regression' for r in comparison_results.values())

def main():
    parser = argparse.ArgumentParser(description='Compare benchmark results')
    parser.add_argument('baseline', help='Path to baseline benchmark results')
    parser.add_argument('current', help='Path to current benchmark results')
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.05,
        help='Regression threshold (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--fail-on-regression',
        action='store_true',
        help='Exit with error code if regressions are found'
    )
    parser.add_argument(
        '--output',
        help='Output file for the report (default: stdout)'
    )
    
    args = parser.parse_args()
    
    # Initialize comparison
    comparator = BenchmarkComparison(args.threshold)
    
    # Load benchmark data
    print(f"Loading baseline data from {args.baseline}...")
    baseline_data = comparator.load_benchmark_data(args.baseline)
    if not baseline_data:
        print("Failed to load baseline data")
        return 1
    
    print(f"Loading current data from {args.current}...")
    current_data = comparator.load_benchmark_data(args.current)
    if not current_data:
        print("Failed to load current data")
        return 1
    
    # Extract metrics
    baseline_metrics = comparator.extract_benchmark_metrics(baseline_data)
    current_metrics = comparator.extract_benchmark_metrics(current_data)
    
    print(f"Found {len(baseline_metrics)} baseline benchmarks")
    print(f"Found {len(current_metrics)} current benchmarks")
    
    # Compare benchmarks
    comparison_results = comparator.compare_benchmarks(baseline_metrics, current_metrics)
    
    # Generate report
    report = comparator.generate_report(comparison_results)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print("\n" + report)
    
    # Check for regressions
    has_regressions = comparator.check_regressions(comparison_results)
    
    if has_regressions:
        print("\n⚠️ Performance regressions detected!")
        if args.fail_on_regression:
            return 1
    else:
        print("\n✅ No performance regressions detected")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())