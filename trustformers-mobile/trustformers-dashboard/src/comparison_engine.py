"""
Comparison engine for analyzing performance across models and configurations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import sqlite3
from datetime import datetime, timedelta
import scipy.stats as stats

@dataclass
class ComparisonResult:
    """Result of model comparison analysis."""
    models: List[str]
    winner_by_metric: Dict[str, str]
    relative_performance: Dict[str, Dict[str, float]]
    statistical_significance: Dict[str, Dict[str, float]]
    trade_offs: List[str]
    recommendations: List[str]
    
class ComparisonEngine:
    def __init__(self, db_path: str = "data/metrics.db"):
        self.db_path = db_path
        self.metrics_weights = {
            'latency': 0.3,
            'throughput': 0.3,
            'memory': 0.2,
            'power': 0.2
        }
    
    def compare_models(self, model_names: List[str], 
                      device: str = None,
                      time_window_hours: int = 24) -> ComparisonResult:
        """Compare multiple models across various metrics."""
        
        # Fetch metrics for all models
        metrics_data = self._fetch_metrics(model_names, device, time_window_hours)
        
        if not metrics_data:
            raise ValueError("No metrics data available for comparison")
        
        # Analyze performance
        winner_by_metric = self._determine_winners(metrics_data)
        relative_perf = self._calculate_relative_performance(metrics_data)
        significance = self._statistical_significance_test(metrics_data)
        trade_offs = self._analyze_trade_offs(metrics_data, relative_perf)
        recommendations = self._generate_recommendations(
            metrics_data, relative_perf, trade_offs
        )
        
        return ComparisonResult(
            models=model_names,
            winner_by_metric=winner_by_metric,
            relative_performance=relative_perf,
            statistical_significance=significance,
            trade_offs=trade_offs,
            recommendations=recommendations
        )
    
    def _fetch_metrics(self, model_names: List[str], device: Optional[str],
                      time_window_hours: int) -> Dict[str, Dict[str, List[float]]]:
        """Fetch metrics data from database."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        metrics_data = {}
        
        with sqlite3.connect(self.db_path) as conn:
            for model in model_names:
                metrics_data[model] = {
                    'latency': [],
                    'throughput': [],
                    'memory': [],
                    'power': []
                }
                
                # Build query
                query = '''
                    SELECT metric_type, value 
                    FROM metrics 
                    WHERE model_name = ? 
                    AND timestamp > ?
                '''
                params = [model, cutoff_time]
                
                if device:
                    query += ' AND device = ?'
                    params.append(device)
                
                cursor = conn.execute(query, params)
                
                for row in cursor:
                    metric_type, value = row
                    if metric_type in metrics_data[model]:
                        metrics_data[model][metric_type].append(value)
        
        return metrics_data
    
    def _determine_winners(self, metrics_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, str]:
        """Determine best performing model for each metric."""
        winners = {}
        
        metrics_to_analyze = ['latency', 'throughput', 'memory', 'power']
        
        for metric in metrics_to_analyze:
            scores = {}
            
            for model, data in metrics_data.items():
                if data[metric]:
                    # Use median for robustness
                    if metric in ['latency', 'memory', 'power']:
                        # Lower is better
                        scores[model] = -np.median(data[metric])
                    else:
                        # Higher is better
                        scores[model] = np.median(data[metric])
            
            if scores:
                winner = max(scores.items(), key=lambda x: x[1])[0]
                winners[metric] = winner
        
        return winners
    
    def _calculate_relative_performance(self, 
                                      metrics_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
        """Calculate relative performance compared to baseline (first model)."""
        if not metrics_data:
            return {}
        
        baseline_model = list(metrics_data.keys())[0]
        relative_perf = {}
        
        for model in metrics_data:
            relative_perf[model] = {}
            
            for metric in ['latency', 'throughput', 'memory', 'power']:
                if metrics_data[model][metric] and metrics_data[baseline_model][metric]:
                    model_median = np.median(metrics_data[model][metric])
                    baseline_median = np.median(metrics_data[baseline_model][metric])
                    
                    if baseline_median != 0:
                        if metric in ['latency', 'memory', 'power']:
                            # Lower is better - invert ratio
                            relative_perf[model][metric] = baseline_median / model_median
                        else:
                            # Higher is better
                            relative_perf[model][metric] = model_median / baseline_median
                    else:
                        relative_perf[model][metric] = 1.0
                else:
                    relative_perf[model][metric] = 1.0
        
        return relative_perf
    
    def _statistical_significance_test(self, 
                                     metrics_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
        """Perform statistical significance testing between models."""
        significance = {}
        models = list(metrics_data.keys())
        
        for i, model1 in enumerate(models):
            significance[model1] = {}
            
            for j, model2 in enumerate(models):
                if i != j:
                    p_values = {}
                    
                    for metric in ['latency', 'throughput', 'memory', 'power']:
                        data1 = metrics_data[model1][metric]
                        data2 = metrics_data[model2][metric]
                        
                        if len(data1) >= 5 and len(data2) >= 5:
                            # Perform Mann-Whitney U test (non-parametric)
                            _, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            p_values[metric] = p_value
                        else:
                            p_values[metric] = 1.0  # Not enough data
                    
                    # Overall significance (Bonferroni correction)
                    min_p = min(p_values.values())
                    adjusted_p = min(min_p * len(p_values), 1.0)
                    significance[model1][model2] = adjusted_p
        
        return significance
    
    def _analyze_trade_offs(self, metrics_data: Dict[str, Dict[str, List[float]]],
                           relative_perf: Dict[str, Dict[str, float]]) -> List[str]:
        """Analyze trade-offs between different models."""
        trade_offs = []
        
        # Identify Pareto-optimal models
        pareto_models = self._find_pareto_optimal(relative_perf)
        
        if len(pareto_models) > 1:
            trade_offs.append(
                f"Multiple Pareto-optimal models found: {', '.join(pareto_models)}. "
                "Choice depends on specific requirements."
            )
        
        # Analyze specific trade-offs
        for model in metrics_data:
            perf = relative_perf.get(model, {})
            
            # High performance but high resource usage
            if perf.get('throughput', 1) > 1.5 and perf.get('memory', 1) < 0.7:
                trade_offs.append(
                    f"{model}: High throughput ({perf['throughput']:.1f}x) "
                    f"but high memory usage ({1/perf['memory']:.1f}x)"
                )
            
            # Low latency but high power
            if perf.get('latency', 1) > 1.3 and perf.get('power', 1) < 0.8:
                trade_offs.append(
                    f"{model}: Low latency ({perf['latency']:.1f}x faster) "
                    f"but high power consumption ({1/perf['power']:.1f}x)"
                )
            
            # Efficient but slower
            if perf.get('power', 1) > 1.2 and perf.get('memory', 1) > 1.2 and perf.get('throughput', 1) < 0.8:
                trade_offs.append(
                    f"{model}: Efficient (low power/memory) but "
                    f"lower throughput ({perf['throughput']:.1f}x)"
                )
        
        return trade_offs
    
    def _find_pareto_optimal(self, relative_perf: Dict[str, Dict[str, float]]) -> List[str]:
        """Find Pareto-optimal models (not dominated by any other model)."""
        models = list(relative_perf.keys())
        pareto_optimal = []
        
        for model1 in models:
            is_dominated = False
            
            for model2 in models:
                if model1 != model2:
                    # Check if model2 dominates model1
                    dominates = True
                    for metric in ['latency', 'throughput', 'memory', 'power']:
                        if relative_perf[model1].get(metric, 1) > relative_perf[model2].get(metric, 1):
                            dominates = False
                            break
                    
                    if dominates:
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_optimal.append(model1)
        
        return pareto_optimal
    
    def _generate_recommendations(self, metrics_data: Dict[str, Dict[str, List[float]]],
                                relative_perf: Dict[str, Dict[str, float]],
                                trade_offs: List[str]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Overall best performer
        overall_scores = {}
        for model in relative_perf:
            score = sum(
                relative_perf[model].get(metric, 1) * self.metrics_weights.get(metric, 0.25)
                for metric in ['latency', 'throughput', 'memory', 'power']
            )
            overall_scores[model] = score
        
        if overall_scores:
            best_model = max(overall_scores.items(), key=lambda x: x[1])[0]
            recommendations.append(
                f"Overall recommendation: {best_model} "
                f"(weighted score: {overall_scores[best_model]:.2f})"
            )
        
        # Scenario-specific recommendations
        scenarios = self._analyze_scenarios(relative_perf)
        recommendations.extend(scenarios)
        
        # Optimization opportunities
        for model in metrics_data:
            perf = relative_perf.get(model, {})
            
            # Low utilization
            if perf.get('throughput', 1) < 0.5:
                recommendations.append(
                    f"{model}: Consider optimization - throughput is "
                    f"{perf['throughput']:.1f}x baseline"
                )
            
            # High variability
            for metric in ['latency', 'throughput']:
                if metrics_data[model][metric]:
                    cv = np.std(metrics_data[model][metric]) / np.mean(metrics_data[model][metric])
                    if cv > 0.3:  # Coefficient of variation > 30%
                        recommendations.append(
                            f"{model}: High {metric} variability (CV={cv:.1%}). "
                            "Consider investigating instability."
                        )
        
        return recommendations
    
    def _analyze_scenarios(self, relative_perf: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate scenario-specific recommendations."""
        scenarios = []
        
        # Real-time inference
        latency_scores = {
            model: perf.get('latency', 1) 
            for model, perf in relative_perf.items()
        }
        if latency_scores:
            best_latency = max(latency_scores.items(), key=lambda x: x[1])[0]
            scenarios.append(
                f"For real-time inference: {best_latency} "
                f"(latency: {latency_scores[best_latency]:.1f}x better)"
            )
        
        # Batch processing
        throughput_scores = {
            model: perf.get('throughput', 1) 
            for model, perf in relative_perf.items()
        }
        if throughput_scores:
            best_throughput = max(throughput_scores.items(), key=lambda x: x[1])[0]
            scenarios.append(
                f"For batch processing: {best_throughput} "
                f"(throughput: {throughput_scores[best_throughput]:.1f}x better)"
            )
        
        # Edge deployment
        edge_scores = {}
        for model, perf in relative_perf.items():
            # Weighted score favoring memory and power efficiency
            edge_score = (
                perf.get('memory', 1) * 0.4 +
                perf.get('power', 1) * 0.4 +
                perf.get('latency', 1) * 0.2
            )
            edge_scores[model] = edge_score
        
        if edge_scores:
            best_edge = max(edge_scores.items(), key=lambda x: x[1])[0]
            scenarios.append(
                f"For edge deployment: {best_edge} "
                f"(efficiency score: {edge_scores[best_edge]:.2f})"
            )
        
        return scenarios
    
    def generate_comparison_matrix(self, comparison_result: ComparisonResult) -> pd.DataFrame:
        """Generate a comparison matrix for visualization."""
        models = comparison_result.models
        metrics = ['latency', 'throughput', 'memory', 'power', 'overall']
        
        # Create matrix
        data = []
        for model in models:
            row = {'model': model}
            for metric in metrics:
                if metric == 'overall':
                    # Calculate overall score
                    score = sum(
                        comparison_result.relative_performance[model].get(m, 1) * 
                        self.metrics_weights.get(m, 0.25)
                        for m in ['latency', 'throughput', 'memory', 'power']
                    )
                    row[metric] = score
                else:
                    row[metric] = comparison_result.relative_performance[model].get(metric, 1)
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.set_index('model')
        
        # Normalize to 0-100 scale for better visualization
        for col in df.columns:
            df[col] = (df[col] / df[col].max()) * 100
        
        return df
    
    def export_comparison_report(self, comparison_result: ComparisonResult,
                               output_path: str):
        """Export detailed comparison report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models': comparison_result.models,
            'summary': {
                'winners': comparison_result.winner_by_metric,
                'overall_ranking': self._calculate_overall_ranking(
                    comparison_result.relative_performance
                )
            },
            'detailed_metrics': comparison_result.relative_performance,
            'statistical_analysis': {
                'significance_matrix': comparison_result.statistical_significance,
                'confidence_level': 0.95
            },
            'trade_offs': comparison_result.trade_offs,
            'recommendations': comparison_result.recommendations,
            'metadata': {
                'metrics_weights': self.metrics_weights,
                'analysis_version': '1.0'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _calculate_overall_ranking(self, relative_perf: Dict[str, Dict[str, float]]) -> List[str]:
        """Calculate overall ranking of models."""
        scores = {}
        
        for model in relative_perf:
            score = sum(
                relative_perf[model].get(metric, 1) * self.metrics_weights.get(metric, 0.25)
                for metric in ['latency', 'throughput', 'memory', 'power']
            )
            scores[model] = score
        
        # Sort by score (descending)
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [model for model, _ in ranking]