#!/usr/bin/env python3
"""
TrustformeRS Leaderboard Manager

This script manages benchmark results, generates leaderboards, and provides
tools for comparing performance across different configurations.

Features:
- Collect and aggregate benchmark results
- Generate HTML/Markdown leaderboards
- Compare performance across versions/hardware
- Submit results to community leaderboard
- Performance regression detection
- Historical trend analysis
"""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import statistics
import subprocess
import hashlib

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: 'requests' not available. Community submission disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Plotting libraries not available. Chart generation disabled.")

@dataclass
class LeaderboardEntry:
    """Single entry in the leaderboard"""
    name: str
    version: str
    hardware: Dict[str, Any]
    software: Dict[str, Any]
    results: List[Dict[str, Any]]
    total_score: float
    timestamp: int
    git_hash: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_json_file(cls, filepath: Path) -> 'LeaderboardEntry':
        """Load entry from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            name=data['config']['name'],
            version=data['config']['version'],
            hardware=data['config']['hardware'],
            software=data['config']['software'],
            results=data['results'],
            total_score=data['total_score'],
            timestamp=data['config']['timestamp'],
            git_hash=data['config'].get('git_hash')
        )

class LeaderboardManager:
    """Manager for benchmark leaderboards"""
    
    def __init__(self, data_dir: Path = Path("benchmark_results")):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        
        # Leaderboard API endpoint (example)
        self.api_endpoint = os.environ.get(
            'TRUSTFORMERS_LEADERBOARD_API',
            'https://api.trustformers.ai/leaderboard'
        )
    
    def run_benchmarks(self, config: Dict[str, Any] = None) -> Path:
        """Run the leaderboard benchmarks and return result file path"""
        print("Running TrustformeRS leaderboard benchmarks...")
        
        # Set environment variables for benchmark configuration
        if config:
            for key, value in config.items():
                os.environ[f'TRUSTFORMERS_{key.upper()}'] = str(value)
        
        # Add git hash if available
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                             cwd=Path(__file__).parent.parent,
                                             universal_newlines=True).strip()
            os.environ['TRUSTFORMERS_GIT_HASH'] = git_hash
        except subprocess.CalledProcessError:
            pass
        
        # Run the benchmark
        cmd = ['cargo', 'bench', '--bench', 'leaderboard_bench']
        
        if config and config.get('features'):
            features = config['features']
            if isinstance(features, list):
                features = ','.join(features)
            cmd.extend(['--features', features])
        
        try:
            subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
        except subprocess.CalledProcessError as e:
            print(f"Benchmark failed: {e}")
            return None
        
        # Generate timestamped result file
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        result_file = self.data_dir / f"leaderboard_{timestamp}.json"
        
        # In a real implementation, this would extract results from criterion output
        # For now, we'll create a mock result file
        mock_result = self._generate_mock_result()
        
        with open(result_file, 'w') as f:
            json.dump(mock_result, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        return result_file
    
    def _generate_mock_result(self) -> Dict:
        """Generate mock result for demonstration"""
        import platform
        import psutil if 'psutil' in sys.modules else None
        
        return {
            'config': {
                'name': os.environ.get('TRUSTFORMERS_BENCHMARK_NAME', 'TrustformeRS Benchmark'),
                'version': self._get_trustformers_version(),
                'hardware': {
                    'cpu_model': platform.processor() or 'Unknown CPU',
                    'cpu_cores': os.cpu_count() or 1,
                    'memory_gb': self._get_memory_gb(),
                    'gpu_model': os.environ.get('TRUSTFORMERS_GPU_MODEL'),
                    'gpu_memory_gb': self._parse_env_int('TRUSTFORMERS_GPU_MEMORY_GB'),
                    'platform': platform.machine()
                },
                'software': {
                    'rust_version': self._get_rust_version(),
                    'trustformers_version': self._get_trustformers_version(),
                    'backend': os.environ.get('TRUSTFORMERS_BACKEND', 'CPU'),
                    'compiler_flags': os.environ.get('RUSTFLAGS', '').split(),
                    'features': os.environ.get('TRUSTFORMERS_FEATURES', '').split(',')
                },
                'timestamp': int(datetime.now(timezone.utc).timestamp()),
                'git_hash': os.environ.get('TRUSTFORMERS_GIT_HASH')
            },
            'results': [
                {
                    'name': 'matmul_1024x1024',
                    'category': 'tensor_ops',
                    'throughput': 1000000.0,
                    'latency': 1.0,
                    'memory_mb': 16.0,
                    'accuracy': None,
                    'energy': None,
                    'metadata': {'precision': 'fp32'}
                },
                {
                    'name': 'gpt2_small_inference',
                    'category': 'model_inference',
                    'throughput': 500.0,
                    'latency': 50.0,
                    'memory_mb': 512.0,
                    'accuracy': 0.85,
                    'energy': 10.0,
                    'metadata': {'model_size': '117M', 'sequence_length': '512'}
                }
            ],
            'total_score': 150.5
        }
    
    def load_results(self, pattern: str = "leaderboard_*.json") -> List[LeaderboardEntry]:
        """Load all benchmark results matching pattern"""
        result_files = list(self.data_dir.glob(pattern))
        entries = []
        
        for file_path in sorted(result_files):
            try:
                entry = LeaderboardEntry.from_json_file(file_path)
                entries.append(entry)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        
        return entries
    
    def generate_leaderboard_html(self, entries: List[LeaderboardEntry], 
                                output_path: Path) -> None:
        """Generate HTML leaderboard"""
        # Sort by total score (descending)
        sorted_entries = sorted(entries, key=lambda x: x.total_score, reverse=True)
        
        html_content = self._generate_html_template(sorted_entries)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML leaderboard generated: {output_path}")
    
    def generate_leaderboard_markdown(self, entries: List[LeaderboardEntry], 
                                    output_path: Path) -> None:
        """Generate Markdown leaderboard"""
        # Sort by total score (descending)
        sorted_entries = sorted(entries, key=lambda x: x.total_score, reverse=True)
        
        md_content = self._generate_markdown_template(sorted_entries)
        
        with open(output_path, 'w') as f:
            f.write(md_content)
        
        print(f"Markdown leaderboard generated: {output_path}")
    
    def compare_versions(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """Compare performance between two versions"""
        entries = self.load_results()
        
        entries_a = [e for e in entries if e.version == version_a]
        entries_b = [e for e in entries if e.version == version_b]
        
        if not entries_a or not entries_b:
            print(f"Warning: Insufficient data for comparison")
            return {}
        
        comparison = {
            'version_a': version_a,
            'version_b': version_b,
            'score_a': statistics.mean(e.total_score for e in entries_a),
            'score_b': statistics.mean(e.total_score for e in entries_b),
            'improvement': 0.0,
            'detailed_comparison': {}
        }
        
        comparison['improvement'] = (comparison['score_b'] - comparison['score_a']) / comparison['score_a'] * 100
        
        return comparison
    
    def detect_regressions(self, threshold: float = 5.0) -> List[Dict[str, Any]]:
        """Detect performance regressions compared to previous versions"""
        entries = self.load_results()
        if len(entries) < 2:
            return []
        
        # Sort by timestamp
        sorted_entries = sorted(entries, key=lambda x: x.timestamp)
        
        regressions = []
        for i in range(1, len(sorted_entries)):
            current = sorted_entries[i]
            previous = sorted_entries[i-1]
            
            score_change = (current.total_score - previous.total_score) / previous.total_score * 100
            
            if score_change < -threshold:
                regressions.append({
                    'current_version': current.version,
                    'previous_version': previous.version,
                    'score_change_percent': score_change,
                    'current_score': current.total_score,
                    'previous_score': previous.total_score,
                    'timestamp': current.timestamp
                })
        
        return regressions
    
    def generate_performance_plots(self, output_dir: Path) -> None:
        """Generate performance trend plots"""
        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available. Skipping chart generation.")
            return
        
        entries = self.load_results()
        if len(entries) < 2:
            print("Insufficient data for plotting.")
            return
        
        output_dir.mkdir(exist_ok=True)
        
        # Sort by timestamp
        sorted_entries = sorted(entries, key=lambda x: x.timestamp)
        
        # Performance over time
        timestamps = [datetime.fromtimestamp(e.timestamp) for e in sorted_entries]
        scores = [e.total_score for e in sorted_entries]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, scores, marker='o', linewidth=2, markersize=6)
        plt.title('TrustformeRS Performance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance by category
        categories = set()
        for entry in sorted_entries:
            for result in entry.results:
                categories.add(result['category'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, category in enumerate(list(categories)[:4]):  # Show up to 4 categories
            if i >= len(axes):
                break
            
            category_scores = []
            category_timestamps = []
            
            for entry in sorted_entries:
                category_results = [r for r in entry.results if r['category'] == category]
                if category_results:
                    # Use throughput as the primary metric
                    throughputs = [r['throughput'] for r in category_results if r['throughput'] is not None]
                    if throughputs:
                        category_scores.append(statistics.mean(throughputs))
                        category_timestamps.append(datetime.fromtimestamp(entry.timestamp))
            
            if category_scores:
                axes[i].plot(category_timestamps, category_scores, marker='o', linewidth=2)
                axes[i].set_title(f'{category.title()} Performance')
                axes[i].set_ylabel('Throughput')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'category_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots generated in: {output_dir}")
    
    def submit_to_community(self, entry: LeaderboardEntry, 
                          api_key: Optional[str] = None) -> bool:
        """Submit results to community leaderboard"""
        if not REQUESTS_AVAILABLE:
            print("Requests library not available. Cannot submit to community leaderboard.")
            return False
        
        api_key = api_key or os.environ.get('TRUSTFORMERS_API_KEY')
        if not api_key:
            print("API key required for community submission.")
            return False
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = entry.to_dict()
        
        try:
            response = requests.post(
                f"{self.api_endpoint}/submit",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                print("Successfully submitted to community leaderboard!")
                result = response.json()
                print(f"Submission ID: {result.get('id', 'Unknown')}")
                return True
            else:
                print(f"Submission failed: {response.status_code} - {response.text}")
                return False
                
        except requests.RequestException as e:
            print(f"Network error during submission: {e}")
            return False
    
    def _generate_html_template(self, entries: List[LeaderboardEntry]) -> str:
        """Generate HTML leaderboard template"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TrustformeRS Performance Leaderboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .timestamp {{ color: #666; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .rank {{ font-weight: bold; font-size: 18px; }}
        .score {{ font-weight: bold; color: #2196F3; }}
        .hardware {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏆 TrustformeRS Performance Leaderboard</h1>
        <p class="timestamp">Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>
    
    <table>
        <tr>
            <th>Rank</th>
            <th>Name</th>
            <th>Score</th>
            <th>Version</th>
            <th>Hardware</th>
            <th>Backend</th>
            <th>Date</th>
        </tr>
"""
        
        for rank, entry in enumerate(entries, 1):
            gpu_info = entry.hardware.get('gpu_model', 'CPU Only')
            date_str = datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d')
            
            html += f"""
        <tr>
            <td class="rank">#{rank}</td>
            <td>{entry.name}</td>
            <td class="score">{entry.total_score:.2f}</td>
            <td>{entry.version}</td>
            <td class="hardware">
                {entry.hardware.get('cpu_model', 'Unknown CPU')}<br>
                {entry.hardware.get('cpu_cores', '?')} cores, {entry.hardware.get('memory_gb', '?')}GB<br>
                GPU: {gpu_info}
            </td>
            <td>{entry.software.get('backend', 'Unknown')}</td>
            <td>{date_str}</td>
        </tr>"""
        
        html += """
    </table>
    
    <div style="margin-top: 40px; text-align: center; color: #666;">
        <p>Submit your results: <code>python scripts/leaderboard_manager.py submit</code></p>
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_markdown_template(self, entries: List[LeaderboardEntry]) -> str:
        """Generate Markdown leaderboard template"""
        md = f"""# 🏆 TrustformeRS Performance Leaderboard

*Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*

| Rank | Name | Score | Version | CPU | GPU | Backend | Date |
|------|------|-------|---------|-----|-----|---------|------|
"""
        
        for rank, entry in enumerate(entries, 1):
            gpu_info = entry.hardware.get('gpu_model', 'CPU Only')
            cpu_info = f"{entry.hardware.get('cpu_cores', '?')}c"
            date_str = datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d')
            
            md += f"| #{rank} | {entry.name} | **{entry.total_score:.2f}** | {entry.version} | {cpu_info} | {gpu_info} | {entry.software.get('backend', '?')} | {date_str} |\n"
        
        md += f"""
## How to Submit

1. Run benchmarks: `cargo bench --bench leaderboard_bench`
2. Submit results: `python scripts/leaderboard_manager.py submit`

## Categories

- **Tensor Operations**: Basic linear algebra operations
- **Model Inference**: Full model forward passes  
- **Tokenization**: Text processing and tokenization
- **Memory**: Memory allocation and management
- **Quantization**: Model compression techniques

Generated by TrustformeRS Leaderboard Manager
"""
        
        return md
    
    def _get_memory_gb(self) -> int:
        """Get system memory in GB"""
        try:
            import psutil
            return int(psutil.virtual_memory().total / (1024**3))
        except ImportError:
            return int(os.environ.get('TRUSTFORMERS_MEMORY_GB', '8'))
    
    def _get_rust_version(self) -> str:
        """Get Rust version"""
        try:
            return subprocess.check_output(['rustc', '--version'], 
                                         universal_newlines=True).strip()
        except subprocess.CalledProcessError:
            return 'Unknown Rust Version'
    
    def _get_trustformers_version(self) -> str:
        """Get TrustformeRS version from Cargo.toml"""
        cargo_toml = Path(__file__).parent.parent / 'Cargo.toml'
        try:
            with open(cargo_toml, 'r') as f:
                for line in f:
                    if line.startswith('version'):
                        return line.split('=')[1].strip().strip('"')
        except FileNotFoundError:
            pass
        return '0.1.0'
    
    def _parse_env_int(self, var_name: str) -> Optional[int]:
        """Parse integer from environment variable"""
        value = os.environ.get(var_name)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
        return None

def main():
    parser = argparse.ArgumentParser(description='TrustformeRS Leaderboard Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run benchmarks
    run_parser = subparsers.add_parser('run', help='Run leaderboard benchmarks')
    run_parser.add_argument('--name', default='TrustformeRS Benchmark', 
                          help='Benchmark submission name')
    run_parser.add_argument('--backend', default='CPU', 
                          help='Backend to use (CPU, CUDA, ROCm, Metal)')
    run_parser.add_argument('--features', 
                          help='Comma-separated list of features to enable')
    
    # Generate leaderboards
    generate_parser = subparsers.add_parser('generate', help='Generate leaderboard files')
    generate_parser.add_argument('--format', choices=['html', 'markdown', 'both'], 
                               default='both', help='Output format')
    generate_parser.add_argument('--output-dir', type=Path, default=Path('.'), 
                               help='Output directory')
    
    # Compare versions
    compare_parser = subparsers.add_parser('compare', help='Compare performance between versions')
    compare_parser.add_argument('version_a', help='First version to compare')
    compare_parser.add_argument('version_b', help='Second version to compare')
    
    # Submit to community
    submit_parser = subparsers.add_parser('submit', help='Submit results to community leaderboard')
    submit_parser.add_argument('--file', type=Path, help='Result file to submit (default: latest)')
    submit_parser.add_argument('--api-key', help='API key for submission')
    
    # Regression detection
    regress_parser = subparsers.add_parser('regressions', help='Detect performance regressions')
    regress_parser.add_argument('--threshold', type=float, default=5.0, 
                              help='Regression threshold percentage')
    
    # Plot generation
    plot_parser = subparsers.add_parser('plot', help='Generate performance plots')
    plot_parser.add_argument('--output-dir', type=Path, default=Path('plots'), 
                           help='Output directory for plots')
    
    args = parser.parse_args()
    
    manager = LeaderboardManager()
    
    if args.command == 'run':
        config = {
            'benchmark_name': args.name,
            'backend': args.backend
        }
        if args.features:
            config['features'] = args.features.split(',')
        
        result_file = manager.run_benchmarks(config)
        if result_file:
            print(f"Benchmark completed. Results in: {result_file}")
    
    elif args.command == 'generate':
        entries = manager.load_results()
        if not entries:
            print("No benchmark results found.")
            return
        
        if args.format in ['html', 'both']:
            manager.generate_leaderboard_html(entries, args.output_dir / 'leaderboard.html')
        
        if args.format in ['markdown', 'both']:
            manager.generate_leaderboard_markdown(entries, args.output_dir / 'leaderboard.md')
    
    elif args.command == 'compare':
        comparison = manager.compare_versions(args.version_a, args.version_b)
        if comparison:
            print(f"Performance Comparison:")
            print(f"  {args.version_a}: {comparison['score_a']:.2f}")
            print(f"  {args.version_b}: {comparison['score_b']:.2f}")
            print(f"  Improvement: {comparison['improvement']:+.2f}%")
    
    elif args.command == 'submit':
        entries = manager.load_results()
        if not entries:
            print("No benchmark results found.")
            return
        
        if args.file:
            entry = LeaderboardEntry.from_json_file(args.file)
        else:
            entry = max(entries, key=lambda x: x.timestamp)  # Latest entry
        
        success = manager.submit_to_community(entry, args.api_key)
        if success:
            print("Submission successful!")
        else:
            print("Submission failed.")
    
    elif args.command == 'regressions':
        regressions = manager.detect_regressions(args.threshold)
        if regressions:
            print(f"Found {len(regressions)} performance regressions:")
            for reg in regressions:
                print(f"  {reg['previous_version']} → {reg['current_version']}: "
                      f"{reg['score_change_percent']:+.2f}%")
        else:
            print("No performance regressions detected.")
    
    elif args.command == 'plot':
        manager.generate_performance_plots(args.output_dir)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()