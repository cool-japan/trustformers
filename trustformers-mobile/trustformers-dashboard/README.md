# TrustformeRS Performance Dashboard

A comprehensive real-time performance monitoring and visualization system for TrustformeRS models.

## Features

### 📊 Real-time Monitoring
- Live performance metrics (latency, throughput, memory, GPU utilization)
- Multi-model and multi-device tracking
- Historical data visualization with interactive charts
- Automatic data collection from model serving endpoints

### 🔬 Deep Profiling
- Layer-wise performance analysis
- Memory allocation tracking
- FLOP utilization calculation
- Bottleneck identification
- Optimization suggestions

### 📈 Benchmarking Suite
- Standardized benchmarks for common models
- Custom benchmark support
- Performance regression detection
- Cross-device comparison

### 🔔 Intelligent Alerts
- Threshold-based alerting system
- Multiple severity levels
- Email, webhook, and console notifications
- Alert history and acknowledgment

### 🔀 Model Comparison
- Side-by-side performance comparison
- Statistical significance testing
- Trade-off analysis
- Scenario-specific recommendations

## Installation

1. Install Python dependencies:
```bash
cd trustformers-dashboard
pip install -r requirements.txt
```

2. Set up Redis (optional, for caching):
```bash
# macOS
brew install redis
brew services start redis

# Linux
sudo apt-get install redis-server
sudo systemctl start redis
```

3. Configure environment variables (optional):
```bash
# For email alerts
export TRUSTFORMERS_SMTP_SERVER=smtp.gmail.com
export TRUSTFORMERS_SMTP_PORT=587
export TRUSTFORMERS_SMTP_USER=your-email@gmail.com
export TRUSTFORMERS_SMTP_PASSWORD=your-app-password
export TRUSTFORMERS_ALERT_EMAIL=alerts@example.com

# For webhook alerts
export TRUSTFORMERS_ALERT_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# For Redis (if not using default)
export REDIS_URL=redis://localhost:6379
```

## Usage

### Starting the Dashboard

```bash
# Basic usage
./run_dashboard.py

# With options
./run_dashboard.py --host 0.0.0.0 --port 8050 --debug

# Disable automatic features
./run_dashboard.py --no-collect --no-alerts
```

### Accessing the Dashboard

Open your browser and navigate to:
```
http://localhost:8050
```

### Dashboard Pages

1. **Overview**: Real-time metrics and system health
2. **Models**: Detailed per-model performance analysis
3. **Benchmarks**: Run and analyze standardized benchmarks
4. **Profiling**: Deep performance profiling and optimization
5. **Comparison**: Compare models across metrics
6. **Alerts**: Configure and monitor performance alerts

## Architecture

### Components

1. **Data Collector** (`data_collector.py`)
   - Collects metrics from model serving endpoints
   - Monitors system resources (CPU, GPU, memory)
   - Stores data in SQLite with configurable retention

2. **Benchmark Runner** (`benchmark_runner.py`)
   - Executes standardized benchmarks
   - Measures latency, throughput, memory, and power
   - Supports custom benchmark configurations

3. **Model Profiler** (`model_profiler.py`)
   - Layer-wise performance analysis
   - Memory allocation tracking
   - Bottleneck identification
   - Optimization recommendations

4. **Comparison Engine** (`comparison_engine.py`)
   - Statistical analysis of model performance
   - Trade-off identification
   - Scenario-specific recommendations

5. **Alert Manager** (`alert_manager.py`)
   - Threshold-based monitoring
   - Multi-channel notifications
   - Alert history and management

### Data Storage

- **SQLite**: Metrics and alert history
- **Redis**: Real-time data caching
- **File System**: Benchmark results and profiles

## API Integration

### Metrics Endpoint

Model serving endpoints should expose metrics at `/metrics`:

```json
{
  "inference_latency_ms": 23.5,
  "throughput_samples_per_sec": 42.0,
  "memory_usage_mb": 1024,
  "error_count": 0,
  "request_count": 1000,
  "device": "cuda:0"
}
```

### Prometheus Integration

The dashboard exposes Prometheus metrics at `/metrics`:

```
# HELP trustformers_inference_latency_seconds Model inference latency
# TYPE trustformers_inference_latency_seconds histogram
trustformers_inference_latency_seconds_bucket{model="bert-base",device="cuda",le="0.01"} 100
```

## Extending the Dashboard

### Adding Custom Metrics

1. Add evaluator to `AlertManager`:
```python
def _evaluate_custom_metric(self, model, device):
    # Your metric collection logic
    return metric_value
```

2. Register in `metric_evaluators`:
```python
self.metric_evaluators['custom_metric'] = self._evaluate_custom_metric
```

### Adding Custom Visualizations

1. Create new chart in `app.py`:
```python
@app.callback(
    Output('custom-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_custom_chart(n):
    # Your visualization logic
    return figure
```

### Adding Custom Benchmarks

1. Add configuration to `benchmark_configs`:
```python
'custom_benchmark': {
    'model': 'your-model',
    'task': 'your-task',
    'default_batch_size': 32,
    'script': 'benchmarks/custom.rs'
}
```

## Performance Tips

1. **Data Retention**: Adjust retention period based on storage capacity
2. **Collection Interval**: Balance between granularity and overhead
3. **Alert Cooldown**: Prevent alert fatigue with appropriate cooldowns
4. **Database Indexing**: Indexes are automatically created for common queries

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   lsof -i :8050
   kill -9 <PID>
   ```

2. **Database Lock**
   ```bash
   rm data/metrics.db-journal
   ```

3. **High Memory Usage**
   - Reduce data retention period
   - Increase collection interval
   - Use Redis for caching

### Debug Mode

Run with `--debug --verbose` for detailed logging:
```bash
./run_dashboard.py --debug --verbose
```

## Security Considerations

1. **Authentication**: Add authentication for production deployments
2. **HTTPS**: Use reverse proxy with SSL termination
3. **Firewall**: Restrict access to dashboard port
4. **Secrets**: Use environment variables for sensitive data

## Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Submit PR with clear description

## License

Same as TrustformeRS project.