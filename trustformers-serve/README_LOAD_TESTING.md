# TrustformeRS Load Testing Suite

Comprehensive load testing framework for TrustformeRS Serve API with advanced features including multiple test scenarios, authentication support, real-time monitoring, and detailed performance analysis.

## Features

- **Multiple Test Scenarios**: Support for complex, weighted test scenarios
- **Authentication**: API Key, JWT, and OAuth2 authentication support
- **Real-time Monitoring**: Live progress reporting and metrics collection
- **Comprehensive Metrics**: Response times, throughput, error rates, and percentiles
- **Flexible Configuration**: JSON configuration files with validation
- **Multiple Output Formats**: Console, JSON, CSV, and HTML reports
- **Advanced Features**: HTTP/2, connection pooling, rate limiting, and ramp-up/down

## Quick Start

### Basic Usage

```bash
# Run a simple health check test
cargo run --bin load_test run --url http://localhost:8080 --duration 60 --concurrent-users 10

# Run with authentication
cargo run --bin load_test run --url http://localhost:8080 --api-key your-api-key --duration 120

# Test a specific endpoint
cargo run --bin load_test run --endpoint /v1/inference --method POST --duration 60
```

### Predefined Scenarios

```bash
# Health check scenario (light load)
cargo run --bin load_test scenario health-check --url http://localhost:8080

# API stress test (heavy load)
cargo run --bin load_test scenario api-stress --url http://localhost:8080

# Mixed workload scenario
cargo run --bin load_test scenario mixed --url http://localhost:8080

# Inference performance test
cargo run --bin load_test scenario inference-test --url http://localhost:8080

# Authentication test
cargo run --bin load_test scenario auth-test --url http://localhost:8080 --api-key your-key
```

### Configuration-based Testing

```bash
# Generate a sample configuration file
cargo run --bin load_test generate-config --output my_config.json

# Validate a configuration file
cargo run --bin load_test validate-config --config my_config.json

# Run test with configuration file
cargo run --bin load_test run --config my_config.json --output results.json
```

## Configuration

### Example Configuration File

See `examples/load_test_config.json` for a comprehensive example configuration.

### Configuration Structure

```json
{
  "base_url": "http://localhost:8080",
  "duration_seconds": 120,
  "concurrent_users": 25,
  "requests_per_second": 50.0,
  "scenarios": [...],
  "auth": {...},
  "timeout_config": {...},
  "output_config": {...},
  "advanced_config": {...}
}
```

### Test Scenarios

Each scenario supports:

- **HTTP Methods**: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS
- **Request Bodies**: JSON templates with variable substitution
- **Query Parameters**: Dynamic query string parameters
- **Headers**: Custom HTTP headers
- **Validation Rules**: Response validation and assertion
- **Weight Distribution**: Percentage of traffic for each scenario

### Validation Rules

Supported validation types:

- `ResponseContains`: Check if response contains specific text
- `ResponseMatches`: Validate response against regex pattern
- `JsonFieldExists`: Verify JSON field presence
- `JsonFieldEquals`: Check JSON field value
- `HeaderExists`: Validate HTTP header presence
- `HeaderEquals`: Check HTTP header value
- `ResponseSizeBounds`: Validate response size within bounds

## Authentication

### API Key Authentication

```bash
cargo run --bin load_test run --api-key your-api-key-here
```

### JWT Bearer Token

```bash
cargo run --bin load_test run --jwt-token your-jwt-token-here
```

### Configuration File

```json
{
  "auth": {
    "auth_type": "ApiKey",
    "api_key": "your-api-key-here",
    "jwt_token": null,
    "oauth2": null
  }
}
```

## Output and Reporting

### Console Output

Real-time progress reporting with key metrics:

```
📈 Progress: 1250 requests | Success: 1247 | Failed: 3 | Avg RT: 45.2ms | RPS: 25.4
```

### JSON Report

Comprehensive test results with detailed statistics:

```json
{
  "summary": {
    "total_requests": 1500,
    "successful_requests": 1485,
    "success_rate": 99.0,
    "average_rps": 25.0,
    "average_response_time_ms": 42.5
  },
  "scenario_results": {...},
  "performance_metrics": {...},
  "error_analysis": {...},
  "time_series": {...}
}
```

### CSV Export

Key metrics in CSV format for spreadsheet analysis.

## Performance Recommendations

The tool provides automatic performance recommendations based on test results:

- **Error Rate Analysis**: Identifies reliability issues and suggests improvements
- **Response Time Optimization**: Recommends caching and optimization strategies
- **Throughput Enhancement**: Suggests scaling and performance tuning approaches
- **Best Practices**: General recommendations for production deployments

## Advanced Features

### Rate Limiting

Control request rate to simulate realistic load patterns:

```bash
cargo run --bin load_test run --requests-per-second 100
```

### Ramp-up and Ramp-down

Gradual load increase and decrease for realistic testing:

```json
{
  "advanced_config": {
    "ramp_up_seconds": 30,
    "ramp_down_seconds": 15
  }
}
```

### Connection Pooling

Optimize connection reuse for better performance:

```json
{
  "advanced_config": {
    "enable_connection_pooling": true,
    "max_idle_per_host": 32,
    "keep_alive_timeout_seconds": 90
  }
}
```

### Think Time

Simulate user behavior with delays between requests:

```json
{
  "advanced_config": {
    "think_time_ms": 100
  }
}
```

## Use Cases

### Development Testing

Quick validation of endpoint functionality:

```bash
cargo run --bin load_test scenario health-check --duration 30
```

### Performance Benchmarking

Establish baseline performance metrics:

```bash
cargo run --bin load_test scenario api-stress --duration 300 --output benchmark.json
```

### Load Testing

Validate system behavior under expected load:

```bash
cargo run --bin load_test run --config production_load.json
```

### Stress Testing

Find system breaking points:

```bash
cargo run --bin load_test scenario api-stress --concurrent-users 100 --requests-per-second 200
```

### CI/CD Integration

Automated performance testing in pipelines:

```bash
# Run quick performance check
cargo run --bin load_test scenario health-check --output ci_results.json

# Validate results programmatically
if [ $(jq '.summary.success_rate' ci_results.json) -lt 95 ]; then
  echo "Performance test failed"
  exit 1
fi
```

## Best Practices

### Test Environment

- Use dedicated test environment similar to production
- Ensure consistent network conditions
- Monitor server resources during testing
- Run tests multiple times for reliable results

### Scenario Design

- Model realistic user behavior patterns
- Include error scenarios and edge cases
- Test different load patterns (steady, burst, gradual)
- Validate both functional and non-functional requirements

### Monitoring

- Monitor both client and server metrics
- Track resource usage (CPU, memory, network)
- Set up alerting for performance degradation
- Analyze trends over time

### Results Analysis

- Focus on percentiles rather than averages
- Identify performance bottlenecks
- Correlate errors with load patterns
- Document and track performance changes

## Troubleshooting

### Common Issues

1. **Connection timeouts**: Increase timeout values or reduce load
2. **High error rates**: Check server capacity and error handling
3. **Inconsistent results**: Ensure stable test environment
4. **Authentication failures**: Verify credentials and permissions

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
RUST_LOG=debug cargo run --bin load_test run --verbose
```

### Performance Issues

If the load test tool itself becomes a bottleneck:

- Reduce concurrent users
- Increase think time
- Use multiple test machines
- Optimize test scenarios

## Contributing

To add new features or improvements:

1. Add new scenario types in `TestScenario`
2. Implement validation rules in `ValidationRule`
3. Extend authentication methods in `AuthConfig`
4. Add output formats in `OutputFormat`

## Examples

See the `examples/` directory for:

- Sample configuration files
- Common test scenarios
- CI/CD integration examples
- Performance analysis scripts