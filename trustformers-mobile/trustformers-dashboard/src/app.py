#!/usr/bin/env python3
"""
TrustformeRS Performance Dashboard
A comprehensive monitoring and visualization system for ML model performance.
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import threading
import time
from collections import deque
import psutil
import redis
from flask_caching import Cache
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest

# Import custom modules
from data_collector import DataCollector
from benchmark_runner import BenchmarkRunner
from model_profiler import ModelProfiler
from comparison_engine import ComparisonEngine
from alert_manager import AlertManager

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

# Configure caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379')
})

# Initialize components
data_collector = DataCollector()
benchmark_runner = BenchmarkRunner()
model_profiler = ModelProfiler()
comparison_engine = ComparisonEngine()
alert_manager = AlertManager()

# Prometheus metrics
registry = CollectorRegistry()
inference_latency = Histogram('trustformers_inference_latency_seconds', 
                            'Model inference latency', 
                            ['model', 'device'], 
                            registry=registry)
throughput_gauge = Gauge('trustformers_throughput_samples_per_second', 
                        'Model throughput', 
                        ['model', 'device'], 
                        registry=registry)
memory_usage = Gauge('trustformers_memory_usage_bytes', 
                    'Memory usage', 
                    ['model', 'device', 'type'], 
                    registry=registry)

# Layout components
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Overview", href="/overview")),
        dbc.NavItem(dbc.NavLink("Models", href="/models")),
        dbc.NavItem(dbc.NavLink("Benchmarks", href="/benchmarks")),
        dbc.NavItem(dbc.NavLink("Profiling", href="/profiling")),
        dbc.NavItem(dbc.NavLink("Comparison", href="/comparison")),
        dbc.NavItem(dbc.NavLink("Alerts", href="/alerts")),
    ],
    brand="TrustformeRS Performance Dashboard",
    brand_href="/",
    color="primary",
    dark=True,
)

# Main layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content'),
    dcc.Interval(id='interval-component', interval=5000),  # Update every 5 seconds
    dcc.Store(id='benchmark-data-store'),
    dcc.Store(id='profile-data-store'),
])

# Overview page
def create_overview_page():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Performance Overview", className="mb-4"),
                html.P("Real-time monitoring of TrustformeRS model performance")
            ])
        ]),
        
        # Key metrics cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Active Models", className="card-title"),
                        html.H2(id="active-models-count", children="0"),
                        html.P("Currently loaded", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Avg Latency", className="card-title"),
                        html.H2(id="avg-latency", children="0ms"),
                        html.P("Last 5 minutes", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Throughput", className="card-title"),
                        html.H2(id="throughput", children="0/s"),
                        html.P("Samples per second", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("GPU Memory", className="card-title"),
                        html.H2(id="gpu-memory", children="0%"),
                        html.P("Current usage", className="text-muted")
                    ])
                ])
            ], width=3),
        ], className="mb-4"),
        
        # Real-time charts
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='latency-timeline', figure={})
            ], width=6),
            dbc.Col([
                dcc.Graph(id='throughput-timeline', figure={})
            ], width=6),
        ]),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='memory-usage-chart', figure={})
            ], width=6),
            dbc.Col([
                dcc.Graph(id='device-utilization', figure={})
            ], width=6),
        ], className="mt-4"),
    ], fluid=True)

# Models page
def create_models_page():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Model Performance", className="mb-4"),
                html.P("Detailed metrics for individual models")
            ])
        ]),
        
        # Model selector
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='model-selector',
                    options=[],
                    value=None,
                    placeholder="Select a model"
                )
            ], width=6),
            dbc.Col([
                dcc.Dropdown(
                    id='device-selector',
                    options=[
                        {'label': 'CPU', 'value': 'cpu'},
                        {'label': 'GPU (CUDA)', 'value': 'cuda'},
                        {'label': 'GPU (Metal)', 'value': 'metal'},
                        {'label': 'Mobile (Android)', 'value': 'android'},
                        {'label': 'Mobile (iOS)', 'value': 'ios'},
                    ],
                    value='cpu',
                    placeholder="Select device"
                )
            ], width=6),
        ], className="mb-4"),
        
        # Model details
        html.Div(id='model-details', children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='model-latency-dist', figure={})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='model-memory-profile', figure={})
                ], width=6),
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Layer-wise Performance"),
                    dash_table.DataTable(
                        id='layer-performance-table',
                        columns=[
                            {'name': 'Layer', 'id': 'layer'},
                            {'name': 'Type', 'id': 'type'},
                            {'name': 'Latency (ms)', 'id': 'latency'},
                            {'name': 'Memory (MB)', 'id': 'memory'},
                            {'name': 'FLOPs', 'id': 'flops'},
                        ],
                        data=[],
                        style_cell={'textAlign': 'left'},
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'latency'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ]
                    )
                ], width=12),
            ], className="mt-4"),
        ])
    ], fluid=True)

# Benchmarks page
def create_benchmarks_page():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Benchmark Suite", className="mb-4"),
                html.P("Run and analyze standardized benchmarks")
            ])
        ]),
        
        # Benchmark controls
        dbc.Row([
            dbc.Col([
                html.H3("Run Benchmark"),
                dbc.Form([
                    dbc.FormGroup([
                        dbc.Label("Select Benchmark"),
                        dcc.Dropdown(
                            id='benchmark-selector',
                            options=[
                                {'label': 'BERT Base Inference', 'value': 'bert_base'},
                                {'label': 'GPT-2 Generation', 'value': 'gpt2_gen'},
                                {'label': 'Vision Transformer', 'value': 'vit'},
                                {'label': 'Mobile BERT', 'value': 'mobile_bert'},
                                {'label': 'Custom Model', 'value': 'custom'},
                            ],
                            value='bert_base'
                        )
                    ]),
                    dbc.FormGroup([
                        dbc.Label("Batch Size"),
                        dbc.Input(id='batch-size', type='number', value=32)
                    ]),
                    dbc.FormGroup([
                        dbc.Label("Sequence Length"),
                        dbc.Input(id='seq-length', type='number', value=128)
                    ]),
                    dbc.Button("Run Benchmark", id='run-benchmark', color='primary', className='mt-3')
                ])
            ], width=4),
            dbc.Col([
                html.H3("Benchmark Results"),
                html.Div(id='benchmark-status', children=[
                    dbc.Alert("No benchmark running", color='info')
                ]),
                dcc.Loading(
                    id='benchmark-results-loading',
                    type='default',
                    children=html.Div(id='benchmark-results')
                )
            ], width=8),
        ]),
        
        # Historical benchmarks
        dbc.Row([
            dbc.Col([
                html.H3("Historical Results", className="mt-4"),
                dcc.Graph(id='benchmark-history', figure={})
            ], width=12)
        ])
    ], fluid=True)

# Profiling page
def create_profiling_page():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Model Profiling", className="mb-4"),
                html.P("Deep performance analysis and optimization insights")
            ])
        ]),
        
        # Profiling controls
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Profile Configuration"),
                    dbc.CardBody([
                        dbc.FormGroup([
                            dbc.Label("Model"),
                            dcc.Dropdown(id='profile-model-selector', options=[])
                        ]),
                        dbc.FormGroup([
                            dbc.Label("Profile Type"),
                            dcc.RadioItems(
                                id='profile-type',
                                options=[
                                    {'label': 'Memory', 'value': 'memory'},
                                    {'label': 'Computation', 'value': 'compute'},
                                    {'label': 'Full', 'value': 'full'}
                                ],
                                value='full'
                            )
                        ]),
                        dbc.Button("Start Profiling", id='start-profile', color='primary')
                    ])
                ])
            ], width=4),
            dbc.Col([
                html.Div(id='profile-visualization', children=[
                    dcc.Graph(id='profile-flamegraph', figure={}),
                    dcc.Graph(id='profile-memory-timeline', figure={})
                ])
            ], width=8)
        ]),
        
        # Optimization suggestions
        dbc.Row([
            dbc.Col([
                html.H3("Optimization Suggestions", className="mt-4"),
                html.Div(id='optimization-suggestions')
            ], width=12)
        ])
    ], fluid=True)

# Comparison page
def create_comparison_page():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Model Comparison", className="mb-4"),
                html.P("Compare performance across models, devices, and configurations")
            ])
        ]),
        
        # Comparison setup
        dbc.Row([
            dbc.Col([
                html.H3("Select Models to Compare"),
                dcc.Dropdown(
                    id='comparison-models',
                    options=[],
                    multi=True,
                    placeholder="Select multiple models"
                )
            ], width=12)
        ], className="mb-4"),
        
        # Comparison charts
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='comparison-latency', figure={})
            ], width=6),
            dbc.Col([
                dcc.Graph(id='comparison-throughput', figure={})
            ], width=6),
        ]),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='comparison-memory', figure={})
            ], width=6),
            dbc.Col([
                dcc.Graph(id='comparison-accuracy', figure={})
            ], width=6),
        ], className="mt-4"),
        
        # Detailed comparison table
        dbc.Row([
            dbc.Col([
                html.H3("Detailed Metrics", className="mt-4"),
                dash_table.DataTable(
                    id='comparison-table',
                    columns=[
                        {'name': 'Model', 'id': 'model'},
                        {'name': 'Device', 'id': 'device'},
                        {'name': 'Latency (ms)', 'id': 'latency'},
                        {'name': 'Throughput', 'id': 'throughput'},
                        {'name': 'Memory (MB)', 'id': 'memory'},
                        {'name': 'Accuracy', 'id': 'accuracy'},
                    ],
                    data=[],
                    style_cell={'textAlign': 'left'},
                    export_format='csv'
                )
            ], width=12)
        ])
    ], fluid=True)

# Alerts page
def create_alerts_page():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Performance Alerts", className="mb-4"),
                html.P("Configure and monitor performance thresholds")
            ])
        ]),
        
        # Alert configuration
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Configure Alert"),
                    dbc.CardBody([
                        dbc.FormGroup([
                            dbc.Label("Metric"),
                            dcc.Dropdown(
                                id='alert-metric',
                                options=[
                                    {'label': 'Latency', 'value': 'latency'},
                                    {'label': 'Memory Usage', 'value': 'memory'},
                                    {'label': 'Throughput', 'value': 'throughput'},
                                    {'label': 'Error Rate', 'value': 'error_rate'}
                                ],
                                value='latency'
                            )
                        ]),
                        dbc.FormGroup([
                            dbc.Label("Threshold"),
                            dbc.Input(id='alert-threshold', type='number', value=100)
                        ]),
                        dbc.FormGroup([
                            dbc.Label("Condition"),
                            dcc.RadioItems(
                                id='alert-condition',
                                options=[
                                    {'label': 'Above', 'value': 'above'},
                                    {'label': 'Below', 'value': 'below'}
                                ],
                                value='above'
                            )
                        ]),
                        dbc.Button("Create Alert", id='create-alert', color='warning')
                    ])
                ])
            ], width=4),
            dbc.Col([
                html.H3("Active Alerts"),
                html.Div(id='active-alerts')
            ], width=8)
        ]),
        
        # Alert history
        dbc.Row([
            dbc.Col([
                html.H3("Alert History", className="mt-4"),
                dash_table.DataTable(
                    id='alert-history-table',
                    columns=[
                        {'name': 'Time', 'id': 'time'},
                        {'name': 'Alert', 'id': 'alert'},
                        {'name': 'Value', 'id': 'value'},
                        {'name': 'Status', 'id': 'status'},
                    ],
                    data=[],
                    style_cell={'textAlign': 'left'}
                )
            ], width=12)
        ])
    ], fluid=True)

# Callbacks
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/models':
        return create_models_page()
    elif pathname == '/benchmarks':
        return create_benchmarks_page()
    elif pathname == '/profiling':
        return create_profiling_page()
    elif pathname == '/comparison':
        return create_comparison_page()
    elif pathname == '/alerts':
        return create_alerts_page()
    else:
        return create_overview_page()

@app.callback(
    [Output('active-models-count', 'children'),
     Output('avg-latency', 'children'),
     Output('throughput', 'children'),
     Output('gpu-memory', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_metrics(n):
    metrics = data_collector.get_current_metrics()
    return (
        str(metrics.get('active_models', 0)),
        f"{metrics.get('avg_latency', 0):.1f}ms",
        f"{metrics.get('throughput', 0):.0f}/s",
        f"{metrics.get('gpu_memory', 0):.0f}%"
    )

@app.callback(
    Output('latency-timeline', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_latency_timeline(n):
    data = data_collector.get_latency_history()
    df = pd.DataFrame(data)
    
    if df.empty:
        return go.Figure()
    
    fig = px.line(df, x='timestamp', y='latency', color='model',
                  title='Inference Latency Over Time')
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Latency (ms)",
        hovermode='x unified'
    )
    return fig

@app.callback(
    Output('throughput-timeline', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_throughput_timeline(n):
    data = data_collector.get_throughput_history()
    df = pd.DataFrame(data)
    
    if df.empty:
        return go.Figure()
    
    fig = px.line(df, x='timestamp', y='throughput', color='model',
                  title='Model Throughput Over Time')
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Samples/sec",
        hovermode='x unified'
    )
    return fig

@app.callback(
    Output('memory-usage-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_memory_chart(n):
    data = data_collector.get_memory_usage()
    
    fig = go.Figure()
    for device in data:
        fig.add_trace(go.Bar(
            name=device['name'],
            x=['Allocated', 'Reserved', 'Free'],
            y=[device['allocated'], device['reserved'], device['free']]
        ))
    
    fig.update_layout(
        title='Memory Usage by Device',
        xaxis_title="Memory Type",
        yaxis_title="Memory (GB)",
        barmode='group'
    )
    return fig

@app.callback(
    Output('benchmark-results', 'children'),
    Output('benchmark-status', 'children'),
    Input('run-benchmark', 'n_clicks'),
    State('benchmark-selector', 'value'),
    State('batch-size', 'value'),
    State('seq-length', 'value')
)
def run_benchmark(n_clicks, benchmark_type, batch_size, seq_length):
    if not n_clicks:
        return None, dbc.Alert("No benchmark running", color='info')
    
    # Start benchmark
    status = dbc.Alert("Running benchmark...", color='warning')
    
    # Run benchmark (this would be async in production)
    results = benchmark_runner.run(
        benchmark_type=benchmark_type,
        batch_size=batch_size,
        seq_length=seq_length
    )
    
    # Display results
    results_display = dbc.Card([
        dbc.CardHeader("Benchmark Complete"),
        dbc.CardBody([
            html.H5(f"Model: {results['model']}"),
            html.P(f"Device: {results['device']}"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H6("Latency"),
                    html.H4(f"{results['latency']:.2f} ms"),
                    html.P(f"P95: {results['p95_latency']:.2f} ms", className="text-muted")
                ], width=4),
                dbc.Col([
                    html.H6("Throughput"),
                    html.H4(f"{results['throughput']:.0f} samples/s"),
                    html.P(f"Batch: {batch_size}", className="text-muted")
                ], width=4),
                dbc.Col([
                    html.H6("Memory"),
                    html.H4(f"{results['memory']:.1f} GB"),
                    html.P(f"Peak: {results['peak_memory']:.1f} GB", className="text-muted")
                ], width=4),
            ])
        ])
    ])
    
    status = dbc.Alert("Benchmark complete!", color='success')
    return results_display, status

# Prometheus metrics endpoint
@app.server.route('/metrics')
def metrics():
    return generate_latest(registry)

# Main execution
if __name__ == '__main__':
    # Start background data collection
    data_collector.start()
    
    # Run the app
    app.run_server(debug=True, host='0.0.0.0', port=8050)