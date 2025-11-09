#!/usr/bin/env python3
"""
Entry point for TrustformeRS Performance Dashboard.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from app import app
from data_collector import DataCollector
from alert_manager import AlertManager, AlertRule, AlertSeverity

def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def setup_default_alerts():
    """Set up default alert rules."""
    alert_manager = AlertManager()
    
    # High latency alert
    alert_manager.add_rule(AlertRule(
        id='high_latency',
        name='High Inference Latency',
        metric='latency',
        model=None,  # All models
        device=None,  # All devices
        threshold=200.0,  # 200ms
        condition='above',
        severity=AlertSeverity.WARNING,
        duration_seconds=60,
        cooldown_seconds=300
    ))
    
    # Low throughput alert
    alert_manager.add_rule(AlertRule(
        id='low_throughput',
        name='Low Throughput',
        metric='throughput',
        model=None,
        device=None,
        threshold=10.0,  # 10 samples/sec
        condition='below',
        severity=AlertSeverity.WARNING,
        duration_seconds=120,
        cooldown_seconds=600
    ))
    
    # High memory usage alert
    alert_manager.add_rule(AlertRule(
        id='high_memory',
        name='High Memory Usage',
        metric='memory',
        model=None,
        device=None,
        threshold=7000.0,  # 7GB
        condition='above',
        severity=AlertSeverity.ERROR,
        duration_seconds=30,
        cooldown_seconds=300
    ))
    
    # High error rate alert
    alert_manager.add_rule(AlertRule(
        id='high_errors',
        name='High Error Rate',
        metric='error_rate',
        model=None,
        device=None,
        threshold=0.01,  # 1% error rate
        condition='above',
        severity=AlertSeverity.CRITICAL,
        duration_seconds=30,
        cooldown_seconds=180
    ))
    
    return alert_manager

def main():
    parser = argparse.ArgumentParser(
        description='TrustformeRS Performance Dashboard'
    )
    parser.add_argument(
        '--host', 
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8050,
        help='Port to bind to (default: 8050)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Run in debug mode'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--no-collect', 
        action='store_true',
        help='Disable automatic data collection'
    )
    parser.add_argument(
        '--no-alerts', 
        action='store_true',
        help='Disable alert monitoring'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory for storing data (default: data)'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Create data directory
    Path(args.data_dir).mkdir(exist_ok=True)
    
    # Start data collection
    if not args.no_collect:
        logger.info("Starting data collection...")
        data_collector = DataCollector(
            db_path=os.path.join(args.data_dir, 'metrics.db')
        )
        data_collector.start()
    
    # Start alert monitoring
    if not args.no_alerts:
        logger.info("Starting alert monitoring...")
        alert_manager = setup_default_alerts()
        alert_manager.start()
    
    # Start dashboard
    logger.info(f"Starting dashboard on {args.host}:{args.port}")
    logger.info(f"Open http://localhost:{args.port} in your browser")
    
    try:
        app.run_server(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if not args.no_collect:
            data_collector.stop()
        if not args.no_alerts:
            alert_manager.stop()

if __name__ == '__main__':
    main()