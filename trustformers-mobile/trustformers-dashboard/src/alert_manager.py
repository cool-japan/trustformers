"""
Alert manager for TrustformeRS performance monitoring.
Handles threshold-based alerts and notifications.
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import sqlite3
from collections import deque
import numpy as np

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"

@dataclass
class AlertRule:
    """Definition of an alert rule."""
    id: str
    name: str
    metric: str
    model: Optional[str]
    device: Optional[str]
    threshold: float
    condition: str  # 'above' or 'below'
    severity: AlertSeverity
    duration_seconds: int = 60  # How long condition must persist
    cooldown_seconds: int = 300  # Minimum time between alerts
    enabled: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class Alert:
    """Active alert instance."""
    id: str
    rule_id: str
    timestamp: datetime
    status: AlertStatus
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    model: Optional[str]
    device: Optional[str]
    metadata: Dict[str, Any] = None

class AlertManager:
    def __init__(self, db_path: str = "data/alerts.db"):
        self.db_path = db_path
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=1000)
        self.running = False
        self.thread = None
        
        # Notification handlers
        self.notification_handlers = {
            'console': self._console_notification,
            'webhook': self._webhook_notification,
            'email': self._email_notification,
        }
        
        # Metric evaluators
        self.metric_evaluators = {
            'latency': self._evaluate_latency,
            'throughput': self._evaluate_throughput,
            'memory': self._evaluate_memory,
            'error_rate': self._evaluate_error_rate,
            'gpu_utilization': self._evaluate_gpu_utilization,
            'queue_depth': self._evaluate_queue_depth,
        }
        
        # Alert state tracking
        self.metric_state = {}
        self.last_alert_time = {}
        
        self._init_database()
        self._load_rules()
    
    def _init_database(self):
        """Initialize alert database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    metric TEXT,
                    model TEXT,
                    device TEXT,
                    threshold REAL,
                    condition TEXT,
                    severity TEXT,
                    duration_seconds INTEGER,
                    cooldown_seconds INTEGER,
                    enabled BOOLEAN,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alert_history (
                    id TEXT PRIMARY KEY,
                    rule_id TEXT,
                    timestamp TIMESTAMP,
                    status TEXT,
                    severity TEXT,
                    message TEXT,
                    value REAL,
                    threshold REAL,
                    model TEXT,
                    device TEXT,
                    metadata TEXT,
                    FOREIGN KEY (rule_id) REFERENCES alert_rules(id)
                )
            ''')
    
    def _load_rules(self):
        """Load alert rules from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM alert_rules WHERE enabled = 1')
            
            for row in cursor:
                rule = AlertRule(
                    id=row[0],
                    name=row[1],
                    metric=row[2],
                    model=row[3],
                    device=row[4],
                    threshold=row[5],
                    condition=row[6],
                    severity=AlertSeverity(row[7]),
                    duration_seconds=row[8],
                    cooldown_seconds=row[9],
                    enabled=bool(row[10]),
                    metadata=json.loads(row[11]) if row[11] else {}
                )
                self.rules[rule.id] = rule
    
    def add_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.rules[rule.id] = rule
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO alert_rules 
                (id, name, metric, model, device, threshold, condition, severity,
                 duration_seconds, cooldown_seconds, enabled, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.id, rule.name, rule.metric, rule.model, rule.device,
                rule.threshold, rule.condition, rule.severity.value,
                rule.duration_seconds, rule.cooldown_seconds, rule.enabled,
                json.dumps(rule.metadata) if rule.metadata else None
            ))
    
    def remove_rule(self, rule_id: str):
        """Remove an alert rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM alert_rules WHERE id = ?', (rule_id,))
    
    def start(self):
        """Start alert monitoring."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """Stop alert monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                for rule_id, rule in self.rules.items():
                    if rule.enabled:
                        self._evaluate_rule(rule)
                
                # Check for resolved alerts
                self._check_resolved_alerts()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Error in alert monitoring: {e}")
                time.sleep(10)
    
    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        # Get metric value
        evaluator = self.metric_evaluators.get(rule.metric)
        if not evaluator:
            return
        
        value = evaluator(rule.model, rule.device)
        if value is None:
            return
        
        # Check threshold
        condition_met = False
        if rule.condition == 'above' and value > rule.threshold:
            condition_met = True
        elif rule.condition == 'below' and value < rule.threshold:
            condition_met = True
        
        # Track state
        state_key = f"{rule.id}:{rule.model}:{rule.device}"
        
        if condition_met:
            # Update or initialize state
            if state_key not in self.metric_state:
                self.metric_state[state_key] = {
                    'first_seen': datetime.now(),
                    'last_value': value,
                    'values': [value]
                }
            else:
                self.metric_state[state_key]['last_value'] = value
                self.metric_state[state_key]['values'].append(value)
            
            # Check if duration requirement is met
            state = self.metric_state[state_key]
            duration = (datetime.now() - state['first_seen']).total_seconds()
            
            if duration >= rule.duration_seconds:
                # Check cooldown
                last_alert = self.last_alert_time.get(state_key)
                if not last_alert or (datetime.now() - last_alert).total_seconds() >= rule.cooldown_seconds:
                    # Create alert
                    self._create_alert(rule, value, state['values'])
                    self.last_alert_time[state_key] = datetime.now()
        else:
            # Condition not met - clear state
            if state_key in self.metric_state:
                del self.metric_state[state_key]
    
    def _create_alert(self, rule: AlertRule, value: float, recent_values: List[float]):
        """Create a new alert."""
        alert_id = f"{rule.id}:{int(time.time() * 1000)}"
        
        # Build message
        avg_value = np.mean(recent_values)
        trend = "increasing" if len(recent_values) > 1 and recent_values[-1] > recent_values[0] else "stable"
        
        message = (
            f"{rule.name}: {rule.metric} is {rule.condition} threshold\n"
            f"Current: {value:.2f}, Threshold: {rule.threshold:.2f}\n"
            f"Average: {avg_value:.2f}, Trend: {trend}"
        )
        
        if rule.model:
            message += f"\nModel: {rule.model}"
        if rule.device:
            message += f"\nDevice: {rule.device}"
        
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            timestamp=datetime.now(),
            status=AlertStatus.ACTIVE,
            severity=rule.severity,
            message=message,
            value=value,
            threshold=rule.threshold,
            model=rule.model,
            device=rule.device,
            metadata={
                'recent_values': recent_values[-10:],  # Last 10 values
                'trend': trend
            }
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Store in database
        self._store_alert(alert)
        
        # Send notifications
        self._send_notifications(alert)
    
    def _store_alert(self, alert: Alert):
        """Store alert in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO alert_history 
                (id, rule_id, timestamp, status, severity, message, value, 
                 threshold, model, device, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.rule_id, alert.timestamp, alert.status.value,
                alert.severity.value, alert.message, alert.value, alert.threshold,
                alert.model, alert.device, json.dumps(alert.metadata)
            ))
    
    def _check_resolved_alerts(self):
        """Check if any active alerts should be resolved."""
        for alert_id, alert in list(self.active_alerts.items()):
            if alert.status != AlertStatus.ACTIVE:
                continue
            
            rule = self.rules.get(alert.rule_id)
            if not rule:
                continue
            
            # Re-evaluate condition
            evaluator = self.metric_evaluators.get(rule.metric)
            if evaluator:
                value = evaluator(rule.model, rule.device)
                if value is not None:
                    condition_met = False
                    if rule.condition == 'above' and value > rule.threshold:
                        condition_met = True
                    elif rule.condition == 'below' and value < rule.threshold:
                        condition_met = True
                    
                    if not condition_met:
                        # Alert resolved
                        alert.status = AlertStatus.RESOLVED
                        self._update_alert_status(alert)
                        del self.active_alerts[alert_id]
                        
                        # Send resolution notification
                        resolution_message = f"RESOLVED: {alert.message}\nCurrent value: {value:.2f}"
                        self._send_notifications(alert, resolution_message)
    
    def _update_alert_status(self, alert: Alert):
        """Update alert status in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE alert_history 
                SET status = ? 
                WHERE id = ?
            ''', (alert.status.value, alert.id))
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            self._update_alert_status(alert)
    
    def silence_alert(self, alert_id: str, duration_minutes: int = 60):
        """Silence an alert for a specified duration."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SILENCED
            alert.metadata['silence_until'] = (
                datetime.now() + timedelta(minutes=duration_minutes)
            ).isoformat()
            self._update_alert_status(alert)
    
    def _send_notifications(self, alert: Alert, custom_message: str = None):
        """Send alert notifications."""
        message = custom_message or alert.message
        
        # Send to all configured handlers
        for handler_name, handler in self.notification_handlers.items():
            try:
                handler(alert, message)
            except Exception as e:
                print(f"Failed to send {handler_name} notification: {e}")
    
    def _console_notification(self, alert: Alert, message: str):
        """Print alert to console."""
        severity_colors = {
            AlertSeverity.INFO: '\033[94m',      # Blue
            AlertSeverity.WARNING: '\033[93m',   # Yellow
            AlertSeverity.ERROR: '\033[91m',     # Red
            AlertSeverity.CRITICAL: '\033[95m'   # Magenta
        }
        
        color = severity_colors.get(alert.severity, '')
        reset = '\033[0m'
        
        print(f"{color}[{alert.severity.value.upper()}] {message}{reset}")
    
    def _webhook_notification(self, alert: Alert, message: str):
        """Send webhook notification."""
        webhook_url = os.environ.get('TRUSTFORMERS_ALERT_WEBHOOK')
        if not webhook_url:
            return
        
        payload = {
            'alert_id': alert.id,
            'severity': alert.severity.value,
            'message': message,
            'timestamp': alert.timestamp.isoformat(),
            'value': alert.value,
            'threshold': alert.threshold,
            'model': alert.model,
            'device': alert.device
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=5)
            response.raise_for_status()
        except Exception as e:
            print(f"Webhook notification failed: {e}")
    
    def _email_notification(self, alert: Alert, message: str):
        """Send email notification."""
        smtp_server = os.environ.get('TRUSTFORMERS_SMTP_SERVER')
        smtp_port = int(os.environ.get('TRUSTFORMERS_SMTP_PORT', 587))
        smtp_user = os.environ.get('TRUSTFORMERS_SMTP_USER')
        smtp_password = os.environ.get('TRUSTFORMERS_SMTP_PASSWORD')
        recipient = os.environ.get('TRUSTFORMERS_ALERT_EMAIL')
        
        if not all([smtp_server, smtp_user, smtp_password, recipient]):
            return
        
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = recipient
        msg['Subject'] = f"[{alert.severity.value.upper()}] TrustformeRS Alert"
        
        body = f"{message}\n\nAlert ID: {alert.id}\nTimestamp: {alert.timestamp}"
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
        except Exception as e:
            print(f"Email notification failed: {e}")
    
    # Metric evaluators
    def _evaluate_latency(self, model: Optional[str], device: Optional[str]) -> Optional[float]:
        """Evaluate latency metric."""
        # This would query actual metrics from data_collector
        # For now, return mock data
        return np.random.uniform(50, 150)
    
    def _evaluate_throughput(self, model: Optional[str], device: Optional[str]) -> Optional[float]:
        """Evaluate throughput metric."""
        return np.random.uniform(100, 1000)
    
    def _evaluate_memory(self, model: Optional[str], device: Optional[str]) -> Optional[float]:
        """Evaluate memory usage metric."""
        return np.random.uniform(1000, 8000)
    
    def _evaluate_error_rate(self, model: Optional[str], device: Optional[str]) -> Optional[float]:
        """Evaluate error rate metric."""
        return np.random.uniform(0, 0.05)
    
    def _evaluate_gpu_utilization(self, model: Optional[str], device: Optional[str]) -> Optional[float]:
        """Evaluate GPU utilization metric."""
        return np.random.uniform(0, 100)
    
    def _evaluate_queue_depth(self, model: Optional[str], device: Optional[str]) -> Optional[float]:
        """Evaluate request queue depth metric."""
        return np.random.uniform(0, 50)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        alerts = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM alert_history 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            for row in cursor:
                alert = Alert(
                    id=row[0],
                    rule_id=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    status=AlertStatus(row[3]),
                    severity=AlertSeverity(row[4]),
                    message=row[5],
                    value=row[6],
                    threshold=row[7],
                    model=row[8],
                    device=row[9],
                    metadata=json.loads(row[10]) if row[10] else {}
                )
                alerts.append(alert)
        
        return alerts