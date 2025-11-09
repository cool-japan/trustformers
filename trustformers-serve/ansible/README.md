# Ansible Automation for TrustformeRS Serve

This directory contains Ansible playbooks and roles for automated deployment and configuration management of TrustformeRS Serve infrastructure and applications.

## Overview

The Ansible automation provides:

- **Infrastructure Deployment**: Automated infrastructure provisioning with Terraform
- **Application Deployment**: Multi-environment application deployment and configuration
- **Configuration Management**: Consistent configuration across all environments
- **Security Hardening**: Automated security baseline implementation
- **Monitoring Setup**: Comprehensive monitoring and alerting configuration
- **Backup Management**: Automated backup and disaster recovery setup
- **Rolling Updates**: Zero-downtime application updates

## Directory Structure

```
ansible/
├── playbooks/           # Main playbooks
│   ├── deploy.yml      # Main deployment playbook
│   ├── update.yml      # Application update playbook
│   ├── backup.yml      # Backup operations playbook
│   └── disaster-recovery.yml
├── roles/              # Ansible roles
│   ├── common/         # Common system configuration
│   ├── docker/         # Docker installation and configuration
│   ├── trustformers-serve/  # Application-specific role
│   ├── postgresql/     # Database configuration
│   ├── redis/          # Redis configuration
│   ├── nginx/          # Load balancer configuration
│   ├── monitoring-agent/    # Monitoring setup
│   └── security-hardening/ # Security baseline
├── inventory/          # Environment inventories
│   ├── production.yml  # Production environment
│   ├── staging.yml     # Staging environment
│   └── development.yml # Development environment
├── group_vars/         # Environment-specific variables
│   ├── production.yml
│   ├── staging.yml
│   └── development.yml
├── host_vars/          # Host-specific variables
└── vault/              # Encrypted secrets (Ansible Vault)
```

## Quick Start

### Prerequisites

1. **Ansible**: Version 2.12 or later
2. **Python**: Version 3.8 or later
3. **AWS CLI**: Configured with appropriate credentials (for AWS deployments)
4. **SSH Access**: To target hosts with sudo privileges

### Installation

1. **Install Ansible**:
   ```bash
   pip install ansible
   
   # Install required collections
   ansible-galaxy install -r requirements.yml
   ```

2. **Configure SSH access**:
   ```bash
   # Add your SSH key to ssh-agent
   ssh-add ~/.ssh/your-key.pem
   
   # Test connectivity
   ansible all -i inventory/production.yml -m ping
   ```

3. **Set up Ansible Vault** (for sensitive data):
   ```bash
   # Create vault password file
   echo "your-vault-password" > .vault_pass
   chmod 600 .vault_pass
   
   # Encrypt sensitive variables
   ansible-vault create group_vars/production/vault.yml
   ```

### Basic Deployment

1. **Deploy to production**:
   ```bash
   # Full deployment (infrastructure + application)
   ansible-playbook -i inventory/production.yml playbooks/deploy.yml
   
   # Application only
   ansible-playbook -i inventory/production.yml playbooks/deploy.yml --skip-tags infrastructure
   
   # Specific environment
   ansible-playbook -i inventory/staging.yml playbooks/deploy.yml -e environment=staging
   ```

2. **Update application**:
   ```bash
   # Rolling update
   ansible-playbook -i inventory/production.yml playbooks/update.yml -e app_version=v1.1.0
   
   # Force update (restart all)
   ansible-playbook -i inventory/production.yml playbooks/update.yml -e force_restart=true
   ```

3. **Run backup**:
   ```bash
   ansible-playbook -i inventory/production.yml playbooks/backup.yml
   ```

## Deployment Options

### 1. Docker-based Deployment

Deploy TrustformeRS Serve as Docker containers:

```bash
ansible-playbook -i inventory/production.yml playbooks/deploy.yml \
  -e deployment_method=docker \
  -e container_tag=v1.0.0
```

### 2. Kubernetes Deployment

Deploy to existing Kubernetes cluster:

```bash
ansible-playbook -i inventory/production.yml playbooks/deploy.yml \
  -e deployment_type=kubernetes \
  -e k8s_namespace=trustformers-prod
```

### 3. Binary Deployment

Deploy as native binary:

```bash
ansible-playbook -i inventory/production.yml playbooks/deploy.yml \
  -e deployment_method=binary \
  -e app_version=v1.0.0
```

## Environment Configuration

### Production Environment

High-availability setup with:
- Multi-AZ deployment
- Load balancers with SSL termination
- Database clustering with automatic failover
- Comprehensive monitoring and alerting
- Automated backups with cross-region replication

```yaml
# group_vars/production.yml
environment: production
multi_az_deployment: true
enable_auto_scaling: true
backup_retention_days: 90
monitoring_enabled: true
```

### Staging Environment

Production-like setup for testing:
- Single-AZ deployment
- Load balancer for testing
- Database with daily backups
- Monitoring enabled

```yaml
# group_vars/staging.yml
environment: staging
multi_az_deployment: false
enable_auto_scaling: false
backup_retention_days: 7
```

### Development Environment

Minimal setup for development:
- Single server deployment
- No load balancer
- Local database
- Basic monitoring

```yaml
# group_vars/development.yml
environment: development
deploy_infrastructure: false
use_load_balancer: false
monitoring_enabled: false
```

## Playbook Usage

### Main Deployment Playbook

```bash
# Full deployment
ansible-playbook -i inventory/production.yml playbooks/deploy.yml

# Infrastructure only
ansible-playbook -i inventory/production.yml playbooks/deploy.yml --tags infrastructure

# Application only
ansible-playbook -i inventory/production.yml playbooks/deploy.yml --tags application

# Specific components
ansible-playbook -i inventory/production.yml playbooks/deploy.yml --tags database,monitoring
```

### Available Tags

- `infrastructure`: Terraform infrastructure deployment
- `application`: Application deployment and configuration
- `database`: Database setup and configuration
- `monitoring`: Monitoring and alerting setup
- `security`: Security hardening
- `backup`: Backup configuration
- `networking`: Network configuration
- `ssl`: SSL/TLS certificate setup

### Update Playbook

Rolling updates with zero downtime:

```bash
# Update application version
ansible-playbook -i inventory/production.yml playbooks/update.yml \
  -e app_version=v1.1.0

# Update configuration only
ansible-playbook -i inventory/production.yml playbooks/update.yml \
  --tags configuration

# Emergency update (faster, but with brief downtime)
ansible-playbook -i inventory/production.yml playbooks/update.yml \
  -e emergency_update=true
```

## Role Documentation

### Common Role

Base system configuration applied to all servers:

- System package management
- User and group creation
- Security hardening baseline
- System monitoring agent
- Log rotation configuration
- Backup script setup

### TrustformeRS Serve Role

Application-specific configuration:

- Application installation (binary/Docker)
- Configuration file generation
- Service management (systemd)
- Health check setup
- Log management
- Performance tuning

### Database Roles

PostgreSQL and Redis configuration:

- Database installation and configuration
- User and database creation
- Backup configuration
- Performance tuning
- Monitoring setup
- Security configuration

### Monitoring Role

Comprehensive monitoring setup:

- Prometheus node exporter
- Log shipping to centralized logging
- Custom application metrics
- Health check endpoints
- Alert rule configuration

## Security Features

### Automated Security Hardening

- SSH configuration hardening
- Firewall setup and configuration
- Fail2ban installation and configuration
- System update automation
- User access control
- File permission hardening

### Secrets Management

Using Ansible Vault for sensitive data:

```bash
# Create encrypted variables
ansible-vault create group_vars/production/vault.yml

# Edit encrypted variables
ansible-vault edit group_vars/production/vault.yml

# Encrypt existing file
ansible-vault encrypt group_vars/production/secrets.yml
```

Example vault file:
```yaml
# group_vars/production/vault.yml
vault_db_password: "super-secret-password"
vault_jwt_secret: "jwt-signing-secret"
vault_api_keys:
  client1: "api-key-1"
  client2: "api-key-2"
vault_slack_webhook_url: "https://hooks.slack.com/..."
```

## Monitoring and Alerting

### Metrics Collection

Automated setup of:
- Prometheus node exporter
- Application metrics endpoint
- Custom business metrics
- System performance metrics

### Log Management

- Centralized log collection
- Log rotation and retention
- Structured logging configuration
- Log forwarding to external systems

### Alerting Rules

Pre-configured alerts for:
- High CPU/memory usage
- Application errors
- Database performance issues
- Disk space warnings
- Service availability
- Security events

## Backup and Disaster Recovery

### Automated Backups

- Database backups with point-in-time recovery
- Application configuration backups
- Model and data backups
- Cross-region backup replication

### Disaster Recovery

- Automated failover procedures
- Recovery time optimization
- Data consistency verification
- Recovery testing automation

## Performance Optimization

### System Tuning

Automated optimization for:
- Kernel parameters
- Network settings
- File system optimization
- Memory management
- Process limits

### Application Tuning

- JVM optimization (if applicable)
- Connection pool sizing
- Cache configuration
- Thread pool optimization
- Resource allocation

## Troubleshooting

### Common Issues

1. **Connection Issues**:
   ```bash
   # Test connectivity
   ansible all -i inventory/production.yml -m ping
   
   # Check SSH configuration
   ansible all -i inventory/production.yml -m setup
   ```

2. **Permission Issues**:
   ```bash
   # Check sudo access
   ansible all -i inventory/production.yml -m shell -a "sudo whoami" --become
   ```

3. **Service Issues**:
   ```bash
   # Check service status
   ansible trustformers_servers -i inventory/production.yml -m systemd -a "name=trustformers-serve state=status"
   ```

### Debug Mode

Run playbooks with increased verbosity:

```bash
# Verbose output
ansible-playbook -vvv -i inventory/production.yml playbooks/deploy.yml

# Debug specific task
ansible-playbook -i inventory/production.yml playbooks/deploy.yml --start-at-task="Task name"
```

### Log Locations

- Ansible logs: `/var/log/ansible.log`
- Application logs: `/var/log/trustformers/`
- System logs: `/var/log/syslog` or `/var/log/messages`

## Best Practices

### Development Workflow

1. **Test in development environment first**
2. **Use staging for pre-production validation**
3. **Implement gradual rollouts for production**
4. **Maintain environment parity**
5. **Use version control for all configurations**

### Security Best Practices

1. **Use Ansible Vault for all secrets**
2. **Implement least-privilege access**
3. **Regular security updates**
4. **Monitor for security events**
5. **Implement network segmentation**

### Operational Best Practices

1. **Monitor deployment metrics**
2. **Implement health checks**
3. **Use rolling deployments**
4. **Maintain backup verification**
5. **Document incident procedures**

## Integration with CI/CD

### GitLab CI Example

```yaml
# .gitlab-ci.yml
deploy_staging:
  stage: deploy
  script:
    - ansible-playbook -i inventory/staging.yml playbooks/deploy.yml -e app_version=$CI_COMMIT_TAG
  only:
    - tags
  environment:
    name: staging

deploy_production:
  stage: deploy
  script:
    - ansible-playbook -i inventory/production.yml playbooks/deploy.yml -e app_version=$CI_COMMIT_TAG
  only:
    - tags
  when: manual
  environment:
    name: production
```

### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
name: Deploy TrustformeRS Serve
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to staging
        run: |
          ansible-playbook -i ansible/inventory/staging.yml ansible/playbooks/deploy.yml \
            -e app_version=${GITHUB_REF#refs/tags/}
```

## Contributing

1. **Role Development**: Follow Ansible best practices
2. **Testing**: Test playbooks in isolated environments  
3. **Documentation**: Update README for any changes
4. **Variables**: Use descriptive variable names
5. **Idempotency**: Ensure all tasks are idempotent

## Support

- **Documentation**: See individual role README files
- **Issues**: Report issues on GitHub
- **Slack**: #platform-team for real-time support
- **Email**: platform-team@example.com

## License

This Ansible configuration is licensed under the same terms as the main TrustformeRS project.