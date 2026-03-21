# TrustformeRS Serve Terraform Infrastructure

This directory contains Terraform modules and configurations for deploying TrustformeRS Serve infrastructure on AWS. The infrastructure supports both traditional EC2-based deployments and modern Kubernetes-based deployments using Amazon EKS.

## Architecture Overview

The Terraform modules deploy a complete, production-ready infrastructure including:

- **Network Layer**: VPC with public/private subnets across multiple AZs
- **Compute Layer**: EKS cluster or Auto Scaling Groups for application hosting  
- **Data Layer**: RDS PostgreSQL and ElastiCache Redis for data persistence
- **Load Balancing**: Application Load Balancer with SSL termination
- **Storage**: S3 buckets for model artifacts and configuration
- **Security**: WAF, Security Groups, IAM roles, and Secrets Manager
- **Monitoring**: CloudWatch dashboards, alarms, and logging
- **Backup**: Automated backup solutions with retention policies

## Directory Structure

```
terraform/
├── modules/                    # Reusable Terraform modules
│   └── trustformers-serve/   # Main module for TrustformeRS Serve
│       ├── main.tf           # Main module configuration
│       ├── variables.tf      # Module input variables
│       ├── outputs.tf        # Module outputs
│       └── modules/          # Sub-modules
│           ├── vpc/          # VPC and networking
│           ├── security/     # Security groups and policies  
│           ├── database/     # RDS configuration
│           ├── redis/        # ElastiCache configuration
│           ├── eks/          # EKS cluster configuration
│           ├── load_balancer/# ALB configuration
│           ├── autoscaling/  # EC2 Auto Scaling
│           ├── monitoring/   # CloudWatch monitoring
│           ├── storage/      # S3 storage
│           └── iam/          # IAM roles and policies
├── environments/             # Environment-specific configurations
│   ├── production/          # Production environment
│   ├── staging/             # Staging environment
│   └── development/         # Development environment
└── examples/                # Example configurations
    ├── simple/              # Simple single-region deployment
    ├── multi-region/        # Multi-region deployment
    └── hybrid/              # Hybrid cloud deployment
```

## Quick Start

### Prerequisites

1. **Terraform**: Version 1.5 or later
2. **AWS CLI**: Configured with appropriate credentials
3. **kubectl**: For EKS cluster management (if using EKS)
4. **Docker**: For local development and testing

### Basic Deployment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cool-japan/trustformers.git
   cd trustformers/trustformers-serve/terraform
   ```

2. **Configure backend** (recommended for production):
   ```bash
   # Create S3 bucket for state storage
   aws s3 mb s3://your-terraform-state-bucket
   
   # Create DynamoDB table for state locking
   aws dynamodb create-table \
     --table-name terraform-locks \
     --attribute-definitions AttributeName=LockID,AttributeType=S \
     --key-schema AttributeName=LockID,KeyType=HASH \
     --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
   ```

3. **Deploy production environment**:
   ```bash
   cd environments/production
   
   # Initialize Terraform
   terraform init
   
   # Plan deployment
   terraform plan -var-file="production.tfvars"
   
   # Apply changes
   terraform apply -var-file="production.tfvars"
   ```

### Environment Configuration

Create a `production.tfvars` file with your specific configuration:

```hcl
# production.tfvars
aws_region = "us-west-2"
allowed_cidr_blocks = ["10.0.0.0/8", "172.16.0.0/12"]
container_registry = "your-registry.com"
container_tag = "v1.0.0"
ssl_certificate_arn = "arn:aws:acm:us-west-2:123456789012:certificate/12345678-1234-1234-1234-123456789012"
monitoring_sns_topic_arn = "arn:aws:sns:us-west-2:123456789012:alerts"

# Security
api_keys = {
  "api-key-1" = "client-1"
  "api-key-2" = "client-2"
}

blocked_country_codes = ["CN", "RU", "KP"]
```

## Module Usage

### Simple Usage

```hcl
module "trustformers_serve" {
  source = "./modules/trustformers-serve"
  
  project_name = "my-project"
  environment  = "production"
  vpc_cidr     = "10.0.0.0/16"
  
  # Enable EKS
  enable_eks = true
  eks_node_groups = {
    main = {
      instance_types = ["m6i.large"]
      min_size      = 2
      max_size      = 10
      desired_size  = 3
    }
  }
  
  # Database configuration
  db_instance_class = "db.r6g.large"
  db_allocated_storage = 100
  
  # Redis configuration  
  redis_node_type = "cache.r6g.large"
  redis_num_cache_nodes = 3
}
```

### Advanced Configuration

```hcl
module "trustformers_serve" {
  source = "./modules/trustformers-serve"
  
  # Basic configuration
  project_name = "trustformers"
  environment  = "production"
  
  # Network configuration
  vpc_cidr = "10.0.0.0/16"
  enable_nat_gateway = true
  allowed_cidr_blocks = ["10.0.0.0/8"]
  
  # Container configuration
  container_registry = "123456789012.dkr.ecr.us-west-2.amazonaws.com"
  container_image = "trustformers-serve"
  container_tag = "v2.1.0"
  
  # EKS configuration
  enable_eks = true
  eks_kubernetes_version = "1.27"
  
  eks_node_groups = {
    general = {
      instance_types = ["m6i.large", "m5.large"]
      capacity_type  = "ON_DEMAND"
      min_size      = 2
      max_size      = 10
      desired_size  = 3
      disk_size     = 100
      labels = { role = "general" }
    }
    
    inference = {
      instance_types = ["c6i.xlarge", "c5.xlarge"]
      capacity_type  = "SPOT"
      min_size      = 1
      max_size      = 20
      desired_size  = 2
      disk_size     = 100
      labels = { role = "inference" }
      taints = [{
        key    = "workload"
        value  = "inference"
        effect = "NO_SCHEDULE"
      }]
    }
  }
  
  # Database configuration
  db_engine_version = "15.4"
  db_instance_class = "db.r6g.xlarge"
  db_allocated_storage = 500
  db_max_allocated_storage = 2000
  db_backup_retention_period = 30
  db_deletion_protection = true
  
  # Redis configuration
  redis_node_type = "cache.r6g.xlarge"
  redis_num_cache_nodes = 3
  redis_engine_version = "7.0"
  
  # Load balancer configuration
  enable_load_balancer = true
  ssl_certificate_arn = var.ssl_certificate_arn
  ssl_policy = "ELBSecurityPolicy-TLS-1-2-Ext-2018-06"
  
  # Monitoring configuration
  enable_monitoring = true
  monitoring_alarm_thresholds = {
    cpu_high_threshold = 70
    memory_high_threshold = 80
    error_rate_threshold = 2
  }
  
  # Storage configuration
  enable_s3_storage = true
  s3_versioning_enabled = true
  s3_encryption_enabled = true
  
  # Tags
  tags = {
    CostCenter = "ml-platform"
    Team       = "platform-team"
    Backup     = "required"
  }
}
```

## Deployment Scenarios

### 1. EKS-based Deployment (Recommended)

Best for: Production workloads, high scalability, modern container orchestration

```hcl
# Enable EKS cluster
enable_eks = true
enable_autoscaling = false  # Disabled when using EKS

eks_node_groups = {
  general = {
    instance_types = ["m6i.large"]
    min_size      = 2
    max_size      = 10
    desired_size  = 3
  }
}
```

### 2. EC2 Auto Scaling Deployment

Best for: Traditional deployments, specific instance requirements

```hcl
# Enable Auto Scaling Groups
enable_eks = false
enable_autoscaling = true

asg_instance_type = "c6i.xlarge"
asg_min_size = 2
asg_max_size = 10
asg_desired_capacity = 3
```

### 3. Development Environment

Minimal setup for development and testing:

```hcl
# Minimal configuration
enable_eks = false
enable_autoscaling = true
enable_nat_gateway = false  # Cost optimization

# Smaller instances
db_instance_class = "db.t3.micro"
redis_node_type = "cache.t3.micro"
asg_instance_type = "t3.small"
```

## Security Features

### Network Security

- **VPC**: Isolated virtual network with public/private subnets
- **Security Groups**: Granular firewall rules for each service
- **NACLs**: Additional network-level security controls
- **VPC Flow Logs**: Network traffic monitoring and auditing

### Application Security

- **WAF**: Web Application Firewall with rate limiting and geo-blocking
- **SSL/TLS**: Automatic HTTPS termination with ACM certificates
- **Secrets Management**: AWS Secrets Manager for sensitive data
- **IAM**: Least-privilege access controls and service roles

### Data Security

- **Encryption at Rest**: All data encrypted using AWS KMS
- **Encryption in Transit**: TLS encryption for all communications
- **Database Security**: RDS with encryption and automated backups
- **Backup Security**: Encrypted backups with retention policies

## Monitoring and Observability

### CloudWatch Integration

- **Metrics**: Custom application metrics and AWS service metrics
- **Alarms**: Automated alerting based on thresholds
- **Dashboards**: Pre-configured monitoring dashboards
- **Logs**: Centralized log aggregation and analysis

### Available Metrics

- Application performance (latency, throughput, errors)
- Infrastructure health (CPU, memory, disk, network)
- Database performance (connections, queries, locks)
- Cache performance (hit ratio, memory usage)
- Load balancer metrics (requests, targets, response times)

### Alerting

Configure SNS topics for alert notifications:

```hcl
monitoring_sns_topic_arn = "arn:aws:sns:us-west-2:123456789012:alerts"

monitoring_alarm_thresholds = {
  cpu_high_threshold = 80
  memory_high_threshold = 85
  error_rate_threshold = 5
  response_time_threshold = 2
}
```

## Cost Optimization

### Instance Right-sizing

- Use Spot instances for non-critical workloads
- Implement auto-scaling policies
- Choose appropriate instance families

### Storage Optimization

- Use GP3 storage for better price-performance
- Implement S3 lifecycle policies
- Enable storage tiering

### Network Optimization

- Use VPC endpoints to reduce NAT Gateway costs
- Optimize data transfer patterns
- Consider regional deployment strategies

### Reserved Capacity

For production workloads, consider:
- RDS Reserved Instances
- ElastiCache Reserved Nodes
- EC2 Reserved Instances or Savings Plans

## Backup and Disaster Recovery

### Automated Backups

- **RDS**: Automated daily backups with point-in-time recovery
- **ElastiCache**: Automated snapshots with configurable retention
- **S3**: Cross-region replication for critical data
- **EKS**: Velero for application-aware backups

### Disaster Recovery

- Multi-AZ deployment for high availability
- Cross-region backup replication
- Infrastructure as Code for rapid recovery
- Automated failover capabilities

## Troubleshooting

### Common Issues

1. **State Lock Issues**:
   ```bash
   # Force unlock if necessary
   terraform force-unlock <lock-id>
   ```

2. **Resource Limits**:
   ```bash
   # Check AWS service limits
   aws service-quotas get-service-quota \
     --service-code ec2 \
     --quota-code L-1216C47A
   ```

3. **EKS Access Issues**:
   ```bash
   # Update kubeconfig
   aws eks update-kubeconfig \
     --region us-west-2 \
     --name trustformers-production
   ```

### Debugging

Enable Terraform debugging:
```bash
export TF_LOG=DEBUG
terraform apply
```

### Support Resources

- [AWS EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)
- [Terraform AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [TrustformeRS Documentation](../README.md)

## Contributing

1. **Module Development**: Follow Terraform best practices
2. **Testing**: Test modules in isolated environments
3. **Documentation**: Update README for any changes
4. **Versioning**: Use semantic versioning for module releases

## Examples

See the `examples/` directory for:
- Simple single-region deployment
- Multi-region deployment with replication
- Hybrid cloud deployment patterns
- Development environment setup
- Production hardening examples

## License

This Terraform configuration is licensed under the same terms as the main TrustformeRS project.