# Production Environment for TrustformeRS Serve

terraform {
  required_version = ">= 1.5"
  
  # Configure remote backend for production
  backend "s3" {
    bucket         = "trustformers-terraform-state"
    key            = "production/trustformers-serve/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "trustformers-terraform-locks"
  }
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = "production"
      Project     = "trustformers-serve"
      ManagedBy   = "terraform"
      Team        = "ml-platform"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Deploy TrustformeRS Serve infrastructure
module "trustformers_serve" {
  source = "../../modules/trustformers-serve"
  
  # General configuration
  project_name = "trustformers"
  environment  = "production"
  
  # Network configuration
  vpc_cidr            = "10.0.0.0/16"
  enable_nat_gateway  = true
  enable_vpn_gateway  = false
  allowed_cidr_blocks = var.allowed_cidr_blocks
  
  # Container configuration
  container_registry = var.container_registry
  container_image   = "trustformers-serve"
  container_tag     = var.container_tag
  
  # Database configuration
  enable_rds                          = true
  db_engine                          = "postgres"
  db_engine_version                  = "15.4"
  db_instance_class                  = "db.r6g.large"
  db_allocated_storage               = 100
  db_max_allocated_storage           = 1000
  db_storage_type                    = "gp3"
  db_backup_retention_period         = 30
  db_backup_window                   = "03:00-04:00"
  db_maintenance_window              = "sun:04:00-sun:05:00"
  db_monitoring_interval             = 60
  db_enabled_cloudwatch_logs_exports = ["postgresql"]
  db_deletion_protection             = true
  db_skip_final_snapshot             = false
  
  # Redis configuration
  enable_redis                   = true
  redis_node_type               = "cache.r6g.large"
  redis_num_cache_nodes         = 3
  redis_engine_version          = "7.0"
  redis_snapshot_retention_limit = 7
  redis_snapshot_window         = "03:00-05:00"
  redis_maintenance_window      = "sun:05:00-sun:07:00"
  
  # EKS configuration
  enable_eks           = true
  eks_kubernetes_version = "1.27"
  eks_endpoint_config = {
    private_access      = true
    public_access       = true
    public_access_cidrs = var.allowed_cidr_blocks
  }
  
  eks_node_groups = {
    general = {
      instance_types = ["m6i.large", "m5.large"]
      capacity_type  = "ON_DEMAND"
      min_size      = 2
      max_size      = 10
      desired_size  = 3
      disk_size     = 100
      ami_type      = "AL2_x86_64"
      labels = {
        role = "general"
      }
      taints = []
    }
    
    inference = {
      instance_types = ["c6i.xlarge", "c5.xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size      = 1
      max_size      = 20
      desired_size  = 2
      disk_size     = 100
      ami_type      = "AL2_x86_64"
      labels = {
        role = "inference"
        "node.kubernetes.io/instance-type" = "compute-optimized"
      }
      taints = [
        {
          key    = "workload"
          value  = "inference"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  eks_addons = {
    "vpc-cni" = {
      version = "v1.13.4-eksbuild.1"
    }
    "coredns" = {
      version = "v1.10.1-eksbuild.1"
    }
    "kube-proxy" = {
      version = "v1.27.3-eksbuild.1"
    }
    "aws-ebs-csi-driver" = {
      version = "v1.21.0-eksbuild.1"
    }
  }
  
  # Load balancer configuration
  enable_load_balancer = true
  lb_type             = "application"
  lb_scheme           = "internet-facing"
  ssl_certificate_arn = var.ssl_certificate_arn
  ssl_policy         = "ELBSecurityPolicy-TLS-1-2-Ext-2018-06"
  
  lb_target_groups = {
    http = {
      port              = 8080
      protocol          = "HTTP"
      target_type       = "ip"  # For EKS
      health_check_path = "/health"
      health_check_port = "8080"
      matcher           = "200"
    }
    grpc = {
      port              = 9090
      protocol          = "HTTP"
      target_type       = "ip"  # For EKS
      health_check_path = "/health"
      health_check_port = "8080"
      matcher           = "200"
    }
  }
  
  # Auto scaling is disabled when using EKS
  enable_autoscaling = false
  
  # Monitoring configuration
  enable_monitoring        = true
  monitoring_sns_topic_arn = var.monitoring_sns_topic_arn
  
  monitoring_alarm_thresholds = {
    cpu_high_threshold        = 70
    memory_high_threshold     = 80
    disk_usage_threshold      = 85
    response_time_threshold   = 2
    error_rate_threshold      = 2
    database_cpu_threshold    = 70
    database_memory_threshold = 80
    redis_cpu_threshold       = 70
    redis_memory_threshold    = 80
  }
  
  # Storage configuration
  enable_s3_storage     = true
  s3_versioning_enabled = true
  s3_encryption_enabled = true
  
  s3_lifecycle_rules = [
    {
      id      = "model_lifecycle"
      enabled = true
      expiration_days = 2555  # 7 years
      transition_days = 90
      transition_storage_class = "STANDARD_IA"
    },
    {
      id      = "logs_lifecycle"
      enabled = true
      expiration_days = 365   # 1 year
      transition_days = 30
      transition_storage_class = "GLACIER"
    }
  ]
  
  # Security configuration
  secrets_recovery_window_days = 30
  jwt_secret                  = var.jwt_secret
  api_keys                    = var.api_keys
  
  # Additional tags
  tags = {
    CostCenter    = "ml-platform"
    Owner         = "platform-team"
    Backup        = "required"
    Monitoring    = "critical"
    Compliance    = "sox"
    DataClass     = "confidential"
  }
}

# Additional production-specific resources

# WAF for additional security
resource "aws_wafv2_web_acl" "main" {
  name  = "trustformers-serve-production"
  scope = "REGIONAL"
  
  default_action {
    allow {}
  }
  
  # Rate limiting rule
  rule {
    name     = "rate-limit"
    priority = 1
    
    action {
      block {}
    }
    
    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }
  }
  
  # Geo blocking rule (example)
  rule {
    name     = "geo-blocking"
    priority = 2
    
    action {
      block {}
    }
    
    statement {
      geo_match_statement {
        country_codes = var.blocked_country_codes
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "GeoBlockingRule"
      sampled_requests_enabled   = true
    }
  }
  
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "TrustformersServeWAF"
    sampled_requests_enabled   = true
  }
  
  tags = {
    Name        = "trustformers-serve-production"
    Environment = "production"
  }
}

# Associate WAF with Load Balancer
resource "aws_wafv2_web_acl_association" "main" {
  resource_arn = module.trustformers_serve.load_balancer_arn
  web_acl_arn  = aws_wafv2_web_acl.main.arn
}

# CloudTrail for audit logging
resource "aws_cloudtrail" "main" {
  name           = "trustformers-serve-production"
  s3_bucket_name = aws_s3_bucket.cloudtrail.bucket
  
  include_global_service_events = true
  is_multi_region_trail        = true
  enable_logging               = true
  
  event_selector {
    read_write_type                 = "All"
    include_management_events       = true
    exclude_management_event_sources = []
    
    data_resource {
      type   = "AWS::S3::Object"
      values = ["${module.trustformers_serve.s3_bucket_arn}/*"]
    }
  }
  
  tags = {
    Name        = "trustformers-serve-production"
    Environment = "production"
  }
}

# S3 bucket for CloudTrail logs
resource "aws_s3_bucket" "cloudtrail" {
  bucket        = "trustformers-serve-cloudtrail-${random_id.bucket_suffix.hex}"
  force_destroy = false
  
  tags = {
    Name        = "trustformers-serve-cloudtrail"
    Environment = "production"
  }
}

resource "aws_s3_bucket_versioning" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_policy" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AWSCloudTrailAclCheck"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.cloudtrail.arn
      },
      {
        Sid    = "AWSCloudTrailWrite"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.cloudtrail.arn}/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
          }
        }
      }
    ]
  })
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Backup vault for additional protection
resource "aws_backup_vault" "main" {
  name        = "trustformers-serve-production"
  kms_key_arn = aws_kms_key.backup.arn
  
  tags = {
    Name        = "trustformers-serve-production"
    Environment = "production"
  }
}

resource "aws_kms_key" "backup" {
  description             = "KMS key for TrustformeRS Serve backups"
  deletion_window_in_days = 30
  
  tags = {
    Name        = "trustformers-serve-backup-key"
    Environment = "production"
  }
}

resource "aws_kms_alias" "backup" {
  name          = "alias/trustformers-serve-backup"
  target_key_id = aws_kms_key.backup.key_id
}