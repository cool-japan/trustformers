# Global Load Balancer Module for TrustformeRS Serve Multi-Cloud Deployment
# Provides global traffic distribution across multiple clouds

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.4"
    }
  }
}

locals {
  # Determine primary cloud provider for DNS and global resources
  is_aws_primary   = var.primary_cloud == "aws"
  is_azure_primary = var.primary_cloud == "azure"
  is_gcp_primary   = var.primary_cloud == "gcp"
  
  # Load balancer name
  lb_name = "${var.project_name}-${var.environment}-global-lb"
  
  # Common tags/labels
  common_tags = merge(var.tags, {
    Component   = "global-load-balancer"
    ManagedBy   = "terraform"
    Environment = var.environment
    Project     = var.project_name
  })
  
  # Active backends (only include enabled clouds)
  active_backends = {
    aws   = var.aws_backend.enabled
    azure = var.azure_backend.enabled
    gcp   = var.gcp_backend.enabled
  }
  
  # Traffic weights based on configuration
  traffic_weights = {
    aws   = var.aws_backend.enabled ? var.traffic_distribution.aws_weight : 0
    azure = var.azure_backend.enabled ? var.traffic_distribution.azure_weight : 0
    gcp   = var.gcp_backend.enabled ? var.traffic_distribution.gcp_weight : 0
  }
}

# Data sources
data "aws_caller_identity" "current" {
  count = local.is_aws_primary ? 1 : 0
}

data "aws_region" "current" {
  count = local.is_aws_primary ? 1 : 0
}

data "azurerm_client_config" "current" {
  count = local.is_azure_primary ? 1 : 0
}

data "google_client_config" "current" {
  count = local.is_gcp_primary ? 1 : 0
}

#############################################
# AWS Global Load Balancer (if AWS is primary)
#############################################

# Route 53 Hosted Zone (AWS Primary)
resource "aws_route53_zone" "main" {
  count = local.is_aws_primary ? 1 : 0
  name  = var.domain_name

  tags = local.common_tags
}

# ACM Certificate for AWS ALB (AWS Primary)
resource "aws_acm_certificate" "main" {
  count           = local.is_aws_primary ? 1 : 0
  domain_name     = var.domain_name
  validation_method = "DNS"

  subject_alternative_names = [
    "*.${var.domain_name}"
  ]

  lifecycle {
    create_before_destroy = true
  }

  tags = local.common_tags
}

# Certificate validation (AWS Primary)
resource "aws_route53_record" "cert_validation" {
  count = local.is_aws_primary ? length(aws_acm_certificate.main[0].domain_validation_options) : 0

  allow_overwrite = true
  name            = tolist(aws_acm_certificate.main[0].domain_validation_options)[count.index].resource_record_name
  records         = [tolist(aws_acm_certificate.main[0].domain_validation_options)[count.index].resource_record_value]
  type            = tolist(aws_acm_certificate.main[0].domain_validation_options)[count.index].resource_record_type
  zone_id         = aws_route53_zone.main[0].zone_id
  ttl             = 60
}

resource "aws_acm_certificate_validation" "main" {
  count           = local.is_aws_primary ? 1 : 0
  certificate_arn = aws_acm_certificate.main[0].arn
  validation_record_fqdns = aws_route53_record.cert_validation[*].fqdn

  timeouts {
    create = "5m"
  }
}

# AWS Application Load Balancer (AWS Primary)
resource "aws_lb" "global" {
  count              = local.is_aws_primary ? 1 : 0
  name               = "${local.lb_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb[0].id]
  subnets           = var.aws_subnet_ids

  enable_deletion_protection = var.enable_deletion_protection

  tags = local.common_tags
}

# Security Group for ALB (AWS Primary)
resource "aws_security_group" "alb" {
  count       = local.is_aws_primary ? 1 : 0
  name_prefix = "${local.lb_name}-alb-"
  vpc_id      = var.aws_vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# Target Groups for each cloud backend (AWS Primary)
resource "aws_lb_target_group" "aws" {
  count    = local.is_aws_primary && var.aws_backend.enabled ? 1 : 0
  name     = "${local.lb_name}-aws-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = var.aws_vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = var.aws_backend.health_check_path
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = local.common_tags
}

resource "aws_lb_target_group" "azure" {
  count    = local.is_aws_primary && var.azure_backend.enabled ? 1 : 0
  name     = "${local.lb_name}-azure-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = var.aws_vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = var.azure_backend.health_check_path
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = local.common_tags
}

resource "aws_lb_target_group" "gcp" {
  count    = local.is_aws_primary && var.gcp_backend.enabled ? 1 : 0
  name     = "${local.lb_name}-gcp-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = var.aws_vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = var.gcp_backend.health_check_path
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = local.common_tags
}

# ALB Listeners (AWS Primary)
resource "aws_lb_listener" "http" {
  count             = local.is_aws_primary ? 1 : 0
  load_balancer_arn = aws_lb.global[0].arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

resource "aws_lb_listener" "https" {
  count             = local.is_aws_primary ? 1 : 0
  load_balancer_arn = aws_lb.global[0].arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = var.ssl_policy
  certificate_arn   = aws_acm_certificate_validation.main[0].certificate_arn

  # Default action with weighted routing
  default_action {
    type = "forward"
    
    forward {
      dynamic "target_group" {
        for_each = var.aws_backend.enabled ? [1] : []
        content {
          arn    = aws_lb_target_group.aws[0].arn
          weight = local.traffic_weights.aws
        }
      }
      
      dynamic "target_group" {
        for_each = var.azure_backend.enabled ? [1] : []
        content {
          arn    = aws_lb_target_group.azure[0].arn
          weight = local.traffic_weights.azure
        }
      }
      
      dynamic "target_group" {
        for_each = var.gcp_backend.enabled ? [1] : []
        content {
          arn    = aws_lb_target_group.gcp[0].arn
          weight = local.traffic_weights.gcp
        }
      }
    }
  }
}

# Route 53 records for global load balancing (AWS Primary)
resource "aws_route53_record" "main" {
  count   = local.is_aws_primary ? 1 : 0
  zone_id = aws_route53_zone.main[0].zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_lb.global[0].dns_name
    zone_id                = aws_lb.global[0].zone_id
    evaluate_target_health = true
  }
}

# Health checks for failover (AWS Primary)
resource "aws_route53_health_check" "aws" {
  count                            = local.is_aws_primary && var.aws_backend.enabled ? 1 : 0
  fqdn                            = var.aws_backend.endpoint
  port                            = 443
  type                            = "HTTPS"
  resource_path                   = var.aws_backend.health_check_path
  failure_threshold               = "3"
  request_interval                = "30"
  cloudwatch_logs_region          = data.aws_region.current[0].name
  cloudwatch_logs_group_name      = "/aws/route53/healthcheck"
  insufficient_data_health_status = "Failure"

  tags = merge(local.common_tags, {
    Name = "${local.lb_name}-aws-health-check"
  })
}

#############################################
# Azure Global Load Balancer (if Azure is primary)
#############################################

# Azure DNS Zone (Azure Primary)
resource "azurerm_dns_zone" "main" {
  count               = local.is_azure_primary ? 1 : 0
  name                = var.domain_name
  resource_group_name = var.azure_resource_group_name

  tags = local.common_tags
}

# Azure Traffic Manager Profile (Azure Primary)
resource "azurerm_traffic_manager_profile" "main" {
  count               = local.is_azure_primary ? 1 : 0
  name                = local.lb_name
  resource_group_name = var.azure_resource_group_name

  traffic_routing_method = var.traffic_distribution.method

  dns_config {
    relative_name = local.lb_name
    ttl           = 60
  }

  monitor_config {
    protocol                     = "HTTPS"
    port                        = 443
    path                        = "/health"
    interval_in_seconds         = 30
    timeout_in_seconds          = 10
    tolerated_number_of_failures = 3
  }

  tags = local.common_tags
}

# Traffic Manager Endpoints (Azure Primary)
resource "azurerm_traffic_manager_external_endpoint" "aws" {
  count              = local.is_azure_primary && var.aws_backend.enabled ? 1 : 0
  name               = "aws-endpoint"
  profile_id         = azurerm_traffic_manager_profile.main[0].id
  target             = var.aws_backend.endpoint
  weight             = local.traffic_weights.aws
  priority           = var.failover_policy.aws_priority
  geo_mappings       = var.traffic_distribution.aws_geo_mappings
}

resource "azurerm_traffic_manager_external_endpoint" "azure" {
  count              = local.is_azure_primary && var.azure_backend.enabled ? 1 : 0
  name               = "azure-endpoint"
  profile_id         = azurerm_traffic_manager_profile.main[0].id
  target             = var.azure_backend.endpoint
  weight             = local.traffic_weights.azure
  priority           = var.failover_policy.azure_priority
  geo_mappings       = var.traffic_distribution.azure_geo_mappings
}

resource "azurerm_traffic_manager_external_endpoint" "gcp" {
  count              = local.is_azure_primary && var.gcp_backend.enabled ? 1 : 0
  name               = "gcp-endpoint"
  profile_id         = azurerm_traffic_manager_profile.main[0].id
  target             = var.gcp_backend.endpoint
  weight             = local.traffic_weights.gcp
  priority           = var.failover_policy.gcp_priority
  geo_mappings       = var.traffic_distribution.gcp_geo_mappings
}

#############################################
# GCP Global Load Balancer (if GCP is primary)
#############################################

# Cloud DNS Managed Zone (GCP Primary)
resource "google_dns_managed_zone" "main" {
  count       = local.is_gcp_primary ? 1 : 0
  name        = replace(var.domain_name, ".", "-")
  dns_name    = "${var.domain_name}."
  description = "DNS zone for TrustformeRS Serve global load balancer"
  project     = var.gcp_project_id

  labels = local.common_tags
}

# Global IP Address (GCP Primary)
resource "google_compute_global_address" "main" {
  count   = local.is_gcp_primary ? 1 : 0
  name    = "${local.lb_name}-ip"
  project = var.gcp_project_id
}

# Backend Services (GCP Primary)
resource "google_compute_backend_service" "main" {
  count                           = local.is_gcp_primary ? 1 : 0
  name                           = local.lb_name
  project                        = var.gcp_project_id
  protocol                       = "HTTP"
  port_name                      = "http"
  load_balancing_scheme          = "EXTERNAL"
  timeout_sec                    = 30
  enable_cdn                     = var.enable_cdn
  
  # Multi-cloud backend configuration
  dynamic "backend" {
    for_each = var.aws_backend.enabled ? [1] : []
    content {
      group           = google_compute_network_endpoint_group.aws[0].self_link
      balancing_mode  = "UTILIZATION"
      capacity_scaler = local.traffic_weights.aws / 100
    }
  }
  
  dynamic "backend" {
    for_each = var.azure_backend.enabled ? [1] : []
    content {
      group           = google_compute_network_endpoint_group.azure[0].self_link
      balancing_mode  = "UTILIZATION"
      capacity_scaler = local.traffic_weights.azure / 100
    }
  }
  
  dynamic "backend" {
    for_each = var.gcp_backend.enabled ? [1] : []
    content {
      group           = google_compute_network_endpoint_group.gcp[0].self_link
      balancing_mode  = "UTILIZATION"
      capacity_scaler = local.traffic_weights.gcp / 100
    }
  }

  health_checks = [google_compute_health_check.main[0].self_link]
}

# Health Check (GCP Primary)
resource "google_compute_health_check" "main" {
  count               = local.is_gcp_primary ? 1 : 0
  name                = "${local.lb_name}-health-check"
  project             = var.gcp_project_id
  timeout_sec         = 5
  check_interval_sec  = 30
  healthy_threshold   = 2
  unhealthy_threshold = 3

  https_health_check {
    port         = 443
    request_path = "/health"
  }
}

# Network Endpoint Groups for external endpoints (GCP Primary)
resource "google_compute_network_endpoint_group" "aws" {
  count                 = local.is_gcp_primary && var.aws_backend.enabled ? 1 : 0
  name                  = "${local.lb_name}-aws-neg"
  project               = var.gcp_project_id
  network_endpoint_type = "INTERNET_FQDN_PORT"
  default_port          = 443
}

resource "google_compute_network_endpoint" "aws" {
  count                  = local.is_gcp_primary && var.aws_backend.enabled ? 1 : 0
  project                = var.gcp_project_id
  network_endpoint_group = google_compute_network_endpoint_group.aws[0].name
  fqdn                   = var.aws_backend.endpoint
  port                   = 443
}

resource "google_compute_network_endpoint_group" "azure" {
  count                 = local.is_gcp_primary && var.azure_backend.enabled ? 1 : 0
  name                  = "${local.lb_name}-azure-neg"
  project               = var.gcp_project_id
  network_endpoint_type = "INTERNET_FQDN_PORT"
  default_port          = 443
}

resource "google_compute_network_endpoint" "azure" {
  count                  = local.is_gcp_primary && var.azure_backend.enabled ? 1 : 0
  project                = var.gcp_project_id
  network_endpoint_group = google_compute_network_endpoint_group.azure[0].name
  fqdn                   = var.azure_backend.endpoint
  port                   = 443
}

resource "google_compute_network_endpoint_group" "gcp" {
  count                 = local.is_gcp_primary && var.gcp_backend.enabled ? 1 : 0
  name                  = "${local.lb_name}-gcp-neg"
  project               = var.gcp_project_id
  network_endpoint_type = "INTERNET_FQDN_PORT"
  default_port          = 443
}

resource "google_compute_network_endpoint" "gcp" {
  count                  = local.is_gcp_primary && var.gcp_backend.enabled ? 1 : 0
  project                = var.gcp_project_id
  network_endpoint_group = google_compute_network_endpoint_group.gcp[0].name
  fqdn                   = var.gcp_backend.endpoint
  port                   = 443
}

# URL Map (GCP Primary)
resource "google_compute_url_map" "main" {
  count           = local.is_gcp_primary ? 1 : 0
  name            = local.lb_name
  project         = var.gcp_project_id
  default_service = google_compute_backend_service.main[0].self_link

  host_rule {
    hosts        = [var.domain_name]
    path_matcher = "allpaths"
  }

  path_matcher {
    name            = "allpaths"
    default_service = google_compute_backend_service.main[0].self_link

    path_rule {
      paths   = ["/*"]
      service = google_compute_backend_service.main[0].self_link
    }
  }
}

# Global Forwarding Rule (GCP Primary)
resource "google_compute_global_forwarding_rule" "main" {
  count      = local.is_gcp_primary ? 1 : 0
  name       = local.lb_name
  project    = var.gcp_project_id
  target     = google_compute_target_https_proxy.main[0].self_link
  ip_address = google_compute_global_address.main[0].address
  port_range = "443"
}

# HTTPS Proxy (GCP Primary)
resource "google_compute_target_https_proxy" "main" {
  count   = local.is_gcp_primary ? 1 : 0
  name    = local.lb_name
  project = var.gcp_project_id
  url_map = google_compute_url_map.main[0].self_link
  ssl_certificates = [google_compute_managed_ssl_certificate.main[0].self_link]
}

# Managed SSL Certificate (GCP Primary)
resource "google_compute_managed_ssl_certificate" "main" {
  count   = local.is_gcp_primary ? 1 : 0
  name    = "${local.lb_name}-ssl-cert"
  project = var.gcp_project_id

  managed {
    domains = [var.domain_name]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# DNS Record (GCP Primary)
resource "google_dns_record_set" "main" {
  count        = local.is_gcp_primary ? 1 : 0
  project      = var.gcp_project_id
  managed_zone = google_dns_managed_zone.main[0].name
  name         = "${var.domain_name}."
  type         = "A"
  ttl          = 300
  rrdatas      = [google_compute_global_address.main[0].address]
}

#############################################
# Cloudflare Integration (Optional)
#############################################

# Cloudflare DNS Records for additional redundancy
resource "cloudflare_record" "main" {
  count   = var.enable_cloudflare_dns ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = var.domain_name
  value   = local.is_aws_primary ? aws_lb.global[0].dns_name : (
    local.is_azure_primary ? azurerm_traffic_manager_profile.main[0].fqdn : 
    google_compute_global_address.main[0].address
  )
  type    = local.is_gcp_primary ? "A" : "CNAME"
  proxied = var.cloudflare_proxied
}

#############################################
# Monitoring and Alerting
#############################################

# CloudWatch Alarms (AWS Primary)
resource "aws_cloudwatch_metric_alarm" "target_response_time" {
  count               = local.is_aws_primary ? 1 : 0
  alarm_name          = "${local.lb_name}-high-response-time"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = "120"
  statistic           = "Average"
  threshold           = "1.0"
  alarm_description   = "This metric monitors ALB target response time"
  alarm_actions       = var.sns_topic_arn != "" ? [var.sns_topic_arn] : []

  dimensions = {
    LoadBalancer = aws_lb.global[0].arn_suffix
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "unhealthy_hosts" {
  count               = local.is_aws_primary ? 1 : 0
  alarm_name          = "${local.lb_name}-unhealthy-hosts"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "UnHealthyHostCount"
  namespace           = "AWS/ApplicationELB"
  period              = "60"
  statistic           = "Average"
  threshold           = "0"
  alarm_description   = "This metric monitors unhealthy ALB targets"
  alarm_actions       = var.sns_topic_arn != "" ? [var.sns_topic_arn] : []

  dimensions = {
    LoadBalancer = aws_lb.global[0].arn_suffix
  }

  tags = local.common_tags
}