# Outputs for Global Load Balancer module

# General Load Balancer Information
output "load_balancer_endpoint" {
  description = "Primary endpoint for the global load balancer"
  value = var.primary_cloud == "aws" ? (
    length(aws_lb.global) > 0 ? aws_lb.global[0].dns_name : ""
  ) : var.primary_cloud == "azure" ? (
    length(azurerm_traffic_manager_profile.main) > 0 ? azurerm_traffic_manager_profile.main[0].fqdn : ""
  ) : var.primary_cloud == "gcp" ? (
    length(google_compute_global_address.main) > 0 ? google_compute_global_address.main[0].address : ""
  ) : ""
}

output "domain_name" {
  description = "Domain name configured for the load balancer"
  value       = var.domain_name
}

output "primary_cloud" {
  description = "Primary cloud provider used for the global load balancer"
  value       = var.primary_cloud
}

# DNS Information
output "dns_zone_id" {
  description = "DNS zone ID"
  value = var.primary_cloud == "aws" ? (
    length(aws_route53_zone.main) > 0 ? aws_route53_zone.main[0].zone_id : ""
  ) : var.primary_cloud == "azure" ? (
    length(azurerm_dns_zone.main) > 0 ? azurerm_dns_zone.main[0].id : ""
  ) : var.primary_cloud == "gcp" ? (
    length(google_dns_managed_zone.main) > 0 ? google_dns_managed_zone.main[0].name : ""
  ) : ""
}

output "dns_zone_name_servers" {
  description = "Name servers for the DNS zone"
  value = var.primary_cloud == "aws" ? (
    length(aws_route53_zone.main) > 0 ? aws_route53_zone.main[0].name_servers : []
  ) : var.primary_cloud == "azure" ? (
    length(azurerm_dns_zone.main) > 0 ? azurerm_dns_zone.main[0].name_servers : []
  ) : var.primary_cloud == "gcp" ? (
    length(google_dns_managed_zone.main) > 0 ? google_dns_managed_zone.main[0].name_servers : []
  ) : []
}

# SSL Certificate Information
output "ssl_certificate_arn" {
  description = "SSL certificate ARN/ID"
  value = var.primary_cloud == "aws" ? (
    length(aws_acm_certificate.main) > 0 ? aws_acm_certificate.main[0].arn : ""
  ) : var.primary_cloud == "gcp" ? (
    length(google_compute_managed_ssl_certificate.main) > 0 ? google_compute_managed_ssl_certificate.main[0].id : ""
  ) : ""
}

output "ssl_certificate_status" {
  description = "SSL certificate status"
  value = var.primary_cloud == "aws" ? (
    length(aws_acm_certificate.main) > 0 ? aws_acm_certificate.main[0].status : ""
  ) : var.primary_cloud == "gcp" ? (
    length(google_compute_managed_ssl_certificate.main) > 0 ? google_compute_managed_ssl_certificate.main[0].managed[0].status : ""
  ) : ""
}

# AWS-specific Outputs
output "aws_load_balancer_arn" {
  description = "AWS Application Load Balancer ARN"
  value       = length(aws_lb.global) > 0 ? aws_lb.global[0].arn : null
}

output "aws_load_balancer_dns_name" {
  description = "AWS Application Load Balancer DNS name"
  value       = length(aws_lb.global) > 0 ? aws_lb.global[0].dns_name : null
}

output "aws_load_balancer_zone_id" {
  description = "AWS Application Load Balancer canonical hosted zone ID"
  value       = length(aws_lb.global) > 0 ? aws_lb.global[0].zone_id : null
}

output "aws_target_group_arns" {
  description = "AWS target group ARNs"
  value = {
    aws_targets   = length(aws_lb_target_group.aws) > 0 ? aws_lb_target_group.aws[0].arn : null
    azure_targets = length(aws_lb_target_group.azure) > 0 ? aws_lb_target_group.azure[0].arn : null
    gcp_targets   = length(aws_lb_target_group.gcp) > 0 ? aws_lb_target_group.gcp[0].arn : null
  }
}

output "aws_listener_arns" {
  description = "AWS listener ARNs"
  value = {
    http_listener  = length(aws_lb_listener.http) > 0 ? aws_lb_listener.http[0].arn : null
    https_listener = length(aws_lb_listener.https) > 0 ? aws_lb_listener.https[0].arn : null
  }
}

output "aws_security_group_id" {
  description = "AWS security group ID for the load balancer"
  value       = length(aws_security_group.alb) > 0 ? aws_security_group.alb[0].id : null
}

output "aws_route53_zone_id" {
  description = "AWS Route 53 hosted zone ID"
  value       = length(aws_route53_zone.main) > 0 ? aws_route53_zone.main[0].zone_id : null
}

output "aws_health_check_ids" {
  description = "AWS Route 53 health check IDs"
  value = {
    aws_health_check = length(aws_route53_health_check.aws) > 0 ? aws_route53_health_check.aws[0].id : null
  }
}

# Azure-specific Outputs
output "azure_traffic_manager_fqdn" {
  description = "Azure Traffic Manager FQDN"
  value       = length(azurerm_traffic_manager_profile.main) > 0 ? azurerm_traffic_manager_profile.main[0].fqdn : null
}

output "azure_traffic_manager_id" {
  description = "Azure Traffic Manager profile ID"
  value       = length(azurerm_traffic_manager_profile.main) > 0 ? azurerm_traffic_manager_profile.main[0].id : null
}

output "azure_dns_zone_id" {
  description = "Azure DNS zone ID"
  value       = length(azurerm_dns_zone.main) > 0 ? azurerm_dns_zone.main[0].id : null
}

output "azure_endpoint_ids" {
  description = "Azure Traffic Manager endpoint IDs"
  value = {
    aws_endpoint   = length(azurerm_traffic_manager_external_endpoint.aws) > 0 ? azurerm_traffic_manager_external_endpoint.aws[0].id : null
    azure_endpoint = length(azurerm_traffic_manager_external_endpoint.azure) > 0 ? azurerm_traffic_manager_external_endpoint.azure[0].id : null
    gcp_endpoint   = length(azurerm_traffic_manager_external_endpoint.gcp) > 0 ? azurerm_traffic_manager_external_endpoint.gcp[0].id : null
  }
}

# GCP-specific Outputs
output "gcp_global_ip_address" {
  description = "GCP global IP address"
  value       = length(google_compute_global_address.main) > 0 ? google_compute_global_address.main[0].address : null
}

output "gcp_global_ip_name" {
  description = "GCP global IP address name"
  value       = length(google_compute_global_address.main) > 0 ? google_compute_global_address.main[0].name : null
}

output "gcp_backend_service_id" {
  description = "GCP backend service ID"
  value       = length(google_compute_backend_service.main) > 0 ? google_compute_backend_service.main[0].id : null
}

output "gcp_url_map_id" {
  description = "GCP URL map ID"
  value       = length(google_compute_url_map.main) > 0 ? google_compute_url_map.main[0].id : null
}

output "gcp_forwarding_rule_id" {
  description = "GCP global forwarding rule ID"
  value       = length(google_compute_global_forwarding_rule.main) > 0 ? google_compute_global_forwarding_rule.main[0].id : null
}

output "gcp_https_proxy_id" {
  description = "GCP HTTPS proxy ID"
  value       = length(google_compute_target_https_proxy.main) > 0 ? google_compute_target_https_proxy.main[0].id : null
}

output "gcp_dns_zone_name" {
  description = "GCP DNS managed zone name"
  value       = length(google_dns_managed_zone.main) > 0 ? google_dns_managed_zone.main[0].name : null
}

output "gcp_health_check_id" {
  description = "GCP health check ID"
  value       = length(google_compute_health_check.main) > 0 ? google_compute_health_check.main[0].id : null
}

output "gcp_network_endpoint_groups" {
  description = "GCP network endpoint group IDs"
  value = {
    aws_neg   = length(google_compute_network_endpoint_group.aws) > 0 ? google_compute_network_endpoint_group.aws[0].id : null
    azure_neg = length(google_compute_network_endpoint_group.azure) > 0 ? google_compute_network_endpoint_group.azure[0].id : null
    gcp_neg   = length(google_compute_network_endpoint_group.gcp) > 0 ? google_compute_network_endpoint_group.gcp[0].id : null
  }
}

# Cloudflare Outputs
output "cloudflare_record_id" {
  description = "Cloudflare DNS record ID"
  value       = length(cloudflare_record.main) > 0 ? cloudflare_record.main[0].id : null
}

output "cloudflare_record_hostname" {
  description = "Cloudflare DNS record hostname"
  value       = length(cloudflare_record.main) > 0 ? cloudflare_record.main[0].hostname : null
}

# Backend Configuration Summary
output "backend_summary" {
  description = "Summary of configured backends"
  value = {
    aws_backend = {
      enabled  = var.aws_backend.enabled
      endpoint = var.aws_backend.endpoint
      region   = var.aws_backend.region
      weight   = local.traffic_weights.aws
    }
    azure_backend = {
      enabled  = var.azure_backend.enabled
      endpoint = var.azure_backend.endpoint
      region   = var.azure_backend.region
      weight   = local.traffic_weights.azure
    }
    gcp_backend = {
      enabled  = var.gcp_backend.enabled
      endpoint = var.gcp_backend.endpoint
      region   = var.gcp_backend.region
      weight   = local.traffic_weights.gcp
    }
  }
}

# Traffic Distribution Configuration
output "traffic_distribution_config" {
  description = "Traffic distribution configuration"
  value = {
    method         = var.traffic_distribution.method
    total_backends = length([for k, v in local.active_backends : k if v])
    weights = {
      aws   = local.traffic_weights.aws
      azure = local.traffic_weights.azure
      gcp   = local.traffic_weights.gcp
    }
  }
}

# Health Check Configuration
output "health_check_config" {
  description = "Health check configuration summary"
  value = {
    interval_seconds     = var.health_check_config.interval_seconds
    timeout_seconds     = var.health_check_config.timeout_seconds
    healthy_threshold   = var.health_check_config.healthy_threshold
    unhealthy_threshold = var.health_check_config.unhealthy_threshold
    path                = var.health_check_config.path
    protocol            = var.health_check_config.protocol
  }
}

# Monitoring Configuration
output "monitoring_config" {
  description = "Monitoring configuration summary"
  value = {
    enabled                   = var.enable_monitoring
    response_time_threshold   = var.monitoring_config.response_time_threshold
    error_rate_threshold     = var.monitoring_config.error_rate_threshold
    detailed_metrics_enabled = var.monitoring_config.enable_detailed_metrics
  }
}

# Security Configuration
output "security_config" {
  description = "Security configuration summary"
  value = {
    waf_enabled            = var.enable_waf
    ddos_protection        = var.enable_ddos_protection
    ssl_policy            = var.ssl_policy
    access_logging_enabled = var.enable_access_logging
  }
}

# CloudWatch Alarm ARNs (AWS Primary)
output "cloudwatch_alarm_arns" {
  description = "CloudWatch alarm ARNs"
  value = var.primary_cloud == "aws" ? {
    response_time_alarm = length(aws_cloudwatch_metric_alarm.target_response_time) > 0 ? aws_cloudwatch_metric_alarm.target_response_time[0].arn : null
    unhealthy_hosts_alarm = length(aws_cloudwatch_metric_alarm.unhealthy_hosts) > 0 ? aws_cloudwatch_metric_alarm.unhealthy_hosts[0].arn : null
  } : null
}

# Connection Information
output "connection_info" {
  description = "Connection information for applications"
  value = {
    # Primary endpoint
    primary_endpoint = var.domain_name
    
    # Load balancer endpoint (cloud-specific)
    load_balancer_endpoint = var.primary_cloud == "aws" ? (
      length(aws_lb.global) > 0 ? aws_lb.global[0].dns_name : ""
    ) : var.primary_cloud == "azure" ? (
      length(azurerm_traffic_manager_profile.main) > 0 ? azurerm_traffic_manager_profile.main[0].fqdn : ""
    ) : var.primary_cloud == "gcp" ? (
      length(google_compute_global_address.main) > 0 ? google_compute_global_address.main[0].address : ""
    ) : ""
    
    # Protocol information
    protocols_supported = ["HTTP", "HTTPS"]
    default_protocol   = "HTTPS"
    ports = {
      http  = 80
      https = 443
    }
    
    # Regional information
    primary_cloud_region = var.primary_cloud == "aws" ? var.aws_backend.region : (
      var.primary_cloud == "azure" ? var.azure_backend.region : var.gcp_backend.region
    )
  }
}

# Cost Information
output "estimated_costs" {
  description = "Estimated monthly costs for the global load balancer"
  value = {
    primary_cloud_cost = var.primary_cloud == "aws" ? "ALB: ~$22.50/month + data processing" : (
      var.primary_cloud == "azure" ? "Traffic Manager: ~$0.54/month + DNS queries" : 
      "GCP Global LB: ~$18/month + data processing"
    )
    dns_costs = var.primary_cloud == "aws" ? "Route 53: $0.50/hosted zone + $0.40/million queries" : (
      var.primary_cloud == "azure" ? "Azure DNS: $0.50/zone + $0.40/million queries" : 
      "Cloud DNS: $0.20/zone + $0.40/million queries"
    )
    ssl_certificate_cost = var.primary_cloud == "aws" ? "ACM: Free for ALB" : (
      var.primary_cloud == "azure" ? "Depends on certificate source" : 
      "GCP Managed SSL: Free"
    )
    cloudflare_cost = var.enable_cloudflare_dns ? "Depends on Cloudflare plan" : "Not applicable"
    note = "Actual costs depend on traffic volume, data transfer, and additional features enabled"
  }
}

# Load Balancer Status
output "load_balancer_status" {
  description = "Current status of the load balancer"
  value = {
    primary_cloud     = var.primary_cloud
    domain_configured = var.domain_name != ""
    ssl_enabled      = true
    backends_active = {
      aws   = var.aws_backend.enabled
      azure = var.azure_backend.enabled
      gcp   = var.gcp_backend.enabled
    }
    total_active_backends = length([for k, v in local.active_backends : k if v])
    cloudflare_enabled   = var.enable_cloudflare_dns
  }
}