# Outputs for Azure AKS module

# Cluster Information
output "cluster_id" {
  description = "AKS cluster ID"
  value       = azurerm_kubernetes_cluster.main.id
}

output "cluster_name" {
  description = "AKS cluster name"
  value       = azurerm_kubernetes_cluster.main.name
}

output "cluster_endpoint" {
  description = "AKS cluster endpoint"
  value       = azurerm_kubernetes_cluster.main.kube_config.0.host
}

output "cluster_version" {
  description = "AKS cluster Kubernetes version"
  value       = azurerm_kubernetes_cluster.main.kubernetes_version
}

output "cluster_fqdn" {
  description = "AKS cluster FQDN"
  value       = azurerm_kubernetes_cluster.main.fqdn
}

output "cluster_private_fqdn" {
  description = "AKS cluster private FQDN"
  value       = azurerm_kubernetes_cluster.main.private_fqdn
}

# Identity Information
output "cluster_identity" {
  description = "AKS cluster managed identity"
  value = {
    principal_id = azurerm_kubernetes_cluster.main.identity[0].principal_id
    tenant_id    = azurerm_kubernetes_cluster.main.identity[0].tenant_id
    type         = azurerm_kubernetes_cluster.main.identity[0].type
  }
}

output "kubelet_identity" {
  description = "AKS cluster kubelet identity"
  value = {
    client_id                 = azurerm_kubernetes_cluster.main.kubelet_identity[0].client_id
    object_id                 = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
    user_assigned_identity_id = azurerm_kubernetes_cluster.main.kubelet_identity[0].user_assigned_identity_id
  }
}

output "user_assigned_identity_id" {
  description = "User assigned identity ID"
  value       = azurerm_user_assigned_identity.aks.id
}

output "user_assigned_identity_principal_id" {
  description = "User assigned identity principal ID"
  value       = azurerm_user_assigned_identity.aks.principal_id
}

output "user_assigned_identity_client_id" {
  description = "User assigned identity client ID"
  value       = azurerm_user_assigned_identity.aks.client_id
}

# Network Information
output "node_resource_group" {
  description = "AKS node resource group name"
  value       = azurerm_kubernetes_cluster.main.node_resource_group
}

output "network_profile" {
  description = "AKS cluster network profile"
  value = {
    network_plugin     = azurerm_kubernetes_cluster.main.network_profile[0].network_plugin
    network_policy     = azurerm_kubernetes_cluster.main.network_profile[0].network_policy
    dns_service_ip     = azurerm_kubernetes_cluster.main.network_profile[0].dns_service_ip
    docker_bridge_cidr = azurerm_kubernetes_cluster.main.network_profile[0].docker_bridge_cidr
    service_cidr       = azurerm_kubernetes_cluster.main.network_profile[0].service_cidr
  }
}

output "virtual_network_id" {
  description = "Virtual network ID"
  value       = data.azurerm_virtual_network.main.id
}

output "subnet_id" {
  description = "Subnet ID used by AKS"
  value       = data.azurerm_subnet.aks.id
}

# Node Pool Information
output "default_node_pool" {
  description = "Default node pool information"
  value = {
    name            = azurerm_kubernetes_cluster.main.default_node_pool[0].name
    node_count      = azurerm_kubernetes_cluster.main.default_node_pool[0].node_count
    vm_size         = azurerm_kubernetes_cluster.main.default_node_pool[0].vm_size
    os_disk_size_gb = azurerm_kubernetes_cluster.main.default_node_pool[0].os_disk_size_gb
    max_pods        = azurerm_kubernetes_cluster.main.default_node_pool[0].max_pods
  }
}

output "additional_node_pools" {
  description = "Additional node pools information"
  value = {
    for k, v in azurerm_kubernetes_cluster_node_pool.additional : k => {
      id              = v.id
      name            = v.name
      node_count      = v.node_count
      vm_size         = v.vm_size
      os_disk_size_gb = v.os_disk_size_gb
      max_pods        = v.max_pods
      availability_zones = v.zones
    }
  }
}

# Resource Group Information
output "resource_group_name" {
  description = "Resource group name"
  value       = data.azurerm_resource_group.main.name
}

output "resource_group_location" {
  description = "Resource group location"
  value       = data.azurerm_resource_group.main.location
}

# Monitoring Information
output "log_analytics_workspace_id" {
  description = "Log Analytics Workspace ID"
  value       = var.enable_oms_agent ? azurerm_log_analytics_workspace.aks[0].id : null
}

output "log_analytics_workspace_primary_shared_key" {
  description = "Log Analytics Workspace primary shared key"
  value       = var.enable_oms_agent ? azurerm_log_analytics_workspace.aks[0].primary_shared_key : null
  sensitive   = true
}

# Kubernetes Configuration
output "kube_config" {
  description = "Kubernetes configuration"
  value = {
    host                   = azurerm_kubernetes_cluster.main.kube_config.0.host
    client_certificate     = azurerm_kubernetes_cluster.main.kube_config.0.client_certificate
    client_key             = azurerm_kubernetes_cluster.main.kube_config.0.client_key
    cluster_ca_certificate = azurerm_kubernetes_cluster.main.kube_config.0.cluster_ca_certificate
  }
  sensitive = true
}

output "kube_config_raw" {
  description = "Raw Kubernetes configuration"
  value       = azurerm_kubernetes_cluster.main.kube_config_raw
  sensitive   = true
}

# Helm Release Information
output "helm_releases" {
  description = "Information about installed Helm releases"
  value = {
    cert_manager = var.enable_cert_manager ? {
      name      = helm_release.cert_manager[0].name
      namespace = helm_release.cert_manager[0].namespace
      version   = helm_release.cert_manager[0].version
      status    = helm_release.cert_manager[0].status
    } : null
    
    nginx_ingress = var.enable_nginx_ingress ? {
      name      = helm_release.nginx_ingress[0].name
      namespace = helm_release.nginx_ingress[0].namespace
      version   = helm_release.nginx_ingress[0].version
      status    = helm_release.nginx_ingress[0].status
    } : null
    
    trustformers_serve = var.install_trustformers_serve ? {
      name      = helm_release.trustformers_serve[0].name
      namespace = helm_release.trustformers_serve[0].namespace
      version   = helm_release.trustformers_serve[0].version
      status    = helm_release.trustformers_serve[0].status
    } : null
  }
}

# DNS Information
output "dns_prefix" {
  description = "DNS prefix for the cluster"
  value       = azurerm_kubernetes_cluster.main.dns_prefix
}

# API Server Information
output "api_server_access_profile" {
  description = "API server access profile"
  value = {
    authorized_ip_ranges = azurerm_kubernetes_cluster.main.api_server_access_profile[0].authorized_ip_ranges
  }
}

# Cost Information
output "estimated_costs" {
  description = "Estimated monthly costs for the AKS cluster"
  value = {
    cluster_cost = var.sku_tier == "Standard" ? "Approximately $73/month for Standard tier control plane" : "Free tier control plane"
    node_pools = {
      for k, v in merge(
        { default = var.default_node_pool },
        var.additional_node_pools
      ) : k => {
        vm_size        = k == "default" ? v.vm_size : v.vm_size
        node_count     = k == "default" ? v.node_count : v.node_count
        estimated_cost = "Varies based on VM size and usage"
      }
    }
    note = "Actual costs depend on VM sizes, usage patterns, and additional Azure services"
  }
}

# Cluster Tags
output "cluster_tags" {
  description = "Tags applied to the AKS cluster"
  value       = azurerm_kubernetes_cluster.main.tags
}

# Resource Information for Monitoring
output "monitoring_targets" {
  description = "Resources that should be monitored"
  value = {
    cluster_name           = azurerm_kubernetes_cluster.main.name
    cluster_id             = azurerm_kubernetes_cluster.main.id
    resource_group_name    = data.azurerm_resource_group.main.name
    node_resource_group    = azurerm_kubernetes_cluster.main.node_resource_group
    log_analytics_workspace_id = var.enable_oms_agent ? azurerm_log_analytics_workspace.aks[0].id : null
  }
}

# Connection Information for Applications
output "connection_info" {
  description = "Connection information for applications"
  value = {
    # Internal cluster endpoint for in-cluster communication
    internal_endpoint = azurerm_kubernetes_cluster.main.kube_config.0.host
    
    # DNS information
    cluster_fqdn        = azurerm_kubernetes_cluster.main.fqdn
    private_cluster_fqdn = azurerm_kubernetes_cluster.main.private_fqdn
    
    # Service discovery
    cluster_domain = "cluster.local"
    
    # Private cluster information
    is_private_cluster = azurerm_kubernetes_cluster.main.private_cluster_enabled
  }
}

# Azure-specific Outputs
output "azure_policy_enabled" {
  description = "Whether Azure Policy is enabled"
  value       = try(azurerm_kubernetes_cluster.main.azure_policy_enabled[0], false)
}

output "oms_agent_enabled" {
  description = "Whether OMS Agent is enabled"
  value       = var.enable_oms_agent
}

output "ingress_application_gateway" {
  description = "Ingress Application Gateway information"
  value = var.enable_ingress_application_gateway ? {
    enabled      = true
    gateway_name = "${local.cluster_name}-agw"
    subnet_cidr  = "10.1.2.0/24"
  } : null
}

# OIDC Issuer (for Workload Identity)
output "oidc_issuer_url" {
  description = "OIDC issuer URL for the cluster"
  value       = azurerm_kubernetes_cluster.main.oidc_issuer_url
}

# Portal FQDN
output "portal_fqdn" {
  description = "Portal FQDN for the cluster"
  value       = azurerm_kubernetes_cluster.main.portal_fqdn
}

# HTTP Application Routing (if enabled)
output "http_application_routing_zone_name" {
  description = "HTTP application routing zone name"
  value       = try(azurerm_kubernetes_cluster.main.http_application_routing_zone_name, null)
}