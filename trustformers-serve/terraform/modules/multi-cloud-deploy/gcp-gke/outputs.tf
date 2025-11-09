# Outputs for GCP GKE module

# Cluster Information
output "cluster_id" {
  description = "GKE cluster ID"
  value       = google_container_cluster.main.id
}

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.main.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = "https://${google_container_cluster.main.endpoint}"
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.main.master_auth.0.cluster_ca_certificate
  sensitive   = true
}

output "cluster_version" {
  description = "GKE cluster Kubernetes version"
  value       = google_container_cluster.main.master_version
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.main.location
}

output "cluster_self_link" {
  description = "GKE cluster self link"
  value       = google_container_cluster.main.self_link
}

# Authentication Information
output "master_auth" {
  description = "GKE cluster master authentication configuration"
  value = {
    cluster_ca_certificate = google_container_cluster.main.master_auth.0.cluster_ca_certificate
  }
  sensitive = true
}

output "access_token" {
  description = "Google Cloud access token for cluster authentication"
  value       = data.google_client_config.default.access_token
  sensitive   = true
}

# Network Information
output "network_name" {
  description = "VPC network name"
  value       = data.google_compute_network.main.name
}

output "network_id" {
  description = "VPC network ID"
  value       = data.google_compute_network.main.id
}

output "network_self_link" {
  description = "VPC network self link"
  value       = data.google_compute_network.main.self_link
}

output "subnet_name" {
  description = "Subnet name"
  value       = data.google_compute_subnetwork.main.name
}

output "subnet_id" {
  description = "Subnet ID"
  value       = data.google_compute_subnetwork.main.id
}

output "subnet_self_link" {
  description = "Subnet self link"
  value       = data.google_compute_subnetwork.main.self_link
}

output "services_ipv4_cidr" {
  description = "Services IPv4 CIDR block"
  value       = google_container_cluster.main.services_ipv4_cidr
}

output "cluster_ipv4_cidr" {
  description = "Cluster IPv4 CIDR block"
  value       = google_container_cluster.main.cluster_ipv4_cidr
}

# Node Pool Information
output "default_node_pool" {
  description = "Default node pool information"
  value = var.enable_autopilot ? null : {
    name               = google_container_node_pool.default[0].name
    node_count         = google_container_node_pool.default[0].node_count
    machine_type       = google_container_node_pool.default[0].node_config[0].machine_type
    disk_size_gb       = google_container_node_pool.default[0].node_config[0].disk_size_gb
    disk_type          = google_container_node_pool.default[0].node_config[0].disk_type
    image_type         = google_container_node_pool.default[0].node_config[0].image_type
    preemptible        = google_container_node_pool.default[0].node_config[0].preemptible
    service_account    = google_container_node_pool.default[0].node_config[0].service_account
    oauth_scopes       = google_container_node_pool.default[0].node_config[0].oauth_scopes
  }
}

output "additional_node_pools" {
  description = "Additional node pools information"
  value = {
    for k, v in google_container_node_pool.additional : k => {
      id               = v.id
      name             = v.name
      node_count       = v.node_count
      machine_type     = v.node_config[0].machine_type
      disk_size_gb     = v.node_config[0].disk_size_gb
      disk_type        = v.node_config[0].disk_type
      image_type       = v.node_config[0].image_type
      preemptible      = v.node_config[0].preemptible
      spot             = v.node_config[0].spot
      service_account  = v.node_config[0].service_account
      oauth_scopes     = v.node_config[0].oauth_scopes
    }
  }
}

output "node_pools_names" {
  description = "List of node pool names"
  value = var.enable_autopilot ? [] : concat(
    [google_container_node_pool.default[0].name],
    [for np in google_container_node_pool.additional : np.name]
  )
}

output "node_pools_versions" {
  description = "Map of node pool names to their Kubernetes versions"
  value = var.enable_autopilot ? {} : merge(
    { (google_container_node_pool.default[0].name) = google_container_node_pool.default[0].version },
    { for k, v in google_container_node_pool.additional : k => v.version }
  )
}

# Service Account Information
output "service_account_email" {
  description = "Service account email for GKE nodes"
  value       = google_service_account.gke_nodes.email
}

output "service_account_name" {
  description = "Service account name for GKE nodes"
  value       = google_service_account.gke_nodes.name
}

output "service_account_unique_id" {
  description = "Service account unique ID for GKE nodes"
  value       = google_service_account.gke_nodes.unique_id
}

# Cluster Features
output "workload_identity_config" {
  description = "Workload Identity configuration"
  value = var.enable_workload_identity ? {
    workload_pool = google_container_cluster.main.workload_identity_config[0].workload_pool
  } : null
}

output "private_cluster_config" {
  description = "Private cluster configuration"
  value = var.private_cluster != null ? {
    enable_private_nodes    = google_container_cluster.main.private_cluster_config[0].enable_private_nodes
    enable_private_endpoint = google_container_cluster.main.private_cluster_config[0].enable_private_endpoint
    master_ipv4_cidr_block  = google_container_cluster.main.private_cluster_config[0].master_ipv4_cidr_block
  } : null
}

output "master_authorized_networks_config" {
  description = "Master authorized networks configuration"
  value = length(var.master_authorized_networks) > 0 ? {
    cidr_blocks = google_container_cluster.main.master_authorized_networks_config[0].cidr_blocks
  } : null
}

# Addons Information
output "addons_config" {
  description = "Cluster addons configuration"
  value = {
    http_load_balancing         = google_container_cluster.main.addons_config[0].http_load_balancing[0].disabled
    horizontal_pod_autoscaling  = google_container_cluster.main.addons_config[0].horizontal_pod_autoscaling[0].disabled
    network_policy_config       = google_container_cluster.main.addons_config[0].network_policy_config[0].disabled
    dns_cache_config           = google_container_cluster.main.addons_config[0].dns_cache_config[0].enabled
    gcp_filestore_csi_driver   = google_container_cluster.main.addons_config[0].gcp_filestore_csi_driver_config[0].enabled
    gcs_fuse_csi_driver        = google_container_cluster.main.addons_config[0].gcs_fuse_csi_driver_config[0].enabled
  }
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

# Logging and Monitoring
output "logging_service" {
  description = "Logging service configuration"
  value       = google_container_cluster.main.logging_service
}

output "monitoring_service" {
  description = "Monitoring service configuration"
  value       = google_container_cluster.main.monitoring_service
}

output "logging_config" {
  description = "Logging configuration"
  value = var.enable_google_cloud_logging ? {
    enable_components = google_container_cluster.main.logging_config[0].enable_components
  } : null
}

output "monitoring_config" {
  description = "Monitoring configuration"
  value = var.enable_google_cloud_monitoring ? {
    enable_components = google_container_cluster.main.monitoring_config[0].enable_components
    managed_prometheus = {
      enabled = google_container_cluster.main.monitoring_config[0].managed_prometheus[0].enabled
    }
  } : null
}

# Cluster Status
output "cluster_status" {
  description = "Current status of the cluster"
  value       = google_container_cluster.main.status
}

output "conditions" {
  description = "Cluster conditions"
  value       = google_container_cluster.main.conditions
}

# Cost Information
output "estimated_costs" {
  description = "Estimated monthly costs for the GKE cluster"
  value = {
    cluster_cost = var.enable_autopilot ? "Autopilot pricing based on resource usage" : "Standard cluster management fee: $0.10/hour per cluster"
    node_pools = var.enable_autopilot ? {} : {
      for k, v in merge(
        { default = var.default_node_pool },
        var.additional_node_pools
      ) : k => {
        machine_type   = k == "default" ? v.machine_type : v.machine_type
        node_count     = k == "default" ? (v.autoscaling_enabled ? "${v.min_node_count}-${v.max_node_count}" : tostring(v.initial_node_count)) : (v.autoscaling_enabled ? "${v.min_node_count}-${v.max_node_count}" : tostring(v.initial_node_count))
        preemptible    = k == "default" ? v.preemptible : v.preemptible
        spot           = k == "default" ? false : v.spot
        estimated_cost = "Varies based on machine type and usage"
      }
    }
    note = "Actual costs depend on machine types, usage patterns, and additional GCP services"
  }
}

# Cluster Labels
output "cluster_labels" {
  description = "Labels applied to the GKE cluster"
  value       = google_container_cluster.main.resource_labels
}

# Resource Information for Monitoring
output "monitoring_targets" {
  description = "Resources that should be monitored"
  value = {
    cluster_name         = google_container_cluster.main.name
    cluster_id           = google_container_cluster.main.id
    project_id           = var.project_id
    location             = google_container_cluster.main.location
    node_pools = var.enable_autopilot ? [] : concat(
      [google_container_node_pool.default[0].name],
      [for np in google_container_node_pool.additional : np.name]
    )
  }
}

# Connection Information for Applications
output "connection_info" {
  description = "Connection information for applications"
  value = {
    # Internal cluster endpoint for in-cluster communication
    internal_endpoint = google_container_cluster.main.endpoint
    
    # External access information
    public_endpoint_enabled = !var.private_cluster.enable_private_endpoint
    
    # DNS information
    cluster_dns_name = replace(google_container_cluster.main.endpoint, "https://", "")
    
    # Service discovery
    cluster_domain = "cluster.local"
    
    # Autopilot information
    is_autopilot_cluster = var.enable_autopilot
  }
}

# Security Information
output "database_encryption" {
  description = "Database encryption configuration"
  value = var.database_encryption_key != "" ? {
    state    = google_container_cluster.main.database_encryption[0].state
    key_name = google_container_cluster.main.database_encryption[0].key_name
  } : null
}

output "network_policy" {
  description = "Network policy configuration"
  value = var.enable_network_policy ? {
    enabled  = google_container_cluster.main.network_policy[0].enabled
    provider = google_container_cluster.main.network_policy[0].provider
  } : null
}

output "binary_authorization" {
  description = "Binary authorization configuration"
  value = var.enable_binary_authorization ? {
    evaluation_mode = google_container_cluster.main.binary_authorization[0].evaluation_mode
  } : null
}

# Maintenance Information
output "maintenance_policy" {
  description = "Cluster maintenance policy"
  value = var.maintenance_window != null ? {
    daily_maintenance_window = {
      start_time = google_container_cluster.main.maintenance_policy[0].daily_maintenance_window[0].start_time
      duration   = google_container_cluster.main.maintenance_policy[0].daily_maintenance_window[0].duration
    }
  } : null
}

# Regional/Zonal Information
output "node_locations" {
  description = "Node locations (zones)"
  value       = google_container_cluster.main.node_locations
}

output "default_max_pods_per_node" {
  description = "Default maximum pods per node"
  value       = google_container_cluster.main.default_max_pods_per_node
}

# Autopilot Information
output "enable_autopilot" {
  description = "Whether Autopilot is enabled"
  value       = var.enable_autopilot
}

# TPU Information (if applicable)
output "tpu_ipv4_cidr_block" {
  description = "TPU IPv4 CIDR block"
  value       = google_container_cluster.main.tpu_ipv4_cidr_block
}

# Notification Configuration
output "notification_config" {
  description = "Notification configuration"
  value = var.notification_config != null ? {
    pubsub = {
      enabled = google_container_cluster.main.notification_config[0].pubsub[0].enabled
      topic   = google_container_cluster.main.notification_config[0].pubsub[0].topic
    }
  } : null
}