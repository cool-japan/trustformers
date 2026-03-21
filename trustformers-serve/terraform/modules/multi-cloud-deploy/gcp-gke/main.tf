# Google Kubernetes Engine (GKE) Module for TrustformeRS Serve
# Provides production-ready GKE cluster with all necessary components

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.4"
    }
  }
}

locals {
  cluster_name = "${var.project_name}-${var.environment}-gke"
  
  common_labels = merge(var.tags, {
    component  = "gke-cluster"
    managed-by = "terraform"
    project    = var.project_name
    environment = var.environment
  })
}

# Data sources
data "google_client_config" "default" {}

data "google_project" "project" {
  project_id = var.project_id
}

# VPC Network (if not provided)
resource "google_compute_network" "main" {
  count                   = var.network_name == "" ? 1 : 0
  name                    = "${local.cluster_name}-network"
  project                 = var.project_id
  auto_create_subnetworks = false
  mtu                     = 1460

  labels = local.common_labels
}

# Use existing or created network
data "google_compute_network" "main" {
  name    = var.network_name != "" ? var.network_name : google_compute_network.main[0].name
  project = var.project_id
}

# Subnet (if not provided)
resource "google_compute_subnetwork" "main" {
  count         = var.subnet_name == "" ? 1 : 0
  name          = "${local.cluster_name}-subnet"
  project       = var.project_id
  region        = var.region
  network       = data.google_compute_network.main.id
  ip_cidr_range = "10.1.0.0/24"

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.2.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.3.0.0/16"
  }

  private_ip_google_access = true
}

# Use existing or created subnet
data "google_compute_subnetwork" "main" {
  name    = var.subnet_name != "" ? var.subnet_name : google_compute_subnetwork.main[0].name
  project = var.project_id
  region  = var.region
}

# Service Account for GKE nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "${local.cluster_name}-nodes"
  display_name = "GKE Node Service Account"
  project      = var.project_id
}

# IAM bindings for GKE node service account
resource "google_project_iam_member" "gke_nodes" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/storage.objectViewer"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# Additional IAM for Workload Identity (if enabled)
resource "google_service_account_iam_member" "workload_identity" {
  count              = var.enable_workload_identity ? 1 : 0
  service_account_id = google_service_account.gke_nodes.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.trustformers_namespace}/trustformers-serve]"
}

# GKE Cluster
resource "google_container_cluster" "main" {
  name     = local.cluster_name
  project  = var.project_id
  location = var.region

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  # Kubernetes version
  min_master_version = var.kubernetes_version

  # Release channel
  dynamic "release_channel" {
    for_each = var.release_channel != "" ? [var.release_channel] : []
    content {
      channel = release_channel.value
    }
  }

  # Network configuration
  network    = data.google_compute_network.main.id
  subnetwork = data.google_compute_subnetwork.main.id

  # IP allocation policy for VPC-native clusters
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Private cluster configuration
  dynamic "private_cluster_config" {
    for_each = var.private_cluster ? [1] : []
    content {
      enable_private_nodes    = true
      enable_private_endpoint = var.private_cluster.enable_private_endpoint
      master_ipv4_cidr_block  = var.master_ipv4_cidr_block
      
      master_global_access_config {
        enabled = var.private_cluster.master_global_access_enabled
      }
    }
  }

  # Master authorized networks
  dynamic "master_authorized_networks_config" {
    for_each = length(var.master_authorized_networks) > 0 ? [1] : []
    content {
      dynamic "cidr_blocks" {
        for_each = var.master_authorized_networks
        content {
          cidr_block   = cidr_blocks.value.cidr_block
          display_name = cidr_blocks.value.display_name
        }
      }
    }
  }

  # Workload Identity
  dynamic "workload_identity_config" {
    for_each = var.enable_workload_identity ? [1] : []
    content {
      workload_pool = "${var.project_id}.svc.id.goog"
    }
  }

  # Network policy
  dynamic "network_policy" {
    for_each = var.enable_network_policy ? [1] : []
    content {
      enabled  = true
      provider = "CALICO"
    }
  }

  # Pod security policy (deprecated)
  dynamic "pod_security_policy_config" {
    for_each = var.enable_pod_security_policy ? [1] : []
    content {
      enabled = true
    }
  }

  # Addons configuration
  addons_config {
    http_load_balancing {
      disabled = !var.enable_http_load_balancing
    }

    horizontal_pod_autoscaling {
      disabled = !var.enable_horizontal_pod_autoscaling
    }

    network_policy_config {
      disabled = !var.enable_network_policy
    }

    dns_cache_config {
      enabled = var.enable_dns_cache
    }

    gcp_filestore_csi_driver_config {
      enabled = var.enable_gcp_filestore_csi
    }

    gcs_fuse_csi_driver_config {
      enabled = var.enable_gcs_fuse_csi
    }

    istio_config {
      disabled = !var.enable_istio
      auth     = var.enable_istio ? "AUTH_MUTUAL_TLS" : null
    }

    cloudrun_config {
      disabled           = !var.enable_cloudrun
      load_balancer_type = var.enable_cloudrun ? "LOAD_BALANCER_TYPE_EXTERNAL" : null
    }
  }

  # Logging and monitoring
  logging_service    = var.enable_google_cloud_logging ? "logging.googleapis.com/kubernetes" : "none"
  monitoring_service = var.enable_google_cloud_monitoring ? "monitoring.googleapis.com/kubernetes" : "none"

  # Cluster-level logging configuration
  dynamic "logging_config" {
    for_each = var.enable_google_cloud_logging ? [1] : []
    content {
      enable_components = [
        "SYSTEM_COMPONENTS",
        "WORKLOADS",
        "API_SERVER"
      ]
    }
  }

  # Cluster-level monitoring configuration
  dynamic "monitoring_config" {
    for_each = var.enable_google_cloud_monitoring ? [1] : []
    content {
      enable_components = [
        "SYSTEM_COMPONENTS",
        "WORKLOADS",
        "APISERVER",
        "SCHEDULER",
        "CONTROLLER_MANAGER"
      ]

      managed_prometheus {
        enabled = var.enable_managed_prometheus
      }
    }
  }

  # Maintenance policy
  dynamic "maintenance_policy" {
    for_each = var.maintenance_window != null ? [1] : []
    content {
      daily_maintenance_window {
        start_time = var.maintenance_window.start_time
      }
    }
  }

  # Resource usage export
  dynamic "resource_usage_export_config" {
    for_each = var.enable_resource_usage_export ? [1] : []
    content {
      enable_network_egress_metering       = true
      enable_resource_consumption_metering = true
      bigquery_destination {
        dataset_id = var.resource_usage_bigquery_dataset
      }
    }
  }

  # Binary authorization
  dynamic "binary_authorization" {
    for_each = var.enable_binary_authorization ? [1] : []
    content {
      evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
    }
  }

  # Database encryption
  dynamic "database_encryption" {
    for_each = var.database_encryption_key != "" ? [1] : []
    content {
      state    = "ENCRYPTED"
      key_name = var.database_encryption_key
    }
  }

  # Shielded nodes
  dynamic "node_config" {
    for_each = var.enable_shielded_nodes ? [1] : []
    content {
      shielded_instance_config {
        enable_secure_boot          = true
        enable_integrity_monitoring = true
      }
    }
  }

  # Cost management
  dynamic "cost_management_config" {
    for_each = var.enable_cost_management ? [1] : []
    content {
      enabled = true
    }
  }

  # Enable Autopilot mode
  dynamic "enable_autopilot" {
    for_each = var.enable_autopilot ? [true] : []
    content {}
  }

  # Notification configuration
  dynamic "notification_config" {
    for_each = var.notification_config != null ? [1] : []
    content {
      pubsub {
        enabled = true
        topic   = var.notification_config.pubsub_topic
      }
    }
  }

  resource_labels = local.common_labels

  # Timeouts
  timeouts {
    create = "30m"
    update = "20m"
    delete = "20m"
  }

  lifecycle {
    ignore_changes = [
      # Ignore changes to node_config as we'll manage nodes separately
      node_config,
      initial_node_count,
    ]
  }
}

# Default Node Pool
resource "google_container_node_pool" "default" {
  count   = var.enable_autopilot ? 0 : 1
  name    = var.default_node_pool.name
  project = var.project_id
  cluster = google_container_cluster.main.name
  location = var.region

  # Specify zones if provided
  node_locations = length(var.zones) > 0 ? var.zones : null

  # Initial node count per zone
  initial_node_count = var.default_node_pool.initial_node_count

  # Auto scaling
  dynamic "autoscaling" {
    for_each = var.default_node_pool.autoscaling_enabled ? [1] : []
    content {
      min_node_count = var.default_node_pool.min_node_count
      max_node_count = var.default_node_pool.max_node_count
    }
  }

  # Node configuration
  node_config {
    preemptible  = var.default_node_pool.preemptible
    machine_type = var.default_node_pool.machine_type
    disk_size_gb = var.default_node_pool.disk_size_gb
    disk_type    = var.default_node_pool.disk_type
    image_type   = var.default_node_pool.image_type

    # Service account
    service_account = google_service_account.gke_nodes.email
    oauth_scopes    = var.default_node_pool.oauth_scopes

    # Labels
    labels = merge(
      local.common_labels,
      var.default_node_pool.node_labels,
      {
        "node-pool" = var.default_node_pool.name
      }
    )

    # Taints
    dynamic "taint" {
      for_each = var.default_node_pool.node_taints
      content {
        key    = taint.value.key
        value  = taint.value.value
        effect = taint.value.effect
      }
    }

    # Workload Identity
    dynamic "workload_metadata_config" {
      for_each = var.enable_workload_identity ? [1] : []
      content {
        mode = "GKE_METADATA"
      }
    }

    # Shielded instance configuration
    dynamic "shielded_instance_config" {
      for_each = var.enable_shielded_nodes ? [1] : []
      content {
        enable_secure_boot          = true
        enable_integrity_monitoring = true
      }
    }

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }

    tags = ["gke-node", "${local.cluster_name}-node"]
  }

  # Node management
  management {
    auto_repair  = var.default_node_pool.auto_repair
    auto_upgrade = var.default_node_pool.auto_upgrade
  }

  # Upgrade settings
  upgrade_settings {
    max_surge       = var.default_node_pool.max_surge
    max_unavailable = var.default_node_pool.max_unavailable
  }

  timeouts {
    create = "30m"
    update = "20m"
    delete = "20m"
  }
}

# Additional Node Pools
resource "google_container_node_pool" "additional" {
  for_each = var.enable_autopilot ? {} : var.additional_node_pools

  name     = each.key
  project  = var.project_id
  cluster  = google_container_cluster.main.name
  location = var.region

  # Specify zones if provided
  node_locations = length(var.zones) > 0 ? var.zones : null

  # Initial node count per zone
  initial_node_count = each.value.initial_node_count

  # Auto scaling
  dynamic "autoscaling" {
    for_each = each.value.autoscaling_enabled ? [1] : []
    content {
      min_node_count = each.value.min_node_count
      max_node_count = each.value.max_node_count
    }
  }

  # Node configuration
  node_config {
    preemptible  = each.value.preemptible
    spot         = each.value.spot
    machine_type = each.value.machine_type
    disk_size_gb = each.value.disk_size_gb
    disk_type    = each.value.disk_type
    image_type   = each.value.image_type

    # Service account
    service_account = google_service_account.gke_nodes.email
    oauth_scopes    = each.value.oauth_scopes

    # Labels
    labels = merge(
      local.common_labels,
      each.value.node_labels,
      {
        "node-pool" = each.key
      }
    )

    # Taints
    dynamic "taint" {
      for_each = each.value.node_taints
      content {
        key    = taint.value.key
        value  = taint.value.value
        effect = taint.value.effect
      }
    }

    # Workload Identity
    dynamic "workload_metadata_config" {
      for_each = var.enable_workload_identity ? [1] : []
      content {
        mode = "GKE_METADATA"
      }
    }

    # Shielded instance configuration
    dynamic "shielded_instance_config" {
      for_each = var.enable_shielded_nodes ? [1] : []
      content {
        enable_secure_boot          = true
        enable_integrity_monitoring = true
      }
    }

    # GPU configuration
    dynamic "guest_accelerator" {
      for_each = each.value.accelerator_count > 0 ? [1] : []
      content {
        type  = each.value.accelerator_type
        count = each.value.accelerator_count
      }
    }

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }

    tags = ["gke-node", "${local.cluster_name}-node", each.key]
  }

  # Node management
  management {
    auto_repair  = each.value.auto_repair
    auto_upgrade = each.value.auto_upgrade
  }

  # Upgrade settings
  upgrade_settings {
    max_surge       = each.value.max_surge
    max_unavailable = each.value.max_unavailable
  }

  timeouts {
    create = "30m"
    update = "20m"
    delete = "20m"
  }
}

# Kubernetes provider configuration
provider "kubernetes" {
  host                   = "https://${google_container_cluster.main.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth.0.cluster_ca_certificate)
}

# Helm provider configuration
provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.main.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth.0.cluster_ca_certificate)
  }
}

# Install NGINX Ingress Controller
resource "helm_release" "nginx_ingress" {
  count = var.enable_nginx_ingress ? 1 : 0

  name       = "ingress-nginx"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  namespace  = "ingress-nginx"
  version    = "4.7.1"

  create_namespace = true

  set {
    name  = "controller.service.type"
    value = "LoadBalancer"
  }

  depends_on = [google_container_cluster.main]
}

# Install cert-manager
resource "helm_release" "cert_manager" {
  count = var.enable_cert_manager ? 1 : 0

  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  namespace  = "cert-manager"
  version    = "v1.11.0"

  create_namespace = true

  set {
    name  = "installCRDs"
    value = "true"
  }

  depends_on = [google_container_cluster.main]
}

# Install TrustformeRS Serve
resource "helm_release" "trustformers_serve" {
  count = var.install_trustformers_serve ? 1 : 0

  name      = "trustformers-serve"
  chart     = var.trustformers_helm_chart_path
  namespace = var.trustformers_namespace
  version   = var.trustformers_helm_chart_version

  create_namespace = true

  values = [
    var.trustformers_helm_values
  ]

  depends_on = [
    google_container_cluster.main,
    helm_release.nginx_ingress,
    helm_release.cert_manager
  ]
}