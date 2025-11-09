# Azure Kubernetes Service (AKS) Module for TrustformeRS Serve
# Provides production-ready AKS cluster with all necessary components

terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
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

# Configure the Microsoft Azure Provider
provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
  }
}

locals {
  cluster_name = "${var.project_name}-${var.environment}-aks"
  
  common_tags = merge(var.tags, {
    Component = "aks-cluster"
    ManagedBy = "terraform"
  })
}

# Data sources
data "azurerm_client_config" "current" {}

# Resource Group (if not provided)
resource "azurerm_resource_group" "main" {
  count    = var.resource_group_name == "" ? 1 : 0
  name     = "${local.cluster_name}-rg"
  location = var.location

  tags = local.common_tags
}

# Use existing or created resource group
data "azurerm_resource_group" "main" {
  name = var.resource_group_name != "" ? var.resource_group_name : azurerm_resource_group.main[0].name
}

# Virtual Network (if not provided)
resource "azurerm_virtual_network" "main" {
  count               = var.vnet_name == "" ? 1 : 0
  name                = "${local.cluster_name}-vnet"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  address_space       = ["10.1.0.0/16"]

  tags = local.common_tags
}

# Use existing or created VNet
data "azurerm_virtual_network" "main" {
  name                = var.vnet_name != "" ? var.vnet_name : azurerm_virtual_network.main[0].name
  resource_group_name = data.azurerm_resource_group.main.name
}

# Subnet for AKS (if not provided)
resource "azurerm_subnet" "aks" {
  count                = var.subnet_name == "" ? 1 : 0
  name                 = "${local.cluster_name}-subnet"
  resource_group_name  = data.azurerm_resource_group.main.name
  virtual_network_name = data.azurerm_virtual_network.main.name
  address_prefixes     = ["10.1.1.0/24"]
}

# Use existing or created subnet
data "azurerm_subnet" "aks" {
  name                 = var.subnet_name != "" ? var.subnet_name : azurerm_subnet.aks[0].name
  virtual_network_name = data.azurerm_virtual_network.main.name
  resource_group_name  = data.azurerm_resource_group.main.name
}

# User Assigned Identity for AKS cluster
resource "azurerm_user_assigned_identity" "aks" {
  location            = data.azurerm_resource_group.main.location
  name                = "${local.cluster_name}-identity"
  resource_group_name = data.azurerm_resource_group.main.name

  tags = local.common_tags
}

# Role assignments for AKS identity
resource "azurerm_role_assignment" "aks_network_contributor" {
  scope                = data.azurerm_virtual_network.main.id
  role_definition_name = "Network Contributor"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

resource "azurerm_role_assignment" "aks_managed_identity_operator" {
  scope                = azurerm_user_assigned_identity.aks.id
  role_definition_name = "Managed Identity Operator"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

# Log Analytics Workspace for monitoring
resource "azurerm_log_analytics_workspace" "aks" {
  count               = var.enable_oms_agent ? 1 : 0
  name                = "${local.cluster_name}-logs"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = local.common_tags
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = local.cluster_name
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  dns_prefix          = "${var.project_name}${var.environment}aks"
  kubernetes_version  = var.kubernetes_version
  sku_tier           = var.sku_tier

  # Node pool configuration
  default_node_pool {
    name                = var.default_node_pool.name
    node_count         = var.default_node_pool.node_count
    vm_size            = var.default_node_pool.vm_size
    vnet_subnet_id     = data.azurerm_subnet.aks.id
    enable_auto_scaling = var.enable_auto_scaling
    min_count          = var.enable_auto_scaling ? var.default_node_pool.min_count : null
    max_count          = var.enable_auto_scaling ? var.default_node_pool.max_count : null
    max_pods           = var.default_node_pool.max_pods
    os_disk_size_gb    = var.default_node_pool.os_disk_size_gb
    os_disk_type       = var.default_node_pool.os_disk_type
    type               = "VirtualMachineScaleSets"
    
    node_labels = var.default_node_pool.node_labels
    tags        = merge(local.common_tags, var.default_node_pool.tags)

    upgrade_settings {
      max_surge = "10%"
    }
  }

  # Identity configuration
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.aks.id]
  }

  # Network configuration
  network_profile {
    network_plugin     = "azure"
    network_policy     = var.enable_network_policy ? "azure" : null
    dns_service_ip     = "10.2.0.10"
    docker_bridge_cidr = "172.17.0.1/16"
    service_cidr       = "10.2.0.0/24"
  }

  # API server access profile
  api_server_access_profile {
    authorized_ip_ranges = var.private_cluster_enabled ? [] : var.api_server_authorized_ip_ranges
  }

  private_cluster_enabled = var.private_cluster_enabled

  # Azure AD integration
  dynamic "azure_active_directory_role_based_access_control" {
    for_each = var.enable_rbac ? [1] : []
    content {
      managed                = true
      azure_rbac_enabled     = true
      admin_group_object_ids = var.admin_group_object_ids
    }
  }

  # OMS Agent (Azure Monitor)
  dynamic "oms_agent" {
    for_each = var.enable_oms_agent ? [1] : []
    content {
      log_analytics_workspace_id = azurerm_log_analytics_workspace.aks[0].id
    }
  }

  # Ingress Application Gateway
  dynamic "ingress_application_gateway" {
    for_each = var.enable_ingress_application_gateway ? [1] : []
    content {
      gateway_name = "${local.cluster_name}-agw"
      subnet_cidr  = "10.1.2.0/24"
    }
  }

  # Azure Policy
  dynamic "azure_policy_enabled" {
    for_each = var.enable_azure_policy ? [true] : []
    content {
      enabled = true
    }
  }

  # Key Vault integration
  dynamic "key_vault_secrets_provider" {
    for_each = var.enable_key_vault_secrets_provider ? [1] : []
    content {
      secret_rotation_enabled  = true
      secret_rotation_interval = "2m"
    }
  }

  # Auto-scaler profile
  dynamic "auto_scaler_profile" {
    for_each = var.enable_auto_scaling ? [1] : []
    content {
      balance_similar_node_groups      = false
      expander                         = "random"
      max_graceful_termination_sec     = "600"
      max_node_provisioning_time       = "15m"
      max_unready_nodes               = 3
      max_unready_percentage          = 45
      new_pod_scale_up_delay          = "10s"
      scale_down_delay_after_add      = "10m"
      scale_down_delay_after_delete   = "10s"
      scale_down_delay_after_failure  = "3m"
      scan_interval                   = "10s"
      scale_down_unneeded             = "10m"
      scale_down_unready              = "20m"
      scale_down_utilization_threshold = 0.5
      empty_bulk_delete_max           = 10
      skip_nodes_with_local_storage   = true
      skip_nodes_with_system_pods     = true
    }
  }

  tags = local.common_tags

  depends_on = [
    azurerm_role_assignment.aks_network_contributor,
    azurerm_role_assignment.aks_managed_identity_operator,
  ]
}

# Additional Node Pools
resource "azurerm_kubernetes_cluster_node_pool" "additional" {
  for_each = var.additional_node_pools

  name                  = each.key
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size              = each.value.vm_size
  node_count           = each.value.node_count
  enable_auto_scaling   = each.value.enable_auto_scaling
  min_count            = each.value.enable_auto_scaling ? each.value.min_count : null
  max_count            = each.value.enable_auto_scaling ? each.value.max_count : null
  max_pods             = each.value.max_pods
  os_disk_size_gb      = each.value.os_disk_size_gb
  os_disk_type         = each.value.os_disk_type
  os_type              = each.value.os_type
  vnet_subnet_id       = data.azurerm_subnet.aks.id
  
  # Spot instances support
  priority        = each.value.priority
  eviction_policy = each.value.priority == "Spot" ? each.value.eviction_policy : null
  spot_max_price  = each.value.priority == "Spot" ? each.value.spot_max_price : null

  node_labels = each.value.node_labels
  node_taints = each.value.node_taints
  tags        = merge(local.common_tags, each.value.tags)

  upgrade_settings {
    max_surge = "10%"
  }

  depends_on = [azurerm_kubernetes_cluster.main]
}

# Kubernetes provider configuration
provider "kubernetes" {
  host                   = azurerm_kubernetes_cluster.main.kube_config.0.host
  client_certificate     = base64decode(azurerm_kubernetes_cluster.main.kube_config.0.client_certificate)
  client_key             = base64decode(azurerm_kubernetes_cluster.main.kube_config.0.client_key)
  cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.main.kube_config.0.cluster_ca_certificate)
}

# Helm provider configuration
provider "helm" {
  kubernetes {
    host                   = azurerm_kubernetes_cluster.main.kube_config.0.host
    client_certificate     = base64decode(azurerm_kubernetes_cluster.main.kube_config.0.client_certificate)
    client_key             = base64decode(azurerm_kubernetes_cluster.main.kube_config.0.client_key)
    cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.main.kube_config.0.cluster_ca_certificate)
  }
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

  depends_on = [azurerm_kubernetes_cluster.main]
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

  set {
    name  = "controller.service.annotations.service\\.beta\\.kubernetes\\.io/azure-load-balancer-health-probe-request-path"
    value = "/healthz"
  }

  depends_on = [azurerm_kubernetes_cluster.main]
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
    azurerm_kubernetes_cluster.main,
    helm_release.nginx_ingress,
    helm_release.cert_manager
  ]
}