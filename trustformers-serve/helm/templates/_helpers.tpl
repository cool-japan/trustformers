{{/*
Expand the name of the chart.
*/}}
{{- define "trustformers-serve.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "trustformers-serve.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "trustformers-serve.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "trustformers-serve.labels" -}}
helm.sh/chart: {{ include "trustformers-serve.chart" . }}
{{ include "trustformers-serve.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: inference-server
app.kubernetes.io/part-of: trustformers
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "trustformers-serve.selectorLabels" -}}
app.kubernetes.io/name: {{ include "trustformers-serve.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "trustformers-serve.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "trustformers-serve.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the config map name
*/}}
{{- define "trustformers-serve.configMapName" -}}
{{- printf "%s-config" (include "trustformers-serve.fullname" .) }}
{{- end }}

{{/*
Create the secret name
*/}}
{{- define "trustformers-serve.secretName" -}}
{{- if .Values.existingSecret }}
{{- .Values.existingSecret }}
{{- else }}
{{- printf "%s-secret" (include "trustformers-serve.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Create the models PVC name
*/}}
{{- define "trustformers-serve.modelsPvcName" -}}
{{- if .Values.persistence.modelCache.existingClaim }}
{{- .Values.persistence.modelCache.existingClaim }}
{{- else }}
{{- printf "%s-models" (include "trustformers-serve.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Generate the configuration TOML content
*/}}
{{- define "trustformers-serve.configToml" -}}
[server]
host = "{{ .Values.config.server.host }}"
port = {{ .Values.config.server.port }}
num_workers = {{ .Values.config.server.numWorkers }}
enable_metrics = {{ .Values.config.server.enableMetrics }}
enable_health_check = {{ .Values.config.server.enableHealthCheck }}
{{- if .Values.config.server.maxConnections }}
max_connections = {{ .Values.config.server.maxConnections }}
{{- end }}
{{- if .Values.config.server.requestTimeout }}
request_timeout_ms = {{ .Values.config.server.requestTimeout }}
{{- end }}

[model]
model_name = "{{ .Values.config.model.modelName }}"
device = "{{ .Values.config.model.device }}"
max_sequence_length = {{ .Values.config.model.maxSequenceLength }}
enable_caching = {{ .Values.config.model.enableCaching }}
{{- if .Values.config.model.modelPath }}
model_path = "{{ .Values.config.model.modelPath }}"
{{- end }}
{{- if .Values.config.model.precision }}
precision = "{{ .Values.config.model.precision }}"
{{- end }}

[batching]
max_batch_size = {{ .Values.config.batching.maxBatchSize }}
max_wait_time_ms = {{ .Values.config.batching.maxWaitTimeMs }}
enable_adaptive_batching = {{ .Values.config.batching.enableAdaptiveBatching }}
{{- if .Values.config.batching.strategy }}
strategy = "{{ .Values.config.batching.strategy }}"
{{- end }}

[caching]
enable_distributed = {{ .Values.config.caching.enableDistributed }}
enable_warming = {{ .Values.config.caching.enableWarming }}
{{- if .Values.redis.enabled }}
redis_url = "redis://{{ .Values.redis.host }}:{{ .Values.redis.port }}/{{ .Values.redis.database }}"
{{- end }}

[caching.result_cache]
max_size_bytes = {{ .Values.config.caching.resultCache.maxSizeBytes }}
max_entries = {{ .Values.config.caching.resultCache.maxEntries }}
default_ttl = {{ .Values.config.caching.resultCache.defaultTtl }}
eviction_policy = "{{ .Values.config.caching.resultCache.evictionPolicy }}"

[caching.embedding_cache]
max_size_bytes = {{ .Values.config.caching.embeddingCache.maxSizeBytes }}
max_entries = {{ .Values.config.caching.embeddingCache.maxEntries }}
default_ttl = {{ .Values.config.caching.embeddingCache.defaultTtl }}

[caching.kv_cache]
max_size_bytes = {{ .Values.config.caching.kvCache.maxSizeBytes }}
max_sequences = {{ .Values.config.caching.kvCache.maxSequences }}
max_layers = {{ .Values.config.caching.kvCache.maxLayers }}
sharing_enabled = {{ .Values.config.caching.kvCache.sharingEnabled }}

[auth]
{{- if .Values.config.auth.jwtSecret }}
jwt_secret = "{{ .Values.config.auth.jwtSecret }}"
{{- end }}
issuer = "{{ .Values.config.auth.issuer }}"
audience = "{{ .Values.config.auth.audience }}"
{{- if .Values.config.auth.tokenExpiration }}
token_expiration = {{ .Values.config.auth.tokenExpiration }}
{{- end }}

[metrics]
enable_prometheus = {{ .Values.config.metrics.enablePrometheus }}
metrics_path = "{{ .Values.config.metrics.metricsPath }}"
{{- if .Values.config.metrics.namespace }}
namespace = "{{ .Values.config.metrics.namespace }}"
{{- end }}

{{- if .Values.tracing.enabled }}
[tracing]
enabled = true
{{- if .Values.tracing.jaegerEndpoint }}
jaeger_endpoint = "{{ .Values.tracing.jaegerEndpoint }}"
{{- end }}
{{- if .Values.tracing.otlpEndpoint }}
otlp_endpoint = "{{ .Values.tracing.otlpEndpoint }}"
{{- end }}
sampling_rate = {{ .Values.tracing.samplingRate | default 0.1 }}
{{- end }}

{{- if .Values.config.logging }}
[logging]
level = "{{ .Values.config.logging.level | default "info" }}"
format = "{{ .Values.config.logging.format | default "json" }}"
{{- if .Values.config.logging.file }}
file = "{{ .Values.config.logging.file }}"
{{- end }}
{{- end }}

{{- with .Values.config.custom }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Common annotations for all resources
*/}}
{{- define "trustformers-serve.annotations" -}}
{{- with .Values.commonAnnotations }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Generate resource limits and requests
*/}}
{{- define "trustformers-serve.resources" -}}
{{- if .Values.resources }}
resources:
  {{- if .Values.resources.limits }}
  limits:
    {{- range $key, $value := .Values.resources.limits }}
    {{ $key }}: {{ $value }}
    {{- end }}
  {{- end }}
  {{- if .Values.resources.requests }}
  requests:
    {{- range $key, $value := .Values.resources.requests }}
    {{ $key }}: {{ $value }}
    {{- end }}
  {{- end }}
{{- end }}
{{- end }}

{{/*
Create Ingress API version
*/}}
{{- define "trustformers-serve.ingress.apiVersion" -}}
{{- if and (.Capabilities.APIVersions.Has "networking.k8s.io/v1") (semverCompare ">= 1.19-0" .Capabilities.KubeVersion.Version) -}}
networking.k8s.io/v1
{{- else if .Capabilities.APIVersions.Has "networking.k8s.io/v1beta1" -}}
networking.k8s.io/v1beta1
{{- else -}}
extensions/v1beta1
{{- end }}
{{- end }}

{{/*
Return the appropriate backend for ingress
*/}}
{{- define "trustformers-serve.ingress.backend" -}}
{{- $apiVersion := include "trustformers-serve.ingress.apiVersion" . -}}
{{- if eq $apiVersion "networking.k8s.io/v1" -}}
service:
  name: {{ include "trustformers-serve.fullname" . }}
  port:
    number: {{ .Values.service.port }}
{{- else -}}
serviceName: {{ include "trustformers-serve.fullname" . }}
servicePort: {{ .Values.service.port }}
{{- end }}
{{- end }}

{{/*
Check if we should create PVC
*/}}
{{- define "trustformers-serve.createPVC" -}}
{{- if and .Values.persistence.modelCache.enabled (not .Values.persistence.modelCache.existingClaim) -}}
true
{{- else -}}
false
{{- end -}}
{{- end }}

{{/*
Storage class for PVC
*/}}
{{- define "trustformers-serve.storageClass" -}}
{{- if .Values.persistence.modelCache.storageClass -}}
{{- if eq "-" .Values.persistence.modelCache.storageClass -}}
storageClassName: ""
{{- else }}
storageClassName: {{ .Values.persistence.modelCache.storageClass | quote }}
{{- end -}}
{{- end -}}
{{- end }}

{{/*
Generate environment variables for containers
*/}}
{{- define "trustformers-serve.environment" -}}
{{- range .Values.env }}
- name: {{ .name }}
  {{- if .value }}
  value: {{ .value | quote }}
  {{- else if .valueFrom }}
  valueFrom:
    {{- toYaml .valueFrom | nindent 4 }}
  {{- end }}
{{- end }}
{{- end }}

{{/*
Generate init container for model downloading
*/}}
{{- define "trustformers-serve.modelDownloaderContainer" -}}
{{- if .Values.initContainers.modelDownloader.enabled }}
- name: model-downloader
  image: {{ .Values.initContainers.modelDownloader.image.repository }}:{{ .Values.initContainers.modelDownloader.image.tag }}
  imagePullPolicy: {{ .Values.initContainers.modelDownloader.image.pullPolicy }}
  command:
    {{- toYaml .Values.initContainers.modelDownloader.command | nindent 4 }}
  {{- with .Values.initContainers.modelDownloader.env }}
  env:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  volumeMounts:
    - name: models-storage
      mountPath: /models
  {{- with .Values.initContainers.modelDownloader.resources }}
  resources:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  securityContext:
    runAsNonRoot: true
    runAsUser: 65532
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities:
      drop:
        - ALL
{{- end }}
{{- end }}