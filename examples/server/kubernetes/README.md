# Kubernetes manifests for the TrustformeRS server example

Plain manifests plus a `kustomization.yaml` for `trustformers-server` (see
`../README.md` for what the server itself does and does not do — read that
first, since several deployment choices below follow directly from it, e.g.
there being no `/metrics` endpoint to scrape).

**Not verified against a real cluster**: no Kubernetes cluster was available
in the environment these manifests were last checked in. What *was* verified
there: `kubectl kustomize kubernetes/` builds successfully (`kubectl` alone,
no `kustomize` binary, was available) and every resource it renders was
inspected by hand against `../src/main.rs` and `../src/handlers.rs` — the
"Recently corrected" section below has specifics. Before applying to a real
cluster, at least run `kubectl apply --dry-run=server -k kubernetes/` (or
`-f kubernetes/`) against it first.

## Recently corrected

A prior version of these manifests described capabilities this server does
not have. Each item below was checked directly against the server's own
source, not assumed:

- **No `/metrics` endpoint.** `src/main.rs`'s route table has no such route.
  `deployment.yaml`'s `prometheus.io/scrape` annotations, `hpa.yaml`'s
  custom-metrics (`http_requests_per_second`) scaling rule, and
  `networkpolicy.yaml`'s "allow traffic from the monitoring namespace" rule
  all assumed one existed and have been removed — CPU/memory autoscaling
  (which needs only the metrics-server most clusters already run) is
  unaffected and still configured in `hpa.yaml`.
- **`configmap.yaml` no longer describes a config file the server doesn't
  read.** It held a `server.yaml` (workers, keep-alive, an auto-load model
  list, batching/timeout settings) and a `log4rs.yaml` (a *different*
  logging framework than the `tracing`/`tracing-subscriber` stack this
  binary actually uses) — the binary has no config-file loader at all, only
  `std::env::var` calls, and this ConfigMap was never even referenced by
  `deployment.yaml`. It now holds the real environment variables (see
  `../README.md`'s table) as plain `data:` entries, and `deployment.yaml`
  consumes them via `envFrom.configMapRef` for real.
- **`kustomization.yaml` could not actually be built.** It combined a plain
  `configmap.yaml` resource with a `configMapGenerator` of the *same name*
  generated from `server.yaml`/`log4rs.yaml` files that do not exist
  anywhere in this directory — `kubectl kustomize kubernetes/` failed
  outright on the missing files (reproduced directly against the pre-fix
  files, not inferred), and would have collided with the plain resource even
  if they had. The generator is gone; the plain `configmap.yaml` resource is
  the only source of that ConfigMap now. Its deprecated `commonLabels:` was
  also modernised to `labels:` with `includeSelectors: false` — the old form
  injected `app.kubernetes.io/version` into every selector, including the
  Deployment's `spec.selector`, which Kubernetes treats as **immutable**
  after creation; bumping that label on a live cluster would have made the
  next `kubectl apply` fail outright with an immutable-field error.
- **`rbac.yaml`'s `Role`/`RoleBinding` are gone.** They granted
  `get/list/watch` on ConfigMaps and `get/list` on Secrets through the
  Kubernetes API — permission the application never exercises, since its
  configuration comes entirely from environment variables kubelet injects,
  not from the binary calling the API server. `deployment.yaml` now also sets
  `automountServiceAccountToken: false` for the same reason. The
  `ServiceAccount` itself is kept (still referenced by `deployment.yaml`, and
  a stable pod identity is good practice regardless).
- **`deployment.yaml`'s `image:` field, and `kustomization.yaml`'s matching
  `images:` entry, are placeholders.** `trustformers/server:latest` is not
  published anywhere. Build `../Dockerfile` yourself, push it to a registry
  you control, and update both before applying.

## Applying

```bash
# Either plain manifests...
kubectl apply -f kubernetes/ -n trustformers

# ...or via Kustomize (adds the commonLabels-successor labels and the
# `trustformers/server` image-tag substitution from kustomization.yaml):
kubectl apply -k kubernetes/
```

Both target the `trustformers` namespace `namespace.yaml` creates; apply that
file (or let either command above create it — it's in `kustomization.yaml`'s
`resources:` and namespaced manifests will fail if it doesn't exist yet)
first if applying individual files out of order.

## What each manifest is for

| File | Kind | Real behaviour it configures |
|---|---|---|
| `namespace.yaml` | `Namespace` | The `trustformers` namespace everything else lives in. |
| `rbac.yaml` | `ServiceAccount` | Pod identity. No `Role`/`RoleBinding` — see above. |
| `configmap.yaml` | `ConfigMap` | The real env vars from `../README.md`'s table. |
| `pvc.yaml` | `PersistentVolumeClaim` ×2 | Storage for `MODEL_CACHE_DIR` — the RWX one (`trustformers-cache`) is what `deployment.yaml` actually mounts; the RWO one (`trustformers-cache-rwo`) is an alternative for single-node/ReadWriteOnce clusters, referenced by neither `deployment.yaml` nor `kustomization.yaml`. |
| `deployment.yaml` | `Deployment` | 2 replicas, real `/health` liveness/readiness probes, the ConfigMap wired in via `envFrom`, the PVC mounted at `MODEL_CACHE_DIR`'s value. |
| `service.yaml` | `Service` | Routes port 80 → the container's `http` (8080) port. |
| `ingress.yaml` | `Ingress` | `api.trustformers.example.com` is a placeholder host; needs an nginx ingress controller and cert-manager (for the TLS annotation) actually installed in the cluster to do anything. |
| `hpa.yaml` | `HorizontalPodAutoscaler` | CPU (70%) / memory (80%) target utilization, 2-10 replicas. |
| `pdb.yaml` | `PodDisruptionBudget` | `minAvailable: 1`. |
| `networkpolicy.yaml` | `NetworkPolicy` | Ingress from the nginx-ingress namespace and same-namespace pods on 8080; egress for DNS and for the real outbound HTTPS/HTTP calls `AutoConfig::from_pretrained`'s hub-config fallback can make (see `../README.md`). |
| `kustomization.yaml` | — | Bundles all of the above; see "Recently corrected" for what changed. |

## Troubleshooting

```bash
kubectl logs -n trustformers deployment/trustformers-server
kubectl describe pod -n trustformers <pod-name>
kubectl top pods -n trustformers
```

- **`CrashLoopBackOff` / readiness never turns green**: `/health` returns 200
  as soon as the router is serving, regardless of whether any model is
  loaded — so a failing probe almost always means the process itself didn't
  start (check `PRELOAD_MODEL_NAME`/`PRELOAD_MODEL_TASK`: if one is set
  without the other, or the checkpoint directory it names has no
  `model.safetensors`, `main()` returns an error and the process exits
  before it ever binds a port — see `../README.md`).
- **`POST /models` returns 503**: `MAX_MODELS` checkpoints are already
  loaded; `DELETE /models/{id}` one first or raise the ConfigMap's
  `MAX_MODELS` value.
- **`OOMKilled`**: lower `MAX_MODELS`, or raise `deployment.yaml`'s memory
  `limits` — this server keeps every loaded checkpoint's weights in process
  memory for as long as it stays loaded.
