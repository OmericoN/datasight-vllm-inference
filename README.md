# datasight-vllm-gpu

Private Qwen3.8-27B inference backend for the CPU
[datasight-llm-server gateway](https://github.com/OmericoN/datasight-llm-server).
Weights come from Hugging Face; vLLM serves them over HTTP. The repository is
configured for DSRI project `ub-ub-datasight`; verify live access before applying.

## Serving configuration

- Model/served name: `Qwen/Qwen3.8-27B`
- Model and tokenizer revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Base image: `vllm/vllm-openai:v0.32.0-cu129` (CUDA 12.9)
- Custom image tag: `datasight-vllm-gpu:qwen3.8-27b-v1`
- BF16, one GPU, tensor parallelism 1, 16,384-token context (input plus output),
  one active sequence, GPU memory utilization 0.90, text only.
- Thinking disabled by default; callers can override with
  `chat_template_kwargs.enable_thinking`. The Qwen reasoning parser is enabled.
- Persistent model cache: `/hf-cache/hub`; compilation caches also live on the PVC.
- Deployment starts at **zero replicas** and uses `Recreate` to avoid overlapping
  GPU consumers during updates. No public GPU Route is provided.

These are initial settings for an H100/A100 80 GB, not a measured capacity or
performance guarantee. Confirm the actual GPU, NVIDIA driver/CUDA compatibility,
quota, storage permissions, and inference during a booked session.

Prepared on 2026-10-21 in `ub-ub-datasight`:

- CPU image assembly, model download, and independent offline verification Jobs completed.
- Published image digest: `sha256:c0172fca44326b3394bc07713b8ddb610413bdcb20700e85d911ad29346a2011`.
- Installed vLLM version verified as `0.32.0+cu129`.
- All 32 snapshot files downloaded; 18 weight shards total **55,563,006,776 bytes**.
- A separate CPU Job verified the pinned snapshot offline through a read-only PVC
  mount, including each weight shard's header and expected file size.
- GPU Deployment remains at **zero replicas**; no GPU was consumed.
- All 14 local tests passed. GPU model loading and inference remain untested.

The cache is already prepared; the steps below also document how to reproduce
setup. Do not rerun create commands against existing resources without inspecting
them. Completed Jobs clean up after 24 hours; the PVC and image remain.

## Before booking GPU time

Commands below use Bash (Git Bash/WSL on Windows), except the Python commands,
which also work in PowerShell. Authenticate with `oc login` using DSRI's console
instructions; never commit kubeconfigs or tokens.

### 1. Inspect the project

```bash
oc whoami
oc -n ub-ub-datasight get deployment,pod,service,pvc,resourcequota,limitrange
oc -n ub-ub-datasight get deployment datasight-vllm-gpu -o yaml
oc -n ub-ub-datasight get serviceaccount anyuid
```

Inspect existing workloads, PVC consumers, resource settings and booking-managed
placement before applying these manifests. Preserve live settings that are not
represented here. Do not apply the zero-replica manifest to active inference.
The PVC manifest requests the existing 200Gi CephFS claim; never delete or shrink
an existing claim. Files outside the persistent mount disappear on replacement.

### 2. Build the versioned image

DSRI's 50Gi temporary-storage cap is too small for a normal OpenShift Docker
build of this base image (the first build was evicted while unpacking it).
The CPU-only assembly Job uses pinned `crane` to stream the original compressed
layers directly into DSRI's registry, adding the startup script/cache directory
and the Dockerfile's environment. No GPU, Docker daemon, or additional privileges
are used. The existing `builder` service account already has registry-push access.

Create the ImageStream once. Inspect an existing stream/Job before reusing it;
use a new image tag for a new release. Only three reviewed source files are sent:

```bash
oc -n ub-ub-datasight create imagestream datasight-vllm-gpu
oc -n ub-ub-datasight create configmap datasight-image-source --from-file=Dockerfile --from-file=entrypoint-vllm.sh --from-file=scripts/build-image.sh
oc -n ub-ub-datasight create -f dsri/image-build-job.yaml
oc -n ub-ub-datasight logs -f job/datasight-image-build
oc -n ub-ub-datasight wait --for=condition=complete job/datasight-image-build --timeout=3600s
```

Assembly implements this Dockerfile's small COPY/chmod layer and inherits its
ENV values. If the Dockerfile gains dependency-installing RUN instructions,
change the assembly path or use a full builder with sufficient storage.
The Job has explicit CPU/RAM/temporary-storage limits and a 24-hour cleanup TTL.
For reruns, inspect the completed Job and update its source ConfigMap deliberately;
delete only the terminal image-build Job before creating it again.

Alternatively, build locally:

```bash
docker build -t datasight-vllm-gpu:qwen3.8-27b-v1 .
```

A local build does not publish the image into DSRI. Use a new custom-image tag
for subsequent revisions and update both manifests rather than overwriting a
released tag. The pinned base image still needs a compatible host driver.

### 3. Validate and prepare storage/service

```bash
oc -n ub-ub-datasight apply --dry-run=server -f dsri/pvc.yaml -f dsri/service.yaml -f dsri/deployment.yaml -f dsri/download-job.yaml
oc -n ub-ub-datasight apply -f dsri/pvc.yaml -f dsri/service.yaml -f dsri/deployment.yaml
oc -n ub-ub-datasight get pvc pvc-datasight-hf-cache
```

Server dry-run does not validate model fit, pull the image, prove volume
permissions, or guarantee GPU availability. CPU, memory and ephemeral-storage
requests/limits are explicit, as [DSRI requires](https://dsri.maastrichtuniversity.nl/docs/anatomy-of-an-application/).

### 4. Download the model on CPU

This Job uses the serving image and same pinned snapshot/cache, requests **no
GPU**, and downloads roughly 55 GB. It has a two-hour deadline, one retry, and
24-hour cleanup TTL; its output remains on the PVC.

```bash
oc -n ub-ub-datasight create -f dsri/download-job.yaml
oc -n ub-ub-datasight logs -f job/datasight-model-download
oc -n ub-ub-datasight wait --for=condition=complete job/datasight-model-download --timeout=7200s
```

A successful rerun reuses the cache. To rerun before TTL cleanup, first confirm
the old Job is terminal, then delete **only that Job** and create it again:

```bash
oc -n ub-ub-datasight get job datasight-model-download
oc -n ub-ub-datasight delete job datasight-model-download
oc -n ub-ub-datasight create -f dsri/download-job.yaml
```

If it fails, inspect pod Events and logs. Both download and serve modes fail if
the persistent cache is unwritable; neither falls back to temporary model
storage. Fix the actual PVC permissions for the chosen service account instead
of adding broad privileges or downloading weights into `/tmp`. Existing weights
under the older `/hf-cache/models` location are not deleted; the new shared
location is `/hf-cache/hub` and may need another download.

Run local checks without a GPU or model download:

```bash
python -m unittest discover -s tests -v
bash -n entrypoint-vllm.sh
```

Manifest tests require PyYAML (`python -m pip install PyYAML` if absent).
The smoke tool itself has no dependencies beyond Python's standard library.

## During your booked GPU window

Book the exact project/app IDs with the
[DSRI booking form](https://dsri.maastrichtuniversity.nl/gpu-booking/).
Verify your confirmed start/end times; reservation activation and application
readiness are separate. Do not assume the booking system selects a particular
GPU model or automatically starts a zero-replica deployment.

```bash
oc -n ub-ub-datasight get resourcequota
oc -n ub-ub-datasight get deployment datasight-vllm-gpu -o yaml
oc -n ub-ub-datasight scale deployment/datasight-vllm-gpu --replicas=1
oc -n ub-ub-datasight get pods -l app=datasight-vllm-gpu -o wide
oc -n ub-ub-datasight exec deployment/datasight-vllm-gpu -c vllm -- nvidia-smi
oc -n ub-ub-datasight logs -f deployment/datasight-vllm-gpu -c vllm
```

In another terminal, wait for readiness:

```bash
oc -n ub-ub-datasight rollout status deployment/datasight-vllm-gpu --timeout=2400s
oc -n ub-ub-datasight port-forward --address=127.0.0.1 service/datasight-vllm-gpu 18000:8000
```

Run the smoke test in another terminal:

```bash
python scripts/smoke_test.py --base-url http://127.0.0.1:18000
```

It checks health, the expected model, and one small schema-constrained extraction
from synthetic text. It requires an exact supported name/quotation, normal
completion, and valid JSON; it fails on unavailable/wrong models and truncation.
This is a serving check, not an extraction-quality benchmark. It uses bounded
per-request timeouts (120 seconds by default), and never scales workloads.

For startup failures, inspect the pod's Events and logs before restarting.
The startup probe allows approximately 30 minutes. Record actual GPU memory,
driver, custom-image digest, model revision, and settings with test results.

## Through the CPU gateway

Configure the gateway separately with:

```text
LLM_BACKEND_URL=http://datasight-vllm-gpu:8000
LLM_MODEL=Qwen/Qwen3.8-27B
```

Keep gateway authentication enabled. Set `SMOKE_API_KEY` in your shell from a
secret source and run the same smoke command with its service root URL
(**without `/v1`**). The tool sends the key as a bearer header and does not print
it. For example, using an existing authenticated localhost tunnel to the gateway:

```bash
python scripts/smoke_test.py --base-url http://127.0.0.1:18001
```

The gateway's `/health` and `/ready` indicate gateway health only; successful
`/v1/models` and inference prove the backend path. Its `/gpu-status` is based on
backend HTTP health, not a direct GPU/reservation check. Use the internal Service
when callers run on DSRI; avoid exposing the GPU backend with a public Route.

## Stop and verify unavailability

Stop dispatching new inference, allow in-flight work to finish, and persist
results in DataSight before the booking ends. Then:

```bash
oc -n ub-ub-datasight scale deployment/datasight-vllm-gpu --replicas=0
oc -n ub-ub-datasight get pods -l app=datasight-vllm-gpu
oc -n ub-ub-datasight get resourcequota
python scripts/smoke_test.py --base-url http://127.0.0.1:18001 --expect-unavailable
```

Run the final command against the **gateway**, whose tunnel must remain running;
the direct GPU tunnel ends with the backend pod. The test expects gateway health
and the specific HTTP 503 `llm_backend_unavailable` error. Do not mark pending
extractions successful or silently substitute CPU inference.

`VLLM_START_MODE=hold` remains available for debugging: no Python, download, or
inference starts, readiness never succeeds, and an allocated GPU is still
consumed. Use scaling to zero for shutdown. `download` exits after the snapshot
is cached; `serve` runs vLLM. Existing `VLLM_EXTRA_ARGS` accepts simple
whitespace-separated flags only; JSON arguments belong in the shell argument
array. Override both model and revision together when testing another checkpoint.

No backup is provided by a model-cache PVC alone; preserve research outputs
separately. This repository does not own DataSight prompts, task persistence,
retries, or extraction evaluation.
