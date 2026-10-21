#!/busybox/sh
set -eu

registry=image-registry.openshift-image-registry.svc:5000
target="${registry}/${NAMESPACE}/datasight-vllm-gpu:qwen3.8-27b-v1"
export HOME=/tmp
export DOCKER_CONFIG=/tmp/docker
mkdir -p "$DOCKER_CONFIG" /tmp/layer/hf-cache
cp /source/entrypoint-vllm.sh /tmp/layer/entrypoint-vllm.sh
chmod 0755 /tmp/layer/entrypoint-vllm.sh
chmod 0777 /tmp/layer/hf-cache
tar -cf /tmp/layer.tar -C /tmp/layer .

# Keep public roots and trust the cluster's internal registry CA; never disable TLS.
cat /etc/ssl/certs/ca-certificates.crt \
  /var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt > /tmp/ca-bundle.crt
export SSL_CERT_FILE=/tmp/ca-bundle.crt
cat /var/run/secrets/kubernetes.io/serviceaccount/token | \
  crane auth login "$registry" --username serviceaccount --password-stdin

# shortcut: assembly implements this Dockerfile's COPY/chmod only; use a full builder if RUN gains dependency installation.
base="$(sed -n 's/^FROM /docker.io\//p' /source/Dockerfile)"
set -- crane mutate "$base" --platform linux/amd64 \
  --append /tmp/layer.tar --entrypoint /entrypoint-vllm.sh \
  --exposed-ports 8000 --tag "$target"
while IFS= read -r instruction; do
  case "$instruction" in
    'ENV '*) set -- "$@" --env "${instruction#ENV }" ;;
  esac
done < /source/Dockerfile
"$@"
echo "Image ready: ${target}"
crane digest "$target"
