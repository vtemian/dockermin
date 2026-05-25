#!/bin/bash
# scripts/setup_pod_docker.sh
set -euo pipefail

# Install docker if not present
if ! command -v docker >/dev/null; then
  apt-get update && apt-get install -y docker.io docker-buildx-plugin
fi

# Verify daemon is up
docker ps >/dev/null || { echo "docker daemon not reachable"; exit 1; }

# daemon.json tuning - apply and restart BEFORE creating the buildx builder,
# otherwise the restart invalidates the builder we just created.
# Prime Intellect pods are containers with a host-mounted docker socket and no
# systemd, so the daemon.json write + restart only makes sense on a real host.
if command -v systemctl >/dev/null && systemctl list-units >/dev/null 2>&1; then
  mkdir -p /etc/docker
  cat > /etc/docker/daemon.json <<EOF
{
  "max-concurrent-downloads": 16,
  "max-concurrent-uploads": 16,
  "storage-driver": "overlay2"
}
EOF
  systemctl restart docker || service docker restart

  # Wait for daemon to come back after the restart.
  for _ in $(seq 1 30); do
    docker ps >/dev/null 2>&1 && break
    sleep 1
  done
else
  echo "no systemd (containerised pod), using host socket as-is"
fi

# buildx builder with cache export (post-restart, so its state survives).
docker buildx create --name dockermin \
  --driver docker-container \
  --driver-opt network=host \
  --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=10000000 \
  --buildkitd-flags '--allow-insecure-entitlement=network.host --oci-worker-gc-keepstorage=20480' \
  --bootstrap --use

# Periodic prune cron
cat > /etc/cron.d/dockermin-prune <<'EOF'
*/30 * * * * root docker system prune -af --filter "until=2h" >/dev/null 2>&1
*/30 * * * * root docker buildx prune -af --filter "until=2h" --keep-storage=20GB --builder dockermin >/dev/null 2>&1
EOF

# Preflight: the reward path needs a working docker + buildx. Fail loud now
# rather than scoring every rollout as a build failure during training.
docker run --rm hello-world >/dev/null 2>&1 || { echo "FATAL: docker not available on this pod (reward needs it)"; exit 1; }
docker buildx version >/dev/null 2>&1 || { echo "FATAL: buildx missing"; exit 1; }

echo "docker setup ok"
