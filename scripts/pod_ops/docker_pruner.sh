#!/bin/bash
# Periodic FULL docker cleanup — runs every 5 min.
# dockermin's reward gates do 256 builds/step. Even with per-rollout image.remove(),
# buildkit's builder-cache (`docker buildx`) accumulates GB/hour and overlayfs mount
# points pile up. Three prior pods died with this pattern. The fix: keep churn bounded.
set -uo pipefail
LOG="$HOME/docker_prune.log"
while true; do
  TS=$(date -u +%FT%TZ)
  echo "[$TS] prune cycle start" >>"$LOG"
  # Containers + dangling images + build cache + unused networks + volumes
  sg docker -c "docker system prune -af --volumes" >>"$LOG" 2>&1
  # Buildx builder cache (separate from `docker image prune`)
  sg docker -c "docker buildx prune -af" >>"$LOG" 2>&1
  # Evidence-collect: mount-table + disk + dockerd process metrics
  MOUNTS=$(cat /proc/mounts | wc -l)
  OVERLAY_MOUNTS=$(grep -c overlay /proc/mounts || echo 0)
  DISK=$(df -h / | awk 'NR==2 {print $3"/"$2" ("$5")"}')
  echo "[$TS] mounts=$MOUNTS overlay=$OVERLAY_MOUNTS disk=$DISK" >>"$LOG"
  sleep 300
done
