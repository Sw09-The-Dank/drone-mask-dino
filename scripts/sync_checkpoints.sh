#!/bin/bash
# sync_checkpoints.sh — run in the background on rank 0 (spark02) during training.
# Polls the output directory and syncs new checkpoints to rank 1 (spark01) via scp.
#
# Usage (run before starting training on rank 0):
#   bash scripts/sync_checkpoints.sh &
#   SYNC_PID=$!
#   # ... start training ...
#   wait $SYNC_PID   # or kill $SYNC_PID when done
#
# Environment variables (override defaults):
#   WORKER_IP       IP of rank 1 node         (default: 169.254.217.232)
#   WORKER_USER     SSH user on rank 1         (default: spark2gm)
#   REMOTE_DIR      Output dir on rank 1       (default: ~/Documents/drone-mask-dino/output)
#   LOCAL_DIR       Output dir on rank 0       (default: ./output)
#   POLL_INTERVAL   Seconds between sync runs  (default: 30)

set -uo pipefail

WORKER_IP="${WORKER_IP:-169.254.217.232}"
WORKER_USER="${WORKER_USER:-spark2gm}"
REMOTE_DIR="${REMOTE_DIR:-~/Documents/drone-mask-dino/output}"
LOCAL_DIR="${LOCAL_DIR:-$(pwd)/output}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

echo "[sync] Starting checkpoint sync: $LOCAL_DIR -> $WORKER_USER@$WORKER_IP:$REMOTE_DIR"
echo "[sync] Poll interval: ${POLL_INTERVAL}s  (PID $$)"

while true; do
    sleep "$POLL_INTERVAL"

    synced=0
    for f in \
        "$LOCAL_DIR"/model_final.pth \
        "$LOCAL_DIR"/last_checkpoint \
        "$LOCAL_DIR"/model_0*.pth; do
        [ -f "$f" ] || continue
        if scp -q "$f" "$WORKER_USER@$WORKER_IP:$REMOTE_DIR/" 2>/dev/null; then
            echo "[sync] $(date '+%H:%M:%S') synced $(basename "$f")"
            synced=$((synced + 1))
        else
            echo "[sync] $(date '+%H:%M:%S') WARNING: failed to sync $(basename "$f")" >&2
        fi
    done

    [ "$synced" -eq 0 ] || echo "[sync] $(date '+%H:%M:%S') sync complete ($synced files)"
done
