#!/bin/bash
# sync_checkpoints.sh — run in the background on rank 0 (spark02) during training.
# Polls the output directory and syncs new checkpoints to rank 1 (spark01) via scp.
# Requires passwordless SSH from rank 0 -> rank 1, because background jobs cannot
# answer password prompts.
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
#   SSH_KEY         Optional private key path   (default: unset)

set -uo pipefail

WORKER_IP="${WORKER_IP:-169.254.217.232}"
WORKER_USER="${WORKER_USER:-spark2gm}"
REMOTE_DIR="${REMOTE_DIR:-~/Documents/drone-mask-dino/output}"
LOCAL_DIR="${LOCAL_DIR:-$(pwd)/output}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
SSH_KEY="${SSH_KEY:-}"

SSH_OPTS=(
    -o BatchMode=yes
    -o ConnectTimeout=5
    -o StrictHostKeyChecking=accept-new
)

if [ -n "$SSH_KEY" ]; then
    SSH_OPTS+=( -i "$SSH_KEY" -o IdentitiesOnly=yes )
fi

log_warn_once=0

echo "[sync] Starting checkpoint sync: $LOCAL_DIR -> $WORKER_USER@$WORKER_IP:$REMOTE_DIR"
echo "[sync] Poll interval: ${POLL_INTERVAL}s  (PID $$)"

if [ ! -d "$LOCAL_DIR" ]; then
    echo "[sync] ERROR: local output directory does not exist: $LOCAL_DIR" >&2
    exit 1
fi

if ! ssh "${SSH_OPTS[@]}" "$WORKER_USER@$WORKER_IP" "mkdir -p $REMOTE_DIR" 2>/dev/null; then
    echo "[sync] ERROR: cannot access worker over passwordless SSH: $WORKER_USER@$WORKER_IP" >&2
    echo "[sync]        Run on rank 0: ssh-copy-id $WORKER_USER@$WORKER_IP" >&2
    echo "[sync]        Then verify:   ssh -o BatchMode=yes $WORKER_USER@$WORKER_IP 'echo ok'" >&2
    exit 1
fi

while true; do
    sleep "$POLL_INTERVAL"

    if ! ssh "${SSH_OPTS[@]}" "$WORKER_USER@$WORKER_IP" "test -d $REMOTE_DIR && test -w $REMOTE_DIR" 2>/dev/null; then
        if [ "$log_warn_once" -eq 0 ]; then
            echo "[sync] WARNING: worker dir not writable or SSH unavailable: $REMOTE_DIR" >&2
            echo "[sync]          Sync paused; will retry." >&2
            log_warn_once=1
        fi
        continue
    fi
    log_warn_once=0

    synced=0
    for f in \
        "$LOCAL_DIR"/model_final.pth \
        "$LOCAL_DIR"/last_checkpoint \
        "$LOCAL_DIR"/model_0*.pth; do
        [ -f "$f" ] || continue
        if scp "${SSH_OPTS[@]}" -q "$f" "$WORKER_USER@$WORKER_IP:$REMOTE_DIR/" 2>/dev/null; then
            echo "[sync] $(date '+%H:%M:%S') synced $(basename "$f")"
            synced=$((synced + 1))
        else
            echo "[sync] $(date '+%H:%M:%S') WARNING: failed to sync $(basename "$f")" >&2
        fi
    done

    [ "$synced" -eq 0 ] || echo "[sync] $(date '+%H:%M:%S') sync complete ($synced files)"
done
