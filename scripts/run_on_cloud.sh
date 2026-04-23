#!/usr/bin/env bash
# End-to-end cloud runner. Assumes a fresh Ubuntu + CUDA box
# (RunPod / vast.ai / Lambda) with Python 3.10+, pip, and git.
#
# Usage:
#   REPO_URL=https://github.com/staffman76/HDNA-Workbench.git \
#       bash run_on_cloud.sh
#
# Optional env overrides:
#   REPO_BRANCH=main
#   ARTIFACT_DIR=/workspace/artifacts
#   SKIP_PARITY=1          # skip the scaling sweep
#   SKIP_TINYSTORIES=1     # skip the head-to-head bench
#   D_MODEL_SWEEP=384,512,768,1024
#   TS_D_MODEL=768 TS_N_LAYERS=8 TS_BATCH_SIZE=64 TS_STEPS=5000

set -euo pipefail

REPO_URL="${REPO_URL:?REPO_URL is required}"
REPO_BRANCH="${REPO_BRANCH:-main}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/workspace/artifacts}"
WORKDIR="${WORKDIR:-/workspace/hdna}"

# Force unbuffered Python so `| tee` doesn't swallow progress output until
# the 4KB pipe buffer fills (previous cloud run lost ~15 min of visibility
# to this before we caught it).
export PYTHONUNBUFFERED=1

log() { printf '\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }

log "GPU info"
nvidia-smi | head -20 || log "nvidia-smi not available (CPU-only?)"

log "cloning repo"
if [[ ! -d "$WORKDIR" ]]; then
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

log "installing package + torch"
pip install --upgrade pip
pip install -e ".[pytorch]"
pip install matplotlib

log "smoke test: cuda available?"
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    p = torch.cuda.get_device_properties(0)
    print(f"VRAM: {p.total_memory / 1024**3:.1f} GB")
PY

mkdir -p "$ARTIFACT_DIR"

# ---------------------------------------------------------------------------
# Scaling sweep
# Defaults sized for A100 80GB SXM. Extends to d=2048 for a full claim.
# ---------------------------------------------------------------------------
if [[ "${SKIP_PARITY:-0}" != "1" ]]; then
    log "== parity_transformer sweep =="
    D_MODEL_SWEEP="${D_MODEL_SWEEP:-384,512,768,1024,1536,2048}" \
    N_LAYERS="${PARITY_N_LAYERS:-6}" \
    N_HEADS="${PARITY_N_HEADS:-8}" \
    SEQ_LEN="${PARITY_SEQ_LEN:-256}" \
    BATCH_SIZE="${PARITY_BATCH_SIZE:-32}" \
    STEPS="${PARITY_STEPS:-1500}" \
    RESULTS_DIR="$ARTIFACT_DIR/parity_cloud" \
        python -m experiments.parity_transformer.run_sweep 2>&1 \
        | tee "$ARTIFACT_DIR/parity_sweep.log"
fi

# ---------------------------------------------------------------------------
# TinyStories matched-baseline
# Defaults sized for A100 80GB SXM: d=1024, 12 layers, ~200M params — the
# size class people actually train from scratch. VRAM-headroom permits
# batch 96 / seq 512 without gradient checkpointing.
#
# BF16 + torch.compile + fused AdamW are enabled by default on cloud runs
# for ~4-6x speedup over FP32 eager mode. Set BF16=0 COMPILE=0 to reproduce
# the earlier FP32 numbers byte-for-byte.
#
# SAVE_CHECKPOINT_DIR dumps each trained condition's state_dict so downstream
# inspection / case-study work can load the actual weights locally.
# ---------------------------------------------------------------------------
if [[ "${SKIP_TINYSTORIES:-0}" != "1" ]]; then
    log "== tinystories_bench =="
    D_MODEL="${TS_D_MODEL:-1024}" \
    N_LAYERS="${TS_N_LAYERS:-12}" \
    N_HEADS="${TS_N_HEADS:-16}" \
    N_EXPERTS="${TS_N_EXPERTS:-4}" \
    BATCH_SIZE="${TS_BATCH_SIZE:-96}" \
    SEQ_LEN="${TS_SEQ_LEN:-512}" \
    STEPS="${TS_STEPS:-8000}" \
    LR="${TS_LR:-3e-4}" \
    BF16="${TS_BF16:-1}" \
    COMPILE="${TS_COMPILE:-1}" \
    SAVE_CHECKPOINT_DIR="${TS_CHECKPOINT_DIR:-$ARTIFACT_DIR/checkpoints}" \
    TINYSTORIES_MAX_BYTES="${TS_MAX_BYTES:-500000000}" \
    RESULTS_DIR="$ARTIFACT_DIR/tinystories_cloud" \
        python -m experiments.tinystories_bench.run 2>&1 \
        | tee "$ARTIFACT_DIR/tinystories.log"
fi

log "archiving artifacts"
cd "$ARTIFACT_DIR"
tar czf "$ARTIFACT_DIR/run_$(date +%Y%m%d_%H%M%S).tar.gz" \
    parity_cloud/ tinystories_cloud/ *.log 2>/dev/null || true

log "done. artifacts in $ARTIFACT_DIR"
ls -la "$ARTIFACT_DIR"
