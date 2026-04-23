# Cloud-run scripts

`run_on_cloud.sh` clones the repo, installs dependencies, runs the scaling
sweep + TinyStories matched-baseline bench, and archives the artifacts.

## Launching on RunPod

1. Deploy a new pod. **A100 80GB SXM** (community) is the recommended config
   (~$1.50–2/hr). The defaults below assume 80GB VRAM; for a 40GB card set
   `TS_D_MODEL=768 TS_BATCH_SIZE=48` and the script stays within limits.
   - Template: **RunPod PyTorch 2.x** (has Python + CUDA preinstalled).
   - Volume: at least 20GB for the TinyStories corpus + artifacts.
2. Once the pod is running, open the web terminal and:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/staffman76/HDNA-Workbench/main/scripts/run_on_cloud.sh \
  -o run_on_cloud.sh
REPO_URL=https://github.com/staffman76/HDNA-Workbench.git \
  bash run_on_cloud.sh
```

3. When the run completes, artifacts are in `/workspace/artifacts/`:
   - `parity_cloud/_summary.json` + per-run JSON files (scaling sweep).
   - `tinystories_cloud/_summary.json` + `vanilla.json` / `inspectable_trace_off.json`.
   - `parity_sweep.log` / `tinystories.log`.
   - A tarball of all of the above.
4. Download the tarball via RunPod's file browser or `scp`, then terminate the pod.

## Launching on vast.ai

Similar flow. Pick an on-demand instance with a recent PyTorch image; run the
same `curl | bash` from the SSH console.

## Environment overrides

All knobs are env vars so the same script runs at different scales without
code edits.

| var | default | meaning |
|:---|:---|:---|
| `REPO_URL` | (required) | git URL to clone |
| `REPO_BRANCH` | `main` | branch to check out |
| `ARTIFACT_DIR` | `/workspace/artifacts` | where results land |
| `SKIP_PARITY` | unset | `1` to skip scaling sweep |
| `SKIP_TINYSTORIES` | unset | `1` to skip the bench |
| `D_MODEL_SWEEP` | `384,512,768,1024,1536,2048` | parity sweep d_models (comma-separated) |
| `PARITY_N_LAYERS` | `6` | layers in parity sweep |
| `PARITY_N_HEADS` | `8` | heads |
| `PARITY_SEQ_LEN` | `256` | context length |
| `PARITY_BATCH_SIZE` | `32` | batch size |
| `PARITY_STEPS` | `1500` | optimizer steps per model |
| `TS_D_MODEL` | `1024` | TinyStories d_model |
| `TS_N_LAYERS` | `12` | layers |
| `TS_N_HEADS` | `16` | heads |
| `TS_N_EXPERTS` | `4` | experts (inspectable only) |
| `TS_BATCH_SIZE` | `96` | batch size |
| `TS_SEQ_LEN` | `512` | context length |
| `TS_STEPS` | `8000` | training steps |
| `TS_LR` | `3e-4` | learning rate |
| `TS_BF16` | `1` | BF16 autocast (2-3× speedup on A100; set `0` for byte-for-byte FP32) |
| `TS_COMPILE` | `1` | try `torch.compile` (1.5-3× more); falls back gracefully on failure |
| `TS_CHECKPOINT_DIR` | `$ARTIFACT_DIR/checkpoints` | where per-condition state_dicts save. Set empty to skip. |
| `TS_MAX_BYTES` | `500000000` | corpus size cap (bytes) — 500MB |

## Rough budget

On an A100 80GB SXM (~$1.79/hr community), **with default BF16 + compile enabled**:
- Parity sweep, defaults (6 sizes up to d=2048, FP32): ~45–60 min. **~$1.50.**
- TinyStories bench at d=1024 / 12 layers / 8000 steps × 2 conditions, BF16+compile: ~25–40 min. **~$1–1.50.**
- Buffer: ~15 min. **~$0.50.**
- Total: **~$3–4** for a full run producing trained checkpoints.

For the original April 2026 FP32 run (d=768, 5000 steps, no BF16/compile, no checkpoints): ~2 hours, **$5.88** actual.

On an A100 40GB (~$0.80/hr):
- Override `TS_D_MODEL=768 TS_BATCH_SIZE=48 TS_N_LAYERS=8 TS_N_HEADS=12`.
- Parity sweep up to `D_MODEL_SWEEP="384,512,768,1024"` and `PARITY_BATCH_SIZE=24`.
- Total: **~$3–4**.

An H100 SXM cuts wall time ~2× but at ~2× the hourly. Dollar-even; A100
80GB recommended.

## Smoke testing locally first

On the 4060 Ti before renting:

```bash
# Parity sweep up to d=384 (fits in 8GB)
D_MODEL_SWEEP=64,128,256,384 python -m experiments.parity_transformer.run_sweep

# TinyStories mini-run
D_MODEL=256 N_LAYERS=4 N_HEADS=4 BATCH_SIZE=16 SEQ_LEN=256 STEPS=500 \
    python -m experiments.tinystories_bench.run
```

If these both complete without OOM, the cloud run will work at the bigger
configs.
