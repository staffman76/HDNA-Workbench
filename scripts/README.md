# Cloud-run scripts

`run_on_cloud.sh` clones the repo, installs dependencies, runs the scaling
sweep + TinyStories matched-baseline bench, and archives the artifacts.

## Launching on RunPod

1. Deploy a new pod. Community Cloud A100 40GB is the sweet spot (~$0.50–1/hr).
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
| `D_MODEL_SWEEP` | `384,512,768,1024` | parity sweep d_models (comma-separated) |
| `PARITY_N_LAYERS` | `6` | layers in parity sweep |
| `PARITY_N_HEADS` | `8` | heads |
| `PARITY_SEQ_LEN` | `256` | context length |
| `PARITY_BATCH_SIZE` | `32` | batch size |
| `PARITY_STEPS` | `1500` | optimizer steps per model |
| `TS_D_MODEL` | `768` | TinyStories d_model |
| `TS_N_LAYERS` | `8` | layers |
| `TS_N_HEADS` | `12` | heads |
| `TS_N_EXPERTS` | `4` | experts (inspectable only) |
| `TS_BATCH_SIZE` | `64` | batch size |
| `TS_SEQ_LEN` | `512` | context length |
| `TS_STEPS` | `5000` | training steps |
| `TS_LR` | `3e-4` | learning rate |
| `TINYSTORIES_MAX_BYTES` | `200_000_000` | corpus size cap (bytes) |

## Rough budget

On an A100 40GB (~$0.80/hr):
- Parity sweep, defaults: ~20 min. **~$0.30.**
- TinyStories bench, defaults: ~2–3 hours for 5000 steps × 2 conditions. **~$1.50–2.50.**
- Total: under **$3** for a full run.

An H100 cuts TinyStories to ~1 hour but at 2–3× the hourly. Roughly break-even
in dollars; A100 is recommended for the cost profile.

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
