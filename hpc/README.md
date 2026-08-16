# SLURM job scripts

Batch scripts for the full pipeline. They were written for one specific H100/L40S
cluster, so **treat them as a starting point, not a drop-in**.

## Running

Submit from the repository root — the log paths are relative to the submit
directory:

```bash
mkdir -p logs/slurm
sbatch hpc/model_scripts/create_moe_model.sbatch   # Stage 0, CPU, run once
sbatch hpc/training_scripts/train_stage_1.sbatch   # 1 GPU
sbatch hpc/training_scripts/train_stage_2.sbatch   # 4 GPUs, FSDP
sbatch hpc/training_scripts/train_stage_2.5.sbatch # 2 GPUs, FSDP
sbatch hpc/training_scripts/train_stage_3.sbatch   # 4 GPUs, FSDP
sbatch hpc/training_scripts/train_dense.sbatch     # 4 GPUs, control baseline
```

Logs land in `logs/slurm/<job-name>_<job-id>.{out,err}`.

## What to adapt

### 1. Paths and environment — `hpc/cluster_env.sh`

Every script sources this file. Edit the defaults there, or export overrides at
submission time:

| Variable | Default | What it is |
|---|---|---|
| `MOE_PROJECT_DIR` | `$HOME/MoE-Specialisation` | Repository checkout |
| `MOE_VENV` | `$HOME/pytorch_latest_venv` | Virtualenv with `requirements.txt` installed |
| `MOE_MODULES` | `GCCcore/11.3.0 Python/3.11.3 CUDA` | Environment modules, loaded in order |
| `MOE_BASE_MODEL` | `$HOME/models/Mistral-7B-v0.3` | Stage 0 input (that script only) |
| `MOE_OUTPUT_MODEL` | `$HOME/models/Mistral-7B-MoE` | Stage 0 output (that script only) |

```bash
MOE_VENV="$HOME/envs/moe" sbatch hpc/training_scripts/train_stage_2.sbatch
```

The setup fails loudly if the virtualenv or the checkout is missing, rather than
running against the wrong Python.

### 2. Scheduler directives — the `#SBATCH` header in each script

Partition names (`gpu-h100`, `gpu-l40s`, `bigmem`), `--gres`, `--mem` and
`--time` are specific to that cluster and **cannot** be moved into
`cluster_env.sh`: SLURM parses the header before any shell runs. Edit them in
place.

Where the GPU count changes, change `--nproc_per_node` in the same script to
match — the two are independent and a mismatch wastes the allocation.

### 3. Data and model paths — `configs/training_config.yaml`

The scripts themselves read every path from there, not from the environment.

## Notes

- Stage 1 is single-GPU (`python -u`); Stages 2, 2.5, 3 and the dense baseline
  use `srun torchrun` with FSDP.
- Stage 3 picks a random rendezvous port, because a fixed one collides with
  other jobs sharing a node.
- `NCCL_TIMEOUT` is ignored from PyTorch 2.5 onwards; the scripts set
  `TORCH_NCCL_BLOCKING_WAIT` and `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` instead.
- No GPU to hand? `make demo` runs the whole pipeline on CPU in about 20
  seconds against synthetic fixtures.
