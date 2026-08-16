#!/bin/bash
# Cluster-specific settings, sourced by every sbatch script under hpc/.
#
# These three values are the ones that differ between clusters and users.
# Override any of them by exporting before submitting:
#
#   MOE_VENV="$HOME/my_env" sbatch hpc/training_scripts/train_stage_2.sbatch
#
# The #SBATCH headers in each script — partition, GPU type, memory, walltime —
# cannot be read from here, because SLURM parses them before any shell runs.
# Those must be edited per cluster. See hpc/README.md.

# Repository checkout. Also where the scripts cd to before running Python.
: "${MOE_PROJECT_DIR:=$HOME/MoE-Specialisation}"

# Virtualenv holding the dependencies from requirements.txt.
: "${MOE_VENV:=$HOME/pytorch_latest_venv}"

# Environment modules to load, in order. A script that needs a different set
# (the CPU-only model build skips CUDA) assigns MOE_MODULES before sourcing
# this file; the := defaults above leave a pre-set value alone.
: "${MOE_MODULES:=GCCcore/11.3.0 Python/3.11.3 CUDA}"

moe_setup_environment() {
    if command -v module >/dev/null 2>&1; then
        module purge
        for mod in ${MOE_MODULES}; do
            module load "${mod}"
        done
    else
        echo "No 'module' command found — assuming the environment is already set up."
    fi

    if [ ! -f "${MOE_VENV}/bin/activate" ]; then
        echo "No virtualenv at ${MOE_VENV}. Set MOE_VENV to your environment." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${MOE_VENV}/bin/activate"

    if [ ! -d "${MOE_PROJECT_DIR}" ]; then
        echo "No checkout at ${MOE_PROJECT_DIR}. Set MOE_PROJECT_DIR." >&2
        exit 1
    fi
    cd "${MOE_PROJECT_DIR}" || exit 1

    # trust_remote_code resolves the custom model classes through this.
    export PYTHONPATH="${PWD}:${PYTHONPATH}"

    echo "=== Environment ==="
    echo "Project: ${PWD}"
    echo "Python:  $(command -v python)"
    python -c "import torch, transformers; print(f'torch {torch.__version__} | transformers {transformers.__version__} | cuda {torch.cuda.is_available()}')"
    echo "==================="
}
