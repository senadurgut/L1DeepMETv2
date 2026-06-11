#!/bin/bash
#SBATCH --job-name=l1deepmetv2_train
#SBATCH --partition=work
#SBATCH --gres=mps:50
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Usage:
#   sbatch slurm/train.sh [config_path] [data_folder] [ckpts_root]
#   sbatch slurm/train.sh configs/config1.yaml data_ttbar ckpts_ttbar
# Defaults to configs/config1.yaml. The run is written to a fresh
# auto-named subfolder <ckpts_root>/<config_id>_run<N>.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/export/sdurgut/scratch/L1DeepMETv2}"
CFG="${1:-configs/config1.yaml}"
DATA="${2:-data_ttbar}"
CKPTS="${3:-ckpts}"
ENV_PREFIX="/home/export/sdurgut/scratch/mamba_envs/l1deepmet_distill_env"
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"

cd "$PROJECT_DIR"
mkdir -p slurm/logs

echo "Host:        $(hostname)"
echo "Node:        ${SLURM_JOB_NODELIST:-?}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES:-?}"
echo "Project dir: $PROJECT_DIR"
echo "Config:      $CFG"
echo "Data:        $DATA"
echo "Ckpts:       $CKPTS"
nvidia-smi -L || true

micromamba run -p "$ENV_PREFIX" python train.py --cfg "$CFG" --data "$DATA" --ckpts "$CKPTS"
