#!/bin/bash
#SBATCH --job-name=l1deepmetv2_train
#SBATCH --partition=work
#SBATCH --gres=mps:100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-0:00:00 
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Usage:
#   sbatch slurm/train.sh [config_path] [data_folder] [ckpts_root] [restore_file]
#   sbatch slurm/train.sh configs/config1.yaml data_ttbar ckpts_ttbar
# Defaults to configs/config1.yaml. The run is written to a fresh
# auto-named subfolder <ckpts_root>/<config_id>_run<N>.
#
# To RESUME a cancelled run, pass the run folder as ckpts_root and a
# restore_file (last|best); training continues in that same folder:
#   sbatch slurm/train.sh ckpts/config1_run1/config.yaml data_ttbar ckpts/config1_run1 last

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/export/sdurgut/scratch/L1DeepMETv2}"
CFG="${1:-configs/config2.yaml}"
DATA="${2:-data_ttbar}"
CKPTS="${3:-ckpts}"
RESTORE="${4:-}"
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
echo "Restore:     ${RESTORE:-<none>}"
nvidia-smi -L || true

RESTORE_ARG=""
if [ -n "$RESTORE" ]; then RESTORE_ARG="--restore_file $RESTORE"; fi

# Send the run's stdout/stderr to node-local disk instead of the NFS-backed
# slurm/logs/*.out. Per-batch tqdm writes to an NFS file handle are what produced
# the ESTALE ("stale file handle") crash; a local fd cannot go stale. The log is
# synced back to NFS every 5 min and once more at exit, so checkpoints (which stay
# on NFS for durability/resume) and the full log both survive a job stop.
LOCAL_LOG="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/${SLURM_JOB_NAME:-l1deepmet}_${SLURM_JOB_ID:-local}.log"
NFS_LOG="slurm/logs/${SLURM_JOB_NAME:-l1deepmet}_${SLURM_JOB_ID:-local}.full.log"
copy_log() { cp -f "$LOCAL_LOG" "$NFS_LOG" 2>/dev/null || true; }
( while true; do sleep 300; copy_log; done ) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; copy_log' EXIT
echo "Run log (node-local, live): $LOCAL_LOG"
echo "Synced to (NFS, every 5min): $NFS_LOG"

micromamba run -p "$ENV_PREFIX" python -u train.py --cfg "$CFG" --data "$DATA" --ckpts "$CKPTS" $RESTORE_ARG > "$LOCAL_LOG" 2>&1
