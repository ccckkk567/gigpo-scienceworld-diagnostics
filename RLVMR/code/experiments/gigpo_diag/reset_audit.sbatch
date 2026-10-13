#!/usr/bin/env bash
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=ggp-sci-reset-audit
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/ggp-scienceworld-reset-audit-%j.out
set -euo pipefail
export GIGPO_ENV=scienceworld
module purge
module load python/3.12.5 cuda/12.6.2
source experiments/gigpo_diag/common.sh
test -f "$GIGPO_VENV/.setup-complete"
source "$GIGPO_VENV/bin/activate"
export GIGPO_CODE_COMMIT="$(git rev-parse HEAD)"
export GIGPO_RUN_ID="science-reset-audit-${SLURM_JOB_ID}"
export GIGPO_RUN_DIR="$ORION_ROOT/logs/gigpo-reproduction/scienceworld/$GIGPO_RUN_ID"
export GIGPO_DIAG_DIR="$GIGPO_RUN_DIR/diagnostics"
mkdir -p "$GIGPO_DIAG_DIR"
trap 'echo "SCIENCE_RESET_AUDIT_END job=$SLURM_JOB_ID exit=$? output=$GIGPO_RUN_DIR"' EXIT
python -m experiments.gigpo_diag.runtime science-reset-audit
