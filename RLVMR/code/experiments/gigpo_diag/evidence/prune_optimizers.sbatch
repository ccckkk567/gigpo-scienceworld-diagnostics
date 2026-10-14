#!/bin/bash
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=ioqueue
#SBATCH --job-name=prune-old-optim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=prune-optim-%j.log
set -euo pipefail
ulimit -c 0
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
base=/scratch/prjsnnn0000269/Students/shuming/cjk
"$base/runtimes/bin/python3.10" -I -B -u "$SLURM_SUBMIT_DIR/prune_optimizers.py" --root "$base/logs/gigpo-reproduction" --audit "$base/analysis/checkpoint-prune/job-${SLURM_JOB_ID}" --delete
