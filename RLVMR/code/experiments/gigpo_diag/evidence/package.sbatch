#!/bin/bash
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=ioqueue
#SBATCH --job-name=sci-evidence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=sci-evidence-%j.out
set -euo pipefail
ulimit -c 0
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
root=/scratch/prjsnnn0000269/Students/shuming/cjk
export TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/sci-evidence-${SLURM_JOB_ID}.XXXXXX")"
export TMP="$TMPDIR" TEMP="$TMPDIR" MPLCONFIGDIR="$TMPDIR/matplotlib"
export CUDA_VISIBLE_DEVICES=''
mkdir -p "$MPLCONFIGDIR" "$root/evidence"
# Same portable interpreter used by the successful ALFWorld ioqueue export.
# The GPU training venv/module Python raised SIGILL on this CPU partition.
python_bin="$root/runtimes/bin/python3.10"
echo "EVIDENCE_NODE=$(hostname) job=$SLURM_JOB_ID scratch=$TMPDIR source_read_limit=16MiB/s"
nice -n 10 "$python_bin" "$SLURM_SUBMIT_DIR/package.py" --root "$root/logs/gigpo-reproduction/scienceworld" --job 161254 --expected-group-size 8 --output "$root/evidence/scienceworld-161254-latest-pack${SLURM_JOB_ID}"
