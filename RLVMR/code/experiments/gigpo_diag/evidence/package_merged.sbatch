#!/bin/bash
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=ioqueue
#SBATCH --job-name=ggp-sci-export150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/analysis/gigpo-reproduction/scienceworld/export150-%j.log
set -euo pipefail
ulimit -c 0
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
base=/scratch/prjsnnn0000269/Students/shuming/cjk
export TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/sci-export150-${SLURM_JOB_ID}.XXXXXX")"
export TMP="$TMPDIR" TEMP="$TMPDIR" MPLCONFIGDIR="$TMPDIR/matplotlib"
dest="$base/analysis/gigpo-reproduction/scienceworld"
mkdir -p "$MPLCONFIGDIR" "$dest"
echo "EVIDENCE_NODE=$(hostname) job=$SLURM_JOB_ID read_limit=16MiB/s source_selection=161254:1-110,162461:111-150"
nice -n 10 "$base/runtimes/bin/python3.10" -I -B -u "$SLURM_SUBMIT_DIR/package.py" \
  --root "$base/logs/gigpo-reproduction/scienceworld" \
  --slurm-log-root "$base/logs" \
  --segment scienceworld-qwen25-15b-gigpo-161254-20261014T023812Z 0 110 \
  --segment scienceworld-qwen25-15b-gigpo-resume161254-20261014T161429Z 111 150 \
  --require-through 150 --expected-group-size 8 \
  --output "$dest/scienceworld-161254-resume162461-u150-pack${SLURM_JOB_ID}"
