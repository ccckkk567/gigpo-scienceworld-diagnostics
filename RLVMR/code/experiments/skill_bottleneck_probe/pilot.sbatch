#!/usr/bin/env bash
#SBATCH -A prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --job-name=sci-skill-pilot
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/sci-skill-pilot-%j.out
set -euo pipefail
ulimit -c 0
module purge
module load python/3.12.5 cuda/12.6.2
root=/scratch/prjsnnn0000269/Students/shuming/cjk
study="$SLURM_SUBMIT_DIR"
native="$root/code/gigpo-reproduction/scienceworld-digitalhuman/RLVMR/code"
test "$(git -C "$study" branch --show-current)" = codex/scienceworld-skill-bottleneck-probe
test -n "${SCI_PROBE_EXPECTED_COMMIT:-}"
test "$(git -C "$study" rev-parse HEAD)" = "$SCI_PROBE_EXPECTED_COMMIT"
source "$root/envs/gigpo-scienceworld-py312-v1/bin/activate"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE GIGPO_DIAG_DIR GIGPO_RUN_ID
unset SWANLAB_RUN_ID WANDB_RUN_ID
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export JAVA_HOME="$root/runtimes/jdk-11.0.28+6"
export PATH="$JAVA_HOME/bin:$PATH"
export JAVA_TOOL_OPTIONS='-Xmx512m -XX:ActiveProcessorCount=1'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled SWANLAB_MODE=disabled
scratch=$(mktemp -d "/tmp/sci-probe-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR="$scratch" TMP="$scratch" TEMP="$scratch"
export XDG_CACHE_HOME="$scratch/cache" TORCHINDUCTOR_CACHE_DIR="$scratch/inductor" TRITON_CACHE_DIR="$scratch/triton" MPLCONFIGDIR="$scratch/mpl"
export PYTHONPATH="$native"
out="$root/analysis/skill_bottleneck_probe/pilot-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
# Each study process uses only its own worker. No global Ray/JVM/cache cleanup.
finish() {
  result=$?
  trap - EXIT
  if test -f "$out/RUN_MANIFEST.json"; then python "$study/analyze.py" "$out" || true; fi
  printf 'SCI_PROBE_END exit=%s output=%s node_local_exports=%s\n' "$result" "$out" "$scratch/exports"
  exit "$result"
}
trap finish EXIT
printf 'SCI_PROBE_START commit=%s output=%s\n' "$SCI_PROBE_EXPECTED_COMMIT" "$out"
python "$study/study.py" run --root "$root" --native-code "$native" --output "$out" --work "$scratch/exports"
