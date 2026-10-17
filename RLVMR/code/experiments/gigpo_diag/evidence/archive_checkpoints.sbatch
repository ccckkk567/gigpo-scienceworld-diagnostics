#!/bin/bash
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=ioqueue
#SBATCH --job-name=sci-ckpt-hf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=sci-ckpt-hf-%j.log
set -euo pipefail
ulimit -c 0
umask 077
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
base=/scratch/prjsnnn0000269/Students/shuming/cjk
jobs=$(squeue -h -u "$USER" -o '%j')
if printf '%s\n' "$jobs" | grep -Eq '^(ggp-sci-(train|resume)|sci-skill-(pilot|exp2))'; then
    echo 'ScienceWorld training or checkpoint evaluation is active; archive and cleanup were not started.'
    exit 1
fi
# Load the existing token without echoing it or placing it in command arguments.
set +x
source "$base/dep-prep-credentials.env"
: "${HF_TOKEN:?HF_TOKEN missing in dep-prep-credentials.env}"
export HF_HOME="$base/.cache/huggingface"
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120 PYTHONDONTWRITEBYTECODE=1
transfer_env="$base/envs/hf-checkpoint-transfer-py310-v1"
if [[ ! -x "$transfer_env/bin/python" ]]; then
    "$base/runtimes/bin/python3.10" -I -m venv "$transfer_env"
fi
if ! "$transfer_env/bin/python" -I -c 'import importlib.metadata as m; assert m.version("huggingface-hub") == "0.36.2"' 2>/dev/null; then
    "$transfer_env/bin/python" -I -m pip install --disable-pip-version-check 'huggingface-hub==0.36.2'
fi
echo "ARCHIVE_NODE=$(hostname) job=$SLURM_JOB_ID repo=Fyvoid/scienceworld-gigpo-qwen2.5-1.5b-checkpoints"
"$transfer_env/bin/python" -I -B -u "$SLURM_SUBMIT_DIR/archive_checkpoints.py" \
    --root "$base/logs/gigpo-reproduction/scienceworld" \
    --state "$base/analysis/checkpoint-hf/scienceworld-u150" --delete
