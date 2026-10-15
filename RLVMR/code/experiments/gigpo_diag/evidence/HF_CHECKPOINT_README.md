---
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- scienceworld
- gigpo
- checkpoint-archive
---

# ScienceWorld GiGPO native checkpoint archive

This repository archives the existing native FSDP checkpoints from ScienceWorld
training jobs 161254 and 162461. The second job resumed from update 110 and
completed update 150. These are training artifacts, not a converted Transformers
`from_pretrained` model. There is no new training or changed reward in this archive.

Files preserve their original bytes and run identity under
`checkpoints/<source-run>/checkpoints/global_step_<N>/`.
Both jobs' checkpoints are retained separately, including any partial checkpoint
left by the original job's out-of-memory failure at step 120. For the continued
training history after step 110, use job 162461's run:
`scienceworld-qwen25-15b-gigpo-resume161254-20261014T161429Z`.

`manifests/` records each archived checkpoint's file sizes, SHA256 values, Git
blob hashes, immutable data revision, and missing files required for native
two-rank training resumption. Previously deleted optimizer shards are not
reconstructed. A directory with missing shards is not a resumable checkpoint.
Presence of all expected files is a structural check, not a tensor load test.

The complete step 150 checkpoint is also kept on the source HPC filesystem.
Older checkpoints are removed locally only after their public Hub file sizes
and content hashes have been verified at a pinned revision. Large files are
checked against Hub LFS SHA256 metadata; Git files against their Git blob hashes.
The source run's old `latest_checkpointed_iteration.txt` remains historical
metadata and may point to a checkpoint that is now archived only on the Hub.

To restore an archived checkpoint, download its whole directory at the
`data_revision` in its manifest. Keep the original actor shards, extra state,
dataloader state and any environment sampler sidecar together. Point the native
trainer's `resume_from_path` to that checkpoint directory; only checkpoints with
all required optimizer/model/extra-state shards can resume full training. The
archive does not claim to capture a complete live ScienceWorld environment.

Training metrics, trajectories, credentials and unrelated environments are not
part of this checkpoint-only repository. During an upload the repository may be
incomplete: a checkpoint's published manifest identifies its verified files.
