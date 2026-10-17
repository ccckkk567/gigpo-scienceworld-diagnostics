# ScienceWorld read-only evidence export

Target training job: 161254. Export the latest finite snapshot, with no update cap.
CPU job: ioqueue, 4 CPUs, 16 GiB RAM, one hour, no GPU.

The tool belongs in the EXISTING ScienceWorld experimental branch under
RLVMR/code/experiments/gigpo_diag/evidence/. Get a pinned commit through GitHub
into a separate tools directory. Never pull/reset/checkout in the running training
checkout. Submit package.sbatch from the tool directory; do not run package.py
on a login node.

The source run is opened read-only. File lengths are frozen during enumeration;
files are copied once sequentially at 16 MiB/s. All analysis then reads copies.
Compression uses one process at nice=10. Shared-storage I/O is additional load,
so absolute zero performance impact is not claimed. No calls to training,
environment APIs, Ray, GPUs, reseeding or replay are made.

The existing Python environment is only read; no packages are installed.
Temporary files and matplotlib cache use a private directory on the CPU node.
Outputs go to /scratch/prjsnnn0000269/Students/shuming/cjk/evidence/ with the
packaging job ID. Checkpoint weights are neither opened nor copied.

Contents: raw_snapshot, original-field records, source line references, field
presence/null/missing inventories, complete and partial episode groups, AE/AS,
anchor groups and alternative action support, reward boundary deltas, native
metrics, per-task/update/stage CSVs, figures, preliminary report, code/config
snapshots, checkpoint metadata, local tracker files, full log prefixes and hashes.

Limitations:
- Not an atomic cross-file snapshot. Changed files, incomplete gzip tails and
  unavailable metadata are disclosed. No retry loop against the active writer.
- Training aggregates require the exact actor-update-returned record. Finished
  updates, complete trajectories, metrics and checkpoints have separate coverage.
- Detailed groups retain the original preselection, not outcome selection.
- Focus/connection feedback labels do not prove correct target or persistent state.
- No fabricated counterfactual advantage decomposition or world-state equivalence.
- W&B/SwanLab local files are copied where present; credential-named files and
  weights are excluded. No cloud tracker history is fetched. An active tracker
  database snapshot is raw evidence, not guaranteed transactionally consistent.
- Checkpoint directory existence is not proof of a verified resumable checkpoint.
- records contains actual observations; reanalysis contains documented derivations.
- Tests use synthetic fixtures only and are not training evidence.

Local verification: python test_package.py

## Final 150-update export with resume (2026-10-15)

Use `package_merged.sbatch` from the existing evidence checkout. Same ioqueue,
portable Python 3.10, 4 CPUs / 16G / one hour. No training checkout update is needed.
The run directory for job 162461 contains `resume161254`, not `162461`; use the
explicit run directory from its completion log, not a substring job search.

Selection is inclusive: original job 161254 updates 0..110 (0 = baseline),
resumed job 162461 updates 111..150. The original job's updates 111..120 and
the resumed job's startup validation at 110 do not enter merged records,
statistics, detailed groups or top-level plots. Missing resumed records are
reported, never replaced with old-run records. All selected joins are scoped
to source run as well as native IDs. Source JSON fields/IDs are not rewritten.

`raw_snapshot/<run>/` retains each job's captured original files, including
the superseded overlap, for provenance only. All cross-run analyses must use
`records/` and `reanalysis/`, not concatenate raw snapshots. Original per-run
reports and tracker databases remain under the source snapshots, not merged.
`selection_audit.csv` counts exclusions; `update_coverage.csv` shows source
ownership and actual counts for each kind at each update. Worker-local
optimizer/weight_sync and lifecycle records without a reliable update stay
in `auxiliary_records/<run>/`. Native cumulative counters restart per run.

`--require-through 150` checks actor-update completion records and native
metrics for every update 1..150. A missing update, missing metrics, or duplicate
completion record returns a failure status AFTER preserving the archive and
coverage report. Passing does not prove every optional field/detailed episode
exists: field_inventory, group completeness flags and read_issues still apply.
No checkpoint weights/optimizer tensors are read. Concurrent optimizer cleanup
can change the checkpoint inventory; it does not remove trajectories or metrics.

Stdlib SVG previews of the selected learning curves are also included, so the
portable interpreter does not need matplotlib installed. CSVs are the plotted
source, unknown values stay absent, and lines break at gaps/source boundaries.
Output: `analysis/gigpo-reproduction/scienceworld/scienceworld-161254-resume162461-u150-pack<export-job>.tar.gz`.
The final log prints the archive path and SHA256. No HPC-to-local transfer is
performed by this job; download the resulting archive using the usual client.

## Public checkpoint archive and cleanup (2026-10-15)

`archive_checkpoints.sbatch` runs on ioqueue (2 CPUs, 4G, 4 hours, no GPU).
It uses the existing portable Python 3.10 to create a separate transfer venv,
pinned to huggingface-hub 0.36.2. Xet/hf_transfer acceleration is disabled for
older ioqueue CPUs. The existing `dep-prep-credentials.env` supplies HF_TOKEN;
no credential value is printed or uploaded. The token must belong to Fyvoid
and permit repository writes. Auth/visibility failures stop the job.

Scope is exactly ScienceWorld runs 161254 and 162461 listed in the script.
Destination is the public model repository
`Fyvoid/scienceworld-gigpo-qwen2.5-1.5b-checkpoints`. Checkpoints are archived
in their native layout, including previously optimizer-pruned and partial
historical directories. Missing required resume shards are recorded explicitly.
Latest complete step 150 is uploaded first and retained locally. All its
model/optimizer/extra-state shards and data.pt must exist and remain unchanged.
No other environment, source code, training metrics or trajectory is deleted.

Every checkpoint is hashed locally and uploaded only where remote paths are
absent; conflicting remote files stop the job. Public, anonymous metadata at
the exact commit must match every file (LFS SHA256 or Git blob SHA1 plus size).
A public per-checkpoint manifest and local durable receipt precede deletion.
Only inventoried unchanged regular files are unlinked; empty old checkpoint
directories are removed afterward. Symlinks, path escapes, changed files,
newer checkpoint steps or missing retained shards stop cleanup. Structural
completeness does not imply a tensor deserialization test was performed.

Persistent state: `analysis/checkpoint-hf/scienceworld-u150/` (inventory,
content hashes, pinned commit receipts, per-file delete audit and result).
Keep this directory. Resubmit the same job after an interruption; it verifies
already uploaded files and can continue after a partially completed cleanup.
Data removed locally remains addressable by the recorded Hub commit. No
additional full copy of the checkpoints is staged on HPC disk. Historical run
checkpoint pointers remain in place and can refer to now-archived directories.

Submit from the existing evidence tool checkout with
`sbatch archive_checkpoints.sbatch`. Log: `sci-ckpt-hf-<job>.log` in that directory.
As authorized again on 2026-10-17, the batch job enables verified cleanup with
`--delete`, retaining the complete step 150 locally. It refuses to start while
ScienceWorld training/resume or `sci-skill-pilot`/`sci-skill-exp2` jobs are queued
or running. Do not start another checkpoint-consuming experiment during cleanup.
The initial log reports inventoried GiB and planned old-checkpoint cleanup GiB;
on retries the latter is the original inventory total, not remaining bytes.
The final result reports bytes actually deleted in that invocation.
Tests: `python -m unittest test_archive_checkpoints -v` (synthetic local Hub
fixtures; actual HPC transfer and free-space results are confirmed by job logs).
