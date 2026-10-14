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
