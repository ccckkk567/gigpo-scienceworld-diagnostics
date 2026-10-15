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
