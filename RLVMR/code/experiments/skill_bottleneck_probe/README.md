# ScienceWorld compositional skill checkpoint probe

Branch: `codex/scienceworld-skill-bottleneck-probe`, based on latest experiment commit
`7972a6a69dbca96099c7975ab82ab265077ea003`. Original experiment branch is unchanged.
This directory implements the 2026-10-15 Pilot and Exp2 v2 study plans. No new training, reward changes, or checkpoint deletion.

## Exp2 v2 (current experiment)

Keep this existing experiment branch and tools checkout. The frozen specification is
`PROTOCOL_V2.md`; exact differences from Pilot 163396 are in `CONFIG_DIFF_V2.md`.
Pilot 163396 completed 64 execution trials across base/u50/u110/u150. Its small results
motivate this fixed follow-up; they do not establish a general skill-integration mechanism.

Exp2 uses only u110 and u150, four historical prefixes (variations 40/55/50/0), three
fixed seeds and P/H/D, plus N on the same four variations: 96 core records. Complete,
compatible Pilot P/H and N records may supply at most 20 of them. New D states the
remaining shortest-life-span objective without naming an answer. Natural trajectories
can trigger up to 16 additional P/D branch trials under the preregistered selection rule.

Submit from this directory on HPC after syncing the pinned experiment commit:

```bash
sbatch --export="ALL,SCI_PROBE_EXPECTED_COMMIT=$(git rev-parse HEAD),SCI_EXP2_RESUME_FROM=" exp2.sbatch
```

The request remains testqueue, 1 GPU, 8 CPUs, 64G RAM and 2 hours, as in the successful
Pilot. Outputs go to `$base/analysis/skill_bottleneck_probe/scienceworld-skill-bottleneck-exp2-JOB-TIMESTAMP/`;
the log is `$base/logs/sci-skill-exp2-JOB.out`. The first var40 seed P/H/D supplies the
in-job smoke; D task success is not required. No extra submission is needed for smoke.

Core raw evidence lives in `u110/trials/TRIAL_ID/` and `u150/trials/TRIAL_ID/`.
Use `postA_by_prefix.csv`, `P_H_D_matched.csv`, `natural_by_checkpoint.csv`,
`onpolicy_branch_results.csv`, and `reports/final_or_partial.md` for the initial review.
`steps_index.csv` links every full prompt, response, executed action and native feedback.
Imported Pilot rows keep their original run IDs; separate evidence manifests record reuse.

If interrupted, explicitly resubmit the same code with `SCI_EXP2_RESUME_FROM` set to the
old Exp2 output. A new output imports completed compatible trials; the source stays read-only.
Partial or unavailable checkpoint/replay evidence remains visible, with unknown, excluded
and not-run counts; no nearest-model substitution or automatic extra sampling occurs.
`exp2_complete` means the planned execution finished, not that tasks succeeded.
`exp2_partial`/exit 2 is an inspectable delivery, not a request to automatically rerun.

Local checks use synthetic environments/model responses and test the production trial loop,
reuse and partial failures. They do not replace actual HPC checkpoint loading and JVM smoke.

## What has actually run

Local read-only preparation recovered 408 training episodes, 58 A transitions, 58 linked historical AE/AS rows,
and 10 full-text candidate prefixes from the deduplicated evidence package. See `data/inventory_report.md`.
No local GPU checkpoint inference has run. HPC results below come from the user's actual run logs.
The model-file availability, merge and replay quality must be inspected before research conclusions.

Pilot job 163357 (commit 37189d9) stopped during inventory with zero study episodes. Its preparation incorrectly
used position 22 in documentation `tasks.json` as a native routing index. The native JVM sorts task names;
the original training load receipts consistently record index 18 for this task. Protocol `skill-bottleneck-v1.1-routing`
uses those receipts and validates the frozen natural/prefix variations against the real L0 training pool during
preparation and inventory. Task name, selected variations/prefixes, sampling and labels are unchanged.

Pilot 163386 (e2eee7c) produced 16 completed base-model trials: N reached A in 0/8; P and H each completed B
in 0/4, with all eight prefix replays verified. The trained u50/u110/u150 models never loaded: the native merger
imported `verl.utils.megatron_utils` unconditionally, which requires the absent `megatron` package even for FSDP.
These results do not measure checkpoint evolution. The prior exit=0 meant the driver finished, not all models finished.

`fsdp_export.py` now runs the same locked native merger with the Megatron-only import moved inside the Megatron
function in memory. No FSDP operation changes, no package installation, and no edits to the native source checkout.
`conversion_manifest.json` records the original/adapter/executed-source hashes. CPU tests compare the entire transformed
AST and merge synthetic two-rank tensors through the real native FSDP body, checking every exported bf16 tensor exactly.
This is not a claim that the actual HPC checkpoint shards have already been converted successfully.

## Frozen pilot

- Checkpoints: base, u50, u110, u150; unavailable checkpoints use nearest eligible lineage point with a recorded reason.
- N: train variations 0,1,2,3, two fixed sampling seeds, at most 30 actions.
- P/H: two historical prefixes (variations 40 and 55), two corresponding sampling seeds each.
- Total target: 64 trials; missing models or failed local execution remain partial results.
- H: the generic stage hint in `data/protocol.json`, appended on every H continuation decision, without the B answer.
- Source lineage: original job 161254 through u110; resume162461 u111–u150. Original abandoned u111+ is excluded.

N and P use different variation sets. Do not interpret their aggregate gap as a paired state-visitation causal effect.
P/H are paired only when the observed boundary evidence matches and both replays are verified.
Repeated focus is a B-attempt proxy, not inferred intent. Label details and unknown handling: `labels_spec.md`.
Reserved variations/prefixes are recorded but not auto-substituted after seeing a failure/outcome.

## HPC execution

Reuse the existing tools checkout `$base/code/scienceworld-evidence-d7c8557`, switch it to this new branch,
and add `RLVMR/code/experiments/skill_bottleneck_probe` to its sparse checkout.
Leave `$base/code/gigpo-reproduction/scienceworld-digitalhuman` unchanged; native modules are read from there,
checked against reviewed hashes, and run with bytecode writing disabled. Do not update the checkout while this study job runs.

From this directory, submit `sbatch --export="ALL,SCI_PROBE_EXPECTED_COMMIT=$(git rev-parse HEAD)" pilot.sbatch`.
The script requests testqueue, one GPU, 8 CPU cores, 64 GiB RAM, 2 hours. It reuses the installed Python3.12 GPU
environment and Java11. It installs nothing, uses fresh node-local compilation caches, and exports weights to node-local storage.
Run this Python3.12 environment on testqueue GPU nodes, where the original training used it; do not move it to the older ioqueue CPUs that previously raised SIGILL.

Entry points (execute on an allocated compute node):

```bash
python study.py inventory --root "$base" --native-code "$native" --output "$new_inventory_dir"
python study.py run --root "$base" --native-code "$native" --output "$new_output_dir" --work "$fresh_node_local_dir"
python analyze.py "$existing_output_dir"
```

The inventory includes model shard presence and the saved checkpoint pointer. It does not require optimizer files.
The pilot hashes sources and exported weights, uses the original FSDP body via `fsdp_export.py`, and records actual vLLM loaded path.
Each model is loaded in a fresh process. Each trial gets a new native worker and explicit task/variation assignment.
Original manager and collector render all model-visible prompts and tokenize them; the original parser executes actions.

Outputs: `$base/analysis/skill_bottleneck_probe/pilot-JOB-TIMESTAMP/`.
Slurm log: `$base/logs/sci-skill-pilot-JOB.out`.
Each checkpoint directory has run_manifest.json, conversion/inference logs, episodes.jsonl.gz, steps.jsonl.gz,
decisions.jsonl.gz, resets.jsonl.gz, and replay_audit.jsonl. Compressed records are complete concatenated gzip members.
Root tables/report link the actual per-checkpoint records. Large model weights are not included in analysis outputs.
No training dashboard is reused. Training archives, optimizers, weights, and original branches remain untouched.

To complete pilot 163386, set `SCI_PROBE_REUSE_BASE_FROM` to its output directory when submitting `pilot.sbatch`
(or pass `study.py --reuse-base-from DIR`). A new output directory receives byte-identical base evidence with source
paths and hashes; original episode run IDs, git commit, and manifests are preserved. The driver checks the frozen protocol,
prefixes, native source identity and full completed trial set. It then runs only u50/u110/u150 (48 new trials).
No base inference repeats and no previous output is overwritten. The combined report contains 64 trials if all finish.
This reuse is valid for this export/orchestration-only repair: runner, prompts, task/variation/seeds, and labels are unchanged.

The summary now reports per-checkpoint and per-trial status. `pilot_complete`/exit 0 requires every planned trial
to finish execution (task success is not required). A checkpoint conversion, loading, trial error, blocked replay,
or incomplete trial set is retained as `pilot_partial`/exit 2 after the remaining checkpoints and reports finish.
Ambiguous replay remains available as auxiliary evidence. Neither partial status nor low success selects new trials.

## Local preparation and validation

`prepare.py` reads the existing 3-part evidence ZIP, never the source HPC log tree, and freezes prefixes before evaluation.
It keeps historical full responses through A (even if a response contained multiple action tags, exactly as originally parsed),
and only the next decision input; it never copies future executed steps into continuation history.

`python -m unittest discover -s . -p 'test_*.py'` checks labels, historical replay comparisons, budget/lineage,
and partial-result denominators without GPUs or JVMs. Native integration and model loading are verified in the actual pilot,
not simulated into a claimed successful experiment.
