# ScienceWorld compositional skill checkpoint probe

Branch: `codex/scienceworld-skill-bottleneck-probe`, based on latest experiment commit
`7972a6a69dbca96099c7975ab82ab265077ea003`. Original experiment branch is unchanged.
This directory implements the 2026-10-15 v1 study plan. No new training, reward changes, or checkpoint deletion.

## What has actually run

Local read-only preparation recovered 408 training episodes, 58 A transitions, 58 linked historical AE/AS rows,
and 10 full-text candidate prefixes from the deduplicated evidence package. See `data/inventory_report.md`.
No local GPU checkpoint inference or actual ScienceWorld study rollout has run. HPC execution is the next step.
The model-file availability, merge and replay quality reported by the pilot must be inspected before research conclusions.

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
The pilot hashes sources and exported weights, uses the original model_merger, and records actual vLLM loaded path.
Each model is loaded in a fresh process. Each trial gets a new native worker and explicit task/variation assignment.
Original manager and collector render all model-visible prompts and tokenize them; the original parser executes actions.

Outputs: `$base/analysis/skill_bottleneck_probe/pilot-JOB-TIMESTAMP/`.
Slurm log: `$base/logs/sci-skill-pilot-JOB.out`.
Each checkpoint directory has run_manifest.json, conversion/inference logs, episodes.jsonl.gz, steps.jsonl.gz,
decisions.jsonl.gz, resets.jsonl.gz, and replay_audit.jsonl. Compressed records are complete concatenated gzip members.
Root tables/report link the actual per-checkpoint records. Large model weights are not included in analysis outputs.
No training dashboard is reused. Training archives, optimizers, weights, and original branches remain untouched.

## Local preparation and validation

`prepare.py` reads the existing 3-part evidence ZIP, never the source HPC log tree, and freezes prefixes before evaluation.
It keeps historical full responses through A (even if a response contained multiple action tags, exactly as originally parsed),
and only the next decision input; it never copies future executed steps into continuation history.

`python -m unittest discover -s . -p 'test_*.py'` checks labels, historical replay comparisons, budget/lineage,
and partial-result denominators without GPUs or JVMs. Native integration and model loading are verified in the actual pilot,
not simulated into a claimed successful experiment.
