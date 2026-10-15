# Exp2 versus successful Pilot 163396

Changed: checkpoints restricted to u110/u150; historical prefixes extended to the
four already frozen candidates; N variations changed to 0/40/50/55; one additional
fixed sampling seed 20261017; one goal-explicit D condition; up to four own-policy
prefixes with P/D continuations (16 trials); per-trial incremental raw output and
identity-checked reuse/resumption; prefix-first and first-action analysis.

Unchanged execution for compatible N/P/H trials: native task index18, easy,
training worker/manager, renderer, parser, prompt budget, response budget,
temperature/top_p/top_k, native seed and generation seed derivation, original H
text/insertion frequency, score/done labels, and single-GPU FSDP export/inference.
V2-derived analysis fields are separate from immutable imported Pilot raw rows.

No new training, reward, AE/AS, loss, optimizer, evaluation task target, oracle
answer, global cleanup, checkpoint removal, or original training branch mutation.
The user explicitly requested reusing the existing experiment branch, so no new
Git branch is created. Each Slurm submission creates a new research output.

Resource request: testqueue, 1 GPU, 8 CPU, 64G RAM, 02:00:00, following the successful
Pilot. At timeout the last complete trial records remain usable; Slurm does not
guarantee the final report hook runs. Resume into a new output using the prior
output as a read-only evidence source.
