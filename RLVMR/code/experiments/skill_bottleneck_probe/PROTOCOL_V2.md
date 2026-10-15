# ScienceWorld Exp2 — frozen execution protocol

This is an existing-checkpoint evaluation, not training. Work stays on
`codex/scienceworld-skill-bottleneck-probe` by explicit user instruction. Pilot
outputs, original training code, rewards, advantages and checkpoint files stay read-only.

## Fixed core design

- Native task 18: `lifespan-longest-lived-then-shortest-lived`; `easy`, train_diagnostic.
- Exactly u110 and u150 from the established 161254 ≤110 / 162461 111–150 lineage.
  No nearest-checkpoint or Base substitution. Optimizer states are not needed.
- Historical prefixes, in order: 48959b57b45b8fb9 (40), 1d7c46c19a7de795 (55),
  ba8d7dd0bae49fc0 (50), 5fc57e0133a566e6 (0). Original actions, raw responses,
  aliases and histories are replayed unchanged, including invalid prefix actions.
- Historical P/H/D: 2 checkpoints × 4 prefixes × 3 seeds × 3 conditions = 72.
- N: variations 0/40/50/55 × 3 seeds × 2 checkpoints = 24.
- Sampling seeds: 20261015, 20261016, 20261017. Actual generation seed remains
  `sample_seed + 1009 * absolute_policy_step_index`.
- Temperature .4, top_p 1, top_k −1; response 1024, prompt 6000, max 30 policy actions.
  Replayed policy actions consume the budget; implicit reset look-around does not.
- Original renderer/parser/tokenizer; one GPU, TP1, bf16, eager, sequential trials.

P has no appended hint. H is byte-identical to the Pilot:

> According to the previous environment feedback, the first subgoal of the original task has been completed. Continue with the remaining part of the original task.

D is the only new major condition:

> According to the previous environment feedback, the first subgoal of the original task has been completed. Your current objective is to find and focus on the animal with the shortest life span.

H/D are appended to the same original user content, on every continuation decision,
never during replay or N. D names the requested subgoal, not the correct animal or
action parameter. No prompt search, system prompt replacement or environment task change.

## Execution order and smoke

Run each checkpoint sequentially: historical P/H/D, then all its N trials, then
its conditional own-policy branches. This keeps one model loaded and avoids a
second export/load solely for branching. u110 precedes u150.
The first var40 P/H/D seed is the in-job smoke. P/H may be imported from complete
compatible Pilot evidence; D runs for real unless resuming an already completed
Exp2 D. Its task failure is not a smoke failure. Local replay/error records do not
stop other prefixes or checkpoints; confirmed identity/routing failures remain visible.

## Natural-prefix branches: fixed selection and cap

Capture the exact unhinted actor prompt immediately after the first confirmed A
transition, with all preceding raw responses/actions/feedback and remaining budget.
Rendering is a pure operation; there is no extra JVM query/action. N continues to
its normal termination regardless of whether a prefix was captured.

Within each checkpoint, sort candidates by variation, source sample_seed, trial_id;
take the first prefix for each variation, without looking at later B success.
u110 receives at most 2 prefix slots; u150 fills the remaining slots up to the
global total of 4. Every selected prefix is tested only with its source checkpoint.
P/D each use the fixed additional seeds 20261115 and 20261116: at most 16 branch trials.
These seeds differ from the source natural seeds. No A or no usable boundary input
means not_triggered, not forced extra natural sampling. A model/checkpoint that
never ran is not_run, not an observed zero-A result.

All branches have `prefix_origin=onpolicy_natural`; historical results are separate.
Shared variation alone does not make N and historical P/H/D the same full state.

## Reuse and interruption

Default source: completed Pilot 163396. Eligible old P/H on variations 40/55 and
N on variation 0 cover at most 20 core records. This is an upper bound, not a
promise of reuse. D has no Pilot counterpart.

Reuse requires the known Pilot commit, or an Exp2 run with identical execution
source hashes; matching actual source weight fingerprints, tokenizer revision,
chat template, placeholder, native source hashes, JAR identity and inference
settings; matching task/variation/reset seed, generation seed/settings, exact
source prefix, and condition-specific prompt semantics. Protocol version change
alone is not treated as equivalence: the explicit Pilot-compatible N/P/H contract
is tested. Complete raw episode/step/decision/reset evidence (and replay audit for
continuations) is required. Task failure is reusable; incomplete execution is not.

Keep original imported rows/run IDs. `evidence_manifest.json` separately records
source run, trial, code, hashes and `reused=true`. Missing/incompatible evidence is
logged and that planned trial is executed once. No retry-until-success or until-match.

Each trial writes its own closed gzip members immediately. A new job can import
completed compatible trials with `SCI_EXP2_RESUME_FROM=/absolute/old/output` into
a NEW output; it never appends to the old source. Failed/incomplete old trials may
be attempted once in this explicit resumption, preserving old evidence and source.

## Labels, comparisons and reporting

Keep Pilot A/B semantics in labels_spec.md. First post-A action categories:
B_success, exact repeat_A, invalid_action, wrong_focus, navigation/inspection,
other_or_unknown (precedence in that order). Multi-label wrong-focus/repeat metrics
remain available separately. Exact string repeat is conservative; entity aliases
are not silently unified. B_attempted is any post-A focus, not intent recognition.
Manual text-B/action-A labels default to not_reviewed; no keyword classifier.

Primary rates require completed execution, verified observed replay, and a known
outcome. Unknown, excluded, not_run and reused counts are separate. Each pairwise
P/H/D contrast also requires the same observed unhinted boundary hash. A missing H
does not invalidate an otherwise evaluable P/D pair. Observed equivalence is not
full hidden JVM equivalence. Same seed is not token-level causal control.

Report per-prefix rates first; then equal-prefix macro means with coverage and
pooled trial fractions. Own-policy branches remain separate. Preserve zero-denominator
NA. No new AE/AS; no causality attribution to GiGPO from these evaluations alone.

## Stop boundary and outputs

One pass through available planned core trials plus conditional branches, then one
analysis. No automatic u130, new seeds, prefixes, hints, training or other environments.
Local failures yield partial evidence and exit code 2, not fabricated successes.

Actual raw files are under `<run>/<checkpoint>/trials/<trial_id>/`; root
episodes.jsonl.gz is an explicitly enriched analysis view. steps_index.csv points
to full raw steps/decisions including prompts and responses. Root tables:
postA_by_prefix.csv, postA_macro_and_trial.csv, P_H_D_matched.csv,
natural_by_checkpoint.csv, onpolicy_branch_results.csv, failure_classification.csv,
checkpoint_identity.csv. reports/final_or_partial.md includes actual counts and
interpretation boundaries; up to three figures are optional. Code/config snapshots
and per-trial source provenance make both new and reused evidence inspectable.
