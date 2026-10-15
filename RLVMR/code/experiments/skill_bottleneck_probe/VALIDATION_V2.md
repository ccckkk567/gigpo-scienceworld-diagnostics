# Exp2 v2 local validation — 2026-10-15

Baseline: successful Pilot commit `1ccc08529b59cfd967f632a6d9ab84e9ad9bc68b`.
Work remains on `codex/scienceworld-skill-bottleneck-probe`.

Executed in the actual study directory on Windows with the existing local Python
environment and a workspace-owned temporary directory:

```text
python -m unittest discover -s . -p test_*.py -v
Ran 21 tests in 18.049s
OK
```

The suite contains the 14 existing Pilot/export tests and 7 Exp2 tests:

- Frozen 96-trial design, historical alias preservation, unchanged P/H/N hint behavior,
  and exact answer-free D text.
- Production trial loop with synthetic native feedback/model responses: natural A
  capture excludes later outcomes; independent P/D replay preserves history/budget;
  D is absent during replay; first-action classes and absolute-step generation seeds.
- Compatible Pilot subset reuse preserves original raw rows and run IDs; changed
  source weights/generation settings or incomplete raw evidence reject reuse.
- Exp2 incremental reuse accepts completed compatible records and rejects explicit
  execution/import failures or changed execution code.
- Natural source selection is deterministic and does not inspect later B outcome;
  fixed branch seeds and prefix caps are respected.
- Injected var50 replay failures leave other core trials running, keep zero-denominator
  NA and exclusions, and produce a not-triggered branch record when no A occurs.
- Full driver with missing u110 and synthetic u150 export/inference keeps the 96-trial
  plan, runs only the exact available u150, and delivers a partial report plus all three
  optional plots. This test deliberately uses synthetic rows, not real model evidence.

The original FSDP adapter tests also passed exact synthetic two-rank bf16 tensor
comparison and AST checks that leave native FSDP operations unchanged.

`sh -n exp2.sbatch` and `git diff --check` passed. Windows sandbox restrictions on
MSYS startup required running the syntax-only check outside the sandbox. No Slurm
script was executed locally. Existing native merger escape/deprecation warnings
were non-fatal; empty-denominator plot fixtures emitted empty-legend warnings.

No actual Exp2 HPC model/JVM trial has been run at this validation point. Real
checkpoint hashes, model-load status, replay quality and results will be recorded
by the submitted job. The first var40 P/H/D group is its small smoke check; task
success is not required. Missing/incompatible old Pilot evidence is recorded and
the corresponding planned trial runs once. No extra seeds or replacement models
are selected in response to outcomes.
