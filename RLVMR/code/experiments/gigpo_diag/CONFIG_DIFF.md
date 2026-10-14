# scienceworld: reference recipe and execution differences

Reference: `examples/gigpo_trainer/run_sciworld.sh`, upstream commit in SOURCE.json. This run follows the locked source recipe, not an exact reproduction of a historical paper table.

- Initialization: Qwen/Qwen2.5-1.5B-Instruct + tokenizer revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Immutable named directory; full file hashes checked before loading. No old pilot/RL checkpoint, no SFT.
- Preserve train batch 16, group 8, val batch 128, native prompt/parser/history, native reward and invalid penalty 0.1, KL loss 0.01, learning rate 1e-6, gamma .95, step weight 1, native normalization, 150 epochs. Dynamic filtering disabled; exact anchors.
- Resource adaptation: 2 Slurm GPUs, TP from reference; actor and logprob microbatch 2. No hardcoded CUDA devices. Separate Python 3.12.5 venv; vLLM version from each source recipe, pinned dependencies in requirements-runtime.txt.
- Native checkpoint every 10 updates and at native final boundary; keep all native saved checkpoints, including optimizer/extra/tokenizer. Initial model is a pinned reference. A walltime interruption may lose work since the latest complete checkpoint; no claim of automatic final checkpoint on SIGKILL.
- Initial eval enabled, native eval every 5 updates. Native ALF/Sci holdout evaluated in aggregate; full holdout text excluded. Fixed train/dev study slices and prefix probes are not yet implemented and do not block native training.
- SwanLab uses the existing Tracking owner once; local process-sharded JSONL always saved. Failed network logging is nonfatal. Raw trace/model artifacts are not uploaded to GitHub or tracker.
- Diagnostics detached/read-only; explicit step identity before reordering. Native invalid penalty is captured before/after mutation; actual AE/AS/joint captured within each native core. Original sampled tokens remain the loss target. Local parameter samples and successful native weight-loading calls are logged separately from driver update count.
- ScienceWorld-only: parent samples original L0 pool with replacement, repeats explicit task/variation/reset seed, worker loads it; same task/observation/state-summary verified before training. Python sampler state has a checkpoint sidecar. Original easy preset and reference `10 * (done and score>0)` retained; no score-delta/milestone shaping. Strict uses Java completion and unrounded score>=1 only as a diagnostic. Two groups x two live resets precede training.
- AppWorld-only: native one-response code turn, train/dev split, max_interactions aligned to max_steps. `multi_episode_rollout=False`, success_coef=0 and prompt_type=vanilla; the latter two are not read by the AppWorld ordinary prompt/reward path (ALF/WebShop shaping is a separate branch). No claim that AppWorld's previous default success_coef caused shaping. 16*8 + 16*1 = 144 services derived from resolved config. Unique job/worker/reset experiments preserve native API traces. No global process cleanup. AppWorld library pinned to upstream Git commit 42b5bcf3cd334fee33f0c37c02070a9f5807add5 per Git-install recipe (PyPI 0.1.3.post1 requires pydantic1 and conflicts with vLLM).

Native schedules and environment RNG may not resume bitwise identically (ALFWorld/TextWorld and AppWorld worker state not checkpointed). ScienceWorld task sampler sidecar restores its newly introduced independent RNG. Unknown labels remain null. No actual training has run locally; only HPC results can establish updates or learning.

## Installation repair after first HPC attempt (2026-10-13)

Observed: ALFWorld job 160221 built/installed flash-attn, then pip rejected
`ray[default]` in the constraints file. AppWorld job 160225 failed resolution:
vLLM 0.8.5 requires OpenTelemetry SDK 1.26.x but Ray[default] 2.49.0 requires >=1.30.
ScienceWorld job 160223 received an unknown Content-Type for the datasets index;
the same datasets 3.6.0 was found in the other jobs. Module pip/setuptools also
leaked from /apps/spack into the activated venv in all logs.

Separate constraints-runtime.txt strips extras; requirements-runtime.txt retains
the requested extras. Clear module PYTHONPATH/PYTHONHOME in both setup and train,
invoke installation using the absolute venv Python with -I, and verify imported
pip + distribution metadata reside in that venv. Retry only recognized transient
transport/index errors, at most three attempts. Each project gets its own pip cache.
Existing venvs and installed flash-attn are reused. No directory/cache purge or
global pip installation. Training recipe, model, reward and algorithm are unchanged.

ScienceWorld resolver check also found that the required xgrammar 0.1.16 is no
longer listed by PyPI (release JSON 404). Keep vLLM 0.8.2 and its exact xgrammar
version: install xgrammar from the official v0.1.16 source commit
989222175c2a30fb7987d8bcce35bec1bf6817f2. pip checks out its submodules and builds
inside the allocated prepare job; CMake parallelism is bounded to four workers.
Source: https://github.com/mlc-ai/xgrammar/releases/tag/v0.1.16 . No alternate
package version or no-deps workaround is used.

## ScienceWorld prepare job 160512: missing package description

The new HPC log confirms xgrammar 0.1.16 and FlashAttention 2.7.4.post1 were
built and installed successfully. Editable installation of this project then
failed while setup.py tried to read the absent RLVMR/code/README.md. Both
setup.py and pyproject.toml expect that file; add a concise local package README.
The pyproject also referenced an absent code/LICENSE file. Preserve the existing
setup.py Apache-2.0 metadata declaration as inline text instead of a missing file
reference. The repository's root LICENSE.txt and all source notices are untouched.
Stop tracking the five upstream-generated verl.egg-info files (editable builds
regenerate them), and ignore egg-info/build/dist directories so installation
artifacts do not make the experiment checkout appear dirty.

This is a packaging-only repair. No environment, model, task pool, reward, parser,
training launcher or dependency version changes. The existing venv, compiled
xgrammar/FlashAttention and cached dependencies are reused on the next prepare.
Retry this project alone with experiments/gigpo_diag/submit.sh after pulling.
Actual runtime and training remain unverified until new HPC logs are available.

## Shared data stack: ALFWorld prepare job 160563

ALFWorld's full pip check, native decord call, actual environment reset/look,
torch CUDA availability and pinned model preparation now passed. The final data
preparation import then failed: installed PyArrow 26.0.0 requires NumPy >=2.0,
while these reference runtimes pin NumPy 1.26.4. Dependency metadata resolution
and pip check alone did not detect this runtime restriction. Training has not
started in the supplied log.

Keep NumPy 1.26.4 and datasets 3.6.0, explicitly pin PyArrow 19.0.1 in requirements
and constraints. All three complete Linux/Python3.12 dependency graphs resolve.
Actual local Python3.12 checks passed datasets mapping, nested prompt/extra_info
Parquet write/read, pandas/pyarrow reads and NumPy conversion. Add the same small
check after pip check in the existing prepare job; outputs are dependency checks
under the job TMPDIR, not experimental rollouts. Do not change native prepare.py,
placeholder sizes, task distribution, reward or trainer. Existing environments,
model files and native libraries are reused. Other projects receive the same
version constraint to prevent this shared failure; their jobs are not restarted
or cancelled automatically. HPC data preparation/real training remain unverified.

## Prepare 160581 / 160583: multiprocessing AF_UNIX path length

ALFWorld and ScienceWorld passed dependency/CUDA/model checks, then both failed
in the native prepare.py datasets.map(num_proc=8) SyncManager listener bind:
OSError: AF_UNIX path too long. EOFError is the subsequent parent-side symptom.
The long scratch TMPDIR, not the separately shortened RAY_TMPDIR, determines
Python multiprocessing listener paths. No optimizer updates occurred.

Keep the scratch storage directory as GIGPO_JOB_TMPDIR. Give TMPDIR/TMP/TEMP a
short alias inside a unique mode-700 mktemp directory on the compute node;
the alias points to that job's original scratch directory. Bulk builds and
diagnostic files still persist under the original scratch path. This applies
to both preparation and training, whose dataset filtering can also spawn
workers. Do not change native num_proc, datasets, tasks or training recipe.
AppWorld gets the same preventative fix; no remote running job is modified.
Local Linux validation is recorded in IPC_PATH_VALIDATION.json. The actual
HPC retry remains necessary; no preparation/training success is claimed yet.

## ScienceWorld train 160625: first grouped training reset rejected

HPC completed model initialization and 128 initial reference-evaluation episodes,
then the reset guard rejected the first training batch before policy updates.
The log alone cannot identify the mismatching group or field. Do not attribute
this to formatting or nondeterminism without raw evidence, and do not weaken
the existing comparison. Reference val/success_rate=0.0234375 is not strict success.

Local ScienceWorld 1.2.3 / Java 11: compared the seed-0 first 16 assignments in
two live Java environments, then repeated with fresh Java instances per task;
all paired observations, task descriptions, object lists, valid commands and
scores matched. This does not reproduce the 128-worker Linux run and does not
prove the HPC guard is incorrect. Preserve initial hash components verbatim and
record every rejected group's raw reset info, hashes and differing fields before
raising the original failure. Logging failures cannot bypass this guard.

reset_audit.sbatch runs the exact native seed-0 first 16x8 training reset on the
existing HPC environment, with no model, optimizer, Ray or evaluation workers.
It requests one testqueue GPU node for compatibility with the installed Python
module (the reset itself is CPU/Java work), 32 CPUs, 192 GiB, and 20 minutes.
It does not retry/resample assignments or change pool, reward, prompt or seeds.
This is diagnosis, not a repair of an as-yet unidentified mismatch. Inspect its
evidence before another full training submission. Existing failed run retained.


## ScienceWorld 160656: correct the reset guard's comparison basis

The 16x8 reset audit rejected group 10, inclined-plane-friction-named-surfaces/469,
reset_seed=374217481. The user's raw evidence shows six workers (80,81,82,83,85,87)
list tin then caesium planes, while two (84,86) reverse those two description lines.
Task/variation/seed, objects, legal commands and native score=10 match. There are
zero policy rollouts or updates in this audit. The old raw state hash includes
observation text, so its mismatch is not independent evidence of a different world.

ScienceWorld reset validation now compares task ID/number, variation, Python seed,
simplifications, task description, score and the recorded non-observation state
components. Missing evidence, inconsistent raw evidence hashes, or mismatches in
these fields still abort, even if diagnostic logging is disabled or fails. This
checks recorded evidence only; it does not certify full hidden-state determinism.
Python reset_seed is not a Java RNG reset. Other environments retain their old
exact-observation guard. Both the environment and native collector UID checks use
the ScienceWorld rule; parent task sampling/distribution stays unchanged.

Raw observations and legacy initial_state_hash are retained unchanged. Differences
are written to group_reset_text_differences.pid-*.jsonl.gz, classified only as
line_order_only or content_difference_unverified. The latter is also diagnostic,
not silently certified equivalent. No text is canonicalized for the model or for
GiGPO; native exact-anchor groups can remain separate (six versus two initially
in this reported example). This is a general separation of reset identity from
rendering, not a task-specific inclined-plane exception. Core advantage, prompt,
parser, rewards, environment/JAR version and sampling seeds are not changed.

Upstream references: RLVMR 7e22784 sends None and draws tasks independently inside
workers; BEACON samples in the parent and repeats indices. Neither reset path has
our raw-observation equality abort. Keep the prior parent-allocation repair.
https://github.com/Tencent/digitalhuman/blob/7e22784db51070fd3ed9b03a23edfca59cb5e800/RLVMR/code/agent_system/environments/env_package/sciworld/envs.py
https://github.com/ZJU-REAL/BEACON/blob/main/agent_system/environments/env_package/sciworld/envs.py

Local regression fixtures use the user-reported observation pair; other fields
are synthetic. They are not new HPC rollouts. Prior local Java reset captures can
also be replayed through the guard without invoking the simulator. HPC training
must still be retried on the existing prepared environment, with a new run ID.

Validation for this correction: 23 local regression cases passed. Offline replay
of two existing local Java capture files also passed: 16 groups / 32 reset
records per file (64 records total), with zero new resets or policy rollouts.
Capture SHA256: fdb550300d91c22e368204e1b1ef2b5451e87855bdc0bad1699375794186ca5b
and bb66373dcb5bea3012b9b18558732ba27a95106544df7a1864fac79a954cba8e.


## ScienceWorld 160694: action aliases are not structural action identities

HPC 160694 (4acf730) completed initial reference validation and one native actor
update. The second batch reset guard then rejected groups 1, 11 and 14 because
observation and sorted possible_actions text differed. SwanLab 0.10.2 was online;
this was not a tracker failure. The user's returned action-list excerpt contains
different paint referents. It is only the tail of the output, not all three raw
HPC groups; do not claim the old run contains structural action-ID evidence.

The bundled upstream Scala InputParser.chooseUniqueReferents randomly chooses
referents after repeated collisions (attempts >=10). Sorting action strings does
not remove this variation. Repeated reads in a single unchanged local 1.2.3 world
reproduce different text with identical (template_id, ordered obj_ids, type_ids).
PythonInterface.doLoad resets object UUID allocation; local repeated loads also
confirm the IDs match for the same sampled task/variation. This is evidence about
the exposed reset state and available action structure, not full hidden state.

New reset_evidence_version=2 uses the existing native JSON action enumeration,
get_valid_action_object_combinations_with_templates(), once IN PLACE OF the old
string enumeration. The two APIs use the same native action generator. Sorted
action strings from that one result still feed the existing possible_actions
field; raw JSON records are retained as possible_action_templates. No additional
enumeration, Java reseeding, simulator/JAR changes or action execution is added.
Step-time enumeration, prompts, parser, rewards, task pool, grouping/seed draws,
GiGPO exact observation anchors and update logic are unchanged.

The guard compares template ID and ordered object/type IDs, retaining duplicate
actions. It still validates task identity, variation, reset_seed, simplifications,
task description, objects, native score, raw evidence hash and agreement between
recorded action strings and JSON records. Missing or inconsistent v2 evidence
aborts. Historical v1 records keep the old string-based comparison; no invented
object IDs or retrospective equivalence claims. Successful structure matches
with different names are saved to group_reset_action_text_differences JSONL;
observations retain their prior separate diagnostic. No model input is rewritten.

Validation: 41 local regression tests pass (including core diagnostic invariance,
real mismatch rejection, argument order, multiplicity and missing evidence).
Two persistent local Java instances followed seed-0 train assignments for the
first two batches: 32 groups, 64 resets, all guards passed. Second-batch groups
1/11/14 reproduced alias differences: inclined-plane-friction-named-surfaces/127,
lifespan-longest-lived-then-shortest-lived/35, test-conductivity/261. On those three
cases, under the same test-only Scala RNG seed, old/new enumeration APIs returned
equal sorted commands and equal next RNG draws. Production never applies this
test seed. Raw capture hashes and bounded alias examples are stored in
SCIENCE_ACTION_BASIS_VALIDATION.json. These are compatibility checks with zero
policy rollouts/updates, not a new HPC training result. The 128-worker HPC rerun
and further optimizer updates remain unverified until the user executes them.

Reference source at the recipe's locked upstream commit:
https://github.com/Tencent/digitalhuman/blob/7e22784db51070fd3ed9b03a23edfca59cb5e800/RLVMR/code/agent_system/environments/env_package/sciworld/ScienceWorld/simulator/src/main/scala/scienceworld/input/InputParser.scala
Action.toTemplate preserves ordered UUID/type arguments; AgentInterface exposes
both text and JSON from mkPossibleActions in the same source tree.


## ScienceWorld 160841: isolate compiler caches on actual node-local storage

Run scienceworld-qwen25-15b-gigpo-160841-20261013T133011Z used 1eb0da9,
torch 2.6.0, Triton 3.2.0 and vLLM 0.8.2. Its small live reset preflight passed.
Initial validation generation failed before any completed evaluation batch or
optimizer update: loading __triton_launcher.so raised ImportError / ENOENT.
The path was under TMPDIR/work/torchinductor_linz3/triton/0. Despite its /tmp
prefix, work is a symlink to shared scratch. The user's later read found the
same .so file present (25760 bytes). This supports a transient cache visibility
or concurrent-access failure, not a missing installed package. The exact shared
filesystem/loader race has not been established or locally reproduced.

common.sh now assigns TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR under the
job's private node-local mktemp directory, outside the work symlink. Ray worker
creation overrides both paths with separate directories keyed by the full actor
name (worker group and rank) through runtime_env, before the actor imports
Torch/Triton. This avoids sharing files even when different workers expose their
GPU as logical device 0. The current recipe remains one node / two GPUs.
Cache path mappings are persisted under compiler_cache_workers/*.json, and
environment provenance now includes Triton and compiler-cache placement.

No caches are deleted, no training batches/actions are retried, and no errors
are swallowed. Bulk temp files, logs and checkpoints stay on scratch. Existing
run directories are retained. Torch compile, attention backend, CUDA graphs,
tensor parallelism, prompts, rewards, reset guards and sampling are unchanged.
Fresh caches can add compilation time. No dependency/prepare rerun is needed.

Validation: six local regression tests passed, covering separate worker/group paths,
repeat configuration preserving
cache contents, child-process environment propagation and concurrent same-key
writes in separate directories, provenance records, and fail-fast unusable paths.
Shell syntax and Python compilation checks also passed locally. These checks do not
exercise Linux GPU compilation or reproduce the original shared-filesystem race;
HPC evaluation and optimizer progress remain unverified pending a new run.

Primary source for supported cache environment variables and launcher loading:
https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/runtime/cache_dir_utils.py
https://github.com/triton-lang/triton/blob/v3.2.0/python/triton/runtime/cache.py
https://github.com/triton-lang/triton/blob/v3.2.0/third_party/nvidia/backend/driver.py


## ScienceWorld 160878: object aliases must not abort a matching reset

Run 160878 completed initial evaluation and two optimizer updates before the
third training reset rejected group 5: mendelian-genetics-unknown-plant / 141,
reset_seed 1562125877. The user's bounded comparison showed equal action
structure across all eight workers, but different book referent strings.
The added object-string equality guard was too strict for ScienceWorld's native
random alias selection. This was a diagnostic guard failure, not evidence that
GiGPO's advantage calculation or the simulator task had failed.

Reset evidence v3 replaces one get_possible_objects call with one native
get_possible_object_referent_types_LUT call. Both APIs enumerate the same native
referents once. Sorting the returned aliases in Java UTF-16 order preserves the
original object-list input, including duplicate names. No additional enumeration
or production Java reseeding is introduced. Step-time behavior is unchanged.

The reset guard compares sorted object UUID/type pairs and retains the existing
task, variation, seed, score and action-structure checks. Raw aliases and their
identity mapping stay in the evidence hash. Matching identities with different
names are recorded in group_reset_object_text_differences, without aborting.
Historical v1/v2 records retain their old comparison rules because they lack the
new object evidence. This does not certify full hidden-world equality. Prompts,
native observation anchors, rewards, task sampling and GiGPO updates are unchanged.

Validation is deliberately bounded to this failure and existing regressions:
59 tests passed. The actual worker reset branch ran the reported task eight times
in two local ScienceWorld 1.2.3 Java processes (four resets each, sequential).
It produced seven object-name variants, one object-identity structure and one
action structure; all passed the guard without mutating raw inputs. A separate
same-world API comparison at three test-only Scala seeds returned equal ordered
names and equal next RNG draws. See SCIENCE_OBJECT_BASIS_VALIDATION.json for
capture hashes and scope. No GPU policy rollout or HPC optimizer update was run.

HPC full job 160883 was RUNNING when this patch was prepared. Do not pull into
its active checkout; apply this patch after it exits, or use an independent
checkout for a later job. No environment preparation rerun is needed.


## ScienceWorld routing policy v4 (base 81c6bfd)

`SCIENCE_ROUTING_POLICY.md` supersedes earlier ScienceWorld content-equality guard descriptions.
Production resets again use native string getters. Parent request/load receipts and actual collector UID/token binding remain fatal contracts; optional content evidence only reports differences/unknowns. Structured evidence decoding is offline only. No task filtering, retry-to-match, reward/anchor/GiGPO change. Startup reset smoke is opt-in via `GIGPO_SCIENCE_ROUTING_CHECK=1`; update 2 now saves a native checkpoint before the existing period-10 schedule. Evaluation UID groups follow actual group_n=1. Evidence and limits: `SCIENCE_ROUTING_VALIDATION.json`.
