# Study vs original training evaluation

Unchanged: model family and base tokenizer; native worker/load/reset/step; native manager, parser, collector rendering;
easy simplifications; max 30 policy actions; prompt limit 6000 (error on overflow); response limit 1024;
do_sample=true, temperature 0.4, top_p 1.0, top_k -1, bfloat16; native scores and reward.

Study changes: fixed train-diagnostic task/variation/seed manifest; one environment per trial; N/P/H conditions;
one GPU with tensor parallel size 1 instead of 2, eager inference, GPU memory utilization .35 instead of .6,
sequential inference instead of rollout batching; per-decision seeds = sample_seed + 1009 * absolute policy index.
These resource and RNG choices are recorded; exact token replay of original training sampling is not claimed.
No optimizer, loss, GiGPO, KL or format-penalty optimization is run. Format validity is only observed.

Sources are read-only. The existing FSDP merger exports actor weights into a fresh node-local directory using
the original base model config. Only trusted own checkpoint shards are loaded with PyTorch's
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 compatibility flag (the native rank-0 call does not specify weights_only).
This does not require old optimizer files. A failed merge never silently substitutes base weights.

Each checkpoint uses a separate process; each trial uses a fresh owned native worker/JVM.
Output and caches are separate from training. No shared deletion, global process cleanup, HF upload, or SwanLab
training-run reuse occurs. This is a local-evidence study with independent Slurm output, not a new training run.
