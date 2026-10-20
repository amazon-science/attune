# Glossary — abbreviations used across this skill

## Parallelism dimensions (Megatron)
Divisibility/sizing constraints among these dimensions are enforced by Megatron — verify the exact
rule in source (arguments.py / parallel_state.py) rather than trusting a remembered formula.
- **TP** — tensor-model-parallel size. Shards each layer's matmuls across ranks (attention). Pair with `--sequence-parallel` at TP>1.
- **PP** — pipeline-model-parallel size. Splits the model into sequential stages across ranks. More PP = less memory/GPU but more bubble.
- **EP** — expert-model-parallel size (MoE). Number of ranks the expert set is sharded across.
- **ETP** — expert-tensor-parallel size (MoE). Shards each individual expert's matmul.
- **CP** — context-parallel size. Shards the sequence dimension (helps long-seq attention memory).
- **DP** — data-parallel size (derived from the other dimensions and world size). Attention data replication.
- **EDP** — expert-data-parallel size (derived, MoE analogue of DP).
- **vpp** — virtual pipeline parallelism (`--num-layers-per-virtual-pipeline-stage`). Interleaves PP stages to shrink the bubble; starves on short-seq / few-microbatch runs (the minimum-microbatch requirement is enforced by Megatron — verify in source).
- **world / world_size** — total GPUs across all parallelism dimensions.

## Batching / throughput
- **bs** — global batch size (`--global-batch-size`). Workload-coupled (affects convergence); NOT a pure throughput knob.
- **mt** — max-tokens-per-gpu (`--max-tokens-per-gpu`). With `--use-dynamic-batch-size`, packs variable-length samples into microbatches; the main fixed-bs throughput knob.
- **microbatch** — one forward/backward chunk; `num_microbatches` per step ≈ tokens-per-step / (DP × mt) — only DP replicas consume separate microbatches (TP/CP cooperate on the same one, PP pipelines the same set), so it's DP, not world, in the denominator.
- **tok/s/GPU** — the north-star metric: `aggregate_tok_per_s_per_gpu = total_tokens / total_step_time / num_gpus` over the measured window (steps 5-19 at the defaults).
- **MFU** — Model FLOPs Utilization: achieved FLOPs / peak GPU FLOPs. Far below dense-training levels for comm-bound large-MoE SFT, even when well-tuned.
- **IQR** — inter-quartile range (`p75 − p25`) of step times; a measure of how much step time varies within a run. Wide IQR / bimodal step times signal instability (per-rank memory or JIT pressure).

## Stack / infra
- **slime** — the RL/SFT training wrapper around Megatron used here (rollout + actor-train loop).
- **SGLang** — the inference engine slime uses for rollout generation.
- **rollout** — the generation phase each step (trivial in SFT; dominant in RL). `actor_train` is the training phase.
- **benchmark run** — one N-step job (`AUTOTUNE_NUM_STEPS`, default 20 unprofiled / 23 profiled) over the ranking workload's frozen rows (by default rows of exactly `AUTOTUNE_SEQ_LEN` tokens, 8K): discard the warm-up steps (0-4 by default, `AUTOTUNE_WINDOW_START`) and measure the rest (5..19; the same window profiled or not); a profiled run adds the profiler's setup step N, its warm-up step N+1 and the captured step N+2 (20, 21, 22 by default; up to 22-24); steps 0-19 of an unprofiled run of at least 20 steps also serve the correctness check, and a 20-step confirm run serves it otherwise.
- **EFA** — AWS Elastic Fabric Adapter (the inter-node NIC fabric); intranode uses NVLink/NVSwitch instead.
