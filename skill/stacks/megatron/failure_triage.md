# Failure triage: Megatron-LM (slime, miles)

Classify a crash before relaunching; an identical relaunch is not a fix. The entries below come from past runs, and
signatures change with the image, the hardware and the nodes: confirm the cause in the current source or logs before
acting on an entry.

## Triage procedure

1. Read the last 200 lines of the run's `train.log` through the log-access pod (`poll_run.sh` prints the path).
2. Match them against the entries below, memory signals first.
3. Ledger the run as `errored` with the triage result and propose a specific config change. An infrastructure failure
   (node, network, preemption) gets one retry of the identical recipe on other nodes (SKILL.md, Budget accounting).
4. If nothing matches, record the error and the last 50 lines of the log verbatim, form a hypothesis from the message
   (for an assertion, read the asserting source), and try one reasoned fix. If it fails too, record the pattern in
   `RESULTS.md`, propose it in `LESSONS.md` when it generalizes, and try a different direction. After three
   consecutive novel crashes that no fix unblocks, record "Blocked on human" in `RESULTS.md`.

## Out of memory

**Pattern:** `torch.OutOfMemoryError` or `CUDA out of memory`.

**Fix by phase:** before the first step (model build or checkpoint load) the weights and optimizer state do not fit,
so raise PP or EP; batch size and mt do not help. In a training step, lower `--max-tokens-per-gpu`, add recompute or
raise PP. Selective recompute can run out of memory where full recompute fits, since its default modules keep expert
and recurrent-state activations (levers.md, Selective recompute).

**Check which PP stage failed:** under 1F1B the first stage holds the most in-flight microbatches (up to PP), and the
last stage holds the output layer and the fp32 logits (`mt × vocab × 4 B / TP`). A failure on one stage points at an
uneven split (`--decoder-first/last-pipeline-num-layers`) or more recompute on that stage.

**Observed on:** slime and miles (Megatron-LM 1dcf0da, 37f91617) on H100, H200 and B200, 2026-05 to 2026-10.

## NCCL "unhandled cuda error" at init is often an out-of-memory

**Pattern:** `NCCL WARN Cuda failure 'out of memory'` or `NCCL WARN Failed to CUDA calloc <N> bytes` near NCCL init or
a first collective, followed by "unhandled cuda error" or a `DistBackendError`.

**Cause:** NCCL allocates its buffers after PyTorch has reserved nearly all GPU memory and reports the failed
allocation as a CUDA error, which reads like a network fault.

**Fix:** confirm with the WARN lines, then treat it as an out-of-memory (above). Recorded triggers: halving EP without
adding PP, and switching full recompute to selective.

**Observed on:** slime (Megatron-LM 1dcf0da) on H200 and B200, and Megatron-LM on H100, 2026-05 to 2026-09.

## Overlap flags on slime: `no_sync_func must be None`, `KeyError: Float16Module`

**Pattern:** with `--overlap-grad-reduce` (and `--overlap-param-gather`), at step 1 or at a checkpoint save:
`AssertionError: When overlap_grad_reduce is True, config.no_sync_func must be None ...`, or, on an image that patched
that assert, `KeyError: Float16Module(...)` from `disable_forward_pre_hook`.

**Cause:** slime calls its `train()` once per rollout, and `train()` is not re-entrant
(`slime/backends/megatron_utils/model.py`, slime a897e1f). Under grad-reduce overlap it asserts
`config.no_sync_func is None`, then sets it and never clears it, so the second call fails. With param-gather overlap
and the distributed optimizer it also disables the forward pre-hooks on entry, which the previous call already
disabled, and Megatron's DDP raises `KeyError` on the missing hook handle. `save()` does the same. Both failures are
deterministic. miles v0.1.0 has the same loop.

**Fix:** drop both flags unless the image carries a re-entrancy patch. Grep the image's `model.py` for the assert
message `config.no_sync_func must be None`; a patched `train()` (slime PR #1857, open as of 2026-11) keeps an
`if config.no_sync_func is None:` line, so grep the message, not the condition.

**Observed on:** slime 0.3.0 (Megatron-LM 1dcf0da) on H200 and B200, 2026-09; miles v0.1.0 on H200, 2026-10.

## NCCL or c10d timeout at init or step 0

**Pattern:** before the first `perf N:` line, `DistBackendError: ... wait timeout after 600000ms` (a c10d store wait
while a communicator is created) or a NCCL watchdog `... ran for 600000 milliseconds before timing out`.

**Cause:** a collective did not complete within slime's `--distributed-timeout-minutes` (default 10; `NCCL_TIMEOUT`
does not set it). Recorded causes: one rank still compiling or autotuning kernels when its peers reached a collective,
one rank near out-of-memory, an expert-parallel group that spans nodes, a pinned `NCCL_PROTO`, and a bad node.

**Fix:**
1. Strip any `NCCL_PROTO` pin (SKILL.md).
2. Diff the config against the last run that passed init: a group that moved across nodes (EP placement: levers.md
   §b), vpp, recompute, overlap flags.
3. If a first-step compile is the suspect (cold kernel cache, deep PP, a new GEMM backend), raise
   `--distributed-timeout-minutes` for that run; a true hang still times out.
4. To see where ranks wait, set `TORCH_NCCL_DUMP_ON_TIMEOUT=1` for a Flight Recorder dump on a NCCL watchdog timeout;
   a c10d store timeout writes none (`measurement.md` §10).
5. If the same config passed before, or the run goes silent in bring-up with no timeout, tear it down and retry once
   on other nodes as an infrastructure failure.

**Observed on:** compile skew past 600 s on Megatron-LM (H100, three runs) and slime (128×H100, 2026-07; cold runs
completed with a 30-minute timeout); EP-group and node hangs on slime (Megatron-LM 1dcf0da) on H100, H200 and B200,
2026-07 to 2026-09.
