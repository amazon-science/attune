# print_topology.md — verifying Megatron process-group placement

Use this when you deploy a new topology (a new TP/PP/EP/CP/ETP combination), move to new hardware, or suspect a
parallelism group is not where you expected (which nodes, which GPUs, intranode vs internode). Two read-only sources
answer it, and they check each other:

| Source | What it shows | How |
|---|---|---|
| **A. The rank layout in the image's Megatron source** | the global ranks of every named group (`tp`, `cp`, `dp`, `pp`, `ep`, `etp`, `edp`) | compute it offline from `parallel_state.py` |
| **B. NCCL INFO fingerprints** | per communicator: ranks, nodes, ranks on this node | read them from the run's `train.log` |

Do not add a probe to the pods for this. In-pod hooks load only through the hook chain (`scripts/hook_chain.py`),
which checks each hook's sha256 and logs it per process; a probe in another `sitecustomize.py` would skip that check.

---

## 1. Source A — the rank layout from source

`megatron/core/parallel_state.py` builds every process group with `RankGenerator`. `initialize_model_parallel`
creates two generators with the same `order` string:

- the decoder (attention) generator, sizes `tp` = TP, `cp` = CP, `dp` = DP, `pp` = PP (its `ep` is 1);
- the expert generator, sizes `tp` = ETP, `ep` = EP, `dp` = EDP = world / (ETP × EP × PP), `pp` = PP (its `cp` is 1).

Earlier tokens in `order` vary fastest. `megatron/training/initialize.py` passes `order='tp-cp-ep-dp-pp'`, or
`'tp-cp-ep-pp-dp'` with `--use-tp-pp-dp-mapping` (Megatron-LM commit 1dcf0daf; read your image's copy). Under the
default, attention ranks fold as `[TP][CP][DP][PP]` and expert ranks as `[ETP][EP][EDP][PP]`, which is why the EP
group is contiguous and intranode exactly when `ETP × EP ≤ GPUs per node` (levers.md §b).

`RankGenerator` and `generate_masked_orthogonal_rank_groups` are plain Python. Copy both from the image's
`parallel_state.py` into a scratch script in your campaign directory and print the groups for your sizes. The tokens
are the ones `parallel_state.py` itself passes to `get_ranks`: `tp`, `cp`, `dp`, `pp` on the decoder generator;
`ep`, `tp` (ETP) and `dp` (EDP) on the expert generator.

```python
# placement.py — RankGenerator and generate_masked_orthogonal_rank_groups pasted above, from the image's source
world, tp, pp, cp, ep, etp, gpus_per_node = 32, 2, 4, 1, 8, 1, 8
dp = world // (tp * pp * cp)
edp = world // (etp * ep * pp)
order = "tp-cp-ep-dp-pp"
dec = RankGenerator(tp=tp, ep=1, dp=dp, pp=pp, cp=cp, order=order)
exp = RankGenerator(tp=etp, ep=ep, dp=edp, pp=pp, cp=1, order=order)
for name, gen, token in [("tp", dec, "tp"), ("cp", dec, "cp"), ("dp", dec, "dp"), ("pp", dec, "pp"),
                         ("ep", exp, "ep"), ("etp", exp, "tp"), ("edp", exp, "dp")]:
    groups = gen.get_ranks(token)
    span = max(len({r // gpus_per_node for r in g}) for g in groups)
    print(f"{name:4s} size {len(groups[0]):2d}  first group {groups[0]}  max nodes per group {span}")
```

For `TP=2 PP=4 EP=8 DP=4 ETP=1 CP=1` on 4×8-GPU nodes it prints:

```
tp   size  2  first group [0, 1]  max nodes per group 1
cp   size  1  first group [0]  max nodes per group 1
dp   size  4  first group [0, 2, 4, 6]  max nodes per group 1
pp   size  4  first group [0, 8, 16, 24]  max nodes per group 4
ep   size  8  first group [0, 1, 2, 3, 4, 5, 6, 7]  max nodes per group 1
etp  size  1  first group [0]  max nodes per group 1
edp  size  1  first group [0]  max nodes per group 1
```

TP, DP and EP stay inside a node; PP crosses all four, as expected. Rank `r` sits on node `r // GPUs-per-node` when
Ray places ranks in node order; source B confirms that for the run.

Megatron also creates groups outside the named accessors (for example the embedding group of the first and last
pipeline stages, `default_embedding_ranks`), and slime wraps process groups of its own. Read `parallel_state.py`
and slime's Megatron backend in the image if a communicator in source B has no match here.

---

## 2. Source B — NCCL INFO fingerprints

When NCCL creates a communicator, each rank prints one line in this format:

```
<hostname>:<pid>:<tid> [<local_gpu>] NCCL INFO comm 0x<id> rank <R> nRanks <N> nNodes <M> localRanks <L> localRank <k> MNNVL <n>
```

| Field | Meaning |
|---|---|
| `comm 0x<id>` | Unique hex ID of this communicator. All ranks in the group print the same ID. |
| `rank <R>` | This rank's position **within the group** (0-indexed, NOT the global rank). |
| `nRanks <N>` | Total ranks in the group. |
| `nNodes <M>` | Number of **distinct physical nodes** NCCL detected. `1` → intranode, `>1` → at least one cross-node link. |
| `localRanks <L>` | How many ranks of this group are co-located on the **current** node. |

**The intranode vs internode verdict comes directly from `nNodes`.** It is a hardware-level signal, not an inference.

### How to enable

Set `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,GRAPH` on a short diagnostic launch. Both are structural env vars
(`scripts/env_rules.py`), so `make_run_yaml.py --env NCCL_DEBUG=INFO --env NCCL_DEBUG_SUBSYS=INIT,GRAPH` sets them on
every pod. The communicators form during init and the first step, so a short unranked diagnostic run carries the
data.

**Do not enable** `NET` or `ENV` in `NCCL_DEBUG_SUBSYS` on long runs: their log volume can OOM an actor or time out
Ray's keepalive thread. Do not set `TORCH_DISTRIBUTED_DEBUG=DETAIL` together with heavy NCCL debug for the same
reason.

### Extract tuples

```bash
kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" exec "$AUTOTUNE_LOG_POD" -- \
  grep "NCCL INFO comm 0x" <path to train.log> \
  | grep -oE "nRanks [0-9]+ nNodes [0-9]+ localRanks [0-9]+" | sort -u | sort -n
```

You get a table like this one (TP=2 PP=4 EP=8 on 4×8-GPU nodes):

```
nRanks  nNodes  localRanks
  2       1        2        ← a 2-rank intranode group → TP pair
  4       1        4        ← 4-rank intranode          → DP subgroup
  4       4        1        ← 4-rank, 1 per node        → PP (cross-node)
  8       1        8        ← 8-rank whole node          → EP group (intranode)
 32       4        8        ← world group
```

Ray's driver log dedupes identical lines with `[repeated N across cluster]`, so you see the *shapes* of the groups,
not every rank list. Source A gives the rank lists.

---

## 3. Cross-checks

- Every group from source A should have a fingerprint of the same size in source B, with `nNodes=1` exactly when
  source A puts the whole group on one node.
- A fingerprint with no matching named group is a group created outside the named accessors (§1, last paragraph).
- If the two disagree, trust source B for the run (it is what NCCL built) and re-read the `order` and sizes the
  image actually used: the resolved-argument dump in `train.log` lists them.

| Question | Best source |
|---|---|
| "Is my EP group intranode?" | A (`ep` group within one node) or B (`nNodes=1` on the EP-sized communicator) |
| "What are the global ranks in the DP group?" | A |
| "Is the PP group 1 rank per node as expected?" | B (`nRanks=PP nNodes=PP localRanks=1`) |
| "Did my new topology break placement?" | A before launch, B on the first launch |
