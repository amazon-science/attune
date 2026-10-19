# Layer-1 rule lists

Layer 1 of the correctness check (`../layer1.py`, called by `../parity_verdict.py`) compares the resolved-argument
dumps of a baseline arm and a candidate arm. It fails the candidate when an argument that defines the objective moved
or a change hits a known stack defect, accepts placement, packing and tested swaps, and passes every other change as
unknown, for the step-0 check to judge and a human to review. These files decide which arguments fall where.
`frozen_args.py` finds this directory as `$AUTOTUNE_RULES_DIR`, else `rules/` beside it. The lists here are a starting
point and a format example; write the per-setup lists for your own model and stack.

## Argument classes

`numerics_list.yaml` puts every resolved argument of one Megatron-LM + slime build, of miles v0.1.0 and of miles
bfdcff0e (Muse Spark), in one class. For miles bfdcff0e only the argument classes ship: the metric readers do not yet
parse the perf lines it logs from `perf.py`.

| Class | A change of the argument |
|---|---|
| `objective` | fails layer 1 (data, sampling, seed, batch, run length, LR and weight-decay schedule, optimizer and trained parameters, loss terms, dropout, precision format, token dropping, routing semantics) |
| `numeric` | unknown: passes layer 1 and is listed for review (model shape, kernel numerics, uncertain names, ...) |
| `placement` | passes (parallel layout and scheduling) |
| `packing` | passes (microbatch packing and size) |
| `allowlisted` | passes to a value an allowlist lists once its component test passed; fails when that test is recorded `fail`; else unknown |
| `derived` | passes when it equals what the framework derives from the other changes (`padded_vocab_size`), fails when it does not; unknown when either dump does not print it as an integer |
| `ignored` | passes (logging, paths, run names) |

A name missing from the list is unknown, like a numeric one, so a new image's arguments are judged by the step-0 check
and listed for review until you class them. So is a name only one of the two dumps has (an objective-class one is
flagged to check by hand). A name listed in two classes, or an unknown top-level key, fails layer 1.

## Per-setup directory

`layer1/<name>/` holds the rules of one model and stack. It is the directory of the frozen-arguments file: point
`$AUTOTUNE_FROZEN_ARGS` at its `frozen_args.yaml`, and layer 1 reads `allowlist.yaml`, `ct_status.yaml` and
`defects.yaml` beside it. Without a frozen-arguments file, or without an allowlist beside it, only the image's swaps
(below) are admitted and every other allowlisted change is unknown. `layer1/example/` shows the four files:

- `allowlist.yaml`: `swaps: [{arg, values, component_test}]`. `values` is the list of values baseline and candidate
  may take (`none` means the flag is absent), or `any` to admit every value; `component_test` names the test in
  `ct_status.yaml` that must have passed (`none` admits the swap without a test).
- `ct_status.yaml`: `{tests: {<test>: pass | fail}}` (case and whitespace are ignored). A swap whose test is `fail`
  here or in the image's results fails layer 1; a missing test, or any other status, leaves the swap unknown.
- `frozen_args.yaml` (YAML; the example uses JSON syntax): arguments no candidate may change. `args` maps a resolved
  name to `{"eq": v}`, `{"in": [...]}` or `{"unchanged": true}` (optionally `default`, the value when the flag is
  absent); `products` freezes a product of arguments (for example tokens per GPU x CP); `env` freezes environment
  variables (`eq` or `absent`); `couplings` are conditional rules evaluated by `../couplings.py` (kinds: `require`,
  `vpp_middle_stages`, `vpp_off_at_cp`, `edp_dp_overlap`, `flex_shared_overlap`, `min_value`, `max_value`,
  `max_when_eq`, `a2a_overlap_requires`). `make_run_yaml.py` refuses a run that violates the file, and layer 1 fails
  a candidate whose resolved arguments violate it. A file with only `{"args": {}}` admits the setup's swaps and
  freezes nothing else.
- `defects.yaml` (optional): `defects: [{arg, values, when, reason, evidence}]`, known stack defects. A candidate
  whose resolved arguments put `arg` at one of `values` (`any`: any value other than the baseline's) with every
  `when: {other_arg: value}` condition holding, when the baseline's do not, fails layer 1 whichever of those
  arguments changed, and the verdict prints the reason and evidence. Each entry needs a reason and evidence that the
  state is invalid on this stack.

Each of these files may hold only its own top-level keys (`version` and `swaps`, `tests` or `defects`); a path that
exists but is not a readable file, or an unknown key, fails layer 1. Values are compared after the dump's
normalization (case, whitespace, YAML booleans and null).

The example's values are illustrative, not validated: run your own component tests before listing a swap.

## Per-image directory

A component test that checks a swap independent of model shape (an optimizer step, a kernel against its reference)
holds for every model trained on the image it ran on. `images/<digest prefix>/` (12 to 64 hex characters that begin
the image's sha256 digest) holds an `allowlist.yaml` and a `ct_status.yaml` in the formats above for such tests, and a
`defects.yaml` for defects of the image's own stack. Layer 1 reads the directory of the image both arms ran:
`parity_verdict.py` takes it from the pair's `images.json` when both arms name the same image pinned by digest, and
`layer1.py` from `--image`. A swap listed there passes on every setup that runs the image. A setup entry for the same
argument replaces the image's and passes only on the setup's own `ct_status.yaml`. A swap fails when the image or the
setup marks its test `fail`, whichever allowlist lists it; an image entry is unknown when the setup marks its test
anything else but `pass`. The defects of the setup and the image both apply. More than one matching directory fails
layer 1.
