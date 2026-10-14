# Layer-1 rule lists

Layer 1 of the correctness check (`../layer1.py`, called by `../parity_verdict.py`) compares the resolved-argument
dumps of a baseline arm and a candidate arm and fails the candidate when an argument that changes the math moved.
These files decide which arguments those are. `frozen_args.py` finds this directory as `$AUTOTUNE_RULES_DIR`, else
`rules/` beside it. The lists here are a starting point and a format example; write the per-setup lists for your own
model and stack.

## Argument classes

`numerics_list.yaml` puts every resolved argument of one Megatron-LM + slime build, and of miles v0.1.0, in one class:

| Class | A change of the argument |
|---|---|
| `numeric` | fails layer 1 (dtype, loss, optimizer, data, seed, batch, model shape, routing, ...) |
| `placement` | passes (parallel layout and scheduling) |
| `packing` | passes (microbatch packing and size) |
| `allowlisted` | passes only to a value the setup's allowlist lists, after its component test passed |
| `derived` | passes only when it equals what the framework derives from accepted changes (`padded_vocab_size`) |
| `ignored` | passes (logging, paths, run names) |

Names missing from the list are numeric, so a new image fails closed until you class its new arguments.

## Per-setup directory

`layer1/<name>/` holds the rules of one model and stack. It is the directory of the frozen-arguments file: point
`$AUTOTUNE_FROZEN_ARGS` at its `frozen_args.yaml`, and layer 1 reads `allowlist.yaml` and `ct_status.yaml` beside it.
Without a frozen-arguments file, or without an allowlist beside it, only the image's swaps (below) are admitted and
every other allowlisted argument fails closed. `layer1/example/` shows the three files:

- `allowlist.yaml`: `swaps: [{arg, values, component_test}]`. `values` are the values baseline and candidate may take
  (lower case; `none` means the flag is absent, `any` admits every value) and `component_test` names the test in
  `ct_status.yaml` that must have passed (`none` admits the swap without a test).
- `ct_status.yaml`: `{tests: {<test>: pass | fail}}`; a missing test is not passed.
- `frozen_args.yaml` (YAML; the example uses JSON syntax): arguments no candidate may change. `args` maps a resolved
  name to `{"eq": v}`, `{"in": [...]}` or `{"unchanged": true}` (optionally `default`, the value when the flag is
  absent); `products` freezes a product of arguments (for example tokens per GPU x CP); `env` freezes environment
  variables (`eq` or `absent`); `couplings` are conditional rules evaluated by `../couplings.py` (kinds: `require`,
  `vpp_middle_stages`, `vpp_off_at_cp`, `edp_dp_overlap`, `flex_shared_overlap`, `min_value`, `max_value`,
  `max_when_eq`, `a2a_overlap_requires`). `make_run_yaml.py` refuses a run that violates the file, and layer 1 fails
  a candidate whose resolved arguments violate it. A file with only `{"args": {}}` admits the setup's swaps and
  freezes nothing else.

The example's values are illustrative, not validated: run your own component tests before listing a swap.

## Per-image directory

A component test that checks a swap independent of model shape (an optimizer step, a kernel against its reference)
holds for every model trained on the image it ran on. `images/<digest prefix>/` (12 to 64 hex characters that begin
the image's sha256 digest) holds an `allowlist.yaml` and a `ct_status.yaml` in the formats above for such tests.
Layer 1 reads the directory of the image both arms ran: `parity_verdict.py` takes it from the pair's `images.json`
when both arms name the same image pinned by digest, and `layer1.py` from `--image`. A swap listed there passes on
every setup that runs the image. A setup entry for the same argument replaces the image's and is judged by the
setup's own `ct_status.yaml`; an image entry fails when the setup marks its test anything but `pass`. More than one
matching directory fails layer 1.
