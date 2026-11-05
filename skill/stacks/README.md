# Stacks

Each training stack has its own folder, selected with `ATTUNE_STACK` (default `megatron`):

- `levers.md`: the lever catalog. What to change for speed on this stack, what each change does, when it helps or
  hurts, and the scope (stack version, model class, GPUs, workload) and evidence of every entry.
- `failure_triage.md`: crash, hang and out-of-memory signatures seen on this stack. Each entry names the image or
  framework commit and the hardware it was observed on. Signatures change with the image, the hardware and the
  nodes, so an entry is a hint to check on the current image, not a rule.
- `correctness/rules/`: the layer-1 argument classes of the correctness check (which arguments change the training
  objective), as `correctness/rules/README.md` describes.

Campaigns read these files and never edit them: the launcher gives each campaign a read-only copy and records its
hashes. Lessons a campaign learns go to its own `LESSONS.md`, and enter a stack's `levers.md` or
`failure_triage.md` only through curation (`harness/CURATION.md`).

| Stack | Frameworks | Status |
|---|---|---|
| `megatron` | Megatron-LM, with slime or miles | supported |
| `torchtitan` | TorchTitan | catalog only; the launch, metrics and correctness adapter is not written yet |
