# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the skill-script tests: a minimal two-group RayJob and synthetic train.log text."""
import json

BASE_RUN = "alice-tune-base"

BASE_YAML = """apiVersion: v1
kind: ConfigMap
metadata:
  name: alice-tune-base-scripts
  namespace: training
data:
  run.sh: |
    export PYTHONPATH=/root/slime:$PYTHONPATH
    echo start
---
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: alice-tune-base
  namespace: training
  labels:
    app: alice-tune-base
spec:
  entrypoint: |
    bash -c '
    python3 /root/slime/train.py \\
      --tensor-model-parallel-size 2 \\
      --context-parallel-size 2 \\
      --global-batch-size 96 \\
      --rollout-batch-size 96 \\
      --num-epoch 1 \\
      --prompt-data /shared/data/orig.parquet \\
      --moe-aux-loss-coeff 0.02 \\
      --max-tokens-per-gpu 3072 \\
      2>&1 | tee /shared/experiments/alice-tune-base/logs/train.log
    '
  runtimeEnvYAML: |
    env_vars:
      PYTHONPATH: "/root/slime"
  rayClusterSpec:
    headGroupSpec:
      template:
        spec:
          containers:
            - name: ray-head
              image: registry.example.com/img@sha256:abc
              env:
                - {name: PYTHONPATH, value: "/root/slime"}
              resources:
                limits: {nvidia.com/gpu: 8}
              volumeMounts:
                - {name: scripts, mountPath: /scripts}
          volumes:
            - {name: scripts, configMap: {name: alice-tune-base-scripts}}
    workerGroupSpecs:
      - replicas: 2
        groupName: gpu-worker
        template:
          spec:
            containers:
              - name: ray-worker
                image: registry.example.com/img@sha256:abc
                env:
                  - {name: PYTHONPATH, value: "/root/slime"}
                resources:
                  limits: {nvidia.com/gpu: 8}
"""


def stamp(step, step_time, t0="2026-10-01 00:00:00"):
    """The log clock of step `step`'s perf line when every step takes step_time s (read_metrics checks it)."""
    import datetime as dt
    return "[%s]" % (dt.datetime.fromisoformat(t0) + dt.timedelta(seconds=step * step_time)).strftime("%Y-%m-%d %H:%M:%S")


def train_log(loss0=2.5, gn0=1.0, steps=20, args=None, counts=True):
    """Synthetic slime train.log: resolved-argument dump, per-step loss/grad-norm, probe counts and perf lines."""
    a = {"moe_router_load_balancing_type": "aux_loss", "moe_aux_loss_coeff": "0.02", "context_parallel_size": "2",
         "tensor_model_parallel_size": "2", "global_batch_size": "96", "max_tokens_per_gpu": "3072", "micro_batch_size": "2",
         "virtual_pipeline_model_parallel_size": "None"}
    a.update(args or {})
    lines = ["  %s %s %s" % (k, "." * 8, v) for k, v in a.items()]
    for t in range(steps):
        lines.append("%s train.py:1 - step %d: {'train/loss': %r, 'train/grad_norm': %r}"
                     % (stamp(t, 10.0), t, loss0 - 0.01 * t, gn0))
        if counts:
            lines.append("PARITY_COUNTS rollout %d step %d: %s" % (t, t, json.dumps(
                {"rows": 96, "tokens": 96 * 3072, "loss_tokens": 96 * 3000, "microbatches": 12, "basis": "chunk0"})))
        lines.append("%s train_metric_utils.py:1 - perf %d: {'perf/actor_train_tok_per_s': 1000.0, "
                     "'perf/actor_train_time': 9.0, 'perf/step_time': 10.0}" % (stamp(t, 10.0), t))
    return "\n".join(lines) + "\n"
