# H15 long-chunk pilot

This batch is strictly opt-in. Existing H10 training configs, the legacy
`fixed_h9` evaluator mode, and the evaluator's default mode list are unchanged.
No predictor is trained or loaded in this phase.

## 1. Data/checkpoint/forward smoke

Run on the training server before starting a sweep:

```bash
python scripts/smoke_acot_long_chunk.py \
  --config-name acot_libero_long_chunk_h15 \
  --output-json /root/autodl-tmp/acotvla/long_chunk_h15/smoke.json
```

The command must report `status: passed`, final actions `[1,15,32]`, coarse
actions `[1,15,32]`, a restored 50999 anchor, finite long-flow/retention losses,
and a sampling-forward output `[1,15,32]`. It performs zero optimizer updates.

## 2. Small learning-rate scan

The config defaults to peak LR `3e-6`. Run a small three-point scan by changing
only the peak/decay LR and experiment name:

```bash
bash scripts/train.sh acot_libero_long_chunk_h15 h15_lr1e6 \
  --lr-schedule.peak-lr=1e-6 --lr-schedule.decay-lr=1e-7
bash scripts/train.sh acot_libero_long_chunk_h15 h15_lr3e6 \
  --lr-schedule.peak-lr=3e-6 --lr-schedule.decay-lr=3e-7
bash scripts/train.sh acot_libero_long_chunk_h15 h15_lr1e5 \
  --lr-schedule.peak-lr=1e-5 --lr-schedule.decay-lr=1e-6
```

Training uses an episode-disjoint 90/10 split. Every 250 updates it evaluates
32 fixed validation batches with deterministic flow noise, scores the EMA
parameters that are actually written for inference, saves every new best, and
stops after six non-improving checks. Read `validation_best.json` under each
experiment directory and serve its `best_step`; do not select the last step by
default. H15 runs retain at most two full checkpoints (the latest validation
best and, when different, the final checkpoint) to bound disk usage; legacy
configs keep their existing unbounded checkpoint policy.

The held-out episode assignment is fixed with `validation_split_seed=42`,
independently of the model/data-order `--seed`. First compare the three learning
rates with the same model seed. After choosing the best LR, rerun that LR with
at least three model seeds (for example 7, 42, and 101); these replications see
exactly the same validation episodes. Reject a checkpoint that wins only on
validation loss but degrades the first-10-step retention metrics or has an
unstable multi-seed Fixed-H result.

The trainable scope is the final 300M expert, its input/time/output projections,
and final-token reasoning/fusion modules. Vision, the base LLM, coarse expert,
and implicit prefix extractor remain frozen. The loss is:

```text
long_flow_loss + lambda10 * 0.5 * (prefix_velocity_mse + prefix_action_mse)
```

The frozen 50999 teacher and H15 student share the same augmented observation,
flow time, and first ten noise tokens.

## 3. Fixed-H evaluation

Serve the selected H15 checkpoint:

```bash
python scripts/serve_policy.py policy:checkpoint \
  --env LIBERO --port 8000 \
  --policy.config acot_libero_long_chunk_h15 \
  --policy.dir /root/autodl-tmp/acotvla/checkpoints/acot_libero_long_chunk_h15/EXP/BEST_STEP
```

Run paired H10 and H15 evaluations in separate output directories while keeping
task IDs, initial-state IDs, seed, profiling, and denoising settings identical:

```bash
python scripts/eval_libero_execution_horizon.py \
  --host 127.0.0.1 --port 8000 \
  --output-dir /root/autodl-tmp/acotvla/long_chunk_h15/fixed_h10 \
  --modes fixed_h --fixed-horizon 10 \
  --model-action-horizon 15 \
  --task-suite-name libero_10 --max-tasks 10 \
  --num-trials-per-task 20 --seed 7 \
  --action-cot-denoising-steps 10

python scripts/eval_libero_execution_horizon.py \
  --host 127.0.0.1 --port 8000 \
  --output-dir /root/autodl-tmp/acotvla/long_chunk_h15/fixed_h15 \
  --modes fixed_h --fixed-horizon 15 \
  --model-action-horizon 15 \
  --task-suite-name libero_10 --max-tasks 10 \
  --num-trials-per-task 20 --seed 7 \
  --action-cot-denoising-steps 10
```

The generic `fixed_h` mode fails instead of silently clipping when the served
checkpoint returns fewer actions than requested. Episode-tail truncation remains
allowed. The H15 gate still requires the old 50999 Fixed-H10 baseline and the new
H15 checkpoint's Fixed-H10/Fixed-H15 runs; this code batch contains no success or
latency result.
