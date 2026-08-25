# Hierarchical Transformer execution-horizon predictor

This path is opt-in. Legacy H1-H10 local-MLP sidecars, ordinary policy serving,
and the evaluator's existing default modes are unchanged. Do not start this
pipeline until a long-chunk checkpoint has passed its Fixed-H gate; the full
`{3,5,7,10,15,20}` pipeline requires an H20 checkpoint.

## Architecture and deployment contract

The sidecar reuses one VLA call and adds no image encoder or base-LLM forward.
It consumes the 20 final actions, 15 stride-2 coarse actions, state/controller
context, previous-chunk overlap, derivatives through jerk, gripper events, and
four learned attention queries over the already-computed prefix activations.
Coarse tokens are interpolated at physical times `0,2,4,...`; they are not
stretched uniformly across the final-action endpoint.

The recommended model is a two-layer, hidden-256, four-head Transformer. Its
deployment heads are:

- per-step hazard and monotone survival for `{3,5,7,10}`;
- success advantage of H15/H20 relative to H10, with learned uncertainty;
- elapsed-time advantage of H15/H20 relative to H10, with learned uncertainty;
- a conformal feature-distance OOD gate fitted on an independent calibration
  split.

Raw-H classification and absolute success/timeout/calls/steps heads remain
auxiliary. No PPO path is used in this version.

The selector defaults to H10. A long horizon is allowed only when its calibrated
95% success lower bound is at least `-0.01`, its elapsed-time upper bound is
below zero, and the OOD gate accepts the root. If neither long horizon passes,
the hazard curve may shorten H10 to H3/H5/H7. Missing calibration is a hard
error in closed-loop evaluation, and absence of a detected event never implies
H20.

## 1. Multi-seed counterfactual collection

Use a served H20 checkpoint. The prefix-token export is opt-in and is removed
from ordinary RPC responses. `1024` safely pads the expected prefix while the
stored mask excludes padding. A pilot should contain roughly 200 independent
roots; full collection should contain at least 2000. Every candidate receives
at least three paired continuation seeds.

```bash
python scripts/collect_execution_horizon_counterfactuals.py \
  --host 127.0.0.1 --port 8000 \
  --output-dir /root/autodl-tmp/acotvla/execution_horizon_h20/pilot \
  --candidate-horizons 3 5 7 10 15 20 \
  --reference-horizon 10 \
  --model-action-horizon 20 --model-coarse-horizon 15 \
  --branch-repeats 3 \
  --repeat-branch-horizons 3 5 7 10 15 20 \
  --prefix-token-count 1024 \
  --root-stride-calls 1 --max-roots-per-episode 1 \
  --num-trials-per-task 20 --max-tasks 10 --seed 7
```

Each schema-v2 record stores all H and seeds for one physical root: success and
timeout counts, trial count, calls/steps/elapsed means and unbiased sample
variances, fixed-size
raw trial outcomes, paired H10-success/long-H-failure counts, and the simulator
state. Paired short-H outcomes are also converted to per-step hazard-event and
at-risk counts, so survival training remains count-aware instead of treating a
single repeat as certain. Non-monotone short-H outcome patterns are excluded
from survival timing labels instead of being forced into a contradictory event
step. Splitting is episode-disjoint, so no H or continuation seed from an
episode can cross partitions. Branches are
interleaved by repeat and deterministically shuffled across H to avoid a
systematic elapsed-time bias from always evaluating H20 last. Full prefix
tokens are exported only on the recorded root request, never on continuation
RPCs, so token transport cannot create an artificial elapsed-time advantage
for long H.

## 2. Three quality-first bootstrap fits

Use the same explicit `split_seed=42` for every fit and vary only the training
seed. The 70/15/15 split is train/validation/calibration; validation selects the
checkpoint and calibration is never used by gradient updates or early stopping.
The Transformer path refuses schema-v1 labels, fewer than three trials for any
root/H, missing elapsed labels, missing prefix tokens, or an implicit split
seed.

Run the command below for seeds 7, 42, and 101 with distinct output directories:

```bash
python scripts/train_execution_horizon_predictor.py \
  --dataset /root/autodl-tmp/acotvla/execution_horizon_h20/full \
  --output-dir /root/autodl-tmp/acotvla/execution_horizon_h20/predictor_s7 \
  --seed 7 --split-seed 42 --bootstrap-episode-groups \
  --temporal-backbone transformer --temporal-layers 2 \
  --hidden-dim 256 --num-heads 4 --feed-forward-multiplier 4 \
  --reference-horizon 10 --coarse-stride 2 --final-stride 1 \
  --physical-action-dim 7 \
  --visual-num-queries 4 \
  --validation-fraction 0.15 --calibration-fraction 0.15 \
  --minimum-trials-per-candidate 3 \
  --early-stopping-patience-logs 10 --early-stopping-min-delta 1e-4
```

Model selection uses the complete fixed validation split. A checkpoint is
deployment-feasible only when long-H coverage is nonzero, the Wilson 95% upper
bound on false-long is controlled, the one-sided success-advantage lower bound
is at least `-1%`, the elapsed-advantage upper bound is below zero, and success
Brier has not regressed from the initial validation check. Among feasible
checkpoints it maximizes long-H coverage, breaking ties with validation loss.

## 3. Independent temperature, conformal, and OOD calibration

Calibrate each bootstrap fit independently. The command reads the exact
calibration episode IDs from `split_manifest.json`, fits shared temperature
scales, one-sided conformal residuals for each long H, and a robust feature
distance distribution using train features plus calibration roots.

```bash
python scripts/calibrate_execution_horizon_predictor.py \
  --dataset /root/autodl-tmp/acotvla/execution_horizon_h20/full \
  --predictor-dir /root/autodl-tmp/acotvla/execution_horizon_h20/predictor_s7 \
  --confidence-level 0.95 --ood-probability-threshold 0.95
```

Inspect `calibration_report.json`-style output next to `calibration.json` for
Brier, ECE, interval coverage, long-H coverage, false-long rate, and OOD
fallback rate. Bootstrap agreement is a quality diagnostic; closed-loop serving
uses one selected/calibrated sidecar until an explicitly validated ensemble
distillation stage is added.

For an untouched counterfactual dataset (or an explicitly named manifest
split), run the offline constrained-selector audit before any rollout:

```bash
python scripts/audit_hierarchical_execution_horizon.py \
  --dataset /root/autodl-tmp/acotvla/execution_horizon_h20/offline_test \
  --predictor-dir /root/autodl-tmp/acotvla/execution_horizon_h20/predictor_s7 \
  --bootstrap-samples 5000 \
  --output-json /root/autodl-tmp/acotvla/execution_horizon_h20/predictor_s7/offline_audit.json
```

The audit reports the selected-H distribution, fixed-H counterfactual
baselines, task-by-initial-state cluster-bootstrap intervals for
success/elapsed/calls advantage, rescues, regressions, false-long Wilson bound,
Brier, and OOD fallbacks. It is a model-selection diagnostic, not a substitute
for the closed-loop pilot.

## 4. Serving and closed-loop pilot

Pass the predictor output directory, not only its `params` child, so serving can
restore the exact Transformer architecture from `predictor_config.json`.

```bash
python scripts/serve_policy.py policy:checkpoint \
  --env LIBERO --port 8000 \
  --policy.config H20_CONFIG \
  --policy.dir /root/autodl-tmp/acotvla/checkpoints/H20_CONFIG/EXP/BEST_STEP \
  --policy.execution-horizon-predictor-params \
    /root/autodl-tmp/acotvla/execution_horizon_h20/predictor_s7

python scripts/eval_libero_execution_horizon.py \
  --host 127.0.0.1 --port 8000 \
  --output-dir /root/autodl-tmp/acotvla/execution_horizon_h20/pilot_eval \
  --modes hierarchical_transformer \
  --hierarchical-calibration-json \
    /root/autodl-tmp/acotvla/execution_horizon_h20/predictor_s7/calibration.json \
  --model-action-horizon 20 \
  --task-suite-name libero_10 --max-tasks 10 \
  --num-trials-per-task 20 --seed 7
```

The first closed-loop gate remains 10 tasks by 20 paired episodes. Formal
10x100 evaluation and any DAgger relabel round begin only after the pilot is
success-noninferior and lowers measured end-to-end policy/RPC time. Fixed H10,
H15, and H20 runs from the same H20 checkpoint remain the required baselines.
