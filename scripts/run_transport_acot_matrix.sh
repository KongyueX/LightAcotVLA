#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${TRANSPORT_PYTHON:-python}
dataset=${TRANSPORT_DATASET:?Set TRANSPORT_DATASET to the multirate window directory.}
output_root=${TRANSPORT_OUTPUT_ROOT:?Set TRANSPORT_OUTPUT_ROOT to a new experiment directory.}
train_steps=${TRANSPORT_TRAIN_STEPS:-1500}
seed=${TRANSPORT_SEED:-7}

mkdir -p "${output_root}/logs"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-"${output_root}/jax-cache"}

common_args=(
  --dataset "${dataset}"
  --selection-mode action
  --seed "${seed}"
  --train-steps "${train_steps}"
  --batch-size 128
  --eval-batch-size 512
  --log-interval 100
  --early-stopping-patience-logs 8
  --profile-warmup 50
  --profile-iterations 2000
)

run_case() {
  local name=$1
  shift
  local case_dir="${output_root}/${name}"
  local log_path="${output_root}/logs/${name}.log"
  if [[ -f "${case_dir}/summary.json" ]]; then
    echo "Skipping completed case ${name}: ${case_dir}/summary.json"
    return
  fi
  echo "Starting ${name} at $(date --iso-8601=seconds)"
  "${python_bin}" "${repo_root}/scripts/train_transport_acot.py" \
    "${common_args[@]}" \
    --output-dir "${case_dir}" \
    "$@" 2>&1 | tee "${log_path}"
  echo "Finished ${name} at $(date --iso-8601=seconds)"
}

run_case phase \
  --correction-mode phase

run_case direct \
  --correction-mode direct

run_case plan_base \
  --correction-mode plan

run_case plan_event_low \
  --correction-mode plan \
  --velocity-loss-weight 0.25 \
  --gripper-state-loss-weight 0.01 \
  --action-gripper-loss-weight 0.01 \
  --event-loss-weight 0.01 \
  --geometry-regularization-weight 0.001

run_case plan_event_strong \
  --correction-mode plan \
  --velocity-loss-weight 0.25 \
  --gripper-state-loss-weight 0.05 \
  --action-gripper-loss-weight 0.05 \
  --event-loss-weight 0.05 \
  --geometry-regularization-weight 0.001

touch "${output_root}/matrix.complete"
