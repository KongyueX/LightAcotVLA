# IR-ACoT aggressive MVP

This pilot tests an algorithmic speedup rather than another execution-horizon
selector. The original 10-step EAR and 10-step final flow are retained as the
teacher. The student reuses the same two branches, but learns the teacher's
clean endpoint directly at flow time `t=1`:

```text
coarse_student = coarse_noise - v_coarse(coarse_noise, t=1)
action_student = action_noise - v_final(action_noise, t=1, EAR, IAR)
```

Therefore deployment performs one EAR forward and one final-action forward.
`H_exe` remains fixed at 9 in the first closed-loop comparison; it is not part
of this method.

IR-lite adds a causal response constraint. For an EAR intervention `do(c -> c')`
with the observation, IAR, and final-action noise held fixed, it aligns

```text
student_action(c') - student_action(c)
```

with the same delta from the 10-step teacher. B6-lite is the matched control
with only clean endpoint losses. B6-lite and IR-lite have identical deployed
architectures and inference call counts.

## Frozen/trainable contract

Always frozen:

- vision encoder and base VLM;
- implicit action reasoner (IAR);
- EAR/IAR interaction and fusion modules.

Trainable in the coarse stage:

- coarse 300M expert LLM branch;
- `coarse_action_in_proj`, coarse time MLP, and `coarse_action_out_proj`.

Trainable in the final stage:

- final 300M expert LLM branch;
- `action_in_proj`, final time MLP, and `action_out_proj`.

The trainer saves only these selected parameters as an Orbax delta sidecar.
The 50999 teacher checkpoint is never overwritten.

## Recommended quick gate on the current server

The commands below target the current RTX PRO 6000 server layout. Run them
after pulling the code into `/root/ACoT-VLA`.

```bash
cd /root/ACoT-VLA
export LD_LIBRARY_PATH=/root/autodl-tmp/acotvla/ffmpeg-7.1.1/lib:${LD_LIBRARY_PATH:-}
PY=/root/autodl-tmp/acotvla/envs/acotvla-py311/bin/python
CKPT=/root/autodl-tmp/acotvla/checkpoints/acot_libero_action_cot_explicit_implicit_co_fusion/acot_libero_long_run1/50999
PILOT=/root/autodl-tmp/acotvla/ir_acot_pilot
```

First run a 200-state causal audit with all interventions:

```bash
$PY scripts/audit_ear_interventions.py \
  --config-name acot_libero_action_cot_explicit_implicit_co_fusion \
  --checkpoint-dir "$CKPT" \
  --output-dir "$PILOT/audit200" \
  --max-items 200 \
  --selection random \
  --seed 7
```

Do not start IR training when `audit_summary.json` reports
`ear_causal_audit_pass=false`. This gate only establishes that final actions
respond to semantic EAR changes above the null rerun; it is not a success-rate
result.

For an aggressive 2k-record pilot, export clean 10+10-step teacher endpoints
with one policy call per frame:

```bash
$PY scripts/audit_ear_interventions.py \
  --config-name acot_libero_action_cot_explicit_implicit_co_fusion \
  --checkpoint-dir "$CKPT" \
  --output-dir "$PILOT/coarse_teacher_2k" \
  --max-items 2000 \
  --selection random \
  --seed 7 \
  --clean-only
```

Train only the one-step coarse branch:

```bash
$PY scripts/train_acot_endpoint_distillation.py \
  --dataset "$PILOT/coarse_teacher_2k" \
  --config-name acot_libero_action_cot_explicit_implicit_co_fusion \
  --checkpoint-dir "$CKPT" \
  --output-dir "$PILOT/coarse_b6" \
  --stage coarse \
  --variant b6 \
  --train-steps 300 \
  --checkpoint-interval 300 \
  --batch-size 8
```

Generate deployable student EAR trajectories. The final output of this
intermediate pass is intentionally ignored:

```bash
$PY scripts/audit_ear_interventions.py \
  --config-name acot_libero_action_cot_explicit_implicit_co_fusion \
  --checkpoint-dir "$CKPT" \
  --endpoint-student-params "$PILOT/coarse_b6/final/params" \
  --output-dir "$PILOT/student_coarse_2k" \
  --max-items 2000 \
  --selection random \
  --seed 7 \
  --num-steps 1 \
  --action-cot-denoising-steps 1 \
  --clean-only
```

Relabel final actions with the 10-step teacher conditioned on those student
EARs. One semantic intervention is cycled per record, reducing this stage from
five extra teacher calls to one:

```bash
$PY scripts/audit_ear_interventions.py \
  --config-name acot_libero_action_cot_explicit_implicit_co_fusion \
  --checkpoint-dir "$CKPT" \
  --coarse-overrides-from "$PILOT/student_coarse_2k" \
  --output-dir "$PILOT/final_teacher_2k" \
  --max-items 2000 \
  --selection random \
  --seed 7 \
  --interventions translate rotate gripper_shift swap \
  --one-intervention-per-record
```

Train the matched final heads from the same labels and same coarse sidecar:

```bash
$PY scripts/train_acot_endpoint_distillation.py \
  --dataset "$PILOT/final_teacher_2k" \
  --config-name acot_libero_action_cot_explicit_implicit_co_fusion \
  --checkpoint-dir "$CKPT" \
  --resume-sidecar-params "$PILOT/coarse_b6/final/params" \
  --output-dir "$PILOT/final_b6" \
  --stage final \
  --variant b6 \
  --train-steps 300 \
  --checkpoint-interval 300 \
  --batch-size 8

$PY scripts/train_acot_endpoint_distillation.py \
  --dataset "$PILOT/final_teacher_2k" \
  --config-name acot_libero_action_cot_explicit_implicit_co_fusion \
  --checkpoint-dir "$CKPT" \
  --resume-sidecar-params "$PILOT/coarse_b6/final/params" \
  --causal-audit-summary "$PILOT/audit200/audit_summary.json" \
  --output-dir "$PILOT/final_ir" \
  --stage final \
  --variant ir \
  --ir-loss-weight 0.5 \
  --train-steps 300 \
  --checkpoint-interval 300 \
  --batch-size 8
```

## Serving and comparison

A complete final sidecar automatically selects one EAR and one final denoising
step. Explicit `sample_kwargs` can still override those values for ablations.

```bash
$PY scripts/serve_policy.py \
  --env LIBERO \
  --port 8000 \
  policy:checkpoint \
  --policy.config acot_libero_action_cot_explicit_implicit_co_fusion \
  --policy.dir "$CKPT" \
  --policy.acot-endpoint-student-params "$PILOT/final_ir/final/params"
```

The first closed-loop table should use the unchanged formal settings:
checkpoint 50999, LIBERO-10, seed 7, `num_steps_wait=10`,
`profile_policy_timing=True`, and fixed `H_exe=9`.

Required rows:

1. original ACoT-VLA, EAR=10 and final=10;
2. untrained one-step ablation, EAR=1 and final=1;
3. B6-lite sidecar, EAR=1 and final=1;
4. IR-lite sidecar, EAR=1 and final=1.

Report task success, policy seconds/episode, calls/episode, coarse/final stage
milliseconds, and easy-task retention. Advance beyond the 2k pilot only if
B6-lite recovers useful success and IR-lite improves the success/latency
Pareto point over B6-lite.

## Storage rule

Do not save full train states for this pilot. A final delta sidecar contains
only the EAR/final student branches. With `train_steps=300` and
`checkpoint_interval=300`, each run writes only the final sidecar.
