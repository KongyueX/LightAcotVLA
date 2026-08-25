# H25 long-action-chunk training

`acot_libero_long_chunk_h25` is strictly opt-in. It warm-starts the 25-action
student and its frozen prefix-retention teacher from the validation-best H20
step4000 checkpoint. The first 20 final actions are anchored with paired
observation, flow time, and noise; legacy H10/H15/H20 configs and evaluation
defaults remain unchanged.

Run the basic restore/shape/loss/sampling smoke before training:

```bash
python scripts/smoke_acot_long_chunk.py \
  --config-name acot_libero_long_chunk_h25 \
  --checkpoint-params \
    /root/autodl-tmp/acotvla/checkpoints/acot_libero_long_chunk_h20/h20_from_h15_lr1e5_seed42_split42_6e35bad/4000/params \
  --batch-size 1 --inference-steps 1 \
  --output-json /root/autodl-tmp/acotvla/long_chunk_h25/smoke/summary.json
```

The first quality run uses the configured `1e-5` peak learning rate, fixed
validation split42, validation-best selection, early stopping, and at most two
retained checkpoints:

```bash
python scripts/train.py acot_libero_long_chunk_h25 \
  --exp-name h25_from_h20_lr1e5_seed42_split42 \
  --seed 42 --overwrite
```

Do not collect H25 counterfactual labels until the selected checkpoint restores,
returns `[batch,25,32]` final actions, has finite long-flow/prefix-retention
loss, and completes a minimal Fixed H25 closed-loop smoke.
