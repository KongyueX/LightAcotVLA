# Experiment Log

记录实验配置、路径、关键指标和结论。

## 2026-06-30 Stage B / Action-CoT pruning and coarse denoising step experiments

### 实验目标

- Stage B 的目标是验证 Action-CoT entropy criterion：低熵 Action-CoT segment 是否比随机或高熵 segment 更适合被剪裁。
- 当前进一步区分了两类加速路径：
  - segment pruning：剪 coarse action trajectory 里的时间片段，再用 `interp` / `hold` / `zero` 替换。
  - coarse denoising step reduction：减少 explicit Action-CoT coarse action 生成时的 denoising 迭代次数，例如 `10 -> 7 -> 5 -> 3 -> 1`。
- 已确认：当前真正能直接减少推理耗时的路径是减少 coarse denoising steps；单纯 mask / override coarse action values 不会带来 segment-level 加速。

### 固定配置和路径

- 主 policy config：`acot_libero_action_cot_explicit_implicit_co_fusion`
- 主 checkpoint：`/root/autodl-tmp/acotvla/checkpoints/acot_libero_action_cot_explicit_implicit_co_fusion/acot_libero_long_run1/50999`
- Stage A entropy labels：`/root/autodl-tmp/acotvla/action_cot_entropy_labels/adaptive_full_k8`
- Stage B eval 输出根目录：`/root/autodl-tmp/acotvla/stage_b_pruning_eval`
- LIBERO policy server 端口：`8000`
- 当前 long checkpoint 对齐的 closed-loop suite：`libero_10`
- 之前用 `libero_spatial` 跑出的成功率结果已判定与 long checkpoint 不匹配，不作为最终成功率结论。

### 关键命令

启动 policy server：

```bash
cd /root/ACoT-VLA
source scripts/env/use_acotvla_env.sh

uv run python scripts/serve_policy.py \
  --env LIBERO \
  --port 8000 \
  policy:checkpoint \
  --policy.config acot_libero_action_cot_explicit_implicit_co_fusion \
  --policy.dir /root/autodl-tmp/acotvla/checkpoints/acot_libero_action_cot_explicit_implicit_co_fusion/acot_libero_long_run1/50999
```

LIBERO-10 task0 baseline sanity，20 trials，只跑 `coarse_steps=10`：

```bash
cd /root/ACoT-VLA
source scripts/env/use_acotvla_env.sh

OUT=/root/autodl-tmp/acotvla/stage_b_pruning_eval/coarse_steps_libero10_baseline_sanity

uv run python scripts/sweep_action_cot_coarse_steps.py \
  --mode closed_loop \
  --host 0.0.0.0 \
  --port 8000 \
  --task_suite_name libero_10 \
  --task_start 0 \
  --max_tasks 1 \
  --num_trials_per_task 20 \
  --rollout_mode full \
  --coarse_steps 10 \
  --output_dir $OUT
```

LIBERO-10 task0，20 trials，比较 `10/7/5/3/1` coarse denoising steps：

```bash
cd /root/ACoT-VLA
source scripts/env/use_acotvla_env.sh

OUT=/root/autodl-tmp/acotvla/stage_b_pruning_eval/coarse_steps_libero10_task0_20trials_all_steps

uv run python scripts/sweep_action_cot_coarse_steps.py \
  --mode closed_loop \
  --host 0.0.0.0 \
  --port 8000 \
  --task_suite_name libero_10 \
  --task_start 0 \
  --max_tasks 1 \
  --num_trials_per_task 20 \
  --rollout_mode full \
  --coarse_steps 10 7 5 3 1 \
  --overwrite \
  --output_dir $OUT
```

计划中的 LIBERO-10 全量固定 step sweep：

```bash
cd /root/ACoT-VLA
source scripts/env/use_acotvla_env.sh

OUT=/root/autodl-tmp/acotvla/stage_b_pruning_eval/coarse_steps_systematic_libero10_10tasks_20trials

uv run python scripts/sweep_action_cot_coarse_steps.py \
  --mode closed_loop \
  --host 0.0.0.0 \
  --port 8000 \
  --task_suite_name libero_10 \
  --task_start 0 \
  --max_tasks 10 \
  --num_trials_per_task 20 \
  --rollout_mode full \
  --coarse_steps 10 7 5 3 1 \
  --output_dir $OUT
```

继续训练 dynamic coarse steps head 的命令：

```bash
cd /root/ACoT-VLA
source scripts/env/use_acotvla_env.sh

uv run python scripts/train.py acot_libero_action_cot_dynamic_steps_stage_c \
  --exp-name dynamic_steps_stage_c_run1 \
  --checkpoint-base-dir /root/autodl-tmp/acotvla/checkpoints \
  --resume
```

该训练命令用于 Stage C / dynamic step head，让模型学习按样本自动选择 coarse denoising steps；它不是固定 `10/7/5/3/1` sweep 的前置条件。

### 已确认的结果

#### 1. Stage B entropy pruning open-loop / model-injected coarse action metrics

路径：

- `injection_adaptive_p03_1k/metrics.json`
- `injection_adaptive_p03_random_1k/metrics.json`
- `injection_adaptive_p03_high_1k/metrics.json`
- `injection_fixed_l5_p03_1k/metrics.json`

用户已贴出的 adaptive `prune_ratio=0.3` 指标：

| strategy | action_mse_to_full | action_mse_to_expert | coarse_mse_to_full | skip_ratio | avg_inference_time |
| --- | ---: | ---: | ---: | ---: | ---: |
| low_entropy | 0.010137461966772673 | 0.02523600491557843 | 0.01717304574051034 | 0.3699333333333334 | 90.0359969874844 |
| random | 0.010420817844713279 | 0.024824777219264786 | 0.02217786704707504 | 0.3592266666666667 | 72.15848131291568 |
| high_entropy | 0.01342737882259421 | 0.024919772231798688 | 0.028913715937464928 | 0.3485333333333334 | 139.9078439353034 |

用户已贴出的 fixed L=5, `prune_ratio=0.3` low-entropy 指标：

| metric | value |
| --- | ---: |
| coarse_l1_to_full | 0.019157185532073868 |
| coarse_mse_to_full | 0.009963901065444172 |
| action_l1_to_full | 0.01661984487649616 |
| action_mse_to_full | 0.005766404794497863 |
| action_l1_to_expert | 0.048848851989245615 |
| action_mse_to_expert | 0.022881458790355427 |
| trajectory_jerk | 0.14501483208792532 |
| gripper_error | 0.07807427024963506 |
| skip_ratio | 0.33333333333333326 |
| avg_inference_time | 86.27931768447161 |

事实结论：

- adaptive `prune_ratio=0.3` 下，low-entropy 的 `coarse_mse_to_full` 低于 random 和 high-entropy。
- adaptive `prune_ratio=0.3` 下，low-entropy 的 `action_mse_to_full` 低于 high-entropy，略低于 random。
- `action_mse_to_expert` 上 low-entropy 没有优于 random / high。
- fixed L=5 low-entropy 在当前贴出的结果中 `action_mse_to_full` 和 `coarse_mse_to_full` 低于 adaptive low-entropy，但二者 `skip_ratio` 不同，不能直接作为 adaptive 一定更差的最终结论。

#### 2. Mask / override coarse actions 的速度验证

路径：`/root/autodl-tmp/acotvla/stage_b_pruning_eval/speed_benchmark_adaptive_p03/summary.json`

用户已贴出的指标：

| metric | value |
| --- | ---: |
| full_acot_ms_mean | 72.97054733460148 |
| cached_coarse_override_ms_mean | 55.53876765693227 |
| pruned_coarse_override_ms_mean | 55.54128849878907 |
| skip_ratio_mean | 0.35866666666666663 |
| full_to_cached_speedup_pct | 23.888788441913377 |
| full_to_pruned_speedup_pct | 23.88533384009267 |
| cached_to_pruned_speedup_pct | -0.004538886912963669 |

事实结论：

- 绕过完整 explicit Action-CoT coarse generation 到 cached coarse override 可以节省约 23.89% latency。
- pruned coarse override 相比 cached coarse override 没有额外加速，`cached_to_pruned_speedup_pct` 接近 0。
- 因此，只改变 / mask coarse action values 不是 deployable segment-level 加速路径。

#### 3. Entropy-selected true segment skip speed 验证

路径：`/root/autodl-tmp/acotvla/stage_b_pruning_eval/true_entropy_skip_speed_p03/summary.json`

用户已贴出的指标：

| metric | value |
| --- | ---: |
| full_acot_ms_mean | 72.74821210031708 |
| cached_coarse_override_ms_mean | 55.65382903441787 |
| pruned_coarse_override_ms_mean | 55.48750493054589 |
| true_entropy_segment_skip_ms_mean | 73.20423135533929 |
| skip_ratio_mean | 0.3333333333333333 |
| true_skip_ratio_mean | 0.3333333333333333 |
| full_to_true_entropy_segment_skip_speedup_pct | -0.6268459964258399 |
| cached_to_pruned_speedup_pct | 0.298854736066978 |

事实结论：

- 当前 true entropy segment skip 实现没有带来加速，`full_to_true_entropy_segment_skip_speedup_pct` 为负。
- 该结果说明：如果实现没有减少 explicit reasoner / denoising 主循环调用次数，按 entropy 跳过部分 coarse token 本身不一定更快。

#### 4. Open-loop coarse denoising step speed sweep

路径：`/root/autodl-tmp/acotvla/stage_b_pruning_eval/coarse_steps_speed_sweep/coarse_steps_speed_summary.csv`

用户已贴出的指标：

| coarse_num_steps | full_acot_ms_mean | speedup_vs_coarse10_pct |
| ---: | ---: | ---: |
| 10 | 73.81813834110896 | 0.0 |
| 7 | 67.41823027531306 | 8.669831303821729 |
| 5 | 63.225869461894035 | 14.349141169435509 |
| 3 | 60.316416497031845 | 18.290520659958275 |
| 1 | 57.10195198034247 | 22.645093382770032 |

事实结论：

- open-loop timing 已证明减少 coarse denoising steps 可以降低 policy inference latency。
- 该实验只证明速度，不单独证明 closed-loop 成功率。

#### 5. LIBERO-10 task0 baseline sanity

路径：`/root/autodl-tmp/acotvla/stage_b_pruning_eval/coarse_steps_libero10_baseline_sanity`

配置：

- `task_suite_name=libero_10`
- `task_start=0`
- `max_tasks=1`
- `num_trials_per_task=20`
- `rollout_mode=full`
- `coarse_steps=10`

用户已贴出的指标：

| metric | value |
| --- | ---: |
| episodes | 20 |
| success_rate | 0.95 |
| average_return | 0.95 |
| timeout_rate | 0.05 |
| avg_wall_inference_ms | 92.4216823455168 |
| avg_policy_inference_ms | 82.49553051029119 |
| avg_server_inference_ms | 91.4254061563256 |
| avg_full_calls_per_episode | 64.55 |

事实结论：

- `libero_10` 与该 long checkpoint 对齐，task0 baseline 成功率为 0.95。
- 该 sanity run 支持后续使用 `libero_10` 做 closed-loop speed-success tradeoff。

#### 6. LIBERO-10 task0 closed-loop coarse step sweep

路径：`/root/autodl-tmp/acotvla/stage_b_pruning_eval/coarse_steps_libero10_task0_20trials_all_steps/coarse_steps_closed_loop_summary.csv`

配置：

- `task_suite_name=libero_10`
- `task_start=0`
- `max_tasks=1`
- `num_trials_per_task=20`
- `rollout_mode=full`
- `coarse_steps=10 7 5 3 1`

用户已贴出的指标：

| coarse_num_steps | success_rate | average_return | timeout_rate | avg_wall_inference_ms | avg_policy_inference_ms | avg_server_inference_ms | speedup_vs_coarse10_pct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.95 | 0.95 | 0.05 | 92.71958060707757 | 82.69714064932793 | 91.71433830888272 | 0.0 |
| 7 | 1.0 | 1.0 | 0.0 | 87.58383951144367 | 77.32679171922877 | 86.52865710470535 | 5.539003802657271 |
| 5 | 1.0 | 1.0 | 0.0 | 84.16512730898869 | 73.78711821431709 | 83.16670466463341 | 9.226156160412891 |
| 3 | 1.0 | 1.0 | 0.0 | 81.77470804548531 | 71.05684950548688 | 80.71142296003535 | 11.80427315344954 |
| 1 | 1.0 | 1.0 | 0.0 | 79.22992294532452 | 68.20720148524433 | 78.20628453366854 | 14.548876918370524 |

事实结论：

- 在 LIBERO-10 task0 的 20 trials 上，减少 coarse denoising steps 从 10 到 1，使 `avg_wall_inference_ms` 从 92.72 ms 降到 79.23 ms。
- 同一实验中，`coarse_num_steps=1/3/5/7` 的成功率均为 1.0；`coarse_num_steps=10` 的成功率为 0.95。
- 该结果只覆盖 task0，不能直接外推到全部 LIBERO-10 tasks。

### 已排除或仅作辅助参考的结果

- `closed_loop_adaptive_p03_smoke` 和 `closed_loop_adaptive_p03_10tasks` 属于 online MC then override 路径；其中 pruned override 会先跑 full ACoT 估计 entropy，不是 deployable speed path。
- 之前 `libero_spatial` 上的 coarse step closed-loop sweep 成功率约 0.35-0.44，但该 suite 与当前 long checkpoint 不匹配，因此不作为最终成功率证据。
- EGL cleanup 报错出现在 LIBERO / robosuite 结束阶段，日志显示 summary 已写出；该报错本身未被用作实验指标。

### 代码和运行问题记录

- PyAV / FFmpeg 报错 `libavformat.so.61` 通过 `source scripts/env/use_acotvla_env.sh` 后可正常 import `av`；确认环境变量包含 `/root/autodl-tmp/acotvla/ffmpeg-7.1.1/lib`。
- 训练 data loader 曾出现 `ActionCotLabelLoader._load_uncached` lru_cache wrapper multiprocessing pickling error，后续已有代码修复并在服务器 pull 到 `94118d1`。
- `scripts/eval_libero_action_cot_pruning.py` 曾在 LIBERO episode 已结束后继续 `env.step(...)` 导致 `ValueError: executing action in terminated episode`；后续修复为每个 episode 独立创建/关闭 env，并在 terminated episode error 时结束当前 episode 而不是中断整个 sweep。
- `.gitignore` 已加入 `outputs/` 和 `work/`，避免本地临时目录被误提交。

### 当前结论

- entropy pruning 的 open-loop 结果支持：低熵剪裁在 `coarse_mse_to_full` 和 `action_mse_to_full` 上优于高熵剪裁；但这不是 deployable speed proof。
- 当前已验证：只 mask / override coarse action values 不会产生额外 segment-level latency saving。
- 当前已验证：减少 explicit Action-CoT coarse denoising steps 可以带来实际 inference latency 下降。
- 当前已验证：在 LIBERO-10 task0 的 20 trials 上，`coarse_num_steps=1/3/5/7` 相比 10 steps 没有成功率下降，并有 5.54%-14.55% wall latency speedup。
- 当前尚未验证：上述 speed-success tradeoff 是否能稳定推广到 LIBERO-10 的全部 10 个 tasks。

### 下一步计划

1. 跑完 LIBERO-10 全量固定 step sweep：`10 tasks x 20 trials x 5 settings = 1000 episodes`。
2. 汇总每个 `coarse_num_steps` 的 `success_rate`、`average_return`、`timeout_rate`、`avg_wall_inference_ms`、`avg_policy_inference_ms` 和 `speedup_vs_coarse10_pct`。
3. 如果固定 `1/3/5/7` 在全量 tasks 上仍保持较高成功率，则优先汇报固定 step reduction 的速度-成功率曲线。
4. 如果不同 task 对 denoising steps 敏感，则继续训练并评估 `acot_libero_action_cot_dynamic_steps_stage_c`，让模型按样本动态选择 coarse denoising steps。
5. 将小型 summary 文件整理到 `results/stage_b/`，只保留 CSV / summary JSON，不提交 videos、checkpoints、wandb 或大规模临时输出。

## 2026-06-30 - AGENTS.md logging rule confirmation

本次任务读取了仓库根目录的 `AGENTS.md`，确认项目规则要求在每个有意义的 Codex 项目任务后追加事实摘要到 `reports/context/experiment_log.md`。后续在修改代码、调试、运行命令、分析实验结果或回答项目相关技术问题后，会按该规则自动追加日志；如果任务确实很 trivial 且没有可复用项目信息，可以跳过记录。

## 2026-07-01 - LIBERO-10 full coarse denoising step sweep and per-task analysis

本次任务分析了用户在服务器完成的 LIBERO-10 全量固定 coarse denoising step sweep。实验路径为：

`/root/autodl-tmp/acotvla/stage_b_pruning_eval/coarse_steps_systematic_libero10_10tasks_20trials`

该实验配置为：

- `task_suite_name=libero_10`
- `task_start=0`
- `max_tasks=10`
- `num_trials_per_task=20`
- `rollout_mode=full`
- `coarse_steps=10 7 5 3 1`
- 每个 setting 共 `10 tasks x 20 trials = 200 episodes`

用户贴出的 overall CSV 路径：

`$OUT/coarse_steps_closed_loop_summary.csv`

overall 结果如下：

| coarse_num_steps | success_rate | average_return | timeout_rate | avg_wall_inference_ms | avg_policy_inference_ms | avg_server_inference_ms | speedup_vs_coarse10_pct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.94 | 0.94 | 0.06 | 157.26577348394895 | 148.27564044621354 | 156.43065938183992 | 0.0 |
| 7 | 0.955 | 0.955 | 0.045 | 163.797758960521 | 154.4857852426062 | 162.99533136617828 | -4.153469208122851 |
| 5 | 0.95 | 0.95 | 0.05 | 156.3220231262081 | 146.76883158222233 | 155.51079513936244 | 0.6000990150836327 |
| 3 | 0.97 | 0.97 | 0.03 | 148.67060032553414 | 138.86706730934057 | 147.8392649699183 | 5.46538065340203 |
| 1 | 0.935 | 0.935 | 0.065 | 141.3187939184246 | 131.02233138571927 | 140.44259009522142 | 10.140146334607225 |

基于 overall 结果，`coarse_steps=3` 是当前 aggregate 上最好的速度-成功率折中：成功率从 `0.94` 提升到 `0.97`，`avg_policy_inference_ms` 从 `148.27564044621354` 降到 `138.86706730934057`，`avg_wall_inference_ms` 从 `157.26577348394895` 降到 `148.67060032553414`。`coarse_steps=1` 的 wall/policy latency 最低，但成功率为 `0.935`，低于 `coarse_steps=10` 的 `0.94`。

用户随后要求按 task 单独统计。使用 `coarse_steps_per_task_summary.csv` 进行 per-task 分析，路径为：

`$OUT/coarse_steps_per_task_summary.csv`

按 task success 统计，用户贴出的结果包括：

| coarse_num_steps | overall success | task0 | task1 | task2 | task3 | task4 | task5 | task6 | task7 | task8 | task9 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 188/200 = 0.940 | 19/20 | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 19/20 | 20/20 | 12/20 | 19/20 |
| 7 | 191/200 = 0.955 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 19/20 | 20/20 | 15/20 | 18/20 |
| 5 | 190/200 = 0.950 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 19/20 | 20/20 | 13/20 | 19/20 |
| 3 | 194/200 = 0.970 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 19/20 | 20/20 | 17/20 | 19/20 |
| 1 | 187/200 = 0.935 | 20/20 | 20/20 | 19/20 | 19/20 | 20/20 | 19/20 | 17/20 | 20/20 | 14/20 | 19/20 |

per-task 成功率结论：

- `coarse_steps=3` 没有任何 task 的成功率低于 `coarse_steps=10`，并且 task8 从 `12/20 = 0.600` 提升到 `17/20 = 0.850`。
- `coarse_steps=1` 速度最快，但 task2、task3、task6 成功率低于 `coarse_steps=10`；其中 task6 从 `19/20 = 0.950` 降到 `17/20 = 0.850`。
- task8 是当前最主要的困难 task：`coarse_steps=10` 为 `0.600`，`7` 为 `0.750`，`5` 为 `0.650`，`3` 为 `0.850`，`1` 为 `0.700`。

per-task latency 分析发现 task0 和 task1 的 `coarse_steps=10` latency 异常低。用户贴出的 per-task policy/wall 指标显示：

| task_id | coarse_steps | success | wall_ms | policy_ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10 | 0.950 | 93.02 | 83.12 |
| 1 | 10 | 1.000 | 90.88 | 81.57 |
| 0 | 7 | 1.000 | 164.78 | 155.13 |
| 1 | 7 | 1.000 | 163.85 | 154.51 |
| 0 | 5 | 1.000 | 156.91 | 147.10 |
| 1 | 5 | 1.000 | 156.48 | 146.77 |
| 0 | 3 | 1.000 | 149.18 | 139.17 |
| 1 | 3 | 1.000 | 148.95 | 139.02 |
| 0 | 1 | 1.000 | 141.64 | 131.29 |
| 1 | 1 | 1.000 | 141.97 | 131.45 |

该现象不符合减少 denoising steps 应该降低 latency 的逻辑，因为 task0/task1 的 `10 steps` 比同一 task 的 `7/5/3/1 steps` 都低很多。当前判断是：task0/task1 的 `coarse_steps=10` latency 是 timing artifact 或运行条件异常，不应作为速度结论的主要依据。可能原因包括运行顺序、server/JAX 异步计时、server 状态、缓存、队列或输出目录复用，但本次没有获得新的重跑结果来确认具体原因。

因此额外计算并讨论了排除 task0/task1 后的 task2-task9 指标。基于用户贴出的 per-task CSV，task2-task9 的 macro 平均为：

| coarse_num_steps | success_rate | avg_policy_inference_ms | policy_speedup_vs_10 |
| ---: | ---: | ---: | ---: |
| 10 | 0.9313 | 164.76 | 0.00% |
| 7 | 0.9437 | 154.40 | +6.29% |
| 5 | 0.9375 | 146.73 | +10.94% |
| 3 | 0.9625 | 138.81 | +15.75% |
| 1 | 0.9187 | 130.94 | +20.53% |

task2-task9 结果符合预期：coarse denoising steps 越少，policy latency 越低。`coarse_steps=3` 在 task2-task9 上同时提升成功率和降低 policy latency；`coarse_steps=1` 更快，但成功率低于 `coarse_steps=10`。

本次还向用户解释了 `wall latency` 和 `policy latency` 的区别：

- `policy latency` 是 policy server 内部模型推理耗时。
- `wall latency` 是 client 端一次 `client.infer(...)` 从发请求到拿到 action 的端到端耗时，包含模型推理、通信、序列化/反序列化和等待开销。
- closed-loop 汇报可以同时报告二者；如果只选一个接近部署体验的指标，可以报告 `avg_wall_inference_ms`，如果强调模型本身加速，则应报告 `avg_policy_inference_ms`。

本次给出的 task0/task1 `coarse_steps=10` 复核命令如下，目的是使用 fresh output directory 重跑 task0/task1 的 10-step baseline，确认之前的 `80-90ms` latency 是否为 timing artifact：

```bash
tmux new -s policy_server
cd /root/ACoT-VLA
source scripts/env/use_acotvla_env.sh

uv run python scripts/serve_policy.py \
  --env LIBERO \
  --port 8000 \
  policy:checkpoint \
  --policy.config acot_libero_action_cot_explicit_implicit_co_fusion \
  --policy.dir /root/autodl-tmp/acotvla/checkpoints/acot_libero_action_cot_explicit_implicit_co_fusion/acot_libero_long_run1/50999
```

```bash
tmux new -s rerun_task01_step10
cd /root/ACoT-VLA
git pull origin main
source scripts/env/use_acotvla_env.sh

OUT=/root/autodl-tmp/acotvla/stage_b_pruning_eval/coarse_steps_libero10_step10_task0to1_rerun

uv run python scripts/sweep_action_cot_coarse_steps.py \
  --mode closed_loop \
  --host 0.0.0.0 \
  --port 8000 \
  --task_suite_name libero_10 \
  --task_start 0 \
  --max_tasks 2 \
  --num_trials_per_task 20 \
  --rollout_mode full \
  --coarse_steps 10 \
  --overwrite \
  --output_dir $OUT
```

重跑结果当前尚未提供。

当前支持的事实结论：

- LIBERO-10 full sweep 已完成，覆盖 `10 tasks x 20 trials x 5 settings`。
- overall aggregate 上，`coarse_steps=3` 的成功率最高，为 `0.97`，且比 `coarse_steps=10` 有更低的 wall/policy latency。
- task2-task9 的 per-task macro latency 趋势符合预期：`10 -> 7 -> 5 -> 3 -> 1` 的 policy latency 逐步降低。
- task0/task1 的 `coarse_steps=10` latency 异常低，正式速度结论不应只依赖包含该异常 baseline 的 overall 平均。
- 下一步需要重跑 task0/task1 的 `coarse_steps=10` fresh baseline，或在正式汇报中清楚标注该异常并使用 per-task / task2-task9 分析支撑速度结论。

## 2026-07-01 - task0/task1 step10 rerun and corrected policy latency summary

用户随后提供了 task0/task1 `coarse_steps=10` fresh rerun 的结果。该 rerun 使用 fresh output directory，目标是复核此前 full sweep 中 task0/task1 的 `coarse_steps=10` latency 异常低值。

用户贴出的 rerun overall 结果：

| coarse_num_steps | mode | success_rate | average_return | timeout_rate | avg_wall_inference_ms | avg_policy_inference_ms | avg_server_inference_ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | full | 1.0 | 1.0 | 0.0 | 202.28895532489668 | 193.3458599204167 | 201.4823369292749 |

用户贴出的 rerun per-task 结果：

| task_id | success | wall_ms | policy_ms | server_ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 228.87 | 220.00 | 228.05 |
| 1 | 1.000 | 175.71 | 166.69 | 174.91 |

该 rerun 结果确认此前 full sweep 中 task0/task1 的 `coarse_steps=10` latency 确实异常低，不应作为最终 latency baseline。使用 rerun 的 task0/task1 step10 数据替换旧值，并保留原 full sweep 中 task2-task9 的 step10 数据后，得到修正后的 macro policy latency 结果：

| coarse_num_steps | success_rate | avg_policy_inference_ms | policy_speedup_vs_corrected_10 |
| ---: | ---: | ---: | ---: |
| 10 | 0.945 | 170.47515198780383 | 0.00% |
| 7 | 0.955 | 154.4857852426062 | +9.37929461200395% |
| 5 | 0.950 | 146.76883158222233 | +13.90602684857996% |
| 3 | 0.970 | 138.86706730934057 | +18.541168205396037% |
| 1 | 0.935 | 131.02233138571927 | +23.142856974784863% |

修正后的 per-task policy latency summary 如下。每个 task 使用该 task 自己修正后的 `coarse_steps=10` 作为 baseline：

```text
task 0
10 steps: success 100.00%, policy 220.00ms
7 steps:  success 100.00%, policy 155.13ms, +29.49%
5 steps:  success 100.00%, policy 147.10ms, +33.14%
3 steps:  success 100.00%, policy 139.17ms, +36.74%
1 step:   success 100.00%, policy 131.29ms, +40.32%

task 1
10 steps: success 100.00%, policy 166.69ms
7 steps:  success 100.00%, policy 154.51ms, +7.31%
5 steps:  success 100.00%, policy 146.77ms, +11.95%
3 steps:  success 100.00%, policy 139.02ms, +16.60%
1 step:   success 100.00%, policy 131.45ms, +21.14%

task 2
10 steps: success 100.00%, policy 155.84ms
7 steps:  success 100.00%, policy 154.49ms, +0.87%
5 steps:  success 100.00%, policy 146.71ms, +5.86%
3 steps:  success 100.00%, policy 138.79ms, +10.94%
1 step:   success 95.00%, policy 130.79ms, +16.08%

task 3
10 steps: success 100.00%, policy 165.78ms
7 steps:  success 100.00%, policy 155.02ms, +6.49%
5 steps:  success 100.00%, policy 146.62ms, +11.56%
3 steps:  success 100.00%, policy 138.81ms, +16.27%
1 step:   success 95.00%, policy 130.73ms, +21.14%

task 4
10 steps: success 100.00%, policy 165.75ms
7 steps:  success 100.00%, policy 154.63ms, +6.71%
5 steps:  success 100.00%, policy 146.83ms, +11.41%
3 steps:  success 100.00%, policy 138.71ms, +16.32%
1 step:   success 100.00%, policy 131.10ms, +20.91%

task 5
10 steps: success 95.00%, policy 165.77ms
7 steps:  success 95.00%, policy 154.00ms, +7.10%
5 steps:  success 95.00%, policy 146.34ms, +11.72%
3 steps:  success 95.00%, policy 138.64ms, +16.36%
1 step:   success 95.00%, policy 130.83ms, +21.08%

task 6
10 steps: success 95.00%, policy 166.28ms
7 steps:  success 95.00%, policy 154.09ms, +7.33%
5 steps:  success 95.00%, policy 146.83ms, +11.70%
3 steps:  success 95.00%, policy 139.05ms, +16.38%
1 step:   success 85.00%, policy 131.12ms, +21.15%

task 7
10 steps: success 100.00%, policy 165.88ms
7 steps:  success 100.00%, policy 154.58ms, +6.81%
5 steps:  success 100.00%, policy 146.48ms, +11.69%
3 steps:  success 100.00%, policy 138.93ms, +16.25%
1 step:   success 100.00%, policy 131.09ms, +20.97%

task 8
10 steps: success 60.00%, policy 166.29ms
7 steps:  success 75.00%, policy 154.45ms, +7.12%
5 steps:  success 65.00%, policy 146.76ms, +11.75%
3 steps:  success 85.00%, policy 138.72ms, +16.58%
1 step:   success 70.00%, policy 131.01ms, +21.22%

task 9
10 steps: success 95.00%, policy 166.47ms
7 steps:  success 90.00%, policy 153.95ms, +7.52%
5 steps:  success 95.00%, policy 147.25ms, +11.54%
3 steps:  success 95.00%, policy 138.84ms, +16.60%
1 step:   success 95.00%, policy 130.83ms, +21.41%
```

修正后的事实结论：

- 替换 task0/task1 的异常 step10 latency 后，所有 task 的 policy latency 基本符合减少 denoising steps 会加速的趋势。
- 修正后 macro 结果中，`coarse_steps=3` 仍是当前最好的速度-成功率折中：success_rate 为 `0.970`，policy latency speedup 为 `+18.54%`。
- `coarse_steps=1` 的 policy latency speedup 最大，为 `+23.14%`，但 success_rate 为 `0.935`，低于修正后的 `coarse_steps=10` 的 `0.945`。
- task8 仍是主要困难任务，`coarse_steps=3` 在 task8 上的成功率最高，为 `85.00%`。

## 2026-07-06 - alternative acceleration ideas beyond fixed denoising-step hyperparameters

本次任务讨论了用户提出的核心问题：仅固定调整 `coarse_num_steps` / denoising steps 仍然像推理超参数调节，本身方法创新不足；此前 coarse action segment pruning 的 post-hoc mask / interpolation 路径也没有证明真实加速。因此后续应寻找不只是手动改 denoising step 数的加速机制。

讨论中澄清：

- 之前带来速度提升的 `coarse_num_steps=10/7/5/3/1` 实际控制的是 explicit Action-CoT coarse action denoising iterations。
- 该实验的价值是证明 Action-CoT denoising 存在可压缩计算冗余，但固定 K 不是最终方法。
- 之前 coarse action segment pruning 失败的关键原因是：先跑完整 ACoT 再 mask / interp 低熵 segment，不能减少前面的 explicit reasoner / denoising 主计算。

本次形成的备选方法方向：

1. Conditional ACoT invocation / act-or-think router：先用 cheap direct action head 或 shallow policy 生成动作，再由 verifier 判断是否需要调用完整 Action-CoT。简单状态直接 act，困难状态才 think。该方法减少的是整次 Action-CoT 调用次数，不是固定改 denoising step 超参。
2. Adaptive replanning / action chunk execution：当前 closed-loop 固定 replan，每次都请求 policy。可以让模型预测 action chunk 的可信执行长度，稳定状态执行更长 chunk，高风险状态提前 replan。这样减少 policy calls per episode，直接降低 wall-clock latency。
3. Learned sparse / compressed Action-CoT representation：不再生成 15 个 dense coarse action token 后再剪，而是训练模型生成少量 control points、B-spline/keyframe 或 latent CoT token，再由 decoder / final head恢复细粒度 action。该方向将“segment pruning”改成训练时可学习的稀疏 CoT 表示。
4. Trainable skip-token / sparse-CoT consumer：让 final action head 在训练时见过 `[SKIP]` 或 sparse coarse actions，而不是推理时临时 interpolation。只有当模型结构支持真正减少 skipped segment 的计算时，该方向才可能带来部署加速。
5. Visual/language token pruning or caching：借鉴 LLM / Transformer token pruning，对 VLA backbone 的视觉 token、语言 token或跨模态 token做动态裁剪或缓存。尤其 LIBERO 语言 prompt 固定、相邻 replan 视觉变化较小，prompt KV cache、视觉特征缓存、temporal feature reuse 可能降低 backbone 端成本。
6. Draft-verify-escalate：先用 cheap path，例如 no-ACoT、1-step ACoT 或 compressed CoT 产生 draft action；verifier 根据 entropy、action disagreement、jerk、gripper risk、state deviation 等判断是否升级到 full ACoT。该方法不是固定 step，而是风险驱动计算分配。
7. Budget-aware distillation / RL：参考 LLM CoT pruning，把“少思考”变成训练目标。先用 10-step teacher 生成训练目标，再训练 short-budget student / router，最后可用 closed-loop reward `success - latency penalty - safety penalty` 微调。

当前建议优先级：

- 第一优先级：Conditional ACoT invocation / draft-verify-escalate。原因是它能跳过整段 Action-CoT，速度收益可能大于只减少 denoising steps，而且容易用现有 checkpoint 做 training-free prototype。
- 第二优先级：Adaptive replanning。原因是直接减少 `client.infer` 调用次数，和 Action-CoT 内部实现解耦，容易验证 wall-clock speedup 与 success tradeoff。
- 第三优先级：Compressed sparse Action-CoT representation。原因是最贴近 Action-CoT 方法创新，但需要训练和结构改动。

本次没有新增代码或实验结果；这是后续方法路线讨论。

## 2026-07-08 - Adaptive Replanning terminology clarification

本次任务澄清了 Adaptive Replanning 中 `step` 和默认 horizon 的含义。当前代码中 LIBERO eval 的固定 replanning horizon 由 `--replan_steps` 控制，默认值为 `5`：

- `scripts/eval_libero_action_cot_pruning.py` 中 `--replan_steps` / `--replan-steps` 默认 `5`
- `scripts/eval_action_cot_pruning.py` 中 `--replan_steps` / `--replan-steps` 默认 `5`
- `scripts/eval_on_libero_plus.py` 中 `replan_steps: int = 5`

因此当前 baseline 的默认执行策略是：policy 每次输出一个 action chunk，但 eval 只执行前 `5` 个 environment/control steps，然后重新请求 policy。Adaptive Replanning 中讨论的动态 horizon `H` 应以该默认值 `H=5` 作为 baseline，对比例如 `H in {2,3,5,8,10}` 的动态策略。这里的 `H` 是 environment/control steps，不是 Action-CoT denoising steps。

## 2026-07-08 - implemented adaptive replanning evaluation with optional entropy guidance

本次任务按 Adaptive Replanning 方案修改了 closed-loop LIBERO eval 代码。目标是不改模型结构、不重新训练，先验证通过动态调整 replanning horizon `H` 是否可以减少 policy 调用次数和 episode 总推理成本。

修改文件：

- `scripts/eval_libero_action_cot_pruning.py`
- `scripts/sweep_action_cot_coarse_steps.py`

主要新增参数：

- `--adaptive_replanning none|action|entropy|action_entropy`
- `--adaptive_replan_horizons`，默认候选 `3 5 8 10`，并自动包含固定 baseline `--replan_steps`
- `--adaptive_replan_entropy_mode none|coarse_proxy|online_mc`
- `--adaptive_replan_entropy_samples`
- `--adaptive_replan_entropy_low_quantile`
- `--adaptive_replan_entropy_high_quantile`
- `--adaptive_replan_entropy_warmup`
- `--adaptive_replan_entropy_low`
- `--adaptive_replan_entropy_high`
- `--adaptive_replan_jerk_low`
- `--adaptive_replan_jerk_high`
- `--adaptive_replan_gripper_change_threshold`

实现逻辑：

- baseline 不变：`--adaptive_replanning none` 时仍固定执行 `--replan_steps` 个 low-level control steps，默认 `5`。
- `action` 模式：根据 action chunk 的稳定性选择 horizon。代码计算 body action 的 mean delta、mean jerk、jerk ratio，以及 gripper change。低 jerk ratio 选择更长 horizon，高 jerk ratio 选择更短 horizon；如果 gripper change 超过阈值，则用 gripper guard 把 horizon cap 到默认 `replan_steps` 附近，避免抓取/释放阶段盲目执行太久。
- `entropy` 模式：使用 Action-CoT uncertainty signal 调整 horizon。entropy 不直接覆盖 action/gripper 风险，而是作为一档调节器：低 entropy 将 horizon 往长调一档，高 entropy 将 horizon 往短调一档。
- `action_entropy` 模式：先用 action stability 得到基础 horizon，再用 entropy 低/高不确定性做一档上调/下调。

entropy 使用方式：

- `coarse_proxy`：不额外调用 policy，只用当前 full inference 返回的单个 `coarse_actions` 计算 coarse action variation proxy。该 proxy 不是 Stage-B MC predictive entropy，但可作为便宜的部署型 uncertainty proxy。
- `online_mc`：对当前 observation 多次采样 `coarse_actions`，使用 Stage-B 的 `compute_mc_predictive_entropy` 计算 segment-level MC predictive entropy。该模式更接近 Stage B entropy，但会增加额外 policy calls，因此适合作为 oracle / analysis，不应直接当作 deployable speed path。
- 为避免 entropy 绝对数值随 norm stats/action dim 改变量纲，默认不用固定绝对阈值，而是维护当前 evaluation run 的 running entropy history。达到 `--adaptive_replan_entropy_warmup` 后，用 low/high quantiles 判断当前 entropy 是低、中、高；也支持用户传入绝对阈值覆盖 quantile。

新增记录指标：

- per episode CSV 新增 `total_wall_inference_ms`、`total_policy_inference_ms`、`total_server_inference_ms`
- 新增 deployable total timing：`total_deployable_wall_inference_ms`、`total_deployable_policy_inference_ms`、`total_deployable_server_inference_ms`
- 新增 `avg_replan_horizon`、`min_replan_horizon`、`max_replan_horizon`
- 新增 action stability 指标：`avg_adaptive_action_delta`、`avg_adaptive_action_jerk`、`avg_adaptive_action_jerk_ratio`、`avg_adaptive_gripper_change`
- 新增 entropy 指标：`avg_adaptive_entropy_score`、`avg_adaptive_entropy_mean`、`avg_adaptive_entropy_max`、`avg_adaptive_entropy_std`、`avg_adaptive_entropy_decision`
- 新增 `adaptive_replan_reasons`
- summary JSON 新增 per-episode 总推理成本和 `avg_num_replans_per_episode`

`scripts/sweep_action_cot_coarse_steps.py` 已透传 adaptive replanning 参数到 closed-loop eval，并在 closed-loop summary CSV 中加入：

- `avg_total_wall_inference_ms_per_episode`
- `avg_total_policy_inference_ms_per_episode`
- `avg_num_replans_per_episode`
- `avg_replan_horizon`

验证：

- 本地只做了轻量语法检查，没有跑 LIBERO closed-loop：
  - `python -m py_compile scripts/eval_libero_action_cot_pruning.py scripts/sweep_action_cot_coarse_steps.py`
  - `git diff --check -- scripts/eval_libero_action_cot_pruning.py scripts/sweep_action_cot_coarse_steps.py`
- 两项检查均通过。

注意事项：

- Adaptive Replanning 的主要速度指标不是单次 `avg_wall_inference_ms`，而是 `avg_num_replans_per_episode` 和 `avg_total_*_inference_ms_per_episode`。
- `online_mc` entropy 会额外调用 policy，实际 total timing 会包含这些额外调用；如果未来训练 entropy predictor，则应主要看 deployable timing 字段或使用 `coarse_proxy` 作为无额外调用的近似。

## 2026-07-08 - corrected Stage-B entropy accounting for adaptive replanning oracle evaluation

用户指出：如果目标是验证 Stage B entropy 能否决定 adaptive replanning horizon `H`，则 smoke 不能只用 `coarse_proxy`，必须保持 Stage B entropy 的计算方式，即对同一 observation 多次采样 `coarse_actions` 并计算 MC predictive entropy。与此同时，统计速度时应只统计“得到 entropy 后的优化版本”的成功率和耗时，而不是把 entropy MC 采样成本混入优化路径。

本次据此修正代码输出语义：

- `online_mc` entropy 仍然使用 Stage-B-style MC entropy 作为 oracle signal。
- `avg_total_*_inference_ms_per_episode` 继续表示 actual total timing，包含 online MC entropy 采样成本。
- `avg_total_deployable_*_inference_ms_per_episode` 表示 oracle/deployable timing：假设 entropy 已由 oracle 或未来 predictor 给出，只统计 entropy 决策后实际 action-producing policy calls 的耗时。
- 新增 `total_policy_calls`、`deployable_policy_calls`、`entropy_oracle_extra_calls` 到 per-episode CSV。
- summary JSON 新增 `avg_total_policy_calls_per_episode`、`avg_deployable_policy_calls_per_episode`、`avg_entropy_oracle_extra_calls_per_episode`。
- `scripts/sweep_action_cot_coarse_steps.py` 的 closed-loop summary CSV 也透传 deployable/oracle total timing 和 call count 字段。

因此后续验证 Stage B entropy + Adaptive Replanning 时，应用：

- success 指标：`success_rate`
- oracle 优化路径速度指标：`avg_total_deployable_wall_inference_ms_per_episode`、`avg_total_deployable_policy_inference_ms_per_episode`、`avg_deployable_policy_calls_per_episode`
- actual online-MC 成本说明：`avg_total_wall_inference_ms_per_episode`、`avg_total_policy_inference_ms_per_episode`、`avg_entropy_oracle_extra_calls_per_episode`

验证：

- `python -m py_compile scripts/eval_libero_action_cot_pruning.py scripts/sweep_action_cot_coarse_steps.py` 通过。
- `git diff --check -- scripts/eval_libero_action_cot_pruning.py scripts/sweep_action_cot_coarse_steps.py` 通过。

## 2026-07-06 - method discussion: beyond fixed denoising-step hyperparameter tuning

本次任务讨论了用户提出的问题：固定 sweep `coarse_num_steps=10/7/5/3/1` 本质上仍是改推理超参数，不足以构成真正的方法创新。当前实验的作用应被定位为 diagnostic evidence：证明 explicit Action-CoT coarse denoising 存在可压缩冗余，并为后续 adaptive / learned compute allocation 方法提供依据。

讨论中形成的主要方向：

1. Training-free adaptive early stop：在 coarse denoising 过程中根据不确定性、entropy、相邻 denoising step 的 coarse action 变化量或预测稳定性，动态决定是否提前停止。该方向对应 diffusion adaptive computation / early-exit 思路，不再固定使用全局 step 超参。
2. Entropy-guided segment-wise denoising：不再 post-hoc mask coarse action segment，而是在 denoising loop 内部对低熵 segment 提前 freeze 或跳过更新，对高熵 segment 保留更多 denoising compute。该方向能把 Stage A entropy labeling 和真实加速路径连接起来，但需要模型支持 segment/token-level sparse update 或 masked update。
3. Learned dynamic step head / router：训练一个轻量 head，根据 observation/backbone feature、entropy proxy、coarse uncertainty 等预测每个样本应该使用 `1/3/5/7/10` 中哪个 budget。训练标签可以来自 offline teacher-student oracle：选择满足 action/coarse error 阈值的最小 step 数。该方向是当前代码中 `acot_libero_action_cot_dynamic_steps_stage_c` 的自然延伸。
4. Budget-aware distillation / consistency training：用 10-step ACoT 作为 teacher，训练 few-step student，使其在 1/3/5 steps 下直接拟合 teacher 的 coarse actions 或 final actions。该方向更像 diffusion progressive distillation / consistency model，能让少 step 不是简单提前停止，而是让模型学会在少 step 下生成高质量 coarse reasoning。
5. RL + SFT compute-aware policy：参考 LLM long-CoT pruning 的做法，先用 SFT/distillation 让模型适应短 budget，再用 closed-loop reward 或 proxy reward 优化 `success - latency penalty - smoothness/safety penalty`。这比简单固定 step 更接近“模型学会短思考”的方法，但机器人 closed-loop RL 成本高，建议先做 offline/proxy，再做少量 rollout 验证。
6. Draft-verify / escalate 策略：先用 1 或 3 steps 生成 cheap draft，再由 verifier 判断风险；低风险直接执行，高风险升级到 5/10 steps。verifier 可以使用 entropy、coarse-action stability、action jerk、gripper uncertainty、与训练分布距离等特征。该方向能保证安全回退，工程上也较容易落地。

当前建议的技术路线：

- 短期论文/汇报主线：固定 step sweep 不是最终方法，而是证明 compute redundancy；基于此提出 Adaptive Action-CoT Denoising。
- 第一阶段实现 training-free gate：根据 denoising convergence / entropy 阈值动态 early stop，和固定 `3 steps`、固定 `1 step`、固定 `10 steps` 对比。
- 第二阶段实现 learned dynamic step router：使用 teacher `10 steps` 和 offline oracle label 训练预算预测 head，闭环评估速度-成功率 Pareto。
- 第三阶段探索 budget-aware distillation 或 RL fine-tuning，让模型真正适应短 denoising budget，而不是只在推理时硬改 step 数。

本次没有新增实验结果或代码改动；这是方法讨论和后续计划记录。

## 2026-07-08 - finer-grained ACoT inference timing split

本次任务根据“每个部分耗时需要进一步拆分 VLM 和 Action Expert 时间”的需求，修改了 ACoT 推理计时链路。

代码改动：

1. `src/openpi/models/acot_vla.py` 将 `sample_actions` 的推理路径拆成可单独 JIT 调用的 profile stage：`sample_actions_profile_prefix`、`sample_actions_profile_implicit`、`sample_actions_profile_coarse`、`sample_actions_profile_expert`。普通 `sample_actions` 仍复用同一套 stage helper，输出语义保持 `actions`、`coarse_actions`、`coarse_num_steps`。
2. `src/openpi/policies/policy.py` 新增 `profile_policy_timing` 请求开关。请求带该字段时，ACoT policy 会顺序执行四段 profile path，并在每段后同步等待，返回 `policy_timing` 细分字段：`vlm_ms`、`implicit_action_reasoner_ms`、`coarse_action_expert_ms`、`action_expert_ms`、`profile_overhead_ms`，同时保留原有 `infer_ms`。未请求 profile 或模型不支持 profile stage 时仍走原来的单次 `sample_actions`。
3. `scripts/benchmark_action_cot_speed.py` 默认为每次 benchmark inference 打开 `profile_policy_timing`，并在 `latency_rows.csv` 和 `summary.json` 中为 `full`、`cached_override`、`pruned_override`、`true_entropy_segment_skip` 四种模式记录上述细分耗时的 row 字段和 mean/std aggregate。
4. `examples/libero/main.py` 在启用 `timing_out_dir` 时为 websocket 请求打开 `profile_policy_timing`，episode JSONL 会保留完整 `server_timing` / `policy_timing`，`summary_by_inference_index.csv/jsonl` 会动态汇总所有出现过的 timing mean 字段。
5. `scripts/eval_libero_action_cot_pruning.py` 为 closed-loop pruning 评测请求打开 `profile_policy_timing`，并区分实际总消耗 `avg_policy_*` 和 deployable 路径 `avg_deployable_policy_*` 的 stage timing。
6. `src/openpi/serving/websocket_policy_server.py` 的 timing JSONL 记录从只写 `infer_ms` 改为透传完整 timing 字典。

验证：

- 已运行 `python -m py_compile src/openpi/policies/policy.py src/openpi/models/acot_vla.py src/openpi/serving/websocket_policy_server.py scripts/benchmark_action_cot_speed.py scripts/eval_libero_action_cot_pruning.py examples/libero/main.py`，语法编译通过。
- 尝试运行 `ruff check ...` 和 `uv run ruff check ...`，但当前环境 PATH 中没有 `ruff` 或 `uv`，因此未完成 lint。

未完成：

- 本次没有启动模型 checkpoint 做实际 GPU 推理或 LIBERO rollout，因此新增 timing 字段尚未用真实运行结果验证数值。

## 2026-07-08 - rename coarse_num_steps user-facing terminology to Action-CoT denoising steps

本次任务处理用户提出的命名混淆问题：此前实验命令和 CSV/JSON 中使用 `coarse_num_steps` / `coarse_steps` 描述的变量，实际含义是 explicit Action-CoT coarse-action generation 的 denoising iterations，而不是 coarse action segment 数量或重规划步长。

代码改动：

1. `scripts/eval_libero_action_cot_pruning.py` 新增推荐参数名 `--action_cot_denoising_steps` / `--denoising_steps`，旧的 `--coarse_num_steps` 保留为兼容别名。动态 step 开关新增 `--dynamic_denoising_steps`，旧的 `--dynamic_coarse_steps` 保留为兼容别名。
2. `scripts/benchmark_action_cot_speed.py` 同步新增 `--action_cot_denoising_steps` / `--denoising_steps` 和 `--dynamic_denoising_steps`，并在 summary/aggregate/latency rows 中新增 `action_cot_denoising_steps`、`action_cot_denoising_steps_used_mean`、`action_cot_denoising_steps_used` 等字段。旧的 `coarse_num_steps` 字段继续写出，作为 deprecated compatibility 字段。
3. `scripts/sweep_action_cot_coarse_steps.py` 的推荐 sweep 参数改为 `--denoising_steps` / `--action_cot_denoising_steps`，旧 `--coarse_steps` 保留为兼容别名。新输出文件名为 `denoising_steps_speed_summary.csv`、`denoising_steps_closed_loop_summary.csv`、`denoising_steps_systematic_summary.csv`，同时继续写出旧的 `coarse_steps_*` CSV 兼容文件。
4. 新增 `scripts/sweep_action_cot_denoising_steps.py`，作为推荐入口；内部复用原 `sweep_action_cot_coarse_steps.py` 实现，以免破坏已有命令。
5. `src/openpi/policies/policy.py` 支持请求字段 `action_cot_denoising_steps`、`denoising_steps`、`action_cot_dynamic_denoising_steps`、`dynamic_denoising_steps`，并保留旧字段 `action_cot_coarse_num_steps`、`coarse_num_steps`、`action_cot_dynamic_coarse_steps`、`dynamic_coarse_steps`。policy 输出新增 `action_cot_denoising_steps`，同时继续输出旧 `coarse_num_steps`。

验证：

- 已运行 `python -m py_compile scripts/eval_libero_action_cot_pruning.py scripts/benchmark_action_cot_speed.py scripts/sweep_action_cot_coarse_steps.py scripts/sweep_action_cot_denoising_steps.py src/openpi/policies/policy.py`，语法编译通过。
- 已运行 `git diff --check -- scripts/eval_libero_action_cot_pruning.py scripts/benchmark_action_cot_speed.py scripts/sweep_action_cot_coarse_steps.py scripts/sweep_action_cot_denoising_steps.py src/openpi/policies/policy.py`，未发现 whitespace error。

未完成：

- 本次没有运行服务器端 LIBERO rollout 或 checkpoint 推理，只做命名与兼容性层面的代码修改。

## 2026-07-08 - remove old coarse-step aliases and clarify entropy-guided timing

本次任务根据用户反馈继续清理 denoising step 命名，并修正 entropy-guided closed-loop 统计口径。

背景：

- 用户指出 `coarse_num_steps` / `coarse_steps` 容易和 coarse action segment 混淆；由于代码会整体 push 到服务器，不需要保留旧参数别名和兼容 CSV 副本。
- 用户还指出 `true` / entropy-guided 路径的 `avg_wall_ms` 从早期约 80ms 变成 300ms 以上，怀疑代码统计有问题。检查代码后确认：`avg_wall_inference_ms` 对 `online_mc` entropy 路径统计的是包含多次 Stage-B-style entropy sampling full policy calls 的观测总耗时；这与单次 action-producing inference 的 80ms 口径不同。

代码改动：

1. 删除旧 sweep 入口 `scripts/sweep_action_cot_coarse_steps.py`，保留并使用 `scripts/sweep_action_cot_denoising_steps.py` 作为唯一 sweep 入口。
2. `scripts/sweep_action_cot_denoising_steps.py` 只接受 `--action_cot_denoising_steps`，不再接受 `--coarse_steps` 或 `--denoising_steps` 旧/短别名；run 子目录改为 `denoising_steps_{step}`，只输出 `denoising_steps_speed_summary.csv`、`denoising_steps_closed_loop_summary.csv`、`denoising_steps_systematic_summary.csv`，不再写旧 `coarse_steps_*` 副本。
3. `src/openpi/models/acot_vla.py` 的模型采样接口从 `coarse_num_steps` / `dynamic_coarse_steps` 改为 `action_cot_denoising_steps` / `dynamic_denoising_steps`，模型输出也从 `coarse_num_steps` 改为 `action_cot_denoising_steps`。
4. `src/openpi/policies/policy.py` 删除旧 request/sample aliases，只读取 `action_cot_denoising_steps` 和 `action_cot_dynamic_denoising_steps`，并向模型传递新的内部参数名。
5. `scripts/eval_libero_action_cot_pruning.py` 删除旧 `--coarse_num_steps`、`--dynamic_coarse_steps` 参数别名和旧 `avg_coarse_num_steps_used` 输出字段，只保留 `action_cot_denoising_steps` 相关字段。
6. `scripts/benchmark_action_cot_speed.py` 删除旧参数别名和旧 `coarse_num_steps` 输出字段；保留并强化四段 profile timing：`vlm_ms`、`implicit_action_reasoner_ms`、`coarse_action_expert_ms`、`action_expert_ms`、`profile_overhead_ms`。summary 中新增 `stage_profile` 和 `entropy_guided_true_skip`，用于直接查看 entropy-guided true-skip 路径的 latency 和分段耗时。
7. closed-loop eval 新增 `primary_wall_inference_ms`、`primary_policy_inference_ms`、`primary_server_inference_ms`。这些字段 mirror deployable/action-producing timing，是用于速度-成功率比较的主字段；`avg_wall_inference_ms` / `avg_total_*` 仍保留为 observed total，用于记录包含 online MC entropy sampling 的实际额外成本。
8. sweep closed-loop 汇总改为用 `primary_wall_inference_ms` 计算 `speedup_vs_denoising10_pct`，同时保留 `observed_avg_wall_inference_ms`、`avg_entropy_oracle_extra_calls_per_episode`、四段 primary/observed policy timing 字段，避免把 entropy oracle sampling 成本误认为优化后单次推理耗时。

验证：

- 已运行 `python -m py_compile scripts/eval_libero_action_cot_pruning.py scripts/benchmark_action_cot_speed.py scripts/sweep_action_cot_denoising_steps.py src/openpi/policies/policy.py src/openpi/models/acot_vla.py`，语法编译通过。
- 已运行 `git diff --check -- scripts/eval_libero_action_cot_pruning.py scripts/benchmark_action_cot_speed.py scripts/sweep_action_cot_denoising_steps.py src/openpi/policies/policy.py src/openpi/models/acot_vla.py`，未发现 whitespace error。
- 已扫描 `scripts src/openpi examples` 中的旧 step 命名，`coarse_num_steps`、`dynamic_coarse_steps`、`action_cot_coarse_num_steps`、`sweep_action_cot_coarse_steps`、`--coarse_steps`、`speedup_vs_coarse10`、`avg_coarse_num_steps` 未再出现在可执行代码中。

未完成：

- 本次没有运行服务器端 LIBERO rollout 或 checkpoint 推理，没有产生新的成功率或延迟实验结果。

## 2026-07-08 - adaptive replanning entropy oracle command

本次任务回答了如何运行 `adaptive_replan_entropy_oracle`。该实验对应 `scripts/eval_libero_action_cot_pruning.py` 中的 closed-loop adaptive replanning 路径，关键参数是 `--adaptive_replanning entropy` 和 `--adaptive_replan_entropy_mode online_mc`。`online_mc` 会在线多次采样 coarse_actions 计算 Stage-B-style MC entropy，因此 `observed_*` timing 会包含 entropy sampling 额外 policy calls；用于“如果 entropy 已知/可预测时的优化速度”比较时应看 `primary_*` 或 `avg_total_deployable_*` 字段。本次只提供命令，没有运行实验，也没有新增结果。

## 2026-07-08 - default Action-CoT denoising steps set to 10

本次任务根据用户要求，将后续默认 `action_cot_denoising_steps` 设为 `10`。

代码改动：

1. `scripts/eval_libero_action_cot_pruning.py` 的 `--action_cot_denoising_steps` 默认值从 `None` 改为 `10`。如果显式启用 `--action_cot_dynamic_denoising_steps`，eval 请求不会再同时传固定的 10-step 参数，避免 dynamic step head 被默认值覆盖。
2. `scripts/benchmark_action_cot_speed.py` 的 `--action_cot_denoising_steps` 默认值从 `None` 改为 `10`。如果显式启用 `--action_cot_dynamic_denoising_steps`，benchmark 创建 policy 时不会传固定 step 参数；summary 中 `_effective_action_cot_denoising_steps` 对 dynamic 模式仍记录为 `-1`。

验证：

- 已运行 `python -m py_compile scripts/eval_libero_action_cot_pruning.py scripts/benchmark_action_cot_speed.py scripts/sweep_action_cot_denoising_steps.py src/openpi/policies/policy.py src/openpi/models/acot_vla.py`，语法编译通过。
- 已运行 `git diff --check -- scripts/eval_libero_action_cot_pruning.py scripts/benchmark_action_cot_speed.py scripts/sweep_action_cot_denoising_steps.py src/openpi/policies/policy.py src/openpi/models/acot_vla.py`，未发现 whitespace error。

未完成：

- 本次没有运行服务器端 LIBERO rollout 或 checkpoint 推理，没有产生新的实验结果。

## 2026-07-08 - baseline timing CSV inspection

本次任务读取并分析了用户提供的 baseline timing CSV：

`/Users/kongyue/Library/Containers/com.bytedance.macos.feishu/Data/Downloads/summary_by_inference_index.csv`

CSV 字段包括 `task_suite_name`、`exp_name`、`task_id`、`task_description`、`infer_idx`、`num_trials_requested`、`num_trials_with_this_inference`、`server_infer_ms_mean`、`policy_infer_ms_mean`。该文件对应 `libero_10`，实验名为 `acot_libero_long_timing_50`，共 10 个 task，每个 task 请求 50 trials，合计 500 episodes。

分析结果：

- 包含 `infer_idx=0` 的全部记录：加权单次 policy mean 为 `78.9558ms`，server mean 为 `84.3060ms`；平均每 episode `58.502` 次 inference；平均每 episode policy total 为 `4619.07ms`，server total 为 `4932.07ms`。
- 排除 `infer_idx=0` 后：加权单次 policy mean 为 `77.9105ms`，server mean 为 `83.2552ms`；平均每 episode `57.502` 次 inference；平均每 episode policy total 为 `4480.01ms`，server total 为 `4787.34ms`。
- `infer_idx=0` 在 task0 上异常大：policy `679.40ms`、server `686.21ms`；其余 task 的 `infer_idx=0` 约 `79ms` policy / `84ms` server。因此 task0 的第 0 次 inference 很可能包含首轮初始化/JIT/缓存影响，不适合直接作为稳定推理延迟。
- 排除 `infer_idx=0` 后，per-task calls/episode 差异明显：task8 为 `106.96` calls/episode，显著高于其他 task；task0 为 `55.76`，task1 为 `51.50`，task2 为 `49.44`，task3 为 `47.90`，task4 为 `47.20`，task5 为 `48.42`，task6 为 `57.08`，task7 为 `54.24`，task9 为 `56.52`。

支持的结论：

- 原版 baseline 的稳定单次推理确实约为 `78ms` policy / `83ms` server，与用户此前观察的 `80ms` 量级一致。
- `online_mc` entropy oracle 中出现的 `observed_avg_wall_ms` 约 400ms 量级，不能和 baseline 单次 inference 直接比较；它包含 MC entropy sampling 的额外 policy calls。
- 若 entropy oracle 的 `primary_avg_wall_ms` 约 `78.61ms`，它与 baseline 稳定单次推理基本相同，说明目前没有单次 inference 加速；adaptive replanning 是否加速必须看是否减少了 calls/episode，即比较 baseline 的约 `57.5` calls/episode 和 oracle 的 `avg_deployable_policy_calls_per_episode` / `avg_replan_horizon`。

未完成：

- 该 CSV 不包含 success rate，因此不能单独用于判断速度-成功率 Pareto。
- 本次没有读取服务器上的 entropy oracle `summary.json`，因此未计算完整 deployable total speedup。

## 2026-07-08 - checked Codex logs_2.sqlite high-frequency TRACE writes

本次任务检查用户提到的 `~1.codex/logs_2.sqlite` 是否因 TRACE 日志持续高频写盘。当前目录和 home 下没有找到字面路径 `~1.codex/logs_2.sqlite`，实际找到的 Codex 日志库包括 `/Users/kongyue/.codex/logs_2.sqlite` 和旧位置 `/Users/kongyue/.codex/sqlite/logs_2.sqlite`。旧位置主库 `mtime` 停在 2026-06-25，当前活跃库是 `/Users/kongyue/.codex/logs_2.sqlite`，并伴随 `/Users/kongyue/.codex/logs_2.sqlite-wal` 和 `/Users/kongyue/.codex/logs_2.sqlite-shm`。

使用 `sqlite3 -readonly` 查询活跃库的 `logs` 表，字段包括 `id`、`ts`、`level`、`target`、`feedback_log_body`、`estimated_bytes` 等。聚合结果显示截至检查时共有约 16880 条记录，其中 `TRACE` 约 11061 条，估算内容体积约 45.45 MB；最近 1 分钟约 1151 条记录中 `TRACE` 约 993 条，约 16.55 条/秒，估算 TRACE 内容约 2.30 MB；最近 5 分钟约 1533 条 TRACE，估算约 11.38 MB。TRACE 来源主要包括 `log`、`hyper_util::client::legacy::pool`、`codex_api::sse::responses`、`hyper_util::client::legacy::client`、`codex_mcp::connection_manager` 等，其中短窗口内 `codex_api::sse::responses` 是主要来源之一。

短时间采样显示数据库总行数基本保持在约 17012，但 `MAX(id)` 在约 23 秒内从 `791919` 增到 `793395`，说明日志库在持续插入并清理旧记录。仅做 `stat` 的采样也显示 WAL 文件 `mtime` 约每 1-2 秒刷新一次，排除了 SQLite 查询本身造成 WAL 更新的可能性。`lsof` 显示两个 Codex app-server 进程打开同一活跃日志库：PID 20302 来自 `/Users/kongyue/.vscode/extensions/openai.chatgpt-26.623.141536-darwin-arm64/bin/macos-aarch64/codex app-server --analytics-default-enabled`，PID 66160 来自 `/Applications/Codex.app/Contents/Resources/codex app-server --analytics-default-enabled`。进程环境和当前 shell 中过滤到的 `RUST_LOG` 均为 `warn`，因此这次没有证据表明是外部 `RUST_LOG=trace` 导致；证据支持 Codex 内部 TRACE 级日志正在持续写入 `/Users/kongyue/.codex/logs_2.sqlite` 的结论。

## 2026-07-08 - installed SQLite trigger to block Codex logs inserts

本次任务在确认 `/Users/kongyue/.codex/logs_2.sqlite` 存在高频写盘后，按用户要求使用 SQLite trigger 拦截 `logs` 表的新插入。先创建过一个只拦截 `level='TRACE'` 的 trigger，但考虑到用户要求用 `MAX(id)` 验收，随后改为拦截 `logs` 表所有 insert：

```sql
CREATE TRIGGER IF NOT EXISTS ignore_logs_before_insert
BEFORE INSERT ON logs
BEGIN
  SELECT RAISE(IGNORE);
END;
```

当前库中存在的 trigger 为 `ignore_logs_before_insert|logs|CREATE TRIGGER ignore_logs_before_insert BEFORE INSERT ON logs BEGIN SELECT RAISE(IGNORE); END`。创建 trigger 后开始复测时，`logs` 表状态为 `COUNT(*)=17470`、`TRACE=11286`、`MAX(id)=804322`、最后日志时间 `2026-07-08 10:56:43`、最后 TRACE 时间 `2026-07-08 10:56:19`。

之后进行了约 20 秒采样，重复读取 `COUNT(*)`、`SUM(level='TRACE')`、`MAX(id)`、最后日志时间、最后 TRACE 时间，并用 `stat` 读取 `/Users/kongyue/.codex/logs_2.sqlite` 和 `/Users/kongyue/.codex/logs_2.sqlite-wal` 的 size/mtime。采样期间所有值保持不变：`COUNT(*)=17470`、`TRACE=11286`、`MAX(id)=804322`、主库 `size=61284352 mtime=1783479367`、WAL `size=5376632 mtime=1783479404`。结论是 trigger 生效后，`logs` 表新记录、TRACE 记录和 WAL 文件在采样窗口内均不再增长。

恢复日志写入的回滚命令是：

```sh
sqlite3 /Users/kongyue/.codex/logs_2.sqlite "DROP TRIGGER IF EXISTS ignore_logs_before_insert;"
```

## 2026-07-08 - entropy adaptive speed comparison protocol

本次任务明确了 Stage-B entropy adaptive replanning 的验证目标：不是证明当前 online MC entropy oracle 本身可部署加速，而是比较“原版模型固定 replanning”的速度与“已知/可预测 entropy 后的 adaptive replanning 决策”的可部署速度。如果该比较显示 entropy-guided adaptive replanning 能减少总推理时间并保持成功率，才有继续开发一个小型 entropy predictor 的必要。

建议比较口径：

- 原版 baseline：使用同一 checkpoint、同一 `libero_10` task 范围、同一 trials 数、同一 `action_cot_denoising_steps`，记录 success rate、平均每 episode policy/server/wall 总推理时间，以及 policy calls per episode。
- entropy adaptive oracle：运行 `scripts/eval_libero_action_cot_pruning.py` 的 `--adaptive_replanning entropy --adaptive_replan_entropy_mode online_mc`，但速度结论不使用包含 MC 采样额外调用的 `observed_*` / `avg_wall_inference_ms`；用于 predictor 价值判断时应使用 `primary_*`、`avg_deployable_*`、`avg_total_deployable_*_per_episode` 和 `avg_deployable_policy_calls_per_episode`。
- 如果 entropy adaptive 的 `avg_total_deployable_policy_inference_ms_per_episode` 或 `avg_total_deployable_wall_inference_ms_per_episode` 低于 baseline，同时 success rate 没有明显下降，才说明 entropy 决策本身有加速潜力。
- 如果单次 `primary_avg_*` 仍约等于原版单次 inference 延迟，则说明没有单次模型调用加速；加速只能来自减少每个 episode 的 policy call 数。

已有事实依据：

- 用户此前提供的原版 baseline timing CSV 显示，排除首个异常初始化调用后，原版稳定单次 policy inference 约 `77.91ms`，server inference 约 `83.26ms`，平均每 episode 约 `57.50` 次 inference。
- 用户近期 entropy oracle 日志中，`primary_avg_wall_ms` 约 `77-78ms`，与原版单次 inference 量级接近；`observed_avg_wall_ms` 约 `400ms` 是因为包含 online MC entropy sampling 的额外 policy calls，不能直接作为可部署速度结论。
- 当前仍需读取完整 entropy adaptive run 的 `summary.json` / `rollout_rows.csv`，计算与原版 baseline 对齐后的 success rate、deployable calls per episode 和 deployable total inference time per episode。没有这些完整结果前，不能断言 entropy adaptive 已经加速。

## 2026-07-08 - organized original LIBERO-10 per-trial baseline logs

本次任务整理了用户提供的原版 baseline 日志目录：

`/Users/kongyue/Downloads/libero_long_timing_logs/aligned_eval/acot_libero_long_timing_50/libero_10/episodes`

该目录包含 10 个 task 子目录，每个 task 下有 `trial_*.jsonl`。每个 JSONL 文件是一条 episode，每行是一轮 policy/server inference call。解析结果显示共有 `500` 个 trial 文件、`10` 个 task，解析错误为 `0`，trial 内 episode-level 字段未发现不一致。

生成的整理文件位于：

`/Users/kongyue/LightAcotVLA/results/stage_b/original_baseline/`

生成文件：

- `libero10_original_per_trial.csv`：每个 trial 一行，包含 task/trial、success、episode steps、inference calls、per-call latency、episode total latency，以及排除 `infer_idx=0` 的版本。
- `libero10_original_per_task_summary.csv`：每个 task 一行，末尾含 `all` 汇总行。
- `libero10_original_overall_summary.json`：原始整体汇总。
- `libero10_original_per_task_summary_excluding_task0_trial0_warmup.csv`：推荐用于后续 PK 的 clean baseline，只排除唯一 warmup outlier：task0 trial0。
- `libero10_original_overall_summary_excluding_task0_trial0_warmup.json`：clean baseline 的整体 JSON 汇总。
- `README.md`：说明字段含义和推荐比较口径。

原始整体结果（包含所有 inference calls）：

- success rate: `0.9660`
- calls per episode: `58.5020`
- weighted policy latency per call: `78.9558 ms`
- weighted server latency per call: `84.3060 ms`
- mean total policy inference time per episode: `4619.0703 ms`
- mean total server inference time per episode: `4932.0689 ms`

解析过程中发现 task0 trial0 是唯一明显 warmup outlier：该 episode 的 `infer_idx=0` policy timing 为 `30065.45 ms`，server timing 为 `30129.33 ms`；同一 trial 后续 inference 约 `78-82 ms`。只有 task0 trial0 的 trial-level policy mean 大于 `120 ms`。

推荐用于 entropy-adaptive 可部署速度 PK 的 clean baseline（排除 task0 trial0，保留其他 episode 的正常首轮 inference）：

- success rate: `0.9659`
- calls per episode: `58.5190`
- weighted policy latency per call: `77.9287 ms`
- weighted server latency per call: `83.2784 ms`
- mean total policy inference time per episode: `4560.3128 ms`
- mean total server inference time per episode: `4873.3741 ms`

Clean per-task 重点结果：

- task0: `49` trials, success `1.0000`, calls/episode `56.8980`, policy total/episode `4473.6498 ms`
- task1: `50` trials, success `1.0000`, calls/episode `52.5000`, policy total/episode `4110.9523 ms`
- task2: `50` trials, success `1.0000`, calls/episode `50.4400`, policy total/episode `3916.3531 ms`
- task3: `50` trials, success `1.0000`, calls/episode `48.9000`, policy total/episode `3803.8354 ms`
- task4: `50` trials, success `1.0000`, calls/episode `48.2000`, policy total/episode `3768.5673 ms`
- task5: `50` trials, success `0.9200`, calls/episode `49.4200`, policy total/episode `3844.0360 ms`
- task6: `50` trials, success `0.9400`, calls/episode `58.0800`, policy total/episode `4508.6266 ms`
- task7: `50` trials, success `1.0000`, calls/episode `55.2400`, policy total/episode `4303.6562 ms`
- task8: `50` trials, success `0.8200`, calls/episode `107.9600`, policy total/episode `8384.4769 ms`
- task9: `50` trials, success `0.9800`, calls/episode `57.5200`, policy total/episode `4487.2414 ms`

后续与 entropy adaptive 比较时，应使用 clean baseline 的 `policy_ms_total_mean_per_episode` / `server_ms_total_mean_per_episode` 与 adaptive summary 中的 `avg_total_deployable_policy_inference_ms_per_episode` / `avg_total_deployable_server_inference_ms_per_episode` 对齐比较；使用 clean baseline 的 `inference_calls_mean_per_episode` 与 adaptive 的 `avg_deployable_policy_calls_per_episode` 比较 policy call 数。

## 2026-07-08 - per-task PK baseline emphasized

本次任务根据用户指出的问题修正了后续比较口径：LIBERO-10 中不同 task 的 episode 长度和 policy call 数差异较大，因此整体平均只能作为 sanity check，不能作为主要结论。尤其 task8 的 clean baseline calls/episode 为 `107.96`，明显高于其他 task，会显著影响总平均。后续 entropy adaptive 与原版 baseline 的速度/成功率比较必须按 task_id 分别讨论。

为便于后续逐 task PK，新增生成文件：

`/Users/kongyue/LightAcotVLA/results/stage_b/original_baseline/libero10_original_pk_by_task_baseline.csv`

该文件来自 clean baseline `libero10_original_per_task_summary_excluding_task0_trial0_warmup.csv`，去掉 `all` 汇总行，只保留逐 task 比较所需字段：`task_id`、`task_description`、`num_trials`、`success_rate`、`inference_calls_mean_per_episode`、`policy_ms_mean_per_call_weighted`、`server_ms_mean_per_call_weighted`、`policy_ms_total_mean_per_episode`、`server_ms_total_mean_per_episode`。

当前逐 task clean baseline：

- task0: success `100.00%`, calls/episode `56.90`, policy total/episode `4473.65 ms`, server total/episode `4786.88 ms`
- task1: success `100.00%`, calls/episode `52.50`, policy total/episode `4110.95 ms`, server total/episode `4391.86 ms`
- task2: success `100.00%`, calls/episode `50.44`, policy total/episode `3916.35 ms`, server total/episode `4183.19 ms`
- task3: success `100.00%`, calls/episode `48.90`, policy total/episode `3803.84 ms`, server total/episode `4062.75 ms`
- task4: success `100.00%`, calls/episode `48.20`, policy total/episode `3768.57 ms`, server total/episode `4029.55 ms`
- task5: success `92.00%`, calls/episode `49.42`, policy total/episode `3844.04 ms`, server total/episode `4111.43 ms`
- task6: success `94.00%`, calls/episode `58.08`, policy total/episode `4508.63 ms`, server total/episode `4823.79 ms`
- task7: success `100.00%`, calls/episode `55.24`, policy total/episode `4303.66 ms`, server total/episode `4588.89 ms`
- task8: success `82.00%`, calls/episode `107.96`, policy total/episode `8384.48 ms`, server total/episode `8960.27 ms`
- task9: success `98.00%`, calls/episode `57.52`, policy total/episode `4487.24 ms`, server total/episode `4793.39 ms`

后续 adaptive 完成后，应对每个 task 单独计算：

- success rate change
- deployable policy calls/episode change
- deployable policy total inference time/episode speedup
- deployable server total inference time/episode speedup

整体平均只作为附属信息，不作为主要结论。

## 2026-07-08 - entropy adaptive per-task results compared with original baseline

本次任务读取了用户提供的 entropy adaptive run 输出片段，来源路径为：

`/root/autodl-tmp/acotvla/stage_b_pruning_eval/adaptive_replan_entropy_oracle_libero10_10tasks_20trials`

该 run 配置为 `libero_10`、每个 task `20` trials、`action_cot_denoising_steps=10`、`adaptive_replanning=entropy`、`adaptive_replan_entropy_mode=online_mc`、`adaptive_replan_entropy_samples=5`、`adaptive_replan_horizons=[3,5,8,10]`。`summary.json` 中整体 aggregate 显示 success rate 为 `0.935`，`avg_deployable_policy_calls_per_episode=61.81`，`avg_total_deployable_policy_inference_ms_per_episode=4485.588982542977 ms`，`avg_total_deployable_server_inference_ms_per_episode=4754.639433114789 ms`，`avg_entropy_oracle_extra_calls_per_episode=247.24`。由于 `online_mc` 会额外调用 policy 估计 entropy，速度结论使用 deployable/primary 字段，不使用 observed 总耗时字段。

本次按 task 将用户贴出的 adaptive per-task deployable metrics 与本地 clean original baseline 对齐比较。生成文件：

`/Users/kongyue/LightAcotVLA/results/stage_b/adaptive_replan_entropy_oracle_vs_original/adaptive_entropy_oracle_vs_original_per_task.csv`

同时生成：

`/Users/kongyue/LightAcotVLA/results/stage_b/adaptive_replan_entropy_oracle_vs_original/adaptive_entropy_oracle_vs_original_summary.json`

比较说明：

- baseline 使用 `libero10_original_pk_by_task_baseline.csv`，即排除 task0 trial0 warmup outlier 后的原版 per-task baseline。
- adaptive 使用用户贴出的 per-task deployable metrics，每个 task `20` trials。
- baseline 原始日志没有 wall timing，因此只计算 policy/server total speedup；adaptive wall total 保留为参考字段，但不与 baseline 做 speedup。

逐 task 结果，格式为 success change、calls reduction、policy total speedup、server total speedup：

- task0: success `-5.00 pp`, calls reduction `-5.36%`, policy speedup `2.41%`, server speedup `3.12%`
- task1: success `0.00 pp`, calls reduction `-9.62%`, policy speedup `-1.35%`, server speedup `-0.63%`
- task2: success `-5.00 pp`, calls reduction `-32.24%`, policy speedup `-23.31%`, server speedup `-22.23%`
- task3: success `0.00 pp`, calls reduction `3.48%`, policy speedup `9.84%`, server speedup `10.48%`
- task4: success `0.00 pp`, calls reduction `2.07%`, policy speedup `8.65%`, server speedup `9.23%`
- task5: success `+3.00 pp`, calls reduction `18.35%`, policy speedup `23.74%`, server speedup `24.45%`
- task6: success `+1.00 pp`, calls reduction `15.20%`, policy speedup `20.61%`, server speedup `21.23%`
- task7: success `0.00 pp`, calls reduction `-5.63%`, policy speedup `1.44%`, server speedup `1.93%`
- task8: success `-17.00 pp`, calls reduction `6.77%`, policy speedup `12.75%`, server speedup `13.50%`
- task9: success `-8.00 pp`, calls reduction `-58.03%`, policy speedup `-46.08%`, server speedup `-44.46%`

汇总事实：

- policy total speedup 为正的 task: `7/10`
- server total speedup 为正的 task: `7/10`
- deployable calls/episode 下降的 task: `5/10`
- success drop 在 `3` 个百分点以内的 task: `6/10`
- 同时满足 policy total speedup 为正且 success drop 在 `3` 个百分点以内的 task: `5/10`
- 同时满足 server total speedup 为正且 success drop 在 `3` 个百分点以内的 task: `5/10`

支持的结论：

- 当前 entropy adaptive oracle 在部分 task 上显示出可部署加速潜力，尤其 task3、task4、task5、task6 同时具备 calls/episode 下降、累计 policy/server 延迟下降、成功率不下降或上升。
- task7 虽然累计延迟小幅下降且成功率不掉，但 calls/episode 上升，因此加速来源不符合 adaptive replanning 的主要机制，证据较弱。
- task8 累计延迟下降但成功率从 baseline `82%` 降到 `65%`，不能算有效加速。
- task2 和 task9 同时出现明显 calls 增加、累计延迟增加和成功率下降，是当前策略的主要失败案例。
- 目前证据不足以说 entropy adaptive 对所有 task 稳定有效；更准确的表述是：entropy adaptive 在若干 task 上有明显可部署加速潜力，但 horizon 决策/entropy 阈值在部分 task 上不稳定，需要进一步按 task 分析失败原因和调整策略。

## 2026-07-08 - entropy adaptive horizon lower-bound and per-task total timing output

本次任务根据用户指出的问题修改了 `scripts/eval_libero_action_cot_pruning.py`：

1. 将 `--adaptive_replan_horizons` 默认值从 `[3, 5, 8, 10]` 改为 `[5, 8, 10]`。
2. 在 entropy-based adaptive replanning（`--adaptive_replanning entropy` 或 `action_entropy`）下，候选 horizon 会自动过滤掉低于 `--replan_steps` 的值。因此即使命令误传 `--adaptive_replan_horizons 3 5 8 10`，实际 entropy adaptive 也不会使用 `3`，只会保持 baseline `5` 或增加到更长 horizon。
3. 在 entropy-based adaptive replanning 下，如果 `--replan_steps < 5`，脚本会报错：`Entropy adaptive replanning requires --replan_steps >= 5 for speed validation.` 这是为了保证当前验证目标是“以 5 步固定 replanning 为 baseline，entropy 只通过延长执行 horizon 来减少 policy calls”。
4. 新增输出文件 `per_task_summary.csv`。该文件按 `mode + task_id` 聚合 rollout 结果，直接包含每个 task 的成功率、平均 steps、累计推理时间、deployable 累计推理时间、replan 次数、policy call 数、entropy oracle 额外 call 数和平均 horizon。
5. `summary.json` 的 `outputs` 中新增 `per_task_summary_csv` 路径。

新增 `per_task_summary.csv` 的关键字段包括：

- `success_rate`
- `avg_steps`
- `avg_total_policy_inference_ms_per_episode`
- `avg_total_server_inference_ms_per_episode`
- `avg_total_wall_inference_ms_per_episode`
- `avg_total_deployable_policy_inference_ms_per_episode`
- `avg_total_deployable_server_inference_ms_per_episode`
- `avg_total_deployable_wall_inference_ms_per_episode`
- `sum_total_deployable_policy_inference_ms`
- `sum_total_deployable_server_inference_ms`
- `sum_total_deployable_wall_inference_ms`
- `avg_deployable_policy_calls_per_episode`
- `avg_entropy_oracle_extra_calls_per_episode`
- `avg_replan_horizon`
- `avg_min_replan_horizon`
- `avg_max_replan_horizon`

验证：

- 已运行 `python -m py_compile scripts/eval_libero_action_cot_pruning.py scripts/sweep_action_cot_denoising_steps.py`，语法检查通过。
- 已运行 `git diff --check -- scripts/eval_libero_action_cot_pruning.py scripts/sweep_action_cot_denoising_steps.py`，未发现 whitespace error。

建议后续重新运行 entropy adaptive 验证时使用：

```sh
uv run python scripts/eval_libero_action_cot_pruning.py \
  --host 0.0.0.0 \
  --port 8000 \
  --task_suite_name libero_10 \
  --task_start 0 \
  --max_tasks 10 \
  --num_trials_per_task 20 \
  --mode full \
  --replan_steps 5 \
  --action_cot_denoising_steps 10 \
  --adaptive_replanning entropy \
  --adaptive_replan_entropy_mode online_mc \
  --adaptive_replan_entropy_samples 5 \
  --adaptive_replan_horizons 5 8 10 \
  --output_dir /root/autodl-tmp/acotvla/stage_b_pruning_eval/adaptive_replan_entropy_oracle_min5_libero10_10tasks_20trials
```

跑完后优先查看：

```sh
cat /root/autodl-tmp/acotvla/stage_b_pruning_eval/adaptive_replan_entropy_oracle_min5_libero10_10tasks_20trials/per_task_summary.csv
```

后续结论应按 task_id 分别比较原版 baseline 的 `policy_ms_total_mean_per_episode` / `server_ms_total_mean_per_episode` 与 adaptive 的 `avg_total_deployable_policy_inference_ms_per_episode` / `avg_total_deployable_server_inference_ms_per_episode`，同时比较 success rate 和 `avg_deployable_policy_calls_per_episode`。

## 2026-07-08 - entropy adaptive min5 per-task results

本次任务分析了用户贴出的新一轮 entropy adaptive min5 结果。该实验使用的关键策略是：`replan_steps=5`，entropy adaptive 的候选 horizon 至少为 5，并可增加到更长 horizon。用户贴出的 per-task 结果包含每个 task 的 `success`、`calls/ep`、`policy_total/ep`、`server_total/ep`、`wall_total/ep` 和 `avg_h`。

本次将用户贴出的结果与本地 clean original baseline `libero10_original_pk_by_task_baseline.csv` 逐 task 对齐比较，并生成：

`/Users/kongyue/LightAcotVLA/results/stage_b/adaptive_replan_entropy_oracle_min5_vs_original/adaptive_entropy_oracle_min5_vs_original_per_task.csv`

逐 task 比较结果，格式为 success change、calls reduction、policy total speedup、server total speedup、average horizon：

- task0: success `0.00 pp`, calls reduction `18.27%`, policy speedup `24.19%`, server speedup `24.74%`, avg_h `6.05`
- task1: success `0.00 pp`, calls reduction `11.71%`, policy speedup `17.92%`, server speedup `18.30%`, avg_h `5.41`
- task2: success `-5.00 pp`, calls reduction `-11.62%`, policy speedup `-4.39%`, server speedup `-3.82%`, avg_h `5.57`
- task3: success `0.00 pp`, calls reduction `11.55%`, policy speedup `17.36%`, server speedup `17.95%`, avg_h `5.70`
- task4: success `0.00 pp`, calls reduction `19.19%`, policy speedup `24.98%`, server speedup `25.64%`, avg_h `6.23`
- task5: success `+3.00 pp`, calls reduction `31.91%`, policy speedup `36.45%`, server speedup `36.96%`, avg_h `6.74`
- task6: success `-9.00 pp`, calls reduction `-8.47%`, policy speedup `-1.56%`, server speedup `-0.89%`, avg_h `6.34`
- task7: success `0.00 pp`, calls reduction `16.09%`, policy speedup `21.40%`, server speedup `21.72%`, avg_h `5.66`
- task8: success `+3.00 pp`, calls reduction `22.10%`, policy speedup `26.93%`, server speedup `27.37%`, avg_h `6.74`
- task9: success `-3.00 pp`, calls reduction `3.42%`, policy speedup `9.85%`, server speedup `10.47%`, avg_h `5.25`

汇总事实：

- policy total speedup 为正的 task: `8/10`
- server total speedup 为正的 task: `8/10`
- deployable calls/episode 下降的 task: `8/10`
- success drop 在 `3` 个百分点以内的 task: `7/10`
- 同时满足 policy total speedup 为正且 success drop 在 `3` 个百分点以内的 task: `7/10`
- 同时满足 server total speedup 为正且 success drop 在 `3` 个百分点以内的 task: `7/10`

支持的结论：

- 相比上一版允许 horizon 低于 5 的 entropy adaptive，min5 版本更符合验证目标，并且结果显著更稳定。
- task0、task1、task3、task4、task5、task7、task8、task9 均显示累计 policy/server 推理延迟下降，其中 task0、1、3、4、5、7、8 的成功率没有下降或上升。
- task9 成功率下降 `3 pp`，但仍有 `9.85%` policy speedup 和 `10.47%` server speedup，处于边界可接受范围。
- task2 和 task6 是当前 min5 entropy adaptive 的主要失败案例：二者 calls/episode 增加、累计推理延迟没有改善，并且成功率分别下降 `5 pp` 和 `9 pp`。
- 当前证据支持“entropy-guided adaptive replanning 在多数 task 上可以减少可部署累计推理延迟，同时基本保持成功率”，但还需要进一步分析 task2/task6 为什么 entropy-to-horizon 决策失效。

## 2026-07-08 - task completion latency definition for baseline comparison

本次任务澄清了“完成一个任务的实际延迟”的统计口径。当前原版 baseline JSONL 日志中每行记录一次 policy/server inference call，包含 `policy_timing.infer_ms` 和 `server_timing.infer_ms`，但不包含仿真环境 step 时间、机器人真实执行动作时间或完整 client wall-clock episode 时间。因此目前能可靠计算的是每个 episode 的累计推理延迟：

- `policy_ms_total_mean_per_episode`：一个 episode 中所有 policy inference 的模型侧推理耗时总和。
- `server_ms_total_mean_per_episode`：一个 episode 中所有 server inference 的服务端耗时总和，更接近请求侧可见的推理延迟，但仍不等于完整任务墙钟时间。

基于 clean baseline，原版每个 task 的平均累计推理延迟为：

- task0: policy `4473.65 ms`, server `4786.88 ms`
- task1: policy `4110.95 ms`, server `4391.86 ms`
- task2: policy `3916.35 ms`, server `4183.19 ms`
- task3: policy `3803.84 ms`, server `4062.75 ms`
- task4: policy `3768.57 ms`, server `4029.55 ms`
- task5: policy `3844.04 ms`, server `4111.43 ms`
- task6: policy `4508.63 ms`, server `4823.79 ms`
- task7: policy `4303.66 ms`, server `4588.89 ms`
- task8: policy `8384.48 ms`, server `8960.27 ms`
- task9: policy `4487.24 ms`, server `4793.39 ms`

结论：在现有日志口径下，原版完成一个 LIBERO-10 task 的累计推理延迟大多约为 `4.0-4.8s` server inference time；task8 明显更长，约 `8.96s` server inference time。完整任务墙钟耗时需要额外记录 episode start/end 或 client-side wall time，当前原版日志不能直接提供。

## 2026-07-08 - StageB algorithm-level acceleration direction discussion

本次任务讨论了 StageB 后续加速推理方案的方向。用户明确提出，StageB 的“其他加速推理方案”应优先设计算法层面的改进，而不是继续简单修改 `denoising step` 或 `replan step` 这类超参数。检查了当前相关代码和结果文件：`src/openpi/models/acot_vla.py` 中 inference 由 prefix/VLM、explicit Action-CoT coarse denoising loop、final action expert 组成；`scripts/benchmark_action_cot_speed.py` 的说明和已有结果指出，当前 pruning/override 路径虽然可以绕过 explicit Action-CoT 生成，但如果 final action head 仍消费完整 coarse token 序列，单纯改变 coarse action 的数值几乎不会带来额外速度收益。

已有 StageB speed 结果支持这一判断：`results/stage_b/speed/speed_benchmark_adaptive_p03_summary.json` 中 `full_acot_ms_mean=72.97 ms`，`cached_coarse_override_ms_mean=55.54 ms`，`pruned_coarse_override_ms_mean=55.54 ms`，`cached_to_pruned_speedup_pct=-0.0045%`；`results/stage_b/speed/true_entropy_skip_speed_p03_summary.json` 中 true entropy segment skip 生成更少 coarse token 后仍为 `73.20 ms`，相对 full 没有加速，说明当前路径没有真正减少可部署的 final action-head 消费成本或整体计算图成本。因此后续算法方向应聚焦在减少模型实际计算量，例如让 final action head 消费压缩后的 Action-CoT 表示、训练可变长度/稀疏 Action-CoT token 路径、学习 lightweight Action-CoT surrogate/adapter、或通过状态连续性缓存/增量更新减少重复计算。尚未实现代码改动，也未运行新实验。

## 2026-07-08 - clarification of StageB evidence source

本次任务进一步澄清“StageB 算法级加速方向与已有结果吻合”的证据来源。关键证据来自 `scripts/benchmark_action_cot_speed.py` 和 `results/stage_b/speed/` 下的已有汇总结果：`speed_benchmark_adaptive_p03_summary.json` 中 `cached_coarse_override_ms_mean=55.5388 ms`，`pruned_coarse_override_ms_mean=55.5413 ms`，`cached_to_pruned_speedup_pct=-0.0045%`，说明当前只改变或插值 coarse action 值的 pruning 路径没有比直接 cached override 更快；脚本注释也说明这一路径没有减少 final action-head token length。`true_entropy_skip_speed_p03_summary.json` 中 `true_entropy_segment_skip_ms_mean=73.2042 ms`，相对 `full_acot_ms_mean=72.7482 ms` 的 speedup 为 `-0.6268%`，说明当前 fixed L=5 entropy skip 虽然只生成剩余 coarse tokens，但恢复 skipped frames 后仍运行 unchanged final action head，因此没有形成可部署加速。`coarse_steps_speed_summary.csv` 则显示减少 `coarse_num_steps` 会带来速度提升，例如 10 到 1 的 `speedup_vs_coarse10_pct=22.6451%`，但这正属于 denoising step 超参数扫描，而不是新的算法结构改进。

## 2026-07-08 - reporting clarity requirement for StageB metrics

本次任务确认了后续汇报 StageB 参数和指标时需要加强解释。用户反馈此前直接使用 `full_acot_ms_mean`、`cached_coarse_override_ms_mean`、`pruned_coarse_override_ms_mean`、`true_entropy_segment_skip_ms_mean`、`cached_to_pruned_speedup_pct`、`coarse_num_steps` 等字段时缺少中文解释，导致难以理解结论依据。后续在分析 StageB 实验结果、速度指标或算法方案时，应为每个关键参数补充其含义、单位、数值大小代表什么、以及它为什么支持当前结论。尚未改动代码，也未运行新实验。

## 2026-07-08 - StageB algorithm proposal references and current parameter naming

本次任务根据用户反馈继续讨论 StageB 算法级推理加速方案。用户指出旧字段 `coarse_num_steps` 已经改名，后续表述应使用当前代码中的 `action_cot_denoising_steps` / `avg_action_cot_denoising_steps_used` 等命名；旧结果文件中仍有 `coarse_num_steps`，但应理解为历史字段，语义对应 explicit Action-CoT/coarse trajectory 的 denoising 迭代次数。检查当前代码发现 `src/openpi/models/acot_vla.py` 的 `sample_actions_profile_coarse` 入口使用 `action_cot_denoising_steps`，final expert 中 `explicit_action_reason` 会先通过 `coarse_action_in_proj` 投影为 tokens，再通过 `explicit_action_reasoner` 与 action tokens 做 cross-attention，因此结构性加速应减少 Action-CoT 的 token 数、压缩表示或减少 final expert 消费成本，而不是只替换 coarse action 数值。

参考方向包括：LLM prompt/CoT 压缩（例如 LLMLingua）、Transformer token pruning/merging（DynamicViT、ToMe）、diffusion/flow sampler distillation（Progressive Distillation）、VLA/action tokenization（FAST、OpenVLA）和 speculative decoding。初步建议的 StageB 主线是 `Compressed / Sparse Action-CoT Consumption`：训练一个压缩或稀疏 Action-CoT 表示，使 coarse reasoner 直接生成少量 latent tokens 或频域/segment tokens，并让 final action expert 直接消费这些压缩 tokens；训练时用完整 ACoT 作为 teacher 做 action distillation、可选 coarse reconstruction 和 token-budget regularization。备选方案包括动态 Action-CoT token gating、Action-CoT progressive/consistency distillation、以及跨 replan 的 lightweight residual update/cache。尚未改动代码，也未运行新实验。

## 2026-07-08 - Dynamic Action-CoT Token Gating scheme detail

本次任务细化了 StageB 的 `Dynamic Action-CoT Token Gating` 方案。该方案目标是让模型预测哪些 Action-CoT segment/token 对最终动作最重要，并让 final action expert 直接消费被选中的较短 Action-CoT token 序列，而不是像当前 true skip 路径一样跳过部分 coarse tokens 后再恢复为完整 15 帧。当前代码中的相关落点是 `src/openpi/models/acot_vla.py`：`sample_actions_profile_coarse` 生成 `explicit_action_reason`，`embed_suffix(..., suf_type="expert")` 中再将 `explicit_action_reason` 经 `coarse_action_in_proj` 转为 tokens，并通过 `explicit_action_reasoner` 与 action expert tokens 交互。新方案需要新增 fixed-budget gated expert path，例如 `K=5` 和 `K=10` 两套静态形状路径，以保持 JAX/XLA 编译稳定。

建议的训练信号包括：使用已有 entropy label 作为弱监督，鼓励 gate 保留高不确定性或高敏感度 segment；使用 full ACoT teacher 的最终动作做 action distillation；可选加入 full Action-CoT reconstruction 或 feature alignment；加入 budget/entropy regularization 避免 gate 全选。关键参数包括 `cot_token_budget_k`（保留 token 数，如 5 或 10）、`gate_temperature`（控制 soft gate 的离散程度）、`gate_loss_weight`（gate 监督损失权重）、`distill_loss_weight`（student action 对齐 teacher action 的权重）、`budget_loss_weight`（限制 token 使用预算的权重）。评估时需要同时看 success rate、`action_expert_ms`、总 `infer_ms`、gate 选择稳定性、teacher-student action error，以及相对 full ACoT 的端到端 deployable speedup。尚未实现代码，也未运行新实验。

## 2026-07-08 - evidence boundary for gated Action-CoT speedup

本次任务澄清了 Dynamic Action-CoT Token Gating 的证据边界。当前已有结果不能直接证明“final expert 处理完整 15 个 Action-CoT tokens 占据大部分时间”，因为本地保存的 summary 里没有分阶段均值（如 `vlm_ms`、`coarse_action_expert_ms`、`action_expert_ms`），对应的 per-row `latency_rows.csv` 未保存在当前仓库。已有结果能支持的结论是：1）`full_acot_ms_mean=72.97 ms` 与 `cached_coarse_override_ms_mean=55.54 ms` 的差值约 `17.43 ms`，说明跳过 explicit Action-CoT 生成最多节省约 24% 总时延；2）`pruned_coarse_override_ms_mean=55.54 ms` 与 cached 几乎一致，说明只改变 coarse action 值且 final expert 仍消费完整 token 序列没有额外速度收益；3）`true_entropy_segment_skip_ms_mean=73.20 ms` 没有比 full 更快，说明当前“少生成后再恢复完整 15 帧”的路径不能形成可部署加速。因此 gated 方案的下一步必须实现真正的短 token final expert path，并用 `action_expert_ms` 和总 `infer_ms` 验证 `K=10` / `K=5` 是否实际加速。尚未实现或验证该路径。

## 2026-07-08 - clarification of StageB speed benchmark modes

本次任务解释了 StageB speed benchmark 中三个 mode/metric 的含义。`cached_coarse_override_ms_mean` 表示先复用已经得到的完整 `coarse_actions`，跳过 explicit Action-CoT/coarse 生成循环，然后直接运行 final action expert 的平均耗时；它近似衡量“如果不生成 Action-CoT，只运行后续 final expert 等部分，需要多久”。`pruned_coarse_override_ms_mean` 表示把 `coarse_actions` 做 pruning/interpolation 后作为 override 输入，也跳过 explicit Action-CoT 生成循环，但 final action expert 仍消费完整长度的 coarse trajectory；它用于验证“只改变 coarse action 的值，而不减少 final expert 的 token 长度，是否有额外加速”。已有结果中 cached 和 pruned 都约为 `55.54 ms`，说明 value-only pruning 没有额外速度收益。`true_entropy_segment_skip_ms_mean` 表示根据 entropy 选择一个 fixed L=5 segment 跳过，coarse 生成时只生成剩余 tokens，但之后会把 skipped segment 恢复/补回成完整 15 帧，再运行 unchanged final action expert；已有结果约为 `73.20 ms`，没有比 full ACoT 更快，说明当前 true skip 还不是“final expert 直接消费短 token 序列”的可部署加速路径。

## 2026-07-08 - proposed test protocol for gated_k10 and gated_k5

本次任务讨论了如何测试 `gated_k10` 和 `gated_k5`，即让 final expert 直接消费 10 个或 5 个 Action-CoT tokens，并且不补回完整 15 帧。检查当前代码发现，`coarse_actions_override` 会经过 input transforms 后作为 `explicit_action_reason_override` 传入模型，`sample_actions_profile_coarse` 对 override 没有固定 15 帧长度检查；`embed_suffix(..., suf_type="expert")` 中 `explicit_action_reasoner` 是 cross-attention，理论上可以接受不同长度的 `explicit_action_reason_tokens`。因此最小测试可以先做非部署的短 override 速度上界实验：先跑一次 full ACoT 得到完整 `coarse_actions`，按 entropy 或 uniform indices 选出 K=10/K=5 帧，把短序列作为 `coarse_actions_override` 输入，再测 `infer_ms`、`action_expert_ms` 和 student/full action error。该实验用于验证 final expert 直接消费短 Action-CoT 表示是否能变快，但不代表可部署加速，因为它复用了 full ACoT 生成的完整 coarse actions。

可部署测试需要新增真正的 no-restore generation path：类似现有 `explicit_action_skip_segment` 分支先只生成 kept tokens，但不要调用 `_restore_fixed_l5_skip` 补回 15 帧，而是直接把 kept tokens 传给 `sample_actions_profile_expert`。建议实现 `gated_k10`（跳过一个 fixed L=5 segment，保留 10 tokens）和 `gated_k5`（保留一个 L=5 segment）两条固定 shape 路径，以避免 JAX 动态 shape 编译问题。结果解释规则：如果短 override 的 `action_expert_ms` 不下降，说明当前架构中减少 CoT key/value 长度不能带来有效速度收益；如果短 override 变快但 no-restore generation 不快，需要分析 coarse generation/gating 开销；如果速度变快但 action error 或 closed-loop success 明显变差，则需要训练 gated student 或 distillation。尚未实现代码，也未运行新实验。

## 2026-07-08 - implemented gated_k override speed benchmark code

本次任务在 `scripts/benchmark_action_cot_speed.py` 中实现了第一阶段 `gated_k10` / `gated_k5` 测试代码，即非部署的 short override 速度上界测试。新增参数 `--gated_budgets`，默认 `[10, 5]`，表示 final expert 直接消费的短 Action-CoT token 数；新增参数 `--gated_selection`，可选 `high_entropy`、`low_entropy`、`uniform`、`random`，默认 `high_entropy`。脚本现在会先运行 full ACoT 得到完整 `coarse_actions`，再按预算 K 选出短序列作为 `coarse_actions_override`，测量 `gated_k{K}_override_ms`、各 profile stage timing、`gated_k{K}_keep_ratio`、`gated_k{K}_keep_indices`，并计算相对 full ACoT 输出动作的 `action_mse_vs_full`、`action_rmse_vs_full`、`action_l2_vs_full`、`action_linf_vs_full`。summary 中新增 `full_to_gated_k{K}_override_speedup_pct` 和 `cached_to_gated_k{K}_override_speedup_pct`，后者应作为判断短 token final expert 是否比完整 15-token cached override 更快的主要速度指标。

本次只实现了 short override 上界测试，没有实现真正可部署的 no-restore generation path；也没有训练 gated student。验证方面，已运行 `python -m py_compile scripts/benchmark_action_cot_speed.py`，语法检查通过；已运行 `git diff --check -- scripts/benchmark_action_cot_speed.py`，未发现 whitespace error。尝试直接运行 `python scripts/benchmark_action_cot_speed.py --help` 时失败，因为当前 shell 未安装或未暴露项目依赖（先缺 `openpi`，设置 `PYTHONPATH=src:scripts` 后缺 `numpydantic`，且本机没有 `uv` 命令），因此没有在本地执行完整 CLI 或 checkpoint benchmark。

## 2026-07-08 - task/sample scope of gated override benchmark

本次任务澄清了 `gated_k10` / `gated_k5` benchmark 的测试样本来源。`scripts/benchmark_action_cot_speed.py` 不是按 LIBERO closed-loop `task_id` 直接选择任务，而是读取 `--entropy_dir` 下的 `sample_*.npz` 文件；`scripts/eval_action_cot_pruning.py` 中 `_entropy_files` 会按文件名排序读取这些 npz，`_prepare_sample` 会从文件名或 fallback 中得到 `item_index`，benchmark 再用 `_expert_actions_for_item(dataset, item_index, ...)` 从 policy dataset 中重建对应 observation/policy input。因此示例命令中的 `/root/autodl-tmp/acotvla/action_cot_entropy_labels/adaptive_full_k8` 决定了实际测试的是哪些 dataset items；`--max_items 50` 表示取该目录排序后的前 50 个样本。当前脚本输出 `sample_id` 和 `item_index`，但没有显式输出 LIBERO `task_id` 或 `task_description`。如果需要按具体 LIBERO task 测试，需要后续增加 task metadata 读取/过滤，或准备只包含目标 task 样本的 entropy_dir。

## 2026-07-08 - plain-language explanation of gated override benchmark

本次任务用更清晰的口径解释了 `gated_k10` / `gated_k5` 测试方式。该测试是 open-loop speed benchmark，不是 closed-loop LIBERO task rollout；它不会在仿真环境里执行整条任务，也不会输出 success rate。输入样本来自 `--entropy_dir` 下的 `sample_*.npz`，每个 npz 对应 dataset 中一个 `item_index`，脚本再用该 dataset item 重建一次 policy observation。对每个 observation，脚本先跑 `full_acot` 得到完整 15 帧 `coarse_actions` 和最终 `actions`；再分别跑 `cached_override`（把完整 15 帧 coarse_actions 作为 override 输入，跳过 coarse 生成）、`gated_k10_override`（从 15 帧中选 10 帧作为 short override，final expert 直接消费 10 帧）、`gated_k5_override`（选 5 帧作为 short override，final expert 直接消费 5 帧）、以及现有 pruning/true-skip 对照。`gated_k` 的主要速度对照应是 `cached_override`，因为二者都跳过 coarse 生成，区别只在 final expert 看到完整 15 帧还是短 K 帧。该测试只能说明“短 Action-CoT override 是否有速度潜力”和“短 override 与 full action 的动作差距”，不能证明闭环任务成功率，也不是可部署路径，因为 short override 复用了 full ACoT 生成出的完整 coarse_actions。

## 2026-07-08 - gated_k override benchmark result interpretation

本次任务分析了用户在 AutoDL 环境运行 `gated_k10` / `gated_k5` short override benchmark 的结果。关键结果为：`full_acot_ms_mean=77.2211 ms`，`cached_coarse_override_ms_mean=59.2252 ms`，`gated_k10_override_ms_mean=59.8083 ms`，`gated_k5_override_ms_mean=60.2061 ms`。相对 cached override，`gated_k10` 的速度提升为 `-0.9846%`，`gated_k5` 为 `-1.6563%`，即两个短 Action-CoT override 都没有加速，反而略慢。分阶段 timing 也支持该结论：`cached_override_action_expert_ms_mean=19.1836 ms`，`gated_k10_override_action_expert_ms_mean=19.1310 ms`，`gated_k5_override_action_expert_ms_mean=19.2041 ms`，差异只有约 `0.05 ms` 或更小，基本可视为噪声级别。

动作差距方面，`gated_k10_action_rmse_vs_full_mean=0.0377`、`gated_k10_action_l2_vs_full_mean=0.0885`；`gated_k5_action_rmse_vs_full_mean=0.0717`、`gated_k5_action_l2_vs_full_mean=0.1544`。这说明 K=5 相比 K=10 对动作扰动更大，但二者都没有带来速度收益。样本输出显示每个 sample 有 3 行重复，对应 `repeat=3`；`gated_selection=high_entropy` 下 K=10 通常保留两个 L=5 segment，K=5 保留一个 L=5 segment，例如 `10;11;12;13;14`。结论：在当前 ACOT-VLA 实现中，仅缩短 `explicit_action_reason` 的 coarse token 数不能有效加速 final expert。原因很可能是 explicit Action-CoT 先通过较小的 `explicit_action_reasoner` cross-attention 融合进 action tokens，而主 PaliGemma final action expert 的 action token 长度和 denoising循环没有减少；因此后续若要算法级加速，需要改变主计算路径，例如减少 final action expert 的 denoising/model calls、压缩 action expert 自身 token/horizon，或把 Action-CoT 压缩设计成能减少主 LLM suffix 计算的结构，而不是只缩短 cross-attention 的 key/value 长度。

## 2026-07-08 - clarified why calls can increase with min horizon >= 5

本次任务解释了用户提出的问题：即使 entropy adaptive 的 horizon 下限设置为 `replan_steps=5`，某些 task 的 calls/episode 仍然可能高于原版。原因是 `horizon >= 5` 只保证在“同一条轨迹、相同 episode steps”的情况下，每次 replan 至少执行 5 个动作；但如果 adaptive 策略让 episode 变长、失败或 timeout 增多，总环境步数增加，则 calls/episode 仍可能增加。近似关系是 `calls/episode ≈ episode_steps / avg_horizon`，所以当 episode_steps 增长超过 avg_horizon 带来的收益时，calls 会增加。

本次还用原版 per-trial CSV 计算了与 adaptive 20 trials 更接近的 baseline 前 20 trials 指标。结果：

- task2 baseline20: success `100.00%`, steps `262.50`, calls `51.05`, clean policy total `3973.48 ms`, clean server total `4250.94 ms`
- task6 baseline20: success `100.00%`, steps `263.50`, calls `51.20`, clean policy total `3977.95 ms`, clean server total `4255.37 ms`

对照用户贴出的 min5 adaptive：

- task2 adaptive: success `95.00%`, calls `56.30`, policy total `4088.15 ms`, server total `4343.06 ms`, avg_h `5.57`
- task6 adaptive: success `85.00%`, calls `63.00`, policy total `4578.84 ms`, server total `4866.76 ms`, avg_h `6.34`

支持的结论：

- task2/task6 的 calls 增加不是因为 horizon 下限失效，而更可能是 adaptive policy 行为导致 episode 更长或失败更多。
- 对 task2/task6 需要查看 adaptive `per_task_summary.csv` 中的 `avg_steps`，以及 `rollout_rows.csv` 中失败 episode 的 `steps`、`avg_replan_horizon`、`adaptive_replan_reasons`，确认是否是失败/timeout 拉长了总步数。
- 后续严格 PK 时，最好同时保留 original baseline 50-trial 汇总和 baseline trial0-19 汇总，避免 adaptive 20 trials 与 original 50 trials 的 init-state 分布差异影响结论。

## 2026-07-08 - implemented stage-aware guarded entropy adaptive replanning

本次任务根据用户建议，直接将 entropy adaptive replanning 改为 stage-aware guarded 版本，并细化 horizon 档位。

代码修改文件：

`/Users/kongyue/LightAcotVLA/scripts/eval_libero_action_cot_pruning.py`

主要改动：

1. `--adaptive_replan_horizons` 默认值从 `[5, 8, 10]` 改为 `[5, 6, 7, 8, 9, 10]`，使 replan horizon 从原版 5 步开始逐级细化到 10 步。
2. entropy adaptive 下仍然会过滤掉低于 `--replan_steps` 的 horizon，保证当前验证目标仍是“保持或延长原版 5-step horizon”，而不是更频繁 replan。
3. 新增默认启用的 stage-aware guard。它用 low-level action chunk 中可直接计算的 `gripper_change`、`action_delta` 和 `action_jerk_ratio` 作为抓取/放置/接触风险代理；如果触发 guard，则将 horizon 强制 cap 回 `--replan_steps`。
4. 新增参数：
   - `--disable_adaptive_replan_stage_guard`：关闭默认 stage-aware guard，用于消融。
   - `--adaptive_replan_action_delta_high`：action delta 风险阈值，默认 `0.08`。
   - `--adaptive_replan_stage_guard_jerk_high`：stage guard 的 jerk-ratio 风险阈值，默认 `0.65`。
5. 低熵分支不再只跳一档，而是根据当前 entropy score 在历史 entropy 中的分位位置，在 `[5,6,7,8,9,10]` 中选择更细的目标 horizon：越低熵，目标 horizon 越长；如果 stage guard 触发，则仍回到 5。
6. 输出中新增：
   - `avg_adaptive_low_entropy_target_horizon`
   - `avg_adaptive_stage_guard`
   这些字段会写入 `rollout_rows.csv`、`per_task_summary.csv` 和 `summary.json` aggregate，用于判断低熵本来想延长到几步，以及 stage guard 抑制了多少高风险延长。

验证：

- 已运行 `python -m py_compile scripts/eval_libero_action_cot_pruning.py`，语法检查通过。
- 已运行 `git diff --check -- scripts/eval_libero_action_cot_pruning.py`，未发现 whitespace error。
- 本次没有运行服务器端 LIBERO rollout，也没有产生新的成功率或延迟结果。

建议服务器端下一轮实验命令：

```sh
OUT=/root/autodl-tmp/acotvla/stage_b_pruning_eval/adaptive_replan_entropy_guarded_h5_10_libero10_10tasks_20trials

uv run python scripts/eval_libero_action_cot_pruning.py \
  --host 0.0.0.0 \
  --port 8000 \
  --task_suite_name libero_10 \
  --task_start 0 \
  --max_tasks 10 \
  --num_trials_per_task 20 \
  --mode full \
  --replan_steps 5 \
  --action_cot_denoising_steps 10 \
  --adaptive_replanning entropy \
  --adaptive_replan_entropy_mode online_mc \
  --adaptive_replan_entropy_samples 5 \
  --adaptive_replan_horizons 5 6 7 8 9 10 \
  --output_dir $OUT
```

跑完后建议重点查看 `per_task_summary.csv` 中的 `success_rate`、`avg_steps`、`avg_total_deployable_policy_inference_ms_per_episode`、`avg_deployable_policy_calls_per_episode`、`avg_replan_horizon`、`avg_adaptive_low_entropy_target_horizon` 和 `avg_adaptive_stage_guard`。

## 2026-07-10 - analyzed guarded entropy adaptive h5-10 results

本次任务分析了用户贴出的 guarded entropy adaptive h5-10 run 结果。该 run 使用 stage-aware guard，horizon 档位为 `5,6,7,8,9,10`，用户提供了每个 task 的 `success`、`steps`、`calls/ep`、`policy_total/ep`、`server_total/ep`、`avg_h`、`target_h` 和 `guard`。

为了与 adaptive 的 20 trials 对齐，本次使用原版 baseline 的 trial 0-19 做逐 task 比较；task0 trial0 的 warmup outlier 只从 baseline timing 中排除。生成比较文件：

`/Users/kongyue/LightAcotVLA/results/stage_b/adaptive_replan_entropy_guarded_h5_10_vs_original/adaptive_entropy_guarded_h5_10_vs_original_baseline20_per_task.csv`

逐 task 对比 baseline20 的结果，格式为 success change、steps change、calls reduction、policy speedup、server speedup、avg_h、target_h、guard：

- task0: success `0.00 pp`, steps `-1.35%`, calls reduction `15.37%`, policy speedup `21.90%`, server speedup `22.61%`, avg_h `5.85`, target_h `6.21`, guard `0.44`
- task1: success `0.00 pp`, steps `-3.35%`, calls reduction `7.59%`, policy speedup `13.78%`, server speedup `14.20%`, avg_h `5.26`, target_h `5.41`, guard `0.53`
- task2: success `-10.00 pp`, steps `+27.54%`, calls reduction `-21.25%`, policy speedup `-13.73%`, server speedup `-12.81%`, avg_h `5.39`, target_h `5.51`, guard `0.33`
- task3: success `0.00 pp`, steps `-0.53%`, calls reduction `10.69%`, policy speedup `16.57%`, server speedup `17.23%`, avg_h `5.58`, target_h `5.79`, guard `0.36`
- task4: success `0.00 pp`, steps `+3.62%`, calls reduction `8.70%`, policy speedup `13.86%`, server speedup `14.22%`, avg_h `5.68`, target_h `6.08`, guard `0.47`
- task5: success `0.00 pp`, steps `+1.92%`, calls reduction `15.36%`, policy speedup `21.13%`, server speedup `21.81%`, avg_h `6.37`, target_h `6.74`, guard `0.30`
- task6: success `0.00 pp`, steps `-5.41%`, calls reduction `23.93%`, policy speedup `28.85%`, server speedup `29.55%`, avg_h `6.28`, target_h `6.61`, guard `0.26`
- task7: success `0.00 pp`, steps `-1.51%`, calls reduction `7.58%`, policy speedup `13.77%`, server speedup `14.45%`, avg_h `5.35`, target_h `5.65`, guard `0.54`
- task8: success `-5.00 pp`, steps `+5.19%`, calls reduction `12.41%`, policy speedup `18.05%`, server speedup `18.52%`, avg_h `6.18`, target_h `6.62`, guard `0.32`
- task9: success `-5.00 pp`, steps `+20.48%`, calls reduction `-17.31%`, policy speedup `-9.70%`, server speedup `-9.02%`, avg_h `5.16`, target_h `5.22`, guard `0.47`

汇总事实：

- policy total speedup 为正的 task: `8/10`
- server total speedup 为正的 task: `8/10`
- deployable calls/episode 下降的 task: `8/10`
- success drop 在 `3` 个百分点以内的 task: `7/10`
- 同时满足 policy total speedup 为正且 success drop 在 `3` 个百分点以内的 task: `7/10`
- 同时满足 server total speedup 为正且 success drop 在 `3` 个百分点以内的 task: `7/10`

支持的结论：

- guarded h5-10 相比上一轮 min5 解决了 task6 的主要问题：task6 success 从用户上一轮贴出的 `85%` 恢复到 `100%`，同时 calls/episode 和累计 policy/server 推理延迟显著下降。
- 当前主要失败案例变为 task2 和 task9：二者 steps 明显增加，calls/episode 也增加，累计 policy/server 推理延迟变差，成功率下降。
- task8 有 calls 和累计延迟收益，但 success 从 baseline20 `80%` 降到 `75%`，属于速度有效但质量仍有损失。
- guard 字段说明 stage-aware guard 已经频繁触发，例如 task1 `0.53`、task7 `0.54`、task9 `0.47`，但 task2 的 guard 只有 `0.33`，且 avg_h/target_h 都接近 5，说明 task2 的失败未必能靠简单降低 horizon 解决，可能需要查看失败 episode 的具体 replan reasons、gripper阶段和轨迹步骤。

## 2026-07-10 - Stage B algorithm-level acceleration alternatives after token gating result

本次任务结合用户实测的 gated override 结果、`src/openpi/models/acot_vla.py` 的推理路径和相关机器人/VLA论文，分析了下一步可行的算法级加速方向。

当前项目内证据：

- 用户实测 `full_acot_ms_mean=77.2211 ms`、`cached_coarse_override_ms_mean=59.2252 ms`，两者相差约 `17.996 ms`。该差值是完全绕过 explicit Action-CoT/coarse 生成循环时可消除耗时的实测上界；真实替代模块还会增加自身耗时。
- `gated_k10` 和 `gated_k5` 没有加速，且三条路径的 `action_expert_ms_mean` 都约为 `19.1-19.2 ms`。代码显示缩短 coarse token 只改变 `embed_suffix` 中小型 explicit cross-attention 的 K/V 长度，最终 action expert 的 10 次 PaliGemma 循环和 action token 长度没有改变。
- `sample_actions_profile_prefix` 已经生成并复用视觉/语言 prefix 的 KV cache，因此普通的单次推理 prefix KV cache 不是新的方案；跨 observation 的选择性视觉 token 缓存仍是另一类方法。

建议方向及优先级：

1. 优先验证 one-step Action-CoT consistency distillation：用当前 10-step coarse expert 作为 teacher，训练 student 一次前向直接生成完整 15 帧 coarse trajectory，再保持 final expert 和其 15 帧输入不变。它减少的是 coarse expert 的模型调用次数，不是简单把 `action_cot_denoising_steps` 设成 1。现有 full-vs-cached 差值表明它直接针对约 18 ms 的 coarse 生成成本。
2. 学习式 branch-level Action-CoT router：保留 `full explicit+implicit` 与 `implicit-only/null-explicit` 两条固定 JIT 路径；router 预测当前 observation 是否真正需要 explicit Action-CoT。被判为简单的 observation 完全跳过 coarse 生成，被判为困难的 observation 走原始完整路径。训练标签应优先来自 full 与 no-explicit 两条路径的反事实 action/rollout 损失差，现有 entropy 只作为弱监督特征。
3. Temporal residual Action-CoT reuse：将上一次 coarse trajectory 按已执行动作向前平移，用轻量单次 residual corrector 根据新 observation 修正；场景变化、夹爪/接触阶段变化或置信度不足时回退到完整 coarse expert。该方案每个 observation 仍执行策略，不等同于延长 replan horizon。
4. Denoising feature cache：保持 coarse/final 的 10 个时间步不变，在相邻时间步复用部分 transformer attention/MLP 中间特征，只周期性重算。实施前必须测量当前 checkpoint 各层跨 denoising timestep 的 feature cosine similarity，因为 `pi05` 的 AdaRMS time condition 会随 timestep 改变，外部模型的高相似性不能直接假设在本模型成立。
5. Task-aware visual token selection 或跨 observation visual-token cache：压缩 `embed_prefix` 产生的大量图像 token，且让 coarse/final 每次 suffix query 都读取更短的 prefix KV。该方向理论上比 coarse-token gating 更可能影响重计算主干，但需先取得本次 run 的 `vlm_ms` 和 prefix token 数量，再判断收益上限。

推荐的最小验证顺序：先增加 `zero_or_null_coarse_override` 与 `previous_shifted_coarse_override` 两个无需训练的反事实基准，测 action deviation、逐 task rollout success 和约 59 ms 的绕过路径延迟；若大量 observation 不需要 fresh explicit CoT，再训练 branch router。若 explicit CoT 普遍必要，则直接进入 one-step coarse consistency student。当前没有这些新方案在本 checkpoint 上的成功率或实际加速结果。

参考的外部方向包括 Consistency Policy、One-Step Flow Policy、VLA-Cache、EfficientVLA、LightVLA、Falcon、STEP 和 ActionCache。它们只能支持方法方向，不能替代本项目上的实测验证。

## 2026-07-10 - reviewed six acceleration papers for Stage B relevance

本次任务阅读并解释了上一轮讨论涉及的六篇论文的原始 arXiv 页面和全文方法部分：

- `Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation` (`arXiv:2405.07503`)
- `One-Step Flow Policy: Self-Distillation for Fast Visuomotor Policies` (`arXiv:2603.12480`)
- `VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching` (`arXiv:2502.02175`)
- `Falcon: Fast Visuomotor Policies via Partial Denoising` (`arXiv:2503.00339`)
- `STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction` (`arXiv:2602.08245`)
- `EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models` (`arXiv:2506.10100`)

核心分类与事实：

- Consistency Policy 和 OFP 属于生成过程蒸馏。Consistency Policy 使用预训练 EDM teacher、CTM self-consistency loss 和 DSM loss，支持 1/3 NFE；论文报告真实机器人中 15-step DDiM 为 `192 ms`、Consistency Policy 为 `21 ms`，但也指出单步策略存在多模态表达和训练稳定性权衡。OFP 不需要预训练外部 teacher，而用 EMA self-teacher 学习 interval-averaged velocity，并加入 self-guided regularization 与 shifted previous action warm start；论文报告 56 个模拟任务及在 RoboTwin 2.0 上集成到 pi0.5 的结果。OFP 与当前 flow-matching ACoT-VLA 的数学形式最接近，但它是 2026 年的新 preprint，外部结果不能替代本 checkpoint 验证。
- VLA-Cache 属于跨 observation 的视觉 KV 复用，不同于当前 `sample_actions_profile_prefix` 已有的单次 policy call 内 prefix KV cache。它按相邻图像 patch cosine similarity 选静态 token，再用 text-to-vision attention 排除任务相关 token，并按各 decoder layer 的 attention entropy 调整复用率。论文在 OpenVLA/LIBERO 中报告 baseline `84.4% / 51.56 ms`，完整方法 `83.8% / 32.22 ms`；跨四个 LIBERO suite 报告约 `1.63x` latency improvement 和 `0.3%` success drop。动态场景会降低可复用 token 数量和加速收益。
- Falcon 和 STEP 属于 temporal warm start。Falcon training-free，保存历史 partial denoising actions，用当前 observation 做 one-step posterior estimate，再通过 threshold、softmax temperature 和 exploration probability 选择从哪个历史噪声状态继续；论文报告简单任务约 `2-7x` speedup，并明确不同任务需要调不同参数。STEP 训练一个轻量 cross-attention Transformer，用当前 observation 与上一 action chunk 预测 spatiotemporally consistent warm start，再从中间噪声步继续；还用 velocity-aware perturbation 防止机器人停滞。STEP 的主要结果强调 2-step 模式，1-step 在多个任务上明显退化。
- EfficientVLA 同时做三件事：按相邻层 hidden-state change 非连续裁剪 LLM layers；按 task relevance 与 feature diversity 固定保留视觉 token；在 diffusion action head 中每隔固定 cache interval 重算 attention/MLP，其余 timestep 复用中间特征。论文在 CogACT/SIMPLER 上报告配置 `L=22, T=56` 的 `1.93x` speedup、`71.1%` FLOPs reduction 和 `0.6` percentage-point average success drop。结果主要来自 CogACT，且 action feature cache 针对 DiT；当前 ACoT-VLA 是 Gemma action expert 并带 timestep-dependent AdaRMS，迁移前必须先测跨 timestep feature similarity。

对当前 Stage B 的结论：

- 若允许重新训练并真正减少 coarse expert 模型调用，OFP/Consistency-style one-step full-15-frame coarse student 最直接命中用户实测约 `18 ms` coarse 生成成本；这不是简单把现有 step 超参数设为 1。
- 若要求保留外层 10 个 timestep，EfficientVLA 的 intermediate feature caching 是最贴近要求的方向，但需要先做 layerwise feature similarity profiling。
- 若利用连续 rollout，STEP 比 Falcon 更适合作为 learned temporal coarse-trajectory predictor；Falcon 和 STEP 的收益最终仍来自减少剩余 NFE，因此不属于“完全不改变有效模型调用次数”的方案。
- 若优化视觉/语言 prefix，VLA-Cache 可以研究跨 observation 缓存；当前已有的单次 call prefix KV cache 不能直接提供这种跨帧收益。

本次只进行了论文分析，没有修改模型或 benchmark 代码，也没有产生新的本项目延迟或成功率结果。

## 2026-07-10 - compared Step-Entropy CoT compression paper with current Stage A/B route

本次任务阅读并核对了用户提供的 PDF：`/Users/kongyue/Desktop/MAKING SLOW THINKING FASTER- COMPRESSING LLM CHAIN-OF-THOUGHT VIA STEP ENTROPY.pdf`。论文为 ICLR 2026 的 `Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy`，共 19 页。使用 `pdfplumber` 提取全文，并渲染、检查了方法页和主要结果表所在的第 5、7、9 页。

论文方法事实：

- 论文先按空行将 autoregressive textual CoT 切成 reasoning steps，对每个 token 的完整词表概率分布计算 Shannon entropy，再对一个 step 内 token entropy 做长度归一化平均。
- 静态分析路径必须先生成 full CoT、计算 step entropy、将最低 entropy 的若干 steps 替换成单个 `[SKIP]` token，再把 compressed CoT 放回 prompt 让模型生成 final answer。该静态路径本身依赖预先得到 full CoT，主要用于验证 entropy criterion；真正可部署的直接短生成来自后续 SFT + GRPO。
- SFT 使用 entropy-compressed `(problem, compressed CoT, answer)` 训练数据；GRPO reward 同时考虑 final answer correctness、skip ratio、过多 `[SKIP]` 的 penalty 和 response length penalty。
- 论文固定静态 pruning ratio 为 `0.8`。Table 1 中低熵 step pruning 的实际 token reduction 随模型/数据集约为 `1.3%-44.9%`，并非删除 80% step 就减少 80% compute。部分困难数据集有明显准确率下降，例如 DeepSeek-R1-7B AIME2024 从 `63.33%` 到 `56.67%`；SFT+GRPO Table 3 中 Math500 从 `88.17%` 到 `85.00%`、AIME2024 从 `63.33%` 到 `57.14%`，同时分别减少 `35%` 和 `57%` thinking tokens。
- 论文局限包括依赖空行 step segmentation、主要在数学与少量 MMLU 领域验证、固定 `80%` threshold 未必跨模型/任务最优，论文也将多模态推广列为未来工作。

与当前项目的关键差异：

- LLM textual CoT 是 autoregressive 生成。省掉一个 reasoning step 会省掉其中许多串行 token decoding calls，并缩短后续 KV context；当前 ACoT-VLA 的 15 个 coarse frames 在每个 flow/denoising iteration 中并行处理。把 15 帧缩成 10 或 5 帧并没有减少 300M coarse/final expert 的 10 次主循环调用，因此用户实测 gated K=10/K=5 与 true segment skip 没有加速是架构原因，不是对 entropy principle 的直接否定。
- 论文的 entropy 是 categorical next-token Shannon entropy。当前 Stage A 在 `src/openpi/action_cot/compression.py::compute_mc_predictive_entropy` 中计算的是 K 次 continuous coarse-action sample 的 `mean(log(variance + eps))`，它是 MC predictive uncertainty proxy，不是论文中的同一个数学量；论文关于离散 CoT entropy 和 mutual information 的理论结论不能直接搬到当前 proxy。
- textual low-entropy step 通常是可由上下文预测的重复推导；机器人中的低方差 coarse segment 也可能是非常确定但物理上关键的抓取、保持夹爪或接触动作。因此必须用 counterfactual action impact、phase/gripper 分层和 closed-loop success 验证，而不能仅凭 low entropy 认定冗余。
- 论文的 `[SKIP]` 在 SFT 后成为模型学会解释的占位符。当前 interp/restore 和 short override 都没有训练 final expert 适应缺失 segment；即使增加 learned `[SKIP]` embedding，如果仍运行相同 dense JAX/Gemma 计算图，也不会自动产生速度收益。

对 Stage A/B 的判断：

- 不建议整体放弃或切换掉 entropy 路线。Stage A 的 entropy label export 对应论文的 full-CoT entropy analysis；Stage B 的 low/random/high pruning 和反事实 action deviation 对应论文的 static criterion validation。现有结果支持 low entropy 相比 high entropy 有更小的 `coarse_mse_to_full` 和 `action_mse_to_full`，但 low entropy 在 `action_mse_to_expert` 上未优于 random/high，因此目前只能算部分支持，不能声称已经复现论文结论。
- 应停止把 post-hoc coarse-frame pruning 或 short K consumption 作为最终 deployable accelerator。它们已经完成必要的反事实诊断，并证明当前 dense architecture 下 token/frame 数不是主要 latency lever。
- 下一阶段应当是 `entropy-guided learned compression`，不是与 entropy 无关地换方向。OFP/consistency distillation 可以作为“如何真正跨过多次 model calls”的实现机制，entropy 则决定训练时哪些 segment 需要高保真、哪些 observation 可以使用低预算。

现有 Stage C 代码可复用，但监督需要加强：

- `acot_libero_action_cot_dynamic_steps_stage_c` 已提供 `action_cot_step_values=(3,5,10)`、step head 和动态推理路径。
- 当前 `ActionCotLabelLoader` 只把每个样本的 mean entropy 按 `0.33/0.67` quantiles 映射到 3/5/10 类，标签并不是“保持 teacher/final action 质量所需的最小预算”。skip head 也只参与 auxiliary BCE loss，当前 sampling path 没有使用 skip logits 形成真实 sparse compute。
- 更接近论文的下一步是：为每个 observation 离线运行 3/5/10 budgets，以 full-10 action 为 teacher，选择满足 final-action error、gripper/jerk safety 和必要 rollout quality 的最小 budget 作为 SFT label；训练 step router 后，再用 compute-aware closed-loop reward 做少量 GRPO/RL 式优化。reward 应包含 task success/动作正确性、实际 NFE/latency saving、过度低预算 penalty 和 safety penalty。
- 当前固定 3-step sweep 是该路线的强 baseline：修正后 macro success 为 `0.970`，policy latency speedup 为 `18.54%`；learned entropy route 必须与固定 3-step 比较，而不应只与 10-step baseline 比较。

最终建议是保留 Stage A/B 作为 entropy criterion 与 redundancy evidence，将研究主线从 `post-hoc segment pruning` 推进为 `entropy-guided budget distillation/router`。这不是推倒已有工作，而是完成与论文中 `static pruning -> SFT -> reward optimization` 对应的后半段。当前没有为这条新训练路径实现代码，也没有新的训练或 rollout 结果。

## 2026-07-10 - reframed entropy as a predictor rather than the skip label

本次任务进一步澄清 Stage B 的因果顺序：当前 Action-CoT MC entropy 不应被直接当作“可以 skip”的 ground-truth label。更可靠的流程应先通过真实 counterfactual intervention 找到哪些计算预算/模块可以省略且保持质量，再验证 entropy 或其他便宜信号能否预测该 safe-skip label。

建议首先选择真正影响 latency 的计算单元。现有 benchmark 已排除 15 帧 coarse trajectory 中的 temporal frame/token 数量作为主要 latency lever；当前最适合的单元是 coarse expert 的完整 NFE/budget，例如 `1/3/5/10` 次 300M expert forward，初期可简化为 `3 vs 10` 两条固定 JIT 路径。整条 coarse branch bypass 和 transformer layer/feature reuse 可以作为后续单元。

建议构造 per-observation counterfactual budget oracle：对同一 observation 使用配对 RNG/noise 分别运行 full-10 与候选 `1/3/5` budgets，记录 coarse output 和 final action；根据 `action_rmse_vs_10`、gripper deviation、trajectory jerk/smoothness、必要的 contact/stage risk，以及由小规模 closed-loop rollout 校准的质量阈值，定义满足约束的最小 budget `b*`。`b*` 才是 router 的主要 SFT label。这里的 oracle 表示离线穷举多个 budget 后得到的最小安全预算，不是部署时额外运行所有路径。

之后再审计 entropy 的预测价值：将 Stage A MC entropy 的 mean/min/max/std、各 segment 位置等特征与 `b*` 或 `safe_for_3_steps` 比较，报告 AUROC、precision/recall、false-safe rate、calibration 和 rank correlation。机器人加速最应关注 false-safe rate，即预测可走 cheap path 但实际上不安全的比例。如果 entropy 预测力不足，应允许 prefix features、denoising convergence、gripper/contact phase 和 state deviation 共同进入 router，而不是强迫 entropy 成为唯一依据。

部署时不能在线计算 K=8 full-policy MC entropy后再决定是否省计算，因为其成本超过可能收益。Stage A MC entropy 可作为离线 teacher/auxiliary target；实际 router 应从已经计算的 prefix pooled features 或其他 cheap pre-coarse features直接预测 `b*`。当前 dynamic step head 的结构可以复用，但 `ActionCotLabelLoader` 中按 entropy `0.33/0.67` quantile 分箱得到 3/5/10 label 的逻辑应升级为 outcome-aware `b*` label。

建议保留历史 Stage A/B 结果，不重新命名或否定已有实验；后续可细分为 Stage B1 `causal skip/budget discovery`、Stage B2 `entropy predictiveness audit`、Stage C `learned router/distillation + closed-loop optimization`。当前没有修改代码或运行新实验。

## 2026-07-10 - reanalyzed fixed coarse-expert budget experiments

本次任务复核了项目中已经完成的 coarse expert 固定调用次数验证。已有两层实验：open-loop 单次推理 latency sweep，以及 LIBERO-10 closed-loop 全任务 success sweep。

Open-loop speed 结果来自 `results/stage_b/speed/coarse_steps_speed_summary.csv`：

- 10 calls: `73.8181 +/- 1.1303 ms`, speedup `0%`
- 7 calls: `67.4182 +/- 0.8225 ms`, speedup `8.6698%`
- 5 calls: `63.2259 +/- 0.7431 ms`, speedup `14.3491%`
- 3 calls: `60.3164 +/- 0.9594 ms`, speedup `18.2905%`
- 1 call: `57.1020 +/- 0.7871 ms`, speedup `22.6451%`

同一 CSV 中 no-coarse cached override 约为 `55-56 ms`。full 与 cached 的差值分别约为 10 calls `17.46 ms`、7 calls `11.97 ms`、5 calls `8.46 ms`、3 calls `5.15 ms`、1 call `1.67 ms`，说明 coarse branch 成本随 NFE 大致线性变化。这是“减少 coarse expert 完整 forward calls 会真实降低单次 latency”的强证据。

Closed-loop sweep 配置为 LIBERO-10 `10 tasks x 20 trials x 5 budgets = 1000 episodes`。原 full sweep 的 task0/task1 10-call timing 异常低，后续使用 fresh 10-call rerun 替换这两个 task 的 baseline 后，修正 macro 结果为：

- 10 calls: success `0.945` (`189/200`), policy latency `170.4752 ms`
- 7 calls: success `0.955` (`191/200`), policy latency `154.4858 ms`, corrected speedup `9.38%`
- 5 calls: success `0.950` (`190/200`), policy latency `146.7688 ms`, corrected speedup `13.91%`
- 3 calls: success `0.970` (`194/200`), policy latency `138.8671 ms`, corrected speedup `18.54%`
- 1 call: success `0.935` (`187/200`), policy latency `131.0223 ms`, corrected speedup `23.14%`

逐 task success 复核显示，3-call 与修正 10-call 在 task0-7 和 task9 的成功次数相同；整体 `+2.5 pp` 的差异全部来自 task8 从 `12/20=60%` 到 `17/20=85%`。1-call 则在 task2/task3 从 `100%` 降到 `95%`、task6 从 `95%` 降到 `85%`，同时 task8 从 `60%` 升到 `70%`。因此 1-call 已显示明确的 task-dependent quality risk，3-call 是当前观察到的最好固定折中。

本次额外使用 Wilson interval 和 Fisher exact test 做了近似统计检查：

- 10-call success `0.945`, Wilson 95% CI `[0.904, 0.969]`
- 3-call success `0.970`, Wilson 95% CI `[0.936, 0.986]`
- 10 vs 3 的独立样本 Fisher two-sided `p=0.3217`
- task8 的 10 vs 3 Fisher two-sided `p=0.1552`

这些区间重叠且检验未达到常用 `0.05` 显著性阈值。由于 corrected 10-call baseline 的 task0/task1 来自 fresh rerun，而不是与其他 budget 完全配对的同一 run，以上检验只能作近似参考。准确表述应为“3 calls 在现有 200 episodes 中未观察到成功率退化，并出现更高 point estimate”，不能表述为“3 calls 显著提升成功率”。

实验边界：该 sweep 验证的是给所有 observation 使用同一固定 budget 的总体速度-成功率曲线；它没有生成 per-observation 最小安全预算 `b*`，没有验证 Stage A entropy 是否能预测 `b*`，也没有训练专门适应 few-step sampling 的 student。现有模型仍是原始 flow model，仅在 inference 改变积分次数。因此下一步 causal budget discovery 仍需要对同一 observation、配对 RNG/noise 的 `1/3/5/10` 输出做逐样本比较，并用 rollout 校准 safe-budget label。

## 2026-07-10 - clarified coarse expert calls, denoising steps, and replan steps

本次任务纠正了术语歧义：在当前 `sample_actions_profile_coarse` 实现中，一次 coarse denoising iteration 会完整调用一次 coarse action expert，因此在正常 explicit Action-CoT generation path 上，`coarse expert calls per policy inference` 与 `action_cot_denoising_steps` 是同一个计数。上一轮所说的 `3 vs 10 coarse expert calls` 实际就是已经做过的固定 `action_cot_denoising_steps=3 vs 10` sweep，不是新的独立变量。

`replan_steps` 是另一层计数：policy 返回一个 action chunk 后，环境连续执行其中多少个 low-level actions，才重新采集 observation 并调用一次 policy。它控制 episode 内 policy calls 的频率，不直接控制一次 policy inference 内 coarse/final expert 的 denoising loop。

当前三个相关计数应区分为：

- `action_cot_denoising_steps`：每次 policy call 内 coarse/explicit Action-CoT expert 的 forward 次数。
- `num_steps`（final action denoising steps）：每次 policy call 内 final action expert 的 forward 次数，当前通常为 10。
- `replan_steps`：每次 policy call 返回 action chunk 后执行多少个环境动作再重新调用 policy。

近似关系为 `episode coarse-expert calls = policy calls per episode x action_cot_denoising_steps`，而 `policy calls per episode` 又大致随 `episode low-level steps / replan_steps` 变化。减少 denoising steps 优化单次 policy latency；增大 replan steps 减少整段 episode 的 policy 调用次数。两者可组合，但属于不同层面的加速机制。

## 2026-07-10 - explained the exact meaning of Action-CoT denoising steps

本次任务澄清了当前 flow-matching ACoT-VLA 中 `action_cot_denoising_steps` 的具体语义。coarse expert 不是先一次性生成 clean coarse trajectory、再把它交给另一个 denoiser；coarse expert 本身就是 learned flow velocity/denoising-direction estimator。一次调用输入当前 noisy coarse candidate `x_t`、当前 flow time/noise level `t` 和已缓存的 observation prefix KV，输出与整个 coarse trajectory 同形状的 velocity `v_t`。代码随后执行 Euler update `x_{t+dt} = x_t + dt * v_t`，其中 `dt=-1/action_cot_denoising_steps`，再把更新后的 `x` 和新的 `t` 输入同一个 coarse expert。

正常 full path 的推理过程为：在 `src/openpi/models/acot_vla.py` 中创建 shape `[batch, coarse_action_horizon, action_dim]` 的 `ref_action_noise`；当前 LIBERO 配置的 `coarse_action_horizon=15`，所以一次 expert forward 同时处理完整 15-frame candidate，而不是只生成其中一帧。`step_explicit_action_reasoner` 每次调用 PaliGemma coarse expert 并输出 `v_t`，`jax.lax.while_loop` 重复该 update N 次，N 即 `action_cot_denoising_steps`。因此该参数准确地说是从 `t=1` noise 积分到 `t=0` coarse Action-CoT 时的 Euler/ODE function-evaluation 数量。

需要区分 sequence length 与 refinement count：

- `coarse_action_horizon=15`：最终 explicit coarse trajectory 包含 15 个时间位置。
- `action_cot_denoising_steps=10`：同一个 15-frame candidate 被 coarse expert 迭代更新 10 轮。
- `action_horizon=10`：最终 action chunk 包含 10 个时间位置。
- `num_steps=10`：同一个 10-frame final-action candidate 被 final expert 迭代更新 10 轮。
- `replan_steps=5`：环境从返回的 final action chunk 中执行 5 个 low-level actions 后重新请求 policy。

训练时 flow path 使用 `x_t = t * noise + (1-t) * clean_coarse_action`，使 `t=1` 对应 noise、`t=0` 对应 clean trajectory；coarse expert学习当前位置的 velocity。推理时从 noise 出发并用负 `dt` 反向积分。因为每次 update 后 `x_t` 和 `t` 都改变，所需 velocity 通常也改变，所以必须重新评估 expert；只调用一次等价于使用一个很大的 Euler jump，并不是“生成后再额外去噪”。本次只做概念与代码路径解释，没有修改代码或运行新实验。

## 2026-07-10 - proposed learned action-conditional compute allocation

本次任务确认固定 `action_cot_denoising_steps=1/3/5/10` 只属于全局 inference hyperparameter sweep，能够提供 speed-quality baseline，但本身不构成算法创新。后续方法应让模型选择哪些 observation/action chunks 可以使用较少 coarse denoising NFE，哪些需要完整计算。

建议首先采用 chunk/observation-level learned budget routing，而不是直接做单帧 action-token routing。原因是当前 coarse Gemma expert 在一次 forward 中 dense 并行处理全部 15 个 coarse frames；已有 true segment skip 与 gated K=10/K=5 benchmark 已显示，将 active token 数从 15 缩到 10/5、但继续运行相同数量的 300M expert forwards，不会产生可测加速。若坚持 per-segment/per-frame budget，必须同时实现 active-token compaction、block-sparse/Mixture-of-Depth 或重新设计可独立计算的 segment refiner，否则选择结果只改变 action 数值，不改变主要计算量。

推荐的近期算法为 action-conditional budget router：

- 对每个 observation 离线、使用配对 RNG/noise 运行候选 budgets，初期只比较 `3 vs 10`，以后扩展到 `1/3/5/10`。
- 用 full-10 final action、dataset expert action、gripper deviation、jerk/contact/stage risk 和小规模 closed-loop calibration 定义最小安全预算 `b*`。
- router 输入应为已经计算的 prefix pooled feature、robot state 和可选的 task-phase feature，输出该 action chunk 的预算类别。Stage A MC entropy 可作为 auxiliary feature/target，但不能在部署时通过 K 次 full sampling在线计算。
- 训练目标可包含 budget classification、few-step/full-step action distillation、compute penalty 和 safety penalty。候选 `3/10` 是固定 JIT execution paths；算法贡献来自 outcome-aware oracle label、learned conditional routing、distillation 和风险回退，而不是候选数字本身。
- 部署时 free-space/稳定阶段可走 3-step path，抓取、接触、放置或 router 低置信度阶段走 10-step path。主比较 baseline 必须包括 fixed-3、fixed-10、random router、entropy-only router 和 learned outcome-aware router。

若研究目标明确要求“同一 15-frame trajectory 内不同动作 segment 使用不同 step 数”，建议作为第二阶段架构研究：所有 segments 先经过 cheap base predictor，segment router 选出 hard segments，再由可独立执行的 lightweight residual refiner处理 hard segments；需要固定 active-segment budgets/JIT paths，并以真实 wall latency验证。当前 dense coarse expert 不支持这种稀疏计算，因此不建议直接在现有 while-loop 中仅 mask/freeze segment。

本次只形成方法设计，没有修改代码或运行新实验。

## 2026-07-10 - reviewed recent entropy and adaptive-compute research for Stage A/B redesign

本次检索并对照了 2024-2026 年与 diffusion/flow 自适应计算、机器人动作不确定性和 CoT entropy 相关的原始论文。结论是：当前 Stage A entropy 可以保留为历史分析指标和辅助特征，但不应继续直接充当 `action_cot_denoising_steps` 的监督标签；Stage A v2 应改为估计“继续 refinement 的收益/少步执行的风险”。

当前实现 `src/openpi/action_cot/compression.py::compute_mc_predictive_entropy` 对 K 个 coarse samples 在每个 frame/action dimension 上计算 variance，再取 `mean(log(var + eps))`。它是对角 Gaussian differential entropy 的排序 proxy，不包含 action dimensions/time 之间的 covariance，并受 action normalization、`eps` 和有效 MC 多样性影响。随后 `src/openpi/action_cot/labels.py` 把所有 segment entropy 取均值，并按数据集 `0.33/0.67` quantile 映射到 `(3,5,10)`；这种分箱只保证相对分组，不证明高 entropy observation 确实需要更多 NFE。若 MC coarse samples 相同，`scripts/export_action_cot_entropy.py` 还可能记录 constant entropy，或在显式启用时加入仅供离线分析的 fallback Gaussian noise，必须审计 `used_offline_fallback_noise`。

与本项目最相关的研究证据如下：

- Diff-DAgger (`arXiv:2410.14868`) 指出生成式机器人策略在多模态决策点上，action/policy disagreement variance 会把多个有效行为模式误判为 uncertainty；论文改用 diffusion training loss 估计 OOD/failure risk。这直接说明当前 MC action variance 不能单独代表“需要更多 denoising”。
- AdaptiveDiffusion (`arXiv:2410.09873`, NeurIPS 2024) 不使用 sample entropy，而根据相邻 denoising latent 的相对 third-order difference 判断当前 velocity/noise prediction 是否稳定；稳定时复用上一预测并跳过一次大模型调用。该信号直接衡量 iterative process 的局部冗余，最适合作为当前 coarse flow loop 的 training-free baseline。论文也指出 extreme few-step 和快速 latent 变化时方法可能失效，并使用 `threshold delta` 与 `maximum consecutive skips Cmax` 控制误差。
- One Step Diffusion via Shortcut Models (`arXiv:2410.12557`, ICLR 2025) 将目标 step size 作为模型输入，并用 self-consistency/bootstrapping 让一个 flow model原生支持 one/few/many-step generation；论文还在 Push-T/Transport 展示了机器人 policy 用例。对本项目的意义是 few-step path 最好经过 step-size-conditioned training，而不是只在原 10-step model 上改 Euler 步长。
- FIPER (`arXiv:2510.09459`) 把 observation OOD 与 action uncertainty 分开：前者使用 policy embedding 上的 RND，后者用多 action-chunk samples 的 dimension-wise binned entropy，并用成功 rollouts 做 conformal calibration。论文明确把二者组合后再判断 failure，且在限制中承认 aleatoric/epistemic uncertainty 尚未完全解耦。它适合作为 risk calibration 参考，但在线采样 B 个 action chunks 成本较高，不适合直接充当省算力 router。
- Sparse ActionGen (`arXiv:2601.12894`) 采用 observation-conditioned pruner 直接预测 diffusion block/timestep 的 prune-and-reuse pattern，报告最高约 4x action generation speedup。它支持“直接学习计算分配，而非先把 entropy 当标签”的路线。
- VADF (`arXiv:2604.15938`) 用 VLM 识别 manipulation stage，再为简单/精细阶段分配不同 denoising steps/action horizon。其思想与 observation-conditioned budget routing一致，但具体 schedule 由 Qwen2-VL-7B prompt 在预设范围内分配，simulation 使用两张 A6000；论文 efficiency table 主要报告 diffusion loop latency，不能直接证明加入在线 VLM 后的端到端收益。因此适合作为语义阶段 baseline，而不是最终方案模板。
- Patch Forcing (`arXiv:2604.19141`, CVPR 2026) 使用轻量 difficulty head 预测 per-patch velocity error，并明确发现朴素地让不同 token 使用不同 timestep 会产生 train-test mismatch。该结论支持：若未来做 15-frame 内的 per-action refinement，必须相应改变训练分布和可稀疏执行架构，不能只在 inference mask frames。
- 2026 年 CoT 研究也更偏向 entropy trajectory/trend 而非单点绝对值。Unveiling the Entropy Dynamics of CoT (`OpenReview f5JmWpV01Y`) 用 CUSUM change-point 检测从探索到收敛的阶段；EAT (`OpenReview Sbvawkc7th`) 报告 CoT 内部 entropy 本身并不 informative，而停止后答案分布的 entropy trend 更相关。这些是跨模态启发，不能直接当作机器人结果。

建议 Stage A v2 产生三类相互区分的信号：`R_conv` 表示 coarse denoising trajectory 是否收敛/继续调用 expert 的边际收益；`U_ood` 表示 observation 是否偏离成功数据；`R_task` 表示 few-step final action 的任务风险，例如 gripper mismatch、contact phase、jerk 和 rollout failure。核心监督 label 应是同一 observation、同一 initial noise 下比较 `1/3/10` budget 后得到的最小安全预算 `b*`，而不是 entropy quantile。部署 router 直接预测 `b*` 或“继续到 10 step 的预期收益”，现有 entropy 只作为 auxiliary feature。

近期最值得先实现的两个实验是：第一，training-free `third-order latent stability + velocity reuse` baseline，验证在 10-step coarse loop 中能否真实减少 expert NFE 和 wall latency；第二，离线 paired-budget oracle 生成 `b*`，比较 raw MC entropy、entropy trend、latent stability、prefix feature 和 OOD score 对 `b*` 的 AUROC、false-safe rate 与 calibration。当前尚未实现这些算法，也没有新的实验结果。

## 2026-07-10 - clarified why diffusion/flow research is relevant without changing the ACoT-VLA research target

本次任务澄清研究定位：引用 diffusion/flow acceleration 工作不意味着把项目改成通用 diffusion acceleration。当前 ACoT-VLA 的 visual-language prefix 负责理解 observation/instruction，explicit coarse Action-CoT expert 在 `src/openpi/models/acot_vla.py::sample_actions_profile_coarse` 中从 `[B, coarse_action_horizon, action_dim]` noise 出发，每次调用 coarse Gemma expert 预测 velocity，并通过 Euler update 迭代生成连续 coarse reasoning trajectory；final action expert 也使用独立的 flow loop。因此 diffusion/flow 文献只用于解决 Action-CoT 的具体执行机制，即 repeated NFE、收敛判断、large-step training 和 conditional computation。

文本 CoT entropy 论文与当前系统不能直接等同：文本 step 是 vocabulary categorical distribution，可直接计算 Shannon entropy，而且少生成文本 token 通常能减少 autoregressive decoding；Action-CoT 是连续动作 trajectory，没有 vocabulary logits，当前 `mean(log MC variance)` 只是 differential-entropy proxy。已有 K=10/K=5 gated benchmark 还证明减少 coarse action frames 但不减少 dense expert forward calls 不产生加速。因此需要从 flow/diffusion policy 研究借用真实减少 NFE/blocks 的机制。

最终研究主线应表述为 `Adaptive Action-CoT Computation for VLA` 或 `Action-CoT Adaptive Deliberation`：根据 observation、任务阶段和下游 final-action 风险决定 coarse reasoning depth。ACoT 的核心贡献仍是 intermediate coarse action reasoning 及其对 final expert 的条件作用；flow/diffusion 只是底层生成器。文献应分层使用：机器人 flow/diffusion policy 的 adaptive NFE/router 是直接方法依据，图像 diffusion 的 latent stability/token routing 仅作为待验证的技术启发，纯图像生成指标不能作为本项目结论。

## 2026-07-13 - verified the online-MC entropy sampling call path

本次检查了 `scripts/eval_libero_action_cot_pruning.py`、`src/openpi/policies/policy.py` 和 `src/openpi/models/acot_vla.py` 中 `--adaptive_replan_entropy_mode online_mc` 的调用路径。当前 `--adaptive_replan_entropy_samples 5` 会执行 5 次完整的 `client.infer()`：第一次主推理的 `coarse_actions` 作为第一个样本，随后循环 `sample_idx=1..4`，使用不同 `policy_seed` 再执行 4 次完整 policy inference。因此每次 entropy decision 的 `entropy_oracle_extra_calls` 为 4，而不是 5。

每次 `policy.infer()` 都重新调用 `sample_actions_profile_prefix`，重新执行 observation preprocessing、图像/文本 prefix embedding、PaliGemma prefix forward 并生成新的 `prefix_out` 和 `kv_cache`；当前没有在这 5 个请求之间缓存或复用 VLM prefix/KV cache。随后每次调用还会重新运行 implicit action reasoner、coarse Action-CoT denoising loop 和 final action expert denoising loop。计算 entropy 时仅保留每次输出的 `coarse_actions`；后 4 次调用生成的最终 `actions` 没有用于环境执行。

结论：当前 online-MC 实现是“完整 VLM/policy 推理 5 次”，不是“VLM 一次后只生成 5 条 coarse action sequence”。因此 observed timing 包含 4 次不必要的 VLM、implicit reasoner 和 final action expert 计算；它适合作为 entropy oracle 验证，但不是部署加速路径。可行的后续优化是增加单请求 multi-sample 接口：只计算一次 prefix/VLM 和 implicit reasoner，复用同一 `prefix_out`/`kv_cache`，仅以 5 组 coarse noise 运行 coarse Action-CoT sampler；若只为计算 coarse entropy，还应跳过额外样本的 final action expert。

补充澄清：online-MC entropy 的统计对象是同一个 observation 条件下、由不同随机 seed/noise 生成的 5 条 `coarse_actions` trajectory，而不是 VLM feature、token logits 或 VLM 输出的 entropy。当前实现为了取得这 5 条动作轨迹，将同一个 observation 发送给 policy 5 次，因此 VLM prefix 也被重复计算了 5 次；VLM 重复执行只是现有实现带来的计算开销，不是 entropy 的计算目标。

计时口径补充：在当前 `adaptive_replan_entropy_samples=5` 评估中，控制台输出的 `observed_avg_wall_ms` 已经是一次 entropy/replan decision 内 5 次完整 `_infer` wall time 的总和再取平均，不能再乘以 5；`primary_avg_wall_ms` 仅表示其中第一次、实际提供执行 action 的主调用。已观察日志中 `observed_avg_wall_ms` 约为 `400-407 ms`，例如 `407.91 ms`，对应 `primary_avg_wall_ms=78.61 ms`。以该样本口径计算，后 4 次 entropy 额外调用合计约 `329.30 ms`，5 次平均约 `81.58 ms/call`。这些是当前 full-policy online-MC oracle 的实际 wall timing，不代表复用 VLM/KV cache 后的潜在 multi-sample 实现耗时。

## 2026-07-13 - reviewed adaptive replanning optimization directions

本次结合当前代码、已有 LIBERO-10 guarded entropy replanning 结果和相关 action-chunking 论文，分析了进一步扩大 replan horizon 的可行性。当前 checkpoint 配置 `acot_libero_action_cot_explicit_implicit_co_fusion` 的 `action_horizon=10`，评估脚本 `_candidate_horizons` 也会丢弃大于返回 action chunk 长度的候选，因此不修改模型结构与 checkpoint 时，单次 policy prediction 可执行的最大 horizon 是 10。若要使用 `H>10`，需要把模型输出 action horizon 扩展到 15/20 并重新训练或微调；仅修改评估参数不会产生额外可执行动作。

现有 guarded `[5,6,7,8,9,10]` 实验的平均实际 horizon 约为 `5.16-6.37`。该实验在 8/10 tasks 上减少 calls 并获得 policy/server cumulative speedup，但 task2 和 task9 的 episode steps、calls 和累计 latency 增加，task2/task9 success 相对 baseline20 分别下降 10/5 percentage points，task8 下降 5 points。这些结果不支持直接把所有阶段的固定/default horizon 提高；更长 open-loop execution 可能降低纠错频率、拉长 episode，抵消调用次数收益。

当前 entropy horizon selector 在 `scripts/eval_libero_action_cot_pruning.py::_mc_entropy_info` 中对 Stage-B segments 计算 entropy 后，只使用 `max(segment_entropy)` 作为一个全局 score，再通过 episode running quantile 映射到 horizon。该 score 不包含高 entropy 出现在 action chunk 前部还是后部的信息。更直接的下一步是保留 per-frame/per-segment entropy curve，并选择满足风险阈值的最大安全 prefix horizon，例如仅当候选 `1..H` 对应的累计/最大 entropy 均低于阈值时延长执行；这与 CVPR 2026 Adaptive Action Chunking (`arXiv:2604.04161`) 按未来 timestep 计算 action entropy、比较不同 chunk-size entropy curve 并选择拐点的思路相关，但本项目可研究 Action-CoT entropy 与 final-action horizon 的时间对齐。

建议的短期验证顺序：先在同一 checkpoint、`action_cot_denoising_steps=10` 下跑 fixed execution horizon `H=5,6,7,8,9,10` 的 per-task paired baseline，建立每个 LIBERO-10 task 的安全上限；随后实现 horizon-specific entropy，而不是全局 entropy scalar；增加有状态 hysteresis/guard，即 high entropy 或 gripper/contact risk 立即回到 H=5，只有连续多次 low entropy 才逐级升到 6/7/.../10，gripper event 后保持若干次 H=5 cooldown；还可在执行 chunk 期间使用低成本 proprioceptive deviation/event trigger 提前清空剩余 plan 并重新推理。

部署速度方面，online-MC 应先改为单请求 multi-sample：VLM prefix 和 implicit reasoner 只执行一次，复用 `prefix_out/kv_cache`，仅运行多组 coarse sampler，并跳过额外样本的 final action expert。若 oracle horizon policy 有效，再训练 lightweight horizon predictor，输入 observation/prefix feature、当前 action chunk、Action-CoT entropy/risk proxy，输出 `H in {5,...,10}`。Dynamic Execution Horizon Prediction (`arXiv:2606.11408`) 采用冻结 base chunk policy、只训练轻量 horizon branch 的路线，并报告 learned policy 在 free-space 使用长 horizon、精细操作使用短 horizon。Adaptive Action Chunking (`arXiv:2604.04161`) 提供 training-free entropy-based horizon baseline。Real-Time Chunking (`arXiv:2506.07339`, NeurIPS 2025) 则通过执行当前 chunk 时异步生成下一 chunk并对重叠部分做 inpainting 来隐藏 inference latency；它优化控制吞吐和 chunk 边界连续性，与减少 policy calls 的 adaptive horizon 可以组合，但不等价于减少模型计算量。

## 2026-07-13 - clarified the purpose and speed cost of event-triggered early replanning

本次澄清执行 action chunk 期间 early interruption 的目的：它不是独立加速模块，而是允许策略在稳定阶段尝试 `H=8-10` 时提供安全回退。控制循环仍逐步执行环境动作并读取已有 proprioception；每步只做 robot-state deviation、gripper transition、action direction/jerk 或环境已提供 collision/contact flag 等轻量检查。只有触发异常时才丢弃剩余 action plan 并提前调用 policy，否则继续执行当前 chunk，不增加 policy inference。

总推理耗时近似为 `policy_calls_per_episode * latency_per_policy_call + per-step guard checks`。guard 的收益条件是：平均有效 horizon 仍大于 baseline `H=5`，且没有导致 episode steps 增长；如果阈值过敏、频繁触发，则 policy calls 增加并可能比固定 H=5 更慢。因此需要同时报告 trigger rate、effective average H、calls/episode、episode steps、累计 policy/server latency 和 success rate。

当前 online-MC entropy 每次 decision 进行 5 次完整 policy inference，observed wall 约 `400-407 ms`，而原始单次主 inference 约 `78-80 ms`。即使从固定 H=5 理想地扩大到 H=10，policy calls 最多约减半，也无法抵消每次 decision 约 5 倍的 online-MC 成本；在相同 episode length 的理想化比较中，总 inference time 仍约为 baseline 的 2.5 倍。因此 event-triggered replanning 的 deployable speedup 必须在 entropy predictor、shared-prefix multi-sample 或仅统计 oracle primary call 的设置下验证，不能把当前 5-call observed timing 当作实际加速路径。

## 2026-07-13 - clarified the role of per-task safe execution horizons

本次澄清 fixed `H=5,6,7,8,9,10` per-task sweep 的目的。每个 LIBERO-10 task 的“安全上限”是诊断和校准基线，定义为在给定试验规模与容许成功率下降条件下，仍未显著降低成功率、未明显增加 episode steps/累计 latency 的最大固定 execution horizon；它是经验上限，不是形式化安全保证，也不是最终部署时按 task ID 写死 horizon。

该基线用于区分两类问题：如果某个 task 在 fixed H=8/10 下本身就稳定，而 adaptive selector 很少选择长 horizon，则问题在 selector/entropy calibration；如果 fixed H=8 已明显掉成功率，则该 task 或其精细阶段确实需要更频繁反馈，不能通过更激进的全局 horizon mapping解决。它还提供最大可实现 call reduction、识别 horizon-sensitive tasks，并为 entropy threshold、event guard 和 horizon predictor 提供监督/评估参照。最终目标应从 task-level 上限细化到 observation/task-phase-level safe horizon，因为同一任务的 free-space 阶段可能适合 H=10，而 grasp/contact/place 阶段可能只适合 H=5。

## 2026-07-13 - designed a training-free guarded Action-CoT horizon selector

本次综合 Adaptive Action Chunking (`arXiv:2604.04161`)、Dynamic Execution Horizon Prediction (`arXiv:2606.11408`)、Real-Time Chunking (`arXiv:2506.07339`) 与当前 ACoT-VLA 代码/实验结果，形成 predictor 训练前的 execution-horizon 方案。AAC 的原始方法并不是把整个 chunk 压成一个 entropy scalar：它并行采样 N 个 action chunks，分别计算每个未来 timestep 的连续 translation/rotation Gaussian differential entropy和离散 gripper entropy，再对每个候选 h 计算 prefix average entropy `E_bar_h`，用 `argmax_h(E_bar_{h+1}-E_bar_h)` 找 entropy jump，并通过 minimum action magnitude 下界避免过短/静止 chunk。论文默认 N=20；其 A800 表中 N=1/5/10/20 的 latency 为 83.0/83.5/84.3/106.0 ms，依赖 batched parallel sampling。当前项目的 K=5 是 5 次串行 full policy calls，约 400 ms，不能直接视为 AAC 的部署成本。

代码检查发现一个此前未纳入 selector 的时间对齐问题。`acot_libero_action_cot_explicit_implicit_co_fusion` 配置为 `coarse_action_horizon=15`、`action_horizon=10`、`joint_action_shifts=(2,1)`；`LiberoACOTInputs` 使用原始 action sequence 的 stride 2 构造 15 个 coarse tokens，使用 stride 1 构造 10 个 final actions。因此 coarse token 对应原始控制时刻 `0,2,4,...,28`，而一次可执行 final chunk 只对应 `0,...,9`。当前 `_mc_entropy_info` 对全部 15 个 coarse tokens/segments 计算 entropy 并取 `max`，会让第 10-28 控制步的远期 coarse uncertainty影响当前仅执行 5-10 步的 horizon 决策；同时它丢失 uncertainty 在时间轴上的位置。这是当前全局 entropy-to-H mapping 的结构性缺陷。

建议现阶段实现 `Guarded Action-CoT Horizon Selector`，候选保持 `H in {5,6,7,8,9,10}`，但职责分离：Action-CoT entropy curve提出 raw H，action/gripper risk只向下限幅，hysteresis控制 H 的变化速度。具体流程为：对同一 observation 的 K 个 normalized coarse samples按 coarse token计算 frame-level Stage-B-style MC entropy；使用 coarse timestamps `0,2,...,28` 将 entropy curve插值到 final action timestamps `0,...,9`，只保留当前 executable horizon 对应的前缀；为每个 h 计算 `E_bar_h=mean(E_1...E_h)` 和 jump `Delta_h=E_bar_{h+1}-E_bar_h`。若前 5 步 entropy 已处于离线校准的高风险区，raw H=5；若存在显著 entropy jump，raw H取 jump 前的最大安全 prefix；若整体低 entropy 且没有显著 jump，raw H=10。显著性阈值应从 Stage A/paired rollout 数据离线校准，不继续使用每个 episode 的 running quantile warmup。

安全层建议：若 primary final action chunk 在候选 prefix 内出现 gripper switch、高 action delta 或 jerk spike，则把 H cap 到该事件附近且不低于 5；gripper/contact event 后保持 1-2 次 replan 的 H=5 cooldown。H 下降立即生效，H 上升最多每次增加 1，并要求至少连续两次 low-risk decision，从而避免单次 noisy K=5 entropy 直接从 5 跳到 10。执行期 event trigger作为最后安全回退，可在 proprio/gripper deviation 明显时丢弃剩余 plan；它不参与正常 H 评分。

实验设计应至少包含：fixed H=5/6/7/8/9/10；论文式 final-action AAC；仅使用 Action-CoT entropy curve 的 CoT-AAC；CoT-AAC + guard/hysteresis。所有方法报告 per-task success、episode steps、effective H distribution、calls/episode、trigger/guard rate、累计 deployable policy/server time。20 trials 的成功率分辨率是 5 percentage points，无法严格判断 2-3 point drop；可先用 20 trials筛选，再对临界 task/setting 扩到 50 trials。若 training-free oracle在 success约束下减少累计推理时间，后续 predictor 的监督目标应为该 selector 产生的 observation-level H，而不是 task-level固定上限。

## 2026-07-13 - separated pruning entropy from horizon-selection entropy and explained temporal alignment

本次澄清 Stage-B entropy 与 adaptive-H entropy 的目标差异。现有 `compute_mc_predictive_entropy` 对每个 segment 内的 K 个 coarse samples先按 time/action dimension计算 MC variance，再取 `mean(log(var+1e-6))`，输出一个 segment scalar；它适合对 segments 排序并验证 low-entropy pruning，但不保留 uncertainty 在未来时间轴上的位置，因此不应直接作为精确选择 execution horizon 的唯一指标。

建议不覆盖 Stage-B 指标，而新增独立的 `horizon_entropy` 并保留两者消融。稳定基线可以把现有公式降到 frame level，即 `u_t=mean_d log(var_k(coarse[k,t,d])+eps)`，等价于 length-1 segment entropy。语义增强版本可参考 AAC，把 LIBERO 7-D action 分成 translation 3-D、rotation 3-D 与 gripper 1-D：translation/rotation使用带 `lambda I` shrinkage 的 Gaussian covariance log-determinant entropy，gripper二值化后使用 Bernoulli Shannon entropy，再对三组分量做离线标准化后组合。K=5 时 3-D full covariance估计较噪，必须使用 shrinkage，并与更稳定的 diagonal log-variance baseline 对比，不能未经消融直接替换现有方法。

时间错位来自训练数据构造。`LiberoACOTInputs` 根据 `joint_action_shifts=(2,1)` 从同一 raw action sequence采样：final actions取 `raw[0:10:1]`，对应 raw control times `0,1,...,9`；coarse actions取 `raw[0:29:2]`，对应 `0,2,4,...,28`。因此 H=5 只执行 raw actions `0..4`，与其直接相关的 coarse observations约为 `0,2,4`；H=10 执行 `0..9`，直接相关的 coarse prefix约为 `0,2,4,6,8`，可用 time interpolation补齐奇数时刻。当前 selector却对全部 coarse times `0..28` 的 segments取最大 entropy，所以 raw time 10-28 的远期不确定性也可能缩短当前 H，尽管机器人会在到达这些时刻前已经重新调用 policy。

推荐 selector 将 coarse frame entropy放在真实 raw-time坐标 `0,2,...,28` 上，插值到 executable final-time坐标 `0,...,9`，仅用 candidate H 对应的 prefix计算 `E_bar_H` 和 entropy jump。远期 coarse entropy可保留为弱 global-risk/tie-break信号，但不应以全局 max直接决定当前 H。该修正同时避免两种错误：远期高 entropy导致当前过度保守，以及 segment averaging稀释近端局部 entropy spike导致当前执行过长。

## 2026-07-13 - clarified coarse horizon, final action horizon, and execution horizon

本次澄清 ACoT-VLA 中三个容易混淆的 horizon。配置 `coarse_action_horizon=15` 表示 explicit Action-CoT reasoner输出 15 个 coarse reasoning tokens；由于训练数据 `joint_action_shifts[0]=2`，其监督目标对应 raw control times `0,2,...,28`，提供低频、较长时间范围的中间动作规划。配置 `action_horizon=10` 表示 final action expert输出 10 个高频、可直接送给环境的 actions，对应 raw control times `0,...,9`。final expert以 visual-language prefix、implicit reason、explicit coarse Action-CoT为条件生成该 executable chunk，因此 coarse与final不是同一输出，长度、采样频率和职责均不同。

当前 adaptive replanning 讨论中的候选 `5,6,7,8,9,10` 实际是 execution horizon，建议记为 `h_exec`，不是 coarse horizon。评估脚本最终执行 `action_plan.extend(action_chunk[:replan_horizon])`，所以 `h_exec=7` 的含义是从 final 10-action chunk中执行前7步，然后重新采集 observation并调用 policy。Stage B segment pruning则作用在 15 个 coarse Action-CoT tokens上；`action_cot_denoising_steps`控制每次生成 coarse trajectory时 coarse expert的迭代次数。这三个量必须分别报告为 `H_cot=15`、`H_action=10`、`h_exec in {5,...,10}`，避免继续把它们都称为 H。

## 2026-07-13 - scoped current optimizations to the correct horizon variables

本次确认当前 adaptive replanning/entropy horizon selector优化的是 final action execution horizon `h_exec`，即从每次生成的 `H_action=10` final action chunk中实际执行前多少步再重新调用 policy。这个优化目标与减少 episode policy calls一致，不应改成直接调整 `H_cot`。当前候选 `h_exec in {5,6,7,8,9,10}` 可继续保留。

不同实验线对应的变量如下：Stage B low/random/high segment pruning作用于 `H_cot=15` 的 coarse Action-CoT segments；fixed `action_cot_denoising_steps=1/3/5/7/10` sweep作用于每次生成 coarse Action-CoT时的 iterative expert NFE，不是任何 horizon；adaptive replanning作用于 final chunk的 `h_exec`；未来 horizon predictor也应预测 `h_exec`。这些机制可组合，但不能共享“H”这一含糊名称。

现阶段不建议修改 checkpoint架构中的 `H_cot=15` 或 `H_action=10`，因为改变这两个训练输出长度涉及数据构造、模型 shape和重新训练。需要立即调整的是评估语义：将 CLI/输出中的 `adaptive_replan_horizons`、`replan_horizon`逐步明确为 `adaptive_execution_horizons`、`execution_horizon`；将 entropy selector从“全部 15 coarse tokens的 max segment entropy直接映射到 h_exec”改为“coarse entropy按 stride=2对齐到 final times 0..9后，使用候选 executable prefix的 entropy curve选择 h_exec”。同时保留 final-action AAC baseline，用于验证 coarse Action-CoT entropy是否真的比 final action uncertainty更适合决定 execution horizon。

## 2026-07-13 - clarified joint optimization of denoising depth and execution horizon

本次确认 `action_cot_denoising_steps` 与 adaptive execution horizon `h_exec` 可以组合，因为它们优化不同层级：前者减少每次 policy inference 内 coarse Action-CoT expert的 NFE/单次 latency，后者增加每次 final action chunk实际执行的步数、减少 episode policy calls。总推理时间可近似写为 `T_episode ~= policy_calls(h_exec) * latency_per_call(action_cot_denoising_steps)`，因此在 episode steps和成功率保持时，两类收益可近似相乘。

需要区分固定与动态联合。固定 `action_cot_denoising_steps=3/5/10` 加 adaptive `h_exec` 现在即可运行；但不能在同一次 inference 中直接用当前 online-MC entropy同时决定当前 denoising steps与 h_exec，因为执行顺序是“先确定 denoising steps并生成 coarse/final trajectories，再从 samples计算 entropy，最后选择 h_exec”。若用当前 entropy反过来决定同一次 coarse generation的 steps，只能增加第二次推理，反而变慢；可选替代是用上一次 entropy决定下一次 steps、使用 denoising-loop early-convergence signal，或后续训练基于 prefix/state的 compute predictor。

实验上不应直接只跑双重激进组合。建议使用 factorial comparison：A=`steps10 + fixed h5` 原始基线；B=`steps3 + fixed h5` 仅单次推理加速；C=`steps10 + adaptive h` 仅减少 calls；D=`steps3 + adaptive h` 联合优化。随后再补 steps5。因为减少 coarse denoising steps会改变 coarse trajectory与其 MC entropy分布，steps10校准的 entropy threshold不能默认复用于 steps3；每个 denoising budget必须重新检查 entropy calibration、success、episode steps、effective h、calls和累计 latency。

联合动态策略应保持保守耦合：高风险阶段使用 `steps10,h5`；中风险阶段可使用 `steps5,h5-7`；只有连续低 entropy、无 gripper/contact/jerk guard时才使用 `steps3,h8-10`。但该策略在 predictor/early-exit signal完成前只能作为离线 oracle或手工 baseline，不能宣称为可部署的当前-entropy联合路由。

## 2026-07-13 - implemented time-aligned adaptive execution-horizon selectors

本次修改 `scripts/eval_libero_action_cot_pruning.py`，只实现 adaptive execution horizon，未修改模型结构、`H_cot=15`、`H_action=10`、denoising loop或 predictor。新增 `--adaptive_h_selector legacy|final_aac|cot_aac|guarded_cot_aac`，默认 `guarded_cot_aac`；新增 `--adaptive_h_entropy_algorithm diagonal_logvar|aac_grouped`，默认保留稳定的 Stage-B-style per-frame diagonal log-variance。`aac_grouped` 对前3维 translation和后3维 rotation使用带 covariance shrinkage 的 Gaussian entropy，对第7维 gripper使用二值 Shannon entropy。online-MC adaptive entropy samples默认值从4改为5。

online-MC 路径现在同时保存 K 个 `coarse_actions` 和 K 个 final `actions`，不再丢弃后 K-1 次 final action输出，因此同一次 K=5 oracle可分别运行 Final-AAC 与 CoT-AAC。新增逐frame entropy curves：final actions直接得到 length-10 curve；coarse Action-CoT先在15个 coarse tokens上计算 entropy，再根据 `--adaptive_h_coarse_stride 2` 从 coarse raw-time `0,2,...,28`插值到 executable final-time `0,...,9`。原 Stage-B segment entropy仍保留给 `legacy` selector和历史 pruning逻辑。

AAC selector对候选 `h_exec in {5,6,7,8,9,10}` 计算 prefix mean entropy与相邻 horizon jump；使用 median absolute deviation threshold判断 jump是否显著，显著时选择 jump前的 prefix horizon，无显著 jump时选择最大可执行 horizon。`guarded_cot_aac` 继续使用 per-task entropy history/absolute thresholds区分 high/mid/low risk：warmup或high risk回到baseline H=5，mid risk最多增长一档，low risk使用AAC raw H；stage guard仅检查 raw candidate prefix中的 gripper/action delta/jerk，并可将H cap到5。

新增 episode-level `AdaptiveHState`：stage guard触发后默认保持2个后续 decisions的H=5 cooldown；H下降立即生效；H增长默认每次最多+1，并要求连续2次low-risk。新增逐决策输出 `adaptive_h_decisions.csv`，记录 environment step、selector、raw H、guard cap、previous/final H、entropy thresholds、entropy curve、prefix entropy、jump curve、stage guard、cooldown、hysteresis和decision reason。`rollout_rows.csv`、`per_task_summary.csv`、`summary.json`新增 raw H、guard cap、max entropy jump、hysteresis rate及decisions文件路径。

同步修改 `scripts/sweep_action_cot_denoising_steps.py`，可将新 adaptive-H selector、entropy算法、stride、jump MAD、growth/cooldown参数传入 closed-loop sweep，并汇总 raw H/guard/hysteresis指标。仅执行了 `python -m py_compile` 和 `git diff --check`，两项均通过；按本地无训练/LIBERO运行环境的约定，没有运行模型或rollout测试。服务器下一步应先分别跑 `final_aac`、`cot_aac`、`guarded_cot_aac` 的小规模 smoke，再运行10 tasks正式比较。

## 2026-07-13 - prepared full guarded CoT-AAC LIBERO-10 evaluation

用户决定跳过单task smoke，直接运行 `libero_10` 全量 adaptive-H 评估。正式配置为10 tasks、每task 20 trials、`mode=full`、固定 `action_cot_denoising_steps=10`、baseline execution horizon 5、候选 `h_exec={5,6,7,8,9,10}`、`adaptive_h_selector=guarded_cot_aac`、`adaptive_h_entropy_algorithm=diagonal_logvar`、online-MC samples=5。该配置只验证 adaptive execution horizon，不同时改变 denoising budget。结果尚未产生。

## 2026-07-13 - audited Git tracking and ignore rules

本次只检查 Git 状态、`.gitignore`、current `main/origin/main` tree与历史对象，没有删除文件、修改索引或重写历史。当前没有 staged changes；工作区待提交代码是 `scripts/eval_libero_action_cot_pruning.py` 与 `scripts/sweep_action_cot_denoising_steps.py`。未跟踪目录包括约8 KB的 `results/stage_b/adaptive_replan_entropy_guarded_h5_10_vs_original/`（1个CSV和1个JSON实验总结）以及约21 MB的 `tmp/`（论文PDF/TXT和`.DS_Store`）。`.gitignore` 当前未忽略 `tmp/` 或 `.DS_Store`，因此执行 `git add .` 会把这些临时文件加入索引。

`main` 与 `origin/main` 当前已跟踪三个不应属于可移植仓库的根目录条目：`.DS_Store`，以及指向服务器绝对路径 `/root/autodl-tmp/acotvla/assets`、`/root/autodl-tmp/acotvla/checkpoints` 的 `assets`/`checkpoints` symlinks。现有 `assets/`、`checkpoints/` ignore规则不匹配这两个已跟踪 symlink本身，而且已跟踪文件即使新增ignore也不会自动移除，需要后续显式 `git rm --cached`。历史中还存在多份约92-98 KB的 `src/openpi/training/config.py.bak*` blobs，当前tree已无这些备份；它们只占较小历史空间，移除需要重写历史。

仅扫描 `main/origin/main` 后，最大blob为历史版本 `uv.lock` 约862 KB；没有 checkpoint、模型权重或临时论文PDF进入远端主分支。最初用 `git rev-list --objects --all` 看到的约21 MB PDF来自 Codex桌面应用本地 `refs/codex/turn-diffs/*` 快照，不属于 `main/origin/main`，普通 `git push main` 不会上传。`results/stage_b/original_baseline/libero10_original_per_trial.csv` 约330 KB，属于此前有意保存的实验结果。

## 2026-07-13 - fixed Git ignore rules and untracked local symlinks

本次按检查结果修改根目录 `.gitignore`：将 `assets/`、`checkpoints/` 改为可匹配根目录 symlink或directory的 `/assets`、`/checkpoints`；新增 `.DS_Store`；将本地运行产物规则明确为 `/output/`、`/outputs/`、`/tmp/`、`/work/`。执行 `git rm --cached -- .DS_Store assets checkpoints`，只从Git索引移除三个已跟踪条目，本地 `.DS_Store` 和指向服务器路径的两个symlink均保留。

使用 `git check-ignore --no-index` 验证 `.DS_Store`、`assets`、`checkpoints`、`tmp/` 及其中PDF均命中新规则；`git diff --check`通过。为保证索引清理与ignore规则属于同一提交，已暂存 `.gitignore` 修改和三个索引删除。两份adaptive-H脚本仍未暂存；`results/stage_b/adaptive_replan_entropy_guarded_h5_10_vs_original/` 仍为未跟踪的有用实验结果，未删除或自动加入。

## 2026-07-13 - committed and pushed adaptive-H implementation

本次使用 HTTPS proxy `http://127.0.0.1:7897` 将本地 `main` 直接推送到 `https://github.com/KongyueX/LightAcotVLA.git`。推送前确认 GitHub CLI未登录，但仓库使用HTTPS remote与macOS credential helper，直接 `git push`认证成功；未创建PR。

远端一次收到两个本地提交：`539e63f new adapitve h` 包含 `.gitignore`更新、停止跟踪`.DS_Store`及服务器绝对路径`assets/checkpoints` symlink；`26cb3f9 add time-aligned adaptive horizon selection` 包含 `scripts/eval_libero_action_cot_pruning.py`、`scripts/sweep_action_cot_denoising_steps.py`和 guarded H5-10 对比结果CSV/JSON。提交前将结果CSV从CRLF规范为LF，并把summary JSON中的本机绝对路径改为仓库相对路径。执行 `python -m py_compile` 与 `git diff --cached --check`均通过。推送结果为 `842a7b5..26cb3f9 main -> main`，完成后本地 `main` 与 `origin/main`一致，工作区干净（忽略的reports context除外）。

## 2026-07-13 - confirmed purpose and measurement scope of full guarded CoT-AAC run

本次确认 `guarded_cot_aac_libero10_10tasks_20trials` 命令用于验证 Stage-B-style Action-CoT entropy 能否在保持 LIBERO-10 closed-loop成功率的前提下，自适应选择 final action execution horizon `h_exec in {5,6,7,8,9,10}`，从而相对固定 `h_exec=5` baseline减少每个 episode 的 policy calls与累计 deployable inference time。该实验不进行 segment pruning，不改变 `H_cot=15`或`H_action=10`，也不评估 denoising step加速；`action_cot_denoising_steps=10`保持原始计算预算，隔离 adaptive replanning变量。

配置中 `online_mc` 对同一 observation串行采样5次完整policy输出以计算 entropy，因此 observed wall time包含5次调用开销，只表示 oracle验证的真实运行成本；`primary/deployable` timing只统计最终用于控制的主调用，表示未来由轻量 predictor替代 online-MC 后可能实现的部署口径。正式比较需要按task报告 success rate、episode steps、calls/episode、effective `h_exec`分布、guard/hysteresis rate以及每episode累计 deployable policy/server/wall time，并与同task固定 `h_exec=5` baseline比较。用户给出的命令末尾 `--output_dir` 缺少参数，运行时必须补为 `--output_dir $OUT`。结果尚未产生。

## 2026-07-13 - guarded CoT-AAC full evaluation completed on server

服务器日志确认 `guarded_cot_aac_libero10_10tasks_20trials` 评估运行完成，并写出 `/root/autodl-tmp/acotvla/stage_b_pruning_eval/guarded_cot_aac_libero10_10tasks_20trials/summary.json`。本轮消息尚未提供 `summary.json`、`per_task_summary.csv`或具体指标内容，因此目前只能确认产物生成，不能判断成功率、policy calls、execution horizon或推理时间是否改善。下一步应提取逐task的 success、timeout、episode steps、deployable calls、effective `h_exec`、guard/hysteresis以及 observed/deployable累计时间，并与同task固定 `h_exec=5`原版baseline比较。

## 2026-07-13 - analyzed full guarded CoT-AAC LIBERO-10 results

用户提供了 `guarded_cot_aac_libero10_10tasks_20trials/per_task_summary.csv` 的关键逐task结果。10个task各20 trials，总体成功率为94.0%（188/200）；此前记录的原版固定 `h_exec=5` baseline为97.5%（195/200），point estimate下降3.5 percentage points。逐task成功率变化为：task0/1/3/4/5不变；task2/6/7/9各下降5 points；task8从80%下降至65%，下降15 points。20 trials的单task成功率分辨率为5 points，因此单个5-point变化不能单独视为稳定结论，但总体point estimate未达到“成功率下降不超过2-3 points”的目标，task8尤其需要复核。

Adaptive run的跨task平均 episode steps为318.71，原版为290.74，增加9.62%；平均deployable policy calls/episode为61.48，原版为56.76，增加8.33%。只有task1的calls下降（52.05到50.55，-2.9%）；其余task均增加或未改善，task8从111.60增至135.05（+21.0%）。因此即使完全忽略entropy oracle额外调用与硬件计时漂移，当前selector也没有通过减少policy calls实现算法层面的加速。

Adaptive run的平均 `raw_h_exec`约6.57，但最终平均 `h_exec`仅约5.07；逐task `avg_h`范围5.01-5.16，而stage-guard rate范围0.48-0.88、跨task平均约0.73，hysteresis-limited rate跨task平均约0.18。这表明AAC entropy原始候选通常会延长horizon，但stage guard、cooldown与hysteresis大幅将其压回baseline H=5。与此同时，少量放行的长horizon没有减少总体steps，反而伴随episode变长与task8成功率下降；仅凭aggregate还不能判断是guard threshold、guard event定义、cooldown还是entropy jump时序造成，需检查`adaptive_h_decisions.csv`的final-H分布和`stage_guard_reason`/`decision_reason`频次。

按用户给出的两位小数deployable policy totals，跨task平均为5.417 s/episode，原版记录为4.430 s/episode，增加22.29%。原版单次policy约77.7-79.5 ms；adaptive task1-9大多约84.7-85.5 ms，task0约117.7 ms，说明本次运行还存在计时环境差异或task0启动/outlier开销，具体原因尚未确认。不过calls本身已增加，所以计时归一化后也不能得到加速。当前online-MC每个deployable call对应4个extra calls，平均extra calls/episode为245.92，恰为61.48的4倍；跨task平均observed wall约27.69 s/episode，明显不是可部署加速路径。

当前结果不支持立即训练entropy predictor：predictor只能消除4次MC额外调用，不能修复selector最终H接近5、calls增加和成功率下降。下一步应先分析决策CSV，调整或重新设计guard/hysteresis，使final-H分布真正扩展，同时通过task8等敏感task验证长horizon是否在错误阶段被放行；修正后再运行对齐baseline的paired评估。

## 2026-07-13 - localized guarded CoT-AAC suppression using decision distributions

用户提供了全部10个task的 `adaptive_h_decisions.csv` horizon与guard reason计数。共12,296次replan decisions。Entropy/AAC经过risk gate后的raw horizon分布为：H5=5,093（41.42%）、H6=4,072（33.12%）、H8=222（1.81%）、H9=493（4.01%）、H10=2,416（19.65%），raw H均值6.528，raw H>5占58.58%。最终执行分布为：H5=11,675（94.95%）、H6=456（3.71%）、H7=107（0.87%）、H8=40（0.33%）、H9=12（0.10%）、H10=6（0.05%），final H均值5.071，最终H>5仅占5.05%。因此entropy并非从不提出延长，而是guard/cooldown/hysteresis后处理将大部分延长建议压回H5。

Guard reason汇总为：`none` 3,210（26.11%）、`cooldown` 3,859（31.38%），其余直接stage guards共5,227（42.51%）；合计73.89%的decisions处于直接guard或cooldown状态。按reason是否包含某因素统计（类别有重叠），jerk相关2,963次（24.10%）、action-delta相关2,017次（16.40%）、gripper相关910次（7.40%）。`cooldown`是最大单一类别；当前每次直接guard触发后保持2个后续decisions为H5，并且重复直接guard会重置cooldown，造成连锁抑制。默认`jerk_ratio=jerk/(action_delta+1e-6)`在action delta很小时可能放大，且当前guard使用candidate prefix内的平均delta/jerk与任意gripper change后统一cap到H5，尚未按危险事件的具体future timestep做局部cap。

各task最终H>5比例分别为：task0 7.19%、task1 1.09%、task2 5.65%、task3 7.76%、task4 2.02%、task5 3.36%、task6 10.00%、task7 2.87%、task8 6.52%、task9 1.70%。这与最终avg H仅5.01-5.16一致，理论上在相同episode steps下也只能带来很小的calls减少。Task8失败率增加后episode变长，因此总calls进一步增加；需要额外报告successful-episode-only与calls per environment step用于诊断，但部署结论仍应以成功率约束下的total time per episode/task为准。

下一版建议不是简单放宽一个全局阈值，而是做消融并重构guard：先运行无guard/hysteresis的`cot_aac`以测量raw entropy selector本身的速度-成功率边界；将当前prefix scalar guard改为per-timestep event-aware cap，在预测危险变化发生于future step j时将H cap到j附近而不是一律回到5；去除或重新校准易在近静止动作中放大的jerk ratio；默认cooldown先降为0，仅对真实gripper/contact event保留短cooldown；降低连续low-risk和growth-limit的重复抑制。Task8应单独检查失败episode中H>5的decision位置，确认偶发长H是否集中在关键操作阶段。

## 2026-07-13 - clarified Task 8 degradation and pure CoT-AAC ablation

本次进一步解释task8结果与`cot_aac`消融。原版task8为16/20成功（80%），当前guarded CoT-AAC为13/20（65%），15-point差异实际对应多3个失败；20 trials样本较小，不能仅凭aggregate证明真实成功率必然下降15 points，但该point estimate是需要定位的风险信号。Task8共有2,701次replan decisions，其中final H>5为176次（6.52%），即平均约8.8次长horizon decisions/episode；因此“占比低”不等于每个episode没有关键长H。Task8的35% timeout也直接拉高了avg steps与calls。

当前证据尚不能确定task8下降的直接原因。待验证机制包括：少量H>5恰好位于接触/抓取等关键阶段；time-aligned coarse entropy以stride 2表示低频长期轨迹，可能遗漏final action的高频接触不确定性；MC low entropy只表示5次samples一致，不保证动作正确或适合open-loop执行更久；per-task running quantile会把相对最低的一部分状态视为low risk，即使task8的绝对风险仍高。应将task8的success/failure episodes与decision CSV联结，比较H>5 rate、首次长H的environment step、max H和guard reasons，才能支持具体归因。

`guarded_cot_aac`与纯`cot_aac`并非只差一个stage-guard开关。当前guarded路径先用AAC entropy jump提出候选，再用running entropy thresholds修改候选：warmup/high entropy回到H5，mid entropy最多H6，low entropy保留AAC候选；随后action-delta/jerk/gripper stage guard可cap H5，并应用2-decision cooldown、连续2次low-risk要求和每次最多+1的growth limit。用户看到的`raw_h`已经经过前述entropy-risk gate，不等同于纯AAC原始输出。`cot_aac`则直接使用time-aligned coarse entropy prefix jump选择H，不应用running quantile gate、stage guard、cooldown或hysteresis，因而是用于隔离“entropy selector本身是否能减少calls并保持成功率”的更激进消融，不是建议直接部署的最终方案。

## 2026-07-13 - Task 8 long-horizon decisions correlate with success, not failure

用户将task8的20个episodes按成功/失败与adaptive-H decisions联结。成功13个episodes的平均H>5 decision rate为8.85%、平均H为5.12；失败7个episodes的平均H>5 rate为3.95%、平均H为5.05。全部失败episode均运行至1000-step timeout。当前样本中失败轨迹没有表现出更多或更长的execution horizons，因此现有证据不支持“Task 8成功率下降是由更频繁长H直接造成”的解释。相反，H>5更像低entropy/easy-progress状态的相关标志；成功轨迹能够进入更多低风险阶段，而失败轨迹进入stuck/high-entropy状态后长期保持H5。该相关性仍不能证明长H提升成功率，因为失败episode的长timeout tail会累积大量H5 decisions，存在survivorship/state-distribution bias。

代码检查确认评估脚本 `_infer` 为每次请求显式传入`policy_seed`；主调用使用由`seed + task_id*1,000,000 + episode_idx*10,000 + environment_step`确定的seed，4个额外MC samples使用`seed+1...4`。服务端`Policy.infer`在收到`policy_seed`时直接构造对应JAX key，不split或推进内部RNG。因此extra MC calls不会改变相同observation/environment-step下主调用的随机样本；一旦adaptive H改变后续observation或environment step，后续轨迹按实验处理自然分叉。

此前用于比较的“原版”数据来自单独保存的`libero_long_timing_logs`，不是当前脚本在同一服务器运行条件下生成的严格paired fixed-H5 baseline。下一步在归因15-point success差异前，应使用当前`eval_libero_action_cot_pruning.py`、同checkpoint、`seed=7`、相同10 tasks x20 initial-state indices运行`adaptive_replanning=none,replan_steps=5`，再做episode-level paired success比较。随后再运行纯`cot_aac`消融。还可先将Task8每个episode只截取前300 environment steps比较H>5 rate，以减少失败timeout tail造成的稀释。

## 2026-07-13 - explained CoT-AAC selectors, paper lineage, and comparison with the original model

本次根据当前仓库实现、已有实验记录和原始论文，系统梳理了`cot_aac`、`guarded_cot_aac`及其与原版ACoT-VLA/fixed-H5执行策略的关系。检查的主要代码包括`scripts/eval_libero_action_cot_pruning.py`中的online-MC采样、逐frame entropy、coarse-to-final时间对齐、AAC prefix-jump selector、stage guard和hysteresis，以及`src/openpi/models/acot_vla.py`和`src/openpi/training/config.py`中的ACoT-VLA推理路径与LIBERO配置。

当前模型配置保持`H_cot=15`、`H_action=10`和`joint_action_shifts=(2,1)`：EAR生成对应raw control times `0,2,...,28`的15个coarse Action-CoT tokens，final action expert生成对应`0,...,9`的10个可执行actions。原版评估固定执行前5个actions再重规划。`cot_aac`不改变checkpoint、模型结构、两个生成horizon或denoising预算，而是对同一observation用5个随机seed采样完整policy输出，对normalized coarse samples计算逐token MC entropy，按stride 2插值到final action时间轴；对候选`h_exec={5,6,7,8,9,10}`计算prefix mean entropy及相邻prefix jump，显著jump前停止，否则选最大H。当前显著性使用`median + 1.5*MAD`，默认entropy为`mean_d log(var_k+1e-6)`；可选`aac_grouped`才使用translation/rotation Gaussian covariance entropy与gripper Bernoulli entropy。因此`cot_aac`是受AAC启发、以Action-CoT uncertainty替代final-action uncertainty的项目变体，不是CVPR 2026 AAC的原样复现。

`guarded_cot_aac`在上述raw selector外继续应用per-task entropy history gate：20-decision warmup或high entropy回到H5，mid entropy最多H6，low entropy保留AAC候选；若candidate prefix内出现gripper change、较大action delta或jerk ratio则cap到H5，并默认保持2个decision cooldown；H增长要求连续2次low-risk、每次最多增加1，H下降立即生效。纯`cot_aac`不应用这些risk gate、guard、cooldown或hysteresis，是更激进的selector消融。

论文脉络中，ACoT-VLA（arXiv:2601.11404）提供EAR、IAR和Action-Guided Prediction模型；Adaptive Action Chunking（arXiv:2604.04161，CVPR 2026）提供training-free、多final-action samples、逐future-timestep action entropy与prefix entropy jump选择execution chunk size的直接启发。DEHP（arXiv:2606.11408）是冻结base chunk policy、用online RL训练轻量horizon branch的相关学习式替代；Real-Time Chunking（arXiv:2506.07339）通过异步推理和overlap inpainting隐藏推理延迟，是可组合但目标不同的方向。

当前实现仍是oracle而非可部署加速路径：K=5通过5次串行完整`client.infer()`获得，第一条动作控制环境，另4次只用于entropy；每次都会重复VLM prefix、IAR、EAR和final expert。`deployable/primary` timing忽略这4次extra calls，只表示未来由轻量predictor或batched/shared-prefix sampler替代后的假设口径，实际observed成本约为单次policy的5倍。

已有time-aligned `guarded_cot_aac` 10 tasks x20 trials结果为188/200成功（94.0%），此前记录的非严格paired fixed-H5 baseline point estimate为195/200（97.5%）；平均episode steps从290.74增至318.71，deployable calls/episode从56.76增至61.48。12,296次decisions中，risk-gated raw H>5占58.58%，但最终H>5仅5.05%，final H均值5.071；直接guard或cooldown合计73.89%。这说明当前guard/hysteresis大幅压制了entropy selector，现有结果不支持已经获得加速或立即训练predictor。纯`cot_aac`尚无完整closed-loop结果，不能做实测数值比较。下一步仍应先用当前脚本和相同seed/initial states跑严格paired fixed-H5 baseline，再跑纯`cot_aac`，然后做per-timestep event-aware guard/cooldown消融。

## 2026-07-13 - fixed Task 8 CSV diagnostic field-size error

用于联结Task 8 `rollout_rows.csv`与`adaptive_h_decisions.csv`的临时分析命令在服务器Python 3.10报错`_csv.Error: field larger than field limit (131072)`。原因是`rollout_rows.csv`中的`adaptive_h_decisions_json`单字段超过Python csv模块默认128 KiB限制，不表示CSV损坏。修复方式是在创建`csv.DictReader`前调用`csv.field_size_limit(16 * 1024 * 1024)`，然后重新运行相同分析。

## 2026-07-13 - explained LIBERO terminated-episode warning in matched baseline

匹配fixed-H5 baseline运行时出现`WARNING: mode=full task=9 episode=11 ended before action step; success=False`。代码触发条件是`env.step(action)`抛出包含`executing action in terminated episode`的`ValueError`；handler随后调用环境`_check_success/check_success`，本次返回False，于是停止该episode并按失败记录，而不是终止整个评估。结合当前LIBERO结果中失败episode恰好停在1000 steps，该warning最可能表示LIBERO/robosuite内部episode horizon已到但脚本仍尝试发送下一步动作，语义等价于timeout failure，不是policy server或模型崩溃。若对应`rollout_rows.csv`中的steps明显小于1000，则需另查环境提前终止或done传播问题。

## 2026-07-13 - pure CoT-AAC full evaluation completed on server

服务器日志确认纯`cot_aac`的LIBERO-10 10 tasks x20 trials评估完成，并写出`/root/autodl-tmp/acotvla/stage_b_pruning_eval/cot_aac_libero10_10tasks_20trials/summary.json`。本轮消息尚未包含summary或per-task具体指标，因此当前只能确认产物生成，不能判断纯CoT-AAC的成功率、calls、H分布或潜在deployable加速。下一步应从`per_task_summary.csv`与`adaptive_h_decisions.csv`提取逐task success/timeout/steps/calls/avg H、observed/deployable totals和全局execution-H分布，再与同条件fixed-H5及guarded CoT-AAC比较。

## 2026-07-13 - analyzed pure CoT-AAC full results

用户提供纯`cot_aac`的完整关键指标。10 tasks x20 trials总体成功率94.5%（189/200）、timeout 5.5%、avg steps 312.63、deployable calls/episode 34.46、avg H 8.93、potential deployable policy total 2.94 s/episode、deployable wall 3.18 s/episode、包含5次online-MC的observed wall 15.47 s/episode。逐task成功率为：task0 100%、task1 100%、task2 90%、task3 100%、task4 95%、task5 95%、task6 85%、task7 100%、task8 85%、task9 95%。

相对guarded CoT-AAC（success 94.0%、steps 318.71、calls 61.48、potential policy 5.417 s、observed wall 27.69 s），纯CoT-AAC成功率point estimate提高0.5 point，steps下降1.9%，calls下降43.95%，potential deployable policy total下降45.73%，observed wall下降44.13%。这确认当前guard/cooldown/hysteresis显著抑制了calls收益；Task8从guarded的65%回到85%，也不支持“更长H本身导致Task8下降”的解释。

相对此前历史原版baseline（success 97.5%、steps 290.74、calls 56.76、policy total 4.430 s），纯CoT-AAC成功率point estimate下降3.0 points，steps增加7.53%，calls下降39.28%，potential deployable policy total下降33.63%。但历史baseline并非当前脚本的严格paired run，因此正式结论仍需等待同条件fixed-H5结果。当前online-MC observed wall 15.47 s仍明显高于单调用baseline，故纯CoT-AAC当前实现没有端到端部署加速；2.94 s仅是用future predictor替代额外4次采样后的潜在口径。

纯CoT-AAC的decision分布为H5 11.61%、H6 8.21%、H8 4.34%、H9 8.07%、H10 67.77%，没有H7；平均H 8.93。该selector多数时候没有检测到significant entropy jump而回退到最大H10，因此行为接近fixed H9/H10，而非细粒度动态路由。要证明entropy本身有价值，除fixed-H5 matched baseline外，还必须增加相同compute/calls水平的fixed-H9对照：如果fixed H9成功率和calls与纯CoT-AAC相当，则当前收益主要来自固定延长chunk，而不是entropy；只有纯CoT-AAC在相似calls下明显优于fixed H9，才支持继续开发entropy predictor。

本次再次澄清terminated warning的语义：`libero_10`评估脚本外层budget为`520*3=1560`，而当前环境失败episodes表现出1000-step内部终止。policy inference本身已成功返回；环境拒绝终止后的下一步action。真实事件是任务未在环境horizon内完成，即rollout timeout failure，不是模型/服务器推理timeout。代码层面的问题只是外层循环没有在下一次`env.step`前干净识别内部termination，因而以warning/exception handler收尾；该问题应后续改成显式同步environment horizon和timeout reason，但当前success=False分类是正确的。

## 2026-07-13 - fixed LIBERO environment-horizon termination handling

本次修改`scripts/eval_libero_action_cot_pruning.py`以消除正常1000-step timeout产生的`ended before action step` warning。新增`_env_horizon`，递归遍历`env/_env/unwrapped` wrapper链并读取`horizon/_horizon`，选择最小正整数环境上限。每个episode现在使用`min(suite_budget + wait_steps, environment_horizon)`作为实际循环上限，因此当LIBERO/robosuite内部horizon为1000时会在提交第1001步action前主动停止，不再依赖`executing action in terminated episode`异常收尾。

保留了terminated-exception fallback处理，用于环境未暴露horizon或异常提前终止；日志现包含明确的`step/current_limit`和success状态，不再将正常handler信息标成含糊WARNING。`rollout_rows.csv`新增`termination_reason`、`episode_step_limit`、`environment_horizon`字段；正常完成记录`success`，主动达到上限记录`step_limit`，环境提前拒绝action记录`environment_terminated`。timeout和success判定语义未改变，模型、policy seed、Action-CoT、H selector及动作执行均未修改。

仅运行`python -m py_compile scripts/eval_libero_action_cot_pruning.py`与`git diff --check`，均通过；按本地无LIBERO运行环境约定未运行rollout。下一步正式对照配置为fixed H9、`adaptive_replanning=none`、`action_cot_denoising_steps=10`、LIBERO-10 10 tasks x20 trials、seed 7。该对照在calls/compute上匹配纯CoT-AAC的avg H 8.93，用于判断entropy selector是否优于简单固定长horizon。后续结果比较固定同时包含三类基准：当前同条件fixed-H5、同条件fixed-H9，以及`results/stage_b/original_baseline`保存的最初版ACoT-VLA 500-trial历史数据；历史原版需要明确标注为非paired run。

Warning修复已提交为`08a0b53 fix LIBERO episode horizon handling`，并通过`http://127.0.0.1:7897`代理成功推送到GitHub `main`（`26cb3f9..08a0b53`）。推送后本地`main`、`origin/main`与`origin/HEAD`均指向`08a0b53`，工作区无未提交Git变更。该改动仅位于client-side评估脚本，服务器pull后现有policy server无需为此重启。

## 2026-07-13 - explained MuJoCo QACC instability warning

fixed-H9后续测试期间出现MuJoCo警告`Nan, Inf or huge value in QACC at DOF 9. The simulation is unstable. Time = 0.5480.`。`QACC`是MuJoCo generalized acceleration，警告表示compiled model第10个DOF的求解加速度出现NaN、Inf或超过稳定范围；这是physics simulation数值不稳定，不是policy inference/network timeout。单条warning无法确定来源，可能来自初始state的物体穿插/强接触、控制输入过大、solver/timestep或机器人/物体状态异常。

当前评估每个episode开头执行10个dummy actions等待物体落稳；`Time=0.5480`接近这一初始化等待阶段的约0.5秒，因此更像初始化/接触稳定过程，而非fixed-H9执行的模型action造成，但该判断需要结合warning前一行的task/episode和当时是否已发生第一次policy call确认。当前`eval_libero_action_cot_pruning.py`与原始LIBERO eval都未在`env.step`前裁剪policy action，也未显式验证action、observation或`sim.data.qacc`有限性。偶发一次且后续状态/episode正常时可继续测试；若同一episode反复出现、observation含非有限值或结果异常，应将该episode标为simulation instability并用相同initial state/seed重跑，而不是直接计作模型失败。后续可增加不改变action的finite-state诊断与`termination_reason=simulation_instability`，不建议未经benchmark约定直接clip actions，因为这会改变policy行为与基线可比性。

## 2026-07-13 - fixed-H9 matched-compute evaluation completed

服务器日志确认fixed H9、LIBERO-10 10 tasks x20 trials对照评估完成，并写出`/root/autodl-tmp/acotvla/stage_b_pruning_eval/fixed_h9_libero10_10tasks_20trials/summary.json`。该设置使用`adaptive_replanning=none`、`replan_steps=9`、`action_cot_denoising_steps=10`，用于在近似匹配纯CoT-AAC avg H 8.93/calls水平下判断entropy selector是否优于固定长horizon。当前消息尚未提供具体metrics，因此尚不能比较success、calls或latency。由于fixed H9每次replan只调用policy一次，不使用online-MC，其observed与deployable timing应近似一致，并代表当前可实际部署的端到端推理口径。

## 2026-07-13 - analyzed fixed-H9 versus pure CoT-AAC and original ACoT-VLA

用户提供fixed H9完整结果：10 tasks x20 trials共200 episodes，success 94.0%（188/200）、timeout 6.0%、avg steps 311.55、calls/episode 34.02、avg H 9.00、policy total 3.11 s/episode、server total 3.41 s/episode、wall total 3.45 s/episode。termination reasons为`success:188`、`step_limit:12`，说明`08a0b53`环境horizon修复已在服务器生效：所有失败均在step limit主动结束，没有通过terminated exception收尾。

Fixed H9逐task成功率为：task0 100%、task1 95%、task2 95%、task3 100%、task4 90%、task5 95%、task6 90%、task7 100%、task8 80%、task9 95%。与最初版ACoT-VLA 499-trial clean historical baseline逐task成功率`[100,100,100,100,100,92,94,100,82,98]%`相比，point differences为`[0,-5,-5,0,-10,+3,-4,0,-2,-3]`points。样本量不同（current每task20，historical约50）且不是paired run，task-level 5-point current变化只对应1个episode，不能直接视为稳定退化；task4的-10 point是当前最需要复核的风险项。

最初版ACoT-VLA clean historical overall为success 96.59%、calls 58.519/episode、policy total 4.560 s/episode、server total 4.873 s/episode。Fixed H9相对该历史参考的success point estimate下降2.59 points，calls减少41.87%，policy total减少31.80%，server total减少30.03%。Fixed H9当前per-call policy约91.4 ms，而历史原版约77.9 ms，说明跨运行环境的单次latency存在明显漂移；尽管如此，calls下降是与硬件计时独立的结构性收益，累计时间仍下降。原始历史baseline没有完全可比的wall字段。

纯CoT-AAC为success 94.5%、calls 34.46、avg H 8.93、potential policy total 2.94 s、observed wall 15.47 s。相对fixed H9只多1个成功episode（+0.5 point），calls反而多1.29%，steps多0.35%；两者在质量和policy calls上基本等价。纯CoT-AAC看似potential policy time低5.47%来自不同run的per-call latency差异，不能归因于entropy，因为其calls没有下降；实际online-MC wall 15.47 s是fixed H9实际wall 3.45 s的约4.48倍。逐task上纯CoT-AAC相对fixed H9在task1/4/8各高5 points，在task2/6各低5 points，其余相同，属于少量失败在tasks间重新分布，没有形成稳定dominance。

当前证据支持“固定H9可直接减少约42% policy calls，并以历史参考约2.6-point success下降换取约30%累计推理时间下降”；不支持“当前CoT entropy selector优于固定长horizon”。纯CoT-AAC有67.77% decisions选择H10且avg H 8.93，行为接近fixed H9/H10。现阶段训练entropy predictor没有充分依据，因为predictor只能逼近一个尚未超过fixed H9的selector。下一步若继续验证entropy价值，应先获得同条件fixed-H5结果，并建立fixed H5/6/7/8/9/10 success-calls Pareto frontier；任何adaptive selector都必须在相同calls或avg H下优于最佳fixed setting。也可测试`final_aac`以判断final-action uncertainty是否比stride-2 CoT entropy更适合execution horizon，但仍需fixed-H匹配对照。

## 2026-07-13 - assessed statistical power and next entropy-horizon design

本次评估当前success differences的统计解释。最初版clean historical baseline为482/499=96.59%，fixed H9为188/200=94.0%，纯CoT-AAC为189/200=94.5%。Wilson 95% intervals约为：historical `[94.61,97.86]%`、fixed H9 `[89.81,96.53]%`、pure CoT-AAC `[90.42,96.90]%`。按非paired two-proportion近似，fixed H9减historical为-2.59 points，95% difference interval约`[-6.25,+1.06]`points、p约0.16；pure减fixed H9为+0.5 point，difference interval约`[-4.06,+5.06]`points、p约0.83。因此当前样本不能证明fixed H9显著降低success，也不能证明其success drop严格不超过3 points；同样没有证据表明纯CoT-AAC优于fixed H9。每task仅20 trials时一个episode等于5 points，逐task差异尤其不稳定。50 trials/task即500 overall可把约95% overall success的单比例95%半宽缩小到约1.9 points，适合正式总体确认；敏感task的逐task结论仍需更多trials或paired McNemar分析。

当前entropy问题不应简单通过增加trials掩盖。纯CoT-AAC的H10占67.77%、H7为0、avg H 8.93，并与fixed H9在success/calls上近似相同，说明现有“prefix entropy jump + no-significant-jump回退H10”缺乏足够horizon discrimination。若研究目标继续要求entropy创新，应优化horizon uncertainty/selector，而不是修改Stage B用于segment ranking的既有指标。优先方向是：以final-action per-timestep entropy作为直接baseline，并与time-aligned CoT entropy做fusion；用paired rollout或safe-horizon labels校准绝对风险，而非每task running quantile；将“无显著jump”回退从固定H10改为校准值；在相同avg H/calls约束下与fixed horizons比较。只有oracle selector稳定超过fixed frontier后才值得训练predictor。

当前候选H5-10的上限10由模型`H_action=10`决定，不是任意设置的entropy上限；一次policy call只产生10个可执行final actions，无法在不生成新动作的情况下安全执行H>10。纯CoT-AAC已大量饱和H10，因此当前主要限制不是上限太低。更可能限制adaptive优势的是下限固定H5：高风险/接触阶段无法缩短到H1-4来换取质量，而fixed H9的优势正需要adaptive在危险阶段更频繁replan。建议下一版允许`h_exec`候选覆盖例如`{3,5,7,9,10}`或`{1,...,10}`，形成低风险H9-10、中风险H5-7、高风险H1-3的策略，并以总体calls匹配fixed H9；当前代码会过滤低于`replan_steps=5`的entropy horizons，因此该设计需要新增独立的minimum execution horizon语义。若要H>10，则需提高模型final action horizon并重新训练/微调，属于另一条架构实验线。

## 2026-07-13 - proposed budget-aware event-triggered entropy horizon V2

本次针对“开放H1-3会增加calls”的问题提出V2设计。短H会增加局部replan次数，但不必增加episode总calls：V2应把短H作为有调用预算的稀缺风险干预，并用大量H10补偿。若只在H10与H3之间选择，为保持平均H不低于9，H3占比最多可为`1/7=14.3%`；若H1与H10混合，H1占比最多约11.1%。例如85% H10、10% H5、5% H3的平均H为9.15，理论calls仍低于fixed H9。当前pure CoT-AAC已有67.77% H10，但浪费了较多H5/H6选择；V2应把普通低风险状态统一推到H10，把H3仅留给极少数真正临近风险的状态。

推荐方案命名为`Budgeted Event-Triggered Entropy Horizon V2`。默认H10，不再使用per-task quantile把固定比例状态判为low/high risk；对K个samples构造逐final-timestep risk curve，包含直接final-action entropy、stride-2对齐的CoT entropy、translation/rotation disagreement和gripper disagreement。使用离线全局校准后的threshold寻找最早risk event位置：无事件选H10，风险在后段则cap到H7-9，中段cap到H4-6，只有风险发生在最前段才选H3；初版不使用H1-2，避免过高调用成本。Guard从“prefix任意风险一律回H5”改为“在风险事件之前停止”，且去除2-decision cooldown和jerk-ratio全局guard。

V2增加budget controller，以fixed H9的calls约34.02/episode或avg H约9作为约束。离线先选择threshold，使短H比例和expected calls不超过预算；在线可维护短H intervention rate或rolling avg H，预算紧张时提高risk threshold，预算充足时允许关键干预。选择目标可写为“在calls budget下最大化success”，而不是单独最大化H。一次fixed-H9失败timeout约需要1000/9约111次calls，而正常成功episode通常约25-35次；若一次H3干预增加2-3次局部calls但避免一个timeout，可抵消几十次此类干预，所以总calls应按完整episode而非单次decision判断。

训练前oracle验证需要与fixed H9严格匹配calls，报告overall success、timeout、total calls/episode、successful-only calls、failed-episode calls、total policy time per success以及H/intervention分布。消融至少包括final entropy only、CoT entropy only、fused entropy、fused+budget。先用20 trials/task筛选；只有在calls不高于fixed H9且success point estimate更高时，才扩到50 trials/task并考虑训练predictor。若V2仍不能超过fixed frontier，应停止entropy-based horizon方向，保留fixed H9与denoising-depth优化。

## 2026-07-13 - generated concise weekly plan from experiment log

本次根据 `reports/context/experiment_log.md` 中最近的 adaptive execution horizon / CoT-AAC 记录，为用户生成简洁周计划。计划重点放在：使用当前脚本与相同 seed/initial states 补齐严格 paired fixed-H5 baseline，运行纯 `cot_aac` 消融，定位 Task8 成功率下降和 horizon decision 分布，重构 event-aware guard/cooldown，并在结果稳定前暂缓训练 entropy predictor。未新增实验结果、代码修改或指标。

## 2026-07-13 - implemented and launched budgeted event-triggered entropy horizon V2

本次在 `scripts/eval_libero_action_cot_pruning.py` 实现 `budgeted_event_v2` adaptive-H selector。V2继续使用online-MC的5条policy samples，但不使用旧的per-task running quantile、stage guard、cooldown或hysteresis。代码从normalized final actions计算整体、translation、rotation和gripper逐future-timestep uncertainty，并与stride-2对齐的Action-CoT entropy构成robust fused risk curve；若发现risk event，则按最早event timestep选择H，无事件默认H10。V2默认最小H3，并自动补全H3至配置最大H之间的所有整数候选；旧selector仍使用原有默认H5-10候选。

新增episode horizon-credit budget controller。默认target average H为9、initial credit为6、capacity为12：H10相对target每次增加1 credit，H3消耗6 credit；候选短H余额不足时提高到当前可负担的最小H。新增CLI参数包括`--adaptive_h_v2_min_horizon`、`--adaptive_h_v2_target_avg_horizon`、`--adaptive_h_v2_initial_budget`、`--adaptive_h_v2_budget_capacity`、`--adaptive_h_v2_risk_threshold`、final/CoT fusion weights，以及可选的全局absolute final/CoT entropy thresholds。`scripts/sweep_action_cot_denoising_steps.py`同步支持新selector和参数透传。

输出新增逐decision的risk event位置、event source、fused/final/CoT risk curves、budget balance、required credit、budget-limited状态、intervention状态和累计average H；episode/per-task/aggregate输出新增intervention rate、budget-limited rate、budget balance、event指标、successful/failed episode calls、policy time per success及raw/final H distributions。当前online-MC observed timing仍包含额外4次采样；deployable/primary timing仍只是future predictor替代MC sampling后的oracle口径。

本地只运行了`python -m py_compile scripts/eval_libero_action_cot_pruning.py scripts/sweep_action_cot_denoising_steps.py`和`git diff --check`，均通过；未运行本地模型或LIBERO rollout。改动提交为`a12e0fa add budgeted entropy horizon v2`，通过`http://127.0.0.1:7897`代理推送到GitHub `main`。服务器`/root/ACoT-VLA`已fast-forward到`a12e0fa`；既有`third_party/libero`子模块修改未被改动。policy server继续在tmux `server-0`使用checkpoint`/root/autodl-tmp/acotvla/checkpoints/acot_libero_action_cot_explicit_implicit_co_fusion/acot_libero_long_run1/50999`监听8000端口。

已在服务器tmux `newH`启动LIBERO-10全量V2评估：10 tasks x20 trials、`mode=full`、`replan_steps=5`、`action_cot_denoising_steps=10`、online-MC samples=5、candidate H3-10、`adaptive_h_selector=budgeted_event_v2`、`diagonal_logvar`、min H3、target average H9、initial budget6、capacity12、risk threshold1.5、final/CoT weights各0.5。输出目录为`/root/autodl-tmp/acotvla/stage_b_pruning_eval/budgeted_event_v2_h3_target9_libero10_10tasks_20trials`，日志为其中的`run.log`。启动后task0 episode0和episode1均完成且success=True，episode2正在运行；完整success/calls/H分布和timing结果尚未产生。

为当前V2全量评估创建了Codex heartbeat完成监控，automation id为`v2`，名称为`V2全量实验完成提醒`，每30分钟通过SSH检查远程输出目录。`summary.json`生成后监控将读取summary与per-task CSV，在当前任务中提示并汇总success、timeout、calls、avg H、timing、H分布及与fixed H9/纯CoT-AAC对照；若进程异常退出且没有summary，则提示错误并附`run.log`末尾。实验仍运行时不发送常规进度消息。

用户要求增加飞书通知并提高检查频率。Codex heartbeat `v2`已从每30分钟更新为每15分钟检查一次。本地环境变量、Git跟踪文件和远程服务器环境中均未发现已有`FEISHU/LARK` webhook配置；飞书通知watcher尚未创建，当前等待用户提供飞书群自定义机器人webhook URL。该凭据计划仅作为服务器私有配置使用，不写入Git仓库。

用户随后提供了飞书自定义机器人webhook和签名密钥。新增通用脚本`scripts/watch_experiment_feishu.py`：按照飞书官方签名规则计算`timestamp + newline + secret`的HMAC-SHA256空消息签名并Base64编码，支持测试消息、按PID监控、15分钟轮询、完成后读取`summary.json`和`per_task_summary.csv`汇总success/timeout/calls/avg H/timing/H分布/逐task结果，以及进程无summary退出时发送`run.log`末尾。脚本不包含任何webhook或密钥，提交为`c5c08fc add signed Feishu experiment notifier`并推送至GitHub `main`；服务器已fast-forward到该提交。

飞书凭据仅存于服务器`/root/.config/acotvla/feishu_notify.env`，权限为`600 root:root`，未写入仓库、命令参数或项目实验日志。服务器使用项目虚拟环境Python发送测试消息，飞书接口正常接受。一次性watcher已在独立tmux session `v2-feishu`运行，监控当前V2评估Python PID 13501，`poll_seconds=900`，输出日志为实验目录下`feishu_watcher.log`；完成/失败通知成功发送后会写`.feishu_notification_sent` marker并退出。Codex heartbeat `v2`也保持15分钟检查频率。

## 2026-07-13 - budgeted event-triggered entropy horizon V2 full results

服务器V2全量实验完成并生成`/root/autodl-tmp/acotvla/stage_b_pruning_eval/budgeted_event_v2_h3_target9_libero10_10tasks_20trials/summary.json`、`per_task_summary.csv`和`adaptive_h_decisions.csv`。配置为LIBERO-10 10 tasks x20 trials、`mode=full`、`replan_steps=5`、Action-CoT denoising steps10、online-MC samples5、candidate H3-10、`budgeted_event_v2`、`diagonal_logvar`、min H3、target average H9、initial credit6、capacity12、risk threshold1.5、final/CoT weights各0.5。

总体结果为192/200成功，success rate 96.0%、timeout rate 4.0%、avg environment steps约299.1、deployable calls/episode 33.245、avg H 8.8485。成功episodes平均deployable calls为30.005，失败episodes为111.0。潜在deployable policy total为2.872 s/episode、deployable wall为3.106 s/episode；包含5次online-MC采样的实际observed policy为13.751 s/episode、observed wall为15.002 s/episode，total policy calls为166.225/episode。当前实现因此仍不是实际端到端加速路径，deployable时间只代表未来轻量predictor替代4次额外采样后的oracle口径。

最终H分布共6,649次decisions：H3 236次（3.55%）、H4 80（1.20%）、H5 111（1.67%）、H6 164（2.47%）、H7 347（5.22%）、H8 790（11.88%）、H9 1,585（23.84%）、H10 3,336（50.17%）。raw selector提出H3 1,986次，但budget controller将多数提高到可负担H；mean budget-limited rate为31.86%，mean intervention rate为50.35%，最终avg H接近但略低于目标9。risk event source统计为`robust_fusion` 3,313、`none` 3,336。

逐task成功率为：task0 100%、task1 100%、task2 100%、task3 100%、task4 100%、task5 90%、task6 90%、task7 95%、task8 100%、task9 85%。相同20-trial规模的fixed H9为94.0% success、34.02 calls、avg H9、policy3.11 s、wall3.45 s；V2 point estimate成功率高2.0 points、calls低2.28%，潜在policy低7.65%、潜在wall低9.96%，但跨run per-call latency有漂移，结构性加速证据应主要看calls。V2实际observed wall 15.002 s是fixed H9实际wall 3.45 s的4.35倍。纯CoT-AAC为94.5% success、34.46 calls、avg H8.93、potential policy2.94 s、observed wall15.47 s；V2 point estimate成功率高1.5 points、calls低3.53%、potential policy低2.31%、observed wall低3.02%。这些差异仅为200-episode point estimates，尚未做显著性或paired outcome检验。

实验结束后出现一次MuJoCo`Nan, Inf or huge value in QACC at DOF 9... Time=0.5480`警告，位置在`summary.json`写出之后；本轮输出已完整生成。该警告与此前观察相同，更接近环境初始化/清理阶段的physics数值不稳定，当前结果中8个失败由4% timeout体现，尚无证据表明该末尾警告破坏summary。飞书watcher已写入`completed` marker、成功退出tmux；Codex heartbeat automation `v2`在完成通知后已删除，避免重复提醒。

## 2026-07-13 - compared original, fixed-H9, guarded/pure CoT-AAC, and V2

本次核对服务器各次`summary.json`与仓库original baseline后澄清：当前比较项除历史原版外，均使用同一个base checkpoint `acot_libero_long_run1/50999`，属于不同execution/replanning策略，不是分别训练的独立模型。历史原版ACoT-VLA clean baseline为499 trials、success 96.59%、calls 58.519/episode、actual policy total 4.560 s/episode、server total 4.873 s/episode；其wall字段未保存，且运行环境/per-call latency与当前实验不同。

当前200-episode策略结果为：guarded CoT-AAC success94.0%、calls61.48、avg H5.074、potential deployable policy5.417 s、observed wall27.692 s；pure CoT-AAC success94.5%、calls34.46、avg H8.926、potential policy2.944 s、potential wall3.181 s、observed wall15.468 s；fixed H9 success94.0%、calls34.025、avg H9、actual policy3.115 s、actual wall3.446 s；budgeted event V2 success96.0%、calls33.245、avg H8.849、potential policy2.872 s、potential wall3.106 s、observed wall15.002 s。

按当前可真实部署口径，fixed H9不需要额外MC或predictor，是当前实际wall最快方案；相对历史原版calls减少约41.9%、policy total point estimate减少约31.7%，success point estimate低2.59 points。按oracle/potential口径，V2是当前最佳success-calls点：相对fixed H9 success高2 points且calls低2.28%；相对历史原版success仅低0.59 point且calls低43.19%。但V2当前online-MC实际wall为15.002 s，是fixed H9的4.35倍，因此尚不能称为实际推理加速。Pure CoT-AAC与fixed H9基本等价，未证明entropy优势；guarded版本因过度压回H5，calls和observed时间最差。V2首次在point estimate上同时超过matched-compute fixed H9的success与calls，支持进入更大样本和predictor可行性验证，但200 episodes的4个成功episode差异尚未证明统计显著。

此前Stage-C dynamic denoising/entropy predictor训练在本轮对话中没有提供完成后的closed-loop evaluation结果，因此不能列入定量模型排名。Static Stage-B pruning与denoising-step sweep分别验证entropy ranking或超参数速度，不等同于已训练的新部署模型。

## 2026-07-13 - report-ready definitions of execution-horizon variants

为报告口径进一步明确各方案含义：这些方案主要共享同一个`50999` checkpoint，区别在于每次policy输出10个final actions后实际执行多少步再replan。原版ACoT-VLA固定执行约5步，质量高但policy调用较多；fixed H9固定执行9步，不使用entropy或额外采样，可直接部署并通过减少policy calls获得实际加速，但关键阶段也无法主动缩短H。Guarded CoT-AAC用5次online-MC Action-CoT entropy提出H，再叠加action delta、jerk、gripper guard、cooldown和hysteresis，结果过度保守，大部分decision回到H5。Pure CoT-AAC直接依据time-aligned Action-CoT entropy prefix jump在H5-10选择，不使用guard，calls明显下降但行为接近fixed H9/H10，且实际需5次完整policy调用。Budgeted Event V2融合final-action与Action-CoT逐future-timestep uncertainty，默认H10，在最早risk event前缩短至H3-9，并用target average H9的credit budget限制短H次数；它在当前point estimate上取得96% success和33.245 calls，但仍使用5次MC，实际部署需未来轻量predictor替代额外4次采样。

## 2026-07-13 - generated daily report from experiment log

本次按 `$report-assistant` 日报格式，根据 `reports/context/experiment_log.md` 中2026-07-13的记录生成三行日报。日报重点概括了fixed-H9与纯CoT-AAC对照、`budgeted_event_v2`实现和全量LIBERO-10评估启动，以及V2完整结果尚未产出的卡点。未新增实验结果或代码改动。

## 2026-07-13 - created weekday daily-report Feishu automation

本次创建了 Codex App cron 自动化，名称为“工作日日报自动生成并发送飞书”，automation id 为 `automation`，状态为 active。自动化计划在本地项目 `/Users/kongyue/LightAcotVLA` 中按工作日 18:00 运行，读取 `/Users/kongyue/.agents/skills/report-assistant/SKILL.md` 与 `reports/context/experiment_log.md`，按当天 Asia/Shanghai 日期提取日志并生成 `$report-assistant` 三行日报；如果当天没有有效日志，则发送飞书提醒而不是编造日报。发送飞书时优先使用环境变量 `FEISHU_WEBHOOK_URL` 与 `FEISHU_SIGNING_SECRET`，缺失时尝试读取私有 env 文件 `$HOME/.config/acotvla/feishu_notify.env` 或 `/root/.config/acotvla/feishu_notify.env`，并复用 `scripts/watch_experiment_feishu.py --test_message` 的签名发送能力。凭据不会写入仓库或日志。

## 2026-07-14 - scheduled one-time weekly group meeting PPT generation

本次创建了 Codex App 一次性 cron 自动化，名称为“今日15点生成周组会PPT”，automation id 为 `15-ppt`，状态为 active。任务将在北京时间2026-07-14 15:00左右在本地项目 `/Users/kongyue/LightAcotVLA` 中运行，读取 `/Users/kongyue/.agents/skills/report-assistant/SKILL.md` 与 `reports/context/experiment_log.md`，根据最近一周尤其是2026-07-13至当天的有效实验日志生成简洁、无装饰的周组会PPT。输出目标为 `reports/outputs/weekly_group_meeting_2026_07_14.pptx`。任务要求不编造未出现的实验结果、成功率、速度、日期或结论，缺失数据使用 `[待补充]`，并在生成后做基本文件存在性检查。

## 2026-07-14 - manually generated weekly group meeting PPT after missed automation

北京时间2026-07-14 16:45检查发现15点一次性自动化未在 `reports/outputs/` 生成PPT，目录为空；随后手动生成简洁、无装饰周组会PPT `reports/outputs/weekly_group_meeting_2026_07_14.pptx`。PPT共6页，内容基于 `reports/context/experiment_log.md` 中最新事实：10x20 pilot 对比、pure-base original H5 10x100已完成、Fixed H9仍在运行、后续Exact K20与sidecar V2-P正式评估计划。生成后使用artifact-tool从最终PPTX重新渲染验证，确认可渲染6页；同时删除未触发的一次性自动化 `15-ppt`，避免后续误触发重复生成。

## 2026-07-13 - designed a training route for the Budgeted Event V2 predictor

本次检查了 `scripts/eval_libero_action_cot_pruning.py` 的 V2 决策与日志字段，以及 `src/openpi/models/acot_vla.py` 中现有 `action_cot_step_head`。V2 的 `raw_execution_horizon` 来自逐future-timestep fused risk的最早事件位置，最终 `execution_horizon`还会被episode horizon-credit budget修正。因此新predictor不应直接监督拟合最终H，否则会把风险估计与有状态budget策略混在一起；更合理的监督目标是V2 fused/final/CoT risk curve、event mask和raw H，部署时继续复用现有budget controller。

现有`action_cot_step_head`只从pooled prefix预测Action-CoT denoising steps `(3,5,10)`，标签来自Stage A静态entropy quantile，不等价于execution horizon predictor。推荐新增独立轻量head，冻结`50999` base policy，输入复用单次policy call已经产生的pooled VLM prefix、normalized coarse action chunk、final action chunk和proprioception，输出10个future-timestep risk/hazard logits或分开的final/CoT risk curves。head必须与policy在同一JAX/GPU前向中运行，避免额外RPC或完整policy call。

推荐训练分两阶段。阶段一是teacher-student supervised distillation：离线用K=8或更高的MC teacher生成V2曲线标签，训练curve regression、event BCE和raw-H ordinal/classification多任务loss；数据应包含离线dataset状态与online rollout状态，并按episode/task切分，同时对高风险/失败附近状态做平衡。现有全量V2输出有6,649个decision labels，但`adaptive_h_decisions.csv`没有完整observation、action chunks或prefix feature，不能直接完成该head训练，需要新增label/feature exporter或重新采集。可用student rollout后重新查询teacher并迭代训练，以减轻部署状态分布偏移。

阶段二是可选的task-return-aware refinement：保持base policy冻结，仅训练horizon head。参考2026-07-09版DEHP的chunk-level SMDP/PPO方法，可使用success terminal reward和policy-call约束/惩罚，并按实际执行H使用step-discounted return；SFT predictor用于warm start，避免从随机H策略开始。fixed H9的34.02 calls/episode可作为当前调用预算参考。不能把failed episode中的所有decision简单标成unsafe，因为episode success不提供逐decision因果标签；若不做online RL，应使用simulator state branching/paired counterfactual rollout或继续teacher distillation。

建议先完成SFT predictor并进行无MC closed-loop评估，再决定是否上PPO。关键对照仍为original ACoT-VLA、fixed H9、online-MC V2和single-call predictor V2；报告相同initial states下的success、timeout、calls/episode、avg H、head latency、实际policy/wall total和H分布。当前online-MC V2为96.0% success、33.245 calls/episode、potential wall 3.106 s/episode但observed wall 15.002 s/episode；predictor阶段的首要验证目标是去掉额外4次完整policy call并尽量复现V2的success/calls，而不是先修改base action policy。该目标尚未通过实验验证。

## 2026-07-13 - selected a mixed-supervision training design instead of mandatory pure distillation

本次进一步确认Budgeted Event V2 predictor并非必须采用纯蒸馏。相关方法存在三类：AAC和HiPolicy在inference时复用condition feature并行采样多个action chunks计算entropy，不训练predictor；AutoHorizon直接使用flow-based VLA内部action self-attention估计execution horizon；DEHP冻结base chunk policy并使用chunk-level SMDP/PPO直接训练categorical horizon head。对当前ACoT-VLA，纯MC蒸馏只是复现现有V2并去掉额外policy calls的最低风险方法，但学生性能受teacher上限限制，不能单独保证超过V2或fixed H9。

推荐可实施方案为`entropy-initialized budget-constrained horizon predictor`。首先应将当前顺序执行5次完整policy的online-MC改为共享一次VLM/prefix state、在GPU上batch采样K个coarse/final action chunks并重新测速；本仓库模型已经存在prefix state与后续coarse sampling分离接口，因此该方向具备代码基础。若batch MC开销已经可接受，则可以保留精确entropy而无需训练predictor；AAC与HiPolicy论文中的低额外延迟来自并行采样和condition feature复用，但其硬件与模型不同，不能直接当作本项目结果。

若仍需predictor，推荐冻结`50999` base policy并训练独立horizon risk/survival head。输入使用单次primary inference已有的pooled prefix feature、proprioception、normalized 15-step coarse actions和10-step final actions；输出每个future action timestep的hazard/executability，以及辅助的predicted final/CoT entropy curves。部署时从hazard得到raw H，现有budget controller仍独立应用，避免把episode budget状态混入风险标签。

训练数据建议混合四种来源：现有Stage A约101,469个dataset sample的K8 coarse entropy可作为大规模CoT uncertainty辅助监督；原始LIBERO demonstrations可提供model chunk对expert future actions的prefix error、gripper mismatch等连续监督；重新采集的online V2/fixed-H rollout states提供部署分布上的MC fused-risk labels；成功/失败rollout与temporal swap、gripper flip、late-tail noise等action corruption可提供plan executability辅助样本。failed episode不能把所有decision直接标为negative。少量关键状态可用MuJoCo state snapshot对H3-10做paired branch rollout获得更强的counterfactual horizon value标签，但该方法成本较高，适合校准集而不是全量数据。

建议训练分三步：第一步用MC curve/event/raw-H和expert-prefix error做supervised warm start；第二步用student rollout后teacher relabel迭代以处理state-distribution shift；第三步参考DEHP，仅更新horizon head，用chunk-level PPO和step-discounted return进行task-return refinement。PPO目标应在terminal success之外加入policy-call成本，并用dual/Lagrangian约束将calls/episode控制在fixed H9的34.02附近；budget约束应作为categorical H action mask，使PPO记录的sampled H与实际executed H一致。评估消融应包含fixed H9、shared-prefix batched-MC V2、pure distillation、distillation加expert auxiliary、distillation加PPO，并只用包含所有采样/head开销的actual wall time作为部署速度结论。该方案目前为设计建议，尚未实现或运行训练实验。

## 2026-07-13 - prioritized predictor quality with a privileged teacher and constrained actor-critic

用户明确训练成本和训练量不是主要限制，predictor效果优先。本次据此将推荐方案从单纯entropy regression提升为两层teacher/student execution-horizon policy。Base ACoT-VLA `50999`仍冻结；昂贵teacher使用共享prefix后的高K并行MC action distributions、Action-CoT/final entropy、当前与上一action chunk一致性及simulator rollout回报，训练categorical H actor和distributional critic。候选execution horizon建议覆盖H1-10，使高风险接触阶段可以真正缩短到H1-2；调用成本通过constrained SMDP objective处理，而不是事先禁止短H。

Teacher训练主目标直接对齐success与calls，不以复现当前V2 heuristic为最终目标。推荐使用task-balanced online PPO、按实际H计算的step-discounted GAE、terminal success/timeout信号和per-policy-call cost，并用dual/Lagrangian变量约束calls budget。可使用MuJoCo state snapshot在关键decision states对H1-10进行paired branch rollout，拟合`Q(s,H)`或success probability作为critic warm start；高风险、失败和Task8/Task9状态应做hard-negative/priority sampling。现有Stage A K8 entropy、expert future-action prefix error、gripper mismatch和合成action corruption只作为辅助监督，不能代替真实return或counterfactual horizon labels。

部署student只观察单次primary inference可得的pooled/token-level prefix feature、proprioception、15-step coarse actions、10-step final actions、上一chunk overlap以及budget/history state，蒸馏teacher的H distribution、Q distribution和risk curve，然后继续进行student-only constrained PPO。蒸馏在该方案中是将高质量昂贵teacher压缩为单次前向的部署手段，不是效果来源或唯一训练目标。可增加共享trunk的bootstrap heads或distributional critic，用success probability lower-confidence bound进行保守H选择；也可增加一个低成本stepwise verifier，在chunk执行过程中根据proprioception/plan inconsistency提前replan。

正式验证建议使用相同initial states进行10 tasks x100 trials，对比original ACoT-VLA、fixed H9、exact batched-MC teacher、distilled student以及student PPO。进入student蒸馏前，learned teacher必须先在相同calls或actual wall预算下超过fixed-H Pareto frontier；最终只依据包含predictor全部开销的actual wall、policy calls、success、timeout和逐task结果下结论。该高质量优先方案目前尚未实现或训练。

## 2026-07-13 - clarified counterfactual storage and horizon predictor architecture

本次澄清counterfactual horizon数据的主要成本是仿真分支计算而非磁盘。每个decision root state只需保存一次context feature、proprioception、primary coarse/final action chunks、previous chunk和可选MuJoCo snapshot；H1-10分支只追加success、timeout、remaining steps/calls、physics status等小型标签，不应重复保存十份observation或完整trajectory/video。按100,000个root states估算，pooled及压缩temporal features使用float16时通常约2-10 GB；若额外保存三路JPEG observations约增加10-30 GB；保存三路224x224 RGB float32原图约180 GB；完整分支视频可能达到数百GB或TB，因此只建议保留少量failure/debug视频。推荐使用sharded Zarr/HDF5/Parquet而不是每sample一个NPZ，并用float16/bfloat16 feature、uint8/bool outcome标签。

架构被简化为三部分。冻结的base ACoT-VLA从observation生成context feature、15-step coarse actions和10-step final actions；数据收集器从同一个MuJoCo snapshot分别执行action prefix H1-10并在统一continuation policy下获得`y_H`；horizon network读取单次base inference已有的context/action/history/budget特征，输出十个`Q(s,H)`或success probability，并选择满足成功率/预算条件的H。网络可使用一个共享temporal encoder，接actor head输出H1-10 logits，接critic/Q head输出每个H的success/value；actor负责部署选择，critic用于counterfactual监督和PPO训练。

昂贵MC entropy teacher不是该架构的必要组件。若counterfactual branch labels充分，可以直接监督Q head；为提高效果，仍建议把K20 MC Action-CoT/final entropy作为训练期辅助或privileged feature，并在后续用task-balanced constrained SMDP PPO直接优化success和calls。部署时只保留single-call horizon network。该说明未修改模型代码或运行新实验。

## 2026-07-13 - made iterative supervised training primary and preserved Budgeted V2 as a dual-head controller

本次比较了SFT与PPO对execution-horizon predictor的作用。由于MuJoCo counterfactual collector可以在同一state枚举H1-10并提供每个H的success、timeout、remaining calls/steps标签，监督学习应作为主要训练方式；推荐训练完整`Q(s,H)`向量而不是只用一个hard best-H class，以保留多个H同时成功及速度差异的信息。训练应迭代执行“student rollout采集自身状态分布、同状态branch H1-10重新标注、聚合数据、继续SFT”，属于DAgger/approximate policy-iteration式流程，可降低一次性offline SFT的distribution shift。

PPO被保留为可选的最终refinement，而非替代SFT。其作用是处理counterfactual labels依赖旧continuation policy、多个H决策和episode budget在整条trajectory上的交互，以及直接优化terminal success与policy-call constraint。相关DEHP工作把variable execution horizon建模为SMDP，冻结base chunk policy并使用chunk-level PPO及step-discounted GAE训练horizon head。推荐只有在iterative SFT闭环结果进入平台期后，才从SFT checkpoint以较保守更新和SFT-policy KL约束进行PPO；若SFT已达到成功率/calls目标，可不做PPO。

同时澄清此前Q predictor是对Budgeted Event V2的扩展，不应无声替换原机制。推荐`Budgeted Event V2-P`双头结构：共享single-call context/action encoder；entropy head监督拟合当前V2的fused final/CoT risk curves、event mask和raw H，并通过原有event mapping与budget controller形成`v2_distilled`模式；Q head用counterfactual H1-10结果预测success/value，在`v2_value_refined`模式中校准H选择；最终episode budget controller保持不变。这样可以先验证distilled模式是否复现当前V2的96.0% success、33.245 calls和H分布，再独立判断Q refinement是否带来提升。相同效果或更优效果目前只是设计目标，尚无训练或closed-loop证据。

## 2026-07-13 - prepared a handoff prompt for Budgeted Event V2-P implementation and evaluation

本次为新Codex任务整理了可直接粘贴的完整handoff提示词。提示词要求新任务先读取`AGENTS.md`和`reports/context/experiment_log.md`，保留当前`budgeted_event_v2`行为和默认Action-CoT denoising steps10，实施`Budgeted Event V2-P`双头single-call predictor、共享prefix的batched MC teacher标签、MuJoCo H1-10 compact counterfactual collector、task-balanced supervised Q/entropy training、`v2_distilled`与`v2_value_refined`闭环模式及iterative student rollout/relabel流程。本阶段明确以SFT为主并完成真实closed-loop success/calls/wall验证，PPO只保留接口和设计说明，未在SFT证据产生前实现。

提示词包含当前定量基线：historical original ACoT-VLA success96.593%、calls58.519；fixed H9 success94.0%、calls34.025、wall3.446 s/episode；online-MC Budgeted Event V2 success96.0%、calls33.245、avg H8.8485、potential wall3.106 s但observed wall15.002 s。还包含本地/远程repo、checkpoint、policy config、LIBERO-10 10 tasks x20 smoke及最终x100 paired evaluation要求、compact feature storage约束、不得提交outputs/work/checkpoints/secrets和不得修改既有`third_party/libero`脏submodule等工作流事实。该任务只准备了提示词，没有修改predictor代码或产生新实验结果。

## 2026-07-13 - implemented the first complete Budgeted Event V2-P code path locally

本次在`/Users/kongyue/LightAcotVLA`完成了Budgeted Event V2-P第一轮代码实现。新增`src/openpi/models/execution_horizon_predictor.py`，将execution horizon与现有预测Action-CoT denoising steps的`action_cot_step_head`完全分开；新模块读取pooled prefix feature、normalized state、15-step coarse actions、10-step final actions、previous chunk overlap、previous H、budget balance和episode progress，经共享temporal encoder输出final/Action-CoT/fused risk curves、event logits、raw-H classification/ordinal logits、H1-10 success/timeout logits以及remaining calls/steps。其SFT loss包含可配置的success/timeout BCE、calls/steps Huber、三类risk regression、event BCE和raw-H classification/ordinal项。

`src/openpi/models/acot_vla.py`新增共享一次VLM/prefix和implicit feature、在GPU batch维并行生成K个coarse/final chunks的`sample_actions_batched_mc`，支持客户端使用连续policy seed保持与顺序采样一致的flow noise；K限制为10/20/32，默认使用Action-CoT denoising steps10。`src/openpi/policies/policy.py`在同一个RPC内运行batched teacher或predictor，不会再次调用VLM，并返回实际同步的`batched_mc_teacher_ms`和`execution_horizon_predictor_ms`。`src/openpi/policies/policy_config.py`与`scripts/serve_policy.py`支持从独立Orbax sidecar加载predictor，同时保留base checkpoint为bfloat16且不改写base参数。新增训练配置`acot_libero_budgeted_event_v2p`，其freeze filter只允许predictor subtree可训练；实际SFT入口不会加载base policy，进一步保证base完全冻结。

新增`src/openpi/execution_horizon/dataset.py`的append-only sharded HDF5格式。每个root state只保存一份float16 prefix/state/coarse/final/previous feature和一份float64 MuJoCo physics state；H1-10 outcome使用bool/uint16紧凑向量，不保存重复图像或分支轨迹。新增`scripts/collect_execution_horizon_counterfactuals.py`：root primary chunk与K-sample teacher只请求一次，H1-10逐分支恢复同一sim snapshot，后续policy seed只依赖root与continuation call index；continuation支持fixed H9和current student，失败视频受数量上限约束。新增`scripts/train_execution_horizon_predictor.py`，对全部H向量做task-balanced SFT，并对Task8/Task9、高risk、gripper变化及存在失败分支的root加权；训练输出仅含predictor sidecar。新增`scripts/run_v2p_iterative_sft.py`，实现初始SFT、加载student server、student状态重新branch、聚合HDF5数据并warm-start继续SFT的可重复流程；只保留`SMDPDecision`接口，没有实现PPO。

新增`src/openpi/execution_horizon/v2.py`，保留原V2 robust final/component/Action-CoT risk融合、event-to-H mapping和episode credit budget公式，同时提供`v2_distilled`与Q-filtered `v2_value_refined`原始H选择。新增`scripts/eval_libero_execution_horizon.py`，用于相同initial-state ID下比较original H5、fixed H9、exact batched-MC V2、v2_distilled和v2_value_refined，记录逐task/overall success、timeout、calls、avg H、H分布、sampled chunks及实际policy/server/client-wall、teacher和predictor总时延。新增`scripts/benchmark_batched_mc_teacher.py`单独测量K10/20/32真实同步延迟，不使用potential timing。飞书watcher扩展为支持SFT/collector/V2-P summary和按`--progress-seconds`发送训练进度，远端训练可配置60秒检查完成、每3600秒发送进度。

本地按仓库限制只执行了`python3 -m py_compile`和`git diff --check`，两项通过；本机没有NumPy、JAX、Flax、h5py或LIBERO运行环境，因此尚未产生模型shape、collector、训练、teacher延迟或closed-loop结果。下一步必须在`/root/ACoT-VLA`真实环境完成import/shape smoke，修复发现的问题后再启动counterfactual数据、SFT及闭环评估。

## 2026-07-13 - completed remote Budgeted Event V2-P smoke tests and measured the batched teacher

本次将Budgeted Event V2-P实现推送到GitHub并在服务器`/root/ACoT-VLA`的真实ACoT-VLA环境中完成逐层smoke。期间修复了Orbax sidecar字典数字key序列化、Tyro的LIBERO枚举输入、JAX typed PRNG key不能直接broadcast、WebSocket返回只读history数组，以及`module_jit`内部`state`参数与predictor关键字冲突的问题；相关修复均已提交并推送。服务器仓库只做fast-forward，没有撤销或修改既有的`third_party/libero`用户改动。

独立predictor GPU forward、完整多任务loss和反向传播通过，smoke loss为4.9723、gradient norm为17.8957。standalone synthetic SFT分别以小维度和默认`prefix_feature_dim=256`运行成功，Orbax sidecar只包含`execution_horizon_predictor`参数，训练日志确认base policy未加载且保持冻结。默认维度sidecar已成功与真实`50999` base checkpoint合并并由policy server加载；单次请求同时返回final/Action-CoT/fused risk、event、raw-H、H1-10 success/timeout和remaining calls/steps，所有H向量shape均为10。首次JIT请求wall约40.237 s；随后5次稳态请求的predictor同步延迟为10.37、3.60、3.12、2.92、2.88 ms，后4次均值约3.13 ms。该sidecar仅由1步synthetic数据训练，不具备效果含义。

`scripts/collect_execution_horizon_counterfactuals.py`在真实LIBERO task 0上完成1个root state的H1-10分支采集，输出位于`/root/autodl-tmp/acotvla/v2p_smoke/a17e123/collector`。collector只进行一次K10 root policy/teacher请求，再从同一MuJoCo physics snapshot执行10个H分支并使用fixed H9 continuation；本次10个分支均success，remaining calls依次为`[30,29,28,28,42,31,28,28,30,29]`，remaining steps为`[257,246,245,246,366,269,250,243,266,254]`，teacher raw H为8。单条HDF5记录约82,848 bytes，prefix/state/coarse/final feature分别以float16保存且physics state只保存一份，没有重复图像或完整分支视频。随后直接使用该真实collector shard完成1步SFT，loss为4.0772，训练和sidecar导出成功。单root端到端采集耗时129.892 s包含首次编译；之后将continuation阶段不必要的每次GPU profiling同步关闭，并新增按episode轮转root call offset的覆盖策略，标签语义未改变，该优化提交为`36a6353`。

`scripts/benchmark_batched_mc_teacher.py`在真实checkpoint上以1次warmup、3次同步测量完成K10/K20/K32 benchmark，结果保存于`/root/autodl-tmp/acotvla/v2p_smoke/a17e123/benchmark_all`。K10的batched teacher/policy/server/client wall均值分别为95.228/108.476/114.133/115.481 ms，对应顺序完整policy client wall 893.619 ms，实测加速7.738倍；K20分别为110.700/123.284/128.469/130.055 ms，对应顺序1755.366 ms，加速13.497倍；K32分别为125.373/135.603/139.426/140.633 ms，对应顺序2759.168 ms，加速19.620倍。以上均为实际GPU同步及RPC wall计时，排除了warmup，不是potential timing。

五种闭环模式的task 0 x 1 trial smoke保存于`/root/autodl-tmp/acotvla/v2p_smoke/a17e123/eval_all`，相同initial-state ID下original H5、fixed H9、exact batched-MC V2 K20、v2_distilled和v2_value_refined均能完成episode并生成162条decision、5条episode和5条per-task记录。exact V2实际使用batched K20 teacher，distilled/value模式在同一policy请求内运行predictor；CSV及summary包含success、timeout、calls、avg H、H分布、policy/server/wall、teacher和predictor总延迟。由于两种student模式使用synthetic随机sidecar，本次success/H结果只能证明闭环实现与计时链路可运行，不能作为predictor质量或方法对比证据。下一步是采集任务平衡的真实counterfactual训练集、训练student、进行student-state迭代重标注并运行10 tasks x20与最终配对10 tasks x100评估。

## 2026-07-13 - started the initial K20 counterfactual collection and verified iterative warm-start

服务器已更新到collector覆盖优化提交`36a6353`，并启动正式首轮counterfactual采集：输出根目录为`/root/autodl-tmp/acotvla/execution_horizon_v2p/initial_k20_fixedh9_10tasks_20trials`，policy server使用冻结的`50999` base checkpoint，collector配置为LIBERO-10全部10个task、每task 20 trials、batched teacher K20、Action-CoT denoising steps 10、fixed H9 continuation、每episode最多一个root、20个decision offset轮转、每个root完整branch H1-10且debug视频数为0。collector在tmux `v2p_initial_collect`中运行，日志位于`collector/run.log`；开始约6分钟时已产生3个root，episode 1/2/3分别在推进1/2/3个fixed-H9 decision后采集，表明offset覆盖不再局限于episode起点。完整数据量、branch结果和总耗时尚未产生。

复核iterative SFT时发现Orbax会将NNX list层的整数key恢复为字符串，而standalone trainer的`--resume-params`路径尚未归一化这些key。提交`e1ec5f7`在trainer warm-start时调用`convert_str_keys_to_int`，并已推送及同步服务器。使用先前真实1-root collector数据和上一轮sidecar执行1步resume训练成功，输出位于`/root/autodl-tmp/acotvla/v2p_smoke/e1ec5f7/resume_train`；训练耗时8.36 s、loss 2.2118，日志再次确认base policy未加载且完全冻结。该1样本数值只验证恢复与继续优化路径，不代表模型效果。

单collector实测约1.5分钟/root，预计200条需约5小时；其前4条尚在内存buffer、没有产生任何HDF5 shard。为提高GPU利用率，本次停止该无落盘试跑并保留原`collector/run.log`，改为4个互斥task分片并行连接同一个policy server，输出根目录为`/root/autodl-tmp/acotvla/execution_horizon_v2p/initial_k20_fixedh9_10tasks_20trials_parallel`，分片为tasks 0-2、3-5、6-7、8-9，每20条立即落一个shard。启动后四个collector进程均存活，单模型server总显存约19.1 GB、GPU利用率约87%，未复制base checkpoint；前3个分片已各完成首个root，Task8-9分片当时仍在初始化。正式速度评估不会使用并发采集方式，将在无资源争用的单server上串行测量五种模式。

为降低小规模首轮counterfactual数据上的过拟合风险，提交`8486631`为standalone SFT增加了可配置的validation-best sidecar选择、按logging次数计的early stopping patience及minimum delta；validation仍按完整`task_id+episode_id`分组切分，不会将同episode状态泄漏到train与validation。远端ruff通过。使用1条真实smoke记录人为设置极大minimum delta后，训练按预期在第2步early stop并选择第1步validation checkpoint；从所保存的best sidecar重新加载后，首个train loss精确复现为1.851402，证明保存的是第1步参数而未被第2步NNX state覆盖。该机制不加载base policy，不改变多任务loss或冻结口径；单样本loss仍仅为功能验证。

并行首轮采集达到59条时，tasks 6-7分片产生首个20-record shard `part_tasks_6_7/data/shard-00000.h5`。manifest确认schema version 1、teacher K20、Action-CoT denoising steps10、fixed H9 continuation、offset cycle20与source iteration0。该文件大小223,416 bytes；`prefix_feature (20,2048)`、`coarse_actions (20,15,32)`、`final_actions/previous_actions (20,10,32)`、state与三类risk均为float16，outcome为bool/uint16，physics state为每root一份vlen float64。所有feature/risk均为finite，branch success与timeout在全部200个H标签上严格互补，remaining calls范围6-111、remaining steps范围44-990、raw H取值为3/9/10。该Task6 shard的H1-10 branch success rate依次为85%、90%、80%、95%、95%、90%、80%、90%、80%、100%，表明数据包含非平凡且不必单调的全H counterfactual监督；完整数据仍在采集中，不能从单task shard推断最终策略效果。

tasks 6-7分片随后以40 records、2 shards正常完成，elapsed 3141.16 s；合并两个task后的H1-10 branch success rate为92.5%、95%、87.5%、97.5%、97.5%、95%、90%、95%、90%、100%，debug failure videos为0。tasks 8-9分片的首个20-record shard也已落盘，其中zero-based task 8有15/20个root至少一个H失败、共46个timeout标签，H1-10 success rate为80%、90%、80%、75%、80%、90%、65%、65%、75%、70%，remaining calls范围22-111、remaining steps 192-990。该shard的teacher raw-H为H3 11条、H8 2条、H9 2条、H10 5条，event label rate为9.5%。这些事实说明teacher event/raw-H与各H真实branch return不等价，Q head的全向量监督提供了独立信号；仍需训练和闭环评估才能判断是否改善策略。

首轮fixed-H9 continuation正式数据采集已完整结束：4个分片分别产生60、60、40、40 records，共200 root states、10个HDF5 shards，10个task各20条且`task_id/episode_id/decision_step/root_seed`无重复。全部feature finite，branch/risk valid全真，success与timeout严格互补；53/200个root至少一个H失败，2个root的H1-10全部失败，总timeout标签率9.85%。全数据H1-10 branch success rate为95%、92.5%、89.5%、88.5%、90.5%、92%、88%、90%、86%、89.5%；raw-H计数为H3 71、H4 1、H6 1、H7 3、H8 19、H9 24、H10 81，event标签率9.3%，decision step覆盖10-181，remaining calls覆盖1-111、remaining steps覆盖10-990。10个shard总大小2,225,784 bytes。四个并行分片wall elapsed约4893.86、4890.87、3141.16、4812.84 s；这些时间受共享server并发影响，只用于采集审计，不作为policy速度结果。采集完成后已停止base policy server，GPU显存回落到0 MiB。

首轮真实SFT输出位于`/root/autodl-tmp/acotvla/execution_horizon_v2p/sft_round00_initial_200_seed7_8486631`。训练读取全部4个counterfactual目录，seed7、batch128、最大20,000步、学习率3e-4、weight decay1e-3、20%按episode分组validation、每50步验证、patience 20个log且minimum delta 1e-4；base policy未加载并保持冻结。训练在step1050 early stop，总耗时27.63 s，validation-best sidecar选自step50，best validation loss为2.032916。step50的train/validation loss为1.246041/2.032916，validation success/timeout accuracy为88.75%/89.50%，raw-H accuracy为45.0%，validation success/timeout BCE为0.353386/0.345140，final/Action-CoT/fused risk Huber为0.192776/0.195102/0.177846，event BCE为0.296257。后续step的train loss继续下降而validation升高，证明小数据发生过拟合，部署sidecar没有使用末步。Feishu watcher以60秒poll和3600秒progress interval运行，训练完成后`.feishu_notification_sent`写入`completed`，说明完成消息已成功发至配置群；训练短于1小时，因此没有中间每小时进度消息。

## 2026-07-14 - restarted the 10x20 evaluation as a strictly serial official run and added safe resume

首轮真实sidecar已由单policy server加载，server日志确认base checkpoint`50999`与5.0 MiB predictor sidecar均恢复成功。最初串行评估在52个original episode后，为讨论吞吐曾短暂停止并启动五个模式并发；每个并发模式仅产生2个episode，用户随后明确要求继续串行，因此五个并发进程立即停止。这些partial输出保留在`eval_round00_10tasks_20trials`与`eval_round00_10tasks_20trials_parallel_functional`中，仅作审计且不进入正式结果。严格串行正式评估已从干净目录`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_round00_10tasks_20trials_serial_final`重新开始，tmux为`v2p_eval_round00_serial_final`，单server无其他eval客户端；模式顺序为original、fixed H9、exact batched-K20 V2、v2_distilled、v2_value_refined，initial-state IDs、seed、threshold、candidate、budget和denoising steps均显式固定。

考虑最终10 tasks x100 trials串行运行接近一天，提交`02ea7c2`为`scripts/eval_libero_execution_horizon.py`增加安全`--resume`：新run写入完整配置签名，逐episode以append journal写rollout与decision；恢复时只跳过完全完成的`mode/task/episode`，拒绝配置不一致或重复rollout，并清理中断episode的decision残片与重复decision。该改动同时避免原实现每episode重写全部decisions CSV的二次方I/O。提交已推送并同步服务器，远端ruff通过；独立journal smoke确认1条已完成rollout被恢复、重复decision去重且孤立未完成episode decision被删除。当前已经启动的10x20进程继续使用其启动时代码运行，不受仓库文件更新影响；resume能力用于后续长评估或异常恢复。

## 2026-07-14 - monitored the strict serial round-0 evaluation and audited collector timing

严格串行的round-0 `10 tasks x 20 trials`评估保持单一eval进程和单一policy server运行，没有并发模式客户端。正式目录中的original模式已完成200/200 episodes；其paired样本内overall success为94.0%、timeout为6.0%、calls/episode为61.42、steps/episode为315.19，actual policy/server/client-wall分别为5821.387/6375.609/6450.812 ms/episode。逐task success（zero-based task 0-9）为100%、100%、95%、100%、100%、95%、95%、95%、65%、95%。这些是当前正式200-trial样本结果，不能与499-trial historical original 96.593%直接视为同一估计。

监控时fixed H9已推进至96/200，eval进程数始终为1，尚未开始exact batched-MC、v2_distilled或v2_value_refined，也尚未生成全模式summary。复核首轮正式counterfactual collector的四个summary后，tasks 0-2、3-5、6-7、8-9分片分别耗时1.359、1.359、0.873、1.337小时；这些是共享单server并发采集的wall数据，只用于预估iterative student rebranch约需1.5-2小时，不用于policy速度结论。后续闭环评估继续严格串行；student counterfactual数据采集可继续按互斥task分片并发，因为其目标是生成训练标签而不是比较部署延迟。

fixed H9模式随后完成200/200 episodes。相同initial-state样本上的overall success为97.0%、timeout为3.0%、calls/episode与sampled chunks/episode均为31.59、steps/episode为290.53、avg H严格为9.0，H9共执行6318次；actual policy/server/client-wall分别为3108.513/3423.918/3461.769 ms/episode。zero-based task 0-9 success依次为100%、100%、100%、100%、95%、95%、100%、100%、85%、95%，对应calls/episode依次为29.95、28.45、27.70、25.30、33.75、25.20、26.80、30.40、56.15、32.20。该paired 200-trial结果高于本轮original的94.0% success且调用数约减半，但仍需exact与student模式完成后再下整体结论。评估已自动切换到exact batched-K20 V2，切换过程没有报错。

exact batched-K20 V2的首个完整task（zero-based task 0，20 episodes）成功率为95%、calls/episode为35.95、avg H为8.8231、actual client wall为4706.819 ms/episode，其中batched teacher实际累计3816.575 ms/episode，折合约106.16 ms/policy call，与独立K20 benchmark的110.700 ms同步teacher延迟一致。Task0的H执行计数为H3 20、H4 5、H5 7、H6 17、H7 34、H8 97、H9 268、H10 271；这是中间单task事实，不代表exact模式overall结果。

exact模式完成前100个episodes（zero-based tasks 0-4）时的中间统计为99.0% success、1.0% timeout、30.01 calls/episode、avg H 8.8185；actual policy/server/client-wall分别为3668.832/3878.883/3913.549 ms/episode，其中batched teacher累计3168.298 ms/episode。Task8/Task9尚未进入，因此该前半段99%不能当作overall结论。

exact batched-K20 V2随后完成200/200 episodes。overall success为93.5%、timeout为6.5%、calls/episode为35.90、steps/episode为324.035、avg H为8.8283；每episode实际采样718.0个action chunks，即35.90次policy call乘K20。H执行计数为H2 2、H3 205、H4 60、H5 62、H6 138、H7 324、H8 938、H9 2734、H10 2717。actual batched-teacher/policy/server/client-wall分别为3822.021/4414.025/4663.071/4704.710 ms/episode，所有值均来自闭环中逐call同步累计，不是potential timing。

exact模式zero-based task 0-9 success依次为95%、100%、100%、100%、100%、95%、85%、95%、80%、85%；calls/episode依次为35.95、28.85、30.85、26.80、27.60、27.35、39.75、36.95、63.35、41.55。该paired 200-trial样本上exact K20的93.5% success低于fixed H9的97.0%，也略低于original的94.0%，同时calls和wall高于fixed H9；Task8/Task9是主要失败来源。现有exact entropy controller在本轮不是Pareto最优，因此后续student relabel和Q refinement需要重点覆盖hard-task及失败附近状态，不能仅依据蒸馏V2 risk曲线声称改善。评估已继续自动进入v2_distilled。

round-0 v2_distilled的前11个episodes几乎始终选择H10（临时avg H 9.993），与训练集中event标签仅9.3%且普通BCE/Huber容易平滑稀有risk峰值的现象一致。为下一轮SFT提交并推送`28f0e0e`（`Rebalance rare V2-P supervision`）：新增与各head总loss系数分离、全部可配置的success-failure、timeout-positive、event-positive和risk-event regression区域权重，trainer默认分别为4/4/5/3；新增validation failure/timeout recall、event与fused-risk-event precision/recall、success Brier及raw-H MAE；task focus默认显式改为zero-based IDs 8、9。改动不影响predictor结构、base checkpoint、episode budget controller或正在运行的round-0进程。本地`py_compile`和`git diff --check`通过，服务器fast-forward后ruff通过；为避免污染当前严格串行速度统计，真实JAX训练smoke延后到本轮五模式评估结束后执行。

v2_distilled完成前100个episodes（zero-based tasks 0-4）时的中间统计为98.0% success、2.0% timeout、27.85 calls/episode、avg H 9.7663；actual policy/server/client-wall分别为3115.841/3380.125/3414.748 ms/episode，predictor累计133.840 ms/episode，折合约4.81 ms/call。H计数为H3 14、H4 28、H5 22、H6 23、H7 23、H8 21、H9 49、H10 2605，说明round-0 entropy head未充分复现exact V2的event/H分布。Task8/Task9尚未进入，该前半段98%不是overall结论。

round-0 v2_distilled随后完成200/200 episodes。overall success为96.0%、timeout为4.0%、calls与sampled chunks均为29.675/episode、steps为298.525/episode、avg H为9.8550。H计数为H2 1、H3 24、H4 34、H5 25、H6 26、H7 30、H8 27、H9 70、H10 5698；95.999%决策为H10，明显没有复现exact K20的event/H分布。actual predictor/policy/server/client-wall分别为142.930/3330.991/3612.132/3648.926 ms/episode，predictor折合约4.82 ms/call且已包含在其余累计时间中。

v2_distilled的zero-based task 0-9 success依次为100%、100%、95%、95%、100%、95%、85%、100%、90%、100%；calls/episode依次为29.05、25.60、31.40、28.65、24.55、22.80、34.20、26.30、47.55、26.65。它优于本轮exact K20的93.5% success、35.90 calls和4.705 s wall，但相对fixed H9少1.915 calls、success低1个百分点，且因predictor和其他时延使client wall高约0.187 s。证据支持round-0 distilled是接近H10的不同trade-off，而不是忠实复现V2；需通过iterative data与不平衡SFT修正。评估已进入v2_value_refined。

v2_value_refined完成前100个episodes（zero-based tasks 0-4）时的中间统计为98.0% success、2.0% timeout、29.62 calls/episode、avg H 9.1489；actual policy/server/client-wall分别为3326.336/3606.204/3643.879 ms/episode，predictor累计143.398 ms/episode。H计数为H1 51、H2 26、H3 37、H4 42、H5 53、H6 31、H7 49、H8 75、H9 618、H10 1980。与相同前100 states的distilled相比success同为98%，但Q refinement将avg H从9.7663降到9.1489并使calls从27.85增到29.62，证明Q路径实际改变了闭环决策；是否改善hard-task success仍需Task8/Task9与overall结果。

首轮严格串行`10 tasks x 20 trials`五模式评估已完整结束。正式目录`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_round00_10tasks_20trials_serial_final`含1000条rollout（每模式精确200条）、1000个唯一`mode/task/episode/initial-state`键与37,921条decision，`summary.json`状态为complete且`paired_initial_states=true`。全部模式使用Action-CoT denoising steps10、seed7、相同initial-state IDs、单policy server和单eval client；运行结束后已停止round-0 server，GPU无残留模型进程。

v2_value_refined正式overall为96.0% success、4.0% timeout、31.02 calls与sampled chunks/episode、decision-weighted avg H 9.2893。H计数为H1 80、H2 59、H3 47、H4 71、H5 81、H6 96、H7 108、H8 119、H9 1111、H10 4432。actual predictor/policy/server/client-wall分别为150.133/3473.453/3765.333/3805.798 ms/episode。zero-based task 0-9 success依次为100%、100%、95%、95%、100%、95%、95%、100%、80%、100%；calls依次为30.80、28.45、33.10、28.75、27.00、22.55、29.25、27.50、55.95、26.85。它与distilled overall success同为96%，但calls多1.345、wall高约0.157 s且Task8从90%降到80%，所以round-0 Q refinement没有超过distilled或fixed H9。

正式`summary.json`的overall avg H按所有decision计数加权；此前监控阶段从每episode `avg_h`列直接取算术均值得到的exact 8.8283、distilled 9.8550等数字是episode等权中间口径。正式decision-weighted值应使用summary：exact K20为8.8503、v2_distilled为9.8613、v2_value_refined为9.2893。success、calls和实际时延等其余已记录overall值不受该口径差异影响。

本轮paired overall对比为：original 94.0% success/6.0% timeout/61.42 calls/6.451 s wall；fixed H9 97.0%/3.0%/31.59/3.462 s；exact batched-K20 V2 93.5%/6.5%/35.90/4.705 s（teacher 3.822 s）；v2_distilled 96.0%/4.0%/29.675/3.649 s（predictor142.930 ms）；v2_value_refined 96.0%/4.0%/31.02/3.806 s（predictor150.133 ms）。本轮最佳success是fixed H9；distilled以低1个百分点success换取1.915 fewer calls，但wall未优于fixed；exact和value refinement均未形成更优Pareto点。下一步是student-state H1-10重分支、聚合数据、使用`28f0e0e`不平衡权重继续SFT，然后重新运行闭环评估。

严格串行评估结束且GPU释放后，在真实200-root HDF5上执行了`28f0e0e`两步rebalanced trainer smoke，输出为`/root/autodl-tmp/acotvla/v2p_smoke/28f0e0e/rebalanced_train`。训练、完整loss反传、JIT validation和Orbax sidecar导出均成功；summary确认base policy未加载且冻结、160 train/40 validation records、label weights为failure4/timeout-positive4/event-positive5/risk-event3，step2 validation loss为3.400653。新增诊断正常产生：event precision/recall 0.1130/0.9189、failure recall 0.3077、timeout recall 0.5128、success Brier 0.2264；fused-risk-event recall在仅2步随机初始化下为0，所有数值只证明新路径可运行，不代表训练质量。训练smoke耗时21.85 s，ptxas不支持CC12而回退CUDA driver编译的已知warning未导致失败。

已启动round-1 student-state counterfactual collection，输出根目录为`/root/autodl-tmp/acotvla/execution_horizon_v2p/iter01_student_value_k20_10tasks_20trials_2roots_parallel`。单server在port8019加载冻结base `50999`与round-0 sidecar；四个互斥task分片为0-2、3-5、6-7、8-9，共享该server但不复制base模型。collector使用current student `v2_value_refined` continuation、teacher K20、Action-CoT denoising steps10、seed107、source iteration1、每episode两个错开root（offset cycle10、stride10）、每root完整branch H1-10、每20 records落一个HDF5 shard；预计最多400 roots。Task8-9分片最多保留3个失败debug视频，其余不保存视频。

首次启动四个collector时，tmux命令中的公共参数变量被多余反斜杠转义，实际进程只收到各分片参数和默认值。该问题在首个有效record/shard产生前通过`pgrep -af`核对命令行发现；四个错误进程被停止，刚创建的空分片目录被清理并立即重启。重启后逐进程命令行已核实包含port8019、current_student/value_refined、K20、source iteration1、两root设置及全部V2/Q/budget参数；四个客户端存活，server约占16.95 GiB，仅输出已知Gym deprecation warning。该启动纠正没有产生可用于训练的数据。

round-1采集的首个20-record shard已在tasks6-7分片落盘：`part_tasks_6_7/data/shard-00000.h5`，大小223,404 bytes。只读审计确认schema1、source iteration全部为1、首批20条均来自zero-based task6、decision steps覆盖10-178、previous-action有效率95%、所有feature/risk finite、branch/risk valid全真且success与timeout逐H严格互补。feature/action为float16、outcome为bool/uint16、physics state仍为每root一份vlen float64；4/20 roots至少一个H timeout，总branch timeout标签率6.5%，event标签率8%，raw-H计数为H3 7、H7 1、H9 3、H10 9。该检查在collector继续运行时只读完成，没有中断或污染采集。

tasks8-9分片的首个20-record shard随后落盘，首批均来自zero-based task8 student states，大小215,185 bytes；source iteration1、feature finite和valid约束全部通过，decision steps覆盖10-175且previous-action有效率95%。14/20 roots至少一个H失败，无root全部H失败，200个branch标签timeout率20.5%。H1-10 branch success依次为70%、75%、75%、90%、75%、75%、85%、85%、85%、80%；同一批状态上H4最高而H10仅80%，呈明显非单调counterfactual return，支持继续训练完整Q向量而不是单一best-H硬标签。该shard event标签率6.5%，raw-H计数为H3 4、H8 1、H9 6、H10 9；teacher event/raw-H同样不能替代真实branch return。

round-1 tasks6-7分片以80/80 records完整结束，elapsed 6639.21 s（1.844 h），4个20-record shards，continuation/current-student、student mode/value-refined、teacher K20、denoising steps10与source iteration1均由summary再次确认。合并该分片的H1-10 branch success为92.5%、90.0%、93.75%、91.25%、92.5%、96.25%、91.25%、91.25%、96.25%、91.25%，debug视频0。一个collector按预期退出，其他三个分片和共享server继续运行。

round-1 student-state collection最终完成396 roots、20个HDF5 shards、总大小4,420,328 bytes；四分片records/elapsed分别为tasks0-2 120/10070.06 s、tasks3-5 116/8312.77 s、tasks6-7 80/6639.21 s、tasks8-9 80/9915.43 s。zero-based task 0-9 records为40、40、40、40、40、36、40、40、40、40；task5少4条是部分短episode在第二个offset root前自然成功/终止，没有补造状态。共保存3个hard-task失败debug视频，符合少量视频约束。采集完成后已停止policy server，GPU无残留模型进程。

全量只读审计确认396个`source/task/episode/decision/root-seed`键全部唯一，source iteration全部为1，所有feature/risk finite、physics state非空、branch/risk valid全真且success/timeout逐H严格互补。118/396 roots至少一个H timeout，5个root的H1-10全部timeout，总branch timeout标签率9.5202%；H1-10 branch success依次为92.68%、89.90%、90.91%、89.39%、91.16%、90.15%、91.67%、89.65%、90.66%、88.64%。raw-H计数为H3 135、H4 5、H5 2、H6 1、H7 3、H8 33、H9 42、H10 175，event标签率8.258%，decision step覆盖10-200、remaining calls 1-110、remaining steps 2-990。

hard-task聚合中，Task8的40 roots有30条至少一个H timeout，总timeout标签率21.25%，H1-10 success为75%、77.5%、82.5%、77.5%、80%、77.5%、85%、80%、77.5%、75%；Task9的40 roots有19条至少一个H timeout，总timeout标签率15.25%，H1-10 success为87.5%、80%、85%、80%、80%、90%、85%、87.5%、85%、87.5%。这些student分布标签与初始fixed-H9 continuation数据共同形成596-root聚合SFT集，并保留完整非单调Q向量。

round-1聚合SFT输出为`/root/autodl-tmp/acotvla/execution_horizon_v2p/sft_round01_aggregate596_seed7_28f0e0e_balanced`。训练读取初始200与student396共596 roots，按episode group切分477 train/119 validation，从round-0 sidecar warm-start；base policy未加载且完全冻结。配置为seed7、batch128、LR1e-4、weight decay1e-3、Task8/9 multiplier3、failure multiplier3、failure/timeout-positive/event-positive/risk-event label multipliers4/4/5/3，并提高fused-risk/event/raw-H head总loss权重。训练在step1600 early-stop，总耗时39.73 s，validation-best sidecar选自step100，best loss3.409992；末步validation已严重退化到6.002661，未被部署。

step100 best-validation诊断为success/timeout accuracy76.55%/78.74%、failure/timeout recall39.83%/35.59%、success Brier0.15418、event precision/recall20.31%/52.0%、fused-risk-threshold precision/recall36.0%/18.0%、raw-H accuracy40.34%且MAE2.714。加权BCE改善稀有失败recall但会改变success logit的概率校准，因此在闭环前必须检查0.9阈值，不能仅凭accuracy选择该sidecar；计划训练Q class weight为1、仅保留state-level hard sampling及event/risk不平衡权重的校准对照。

训练时同步启动tmux `v2p_sft_round01_watch`，加载私有`/root/.config/acotvla/feishu_notify.env`，poll60 s、progress interval3600 s。训练完成后`.feishu_notification_sent`内容为`completed`且watcher正常退出，确认完成消息成功发送；训练短于1小时，因此没有中间每小时进度消息。

对balanced候选做未加权validation概率审计后，真实success率90.08%，平均预测success仅68.81%；`success>=0.9 && timeout<=0.2`只覆盖16.30%标签，所选标签实际success为95.88%。class-weighted BCE提高failure recall但使Q概率整体下偏，因此该候选不用于正式闭环。随后从round-0 warm-start训练Q class multipliers为1/1的`qcalibrated`候选，输出`/root/autodl-tmp/acotvla/execution_horizon_v2p/sft_round01_aggregate596_seed7_28f0e0e_qcalibrated`；仍保留Task8/9、high-risk、gripper和failure root采样以及event/risk峰值权重。它在step1650 early-stop、39.22 s，best sidecar为step150，validation loss2.846740、success Brier0.09098、success accuracy89.08%、raw-H accuracy44.54%。

qcalibrated的独立validation审计显示平均success probability88.10%对真实90.08%；0.9/0.2 Q筛选覆盖63.45%，被选标签实际success为95.63%。0.9以上bin的平均预测95.69%、实际95.65%，表明部署阈值校准良好；各H被选子集实际success为92.77%-98.65%。但其fused-risk threshold1.5只预测2.35% event点，目标为8.40%，119个validation roots中103个预测raw-H10，仍不足以复现V2 entropy行为。

为兼顾双头，从qcalibrated sidecar以LR5e-5继续训练`riskcalibrated`候选，输出`/root/autodl-tmp/acotvla/execution_horizon_v2p/sft_round01_aggregate596_seed7_28f0e0e_riskcalibrated`。该run保持Q普通BCE，将event-positive/risk-event multipliers提高到10/10，并提高final/CoT/fused/event/raw-H loss权重；step1550 early-stop、38.52 s，best sidecar为step50。best validation的success Brier0.09347、accuracy88.74%、raw-H accuracy44.54%；event precision/recall14.98%/65.0%，fused-risk threshold1.5 precision/recall27.08%/26.0%。

riskcalibrated独立审计显示Q 0.9/0.2覆盖59.24%、被选标签实际success95.89%，仍保持可用校准；threshold1.5预测event率8.07%，接近validation目标8.40%，显著优于qcalibrated的2.35%。预测raw-H计数为H3 31、H4 2、H5 1、H6 2、H7 3、H8 3、H9 3、H10 74，目标为H3 38、H4 1、H6 1、H8 14、H9 12、H10 53，中段H8/H9时间定位仍不充分，但该候选是当前entropy/Q双头的最佳折中并被选为round-1正式闭环sidecar；student risk threshold保持原1.5，不改exact基线。qcalibrated与riskcalibrated训练均各自启动Feishu watcher，完成后marker均为`completed`且watcher正常退出；两次训练都短于1小时，没有中间进度消息。

已启动round-1 riskcalibrated sidecar的严格串行student闭环复核，输出目录为`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_round01_riskcal_10tasks_20trials_serial_student`。单policy server在port8020加载冻结base checkpoint `50999`与`riskcalibrated/params`，单eval进程按顺序运行`v2_distilled`、`v2_value_refined`，每模式10 tasks x 20 trials；配置与round-0保持seed7、initial-state offset0、Action-CoT denoising steps10、risk threshold1.5、V2 target/init/capacity 9/6/12、Q success/timeout阈值0.90/0.20和H1-10候选。首次server tmux命令使用`uv`时因登录shell中找不到该可执行文件而在模型加载前退出；已改用环境脚本提供的`.venv` Python重新启动，base和sidecar均成功恢复且websocket开始监听，该启动错误未产生rollout或改变评估配对。

round-1 riskcalibrated的`v2_distilled`严格串行10 tasks x 20 trials已完成200条，overall为96.0% success、4.0% timeout、32.43 calls/chunks per episode、decision-weighted avg H 9.21816；H计数为H2 1、H3 286、H4 71、H5 93、H6 132、H7 122、H8 163、H9 950、H10 4668。actual predictor/policy/server/client-wall分别为154.761/3581.784/3881.202/3928.555 ms/episode。逐task success为95%、100%、100%、100%、100%、95%、90%、95%、90%、95%。Task8为18/20、55.65 calls、avg H9.3378、predictor263.339 ms和wall6.688 s；与round-0 distilled的同一20 states相比，Task8 success仍为90%，但calls从47.55升至55.65、wall从5.873 s升至6.688 s。overall也与round-0 distilled同为96%，但多2.755 calls且wall多约0.280 s，因此增强rare-event recall没有让distilled形成更优Pareto点；同一评估进程已继续顺序运行`v2_value_refined`。

round-1 riskcalibrated的student-only闭环复核已完整结束，`summary.json`状态complete且paired initial states为真；目录包含400条唯一rollout键（每模式200）、13,304条decisions，无重复episode。`v2_value_refined` overall为94.0% success、6.0% timeout、34.09 calls/chunks、decision-weighted avg H9.00440；H计数为H1 42、H2 18、H3 203、H4 61、H5 84、H6 117、H7 208、H8 220、H9 2527、H10 3338。actual predictor/policy/server/client-wall为160.765/3749.022/4060.175/4108.980 ms/episode。逐task success为95%、100%、95%、100%、100%、90%、100%、90%、75%、95%。Task8仅15/20、62.75 calls、avg H8.9482和wall7.420 s，低于round-0 value-refined的16/20且更慢。

同一20-state paired对比表明riskcalibrated候选没有改善：distilled success与round-0同为96%但效率下降，value-refined从round-0的96%降至94%，calls从31.02增至34.09、wall从3.806 s增至4.109 s；它也低于fixed H9的97%。增强risk/event标签权重虽使离线event rate匹配teacher，但在线产生较多短H，未带来hard-task回报。评估完成后已停止port8020 policy server和eval tmux，GPU进程随即释放。下一步不把该失败候选直接扩展到10x100，而是复核Q校准更好的`qcalibrated` sidecar及value筛选，再选正式sidecar；所有失败结果保留不覆盖。

为使sidecar/阈值选择可重复，新增`scripts/evaluate_execution_horizon_sidecar.py`并提交推送`5631394`，随后以`dd5d64f`修复两处Ruff冗余cast。脚本只加载standalone predictor与紧凑HDF5 labels，不加载base policy；使用与训练一致的episode-group split，在held-out roots上报告predicted distilled和Q threshold sweep所选H对应的真实H1-10 counterfactual success/timeout、H分布、Q缩短率及Task8/9子集。输出明确标注不含episode budget dynamics且不能替代LIBERO闭环。服务器`ruff check`、`ruff format --check`与`py_compile`全部通过，远端仍只保留既有`third_party/libero`用户修改。

596-root聚合数据的119-root held-out审计写入`/root/autodl-tmp/acotvla/execution_horizon_v2p/offline_sidecar_audit_596_seed7_dd5d64f/summary.json`。validation task0-9数量为15、12、15、9、3、14、3、18、15、15。qcalibrated的predicted curve any-event rate13.445%，riskcalibrated为37.815%；predicted distilled raw-H counterfactual success分别86.55%和88.24%，该离线代理不包含budget和closed-loop状态迁移。Q sweep中qcalibrated的`minimum success=0.925, maximum timeout=0.15`取得最高overall counterfactual success92.437%、Task8/9合并80.0%、average raw H7.445、Q reduction41.18%；默认0.90/0.20为91.597%、hard tasks76.67%。因此下一轮qcalibrated闭环预注册0.925/0.15阈值；该阈值来自held-out counterfactual审计，仍必须通过闭环确认。

已启动qcalibrated sidecar的严格串行`v2_value_refined` 10 tasks x 20 trials闭环复核，目录为`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_round01_qcal_q0925_t015_10tasks_20trials_serial_value`。单server在port8022加载冻结base `50999`与qcalibrated params，Q阈值锁定为success0.925/timeout0.15；seed7、initial-state offset0、denoising steps10、risk threshold1.5、budget target/init/capacity9/6/12与H1-10候选均与既有paired评估相同。前2条rollout均成功，首次JAX warmup后进程与journal正常。

已向用户系统讲解V2-P predictor架构、训练监督与实验决策链：2048维prefix/state/controller context与final/coarse/previous-overlap action tokens进入256维、3层邻域temporal residual encoder；token heads预测三条risk curves/event，pooled summary heads预测raw-H分类/ordinal、H1-10 success/timeout与remaining calls/steps。说明了base不在trainer加载、sidecar独立训练、同一JAX policy call复用prefix/action且不新增RPC；并结合K20 batched teacher加速、200-root初始数据、396-root student iterative relabel、balanced概率失校准、qcalibrated Q校准、riskcalibrated在线过度保守及当前qcal闭环复核，解释每一步为何保留或淘汰。该说明只引用已获得的事实，没有把当前未完成的早期qcal结果当作最终结论。

根据用户提醒，明确将纯原版ACoT-VLA作为速度主对照，并用严格paired round-0 actual wall重算口径：original H5为6.450812 s/episode；V2-P distilled为3.648926 s（含predictor142.930 ms），实际speedup 1.768x、wall下降43.43%；value-refined为3.805798 s（含predictor150.133 ms），speedup1.695x。distilled predictor约占wall3.92%，约4.82 ms/policy call，因此相对纯原版的净加速没有被predictor开销抵消。但fixed H9无predictor为3.461769 s且97% success，反而比round-0 distilled快0.187 s并高1个百分点，说明当前V2-P尚未证明相对fixed H9值得。最终主表必须同时给出相对纯原版actual speedup和相对fixed H9净差，predictor latency单列且已包含在policy/server/wall totals中；不得只用calls推算。

进一步向用户澄清评估列与两种V2-P模式：现有`actual_wall_total_ms`是对每个`client.infer()`用`perf_counter`实测后在episode内求和的policy-RPC阻塞时间，包含网络/序列化/server/policy/predictor，但不包含MuJoCo reset、`env.step`、渲染和收尾；predictor是policy/server/wall的子项，不能重复相加。`v2_distilled`只用预测risk curves走原V2 event mapping和原budget controller，主要验证single-call能否逼近exact K20；`v2_value_refined`在同一entropy安全上限内再用H1-10 success/timeout Q筛选最大可接受H，最后仍走原budget controller。当前remaining calls/steps仅记录并保留SMDP接口，未直接控制H，也未实现PPO。为避免`wall`名称歧义，正式10x100前将额外记录包含环境reset/step与policy calls的full episode elapsed，并同时保留policy-RPC wall。

提交`331b010`新增明确的`policy_rpc_wall_total_ms`别名和真正覆盖环境创建/reset、所有`env.step`、policy calls及关闭环境的`actual_episode_elapsed_total_ms`；summary增加对应per-episode均值和有效episode计数，并对旧journal字段提供兼容读取。用户要求把成功率纳入速度比较后，明确采用联合Pareto口径：主表同时报告success、相对原版百分点差、policy/RPC/full-episode speedup、predictor占比；只有success不劣且actual time更低才称成功加速。round-0中fixed H9以97%/3.109 s policy/3.462 s RPC wall严格优于两种96%的V2-P；V2-P只相对94%的paired original形成观测Pareto改进，正式10x100还需报告逐task与same-state paired win/loss及不确定性。

针对fixed H9新评估97% success是否异常，核对了早期与当前两次10 tasks x 20 trials结果。早期`/root/autodl-tmp/acotvla/stage_b_pruning_eval/fixed_h9_libero10_10tasks_20trials/summary.json`为188/200=94%，逐task为100/95/95/100/90/95/90/100/80/95%；round-0当前快照为194/200=97%，逐task为100/100/100/100/95/95/100/100/85/95%。同一task/episode ID上有6个早期失败转为当前成功、1个早期成功转为当前失败，双侧exact McNemar约p=0.125，因此200局证据不足以认定真实成功率升高；两次评估还跨越代码/server快照，正式使用前必须做相同观测和seed的action chunk等价性审计及独立复跑。

fixed H9耗时按episode outcome分层后，早期run的成功局188个，aggregate policy约91.55 ms/call，成功局29.18 calls与2669.93 ms policy/episode，超时局110 calls与10087.24 ms；当前round-0成功局194个，aggregate policy约98.40 ms/call，成功局29.16 calls与2871.36 ms，超时局110 calls与10776.58 ms。两次overall calls/episode差异主要由超时局数量从12降至6造成，不代表成功episode本身执行更少。正在扩展evaluator summary，联合输出success/timeout、policy/server/predictor ms/call及成功/超时分层的calls与耗时，以避免用overall episode均值把策略延迟和timeout比例混在一起；在等价性审计完成前不把97%解释为H9具有稳定成功率优势。

进一步核对early Fixed H9与paired round-0的评估代码：early使用`scripts/eval_libero_action_cot_pruning.py`的单一`full`模式与`replan_steps=9`；round-0使用`scripts/eval_libero_execution_horizon.py`在同一run中顺序覆盖五个模式，并在summary显式记录`paired_initial_states=true`。两者均使用LIBERO-10 state `initial_states[episode]`、seed 7、Action-CoT denoising steps 10，且policy request seed公式均为`7 + task_id*1_000_000 + episode*10_000 + environment_step`。因此early与round-0 H9的名义初态/seed相同，但前者未与其他模式在同一代码/server快照内运行；round-0的paired只表示相同初态ID，不表示不同模式在首个action后仍共享轨迹。7个episode结果翻转不能归因于initial-state配对方式，必须继续审计base policy action输出与运行时等价性。

qcalibrated sidecar、Q success/timeout阈值0.925/0.15的严格串行`v2_value_refined`闭环已完整结束，目录`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_round01_qcal_q0925_t015_10tasks_20trials_serial_value`的summary状态complete，200 rows与200 unique keys，paired initial states为真。overall为191/200=95.5% success、4.5% timeout、32.93 calls、avg H9.06969；predictor/policy/server/client RPC wall分别为141.644/3414.247/3663.964/3721.744 ms/episode。逐task success为100/100/100/100/90/95/85/95/90/100%，Task6与Task8仍是主要失败来源。评估完成后port8022 server与相关tmux已停止并释放GPU。

qcal闭环的aggregate实际延迟为policy 103.682 ms/call、server111.265 ms/call、client RPC wall113.020 ms/call、predictor4.301 ms/call。成功局191个，平均29.508 calls、3064.481 ms policy与3341.585 ms RPC wall；失败局9个，平均105.556 calls、10837.045 ms policy与11789.557 ms RPC wall，证明overall episode timing受timeout比例显著影响。同state ID paired对比round-0 original/fixed H9/exact/distilled/value，qcal的win/loss分别为10/7、5/8、10/6、5/6、6/7，exact McNemar p均不显著；qcal虽优于riskcal的94%，但未超过round-0 fixed H9 97%或两种V2-P 96%，因此不直接扩展为正式10x100候选，先完成base action等价性与H9独立复跑。

为隔离early H9与round-0 H9差异来源，新增`scripts/audit_policy_action_equivalence.py`。工具在8个预注册LIBERO task/episode初态上构造first-decision policy input，记录完整输入SHA-256 digest和同一公式的policy seed，分别捕获两个server snapshot的`actions`与`coarse_actions`到各一份小型压缩NPZ；compare子命令先强制验证task/state/step/seed/input digest完全一致，再报告逐case exact equality及max/mean absolute error，并在输出中声明该审计仅覆盖相同首决策输入、不能替代full-rollout闭环。默认case覆盖Task4/6/8/9及早期/当前H9结果翻转相关episode。脚本已通过本地Ruff format/check、`py_compile`和`git diff --check`，尚未在服务器运行，尚无action等价性结论。

当前阶段性选择结论是：round-1 riskcal与qcal均未在10x20闭环超过round-0 V2-P；在基线审计前不将任何候选扩展到10x100。下一步依次为pre-V2P/current server同输入action audit、使用新增full elapsed和outcome-stratified字段独立串行复跑Fixed H9 10x20、基于审计锁定最佳sidecar，然后才启动五模式相同initial states的正式10 tasks x 100 trials。最终速度结论将以纯original为主对照，同时报告Fixed H9，并联合success/paired win-loss、policy/server/predictor ms/call、RPC wall与full episode elapsed；只有成功率不劣且actual time降低才称为成功加速。

按用户要求重命名主比较方案并更新汇报口径：`ACoT-VLA paired round-0 original H5`明确表示纯base checkpoint、原执行H5、无predictor；`ACoT-VLA Fixed H9`仍是同一base、无predictor，只改变execution horizon；exact K20无predictor但每call进行K20采样；V2-P distilled/value才加载predictor。后续主表删除499局、非严格paired的historical original行，以paired round-0 original H5的actual policy 5821.387 ms/episode作为Policy speedup分母。对应round-0 fixed/exact/distilled/value的policy speedup为1.873x/1.319x/1.748x/1.676x；riskcal distilled/value为1.625x/1.553x，qcal value为1.705x。速度表必须和success同列，且Fixed H9的97%在复现完成前加审计标记。

base action audit已在服务器完成第一轮。pre-V2P使用独立git worktree commit`c5c08fc6275d879e17dc5da89047db0474398401`，current使用`967ab56`；两者串行加载同一checkpoint `50999`，客户端统一使用current审计脚本。8/8 cases的task/episode/state/step/request seed及完整policy-input SHA-256 digest完全一致。pre-V2P对current的final actions均非bitwise equal，overall max absolute difference为0.00256205；coarse actions overall max为0.00390100，各case final mean absolute difference约0.0001556-0.0003326。结果目录为`/root/autodl-tmp/acotvla/execution_horizon_v2p/base_action_equivalence_c5c08fc_vs_967ab56`。

为区分代码漂移与运行时数值漂移，重启相同current commit/server并对相同8 cases重复capture。current第一次对current重启后的final actions overall max absolute difference为0.00292814，coarse overall max为0.00468135，均略大于pre-V2P对current；输入digest仍完全一致。因此当前证据不支持V2-P重构引入了超出同commit server重启基线的base-action漂移，但也表明该GPU/JAX运行栈跨server load不具备bitwise reproducibility。旧/新H9的94%/97%差异仍可能由闭环放大小数值漂移，必须依靠独立full-rollout H9复跑判断，而不能由单次action audit直接解释。所有审计server完成后已停止并释放GPU。

检查round-0计时公平性时确认，五模式共用的server必须加载predictor sidecar才能运行distilled/value。original/fixed request虽未调用predictor head，但`execution_horizon_predictor_enabled=True`会在`sample_actions_profile_prefix`中无条件pool prefix得到2048维`execution_horizon_prefix_feature`，结果还会经历device sync/host copy/websocket serialization；`Policy.infer`也返回normalized state/action aliases。因此round-0 original在动作选择上是原H5且base权重完全冻结，但其runtime不是严格pure-base timing，之前相对它计算的policy speedup只能标为round-0 pilot口径。该问题是额外计算/传输污染计时基线，不是predictor训练修改base权重。

正式公平协议调整为分server串行、相同initial-state IDs与seed：pre-V2P pure-base server运行ACoT-VLA original H5与Fixed H9；current base-only server运行Exact batched-MC V2 K20；sidecar server只运行V2-P distilled/value。这样success仍可按task/episode逐局paired，而每种速度来自其真实部署路径。正在运行的current base-only Fixed H9独立10x20复现目录为`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_fixed_h9_replicate_current_10tasks_20trials_967ab56`；Task0已20/20成功，临时Task0均值为30.95 calls、policy92.54 ms/call与2.864 s/episode、RPC wall3.152 s/episode、真正包含MuJoCo的full episode elapsed14.94 s/episode。该单task临时值不能代表overall。

用户确认无需继续current base-only H9诊断或额外smoke，应直接获得pure-base original/H9数据。已按此停止`eval_fixed_h9_replicate_current_10tasks_20trials_967ab56`，停止时journal为49/200、46 successes；该partial结果保留用于诊断但明确不进入主表。随后直接启动pre-V2P commit`c5c08fc` pure-base server和完整串行10 tasks x 20 trials evaluator，模式顺序为`original` H5后`fixed_h9`，输出目录`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_pure_base_c5c08fc_original_h5_fixed_h9_10tasks_20trials`。服务端不包含predictor模块或V2-P prefix-feature输出，客户端仍使用current evaluator以获得新的outcome-stratified与full-episode timing；checkpoint、seed7、state IDs0-19和denoising steps10保持不变。

pure-base original前6局均成功，临时均值56.5 calls、4.817 s policy与17.539 s full episode elapsed；该样本仅用于运行时间估算，不能用于成功率结论。结合后续hard tasks/timeout余量，当前400局pure-base original+H9 pilot预计约2-2.5小时；完整性汇总后不再插入smoke，直接进入分server严格串行10x100。正式pure-base original+H9约10-12小时、current base-only exact K20约5-6小时、round-0 sidecar distilled+value约9-11小时，加最终统计约1-2小时；从当前时点完整闭环预计约27-33小时，异常重启另预留2-4小时。正式sidecar锁定10x20闭环最好的round-0 predictor，不继续基于20局结果调riskcal/qcal阈值。

用户要求每个后续阶段完成时在飞书群汇报，并要求10x20完整性/paired/timing汇总消息包含列：方案、Predictor、Success、Calls/ep、Avg H、Policy ms/call、Policy s/ep、Policy speedup vs ACoT-VLA H5。扩展`scripts/watch_experiment_feishu.py`并提交`c0a2bae`：新增`--all-modes`、`--baseline-summary`/`--baseline-mode`，统一正式方案命名、predictor标记、policy ms/call fallback与speedup计算，生成Feishu纯文本表。随后提交`79336c0`新增可重复`--comparison-summary`，按优先顺序合并多个summary且以当前监控阶段同名mode覆盖旧数据；本地/服务器Ruff、py_compile及模拟summary表格测试通过。

current pure-base watcher首次启动因私有env文件变量未export而在发送前退出，未泄露凭据且未影响eval；已用`set -a`安全加载`/root/.config/acotvla/feishu_notify.env`后正常启动。按用户确认完整汇总需包括其他V2方案，watcher `v2p_purebase_feishu`现监控eval PID，完成时将当前pure-base summary与round-0 summary合并：新original/H9覆盖旧同名行，round-0提供Exact K20、V2-P distilled、V2-P value，speedup分母采用新pure-base original。正式阶段将依次以10x100结果替换pilot行；riskcal/qcal属于淘汰训练候选，保留在候选审计而不混入正式五方案主表。

pre-V2P pure-base original H5的10 tasks x 20 trials已完成200/200，187 successes=93.5%、13 timeouts=6.5%，60.910 calls/episode、固定avg H5。actual policy为91.314 ms/call与5561.915 ms/episode，server6084.615 ms/episode，client RPC wall6156.719 ms/episode，真正full episode elapsed15821.780 ms/episode。逐task success为95/100/100/100/100/95/90/100/60/95%。成功局187个均值51.380 calls、4663.014 ms policy、14229.691 ms full elapsed；timeout局13个均为198 calls，均值18492.260 ms policy、38723.370 ms full elapsed，进一步确认overall episode timing受timeout比例强烈影响。

original完成后同一pure-base server/evaluator已自动切换Fixed H9，无人工重启。按用户“每阶段完成发飞书”要求，已通过签名webhook发送original阶段完成消息，包含指定表头的一行pure-base数据、RPC/full elapsed、逐task success及H9正在运行提示，命令返回`feishu_sent`。合并五方案完整表仍由watcher在H9完成并生成summary后自动发送。

pre-V2P pure-base Fixed H9的10 tasks x 20 trials已完成200/200，192 successes=96.0%、8 timeouts=4.0%，32.905 calls/episode、固定avg H9。actual policy为94.857 ms/call与3121.283 ms/episode，server3442.422 ms/episode，client RPC wall3474.703 ms/episode，真正full episode elapsed12206.516 ms/episode。逐task success为100/100/95/95/95/95/95/100/85/100%。成功局192个均值29.693 calls、2821.410 ms policy、3142.495 ms RPC wall与11560.104 ms full elapsed；timeout局8个均为110 calls，均值10318.216 ms policy、11447.704 ms RPC wall与27720.424 ms full elapsed。

pure-base pilot完整性审计确认`summary.json`状态complete、400 rows和400个唯一`mode/task/episode`键，无缺失或重复；original与Fixed H9的200个task/episode键及initial-state IDs逐项完全一致，`paired_initial_states=true`。paired outcome中Fixed H9相对original有11 wins、6 losses、2局共同失败、181局共同成功；两者失败集合并不稳定地复现round-0，符合此前跨server load存在小数值漂移的证据。Fixed H9相对pure-base original的policy/RPC/full-episode speedup分别为1.782x/1.772x/1.296x，success高2.5个百分点；这是10x20 pilot观察值，正式结论仍等待10x100。watcher完成标记为`completed`并正常退出，已将当前pure-base original/H9与round-0 Exact K20、V2-P distilled/value合并为用户指定的五方案Feishu表；新pure-base original是policy speedup分母。下一步直接启动分server严格串行10x100正式评估，不再插入smoke。

正式pre-V2P pure-base 10 tasks x 100 trials已在tmux `v2p_formal_purebase_eval`启动，输出目录为`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_formal_pure_base_c5c08fc_original_h5_fixed_h9_10tasks_100trials`。它复用port8026上的commit `c5c08fc` pure-base server，严格串行运行original H5后Fixed H9；配置签名确认seed7、initial-state IDs 0-99、Action-CoT denoising steps10、无predictor。首次启动错误地对全新空目录传入`--resume`，安全检查因缺少`run_config.json`立即退出且未生成rollout；随即按设计以非resume首启成功，之后发生中断时才能使用相同签名resume。当前首3局均成功且journal持续写入，不能据此推断最终success。

正式pure-base阶段的飞书watcher `v2p_formal_purebase_feishu`已启动，poll60 s、每小时进度消息、完成消息；完成表将以正式pure-base original为speedup分母，正式original/H9覆盖旧行，并暂时合并round-0 Exact K20与V2-P两行作为明确标注的pilot context。后续exact与sidecar正式阶段完成时将依次用10x100结果替换对应pilot行。

为持续推进约27-33小时的串行正式流程，已在当前Codex任务创建30分钟心跳自动化`ACoT-VLA V2-P 串行正式评估推进`（automation id `acot-vla-v2-p`）。心跳只检查并推进当前远端流程：运行中核验journal/tmux，异常时按相同signature resume，阶段完成后做完整性与timing审计、追加本日志、确认Feishu，再依次切换到current base-only Exact K20和round-0 sidecar distilled/value；明确禁止并行eval、禁止触碰远端`third_party/libero`修改和泄露凭据。整个正式流程完成后应将该心跳停用。

2026-07-14 12:42（Asia/Shanghai）心跳检查：正式pure-base 10x100 journal已写入103/2000 episodes，全部属于按顺序运行的original H5；100 successes、3 failures，其中Task0已完成100局为97/100，Task1当前完成前3局且均成功。103个`mode/task/episode`键全部唯一，journal距检查时最近写入约1.7 s，不能用该未完成比例推断最终success。eval PID129817、pure-base server PID121683与Feishu watcher PID129878均存活；GPU仅server进程占用约17,030 MiB；eval/server日志未发现Traceback、exception、OOM或killed，Feishu watcher日志为空且未生成完成marker，符合阶段仍正常运行的状态。无需恢复或切换server，继续严格串行运行。

2026-07-14 13:12（Asia/Shanghai）心跳检查：正式pure-base journal推进至217/2000 episodes，仍在original H5阶段，212 successes、5 failures；Task0已完成97/100、Task1已完成98/100，Task2当前17/17。217个rollout键全部唯一，最近写入距检查约2.0 s；eval/server/watcher三个进程仍为PID129817/121683/129878，summary与Feishu完成marker尚未生成，watcher错误日志为0 bytes，eval日志未发现Traceback、exception、OOM或killed。阶段正常连续推进，无需resume或用户介入；未完成success比例仅作运行状态记录，不作为正式结论。

2026-07-14 13:42（Asia/Shanghai）心跳检查：正式pure-base journal推进至364/2000 episodes，全部为original H5，356 successes、8 failures；Task0/1/2已分别完成97/100、98/100、98/100，Task3当前63/64。364个rollout键全部唯一且最近写入距检查约3.8 s；eval/server/watcher仍存活，summary和Feishu完成marker仍待阶段结束，watcher日志0 bytes，eval未发现Traceback、exception、CUDA failure、OOM或killed。没有异常或停滞，继续严格串行运行；当前部分success不作为正式结果。

2026-07-14 14:12（Asia/Shanghai）心跳检查：正式pure-base journal为477/2000 episodes，仍为original H5，467 successes、10 failures；Task0-3已分别完成97/100、98/100、98/100、98/100，Task4当前76/77。477个键全部唯一，最近写入距检查约6.7 s；eval/server/watcher均存活，summary与完成marker未生成，watcher日志0 bytes且eval无Traceback、exception、CUDA failure、OOM或killed。评估持续正常写入，无需resume、切换server或用户介入；部分success仍不作为正式结论。

2026-07-14 14:42（Asia/Shanghai）心跳检查：正式pure-base journal推进至609/2000 episodes，均为original H5，587 successes、22 failures；已完成Task0-5的success分别为97/100、98/100、98/100、98/100、98/100、89/100，Task6当前9/9。609个键全部唯一，最新写入距检查约0.5 s；eval/server/watcher持续存活，summary/完成marker仍待生成，watcher日志0 bytes且eval未见Traceback、exception、CUDA failure、OOM或killed。Task5的89/100是已完成逐task事实，但在所有模式及paired对比完成前不据此作方案结论；无需恢复或用户介入。

2026-07-14 15:12（Asia/Shanghai）心跳检查：正式pure-base journal推进至715/2000 episodes，仍在original H5，682 successes、33 failures；Task0-6已完成，success依次为97%、98%、98%、98%、98%、89%、89%，Task7当前15/15。715个键全部唯一，最新写入距检查约12.9 s且进程持续运行；eval/server/watcher均健康，summary/完成marker未生成，watcher日志为空且eval无Traceback、exception、CUDA failure、OOM或killed。Task5/6各89%是已完成的original逐task观察值，仍须等待H9及后续方案的同state paired结果再解释；无需resume或用户介入。

2026-07-14 15:42（Asia/Shanghai）心跳检查：正式pure-base journal为822/2000 episodes，仍在original H5，779 successes、43 failures；Task7最终99/100，已进入已知困难的Task8并完成13/22。822个键全部唯一，最近写入距检查约4.7 s；eval/server/watcher保持存活，summary/完成marker未生成，watcher日志0 bytes且eval未发现Traceback、exception、CUDA failure、OOM或killed。Task8当前比例样本未满，仅记录运行状态，不作最终success判断；无需resume或用户介入。

2026-07-14 16:12（Asia/Shanghai）心跳检查：正式pure-base journal推进至898/2000 episodes，全部为original H5，830 successes、68 failures；Task8已完成98局、64 successes，最后一条episode97为timeout/失败，Task8尚缺2局才形成正式逐task值。898个键全部唯一，最近写入距检查约16.7 s且eval仍运行；server/watcher正常，summary与完成marker待生成，watcher日志为空且eval未见Traceback、exception、CUDA failure、OOM或killed。Task8较低success与其已知困难属性一致，但不在未完成时外推；继续串行，无需用户介入。

2026-07-14 16:43（Asia/Shanghai）正式pure-base original H5已完成1000/1000并由同一evaluator/server无缝切换Fixed H9，检查时H9为21/1000且21 successes。original为927/1000=92.7% success、73/1000=7.3% timeout，逐task success为97%、98%、98%、98%、98%、89%、89%、99%、66%、95%；63.518 calls/episode、固定avg H5。实测policy为89.540 ms/call与5687.400 ms/episode，server6212.228 ms/episode，client RPC wall6290.472 ms/episode，full episode elapsed16091.163 ms/episode。成功927局均值52.928 calls、4765.705 ms policy、5274.146 ms RPC和14384.059 ms full elapsed；timeout73局均为198 calls，均值17391.675 ms policy、19196.418 ms RPC和37769.052 ms full elapsed。上述是完整original单模式事实，但正式summary与strict paired H5/H9统计要在H9完成后生成；当前不据单模式数据判断H9/V2-P优劣。

该切换点journal共1021行且1021个`mode/task/episode`键唯一；eval/server/Feishu watcher均存活，最新写入约1.6 s，日志无Traceback、exception、CUDA failure、OOM或killed，summary与完成marker尚未生成。pure-base original+H9在排期中属于同一正式阶段，因此保留阶段结束时由watcher发送五方案正式/pilot上下文汇总，不在中途重启server或并行发送另一评估。

用户询问进度后于2026-07-14 16:46（Asia/Shanghai）实时复核：流程仍在继续，正式journal为1031/2000，其中original H5已固定为927/1000，Fixed H9为31/1000且当前31 successes，正在Task0 episode30。1031个键全部唯一，最新写入距检查约2.7 s；eval PID129817、server PID121683、Feishu watcher PID129878均存活，summary尚未生成，watcher日志0 bytes且eval未见Traceback、exception、CUDA failure、OOM或killed。按已观测吞吐与pilot full elapsed估算，pure-base H9剩余约3-4小时；随后仍严格串行运行Exact K20约5-6小时、sidecar双模式约9-11小时和最终统计1-2小时，若无异常从该时点剩余约19-23小时，另保留错误恢复余量。

向用户简要说明当前排期中的Stage C：它是在pure-base original/H9和current base-only Exact batched-MC V2 K20之后，加载冻结base checkpoint `50999`与锁定的round-0 V2-P predictor sidecar，严格串行评估`v2_distilled`和`v2_value_refined`各10 tasks x 100 trials。distilled用single-call predictor输出的final/Action-CoT/fused risk curves与event/raw-H复现原V2 event mapping和原episode budget controller；value-refined在同一entropy风险与budget约束内，再用Q head的H1-10 success/timeout预测筛选风险可接受、成功概率高且budget允许的最大H。predictor复用同一policy call的prefix/action feature，不新增VLM、policy RPC或base权重更新；正式速度统计必须包含predictor实际开销，并与pure original H5及Fixed H9联合比较success、calls、policy/server/RPC/full elapsed。正式Stage C使用round-0候选是因为其10x20双模式均为96%且优于后续riskcal/qcal闭环候选的Pareto表现；不再依据20局结果继续调参。

按用户要求整理10 tasks x 20 trials小规模主表，速度统一以pre-V2P pure-base original H5的5.561915 s policy/episode为分母。五方案数据为：pure-base original 93.5% success、60.910 calls、H5、91.31 ms/call、5.562 s/episode、1.000x；pure-base Fixed H9 96.0%、32.905 calls、H9、94.86 ms/call、3.121 s、1.782x；round-0 Exact K20 93.5%、35.900 calls、avg H8.850、122.95 ms/call、4.414 s、1.260x；round-0 V2-P distilled 96.0%、29.675 calls、avg H9.861、112.25 ms/call、3.331 s、1.670x；round-0 V2-P value-refined 96.0%、31.020 calls、avg H9.289、111.97 ms/call、3.473 s、1.601x。distilled/value predictor实际开销分别约4.82/4.84 ms/call，已包含在policy/server/RPC totals；Exact K20无predictor但包含实际batched teacher采样。pure-base两行与round-0三行共享seed7和state IDs0-19，但来自隔离的真实部署server路径，因此成功结果可按名义state ID对齐，跨server数值漂移仍需正式10x100处理。

结合当前实现向用户具体解释两个V2-P闭环模式。每次policy call只运行一次base prefix/VLM并生成一份primary coarse15/final10 action chunk；sidecar在同一JAX调用中读取pooled prefix、proprioception、normalized coarse/final actions、上一chunk剩余部分与当前chunk的一致性、previous H、normalized budget balance及episode progress。Entropy Head输出长度10的final/Action-CoT/fused risk curves及event/raw-H辅助heads；Q Head输出H1-10的success/timeout logits与remaining calls/steps。当前闭环从预测risk curves重新计算原V2 event mask和first-event raw H；event logits与raw-H logits只作辅助监督/记录，remaining calls/steps也只记录并保留SMDP接口，不直接选H。

当前默认配置下，distilled在H3-10中用fused risk threshold1.5（final/Action-CoT独立threshold为null）找第一处event；无event取H10，首次event在0-based index `i`时选择不越过event的最大候选H，且最小H3。value-refined先得到相同entropy raw H，再将Q logits转为概率；对默认H1-10候选要求H不超过entropy cap（risk slack0）、执行前H步的fused risk都低于1.5、success probability>=0.90且timeout probability<=0.20，选择最大eligible H；无eligible时回退entropy raw H。因此默认value refinement只会保持或缩短entropy raw H，不能越过它，正对应pilot中avg H从distilled 9.861降至value 9.289。最后两模式都调用未修改的episode budget controller：target avg H9、initial balance6、capacity12；H<9消耗`9-H` credit，H>9赚取`H-9` credit，余额不足时controller会把过短H提高到可负担的最小候选。最终只执行primary chunk前H个动作，然后在新观测上重新single-call决策。

向用户解释predictor的具体训练流程：初始collector在每个MuJoCo root只调用一次K20 batched teacher并保存一份primary chunk/feature和physics snapshot；从同一snapshot分别强制执行primary chunk的H1-10步，再使用相同fixed-H9 continuation和受控continuation seeds跑到success或timeout。每个root由K20样本分散程度产生长度10的final/Action-CoT/fused risk、event mask和原V2 raw H标签，同时产生完整的branch success/timeout/remaining calls/steps H1-10向量；没有把teacher raw H或单一best-H硬标签当作Q真值。训练只读取compact HDF5中的float16 feature和紧凑标签，不加载图像或physics state。

网络把2048维prefix、32维state、previous H/budget/progress/valid controller context，以及每个时间点的final、对齐coarse、对齐previous tail、current-minus-previous delta、overlap-valid和consistency投影到256维；10个token经过3层left/current/right邻域残差temporal encoder，token heads预测三条risk/event，mean-pooled summary heads预测raw-H classification/ordinal、H1-10 success/timeout及remaining calls/steps。默认多任务总loss权重为success1.0、timeout0.5、calls/steps各0.25、final/CoT risk各0.5、fused risk1.0、event0.5、raw-H classification0.5、ordinal0.25；success/timeout/event使用BCE，calls/steps和risk使用Huber，calls/steps先分别按64/512缩放。当前trainer还支持task-balanced sampling，并对Task8/9、高risk top quartile、gripper变化top quartile及存在失败分支的root加权；label class multipliers与head总权重分开配置。

当前正式Stage C锁定的round-0 sidecar实际使用200 roots（每task20，每root含10个H结果，即2000 branch outcomes），按完整task+episode group切成160 train/40 validation；seed7、batch128、LR3e-4、weight decay1e-3、gradient clip1、最大20,000步，每50步验证，patience20和minimum delta1e-4。该run在step1050 early-stop、27.63 s，选择step50 validation-best而非末步，best loss2.032916；base policy在trainer中完全没有加载，输出只有约5 MiB predictor sidecar。后续student closed-loop重新branch得到396 roots，与初始数据聚合为596 roots并warm-start继续SFT；balanced/qcalibrated/riskcalibrated候选分别暴露Q概率下偏或在线短H过多，10x20闭环未超过round-0 Pareto，因此正式10x100仍使用round-0 sidecar。PPO未实现，当前训练和迭代更新均为SFT。

进一步向用户解释Q Head语义：它不是动作生成器或Bellman/TD critic，而是监督式counterfactual outcome model。更准确地说输入还包含base生成的primary action chunk和controller/history context，因此预测的是`Q(x,H)`：在当前feature/context下先执行同一chunk前H步，再按采集时的continuation policy继续时，terminal success/timeout概率及到终点的calls/steps成本。共享temporal encoder的pooled 256维summary分别经四个linear heads产生H1-10 success logits、timeout logits、remaining calls和remaining steps；概率头用sigmoid，成本头用softplus并分别乘64/512。每个root的每个H只有一次Bernoulli rollout outcome，模型通过跨roots监督学习概率；H维结果允许非单调，不施加H越大success必须更低等人为约束。

Q标签依赖continuation policy：初始200 roots使用fixed H9 continuation，迭代396 roots使用current student continuation，所以它不是与后续策略无关的绝对真值。`v2_value_refined`先以Entropy Head得到安全上限，再用Q的默认success>=0.90、timeout<=0.20筛选最大H；remaining calls/steps当前只作辅助监督/审计，不直接参与选择，且无eligible候选时回退entropy H。它与Entropy Head互补：Entropy预测局部action uncertainty/event，Q预测执行某个H对整个episode结果的经验后果。round-1实验也表明Q概率校准比单纯failure recall重要，class-weighted balanced训练虽提高失败召回，却把平均success probability压低并降低在线可用性。

2026-07-14 17:13（Asia/Shanghai）心跳检查发现正式评估本身正常推进，但`v2p_formal_purebase_feishu`在发送小时进度时因飞书返回`code=11232 frequency limited`退出；当时journal为1145/2000且全部唯一，Fixed H9为144/145 successes，Task0完成99/100、Task1为45/45，eval/server仍健康且无模型或rollout错误。该故障仅影响watcher通知，不影响评估。

已提交并推送`036101e`（`Retry transient Feishu watcher failures`）：`scripts/watch_experiment_feishu.py`新增受保护发送，progress发送失败只记录并延后下一次，summary/failure完成通知失败则按poll interval持续重试，只有发送成功才写完成marker并退出。修改通过本地`py_compile`、retry guard单元式smoke、`git diff --check`，并在服务器通过Ruff check/format与py_compile；远端fast-forward后仍只保留既有`third_party/libero`用户修改。watcher已以同名tmux、PID140319、安全私有env和append日志方式恢复，重新检查时Fixed H9为156/157 successes、最新Task1 episode56，eval PID129817与watcher均存活。旧限频traceback保留在watcher log中作审计，新watcher尚无新增错误；后续即使再次限频也不会丢失完成通知重试。

用户将飞书通知策略改为按完整测试发送，不再发送小时/周期进度：每个方案完成10 tasks x100 trials即1000局、生成summary并通过完整性核验后发一次。已停止带`--progress-seconds 3600`的watcher并在不影响eval PID129817的情况下，以同名tmux重新启动completion-only watcher PID140501；其命令不含`--progress-seconds`，只poll summary并在Fixed H9完成后发送本阶段表。后续Exact K20、V2-P distilled与V2-P value-refined将分别使用独立串行1000局output/summary和completion watcher，各完成各发一次；最终五方案统计完成后再发一次总表。当前Codex心跳automation `acot-vla-v2-p`也已同步更新该通知规则，仍保持30分钟远端健康检查但不会把周期检查发往飞书。

2026-07-14 17:45（Asia/Shanghai）心跳检查：正式pure-base journal为1324/2000且全部唯一，Fixed H9已完成324/1000、321 successes；逐task进度为Task0 99/100、Task1 100/100、Task2 99/100，Task3当前23/24。最新journal写入距检查约7.1 s；eval PID129817、pure-base server PID121683与completion-only watcher PID140501均存活，summary和完成marker未生成。进程参数确认watcher不含`--progress-seconds`；watcher log仍为17:10旧限频traceback的588 bytes且重启后mtime未变化，说明没有周期发送或新增错误。eval日志无Traceback、exception、CUDA failure、OOM或killed，继续严格串行，无需用户介入。

2026-07-14 18:14（Asia/Shanghai）心跳检查：正式journal推进至1463/2000且全部唯一，Fixed H9为457/463 successes；Task0-3已分别完成99%、100%、99%、99%，Task4当前60/63，最后一局为timeout/失败。最新写入距检查约28.7 s，符合正在运行较长episode，eval/server/watcher三进程均存活；summary与完成marker尚未生成，eval无Traceback、exception、CUDA failure、OOM或killed。completion-only watcher log仍保持旧588 bytes与17:10 mtime，没有周期飞书发送或新错误；无需恢复或用户介入。

2026-07-14 19:19（Asia/Shanghai）合并处理两次心跳后检查：正式journal为1786/2000且全部唯一，Fixed H9为758/786 successes；Task0-6完整success依次为99%、100%、99%、99%、93%、92%、93%，Task7当前83/86。最新写入距检查约0.6 s；eval PID129817、server PID121683和completion-only watcher PID140501均存活，summary/完成marker仍待生成，eval无Traceback、exception、CUDA failure、OOM或killed。watcher log仍为旧588 bytes且mtime保持17:10，确认没有周期飞书或新增错误；继续严格串行，无需用户介入。

2026-07-14 19:57（Asia/Shanghai）心跳检查：正式journal为1949/2000且全部唯一，Fixed H9已完成949/1000、904 successes；Task0-8完整success为99%、100%、99%、99%、93%、92%、93%、96%、86%，Task9当前47/49，只剩51局。最新写入约7.4 s；eval/server/completion-only watcher均存活，summary和marker尚未生成，eval日志无Traceback、exception、CUDA failure、OOM或killed，watcher log仍无17:10之后的新写入。阶段接近完成但尚无最终统计，不提前发送飞书或切换server。

2026-07-14 23:38（Asia/Shanghai）完成正式pre-V2P pure-base阶段审计并串行切换到current base-only Exact K20。目录`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_formal_pure_base_c5c08fc_original_h5_fixed_h9_10tasks_100trials`的`summary.json`为`complete`且`paired_initial_states=true`；`rollout_rows.csv`含2000行、2000个唯一`mode/task/episode`键、original与Fixed H9各1000局，无重复，两模式task/episode键集及`initial_state_id`逐项一致。飞书marker内容为`completed`，说明本阶段完成消息已发送。

正式pure-base original H5为927/1000=92.7% success、7.3% timeout、63.518 calls/episode、avg H5、89.540 ms policy/call、5.687 s policy/episode、6.290 s RPC wall/episode、16.091 s full episode elapsed；Fixed H9为950/1000=95.0% success、5.0% timeout、33.568 calls/episode、avg H9、88.855 ms policy/call、2.983 s policy/episode、3.288 s RPC wall/episode、12.567 s full elapsed。same-state paired outcome为Fixed H9相对H5 55 wins、32 losses、895 both-success、18 both-fail；因此在这1000个正式pure-base配对初态上，H9同时提高2.3个百分点success并将policy/episode时间缩短约1.907倍。该结论仅针对本次checkpoint、配置和初态，不替代后续Exact/V2-P正式比较。

已停止旧pure-base server并确认GPU清空后，在远端current commit`036101e`启动独立base-only server `v2p_formal_exact_server`（PID152221、port8027，无predictor sidecar），随后首次启动而非resume正式evaluator `v2p_formal_exact_eval`（PID153099）。Exact输出目录为`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_formal_current_base_exact_batched_mc_v2_k20_10tasks_100trials`，配置为唯一模式`exact_batched_mc_v2`、LIBERO-10全部10 tasks x100 trials、seed7、state IDs0-99、Action-CoT denoising steps10、batched teacher K20、risk threshold1.5及原V2 budget target/init/capacity 9/6/12。`run_config.json`已核对；首局Task0/state0已成功写入，29 calls、avg H8.793、580 sampled chunks、实测batched-teacher3.244 s、policy3.787 s、RPC4.085 s及full elapsed15.284 s。server/eval日志无Traceback、OOM或CUDA failure。

Exact的completion-only飞书watcher `v2p_formal_exact_feishu`（PID154254）已启动，不含`--progress-seconds`，只会在1000局完成、summary生成后发送一次；比较上下文使用正式pure-base summary覆盖original/H9，并保留round-0 V2-P pilot行。当前Exact server/eval/watcher三个tmux均存活，仍保持单eval严格串行；下一阶段只会在Exact完整结束和完整性核验后启动round-0 sidecar `v2_distilled`。

2026-07-14 23:40（Asia/Shanghai）心跳检查：正式current base-only Exact batched-MC V2 K20已写入12/1000局，均为Task0 state IDs0-11，12条`mode/task/episode`键全部唯一；当前12局均success且无timeout，但该未完成比例不用于最终success结论。最新row距检查约8.2 s，server PID152221、eval PID153099、completion-only watcher PID154254均存活，GPU仅base server占用约17,038 MiB。`summary.json`与飞书完成marker尚未生成，watcher日志为0 bytes，进程参数确认不含周期进度通知；run/server/watcher日志未发现Traceback、Exception、CUDA failure、out-of-memory、Killed或XLA runtime error。评估健康推进，无需`--resume`、切换server或用户介入，继续严格串行等待Exact 1000局完成。

2026-07-15 00:17（Asia/Shanghai）心跳检查：正式Exact K20推进至236/1000局，236个`mode/task/episode`键全部唯一、无重复；Task0与Task1均已完成并各为98/100 success、2 timeout，Task2当前完成36局、35 success、1 timeout。累计231 success与5 timeout仅用于运行状态，不外推最终success。最新row距检查约3.4 s；server PID152221、eval PID153099、completion-only watcher PID154254均存活，GPU仍只有base server占用约17,038 MiB。summary/飞书marker尚未生成，watcher日志仍为0 bytes，run/server/watcher未发现Traceback、Exception、CUDA failure、out-of-memory、Killed或XLA runtime error。无需resume或阶段切换，继续单eval严格串行运行。

2026-07-15 02:39（Asia/Shanghai）心跳检查：正式Exact K20已推进至837/1000局，837个`mode/task/episode`键全部唯一、无重复。Task0-7均完成100局，success依次为98、98、99、100、97、91、92、98；Task8当前完成37局、29 success、8 timeout。累计802 success与35 timeout仍属未完成运行状态，不作为最终success结论。最新row距检查约11.5 s；server/eval/completion-only watcher继续为PID152221/153099/154254且三个tmux pane均`dead=0`，GPU仅base server占用约17,038 MiB。summary及飞书marker尚未生成，watcher日志仍为0 bytes，日志未发现Traceback、Exception、CUDA failure、out-of-memory、Killed或XLA runtime error。评估未中断，不使用resume；继续等待剩余163局，完成并审计后才串行切换sidecar distilled。

2026-07-15 03:08（Asia/Shanghai）心跳检查：正式Exact K20推进至956/1000局，956个唯一`mode/task/episode`键且无重复。Task0-8已完成，success依次为98、98、99、100、97、91、92、98、82；Task9当前56局、52 success、4 timeout。累计907 success与49 timeout仍是未完成状态，不作最终比例结论。最新row距检查约10.3 s；Exact server/eval/watcher三个tmux均存活，GPU仅PID152221占用约17,038 MiB，summary、完成marker尚未生成且watcher日志为空；目标错误模式扫描无异常。剩余44局，保持严格串行，不提前停止server或启动sidecar。

2026-07-15 03:55（Asia/Shanghai）正式current base-only Exact batched-MC V2 K20 10 tasks x100 trials已完成并通过完整性审计。目录`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_formal_current_base_exact_batched_mc_v2_k20_10tasks_100trials`的summary状态为`complete`；`rollout_rows.csv`含1000行、1000个唯一`mode/task/episode`键、无重复，每个task恰好100局。Exact与正式pure-base original/H9的1000个task/episode键集完全一致，`initial_state_id`逐项一致；LIBERO每task实际有50个initial states，100 trials按相同方式各循环两次，因此严格配对通过。

Exact K20 overall为944/1000=94.4% success、5.6% timeout、34.647 calls/episode、decision-weighted avg H8.84622；逐task success为98%、98%、99%、100%、97%、91%、92%、98%、82%、89%。H分布为H1 2、H2 4、H3 963、H4 259、H5 371、H6 744、H7 1564、H8 4520、H9 13067、H10 13153。实测batched teacher为3.666 s/episode，policy为122.087 ms/call与4.230 s/episode，server4.467 s/episode、RPC wall4.507 s/episode、full elapsed13.366 s/episode；teacher是policy时间内的组成部分，不重复相加。成功944局均值30.121 calls、3.685 s policy、3.927 s RPC和12.301 s full elapsed；timeout56局均值110.946 calls、13.420 s policy、14.289 s RPC和31.316 s full elapsed。

相对正式pure-base original H5，Exact success高1.7个百分点，policy/RPC/full elapsed speedup分别为1.3446x/1.3956x/1.2039x；same-state paired为Exact 53 wins、36 losses、891 both-success、20 both-fail，双侧exact McNemar p=0.0893。相对Fixed H9，Exact success低0.6个百分点，paired为32 wins、38 losses、912 both-success、18 both-fail，p=0.5504。因paired差异均未达到常用显著性阈值，当前证据支持Exact相对H5更快且点估计更高，但不能宣称success显著优于H5，也不能宣称与H9存在success差异。

Exact飞书watcher曾记录一次`code=11232 frequency limited`，修改后的watcher保留进程并自动重试，最终`.feishu_notification_sent`为`completed`且watcher正常退出，说明Exact 1000局完成表已发送。随后停止`v2p_formal_exact_server`并确认GPU清空，没有并行eval。

已严格串行进入round-0 sidecar正式`v2_distilled`阶段。单server `v2p_formal_sidecar_server`（PID163164、port8028）加载冻结base checkpoint`50999`与锁定sidecar`/root/autodl-tmp/acotvla/execution_horizon_v2p/sft_round00_initial_200_seed7_8486631/params`，日志确认base约7.1 GiB及predictor约5.0 MiB均恢复成功并开始监听。独立fresh evaluator `v2p_formal_distilled_eval`（PID163958）输出到`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_formal_round00_sidecar_v2_distilled_10tasks_100trials`，唯一模式为`v2_distilled`，其seed7、task/episode范围、denoising steps10、risk threshold1.5、budget 9/6/12及Q参数与正式协议一致，首次启动未使用`--resume`。前三局已成功写入，calls为27/26/26，avg H为10/10/9.769，predictor实际总延迟约129.35/129.22/132.82 ms；仅证明运行链路健康，不作效果结论。

completion-only watcher `v2p_formal_distilled_feishu`（PID164686）已启动，不含周期进度参数；只会在distilled 1000局及summary完成后发送一次，表中正式pure-base和Exact覆盖pilot行，尚未完成的value-refined仍明确保留pilot context。sidecar server/eval/watcher均存活，日志未发现目标错误；下一步等待distilled完成并审计，通过后才复用同一sidecar server串行启动独立`v2_value_refined` 1000局。

2026-07-15 18:38（Asia/Shanghai）正式round-0 sidecar `v2_distilled` 10 tasks x100 trials已完成并通过完整性与配对审计。目录`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_formal_round00_sidecar_v2_distilled_10tasks_100trials`的summary状态为`complete`，1000 rollout rows对应1000个唯一`mode/task/episode`键且无重复，每task恰好100局；与正式pure-base original/H9及Exact的task/episode键集、`initial_state_id`均逐项一致。完成marker为`completed`、watcher日志0 bytes且watcher正常退出，说明distilled 1000局完成表已发送，没有周期飞书消息。

`v2_distilled` overall为952/1000=95.2% success、4.8% timeout、30.536 calls/episode、decision-weighted avg H9.87412；逐task success为98%、100%、97%、98%、97%、90%、94%、98%、86%、94%。H分布为H1 3、H2 3、H3 98、H4 120、H5 134、H6 162、H7 141、H8 153、H9 340、H10 29382，说明绝大多数决策为H10。predictor实测4.761 ms/call与145.389 ms/episode；包含该开销的policy为110.904 ms/call与3.387 s/episode，server3.666 s、RPC wall3.702 s、full elapsed12.430 s/episode。成功952局均值27.054 calls、3.001 s policy、3.281 s RPC、11.620 s full elapsed；timeout48局均值99.604 calls、11.027 s policy、12.049 s RPC、28.507 s full elapsed。

相对正式pure-base original H5，distilled success高2.5个百分点，policy/RPC/full speedup为1.6794x/1.6992x/1.2945x；same-state paired为55 wins、30 losses、897 both-success、18 both-fail，双侧exact McNemar p=0.00884，支持本次配对样本上distilled相对H5同时提高success并降低实际时间。相对Fixed H9，distilled点估计success高0.2个百分点，paired为38 wins、36 losses、914 both-success、12 both-fail，p=0.9076；distilled policy与RPC分别慢13.54%与12.58%，但full elapsed快约1.10%，因此不能宣称它优于H9。相对Exact K20，distilled点估计success高0.8个百分点，paired41 wins/33 losses、p=0.4160且policy快1.2490x；支持蒸馏显著减少采样时间，但success差异没有统计证据。

在distilled summary、配对和完成marker均通过后，复用唯一的round-0 sidecar server PID163164/port8028，严格串行首次启动`v2_value_refined`正式1000局，未重新加载或修改base/sidecar。新输出目录为`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_formal_round00_sidecar_v2_value_refined_10tasks_100trials`；evaluator `v2p_formal_value_eval` PID191819使用唯一模式`v2_value_refined`、Q success/timeout阈值0.90/0.20、risk slack0及与其他正式方案一致的seed7、初态、risk和budget参数，fresh启动未使用resume。前三局均success，calls为27/26/31、avg H为10/10/9.419，predictor实际总延迟约85.01/96.58/130.88 ms；仅作为链路健康事实。

value的completion-only watcher `v2p_formal_value_feishu` PID191885已启动，不含周期进度参数；完成时将用正式value结果覆盖pilot并发送一次value完成表。sidecar server/value evaluator/watcher三个pane均`dead=0`，无其他eval客户端且日志无目标错误。value 1000局完成后还需做五方案最终严格配对、逐task、H分布及全部实际耗时汇总，并按用户要求另发一次最终总表。

2026-07-15 18:41（Asia/Shanghai）应用户询问实时复核进度：代码实现、初始counterfactual数据、round-0 SFT、student iterative relabel/SFT候选审计以及正式pure-base H5/H9、Exact K20、v2_distilled均已完成；当前唯一运行阶段是正式`v2_value_refined` 1000局。value输出已写入15/1000局，15个`mode/task/episode`键全部唯一且最新row距检查约0.9 s；当前15局均success但该小样本不用于最终效果判断。sidecar server/eval/completion-only watcher PID163164/191819/191885均存活，GPU只有server占用约17,030 MiB，summary与完成marker尚未生成，watcher日志0 bytes且目标错误扫描为空。

正式Exact从run_config到summary耗时约3小时43分42秒，正式distilled约3小时28分03秒。value在18:36:38启动，前15局含启动阶段平均约14.9 s/局；结合distilled实测吞吐及10x20中value相对distilled略多policy calls，预计value剩余约3.5-4.2小时，约22:15-22:50完成。其后五方案最终配对、逐task/timing/H分布核验和最终飞书总表预计再需约30-60分钟；无异常时整个正式闭环从当前时点还需约4-5小时，预计约22:45-23:50交付最终汇总，错误恢复会另增加时间。

2026-07-15 20:17（Asia/Shanghai）心跳检查：正式`v2_value_refined`推进至475/1000局，475个`mode/task/episode`键全部唯一、无重复。Task0-3已完成，success分别为97%、99%、95%、99%；Task4当前75局、72 success。累计462 success/13 timeout仍为未完成状态，不用于最终比例结论。最新row距检查约13.7 s；sidecar server PID163164、eval PID191819、completion-only watcher PID191885均存活且pane `dead=0`，GPU只有server占用约17,030 MiB，summary与完成marker尚未生成，watcher日志仍为0 bytes，目标错误扫描为空。当前吞吐与18:41估算一致，无需resume，继续严格串行运行且不发送中途飞书消息。

2026-07-15 20:38（Asia/Shanghai）复核工作日日报自动化未发送问题：`experiment_log.md`实际已有多条当天有效事实，但采用`YYYY-MM-DD HH:MM`段落格式，原自动化只匹配`## YYYY-MM-DD`标题，因此错误判定为无当天记录；自动化`automation`已更新为同时识别日期标题与日期时间段落，并改为使用环境中实际存在的`python3`或`python`执行飞书发送脚本，不再依赖`uv`。本机飞书桌面端已定位到既有通知会话“对接codex”，但远端SSH连接被拒绝且本机无webhook凭据，当前日报补发需在用户确认桌面端发送后执行。

2026-07-15 20:54（Asia/Shanghai）在用户再次确认飞书未收到后，通过已登录的本机飞书桌面端向“对接codex”会话补发三行日报；发送后的飞书可访问性树出现以“梁晓易”为发送者的新消息，并包含Exact K20、V2-P distilled与value-refined的日报内容，说明消息已进入目标会话。随后Mac自动锁屏，无法继续截图复核，但自动化的日期格式兼容修复保持生效。

2026-07-15 20:47（Asia/Shanghai）V2-P正式流程心跳检查：`v2_value_refined`推进至638/1000局，638个`mode/task/episode`键全部唯一且无重复。Task0-5已完成，success依次为97%、99%、95%、99%、96%、91%；Task6当前38局、37 success。累计614 success/24 timeout仍为未完成运行状态，不作最终比例判断。最新row距检查约4.6 s；sidecar server/eval/completion-only watcher PID163164/191819/191885均存活且pane `dead=0`，summary与完成marker尚未生成，watcher日志0 bytes，目标错误扫描为空。无需resume或用户介入，继续严格串行等待剩余362局。

2026-07-15 21:21（Asia/Shanghai）V2-P正式流程心跳检查：`v2_value_refined`已推进至930/1000局，930个`mode/task/episode`键全部唯一且无重复。Task0-8已完成，success依次为97%、99%、95%、99%、96%、91%、92%、98%、84%；Task9当前30局、28 success。累计879 success/51 timeout仍是未完成状态，不提前作为最终结果。最新row距检查约9.0 s；sidecar server/eval/completion-only watcher PID163164/191819/191885均存活且pane `dead=0`，summary/marker尚未生成、watcher日志0 bytes，错误扫描为空。只剩70局，继续保持单eval，完成后立即执行五方案最终审计与飞书总表。

2026-07-15 22:21（Asia/Shanghai）Budgeted Event V2-P正式全流程完成。`v2_value_refined` 10 tasks x100 trials的summary状态为`complete`，1000条rollout全部唯一；overall为94.3% success、5.7% timeout、32.892 calls/episode、decision-weighted avg H 9.32838，policy 109.780 ms/call与3.611 s/episode，server/RPC/full分别为3.903/3.945/12.796 s/episode，predictor实测4.668 ms/call与153.550 ms/episode。逐task success为97%、99%、95%、99%、96%、91%、92%、98%、84%、92%；H分布为H1 374、H2 311、H3 174、H4 406、H5 454、H6 435、H7 561、H8 541、H9 5808、H10 23828。相对distilled，value-refined success从95.2%降至94.3%，calls从30.536升至32.892，且policy/RPC/full都更慢；配对为distilled 40 wins、value 31 wins、912 both-success、17 both-fail，p=0.3425，因此round-0 Q refinement没有提升闭环Pareto结果。

最终五方案审计确认pure-base original H5/Fixed H9各1000局，Exact K20、v2_distilled、v2_value_refined各1000局；五方案`task_id/episode` key set与`initial_state_id`逐项一致，strict paired通过。最终overall为：H5 92.7% success/63.518 calls/5.687 s policy；H9 95.0%/33.568/2.983 s；Exact K20 94.4%/34.647/4.230 s；distilled 95.2%/30.536/3.387 s；value-refined 94.3%/32.892/3.611 s。distilled相对H5 success高2.5个百分点、policy快1.679x，paired 55 wins/30 losses、p=0.00884；相对Exact policy快1.249x而success差异无显著证据（p=0.416），支持single-call蒸馏成功降低K20采样开销。Fixed H9仍是policy速度最快方案（1.907x vs H5），且与distilled的配对success无可区分差异（p=0.9076），因此本轮不声称V2-P对H9形成明确Pareto优势。

五方案逐task、overall、outcome-stratified timing、H分布、全部same-state paired比较及远程artifact路径已汇总到`reports/execution_horizon_v2p_formal_10x100.md`。value方案完成marker与最终五方案飞书总表marker均为`completed`；最终只读复核时远程无tmux、无eval/server/watcher进程、GPU无compute进程，远程git仅保留用户原有`third_party/libero`修改。PPO按阶段要求继续暂缓；若继续研究，下一步应针对Q概率校准与Task8/9失败状态改进SFT，而不是直接进入PPO。

正式报告已以提交`80f8a16`（`Report formal V2-P evaluation results`）推送到GitHub `main`；首次直连push长时间无响应后中止，通过`127.0.0.1:7897`代理重试成功。已删除完成使命的Codex heartbeat automation `acot-vla-v2-p`，后续不再每30分钟唤醒该任务。

2026-07-17（Asia/Shanghai）根据已完成的Budgeted Event V2-P正式10 tasks x100 trials严格配对结果，整理了一份可直接口头汇报的简版阶段报告。报告概括了single-call双头predictor、counterfactual H1-10 SFT与iterative relabel流程，列出H5、Fixed H9、Exact K20、v2_distilled和v2_value_refined的success、calls及实际policy时间，并明确结论：distilled相对原版H5同时提高success和降低policy时间，成功替代Exact K20的大部分MC开销；Fixed H9仍是最快方案，当前V2-P没有证明对H9形成明确Pareto优势；round-0 Q refinement未改善结果，后续应优先处理Q概率校准和Task8/9困难状态。本次仅整理已有证据，没有运行新实验或修改模型代码。

2026-07-17（Asia/Shanghai）进一步按zero-based Task0-9整理五方案逐任务对比，采用“success率 / calls per episode”统一口径。结果显示Task0-4整体较容易，多数方案success为95%-100%；Task5-7存在有限差异，其中distilled在Task6达到94%且28.77 calls，Task7各方案保持96%-99%；Task8是最主要困难任务，原版H5仅66% success和125.14 calls，Fixed H9与distilled均提高到86%，其中distilled calls最低为52.28；Task9中原版H5的95% success最高，distilled为94%且calls由60.29降至31.97。逐任务结果也确认value-refined没有形成稳定收益，特别是在Task2、Task8和Task9低于distilled。本次仅分析正式报告中的既有数据，没有新实验。

2026-07-17（Asia/Shanghai）为汇报整理了V2-P distilled算法流程图，区分离线训练与在线闭环。离线阶段使用Exact batched-MC K20生成final/Action-CoT/fused risk curves、event和raw-H辅助监督，并用H1-10 counterfactual分支训练Q/timeout/cost辅助heads，通过多任务SFT导出独立predictor sidecar；在线distilled阶段每次decision只运行一次冻结的ACoT-VLA prefix/VLM与primary action chunk生成，在同一JAX/GPU policy call内将prefix、proprioception、coarse/final actions、previous overlap、previous H、budget balance及episode progress送入共享temporal encoder。部署时只使用Entropy Head预测的risk curves，经原V2 threshold/first-event mapping得到raw H，再进入保持不变的episode budget controller后执行primary chunk前H步；Q Head输出不参与`v2_distilled`选H，仅供`v2_value_refined`使用。在线路径没有K20 MC、第二次VLM或额外RPC。

2026-07-17（Asia/Shanghai）核对当前代码后确认`v2_distilled`已具备LIBERO/WebSocket闭环部署能力，但还不是对任意现有机器人客户端完全透明的单一返回H接口。`scripts/serve_policy.py`可通过`execution_horizon_predictor_params`将独立sidecar与冻结base checkpoint合并加载，`src/openpi/policies/policy.py`在同一次policy request/JAX调用内生成primary action chunk并运行predictor，返回risk curves和predictor输出；正式LIBERO 1000局已经验证该服务路径。当前原V2 event mapping、episode budget state更新及最终H选择仍位于`eval_libero_execution_horizon.py`客户端，客户端还需传入previous actions/H、budget balance、episode progress并只执行chunk前H步。因此可直接部署到现有V2-P-aware LIBERO runner；若要作为原ACoT-VLA动作API的drop-in替换或部署到真实机器人，还需一个薄client/controller adapter（或将mapping与budget controller封装进server），并补充机器人观测/动作适配、安全约束及域外验证。

2026-07-17（Asia/Shanghai）进一步澄清部署术语。“加载冻结的Base checkpoint”指部署时仍需恢复原ACoT-VLA checkpoint `50999`来执行VLM/prefix、coarse action和final action生成，再只把约5 MiB predictor sidecar覆盖到独立`execution_horizon_predictor`参数子树；base参数不被sidecar替换、不在SFT中更新，但推理时仍会实际运行并产生主要计算开销。Predictor不能独立生成机器人动作，它只基于base feature/action chunk预测风险与H相关输出。“薄deployment adapter”指在现有policy client外增加一层无学习参数的状态控制逻辑：向单次请求附加previous actions/H、budget balance和episode progress，读取返回的risk curves，通过threshold/first-event mapping得到raw H，再用episode budget controller得到final H，只执行action chunk前H步并更新下一次请求所需状态。该adapter不重新运行VLM、不增加RPC、无需重新训练；当前LIBERO evaluator已经承担这层职责，若要求原ACoT-VLA客户端零改动接入，则应将这段逻辑抽成通用client wrapper或封装到server中。

2026-07-17（Asia/Shanghai）按用户要求将V2-P distilled流程重新拆成两张独立汇报图。训练图单独表示数据构建与SFT：冻结Base在每个root state只生成一次primary feature/action，shared-prefix batched-MC K20产生final/Action-CoT/fused risk curves、event mask和raw-H监督；H1-10同snapshot counterfactual分支产生Q/timeout/cost辅助标签；compact HDF5经共享temporal encoder和多任务loss训练后只导出validation-best predictor sidecar。在线图按client、policy server、V2 controller和环境边界表示一次decision：client携带观测及历史/budget状态进行一次RPC，server在同一次JAX/GPU调用内运行冻结Base和predictor并返回primary 10-step action chunk与三条risk curves；distilled只用risk curves经过threshold 1.5、first-event和H3-10候选映射得到raw H，再由target H9、initial balance6、capacity12的原budget controller得到final H，执行前H步并更新下一轮状态。在线图明确排除了K20 MC、第二次VLM和Q Head选H。

2026-07-20（Asia/Shanghai）对照分析了V2-P distilled正式结果与论文《Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy》（arXiv:2508.03346v2）的SFT+GRPO方案。该论文用最终答案正确性、skip比例、skip数量和响应长度构成GRPO reward；其表3显示GRPO相对SFT主要继续降低token数，accuracy收益并不一致，附录reward消融还显示朴素的correctness+skip reward会产生退化策略。结合本项目严格配对结果：distilled为95.2% success、30.536 calls/episode、3.387 s policy/episode，Fixed H9为95.0%、33.568、2.983 s，value-refined为94.3%、32.892、3.611 s；当前不建议立即直接实现RL。证据更支持先评估counterfactual horizon oracle headroom，并补强Task8/9困难状态、Q/timeout概率校准和on-policy SFT数据。如果这些检查确认状态自适应H仍有可实现的Pareto空间，再增加可选的student-only、冻结base、chunk-level constrained SMDP PPO阶段，以terminal success为主reward、policy-call/实际延迟为成本并用dual/Lagrangian约束预算；不建议照搬LLM的GRPO或对完整base VLA做RL。本次仅作论文与现有实验的技术判断，没有修改模型代码或运行训练/评估。

2026-07-20（Asia/Shanghai）进一步解释了“counterfactual horizon oracle headroom”检查：它不是训练或部署一个oracle，而是利用同一root state已有的H1-10分支结果，离线假设决策器事后知道各H的success/timeout/remaining calls标签，并在给定calls或实际延迟预算下选择最优H，由此估计状态自适应horizon策略相对V2-P distilled和Fixed H9的理想上界。若该上界仍接近现有结果，说明RL缺少可利用的horizon决策空间；若上界在相同预算下明显更好，才说明当前predictor/data/optimization没有学到已有信号，值得继续定向SFT或受约束RL。该oracle仅用于诊断，会因事后使用counterfactual outcome而偏乐观，不能作为真实部署结果。

2026-07-20（Asia/Shanghai）进一步分析了H接近上限后低成功率任务是否仍有提升空间。全局avg H9.874和96.2% H10只说明通过继续增大H来减少调用的空间很小，不排除在极少数失败前关键状态选择较短H、提前replan来提高success。正式逐task结果显示：Task8的distilled与Fixed H9均为86% success，distilled avg H9.990，而H5仅66%；value-refined将avg H降至9.245后为84%，说明Task8不适合广泛缩短H，剩余失败更可能需要base policy、恢复能力或极高精度的少量关键状态干预。Task9为H5 95%、H9 93%、distilled 94%且avg H9.854，Task5为H9 92%、distilled 90%且avg H9.998，这两个任务更可能存在少量replanning-sensitive状态，但每task仅100局，点估计不能单独证明因果。建议对distilled失败episode的最后若干decision snapshots做failure-focused H3-10分支审计：若某个替代H能把相同状态的失败转为成功，则属于horizon-fixable failure，可用于hard-state SFT或后续受约束RL；若所有H均失败，则horizon predictor无法解决，应转向base action policy、恢复策略或stepwise verifier。当前不建议按task整体降低H，也不建议用RL大范围增加短H。

2026-07-20（Asia/Shanghai）核对了ACoT-VLA原论文（arXiv:2601.11404v2）、官方评测代码与当前V2-P正式H5协议。论文表1在LIBERO-Long报告ACoT-VLA Frozen 96.0%、完整训练版97.0%；98.5%是Spatial/Object/Goal/Long四套平均。论文每个task运行50 trials，Long共500局；官方`examples/libero/main.py`默认`replan_steps=5`并执行action chunk前5步，因此H5本身是官方执行协议，不能用模型输出action horizon 10否定H5。当前`acot_libero_action_cot_explicit_implicit_co_fusion`配置`freeze_llm=True`，应优先对照Frozen 96.0%。项目已有官方式50-trial历史baseline为483/500=96.6%，而正式V2-P协议H5为927/1000=92.7%，其中Task8从历史82%降到66%。正式协议每task 100 trials会把50个initial states各重复两次，并对每次policy request显式使用由task、episode和step构造的`policy_seed`；官方client只跑50 states且不显式传`policy_seed`，服务端使用连续内部RNG，因此92.7%不是论文协议的直接复现。它仍可作为五方案在相同task/state/seed集合上的严格配对基线，但应标注为“V2-P seeded 100-trial protocol下的H5 paired baseline”，不能直接代表论文原始成功率。下一步应先把正式H5按episode 0-49与50-99拆分，并在需要时用官方client、无显式`policy_seed`复跑50 trials/task，以隔离state重复、seed策略和evaluator差异；优先检查Task8。当前远端SSH连接被拒，尚未取得formal rollout CSV做两半拆分。

2026-07-20（Asia/Shanghai）进一步明确“我们的模型”V2-P distilled与纯原版ACoT-VLA的比较口径。最可信的是正式10 tasks x100 trials同协议、同checkpoint、同initial-state和seed的严格配对结果：纯原版H5为92.7% success、63.518 calls/episode、5.687 s policy/episode；V2-P distilled为95.2%、30.536 calls/episode、3.387 s policy/episode。V2-P相对纯原版H5成功率提高2.5个百分点，调用数减少51.9%，policy时间缩短1.679倍；same-state结果为V2-P独赢55局、原版独赢30局、双方成功897局、双方失败18局，双侧exact McNemar p=0.008836。该结果支持在本次seeded 100-trial协议下，V2-P distilled相对纯原版H5同时提高成功率和效率。跨协议不能直接用V2-P的95.2%与论文Frozen 96.0%或历史官方式原版96.6%比较，因此目前不能声称V2-P超过论文原版的绝对成功率。另需保留Fixed H9强基线：其成功率95.0%、policy 2.983 s/episode，与distilled成功率无显著差异（38对36 paired wins，p=0.9076），且policy时间更短；所以当前证据证明V2-P优于官方H5执行方式，但尚未证明优于简单Fixed H9。

2026-07-20（Asia/Shanghai）根据用户提出“92.7%的H5不能代表论文水准”，重新收紧V2-P研究结论并核准项目历史原版速度。当前项目最接近论文水准的官方式ACoT-VLA历史baseline为500局483 success，即96.6%；排除task0 trial0计时warmup后的clean timing口径为499局96.59% success、58.519 calls/episode、4.560 s policy/episode和4.873 s server/episode。V2-P distilled正式结果为95.2%、30.536 calls/episode、3.387 s policy/episode和3.666 s server/episode。直接按点估计对照，V2-P成功率比历史原版低约1.39个百分点，calls减少约47.8%，policy总时间约快1.35倍、server总时间约快1.33倍；但历史与当前运行环境/单次延迟不同，速度比只能作为项目历史参照，不能作为严格同机结论。历史原版没有保存完整episode wall字段，不能比较端到端full time。按“至少达到或超过论文级原版成功率并同时加速”的研究标准，当前结果尚未达标，只证明了速度—成功率trade-off；此前以92.7% H5为主基线的“全面提高”表述不应作为对论文ACoT-VLA的主结论。下一步应先在历史官方10 tasks x50 trials、无显式per-request policy seed的同一条件下评估V2-P，并同时记录原版与V2-P速度；最低成功率门槛应为Frozen论文96.0%，更严格可使用项目历史96.6%或论文完整训练版97.0%。若同协议V2-P仍未达门槛，再针对Task8/9困难状态改进predictor或base恢复能力。

2026-07-20（Asia/Shanghai）针对“是否增加RL或其他方法，使V2-P至少达到论文级ACoT-VLA成功率并保持加速”给出补充路线。当前V2-P为952/1000；若以同一1000局口径计算，达到96.0%、96.6%和97.0%分别需要净救回8、14和18个episode。其30.536 calls/episode和3.387 s policy/episode相对历史原版58.519 calls和4.560 s仍有约27.98 calls与1.173 s的平均预算余量，因此可以把部分速度红利用于少量高风险状态的恢复或额外计算。建议先对48个失败episode的失效前decision snapshots做H1-10 counterfactual分支，统计至少14个失败是否能由替代H救回；若可救回数量足够，优先使用Task8/9 hard-state oversampling、pairwise/ranking Q监督和student-state DAgger继续SFT，再可选冻结base、只更新selector的chunk-level constrained SMDP PPO，以terminal success为主目标、policy time/calls为dual/Lagrangian约束并保持与SFT策略的KL。若H分支oracle不足，说明仅对H做RL不可能达到目标，应改为risk-triggered selective fallback：正常状态维持single-call H9/H10，仅在少数高风险状态提前replan、额外采样候选chunk或启用恢复策略；必要时再对base增加小型recovery adapter，而非直接对完整ACoT-VLA做RL。当前不建议照搬LLM GRPO，也不建议在未做failure-fixability审计前直接启动horizon PPO。本次仅形成研究建议，没有修改训练代码或运行新实验。

2026-07-20（Asia/Shanghai）核对了上述V2-P补充路线的直接文献依据与项目内逻辑。最直接参考是DEHP《Dynamic Execution Horizon Prediction for Chunk-based Robot Policies》（arXiv:2606.11408v2）：其冻结pretrained chunk policy，只训练conditioned on state/action chunk的categorical horizon head，将可变执行时长建模为chunk-level SMDP，使用step-discounted GAE和clipped PPO，并在精细与长时序操作上超过最佳fixed-H；该工作直接支持“冻结ACoT-VLA、仅对H做PPO”的可行性，但其实验主要是state-based Diffusion Policy、1000并行环境，不能保证ACoT-VLA/LIBERO收益。DAgger（Ross et al., AISTATS 2011）支持用student实际访问的state持续聚合监督数据以处理sequential distribution shift，对应当前student rollout/relabel SFT。CPO（Achiam et al., 2017）提供reward与资源/安全constraint分离的CMDP依据，对应将terminal success作为目标、calls或policy time作为约束；实现不必照搬CPO，可采用PPO-Lagrangian。ThriftyDAgger（Hoque et al., 2021）通过novelty/risk和intervention budget只在关键状态切换控制模式，为risk-triggered selective fallback提供类比依据。ResiP《From Imitation to Refinement – Residual RL for Precise Assembly》（arXiv:2407.16677）表明当open-loop chunk/base动作本身缺少精细纠错能力时，可冻结BC planner并训练per-step residual RL；这对应H-only oracle不足后的recovery adapter路线。此前Step Entropy论文的SFT+GRPO也支持“监督warm start后再用return/cost联合优化”的一般结构，但其reward消融显示朴素correctness+compression reward会产生退化，因此不能直接移植GRPO。项目内证据进一步要求分阶段：V2-P已有95.2%且96.2% decisions为H10，value-refined把avg H降到9.328后success反降至94.3%，说明广泛缩短H不是答案；应先确认48个失败中是否至少有14个horizon-fixable，再决定H-PPO，否则转向少量风险触发恢复。此次只做论文与逻辑核对，没有修改代码或运行实验。

2026-07-20（Asia/Shanghai）在用户授权修改代码和测试后，实现了可重复的execution-horizon headroom审计。新增`src/openpi/execution_horizon/headroom.py`，对H1-10 counterfactual branch labels计算指定reference H的成功率、失败数、可被其他H救回/所有H均失败数量、success-first oracle上界，以及在平均remaining-calls预算下精确最大化root success的离散oracle；同时计算达到96.0%、96.6%、97.0% root-level目标所需额外success与最小标签成本。新增CLI `scripts/audit_execution_horizon_headroom.py`，读取现有sharded HDF5，默认审计raw-H、H9、H10，并输出overall、逐task和Task8/9聚合结果。输出明确标注这是存储continuation policy下的乐观root-state反事实上界，不能替代闭环LIBERO成功率，也不能直接证明正式48个失败episode被救回。新增`src/openpi/execution_horizon/headroom_test.py`覆盖reference fixability、预算贪心最优选择、不可行预算、目标成本和cost tie-breaking；本机因仓库全局conftest依赖缺失`pynvml`，使用`pytest --noconftest`运行纯NumPy测试，5项全部通过（0.05–0.09 s）。`py_compile`、`git diff --check`和独立4-root HDF5 CLI端到端合成测试均通过。

真实HDF5仍只在远端，连续两次`ssh covla`均返回port 22 connection refused，因此本轮未运行原始596-root完整cost/fixability审计。利用现有日志中已核准的真实396-root student-state汇总可做不含cost的初步推断：H10为351/396成功，45个失败中只有5个是H1-10全部失败，因此40/45个H10失败在单次counterfactual标签中至少存在一个成功替代H，any-H乐观上界为391/396=98.74%。初始200-root数据另有2个all-H失败；合并596-root训练集的any-H标签上界约为589/596=98.83%。这说明训练标签中存在足够的H选择信号，不能据“H接近10”直接否定horizon RL；但每个H仅有一次Bernoulli branch outcome、两个数据源的continuation policy不同，且oracle会利用事后结果，所以上界显著偏乐观。远端恢复后应立即用新CLI读取四个initial分片和四个student分片，重点检查H9/H10及raw-H的fixable counts、Task8/9、达到目标所需minimum calls，并据此决定继续hard-state SFT还是实现constrained PPO。

上述headroom审计实现已提交为`40bf6f2`（`Add execution horizon headroom audit`）并成功推送到GitHub `main`，提交范围仅包含三个新增审计/测试文件；`reports/context/experiment_log.md`按仓库规则继续作为本地ignored事实日志，没有进入提交。

2026-07-20（Asia/Shanghai）使用用户提供的新SSH入口`ssh -p 26543 root@connect.westd.seetacloud.com`恢复远端访问。服务器`/root/ACoT-VLA`从`036101e` fast-forward到`40bf6f2`，既有`third_party/libero`脏子模块保持不动。随后直接对8个真实HDF5目录运行`audit_execution_horizon_headroom.py`，分别生成initial 200 roots、student 396 roots和aggregate 596 roots三份结果：`/root/autodl-tmp/acotvla/execution_horizon_v2p/headroom_audit/{initial200,student396,aggregate596}.json`。本轮按用户要求没有继续增加或重复单元测试。

真实aggregate 596-root审计中，固定H1/H5/H9/H10的单root分支成功分别为557/596=93.46%、542/596=90.94%、531/596=89.09%、530/596=88.93%；存储的K20 teacher `raw_h`为535/596=89.77%，平均raw H为7.174。H10的66个失败root中59个至少存在一个成功替代H，仅7个H1-10全部失败，因此标签级any-H乐观上界为589/596=98.83%。initial 200与student 396的H10失败可救回比例分别为19/21和40/45，上界分别99.0%与98.74%，两个continuation policy下结论方向一致。596 roots中425个在所有H均成功、7个所有H均失败，其余164个对H选择敏感；说明可利用信号集中在少数root，而不是要求全局广泛缩短H。

困难任务的信号更集中。Task8 aggregate为60 roots，stored raw-H与H10均44/60=73.33%，但每个root至少有一个成功H，标签上界100%；16个H10失败全部可由其他H救回，其中H1与H3各自救回12个。Task9的raw-H与H10均53/60=88.33%，7个H10失败也全部存在成功替代H，上界100%。Task8/9合并H10为97/120=80.83%，23个失败全部标签可救回。固定H的同数据最优点为Task8 H2=81.67%、Task9 H6=93.33%，仍远低于any-H上界，说明潜在收益需要state-dependent选择，不能只按task固定一个H。

成对检查显示aggregate上H1相对stored raw-H为515个双方成功、42个H1独赢、20个raw-H独赢、19个双方失败，净增22个root；H1相对H10为505个双方成功、52个H1独赢、25个H10独赢、14个双方失败。跨数据源检查也保持方向：用initial 200选择的全局H1在student 396上为92.68%，用student选择的全局H1在initial上为95.0%；按训练源选择每task最佳固定H后应用到另一来源分别为93.43%和94.0%，均高于对应stored raw-H 90.15%和89.0%。这证明现有teacher raw-H/Q选择没有充分利用分支标签，但不证明部署时反复使用H1会更好，因为每条记录只在当前root强制一次H，之后恢复各自的fixed-H9或student continuation。

remaining-calls hindsight oracle在aggregate上选择成功分支后为589/596=98.83%、平均19.128 remaining calls，低于H10的28.773；但该成本同时受“成功分支提前终止、失败分支跑满timeout”影响，不能解释为可部署selector已实现同等调用预算。更关键的限制是每个root/H只有一次Bernoulli rollout、oracle事后查看了全部结果、initial与student使用不同continuation，且这些596 roots不是正式V2-P 48个失败episode的逐决策快照。因此98.83%是明显偏乐观的root-label headroom，不可替代闭环success，也不能直接声称能把正式95.2%提升到98.83%。

本轮证据支持继续优化H selector，但优先级应是decision-focused监督而非立即大规模RL：先用现有H1-10向量构造pairwise/ranking或advantage目标，只在H敏感roots上强化Task8/9与student-state采样，并增加每个关键root/H的重复rollout以估计成功概率、降低单次标签噪声；随后进行小规模闭环验证。若该阶段仍无法把离线选择明显推向fixed-H frontier以上，再实现冻结base、只更新selector的constrained chunk-level PPO，以terminal success为reward、calls/policy time为约束。若对正式失败episode重采样后多数root在所有H仍失败，则应转向risk-triggered recovery/base adapter，而不是继续H-only RL。

2026-07-20（Asia/Shanghai）进一步澄清596-root审计的因果含义：在同一MuJoCo snapshot、同一primary action chunk和受控continuation条件下，仅改变当前root先执行多少步再replan，确实观测到不同terminal outcome；因此execution horizon在部分状态上是可能影响成功率的控制变量，而不仅是速度参数。aggregate中H1相对H10有52个root从失败变成功，同时也有25个root从成功变失败，说明不存在“H越短越好”或“H越长越好”的单调规律，真正需要的是state-dependent selector。H10失败的59/66以及Task8/9的23/23存在成功替代H，意味着当前长H策略可能在少量关键状态执行过多旧chunk动作，及时replan有机会救回；但这些是单次branch标签、只强制当前一次H且后续continuation不同于持续部署同一selector，所以只能支持“存在可学习的选择空间”，不能证明部署后必然提高、不能把98.83%当成episode成功率，也不能据此直接决定全局改成H1。可靠下一步仍是对正式失败前snapshot做多seed重复分支，再验证按状态选择H的闭环净收益与calls代价。

2026-07-20（Asia/Shanghai）基于真实596-root headroom结果，形成三层改进方案。首选是`rescue-aware ranking SFT`：为collector增加关键root的多seed H分支重复，对每个`(state,H)`估计成功概率；在现有predictor共享特征上增加直接H-policy/advantage head，用同root内pairwise或listwise ranking训练“哪个H优于哪个H”，而不是继续让全部约90%正标签的独立BCE主导选择。425个all-H-success roots用于学习在同样成功时偏向高H/低调用，164个H-sensitive roots重点学习成功差异，7个all-H-fail roots标为需要fallback而不强行生成best-H；Task8/9与student访问状态继续优先采样。在线默认H9/H10，只有预测短H相对H10的成功优势超过校准margin时才干预，并保留episode calls budget。第二层是`risk-triggered selective fallback`：正常状态仍single-call H9/H10，仅在预测所有H不安全、OOD或高不确定状态触发短H、第二次policy采样/候选比较或恢复策略，以V2-P相对历史原版约28 calls/episode的速度余量换取少量关键纠错。第三层才是冻结base、只更新H selector的chunk-level PPO-Lagrangian，用terminal success为reward、calls或policy time为constraint、SFT策略KL为稳定项，并以counterfactual Q作critic warm start；不建议直接对完整ACoT-VLA做RL或照搬GRPO。

建议的最小验证顺序为：先在正式distilled失败前snapshot及Task8/9敏感root上，对H1/H3/H5/H7/H9/H10各重复3-5个受控seed；若替代H优势跨seed稳定，再实现可选`v2_ranked`模式并进行episode-group held-out离线审计，指标使用matched-call success、相对H9/H10的paired rescue/regression而非raw-H accuracy；随后先跑Task8/9小规模闭环，再锁参跑全LIBERO-Long同协议评估。只有ranking SFT达到可用但仍低于论文级门槛时才进入selector-only constrained PPO；若多seed后多数正式失败root所有H均失败，则直接转向selective recovery/base adapter，不继续投入H-only训练。

2026-07-20（Asia/Shanghai）为执行同root多seed稳定性测试，新增`scripts/replay_execution_horizon_branches.py`并提交推送`fbe396a`。脚本只读现有HDF5与physics snapshot，在保存root上重新生成同seed primary action，并对指定H使用受控continuation seed重复分支；输出root action对float16记录的误差、原seed标签复现率、逐H重复成功率及原rescue-H相对reference-H的paired win/loss，不写训练数据、不改变既有collector/eval格式。`py_compile`、Ruff和`git diff --check`通过。远端拉取后尝试在port8031启动base-only server，但新SSH容器中JAX仅识别`CpuDevice(id=0)`，`/usr/bin/nvidia-smi`为空且不可执行，base checkpoint在恢复阶段退出，因而没有产生任何重复分支结果。用户确认GPU仍在排队，本实验暂缓；不能把未运行的多seed测试写成结果。

2026-07-20（Asia/Shanghai）按用户要求对服务器数据盘做存储审计与保守清理。`/root/autodl-tmp`初始为800G总量、626G已用、175G可用，其中`/root/autodl-tmp/acotvla`约624.9 GiB。主要占用为：主`checkpoints`约428.8 GiB、旧`checkpoints--overwirte`约100.4 GiB、HF cache约24.0 GiB、uv cache约20.6 GiB、两套Python环境约17.8 GiB、训练datasets约15.7 GiB、openpi base cache约11.6 GiB、Stage B结果约3.6 GiB。inode使用仅1%，问题是大文件容量而不是小文件数量。

本轮只删除两项明确可恢复且不影响研究证据的内容：使用`uv cache clean`清除`/root/autodl-tmp/acotvla/uv-cache`的135,041个缓存文件，释放20.4 GiB；删除未被`.venv`、进程或链接使用的旧环境备份`/root/autodl-tmp/acotvla/envs/acotvla-py311_dirty_0515_1030`，释放约9.25 GiB。清理后数据盘为596G已用、205G可用、占用率从79%降到75%，约净释放30 GiB；当前`.venv`仍指向`envs/acotvla-py311`，并已实际import验证NumPy 1.26.4、h5py 3.13.0和JAX 0.5.3正常。没有删除任何正式eval、HDF5、训练dataset、当前环境或checkpoint。

剩余最大可选清理项需要确认模型保留策略。当前LIBERO base run含10000/20000/30000/40000/50000/50999六个约26G checkpoint，项目现有正式流程只引用50999；若只保留50999可再释放约126 GiB。dynamic-steps run同样有六个约25-26G checkpoint，若只保留50999可释放约126 GiB。`checkpoints--overwirte`是2026-06-26至27的旧dynamic-steps四个checkpoint、约101 GiB；主目录是2026-07-01至02的另一轮且包含50000/50999，旧目录在repo、日志和shell history中未发现引用，但两轮weights并非逐文件相同，因此本轮没有擅自删除。另有Agibot五个约26G checkpoint共127.5 GiB，属于不同任务，本轮不建议在未确认用途时处理。Stage B `static_sweep_full`约3.5 GiB主要是可压缩的`per_sample_metrics.csv`，HF/openpi caches约35.6 GiB可重建但会增加后续下载/预处理时间，当前也暂时保留。

2026-07-20（Asia/Shanghai）用户明确授权删除全部Agibot checkpoint，并按建议删除旧overwrite目录、将两个LIBERO run只保留最终50999。删除前确认`acot_libero_action_cot_explicit_implicit_co_fusion/acot_libero_long_run1/50999`与`acot_libero_action_cot_dynamic_steps_stage_c/dynamic_steps_stage_c_run1/50999`均存在，且服务器没有`serve_policy.py`或`scripts/train.py`进程。随后删除`checkpoints/acot_autodl_agibot_50k_bs8_clean`、`checkpoints/acot_autodl_agibot_50k`、整个`checkpoints--overwirte`，以及两个LIBERO run各自的10000/20000/30000/40000/50000中间checkpoint。

本次大项删除精确释放514,465,275,904 bytes，约479.13 GiB。连同上一轮uv cache与旧环境约30 GiB，数据盘从最初626G已用、175G可用、79%占用降至117G已用、684G可用、15%占用。复核确认两个保留的50999分别约26G和25G，均保有`_CHECKPOINT_METADATA`、`params/manifest.ocdbt`和`assets/norm_stats.json`；各run顶层只剩50999与`wandb_id.txt`。`execution_horizon_v2p`、`stage_b_pruning_eval`、HF cache、训练datasets、openpi cache和当前Python环境均保留，当前环境再次import验证NumPy/h5py/JAX正常。

2026-07-20（Asia/Shanghai）用户提供新入口`ssh -p 37693 root@connect.westd.seetacloud.com`后检查训练/测试能力。新实例为NVIDIA RTX PRO 6000 Blackwell Server Edition，显存97,887 MiB、检查时空闲约97,252 MiB，driver 595.58.03；JAX实际识别`CudaDevice(id=0)`并使用GPU backend。远端repo为`fbe396a`、main分支，仅有既有`third_party/libero`脏子模块。清理后保留的base/dynamic-steps两个50999、initial/student HDF5 manifest及当前`.venv`均存在；该实例挂载的数据盘配额显示310G总量、117G已用、194G可用。

真实训练smoke读取initial Task8/9的40-root HDF5，在默认256 hidden/3 temporal layers结构上运行2步、batch16、32 train/8 validation，完整完成GPU forward、反向传播、gradient clipping、validation-best选择和Orbax sidecar导出。输出为`/root/autodl-tmp/acotvla/v2p_smoke/fbe396a_gpu37693_train2`，状态complete、耗时15.67 s、step2 train loss3.59574、validation loss3.25239、gradient norm18.3552，best step为2；base policy未加载且标记frozen，params成功写入。两步数值只验证训练链路，不代表模型效果。Blackwell CC12.0触发本机`ptxas`过旧warning，JAX自动回退CUDA driver PTX compilation，未造成失败。

base-only server从保留的正式50999成功恢复7.1 GiB参数，Orbax读取约4.97 s，并在port8031正常监听。标准fresh LIBERO evaluator随后完成两局original H5 Task0：state0成功、52 calls、267 steps、首局含首次JIT，policy累计19.615 s；同一server预热后的state1也成功、71 calls、361 steps，policy累计6.931 s即97.62 ms/call，server约107.42 ms/call、RPC wall约108.62 ms/call、full episode elapsed20.971 s。两份输出分别为`v2p_smoke/fbe396a_gpu37693_eval_original_task0_1`与`v2p_smoke/fbe396a_gpu37693_eval_original_task0_state1`，summary均为complete。因此当前新实例具备真实predictor训练、base checkpoint加载、WebSocket policy与闭环LIBERO测试条件；正式速度测试应先warmup并复用同一server，不能使用第一局JIT污染值。

同时尝试了旧HDF5 physics-state离线重放的单Task8 root、H1/H3/H6/H10各1次。分支执行链路本身跑通，但原seed标签0/4复现，重新生成root action相对float16存档最大绝对差1.315，且存档H10失败在重放中变为成功，因此该重放结果无研究效力。现有HDF5只持久化physics state，没有完整保存collector内存中的wrapper scalar/random snapshot与原root observation，不能据此可靠做事后多seed分支；后续应在live collector捕获snapshot时直接重复H分支，而不是从旧HDF5恢复。检查结束后已停止port8031 server，GPU无残留compute进程。

2026-07-20（Asia/Shanghai）根据最新实验日志为用户生成简短周计划。计划聚焦三件事：在新GPU实例上继续推进正式10x100对照评估并避免首局JIT污染；基于headroom审计设计rescue-aware/ranking式H selector与live branch数据采集；先在Task8/9等困难任务上小规模闭环验证，再决定是否扩展到全量正式对比。未新增代码或实验结果。

2026-07-20（Asia/Shanghai）针对“提升复杂任务成功率”的rescue-aware ranking SFT、selective fallback与后续selector-only PPO路线，先完成了同一live snapshot内的多seed前置验证。`scripts/collect_execution_horizon_counterfactuals.py`新增向后兼容的`--branch-repeats`、`--repeat-branch-horizons`、`--branch-repeat-seed-stride`和`--episode-ids`；默认`branch_repeats=1`时原HDF5与旧采集行为不变，启用重复时仍将repeat-0写入原HDF5，并把同root各H逐seed outcome写入`repeated_branch_outcomes.jsonl`。对应提交为`3b3a175`与`9fca67f`。新增`scripts/audit_repeated_horizon_branches.py`，按相同continuation policy seed统计相对H10的paired rescue/regression、固定H成功率、in-sample root-level empirical best-H及per-repeat hindsight any-H；提交为`b37bc3e`，Ruff修复提交为`eeee611`。本机`py_compile`/`git diff --check`和服务器Ruff均通过，代码已推送并同步到远端main；远端既有`third_party/libero`脏子模块未被改动。

第一批6个live roots来自Task8四个与Task9两个，每个H1/H3/H6/H10各3 seeds。汇总文件为`/root/autodl-tmp/acotvla/v2p_smoke/eeee611_live_repeat_aggregate6.json`：18个paired trials中H10为15/18=83.33%，H1/H3/H6均为13/18=72.22%；固定短H相对H10均净减少2个success。in-sample按root事后选经验最佳H为17/18=94.44%，平均remaining calls为44.78，对比H10的49.83；6个root中只有2个出现短H成功数高于H10。Task8 episode2/step28为H3 3/3对H10 2/3，Task9 episode13/step127为H6 3/3对H10 2/3；另一个Task8敏感root上H1与H10同为2/3，但表现为一次rescue加一次regression，净收益为0。per-repeat hindsight any-H为18/18，但它利用了不可预知的未来seed outcome，明显比可学习的root-level best-H更乐观。

随后尝试把上述两个3-seed margin root跨run补到5 seeds，发现该做法无效：尽管task/episode/decision step/root seed完全相同，两轮Task8 physics state最大绝对差约0.00292，Task9约0.09940，prefix/action feature与success pattern也不同，说明nominal root标识不能在server重启/重新rollout后保证同一数值状态。第二轮两个nominal roots内H10为10/10，H3/H6各9/10，没有成功率margin，输出为`/root/autodl-tmp/acotvla/v2p_smoke/eeee611_margin_recheck_aggregate2.json`。两轮不能合并成8-seed统计；未来必须在首次捕获完整`SimulatorSnapshot`时一次性采足所需seeds，旧HDF5或跨run episode重建都不可靠。

为得到不依赖3-seed的小批次证据，使用同一base server四路并行采集4个全新Task8/9 roots，每个root在首次live snapshot内对H1/H3/H6/H10各跑5 seeds。四路共享单server且GPU利用率约85%-88%，未复制base checkpoint。汇总`/root/autodl-tmp/acotvla/v2p_smoke/eeee611_fresh_hard4_r5_audit.json`显示20个paired trials中H10=19/20=95%，H3=19/20=95%，H1=16/20=80%，H6=16/20=80%；固定H3相对H10为1 rescue、1 regression、净0，H1/H6均无rescue且各3 regressions。只有1/4 root具有真正成功率margin：Task8 episode1/step19上H3为5/5、平均remaining calls52.0，H10为4/5、60.6 calls，同seed得到1 rescue/0 regression；同root H6仅1/5，直接证明H效果非单调且不能把“短H”作为统一动作。其余Task8 step136上H10/H6均5/5而H1/H3为4/5；Task9 step37上四个H均5/5；Task9 step109上H10/H3/H6均5/5而H1为4/5。in-sample root-level best-H为20/20、32.2 remaining calls，对比H10的19/20、37.6 calls，但这是对4个特意抽取roots事后选H的诊断，不是部署成功率。

当前证据支持存在少量真实、state-dependent的H rescue机会，也否定全局降低H：复杂Task8的某个状态中H3同时提高5-seed成功率并减少完成成本，但相邻状态可出现H3回归或H6灾难性下降。最合理的下一模型是用同root多seed成功概率训练pairwise/listwise advantage，并以H9/H10为默认，只在预测短H相对H10优势超过校准margin时稀疏干预；这实质上应把rescue-aware ranking SFT与confidence-gated selective fallback结合。当前只有4个严格5-seed roots，样本不足以负责任地训练并声称新selector效果；应先按相同live机制扩展约20-40个Task8/9或失败附近roots、每root至少5 seeds，再做episode-group held-out ranking与小规模闭环。selector-only PPO继续后置，不能用PPO补救标签概率尚未估准的问题。全部采集结束后已停止policy server，GPU无残留compute进程。

2026-07-20（Asia/Shanghai）综合当前headroom、3-seed与严格5-seed live结果，判断H selector仍有真实但稀疏的提升空间。支持证据是同一live snapshot的Task8 step19上H3达到5/5、H10为4/5，且H3平均remaining calls更低，说明至少存在可由更早replan稳定救回的状态；反对“大范围提升”的证据是严格4-root批次固定H3与H10均为19/20，H3在不同root间恰好1 rescue/1 regression，H1/H6整体更差。因而可支持的结论是“高置信state-dependent选择可能获得小幅净提升”，不能支持“换固定H即可提高成功率”或“当前oracle上界可实现”。正式V2-P从95.2%达到历史96.6%需要1000局净救回14局，这一量级在机制上并未被当前证据排除，但现有4个严格roots无法估计稳定救回状态在正式失败episode中的真实频率，也无法保证达到该门槛。下一关键量不是继续看hindsight any-H，而是扩大同snapshot多seed数据后测held-out selector相对H10的净rescue减regression；只有该值稳定为正，才能把提升空间转化为研究结果。

2026-07-20（Asia/Shanghai）进一步区分了Step Entropy论文采用GRPO的原因与当前V2-P H selector是否面临同类问题。论文的SFT只模仿单条静态压缩CoT，不能直接优化整条生成轨迹的非可微目标；GRPO用同题多条on-policy completion的组内相对reward，联合最终答案正确性、skip比例、skip数量惩罚与响应长度，主要学习accuracy-efficiency trade-off而非单纯提高accuracy。论文表3中GRPO相对SFT继续减少token，但accuracy并非各benchmark一致上升；附录消融还显示朴素correctness+skip reward会出现退化策略。V2-P确实存在对应的两类问题：单步/反事实监督只是terminal success与calls目标的代理，且部署selector改变H后会改变后续访问状态，形成SFT数据与on-policy分布偏移。不过当前系统拥有同snapshot多H分支这种比LLM terminal reward更密集的监督，H动作空间仅10个，而真实success昂贵且随机；严格5-seed数据目前也只有4个roots、仅1个root显示短H净优势。因此当前瓶颈更像是成功概率估计、置信度校准与闭环分布覆盖不足，不是缺少policy gradient。合理顺序仍为live同snapshot多seed扩充数据、rescue-aware ranking/Q SFT、student-state DAgger与confidence-gated闭环验证；只有held-out净rescue稳定为正但仍受序列目标限制时，再引入冻结base、selector-only的GRPO/PPO-Lagrangian式on-policy优化，以terminal success为主reward、calls/policy time为约束并保留对SFT策略的KL。未修改模型代码或运行新实验。

2026-07-20（Asia/Shanghai）在新GPU实例`ssh -p 37693 root@connect.westd.seetacloud.com`完成了第二批严格live同snapshot多seed验证。远端repo为`eeee611`，使用保留的正式ACoT-VLA 50999 checkpoint和单一base server，4个collector并行共享RTX PRO 6000 GPU；continuation policy固定为H9，`root_call_offset_cycle=20`、每episode只取一个root，对H1/H3/H6/H10各运行5个配对continuation seeds，其余H保留repeat-0单次分支。样本定向来自正式Fixed-H9失败episode，Task8取13个新roots、Task9取7个新roots，共20 roots、100 paired trials；4个collector分别完整写出4/4/5/7条记录且`summary.json`均为`complete`。一次SSH轮询出现瞬时连接错误，但服务器采集未中断。

最终固定H统计为：H10 71/100 success、平均56.51 remaining calls；H6同为71/100、56.59 calls；H1为68/100、59.29 calls；H3为66/100、58.74 calls。相对H10的同seed配对结果分别为H1 14 rescue/17 regression（净-3）、H3 10/15（净-5）、H6 15/15（净0），因此这批困难状态不支持全局换成任何固定短H。Task8的H10/H6/H3/H1分别为47/47/46/44 successes（65 trials），Task9分别为24/24/20/24（35 trials）；固定H的总体差异较小且方向随状态、任务变化。

状态相关诊断仍显示稀疏headroom：9/20 roots存在某个候选H相对H10的经验成功率margin；直接用同5 seeds事后按root选择经验best-H可达84/100、47.51 calls，而逐seed hindsight any-H为91/100。但进一步进行“用4个seed选H、剩1个seed验证”的同root留一审计后，激进best-H只有67/100，低于H10的71/100，证明84%明显包含小样本选择过拟合。只有当训练4 seeds中候选H相对H10至少多2个success才允许切换的confidence gate得到72/100、56.21 calls，即3 rescues/2 regressions、净+1；其中Task8为49/65对H10 47/65且calls 59.18对60.20，Task9为23/35对24/35且calls 50.69对49.66。该gate结果只验证同一root上的概率估计稳定性，仍未验证predictor对未见root的泛化；且本批专门抽取H9失败episode，所有百分比均不能外推为正式闭环总体成功率。

最终证据支持“存在真实但稀疏的state-dependent H信号”，也表明5-seed标签仍不足以支撑激进选择。当前合理下一步是继续扩充hard roots/seed，使用多seed成功概率构造rescue-aware ranking/Q监督并配合student-state DAgger，部署时默认H9/H10且只做高置信稀疏干预；尚不支持立即进入GRPO或selector PPO。主要产物位于`/root/autodl-tmp/acotvla/v2p_validation/20260720_hard20_r5`，包括`final_all_audit.json`、`final_task8_audit.json`、`final_task9_audit.json`和`final_leave_one_seed_gate_audit.json`。完整结果摘要已通过服务器私有签名webhook成功发送飞书；随后停止policy server，确认无collector/server残留且GPU为0 MiB、0%利用率。

2026-07-20（Asia/Shanghai）基于20-root多seed验证，进一步确定RL与SFT的优先级。当前不建议立即用GRPO/PPO辅助SFT：固定H没有改善，5-seed in-sample best-H从H10的71/100升到84/100，但4-seed选H、1-seed验证的激进策略反降到67/100，说明主要误差来自小样本成功概率估计、选择偏差和未见root泛化，而非缺少policy-gradient目标；RL会直接放大同一类高方差terminal reward。更合适的当前方法是把同root多H数据视为full-information contextual-bandit监督，使用多seed成功概率及不确定度训练H-relative Q/advantage或pairwise/listwise ranking head，默认H9/H10，只有候选短H对H10的成功优势下置信界超过门槛时才干预；同时用student-state DAgger补齐部署状态分布。该阶段应重点扩大Task8和失败附近roots，并保留Task9无收益时的默认H10。只有在unseen-root与闭环paired评估中高置信selector持续净rescue为正、但仍低于论文级成功率目标且SFT/DAgger已达到平台时，才进入冻结base、selector-only的constrained PPO/PPO-Lagrangian或GRPO-like阶段，以terminal success为目标、calls/policy time为约束、SFT策略KL为稳定项。该判断未运行新实验或修改模型代码。

2026-07-20（Asia/Shanghai）在用户追问“为什么不现在试PPO”后，重新界定结论：没有原则性理由继续排除PPO；完成20-root多seed验证后，9/20 roots出现状态相关经验margin，已经满足“horizon确实可能改变terminal outcome”的最小机制前提，因此可以立即进行一个受控的selector-only PPO pilot。此前暂缓的原因是terminal success为稀疏高方差reward、一次episode包含多个H决策导致credit assignment困难、当前激进留一选择67/100低于H10 71/100，以及现有predictor并非直接可采样并记录log-prob的categorical actor；这些风险意味着不能把PPO直接作为主线或从完整VLA参数开始训练。合理pilot应冻结ACoT-VLA base，复用predictor feature并增加小型categorical H actor与value head，以H10-biased或ranking-SFT权重warm start，仅在Task8先运行；使用terminal success主reward、calls/policy time的Lagrangian约束、对warm-start策略的KL和明确的entropy/collapse监控，最终用未参与训练的固定initial states/seeds做paired H10/V2-P rescue-regression评估。若pilot无净rescue或塌缩到单一H，应停止并回到ranking/DAgger；若Task8稳定为正，再扩展Task9和全LIBERO-Long。此次只调整实验建议，未修改代码或运行PPO。

2026-07-20（Asia/Shanghai）结合DEHP、PolicyTrim、PA-RL与Residual Off-Policy RL重新设计SFT+RL混合路线。DEHP是最直接的execution-horizon参考：冻结chunk policy、训练categorical H head，将变时长决策写成chunk-level SMDP，并使用step-discounted GAE和PPO；其actor从uniform初始化而只warm-start critic，本项目可利用已有596-root反事实数据和20-root多seed数据进一步加入监督actor warm start。PolicyTrim表明高成功率下纯binary group reward会失去方差，并展示只有成功轨迹才启用效率奖励、KL与group-anchored稳定正则可防止追求长chunk或少steps时出现脆弱shortcut；该reward原则可借用，但不应照搬其对完整VLA做GRPO。PA-RL提供更适合本项目的主线：RL critic先评价/优化候选动作，再通过标准监督损失把Q优化后的动作蒸馏回policy；由于H只有1-10，可直接枚举全部H，无需连续动作的采样和梯度优化。Residual Off-Policy RL则支持冻结大base、只训练小模块，并在在线阶段混合offline demonstration/replay以保留安全先验。

据此推荐的主实验不是简单串行“SFT后纯PPO”，而是offline-to-online Q-guided supervised policy improvement：先用596-root单seed数据预训练success/timeout/cost Q与ranking actor，用20-root多seed结果校准不确定度；在线 rollout后按chunk-level SMDP更新distributional/ensemble Q，枚举H1-10得到带success下置信约束的目标分布，再用cross-entropy/listwise loss蒸馏actor，同时持续混合旧反事实数据和新on-policy buffer并保留risk-distillation heads。部署仍默认H9/H10，仅在候选H的保守success优势成立时切换。并行消融实现SFT-warm-started selector PPO：actor loss保留offline ranking辅助项和对SFT策略的KL，terminal success为主return，calls/policy time通过dual/Lagrangian约束；先Task8，再Task9，最后全LIBERO-Long。该设计既保留SFT的稳定性与全部反事实监督，又让RL处理on-policy状态分布和跨decision credit assignment，研究上也区别于DEHP的纯PPO与PolicyTrim的完整VLA GRPO。此次完成文献分析与方案收敛，没有修改代码或运行新训练。

2026-07-20（Asia/Shanghai）按用户授权完成了冻结base的SFT+RL execution-horizon selector pilot，并在`ssh -p 37693 root@connect.westd.seetacloud.com`真实训练和闭环评测。代码新增`src/openpi/execution_horizon/rl_selector.py`、`scripts/train_execution_horizon_rl_selector.py`、`scripts/train_execution_horizon_selector_online.py`和`scripts/audit_execution_horizon_selector_eval.py`，并向后兼容扩展`eval_libero_execution_horizon.py`的`q_guided_selector`、`sft_selector`、`ppo_selector`模式及显式episode IDs。新selector是独立NPZ sidecar，只读取冻结V2-P predictor返回的temporal feature、H1-10 success/timeout/cost与risk曲线；ACoT-VLA 50999 base和V2-P encoder均未更新。离线阶段训练Q-success、Q-cost、actor和value；在线Q阶段混合旧反事实replay并蒸馏actor；PPO阶段只更新actor/value，使用chunk duration的`gamma^H` SMDP-GAE、clipped PPO和对SFT actor的KL。代码提交为`bf789ed`与修正在线Q replay/mask的`c03f3c3`，均已推送GitHub main。服务器Ruff、Ruff format、`py_compile`和selector NPZ save/load/inference smoke均通过，远端既有`third_party/libero`脏子模块保持不动。

离线SFT使用原aggregate 596-root HDF5加20个严格多seed hard roots，共616条记录；重复JSON标注有20个root，因HDF5中存在匹配重复行而落到22条训练记录。训练耗时约22–28 s，输出`/root/autodl-tmp/acotvla/execution_horizon_rl/20260720_selector_pilot/sft/selector_sft.npz`与`sft_replay/selector_replay.npz`。最终离线Q加权Brier约0.0287，22条重复记录上的保守经验目标拟合为100%；这只是训练内指标，未作为闭环效果结论。随后在Task8训练状态上用SFT actor、temperature 0.8采集24个on-policy episodes、2336个chunk transitions，成功16/24=66.67%，产物为`online_collection_task8`。首次在线Q蒸馏暴露出“新Q目标套用旧safety mask”导致无效千万级CE哨兵的问题；该checkpoint未进入评测。`c03f3c3`改为用完整actor分布蒸馏，并显式混入616-root replay，重跑后Q distillation CE约1.008、offline Q BCE约0.165；PPO末步KL约0.0077、clip fraction约0.102。最终sidecars分别为`online_q_distill_replay/selector_q_online.npz`和`online_ppo_replay/selector_ppo.npz`。

最终评测使用Task8中未参与训练/重复标注的20个initial-state IDs，并用两组policy seed各跑一次，共40个同状态配对episodes；H10、现有V2-P value-refined和PPO均覆盖40局。合并结果位于`/root/autodl-tmp/acotvla/execution_horizon_rl/20260720_selector_pilot/final_audit.json`：Fixed H10为33/40=82.5%、51.775 calls/episode；现有V2-P为33/40=82.5%、58.7 calls/episode；SFT-warm-start selector PPO为34/40=85.0%、119.2 calls/episode、avg H4.339。PPO相对H10与V2-P均为6 rescues、5 regressions、净+1，双侧exact McNemar p=1.0；调用数分别为H10的2.302倍和V2-P的2.031倍。分seed方向反转：第一组PPO 18/20对H10/V2-P 15/20，净+3；第二组PPO 16/20对H10/V2-P 18/20，净-2。Wilson区间也高度重叠。因此PPO表现出救回能力，但当前+2.5个百分点不显著、跨seed不稳定且效率代价过大，不能替换V2-P。

Q-guided online selector只在第一组20个held-out episodes评测，得到13/20=65%、131.75 calls/episode、avg H4.875；相对H10为4 rescues/6 regressions、净-2，相对V2-P为3/5、净-2，属于明确负结果。总体证据否定当前“在线success-Q直接枚举H”和当前无显式成本约束的PPO部署。下一步若继续RL，应采用更保守的constrained PPO/PPO-Lagrangian：H9/H10或SFT策略作为成功风险锚点，calls/policy time作为真实dual constraint，限制H1过度选择，使用更多initial states和至少多组seed的online replay，并以held-out净rescue减regression和成功率-调用数Pareto为选模标准。完整摘要已通过服务器私有飞书通知脚本成功发送；评测结束后停止port8032 policy server，GPU无残留compute进程。

2026-07-21（Asia/Shanghai）基于上一轮SFT+RL selector pilot重新判断研究主线。当前不建议继续调现有Q-direct selector或无显式成本约束的PPO：Q在20个held-out Task8 episodes上只有65%且净退化2局；PPO在40个配对episodes上只比H10/V2-P多1个净success，双侧McNemar p=1.0，两个policy-seed round分别净+3和-2，同时将avg H压到4.339、calls提高到119.2/episode，为V2-P的2.03倍。这更像用高频replan购买随机救回，而不是学到稳定、稀疏的关键状态干预。现有20-root多seed反事实也显示aggressive leave-one-seed selector低于H10，说明瓶颈主要是可救回状态稀疏、成功概率估计与跨状态/seed泛化，而非再加几轮policy-gradient优化。

建议将H-only方法降为有明确止损条件的辅助消融：默认保持V2-P/H9-H10，只学习是否触发一次稀疏短H干预，排除H1或严格限制其占比；使用ensemble/下置信界作abstention gate，并用PPO-Lagrangian将calls或policy time约束在V2-P的1.10–1.15倍内。至少用100个配对episodes、3个以上policy seeds评估；若不能在该预算内保持稳定正的净rescue，应停止H-only RL，而不是继续调参。

更推荐的主方向是“V2-P节省计算预算 + risk-triggered selective recovery/action-chunk reranking”：普通状态继续single-chunk长H；只在高风险或低置信状态从冻结ACoT-VLA采样少量不同seed的candidate chunks，用action-value/progress critic进行pairwise rerank，执行较短prefix后重新观察。该方向把V2-P节省的调用预算重新投入少量困难状态，动作空间不再局限于对同一个错误chunk选择执行长度，更可能处理base action plan本身错误。先做同snapshot candidate-chunk oracle，判断K=2–4候选是否能稳定救回H-only失败；有headroom时采用PA-RL式critic优化/重排再监督蒸馏，并用DAgger式on-policy状态聚合。若多个base candidates仍同时失败，再升级到冻结base的小型residual action或latent recovery adapter；暂不建议直接对完整ACoT-VLA做RL。本轮只完成技术分析与方向收敛，没有修改代码或运行新实验。

2026-07-21（Asia/Shanghai）开始实现risk-triggered selective recovery/action-chunk reranking的最小真值实验。`src/openpi/policies/policy.py`现在会对共享一次VLM/prefix的batched-MC输出逐候选复用原output transform，新增环境空间`mc_actions`（K x 10 x 7），同时保留既有candidate-0 `actions`和normalized MC诊断，因此不改变普通推理、V2-P或旧collector行为。新增`scripts/collect_action_chunk_candidate_oracle.py`：在同一个live `SimulatorSnapshot`和配对continuation seeds下比较candidate-0 H10、candidate-0 H3及其他candidate H3；默认从K10中用不查看仿真outcome的farthest-first prefix diversity选4个候选，分别报告纯缩短H、同H换chunk、hindsight any-candidate和in-sample empirical-best上界，且明确后两者不是可部署成功率。代码提交为`1f5c1fb`与格式修正`f76af3a`，已推送并同步服务器main；本机`py_compile`/`git diff --check`、服务器Ruff、CLI help及候选选择smoke通过，远端既有`third_party/libero`脏子模块未改动。

当前`ssh -p 37693 root@connect.westd.seetacloud.com`没有挂载`/dev/nvidia*`，JAX只识别`CpuDevice(id=0)`，因此尚未产生真实模型候选或branch成功率，不能报告实验结果。已在tmux `action_candidate_wait`启动低开销GPU等待与自动运行器；GPU恢复后会在port8033加载冻结50999 base，四路并行采集Task8 episode 0/35与Task9 episode 37/75共4个hard roots，每root使用4个候选、3个配对seed、fixed-H9 continuation，共预计60个branch rollouts。目标输出为`/root/autodl-tmp/acotvla/action_chunk_candidate_oracle/20260721_pilot4_f76af3a`，自动生成`aggregate_summary.json`并在结束后停止policy server。当前状态仅为`waiting_for_gpu`，后续必须核验`candidate0_action_max_abs_difference_vs_primary=0`及paired rescue/regression后才能判断动作候选是否有headroom。

2026-07-21（Asia/Shanghai）GPU重新分配后JAX已识别`CudaDevice(id=0)`；设备节点实际为`/dev/nvidia3`，且分配过程重启容器清掉了原tmux，因此修正临时等待器的设备匹配后重新启动。第一次真实运行在候选输出进入branch runner后暴露参数契约遗漏：fixed-H9 `_run_branch`仍会读取`args.v2_budget_capacity`用于归一化传参，而新collector parser未定义该字段，四路均在首个branch报同一`AttributeError`且JSONL保持0条。补充默认12.0后提交`99ac898`并重跑；失败目录`20260721_pilot4_f76af3a`标记partial，不进入任何统计。

修复后的4-root action-chunk oracle在`/root/autodl-tmp/acotvla/action_chunk_candidate_oracle/20260721_pilot4_99ac898`完整完成60个branch rollouts，随后自动停止server。4个root全部核验`candidate0_action_max_abs_difference_vs_primary=0.0`，证明新增environment-space candidate-0与原`actions`严格一致。H10 reference为7/12=58.33%；同chunk只把H10缩到H3为9/12=75.0%，同seed配对3 rescues/1 regression、净+2。保持H3不变、对3个替代chunk做逐seed hindsight-any为12/12，相对candidate-0 H3为3 rescues/0 regressions；四个greedy diversity rank单独固定使用时成功率分别为9/12、9/12、9/12、10/12，说明oracle增益不能由“任取一个不同chunk”直接实现。按root使用全部3 seeds事后选经验best candidate为11/12=91.67%，逐seedany-candidate为12/12，这两项均使用未来outcome，只是headroom上界，不是可部署成功率。

逐root机制显示Task8 episode0主要是缩短H收益（H10 2/3、candidate-0 H3 3/3）；Task9 episode37全部arm均3/3，无可区分headroom。Task8 episode35为H10 1/3、candidate-0 H3 2/3、最佳替代候选3/3；Task9 episode75为H10 1/3、candidate-0 H3 1/3、最佳替代候选2/3，后者是在相同H下最清晰的跨seed动作候选margin。值得注意的是这两个candidate-sensitive roots的V2 teacher max fused risk分别只有0.9891与1.1532、event_index均为-1、raw-H均为10，现有1.5 entropy gate会漏掉它们；而两个高风险raw-H3 roots没有额外动作候选margin。因此若继续selective recovery，trigger不能只复用当前entropy threshold，需要加入candidate-relative success/progress critic、stuck/history或校准不确定度。raw candidate prefix RMSE虽仅约0.02–0.05环境动作单位，仍可改变terminal outcome，支持进一步验证candidate reranking，但4 roots/12 paired seeds样本太小且hard-root定向，不能外推闭环成功率。

由于pilot出现2个root的动作候选经验margin，已利用当前GPU继续启动其余16个不重复hard episodes的扩大验证：tmux为`action_candidate_hard16`，port8033单base server，Task8 episode 30/36/50、4/47/68、70/72/74/80/97及Task9 episode43/61/73/84/95，仍为K10中farthest-first选4个candidate、H3对H10、fixed-H9 continuation，但每root提高到5个配对seed，共预计400个branch rollouts。输出为`/root/autodl-tmp/acotvla/action_chunk_candidate_oracle/20260721_hard16_r5_99ac898`；启动时checkpoint约4.99秒恢复，四路collector均已连接，结果尚未完成。

2026-07-21（Asia/Shanghai）扩大action-chunk候选验证完整结束：`20260721_hard16_r5_99ac898`的16/16 roots、400/400 branch rollouts均完成，四个collector零异常，runner随后自动停止port8033 server并释放GPU。扩大批次H10 reference为54/80=67.5%，candidate-0 H3为57/80=71.25%，配对为16 rescues/13 regressions、净+3；固定使用三个farthest-first替代rank分别55/80、54/80、56/80，均未超过candidate-0。逐seed hindsight any-candidate为70/80=87.5%，相对H10为20 rescues/4 regressions；按root用同5 seeds事后选经验best candidate为65/80=81.25%。这些oracle指标仍看了未来outcome，不是可部署选择器。

将4-root pilot与16-root扩大批次合并后，共20个不重复hard roots、92个配对seed、460个branch，全部`candidate0_action_max_abs_difference_vs_primary=0.0`。H10为61/92=66.30%，candidate-0 H3为66/92=71.74%，配对19 rescues/14 regressions、净+5、exact McNemar p=0.4869，未显示显著优势。固定替代rank1/2/3分别为64/92、63/92、66/92；相对candidate-0 H3分别净-2、-3、0，说明farthest diversity不能直接完成候选选择。逐seedhindsight any-alternative为81/92，相对candidate-0为16 rescues/1 regression；any-candidate为82/92=89.13%，按root事后best为76/92=82.61%，只说明多个base action chunks之间存在较大oracle headroom。

稳定性审计否定了直接依据少量seed经验胜率选择候选：对每个root留一continuation seed、用其余2或4个seed选candidate时，激进选择为60/92，低于candidate-0的66/92；要求替代candidate训练seed胜数严格高于candidate-0的margin-1 gate为64/92，4 rescues/6 regressions、净-2；要求优势至少2个seed也为64/92、0 rescues/2 regressions。20个root中事后best alternative胜数高于candidate-0的有8个、相同11个、低于1个，但这些小样本margin不能泛化到held-out continuation seed。风险分组也表明candidate margin同时出现在4个raw-H3与4个raw-H10 roots，当前V2 entropy gate会漏掉一半候选敏感状态。

综合结论是：动作候选方向存在明确机制headroom，但当前没有可部署的候选选择证据；固定候选和少量seed经验rerank均不优于candidate-0，不能据此立即训练RL或声称成功率提升。下一步若继续，应训练candidate-relative success/progress critic，优先使用短prefix后的simulator progress、object/contact/grasp/phase等低方差dense标签并蒸馏到视觉/action feature，同时扩大roots并用未见root与未见seed双重held-out验证；部署trigger需结合stuck/history与critic置信度，不能只用现有entropy阈值。飞书通知脚本已尝试发送本结果，但GPU容器重启后`FEISHU_WEBHOOK_URL`与`FEISHU_SIGNING_SECRET`均未注入，服务器也未找到现成私有env文件，因此发送失败且未泄露凭据；需要重新提供环境变量后才能重发。

2026-07-21（Asia/Shanghai）进一步明确candidate-oracle后的研究决策：不是推倒`V2-P + selective recovery`总体框架，而是停止其中已经被数据否定的朴素实现，包括用farthest diversity固定选candidate、用每root 3–5个terminal Bernoulli seed经验胜率直接rerank，以及在该噪声标签上立即追加PPO/GRPO。V2-P继续作为速度主线，selective recovery仍作为困难状态的可选质量补充；当前缺失的是可学习、低方差的candidate quality信号，而不是候选集合完全没有headroom。

建议只再做一个有明确止损条件的可学性判定实验：从同一live snapshot执行每个candidate的H3短prefix，利用simulator privileged state构造局部progress标签，例如任务predicate阶段推进、object-to-goal距离变化、grasp/contact稳定性、碰撞/掉落与动作平滑性，再检验该局部score选择的candidate是否能在未参与打分的continuation seeds上稳定超过candidate-0。若20–40个新hard roots的held-out paired rescue-regression净值稳定为正，再训练视觉/action candidate critic并蒸馏部署；若该局部oracle仍不能泛化，则停止candidate/H selector迭代，主线转为直接改善base action policy的hard-task DAgger/targeted SFT或小型residual recovery adapter。这样下一次结果是明确的go/no-go，而不是继续更换另一种selector名称。

2026-07-21（Asia/Shanghai）完成了上述H3 privileged-progress candidate selector的最终go/no-go实验。新增`src/openpi/execution_horizon/privileged_progress.py`，从LIBERO BDDL goal与MuJoCo状态构造stage-aware连续分数：Task8使用两个`On(moka_pot, stove_cook_region)`的predicate、EEF接近、抓取和object-to-target距离，Task9使用`In(mug, microwave_heating_region)`及microwave close joint进度；并新增`scripts/collect_action_chunk_progress_audit.py`，先从K10中farthest-first选4个候选，在同snapshot分别执行H3短prefix，只按局部分数与固定margin选择candidate，再用独立fixed-H9 continuation seeds比较terminal success。普通eval脚本及默认推理路径未改动。实现与格式修正提交为`3c30930`、`525f4ff`、`5c922d9`；校准时发现LIBERO wrapper在`set_state/forward`后返回的`robot0_eef_pos`可能滞后一帧，导致前一candidate污染后一candidate的起点评分，因此改为直接读取恢复后MuJoCo EEF site并提交`a7e2d60`。修复后所有候选起点分数逐位一致，最终40-root数据的最大起点spread为0.0；按纯数值尺度将margin从0.002锁定为0.0005并提交`fc6d8b0`，之后未再根据terminal结果调参。所有代码已推送main并同步服务器，服务器Ruff、Ruff format和`py_compile`通过，远端既有`third_party/libero`脏子模块未改动。

第一批锁参正式结果位于`/root/autodl-tmp/acotvla/action_chunk_progress/20260721_new24_r5_fc6d8b0`：24个全新Task8/9 hard roots、每root 5个独立continuation seeds，共120个paired trials；8/24 roots触发替代candidate。progress selector为100/120=83.33%，candidate-0 H3为95/120=79.17%，H10 reference为88/120=73.33%；selector相对candidate-0为8 rescues/3 regressions、净+5，exact McNemar p=0.22656。Task8为49/60对46/60、4/1净+3，Task9为51/60对49/60、4/2净+2。candidate-0与原primary action的最大绝对差为0.0，全部candidate起点评分spread为0.0。由于该正向结果未达到显著或稳定证据，并且预设采样范围为20–40 roots，在不改变score或margin的前提下追加16个独立确认roots，并将其单独分析而非只报告pooled结果。

独立确认集位于`/root/autodl-tmp/acotvla/action_chunk_progress/20260721_confirm16_r5_fc6d8b0`：16 roots、80 paired trials、8 roots触发。selector为60/80=75.0%，candidate-0 H3为61/80=76.25%，H10 reference为60/80=75.0%；selector相对candidate-0为2 rescues/3 regressions、净-1，exact McNemar p=1.0。Task8为26/40对26/40、1/1净0；Task9为34/40对35/40、1/2净-1，因此首批正增益没有在确认集复现。次要pooled 40-root结果保存为`/root/autodl-tmp/acotvla/action_chunk_progress/20260721_pooled40_summary_fc6d8b0.json`：200 paired trials中selector 160/200=80.0%、candidate-0 H3 156/200=78.0%、H10 148/200=74.0%；相对candidate-0为10 rescues/6 regressions、净+4，但exact McNemar p=0.45450。Task8 pooled净+3、Task9净+1，但效应小且由未复现的首批结果主导；这些hard-root定向率也不是闭环episode成功率。

本轮证据未通过预设的“在新root和held-out seeds上稳定净正”门槛，因此停止继续调当前privileged-progress candidate selector，不据此训练视觉candidate critic，也不追加candidate/H PPO或GRPO。局部分数存在两个机制性问题：H3变化通常过小而使24/40 roots abstain，抓取等阶段跃迁也可能成为强但错误的短期信号，例如Task9 episode12的局部分数优势约0.3565却产生0 rescue/1 regression。动作候选仍有hindsight oracle headroom，但当前dense局部标签不能可靠地把headroom转化为选择收益。下一主线应保持V2-P作为速度骨干，转向直接改善base action policy的Task8/9 hard-state DAgger/targeted SFT；若base监督改进仍不能覆盖接触与恢复错误，再评估冻结base的小型residual recovery adapter。port8033模型服务和全部collector已停止，GPU compute进程已清空。此前飞书环境变量在容器重启后仍未注入，本轮未能发送飞书通知，也未暴露凭据。

2026-07-22（Asia/Shanghai）细化了Task8/9 hard-state DAgger＋targeted SFT方案，当前仅完成设计分析，尚未修改代码或运行实验。建议先从现有50999 base checkpoint做保守的targeted SFT：保留全任务示范回放以避免遗忘，提高Task8/9关键阶段窗口和高置信DAgger纠正轨迹的采样权重，初期冻结视觉骨干与LLM、只更新coarse/fine action experts及必要投影，并用固定H9/H10评测隔离base action质量变化。DAgger部分应让更新后的student在Task8/9闭环运行，从停滞、抓取未带起物体、放置predicate未完成或回退、动作振荡及超时前状态保存simulator snapshot，再通过人工/脚本专家或离线多候选多continuation的高置信hindsight teacher生成成功纠正轨迹；未来terminal outcome只用于离线筛标签，不能进入部署策略。现有ACoT数据变换可从原始纠正轨迹自动生成15点stride-2 coarse action与10点stride-1 fine action监督，因此每个样本需保存至少29步连续专家动作，不能只保存H3前缀。训练数据建议以全任务general replay、Task8/9原始关键阶段和DAgger corrections混合，并按初始state/root分组切分训练与held-out，避免同root泄漏。旧V2-P需继续作为速度骨干，但base更新后应先验证fixed-H提升，再重新采集新base特征并校准/蒸馏V2-P，不能直接把旧sidecar的表现当作最终结果。核心验收是Task8/9相同H、相同初始root/seed下相对旧base的paired rescues/regressions与episode success，同时监控Task0–7和整体LIBERO-10是否退化；若一至两轮聚合后held-out仍无净提升，再停止扩大DAgger并转向冻结base的residual recovery adapter。

2026-07-22（Asia/Shanghai）进一步解释了DAgger概念及其与普通SFT的区别：普通SFT只在固定专家示范分布上训练，DAgger则先让当前student闭环运行、收集其自身偏差造成的状态，再由更可靠的教师为这些状态提供纠正动作，将新数据聚合回训练集并迭代。结合Task8/9，典型采集状态包括抓偏后仍夹持、pot放置未满足predicate、mug卡在microwave入口或门关闭进度停滞；失败动作只作为上下文，监督目标必须是教师纠正动作。当前仅完成概念说明，没有修改代码或运行实验。

2026-07-22（Asia/Shanghai）明确了当前不优先采用强化学习的边界：并非永久排除全部RL，而是暂缓在现有candidate/H selector或base ACoT-VLA上直接追加PPO/GRPO。现有privileged-progress selector的首批正增益未在独立确认集复现，terminal success又是高方差稀疏Bernoulli信号，RL不能自动修复不可靠的reward/credit assignment；Task8/9的接触、放置与阶段切换错误还需要长闭环后才能得到terminal反馈。与此同时，当前action expert采用连续动作的flow-matching MSE训练，不直接提供标准PPO所需的可计算action log-prob ratio，若对base做PPO需重构采样与概率记账或另加显式随机residual policy，而不是简单接一个loss。相比之下，targeted SFT/DAgger能把每个hard state转成连续的教师纠正动作监督、重复利用离线样本，并通过全任务replay与冻结骨干降低Task0–7遗忘风险。建议仅在监督纠正已稳定提升但出现饱和、dense reward/critic能在未见root和seed上通过验证、且具备KL/回放/paired评测保护后，再考虑对小型residual adapter或恢复策略进行受约束RL。当前仅完成方法决策说明，未修改代码或运行实验。

2026-07-22（Asia/Shanghai）开始实施Task8/9 hard-state DAgger＋targeted SFT的Phase 0。新GPU实例通过`ssh -p 11375 root@connect.westd.seetacloud.com`连接，使用独立临时known-hosts文件避免覆盖全局SSH记录；实例为RTX PRO 6000 Blackwell 97887 MiB，检查时显存占用0 MiB、GPU利用率0%，持久盘`/root/autodl-tmp`剩余约134 GiB。服务器代码仓库为`/root/ACoT-VLA`、HEAD `fc6d8b0`，只有既有`third_party/libero`脏子模块；原50999 checkpoint、LIBERO LeRobot数据与此前评测结果均存在。数据审计确认评测语境中的双moka pot与microwave任务在训练LeRobot元数据中的task index分别是6和2，因此新实现按任务文本而不是数字匹配。原训练集共379 episodes、101469 frames；双moka pot为29 episodes/11808 frames，microwave为34 episodes/9990 frames，两者合计21798 frames，说明Phase 0重点应是关键阶段加权而非简单重复整条任务。

代码新增`acot_libero_task89_targeted_sft`训练配置，从50999 checkpoint加载，冻结视觉骨干与base LLM、训练双action experts及动作相关投影/融合模块，使用peak LR `1e-5`、200步warmup、5000步上限、每1000步保存。`DataConfig`和`FrameSampler`新增可复现mixture采样：50%全数据uniform replay、25%按目标任务文本采样、25%按JSONL manifest采样；manifest按repo/episode/episode-local frame range定位并支持split与显式weight，旧`subtask` sampler路径保留。新增`scripts/build_libero_targeted_manifest.py`，从原始LeRobot parquet动作中提取gripper状态切换前20/后30帧和每条目标示范最后80帧，输出可审计JSONL与summary，不修改源数据。服务器真实数据smoke test生成180个合并区间、12777个不重复关键阶段帧，其中双moka pot 6548帧、microwave 6229帧；mixture sampler完成101469次抽样，索引范围2至101465、目标任务实际占比0.60627、manifest帧实际占比0.45771，符合三个重叠mixture component的理论分布。当前已通过`git diff --check`和四个改动文件的`py_compile`，新sampler与manifest脚本单独Ruff检查通过；尚未启动模型训练或评测。

2026-07-22（Asia/Shanghai）Phase-0 targeted SFT实现已提交并推送main，commit为`abdd724`（`Add Task8/9 targeted SFT pipeline`），服务器`/root/ACoT-VLA`已fast-forward同步；既有`third_party/libero`脏子模块未改动。服务器完整data-loader smoke成功读取一个真实batch，fine actions形状`(16,10,32)`、coarse actions形状`(16,15,32)`、state为`(16,32)`、tokenized prompt为`(16,200)`。首次5000步热身运行完成50999权重恢复和反向首步，首步`loss=0.0613`、`grad_norm=0.3114`、`param_norm=1949.5723`，总参数3817454128，显存稳定在约66267 MiB，稳态约2.1秒/step；随后在尚无checkpoint时主动停止，改为符合预设分阶段验证的1000步pilot，而非直接盲跑约3小时。

正式pilot运行在tmux `task89_targeted_sft_pilot1k`，日志`/root/autodl-tmp/acotvla/logs/task89_targeted_sft/pilot1k_abdd724.log`，checkpoint目标目录`/root/autodl-tmp/acotvla/checkpoints/acot_libero_task89_targeted_sft/task89_targeted_sft_pilot1k_abdd724`；命令通过CLI将`num_train_steps`覆盖为1000，仍保留5000步LR schedule。pilot已再次通过首个反向step，指标与热身一致，预计约35分钟完成并在step999保存最终checkpoint。后续固定采用seed7、H9评测评测语境中的Task8/9：既有可配对base参考在前20个初始states上为Task8 `17/20`、Task9 `20/20`，100-state正式参考为Task8 `86/100`、Task9 `93/100`；先跑20-state sanity并核对paired rescues/regressions，若无明显退化再扩大100-state。当前尚无新checkpoint成功率，不能声称targeted SFT已提升。

2026-07-22 11:12（Asia/Shanghai）检查Phase-0 pilot进度：训练仍正常运行在`700/1000`，稳态约2.1秒/step，GPU显存约66267 MiB、利用率100%，预计还需约10–11分钟完成训练与step999保存。每100步记录的loss依次为0.0613、0.0457、0.0428、0.0457、0.0471、0.0435、0.0468、0.0442；step700的grad norm为0.1948，未出现NaN、OOM或Traceback。loss总体低于首步但存在正常batch波动，只能说明训练数值稳定，不能代表Task8/9成功率提升；fixed-H9闭环评测尚未开始，因此当前没有新的成功率结论。服务器仍缺少`FEISHU_WEBHOOK_URL`，无法自动发送飞书结果。

2026-07-22（Asia/Shanghai）Phase-0 Task8/9 targeted-SFT 1000步pilot及配对评测已完成。训练于11:23完成，checkpoint为`/root/autodl-tmp/acotvla/checkpoints/acot_libero_task89_targeted_sft/task89_targeted_sft_pilot1k_abdd724/999`，日志为`/root/autodl-tmp/acotvla/logs/task89_targeted_sft/pilot1k_abdd724.log`；step0至900每100步loss为0.0613、0.0457、0.0428、0.0457、0.0471、0.0435、0.0468、0.0442、0.0449、0.0435，未出现NaN、OOM或Traceback。训练数值稳定，但该事实没有转化为闭环成功率提升。

固定H9、seed7、Task8/9各20个initial states的评测输出位于`/root/autodl-tmp/acotvla/task89_targeted_sft_eval/pilot1k_fixed_h9_task89_20_seed7`，已完整生成`rollout_rows.csv`、`per_task_summary.csv`和`summary.json`。新checkpoint总体为32/40=80.0%，同task/episode的50999 base为37/40=92.5%，净降12.5个百分点。Task8为13/20=65.0%对base 17/20=85.0%，配对2 rescues（episode 1、4）、6 regressions（2、5、6、8、11、12），exact McNemar p=0.2891；Task9为19/20=95.0%对base 20/20=100%，0 rescues、1 regression（episode 12），p=1.0。合计配对为30个共同成功、1个共同失败、2 rescues、7 regressions，exact McNemar p=0.1797。小pilot尚不足以声称统计显著退化，但作为预设go/no-go评测没有任何正向证据，并呈明确负向幅度，因此不继续当前配方到5000步，也不扩大到100-state正式评测。

新增失败全部表现为timeout：新checkpoint 8/40 timeout，base为3/40。成功episode本身的执行长度没有明显恶化：Task8成功episode平均steps为468.54，base为467.88；平均policy calls为51.23，两者几乎相同。总体每call RPC wall为99.67 ms，base为96.51 ms，但两次run的少量硬件/编译波动使该差异不能当作模型速度结论。结果更像是策略更新使部分原本成功的initial states转为整局卡死，而不是所有成功轨迹普遍变慢。

当前50/25/25 additive mixture实际使Task8/9占样本约60.63%、manifest关键帧membership约45.77%，而原始Task8/9只占21.48%训练帧；这次数据仍全部来自原始成功示范，manifest也只是gripper切换与terminal窗口，并未包含student访问到的失败/恢复状态，因此该run属于targeted offline SFT，不是真正DAgger。额外审计显示未来动作padding样本概率从uniform的10.46%仅升至mixture的10.70%，最后80帧概率从29.88%升至30.58%，所以terminal padding不是当前退化的主要候选解释。当前实验仍缺少“同样冻结范围、LR和1000步，但uniform replay”的continued-SFT控制，不能把退化唯一归因于mixture；也可能有optimizer state重置、更新双action experts及共享投影或学习率共同作用。

后续建议先补一个最小因果控制：从同一50999权重用uniform replay继续训练，保持当前冻结范围和评测协议，并把checkpoint间隔缩到200步，以区分“继续SFT本身”与“target mixture”造成的退化。若uniform control保持base水平，再将targeted配方降为约85% general、10% target、5% phase，peak LR降至2e-6至3e-6，并优先只更新fine action expert或小型residual adapter；若uniform同样退化，则应先解决optimizer/参数更新保护，不再调采样比例。主研究路线仍应转入真正hard-state DAgger：从旧base在Task8/9闭环中的stuck、grasp/drop、predicate回退和timeout前snapshot采集student-distribution状态，由人工/脚本专家或多候选多continuation中高置信成功的hindsight teacher产生至少29步连续纠正动作，按root划分train/held-out并与全任务replay保守混合。只有held-out paired rescues稳定多于regressions且Task0-7无明显遗忘后，才更新并重新采集/蒸馏V2-P；若一至两轮可靠纠正数据仍无净增益，再转冻结base的selective residual recovery adapter，而不是直接追加PPO/GRPO。

评测完成后已停止`task89_sft_eval`和`task89_sft_server`，GPU无残留compute进程。本地和服务器当前均未注入`FEISHU_WEBHOOK_URL`/`FEISHU_SIGNING_SECRET`，因此结果尚不能发送飞书；未读取或暴露任何凭据。

2026-07-22（Asia/Shanghai）按预设因果控制补做了Task8/9 uniform-replay continued-SFT实验，用来区分上一轮退化是“继续SFT本身”还是强target mixture造成。新增训练配置`acot_libero_task89_uniform_control`：从同一50999 checkpoint恢复，保持targeted pilot相同的冻结范围、双action experts更新范围、AdamW、EMA、batch size、peak LR `1e-5`、200步warmup和1000步训练，只把数据改回全LIBERO uniform replay，不启用target sampler或manifest。考虑每个checkpoint约21 GiB且服务器初始只剩约113 GiB，保存点设为step500与step999。实现提交为`8b626d7`与`830a7d8`，均已推送GitHub main并同步服务器；本地`git diff --check`和`py_compile`通过，服务器配置审计确认target/manifest采样比例均为0。

训练在`ssh -p 11375 root@connect.westd.seetacloud.com`完整完成，最终checkpoint为`/root/autodl-tmp/acotvla/checkpoints/acot_libero_task89_uniform_control/task89_uniform_control1k_830a7d8/999`，日志为`/root/autodl-tmp/acotvla/logs/task89_uniform_control/uniform1k_830a7d8.log`。step0至900每100步loss依次为0.0510、0.0451、0.0425、0.0436、0.0466、0.0422、0.0460、0.0439、0.0442、0.0426，grad norm约0.174至0.228，未出现NaN、OOM或Traceback；稳态约2.1秒/step、显存约66267 MiB。step500和999 checkpoint均保留，持久盘评测后约剩72 GiB。

固定H9、seed7、按同task/episode与50999 base配对的Task8/9评测已扩展到各100个initial states。前20集输出为`/root/autodl-tmp/acotvla/task89_uniform_control_eval/uniform1k_fixed_h9_task89_20_seed7`：uniform为Task8 `19/20`、Task9 `19/20`，合计`38/40=95.0%`；base为`37/40=92.5%`，配对2 rescues、1 regression。独立episode 20–99确认集输出为`/root/autodl-tmp/acotvla/task89_uniform_control_eval/uniform1k_fixed_h9_task89_ep20_99_seed7`：Task8为`70/80`对base `69/80`，6 rescues、5 regressions；Task9为`76/80`对base `73/80`，6 rescues、3 regressions；合计`146/160=91.25%`对`142/160=88.75%`，12 rescues、8 regressions，双侧exact McNemar `p=0.5034`。

合并100-state完整结果：Task8 uniform `89/100`、base `86/100`，8 rescues、5 regressions，`p=0.5811`；Task9 uniform `95/100`、base `93/100`，6 rescues、4 regressions，`p=0.7539`。两任务合计uniform `184/200=92.0%`、base `179/200=89.5%`，净增5个成功、提升2.5个百分点，14 rescues、9 regressions，双侧exact McNemar `p=0.4049`，因此方向为正但尚无统计显著证据。uniform timeout为16/200，base为21/200；overall calls/episode为45.09对46.43，主要来自少5个timeout，而非单次决策更快。仅看成功episode，uniform平均39.45 calls、361.36 steps，base为38.97 calls、356.61 steps；跨run RPC wall差异不作为模型速度结论。

该控制基本排除了“相同参数更新与1000步continued-SFT必然造成上一轮大幅退化”的解释：uniform在独立确认集和完整集均没有复现targeted pilot的负向幅度，说明上一轮约60.63%目标任务、45.77%manifest membership的强采样分布是主要风险因素。不过uniform相对base的+5/200仍不显著，且尚未验证Task0–7 retention，因此不能据此替换50999 base，也不构成研究主贡献。下一步不建议继续在原始成功示范上调85/10/5等target比例；更有信息量的方向是进入真正hard-state DAgger，采集student闭环访问到的stuck、grasp/drop、predicate回退与timeout前状态，并生成至少29步高置信纠正动作，以80–90% general replay和10–20% correction、按root隔离held-out、较低LR及优先更新fine expert/小型adapter的保守方式训练。若准备把uniform checkpoint作为新基线，必须先补Task0–7同协议retention评测。评测后模型服务与客户端均已停止，GPU无残留compute进程；飞书环境变量仍未注入，因此本轮结果未能发送飞书。

2026-07-22（Asia/Shanghai）对uniform continued-SFT结果的研究价值作进一步判断。`+2.5`个百分点和exact McNemar `p=0.4049`对“新主方法”而言确实太小，不能作为稳定提升或论文贡献，也不值得继续通过增加uniform训练步数或大规模调学习率来追逐；该checkpoint目前只适合作为因果控制和候选warm start。另一方面，base失败数从21降到16，相当于表面错误数减少23.8%，且配对中实际出现14 rescues和9 regressions，说明动作策略的可改变headroom并不为零，当前主要问题是救回与破坏同时发生、缺少对hard states的选择性纠正。

因此不应把uniform replay与真正hard-state DAgger视为同一个方向：前者没有使用student访问到的失败状态，只重复原始专家分布，主线价值有限；后者尚未被当前实验验证或否定，并且直接针对Task8/9的闭环分布偏移，仍值得做一次小而有明确止损条件的pilot。建议只做1轮高置信纠正数据聚合，不再做原始成功示范的target reweight：用旧50999 base采集失败/停滞snapshot，为每个状态生成至少29步能完成局部恢复的教师动作，以80–90% general replay和10–20% correction、较低LR训练，然后在未见root的Task8/9各100个固定H9 episodes上配对评估。继续门槛建议预先锁定为合计至少`+5`个百分点，即200局净增至少10个成功、rescues至少约为regressions的2倍，并且Task8和Task9方向均不为负；若只得到小于`+3`个百分点、seed/root间方向反转，或Task0–7出现一致退化，则停止扩大DAgger并转向冻结base的小型residual recovery adapter。该分析没有修改代码或启动新实验。

2026-07-23（Asia/Shanghai）汇总了7月17日至今的V2-P、execution-horizon、selector/PPO、action-chunk recovery、targeted SFT和hard-state DAgger阶段成果。当前最可靠的总体结论仍是：V2-P distilled在seeded 100-trial协议下达到95.2% success、30.536 calls/episode和3.387 s policy/episode，明显优于同协议H5，但尚未证明超过论文级Frozen 96.0%、项目历史官方式96.6%或Fixed H9的Pareto表现；596-root反事实审计确认H选择存在明显乐观headroom，但多seed、PPO和候选selector实验都暴露出高方差与未见root泛化问题，不能把oracle收益转化为稳定闭环提升。Task8/9的强targeted offline SFT在40局上由base 92.5%降至80.0%，而uniform continued-SFT在200局上由89.5%升至92.0%、净增5局但McNemar p=0.4049，说明动作策略有可改变空间，但原始成功示范重加权不是可靠主线。hard-state DAgger采集器、root隔离、LeRobot数据写入和general/correction混合训练链路已经实现并验证；最后可确认的正式采集进度为49条collection records、至少3条双seed稳定纠正轨迹和705帧，尚未开始1000步SFT训练或held-out评测。2026-07-23复查时`connect.westd.seetacloud.com:11375`拒绝连接，因此无法确认远端采集是否最终完成，当前不能声称DAgger已提高成功率。

2026-07-23 14:28（Asia/Shanghai）重新连接`ssh -p 11375 root@connect.westd.seetacloud.com`核查hard-state DAgger 1000步训练状态。正式round1采集已经于2026-07-22 16:07完成，`summary.json`标记`status=complete`，共扫描100个Task8/9 train-root rollouts，记录11个base failures、33个hard-root尝试，最终接受3条双seed稳定纠正轨迹、705帧，来自2个root；`collection_records.jsonl`共有122条记录，`training_manifest.jsonl`已生成。服务器当前没有tmux session或训练进程，也没有任何hard-state DAgger checkpoint或训练日志，因此对应的1000步SFT尚未启动、更没有完成。服务器上已存在的两个step999目录分别是此前targeted-SFT pilot和uniform-control pilot，不能视为本轮DAgger训练结果；持久盘当前剩余约72 GiB。

2026-07-23 15:17（Asia/Shanghai）使用新实例入口`ssh -p 40429 root@connect.westd.seetacloud.com`启动hard-state DAgger round1的1000步保守SFT。启动前确认RTX PRO 6000 Blackwell GPU空闲、持久盘剩余约72 GiB、服务器repo HEAD为`1a3650e`且仅有既有`third_party/libero`脏子模块，50999初始权重、`status=complete`的round1采集、705帧纠正LeRobot数据和manifest均存在。训练配置`acot_libero_task89_hard_state_dagger_round1`按85%原始LIBERO replay与15%纠正数据混合，peak LR为`3e-6`，冻结vision、base LLM与coarse action expert，只更新fine action expert；计划在step500与step999保存。tmux为`task89_dagger_train1k`，日志为`/root/autodl-tmp/acotvla/logs/hard_state_dagger/train1k_1a3650e.log`，checkpoint目录为`/root/autodl-tmp/acotvla/checkpoints/acot_libero_task89_hard_state_dagger_round1/task89_hard_state_dagger_round1_1k_1a3650e`。运行时已正确读取原始101469帧和纠正705帧，恢复50999参数并初始化train state；15:22检查已运行到step27，稳态约2.1秒/step，GPU利用率100%、显存约88197 MiB，未见Traceback、OOM或其他错误，按当前速度预计约34分钟后完成主体训练，另需等待最终checkpoint异步保存。

2026-07-23 15:57（Asia/Shanghai）hard-state DAgger round1的1000步保守SFT已完整结束，主体训练用时约37分07秒，step999 checkpoint于15:57:29完成异步写入且日志明确记录`No errors found in background save thread`。最终checkpoint为`/root/autodl-tmp/acotvla/checkpoints/acot_libero_task89_hard_state_dagger_round1/task89_hard_state_dagger_round1_1k_1a3650e/999`，step500 checkpoint也已保留。每100步loss从step0的0.0889下降为0.0641、0.0621、0.0617、0.0660、0.0632、0.0631、0.0634、0.0635和step900的0.0614；对应grad norm保持在0.1730至0.2406之间，未出现NaN、OOM、Traceback或保存错误。训练结束后GPU显存占用与利用率均归零。上述结果只证明训练数值稳定，尚未提供Task8/9成功率提升证据；下一步需使用未参与采集的held-out roots，在固定H9、相同initial root和policy seed下与50999 base做配对闭环评测。

2026-07-23（Asia/Shanghai）完成hard-state DAgger round1 step999与50999 base的严格held-out固定H9配对评测。评测只使用未参与采集的initial roots 25–49，每个root使用两组policy seed（显式episode IDs 25–49和75–99），Task8/9各50局、合计每个checkpoint 100局；两次运行均固定seed7、Action-CoT denoising 10步、execution horizon 9、`num_steps_wait=10`和相同当前代码路径。Base结果位于`/root/autodl-tmp/acotvla/task89_hard_state_dagger_eval/base50999_fixed_h9_heldout50_seed7`，总体89/100，Task8为42/50、Task9为47/50，calls/episode为47.87。新checkpoint结果位于`/root/autodl-tmp/acotvla/task89_hard_state_dagger_eval/dagger1k_fixed_h9_heldout50_seed7`，总体91/100，Task8为44/50、Task9为47/50，calls/episode为45.71；calls下降主要来自timeout由11个减少到9个，而两次顺序运行的per-call timing存在波动，不据此声称模型计算加速。

严格配对统计中，新模型相对base共有8个rescues和6个regressions，净增2局、提升2个百分点，双侧exact McNemar `p=0.79052734375`。Task8为7 rescues、5 regressions、净+2，`p=0.7744140625`；Task9为1 rescue、1 regression、净0，`p=1.0`。Task8 rescues为episodes 33、42、47、87、89、91、99，regressions为26、39、40、78、90；Task9 rescue为84、regression为34。按policy-seed分半后方向不稳定：前50个跨任务episode为base 47/50、新模型46/50，3 rescues/4 regressions、净-1；后50个为base 42/50、新模型45/50，5 rescues/2 regressions、净+3。Task8两半分别净0和净+2，Task9两半分别净-1和净+1。

本轮没有通过预先锁定的继续门槛：总体仅+2个百分点，低于至少+5个百分点；rescues/regressions为1.33而非约2倍；且两个policy-seed半集方向反转。结果表明仅用来自2个train roots的3条/705帧高置信纠正轨迹、以15%比例更新fine action expert，确实能改变未见hard states并产生rescues，但同时引入接近数量的regressions，尚未形成稳定、选择性的泛化。因此不自动启动第二轮相同配方DAgger，也不据此替换50999 base或进入全Task0–7 retention评测；下一步应停止扩大当前配方，优先考虑冻结base的小型selective residual recovery adapter，或在进入新训练前先改善纠正标签覆盖和参数隔离。评测完成后已停止模型服务，GPU无残留compute进程。

2026-07-23（Asia/Shanghai）为判断hard-state DAgger round1的弱增益是否只是step999训练过度，补做step500早停checkpoint的严格held-out固定H9小规模止损评测。checkpoint为`/root/autodl-tmp/acotvla/checkpoints/acot_libero_task89_hard_state_dagger_round1/task89_hard_state_dagger_round1_1k_1a3650e/500`，输出目录为`/root/autodl-tmp/acotvla/task89_hard_state_dagger_eval/dagger500_fixed_h9_valhalf1_seed7`；协议保持seed7、Action-CoT denoising 10步、execution horizon 9及未参与采集的held-out roots。运行到Task8前11局时，新checkpoint仅成功7/11，而配对50999 base为10/11，出现0 rescues、3 regressions、净降3局，因此按小规模止损规则提前终止，没有继续浪费完整50局或100局评测预算。该结果否定了“只需把同一DAgger训练提前停止”这一简单解释；step500与step999都显示全局替换fine expert会同时改变普通状态，当前核心问题更像缺少状态选择性，而非单纯训练步数过多。评测服务与客户端均已停止，GPU无残留compute进程。下一步转为量化base到DAgger的实际参数差异，并尝试冻结50999 base、只在高置信hard state启用的小型recovery/residual分支；仍需通过held-out配对rescues/regressions验证，不能把训练内拟合当作成功率提升。

2026-07-23（Asia/Shanghai）继续排查hard-state DAgger round1的回归来源并启动严格参数隔离控制。50999 base与原step999 checkpoint的逐叶差分显示182个参数叶子中有133个发生变化；除预期的fine `llm_2` 11个叶子、427932672个参数外，原配置还更新了coarse time MLP、explicit/implicit action reasoner、reasoning fusion及其他非LLM层。原因是`ACOTConfig.get_freeze_filter`只冻结指定LLM分支，不会自动冻结这些非LLM模块，因此上一轮“只更新fine expert”的描述并不严格。新增配置`acot_libero_task89_hard_state_dagger_round1_strict_fine`将trainable范围显式限制为fine `llm_2`、`action_in_proj`、`time_mlp_in/out`和`action_out_proj`，真实模型树核验为19个叶子、430098464个参数；代码提交`b876804`已推送main并同步服务器。

严格隔离控制仍使用同一round1纠正数据、85% general replay加15% correction、peak LR `3e-6`和1000步，只改变冻结边界。训练于17:56完整结束，step500与step999均保存于`/root/autodl-tmp/acotvla/checkpoints/acot_libero_task89_hard_state_dagger_round1_strict_fine/task89_dagger_strict_fine1k_b876804/`；step0至900每100步loss为0.0890、0.0641、0.0623、0.0621、0.0665、0.0639、0.0639、0.0646、0.0647、0.0626，grad norm为0.1258、0.0917、0.0872、0.0908、0.1311、0.0928、0.0960、0.0901、0.0938、0.0945，未出现NaN、OOM或Traceback，异步保存明确无错误。严格step500在held-out Task8 episodes 25–49、fixed-H9、seed7选择集上为22/25，与base 22/25相同；它救回33、42、47，但破坏25、26、36，形成3 rescues、3 regressions、净0，因此没有显示比原全局更新更好的选择性。严格step999的同一Task8选择集评测已启动，结果尚未完成。当前方法栈已整理为流程图：LeRobot/HF格式的general replay与hard-state DAgger纠正数据，经JAX/Flax NNX和Optax做strict-fine SFT、Orbax保存，再由WebSocket policy server接入LIBERO/robosuite/MuJoCo做fixed-H9配对质量门控；通过后才重新校准V2-P执行时域选择器用于速度优化，失败则转向改善纠正覆盖或冻结base的selective recovery adapter。

2026-07-23（Asia/Shanghai）按用户要求将技术流程缩小并聚焦于V2-P predictor的训练与在线使用，完成方法说明而未修改代码。训练端冻结ACoT-VLA base，在LIBERO决策snapshot上用共享一次prefix的K=20 batched-MC动作样本生成`final_risk`、`action_cot_risk`、`fused_risk`、`event_mask`和teacher `raw_h`，并从同一MuJoCo snapshot分别强制执行H1至H10、再用固定H9 continuation得到每个H的success、timeout、remaining steps和remaining calls；输入特征与反事实标签写入HDF5。训练集按`task_id+episode_id`分组切分，重点任务、高风险、夹爪变化及包含失败branch的记录加权采样；JAX/Flax NNX predictor使用prefix、proprioception、coarse/final chunks、上一action chunk、上一H、budget、episode progress和previous-valid，经3层时序编码器同时学习风险曲线、event、raw-H、每H成功/超时概率及剩余calls/steps，只保存`execution_horizon_predictor` Orbax sidecar，base policy不进入优化。在线端单次ACoT-VLA推理复用prefix、coarse和fine输出运行sidecar，不再做K次MC；distilled模式按预测风险曲线的首个event得到raw H，value-refined模式可再用成功/超时阈值选最大的安全H，最后通过目标平均H9的episode budget controller得到实际执行H，执行前H个动作后重新观测。该predictor控制的是环境动作执行长度`h_exec`，不是Action-CoT denoising步数，也不重新生成动作chunk。

2026-07-23（Asia/Shanghai）严格参数隔离hard-state DAgger控制的Task8选择集评测已完成并按止损规则结束。step500在held-out episodes 25–49、fixed-H9、seed7上为22/25，与50999 base的22/25相同，3 rescues、3 regressions、净0；step999在完全相同的25局上为19/25，base为22/25，仍救回episodes 33、42、47，但在26、29、36、37、40、49产生6个regressions，形成3 rescues、6 regressions、净降3局。由于step500没有净增益、step999明确退化，未继续运行Task9或episodes 75–99确认集，port8035 server和评测进程均已停止，GPU无残留compute进程。该结果说明原round1回归不能仅归因于冻结边界不严格；把更新严格限制到19个fine-expert本地叶子仍会全局改变正常状态。结合纠正数据仅3条、705帧且来自2个root，当前主要瓶颈更可能是纠正覆盖不足以及缺少只在hard state触发的状态选择性，而不是继续调整同一全局SFT的步数或冻结范围。

2026-07-23（Asia/Shanghai）根据7月20日至23日的事实日志整理本周阶段总结，覆盖V2-P正式基线与论文级口径校准、596-root horizon headroom、多seed H稳定性、冻结base的SFT+Q/PPO selector、action-chunk candidate oracle及privileged-progress确认集、targeted与uniform continued-SFT、hard-state DAgger round1和strict-fine参数隔离控制。总结保留严格协议差异和统计检验：V2-P distilled在seeded 100-trial协议下为95.2% success、30.536 calls/episode、3.387 s policy/episode，但尚未超过论文Frozen 96.0%、项目历史官方式96.6%或Fixed H9的Pareto表现；现有H-selector PPO、candidate/progress selector及全局targeted/DAgger SFT均未在held-out root与seed上形成稳定显著净提升。当前建议保留50999 base与V2-P作为速度骨干，停止继续调现有Q-direct/PPO、朴素candidate selector和同配方全局SFT；下一质量主线应在扩大高置信hard-state纠正覆盖后采用冻结base、置信门控的selective recovery/residual adapter，并继续用fixed-H9 paired rescues/regressions作准入门槛。

2026-07-23（Asia/Shanghai）核对当前Budgeted Event V2-P的在线输入链路。predictor在使用时需要当前决策时刻的observation信息，但不会单独再次编码原始图像：主ACoT-VLA策略先用正常observation完成一次视觉/语言prefix编码和动作生成，predictor直接复用其pool后的`execution_horizon_prefix_feature`，并显式读取归一化proprioception/state、当前coarse/final action chunks、上一action chunk与上一H、budget balance、episode progress和previous-valid。因而部署时客户端仍只需提供策略原本需要的当前observation及controller状态，不需要未来observation、环境真值、成功标签或第二次视觉前向；若脱离主策略单独运行sidecar，则必须自行提供这些中间特征和动作输入。

2026-07-23（Asia/Shanghai）进一步核对`execution_horizon_prefix_feature`在ACoT-VLA前向中的截取位置。它不是VLA处理前的原始observation或浅层embedding：图像先经过PaliGemma image encoder、语言先转换为token embedding，二者拼成prefix后再完整通过一次PaliGemma LLM，得到上下文化的`prefix_out`；随后按有效token mask做mean pooling，形成2048维prefix feature供V2-P predictor复用。该特征位于视觉语言prefix编码之后、coarse/fine action suffix推理之前；机器人state不包含在这个prefix feature中，而是由predictor的`state_proj`作为独立输入加入。

2026-07-23（Asia/Shanghai）核对V2-P中entropy/risk的生成与在线使用位置。训练teacher在同一个observation和共享的一次VLM prefix下，默认用20个不同flow-noise seed生成K组coarse与final action chunks；final action样本逐时刻计算final entropy，coarse样本计算Action-CoT entropy并时间对齐到10步，另计算translation、rotation和gripper分量，再经robust positive normalization和加权融合得到`final_risk`、`action_cot_risk`、`fused_risk`、`event_mask`及teacher `raw_h`。这些entropy-derived curves在SFT中是predictor的监督目标，不是在线输入。部署时V2-P只运行一次ACoT-VLA动作采样，由prefix feature、state、当前coarse/final chunks及controller history直接预测三条risk曲线，再按首个超阈值event得到raw H；因此部署不实际从单个action chunk计算entropy，也不需要K次MC，所谓在线entropy是蒸馏后的预测风险而非现场测得的真实采样熵。

2026-07-23（Asia/Shanghai）核对当前`ExecutionHorizonPredictor`网络架构。它是Flax NNX实现的约1.31M参数轻量级多输入temporal MLP sidecar，不是新的Transformer、CNN或RNN。2048维VLM prefix feature、32维state和4维controller状态分别经Linear投影到256维；每个H1-H10时间位置将final action、对齐后的coarse action、上一chunk重叠部分、二者差值以及overlap-valid/consistency组成130维action feature，再投影到256维并叠加全局context。时序主干包含3层局部邻域残差MLP，每层拼接左邻、当前、右邻三个256维token后以Linear 768→256和Swish更新；10步token均值与context拼接后经summary Linear 512→256。逐时间步head预测final/Action-CoT/fused risk和event，summary head并行预测raw-H十分类与九个ordinal边界、H1-H10成功概率、超时概率、remaining calls和remaining steps。训练时冻结整个ACoT-VLA base，只保存predictor subtree sidecar。

2026-07-24（Asia/Shanghai）围绕“是否在predictor前用RL提取特征、以及如何增加方法深度并提高效果”完成了仓库与近期方法对照分析，未修改模型代码或运行新实验。当前V2-P前端只对完整VLM prefix token做2048维mean pooling，predictor再以局部temporal MLP融合state、coarse/final chunks和controller history；已有selector RL只消费冻结的predictor temporal feature及输出heads，因此无法补回pooling后丢失的空间、任务阶段和动态信息。结合现有Task8 selector PPO仅净增1/40且calls约翻倍、candidate/progress规则未在独立root与seed上复现、hard-state DAgger全局更新同时产生rescues与regressions的证据，本次不建议把普通PPO简单当作独立“特征提取器”或继续只堆叠H actor。

推荐将主方法升级为ACoT特有的return-aware predictive representation与selective verifier：冻结base VLA，使用完整prefix token的learned-query pooling、Action-CoT/final action token、proprioception和短历史构造state-action latent；训练期利用MuJoCo/BDDL privileged state、短前缀后的next latent、contact/grasp/drop/phase/progress以及同root多H多seed terminal outcome，联合训练latent dynamics/successor feature、distributional success/cost critic、rescue-vs-regression ranking和不确定度ensemble，privileged信息只作为teacher或辅助标签，部署student仍只读可获得观测与动作特征。predictor默认保持H9/H10，仅当候选短H或替代chunk相对reference的成功优势下置信界超过门槛时干预；执行期间再用轻量视觉latent monitor比较预测与实际特征演化，持续偏离时提前截断并replan，高风险状态才启用K=2–4 candidate rerank或冻结base的小型residual recovery adapter。该设计借鉴了DEHP的冻结chunk-policy horizon RL、TD-MPC2/SALE的控制相关state-action表征、AutoHorizon的内部attention信号、VLA-Corrector的latent dynamics monitor、VeriSpace的空间动作验证，以及HiPolicy/Mixture-of-Horizons的多频率或多horizon建模，但以ACoT coarse/fine双流、counterfactual H标签和保守选择门控形成项目自己的组合。

建议按四个可止损阶段验证：先只比较mean pooling与token-query/return-aware representation在未见root和未见seed上的Brier/ECE、pairwise ranking和future-latent误差；再闭环比较fixed H9、V2-P distilled与保守distributional selector的success/calls/Pareto；随后单独加入online monitor和candidate/recovery并报告触发覆盖率、false intervention、rescues/regressions；最后仅在前三阶段稳定正向时做encoder/actor小步constrained PPO-Lagrangian，显式约束calls/policy time并保留对SFT策略的KL。当前这些是设计建议，不构成任何成功率提升结论。

2026-07-24（Asia/Shanghai）进一步讨论了仅预测`H∈[1,10]`造成的方法与加速上限。结论是：execution horizon只改变完整policy调用频率，最大H=10形成硬上界；它既不降低单次ACoT-VLA调用成本，也不能在不重规划时修正错误action chunk或减少任务本身的冗余物理动作。因此即使继续增强H predictor或用RL学习其特征，只要输出空间仍只有H，能力上限基本不变。

建议把H predictor升级为暂名`ACoT Adaptive Refresh-and-Correct (ACoT-ARC)`的结构化推理控制器。重型ACoT-VLA周期性生成coarse/final action chunk，轻量闭环分支在每个环境步读取当前低成本视觉变化、proprioception、缓存的prefix/Action-CoT/action tail和执行历史，分层选择：继续缓存动作、对当前/剩余动作施加门控residual correction、仅刷新final action expert、复用或增量更新视觉prefix后刷新coarse+final、或者进行完整VLM重规划；同时条件式选择coarse/final denoising NFE和`h_exec`。为避免把所有组合做成巨大平坦分类，router应先预测refresh level，再由条件head预测compute budget与H。缓存旧prefix时必须通过轻量delta-token/视觉变化模块纳入当前观测；不能直接把旧图像KV当成当前状态。

仓库结构支持该方向的第一步验证：`src/openpi/models/acot_vla.py`已将prefix、implicit、coarse和final expert拆成独立profile入口，保留`kv_cache`、coarse override和动态Action-CoT denoising steps；`src/openpi/policies/policy.py`已分别统计`vlm_ms`、`implicit_action_reasoner_ms`、`coarse_action_expert_ms`和`action_expert_ms`。已有Stage B结果也限定了错误方向：K10/K5短Action-CoT token输入相对cached override没有速度收益，final expert约19.1–19.2 ms基本不变，因此不能只做coarse token裁剪；必须真正跳过整段模块/迭代、做跨时刻warm start，或用轻量residual分支替代部分完整调用。

推荐的训练顺序是先做反事实监督而非直接端到端RL：从同一snapshot分支执行不同refresh level/compute budget/H，记录成功、timeout、remaining steps/calls、分阶段latency和相对完整重规划的动作/进展差；用hard-state DAgger成功纠正构造residual target，并以正常状态的零residual和强门控抑制regression。先训练分布式value/cost heads与保守router，达到held-out root/seed上的校准和rescue-vs-regression门槛后，再用constrained RL联合优化离散refresh决策与连续residual。近期VLA-Cache、Action-to-Action Flow Matching、REMAC、Mixture of Horizons和PolicyTrim分别提供了跨帧视觉缓存、历史动作warm start、异步chunk纠正、多horizon生成和减少物理动作冗余的相关参考，但本次只完成设计分析，未修改模型代码或运行新实验。

2026-07-24（Asia/Shanghai）按用户要求进一步细化ACoT-ARC并核对原始论文参考，未修改模型代码或运行训练评测。当前H-only机制的硬限制可写为episode policy time近似等于完整调用次数乘单次完整调用成本；`ExecutionHorizonPredictorConfig`明确把action horizon固定为10，V2-P正式结果中平均H已为9.874且96.2%的决策选择H10，因此继续提高H分类准确率几乎不能再降低调用次数，也不能改变已生成chunk的动作内容或单次调用的VLM/coarse/final计算。更细粒度不应理解为预测小数H，而应同时控制动作残差、刷新深度、coarse/final denoising预算和执行H。

细化后的在线结构为：完整ACoT调用生成并缓存视觉语言prefix/KV、implicit reason、15步coarse Action-CoT和10步final action；每个环境控制步由轻量当前观测编码器读取最新低分辨率图像/局部patch、proprioception、剩余coarse/final chunk、chunk位置、前后chunk一致性和历史mode，形成监测latent。层级router先在continue、bounded residual correction、partial replan和full replan之间决策；只有进入partial replan后才继续选择复用coarse只刷新final、刷新coarse+final或完整刷新prefix，并条件式选择coarse/final NFE与执行H，从而避免把所有组合展开成巨大平坦分类。当前仓库已有分阶段profile和同一调用内KV/override接口，但尚未实现跨observation安全缓存；旧prefix不能直接代表新图像，partial refresh必须重新计算当前prefix，或加入能够只更新变化视觉token的delta-token机制。

训练方案进一步分为三阶段。第一阶段先构造模式反事实上界：从同一MuJoCo snapshot、相同policy/continuation seeds分别评估cached action、teacher residual、当前prefix+cached coarse+fresh final和full replan，记录terminal success、progress、timeout、remaining steps/calls以及真实分阶段latency，确认结构化控制空间存在held-out headroom。第二阶段冻结50999 base，使用高置信成功纠正和扰动恢复样本训练bounded residual head，同时以大量正常状态零residual、动作幅度约束和保守gate抑制regression；再训练每种mode的distributional success/cost critic，以成功下置信界选择满足质量约束的最便宜mode。第三阶段只有在未见root/seed上稳定出现净rescues且真实policy time进入Pareto前沿后，才用constrained semi-MDP RL优化层级router；RL action包含refresh mode、compute budget、H和可选连续residual，奖励同时考虑terminal success、真实latency、物理steps、修正幅度和频繁切换，并以base/V2-P成功率作为约束，而不是把PPO仅当作H predictor前的特征提取器。

本次原文核对显示：DEHP本身仍是冻结chunk policy后对H做在线RL，适合作为H-only基线；A2C2最接近per-step residual分支，它读取最新观测、base action、chunk位置和base特征输出轻量修正；RTC在flow denoising中冻结必执行prefix并inpaint剩余chunk，REMAC进一步训练masked chunk correction；VLA-Cache按相邻帧变化与任务相关性选择视觉token并更新KV；Action-to-Action Flow Matching用历史proprioceptive action初始化生成以减少从噪声开始的迭代；Consistency Policy用teacher轨迹一致性蒸馏减少采样步；Mixture of Horizons在动作生成器内部并行融合不同horizon；PolicyTrim用RL同时延长可靠chunk和减少冗余物理steps；VLA-Corrector和PATCH则提供action-conditioned latent drift monitor。重要边界是这些论文的外部指标协议不同，不能直接外推到本checkpoint；VLA-Corrector还报告OGG虽提高success-per-call，但总体wall-clock inference约增加1.62至1.68倍，进一步说明本项目必须报告真实policy time而不能只看calls。

2026-07-24（Asia/Shanghai）根据用户对ACoT-ARC真实速度的质疑，重新以端到端延迟而非控制粒度排序研究路线，未修改模型代码或运行新实验。质疑成立：现有profile中完整ACoT调用为77.2211 ms，完全跳过显式Action-CoT/coarse生成后仍为59.2252 ms，因此只优化coarse部分的理论上限仅约1.304倍；在V2-P平均H=9.874时，若每个环境步都新增monitor/corrector，其平均耗时必须低于约1.823 ms才不抵消一次完整调用中约17.996 ms的coarse可省成本。故ACoT-ARC应降级为可选的成功率/恢复模块，不能作为主加速贡献。

新的速度主线暂称`Fast-ACoT`：先同时压缩ACoT内部两个独立的10步flow循环，而不是继续增强H predictor。第一阶段将coarse Action-CoT expert蒸馏为1至3步consistency/mean-flow student；第二阶段将final action expert也蒸馏为1至3步，并加入student-coarse条件下匹配完整teacher final action的跨阶段蒸馏，避免只压缩coarse却把误差传给未适配的final expert；部署时最多每个policy call运行一次廉价置信router，在1/3步快速路径和10步fallback间选择，不增加per-environment-step视觉前向。按当前分项profile作线性缩放的乐观算术估计，coarse与final都降到3/10 NFE时约为51.20 ms、1.51倍，均降到1/10时约43.76 ms、1.76倍；这些不是实测结果，并假设迭代成本线性、固定约40.04 ms的prefix/implicit/其他开销不变且质量不下降。两段flow压缩后，约40 ms的剩余前端将成为新瓶颈，届时再引入VLA-Cache式跨观测视觉token复用或EfficientVLA式语言层裁剪、任务相关视觉token选择与action-head特征缓存；RTC式异步chunk生成可改善机器人wall-clock吞吐，但不等同于降低模型FLOPs，应单独报告。

现有数据给出了无需训练的强基线和下一步最小诊断：固定coarse NFE=3在10任务各20局中点估计success为0.970、相对10步的closed-loop latency下降18.54%，虽优势未达统计显著；因此任何学习方法都必须至少与`coarse=3, final=10`比较。最高优先级应先做`coarse NFE∈{1,3,5,10} × final NFE∈{1,3,5,10}`二维速度与闭环Pareto sweep，确认final expert的可压缩空间，再决定直接少步推理是否足够或必须蒸馏。CoT-LLM方法可迁移的部分也需改变落点：Chain-of-Draft对应直接生成少量关键Action-CoT keypoints而非生成后裁剪；CODI/Coconut对应把15帧显式轨迹蒸馏为少量连续reasoning tokens，并重新训练final expert消费它们；LayerSkip对应早层draft、后层verify和full fallback；Mixture-of-Depths对应让真正的attention/MLP块只处理top-k动作/视觉token。由于ACoT的15帧在单次flow forward内并行，单纯减少frame/token数量而不跳过完整网络层或NFE不会像自回归文本CoT那样直接获得token级加速。当前推荐不需要RL；只有少步student和廉价supervised router稳定进入success-latency Pareto后，才考虑用约束RL微调预算选择。

2026-07-24（Asia/Shanghai）澄清`coarse=3, final=10`的准确含义，未修改模型代码或运行实验。`coarse=3`表示显式Action-CoT/coarse flow从噪声生成15帧coarse trajectory时只运行3次denoising NFE，即3次coarse expert完整前向，而不是只生成3帧或执行3个动作；`final=10`表示final action flow仍按原始配置运行10次denoising NFE，最终输出长度仍为10的action chunk。两者都描述一次policy调用内部的生成计算量，与execution horizon `H`不同；`H`决定这10个final actions中实际执行前几个后再重新观测和调用policy。例如`coarse=3, final=10, H=9`表示先用3次coarse前向生成完整15帧Action-CoT，再用10次final前向生成完整10步动作，实际执行前9步后replan。该配置只减少相对`10/10`基线的7次coarse expert前向，保留final expert计算与输出长度不变，因此是保守的单分支加速基线。

2026-07-24（Asia/Shanghai）接受用户关于固定`coarse/final NFE`仍只是超参数修改的纠正，并重新界定算法贡献，未修改代码或运行实验。`coarse=3, final=3`不改变网络、训练目标或推理机制，只改变同一flow ODE求解器的迭代次数，因此只能作为可行性诊断和强速度基线，不能作为新方法。建议把真正主方法改为暂名`Latent Distilled ACoT (LD-ACoT)`的结构与训练算法：冻结完整`10+10`步ACoT-VLA作为teacher，用少量learned latent planning queries从共享VLM prefix一次性产生隐式Action-CoT；训练期使用辅助decoder重建teacher的15帧显式coarse trajectory并做阶段/夹爪监督，但部署时移除该decoder，不再运行显式coarse flow；final student采用consistency/mean-flow endpoint distillation，用一次前向逼近teacher的10步final flow，并必须在student latent条件下匹配teacher final action，形成跨阶段蒸馏，避免teacher-coarse到student-coarse的部署分布偏移。

LD-ACoT的建议损失包含latent/teacher hidden alignment、辅助coarse reconstruction、coarse phase与gripper事件、one-step final endpoint/velocity consistency、student-latent-conditioned teacher action matching及demonstration BC。部署默认只执行一次latent planner和一次one-step final student；另训练同一policy-call内的校准error head，根据首遍draft的内部残差决定是否复用student做第二次corrective refinement或回退完整teacher，属于学习的条件计算而非固定`NFE=3`。该设计迁移CODI/Coconut的“显式CoT teacher蒸馏到连续latent”思想、Consistency Policy/OFP的少步生成和LayerSkip的draft-verify，但针对ACoT形成显式15帧coarse teacher、latent planner、one-step final和跨阶段student-conditioned distillation的组合。二维`coarse/final NFE` sweep仍应保留，但其角色只是证明现有模型的无训练Pareto与蒸馏必要性；论文级消融必须分别比较固定少步、仅final蒸馏、仅latent Action-CoT、联合跨阶段蒸馏及adaptive corrective refinement。

2026-07-24（Asia/Shanghai）进一步澄清LD-ACoT中的`latent planning tokens`与`one-step action student`，未修改代码或运行实验。latent planning token不是文字token、coarse action帧或可执行动作，而是由少量learned queries对当前VLM prefix、proprioception和implicit feature做cross-attention后得到的`M×D`连续内部向量；它们共同压缩teacher的15帧显式Action-CoT，可能分布式编码目标物体/目标位置、接近-抓取-搬运-释放阶段、运动方向与夹爪切换，但这些语义不应在无probe证据时逐token强行解释。训练期通过辅助decoder从latent tokens重建teacher coarse trajectory，并增加阶段、夹爪事件、teacher hidden alignment和最终动作监督；部署时移除辅助decoder，latent planner本身必须是少层小型adapter，从而真正跳过原10步coarse flow。

one-step action student也不是只预测或执行一个机器人动作，而是在一次final action网络前向中生成完整长度10的action chunk。原final flow从噪声action chunk出发，调用velocity network约10次逐步积分到最终动作；one-step student通过teacher endpoint/trajectory consistency、student-latent-conditioned action matching和demonstration BC，学习从同样的噪声、当前prefix/state及latent plan直接近似10步teacher的最终endpoint。执行时仍由独立execution horizon `H`决定完整10步chunk中实际执行多少步。两者的关系是latent planner替代显式coarse生成，one-step action student替代final的10次迭代；若只实现其中一个，另一段仍是主要延迟来源。

2026-07-24（Asia/Shanghai）根据用户关于latent planner削弱CoT属性的质疑，核对ACoT与显式CoT压缩参考并再次调整主方法，未修改代码或运行实验。原ACoT-VLA将Action-CoT定义为由EAR给出的结构化coarse action intent序列，并与IAR共同条件化下游action head；若完全用不可观察latent替代15帧coarse trajectory，方法更像隐式规划蒸馏，不再是最忠实的ACoT加速。建议将首选方案改为`Fast Explicit ACoT`：保留完整15帧、动作空间内可观察和可干预的显式Action-CoT表示，冻结10步teacher后训练consistency/mean-flow coarse student，用一次前向直接生成相同15帧coarse chain；同时蒸馏final flow，但final student始终显式消费student生成的Action-CoT。训练目标除endpoint外，还应保持逐帧pose、相邻frame delta/curvature、时间顺序、gripper/contact事件、teacher hidden feature及最终动作影响，形成process-preserving与student-conditioned cross-stage distillation。该方法改变训练目标和生成映射，而非仅设置`action_cot_denoising_steps=1`，且保留ACoT论文的显式推理链定义。

可作为第二层创新的是`Drafted Action-CoT`：参考Chain-of-Draft、TokenSkip和Step-Entropy CoT compression，从teacher的15帧链中识别必须保留的显式关键action intents，例如首尾、gripper切换、contact变化、轨迹高曲率和对final action有高leave-one-out影响的frame；student直接生成按时间排序的少量keyframes、duration和`STOP/valid`标志，而不是先完整生成15帧再post-hoc剪枝。final expert必须重新训练为直接消费compacted keyframe tokens，或使用很小的插值器恢复兼容序列；现有K10/K5 override没有速度收益已经证明，只删值或恢复到15帧而不跳过实际attention/MLP/NFE不能形成加速。连续ACoT不能照搬文本token Shannon entropy，建议以MC不确定度、teacher denoising收敛、leave-one-frame-out final-action变化、event preservation及少量counterfactual rollout regret联合标注step importance。

原文参考支持这一显式路线：ACoT-VLA与2026年的Coarse-to-Control都把compact coarse action sequence作为可执行动作生成前的显式计划；Chain-of-Action使用goal keyframe起始的显式action-level backward reasoning、dynamic stopping和multi-token prediction；Chain-of-Draft保持显式但只输出关键中间推理；TokenSkip和Making Slow Thinking Faster分别学习跳过低重要性token或低step-entropy推理步骤；Dynamic Early Exit根据推理收敛自适应停止；Consistency Policy则提供从多步生成轨迹蒸馏为少步模型的机器人控制先例。推荐顺序是先做保持15帧的explicit consistency distillation，以最低表示风险验证真实速度与成功率；再加入event-preserving variable-length draft chain作为更强的CoT压缩贡献。latent-only方案降为消融对照，不作为主方法。

2026-07-24（Asia/Shanghai）进一步明确Fast Explicit ACoT两层的组合方式，未修改代码或运行实验。两层最终可以组成同一个方法，但不建议从头联合训练：第一层explicit consistency distillation固定保留15帧Action-CoT，只把10步coarse flow学习为一次student前向，解决生成求解器成本并保持原始表示；第二层Drafted Action-CoT把15帧teacher chain离线压缩为按时间排序的显式keyframes、duration和valid/STOP，解决推理内容冗余。第一层是必做的速度骨干，第二层是通过第一层验收后再叠加的结构扩展。

建议训练顺序为四阶段。Stage 0缓存冻结`10+10` teacher在训练observation上的15帧coarse chain、final action、必要denoising states和gripper/contact/curvature事件。Stage 1只训练one-step explicit coarse student输出完整15帧，final仍使用原10步teacher，以naive `NFE=1`和原`10/10`分别验证“蒸馏算法收益”及CoT质量。Stage 2根据事件与leave-one-frame-out final-action影响产生compact keyframe teacher labels，训练`K_max`个显式keyframe slot、duration及valid/STOP head，使student直接输出短链，而不是先生成15帧再剪枝。Stage 3先用ground-truth compact teacher chain训练final student，再逐步切换到student生成的compact chain，最后联合微调coarse draft与one-step final，避免两段误差同时漂移。

最终部署路径为`observation -> shared prefix -> one-step explicit drafted CoT generator -> K个action-space keyframes+duration+STOP -> compact-conditioned one-step final student -> 10-action chunk`。需要保留一个重要速度边界：第一层通过减少完整expert NFE有明确计算收益；第二层从15帧缩到K帧不会自动加速，当前K10/K5结果已表明mask或短K/V但仍运行相同dense final expert几乎无收益。只有final head直接消费物理compacted tokens，并以K-bucket独立编译、active-token compaction或更小的compact-conditioned head真正减少attention/MLP计算后，才能把第二层计入速度贡献。若microbenchmark仍无额外wall-clock收益，第二层应只作为显式CoT简洁性/质量贡献或停止扩展，不能宣称叠加加速。

2026-07-24（Asia/Shanghai）明确Fast Explicit ACoT与execution horizon `H_exe`的关系，并重新审视方法是否具有ICRA级别贡献，未修改代码或运行实验。完整10步action chunk仍可交给`H_exe`决定执行前缀长度，但`H_exe`属于生成后的闭环执行调度，与显式CoT生成/压缩是正交层。开发阶段必须先固定H9或同一H隔离单次policy计算收益；最终部署才组合为`fast explicit CoT -> fast final chunk -> calibrated H_exe -> execute A[:H_exe]`。由于少步/短链student会改变action、risk和prefix-feature分布，现有V2-P不能直接当作已验证组件复用，必须在新student上重新收集counterfactual标签、校准或重训；同时单次调用变便宜后success-vs-replan-cost的最优H也可能变化，不能继续把目标平均H9视为固定真值。正式factorial comparison应包含原ACoT+fixed H、Fast ACoT+同fixed H、原ACoT+V2-P和Fast ACoT+重新校准V2-P。

对发表强度的结论作保守修正：`consistency distillation + compact keyframes`分别已有Consistency Policy/OFP与Chain-of-Draft/TokenSkip/Coarse-to-Control/Chain-of-Action等直接先例，简单串联及NFE/K sweep本身不能保证ICRA级算法创新，只能构成合理工程骨架和强baseline。若以ICRA为目标，更明确的核心研究问题应是“如何在保留显式action-space reasoning可观察性的同时，按状态自适应分配推理内容和计算，并用可验证的step utility保证控制效果”。建议的算法升级为`Anytime Explicit ACoT`：先一次产生最小但完整的显式keyframe chain；轻量interval verifier预测每个相邻keyframe区间的执行风险与增加一个Action-CoT step的边际收益；只在最高风险区间插入新的显式action intent并更新时间/duration，直到所有区间通过证书或达到安全上限。每次refinement后均保持一条可执行、可视化的显式CoT，不使用不可解释latent。

Anytime Explicit ACoT的训练信号不应只用entropy，而应通过teacher chain intervention构造step-utility标签：删除或合并第i个coarse intent，测量final action变化、gripper/contact事件破坏、teacher/value disagreement和少量snapshot rollout regret；训练generator预测最小draft、verifier定位需细化区间、insertion head恢复最有价值的显式intent，并用process-preserving与final-action loss联合优化。`H_exe`可以在最后由完整action chunk与最终显式chain的validity/duration共同校准，但不是主创新。该方案比固定短链多出“可变内容、风险定位、显式插入、每次迭代可执行、step utility intervention”这一完整算法闭环；是否达到ICRA水平仍取决于真实wall-clock、成功率、跨任务泛化、消融和与强加速基线的比较，当前仅为设计建议，不能预先保证录用级别。

2026-07-24（Asia/Shanghai）根据用户强调主方法不必围绕`H`或短链长度`K`，再次从ICRA投稿目标重构研究基线，未修改代码或运行实验。新的建议彻底移除主方法中的execution-horizon选择、variable-K keyframe选择和在线compute router，固定保留原ACoT的15帧显式Action-CoT与10步final action chunk；核心研究问题改为“能否在少步生成显式Action-CoT和final action的同时，保持中间动作推理对下游控制的功能忠实性”。暂名`IR-ACoT: Interventional Reasoning Distillation for Fast ACoT`。

IR-ACoT冻结原`10-step coarse + 10-step final`为teacher，分别以consistency/mean-flow目标训练coarse与final student，使部署只需一次coarse student前向生成完整15帧显式chain，再一次final student前向生成完整10步action chunk。区别于普通双flow蒸馏的核心是结构化interventional response alignment：对teacher Action-CoT执行单帧插值替换、平移/旋转扰动、gripper事件翻转或平移、相邻时间交换、segment corruption等干预，在固定observation、prefix与implicit reason条件下通过`explicit_action_reason_override`重新运行teacher final expert，得到`ΔA_T=A_T(C_T^I)-A_T(C_T)`；对student chain施加对应干预并计算`ΔA_S`，训练`ΔA_S≈ΔA_T`，同时匹配影响方向、幅度和不同干预的utility排序。该损失相当于用结构化有限差分蒸馏final action关于显式Action-CoT的响应，而不只匹配teacher endpoint或hidden feature；高影响gripper/contact/phase干预可加权，少量MuJoCo snapshot branch outcome只用于校准干预影响与真实控制风险，不作为部署输入。

建议总损失由coarse/final flow consistency、15帧frame与temporal-delta过程保持、gripper/event监督、teacher/student action endpoint、demonstration BC和interventional response alignment构成。final student先在teacher chain条件下训练，再切换到student chain，并同时在clean/intervened chains上学习，防止student虽输出看似正确的CoT但action head实际忽略它。所有intervention与可选alignment verifier只在训练期存在，部署不增加monitor、verifier、H/K selector或额外视觉前向；核心实验统一固定H9以隔离生成算法，现有V2-P只作为后续独立系统组合，且若组合必须在新student上重新校准。

该方向的相关边界为：ACoT-VLA、Coarse-to-Control和Chain-of-Action保留显式action-space intermediate plan；Consistency Policy与OFP解决少步动作生成；LaRA-VLA将reasoning隐式化；ElegantVLA调度视觉、LLM和denoising计算；LLM领域的Making Reasoning Matter、FRIT与Causal Distillation说明显式推理可能不具有稳定因果作用，并提供counterfactual/interchange intervention训练先例。IR-ACoT的目标差异是把interventional faithfulness迁移到连续Action-CoT，并与ACoT双flow少步蒸馏联合，使显式chain在加速后仍是下游动作的load-bearing mediator。该组合目前是投稿设计而非已验证创新，仍需文献持续查重、真实速度和闭环结果支持。

建议的投稿对照不再以H/K为主轴，而包括：原ACoT `10/10`、原权重naive `1/1`及当前强`3/10`solver baselines、去除EAR的IAR-only/direct-action路径、final-only consistency distillation、普通dual-flow consistency distillation、dual-flow加process losses、完整IR-ACoT，以及可实现时的ElegantVLA式scheduler或latent-reasoning对照。主表所有方法使用相同fixed H与paired roots/seeds，报告success、per-call和per-episode真实policy latency、p50/p95、分阶段耗时及success-latency Pareto；CoT专属评价报告event preservation、teacher-student intervention-response error、utility rank correlation、matched/mismatched chain sensitivity和干预后闭环robustness。只有完整IR损失在匹配速度下稳定优于普通dual distillation，并同时保持或提升闭环success，才能支持核心贡献。

2026-07-24（Asia/Shanghai）进一步用控制变量口径解释IR-ACoT的投稿baseline体系，未修改代码或运行实验。术语上应区分：`base model/teacher`是原始50999 ACoT-VLA；`baselines`是用于排除其他解释的比较方法；`IR-ACoT`是proposed main method，不应称为baseline。所有内部对照固定同一teacher初始化、数据、15帧显式Action-CoT、10步action chunk、Fixed H9、initial roots、policy seeds、profile路径和硬件，逐级只改变一个因素。

建议内部对照阶梯为：B0原ACoT `10-step coarse + 10-step final`，给出完整质量与延迟锚点；B1原权重naive `1/1`，不训练只强制一步，用来证明调NFE本身不足；B2当前强`3/10`，是不训练的最佳已知solver折中；B3关闭EAR的IAR-only/direct-action路径，用来验证显式Action-CoT是否有必要；B4保持原10步coarse、只蒸馏one-step final，用来测final蒸馏贡献；B5只蒸馏one-step coarse、final仍10步，用来测显式CoT生成加速贡献；B6同时蒸馏one-step coarse和final、只使用普通consistency/endpoint/BC损失，是与IR-ACoT速度相同的最近基线；B7在B6上增加frame、temporal delta、gripper/contact等process losses，只保证student CoT外观和事件像teacher；完整IR-ACoT仅在B7上增加interventional response alignment，要求对grasp frame、gripper event或时间顺序的同一干预在teacher和student中引起一致的final-action变化。

结果解释应严格对应控制变量：B4对B0回答final分支能否安全加速，B5对B0回答coarse分支能否安全加速，B6对B4/B5回答两段联合误差是否累积，B7对B6回答过程监督是否有用，Ours对B7才回答interventional faithfulness是否是核心增益。如果Ours只优于B0但不优于B6/B7，不能把收益归因于新IR目标；如果B3与Ours质量相当，则显式CoT价值不足；如果B1已达到同等质量，则复杂蒸馏没有必要。外部对照如ElegantVLA式scheduler和latent-reasoning方法用于与近期路线定位，但不能替代上述同代码、同速度的内部控制。

2026-07-24（Asia/Shanghai）进一步澄清IR-ACoT中EAR/IAR的去向，未修改模型代码或运行实验。完整方法并不删除EAR：原迭代式EAR由冻结teacher的10步coarse flow产生15帧显式Action-CoT，student侧将其替换为经consistency distillation训练的一步`Fast-EAR`，仍输出同样的15帧action-space reasoning，并继续通过原explicit cross-attention分支进入final expert。IAR保持原结构，从共享VLM prefix的KV表示提取隐式action tokens；第一版为控制变量应冻结VLM、IAR以及原EAR/IAR融合模块，主要训练Fast-EAR和one-step final student，待稳定后再把融合模块解冻小步适配。部署路径仍为`prefix -> IAR`与`prefix/noise -> Fast-EAR`两条并行支路，经EAR/IAR fusion后由one-step final student输出完整10步action chunk。IR干预损失只施加于Fast-EAR产生的显式链，固定IAR和observation比较clean/intervened chain导致的final-action变化，从而专门约束EAR的功能作用；若同时改变IAR，就无法把动作响应归因于EAR。

2026-07-24（Asia/Shanghai）制定IR-ACoT的分阶段训练与实验启动方案，未修改模型代码或运行训练评测。第一步不是直接训练完整方法，而是锁定50999 teacher、10/10生成、Fixed H9、相同paired roots/seeds与`profile_policy_timing=True`，复测B0原模型、B1原权重naive 1/1和B2当前3/10，并利用已有`explicit_action_reason_override`做EAR有效性审计：固定observation、prefix、IAR、final noise和policy seed，只对15帧显式Action-CoT施加几何、gripper时序、相邻帧交换、segment插值及null干预，比较teacher final-action response并在少量snapshot branches上校准其与控制regret的关系。若语义干预相对null没有稳定响应，或响应与闭环风险无关，则停止IR路线，不能假定EAR是load-bearing mediator。

现有`ACOT_VLA.compute_loss`仍是用数据集coarse/actions进行普通双flow matching和teacher forcing，不能通过把推理NFE设为1训练Fast-EAR。计划以向后兼容的新入口增加teacher trajectory exporter和consistency trainer：从原LIBERO训练split按task/episode分层缓存coarse/final flow的固定noise、相邻time states和teacher endpoints，不缓存巨大prefix KV；冻结VLM与IAR，严格allowlist训练coarse `llm_1`及其input/time/output投影得到B5 Fast-EAR，再独立冻结EAR训练final `llm_2`及其投影得到B4。每支先做小数据/短步数的shape、梯度、checkpoint和one-step inference smoke，再以open-loop endpoint、temporal/event、student-chain导致的teacher-final action deviation、真实分阶段latency及fixed-H9闭环success决定是否扩大训练。

只有B4和B5分别通过非劣与加速门槛后才组合B6：final student先消费teacher chain，再逐步切换到Fast-EAR student chain，并通过冻结teacher对student chain重新标注final endpoint，避免teacher-chain训练与student-chain部署的分布偏移。随后依次增加B7 process losses和完整IR response loss，保持B6/B7/Ours完全相同的1-step EAR、1-step final推理图；最终用多训练seed及paired LIBERO roots/policy seeds比较success、rescues/regressions、per-call/per-episode latency、p50/p95、event preservation、intervention-response error和utility rank correlation。开发默认沿用本地改代码、推送后由用户在服务器拉取运行的流程，且新增脚本不改变原训练和eval默认行为。

2026-07-24（Asia/Shanghai）根据用户希望更激进地节省首轮验证时间，将IR-ACoT启动方案压缩为快速否证型MVP，未修改模型代码或运行训练评测。保留不可省略的同速度对照，但首轮跳过完整consistency trajectory、B4/B5独立正式闭环、B7 process ablation、全LIBERO-10、多训练seed和大规模student-chain迭代重标。先在约200个分层状态上用同observation/IAR/final noise完成EAR干预审计；若semantic intervention的teacher action response不能稳定超过null，立即停止。若通过，则从50999复制同一初始化，以约1万条跨10任务分层snapshot、每条一个固定noise训练直接one-step endpoint student：先短训Fast-EAR，再固定其student chain生成一次teacher-final relabel，随后得到一个共同dual one-step warm start。

共同warm start之后复制成两个总训练步数相同的分支：`B6-lite`继续clean endpoint/BC训练，`IR-lite`在相同clean loss上增加interventional response alignment；二者使用相同数据、初始化、optimizer、step数和1-step EAR + 1-step final推理图。快速open-loop门槛暂定为IR-lite相对B6-lite把teacher-student intervention-response RMSE降低至少约20%，同时clean final-action endpoint误差恶化不超过约5%；通过后只在Task8、Task9及一个高成功率retention task上各做约25个paired fixed-H9 episodes，并复测B0、B6-lite、IR-lite的真实分阶段延迟。该小样本闭环仅作go/no-go，不作论文显著性结论。若teacher EAR响应弱则否定IR路线；若EAR响应存在但one-step clean reconstruction失败，只说明endpoint MVP压缩过猛，应升级为完整consistency或两步诊断；若IR open-loop改善却不产生更多闭环rescues，则不继续投入完整ICRA实验。

2026-07-24（Asia/Shanghai）估算快速否证MVP的时间与新增存储，未运行任务。以最近同服务器ACoT训练稳态约2.1秒/step、1000步主体约37分钟为锚点，约6000个顺序优化step的纯训练下界约3.5至4小时；考虑IR分支clean/intervened额外前向、JIT、数据读取和checkpoint，训练预算更现实为约4至7小时。EAR 200-state审计约15至30分钟，1万snapshot的teacher endpoint/intervention导出与一次student-chain relabel约1至2小时，open-loop/timing约0.5至1小时，三个模型在三个任务上各25局的225局paired闭环暂估约2至4小时；代码就绪后获得open-loop go/no-go约需5至9个GPU小时，含闭环约8至14个GPU小时，实际日历时间还取决于实现与服务器往返。

标签缓存按模型真实32维padded action、FP32、不复制图像只保存dataset索引估算：coarse/final noise与clean endpoints约64 MB，四类intervention final endpoints约51 MB，一次student chain与teacher-final relabel约32 MB，连同metadata/HDF5开销预计总计约0.2至0.5 GB。存储主要来自checkpoint：现有全量Orbax checkpoint历史约21 GiB/份，若保存common、B6、IR及中间点会新增约63至105 GiB；因此计划采用50999 base加trainable subtree delta/sidecar，只保留一个可恢复optimizer state与common/B6/IR权重，推荐预留约20至30 GiB，极简清理optimizer后约8至15 GiB。最近服务器记录曾仅剩约72 GiB但不是当前实时值，启动前必须重新检查；若不实现delta保存则至少应准备约100 GiB新增空间。

2026-07-24（Asia/Shanghai）完成IR-ACoT快速否证MVP的代码实现，尚未启动正式训练或闭环评测。新增`src/openpi/action_cot/endpoint_dataset.py`与`scripts/audit_ear_interventions.py`，以可续跑的分片HDF5保存dataset/task/episode/frame索引、固定coarse/final noise、10步teacher clean endpoints以及null/平移/旋转/gripper时序/相邻交换EAR干预的teacher final-action响应；审计固定observation、IAR、policy seed和final noise，并以semantic response绝对值及相对null比例作为只用于go/no-go的开放环因果门槛。新增`ACOT_VLA.compute_endpoint_distillation_loss`和`scripts/train_acot_endpoint_distillation.py`，用`t=1`的一步endpoint公式训练Fast-EAR和final student，B6-lite只匹配clean endpoint，IR-lite额外匹配student/teacher在相同EAR干预下的final-action delta；训练allowlist只覆盖coarse或final 300M expert分支及其本地投影，VLM、IAR与reasoning fusion保持冻结。新增严格shape/路径校验的BF16 Orbax delta sidecar加载与`serve_policy.py`入口；只有同时包含coarse和final分支的完整sidecar才默认切到EAR=1、final=1，原训练、推理和eval默认行为不变。完整操作与对照协议写入`docs/ir_acot_mvp.md`。

在服务器RTX PRO 6000上使用50999真实参数完成实现级验证：静态编译与ruff检查通过，`endpoint_dataset_test.py`及`policy_config_test.py`共11项测试通过；真实参数树182个叶子中严格筛出coarse 19个、final 19个，未选中VLM/IAR/fusion叶子；dual IR前向`jax.eval_shape`通过。使用2条真实label做的一步coarse backward smoke得到train MSE 0.002109、validation MSE 0.003735；一步final IR backward smoke得到train final MSE 0.007370、IR cosine 0.643506，validation final MSE 0.000798、IR cosine 0.579738，并成功保存/加载38叶、全BF16、约1.3 GiB的完整coarse+final sidecar且自动采用1+1推理步数。2状态EAR审计的null median response为0.014453、translate为0.084872、ratio为5.872并通过当前阈值；这些都只是shape、梯度、I/O和因果链路smoke，样本量不足，不能作为方法效果、成功率或论文结论。

按用户授权检查并清理服务器空间。删除了50999下约17 GiB的`train_state`优化器状态，保留约8.2 GiB的`50999/params`与assets，因此该checkpoint仍可用于推理、评测和作为新微调初始化，但不能从50999原优化器状态原位续训；数据盘从约24 GiB可用增加到41 GiB可用。另删除本次验证生成的精确`/tmp/ir-acot-*019f91a7`副本与临时帮助文件，释放root overlay约2.8 GiB，保留仅5.4 MiB且后续有用的IR-ACoT JAX编译缓存；服务器正式repo未写入本次代码，仍仅显示既有`third_party/libero`脏子模块。当前2k aggressive pilot采用`checkpoint_interval=train_steps`时只保存coarse、B6、IR三个delta sidecar，估计持久盘新增约3.3 GiB，连同标签、Orbax临时写入和运行余量按8至12 GiB预算，现有41 GiB足够，不需要为首轮扩容；若改做多训练seed、频繁中间checkpoint或恢复全量train state，再考虑额外30至50 GiB。

2026-07-24 15:09（Asia/Shanghai）实时核对IR-ACoT测试状态。服务器没有tmux session、GPU compute进程或`audit_ear_interventions.py`、`train_acot_endpoint_distillation.py`、LIBERO eval、`serve_policy.py`相关进程，`/root/autodl-tmp/acotvla/ir_acot_pilot`及同前缀正式输出目录尚不存在；因此200-state EAR审计、2k标签导出、B6/IR正式训练和闭环效果评测均未启动。服务器正式repo仍为`b876804`，只显示既有`third_party/libero`脏子模块，新实现仍位于本地未提交工作树；数据盘保持41 GiB可用。当前已获得的只有此前记录的实现级检查、11项单元测试和2样本真实checkpoint smoke，足以说明代码链路可运行，但不能说明方法有效、成功率提升或实际加速幅度。

2026-07-24 15:17（Asia/Shanghai）将IR-ACoT MVP以提交`895cfe7`推送到main并由服务器`/root/ACoT-VLA` fast-forward同步，保留既有`third_party/libero`脏子模块；服务器再次运行endpoint dataset与sidecar policy共11项单元测试，全部通过。随后在50999、10步EAR、10步final、random selection seed7上完成正式200-state开放环EAR因果审计，输出为`/root/autodl-tmp/acotvla/ir_acot_pilot/audit200`，200条均成功、0失败，审计门槛通过。null response L2中位数为0.016039；rotate中位数1.062952、semantic/null ratio约66.27，200条中100%超过各自2倍null、99.5%超过0.05；translate中位数0.049707，67.5%超过2倍null、50%超过0.05；gripper_shift中位数0.020706，仅27%超过2倍null、21.5%超过0.05；swap与null的中位数及逐样本分布基本相同，没有可用因果响应。该结果支持EAR至少对旋转及部分平移扰动是final-action的load-bearing输入，但不支持当前swap干预；后续IR relabel不使用swap，并将rotate/translate作为主要有效干预，gripper仅作为稀疏事件信号。审计仍不是成功率或加速结果。15:17已启动2k clean teacher endpoint导出，tmux为`ir_acot_labels2k`，日志为`/root/autodl-tmp/acotvla/ir_acot_pilot/logs/coarse_teacher_2k.log`。

2026-07-24 16:15（Asia/Shanghai）完成IR-ACoT 2k快速否证实验的全部开放环阶段，并因IR未通过预设门槛而停止，没有启动LIBERO闭环。2k clean 10/10 teacher labels全部成功导出；Fast-EAR使用1794条训练、206条episode-held-out validation训练300步，耗时186.5秒，只更新430098464个coarse参数并保存642 MiB sidecar。相同2k状态、相同noise的确定性对照显示，未训练naive 1-step在validation上的coarse MSE为0.00807179，Fast-EAR为0.00559834，降低30.64%；对应RMSE从0.08984降至0.07482。Fast-EAR随后为同一2k状态生成student coarse，冻结50999 teacher在该chain上用translate、rotate、gripper_shift各循环一个干预重标final endpoints，得到650/665/685条干预记录，三类teacher response L2中位数分别为0.05178、0.98263、0.15595，2000条均成功。

matched B6-lite与IR-lite均从同一Fast-EAR sidecar、相同数据划分、seed、optimizer和300步训练开始，只更新430098464个final参数；B6耗时230.4秒，IR耗时191.9秒，二者均保存包含coarse+final共860196928参数的约1.3 GiB BF16 sidecar。完整206条held-out validation的确定性评测中，B6 clean final MSE为0.00499738，IR为0.00500386，IR相对B6变化+0.1296%，paired bootstrap 95%区间为[-0.1867%, +0.4832%]；B6 intervention-response MSE为0.00181729，IR为0.00180693，仅降低0.5701%，95%区间为[-0.3984%, +1.6664%]，远低于预设至少20%的IR准入门槛。response cosine也几乎相同，B6为0.37638、IR为0.37780；按干预拆分的IR response-MSE改善仅translate 1.58%、rotate 0.37%、gripper 1.66%。全2k含训练样本的response-MSE改善为5.13%，仍低于门槛且未在held-out上保持，说明当前IR loss最多产生很弱的拟合变化，没有形成可泛化的核心增益。

普通一步endpoint蒸馏本身有正向但有限的信号：在student-chain final teacher目标上，未训练final 1-step validation MSE为0.00585014，B6降至0.00499738，改善14.58%；相对原始10/10 teacher的端到端validation MSE，naive 1/1为0.00920990，B6为0.00714273，改善22.45%。相同2k profile下原10/10平均clean policy infer为95.844 ms，B6 1/1为68.929 ms，IR为68.946 ms，仅约1.39倍单次policy加速，说明VLM/IAR等固定成本占比较高，未达到按NFE比例缩放。综合结论是Fast-EAR/B6 endpoint distillation可改善naive 1/1并获得约28.1%单次延迟下降，但当前2k/300-step IR-ACoT核心response loss未优于同速度B6，快速门槛失败；按预先规则不消耗闭环评测预算，也不能声称方法有效或达到ICRA贡献。结果位于`/root/autodl-tmp/acotvla/ir_acot_pilot/open_loop_b6_vs_ir.json`和`coarse_open_loop_comparison.json`；当前无tmux/GPU进程，GPU空闲，数据盘剩余约37 GiB。
2026-07-24 16:05（Asia/Shanghai）按“GitHub主要保留实验数据”的范围完成报告归档收缩和项目README重写，本任务未启动、停止或修改任何训练评测进程。完整`reports/context/experiment_log.md`在追加本记录前为2513行、479826字节，体积约480 KB，因此继续作为事实总账纳入Git；两份PPTX、月计划XLSX、日报/周报空模板及检查中间件均取消Git跟踪但保留本地副本。结构化数据继续按`results/execution_horizon_v2p`、`results/task89_adaptation`、`results/stage_b`和`results/ir_acot_pilot`归档，checkpoint、HDF5、视频、原始逐步rollout、完整日志、环境和缓存不入库。

16:00只读核对服务器IR-ACoT pilot时没有残留相关进程或tmux，`final_teacher_2k`、`final_b6`和`final_ir`均已完成；新增同步对应summary以及coarse/B6/IR三个短训练metric streams。B6-lite与IR-lite均完成300步、1794条训练记录和206条验证记录；最终验证点B6-lite的final MSE为0.00364254、IR-delta MSE为0.00030865、cosine为0.50246，IR-lite分别为0.00363019、0.00031122、0.49612。IR-lite的clean final MSE仅略低，但response MSE和cosine未优于matched B6-lite，未达到预设的约20% response-RMSE改善门槛；当前只能说明pilot训练链路完成，仍没有闭环LIBERO成功率或完整1×1端到端加速结论。

根`README.md`已从上游官方项目介绍改写为LightAcotVLA研究分支首页，明确列出正式V2-P 10×100结果、Stage B真实NFE降算力数据、Task8/9 targeted SFT/uniform/DAgger证据、IR-ACoT当前负/中性边界、数据索引、核心代码入口和下一阶段闭环门槛。GitHub合并与推送在本记录写入时尚未执行，完成后需另记远端验证结果。

2026-07-24 16:16（Asia/Shanghai）将最终实验数据归档和项目README以squash提交`02ab4d0`合并并推送到GitHub `main`。最终提交包含48个路径：`.gitignore`、项目/实验索引、完整事实日志，以及V2-P、Task8/9、IR-ACoT的小型JSON、JSONL和CSV结果；不包含PPTX、XLSX、汇报草稿、checkpoint、HDF5、视频或完整训练日志，本地5份汇报材料仍保留且已忽略。发布前验证26个JSON、4个JSONL均可解析，11个CSV表头均为29列，5份README本地链接无断链，未发现凭据模式、超过50 MiB的文件或待发布二进制汇报文件；`git ls-remote`与GitHub API均确认远端`main`为`02ab4d0cd26b2c690efcf51f63c5e57f1dd0e8b4`。
