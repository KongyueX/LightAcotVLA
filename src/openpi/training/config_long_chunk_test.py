from openpi.training import config


def test_long_chunk_configs_are_strictly_opt_in():
    legacy = config.get_config("acot_libero_action_cot_explicit_implicit_co_fusion")
    h15 = config.get_config("acot_libero_long_chunk_h15")
    h20 = config.get_config("acot_libero_long_chunk_h20")
    h25 = config.get_config("acot_libero_long_chunk_h25")

    assert legacy.model.action_horizon == 10
    assert legacy.prefix_retention_loss_weight == 0.0
    assert legacy.validation_fraction == 0.0

    assert h15.model.action_horizon == 15
    assert h15.model.coarse_action_horizon == 15
    assert h15.prefix_retention_horizon == 10
    assert h15.prefix_retention_loss_weight > 0.0
    assert h15.prefix_retention_teacher_weight_loader is not None
    assert h15.validation_fraction == 0.10
    assert h15.validation_split_seed == 42
    assert h15.early_stopping_patience is not None
    assert h15.save_interval == h15.num_train_steps
    assert h15.keep_period is None
    assert h15.max_checkpoints_to_keep == 2

    assert h20.model.action_horizon == 20
    assert h20.model.coarse_action_horizon == 15
    assert h20.prefix_retention_horizon == 15
    assert h20.prefix_retention_loss_weight > 0.0
    assert h20.prefix_retention_teacher_weight_loader is not None
    assert h20.validation_fraction == 0.10
    assert h20.validation_split_seed == 42
    assert h20.early_stopping_patience is not None
    assert h20.save_interval == h20.num_train_steps
    assert h20.keep_period is None
    assert h20.max_checkpoints_to_keep == 2

    assert h25.model.action_horizon == 25
    assert h25.model.coarse_action_horizon == 15
    assert h25.prefix_retention_horizon == 20
    assert h25.prefix_retention_loss_weight > 0.0
    assert h25.prefix_retention_teacher_weight_loader is not None
    assert h25.validation_fraction == 0.10
    assert h25.validation_split_seed == 42
    assert h25.early_stopping_patience is not None
    assert h25.save_interval == h25.num_train_steps
    assert h25.keep_period is None
    assert h25.max_checkpoints_to_keep == 2
