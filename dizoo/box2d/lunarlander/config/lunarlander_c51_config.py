from easydict import EasyDict

lunarlander_c51_config = dict(
    exp_name='lunarlander_c51',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=200,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[128, 128, 64],
            v_min=-10,
            v_max=10,
            n_atom=51,
        ),
        discount_factor=0.97,
        nstep=1,   # 3*3
        learn=dict(
            update_per_collect=4,
            batch_size=64,
            learning_rate=0.0005, #0.0005
            target_update_freq=100,
        ),
        collect=dict(
            n_sample=128,  #
            unroll_len=1,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ), replay_buffer=dict(replay_buffer_size=20000, )
        ),
    ),
)
lunarlander_c51_config = EasyDict(lunarlander_c51_config)
main_config = lunarlander_c51_config

lunarlander_c51_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='c51'),
)
lunarlander_c51_create_config = EasyDict(lunarlander_c51_create_config)
create_config = lunarlander_c51_create_config
