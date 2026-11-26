"""Short vertical smoke test of configuration, environment and TD3 learning."""

import keras
import numpy as np
import yaml

from rl_system.agents.td3.agent import TD3Agent
from rl_system.agents.td3.config import TD3Config
from rl_system.config import load_config
from rl_system.environment import make_environment


def test_td3_pendulum_smoke(tmp_path):
    seed = 42
    config_path = tmp_path / 'td3_pendulum.yaml'
    config_path.write_text(yaml.safe_dump({
        'environment': {'id': 'Pendulum-v1', 'kwargs': {}},
        'algorithm': {'name': 'td3', 'params': {
            'replay_capacity': 32, 'batch_size': 4,
            'gamma': 0.99, 'tau': 0.005,
            'actor_learning_rate': 0.001, 'critic_learning_rate': 0.001,
            'policy_delay': 2, 'exploration_noise_std': 0.1,
            'target_policy_noise_std': 0.2, 'target_noise_clip': 0.5,
            'actor_output_init_limit': 0.003,
            'actor': {'hidden_units': [8, 8], 'hidden_activations': ['relu', 'relu']},
            'critic': {'hidden_units': [8, 8], 'hidden_activations': ['relu', 'relu']},
        }},
        'training': {'total_steps': 10, 'seed': seed},
    }), encoding = 'utf-8')
    config = load_config(config_path)
    td3_config = TD3Config.model_validate(config.algorithm.params)
    keras.utils.set_random_seed(config.training.seed)
    env, spec = make_environment(config.environment)
    try:
        agent = TD3Agent(spec, td3_config, seed = config.training.seed)
        obs, _ = env.reset(seed = config.training.seed)
        updates = 0
        actor_updates = 0
        targets = [*agent._target_actor.weights, *agent._target_critic_1.weights,
                   *agent._target_critic_2.weights]
        initial_targets = [v.numpy().copy() for v in targets]
        for step in range(config.training.total_steps):
            action = agent.select_action(obs, explore = True)
            assert action.shape == (spec.action_dim,)
            assert action.dtype == np.float32
            assert np.all(np.isfinite(action))
            assert env.action_space.contains(action)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, terminated, truncated)
            metrics = agent.update()
            if step + 1 < td3_config.batch_size:
                assert metrics is None
            else:
                updates += 1
                assert metrics is not None
                expected = {'critic_1_loss', 'critic_2_loss'}
                if updates % td3_config.policy_delay == 0:
                    expected.add('actor_loss')
                    actor_updates += 1
                assert set(metrics) == expected
                assert all(type(v) is float and np.isfinite(v) for v in metrics.values())
            obs = env.reset()[0] if terminated or truncated else next_obs

        assert updates == config.training.total_steps - td3_config.batch_size + 1
        assert actor_updates == updates // td3_config.policy_delay > 0
        assert agent._update_count == updates
        assert int(agent._critic_1_optimizer.iterations.numpy()) == updates
        assert int(agent._critic_2_optimizer.iterations.numpy()) == updates
        assert int(agent._actor_optimizer.iterations.numpy()) == actor_updates
        assert any(not np.array_equal(v.numpy(), initial) for v, initial in
                   zip(targets, initial_targets, strict = True))
        action = agent.select_action(obs, explore = False)
        assert action.dtype == np.float32
        assert action.shape == (spec.action_dim,)
        assert np.all(np.isfinite(action)) and env.action_space.contains(action)
    finally:
        env.close()
