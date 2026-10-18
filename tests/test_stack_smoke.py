"""Smoke tests for the local RL development stack (subphase 1.0)."""

import gymnasium as gym
import keras
import numpy as np
import tensorflow as tf

import rl_system


def test_rl_system_import():
    assert rl_system.__file__ is not None


def test_keras_tensorflow_cpu_gradient_update():
    assert keras.backend.backend() == "tensorflow"
    assert tf.config.list_physical_devices("CPU")

    with tf.device("/CPU:0"):
        model = keras.Sequential(
            [
                keras.Input(shape = (2,)),
                keras.layers.Dense(
                    1, use_bias = False, kernel_initializer = "ones"
                ),
            ]
        )
        inputs = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype = tf.float32)
        targets = tf.constant([[1.0], [2.0]], dtype = tf.float32)
        optimizer = keras.optimizers.SGD(learning_rate = 0.01)

        with tf.GradientTape() as tape:
            predictions = model(inputs, training = True)
            loss = tf.reduce_mean(tf.square(predictions - targets))

        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)

        assert predictions.shape == targets.shape
        assert bool(tf.reduce_all(tf.math.is_finite(predictions)).numpy())
        assert bool(tf.math.is_finite(loss).numpy())
        assert variables
        assert all(gradient is not None for gradient in gradients)
        assert all(
            bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
            for gradient in gradients
        )
        assert any(
            bool(tf.reduce_any(tf.not_equal(gradient, 0)).numpy())
            for gradient in gradients
        )

        weights_before = [variable.numpy().copy() for variable in variables]
        optimizer.apply_gradients(zip(gradients, variables))
        weights_after = [variable.numpy() for variable in variables]

        assert all(np.isfinite(weights).all() for weights in weights_after)
        assert any(
            not np.array_equal(before, after)
            for before, after in zip(weights_before, weights_after)
        )


def test_keras_save_load(tmp_path):
    model = keras.Sequential(
        [
            keras.Input(shape = (3,)),
            keras.layers.Dense(4, activation = "relu"),
            keras.layers.Dense(2),
        ]
    )
    inputs = tf.constant([[0.5, -1.0, 2.0]], dtype = tf.float32)
    predictions_before = model(inputs, training = False).numpy()

    checkpoint_path = tmp_path / "smoke_model.keras"
    model.save(checkpoint_path)
    assert checkpoint_path.is_file()

    restored_model = keras.models.load_model(checkpoint_path)
    predictions_after = restored_model(inputs, training = False).numpy()
    np.testing.assert_allclose(
        predictions_before, predictions_after, rtol = 1e-6, atol = 1e-6
    )


def test_gymnasium_pendulum_step():
    env = gym.make("Pendulum-v1")
    try:
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.observation_space.shape == (3,)
        assert env.action_space.shape == (1,)

        observation, reset_info = env.reset(seed = 123)
        assert isinstance(reset_info, dict)
        assert env.observation_space.contains(observation)

        action = env.action_space.sample()
        assert env.action_space.contains(action)
        next_observation, reward, terminated, truncated, step_info = env.step(action)

        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
    finally:
        env.close()
