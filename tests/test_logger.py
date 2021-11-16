import shutil

import numpy as np

from anvilrl.common.logging_ import Logger
from anvilrl.common.type_aliases import Log

path = "runs/tests"


def test_init():
    Logger(tensorboard_log_path=path)


def test_reset_episode_log():
    logger = Logger(tensorboard_log_path=path)
    logger.reset_episode_log()

    assert logger.episode_rewards == []
    assert logger.episode_actor_losses == []
    assert logger.episode_critic_losses == []
    assert logger.episode_entropies == []
    assert logger.episode_kl_divergences == []


def test_make_episode_log():
    logger = Logger(tensorboard_log_path=path)
    logger.episode_rewards = [0]

    actual_log = logger._make_episode_log()
    expected_log = Log()

    assert actual_log == expected_log


def test_add_train_log():
    logger = Logger(tensorboard_log_path=path)
    train_log = Log()

    logger.add_train_log(train_log)
    assert logger.episode_actor_losses == []
    assert logger.episode_critic_losses == []
    assert logger.episode_kl_divergences == []
    assert logger.episode_entropies == []


def test_add_reward():
    logger = Logger(tensorboard_log_path=path)
    vec_logger = Logger(num_envs=2)

    logger.add_reward(1.0)
    assert logger.episode_rewards == [1.0]

    reward = np.array([1, 1])
    vec_logger.add_reward(reward)
    shutil.rmtree(path)
    np.testing.assert_array_almost_equal([reward], vec_logger.episode_rewards)
