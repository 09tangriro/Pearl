import shutil

import numpy as np

from anvilrl.common.logging_ import Logger
from anvilrl.common.type_aliases import Log

path = "runs/tests"


def test_init():
    Logger(tensorboard_log_path=path)


def test_reset_log():
    logger = Logger(tensorboard_log_path=path)
    logger.reset_log()

    assert logger.rewards == []
    assert logger.actor_losses == []
    assert logger.critic_losses == []
    assert logger.entropies == []
    assert logger.divergences == []


def test_make_episode_log():
    logger = Logger(tensorboard_log_path=path)
    logger.rewards = [0]

    actual_log = logger._make_episode_log()
    expected_log = Log()

    assert actual_log == expected_log


def test_add_train_log():
    logger = Logger(tensorboard_log_path=path)
    train_log = Log()

    logger.add_train_log(train_log)
    assert logger.actor_losses == []
    assert logger.critic_losses == []
    assert logger.divergences == []
    assert logger.entropies == []


def test_add_reward():
    logger = Logger(tensorboard_log_path=path)
    vec_logger = Logger(num_envs=2)

    logger.add_reward(1.0)
    assert logger.rewards == [1.0]

    reward = np.array([1, 1])
    vec_logger.add_reward(reward)
    shutil.rmtree(path)
    np.testing.assert_array_almost_equal([1], vec_logger.rewards)
