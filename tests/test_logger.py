import shutil

import numpy as np

from pearll.common.logging_ import Logger
from pearll.common.type_aliases import Log

path = "runs/tests"


def test_init():
    Logger(tensorboard_log_path=path)


def test_reset_log():
    logger = Logger(tensorboard_log_path=path)
    logger.reset_log()
    shutil.rmtree(path)

    assert logger.rewards == []
    assert logger.actor_losses == []
    assert logger.critic_losses == []
    assert logger.entropies == []
    assert logger.divergences == []


def test_make_episode_log():
    logger = Logger(tensorboard_log_path=path)
    logger.rewards = [0]
    logger.actor_losses = [0]
    logger.critic_losses = [0]
    logger.entropies = [0]
    logger.divergences = [0]

    actual_log = logger._make_episode_log()
    expected_log = Log(actor_loss=0, critic_loss=0, entropy=0, divergence=0)

    shutil.rmtree(path)
    assert actual_log == expected_log


def test_add_train_log():
    logger = Logger(tensorboard_log_path=path)
    train_log = Log(actor_loss=0, critic_loss=0, entropy=0, divergence=0)

    logger.add_train_log(train_log)
    shutil.rmtree(path)
    assert logger.actor_losses == [0]
    assert logger.critic_losses == [0]
    assert logger.divergences == [0]
    assert logger.entropies == [0]


def test_add_reward():
    logger = Logger(tensorboard_log_path=path)
    vec_logger = Logger(num_envs=2, tensorboard_log_path=path)

    logger.add_reward(1.0)
    assert logger.rewards == [1.0]

    reward = np.array([1, 1])
    vec_logger.add_reward(reward)
    shutil.rmtree(path)
    np.testing.assert_array_almost_equal([1], vec_logger.rewards)


def test_episode_done():
    logger = Logger(tensorboard_log_path=path)
    done = False
    flag = logger.check_episode_done(done)
    assert not flag

    done = True
    flag = logger.check_episode_done(done)
    shutil.rmtree(path)
    assert flag


def test_stream_log():
    logger = Logger(tensorboard_log_path=path)
    logger.warning("test")
    logger.error("test")
    logger.exception("test")
    shutil.rmtree(path)
