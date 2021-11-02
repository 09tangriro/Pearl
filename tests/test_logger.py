from anvilrl.common.logging import Logger
from anvilrl.common.type_aliases import Log


def test_init():
    Logger()


def test_reset_episode_log():
    logger = Logger()
    logger.reset_episode_log()

    assert logger.episode_rewards == []
    assert logger.episode_actor_losses == []
    assert logger.episode_critic_losses == []
    assert logger.episode_entropies == []
    assert logger.episode_kl_divergences == []


def test_make_episode_log():
    logger = Logger()
    logger.episode_rewards = [0]

    actual_log = logger._make_episode_log()
    expected_log = Log()

    assert actual_log == expected_log
