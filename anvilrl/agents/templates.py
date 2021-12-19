from typing import List, Optional, Type, Union

import torch as T
from gym import Env
from gym.vector import VectorEnv

from anvilrl.agents.base_agents import BaseDeepAgent, BaseEvolutionAgent
from anvilrl.buffers.base_buffer import BaseBuffer
from anvilrl.callbacks.base_callback import BaseCallback
from anvilrl.common.type_aliases import Log
from anvilrl.explorers.base_explorer import BaseExplorer
from anvilrl.models.actor_critics import ActorCritic
from anvilrl.settings import (
    BufferSettings,
    CallbackSettings,
    ExplorerSettings,
    LoggerSettings,
    PopulationInitializerSettings,
)
from anvilrl.updaters.evolution import BaseEvolutionUpdater


class YourDeepAgent(BaseDeepAgent):
    """
    A template for a new deep RL agent :)

    Add here any other modules you like; for example, an actor updater or a critic updater. If you
    want to use a new module which inherits from an existing module and need to pass through unique
    settings, simply inherit the setting object from which the module is based and update as required
    before passing here.

    For example:
    ```
    @dataclass
    class HERBufferSettings(BufferSettings):
        goal_selection_strategy: Union[str, GoalSelectionStrategy] = "future"
        n_sampled_goal: int = 4
    ```

    :param env: the gym-like environment to be used
    :param model: the neural network model
    :param action_explorer_class: the explorer class for random search at beginning of training and
            adding noise to actions
    :param explorer settings: settings for the action explorer
    :param buffer_class: the buffer class for storing and sampling trajectories
    :param buffer_settings: settings for the buffer
    :param logger_settings: settings for the logger
    :param callbacks: an optional list of callbacks (e.g. if you want to save the model)
    :param callback_settings: settings for callbacks
    :param device: device to run on, accepts "auto", "cuda" or "cpu"
    :param render: whether to render the environment or not
    :param seed: optional seed for the random number generator
    """

    def __init__(
        self,
        env: Env,
        model: ActorCritic,
        action_explorer_class: Type[BaseExplorer] = ...,
        explorer_settings: ExplorerSettings = ...,
        buffer_class: Type[BaseBuffer] = ...,
        buffer_settings: BufferSettings = ...,
        logger_settings: LoggerSettings = ...,
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        callback_settings: Optional[List[CallbackSettings]] = None,
        device: Union[str, T.device] = "auto",
        render: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            env,
            model,
            action_explorer_class=action_explorer_class,
            explorer_settings=explorer_settings,
            buffer_class=buffer_class,
            buffer_settings=buffer_settings,
            logger_settings=logger_settings,
            callbacks=callbacks,
            callback_settings=callback_settings,
            device=device,
            render=render,
            seed=seed,
        )

    def _fit(
        self, batch_size: int, actor_epochs: int = 1, critic_epochs: int = 1
    ) -> Log:
        """
        Specify your algorithm logic here! Should call your critic_updater and actor_updater
        and return a `Log` object, details of which can be found in anvilrl/common/type_aliases.py
        """


class YourSearchAgent(BaseEvolutionAgent):
    """
    A template for a new search agent.

    Add here any other modules you like.  If you want to use a new module which inherits from an existing
    module and need to pass through unique settings, simply inherit the setting object from which the
    module is based and update as required before passing here.

    For example:
    ```
    @dataclass
    class YourSpecialLoggerSettings(LoggerSettings):
        setting1: int = 1
        setting2: str = "hello"
    ```

    :param env: the gym vecotrized environment
    :param updater_class: the class to use for the updater handling the actual update algorithm
    :param population_initializer_settings: the settings object for population initialization
    :param buffer_class: the buffer class for storing and sampling trajectories
    :param buffer_settings: settings for the buffer
    :param logger_settings: settings for the logger
    :param device: device to run on, accepts "auto", "cuda" or "cpu" (needed to pass to buffer,
        can mostly be ignored)
    :param seed: optional seed for the random number generator
    """

    def __init__(
        self,
        env: VectorEnv,
        updater_class: Type[BaseEvolutionUpdater],
        population_initializer_settings: PopulationInitializerSettings = ...,
        buffer_class: Type[BaseBuffer] = ...,
        buffer_settings: BufferSettings = ...,
        logger_settings: LoggerSettings = ...,
        device: Union[str, T.device] = "auto",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            env,
            updater_class,
            population_initializer_settings=population_initializer_settings,
            buffer_class=buffer_class,
            buffer_settings=buffer_settings,
            logger_settings=logger_settings,
            device=device,
            seed=seed,
        )

    def _fit(self) -> Log:
        """
        Specify your algorithm logic here! Should call your updater and return a `Log` object,
        details of which can be found in anvilrl/common/type_aliases.py
        """
