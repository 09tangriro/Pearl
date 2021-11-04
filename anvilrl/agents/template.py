from typing import List, Optional, Type, Union

import torch as T
from gym import Env

from anvilrl.agents.base_agents import BaseDeepAgent
from anvilrl.buffers.base_buffer import BaseBuffer
from anvilrl.callbacks.base_callback import BaseCallback
from anvilrl.common.type_aliases import (
    BufferSettings,
    CallbackSettings,
    ExplorerSettings,
    Log,
    LoggerSettings,
)
from anvilrl.explorers.base_explorer import BaseExplorer
from anvilrl.models.actor_critics import ActorCritic


class YourAlgorithm(BaseDeepAgent):
    """
    A template for a new deep RL algorithm :)

    Add here any other modules you like; for example, an actor updater or a critic updater. If you
    want to use a new module which inherits from an existing module and need to pass through unique
    settings, simply inherit the setting object from which the module is based and update as required
    before passing here.

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
    """

    def __init__(
        self,
        env: Env,
        model: ActorCritic,
        action_explorer_class: Type[BaseExplorer] = ...,
        explorer_settings: ExplorerSettings = ...,
        buffer_class: BaseBuffer = ...,
        buffer_settings: BufferSettings = ...,
        logger_settings: LoggerSettings = ...,
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        callback_settings: Optional[List[CallbackSettings]] = None,
        device: Union[str, T.device] = "auto",
        render: bool = False,
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
        )

    def _fit(
        self, batch_size: int, actor_epochs: int = 1, critic_epochs: int = 1
    ) -> Log:
        """
        Specify your algorithm logic here! Should call your critic_updater and actor_updater
        and return a `Log` object, details of which can be found in anvilrl/common/type_aliases.py
        """
