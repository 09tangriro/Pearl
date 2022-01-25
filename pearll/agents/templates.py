from typing import List, Optional, Type

from gym import Env

from pearll.agents.base_agents import BaseAgent
from pearll.buffers.base_buffer import BaseBuffer
from pearll.callbacks.base_callback import BaseCallback
from pearll.common.type_aliases import Log
from pearll.explorers.base_explorer import BaseExplorer
from pearll.models.actor_critics import ActorCritic
from pearll.settings import (
    BufferSettings,
    ExplorerSettings,
    LoggerSettings,
    MiscellaneousSettings,
    Settings,
)


class YourRLAgent(BaseAgent):
    """
    A template for a new agent :)

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
    :param buffer_class: the buffer class for storing and sampling trajectories
    :param buffer_settings: settings for the buffer
    :param action_explorer_class: the explorer class for random search at beginning of training and
        adding noise to actions
    :param explorer settings: settings for the action explorer
    :param callbacks: an optional list of callbacks (e.g. if you want to save the model)
    :param callback_settings: settings for callbacks
    :param logger_settings: settings for the logger
    :param misc_settings: settings for miscellaneous parameters
    """

    def __init__(
        self,
        env: Env,
        model: ActorCritic,
        buffer_class: Type[BaseBuffer] = ...,
        buffer_settings: BufferSettings = ...,
        action_explorer_class: Type[BaseExplorer] = ...,
        explorer_settings: ExplorerSettings = ...,
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        callback_settings: Optional[List[Settings]] = None,
        logger_settings: LoggerSettings = ...,
        misc_settings: MiscellaneousSettings = ...,
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
            misc_settings=misc_settings,
        )

    def _fit(
        self, batch_size: int, actor_epochs: int = 1, critic_epochs: int = 1
    ) -> Log:
        """
        Specify your algorithm logic here! Should call your critic_updater and actor_updater
        and return a `Log` object, details of which can be found in pearll/common/type_aliases.py
        """
