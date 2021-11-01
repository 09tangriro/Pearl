[![pipeline status](https://github.com/LondonNode/AnvilRL/actions/workflows/ci.yaml/badge.svg)](https://github.com/LondonNode/AnvilRL/actions/workflows/ci.yaml)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="docs/images/logo.png" align="right" width="50%"/>

# AnvilRL
AnvilRL is a pytorch based library with the goal of being excellent for rapid prototyping of new algorithms and ideas over benchmarking. As such, this is **not** intended to provide template pre-built algorithms as a baseline, but rather flexible tools to allow the user to quickly build and test their own implementations and ideas.

## Developer Guide
### Scripts
**Linux**
1. `scripts/setup_dev.sh`: setup your virtual environment
2. `scripts/run_tests.sh`: run tests

**Windows**
1. `scripts/windows_setup_dev.bat`: setup your virtual environment
2. `scripts/windows_run_tests.bat`: run tests

### Dependency Management
AnvilRL uses [poetry](https://python-poetry.org/docs/basic-usage/) for dependency management and build release over pip. As a quick guide:
1. Run `poetry add [package]` to add more package dependencies.
2. Poetry automatically handles the virtual environment used, check `pyproject.toml` for specifics on the virtual environment setup.
3. If you want to run something in the poetry virtual environment, add `poetry run` as a prefix to the command you want to execute. For example, to run a python file: `poetry run python3 script.py`.

## User Guide

### Installation
There are two options to install this package:
1. `pip install anvil_rl`
2. `git clone git@github.com:LondonNode/AnvilRL.git`

### Algorithm Template
Below is a template of a new algorithm implementation:

```
class YourAlgorithm(BaseAgent):
    """
    A template for a new algorithm :)
    
    :param env: the environment
    :param model: an actor critic model
    :param actor_updater_class: the actor updater class, from anvil_rl/updaters/actors.py
    :param actor_optimizer_settings: optimizer settings for the actor updater, from anvil_rl/common/type_aliases.py
    :param critic_updater_class: the critic updater class, from anvil_rl/updaters/critics.py
    :param critic_optimizer_settings: optimizer settings for the actor updater, from anvil_rl/common/type_aliases.py
    :param buffer_class: the buffer class, from anvil_rl/buffers
    :param buffer_settings: the buffer settings, from anvil_rl/common/type_aliases.py
    :param action_explorer_class: action explorer class allows for uniform action sampling at beginning of training and adding noise to actions, from anvil_rl/explorers
    :param explorer_settings: settings for the action explorer, from anvil_rl/common/type_aliases.py
    :param callbacks: e.g. saving model, from anvil_rl/callbacks
    """
    def __init__(
        self,
        env: Env,
        model: ActorCritic,
        actor_updater_class: Type[BaseCriticUpdater],
        actor_optimizer_settings: OptimizerSettings,
        critic_updater_class: Type[BaseCriticUpdater],
        critic_optimizer_settings: OptimizerSettings,
        buffer_class: Type[BaseBuffer],
        buffer_settings: BufferSettings,
        action_explorer_class: Type[BaseExplorer],
        explorer_settings: ExplorerSettings,
        callbacks: Optional[List[Type[BaseCallback]]] = None,
        device: Union[T.device, str] = "auto",
        verbose: bool = True,
        render: bool = False,
        model_path: Optional[str] = None,
        tensorboard_log_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            env=env,
            model=model,
            callbacks=callbacks,
            device=device,
            verbose=verbose,
            model_path=model_path,
            tensorboard_log_path=tensorboard_log_path,
            render=render,
        )

        self.buffer = buffer_class(
            env=env,
            buffer_size=buffer_settings.buffer_size,
            n_envs=buffer_settings.n_envs,
            device=device,
        )
        self.actor_updater = actor_updater_class(
            optimizer_class=optimizer_settings.optimizer_class,
            lr=optimizer_settings.learning_rate,
            max_grad=optimizer_settings.max_grad,
        )
        self.critic_updater = critic_updater_class(
            optimizer_class=optimizer_settings.optimizer_class,
            lr=optimizer_settings.learning_rate,
            max_grad=optimizer_settings.max_grad,
        )

        if explorer_settings.scale is not None:
            self.action_explorer = action_explorer_class(
                action_space=env.action_space,
                start_steps=explorer_settings.start_steps,
                scale=explorer_settings.scale,
            )
        else:
            self.action_explorer = action_explorer_class(
                action_space=env.action_space, start_steps=explorer_settings.start_steps
            )

    def _fit(
        self, batch_size: int, actor_epochs: int = 1, critic_epochs: int = 1
    ) -> Log:
    """
    Specify your algorithm logic here! Should call your critic_updater and actor_updater
    and return a `Log` object, details of which can be found in anvil_rl/common/type_aliases.py
    """
```
For more examples, see implementations under `anvil_rl/agents`!

### Agent performance
To see training performance, use the command `tensorboard --logdir runs` or `tensorboard --logdir <tensorboard_log_path>` defined in your algorithm class initialization.

## Credit
AnvilRL was inspired by [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) and [Tonic](https://github.com/fabiopardo/tonic)
