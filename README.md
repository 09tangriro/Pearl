[![pipeline status](https://github.com/LondonNode/AnvilRL/actions/workflows/ci.yaml/badge.svg)](https://github.com/LondonNode/AnvilRL/actions/workflows/ci.yaml)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="docs/images/logo.png" align="right" width="50%"/>

# AnvilRL
AnvilRL (referred to as *Anvil*) is a pytorch based library with the goal of being excellent for rapid prototyping of new algorithms and ideas. As such, this is **not** intended to provide template pre-built algorithms as a baseline, but rather flexible tools to allow the user to quickly build and test their own implementations and ideas.

## Developer Guide
### Scripts
**Linux**
1. `scripts/setup_dev.sh`: setup your virtual environment
2. `scripts/run_tests.sh`: run tests

**Windows**
1. `scripts/windows_setup_dev.bat`: setup your virtual environment
2. `scripts/windows_run_tests.bat`: run tests

### Dependency Management
Anvil uses [poetry](https://python-poetry.org/docs/basic-usage/) for dependency management and build release instead of pip. As a quick guide:
1. Run `poetry add [package]` to add more package dependencies.
2. Poetry automatically handles the virtual environment used, check `pyproject.toml` for specifics on the virtual environment setup.
3. If you want to run something in the poetry virtual environment, add `poetry run` as a prefix to the command you want to execute. For example, to run a python file: `poetry run python3 script.py`.

## User Guide

### Installation
There are two options to install this package:
1. `pip install anvilrl`
2. `git clone git@github.com:LondonNode/AnvilRL.git`

### Algorithm Template
See `anvilrl/agents/template.py` for the templates to create your own algorithm! 
For more examples, see specific algorithm implementations under `anvilrl/agents`.

### Agent performance
To see training performance, use the command `tensorboard --logdir runs` or `tensorboard --logdir <tensorboard_log_path>` defined in your algorithm class initialization.

## Credit
Anvil was inspired by [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) and [Tonic](https://github.com/fabiopardo/tonic)
