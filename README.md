<img src="docs/images/logo.png" align="right" width="50%"/>

# Anvil
Anvil is a pytorch based library with the goal of being excellent for rapid prototyping of new algorithms and ideas over benchmarking. As such, this is **not** intended to provide template pre-built algorithms as a baseline, but rather flexible tools to allow the user to quickly build and test their own implementations and ideas.

## Developer Guide
### Scripts
1. `scripts/setup_dev.sh`: setup your virtual environment
2. `scripts/run_tests.sh`: run tests

### Dependency Management
Anvil uses [poetry](https://python-poetry.org/docs/basic-usage/) for dependency management and build release over pip. As a quick guide:
1. Run `poetry add [package]` to add more package dependencies.
2. Poetry automatically handles the virtual environment used, check `pyproject.toml` for specifics on the virtual environment setup.
3. If you want to run something in the poetry virtual environment, add `poetry run` as a prefix to the command you want to execute. For example, to run a python file: `poetry run python3 script.py`.

## Credit
Anvil was inspired by [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) and [Tonic](https://github.com/fabiopardo/tonic)
