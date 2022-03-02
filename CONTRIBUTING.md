# Contributing to Pearl

1. Create an issue proposing your change.
2. Once we've discussed and agree on the approach, create a new fork or branch to implement the change in.
3. If this is your first time working on the code, 🚨🚨 run `scripts/setup_dev.sh` 🚨🚨 to make sure your environment is setup properly, this will prevent a lot of potential problems when it comes to merging.
4. Implement your change.
5. Create a pull request (PR).
6. If the criteria for merging is met and everything looks good, we can merge into main 🎉🎉

# CI Pipeline Checks
1. Linting with `flake8` E9, F63, F7, F82
2. Check imports with `isort`.
3. Check format with `black`.
4. Check tests pass with `pytest`
5. Check code coverage with Codecov

## Tips for Passing the CI Pipeline
🚨🚨 Again, make sure you have run `scripts/setup_dev.sh` 🚨🚨 This sets up the pre-commit hooks and poetry environment with the correct package and python versions.

The pre-commit hooks should make sure any commits you make pass the `flake8`, `black` and `isort` checks, so you should only need to make sure that the tests pass! 

If you're tests pass when you run them but fail in the CI pipeline, it's probably because you're not running the tests in the [poetry](https://python-poetry.org/docs/basic-usage/) virtual environment so your dependency versions might be different. To fix this, use the command `scripts/run_tests.sh` to run your tests.

# Criteria for merging a PR

1. Good variable names and readable code.
2. Type hints for parameters in function definitions.
3. Docstrings specifying input and return parameters.
4. Pass CI pipeline.