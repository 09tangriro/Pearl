echo Installing and creating poetry virtual environment
pip install poetry
poetry install

echo Installing pre-commit hooks
pip install pre-commit
pre-commit install

echo DONE
