#!/bin/sh
echo ---------------------
echo Installing pipx and ensuring path
echo ---------------------
python3 -m pip install --user pipx
python3 -m pipx ensurepath
sudo apt install python3.8-venv

echo ---------------------
echo Installing and creating poetry virtual environment
echo ---------------------
pipx install poetry
poetry install
echo ""

echo ---------------------
echo Installing pre-commit hooks
echo ---------------------
pipx install pre-commit
pre-commit install
echo ""

echo ---------------------
echo DONE
echo ---------------------
