#!/bin/sh
echo ---------------------
echo Installing and creating poetry virtual environment
echo ---------------------
pip3 install poetry
poetry install
echo ""

echo ---------------------
echo Installing pre-commit hooks
echo ---------------------
pip3 install pre-commit
pre-commit install
echo ""

echo ---------------------
echo DONE
echo ---------------------
