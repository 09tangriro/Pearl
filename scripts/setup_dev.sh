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
poetry run pre-commit install
echo ""

echo ---------------------
echo DONE
echo ---------------------

echo ""
echo =================================================================================
echo ""

echo A python virtual environment has been created at:
poetry config --local virtualenvs.path

echo ""
echo To run unit tests locally:
echo    poetry run pytest

echo ""
echo =================================================================================
echo ""
