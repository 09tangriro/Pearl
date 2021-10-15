echo ---------------------
echo Installing and creating poetry virtual environment
echo ---------------------
pip install poetry
poetry install
echo ""

echo ---------------------
echo Installing pre-commit hooks
echo ---------------------
pip install pre-commit
pre-commit install
echo ""

echo ---------------------
echo DONE
echo ---------------------