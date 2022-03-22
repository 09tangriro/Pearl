echo Installing and creating poetry virtual environment
pip install poetry
poetry install

echo Installing pre-commit hooks
poetry run pre-commit install

echo DONE

echo.
echo =================================================================================
echo.

echo A python virtual environment has been created at:
poetry config --local virtualenvs.path

echo.
echo To run unit tests locally, run:
echo.
echo    poetry run pytest OR poetry run scripts/run_tests.sh

echo .
echo =================================================================================
echo.
