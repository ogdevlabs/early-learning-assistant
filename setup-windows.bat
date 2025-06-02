@echo off
set PYTHON_VERSION=3.10.13
set PYTHON_DIR=python-%PYTHON_VERSION%
set ZIP_FILE=python-%PYTHON_VERSION%-embed-amd64.zip
set VENV_DIR=venv
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%ZIP_FILE%

IF EXIST %VENV_DIR% (
    echo ✅ Virtual environment already exists. Skipping setup.
    exit /b 0
)

echo 📥 Downloading embeddable Python %PYTHON_VERSION%...
powershell -Command "Invoke-WebRequest -Uri %PYTHON_URL% -OutFile %ZIP_FILE%"

echo 📦 Extracting...
powershell -Command "Expand-Archive -Path %ZIP_FILE% -DestinationPath %PYTHON_DIR%"

:: Create virtualenv using embedded python
cd %PYTHON_DIR%
echo import venv; venv.create(r'..\%VENV_DIR%', with_pip=True) > create_venv.py
python create_venv.py
cd ..

call %VENV_DIR%\Scripts\activate

echo ⬆️ Installing pip and dependencies...
python -m ensurepip
python -m pip install --upgrade pip
if exist requirements.txt (
    pip install -r requirements.txt
)

echo ✅ Python 3.10.13 virtual environment created at .\%VENV_DIR%