#!/bin/bash
set -e

PYTHON_VERSION="3.10"
VENV_DIR=".venv"

# Check for python3.10
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "❌ Python 3.10 is required. Please install it (e.g., via pyenv or your package manager)."
  exit 1
fi

# Ensure venv module is available
if ! python3.10 -m venv --help >/dev/null 2>&1; then
  echo "🔧 Installing python3.10-venv module..."
  sudo apt-get update && sudo apt-get install -y python3.10-venv
fi

if [ -d "$VENV_DIR" ]; then
  echo "✅ Virtual environment already exists. Skipping setup."
  exit 0
fi

python3.10 -m venv "$VENV_DIR"
echo "🔗 Activating virtual environment..."
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
[ -f requirements.txt ] && pip install -r requirements.txt

echo "✅ Python 3.10 virtual environment created and activated at ./$VENV_DIR"