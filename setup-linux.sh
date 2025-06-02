#!/bin/bash
set -e

VENV_DIR=".venv"

# Find python3.10+ (prefer 3.10, fallback to highest >=3.10)
PYTHON_BIN=$(command -v python3.10 || command -v python3.11 || command -v python3.12)

if [ -z "$PYTHON_BIN" ]; then
  echo "❌ Python 3.10 or newer is required but not found."
  echo "👉 Install with: sudo apt-get update && sudo apt-get install python3.10 python3.10-venv"
  echo "   Or use pyenv: https://github.com/pyenv/pyenv"
  exit 1
fi

PYTHON_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ $(echo "$PYTHON_VERSION < 3.10" | bc) -eq 1 ]]; then
  echo "❌ Python version $PYTHON_VERSION found, but 3.10 or newer is required."
  exit 1
fi

if ! $PYTHON_BIN -m venv --help >/dev/null 2>&1; then
  echo "🔧 Installing venv module for $PYTHON_BIN..."
  sudo apt-get update && sudo apt-get install -y "python${PYTHON_VERSION}-venv"
fi

if [ -d "$VENV_DIR" ]; then
  echo "✅ Virtual environment already exists. Skipping setup."
  exit 0
fi

$PYTHON_BIN -m venv "$VENV_DIR"
echo "🔗 Activating virtual environment..."
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
[ -f requirements.txt ] && pip install -r requirements.txt

echo "✅ Python $PYTHON_VERSION virtual environment created and activated at ./$VENV_DIR"