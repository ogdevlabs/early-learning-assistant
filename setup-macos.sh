#!/bin/bash
set -e

PYTHON_VERSION="3.10.11"
VENV_DIR=".venv"
PKGFILE="python-$PYTHON_VERSION-macos11.pkg"
PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10"

command -v curl >/dev/null 2>&1 || { echo "❌ curl is required but not installed."; exit 1; }
command -v installer >/dev/null 2>&1 || { echo "❌ macOS 'installer' command not found."; exit 1; }

if [ -d "$VENV_DIR" ]; then
  echo "✅ Virtual environment already exists. Skipping setup."
  exit 0
fi

if [ ! -f "$PKGFILE" ]; then
  echo "📥 Downloading Python $PYTHON_VERSION..."
  curl -fLO "https://www.python.org/ftp/python/$PYTHON_VERSION/$PKGFILE"
  if [ $? -ne 0 ] || [ ! -f "$PKGFILE" ]; then
    echo "❌ Download failed. Check your internet connection or the Python version."
    exit 1
  fi
fi

echo "📦 Installing Python $PYTHON_VERSION..."
if ! sudo installer -pkg "$PKGFILE" -target /; then
  echo "❌ Python installer failed. Try running the script again or install Python manually."
  exit 1
fi

if ! [ -x "$PYTHON_BIN" ]; then
  echo "❌ Could not locate installed Python binary at $PYTHON_BIN."
  echo "   Please check the installation or adjust the script to your Python path."
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
[ -f requirements.txt ] && pip install -r requirements.txt

echo "✅ Python $PYTHON_VERSION virtual environment created at ./$VENV_DIR"