#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/assets.venv"
SHARK_DIR="$HOME/shark-ai"
ASSETS_DIR="${SCRIPT_DIR}/assets"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "INITIALIZING VIRTUAL ENVIRONMENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH"
    python -m venv "$VENV_PATH"
else
    echo "Using existing virtual environment at $VENV_PATH"
fi
source "$VENV_PATH/bin/activate"

if [ ! -d "$SHARK_DIR" ]; then
    echo "Cloning shark-ai repository"
    git clone https://github.com/nod-ai/shark-ai.git "$SHARK_DIR"
else
    echo "Using existing shark-ai repository"
fi
cd "$SHARK_DIR"

SHARK_COMMIT=$(git rev-parse HEAD)
echo "Found shark-ai commit: $SHARK_COMMIT"

echo "Cleaning up previous installations"
pip uninstall -y iree-compiler iree-base-compiler

echo "Installing dependencies"
pip install --no-compile -r pytorch-cpu-requirements.txt
pip install --no-compile -r requirements-iree-pinned.txt
pip install --no-compile -e sharktank/

mkdir -p "$ASSETS_DIR"
echo "$SHARK_COMMIT" > "$ASSETS_DIR/shark-ai-commit.txt"

cd - > /dev/null
echo "Environment setup complete"
echo "Shark-AI commit hash saved to $ASSETS_DIR/shark-ai-commit.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"