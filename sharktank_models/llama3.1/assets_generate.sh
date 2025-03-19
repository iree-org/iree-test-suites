#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/assets.venv"
ASSETS_DIR="${SCRIPT_DIR}/assets"
IRPA_FILE="${ASSETS_DIR}/toy_llama.irpa"
BATCH_SIZES=(1 4 32)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GENERATING BATCHED MODELS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

source "$VENV_PATH/bin/activate"

if [ -f "$ASSETS_DIR/shark-ai-commit.txt" ]; then
    SHARK_COMMIT=$(cat "$ASSETS_DIR/shark-ai-commit.txt")
    echo "Using Shark-AI commit: $SHARK_COMMIT"
else
    echo "ERROR: Shark-AI commit file not found. Please run assets_generate_venv_init.sh first."
    exit 1
fi

echo "Recording environment information..."
{
    echo "# Environment Information"
    echo "Date: $(date)"
    echo "Shark-AI Commit: $SHARK_COMMIT"
    echo "IREE Base Compiler: $(pip freeze | grep iree-base-compiler)"
    echo "IREE Turbine: $(pip freeze | grep iree-turbine)"
    echo "Python Version: $(python --version 2>&1)"
    echo "IREE Compile Path: $(which iree-compile)"
    echo "IREE Compile Version: $(iree-compile --version 2>&1 | head -n 1)"
} > "$ASSETS_DIR/version_info.txt"

echo "Generating MLIR files for batch sizes: ${BATCH_SIZES[*]}"
for BS in "${BATCH_SIZES[@]}"; do
    echo "   Processing batch size $BS..."
    
    BS_DIR="$ASSETS_DIR/bs$BS"
    mkdir -p "$BS_DIR"
    
    OUTPUT_MLIR="$BS_DIR/toy_llama_bs$BS.mlir"
    
    python -m sharktank.examples.export_paged_llm_v1 --bs-prefill=$BS --bs-decode=$BS \
        --irpa-file "$IRPA_FILE" --output-mlir "$OUTPUT_MLIR"
    
    echo "   Generated $OUTPUT_MLIR"
done

echo "Generation complete!"
echo "Generated MLIR files:"
for BS in "${BATCH_SIZES[@]}"; do
    echo "   - Batch size $BS: $ASSETS_DIR/bs$BS/toy_llama_bs$BS.mlir"
done
echo "Environment details saved to $ASSETS_DIR/version_info.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"