#!/bin/bash
# ==============================================================================
# Entrypoint script for Wan2.2-TI2V-5B Docker container
# ==============================================================================
# This script validates the environment before running inference:
# 1. Verifies RTX 5090 GPU detection
# 2. Checks CUDA 12.8 and PyTorch 2.7.0 compatibility
# 3. Validates model checkpoint availability
# 4. Reports Flash Attention status
# ==============================================================================

set -e

echo "========================================="
echo "  Wan2.2-TI2V-5B RTX 5090 Container"
echo "========================================="
echo ""

# ==============================================================================
# GPU Detection
# ==============================================================================
echo "=== GPU Detection ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is NVIDIA Docker runtime installed?"
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
echo ""

# ==============================================================================
# CUDA and PyTorch Information
# ==============================================================================
echo "=== CUDA/PyTorch Information ==="
python3 -c "
import torch
import sys

print(f'Python Version: {sys.version.split()[0]}')
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'CUDA Available: {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    print('ERROR: CUDA is not available!')
    sys.exit(1)

print(f'GPU Count: {torch.cuda.device_count()}')
print(f'GPU Name: {torch.cuda.get_device_name(0)}')
capability = torch.cuda.get_device_capability(0)
print(f'Compute Capability: {capability[0]}.{capability[1]}')
print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

# Verify RTX 5090 (compute capability 12.0)
if capability[0] != 12:
    print(f'WARNING: Expected compute capability 12.x for RTX 5090, got {capability[0]}.{capability[1]}')
else:
    print('✓ RTX 5090 (Blackwell) detected correctly')
" || exit 1

echo ""

# ==============================================================================
# Flash Attention Check
# ==============================================================================
echo "=== Flash Attention Status ==="
python3 -c "
import sys
try:
    import flash_attn
    print(f'✓ Flash Attention Version: {flash_attn.__version__}')
    print('✓ Flash Attention 2 Available: True')
    print('  (Optimized attention kernels enabled)')
except ImportError as e:
    print(f'⚠ Flash Attention not available: {e}')
    print('  Will fallback to standard attention (slower but functional)')
    # Don't exit - code has automatic fallback
" || true

echo ""

# ==============================================================================
# Model Checkpoint Verification
# ==============================================================================
echo "=== Model Checkpoint Verification ==="
MODEL_DIR="/models/Wan2.2-TI2V-5B"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found at $MODEL_DIR"
    echo ""
    echo "Please mount the model directory with:"
    echo "  -v /home/humanoid/Downloads/Wan2.2/Wan2.2-TI2V-5B:/models/Wan2.2-TI2V-5B:ro"
    echo ""
    echo "Or using docker-compose:"
    echo "  docker-compose run --rm wan-ti2v <command>"
    exit 1
fi

echo "✓ Model directory found at $MODEL_DIR"

# Check for essential model files
REQUIRED_FILES=("config.json" "Wan2.2_VAE.pth" "diffusion_pytorch_model.safetensors.index.json")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$MODEL_DIR/$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "ERROR: Missing required model files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

echo "✓ All required model files present"
echo ""

# ==============================================================================
# Ready to Execute
# ==============================================================================
echo "========================================="
echo "Container ready. Executing command: $@"
echo "========================================="
echo ""

# Execute the provided command
exec "$@"
