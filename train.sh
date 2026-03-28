#!/bin/bash
# train.sh — manufacturing MARL training wrapper
# Fixes all known Python 3.13 + Windows venv issues before training.

set -e
cd "$(dirname "$0")"

echo "[1/4] Fixing distutils pth..."
python -c "
import os
pth = os.path.join('venv','Lib','site-packages','distutils-precedence.pth')
if os.path.exists(pth):
    open(pth,'w').close()
    print('  cleared:', pth)
"

echo "[2/4] Clearing __init__.py files..."
python -c "
import os
root = os.getcwd()
count = 0
for dirpath, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in ('venv','.git','__pycache__')]
    for f in files:
        if f == '__init__.py':
            p = os.path.join(dirpath, f)
            if os.path.getsize(p) > 0:
                open(p,'w').close()
                count += 1
print(f'  cleared {count} __init__.py files')
"

echo "[3/4] Verifying packages..."
python -c "
import importlib, sys

# Force reimport from scratch
for mod in list(sys.modules.keys()):
    if mod in ('yaml','numpy','torch') or mod.startswith(('yaml.','numpy.','torch.')):
        del sys.modules[mod]

import yaml
assert hasattr(yaml, 'safe_load'), f'yaml missing safe_load, file={yaml.__file__}'

import numpy
assert hasattr(numpy, 'ndarray'), f'numpy missing ndarray, file={numpy.__file__}'

import torch
assert hasattr(torch, 'cuda'), f'torch missing cuda, file={torch.__file__}'
assert torch.cuda.is_available(), 'CUDA not available - check GPU drivers'

print(f'  yaml      {yaml.__version__}')
print(f'  numpy     {numpy.__version__}')
print(f'  torch     {torch.__version__}  CUDA={torch.version.cuda}')
print(f'  GPU       {torch.cuda.get_device_name(0)}')
" || { echo "FAILED - try: deactivate && source venv/Scripts/activate && bash train.sh"; exit 1; }

echo "[4/4] Starting training..."
echo ""

# Parse args
CONFIG="configs/phase1.yaml"
TIMESTEPS="500000"
EXTRA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)    CONFIG="$2";   shift 2 ;;
        --timesteps) TIMESTEPS="$2"; shift 2 ;;
        --resume)    EXTRA="$EXTRA --resume $2"; shift 2 ;;
        *)           EXTRA="$EXTRA $1"; shift ;;
    esac
done

echo "  Config:    $CONFIG"
echo "  Steps:     $TIMESTEPS"
[ -n "$EXTRA" ] && echo "  Extra:     $EXTRA"
echo ""

python scripts/train.py --config "$CONFIG" --timesteps "$TIMESTEPS" $EXTRA