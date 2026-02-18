#!/usr/bin/env bash
set -e

echo "🔧 Creating virtual environment"
python -m venv .venv

echo "🐍 Activating venv"
source .venv/bin/activate

echo "⬆️ Upgrading pip"
pip install --upgrade pip setuptools wheel

echo " Installing requirements"
pip install -r requirements.txt

echo "✅ Installation complete"
python - << EOF
import torch
print("CUDA available:", torch.cuda.is_available())
EOF
