#!/bin/bash
# ============================================================
# Download CLIP Models Script for IFP Project
# ============================================================
# Downloads CLIP weights from HuggingFace for offline use
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Downloading CLIP models ==="
CLIP_DIR="${PROJECT_DIR}/clip"
mkdir -p "${CLIP_DIR}"

echo "[1/2] CLIP ViT-B/16:"
echo "  -> From: https://huggingface.co/openai/clip-vit-base-patch16"
echo "  -> Run: python -c \"from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch16', cache_dir='${CLIP_DIR}/clip-vit-base-patch16'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16', cache_dir='${CLIP_DIR}/clip-vit-base-patch16')\""

echo ""
echo "[2/2] CLIP ViT-L/14:"
echo "  -> From: https://huggingface.co/openai/clip-vit-large-patch14"
echo "  -> Run: python -c \"from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-large-patch14', cache_dir='${CLIP_DIR}/clip-vit-large-patch14'); CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', cache_dir='${CLIP_DIR}/clip-vit-large-patch14')\""

echo ""
echo "=== CLIP model download guide complete ==="
echo "Supported CLIP variants: clipb (vit-base-patch16), clipl (vit-large-patch14)"
echo "CLIP path format: ./clip/clip-vit-base-patch16 or ./clip/clip-vit-large-patch14"
