#!/bin/bash
# ============================================================
# Download DINO Models Script for IFP Project
# ============================================================
# Downloads DINOv2 pre-trained weights.
# Note: DINOv3 requires applying for access on HuggingFace first.
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Downloading DINO models ==="

# ---- DINOv3 (HuggingFace) ----
echo "[1/2] DINOv3 models (HuggingFace access required):"
echo "  -> Apply for access at: https://huggingface.co/facebook/dinov3"
echo "  -> After approval, download weights and place in: ${PROJECT_DIR}/dinov3/download_models/"
echo "  -> Required files:"
echo "       dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth  (ViT-H/16+, default)"
echo "       dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth       (ViT-L/16)"

# ---- DINOv2 ----
echo ""
echo "[2/2] Setting up DINOv2 models..."
DINOV2_DIR="${PROJECT_DIR}/dinov2/download_models"
mkdir -p "${DINOV2_DIR}"

echo "  DINOv2 ViT-L/14:"
echo "    -> wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
echo "    -> Place in: ${DINOV2_DIR}/dinov2_vitl14_pretrain.pth"

echo "  DINOv2 ViT-G/14:"
echo "    -> wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth"
echo "    -> Place in: ${DINOV2_DIR}/dinov2_vitg14_pretrain.pth"

echo "  DINOv2 ViT-B/14:"
echo "    -> wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
echo "    -> Place in: ${DINOV2_DIR}/dinov2_vitb14_pretrain.pth"

echo ""
echo "=== DINO model download guide complete ==="
echo "Supported DINO variants: dinov3_vith16plus, dinov3_vitl16, dino2l, dino2g, dino2b"
